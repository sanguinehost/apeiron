//! The affine-state carrier of the sequence-mixer primitive.
//!
//! A mixer layer either keeps the append-only key/value list ([`super::KvState`], read by softmax
//! attention) or a fixed-size state per head updated by an affine map and read by a vector:
//!
//! `S_t = A_t · S_{t−1} + B_t`, `o_t = S_tᵀ · q_t`.
//!
//! Gated `DeltaNet` (Yang, Kautz and Hatamizadeh, 2024; llama.cpp `delta-net-base.cpp`
//! `build_delta_net_autoregressive`) is the instance `A_t = α_t (I − β_t k_t k_tᵀ)` and
//! `B_t = β_t k_t v_tᵀ`.
//!
//! # Algebraic Minimum and Exact Integer Contract (PERF-P1)
//!
//! Today's baseline evaluated the recurrence as decay, read, delta, write:
//!
//! 1. `S ← α S` (decay full 128×128 state matrix);
//! 2. `m = Sᵀ k` (first read pass);
//! 3. `d = β (v − m)`;
//! 4. `S ← S + k dᵀ` (first write pass);
//! 5. `o = Sᵀ q̃` (second read pass).
//!
//! This required 4 products per element across 2 passes over 16 384 state words.
//!
//! We optimize this to the algebraic minimum while maintaining an exact integer contract:
//!
//! (a) **Central Decay Deferral**: The scalar decay matrix `A = α·I` commutes with every rank-1
//! update. We factor `S = c · S̃` where `c` is a single scalar per head initialized to `1.0` (in
//! Q32.32, `1 << 32`), and `S̃` is the unscaled state accumulator. Per token, `c` updates as:
//! `c ← round_shift_32(α · c)`.
//! The state write becomes `S̃ ← S̃ + k · (d / c)ᵀ`.
//! When `c` drops below `2⁻⁸` (`RENORM_THRESHOLD = 1 << 24`), `c` is folded into `S̃` in one pass:
//! `S̃ ← round_shift_32(S̃ · c)`, and `c` is reset to `1.0` (`1 << 32`).
//!
//! **Bound on `|S̃|` and Overflow Refusal**:
//! We derive `STATE_BOUND = 1 << 48` (magnitude `65 536.0` in Q32.32).
//! In the read pass, `S̃ᵀ k` and `S̃ᵀ q̃` accumulate over `dk = 128` rows. Since `k` and `q̂` are
//! L2-normalized (`‖k‖₂ = 1`, `‖q̃‖₂ ≤ 1/√128`), every `|k[i]| ≤ 2³²` and `|q̃[i]| ≤ 2³²`.
//! In `i128` arithmetic, each product `|S̃[i,j] · k[i]| ≤ 2⁴⁸ · 2³² = 2⁸⁰`, and the sum of 128 terms
//! satisfies `|Σ S̃[i,j] · k[i]| ≤ 128 · 2⁸⁰ = 2⁸⁷ < 2¹²⁷ − 1` (40 bits of headroom).
//! In AVX2 SIMD using 64-bit lanes (splitting Q32.32 words into 32-bit halves):
//! `S̃[i,j] = S_hi · 2³² + S_lo` where `|S_hi| ≤ 2¹⁶` and `S_lo ∈ [0, 2³² − 1]`.
//! For operand `k[i] = k_hi · 2³² + k_lo`, `k_hi ∈ {−1, 0, 1}` and `k_lo ∈ [0, 2³² − 1]`.
//! The middle term `S_hi · k_lo + k_hi · S_lo + (S_lo · k_lo)_hi` has magnitude `≤ 2¹⁶ · 2³² + 2³² + 2³² = 2⁴⁸ + 2³³`.
//! Summing across 128 rows yields `≤ 128 · (2⁴⁸ + 2³³) = 2⁵⁵ + 2⁴⁰ < 2⁶³ − 1`, fitting strictly
//! inside signed 64-bit lanes with 8 bits of headroom.
//! Any state element exceeding `STATE_BOUND` or non-positive `c` immediately refuses with
//! `StepError::Overflow`.
//!
//! (b) **Rank-1 Read Identity**:
//! With `S_new = S_old + k dᵀ`, the output readout satisfies:
//! `o = S_newᵀ q̃ = (S_old + k dᵀ)ᵀ q̃ = S_oldᵀ q̃ + d · (kᵀ q̃)`.
//! Because `S_old = c · S̃`, we compute both `m̃ = S̃ᵀ k` and `õ = S̃ᵀ q̃` simultaneously in **ONE
//! read pass** over `S̃` (two MACs per element, exact accumulation without intermediate rounding).
//! We then evaluate:
//! `m = round_shift_32(c · m̃)`,
//! `d = round_shift_32(β · (v − m))`,
//! `d' = round(d / c)`,
//! `kᵀ q̃ = round_shift_32(Σ k[i] · q̃[i])`,
//! `o = round_shift_32(c · õ) + round_shift_32(d · (kᵀ q̃))`.
//! Finally, **ONE write pass** executes `S̃ += k d'ᵀ`.
//! This reduces operations from 4 products to 3 per element, and roundings from 2 to 1 per written word.
//!
//! Every product is the one checked Q32.32 multiply [`mul_q32`]. Heads are independent, so running
//! them across the pool is bit-identical to the serial loop.

use std::mem::MaybeUninit;
use std::sync::Mutex;

use super::chart::ffn::attn_heads;
use super::step::StepError;
use super::value::{mul_q32, round_shift_32};
use super::{rms_norm_fixed, silu_fixed, AffineGeom, AffineVectors};
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_ln::{fixed_sigmoid, fixed_softplus};
use crate::numeric::fixed_rsqrt::{fixed_rsqrt, isqrt};

/// Threshold below which the central decay scalar `c` is folded back into `S̃` (2⁻⁸ in Q32.32).
pub const RENORM_THRESHOLD: i64 = 1 << 24;

/// Upper bound on `|S̃|` ensuring products fit within `i128` and SIMD `i64` accumulation without overflow.
pub const STATE_BOUND: i64 = 1 << 48;

/// State for a single affine head: the unscaled accumulator `s` and the deferred scalar decay `c`.
#[derive(Debug)]
pub struct AffineHead {
    /// The unscaled state matrix `S̃`, `head_k × head_v` elements, row `i` major.
    pub s: Vec<i64>,
    /// The deferred scalar decay factor `c`, in Q32.32 (starts at `1.0 = 1 << 32`).
    pub c: i64,
}

impl AffineHead {
    /// New zeroed head state with `c = 1.0`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        AffineHead {
            s: vec![0; len],
            c: 1i64 << 32,
        }
    }
}

impl Clone for AffineHead {
    fn clone(&self) -> Self {
        AffineHead {
            s: self.s.clone(),
            c: self.c,
        }
    }

    /// In-place copy reusing `s`'s buffer: the speculative rollback copies both the state and the
    /// deferred decay `c`, so a renormalisation inside a rejected column is undone too.
    fn clone_from(&mut self, src: &Self) {
        self.s.clone_from(&src.s);
        self.c = src.c;
    }
}

/// One affine layer's state.
///
/// - `conv`: the last `conv_kernel − 1` input columns, oldest first, `(k − 1)·channels` values.
/// - `heads`: `slices` [`AffineHead`]s per value head, head `h`'s slice `t` at `h·slices + t`,
///   each the `head_k × (head_v / slices)` value columns `t·w..(t + 1)·w` of the head, row `i`
///   (key index) major.
///
/// The delta rule is separable by value column: column `j` of the update, of `o` and of `S̃` reads
/// only column `j` and the head's shared scalars (`k`, `q̃`, `β`, the decay and hence `c`). So a
/// slice is itself a head of value width `w`, and slicing is bit-identical to the whole head
/// (`sliced_heads_equal_whole_heads`). Each slice sits behind its own lock, so the pooled step
/// hands each task exactly one slice without sharing. The lock is uncontended.
#[derive(Debug)]
pub struct AffineLayer {
    conv: Vec<i64>,
    heads: Vec<Mutex<AffineHead>>,
    slices: usize,
}

/// Value-column slices per affine head: four 32-column slices of a 128-wide head keep a slice's
/// state at 32 KiB (an L1d) and give the pool `4·n_v_heads` tasks. `APEIRON_HEAD_SLICES`
/// overrides it (read once); a count that does not divide the head width falls back to 1.
fn head_slices(head_v: usize) -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let n = *N.get_or_init(|| {
        std::env::var("APEIRON_HEAD_SLICES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    });
    if n > 0 && head_v.is_multiple_of(n) && (head_v / n).is_multiple_of(4) {
        n
    } else {
        1
    }
}

impl Clone for AffineLayer {
    fn clone(&self) -> Self {
        AffineLayer {
            slices: self.slices,
            conv: self.conv.clone(),
            heads: self
                .heads
                .iter()
                .map(|m| {
                    Mutex::new(
                        m.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .clone(),
                    )
                })
                .collect(),
        }
    }

    /// Copy `src` into `self` reusing every buffer: no allocation once the shapes match (the
    /// speculative decoder's per-round rollback snapshot).
    fn clone_from(&mut self, src: &Self) {
        self.conv.clone_from(&src.conv);
        if self.heads.len() != src.heads.len() {
            *self = src.clone();
            return;
        }
        for (dst, s) in self.heads.iter_mut().zip(&src.heads) {
            let s = s.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            dst.get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone_from(&s);
        }
    }
}

/// The affine state of every affine layer, in schedule order.
///
/// Its size is `O(1)` in the context length: per layer, `(k − 1)·channels + n_v_heads·head_k·head_v`
/// Q32.32 words. For Qwen3.5-0.8B that is 18 layers × (3·6144 + 16·128·128) words ≈ 38 MiB.
#[derive(Debug, Clone)]
pub struct AffineState {
    layers: Vec<AffineLayer>,
}

impl AffineState {
    /// Zeroed state for `n_layers` affine layers of geometry `g`.
    ///
    /// # Complexity
    ///
    /// `O(n_layers · state words)` zeroed words, allocated once.
    #[must_use]
    pub fn new(g: &AffineGeom, n_layers: usize) -> AffineState {
        Self::with_slices(g, n_layers, head_slices(g.head_v))
    }

    /// Zeroed state whose heads are split into `slices` value-column slices (1 is the whole head).
    ///
    /// # Complexity
    ///
    /// As [`AffineState::new`].
    #[must_use]
    pub fn with_slices(g: &AffineGeom, n_layers: usize, slices: usize) -> AffineState {
        let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
        let conv_len = g.conv_kernel.saturating_sub(1) * channels;
        let slices = slices.max(1);
        let slice_len = g.head_k * (g.head_v / slices);
        let layers = (0..n_layers)
            .map(|_| AffineLayer {
                conv: vec![0; conv_len],
                heads: (0..g.n_v_heads * slices)
                    .map(|_| Mutex::new(AffineHead::new(slice_len)))
                    .collect(),
                slices,
            })
            .collect();
        AffineState { layers }
    }

    /// Affine layers held.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layers(&self) -> &[AffineLayer] {
        &self.layers
    }

    pub(crate) fn layer_mut(&mut self, slot: usize) -> Option<&mut AffineLayer> {
        self.layers.get_mut(slot)
    }
}

/// The per-column spare-capacity writers that [`conv_silu_cols`] shares with the pool: element `j`
/// points at the first unwritten word of output column `j`.
struct ColWriters(Vec<*mut MaybeUninit<i64>>);

// SAFETY: the dispatch gives task `t` the channel range `[t·per, min(t·per + per, ch))` of every
// column, so no two tasks write the same word, and `by_col` is neither moved nor reallocated while
// the pool runs. The disjointness argument is the same one `SendPtr` (engine::chart::pool) uses for
// its disjoint chunks.
unsafe impl Sync for ColWriters {}

impl ColWriters {
    /// Write channel `c` of output column `j`.
    ///
    /// # Safety
    ///
    /// `j` is a valid column index, `c` is below that column's reserved capacity, and no other
    /// live writer targets the same `(j, c)` word.
    #[inline]
    unsafe fn write(&self, j: usize, c: usize, value: i64) {
        // SAFETY: the caller guarantees the index and the disjointness.
        unsafe { self.0[j].add(c).write(MaybeUninit::new(value)) };
    }
}

/// The causal depthwise convolution of `k` consecutive columns, then `SiLU`, shifting them into the
/// state (llama.cpp `build_conv_state`, ggml `ssm_conv`). Over the history
/// `[conv window rows | xs rows]`, column `j` reads rows `j..j + kernel`; the new window is the
/// history's last `kernel − 1` rows. Column `j` therefore equals the `j`-th of `k` one-column
/// calls. Channels are independent, so pool tasks own channel ranges and run every column.
///
/// # Complexity
///
/// `O(k · conv_kernel · channels)` `i128` multiply-adds and `k · channels` `SiLU`s over the pool,
/// `O((conv_kernel − 1)·channels)` window moves, and `k · channels` output words written once with
/// no channel-major staging buffer or transposing copy. One pool task per `per` channels.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "w (taps), x (input), c and t (channel and tap) are the convolution's symbols; the
              direct-to-column store needs the pool frame inline"
)]
pub(crate) fn conv_silu_cols(
    layer: &mut AffineLayer,
    w: &[i64],
    kernel: usize,
    xs: &[&[i64]],
) -> Result<Vec<Vec<i64>>, StepError> {
    let k = xs.len();
    let ch = xs.first().map_or(0, |x| x.len());
    let cols = kernel
        .checked_sub(1)
        .ok_or(StepError::BadConfig("conv_kernel"))?;
    if k == 0
        || xs.iter().any(|x| x.len() != ch)
        || w.len() != kernel * ch
        || layer.conv.len() != cols * ch
    {
        return Err(StepError::BadConfig("conv geometry"));
    }
    // History row `r`: window row `r` for `r < cols`, else column `r − cols`.
    let window = &layer.conv;
    let row = |r: usize, c: usize| {
        if r < cols {
            window[r * ch + c]
        } else {
            xs[r - cols][c]
        }
    };
    // Per-column outputs written in place: `by_col[j][c]` is channel `c` of column `j`. Every
    // column reserves `ch` words, so a task writes its channel's word of each column directly and
    // no channel-major staging buffer or serial transpose is needed.
    let mut by_col: Vec<Vec<i64>> = (0..k).map(|_| Vec::with_capacity(ch)).collect();
    let writers = ColWriters(
        by_col
            .iter_mut()
            .map(|v| v.spare_capacity_mut().as_mut_ptr())
            .collect(),
    );
    let first: std::sync::Mutex<Option<(usize, StepError)>> = std::sync::Mutex::new(None);
    let per = ch
        .div_ceil(crate::engine::par::row_chunk_budget(ch))
        .max(64);
    let tasks = ch.div_ceil(per);
    crate::engine::chart::pool::Pool::global().run(tasks, &|t| {
        let c0 = t * per;
        let c1 = (c0 + per).min(ch);
        for c in c0..c1 {
            let base = c * kernel;
            for j in 0..k {
                let mut acc: i128 = 0;
                let mut tap = 0usize;
                let mut acc_ok = true;
                while tap < kernel && acc_ok {
                    let prod = i128::from(w[base + tap]) * i128::from(row(j + tap, c));
                    match acc.checked_add(prod) {
                        Some(total) => acc = total,
                        None => acc_ok = false,
                    }
                    tap += 1;
                }
                if !acc_ok {
                    let mut f = first
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if f.as_ref().is_none_or(|(at, _)| c < *at) {
                        *f = Some((c, StepError::Overflow));
                    }
                    return;
                }
                // The shared `round_shift_32` adds 2^31 before shifting; guard that addition here
                // because this dot may be as large as i128::MAX.
                let v = match acc.checked_add(1_i128 << 31) {
                    Some(_) => i64::try_from(round_shift_32(acc)).map_err(|_| StepError::Overflow),
                    None => Err(StepError::Overflow),
                };
                match v.and_then(|v| silu_fixed(v).map_err(StepError::Ffn)) {
                    // SAFETY: `by_col[j]` reserved `ch` words; task `t` writes slot `c` of column
                    // `j` exactly once and no other task writes slot `c`.
                    Ok(y) => unsafe { writers.write(j, c, y) },
                    Err(e) => {
                        let mut f = first
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if f.as_ref().is_none_or(|(at, _)| c < *at) {
                            *f = Some((c, e));
                        }
                        return;
                    }
                }
            }
        }
    });
    if let Some((_, e)) = first
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return Err(e);
    }
    // Every index `0..ch` of every column was written exactly once above; this point is reached
    // only after a refusal-free dispatch, so the lengths can be published. The refusal path
    // returns before it, dropping length-0 vectors that never read an unwritten word.
    for col in &mut by_col {
        // SAFETY: the dispatch set no refusal, so all `ch` words are initialised.
        unsafe { col.set_len(ch) };
    }
    // The new window: the history's last `cols` rows.
    let mut next = vec![0i64; cols * ch];
    for r in 0..cols {
        for c in 0..ch {
            next[r * ch + c] = row(k + r, c);
        }
    }
    layer.conv = next;
    Ok(by_col)
}

/// `x / max(‖x‖₂, eps)` in Q32.32 (ggml `l2_norm`: `scale = 1 / fmaxf(sqrtf(Σx²), eps)`).
///
/// The norm is the integer square root of the exact `i128` sum of squares, rounded to nearest.
/// Each output is one round-to-nearest division.
///
/// # Complexity
///
/// `O(len)` `i128` multiply-adds, one `O(64)` integer square root and `O(len)` divisions.
pub(crate) fn l2_normalize(x: &[i64], eps: i64) -> Result<Vec<i64>, StepError> {
    let mut sum: i128 = 0;
    for &v in x {
        sum = sum
            .checked_add(i128::from(v) * i128::from(v))
            .ok_or(StepError::Overflow)?;
    }
    let s = isqrt(sum);
    let norm = if sum - s * s > s { s + 1 } else { s };
    let den = norm.max(i128::from(eps));
    if den <= 0 {
        return Err(StepError::BadConfig("l2 norm eps"));
    }
    let half = den >> 1;
    let mut out = Vec::with_capacity(x.len());
    for &v in x {
        let num = i128::from(v) << 32;
        let q = if num >= 0 {
            (num + half) / den
        } else {
            -((-num + half) / den)
        };
        out.push(i64::try_from(q).map_err(|_| StepError::Overflow)?);
    }
    Ok(out)
}

/// Exact Q32.32 division `round((num · 2³²) / den)` with round to nearest.
#[inline]
fn div_q32(num: i64, den: i64) -> Result<i64, StepError> {
    if den <= 0 {
        return Err(StepError::BadConfig("affine decay factor c <= 0"));
    }
    let num128 = i128::from(num) << 32;
    let half = i128::from(den >> 1);
    let q = if num128 >= 0 {
        (num128 + half) / i128::from(den)
    } else {
        -((-num128 + half) / i128::from(den))
    };
    i64::try_from(q).map_err(|_| StepError::Overflow)
}

/// Baseline Gated `DeltaNet` update in two fused passes (unoptimized v0).
/// Retained under `#[cfg(test)]` to verify and measure drift against `delta_head_step_ref`.
#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    reason = "S, q, k, v, d and o are the recurrence's standard symbols"
)]
pub(crate) fn delta_head_step_v0(
    s: &mut [i64],
    q: &[i64],
    k: &[i64],
    v: &[i64],
    decay: i64,
    beta: i64,
) -> Result<Vec<i64>, StepError> {
    let (dk, dv) = (k.len(), v.len());
    if s.len() != dk * dv || q.len() != dk {
        return Err(StepError::BadConfig("affine head geometry"));
    }
    let mul = |a: i64, b: i64| mul_q32(a, b).ok_or(StepError::Overflow);
    let mut d = Vec::with_capacity(dv);
    let mut j = 0usize;
    while j < dv {
        let mut acc: i128 = 0;
        let mut i = 0usize;
        while i < dk {
            let at = i * dv + j;
            let decayed = mul(s[at], decay)?;
            s[at] = decayed;
            acc += i128::from(decayed) * i128::from(k[i]);
            i += 1;
        }
        let m = i64::try_from(round_shift_32(acc)).map_err(|_| StepError::Overflow)?;
        d.push(mul(beta, v[j].checked_sub(m).ok_or(StepError::Overflow)?)?);
        j += 1;
    }
    let mut o = Vec::with_capacity(dv);
    let mut j = 0usize;
    while j < dv {
        let mut acc: i128 = 0;
        let mut i = 0usize;
        while i < dk {
            let at = i * dv + j;
            let w = s[at]
                .checked_add(mul(k[i], d[j])?)
                .ok_or(StepError::Overflow)?;
            s[at] = w;
            acc += i128::from(w) * i128::from(q[i]);
            i += 1;
        }
        o.push(i64::try_from(round_shift_32(acc)).map_err(|_| StepError::Overflow)?);
        j += 1;
    }
    Ok(o)
}

/// The scalar oracle reference for Gated `DeltaNet` at the algebraic minimum.
///
/// Features:
/// 1. Scalar decay factor `c` deferred across tokens;
/// 2. Renormalization fold `S̃ ← S̃ · c` when `c < 2⁻⁸`;
/// 3. One fused read pass computing `m̃ = S̃ᵀ k` and `õ = S̃ᵀ q̃`;
/// 4. Rank-1 update readout: `o = c·õ + d·(kᵀ q̃)`;
/// 5. One write pass: `S̃ += k · (d / c)ᵀ`.
///
/// Every optimized implementation must match this function bit for bit.
#[allow(
    clippy::many_single_char_names,
    reason = "s, c, q, k, v, d, o are the recurrence symbols"
)]
pub(crate) fn delta_head_step_ref(
    s: &mut [i64],
    c: &mut i64,
    q: &[i64],
    k: &[i64],
    v: &[i64],
    decay: i64,
    beta: i64,
) -> Result<Vec<i64>, StepError> {
    let (dk, dv) = (k.len(), v.len());
    if s.len() != dk * dv || q.len() != dk {
        return Err(StepError::BadConfig("affine head geometry"));
    }
    let mul = |a: i64, b: i64| mul_q32(a, b).ok_or(StepError::Overflow);

    // 1. Update deferred decay c
    *c = mul(*c, decay)?;
    if *c <= 0 {
        return Err(StepError::Overflow);
    }

    // Fold c into S̃ when c drops below RENORM_THRESHOLD (2⁻⁸)
    if *c < RENORM_THRESHOLD {
        let c_val = *c;
        for x in s.iter_mut() {
            *x = mul(*x, c_val)?;
        }
        *c = 1i64 << 32;
    }
    let c_val = *c;

    // Check bounds on S̃
    for &x in s.iter() {
        if x.abs() > STATE_BOUND {
            return Err(StepError::Overflow);
        }
    }

    // 2. Fused read pass: compute S̃ᵀ k and S̃ᵀ q̃ simultaneously in row-major order
    let mut acc_m = vec![0i128; dv];
    let mut acc_o = vec![0i128; dv];
    let mut acc_kq: i128 = 0;

    for i in 0..dk {
        let ki = i128::from(k[i]);
        let qi = i128::from(q[i]);
        acc_kq += ki * qi;
        let row = &s[i * dv..(i + 1) * dv];
        for j in 0..dv {
            let sij = i128::from(row[j]);
            acc_m[j] += sij * ki;
            acc_o[j] += sij * qi;
        }
    }
    let k_dot_q = i64::try_from(round_shift_32(acc_kq)).map_err(|_| StepError::Overflow)?;

    let mut d_prime = Vec::with_capacity(dv);
    let mut o = Vec::with_capacity(dv);

    for j in 0..dv {
        let m_tilde = i64::try_from(round_shift_32(acc_m[j])).map_err(|_| StepError::Overflow)?;
        let m = mul(c_val, m_tilde)?;
        let dj = mul(beta, v[j].checked_sub(m).ok_or(StepError::Overflow)?)?;
        d_prime.push(div_q32(dj, c_val)?);

        let o_tilde = i64::try_from(round_shift_32(acc_o[j])).map_err(|_| StepError::Overflow)?;
        let term1 = mul(c_val, o_tilde)?;
        let term2 = mul(dj, k_dot_q)?;
        o.push(term1.checked_add(term2).ok_or(StepError::Overflow)?);
    }

    // 3. Fused write pass: S̃ += k · d'ᵀ
    for i in 0..dk {
        let ki = k[i];
        if ki == 0 {
            continue;
        }
        let row = &mut s[i * dv..(i + 1) * dv];
        for j in 0..dv {
            let delta = mul(ki, d_prime[j])?;
            let updated = row[j].checked_add(delta).ok_or(StepError::Overflow)?;
            if updated.abs() > STATE_BOUND {
                return Err(StepError::Overflow);
            }
            row[j] = updated;
        }
    }

    Ok(o)
}

/// Fixed intermediate precision of the chunked state composition, in fractional bits.
///
/// Eight guard bits above the Q32.32 carrier: every composed word is rounded to `2^−40`, and the
/// entry decode, the readouts and the chunk-boundary state are the only roundings that reach
/// `2^−32`.
pub const CHUNK_PREC: u32 = 40;

/// Largest admissible chunk length. A `chunk` outside `1..=CHUNK_MAX` refuses with
/// [`StepError::BadConfig`] before any word is written.
///
/// Width derivation at `2^−CHUNK_PREC`: a state word is `|S̃| ≤ 2^56` (the Q32.32
/// [`STATE_BOUND`] at the guard precision), an injection `≤ 2^56` and a composed coefficient
/// `≤ 2^40`; the largest accumulator in [`delta_head_chunk`] is `2^96 + log₂(chunk)`, below
/// `2^127`, so the checked `i128` arithmetic holds for any `chunk < 2^31`. `CHUNK_MAX = 64` is a
/// throughput choice (the `u` and `D` working sets), not a width limit.
pub const CHUNK_MAX: usize = 64;

/// Round-to-nearest `v / 2^shift` for `shift >= 1`, ties upward: the [`round_shift_32`]
/// convention at the chunk precision.
#[inline]
fn chunk_round(v: i128, shift: u32) -> i128 {
    (v + (1_i128 << (shift - 1))) >> shift
}

/// [`chunk_round`] narrowed to `i64`; a value that leaves the carrier refuses.
#[inline]
fn chunk_narrow(v: i128, shift: u32) -> Result<i64, StepError> {
    i64::try_from(chunk_round(v, shift)).map_err(|_| StepError::Overflow)
}

/// `round(a·b / 2^{64−CHUNK_PREC})`: two Q32.32 words multiplied at the chunk precision.
#[inline]
fn mul_q32_prec(a: i64, b: i64) -> Result<i64, StepError> {
    chunk_narrow(i128::from(a) * i128::from(b), 64 - CHUNK_PREC)
}

/// `round(a·b / 2^32)`: a precision-`CHUNK_PREC` word times a Q32.32 word.
#[inline]
fn mul_prec_q32(a: i64, b: i64) -> Result<i64, StepError> {
    chunk_narrow(i128::from(a) * i128::from(b), 32)
}

/// `round(a·b / 2^CHUNK_PREC)`: two precision-`CHUNK_PREC` words.
#[inline]
fn mul_prec(a: i64, b: i64) -> Result<i64, StepError> {
    chunk_narrow(i128::from(a) * i128::from(b), CHUNK_PREC)
}

/// `round(⟨a, b⟩ / 2^{64−CHUNK_PREC})`: the Q32.32 inner product at the chunk precision.
fn dot_q32_prec(a: &[i64], b: &[i64]) -> Result<i64, StepError> {
    let mut acc: i128 = 0;
    for (&x, &y) in a.iter().zip(b) {
        acc = acc
            .checked_add(i128::from(x) * i128::from(y))
            .ok_or(StepError::Overflow)?;
    }
    chunk_narrow(acc, 64 - CHUNK_PREC)
}

/// `round(S0ᵀx / 2^32)`: every value column of the `dk × dv` chunk-precision state read against
/// the `dk`-long Q32.32 vector `x`, at the chunk precision.
///
/// # Complexity
///
/// `O(dk · dv)` checked `i128` multiply-adds and `O(dv)` outputs.
fn state_read_prec(s0: &[i64], dv: usize, x: &[i64]) -> Result<Vec<i64>, StepError> {
    let mut acc = vec![0i128; dv];
    for (i, &xi) in x.iter().enumerate() {
        let xw = i128::from(xi);
        let row = &s0[i * dv..(i + 1) * dv];
        for (a, &w) in acc.iter_mut().zip(row) {
            *a = a
                .checked_add(i128::from(w) * xw)
                .ok_or(StepError::Overflow)?;
        }
    }
    acc.into_iter().map(|a| chunk_narrow(a, 32)).collect()
}

/// One head's `chunk`-step Gated `DeltaNet` composition in the chunked (WY/UT) form, at the fixed
/// intermediate precision [`CHUNK_PREC`], with one rounding to Q32.32 at the chunk boundary.
///
/// Inputs: the entry state `s` (`dk × dv` Q32.32, row major) with its deferred decay `c`; per
/// step `t`, `q[t·dk..]`, `k[t·dk..]`, `v[t·dv..]` and the gates `decay[t]`, `beta[t]`. Output:
/// `chunk` readouts of `dv` words (`chunk·dv`, step major); `s` and `c` become the composed exit
/// state (real state at `c = 1`).
///
/// With `Q_t = α_t Q_{t−1} + k_t u_tᵀ`, `u_t = β_t(v_t − (α_t Q_{t−1})ᵀk_t)`, `G_t = ∏_{l≤t} α_l`
/// and `D(t←j) = ∏_{j<l≤t} α_l`, the composition is the forward substitution
/// `u_t = β_t v_t − α_t β_t G_{t−1}(S_0ᵀk_t) − Σ_{j<t} α_t β_t D(t−1←j)(k_jᵀk_t) u_j`, then
/// `o_t = G_t(S_0ᵀq_t) + Σ_{j≤t} D(t←j)(k_jᵀq_t) u_j` and
/// `S_chunk = G_{C−1}S_0 + Σ_j D(C−1←j) k_j u_jᵀ`. Every product is rounded to `2^−CHUNK_PREC` in
/// checked `i128` (the one fixed bracketing); the entry decode, the readouts and the exit state
/// are the roundings that reach the Q32.32 carrier. A refusal leaves `s` and `c` untouched.
/// `chunk == 1` has no composition to amortise and reduces to [`delta_head_step_ref`].
///
/// # Complexity
///
/// `O(chunk·dk·dv + chunk²·(dk + dv))` checked `i128` multiply-adds over `O(dk·dv + chunk·dv)`
/// scratch words; the state read and write are each one pass over `dk·dv`.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    dead_code,
    reason = "dk, dv, C, G, D, u, o are the recurrence's symbols; geometry, entry state and the \
              per-step gates are the kernel's input; the chunk path is reached from \
              engine::mixer::tests until the prefill wiring lands"
)]
pub(crate) fn delta_head_chunk(
    s: &mut [i64],
    c: &mut i64,
    q: &[i64],
    k: &[i64],
    v: &[i64],
    decay: &[i64],
    beta: &[i64],
    dk: usize,
    dv: usize,
    chunk: usize,
) -> Result<Vec<i64>, StepError> {
    if chunk == 0 || chunk > CHUNK_MAX || decay.len() != chunk || beta.len() != chunk {
        return Err(StepError::BadConfig("affine chunk length"));
    }
    if dk == 0
        || dv == 0
        || s.len() != dk * dv
        || q.len() != chunk * dk
        || k.len() != chunk * dk
        || v.len() != chunk * dv
    {
        return Err(StepError::BadConfig("affine head geometry"));
    }
    if *c <= 0 {
        return Err(StepError::Overflow);
    }
    for &w in s.iter() {
        if w.abs() > STATE_BOUND {
            return Err(StepError::Overflow);
        }
    }
    // A one-step chunk has no composition to amortise: the step oracle is the chunk.
    if chunk == 1 {
        return delta_head_step_ref(s, c, q, k, v, decay[0], beta[0]);
    }

    // Entry state decoded to the chunk precision. `|S̃| ≤ STATE_BOUND` and `c ≤ 1`, so the product
    // fits i128 and the rounded word fits i64.
    let s0: Vec<i64> = s
        .iter()
        .map(|&w| mul_q32_prec(w, *c))
        .collect::<Result<_, _>>()?;

    // Cumulative decays `G_t` at the chunk precision.
    let one = 1i64 << CHUNK_PREC;
    let mut g = Vec::with_capacity(chunk);
    let mut g_prev = one;
    for &a in decay {
        g_prev = mul_prec_q32(g_prev, a)?;
        g.push(g_prev);
    }

    let mut u = vec![0i64; chunk * dv];
    let mut out = vec![0i64; chunk * dv];
    for t in 0..chunk {
        let kt = &k[t * dk..(t + 1) * dk];
        // The read at step `t` sees the decayed state `α_t Q_{t−1}`: both the entry term and the
        // substitution coefficient carry `α_t β_t`, the injection term only `β_t`.
        let alpha_beta = mul_q32_prec(beta[t], decay[t])?;
        // Right-hand side `β_t v_t − α_t β_t G_{t−1}(S_0ᵀk_t)`.
        let s0k = state_read_prec(&s0, dv, kt)?;
        let beta_g = mul_prec(alpha_beta, if t == 0 { one } else { g[t - 1] })?;
        let mut ut = Vec::with_capacity(dv);
        for (jv, &vj) in v[t * dv..(t + 1) * dv].iter().enumerate() {
            let vt = mul_q32_prec(beta[t], vj)?;
            let sub = mul_prec(beta_g, s0k[jv])?;
            ut.push(vt.checked_sub(sub).ok_or(StepError::Overflow)?);
        }
        // Forward substitution, `D(t−1←t−1) = 1` walking `j` downward.
        let mut d = one;
        for j in (0..t).rev() {
            let dot = dot_q32_prec(&k[j * dk..(j + 1) * dk], kt)?;
            let coef = mul_prec(mul_prec(alpha_beta, d)?, dot)?;
            let uj = &u[j * dv..(j + 1) * dv];
            for (a, &b) in ut.iter_mut().zip(uj) {
                *a = a
                    .checked_sub(mul_prec(coef, b)?)
                    .ok_or(StepError::Overflow)?;
            }
            d = mul_prec_q32(d, decay[j])?;
        }
        u[t * dv..(t + 1) * dv].copy_from_slice(&ut);

        // Readout `o_t = G_t(S_0ᵀq_t) + Σ_{j≤t} D(t←j)(k_jᵀq_t) u_j`.
        let qt = &q[t * dk..(t + 1) * dk];
        let s0q = state_read_prec(&s0, dv, qt)?;
        let ot = &mut out[t * dv..(t + 1) * dv];
        for (a, &b) in ot.iter_mut().zip(&s0q) {
            *a = mul_prec(g[t], b)?;
        }
        let mut d = one;
        for j in (0..=t).rev() {
            let dot = dot_q32_prec(&k[j * dk..(j + 1) * dk], qt)?;
            let coef = mul_prec(d, dot)?;
            let uj = &u[j * dv..(j + 1) * dv];
            for (a, &b) in ot.iter_mut().zip(uj) {
                *a = a
                    .checked_add(mul_prec(coef, b)?)
                    .ok_or(StepError::Overflow)?;
            }
            if j > 0 {
                d = mul_prec_q32(d, decay[j])?;
            }
        }
    }

    // Composed exit state `S_C = G_{C−1}S_0 + Σ_j D(C−1←j) k_j u_jᵀ`.
    let mut acc: Vec<i128> = s0
        .iter()
        .map(|&w| Ok(i128::from(mul_prec(g[chunk - 1], w)?)))
        .collect::<Result<_, StepError>>()?;
    let mut d = one;
    for j in (0..chunk).rev() {
        let uj = &u[j * dv..(j + 1) * dv];
        let kj = &k[j * dk..(j + 1) * dk];
        for (i, &ki) in kj.iter().enumerate() {
            for (jv, &b) in uj.iter().enumerate() {
                let term = mul_prec(d, mul_prec_q32(ki, b)?)?;
                let slot = &mut acc[i * dv + jv];
                *slot = slot
                    .checked_add(i128::from(term))
                    .ok_or(StepError::Overflow)?;
            }
        }
        if j > 0 {
            d = mul_prec_q32(d, decay[j])?;
        }
    }
    // Narrow once to Q32.32 and check the width before the state is written.
    let mut next_s = Vec::with_capacity(dk * dv);
    for a in acc {
        let w = chunk_narrow(a, CHUNK_PREC - 32)?;
        if w.abs() > STATE_BOUND {
            return Err(StepError::Overflow);
        }
        next_s.push(w);
    }
    let mut next_o = Vec::with_capacity(chunk * dv);
    for w in out {
        next_o.push(chunk_narrow(i128::from(w), CHUNK_PREC - 32)?);
    }
    s.copy_from_slice(&next_s);
    *c = 1i64 << 32;
    Ok(next_o)
}

#[cfg(target_arch = "x86_64")]
mod avx2_kern {
    use super::{StepError, STATE_BOUND};
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_loadu_si256,
        _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi64x, _mm256_setzero_si256,
        _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
        _mm256_testz_si256, _mm256_xor_si256,
    };

    /// Pure AVX2 vector `mul_q32` evaluating 4 lanes of `round_shift_32(a · b[lane])`.
    ///
    /// Operand bounds:
    /// `|a| ≤ 2³²` (scalar `k[i]`, where `‖k‖₂ = 1`), `|b[lane]| ≤ 2⁴⁶` (`d' = d / c`, where `c ≥ 2⁻⁸`).
    ///
    /// The formula:
    /// `(a·b + 2³¹) >> 32 = a_hi·b_hi·2³² + a_hi·b_lo + a_lo·b_hi + ((a_lo·b_lo + 2³¹) >> 32)`.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn mul_q32_4x_avx2(a_hi: i64, a_lo: i64, b: __m256i) -> __m256i {
        let v_a_lo = _mm256_set1_epi64x(a_lo);
        let mask32 = _mm256_set1_epi64x(0xFFFF_FFFF);
        let sign_bit = _mm256_set1_epi64x(0x8000_0000);
        let half = _mm256_set1_epi64x(1 << 31);

        let b_lo = _mm256_and_si256(b, mask32);
        let b_srli = _mm256_srli_epi64(b, 32);
        let b_hi = _mm256_sub_epi64(_mm256_xor_si256(b_srli, sign_bit), sign_bit);

        // part 4: ((a_lo * b_lo + 2^31) >> 32)
        let prod_lo = _mm256_mul_epu32(v_a_lo, b_lo);
        let part4 = _mm256_srli_epi64(_mm256_add_epi64(prod_lo, half), 32);

        // part 1: a_hi * b_hi * 2^32 (a_hi in {-1, 0, 1})
        let part1 = if a_hi == 1 {
            _mm256_slli_epi64(b_hi, 32)
        } else if a_hi == -1 {
            _mm256_slli_epi64(_mm256_sub_epi64(_mm256_setzero_si256(), b_hi), 32)
        } else {
            _mm256_setzero_si256()
        };

        // part 2: a_hi * b_lo
        let part2 = if a_hi == 1 {
            b_lo
        } else if a_hi == -1 {
            _mm256_sub_epi64(_mm256_setzero_si256(), b_lo)
        } else {
            _mm256_setzero_si256()
        };

        // part 3: a_lo * b_hi (a_lo in [0, 2^32-1], b_hi signed)
        let is_neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), b_hi);
        let abs_b_hi = _mm256_sub_epi64(_mm256_xor_si256(b_hi, is_neg), is_neg);
        let prod_u = _mm256_mul_epu32(v_a_lo, abs_b_hi);
        let part3 = _mm256_sub_epi64(_mm256_xor_si256(prod_u, is_neg), is_neg);

        let sum1 = _mm256_add_epi64(part1, part2);
        let sum2 = _mm256_add_epi64(part3, part4);
        _mm256_add_epi64(sum1, sum2)
    }

    /// AVX2 write pass: `S̃[i*dv + j] += mul_q32(k[i], d'[j])` across 4 lanes at a time.
    /// For 128 key rows and `dv` values, it performs at most `128 * dv / 4` vector updates, each
    /// OR-ing the `|res| > STATE_BOUND` predicate into a flag register, and `O(1)` extra words: the
    /// bound refusal needs no second pass over the state.
    ///
    /// The predicate `cmpgt(res, B) | cmpgt(-B-1, res)` is `|res| > B` on the signed i64 lanes,
    /// including `i64::MIN` (`|i64::MIN|` is not representable; the second compare catches it).
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::cast_ptr_alignment,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::needless_range_loop,
        reason = "unaligned SIMD load/store and 32-bit integer sign/bit operations"
    )]
    pub(super) unsafe fn write_pass_avx2(
        s: &mut [i64],
        k: &[i64],
        d_prime: &[i64],
        dv: usize,
    ) -> Result<(), StepError> {
        let dk = 128;
        let bound = _mm256_set1_epi64x(STATE_BOUND);
        let below = _mm256_set1_epi64x(-STATE_BOUND - 1);
        let mut over = _mm256_setzero_si256();
        for i in 0..dk {
            let ki = k[i];
            if ki == 0 {
                continue;
            }
            let a_hi = ki >> 32;
            let a_lo = (ki as u64 & 0xFFFF_FFFF) as i64;
            let row_ptr = s.as_mut_ptr().add(i * dv);
            let d_ptr = d_prime.as_ptr();

            for j in (0..dv).step_by(4) {
                let b = _mm256_loadu_si256(d_ptr.add(j).cast::<__m256i>());
                let prod = mul_q32_4x_avx2(a_hi, a_lo, b);
                let cur = _mm256_loadu_si256(row_ptr.add(j).cast::<__m256i>());
                let res = _mm256_add_epi64(cur, prod);
                over = _mm256_or_si256(
                    over,
                    _mm256_or_si256(
                        _mm256_cmpgt_epi64(res, bound),
                        _mm256_cmpgt_epi64(below, res),
                    ),
                );
                _mm256_storeu_si256(row_ptr.add(j).cast::<__m256i>(), res);
            }
        }
        if _mm256_testz_si256(over, over) == 0 {
            return Err(StepError::Overflow);
        }
        Ok(())
    }

    /// AVX2 fused read accumulation: computes 4 columns of `S̃ᵀ k` and `S̃ᵀ q̃` simultaneously.
    #[target_feature(enable = "avx2")]
    #[allow(
        clippy::cast_ptr_alignment,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::needless_range_loop,
        reason = "unaligned SIMD load/store and 32-bit integer sign/bit operations"
    )]
    pub(super) unsafe fn read_accum_fused_4_avx2(
        s: &[i64],
        k: &[i64],
        q: &[i64],
        j_col: usize,
        dv: usize,
    ) -> (__m256i, __m256i) {
        let mask32 = _mm256_set1_epi64x(0xFFFF_FFFF);
        let sign_bit = _mm256_set1_epi64x(0x8000_0000);
        let half = _mm256_set1_epi64x(1 << 31);

        let mut sum_hi_k = _mm256_setzero_si256();
        let mut sum_mid_k = _mm256_setzero_si256();
        let mut sum_lo_k = _mm256_setzero_si256();

        let mut sum_hi_q = _mm256_setzero_si256();
        let mut sum_mid_q = _mm256_setzero_si256();
        let mut sum_lo_q = _mm256_setzero_si256();

        let dk = 128;

        for i in 0..dk {
            let ki = k[i];
            let qi = q[i];
            if ki == 0 && qi == 0 {
                continue;
            }

            let row_ptr = s.as_ptr().add(i * dv + j_col);
            let s_vec = _mm256_loadu_si256(row_ptr.cast::<__m256i>());

            let s_lo = _mm256_and_si256(s_vec, mask32);
            let s_srli = _mm256_srli_epi64(s_vec, 32);
            let s_hi = _mm256_sub_epi64(_mm256_xor_si256(s_srli, sign_bit), sign_bit);

            let is_neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), s_hi);
            let abs_s_hi = _mm256_sub_epi64(_mm256_xor_si256(s_hi, is_neg), is_neg);

            if ki != 0 {
                let k_hi = ki >> 32;
                let k_lo = (ki as u64 & 0xFFFF_FFFF) as i64;
                let v_k_lo = _mm256_set1_epi64x(k_lo);

                let prod_lo = _mm256_mul_epu32(s_lo, v_k_lo);
                let p_lo_lo = _mm256_and_si256(prod_lo, mask32);
                let p_lo_hi = _mm256_srli_epi64(prod_lo, 32);
                sum_lo_k = _mm256_add_epi64(sum_lo_k, p_lo_lo);

                let term_k_hi_s_lo = if k_hi == 1 {
                    s_lo
                } else if k_hi == -1 {
                    _mm256_sub_epi64(_mm256_setzero_si256(), s_lo)
                } else {
                    _mm256_setzero_si256()
                };

                let prod_u = _mm256_mul_epu32(abs_s_hi, v_k_lo);
                let s_hi_k_lo = _mm256_sub_epi64(_mm256_xor_si256(prod_u, is_neg), is_neg);
                let mid = _mm256_add_epi64(_mm256_add_epi64(s_hi_k_lo, term_k_hi_s_lo), p_lo_hi);
                sum_mid_k = _mm256_add_epi64(sum_mid_k, mid);

                let term_hi = if k_hi == 1 {
                    s_hi
                } else if k_hi == -1 {
                    _mm256_sub_epi64(_mm256_setzero_si256(), s_hi)
                } else {
                    _mm256_setzero_si256()
                };
                sum_hi_k = _mm256_add_epi64(sum_hi_k, term_hi);
            }

            if qi != 0 {
                let q_hi = qi >> 32;
                let q_lo = (qi as u64 & 0xFFFF_FFFF) as i64;
                let v_q_lo = _mm256_set1_epi64x(q_lo);

                let prod_lo = _mm256_mul_epu32(s_lo, v_q_lo);
                let p_lo_lo = _mm256_and_si256(prod_lo, mask32);
                let p_lo_hi = _mm256_srli_epi64(prod_lo, 32);
                sum_lo_q = _mm256_add_epi64(sum_lo_q, p_lo_lo);

                let term_q_hi_s_lo = if q_hi == 1 {
                    s_lo
                } else if q_hi == -1 {
                    _mm256_sub_epi64(_mm256_setzero_si256(), s_lo)
                } else {
                    _mm256_setzero_si256()
                };

                let prod_u = _mm256_mul_epu32(abs_s_hi, v_q_lo);
                let s_hi_q_lo = _mm256_sub_epi64(_mm256_xor_si256(prod_u, is_neg), is_neg);
                let mid = _mm256_add_epi64(_mm256_add_epi64(s_hi_q_lo, term_q_hi_s_lo), p_lo_hi);
                sum_mid_q = _mm256_add_epi64(sum_mid_q, mid);

                let term_hi = if q_hi == 1 {
                    s_hi
                } else if q_hi == -1 {
                    _mm256_sub_epi64(_mm256_setzero_si256(), s_hi)
                } else {
                    _mm256_setzero_si256()
                };
                sum_hi_q = _mm256_add_epi64(sum_hi_q, term_hi);
            }
        }

        let lo_rounded_k = _mm256_srli_epi64(_mm256_add_epi64(sum_lo_k, half), 32);
        let hi_shifted_k = _mm256_slli_epi64(sum_hi_k, 32);
        let res_k = _mm256_add_epi64(_mm256_add_epi64(hi_shifted_k, sum_mid_k), lo_rounded_k);

        let lo_rounded_q = _mm256_srli_epi64(_mm256_add_epi64(sum_lo_q, half), 32);
        let hi_shifted_q = _mm256_slli_epi64(sum_hi_q, 32);
        let res_q = _mm256_add_epi64(_mm256_add_epi64(hi_shifted_q, sum_mid_q), lo_rounded_q);

        (res_k, res_q)
    }
}

/// Vectorized AVX2 Gated `DeltaNet` head update.
#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_ptr_alignment,
    reason = "s, c, q, k, v, d, o are recurrence symbols; SIMD store unaligned"
)]
pub(crate) fn delta_head_step_fast(
    s: &mut [i64],
    c: &mut i64,
    q: &[i64],
    k: &[i64],
    v: &[i64],
    decay: i64,
    beta: i64,
) -> Result<Vec<i64>, StepError> {
    let (dk, dv) = (k.len(), v.len());
    // Any value width that is a whole number of 4-lane vectors up to 128: a value-column slice of
    // a head (`HEAD_SLICES`) runs the same kernel on its own columns.
    if dk != 128 || dv % 4 != 0 || dv > 128 {
        return delta_head_step_ref(s, c, q, k, v, decay, beta);
    }
    if s.len() != dk * dv || q.len() != dk {
        return Err(StepError::BadConfig("affine head geometry"));
    }
    let mul = |a: i64, b: i64| mul_q32(a, b).ok_or(StepError::Overflow);

    // 1. Update deferred decay c
    *c = mul(*c, decay)?;
    if *c <= 0 {
        return Err(StepError::Overflow);
    }

    if *c < RENORM_THRESHOLD {
        let c_val = *c;
        for x in s.iter_mut() {
            *x = mul(*x, c_val)?;
        }
        *c = 1i64 << 32;
    }
    let c_val = *c;

    for &x in s.iter() {
        if x.abs() > STATE_BOUND {
            return Err(StepError::Overflow);
        }
    }

    // 2. Fused read pass in AVX2
    let mut m_tilde = [0i64; 128];
    let mut o_tilde = [0i64; 128];
    let mut acc_kq: i128 = 0;
    for i in 0..dk {
        acc_kq += i128::from(k[i]) * i128::from(q[i]);
    }
    let k_dot_q = i64::try_from(round_shift_32(acc_kq)).map_err(|_| StepError::Overflow)?;

    // SAFETY: dk == 128 and dv ≤ 128, a multiple of 4, checked above; host supports AVX2.
    unsafe {
        use std::arch::x86_64::_mm256_storeu_si256;
        for j_col in (0..dv).step_by(4) {
            let (rk, rq) = avx2_kern::read_accum_fused_4_avx2(s, k, q, j_col, dv);
            _mm256_storeu_si256(
                m_tilde
                    .as_mut_ptr()
                    .add(j_col)
                    .cast::<std::arch::x86_64::__m256i>(),
                rk,
            );
            _mm256_storeu_si256(
                o_tilde
                    .as_mut_ptr()
                    .add(j_col)
                    .cast::<std::arch::x86_64::__m256i>(),
                rq,
            );
        }
    }

    let mut d_prime = [0i64; 128];
    let mut o = Vec::with_capacity(dv);

    for j in 0..dv {
        let m = mul(c_val, m_tilde[j])?;
        let dj = mul(beta, v[j].checked_sub(m).ok_or(StepError::Overflow)?)?;
        d_prime[j] = div_q32(dj, c_val)?;

        let term1 = mul(c_val, o_tilde[j])?;
        let term2 = mul(dj, k_dot_q)?;
        o.push(term1.checked_add(term2).ok_or(StepError::Overflow)?);
    }

    // 3. Fused write pass in AVX2
    // SAFETY: dk == 128 and dv ≤ 128, a multiple of 4, checked above; host supports AVX2.
    unsafe {
        avx2_kern::write_pass_avx2(s, k, &d_prime[..dv], dv)?;
    }

    Ok(o)
}

/// The Gated `DeltaNet` instance of the affine update on one head.
/// Dispatches to the AVX2 SIMD core when available, falling back to the scalar reference oracle.
#[allow(
    clippy::many_single_char_names,
    reason = "S, q, k, v, d and o are the recurrence's standard symbols"
)]
pub(crate) fn delta_head_step(
    s: &mut [i64],
    c: &mut i64,
    q: &[i64],
    k: &[i64],
    v: &[i64],
    decay: i64,
    beta: i64,
) -> Result<Vec<i64>, StepError> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return delta_head_step_fast(s, c, q, k, v, decay, beta);
    }
    delta_head_step_ref(s, c, q, k, v, decay, beta)
}

/// One column's inputs to the affine heads: the convolved `[q | k | v]`, the gate input `z`, and
/// the per-head `β` and decay inputs `b`, `a`.
pub(crate) type AffineCol<'a> = (&'a [i64], &'a [i64], &'a [i64], &'a [i64]);

/// Every value head of one affine layer for one column: [`delta_heads_cols`] with `k = 1` (the
/// microbenchmark's entry point).
///
/// # Complexity
///
/// As [`delta_heads_cols`] with one column.
#[cfg(test)]
#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "the layer state, geometry, parameters and the four input vectors are the step's \
              input; z, b, a are the architecture's names for the gate and the β/decay inputs"
)]
pub(crate) fn delta_heads(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    conv: &[i64],
    z: &[i64],
    b: &[i64],
    a: &[i64],
    eps: i64,
    parallel: bool,
) -> Result<Vec<i64>, StepError> {
    let mut out = delta_heads_cols(layer, g, p, &[(conv, z, b, a)], eps, parallel, None)?;
    out.pop().ok_or(StepError::BadConfig("affine column"))
}

/// Every value head of one affine layer over `k` columns in token order, from each column's
/// convolved `[q | k | v]`, gate input `z` and per-head `β`/`α` inputs `b`/`a`. Returns each
/// column's gated head outputs, `n_v_heads·head_v` wide.
///
/// Per head `h` (key head `h mod n_k_heads`, llama.cpp `qwen35.cpp` repeat order), for each
/// column in order:
/// 1. `q̂, k̂` are L2-normalised; `q̃ = q̂/√head_k`.
/// 2. `β = σ(b_h)`; `α = exp(ssm_a_h · softplus(a_h + dt_h))`.
/// 3. `o_h = delta_head_step(S_h, q̃, k̂, v_h, α, β)`.
/// 4. `out_h = rms(o_h) ⊙ ssm_norm ⊙ SiLU(z_h)`.
///
/// Three pooled stages: (0) step 1 per key head and column, once; (0.5) the per-head `β`/`α`
/// gates per (value head, column), once, shared by every value slice; (1) steps 2–3 per
/// value-column slice of each head ([`AffineLayer`]), one task carrying its slice through all `k`
/// columns in token order while its state is hot in cache; (2) step 4 per head on the reassembled
/// `o_h`.
/// Heads and slices are independent and each slice's updates run in token order, so the result
/// equals `k` one-column calls on whole heads. With `snapshot`, each stage-1 task copies its slice
/// (state and deferred decay) into the snapshot after column 0: the speculative rollback point.
/// A refusal in stage 0 leaves the state untouched; later refusals leave it unspecified.
///
/// # Complexity
///
/// `O(k · n_v_heads · head_k · head_v)` for the updates, plus `O(k · channels)` for the norms, over
/// `n_k_heads`, `n_v_heads · slices` and `n_v_heads` pool tasks when `parallel`.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "the layer state, geometry, parameters, columns and snapshot are the step's input; \
              z, b, a are the architecture's names for the gate and the β/decay inputs; the three \
              stages read in order"
)]
pub(crate) fn delta_heads_cols(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    cols: &[AffineCol<'_>],
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
) -> Result<Vec<Vec<i64>>, StepError> {
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let k = cols.len();
    for (conv, z, b, a) in cols {
        if conv.len() != 2 * key_dim + val_dim
            || z.len() != val_dim
            || b.len() != g.n_v_heads
            || a.len() != g.n_v_heads
        {
            return Err(StepError::BadConfig("affine layer inputs"));
        }
    }
    let slices = layer.slices;
    let w = g.head_v / slices;
    let n_tasks = g.n_v_heads * slices;
    if layer.heads.len() != n_tasks || snapshot.is_some_and(|s| s.heads.len() != n_tasks) {
        return Err(StepError::BadConfig("affine layer heads"));
    }
    let head_k_fixed = i64::try_from(g.head_k)
        .ok()
        .and_then(|v| v.checked_shl(32))
        .ok_or(StepError::Overflow)?;
    let q_scale = fixed_rsqrt(head_k_fixed).map_err(|_| StepError::Overflow)?;
    let hk = g.head_k;
    // Stage 0, per key head: every column's `q̃` then `k̂`, `qk[kh·k·2hk + c·2hk ..]`, computed once
    // and shared by every value head and slice that reads that key head.
    let mut qk = vec![0i64; k * 2 * key_dim];
    let norm_qk = |kh: usize, dst: &mut [i64]| -> Result<(), StepError> {
        let kr = kh * hk..(kh + 1) * hk;
        for (c, (conv, ..)) in cols.iter().enumerate() {
            let (q_all, rest) = conv.split_at(key_dim);
            let q_hat = l2_normalize(&q_all[kr.clone()], eps)?;
            let k_hat = l2_normalize(&rest[kr.clone()], eps)?;
            let d = &mut dst[c * 2 * hk..(c + 1) * 2 * hk];
            for (x, &q) in d[..hk].iter_mut().zip(&q_hat) {
                *x = mul_q32(q, q_scale).ok_or(StepError::Overflow)?;
            }
            d[hk..].copy_from_slice(&k_hat);
        }
        Ok(())
    };
    if let Some((_, e)) = attn_heads(g.n_k_heads, k * 2 * hk, &mut qk, parallel, norm_qk) {
        return Err(e);
    }
    // Stage 0.5, per value head and column: β = σ(b_h) and α = exp(ssm_a_h·softplus(a_h + dt_h))
    // read only the column and the head, never the value slice, so compute them once for all
    // `slices` slice tasks (`bd[h·k·2 + c·2]`, `bd[h·k·2 + c·2 + 1]`). Every fixed_* refusal maps
    // to Overflow, exactly as it did inside stage 1, but now before any state update.
    let mut bd = vec![0i64; g.n_v_heads * k * 2];
    let head_gates = |h: usize, dst: &mut [i64]| -> Result<(), StepError> {
        for (c, (_, _, b, a)) in cols.iter().enumerate() {
            dst[c * 2] = fixed_sigmoid(b[h]).map_err(|_| StepError::Overflow)?;
            let sp = fixed_softplus(a[h].checked_add(p.dt[h]).ok_or(StepError::Overflow)?)
                .map_err(|_| StepError::Overflow)?;
            dst[c * 2 + 1] = fixed_exp(mul_q32(p.a[h], sp).ok_or(StepError::Overflow)?)
                .map_err(|_| StepError::Overflow)?;
        }
        Ok(())
    };
    if let Some((_, e)) = attn_heads(g.n_v_heads, k * 2, &mut bd, parallel, head_gates) {
        return Err(e);
    }
    // Stage 1, per (value head, slice): the slice's update through all k columns in token order,
    // its state hot in cache; `raw[t·k·w + c·w ..]`. The rollback copy is taken inside the task.
    let mut raw = vec![0i64; k * val_dim];
    let update = |t: usize, dst: &mut [i64]| -> Result<(), StepError> {
        let (h, sl) = (t / slices, t % slices);
        let kh = h % g.n_k_heads;
        let vr = h * g.head_v + sl * w..h * g.head_v + (sl + 1) * w;
        let mut guard = layer.heads[t]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (c, (conv, ..)) in cols.iter().enumerate() {
            if c == 1 {
                if let Some(snap) = snapshot {
                    snap.heads[t]
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone_from(&guard);
                }
            }
            let at = kh * k * 2 * hk + c * 2 * hk;
            let (q_tilde, k_hat) = qk[at..at + 2 * hk].split_at(hk);
            let beta = bd[h * k * 2 + c * 2];
            let decay = bd[h * k * 2 + c * 2 + 1];
            let v = &conv[2 * key_dim + vr.start..2 * key_dim + vr.end];
            let AffineHead { s, c: scale } = &mut *guard;
            let o = delta_head_step(s, scale, q_tilde, k_hat, v, decay, beta)?;
            dst[c * w..(c + 1) * w].copy_from_slice(&o);
        }
        Ok(())
    };
    if let Some((_, e)) = attn_heads(n_tasks, k * w, &mut raw, parallel, update) {
        return Err(e);
    }
    // Stage 2, per value head: reassemble each column's `o_h` from its slices, then the gated
    // norm. Per head, the k columns' outputs side by side: `out[h·k·head_v + c·head_v + j]`.
    let mut out = vec![0i64; k * val_dim];
    let gate_norm = |h: usize, dst: &mut [i64]| -> Result<(), StepError> {
        let vr = h * g.head_v..(h + 1) * g.head_v;
        let mut o = vec![0i64; g.head_v];
        for (c, (_, z, ..)) in cols.iter().enumerate() {
            for sl in 0..slices {
                let at = (h * slices + sl) * k * w + c * w;
                o[sl * w..(sl + 1) * w].copy_from_slice(&raw[at..at + w]);
            }
            let normed = rms_norm_fixed(&o, eps).map_err(StepError::Norm)?;
            let zh = &z[vr.clone()];
            let dcol = &mut dst[c * g.head_v..(c + 1) * g.head_v];
            let mut j = 0usize;
            while j < g.head_v {
                let gained = mul_q32(normed[j], p.norm[j]).ok_or(StepError::Overflow)?;
                let gate = silu_fixed(zh[j]).map_err(StepError::Ffn)?;
                dcol[j] = mul_q32(gained, gate).ok_or(StepError::Overflow)?;
                j += 1;
            }
        }
        Ok(())
    };
    if let Some((_, e)) = attn_heads(g.n_v_heads, k * g.head_v, &mut out, parallel, gate_norm) {
        return Err(e);
    }
    let mut cols_out: Vec<Vec<i64>> = (0..k).map(|_| Vec::with_capacity(val_dim)).collect();
    for h in 0..g.n_v_heads {
        for (c, col) in cols_out.iter_mut().enumerate() {
            let at = h * k * g.head_v + c * g.head_v;
            col.extend_from_slice(&out[at..at + g.head_v]);
        }
    }
    Ok(cols_out)
}

impl AffineLayer {
    /// The rollback point's conv half: `src`'s window after one more column `x0`, i.e. its rows
    /// `1..` followed by `x0`, written in place.
    pub(crate) fn snapshot_conv_after(&mut self, src: &AffineLayer, x0: &[i64]) {
        let ch = x0.len();
        self.conv.clear();
        if src.conv.len() >= ch && ch > 0 {
            self.conv.extend_from_slice(&src.conv[ch..]);
            self.conv.extend_from_slice(x0);
        }
    }
}

#[cfg(test)]
mod tests;
