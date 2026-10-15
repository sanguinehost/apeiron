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

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::ref_option,
    clippy::too_many_arguments,
    reason = "dense numeric kernels name operands as in the recurrences they implement; flat-buffer helpers take the geometry and buffers explicitly, and a borrowed Option keeps one parameter level for the extraction"
)]

use crate::engine::chart::pool::{set_written_lens, Excl};
use crate::engine::chart::task::{ChunkTask, HeadTask, IndexTask};
use std::mem::MaybeUninit;

use super::chart::ffn::attn_heads;
use super::step::StepError;
use super::value::{mul_q32, round_shift_32};
use super::{rms_norm_fixed, silu_fixed, AffineGeom, AffineVectors};
use crate::engine::fp::F32;
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_ln::{fixed_sigmoid, fixed_softplus};
use crate::numeric::fixed_rsqrt::{fixed_rsqrt, isqrt};

#[path = "mixer/conv_f32.rs"]
mod conv_f32;
#[path = "mixer/f32_carrier.rs"]
pub(crate) mod f32_carrier;
use conv_f32::conv_silu_cols_f32;

/// Threshold below which the central decay scalar `c` is folded back into `S̃` (2⁻⁸ in Q32.32).
pub const RENORM_THRESHOLD: i64 = 1 << 24;

/// Upper bound on `|S̃|` ensuring products fit within `i128` and SIMD `i64` accumulation without overflow.
pub const STATE_BOUND: i64 = 1 << 48;

/// State for a single affine head: unscaled accumulator `s` and deferred scalar decay `c` for Q32.32,
/// or IEEE binary32 state matrix for F32.
#[derive(Debug)]
pub enum HeadState {
    Q32 { s: Vec<i64>, c: i64 },
    F32 { s: Vec<F32> },
}

/// State for a single affine head.
#[derive(Debug)]
pub struct AffineHead {
    pub state: HeadState,
}

impl AffineHead {
    /// New zeroed head state with `c = 1.0` in Q32.32.
    #[allow(dead_code, reason = "convenience constructor for Q32 head")]
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::with_carrier(len, Carrier::Q32)
    }

    /// New head state configured for the given carrier.
    #[must_use]
    pub fn with_carrier(len: usize, carrier: Carrier) -> Self {
        match carrier {
            Carrier::Q32 => AffineHead {
                state: HeadState::Q32 {
                    s: vec![0; len],
                    c: 1i64 << 32,
                },
            },
            Carrier::F32 => AffineHead {
                state: HeadState::F32 {
                    s: vec![F32::ZERO; len],
                },
            },
        }
    }
}

impl Clone for AffineHead {
    fn clone(&self) -> Self {
        AffineHead {
            state: match &self.state {
                HeadState::Q32 { s, c } => HeadState::Q32 {
                    s: s.clone(),
                    c: *c,
                },
                HeadState::F32 { s } => HeadState::F32 { s: s.clone() },
            },
        }
    }

    /// In-place copy reusing the state buffer: the speculative rollback copies both the state and the
    /// deferred decay `c`, so a renormalisation inside a rejected column is undone too.
    fn clone_from(&mut self, src: &Self) {
        match (&mut self.state, &src.state) {
            (HeadState::Q32 { s: dst_s, c: dst_c }, HeadState::Q32 { s: src_s, c: src_c }) => {
                dst_s.clone_from(src_s);
                *dst_c = *src_c;
            }
            (HeadState::F32 { s: dst_s }, HeadState::F32 { s: src_s }) => {
                dst_s.clone_from(src_s);
            }
            _ => {
                *self = src.clone();
            }
        }
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
/// State for the causal depthwise convolution: Q32.32 integer words or IEEE binary32 floats.
#[derive(Debug, Clone)]
pub enum ConvState {
    Q32(Vec<i64>),
    F32(Vec<F32>),
}

// Bitwise equality: `F32` has no `PartialEq` (IEEE `==` differs on `-0` and NaN), and state
// words are compared as the bit patterns they store.
impl PartialEq for ConvState {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Q32(a), Self::Q32(b)) => a == b,
            (Self::F32(a), Self::F32(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            _ => false,
        }
    }
}

#[allow(dead_code, reason = "typed conv state accessors")]
impl ConvState {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Q32(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_q32_slice(&self) -> &[i64] {
        match self {
            Self::Q32(v) => v,
            Self::F32(_) => panic!("expected Q32 conv state"),
        }
    }

    pub fn as_q32_mut(&mut self) -> &mut Vec<i64> {
        match self {
            Self::Q32(v) => v,
            Self::F32(_) => panic!("expected Q32 conv state"),
        }
    }

    pub fn as_f32_slice(&self) -> &[F32] {
        match self {
            Self::F32(v) => v,
            Self::Q32(_) => panic!("expected F32 conv state"),
        }
    }

    pub fn as_f32_mut(&mut self) -> &mut Vec<F32> {
        match self {
            Self::F32(v) => v,
            Self::Q32(_) => panic!("expected F32 conv state"),
        }
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
#[derive(Debug, Clone)]
pub(crate) struct StagedConv {
    pub(crate) xs_f32: Vec<F32>,
    pub(crate) window_f32: Vec<F32>,
    #[allow(dead_code, reason = "kept for debugging and dimension checks")]
    pub(crate) k: usize,
    pub(crate) ch: usize,
}

/// One f32 conv output buffer per [`AffineState`], written by layer `owner`'s conv and read by
/// that layer's heads before the next layer's conv overwrites it. `k == 0` marks it invalid;
/// otherwise `buf` holds `k` columns of layer `owner`. Capacity is kept at its high-water mark.
#[derive(Debug, Default)]
pub(crate) struct ConvF32Shared {
    pub(crate) buf: Vec<F32>,
    pub(crate) owner: usize,
    pub(crate) k: usize,
}

#[derive(Debug)]
pub struct AffineLayer {
    pub conv: ConvState,
    pub heads: Vec<Excl<AffineHead>>,
    pub slices: usize,
    /// Index of this layer in its [`AffineState`]; the tag `conv_shared` checks.
    pub(crate) slot: usize,
    /// The f32 conv output buffer shared by every layer of one [`AffineState`].
    pub(crate) conv_shared: std::sync::Arc<Excl<ConvF32Shared>>,
    pub(crate) staged_conv: Excl<Option<StagedConv>>,
    pub(crate) conv_w_packed: Excl<Vec<F32>>,
    pub(crate) conv_w_scalar: Excl<Vec<F32>>,
}

/// The `APEIRON_HEAD_SLICES` setting, read once (default 4); the process-flag boundary.
fn head_slices_setting() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("APEIRON_HEAD_SLICES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
}

/// Value-column slices per affine head: four 32-column slices of a 128-wide head keep a slice's
/// state at 32 KiB (an L1d) and give the pool `4·n_v_heads` tasks. `APEIRON_HEAD_SLICES`
/// overrides it (read once); a count that does not divide the head width falls back to 1.
fn head_slices(head_v: usize) -> usize {
    let n = head_slices_setting();
    if n > 0 && head_v.is_multiple_of(n) && (head_v / n).is_multiple_of(4) {
        n
    } else {
        1
    }
}

/// `APEIRON_EXACT_SCAN`: route the affine-head block scan through the chunked composition
/// ([`delta_head_chunk`]) instead of the per-step recurrence. Parsed once per process; unset, `0`
/// and every other value leave the per-step path in place, so the default is bit-identical.
pub(crate) fn exact_scan() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("APEIRON_EXACT_SCAN")
            .is_ok_and(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Carrier {
    Q32,
    F32,
}

/// Reads `APEIRON_F32_CARRIER` once via `OnceLock`. The `f32` carrier is the default
/// (it is the path the prefill receipts measure); `"0"` selects the Q32.32 carrier.
pub(crate) fn carrier_from_env() -> Carrier {
    static CARRIER: std::sync::OnceLock<Carrier> = std::sync::OnceLock::new();
    *CARRIER.get_or_init(|| match std::env::var("APEIRON_F32_CARRIER").as_deref() {
        Ok("0") => Carrier::Q32,
        _ => Carrier::F32,
    })
}

impl Clone for AffineLayer {
    fn clone(&self) -> Self {
        let mut heads = Vec::with_capacity(self.heads.len());
        let mut hi = 0usize;
        while hi < self.heads.len() {
            heads.push(Excl::new(self.heads[hi].lock().clone()));
            hi += 1;
        }
        AffineLayer {
            slices: self.slices,
            conv: self.conv.clone(),
            heads,
            slot: self.slot,
            conv_shared: std::sync::Arc::new(Excl::new(ConvF32Shared::default())),
            staged_conv: Excl::new(self.staged_conv.lock().clone()),
            conv_w_packed: Excl::new(self.conv_w_packed.lock().clone()),
            conv_w_scalar: Excl::new(self.conv_w_scalar.lock().clone()),
        }
    }

    /// Copy `src` into `self` reusing every buffer: no allocation once the shapes match (the
    /// speculative decoder's per-round rollback snapshot).
    fn clone_from(&mut self, src: &Self) {
        match (&mut self.conv, &src.conv) {
            (ConvState::Q32(dst), ConvState::Q32(src_conv)) => dst.clone_from(src_conv),
            (ConvState::F32(dst), ConvState::F32(src_conv)) => dst.clone_from(src_conv),
            _ => self.conv = src.conv.clone(),
        }
        if self.heads.len() != src.heads.len() {
            *self = src.clone();
            return;
        }
        let mut hi = 0usize;
        while hi < self.heads.len() {
            let dst = &mut self.heads[hi];
            let s = &src.heads[hi];
            hi += 1;
            let s = s.lock();
            dst.get_mut().clone_from(&s);
        }
        self.conv_shared.lock().k = 0;
        *self.staged_conv.lock() = None;
    }
}

/// The affine state of every affine layer, in schedule order.
///
/// Its size is `O(1)` in the context length: per layer, `(k − 1)·channels + n_v_heads·head_k·head_v`
/// Q32.32 words. For Qwen3.5-0.8B that is 18 layers × (3·6144 + 16·128·128) words ≈ 38 MiB.
#[derive(Debug)]
pub struct AffineState {
    layers: Vec<AffineLayer>,
}

impl Clone for AffineState {
    /// The clone owns its own conv buffer, shared by its layers only (a layer's own clone starts
    /// with a private one).
    fn clone(&self) -> Self {
        let mut layers = self.layers.clone();
        let shared = std::sync::Arc::new(Excl::new(ConvF32Shared::default()));
        for l in &mut layers {
            l.conv_shared = std::sync::Arc::clone(&shared);
        }
        AffineState { layers }
    }
}

impl AffineState {
    /// Zeroed state for `n_layers` affine layers of geometry `g`.
    ///
    /// # Complexity
    ///
    /// `O(n_layers · state words)` zeroed words, allocated once.
    #[must_use]
    pub fn new(g: &AffineGeom, n_layers: usize) -> AffineState {
        Self::with_slices_carrier(g, n_layers, head_slices(g.head_v), carrier_from_env())
    }

    /// Zeroed state whose heads are split into `slices` value-column slices (1 is the whole head).
    ///
    /// # Complexity
    ///
    /// As [`AffineState::new`].
    #[must_use]
    pub fn with_slices(g: &AffineGeom, n_layers: usize, slices: usize) -> AffineState {
        Self::with_slices_carrier(g, n_layers, slices, carrier_from_env())
    }

    /// Zeroed state configured with explicit slice count and carrier.
    #[must_use]
    pub(crate) fn with_slices_carrier(
        g: &AffineGeom,
        n_layers: usize,
        slices: usize,
        carrier: Carrier,
    ) -> AffineState {
        let channels = 2 * g.n_k_heads * g.head_k + g.n_v_heads * g.head_v;
        let conv_len = g.conv_kernel.saturating_sub(1) * channels;
        let slices = slices.max(1);
        let slice_len = g.head_k * (g.head_v / slices);
        let shared = std::sync::Arc::new(Excl::new(ConvF32Shared::default()));
        let mut layers = Vec::with_capacity(n_layers);
        let mut slot = 0usize;
        while slot < n_layers {
            let mut heads = Vec::with_capacity(g.n_v_heads * slices);
            let mut h = 0usize;
            while h < g.n_v_heads * slices {
                heads.push(Excl::new(AffineHead::with_carrier(slice_len, carrier)));
                h += 1;
            }
            layers.push(AffineLayer {
                conv: match carrier {
                    Carrier::Q32 => ConvState::Q32(vec![0; conv_len]),
                    Carrier::F32 => ConvState::F32(vec![F32::ZERO; conv_len]),
                },
                heads,
                slices,
                slot,
                conv_shared: std::sync::Arc::clone(&shared),
                staged_conv: Excl::new(None),
                conv_w_packed: Excl::new(Vec::new()),
                conv_w_scalar: Excl::new(Vec::new()),
            });
            slot += 1;
        }
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

    /// A state with no layers: `layer_mut` yields `None` for every slot. `O(1)`, no allocation.
    #[must_use]
    pub(crate) fn empty() -> AffineState {
        AffineState { layers: Vec::new() }
    }

    pub(crate) fn layer_mut(&mut self, slot: usize) -> Option<&mut AffineLayer> {
        self.layers.get_mut(slot)
    }
}

/// The per-column spare-capacity writers that [`conv_silu_cols_vecs`] shares with the pool: element `j`
/// points at the first unwritten word of output column `j`.
pub(crate) struct ColWriters(pub(crate) Vec<*mut MaybeUninit<i64>>);

/// `k` empty columns, each reserving `cap` words up front.
///
/// # Complexity
///
/// Time `O(k)`: one `Vec::with_capacity` and one push per column, the loop runs `k` times.
/// Space `O(k · cap)` words; no column grows past `cap` on its first `cap` pushes.
fn alloc_cols(k: usize, cap: usize) -> Vec<Vec<i64>> {
    let mut cols = Vec::with_capacity(k);
    let mut j = 0usize;
    while j < k {
        cols.push(Vec::with_capacity(cap));
        j += 1;
    }
    cols
}

/// The write pointers of [`ColWriters`] for `cols`, in column order.
///
/// # Complexity
///
/// Time `O(k)` for `k = cols.len()`: one spare-capacity pointer per column. Space `O(k)` pointers.
fn col_writers(cols: &mut [Vec<i64>]) -> ColWriters {
    let mut ptrs = Vec::with_capacity(cols.len());
    let mut j = 0usize;
    while j < cols.len() {
        ptrs.push(cols[j].spare_capacity_mut().as_mut_ptr());
        j += 1;
    }
    ColWriters(ptrs)
}

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

    /// Write a contiguous slice of channels starting at `offset` in output column `j`.
    ///
    /// # Safety
    ///
    /// `j` is a valid column index, `offset + slice.len()` is within that column's reserved
    /// capacity, and no other live writer targets the same channel range of column `j`.
    #[inline]
    unsafe fn write_slice(&self, j: usize, offset: usize, slice: &[i64]) {
        // SAFETY: the caller guarantees the index and disjointness.
        unsafe {
            let dst_ptr = self.0[j].add(offset).cast::<i64>();
            std::ptr::copy_nonoverlapping(slice.as_ptr(), dst_ptr, slice.len());
        }
    }
}

/// `k` input rows of `ch` words, copied into one row-major buffer so the convolution reads plain
/// slices. `ragged` is set when a row is not `ch` long (the geometry refusal); `data` then holds
/// only the rows before it.
pub(super) struct Flat {
    pub(super) data: Vec<i64>,
    pub(super) k: usize,
    pub(super) ch: usize,
    pub(super) ragged: bool,
}

/// Append `row` to `out`.
fn push_row(row: &[i64], out: &mut Vec<i64>) {
    out.extend_from_slice(row);
}

/// The marshalling boundary of the column convolutions: the rows of `xs` become the flat [`Flat`]
/// that every extracted body reads. Row 0 fixes `ch`; the copy stops at the first row of another
/// width (`ragged`).
///
/// # Complexity
///
/// Time `O(k · ch)` words copied; space `O(k · ch)` words. One loop over the `k` rows, one
/// `ch`-word copy each.
pub(super) fn flatten_vecs(xs: &[Vec<i64>]) -> Flat {
    let ch = if xs.is_empty() { 0 } else { xs[0].len() };
    let mut data = Vec::with_capacity(xs.len() * ch);
    let mut ragged = false;
    let mut i = 0usize;
    while i < xs.len() && !ragged {
        if xs[i].len() == ch {
            push_row(&xs[i], &mut data);
        } else {
            ragged = true;
        }
        i += 1;
    }
    Flat {
        data,
        k: xs.len(),
        ch,
        ragged,
    }
}

/// Owned copies of borrowed rows: the test entry points' adapter onto the `Vec` row form.
#[cfg(test)]
pub(super) fn owned_rows(xs: &[&[i64]]) -> Vec<Vec<i64>> {
    xs.iter().map(|r| r.to_vec()).collect()
}

/// [`flatten_vecs`] over borrowed rows.
#[cfg(test)]
pub(super) fn flatten_cols(xs: &[&[i64]]) -> Flat {
    flatten_vecs(&owned_rows(xs))
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
/// no channel-major staging buffer or transposing copy. One pool task per `per` channels. The input
/// rows are copied once into a flat buffer: `k · channels` words.
pub(crate) fn conv_silu_cols_vecs(
    layer: &mut AffineLayer,
    w: &[i64],
    kernel: usize,
    xs: &[Vec<i64>],
) -> Result<Vec<Vec<i64>>, StepError> {
    let flat = flatten_vecs(xs);
    conv_silu_cols_flat(layer, w, kernel, &flat)
}

/// [`conv_silu_cols_vecs`] over borrowed rows.
#[cfg(test)]
pub(crate) fn conv_silu_cols(
    layer: &mut AffineLayer,
    w: &[i64],
    kernel: usize,
    xs: &[&[i64]],
) -> Result<Vec<Vec<i64>>, StepError> {
    conv_silu_cols_vecs(layer, w, kernel, &owned_rows(xs))
}

fn conv_silu_cols_flat(
    layer: &mut AffineLayer,
    w: &[i64],
    kernel: usize,
    flat: &Flat,
) -> Result<Vec<Vec<i64>>, StepError> {
    match layer.conv {
        ConvState::Q32(_) => conv_silu_cols_q32(layer, w, kernel, flat),
        // Decode (k = 1) fuses conv+SiLU into the pooled heads dispatch. Prefill (k > 1) keeps
        // the per-column conv: the staged path holds the window per layer and cost 2x prefill
        // (f32 carrier).
        ConvState::F32(_) if flat.k > 1 => conv_silu_cols_f32(layer, w, kernel, flat),
        ConvState::F32(_) => conv_silu_cols_f32_stage(layer, w, kernel, flat),
    }
}

/// [`conv_silu_cols`] on a chosen carrier.
#[cfg(test)]
pub(crate) fn conv_silu_cols_carrier(
    layer: &mut AffineLayer,
    w: &[i64],
    kernel: usize,
    xs: &[&[i64]],
    carrier: Carrier,
) -> Result<Vec<Vec<i64>>, StepError> {
    let flat = flatten_cols(xs);
    match carrier {
        Carrier::Q32 => conv_silu_cols_q32(layer, w, kernel, &flat),
        Carrier::F32 => conv_silu_cols_f32(layer, w, kernel, &flat),
    }
}

/// The shared borrows of the staged 4-tap convolution dispatch; `run` is the single call to [`stage_task`].
struct StageCtx<'a> {
    packed: &'a [F32],
    scalar: &'a [F32],
    window: &'a [F32],
    xsf: &'a [F32],
    ch: usize,
    k: usize,
    per: usize,
    writers: &'a ColWriters,
    first: &'a crate::engine::chart::pool::FirstErr<usize, StepError>,
}

impl IndexTask for StageCtx<'_> {
    fn run(&self, t: usize) {
        stage_task(
            self.packed,
            self.scalar,
            self.window,
            self.xsf,
            self.ch,
            self.k,
            self.per,
            t,
            self.writers,
            self.first,
        );
    }
}

/// Pool task `t` of the staged convolution: channels `[t·per, (t+1)·per) ∩ [0, ch)` through
/// [`stage_range`]; a refusal is recorded under the range's first channel.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
fn stage_task(
    packed: &[F32],
    scalar: &[F32],
    window: &[F32],
    xsf: &[F32],
    ch: usize,
    k: usize,
    per: usize,
    t: usize,
    writers: &ColWriters,
    first: &crate::engine::chart::pool::FirstErr<usize, StepError>,
) {
    let c0 = t * per;
    let c1 = (c0 + per).min(ch);
    if c0 >= c1 {
        return;
    }
    let res = stage_range(packed, scalar, window, xsf, ch, k, c0, c1, writers);
    if let Err(e) = res {
        first.record(c0, e);
    }
}

/// Channels `[c0, c1)` of every column of the staged 4-tap convolution, read out to Q32 into `writers`.
///
/// # Complexity
///
/// `O(k · (c1 − c0))` f32 convolution outputs and readouts, `k · (c1 − c0)` scratch words.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
fn stage_range(
    packed: &[F32],
    scalar: &[F32],
    window: &[F32],
    xsf: &[F32],
    ch: usize,
    k: usize,
    c0: usize,
    c1: usize,
    writers: &ColWriters,
) -> Result<(), StepError> {
    let width = c1 - c0;
    let mut cols_f32 = vec![F32::ZERO; k * width];
    conv_silu_channel_range(packed, scalar, window, xsf, ch, k, c0, c1, &mut cols_f32)?;
    let mut err: Option<StepError> = None;
    let mut col_ix = 0;
    while col_ix < k && err.is_none() {
        let mut ch_ix = 0;
        while ch_ix < width && err.is_none() {
            match f32_carrier::f32_readout_to_q32(cols_f32[col_ix * width + ch_ix]) {
                // SAFETY: the caller's task owns channels `[c0, c1)` of every column, each
                // `by_col[j]` reserved `ch` words, and no other task writes `c0 + i`.
                Ok(v) => unsafe { writers.write(col_ix, c0 + ch_ix, v) },
                Err(e) => err = Some(e),
            }
            ch_ix += 1;
        }
        col_ix += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The decode (`k = 1`) f32 arm of [`conv_silu_cols_vecs`]: returns the complete column outputs and
/// stages the pre-call window and the converted inputs for the fused heads dispatch of
/// [`delta_heads_vecs_impl`].
///
/// The returned words are the same convolution the fused dispatch computes per channel range, so a
/// `k`-column call equals `k` one-column calls (`conv_silu_cols_k_columns_equal_k_single_column_calls`);
/// a placeholder would make that identity hold only vacuously.
///
/// # Complexity
///
/// `O(conv_kernel · channels)` f32 multiply-adds and `SiLU` evaluations,
/// `O((conv_kernel − 1)·channels)` window
/// moves, and `channels` Q32 readouts per column, pooled over channel ranges; the same per-column
/// work `conv_silu_channel_range` does for a head's channel range, taken once over
/// `[0, channels)`. Taps come from the layer's packed cache, so no Q32 tap conversion runs on this
/// path.
#[allow(
    clippy::too_many_lines,
    reason = "the complete-column store, the window advance and the stage are one decode step"
)]
fn conv_silu_cols_f32_stage(
    layer: &mut AffineLayer,
    w: &[i64],
    kernel: usize,
    flat: &Flat,
) -> Result<Vec<Vec<i64>>, StepError> {
    let (k, ch) = (flat.k, flat.ch);
    let xs: &[i64] = &flat.data;
    let cols = kernel
        .checked_sub(1)
        .ok_or(StepError::BadConfig(String::from("conv_kernel")))?;
    if k == 0 || flat.ragged || w.len() != kernel * ch || layer.conv.len() != cols * ch {
        return Err(StepError::BadConfig(String::from("conv geometry")));
    }

    layer.ensure_packed_w(w, ch, kernel);

    let mut xs_f32 = vec![F32::ZERO; k * ch];
    let mut i = 0usize;
    while i < k * ch {
        xs_f32[i] = f32_carrier::q32_to_f32(xs[i]);
        i += 1;
    }

    // This call's history: the window as it stood before the window advance below, plus the
    // converted inputs. The convolution and the fused heads both read this history.
    let window_f32: Vec<F32> = match &layer.conv {
        ConvState::F32(v) => v.clone(),
        ConvState::Q32(v) => {
            let mut w: Vec<F32> = Vec::with_capacity(v.len());
            let mut wi = 0usize;
            while wi < v.len() {
                w.push(f32_carrier::q32_to_f32(v[wi]));
                wi += 1;
            }
            w
        }
    };

    let by_col: Vec<Vec<i64>> = if kernel == 4 {
        // The complete columns, over every channel, with the packed taps. The per-column SiLU and
        // the f32 readout are the ones `conv_silu_cols_f32` applies, so the k-column calls of the
        // wave-4 arm and these one-column calls agree bit for bit. Channel ranges go to the pool:
        // serial, the `ch` readouts alone cost more than the decode step they exist for.
        let mut by_col = alloc_cols(k, ch);
        {
            let packed_guard = layer.conv_w_packed.lock();
            let scalar_guard = layer.conv_w_scalar.lock();
            let packed: &[F32] = &packed_guard;
            let scalar: &[F32] = &scalar_guard;
            let window: &[F32] = &window_f32;
            let xsf: &[F32] = &xs_f32;
            let writers = col_writers(&mut by_col);
            let first: crate::engine::chart::pool::FirstErr<usize, StepError> =
                crate::engine::chart::pool::FirstErr::new();
            let per = ch
                .div_ceil(crate::engine::par::row_chunk_budget(ch))
                .max(128);
            let per = per.div_ceil(8) * 8;
            let tasks = ch.div_ceil(per).max(1);
            let ctx = StageCtx {
                packed,
                scalar,
                window,
                xsf,
                ch,
                k,
                per,
                writers: &writers,
                first: &first,
            };
            crate::engine::chart::pool::run_global(tasks, &ctx);
            if let Some((_, e)) = first.take() {
                return Err(e);
            }
        }
        // SAFETY: every refusal-free dispatch wrote all `ch` words of every column.
        unsafe { set_written_lens(&mut by_col, ch) };

        match &mut layer.conv {
            ConvState::F32(window) => {
                if k >= cols {
                    window.copy_from_slice(&xs_f32[(k - cols) * ch..k * ch]);
                } else {
                    window.copy_within(k * ch..cols * ch, 0);
                    window[(cols - k) * ch..cols * ch].copy_from_slice(&xs_f32[..k * ch]);
                }
            }
            ConvState::Q32(window) => {
                let mut next = vec![0i64; cols * ch];
                let mut r = 0;
                while r < cols {
                    let mut c = 0;
                    while c < ch {
                        next[r * ch + c] = if k + r < cols {
                            window[(k + r) * ch + c]
                        } else {
                            xs[(k + r - cols) * ch + c]
                        };
                        c += 1;
                    }
                    r += 1;
                }
                *window = next;
            }
        }
        by_col
    } else {
        // No packed cache exists outside the 4-tap kernel: the wave-4 f32 kernel computes the
        // complete columns, advances the window and refreshes the f32 column cache.
        conv_silu_cols_f32(layer, w, kernel, flat)?
    };

    {
        let mut guard = layer.staged_conv.lock();
        *guard = Some(StagedConv {
            xs_f32,
            window_f32,
            k,
            ch,
        });
    }

    Ok(by_col)
}

/// History row `r` of the Q32 convolution: window row `r` for `r < cols`, else input row `r − cols`
/// of the flat `xs`, at channel `c`.
#[inline]
fn hist_q32(window: &[i64], xs: &[i64], cols: usize, ch: usize, r: usize, c: usize) -> i64 {
    if r < cols {
        return window[r * ch + c];
    }
    xs[(r - cols) * ch + c]
}

/// Channel `c` of all `k` columns of the Q32 convolution into `ys`; `taps` and `hist` are
/// `kernel`-word scratch. Returns the first refusal.
///
/// # Complexity
///
/// `O(k · kernel)` `i128` multiply-adds and `k` `SiLU`s, `O(kernel)` scratch words.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
fn conv_channel_q32(
    w: &[i64],
    window: &[i64],
    xs: &[i64],
    kernel: usize,
    cols: usize,
    ch: usize,
    k: usize,
    c: usize,
    taps: &mut [i64],
    hist: &mut [i64],
    ys: &mut [i64],
) -> Result<(), StepError> {
    let base = c * kernel;
    let mut tap = 0usize;
    while tap < kernel {
        taps[tap] = w[base + tap];
        hist[tap] = hist_q32(window, xs, cols, ch, tap, c);
        tap += 1;
    }
    let mut err: Option<StepError> = None;
    let mut j = 0;
    while j < k && err.is_none() {
        let mut acc: i128 = 0;
        let mut tap = 0usize;
        let mut acc_ok = true;
        while tap < kernel && acc_ok {
            let prod = i128::from(taps[tap]) * i128::from(hist[tap]);
            match acc.checked_add(prod) {
                Some(total) => acc = total,
                None => acc_ok = false,
            }
            tap += 1;
        }
        if acc_ok {
            // The shared `round_shift_32` adds 2^31 before shifting; guard that addition
            // here because this dot may be as large as i128::MAX.
            let v = match acc.checked_add(1_i128 << 31) {
                Some(_) => i64::try_from(round_shift_32(acc)).map_err(|_| StepError::Overflow),
                None => Err(StepError::Overflow),
            };
            let y = match v {
                Ok(v) => silu_fixed(v).map_err(StepError::Ffn),
                Err(e) => Err(e),
            };
            match y {
                Ok(y) => {
                    ys[j] = y;
                    // Slide the history window one column: after the shift `hist[t]`
                    // holds `row(j + 1 + t, c)`, the next column's window. The last column
                    // has no successor row, so it leaves `hist` as it is.
                    if j + 1 < k {
                        hist.copy_within(1..kernel, 0);
                        hist[kernel - 1] = hist_q32(window, xs, cols, ch, j + kernel, c);
                    }
                }
                Err(e) => err = Some(e),
            }
        } else {
            err = Some(StepError::Overflow);
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// Channels `[c0, c1)` of the Q32 convolution, written into `writers`. A refusal is recorded in
/// `first` under the channel and ends the range.
///
/// # Complexity
///
/// `O((c1 − c0) · k · kernel)` `i128` multiply-adds, `O(k + kernel)` scratch words.
#[allow(
    clippy::too_many_arguments,
    reason = "the geometry is scalars so every reference parameter is single-level"
)]
fn range_q32(
    w: &[i64],
    window: &[i64],
    xs: &[i64],
    kernel: usize,
    cols: usize,
    ch: usize,
    k: usize,
    c0: usize,
    c1: usize,
    writers: &ColWriters,
    first: &crate::engine::chart::pool::FirstErr<usize, StepError>,
) {
    // The channel's `kernel` taps and its sliding `kernel`-row history, in task-local
    // buffers: the raw output stores through `writers.write` may alias `w` and `xs` to LLVM,
    // so a value read from either is reloaded per column; a task-local buffer is not.
    let mut taps = vec![0i64; kernel];
    let mut hist = vec![0i64; kernel];
    let mut ys = vec![0i64; k];
    let mut stop = false;
    let mut c = c0;
    while c < c1 && !stop {
        match conv_channel_q32(
            w, window, xs, kernel, cols, ch, k, c, &mut taps, &mut hist, &mut ys,
        ) {
            Ok(()) => {
                let mut j = 0;
                while j < k {
                    // SAFETY: `by_col[j]` reserved `ch` words; task `t` writes slot `c`
                    // of column `j` exactly once and no other task writes slot `c`.
                    unsafe { writers.write(j, c, ys[j]) };
                    j += 1;
                }
            }
            Err(e) => {
                first.record(c, e);
                stop = true;
            }
        }
        c += 1;
    }
}

/// The shared borrows of the Q32 convolution dispatch; `run` is the single call to [`q32_task`].
struct Q32Ctx<'a> {
    w: &'a [i64],
    window: &'a [i64],
    xs: &'a [i64],
    kernel: usize,
    cols: usize,
    ch: usize,
    k: usize,
    per: usize,
    writers: &'a ColWriters,
    first: &'a crate::engine::chart::pool::FirstErr<usize, StepError>,
}

impl IndexTask for Q32Ctx<'_> {
    fn run(&self, t: usize) {
        q32_task(
            self.w,
            self.window,
            self.xs,
            [self.kernel, self.cols, self.ch, self.k, self.per],
            t,
            self.writers,
            self.first,
        );
    }
}

/// Pool task `t` of the Q32 convolution: channels `[t·per, (t+1)·per) ∩ [0, ch)` through
/// [`range_q32`]. `dims = [kernel, cols, ch, k, per]`.
fn q32_task(
    w: &[i64],
    window: &[i64],
    xs: &[i64],
    dims: [usize; 5],
    t: usize,
    writers: &ColWriters,
    first: &crate::engine::chart::pool::FirstErr<usize, StepError>,
) {
    let [kernel, cols, ch, k, per] = dims;
    let c0 = t * per;
    let c1 = (c0 + per).min(ch);
    range_q32(w, window, xs, kernel, cols, ch, k, c0, c1, writers, first);
}

#[allow(
    clippy::many_single_char_names,
    reason = "w (taps), x (input), c and t (channel and tap) are the convolution's symbols"
)]
fn conv_silu_cols_q32(
    layer: &mut AffineLayer,
    w: &[i64],
    kernel: usize,
    flat: &Flat,
) -> Result<Vec<Vec<i64>>, StepError> {
    let (k, ch) = (flat.k, flat.ch);
    let xs: &[i64] = &flat.data;
    let cols = kernel
        .checked_sub(1)
        .ok_or(StepError::BadConfig(String::from("conv_kernel")))?;
    if k == 0 || flat.ragged || w.len() != kernel * ch || layer.conv.len() != cols * ch {
        return Err(StepError::BadConfig(String::from("conv geometry")));
    }
    // History row `r`: window row `r` for `r < cols`, else column `r − cols`.
    let window = layer.conv.as_q32_slice();
    // Per-column outputs written in place: `by_col[j][c]` is channel `c` of column `j`. Every
    // column reserves `ch` words, so a task writes its channel's word of each column directly and
    // no channel-major staging buffer or serial transpose is needed.
    let mut by_col = alloc_cols(k, ch);
    let writers = col_writers(&mut by_col);
    let first: crate::engine::chart::pool::FirstErr<usize, StepError> =
        crate::engine::chart::pool::FirstErr::new();
    let per = ch
        .div_ceil(crate::engine::par::row_chunk_budget(ch))
        .max(64);
    let tasks = ch.div_ceil(per);
    let ctx = Q32Ctx {
        w,
        window,
        xs,
        kernel,
        cols,
        ch,
        k,
        per,
        writers: &writers,
        first: &first,
    };
    crate::engine::chart::pool::run_global(tasks, &ctx);
    if let Some((_, e)) = first.take() {
        return Err(e);
    }
    // Every index `0..ch` of every column was written exactly once above; this point is reached
    // only after a refusal-free dispatch, so the lengths can be published. The refusal path
    // returns before it, dropping length-0 vectors that never read an unwritten word.
    // SAFETY: the dispatch set no refusal, so all `ch` words are initialised.
    unsafe { set_written_lens(&mut by_col, ch) };
    // The new window: the history's last `cols` rows.
    let mut next = vec![0i64; cols * ch];
    let mut r = 0;
    while r < cols {
        let mut c = 0;
        while c < ch {
            next[r * ch + c] = hist_q32(window, xs, cols, ch, k + r, c);
            c += 1;
        }
        r += 1;
    }
    *layer.conv.as_q32_mut() = next;
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
    let mut err: Option<StepError> = None;
    let mut sum: i128 = 0;
    let mut pos = 0;
    while pos < x.len() && err.is_none() {
        let v = i128::from(x[pos]);
        match sum.checked_add(v * v) {
            Some(total) => sum = total,
            None => err = Some(StepError::Overflow),
        }
        pos += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let s = isqrt(sum);
    let norm = if sum - s * s > s { s + 1 } else { s };
    let den = norm.max(i128::from(eps));
    if den <= 0 {
        return Err(StepError::BadConfig(String::from("l2 norm eps")));
    }
    let half = den >> 1;
    let mut out = Vec::with_capacity(x.len());
    let mut pos = 0;
    while pos < x.len() && err.is_none() {
        let num = i128::from(x[pos]) << 32;
        let q = if num >= 0 {
            (num + half) / den
        } else {
            -((-num + half) / den)
        };
        match i64::try_from(q) {
            Ok(word) => out.push(word),
            Err(_) => err = Some(StepError::Overflow),
        }
        pos += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(out)
}

/// Exact Q32.32 division `round((num · 2³²) / den)` with round to nearest.
#[inline]
fn div_q32(num: i64, den: i64) -> Result<i64, StepError> {
    if den <= 0 {
        return Err(StepError::BadConfig(String::from(
            "affine decay factor c <= 0",
        )));
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
        return Err(StepError::BadConfig(String::from("affine head geometry")));
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
        return Err(StepError::BadConfig(String::from("affine head geometry")));
    }
    let mul = |a: i64, b: i64| mul_q32(a, b).ok_or(StepError::Overflow);

    // 1. Update deferred decay c
    *c = mul(*c, decay)?;
    if *c <= 0 {
        return Err(StepError::Overflow);
    }

    let mut err: Option<StepError> = None;

    // Fold c into S̃ when c drops below RENORM_THRESHOLD (2⁻⁸)
    if *c < RENORM_THRESHOLD {
        let c_val = *c;
        let mut i = 0;
        while i < s.len() && err.is_none() {
            match mul(s[i], c_val) {
                Ok(x) => s[i] = x,
                Err(e) => err = Some(e),
            }
            i += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        *c = 1i64 << 32;
    }
    let c_val = *c;

    // Check bounds on S̃
    let mut i = 0;
    while i < s.len() && err.is_none() {
        if s[i].abs() > STATE_BOUND {
            err = Some(StepError::Overflow);
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
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

    let mut j = 0;
    while j < dv && err.is_none() {
        match delta_head_row(c_val, beta, v[j], acc_m[j], acc_o[j], k_dot_q) {
            Ok((dj, oj)) => {
                d_prime.push(dj);
                o.push(oj);
            }
            Err(e) => err = Some(e),
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    // 3. Fused write pass: S̃ += k · d'ᵀ
    let mut i = 0;
    while i < dk && err.is_none() {
        let ki = k[i];
        if ki != 0 {
            let row = &mut s[i * dv..(i + 1) * dv];
            let mut j = 0;
            while j < dv && err.is_none() {
                match mul(ki, d_prime[j]) {
                    Ok(delta) => match row[j].checked_add(delta) {
                        Some(updated) if updated.abs() <= STATE_BOUND => row[j] = updated,
                        _ => err = Some(StepError::Overflow),
                    },
                    Err(e) => err = Some(e),
                }
                j += 1;
            }
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    Ok(o)
}

/// Value column `j` of the read pass of [`delta_head_step_ref`]: `(d'_j, o_j)` from the column's
/// accumulated reads `S̃ᵀk` and `S̃ᵀq̃`.
fn delta_head_row(
    c_val: i64,
    beta: i64,
    v_j: i64,
    acc_m: i128,
    acc_o: i128,
    k_dot_q: i64,
) -> Result<(i64, i64), StepError> {
    let m_tilde = i64::try_from(round_shift_32(acc_m)).map_err(|_| StepError::Overflow)?;
    let m = mul_q32(c_val, m_tilde).ok_or(StepError::Overflow)?;
    let dj =
        mul_q32(beta, v_j.checked_sub(m).ok_or(StepError::Overflow)?).ok_or(StepError::Overflow)?;
    let d_prime = div_q32(dj, c_val)?;

    let o_tilde = i64::try_from(round_shift_32(acc_o)).map_err(|_| StepError::Overflow)?;
    let term1 = mul_q32(c_val, o_tilde).ok_or(StepError::Overflow)?;
    let term2 = mul_q32(dj, k_dot_q).ok_or(StepError::Overflow)?;
    Ok((
        d_prime,
        term1.checked_add(term2).ok_or(StepError::Overflow)?,
    ))
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
    let mut err: Option<StepError> = None;
    let mut i = 0;
    while i < x.len() && err.is_none() {
        let xw = i128::from(x[i]);
        let row = &s0[i * dv..(i + 1) * dv];
        let mut j = 0;
        while j < dv && err.is_none() {
            match acc[j].checked_add(i128::from(row[j]) * xw) {
                Some(total) => acc[j] = total,
                None => err = Some(StepError::Overflow),
            }
            j += 1;
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut out = Vec::with_capacity(acc.len());
    let mut j = 0usize;
    while j < acc.len() {
        out.push(chunk_narrow(acc[j], 32)?);
        j += 1;
    }
    Ok(out)
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
    reason = "dk, dv, C, G, D, u, o are the recurrence's symbols; geometry, entry state and the \
              per-step gates are the kernel's input; the chunked stage-1 routing is \
              delta_heads_cols"
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
        return Err(StepError::BadConfig(String::from("affine chunk length")));
    }
    if dk == 0
        || dv == 0
        || s.len() != dk * dv
        || q.len() != chunk * dk
        || k.len() != chunk * dk
        || v.len() != chunk * dv
    {
        return Err(StepError::BadConfig(String::from("affine head geometry")));
    }
    if *c <= 0 {
        return Err(StepError::Overflow);
    }
    let mut err: Option<StepError> = None;
    let mut i = 0;
    while i < s.len() && err.is_none() {
        if s[i].abs() > STATE_BOUND {
            err = Some(StepError::Overflow);
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    // A one-step chunk has no composition to amortise: the step oracle is the chunk.
    if chunk == 1 {
        return delta_head_step_ref(s, c, q, k, v, decay[0], beta[0]);
    }

    // Entry state decoded to the chunk precision. `|S̃| ≤ STATE_BOUND` and `c ≤ 1`, so the product
    // fits i128 and the rounded word fits i64.
    let mut s0: Vec<i64> = Vec::with_capacity(s.len());
    let mut i = 0usize;
    while i < s.len() && err.is_none() {
        match mul_q32_prec(s[i], *c) {
            Ok(x) => s0.push(x),
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    // Cumulative decays `G_t` at the chunk precision.
    let one = 1i64 << CHUNK_PREC;
    let mut g = Vec::with_capacity(chunk);
    let mut g_prev = one;
    let mut i = 0;
    while i < chunk && err.is_none() {
        match mul_prec_q32(g_prev, decay[i]) {
            Ok(x) => {
                g_prev = x;
                g.push(g_prev);
            }
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    let mut u = vec![0i64; chunk * dv];
    let mut out = vec![0i64; chunk * dv];
    let mut t = 0;
    while t < chunk && err.is_none() {
        if let Err(e) = chunk_step(t, k, q, v, beta, decay, &s0, &g, dk, dv, &mut u, &mut out) {
            err = Some(e);
        }
        t += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    // Composed exit state `S_C = G_{C−1}S_0 + Σ_j D(C−1←j) k_j u_jᵀ`.
    let mut acc: Vec<i128> = Vec::with_capacity(s0.len());
    let mut i = 0usize;
    while i < s0.len() && err.is_none() {
        match mul_prec(g[chunk - 1], s0[i]) {
            Ok(x) => acc.push(i128::from(x)),
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut d = one;
    let mut jj = chunk;
    while jj > 0 && err.is_none() {
        let j = jj - 1;
        let (uj, kj) = (&u[j * dv..(j + 1) * dv], &k[j * dk..(j + 1) * dk]);
        match exit_accumulate(&mut acc, kj, uj, d, dv) {
            Ok(()) if j > 0 => match mul_prec_q32(d, decay[j]) {
                Ok(x) => d = x,
                Err(e) => err = Some(e),
            },
            Ok(()) => {}
            Err(e) => err = Some(e),
        }
        jj -= 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    // Narrow once to Q32.32 and check the width before the state is written.
    let mut next_s = Vec::with_capacity(dk * dv);
    let mut i = 0;
    while i < acc.len() && err.is_none() {
        match chunk_narrow(acc[i], CHUNK_PREC - 32) {
            Ok(w) if w.abs() <= STATE_BOUND => next_s.push(w),
            Ok(_) => err = Some(StepError::Overflow),
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut next_o = Vec::with_capacity(chunk * dv);
    let mut i = 0;
    while i < out.len() && err.is_none() {
        match chunk_narrow(i128::from(out[i]), CHUNK_PREC - 32) {
            Ok(w) => next_o.push(w),
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    s.copy_from_slice(&next_s);
    *c = 1i64 << 32;
    Ok(next_o)
}

/// Step `t` of [`delta_head_chunk`]'s forward substitution and readout: writes `u[t]` and
/// `out[t]` from the entry state `s0` and the cumulative decays `g`.
///
/// # Complexity
///
/// `O(t·(dk + dv) + dk·dv)` checked `i128` multiply-adds, `O(dv)` scratch words: the two state
/// reads are `dk·dv` each, and each of the `t` (resp. `t + 1`) substitution (resp. readout)
/// terms is one `dk` dot and one `dv` row.
#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    reason = "dk, dv, G, u, o are the recurrence's symbols; the step reads the chunk's inputs and \
              the composed prefix"
)]
fn chunk_step(
    t: usize,
    k: &[i64],
    q: &[i64],
    v: &[i64],
    beta: &[i64],
    decay: &[i64],
    s0: &[i64],
    g: &[i64],
    dk: usize,
    dv: usize,
    u: &mut [i64],
    out: &mut [i64],
) -> Result<(), StepError> {
    let one = 1i64 << CHUNK_PREC;
    let kt = &k[t * dk..(t + 1) * dk];
    // The read at step `t` sees the decayed state `α_t Q_{t−1}`: both the entry term and the
    // substitution coefficient carry `α_t β_t`, the injection term only `β_t`.
    let alpha_beta = mul_q32_prec(beta[t], decay[t])?;
    // Right-hand side `β_t v_t − α_t β_t G_{t−1}(S_0ᵀk_t)`.
    let s0k = state_read_prec(s0, dv, kt)?;
    let beta_g = mul_prec(alpha_beta, if t == 0 { one } else { g[t - 1] })?;
    let mut err: Option<StepError> = None;
    let mut ut = Vec::with_capacity(dv);
    let mut jv = 0;
    while jv < dv && err.is_none() {
        match rhs_word(beta[t], v[t * dv + jv], beta_g, s0k[jv]) {
            Ok(x) => ut.push(x),
            Err(e) => err = Some(e),
        }
        jv += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    // Forward substitution, `D(t−1←t−1) = 1` walking `j` downward.
    let mut d = one;
    let mut jj = t;
    while jj > 0 && err.is_none() {
        let j = jj - 1;
        let uj = &u[j * dv..(j + 1) * dv];
        match subst_term(&mut ut, uj, &k[j * dk..(j + 1) * dk], kt, alpha_beta, d) {
            Ok(()) => match mul_prec_q32(d, decay[j]) {
                Ok(x) => d = x,
                Err(e) => err = Some(e),
            },
            Err(e) => err = Some(e),
        }
        jj -= 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    u[t * dv..(t + 1) * dv].copy_from_slice(&ut);

    // Readout `o_t = G_t(S_0ᵀq_t) + Σ_{j≤t} D(t←j)(k_jᵀq_t) u_j`.
    let qt = &q[t * dk..(t + 1) * dk];
    let s0q = state_read_prec(s0, dv, qt)?;
    let ot = &mut out[t * dv..(t + 1) * dv];
    let mut i = 0;
    while i < dv && err.is_none() {
        match mul_prec(g[t], s0q[i]) {
            Ok(x) => ot[i] = x,
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    let mut d = one;
    let mut jj = t + 1;
    while jj > 0 && err.is_none() {
        let j = jj - 1;
        let uj = &u[j * dv..(j + 1) * dv];
        match readout_term(ot, uj, &k[j * dk..(j + 1) * dk], qt, d) {
            Ok(()) if j > 0 => match mul_prec_q32(d, decay[j]) {
                Ok(x) => d = x,
                Err(e) => err = Some(e),
            },
            Ok(()) => {}
            Err(e) => err = Some(e),
        }
        jj -= 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// `β_t v_j − β_g s0k_j` at the chunk precision: one word of the right-hand side.
fn rhs_word(beta_t: i64, v_j: i64, beta_g: i64, s0k_j: i64) -> Result<i64, StepError> {
    let vt = mul_q32_prec(beta_t, v_j)?;
    let sub = mul_prec(beta_g, s0k_j)?;
    vt.checked_sub(sub).ok_or(StepError::Overflow)
}

/// `ut −= D·α_tβ_t (k_jᵀk_t) u_j`: one forward-substitution term.
///
/// # Complexity
///
/// `O(dk + dv)` checked `i128` multiply-adds, `O(1)` scratch.
fn subst_term(
    ut: &mut [i64],
    uj: &[i64],
    kj: &[i64],
    kt: &[i64],
    alpha_beta: i64,
    d: i64,
) -> Result<(), StepError> {
    let dot = dot_q32_prec(kj, kt)?;
    let coef = mul_prec(mul_prec(alpha_beta, d)?, dot)?;
    row_axpy_prec(ut, coef, uj, true)
}

/// `ot += D (k_jᵀq_t) u_j`: one readout term.
///
/// # Complexity
///
/// `O(dk + dv)` checked `i128` multiply-adds, `O(1)` scratch.
fn readout_term(
    ot: &mut [i64],
    uj: &[i64],
    kj: &[i64],
    qt: &[i64],
    d: i64,
) -> Result<(), StepError> {
    let dot = dot_q32_prec(kj, qt)?;
    let coef = mul_prec(d, dot)?;
    row_axpy_prec(ot, coef, uj, false)
}

/// `acc[i] ∓= mul_prec(coef, row[i])` for `i < acc.len().min(row.len())`, checked; a refusal
/// leaves the earlier words written.
///
/// # Complexity
///
/// `O(len)` `mul_prec` and checked `i64` add or subtract, `O(1)` scratch.
fn row_axpy_prec(acc: &mut [i64], coef: i64, row: &[i64], subtract: bool) -> Result<(), StepError> {
    let n = acc.len().min(row.len());
    let mut err: Option<StepError> = None;
    let mut i = 0;
    while i < n && err.is_none() {
        match mul_prec(coef, row[i]) {
            Ok(m) => {
                let next = if subtract {
                    acc[i].checked_sub(m)
                } else {
                    acc[i].checked_add(m)
                };
                match next {
                    Some(x) => acc[i] = x,
                    None => err = Some(StepError::Overflow),
                }
            }
            Err(e) => err = Some(e),
        }
        i += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// `acc[i·dv + jv] += D(C−1←j)·ki·uj[jv]` over the outer product `kj ⊗ uj`, checked at the chunk
/// precision.
///
/// # Complexity
///
/// `O(dk·dv)` `mul_prec_q32`, `mul_prec` and checked `i128` adds, `O(1)` scratch.
fn exit_accumulate(
    acc: &mut [i128],
    kj: &[i64],
    uj: &[i64],
    d: i64,
    dv: usize,
) -> Result<(), StepError> {
    let mut err: Option<StepError> = None;
    let mut i = 0;
    while i < kj.len() && err.is_none() {
        let mut jv = 0;
        while jv < uj.len() && err.is_none() {
            match mul_prec_q32(kj[i], uj[jv]) {
                Ok(p) => match mul_prec(d, p) {
                    Ok(term) => match acc[i * dv + jv].checked_add(i128::from(term)) {
                        Some(x) => acc[i * dv + jv] = x,
                        None => err = Some(StepError::Overflow),
                    },
                    Err(e) => err = Some(e),
                },
                Err(e) => err = Some(e),
            }
            jv += 1;
        }
        i += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
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
    /// The predicate `cmpgt(res, B) | cmpgt(-B, res)` is exactly `res > B || res < -B`, i.e.
    /// `|res| > B` on the signed i64 lanes, matching `delta_head_step_ref`'s
    /// `updated.abs() > STATE_BOUND`. `-B` (not `-B - 1`) makes the boundary word `-(B + 1)` refuse;
    /// `i64::MIN` also refuses (`|i64::MIN|` is not representable, and the second compare catches it).
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
        let below = _mm256_set1_epi64x(-STATE_BOUND);
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
        return Err(StepError::BadConfig(String::from("affine head geometry")));
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

    // No pre-read state scan: `|s[i]| ≤ STATE_BOUND` holds before every column step.
    // (1) `AffineHead::new` zeroes every word and the only writers of `s` are `delta_head_step*`,
    // `delta_head_chunk` and `clone_from` (a copy of already-checked words); (2) every word the
    // write pass writes is checked at write time by its
    // `|res| > STATE_BOUND` predicate, and a `k[i] == 0` row is skipped, keeping its checked value;
    // (3) the renormalisation fold `*x = mul(*x, c_val)` runs only when `c_val < RENORM_THRESHOLD`,
    // where `|round(x·c_val / 2³²)| ≤ |x|`, so the fold cannot leave the bound and its own `Err` is
    // unchanged; (4) the read pass does not modify `s`; (5) `delta_head_chunk` narrows its exit
    // state through the same `|w| > STATE_BOUND` refusal before writing. `delta_head_step_ref` keeps
    // its scan: it is the oracle. A state left out of bound by a refusal is unspecified by the
    // contract and is not re-entered.

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
#[cfg(test)]
pub(crate) type AffineCol<'a> = (&'a [i64], &'a [i64], &'a [i64], &'a [i64]);

/// Every value head of one affine layer for one column: [`delta_heads_cols`] with `k = 1` (the
/// microbenchmark's entry point).
///
/// # Complexity
///
/// As [`delta_heads_cols_vecs`] with one column.
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
    out.pop()
        .ok_or(StepError::BadConfig(String::from("affine column")))
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
/// When `APEIRON_EXACT_SCAN` is on, a multi-column, snapshot-free block runs stage 1 through the
/// chunked composition [`delta_head_chunk`] in `<= CHUNK_MAX`-column groups; the readouts and the
/// real exit state agree with the per-step path within the epic's uniform bound, and the state is
/// stored as the real state at `c = 1`. The default (flag off, one column, or a snapshot) is the
/// per-step path bit for bit.
///
/// # Complexity
///
/// `O(k · n_v_heads · head_k · head_v)` for the updates, plus `O(k · channels)` for the norms, over
/// `n_k_heads`, `n_v_heads · slices` and `n_v_heads` pool tasks when `parallel`; the four input
/// lists are copied once into the flat buffers of [`flatten_affine_vecs`]:
/// `k · (2·key_dim + 2·val_dim + 2·n_v_heads)` words.
#[allow(
    clippy::too_many_arguments,
    reason = "the layer state, geometry, parameters, the four column inputs and snapshot are the \
              step's input"
)]
pub(crate) fn delta_heads_cols_vecs(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    conv_cols: &[Vec<i64>],
    z_cols: &[Vec<i64>],
    b_cols: &[Vec<i64>],
    a_cols: &[Vec<i64>],
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
) -> Result<Vec<Vec<i64>>, StepError> {
    let carrier = carrier_from_env();
    delta_heads_vecs_carrier(
        layer, g, p, conv_cols, z_cols, b_cols, a_cols, eps, parallel, snapshot, carrier,
    )
}

/// Whether the block scan takes the chunked composition: the Q32 carrier, `APEIRON_EXACT_SCAN`, more
/// than one column and no rollback snapshot. Scalar arguments keep the branch out of the caller,
/// whose reference parameters Aeneas cannot join across a branch.
fn chunked_route(carrier: Carrier, n_cols: usize, no_snapshot: bool) -> bool {
    if carrier == Carrier::Q32 && exact_scan() {
        return n_cols > 1 && no_snapshot;
    }
    false
}

#[allow(
    clippy::too_many_arguments,
    reason = "the layer state, geometry, parameters, the four column inputs, snapshot and carrier \
              are the step's input"
)]
fn delta_heads_vecs_carrier(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    conv_cols: &[Vec<i64>],
    z_cols: &[Vec<i64>],
    b_cols: &[Vec<i64>],
    a_cols: &[Vec<i64>],
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
    carrier: Carrier,
) -> Result<Vec<Vec<i64>>, StepError> {
    // A one-column block has no composition to amortise, and the speculative rollback copy is
    // materialised per step, so both keep the per-step path whatever the flag says.
    let chunked = chunked_route(carrier, conv_cols.len(), snapshot.is_none());
    delta_heads_vecs_impl(
        layer, g, p, conv_cols, z_cols, b_cols, a_cols, eps, parallel, snapshot, chunked, carrier,
    )
    .map(|(out, _raw)| out)
}

/// The four per-column inputs of borrowed [`AffineCol`] tuples as owned row lists.
#[cfg(test)]
fn split_cols(cols: &[AffineCol<'_>]) -> [Vec<Vec<i64>>; 4] {
    [
        cols.iter().map(|c| c.0.to_vec()).collect(),
        cols.iter().map(|c| c.1.to_vec()).collect(),
        cols.iter().map(|c| c.2.to_vec()).collect(),
        cols.iter().map(|c| c.3.to_vec()).collect(),
    ]
}

/// [`delta_heads_cols_vecs`] over borrowed columns.
#[cfg(test)]
#[allow(
    clippy::too_many_arguments,
    reason = "the layer state, geometry, parameters, columns and snapshot are the step's input"
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
    let [c, z, b, a] = split_cols(cols);
    delta_heads_cols_vecs(layer, g, p, &c, &z, &b, &a, eps, parallel, snapshot)
}

/// [`delta_heads_cols_vecs`] on a chosen carrier, over borrowed columns.
#[cfg(test)]
#[allow(
    clippy::too_many_arguments,
    reason = "the layer state, geometry, parameters, columns, snapshot and carrier are the step's input"
)]
pub(crate) fn delta_heads_cols_carrier(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    cols: &[AffineCol<'_>],
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
    carrier: Carrier,
) -> Result<Vec<Vec<i64>>, StepError> {
    let [c, z, b, a] = split_cols(cols);
    delta_heads_vecs_carrier(
        layer, g, p, &c, &z, &b, &a, eps, parallel, snapshot, carrier,
    )
}

fn attn_heads_f32<E, K>(
    n_head: usize,
    hd: usize,
    out: &mut [F32],
    parallel: bool,
    head: &K,
) -> Option<(usize, E)>
where
    K: HeadTask<F32, E>,
    E: Send,
{
    if !parallel {
        let mut h = 0usize;
        while h < n_head {
            let lo = h * hd;
            if let Err(e) = head.run(h, &mut out[lo..lo + hd]) {
                return Some((h, e));
            }
            h += 1;
        }
        return None;
    }
    let first: crate::engine::chart::pool::FirstErr<usize, E> =
        crate::engine::chart::pool::FirstErr::new();
    let ctx = HeadsF32Ctx {
        head,
        first: &first,
    };
    crate::engine::chart::pool::chunks_mut_global(out, hd, &ctx);
    first.take()
}

/// The shared borrows of the per-head f32 dispatch: the head body and the refusal cell.
struct HeadsF32Ctx<'a, E, K> {
    head: &'a K,
    first: &'a crate::engine::chart::pool::FirstErr<usize, E>,
}

impl<E, K> ChunkTask<F32> for HeadsF32Ctx<'_, E, K>
where
    K: HeadTask<F32, E>,
    E: Send,
{
    fn run(&self, c: usize, _start: usize, dst: &mut [F32]) {
        heads_f32_task(self.head, self.first, c, dst);
    }
}

/// Head `c`: run `head` on its piece and record a refusal under `c`.
fn heads_f32_task<E, K>(
    head: &K,
    first: &crate::engine::chart::pool::FirstErr<usize, E>,
    c: usize,
    dst: &mut [F32],
) where
    K: HeadTask<F32, E>,
    E: Send,
{
    if let Err(e) = head.run(c, dst) {
        first.record(c, e);
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "channel-range causal conv+silu; taps, history, and channels"
)]
fn conv_silu_channel_range(
    packed_w: &[F32],
    scalar_w: &[F32],
    window: &[F32],
    xs: &[F32],
    ch: usize,
    k: usize,
    c0: usize,
    c1: usize,
    out: &mut [F32],
) -> Result<(), StepError> {
    let cols = 3usize;
    let width = c1 - c0;
    let is_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");

    let mut c = c0;
    if is_avx2 && packed_w.len() >= 4 * ch {
        while c + 8 <= c1 {
            let c_offset = c - c0;
            let blk = c / 8;
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
                let v_tap0 = _mm256_loadu_ps(packed_w.as_ptr().cast::<f32>().add(blk * 32));
                let v_tap1 = _mm256_loadu_ps(packed_w.as_ptr().cast::<f32>().add(blk * 32 + 8));
                let v_tap2 = _mm256_loadu_ps(packed_w.as_ptr().cast::<f32>().add(blk * 32 + 16));
                let v_tap3 = _mm256_loadu_ps(packed_w.as_ptr().cast::<f32>().add(blk * 32 + 24));

                for j in 0..k {
                    let p0 = if j < cols {
                        window.as_ptr().cast::<f32>().add(j * ch + c)
                    } else {
                        xs.as_ptr().cast::<f32>().add((j - cols) * ch + c)
                    };
                    let p1 = if j + 1 < cols {
                        window.as_ptr().cast::<f32>().add((j + 1) * ch + c)
                    } else {
                        xs.as_ptr().cast::<f32>().add((j + 1 - cols) * ch + c)
                    };
                    let p2 = if j + 2 < cols {
                        window.as_ptr().cast::<f32>().add((j + 2) * ch + c)
                    } else {
                        xs.as_ptr().cast::<f32>().add((j + 2 - cols) * ch + c)
                    };
                    let p3 = if j + 3 < cols {
                        window.as_ptr().cast::<f32>().add((j + 3) * ch + c)
                    } else {
                        xs.as_ptr().cast::<f32>().add((j + 3 - cols) * ch + c)
                    };

                    let v_h0 = _mm256_loadu_ps(p0);
                    let v_h1 = _mm256_loadu_ps(p1);
                    let v_h2 = _mm256_loadu_ps(p2);
                    let v_h3 = _mm256_loadu_ps(p3);

                    let vy = f32_carrier::conv_silu_tap4_avx2(
                        v_tap0, v_tap1, v_tap2, v_tap3, v_h0, v_h1, v_h2, v_h3,
                    );
                    _mm256_storeu_ps(out.as_mut_ptr().cast::<f32>().add(j * width + c_offset), vy);
                }
            }
            c += 8;
        }
    }

    let mut err: Option<StepError> = None;
    while c < c1 && err.is_none() {
        let c_offset = c - c0;
        let base = c * 4;
        let taps = [
            scalar_w[base],
            scalar_w[base + 1],
            scalar_w[base + 2],
            scalar_w[base + 3],
        ];
        let mut j = 0;
        while j < k && err.is_none() {
            let h0 = if j < cols {
                window[j * ch + c]
            } else {
                xs[(j - cols) * ch + c]
            };
            let h1 = if j + 1 < cols {
                window[(j + 1) * ch + c]
            } else {
                xs[(j + 1 - cols) * ch + c]
            };
            let h2 = if j + 2 < cols {
                window[(j + 2) * ch + c]
            } else {
                xs[(j + 2 - cols) * ch + c]
            };
            let h3 = if j + 3 < cols {
                window[(j + 3) * ch + c]
            } else {
                xs[(j + 3 - cols) * ch + c]
            };

            let y = f32_carrier::conv_silu_tap4_scalar(taps, [h0, h1, h2, h3]);
            if y.is_finite() {
                out[j * width + c_offset] = y;
            } else {
                err = Some(StepError::Overflow);
            }
            j += 1;
        }
        c += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }

    Ok(())
}

/// The affine heads' per-column inputs copied into one row-major buffer per input, so every
/// extracted body reads plain slices: column `c` of `conv` is `conv[c·cdim ..][..cdim]`, of `z`
/// `z[c·val_dim ..]`, of `b` and `a` `[c·n_v_heads ..]`. `bad` is set when a column has the wrong
/// width (the `"affine layer inputs"` refusal); the buffers then hold only the columns before it.
struct FlatAffine {
    conv: Vec<i64>,
    z: Vec<i64>,
    b: Vec<i64>,
    a: Vec<i64>,
    k: usize,
    bad: bool,
}

/// The marshalling boundary of the affine heads: the four per-column row lists become one owned
/// [`FlatAffine`] with `k = conv_cols.len()` columns. A column of the wrong width, or an input
/// list that does not hold `k` columns, sets `bad`.
///
/// # Complexity
///
/// Time `O(k · (cdim + val_dim + 2·n_v_heads))` words copied, the same in space: one loop over the
/// `k` columns, four row copies each; no pointer list is built.
fn flatten_affine_vecs(
    conv_cols: &[Vec<i64>],
    z_cols: &[Vec<i64>],
    b_cols: &[Vec<i64>],
    a_cols: &[Vec<i64>],
    cdim: usize,
    vdim: usize,
    nv: usize,
) -> FlatAffine {
    let k = conv_cols.len();
    let mut out = FlatAffine {
        conv: Vec::with_capacity(k * cdim),
        z: Vec::with_capacity(k * vdim),
        b: Vec::with_capacity(k * nv),
        a: Vec::with_capacity(k * nv),
        k,
        bad: z_cols.len() != k || b_cols.len() != k || a_cols.len() != k,
    };
    let mut i = 0usize;
    while i < k && !out.bad {
        if conv_cols[i].len() != cdim
            || z_cols[i].len() != vdim
            || b_cols[i].len() != nv
            || a_cols[i].len() != nv
        {
            out.bad = true;
        } else {
            push_row(&conv_cols[i], &mut out.conv);
            push_row(&z_cols[i], &mut out.z);
            push_row(&b_cols[i], &mut out.b);
            push_row(&a_cols[i], &mut out.a);
        }
        i += 1;
    }
    out
}

/// Append the gate input of columns `i..` of `cols` to `out`.
#[cfg(test)]
fn flatten_z_rows(cols: &[AffineCol<'_>], i: usize, out: &mut Vec<i64>) {
    if i >= cols.len() {
        return;
    }
    push_row(cols[i].1, out);
    flatten_z_rows(cols, i + 1, out);
}

/// Scalar geometry of the stage 2 task grid.
struct Stage2Geom {
    val_dim: usize,
    head_v: usize,
    slices: usize,
    k: usize,
    num_b: usize,
    per_col: usize,
}

/// Head `h`, columns `[c0, c1)` of stage 2 in f32, with caller-provided scratch `o_slice`
/// (`head_v` f32) and `dcol_slice` (`head_v` words). The refusal carries its `(head, column)`.
///
/// # Complexity
///
/// `O((c1 − c0) · head_v)` f32 operations and words written, no allocation.
#[allow(
    clippy::too_many_arguments,
    reason = "scalars and plain slices keep every reference parameter single-level"
)]
fn stage2_cols(
    raw_f32: &[F32],
    z: &[i64],
    ssm_norm_f: &[F32],
    eps_f: F32,
    sg: &Stage2Geom,
    h: usize,
    c0: usize,
    c1: usize,
    o_slice: &mut [F32],
    dcol_slice: &mut [i64],
    writers: &ColWriters,
) -> Result<(), (usize, usize, StepError)> {
    let head_v = sg.head_v;
    let slices = sg.slices;
    let k = sg.k;
    let w = head_v / slices;
    let mut err: Option<(usize, usize, StepError)> = None;
    let mut c = c0;
    while c < c1 && err.is_none() {
        let mut sl = 0;
        while sl < slices {
            let at = (h * slices + sl) * k * w + c * w;
            o_slice[sl * w..(sl + 1) * w].copy_from_slice(&raw_f32[at..at + w]);
            sl += 1;
        }
        let zo = c * sg.val_dim + h * head_v;
        match f32_carrier::stage2_gated_norm_f32(
            o_slice,
            &z[zo..zo + head_v],
            ssm_norm_f,
            eps_f,
            dcol_slice,
        ) {
            // SAFETY: task `t` owns head `h` (channels `[h·head_v, (h+1)·head_v)`) and column
            // range `[c0, c1)` containing column `c`. No other task writes to channel
            // `h·head_v + i` of column `c`, and `cols_out[c]` has capacity `val_dim`.
            Ok(()) => unsafe { writers.write_slice(c, h * head_v, dcol_slice) },
            Err(e) => err = Some((h, c, e)),
        }
        c += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// The shared borrows of the stage 2 dispatch; `run` is the single call to [`stage2_pool_task`].
struct Stage2Ctx<'a> {
    raw_f32: &'a [F32],
    z: &'a [i64],
    ssm_norm_f: &'a [F32],
    eps_f: F32,
    sg: &'a Stage2Geom,
    writers: &'a ColWriters,
    first: &'a crate::engine::chart::pool::FirstErr<u128, StepError>,
}

impl IndexTask for Stage2Ctx<'_> {
    fn run(&self, t: usize) {
        stage2_pool_task(
            self.raw_f32,
            self.z,
            self.ssm_norm_f,
            self.eps_f,
            self.sg,
            t,
            self.writers,
            self.first,
        );
    }
}

/// [`stage2_task`] on the pool: a refusal is recorded under its `(head, column)` key.
#[allow(
    clippy::too_many_arguments,
    reason = "the stage 2 borrows and the refusal cell, each single-level"
)]
fn stage2_pool_task(
    raw_f32: &[F32],
    z: &[i64],
    ssm_norm_f: &[F32],
    eps_f: F32,
    sg: &Stage2Geom,
    t: usize,
    writers: &ColWriters,
    first: &crate::engine::chart::pool::FirstErr<u128, StepError>,
) {
    if let Err(e) = stage2_task(raw_f32, z, ssm_norm_f, eps_f, sg, t, writers) {
        first.record(crate::engine::chart::task::pair_key(e.0, e.1), e.2);
    }
}

/// Task `t` of the stage 2 grid: head `t / num_b`, column block `t % num_b`.
fn stage2_task(
    raw_f32: &[F32],
    z: &[i64],
    ssm_norm_f: &[F32],
    eps_f: F32,
    sg: &Stage2Geom,
    t: usize,
    writers: &ColWriters,
) -> Result<(), (usize, usize, StepError)> {
    let h = t / sg.num_b;
    let b = t % sg.num_b;
    let c0 = b * sg.per_col;
    let c1 = (c0 + sg.per_col).min(sg.k);
    if c0 >= c1 {
        return Ok(());
    }
    let head_v = sg.head_v;
    if head_v <= 256 {
        let mut o_stack = [F32::ZERO; 256];
        let mut d_stack = [0i64; 256];
        stage2_cols(
            raw_f32,
            z,
            ssm_norm_f,
            eps_f,
            sg,
            h,
            c0,
            c1,
            &mut o_stack[..head_v],
            &mut d_stack[..head_v],
            writers,
        )
    } else {
        let mut o_vec = vec![F32::ZERO; head_v];
        let mut d_vec = vec![0i64; head_v];
        stage2_cols(
            raw_f32, z, ssm_norm_f, eps_f, sg, h, c0, c1, &mut o_vec, &mut d_vec, writers,
        )
    }
}

/// Run the `n_tasks` stage 2 tasks, on the pool when `parallel`, else in task order; the first
/// refusal (smallest `(head, column)` on the pool, first task otherwise) is returned.
fn stage2_dispatch(
    raw_f32: &[F32],
    z: &[i64],
    ssm_norm_f: &[F32],
    eps_f: F32,
    sg: &Stage2Geom,
    n_tasks: usize,
    parallel: bool,
    writers: &ColWriters,
) -> Result<(), StepError> {
    if parallel {
        let first: crate::engine::chart::pool::FirstErr<u128, StepError> =
            crate::engine::chart::pool::FirstErr::new();
        let ctx = Stage2Ctx {
            raw_f32,
            z,
            ssm_norm_f,
            eps_f,
            sg,
            writers,
            first: &first,
        };
        crate::engine::chart::pool::run_global(n_tasks, &ctx);
        return match first.take() {
            Some((_, e)) => Err(e),
            None => Ok(()),
        };
    }
    let mut t = 0;
    while t < n_tasks {
        if let Err((_, _, e)) = stage2_task(raw_f32, z, ssm_norm_f, eps_f, sg, t, writers) {
            return Err(e);
        }
        t += 1;
    }
    Ok(())
}

/// Stage 2 in f32: per-(head, column block) RMS norm × `ssm_norm` × SiLU(z) gate, writing directly
/// into the per-column output vectors via [`ColWriters`] without an intermediate flat `out` buffer
/// or serial reassembly. `z` is the flat gate input, column `c` at `z[c·val_dim ..]`.
///
/// # Task decomposition
///
/// We divide `k` columns into `min(8, k)` column blocks of width `per_col = ceil(k / col_blocks)`.
/// With `n_v_heads` value heads, this gives up to `16 * 8 = 128` tasks, keeping all participants
/// in the pool busy (avoiding the 8-worker idle gap of 16-task head dispatches on a 24-wide pool)
/// with fine-grained load balancing between P and E cores.
///
/// # Safety
///
/// Each task `t` owns a disjoint `(head h, column block b)` slice: head `h` targets channels
/// `[h·head_v, (h+1)·head_v)` of columns `[c0, c1)`. Because `val_dim = n_v_heads · head_v` and
/// column blocks partition `0..k` disjointly, no two tasks write to the same `(column, channel)`
/// word. Each output column reserves capacity `val_dim`, and all words are written before
/// setting the length.
///
/// # Complexity
///
/// `O(k · val_dim)` f32 operations and `k · val_dim` words written, over `n_v_heads · num_b` pool
/// tasks.
#[allow(
    clippy::too_many_arguments,
    reason = "Stage 2 heads arguments match the affine layer geometry and carrier buffers"
)]
fn stage2_heads_flat(
    raw_f32: &[F32],
    z: &[i64],
    ssm_norm_f: &[F32],
    eps_f: F32,
    n_v_heads: usize,
    head_v: usize,
    slices: usize,
    k: usize,
    parallel: bool,
) -> Result<Vec<Vec<i64>>, StepError> {
    let val_dim = n_v_heads * head_v;
    let mut cols_out = alloc_cols(k, val_dim);
    let writers = col_writers(&mut cols_out);

    let col_blocks = 8.min(k).max(1);
    let per_col = k.div_ceil(col_blocks);
    let num_b = k.div_ceil(per_col);
    let n_tasks = n_v_heads * num_b;
    let sg = Stage2Geom {
        val_dim,
        head_v,
        slices,
        k,
        num_b,
        per_col,
    };

    let res = stage2_dispatch(
        raw_f32, z, ssm_norm_f, eps_f, &sg, n_tasks, parallel, &writers,
    );
    res?;

    // SAFETY: every refusal-free dispatch wrote all `val_dim` words of every column.
    unsafe { set_written_lens(&mut cols_out, val_dim) };

    Ok(cols_out)
}

/// [`stage2_heads_flat`] over borrowed columns (the gate input of each is `cols[c].1`).
#[cfg(test)]
#[allow(
    clippy::too_many_arguments,
    reason = "Stage 2 heads arguments match the affine layer geometry and carrier buffers"
)]
pub(crate) fn stage2_heads_cols_f32(
    raw_f32: &[F32],
    cols: &[AffineCol<'_>],
    ssm_norm_f: &[F32],
    eps_f: F32,
    n_v_heads: usize,
    head_v: usize,
    slices: usize,
    k: usize,
    parallel: bool,
) -> Result<Vec<Vec<i64>>, StepError> {
    let mut z = Vec::with_capacity(cols.len() * n_v_heads * head_v);
    flatten_z_rows(cols, 0, &mut z);
    stage2_heads_flat(
        raw_f32, &z, ssm_norm_f, eps_f, n_v_heads, head_v, slices, k, parallel,
    )
}

/// Decode (`k = 1`) fused head step with the staged convolution: head `h`'s q, k and v channels
/// convolved from the staged window, normalised, the delta recurrence over its slices, and the
/// gated norm into `dst`.
///
/// # Complexity
///
/// `O(conv_kernel · (2·head_k + head_v) + slices · head_k · w + head_v)` f32 operations
/// (`w = head_v / slices`), no heap allocation.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the layer, geometry, staged convolution and head state are the step's input"
)]
fn fused_head_staged(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    fa: &FlatAffine,
    staged: &StagedConv,
    packed: &[F32],
    scalar: &[F32],
    ssm_norm_f: &[F32],
    eps_f: F32,
    q_scale: F32,
    h: usize,
    dst: &mut [i64],
) -> Result<(), StepError> {
    let hk = g.head_k;
    let key_dim = g.n_k_heads * hk;
    let slices = layer.slices;
    let w = g.head_v / slices;
    let kh = h % g.n_k_heads;
    let mut q_stack = [F32::ZERO; 128];
    let mut k_stack = [F32::ZERO; 128];
    let mut v_stack = [F32::ZERO; 128];

    let q_buf = if hk <= 128 {
        &mut q_stack[..hk]
    } else {
        return Err(StepError::BadConfig(String::from("head_k > 128")));
    };
    let k_buf = if hk <= 128 {
        &mut k_stack[..hk]
    } else {
        return Err(StepError::BadConfig(String::from("head_k > 128")));
    };
    let v_buf = if g.head_v <= 128 {
        &mut v_stack[..g.head_v]
    } else {
        return Err(StepError::BadConfig(String::from("head_v > 128")));
    };

    // 1. Convolve q for key head kh
    let q_c0 = kh * hk;
    conv_silu_channel_range(
        packed,
        scalar,
        &staged.window_f32,
        &staged.xs_f32,
        staged.ch,
        1,
        q_c0,
        q_c0 + hk,
        q_buf,
    )?;
    f32_carrier::l2_normalize_f32_in_place(q_buf, eps_f, q_scale)?;

    // 2. Convolve k for key head kh
    let k_c0 = key_dim + kh * hk;
    conv_silu_channel_range(
        packed,
        scalar,
        &staged.window_f32,
        &staged.xs_f32,
        staged.ch,
        1,
        k_c0,
        k_c0 + hk,
        k_buf,
    )?;
    f32_carrier::l2_normalize_f32_in_place(k_buf, eps_f, F32::ONE)?;

    // 3. Convolve v for value head h
    let v_c0 = 2 * key_dim + h * g.head_v;
    conv_silu_channel_range(
        packed,
        scalar,
        &staged.window_f32,
        &staged.xs_f32,
        staged.ch,
        1,
        v_c0,
        v_c0 + g.head_v,
        v_buf,
    )?;

    // 4. Gates
    let dt_f = f32_carrier::q32_to_f32(p.dt[h]);
    let ssm_a_f = f32_carrier::q32_to_f32(p.a[h]);
    let b_f = f32_carrier::q32_to_f32(fa.b[h]);
    let a_f = f32_carrier::q32_to_f32(fa.a[h]);
    let beta_f = f32_carrier::sigmoid_f32(b_f);
    let sp = f32_carrier::softplus_f32(a_f + dt_f);
    let decay_f = f32_carrier::exp_f32(ssm_a_f * sp);
    if !beta_f.is_finite() || !decay_f.is_finite() {
        return Err(StepError::Overflow);
    }

    // 5. DeltaNet recurrence for each slice
    let mut o_stack = [F32::ZERO; 256];
    let o_slice = if g.head_v <= 256 {
        &mut o_stack[..g.head_v]
    } else {
        return Err(StepError::BadConfig(String::from("head_v > 256")));
    };

    let mut sl = 0;
    let mut carrier_err = false;
    while sl < slices && !carrier_err {
        let t = h * slices + sl;
        let mut guard = layer.heads[t].lock();
        if let HeadState::F32 { s: ref mut s_f32 } = guard.state {
            let v_slice = &v_buf[sl * w..(sl + 1) * w];
            let out_slice = &mut o_slice[sl * w..(sl + 1) * w];
            f32_carrier::delta_step_f32_avx2(
                s_f32, out_slice, q_buf, k_buf, v_slice, decay_f, beta_f,
            );
        } else {
            carrier_err = true;
        }
        sl += 1;
    }
    if carrier_err {
        return Err(StepError::BadConfig(String::from("affine carrier")));
    }

    // 6. Stage 2 Gated Norm
    let zo = h * g.head_v;
    f32_carrier::stage2_gated_norm_f32(o_slice, &fa.z[zo..zo + g.head_v], ssm_norm_f, eps_f, dst)?;
    Ok(())
}

/// The shared inputs of [`fused_head_staged`] for the decode head dispatch.
struct FusedStagedCtx<'a> {
    layer: &'a AffineLayer,
    g: &'a AffineGeom,
    p: &'a AffineVectors,
    fa: &'a FlatAffine,
    staged: &'a StagedConv,
    packed: &'a [F32],
    scalar: &'a [F32],
    ssm_norm_f: &'a [F32],
    eps_f: F32,
    q_scale: F32,
}

impl HeadTask<i64, StepError> for FusedStagedCtx<'_> {
    fn run(&self, h: usize, dst: &mut [i64]) -> Result<(), StepError> {
        fused_head_staged(
            self.layer,
            self.g,
            self.p,
            self.fa,
            self.staged,
            self.packed,
            self.scalar,
            self.ssm_norm_f,
            self.eps_f,
            self.q_scale,
            h,
            dst,
        )
    }
}

/// The shared inputs of [`norm_qk_f32`] for the key-head dispatch.
struct NormQkF32Ctx<'a> {
    layer: &'a AffineLayer,
    fa: &'a FlatAffine,
    staged_opt: &'a Option<StagedConv>,
    cf: &'a [F32],
    use_cf: bool,
    key_dim: usize,
    cdim: usize,
    hk: usize,
    eps_f: F32,
    q_scale: F32,
}

impl HeadTask<F32, StepError> for NormQkF32Ctx<'_> {
    fn run(&self, h: usize, dst: &mut [F32]) -> Result<(), StepError> {
        norm_qk_f32(
            self.layer,
            self.fa,
            self.staged_opt,
            self.cf,
            self.use_cf,
            self.key_dim,
            self.cdim,
            self.hk,
            self.eps_f,
            self.q_scale,
            h,
            dst,
        )
    }
}

/// The shared inputs of [`head_gates_f32`] for the value-head dispatch.
struct GatesF32Ctx<'a> {
    fa: &'a FlatAffine,
    p: &'a AffineVectors,
    nv: usize,
}

impl HeadTask<F32, StepError> for GatesF32Ctx<'_> {
    fn run(&self, h: usize, dst: &mut [F32]) -> Result<(), StepError> {
        head_gates_f32(self.fa, self.p, self.nv, h, dst)
    }
}

/// The shared inputs of [`fused_head_nostage`] for the decode head dispatch.
struct FusedNoStageCtx<'a> {
    layer: &'a AffineLayer,
    g: &'a AffineGeom,
    fa: &'a FlatAffine,
    cf: &'a [F32],
    use_cf: bool,
    ssm_norm_f: &'a [F32],
    eps_f: F32,
    qk_f32: &'a [F32],
    bd_f32: &'a [F32],
}

impl HeadTask<i64, StepError> for FusedNoStageCtx<'_> {
    fn run(&self, h: usize, dst: &mut [i64]) -> Result<(), StepError> {
        fused_head_nostage(
            self.layer,
            self.g,
            self.fa,
            self.cf,
            self.use_cf,
            self.ssm_norm_f,
            self.eps_f,
            self.qk_f32,
            self.bd_f32,
            h,
            dst,
        )
    }
}

/// The shared inputs of [`update_f32`] for the (value head, slice) dispatch; `t` is the task index.
struct UpdateF32Ctx<'a> {
    layer: &'a AffineLayer,
    g: &'a AffineGeom,
    fa: &'a FlatAffine,
    staged_opt: &'a Option<StagedConv>,
    cf: &'a [F32],
    use_cf: bool,
    snap: &'a AffineLayer,
    has_snap: bool,
    qk_f32: &'a [F32],
    bd_f32: &'a [F32],
}

impl HeadTask<F32, StepError> for UpdateF32Ctx<'_> {
    fn run(&self, t: usize, dst: &mut [F32]) -> Result<(), StepError> {
        update_f32(
            self.layer,
            self.g,
            self.fa,
            self.staged_opt,
            self.cf,
            self.use_cf,
            self.snap,
            self.has_snap,
            self.qk_f32,
            self.bd_f32,
            t,
            dst,
        )
    }
}

/// The shared inputs of [`norm_qk_q32`] for the key-head dispatch.
struct NormQkQ32Ctx<'a> {
    fa: &'a FlatAffine,
    key_dim: usize,
    cdim: usize,
    hk: usize,
    eps: i64,
    q_scale: i64,
}

impl HeadTask<i64, StepError> for NormQkQ32Ctx<'_> {
    fn run(&self, h: usize, dst: &mut [i64]) -> Result<(), StepError> {
        norm_qk_q32(
            self.fa,
            self.key_dim,
            self.cdim,
            self.hk,
            self.eps,
            self.q_scale,
            h,
            dst,
        )
    }
}

/// The shared inputs of [`head_gates_q32`] for the value-head dispatch.
struct GatesQ32Ctx<'a> {
    fa: &'a FlatAffine,
    p: &'a AffineVectors,
    nv: usize,
}

impl HeadTask<i64, StepError> for GatesQ32Ctx<'_> {
    fn run(&self, h: usize, dst: &mut [i64]) -> Result<(), StepError> {
        head_gates_q32(self.fa, self.p, self.nv, h, dst)
    }
}

/// The shared inputs of [`update_q32`] for the (value head, slice) dispatch; `t` is the task index.
struct UpdateQ32Ctx<'a> {
    layer: &'a AffineLayer,
    g: &'a AffineGeom,
    fa: &'a FlatAffine,
    snap: &'a AffineLayer,
    has_snap: bool,
    qk: &'a [i64],
    bd: &'a [i64],
    chunked: bool,
}

impl HeadTask<i64, StepError> for UpdateQ32Ctx<'_> {
    fn run(&self, t: usize, dst: &mut [i64]) -> Result<(), StepError> {
        update_q32(
            self.layer,
            self.g,
            self.fa,
            self.snap,
            self.has_snap,
            self.qk,
            self.bd,
            self.chunked,
            t,
            dst,
        )
    }
}

/// The shared inputs of [`gate_norm_q32`] for the value-head dispatch.
struct GateNormQ32Ctx<'a> {
    fa: &'a FlatAffine,
    p: &'a AffineVectors,
    head_v: usize,
    val_dim: usize,
    slices: usize,
    raw: &'a [i64],
    eps: i64,
}

impl HeadTask<i64, StepError> for GateNormQ32Ctx<'_> {
    fn run(&self, h: usize, dst: &mut [i64]) -> Result<(), StepError> {
        gate_norm_q32(
            self.fa,
            self.p,
            self.head_v,
            self.val_dim,
            self.slices,
            self.raw,
            self.eps,
            h,
            dst,
        )
    }
}

/// The decode (`k = 1`) f32 path with a staged convolution: every value head through
/// [`fused_head_staged`], then the staged state is consumed.
///
/// # Complexity
///
/// `n_v_heads` calls of [`fused_head_staged`] over the pool when `parallel`.
#[allow(
    clippy::too_many_arguments,
    reason = "the layer, geometry, parameters, staged convolution and f32 constants are the input"
)]
fn f32_decode_staged(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    fa: &FlatAffine,
    staged: &StagedConv,
    ssm_norm_f: &[F32],
    eps_f: F32,
    q_scale: F32,
    parallel: bool,
) -> Result<(Vec<Vec<i64>>, Vec<i64>), StepError> {
    let mut out = vec![0i64; g.n_v_heads * g.head_v];
    let packed_guard = layer.conv_w_packed.lock();
    let scalar_guard = layer.conv_w_scalar.lock();
    let packed: &[F32] = &packed_guard;
    let scalar: &[F32] = &scalar_guard;
    let step = FusedStagedCtx {
        layer,
        g,
        p,
        fa,
        staged,
        packed,
        scalar,
        ssm_norm_f,
        eps_f,
        q_scale,
    };
    let res = attn_heads(g.n_v_heads, g.head_v, &mut out, parallel, &step);
    drop(packed_guard);
    drop(scalar_guard);
    *layer.staged_conv.lock() = None;

    if let Some((_, e)) = res {
        return Err(e);
    }
    Ok((vec![out], Vec::new()))
}

/// Stage 0 in f32 for key head `kh`: every column's L2-normalised `q̃` then `k̂` into `dst`, from
/// the staged convolution, the f32 conv cache (`use_cf`) or the Q32 conv input.
///
/// # Complexity
///
/// `O(k · head_k)` f32 operations plus, with a staged convolution, the `k`-column convolution of
/// the head's `2 · head_k` channels.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the layer, staged convolution, conv cache and f32 constants are the step's input"
)]
fn norm_qk_f32(
    layer: &AffineLayer,
    fa: &FlatAffine,
    staged_opt: &Option<StagedConv>,
    cf: &[F32],
    use_cf: bool,
    key_dim: usize,
    cdim: usize,
    hk: usize,
    eps_f: F32,
    q_scale: F32,
    kh: usize,
    dst: &mut [F32],
) -> Result<(), StepError> {
    let k = fa.k;
    let kr0 = kh * hk;
    let kr1 = (kh + 1) * hk;
    if let Some(staged) = staged_opt {
        let packed_guard = layer.conv_w_packed.lock();
        let scalar_guard = layer.conv_w_scalar.lock();
        let mut q_col = vec![F32::ZERO; k * hk];
        let mut k_col = vec![F32::ZERO; k * hk];
        conv_silu_channel_range(
            &packed_guard,
            &scalar_guard,
            &staged.window_f32,
            &staged.xs_f32,
            staged.ch,
            k,
            kr0,
            kr1,
            &mut q_col,
        )?;
        conv_silu_channel_range(
            &packed_guard,
            &scalar_guard,
            &staged.window_f32,
            &staged.xs_f32,
            staged.ch,
            k,
            key_dim + kr0,
            key_dim + kr1,
            &mut k_col,
        )?;
        let mut err: Option<StepError> = None;
        let mut c = 0;
        while c < k && err.is_none() {
            let base = c * 2 * hk;
            dst[base..base + hk].copy_from_slice(&q_col[c * hk..(c + 1) * hk]);
            let mut r =
                f32_carrier::l2_normalize_f32_in_place(&mut dst[base..base + hk], eps_f, q_scale);
            if r.is_ok() {
                dst[base + hk..base + 2 * hk].copy_from_slice(&k_col[c * hk..(c + 1) * hk]);
                r = f32_carrier::l2_normalize_f32_in_place(
                    &mut dst[base + hk..base + 2 * hk],
                    eps_f,
                    F32::ONE,
                );
            }
            if let Err(e) = r {
                err = Some(e);
            }
            c += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
    } else {
        let mut err: Option<StepError> = None;
        let mut c = 0;
        while c < k && err.is_none() {
            let base = c * 2 * hk;
            let q_at = c * cdim + kr0;
            let k_at = c * cdim + key_dim + kr0;
            let mut i = 0;
            while i < hk {
                dst[base + i] = if use_cf {
                    cf[q_at + i]
                } else {
                    f32_carrier::q32_to_f32(fa.conv[q_at + i])
                };
                i += 1;
            }
            let mut r =
                f32_carrier::l2_normalize_f32_in_place(&mut dst[base..base + hk], eps_f, q_scale);
            if r.is_ok() {
                let mut i = 0;
                while i < hk {
                    dst[base + hk + i] = if use_cf {
                        cf[k_at + i]
                    } else {
                        f32_carrier::q32_to_f32(fa.conv[k_at + i])
                    };
                    i += 1;
                }
                r = f32_carrier::l2_normalize_f32_in_place(
                    &mut dst[base + hk..base + 2 * hk],
                    eps_f,
                    F32::ONE,
                );
            }
            if let Err(e) = r {
                err = Some(e);
            }
            c += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
    }
    Ok(())
}

/// Stage 0.5 in f32 for value head `h`: `β = σ(b)` and `α = exp(ssm_a · softplus(a + dt))` for each
/// column into `dst[c·2]`, `dst[c·2 + 1]`.
///
/// # Complexity
///
/// `O(k)` f32 transcendental evaluations.
fn head_gates_f32(
    fa: &FlatAffine,
    p: &AffineVectors,
    nv: usize,
    h: usize,
    dst: &mut [F32],
) -> Result<(), StepError> {
    let dt_f = f32_carrier::q32_to_f32(p.dt[h]);
    let ssm_a_f = f32_carrier::q32_to_f32(p.a[h]);
    let mut c = 0;
    while c < fa.k {
        let b_f = f32_carrier::q32_to_f32(fa.b[c * nv + h]);
        let a_f = f32_carrier::q32_to_f32(fa.a[c * nv + h]);
        let beta = f32_carrier::sigmoid_f32(b_f);
        let sp = f32_carrier::softplus_f32(a_f + dt_f);
        let decay = f32_carrier::exp_f32(ssm_a_f * sp);
        if !beta.is_finite() || !decay.is_finite() {
            return Err(StepError::Overflow);
        }
        dst[c * 2] = beta;
        dst[c * 2 + 1] = decay;
        c += 1;
    }
    Ok(())
}

/// Decode (`k = 1`) fused stage 1 + stage 2 for value head `h` without a staged convolution, with
/// caller-provided scratch `o_slice` (`head_v` f32) and `v_slice` (`w` f32).
///
/// # Complexity
///
/// `O(slices · head_k · w + head_v)` f32 operations, no allocation.
#[allow(
    clippy::too_many_arguments,
    reason = "the layer, geometry, conv cache, stage 0 outputs and scratch are the step's input"
)]
fn fused_head_body(
    layer: &AffineLayer,
    g: &AffineGeom,
    fa: &FlatAffine,
    cf: &[F32],
    use_cf: bool,
    ssm_norm_f: &[F32],
    eps_f: F32,
    qk_f32: &[F32],
    bd_f32: &[F32],
    h: usize,
    dst: &mut [i64],
    o_slice: &mut [F32],
    v_slice: &mut [F32],
) -> Result<(), StepError> {
    let hk = g.head_k;
    let key_dim = g.n_k_heads * hk;
    let slices = layer.slices;
    let w = g.head_v / slices;
    let kh = h % g.n_k_heads;
    let at = kh * 2 * hk;
    let beta_f = bd_f32[h * 2];
    let decay_f = bd_f32[h * 2 + 1];

    let mut sl = 0;
    while sl < slices {
        let t = h * slices + sl;
        let vs = h * g.head_v + sl * w;
        let mut guard = layer.heads[t].lock();
        let HeadState::F32 { s: ref mut s_f32 } = guard.state else {
            return Err(StepError::BadConfig(String::from("affine carrier")));
        };
        if use_cf {
            v_slice.copy_from_slice(&cf[2 * key_dim + vs..2 * key_dim + vs + w]);
        } else {
            let mut i = 0;
            while i < w {
                v_slice[i] = f32_carrier::q32_to_f32(fa.conv[2 * key_dim + vs + i]);
                i += 1;
            }
        }

        let out_slice = &mut o_slice[sl * w..(sl + 1) * w];
        f32_carrier::delta_step_f32_avx2(
            s_f32,
            out_slice,
            &qk_f32[at..at + hk],
            &qk_f32[at + hk..at + 2 * hk],
            v_slice,
            decay_f,
            beta_f,
        );
        sl += 1;
    }

    let zo = h * g.head_v;
    f32_carrier::stage2_gated_norm_f32(o_slice, &fa.z[zo..zo + g.head_v], ssm_norm_f, eps_f, dst)?;
    Ok(())
}

/// [`fused_head_body`] with its two scratch buffers on the stack when `head_v <= 256` and
/// `head_v / slices <= 128`, else on the heap.
#[allow(
    clippy::too_many_arguments,
    reason = "the layer, geometry, conv cache and stage 0 outputs are the step's input"
)]
fn fused_head_nostage(
    layer: &AffineLayer,
    g: &AffineGeom,
    fa: &FlatAffine,
    cf: &[F32],
    use_cf: bool,
    ssm_norm_f: &[F32],
    eps_f: F32,
    qk_f32: &[F32],
    bd_f32: &[F32],
    h: usize,
    dst: &mut [i64],
) -> Result<(), StepError> {
    let w = g.head_v / layer.slices;
    if g.head_v <= 256 && w <= 128 {
        let mut o_stack = [F32::ZERO; 256];
        let mut v_stack = [F32::ZERO; 128];
        fused_head_body(
            layer,
            g,
            fa,
            cf,
            use_cf,
            ssm_norm_f,
            eps_f,
            qk_f32,
            bd_f32,
            h,
            dst,
            &mut o_stack[..g.head_v],
            &mut v_stack[..w],
        )
    } else {
        let mut o_vec = vec![F32::ZERO; g.head_v];
        let mut v_vec = vec![F32::ZERO; w];
        fused_head_body(
            layer, g, fa, cf, use_cf, ssm_norm_f, eps_f, qk_f32, bd_f32, h, dst, &mut o_vec,
            &mut v_vec,
        )
    }
}

/// The rollback snapshot when present, else `layer` (never read when no snapshot is present).
fn snapshot_or<'a>(snapshot: Option<&'a AffineLayer>, layer: &'a AffineLayer) -> &'a AffineLayer {
    match snapshot {
        Some(s) => s,
        None => layer,
    }
}

/// Copy the f32 state `s` into task `t`'s rollback snapshot when that holds an f32 state.
fn snapshot_f32_state(snap: &AffineLayer, t: usize, s: &Vec<F32>) {
    let mut snap_guard = snap.heads[t].lock();
    if let HeadState::F32 { s: ref mut snap_s } = snap_guard.state {
        snap_s.clone_from(s);
    }
}

/// The staged convolution of channels `[lo, lo + w)` over `k` columns, `vc[c·w ..]`; empty when
/// there is no staged convolution.
///
/// # Complexity
///
/// `O(k · w · conv_kernel)` f32 operations, `k · w` f32 allocated.
fn staged_v_col(
    layer: &AffineLayer,
    staged_opt: &Option<StagedConv>,
    k: usize,
    w: usize,
    lo: usize,
) -> Result<Vec<F32>, StepError> {
    let Some(staged) = staged_opt else {
        return Ok(Vec::new());
    };
    let packed_guard = layer.conv_w_packed.lock();
    let scalar_guard = layer.conv_w_scalar.lock();
    let mut vc = vec![F32::ZERO; k * w];
    conv_silu_channel_range(
        &packed_guard,
        &scalar_guard,
        &staged.window_f32,
        &staged.xs_f32,
        staged.ch,
        k,
        lo,
        lo + w,
        &mut vc,
    )?;
    Ok(vc)
}

/// Stage 1 in f32 for task `t` (value head `t / slices`, slice `t % slices`) through all `k`
/// columns in token order, with caller-provided scratch `v_slice` (`w` f32). The rollback copy of
/// the slice state is taken before column 1.
///
/// # Complexity
///
/// `O(k · head_k · w)` f32 operations for the recurrence, plus the `k`-column convolution of the
/// slice's `w` channels when staged; `k · w` scratch f32 when staged.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the layer, geometry, conv cache, snapshot and stage 0 outputs are the step's input"
)]
fn update_f32_body(
    layer: &AffineLayer,
    g: &AffineGeom,
    fa: &FlatAffine,
    staged_opt: &Option<StagedConv>,
    cf: &[F32],
    use_cf: bool,
    snap: &AffineLayer,
    has_snap: bool,
    qk_f32: &[F32],
    bd_f32: &[F32],
    t: usize,
    dst: &mut [F32],
    v_slice: &mut [F32],
) -> Result<(), StepError> {
    let hk = g.head_k;
    let key_dim = g.n_k_heads * hk;
    let val_dim = g.n_v_heads * g.head_v;
    let cdim = 2 * key_dim + val_dim;
    let k = fa.k;
    let slices = layer.slices;
    let w = g.head_v / slices;
    let (h, sl) = (t / slices, t % slices);
    let kh = h % g.n_k_heads;
    let vs = h * g.head_v + sl * w;
    let mut guard = layer.heads[t].lock();
    let HeadState::F32 { s: ref mut s_f32 } = guard.state else {
        return Err(StepError::BadConfig(String::from("affine carrier")));
    };

    let has_vc = staged_opt.is_some();
    let vc = staged_v_col(layer, staged_opt, k, w, 2 * key_dim + vs)?;

    let mut c = 0;
    while c < k {
        if c == 1 && has_snap {
            snapshot_f32_state(snap, t, s_f32);
        }
        let at = kh * k * 2 * hk + c * 2 * hk;
        if has_vc {
            v_slice.copy_from_slice(&vc[c * w..(c + 1) * w]);
        } else if use_cf {
            let v_at = c * cdim + 2 * key_dim + vs;
            v_slice.copy_from_slice(&cf[v_at..v_at + w]);
        } else {
            let v_at = c * cdim + 2 * key_dim + vs;
            let mut i = 0;
            while i < w {
                v_slice[i] = f32_carrier::q32_to_f32(fa.conv[v_at + i]);
                i += 1;
            }
        }
        let beta_f = bd_f32[h * k * 2 + c * 2];
        let decay_f = bd_f32[h * k * 2 + c * 2 + 1];

        let out_slice = &mut dst[c * w..(c + 1) * w];
        f32_carrier::delta_step_f32_avx2(
            s_f32,
            out_slice,
            &qk_f32[at..at + hk],
            &qk_f32[at + hk..at + 2 * hk],
            v_slice,
            decay_f,
            beta_f,
        );
        c += 1;
    }
    Ok(())
}

/// [`update_f32_body`] with its `w`-word scratch on the stack when `w <= 128`, else on the heap.
#[allow(
    clippy::too_many_arguments,
    reason = "the layer, geometry, conv cache, snapshot and stage 0 outputs are the step's input"
)]
fn update_f32(
    layer: &AffineLayer,
    g: &AffineGeom,
    fa: &FlatAffine,
    staged_opt: &Option<StagedConv>,
    cf: &[F32],
    use_cf: bool,
    snap: &AffineLayer,
    has_snap: bool,
    qk_f32: &[F32],
    bd_f32: &[F32],
    t: usize,
    dst: &mut [F32],
) -> Result<(), StepError> {
    let w = g.head_v / layer.slices;
    if w <= 128 {
        let mut v_stack = [F32::ZERO; 128];
        update_f32_body(
            layer,
            g,
            fa,
            staged_opt,
            cf,
            use_cf,
            snap,
            has_snap,
            qk_f32,
            bd_f32,
            t,
            dst,
            &mut v_stack[..w],
        )
    } else {
        let mut v_vec = vec![F32::ZERO; w];
        update_f32_body(
            layer, g, fa, staged_opt, cf, use_cf, snap, has_snap, qk_f32, bd_f32, t, dst,
            &mut v_vec,
        )
    }
}

/// The IEEE binary32 carrier path of [`delta_heads_vecs_impl`]: stages 0, 0.5, 1 and 2 entirely in
/// single-precision floating point, carrying f32 end-to-end between stages.
///
/// # Complexity
///
/// As [`delta_heads_cols_vecs`] with f32 operations; plus `O(conv_kernel · channels)` words for the
/// cloned staged convolution.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the layer state, geometry, parameters, columns and snapshot are the step's input; \
              the three stages read in order"
)]
fn delta_heads_f32(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    fa: &FlatAffine,
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
) -> Result<(Vec<Vec<i64>>, Vec<i64>), StepError> {
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let cdim = 2 * key_dim + val_dim;
    let k = fa.k;
    let slices = layer.slices;
    let w = g.head_v / slices;
    let n_tasks = g.n_v_heads * slices;
    let hk = g.head_k;

    let eps_f = f32_carrier::q32_to_f32(eps);
    if !eps_f.is_finite() || eps_f.le(F32::ZERO) {
        return Err(StepError::BadConfig(String::from("l2 norm eps")));
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "g.head_k is small (typically <= 256)"
    )]
    // head_k is a head width (small usize), exact in i64 and in binary32.
    #[allow(clippy::cast_possible_wrap, reason = "head width < 2^53")]
    let q_scale = F32::ONE / F32::from_i64(g.head_k as i64).sqrt();
    if !q_scale.is_finite() {
        return Err(StepError::Overflow);
    }

    let has_snap = snapshot.is_some();
    let snap = snapshot_or(snapshot, layer);
    let staged_opt = layer.staged_conv.lock().clone();

    if let Some(ref staged) = staged_opt {
        layer.ensure_packed_w(&p.conv, staged.ch, 4);
    }

    let conv_f32_guard = layer.conv_shared.lock();
    let use_cf = conv_f32_guard.owner == layer.slot
        && conv_f32_guard.k == k
        && conv_f32_guard.buf.len() == k * cdim;
    let cf: &[F32] = &conv_f32_guard.buf;

    let mut ssm_norm_f: Vec<F32> = Vec::with_capacity(p.norm.len());
    let mut ni = 0usize;
    while ni < p.norm.len() {
        ssm_norm_f.push(f32_carrier::q32_to_f32(p.norm[ni]));
        ni += 1;
    }

    if k == 1 {
        if let Some(ref staged) = staged_opt {
            return f32_decode_staged(
                layer,
                g,
                p,
                fa,
                staged,
                &ssm_norm_f,
                eps_f,
                q_scale,
                parallel,
            );
        }
    }

    // A one-column block runs its stages serially whatever `parallel` says.
    let par = parallel && k > 1;

    // Stage 0 in f32: L2-normalize q and k, scale q by 1/sqrt(head_k)
    let mut qk_f32 = vec![F32::ZERO; k * 2 * key_dim];
    let norm_qk_f32_step = NormQkF32Ctx {
        layer,
        fa,
        staged_opt: &staged_opt,
        cf,
        use_cf,
        key_dim,
        cdim,
        hk,
        eps_f,
        q_scale,
    };
    if let Some((_, e)) =
        attn_heads_f32(g.n_k_heads, k * 2 * hk, &mut qk_f32, par, &norm_qk_f32_step)
    {
        return Err(e);
    }

    // Stage 0.5 in f32: β = sigmoid(b) and α = exp(ssm_a · softplus(a + dt))
    let mut bd_f32 = vec![F32::ZERO; g.n_v_heads * k * 2];
    let nv = g.n_v_heads;
    let gates_step = GatesF32Ctx { fa, p, nv };
    if let Some((_, e)) = attn_heads_f32(g.n_v_heads, k * 2, &mut bd_f32, par, &gates_step) {
        return Err(e);
    }

    if k == 1 {
        let mut out = vec![0i64; val_dim];
        // Fused Stage 1 + Stage 2 for decode (k = 1) fallback:
        let fused_step = FusedNoStageCtx {
            layer,
            g,
            fa,
            cf,
            use_cf,
            ssm_norm_f: &ssm_norm_f,
            eps_f,
            qk_f32: &qk_f32,
            bd_f32: &bd_f32,
        };
        if let Some((_, e)) = attn_heads(g.n_v_heads, g.head_v, &mut out, parallel, &fused_step) {
            return Err(e);
        }

        return Ok((vec![out], Vec::new()));
    }

    // Stage 1 in f32: DeltaNet recurrence writing directly into raw_f32
    let mut raw_f32 = vec![F32::ZERO; k * val_dim];
    let update_step = UpdateF32Ctx {
        layer,
        g,
        fa,
        staged_opt: &staged_opt,
        cf,
        use_cf,
        snap,
        has_snap,
        qk_f32: &qk_f32,
        bd_f32: &bd_f32,
    };
    if let Some((_, e)) = attn_heads_f32(n_tasks, k * w, &mut raw_f32, parallel, &update_step) {
        return Err(e);
    }

    let cols_out = stage2_heads_flat(
        &raw_f32,
        &fa.z,
        &ssm_norm_f,
        eps_f,
        g.n_v_heads,
        g.head_v,
        slices,
        k,
        parallel,
    )?;
    *layer.staged_conv.lock() = None;
    Ok((cols_out, Vec::new()))
}

/// Stage 0 on the Q32 carrier for key head `kh`: every column's `q̃` then `k̂` into `dst`.
///
/// # Complexity
///
/// `O(k · head_k)` Q32 operations and `k · 2 · head_k` words written.
#[allow(
    clippy::too_many_arguments,
    reason = "scalars and plain slices keep every reference parameter single-level"
)]
fn norm_qk_q32(
    fa: &FlatAffine,
    key_dim: usize,
    cdim: usize,
    hk: usize,
    eps: i64,
    q_scale: i64,
    kh: usize,
    dst: &mut [i64],
) -> Result<(), StepError> {
    let kr0 = kh * hk;
    let mut err: Option<StepError> = None;
    let mut c = 0;
    while c < fa.k && err.is_none() {
        let base = c * cdim;
        match l2_normalize(&fa.conv[base + kr0..base + kr0 + hk], eps) {
            Err(e) => err = Some(e),
            Ok(q_hat) => match l2_normalize(
                &fa.conv[base + key_dim + kr0..base + key_dim + kr0 + hk],
                eps,
            ) {
                Err(e) => err = Some(e),
                Ok(k_hat) => {
                    let d = &mut dst[c * 2 * hk..(c + 1) * 2 * hk];
                    let mut i = 0;
                    while i < hk && err.is_none() {
                        match mul_q32(q_hat[i], q_scale) {
                            Some(x) => d[i] = x,
                            None => err = Some(StepError::Overflow),
                        }
                        i += 1;
                    }
                    if err.is_none() {
                        d[hk..].copy_from_slice(&k_hat);
                    }
                }
            },
        }
        c += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// Stage 0.5 on the Q32 carrier for value head `h`: `β = σ(b_h)` and
/// `α = exp(ssm_a_h·softplus(a_h + dt_h))` per column into `dst[c·2]`, `dst[c·2 + 1]`. Every
/// `fixed_*` refusal maps to Overflow.
///
/// # Complexity
///
/// `O(k)` fixed-point transcendental evaluations.
fn head_gates_q32(
    fa: &FlatAffine,
    p: &AffineVectors,
    nv: usize,
    h: usize,
    dst: &mut [i64],
) -> Result<(), StepError> {
    let mut err: Option<StepError> = None;
    let mut c = 0;
    while c < fa.k && err.is_none() {
        let beta = fixed_sigmoid(fa.b[c * nv + h]);
        if let Ok(x) = beta {
            dst[c * 2] = x;
        } else {
            err = Some(StepError::Overflow);
        }
        if err.is_none() {
            let sum = fa.a[c * nv + h].checked_add(p.dt[h]);
            let sp = match sum {
                Some(x) => fixed_softplus(x).ok(),
                None => None,
            };
            let m = match sp {
                Some(x) => mul_q32(p.a[h], x),
                None => None,
            };
            let decay = match m {
                Some(x) => fixed_exp(x).ok(),
                None => None,
            };
            match decay {
                Some(x) => dst[c * 2 + 1] = x,
                None => err = Some(StepError::Overflow),
            }
        }
        c += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// Stage 1 on the Q32 carrier for task `t` (value head `t / slices`, slice `t % slices`): the
/// slice's update through all `k` columns in token order, into `dst[c·w ..]`. With `chunked`, the
/// whole block runs through one composed affine map per `<= CHUNK_MAX` columns
/// ([`delta_head_chunk`]); a refusal before a chunk boundary writes nothing. Otherwise the rollback
/// copy is taken before column 1.
///
/// # Complexity
///
/// `O(k · head_k · w)` Q32 operations.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the layer, geometry, columns, snapshot and stage 0 outputs are the step's input"
)]
fn update_q32(
    layer: &AffineLayer,
    g: &AffineGeom,
    fa: &FlatAffine,
    snap: &AffineLayer,
    has_snap: bool,
    qk: &[i64],
    bd: &[i64],
    chunked: bool,
    t: usize,
    dst: &mut [i64],
) -> Result<(), StepError> {
    let hk = g.head_k;
    let key_dim = g.n_k_heads * hk;
    let cdim = 2 * key_dim + g.n_v_heads * g.head_v;
    let k = fa.k;
    let slices = layer.slices;
    let w = g.head_v / slices;
    let (h, sl) = (t / slices, t % slices);
    let kh = h % g.n_k_heads;
    let vs = h * g.head_v + sl * w;
    let mut guard = layer.heads[t].lock();
    let mut err: Option<StepError> = None;

    if chunked {
        let HeadState::Q32 {
            ref mut s,
            c: ref mut scale,
        } = guard.state
        else {
            return Err(StepError::BadConfig(String::from("affine carrier")));
        };
        let mut q_buf = vec![0i64; CHUNK_MAX * hk];
        let mut k_buf = vec![0i64; CHUNK_MAX * hk];
        let mut v_buf = vec![0i64; CHUNK_MAX * w];
        let mut beta_buf = vec![0i64; CHUNK_MAX];
        let mut decay_buf = vec![0i64; CHUNK_MAX];
        let mut start = 0usize;
        while start < k && err.is_none() {
            let len = (k - start).min(CHUNK_MAX);
            let mut c = start;
            while c < start + len {
                let at = kh * k * 2 * hk + c * 2 * hk;
                let off = (c - start) * hk;
                q_buf[off..off + hk].copy_from_slice(&qk[at..at + hk]);
                k_buf[off..off + hk].copy_from_slice(&qk[at + hk..at + 2 * hk]);
                beta_buf[c - start] = bd[h * k * 2 + c * 2];
                decay_buf[c - start] = bd[h * k * 2 + c * 2 + 1];
                let v_at = c * cdim + 2 * key_dim + vs;
                v_buf[(c - start) * w..(c - start + 1) * w]
                    .copy_from_slice(&fa.conv[v_at..v_at + w]);
                c += 1;
            }
            match delta_head_chunk(
                s,
                scale,
                &q_buf[..len * hk],
                &k_buf[..len * hk],
                &v_buf[..len * w],
                &decay_buf[..len],
                &beta_buf[..len],
                hk,
                w,
                len,
            ) {
                Err(e) => err = Some(e),
                Ok(o) => {
                    let mut c = start;
                    while c < start + len {
                        let src = (c - start) * w;
                        dst[c * w..(c + 1) * w].copy_from_slice(&o[src..src + w]);
                        c += 1;
                    }
                }
            }
            start += len;
        }
        return match err {
            Some(e) => Err(e),
            None => Ok(()),
        };
    }
    let mut c = 0;
    while c < k && err.is_none() {
        if c == 1 && has_snap {
            snap.heads[t].lock().clone_from(&guard);
        }
        let at = kh * k * 2 * hk + c * 2 * hk;
        let beta = bd[h * k * 2 + c * 2];
        let decay = bd[h * k * 2 + c * 2 + 1];
        let v_at = c * cdim + 2 * key_dim + vs;
        if let HeadState::Q32 {
            ref mut s,
            c: ref mut scale,
        } = guard.state
        {
            match delta_head_step(
                s,
                scale,
                &qk[at..at + hk],
                &qk[at + hk..at + 2 * hk],
                &fa.conv[v_at..v_at + w],
                decay,
                beta,
            ) {
                Err(e) => err = Some(e),
                Ok(o) => dst[c * w..(c + 1) * w].copy_from_slice(&o),
            }
        } else {
            err = Some(StepError::BadConfig(String::from("affine carrier")));
        }
        c += 1;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Stage 2 on the Q32 carrier for value head `h`: each column's `o_h` reassembled from its slices
/// out of `raw`, then the gated norm into `dst[c·head_v ..]`.
///
/// # Complexity
///
/// `O(k · head_v)` Q32 operations.
#[allow(
    clippy::too_many_arguments,
    reason = "scalars and plain slices keep every reference parameter single-level"
)]
fn gate_norm_q32(
    fa: &FlatAffine,
    p: &AffineVectors,
    head_v: usize,
    val_dim: usize,
    slices: usize,
    raw: &[i64],
    eps: i64,
    h: usize,
    dst: &mut [i64],
) -> Result<(), StepError> {
    let k = fa.k;
    let w = head_v / slices;
    let mut o = vec![0i64; head_v];
    let mut err: Option<StepError> = None;
    let mut c = 0;
    while c < k && err.is_none() {
        let zo = c * val_dim + h * head_v;
        let mut sl = 0;
        while sl < slices {
            let at = (h * slices + sl) * k * w + c * w;
            o[sl * w..(sl + 1) * w].copy_from_slice(&raw[at..at + w]);
            sl += 1;
        }
        match rms_norm_fixed(&o, eps) {
            Err(e) => err = Some(StepError::Norm(e)),
            Ok(normed) => {
                let mut j = 0usize;
                while j < head_v && err.is_none() {
                    match mul_q32(normed[j], p.norm[j]) {
                        None => err = Some(StepError::Overflow),
                        Some(gained) => match silu_fixed(fa.z[zo + j]) {
                            Err(e) => err = Some(StepError::Ffn(e)),
                            Ok(gate) => match mul_q32(gained, gate) {
                                Some(x) => dst[c * head_v + j] = x,
                                None => err = Some(StepError::Overflow),
                            },
                        },
                    }
                    j += 1;
                }
            }
        }
        c += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The Q32 carrier path of [`delta_heads_vecs_impl`], returning the columns and the stage 1 `raw`
/// buffer.
///
/// # Complexity
///
/// As [`delta_heads_cols_vecs`].
#[allow(
    clippy::too_many_arguments,
    reason = "the layer state, geometry, parameters, columns and snapshot are the step's input"
)]
fn delta_heads_q32(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    fa: &FlatAffine,
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
    chunked: bool,
) -> Result<(Vec<Vec<i64>>, Vec<i64>), StepError> {
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let cdim = 2 * key_dim + val_dim;
    let k = fa.k;
    let slices = layer.slices;
    let w = g.head_v / slices;
    let n_tasks = g.n_v_heads * slices;
    let hk = g.head_k;
    let nv = g.n_v_heads;
    let head_k_fixed = i64::try_from(g.head_k)
        .ok()
        .and_then(|v| v.checked_shl(32))
        .ok_or(StepError::Overflow)?;
    let q_scale = fixed_rsqrt(head_k_fixed).map_err(|_| StepError::Overflow)?;
    let has_snap = snapshot.is_some();
    let snap = snapshot_or(snapshot, layer);
    // Stage 0, per key head: every column's `q̃` then `k̂`, `qk[kh·k·2hk + c·2hk ..]`, computed once
    // and shared by every value head and slice that reads that key head.
    let mut qk = vec![0i64; k * 2 * key_dim];
    let norm_qk = NormQkQ32Ctx {
        fa,
        key_dim,
        cdim,
        hk,
        eps,
        q_scale,
    };
    if let Some((_, e)) = attn_heads(g.n_k_heads, k * 2 * hk, &mut qk, parallel, &norm_qk) {
        return Err(e);
    }
    // Stage 0.5, per value head and column: β = σ(b_h) and α = exp(ssm_a_h·softplus(a_h + dt_h))
    // read only the column and the head, never the value slice, so compute them once for all
    // `slices` slice tasks (`bd[h·k·2 + c·2]`, `bd[h·k·2 + c·2 + 1]`). Every fixed_* refusal maps
    // to Overflow, exactly as it did inside stage 1, but now before any state update.
    let mut bd = vec![0i64; g.n_v_heads * k * 2];
    let head_gates = GatesQ32Ctx { fa, p, nv };
    if let Some((_, e)) = attn_heads(g.n_v_heads, k * 2, &mut bd, parallel, &head_gates) {
        return Err(e);
    }
    // Stage 1, per (value head, slice): the slice's update through all k columns in token order,
    // its state hot in cache; `raw[t·k·w + c·w ..]`. The rollback copy is taken inside the task.
    let mut raw = vec![0i64; k * val_dim];
    let update = UpdateQ32Ctx {
        layer,
        g,
        fa,
        snap,
        has_snap,
        qk: &qk,
        bd: &bd,
        chunked,
    };
    if let Some((_, e)) = attn_heads(n_tasks, k * w, &mut raw, parallel, &update) {
        return Err(e);
    }
    // Stage 2, per value head: reassemble each column's `o_h` from its slices, then the gated
    // norm. Per head, the k columns' outputs side by side: `out[h·k·head_v + c·head_v + j]`.
    let mut out = vec![0i64; k * val_dim];
    let gate_norm = GateNormQ32Ctx {
        fa,
        p,
        head_v: g.head_v,
        val_dim,
        slices,
        raw: &raw,
        eps,
    };
    if let Some((_, e)) = attn_heads(g.n_v_heads, k * g.head_v, &mut out, parallel, &gate_norm) {
        return Err(e);
    }
    let mut cols_out = alloc_cols(k, val_dim);
    let mut h = 0;
    while h < g.n_v_heads {
        let mut c = 0;
        while c < k {
            let at = h * k * g.head_v + c * g.head_v;
            cols_out[c].extend_from_slice(&out[at..at + g.head_v]);
            c += 1;
        }
        h += 1;
    }
    Ok((cols_out, raw))
}

/// [`delta_heads_vecs_impl`] over the flat inputs.
#[allow(
    clippy::too_many_arguments,
    reason = "the layer state, geometry, parameters, columns, snapshot and carrier are the step's input"
)]
fn delta_heads_flat(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    fa: &FlatAffine,
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
    chunked: bool,
    carrier: Carrier,
) -> Result<(Vec<Vec<i64>>, Vec<i64>), StepError> {
    if fa.bad {
        return Err(StepError::BadConfig(String::from("affine layer inputs")));
    }
    let n_tasks = g.n_v_heads * layer.slices;
    let snapshot_bad = match snapshot {
        Some(s) => s.heads.len() != n_tasks,
        None => false,
    };
    if layer.heads.len() != n_tasks || snapshot_bad {
        return Err(StepError::BadConfig(String::from("affine layer heads")));
    }
    if carrier == Carrier::F32 {
        return delta_heads_f32(layer, g, p, fa, eps, parallel, snapshot);
    }
    delta_heads_q32(layer, g, p, fa, eps, parallel, snapshot, chunked)
}

/// The three-stage affine heads over owned columns: [`flatten_affine_vecs`] then
/// [`delta_heads_flat`].
///
/// # Complexity
///
/// As [`delta_heads_cols_vecs`], plus the `k` flat column copies of [`flatten_affine_vecs`].
#[allow(
    clippy::too_many_arguments,
    reason = "the layer state, geometry, parameters, the four column inputs, snapshot and carrier \
              are the step's input"
)]
fn delta_heads_vecs_impl(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    conv_cols: &[Vec<i64>],
    z_cols: &[Vec<i64>],
    b_cols: &[Vec<i64>],
    a_cols: &[Vec<i64>],
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
    chunked: bool,
    carrier: Carrier,
) -> Result<(Vec<Vec<i64>>, Vec<i64>), StepError> {
    let key_dim = g.n_k_heads * g.head_k;
    let val_dim = g.n_v_heads * g.head_v;
    let fa = flatten_affine_vecs(
        conv_cols,
        z_cols,
        b_cols,
        a_cols,
        2 * key_dim + val_dim,
        val_dim,
        g.n_v_heads,
    );
    delta_heads_flat(layer, g, p, &fa, eps, parallel, snapshot, chunked, carrier)
}

/// [`delta_heads_vecs_impl`] over borrowed columns.
#[cfg(test)]
#[allow(
    clippy::too_many_arguments,
    reason = "the layer state, geometry, parameters, columns, snapshot and carrier are the step's input"
)]
fn delta_heads_cols_impl(
    layer: &AffineLayer,
    g: &AffineGeom,
    p: &AffineVectors,
    cols: &[AffineCol<'_>],
    eps: i64,
    parallel: bool,
    snapshot: Option<&AffineLayer>,
    chunked: bool,
    carrier: Carrier,
) -> Result<(Vec<Vec<i64>>, Vec<i64>), StepError> {
    let [c, z, b, a] = split_cols(cols);
    delta_heads_vecs_impl(
        layer, g, p, &c, &z, &b, &a, eps, parallel, snapshot, chunked, carrier,
    )
}

impl AffineLayer {
    #[allow(
        clippy::similar_names,
        clippy::needless_range_loop,
        reason = "converts and packs Q32 conv weights into tap-major AVX2 format once per layer"
    )]
    pub(crate) fn ensure_packed_w(&self, w: &[i64], ch: usize, kernel: usize) {
        if kernel != 4 {
            return;
        }
        let mut packed = self.conv_w_packed.lock();
        if packed.len() == 4 * ch {
            return;
        }
        let mut scalar = self.conv_w_scalar.lock();

        let mut p = vec![F32::ZERO; 4 * ch];
        let mut s = vec![F32::ZERO; 4 * ch];

        for c in 0..ch {
            let base = c * 4;
            s[base] = f32_carrier::q32_to_f32(w[base]);
            s[base + 1] = f32_carrier::q32_to_f32(w[base + 1]);
            s[base + 2] = f32_carrier::q32_to_f32(w[base + 2]);
            s[base + 3] = f32_carrier::q32_to_f32(w[base + 3]);
        }

        for b in 0..ch / 8 {
            for lane in 0..8 {
                let c = b * 8 + lane;
                let base = c * 4;
                p[b * 32 + lane] = s[base];
                p[b * 32 + 8 + lane] = s[base + 1];
                p[b * 32 + 16 + lane] = s[base + 2];
                p[b * 32 + 24 + lane] = s[base + 3];
            }
        }

        *packed = p;
        *scalar = s;
    }

    /// The rollback point's conv half: `src`'s window after one more column `x0`, i.e. its rows
    /// `1..` followed by `x0`, written in place.
    pub(crate) fn snapshot_conv_after(&mut self, src: &AffineLayer, x0: &[i64]) {
        let ch = x0.len();
        match (&mut self.conv, &src.conv) {
            (ConvState::Q32(dst), ConvState::Q32(src_conv)) => {
                dst.clear();
                if src_conv.len() >= ch && ch > 0 {
                    dst.extend_from_slice(&src_conv[ch..]);
                    dst.extend_from_slice(x0);
                }
            }
            (ConvState::F32(dst), ConvState::F32(src_conv)) => {
                dst.clear();
                if src_conv.len() >= ch && ch > 0 {
                    dst.extend_from_slice(&src_conv[ch..]);
                    dst.reserve(ch);
                    for &x in x0 {
                        dst.push(f32_carrier::q32_to_f32(x));
                    }
                }
            }
            _ => panic!("mismatched conv states in snapshot_conv_after"),
        }
    }
}

#[cfg(test)]
mod tests;
