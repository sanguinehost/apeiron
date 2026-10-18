//! `Q32.32` forward pass of ONE `Qwen2` decoder layer, returning every intermediate the
//! fold-transfer reads: port of `parts`, `attn_a`, `rms` and `rope_tables` in
//! `/home/socol/oc-tmp/lanes/port/spec/fold_transfer.py` (`rope_tables:60-64`, `rot:66-68`,
//! `rms:70-71`, `attn_a:73-81`, `parts:83-92`).
//!
//! ```text
//! xa = rms(h, ln1); q = xa·qᵀ + qb; k/v likewise          # [t, hidden] → [t, head widths]
//! a  = attn_a(q, k, v)                   # rotate-half RoPE, GQA repeat, causal softmax
//! mid = h + a·oᵀ; xf = rms(mid, ln2)
//! m   = silu(xf·gateᵀ) · (xf·upᵀ); out = mid + m·downᵀ
//! ```
//!
//! Every weight is row-major `[out, in]` as the transfer-layer dictionary holds it. With
//! `inv[i] = theta^(-2i/head_dim)` and `emb[t, j] = t · inv[j mod (head_dim/2)]`, the half-rotation
//! is `x ↦ x·cos(emb) + rot(x)·sin(emb)`, `rot(x) = cat(-x[upper], x[lower])`.
//!
//! The projections are [`gemm_nt`] on the grid, each operand's exact `2⁻ᵏ` panel shift keeping it
//! inside the primitive's `2³⁶` bound and cancelling on the read-back. Norms, rotation, `SwiGLU` and
//! the residual adds are serial integer loops: `O(t·width)` is one pass over a projection's output,
//! so a pool dispatch does not pay.
//!
//! The causal attention is the one place where this module contracts on the engine kernel directly
//! ([`crate::engine::kern::limb`], [`LIMBS`] limbs per block): per kv head, per query head and per
//! band of [`BAND`] query positions one [`gemm_q32`] score product against the keys that band can
//! reach and one for the value mix. The bands are the kernel's output rows and the kernel owns the
//! pool, so the whole attention is one serial loop of top-level kernel calls: two nested dispatches
//! would share the pool's single job slot, which has no re-entrancy guard. A band's product covers
//! the masked `j > i` entries too, `BAND/t` of the triangular population.
//!
//! Deviations: the reference (`torch.float64`, CUDA) contracts multiply-adds and evaluates
//! `exp`/`cos`/`sin` in a device library, so this carrier agrees to the grid quantum times the
//! operation's condition growth, not bit-for-bit; the attention products add the kernel's limb
//! truncation, bounded entrywise by the limb module's `G(n, limbs, max|a|, max|b|)`; refusals are
//! typed ([`PartsError`]). The transcendentals are the `crate::numeric` kernels ([`fixed_ln`],
//! [`fixed_exp`], [`fixed_sin_cos`]); the `RMSNorm` row is the engine's [`rms_norm_fixed`] with the
//! gain composed after it, and the attention scale's root is the floored [`isqrt`]. Their documented
//! bounds set this module's agreement (`fixed_sin_cos` is within `16·2⁻³²` absolute).
//!
//! Every value this module computes is a `Q32.32` word, the rotation tables included. The only
//! real-arithmetic input is the [`LayerCfg`] pair `eps`/`theta`, the model configuration's own
//! payload: [`grid1`] decodes each once per call and nothing downstream sees a real.
//!
//! Grade `tested` (`tests/learn_fold_parts.rs`, `tests/learn_fold_attn_speed.rs`); no Lean
//! refinement links this file, and the kernel limb bound is itself `tested` (`src/engine/kern/limb.rs`).
//! Trust boundary: the `Q32.32` arithmetic on `&[i64]` words and the [`LayerWords`] weight view only,
//! not the model loader, the test's real-arithmetic oracle, `torch`/CUDA codegen, or the rest of the
//! transfer; the AVX-VNNI/AVX2 cores the attention products run on are not extracted.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the dense kernels index the row-major buffers by `r/i/j/k/o/d/h`, name the head \
              geometry `nh`/`nkv`/`hd` as the reference does, and cast the position and dimension \
              counts into the exact `i128` accumulator domain"
)]

use crate::engine::kern::limb::{gemm_q32, LimbError};
use crate::engine::{rms_norm_fixed, NormError};
use crate::learn::solve::gemm::{gemm_nt, GemmError};
use crate::learn::solve::q32::{div_rne, ilog2_pos, narrow, rshift_rne, to_grid, GRID_BOUND};
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_ln::{fixed_ln, ONE};
use crate::numeric::fixed_rsqrt::isqrt;
use crate::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians};

/// `round(2*pi * 2^32)`, the rounded `2*pi` word the trig kernel carries privately; the kernel
/// exports its turn entry ([`turn_to_radians`]) rather than the constant, so the radian angle is
/// converted to a turn phase here before the series.
const TWO_PI_Q32: i64 = 26_986_075_409;

/// `a*b` in `Q32.32`, rounded once to nearest, ties to even; `None` off the grid.
///
/// The product is a `Q64.64` word and this is [`round_q32_even`]'s value, written branchlessly:
/// with `q = p >> 32` the floor quotient and `r = p & (2³²−1)` the non-negative floor remainder,
/// round-half-to-even is `q + [r > 2³¹ ∨ (r = 2³¹ ∧ q odd)]`. The truncating form rounds a tie
/// away from zero — `q + 1` for `p ≥ 0`, `q` for `p < 0` — and `q + 1` is the even neighbour
/// exactly when `q` is odd, so both forms choose the same word; the branchless form removes the
/// sign-dependent branches the product's sign makes unpredictable. The `i128` product of two
/// words is below `2¹²⁶`; [`narrow`] keeps the off-grid refusal.
pub(crate) fn mul(a: i64, b: i64) -> Option<i64> {
    let p = i128::from(a) * i128::from(b);
    let q = p >> 32;
    let r = p & 0xffff_ffff;
    let half = 1_i128 << 31;
    let bump = i128::from(r > half || (r == half && (q & 1) != 0));
    narrow(q + bump)
}

/// `a/b` in `Q32.32`, rounded once by [`div_rne`]; `None` on a non-positive divisor or a quotient
/// off the grid.
fn div(a: i64, b: i64) -> Option<i64> {
    div_rne(i128::from(a) << 32, i128::from(b)).and_then(narrow)
}

/// `sqrt x` for `x >= 0`: the floored integer root of `x*2^32` from the numeric root kernel, at
/// most one grid quantum below the true root. `None` for `x < 0`. `O(1)`.
pub(crate) fn sqrt(x: i64) -> Option<i64> {
    if x < 0 {
        return None;
    }
    narrow(isqrt(i128::from(x) << 32))
}

/// `e^x` with the top clamped at `e^21`; below the kernel's `-16` lower edge the value is flushed to
/// zero, where `e^-16 < 2^-23` is below the grid quantum of the logistic it feeds.
fn exp_sat(x: i64) -> Option<i64> {
    let x = if x < 21 * ONE { x } else { 21 * ONE };
    match fixed_exp(x) {
        Ok(v) => Some(v),
        Err(_) if x < -16 * ONE => Some(0),
        Err(_) => None,
    }
}

/// `(sin t, cos t)` from the numeric kernel at an angle `t` reduced modulo one turn, so every
/// position reaches the total turn entry point inside the kernel's `+-4096` radian window. `O(1)`.
pub(crate) fn sin_cos(t: i64) -> Option<(i64, i64)> {
    let turns = narrow(div_rne(i128::from(t) << 32, i128::from(TWO_PI_Q32))?)?;
    let phase = (turns as u64) << 32;
    match fixed_sin_cos(turn_to_radians(phase)) {
        Ok(v) => Some((v.sin, v.cos)),
        Err(_) => None,
    }
}

/// The geometry of one `Qwen2` decoder layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerCfg {
    /// Query head count.
    pub n_head: usize,
    /// Key/value head count; `n_head % n_kv_head == 0` is required.
    pub n_kv_head: usize,
    /// Per-head width; even.
    pub head_dim: usize,
    /// Residual width; `hidden == n_head * head_dim` is required.
    pub hidden: usize,
    /// Feed-forward width.
    pub inter: usize,
    /// `RMSNorm` epsilon, the configuration payload; [`grid1`] decodes it once per call.
    pub eps: f64,
    /// `RoPE` base, the configuration payload; finite and positive, decoded by [`grid1`].
    pub theta: f64,
}

impl LayerCfg {
    /// Refuse an invalid head geometry or a non-finite `eps`/`theta`. `O(1)`.
    fn validate(&self) -> Result<(), PartsError> {
        if !self.eps.is_finite() || !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(PartsError::NonFinite);
        }
        if self.n_head == 0
            || self.n_kv_head == 0
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
        {
            return Err(PartsError::Shape);
        }
        let wide = self
            .n_head
            .checked_mul(self.head_dim)
            .ok_or(PartsError::Shape)?;
        if wide != self.hidden || !self.n_head.is_multiple_of(self.n_kv_head) {
            return Err(PartsError::Shape);
        }
        Ok(())
    }
}

/// A refusal from the layer forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartsError {
    /// A slice length disagrees with the declared shape, a dimension product left `usize`, or the
    /// head geometry is invalid.
    Shape,
    /// `eps` is not finite, or `theta` is not a finite positive value.
    NonFinite,
    /// The sequence length `t` exceeds the `t_max` the [`RopeTables`] were built for.
    RopeTooShort,
    /// A value is outside the `Q32.32` grid or a rounded result left it, including the `gemm` bound
    /// refusal. The entry points refuse instead of saturating.
    OutOfRange,
}

/// Borrowed `Q32.32` views of the twelve weight tensors of one `Qwen2` decoder layer, each
/// row-major as the transfer dictionary holds it.
#[derive(Clone, Copy, Debug)]
pub struct LayerWords<'a> {
    /// `input_layernorm.weight`, length `hidden`.
    pub ln1: &'a [i64],
    /// `post_attention_layernorm.weight`, length `hidden`.
    pub ln2: &'a [i64],
    /// `q_proj.weight`, `[n_head·head_dim, hidden]`.
    pub q: &'a [i64],
    /// `q_proj.bias`, length `n_head·head_dim`.
    pub qb: &'a [i64],
    /// `k_proj.weight`, `[n_kv_head·head_dim, hidden]`.
    pub k: &'a [i64],
    /// `k_proj.bias`, length `n_kv_head·head_dim`.
    pub kb: &'a [i64],
    /// `v_proj.weight`, `[n_kv_head·head_dim, hidden]`.
    pub v: &'a [i64],
    /// `v_proj.bias`, length `n_kv_head·head_dim`.
    pub vb: &'a [i64],
    /// `o_proj.weight`, `[hidden, n_head·head_dim]`.
    pub o: &'a [i64],
    /// `gate_proj.weight`, `[inter, hidden]`.
    pub gate: &'a [i64],
    /// `up_proj.weight`, `[inter, hidden]`.
    pub up: &'a [i64],
    /// `down_proj.weight`, `[hidden, inter]`.
    pub down: &'a [i64],
}

/// The half-rotation tables [`attn_a`] applies to `q` and `k`, each row-major `[t_max, head_dim]` of
/// `Q32.32` words straight from [`fixed_sin_cos`]: the forward reads the table with no real-valued
/// copy beside it, so no entry is re-quantised per token.
#[derive(Clone, Debug)]
pub struct RopeTables {
    /// `cos(emb)`, row-major `[t_max, head_dim]`.
    pub cos: Vec<i64>,
    /// `sin(emb)`, row-major `[t_max, head_dim]`.
    pub sin: Vec<i64>,
    head_dim: usize,
    t_max: usize,
}

impl RopeTables {
    /// Build the tables for `t_max` positions. `O(t_max·head_dim)` time and space.
    /// # Errors
    /// [`PartsError::Shape`] on an invalid geometry; [`PartsError::NonFinite`] for a non-finite
    /// `eps`/`theta`; [`PartsError::OutOfRange`] for a `theta` outside the numeric `exp` kernel's
    /// `[e^-21, e^21]` window, or an entry outside the grid.
    pub fn new(cfg: &LayerCfg, t_max: usize) -> Result<Self, PartsError> {
        cfg.validate()?;
        let (hd, half) = (cfg.head_dim, cfg.head_dim / 2);
        let n = t_max.checked_mul(hd).ok_or(PartsError::Shape)?;
        let ln_theta = fixed_ln(grid1(cfg.theta)?).map_err(|_| PartsError::OutOfRange)?;
        let mut inv = vec![0_i64; half];
        for (i, slot) in inv.iter_mut().enumerate() {
            // theta^(-2i/head_dim) = exp(-(2i/head_dim)·ln theta)
            let e = i128::from(2 * i as i64) << 32;
            let frac = div_rne(-e, i128::try_from(hd).map_err(|_| PartsError::Shape)?)
                .and_then(narrow)
                .ok_or(PartsError::OutOfRange)?;
            *slot = fixed_exp(mul(ln_theta, frac).ok_or(PartsError::OutOfRange)?)
                .map_err(|_| PartsError::OutOfRange)?;
        }
        let (mut cos, mut sin) = (vec![0_i64; n], vec![0_i64; n]);
        for p in 0..t_max {
            let pos = i64::try_from(p).map_err(|_| PartsError::Shape)? << 32;
            for j in 0..hd {
                let emb = mul(pos, inv[j % half]).ok_or(PartsError::OutOfRange)?;
                let (s, c) = sin_cos(emb).ok_or(PartsError::OutOfRange)?;
                (cos[p * hd + j], sin[p * hd + j]) = (c, s);
            }
        }
        Ok(Self {
            cos,
            sin,
            head_dim: hd,
            t_max,
        })
    }

    /// The head width the tables were built for.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The position count the tables cover.
    #[must_use]
    pub fn t_max(&self) -> usize {
        self.t_max
    }
}

/// Every intermediate of one decoder layer, each row-major `[t, width]` as `Q32.32` words.
#[derive(Clone, Debug, PartialEq)]
pub struct Parts {
    /// `rms(h, ln1)`, `[t, hidden]`.
    pub xa: Vec<i64>,
    /// Query projection, `[t, n_head·head_dim]`.
    pub q: Vec<i64>,
    /// Key projection, `[t, n_kv_head·head_dim]`.
    pub k: Vec<i64>,
    /// Value projection, `[t, n_kv_head·head_dim]`.
    pub v: Vec<i64>,
    /// Causal GQA attention output, `[t, n_head·head_dim]`.
    pub a: Vec<i64>,
    /// Residual after attention, `[t, hidden]`.
    pub mid: Vec<i64>,
    /// `rms(mid, ln2)`, `[t, hidden]`.
    pub xf: Vec<i64>,
    /// Gate projection, `[t, inter]`.
    pub g: Vec<i64>,
    /// Up projection, `[t, inter]`.
    pub u: Vec<i64>,
    /// `silu(g)·u`, `[t, inter]`.
    pub m: Vec<i64>,
    /// Layer output, `[t, hidden]`.
    pub out: Vec<i64>,
}

/// `a·b`, refused when the product leaves `usize`.
fn area(a: usize, b: usize) -> Result<usize, PartsError> {
    a.checked_mul(b).ok_or(PartsError::Shape)
}

/// One `f64` configuration value as a `Q32.32` word: the model loader's single decode edge, refusing
/// an off-grid value rather than clamping it.
fn grid1(x: f64) -> Result<i64, PartsError> {
    to_grid(x, GRID_BOUND).ok_or(PartsError::OutOfRange)
}

/// The least `k ≥ 0` with `max·2⁻ᵏ < 2³⁵` raw: one bit below the `gemm`/`wsyrk` bound `2³⁶`.
fn panel_shift(max_mag: u128) -> u32 {
    match ilog2_pos(max_mag) {
        Some(b) => (b + 1).saturating_sub(35),
        None => 0,
    }
}

/// A power-of-two shift of a grid slice: right by `k > 0` (round-to-nearest, ties to even), an
/// exact checked left shift by `−k`, the identity at `0`. `None` when a result leaves `i64`;
/// `O(len)`.
pub(crate) fn shift(a: &mut [i64], k: i32) -> Option<()> {
    if k == 0 {
        return Some(());
    }
    // Flag form: the fragment has no `?` inside a loop, and a `for v in a` over `&mut [i64]`
    // reaches `IntoIterator`'s `into_iter`, which the model does not carry. The first refused entry
    // stops the shift and leaves every later entry untouched, as the early return did.
    let mut ok = true;
    for i in 0..a.len() {
        if ok {
            let v = a[i];
            let shifted = if k > 0 {
                // `k > 0`: the `u32` shift amount selects the model's wrapping shift.
                match rshift_rne(i128::from(v), k as u32) {
                    Some(r) => narrow(r),
                    None => None,
                }
            } else {
                // `k < 0`: `-k` is exact in `i64` (`i32::MIN` included) and fits `u32`; the
                // 128-bit shift wraps the value bits exactly as `checked_shl` did, and `narrow`
                // keeps the off-grid refusal.
                let s = (0i64 - i64::from(k)) as u32;
                if s >= 128 {
                    None
                } else {
                    narrow(i128::from(v) << s)
                }
            };
            match shifted {
                Some(x) => a[i] = x,
                None => ok = false,
            }
        }
    }
    if ok {
        Some(())
    } else {
        None
    }
}

/// `v` brought inside the `gemm` input bound: the exact `2⁻ᵏ` shift from [`panel_shift`], and `k`
/// to undo on the product. `O(len)` time and space.
pub(crate) fn panel(v: &[i64]) -> Option<(Vec<i64>, i32)> {
    // Serial maximum without the closure the fragment does not model; `max` starts at the identity.
    let mut max: u128 = 0;
    for &x in v {
        let y = i128::from(x);
        let m = if y < 0 { -y } else { y };
        let m = m as u128;
        if m > max {
            max = m;
        }
    }
    let k = panel_shift(max) as i32;
    let mut out = v.to_vec();
    shift(&mut out, k)?;
    Some((out, k))
}

/// [`GemmError`] as a [`PartsError`]: a shape or zero dimension is [`PartsError::Shape`], a bound
/// or overflow refusal is [`PartsError::OutOfRange`].
pub(crate) fn map_gemm(e: GemmError) -> PartsError {
    match e {
        GemmError::ZeroDim | GemmError::ShapeMismatch => PartsError::Shape,
        GemmError::InputBound | GemmError::Overflow => PartsError::OutOfRange,
    }
}

/// Limb-count of every attention product: the 3-limb default (24-bit mantissa per block).
const LIMBS: u32 = 3;

/// [`LimbError`] onto [`PartsError`]: `NoKernel` is not a fallback here and reads as a range
/// refusal, never a silent scalar path.
fn map_limb(e: LimbError) -> PartsError {
    match e {
        LimbError::Shape => PartsError::Shape,
        LimbError::Range | LimbError::NoKernel => PartsError::OutOfRange,
    }
}

/// Row-major dense projection `out[r,o] = bias[o] + Σ_i w[o,i]·x[r,i]` on `Q32.32` words: the two
/// shifted panels through [`gemm_nt`], the shift undone once.
/// # Errors
/// [`PartsError::Shape`] on a length disagreement; [`PartsError::OutOfRange`] when a value or a
/// rounded entry leaves the grid.
/// # Complexity
/// Time `O(t·out_w·in_w)` multiply-adds pooled over rows inside [`gemm_nt`]; space `O(t·out_w +
/// t·in_w + out_w·in_w)`.
pub fn linear(
    x: &[i64],
    t: usize,
    in_w: usize,
    w: &[i64],
    bias: Option<&[i64]>,
    out_w: usize,
) -> Result<Vec<i64>, PartsError> {
    if x.len() != area(t, in_w)? || w.len() != area(out_w, in_w)? {
        return Err(PartsError::Shape);
    }
    if let Some(b) = bias {
        if b.len() != out_w {
            return Err(PartsError::Shape);
        }
    }
    let (xs, kx) = panel(x).ok_or(PartsError::OutOfRange)?;
    let (ws, kw) = panel(w).ok_or(PartsError::OutOfRange)?;
    let mut out = gemm_nt(&xs, &ws, t, in_w, out_w).map_err(map_gemm)?;
    shift(&mut out, -(kx + kw)).ok_or(PartsError::OutOfRange)?;
    if let Some(b) = bias {
        // Indexed rather than `zip(b.iter().cycle())`: the fragment models no `Cycle` adaptor. The
        // bias of output column `i % out_w` is the one the cycle visits at that position.
        let mut err = false;
        for (i, slot) in out.iter_mut().enumerate() {
            if !err {
                match slot.checked_add(b[i % out_w]) {
                    Some(v) => *slot = v,
                    None => err = true,
                }
            }
        }
        if err {
            return Err(PartsError::OutOfRange);
        }
    }
    Ok(out)
}

/// [`NormError`] as a [`PartsError`]: an empty row a shape refusal, anything else a range refusal.
fn map_norm(e: NormError) -> PartsError {
    match e {
        NormError::EmptyRow => PartsError::Shape,
        NormError::OutOfRange => PartsError::OutOfRange,
    }
}

/// `RMSNorm` over rows: `out[r,j] = w[j]·x[r,j]/sqrt(mean_j(x[r,:]²) + eps)`, `eps` a word. Each
/// row is the engine's [`rms_norm_fixed`] (one `Q32.32` mean, reciprocal square root and product
/// per entry); the per-channel gain composes as one further rounded product. The row operation is
/// the engine body's: the two differ only in rounding the mean and the reciprocal root, each once
/// per row, so agreement is `O(q)` per row (asserted in `tests/learn_fold_parts.rs`).
/// # Errors
/// [`PartsError::Shape`] when a length disagrees with `t`×`width` or `width == 0`;
/// [`PartsError::OutOfRange`] when a value or a rounded result leaves the grid or the engine root's
/// window.
/// # Complexity
/// Time `O(t·width)` multiply-adds plus one integer mean and root per row; space `O(t·width)`.
pub fn rms(x: &[i64], w: &[i64], eps: i64, t: usize, width: usize) -> Result<Vec<i64>, PartsError> {
    let n = area(t, width)?;
    if width == 0 || x.len() != n || w.len() != width {
        return Err(PartsError::Shape);
    }
    let mut out = vec![0_i64; n];
    // Error slot rather than `?` inside the row loop: the fragment has no early return there. The
    // first refusal stops the accumulation and leaves the later rows unwritten, as the `?` did.
    let mut err: Option<PartsError> = None;
    for r in 0..t {
        if err.is_none() {
            let row = &x[r * width..][..width];
            match rms_norm_fixed(row, eps) {
                Ok(norm) => {
                    for j in 0..width {
                        if err.is_none() {
                            match mul(w[j], norm[j]) {
                                Some(v) => out[r * width + j] = v,
                                None => err = Some(PartsError::OutOfRange),
                            }
                        }
                    }
                }
                Err(e) => err = Some(map_norm(e)),
            }
        }
    }
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// The `RoPE` half-rotation: `dst = src·cos(emb) + rot(src)·sin(emb)`, two rounded products and one
/// rounded sum per entry. `O(t·n_head·head_dim)`, serial as one pass over a projection's output.
pub(crate) fn rope_apply(
    dst: &mut [i64],
    src: &[i64],
    t: usize,
    n_head: usize,
    hd: usize,
    rope: &RopeTables,
) -> Result<(), PartsError> {
    let half = hd / 2;
    // Error slot rather than `?` inside the position loop: the fragment has no early return there.
    let mut err: Option<PartsError> = None;
    for p in 0..t {
        if err.is_none() {
            let cb = p * hd;
            for h in 0..n_head {
                if err.is_none() {
                    let base = p * n_head * hd + h * hd;
                    for d in 0..hd {
                        if err.is_none() {
                            let r = if d < half {
                                -src[base + d + half]
                            } else {
                                src[base + d - half]
                            };
                            match mul(src[base + d], rope.cos[cb + d]).and_then(|a| {
                                mul(r, rope.sin[cb + d]).and_then(|b| a.checked_add(b))
                            }) {
                                Some(v) => dst[base + d] = v,
                                None => err = Some(PartsError::OutOfRange),
                            }
                        }
                    }
                }
            }
        }
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Query positions per causal band: the kernel's output-row count for one score product, and the
/// triangular mask's overhead, which is `BAND/t` of the products the mask leaves unused.
///
/// The band is the dispatch unit because the kernel owns the pool: [`gemm_q8_0`] splits its own
/// output rows across `width_global()` participants, so a caller that is itself a pool task would
/// nest two dispatches into one job slot. [`crate::engine::chart::pool`] has no re-entrancy guard,
/// so [`attn_words`] runs serially and every product is one top-level kernel call.
///
/// `512` is the measured minimum of the ported body's unit cost at `t = 512` on the pooled kernel
/// (`0.230` ns per word multiply-add against `0.248` at 256, `0.280` at 128 and `0.459` at 64; five
/// alternating runs each, [`k5_paired`]): a band is `2·n_head` kernel dispatches, so at the
/// production length the dispatch count outweighs the square's masked `j > i` entries and one band
/// covering the sequence wins. It equals the whole sequence at `t ≤ 512` and bounds the extent of
/// every later band.
///
/// [`gemm_q8_0`]: crate::engine::kern::gemm::gemm_q8_0
const BAND: usize = 512;

/// [`attn_a`] on `Q32.32` words: the rotated `q`/`k`, then per kv head, per query head and per
/// causal band one [`gemm_q32`] score product and one [`gemm_q32`] value mix.
///
/// Every product runs on the kernel at [`LIMBS`] limbs, so an entry is the block-floating
/// approximation of the exact `i128` dot, not bit-equal to it: the score bound is
/// `G(head_dim, LIMBS, max|q|, max|k|)` words and the mix bound `G(t, LIMBS, max|e|, max|v|)`, the
/// limb module's entrywise bound. The softmax, its rounding, the causal mask and the division are
/// unchanged.
///
/// Grade `tested` for the values and `measured` for the cost: at `t = 512` production geometry
/// (`n_head 14`, `n_kv_head 2`, `head_dim 64`) two campaigns of five alternating release runs of
/// this port and of the pooled exact pre-port body it replaces give `0.228`/`0.240` and
/// `0.421`/`0.424` ns per word multiply-add (ratio `0.54`/`0.57`), on a width-24 pool
/// (`cargo test --release --lib k5_paired -- --ignored --nocapture`).
fn attn_words(
    cfg: &LayerCfg,
    rope: &RopeTables,
    q: &[i64],
    k: &[i64],
    v: &[i64],
    t: usize,
) -> Result<Vec<i64>, PartsError> {
    let (nh, nkv, hd) = (cfg.n_head, cfg.n_kv_head, cfg.head_dim);
    let qt = area(area(t, nh)?, hd)?;
    let kt = area(area(t, nkv)?, hd)?;
    if q.len() != qt || k.len() != kt || v.len() != kt {
        return Err(PartsError::Shape);
    }
    let root = sqrt(i64::try_from(hd).map_err(|_| PartsError::Shape)? << 32)
        .ok_or(PartsError::OutOfRange)?;
    let scale = div(ONE, root).ok_or(PartsError::OutOfRange)?;
    let (mut qr, mut kr) = (vec![0_i64; qt], vec![0_i64; kt]);
    rope_apply(&mut qr, q, t, nh, hd, rope)?;
    rope_apply(&mut kr, k, t, nkv, hd, rope)?;
    // Keys per kv head `[n_kv_head][t][head_dim]` and values per kv head transposed
    // `[n_kv_head][head_dim][t]`: a head's keys are one contiguous contraction plane and the mix
    // contracts over `t`.
    let (mut kh, mut vt) = (vec![0_i64; kt], vec![0_i64; kt]);
    for j in 0..t {
        for g in 0..nkv {
            for d in 0..hd {
                kh[(g * t + j) * hd + d] = kr[(j * nkv + g) * hd + d];
                vt[(g * hd + d) * t + j] = v[(j * nkv + g) * hd + d];
            }
        }
    }
    let rep = nh / nkv;
    let mut out = vec![0_i64; qt];
    // One head's queries `[t, head_dim]`, gathered from the interleaved `qr` rows so a band is a
    // contiguous operand.
    let mut qhead = vec![0_i64; t * hd];
    // Per-band scratch: `rows × extent` scores, the same shape of masked weights, the masked
    // values `[head_dim, extent]`, and one denominator per row.
    let mut e = vec![0_i64; BAND * t];
    let mut vtb = vec![0_i64; hd * t];
    let mut denom = vec![0_i64; BAND];
    for g in 0..nkv {
        let keys = &kh[g * t * hd..][..t * hd];
        let vals = &vt[g * hd * t..][..hd * t];
        for h in g * rep..(g + 1) * rep {
            for (i, slot) in qhead.chunks_mut(hd).enumerate() {
                slot.copy_from_slice(&qr[(i * nh + h) * hd..][..hd]);
            }
            let mut i0 = 0;
            while i0 < t {
                let rows = BAND.min(t - i0);
                let extent = i0 + rows;
                let mut scores = gemm_q32(
                    &qhead[i0 * hd..][..rows * hd],
                    rows,
                    &keys[..extent * hd],
                    extent,
                    hd,
                    LIMBS,
                )
                .map_err(map_limb)?;
                e[..rows * extent].fill(0);
                for r in 0..rows {
                    let i = i0 + r;
                    let row = &mut scores[r * extent..][..=i];
                    let mut row_max = i64::MIN;
                    for s in row.iter_mut() {
                        *s = mul(*s, scale).ok_or(PartsError::OutOfRange)?;
                        row_max = row_max.max(*s);
                    }
                    let mut sum: i64 = 0;
                    for (slot, &s) in e[r * extent..][..=i].iter_mut().zip(row.iter()) {
                        *slot = s
                            .checked_sub(row_max)
                            .and_then(exp_sat)
                            .ok_or(PartsError::OutOfRange)?;
                        sum = sum.checked_add(*slot).ok_or(PartsError::OutOfRange)?;
                    }
                    denom[r] = sum;
                }
                for (row, slot) in vtb.chunks_mut(extent).enumerate().take(hd) {
                    slot.copy_from_slice(&vals[row * t..][..extent]);
                }
                let mix = gemm_q32(
                    &e[..rows * extent],
                    rows,
                    &vtb[..hd * extent],
                    hd,
                    extent,
                    LIMBS,
                )
                .map_err(map_limb)?;
                for r in 0..rows {
                    let base = ((i0 + r) * nh + h) * hd;
                    for (o, &a) in out[base..][..hd].iter_mut().zip(&mix[r * hd..][..hd]) {
                        *o = div(a, denom[r]).ok_or(PartsError::OutOfRange)?;
                    }
                }
                i0 += rows;
            }
        }
    }
    Ok(out)
}

/// Causal grouped-query attention with rotate-half `RoPE` (`attn_a`, lines 73-81) in `Q32.32`. `q`
/// is row-major `[t, n_head·head_dim]`, `k` and `v` are `[t, n_kv_head·head_dim]`. Head `h` reads
/// kv head `h / (n_head / n_kv_head)`, the scale is `1/sqrt(head_dim)`, and query position `i`
/// attends to `j ≤ i`. Returns row-major `[t, n_head·head_dim]`.
/// # Errors
/// [`PartsError::Shape`] for an invalid geometry or a slice that is not the stated shape;
/// [`PartsError::NonFinite`] for non-finite `eps`/`theta`; [`PartsError::RopeTooShort`] if `t >
/// rope.t_max` or `rope.head_dim` differs from `cfg.head_dim`; [`PartsError::OutOfRange`] when a
/// value or a rounded result leaves the grid.
/// # Complexity
/// Time `O(t²·n_head·head_dim)` over the causal bands of [`BAND`] positions: each band is one
/// [`gemm_q32`] score product (`BAND × (i0+BAND) × head_dim`) and one value mix, each of
/// `limbs(limbs+1)/2 = 6` retained limb pairs at [`LIMBS`] limbs, plus the `O(t·(n_head +
/// n_kv_head)·head_dim)` rotation and transposes; space `O(t·(n_head + 2·n_kv_head)·head_dim +
/// BAND·t + head_dim·t)`.
pub fn attn_a(
    cfg: &LayerCfg,
    rope: &RopeTables,
    q: &[i64],
    k: &[i64],
    v: &[i64],
    t: usize,
) -> Result<Vec<i64>, PartsError> {
    cfg.validate()?;
    if rope.head_dim != cfg.head_dim {
        return Err(PartsError::Shape);
    }
    if t > rope.t_max {
        return Err(PartsError::RopeTooShort);
    }
    attn_words(cfg, rope, q, k, v, t)
}

/// `silu(g) = g/(1 + exp(−g))` on one `Q32.32` word; the exponent saturates at `e²¹`, so a large
/// negative `g` gives `g·(1+e²¹)⁻¹` with an error below `8·10⁻¹⁰·|g|`. `None` off the grid. `O(1)`.
#[must_use]
pub fn silu(g: i64) -> Option<i64> {
    let den = exp_sat(0i64.checked_sub(g)?)?.checked_add(ONE)?;
    div(g, den)
}

/// `silu(g)·u` on words: one `exp`, one division and one product, each rounded once. `O(1)`.
pub(crate) fn silu_mul(g: i64, u: i64) -> Option<i64> {
    mul(silu(g)?, u)
}

/// One `Qwen2` decoder layer on a whole sequence (`parts`, lines 83-91) in `Q32.32`: the two
/// pre-norms, causal GQA attention and the `SwiGLU` feed-forward, returning every intermediate. `h`
/// is row-major `[t, hidden]` in words and the twelve weight tensors are [`LayerWords`] words, so
/// nothing here re-quantises a weight. Every weight is checked against the geometry before any
/// arithmetic, so a malformed layer is refused without touching an index.
/// # Errors
/// [`PartsError::Shape`] for an invalid geometry or a slice that is not the stated shape;
/// [`PartsError::NonFinite`] for non-finite `eps`/`theta`; [`PartsError::RopeTooShort`] if `t >
/// rope.t_max`; [`PartsError::OutOfRange`] when a value, a rounded result or a projection operand
/// outside the shared `2³⁶` bound is refused.
/// # Complexity
/// Time `O(t·(hidden·n_head·head_dim + inter·hidden) + t²·n_head·head_dim)`: the six projections
/// are `t × out × in` triples pooled inside [`gemm_nt`], the norms and the `SwiGLU` product are
/// `O(t·width)` serial passes, and [`attn_a`] as documented; space `O(t·(hidden + head widths +
/// inter))` for [`Parts`] plus the rope scratch.
pub fn parts(
    cfg: &LayerCfg,
    rope: &RopeTables,
    layer: &LayerWords<'_>,
    h: &[i64],
    t: usize,
) -> Result<Parts, PartsError> {
    cfg.validate()?;
    if rope.head_dim != cfg.head_dim {
        return Err(PartsError::Shape);
    }
    if t > rope.t_max {
        return Err(PartsError::RopeTooShort);
    }
    let (hid, nh, hd, nkv, inter) = (
        cfg.hidden,
        cfg.n_head,
        cfg.head_dim,
        cfg.n_kv_head,
        cfg.inter,
    );
    let (qw, kw) = (area(nh, hd)?, area(nkv, hd)?);
    let shapes = h.len() == area(t, hid)?
        && layer.ln1.len() == hid
        && layer.ln2.len() == hid
        && layer.q.len() == area(qw, hid)?
        && layer.qb.len() == qw
        && layer.k.len() == area(kw, hid)?
        && layer.kb.len() == kw
        && layer.v.len() == area(kw, hid)?
        && layer.vb.len() == kw
        && layer.o.len() == area(hid, qw)?
        && layer.gate.len() == area(inter, hid)?
        && layer.up.len() == area(inter, hid)?
        && layer.down.len() == area(hid, inter)?;
    if !shapes {
        return Err(PartsError::Shape);
    }

    let eps = grid1(cfg.eps)?;
    let xa = rms(h, layer.ln1, eps, t, hid)?;
    let q = linear(&xa, t, hid, layer.q, Some(layer.qb), qw)?;
    let k = linear(&xa, t, hid, layer.k, Some(layer.kb), kw)?;
    let v = linear(&xa, t, hid, layer.v, Some(layer.vb), kw)?;
    let a = attn_words(cfg, rope, &q, &k, &v, t)?;

    let attn_out = linear(&a, t, qw, layer.o, None, hid)?;
    let mut mid = vec![0_i64; area(t, hid)?];
    for (slot, (&x, &y)) in mid.iter_mut().zip(h.iter().zip(&attn_out)) {
        *slot = x.checked_add(y).ok_or(PartsError::OutOfRange)?;
    }

    let xf = rms(&mid, layer.ln2, eps, t, hid)?;
    let g = linear(&xf, t, hid, layer.gate, None, inter)?;
    let u = linear(&xf, t, hid, layer.up, None, inter)?;
    let mut m = vec![0_i64; area(t, inter)?];
    for (slot, (&gv, &uv)) in m.iter_mut().zip(g.iter().zip(&u)) {
        *slot = silu_mul(gv, uv).ok_or(PartsError::OutOfRange)?;
    }

    let down_out = linear(&m, t, inter, layer.down, None, hid)?;
    let mut out = vec![0_i64; area(t, hid)?];
    for (slot, (&x, &y)) in out.iter_mut().zip(mid.iter().zip(&down_out)) {
        *slot = x.checked_add(y).ok_or(PartsError::OutOfRange)?;
    }

    Ok(Parts {
        xa,
        q,
        k,
        v,
        a,
        mid,
        xf,
        g,
        u,
        m,
        out,
    })
}

/// The K5 comparison body: the pre-port attention (`git show 310a5dcb:src/learn/fold/stats/parts.rs`,
/// `AttnRows::rows` and `attn_words`), verbatim, including its own pool call. It lives under
/// `cfg(test)` because its measurement must run on the crate's shipped pool
/// ([`crate::engine::chart::pool`], `pub(crate)`), which no integration test can reach; the
/// integration bench ([`tests/learn_fold_attn_speed.rs`]) can only time a serial reconstruction.
#[cfg(test)]
mod k5 {
    use super::{div, exp_sat, mul, rope_apply, LayerCfg, PartsError, RopeTables};
    use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
    use crate::engine::chart::task::ChunkTask;
    use crate::learn::solve::q32::{round_q32_even, to_grid, GRID_BOUND};
    use std::time::Instant;

    /// [`super::attn_words`]'s pre-port shape: one pooled pass over contiguous
    /// `(position, head)` rows, every score and every value accumulation an exact `i128` dot
    /// rounded once.
    struct AttnRows<'a> {
        cfg: &'a LayerCfg,
        qr: &'a [i64],
        kr: &'a [i64],
        v: &'a [i64],
        t: usize,
        scale: i64,
        first: &'a FirstErr<usize, PartsError>,
    }

    impl AttnRows<'_> {
        /// The rows of `piece`, each scoring keys `j ≤ i` and normalising by its softmax
        /// denominator. `O(rows·t·head_dim)` multiply-adds.
        fn rows(&self, start: usize, piece: &mut [i64]) -> Result<(), PartsError> {
            let (hd, nh, nkv) = (self.cfg.head_dim, self.cfg.n_head, self.cfg.n_kv_head);
            let (rep, row0) = (nh / nkv, start / hd);
            let mut scores = vec![0_i64; self.t];
            let mut acc = vec![0_i64; hd];
            for (r, out) in piece.chunks_mut(hd).enumerate() {
                let (i, h) = ((row0 + r) / nh, (row0 + r) % nh);
                let (gbase, qbase) = ((h / rep) * hd, i * nh * hd + h * hd);
                let qrow = &self.qr[qbase..][..hd];
                let mut row_max = i64::MIN;
                for (j, slot) in scores.iter_mut().enumerate().take(i + 1) {
                    let krow = &self.kr[j * nkv * hd + gbase..][..hd];
                    let mut dot: i128 = 0;
                    for d in 0..hd {
                        dot += i128::from(qrow[d]) * i128::from(krow[d]);
                    }
                    let word = round_q32_even(dot).map_err(|_| PartsError::OutOfRange)?;
                    *slot = mul(word, self.scale).ok_or(PartsError::OutOfRange)?;
                    row_max = row_max.max(*slot);
                }
                acc.fill(0);
                let mut denom: i64 = 0;
                for j in 0..=i {
                    let e = scores[j]
                        .checked_sub(row_max)
                        .and_then(exp_sat)
                        .ok_or(PartsError::OutOfRange)?;
                    denom = denom.checked_add(e).ok_or(PartsError::OutOfRange)?;
                    let vbase = j * nkv * hd + gbase;
                    for (a, &v) in acc.iter_mut().zip(&self.v[vbase..][..hd]) {
                        let t = mul(e, v).ok_or(PartsError::OutOfRange)?;
                        *a = a.checked_add(t).ok_or(PartsError::OutOfRange)?;
                    }
                }
                for (o, &a) in out.iter_mut().zip(&acc) {
                    *o = div(a, denom).ok_or(PartsError::OutOfRange)?;
                }
            }
            Ok(())
        }
    }

    impl ChunkTask<i64> for AttnRows<'_> {
        fn run(&self, c: usize, start: usize, piece: &mut [i64]) {
            if let Err(e) = self.rows(start, piece) {
                self.first.record(c, e);
            }
        }
    }

    /// The pre-port `attn_words`: the rotated `q`/`k`, one pooled pass over the row chunks.
    fn old_attn(
        cfg: &LayerCfg,
        rope: &RopeTables,
        q: &[i64],
        k: &[i64],
        v: &[i64],
        t: usize,
    ) -> Result<Vec<i64>, PartsError> {
        let (nh, nkv, hd) = (cfg.n_head, cfg.n_kv_head, cfg.head_dim);
        let qt = super::area(super::area(t, nh)?, hd)?;
        let kt = super::area(super::area(t, nkv)?, hd)?;
        if q.len() != qt || k.len() != kt || v.len() != kt {
            return Err(PartsError::Shape);
        }
        let root = super::sqrt(i64::try_from(hd).map_err(|_| PartsError::Shape)? << 32)
            .ok_or(PartsError::OutOfRange)?;
        let scale = div(super::ONE, root).ok_or(PartsError::OutOfRange)?;
        let (mut qr, mut kr) = (vec![0_i64; qt], vec![0_i64; kt]);
        rope_apply(&mut qr, q, t, nh, hd, rope)?;
        rope_apply(&mut kr, k, t, nkv, hd, rope)?;
        let mut out = vec![0_i64; qt];
        let first = FirstErr::new();
        let chunk = (t * nh).div_ceil(width_global()).max(1) * hd;
        chunks_mut_global(
            &mut out,
            chunk,
            &AttnRows {
                cfg,
                qr: &qr,
                kr: &kr,
                v,
                t,
                scale,
                first: &first,
            },
        );
        match first.take() {
            Some((_, e)) => Err(e),
            None => Ok(out),
        }
    }

    /// Production geometry and a seeded fixture for the pair.
    fn fixture(seed: u64, t: usize) -> (LayerCfg, RopeTables, Vec<i64>, Vec<i64>, Vec<i64>) {
        let cfg = LayerCfg {
            n_head: 14,
            n_kv_head: 2,
            head_dim: 64,
            hidden: 14 * 64,
            inter: 4864,
            eps: 1e-6,
            theta: 1e4,
        };
        let rope = RopeTables::new(&cfg, t).expect("rope");
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (s >> 11) as f64 / 9_007_199_254_740_992.0;
            to_grid(2.0 * u - 1.0, GRID_BOUND).expect("on grid")
        };
        let q = (0..t * 14 * 64).map(|_| next()).collect();
        let k = (0..t * 2 * 64).map(|_| next()).collect();
        let v = (0..t * 2 * 64).map(|_| next()).collect();
        (cfg, rope, q, k, v)
    }

    /// K5: the port against the pooled exact body it replaces, five alternating runs each at the
    /// production shape `t = 512`, mean and spread in ns per word multiply-add. Ignored so the
    /// default gate run stays fast: `cargo test --release --lib k5_paired -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement: release run, five alternating arms"]
    fn k5_paired() {
        let t = 512;
        let (cfg, rope, q, k, v) = fixture(0x5eed_0007, t);
        let n = (14 * 64 * t * (t + 1)) as f64;
        // Warm the pool and the kernel on both arms before the first timed pair.
        std::hint::black_box(super::attn_a(&cfg, &rope, &q, &k, &v, t).expect("port"));
        std::hint::black_box(old_attn(&cfg, &rope, &q, &k, &v, t).expect("body"));
        let (mut port, mut body) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            let t0 = Instant::now();
            let a = super::attn_a(&cfg, &rope, &q, &k, &v, t).expect("port");
            let dt = t0.elapsed().as_secs_f64();
            std::hint::black_box(&a);
            port.push(dt * 1e9 / n);

            let t0 = Instant::now();
            let b = old_attn(&cfg, &rope, &q, &k, &v, t).expect("body");
            let dt = t0.elapsed().as_secs_f64();
            std::hint::black_box(&b);
            body.push(dt * 1e9 / n);
        }
        let stats = |xs: &[f64]| {
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = xs.iter().copied().fold(0.0_f64, f64::max);
            (mean, hi - lo)
        };
        let (pm, ps) = stats(&port);
        let (bm, bs) = stats(&body);
        println!(
            "K5 t=512 width {} port {pm:.4} ns/MAC (spread {ps:.4}; {port:?})",
            width_global()
        );
        println!("K5 t=512 pooled body {bm:.4} ns/MAC (spread {bs:.4}; {body:?})");
        println!("K5 ratio port/body {:.3}", pm / bm);
    }
}
