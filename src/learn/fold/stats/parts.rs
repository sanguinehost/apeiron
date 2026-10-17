//! `Q32.32` forward pass of ONE `Qwen2` decoder layer, returning every intermediate the
//! fold-transfer reads: port of `parts`, `attn_a`, `rms` and `rope_tables` in
//! `~/lanes/port/spec/fold_transfer.py` (`rope_tables:60-64`, `rot:66-68`,
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
//! so a pool dispatch does not pay. The attention rows are the one pooled task, at
//! `O(t²·n_head·head_dim)`.
//!
//! Deviations: the reference (`torch.float64`, CUDA) contracts multiply-adds and evaluates
//! `exp`/`cos`/`sin` in a device library, so this carrier agrees to the grid quantum times the
//! operation's condition growth, not bit-for-bit; refusals are typed ([`PartsError`]). The
//! transcendentals are the `crate::numeric` kernels ([`fixed_ln`], [`fixed_exp`],
//! [`fixed_sin_cos`]) and the root is the floored [`isqrt`]; their documented bounds set this
//! module's agreement (`fixed_sin_cos` is within `16·2⁻³²` absolute).
//!
//! Grade `tested` (`tests/learn_fold_parts.rs`); no Lean refinement links this file. Trust boundary:
//! the `Q32.32` arithmetic on `&[f64]` inputs and the `f64` grid map only, not the model loader, the
//! `f64` oracle, `torch`/CUDA codegen, or the rest of the transfer.

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

use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::learn::solve::gemm::{gemm_nt, GemmError};
use crate::learn::solve::q32::{
    div_rne, from_grid, ilog2_pos, round_q32_even, rshift_rne, to_grid, GRID_BOUND,
};
use crate::numeric::fixed_exp::fixed_exp;
use crate::numeric::fixed_ln::{fixed_ln, ONE};
use crate::numeric::fixed_rsqrt::isqrt;
use crate::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians};

/// `round(2*pi * 2^32)`, the rounded `2*pi` word the trig kernel carries privately; the kernel
/// exports its turn entry ([`turn_to_radians`]) rather than the constant, so the radian angle is
/// converted to a turn phase here before the series.
const TWO_PI_Q32: i64 = 26_986_075_409;

/// `a*b` in `Q32.32`, rounded once by [`round_q32_even`] (ties to even); `None` off the grid.
pub(crate) fn mul(a: i64, b: i64) -> Option<i64> {
    round_q32_even(i128::from(a) * i128::from(b)).ok()
}

/// `a/b` in `Q32.32`, rounded once by [`div_rne`]; `None` on a non-positive divisor or a quotient
/// off the grid.
fn div(a: i64, b: i64) -> Option<i64> {
    div_rne(i128::from(a) << 32, i128::from(b)).and_then(narrow)
}

/// `v` as an `i64`, or `None` when it leaves the grid.
pub(crate) fn narrow(v: i128) -> Option<i64> {
    i64::try_from(v).ok()
}

/// `sqrt x` for `x >= 0`: the floored integer root of `x*2^32` from the numeric root kernel, at
/// most one grid quantum below the true root. `None` for `x < 0`. `O(1)`.
fn sqrt(x: i64) -> Option<i64> {
    if x < 0 {
        return None;
    }
    narrow(isqrt(i128::from(x) << 32))
}

/// `e^x` with the top clamped at `e^21`; below the kernel's `-16` lower edge the value is flushed to
/// zero, where `e^-16 < 2^-23` is below the grid quantum of the logistic it feeds.
fn exp_sat(x: i64) -> Option<i64> {
    let x = x.min(21 * ONE);
    match fixed_exp(x) {
        Ok(v) => Some(v),
        Err(_) if x < -16 * ONE => Some(0),
        Err(_) => None,
    }
}

/// `(sin t, cos t)` from the numeric kernel at an angle `t` reduced modulo one turn, so every
/// position reaches the total turn entry point inside the kernel's `+-4096` radian window. `O(1)`.
fn sin_cos(t: i64) -> Option<(i64, i64)> {
    let turns = narrow(div_rne(i128::from(t) << 32, i128::from(TWO_PI_Q32))?)?;
    let phase = (turns as u64) << 32;
    fixed_sin_cos(turn_to_radians(phase))
        .ok()
        .map(|v| (v.sin, v.cos))
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
    /// `RMSNorm` epsilon.
    pub eps: f64,
    /// `RoPE` base; finite and positive.
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

/// The twelve `f64` weight tensors of one `Qwen2` decoder layer, each row-major as the transfer
/// dictionary holds it.
#[derive(Clone, Copy, Debug)]
pub struct LayerF64<'a> {
    /// `input_layernorm.weight`, length `hidden`.
    pub ln1: &'a [f64],
    /// `post_attention_layernorm.weight`, length `hidden`.
    pub ln2: &'a [f64],
    /// `q_proj.weight`, `[n_head·head_dim, hidden]`.
    pub q: &'a [f64],
    /// `q_proj.bias`, length `n_head·head_dim`.
    pub qb: &'a [f64],
    /// `k_proj.weight`, `[n_kv_head·head_dim, hidden]`.
    pub k: &'a [f64],
    /// `k_proj.bias`, length `n_kv_head·head_dim`.
    pub kb: &'a [f64],
    /// `v_proj.weight`, `[n_kv_head·head_dim, hidden]`.
    pub v: &'a [f64],
    /// `v_proj.bias`, length `n_kv_head·head_dim`.
    pub vb: &'a [f64],
    /// `o_proj.weight`, `[hidden, n_head·head_dim]`.
    pub o: &'a [f64],
    /// `gate_proj.weight`, `[inter, hidden]`.
    pub gate: &'a [f64],
    /// `up_proj.weight`, `[inter, hidden]`.
    pub up: &'a [f64],
    /// `down_proj.weight`, `[hidden, inter]`.
    pub down: &'a [f64],
}

/// The half-rotation tables [`attn_a`] applies to `q` and `k`, each row-major `[t_max, head_dim]`.
/// The grid words the forward reads sit beside the `f64` copies, so no entry is re-quantised per
/// token.
#[derive(Clone, Debug)]
pub struct RopeTables {
    /// `cos(emb)`, row-major `[t_max, head_dim]`.
    pub cos: Vec<f64>,
    /// `sin(emb)`, row-major `[t_max, head_dim]`.
    pub sin: Vec<f64>,
    head_dim: usize,
    t_max: usize,
    cosq: Vec<i64>,
    sinq: Vec<i64>,
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
        let (mut cosq, mut sinq) = (vec![0_i64; n], vec![0_i64; n]);
        for p in 0..t_max {
            let pos = i64::try_from(p).map_err(|_| PartsError::Shape)? << 32;
            for j in 0..hd {
                let emb = mul(pos, inv[j % half]).ok_or(PartsError::OutOfRange)?;
                let (s, c) = sin_cos(emb).ok_or(PartsError::OutOfRange)?;
                (cosq[p * hd + j], sinq[p * hd + j]) = (c, s);
            }
        }
        let (cos, sin) = (words_grid(&cosq), words_grid(&sinq));
        Ok(Self {
            cos,
            sin,
            head_dim: hd,
            t_max,
            cosq,
            sinq,
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

/// Every intermediate of one decoder layer, each row-major `[t, width]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Parts {
    /// `rms(h, ln1)`, `[t, hidden]`.
    pub xa: Vec<f64>,
    /// Query projection, `[t, n_head·head_dim]`.
    pub q: Vec<f64>,
    /// Key projection, `[t, n_kv_head·head_dim]`.
    pub k: Vec<f64>,
    /// Value projection, `[t, n_kv_head·head_dim]`.
    pub v: Vec<f64>,
    /// Causal GQA attention output, `[t, n_head·head_dim]`.
    pub a: Vec<f64>,
    /// Residual after attention, `[t, hidden]`.
    pub mid: Vec<f64>,
    /// `rms(mid, ln2)`, `[t, hidden]`.
    pub xf: Vec<f64>,
    /// Gate projection, `[t, inter]`.
    pub g: Vec<f64>,
    /// Up projection, `[t, inter]`.
    pub u: Vec<f64>,
    /// `silu(g)·u`, `[t, inter]`.
    pub m: Vec<f64>,
    /// Layer output, `[t, hidden]`.
    pub out: Vec<f64>,
}

/// `a·b`, refused when the product leaves `usize`.
fn area(a: usize, b: usize) -> Result<usize, PartsError> {
    a.checked_mul(b).ok_or(PartsError::Shape)
}

/// One `f64` value as a `Q32.32` word; a value off the grid is refused, never clamped.
fn grid1(x: f64) -> Result<i64, PartsError> {
    to_grid(x, GRID_BOUND).ok_or(PartsError::OutOfRange)
}

/// A whole `f64` slice as `Q32.32` words. `O(len)` time and space.
fn grid_words(x: &[f64]) -> Result<Vec<i64>, PartsError> {
    x.iter().map(|&v| grid1(v)).collect()
}

/// A `Q32.32` slice back across the `f64` boundary. `O(len)` time and space.
pub(crate) fn words_grid(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| from_grid(v)).collect()
}

/// An exact `i128` `Q64` accumulator rounded once to a `Q32.32` word.
fn round64(acc: i128) -> Result<i64, PartsError> {
    round_q32_even(acc).map_err(|_| PartsError::OutOfRange)
}

/// The least `k ≥ 0` with `max·2⁻ᵏ < 2³⁵` raw: one bit below the `gemm`/`wsyrk` bound `2³⁶`.
fn panel_shift(max_mag: u128) -> u32 {
    ilog2_pos(max_mag).map_or(0, |b| (b + 1).saturating_sub(35))
}

/// A power-of-two shift of a grid slice: right by `k > 0` (round-to-nearest, ties to even), an
/// exact checked left shift by `−k`, the identity at `0`. `None` when a result leaves `i64`;
/// `O(len)`.
pub(crate) fn shift(a: &mut [i64], k: i32) -> Option<()> {
    if k == 0 {
        return Some(());
    }
    for v in a {
        *v = if k > 0 {
            narrow(rshift_rne(i128::from(*v), u32::try_from(k).ok()?)?)?
        } else {
            narrow(i128::from(*v).checked_shl(k.unsigned_abs())?)?
        };
    }
    Some(())
}

/// `v` brought inside the `gemm` input bound: the exact `2⁻ᵏ` shift from [`panel_shift`], and `k`
/// to undo on the product. `O(len)` time and space.
pub(crate) fn panel(v: &[i64]) -> Option<(Vec<i64>, i32)> {
    let max = v
        .iter()
        .fold(0_u128, |m, &x| m.max(i128::from(x).unsigned_abs()));
    let k = i32::try_from(panel_shift(max)).ok()?;
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

/// Row-major dense projection `out[r,o] = bias[o] + Σ_i w[o,i]·x[r,i]` at the `f64` boundary.
/// # Errors
/// [`PartsError::Shape`] on a length disagreement; [`PartsError::OutOfRange`] when a value or a
/// rounded entry leaves the grid.
/// # Complexity
/// Time `O(t·out_w·in_w)` multiply-adds pooled over rows inside [`gemm_nt`]; space `O(t·out_w +
/// t·in_w + out_w·in_w)`.
pub fn linear(
    x: &[f64],
    t: usize,
    in_w: usize,
    w: &[f64],
    bias: Option<&[f64]>,
    out_w: usize,
) -> Result<Vec<f64>, PartsError> {
    let bw = match bias {
        Some(b) => Some(grid_words(b)?),
        None => None,
    };
    let out = linear_words(
        &grid_words(x)?,
        t,
        in_w,
        &grid_words(w)?,
        bw.as_deref(),
        out_w,
    )?;
    Ok(words_grid(&out))
}

/// `linear` on `Q32.32` words: the two shifted panels through [`gemm_nt`], the shift undone once.
fn linear_words(
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
        for (slot, &bias) in out.iter_mut().zip(b.iter().cycle().take(t * out_w)) {
            *slot = slot.checked_add(bias).ok_or(PartsError::OutOfRange)?;
        }
    }
    Ok(out)
}

/// `RMSNorm` over rows: `out[r,j] = w[j]·x[r,j]/sqrt(mean_j(x[r,:]²) + eps)`. One serial pass per
/// row, an order below one projection.
/// # Errors
/// [`PartsError::Shape`] when a length disagrees with `t`×`width` or `width == 0`;
/// [`PartsError::NonFinite`] if `eps` is not finite; [`PartsError::OutOfRange`] when a value or a
/// rounded result leaves the grid.
/// # Complexity
/// Time `O(t·width)` multiply-adds plus `t` integer roots and reciprocals; space `O(t·width)`.
fn rms_words(
    x: &[i64],
    w: &[i64],
    eps: f64,
    t: usize,
    width: usize,
) -> Result<Vec<i64>, PartsError> {
    let n = area(t, width)?;
    if width == 0 || x.len() != n || w.len() != width {
        return Err(PartsError::Shape);
    }
    let eps = grid1(eps)?;
    let mut out = vec![0_i64; n];
    for r in 0..t {
        let row = &x[r * width..][..width];
        let mut ss: i128 = 0;
        for &v in row {
            ss += i128::from(v) * i128::from(v);
        }
        // mean = ss/(2³²·width) in Q32.32, one rounding, then + eps.
        let mean = div_rne(
            ss,
            i128::try_from(width).map_err(|_| PartsError::Shape)? << 32,
        )
        .and_then(narrow)
        .ok_or(PartsError::OutOfRange)?;
        let denom = sqrt(mean.checked_add(eps).ok_or(PartsError::OutOfRange)?)
            .and_then(|s| div(ONE, s))
            .ok_or(PartsError::OutOfRange)?;
        for (slot, (&v, &wj)) in out[r * width..][..width].iter_mut().zip(row.iter().zip(w)) {
            *slot = mul(wj, mul(v, denom).ok_or(PartsError::OutOfRange)?)
                .ok_or(PartsError::OutOfRange)?;
        }
    }
    Ok(out)
}

/// `RMSNorm` over rows at the `f64` boundary of [`rms_words`].
/// # Errors
/// As [`rms_words`].
/// # Complexity
/// As [`rms_words`], plus `O(t·width)` grid conversions.
pub fn rms(x: &[f64], w: &[f64], eps: f64, t: usize, width: usize) -> Result<Vec<f64>, PartsError> {
    if !eps.is_finite() {
        return Err(PartsError::NonFinite);
    }
    let out = rms_words(&grid_words(x)?, &grid_words(w)?, eps, t, width)?;
    Ok(words_grid(&out))
}

/// The `RoPE` half-rotation: `dst = src·cos(emb) + rot(src)·sin(emb)`, two rounded products and one
/// rounded sum per entry. `O(t·n_head·head_dim)`, serial as one pass over a projection's output.
fn rope_apply(
    dst: &mut [i64],
    src: &[i64],
    t: usize,
    n_head: usize,
    hd: usize,
    rope: &RopeTables,
) -> Result<(), PartsError> {
    let half = hd / 2;
    for p in 0..t {
        let cb = p * hd;
        for h in 0..n_head {
            let base = p * n_head * hd + h * hd;
            for d in 0..hd {
                let r = if d < half {
                    -src[base + d + half]
                } else {
                    src[base + d - half]
                };
                dst[base + d] = mul(src[base + d], rope.cosq[cb + d])
                    .and_then(|a| mul(r, rope.sinq[cb + d]).and_then(|b| a.checked_add(b)))
                    .ok_or(PartsError::OutOfRange)?;
            }
        }
    }
    Ok(())
}

/// One contiguous `(query position, head)` row chunk of [`attn_words`]: the row owns its scores and
/// its value accumulator, and reads only the rotated `q`/`k` and the values.
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
    /// The rows of `piece`, each scoring keys `j ≤ i` and normalising by its softmax denominator.
    fn rows(&self, start: usize, piece: &mut [i64]) -> Result<(), PartsError> {
        let (hd, nh, nkv) = (self.cfg.head_dim, self.cfg.n_head, self.cfg.n_kv_head);
        let (rep, row0) = (nh / nkv, start / hd);
        let mut scores = vec![0_i64; self.t];
        let mut acc = vec![0_i64; hd];
        for (r, out) in piece.chunks_mut(hd).enumerate() {
            let (i, h) = ((row0 + r) / nh, (row0 + r) % nh);
            let (gbase, qbase) = ((h / rep) * hd, i * nh * hd + h * hd);
            let mut row_max = i64::MIN;
            for (j, slot) in scores.iter_mut().enumerate().take(i + 1) {
                let kbase = j * nkv * hd + gbase;
                let mut dot: i128 = 0;
                for d in 0..hd {
                    dot += i128::from(self.qr[qbase + d]) * i128::from(self.kr[kbase + d]);
                }
                *slot = mul(round64(dot)?, self.scale).ok_or(PartsError::OutOfRange)?;
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

/// [`attn_a`] on `Q32.32` words: the rotated `q`/`k`, one pooled pass over the row chunks.
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
/// Time `O(t²·n_head·head_dim + t·(n_head + n_kv_head)·head_dim)` over `width_global()` contiguous
/// `(position, head)` chunks (each row scores `i+1` keys and updates `i+1` values); space
/// `O(t·(n_head + n_kv_head)·head_dim + t + head_dim)`.
pub fn attn_a(
    cfg: &LayerCfg,
    rope: &RopeTables,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    t: usize,
) -> Result<Vec<f64>, PartsError> {
    cfg.validate()?;
    if rope.head_dim != cfg.head_dim {
        return Err(PartsError::Shape);
    }
    if t > rope.t_max {
        return Err(PartsError::RopeTooShort);
    }
    let out = attn_words(
        cfg,
        rope,
        &grid_words(q)?,
        &grid_words(k)?,
        &grid_words(v)?,
        t,
    )?;
    Ok(words_grid(&out))
}

/// `silu(g) = g/(1 + exp(−g))` in `Q32.32`; the exponent saturates at `e²¹`, so a large negative
/// `g` gives `g·(1+e²¹)⁻¹` with an error below `8·10⁻¹⁰·|g|`.
#[must_use]
pub fn silu(g: f64) -> f64 {
    match grid1(g) {
        Ok(x) => silu_word(x).map_or(f64::NAN, from_grid),
        Err(_) => f64::NAN,
    }
}

/// [`silu`] on one `Q32.32` word. `O(1)`.
fn silu_word(g: i64) -> Option<i64> {
    let den = exp_sat(g.checked_neg()?)?.checked_add(ONE)?;
    div(g, den)
}

/// `silu(g)·u` on words: one `exp`, one division and one product, each rounded once. `O(1)`.
fn silu_mul(g: i64, u: i64) -> Option<i64> {
    mul(silu_word(g)?, u)
}

/// One `Qwen2` decoder layer on a whole sequence (`parts`, lines 83-91) in `Q32.32`: the two
/// pre-norms, causal GQA attention and the `SwiGLU` feed-forward, returning every intermediate. `h`
/// is row-major `[t, hidden]`; every weight is checked against the geometry before any arithmetic,
/// so a malformed layer is refused without touching an index.
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
    layer: &LayerF64,
    h: &[f64],
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

    let (hw, ln1, ln2) = (
        grid_words(h)?,
        grid_words(layer.ln1)?,
        grid_words(layer.ln2)?,
    );
    let xa = rms_words(&hw, &ln1, cfg.eps, t, hid)?;
    let q = linear_words(
        &xa,
        t,
        hid,
        &grid_words(layer.q)?,
        Some(&grid_words(layer.qb)?),
        qw,
    )?;
    let k = linear_words(
        &xa,
        t,
        hid,
        &grid_words(layer.k)?,
        Some(&grid_words(layer.kb)?),
        kw,
    )?;
    let v = linear_words(
        &xa,
        t,
        hid,
        &grid_words(layer.v)?,
        Some(&grid_words(layer.vb)?),
        kw,
    )?;
    let a = attn_words(cfg, rope, &q, &k, &v, t)?;

    let attn_out = linear_words(&a, t, qw, &grid_words(layer.o)?, None, hid)?;
    let mut mid = vec![0_i64; area(t, hid)?];
    for (slot, (&x, &y)) in mid.iter_mut().zip(hw.iter().zip(&attn_out)) {
        *slot = x.checked_add(y).ok_or(PartsError::OutOfRange)?;
    }

    let xf = rms_words(&mid, &ln2, cfg.eps, t, hid)?;
    let g = linear_words(&xf, t, hid, &grid_words(layer.gate)?, None, inter)?;
    let u = linear_words(&xf, t, hid, &grid_words(layer.up)?, None, inter)?;
    let mut m = vec![0_i64; area(t, inter)?];
    for (slot, (&gv, &uv)) in m.iter_mut().zip(g.iter().zip(&u)) {
        *slot = silu_mul(gv, uv).ok_or(PartsError::OutOfRange)?;
    }

    let down_out = linear_words(&m, t, inter, &grid_words(layer.down)?, None, hid)?;
    let mut out = vec![0_i64; area(t, hid)?];
    for (slot, (&x, &y)) in out.iter_mut().zip(mid.iter().zip(&down_out)) {
        *slot = x.checked_add(y).ok_or(PartsError::OutOfRange)?;
    }

    Ok(Parts {
        xa: words_grid(&xa),
        q: words_grid(&q),
        k: words_grid(&k),
        v: words_grid(&v),
        a: words_grid(&a),
        mid: words_grid(&mid),
        xf: words_grid(&xf),
        g: words_grid(&g),
        u: words_grid(&u),
        m: words_grid(&m),
        out: words_grid(&out),
    })
}
