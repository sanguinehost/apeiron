//! The per-stage statistics and the Tikhonov solve of the production `fold_gqa` fold-transfer arm,
//! on the `Q32.32` `i64` grid.
//!
//! Port of `fold_transfer.py:1538-1690` (`/home/socol/oc-tmp/lanes/port/spec/fold_transfer.py`) and
//! its two helpers [`ridge_solve`] (`:426-428`) and [`metric`] (`:430-432`) at the production arm's
//! flags only (`teacher_in = false`, `fold_gqa`, `split_intra_inter`, `sink_gate`,
//! `block_alpha = 1.0`; the teacher-output and R1–R5 paths are out of scope). Item → reference:
//! [`sink_token_weights`] `:1592-1597`; [`StageAccum::add_slab`] `:1541-1648`; [`solve_stage`]
//! `:1652-1682`; [`SolveParams`] `pv_finish.py:75-84`; [`bias_update`] `:1880`.
//!
//! # Carrier
//! Every computed value is a `Q32.32` `i64` word (`v/2³²`): products are exact in `i128` (`Q64.64`),
//! a contraction rounds **once** to `Q32.32` by [`round_q32_even`] (ties to even), a row norm is the
//! floored `isqrt` of the exact squared sum, and a quotient is `round(num/den)` in `i128`. `Hm_w`
//! applies the weight **linearly** (`:1598-1599`, not `√w`: the reference is the weighted least
//! squares `Σ_t w_t |y_t − W x_t|²`); `ridge` stays `Q64.64` so `1e-8` keeps its precision.
//!
//! # Pool
//! Only matrix-sized work is pooled, through `chunks_mut_global` and one `ChunkTask` ([`Dot`]): the
//! `rows·n²` Gram, the `rows·n·out` cross-Gram and the `n²·out` matvec. Every other pass — a
//! conversion, transpose, per-row norm or clamp, column sum, combine — is a serial integer loop: a
//! dispatch costs about what a few instructions per entry save. `hm`, `hm_w`, `cs`, `cts`, `x_sum`
//! and `y_sum` are `Q32.32` words and `n_toks` an integer count; every solve scalar crosses once.
//!
//! # Evidence grade and trust boundary
//! `tested`: `cargo test --test learn_fold_stage` compares every function with the `f64` oracle it
//! replaced inside a derived grid bound; no Lean refinement links this file, so nothing is
//! `proved`. Not covered: rustc/LLVM codegen, the reference's fp32 accumulation, the pool's thread
//! boundary, and any activation the caller does not supply.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the flat-slice kernels index row-major `[rows, n]`/`[n, out]` buffers by `i/j/k/o/r/t`; \
              `pow2_exp` narrows a masked 11-bit exponent field; `param_q64`'s cast follows a range \
              check; `StageAccum`/`StageError` repeat the module name by the brief's naming."
)]

use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;
use crate::learn::fold::stats::hocapture::Kind;
use crate::learn::solve::chol::{chol_solve, SolveError};
use crate::learn::solve::q32::{
    div_rne, fill_scale, from_grid, ilog2_pos, round_q32_even, rshift_rne, to_grid, GRID_BOUND,
};
use crate::numeric::fixed_rsqrt::isqrt;

/// The `Q32.32` word of `1.0`.
const ONE: i64 = 1 << 32;
/// The reference row-norm floor `1e-6` as a `Q32.32` word (`round(1e-6·2³²)`).
const MIN_ROW_NORM: i64 = 4_295;
/// `2⁶⁴`, the scale that takes an `f64` parameter to `Q64.64`.
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

/// A refusal from a per-stage statistics or solve operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageError {
    /// A slice length disagrees with the declared shape.
    Shape,
    /// An empty population: a slab with `t < 2` rows per sequence (no median), or a zero token count.
    Empty,
    /// An input carried a NaN or infinite value.
    NonFinite,
    /// The ridged matrix `H + ridge·mean(diag H)·I` was not positive-definite at a `Cholesky` pivot.
    NotPositiveDefinite,
    /// A boundary conversion, or the primitive's checked `i128` arithmetic, left its range.
    Overflow,
    /// A split intra/inter solve was requested for an output with no `Cts` accumulator.
    MissingCts,
}

impl From<SolveError> for StageError {
    fn from(e: SolveError) -> Self {
        match e {
            SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
                Self::NotPositiveDefinite
            }
            SolveError::ShapeMismatch | SolveError::Empty => Self::Shape,
            SolveError::Overflow => Self::Overflow,
        }
    }
}

/// The `fold_gqa` solve hyper-parameters, the production arm's `a.*` defaults
/// (`/home/socol/oc-tmp/lanes/port/spec/pv_finish.py:75-84`, `arm_args`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveParams {
    /// `a.ridge_intra = 0.045`: ridge of the intra-residual solve.
    pub ridge_intra: f64,
    /// `a.ridge_inter = 0.025`: ridge of the inter-residual solve.
    pub ridge_inter: f64,
    /// `a.inter_gamma = 1.0`: weight of `dW_inter` added to `dW_intra`.
    pub inter_gamma: f64,
    /// `a.ridge_w0_gqa = 0.03`: ridge of the direct solve for q/k/v/gate/up.
    pub ridge_w0_gqa: f64,
    /// `a.inter_gamma_in = 0.25` (the `-gin25` arm): step on `dW_inter` for the direct outputs.
    pub gin: f64,
    /// `a.sink_tau = 1.5`: row-norm cap on `dW` relative to `W0`.
    pub sink_tau: f64,
    /// The width `896` the ridges are scaled by (`n / 896`).
    pub n_ref: f64,
}

impl Default for SolveParams {
    fn default() -> Self {
        Self {
            ridge_intra: 0.045,
            ridge_inter: 0.025,
            inter_gamma: 1.0,
            ridge_w0_gqa: 0.03,
            gin: 0.25,
            sink_tau: 1.5,
            n_ref: 896.0,
        }
    }
}

/// `x` as a `Q32.32` word, or [`StageError::Overflow`] outside `|x| < 2³¹`.
fn word(x: f64) -> Result<i64, StageError> {
    to_grid(x, GRID_BOUND).ok_or(StageError::Overflow)
}

/// `v` as an `i64`, or [`StageError::Overflow`] outside its range.
fn narrow(v: i128) -> Result<i64, StageError> {
    i64::try_from(v).map_err(|_| StageError::Overflow)
}

/// `round(v / 2^s)` to nearest, ties to even, narrowed to `i64`.
fn shift(v: i128, s: u32) -> Result<i64, StageError> {
    narrow(rshift_rne(v, s).ok_or(StageError::Overflow)?)
}

/// A `Q64.64` accumulation rounded once to `Q32.32`.
fn rnd(v: i128) -> Result<i64, StageError> {
    round_q32_even(v).map_err(|_| StageError::Overflow)
}

/// `round(num/den)` to nearest, ties away from zero, for `den > 0`.
fn div_round(num: i128, den: i128) -> Result<i128, StageError> {
    if den <= 0 {
        return Err(StageError::Shape);
    }
    let half = den / 2;
    let adj = if num >= 0 {
        num.checked_add(half)
    } else {
        num.checked_sub(half)
    };
    Ok(adj.ok_or(StageError::Overflow)? / den)
}

/// An `f64` parameter as `round(x·2⁶⁴)`: the exact power-of-two rescale keeps `1e-8`'s precision.
pub(crate) fn param_q64(x: f64) -> Result<i128, StageError> {
    let q = x * TWO_POW_64;
    if !q.is_finite() || q.abs() >= 1.701_411_834_604_692_3e38 {
        return Err(StageError::Overflow);
    }
    Ok(q.round() as i128)
}

/// The integer width an `f64` dimension denotes, or [`StageError::Shape`]. `O(1)`.
pub(crate) fn width_n(x: f64) -> Result<usize, StageError> {
    let r = x.round();
    if !x.is_finite() || r < 1.0 || (x - r).abs() > 1e-9 || r > 16_777_216.0 {
        return Err(StageError::Shape);
    }
    usize::try_from(r as i64).map_err(|_| StageError::Shape)
}

/// The exponent `e` with `s == 2^e` for the exact power of two [`fill_scale`] returns.
fn pow2_exp(s: f64) -> i32 {
    ((s.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// `v·2^e` narrowed to `i64`, or [`StageError::Overflow`] when the shift leaves the range.
fn shift_word(v: i64, e: i32) -> Result<i64, StageError> {
    if e > 100 {
        return Err(StageError::Overflow);
    }
    if e >= 0 {
        return narrow(i128::from(v) << e);
    }
    let k = u32::try_from(-e).map_err(|_| StageError::Overflow)?;
    shift(i128::from(v), k)
}

/// The L2 norm of a `Q32.32` row: the floored `isqrt` of the exact `i128` squared sum.
fn norm_row(row: &[i64]) -> Result<i64, StageError> {
    let acc = row.iter().try_fold(0i128, |a, &v| {
        a.checked_add(i128::from(v) * i128::from(v))
            .ok_or(StageError::Overflow)
    })?;
    narrow(isqrt(acc))
}

/// The `Q32.32` L2 norm of the word row `a[base..base + width]`.
fn norm_at(a: &[i64], base: usize, width: usize) -> Result<i64, StageError> {
    let acc = (base..base + width).try_fold(0i128, |acc, k| {
        let v = i128::from(a[k]);
        acc.checked_add(v * v).ok_or(StageError::Overflow)
    })?;
    narrow(isqrt(acc))
}

/// The largest `|entry|` of a slice, for the boundary fill scale.
fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, &v| m.max(v.abs()))
}

/// `true` iff every entry is finite.
fn finite(a: &[f64]) -> bool {
    a.iter().all(|v| v.is_finite())
}

/// `src·s` on the grid, `s` an exact power of two (`1.0` at the unit boundary).
fn scaled(src: &[f64], s: f64) -> Result<Vec<i64>, StageError> {
    src.iter().map(|&v| word(v * s)).collect()
}

/// `src` row-major `[rows, out]` as a `Q32.32` transpose `[out, rows]`, the same panel weighted
/// linearly by `w`, and the exact unweighted column sums.
#[allow(
    clippy::type_complexity,
    reason = "the three panels are produced and consumed together by every caller."
)]
fn panels(
    src: &[i64],
    w: &[i64],
    rows: usize,
    out: usize,
) -> Result<(Vec<i64>, Vec<i64>, Vec<i128>), StageError> {
    let mut plain = vec![0i64; out * rows];
    let mut weighted = vec![0i64; out * rows];
    let mut col = vec![0i128; out];
    for r in 0..rows {
        for o in 0..out {
            let v = src[r * out + o];
            plain[o * rows + r] = v;
            let c = col[o].checked_add(i128::from(v));
            col[o] = c.ok_or(StageError::Overflow)?;
            weighted[o * rows + r] = rnd(i128::from(v) * i128::from(w[r]))?;
        }
    }
    Ok((plain, weighted, col))
}

/// The diagonal shift `ridge·(n/n_ref)·mean(diag h)` in `Q32.32`: the diagonal sum's rounded mean
/// times one [`div_rne`] `Q64.64` rate, so no `f64` touches the scale.
fn diag_shift(h: &[i64], n: usize, rq: i128, nr: Option<usize>) -> Result<i64, StageError> {
    let sum = (0..n).try_fold(0i128, |a, i| {
        a.checked_add(i128::from(h[i * n + i]))
            .ok_or(StageError::Overflow)
    })?;
    let mean = div_round(sum, i128::from(n as i64))?;
    let rate = match nr {
        Some(nr) => {
            let num = rq
                .checked_mul(i128::try_from(n).map_err(|_| StageError::Shape)?)
                .ok_or(StageError::Overflow)?;
            let den = i128::try_from(nr).map_err(|_| StageError::Shape)?;
            div_rne(num, den).ok_or(StageError::Overflow)?
        }
        None => rq,
    };
    let d = rate.checked_mul(mean).ok_or(StageError::Overflow)?;
    shift(d, 64)
}

/// `h` with [`diag_shift`] added to every diagonal entry.
fn ridged(h: &[i64], n: usize, rq: i128, nr: Option<usize>) -> Result<Vec<i64>, StageError> {
    let mut hr = h.to_vec();
    let s = diag_shift(h, n, rq, nr)?;
    for i in 0..n {
        let at = i * n + i;
        hr[at] = hr[at].checked_add(s).ok_or(StageError::Overflow)?;
    }
    Ok(hr)
}

/// The common right shift `k` that brings `h` and every `c` below `2⁶¹` raw, so `chol_solve`'s
/// `a_ij << 64` products stay inside `i128`; a common factor on both sides leaves `X` unchanged.
fn grid_shift(slices: &[&[i64]]) -> u32 {
    let mx = slices
        .iter()
        .flat_map(|s| s.iter())
        .fold(0u128, |m, &v| m.max(u128::from(v.unsigned_abs())));
    ilog2_pos(mx).map_or(0, |bits| (bits + 1).saturating_sub(61))
}

/// `src / 2^k` on the grid.
fn shift_words(src: &[i64], k: u32) -> Result<Vec<i64>, StageError> {
    src.iter().map(|&v| shift(i128::from(v), k)).collect()
}

/// `a − b` entrywise on the grid.
fn sub_words(a: &[i64], b: &[i64]) -> Result<Vec<i64>, StageError> {
    let mut d = Vec::with_capacity(a.len());
    for (&x, &y) in a.iter().zip(b) {
        d.push(x.checked_sub(y).ok_or(StageError::Overflow)?);
    }
    Ok(d)
}

/// `dst[i·out + o] (=)+= Σ_t x[i·rows + t]·y[o·rows + t]`, one `Q64.64` sum per entry rounded once:
/// the slab Gram (`replace`, `x = y`), the cross-Gram (`Cs`/`Cts`), the `H·W0ᵀ` matvec and the
/// solve's `dW` step. Both operands are contiguous `[·, rows]` panels.
struct Dot<'a> {
    x: &'a [i64],
    y: &'a [i64],
    rows: usize,
    out: usize,
    replace: bool,
    first: &'a FirstErr<usize, StageError>,
}
impl ChunkTask<i64> for Dot<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        let (out, rows) = (self.out, self.rows);
        let lo = start / out;
        for r in 0..piece.len() / out {
            for o in 0..out {
                let mut acc: i128 = 0;
                for t in 0..rows {
                    let p =
                        i128::from(self.x[(lo + r) * rows + t]) * i128::from(self.y[o * rows + t]);
                    acc = match acc.checked_add(p) {
                        Some(s) => s,
                        None => return self.first.record(start, StageError::Overflow),
                    };
                }
                let at = r * out + o;
                let base = if self.replace { 0 } else { piece[at] };
                match rnd(acc).and_then(|w| base.checked_add(w).ok_or(StageError::Overflow)) {
                    Ok(v) => piece[at] = v,
                    Err(e) => return self.first.record(start, e),
                }
            }
        }
    }
}

/// `rows·n·out` `i128` multiply-adds per call, `≥ 10⁶` at the production shapes — the one pooled
/// entry point, called once per Gram, per output per slab, and per solve residual.
fn dot(
    dst: &mut [i64],
    x: &[i64],
    y: &[i64],
    rows: usize,
    out: usize,
    replace: bool,
) -> Result<(), StageError> {
    if dst.is_empty() {
        return Ok(());
    }
    let first: FirstErr<usize, StageError> = FirstErr::new();
    let task = Dot {
        x,
        y,
        rows,
        out,
        replace,
        first: &first,
    };
    // `~2·width` chunks over the output rows, so every participant gets a few.
    let rows = dst.len() / out;
    let chunk = rows.div_ceil(width_global().saturating_mul(2)).max(1) * out;
    chunks_mut_global(dst, chunk, &task);
    first.take().map_or(Ok(()), |(_, e)| Err(e))
}

/// `r_tok > thresh·r_med ? r_med/max(r_tok, 1e-6) : 1` per token (`fold_transfer.py:1592-1597`).
/// `r_t = max(‖h_t‖, ‖p_mid‖, ‖p_out‖)`, `r_s = max(‖h_s‖, ‖p_in_mid‖)`, `r_tok = max(r_t, r_s)`;
/// `r_med` is the median of `r_t` over positions `>= 1` of every sequence (`torch.median`'s lower
/// middle element). Inputs are row-major `[seqs·t, width]` `Q32.32` words, `thresh` a word; the
/// result is one weight word per token.
/// # Complexity
/// Time `O(seqs·t·width + k·log k)`, `k = seqs·(t−1)`, serial; space `O(seqs·t + k)`.
/// # Errors
/// [`StageError::Empty`] for `t < 2`, [`StageError::Shape`] on a length mismatch,
/// [`StageError::Overflow`] on a checked step.
#[allow(
    clippy::too_many_arguments,
    reason = "the reference passes the tapped tensors, the stage input, the optional mid input, the \
              geometry and the threshold separately (`fold_transfer.py:1592-1597`)."
)]
pub fn sink_token_weights(
    h_t: &[i64],
    p_mid: &[i64],
    p_out: &[i64],
    h_s: &[i64],
    p_in_mid: Option<&[i64]>,
    seqs: usize,
    t: usize,
    width: usize,
    thresh: i64,
) -> Result<Vec<i64>, StageError> {
    let rows = seqs * t;
    if rows == 0 || width == 0 || t < 2 {
        return Err(StageError::Empty);
    }
    let ins = [h_t, p_mid, p_out, h_s];
    let bad = |s: &[i64]| s.len() != rows * width;
    if ins.iter().any(|s| bad(s)) || p_in_mid.is_some_and(bad) {
        return Err(StageError::Shape);
    }
    let mut r_t = vec![0i64; rows];
    for r in 0..rows {
        let base = r * width;
        r_t[r] = norm_at(h_t, base, width)?
            .max(norm_at(p_mid, base, width)?)
            .max(norm_at(p_out, base, width)?);
    }
    // Positions >= 1 of every sequence; `torch.median`'s lower middle is `(k-1)/2`.
    let mut med: Vec<i64> = (0..seqs)
        .flat_map(|s| r_t[s * t + 1..(s + 1) * t].iter().copied())
        .collect();
    if med.is_empty() {
        return Err(StageError::Empty);
    }
    med.sort_unstable();
    let r_med = med[(med.len() - 1) / 2];
    let mut w = vec![0i64; rows];
    for r in 0..rows {
        let base = r * width;
        let mut rs = norm_at(h_s, base, width)?;
        if let Some(p) = p_in_mid {
            rs = rs.max(norm_at(p, base, width)?);
        }
        let tok = r_t[r].max(rs).max(MIN_ROW_NORM);
        let num = i128::from(r_med) << 32;
        w[r] = if i128::from(tok) * i128::from(ONE) > i128::from(thresh) * i128::from(r_med) {
            narrow(div_round(num, i128::from(tok))?)?
        } else {
            ONE
        };
    }
    Ok(w)
}

/// The per-stage normal-equation accumulator over slabs of a calibration set
/// (`fold_transfer.py:1541-1648`). Every field is `Q32.32` words; `n_toks` is the integer count.
/// `hm`, `x_sum` and `y_sum` accumulate by checked integer adds, so no `f64` view sits between slabs.
#[derive(Clone, Debug, PartialEq)]
pub struct StageAccum {
    /// Input width `n`.
    pub n: usize,
    /// Output width per stage output (`names`), in stage order.
    pub outs: Vec<usize>,
    /// Whether output `j` carries a `Cts` accumulator (`split_intra_inter` on `o`/`down`).
    pub with_ts: Vec<bool>,
    /// `Hm = Σ XᵀX`, row-major `[n, n]`, unweighted, `Q32.32` words.
    pub hm: Vec<i64>,
    /// `Hm_w = Σ XwᵀXw`, row-major `[n, n]`, `Xw = X·w` (linear weight).
    pub hm_w: Vec<i64>,
    /// `Cs[j] = Σ Xwᵀ(Y_j·w)`, row-major `[n, outs[j]]`.
    pub cs: Vec<Vec<i64>>,
    /// `Cts[j] = Σ Xwᵀ(Y_ts_j·w)`, row-major `[n, outs[j]]`; empty where `with_ts[j]` is `false`.
    pub cts: Vec<Vec<i64>>,
    /// `x_sum = Σ X`, length `n`, unweighted, `Q32.32` words.
    pub x_sum: Vec<i64>,
    /// `y_sum[j] = Σ Y_j`, length `outs[j]`, unweighted, `Q32.32` words.
    pub y_sum: Vec<Vec<i64>>,
    /// The token count `Σ rows` folded in so far.
    pub n_toks: usize,
}

impl StageAccum {
    /// A zeroed accumulator for input width `n` and outputs `outs`; `with_ts[j]` names the outputs
    /// that carry a `Cts` (`o`/`down` under `split_intra_inter`).
    /// # Complexity
    /// Time and space `O(n² + n·Σ outs)`.
    /// # Errors
    /// [`StageError::Shape`] if `outs` and `with_ts` differ in length, or `n == 0`.
    pub fn new(n: usize, outs: &[usize], with_ts: &[bool]) -> Result<Self, StageError> {
        if n == 0 || outs.len() != with_ts.len() {
            return Err(StageError::Shape);
        }
        let cs = outs.iter().map(|&o| vec![0i64; n * o]).collect();
        let cts = outs
            .iter()
            .zip(with_ts)
            .map(|(&o, &ts)| if ts { vec![0i64; n * o] } else { Vec::new() })
            .collect();
        let y_sum = outs.iter().map(|&o| vec![0i64; o]).collect();
        Ok(Self {
            n,
            outs: outs.to_vec(),
            with_ts: with_ts.to_vec(),
            hm: vec![0i64; n * n],
            hm_w: vec![0i64; n * n],
            cs,
            cts,
            x_sum: vec![0i64; n],
            y_sum,
            n_toks: 0,
        })
    }

    /// Fold one slab, every tensor `Q32.32` words: `x` row-major `[rows, n]`, `w_tok` length
    /// `rows`, `ys[j]` row-major `[rows, outs[j]]`, `y_ts[j]` the `o`/`down` target block (`None`
    /// where `with_ts[j]` is `false`). Accumulates `Hm`, `Hm_w`, `Cs`, `Cts`, `x_sum`, `y_sum` and
    /// the token count
    /// (`fold_transfer.py:1590-1647`) with every contraction an exact `i128` sum rounded once.
    /// # Complexity
    /// Time `O(rows·(n² + n·Σ outs))` `i128` multiply-adds; the pooled Gram (`rows·n²`) dominates;
    /// space `O(rows·(n + Σ outs) + n²)`.
    /// # Errors
    /// [`StageError::Shape`] on a length or geometry mismatch (including a missing `y_ts[j]`),
    /// [`StageError::Overflow`] on a checked step.
    pub fn add_slab(
        &mut self,
        x: &[i64],
        w_tok: &[i64],
        ys: &[&[i64]],
        y_ts: &[Option<&[i64]>],
    ) -> Result<(), StageError> {
        let rows = w_tok.len();
        let n = self.n;
        let m = self.outs.len();
        if x.len() != rows * n
            || ys.len() != m
            || y_ts.len() != m
            || (0..m).any(|j| {
                ys[j].len() != rows * self.outs[j]
                    || (self.with_ts[j] && y_ts[j].is_none_or(|s| s.len() != rows * self.outs[j]))
            })
        {
            return Err(StageError::Shape);
        }
        let (xt, xwt, xsum) = panels(x, w_tok, rows, n)?;
        let mut g = vec![0i64; n * n];
        dot(&mut g, &xt, &xt, rows, n, true)?;
        for (h, &v) in self.hm.iter_mut().zip(&g) {
            *h = h.checked_add(v).ok_or(StageError::Overflow)?;
        }
        dot(&mut g, &xwt, &xwt, rows, n, true)?;
        for (h, &v) in self.hm_w.iter_mut().zip(&g) {
            *h = h.checked_add(v).ok_or(StageError::Overflow)?;
        }
        for (s, &v) in self.x_sum.iter_mut().zip(&xsum) {
            *s = s.checked_add(narrow(v)?).ok_or(StageError::Overflow)?;
        }
        for j in 0..m {
            let out = self.outs[j];
            let (_, ywt, ysum) = panels(ys[j], w_tok, rows, out)?;
            dot(&mut self.cs[j], &xwt, &ywt, rows, out, false)?;
            for (s, &v) in self.y_sum[j].iter_mut().zip(&ysum) {
                *s = s.checked_add(narrow(v)?).ok_or(StageError::Overflow)?;
            }
            if let Some(yts) = y_ts[j].filter(|_| self.with_ts[j]) {
                let twt = panels(yts, w_tok, rows, out)?.1;
                dot(&mut self.cts[j], &xwt, &twt, rows, out, false)?;
            }
        }
        self.n_toks = self.n_toks.checked_add(rows).ok_or(StageError::Overflow)?;
        Ok(())
    }
}

/// The `fold_gqa` split intra/inter solve (`fold_transfer.py:1652-1682`) for one stage:
/// `H_solve = Hm_w`. For [`Kind::O`]/[`Kind::Down`] the residual against `W0` is split into
/// `C_intra = Cts − H_solve·W0ᵀ` and `C_inter = Cs − Cts`, solved at `ridge_intra·(n/n_ref)` and
/// `ridge_inter·(n/n_ref)`, combined as `dW = dW_intra + inter_gamma·dW_inter` and row-capped at
/// `sink_tau·max(‖W0_row‖, 1e-6)` under `sink_tau > 0`. Other kinds solve the direct residual
/// `Cs − H_solve·W0ᵀ` at `ridge_w0_gqa·(n/n_ref)` and step by `gin`. `w0` and the returned `Ws`
/// are `Q32.32` words; the caller converts at the fit's `f64` boundary.
/// # Complexity
/// Time `O(Σ_j (n³ + n²·outs[j]))` `i128` multiply-adds: one pooled `H·W0ᵀ`, one `Cholesky`
/// (`n³/3`) and one or two triangular solves per residual; space `O(n² + Σ_j n·outs[j])`.
/// # Errors
/// [`StageError::Shape`] on a `w0`/`kinds` mismatch, [`StageError::MissingCts`] for an `o`/`down`
/// output with no `Cts`, [`StageError::NotPositiveDefinite`] from the solve,
/// [`StageError::Overflow`] on a checked step.
pub fn solve_stage(
    accum: &StageAccum,
    w0: &[&[i64]],
    kinds: &[Kind],
    p: &SolveParams,
) -> Result<Vec<Vec<i64>>, StageError> {
    let n = accum.n;
    let m = accum.outs.len();
    let bad = |j: usize| w0[j].len() != accum.outs[j] * n;
    if n == 0 || w0.len() != m || kinds.len() != m || (0..m).any(bad) {
        return Err(StageError::Shape);
    }
    let mut ws = Vec::with_capacity(m);
    // One conversion per scalar: the ridges to `Q64.64`, the step weights to `Q32.32`, and the
    // reference width to an integer; `ridge·n/n_ref` below is one integer `div_rne` quotient.
    let (ri, rr, rw) = (
        param_q64(p.ridge_intra)?,
        param_q64(p.ridge_inter)?,
        param_q64(p.ridge_w0_gqa)?,
    );
    let (gamma, gin, tau) = (word(p.inter_gamma)?, word(p.gin)?, word(p.sink_tau)?);
    let n_ref = width_n(p.n_ref)?;
    for j in 0..m {
        let out = accum.outs[j];
        let wq = w0[j];
        let mut c0 = vec![0i64; n * out];
        dot(&mut c0, &accum.hm_w, wq, n, out, false)?;
        let d = if matches!(kinds[j], Kind::O | Kind::Down) {
            if !accum.with_ts[j] {
                return Err(StageError::MissingCts);
            }
            let ci = sub_words(&accum.cts[j], &c0)?;
            let cx = sub_words(&accum.cs[j], &accum.cts[j])?;
            let k = grid_shift(&[&accum.hm_w, &ci, &cx]);
            let hq = shift_words(&accum.hm_w, k)?;
            let (ciq, cxq) = (shift_words(&ci, k)?, shift_words(&cx, k)?);
            let hi = ridged(&hq, n, ri, Some(n_ref))?;
            let d_intra = chol_solve(&hi, &ciq, n, out)?;
            let hi = ridged(&hq, n, rr, Some(n_ref))?;
            let d_inter = chol_solve(&hi, &cxq, n, out)?;
            let mut dw = vec![0i64; out * n];
            for (at, (&d_in, &d_ix)) in d_intra.x.iter().zip(&d_inter.x).enumerate() {
                let t = rnd(i128::from(gamma) * i128::from(d_ix))?;
                dw[(at % out) * n + at / out] = d_in.checked_add(t).ok_or(StageError::Overflow)?;
            }
            if tau > 0 {
                for o in 0..out {
                    let row = o * n..o * n + n;
                    let w0_rn = norm_row(&wq[row.clone()])?.max(MIN_ROW_NORM);
                    let dw_rn = norm_row(&dw[row.clone()])?.max(MIN_ROW_NORM);
                    if i128::from(dw_rn) * i128::from(ONE) > i128::from(tau) * i128::from(w0_rn) {
                        let num = i128::from(tau) * i128::from(w0_rn);
                        let s = narrow(div_round(num, i128::from(dw_rn))?)?;
                        for v in &mut dw[row] {
                            *v = rnd(i128::from(*v) * i128::from(s))?;
                        }
                    }
                }
            }
            dw
        } else {
            let cx = sub_words(&accum.cs[j], &c0)?;
            let k = grid_shift(&[&accum.hm_w, &cx]);
            let hq = shift_words(&accum.hm_w, k)?;
            let cxq = shift_words(&cx, k)?;
            let hr = ridged(&hq, n, rw, Some(n_ref))?;
            let d = chol_solve(&hr, &cxq, n, out)?;
            let mut dw = vec![0i64; out * n];
            for (at, &v) in d.x.iter().enumerate() {
                dw[(at % out) * n + at / out] = rnd(i128::from(gin) * i128::from(v))?;
            }
            dw
        };
        let mut wsj = vec![0i64; out * n];
        for at in 0..out * n {
            wsj[at] = wq[at].checked_add(d[at]).ok_or(StageError::Overflow)?;
        }
        ws.push(wsj);
    }
    Ok(ws)
}

/// The ridge normal-equation solve `ridge_solve(H, C, ridge)` (`fold_transfer.py:426-428`):
/// `Hr = H` with `diag(Hr) += ridge·mean(diag H)`, then `Hr·X = C`, returning `Xᵀ` (`[out, n]`);
/// `h` is row-major `[n, n]`, `c` row-major `[n, out]`. The solve is the shared [`chol_solve`].
/// `H` and `C` are each rounded with their own exact power-of-two unit from [`fill_scale`], so each
/// fills `|v| < 2³¹`; the primitive's `X_q = (s_c/s_h)·X` is unscaled back by one exact shift.
/// # Complexity
/// Time `O(n³ + n²·out)`; space `O(n² + n·out)`.
/// # Errors
/// [`StageError::Shape`] if a length disagrees with `n`/`out`, [`StageError::NonFinite`] on a
/// NaN/∞ entry, [`StageError::NotPositiveDefinite`] if `Hr` is not positive-definite,
/// [`StageError::Overflow`] on a boundary or primitive overflow.
pub fn ridge_solve(
    h: &[f64],
    c: &[f64],
    n: usize,
    out: usize,
    ridge: f64,
) -> Result<Vec<f64>, StageError> {
    if n == 0 || h.len() != n * n || c.len() != n * out {
        return Err(StageError::Shape);
    }
    if !finite(h) || !finite(c) {
        return Err(StageError::NonFinite);
    }
    let (s_h, s_c) = (fill_scale(max_abs(h)), fill_scale(max_abs(c)));
    let hq = scaled(h, s_h)?;
    let cq = scaled(c, s_c)?;
    let ridge_q = param_q64(ridge)?;
    let solved = chol_solve(&ridged(&hq, n, ridge_q, None)?, &cq, n, out)?;
    let e = pow2_exp(s_h) - pow2_exp(s_c);
    // `solved.x` is `[n, out]`; the returned `Xᵀ` is `[out, n]`.
    let mut out_t = vec![0.0f64; out * n];
    for (at, &v) in solved.x.iter().enumerate() {
        out_t[(at % out) * n + at / out] = from_grid(shift_word(v, e)?);
    }
    Ok(out_t)
}

/// The metric matrix `Hl = H` with `diag += ridge·mean(diag H)` (`fold_transfer.py:430-432`), added
/// on the grid by [`diag_shift`]; unlike [`crate::learn::fold::fit::icm::ridged_metric`] the shift
/// is not scaled by `n/896`. `hm` is row-major `[n, n]` Q32.32; `ridge` is the Q32.32 word the
/// caller's `rg_q` already is, so no `f64` crosses this entry.
/// # Complexity
/// Time and space `O(n²)`, serial.
/// # Errors
/// [`StageError::Shape`] if `hm.len() != n²` or `n == 0`; [`StageError::Overflow`] on a checked
/// step.
pub fn metric(hm: &[i64], n: usize, ridge: i64) -> Result<Vec<i64>, StageError> {
    if n == 0 || hm.len() != n * n {
        return Err(StageError::Shape);
    }
    ridged(hm, n, i128::from(ridge) << 32, None)
}

/// The post-quantisation bias update `db = (y_sum − Wp·x_sum)/n_toks` (`fold_transfer.py:1880`) for
/// the `q`/`k`/`v` biases after `Wp` (`[out, n]`) is fixed: the exact `i128` dot product subtracted
/// from `y_sum·2³²` is `Q64.64`, divided by the `Q32.32` token count. Every operand and the result
/// are Q32.32 words.
/// # Complexity
/// Time and space `O(out·n)`, serial.
/// # Errors
/// [`StageError::Shape`] on a length mismatch or a non-positive `n_toks`, [`StageError::Empty`] for
/// `n_toks == 0`, [`StageError::Overflow`] on a checked step.
pub fn bias_update(
    y_sum: &[i64],
    x_sum: &[i64],
    wp: &[i64],
    n_toks: i64,
) -> Result<Vec<i64>, StageError> {
    let (n, out) = (x_sum.len(), y_sum.len());
    if n == 0 || out == 0 || wp.len() != out * n {
        return Err(StageError::Shape);
    }
    if n_toks == 0 {
        return Err(StageError::Empty);
    }
    let mut db = vec![0i64; out];
    for o in 0..out {
        let mut acc: i128 = 0;
        for (&p, &q) in wp[o * n..(o + 1) * n].iter().zip(x_sum) {
            let t = i128::from(p) * i128::from(q);
            acc = acc.checked_add(t).ok_or(StageError::Overflow)?;
        }
        let num = (i128::from(y_sum[o]) << 32) - acc;
        db[o] = narrow(div_round(num, i128::from(n_toks))?)?;
    }
    Ok(db)
}
