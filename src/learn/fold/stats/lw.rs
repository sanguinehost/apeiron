//! Analytic shrinkage intensities of the output-side Hessian sketch toward its diagonal: the
//! Schäfer–Strimmer (2005) intensity of the per-sample scatter and its batch-means (sequence-unit)
//! refinement. Ported from `lw_intensity`, `_block_off` and `lw_block_intensity` in the removed
//! one-bit fold transfer (`fold_transfer.py:1101–1161`, read-only spec outside this repository).
//!
//! Target `D`: diagonal, unequal variances (Schäfer and Strimmer, *Stat. Appl. Genet. Mol. Biol.* 4
//! (2005) art. 32; the Ledoit–Wolf (2004) construction for that target). [`lw_intensity`] treats
//! the `n = K·N·T` samples `s_t = √w_t·g_t` as independent; [`lw_block_intensity`] takes one
//! sequence per block, so the correlated tokens of one sequence and the `K` label draws of one
//! token are not counted as independent samples.
//!
//! # Carrier
//!
//! Every value is a `Q32.32` word: the squares and dot products are exact `i128` accumulations and
//! each reported scalar is rounded once ([`div_rne`], ties to even). The `f64` parameters and
//! returned pairs are the cross-file boundary (`to_grid`/`from_grid`); no arithmetic runs in `f64`.
//! With `H = Σ_t s_t s_tᵀ` the raw scatter, `S4 = Σ_t ‖s_t‖⁴`, `Q = Σ_t s_t⁴`, `M = H/n` and
//! `off(X) = ‖X‖_F² − Σ_i X_ii²`:
//!
//! ```text
//! num = (S4 − ΣQ − n·off(M)) / (n(n−1));  λ = clamp(num / off(M), 0, 1)
//! ```
//!
//! The block variant accumulates `P = Σ_b off(M_b)` with `M_b = H_b/n_b`, giving
//! `num = (P − B·off(M)) / (B(B−1))` and the same `λ`. Both return `(num, λ)`; `λ` is the per-linear
//! `α` of `HO ← (1−λ)·HO + λ·diag(HO)`.
//!
//! # Deviations from the Python
//!
//! [`block_off`] forms the `o × o` block scatter `H_b = Σ_k S_kᵀ S_k` with [`wsyrk`] rather than the
//! Python's `(K T)×(K T)` Gram; `Σ_{a,b} ⟨S_a, S_b⟩² = ‖H_b‖_F²`, so the two are the same quantity
//! at `O(o²)` space. `off(M)` is `(‖H‖_F² − Σ_i H_ii²)/n²`: one exact sum and one rounding. Shape
//! validation precedes the `B < 2` early return, [`lw_block_intensity`] refuses `n == 0`, and
//! [`lw_intensity`] refuses `n < 2` where the formula divides by `n(n−1)`.
//!
//! # Evidence grade
//!
//! `tested`: `tests/learn_fold_lw.rs` on the build path. No Lean refinement links this file. Trust
//! boundary: the `Q32.32` arithmetic of the three functions on `&[f64]` inputs (the `f64` boundary
//! is the grid map only); not the capture that produces `S4`, `Q`, `H` or the block stacks, nor any
//! GPU/`torch` codegen.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the estimators index the row-major `[o, o]` scatter and the `[t, o]` blocks by \
              `i/j/a/b`, name the scalars `num`/`nf`/`off` and the intensity `lambda` as the \
              reference does, and cast the sample/block counts `usize → i128`, a checked \
              widening into the exact accumulator domain"
)]

use crate::learn::solve::q32::{div_rne, from_grid, round_q32_even, to_grid, GRID_BOUND};
use crate::learn::solve::syrk::{wsyrk, SyrkError};
use crate::numeric::fixed_ln::ONE;

use super::parts::{panel, shift};

/// A refusal from the shrinkage-intensity estimators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LwError {
    /// A slice length disagrees with the declared shape: `h` is not the `o·o` scatter, `q` is not
    /// the length-`o` vector, or a block is not the `t·o` stack.
    ShapeMismatch,
    /// A sample count below the divisor's floor: [`lw_intensity`] divides by `n(n−1)` (needs `n ≥
    /// 2`), [`lw_block_intensity`] divides the scatter by `n` (needs `n ≥ 1`).
    TooFewSamples,
    /// An input is outside the `Q32.32` grid, a primitive bound was violated, or a quotient left
    /// the grid. The `f64` entry points refuse instead of saturating.
    OutOfRange,
}

/// `v` as `i64`, or [`LwError::OutOfRange`] when it leaves the grid.
fn narrow(v: i128) -> Result<i64, LwError> {
    i64::try_from(v).map_err(|_| LwError::OutOfRange)
}

/// One `f64` input as a `Q32.32` word; a value outside the grid is refused, never clamped.
fn g(x: f64) -> Result<i64, LwError> {
    to_grid(x, GRID_BOUND).ok_or(LwError::OutOfRange)
}

/// A whole `f64` slice as `Q32.32` words. `O(len)` time and space.
fn grid(x: &[f64]) -> Result<Vec<i64>, LwError> {
    x.iter().map(|&v| g(v)).collect()
}

/// [`SyrkError`] as a [`LwError`].
fn map_syrk(e: SyrkError) -> LwError {
    match e {
        SyrkError::ShapeMismatch | SyrkError::ZeroN | SyrkError::ZeroO => LwError::ShapeMismatch,
        SyrkError::InputBound | SyrkError::Overflow => LwError::OutOfRange,
    }
}

/// `off(M) = ‖M‖_F² − Σ_i M_ii²` with `M = h / n`: the sum of the squared off-diagonal entries, as
/// a `Q32.32` word. `h` is the `o·o` scatter in grid words. The `o²` squares are summed exactly in
/// `i128` (a `Q64` value) and divided by `n²` once, so the result carries one rounding.
/// # Complexity
/// Time `O(o²)`, one serial pass over the scatter; space `O(1)`.
fn off_of(h: &[i64], o: usize, n: usize) -> Result<i64, LwError> {
    let mut total: i128 = 0;
    for i in 0..o {
        for j in 0..o {
            if j != i {
                let v = i128::from(h[i * o + j]);
                total += v * v;
            }
        }
    }
    let n2 = i128::try_from(n)
        .ok()
        .and_then(|v| v.checked_mul(v))
        .and_then(|v| v.checked_mul(1_i128 << 32))
        .ok_or(LwError::OutOfRange)?;
    narrow(div_rne(total, n2).ok_or(LwError::OutOfRange)?)
}

/// Schäfer–Strimmer (2005) intensity of the raw scatter `H` toward its diagonal, from the capture's
/// own samples: `(num, λ)` with `num = Σ_{i≠j} Var̂(mean_t s_ti s_tj)` and `λ = clamp(num/off, 0,
/// 1)` (`fold_transfer.py:1101–1121`). `s4 = Σ_t ‖s_t‖⁴`, `q = Σ_t s_t⁴` (length `o`), `h = Σ_t s_t
/// s_tᵀ` (row-major `o·o`), `n` the sample count. Returns `(0, 0)` when `off(H/n) ≤ 0`.
/// # Complexity
/// Time `O(o²)` for `off`, one pass over `h`; space `O(o)` (`O(o²)` at the boundary call).
/// # Errors
/// [`LwError::ShapeMismatch`] if `h.len() != o·o` or `q.len() != o`; [`LwError::TooFewSamples`] if
/// `off > 0` and `n < 2`; [`LwError::OutOfRange`] when an input or a quotient leaves the grid.
pub fn lw_intensity(
    s4: f64,
    q: &[f64],
    h: &[f64],
    o: usize,
    n: usize,
) -> Result<(f64, f64), LwError> {
    if h.len() != o * o || q.len() != o {
        return Err(LwError::ShapeMismatch);
    }
    let (hw, qw, s4w) = (grid(h)?, grid(q)?, g(s4)?);
    let off = off_of(&hw, o, n)?;
    if off <= 0 {
        return Ok((0.0, 0.0));
    }
    if n < 2 {
        return Err(LwError::TooFewSamples);
    }
    let mut q_sum: i128 = 0;
    for &v in &qw {
        q_sum += i128::from(v);
    }
    let nf = i128::try_from(n).map_err(|_| LwError::OutOfRange)?;
    let acc = i128::from(s4w) - q_sum - nf * i128::from(off);
    let num = narrow(div_rne(acc, nf * (nf - 1)).ok_or(LwError::OutOfRange)?)?;
    let lambda = div_rne(i128::from(num) << 32, i128::from(off))
        .and_then(|v| i64::try_from(v).ok())
        .ok_or(LwError::OutOfRange)?;
    Ok((from_grid(num), from_grid(lambda.clamp(0, ONE))))
}

/// `off(H_b) = ‖H_b‖_F² − Σ_i (H_b)_ii²` of ONE block, with `H_b = Σ_k S_kᵀ S_k`
/// (`fold_transfer.py:1123–1133`). `sk` holds the `K` label-draw stacks, each `t·o` row-major; `t`
/// is `T`, `o` is `O`. The scatter is [`wsyrk`] on the stacked rows after the exact `2⁻ᵏ` shift
/// into the primitive's `2³⁶` bound, with `2²ᵏ` undone on the Gram; the two Frobenius sums are
/// exact `i128` squares rounded once.
/// # Complexity
/// Time `O(K·T·o²)` pooled inside [`wsyrk`]; space `O(o² + K·T·o)` (the Python materialises the `(K
/// T)²` Gram).
/// # Errors
/// [`LwError::ShapeMismatch`] if any block's length is not `t·o`; [`LwError::OutOfRange`] when an
/// input or a rounded square leaves the grid.
pub fn block_off(sk: &[&[f64]], t: usize, o: usize) -> Result<f64, LwError> {
    for s in sk {
        if s.len() != t * o {
            return Err(LwError::ShapeMismatch);
        }
    }
    if o == 0 || t == 0 || sk.is_empty() {
        return Ok(0.0);
    }
    let rows = t.checked_mul(sk.len()).ok_or(LwError::OutOfRange)?;
    let mut stacked = Vec::with_capacity(rows * o);
    for s in sk {
        stacked.extend(grid(s)?);
    }
    let (xs, k) = panel(&stacked).ok_or(LwError::OutOfRange)?;
    let mut h = wsyrk(&xs, None, rows, o).map_err(map_syrk)?;
    shift(&mut h, -2 * k).ok_or(LwError::OutOfRange)?;
    let (mut total, mut diag): (i128, i128) = (0, 0);
    for i in 0..o {
        for j in 0..o {
            let v = i128::from(h[i * o + j]);
            total += v * v;
            if i == j {
                diag += v * v;
            }
        }
    }
    Ok(from_grid(
        round_q32_even(total - diag).map_err(|_| LwError::OutOfRange)?,
    ))
}

/// Batch-means (sequence-unit) intensity of the raw scatter `H` toward its diagonal, with the
/// sequence as the sample unit (`fold_transfer.py:1135–1161`). `p = Σ_b off(H_b)/n_b²` accumulated
/// by [`block_off`], `b` the block count `B`, `h = Σ_b H_b` the total scatter (row-major `o·o`),
/// `n` the total sample count. Returns `(num, λ)` with `num = (P − B·off(H/n)) / (B(B−1))` and `λ =
/// clamp(num/off(H/n), 0, 1)`; `B < 2` and `off(H/n) ≤ 0` return `(0, 0)`.
/// # Complexity
/// Time `O(o²)` for `off`, one pass over `h`; space `O(o²)` at the boundary call.
/// # Errors
/// [`LwError::ShapeMismatch`] if `h.len() != o·o`; [`LwError::TooFewSamples`] if `B ≥ 2` and `n ==
/// 0`; [`LwError::OutOfRange`] when an input or a quotient leaves the grid.
pub fn lw_block_intensity(
    p: f64,
    b: usize,
    h: &[f64],
    o: usize,
    n: usize,
) -> Result<(f64, f64), LwError> {
    if h.len() != o * o {
        return Err(LwError::ShapeMismatch);
    }
    if b < 2 {
        return Ok((0.0, 0.0));
    }
    if n == 0 {
        return Err(LwError::TooFewSamples);
    }
    let (pw, hw) = (g(p)?, grid(h)?);
    let off = off_of(&hw, o, n)?;
    if off <= 0 {
        return Ok((0.0, 0.0));
    }
    let bf = i128::try_from(b).map_err(|_| LwError::OutOfRange)?;
    let acc = i128::from(pw) - bf * i128::from(off);
    let num = narrow(div_rne(acc, bf * (bf - 1)).ok_or(LwError::OutOfRange)?)?;
    let lambda = div_rne(i128::from(num) << 32, i128::from(off))
        .and_then(|v| i64::try_from(v).ok())
        .ok_or(LwError::OutOfRange)?;
    Ok((from_grid(num), from_grid(lambda.clamp(0, ONE))))
}
