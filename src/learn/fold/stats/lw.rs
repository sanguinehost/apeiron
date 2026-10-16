//! Analytic shrinkage intensities of the output-side Hessian sketch toward its diagonal: the
//! Schäfer–Strimmer (2005) intensity of the per-sample scatter, and its batch-means (sequence-unit)
//! refinement. Ported from `lw_intensity`, `_block_off` and `lw_block_intensity` in the removed
//! one-bit fold transfer (`fold_transfer.py:1101–1161`, read-only spec outside this repository).
//!
//! Target `D`: diagonal, unequal variances (Schäfer and Strimmer, *Stat. Appl. Genet. Mol. Biol.* 4
//! (2005) art. 32; the Ledoit–Wolf (2004) construction for that target). [`lw_intensity`] treats
//! the `n = K·N·T` samples `s_t = √w_t·g_t` as independent; [`lw_block_intensity`] takes one
//! sequence per block, so the correlated tokens of one sequence and the `K` label draws of one
//! token are not counted as independent samples.
//!
//! With `H = Σ_t s_t s_tᵀ` the raw scatter, `S4 = Σ_t ‖s_t‖⁴`, `Q = Σ_t s_t⁴`, `M = H/n` and
//! `off(X) = ‖X‖_F² − Σ_i X_ii²`:
//!
//! ```text
//! num    = (S4 − ΣQ − n·off(M)) / (n(n−1)) = Σ_{i≠j} Var̂(mean_t s_ti s_tj)
//! λ      = clamp(num / off(M), 0, 1)
//! ```
//!
//! and the block variant accumulates `P = Σ_b off(M_b)` with `M_b = H_b/n_b`, `H_b` the block
//! scatter, giving `num = (P − B·off(M)) / (B(B−1))` and the same `λ`. [`lw_intensity`] and
//! [`lw_block_intensity`] return `(num, λ)`; `λ` is the per-linear `α` of the convex combination
//! `HO ← (1−λ)·HO + λ·diag(HO)`.
//!
//! # Deviations from the Python
//!
//! - [`block_off`] sums the squared row dot products of the stacked samples (`Σ_{a,b} ⟨S_a, S_b⟩² =
//!   ‖H_b‖_F²`) one pair at a time. The Python materialises the `(K T)×(K T)` Gram `Sall Sallᵀ` and
//!   sums its squares in one pass; the streamed sum is the same quantity and holds `O(o)` for the
//!   diagonal plus `O(1)`, never the `(K T)²` Gram nor the `O×O` scatter.
//! - Shape validation precedes the `B < 2` early return, so a malformed `H` is refused rather than
//!   silently returning `(0, 0)` on the `B < 2` branch.
//! - [`lw_block_intensity`] refuses `n == 0`; the Python divides the scatter by `n` and would
//!   produce a non-finite `off`. [`lw_intensity`] refuses `n < 2` where the Python divides by
//!   `n(n−1)`, and returns `(0, 0)` on `off ≤ 0` first, exactly as the Python does.
//!
//! # Evidence grade
//!
//! `tested`: the gate's `cargo test` on the build path (`tests/learn_fold_lw.rs`). No Lean
//! refinement links this file; it is not on the extracted fragment.
//!
//! # Trust boundary
//!
//! Covers the `f64` arithmetic of the three functions on `&[f64]` inputs. It does not cover the
//! capture that produces `S4`, `Q`, `H` or the block stacks, nor any GPU/`torch` codegen.

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    reason = "the estimators index the row-major `[o, o]` scatter and the `[t, o]` blocks by \
              `i/j/a/b`, name the scalars `num`/`nf`/`off` and the intensity `lambda` as the \
              reference does, accumulate `Σ x²` as `x*x` (not `mul_add`) to keep the elementwise \
              arithmetic the reference's, and cast the sample/block counts `usize → f64`, a \
              deliberate lossy cast of a dimension"
)]

/// A refusal from the shrinkage-intensity estimators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LwError {
    /// A slice length disagrees with the declared shape: `h` is not the `o·o` scatter, `q` is not
    /// the length-`o` vector, or a block is not the `t·o` stack.
    ShapeMismatch,
    /// A sample count below the divisor's floor: [`lw_intensity`] divides by `n(n−1)` (needs
    /// `n ≥ 2`), [`lw_block_intensity`] divides the scatter by `n` (needs `n ≥ 1`).
    TooFewSamples,
}

/// `min(1, max(0, x))` as the Python writes it: [`f64::clamp`] over the finite `num/off` the
/// callers pass (a NaN input is not reachable here). Time `O(1)`.
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// `off(M) = ‖M‖_F² − Σ_i M_ii²` with `M = h / n`: the sum of the squared off-diagonal entries.
///
/// # Complexity
///
/// Time `O(o²)`: one pass over the `o·o` scatter. Space `O(1)`: two running sums.
fn off_of(h: &[f64], o: usize, n: f64) -> f64 {
    let mut sumsq = 0.0f64;
    let mut diag = 0.0f64;
    for i in 0..o {
        for j in 0..o {
            let m = h[i * o + j] / n;
            sumsq += m * m;
            if i == j {
                diag += m * m;
            }
        }
    }
    sumsq - diag
}

/// Schäfer–Strimmer (2005) intensity of the raw scatter `H` toward its diagonal, from the capture's
/// own samples: `(num, λ)` with `num = Σ_{i≠j} Var̂(mean_t s_ti s_tj)` and `λ = clamp(num/off, 0, 1)`
/// (`fold_transfer.py:1101–1121`).
///
/// `s4 = Σ_t ‖s_t‖⁴`, `q = Σ_t s_t⁴` (length `o`), `h = Σ_t s_t s_tᵀ` (row-major `o·o`), `n` the
/// sample count. Returns `(0, 0)` when `off(H/n) ≤ 0` (a diagonal scatter has no off-diagonal
/// intensity to shrink).
///
/// # Complexity
///
/// Time `O(o²)` for `off`, one pass over `h`, plus `O(o + n)` for the `S4`/`Q` sums already held by
/// the caller. Space `O(1)`. Cost `O(o²)` once per `(layer, kind)`.
///
/// # Errors
///
/// [`LwError::ShapeMismatch`] if `h.len() != o·o` or `q.len() != o`. [`LwError::TooFewSamples`] if
/// `off > 0` and `n < 2`, where the formula divides by `n(n−1)`.
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
    let off = off_of(h, o, n as f64);
    if off <= 0.0 {
        return Ok((0.0, 0.0));
    }
    if n < 2 {
        return Err(LwError::TooFewSamples);
    }
    let q_sum: f64 = q.iter().sum();
    let nf = n as f64;
    let num = (s4 - q_sum - nf * off) / (nf * (nf - 1.0));
    Ok((num, clamp01(num / off)))
}

/// `off(H_b) = ‖H_b‖_F² − Σ_i (H_b)_ii²` of ONE block, with `H_b = Σ_k S_kᵀ S_k` (`fold_transfer.py:
/// 1123–1133`). `sk` holds the `K` label-draw stacks, each `t·o` row-major; `t` is `T`, `o` is `O`.
///
/// # Complexity
///
/// Time `O(K² T² O)`: `Σ_{a,b}` over the `K·T` stacked rows, each dot product `O(O)`. Space `O(O)`
/// for the diagonal plus `O(1)` for the streamed Gram sum (the Python materialises the `(K T)²`
/// Gram).
///
/// # Errors
///
/// [`LwError::ShapeMismatch`] if any block's length is not `t·o`.
pub fn block_off(sk: &[&[f64]], t: usize, o: usize) -> Result<f64, LwError> {
    for s in sk {
        if s.len() != t * o {
            return Err(LwError::ShapeMismatch);
        }
    }
    // diag(H_b)_i = Σ_a Sall[a][i]², the column sums of squares of the stacked samples.
    let mut dg = vec![0.0f64; o];
    for s in sk {
        for r in 0..t {
            let base = r * o;
            for i in 0..o {
                let v = s[base + i];
                dg[i] += v * v;
            }
        }
    }
    // ‖H_b‖_F² = Σ_{a,b} ⟨S_a, S_b⟩², the squared Frobenius norm of the (K T)×(K T) row Gram,
    // streamed one pair at a time.
    let k = sk.len();
    let rows = k * t;
    let mut total = 0.0f64;
    for a in 0..rows {
        let (ba, ra) = (a / t, a % t);
        let sa = &sk[ba][ra * o..ra * o + o];
        for b in 0..rows {
            let (bb, rb) = (b / t, b % t);
            let sb = &sk[bb][rb * o..rb * o + o];
            let dot: f64 = sa.iter().zip(sb.iter()).map(|(x, y)| x * y).sum();
            total += dot * dot;
        }
    }
    let dg_sq: f64 = dg.iter().map(|x| x * x).sum();
    Ok(total - dg_sq)
}

/// Batch-means (sequence-unit) intensity of the raw scatter `H` toward its diagonal, with the
/// sequence as the sample unit (`fold_transfer.py:1135–1161`).
///
/// `p = Σ_b off(H_b)/n_b²` accumulated by [`block_off`], `b` the block count `B`, `h = Σ_b H_b` the
/// total scatter (row-major `o·o`), `n` the total sample count. Returns `(num, λ)` with
/// `num = (P − B·off(H/n)) / (B(B−1))` and `λ = clamp(num/off(H/n), 0, 1)`; `B < 2` has no
/// between-block variance and returns `(0, 0)`, and so does `off(H/n) ≤ 0`.
///
/// # Complexity
///
/// Time `O(o²)` for `off`, one pass over `h`; the `B·K²T²O` block Grams are paid by the caller when
/// it accumulates `P`. Space `O(1)` beyond `h`.
///
/// # Errors
///
/// [`LwError::ShapeMismatch`] if `h.len() != o·o`. [`LwError::TooFewSamples`] if `B ≥ 2` and
/// `n == 0` (the scatter is divided by `n`).
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
    let off = off_of(h, o, n as f64);
    if off <= 0.0 {
        return Ok((0.0, 0.0));
    }
    let bf = b as f64;
    let num = (p - bf * off) / (bf * (bf - 1.0));
    Ok((num, clamp01(num / off)))
}
