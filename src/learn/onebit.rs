//! One-bit arms on the calibration Gram: the one-bit measurement.
//!
//! Every arm quantizes a row of `W` (`[out, n]`, row-major) in `Q1_0` blocks of 128 weights, each
//! block carrying one sign plane and one `f16` scale `a = mean|w|` over the block (`bits/weight =
//! 1 + 16/128`). The arms differ only in how the sign is chosen and in which column order the block
//! is walked:
//!
//! - [`Arm::Rtn`] — `sign(w)·a`, no feedback.
//! - [`Arm::Sd1`] — first-order sigma-delta: `e₀ = 0`, `s_t = sign(w_t + e_t)` (`sign(0) = +1`),
//!   `q_t = a·s_t`, `e_{t+1} = w_t + e_t − q_t`; the state resets per block.
//! - [`Arm::Sd1Path`] — `sd1` walked in the greedy nearest-neighbour order of the calibration Gram
//!   (`d(i,j) = H_ii + H_jj − 2H_ij`, starting at `argmax H_ii`; one permutation per tensor).
//! - [`Arm::Gptq1`] — GPTQ column feedback ([`crate::learn::gptq::gptq_ternary_absmean`]'s structure,
//!   damping `0.01·mean(diag)`) on the grid `{−a, +a}`.
//! - [`Arm::Gptq1Act`] — `gptq1` in descending `H_ii` (act) order.
//! - [`Arm::Gptq1Path`] — `gptq1` in the path order.
//! - [`Arm::HadRtn`] / [`Arm::HadGptq1`] — normalised 128-block Walsh–Hadamard `H₁₂₈/√128` on the
//!   columns of `W` and the congruence `H ↦ R H R` on the Gram (the transform preserves the output
//!   error because `R` is orthogonal), then `rtn` / `gptq1` in the rotated domain, then `R` again to
//!   return to the weight domain.
//!
//! # Subsumption
//!
//! The empirical base is the in-tree [`crate::learn::gptq`] (`src/learn/gptq.rs`), which ports
//! `fold/modelfold/src/quant.rs` at `d497161f` — error feedback under the calibration Gram. Every
//! feedback arm here reuses that Gram/Hessian machinery: [`feedback_u`] rebuilds the private
//! `feedback_hi` factor `U` (`UᵀU = H⁻¹`) from the public [`ul_cholesky`] and
//! [`upper_tri_inverse`]. No new Hessian routine is introduced.
//!
//! # Partiality
//!
//! [`quantize`] returns [`GptqError`]: [`GptqError::ShapeMismatch`] on a shape or rotation-block
//! mismatch, [`GptqError::ZeroGroup`] on `group == 0`, and [`GptqError::NotPositiveDefinite`] if the
//! damped Cholesky pivot is non-positive. No function here panics on caller input; the Hadamard arms
//! require `n` to be a whole number of 128-column blocks.
//!
//! # Trust boundary
//!
//! The quantizers run in `f64` and are `tested` (`tests/learn_onebit.rs`); the `sd1`/`sd1-path`
//! sign stream runs in the checked-integer [`crate::learn::sigma_delta::sd_binary_block`], refined
//! to `Numeric.SigmaDelta` (`proof/corpus/Numeric/OneBit/SigmaDeltaRefinement.lean`). The `output_error`
//! metric is the existing [`crate::learn::gptq::output_error`]; this module makes no performance
//! claim.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    reason = "the flat-slice kernels index row-major [out, n] buffers by o/i/j/k/p, casts between \
              f32/f64 and integer codes reflect the quantized representation, and doc comments use \
              mathematical subscripts (H_ii)"
)]

use crate::learn::gptq::{ul_cholesky, upper_tri_inverse, GptqError};
use crate::learn::sigma_delta::sd_binary_block;

/// The `Q1_0` one-bit block width: one sign plus one `f16` scale per 128 weights.
pub const ONEBIT_GROUP: usize = 128;

/// Bits per weight including the per-128 `f16` scale: `1 + 16/128`.
pub const ONEBIT_BPW: f64 = 1.0 + 16.0 / 128.0;

/// One-bit quantization arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// `sign(w)·a`, no feedback.
    Rtn,
    /// First-order sigma-delta in natural column order.
    Sd1,
    /// `sd1` in the greedy nearest-neighbour column order of the calibration Gram.
    Sd1Path,
    /// GPTQ column feedback on the grid `{−a, +a}`, natural order.
    Gptq1,
    /// `gptq1` in descending `H_ii` (act) order.
    Gptq1Act,
    /// `gptq1` in the greedy nearest-neighbour path order.
    Gptq1Path,
    /// Normalised 128-block Walsh–Hadamard on the columns, then `rtn`.
    HadRtn,
    /// The same rotation, then `gptq1`.
    HadGptq1,
}

impl Arm {
    /// Every arm, in report order.
    pub const ALL: [Arm; 8] = [
        Arm::Rtn,
        Arm::Sd1,
        Arm::Sd1Path,
        Arm::Gptq1,
        Arm::Gptq1Act,
        Arm::Gptq1Path,
        Arm::HadRtn,
        Arm::HadGptq1,
    ];

    /// The stable report name (`rtn`, `sd1-path`, …).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Arm::Rtn => "rtn",
            Arm::Sd1 => "sd1",
            Arm::Sd1Path => "sd1-path",
            Arm::Gptq1 => "gptq1",
            Arm::Gptq1Act => "gptq1-act",
            Arm::Gptq1Path => "gptq1-path",
            Arm::HadRtn => "had-rtn",
            Arm::HadGptq1 => "had-gptq1",
        }
    }
}

/// One arm's result on one weight matrix.
#[derive(Clone, Debug)]
pub struct OneBit {
    /// The dequantized weights `Ŵ`, row-major `[out, n]`.
    pub what: Vec<f32>,
    /// `max_t |e_t|/a` for the sigma-delta arms; `NaN` for the feedback and rotation arms.
    pub stab: f64,
    /// Bits per weight: `1 + 16/128`.
    pub bpw: f64,
}

/// Quantize `w` (`[out, n]`, row-major) with `arm` over the calibration Gram `gram` (`[n, n]`).
/// `group` is the scale/block granularity along `n` (128 for the `Q1_0` block).
///
/// # Complexity
/// - Temporal: arm-dependent. `rtn`/`sd1` are $O(\text{out} \cdot n)$; `gptq1`/`gptq1-act`/
///   `gptq1-path`/`had-gptq1` are $O(n^3 + \text{out} \cdot n^2)$ (the Hessian factor plus the
///   column feedback), `sd1-path` adds the $O(n^2)$ path build; `had-*` add
///   $O((\text{out}+n) \cdot n \log 128)$ for the transform.
/// - Spatial: $O(\text{out} \cdot n)$ f64 words for the sign plane, plus $O(n^2)$ for the feedback
///   factor $U$ and (for `had-*`) the rotated Gram and its copy.
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `out == 0`, `n == 0`, `w.len() != out * n`,
/// `gram.len() != n * n`, or (for the Hadamard arms) `n % 128 != 0`.
/// Returns [`GptqError::ZeroGroup`] if `group == 0`.
/// Returns [`GptqError::NotPositiveDefinite`] if the damped Cholesky pivot is non-positive.
#[must_use = "the dequantized weights are the result"]
pub fn quantize(
    arm: Arm,
    w: &[f32],
    out: usize,
    n: usize,
    gram: &[f64],
    group: usize,
) -> Result<OneBit, GptqError> {
    if out == 0 || n == 0 || w.len() != out * n || gram.len() != n * n {
        return Err(GptqError::ShapeMismatch);
    }
    if group == 0 {
        return Err(GptqError::ZeroGroup);
    }
    let wf: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
    let (wq, stab) = match arm {
        Arm::Rtn => (rtn_row(&wf, out, n, group), f64::NAN),
        Arm::Sd1 => sd1_row(&wf, out, n, &identity(n), group),
        Arm::Sd1Path => sd1_row(&wf, out, n, &path_order(gram, n), group),
        Arm::Gptq1 => (gptq_row(&wf, out, n, gram, &identity(n), group)?, f64::NAN),
        Arm::Gptq1Act => (
            gptq_row(&wf, out, n, gram, &act_order(gram, n), group)?,
            f64::NAN,
        ),
        Arm::Gptq1Path => (
            gptq_row(&wf, out, n, gram, &path_order(gram, n), group)?,
            f64::NAN,
        ),
        Arm::HadRtn => {
            let (wt, _gt) = had_transform(&wf, gram, out, n)?;
            let wqt = rtn_row(&wt, out, n, group);
            (block_transform(&wqt, out, n), f64::NAN)
        }
        Arm::HadGptq1 => {
            let (wt, gt) = had_transform(&wf, gram, out, n)?;
            let wqt = gptq_row(&wt, out, n, &gt, &identity(n), group)?;
            (block_transform(&wqt, out, n), f64::NAN)
        }
    };
    Ok(OneBit {
        what: wq.iter().map(|&x| x as f32).collect(),
        stab,
        bpw: ONEBIT_BPW,
    })
}

/// `sign(x)` with `sign(0) = +1`.
fn sign(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// The identity permutation over `0..n`.
fn identity(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// The mean absolute value over a block, floored away from 0 so a dead block divides cleanly.
fn mean_abs(w: &[f64], base: usize, g0: usize, g1: usize) -> f64 {
    let mean = (g0..g1).map(|j| w[base + j].abs()).sum::<f64>() / (g1 - g0) as f64;
    mean.max(1e-12)
}

/// `sign(w)·a` per block, natural order.
///
/// # Complexity
/// - Temporal: $O(\text{out} \cdot n)$ comparisons and multiplies.
/// - Spatial: $O(\text{out} \cdot n)$ f64 words for the output.
fn rtn_row(w: &[f64], out: usize, n: usize, group: usize) -> Vec<f64> {
    let mut wq = vec![0f64; out * n];
    for o in 0..out {
        let base = o * n;
        let mut g0 = 0;
        while g0 < n {
            let g1 = (g0 + group).min(n);
            let a = mean_abs(w, base, g0, g1);
            for j in g0..g1 {
                wq[base + j] = a * sign(w[base + j]);
            }
            g0 = g1;
        }
    }
    wq
}

/// The fixed-point fractional bits of the `sd1` recurrence: `S = 24`. An `f16` weight is
/// `k·2^(E−25)` with `k ∈ [1024, 2047]`, `1 ≤ E ≤ 30` (or `m·2^−24` subnormal), so `w·2^24` is an
/// integer for every `f16` weight: the integer body sees the exact W7 weights and only the per-block
/// scale `a = mean|w|` is rounded (`|round(a·2^24) − a·2^24| ≤ 1/2`, relative error below `2^−24`).
pub const SD_FRAC_BITS: u32 = 24;

/// `2^24`, the fixed-point unit the `sd1` blocks are scaled by.
const SD_ONE: f64 = (1u64 << SD_FRAC_BITS) as f64;

/// First-order sigma-delta per block, walked in `order` (natural for identity). Returns the
/// dequantized weights and `max_t |e_t|/a`.
///
/// The recurrence runs in the checked-integer body [`sd_binary_block`] on the `round(w·2^24)`
/// weights and the `round(a·2^24)` scale; the sign stream and the residual are decided there, so no
/// `f64` copy of the recurrence remains. The quantized value `a·s_t` is rebuilt from the integer
/// sign with the same `f64` block scale `a` the W7 arm measured. A `None` from the integer body is a
/// loud panic, never a silent fallback.
///
/// # Complexity
/// - Temporal: $O(\text{out} \cdot n)$ — one checked integer step per weight plus the roundings.
/// - Spatial: $O(\text{out} \cdot n)$ f64 words for the output, plus $O(n)$ words for the two
///   per-row integer block buffers.
///
/// # Panics
/// Panics if a weight is not finite (the `as i64` cast would otherwise saturate or map NaN to 0), or
/// if [`sd_binary_block`] overflows on a block; with `S = 24` and `f16` weights the state is
/// bounded by `2·group·max|w|·2^24 < 2^60`, far inside `i64`, so this is unreachable for the arm.
fn sd1_row(w: &[f64], out: usize, n: usize, order: &[usize], group: usize) -> (Vec<f64>, f64) {
    let mut wq = vec![0f64; out * n];
    let mut stab = 0.0f64;
    assert!(w.iter().all(|v| v.is_finite()), "sd1: non-finite weight");
    let mut wb: Vec<i64> = vec![0i64; n];
    let mut sb: Vec<i8> = vec![0i8; n];
    for o in 0..out {
        let base = o * n;
        let mut g0 = 0;
        while g0 < n {
            let g1 = (g0 + group).min(n);
            let len = g1 - g0;
            let a = (g0..g1).map(|t| w[base + order[t]].abs()).sum::<f64>() / len as f64;
            let a = a.max(1e-12);
            for t in 0..len {
                wb[t] = (w[base + order[g0 + t]] * SD_ONE).round() as i64;
            }
            let af = (a * SD_ONE).round() as i64;
            let (_e, peak) = sd_binary_block(&wb[..len], af, &mut sb[..len])
                .expect("sigma-delta overflow on a fixed-point block");
            let r = peak as f64 / SD_ONE / a;
            if r > stab {
                stab = r;
            }
            for t in 0..len {
                let s = if sb[t] < 0 { -1.0 } else { 1.0 };
                wq[base + order[g0 + t]] = a * s;
            }
            g0 = g1;
        }
    }
    (wq, stab)
}

/// The GPTQ error-feedback factor `U` (`UᵀU = H⁻¹`, upper-triangular) of the damped Gram
/// `H = gram + 0.01·mean(diag)·I`, dead-column guarded. Uses the public [`ul_cholesky`] and
/// [`upper_tri_inverse`]; this is the private `feedback_hi` path rebuilt from public helpers.
///
/// # Complexity
/// - Temporal: $O(n^3)$ f64 operations ($n^3/6$ for the `UL` Cholesky, $n^3/6$ for the inverse).
/// - Spatial: $O(n^2)$ f64 words ($H$, the factor $V$, and the inverse $U$).
fn feedback_u(gram: &[f64], n: usize) -> Result<Vec<f64>, GptqError> {
    let mean_diag = (0..n).map(|i| gram[i * n + i]).sum::<f64>() / n as f64;
    if mean_diag.is_nan() {
        return Err(GptqError::NotPositiveDefinite);
    }
    let damp = 0.01 * mean_diag;
    let mut h = gram.to_vec();
    for i in 0..n {
        h[i * n + i] += damp;
        if h[i * n + i] == 0.0 {
            h[i * n + i] = 1.0;
        }
    }
    let v = ul_cholesky(&h, n)?;
    upper_tri_inverse(&v, n)
}

/// GPTQ column feedback on the grid `{−a, +a}` in the column order `order`. The Gram and the row are
/// permuted into `order`, the feedback factor `U` is built from the permuted Gram, each column is
/// rounded to `a·sign(w)` (block scale `a` = mean of the current error-fed block), the residual
/// `(w − q)/U[j,j]` is pushed onto the later columns, and the result is returned to natural order.
///
/// # Complexity
/// - Temporal: $O(n^3 + \text{out} \cdot n^2)$ — the factor build plus $\sum_j (n - j) =
///   n(n+1)/2$ multiply-subtracts per row; the permutation copy is $O(n^2)$.
/// - Spatial: $O(n^2 + \text{out} \cdot n)$ f64 words for the permuted Gram, the factor, and the
///   working row.
fn gptq_row(
    w: &[f64],
    out: usize,
    n: usize,
    gram: &[f64],
    order: &[usize],
    group: usize,
) -> Result<Vec<f64>, GptqError> {
    let is_identity = order.iter().enumerate().all(|(i, &p)| i == p);
    let u = if is_identity {
        feedback_u(gram, n)?
    } else {
        let mut gp = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gp[i * n + j] = gram[order[i] * n + order[j]];
            }
        }
        feedback_u(&gp, n)?
    };
    let identity_order = identity(n);
    let perm: &[usize] = if is_identity { &identity_order } else { order };

    let mut wk = vec![0f64; out * n];
    for o in 0..out {
        for i in 0..n {
            wk[o * n + i] = w[o * n + perm[i]];
        }
    }
    let mut wqp = vec![0f64; out * n];
    for o in 0..out {
        let base = o * n;
        let mut g0 = 0;
        while g0 < n {
            let g1 = (g0 + group).min(n);
            let a = mean_abs(&wk, base, g0, g1);
            for j in g0..g1 {
                let wv = wk[base + j];
                let q = a * sign(wv);
                wqp[base + j] = q;
                let err = (wv - q) / u[j * n + j];
                for k in (j + 1)..n {
                    wk[base + k] -= err * u[j * n + k];
                }
            }
            g0 = g1;
        }
    }
    let mut wq = vec![0f64; out * n];
    for o in 0..out {
        for i in 0..n {
            wq[o * n + perm[i]] = wqp[o * n + i];
        }
    }
    Ok(wq)
}

/// The greedy nearest-neighbour column order of the calibration Gram: start at `argmax H_ii`, then
/// repeatedly step to the unvisited `j` minimising `d(cur, j) = H_cur,cur + H_jj − 2H_cur,j` (the
/// squared distance between the activation columns). One permutation per tensor.
///
/// # Complexity
/// - Temporal: $O(n^2)$ distance evaluations — $n$ steps each scanning the remaining columns.
/// - Spatial: $O(n)$ bytes for the visited mask and the order.
fn path_order(gram: &[f64], n: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut cur = 0;
    let mut best = f64::NEG_INFINITY;
    for i in 0..n {
        if gram[i * n + i] > best {
            best = gram[i * n + i];
            cur = i;
        }
    }
    order.push(cur);
    visited[cur] = true;
    for _ in 1..n {
        let mut next = cur;
        let mut nbd = f64::INFINITY;
        for j in 0..n {
            if visited[j] {
                continue;
            }
            let d = gram[cur * n + cur] + gram[j * n + j] - 2.0 * gram[cur * n + j];
            if d < nbd {
                nbd = d;
                next = j;
            }
        }
        order.push(next);
        visited[next] = true;
        cur = next;
    }
    order
}

/// Columns by descending `H_ii` (act order), index as the stable tiebreak.
///
/// # Complexity
/// - Temporal: $O(n \log n)$ comparisons: one stable merge sort of the `n` indices.
/// - Spatial: $O(n)$ words: the `n`-index order plus the sort's `n/2` merge buffer.
fn act_order(gram: &[f64], n: usize) -> Vec<usize> {
    let mut p = identity(n);
    p.sort_by(|&a, &b| {
        gram[b * n + b]
            .partial_cmp(&gram[a * n + a])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    p
}

/// The in-place normalised Walsh–Hadamard transform of one 128-vector: `H₁₂₈/√128`.
fn fwht_block(x: &mut [f64; ONEBIT_GROUP]) {
    let mut h = 1;
    while h < ONEBIT_GROUP {
        let mut i = 0;
        while i < ONEBIT_GROUP {
            for j in i..(i + h) {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let inv = 1.0 / (ONEBIT_GROUP as f64).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Apply the `H₁₂₈/√128` transform to each 128-block of every row. The transform is its own
/// inverse (the Hadamard matrix is symmetric and `R = H/√128` is orthogonal), so the same routine
/// rotates into and out of the transformed domain.
///
/// # Complexity
/// - Temporal: $O(\text{out} \cdot n \log 128)$ — the butterfly is `7` stages over each block.
/// - Spatial: $O(\text{out} \cdot n)$ f64 words for the copy.
fn block_transform(w: &[f64], out: usize, n: usize) -> Vec<f64> {
    let mut t = w.to_vec();
    let nb = n / ONEBIT_GROUP;
    for o in 0..out {
        for b in 0..nb {
            let base = o * n + b * ONEBIT_GROUP;
            let mut blk = [0f64; ONEBIT_GROUP];
            blk.copy_from_slice(&t[base..base + ONEBIT_GROUP]);
            fwht_block(&mut blk);
            t[base..base + ONEBIT_GROUP].copy_from_slice(&blk);
        }
    }
    t
}

/// The congruent Gram `R H R` (`R` the block-diagonal `H₁₂₈/√128`): transform the columns of `H`
/// (right multiply by `R`), then its rows (left multiply by `R`). Orthogonality of `R` makes the
/// output error invariant under the rotation.
///
/// # Complexity
/// - Temporal: $O(n^2 \log 128)$ — `7` butterfly stages over each of the `n` row-blocks and `n`
///   column-blocks.
/// - Spatial: $O(n^2)$ f64 words for the working copy.
fn had_transform_gram(gram: &[f64], n: usize) -> Vec<f64> {
    let mut t = gram.to_vec();
    let nb = n / ONEBIT_GROUP;
    for i in 0..n {
        for b in 0..nb {
            let base = i * n + b * ONEBIT_GROUP;
            let mut blk = [0f64; ONEBIT_GROUP];
            blk.copy_from_slice(&t[base..base + ONEBIT_GROUP]);
            fwht_block(&mut blk);
            t[base..base + ONEBIT_GROUP].copy_from_slice(&blk);
        }
    }
    for j in 0..n {
        for b in 0..nb {
            let mut blk = [0f64; ONEBIT_GROUP];
            for k in 0..ONEBIT_GROUP {
                blk[k] = t[(b * ONEBIT_GROUP + k) * n + j];
            }
            fwht_block(&mut blk);
            for k in 0..ONEBIT_GROUP {
                t[(b * ONEBIT_GROUP + k) * n + j] = blk[k];
            }
        }
    }
    t
}

/// Rotate `w` and `gram` into the Hadamard domain.
///
/// # Errors
/// Returns [`GptqError::ShapeMismatch`] if `n` is not a whole number of 128-column blocks.
fn had_transform(
    w: &[f64],
    gram: &[f64],
    out: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), GptqError> {
    if !n.is_multiple_of(ONEBIT_GROUP) {
        return Err(GptqError::ShapeMismatch);
    }
    Ok((block_transform(w, out, n), had_transform_gram(gram, n)))
}
