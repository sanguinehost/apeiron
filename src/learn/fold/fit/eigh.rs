//! Symmetric PSD eigendecomposition and the exact rank-`rank` corrector in the metric `Hl` — the
//! `round_f16=False` path of `lowrank_corrector` in the removed one-bit fold transfer
//! (`fold_transfer.py:560–583`, read-only spec outside this repository): `M = (ΔW·Hl)·ΔWᵀ`, the
//! `rank` largest eigenpairs of `½(M + Mᵀ)`, `Corr = U·(Uᵀ ΔW)`, `top_sum = Σ_k max(λ_k, 0)`.
//!
//! Carrier: Q32.32 `i64`/`i128` in and out. [`eigh`] takes and returns words; the exact power-of-two
//! panel shift is applied to the words and undone on the returned eigenvalue words. [`top_eigh`]
//! supplies the spectrum, the pooled `gemm` kernels the products, and the `O(n²)` symmetrisation is
//! serial (≈10⁶ word operations at `o = 896`, about one pool dispatch's cost).
//!
//! Accepted domain: `a` is PSD — both callers pass a Gram matrix. A diagonal word below `-1`, or a
//! spectrum below [`check_psd`]'s bound, is [`EighError::NotConverged`], never shifted. The
//! recurrence, spectrum and corrector are `tested` (`tests/learn_fold_eigh.rs`), not `proved`;
//! `Corr`'s trace-optimality is Ky Fan / Eckart–Young.

#![allow(
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the transfer's arithmetic indexes row-major `[r, c]` buffers by `i/j/k`; the pedantic \
              lints stay on for the rest of the crate."
)]

use crate::learn::fold::fit::corrector::{eigen_parts, CorrectorError};
use crate::learn::fold::stats::parts::panel;
use crate::learn::solve::eig::{top_eigh, EigError, TopEig};
use crate::learn::solve::gemm::{gemm_nn, gemm_nt, GemmError};
use crate::learn::solve::q32::{ilog2_pos, rshift_rne};

/// A refusal from the symmetric eigendecomposition or the exact low-rank corrector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EighError {
    /// A slice length disagrees with the declared shape (`n·n`, or `o·n` for `ΔW`).
    ShapeMismatch,
    /// The input left the accepted domain (not PSD, a non-finite or out-of-grid entry), or the
    /// eigenstage missed its convergence test, or a checked step left range.
    NotConverged,
    /// The requested `rank` exceeds the row count `o` of `ΔW`.
    RankOutOfRange,
}

/// The exact power-of-two exponent `e` with `max·2^e ≤ 2³⁵ − 16` maximal, the word form of the
/// deleted `fill_scale(max)·2⁻²⁸` boundary (`2³¹ − 1` at `2⁻²⁸` is `2³⁵ − 16` raw), so a word panel
/// reproduces the old scaled input bit for bit from grid-exact operands. `e = 0` for the zero
/// matrix. `O(1)`.
fn input_scale(max: u128) -> i32 {
    const BOUND: u128 = (1u128 << 35) - 16;
    let Some(bits) = ilog2_pos(max) else {
        return 0;
    };
    let mut e = 34 - bits.cast_signed();
    loop {
        let fits = if e >= 0 {
            u32::try_from(e)
                .ok()
                .and_then(|k| max.checked_shl(k))
                .is_some_and(|v| v <= BOUND)
        } else {
            let k = u32::try_from(-e).unwrap_or(u32::MAX);
            max.checked_shr(k).is_some_and(|v| v <= BOUND)
        };
        if fits {
            return e;
        }
        e -= 1;
    }
}

/// The exact power-of-two exponent `e` with the word maximum `max·2^e ∈ (2³⁴, 2³⁵]` — the total
/// shift [`eigen_parts`] applies to the raw Gram words before [`top_eigh`] (`34 − ⌊log₂ max⌋` and
/// `0` for the zero matrix, which [`eigen_parts`] special-cases). `O(len)`.
#[must_use]
pub fn gram_scale(m: &[i64]) -> i32 {
    let max = m
        .iter()
        .fold(0u128, |a, &v| a.max(u128::from(v.unsigned_abs())));
    ilog2_pos(max).map_or(0, |b| 34 - b.cast_signed())
}

/// A [`GemmError`] from a product whose shape and depth were already validated here.
fn gemm_err(_: GemmError) -> EighError {
    EighError::NotConverged
}

/// A [`CorrectorError`] from [`eigen_parts`], whose shape and rank were already validated here.
fn parts_err(e: CorrectorError) -> EighError {
    match e {
        CorrectorError::ShapeMismatch => EighError::ShapeMismatch,
        CorrectorError::RankOutOfRange => EighError::RankOutOfRange,
        _ => EighError::NotConverged,
    }
}

/// An [`EigError`] whose shape and rank were already validated at this call site.
fn eig_err(_: EigError) -> EighError {
    EighError::NotConverged
}

/// `v·2⁻ᵏ` (`k ≥ 0`) rounded to nearest with ties away from zero — the rounding `to_grid` applied
/// to the pre-flip `f64` operands, so the panel words match the old entry bit-for-bit; `None` when
/// the result leaves `i64`. `O(1)`.
fn rshift_away(v: i64, k: u32) -> Option<i64> {
    if k == 0 {
        return Some(v);
    }
    if k > 126 {
        return None;
    }
    let (d, half) = (1_i128 << k, 1_i128 << (k - 1));
    let x = i128::from(v);
    let (q, r) = (x / d, x % d);
    let mag = if r < 0 { -r } else { r };
    let step = if mag >= half { r.signum() } else { 0 };
    i64::try_from(q + step).ok()
}

/// `v·2^e` on the grid with the same half-away rounding; `None` when it leaves `i64`. `O(1)`.
fn scale_away(v: i64, e: i32) -> Option<i64> {
    if e >= 0 {
        let w = i128::from(v).checked_shl(u32::try_from(e).ok()?)?;
        return i64::try_from(w).ok();
    }
    rshift_away(v, u32::try_from(-e).ok()?)
}

/// `x` as a Q32.32 panel and the exponent `k` of the exact `2⁻ᵏ` it was shifted by — the
/// [`panel_shift`](crate::learn::solve::q32::panel_shift) bringing the largest entry under the
/// primitive's input bound, on the words directly; `None` when a shift leaves `i64`.
///
/// # Complexity
/// `O(len)`: one magnitude scan, one shift per entry.
fn panel_words(x: &[i64]) -> Option<(Vec<i64>, i32)> {
    let max = x
        .iter()
        .fold(0u128, |a, &v| a.max(u128::from(v.unsigned_abs())));
    let k = match ilog2_pos(max) {
        Some(b) if max > 1u128 << 35 => {
            if max == 1u128 << b {
                b.cast_signed() - 35
            } else {
                b.cast_signed() + 1 - 35
            }
        }
        _ => 0,
    };
    let q = x
        .iter()
        .map(|&v| {
            if k >= 0 {
                rshift_away(v, u32::try_from(k).ok()?)
            } else {
                i64::try_from(i128::from(v).checked_shl(k.unsigned_abs())?).ok()
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some((q, k))
}

/// Replace each transposed pair of the row-major `[n, n]` word matrix `q` by
/// `round((q_ij + q_ji)/2)`, ties to even, so `q` is exactly symmetric on the grid.
///
/// # Complexity
/// `O(n²)` serial word operations (≈10⁶ at `n² ≤ 896²`, below the pool threshold); space `O(1)`.
fn symmetrise(q: &mut [i64], n: usize) -> Result<(), EighError> {
    for i in 0..n {
        for j in 0..=i {
            let mean = rshift_rne(i128::from(q[i * n + j]) + i128::from(q[j * n + i]), 1)
                .and_then(|m| i64::try_from(m).ok())
                .ok_or(EighError::NotConverged)?;
            q[i * n + j] = mean;
            q[j * n + i] = mean;
        }
    }
    Ok(())
}

/// The domain's non-negativity test: each symmetrised entry is within one grid word of the exact
/// scaled entry (an eigenvalue shift of at most `n` words), and [`top_eigh`] returns each
/// eigenvalue within `√n·λ_max·2⁻²⁰`; a smallest word below `−n·(λ_max·2⁻²⁰ + 1)` is a negative
/// eigenvalue outside the grid noise.
///
/// # Complexity
/// `O(1)`.
fn check_psd(evals: &[i64], n: usize) -> Result<(), EighError> {
    let n = i128::try_from(n).map_err(|_| EighError::NotConverged)?;
    let lam_max = i128::from(evals[evals.len() - 1].max(0));
    if i128::from(evals[0]) < -n * ((lam_max >> 20).max(1) + 1) {
        return Err(EighError::NotConverged);
    }
    Ok(())
}

/// Symmetric eigendecomposition of the PSD `a` (row-major `[n, n]`, Q32.32 words): eigenvalues
/// ascending as words, `eigenvectors` row-major `[n, n]` with column `k` for `eigenvalues[k]`
/// (`Qᵀ Q = I`, largest-magnitude entry positive). Only `½(a + aᵀ)` is decomposed, as the Python
/// does. The input panel is the exact `2^e` shift of [`input_scale`], undone on the eigenvalue
/// words; the eigenvectors are scale-free.
///
/// # Complexity
/// `O(MAX_ITERS·n³)` checked multiply-accumulates (`M·Q`, the Rayleigh quotient, the two rotation
/// products and the integer Jacobi), pooled by row chunks; space `O(n²)`. `r = n` carries the full
/// spectrum in the first Rayleigh–Ritz step.
///
/// # Errors
/// [`EighError::ShapeMismatch`] on a length mismatch; [`EighError::NotConverged`] on a scaled entry
/// or eigenvalue that leaves `i64`, a non-PSD spectrum, or the iteration bound.
pub fn eigh(a: &[i64], n: usize) -> Result<(Vec<i64>, Vec<i64>), EighError> {
    if a.len() != n * n {
        return Err(EighError::ShapeMismatch);
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let max = a
        .iter()
        .fold(0u128, |m, &v| m.max(u128::from(v.unsigned_abs())));
    let e = input_scale(max);
    let mut q = a
        .iter()
        .map(|&v| scale_away(v, e))
        .collect::<Option<Vec<i64>>>()
        .ok_or(EighError::NotConverged)?;
    symmetrise(&mut q, n)?;
    // PSD implies a non-negative diagonal; a word below `-1` is outside the one-word bound.
    if (0..n).any(|i| q[i * n + i] < -1) {
        return Err(EighError::NotConverged);
    }
    let TopEig { evals, evecs, .. } = top_eigh(&q, n, n).map_err(eig_err)?;
    check_psd(&evals, n)?;
    // `μ·2⁻ᵉ` is the eigenvalue word in the input's own units; one rounding, ties away from zero,
    // as the deleted `to_grid` read-back applied.
    let evals = evals
        .iter()
        .map(|&mu| scale_away(mu, -e))
        .collect::<Option<Vec<i64>>>()
        .ok_or(EighError::NotConverged)?;
    Ok((evals, evecs))
}

/// The `o × o` Gram `M = (ΔW·Hl)·ΔWᵀ`: two pooled [`gemm_nn`]/[`gemm_nt`] products on Q32.32
/// words, each operand panel shifted by an exact `2⁻ᵏ` ([`panel_words`], [`panel`]) and the
/// composed scale returned as the exponent `K` of the exact `2^K` undo (`M_real·2³² = w·2^K`); the
/// `O(o²)` symmetrisation is serial. `Hl` need not be symmetric.
///
/// # Complexity
/// `O(o·n² + o²·n)` multiply-accumulates on the engine pool, the serial `O(o²)` symmetrisation and
/// `O(o·n + o²)` panel shifts; space `O(o·n + o²)`.
///
/// # Errors
/// [`EighError::ShapeMismatch`] on a length mismatch; otherwise [`EighError::NotConverged`] for a
/// product's refusal or a shift that leaves `i64`.
pub fn eigen_gram(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
) -> Result<(Vec<i64>, i32), EighError> {
    if delta_w.len() != o * n || hl.len() != n * n {
        return Err(EighError::ShapeMismatch);
    }
    if o == 0 || n == 0 {
        return Ok((vec![0i64; o * o], 0));
    }
    let (dw, ka) = panel_words(delta_w).ok_or(EighError::NotConverged)?;
    let (hlq, kb) = panel_words(hl).ok_or(EighError::NotConverged)?;
    let (w1, k1) =
        panel(&gemm_nn(&dw, &hlq, o, n, n).map_err(gemm_err)?).ok_or(EighError::NotConverged)?;
    let mut m = gemm_nt(&w1, &dw, o, n, o).map_err(gemm_err)?;
    symmetrise(&mut m, o)?;
    Ok((m, 2 * ka + kb + k1))
}

/// The exact rank-`rank` corrector in the metric `Hl`: the stored factors and `Corr = U·V`, all
/// Q32.32 words.
#[derive(Clone, Debug)]
pub struct Corrector {
    /// The dense correction `U·V`, row-major `[o, n]`, Q32.32 words.
    pub corr: Vec<i64>,
    /// `Σ_k max(λ_k, 0)` over the top `rank` eigenvalues — the first-order objective drop attained,
    /// as a Q32.32 word.
    pub top_sum: i64,
    /// The top `rank` eigenvectors of `M`, row-major `[o, rank]`, ascending; column `rank−1` is
    /// the largest, as Q32.32 words.
    pub u: Vec<i64>,
    /// `V = Uᵀ·ΔW`, row-major `[rank, n]`, so `Corr = U·V`, as Q32.32 words.
    pub v: Vec<i64>,
}

/// The exact rank-`rank` corrector of `ΔW` in the metric `Hl` — the `round_f16=False,
/// u8_corr=False` branch of `lowrank_corrector`: `Corr = U·(Uᵀ ΔW)` for the `rank` largest
/// eigenvectors (columns) of `M`, the factors `U`, `V`, and `top_sum = Σ max(λ_k, 0)` over them.
/// `Corr = (I − U Uᵀ) ΔW` minimises `tr(R·Hl·Rᵀ)` over rank-`rank` projections (Ky Fan /
/// Eckart–Young). `ΔW` and `Hl` are Q32.32 words; every returned buffer is a word.
///
/// # Complexity
/// `O(o·n² + o²·n + o²·rank·iters + rank·o·n)`; space `O(o·n + o² + o·rank + rank·n)`.
///
/// # Errors
/// A length mismatch, `rank > o`, or the refusal of [`eigen_gram`] or [`top_eigh`].
pub fn lowrank_corrector_exact(
    delta_w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<Corrector, EighError> {
    if delta_w.len() != o * n || hl.len() != n * n {
        return Err(EighError::ShapeMismatch);
    }
    if rank > o {
        return Err(EighError::RankOutOfRange);
    }
    if rank == 0 {
        return Ok(Corrector {
            corr: vec![0i64; o * n],
            top_sum: 0,
            u: Vec::new(),
            v: Vec::new(),
        });
    }
    let (u, top_sum) = eigen_parts(delta_w, hl, o, n, rank).map_err(parts_err)?;
    if n == 0 {
        return Ok(Corrector {
            corr: Vec::new(),
            top_sum,
            u,
            v: Vec::new(),
        });
    }
    // `V = Uᵀ·ΔW` and `Corr = U·V` on the pooled word kernels; the panels' exact `2⁻ᵏ` is undone
    // once per output word.
    let (dw, kd) = panel_words(delta_w).ok_or(EighError::NotConverged)?;
    let ut: Vec<i64> = (0..rank * o).map(|t| u[(t % o) * rank + t / o]).collect();
    let (v, kv) =
        panel(&gemm_nn(&ut, &dw, rank, o, n).map_err(gemm_err)?).ok_or(EighError::NotConverged)?;
    let corr = gemm_nn(&u, &v, o, rank, n).map_err(gemm_err)?;
    let back = kd + kv;
    let scaled = |x: &[i64]| -> Result<Vec<i64>, EighError> {
        x.iter()
            .map(|&v| scale_away(v, back).ok_or(EighError::NotConverged))
            .collect()
    };
    Ok(Corrector {
        corr: scaled(&corr)?,
        top_sum,
        u,
        v: scaled(&v)?,
    })
}
