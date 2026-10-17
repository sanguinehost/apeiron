//! Symmetric PSD eigendecomposition and the exact rank-`rank` corrector in the metric `Hl` — the
//! `round_f16=False` path of `lowrank_corrector` in the removed one-bit fold transfer
//! (`fold_transfer.py:560–583`, read-only spec outside this repository): `M = (ΔW·Hl)·ΔWᵀ`, the
//! `rank` largest eigenpairs of `½(M + Mᵀ)`, `Corr = U·(Uᵀ ΔW)`, `top_sum = Σ_k max(λ_k, 0)`.
//!
//! Carrier: Q32.32 `i64`/`i128`; `f64` only at the `pub` boundaries, across one exact power-of-two
//! scale and one `to_grid`/`from_grid` rounding per entry. [`top_eigh`] supplies the spectrum, the
//! pooled `gemm` kernels the products, and the `O(n²)` symmetrisation is serial (≈10⁶ word
//! operations at `o = 896`, about one pool dispatch's cost).
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
use crate::learn::solve::q32::{
    fill_scale, from_grid, panel_shift, rshift_rne, to_grid, GRID_BOUND, INPUT_BOUND,
};

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

/// `2⁻²⁸`: the [`fill_scale`] fill `2³¹` divided down below [`top_eigh`]'s `2³⁶`-raw input bound.
const FILL_DOWN: f64 = 1.0 / 268_435_456.0;

/// The exact power of two `s` with `max_mag·s ≤ 8` (`2⁻²⁸` when `max_mag ≤ 0`), exact to undo.
#[must_use]
pub fn gram_scale(max_mag: f64) -> f64 {
    fill_scale(max_mag) * FILL_DOWN
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

/// `x` as a Q32.32 panel and the exponent `k` of the exact `2⁻ᵏ` it was shifted by — the
/// [`panel_shift`] bringing the largest entry under [`INPUT_BOUND`]; `NotConverged` on a
/// non-finite entry.
///
/// # Complexity
/// `O(len)`: one magnitude scan, one shift and one rounding per entry.
fn panel_f64(x: &[f64]) -> Result<(Vec<i64>, i32), EighError> {
    let max = x.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let k = panel_shift(max);
    let scale = 2.0f64.powi(-k);
    let q = x
        .iter()
        .map(|&v| to_grid(v * scale, INPUT_BOUND))
        .collect::<Option<Vec<i64>>>()
        .ok_or(EighError::NotConverged)?;
    Ok((q, k))
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

/// The `f64` read-back: `evals[k] = μ_k/(2³²·scale)` and the scale-free eigenvectors.
/// # Complexity
/// `O(n + n²)` conversions; `O(n²)` space.
fn read_back(top: &TopEig, scale: f64) -> (Vec<f64>, Vec<f64>) {
    let evals = top.evals.iter().map(|&mu| from_grid(mu) / scale).collect();
    let evecs = top.evecs.iter().map(|&v| from_grid(v)).collect();
    (evals, evecs)
}

/// Symmetric eigendecomposition of the PSD `a` (row-major `[n, n]`): eigenvalues ascending,
/// `eigenvectors` row-major `[n, n]` with column `k` for `eigenvalues[k]` (`Qᵀ Q = I`,
/// largest-magnitude entry positive). Only `½(a + aᵀ)` is decomposed, as the Python does.
///
/// # Complexity
/// `O(MAX_ITERS·n³)` checked multiply-accumulates (`M·Q`, the Rayleigh quotient, the two rotation
/// products and the integer Jacobi), pooled by row chunks; space `O(n²)`. `r = n` carries the full
/// spectrum in the first Rayleigh–Ritz step.
///
/// # Errors
/// [`EighError::ShapeMismatch`] on a length mismatch; [`EighError::NotConverged`] on a non-finite
/// entry, an out-of-grid entry, a non-PSD spectrum, or the iteration bound.
pub fn eigh(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), EighError> {
    if a.len() != n * n {
        return Err(EighError::ShapeMismatch);
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if a.iter().any(|x| !x.is_finite()) {
        return Err(EighError::NotConverged);
    }
    let max_mag = a.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let scale = gram_scale(max_mag);
    let mut q = a
        .iter()
        .map(|&v| to_grid(v * scale, INPUT_BOUND))
        .collect::<Option<Vec<i64>>>()
        .ok_or(EighError::NotConverged)?;
    symmetrise(&mut q, n)?;
    // PSD implies a non-negative diagonal; a word below `-1` is outside the one-word bound.
    if (0..n).any(|i| q[i * n + i] < -1) {
        return Err(EighError::NotConverged);
    }
    let top = top_eigh(&q, n, n).map_err(eig_err)?;
    check_psd(&top.evals, n)?;
    Ok(read_back(&top, scale))
}

/// The `o × o` Gram `M = (ΔW·Hl)·ΔWᵀ`: two pooled [`gemm_nn`]/[`gemm_nt`] products on Q32.32
/// words, each operand panel shifted by an exact `2⁻ᵏ` ([`panel_f64`], [`panel`]) and the composed
/// scale undone once on the read-back; the `O(o²)` symmetrisation is serial. Returned in real
/// units, row-major `[o, o]`; `Hl` need not be symmetric.
///
/// # Complexity
/// `O(o·n² + o²·n)` multiply-accumulates on the engine pool, the serial `O(o²)` symmetrisation and
/// `O(o·n + o²)` panel shifts and conversions; space `O(o·n + o²)`.
///
/// # Errors
/// [`EighError::ShapeMismatch`] on a length mismatch; otherwise [`EighError::NotConverged`] for a
/// non-finite entry or a product's refusal (an entry leaving the grid or the `i128` range).
pub fn eigen_gram(delta_w: &[f64], hl: &[f64], o: usize, n: usize) -> Result<Vec<f64>, EighError> {
    if delta_w.len() != o * n || hl.len() != n * n {
        return Err(EighError::ShapeMismatch);
    }
    if o == 0 || n == 0 {
        return Ok(vec![0.0f64; o * o]);
    }
    let (dw, ka) = panel_f64(delta_w)?;
    let (hlq, kb) = panel_f64(hl)?;
    let (w1, k1) =
        panel(&gemm_nn(&dw, &hlq, o, n, n).map_err(gemm_err)?).ok_or(EighError::NotConverged)?;
    let mut m = gemm_nt(&w1, &dw, o, n, o).map_err(gemm_err)?;
    symmetrise(&mut m, o)?;
    let back = 2.0f64.powi(2 * ka + kb + k1);
    Ok(m.into_iter().map(|w| from_grid(w) * back).collect())
}

/// The exact rank-`rank` corrector in the metric `Hl`: the stored factors and `Corr = U·V`.
#[derive(Clone, Debug)]
pub struct Corrector {
    /// The dense correction `U·V`, row-major `[o, n]`.
    pub corr: Vec<f64>,
    /// `Σ_k max(λ_k, 0)` over the top `rank` eigenvalues — the first-order objective drop attained.
    pub top_sum: f64,
    /// The top `rank` eigenvectors of `M`, row-major `[o, rank]`, ascending; column `rank−1` is
    /// the largest.
    pub u: Vec<f64>,
    /// `V = Uᵀ·ΔW`, row-major `[rank, n]`, so `Corr = U·V`.
    pub v: Vec<f64>,
}

/// The exact rank-`rank` corrector of `ΔW` in the metric `Hl` — the `round_f16=False,
/// u8_corr=False` branch of `lowrank_corrector`: `Corr = U·(Uᵀ ΔW)` for the `rank` largest
/// eigenvectors (columns) of `M`, the factors `U`, `V`, and `top_sum = Σ max(λ_k, 0)` over them.
/// `Corr = (I − U Uᵀ) ΔW` minimises `tr(R·Hl·Rᵀ)` over rank-`rank` projections (Ky Fan /
/// Eckart–Young).
///
/// # Complexity
/// `O(o·n² + o²·n + o²·rank·iters + rank·o·n)`; space `O(o·n + o² + o·rank + rank·n)`.
///
/// # Errors
/// A length mismatch, `rank > o`, or the refusal of [`eigen_gram`] or [`top_eigh`].
pub fn lowrank_corrector_exact(
    delta_w: &[f64],
    hl: &[f64],
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
            corr: vec![0.0f64; o * n],
            top_sum: 0.0,
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
    // `V = Uᵀ·ΔW` and `Corr = U·V` on the pooled word kernels; the `ΔW` panel's exact `2⁻ᵏ` is
    // undone once on the read-back.
    let uq: Vec<i64> = u
        .iter()
        .map(|&x| to_grid(x, GRID_BOUND).ok_or(EighError::NotConverged))
        .collect::<Result<_, _>>()?;
    let (dw, kd) = panel_f64(delta_w)?;
    let ut: Vec<i64> = (0..rank * o).map(|t| uq[(t % o) * rank + t / o]).collect();
    let (v, kv) =
        panel(&gemm_nn(&ut, &dw, rank, o, n).map_err(gemm_err)?).ok_or(EighError::NotConverged)?;
    let corr = gemm_nn(&uq, &v, o, rank, n).map_err(gemm_err)?;
    let back = 2.0f64.powi(kd + kv);
    Ok(Corrector {
        corr: corr.iter().map(|&w| from_grid(w) * back).collect(),
        top_sum,
        u,
        v: v.iter().map(|&w| from_grid(w) * back).collect(),
    })
}
