//! The rounded rank-`rank` corrector in the metric `Hl`: the fp16 and u8 carriers of
//! `lowrank_corrector` in the removed one-bit fold transfer (`fold_transfer.py:560–582`, read-only
//! spec outside this repository).
//!
//! [`crate::learn::fold::fit::eigh::lowrank_corrector_exact`] carries the `round_f16=False, u8_corr=False`
//! branch (`Corr = U·(Uᵀ ΔW)` with `U` the top-`rank` eigenvectors of `M = (ΔW·Hl)·ΔWᵀ`). This
//! module ports the two rounded branches that consume those same factors:
//!
//! - [`Mode::F16`] — `Uh = f16r(U)`, then `Vh = f16r((Uhᵀ Uh + 1e-6 I)⁻¹ Uhᵀ ΔW)`, the Python's
//!   `round_f16=True` branch.
//! - [`Mode::U8`] — each column of `U` and each row of the ridge solution `V` quantized by
//!   `quant_c_u8(·, bits=8)` (an `f16` `(c_min, c_max)` header plus 8-bit affine codes), the
//!   `u8_corr=True` branch.
//! - [`Mode::Exact`] — the unrounded `U`, `V = Uᵀ ΔW` of [`crate::learn::fold::fit::eigh`].
//!
//! All three return `Corr = Uh·Vh`, the bit count of the stored factors, `top_sum = Σ_k max(λ_k, 0)`
//! over the top `rank` eigenvalues of `M`, and — when `factors` is requested — the stored `(Uh, Vh)`
//! so a consumer can train the store itself.
//!
//! # Reuse
//!
//! The `M` Gram is built here and its eigendecomposition is the shared
//! [`crate::learn::fold::fit::eigh::eigh`] (cyclic Jacobi); the rank×rank ridge system is
//! [`crate::learn::ridge::solve_spd`] (Cholesky). The `f16` round-trip and the u8 quantizer are
//! [`crate::learn::fold::fit::quant::f16r`] and [`crate::learn::fold::fit::quant::quant_c_u8`].
//!
//! # Deviations from the Python
//!
//! - The Python's `rank == 0` (`evecs[:, -0:]` selects *all* columns) returns the empty corrector
//!   (`Corr = 0`, zero bits, empty factors), the reading
//!   [`crate::learn::fold::fit::eigh::lowrank_corrector_exact`] gives it; no caller passes `rank == 0`.
//! - `torch.linalg.solve` on the possibly-indefinite `Uhᵀ Uh + 1e-6 I` is replaced by the SPD
//!   [`solve_spd`](crate::learn::ridge::solve_spd); a non-positive pivot is refused as
//!   [`CorrectorError::NotPositiveDefinite`] rather than silently producing a result.
//! - The Python broadcasts a non-finite input through the decomposition; here a non-finite `ΔW` or
//!   `Hl` entry is refused as [`CorrectorError::NonFinite`].
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_corrector.rs`). No extraction or refinement links these bodies to a
//! Lean theorem, so nothing here is `proved`.

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the row-major `[o, n]`/`[rank, n]` kernels index by `i/j/k/p` against the spec's \
              `DeltaW[i, j]`/`Uh`/`Vh` arithmetic; `rank as u64` is a deliberate lossy dimension \
              cast in the bit-count formula; the pedantic lints stay on for the rest of the crate."
)]

use crate::learn::fold::fit::eigh::{eigh, lowrank_corrector_exact, EighError};
use crate::learn::fold::fit::quant::{f16r, quant_c_u8};
use crate::learn::ridge::solve_spd;

/// The `1e-6` ridge added to the rank×rank Gram `Uhᵀ Uh` before the `V` solve, the Python's
/// `torch.eye(rank) * 1e-6`.
const RIDGE: f64 = 1e-6;

/// The `bits=8` width of the u8 affine carrier (`quant_c_u8`).
const U8_BITS: u32 = 8;

/// Which carrier `lowrank_corrector` rounds the factors into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Unrounded `U`, `V = Uᵀ ΔW` (the Python's `round_f16=False, u8_corr=False`).
    Exact,
    /// Binary16 rounding of `U` and of the ridge solution `V` (the Python's `round_f16=True`).
    F16,
    /// 8-bit affine quantization of each column of `U` and each row of the ridge solution `V`.
    U8,
}

/// A refusal from the rounded low-rank corrector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectorError {
    /// A slice length disagrees with the declared shape (`o·n` for `ΔW`, `n·n` for `Hl`), or a
    /// quantized column/row was empty.
    ShapeMismatch,
    /// The requested `rank` exceeds the row count `o` of `ΔW`.
    RankOutOfRange,
    /// An entry of `ΔW` or `Hl` is not finite.
    NonFinite,
    /// The underlying eigendecomposition did not converge within its sweep bound.
    NotConverged,
    /// The rank×rank ridge system `Uhᵀ Uh + 1e-6 I` was not positive-definite at some pivot (only
    /// reachable from a degenerate rounded `Uh`).
    NotPositiveDefinite,
}

impl From<EighError> for CorrectorError {
    fn from(e: EighError) -> Self {
        match e {
            EighError::ShapeMismatch => Self::ShapeMismatch,
            EighError::RankOutOfRange => Self::RankOutOfRange,
            EighError::NotConverged => Self::NotConverged,
        }
    }
}

/// The stored factors `(Uh, Vh)` of a rounded corrector, with `Corr = Uh·Vh`.
#[derive(Clone, Debug)]
pub struct Factors {
    /// `Uh`, row-major `[o, rank]` (column `k` is the `k`-th stored basis vector).
    pub u: Vec<f64>,
    /// `Vh`, row-major `[rank, n]`.
    pub v: Vec<f64>,
}

/// A rank-`rank` corrector in the metric `Hl`: the dense correction, its bit count, the top-`rank`
/// eigenvalue sum, and (optionally) the stored factors.
#[derive(Clone, Debug)]
pub struct LowrankCorrector {
    /// The dense correction `Corr = Uh·Vh`, row-major `[o, n]`.
    pub corr: Vec<f64>,
    /// The stored-factor bit count: `16·rank·(o + n)` for [`Mode::Exact`]/[`Mode::F16`],
    /// `rank·(8·(o + n) + 64)` for [`Mode::U8`].
    pub bits: u64,
    /// `Σ_k max(λ_k, 0)` over the top `rank` eigenvalues of `M = (ΔW·Hl)·ΔWᵀ`.
    pub top_sum: f64,
    /// The stored factors, present iff the caller requested them.
    pub factors: Option<Factors>,
}

/// The rank-`rank` corrector of `ΔW` in the metric `Hl`, rounded into the carrier `mode`.
///
/// `ΔW` is row-major `[o, n]`, `Hl` row-major `[n, n]`. Builds `M = (ΔW·Hl)·ΔWᵀ`, takes its top
/// `rank` eigenvectors as `U` by [`eigh`], and returns `Corr = Uh·Vh` for the carrier:
/// [`Mode::Exact`] `(U, Uᵀ ΔW)`, [`Mode::F16`] `Uh = f16r(U)`, `Vh = f16r(solve(Uhᵀ Uh + 1e-6 I,
/// Uhᵀ ΔW))`, [`Mode::U8`] the same ridge solve on the column-quantized `Uh`, with each row of `V`
/// then quantized. Returns the bit count of the stored factors and `top_sum`. `factors` also returns
/// the stored `(Uh, Vh)`.
///
/// # Complexity
///
/// Time `O(o·n² + o²·n + o³·sweeps + rank·o·n + rank²·(o + n) + rank³)` — the `ΔW·Hl` product
/// `O(o·n²)`, the `M` product `O(o²·n)`, the eigendecomposition `O(o³·sweeps)` ([`eigh`]), the
/// rank×rank ridge Gram/right-hand side `O(rank²·o + rank·o·n)`, the `Cholesky` solve
/// `O(rank³ + rank²·n)`, and the `Corr` product `O(rank·o·n)`; [`Mode::Exact`] only needs the
/// eigenvector slice. Space `O(o·n + o² + o·rank + rank·n)`.
///
/// # Errors
///
/// [`CorrectorError::ShapeMismatch`] if `ΔW` is not `o·n` or `Hl` is not `n·n`;
/// [`CorrectorError::RankOutOfRange`] if `rank > o`; [`CorrectorError::NonFinite`] if any `ΔW` or
/// `Hl` entry is not finite; otherwise the refusal of the underlying [`eigh`] or ridge solve.
pub fn lowrank_corrector(
    delta_w: &[f64],
    hl: &[f64],
    o: usize,
    n: usize,
    rank: usize,
    mode: Mode,
    factors: bool,
) -> Result<LowrankCorrector, CorrectorError> {
    if delta_w.len() != o * n || hl.len() != n * n {
        return Err(CorrectorError::ShapeMismatch);
    }
    if rank > o {
        return Err(CorrectorError::RankOutOfRange);
    }
    if delta_w.iter().any(|x| !x.is_finite()) || hl.iter().any(|x| !x.is_finite()) {
        return Err(CorrectorError::NonFinite);
    }
    if rank == 0 {
        return Ok(LowrankCorrector {
            corr: vec![0.0; o * n],
            bits: 0,
            top_sum: 0.0,
            factors: factors.then_some(Factors {
                u: Vec::new(),
                v: Vec::new(),
            }),
        });
    }

    if mode == Mode::Exact {
        let c = lowrank_corrector_exact(delta_w, hl, o, n, rank)?;
        return Ok(LowrankCorrector {
            corr: c.corr,
            bits: f16_bits(o, n, rank),
            top_sum: c.top_sum,
            factors: factors.then_some(Factors { u: c.u, v: c.v }),
        });
    }

    let (u, top_sum) = eigen_parts(delta_w, hl, o, n, rank)?;
    let (uh, vh, bits) = match mode {
        Mode::F16 => {
            let uh: Vec<f64> = u.iter().map(|&x| f16r(x)).collect();
            let v = ridge_solve(&uh, delta_w, o, n, rank)?;
            let vh: Vec<f64> = v.iter().map(|&x| f16r(x)).collect();
            (uh, vh, f16_bits(o, n, rank))
        }
        Mode::U8 => {
            // Column `k` of `U` is quantized independently; `quant_c_u8` returns the grid value.
            let mut uh = vec![0.0f64; o * rank];
            for k in 0..rank {
                let col: Vec<f64> = (0..o).map(|i| u[i * rank + k]).collect();
                let (q, _) =
                    quant_c_u8(&col, U8_BITS).map_err(|_| CorrectorError::ShapeMismatch)?;
                for i in 0..o {
                    uh[i * rank + k] = q[i];
                }
            }
            let v = ridge_solve(&uh, delta_w, o, n, rank)?;
            // Row `k` of `V` is quantized independently.
            let mut vh = vec![0.0f64; rank * n];
            for k in 0..rank {
                let (q, _) = quant_c_u8(&v[k * n..k * n + n], U8_BITS)
                    .map_err(|_| CorrectorError::ShapeMismatch)?;
                vh[k * n..k * n + n].copy_from_slice(&q);
            }
            (uh, vh, u8_bits(o, n, rank))
        }
        Mode::Exact => unreachable!("handled above"),
    };
    let corr = product(&uh, &vh, o, rank, n);
    Ok(LowrankCorrector {
        corr,
        bits,
        top_sum,
        factors: factors.then_some(Factors { u: uh, v: vh }),
    })
}

/// `16·rank·(o + n)`: the stored-factor bit count of the [`Mode::Exact`] and [`Mode::F16`] carriers
/// (`rank` basis vectors of `o` and `rank` rows of `n`, each a binary16 scalar). Saturating in
/// `u64`. Time `O(1)`.
fn f16_bits(o: usize, n: usize, rank: usize) -> u64 {
    let per = 16u64.saturating_mul((o as u64).saturating_add(n as u64));
    (rank as u64).saturating_mul(per)
}

/// `rank·(8·(o + n) + 64)`: the stored-factor bit count of the [`Mode::U8`] carrier — each of the
/// `rank` columns of `Uh` and rows of `Vh` is an 8-bit code per element plus a 32-bit
/// `(c_min, c_max)` `f16` header. Saturating in `u64`. Time `O(1)`.
fn u8_bits(o: usize, n: usize, rank: usize) -> u64 {
    let per = 8u64
        .saturating_mul((o as u64).saturating_add(n as u64))
        .saturating_add(64);
    (rank as u64).saturating_mul(per)
}

/// The top-`rank` eigenvectors of `M = (ΔW·Hl)·ΔWᵀ` and `Σ_k max(λ_k, 0)` over them.
///
/// Colours `U` row-major `[o, rank]` with column `k` the eigenvector for the `(rank−k)`-th largest
/// eigenvalue (ascending order, so column `rank−1` is the largest). The eigendecomposition is the
/// shared [`eigh`]; the `M` Gram mirrors [`lowrank_corrector_exact`].
///
/// # Complexity
///
/// Time `O(o·n² + o²·n + o³·sweeps)`; space `O(o·n + o² + o·rank)`.
///
/// # Errors
///
/// The refusal of [`eigh`] (non-convergence).
fn eigen_parts(
    delta_w: &[f64],
    hl: &[f64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<(Vec<f64>, f64), CorrectorError> {
    // `w1 = ΔW · Hl` (o × n).
    let mut w1 = vec![0.0f64; o * n];
    for i in 0..o {
        for k in 0..n {
            let mut acc = 0.0f64;
            for j in 0..n {
                acc += delta_w[i * n + j] * hl[j * n + k];
            }
            w1[i * n + k] = acc;
        }
    }
    // `m = w1 · ΔWᵀ` (o × o); `eigh` symmetrises it as the Python does.
    let mut m = vec![0.0f64; o * o];
    for i in 0..o {
        for l in 0..o {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += w1[i * n + k] * delta_w[l * n + k];
            }
            m[i * o + l] = acc;
        }
    }
    let (evals, evecs) = eigh(&m, o)?;
    // `U` = the `rank` largest eigenvectors: columns `o − rank .. o` of the ascending-order basis.
    let mut u = vec![0.0f64; o * rank];
    for i in 0..o {
        for k in 0..rank {
            u[i * rank + k] = evecs[i * o + (o - rank + k)];
        }
    }
    let mut top_sum = 0.0f64;
    for k in 0..rank {
        let lam = evals[o - rank + k];
        if lam > 0.0 {
            top_sum += lam;
        }
    }
    Ok((u, top_sum))
}

/// Solve the rank×rank ridge system `(Uhᵀ Uh + 1e-6 I)·V = Uhᵀ ΔW` for `V`, row-major `[rank, n]`.
///
/// # Complexity
///
/// Time `O(rank²·o + rank·o·n + rank³ + rank²·n)` — the Gram `Uhᵀ Uh` `O(rank²·o)`, the right-hand
/// side `Uhᵀ ΔW` `O(rank·o·n)`, and the [`solve_spd`] `Cholesky` `O(rank³ + rank²·n)`. Space
/// `O(rank² + rank·n)`.
///
/// # Errors
///
/// [`CorrectorError::NotPositiveDefinite`] if the ridged Gram is singular at some pivot.
fn ridge_solve(
    u: &[f64],
    delta_w: &[f64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<Vec<f64>, CorrectorError> {
    // A = Uhᵀ Uh + 1e-6 I (rank × rank).
    let mut a = vec![0.0f64; rank * rank];
    for i in 0..rank {
        for j in 0..rank {
            let mut acc = 0.0f64;
            for p in 0..o {
                acc += u[p * rank + i] * u[p * rank + j];
            }
            a[i * rank + j] = acc;
        }
    }
    for i in 0..rank {
        a[i * rank + i] += RIDGE;
    }
    // B = Uhᵀ ΔW (rank × n).
    let mut b = vec![0.0f64; rank * n];
    for k in 0..rank {
        for j in 0..n {
            let mut acc = 0.0f64;
            for i in 0..o {
                acc += u[i * rank + k] * delta_w[i * n + j];
            }
            b[k * n + j] = acc;
        }
    }
    solve_spd(&a, &b, rank, n).map_err(|_| CorrectorError::NotPositiveDefinite)
}

/// `U·V` for row-major `[o, rank] · [rank, n]`, accumulating `Σ_k` per `(i, j)` in ascending `k`
/// order — the order the factor-product test re-computes, so `corr` and the test product agree
/// bit-for-bit. Time `O(o·n·rank)`, space `O(o·n)`.
fn product(u: &[f64], v: &[f64], o: usize, rank: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; o * n];
    for i in 0..o {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..rank {
                acc += u[i * rank + k] * v[k * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}
