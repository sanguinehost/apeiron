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
//! # Q32.32 carrier
//!
//! Every body works on the Q32.32 `i64` grid: [`lowrank_corrector`] takes `ΔW` and `Hl` as words and
//! returns `Corr`, `top_sum` and the stored factors as words; the only `f64` left is the
//! [`crate::learn::fold::fit::eigh`] Gram/eigenstage boundary those words are handed to (that module
//! still has an `f64` entry) and its read-back. The Gram `M` is the shared [`eigen_gram`] (two
//! pooled Q32.32 products under one exact power-of-two panel scale), the eigenstage is [`top_eigh`],
//! the ridge Gram is the pooled [`wsyrk`] and the right-hand side the pooled [`gemm_nn`] on the
//! transposed `Uh`, the ridge solve is [`chol_solve`], the factor product `Corr = Uh·Vh` is the
//! pooled [`gemm_nn`], the `f16` round-trip is [`crate::learn::solve::q32::round_f16`], and the u8
//! quantizer is [`crate::learn::fold::fit::quant::quant_c_u8`] — the same carriers the store uses.
//!
//! # Deviations from the Python
//!
//! - The Python's `rank == 0` (`evecs[:, -0:]` selects *all* columns) returns the empty corrector
//!   (`Corr = 0`, zero bits, empty factors), the reading
//!   [`crate::learn::fold::fit::eigh::lowrank_corrector_exact`] gives it; no caller passes `rank == 0`.
//! - `torch.linalg.solve` on the possibly-indefinite `Uhᵀ Uh + 1e-6 I` is replaced by the SPD
//!   [`chol_solve`]; a non-positive pivot is [`CorrectorError::NotPositiveDefinite`], not a result.
//! - The inputs are Q32.32 words, so a non-finite or off-grid `ΔW`/`Hl` entry cannot reach this
//!   module; [`CorrectorError::NonFinite`]/[`CorrectorError::NotRepresentable`] remain only for the
//!   `f64` read-back of [`crate::learn::fold::fit::eigh`].
//! - The ridge `1e-6` is the grid word `round(1e-6·2³²) = 4295`, i.e. `1.000013e-6`: the carrier
//!   cannot hold a finer value, and the relative change of the ridge is `1.3e-5`.
//! - `f16r(U)` rounds once from the exact integer significand ([`round_f16`]) instead of the
//!   Python's `f64 → f32 → f16`; a factor at binary16's range is [`CorrectorError::Overflow`].
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_corrector.rs`). No extraction or refinement links these bodies to a
//! Lean theorem, so nothing here is `proved`.

#![allow(
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the row-major `[o, n]`/`[rank, n]` kernels index by `i/j/k/p` against the spec's \
              `DeltaW[i, j]`/`Uh`/`Vh` arithmetic, and `rank as u64` is a deliberate lossy \
              dimension cast in the bit-count formula; the pedantic lints stay on for the rest of \
              the crate."
)]

use crate::learn::fold::fit::eigh::{eigen_gram, gram_scale, lowrank_corrector_exact, EighError};
use crate::learn::fold::fit::quant::quant_c_u8;
use crate::learn::fold::stats::parts::{panel, shift};
use crate::learn::solve::chol::chol_solve;
use crate::learn::solve::eig::top_eigh;
use crate::learn::solve::gemm::gemm_nn;
use crate::learn::solve::q32::{from_grid, round_f16, to_grid, Fx, GRID_BOUND, INPUT_BOUND};
use crate::learn::solve::syrk::wsyrk;

/// The `1e-6` ridge added to the rank×rank Gram `Uhᵀ Uh` before the `V` solve, the Python's
/// `torch.eye(rank) * 1e-6`, as the grid word `round(1e-6·2³²) = 4295` (`1.000013e-6`).
const RIDGE_Q: i64 = 4295;

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
    /// A finite entry of `ΔW` or `Hl` lies at or beyond the Q32.32 grid's real range `2³¹`
    /// ([`GRID_BOUND`]), so the carrier cannot represent it.
    NotRepresentable,
    /// The underlying eigendecomposition did not converge within its sweep bound.
    NotConverged,
    /// The rank×rank ridge system `Uhᵀ Uh + 1e-6 I` was not positive-definite at some pivot (only
    /// reachable from a degenerate rounded `Uh`).
    NotPositiveDefinite,
    /// A checked `i128` accumulation, a rounded entry or a power-of-two undo left its integer range.
    Overflow,
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
    /// `Uh`, row-major `[o, rank]` (column `k` is the `k`-th stored basis vector), as Q32.32 words.
    pub u: Vec<i64>,
    /// `Vh`, row-major `[rank, n]`, as Q32.32 words.
    pub v: Vec<i64>,
}

/// A rank-`rank` corrector in the metric `Hl`: the dense correction, its bit count, the top-`rank`
/// eigenvalue sum, and (optionally) the stored factors.
#[derive(Clone, Debug)]
pub struct LowrankCorrector {
    /// The dense correction `Corr = Uh·Vh`, row-major `[o, n]`, as Q32.32 words.
    pub corr: Vec<i64>,
    /// The stored-factor bit count: `16·rank·(o + n)` for [`Mode::Exact`]/[`Mode::F16`],
    /// `rank·(8·(o + n) + 64)` for [`Mode::U8`].
    pub bits: u64,
    /// `Σ_k max(λ_k, 0)` over the top `rank` eigenvalues of `M = (ΔW·Hl)·ΔWᵀ`, as a Q32.32 word.
    pub top_sum: i64,
    /// The stored factors, present iff the caller requested them.
    pub factors: Option<Factors>,
}

/// `x` as Q32.32 words `round(x·2³²)`; `NonFinite` / `NotRepresentable` name the refusal. Time and
/// space `O(len)`, one grid rounding per entry.
fn to_words(x: &[f64]) -> Result<Vec<i64>, CorrectorError> {
    x.iter()
        .map(|&v| {
            to_grid(v, GRID_BOUND).ok_or(if v.is_finite() {
                CorrectorError::NotRepresentable
            } else {
                CorrectorError::NonFinite
            })
        })
        .collect()
}

/// The Q32.32 words `x` as reals (exact). Time and space `O(len)`.
fn to_real(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| from_grid(v)).collect()
}

/// `x` as the Q32.32 word `round(x·2³²)`; `NotRepresentable` off the grid. `O(1)`.
fn to_word(x: f64) -> Result<i64, CorrectorError> {
    to_grid(x, GRID_BOUND).ok_or(CorrectorError::NotRepresentable)
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

/// The top-`rank` eigenvectors of `M = (ΔW·Hl)·ΔWᵀ` and `Σ_k max(λ_k, 0)` over them, as reals.
///
/// The Gram is [`eigen_gram`], scaled by the exact power of two [`gram_scale`] into [`top_eigh`]'s
/// bound; the scale is undone exactly on the summed eigenvalue words. `U` is row-major `[o, rank]`,
/// column `k` the eigenvector for the `(rank−k)`-th largest eigenvalue; `M = 0` returns the identity
/// columns, the basis the full `eigh` gives the zero matrix.
///
/// # Errors
/// [`CorrectorError::ShapeMismatch`] for a wrong `ΔW`/`Hl` length,
/// [`CorrectorError::RankOutOfRange`] for `rank > o`, [`CorrectorError::NotConverged`] for a
/// [`top_eigh`] refusal, [`CorrectorError::Overflow`] on the eigenvalue sum.
///
/// # Complexity
/// Time `O(o·n² + o²·n + o²·rank·iters)`; space `O(o·n + o² + o·rank)`.
pub fn eigen_parts(
    delta_w: &[f64],
    hl: &[f64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<(Vec<f64>, f64), CorrectorError> {
    if rank > o {
        return Err(CorrectorError::RankOutOfRange);
    }
    let m = eigen_gram(delta_w, hl, o, n)?;
    let mmax = m.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
    if mmax <= 0.0 {
        // `M = 0`: the full `eigh` of the zero matrix returns the identity basis; reproduce it.
        let mut u = vec![0.0_f64; o * rank];
        for i in 0..o {
            for k in 0..rank {
                if i == o - rank + k {
                    u[i * rank + k] = 1.0;
                }
            }
        }
        return Ok((u, 0.0));
    }
    let s_m = gram_scale(mmax);
    let mut m_q = Vec::with_capacity(m.len());
    for &v in &m {
        m_q.push(to_grid(v * s_m, INPUT_BOUND).ok_or(CorrectorError::NotConverged)?);
    }
    let top = top_eigh(&m_q, o, rank).map_err(|_| CorrectorError::NotConverged)?;
    let mut sum: i128 = 0;
    for k in 0..rank {
        if top.evals[k] > 0 {
            sum = sum
                .checked_add(i128::from(top.evals[k]))
                .ok_or(CorrectorError::Overflow)?;
        }
    }
    let ws = i64::try_from(sum).map_err(|_| CorrectorError::Overflow)?;
    Ok((to_real(&top.evecs), from_grid(ws) / s_m))
}

/// `x` rounded to binary16 and back to Q32.32 ([`round_f16`] then one grid shift); `Overflow` at
/// binary16's range (`2¹⁶`) or on a shift that leaves `i64`. `O(1)`.
fn f16_word(v: i64) -> Result<i64, CorrectorError> {
    let f = round_f16(Fx { m: v, e: -32 }).map_err(|_| CorrectorError::Overflow)?;
    let s = u32::try_from(f.e + 32).map_err(|_| CorrectorError::Overflow)?;
    let w = i128::from(f.m)
        .checked_shl(s)
        .ok_or(CorrectorError::Overflow)?;
    i64::try_from(w).map_err(|_| CorrectorError::Overflow)
}

/// Solve the rank×rank ridge system `(Uhᵀ Uh + RIDGE_Q·I)·V = Uhᵀ ΔW` for `V`, row-major
/// `[rank, n]` Q32.32: the Gram is the pooled [`wsyrk`], the right-hand side the pooled
/// [`gemm_nn`] on the transposed `Uh` under one exact `2⁻ᵏ` panel (§ [`panel`]), and the solve is
/// [`chol_solve`].
///
/// # Errors
/// [`CorrectorError::NotPositiveDefinite`] if the ridged Gram is singular at some pivot;
/// [`CorrectorError::Overflow`] on a checked `i128` operation or a panel that leaves range.
///
/// # Complexity
/// Time `O(rank²·o + rank·o·n + rank³ + rank²·n)`: the `rank·o·n` right-hand side (≈10⁶
/// multiply-accumulates at `o = n = 896`, one call per corrector) is one pooled row-split, the
/// rank×rank factor and the `rank²·n` substitution serial; space `O(o·rank + rank² + rank·n)`.
fn ridge_solve(
    u: &[i64],
    delta_w: &[i64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<Vec<i64>, CorrectorError> {
    // A = Uhᵀ Uh + RIDGE_Q·I (rank × rank), one rounding per entry on the upper triangle, mirrored.
    let mut a = wsyrk(u, None, o, rank).map_err(|_| CorrectorError::Overflow)?;
    for i in 0..rank {
        a[i * rank + i] = a[i * rank + i]
            .checked_add(RIDGE_Q)
            .ok_or(CorrectorError::Overflow)?;
    }
    // B = Uhᵀ ΔW (rank × n) through the pooled kernel on the transposed `Uh`, the exact panel
    // shift undone once so `B` is the grid word.
    let ut: Vec<i64> = (0..rank * o).map(|t| u[(t % o) * rank + t / o]).collect();
    let (dw, kd) = panel(delta_w).ok_or(CorrectorError::Overflow)?;
    let mut b = gemm_nn(&ut, &dw, rank, o, n).map_err(|_| CorrectorError::Overflow)?;
    shift(&mut b, -kd).ok_or(CorrectorError::Overflow)?;
    Ok(chol_solve(&a, &b, rank, n)
        .map_err(|_| CorrectorError::NotPositiveDefinite)?
        .x)
}

/// The rank-`rank` corrector of `ΔW` in the metric `Hl`, rounded into the carrier `mode`.
///
/// `ΔW` is row-major `[o, n]` and `Hl` row-major `[n, n]`, both Q32.32 words. Builds
/// `M = (ΔW·Hl)·ΔWᵀ` on the Q32.32 grid, takes its top `rank` eigenvectors as `U` —
/// [`eigen_parts`] for the rounded carriers, [`crate::learn::fold::fit::eigh::eigh`] for
/// [`Mode::Exact`] — and returns `Corr = Uh·Vh`, the carrier's stored `(Uh, Vh)` when requested, the
/// bit count and `top_sum`: [`Mode::Exact`] `(U, Uᵀ ΔW)`, [`Mode::F16`] `Uh = f16r(U)`,
/// `Vh = f16r(chol_solve(Uhᵀ Uh + 1e-6 I, Uhᵀ ΔW))`, [`Mode::U8`] the same ridge solve on the
/// column-quantized `Uh` with each row of `V` quantized. Every returned buffer is a Q32.32 word.
///
/// # Complexity
/// Time `O(o·n² + o²·n + o²·rank·iters + rank·o·n + rank²·(o + n) + rank³)`: the pooled `ΔW·Hl`,
/// `M` and `Corr` products, the top-`rank` extraction, and the rank×rank ridge solve. Space
/// `O(o·n + o² + o·rank + rank·n)`.
///
/// # Errors
/// [`CorrectorError::ShapeMismatch`] for a wrong `ΔW`/`Hl` length,
/// [`CorrectorError::RankOutOfRange`] for `rank > o`, and the refusals of the eigenstage and ridge
/// solve.
pub fn lowrank_corrector(
    delta_w: &[i64],
    hl: &[i64],
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

    if rank == 0 {
        return Ok(LowrankCorrector {
            corr: vec![0; o * n],
            bits: 0,
            top_sum: 0,
            factors: factors.then_some(Factors {
                u: Vec::new(),
                v: Vec::new(),
            }),
        });
    }
    // The `eigh` entry is still `f64`; the words round-trip through `from_grid` exactly and the
    // results are read back onto the same grid.
    if mode == Mode::Exact {
        let c = lowrank_corrector_exact(&to_real(delta_w), &to_real(hl), o, n, rank)?;
        let f = if factors {
            Some(Factors {
                u: to_words(&c.u)?,
                v: to_words(&c.v)?,
            })
        } else {
            None
        };
        return Ok(LowrankCorrector {
            corr: to_words(&c.corr)?,
            bits: f16_bits(o, n, rank),
            top_sum: to_word(c.top_sum)?,
            factors: f,
        });
    }

    let (u, top_sum) = eigen_parts(&to_real(delta_w), &to_real(hl), o, n, rank)?;
    let uq = to_words(&u)?;
    let top_sum = to_word(top_sum)?;
    let (uh, vh, bits) = match mode {
        Mode::F16 => {
            let uh: Vec<i64> = uq.iter().map(|&x| f16_word(x)).collect::<Result<_, _>>()?;
            let v = ridge_solve(&uh, delta_w, o, n, rank)?;
            let vh: Vec<i64> = v.iter().map(|&x| f16_word(x)).collect::<Result<_, _>>()?;
            (uh, vh, f16_bits(o, n, rank))
        }
        Mode::U8 => {
            // Column `k` of `U` is quantized independently on the word carrier.
            let mut uh = vec![0_i64; o * rank];
            let mut col = vec![0_i64; o];
            for k in 0..rank {
                for (i, c) in col.iter_mut().enumerate() {
                    *c = uq[i * rank + k];
                }
                let (q, _) =
                    quant_c_u8(&col, U8_BITS).map_err(|_| CorrectorError::ShapeMismatch)?;
                for i in 0..o {
                    uh[i * rank + k] = q[i];
                }
            }
            let v = ridge_solve(&uh, delta_w, o, n, rank)?;
            // Row `k` of `V` is quantized independently.
            let mut vh = vec![0_i64; rank * n];
            for (k, slot) in vh.chunks_mut(n).enumerate() {
                let (q, _) = quant_c_u8(&v[k * n..k * n + n], U8_BITS)
                    .map_err(|_| CorrectorError::ShapeMismatch)?;
                slot.copy_from_slice(&q);
            }
            (uh, vh, u8_bits(o, n, rank))
        }
        Mode::Exact => unreachable!("handled above"),
    };
    let corr = if n == 0 {
        Vec::new()
    } else {
        // `Corr = Uh·Vh` on the pooled `gemm_nn` under one exact `2⁻ᵏ` panel, undone on the words.
        let (uh_p, ku) = panel(&uh).ok_or(CorrectorError::Overflow)?;
        let (vh_p, kv) = panel(&vh).ok_or(CorrectorError::Overflow)?;
        let mut c = gemm_nn(&uh_p, &vh_p, o, rank, n).map_err(|_| CorrectorError::Overflow)?;
        shift(&mut c, ku + kv).ok_or(CorrectorError::Overflow)?;
        c
    };
    Ok(LowrankCorrector {
        corr,
        bits,
        top_sum,
        factors: factors.then_some(Factors { u: uh, v: vh }),
    })
}
