//! The `-resq` production row-scale step for the MLP `up` plane.
//!
//! Ports the `use_res and nm == "up"` block of `fold_transfer.py`'s `run_variant`
//! (`spec/fold_transfer.py:1911-1945`). On the `up` plane the production variant `-resq`, instead of
//! rescaling the reconstruction `Wp` directly (the `Wp *= s[:, None]` of the `-res` path), folds the
//! per-row multiplier into each stored plane's *hierarchical row scale*: `rh = quant_r_hier(rh·s)`,
//! so the multiplier costs no bits beyond `quant_r_hier`'s accounted r-bits.
//!
//! # The solve
//!
//! Given the ffn-stage input slabs `X` (`rows × n`), the teacher gate/up weights `W_g0`, `W_u0`
//! (`inter × n`), the student quantised gate weight `Wq_g` and the current up reconstruction `Wq_u`
//! (`inter × n`), and the teacher down weight `W_down` (`hidden × inter`):
//!
//! - `G = W_downᵀ·W_down` (`inter × inter`).
//! - Per slab: `P_0 = silu(X·W_g0ᵀ) ⊙ (X·W_u0ᵀ)`, `P_q = silu(X·Wq_gᵀ) ⊙ (X·Wq_uᵀ)`;
//!   `P_mat += P_qᵀ·P_q`, `C_mat += P_qᵀ·P_0`.
//! - `GoP = G ⊙ P_mat`, `GoC = G ⊙ C_mat` (Hadamard); `rhs = GoC·1`;
//!   `lam = 0.05·max(mean(diag(GoP)), 1e-12)`; `GoP.diag += lam`, `rhs += lam`.
//! - `s = solve(GoP, rhs)` (Cholesky), clamped to `[0.5, 2.0]`.
//!
//! # Q32.32 carrier
//!
//! Every value this step computes is a Q32.32 `i64` word. The weights cross the grid once at
//! [`resq_solve`]'s entry; the four projections and the two Grams are the shared [`gemm_nt_f64`]
//! Q32.32 products (grid in, exact power-of-two panel scale, grid out); each gate and up dot is
//! re-gridded once; the gate is [`silu_fixed`], an elementwise product is [`round_q32_even`] of the
//! exact `i128` product, the Hadamard is [`hadamard_into`], `rhs = GoC·1` is [`row_dot_fixed`], and
//! the solve is [`chol_solve`]. The per-row multiplier fold is one integer product per stored entry;
//! the only `f64` is the products' documented power-of-two scaling and grid crossing.
//!
//! The Hadamard product of two symmetric positive-semidefinite matrices (`G` is a Gram, `P_mat` is a
//! Gram of the slab stack) is positive-semidefinite (Schur product theorem), so `GoP + lam·I` is
//! positive-definite for `lam > 0`; [`resq_solve`] solves it with [`chol_solve`] and returns
//! [`ResqError::NotPositiveDefinite`] if a `Cholesky` pivot is non-positive, with no fallback.
//!
//! # Partiality
//!
//! Every entry point refuses a shape that disagrees with the declared dims ([`ResqError::ShapeMismatch`]),
//! a zero `inter`/`n`/`hidden` ([`ResqError::Empty`]), a non-finite input ([`ResqError::NonFinite`]),
//! a finite entry at or beyond the grid's range `2³¹` ([`ResqError::NotRepresentable`]), and an
//! integer operation that leaves its range ([`ResqError::Overflow`]). The `1e-12` floor of `lam` is
//! below the carrier's quantum `2⁻³² ≈ 2.3e-10`, so it is the grid's zero: a zero diagonal means
//! `lam == 0` and the solve's own positivity refusal. No function here panics on caller input.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_resq.rs`). No extraction or refinement links these bodies to a Lean
//! theorem, so nothing here is `proved`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the kernels index the row-major `[rows, n]`/`[inter, inter]` buffers by `i/j/r` and \
              name the carriers `W_g0`/`W_u0`/`Wq_g`/`Wq_u`/`P_mat`/`C_mat`/`GoP`/`GoC` as the \
              reference does"
)]

use crate::engine::row_dot_fixed;
use crate::engine::silu_fixed;
use crate::learn::fold::fit::quant::{quant_r_hier, QuantError, RHier};
use crate::learn::solve::chol::{chol_solve, SolveError};
use crate::learn::solve::gemm::gemm_nt_f64;
use crate::learn::solve::q32::{from_grid, round_q32_even, to_grid, GRID_BOUND};
use crate::learn::solve::syrk::hadamard_into;

/// `1.0` in raw Q32.32.
const ONE: i64 = 1_i64 << 32;

/// A refusal from the `-resq` row-scale step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResqError {
    /// A zero `inter`, `n` or `hidden`, or an empty rescale row set.
    Empty,
    /// A slice length that disagrees with the declared shape.
    ShapeMismatch,
    /// A non-finite entry in an input.
    NonFinite,
    /// A finite entry at or beyond the Q32.32 grid's real range `2³¹` ([`GRID_BOUND`]).
    NotRepresentable,
    /// The elementwise-Gram system had a non-positive `Cholesky` pivot.
    NotPositiveDefinite,
    /// A checked integer accumulation, product or narrowing left its range.
    Overflow,
    /// The hierarchical row-scale quantiser refused its input.
    Quant(QuantError),
}

impl From<QuantError> for ResqError {
    fn from(e: QuantError) -> Self {
        ResqError::Quant(e)
    }
}

/// The `-resq` line-search inputs for one `up` plane, row-major.
///
/// `x_slabs` are the ffn-stage slabs (`rows_s × n` each, flattened row-major); `w_g0`/`w_u0` the
/// teacher gate/up (`inter × n`); `wq_g`/`wq_u` the student quantised gate and current up
/// reconstruction (`inter × n`); `w_down` the teacher down (`hidden × inter`).
#[derive(Clone, Copy, Debug)]
pub struct ResqInputs<'a> {
    /// The ffn-stage input slabs, each flattened row-major `[rows, n]`.
    pub x_slabs: &'a [&'a [f64]],
    /// The input width `n`.
    pub n: usize,
    /// The teacher gate weight, row-major `[inter, n]`.
    pub w_g0: &'a [f64],
    /// The teacher up weight, row-major `[inter, n]`.
    pub w_u0: &'a [f64],
    /// The student quantised gate weight, row-major `[inter, n]`.
    pub wq_g: &'a [f64],
    /// The current up reconstruction, row-major `[inter, n]`.
    pub wq_u: &'a [f64],
    /// The teacher down weight, row-major `[hidden, inter]`.
    pub w_down: &'a [f64],
    /// The intermediate width `inter`.
    pub inter: usize,
    /// The hidden width `hidden`.
    pub hidden: usize,
}

/// The `-resq` system and its solution.
///
/// `gop` is `G ⊙ P_mat` with `lam` added to its diagonal (row-major `[inter, inter]`); `rhs` is
/// `GoC·1` with `lam` added; `s_raw` solves `gop·s = rhs`; `s_clamped` is `s_raw` clamped to
/// `[0.5, 2.0]` (the value the production step applies). Every field is a Q32.32 real.
#[derive(Clone, Debug)]
pub struct ResqSolution {
    /// `G ⊙ P_mat + lam·I`, row-major `[inter, inter]`.
    pub gop: Vec<f64>,
    /// `GoC·1 + lam`, length `inter`.
    pub rhs: Vec<f64>,
    /// The ridge `0.05·max(mean(diag(G ⊙ P_mat)), 1e-12)`.
    pub lam: f64,
    /// The unclamped solution `gop⁻¹·rhs`, length `inter`.
    pub s_raw: Vec<f64>,
    /// The solution clamped to `[0.5, 2.0]`, length `inter`.
    pub s_clamped: Vec<f64>,
}

/// `x` as Q32.32 words `round(x·2³²)`; `NonFinite` / `NotRepresentable` name the refusal.
/// Time and space `O(len)`, one grid rounding per entry.
fn to_words(x: &[f64]) -> Result<Vec<i64>, ResqError> {
    x.iter()
        .map(|&v| {
            to_grid(v, GRID_BOUND).ok_or(if v.is_finite() {
                ResqError::NotRepresentable
            } else {
                ResqError::NonFinite
            })
        })
        .collect()
}

/// The Q32.32 words `x` as reals (exact). Time and space `O(len)`.
fn to_real(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| from_grid(v)).collect()
}

/// `a` (row-major `[r, c]`) transposed to `[c, r]`. Time `O(r·c)` moves, space `O(r·c)`.
fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// The Q32.32 product `round(a·b)`: the exact `i128` product, one round-to-nearest shift. `O(1)`;
/// `Overflow` when the rounded product leaves `i64`.
fn mul_q32(a: i64, b: i64) -> Result<i64, ResqError> {
    round_q32_even(i128::from(a) * i128::from(b)).map_err(|_| ResqError::Overflow)
}

/// `round(num / den)` (nearest, ties up) for a positive `den`, or `None` when the result leaves `i64`.
/// `O(1)`.
fn div_q32(num: i128, den: i64) -> Option<i64> {
    let d = i128::from(den);
    let n = if num >= 0 {
        num.checked_add(d / 2)?
    } else {
        num.checked_sub(d / 2)?
    };
    i64::try_from(n / d).ok()
}

/// One slab's `P` rows: `silu(X·W_gᵀ) ⊙ (X·W_uᵀ)`, row-major `[rows, inter]` Q32.32.
///
/// # Errors
/// `NotRepresentable` past `2³¹`; `Overflow` from a `SiLU` or product.
///
/// # Complexity
/// Time `O(rows·n·inter)` word ops in the two pooled `gemm` products plus `O(rows·inter)`
/// elementwise, above the `10⁶` pool threshold at any production slab. Space `O(rows·inter)`.
fn project(
    slab: &[f64],
    rows: usize,
    n: usize,
    inter: usize,
    wg: &[i64],
    wu: &[i64],
) -> Result<Vec<i64>, ResqError> {
    let (wg, wu) = (to_real(wg), to_real(wu));
    let gate = gemm_nt_f64(slab, &wg, rows, n, inter).map_err(|_| ResqError::Overflow)?;
    let up = gemm_nt_f64(slab, &wu, rows, n, inter).map_err(|_| ResqError::Overflow)?;
    let mut out = Vec::with_capacity(rows * inter);
    for k in 0..rows * inter {
        let g = to_grid(gate[k], GRID_BOUND).ok_or(ResqError::NotRepresentable)?;
        let u = to_grid(up[k], GRID_BOUND).ok_or(ResqError::NotRepresentable)?;
        let s = silu_fixed(g).map_err(|_| ResqError::Overflow)?;
        out.push(mul_q32(s, u)?);
    }
    Ok(out)
}

/// The clamped per-row multiplier `s_sol`, length `inter`.
///
/// # Errors
/// As [`resq_solve`].
pub fn resq_scale(inputs: &ResqInputs<'_>) -> Result<Vec<f64>, ResqError> {
    Ok(resq_solve(inputs)?.s_clamped)
}

/// Build and solve the `-resq` elementwise-Gram ridge system on the Q32.32 grid.
///
/// # Errors
/// [`ResqError::Empty`] for a zero `inter`/`n`/`hidden`; [`ResqError::ShapeMismatch`] for a weight
/// or slab length against the declared dims; [`ResqError::NonFinite`] / [`ResqError::NotRepresentable`]
/// for an input entry outside the carrier; [`ResqError::NotPositiveDefinite`] if the `Cholesky`
/// factor of `GoP + lam·I` hits a non-positive pivot; [`ResqError::Overflow`] on checked arithmetic.
///
/// # Complexity
/// Temporal: `O(hidden·inter² + Σ_s rows_s·(n·inter + inter²) + inter³)`, dominant term
/// `2·rows·inter²` in the pooled `P_qᵀ·P_q`/`P_qᵀ·P_0` pair. The per-row loops (`rhs = GoC·1` at
/// `inter²`, the row-scale fold at `o·g`) are serial: at or near `10⁶` word ops, cheaper than a
/// dispatch. Spatial: `O(inter² + rows_max·inter)` Q32.32 words.
pub fn resq_solve(inputs: &ResqInputs<'_>) -> Result<ResqSolution, ResqError> {
    let ResqInputs {
        x_slabs,
        n,
        w_g0,
        w_u0,
        wq_g,
        wq_u,
        w_down,
        inter,
        hidden,
    } = *inputs;

    if n == 0 || inter == 0 || hidden == 0 {
        return Err(ResqError::Empty);
    }
    let inter_n = inter.checked_mul(n).ok_or(ResqError::ShapeMismatch)?;
    let hidden_inter = hidden.checked_mul(inter).ok_or(ResqError::ShapeMismatch)?;
    let inter2 = inter.checked_mul(inter).ok_or(ResqError::ShapeMismatch)?;
    if w_g0.len() != inter_n
        || w_u0.len() != inter_n
        || wq_g.len() != inter_n
        || wq_u.len() != inter_n
    {
        return Err(ResqError::ShapeMismatch);
    }
    if w_down.len() != hidden_inter {
        return Err(ResqError::ShapeMismatch);
    }
    for slab in x_slabs {
        if slab.len() % n != 0 {
            return Err(ResqError::ShapeMismatch);
        }
        if slab.iter().any(|v| !v.is_finite()) {
            return Err(ResqError::NonFinite);
        }
    }

    // The one `f64 → Q32.32` crossing of the weights.
    let down = to_words(w_down)?;
    let g0 = to_words(w_g0)?;
    let u0 = to_words(w_u0)?;
    let qg = to_words(wq_g)?;
    let qu = to_words(wq_u)?;

    // G = W_downᵀ·W_down: the shared pooled product on the transposed down panel.
    let dt = transpose(&to_real(&down), hidden, inter);
    let g =
        to_words(&gemm_nt_f64(&dt, &dt, inter, hidden, inter).map_err(|_| ResqError::Overflow)?)?;

    let mut p_mat = vec![0_i64; inter2];
    let mut c_mat = vec![0_i64; inter2];
    for slab in x_slabs {
        let rows = slab.len() / n;
        if rows == 0 {
            continue;
        }
        let p0 = project(slab, rows, n, inter, &g0, &u0)?;
        let pq = project(slab, rows, n, inter, &qg, &qu)?;
        let pqt = transpose(&to_real(&pq), rows, inter);
        let p0t = transpose(&to_real(&p0), rows, inter);
        // `P_qᵀ·P_q` and `P_qᵀ·P_0`: `inter × inter` over the slab's `rows`.
        let pp = gemm_nt_f64(&pqt, &pqt, inter, rows, inter).map_err(|_| ResqError::Overflow)?;
        let pc = gemm_nt_f64(&pqt, &p0t, inter, rows, inter).map_err(|_| ResqError::Overflow)?;
        for idx in 0..inter2 {
            let a = to_grid(pp[idx], GRID_BOUND).ok_or(ResqError::NotRepresentable)?;
            let b = to_grid(pc[idx], GRID_BOUND).ok_or(ResqError::NotRepresentable)?;
            p_mat[idx] = p_mat[idx].checked_add(a).ok_or(ResqError::Overflow)?;
            c_mat[idx] = c_mat[idx].checked_add(b).ok_or(ResqError::Overflow)?;
        }
    }

    // GoP = G ⊙ P_mat; GoC = G ⊙ C_mat; rhs = GoC·1; the ridge from the GoP diagonal.
    let mut gop = g.clone();
    hadamard_into(&mut gop, &p_mat).map_err(|_| ResqError::Overflow)?;
    let mut goc = g;
    hadamard_into(&mut goc, &c_mat).map_err(|_| ResqError::Overflow)?;
    let ones = vec![ONE; inter];
    let mut rhs = vec![0_i64; inter];
    let mut diag_sum: i128 = 0;
    for i in 0..inter {
        rhs[i] = row_dot_fixed(&goc[i * inter..(i + 1) * inter], &ones)
            .map_err(|_| ResqError::Overflow)?;
        diag_sum = diag_sum
            .checked_add(i128::from(gop[i * inter + i]))
            .ok_or(ResqError::Overflow)?;
    }
    let mean = div_q32(
        diag_sum,
        i64::try_from(inter).map_err(|_| ResqError::Overflow)?,
    )
    .ok_or(ResqError::Overflow)?;
    let lam = div_q32(i128::from(mean.max(0)), 20).ok_or(ResqError::Overflow)?;
    for i in 0..inter {
        gop[i * inter + i] = gop[i * inter + i]
            .checked_add(lam)
            .ok_or(ResqError::Overflow)?;
        rhs[i] = rhs[i].checked_add(lam).ok_or(ResqError::Overflow)?;
    }

    let solved = chol_solve(&gop, &rhs, inter, 1).map_err(|e| match e {
        SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
            ResqError::NotPositiveDefinite
        }
        SolveError::Overflow => ResqError::Overflow,
        SolveError::ShapeMismatch | SolveError::Empty => ResqError::ShapeMismatch,
    })?;
    let half = ONE / 2;
    let twice = 2 * ONE;
    Ok(ResqSolution {
        gop: to_real(&gop),
        rhs: to_real(&rhs),
        lam: from_grid(lam),
        s_raw: to_real(&solved.x),
        s_clamped: solved
            .x
            .iter()
            .map(|&v| from_grid(v.clamp(half, twice)))
            .collect(),
    })
}

/// Fold a per-row multiplier into a plane's stored hierarchical row scale: `rh = quant_r_hier(rh·s)`
/// with `rh` row-major `[o, g]` and `s` length `o`, the product on the Q32.32 grid.
///
/// This is the `-resq` replacement for `Wp *= s[:, None]`: the multiplier enters the row scale before
/// the grid quantization, so it is absorbed into `quant_r_hier`'s bits.
///
/// # Errors
/// [`ResqError::Empty`] for a zero `o`/`g`; [`ResqError::ShapeMismatch`] for a length against the
/// declared shape; [`ResqError::NonFinite`] / [`ResqError::NotRepresentable`] for an input entry
/// outside the carrier; [`ResqError::Quant`] if [`quant_r_hier`] refuses.
///
/// # Complexity
/// Temporal: `O(o·g)` serial products plus `quant_r_hier`'s `O(o·g)` — one product per stored entry,
/// which a pool dispatch does not repay; spatial `O(o·g)`.
pub fn rescale_hier(rh: &[f64], s: &[f64], o: usize, g: usize) -> Result<RHier, ResqError> {
    if o == 0 || g == 0 {
        return Err(ResqError::Empty);
    }
    let n = o.checked_mul(g).ok_or(ResqError::ShapeMismatch)?;
    if rh.len() != n || s.len() != o {
        return Err(ResqError::ShapeMismatch);
    }
    let rhw = to_words(rh)?;
    let sw = to_words(s)?;
    let mut scaled = vec![0.0_f64; n];
    for i in 0..o {
        for j in 0..g {
            scaled[i * g + j] = from_grid(mul_q32(rhw[i * g + j], sw[i])?);
        }
    }
    Ok(quant_r_hier(&scaled, o, g)?)
}

/// The partial-plane (`pk2_top`) variant of [`rescale_hier`]: the same `s` is gathered at `rows`
/// (the plane's global row indices) before the fold, `rh = quant_r_hier(rh·s[rows])` with `rh` the
/// partial plane's `[rows.len(), g]` scale.
///
/// # Errors
/// As [`rescale_hier`], with a gathered index out of range for `s` as [`ResqError::ShapeMismatch`].
///
/// # Complexity
/// Temporal: `O(rows.len()·g)` serial products plus `quant_r_hier`'s `O(rows.len()·g)`; spatial
/// `O(rows.len()·g)`.
pub fn rescale_hier_top(
    rh: &[f64],
    s: &[f64],
    rows: &[usize],
    g: usize,
) -> Result<RHier, ResqError> {
    let o = rows.len();
    if o == 0 || g == 0 {
        return Err(ResqError::Empty);
    }
    let n = o.checked_mul(g).ok_or(ResqError::ShapeMismatch)?;
    if rh.len() != n {
        return Err(ResqError::ShapeMismatch);
    }
    if rows.iter().any(|&r| r >= s.len()) {
        return Err(ResqError::ShapeMismatch);
    }
    let rhw = to_words(rh)?;
    let sw = to_words(s)?;
    let mut scaled = vec![0.0_f64; n];
    for (i, &r) in rows.iter().enumerate() {
        for j in 0..g {
            scaled[i * g + j] = from_grid(mul_q32(rhw[i * g + j], sw[r])?);
        }
    }
    Ok(quant_r_hier(&scaled, o, g)?)
}
