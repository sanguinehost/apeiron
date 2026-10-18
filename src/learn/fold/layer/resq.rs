//! The `-resq` production row-scale step for the MLP `up` plane: instead of rescaling the
//! reconstruction `Wp` (the `Wp *= s[:, None]` of `-res`), the per-row multiplier enters each
//! stored plane's hierarchical row scale, `rh = quant_r_hier(rh·s)`, costing no bits beyond
//! `quant_r_hier`'s r-bits. Ports `fold_transfer.py`'s `use_res and nm == "up"` block
//! (`spec/fold_transfer.py:1911-1945`).
//!
//! # Q32.32 carrier
//!
//! Every value is a Q32.32 `i64` word: [`ResqInputs`], [`ResqSolution`] and [`rescale_hier`]'s
//! `rh`/`s` arguments included, so nothing here reads or writes an `f64`. Products are the pooled
//! [`gemm_nt`]/[`gemm_nn`], Grams [`wsyrk`], the Hadamard [`hadamard_into`], `rhs = GoC·1`
//! [`row_dot_fixed`], and the solve [`chol_solve`] with its exact `sqrt(A_ii)` Jacobi column
//! scaling. An operand above `2³⁶` raw (`16` real) is scaled by one exact power of two and shifted
//! back; no `f64` division, no `crate::learn::ridge`.
//!
//! `G` and `P_mat` are Grams, so `G ⊙ P_mat` is PSD (Schur product theorem) and `GoP + lam·I` is
//! positive-definite for `lam > 0`; a non-positive pivot is [`ResqError::NotPositiveDefinite`].
//!
//! A shape against the declared dims, a zero `inter`/`n`/`hidden`, and any checked integer leaving
//! range are [`ResqError`] refusals; the word domain admits no non-finite value. The `1e-12` `lam`
//! floor is below the quantum `2⁻³² ≈ 2.3e-10`, the grid's zero: a zero diagonal is `lam == 0` and
//! the solve's own positivity refusal. No function here panics on caller input. Grade: `tested`
//! (`tests/learn_fold_resq.rs`); nothing here is `proved`.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the kernels index the row-major `[rows, n]`/`[inter, inter]` buffers by `i/j/r` and \
              name the carriers `W_g0`/`W_u0`/`Wq_g`/`Wq_u`/`P_mat`/`C_mat`/`GoP`/`GoC` as the \
              reference does"
)]

use crate::engine::row_dot_fixed;
use crate::engine::silu_fixed;
use crate::learn::fold::fit::quant::{quant_r_hier, QuantError, RHier};
use crate::learn::solve::chol::{chol_solve, SolveError};
use crate::learn::solve::gemm::{gemm_nn, gemm_nt};
use crate::learn::solve::q32::{ilog2_pos, round_q32_even, rshift_rne};
use crate::learn::solve::syrk::{hadamard_into, wsyrk};

/// `1.0` in raw Q32.32.
const ONE: i64 = 1_i64 << 32;

/// A refusal from the `-resq` row-scale step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResqError {
    /// A zero `inter`, `n` or `hidden`, or an empty rescale row set.
    Empty,
    /// A slice length that disagrees with the declared shape.
    ShapeMismatch,
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

/// The `-resq` line-search inputs for one `up` plane: `x_slabs` (`rows_s × n` each), the teacher
/// gate/up `w_g0`/`w_u0` and student `wq_g`/`wq_u` (`inter × n`), and the teacher down `w_down`
/// (`hidden × inter`), all row-major.
#[derive(Clone, Copy, Debug)]
pub struct ResqInputs<'a> {
    /// The ffn-stage input slabs, each flattened row-major `[rows, n]`, Q32.32 words.
    pub x_slabs: &'a [&'a [i64]],
    /// The input width `n`.
    pub n: usize,
    /// The teacher gate weight, row-major `[inter, n]`, Q32.32 words.
    pub w_g0: &'a [i64],
    /// The teacher up weight, row-major `[inter, n]`, Q32.32 words.
    pub w_u0: &'a [i64],
    /// The student quantised gate weight, row-major `[inter, n]`, Q32.32 words.
    pub wq_g: &'a [i64],
    /// The current up reconstruction, row-major `[inter, n]`, Q32.32 words.
    pub wq_u: &'a [i64],
    /// The teacher down weight, row-major `[hidden, inter]`, Q32.32 words.
    pub w_down: &'a [i64],
    /// The intermediate width `inter`.
    pub inter: usize,
    /// The hidden width `hidden`.
    pub hidden: usize,
}

/// The `-resq` system and its solution: `gop = G ⊙ P_mat + lam·I`, `rhs = GoC·1 + lam`, `s_raw`
/// solves `gop·s = rhs`, `s_clamped` clamps it to `[0.5, 2.0]`.
#[derive(Clone, Debug)]
pub struct ResqSolution {
    /// `G ⊙ P_mat + lam·I`, row-major `[inter, inter]`, Q32.32 words.
    pub gop: Vec<i64>,
    /// `GoC·1 + lam`, length `inter`, Q32.32 words.
    pub rhs: Vec<i64>,
    /// The ridge `0.05·max(mean(diag(G ⊙ P_mat)), 1e-12)`, Q32.32.
    pub lam: i64,
    /// The unclamped solution `gop⁻¹·rhs`, length `inter`, Q32.32 words.
    pub s_raw: Vec<i64>,
    /// The solution clamped to `[0.5, 2.0]`, length `inter`, Q32.32 words.
    pub s_clamped: Vec<i64>,
}

/// `a` (row-major `[r, c]`) transposed to `[c, r]` on the word grid. `O(r·c)` moves and space.
fn transpose(a: &[i64], r: usize, c: usize) -> Vec<i64> {
    let mut out = vec![0_i64; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// `round(a·b)` from the exact `i128` product, one round-to-nearest shift. `O(1)`.
fn mul_q32(a: i64, b: i64) -> Result<i64, ResqError> {
    round_q32_even(i128::from(a) * i128::from(b)).map_err(|_| ResqError::Overflow)
}

/// `round(num / den)` for a positive `den`, `None` when the result leaves `i64`. `O(1)`.
fn div_q32(num: i128, den: i64) -> Option<i64> {
    let d = i128::from(den);
    let n = if num >= 0 {
        num.checked_add(d / 2)?
    } else {
        num.checked_sub(d / 2)?
    };
    i64::try_from(n / d).ok()
}

/// `x` and the least `k ≥ 0` with `max|x|·2⁻ᵏ < 2³⁶` raw, the products' bound, by one exact power
/// of two; `O(len)`.
fn fit(x: &[i64]) -> Result<(Vec<i64>, u32), ResqError> {
    let m = x
        .iter()
        .fold(0_u128, |m, &v| m.max(u128::from(v.unsigned_abs())));
    let k = ilog2_pos(m).map_or(0, |b| (b + 1).saturating_sub(36));
    x.iter()
        .map(|&v| {
            rshift_rne(i128::from(v), k)
                .and_then(|y| i64::try_from(y).ok())
                .ok_or(ResqError::Overflow)
        })
        .collect::<Result<Vec<i64>, ResqError>>()
        .map(|y| (y, k))
}

/// `x·2ᵏ`, the exact scale [`fit`] removed (the shift is exact; `fit`'s rounding is not undone);
/// `O(len)`.
fn unfix(x: &[i64], k: u32) -> Result<Vec<i64>, ResqError> {
    x.iter()
        .map(|&v| v.checked_shl(k).ok_or(ResqError::Overflow))
        .collect()
}

/// `A·B` ([`gemm_nn`]) or `A·Bᵀ` ([`gemm_nt`]), both operands [`fit`] and the product unfitted;
/// `O(m·k·n)` time, `O(m·n)` space.
fn product(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    nt: bool,
) -> Result<Vec<i64>, ResqError> {
    let ((pa, ka), (pb, kb)) = (fit(a)?, fit(b)?);
    let c = if nt {
        gemm_nt(&pa, &pb, m, k, n)
    } else {
        gemm_nn(&pa, &pb, m, k, n)
    };
    unfix(&c.map_err(|_| ResqError::Overflow)?, ka + kb)
}

/// `xᵀ·x`: the pooled unit-weight [`wsyrk`] of a fitted panel; `O(o·n²)` time, `O(n²)` space.
fn gram(x: &[i64], o: usize, n: usize) -> Result<Vec<i64>, ResqError> {
    let (xs, k) = fit(x)?;
    let g = wsyrk(&xs, None, o, n).map_err(|_| ResqError::Overflow)?;
    unfix(&g, 2 * k)
}

/// One slab's `P` rows: `silu(X·W_gᵀ) ⊙ (X·W_uᵀ)`, row-major `[rows, inter]` Q32.32; `Overflow`
/// from a `SiLU`, product or pooled GEMM.
///
/// # Complexity
/// Time `O(rows·n·inter)` word ops in the two pooled products plus `O(rows·inter)` elementwise,
/// above the `10⁶` pool threshold at any production slab. Space `O(rows·inter)`.
fn project(
    slab: &[i64],
    rows: usize,
    n: usize,
    inter: usize,
    wg: &[i64],
    wu: &[i64],
) -> Result<Vec<i64>, ResqError> {
    let gate = product(slab, wg, rows, n, inter, true)?;
    let up = product(slab, wu, rows, n, inter, true)?;
    let mut out = Vec::with_capacity(rows * inter);
    for k in 0..rows * inter {
        let s = silu_fixed(gate[k]).map_err(|_| ResqError::Overflow)?;
        out.push(mul_q32(s, up[k])?);
    }
    Ok(out)
}

/// The clamped per-row multiplier `s_sol`, length `inter`.
///
/// # Errors
/// As [`resq_solve`].
pub fn resq_scale(inputs: &ResqInputs<'_>) -> Result<Vec<i64>, ResqError> {
    Ok(resq_solve(inputs)?.s_clamped)
}

/// Build and solve the `-resq` elementwise-Gram ridge system on the Q32.32 grid: `G` and the
/// per-slab `P_qᵀ·P_q`/`P_qᵀ·P_0`, `GoP = G ⊙ P_mat`, `GoC = G ⊙ C_mat`, the ridge
/// `0.05·max(mean(diag(GoP)), 1e-12)`, and `s = chol_solve(GoP + lam·I, GoC·1 + lam)`.
///
/// # Errors
/// [`ResqError::Empty`] for a zero `inter`/`n`/`hidden`; [`ResqError::ShapeMismatch`] for a length
/// against the declared dims; [`ResqError::NotPositiveDefinite`] on a non-positive `Cholesky` pivot;
/// [`ResqError::Overflow`] on checked arithmetic.
///
/// # Complexity
/// Temporal: `O(hidden·inter² + Σ_s rows_s·(n·inter + inter²) + inter³)`, dominant term
/// `2·rows·inter²` in the pooled `P_qᵀ·P_q`/`P_qᵀ·P_0` pair. The per-row loops (`rhs = GoC·1` at
/// `inter²`, the row-scale fold at `o·g`) are serial: at or near `10⁶` word ops, cheaper than a
/// dispatch. Spatial: `O(inter² + rows_max·inter)` Q32.32 words.
pub fn resq_solve(inputs: &ResqInputs<'_>) -> Result<ResqSolution, ResqError> {
    let (n, inter, hidden) = (inputs.n, inputs.inter, inputs.hidden);
    if n == 0 || inter == 0 || hidden == 0 {
        return Err(ResqError::Empty);
    }
    let inter_n = inter.checked_mul(n).ok_or(ResqError::ShapeMismatch)?;
    let hidden_inter = hidden.checked_mul(inter).ok_or(ResqError::ShapeMismatch)?;
    let inter2 = inter.checked_mul(inter).ok_or(ResqError::ShapeMismatch)?;
    if inputs.w_g0.len() != inter_n
        || inputs.w_u0.len() != inter_n
        || inputs.wq_g.len() != inter_n
        || inputs.wq_u.len() != inter_n
    {
        return Err(ResqError::ShapeMismatch);
    }
    if inputs.w_down.len() != hidden_inter {
        return Err(ResqError::ShapeMismatch);
    }
    for slab in inputs.x_slabs {
        if slab.len() % n != 0 {
            return Err(ResqError::ShapeMismatch);
        }
    }

    let (down, g0, u0, qg, qu) = (
        inputs.w_down,
        inputs.w_g0,
        inputs.w_u0,
        inputs.wq_g,
        inputs.wq_u,
    );

    let g = gram(down, hidden, inter)?;

    let mut p_mat = vec![0_i64; inter2];
    let mut c_mat = vec![0_i64; inter2];
    for slab in inputs.x_slabs {
        let rows = slab.len() / n;
        if rows == 0 {
            continue;
        }
        let p0 = project(slab, rows, n, inter, g0, u0)?;
        let pq = project(slab, rows, n, inter, qg, qu)?;
        let pp = gram(&pq, rows, inter)?;
        let pqt = transpose(&pq, rows, inter);
        let pc = product(&pqt, &p0, inter, rows, inter, false)?;
        for idx in 0..inter2 {
            p_mat[idx] = p_mat[idx].checked_add(pp[idx]).ok_or(ResqError::Overflow)?;
            c_mat[idx] = c_mat[idx].checked_add(pc[idx]).ok_or(ResqError::Overflow)?;
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
        gop,
        rhs,
        lam,
        s_raw: solved.x.clone(),
        s_clamped: solved.x.iter().map(|&v| v.clamp(half, twice)).collect(),
    })
}

/// Fold a per-row multiplier into a plane's stored hierarchical row scale: `rh = quant_r_hier(rh·s)`
/// with `rh` row-major `[o, g]` and `s` length `o`, both Q32.32 words; the product is exact-integer
/// on the grid and feeds [`quant_r_hier`] as words.
///
/// # Errors
/// [`ResqError::Empty`] for a zero `o`/`g`; [`ResqError::ShapeMismatch`] for a length against the
/// shape; [`ResqError::Quant`] if [`quant_r_hier`] refuses.
///
/// # Complexity
/// Temporal: `O(o·g)` serial products plus `quant_r_hier`'s `O(o·g)` — one product per stored entry,
/// which a pool dispatch does not repay; spatial `O(o·g)`.
pub fn rescale_hier(rh: &[i64], s: &[i64], o: usize, g: usize) -> Result<RHier, ResqError> {
    if o == 0 || g == 0 {
        return Err(ResqError::Empty);
    }
    let n = o.checked_mul(g).ok_or(ResqError::ShapeMismatch)?;
    if rh.len() != n || s.len() != o {
        return Err(ResqError::ShapeMismatch);
    }
    let mut scaled = vec![0_i64; n];
    let mut err: Option<ResqError> = None;
    let mut i = 0;
    while i < o && err.is_none() {
        let mut j = 0;
        while j < g && err.is_none() {
            match mul_q32(rh[i * g + j], s[i]) {
                Ok(v) => scaled[i * g + j] = v,
                Err(e) => err = Some(e),
            }
            j += 1;
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(quant_r_hier(&scaled, o, g)?)
}

/// The partial-plane (`pk2_top`) variant of [`rescale_hier`]: the same `s` gathered at `rows`,
/// the plane's global indices.
///
/// # Errors
/// As [`rescale_hier`], with an out-of-range gathered row for `s` as [`ResqError::ShapeMismatch`].
///
/// # Complexity
/// Temporal: `O(rows.len()·g)` serial products plus `quant_r_hier`'s `O(rows.len()·g)`; spatial
/// `O(rows.len()·g)`.
pub fn rescale_hier_top(
    rh: &[i64],
    s: &[i64],
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
    let mut bad = false;
    let mut q = 0;
    while q < rows.len() {
        if rows[q] >= s.len() {
            bad = true;
        }
        q += 1;
    }
    if bad {
        return Err(ResqError::ShapeMismatch);
    }
    let mut scaled = vec![0_i64; n];
    let mut err: Option<ResqError> = None;
    let mut i = 0;
    while i < o && err.is_none() {
        let r = rows[i];
        let mut j = 0;
        while j < g && err.is_none() {
            match mul_q32(rh[i * g + j], s[r]) {
                Ok(v) => scaled[i * g + j] = v,
                Err(e) => err = Some(e),
            }
            j += 1;
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(quant_r_hier(&scaled, o, g)?)
}
