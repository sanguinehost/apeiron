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
//! The Hadamard product of two symmetric positive-semidefinite matrices (`G` is a Gram, `P_mat` is a
//! Gram of the slab stack) is positive-semidefinite (Schur product theorem), so `GoP + lam·I` is
//! positive-definite for `lam > 0`; [`resq_solve`] solves it with [`crate::learn::ridge::solve_spd`]
//! and returns [`ResqError::NotPositiveDefinite`] if a `Cholesky` pivot is non-positive, with no
//! fallback.
//!
//! # Partiality
//!
//! Every entry point refuses a shape that disagrees with the declared dims ([`ResqError::ShapeMismatch`]),
//! a zero `inter`/`n`/`hidden` ([`ResqError::Empty`]) and a non-finite input ([`ResqError::NonFinite`]).
//! No function here panics on caller input.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_resq.rs`). No extraction or refinement links these bodies to a Lean
//! theorem, so nothing here is `proved`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::must_use_candidate,
    reason = "the kernels index the row-major `[rows, n]`/`[inter, inter]` buffers by `i/j/r` and \
              name the carriers `W_g0`/`W_u0`/`Wq_g`/`Wq_u`/`Wg`/`P_mat`/`C_mat`/`GoP`/`GoC` as the \
              reference does; the `inter as f64` cast is the mean's divisor after `inter > 0` is \
              checked"
)]

use crate::learn::fold::fit::quant::{quant_r_hier, QuantError, RHier};
use crate::learn::ridge::solve_spd;

/// A refusal from the `-resq` row-scale step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResqError {
    /// A zero `inter`, `n` or `hidden`, or an empty rescale row set.
    Empty,
    /// A slice length that disagrees with the declared shape.
    ShapeMismatch,
    /// A non-finite entry in an input.
    NonFinite,
    /// The elementwise-Gram system had a non-positive `Cholesky` pivot.
    NotPositiveDefinite,
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
/// `[0.5, 2.0]` (the value the production step applies).
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

/// The clamped per-row multiplier `s_sol` of the `-resq` step, length `inter`.
///
/// # Errors
///
/// As [`resq_solve`].
///
/// # Complexity
///
/// Temporal: `O(hidden·inter² + Σ_s rows_s·(n·inter + inter²) + inter³)` — the `inter³` is the
/// `Cholesky` solve. The dominant term is the `P_mat`/`C_mat` accumulation, `2·rows·inter²` flops
/// over all slabs. Spatial: `O(inter² + rows_max·inter)` f64 words.
pub fn resq_scale(inputs: &ResqInputs<'_>) -> Result<Vec<f64>, ResqError> {
    Ok(resq_solve(inputs)?.s_clamped)
}

/// Build and solve the `-resq` elementwise-Gram ridge system.
///
/// # Errors
///
/// [`ResqError::Empty`] if `inter`, `n` or `hidden` is zero; [`ResqError::ShapeMismatch`] if a
/// weight length disagrees with `inter × n` / `hidden × inter` or a slab length is not a multiple of
/// `n`; [`ResqError::NonFinite`] if any input entry is not finite;
/// [`ResqError::NotPositiveDefinite`] if the `Cholesky` factor of `GoP + lam·I` hits a non-positive
/// pivot.
///
/// # Complexity
///
/// Temporal: `O(hidden·inter² + Σ_s rows_s·(n·inter + inter²) + inter³)`; dominant term
/// `2·rows·inter²` flops (the `P_mat` and `C_mat` accumulations). Spatial:
/// `O(inter² + rows_max·inter)` f64 words.
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
    }
    let mut finite = all_finite(w_g0) && all_finite(w_u0) && all_finite(wq_g) && all_finite(wq_u);
    finite &= all_finite(w_down);
    for slab in x_slabs {
        finite &= all_finite(slab);
    }
    if !finite {
        return Err(ResqError::NonFinite);
    }

    // G = W_downᵀ·W_down, `inter × inter`.
    let mut g = vec![0.0_f64; inter2];
    for h in 0..hidden {
        for i in 0..inter {
            let w_hi = w_down[h * inter + i];
            for j in 0..inter {
                g[i * inter + j] += w_hi * w_down[h * inter + j];
            }
        }
    }

    let mut p_mat = vec![0.0_f64; inter2];
    let mut c_mat = vec![0.0_f64; inter2];
    for slab in x_slabs {
        let rows = slab.len() / n;
        if rows == 0 {
            continue;
        }
        let mut p0 = vec![0.0_f64; rows * inter];
        let mut pq = vec![0.0_f64; rows * inter];
        for r in 0..rows {
            let xr = &slab[r * n..r * n + n];
            for i in 0..inter {
                let mut ag = 0.0_f64;
                let mut au = 0.0_f64;
                let mut agq = 0.0_f64;
                let mut auq = 0.0_f64;
                for j in 0..n {
                    let x = xr[j];
                    ag += x * w_g0[i * n + j];
                    au += x * w_u0[i * n + j];
                    agq += x * wq_g[i * n + j];
                    auq += x * wq_u[i * n + j];
                }
                p0[r * inter + i] = silu(ag) * au;
                pq[r * inter + i] = silu(agq) * auq;
            }
        }
        for i in 0..inter {
            for j in 0..inter {
                let mut pp = 0.0_f64;
                let mut pc = 0.0_f64;
                for r in 0..rows {
                    pp += pq[r * inter + i] * pq[r * inter + j];
                    pc += pq[r * inter + i] * p0[r * inter + j];
                }
                p_mat[i * inter + j] += pp;
                c_mat[i * inter + j] += pc;
            }
        }
    }

    // GoP = G ⊙ P_mat; rhs = (G ⊙ C_mat)·1; the ridge from the GoP diagonal.
    let mut gop = vec![0.0_f64; inter2];
    let mut rhs = vec![0.0_f64; inter];
    for i in 0..inter {
        for j in 0..inter {
            gop[i * inter + j] = g[i * inter + j] * p_mat[i * inter + j];
        }
    }
    let mut mean_diag = 0.0_f64;
    for i in 0..inter {
        mean_diag += gop[i * inter + i];
    }
    mean_diag /= inter as f64;
    let lam = 0.05 * mean_diag.max(1e-12);
    for i in 0..inter {
        gop[i * inter + i] += lam;
    }
    for i in 0..inter {
        let mut s = 0.0_f64;
        for j in 0..inter {
            s += g[i * inter + j] * c_mat[i * inter + j];
        }
        rhs[i] = s + lam;
    }

    let s_raw = solve_spd(&gop, &rhs, inter, 1).map_err(|_| ResqError::NotPositiveDefinite)?;
    let s_clamped: Vec<f64> = s_raw.iter().map(|&v| v.clamp(0.5, 2.0)).collect();
    Ok(ResqSolution {
        gop,
        rhs,
        lam,
        s_raw,
        s_clamped,
    })
}

/// Fold a per-row multiplier into a plane's stored hierarchical row scale: `rh = quant_r_hier(rh·s)`
/// with `rh` row-major `[o, g]` and `s` length `o`.
///
/// This is the `-resq` replacement for `Wp *= s[:, None]`: the multiplier enters the row scale before
/// the grid quantization, so it is absorbed into `quant_r_hier`'s bits.
///
/// # Errors
///
/// [`ResqError::Empty`] if `o` or `g` is zero; [`ResqError::ShapeMismatch`] if `rh.len() != o·g` or
/// `s.len() != o`; [`ResqError::NonFinite`] if any entry is not finite;
/// [`ResqError::Quant`] if [`quant_r_hier`] refuses.
///
/// # Complexity
///
/// Temporal: `O(o·g)` for the elementwise product plus `quant_r_hier`'s `O(o·g)`; spatial `O(o·g)`.
pub fn rescale_hier(rh: &[f64], s: &[f64], o: usize, g: usize) -> Result<RHier, ResqError> {
    if o == 0 || g == 0 {
        return Err(ResqError::Empty);
    }
    let n = o.checked_mul(g).ok_or(ResqError::ShapeMismatch)?;
    if rh.len() != n || s.len() != o {
        return Err(ResqError::ShapeMismatch);
    }
    if !all_finite(rh) || !all_finite(s) {
        return Err(ResqError::NonFinite);
    }
    let mut scaled = vec![0.0_f64; n];
    for i in 0..o {
        let si = s[i];
        for j in 0..g {
            scaled[i * g + j] = rh[i * g + j] * si;
        }
    }
    Ok(quant_r_hier(&scaled, o, g)?)
}

/// The partial-plane (`pk2_top`) variant of [`rescale_hier`]: the same `s` is gathered at `rows`
/// (the plane's global row indices) before the fold, `rh = quant_r_hier(rh·s[rows])` with `rh` the
/// partial plane's `[rows.len(), g]` scale.
///
/// # Errors
///
/// [`ResqError::Empty`] if `rows` or `g` is zero; [`ResqError::ShapeMismatch`] if `rh.len() !=
/// rows.len()·g` or a gathered index is out of range for `s`; [`ResqError::NonFinite`] if any entry
/// is not finite; [`ResqError::Quant`] if [`quant_r_hier`] refuses.
///
/// # Complexity
///
/// Temporal: `O(rows.len()·g)`; spatial `O(rows.len()·g)`.
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
    if !all_finite(rh) || !all_finite(s) {
        return Err(ResqError::NonFinite);
    }
    let mut scaled = vec![0.0_f64; n];
    for (i, &r) in rows.iter().enumerate() {
        let si = *s.get(r).ok_or(ResqError::ShapeMismatch)?;
        for j in 0..g {
            scaled[i * g + j] = rh[i * g + j] * si;
        }
    }
    Ok(quant_r_hier(&scaled, o, g)?)
}

/// `silu(x) = x·sigmoid(x) = x / (1 + e^{-x})`.
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Whether every entry of `x` is finite.
fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}
