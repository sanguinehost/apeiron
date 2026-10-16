//! GPTQ error feedback over the 8-block E8 ∪ O8 codebook: [`gptq_e8`] and its factor [`feedback_u`].
//!
//! Port of `feedback_u` and `gptq_e8` in `tools/onebit/fold_transfer.py` (removed; at `2f15d6a9`),
//! lines 148–158 and 267–397, for `beam = 1`. One code index per 8 consecutive columns of a
//! 128-column group; the quantisation error of a block is propagated to the later columns of the
//! group and then to the later groups through the feedback factor `U` (`UᵀU = H⁻¹` of the damped
//! Hessian).
//!
//! # Codebook scaling
//!
//! [`e8::codeword`]`(idx)` is the Python `e8_o8_codebook` row `idx` in Q32.32: the real value of
//! coordinate `k` is `codeword(idx)[k] / ONE` with `ONE = 2^32`. Rows `0..128` are the spinors `±1`,
//! `128..240` the D8 roots `±2e_a ±2e_b`, `240..256` the O8 vertices `±√8·e_i`
//! ([`e8::SQRT8_Q32`]); every row has real norm `√8`, so the admission of a codeword is a statement
//! about the codebook the Python builds, not a rescaling of it. [`GptqE8::s`] stores exactly
//! `codeword(idx)[k] / ONE` (an exact f64 division by `2^32`), so `s` decodes back from `idx` with
//! no rounding, and `Wq = scale · s` is the dequantised weight.
//!
//! # Selection rule
//!
//! The block selection is the exact block-Cholesky CVP of the Python (`gptq_e8` lines 312–321), not
//! a Euclidean nearest-codeword search. Per 8-block `m` the working row is right-solved through the
//! trailing upper-triangular block `U_sub = U[b0:b1, b0:b1]`, `Z_w = w_sub·U_sub⁻¹`; the 256
//! codeword images `Z_c = C·U_sub⁻¹` and their squared norms `|Z_c|²` are formed; the code index is
//!
//! ```text
//! idx = argmin_k  a²·|Z_c[k]|² − 2a·⟨Z_w, Z_c[k]⟩
//! ```
//!
//! i.e. the codeword nearest to `a·C` in the whitened norm `(U_subᵀU_sub)⁻¹`, which is the norm the
//! group objective is measured in. `err_sub = Z_w − a·Z_c[idx]` is the whitened residual of the
//! block: it is what the group loop writes to the later columns (line 323) and what the group
//! aggregate pushes onto the later groups (line 394). (The same quantity is
//! `(w_sub − a·s_sel)·U_sub⁻¹`, so only the selection changed with respect to an original-frame
//! nearest-codeword rule; the feedback was already in this frame.)
//!
//! Deviations from the Python, each with its reason:
//!
//! - f64 reassociation. The right-solves (`Z_w`, `Z_c`) are back-substitutions against `U_sub`;
//!   `torch.linalg.solve_triangular` is not specified to the bit, and the codeword images are
//!   recomputed per pass instead of cached per Hessian (`zc_list`/`zcn_list`, line 288) — same
//!   values, same code path.
//! - The trailing group is `g1 = min(g0 + GROUP, n)`, so any `n % 8 == 0` is admitted. The Python
//!   indexes `A[:, gi]` and slices `U[g0:g1, g1:]` over `n // group` groups and is defined only for
//!   `n % 128 == 0`.
//! - The act-order permutation sorts ties (equal block traces) by ascending index; `torch.argsort`
//!   leaves the tie order unspecified.
//! - [`GptqE8Error::Unsupported`] for `beam != 1`: the Python beam-delayed-decision branch
//!   (`gptq_e8` lines 333–378) is not ported.
//! - `feedback_u` refuses a non-positive pivot of the damped Hessian; the Python's
//!   `torch.linalg.cholesky` raises there too.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the kernel indexes row-major [o, n] and [n, n] buffers by r/o/i/j/k/m/b/t, and the \
              integer/float casts are the exact Q32.32 decode codeword(idx)/2^32 of gptq_e8 in \
              fold_transfer.py"
)]

use crate::engine::ONE;
use crate::learn::code::e8::{self, E8Error};
use crate::learn::gptq::{ul_cholesky, upper_tri_inverse};

/// Columns per group: the Python `GROUP`. One scale per (row, group).
pub const GROUP: usize = 128;

/// Columns per code index: one codebook row per 8 weights.
pub const BLOCK: usize = 8;

/// The codebook scale factor of `gptq_e8` (line 303): `a = 0.828 · w_rms`.
pub const E8_SCALE: f64 = 0.828;

/// A refusal of [`feedback_u`] or [`gptq_e8`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptqE8Error {
    /// A buffer length or a dimension parameter violates the expected `[o, n]` / `[n, n]` shape.
    ShapeMismatch,
    /// `o` or `n` is zero.
    ZeroDim,
    /// `n` is not a multiple of [`BLOCK`].
    BlockMisaligned,
    /// `damp` is not finite or not positive.
    InvalidDamping,
    /// The damped Hessian has a non-positive Cholesky pivot: it is not positive-definite.
    NotPositiveDefinite,
    /// A weight or Hessian entry is NaN or infinite.
    NonFinite,
    /// A checked `i64`/`i128` quantity left its range in a codebook helper.
    Overflow,
    /// The code index does not decode: a codebook refusal.
    Codebook,
    /// `beam != 1`: the Python beam-search branch is not ported.
    Unsupported,
}

impl core::fmt::Display for GptqE8Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch => write!(f, "matrix shape or buffer length mismatch"),
            Self::ZeroDim => write!(f, "output or column count must be non-zero"),
            Self::BlockMisaligned => write!(f, "column count is not a multiple of 8"),
            Self::InvalidDamping => write!(f, "damping must be finite and positive"),
            Self::NotPositiveDefinite => write!(f, "Hessian is not positive-definite"),
            Self::NonFinite => write!(f, "weight or Hessian entry is not finite"),
            Self::Overflow => write!(f, "Q32.32 value left the i64 range"),
            Self::Codebook => write!(f, "codebook search left its admitted range"),
            Self::Unsupported => write!(f, "beam > 1 is not ported"),
        }
    }
}

impl std::error::Error for GptqE8Error {}

/// The result of [`gptq_e8`]: the codeword values, the code indices, and the group scales.
#[derive(Clone, Debug, PartialEq)]
pub struct GptqE8 {
    /// Row-major `[o, n]` codeword values: exactly `codeword(idx)[k] / ONE` before scaling.
    pub s: Vec<f64>,
    /// Row-major `[o, n/8]` code index of the 8-block starting at column `8·b`, in the caller's
    /// column order (`act_order` reorders whole 8-blocks, never the columns inside one).
    pub idx: Vec<u16>,
    /// Row-major `[o, ceil(n/GROUP)]` per-group scale `a`; the dequantised weight is `scale·s`.
    pub scale: Vec<f64>,
}

/// [`E8Error`] as a [`GptqE8Error`].
fn map_e8(e: E8Error) -> GptqE8Error {
    match e {
        E8Error::Overflow => GptqE8Error::Overflow,
        E8Error::IndexOutOfRange | E8Error::NonPositiveScale => GptqE8Error::Codebook,
    }
}

/// Solve `x·U_sub = e` in place for the row `e` (length `m`), where `U_sub` is the trailing
/// upper-triangular block `u[b0..b0+m, b0..b0+m]` of the row-major `[n, n]` upper-triangular `u`.
/// Back substitution from the last column: `x_j = (e_j − Σ_{k>j} x_k·U_kj) / U_jj`.
///
/// # Complexity
/// - Temporal: `O(m²)` floating-point multiply-subtracts; `m(m−1)/2` for the sums plus `m` divisions.
///   Measure: arithmetic operations on f64.
/// - Spatial: `O(1)`, in place.
fn solve_upper_block(u: &[f64], n: usize, b0: usize, m: usize, e: &mut [f64]) {
    let mut j = m;
    while j > 0 {
        j -= 1;
        let mut s = e[j];
        let mut k = j + 1;
        while k < m {
            s -= e[k] * u[(b0 + k) * n + b0 + j];
            k += 1;
        }
        e[j] = s / u[(b0 + j) * n + b0 + j];
    }
}

/// The inverse feedback factor `U` of the damped Hessian `H`: upper-triangular with `UᵀU = H⁻¹` for
/// `H = h + damp·mean(diag(h))·I`, and a zero diagonal entry replaced by `1.0` after damping (the
/// Python `feedback_u`, lines 148–158). `U = V⁻¹` with the upper factor `V` of `h = V·Vᵀ`; the
/// Python's `cholesky_upper(sym(H⁻¹))` is the same matrix up to f64 reassociation.
///
/// # Errors
/// [`GptqE8Error::ShapeMismatch`] if `h.len() != n·n`; [`GptqE8Error::ZeroDim`] if `n == 0`;
/// [`GptqE8Error::NonFinite`] if `h` holds a NaN or an infinite entry;
/// [`GptqE8Error::NotPositiveDefinite`] if the damped matrix has a non-positive Cholesky pivot.
///
/// # Complexity
/// - Temporal: `O(n³)` floating-point operations — `n³/6` for the upper Cholesky and `n³/6` for the
///   triangular inverse. Measure: arithmetic operations. Derivation: two triangular sweeps, each
///   summing `Σ_j (n−j)²/2` terms.
/// - Spatial: `O(n²)` f64 words for `h`'s copy, the factor `V` and the inverse `U`.
pub fn feedback_u(h: &[f64], n: usize, damp: f64) -> Result<Vec<f64>, GptqE8Error> {
    if n == 0 {
        return Err(GptqE8Error::ZeroDim);
    }
    if h.len() != n * n {
        return Err(GptqE8Error::ShapeMismatch);
    }
    if h.iter().any(|x| !x.is_finite()) {
        return Err(GptqE8Error::NonFinite);
    }
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    if !mean.is_finite() {
        return Err(GptqE8Error::NonFinite);
    }
    let mut hh = h.to_vec();
    for i in 0..n {
        let v = hh[i * n + i] + damp * mean;
        hh[i * n + i] = if v == 0.0 { 1.0 } else { v };
    }
    let v = ul_cholesky(&hh, n).map_err(|_| GptqE8Error::NotPositiveDefinite)?;
    upper_tri_inverse(&v, n).map_err(|_| GptqE8Error::NotPositiveDefinite)
}

/// The act-order permutation of a Hessian diagonal, at 8-block granularity: groups in descending
/// order of their diagonal sum, and inside a group the 8-blocks in descending order of their
/// diagonal sum, ties by ascending index. Columns inside an 8-block keep their order.
fn block_order(h: &[f64], n: usize) -> (Vec<usize>, Vec<usize>) {
    let nb = n / BLOCK;
    let per_group = GROUP / BLOCK;
    let ng = n.div_ceil(GROUP);
    let mut bt = vec![0.0f64; nb];
    for b in 0..nb {
        for k in 0..BLOCK {
            let i = b * BLOCK + k;
            bt[b] += h[i * n + i];
        }
    }
    let mut gt = vec![0.0f64; ng];
    for b in 0..nb {
        gt[b / per_group] += bt[b];
    }
    let mut groups: Vec<usize> = (0..ng).collect();
    groups.sort_by(|&a, &b| gt[b].total_cmp(&gt[a]).then(a.cmp(&b)));
    let mut blk_perm = vec![0usize; nb];
    for (gp, &g) in groups.iter().enumerate() {
        let s0 = g * per_group;
        let s1 = ((g + 1) * per_group).min(nb);
        let mut sub: Vec<usize> = (s0..s1).collect();
        sub.sort_by(|&a, &b| bt[b].total_cmp(&bt[a]).then(a.cmp(&b)));
        for (t, &b) in sub.iter().enumerate() {
            blk_perm[gp * per_group + t] = b;
        }
    }
    let mut inv_blk = vec![0usize; nb];
    for (p, &b) in blk_perm.iter().enumerate() {
        inv_blk[b] = p;
    }
    (blk_perm, inv_blk)
}

/// One quantisation pass over the group `[g0, g1)` of the permuted Hessian factor `u` and the
/// pristine group block `w0` (`o × (g1−g0)`, row-major): per 8-block the code index of the Python's
/// block-Cholesky CVP, `idx = argmin_k a²|Z_c[k]|² − 2a⟨Z_w, Z_c[k]⟩` with `Z_w = w_sub·U_sub⁻¹` and
/// `Z_c = C·U_sub⁻¹`; the whitened residual `e = Z_w − a·Z_c[idx]`; and the residual propagated onto
/// the later columns of the group (the Python lines 312–323).
///
/// # Errors
/// [`GptqE8Error`] from the codebook decode, or [`GptqE8Error::ShapeMismatch`] on a buffer of the
/// wrong length.
///
/// # Complexity
/// - Temporal: `O([gs/8]·256·8² + o·[gs/8]·(256·8 + 8·gs))` — the 256 codeword images `Z_c` (one
///   8×8 right-solve each) per 8-block, then per row and block the 256×8 scan, the 8×8 solve of
///   `Z_w`, and the intra-group propagation `Σ_blocks 8·(gs − b1)`. Measure: f64
///   multiply-subtracts.
/// - Spatial: `O(o·gs + 256·8)` f64 words for the working copy `wk` and the block's `Z_c`; the
///   outputs are caller buffers.
#[allow(
    clippy::too_many_arguments,
    reason = "the pass writes three separate output buffers plus the working block; grouping them \
              into a struct would not reduce the argument count in the caller"
)]
fn quantise_pass(
    u: &[f64],
    n: usize,
    g0: usize,
    g1: usize,
    w0: &[f64],
    a: &[f64],
    o: usize,
    s_out: &mut [f64],
    e_out: &mut [f64],
    i_out: &mut [u16],
) -> Result<(), GptqE8Error> {
    let gs = g1 - g0;
    let nb = gs / BLOCK;
    if w0.len() != o * gs || s_out.len() != o * gs || e_out.len() != o * gs || i_out.len() != o * nb
    {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let mut wk = w0.to_vec();
    // The block's codeword images `Z_c = C·U_sub⁻¹` (row-major `[256, 8]`) and their squared norms
    // (the Python's `zc_list[m]` and `zcn_list[m]`, lines 287–288).
    let mut zc = vec![0.0f64; e8::CODEBOOK_LEN * BLOCK];
    let mut zcn = vec![0.0f64; e8::CODEBOOK_LEN];
    let mut m = 0;
    while m < nb {
        let b0 = m * BLOCK;
        for k in 0..e8::CODEBOOK_LEN {
            let cw = e8::codeword(k).map_err(map_e8)?;
            let row = &mut zc[k * BLOCK..(k + 1) * BLOCK];
            for t in 0..BLOCK {
                row[t] = cw[t] as f64 / ONE as f64;
            }
            solve_upper_block(u, n, g0 + b0, BLOCK, row);
            let mut n2 = 0.0f64;
            for t in 0..BLOCK {
                n2 += row[t] * row[t];
            }
            zcn[k] = n2;
        }
        for r in 0..o {
            // `Z_w = w_sub·U_sub⁻¹` (the Python line 314).
            let mut zw = [0.0f64; BLOCK];
            for t in 0..BLOCK {
                zw[t] = wk[r * gs + b0 + t];
            }
            solve_upper_block(u, n, g0 + b0, BLOCK, &mut zw);
            let ar = a[r];
            // `cost[k] = a²|Z_c[k]|² − 2a⟨Z_w, Z_c[k]⟩`, first index on ties (the Python 317–319).
            let mut best = 0usize;
            let mut best_cost = f64::INFINITY;
            for k in 0..e8::CODEBOOK_LEN {
                let zrow = &zc[k * BLOCK..(k + 1) * BLOCK];
                let mut dot = 0.0f64;
                for t in 0..BLOCK {
                    dot += zw[t] * zrow[t];
                }
                let cost = ar * ar * zcn[k] - 2.0 * ar * dot;
                if cost < best_cost {
                    best_cost = cost;
                    best = k;
                }
            }
            let cw = e8::codeword(best).map_err(map_e8)?;
            let zsel = &zc[best * BLOCK..(best + 1) * BLOCK];
            let mut e = [0.0f64; BLOCK];
            for t in 0..BLOCK {
                s_out[r * gs + b0 + t] = cw[t] as f64 / ONE as f64;
                // `err_sub = Z_w − a·Z_c[idx]` (the Python line 321).
                e[t] = zw[t] - ar * zsel[t];
            }
            i_out[r * nb + m] = u16::try_from(best).map_err(|_| GptqE8Error::Codebook)?;
            for t in 0..BLOCK {
                e_out[r * gs + b0 + t] = e[t];
            }
            // Intra-group feedback: the block's whitened residual onto every later column of the
            // group, `wk[:, b1:] -= err_sub @ U_blk[b0:b1, b1:]` (the Python line 323).
            let mut c = b0 + BLOCK;
            while c < gs {
                let mut acc = 0.0f64;
                for t in 0..BLOCK {
                    acc += e[t] * u[(g0 + b0 + t) * n + g0 + c];
                }
                wk[r * gs + c] -= acc;
                c += 1;
            }
        }
        m += 1;
    }
    Ok(())
}

/// The whitened scale projection of the `wscale` branch: `u_s = S·U_blk⁻¹`,
/// `a_new = max(a + ⟨e, u_s⟩/⟨u_s, u_s⟩, 1e-12)` and `e_new = e − (a_new − a)·u_s`
/// (the Python lines 327–332). Returns `(a_new, e_new)`.
///
/// # Complexity
/// - Temporal: `O(o·gs²)` — the triangular solve `o·gs²/2` multiply-subtracts and the two inner
///   products `O(o·gs)`. Measure: arithmetic operations on f64.
/// - Spatial: `O(o·gs)` f64 words for `u_s` and `e_new`.
#[allow(
    clippy::too_many_arguments,
    reason = "the projection reads the pass's four buffers and writes two; a tuple-in/tuple-out \
              signature would not make the call site clearer"
)]
fn project_scale(
    u: &[f64],
    n: usize,
    g0: usize,
    gs: usize,
    s_cur: &[f64],
    e_cur: &[f64],
    a: &[f64],
    o: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut u_s = s_cur.to_vec();
    for r in 0..o {
        solve_upper_block(u, n, g0, gs, &mut u_s[r * gs..(r + 1) * gs]);
    }
    let mut a_new = vec![0.0f64; o];
    let mut e_new = vec![0.0f64; o * gs];
    for r in 0..o {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for j in 0..gs {
            num += e_cur[r * gs + j] * u_s[r * gs + j];
            den += u_s[r * gs + j] * u_s[r * gs + j];
        }
        let an = (a[r] + num / den.max(1e-12)).max(1e-12);
        a_new[r] = an;
        for j in 0..gs {
            e_new[r * gs + j] = e_cur[r * gs + j] - (an - a[r]) * u_s[r * gs + j];
        }
    }
    (a_new, e_new)
}

/// The GPTQ error feedback of `gptq_e8` (lines 267–397): per 8-block the block-Cholesky CVP
/// `argmin_k a²|Z_c[k]|² − 2a⟨Z_w, Z_c[k]⟩` over the 256-entry E8 ∪ O8 codebook, with each block's
/// whitened residual pushed onto the later columns of the group and the group's residuals pushed
/// onto the later groups through `U`.
///
/// `w` is row-major `[o, n]`, `h` the symmetric `[n, n]` Hessian. `act_order` visits the groups and
/// 8-blocks in descending Hessian-diagonal order and restores the caller's order in the output;
/// `wscale` runs the whitened scale projection and its second pass after the block pass (two refined
/// candidates, the smaller whitened residual row wins). The group scale is `a = 0.828·w_rms` in
/// both settings (the Python line 303).
///
/// # Errors
/// [`GptqE8Error::Unsupported`] if `beam != 1`; [`GptqE8Error::ZeroDim`] if `o == 0` or `n == 0`;
/// [`GptqE8Error::ShapeMismatch`] if `w.len() != o·n` or `h.len() != n·n`;
/// [`GptqE8Error::BlockMisaligned`] if `n % 8 != 0`; [`GptqE8Error::InvalidDamping`] if `damp <= 0`
/// or `damp` is not finite; [`GptqE8Error::NonFinite`] on a non-finite weight or Hessian entry;
/// [`GptqE8Error::NotPositiveDefinite`] if the damped Hessian has a non-positive Cholesky pivot;
/// [`GptqE8Error::Codebook`] from the exact code index decode.
///
/// # Complexity
/// - Temporal: `O(n³ + o·n³/8²·256·8 + o·n²)` — the factor `U` (`n³/3`), the codeword-image
///   right-solves and the `256·8` scan per 8-block per row, and the two feedback sweeps. Measure:
///   f64 multiply-subtracts. Derivation: `[n/8]` blocks × `256` rows × `8` coordinates, plus
///   `Σ_j (n − j)` multiply-subtracts per row per sweep.
/// - Spatial: `O(n² + o·n)` f64 words — `U`, the working copy of `w`, and the per-group buffers.
///
/// `gptq_e8` is the direct port of the Python's single 130-line body (lines 267–397); keeping it
/// one function is what makes a line-by-line audit against the spec possible.
#[allow(
    clippy::too_many_lines,
    reason = "the port is kept parallel to the Python's single 130-line gptq_e8 body"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the dispatch fixes the eight-parameter signature (o, n, damp, act_order, wscale, beam)"
)]
pub fn gptq_e8(
    w: &[f64],
    h: &[f64],
    o: usize,
    n: usize,
    damp: f64,
    act_order: bool,
    wscale: bool,
    beam: usize,
) -> Result<GptqE8, GptqE8Error> {
    if beam != 1 {
        return Err(GptqE8Error::Unsupported);
    }
    if o == 0 || n == 0 {
        return Err(GptqE8Error::ZeroDim);
    }
    if w.len() != o * n || h.len() != n * n {
        return Err(GptqE8Error::ShapeMismatch);
    }
    if !n.is_multiple_of(BLOCK) {
        return Err(GptqE8Error::BlockMisaligned);
    }
    if !damp.is_finite() || damp <= 0.0 {
        return Err(GptqE8Error::InvalidDamping);
    }
    if w.iter().any(|x| !x.is_finite()) || h.iter().any(|x| !x.is_finite()) {
        return Err(GptqE8Error::NonFinite);
    }
    let nb = n / BLOCK;
    let per_group = GROUP / BLOCK;
    let ng = n.div_ceil(GROUP);
    // The act order permutes whole 8-blocks; `perm_col[c]` is the original column at permuted
    // position `c`, `inv_blk[b]` the permuted position of original block `b`.
    let perm = if act_order {
        let (blk_perm, inv_blk) = block_order(h, n);
        let mut perm_col = vec![0usize; n];
        for c in 0..n {
            perm_col[c] = blk_perm[c / BLOCK] * BLOCK + c % BLOCK;
        }
        Some((perm_col, inv_blk))
    } else {
        None
    };
    let (wp, hp) = match &perm {
        Some((perm_col, _)) => {
            let mut wp = vec![0.0f64; o * n];
            let mut hp = vec![0.0f64; n * n];
            for r in 0..o {
                for c in 0..n {
                    wp[r * n + c] = w[r * n + perm_col[c]];
                }
            }
            for i in 0..n {
                for j in 0..n {
                    hp[i * n + j] = h[perm_col[i] * n + perm_col[j]];
                }
            }
            (wp, hp)
        }
        None => (w.to_vec(), h.to_vec()),
    };
    let u = feedback_u(&hp, n, damp)?;
    let mut wk = wp;
    let mut s_p = vec![0.0f64; o * n];
    let mut idx_p = vec![0u16; o * nb];
    let mut scale_p = vec![0.0f64; o * ng];

    for gi in 0..ng {
        let g0 = gi * GROUP;
        let g1 = (g0 + GROUP).min(n);
        let gs = g1 - g0;
        let gb = gs / BLOCK;
        let mut w0 = vec![0.0f64; o * gs];
        for r in 0..o {
            w0[r * gs..(r + 1) * gs].copy_from_slice(&wk[r * n + g0..r * n + g1]);
        }
        // Inverse-variance weights `w_prec = 1/U_jj²` and the group scale `a = 0.828·w_rms`
        // (the Python lines 301–303). `wscale` does not change the scale rule here — it only adds
        // the projection passes below (unlike `gptq_signs`, whose `wscale` branch uses the
        // inverse-variance weighted mean absolute value, `fold_transfer.py` line 188).
        let mut prec = vec![0.0f64; gs];
        let mut prec_sum = 0.0f64;
        for j in 0..gs {
            let d = u[(g0 + j) * n + g0 + j];
            let p = 1.0 / (d * d).max(1e-12);
            prec[j] = p;
            prec_sum += p;
        }
        let mut a = vec![0.0f64; o];
        for r in 0..o {
            let mut acc = 0.0f64;
            for j in 0..gs {
                let x = w0[r * gs + j];
                acc += x * x * prec[j];
            }
            a[r] = (E8_SCALE * (acc / prec_sum).max(1e-24).sqrt()).max(1e-12);
        }
        let mut s_cur = vec![0.0f64; o * gs];
        let mut e_cur = vec![0.0f64; o * gs];
        let mut i_cur = vec![0u16; o * gb];
        quantise_pass(
            &u, n, g0, g1, &w0, &a, o, &mut s_cur, &mut e_cur, &mut i_cur,
        )?;
        let mut a_best = a;
        let mut s_best = s_cur;
        let mut e_best = e_cur;
        let mut i_best = i_cur;
        if wscale {
            let (a1, e1) = project_scale(&u, n, g0, gs, &s_best, &e_best, &a_best, o);
            let mut s2 = vec![0.0f64; o * gs];
            let mut e2 = vec![0.0f64; o * gs];
            let mut i2 = vec![0u16; o * gb];
            quantise_pass(&u, n, g0, g1, &w0, &a1, o, &mut s2, &mut e2, &mut i2)?;
            let (a2, e2b) = project_scale(&u, n, g0, gs, &s2, &e2, &a1, o);
            // Second pass: keep the candidate with the smaller whitened residual per row.
            let mut a_fin = vec![0.0f64; o];
            let mut s_fin = vec![0.0f64; o * gs];
            let mut e_fin = vec![0.0f64; o * gs];
            let mut i_fin = vec![0u16; o * gb];
            for r in 0..o {
                let n1: f64 = (0..gs).map(|j| e1[r * gs + j] * e1[r * gs + j]).sum();
                let n2: f64 = (0..gs).map(|j| e2b[r * gs + j] * e2b[r * gs + j]).sum();
                let (ar, sr, er, ir): (f64, &[f64], &[f64], &[u16]) = if n2 < n1 {
                    (a2[r], &s2, &e2b, &i2)
                } else {
                    (a1[r], &s_best, &e1, &i_best)
                };
                a_fin[r] = ar;
                s_fin[r * gs..(r + 1) * gs].copy_from_slice(&sr[r * gs..(r + 1) * gs]);
                e_fin[r * gs..(r + 1) * gs].copy_from_slice(&er[r * gs..(r + 1) * gs]);
                i_fin[r * gb..(r + 1) * gb].copy_from_slice(&ir[r * gb..(r + 1) * gb]);
            }
            a_best = a_fin;
            s_best = s_fin;
            e_best = e_fin;
            i_best = i_fin;
        }
        for r in 0..o {
            scale_p[r * ng + gi] = a_best[r];
            s_p[r * n + g0..r * n + g1].copy_from_slice(&s_best[r * gs..(r + 1) * gs]);
            idx_p[r * nb + gi * per_group..r * nb + gi * per_group + gb]
                .copy_from_slice(&i_best[r * gb..(r + 1) * gb]);
        }
        // Inter-group feedback: `wk[:, g1:] -= Err_best @ U[g0:g1, g1:]` (line 394).
        if g1 < n {
            for r in 0..o {
                let mut m = g1;
                while m < n {
                    let mut acc = 0.0f64;
                    for j in 0..gs {
                        acc += e_best[r * gs + j] * u[(g0 + j) * n + m];
                    }
                    wk[r * n + m] -= acc;
                    m += 1;
                }
            }
        }
    }

    let (s, idx, scale) = match &perm {
        None => (s_p, idx_p, scale_p),
        Some((_, inv_blk)) => {
            let mut s = vec![0.0f64; o * n];
            let mut idx = vec![0u16; o * nb];
            let mut scale = vec![0.0f64; o * ng];
            for r in 0..o {
                for c in 0..n {
                    // The permuted position of original column `c`.
                    let p = inv_blk[c / BLOCK] * BLOCK + c % BLOCK;
                    s[r * n + c] = s_p[r * n + p];
                }
                for b in 0..nb {
                    idx[r * nb + b] = idx_p[r * nb + inv_blk[b]];
                }
                for g in 0..ng {
                    scale[r * ng + g] = scale_p[r * ng + inv_blk[g * per_group] / per_group];
                }
            }
            (s, idx, scale)
        }
    };
    Ok(GptqE8 { s, idx, scale })
}
