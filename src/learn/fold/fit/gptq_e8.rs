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
//! `idx = argmin_k a²·|Z_c[k]|² − 2a·⟨Z_w, Z_c[k]⟩`, i.e. the codeword nearest to `a·C` in the
//! whitened norm `(U_subᵀU_sub)⁻¹`, which is the norm the
//! group objective is measured in. `err_sub = Z_w − a·Z_c[idx]` is the whitened residual of the
//! block: it is what the group loop writes to the later columns (line 323) and what the group
//! aggregate pushes onto the later groups (line 394). (The same quantity is
//! `(w_sub − a·s_sel)·U_sub⁻¹`, so only the selection changed with respect to an original-frame
//! nearest-codeword rule; the feedback was already in this frame.)
//!
//! # Beam search
//!
//! `beam > 1` ports the Python's delayed-decision branch (`gptq_e8` lines 333–378): per 8-block `m`
//! every live beam `b` is expanded over the 256 codewords, `tot_err2[b,c] = cum_err2[b] +
//! |Z_w[b] − a·Z_c[c]|²`, and the `K = min(beam, B·256)` smallest pairs are kept with `B ← K`
//! (lines 350–366); the parent `flat/256` and code `flat%256` reorder the working block, signs and
//! residuals, and the winner is the least cumulative error (`beam 0` after the ascending selection,
//! line 378) or, under `wscale`, the beam with the smallest projected residual (lines 367–376). The
//! `torch.topk(..., largest=False)` order is an ascending partial selection with the tie rule
//! **smaller flat index first** (`flat = parent·256 + code`, [`select_k`]); [`beam_pass`] runs it on
//! the exact `i128` cost. `beam = 1` takes the Python's `beam <= 1` single path (line 305) unchanged.
//!
//! Deviations from the Python, each with its reason:
//!
//! - The trailing group is `g1 = min(g0 + GROUP, n)`, so any `n % 8 == 0` is admitted. The Python
//!   indexes `A[:, gi]` and slices `U[g0:g1, g1:]` over `n // group` groups and is defined only for
//!   `n % 128 == 0`.
//! - The act-order permutation sorts ties (equal block traces) by ascending index; `torch.argsort`
//!   leaves the tie order unspecified.
//! - [`GptqE8Error::BeamOutOfRange`] for `beam == 0` or `beam > MAX_BEAM` (16). The Python accepts
//!   any `beam >= 1` (its `K = min(beam, B·256)` saturates at 256); the production arm needs
//!   `beam = 4` and `1..=16` is the ported domain.
//! - `feedback_u` refuses a non-positive pivot of the damped Hessian; the Python's
//!   `torch.linalg.cholesky` raises there too.
//! - `feedback_u` factors the damped Hessian through the Q32.32 [`inv_factor_upper`] (one exact
//!   `i128` accumulation and one round-to-nearest per stored entry); the captured `beam = 1` golden
//!   scales move with it by `≈ 4e-11` relative. Measurements: `tests/learn_solve_inv.rs`.
//!
//! # Integer carrier (lanes `q-l1a`–`q-l1d`, `int-gptq-w`)
//!
//! The selection is integer end to end. The feedback factor, the act order, the group scale, both
//! block passes, the whitened scale projection, the second-pass choice and the inter-group feedback
//! are all carried in integers: `feedback_u` returns a [`UFac`] (`U = Uhat·diag(2^-d)`),
//! [`block_order`] orders on `i128` diagonal sums, the group scale is an [`Fx`], and
//! [`quantise_pass`] and [`beam_pass`] evaluate the block objective as an exact `i128` scan,
//! carrying the residual and codeword image as [`Fx`]. The `wscale` projection is
//! `project_scale_int` ([`tri_upper_solve`] then exact `i128` inner products, one division), the
//! second-pass choice compares the two candidates' exact `i128` residual norms, and
//! `inter_group_feedback` updates the integer working rows with one round-to-nearest per stored
//! entry.
//!
//! # The boundary
//!
//! [`gptq_e8`] takes and returns Q32.32 `i64` words: `w`, `h` and `damp` are words in, and
//! [`GptqE8::s`] / [`GptqE8::scale`] are words out (`s` exactly the codeword word, `scale` the
//! word `RNE(a·2³²)`). Inside, the weight is renormalised once to its block floating point
//! ([`w_to_rows`]) and the Hessian is renormed once at its even global exponent ([`h_words`]); the
//! `2^-g` global exponent `g` is even, so it is an exact scaling. No `f64` value, input or output,
//! crosses this boundary.

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

use super::gptq_e8_carrier::{
    blen, block_order, div_shift_rne, fx_to_word, group_scale_int, inter_group_feedback, min_exp,
    narrow_q, project_scale_int, project_scale_row, shift_left_i128, sum_sq_at, w_to_rows,
};
pub use super::gptq_e8_carrier::{feedback_u, UFac};
use crate::learn::code::e8::{self, E8Error};
use crate::learn::solve::q32::{rshift_rne, Fx};

/// The alignment ceiling of a block: `58` keeps `|cw_t·G_t|` and `|H_t|` below `2^58`, so the
/// single-path block's sum of squares stays below `8·2^118` and every live beam's below `2^121`,
/// and the carried `cum` below `2^126`; every `i128` accumulation in the passes is in range.
const ALIGN: i32 = 58;

/// Columns per group: the Python `GROUP`. One scale per (row, group).
pub const GROUP: usize = 128;

/// Columns per code index: one codebook row per 8 weights.
pub const BLOCK: usize = 8;

/// The codebook scale factor of `gptq_e8` (line 303): `a = 0.828 · w_rms`, as the Q32.32 word
/// `RNE(0.828 · 2^32)`. `0.828 · 4_294_967_296 = 3_556_232_921.088`, so the rounded word is
/// `3_556_232_921` and `E8_SCALE · 2^-32` differs from `0.828` by `0.088 · 2^-32 ≈ 2.0e-11`
/// (relative `2.5e-11`). The integer group scale multiplies by this word; the factor never enters
/// `f64`.
pub const E8_SCALE: i64 = 3_556_232_921;

/// The largest admitted beam width. `K = min(beam, B·256)` saturates at 256, and the production arm
/// uses `beam = 4`; the ported domain is `1..=MAX_BEAM`.
pub const MAX_BEAM: usize = 16;

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
    /// `beam` is 0 or greater than [`MAX_BEAM`]: the beam width is outside the ported `1..=16`.
    BeamOutOfRange,
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
            Self::BeamOutOfRange => write!(f, "beam width is outside 1..=16"),
        }
    }
}

impl std::error::Error for GptqE8Error {}

/// The result of [`gptq_e8`]: the codeword values, the code indices, and the group scales.
#[derive(Clone, Debug, PartialEq)]
pub struct GptqE8 {
    /// Row-major `[o, n]` codeword words: exactly `codeword(idx)[k]` (the Q32.32 row) before
    /// scaling.
    pub s: Vec<i64>,
    /// Row-major `[o, n/8]` code index of the 8-block starting at column `8·b`, in the caller's
    /// column order (`act_order` reorders whole 8-blocks, never the columns inside one).
    pub idx: Vec<u16>,
    /// Row-major `[o, ceil(n/GROUP)]` per-group scale word `RNE(a·2³²)`; the dequantised
    /// weight is `scale·s` in Q64.64.
    pub scale: Vec<i64>,
}

/// [`E8Error`] as a [`GptqE8Error`].
fn map_e8(e: E8Error) -> GptqE8Error {
    match e {
        E8Error::Overflow => GptqE8Error::Overflow,
        E8Error::IndexOutOfRange | E8Error::NonPositiveScale => GptqE8Error::Codebook,
    }
}

/// One quantisation pass over the group `[g0, g1)` of the permuted Hessian factor `u` and the
/// group's per-row integer weight rows `mant`/`e_row` (row-major `o × (g1 − g0)`, the real weight
/// is `mant[r·gs + c] · 2^e_row[r]`), at the per-row group scale `a` (the [`group_scale_int`]
/// outputs).
///
/// # The exact step cost
///
/// The block's strictly below-diagonal factor entries are zero, so the block right-solve is the
/// per-coordinate quotient `Z[t] = w[t]·2^(32 + d_j)/u.hat[j·n + j]`. With the block's alignment
/// exponent `K` the argmin of `Σ_t (cw[k][t]·G_t − H_t)²` is the Python's block objective argmin,
/// exactly, in `i128`; the residual `E[t] = 2^K·err_sub[t]` is recomputed from the numerator form so
/// the rounded `G_t`/`H_t` never enter it. The intra-group feedback `wk[:, b1:] -= E @ U[b0:b1,
/// b1:]` is one `i128` contraction and one RNE to the row grid per later column. Rows are processed
/// serially, each independent of the others.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] on a buffer of the wrong length, [`GptqE8Error::Codebook`] from
/// the code index decode, [`GptqE8Error::NotPositiveDefinite`] on a non-positive diagonal and
/// [`GptqE8Error::Overflow`] on a checked `i128` operation.
///
/// # Complexity
///
/// Temporal `O(o·[gs/8]·(256·8 + gs))` checked `i128` multiply-adds; spatial `O(gs)` working words
/// plus the `256·8` `i64` codebook.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the pass writes three separate output buffers plus the working mantissas and reads \
              the factor, the rows and the scale; grouping them into a struct would not reduce the \
              caller's argument count, and the body is kept one block loop parallel to the Python's \
              lines 312–323"
)]
pub(super) fn quantise_pass(
    u: &UFac,
    n: usize,
    g0: usize,
    g1: usize,
    mant: &[i64],
    e_row: &[i32],
    a: &[Fx],
    o: usize,
    s_out: &mut [Fx],
    e_out: &mut [Fx],
    i_out: &mut [u16],
) -> Result<(), GptqE8Error> {
    let gs = g1 - g0;
    let nb = gs / BLOCK;
    if mant.len() != o * gs
        || e_row.len() != o
        || a.len() != o
        || s_out.len() != o * gs
        || e_out.len() != o * gs
        || i_out.len() != o * nb
    {
        return Err(GptqE8Error::ShapeMismatch);
    }
    // The 256 codebook rows once per call.
    let mut cw = [[0i64; BLOCK]; e8::CODEBOOK_LEN];
    for (k, row) in cw.iter_mut().enumerate() {
        *row = e8::codeword(k).map_err(map_e8)?;
    }
    let mut wk = vec![0i64; gs];
    let mut gvec = [0i128; BLOCK];
    let mut hvec = [0i128; BLOCK];
    let mut esel = [0i128; BLOCK];
    for lr in 0..o {
        wk.copy_from_slice(&mant[lr * gs..(lr + 1) * gs]);
        let er = i64::from(e_row[lr]);
        let am = i128::from(a[lr].m);
        let ae = i64::from(a[lr].e);
        // `Δ = er + 32 − ae`, the relative scale of the row grid and the `a·2^32` image.
        let delta = er + 32 - ae;
        for m in 0..nb {
            let b0 = m * BLOCK;
            // (1) The block's alignment exponent `K = ALIGN − kb`, `kb` the magnitude class of
            // the largest whitened term of the block.
            let mut kb = 0i32;
            for t in 0..BLOCK {
                let j = g0 + b0 + t;
                let hjj = u.hat[j * n + j];
                if hjj <= 0 {
                    return Err(GptqE8Error::NotPositiveDefinite);
                }
                let bh = blen(i128::from(hjj));
                let bm = blen(i128::from(wk[b0 + t]));
                if bm > 0 {
                    kb = kb.max(bm + e_row[lr] + 32 + u.d[j] - bh + 1);
                }
                if am != 0 {
                    // The `34` is the magnitude class of the largest codebook word
                    // (`sqrt8·2^32`), which multiplies `G_t`.
                    kb = kb.max(34 + blen(am) + a[lr].e + u.d[j] - bh + 1);
                }
            }
            let kal = i64::from(ALIGN - kb);
            // (2) `G_t = RNE(a·2^-32·S_t)`, `H_t = RNE(w_t·S_t)`.
            for t in 0..BLOCK {
                let j = g0 + b0 + t;
                let hjj = i128::from(u.hat[j * n + j]);
                let dj = i64::from(u.d[j]);
                gvec[t] = div_shift_rne(am, kal + dj + ae, hjj)?;
                hvec[t] = div_shift_rne(i128::from(wk[b0 + t]), er + kal + 32 + dj, hjj)?;
            }
            // (3) The exact 256-way scan, ties to the smaller index.
            let mut best = 0usize;
            let mut best_cost = i128::MAX;
            for (k, row) in cw.iter().enumerate() {
                let mut cost: i128 = 0;
                for t in 0..BLOCK {
                    let Some(e) = i128::from(row[t])
                        .checked_mul(gvec[t])
                        .and_then(|v| v.checked_sub(hvec[t]))
                    else {
                        return Err(GptqE8Error::Overflow);
                    };
                    let Some(e2) = e.checked_mul(e) else {
                        return Err(GptqE8Error::Overflow);
                    };
                    let Some(next) = cost.checked_add(e2) else {
                        return Err(GptqE8Error::Overflow);
                    };
                    cost = next;
                }
                if cost < best_cost {
                    best_cost = cost;
                    best = k;
                }
            }
            // (4) The exact residual `E_t = 2^K·e[t]` from the numerator form: the rounded
            // `G_t`/`H_t` above decide the argmin but must **not** leak their `2^-26` rounding
            // into the residual the projection and the feedback read.
            let row = &cw[best];
            for t in 0..BLOCK {
                let j = g0 + b0 + t;
                let dj = i64::from(u.d[j]);
                let va = am
                    .checked_mul(i128::from(row[t]))
                    .ok_or(GptqE8Error::Overflow)?;
                let (iv, phi) = if delta >= 0 {
                    let sh = u32::try_from(delta).map_err(|_| GptqE8Error::Overflow)?;
                    let vw = i128::from(wk[b0 + t])
                        .checked_shl(sh)
                        .ok_or(GptqE8Error::Overflow)?;
                    (
                        vw.checked_sub(va).ok_or(GptqE8Error::Overflow)?,
                        kal + ae + dj,
                    )
                } else {
                    let sh = u32::try_from(-delta).map_err(|_| GptqE8Error::Overflow)?;
                    let sh = va.checked_shl(sh).ok_or(GptqE8Error::Overflow)?;
                    (
                        i128::from(wk[b0 + t])
                            .checked_sub(sh)
                            .ok_or(GptqE8Error::Overflow)?,
                        kal + er + 32 + dj,
                    )
                };
                let e = div_shift_rne(iv, phi, i128::from(u.hat[j * n + j]))?;
                esel[t] = e;
                s_out[lr * gs + b0 + t] = Fx { m: row[t], e: -32 };
                // `E_t` is already `2^K·(Z_w − a·Z_c)` in the positive convention the Python's
                // `err_sub` uses, so it is stored unnegated.
                e_out[lr * gs + b0 + t] =
                    Fx::from_i128(e, -i32::try_from(kal).map_err(|_| GptqE8Error::Overflow)?)
                        .map_err(|_| GptqE8Error::Overflow)?;
            }
            i_out[lr * nb + m] = u16::try_from(best).map_err(|_| GptqE8Error::Codebook)?;
            let mut c = b0 + BLOCK;
            while c < gs {
                let mut acc: i128 = 0;
                for t in 0..BLOCK {
                    let Some(pr) =
                        esel[t].checked_mul(i128::from(u.hat[(g0 + b0 + t) * n + g0 + c]))
                    else {
                        return Err(GptqE8Error::Overflow);
                    };
                    let Some(next) = acc.checked_add(pr) else {
                        return Err(GptqE8Error::Overflow);
                    };
                    acc = next;
                }
                // `Δw = −acc · 2^(−K − 32 − d_c)`, one RNE rounding to the row's grid.
                let sh = -(kal + 32 + i64::from(u.d[g0 + c]) + er);
                let neg = acc.checked_neg().ok_or(GptqE8Error::Overflow)?;
                let dm = if sh >= 0 {
                    match u32::try_from(sh)
                        .ok()
                        .and_then(|k| 1_i128.checked_shl(k))
                        .and_then(|p| neg.checked_mul(p))
                    {
                        Some(v) => v,
                        None => return Err(GptqE8Error::Overflow),
                    }
                } else {
                    match u32::try_from(-sh).ok().and_then(|k| rshift_rne(neg, k)) {
                        Some(v) => v,
                        None => return Err(GptqE8Error::Overflow),
                    }
                };
                let cur = i128::from(wk[c]);
                wk[c] = narrow_q(cur.checked_add(dm).ok_or(GptqE8Error::Overflow)?)?;
                c += 1;
            }
        }
    }
    Ok(())
}

/// The `k` smallest `(cost, flat index)` pairs of `cost` by ascending cost, ties to the smaller index:
/// the `torch.topk(..., largest=False)` order of the Python line 352 (`O(len·k)` `i128`
/// comparisons on a sorted `k`-window, `O(k)` space; `len = B·256`, `k ≤ MAX_BEAM`).
pub(super) fn select_k(cost: &[i128], k: usize, out: &mut Vec<(usize, i128)>) {
    out.clear();
    for (i, &v) in cost.iter().enumerate() {
        let pos = out.partition_point(|&(oi, ov)| ov < v || (ov == v && oi < i));
        if pos < k {
            out.insert(pos, (i, v));
            out.truncate(k);
        }
    }
}

/// The `wscale` second-pass tie rule: the second candidate replaces the first only on a **strictly**
/// smaller exact `i128` whitened residual norm, so equal norms keep pass 1 (the Python's
/// `(Err_new*Err_new).sum < (Err1*Err1).sum`). `O(1)`.
pub(super) fn second_pass_wins(n1: i128, n2: i128) -> bool {
    n2 < n1
}

/// One beam pass over the group `[g0, g1)` of the permuted factor `u` and the group's per-row
/// integer weight rows `mant`/`e_row` (real weight `mant[r·gs + c]·2^e_row[r]`) at the per-row group
/// scale `a` (the [`group_scale_int`] outputs): the Python's lines 333–378.
///
/// # The exact step cost
///
/// The block right-solve is the per-coordinate quotient of [`quantise_pass`] (the block's strictly
/// below-diagonal entries are zero), so beam `b`'s whitened block row is
/// `Z_w[t] = w[b][t]·2^(32 + d_j)/u.hat[j·n + j]`. With `S_t` the block's scaling at exponent `K`,
/// `G_t = RNE(a·2^-32·S_t)` and `H_t = RNE(w[b][t]·S_t)`:
///
/// `|Z_w − a·Z_c[k]|²·2^(2K) = Σ_t (cw[k][t]·G_t − H_t)²`
///
/// exactly, with no rounding inside the 256-way scan. The beams are compared **across parents with
/// different `Z_w`**, so the `|Z_w|²` term is kept — unlike [`quantise_pass`], where it is common to
/// all 256 candidates — and `cum[b]` is carried on.
///
/// # The cumulative error and its headroom
///
/// `cum[b]` is an `i128` at the **per-row-per-block** exponent `−2K`: the stored integer is
/// `cum_true[b]·2^(2K)`. `K = ALIGN − kb` with `kb` the largest magnitude class of a whitened term
/// over **all live beams** of the block, so every beam shares one `K` and one scale and the costs are
/// directly comparable. When `K` changes, `cum` is rescaled to the new `2^(2K)`: an exact checked
/// left shift for `K` increasing, one round-to-nearest right shift for `K` decreasing.
///
/// The `+1` in `kb` makes `|cw_t·G_t| < 2^58` and `|H_t| < 2^58` strict for every live beam, so
/// `|cw_t·G_t − H_t| < 2^59` and a block's step error is `< 8·2^118 = 2^121`. A group holds
/// `gs/8 ≤ 16` blocks, so at one scale `|cum| < 2^125` and `|cum + step| < 2^126`, inside `i128`
/// (max `2^127 − 1`); the upshift `cum·2^(2ΔK)` is a checked multiply and returns
/// [`GptqE8Error::Overflow`] when the result leaves `i128`.
///
/// # The rebuild, feedback and `wscale` tail
///
/// The `K = min(beam, B·256)` smallest `(cost, flat)` pairs (`flat = parent·256 + code`) are selected
/// by [`select_k`]; each retained beam copies its parent's working row, codeword image and residual,
/// and its block residual `E_t = 2^K·(Z_w − a·Z_c)` is recomputed exactly from the numerator form
/// (as in [`quantise_pass`], so `G_t`/`H_t` decide the argmin but never enter the residual). The
/// residual feeds the later columns, `wk[b][b1:] −= E·U_blk[b0:b1, b1:]`: an `i128` contraction and
/// one round-to-nearest per stored mantissa to the row grid `2^e_row`.
///
/// `wscale` projects the final `≤ MAX_BEAM` beams: `u_S = S_b·U_blk⁻¹`,
/// `a_cand = max(a + ⟨E, u_S⟩/⟨u_S, u_S⟩, floor)`, the smallest projected residual wins
/// (lines 367–376). The projection is `project_scale_int` on the carrier: the triangular solve, the
/// two exact `i128` inner products, one division and one `Fx` rounding per stored residual entry;
/// the beams are compared on their exact `i128` residual norms at a shared exponent
/// ([`sum_sq_at`], [`min_exp`]).
///
/// Rows are processed serially; `B` depends on the block index alone, so each row's beam state is
/// independent.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] on a wrong-length buffer, [`GptqE8Error::Codebook`] from the code
/// index decode, [`GptqE8Error::NotPositiveDefinite`] on a non-positive diagonal and
/// [`GptqE8Error::Overflow`] on a checked `i128` operation.
///
/// # Complexity
///
/// Temporal `O(o·[gs/8]·beam·256·8 + o·Σ_blocks beam·(gs − b1) + o·beam·gs²)` `i128` multiply-adds;
/// spatial `O(beam·gs + beam·256)` row-local beam state plus the `256·8` `i64` codebook.
#[allow(
    clippy::too_many_arguments,
    reason = "the pass writes three output buffers plus the working rows and reads the factor, the \
              rows and the scale; grouping them into a struct would not reduce the caller's \
              argument count"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the port is kept parallel to the Python's single beam-branch body (lines 333–378)"
)]
pub(super) fn beam_pass(
    u: &UFac,
    n: usize,
    g0: usize,
    g1: usize,
    mant: &[i64],
    e_row: &[i32],
    a: &[Fx],
    o: usize,
    beam: usize,
    wscale: bool,
    s_out: &mut [Fx],
    e_out: &mut [Fx],
    i_out: &mut [u16],
) -> Result<Vec<Fx>, GptqE8Error> {
    let gs = g1 - g0;
    let nb = gs / BLOCK;
    if mant.len() != o * gs
        || e_row.len() != o
        || a.len() != o
        || s_out.len() != o * gs
        || e_out.len() != o * gs
        || i_out.len() != o * nb
    {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let cap = beam.min(e8::CODEBOOK_LEN);
    let stride = cap * gs;
    let mut cw = [[0i64; BLOCK]; e8::CODEBOOK_LEN];
    for (k, row) in cw.iter_mut().enumerate() {
        *row = e8::codeword(k).map_err(map_e8)?;
    }
    let mut wk_c = vec![0i64; stride];
    let mut s_c = vec![Fx { m: 0, e: 0 }; stride];
    let mut e_c = vec![Fx { m: 0, e: 0 }; stride];
    let mut cum_c = vec![0i128; cap];
    let mut idx_c = vec![0u16; cap * nb];
    let mut wk_x = vec![0i64; stride];
    let mut s_x = vec![Fx { m: 0, e: 0 }; stride];
    let mut e_x = vec![Fx { m: 0, e: 0 }; stride];
    let mut cum_x = vec![0i128; cap];
    let mut idx_x = vec![0u16; cap * nb];
    let mut cost = vec![0i128; cap * e8::CODEBOOK_LEN];
    let mut base = vec![0i128; cap];
    let mut sel: Vec<(usize, i128)> = Vec::with_capacity(cap);
    let mut eproj = vec![Fx { m: 0, e: 0 }; stride];
    let mut a_cand = vec![Fx { m: 0, e: 0 }; cap];
    let mut cerr = vec![0i128; cap];
    let mut gvec = [0i128; BLOCK];
    let mut hvec = [0i128; BLOCK];
    let mut esel = [0i128; BLOCK];
    let mut a_new = vec![Fx { m: 0, e: 0 }; o];
    for lr in 0..o {
        let am = i128::from(a[lr].m);
        let ae = i64::from(a[lr].e);
        let er = i64::from(e_row[lr]);
        let delta = er + 32 - ae;
        wk_c[..gs].copy_from_slice(&mant[lr * gs..(lr + 1) * gs]);
        s_c[..gs].fill(Fx { m: 0, e: 0 });
        e_c[..gs].fill(Fx { m: 0, e: 0 });
        let mut kal_prev: Option<i64> = None;
        let mut b_cur = 1usize;
        for m in 0..nb {
            let b0 = m * BLOCK;
            let b1 = b0 + BLOCK;
            let b_next = beam.min(b_cur * e8::CODEBOOK_LEN);
            // (1) The block's alignment exponent from the largest whitened term over ALL live
            // beams, so every beam of the block shares `K` and its scale.
            let mut kb = 0i32;
            for b in 0..b_cur {
                for t in 0..BLOCK {
                    let j = g0 + b0 + t;
                    let hjj = u.hat[j * n + j];
                    if hjj <= 0 {
                        return Err(GptqE8Error::NotPositiveDefinite);
                    }
                    let bh = blen(i128::from(hjj));
                    let bm = blen(i128::from(wk_c[b * gs + b0 + t]));
                    if bm > 0 {
                        kb = kb.max(bm + e_row[lr] + 32 + u.d[j] - bh + 1);
                    }
                    if am != 0 {
                        kb = kb.max(34 + blen(am) + a[lr].e + u.d[j] - bh + 1);
                    }
                }
            }
            let kal = i64::from(ALIGN - kb);
            // (2) `G_t = RNE(a·2^-32·S_t)`, the same for every beam of the row.
            for t in 0..BLOCK {
                let j = g0 + b0 + t;
                gvec[t] = div_shift_rne(
                    am,
                    kal + i64::from(u.d[j]) + ae,
                    i128::from(u.hat[j * n + j]),
                )?;
            }
            // (3) The carried `cum` rescaled to this block's `2^(2K)`.
            for b in 0..b_cur {
                base[b] = match kal_prev {
                    None => 0,
                    Some(kp) => {
                        let de = 2 * (kal - kp);
                        if de >= 0 {
                            let sh = i32::try_from(de).map_err(|_| GptqE8Error::Overflow)?;
                            shift_left_i128(cum_c[b], sh).ok_or(GptqE8Error::Overflow)?
                        } else {
                            let sh = u32::try_from(-de).map_err(|_| GptqE8Error::Overflow)?;
                            rshift_rne(cum_c[b], sh).ok_or(GptqE8Error::Overflow)?
                        }
                    }
                };
            }
            // (4) The exact 256-way scan per live beam.
            for b in 0..b_cur {
                for t in 0..BLOCK {
                    let j = g0 + b0 + t;
                    hvec[t] = div_shift_rne(
                        i128::from(wk_c[b * gs + b0 + t]),
                        er + kal + 32 + i64::from(u.d[j]),
                        i128::from(u.hat[j * n + j]),
                    )?;
                }
                let bb = base[b];
                for (k, row) in cw.iter().enumerate() {
                    let mut c: i128 = 0;
                    for t in 0..BLOCK {
                        let Some(e) = i128::from(row[t])
                            .checked_mul(gvec[t])
                            .and_then(|v| v.checked_sub(hvec[t]))
                        else {
                            return Err(GptqE8Error::Overflow);
                        };
                        let Some(e2) = e.checked_mul(e) else {
                            return Err(GptqE8Error::Overflow);
                        };
                        let Some(next) = c.checked_add(e2) else {
                            return Err(GptqE8Error::Overflow);
                        };
                        c = next;
                    }
                    cost[b * e8::CODEBOOK_LEN + k] =
                        bb.checked_add(c).ok_or(GptqE8Error::Overflow)?;
                }
            }
            // (5) Keep the `K` smallest pairs and rebuild the beams from their parents.
            let len = b_cur * e8::CODEBOOK_LEN;
            select_k(&cost[..len], b_next, &mut sel);
            for (bi, &(flat, val)) in sel.iter().enumerate() {
                let parent = flat / e8::CODEBOOK_LEN;
                let c = flat % e8::CODEBOOK_LEN;
                let src = parent * gs;
                let dst = bi * gs;
                wk_x[dst..dst + gs].copy_from_slice(&wk_c[src..src + gs]);
                s_x[dst..dst + gs].clone_from_slice(&s_c[src..src + gs]);
                e_x[dst..dst + gs].clone_from_slice(&e_c[src..src + gs]);
                let row = &cw[c];
                // The exact residual `E_t` from the parent's block, as in `quantise_pass`.
                for t in 0..BLOCK {
                    let j = g0 + b0 + t;
                    let dj = i64::from(u.d[j]);
                    let va = am
                        .checked_mul(i128::from(row[t]))
                        .ok_or(GptqE8Error::Overflow)?;
                    let (iv, phi) = if delta >= 0 {
                        let sh = u32::try_from(delta).map_err(|_| GptqE8Error::Overflow)?;
                        let vw = i128::from(wk_c[src + b0 + t])
                            .checked_shl(sh)
                            .ok_or(GptqE8Error::Overflow)?;
                        (
                            vw.checked_sub(va).ok_or(GptqE8Error::Overflow)?,
                            kal + ae + dj,
                        )
                    } else {
                        let sh = u32::try_from(-delta).map_err(|_| GptqE8Error::Overflow)?;
                        let sh = va.checked_shl(sh).ok_or(GptqE8Error::Overflow)?;
                        (
                            i128::from(wk_c[src + b0 + t])
                                .checked_sub(sh)
                                .ok_or(GptqE8Error::Overflow)?,
                            kal + er + 32 + dj,
                        )
                    };
                    let e = div_shift_rne(iv, phi, i128::from(u.hat[j * n + j]))?;
                    esel[t] = e;
                    s_x[dst + b0 + t] = Fx { m: row[t], e: -32 };
                    e_x[dst + b0 + t] =
                        Fx::from_i128(e, -i32::try_from(kal).map_err(|_| GptqE8Error::Overflow)?)
                            .map_err(|_| GptqE8Error::Overflow)?;
                }
                // The intra-group feedback, one RNE rounding per stored entry.
                let mut col = b1;
                while col < gs {
                    let mut acc: i128 = 0;
                    for t in 0..BLOCK {
                        let Some(pr) =
                            esel[t].checked_mul(i128::from(u.hat[(g0 + b0 + t) * n + g0 + col]))
                        else {
                            return Err(GptqE8Error::Overflow);
                        };
                        let Some(next) = acc.checked_add(pr) else {
                            return Err(GptqE8Error::Overflow);
                        };
                        acc = next;
                    }
                    let sh = -(kal + 32 + i64::from(u.d[g0 + col]) + er);
                    let neg = acc.checked_neg().ok_or(GptqE8Error::Overflow)?;
                    let dm = if sh >= 0 {
                        match u32::try_from(sh)
                            .ok()
                            .and_then(|k| 1_i128.checked_shl(k))
                            .and_then(|p| neg.checked_mul(p))
                        {
                            Some(v) => v,
                            None => return Err(GptqE8Error::Overflow),
                        }
                    } else {
                        match u32::try_from(-sh).ok().and_then(|k| rshift_rne(neg, k)) {
                            Some(v) => v,
                            None => return Err(GptqE8Error::Overflow),
                        }
                    };
                    let cur = i128::from(wk_c[src + col]);
                    wk_x[dst + col] = narrow_q(cur.checked_add(dm).ok_or(GptqE8Error::Overflow)?)?;
                    col += 1;
                }
                cum_x[bi] = val;
                for mm in 0..m {
                    idx_x[bi * nb + mm] = idx_c[parent * nb + mm];
                }
                idx_x[bi * nb + m] = u16::try_from(c).map_err(|_| GptqE8Error::Codebook)?;
            }
            core::mem::swap(&mut wk_c, &mut wk_x);
            core::mem::swap(&mut s_c, &mut s_x);
            core::mem::swap(&mut e_c, &mut e_x);
            core::mem::swap(&mut cum_c, &mut cum_x);
            core::mem::swap(&mut idx_c, &mut idx_x);
            kal_prev = Some(kal);
            b_cur = b_next;
        }
        if wscale {
            // The integer project row body: one `u_s` solve and one division per beam.
            for b in 0..b_cur {
                let src = b * gs;
                a_cand[b] = project_scale_row(
                    u,
                    n,
                    g0,
                    gs,
                    &s_c[src..src + gs],
                    &e_c[src..src + gs],
                    &a[lr],
                    &mut eproj[src..src + gs],
                )?;
            }
            // The smallest projected residual wins, on the exact `i128` norms at one shared
            // exponent over the row's beams.
            let mut rows: Vec<&[Fx]> = Vec::with_capacity(b_cur);
            for b in 0..b_cur {
                rows.push(&eproj[b * gs..(b + 1) * gs]);
            }
            let e0 = min_exp(&rows);
            for b in 0..b_cur {
                cerr[b] = sum_sq_at(&eproj[b * gs..(b + 1) * gs], e0)?;
            }
            let mut win = 0usize;
            for b in 1..b_cur {
                if cerr[b] < cerr[win] {
                    win = b;
                }
            }
            a_new[lr] = a_cand[win].clone();
            let src = win * gs;
            for j in 0..gs {
                s_out[lr * gs + j] = s_c[src + j].clone();
                e_out[lr * gs + j] = eproj[src + j].clone();
            }
            for mm in 0..nb {
                i_out[lr * nb + mm] = idx_c[win * nb + mm];
            }
        } else {
            a_new[lr] = a[lr].clone();
            s_out[lr * gs..(lr + 1) * gs].clone_from_slice(&s_c[..gs]);
            e_out[lr * gs..(lr + 1) * gs].clone_from_slice(&e_c[..gs]);
            for mm in 0..nb {
                i_out[lr * nb + mm] = idx_c[mm];
            }
        }
    }
    Ok(a_new)
}

/// One `wscale` candidate of a group: the per-row scale, codeword image, whitened residual and
/// codes. Referenced, not owned, so the two passes keep their buffers until the pick.
struct Cand<'a> {
    a: &'a [Fx],
    s: &'a [Fx],
    e: &'a [Fx],
    i: &'a [u16],
}

/// The `wscale` second-pass choice over the `o` rows: the candidate with the strictly smaller exact
/// `i128` whitened residual norm at the row's shared exponent ([`min_exp`], [`sum_sq_at`]), equal
/// norms keeping pass 1. `(a, s, e, i)` of the winner is returned in the candidate order.
///
/// # Errors
///
/// [`GptqE8Error::Overflow`] on a checked `i128` operation.
///
/// # Complexity
///
/// Temporal `O(o·gs)` checked `i128` multiply-adds (the two norms) plus the `O(o·(gs + gb))`
/// copies; spatial `O(o·(gs + gb))` for the winner's buffers.
#[allow(clippy::type_complexity, reason = "the caller's four group buffers")]
fn second_pass_pick(
    c1: &Cand<'_>,
    c2: &Cand<'_>,
    o: usize,
    gs: usize,
    gb: usize,
) -> Result<(Vec<Fx>, Vec<Fx>, Vec<Fx>, Vec<u16>), GptqE8Error> {
    let mut a = vec![Fx { m: 0, e: 0 }; o];
    let mut s = vec![Fx { m: 0, e: 0 }; o * gs];
    let mut e = vec![Fx { m: 0, e: 0 }; o * gs];
    let mut i = vec![0u16; o * gb];
    for r in 0..o {
        let (e1, e2) = (&c1.e[r * gs..(r + 1) * gs], &c2.e[r * gs..(r + 1) * gs]);
        let e0 = min_exp(&[e1, e2]);
        let n1 = sum_sq_at(e1, e0)?;
        let n2 = sum_sq_at(e2, e0)?;
        let (c, e_win) = if second_pass_wins(n1, n2) {
            (&c2, e2)
        } else {
            (&c1, e1)
        };
        a[r] = c.a[r].clone();
        s[r * gs..(r + 1) * gs].clone_from_slice(&c.s[r * gs..(r + 1) * gs]);
        i[r * gb..(r + 1) * gb].copy_from_slice(&c.i[r * gb..(r + 1) * gb]);
        e[r * gs..(r + 1) * gs].clone_from_slice(e_win);
    }
    Ok((a, s, e, i))
}

/// The GPTQ error feedback of `gptq_e8` (lines 267–397): per 8-block the block-Cholesky CVP
/// `argmin_k a²|Z_c[k]|² − 2a⟨Z_w, Z_c[k]⟩` over the 256-entry E8 ∪ O8 codebook, with each block's
/// whitened residual pushed onto the later columns of the group and the group's residuals pushed
/// onto the later groups through `U`.
///
/// `w` is row-major `[o, n]` Q32.32 words, `h` the symmetric `[n, n]` Q32.32 Hessian and `damp`
/// their Q32.32 damping coefficient. `act_order` visits the groups and 8-blocks in descending
/// Hessian-diagonal order and restores the caller's order in the output; `wscale` runs the whitened
/// scale projection and its second pass after the block pass (two refined candidates, the smaller
/// whitened residual row wins). The group scale is `a = 0.828·w_rms` in both settings (the Python
/// line 303), on the [`E8_SCALE`] word.
///
/// `beam = 1` is the single-path block pass; `beam in 2..=MAX_BEAM` runs the delayed-decision beam
/// search (`beam_pass`), which under `wscale` also runs its second pass as a beam search and keeps
/// the smaller-residual pass per row.
///
/// # Errors
/// [`GptqE8Error::BeamOutOfRange`] if `beam == 0` or `beam > MAX_BEAM`;
/// [`GptqE8Error::ZeroDim`] if `o == 0` or `n == 0`;
/// [`GptqE8Error::ShapeMismatch`] if `w.len() != o·n` or `h.len() != n·n`;
/// [`GptqE8Error::BlockMisaligned`] if `n % 8 != 0`; [`GptqE8Error::InvalidDamping`] if `damp <= 0`;
/// [`GptqE8Error::NotPositiveDefinite`] if the damped Hessian has a non-positive Cholesky pivot;
/// [`GptqE8Error::Codebook`] from the exact code index decode.
///
/// # Complexity
/// - Temporal: `O(n³ + o·n³/8²·256·8·beam + o·n²·beam)` checked `i128` multiply-adds — the factor
///   `U` (`n³/3`), the `256·8` scan per 8-block per row (×`beam` for the candidate expansion), and
///   the two feedback sweeps. Measure: multiply-adds. Derivation: `[n/8]` blocks × `256` rows ×
///   `8` coordinates × `beam` beams, plus `Σ_j (n − j)` per row per sweep. At `beam = 1` this is
///   the pre-beam cost.
/// - Spatial: `O(n² + o·n·beam)` words — `U`, the working copy of `w`, and the per-group beam
///   buffers.
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
    w: &[i64],
    h: &[i64],
    o: usize,
    n: usize,
    damp: i64,
    act_order: bool,
    wscale: bool,
    beam: usize,
) -> Result<GptqE8, GptqE8Error> {
    if beam == 0 || beam > MAX_BEAM {
        return Err(GptqE8Error::BeamOutOfRange);
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
    if damp <= 0 {
        return Err(GptqE8Error::InvalidDamping);
    }
    let nb = n / BLOCK;
    let per_group = GROUP / BLOCK;
    let ng = n.div_ceil(GROUP);
    // The weight crosses into its per-row integer carrier once and never leaves it.
    let (mant, e_row) = w_to_rows(w, o, n)?;
    // The act order permutes whole 8-blocks; `perm_col[c]` is the original column at permuted
    // position `c`, `inv_blk[b]` the permuted position of original block `b`.
    let perm = if act_order {
        let (blk_perm, inv_blk) = block_order(h, n)?;
        let mut perm_col = vec![0usize; n];
        for c in 0..n {
            perm_col[c] = blk_perm[c / BLOCK] * BLOCK + c % BLOCK;
        }
        Some((perm_col, inv_blk))
    } else {
        None
    };
    // The permuted working rows are an exact copy of the integer mantissas (`act_order` copies
    // columns and performs no arithmetic); only the Hessian's column order is rebuilt.
    let (mut wk, mut er, hp) = match &perm {
        Some((perm_col, _)) => {
            let mut wk = vec![0i64; o * n];
            for r in 0..o {
                for c in 0..n {
                    wk[r * n + c] = mant[r * n + perm_col[c]];
                }
            }
            let mut hp = vec![0i64; n * n];
            for i in 0..n {
                let ri = perm_col[i];
                for c in 0..n {
                    hp[i * n + c] = h[ri * n + perm_col[c]];
                }
            }
            (wk, e_row, hp)
        }
        None => (mant, e_row, h.to_vec()),
    };
    let ufac = feedback_u(&hp, n, damp)?;
    let mut s_p = vec![Fx { m: 0, e: 0 }; o * n];
    let mut idx_p = vec![0u16; o * nb];
    let mut scale_fx = vec![Fx { m: 0, e: 0 }; o * ng];

    for gi in 0..ng {
        let g0 = gi * GROUP;
        let g1 = (g0 + GROUP).min(n);
        let gs = g1 - g0;
        let gb = gs / BLOCK;
        // The group's integer row slice at the row's exponent (an exact copy, no re-normalisation)
        // and its scale `a = 0.828·w_rms` with inverse-variance weights `prec_j = 1/U_jj²` (the
        // Python lines 301–303), integer through [`group_scale_int`]; `wscale` does not change the
        // scale rule here — it only adds the projection passes below.
        let mut gm = vec![0i64; o * gs];
        let mut a_fx = vec![Fx { m: 0, e: 0 }; o];
        for r in 0..o {
            gm[r * gs..(r + 1) * gs].copy_from_slice(&wk[r * n + g0..r * n + g0 + gs]);
            a_fx[r] = group_scale_int(&gm[r * gs..(r + 1) * gs], er[r], &ufac, g0)?;
        }
        let (a_best, s_best, e_best, i_best) = if beam == 1 {
            let mut s_best = vec![Fx { m: 0, e: 0 }; o * gs];
            let mut e_best = vec![Fx { m: 0, e: 0 }; o * gs];
            let mut i_best = vec![0u16; o * gb];
            quantise_pass(
                &ufac,
                n,
                g0,
                g1,
                &gm,
                &er,
                &a_fx,
                o,
                &mut s_best,
                &mut e_best,
                &mut i_best,
            )?;
            if wscale {
                let (a1, e1) = project_scale_int(&ufac, n, g0, gs, &s_best, &e_best, &a_fx)?;
                let mut s2 = vec![Fx { m: 0, e: 0 }; o * gs];
                let mut e2 = vec![Fx { m: 0, e: 0 }; o * gs];
                let mut i2 = vec![0u16; o * gb];
                quantise_pass(
                    &ufac, n, g0, g1, &gm, &er, &a1, o, &mut s2, &mut e2, &mut i2,
                )?;
                let (a2, e2b) = project_scale_int(&ufac, n, g0, gs, &s2, &e2, &a1)?;
                // Second pass: keep the candidate with the smaller whitened residual per row, on
                // the exact `i128` norms at a shared exponent; equal norms keep pass 1.
                second_pass_pick(
                    &Cand {
                        a: &a1,
                        s: &s_best,
                        e: &e1,
                        i: &i_best,
                    },
                    &Cand {
                        a: &a2,
                        s: &s2,
                        e: &e2b,
                        i: &i2,
                    },
                    o,
                    gs,
                    gb,
                )?
            } else {
                (a_fx, s_best, e_best, i_best)
            }
        } else {
            let mut s1 = vec![Fx { m: 0, e: 0 }; o * gs];
            let mut e1 = vec![Fx { m: 0, e: 0 }; o * gs];
            let mut i1 = vec![0u16; o * gb];
            let a1 = beam_pass(
                &ufac, n, g0, g1, &gm, &er, &a_fx, o, beam, wscale, &mut s1, &mut e1, &mut i1,
            )?;
            if wscale {
                // The second pass is a beam search again, at the projected scale (lines 304,
                // 379–389); the projected scale stays on the carrier as an [`Fx`].
                let mut s2 = vec![Fx { m: 0, e: 0 }; o * gs];
                let mut e2 = vec![Fx { m: 0, e: 0 }; o * gs];
                let mut i2 = vec![0u16; o * gb];
                let a2 = beam_pass(
                    &ufac, n, g0, g1, &gm, &er, &a1, o, beam, true, &mut s2, &mut e2, &mut i2,
                )?;
                second_pass_pick(
                    &Cand {
                        a: &a1,
                        s: &s1,
                        e: &e1,
                        i: &i1,
                    },
                    &Cand {
                        a: &a2,
                        s: &s2,
                        e: &e2,
                        i: &i2,
                    },
                    o,
                    gs,
                    gb,
                )?
            } else {
                (a1, s1, e1, i1)
            }
        };
        for r in 0..o {
            scale_fx[r * ng + gi] = a_best[r].clone();
            s_p[r * n + g0..r * n + g1].clone_from_slice(&s_best[r * gs..(r + 1) * gs]);
            idx_p[r * nb + gi * per_group..r * nb + gi * per_group + gb]
                .copy_from_slice(&i_best[r * gb..(r + 1) * gb]);
        }
        // Inter-group feedback: `wk[:, g1:] -= Err_best @ U[g0:g1, g1:]` (line 394), on the integer
        // rows with one round-to-nearest per stored entry.
        if g1 < n {
            inter_group_feedback(&mut wk, &mut er, &e_best, &ufac, n, g0, gs)?;
        }
    }

    let (s_fx, idx, scale_fx) = match &perm {
        None => (s_p, idx_p, scale_fx),
        Some((_, inv_blk)) => {
            let mut s = vec![Fx { m: 0, e: 0 }; o * n];
            let mut idx = vec![0u16; o * nb];
            let mut scale = vec![Fx { m: 0, e: 0 }; o * ng];
            for r in 0..o {
                for c in 0..n {
                    let p = inv_blk[c / BLOCK] * BLOCK + c % BLOCK;
                    s[r * n + c] = s_p[r * n + p].clone();
                }
                for b in 0..nb {
                    idx[r * nb + b] = idx_p[r * nb + inv_blk[b]];
                }
                for g in 0..ng {
                    scale[r * ng + g] =
                        scale_fx[r * ng + inv_blk[g * per_group] / per_group].clone();
                }
            }
            (s, idx, scale)
        }
    };
    // The boundary's output words: `s` is the codeword word itself (`e = -32`) and `scale` is one
    // round-to-nearest of the group scale's `Fx` to the Q32.32 grid.
    let s = s_fx.iter().map(fx_to_word).collect::<Result<Vec<_>, _>>()?;
    let scale = scale_fx
        .iter()
        .map(fx_to_word)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GptqE8 { s, idx, scale })
}
