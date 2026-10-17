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
//! # Integer carrier (lanes `q-l1a`–`q-l1c`)
//!
//! The feedback factor, the act order, the group scale and both block passes are carried in
//! integers: `feedback_u` returns a [`UFac`] (`U = Uhat·diag(2^-d)`), [`block_order`] orders on
//! `i128` diagonal sums, the group scale is an [`Fx`], and [`quantise_pass`] and [`beam_pass`]
//! evaluate the block objective as an exact `i128` scan, carrying the residual and codeword image as
//! [`Fx`]. The only `f64` inputs that cross into integers are the damped Hessian ([`h_to_spdq`],
//! which L6 deletes when the caller hands an [`SpdQ`]) and the weight ([`w_to_rows`]); the `2^-g`
//! global exponent `g` is even, so it is an exact scaling. What remains `f64` is the scale projection
//! ([`project_scale`]), the group feedback ([`inter_group_feedback`]) and the beam pass's `wscale`
//! tail, which L1d replaces; they receive their views through [`ufac_f64`], [`rows_to_f64`],
//! [`fx_slice_to_f64`] and [`f64_slice_to_fx`], marked `L1b-L1d delete this`.

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
    block_order, div_rne_i128, f64_int_exp, fx_to_f64, group_scale_int, mag_u128, narrow_q, pow2i,
    rows_to_f64, ufac_f64, w_to_rows,
};
pub use super::gptq_e8_carrier::{feedback_u, UFac};
use crate::engine::ONE;
use crate::learn::code::e8::{self, E8Error};
use crate::learn::solve::q32::{ilog2_pos, rshift_rne, Fx};

/// Columns per group: the Python `GROUP`. One scale per (row, group).
pub const GROUP: usize = 128;

/// Columns per code index: one codebook row per 8 weights.
pub const BLOCK: usize = 8;

/// The codebook scale factor of `gptq_e8` (line 303): `a = 0.828 · w_rms`.
pub const E8_SCALE: f64 = 0.828;

/// The [`E8_SCALE`] factor as a Q32.32 word: `round(0.828 · 2^32)`. Derived from
/// `0.828 · 4_294_967_296 = 3_556_232_921.088`, so the rounded literal is `3_556_232_921`; the
/// product `E8_SCALE_Q32 · 2^-32` differs from `0.828` by `0.088·2^-32 ≈ 2.0e-11`. The integer
/// group scale multiplies by this word, so the factor never re-enters `f64`.
pub const E8_SCALE_Q32: i64 = 3_556_232_921;

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

/// Solve `x·U_sub = e` in place for the row `e` (length `m`), `U_sub` the trailing upper-triangular
/// block of the row-major `[stride, stride]` upper-triangular `u`: `x_j = (e_j − Σ_{k>j}
/// x_k·U_kj)/U_jj`. Temporal `O(m²)` f64 multiply-subtracts, spatial `O(1)`.
fn solve_upper_block(u: &[f64], stride: usize, b0: usize, m: usize, e: &mut [f64]) {
    let mut j = m;
    while j > 0 {
        j -= 1;
        let mut s = e[j];
        let mut k = j + 1;
        while k < m {
            s -= e[k] * u[(b0 + k) * stride + b0 + j];
            k += 1;
        }
        e[j] = s / u[(b0 + j) * stride + b0 + j];
    }
}

/// The worker count handed to [`inv_factor_upper`] by [`feedback_u`]. `feedback_u` takes no thread
/// parameter and its factor is bit-identical for every count, so the count changes only wall-clock;
/// `8` matches the other solve callers' fixed count and leaves headroom on the shared box.
pub(super) const INV_THREADS: usize = 8;

/// The magnitude class `⌈log2(|v| + 1)⌉` of an `i128`: `0` for `0`, else `ilog2_pos(|v|) + 1`, so
/// `2^(b − 1) ≤ |v| < 2^b`.
///
/// # Complexity
///
/// `O(1)`.
fn blen(v: i128) -> i32 {
    if v == 0 {
        0
    } else {
        match ilog2_pos(mag_u128(v)) {
            Some(b) => b.cast_signed() + 1,
            None => 0,
        }
    }
}

/// `RNE(v · 2^s / d)` for `d > 0`: an exact checked left shift of the numerator when `s ≥ 0`, the
/// exact denominator `d · 2^-s` when `s < 0`, and one round-to-nearest quotient ([`div_rne_i128`]).
/// `0` for `v == 0`.
///
/// # Errors
///
/// [`GptqE8Error::Overflow`] on a shift or quotient that leaves `i128`.
///
/// # Complexity
///
/// `O(1)`.
fn div_shift_rne(v: i128, s: i64, d: i128) -> Result<i128, GptqE8Error> {
    if v == 0 {
        return Ok(0);
    }
    let q = if s >= 0 {
        match u32::try_from(s).ok().and_then(|k| 1_i128.checked_shl(k)) {
            Some(p) => match v.checked_mul(p) {
                Some(num) => div_rne_i128(num, d),
                None => None,
            },
            None => None,
        }
    } else {
        match u32::try_from(-s).ok().and_then(|k| d.checked_shl(k)) {
            Some(den) => div_rne_i128(v, den),
            None => None,
        }
    };
    q.ok_or(GptqE8Error::Overflow)
}

/// The `f64` view (`m · 2^e`) of a carrier slice: `// L1b-L1d delete this`, the transient boundary
/// the still-`f64` passes read (a conversion, not a second path).
fn fx_slice_to_f64(x: &[Fx]) -> Vec<f64> {
    x.iter().map(fx_to_f64).collect()
}

/// The [`Fx`] of a finite `f64`, exact (`|significand| ≤ 2^53`): `// L1b-L1d delete this`, the
/// transient boundary the projected scale returns through; [`GptqE8Error::NonFinite`] on a NaN or an
/// infinite, [`GptqE8Error::Overflow`] on a carrier outside `i64`.
fn f64_slice_to_fx(x: &[f64]) -> Result<Vec<Fx>, GptqE8Error> {
    let mut out = Vec::with_capacity(x.len());
    for &v in x {
        if !v.is_finite() {
            return Err(GptqE8Error::NonFinite);
        }
        let (m, e) = f64_int_exp(v);
        out.push(Fx::from_i128(m, e).map_err(|_| GptqE8Error::Overflow)?);
    }
    Ok(out)
}

/// One quantisation pass over the group `[g0, g1)` of the permuted Hessian factor `u` and the
/// group's per-row integer weight rows `mant`/`e_row` (row-major `o × (g1 − g0)`, the real weight
/// is `mant[r·gs + c] · 2^e_row[r]`), at the per-row group scale `a` (the [`group_scale_int`]
/// outputs).
///
/// # The block right-solve
///
/// The stored factor is `U_ij = u.hat[i·n + j] · 2^-32 · 2^-d_j`, row-major `[n, n]` and upper
/// triangular (`u.hat[i·n + j] = 0` for `j < i`). The pre-lane `solve_upper_block` contraction
/// `Σ_{k>j} e[k] · u[(b0 + k)·n + (b0 + j)]` reads the strictly below-diagonal entries of that
/// block — all zero — so the block right-solve it performs is the per-coordinate quotient
/// `Z[t] = w_sub[t] / U_tt = w_sub[t] · 2^(32 + d_j) / u.hat[j·n + j]`, `j = g0 + b0 + t`. This pass
/// carries that quotient; a full 8×8 triangular inverse is **not** what the pass computes and would
/// move every code (measured on the f64 arm at this revision: the quotient reproduces the arm's
/// argmin 16/16 on the `n = 8` fixture, the full inverse 14/16).
///
/// # The exact cost
///
/// With the block's alignment exponent `K` and `S_t = 2^(K + 32 + d_j) / u.hat[j·n + j]`, the pair
/// `G_t = RNE(a · 2^-32 · S_t)` and `H_t = RNE(w_sub[t] · S_t)` makes the Python's block objective
/// (line 318)
/// `a²|Z_c[k]|² − 2a⟨Z_w, Z_c[k]⟩ = Σ_t ((cw[k][t]·G_t − H_t)·2^-K)² − |Z_w|²`, so its argmin is the
/// argmin of the exact `i128` sum `Σ_t (cw[k][t]·G_t − H_t)²` with no rounding inside the 256-way
/// scan and ties to the smaller index. `K` is chosen per (row, block) as `58 − kb`, where `kb` is
/// the magnitude class of the largest whitened term of the block, which holds
/// `|cw[k][t]·G_t| < 2^58` and `|H_t| < 2^58`; hence `Σ_t (…)² < 8·2^118` and every accumulation
/// below stays inside `i128` (all of them checked, [`GptqE8Error::Overflow`] otherwise). The
/// residual is **not** `cw[idx][t]·G_t − H_t` (that carries `G_t`'s `2^-26` rounding); it is
/// recomputed exactly from the numerator form,
/// `E[t] = 2^K·err_sub[t] = 2^K·(Z_w[t] − a·Z_c[idx][t])`, so `err_sub[t] = E[t]·2^-K` and the
/// codeword image `s[t] = cw[idx][t]·2^-32` are written on the integer carrier (`Fx`) at full
/// precision.
///
/// # The intra-group feedback
///
/// `wk[:, b1:] -= err_sub @ U_blk[b0:b1, b1:]` (line 323): per later column `c` the contraction
/// `Σ_t E[t] · u.hat[(b0 + t)·n + (g0 + c)]` is accumulated in `i128` and applied to the working
/// mantissa with a **single** RNE rounding to the row's grid `2^e_row`. One rounding per stored
/// entry bounds the update's absolute error by `2^(e_row − 1)`, i.e. `2^-35` of the row's maximum
/// magnitude; the per-entry errors of one group (at most `gs/BLOCK` of them per column) move the
/// objective by `≈ 2^-31` relative, far inside the `2^-20` the `J` gate allows.
///
/// # Threading
///
/// Rows are independent given the read-only `u`, `a` and codebook, so the rows are split into
/// `threads` contiguous ranges and each range runs the whole block loop on its own scoped worker:
/// one spawn per call. The per-row operation order is unchanged and no quantity is reduced across
/// rows, so the output — and any refusal — is identical for every `threads`, including `1`.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] on a buffer of the wrong length, [`GptqE8Error::Codebook`] from
/// the exact code index decode, [`GptqE8Error::NotPositiveDefinite`] on a non-positive feedback
/// diagonal and [`GptqE8Error::Overflow`] on a checked `i128` operation. A refusal from a worker is
/// returned from the lowest row range that produced one.
///
/// # Complexity
///
/// - Temporal: `O(o·[gs/8]·256·8 + o·Σ_blocks (gs − b1))` checked `i128` multiply-adds — the
///   `256·8` scan of the block objective and the intra-group propagation `Σ_blocks 8·(gs − b1)`,
///   plus the `O([gs/8]·8)` per-row alignment and pair computation. Measure: `i128` multiply-adds.
///   The row term is independent per row, so the wall-clock span is that term `/ threads`.
/// - Spatial: `O(o·gs)` working `i64` mantissas (one copy per worker of its own rows) plus the
///   `256·8` `i64` codebook; the outputs are caller buffers.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the pass writes three separate output buffers plus the working mantissas and reads \
              the factor, the rows and the scale; grouping them into a struct would not reduce the \
              caller's argument count, and the body is kept one block loop parallel to the Python's \
              lines 312–323"
)]
fn quantise_pass(
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
    threads: usize,
) -> Result<(), GptqE8Error> {
    // The alignment ceiling: `58` keeps `|cw_t·G_t|` and `|H_t|` below `2^58`, so the block's
    // sum of squares stays below `8·2^118` and every `i128` accumulation below is in range.
    const ALIGN: i32 = 58;
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
    // The 256 codebook rows once per call, read-only inside the workers.
    let mut cw = [[0i64; BLOCK]; e8::CODEBOOK_LEN];
    for (k, row) in cw.iter_mut().enumerate() {
        *row = e8::codeword(k).map_err(map_e8)?;
    }
    let cw = &cw;
    let per = o.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|scope| -> Result<(), GptqE8Error> {
        let mut handles = Vec::new();
        for (((((mc, ec), ac), sc), eoc), ic) in mant
            .chunks(per * gs)
            .zip(e_row.chunks(per))
            .zip(a.chunks(per))
            .zip(s_out.chunks_mut(per * gs))
            .zip(e_out.chunks_mut(per * gs))
            .zip(i_out.chunks_mut(per * nb))
        {
            handles.push(scope.spawn(move || -> Result<(), GptqE8Error> {
                let rows = ec.len();
                // The group's working mantissas for this worker's rows: the pristine rows, which
                // only this worker's intra-group feedback below updates.
                let mut wk = mc.to_vec();
                let mut gvec = [0i128; BLOCK];
                let mut hvec = [0i128; BLOCK];
                for lr in 0..rows {
                    let er = i64::from(ec[lr]);
                    let am = i128::from(ac[lr].m);
                    let ae = i64::from(ac[lr].e);
                    // `Δ = er + 32 − ae`, the relative scale of the row grid and the `a·2^32` image;
                    // `≈ 62..66` for every fixture, and the sign picks which term of `I_t` shifts.
                    let delta = er + 32 - ae;
                    for m in 0..nb {
                        let b0 = m * BLOCK;
                        // (1) The block's alignment exponent `K = ALIGN − kb`, `kb` the magnitude
                        // class of the largest whitened term of the block. `kb` stays `0` when the
                        // row and the scale are both zero, which makes every cost `0` and leaves
                        // the decision to the tie rule.
                        let mut kb = 0i32;
                        for t in 0..BLOCK {
                            let j = g0 + b0 + t;
                            let hjj = u.hat[j * n + j];
                            if hjj <= 0 {
                                return Err(GptqE8Error::NotPositiveDefinite);
                            }
                            let bh = blen(i128::from(hjj));
                            let bm = blen(i128::from(wk[lr * gs + b0 + t]));
                            if bm > 0 {
                                kb = kb.max(bm + ec[lr] + 32 + u.d[j] - bh + 1);
                            }
                            if am != 0 {
                                // The `34` is the magnitude class of the largest codebook word
                                // (`sqrt8·2^32`), which multiplies `G_t`.
                                kb = kb.max(34 + blen(am) + ac[lr].e + u.d[j] - bh + 1);
                            }
                        }
                        let kal = i64::from(ALIGN - kb);
                        // (2) `G_t = RNE(a·2^-32·S_t)`, `H_t = RNE(w_t·S_t)`.
                        for t in 0..BLOCK {
                            let j = g0 + b0 + t;
                            let hjj = i128::from(u.hat[j * n + j]);
                            let dj = i64::from(u.d[j]);
                            gvec[t] = div_shift_rne(am, kal + dj + ae, hjj)?;
                            hvec[t] = div_shift_rne(
                                i128::from(wk[lr * gs + b0 + t]),
                                er + kal + 32 + dj,
                                hjj,
                            )?;
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
                        // (4) The exact residual `E_t = 2^K·e[t]` from the numerator form — the
                        // rounded `G_t`/`H_t` above decide the argmin but must **not** leak their
                        // `2^-26` rounding into the residual the projection and the feedback read.
                        // `E_t = [wk·2^(er+32+dj) − a.m·row[t]·2^(ae+dj)]/hat_jj`; factor the common
                        // `2^(min(er+32,ae)+dj+K)`, so `I_t = wk·2^Δ − a.m·row[t]` (`Δ = er+32−ae`)
                        // when `Δ ≥ 0`, else `wk − a.m·row[t]·2^(−Δ)`, and `E_t = RNE(I_t·2^φ/hat_jj)`.
                        let row = &cw[best];
                        let mut esel = [0i128; BLOCK];
                        for t in 0..BLOCK {
                            let j = g0 + b0 + t;
                            let dj = i64::from(u.d[j]);
                            let va = am
                                .checked_mul(i128::from(row[t]))
                                .ok_or(GptqE8Error::Overflow)?;
                            let (iv, phi) = if delta >= 0 {
                                let sh = u32::try_from(delta).map_err(|_| GptqE8Error::Overflow)?;
                                let vw = i128::from(wk[lr * gs + b0 + t])
                                    .checked_shl(sh)
                                    .ok_or(GptqE8Error::Overflow)?;
                                (
                                    vw.checked_sub(va).ok_or(GptqE8Error::Overflow)?,
                                    kal + ae + dj,
                                )
                            } else {
                                let sh =
                                    u32::try_from(-delta).map_err(|_| GptqE8Error::Overflow)?;
                                let sh = va.checked_shl(sh).ok_or(GptqE8Error::Overflow)?;
                                (
                                    i128::from(wk[lr * gs + b0 + t])
                                        .checked_sub(sh)
                                        .ok_or(GptqE8Error::Overflow)?,
                                    kal + er + 32 + dj,
                                )
                            };
                            let e = div_shift_rne(iv, phi, i128::from(u.hat[j * n + j]))?;
                            esel[t] = e;
                            sc[lr * gs + b0 + t] = Fx { m: row[t], e: -32 };
                            // `E_t` is already `2^K·e[t] = 2^K·(Z_w − a·Z_c)` in the positive
                            // convention the Python's `err_sub` uses, so it is stored unnegated.
                            eoc[lr * gs + b0 + t] = Fx::from_i128(
                                e,
                                -i32::try_from(kal).map_err(|_| GptqE8Error::Overflow)?,
                            )
                            .map_err(|_| GptqE8Error::Overflow)?;
                        }
                        ic[lr * nb + m] = u16::try_from(best).map_err(|_| GptqE8Error::Codebook)?;
                        let mut c = b0 + BLOCK;
                        while c < gs {
                            let mut acc: i128 = 0;
                            for t in 0..BLOCK {
                                let Some(pr) = esel[t]
                                    .checked_mul(i128::from(u.hat[(g0 + b0 + t) * n + g0 + c]))
                                else {
                                    return Err(GptqE8Error::Overflow);
                                };
                                let Some(next) = acc.checked_add(pr) else {
                                    return Err(GptqE8Error::Overflow);
                                };
                                acc = next;
                            }
                            // `Δw = −acc · 2^(−K − 32 − d_c)`, one RNE rounding to the row's grid
                            // `2^e_row`.
                            let sh = -(kal + 32 + i64::from(u.d[g0 + c]) + er);
                            let Some(neg) = acc.checked_neg() else {
                                return Err(GptqE8Error::Overflow);
                            };
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
                            let cur = i128::from(wk[lr * gs + c]);
                            let Some(next) = cur.checked_add(dm) else {
                                return Err(GptqE8Error::Overflow);
                            };
                            wk[lr * gs + c] = narrow_q(next)?;
                            c += 1;
                        }
                    }
                }
                Ok(())
            }));
        }
        for h in handles {
            match h.join() {
                Ok(r) => r?,
                Err(p) => std::panic::resume_unwind(p),
            }
        }
        Ok(())
    })
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

/// The `k` smallest `(cost, flat index)` pairs of `cost` by ascending cost, ties to the smaller index:
/// the `torch.topk(..., largest=False)` order of the Python line 352 (`O(len·k)` `i128`
/// comparisons on a sorted `k`-window, `O(k)` space; `len = B·256`, `k ≤ MAX_BEAM`).
fn select_k(cost: &[i128], k: usize, out: &mut Vec<(usize, i128)>) {
    out.clear();
    for (i, &v) in cost.iter().enumerate() {
        let pos = out.partition_point(|&(oi, ov)| ov < v || (ov == v && oi < i));
        if pos < k {
            out.insert(pos, (i, v));
            out.truncate(k);
        }
    }
}

/// `v·2^s` for `s ≥ 0`, `Some(0)` for zero, `None` when the product leaves `i128` (`O(1)`).
fn shift_left_checked(v: i128, s: u32) -> Option<i128> {
    if v == 0 {
        return Some(0);
    }
    if s >= 127 {
        return None;
    }
    v.checked_mul(1_i128 << s)
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
/// `a_cand = max(a + ⟨E, u_S⟩/max(|u_S|², 1e-12), 1e-12)`, the smallest projected residual wins
/// (lines 367–376). It reads `s`/`e` through [`fx_slice_to_f64`] and writes the residual back
/// through [`f64_slice_to_fx`]: the pass's only `f64` part
/// (`// L1d: replaced by the integer project helper`); everything before it is integer.
///
/// Rows are independent given the read-only `u`, the per-row `a[r]` and the codebook, and `B`
/// depends on the block index alone, so the rows are split into `threads` contiguous ranges, each
/// running the whole block loop on its own scoped worker. The per-row operation order is unchanged
/// and nothing is reduced across rows, so the output — and any refusal — is identical for every
/// `threads`, including `1`.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] on a wrong-length buffer, [`GptqE8Error::Codebook`] from the code
/// index decode, [`GptqE8Error::NotPositiveDefinite`] on a non-positive feedback diagonal and
/// [`GptqE8Error::Overflow`] on a checked `i128` operation; a worker refusal names its lowest row.
///
/// # Complexity
///
/// Temporal `O(o·[gs/8]·beam·256·8 + o·Σ_blocks beam·(gs − b1) + o·beam·gs²)` `i128` multiply-adds
/// (256-codeword expansion, retained-beam intra-group feedback, `wscale` projection; row term split
/// `/ threads`); spatial `O(ceil(o/threads)·beam·gs)` beam state, a `beam·256` cost buffer per
/// worker, the shared `256·8` `i64` codebook and caller-owned outputs.
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
fn beam_pass(
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
    threads: usize,
) -> Result<Vec<f64>, GptqE8Error> {
    // The alignment ceiling: `58` keeps `|cw_t·G_t|` and `|H_t|` below `2^58` for every live beam,
    // so the block's sum of squares stays below `2^121` and the carried `cum` below `2^126`.
    const ALIGN: i32 = 58;
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
    let cw = &cw;
    // `// L1d: replaced by the integer project helper` — the group block of the feedback factor as
    // an `f64` view for the `wscale` tail only: `gs²` words, not the full `[n, n]`.
    let mut ublk = Vec::new();
    if wscale {
        ublk = vec![0.0f64; gs * gs];
        for i in 0..gs {
            for j in i..gs {
                ublk[i * gs + j] =
                    u.hat[(g0 + i) * n + g0 + j] as f64 / ONE as f64 * pow2i(-u.d[g0 + j]);
            }
        }
    }
    let ublk = &ublk;
    let per = o.div_ceil(threads.max(1)).max(1);
    let mut a_new = Vec::with_capacity(o);
    std::thread::scope(|scope| -> Result<(), GptqE8Error> {
        let mut handles = Vec::new();
        for (((((mc, ec), ac), sc), eoc), ic) in mant
            .chunks(per * gs)
            .zip(e_row.chunks(per))
            .zip(a.chunks(per))
            .zip(s_out.chunks_mut(per * gs))
            .zip(e_out.chunks_mut(per * gs))
            .zip(i_out.chunks_mut(per * nb))
        {
            handles.push(scope.spawn(move || -> Result<Vec<f64>, GptqE8Error> {
                let rows = ec.len();
                // Two beam-state buffer sets for the worker's rows, swapped per block.
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
                let mut us = vec![0.0f64; gs];
                let mut eproj = vec![0.0f64; stride];
                let mut a_cand = vec![0.0f64; cap];
                let mut cerr = vec![0.0f64; cap];
                let mut a_row = vec![0.0f64; rows];
                let mut gvec = [0i128; BLOCK];
                let mut hvec = [0i128; BLOCK];
                let mut esel = [0i128; BLOCK];
                for lr in 0..rows {
                    let ar = fx_to_f64(&ac[lr]);
                    let am = i128::from(ac[lr].m);
                    let ae = i64::from(ac[lr].e);
                    let er = i64::from(ec[lr]);
                    let delta = er + 32 - ae;
                    wk_c[..gs].copy_from_slice(&mc[lr * gs..(lr + 1) * gs]);
                    s_c[..gs].fill(Fx { m: 0, e: 0 });
                    e_c[..gs].fill(Fx { m: 0, e: 0 });
                    let mut kal_prev: Option<i64> = None;
                    let mut b_cur = 1usize;
                    for m in 0..nb {
                        let b0 = m * BLOCK;
                        let b1 = b0 + BLOCK;
                        let b_next = beam.min(b_cur * e8::CODEBOOK_LEN);
                        // (1) The block's alignment exponent from the largest whitened term over
                        // ALL live beams, so every beam of the block shares `K` and its scale.
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
                                    kb = kb.max(bm + ec[lr] + 32 + u.d[j] - bh + 1);
                                }
                                if am != 0 {
                                    kb = kb.max(34 + blen(am) + ac[lr].e + u.d[j] - bh + 1);
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
                        // (3) The carried `cum` rescaled to this block's `2^(2K)`: an exact checked
                        // left shift for an increasing `K`, one RNE shift for a decreasing one.
                        for b in 0..b_cur {
                            base[b] = match kal_prev {
                                None => 0,
                                Some(kp) => {
                                    let de = 2 * (kal - kp);
                                    if de >= 0 {
                                        let sh =
                                            u32::try_from(de).map_err(|_| GptqE8Error::Overflow)?;
                                        shift_left_checked(cum_c[b], sh)
                                            .ok_or(GptqE8Error::Overflow)?
                                    } else {
                                        let sh = u32::try_from(-de)
                                            .map_err(|_| GptqE8Error::Overflow)?;
                                        rshift_rne(cum_c[b], sh).ok_or(GptqE8Error::Overflow)?
                                    }
                                }
                            };
                        }
                        // (4) The exact 256-way scan per live beam: `cum + |Z_w − a·Z_c[k]|²` at
                        // the shared scale `2^(2K)`, ties by the selection order.
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
                            // The exact residual `E_t` from the parent's block, as in
                            // `quantise_pass`: the rounded `G_t`/`H_t` decide the argmin but do not
                            // enter the residual the projection and the feedback read.
                            for t in 0..BLOCK {
                                let j = g0 + b0 + t;
                                let dj = i64::from(u.d[j]);
                                let va = am
                                    .checked_mul(i128::from(row[t]))
                                    .ok_or(GptqE8Error::Overflow)?;
                                let (iv, phi) = if delta >= 0 {
                                    let sh =
                                        u32::try_from(delta).map_err(|_| GptqE8Error::Overflow)?;
                                    let vw = i128::from(wk_c[src + b0 + t])
                                        .checked_shl(sh)
                                        .ok_or(GptqE8Error::Overflow)?;
                                    (
                                        vw.checked_sub(va).ok_or(GptqE8Error::Overflow)?,
                                        kal + ae + dj,
                                    )
                                } else {
                                    let sh =
                                        u32::try_from(-delta).map_err(|_| GptqE8Error::Overflow)?;
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
                                e_x[dst + b0 + t] = Fx::from_i128(
                                    e,
                                    -i32::try_from(kal).map_err(|_| GptqE8Error::Overflow)?,
                                )
                                .map_err(|_| GptqE8Error::Overflow)?;
                            }
                            // The intra-group feedback, one RNE rounding per stored entry.
                            let mut col = b1;
                            while col < gs {
                                let mut acc: i128 = 0;
                                for t in 0..BLOCK {
                                    let Some(pr) = esel[t].checked_mul(i128::from(
                                        u.hat[(g0 + b0 + t) * n + g0 + col],
                                    )) else {
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
                                wk_x[dst + col] =
                                    narrow_q(cur.checked_add(dm).ok_or(GptqE8Error::Overflow)?)?;
                                col += 1;
                            }
                            cum_x[bi] = val;
                            for mm in 0..m {
                                idx_x[bi * nb + mm] = idx_c[parent * nb + mm];
                            }
                            idx_x[bi * nb + m] =
                                u16::try_from(c).map_err(|_| GptqE8Error::Codebook)?;
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
                        // `// L1d: replaced by the integer project helper` — the projected scale of
                        // the final beams of this row, through the transient `f64` views.
                        for b in 0..b_cur {
                            let src = b * gs;
                            us.copy_from_slice(&fx_slice_to_f64(&s_c[src..src + gs]));
                            solve_upper_block(ublk, gs, 0, gs, &mut us);
                            let ev = fx_slice_to_f64(&e_c[src..src + gs]);
                            let mut num = 0.0f64;
                            let mut den = 0.0f64;
                            for j in 0..gs {
                                num += ev[j] * us[j];
                                den += us[j] * us[j];
                            }
                            let ac2 = (ar + num / den.max(1e-12)).max(1e-12);
                            a_cand[b] = ac2;
                            let dlt = ac2 - ar;
                            let mut ce = 0.0f64;
                            for j in 0..gs {
                                let v = ev[j] - dlt * us[j];
                                eproj[src + j] = v;
                                ce += v * v;
                            }
                            cerr[b] = ce;
                        }
                        let mut win = 0usize;
                        for b in 1..b_cur {
                            if cerr[b] < cerr[win] {
                                win = b;
                            }
                        }
                        a_row[lr] = a_cand[win];
                        let src = win * gs;
                        let efx = f64_slice_to_fx(&eproj[src..src + gs])?;
                        for j in 0..gs {
                            sc[lr * gs + j] = s_c[src + j].clone();
                            eoc[lr * gs + j] = efx[j].clone();
                        }
                        for mm in 0..nb {
                            ic[lr * nb + mm] = idx_c[win * nb + mm];
                        }
                    } else {
                        a_row[lr] = ar;
                        sc[lr * gs..(lr + 1) * gs].clone_from_slice(&s_c[..gs]);
                        eoc[lr * gs..(lr + 1) * gs].clone_from_slice(&e_c[..gs]);
                        for mm in 0..nb {
                            ic[lr * nb + mm] = idx_c[mm];
                        }
                    }
                }
                Ok(a_row)
            }));
        }
        for h in handles {
            match h.join() {
                Ok(r) => a_new.extend(r?),
                Err(p) => std::panic::resume_unwind(p),
            }
        }
        Ok(())
    })?;
    Ok(a_new)
}
/// every row in it.
const FEEDBACK_TILE: usize = 8;

/// `wk[:, g0+gs:] -= E · U[g0:g0+gs, g0+gs:]` for the `O × gs` group error `e` (the inter-group
/// feedback of the scalar loop it replaces, line 394 of the script). Per entry the `gs` products
/// are summed in ascending `j` from `0.0` and then subtracted, as the scalar loop did, so the result
/// is bit-identical to it and independent of the thread count. Rows are independent, so they are
/// split across [`INV_THREADS`] scoped workers in [`FEEDBACK_TILE`]-row tiles.
///
/// # Complexity
///
/// Temporal: `O·gs·(n − g0 − gs)` multiply-adds (one contiguous axpy over the tail of each `U` row
/// per `(row, j)`), split over the workers. Spatial: `FEEDBACK_TILE · (n − g0 − gs)` `f64` of
/// accumulator per worker.
fn inter_group_feedback(wk: &mut [f64], e: &[f64], u: &[f64], n: usize, g0: usize, gs: usize) {
    let g1 = g0 + gs;
    let o = wk.len() / n;
    let per = o.div_ceil(INV_THREADS).max(1);
    std::thread::scope(|scope| {
        for (w, ec) in wk.chunks_mut(per * n).zip(e.chunks(per * gs)) {
            scope.spawn(move || {
                let rows = ec.len() / gs;
                let mut acc = vec![0.0f64; FEEDBACK_TILE * (n - g1)];
                let mut r0 = 0;
                while r0 < rows {
                    let tile = FEEDBACK_TILE.min(rows - r0);
                    acc[..tile * (n - g1)].fill(0.0);
                    for j in 0..gs {
                        let urow = &u[(g0 + j) * n + g1..(g0 + j + 1) * n];
                        for t in 0..tile {
                            let coef = ec[(r0 + t) * gs + j];
                            let at = &mut acc[t * (n - g1)..(t + 1) * (n - g1)];
                            for (av, uv) in at.iter_mut().zip(urow) {
                                *av += coef * uv;
                            }
                        }
                    }
                    for t in 0..tile {
                        let wrow = &mut w[(r0 + t) * n + g1..(r0 + t + 1) * n];
                        for (wv, av) in wrow.iter_mut().zip(&acc[t * (n - g1)..(t + 1) * (n - g1)])
                        {
                            *wv -= av;
                        }
                    }
                    r0 += tile;
                }
            });
        }
    });
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
/// `beam = 1` is the single-path block pass; `beam in 2..=MAX_BEAM` runs the delayed-decision beam
/// search (`beam_pass`), which under `wscale` also runs its second pass as a beam search and keeps
/// the smaller-residual pass per row.
///
/// # Errors
/// [`GptqE8Error::BeamOutOfRange`] if `beam == 0` or `beam > MAX_BEAM`;
/// [`GptqE8Error::ZeroDim`] if `o == 0` or `n == 0`;
/// [`GptqE8Error::ShapeMismatch`] if `w.len() != o·n` or `h.len() != n·n`;
/// [`GptqE8Error::BlockMisaligned`] if `n % 8 != 0`; [`GptqE8Error::InvalidDamping`] if `damp <= 0`
/// or `damp` is not finite; [`GptqE8Error::NonFinite`] on a non-finite weight or Hessian entry;
/// [`GptqE8Error::NotPositiveDefinite`] if the damped Hessian has a non-positive Cholesky pivot;
/// [`GptqE8Error::Codebook`] from the exact code index decode.
///
/// # Complexity
/// - Temporal: `O(n³ + o·n³/8²·256·8·beam + o·n²·beam)` checked `i128` and `f64` multiply-adds —
///   the factor `U` (`n³/3`), the `256·8` scan per 8-block per row (×`beam` for the candidate
///   expansion), and the two feedback sweeps. Measure: multiply-adds. Derivation: `[n/8]` blocks ×
///   `256` rows × `8` coordinates × `beam` beams, plus `Σ_j (n − j)` per row per sweep. At
///   `beam = 1` this is the pre-beam cost.
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
    w: &[f64],
    h: &[f64],
    o: usize,
    n: usize,
    damp: f64,
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
    if !damp.is_finite() || damp <= 0.0 {
        return Err(GptqE8Error::InvalidDamping);
    }
    if w.iter().any(|x| !x.is_finite()) || h.iter().any(|x| !x.is_finite()) {
        return Err(GptqE8Error::NonFinite);
    }
    let nb = n / BLOCK;
    let per_group = GROUP / BLOCK;
    let ng = n.div_ceil(GROUP);
    // The weight crosses into its per-row integer carrier once; the still-`f64` selection passes
    // read the transient view `wfull` (`// L1b-L1d delete this`).
    let (mant, e_row) = w_to_rows(w, o, n)?;
    let wfull = rows_to_f64(&mant, &e_row, o, n);
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
    let (wp, hp) = match &perm {
        Some((perm_col, _)) => {
            let mut wp = vec![0.0f64; o * n];
            let mut hp = vec![0.0f64; n * n];
            for r in 0..o {
                for c in 0..n {
                    wp[r * n + c] = wfull[r * n + perm_col[c]];
                }
            }
            for i in 0..n {
                for j in 0..n {
                    hp[i * n + j] = h[perm_col[i] * n + perm_col[j]];
                }
            }
            (wp, hp)
        }
        None => (wfull, h.to_vec()),
    };
    let ufac = feedback_u(&hp, n, damp)?;
    let u = ufac_f64(&ufac); // L1b-L1d delete this
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
        // The group scale `a = 0.828·w_rms` with inverse-variance weights `prec_j = 1/U_jj²` (the
        // Python lines 301–303), integer through [`group_scale_int`]; `wscale` does not change the
        // scale rule here — it only adds the projection passes below.
        let mut a = vec![0.0f64; o];
        let mut a_fx = Vec::with_capacity(o);
        for r in 0..o {
            let af = group_scale_int(&w0[r * gs..(r + 1) * gs], &ufac, g0)?;
            a[r] = fx_to_f64(&af);
            a_fx.push(af);
        }
        // The group block's per-row integer rows `w = mant·2^e_row` ([`w_to_rows`]), the only
        // `f64` → integer conversion of the weight on either branch; the passes carry them.
        let (gm, ge) = w_to_rows(&w0, o, gs)?;
        let (a_best, s_best, e_best, i_best) = if beam == 1 {
            let mut s_cur = vec![Fx { m: 0, e: 0 }; o * gs];
            let mut e_cur = vec![Fx { m: 0, e: 0 }; o * gs];
            let mut i_cur = vec![0u16; o * gb];
            quantise_pass(
                &ufac,
                n,
                g0,
                g1,
                &gm,
                &ge,
                &a_fx,
                o,
                &mut s_cur,
                &mut e_cur,
                &mut i_cur,
                INV_THREADS,
            )?;
            let mut a_best = a.clone();
            let mut s_best = fx_slice_to_f64(&s_cur);
            let mut e_best = fx_slice_to_f64(&e_cur);
            let mut i_best = i_cur;
            if wscale {
                // `project_scale` still reads and returns `f64` (`// L1c-L1d delete this`); the
                // projected scale re-enters the integer pass through [`f64_slice_to_fx`].
                let (a1, e1) = project_scale(&u, n, g0, gs, &s_best, &e_best, &a_best, o);
                let a1_fx = f64_slice_to_fx(&a1)?;
                let mut s2 = vec![Fx { m: 0, e: 0 }; o * gs];
                let mut e2 = vec![Fx { m: 0, e: 0 }; o * gs];
                let mut i2 = vec![0u16; o * gb];
                quantise_pass(
                    &ufac,
                    n,
                    g0,
                    g1,
                    &gm,
                    &ge,
                    &a1_fx,
                    o,
                    &mut s2,
                    &mut e2,
                    &mut i2,
                    INV_THREADS,
                )?;
                let s2f = fx_slice_to_f64(&s2);
                let e2f = fx_slice_to_f64(&e2);
                let (a2, e2b) = project_scale(&u, n, g0, gs, &s2f, &e2f, &a1, o);
                // Second pass: keep the candidate with the smaller whitened residual per row.
                let mut a_fin = vec![0.0f64; o];
                let mut s_fin = vec![0.0f64; o * gs];
                let mut e_fin = vec![0.0f64; o * gs];
                let mut i_fin = vec![0u16; o * gb];
                for r in 0..o {
                    let n1: f64 = (0..gs).map(|j| e1[r * gs + j] * e1[r * gs + j]).sum();
                    let n2: f64 = (0..gs).map(|j| e2b[r * gs + j] * e2b[r * gs + j]).sum();
                    let (ar, sr, er, ir): (f64, &[f64], &[f64], &[u16]) = if n2 < n1 {
                        (a2[r], &s2f, &e2b, &i2)
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
            (a_best, s_best, e_best, i_best)
        } else {
            let mut s1 = vec![Fx { m: 0, e: 0 }; o * gs];
            let mut e1 = vec![Fx { m: 0, e: 0 }; o * gs];
            let mut i1 = vec![0u16; o * gb];
            let a1 = beam_pass(
                &ufac,
                n,
                g0,
                g1,
                &gm,
                &ge,
                &a_fx,
                o,
                beam,
                wscale,
                &mut s1,
                &mut e1,
                &mut i1,
                INV_THREADS,
            )?;
            let s1f = fx_slice_to_f64(&s1);
            let e1f = fx_slice_to_f64(&e1);
            if wscale {
                // The second pass is a beam search again, at the projected scale (lines 304,
                // 379–389); the projected scale re-enters the integer pass through an [`Fx`].
                let a1_fx = f64_slice_to_fx(&a1)?;
                let mut s2 = vec![Fx { m: 0, e: 0 }; o * gs];
                let mut e2 = vec![Fx { m: 0, e: 0 }; o * gs];
                let mut i2 = vec![0u16; o * gb];
                let a2 = beam_pass(
                    &ufac,
                    n,
                    g0,
                    g1,
                    &gm,
                    &ge,
                    &a1_fx,
                    o,
                    beam,
                    true,
                    &mut s2,
                    &mut e2,
                    &mut i2,
                    INV_THREADS,
                )?;
                let s2f = fx_slice_to_f64(&s2);
                let e2f = fx_slice_to_f64(&e2);
                let mut a_fin = vec![0.0f64; o];
                let mut s_fin = vec![0.0f64; o * gs];
                let mut e_fin = vec![0.0f64; o * gs];
                let mut i_fin = vec![0u16; o * gb];
                for r in 0..o {
                    let n1: f64 = (0..gs).map(|j| e1f[r * gs + j] * e1f[r * gs + j]).sum();
                    let n2: f64 = (0..gs).map(|j| e2f[r * gs + j] * e2f[r * gs + j]).sum();
                    let (ar, sr, er, ir): (f64, &[f64], &[f64], &[u16]) = if n2 < n1 {
                        (a2[r], &s2f, &e2f, &i2)
                    } else {
                        (a1[r], &s1f, &e1f, &i1)
                    };
                    a_fin[r] = ar;
                    s_fin[r * gs..(r + 1) * gs].copy_from_slice(&sr[r * gs..(r + 1) * gs]);
                    e_fin[r * gs..(r + 1) * gs].copy_from_slice(&er[r * gs..(r + 1) * gs]);
                    i_fin[r * gb..(r + 1) * gb].copy_from_slice(&ir[r * gb..(r + 1) * gb]);
                }
                (a_fin, s_fin, e_fin, i_fin)
            } else {
                (a1, s1f, e1f, i1)
            }
        };
        for r in 0..o {
            scale_p[r * ng + gi] = a_best[r];
            s_p[r * n + g0..r * n + g1].copy_from_slice(&s_best[r * gs..(r + 1) * gs]);
            idx_p[r * nb + gi * per_group..r * nb + gi * per_group + gb]
                .copy_from_slice(&i_best[r * gb..(r + 1) * gb]);
        }
        // Inter-group feedback: `wk[:, g1:] -= Err_best @ U[g0:g1, g1:]` (line 394).
        if g1 < n {
            inter_group_feedback(&mut wk, &e_best, &u, n, g0, gs);
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

#[cfg(test)]
mod tests {
    use super::super::gptq_e8_carrier::{group_scale_int, w_to_rows};
    use super::{beam_pass, feedback_u, quantise_pass, Fx, BLOCK};

    /// The block passes are bit-identical for `threads = 1` and `threads = 8`: rows are independent
    /// and no quantity is reduced across rows, so the split changes only wall-clock. Exercises the
    /// `o = 37 > 8` split with a short last range, the `beam = 1` pass and the `beam = 4` beam pass
    /// under both `wscale` settings.
    #[test]
    fn q_l1b_l1c_worker_count_is_invariant() {
        let (o, n) = (37usize, 128usize);
        let mut h = vec![0.0f64; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        let ufac = feedback_u(&h, n, 0.01).expect("SPD factor");
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut w = vec![0.0f64; o * n];
        for x in &mut w {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *x = (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
        }
        let mut a = Vec::with_capacity(o);
        for r in 0..o {
            a.push(group_scale_int(&w[r * n..(r + 1) * n], &ufac, 0).expect("scale"));
        }
        let (mant, e_row) = w_to_rows(&w, o, n).expect("rows");
        let gb = n / BLOCK;
        let mut prev: Option<(Vec<Fx>, Vec<Fx>, Vec<u16>)> = None;
        for threads in [1usize, 8] {
            let mut s = vec![Fx { m: 0, e: 0 }; o * n];
            let mut e = vec![Fx { m: 0, e: 0 }; o * n];
            let mut idx = vec![0u16; o * gb];
            quantise_pass(
                &ufac, n, 0, n, &mant, &e_row, &a, o, &mut s, &mut e, &mut idx, threads,
            )
            .expect("quantise_pass");
            if let Some((ps, pe, pi)) = &prev {
                assert_eq!(&s, ps, "worker count changed `s`");
                assert_eq!(&e, pe, "worker count changed `e`");
                assert_eq!(&idx, pi, "worker count changed `idx`");
            }
            prev = Some((s, e, idx));
        }
        for wscale in [false, true] {
            let mut prev: Option<(Vec<Fx>, Vec<Fx>, Vec<u16>, Vec<f64>)> = None;
            for threads in [1usize, 8] {
                let mut s = vec![Fx { m: 0, e: 0 }; o * n];
                let mut e = vec![Fx { m: 0, e: 0 }; o * n];
                let mut idx = vec![0u16; o * gb];
                let a_row = beam_pass(
                    &ufac, n, 0, n, &mant, &e_row, &a, o, 4, wscale, &mut s, &mut e, &mut idx,
                    threads,
                )
                .expect("beam_pass");
                if let Some((ps, pe, pi, pa)) = &prev {
                    assert_eq!(&s, ps, "worker count changed `s` (wscale={wscale})");
                    assert_eq!(&e, pe, "worker count changed `e` (wscale={wscale})");
                    assert_eq!(&idx, pi, "worker count changed `idx` (wscale={wscale})");
                    assert_eq!(
                        a_row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        pa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "worker count changed the scale (wscale={wscale})"
                    );
                }
                prev = Some((s, e, idx, a_row));
            }
        }
    }

    #[test]
    fn select_k_ties_take_the_smaller_flat_index() {
        let mut out = Vec::new();
        super::select_k(&[7_i128, 2, 2, 7, 2], 3, &mut out);
        assert_eq!(out, [(1, 2), (2, 2), (4, 2)]); // ties: smaller flat index
    }
}
