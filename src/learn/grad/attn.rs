//! Causal grouped-query attention with `RoPE`: forward and backward in Q32.32 `i64`.
//!
//! Ports `tools/onebit/fold_transfer.py` `rope_tables` (lines 60-64), `rot` (lines 66-68) and `attn_a`
//! (lines 73-81). The rotation pairs `(i, i + hd/2)`: `rot(x) = cat(-x[h:], x[:h])` with
//! `out = x*cos + rot(x)*sin` is `out[i] = x[i]cos_i - x[i+h]sin_i`, `out[i+h] = x[i+h]cos_i + x[i]sin_i`,
//! the engine's `RopeTable::rotate_head` pairing. Query head `h` reads kv head `h / rep`,
//! `rep = nh / nkv` (`repeat_interleave`). The forward is `softmax(scale * q kᵀ + causal mask) v`,
//! `scale = 1/sqrt(hd)`.
//!
//! Backward is the transpose of the forward at the stated roundings:
//! `dV = Pᵀ dO`, `dP = dO Vᵀ`, `dS = P ⊙ (dP − rowsum(dP ⊙ P))`, `dQ = scale · dS K`, `dK = scale · dSᵀ Q`,
//! with the kv-head gradients summed over the `rep` query heads that share the kv head.
//!
//! Prior art for the recurrence: `../fold` rev `d497161f`, `modelfold/src/arch.rs` `attention_bwd`
//! (line 3790, sha256 `2e243911bc9d46d58f7198783ee066c3f9ecf83d36d08ca24f2136c451a9934d`), in a single-precision float carrier.
//! No `RoPE` backward exists in `~/Workspace`, so [`rope_bwd`] is new (epic W19 Spec).
//!
//! Caller contract: `q` and `k` are rotated per token by [`rope_fwd`] with `pos` = the token index
//! before [`attn_fwd`]; the forward does not rotate. Layouts are token-major:
//! `q[(i·nh + h)·hd + d]`, `k[(j·nkv + g)·hd + d]`, `P[(h·t + i)·t + j]` with `P = 0` for `j > i`.
//!
//! Scalar and vector bodies reused, not restated: `fixed_rsqrt` (scale), `fixed_exp` (softmax
//! numerator), `RopeTable::rotate_one` and the engine's `fixed_sin_cos`/`turn_to_radians` (`RoPE`).
//! Rounding everywhere is round-half-up `(v + 2^31) >> 32`, the engine's `round_shift_32`
//! (`pub(super)`, so restated here as [`round32`]).

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the attention recurrence names its tensors q, k, v, p, d, h, i, j, t, n as in the frozen \
              Python (`attn_a`) and `Numeric/SigmaDelta.lean`; the casts are narrowings or same-width \
              reinterpretations each preceded by an explicit range check (`acc_round`, `to_i64`) or a \
              documented bound (`hd ≤ 2^40`, `pos` and `turns` as `u128` phase products masked to 64 bits)"
)]

use crate::engine::{RopeError, RopeTable, ONE};
use crate::numeric::fixed_exp::{fixed_exp, ExpError};
use crate::numeric::fixed_rsqrt::fixed_rsqrt;
use crate::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians};

/// A refused attention or `RoPE` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnError {
    /// A dimension is zero, `nh` is not a multiple of `nkv`, a buffer is empty, or a length is not a
    /// whole number of tokens or does not match the other operands.
    BadShape,
    /// A checked `i128`/`i64` operation left its range, or a rounded value does not fit `i64`.
    Overflow,
    /// Names the [`fixed_exp`] refusal type. Unreachable: every shift `s − max_j s_j ≤ 0`, and a shift
    /// below `-16·ONE` saturates to `0` before [`fixed_exp`] is called.
    ExpOutOfRange,
    /// `head_dim · ONE` lies outside the [`fixed_rsqrt`] window `[ONE >> 20, ONE << 20]`.
    RsqrtOutOfRange,
    /// A `RoPE` refusal: a length that is not a multiple of `head_dim`, or a rotated component outside
    /// `i64`.
    Rope(RopeError),
}

/// Round-half-up `>> 32` of an `i128` product or sum, the engine's value-path epilogue.
fn round32(v: i128) -> i128 {
    (v + (1_i128 << 31)) >> 32
}

/// Round an `i128` Q64 accumulator to Q32 and check that it fits `i64`.
fn acc_round(acc: i128) -> Result<i64, AttnError> {
    let r = round32(acc);
    if r > i128::from(i64::MAX) || r < i128::from(i64::MIN) {
        Err(AttnError::Overflow)
    } else {
        Ok(r as i64)
    }
}

/// `round(a · b / 2^32)` in Q32.32, checked.
fn mul_round(a: i64, b: i64) -> Result<i64, AttnError> {
    match i128::from(a).checked_mul(i128::from(b)) {
        Some(p) => acc_round(p),
        None => Err(AttnError::Overflow),
    }
}

/// `round(Σ_d a[d] · b[d] / 2^32)` over two equal-length rows, checked.
fn dot_round(a: &[i64], b: &[i64]) -> Result<i64, AttnError> {
    let n = a.len();
    let mut acc: i128 = 0;
    let mut i = 0usize;
    let mut ok = true;
    while i < n && ok {
        match i128::from(a[i]).checked_mul(i128::from(b[i])) {
            Some(p) => match acc.checked_add(p) {
                Some(s) => acc = s,
                None => ok = false,
            },
            None => ok = false,
        }
        i += 1;
    }
    if ok {
        acc_round(acc)
    } else {
        Err(AttnError::Overflow)
    }
}

/// `a · b` as a checked product, for counts and sizes.
fn dim_product(a: usize, b: usize) -> Result<usize, AttnError> {
    match a.checked_mul(b) {
        Some(x) => Ok(x),
        None => Err(AttnError::Overflow),
    }
}

/// The number of tokens in a buffer of `row`-wide rows: `len / row`, refused unless exact and nonzero.
///
/// # Complexity
/// `O(1)`: one remainder.
fn rows_of(len: usize, row: usize) -> Result<usize, AttnError> {
    if row == 0 || len == 0 || !len.is_multiple_of(row) {
        Err(AttnError::BadShape)
    } else {
        Ok(len / row)
    }
}

/// The attention scale `1/sqrt(hd)` in Q32.32, via [`fixed_rsqrt`] of `hd · ONE`.
///
/// # Errors
/// [`AttnError::RsqrtOutOfRange`] when `hd · ONE` leaves the [`fixed_rsqrt`] window.
fn rsqrt_scale(hd: usize) -> Result<i64, AttnError> {
    if hd as u64 > (1_u64 << 40) {
        return Err(AttnError::RsqrtOutOfRange);
    }
    match (hd as i64).checked_mul(ONE) {
        None => Err(AttnError::RsqrtOutOfRange),
        Some(x) => match fixed_rsqrt(x) {
            Ok(s) => Ok(s),
            Err(_) => Err(AttnError::RsqrtOutOfRange),
        },
    }
}

/// `(nh, nkv, hd)` must be nonzero with `nkv | nh`.
fn gqa_shape(nh: usize, nkv: usize, hd: usize) -> Result<(), AttnError> {
    if nh == 0 || nkv == 0 || hd == 0 || !nh.is_multiple_of(nkv) {
        Err(AttnError::BadShape)
    } else {
        Ok(())
    }
}

/// Rotate one token's `x` (`n_heads · hd`) in place-equivalent form: the engine's
/// [`RopeTable::rotate_one`] at position `pos`, reused unchanged.
///
/// `pos` is the token index. The rotation of pair `(j, j + hd/2)` is by the angle
/// `turn_to_radians((turns[j] · pos) mod 2^64)`, the engine's `rotate_head`.
///
/// # Errors
/// [`AttnError::BadShape`] for an empty `x`; [`AttnError::Rope`] wrapping the engine's
/// `VectorLength` (length not a multiple of `hd`) or `OutOfRange` (a rotated component leaves `i64`).
///
/// # Complexity
/// `O(x.len())` time in multiplications and one `fixed_sin_cos` per pair; `O(x.len())` space for the copy.
pub fn rope_fwd(x: &[i64], pos: u32, table: &RopeTable) -> Result<Vec<i64>, AttnError> {
    if x.is_empty() {
        return Err(AttnError::BadShape);
    }
    let mut y: Vec<i64> = x.to_vec();
    match table.rotate_one(&mut y, pos) {
        Ok(()) => Ok(y),
        Err(e) => Err(AttnError::Rope(e)),
    }
}

/// The adjoint of [`rope_fwd`]: the transpose of the rotation by `+angle`, i.e. by `−angle`.
///
/// For a pair with `c = cos`, `s = sin` of the forward phase, `out = round(R x)` with
/// `out0 = x0 c − x1 s`, `out1 = x1 c + x0 s`, so the transpose is `dx0 = dy0 c + dy1 s`,
/// `dx1 = dy1 c − dy0 s`. The `sin`/`cos` are the same values [`rope_fwd`] uses; the phase is
/// computed as the engine does.
///
/// # Errors
/// [`AttnError::BadShape`] for an empty `dy`; [`AttnError::Rope`] with `VectorLength` for a length
/// that is not a multiple of `head_dim`, and `OutOfRange` for a component outside `i64`.
///
/// # Complexity
/// `O(dy.len())` time, one `fixed_sin_cos` per pair; `O(dy.len())` space for the output.
pub fn rope_bwd(dy: &[i64], pos: u32, table: &RopeTable) -> Result<Vec<i64>, AttnError> {
    if dy.is_empty() {
        return Err(AttnError::BadShape);
    }
    let hd = table.head_dim();
    if !dy.len().is_multiple_of(hd) {
        return Err(AttnError::Rope(RopeError::VectorLength));
    }
    let d2 = hd / 2;
    let turns = table.turns();
    let heads = dy.len() / hd;
    let mut dx: Vec<i64> = vec![0; dy.len()];
    let mut h = 0usize;
    let mut ok = true;
    let err = AttnError::Rope(RopeError::OutOfRange);
    while h < heads && ok {
        let base = h * hd;
        let mut j = 0usize;
        while j < d2 && ok {
            let phase = ((u128::from(turns[j]) * u128::from(pos)) & u128::from(u64::MAX)) as u64;
            match fixed_sin_cos(turn_to_radians(phase)) {
                Err(_) => ok = false,
                Ok(sc) => {
                    let y0 = i128::from(dy[base + j]);
                    let y1 = i128::from(dy[base + j + d2]);
                    let c = i128::from(sc.cos);
                    let s = i128::from(sc.sin);
                    match (acc_round(y0 * c + y1 * s), acc_round(y1 * c - y0 * s)) {
                        (Ok(a), Ok(b)) => {
                            dx[base + j] = a;
                            dx[base + j + d2] = b;
                        }
                        _ => ok = false,
                    }
                }
            }
            j += 1;
        }
        h += 1;
    }
    if ok {
        Ok(dx)
    } else {
        Err(err)
    }
}

/// `round(scale · Σ_d q[d] k[d])` for the scaled score of key `j` in kv head `kvh`.
///
/// # Complexity
/// `O(hd)` checked `i128` multiply-adds, `O(1)` space.
fn score_scaled(
    qi: &[i64],
    k: &[i64],
    kvh: usize,
    nkv: usize,
    hd: usize,
    j: usize,
    scale: i64,
) -> Result<i64, AttnError> {
    let base = (j * nkv + kvh) * hd;
    match dot_round(qi, &k[base..base + hd]) {
        Ok(d) => mul_round(d, scale),
        Err(e) => Err(e),
    }
}

/// One softmax numerator `exp(s − m)` in Q32.32, with the rule of `exp_at` in `src/engine/attn.rs`:
/// the subtraction is checked; a shift below `-16·ONE` saturates to `0`; any other shift is
/// [`fixed_exp`]. The backward's `P` must equal the engine's forward, so this mirrors that rule.
///
/// `s ≤ m` by construction, so the shift is in `(-∞, 0]`. The [`ExpError`] arm is unreachable for
/// shifts in `[-16·ONE, 0]`, the [`fixed_exp`] window.
///
/// # Complexity
/// `O(1)`.
fn exp_gap(s: i64, m: i64) -> Result<i64, AttnError> {
    let shifted = s.checked_sub(m).ok_or(AttnError::Overflow)?;
    if shifted < -16 * ONE {
        return Ok(0);
    }
    match fixed_exp(shifted) {
        Ok(e) => Ok(e),
        Err(ExpError::OutOfRange) => Err(AttnError::ExpOutOfRange),
    }
}

/// `round(e · ONE / z)`: one softmax probability from its numerator `e` and the row sum `z`.
///
/// # Complexity
/// `O(1)`: one `i128` multiply and one checked division.
fn prob(e: i64, z: i128) -> Result<i64, AttnError> {
    let num = i128::from(e) * i128::from(ONE);
    match (num + z / 2).checked_div(z) {
        None => Err(AttnError::Overflow),
        Some(q) => {
            if q > i128::from(i64::MAX) || q < i128::from(i64::MIN) {
                Err(AttnError::Overflow)
            } else {
                Ok(q as i64)
            }
        }
    }
}

/// One query row `(i, h)`: scores over keys `0..n` (`n = i + 1`, causal), max-subtracted softmax into
/// `prow[0..n]`, then the output row `orow = Σ_j P_j v_j`.
///
/// `prow` and `orow` are written; `prow[n..]` is left as the caller set it (zero).
///
/// # Complexity
/// `O(n · hd)` checked `i128` operations: `n` scores of `hd` products, `n` exponentials, `hd · n`
/// output products. Space `O(1)` beyond the two row slices.
#[allow(
    clippy::too_many_arguments,
    reason = "the row needs its query, the k/v buffers, the kv head, the head counts and the scale \
              as separate values; a struct would add a wrapper the extraction would have to unfold"
)]
fn attn_row_fwd(
    qi: &[i64],
    k: &[i64],
    v: &[i64],
    kvh: usize,
    nkv: usize,
    hd: usize,
    n: usize,
    scale: i64,
    prow: &mut [i64],
    orow: &mut [i64],
) -> Result<(), AttnError> {
    let mut m: i64 = i64::MIN;
    let mut j = 0usize;
    let mut ok = true;
    let mut err = AttnError::Overflow;
    while j < n && ok {
        match score_scaled(qi, k, kvh, nkv, hd, j, scale) {
            Ok(s) => {
                prow[j] = s;
                if s > m {
                    m = s;
                }
            }
            Err(e) => {
                ok = false;
                err = e;
            }
        }
        j += 1;
    }
    let mut z: i128 = 0;
    j = 0;
    while j < n && ok {
        match exp_gap(prow[j], m) {
            Ok(e) => {
                prow[j] = e;
                match z.checked_add(i128::from(e)) {
                    Some(s) => z = s,
                    None => ok = false,
                }
            }
            Err(e) => {
                ok = false;
                err = e;
            }
        }
        j += 1;
    }
    j = 0;
    while j < n && ok {
        match prob(prow[j], z) {
            Ok(p) => prow[j] = p,
            Err(e) => {
                ok = false;
                err = e;
            }
        }
        j += 1;
    }
    let mut d = 0usize;
    while d < hd && ok {
        let mut acc: i128 = 0;
        let mut jj = 0usize;
        while jj < n && ok {
            let vj = v[(jj * nkv + kvh) * hd + d];
            match i128::from(prow[jj]).checked_mul(i128::from(vj)) {
                Some(prod) => match acc.checked_add(prod) {
                    Some(s) => acc = s,
                    None => ok = false,
                },
                None => ok = false,
            }
            jj += 1;
        }
        if ok {
            match acc_round(acc) {
                Ok(o) => orow[d] = o,
                Err(e) => {
                    ok = false;
                    err = e;
                }
            }
        }
        d += 1;
    }
    if ok {
        Ok(())
    } else {
        Err(err)
    }
}

/// Causal GQA attention forward: returns `(out, P)`.
///
/// `q` is `t × nh × hd`, `k` and `v` are `t × nkv × hd` with `t = q.len() / (nh · hd)`. `out` is
/// `t × nh × hd`. `P` is `nh × t × t` (token-major, the probabilities of the causal lower triangle,
/// zero above it), saved for [`attn_bwd`].
///
/// # Errors
/// [`AttnError::BadShape`] for zero dimensions, `nkv ∤ nh`, an empty or ragged buffer, or
/// mismatched `k`/`v` lengths; [`AttnError::RsqrtOutOfRange`] for `hd` outside the scale window;
/// [`AttnError::ExpOutOfRange`] is unreachable (see its variant): a score gap below `-16·ONE` saturates
/// to `0`, as in the engine's `exp_at`; [`AttnError::Overflow`] for
/// any checked operation or a rounded value outside `i64`.
///
/// # Complexity
/// Time `O(nh · t² · hd)` checked `i128` operations: for each head and row `i`, `(i+1)` scores of `hd`
/// products, `(i+1)` exponentials and `hd · (i+1)` output products; summing `i + 1` over `t` rows gives
/// `t(t+1)/2`. Space `O(nh · t² + t · nh · hd)` words for `P` and `out`.
pub fn attn_fwd(
    q: &[i64],
    k: &[i64],
    v: &[i64],
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Result<(Vec<i64>, Vec<i64>), AttnError> {
    gqa_shape(nh, nkv, hd)?;
    let qrow = dim_product(nh, hd)?;
    let kvrow = dim_product(nkv, hd)?;
    let t = rows_of(q.len(), qrow)?;
    if rows_of(k.len(), kvrow)? != t || v.len() != k.len() {
        return Err(AttnError::BadShape);
    }
    let scale = rsqrt_scale(hd)?;
    let rep = nh / nkv;
    let pn = dim_product(dim_product(nh, t)?, t)?;
    let on = dim_product(t, qrow)?;
    let mut p: Vec<i64> = vec![0; pn];
    let mut out: Vec<i64> = vec![0; on];
    let mut h = 0usize;
    let mut ok = true;
    let mut err = AttnError::Overflow;
    while h < nh && ok {
        let kvh = h / rep;
        let mut i = 0usize;
        while i < t && ok {
            let qbase = (i * nh + h) * hd;
            let pbase = (h * t + i) * t;
            let r = attn_row_fwd(
                &q[qbase..qbase + hd],
                k,
                v,
                kvh,
                nkv,
                hd,
                i + 1,
                scale,
                &mut p[pbase..pbase + t],
                &mut out[qbase..qbase + hd],
            );
            match r {
                Ok(()) => {}
                Err(e) => {
                    ok = false;
                    err = e;
                }
            }
            i += 1;
        }
        h += 1;
    }
    if ok {
        Ok((out, p))
    } else {
        Err(err)
    }
}

/// The softmax Jacobian for one row `(i, h)`: `dP_j = dO · v_j`, `r = Σ_j dP_j P_j`, then
/// `dS_j = P_j · (dP_j − r)`, written over `dprow[0..n]`.
///
/// `r` is `rowsum(dP ⊙ P)`, the term whose omission is the mutation control.
///
/// # Complexity
/// `O(n · hd)` checked `i128` operations for the `n` dot products, `O(n)` for the rowsum and the
/// Jacobian; space `O(1)` beyond the row.
#[allow(
    clippy::too_many_arguments,
    reason = "the row needs the output gradient, the v buffer, the kv head, the head counts, the row \
              length, the probabilities and the gradient slot as separate values"
)]
fn attn_row_ds(
    dout_i: &[i64],
    v: &[i64],
    kvh: usize,
    nkv: usize,
    hd: usize,
    n: usize,
    prow: &[i64],
    dprow: &mut [i64],
) -> Result<(), AttnError> {
    let mut j = 0usize;
    let mut ok = true;
    let mut err = AttnError::Overflow;
    while j < n && ok {
        let base = (j * nkv + kvh) * hd;
        match dot_round(dout_i, &v[base..base + hd]) {
            Ok(d) => dprow[j] = d,
            Err(e) => {
                ok = false;
                err = e;
            }
        }
        j += 1;
    }
    let mut acc: i128 = 0;
    j = 0;
    while j < n && ok {
        match i128::from(dprow[j]).checked_mul(i128::from(prow[j])) {
            Some(prod) => match acc.checked_add(prod) {
                Some(s) => acc = s,
                None => ok = false,
            },
            None => ok = false,
        }
        j += 1;
    }
    let mut rsum: i64 = 0;
    if ok {
        match acc_round(acc) {
            Ok(r) => rsum = r,
            Err(e) => {
                ok = false;
                err = e;
            }
        }
    }
    j = 0;
    while j < n && ok {
        if let Some(diff) = dprow[j].checked_sub(rsum) {
            match mul_round(prow[j], diff) {
                Ok(ds) => dprow[j] = ds,
                Err(e) => {
                    ok = false;
                    err = e;
                }
            }
        } else {
            ok = false;
            err = AttnError::Overflow;
        }
        j += 1;
    }
    if ok {
        Ok(())
    } else {
        Err(err)
    }
}

/// `acc[idx] += a · b` in `i128`, checked.
///
/// # Complexity
/// `O(1)`.
fn acc_add(acc: &mut [i128], idx: usize, a: i64, b: i64) -> Result<(), AttnError> {
    match i128::from(a).checked_mul(i128::from(b)) {
        None => Err(AttnError::Overflow),
        Some(prod) => match acc[idx].checked_add(prod) {
            Some(s) => {
                acc[idx] = s;
                Ok(())
            }
            None => Err(AttnError::Overflow),
        },
    }
}

/// The three input gradients of [`attn_bwd`], in the order `(dq, dk, dv)`.
pub type AttnGrads = (Vec<i64>, Vec<i64>, Vec<i64>);

/// Causal GQA attention backward: returns `(dq, dk, dv)` from the forward inputs, the saved `P` and
/// the output gradient `dout` (`t × nh × hd`).
///
/// `dV_j = Σ_{h,i ≥ j} P_ij dO_i`, `dS_ij = P_ij (dP_ij − Σ_j' dP_ij' P_ij')` with `dP_ij = dO_i · v_j`,
/// `dQ_i = scale Σ_j dS_ij k_j`, `dK_j = scale Σ_{h,i ≥ j} dS_ij q_i`. The kv gradients accumulate
/// unrounded in `i128` over the `rep` query heads and rows, and round once.
///
/// # Errors
/// As [`attn_fwd`] for the shapes and scale, plus [`AttnError::BadShape`] when `p` is not `nh × t × t`
/// or `dout` is not `t × nh × hd`; [`AttnError::Overflow`] for any checked operation.
///
/// # Complexity
/// Time `O(nh · t² · hd)` checked `i128` operations: the `dP` dot products and the three accumulations
/// per (head, row, key, dimension). Space `O(nh · t² + t · nh · hd)` words for the `dS` buffer and the
/// `i128` accumulators (`dq`, `dk`, `dv`).
#[allow(
    clippy::too_many_arguments,
    reason = "the backward takes the three forward inputs, the saved P, the output gradient and the \
              head counts; the fixed signature is the one the extraction and its refinement name"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one pass per (head, row) with the dS row helper, three i128 accumulations and three final \
              roundings; splitting the carried error flag across helpers would not shorten the recurrence"
)]
pub fn attn_bwd(
    q: &[i64],
    k: &[i64],
    v: &[i64],
    p: &[i64],
    dout: &[i64],
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Result<AttnGrads, AttnError> {
    gqa_shape(nh, nkv, hd)?;
    let qrow = dim_product(nh, hd)?;
    let kvrow = dim_product(nkv, hd)?;
    let t = rows_of(q.len(), qrow)?;
    if rows_of(k.len(), kvrow)? != t || v.len() != k.len() {
        return Err(AttnError::BadShape);
    }
    let pn = dim_product(dim_product(nh, t)?, t)?;
    if p.len() != pn || dout.len() != dim_product(t, qrow)? {
        return Err(AttnError::BadShape);
    }
    let scale = rsqrt_scale(hd)?;
    let rep = nh / nkv;
    let kvn = dim_product(t, kvrow)?;
    let qn = dim_product(t, qrow)?;
    let mut ds: Vec<i64> = vec![0; pn];
    let mut dq_acc: Vec<i128> = vec![0; qn];
    let mut dk_acc: Vec<i128> = vec![0; kvn];
    let mut dv_acc: Vec<i128> = vec![0; kvn];
    let mut h = 0usize;
    let mut ok = true;
    let mut err = AttnError::Overflow;
    while h < nh && ok {
        let kvh = h / rep;
        let mut i = 0usize;
        while i < t && ok {
            let qbase = (i * nh + h) * hd;
            let pbase = (h * t + i) * t;
            let n = i + 1;
            match attn_row_ds(
                &dout[qbase..qbase + hd],
                v,
                kvh,
                nkv,
                hd,
                n,
                &p[pbase..pbase + n],
                &mut ds[pbase..pbase + n],
            ) {
                Ok(()) => {}
                Err(e) => {
                    ok = false;
                    err = e;
                }
            }
            let mut j = 0usize;
            while j < n && ok {
                let kbase = (j * nkv + kvh) * hd;
                let pij = p[pbase + j];
                let dsij = ds[pbase + j];
                let mut d = 0usize;
                while d < hd && ok {
                    match acc_add(&mut dv_acc, kbase + d, pij, dout[qbase + d]) {
                        Ok(()) => {}
                        Err(e) => {
                            ok = false;
                            err = e;
                        }
                    }
                    if ok {
                        match acc_add(&mut dk_acc, kbase + d, dsij, q[qbase + d]) {
                            Ok(()) => {}
                            Err(e) => {
                                ok = false;
                                err = e;
                            }
                        }
                    }
                    if ok {
                        match acc_add(&mut dq_acc, qbase + d, dsij, k[kbase + d]) {
                            Ok(()) => {}
                            Err(e) => {
                                ok = false;
                                err = e;
                            }
                        }
                    }
                    d += 1;
                }
                j += 1;
            }
            i += 1;
        }
        h += 1;
    }
    let mut dq: Vec<i64> = vec![0; qn];
    let mut dk: Vec<i64> = vec![0; kvn];
    let mut dv: Vec<i64> = vec![0; kvn];
    let mut x = 0usize;
    while x < qn && ok {
        match acc_round(dq_acc[x]) {
            Ok(a) => match mul_round(a, scale) {
                Ok(b) => dq[x] = b,
                Err(e) => {
                    ok = false;
                    err = e;
                }
            },
            Err(e) => {
                ok = false;
                err = e;
            }
        }
        x += 1;
    }
    x = 0;
    while x < kvn && ok {
        match acc_round(dk_acc[x]) {
            Ok(a) => match mul_round(a, scale) {
                Ok(b) => dk[x] = b,
                Err(e) => {
                    ok = false;
                    err = e;
                }
            },
            Err(e) => {
                ok = false;
                err = e;
            }
        }
        x += 1;
    }
    x = 0;
    while x < kvn && ok {
        match acc_round(dv_acc[x]) {
            Ok(a) => dv[x] = a,
            Err(e) => {
                ok = false;
                err = e;
            }
        }
        x += 1;
    }
    if ok {
        Ok((dq, dk, dv))
    } else {
        Err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64 values in `[-ONE/2, ONE/2)`: deterministic, no float, no wrapping arithmetic.
    fn fixture(state: &mut u64, n: usize) -> Vec<i64> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            out.push((*state % (ONE as u64)) as i64 - ONE / 2);
        }
        out
    }

    /// `Σ dO · out` in `i128` (real value times `2^64`).
    fn loss(
        q: &[i64],
        k: &[i64],
        v: &[i64],
        nh: usize,
        nkv: usize,
        hd: usize,
        dout: &[i64],
    ) -> i128 {
        let (out, _) = attn_fwd(q, k, v, nh, nkv, hd).expect("forward");
        out.iter()
            .zip(dout)
            .map(|(o, d)| i128::from(*o) * i128::from(*d))
            .sum()
    }

    /// Central difference of every entry of `q`, `k`, `v` against the backward.
    ///
    /// Step `h = 2^-12` (raw `2^20`). With `L_raw = Σ dO·out` (real `· 2^64`), the central difference
    /// in raw units is `(L+ − L−) / 2^21`, so `|(L+ − L−) − grad_raw · 2^21| / 2^21` is the error in ULPs.
    ///
    /// Tolerance `T`, derived: each forward output carries error `≤ E` ULP with `E = 1472`, the
    /// worst case of [`fixed_exp`] on `x ≥ -8` (`3.43e-7` relative, `3.43e-7 · 2^32 ≈ 1472`), which is
    /// conservative for the fixtures' score gaps `≤ 1`. The two evaluations therefore differ from the
    /// real `L` by at most `Σ|dO| · E` ULP each, and dividing by `2h = 2^-11` multiplies that by `2^11`:
    /// the FD noise is `≤ 2 · 2^11 · E · Σ|dO_real|` ULP, `= 2^12 · E · Σ|dO_raw| / 2^32`. The truncation
    /// term `h²/6 · f'''` is `≤ 64` ULP for these magnitudes. The backward's own rounding is at most a
    /// few ULP and is absorbed in the same bound.
    fn check_fd(
        q: &[i64],
        k: &[i64],
        v: &[i64],
        nh: usize,
        nkv: usize,
        hd: usize,
        dout: &[i64],
    ) -> i128 {
        let (_, p) = attn_fwd(q, k, v, nh, nkv, hd).expect("forward");
        let (dq, dk, dv) = attn_bwd(q, k, v, &p, dout, nh, nkv, hd).expect("backward");
        let sum_abs: i128 = dout.iter().map(|d| i128::from(*d).abs()).sum();
        let tol: i128 = ((1_i128 << 12) * 1472 * sum_abs >> 32) + 64;
        let mut worst: i128 = 0;
        type Named<'a> = (usize, &'a [i64], &'a [i64]);
        let tensors: [Named; 3] = [(0, q, &dq), (1, k, &dk), (2, v, &dv)];
        for (which, base, grad) in tensors {
            for idx in 0..base.len() {
                let mut plus = base.to_vec();
                let mut minus = base.to_vec();
                plus[idx] += 1 << 20;
                minus[idx] -= 1 << 20;
                let (lp, lm) = match which {
                    0 => (
                        loss(&plus, k, v, nh, nkv, hd, dout),
                        loss(&minus, k, v, nh, nkv, hd, dout),
                    ),
                    1 => (
                        loss(q, &plus, v, nh, nkv, hd, dout),
                        loss(q, &minus, v, nh, nkv, hd, dout),
                    ),
                    _ => (
                        loss(q, k, &plus, nh, nkv, hd, dout),
                        loss(q, k, &minus, nh, nkv, hd, dout),
                    ),
                };
                let diff = (lp - lm) - i128::from(grad[idx]) * (1 << 21);
                worst = worst.max(diff.abs() >> 21);
                assert!(
                    diff.abs() <= tol << 21,
                    "tensor {which} index {idx}: |FD - grad| = {} ULP > T = {tol}",
                    diff.abs() >> 21
                );
            }
        }
        worst
    }

    fn dims_fixture(
        seed: u64,
        t: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
        let mut s = seed;
        // Values in ±1/2: |q·k| ≤ hd/4 and the scale is 1/sqrt(hd), so |score| ≤ sqrt(hd)/4 ≤ 3/4 and
        // every score gap is below 2, inside the fixed_exp window with the documented error.
        let q = fixture(&mut s, t * nh * hd);
        let k = fixture(&mut s, t * nkv * hd);
        let v = fixture(&mut s, t * nkv * hd);
        let dout = fixture(&mut s, t * nh * hd);
        (q, k, v, dout)
    }

    #[test]
    fn rope_adjoint_exact_and_rounded_bound() {
        let table = RopeTable::from_turns(4, vec![1 << 61, 1 << 60]).expect("table");
        let mut s = 7u64;
        for pos in [0u32, 1, 3, 17] {
            let x = fixture(&mut s, 8);
            let dy = fixture(&mut s, 8);
            // Exact identity on the unrounded products, same sin/cos as the engine phase.
            let mut lhs: i128 = 0;
            let mut rhs: i128 = 0;
            for head in 0..2 {
                for j in 0..2 {
                    let phase =
                        ((table.turns()[j] as u128 * pos as u128) & u64::MAX as u128) as u64;
                    let sc = fixed_sin_cos(turn_to_radians(phase)).expect("trig");
                    let (c, sn) = (i128::from(sc.cos), i128::from(sc.sin));
                    let (x0, x1) = (i128::from(x[head * 4 + j]), i128::from(x[head * 4 + j + 2]));
                    let (y0, y1) = (
                        i128::from(dy[head * 4 + j]),
                        i128::from(dy[head * 4 + j + 2]),
                    );
                    lhs += y0 * (x0 * c - x1 * sn) + y1 * (x1 * c + x0 * sn);
                    rhs += x0 * (y0 * c + y1 * sn) + x1 * (y1 * c - y0 * sn);
                }
            }
            assert_eq!(lhs, rhs, "exact adjoint identity at pos {pos}");
            // Rounded: each side carries a half-ULP error per output component.
            let y = rope_fwd(&x, pos, &table).expect("fwd");
            let dx = rope_bwd(&dy, pos, &table).expect("bwd");
            let sl: i128 = dy
                .iter()
                .zip(&y)
                .map(|(a, b)| i128::from(*a) * i128::from(*b))
                .sum();
            let sr: i128 = x
                .iter()
                .zip(&dx)
                .map(|(a, b)| i128::from(*a) * i128::from(*b))
                .sum();
            let bound: i128 = (dy.iter().map(|a| i128::from(*a).abs()).sum::<i128>()
                + x.iter().map(|a| i128::from(*a).abs()).sum::<i128>())
                / 2;
            assert!(
                (sl - sr).abs() <= bound,
                "rounded adjoint at pos {pos}: {} > {bound}",
                (sl - sr).abs()
            );
        }
    }

    #[test]
    fn rope_refusals_reached() {
        let table = RopeTable::from_turns(4, vec![1 << 61, 1 << 60]).expect("table");
        assert_eq!(rope_fwd(&[], 0, &table), Err(AttnError::BadShape));
        assert_eq!(rope_bwd(&[], 0, &table), Err(AttnError::BadShape));
        assert_eq!(
            rope_fwd(&[1, 2, 3], 0, &table),
            Err(AttnError::Rope(RopeError::VectorLength))
        );
        assert_eq!(
            rope_bwd(&[1, 2, 3], 0, &table),
            Err(AttnError::Rope(RopeError::VectorLength))
        );
        // 45 degrees: cos = sin > 0, so MAX·cos + MAX·sin leaves i64.
        let t2 = RopeTable::from_turns(2, vec![1 << 61]).expect("table");
        assert_eq!(
            rope_fwd(&[i64::MAX, i64::MAX], 1, &t2),
            Err(AttnError::Rope(RopeError::OutOfRange))
        );
        assert_eq!(
            rope_bwd(&[i64::MAX, i64::MAX], 1, &t2),
            Err(AttnError::Rope(RopeError::OutOfRange))
        );
    }

    #[test]
    fn fd_fixture_two_heads_gqa_t3_hd4() {
        let (q, k, v, dout) = dims_fixture(11, 3, 2, 1, 4);
        let w = check_fd(&q, &k, &v, 2, 1, 4, &dout);
        eprintln!("fd t3 nh2 nkv1 hd4 worst = {w} ULP");
    }

    #[test]
    fn fd_fixture_two_heads_gqa_t4_hd8() {
        let (q, k, v, dout) = dims_fixture(23, 4, 2, 1, 8);
        let w = check_fd(&q, &k, &v, 2, 1, 8, &dout);
        eprintln!("fd t4 nh2 nkv1 hd8 worst = {w} ULP");
    }

    #[test]
    fn fd_fixture_mha_t2_hd2() {
        let (q, k, v, dout) = dims_fixture(37, 2, 2, 2, 2);
        let w = check_fd(&q, &k, &v, 2, 2, 2, &dout);
        eprintln!("fd t2 nh2 nkv2 hd2 worst = {w} ULP");
    }

    #[test]
    fn one_token_row_is_exact() {
        // One key: P = ONE, dS = 0, so dq = dk = 0 and dv = dO exactly; out = v exactly.
        let q = vec![3 << 30, -(1 << 31)];
        let k = vec![1 << 30, 5 << 29];
        let v = vec![-(7 << 29), 1 << 31];
        let dout = vec![1 << 30, -(3 << 29)];
        let (out, p) = attn_fwd(&q, &k, &v, 1, 1, 2).expect("forward");
        assert_eq!(out, v);
        assert_eq!(p, vec![ONE]);
        let (dq, dk, dv) = attn_bwd(&q, &k, &v, &p, &dout, 1, 1, 2).expect("backward");
        assert_eq!(dq, vec![0, 0]);
        assert_eq!(dk, vec![0, 0]);
        assert_eq!(dv, dout);
        check_fd(&q, &k, &v, 1, 1, 2, &dout);
    }

    #[test]
    fn zero_row_is_uniform() {
        // q = 0: all scores 0, P = ONE/t, out = mean of v.
        let (_, k, v, dout) = dims_fixture(5, 2, 1, 1, 2);
        let q = vec![0; 4];
        let (_, p) = attn_fwd(&q, &k, &v, 1, 1, 2).expect("forward");
        assert_eq!(p[0], ONE);
        // Row 1 has two equal keys: each probability is ONE/2 to the rounding of prob().
        assert_eq!(p[2], p[3]);
        assert!((p[2] - ONE / 2).abs() <= 1, "uniform row: {}", p[2]);
        check_fd(&q, &k, &v, 1, 1, 2, &dout);
    }

    #[test]
    fn causal_mask_later_keys_do_not_leak() {
        let (q, k, v, _) = dims_fixture(41, 3, 2, 1, 4);
        let (out, _) = attn_fwd(&q, &k, &v, 2, 1, 4).expect("forward");
        let mut k2 = k.clone();
        let mut v2 = v.clone();
        // Token 2 is the last; change its k and v entirely.
        for d in 0..4 {
            k2[2 * 4 + d] = -k2[2 * 4 + d] + 12345;
            v2[2 * 4 + d] = 3 * v2[2 * 4 + d] + 99_999;
        }
        let (out2, _) = attn_fwd(&q, &k2, &v2, 2, 1, 4).expect("forward");
        let row = 2 * 4;
        assert_eq!(
            out[..row],
            out2[..row],
            "rows before the changed token must not move"
        );
        assert_ne!(out[row..], out2[row..]);
    }

    #[test]
    fn attn_refusals_reached() {
        let one = vec![ONE; 2];
        // BadShape: nh not a multiple of nkv, empty q, ragged k, bad P / dout lengths.
        assert_eq!(
            attn_fwd(&one, &one, &one, 3, 2, 1),
            Err(AttnError::BadShape)
        );
        assert_eq!(attn_fwd(&[], &one, &one, 1, 1, 2), Err(AttnError::BadShape));
        assert_eq!(
            attn_fwd(&one, &one[..1], &one[..1], 1, 1, 2),
            Err(AttnError::BadShape)
        );
        let (_, p) = attn_fwd(&one, &one, &one, 1, 1, 2).expect("forward");
        assert_eq!(
            attn_bwd(&one, &one, &one, &p[..0], &one, 1, 1, 2),
            Err(AttnError::BadShape)
        );
        assert_eq!(
            attn_bwd(&one, &one, &one, &p, &one[..1], 1, 1, 2),
            Err(AttnError::BadShape)
        );
        // RsqrtOutOfRange: hd = 2^20 + 1 puts hd · ONE above ONE << 20.
        let hd = (1 << 20) + 1;
        let big = vec![0; hd];
        assert_eq!(
            attn_fwd(&big, &big, &big, 1, 1, hd),
            Err(AttnError::RsqrtOutOfRange)
        );
        // Overflow: q = k = MAX, round(MAX·MAX·2 / 2^32) leaves i64.
        let m = vec![i64::MAX, i64::MAX];
        assert_eq!(attn_fwd(&m, &m, &m, 1, 1, 2), Err(AttnError::Overflow));
        // Overflow in the backward: dP = round(dO · v) with dO = v = MAX, v forward-safe at t = 1.
        let mx = vec![i64::MAX];
        let (_, pm) = attn_fwd(&[0], &[0], &mx, 1, 1, 1).expect("forward");
        assert_eq!(
            attn_bwd(&[0], &[0], &mx, &pm, &mx, 1, 1, 1),
            Err(AttnError::Overflow)
        );
    }

    #[test]
    fn exp_saturation_far_entry_is_exact_zero() {
        // hd = 1, q = 5, k = [+5, -5] at Q32.32: row 1 scores +25 and -25, gap -50 < -16·ONE, so the
        // far entry saturates to 0 by the engine's exp_at rule (no refusal). Row 1 is [ONE, 0].
        let q = vec![5 << 32, 5 << 32];
        let k = vec![5 << 32, -(5 << 32)];
        let v = vec![3 << 32, 7 << 32];
        let (_out, p) = attn_fwd(&q, &k, &v, 1, 1, 1).expect("saturating row must succeed");
        assert_eq!(p[3], 0, "far entry must saturate to exact 0");
        assert!(
            (p[2] + p[3] - ONE).abs() <= 1,
            "row 1 sums to ONE: {}",
            p[2] + p[3]
        );
        // Zero gradient through the saturated entry: P[1][1] = 0 gives dV[1] = 0 and dK[1] = 0.
        let dout = vec![ONE, ONE];
        let (_dq, dk, dv) = attn_bwd(&q, &k, &v, &p, &dout, 1, 1, 1).expect("backward");
        assert_eq!(dv[1], 0, "dV of the saturated key");
        assert_eq!(dk[1], 0, "dK of the saturated key");
        // The threshold is -16·ONE: a shift of -10·ONE is inside the window and must not saturate.
        let mid = exp_gap(-10 * ONE, 0).expect("shift -10·ONE is inside the fixed_exp window");
        match fixed_exp(-10 * ONE) {
            Ok(e) => assert_eq!(mid, e),
            Err(_) => panic!("fixed_exp(-10·ONE) must be in window"),
        }
        assert!(mid != 0, "shift -10·ONE must not saturate");
    }
}
