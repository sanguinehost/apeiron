//! Head losses over vocabulary logits: `KL(p_t‖p_s)` and `TV = ½Σ|p_s − p_t|` on softmax rows, their
//! logit gradients, and the chunked head that maps a hidden block through `W_head`.
//!
//! Ports the chunked head loss of `tools/onebit/pv_finish.py` (removed; at 2f15d6a9) (frozen, read only): `KLHead`
//! (lines 829–849, objective `kl`, `sum_t KL(p_t ‖ q_t)`, logit gradient `q − p` formed in the forward)
//! and `TVHead` (lines 852–874, objective `tv`, `sum_t TV(p_t, q_t)`, logit gradient
//! `½ q (s − ⟨s, q⟩)` with `s = sign(q − p)`). Here `p` is the teacher (`z_t`) and `q` the student
//! (`z_s`). The chunking over positions is this module's own (`kl_chunk` was not read for this port).
//! Fold prior art: none read for a fixed-point head.
//!
//! # Carrier and rules
//!
//! - Values are Q32.32 in `i64`; `ONE = 2^32` is `1.0`. Products of two Q32 values are `i128` and are
//!   rounded once per sum, round-half-away-from-zero ([`round_div`]).
//! - Softmax of a row `z`: shift by the row maximum, `u_i = z_i − max`, so `u_i ≤ 0`. Let
//!   `e_i = exp(u_i)` by [`crate::numeric::fixed_exp::fixed_exp`], `S = Σ e_i`, `L = ln S` by
//!   [`crate::numeric::fixed_ln::fixed_ln`]. Then `log p_i = u_i − L` (exact `i128`, refused on `i64`
//!   overflow) and `p_i = round(e_i / S)`.
//! - **Underflow rule.** A shifted value `u_i < −16·ONE` (the lower edge of the `fixed_exp` window) has
//!   `e_i = 0` and `p_i = 0`. The rule is the same in the forward and the backward, because both read
//!   the one `p` vector. `log p_i` stays the exact `u_i − L` there; a flushed `p` contributes nothing
//!   to any sum that weights it.
//! - Gradients are formed in the forward and returned with the loss, as `KLHead` and `TVHead` do.
//!
//! # Derived bounds
//!
//! - `Σ p_i` against `ONE`: `Σ e_i·ONE/S = ONE` exactly, and each `p_i` rounds by at most `½` ULP, so
//!   `|Σ p_i − ONE| ≤ v/2` ULP.
//! - `Σ fixed_exp(log p_i)` against `ONE`: per term `fixed_exp` errs by at most `4 + 10⁻⁸·e_i·2^32`
//!   ULP, the `ln` error (`≤ 7` ULP, `fixed_ln`'s measured bound) shifts every `log p_i` by one common
//!   amount `δ` whose effect on the sum is `≤ 7` ULP, and `Σ e_i·2^32 ≤ v·2^32`. Total `≤ 5v + 50` ULP.
//! - KL: the loss sums `p_t·(log p_t − log p_s)` in exact `i128`, one rounding. The two `ln` terms
//!   are independent, so a true KL of `0` can come out at `≥ −(14 + v/2 + 1)` ULP. Equal logits give
//!   `0` exactly, because both sides then compute the same `log p` and the same `p`.
//! - TV: `½Σ|p_s − p_t| ∈ [0, ONE]` as `Σ|·| ≤ 2·ONE`.

#![allow(
    clippy::many_single_char_names,
    clippy::question_mark,
    reason = "the softmax, gradient and head recurrences name the logit index `i`, the vocabulary index `k`, the hidden index `j`, the row `r`, the shifted value `u` and the probability `p`, as in the frozen script; `?` is outside the pinned extraction fragment, so refusals propagate by explicit `match`"
)]

use crate::engine::ONE;
use crate::numeric::fixed_exp::{fixed_exp, ExpError};
use crate::numeric::fixed_ln::{fixed_ln, LnError};

/// Lower edge of the `fixed_exp` window, `-16 * ONE`. A shifted logit below it has `p = 0`.
const FLUSH: i128 = -16 * (1_i128 << 32);

/// The refusals of the head.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum HeadError {
    /// Empty row, a zero dimension or chunk, or a slice length that is not the stated shape.
    Shape,
    /// A checked `i128` operation or an `i64` result left its range.
    Overflow,
    /// `fixed_exp` refused a shifted logit. Unreachable: the underflow rule sends every value
    /// below `-16 * ONE` to `e = 0` before the call, and shifted values are `≤ 0`.
    Exp(ExpError),
    /// `fixed_ln` refused the softmax sum. Unreachable at `S ≥ ONE − 4` below `i64::MAX − ONE`,
    /// which holds for `v < 2^31` rows.
    Ln(LnError),
}

/// The loss selected by `--objective`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Objective {
    /// `KL(p_t ‖ p_s)`, gradient `p_s − p_t`.
    Kl,
    /// `TV(p_s, p_t) = ½Σ|p_s − p_t|`.
    Tv,
}

/// The shape of a head: `n` positions, hidden width `d`, vocabulary `v`.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Dims {
    /// Positions (rows of `hidden` and `z_t`).
    pub n: usize,
    /// Hidden width (columns of `hidden` and `w_head`).
    pub d: usize,
    /// Vocabulary size (rows of `w_head`, columns of `z_t`).
    pub v: usize,
}

/// `|x|` for `x > i128::MIN` (both call sites: a remainder `n % d` with `d > 0`, and a difference
/// of two `i64`). Written out because Aeneas has no model of `i128::abs`.
fn abs128(x: i128) -> i128 {
    if x < 0 {
        0 - x
    } else {
        x
    }
}

/// Round `n / d` to nearest, ties away from zero; `d > 0`, and `|n| / d` fits `i128`.
fn round_div(n: i128, d: i128) -> i128 {
    let q = n / d;
    let r = n % d;
    if 2 * abs128(r) >= d {
        if n >= 0 {
            q + 1
        } else {
            q - 1
        }
    } else {
        q
    }
}

/// Narrow to `i64` by the bounds check; the cast is exact on the checked range.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the value is range-checked against the i64 bounds on the line above"
)]
fn to64(x: i128) -> Result<i64, HeadError> {
    if x < i128::from(i64::MIN) || x > i128::from(i64::MAX) {
        Err(HeadError::Overflow)
    } else {
        Ok(x as i64)
    }
}

fn add128(a: i128, b: i128) -> Result<i128, HeadError> {
    match a.checked_add(b) {
        Some(s) => Ok(s),
        None => Err(HeadError::Overflow),
    }
}

fn mul128(a: i128, b: i128) -> Result<i128, HeadError> {
    match a.checked_mul(b) {
        Some(s) => Ok(s),
        None => Err(HeadError::Overflow),
    }
}

/// `usize` widened to `i128`: every `usize` value is representable, so the widening is exact.
fn dim128(a: usize) -> i128 {
    a as i128
}

fn mul_dims(a: usize, b: usize) -> Result<usize, HeadError> {
    match a.checked_mul(b) {
        Some(s) => Ok(s),
        None => Err(HeadError::Shape),
    }
}

/// Sign of `x` as `-1`, `0` or `1`.
fn sgn(x: i128) -> i128 {
    match x.cmp(&0) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

/// Softmax weight `e = fixed_exp(z − m)` of one entry; `0` below the flush edge.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the shifted value is range-checked against i64 on the line above the cast"
)]
fn exp_entry(zi: i64, m: i64) -> Result<i64, HeadError> {
    let ui = i128::from(zi) - i128::from(m);
    if ui < FLUSH {
        return Ok(0);
    }
    if ui > i128::from(i64::MAX) {
        return Err(HeadError::Overflow);
    }
    match fixed_exp(ui as i64) {
        Ok(ev) => Ok(ev),
        Err(x) => Err(HeadError::Exp(x)),
    }
}

/// Maximum of a non-empty row; the caller refuses the empty row first.
fn row_max(z: &[i64]) -> i64 {
    let v = z.len();
    let mut m: i64 = z[0];
    let mut i: usize = 1;
    while i < v {
        if z[i] > m {
            m = z[i];
        }
        i += 1;
    }
    m
}

/// Fills `e[i] = exp_entry(z[i], m)` and returns `Σ e`.
///
/// # Complexity
///
/// `O(v)` `fixed_exp` calls and `v` checked `i128` additions; space is the caller's `e`.
fn exp_fill(z: &[i64], m: i64, e: &mut [i64]) -> Result<i128, HeadError> {
    let v = z.len();
    let mut sum: i128 = 0;
    let mut fail: Option<HeadError> = None;
    let mut i: usize = 0;
    while i < v && fail.is_none() {
        match exp_entry(z[i], m) {
            Ok(ev) => {
                e[i] = ev;
                match add128(sum, i128::from(ev)) {
                    Ok(s) => sum = s,
                    Err(x) => fail = Some(x),
                }
            }
            Err(x) => fail = Some(x),
        }
        i += 1;
    }
    match fail {
        Some(x) => Err(x),
        None => Ok(sum),
    }
}

/// `log p_i = (z_i − m) − l` narrowed to `i64`.
fn lp_entry(zi: i64, m: i64, l: i64) -> Result<i64, HeadError> {
    to64(i128::from(zi) - i128::from(m) - i128::from(l))
}

/// `p_i = round(e_i · ONE / S)` narrowed to `i64`; `S > 0`.
fn p_entry(ei: i64, sum: i128) -> Result<i64, HeadError> {
    match mul128(i128::from(ei), i128::from(ONE)) {
        Ok(num) => to64(round_div(num, sum)),
        Err(x) => Err(x),
    }
}

/// Fills `lp` and `p` for every entry; returns the first refusal.
///
/// # Complexity
///
/// `O(v)`: two checked `i128` operations and one division per entry; space is the caller's `lp`
/// and `p`.
fn fill_lp_p(
    z: &[i64],
    m: i64,
    l: i64,
    e: &[i64],
    sum: i128,
    lp: &mut [i64],
    p: &mut [i64],
) -> Option<HeadError> {
    let v = z.len();
    let mut fail: Option<HeadError> = None;
    let mut i: usize = 0;
    while i < v && fail.is_none() {
        match lp_entry(z[i], m, l) {
            Ok(x) => lp[i] = x,
            Err(x) => fail = Some(x),
        }
        if fail.is_none() {
            match p_entry(e[i], sum) {
                Ok(x) => p[i] = x,
                Err(x) => fail = Some(x),
            }
        }
        i += 1;
    }
    fail
}

/// Softmax of one row: returns `(log p, p)` under the rules of the module documentation.
///
/// # Errors
///
/// [`HeadError::Shape`] for an empty row, [`HeadError::Overflow`] when `log p_i = u_i − L` leaves
/// `i64`, and the unreachable [`HeadError::Exp`] and [`HeadError::Ln`] arms.
///
/// # Complexity
///
/// `O(v)` comparisons, `v` `fixed_exp` calls and `v` divisions in `i128`, plus one `fixed_ln`;
/// `O(v)` words of output for `log p` and `p`. The temporaries are the weight vector `e` (`v`
/// words) and the sum `S` (one `i128` word).
fn softmax_row(z: &[i64]) -> Result<(Vec<i64>, Vec<i64>), HeadError> {
    let v = z.len();
    if v == 0 {
        return Err(HeadError::Shape);
    }
    let m = row_max(z);
    let mut e: Vec<i64> = vec![0; v];
    let sum = match exp_fill(z, m, &mut e) {
        Ok(s) => s,
        Err(x) => return Err(x),
    };
    let s64 = match to64(sum) {
        Ok(s) => s,
        Err(x) => return Err(x),
    };
    let l = match fixed_ln(s64) {
        Ok(l) => l,
        Err(x) => return Err(HeadError::Ln(x)),
    };
    let mut lp: Vec<i64> = vec![0; v];
    let mut p: Vec<i64> = vec![0; v];
    match fill_lp_p(z, m, l, &e, sum, &mut lp, &mut p) {
        Some(x) => Err(x),
        None => Ok((lp, p)),
    }
}

/// `log softmax` of one row, `log p_i = u_i − ln S` with `u` shifted by the row maximum.
///
/// Positions with `u_i < −16 * ONE` have `p_i = 0` (the underflow rule); their `log p_i` is still
/// the exact `u_i − ln S`.
///
/// # Errors
///
/// As [`softmax_row`]: [`HeadError::Shape`] for an empty row and [`HeadError::Overflow`] when a
/// log-probability leaves `i64`.
///
/// # Complexity
///
/// `O(v)`, as [`softmax_row`].
pub fn log_softmax_row(z: &[i64]) -> Result<Vec<i64>, HeadError> {
    match softmax_row(z) {
        Ok((lp, _p)) => Ok(lp),
        Err(x) => Err(x),
    }
}

/// `Σ p_t (log p_t − log p_s)` in exact `i128`.
///
/// # Complexity
///
/// `O(v)` checked `i128` multiply-adds; space `O(1)`.
fn kl_acc(p_t: &[i64], lp_t: &[i64], lp_s: &[i64]) -> Result<i128, HeadError> {
    let v = p_t.len();
    let mut acc: i128 = 0;
    let mut fail: Option<HeadError> = None;
    let mut i: usize = 0;
    while i < v && fail.is_none() {
        let delta = i128::from(lp_t[i]) - i128::from(lp_s[i]);
        match mul128(i128::from(p_t[i]), delta) {
            Err(x) => fail = Some(x),
            Ok(term) => match add128(acc, term) {
                Ok(s) => acc = s,
                Err(x) => fail = Some(x),
            },
        }
        i += 1;
    }
    match fail {
        Some(x) => Err(x),
        None => Ok(acc),
    }
}

/// Fills `dz[i] = p_s,i − p_t,i`, narrowed to `i64`; returns the first refusal.
fn dz_diff(p_s: &[i64], p_t: &[i64], dz: &mut [i64]) -> Option<HeadError> {
    let v = p_s.len();
    let mut fail: Option<HeadError> = None;
    let mut i: usize = 0;
    while i < v && fail.is_none() {
        match to64(i128::from(p_s[i]) - i128::from(p_t[i])) {
            Ok(g) => dz[i] = g,
            Err(x) => fail = Some(x),
        }
        i += 1;
    }
    fail
}

/// `KL(p_t ‖ p_s)` of one row and its gradient `dz_s = p_s − p_t`, both in Q32.32.
///
/// `z_s` is the student logits, `z_t` the teacher logits. The loss is `Σ p_t (log p_t − log p_s)`
/// in exact `i128`, rounded once.
///
/// # Errors
///
/// [`HeadError::Shape`] when the lengths differ or a row is empty; [`HeadError::Overflow`] from
/// the softmax or the gradient narrowing.
///
/// # Complexity
///
/// `O(v)` for two softmaxes and one pass over the `v` terms; output `v + 1` words.
pub fn kl_row(z_s: &[i64], z_t: &[i64]) -> Result<(i64, Vec<i64>), HeadError> {
    if z_s.len() != z_t.len() {
        return Err(HeadError::Shape);
    }
    let (lp_s, p_s) = match softmax_row(z_s) {
        Ok(pair) => pair,
        Err(x) => return Err(x),
    };
    let (lp_t, p_t) = match softmax_row(z_t) {
        Ok(pair) => pair,
        Err(x) => return Err(x),
    };
    let acc = match kl_acc(&p_t, &lp_t, &lp_s) {
        Ok(a) => a,
        Err(x) => return Err(x),
    };
    let mut dz: Vec<i64> = vec![0; z_s.len()];
    if let Some(x) = dz_diff(&p_s, &p_t, &mut dz) {
        return Err(x);
    }
    match to64(round_div(acc, i128::from(ONE))) {
        Ok(loss) => Ok((loss, dz)),
        Err(x) => Err(x),
    }
}

/// Scan of one row for the TV loss: fills `sg[i] = sign(p_s,i − p_t,i)` and returns
/// `(Σ |p_s − p_t|, Σ p_s,j s_j)`.
///
/// # Complexity
///
/// `O(v)` checked `i128` operations; space is the caller's `sg`.
fn tv_scan(p_s: &[i64], p_t: &[i64], sg: &mut [i128]) -> Result<(i128, i128), HeadError> {
    let v = p_s.len();
    let mut abs_sum: i128 = 0;
    let mut signed: i128 = 0;
    let mut fail: Option<HeadError> = None;
    let mut i: usize = 0;
    while i < v && fail.is_none() {
        let diff = i128::from(p_s[i]) - i128::from(p_t[i]);
        sg[i] = sgn(diff);
        match add128(abs_sum, abs128(diff)) {
            Ok(s) => abs_sum = s,
            Err(x) => fail = Some(x),
        }
        match mul128(i128::from(p_s[i]), sg[i]) {
            Ok(term) => match add128(signed, term) {
                Ok(s) => signed = s,
                Err(x) => fail = Some(x),
            },
            Err(x) => fail = Some(x),
        }
        i += 1;
    }
    match fail {
        Some(x) => Err(x),
        None => Ok((abs_sum, signed)),
    }
}

/// Fills `dz[i] = ½ p_s,i (s_i·ONE − A) / ONE` rounded once, with `A = Σ p_s,j s_j` in Q32.
fn tv_dz(p_s: &[i64], sg: &[i128], signed: i128, dz: &mut [i64]) -> Option<HeadError> {
    let v = p_s.len();
    let one = i128::from(ONE);
    let mut fail: Option<HeadError> = None;
    let mut i: usize = 0;
    while i < v && fail.is_none() {
        match mul128(i128::from(p_s[i]), sg[i] * one - signed) {
            Err(x) => fail = Some(x),
            Ok(num) => match to64(round_div(num, 2 * one)) {
                Ok(g) => dz[i] = g,
                Err(x) => fail = Some(x),
            },
        }
        i += 1;
    }
    fail
}

/// `TV(p_s, p_t) = ½Σ|p_s − p_t|` of one row and its gradient, both in Q32.32.
///
/// With `s_i = sign(p_s,i − p_t,i)` (`sign(0) = 0`), the gradient is
/// `dz_s,i = ½ p_s,i (s_i − Σ_j p_s,j s_j)`, rounded once in `i128`.
///
/// # Errors
///
/// As [`kl_row`].
///
/// # Complexity
///
/// `O(v)` for two softmaxes, one pass for the loss and the signed sum `A = Σ_j p_s,j s_j`, and one
/// pass for the gradient; output `v + 1` words.
pub fn tv_row(z_s: &[i64], z_t: &[i64]) -> Result<(i64, Vec<i64>), HeadError> {
    if z_s.len() != z_t.len() {
        return Err(HeadError::Shape);
    }
    let (_lp_s, p_s) = match softmax_row(z_s) {
        Ok(pair) => pair,
        Err(x) => return Err(x),
    };
    let (_lp_t, p_t) = match softmax_row(z_t) {
        Ok(pair) => pair,
        Err(x) => return Err(x),
    };
    let v = z_s.len();
    let mut sg: Vec<i128> = vec![0; v];
    let (abs_sum, signed) = match tv_scan(&p_s, &p_t, &mut sg) {
        Ok(pair) => pair,
        Err(x) => return Err(x),
    };
    let mut dz: Vec<i64> = vec![0; v];
    if let Some(x) = tv_dz(&p_s, &sg, signed, &mut dz) {
        return Err(x);
    }
    match to64(round_div(abs_sum, 2)) {
        Ok(loss) => Ok((loss, dz)),
        Err(x) => Err(x),
    }
}

/// One row's loss and logit gradient under `objective`.
fn row_objective(
    objective: Objective,
    z_s: &[i64],
    z_t: &[i64],
) -> Result<(i64, Vec<i64>), HeadError> {
    match objective {
        Objective::Kl => kl_row(z_s, z_t),
        Objective::Tv => tv_row(z_s, z_t),
    }
}

/// `round((a[abase..abase+d] · b[bbase..bbase+d]) / ONE)` with a checked `i128` accumulator.
///
/// # Complexity
///
/// `O(d)` checked `i128` multiply-adds and one rounding; space `O(1)`.
fn dot_round(a: &[i64], abase: usize, b: &[i64], bbase: usize, d: usize) -> Result<i64, HeadError> {
    let mut acc: i128 = 0;
    let mut fail: Option<HeadError> = None;
    let mut j: usize = 0;
    while j < d && fail.is_none() {
        match mul128(i128::from(a[abase + j]), i128::from(b[bbase + j])) {
            Err(x) => fail = Some(x),
            Ok(t) => match add128(acc, t) {
                Ok(s) => acc = s,
                Err(x) => fail = Some(x),
            },
        }
        j += 1;
    }
    match fail {
        Some(x) => Err(x),
        None => to64(round_div(acc, i128::from(ONE))),
    }
}

/// The student logits of row `r`: `z_k = round(Σ_j hidden_rj · w_kj / ONE)` for `k < v`.
///
/// # Errors
///
/// [`HeadError::Overflow`] when a dot product or a logit leaves its range.
///
/// # Complexity
///
/// `O(v·d)` `i128` multiply-adds, `v` roundings; output `v` words.
fn logits_row(
    hidden: &[i64],
    w_head: &[i64],
    r: usize,
    d: usize,
    v: usize,
) -> Result<Vec<i64>, HeadError> {
    let mut zs: Vec<i64> = vec![0; v];
    let mut fail: Option<HeadError> = None;
    let mut k: usize = 0;
    while k < v && fail.is_none() {
        match dot_round(hidden, r * d, w_head, k * d, d) {
            Ok(z) => zs[k] = z,
            Err(x) => fail = Some(x),
        }
        k += 1;
    }
    match fail {
        Some(x) => Err(x),
        None => Ok(zs),
    }
}

/// `d_hidden[r,j] = round((Σ_k dz_k · w_kj) / n_scale)` for one `j`.
///
/// # Complexity
///
/// `O(v)` checked `i128` multiply-adds and one rounding.
fn dh_entry(
    dz: &[i64],
    w_head: &[i64],
    d: usize,
    j: usize,
    n_scale: i128,
) -> Result<i64, HeadError> {
    let v = dz.len();
    let mut acc: i128 = 0;
    let mut fail: Option<HeadError> = None;
    let mut k: usize = 0;
    while k < v && fail.is_none() {
        match mul128(i128::from(dz[k]), i128::from(w_head[k * d + j])) {
            Err(x) => fail = Some(x),
            Ok(t) => match add128(acc, t) {
                Ok(s) => acc = s,
                Err(x) => fail = Some(x),
            },
        }
        k += 1;
    }
    match fail {
        Some(x) => Err(x),
        None => to64(round_div(acc, n_scale)),
    }
}

/// Writes `dh[dbase + j] = dh_entry(j)` for `j < d`.
fn dh_fill(
    dz: &[i64],
    w_head: &[i64],
    d: usize,
    n_scale: i128,
    dh: &mut [i64],
    dbase: usize,
) -> Option<HeadError> {
    let mut fail: Option<HeadError> = None;
    let mut j: usize = 0;
    while j < d && fail.is_none() {
        match dh_entry(dz, w_head, d, j, n_scale) {
            Ok(g) => dh[dbase + j] = g,
            Err(x) => fail = Some(x),
        }
        j += 1;
    }
    fail
}

/// `dw[t] += dz[k] · hidden[hbase + j]` for `t = k·d + j`, one flat loop over `v·d` entries.
///
/// # Complexity
///
/// `v·d` checked `i128` multiply-adds, one per entry of `dw`; the index is split by `t / d` and
/// `t % d` with `d > 0`.
fn dw_fill(
    dz: &[i64],
    hidden: &[i64],
    hbase: usize,
    d: usize,
    dw: &mut [i128],
) -> Option<HeadError> {
    let total = dz.len() * d;
    let mut fail: Option<HeadError> = None;
    let mut t: usize = 0;
    while t < total && fail.is_none() {
        let k = t / d;
        let j = t % d;
        match mul128(i128::from(dz[k]), i128::from(hidden[hbase + j])) {
            Err(x) => fail = Some(x),
            Ok(term) => match add128(dw[t], term) {
                Ok(s) => dw[t] = s,
                Err(x) => fail = Some(x),
            },
        }
        t += 1;
    }
    fail
}

/// Fold one row's loss and gradient into the head's outputs; returns the new loss accumulator.
///
/// Adds `loss` into `loss_acc`, writes `d_hidden` row `r` (at `dbase`) and adds `dz_k · hidden_r,j`
/// into `dw` (exact, unrounded; rounded once by the caller).
///
/// # Errors
///
/// [`HeadError::Overflow`] when an accumulator or an output leaves its range.
///
/// # Complexity
///
/// `O(v·d)` `i128` multiply-adds: one `v`-term sum per output of `dh` row and one `d`-term update
/// per `(k, j)` pair of `dw`.
#[allow(
    clippy::too_many_arguments,
    reason = "the row's loss, gradient, inputs and two outputs"
)]
fn absorb_row(
    loss: i64,
    dz: &[i64],
    hidden: &[i64],
    hbase: usize,
    w_head: &[i64],
    d: usize,
    n_scale: i128,
    loss_acc: i128,
    dh: &mut [i64],
    dbase: usize,
    dw: &mut [i128],
) -> Result<i128, HeadError> {
    let acc = match add128(loss_acc, i128::from(loss)) {
        Ok(s) => s,
        Err(x) => return Err(x),
    };
    if let Some(x) = dh_fill(dz, w_head, d, n_scale, dh, dbase) {
        return Err(x);
    }
    if let Some(x) = dw_fill(dz, hidden, hbase, d, dw) {
        return Err(x);
    }
    Ok(acc)
}

/// One row of the chunked head: logits, objective and the fold into the outputs.
///
/// # Complexity
///
/// `O(v·d)`, as [`absorb_row`] and [`logits_row`].
#[allow(
    clippy::too_many_arguments,
    reason = "the objective, inputs, row index, scale, accumulator and two outputs"
)]
fn row_step(
    objective: Objective,
    hidden: &[i64],
    w_head: &[i64],
    z_t: &[i64],
    dims: Dims,
    r: usize,
    n_scale: i128,
    loss_acc: i128,
    dh: &mut [i64],
    dw: &mut [i128],
) -> Result<i128, HeadError> {
    let d = dims.d;
    let v = dims.v;
    let zs = match logits_row(hidden, w_head, r, d, v) {
        Ok(z) => z,
        Err(x) => return Err(x),
    };
    let (loss, dz) = match row_objective(objective, &zs, &z_t[r * v..(r + 1) * v]) {
        Ok(pair) => pair,
        Err(x) => return Err(x),
    };
    absorb_row(
        loss,
        &dz,
        hidden,
        r * d,
        w_head,
        d,
        n_scale,
        loss_acc,
        dh,
        r * d,
        dw,
    )
}

/// Runs [`row_step`] over every row in order; returns the exact loss accumulator.
///
/// # Complexity
///
/// `O(n·v·d)` checked `i128` operations; space is the outputs `dh`, `dw` and one row of logits.
#[allow(
    clippy::too_many_arguments,
    reason = "the objective, inputs, dims, scale and two outputs"
)]
fn rows_loop(
    objective: Objective,
    hidden: &[i64],
    w_head: &[i64],
    z_t: &[i64],
    dims: Dims,
    n_scale: i128,
    dh: &mut [i64],
    dw: &mut [i128],
) -> Result<i128, HeadError> {
    let n = dims.n;
    let mut loss_acc: i128 = 0;
    let mut fail: Option<HeadError> = None;
    let mut r: usize = 0;
    while r < n && fail.is_none() {
        match row_step(
            objective, hidden, w_head, z_t, dims, r, n_scale, loss_acc, dh, dw,
        ) {
            Ok(a) => loss_acc = a,
            Err(x) => fail = Some(x),
        }
        r += 1;
    }
    match fail {
        Some(x) => Err(x),
        None => Ok(loss_acc),
    }
}

/// `out[j] = round(dw[j] / n_scale)` narrowed to `i64`.
fn dw_out_fill(dw: &[i128], n_scale: i128, out: &mut [i64]) -> Option<HeadError> {
    let len = dw.len();
    let mut fail: Option<HeadError> = None;
    let mut j: usize = 0;
    while j < len && fail.is_none() {
        match to64(round_div(dw[j], n_scale)) {
            Ok(x) => out[j] = x,
            Err(x) => fail = Some(x),
        }
        j += 1;
    }
    fail
}

/// Mean loss, `d_hidden` and `d_w_head` of the chunked head.
///
/// The student logits are `z_s = hidden · w_headᵀ`, one rounded dot product per entry. Rows are
/// folded one at a time in order, so the logits held at one time are one row of `v` words. The
/// per-position gradient `dz_s` of the row objective is applied to `d_hidden` and accumulated into
/// `d_w_head`; both are scaled by `1/n` for the mean. Every accumulation is exact in `i128`, and
/// each output is rounded once, so the result does not depend on `chunk` (validated nonzero).
///
/// Returns `(mean loss, d_hidden[n×d], d_w_head[v×d])`.
///
/// # Errors
///
/// [`HeadError::Shape`] when a dimension or the chunk is zero, when a slice length is not the
/// stated shape, or when a product of dimensions overflows `usize`; [`HeadError::Overflow`] when a
/// logit, a gradient or an accumulator leaves its range, and any refusal of [`kl_row`] or
/// [`tv_row`].
///
/// # Complexity
///
/// Count: `3·n·v·d` `i128` multiply-adds (logits, `d_hidden`, `d_w_head`) plus `O(n·v)` for the row
/// objectives. Derivation: each of the `n` rows runs one logit dot product per `k < v` of length
/// `d`, and the two gradient products of the same shape. Space: `v` logit words for one row,
/// `v·d` `i128` accumulator words, and the `n·d` output.
pub fn head_chunked(
    hidden: &[i64],
    w_head: &[i64],
    z_t: &[i64],
    dims: Dims,
    chunk: usize,
    objective: Objective,
) -> Result<(i64, Vec<i64>, Vec<i64>), HeadError> {
    let (n, d, v) = (dims.n, dims.d, dims.v);
    if n == 0 || d == 0 || v == 0 || chunk == 0 {
        return Err(HeadError::Shape);
    }
    let nd = match mul_dims(n, d) {
        Ok(x) => x,
        Err(x) => return Err(x),
    };
    let vd = match mul_dims(v, d) {
        Ok(x) => x,
        Err(x) => return Err(x),
    };
    let nv = match mul_dims(n, v) {
        Ok(x) => x,
        Err(x) => return Err(x),
    };
    if hidden.len() != nd || w_head.len() != vd || z_t.len() != nv {
        return Err(HeadError::Shape);
    }
    let n_scale = match mul128(dim128(n), i128::from(ONE)) {
        Ok(x) => x,
        Err(x) => return Err(x),
    };
    let mut dh: Vec<i64> = vec![0; nd];
    let mut dw: Vec<i128> = vec![0; vd];
    let loss_acc = match rows_loop(
        objective, hidden, w_head, z_t, dims, n_scale, &mut dh, &mut dw,
    ) {
        Ok(a) => a,
        Err(x) => return Err(x),
    };
    let loss = match to64(round_div(loss_acc, dim128(n))) {
        Ok(x) => x,
        Err(x) => return Err(x),
    };
    let mut dw_out: Vec<i64> = vec![0; vd];
    if let Some(x) = dw_out_fill(&dw, n_scale, &mut dw_out) {
        return Err(x);
    }
    Ok((loss, dh, dw_out))
}

#[cfg(test)]
mod tests {
    use super::{
        head_chunked, kl_row, log_softmax_row, logits_row, round_div, row_objective, softmax_row,
        tv_row, Dims, HeadError, Objective, FLUSH,
    };
    use crate::engine::ONE;
    use crate::numeric::fixed_exp::fixed_exp;

    /// A value in thousandths of a unit, in Q32.32.
    fn m(x: i64) -> i64 {
        x * ONE / 1000
    }

    fn milli(xs: &[i64]) -> Vec<i64> {
        xs.iter().map(|&x| m(x)).collect()
    }

    /// Central-difference step `h = 2^-12` in Q32.32 units.
    const H: i64 = ONE >> 12;

    /// The FD quotient is `(L(+h) − L(−h)) / (2h)` in real units. Multiplied by `2^32` to Q32 units
    /// that is `(ΔL) · 2^32 / (2 · 2^-12 · 2^32) = ΔL · 2^11`.
    const FD_SCALE: i128 = 2048;

    /// Derived tolerance, in Q32 ULPs (`2^-32`). One loss evaluation is off by at most
    /// `2v + 14` ULP (`½` ULP of rounding per `p`, times `|Δ| ≤ 2` real in these fixtures, `v` terms,
    /// plus the `ln` error of `≤ 7` ULP on each side). The difference of two evaluations is off by
    /// at most twice that, and the quotient multiplies it by `2^11`. For `v ≤ 8`:
    /// `2^11 · 2 · (2·8 + 14) = 2^11 · 60 < 2^17`. The gate is `2^18`.
    const FD_TOL: i128 = 1 << 18;

    /// `(f(z + h e_i) − f(z − h e_i)) · 2^11`, the central difference in Q32 units.
    fn central(f: &dyn Fn(&[i64]) -> i64, z: &[i64], i: usize) -> i128 {
        let mut zp = z.to_vec();
        let mut zm = z.to_vec();
        zp[i] += H;
        zm[i] -= H;
        (i128::from(f(&zp)) - i128::from(f(&zm))) * FD_SCALE
    }

    fn fixtures() -> Vec<(Vec<i64>, Vec<i64>)> {
        vec![
            (milli(&[200, 1000, -300, 500]), milli(&[0, 600, 100, -400])),
            (milli(&[-1500, 0, 800]), milli(&[900, -200, 300])),
            // The max-magnitude row: the spread is 15 units, so every shifted value is in the window.
            (milli(&[-7500, 7500, 0]), milli(&[5000, 3000, 0])),
        ]
    }

    #[test]
    fn softmax_sums_to_one_within_derived_bound() {
        for (zs, _) in fixtures() {
            let v = i128::try_from(zs.len()).unwrap();
            let (lp, p) = softmax_row(&zs).unwrap();
            let sum_p: i128 = p.iter().map(|&x| i128::from(x)).sum();
            // Each p rounds by at most ½ ULP and the exact sum is ONE: |Σp − ONE| ≤ v/2.
            assert!((sum_p - i128::from(ONE)).abs() <= v);
            // Σ fixed_exp(log p): ≤ 5v + 50 ULP, derived in the module header.
            let mut s: i128 = 0;
            for &l in &lp {
                s += i128::from(fixed_exp(l).unwrap());
            }
            assert!(
                (s - i128::from(ONE)).abs() <= 5 * v + 50,
                "sum exp(log p) = {s}"
            );
        }
        // The one-element and zero rows: p = ONE exactly, log p = 0.
        let (lp1, p1) = softmax_row(&[m(7)]).unwrap();
        assert!(lp1[0].abs() <= 7 && p1 == vec![ONE]);
        let (lp, p) = softmax_row(&[0, 0, 0, 0]).unwrap();
        assert!(lp.iter().all(|&x| x == lp[0]));
        // log p = -ln 4 = -2 ln 2; 2 * LN2_Q32 = 5_954_088_944, within the fixed_ln and fixed_exp bounds.
        assert!((lp[0] + 5_954_088_944).abs() <= 7 + 4);
        assert!(p.iter().all(|&x| (x - ONE / 4).abs() <= 1));
    }

    #[test]
    fn underflow_flushes_p_and_keeps_log_p() {
        // u = −20 units of ONE is below the window: p = 0 there, and log p = u − ln S exactly.
        let z = [0, -20 * ONE];
        let (lp, p) = softmax_row(&z).unwrap();
        assert_eq!(p[1], 0);
        assert_eq!(lp[1], -20 * ONE - lp[0]);
        assert!(i128::from(-20 * ONE) < FLUSH);
    }

    #[test]
    fn kl_gradient_matches_central_difference() {
        for (zs, zt) in fixtures() {
            let (_, dz) = kl_row(&zs, &zt).unwrap();
            for (i, &g) in dz.iter().enumerate() {
                let fd = central(&|z: &[i64]| kl_row(z, &zt).unwrap().0, &zs, i);
                let err = (i128::from(g) - fd).abs();
                assert!(err <= FD_TOL, "kl dz[{i}] = {g} fd = {fd} err = {err}");
            }
        }
        // Zero row: both logits zero, gradient zero.
        let z = vec![0, 0, 0, 0];
        let (loss, dz) = kl_row(&z, &z).unwrap();
        assert_eq!((loss, dz), (0, vec![0; 4]));
    }

    #[test]
    fn tv_gradient_matches_central_difference() {
        // Margin check: every |p_s − p_t| exceeds 2^-10 (2^22 ULP), well above the step's effect on p
        // (≈ 2h · p(1−p) ≤ 2^-11 · ¼), so no sign flips inside the central difference.
        let margin: i128 = 1 << 22;
        for (zs, zt) in fixtures() {
            let (_, ps) = softmax_row(&zs).unwrap();
            let (_, pt) = softmax_row(&zt).unwrap();
            for i in 0..zs.len() {
                assert!(
                    (i128::from(ps[i]) - i128::from(pt[i])).abs() > margin,
                    "fixture margin at {i}"
                );
            }
            let (_, dz) = tv_row(&zs, &zt).unwrap();
            for (i, &g) in dz.iter().enumerate() {
                let fd = central(&|z: &[i64]| tv_row(z, &zt).unwrap().0, &zs, i);
                let err = (i128::from(g) - fd).abs();
                assert!(err <= FD_TOL, "tv dz[{i}] = {g} fd = {fd} err = {err}");
            }
        }
    }

    #[test]
    fn kl_and_tv_vanish_at_equal_logits_and_one_element_rows() {
        for (zs, _) in fixtures() {
            assert_eq!(kl_row(&zs, &zs).unwrap(), (0, vec![0; zs.len()]));
            assert_eq!(tv_row(&zs, &zs).unwrap(), (0, vec![0; zs.len()]));
        }
        assert_eq!(kl_row(&[m(3)], &[m(-2)]).unwrap(), (0, vec![0]));
        assert_eq!(tv_row(&[m(3)], &[m(-2)]).unwrap(), (0, vec![0]));
    }

    #[test]
    fn kl_is_nonnegative_within_derived_bound() {
        for (zs, zt) in fixtures() {
            let (loss, _) = kl_row(&zs, &zt).unwrap();
            // ≥ −(14 + v/2 + 1) ULP, from the two independent ln errors (module header).
            assert!(loss >= -(14 + 2 + 1), "kl = {loss}");
        }
    }

    #[test]
    fn tv_lies_in_zero_one() {
        for (zs, zt) in fixtures() {
            let (loss, _) = tv_row(&zs, &zt).unwrap();
            assert!((0..=ONE).contains(&loss), "tv = {loss}");
        }
    }

    fn hidden_fixture() -> (Vec<i64>, Vec<i64>, Vec<i64>, Dims) {
        let hidden = milli(&[500, -200, 300, -400, 700, 100, 900, 0, -600, 250, 250, -800]);
        let w = milli(&[300, -100, 200, -500, 400, 0, 100, 100, 100, 0, -700, 600]);
        let zt = milli(&[
            100, -200, 300, 0, 0, 500, -100, 200, -300, 300, 600, -50, 800, -800, 0, 100,
        ]);
        (hidden, w, zt, Dims { n: 4, d: 3, v: 4 })
    }

    #[test]
    fn head_chunked_is_independent_of_chunk() {
        let (h, w, zt, dims) = hidden_fixture();
        for objective in [Objective::Kl, Objective::Tv] {
            let one = head_chunked(&h, &w, &zt, dims, 1, objective).unwrap();
            let three = head_chunked(&h, &w, &zt, dims, 3, objective).unwrap();
            let all = head_chunked(&h, &w, &zt, dims, dims.n, objective).unwrap();
            assert_eq!(one, three);
            assert_eq!(one, all);
            assert_eq!(one.1.len(), 12);
            assert_eq!(one.2.len(), 12);
        }
    }

    /// The `n = 2`, `d = 3`, `v = 4` fixture of the head gradient tests.
    fn grad_fixture() -> (Vec<i64>, Vec<i64>, Vec<i64>, Dims) {
        let hidden = milli(&[600, -300, 200, -100, 800, 400]);
        let w = milli(&[
            500, 100, -200, -400, 300, 700, 200, -600, 400, 100, 500, -300,
        ]);
        let zt = milli(&[300, -500, 100, 700, -200, 400, 0, 600]);
        (hidden, w, zt, Dims { n: 2, d: 3, v: 4 })
    }

    #[test]
    fn head_gradients_match_central_difference() {
        const HEAD_FD_TOL: i128 = 1 << 17;
        let (h, w, zt, dims) = grad_fixture();
        let (n, d, v) = (dims.n, dims.d, dims.v);
        // Margin check, as in the TV row test: every |p_s − p_t| exceeds 2^-10 at the student logits
        // that `logits_row` produces, so no sign flips inside the central difference.
        let margin: i128 = 1 << 22;
        for r in 0..n {
            let zs = logits_row(&h, &w, r, d, v).unwrap();
            let (_, ps) = softmax_row(&zs).unwrap();
            let (_, pt) = softmax_row(&zt[r * v..(r + 1) * v]).unwrap();
            for k in 0..v {
                assert!(
                    (i128::from(ps[k]) - i128::from(pt[k])).abs() > margin,
                    "fixture margin at row {r} k {k}"
                );
            }
        }
        // Derived tolerance, in Q32 ULPs. Per evaluation of the mean loss L, one row's loss is off by
        // at most 2v + 14 (the row bound of the kl and tv tests); the mean keeps that bound. Rounding
        // the logits to Q32 moves each z_k by ≤ ½ ULP, and between the two evaluations the rounding
        // of z_k changes by ≤ 1 ULP, so ΔL moves by ≤ Σ_k |∂ℓ/∂z_k| ≤ v ULP (|∂ℓ/∂z_k| ≤ 1). The
        // mean's own rounding adds ½ per evaluation. So |ΔL| ≤ 2(2v + 14) + 1 + v = 49 ULP, and the
        // quotient multiplies by 2^11. The central-difference truncation is ≤ 43 ULP at |f'''| ≤ 1,
        // and the dz and output roundings are ≤ 13 ULP after the weights |w| ≤ 0.8 and the 1/n.
        // Total ≤ 2^11 · 49 + 128 < 2^17.
        for objective in [Objective::Kl, Objective::Tv] {
            let loss_at =
                |hh: &[i64], ww: &[i64]| head_chunked(hh, ww, &zt, dims, 1, objective).unwrap().0;
            let (_, dh, dw) = head_chunked(&h, &w, &zt, dims, 1, objective).unwrap();
            assert_eq!((dh.len(), dw.len()), (n * d, v * d));
            for (i, &g) in dh.iter().enumerate() {
                let fd = central(&|x: &[i64]| loss_at(x, &w), &h, i);
                let err = (i128::from(g) - fd).abs();
                assert!(err <= HEAD_FD_TOL, "dh[{i}] = {g} fd = {fd} err = {err}");
            }
            for (i, &g) in dw.iter().enumerate() {
                let fd = central(&|x: &[i64]| loss_at(&h, x), &w, i);
                let err = (i128::from(g) - fd).abs();
                assert!(err <= HEAD_FD_TOL, "dw[{i}] = {g} fd = {fd} err = {err}");
            }
        }
    }

    #[test]
    fn head_gradients_are_the_exact_chain_rule() {
        let (h, w, zt, dims) = grad_fixture();
        let (n, d, v) = (dims.n, dims.d, dims.v);
        let n_i = i128::try_from(n).unwrap();
        // The head scales both gradients by 1/n: the divisor is n·ONE, one ONE for the Q32 product.
        let n_scale = n_i * i128::from(ONE);
        for objective in [Objective::Kl, Objective::Tv] {
            let (loss, dh, dw) = head_chunked(&h, &w, &zt, dims, 1, objective).unwrap();
            let mut loss_acc: i128 = 0;
            let mut dz_all: Vec<Vec<i64>> = Vec::new();
            for r in 0..n {
                let zs = logits_row(&h, &w, r, d, v).unwrap();
                let (l, dz) = row_objective(objective, &zs, &zt[r * v..(r + 1) * v]).unwrap();
                loss_acc += i128::from(l);
                dz_all.push(dz);
            }
            assert_eq!(loss, i64::try_from(round_div(loss_acc, n_i)).unwrap());
            // d_hidden[r,j] = round(Σ_k dz[r,k]·w[k,j] / (n·ONE)).
            for r in 0..n {
                for j in 0..d {
                    let acc: i128 = (0..v)
                        .map(|k| i128::from(dz_all[r][k]) * i128::from(w[k * d + j]))
                        .sum();
                    assert_eq!(
                        dh[r * d + j],
                        i64::try_from(round_div(acc, n_scale)).unwrap()
                    );
                }
            }
            // d_w_head[k,j] = round(Σ_r dz[r,k]·hidden[r,j] / (n·ONE)).
            for k in 0..v {
                for j in 0..d {
                    let acc: i128 = (0..n)
                        .map(|r| i128::from(dz_all[r][k]) * i128::from(h[r * d + j]))
                        .sum();
                    assert_eq!(
                        dw[k * d + j],
                        i64::try_from(round_div(acc, n_scale)).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn head_chunked_refusals_are_reached() {
        let (h, w, zt, dims) = hidden_fixture();
        let k = Objective::Kl;
        // Zero chunk and zero dimension.
        assert_eq!(
            head_chunked(&h, &w, &zt, dims, 0, k).unwrap_err(),
            HeadError::Shape
        );
        let zero = Dims { n: 0, d: 3, v: 4 };
        assert_eq!(
            head_chunked(&h, &w, &zt, zero, 1, k).unwrap_err(),
            HeadError::Shape
        );
        // A slice that is not the stated shape.
        assert_eq!(
            head_chunked(&h[1..], &w, &zt, dims, 1, k).unwrap_err(),
            HeadError::Shape
        );
        // A dimension product that overflows usize.
        let huge = Dims {
            n: usize::MAX,
            d: 2,
            v: 1,
        };
        assert_eq!(
            head_chunked(&h, &w, &zt, huge, 1, k).unwrap_err(),
            HeadError::Shape
        );
        // A logit that leaves i64: (2^63−1)² / 2^32 ≈ 2^94.
        assert_eq!(
            head_chunked(
                &[i64::MAX],
                &[i64::MAX],
                &[0],
                Dims { n: 1, d: 1, v: 1 },
                1,
                k
            )
            .unwrap_err(),
            HeadError::Overflow
        );
    }

    #[test]
    fn row_refusals_are_reached() {
        assert_eq!(log_softmax_row(&[]).unwrap_err(), HeadError::Shape);
        assert_eq!(kl_row(&[], &[]).unwrap_err(), HeadError::Shape);
        assert_eq!(tv_row(&[0], &[0, 0]).unwrap_err(), HeadError::Shape);
        // u = MIN − MAX is below i64, so log p leaves i64.
        assert_eq!(
            log_softmax_row(&[i64::MIN, i64::MAX]).unwrap_err(),
            HeadError::Overflow
        );
        assert_eq!(
            kl_row(&[i64::MIN, i64::MAX], &[0, 0]).unwrap_err(),
            HeadError::Overflow
        );
        assert_eq!(
            tv_row(&[i64::MIN, i64::MAX], &[0, 0]).unwrap_err(),
            HeadError::Overflow
        );
    }
}
