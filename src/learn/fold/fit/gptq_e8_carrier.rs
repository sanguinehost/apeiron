//! The integer-carrier boundary of [`super::gptq_e8`]: the feedback factor [`UFac`], the act order
//! [`block_order`], the group scale [`group_scale_int`], the triangular solve [`tri_upper_solve`],
//! the whitened scale projection, the inter-group feedback, and the `f64` → integer conversions of
//! the Hessian ([`h_words`]) and of the weight ([`w_to_rows`]). Everything downstream is exact
//! integer arithmetic, damping included; `f64` remains only at the input boundary and [`fx_to_f64`].
//!
//! [`h_words`] and the whole-weight pass [`w_to_rows`] are matrix-sized and run on the engine pool;
//! every other pass is a serial integer loop. The test module exercises the full-body passes
//! against their one-row forms.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the conversions index row-major [o, n] and [n, n] buffers and move between the \
              Q32.32 words, their binary exponents and the f64 boundary views"
)]

use super::gptq_e8::{GptqE8Error, BLOCK, E8_SCALE_Q32, GROUP};
use crate::engine::chart::pool::{for_chunks, for_chunks2, width_global};
use crate::engine::chart::task::{Task1, Task2};
use crate::learn::solve::chol::SolveError;
use crate::learn::solve::inv::inv_factor_upper;
use crate::learn::solve::q32::{ilog2_pos, round_q32_even, rshift_rne, Fx, Q32Error, SpdQ};
use crate::numeric::fixed_rsqrt::isqrt;

/// The exact signed-integer decomposition `x = m · 2^e` of a finite nonzero `f64`: `m` the 53-bit
/// significand (with sign), `e` the binary exponent. `O(1)`.
pub(super) fn f64_int_exp(x: f64) -> (i128, i32) {
    let bits = x.to_bits();
    let sign: i128 = if bits >> 63 == 0 { 1 } else { -1 };
    let efield = ((bits >> 52) & 0x7ff) as i32;
    let frac = i128::from(bits & 0x000f_ffff_ffff_ffff);
    if efield == 0 {
        (sign * frac, -1074)
    } else {
        (sign * ((1_i128 << 52) | frac), efield - 1023 - 52)
    }
}

/// `|v|` as a `u128` (`i128::MIN` included).
pub(super) fn mag_u128(v: i128) -> u128 {
    v.unsigned_abs()
}

/// `v` as `i64`, or [`GptqE8Error::Overflow`] outside `i64`.
pub(super) fn narrow_q(v: i128) -> Result<i64, GptqE8Error> {
    if v > i128::from(i64::MAX) || v < i128::from(i64::MIN) {
        Err(GptqE8Error::Overflow)
    } else {
        Ok(v as i64)
    }
}

/// The even global exponent `g` that scales a positive magnitude `max_abs` into `[2^29, 2^31)`:
/// `floor(log2 max_abs) − g ∈ {29, 30}` by that floor-log's parity. Even `g` keeps the half-exponent
/// `g/2` an integer for the [`SpdQ`] fold; `0` for a zero or non-finite magnitude. `O(1)`.
pub(super) fn even_g(max_abs: f64) -> i32 {
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return 0;
    }
    let (m, e) = f64_int_exp(max_abs);
    match ilog2_pos(mag_u128(m)) {
        Some(b) => {
            let l = e + b.cast_signed(); // floor(log2 max_abs)
            (l - 29) & !1
        }
        None => 0,
    }
}

/// The Q32.32 word `RNE(x · 2^k)` of a finite `x`, with one round-to-nearest (ties to even) on the
/// significand: the exact Q64.64 integer via [`round_q32_even`] when `e + k ≥ 0`, else
/// [`rshift_rne`].
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] on a NaN or an infinite, [`GptqE8Error::Overflow`] if the rounded word
/// leaves `i64`.
///
/// # Complexity
///
/// `O(1)`.
pub(super) fn rne_scaled_i64(x: f64, k: i32) -> Result<i64, GptqE8Error> {
    if x == 0.0 {
        return Ok(0);
    }
    if !x.is_finite() {
        return Err(GptqE8Error::NonFinite);
    }
    let (m, e) = f64_int_exp(x);
    let sh = e + k;
    if sh >= 0 {
        if sh > 62 {
            return Err(GptqE8Error::Overflow);
        }
        let scale = 1_i128 << (sh + 32);
        let vv = m.checked_mul(scale).ok_or(GptqE8Error::Overflow)?;
        round_q32_even(vv).map_err(|_| GptqE8Error::Overflow)
    } else {
        match rshift_rne(m, (-sh).cast_unsigned()) {
            Some(q) => narrow_q(q),
            None => Err(GptqE8Error::Overflow),
        }
    }
}

/// Pool task context for [`to_words`]: the source, the scale exponent and the entries per task.
struct WordsTask<'a> {
    x: &'a [f64],
    k: i32,
    per: usize,
}

impl Task1<i64, GptqE8Error> for WordsTask<'_> {
    /// The rounding of [`to_words`] over this task's entries.
    fn run(&self, t: usize, out: &mut [i64]) -> Result<(), GptqE8Error> {
        let lo = t * self.per;
        for (j, w) in out.iter_mut().enumerate() {
            *w = rne_scaled_i64(self.x[lo + j], self.k)?;
        }
        Ok(())
    }
}

/// The Q32.32 words of `x · 2^-g` through [`rne_scaled_i64`], split across the pool by entry range;
/// each entry reads only itself.
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] or [`GptqE8Error::Overflow`] from [`rne_scaled_i64`].
///
/// # Complexity
///
/// Time `O(len)`: `len` word ops per call, two calls per fit ([`h_words`] from [`feedback_u`] and
/// from [`block_order`]); space `O(len)`.
pub(super) fn to_words(x: &[f64], g: i32) -> Result<Vec<i64>, GptqE8Error> {
    let mut out = vec![0i64; x.len()];
    let per = x.len().div_ceil(width_global()).max(1);
    for_chunks(&mut out, per, &WordsTask { x, k: 32 - g, per })?;
    Ok(out)
}

/// The Q32.32 words of the row-major `[n, n]` `h` at the even global exponent [`even_g`] of its
/// largest entry, and that exponent: `h[i][j] = words[i·n + j] · 2^(g − 32)` up to the one rounding
/// of each entry. The **only** conversion of the Hessian.
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] on a non-finite entry, [`GptqE8Error::Overflow`] from [`to_words`].
///
/// # Complexity
///
/// Time `O(len)` (one maximum scan, one pooled conversion pass), one call per fit; space `O(len)`.
pub(super) fn h_words(h: &[f64]) -> Result<(Vec<i64>, i32), GptqE8Error> {
    let mut max_abs = 0.0f64;
    for &x in h {
        if !x.is_finite() {
            return Err(GptqE8Error::NonFinite);
        }
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let g = even_g(max_abs);
    Ok((to_words(h, g)?, g))
}

/// [`Q32Error`] as a [`GptqE8Error`]: an asymmetric input is reported as
/// [`GptqE8Error::ShapeMismatch`] (the Hessian contract is symmetric), a non-positive diagonal as
/// [`GptqE8Error::NotPositiveDefinite`].
pub(super) fn map_q32(e: Q32Error) -> GptqE8Error {
    match e {
        Q32Error::ShapeMismatch | Q32Error::Asymmetric => GptqE8Error::ShapeMismatch,
        Q32Error::NotSpd => GptqE8Error::NotPositiveDefinite,
        Q32Error::Overflow => GptqE8Error::Overflow,
    }
}

/// [`SolveError`] as a [`GptqE8Error`]: both non-positive cases as
/// [`GptqE8Error::NotPositiveDefinite`], a grid-range refusal as [`GptqE8Error::Overflow`].
pub(super) fn map_inv(e: SolveError) -> GptqE8Error {
    match e {
        SolveError::ShapeMismatch => GptqE8Error::ShapeMismatch,
        SolveError::Empty => GptqE8Error::ZeroDim,
        SolveError::NonPositiveDiagonal | SolveError::NotPositiveDefinite => {
            GptqE8Error::NotPositiveDefinite
        }
        SolveError::Overflow => GptqE8Error::Overflow,
    }
}

/// The inverse feedback factor `U` of the damped Hessian as an equilibrated integer carrier:
/// `U = Uhat · diag(2^-d)`, so the real `U` satisfies `UᵀU = H⁻¹`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UFac {
    /// Row-major `[n, n]` Q32.32 upper-triangular factor `Uhat` (zero below the diagonal).
    pub hat: Vec<i64>,
    /// The per-column base-2 exponents `d`: column `j` is scaled by `2^-d_j`.
    pub d: Vec<i32>,
    /// The matrix order.
    pub n: usize,
}

/// The inverse feedback factor `U` (`UᵀU = H⁻¹`) of the damped Hessian
/// `H = h + damp·mean(diag(h))·I`, a zero damped diagonal entry replaced by `1.0`.
///
/// The damping is integer: [`h_words`] converts `h` once, the diagonal mean is an exact `i128` word
/// sum divided once by `n`, and `damp`'s product with it is rounded once by [`round_q32_even`]. The
/// damped `H` is mirrored from its lower triangle and equilibrated by [`SpdQ::from_q32`];
/// [`inv_factor_upper`] returns `Uhat`, and with `d = spdq.d + g/2`, `U = Uhat·diag(2^-d)`.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] if `h.len() != n·n`; [`GptqE8Error::ZeroDim`] if `n == 0`;
/// [`GptqE8Error::NonFinite`] on a non-finite `h` or `damp`; [`GptqE8Error::NotPositiveDefinite`]
/// on a non-positive damped diagonal or `Cholesky` pivot; [`GptqE8Error::Overflow`] on a checked
/// `i128`/`i64` operation or a `damp` outside `|damp| < 2^31`.
///
/// # Complexity
///
/// Temporal `O(n³)` `i128` multiply-adds (factor `n³/3` plus inverse `n³/6`, damping and mirror
/// `O(n²)`); spatial `O(n²)` words.
pub fn feedback_u(h: &[f64], n: usize, damp: f64) -> Result<UFac, GptqE8Error> {
    if n == 0 {
        return Err(GptqE8Error::ZeroDim);
    }
    if h.len() != n * n {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let (w, g) = h_words(h)?;
    // `mean(diag) · 2^(32 − g)` as one word: the exact `i128` sum of the diagonal words, rounded
    // once by the division.
    let mut sum: i128 = 0;
    for i in 0..n {
        sum += i128::from(w[i * n + i]);
    }
    let nn = i128::try_from(n).map_err(|_| GptqE8Error::Overflow)?;
    let mean = div_rne_i128(sum, nn).ok_or(GptqE8Error::Overflow)?;
    let damp_w = i128::from(rne_scaled_i64(damp, 32)?);
    let dlt =
        round_q32_even(mean.checked_mul(damp_w).ok_or(GptqE8Error::Overflow)?).map_err(map_q32)?;
    let one = rne_scaled_i64(1.0, 32 - g)?;
    let mut sym = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            sym[i * n + j] = if j > i { w[j * n + i] } else { w[i * n + j] };
        }
        let v = w[i * n + i].checked_add(dlt).ok_or(GptqE8Error::Overflow)?;
        sym[i * n + i] = if v == 0 { one } else { v };
    }
    let spdq = SpdQ::from_q32(&sym, n).map_err(map_q32)?;
    let hat = inv_factor_upper(&spdq.hat, n).map_err(map_inv)?;
    let half = g / 2;
    let mut d = vec![0i32; n];
    for (di, &d0) in d.iter_mut().zip(&spdq.d) {
        *di = d0.checked_add(half).ok_or(GptqE8Error::Overflow)?;
    }
    Ok(UFac { hat, d, n })
}

/// Solve `x·U_blk = e` in place for the group's upper-triangular block `U_blk` of the carrier: with
/// `U_kj = hat[(g0+k)·n + g0+j]·2^-32·2^-d_j` the back substitution is
/// `x_j = (e_j − Σ_{k>j} x_k·U_kj)/U_jj` for `j` descending, aligned at the row's common exponent
/// with one round-to-nearest quotient of the checked `i128` accumulation.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] if `e.len() != gs`; [`GptqE8Error::NotPositiveDefinite`] on a
/// non-positive block diagonal; [`GptqE8Error::Overflow`] on a checked `i128` operation.
///
/// # Complexity
///
/// Temporal `O(gs²)` checked `i128` multiply-adds per call, spatial `O(1)` beyond the row.
pub(super) fn tri_upper_solve(
    u: &UFac,
    n: usize,
    g0: usize,
    gs: usize,
    e: &mut [Fx],
) -> Result<(), GptqE8Error> {
    if e.len() != gs || g0 + gs > n || g0 + gs > u.n {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let mut j = gs;
    while j > 0 {
        j -= 1;
        let dj = u.d[g0 + j];
        // The common exponent of the contraction terms `x_k·U_kj` (zero factors are skipped).
        let mut emin = e[j].e;
        for k in (j + 1)..gs {
            let h = u.hat[(g0 + k) * n + g0 + j];
            if h != 0 && e[k].m != 0 {
                emin = emin.min(e[k].e - 32 - dj);
            }
        }
        let mut acc =
            shift_left_i128(i128::from(e[j].m), e[j].e - emin).ok_or(GptqE8Error::Overflow)?;
        for k in (j + 1)..gs {
            let h = u.hat[(g0 + k) * n + g0 + j];
            if h == 0 || e[k].m == 0 {
                continue;
            }
            let te = e[k].e - 32 - dj;
            let pr = i128::from(e[k].m)
                .checked_mul(i128::from(h))
                .ok_or(GptqE8Error::Overflow)?;
            let pr = shift_left_i128(pr, te - emin).ok_or(GptqE8Error::Overflow)?;
            acc = acc.checked_sub(pr).ok_or(GptqE8Error::Overflow)?;
        }
        let hjj = u.hat[(g0 + j) * n + g0 + j];
        if hjj <= 0 {
            return Err(GptqE8Error::NotPositiveDefinite);
        }
        // `x_j = acc·2^(emin + 32 + d_j)/hjj`; the quotient is scaled to at most 62 bits so the one
        // division is the only rounding.
        let s = i64::from(emin) + 32 + i64::from(dj);
        let est = blen(acc) - blen(i128::from(hjj)) + i32::try_from(s).unwrap_or(i32::MAX);
        let shift = est - 61;
        let q = div_shift_rne(acc, s - i64::from(shift), i128::from(hjj))?;
        e[j] = Fx {
            m: narrow_q(q)?,
            e: shift,
        };
    }
    Ok(())
}

/// The exact power of two `2^e` as an `f64` (the exponent field), or `0.0`/`INFINITY` outside the
/// finite range.
pub(super) fn pow2i(e: i32) -> f64 {
    if (-1022..=1023).contains(&e) {
        2.0f64.powi(e)
    } else if e < -1022 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// Pool task context for [`w_to_rows`]: the `f64` weight, the row length and the rows per task.
struct RowsTask<'a> {
    w: &'a [f64],
    n: usize,
    per: usize,
}

impl Task2<i64, i32, GptqE8Error> for RowsTask<'_> {
    /// The block floating point of [`w_to_rows`] over this task's rows.
    fn run(&self, t: usize, mant: &mut [i64], e_row: &mut [i32]) -> Result<(), GptqE8Error> {
        let n = self.n;
        for (r, er_slot) in e_row.iter_mut().enumerate() {
            let r0 = t * self.per + r;
            let row = &self.w[r0 * n..(r0 + 1) * n];
            let mut max_abs = 0.0f64;
            for &x in row {
                if !x.is_finite() {
                    return Err(GptqE8Error::NonFinite);
                }
                let a = x.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            if max_abs == 0.0 {
                continue;
            }
            let (m, e) = f64_int_exp(max_abs);
            let bl = match ilog2_pos(mag_u128(m)) {
                Some(b) => b + 1,
                None => return Err(GptqE8Error::Overflow),
            };
            let l = e + bl.cast_signed() - 1;
            let er = l - 34;
            *er_slot = er;
            for j in 0..n {
                mant[r * n + j] = rne_scaled_i64(row[j], -er)?;
            }
        }
        Ok(())
    }
}

/// The per-row block floating point of the weight `w` (row-major `[o, n]`): mantissas `mant` and
/// per-row exponents `e_row` with `max |mant[r]| ∈ [2^34, 2^35)` (all-zero row: zeros, `e = 0`), so
/// `w[r][j] = mant[r·n+j]·2^e_row[r]` up to the rounding. The **only** conversion of the weight.
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`], or [`GptqE8Error::Overflow`] if a mantissa leaves `i64`.
///
/// # Complexity
///
/// Time `O(o·n)`: `o·n` word ops per call, one call per fit; space `O(o·n)` plus `O(o)`.
pub(super) fn w_to_rows(
    w: &[f64],
    o: usize,
    n: usize,
) -> Result<(Vec<i64>, Vec<i32>), GptqE8Error> {
    let mut mant = vec![0i64; o * n];
    let mut e_row = vec![0i32; o];
    let per = o.div_ceil(width_global()).max(1);
    for_chunks2(&mut mant, per * n, &mut e_row, per, &RowsTask { w, n, per })?;
    Ok((mant, e_row))
}

/// The renormalisation precision of a row that enters an inner product, a sum of squares or the
/// inter-group contraction: 56 significant bits, so a product of two mantissas is below `2^114` and
/// a 128-term sum stays below `2^121` inside `i128` with no exponent-alignment shift.
pub(super) const ROW_BITS: i32 = 56;

/// The coarsest exponent `e` at which every non-zero entry of `x` keeps at most [`ROW_BITS`]
/// significand bits: `e = max_j(e_j + ⌈log2|m_j|⌉) − ROW_BITS`; `0` for an all-zero slice.
/// `O(len)`.
pub(super) fn row_exp(x: &[Fx]) -> i32 {
    let mut e = i32::MIN;
    for v in x {
        if v.m != 0 {
            e = e.max(v.e + blen(i128::from(v.m)) - ROW_BITS);
        }
    }
    if e == i32::MIN {
        0
    } else {
        e
    }
}

/// The mantissas `m_j = RNE(x_j·2^-e)` of `x` on the row exponent `e`, each
/// `|m_j| ≤ 2^(ROW_BITS+1)`. `O(len)`, `O(len)` output.
pub(super) fn renorm_row(x: &[Fx], e: i32) -> Result<Vec<i128>, GptqE8Error> {
    let mut out = Vec::with_capacity(x.len());
    for v in x {
        out.push(round_at(i128::from(v.m), i64::from(v.e - e))?);
    }
    Ok(out)
}

/// One row of the whitened scale projection of the `wscale` branch: `u_s = S·U_blk⁻¹` by
/// [`tri_upper_solve`], then `a_new = max(a + ⟨e, u_s⟩/⟨u_s, u_s⟩, floor)` and
/// `e_new = e − (a_new − a)·u_s`. Both inner products are exact `i128` sums at one [`ROW_BITS`]-bit
/// exponent per operand, their ratio is one [`div_rne_i128`], and `e_new` is written to `e_out` with
/// one rounding per entry. `floor` is the smallest positive [`Fx`] at the scale's exponent
/// (`A_CLAMP` when `a` is zero); a zero `⟨u_s,u_s⟩` returns `a` unchanged with `e_new = e`.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] or [`GptqE8Error::Overflow`].
///
/// # Complexity
///
/// Temporal `O(gs²)` `i128` multiply-adds; spatial `O(gs)` [`Fx`].
#[allow(
    clippy::too_many_arguments,
    reason = "one buffer and geometry value per parameter"
)]
pub(super) fn project_scale_row(
    u: &UFac,
    n: usize,
    g0: usize,
    gs: usize,
    s_row: &[Fx],
    e_row: &[Fx],
    a: &Fx,
    e_out: &mut [Fx],
) -> Result<Fx, GptqE8Error> {
    if s_row.len() != gs || e_row.len() != gs || e_out.len() != gs {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let mut us = s_row.to_vec();
    tri_upper_solve(u, n, g0, gs, &mut us)?;
    // Both operands of the inner products on one row exponent each: exact `i128` sums.
    let e_us = row_exp(&us);
    let um = renorm_row(&us, e_us)?;
    let e_e = row_exp(e_row);
    let em = renorm_row(e_row, e_e)?;
    let mut num: i128 = 0;
    let mut den: i128 = 0;
    for j in 0..gs {
        let pr = em[j].checked_mul(um[j]).ok_or(GptqE8Error::Overflow)?;
        num = num.checked_add(pr).ok_or(GptqE8Error::Overflow)?;
        let sq = um[j].checked_mul(um[j]).ok_or(GptqE8Error::Overflow)?;
        den = den.checked_add(sq).ok_or(GptqE8Error::Overflow)?;
    }
    let floor = if a.m > 0 {
        Fx { m: 1, e: a.e }
    } else {
        A_CLAMP
    };
    let an = if den == 0 || num == 0 {
        a.clone()
    } else {
        // `delta = (num/den)·2^(e_e − e_us)`: each operand is normalised to 62 bits (one RNE)
        // so the one division's numerator `n·2^61` stays inside `i128`, then one RNE on the
        // quotient at 61 bits and the exact exponent `sigma`.
        let bn = blen(num);
        let bd = blen(den);
        let n62 = round_at(num, i64::from(62 - bn))?;
        let d62 = round_at(den, i64::from(62 - bd))?;
        let num_q = shift_left_i128(n62, 61).ok_or(GptqE8Error::Overflow)?;
        let q = div_rne_i128(num_q, d62).ok_or(GptqE8Error::Overflow)?;
        let sigma = i64::from(e_e) - i64::from(e_us) + i64::from(bn) - i64::from(bd) - 61;
        if q == 0 {
            a.clone()
        } else {
            let dlt = Fx::from_i128(q, i32::try_from(sigma).map_err(|_| GptqE8Error::Overflow)?)
                .map_err(|_| GptqE8Error::Overflow)?;
            let cand = fx_add(a, &dlt)?;
            if cand.m <= 0 || fx_lt(&cand, &floor) {
                floor
            } else {
                cand
            }
        }
    };
    let dlt = fx_sub(&an, a)?;
    // `e_new = e − dlt·u_s`, exact per entry, then one rounding to the output row exponent.
    let mut ve = vec![0i128; gs];
    let mut ex = vec![0i32; gs];
    for j in 0..gs {
        let pe = dlt.e + e_us;
        let p = i128::from(dlt.m)
            .checked_mul(um[j])
            .ok_or(GptqE8Error::Overflow)?;
        let e0 = e_e.min(pe);
        let ea = shift_left_i128(em[j], e_e - e0).ok_or(GptqE8Error::Overflow)?;
        let eb = shift_left_i128(p, pe - e0).ok_or(GptqE8Error::Overflow)?;
        ve[j] = ea.checked_sub(eb).ok_or(GptqE8Error::Overflow)?;
        ex[j] = e0;
    }
    let mut eout = i32::MIN;
    for j in 0..gs {
        if ve[j] != 0 {
            eout = eout.max(ex[j] + blen(ve[j]) - ROW_BITS);
        }
    }
    let eout = if eout == i32::MIN { 0 } else { eout };
    for j in 0..gs {
        let m = if ve[j] == 0 {
            0
        } else {
            narrow_q(round_at(ve[j], i64::from(ex[j] - eout))?)?
        };
        e_out[j] = Fx { m, e: eout };
    }
    Ok(an)
}

/// The whitened scale projection of the `wscale` branch over all `o` rows: [`project_scale_row`] per
/// row, serially.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] or [`GptqE8Error::Overflow`] on a checked `i128` operation.
///
/// # Complexity
///
/// Temporal `O(o·gs²)` `i128` multiply-adds; spatial `O(o·gs)` [`Fx`].
pub(super) fn project_scale_int(
    u: &UFac,
    n: usize,
    g0: usize,
    gs: usize,
    s_cur: &[Fx],
    e_cur: &[Fx],
    a: &[Fx],
) -> Result<(Vec<Fx>, Vec<Fx>), GptqE8Error> {
    let o = a.len();
    if s_cur.len() != o * gs || e_cur.len() != o * gs {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let mut a_new = vec![Fx { m: 0, e: 0 }; o];
    let mut e_new = vec![Fx { m: 0, e: 0 }; o * gs];
    for i in 0..o {
        a_new[i] = project_scale_row(
            u,
            n,
            g0,
            gs,
            &s_cur[i * gs..(i + 1) * gs],
            &e_cur[i * gs..(i + 1) * gs],
            &a[i],
            &mut e_new[i * gs..(i + 1) * gs],
        )?;
    }
    Ok((a_new, e_new))
}

/// `wk[:, g1:] -= E·U[g0:g1, g1:]` on the integer working rows: `e` is the group's `Fx` residual
/// (`o × gs`), `wm` the row-major `o × n` mantissas at the per-row exponents `e_row` and
/// `U_jc = hat[(g0+j)·n + c]·2^-32·2^-d_c`. Per stored entry the `gs` products are one exact `i128`
/// accumulation and the update is **one** round-to-nearest to the row's grid `2^e_row`, with the
/// row exponent rising by one (all mantissas halved with RNE) only at `|m| < 2^62`.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] or [`GptqE8Error::Overflow`].
///
/// # Complexity
///
/// Temporal `O(o·gs·(n − g1))` checked `i128` multiply-adds; spatial `O(1)` beyond the row.
#[allow(
    clippy::too_many_arguments,
    reason = "one distinct buffer and geometry value per parameter; the call sites name them"
)]
pub(super) fn inter_group_feedback(
    wm: &mut [i64],
    e_row: &mut [i32],
    e: &[Fx],
    u: &UFac,
    n: usize,
    g0: usize,
    gs: usize,
) -> Result<(), GptqE8Error> {
    let g1 = g0 + gs;
    if g1 > n
        || n == 0
        || !wm.len().is_multiple_of(n)
        || e_row.len() != wm.len() / n
        || e.len() != e_row.len() * gs
    {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let o = e_row.len();
    let tail = n - g1;
    for r in 0..o {
        // The shared exponent of this row's residual terms.
        let mut emin = i32::MAX;
        for j in 0..gs {
            let x = &e[r * gs + j];
            if x.m != 0 {
                emin = emin.min(x.e);
            }
        }
        let emin = if emin == i32::MAX { 0 } else { emin };
        for t in 0..tail {
            let c = g1 + t;
            let dc = u.d[c];
            let mut sum: i128 = 0;
            for j in 0..gs {
                let h = u.hat[(g0 + j) * n + c];
                let x = &e[r * gs + j];
                if h == 0 || x.m == 0 {
                    continue;
                }
                let pr = i128::from(x.m)
                    .checked_mul(i128::from(h))
                    .ok_or(GptqE8Error::Overflow)?;
                let term = shift_left_i128(pr, x.e - emin).ok_or(GptqE8Error::Overflow)?;
                sum = sum.checked_add(term).ok_or(GptqE8Error::Overflow)?;
            }
            // `Δw = −sum·2^(emin − 32 − d_c)`, one RNE to the row grid.
            let mut sh = i64::from(emin) - 32 - i64::from(dc) - i64::from(e_row[r]);
            let mut v = i128::from(wm[r * n + c])
                .checked_add(round_at(
                    sum.checked_neg().ok_or(GptqE8Error::Overflow)?,
                    sh,
                )?)
                .ok_or(GptqE8Error::Overflow)?;
            if blen(v) > 62 {
                // The headroom rule: halve the whole row (RNE) and re-round this entry.
                for q in 0..n {
                    let cur = i128::from(wm[r * n + q]);
                    let half = rshift_rne(cur, 1).ok_or(GptqE8Error::Overflow)?;
                    wm[r * n + q] = narrow_q(half)?;
                }
                e_row[r] = e_row[r].checked_add(1).ok_or(GptqE8Error::Overflow)?;
                sh -= 1;
                v = i128::from(wm[r * n + c])
                    .checked_add(round_at(
                        sum.checked_neg().ok_or(GptqE8Error::Overflow)?,
                        sh,
                    )?)
                    .ok_or(GptqE8Error::Overflow)?;
            }
            wm[r * n + c] = narrow_q(v)?;
        }
    }
    Ok(())
}

/// Signed `round(v / d)` for `d > 0`, nearest with ties away from zero, or `None` if the result or
/// an intermediate leaves `i128`.
pub(super) fn div_rne_i128(v: i128, d: i128) -> Option<i128> {
    if d <= 0 {
        return None;
    }
    if v >= 0 {
        v.checked_add(d / 2).map(|n| n / d)
    } else {
        v.checked_sub(d / 2).map(|n| n / d)
    }
}

/// `RNE(v · 2^k)`: an exact checked left shift for `k ≥ 0`, a round-to-nearest ties-to-even right
/// shift for `k < 0`, or `None` when a checked operation leaves `i128`.
pub(super) fn shift_rne(v: i128, k: i32) -> Option<i128> {
    if k >= 0 {
        if k > 62 {
            return None;
        }
        v.checked_mul(1_i128 << k)
    } else {
        rshift_rne(v, (-k).cast_unsigned())
    }
}

/// The magnitude class `⌈log2(|v| + 1)⌉` of an `i128`: `0` for `0`, else `ilog2_pos(|v|) + 1`, so
/// `2^(b − 1) ≤ |v| < 2^b`.
///
/// # Complexity
///
/// `O(1)`.
pub(super) fn blen(v: i128) -> i32 {
    if v == 0 {
        0
    } else {
        match ilog2_pos(mag_u128(v)) {
            Some(b) => b.cast_signed() + 1,
            None => 0,
        }
    }
}

/// `v · 2^k` for `k ≥ 0` in `i128`, `Some(0)` for `v == 0`, `None` when the product leaves `i128`.
///
/// # Complexity
///
/// `O(1)`.
pub(super) fn shift_left_i128(v: i128, k: i32) -> Option<i128> {
    if v == 0 {
        return Some(0);
    }
    if !(0..127).contains(&k) {
        return None;
    }
    v.checked_mul(1_i128 << k)
}

/// `RNE(v · 2^k)` with the shift taken as `i64`: the [`shift_rne`] rule with an overflow check on
/// the cast. `O(1)`.
fn round_at(v: i128, k: i64) -> Result<i128, GptqE8Error> {
    if k >= 0 {
        shift_left_i128(v, i32::try_from(k).map_err(|_| GptqE8Error::Overflow)?)
            .ok_or(GptqE8Error::Overflow)
    } else {
        rshift_rne(v, u32::try_from(-k).map_err(|_| GptqE8Error::Overflow)?)
            .ok_or(GptqE8Error::Overflow)
    }
}

/// `RNE(v · 2^s / d)` for `d > 0`: an exact checked numerator shift when `s ≥ 0`, the exact
/// denominator `d · 2^-s` when `s < 0`, and one round-to-nearest quotient ([`div_rne_i128`]).
///
/// # Errors
///
/// [`GptqE8Error::Overflow`] on a shift or quotient that leaves `i128`.
///
/// # Complexity
///
/// `O(1)`.
pub(super) fn div_shift_rne(v: i128, s: i64, d: i128) -> Result<i128, GptqE8Error> {
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

/// `x + y` as an [`Fx`], aligned at the finer exponent with one [`Fx::from_i128`]. `O(1)`.
fn fx_add(x: &Fx, y: &Fx) -> Result<Fx, GptqE8Error> {
    let e = x.e.min(y.e);
    let a = shift_left_i128(i128::from(x.m), x.e - e).ok_or(GptqE8Error::Overflow)?;
    let b = shift_left_i128(i128::from(y.m), y.e - e).ok_or(GptqE8Error::Overflow)?;
    Fx::from_i128(a.checked_add(b).ok_or(GptqE8Error::Overflow)?, e)
        .map_err(|_| GptqE8Error::Overflow)
}

/// `x − y` as an [`Fx`], aligned at the finer exponent with one [`Fx::from_i128`]. `O(1)`.
fn fx_sub(x: &Fx, y: &Fx) -> Result<Fx, GptqE8Error> {
    let e = x.e.min(y.e);
    let a = shift_left_i128(i128::from(x.m), x.e - e).ok_or(GptqE8Error::Overflow)?;
    let b = shift_left_i128(i128::from(y.m), y.e - e).ok_or(GptqE8Error::Overflow)?;
    Fx::from_i128(a.checked_sub(b).ok_or(GptqE8Error::Overflow)?, e)
        .map_err(|_| GptqE8Error::Overflow)
}

/// `x < y` exactly for two positive [`Fx`], compared at the finer exponent with the magnitude class
/// deciding only when both aligned values leave `i128`. `O(1)`.
fn fx_lt(x: &Fx, y: &Fx) -> bool {
    let e = x.e.min(y.e);
    match (
        shift_left_i128(i128::from(x.m), x.e - e),
        shift_left_i128(i128::from(y.m), y.e - e),
    ) {
        (Some(a), Some(b)) => a < b,
        (None, Some(_)) => false,
        (Some(_), None) => true,
        (None, None) => x.e < y.e,
    }
}

/// The shared exponent [`sum_sq_at`] needs: the smallest over the non-zero entries, or `0`. `O(Σ len)`.
pub(super) fn min_exp(rows: &[&[Fx]]) -> i32 {
    let mut e = i32::MAX;
    for row in rows {
        for x in *row {
            if x.m != 0 && x.e < e {
                e = x.e;
            }
        }
    }
    if e == i32::MAX {
        0
    } else {
        e
    }
}

/// `Σ m_i²·2^(2(e_i − e0))` for a shared exponent `e0 ≤ min_i e_i`, as an exact checked `i128`
/// accumulation (`Σ v_i² · 2^(-2 e0)` of the represented values).
///
/// # Errors
///
/// [`GptqE8Error::Overflow`] when an aligned square or the sum leaves `i128`
/// (`2·(e_i − e0) + 2·blen(m_i) ≤ 126` per term).
///
/// # Complexity
///
/// `O(len)` checked `i128` multiply-adds, `O(1)` space.
pub(super) fn sum_sq_at(e: &[Fx], e0: i32) -> Result<i128, GptqE8Error> {
    let mut acc: i128 = 0;
    for x in e {
        if x.m == 0 {
            continue;
        }
        let m2 = i128::from(x.m)
            .checked_mul(i128::from(x.m))
            .ok_or(GptqE8Error::Overflow)?;
        let t = shift_left_i128(m2, 2 * (x.e - e0)).ok_or(GptqE8Error::Overflow)?;
        acc = acc.checked_add(t).ok_or(GptqE8Error::Overflow)?;
    }
    Ok(acc)
}

/// The real value of an [`Fx`] as an `f64`: the significand rounds once to 53 bits (RNE) and the
/// power of two is exact.
pub(super) fn fx_to_f64(x: &Fx) -> f64 {
    if x.m == 0 {
        return 0.0;
    }
    let bits = match ilog2_pos(mag_u128(i128::from(x.m))) {
        Some(b) => b + 1,
        None => return 0.0,
    };
    let s = bits.saturating_sub(53);
    let (m, e) = if s == 0 {
        (x.m, x.e)
    } else {
        match rshift_rne(i128::from(x.m), s) {
            Some(v) => (v as i64, x.e + s.cast_signed()),
            None => (x.m, x.e),
        }
    };
    // `|m| ≤ 2^53` here, so `m as f64` is exact and the product is exact (the exponent is in range
    // for every scale this kernel produces; an out-of-range one is the `pow2i` 0/INFINITY edge).
    debug_assert!(m.unsigned_abs() <= 1u64 << 53);
    m as f64 * pow2i(e)
}

/// `x < 2^k` for a positive-mantissa [`Fx`] (used by the group scale's lower clamp).
pub(super) fn fx_below_pow2(x: &Fx, k: i32) -> bool {
    if x.m <= 0 {
        return true;
    }
    let t = k - x.e;
    if t >= 63 {
        return true;
    }
    if t < 0 {
        return false;
    }
    x.m < (1_i64 << t)
}

/// The `O(1)` lower clamp of the group scale: the f64 `a = max(a, 1e-12)` is carried as one
/// Q32.32 unit's worth of mantissa (`Fx { m: 1, e: -40 }`, `2^-40 ≈ 9.1e-13`).
pub(super) const A_CLAMP: Fx = Fx { m: 1, e: -40 };

/// The integer group scale `a = 0.828·w_rms` of the working row: `prec_j = 2^(2 d_j)/Uhat_jj²` are
/// the inverse-variance weights, `w_rms² = Σ_j m_j²·prec_j / Σ_j prec_j` is an `i128` accumulation
/// aligned to `2 d_j`, `sqrt` by [`isqrt`] and the `0.828` by [`E8_SCALE_Q32`]; the result's
/// exponent carries `e_row`.
///
/// # Errors
///
/// [`GptqE8Error::NotPositiveDefinite`] on a non-positive diagonal, or [`GptqE8Error::Overflow`].
///
/// # Complexity
///
/// `O(gs)` checked `i128` operations, `O(1)` space.
pub(super) fn group_scale_int(
    mant_row: &[i64],
    e_row: i32,
    u: &UFac,
    g0: usize,
) -> Result<Fx, GptqE8Error> {
    let gs = mant_row.len();
    let n = u.n;
    let mut num: i128 = 0;
    let mut den: i128 = 0;
    for j in 0..gs {
        let ujj = u.hat[(g0 + j) * n + g0 + j];
        if ujj <= 0 {
            return Err(GptqE8Error::NotPositiveDefinite);
        }
        let ujj2 = i128::from(ujj) * i128::from(ujj);
        let Some(m_j) = div_rne_i128(1_i128 << 96, ujj2) else {
            return Err(GptqE8Error::Overflow);
        };
        let Some(z) = shift_rne(m_j, 2 * u.d[g0 + j]) else {
            return Err(GptqE8Error::Overflow);
        };
        if z <= 0 {
            continue;
        }
        let m2 = i128::from(mant_row[j]) * i128::from(mant_row[j]);
        let Some(term) = m2.checked_mul(z) else {
            return Err(GptqE8Error::Overflow);
        };
        num = match num.checked_add(term) {
            Some(v) => v,
            None => return Err(GptqE8Error::Overflow),
        };
        den = match den.checked_add(z) {
            Some(v) => v,
            None => return Err(GptqE8Error::Overflow),
        };
    }
    if num <= 0 || den <= 0 {
        return Ok(A_CLAMP);
    }
    let Some(bl_num) = ilog2_pos(num.cast_unsigned()) else {
        return Err(GptqE8Error::Overflow);
    };
    let bl_num = bl_num.cast_signed();
    // `num << shift2` stays inside `i128` (the shifted magnitude is below 2^125) and `shift2` is
    // even, so `sqrt((num << shift2)/den)` halves exactly.
    let mut shift2 = 124 - bl_num;
    if shift2 < 0 {
        shift2 = 0;
    }
    shift2 -= shift2 % 2;
    let Some(scaled) = num.checked_mul(1_i128 << shift2) else {
        return Err(GptqE8Error::Overflow);
    };
    let q = scaled / den;
    if q <= 0 {
        return Ok(A_CLAMP);
    }
    let s = isqrt(q);
    let fxm = i128::from(E8_SCALE_Q32) * s;
    let fxe = e_row - shift2 / 2 - 32;
    let a = Fx::from_i128(fxm, fxe).map_err(|_| GptqE8Error::Overflow)?;
    if fx_below_pow2(&a, -40) {
        return Ok(A_CLAMP);
    }
    Ok(a)
}

/// The act-order permutation of a Hessian diagonal, at 8-block granularity: groups in descending
/// order of their diagonal sum, and inside a group the 8-blocks in descending order of their
/// diagonal sum, ties by ascending index. Columns inside an 8-block keep their order. The block
/// sums are exact `i128` word sums of the [`h_words`] conversion.
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] or [`GptqE8Error::Overflow`] from [`h_words`].
///
/// # Complexity
///
/// `O(n² + nb·log nb + ng·log ng)`; space `O(n² + nb + ng)`.
pub(super) fn block_order(h: &[f64], n: usize) -> Result<(Vec<usize>, Vec<usize>), GptqE8Error> {
    let nb = n / BLOCK;
    let per_group = GROUP / BLOCK;
    let ng = n.div_ceil(GROUP);
    let (words, _) = h_words(h)?;
    let mut bt = vec![0i128; nb];
    for (b, s) in bt.iter_mut().enumerate() {
        for k in 0..BLOCK {
            let i = b * BLOCK + k;
            *s += i128::from(words[i * n + i]);
        }
    }
    let mut gt = vec![0i128; ng];
    for b in 0..nb {
        gt[b / per_group] += bt[b];
    }
    let mut groups: Vec<usize> = (0..ng).collect();
    groups.sort_by_key(|&g| (core::cmp::Reverse(gt[g]), g));
    let mut blk_perm = vec![0usize; nb];
    for (gp, &g) in groups.iter().enumerate() {
        let s0 = g * per_group;
        let s1 = ((g + 1) * per_group).min(nb);
        let mut sub: Vec<usize> = (s0..s1).collect();
        sub.sort_by_key(|&b| (core::cmp::Reverse(bt[b]), b));
        for (t, &b) in sub.iter().enumerate() {
            blk_perm[gp * per_group + t] = b;
        }
    }
    let mut inv_blk = vec![0usize; nb];
    for (p, &b) in blk_perm.iter().enumerate() {
        inv_blk[b] = p;
    }
    Ok((blk_perm, inv_blk))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        reason = "the tests compare exact grid images and the LCG maps a shifted `u64` state to the \
                  `i64` row words"
    )]

    use super::super::gptq_e8::{beam_pass, second_pass_wins, select_k, BLOCK};
    use super::{
        feedback_u, fx_to_f64, group_scale_int, inter_group_feedback, tri_upper_solve, w_to_rows,
        Fx, UFac,
    };

    /// `len` LCG values in `[-1, 1)` from `seed`.
    fn lcg(len: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    /// The full-row beam pass equals one call per row: rows are independent and nothing is reduced
    /// across them. Covers `wscale` on and off.
    #[test]
    fn full_body_equals_the_per_row_body() {
        let (o, n) = (17usize, 32usize);
        let gb = n / BLOCK;
        let buf = |rows: usize| {
            (
                vec![Fx { m: 0, e: 0 }; rows * n],
                vec![Fx { m: 0, e: 0 }; rows * n],
                vec![0u16; rows * gb],
            )
        };
        let mut h = vec![0.0f64; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        let ufac = feedback_u(&h, n, 0.01).expect("SPD factor");
        let w = lcg(o * n, 0x9E37_79B9_7F4A_7C15);
        let (mant, e_row) = w_to_rows(&w, o, n).expect("rows");
        let a: Vec<Fx> = (0..o)
            .map(|r| group_scale_int(&mant[r * n..(r + 1) * n], e_row[r], &ufac, 0).expect("scale"))
            .collect();
        let run_b = |m: &[i64], er: &[i32], a: &[Fx], rows: usize, wscale: bool| {
            let (mut s, mut e, mut i) = buf(rows);
            let an = beam_pass(
                &ufac, n, 0, n, m, er, a, rows, 4, wscale, &mut s, &mut e, &mut i,
            )
            .expect("beam");
            (s, e, i, an)
        };
        for wscale in [false, true] {
            let full = run_b(&mant, &e_row, &a, o, wscale);
            let mut per_row = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for r in 0..o {
                let (s, e, i, an) = run_b(
                    &mant[r * n..(r + 1) * n],
                    &e_row[r..=r],
                    &a[r..=r],
                    1,
                    wscale,
                );
                per_row.0.extend(s);
                per_row.1.extend(e);
                per_row.2.extend(i);
                per_row.3.extend(an);
            }
            assert_eq!(
                full, per_row,
                "per-row calls changed the beam (wscale={wscale})"
            );
        }
    }

    #[test]
    fn select_k_ties_take_the_smaller_flat_index() {
        let mut out = Vec::new();
        select_k(&[7_i128, 2, 2, 7, 2], 3, &mut out);
        assert_eq!(out, [(1, 2), (2, 2), (4, 2)]); // ties: smaller flat index
    }

    /// `tri_upper_solve` on a factor whose `k > j` entries are non-zero (the stored factor has zeros
    /// there): it recovers the exact `x` of `x·U = e`.
    #[test]
    fn tri_upper_solve_inverts_a_dense_factor() {
        let n = 4usize;
        let u = UFac {
            hat: vec![10, 0, 0, 0, 2, 11, 0, 0, 3, 4, 12, 0, -1, 5, 6, 13],
            d: vec![0; n],
            n,
        };
        let x = [1_i64, -2, 3, -1];
        let mut e = Vec::with_capacity(n);
        for j in 0..n {
            let v: i64 = (j..n).map(|k| x[k] * u.hat[k * n + j]).sum();
            e.push(Fx { m: v, e: -32 }); // `U_kj = hat_kj·2^-32`
        }
        tri_upper_solve(&u, n, 0, n, &mut e).expect("solve");
        for (j, xj) in x.iter().enumerate() {
            assert_eq!(fx_to_f64(&e[j]), *xj as f64, "x[{j}]");
        }
    }

    /// The second pass replaces the first only on a strictly smaller whitened norm.
    #[test]
    fn second_pass_needs_a_strictly_smaller_norm() {
        assert!(second_pass_wins(5, 4));
        assert!(!second_pass_wins(5, 5));
        assert!(!second_pass_wins(5, 6));
    }

    /// The inter-group feedback on all rows equals one call per row.
    #[test]
    fn inter_group_feedback_is_per_row() {
        let (o, n, gs) = (16usize, 32usize, 8usize);
        let mut h = vec![0.0f64; n * n];
        for i in 0..n {
            h[i * n + i] = 8.0;
            for j in i.saturating_sub(3)..i {
                h[i * n + j] = 1.0;
                h[j * n + i] = 1.0;
            }
        }
        let ufac = feedback_u(&h, n, 0.01).expect("SPD factor");
        let wm: Vec<i64> = lcg(o * n, 0x1234_5678_9ABC_DEF0)
            .iter()
            .map(|x| (x * 1_000_000_000.0) as i64)
            .collect();
        let er = vec![-30i32; o];
        // A residual large enough that the update moves the row grid `2^-30`.
        let e: Vec<Fx> = (0..o * gs)
            .map(|r| Fx {
                m: (1_i64 << 24) + r as i64,
                e: 0,
            })
            .collect();
        let (mut w1, mut er1) = (wm.clone(), er.clone());
        inter_group_feedback(&mut w1, &mut er1, &e, &ufac, n, 0, gs).expect("feedback");
        let (mut w2, mut er2) = (wm.clone(), er.clone());
        for r in 0..o {
            inter_group_feedback(
                &mut w2[r * n..(r + 1) * n],
                &mut er2[r..=r],
                &e[r * gs..(r + 1) * gs],
                &ufac,
                n,
                0,
                gs,
            )
            .expect("feedback, one row");
        }
        assert_ne!(w1, wm, "the feedback did not move the rows");
        assert_eq!((w1, er1), (w2, er2), "per-row calls changed the feedback");
    }
}
