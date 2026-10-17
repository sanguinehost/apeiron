//! The integer-carrier boundary of [`super::gptq_e8`]: the feedback factor [`UFac`], the act order
//! [`block_order`], the group scale [`group_scale_int`], the triangular solve [`tri_upper_solve`],
//! the whitened scale projection [`project_scale_int`], the inter-group feedback
//! [`inter_group_feedback`], and the `f64` → integer conversions of the damped Hessian
//! ([`h_to_spdq`]) and of the weight ([`w_to_rows`]). Everything downstream of those conversions is
//! exact integer arithmetic; the only `f64` left is the public input boundary and the [`fx_to_f64`]
//! output boundary.

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

use super::gptq_e8::{GptqE8Error, BLOCK, E8_SCALE_Q32, GROUP, INV_THREADS};
use crate::learn::solve::chol::SolveError;
use crate::learn::solve::inv::inv_factor_upper;
use crate::learn::solve::q32::{ilog2_pos, round_q32_even, rshift_rne, Fx, Q32Error, SpdQ};
use crate::numeric::fixed_rsqrt::isqrt;

/// The exact signed-integer decomposition `x = m · 2^e` of a finite nonzero `f64`: `m` is the
/// 53-bit significand (with sign) and `e` the binary exponent, both exact.
///
/// # Complexity
///
/// `O(1)`.
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
/// `floor(log2 max_abs) − g ∈ {29, 30}`, the parity of `floor(log2 max_abs)` choosing which.
/// Even `g` keeps the half-exponent `g/2` an integer for the [`SpdQ`] fold, and the two possible
/// magnitudes both keep the Q32.32 word below `2^63`. `0` for a zero or non-finite magnitude.
///
/// # Complexity
///
/// `O(1)`.
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

/// The Q32.32 word `RNE(x · 2^k)` of a finite `x`: the value crosses into integers with a single
/// round-to-nearest, ties to even, on the significand. When `e + k ≥ 0` the exact Q64.64 integer is
/// formed and rounded by [`round_q32_even`]; otherwise the value needs fewer than 64 fractional bits
/// and [`rshift_rne`] rounds it directly.
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] if `x` is a NaN or an infinite [`GptqE8Error::Overflow`] if the
/// rounded word leaves `i64`.
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

/// The Q32.32 words of `x · 2^-g`, one per entry, through [`rne_scaled_i64`].
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] or [`GptqE8Error::Overflow`] from [`rne_scaled_i64`].
///
/// # Complexity
///
/// Time `O(len)`, space `O(len)` for the words.
pub(super) fn to_words(x: &[f64], g: i32) -> Result<Vec<i64>, GptqE8Error> {
    let k = 32 - g;
    let mut out = vec![0i64; x.len()];
    for (w, &v) in out.iter_mut().zip(x) {
        *w = rne_scaled_i64(v, k)?;
    }
    Ok(out)
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

/// [`SolveError`] as a [`GptqE8Error`]: the primitive's refusals onto the variants this module
/// already returns, with the equilibration's non-positive diagonal and a non-positive `Cholesky`
/// pivot both reported as [`GptqE8Error::NotPositiveDefinite`] and a grid-range refusal as
/// [`GptqE8Error::Overflow`].
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

/// The `f64` → integer boundary of the damped Hessian, the **only** such conversion (L6 deletes it
/// when the caller hands an [`SpdQ`]): the even global exponent `g` ([`even_g`]) brings the largest
/// entry of `hh · 2^-g` into `[2^29, 2^31)`, [`to_words`] rounds the scaled entries to Q32.32 words
/// (`round_q32_even`, no `to_grid`), and [`SpdQ::from_q32`] equilibrates them. The returned `g` is
/// even; the caller folds `g/2` into `d` so the relation `hh = 2^d · Hhat · 2^d` holds for the
/// undamped-`g` scaling.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] if `n == 0` or `hh.len() != n·n`; [`GptqE8Error::NonFinite`] on a
/// non-finite entry; [`GptqE8Error::NotPositiveDefinite`] if a damped diagonal is below one word;
/// [`GptqE8Error::Overflow`] if a word leaves `i64`.
///
/// # Complexity
///
/// Time `O(n²)`, space `O(n²)` for the words and the [`SpdQ`]: one scale scan, one conversion pass
/// and one equilibration pass.
pub(super) fn h_to_spdq(hh: &[f64], n: usize) -> Result<(SpdQ, i32), GptqE8Error> {
    if n == 0 {
        return Err(GptqE8Error::ZeroDim);
    }
    if hh.len() != n * n {
        return Err(GptqE8Error::ShapeMismatch);
    }
    let mut max_abs = 0.0f64;
    for &x in hh {
        if !x.is_finite() {
            return Err(GptqE8Error::NonFinite);
        }
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let g = even_g(max_abs);
    let words = to_words(hh, g)?;
    let spdq = SpdQ::from_q32(&words, n).map_err(map_q32)?;
    Ok((spdq, g))
}

/// The inverse feedback factor `U` of the damped Hessian `H`: upper-triangular with `UᵀU = H⁻¹` for
/// `H = h + damp·mean(diag(h))·I`, and a zero diagonal entry replaced by `1.0` after damping (the
/// Python `feedback_u`, lines 148–158). The damping and the zero-diagonal rule stay in `f64`; the
/// damped `H` is mirrored to exact symmetry (its lower triangle into the upper one, the triangle the
/// blocked `Cholesky` reads) and crosses into Q32.32 once, through [`h_to_spdq`]; [`inv_factor_upper`]
/// (`src/learn/solve/inv.rs`) returns the Q32.32 upper factor `Uhat` of the equilibrated matrix.
/// With `d = spdq.d + g/2`, `hh = 2^d·Hhat·2^d` and `U = Uhat·diag(2^-d)` gives `UᵀU = hh⁻¹`.
///
/// # Errors
/// [`GptqE8Error::ShapeMismatch`] if `h.len() != n·n`; [`GptqE8Error::ZeroDim`] if `n == 0`;
/// [`GptqE8Error::NonFinite`] if `h` or `mean(diag(h))` is a NaN or an infinite;
/// [`GptqE8Error::NotPositiveDefinite`] if the damped matrix has a non-positive diagonal or
/// `Cholesky` pivot; [`GptqE8Error::Overflow`] if a checked `i128`/`i64` operation overflows.
///
/// # Complexity
/// - Temporal: `O(n³)` checked `i128` integer operations — the factor `n³/3` ([`inv_factor_upper`]'s
///   equilibrated `Cholesky`) plus the inverse `n³/6`. Measure: `i128` multiply-adds. Derivation:
///   one blocked factor and one blocked triangular inverse.
/// - Spatial: `O(n²)` words — the `f64` damped copy, the Q32.32 words, the [`SpdQ`] and
///   `inv_factor_upper`'s buffers.
pub fn feedback_u(h: &[f64], n: usize, damp: f64) -> Result<UFac, GptqE8Error> {
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
    // The Hadamard Gram arrives only approximately symmetric (the transform rounds each element
    // once). The blocked `Cholesky` reads the lower triangle, so the `f64` baseline's effective
    // matrix is that lower triangle; mirror it into the upper one, which makes the integer
    // carrier's exact-symmetry contract (`SpdQ::from_q32`) hold without changing the effective
    // matrix.
    for i in 0..n {
        for j in (i + 1)..n {
            hh[i * n + j] = hh[j * n + i];
        }
    }
    let (spdq, g) = h_to_spdq(&hh, n)?;
    let hat = inv_factor_upper(&spdq.hat, n, INV_THREADS).map_err(map_inv)?;
    let half = g / 2;
    let mut d = vec![0i32; n];
    for (di, &d0) in d.iter_mut().zip(&spdq.d) {
        *di = d0.checked_add(half).ok_or(GptqE8Error::Overflow)?;
    }
    Ok(UFac { hat, d, n })
}

/// Solve `x·U_blk = e` in place for the group's upper-triangular block `U_blk = U[g0:g0+gs,
/// g0:g0+gs]` of the carrier: with `U_kj = hat[(g0+k)·n + g0+j]·2^-32·2^-d_j` the back substitution
/// is `x_j = (e_j − Σ_{k>j} x_k·U_kj)/U_jj` for `j` descending. Every term is aligned exactly at the
/// common exponent of the row's `U_kj` factors and the whole contraction is one checked `i128`
/// accumulation; the single stored `x_j` is one round-to-nearest quotient of that `i128`
/// ([`div_shift_rne`]) with the output exponent chosen so the quotient needs at most 62 bits, so
/// no second rounding enters.
///
/// The contraction over `k > j` reads the strictly below-diagonal entries of the stored factor,
/// which are zero: for an upper-triangular `U` the split `Σ_i x_i U_ij = x_j U_jj + Σ_{i<j} x_i
/// U_ij` runs over the *earlier* columns, so `x·U = e` is the per-coordinate quotient
/// `x_j = e_j/U_jj`. The recurrence is kept explicit (and is exercised on a dense `UFac` by the
/// carrier test) so a factor whose below-diagonal entries are ever non-zero is solved as written.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] if `e.len() != gs`; [`GptqE8Error::NotPositiveDefinite`] on a
/// non-positive block diagonal; [`GptqE8Error::Overflow`] on a checked `i128` operation.
///
/// # Complexity
///
/// Temporal `O(gs²)` checked `i128` multiply-adds per call (`gs/8` block columns of the group loop),
/// spatial `O(1)` beyond the in-place row.
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

/// The per-row block floating point of the weight `w` (row-major `[o, n]`): mantissas `mant` and
/// per-row exponents `e_row` with `max |mant[r]| ∈ [2^34, 2^35)` (all-zero row: zeros, `e = 0`),
/// so `w[r][j] = mant[r·n+j]·2^e_row[r]` up to the rounding. `w` crosses into integers here, the
/// **only** such conversion of the weight.
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] on a non-finite entry; [`GptqE8Error::Overflow`] if a mantissa leaves
/// `i64`.
///
/// # Complexity
///
/// Time `O(o·n)`, space `O(o·n)` for the mantissas plus `O(o)` for the exponents: one maximum scan
/// and one rounding pass per row.
pub(super) fn w_to_rows(
    w: &[f64],
    o: usize,
    n: usize,
) -> Result<(Vec<i64>, Vec<i32>), GptqE8Error> {
    let mut mant = vec![0i64; o * n];
    let mut e_row = vec![0i32; o];
    for r in 0..o {
        let row = &w[r * n..(r + 1) * n];
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
        e_row[r] = er;
        for j in 0..n {
            mant[r * n + j] = rne_scaled_i64(row[j], -er)?;
        }
    }
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

/// The mantissas of `x` on the row exponent `e` (from [`row_exp`]): `m_j = RNE(x_j·2^-e)`, one
/// rounding per entry, each `|m_j| ≤ 2^(ROW_BITS+1)`.
///
/// # Errors
///
/// [`GptqE8Error::Overflow`] on a shift that leaves `i128`.
///
/// # Complexity
///
/// `O(len)`, `O(len)` output.
pub(super) fn renorm_row(x: &[Fx], e: i32) -> Result<Vec<i128>, GptqE8Error> {
    let mut out = Vec::with_capacity(x.len());
    for v in x {
        out.push(round_at(i128::from(v.m), i64::from(v.e - e))?);
    }
    Ok(out)
}

/// The whitened scale projection of the `wscale` branch: the row `u_s = S·U_blk⁻¹` by
/// [`tri_upper_solve`], then `a_new = max(a + ⟨e, u_s⟩/⟨u_s, u_s⟩, floor)` and
/// `e_new = e − (a_new − a)·u_s`. The input residual and `u_s` are renormalised to one
/// [`ROW_BITS`]-bit exponent per row (one rounding per entry), so both inner products are exact
/// `i128` sums with no alignment shift. Their ratio `num/den` is formed at 61 bits — each operand is
/// normalised to 62 bits (one RNE each) so the division's numerator stays inside `i128` — and one
/// [`div_rne_i128`] rounds the quotient. `e_new` is stored on the row's own [`ROW_BITS`]-bit
/// exponent with one rounding per entry.
///
/// `floor` is the smallest positive [`Fx`] at the scale's own exponent, `Fx { m: 1, e: a.e }` (the
/// grid unit of `a`; `A_CLAMP` when `a` is zero). The `f64` `1e-12` clamp has no carrier meaning, so
/// the floor states the representable-positive bound instead of a decimal constant.
/// `⟨u_s,u_s⟩ = 0` returns `a` unchanged with `e_new = e`.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] on a buffer of the wrong length; [`GptqE8Error::Overflow`] on a
/// checked `i128` operation.
///
/// # Complexity
///
/// Temporal `O(o·gs²)` `i128` multiply-adds — the `o` triangular solves `o·gs²/2` and the two inner
/// products `O(o·gs)`; spatial `O(o·gs)` [`Fx`] for `u_s` and `e_new`.
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
    let mut a_new = Vec::with_capacity(o);
    let mut e_new = vec![Fx { m: 0, e: 0 }; o * gs];
    let mut us = vec![Fx { m: 0, e: 0 }; gs];
    for r in 0..o {
        us.clone_from_slice(&s_cur[r * gs..(r + 1) * gs]);
        tri_upper_solve(u, n, g0, gs, &mut us)?;
        // Both operands of the inner products on one row exponent each: exact `i128` sums.
        let e_us = row_exp(&us);
        let um = renorm_row(&us, e_us)?;
        let er = &e_cur[r * gs..(r + 1) * gs];
        let e_e = row_exp(er);
        let em = renorm_row(er, e_e)?;
        let mut num: i128 = 0;
        let mut den: i128 = 0;
        for j in 0..gs {
            let pr = em[j].checked_mul(um[j]).ok_or(GptqE8Error::Overflow)?;
            num = num.checked_add(pr).ok_or(GptqE8Error::Overflow)?;
            let sq = um[j].checked_mul(um[j]).ok_or(GptqE8Error::Overflow)?;
            den = den.checked_add(sq).ok_or(GptqE8Error::Overflow)?;
        }
        let floor = if a[r].m > 0 {
            Fx { m: 1, e: a[r].e }
        } else {
            A_CLAMP
        };
        let an = if den == 0 || num == 0 {
            a[r].clone()
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
                a[r].clone()
            } else {
                let dlt =
                    Fx::from_i128(q, i32::try_from(sigma).map_err(|_| GptqE8Error::Overflow)?)
                        .map_err(|_| GptqE8Error::Overflow)?;
                let cand = fx_add(&a[r], &dlt)?;
                if cand.m <= 0 || fx_lt(&cand, &floor) {
                    floor
                } else {
                    cand
                }
            }
        };
        let dlt = fx_sub(&an, &a[r])?;
        a_new.push(an);
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
            e_new[r * gs + j] = Fx { m, e: eout };
        }
    }
    Ok((a_new, e_new))
}

/// `wk[:, g1:] -= E·U[g0:g1, g1:]` on the integer working rows: `e` is the group's `Fx` residual
/// (`o × gs`), `wm` the row-major `o × n` mantissas at the per-row exponents `e_row` and
/// `U_jc = hat[(g0+j)·n + c]·2^-32·2^-d_c`. Per stored entry `c` the `gs` products are summed in an
/// exact `i128` accumulation aligned at the common exponent of the residual terms, and the update is
/// **one** round-to-nearest to the row's grid `2^e_row`, so the update's absolute error is at most
/// `2^(e_row − 1)`.
///
/// The row exponent rises by one — every mantissa of that row halved with round-to-nearest — only
/// when an updated entry would reach the 62-bit headroom bound `|m| < 2^62`; the halving keeps the
/// row's real values and the one-rounding-per-entry rule applies to the shifted entry, so a row that
/// needs no shift (all fixtures: the update is a contraction of the row's own magnitude) is never
/// rounded a second time.
///
/// Rows are independent given the read-only `u` and `e`, so they are split into contiguous ranges
/// across `threads` scoped workers; each row is computed by one worker in the same order for every
/// count, so the output is bit-identical for `threads = 1` and `threads = 8`.
///
/// # Errors
///
/// [`GptqE8Error::ShapeMismatch`] on a wrong-length buffer; [`GptqE8Error::Overflow`] on a checked
/// `i128` operation.
///
/// # Complexity
///
/// Temporal `O(o·gs·(n − g1))` checked `i128` multiply-adds (one contraction per stored entry),
/// split across the workers; spatial `O((n − g1))` `i128` accumulator per worker row.
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
    threads: usize,
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
    let per = o.div_ceil(threads.max(1)).max(1);
    let tail = n - g1;
    std::thread::scope(|scope| -> Result<(), GptqE8Error> {
        let mut handles = Vec::new();
        for (ci, (wchunk, echunk)) in wm
            .chunks_mut(per * n)
            .zip(e_row.chunks_mut(per))
            .enumerate()
        {
            let r0 = ci * per;
            handles.push(scope.spawn(move || -> Result<(), GptqE8Error> {
                let rows = echunk.len();
                for r in 0..rows {
                    // The shared exponent of this row's residual terms.
                    let mut emin = i32::MAX;
                    for j in 0..gs {
                        let x = &e[(r0 + r) * gs + j];
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
                            let x = &e[(r0 + r) * gs + j];
                            if h == 0 || x.m == 0 {
                                continue;
                            }
                            let pr = i128::from(x.m)
                                .checked_mul(i128::from(h))
                                .ok_or(GptqE8Error::Overflow)?;
                            let term =
                                shift_left_i128(pr, x.e - emin).ok_or(GptqE8Error::Overflow)?;
                            sum = sum.checked_add(term).ok_or(GptqE8Error::Overflow)?;
                        }
                        // `Δw = −sum·2^(emin − 32 − d_c)`, one RNE to the row grid.
                        let mut sh = i64::from(emin) - 32 - i64::from(dc) - i64::from(echunk[r]);
                        let mut v = i128::from(wchunk[r * n + c])
                            .checked_add(round_at(
                                sum.checked_neg().ok_or(GptqE8Error::Overflow)?,
                                sh,
                            )?)
                            .ok_or(GptqE8Error::Overflow)?;
                        if blen(v) > 62 {
                            // The headroom rule: halve the whole row (RNE) and re-round this entry.
                            for q in 0..n {
                                let cur = i128::from(wchunk[r * n + q]);
                                let half = rshift_rne(cur, 1).ok_or(GptqE8Error::Overflow)?;
                                wchunk[r * n + q] = narrow_q(half)?;
                            }
                            echunk[r] = echunk[r].checked_add(1).ok_or(GptqE8Error::Overflow)?;
                            sh -= 1;
                            v = i128::from(wchunk[r * n + c])
                                .checked_add(round_at(
                                    sum.checked_neg().ok_or(GptqE8Error::Overflow)?,
                                    sh,
                                )?)
                                .ok_or(GptqE8Error::Overflow)?;
                        }
                        wchunk[r * n + c] = narrow_q(v)?;
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
fn shift_left_i128(v: i128, k: i32) -> Option<i128> {
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

/// `x + y` as an [`Fx`]: both terms aligned exactly at the finer exponent and one
/// [`Fx::from_i128`] normalisation. `O(1)`.
fn fx_add(x: &Fx, y: &Fx) -> Result<Fx, GptqE8Error> {
    let e = x.e.min(y.e);
    let a = shift_left_i128(i128::from(x.m), x.e - e).ok_or(GptqE8Error::Overflow)?;
    let b = shift_left_i128(i128::from(y.m), y.e - e).ok_or(GptqE8Error::Overflow)?;
    Fx::from_i128(a.checked_add(b).ok_or(GptqE8Error::Overflow)?, e)
        .map_err(|_| GptqE8Error::Overflow)
}

/// `x − y` as an [`Fx`], aligned exactly at the finer exponent with one [`Fx::from_i128`]
/// normalisation. `O(1)`.
fn fx_sub(x: &Fx, y: &Fx) -> Result<Fx, GptqE8Error> {
    let e = x.e.min(y.e);
    let a = shift_left_i128(i128::from(x.m), x.e - e).ok_or(GptqE8Error::Overflow)?;
    let b = shift_left_i128(i128::from(y.m), y.e - e).ok_or(GptqE8Error::Overflow)?;
    Fx::from_i128(a.checked_sub(b).ok_or(GptqE8Error::Overflow)?, e)
        .map_err(|_| GptqE8Error::Overflow)
}

/// `x < y` exactly for two positive [`Fx`] (`x.m > 0`, `y.m > 0`): a common-exponent `i128`
/// comparison, with a magnitude class deciding only when both aligned values leave `i128`.
/// `O(1)`.
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

/// The smallest exponent over the non-zero entries of every slice in `rows`, or `0` when all of
/// them are zero; the shared exponent of [`sum_sq_at`]. `O(Σ len)`.
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

/// `Σ m_i²·2^(2(e_i − e0))` for a shared exponent `e0 ≤ min_i e_i`: an exact checked `i128`
/// accumulation. The sum is `Σ v_i² · 2^(-2 e0)` of the represented values.
///
/// # Errors
///
/// [`GptqE8Error::Overflow`] when an aligned square or the sum leaves `i128`. The headroom bound is
/// `2·(e_i − e0) + 2·blen(m_i) ≤ 126` for every term.
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

/// The real value of an [`Fx`] as an `f64` at the public output boundary: the significand is rounded
/// once to 53 bits (RNE) and the power of two is exact, so the conversion is exact after that
/// rounding and `|m| ≤ 2^53` holds by construction.
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

/// The integer group scale `a = 0.828·w_rms` of the working row: `mant_row` is the group's `g1−g0`
/// mantissas at the row exponent `e_row` (`w = mant·2^e_row`, the [`w_to_rows`] form) and `prec_j =
/// 2^(2 d_j)/Uhat_jj²` the inverse-variance weights. `w_rms² = Σ_j m_j²·prec_j / Σ_j prec_j` is
/// accumulated in `i128` with every term aligned to the common exponent `2 d_j` (the `z_j =
/// RNE(m_ĵ·2^{2d_j})` scale below, so the term is `prec_j·2^96/2^64`), `sqrt` by [`isqrt`] and the
/// `0.828` by [`E8_SCALE_Q32`]. The result's exponent carries `e_row`: no re-normalisation of the
/// row happens here because the caller already holds the row on its integer grid.
///
/// # Errors
///
/// [`GptqE8Error::NotPositiveDefinite`] if a feedback diagonal is non-positive;
/// [`GptqE8Error::Overflow`] on a checked `i128` operation.
///
/// # Complexity
///
/// Time `O(gs)` checked `i128` operations (one accumulation pass), space `O(1)` beyond the input.
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
/// diagonal sum, ties by ascending index. Columns inside an 8-block keep their order.
///
/// Integer: the undamped `h` is converted once to Q32.32 words with the shared even exponent
/// [`even_g`]/[`to_words`] rule, and the block and group diagonal sums are exact `i128` word sums
/// (`sort_by_key` over `(Reverse(sum), index)`, ties by ascending index). No `f64` comparison.
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] or [`GptqE8Error::Overflow`] from [`to_words`].
///
/// # Complexity
///
/// Time `O(n² + nb·log nb + ng·log ng)` (`n²` for the conversion, the sums `O(n)`, two sorts over
/// the `n/8` blocks and the `⌈n/128⌉` groups), space `O(n²)` for the words plus `O(nb + ng)`.
pub(super) fn block_order(h: &[f64], n: usize) -> Result<(Vec<usize>, Vec<usize>), GptqE8Error> {
    let nb = n / BLOCK;
    let per_group = GROUP / BLOCK;
    let ng = n.div_ceil(GROUP);
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
    let words = to_words(h, even_g(max_abs))?;
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
