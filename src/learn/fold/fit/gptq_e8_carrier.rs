//! The integer-carrier boundary of [`super::gptq_e8`]: the feedback factor [`UFac`], the act order
//! [`block_order`], the group scale [`group_scale_int`] and the `f64` → integer conversions of the
//! damped Hessian ([`h_to_spdq`]) and of the weight ([`w_to_rows`]). The still-`f64` selection
//! passes receive their views through [`ufac_f64`] and [`rows_to_f64`]; lanes L1b–L1d delete both.

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
use crate::engine::ONE;
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

/// The `f64` view of a [`UFac`]: `U_ij = (hat_ij / 2^32)·2^-d_j` above the diagonal, `0` below.
///
/// `// L1b-L1d delete this` — the transient `f64` boundary while the selection passes are still
/// `f64`; it is a conversion step, not a second path.
pub(super) fn ufac_f64(u: &UFac) -> Vec<f64> {
    let n = u.n;
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            out[i * n + j] = u.hat[i * n + j] as f64 / ONE as f64 * pow2i(-u.d[j]);
        }
    }
    out
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

/// The `f64` view of a [`w_to_rows`] pair: `mant · 2^e_row` per row.
///
/// `// L1b-L1d delete this` — the transient `f64` boundary while the selection passes are still
/// `f64`.
pub(super) fn rows_to_f64(mant: &[i64], e_row: &[i32], o: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; o * n];
    for r in 0..o {
        let s = pow2i(e_row[r]);
        for j in 0..n {
            out[r * n + j] = mant[r * n + j] as f64 * s;
        }
    }
    out
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

/// The real value of an [`Fx`] as an `f64` (`m · 2^e`), the transient boundary the still-`f64`
/// passes read.
pub(super) fn fx_to_f64(x: &Fx) -> f64 {
    x.m as f64 * pow2i(x.e)
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

/// The integer group scale `a = 0.828·w_rms` of the working row `w_row` (the group's `g1−g0`
/// columns of the current `wk`) with inverse-variance weights `prec_j = 2^(2 d_j)/Uhat_jj²`:
/// `w_rms² = Σ_j w_j²·prec_j / Σ_j prec_j`, accumulated in `i128` with every term aligned to the
/// common exponent `2 d_j` (the `m_j = round(2^96/Uhat_jj²)` scale below, so `z_j = RNE(m_j·2^{2d_j})`
/// is `prec_j·2^96/2^64`, ~`2^32` bits of resolution at `d_j = 0`), `sqrt` by [`isqrt`] and the
/// `0.828` by [`E8_SCALE_Q32`]. A term or sum that leaves `i128` is [`GptqE8Error::Overflow`].
///
/// # Errors
///
/// [`GptqE8Error::NonFinite`] on a non-finite weight; [`GptqE8Error::NotPositiveDefinite`] if a
/// feedback diagonal is non-positive; [`GptqE8Error::Overflow`] on a checked `i128` operation.
///
/// # Complexity
///
/// Time `O(gs)` checked `i128` operations (one scan, one accumulation pass), space `O(gs)` for the
/// mantissas.
pub(super) fn group_scale_int(w_row: &[f64], u: &UFac, g0: usize) -> Result<Fx, GptqE8Error> {
    let gs = w_row.len();
    let n = u.n;
    let mut max_abs = 0.0f64;
    for &x in w_row {
        if !x.is_finite() {
            return Err(GptqE8Error::NonFinite);
        }
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let mut mant = vec![0i64; gs];
    let e_row = if max_abs == 0.0 {
        0
    } else {
        let (m, e) = f64_int_exp(max_abs);
        let bl = match ilog2_pos(mag_u128(m)) {
            Some(b) => b + 1,
            None => return Err(GptqE8Error::Overflow),
        };
        let l = e + bl.cast_signed() - 1;
        let er = l - 34;
        for j in 0..gs {
            mant[j] = rne_scaled_i64(w_row[j], -er)?;
        }
        er
    };
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
        let m2 = i128::from(mant[j]) * i128::from(mant[j]);
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
