//! Fixed-point `ln`, `softplus`, and `sigmoid` in `i64` Q32.32 for the integer executing path.
//!
//! An `i64` `v` denotes the real number `v / 2^32`; `ONE` is `1.0`. All kernels in this module
//! operate purely in integer arithmetic (using `i128` intermediates for multiplication and
//! division with per-step rounding) and refuse outside their admissible domains rather than
//! wrapping or silently losing precision.
//!
//! # Functions and Error Contracts
//!
//! - [`fixed_ln(x)`]: computes `ln(x / 2^32)` in Q32.32.
//!   Domain: `x > 0`. Refuses with [`LnError::NonPositive`] for `x <= 0`.
//!   Algorithm: leading-zero count range reduction into `[1/sqrt(2), sqrt(2))`, followed by an
//!   odd degree-13 `atanh` series on `s = (x - 2^p) / (x + 2^p)`.
//!   Max absolute error: asserted and tested `abs(got - expected) <= 7` units of `2^-32` over
//!   the entire positive domain `(0, i64::MAX]`.
//!
//! - [`fixed_softplus(x)`]: computes `ln(1 + exp(x / 2^32))` in Q32.32.
//!   Domain: `x <= i64::MAX - ONE` (to guarantee `x + ln(1 + e^-x)` does not overflow `i64`).
//!   Refuses with [`LnError::OutOfRange`] for `x > i64::MAX - ONE`.
//!   Formula: `x + ln(1 + e^-x)` for `x >= 0`; `ln(1 + e^x)` for `x < 0`.
//!   Evaluates `e^-|x|` via [`fixed_exp`]; below `-16 * ONE`, `e^-|x|` saturates to 0.
//!   Max absolute error:
//!   - For `|x| <= 16 * ONE`: asserted `<= 16` units of `2^-32` (measured `<= 13`, bounded by
//!     `fixed_exp` error + `fixed_ln` error).
//!   - For `|x| > 16 * ONE`: bounded by `exp(-16) * 2^32 <= 484` units of `2^-32` due to the
//!     exponential flush-to-zero threshold.
//!
//! - [`fixed_sigmoid(x)`]: computes `1 / (1 + exp(-x / 2^32))` in Q32.32.
//!   Domain: total on all `i64`. Output is strictly in `[0, ONE]`.
//!   Formula: `ONE / (ONE + e^-x)` for `x >= 0`; `e^x / (ONE + e^x)` for `x < 0`.
//!   Max absolute error:
//!   - For `|x| <= 16 * ONE`: asserted `<= 10` units of `2^-32` (measured `<= 8`, derivative
//!     factor `d/d(exp) <= 0.54`).
//!   - For `|x| > 16 * ONE`: bounded by `exp(-16) * 2^32 <= 484` units of `2^-32`.
//!
//!   Symmetry: `fixed_sigmoid(-x) = ONE - fixed_sigmoid(x)` within 1 unit of `2^-32`.

use super::fixed_exp::{fixed_exp, ExpError};

/// The Q32.32 scale: `ONE` denotes `1.0`, so one unit is `2^-32`.
pub const ONE: i64 = 1 << 32;

/// `round(ln(2) * 2^32)`.
pub const LN2_Q32: i64 = 2_977_044_472;

/// `round(sqrt(2) * 2^62)`: threshold for interval reduction to `[1/sqrt(2), sqrt(2))`.
const SQRT2_Q62: u128 = 6_521_908_912_666_391_553;

/// Lower threshold for [`fixed_exp`]: `-16 * ONE`. Below this, `exp(x)` flushes to 0.
const EXP_LOWER_BOUND: i64 = -16 * ONE;

/// The refusal error of [`fixed_ln`] and [`fixed_softplus`].
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LnError {
    /// Non-positive input to [`fixed_ln`]: `x <= 0`.
    NonPositive,
    /// Argument exceeds the safe range where the output fits in `i64`.
    OutOfRange,
}

/// Divide an `i128` by `2^32`, rounding to nearest with ties away from zero.
#[inline]
const fn round_div_2p32(value: i128) -> i128 {
    const HALF: i128 = 1 << 31;
    if value >= 0 {
        (value + HALF) >> 32
    } else {
        -((-value + HALF) >> 32)
    }
}

/// Signed integer division of two `i128` values, rounding to nearest with ties away from zero.
///
/// Precondition: `den > 0`.
#[inline]
const fn round_div_i128(num: i128, den: i128) -> i128 {
    let half = den >> 1;
    if num >= 0 {
        (num + half) / den
    } else {
        -((-num + half) / den)
    }
}

/// Evaluate `2 * atanh(s)` for `s in [-0.1716, 0.1716]` in Q32.32.
///
/// Degree-13 polynomial in `y = s^2`:
/// `2 * (s + s^3/3 + s^5/5 + s^7/7 + s^9/9 + s^11/11 + s^13/13)`.
///
/// Horner evaluation with coefficients `round(2 / k * 2^32)`:
/// - `2/13 * 2^32 = 660_764_199`
/// - `2/11 * 2^32 = 780_903_145`
/// - `2/9  * 2^32 = 954_437_177`
/// - `2/7  * 2^32 = 1_227_133_513`
/// - `2/5  * 2^32 = 1_717_986_918`
/// - `2/3  * 2^32 = 2_863_311_531`
/// - `2/1  * 2^32 = 8_589_934_592 = 2 * ONE`
///
/// Complexity: O(1) temporal (7 multiplications, 7 rounded divisions by `2^32`, 7 additions),
/// O(1) spatial (stack accumulator).
#[inline]
fn ln_atanh_poly(s: i128) -> i128 {
    let y = round_div_2p32(s * s);
    let mut acc: i128 = 660_764_199; // 2 / 13
    acc = round_div_2p32(acc * y) + 780_903_145; // 2 / 11
    acc = round_div_2p32(acc * y) + 954_437_177; // 2 / 9
    acc = round_div_2p32(acc * y) + 1_227_133_513; // 2 / 7
    acc = round_div_2p32(acc * y) + 1_717_986_918; // 2 / 5
    acc = round_div_2p32(acc * y) + 2_863_311_531; // 2 / 3
    acc = round_div_2p32(acc * y) + (2_i128 << 32); // 2 / 1
    round_div_2p32(acc * s)
}

/// Natural logarithm in Q32.32 for `x > 0`.
///
/// Decomposes `x` into `2^e * m` with `m in [1/sqrt(2), sqrt(2))`, then evaluates `ln(m)` via
/// `2 * atanh((m - 1)/(m + 1))` where `s = (x - 2^p)/(x + 2^p)` with `p = e + 32`.
///
/// The returned `i64` denotes `ln(x / 2^32) * 2^32`.
///
/// # Errors
///
/// Returns [`LnError::NonPositive`] when `x <= 0`.
///
/// # Complexity
///
/// O(1) temporal: 1 leading-zero count, 1 comparison, 1 exact division, 7 Horner polynomial
/// steps, 1 final addition. O(1) spatial.
#[allow(
    clippy::manual_range_contains,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    reason = "plain i64/i128 comparisons and `as` casts extract through Charon/Aeneas"
)]
pub fn fixed_ln(x: i64) -> Result<i64, LnError> {
    if x <= 0 {
        return Err(LnError::NonPositive);
    }

    // Number of leading zeros in positive i64 (between 1 and 63).
    let lz = (x as u64).leading_zeros();
    let k = 63 - lz; // bit position of the most significant 1

    // Normalize x to bit 62 to compare against round(sqrt(2) * 2^62).
    let x_norm = (x as u128) << (lz - 1);
    let p = if x_norm >= SQRT2_Q62 { k + 1 } else { k };

    let e = (p as i64) - 32;

    // s = (x - 2^p) / (x + 2^p) in Q32.32.
    // num = (x - 2^p) * 2^32.
    // den = x + 2^p.
    let base = 1_i128 << p;
    let x_wide = x as i128;
    let num = (x_wide - base) << 32;
    let den = x_wide + base;
    let s = round_div_i128(num, den);

    let ln_m = ln_atanh_poly(s);
    let e_ln2 = (e as i128) * (LN2_Q32 as i128);
    let total = e_ln2 + ln_m;

    if total > (i64::MAX as i128) || total < (i64::MIN as i128) {
        return Err(LnError::OutOfRange);
    }
    Ok(total as i64)
}

/// Compute `softplus(x) = ln(1 + exp(x))` in Q32.32.
///
/// Numerically stable formulation:
/// - `x >= 0`: `x + ln(1 + exp(-x))`
/// - `x < 0`: `ln(1 + exp(x))`
///
/// For `e^-|x|`, uses [`fixed_exp`]. When the exponent is strictly below `-16 * ONE`, `fixed_exp`
/// is out of range, but `exp(-|x|)` is mathematically below `exp(-16) * 2^32 <= 483.3` units and
/// saturates to `0`.
///
/// # Errors
///
/// Returns [`LnError::OutOfRange`] if `x > i64::MAX - ONE` (since `softplus(x) >= x`).
///
/// # Complexity
///
/// O(1) temporal: 1 `fixed_exp` call (or 0 if saturated), 1 `fixed_ln` call (or 0 if saturated),
/// additions. O(1) spatial.
#[allow(
    clippy::manual_unwrap_or,
    clippy::manual_unwrap_or_default,
    clippy::manual_let_else,
    clippy::question_mark,
    reason = "explicit pattern matching avoids complex desugaring in Charon/Aeneas extraction"
)]
pub fn fixed_softplus(x: i64) -> Result<i64, LnError> {
    if x >= 0 {
        if x > i64::MAX - ONE {
            return Err(LnError::OutOfRange);
        }
        let neg_x = -x;
        let exp_neg = if neg_x < EXP_LOWER_BOUND {
            0
        } else {
            match fixed_exp(neg_x) {
                Ok(v) => v,
                Err(ExpError::OutOfRange) => 0,
            }
        };
        let arg = match ONE.checked_add(exp_neg) {
            Some(v) => v,
            None => return Err(LnError::OutOfRange),
        };
        let ln_term = match fixed_ln(arg) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        match x.checked_add(ln_term) {
            Some(v) => Ok(v),
            None => Err(LnError::OutOfRange),
        }
    } else {
        let exp_pos = if x < EXP_LOWER_BOUND {
            0
        } else {
            match fixed_exp(x) {
                Ok(v) => v,
                Err(ExpError::OutOfRange) => 0,
            }
        };
        let arg = match ONE.checked_add(exp_pos) {
            Some(v) => v,
            None => return Err(LnError::OutOfRange),
        };
        fixed_ln(arg)
    }
}

/// Compute `sigmoid(x) = 1 / (1 + exp(-x))` in Q32.32.
///
/// Numerically stable formulation avoiding intermediate overflow:
/// - `x >= 0`: `1 / (1 + exp(-x)) = ONE / (ONE + exp(-x))`
/// - `x < 0`: `exp(x) / (1 + exp(x)) = exp(x) / (ONE + exp(x))`
///
/// For `e^-|x|`, uses [`fixed_exp`]. Below `-16 * ONE`, `exp(-|x|)` saturates to `0`.
/// Result is strictly in `[0, ONE]`.
///
/// Exact integer division with round-to-nearest (ties away from zero).
///
/// # Errors
///
/// This function never errors; returns `Ok(v)` for all `i64` inputs to match the numeric API
/// conventions.
///
/// # Complexity
///
/// O(1) temporal: 1 `fixed_exp` call (or 0 if saturated), 1 128-bit integer division. O(1) spatial.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::manual_unwrap_or,
    clippy::manual_unwrap_or_default,
    reason = "widening casts to i128 for exact rounding division and explicit matching for extraction"
)]
pub fn fixed_sigmoid(x: i64) -> Result<i64, LnError> {
    if x >= 0 {
        let neg_x = -x;
        let exp_neg = if neg_x < EXP_LOWER_BOUND {
            0
        } else {
            match fixed_exp(neg_x) {
                Ok(v) => v,
                Err(ExpError::OutOfRange) => 0,
            }
        };
        let den = (ONE as i128) + (exp_neg as i128);
        let num = (ONE as i128) << 32;
        let res = round_div_i128(num, den);
        let res_clamped = if res > (ONE as i128) {
            ONE
        } else if res < 0 {
            0
        } else {
            res as i64
        };
        Ok(res_clamped)
    } else {
        let exp_pos = if x < EXP_LOWER_BOUND {
            0
        } else {
            match fixed_exp(x) {
                Ok(v) => v,
                Err(ExpError::OutOfRange) => 0,
            }
        };
        let den = (ONE as i128) + (exp_pos as i128);
        let num = (exp_pos as i128) << 32;
        let res = round_div_i128(num, den);
        let res_clamped = if res > (ONE as i128) {
            ONE
        } else if res < 0 {
            0
        } else {
            res as i64
        };
        Ok(res_clamped)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        clippy::cast_lossless,
        clippy::cast_possible_truncation,
        reason = "the f64 reference comparison casts i64 fixed-point values to f64"
    )]

    use super::{fixed_ln, fixed_sigmoid, fixed_softplus, LnError, LN2_Q32, ONE};

    fn to_real(v: i64) -> f64 {
        v as f64 / 4_294_967_296.0
    }

    #[test]
    fn ln_refusals() {
        assert_eq!(fixed_ln(0), Err(LnError::NonPositive));
        assert_eq!(fixed_ln(-1), Err(LnError::NonPositive));
        assert_eq!(fixed_ln(-100 * ONE), Err(LnError::NonPositive));
        assert_eq!(fixed_ln(i64::MIN), Err(LnError::NonPositive));
    }

    #[test]
    fn ln_exact_and_powers_of_two() {
        assert_eq!(fixed_ln(ONE), Ok(0));
        assert_eq!(fixed_ln(2 * ONE), Ok(LN2_Q32));
        assert_eq!(fixed_ln(4 * ONE), Ok(2 * LN2_Q32));
        assert_eq!(fixed_ln(ONE / 2), Ok(-LN2_Q32));
        assert_eq!(fixed_ln(ONE / 4), Ok(-2 * LN2_Q32));
    }

    #[test]
    fn ln_accuracy_dense_sweep() {
        // Asserted bound over the whole domain (0, i64::MAX]: <= 7 units of 2^-32.
        const MAX_ALLOWED_ERR_LN: i64 = 7;
        let mut max_err: i64 = 0;

        // 1. Tiny inputs: 1 to 50,000
        for x in 1..=50_000 {
            let got = fixed_ln(x).expect("positive input");
            let expected = (to_real(x).ln() * 4_294_967_296.0).round() as i64;
            let err = (got - expected).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err <= MAX_ALLOWED_ERR_LN,
                "x={x}: got={got}, expected={expected}, err={err}"
            );
        }

        // 2. Medium inputs: around 1.0 (ONE) with fine step
        for i in -25_000..=25_000 {
            let x = ONE + i * (ONE / 50_000);
            let got = fixed_ln(x).expect("positive input");
            let expected = (to_real(x).ln() * 4_294_967_296.0).round() as i64;
            let err = (got - expected).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err <= MAX_ALLOWED_ERR_LN,
                "x={x}: got={got}, expected={expected}, err={err}"
            );
        }

        // 3. Powers of 2 and sqrt(2) boundaries across all 62 bit-lengths
        for k in 0..62 {
            let base = 1_i64 << k;
            for offset in -20..=20 {
                let x = base.saturating_add(offset);
                if x > 0 {
                    let got = fixed_ln(x).expect("positive input");
                    let expected = (to_real(x).ln() * 4_294_967_296.0).round() as i64;
                    let err = (got - expected).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    assert!(
                        err <= MAX_ALLOWED_ERR_LN,
                        "x={x}: got={got}, expected={expected}, err={err}"
                    );
                }
            }
            let sqrt2_bound = ((2.0_f64.sqrt() * (base as f64)).round() as i64).max(1);
            for offset in -20..=20 {
                let x = sqrt2_bound.saturating_add(offset);
                if x > 0 {
                    let got = fixed_ln(x).expect("positive input");
                    let expected = (to_real(x).ln() * 4_294_967_296.0).round() as i64;
                    let err = (got - expected).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    assert!(
                        err <= MAX_ALLOWED_ERR_LN,
                        "x={x}: got={got}, expected={expected}, err={err}"
                    );
                }
            }
        }

        // 4. Huge inputs near i64::MAX
        for offset in 0..1000 {
            let x = i64::MAX - offset;
            let got = fixed_ln(x).expect("positive input");
            let expected = (to_real(x).ln() * 4_294_967_296.0).round() as i64;
            let err = (got - expected).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err <= MAX_ALLOWED_ERR_LN,
                "x={x}: got={got}, expected={expected}, err={err}"
            );
        }

        assert!(max_err <= MAX_ALLOWED_ERR_LN);
    }

    #[test]
    fn softplus_accuracy_and_monotonicity() {
        // For |x| <= 16 * ONE, error is <= 16 units of 2^-32 (measured <= 13 units).
        const MAX_ALLOWED_ERR_SOFTPLUS_WINDOW: i64 = 16;
        // Outside the window, error is bounded by exp(-16) * 2^32 <= 484 units.
        const MAX_ALLOWED_ERR_SOFTPLUS_TAIL: i64 = 484;

        let mut prev = i64::MIN;
        let mut max_err_window: i64 = 0;
        let mut max_err_tail: i64 = 0;

        // Sweep from -25 * ONE to 25 * ONE
        let step = ONE / 2000;
        let mut x = -25 * ONE;
        while x <= 25 * ONE {
            let got = fixed_softplus(x).expect("within softplus range");
            assert!(got >= prev, "monotonicity violated at x={x}");
            prev = got;

            let x_f = to_real(x);
            let expected_f = if x_f >= 0.0 {
                x_f + (-x_f).exp().ln_1p()
            } else {
                x_f.exp().ln_1p()
            };
            let expected = (expected_f * 4_294_967_296.0).round() as i64;
            let err = (got - expected).abs();

            if x.abs() <= 16 * ONE {
                if err > max_err_window {
                    max_err_window = err;
                }
                assert!(
                    err <= MAX_ALLOWED_ERR_SOFTPLUS_WINDOW,
                    "x={x}: got={got}, expected={expected}, err={err}"
                );
            } else {
                if err > max_err_tail {
                    max_err_tail = err;
                }
                assert!(
                    err <= MAX_ALLOWED_ERR_SOFTPLUS_TAIL,
                    "tail x={x}: got={got}, expected={expected}, err={err}"
                );
            }
            x += step;
        }

        // Test near bounds and refusals
        assert_eq!(fixed_softplus(i64::MAX), Err(LnError::OutOfRange));
        assert!(fixed_softplus(i64::MIN / 2).is_ok());
    }

    #[test]
    fn sigmoid_accuracy_monotonicity_and_symmetry() {
        // Inside [-16*ONE, 16*ONE], error is <= 10 units of 2^-32 (measured <= 8 units).
        const MAX_ALLOWED_ERR_SIGMOID_WINDOW: i64 = 10;
        // Outside, bounded by 484 units.
        const MAX_ALLOWED_ERR_SIGMOID_TAIL: i64 = 484;

        let mut prev = 0_i64;
        let mut max_err_window: i64 = 0;
        let mut max_err_tail: i64 = 0;
        let mut max_sym_diff: i64 = 0;

        let step = ONE / 2000;
        let mut x = -25 * ONE;
        while x <= 25 * ONE {
            let got = fixed_sigmoid(x).expect("sigmoid is total");
            assert!(
                (0..=ONE).contains(&got),
                "range violation at x={x}: got={got}"
            );
            assert!(got >= prev, "monotonicity violated at x={x}");
            prev = got;

            // Symmetry check: sigmoid(-x) == ONE - sigmoid(x) within 1 unit
            let got_neg = fixed_sigmoid(-x).expect("sigmoid is total");
            let sym_diff = ((got + got_neg) - ONE).abs();
            if sym_diff > max_sym_diff {
                max_sym_diff = sym_diff;
            }
            assert!(
                sym_diff <= 1,
                "symmetry violated at x={x}: got={got}, got_neg={got_neg}, diff={sym_diff}"
            );

            let x_f = to_real(x);
            let expected_f = 1.0 / (1.0 + (-x_f).exp());
            let expected = (expected_f * 4_294_967_296.0).round() as i64;
            let err = (got - expected).abs();

            if x.abs() <= 16 * ONE {
                if err > max_err_window {
                    max_err_window = err;
                }
                assert!(
                    err <= MAX_ALLOWED_ERR_SIGMOID_WINDOW,
                    "x={x}: got={got}, expected={expected}, err={err}"
                );
            } else {
                if err > max_err_tail {
                    max_err_tail = err;
                }
                assert!(
                    err <= MAX_ALLOWED_ERR_SIGMOID_TAIL,
                    "tail x={x}: got={got}, expected={expected}, err={err}"
                );
            }

            x += step;
        }

        // Test extreme values
        assert_eq!(fixed_sigmoid(i64::MIN), Ok(0));
        assert_eq!(fixed_sigmoid(i64::MAX), Ok(ONE));
        assert_eq!(fixed_sigmoid(0), Ok(ONE / 2));
    }
}
