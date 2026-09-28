//! Fixed-point `sin`/`cos` in `i64` Q32.32 for the integer executing path.
//!
//! An `i64` `v` denotes the real number `v / 2^32`; `ONE` is `1.0`. [`fixed_sin_cos`] is total
//! over the window `[-4096 * ONE, 4096 * ONE]` radians and returns [`TrigError::OutOfRange`]
//! outside it; it never wraps or saturates. The refusal is a documented domain, not a failure mode
//! of the caller. [`turn_to_radians`] maps any turn phase in `Z/2^64` into that window, so `RoPE`
//! reaches `fixed_sin_cos` only through the total turn entry point.
//!
//! The algorithm is integer-only.
//!
//! **Range reduction.** `k = round(x / 2pi)` is computed from the Q32.32 reciprocal
//! `TWO_PI_RECIP_Q32 = round(2^32 / 2pi)` as `(x * TWO_PI_RECIP_Q32 + 2^63) >> 64`, so
//! `|x / 2pi - k| <= 1/2 + 3e-6`. Then `r = x - k * TWO_PI_Q32` with
//! `TWO_PI_Q32 = round(2pi * 2^32)`, so `|r| <= pi + 2e-5` in `2^32` units. `|k| <= 652` over the
//! window, so the `i128` product stays below `2^74`.
//!
//! **Quadrature.** `r` is folded into `[-pi/2, pi/2]` as `r - pi` (or `r + pi`) with a sign flip
//! on both outputs, then into `[-pi/4, pi/4]` by the quadrant swap `sin <-> cos` at `pi/2 - r`
//! (or `r + pi/2`). The fold uses the rounded `PI_Q32`, `HALF_PI_Q32` and `QUARTER_PI_Q32`
//! constants, so the folded argument exceeds `pi/4` by at most the reduction slack `2e-5`.
//!
//! **Series.** On `|t| <= pi/4` the degree-8 Taylor truncations are evaluated by `i128` Horner in
//! `u = t^2`: `sin t = t * (1 - u/3! + u^2/5! - u^3/7!)` (the `t^8` term of the degree-8
//! polynomial is zero) and `cos t = 1 - u/2! + u^2/4! - u^3/6! + u^4/8!`. The Q32.32
//! coefficients are `round(2^32 / n!)`; every partial product is rounded at `2^32` before the
//! next Horner step.
//!
//! **Rounding policy.** Per step, round to nearest, ties away from zero, at `2^-32`.
//!
//! **Accuracy**, measured by the dense-sample unit test over the whole window. The achieved
//! relative bound is `5e-7` for outputs above the floor, below the `2^-20` target; its dominant
//! term is the `sin` remainder `|t|^9 / 9!`, at most `4.5e-7` relative at `t = pi/4`. Near the
//! zeros the fixed-point reduction dominates: the rounded `TWO_PI_Q32` displaces `r` by up to
//! `|k| * 2^-33`, so each output carries an absolute floor. The asserted combined bound is
//! `|result / 2^32 - f(x / 2^32)| <= 5e-7 * |f(x / 2^32)| + 2^-24`, and the measured maxima over
//! the dense grid are `4.47e-7` relative and `1.42e-8` absolute.

/// The Q32.32 scale: `ONE` denotes `1.0`, so one unit is `2^-32`.
const ONE: i64 = 1 << 32;

/// `round(2pi * 2^32)`.
const TWO_PI_Q32: i64 = 26_986_075_409;

/// `round(2^32 / 2pi)`, the Q32.32 reciprocal used by the range reduction.
const TWO_PI_RECIP_Q32: i64 = 683_565_276;

/// `round(pi * 2^32)`.
const PI_Q32: i64 = 13_493_037_705;

/// `round(pi/2 * 2^32)`.
const HALF_PI_Q32: i64 = 6_746_518_852;

/// `round(pi/4 * 2^32)`.
const QUARTER_PI_Q32: i64 = 3_373_259_426;

/// `round(2^32 / 3!)`, the `t^3` coefficient of `sin`.
const SIN_C1: i64 = 715_827_883;

/// `round(2^32 / 5!)`, the `t^5` coefficient of `sin`.
const SIN_C2: i64 = 35_791_394;

/// `round(2^32 / 7!)`, the `t^7` coefficient of `sin`.
const SIN_C3: i64 = 852_176;

/// `round(2^32 / 2!)`, the `t^2` coefficient of `cos`.
const COS_C1: i64 = 2_147_483_648;

/// `round(2^32 / 4!)`, the `t^4` coefficient of `cos`.
const COS_C2: i64 = 178_956_971;

/// `round(2^32 / 6!)`, the `t^6` coefficient of `cos`.
const COS_C3: i64 = 5_965_232;

/// `round(2^32 / 8!)`, the `t^8` coefficient of `cos`.
const COS_C4: i64 = 106_522;

/// The refusal of a [`fixed_sin_cos`] argument outside its documented window.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TrigError {
    /// The argument lies outside `[-4096 * ONE, 4096 * ONE]`.
    OutOfRange,
}

/// The Q32.32 `sin` and `cos` of one angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinCos {
    /// `sin` of the angle, in Q32.32.
    pub sin: i64,
    /// `cos` of the angle, in Q32.32.
    pub cos: i64,
}

/// Divide an `i128` by `2^32`, rounding to nearest with ties away from zero.
const fn round_div_2p32(value: i128) -> i128 {
    const HALF: i128 = 1 << 31;
    if value >= 0 {
        (value + HALF) >> 32
    } else {
        -((-value + HALF) >> 32)
    }
}

/// Evaluate the degree-8 Taylor truncations of `sin` and `cos` at `t` in `[-pi/4, pi/4]`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "both results are bounded by ONE in magnitude, and `as` casts extract through \
              Charon/Aeneas while TryFrom emits an unmodeled external"
)]
fn taylor_sin_cos(t: i64) -> (i64, i64) {
    let t = i128::from(t);
    let u = round_div_2p32(t * t);

    // sin(t) = t * (1 - u/6 + u^2/120 - u^3/5040)
    let mut q = -i128::from(SIN_C3);
    q = round_div_2p32(q * u) + i128::from(SIN_C2);
    q = round_div_2p32(q * u) - i128::from(SIN_C1);
    q = round_div_2p32(q * u) + i128::from(ONE);
    let sin_t = round_div_2p32(t * q) as i64;

    // cos(t) = 1 - u/2 + u^2/24 - u^3/720 + u^4/40320
    let mut p = i128::from(COS_C4);
    p = round_div_2p32(p * u) - i128::from(COS_C3);
    p = round_div_2p32(p * u) + i128::from(COS_C2);
    p = round_div_2p32(p * u) - i128::from(COS_C1);
    p = round_div_2p32(p * u) + i128::from(ONE);
    let cos_t = p as i64;

    (sin_t, cos_t)
}

/// Evaluate `sin(x)` and `cos(x)` in Q32.32 by range reduction and degree-8 Taylor polynomials.
///
/// Each field of the result `v` denotes `v / 2^32 = sin(x / 2^32)` or `cos(x / 2^32)`. The
/// arithmetic is exact integer arithmetic with per-step rounding; see the module documentation
/// for the algorithm and the measured error.
///
/// # Errors
///
/// Returns [`TrigError::OutOfRange`] when `x` is outside `[-4096 * ONE, 4096 * ONE]`.
#[allow(
    clippy::manual_range_contains,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    reason = "plain i64/i128 comparisons and `as` casts after an explicit range check extract \
              through Charon/Aeneas, while RangeInclusive::contains, TryFrom and unsigned_abs \
              each emit an unmodeled external (an `axiom` in the emitted Lean) that the \
              fail-closed audit rejects"
)]
pub fn fixed_sin_cos(x: i64) -> Result<SinCos, TrigError> {
    const LOWER: i64 = -4096 * ONE;
    const UPPER: i64 = 4096 * ONE;

    if x < LOWER || x > UPPER {
        return Err(TrigError::OutOfRange);
    }

    // k = round(x / 2pi): the product is Q64.64, so shifting by 64 with a 2^63 half rounds to
    // nearest. |x| <= 2^44 and the reciprocal is below 2^30, so the product stays inside i128.
    let product = i128::from(x) * i128::from(TWO_PI_RECIP_Q32);
    let k_wide = (product + (1_i128 << 63)) >> 64;
    if k_wide > i64::MAX as i128 || k_wide < i64::MIN as i128 {
        return Err(TrigError::OutOfRange);
    }
    let k = k_wide as i64;

    // r = x - k * 2pi. |k| <= 652 in the window, so the product fits i64.
    let r_wide = i128::from(x) - i128::from(k) * i128::from(TWO_PI_Q32);
    if r_wide > i64::MAX as i128 || r_wide < i64::MIN as i128 {
        return Err(TrigError::OutOfRange);
    }
    let r = r_wide as i64;

    // Fold into [-pi/2, pi/2]: sin(r) = +-sin(s), cos(r) = +-cos(s).
    let (s, negate) = if r > HALF_PI_Q32 {
        (r - PI_Q32, true)
    } else if r < -HALF_PI_Q32 {
        (r + PI_Q32, true)
    } else {
        (r, false)
    };

    // Fold into [-pi/4, pi/4] by the quadrant swap sin <-> cos.
    let (arg, swap) = if s > QUARTER_PI_Q32 {
        (HALF_PI_Q32 - s, 1_i32)
    } else if s < -QUARTER_PI_Q32 {
        (s + HALF_PI_Q32, 2_i32)
    } else {
        (s, 0_i32)
    };

    let (sin_t, cos_t) = taylor_sin_cos(arg);
    let (mut sin_out, mut cos_out) = if swap == 1 {
        (cos_t, sin_t)
    } else if swap == 2 {
        (-cos_t, sin_t)
    } else {
        (sin_t, cos_t)
    };
    if negate {
        sin_out = -sin_out;
        cos_out = -cos_out;
    }

    Ok(SinCos {
        sin: sin_out,
        cos: cos_out,
    })
}

/// The Q32.32 radian angle denoted by a turn phase in `Z/2^64`.
///
/// A `u64` phase `p` denotes the fraction `p / 2^64` of one full turn. The two's-complement lift
/// `s = p as i64` selects the representative in `[-2^63, 2^63)` (a signed half-turn), and
/// `r = (s * TWO_PI_Q32 + 2^63) >> 64` is that phase's angle in Q32.32 radians, rounded to
/// nearest. The function is total: `|s| <= 2^63` gives `|r| <= round(pi * 2^32) + 1`, inside
/// [`fixed_sin_cos`]'s `[-4096 * ONE, 4096 * ONE]` window, so `fixed_sin_cos` never refuses it for
/// any `u64` phase.
///
/// # Complexity
///
/// `O(1)` time and space: one `i128` multiply and one shift, no branch.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "explicit `as` casts extract through Charon/Aeneas, while `i128::from` and `TryFrom` \
              emit an unmodeled external the fail-closed audit rejects; `p as i64` is the intended \
              two's-complement lift and the shifted result is bounded by pi * 2^32 + 1"
)]
#[must_use]
pub fn turn_to_radians(phase: u64) -> i64 {
    let s = phase as i64;
    let r = ((s as i128) * (TWO_PI_Q32 as i128) + (1_i128 << 63)) >> 64;
    r as i64
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        reason = "the f64 oracle and the Q32.32 result are compared in floating point; sample \
                  magnitudes exceed 2^53, so the cast is lossy by construction and the tolerance \
                  absorbs it"
    )]

    use super::{fixed_sin_cos, SinCos, TrigError, HALF_PI_Q32, ONE, PI_Q32, QUARTER_PI_Q32};

    /// `2^-20`, the relative-error target for outputs above the fixed-point floor.
    const TARGET_REL: f64 = 9.536_743_164_062_5e-7;

    /// The achieved relative-error bound asserted by the dense sample.
    const ACHIEVED_REL: f64 = 5.0e-7;

    /// `2^-24`, the absolute floor near the zeros of the outputs.
    const FLOOR: f64 = 5.960_464_477_539_063e-8;

    /// The dense-sample grid step, exact in Q32.32.
    const STEP: i64 = ONE / 256;

    /// Outputs with magnitude at least this are judged on relative error.
    const HALF: f64 = 0.5;

    fn to_real(v: i64) -> f64 {
        v as f64 / 4_294_967_296.0
    }

    #[test]
    fn exact_and_near_values() {
        assert_eq!(fixed_sin_cos(0), Ok(SinCos { sin: 0, cos: ONE }));
        assert_eq!(fixed_sin_cos(HALF_PI_Q32), Ok(SinCos { sin: ONE, cos: 0 }));
        assert_eq!(
            fixed_sin_cos(-HALF_PI_Q32),
            Ok(SinCos { sin: -ONE, cos: 0 })
        );

        let pi = fixed_sin_cos(PI_Q32).expect("pi is inside the window");
        assert_eq!(pi.cos, -ONE, "cos(pi) = {}", pi.cos);
        assert!(pi.sin.abs() <= 4, "sin(pi) = {}", pi.sin);

        let minus_quarter = fixed_sin_cos(-QUARTER_PI_Q32).expect("inside the window");
        let expected = -core::f64::consts::FRAC_1_SQRT_2;
        let got = to_real(minus_quarter.sin);
        assert!(
            (got - expected).abs() <= ACHIEVED_REL * expected.abs() + FLOOR,
            "sin(-pi/4) = {got}, expected {expected}"
        );
    }

    #[test]
    fn refuses_outside_the_window() {
        assert!(fixed_sin_cos(-4096 * ONE).is_ok());
        assert!(fixed_sin_cos(4096 * ONE).is_ok());
        assert_eq!(fixed_sin_cos(-4096 * ONE - 1), Err(TrigError::OutOfRange));
        assert_eq!(fixed_sin_cos(4096 * ONE + 1), Err(TrigError::OutOfRange));
        assert_eq!(fixed_sin_cos(i64::MIN), Err(TrigError::OutOfRange));
        assert_eq!(fixed_sin_cos(i64::MAX), Err(TrigError::OutOfRange));
    }

    #[test]
    fn accuracy_against_f64_sin_cos() {
        let mut max_rel_sin = 0.0f64;
        let mut max_rel_cos = 0.0f64;
        let mut max_floor = 0.0f64;

        let mut x = -4096 * ONE;
        while x <= 4096 * ONE {
            let result = fixed_sin_cos(x).expect("sampled inside the window");
            let xr = to_real(x);
            let (expected_sin, expected_cos) = (xr.sin(), xr.cos());
            let got_sin = to_real(result.sin);
            let got_cos = to_real(result.cos);
            let err_sin = (got_sin - expected_sin).abs();
            let err_cos = (got_cos - expected_cos).abs();

            assert!(
                err_sin <= ACHIEVED_REL * expected_sin.abs() + FLOOR,
                "sin x={x} expected={expected_sin} got={got_sin} err={err_sin}"
            );
            assert!(
                err_cos <= ACHIEVED_REL * expected_cos.abs() + FLOOR,
                "cos x={x} expected={expected_cos} got={got_cos} err={err_cos}"
            );

            if expected_sin.abs() >= HALF {
                max_rel_sin = max_rel_sin.max(err_sin / expected_sin.abs());
            }
            if expected_cos.abs() >= HALF {
                max_rel_cos = max_rel_cos.max(err_cos / expected_cos.abs());
            }
            if expected_sin.abs() < HALF {
                max_floor = max_floor.max(err_sin);
            }
            if expected_cos.abs() < HALF {
                max_floor = max_floor.max(err_cos);
            }

            let next = x + STEP;
            if next <= x {
                break;
            }
            x = next;
        }

        assert!(
            max_rel_sin <= TARGET_REL,
            "max relative error of sin: {max_rel_sin}, target {TARGET_REL}"
        );
        assert!(
            max_rel_cos <= TARGET_REL,
            "max relative error of cos: {max_rel_cos}, target {TARGET_REL}"
        );
        assert!(
            max_floor <= FLOOR,
            "max absolute error below the floor: {max_floor}, floor {FLOOR}"
        );
    }

    #[test]
    fn monotone_across_the_base_interval() {
        // sin increases on the whole base interval [-pi/4, pi/4].
        let mut x = -QUARTER_PI_Q32;
        let mut prev = fixed_sin_cos(x).expect("left end is inside the window");
        while x <= QUARTER_PI_Q32 - STEP {
            x += STEP;
            let cur = fixed_sin_cos(x).expect("sampled inside the window");
            assert!(
                cur.sin >= prev.sin,
                "sin not increasing at x={x}: {} -> {}",
                prev.sin,
                cur.sin
            );
            prev = cur;
        }

        // cos decreases on [0, pi/4] and increases on [-pi/4, 0].
        let mut x = 0;
        let mut prev = fixed_sin_cos(x).expect("inside the window");
        while x <= QUARTER_PI_Q32 - STEP {
            x += STEP;
            let cur = fixed_sin_cos(x).expect("sampled inside the window");
            assert!(
                cur.cos <= prev.cos,
                "cos not decreasing at x={x}: {} -> {}",
                prev.cos,
                cur.cos
            );
            prev = cur;
        }
        let mut x = -QUARTER_PI_Q32;
        let mut prev = fixed_sin_cos(x).expect("inside the window");
        while x <= -STEP {
            x += STEP;
            let cur = fixed_sin_cos(x).expect("sampled inside the window");
            assert!(
                cur.cos >= prev.cos,
                "cos not increasing at x={x}: {} -> {}",
                prev.cos,
                cur.cos
            );
            prev = cur;
        }
    }
}
