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
//! **Range reduction.** `k = round(x / (pi/2))` is computed in Q64 as
//! `q_rne(x << 32, HALF_PI_Q64)`, with `HALF_PI_Q64 = round(pi/2 * 2^64)`, and the remainder
//! `r = x - k * pi/2` is rounded once into Q32.32. The quotient's `1/2` rounding puts `|r|` within
//! `pi/4` of zero; the Q64 constant's own rounding, times `|k| <= 2608` over the window
//! (`4096 * 2 / pi`), adds at most `2^-34` of a Q32.32 unit, and the final rounding `1/2`, so
//! `|r| <= pi/4 + 2^-32`.
//!
//! **Quadrature.** `k mod 4` selects `(sin r, cos r)`, `(cos r, -sin r)`, `(-sin r, -cos r)` or
//! `(-cos r, sin r)`; the sign changes are `i64` negations bounded by one.
//!
//! **Series.** On `|r| <= pi/4 + 2^-32` the Taylor series
//! `sin r = r - r^3/3! + ... + r^13/13!` and `cos r = 1 - r^2/2! + ... + r^12/12!` are summed
//! with the exact factorial divisors; the first omitted terms are `r^15/15! <= 2.1e-14` and
//! `r^14/14! <= 3.9e-13`. Per step, round to nearest, ties to even, at `2^-32`.
//!
//! **Accuracy**, measured by the dense-sample unit test over the whole window. The series
//! truncation is below `4e-13`; the per-term rounding dominates. Each `sin` term `r^(2i+1)` costs
//! `i + 1` multiplications and one exact-factorial division, each rounded once, so the six terms
//! and the `i128` sum carry at most `16` Q32.32 units of absolute error; the asserted bound is
//! `|result / 2^32 - f(x / 2^32)| <= 16 * 2^-32`. The measured maxima over the dense grid are
//! `1.10e-9` relative for outputs above `1/2` and `5.76e-10` absolute.
//!
//! **`atan2`.** [`fixed_atan2`] shares the carrier and is total on every `i64` pair whose
//! magnitudes are representable: `atan r` for `r in [0, 1]` is reduced twice by
//! `atan r = 2*atan(r / (1 + sqrt(1 + r^2)))` to `t <= tan(pi/16)`, summed through `t^19` (first
//! omitted `t^21/21 < 2^-68` at `t = 0.199`), scaled by four and placed in the quadrant from the
//! signs and the `|x| < |y|` swap. The quotient is [`div_rne`] and the root is the floored integer
//! [`isqrt`], the same words the `sin`/`cos` reduction uses.

use super::fixed_ln::div_rne;
use super::fixed_rsqrt::isqrt;

/// The Q32.32 scale: `ONE` denotes `1.0`, so one unit is `2^-32`.
const ONE: i64 = 1 << 32;

/// `round(2pi * 2^32)`.
const TWO_PI_Q32: i64 = 26_986_075_409;

/// `round(pi * 2^32)`.
const PI_Q32: i64 = 13_493_037_705;

/// `round(pi/2 * 2^32)`.
const HALF_PI_Q32: i64 = 6_746_518_852;

/// `round(pi/2 * 2^64)`, the Q64 half-turn the `sin`/`cos` range reduction divides by.
const HALF_PI_Q64: i128 = 28_976_077_832_308_490_240;

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

/// `?` in the extraction-fragment form: the expanded `match` is the term Charon sees; `?`
/// desugars through `Try::branch`, which the pinned fragment does not model.
macro_rules! out {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return Err(TrigError::OutOfRange),
        }
    };
}

/// `num / den` rounded to nearest, ties to even; `None` when `den <= 0`, the doubled remainder
/// leaves `i128`, or the rounding step leaves the grid.
///
/// The extraction-fragment form of [`div_rne`]: the shared body's `i128::unsigned_abs` has no
/// model in `proof/patches/aeneas-lean-model.patch` (only the `i64` form does), so calling it from
/// the extracted `fixed_sin_cos` path would declare the unlisted axiom
/// `core.num.I128.unsigned_abs`. The two magnitude comparisons here are plain `i128` comparisons
/// of `2r` with `±den`, which the fragment models. `fixed_atan2` keeps the shared [`div_rne`]; it
/// is not an extraction root.
#[allow(
    clippy::question_mark,
    clippy::manual_let_else,
    clippy::bool_to_int_with_if,
    reason = "the explicit `match` is the pinned extraction fragment's form of `?` and `let .. \
              else`, and the `if`-returned `0/1` step keeps the rounding branch visible to that \
              fragment instead of `i128::from(bool)`"
)]
fn q_rne(num: i128, den: i128) -> Option<i128> {
    if den <= 0 {
        return None;
    }
    let q = num / den;
    let r = num % den;
    if r == 0 {
        return Some(q);
    }
    let twice = match r.checked_mul(2) {
        Some(v) => v,
        None => return None,
    };
    let step: i128 =
        if twice > den || twice < -den || ((twice == den || twice == -den) && q % 2 != 0) {
            1
        } else {
            0
        };
    Some(if r > 0 { q + step } else { q - step })
}

/// `p / f` in Q32.32, `f` an exact factorial divisor, rounded once; `None` off the grid.
fn div_fac(p: i64, f: i64) -> Option<i64> {
    match q_rne(i128::from(p), i128::from(f)) {
        Some(v) => narrow(v),
        None => None,
    }
}

/// `(sin r, cos r)` from the Taylor series through `r^13/13!` and `r^12/12!`, each term divided
/// by its exact factorial; `r` is Q32.32 with `|r| <= pi/4 + 2^-32`. `O(1)` time and space:
/// twelve rounded products, twelve divisions and thirteen `i128` sums.
#[allow(
    clippy::question_mark,
    clippy::many_single_char_names,
    reason = "the `out!` macro expands to the `match` form the pinned Charon/Aeneas extraction \
              fragment models, in place of `?`, and the single-letter names are the series symbols"
)]
fn sin_cos_series(r: i64) -> Result<SinCos, TrigError> {
    let r2 = out!(q_mul(r, r));
    let mut p = out!(q_mul(r, r2)); // r^3
    let mut q = r2; // r^2
    let (mut s, mut c) = (i128::from(r), i128::from(ONE));
    s -= i128::from(out!(div_fac(p, 6)));
    c -= i128::from(out!(div_fac(q, 2)));
    q = out!(q_mul(q, r2)); // r^4
    p = out!(q_mul(p, r2)); // r^5
    s += i128::from(out!(div_fac(p, 120)));
    c += i128::from(out!(div_fac(q, 24)));
    q = out!(q_mul(q, r2)); // r^6
    p = out!(q_mul(p, r2)); // r^7
    s -= i128::from(out!(div_fac(p, 5_040)));
    c -= i128::from(out!(div_fac(q, 720)));
    q = out!(q_mul(q, r2)); // r^8
    p = out!(q_mul(p, r2)); // r^9
    s += i128::from(out!(div_fac(p, 362_880)));
    c += i128::from(out!(div_fac(q, 40_320)));
    q = out!(q_mul(q, r2)); // r^10
    p = out!(q_mul(p, r2)); // r^11
    s -= i128::from(out!(div_fac(p, 39_916_800)));
    c -= i128::from(out!(div_fac(q, 3_628_800)));
    q = out!(q_mul(q, r2)); // r^12
    p = out!(q_mul(p, r2)); // r^13
    s += i128::from(out!(div_fac(p, 6_227_020_800)));
    c += i128::from(out!(div_fac(q, 479_001_600)));
    Ok(SinCos {
        sin: out!(narrow(s)),
        cos: out!(narrow(c)),
    })
}

/// Evaluate `sin(x)` and `cos(x)` in Q32.32 by half-turn range reduction and Taylor polynomials.
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
    clippy::question_mark,
    reason = "plain i64/i128 comparisons and `as` casts after an explicit range check extract \
              through Charon/Aeneas, while RangeInclusive::contains, TryFrom and unsigned_abs \
              each emit an unmodeled external (an `axiom` in the emitted Lean) that the \
              fail-closed audit rejects; `out!` expands to the `match` form in place of `?`"
)]
pub fn fixed_sin_cos(x: i64) -> Result<SinCos, TrigError> {
    const LOWER: i64 = -4096 * ONE;
    const UPPER: i64 = 4096 * ONE;

    if x < LOWER || x > UPPER {
        return Err(TrigError::OutOfRange);
    }

    // k = round(x / (pi/2)) in Q64, r = x - k*pi/2 in [-pi/4, pi/4]. |x| <= 2^44 and |k| <= 2608,
    // so both `i128` products stay below 2^76.
    let t64 = i128::from(x) << 32;
    let k = out!(q_rne(t64, HALF_PI_Q64));
    let r = out!(narrow(out!(q_rne(t64 - k * HALF_PI_Q64, 1_i128 << 32))));

    let sc = match sin_cos_series(r) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    let quarter = ((k % 4) + 4) % 4;
    Ok(match quarter {
        0 => sc,
        1 => SinCos {
            sin: sc.cos,
            cos: -sc.sin,
        },
        2 => SinCos {
            sin: -sc.sin,
            cos: -sc.cos,
        },
        _ => SinCos {
            sin: -sc.cos,
            cos: sc.sin,
        },
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

/// `v` as an `i64`, or `None` when it leaves the grid.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `as` narrowing follows an explicit `i64` bound check and extracts through \
              Charon/Aeneas, while `TryFrom` emits an unmodeled external the audit rejects"
)]
fn narrow(v: i128) -> Option<i64> {
    if v > i128::from(i64::MAX) || v < i128::from(i64::MIN) {
        None
    } else {
        Some(v as i64)
    }
}

/// `a*b` in Q32.32, rounded once to nearest, ties to even.
fn q_mul(a: i64, b: i64) -> Option<i64> {
    match q_rne(i128::from(a) * i128::from(b), i128::from(ONE)) {
        Some(v) => narrow(v),
        None => None,
    }
}

/// `a/b` in Q32.32, rounded once; `None` on a non-positive divisor or a quotient off the grid.
fn q_div(a: i64, b: i64) -> Option<i64> {
    match q_rne(i128::from(a) << 32, i128::from(b)) {
        Some(v) => narrow(v),
        None => None,
    }
}

/// The `Q32.32` angle `atan2(y, x)` in radians on `(-pi, pi]`, `0` at `(0, 0)`.
///
/// `atan r` for `r in [0, 1]` is reduced twice by `atan r = 2*atan(r / (1 + sqrt(1 + r^2)))` to
/// `t <= tan(pi/16)` and summed through `t^19`; the quadrant is restored from the signs and the
/// `|x| < |y|` swap with the rounded `PI_Q32` and `HALF_PI_Q32` words.
///
/// # Errors
///
/// Returns [`TrigError::OutOfRange`] when an argument magnitude is not representable (`i64::MIN`),
/// or a product, quotient or the `4*atan` scale leaves the grid.
///
/// # Complexity
///
/// `O(1)` time and space: two root-and-quotient reductions, nine series terms and one scale, all in
/// `i128`.
#[allow(
    clippy::manual_let_else,
    clippy::question_mark,
    clippy::many_single_char_names,
    reason = "the `out!` macro expands to the `match` form the pinned Charon/Aeneas extraction \
              fragment models, in place of `let .. else` and `?`, and the single-letter names are \
              the series symbols"
)]
pub fn fixed_atan2(y: i64, x: i64) -> Result<i64, TrigError> {
    if x == 0 && y == 0 {
        return Ok(0);
    }
    let (ax, ay) = match (x.checked_abs(), y.checked_abs()) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(TrigError::OutOfRange),
    };
    let (mut r, flip) = if ax >= ay {
        (out!(q_div(ay, ax)), false)
    } else {
        (out!(q_div(ax, ay)), true)
    };
    // atan r = 2*atan(r / (1 + sqrt(1 + r^2))), twice: r <= tan(pi/8), then r <= tan(pi/16).
    for _ in 0..2 {
        let r2 = out!(q_mul(r, r));
        let radicand = out!(ONE.checked_add(r2));
        let s = out!(narrow(isqrt(i128::from(radicand) << 32)));
        r = out!(q_div(r, out!(ONE.checked_add(s))));
    }
    let r2 = out!(q_mul(r, r));
    let (mut p, mut sum) = (r, r);
    for k in 1..=9_i64 {
        p = out!(q_mul(p, r2));
        let term = out!(div_rne(i128::from(p), i128::from(2 * k + 1)).and_then(narrow));
        sum = out!(if k % 2 == 0 {
            sum.checked_add(term)
        } else {
            sum.checked_sub(term)
        });
    }
    let mut a = out!(sum.checked_mul(4));
    if flip {
        a = out!(HALF_PI_Q32.checked_sub(a));
    }
    if x < 0 {
        a = out!(PI_Q32.checked_sub(a));
    }
    if y < 0 {
        a = out!(a.checked_neg());
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        reason = "the f64 oracle and the Q32.32 result are compared in floating point; sample \
                  magnitudes exceed 2^53, so the cast is lossy by construction and the tolerance \
                  absorbs it"
    )]

    use super::{fixed_atan2, fixed_sin_cos, SinCos, TrigError, HALF_PI_Q32, ONE, PI_Q32};

    /// `round(pi/4 * 2^32)`, the base-interval endpoint.
    const QUARTER_PI_Q32: i64 = 3_373_259_426;

    /// The derived absolute error bound: sixteen `2^-32` units, the next power of two above the
    /// per-term rounding count (six terms, each at most `(i + 2)/2` rounded units, plus the `i128`
    /// sums). The series truncation itself is below `4e-13`.
    const FLOOR: f64 = 16.0 / 4_294_967_296.0;

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
            (got - expected).abs() <= FLOOR,
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
                err_sin <= FLOOR,
                "sin x={x} expected={expected_sin} got={got_sin} err={err_sin}"
            );
            assert!(
                err_cos <= FLOOR,
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
            max_rel_sin <= 2.0 * FLOOR,
            "max relative error of sin: {max_rel_sin}, bound {}",
            2.0 * FLOOR
        );
        assert!(
            max_rel_cos <= 2.0 * FLOOR,
            "max relative error of cos: {max_rel_cos}, bound {}",
            2.0 * FLOOR
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

    /// The `atan2` agreement bound, `128` units of `2^-32`, at the next power of two above the
    /// derivation: each of the two `atan r = 2*atan(r/(1+sqrt(1+r^2)))` reductions carries a floored
    /// root (1 unit) and two quotients plus one sum (2 units), the nine series terms carry one
    /// rounding each, and the `*4` scale with one `pi/2` or `pi` subtraction adds 2 — `4*(3+3+9+2)
    /// = 68` units.
    const ATAN2_ABS: f64 = 128.0 / 4_294_967_296.0;

    fn atan2_real(v: i64) -> f64 {
        v as f64 / 4_294_967_296.0
    }

    #[test]
    fn atan2_matches_f64_within_the_derived_bound() {
        assert_eq!(
            [
                fixed_atan2(0, 0),
                fixed_atan2(ONE, 0),
                fixed_atan2(-ONE, 0),
                fixed_atan2(0, -ONE),
            ],
            [Ok(0), Ok(HALF_PI_Q32), Ok(-HALF_PI_Q32), Ok(PI_Q32)]
        );
        // Quadrant grid at a 1/16 radian step: every sign pattern and both `|x| < |y|` arms.
        let (step, mut worst) = (ONE / 16, 0.0_f64);
        for iy in -48_i64..=48 {
            for ix in -48_i64..=48 {
                let (y, x) = (iy * step, ix * step);
                let got = atan2_real(fixed_atan2(y, x).expect("magnitudes are representable"));
                let want = atan2_real(y).atan2(atan2_real(x));
                worst = worst.max((got - want).abs());
            }
        }
        assert!(worst <= ATAN2_ABS, "worst |atan2 error| = {worst}");
    }

    #[test]
    fn atan2_refuses_only_unrepresentable_magnitudes() {
        assert_eq!(fixed_atan2(i64::MIN, ONE), Err(TrigError::OutOfRange));
        assert_eq!(fixed_atan2(i64::MIN, i64::MIN), Err(TrigError::OutOfRange));
    }
}
