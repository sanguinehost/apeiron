//! Fixed-point `rsqrt` in `i64` Q32.32 for the integer executing path.
//!
//! An `i64` `v` denotes the real number `v / 2^32`; `ONE` is `1.0`. [`fixed_rsqrt`] is total over
//! the window `[2^-20, 2^20]`, i.e. `ONE >> 20 <= x <= ONE << 20`, and returns
//! [`RsqrtError::OutOfRange`] outside it, including for `x <= 0`; it never wraps or saturates.
//!
//! The algorithm is integer-only and scales before the root. For `x` in the window,
//! `s = isqrt(x << 64)` is `2^32 * sqrt(x)` rounded down, so `s >= 2^38` and the root rounding
//! error is below `2^-38` relative. The Q32.32 result is `round(2^80 / s)` with round-to-nearest:
//! `2^80 / s` denotes `2^48 / sqrt(x) = rsqrt(x / 2^32) * 2^32`. The quotient is at most `2^42`
//! (at `x = ONE >> 20`) and at least `2^22` (at `x = ONE << 20`), so it fits `i64`; the final
//! `as i64` is guarded by an explicit `i64::MAX` check.
//!
//! Rounding policy: the root is floored, the reciprocal rounds to nearest. The dominant term is
//! the one-unit quotient rounding, at most `0.5 / 2^22 = 1.19e-7` relative at the high end of the
//! window, below the documented `2^-20` target.
//!
//! Accuracy, measured by the dense-sample unit test over the window. The asserted per-sample
//! bound is `relative error <= 2^-20`; the measured maximum over 43,900 samples is `1.164e-7`.

/// The Q32.32 scale: `ONE` denotes `1.0`, so one unit is `2^-32`.
const ONE: i64 = 1 << 32;

/// The refusal of a [`fixed_rsqrt`] argument outside its documented window.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RsqrtError {
    /// The argument lies outside `[ONE >> 20, ONE << 20]`, or is non-positive.
    OutOfRange,
}

/// Floor of the integer square root of a positive `i128`.
///
/// Bit-by-bit restoring square root (Warren, *Hacker's Delight*, 2nd ed., §11-1, `isqrt`): one
/// iteration per two bits of `n`, at most 64 here since the largest `n` is `2^116` from the
/// largest window argument. Temporal cost is O(64) `i128` adds, subtracts and shifts, O(1) space.
pub(crate) fn isqrt(n: i128) -> i128 {
    let mut rem = n;
    let mut res: i128 = 0;
    let mut bit: i128 = 1_i128 << 126;
    while bit > rem {
        bit >>= 2;
    }
    while bit != 0 {
        let step = res + bit;
        if rem >= step {
            rem -= step;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    res
}

/// Evaluate `rsqrt(x)` in Q32.32 by scaling the radicand before the integer square root.
///
/// The result `v` denotes `v / 2^32 = 1 / sqrt(x / 2^32)`. The arithmetic is exact integer
/// arithmetic with an explicit range check and one round-to-nearest quotient; see the module
/// documentation for the algorithm and the measured error.
///
/// # Errors
///
/// Returns [`RsqrtError::OutOfRange`] when `x` is outside `[ONE >> 20, ONE << 20]`.
#[allow(
    clippy::manual_range_contains,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    reason = "plain i64 comparisons and `as` casts after an explicit range check extract through \
              Charon/Aeneas, while RangeInclusive::contains and TryFrom each emit an unmodeled \
              external (an `axiom` in the emitted Lean) that the fail-closed audit rejects"
)]
pub fn fixed_rsqrt(x: i64) -> Result<i64, RsqrtError> {
    const LOWER: i64 = ONE >> 20;
    const UPPER: i64 = ONE << 20;

    if x < LOWER || x > UPPER {
        return Err(RsqrtError::OutOfRange);
    }

    // s = isqrt(x * 2^64) = 2^32 * sqrt(x) rounded down. The largest radicand is 2^116, inside
    // i128; the result lies in [2^38, 2^58].
    let scaled = i128::from(x) << 64;
    let s = isqrt(scaled);

    // out = round(2^80 / s): the half-unit bias gives round-to-nearest, ties up.
    let num = (1_i128 << 80) + (s / 2);
    let out = num / s;

    if out > i64::MAX as i128 {
        return Err(RsqrtError::OutOfRange);
    }
    Ok(out as i64)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        reason = "the f64 oracle and the Q32.32 result are compared in floating point; sample \
                  magnitudes exceed 2^53, so the cast is lossy by construction and the tolerance \
                  absorbs it"
    )]

    use super::{fixed_rsqrt, RsqrtError, ONE};

    /// `2^-20`, the documented relative-error target.
    const TARGET_REL: f64 = 9.536_743_164_062_5e-7;

    const LOWER: i64 = ONE >> 20;
    const UPPER: i64 = ONE << 20;

    fn to_real(v: i64) -> f64 {
        v as f64 / 4_294_967_296.0
    }

    /// Relative error of a result at `x` against the f64 oracle `1 / sqrt(x / 2^32)`.
    fn rel_error(x: i64) -> (f64, f64) {
        let result = fixed_rsqrt(x).expect("sampled inside the window");
        let expected = 1.0 / to_real(x).sqrt();
        let got = to_real(result);
        let err = (got - expected).abs();
        (err / expected, expected)
    }

    #[test]
    fn exact_identities_and_boundaries() {
        assert_eq!(fixed_rsqrt(ONE), Ok(ONE));
        assert_eq!(fixed_rsqrt(4 * ONE), Ok(ONE / 2));
        assert_eq!(fixed_rsqrt(ONE / 4), Ok(2 * ONE));

        assert!(fixed_rsqrt(LOWER).is_ok());
        assert!(fixed_rsqrt(UPPER).is_ok());
        assert_eq!(fixed_rsqrt(LOWER - 1), Err(RsqrtError::OutOfRange));
        assert_eq!(fixed_rsqrt(UPPER + 1), Err(RsqrtError::OutOfRange));
        assert_eq!(fixed_rsqrt(0), Err(RsqrtError::OutOfRange));
        assert_eq!(fixed_rsqrt(-1), Err(RsqrtError::OutOfRange));
        assert_eq!(fixed_rsqrt(i64::MIN), Err(RsqrtError::OutOfRange));
        assert_eq!(fixed_rsqrt(i64::MAX), Err(RsqrtError::OutOfRange));
    }

    #[test]
    fn accuracy_on_a_dense_grid() {
        let mut max_rel = 0.0f64;

        // Geometric sweep: ~1024 samples per octave across the 40-octave window.
        let mut x = LOWER;
        while x <= UPPER {
            let (rel, expected) = rel_error(x);
            assert!(
                rel <= TARGET_REL,
                "x={x} expected={expected} relative={rel} target={TARGET_REL}"
            );
            max_rel = max_rel.max(rel);
            let step = (x / 1024).max(1);
            let next = x + step;
            if next <= x {
                break;
            }
            x = next;
        }

        // Linear sweep over the four octaves around ONE, where the quotient rounding dominates.
        let mut x = ONE / 4;
        while x <= 4 * ONE {
            let (rel, expected) = rel_error(x);
            assert!(
                rel <= TARGET_REL,
                "x={x} expected={expected} relative={rel} target={TARGET_REL}"
            );
            max_rel = max_rel.max(rel);
            x += ONE / 4096;
        }

        assert!(
            max_rel <= TARGET_REL,
            "max relative error over the dense grid: {max_rel}"
        );
        assert!(max_rel > 0.0, "the grid measured no error at all");
    }

    #[test]
    fn monotone_decreasing_on_a_sample() {
        let mut x = LOWER;
        let mut prev = fixed_rsqrt(x).expect("left end is inside the window");
        while x <= UPPER {
            let step = (x / 1024).max(1);
            let next = x + step;
            if next <= x {
                break;
            }
            x = next;
            if x > UPPER {
                x = UPPER;
            }
            let cur = fixed_rsqrt(x).expect("sampled inside the window");
            assert!(cur <= prev, "not monotone at x={x}: {prev} -> {cur}");
            prev = cur;
            if x == UPPER {
                break;
            }
        }
    }
}
