//! Fixed-point `exp` in `i64` Q32.32 for the integer executing path.
//!
//! An `i64` `v` denotes the real number `v / 2^32`; `ONE` is `1.0`. [`fixed_exp`] is total and
//! returns [`ExpError::OutOfRange`] outside the window `[-16 * ONE, 21 * ONE]`; it never wraps or
//! saturates. `21 * ONE` is the top because `exp(21) * 2^32 <= 5.66e18 < i64::MAX`; `-16 * ONE`
//! keeps the scaled result at `exp(-16) * 2^32 >= 483`, above the flush-to-zero threshold.
//!
//! The algorithm is integer-only. It computes `k = round(x / ln 2)` from the Q32 reciprocal as
//! `(x * LN2_RECIP_Q32 + 2^63) >> 64`, so `|x / ln 2 - k| <= 1/2 + 5e-9`, then
//! `r = x - k * LN2_Q32` with `|r| <= ln 2 / 2 + 4e-9` in `2^32` units. It evaluates the degree-7
//! Taylor polynomial of `exp` at `r` in Horner form over the Q32 coefficients `1 / n!`, each
//! product accumulated in `i128`. It scales by `2^k`: `k >= 0` is a checked left shift that
//! returns `Err` on `i64` overflow, `k < 0` is a round-to-nearest right shift that flushes to `0`
//! below half a unit. A shift count `>= 64` returns before any Rust shift whose count would be
//! out of range.
//!
//! Rounding policy: per step. Each `i128` product is rounded to nearest, ties away from zero,
//! before the next Horner step, and the final scale rounds once more. The degree-7 remainder
//! dominates the error: `|r|^8 / 8! <= 5.2e-9` for `|r| <= ln 2 / 2`.
//!
//! Accuracy, measured by the dense-sample unit test over the whole window. The asserted combined
//! bound `|result / 2^32 - exp(x / 2^32)| <= 1e-8 * exp(x / 2^32) + 4 / 2^32` holds. The max
//! relative error is `8.44e-9` for `x >= 0` (asserted `<= 1e-8`) and `3.43e-7` for `x >= -8`,
//! below the `2^-20` target; below `x = -8` the fixed-point grid, not the algorithm, sets the
//! relative error, since the output `483` units at `x = -16` makes half a unit `6.9e-4` relative.
//!
//! The body is split into three named helpers ([`exp_reduce`], [`exp_horner`], [`exp_scale`]) so
//! the extracted Aeneas term has call/bind boundaries rather than one flat `do` block; the
//! arithmetic and the refusals are unchanged.

/// The Q32.32 scale: `ONE` denotes `1.0`, so one unit is `2^-32`.
const ONE: i64 = 1 << 32;

/// `round(ln 2 * 2^32)`.
const LN2_Q32: i64 = 2_977_044_472;

/// `round(2^32 / ln 2) = round(log2(e) * 2^32)`.
const LN2_RECIP_Q32: i64 = 6_196_328_019;

/// The refusal of a [`fixed_exp`] argument outside its documented window.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ExpError {
    /// The argument lies outside `[-16 * ONE, 21 * ONE]`, or the Q32.32 result would overflow
    /// `i64`.
    OutOfRange,
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

/// Range reduction: check the window, then compute the quotient `k = round(x / ln 2)` and the
/// remainder `r = x - k * LN2_Q32`, both in Q32.32.
///
/// Returns `(k, r)`. The refusals are the same as [`fixed_exp`]'s: the window check, and the two
/// narrowing checks that keep `k` and `k * LN2_Q32` inside `i64`.
#[allow(
    clippy::manual_range_contains,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    reason = "plain i64/i128 comparisons and `as` casts after an explicit range check extract \
              through Charon/Aeneas, while RangeInclusive::contains, TryFrom and unsigned_abs \
              each emit an unmodeled external (an `axiom` in the emitted Lean) that the \
              fail-closed audit rejects"
)]
fn exp_reduce(x: i64) -> Result<(i64, i64), ExpError> {
    const LOWER: i64 = -16 * ONE;
    const UPPER: i64 = 21 * ONE;

    if x < LOWER || x > UPPER {
        return Err(ExpError::OutOfRange);
    }

    // k = round(x / ln 2): the product is Q64.64, so shifting by 64 with a 2^63 half rounds to
    // nearest. |x| <= 21 * 2^32 keeps the product below 2^70, well inside i128.
    let product = i128::from(x) * i128::from(LN2_RECIP_Q32);
    let shifted = (product + (1_i128 << 63)) >> 64;
    if shifted > i64::MAX as i128 || shifted < i64::MIN as i128 {
        return Err(ExpError::OutOfRange);
    }
    let k = shifted as i64;

    // r = x - k * LN2_Q32. |k| <= 31 in the window, so the product fits i64.
    let k_ln2_wide = i128::from(k) * i128::from(LN2_Q32);
    if k_ln2_wide > i64::MAX as i128 || k_ln2_wide < i64::MIN as i128 {
        return Err(ExpError::OutOfRange);
    }
    let k_ln2 = k_ln2_wide as i64;
    let r = x - k_ln2;

    Ok((k, r))
}

/// Horner form of `sum_{n=0}^{7} r^n / n!`, with the Q32 coefficients `round(2^32 / n!)`. `r` is
/// Q32.32; the accumulator is Q32.32.
#[allow(
    clippy::cast_lossless,
    reason = "the widening `i128::from` is required for the product to fit; the extraction maps \
              it to a pure conversion"
)]
fn exp_horner(r: i64) -> i128 {
    let r = i128::from(r);
    let mut acc: i128 = 852_176; // 1 / 7!
    acc = round_div_2p32(acc * r) + 5_965_232; // 1 / 6!
    acc = round_div_2p32(acc * r) + 35_791_394; // 1 / 5!
    acc = round_div_2p32(acc * r) + 178_956_971; // 1 / 4!
    acc = round_div_2p32(acc * r) + 715_827_883; // 1 / 3!
    acc = round_div_2p32(acc * r) + 2_147_483_648; // 1 / 2!
    acc = round_div_2p32(acc * r) + 4_294_967_296; // 1 / 1!
    acc = round_div_2p32(acc * r) + 4_294_967_296; // 1 / 0!

    acc
}

/// Scale the Horner accumulator `acc` by `2^k`: a checked left shift for `k >= 0`, a
/// round-to-nearest right shift (flush to `0` below half a unit) for `k < 0`. Refuses with
/// [`ExpError::OutOfRange`] when the `i64` result would overflow and when a left shift count is
/// `>= 64`.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "plain i64/i128 comparisons and `as` casts after an explicit range check extract \
              through Charon/Aeneas, while RangeInclusive::contains, TryFrom and unsigned_abs \
              each emit an unmodeled external (an `axiom` in the emitted Lean) that the \
              fail-closed audit rejects"
)]
fn exp_scale(k: i64, acc: i128) -> Result<i64, ExpError> {
    if k >= 0 {
        if k >= 64 {
            return Err(ExpError::OutOfRange);
        }
        let n = k as u32;
        let scaled = acc << n;
        if scaled > i64::MAX as i128 || scaled < i64::MIN as i128 {
            return Err(ExpError::OutOfRange);
        }
        Ok(scaled as i64)
    } else {
        let n = if k < 0 { (-k) as u64 } else { k as u64 };
        if n >= 64 {
            return Ok(0);
        }
        // Round to nearest at 2^-n, flushing to 0 when the value is below half a unit.
        let half: i128 = 1 << (n - 1);
        let rounded = (acc + half) >> n;
        if rounded > i64::MAX as i128 || rounded < i64::MIN as i128 {
            return Err(ExpError::OutOfRange);
        }
        Ok(rounded as i64)
    }
}

/// Evaluate `exp(x)` in Q32.32 by range reduction to `[-ln 2 / 2, ln 2 / 2]` and a degree-7
/// Taylor polynomial.
///
/// The result `v` denotes `v / 2^32`. The arithmetic is exact integer arithmetic with per-step
/// rounding; see the module documentation for the algorithm and the measured error.
///
/// # Errors
///
/// Returns [`ExpError::OutOfRange`] when `x` is outside `[-16 * ONE, 21 * ONE]`, or when the
/// scaled result does not fit `i64`.
#[allow(
    clippy::question_mark,
    reason = "an explicit `match` keeps the `Err` early return in the extracted term; `?` \
              desugars through `Try::branch`/`FromResidual`, which the pinned extraction \
              fragment does not model and which would emit an unmodeled external"
)]
pub fn fixed_exp(x: i64) -> Result<i64, ExpError> {
    let (k, r) = match exp_reduce(x) {
        Ok(kr) => kr,
        Err(e) => return Err(e),
    };
    let acc = exp_horner(r);
    exp_scale(k, acc)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_precision_loss,
        reason = "the f64 oracle and the Q32.32 result are compared in floating point; sample \
                  magnitudes exceed 2^53, so the cast is lossy by construction and the tolerance \
                  absorbs it"
    )]

    use super::{fixed_exp, ExpError, ONE};

    /// `2^-20`, the relative-error target for outputs above the fixed-point grid.
    const TARGET_REL: f64 = 9.536_743_164_062_5e-7;

    /// The achieved relative-error bound on `x >= 0`, asserted by the dense sample.
    const ACHIEVED_REL: f64 = 1.0e-8;

    /// Four units of `2^-32`, the absolute floor for small outputs.
    const ULP4: f64 = 4.0 / 4_294_967_296.0;

    /// The dense-sample grid step, exact in Q32.32.
    const STEP: i64 = ONE / 1024;

    fn to_real(v: i64) -> f64 {
        v as f64 / 4_294_967_296.0
    }

    #[test]
    fn zero_and_boundaries() {
        assert_eq!(fixed_exp(0), Ok(ONE));
        assert!(fixed_exp(-16 * ONE).is_ok());
        assert!(fixed_exp(21 * ONE).is_ok());
        assert_eq!(fixed_exp(-16 * ONE - 1), Err(ExpError::OutOfRange));
        assert_eq!(fixed_exp(21 * ONE + 1), Err(ExpError::OutOfRange));
        assert_eq!(fixed_exp(i64::MIN), Err(ExpError::OutOfRange));
        assert_eq!(fixed_exp(i64::MAX), Err(ExpError::OutOfRange));
    }

    #[test]
    fn accuracy_against_f64_exp() {
        let mut max_rel_nonneg = 0.0f64;
        let mut x = -16 * ONE;
        while x <= 21 * ONE {
            let result = fixed_exp(x).expect("sampled inside the window");
            let expected = to_real(x).exp();
            let got = to_real(result);
            let err = (got - expected).abs();
            let tol = ACHIEVED_REL * expected + ULP4;
            assert!(
                err <= tol,
                "x={x} expected={expected} got={got} err={err} tol={tol}"
            );
            if x >= 0 {
                max_rel_nonneg = max_rel_nonneg.max(err / expected);
            }
            if x >= -8 * ONE {
                assert!(
                    err / expected <= TARGET_REL,
                    "x={x} relative={} target={TARGET_REL}",
                    err / expected
                );
            }
            x += STEP;
        }
        assert!(
            max_rel_nonneg <= ACHIEVED_REL,
            "max relative error on x >= 0: {max_rel_nonneg}"
        );
    }

    #[test]
    fn monotone_on_a_sample() {
        let mut x = -16 * ONE;
        let mut prev = fixed_exp(x).expect("left end is inside the window");
        while x <= 21 * ONE - STEP {
            x += STEP;
            let cur = fixed_exp(x).expect("sampled inside the window");
            assert!(cur >= prev, "not monotone at x={x}: {prev} -> {cur}");
            prev = cur;
        }
    }
}
