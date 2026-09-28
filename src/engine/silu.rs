//! The scalar `SiLU` of the dense FFN (port-map row 6), isolated from the projection kernels so
//! the Charon/Aeneas extraction can root it without pulling `matvec`, `dequant` or `EngineError`.
//!
//! `engine::ffn::silu_fixed` is the public entry point and delegates here; `proof/extract/lib.rs`
//! includes this file as its own module and roots the real body.

use crate::engine::value::div_round_nearest;
use crate::engine::ONE;
use crate::numeric::fixed_exp::fixed_exp;

/// A refused fixed-point `SiLU` evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SiluError {
    /// A [`fixed_exp`] argument on the exact `SiLU` path left its window `[-16·ONE, 21·ONE]`.
    /// Values outside the open interval `(-21·ONE, 16·ONE)` saturate instead, so this is
    /// unreachable for a well-formed call and is retained for the in-window path.
    ExpOutOfRange,
    /// The `SiLU` denominator `ONE + fixed_exp(-v)` is zero, so the division is undefined.
    ZeroDenominator,
    /// An `i128` product or an `i64` narrowing left its range.
    Overflow,
}

/// Fixed-point `SiLU` (port-map row 6): `silu(v) = v / (1 + exp(-v))` in Q32.32.
///
/// This is the scalar the fold `moe.rs::swiglu` applies to the gate projection. `v` and the result
/// are Q32.32; `exp(-v)` is [`fixed_exp`] and the quotient is the single round-to-nearest division
/// `round(|v|·2^32 / (2^32 + exp(-v)))` with the sign reapplied (ties away from zero). The exact
/// path is the open interval `-21·ONE < v < 16·ONE`, exactly the interval keeping `-v` strictly
/// inside the `fixed_exp` window `[-16·ONE, 21·ONE]`. Outside it the result saturates at the
/// underflow (`v <= -21·ONE`, return `0`) and overflow (`v >= 16·ONE`, return `v`) ends; the
/// in-window quotient is unchanged. `silu(0) = 0` exactly.
///
/// The `Result` is written with explicit `match` arms rather than `?` and with a range check rather
/// than `i64::try_from`, so the early returns and the narrowing appear directly in the extracted
/// term; the arithmetic and the refusals are unchanged.
///
/// # Errors
///
/// Returns [`SiluError::ExpOutOfRange`] only if `fixed_exp` refuses on the exact path (unreachable
/// for `v` in `(-21·ONE, 16·ONE)`), [`SiluError::ZeroDenominator`] when the denominator is zero, and
/// [`SiluError::Overflow`] when the `i128` product or the `i64` narrowing leaves its range. No float
/// appears on the executing path.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::manual_let_else,
    reason = "the match/early-return forms are written explicitly so the early returns appear in \
              the extracted term; let-else and `?` desugar through constructs the pinned fragment \
              does not model. The one narrowing `as i64` is range-checked against i64::MAX/MIN \
              immediately before it, and the widening `as i128` forms are total"
)]
pub fn silu_fixed(v: i64) -> Result<i64, SiluError> {
    const LOWER: i64 = -21 * ONE;
    const UPPER: i64 = 16 * ONE;
    // Overflow saturation, `v >= 16·ONE`. Here `silu(v) = v/(1 + exp(-v))`, so
    // `|silu(v) - v| = |v|·exp(-v)/(1 + exp(-v)) <= |v|·exp(-v) <= 16·exp(-16) ≈ 1.8e-6`, the bound
    // decreasing in `v`. Return `v` exactly.
    if v >= UPPER {
        return Ok(v);
    }
    // Underflow saturation, `v <= -21·ONE`. Here `|silu(v)| = |v|·exp(v)/(1 + exp(v)) <= |v|·exp(v)
    // <= 21·exp(-21) ≈ 1.6e-8`, the bound decreasing as `v` falls. Return `0`.
    if v <= LOWER {
        return Ok(0);
    }
    // Exact path, `-21·ONE < v < 16·ONE`: `|v| < 21·ONE`, so `-v` cannot overflow and stays
    // strictly inside `fixed_exp`'s domain.
    let w = -v;
    let e = match fixed_exp(w) {
        Ok(e) => e,
        Err(_) => return Err(SiluError::ExpOutOfRange),
    };
    let den = match ONE.checked_add(e) {
        Some(den) => den,
        None => return Err(SiluError::Overflow),
    };
    if den <= 0 {
        return Err(SiluError::ZeroDenominator);
    }
    let mag = if v < 0 { -i128::from(v) } else { i128::from(v) };
    let num = match mag.checked_mul(i128::from(ONE)) {
        Some(num) => num,
        None => return Err(SiluError::Overflow),
    };
    let q = div_round_nearest(num, i128::from(den));
    let signed = if v < 0 { -q } else { q };
    if signed > i64::MAX as i128 || signed < i64::MIN as i128 {
        return Err(SiluError::Overflow);
    }
    Ok(signed as i64)
}
