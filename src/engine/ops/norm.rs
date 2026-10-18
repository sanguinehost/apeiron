//! Fixed-point `RMSNorm` (port-map row 2).

use crate::engine::num::value::{div_round_nearest, round_shift_32};
use crate::engine::FIXED_FRAC_BITS;
use crate::numeric::fixed_rsqrt::{fixed_rsqrt, RsqrtError};

/// A refused fixed-point `RMSNorm` (port-map row 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormError {
    /// The input row is empty, so the mean square is undefined.
    EmptyRow,
    /// The mean square plus epsilon, an accumulator, or an output fell outside its admitted
    /// fixed-point window.
    OutOfRange,
}

impl From<RsqrtError> for NormError {
    fn from(_: RsqrtError) -> Self {
        NormError::OutOfRange
    }
}

/// Fixed-point `RMSNorm` (port-map row 2): `y_i = x_i * fixed_rsqrt(mean(x^2) + eps_fixed)`.
///
/// `x` and `eps_fixed` are Q32.32. The mean square is `round(sum x_i^2 / (n * 2^32))` in Q32.32:
/// the sum of squares is accumulated in `i128` (an `i64` square overflows `i64`), then divided by
/// `n * 2^32` with one round-to-nearest. The output applies the documented value-path epilogue
/// `round(x_i * r / 2^32)`, `r` the Q32.32 `fixed_rsqrt` result. The per-channel gain of the
/// reference `cpu_rms` is not part of this kernel; it composes as a separate elementwise Q32.32
/// product.
///
/// # Errors
///
/// Returns [`NormError::EmptyRow`] for an empty row and [`NormError::OutOfRange`] when the argument
/// to `fixed_rsqrt` leaves `[2^-20, 2^20]`, when an `i128` accumulation leaves its range, or when an
/// output does not fit `i64`. No input is clamped and no operation wraps.
///
/// The two accumulation loops carry a `bool` failure flag and stop on the first refusal, so no
/// `return` appears inside a loop; the refusals, the `i128` arithmetic, the `fixed_rsqrt` call and
/// the single round-to-nearest epilogue are the same operations in the same order. The pinned
/// Charon/Aeneas fragment lowers an early return inside one of two sequential loops to an
/// unsupported control-flow pattern, so the explicit flag form is the extractable one. The one
/// `as i64` narrowing is range-checked against `i64::MAX`/`MIN` immediately before it; `TryFrom`
/// and `?` inside a loop each emit an unmodeled external (an `axiom` in the emitted Lean).
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    reason = "plain i64/i128 comparisons and `as` casts after an explicit range check extract \
              through Charon/Aeneas, while TryFrom and a return inside a loop emit an unmodeled \
              external or an unsupported control-flow pattern the fail-closed audit rejects. The \
              explicit `match` early returns are written out because let-else desugars through a \
              construct the pinned fragment does not model"
)]
pub fn rms_norm_fixed(row: &[i64], eps_fixed: i64) -> Result<Vec<i64>, NormError> {
    let n = row.len();
    if n == 0 {
        return Err(NormError::EmptyRow);
    }
    let mut sum_sq: i128 = 0;
    let mut i = 0usize;
    let mut sum_ok = true;
    while i < n && sum_ok {
        let v = i128::from(row[i]);
        match v.checked_mul(v) {
            Some(sq) => match sum_sq.checked_add(sq) {
                Some(total) => sum_sq = total,
                None => sum_ok = false,
            },
            None => sum_ok = false,
        }
        i += 1;
    }
    if !sum_ok {
        return Err(NormError::OutOfRange);
    }
    // `n >= 0`, so `checked_shl(FIXED_FRAC_BITS)` and `checked_mul(2^FIXED_FRAC_BITS)` accept the
    // same set; the multiply is the form the pinned fragment models (`I128.checked_shl` is an
    // unmodeled external the fail-closed audit rejects).
    let n_i128 = n as i128;
    let den = match n_i128.checked_mul(1_i128 << FIXED_FRAC_BITS) {
        Some(den) => den,
        None => return Err(NormError::OutOfRange),
    };
    let mean = div_round_nearest(sum_sq, den);
    let arg = match mean.checked_add(i128::from(eps_fixed)) {
        Some(arg) => arg,
        None => return Err(NormError::OutOfRange),
    };
    if arg > i64::MAX as i128 || arg < i64::MIN as i128 {
        return Err(NormError::OutOfRange);
    }
    let arg_i64 = arg as i64;
    let root = match fixed_rsqrt(arg_i64) {
        Ok(root) => i128::from(root),
        Err(_) => return Err(NormError::OutOfRange),
    };
    let mut out: Vec<i64> = Vec::with_capacity(n);
    let mut i = 0usize;
    let mut out_ok = true;
    while i < n && out_ok {
        match i128::from(row[i]).checked_mul(root) {
            Some(prod) => {
                let normed = round_shift_32(prod);
                if normed > i64::MAX as i128 || normed < i64::MIN as i128 {
                    out_ok = false;
                } else {
                    out.push(normed as i64);
                }
            }
            None => out_ok = false,
        }
        i += 1;
    }
    if !out_ok {
        return Err(NormError::OutOfRange);
    }
    Ok(out)
}
