//! The `F32` load-boundary conversion: Q32.32 norm gains and attention biases.

/// A refused `F32` norm-gain or attention-bias conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32Error {
    /// The value was `NaN` or infinite.
    NonFinite,
    /// The rounded value does not fit `i64`.
    OutOfRange,
    /// The vector payload was shorter than the declared length.
    Truncated {
        /// Bytes needed.
        need: usize,
        /// Bytes supplied.
        have: usize,
    },
}

impl core::fmt::Display for F32Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            F32Error::NonFinite => write!(f, "non-finite f32 gain or bias"),
            F32Error::OutOfRange => write!(f, "f32 gain or bias out of Q32.32 range"),
            F32Error::Truncated { need, have } => {
                write!(f, "f32 vector needs {need} bytes, has {have}")
            }
        }
    }
}

/// The Q32.32 image of an `F32` norm gain or attention bias: round-to-nearest at `2^-32`.
///
/// `v` denotes `round(v · 2^32)`. Unlike a weight row, the domain here is the per-channel norm
/// gain and the Qwen2 attention bias, whose magnitudes are bounded near one at load; the
/// conversion is exact for every `f32` that is an integer multiple of `2^-32`, and otherwise the
/// single documented round-to-nearest is the Q32.32 epilogue. `NaN` and `Inf` refuse.
///
/// # Errors
///
/// Returns [`F32Error::NonFinite`] for `NaN`/`Inf` and [`F32Error::OutOfRange`] when the rounded
/// value does not fit `i64`. No value is clamped.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the finite range guard above `as i64` keeps the cast inside the Q32.32 integer domain"
)]
pub fn f32_to_fixed(v: f32) -> Result<i64, F32Error> {
    if !v.is_finite() {
        return Err(F32Error::NonFinite);
    }
    let q = (f64::from(v) * 4_294_967_296.0).round();
    if !(q.is_finite() && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&q))
    {
        return Err(F32Error::OutOfRange);
    }
    Ok(q as i64)
}
