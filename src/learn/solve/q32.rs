//! The `f64` ↔ Q32.32 `i64` boundary the solve primitives share.
//!
//! A real `x` is the grid word `round(x·2³²)` and a word `v` is `v/2³²`. Every caller that hands an
//! `f64` fold quantity to [`super::chol`], [`super::syrk`] or [`super::gemm`] converts here, so the
//! range test, the rounding and the power-of-two scaling rule exist once.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the `f64 as i64` in `to_grid` is the Q32.32 boundary rounding, range-checked \
              immediately before it; the `f64 as i32` in `panel_shift` and `fill_scale` is an \
              exponent clamped to a small range; the `i64 as f64` in `from_grid` is the \
              documented grid-to-real map"
)]

/// The `Q32.32` scale `2³²`: the grid word of a real `x` is `round(x·2³²)` and a word `v` is `v/2³²`.
pub const GRID_SCALE: f64 = 4_294_967_296.0;

/// The `wsyrk` input bound in the reals: the primitive's documented bound is `|v| < 2³⁶` in raw
/// `Q32.32`, i.e. `2³⁶/2³² = 16` on the `f64` side.
pub const INPUT_BOUND: f64 = 16.0;

/// The grid's own real range bound `2³¹`: `|x| < 2³¹` keeps `round(x·2³²)` inside `i64`. The bound
/// for a value the primitives consume without a tighter input bound — a metric factor or a
/// [`super::chol::chol_solve`] operand.
pub const GRID_BOUND: f64 = 2_147_483_648.0;

/// `round(x·2³²)` as a `Q32.32` word, or `None` when `x` is not finite or `|x|` reaches `bound`.
/// `bound` is [`INPUT_BOUND`] for a `wsyrk` input and [`GRID_BOUND`] for a value the solve consumes.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn to_grid(x: f64, bound: f64) -> Option<i64> {
    let q = x * GRID_SCALE;
    if !x.is_finite() || x.abs() >= bound || q.abs() >= 9_223_372_036_854_775_808.0 {
        return None;
    }
    Some(q.round() as i64)
}

/// `v/2³²`, the `Q32.32` word as an `f64`.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn from_grid(v: i64) -> f64 {
    v as f64 / GRID_SCALE
}

/// The least `k ≥ 0` with `max·2⁻ᵏ ≤ 8` — the exact power-of-two shift that brings a panel below the
/// `wsyrk` input bound `16` with head-room; `k = 0` when `max ≤ 8`.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
pub fn panel_shift(max: f64) -> i32 {
    if max <= 8.0 {
        0
    } else {
        (max / 8.0).log2().ceil() as i32
    }
}

/// `2^e` for an integer `e` (exact, by the IEEE exponent field), or `None` outside `f64`'s finite
/// normal range `[-1022, 1023]`.
fn pow2(e: i32) -> Option<f64> {
    if (-1022..=1023).contains(&e) {
        Some(f64::from_bits(u64::try_from(e + 1023).ok()? << 52))
    } else {
        None
    }
}

/// The exact power of two `s` that scales a magnitude `max_mag` into `(2³⁰, 2³¹)`, the fill of the
/// grid range, so the scaled entries keep the most significant bits available; `1.0` when
/// `max_mag ≤ 0`. `max_mag` is a finite non-negative magnitude.
///
/// Two operands filled by separate powers of two `s_a`, `s_b` solve to `X_q = (s_b/s_a)·X`, which
/// lands the quotient near the top of the grid range rather than at the bottom; the caller undoes it
/// as `X = (s_a/s_b)·X_q`.
///
/// The `log2` estimate is corrected by exact halvings and doublings; the exponent is capped so
/// `s·2³²` stays a finite `f64`, which under-fills only for a magnitude below `2⁻⁸⁶⁹`.
///
/// # Complexity
///
/// `O(1)`: the correction loops run at most twice after the `log2` estimate.
#[must_use]
pub fn fill_scale(max_mag: f64) -> f64 {
    const MAX_VALUE: f64 = 2_147_483_647.0;
    if max_mag <= 0.0 {
        return 1.0;
    }
    let estimate = (MAX_VALUE / max_mag).log2();
    let exponent = if estimate.is_finite() {
        estimate.clamp(-1022.0, 900.0).floor() as i32
    } else {
        900
    };
    let mut scale = pow2(exponent).unwrap_or(1.0);
    while max_mag * scale > MAX_VALUE {
        scale *= 0.5;
    }
    while scale < 1.0e270 && max_mag * (scale * 2.0) <= MAX_VALUE {
        scale *= 2.0;
    }
    scale
}
