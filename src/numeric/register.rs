//! Register stages over `R[x]/(x^2 - sigma)`: the integer executing path of
//! `Numeric.Register.cellStage`.
//!
//! A register is a slice read as consecutive blocks of `2 * span` entries; at lane `i < span` the
//! pair `(x[b + i], x[b + i + span])` is multiplied by the cell `w` in `R[x]/(x^2 - sigma)`. This
//! is the body a later task extracts and refines to `Numeric.Register.cellStage`, whose elliptic
//! case `sigma = -1` is `register_stage_normForm`.

use crate::cell::{Cell, Sigma};

/// The refusal of a [`stage_mul`] argument or pair outside its documented domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterError {
    /// `x.len()` is not a multiple of `2 * span`.
    LengthMismatch,
    /// `span` is zero, not a power of two, or at least `2^16`.
    NotPowerOfTwo,
    /// A pair's product or sum overflows `i64`.
    Overflow,
}

/// Multiply every pair of `x` by the cell `w` in `R[x]/(x^2 - sigma)`.
///
/// `x` is read as consecutive blocks of `2 * span` entries; at lane `i < span` the pair
/// `(x[b + i], x[b + i + span])` becomes `w * (x[b + i], x[b + i + span])`, the pair map of
/// `Numeric.Register.cellStage` (`register_stage_normForm` at `sigma = -1`).
///
/// # Errors
///
/// [`RegisterError::NotPowerOfTwo`] when `span` is zero, not a power of two, or at least `2^16`;
/// [`RegisterError::LengthMismatch`] when `x.len()` is not a multiple of `2 * span`;
/// [`RegisterError::Overflow`] when a pair's `i64` product or sum overflows. A refused pair leaves
/// the pairs written before it in place: there is **no partial claim**, the caller discards `x`.
///
/// # Complexity
///
/// `x.len() / 2` cell multiplies, each `4` `i64` products and `2` checked sums, and `O(1)` space.
/// The outer loop runs `x.len() / (2 * span)` times and the inner `span` times, so the pair count
/// is `x.len() / 2` (`Numeric.Register.register_stage_pairs`).
#[allow(
    clippy::manual_is_multiple_of,
    reason = "division and remainder keep this extraction body free of the unmodelled is_multiple_of method"
)]
pub fn stage_mul(
    x: &mut [i64],
    span: usize,
    w: Cell<i64>,
    sigma: Sigma,
) -> Result<(), RegisterError> {
    if !span.is_power_of_two() || span >= (1usize << 16) {
        return Err(RegisterError::NotPowerOfTwo);
    }
    let block = 2 * span;
    if x.len() % block != 0 {
        return Err(RegisterError::LengthMismatch);
    }

    let mut base = 0usize;
    let mut err: Option<RegisterError> = None;
    while base < x.len() && err.is_none() {
        err = stage_block(x, base, span, w, sigma);
        base += block;
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Multiply the `span` pairs of the block at `base`, or return the first refusal.
///
/// The pair at lane `i` is `(x[base + i], x[base + i + span])`. The index is carried by value and
/// the loop stops on the first error, so this is the lane loop a later task extracts.
fn stage_block(
    x: &mut [i64],
    base: usize,
    span: usize,
    w: Cell<i64>,
    sigma: Sigma,
) -> Option<RegisterError> {
    let mut lane = 0usize;
    let mut err: Option<RegisterError> = None;
    while lane < span && err.is_none() {
        let lo = base + lane;
        let hi = lo + span;
        let pair = Cell { a: x[lo], b: x[hi] };
        match w.checked_mul(sigma, pair) {
            Some(out) => {
                x[lo] = out.a;
                x[hi] = out.b;
            }
            None => err = Some(RegisterError::Overflow),
        }
        lane += 1;
    }
    err
}

#[cfg(test)]
mod tests {
    use super::{stage_mul, RegisterError};
    use crate::cell::{Cell, Sigma};

    /// Sum of squares as `i128`, so the test never overflows before the assertion.
    fn sum_sq(x: &[i64]) -> i128 {
        let mut total = 0i128;
        let mut i = 0usize;
        while i < x.len() {
            total += i128::from(x[i]) * i128::from(x[i]);
            i += 1;
        }
        total
    }

    /// At `w = (1, 1)` and `sigma = -1` a pair `(u, v)` maps to `(u - v, u + v)`.
    #[test]
    fn elliptic_stage_at_one_one_subtracts_adds() {
        let mut x = [1i64, 2, 3, 4];
        stage_mul(&mut x, 1, Cell::new(1, 1), Sigma::Elliptic).expect("no overflow");
        assert_eq!(x, [-1, 3, -1, 7]);

        let mut y = [1i64, 2, 3, 4];
        stage_mul(&mut y, 2, Cell::new(1, 1), Sigma::Elliptic).expect("no overflow");
        assert_eq!(y, [-2, -2, 4, 6]);
    }

    /// At `sigma = -1` and `w = (3, 4)` the sum of squares scales by `normForm = 25`.
    #[test]
    fn elliptic_stage_scales_the_sum_of_squares_by_the_norm_form() {
        let w = Cell::new(3i64, 4);
        let vectors: [[i64; 8]; 3] = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, -1, 4, 0, -3, 5, 1, -2],
            [7, 0, 0, 1, 0, 0, 2, 3],
        ];
        for v in vectors {
            let before = sum_sq(&v);
            for span in [1usize, 2, 4] {
                let mut x = v;
                stage_mul(&mut x, span, w, Sigma::Elliptic).expect("no overflow");
                assert_eq!(sum_sq(&x), 25 * before, "span {span}");
            }
        }
    }

    /// The conjugate stage after the stage is the norm form, on every face.
    #[test]
    fn stage_by_w_then_conj_w_is_the_norm_form() {
        let w = Cell::new(3i64, 4);
        let conj = Cell::new(3i64, -4);
        let x0: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        for sigma in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
            let norm = 9i64 - sigma.value() * 16;
            for span in [1usize, 2] {
                let mut x = x0;
                stage_mul(&mut x, span, w, sigma).expect("no overflow");
                stage_mul(&mut x, span, conj, sigma).expect("no overflow");
                let mut i = 0usize;
                while i < x.len() {
                    assert_eq!(x[i], norm * x0[i], "sigma {sigma:?} span {span} i {i}");
                    i += 1;
                }
            }
        }
    }

    /// Without overflow the checked multiply is the infallible `Cell::mul`.
    #[test]
    fn checked_mul_agrees_with_cell_mul_without_overflow() {
        let cases: [(Cell<i64>, Cell<i64>, Sigma); 6] = [
            (Cell::new(3, 4), Cell::new(1, -2), Sigma::Elliptic),
            (Cell::new(1, 1), Cell::new(5, 7), Sigma::Elliptic),
            (Cell::new(2, -3), Cell::new(0, 6), Sigma::Parabolic),
            (Cell::new(7, 0), Cell::new(-1, -1), Sigma::Parabolic),
            (Cell::new(4, 5), Cell::new(6, -7), Sigma::Hyperbolic),
            (Cell::new(-2, 3), Cell::new(8, 9), Sigma::Hyperbolic),
        ];
        for (w, p, sigma) in cases {
            assert_eq!(w.checked_mul(sigma, p), Some(w.mul(sigma, p)));
        }
    }

    /// The domain refusals: length, power of two, and a pair that overflows `i64`.
    #[test]
    fn refuses_bad_domain_and_overflow() {
        let mut x = [i64::MAX, 1];
        assert_eq!(
            stage_mul(&mut x, 1, Cell::new(2, 0), Sigma::Elliptic),
            Err(RegisterError::Overflow)
        );

        let mut short = [1i64, 2, 3];
        assert_eq!(
            stage_mul(&mut short, 1, Cell::new(1, 1), Sigma::Elliptic),
            Err(RegisterError::LengthMismatch)
        );

        let mut odd_span = [1i64, 2, 3, 4, 5, 6];
        assert_eq!(
            stage_mul(&mut odd_span, 3, Cell::new(1, 1), Sigma::Elliptic),
            Err(RegisterError::NotPowerOfTwo)
        );

        let mut zero_span = [1i64, 2];
        assert_eq!(
            stage_mul(&mut zero_span, 0, Cell::new(1, 1), Sigma::Elliptic),
            Err(RegisterError::NotPowerOfTwo)
        );
    }

    /// `checked_mul` refuses an `i64` overflow and equals `mul` where it fits.
    #[test]
    fn checked_mul_refuses_overflow_and_agrees_where_it_fits() {
        assert_eq!(
            Cell::new(i64::MAX, 0).checked_mul(Sigma::Elliptic, Cell::new(2, 0)),
            None
        );
        assert_eq!(
            Cell::new(3i64, 4).checked_mul(Sigma::Elliptic, Cell::new(1, -2)),
            Some(Cell::new(3i64, 4).mul(Sigma::Elliptic, Cell::new(1, -2)))
        );
    }
}
