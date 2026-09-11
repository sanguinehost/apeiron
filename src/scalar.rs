//! The scalars a cell is built from.
//!
//! The algebra needs a commutative ring. Which one is a choice with consequences: `i64` is the
//! fast lattice and cannot represent a half, so the nontrivial idempotents `(1 +/- j)/2` do not
//! live in it. That is not a reason to weaken the claim -- it is a reason to have a scalar that
//! holds them.
//!
//! [`Dyadic`] is that scalar: a mantissa and a power-of-two scale, `m / 2^e`. The same shape a
//! float has, minus the rounding -- a float is a sign, an exponent and a mantissa, and its
//! inexactness is that it truncates the mantissa to a fixed width. `Dyadic` is exact in that
//! sense, but it is **not** a total ring either: the mantissa is a fixed `i128` and the scale a
//! `u32`, so alignment, multiplication and scale growth can exceed the representation. The
//! `checked_*` methods return `None` there, and the [`Ring`]/[`Halve`] implementations panic
//! rather than dropping bits silently.
//!
//! The split is expressed in the bounds rather than in a comment. [`Ring`] is what the algebra
//! needs; [`Halve`] is the extra thing the idempotents need. `i64` implements the first and not
//! the second, so a construction that requires a half will not compile over `i64` -- which is the
//! honest form of "that theorem does not transfer".

use std::fmt::Debug;

/// A commutative ring: what the two-slot algebra is built over.
///
/// It deliberately does **not** require `Debug`: the algebra is arithmetic, and a formatting
/// supertrait would drag `core::fmt` into the extracted Lean model through
/// `Aeneas.Std.core.fmt.Formatter` for no arithmetic reason. Types that want `Debug` derive it.
pub trait Ring: Copy + PartialEq {
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;

    #[must_use]
    /// Ring addition.
    fn add(self, other: Self) -> Self;
    #[must_use]
    /// Ring subtraction.
    fn sub(self, other: Self) -> Self;
    #[must_use]
    /// Ring multiplication.
    fn mul(self, other: Self) -> Self;
    #[must_use]
    /// The ring element an integer names.
    fn from_i64(v: i64) -> Self;
}

/// A ring in which two is invertible.
///
/// The nontrivial idempotents are `(1 +/- j)/2` and the change into the null basis is
/// `a = (u + l)/2`, so both need exactly this and nothing more. Splitting it out is what makes
/// `Idempotents::exists_nontrivial_idempotent_iff_disc_pos` available over a scalar that has
/// halves and unavailable over one that does not, at compile time.
pub trait Halve: Ring {
    #[must_use]
    /// Division by two, exactly. What the idempotents need and `i64` cannot give.
    fn halve(self) -> Self;
}

/// A narrow ring, for proof only.
///
/// The algebra's laws are polynomial identities: `mul_assoc` is an equality of two polynomials in
/// the scalar variables, with integer coefficients, and such an identity holds in *every*
/// commutative ring or in none. It says nothing about a width.
///
/// A bit-precise checker does not know that. It bit-blasts a multiply to the width of the type
/// whatever the operands hold, so the same identity that decides in a second over 8 bits does not
/// return over 64. Proving it here is proving the structure; the engine keeps using `i64`.
impl Ring for i8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
    fn mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The narrowing is the caller's declared intent -- this is the `i64` -> `Self` direction \
                      of a scalar conversion whose contract is wrapping, matching the rest of the trait."
    )]
    fn from_i64(v: i64) -> Self {
        v as Self
    }
}

impl Ring for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }
    fn mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }
    fn from_i64(v: i64) -> Self {
        v
    }
}

/// A dyadic rational, `m / 2^e`, kept normalised.
///
/// Exact for `+`, `-` and `*` **when the result fits**: the mantissa is an `i128` and the scale a
/// `u32`. It is not a total ring. Aligning two values at very different scales, multiplying two
/// large mantissas, or growing the scale can exceed the representation, and the `checked_*` methods
/// return `None` for those cases instead of dropping bits. The [`Ring`] and [`Halve`]
/// implementations are correspondingly partial: they panic on an unrepresentable result rather than
/// returning a silently wrong one. That boundary is the correction to the previous `aligned`, which
/// shifted the smaller mantissa up by a plain `<<` and discarded the overflow without saying so.
///
/// Trailing zero bits are stripped from the mantissa, so equality is structural and `1/2` has one
/// representation rather than an infinite family.
#[derive(Debug, Clone, Copy)]
pub struct Dyadic {
    mantissa: i128,
    scale: u32,
}

/// Shift `value` left by `bits`, or `None` if any bit would be lost or the shift is out of range.
///
/// `i128`'s `<<` discards the shifted-out high bits silently, which is exactly how an aligned sum
/// can drop a term and still compare equal to a wrong answer. `value == 0` is representable at
/// every scale, so it is allowed even for an out-of-range `bits`.
fn checked_shl_exact(value: i128, bits: u32) -> Option<i128> {
    if value == 0 {
        return Some(0);
    }
    if bits >= i128::BITS {
        return None;
    }
    let shifted = value << bits;
    (shifted >> bits == value).then_some(shifted)
}

impl Dyadic {
    /// `m / 2^e`, normalised.
    #[must_use]
    pub fn new(mantissa: i128, scale: u32) -> Self {
        Self { mantissa, scale }.normalised()
    }

    /// A whole number.
    #[must_use]
    pub fn whole(v: i64) -> Self {
        Self {
            mantissa: i128::from(v),
            scale: 0,
        }
    }

    /// Strip the common factors of two, so equal values compare equal.
    fn normalised(self) -> Self {
        if self.mantissa == 0 {
            return Self {
                mantissa: 0,
                scale: 0,
            };
        }
        let shed = self.mantissa.trailing_zeros().min(self.scale);
        Self {
            mantissa: self.mantissa >> shed,
            scale: self.scale - shed,
        }
    }

    /// Both values at a common scale, for adding, or `None` if raising either to that scale would
    /// lose bits.
    fn checked_aligned(self, other: Self) -> Option<(i128, i128, u32)> {
        let scale = self.scale.max(other.scale);
        let a = checked_shl_exact(self.mantissa, scale - self.scale)?;
        let b = checked_shl_exact(other.mantissa, scale - other.scale)?;
        Some((a, b, scale))
    }

    /// `self + other`, or `None` if the aligned sum is not representable.
    #[must_use]
    pub fn checked_add(self, other: Self) -> Option<Self> {
        let (a, b, scale) = self.checked_aligned(other)?;
        Some(Self::new(a.checked_add(b)?, scale))
    }

    /// `self - other`, or `None` if the aligned difference is not representable.
    #[must_use]
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        let (a, b, scale) = self.checked_aligned(other)?;
        Some(Self::new(a.checked_sub(b)?, scale))
    }

    /// `self * other`, or `None` if the product mantissa or the combined scale overflows.
    #[must_use]
    pub fn checked_mul(self, other: Self) -> Option<Self> {
        let mantissa = self.mantissa.checked_mul(other.mantissa)?;
        let scale = self.scale.checked_add(other.scale)?;
        Some(Self::new(mantissa, scale))
    }

    /// `self / 2`, or `None` if the scale would overflow.
    #[must_use]
    pub fn checked_halve(self) -> Option<Self> {
        Some(Self::new(self.mantissa, self.scale.checked_add(1)?))
    }

    /// The exact numerator and its power-of-two denominator.
    #[must_use]
    pub const fn parts(self) -> (i128, u32) {
        (self.mantissa, self.scale)
    }

    /// Whether this is a whole number.
    #[must_use]
    pub const fn is_whole(self) -> bool {
        self.scale == 0
    }
}

impl PartialEq for Dyadic {
    fn eq(&self, other: &Self) -> bool {
        let a = self.normalised();
        let b = other.normalised();
        a.mantissa == b.mantissa && a.scale == b.scale
    }
}

impl Eq for Dyadic {}

impl Ring for Dyadic {
    const ZERO: Self = Self {
        mantissa: 0,
        scale: 0,
    };
    const ONE: Self = Self {
        mantissa: 1,
        scale: 0,
    };

    /// # Panics
    ///
    /// Panics if the aligned sum is not representable. Use [`Dyadic::checked_add`] to handle that
    /// case; the [`Ring`] trait cannot express it.
    fn add(self, other: Self) -> Self {
        self.checked_add(other)
            .expect("dyadic addition overflow: the aligned sum is not representable")
    }

    /// # Panics
    ///
    /// Panics if the aligned difference is not representable. Use [`Dyadic::checked_sub`] to handle
    /// that case.
    fn sub(self, other: Self) -> Self {
        self.checked_sub(other)
            .expect("dyadic subtraction overflow: the aligned difference is not representable")
    }

    /// # Panics
    ///
    /// Panics if the product mantissa or the combined scale overflows. Use [`Dyadic::checked_mul`]
    /// to handle that case.
    fn mul(self, other: Self) -> Self {
        self.checked_mul(other)
            .expect("dyadic multiplication overflow: mantissa or scale exceeds its representation")
    }

    fn from_i64(v: i64) -> Self {
        Self::whole(v)
    }
}

impl Halve for Dyadic {
    /// # Panics
    ///
    /// Panics if the scale would overflow. Use [`Dyadic::checked_halve`] to handle that case.
    fn halve(self) -> Self {
        self.checked_halve()
            .expect("dyadic halving overflow: scale exceeds u32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_half_is_exact_and_round_trips() {
        let half = Dyadic::ONE.halve();
        assert_eq!(half.parts(), (1, 1));
        assert_eq!(half.add(half), Dyadic::ONE);
        assert!(!half.is_whole());
        assert!(Dyadic::ONE.is_whole());
    }

    #[test]
    fn equality_is_independent_of_scale() {
        assert_eq!(Dyadic::new(2, 1), Dyadic::ONE);
        assert_eq!(Dyadic::new(8, 3), Dyadic::ONE);
        assert_eq!(Dyadic::new(0, 7), Dyadic::ZERO);
        assert_ne!(Dyadic::new(1, 1), Dyadic::ONE);
    }

    /// The property the whole type exists for: quartering, then multiplying back, is exact.
    #[test]
    fn arithmetic_stays_exact_where_a_float_would_drift() {
        let mut v = Dyadic::ONE;
        for _ in 0..80 {
            v = v.halve();
        }
        let mut back = v;
        for _ in 0..80 {
            back = back.add(back);
        }
        assert_eq!(back, Dyadic::ONE, "80 halvings and 80 doublings round-trip");
    }

    /// The alignment bug: `2` at scale 0, raised to scale 127, needs 129 bits of mantissa. Before
    /// the fix it was shifted with a plain `<<`, the overflow was discarded, and the sum collapsed
    /// onto the first addend. It is now reported instead.
    #[test]
    fn checked_arithmetic_reports_every_overflow() {
        assert_eq!(Dyadic::new(1, 127).checked_add(Dyadic::whole(2)), None);
        assert_eq!(Dyadic::whole(3).checked_sub(Dyadic::new(1, 127)), None);
        assert_eq!(
            Dyadic::new(i128::MAX, 0).checked_mul(Dyadic::whole(2)),
            None
        );
        assert_eq!(
            Dyadic::new(1, u32::MAX).checked_mul(Dyadic::new(1, 1)),
            None
        );
        assert_eq!(Dyadic::new(1, u32::MAX).checked_halve(), None);
    }

    /// The same cases still work where the result is representable.
    #[test]
    fn checked_arithmetic_is_exact_where_it_is_representable() {
        assert_eq!(
            Dyadic::ONE.checked_add(Dyadic::whole(2)),
            Some(Dyadic::whole(3))
        );
        assert_eq!(
            Dyadic::whole(3).checked_sub(Dyadic::ONE),
            Some(Dyadic::whole(2))
        );
        assert_eq!(
            Dyadic::new(1, 1).checked_mul(Dyadic::new(1, 1)),
            Some(Dyadic::new(1, 2))
        );
        assert_eq!(Dyadic::new(1, 1).checked_halve(), Some(Dyadic::new(1, 2)));
    }

    /// The infallible [`Ring`] surface is partial and says so by panicking rather than returning a
    /// value with the overflow already gone.
    #[test]
    #[should_panic(expected = "dyadic addition overflow")]
    fn the_infallible_api_refuses_to_drop_bits() {
        let _ = Dyadic::new(1, 127).add(Dyadic::whole(2));
    }
}
