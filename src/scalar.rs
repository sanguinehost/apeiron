//! The scalars a cell is built from.
//!
//! The algebra needs a commutative ring. Which one is a choice with consequences: `i64` is the
//! fast lattice and cannot represent a half, so the nontrivial idempotents `(1 +/- j)/2` do not
//! live in it. That is not a reason to weaken the claim -- it is a reason to have a scalar that
//! holds them.
//!
//! [`Dyadic`] is that scalar: a mantissa and a power-of-two scale, `m / 2^e`, exact under
//! addition, subtraction and multiplication, with halving free. The same shape a float has, minus
//! the rounding -- a float is a sign, an exponent and a mantissa, and the only reason it is
//! inexact is that it truncates the mantissa to a fixed width.
//!
//! The split is expressed in the bounds rather than in a comment. [`Ring`] is what the algebra
//! needs; [`Halve`] is the extra thing the idempotents need. `i64` implements the first and not
//! the second, so a construction that requires a half will not compile over `i64` -- which is the
//! honest form of "that theorem does not transfer".

use std::fmt::Debug;

/// A commutative ring: what the two-slot algebra is built over.
pub trait Ring: Copy + PartialEq + Debug {
    const ZERO: Self;
    const ONE: Self;

    #[must_use]
    fn add(self, other: Self) -> Self;
    #[must_use]
    fn sub(self, other: Self) -> Self;
    #[must_use]
    fn mul(self, other: Self) -> Self;
    #[must_use]
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
    #[allow(clippy::cast_possible_truncation)]
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

/// An exact dyadic rational, `m / 2^e`.
///
/// Closed under `+`, `-` and `*`, and halving costs one increment of the scale. Kept normalised --
/// trailing zero bits stripped from the mantissa -- so equality is structural and `1/2` has one
/// representation rather than an infinite family.
#[derive(Debug, Clone, Copy)]
pub struct Dyadic {
    mantissa: i128,
    scale: u32,
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

    /// Both values at a common scale, for adding.
    fn aligned(self, other: Self) -> (i128, i128, u32) {
        let scale = self.scale.max(other.scale);
        (
            self.mantissa << (scale - self.scale),
            other.mantissa << (scale - other.scale),
            scale,
        )
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

    fn add(self, other: Self) -> Self {
        let (a, b, scale) = self.aligned(other);
        Self::new(a + b, scale)
    }

    fn sub(self, other: Self) -> Self {
        let (a, b, scale) = self.aligned(other);
        Self::new(a - b, scale)
    }

    fn mul(self, other: Self) -> Self {
        Self::new(self.mantissa * other.mantissa, self.scale + other.scale)
    }

    fn from_i64(v: i64) -> Self {
        Self::whole(v)
    }
}

impl Halve for Dyadic {
    fn halve(self) -> Self {
        Self::new(self.mantissa, self.scale + 1)
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
}
