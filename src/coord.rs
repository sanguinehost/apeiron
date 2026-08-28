//! Turning a key into an address, without leaving the ring.
//!
//! A coordinate fold is arithmetic in `Z/2^64`, which is a commutative ring. So is the two-slot
//! algebra. Mixing in an operation from a *different* structure -- XOR, which is addition in
//! `GF(2)^64` -- is not a smaller version of the same thing; it is a different algebra, and the
//! multiplicative spread that made the fold work does not survive the crossing.
//!
//! Measured on hashtrinity's `(A, B)` fold, `a.wrapping_mul(PHI) ^ b`:
//!
//! | key set | XOR | `wrapping_add` |
//! |---|---|---|
//! | 2-tuple, `-64..63` | 49.2% collide | 0% |
//! | 2-tuple, `-512..511` | 49.9% collide | 0% |
//! | 3-tuple, `-16..15` | 71.8% collide | 0% |
//!
//! The cause is exact. A signed integer widens by sign extension, so a negative coordinate has
//! its whole high half set, and `PHI ^ -PHI` is `0xFFFF_FFFF_FFFF_FFFE` -- for any odd `v`,
//! `v ^ -v` is `!1`. XOR is self-inverse and linear over `GF(2)`, so it cannot tell that flip
//! apart from the matching flip in the other coordinate, and the two cancel. Addition carries,
//! so it cannot cancel: `(a1 - a2) * PHI = b2 - b1` has no small solutions when `PHI` is a large
//! odd multiplier.
//!
//! It is worth being clear about what this is *not*. The collisions do not lie on the null cone
//! `|a| = |b|` -- 126 of 8064 do, against 7938 that do not -- so this is not the algebra showing
//! through as a bug. It is a plain ring/field mismatch, and the fix is to stay in one ring.

/// The golden-ratio odd multiplier. Odd, so multiplication by it is a bijection on `Z/2^64`.
pub const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

/// A key's 64-bit address.
pub trait Coord {
    fn coord(&self) -> u64;
}

macro_rules! int_coord {
    ($($t:ty),*) => {$(
        impl Coord for $t {
            #[inline]
            fn coord(&self) -> u64 {
                // Sign extension is deliberate and is what makes `from_coord` round-trip.
                #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
                { *self as u64 }
            }
        }
    )*};
}
int_coord!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// Mix one more coordinate into a running address.
///
/// Multiply then add, both in `Z/2^64`. Multiplying by an odd constant is a bijection, and adding
/// is a bijection, so each step is invertible in its own right -- which is why the composite
/// separates keys instead of folding them together.
#[inline]
#[must_use]
pub fn mix(acc: u64, next: u64) -> u64 {
    acc.wrapping_mul(PHI).wrapping_add(next)
}

impl<A: Coord, B: Coord> Coord for (A, B) {
    #[inline]
    fn coord(&self) -> u64 {
        mix(self.0.coord(), self.1.coord())
    }
}

impl<A: Coord, B: Coord, C: Coord> Coord for (A, B, C) {
    #[inline]
    fn coord(&self) -> u64 {
        mix(mix(self.0.coord(), self.1.coord()), self.2.coord())
    }
}

impl<A: Coord, B: Coord, C: Coord, D: Coord> Coord for (A, B, C, D) {
    #[inline]
    fn coord(&self) -> u64 {
        mix(mix(mix(self.0.coord(), self.1.coord()), self.2.coord()), self.3.coord())
    }
}
