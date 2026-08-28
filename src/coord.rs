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
//! through as a bug. It is a plain ring/field mismatch.
//!
//! # One fold is the wrong number of folds
//!
//! Staying in the ring is necessary and not sufficient, and the earlier claim that multiply-add
//! collides on 0% of keys was a property of the grids it was measured on. It is **not injective**
//! on `u32 x u32`: `f(a, b) = a * PHI + b` collides whenever `d * PHI = -e (mod 2^64)` for
//! `|d|, |e| < 2^32`, and the golden ratio's convergents are Fibonacci numbers, so
//! `d = 2_971_215_073` gives `e = 50_920_843` and `(2971215073, 50920843)` folds to the same
//! address as `(0, 0)`.
//!
//! The algebra says that is the wrong question to ask once, because the faces want different
//! things from an address:
//!
//! | face | wants | so the fold is |
//! |---|---|---|
//! | elliptic (**B**) | the address *determines* the key -- nothing is degenerate | a bijection: [`pack`], and it refuses keys too wide to fit |
//! | parabolic (**B/U**) | the address need not determine the key, which is carried beside it | any spreading fold; misses are caught by comparing the stored key |
//! | hyperbolic (**NB**) | the address is a discard, and the key is gone | a scrambler, judged on *distribution*, never on injectivity |
//!
//! So injectivity is a requirement on one face, a non-requirement on the second, and impossible
//! on the third. A single fold cannot be right for all three, and measuring "the" collision rate
//! is measuring three different things at once.

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

/// Mix one more coordinate into a running address, losing information.
///
/// Multiply then add, both in `Z/2^64`. Each step is a bijection on a single word, but the
/// composite maps two words to one and therefore cannot be injective: it is a **discard**, and
/// belongs to the hyperbolic face. Judge it on how evenly it spreads, never on whether it
/// separates -- there are `2^64` addresses and `2^128` pairs, so it does not.
#[inline]
#[must_use]
pub fn mix(acc: u64, next: u64) -> u64 {
    acc.wrapping_mul(PHI).wrapping_add(next)
}

/// A bijective address for keys narrow enough to have one.
///
/// Concatenation, not arithmetic: `hi` in the top `shift` bits, `lo` beneath. Exactly invertible
/// by construction, which is what the elliptic face requires and what no multiply-add fold of two
/// full words can provide. `shift` is the width of `lo`, and the caller is responsible for `hi`
/// fitting in the rest -- see [`CoordInv`] for the types where that is checked by construction.
#[inline]
#[must_use]
pub const fn pack(hi: u64, lo: u64, shift: u32) -> u64 {
    (hi << shift) | lo
}

/// A key whose address determines it exactly: `unpack(k.packed()) == k`.
///
/// The elliptic face. Deliberately a **separate trait from [`Coord`]**, not a refinement of it,
/// because the two are different operations that give different answers: `packed` is a bijection
/// and `coord` is a discard. A key can have both, and it should not be possible to reach for one
/// while meaning the other.
///
/// Implemented only for keys that fit in 64 bits, because that is exactly when the bijection
/// exists. There is no honest `Packed` for two full `u64`s.
pub trait Packed: Sized {
    fn packed(&self) -> u64;
    fn unpack(c: u64) -> Self;
}

macro_rules! int_packed {
    ($($t:ty),*) => {$(
        impl Packed for $t {
            #[inline]
            #[allow(clippy::cast_sign_loss, clippy::cast_lossless)]
            fn packed(&self) -> u64 { *self as u64 }
            #[inline]
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            fn unpack(c: u64) -> Self { c as Self }
        }
    )*};
}
int_packed!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// Two 32-bit halves fill one word exactly.
impl Packed for (u32, u32) {
    #[inline]
    fn packed(&self) -> u64 {
        pack(u64::from(self.0), u64::from(self.1), 32)
    }
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn unpack(c: u64) -> Self {
        ((c >> 32) as u32, c as u32)
    }
}

/// Signed halves, bit-cast. Two's complement makes the round-trip exact.
impl Packed for (i32, i32) {
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn packed(&self) -> u64 {
        pack(u64::from(self.0 as u32), u64::from(self.1 as u32), 32)
    }
    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn unpack(c: u64) -> Self {
        (((c >> 32) as u32) as i32, (c as u32) as i32)
    }
}

/// Three signed 21-bit coordinates and a spare bit: a world of +/- one million units a side,
/// larger than any Aurora area and most worlds built out of them.
impl Packed for (i21, i21, i21) {
    #[inline]
    fn packed(&self) -> u64 {
        pack(pack(self.0.bits(), self.1.bits(), 21), self.2.bits(), 21)
    }
    #[inline]
    fn unpack(c: u64) -> Self {
        (
            i21::from_bits((c >> 42) & 0x1F_FFFF),
            i21::from_bits((c >> 21) & 0x1F_FFFF),
            i21::from_bits(c & 0x1F_FFFF),
        )
    }
}

/// A 21-bit signed integer, so three fit in a word with a bit to spare.
///
/// Exists because the elliptic face's bijection is only available when the key actually fits, and
/// a type that cannot hold an out-of-range value is how that is enforced rather than hoped for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(non_camel_case_types)]
pub struct i21(i32);

impl i21 {
    pub const MIN: i32 = -(1 << 20);
    pub const MAX: i32 = (1 << 20) - 1;

    /// `None` if `v` will not fit, which is the whole point of the type.
    #[must_use]
    pub const fn new(v: i32) -> Option<Self> {
        if v >= Self::MIN && v <= Self::MAX {
            Some(Self(v))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    const fn bits(self) -> u64 {
        (self.0 as u64) & 0x1F_FFFF
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    const fn from_bits(b: u64) -> Self {
        // Sign-extend from bit 20.
        Self((((b << 43) as i64) >> 43) as i32)
    }
}

impl Coord for i21 {
    #[inline]
    fn coord(&self) -> u64 {
        self.bits()
    }
}

/// The lossy folds, for the faces that do not need a bijection. Generic over any `Coord`, and
/// deliberately *not* `CoordInv`: a fold of two words into one cannot round-trip.
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
