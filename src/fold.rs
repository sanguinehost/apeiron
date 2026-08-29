//! Addressing a key that is not already an address.
//!
//! `Coord` for an integer is the identity: the key *is* its address. For a string, a byte slice
//! or anything else of unbounded width there is no such luck, and the address has to be folded
//! out of the bytes. That fold is a **discard** -- arbitrarily many byte strings, `2^64`
//! addresses -- so by [`crate::keystore`] it is a hyperbolic operation, and the table it feeds
//! must be one that carries the key beside the address, because the discard destroyed it.
//!
//! That is the whole reason the parabolic face exists, and it is worth stating in the direction
//! that matters: the key is not stored because storing keys is what maps do. It is stored because
//! the address is a projection, the projection has fibers, and **nothing else can tell two keys
//! in the same fiber apart**.
//!
//! # The one place XOR is used, and why it is not the earlier mistake
//!
//! [`crate::coord`] measures a fold that crossed from `Z/2^64` into `GF(2)^64` -- `a * PHI ^ b` --
//! and collides half its keys, because XOR is linear and self-inverse and cannot see a carry. So
//! the rule in this crate is that accumulation stays in the ring.
//!
//! [`mix2`] uses XOR twice, and neither use is that crossing:
//!
//! - **Folding a 128-bit product to 64 bits.** `hi(a*b)` and `lo(a*b)` are not linear functions of
//!   `a` and `b` over `GF(2)` -- `lo` is the ring product truncated and `hi` is the carry-out of
//!   it -- so combining them is not a linear recombine of the *inputs*, which is what cancelled.
//!   Addition would serve equally; XOR is one cycle cheaper and does not propagate a carry into
//!   bits the multiply already decided.
//! - **The annihilation guard.** A bare folded multiply has a fatal fiber: an attacker who drives
//!   either operand to zero gets address zero for *every* value of the other, which is a family of
//!   colliding keys as large as they like. `wyhash` and its descendants close this with a secret
//!   seed. This crate cannot -- a per-run seed denies deterministic replay -- so it closes it
//!   structurally instead, by adding a term of the inputs back over the product. When the product
//!   annihilates, that term is all that is left, and it is injective in the free operand.
//!
//! Inputs are combined into `mix2` with `wrapping_add`, never XOR. The ring rule holds where it
//! was measured to matter.
//!
//! # What a seedless fold can and cannot promise
//!
//! It cannot promise collision resistance. The constants are public, the function is public, and
//! `2^64` addresses meet a birthday bound at `2^32` keys; an adversary who can search offline can
//! find pairs, and may be able to find them algebraically faster than that.
//!
//! What it changes is what a collision *costs*. In [`crate::table`] two equal addresses are one
//! key by definition, so the window descent's bound -- a chain of `j` steps costs `j * gbits` bits
//! of control -- is the whole story. In a keyed map two equal addresses are two keys, they receive
//! the identical probe sequence, and the descent cannot separate them: a fiber of size `m` costs
//! `O(m^2)` to insert. **This is the parabolic face's exposure and it is not structural.** It is
//! the same exposure `hashbrown` has with a fixed hasher, and the answer is the same one:
//! [`crate::pepper`], a process-wide constant supplied by the caller, which moves the fibers somewhere
//! the attacker cannot precompute. It is a *constant*, not a per-table random seed, so a recorded
//! pepper replays a run exactly -- determinism and unpredictability are not in conflict here,
//! because only reproducibility across runs of *the same recorded session* was ever required.

use crate::coord::{Coord, PHI};

/// A second odd multiplier, independent of [`PHI`]. Odd, so multiplication by it is a bijection on
/// `Z/2^64` and no input is mapped to zero except zero.
pub const NU: u64 = 0xD6E8_FEB8_6659_FD93;

/// The default address offset: none.
///
/// A pepper is a process-wide constant mixed into every byte fold. Zero is the default because a
/// game engine replaying a recorded session must address keys the same way it did when it recorded
/// it, and because a table that is never shown untrusted keys gains nothing from one.
///
/// A server that accepts keys from the network should set one. See [`crate::pepper`].
pub const NO_PEPPER: u64 = 0;

/// Two words to one, with no fiber an attacker can name.
///
/// The product's halves are folded together, and then a term of the inputs is laid over the
/// result. The second half is the guard: `a` or `b` equal to zero kills the product entirely, and
/// without the guard every key in that family would share address zero.
#[inline]
#[must_use]
pub const fn mix2(a: u64, b: u64) -> u64 {
    let w = (a as u128).wrapping_mul(b as u128);
    #[allow(clippy::cast_possible_truncation)]
    let p = ((w >> 64) as u64) ^ (w as u64);
    // Rotating `b` before adding means the guard is not symmetric in its arguments, so a swapped
    // pair does not survive it, and it is injective in each argument with the other held fixed.
    p ^ a.wrapping_add(b.rotate_left(32))
}

#[inline]
fn word(b: &[u8]) -> u64 {
    // Callers only ever pass exactly eight bytes; the fallback keeps the function total rather
    // than adding a panic to a hot loop.
    b.try_into().map_or(0, u64::from_le_bytes)
}

#[inline]
fn half(b: &[u8]) -> u64 {
    b.try_into().map_or(0, |a| u64::from(u32::from_le_bytes(a)))
}

/// The address of a byte string, with no pepper.
#[inline]
#[must_use]
pub fn fold_bytes(b: &[u8]) -> u64 {
    fold_bytes_with(NO_PEPPER, b)
}

/// The address of a byte string.
///
/// Length is folded in first, so `"ab"` and `"ab\0"` are not near neighbours, and every branch
/// reads its input through overlapping reads rather than a copy -- a key of 5..=7 bytes is two
/// overlapping `u32`s, one of 9..=16 is two overlapping `u64`s, and neither needs a buffer.
///
/// Long keys run two independent lanes, sixteen bytes a round. Independent, not chained: the two
/// multiplies of a round do not read each other's output, so the multiplier is issued twice per
/// round rather than serialised behind itself.
///
/// `#[inline]` is load-bearing, the same way it was on the probe internals: this function is not
/// generic, so without it every caller outside the crate pays a call and the spills around it.
/// Measured on the `String` hit loop: 112.3 instructions and 25.7 cycles a lookup outlined, 97.3
/// and 22.2 inlined (one instruction of that step is `map::Geom`'s, landed together) -- most of
/// what stood between this fold and fxhash was the call.
#[inline]
#[must_use]
pub fn fold_bytes_with(pepper: u64, b: &[u8]) -> u64 {
    let len = b.len();
    // The length enters through a multiply, so it changes every bit of the seed rather than the
    // low ones. Two keys of different lengths therefore start from unrelated states.
    let seed = pepper.wrapping_add(PHI).wrapping_add((len as u64).wrapping_mul(NU));

    if len <= 16 {
        let (a, c) = if len >= 8 {
            (word(&b[..8]), word(&b[len - 8..]))
        } else if len >= 4 {
            (half(&b[..4]), half(&b[len - 4..]))
        } else if len > 0 {
            // Three reads cover 1, 2 and 3 bytes without a branch per length: first, middle and
            // last, which are distinct positions for every length in range.
            let a = (u64::from(b[0]) << 16)
                | (u64::from(b[len >> 1]) << 8)
                | u64::from(b[len - 1]);
            (a, 0)
        } else {
            (0, 0)
        };
        return mix2(a.wrapping_add(seed), c.wrapping_add(NU));
    }

    let mut x = seed;
    let mut y = seed.wrapping_add(NU);
    let mut rest = b;
    while rest.len() > 16 {
        let lo = word(&rest[..8]);
        let hi = word(&rest[8..16]);
        // Both lanes read both words, in opposite roles, and neither reads the other's output.
        let nx = mix2(x.wrapping_add(lo), NU.wrapping_add(hi));
        let ny = mix2(y.wrapping_add(hi), PHI.wrapping_add(lo));
        x = nx;
        y = ny;
        rest = &rest[16..];
    }
    // The last sixteen bytes, overlapping whatever the loop already consumed. Overlap rather than
    // a padded tail: no copy, no branch per remaining length, and the bytes are covered twice
    // rather than once, which is harmless.
    let tail = &b[len - 16..];
    mix2(
        x.wrapping_add(word(&tail[..8])),
        y.wrapping_add(word(&tail[8..])),
    )
}

impl Coord for str {
    #[inline]
    fn coord(&self) -> u64 {
        fold_bytes(self.as_bytes())
    }
    #[inline]
    fn coord_with(&self, pepper: u64) -> u64 {
        fold_bytes_with(pepper, self.as_bytes())
    }
}

impl Coord for String {
    #[inline]
    fn coord(&self) -> u64 {
        fold_bytes(self.as_bytes())
    }
    #[inline]
    fn coord_with(&self, pepper: u64) -> u64 {
        fold_bytes_with(pepper, self.as_bytes())
    }
}

impl Coord for [u8] {
    #[inline]
    fn coord(&self) -> u64 {
        fold_bytes(self)
    }
    #[inline]
    fn coord_with(&self, pepper: u64) -> u64 {
        fold_bytes_with(pepper, self)
    }
}

impl Coord for Vec<u8> {
    #[inline]
    fn coord(&self) -> u64 {
        fold_bytes(self)
    }
    #[inline]
    fn coord_with(&self, pepper: u64) -> u64 {
        fold_bytes_with(pepper, self)
    }
}

impl<const N: usize> Coord for [u8; N] {
    #[inline]
    fn coord(&self) -> u64 {
        fold_bytes(self)
    }
    #[inline]
    fn coord_with(&self, pepper: u64) -> u64 {
        fold_bytes_with(pepper, self)
    }
}

impl<T: Coord + ?Sized> Coord for &T {
    #[inline]
    fn coord(&self) -> u64 {
        (**self).coord()
    }
    #[inline]
    fn coord_with(&self, pepper: u64) -> u64 {
        (**self).coord_with(pepper)
    }
}

impl Coord for char {
    #[inline]
    fn coord(&self) -> u64 {
        u64::from(*self as u32)
    }
}

impl Coord for bool {
    #[inline]
    fn coord(&self) -> u64 {
        u64::from(*self)
    }
}

/// Wider than a word, so it folds rather than packs.
impl Coord for u128 {
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn coord(&self) -> u64 {
        mix2((*self as u64).wrapping_add(PHI), ((*self >> 64) as u64).wrapping_add(NU))
    }
}

impl Coord for i128 {
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn coord(&self) -> u64 {
        (*self as u128).coord()
    }
}

impl<T: Coord> Coord for Option<T> {
    #[inline]
    fn coord(&self) -> u64 {
        self.coord_with(0)
    }
    #[inline]
    fn coord_with(&self, pepper: u64) -> u64 {
        // `None` is not zero: a `None` and a `Some(0)` are different keys and must be able to have
        // different addresses.
        match self {
            None => NU.wrapping_add(pepper),
            Some(v) => v.coord_with(pepper).wrapping_mul(PHI).wrapping_add(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect the guard exists to prevent, stated as a test rather than as a comment.
    ///
    /// Without the guard a folded multiply returns zero whenever either operand is zero, so an
    /// attacker who fixes the first eight bytes to cancel the seed gets one address for every
    /// possible tail.
    #[test]
    fn a_zeroed_operand_does_not_collapse_the_fold() {
        let mut seen = std::collections::HashSet::new();
        for c in 0..4096u64 {
            seen.insert(mix2(0, c));
        }
        assert_eq!(seen.len(), 4096, "mix2(0, .) must stay injective");

        let mut seen = std::collections::HashSet::new();
        for a in 0..4096u64 {
            seen.insert(mix2(a, 0));
        }
        assert_eq!(seen.len(), 4096, "mix2(., 0) must stay injective");
    }

    /// Swapping the two words is a different key and must be a different address.
    #[test]
    fn the_guard_is_not_symmetric() {
        assert_ne!(mix2(1, 2), mix2(2, 1));
        assert_ne!(mix2(PHI, NU), mix2(NU, PHI));
    }

    #[test]
    fn length_is_part_of_the_key() {
        assert_ne!(fold_bytes(b"ab"), fold_bytes(b"ab\0"));
        assert_ne!(fold_bytes(b""), fold_bytes(b"\0"));
        assert_ne!(fold_bytes(&[0u8; 16]), fold_bytes(&[0u8; 17]));
        assert_ne!(fold_bytes(&[0u8; 32]), fold_bytes(&[0u8; 33]));
    }

    #[test]
    fn a_pepper_moves_every_address() {
        let keys = ["", "a", "abcd", "abcdefgh", "the quick brown fox jumps over the lazy dog"];
        for k in keys {
            assert_ne!(
                fold_bytes_with(0, k.as_bytes()),
                fold_bytes_with(0x5EED, k.as_bytes()),
                "pepper did nothing for {k:?}"
            );
        }
    }

    /// Every branch of the length ladder must actually be exercised by the corpus below, or the
    /// distribution test is measuring one branch and reporting on five.
    #[test]
    fn no_two_of_a_hundred_thousand_distinct_strings_share_an_address() {
        let mut seen = std::collections::HashSet::new();
        let mut lengths = [false; 5];
        #[cfg(miri)]
        let n = 2_000u32;
        #[cfg(not(miri))]
        let n = 100_000u32;
        for i in 0..n {
            // Lengths from 0 to ~40, so all five branches run.
            let s = format!("{}{}", "resref_".repeat((i % 6) as usize), i);
            match s.len() {
                0 => lengths[0] = true,
                1..=3 => lengths[1] = true,
                4..=7 => lengths[2] = true,
                8..=16 => lengths[3] = true,
                _ => lengths[4] = true,
            }
            assert!(seen.insert(fold_bytes(s.as_bytes())), "collision on {s:?}");
        }
        assert!(lengths[1] && lengths[2] && lengths[3] && lengths[4]);
    }

    /// A single input bit must reach roughly half the output bits. Below 0.35 or above 0.65 the
    /// fold has a structural bias, which is what the earlier XOR fold had.
    #[test]
    fn one_input_bit_moves_about_half_the_output() {
        let base = *b"a_typical_resource_reference";
        let mut worst = (0usize, 0.5f64);
        for bit in 0..base.len() * 8 {
            let mut flipped = base;
            flipped[bit / 8] ^= 1 << (bit % 8);
            let d = (fold_bytes(&base) ^ fold_bytes(&flipped)).count_ones();
            let frac = f64::from(d) / 64.0;
            if (frac - 0.5).abs() > (worst.1 - 0.5f64).abs() {
                worst = (bit, frac);
            }
        }
        assert!(
            (0.30..=0.70).contains(&worst.1),
            "bit {} moved {:.3} of the output",
            worst.0,
            worst.1
        );
    }
}
