//! Folding a key of unbounded width into a `u64` address.
//!
//! `Coord` for an integer is the identity. For bytes there is no such luck, so this folds them:
//! `mix2(a, b)` is the folded 128-bit product `lo ^ hi`, plus a guard term `a + rotl(b, 32)`.
//!
//! **The guard is not optional.** A bare folded multiply returns zero whenever either operand is
//! zero, so an attacker who cancels the seed gets one address for every tail. `wyhash` closes this
//! with a secret seed; this crate cannot (a per-run seed denies deterministic replay), so it
//! closes it structurally -- when the product annihilates the guard term is all that is left, and
//! it is injective in the free operand. `hashtrinity` shipped without one: 4,096 keys built from
//! published constants gave it **one** address, and its table answered 4,095 of 4,096 lookups
//! with another key's value.
//!
//! Inputs enter `mix2` with `wrapping_add`, never XOR -- see [`crate::coord`] for why.
//!
//! **Length ladder:** <= 16 bytes, one `mix2`; 17..=32, two overlapping halves; above that,
//! 32-byte rounds of two `mix2`s, one per lane. Sixteen bytes per `mix2` is the input-rate
//! ceiling.
//!
//! **Cost**, instructions per hash (`examples/fold_instr.rs`):
//!
//! | corpus | apeiron | foldhash |
//! |---|---|---|
//! | resref, 13 B | **23.1** | 24.1 |
//! | path, 42 B | **68.1** | 74.1 |
//! | long, 231 B | 199.3 | **180.3** |
//!
//! It cannot promise collision resistance: the constants are public and `2^64` addresses meet a
//! birthday bound at `2^32` keys. What it changes is what a collision *costs* -- see
//! [`crate::pepper`].

use crate::coord::{Coord, PHI, PSI};

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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The fold *is* the truncation: `lo ^ hi` of a 128-bit product keeps 64 bits on purpose, \
                      which is what makes a widening multiply a mixing step rather than a widening one."
    )]
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
    settle(absorb(pepper, b))
}

/// The last step, and it is not optional: a multiply then an xor-shift.
///
/// Multiplication in `Z/2^64` carries information **upward only**, so the top bits of a product
/// are the ones no later multiply can bring back. `2^63` is a zero divisor there with annihilator
/// `2Z`, which means flipping the top bit of a `mix2` operand moves the product by
/// `2^63 * (b & 1)` -- a function of the *other* operand's low bit and not of the flipped one at
/// all. The differences then cluster, which is what `SMHasher`'s `DiffDist` reports at bit 63:
/// 642,947 high-32 collisions against 512 expected, and 99.7% distribution bias.
///
/// No further multiply repairs that. The repair has to come from the other structure -- a
/// `GF(2)`-linear shift that moves information *down*, which is why splitmix alternates the two
/// algebras and why [`crate::coord::spread`] ends in `rotl(hi, 32) ^ lo`. Measured, distinct
/// high-32 differences over 200,000 keys with the top bit flipped:
///
/// | final step | bit 63 | |
/// |---|---|---|
/// | none | 167,240 | |
/// | `h ^ (h >> 32)` | 167,240 | a `GF(2)`-linear map permutes differences, it cannot unclump them |
/// | `h ^ rotr(h, 32)` | 199,947 | **unusable**: sends `(hi, lo)` to `(hi^lo, hi^lo)`, so the range is `2^32` |
/// | **`h * PHI` then `x ^ (x >> 32)`** | **199,997** | bijective, one multiply |
/// | `finish` (splitmix) | 199,995 | bijective, two multiplies |
///
/// A finalizer must be a bijection or it manufactures collisions of its own -- which is what the
/// rotate does, and why the best-scoring candidate was the wrong one.
#[expect(
    clippy::inline_always,
    reason = "the split into `absorb` and `settle` exists for readability; neither may become a \
              call, and `fold_bytes_with` is on every string lookup"
)]
#[inline(always)]
fn settle(h: u64) -> u64 {
    let x = h.wrapping_mul(PHI);
    x ^ (x >> 32)
}

/// The fold proper, before [`settle`].
#[expect(clippy::inline_always, reason = "as `settle`")]
#[inline(always)]
fn absorb(pepper: u64, b: &[u8]) -> u64 {
    let len = b.len();
    // The length enters through a multiply, so it changes every bit of the seed rather than the
    // low ones. Two keys of different lengths therefore start from unrelated states.
    // The pepper enters **multiplicatively**, and by two different odd constants, one per lane.
    //
    // Adding it made the key and the pepper interchangeable: with `mix2(a + seed, c + NU)` the
    // hash is a function of `a + seed` whenever two keys share `c`, so every pair with the same
    // sum collides. At one to three bytes `c` was always zero and *every* pair did -- SMHasher's
    // PerlinNoise keyset found 14,741,760 collisions in 16,777,216 hashes.
    //
    // Two weaker repairs failed and are worth recording. Making `c` key-derived at short lengths
    // fixed those but left it alive wherever `c` happens to be zero: a five-byte key reads `c`
    // from bytes 1..5, so key 0 under pepper 1 still met key 1 under pepper 0. Seeding the second
    // lane with `seed.rotate_left(32)` failed too, because a rotate is a bit permutation and so
    // commutes with the addition it was meant to break: key 0 under pepper 257 met key 256 under
    // pepper 0.
    //
    // A multiply does not commute with a translation. A collision now needs
    // `a1 + p1*PHI == a2 + p2*PHI` *and* `c1 + p1*PSI == c2 + p2*PSI`; for equal `c` the second
    // forces `p1 == p2` because `PSI` is odd and invertible, and then the first forces `a1 == a2`.
    // This is exactly the rule `Salt` records for the address bijection -- an additive secret is a
    // translation and buys nothing -- which the fold was not following.
    // Both lanes keep a constant offset and a length term; only the *pepper* changed, from
    // additive to multiplicative.
    //
    // Two cheaper shapes were tried and rejected. Dropping `PHI` from one lane and the length from
    // the other traded PerlinNoise's collisions for new ones in Sparse, Permutation and TwoBytes.
    // Deriving the second lane as `seed * PSI` -- one multiply instead of two -- fails
    // `every_input_bit_moves_half_the_output_at_every_length`, which averages over 512 keys and so
    // is measuring a real correlation between the lanes rather than a lucky draw.
    let seed = pepper
        .wrapping_mul(PHI)
        .wrapping_add(PHI)
        .wrapping_add((len as u64).wrapping_mul(NU));
    let seed2 = pepper
        .wrapping_mul(PSI)
        .wrapping_add(PSI)
        .wrapping_add((len as u64).wrapping_mul(PHI));

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
            // The second operand is the key again, rotated -- **not** zero, which is what it was.
            //
            // With `c = 0` the whole hash is `mix2(a + seed, NU)`, a function of `a + seed` alone,
            // so the key and the seed are additively interchangeable and any two pairs with the
            // same sum collide. SMHasher's PerlinNoise keyset hashes one key under many seeds and
            // found 14,741,760 collisions in 16,777,216 hashes; `examples/perlin.rs` reproduces
            // the number exactly. It is also a broken pepper, not merely a bad distribution: one
            // known (key, hash) pair on a short key yields `a + pepper`, hence the pepper, hence
            // every other short-key address.
            //
            // Longer keys never had it because `c` is key-derived there -- at four bytes `c == a`,
            // which is enough to tie the key into both operands. This is the same lesson `Salt`
            // records for the address bijection, in the fold: an additive secret is a translation.
            (a, a.rotate_left(32))
        } else {
            (0, 0)
        };
        return mix2(a.wrapping_add(seed), c.wrapping_add(seed2));
    }

    // 17..=32 bytes: two overlapping halves, one `mix2` each. The overlap is what avoids a padded
    // tail -- the middle bytes are read twice, which is harmless, and there is no branch per length.
    if len <= 32 {
        let x = mix2(word(&b[..8]).wrapping_add(seed), word(&b[8..16]).wrapping_add(seed2));
        let y = mix2(
            word(&b[len - 16..len - 8]).wrapping_add(x),
            word(&b[len - 8..]).wrapping_add(PHI),
        );
        return y;
    }

    // Longer: **thirty-two** bytes a round, two `mix2`s, one per sixteen.
    //
    // It was sixteen bytes a round with two `mix2`s, because the two lanes were made independent
    // for instruction-level parallelism and each read *both* words -- so the fold spent two
    // widening multiplies on every sixteen bytes where the input rate allows one. A `mix2` consumes
    // two words, so one per sixteen bytes is the ceiling and this now sits on it.
    //
    // Measured, 231-byte keys: **359.8 instructions per fold against foldhash's 179.3** before,
    // and the ratio in time was 2.20x. Instruction counts are the honest instrument here -- they do
    // not move with machine load, and the cycle measurements this was first diagnosed from were
    // taken on a busy machine and were not trustworthy.
    //
    // The lanes stay independent: `x` reads the round's first sixteen bytes, `y` the second, and
    // neither reads the other's output, so the multiplier is issued twice per round rather than
    // serialised behind itself.
    let mut x = seed;
    let mut y = seed2;
    let mut rest = b;
    while rest.len() > 32 {
        let a0 = word(&rest[..8]);
        let a1 = word(&rest[8..16]);
        let b0 = word(&rest[16..24]);
        let b1 = word(&rest[24..32]);
        let nx = mix2(x.wrapping_add(a0), NU.wrapping_add(a1));
        let ny = mix2(y.wrapping_add(b0), PHI.wrapping_add(b1));
        x = nx;
        y = ny;
        rest = &rest[32..];
    }
    // The last thirty-two bytes, overlapping whatever the loop consumed. `len > 32` here, so the
    // slice exists; the overlap covers those bytes twice rather than once, which is harmless and is
    // what a padded tail would otherwise cost.
    let tail = &b[len - 32..];
    let x = mix2(x.wrapping_add(word(&tail[..8])), NU.wrapping_add(word(&tail[8..16])));
    let y = mix2(y.wrapping_add(word(&tail[16..24])), PHI.wrapping_add(word(&tail[24..])));
    mix2(x, y.wrapping_add(NU))
}

impl Coord for str {
    const SPREAD: bool = true;
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
    const SPREAD: bool = true;
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
    const SPREAD: bool = true;
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
    const SPREAD: bool = true;
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
    const SPREAD: bool = true;
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
    const SPREAD: bool = T::SPREAD;
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Both halves of the `u128` are consumed -- `as u64` takes the low one, the shift takes \
                      the high one -- so nothing is dropped; the cast is how each half is named."
    )]
    fn coord(&self) -> u64 {
        mix2((*self as u64).wrapping_add(PHI), ((*self >> 64) as u64).wrapping_add(NU))
    }
}

impl Coord for i128 {
    #[inline]
    #[expect(
        clippy::cast_sign_loss,
        reason = "An `i128` is folded through the `u128` impl by reinterpretation, so a negative key and \
                      its two's-complement bit pattern get the same address. That is required, not tolerated: \
                      `Eq` agrees with the bit pattern here."
    )]
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`N` is 4,096 at most and `usize` is at least 32 bits on every target this \
                  builds for"
    )]
    fn a_zeroed_operand_does_not_collapse_the_fold() {
        const N: u64 = if cfg!(miri) { 256 } else { 4096 };

        let mut seen = std::collections::HashSet::new();
        for c in 0..N {
            seen.insert(mix2(0, c));
        }
        assert_eq!(seen.len(), N as usize, "mix2(0, .) must stay injective");

        let mut seen = std::collections::HashSet::new();
        for a in 0..N {
            seen.insert(mix2(a, 0));
        }
        assert_eq!(seen.len(), N as usize, "mix2(., 0) must stay injective");
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
        // Miri interprets every fold, and a hundred thousand of them is a ten-minute test. The
        // birthday bound is what this checks and it is a property of the fold, not of the count.
        let n = if cfg!(miri) { 2_000u32 } else { 100_000u32 };
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

    /// A key and the pepper must not be interchangeable.
    ///
    /// The defect `SMHasher`'s `PerlinNoise` keyset found: at one to three bytes the second `mix2`
    /// operand was `0`, so the whole hash was `mix2(a + seed, NU)` -- a function of `a + seed`,
    /// with the key and the seed additively swappable. Two pairs with the same sum collided, and a
    /// single known short key gave up the pepper.
    #[test]
    fn a_short_key_and_the_pepper_are_not_interchangeable() {
        // 512 x 512 x 8 folds is a second natively and minutes under Miri, where the point is
        // the unsafe and not the coverage. The crate's convention, as in `fold_is_deterministic`.
        const N: u64 = if cfg!(miri) { 48 } else { 512 };
        for len in 1..=8usize {
            let mut seen = std::collections::HashMap::new();
            for k in 0..N {
                let key = &k.to_le_bytes()[..len];
                for pepper in 0..N {
                    let h = fold_bytes_with(pepper, key);
                    let id = (u64::from_le_bytes({
                        let mut w = [0u8; 8];
                        w[..len].copy_from_slice(key);
                        w
                    }), pepper);
                    if let Some(prev) = seen.insert(h, id) {
                        assert_eq!(prev, id, "len {len}: {prev:?} and {id:?} share an address");
                    }
                }
            }
        }
    }

    /// Each input bit must move about half the output bits, **averaged over many keys**.
    ///
    /// One base key per length is one Bernoulli sample per bit: the fraction has a standard
    /// deviation of `sqrt(0.25 / 64) = 0.0625`, so over a few hundred bit positions the worst will
    /// stray past 0.75 by luck and the test passes or fails on which key it happened to pick. That
    /// is what it used to do.
    ///
    /// Averaging over `KEYS` base keys cuts the deviation by `sqrt(KEYS)` and lets the band be
    /// tight enough to mean something: a real bias shows, a lucky draw does not.
    #[test]
    #[expect(clippy::cast_possible_truncation, reason = "a key byte is a byte of a mixed word")]
    fn every_input_bit_moves_half_the_output_at_every_length() {
        const KEYS: u32 = if cfg!(miri) { 8 } else { 96 };
        const LENS: &[usize] = if cfg!(miri) {
            &[1, 4, 8, 16, 33, 64]
        } else {
            &[1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 200]
        };
        for &len in LENS {
            let mut worst = (0usize, 0.5f64);
            for bit in 0..len * 8 {
                let mut moved = 0u32;
                for k in 0..KEYS {
                    let base: Vec<u8> = (0..len)
                        .map(|i| {
                            let x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                ^ u64::from(k).wrapping_mul(0xD1B5_4A32_D192_ED03);
                            (x >> 24) as u8
                        })
                        .collect();
                    let mut flipped = base.clone();
                    flipped[bit / 8] ^= 1 << (bit % 8);
                    moved += (fold_bytes(&flipped) ^ fold_bytes(&base)).count_ones();
                }
                let frac = f64::from(moved) / f64::from(KEYS * 64);
                if (frac - 0.5).abs() > (worst.1 - 0.5f64).abs() {
                    worst = (bit, frac);
                }
            }
            // The band widens with the sample: 96 keys is 6144 bit-observations per input bit and
            // 0.44..=0.56 is tight on that; 8 keys is 512 and the same band would fail on sampling
            // noise alone. Miri runs this for the code path, not the statistic.
            let band = if cfg!(miri) { 0.36..=0.64 } else { 0.44..=0.56 };
            assert!(
                band.contains(&worst.1),
                "len {len}: bit {} moved {:.3} of the output, averaged over {KEYS} keys",
                worst.0,
                worst.1
            );
        }
    }

    /// Every length in the ladder must separate keys, including across a boundary. A round that
    /// consumed the wrong bytes would collide two keys that differ only in the region it skipped.
    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the truncation is the point: the key bytes are a repeating pattern derived from \
                  the index, and wrapping is what makes it repeat"
    )]
    fn no_two_keys_collide_across_the_length_ladder() {
        let mut seen = std::collections::HashMap::new();
        let (max_len, variants) = if cfg!(miri) { (40usize, 8u8) } else { (140usize, 64u8) };
        for len in 0..=max_len {
            for variant in 0..variants {
                let mut k: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(17)).collect();
                if len > 0 {
                    // Vary a byte at each end and in the middle, so a round that dropped a slice
                    // of the input shows up rather than being averaged away.
                    k[0] ^= variant;
                    k[len / 2] ^= variant.rotate_left(3);
                    k[len - 1] ^= variant.rotate_left(5);
                }
                let a = fold_bytes(&k);
                if let Some(prev) = seen.insert(a, k.clone()) {
                    assert_eq!(prev, k, "two distinct keys share an address: {prev:?} and {k:?}");
                }
            }
        }
    }
}
