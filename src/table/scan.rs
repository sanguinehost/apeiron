//! `u8`-lane tag-line scans for the growable table's group probe.
//!
//! A group is `N` lanes and a tag line is one `u8` per lane. A scan answers one of two questions
//! about a line, as a lane bitmask: which lanes equal a tag ([`match_tag`]), and which lanes are
//! empty ([`match_empty`]). Bit `i` of the mask is lane `i`.
//!
//! The equality and zero compares are SWAR over `u64` words, eight little-endian lanes per word.
//! They are exact: every lane's answer is independent of its neighbours, so a lane whose value
//! differs from the compared byte by one cannot borrow into the neighbouring lane and be reported
//! as a match. The scalar tail scans the final fewer-than-eight lanes, so a group whose width is
//! not a multiple of eight is handled on the same real path rather than by a second locate.

/// Lanes carried by one `u64` word.
pub const LANES_PER_WORD: usize = 8;

/// One `0x01` per lane: the value that turns a per-lane subtraction into a per-lane decrement.
const LO: u64 = 0x0101_0101_0101_0101;
/// One `0x80` per lane: the marker bit, and the value that keeps a borrow inside its lane.
const HI: u64 = 0x8080_8080_8080_8080;

/// Gather eight lane markers into the low eight bits.
///
/// A compare leaves `0x80` at bit `7 + 8i` of lane `i`. Multiplying by `GATHER` moves lane `i`'s
/// marker by `49 - 7i`; the cross terms land on distinct positions, so no carry is produced and
/// `>> 56` reads one marker per lane.
const GATHER: u64 = 0x0002_0408_1020_4081;

/// `0x80` in lane `i` of `word` iff that lane equals `byte`, zero elsewhere.
///
/// The textbook `(x - LO) & !x & HI` is a superset: a matching lane borrows out of the
/// subtraction, and a neighbour whose difference from `byte` is one is then flagged too. Setting
/// every lane's top bit before subtracting stops the borrow at the lane boundary, and or-ing the
/// unmodified `x` back in restores the lanes the forcing overwrote.
#[inline]
const fn eq_swar(word: u64, byte: u8) -> u64 {
    let x = word ^ LO.wrapping_mul(byte as u64);
    !(x | (x | HI).wrapping_sub(LO)) & HI
}

/// `0x80` in lane `i` of `word` iff that lane is zero, zero elsewhere.
///
/// The same exact construction as [`eq_swar`] at `byte = 0`, where `x` is `word` itself.
#[inline]
const fn zero_swar(word: u64) -> u64 {
    !(word | (word | HI).wrapping_sub(LO)) & HI
}

/// Compress the eight lane markers of `marked` into the low eight bits.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the multiply gathers one marker per lane into the top byte, so the low eight bits \
              hold every marker and the cast drops none of them"
)]
const fn gather(marked: u64) -> u8 {
    (marked.wrapping_mul(GATHER) >> 56) as u8
}

/// The eight bytes at `off` as one little-endian word; lane `i` is `line[off + i]`.
///
/// Temporal O(1): one eight-byte copy and one byte-order read, a fixed count. Spatial O(1): one
/// eight-byte buffer.
#[inline]
fn word_at(line: &[u8], off: usize) -> u64 {
    let mut buf = [0u8; LANES_PER_WORD];
    buf.copy_from_slice(&line[off..off + LANES_PER_WORD]);
    u64::from_le_bytes(buf)
}

/// Lane mask of the lanes of `line` equal to `tag`.
///
/// Temporal O(L/8) word compares plus O(L mod 8) scalar compares for `L = line.len()` lanes,
/// counted in byte comparisons; the loop advances `off` by eight and the tail by one. Spatial
/// O(1): one word temporary and one mask.
#[inline]
#[must_use]
pub fn match_tag(line: &[u8], tag: u8) -> u64 {
    let mut mask = 0u64;
    let mut off = 0usize;
    let mut base = 0u32;
    while off + LANES_PER_WORD <= line.len() && base + 8 <= 64 {
        let word = word_at(line, off);
        mask |= u64::from(gather(eq_swar(word, tag))) << base;
        off += LANES_PER_WORD;
        base += 8;
    }
    while off < line.len() && base < 64 {
        if line[off] == tag {
            mask |= 1u64 << base;
        }
        off += 1;
        base += 1;
    }
    mask
}

/// Lane mask of the empty lanes of `line`.
///
/// The same shape as [`match_tag`]: O(L/8) word compares plus an O(L mod 8) scalar tail, in byte
/// comparisons; spatial O(1).
#[inline]
#[must_use]
pub fn match_empty(line: &[u8]) -> u64 {
    let mut mask = 0u64;
    let mut off = 0usize;
    let mut base = 0u32;
    while off + LANES_PER_WORD <= line.len() && base + 8 <= 64 {
        let word = word_at(line, off);
        mask |= u64::from(gather(zero_swar(word))) << base;
        off += LANES_PER_WORD;
        base += 8;
    }
    while off < line.len() && base < 64 {
        if line[off] == 0 {
            mask |= 1u64 << base;
        }
        off += 1;
        base += 1;
    }
    mask
}

/// Least significant set-lane index of `bits`, or 64 when no lane is set.
///
/// The read path calls this once per fingerprint match, so its cost is the per-present-hit lane
/// search. The bit-at-a-time loop it replaces takes a branch per four lanes and mispredicts at the
/// match lane; this halving search takes six constant branch tests and no data-dependent trip
/// count. It is exact for every `bits`: when the low half is zero the lowest set lane lies in the
/// high half, so shifting the half in and adding its width leaves the lowest set lane's position.
/// Temporal O(6) branch tests at a fixed 64-bit width, spatial O(1).
#[inline]
#[must_use]
#[allow(
    clippy::verbose_bit_mask,
    reason = "the halving search must stay a shift-and-mask loop; `trailing_zeros` is the \
              unmodeled intrinsic this exact loop replaces, so the lint's rewrite is not available"
)]
pub fn least_set_lane(bits: u64) -> usize {
    if bits == 0 {
        return 64;
    }
    let mut lane = 0usize;
    let mut rest = bits;
    if rest & 0xFFFF_FFFF == 0 {
        rest >>= 32;
        lane += 32;
    }
    if rest & 0xFFFF == 0 {
        rest >>= 16;
        lane += 16;
    }
    if rest & 0xFF == 0 {
        rest >>= 8;
        lane += 8;
    }
    if rest & 0xF == 0 {
        rest >>= 4;
        lane += 4;
    }
    if rest & 0x3 == 0 {
        rest >>= 2;
        lane += 2;
    }
    if rest & 0x1 == 0 {
        lane += 1;
    }
    lane
}

#[cfg(test)]
mod tests {
    use super::{match_empty, match_tag};

    /// The scalar reference the SWAR scan must agree with: one comparison per lane.
    fn scalar_tag(line: &[u8], tag: u8) -> u64 {
        let mut mask = 0u64;
        let mut i = 0usize;
        while i < line.len() {
            if line[i] == tag {
                mask |= 1u64 << i;
            }
            i += 1;
        }
        mask
    }

    fn scalar_empty(line: &[u8]) -> u64 {
        let mut mask = 0u64;
        let mut i = 0usize;
        while i < line.len() {
            if line[i] == 0 {
                mask |= 1u64 << i;
            }
            i += 1;
        }
        mask
    }

    /// Control 1: a lane equal to a neighbour's high byte at a word boundary is not flagged.
    ///
    /// `0xB0` is one less than `0xB1`, so the loose `(x - LO) & !x & HI` form borrows out of
    /// lane 0 and reports lane 1. The exact form reports only lane 0.
    #[test]
    fn a_neighbouring_high_byte_is_not_a_match() {
        let mut line = [0u8; 16];
        line[0] = 0xB1;
        line[1] = 0xB0;
        line[8] = 0xB0;
        line[9] = 0xB1;
        assert_eq!(match_tag(&line, 0xB1), 0b10_0000_0001);
        assert_eq!(match_tag(&line, 0xB1), scalar_tag(&line, 0xB1));

        // The fixture must separate the exact form from the loose one, or the control would pass
        // even if `eq_swar` regressed to the superset form.
        let mut buf = [0u8; super::LANES_PER_WORD];
        buf.copy_from_slice(&line[..super::LANES_PER_WORD]);
        let word = u64::from_le_bytes(buf);
        let x = word ^ super::LO.wrapping_mul(u64::from(0xB1u8));
        let loose = x.wrapping_sub(super::LO) & !x & super::HI;
        let exact = super::eq_swar(word, 0xB1);
        assert_ne!(loose & 0x8000, 0, "the loose form flags the neighbour");
        assert_eq!(exact & 0x8000, 0, "the exact form does not");
    }

    /// The SWAR scan and the scalar scan agree on every tag and every lane value.
    #[test]
    fn swar_and_scalar_agree_on_every_tag() {
        let mut line = [0u8; 16];
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut i = 0usize;
        while i < line.len() {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            line[i] = (seed >> 33).to_le_bytes()[0];
            i += 1;
        }
        let mut value = 0u16;
        while value < 256 {
            let tag = u8::try_from(value).expect("0..256 fits u8");
            assert_eq!(match_tag(&line, tag), scalar_tag(&line, tag), "tag {tag}");
            value += 1;
        }
        // Force every lane to the same tag, so the all-match and all-empty boundaries are seen.
        let full = [0xA5u8; 16];
        assert_eq!(match_tag(&full, 0xA5), 0xFFFF);
        assert_eq!(match_tag(&full, 0x00), 0);
    }

    /// A partial final word is scanned by the scalar tail and still agrees.
    #[test]
    fn the_scalar_tail_scans_a_partial_word() {
        let line = [7u8, 0, 9, 0, 7, 7, 0, 0, 7, 0, 0, 9];
        assert_eq!(match_tag(&line, 7), 0b1_0011_0001);
        assert_eq!(match_empty(&line), 0b110_1100_1010);
        assert_eq!(match_tag(&line, 7), scalar_tag(&line, 7));
        assert_eq!(match_empty(&line), scalar_empty(&line));
    }

    /// The scalar reference for the least set lane: one shift test per lane.
    fn scalar_least_set_lane(bits: u64) -> usize {
        let mut lane = 0usize;
        while lane < 64 && ((bits >> lane) & 1) == 0 {
            lane += 1;
        }
        lane
    }

    /// The halving lane search agrees with the bit-at-a-time reference on every single bit and on
    /// deterministic multi-bit masks.
    ///
    /// The single-bit sweep pins every lane position, including the 32/16/8/4/2/1 halving
    /// boundaries; the multi-bit masks pin that the *least* set lane is selected, not any set lane.
    #[test]
    fn least_set_lane_matches_the_scalar_search() {
        assert_eq!(super::least_set_lane(0), 64);
        let mut lane = 0u32;
        while lane < 64 {
            let bits = 1u64 << lane;
            assert_eq!(
                super::least_set_lane(bits),
                lane as usize,
                "single bit {lane}"
            );
            assert_eq!(
                super::least_set_lane(bits),
                scalar_least_set_lane(bits),
                "single bit {lane} against the reference"
            );
            lane += 1;
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut case = 0usize;
        while case < 4096 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let bits = seed | (1u64 << (case % 64));
            assert_eq!(
                super::least_set_lane(bits),
                scalar_least_set_lane(bits),
                "mask {bits:#018x}"
            );
            case += 1;
        }
    }
}
