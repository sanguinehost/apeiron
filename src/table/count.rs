#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "extraction-facing scalar forms: the pinned extractor emits an `axiom` for `TryFrom` \
              and the narrowing casts, so the explicit forms are kept and these lints stand. \
              `COORDS_BITS`-wide words are masked or guarded by `admit_coord` at every narrowing."
)]
//! Coordinate-keyed counting container: one single-region, open-addressed slot array over packed
//! `(Sigma, u32)` coordinate words, with the slot tag as the sole occupancy authority.
//!
//! The container answers one question — how many times has this exact coordinate, on this exact
//! face, been merged — and it answers it without retaining a key. One `u64` per live coordinate
//! carries both the face and the coordinate, so no `(Sigma, K)` widening is materialized per query.
//!
//! The codec and its frozen boundary values are stated once, in [`U32CoordWord`] and
//! [`SENTINELS`]; every other item here is a consumer of that statement.

use core::marker::PhantomData;

use crate::cell::Sigma;

use super::resources::try_reserve_exact;
use super::StorageError;

/// Width of the coordinate half of a stored word. The coordinate occupies bits `0..32`.
pub const COORD_BITS: u32 = 32;
/// Mask of the coordinate half of a stored word.
pub const COORD_MASK: u64 = (1 << COORD_BITS) - 1;
/// Live bit of a slot tag. Every written tag carries it; no vacancy marker is carried in the word.
pub const TAG_LIVE: u8 = 0x80;
/// Count returned for a coordinate that was never written: the additive identity of `Int64`.
pub const COUNT_DEFAULT: i64 = 0;
/// Smallest region width that can resolve a collision. A narrower region cannot hold a probe pair,
/// so construction refuses it as an invalid group width (`StorageError::SizeOverflow`'s own words).
pub const MIN_REGION_SLOTS: usize = 4;
/// Largest region this container will build. Growth stops here and a further absent coordinate is
/// refused rather than resized.
pub const MAX_REGION_SLOTS: usize = 1 << 24;

/// The frozen sentinel and boundary values of [`U32CoordWord`], with the admitted-domain edge.
pub const SENTINELS: [(Sigma, u64, bool); 4] = [
    (Sigma::Elliptic, 0, true),
    (Sigma::Parabolic, 1 << 31, true),
    (Sigma::Hyperbolic, (1 << COORD_BITS) - 1, true),
    (Sigma::Hyperbolic, 1 << COORD_BITS, false),
];

/// Refusal outcomes of the counting face.
///
/// A total widening of [`StorageError`] by [`CountRefusal::CountOverflow`]: the first two arms are
/// the checked-layout and failed-reservation refusals the growable table already returns
/// (the resource/refusal contract), and
/// the third is the count merge's checked addition. No arm wraps and no arm commits a partial write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CountRefusal {
    /// A checked calculation did not fit: a coordinate above the admitted `u32` domain, a region
    /// width below [`MIN_REGION_SLOTS`] or above [`MAX_REGION_SLOTS`], or a growth width whose
    /// doubling overflowed `usize`.
    SizeOverflow,
    /// A checked reservation could not be obtained; the committed region is unchanged.
    AllocationFailed,
    /// The new-first merge `new + old` left the `i64` carrier; the committed count is unchanged.
    CountOverflow,
}

impl From<StorageError> for CountRefusal {
    /// The mapping is total. The third [`StorageError`] arm, `ShardFull`, is declared by its own
    /// source to be compatibility-only and not returned by a growable table
    /// (`StorageError::ShardFull`: "this growable table does not return it"), and the only producer
    /// of a [`StorageError`] in this module is `try_reserve_exact`, which returns the first two
    /// arms. It therefore has no counting analogue and is folded onto `SizeOverflow`.
    fn from(error: StorageError) -> Self {
        match error {
            StorageError::SizeOverflow | StorageError::ShardFull => Self::SizeOverflow,
            StorageError::AllocationFailed => Self::AllocationFailed,
        }
    }
}

/// The set of coordinate codecs this container may store under, sealed.
///
/// The required law is, on the admitted domain: `decode(s, encode(s, c)) == c`, and the required
/// separation is: `encode(s1, c1) == encode(s2, c2)` implies `s1 == s2` and `c1 == c2`. The seal
/// means no unreviewed instance can be selected.
pub trait CountKeyCodec: sealed::Sealed {
    /// The stored word for one admitted `(face, coordinate)` key. Total: its domain is the admitted
    /// `u32` coordinate, so there is no `Option` and no refusal arm.
    fn encode(sigma: Sigma, coord: u32) -> u64;
    /// The identity comparison, and the only operation a probe performs against a live slot.
    fn matches(sigma: Sigma, stored: u64, coord: u32) -> bool;
    /// The verified inverse of [`Self::encode`] on its image. Reached only after
    /// [`Self::matches`] returned true, so the face half is already known to agree.
    fn decode(sigma: Sigma, stored: u64) -> u32;
}

/// The only admitted codec instance, in the shape of `ExactCodec`/`I64WordCodec`
/// (`src/table/keyrep.rs`) so that no unreviewed instance can be selected.
///
/// **Law.** `encode(s, c) = (u64::from((s.value() + 1) as u32) << 32) + u64::from(c)`.
/// `Sigma::value()` (`src/cell.rs`) is `-1`, `0`, `1` on `Elliptic`, `Parabolic`, `Hyperbolic`, so
/// `value() + 1` is the injective face tag `0`, `1`, `2`. The face half occupies bits 32-33 and the
/// coordinate half bits 0-31, two disjoint ranges, so the sum above is the bitwise-or packing and
/// the image is exactly `{0, 1, 2} x 2^32` — `12_884_901_888` distinct words.
///
/// The two obligations this law discharges:
///
/// * `decode(s, encode(s, c)) = c` for every `s` and every `c < 2^32`, because `decode` reads the
///   low half and the low half is `c`.
/// * `encode(s1, c1) = encode(s2, c2) ⟹ s1 = s2 ∧ c1 = c2`, by disjointness of the two halves. So
///   two admitted keys never share a stored word, and `(s, c)` and `(s', c)` differ for `s ≠ s'`
///   at one coordinate and one placement.
///
/// **Sentinels.** See [`SENTINELS`] for the executed table. `0` is `encode(Elliptic, 0)`, so a
/// stored word of `0` is a *live* coordinate and never a vacancy marker: occupancy is the slot tag
/// ([`CountSlot::is_live`]), never the word. Bits 34-63 lie outside the image for every admitted
/// key, so no stored word above `0x2_FFFF_FFFF` is a live coordinate.
pub struct U32CoordWord;

impl sealed::Sealed for U32CoordWord {}

impl CountKeyCodec for U32CoordWord {
    #[inline]
    fn encode(sigma: Sigma, coord: u32) -> u64 {
        (u64::from((sigma.value() + 1) as u32) << COORD_BITS) + u64::from(coord)
    }

    #[inline]
    fn matches(sigma: Sigma, stored: u64, coord: u32) -> bool {
        stored == Self::encode(sigma, coord)
    }

    #[inline]
    fn decode(_: Sigma, stored: u64) -> u32 {
        (stored & COORD_MASK) as u32
    }
}

mod sealed {
    /// Prevents any type outside this module from implementing [`super::CountKeyCodec`].
    pub trait Sealed {}
}

/// Admits a raw coordinate tag into the packed `u32` word, or refuses it.
///
/// `2^32` is the first refused coordinate, matching the first `u64` tag
/// `u32::try_from(key.tag).ok()` rejects in the harness's own address adapter. There is no
/// truncation to `0` and no wrap to `2^32 - 1`: the arm is a refusal, and because the refusal is
/// the caller's first action no slot is touched.
///
/// The refusal is [`CountRefusal::SizeOverflow`], the frozen set's checked-narrowing arm. The
/// contract froze three variants and none of them names a coordinate-domain arm, so the domain
/// refusal is carried by the closest one rather than by an invented fourth.
///
/// # Errors
///
/// [`CountRefusal::SizeOverflow`] when `raw` is `2^32` or above. `Result` is already
/// `#[must_use]`, so no attribute is repeated here.
#[inline]
pub fn admit_coord(raw: u64) -> Result<u32, CountRefusal> {
    if raw > u64::from(u32::MAX) {
        return Err(CountRefusal::SizeOverflow);
    }
    Ok(raw as u32)
}

/// One open-addressed slot: the occupancy authority, and, when live, the packed word and the count.
///
/// The word of a vacant slot is `0`, which is also `encode(Elliptic, 0)`. Nothing reads it: the tag
/// decides occupancy. That is what keeps the zero coordinate live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CountSlot {
    /// Occupancy authority. `TAG_LIVE` set means occupied; the word below is then the stored word.
    pub tag: u8,
    /// `encode(sigma, coord)` of the live coordinate; meaningless and zero when vacant.
    pub word: u64,
    /// The merged count at the live coordinate; zero when vacant.
    pub count: i64,
}

impl CountSlot {
    /// The vacant slot. Its word is `0` and carries no meaning; see [`CountSlot`].
    pub const VACANT: Self = Self {
        tag: 0,
        word: 0,
        count: 0,
    };

    /// Whether this slot is occupied. The **sole** occupancy authority in this module.
    #[inline]
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.tag & TAG_LIVE != 0
    }
}

/// The result of one forward probe of the region at a stored word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe {
    /// A live slot whose word is equal: the identity match, at this index.
    Matched(usize),
    /// The probe reached a vacant slot, ending it: the word is absent, and this index is free.
    Vacant(usize),
    /// The probe stepped `slots.len()` times without a match or a vacancy: the region is full.
    Absent,
}

/// One single-region open-addressed counting table over packed `(Sigma, u32)` coordinate words.
///
/// Occupancy is the slot tag. Identity is the packed word equality, decided by
/// [`CountKeyCodec::matches`]. Probing is linear from a spread start, with no tombstone and no
/// deletion, so a probe that steps the whole region width has proved the coordinate absent and the
/// region full.
#[derive(Debug)]
pub struct CountingTable<C: CountKeyCodec = U32CoordWord> {
    slots: Vec<CountSlot>,
    live: usize,
    codec: PhantomData<C>,
}

impl<C: CountKeyCodec> CountingTable<C> {
    /// Builds an empty table over a region of `region_slots` slots, all vacant.
    ///
    /// # Complexity
    ///
    /// Temporal `O(region_slots)` in slot writes: one `try_reserve_exact` reservation, one
    /// `resize` filling the reserved region, and no probe. Spatial `O(region_slots)` slots, one tag
    /// byte plus one packed word and one `i64` count each.
    ///
    /// # Errors
    ///
    /// [`CountRefusal::SizeOverflow`] when `region_slots` is below [`MIN_REGION_SLOTS`] or above
    /// [`MAX_REGION_SLOTS`], and [`CountRefusal::AllocationFailed`] when the reservation fails. On
    /// either arm no table exists, so there is no committed state to preserve.
    pub fn new(region_slots: usize) -> Result<Self, CountRefusal> {
        if !(MIN_REGION_SLOTS..=MAX_REGION_SLOTS).contains(&region_slots) {
            return Err(CountRefusal::SizeOverflow);
        }
        let mut slots: Vec<CountSlot> = Vec::new();
        try_reserve_exact(&mut slots, region_slots).map_err(CountRefusal::from)?;
        slots.resize(region_slots, CountSlot::VACANT);
        Ok(Self {
            slots,
            live: 0,
            codec: PhantomData,
        })
    }

    /// Number of live coordinates.
    #[inline]
    #[must_use]
    pub fn live(&self) -> usize {
        self.live
    }

    /// The merged count at `(sigma, raw_coord)`, or [`COUNT_DEFAULT`] if that coordinate was never
    /// written. Total: absence is the default, which is `HMap K V = K → V` read at its default.
    ///
    /// # Complexity
    ///
    /// Temporal `O(1)` in the region width, counted in slot comparisons: one `wrapping_mul` and one
    /// remainder to compute the probe start, then a single forward pass of at most one comparison
    /// per slot. The pass is `for step in 0..slots.len()`, so it is bounded by the region length
    /// and closes without a break condition; no slot is written and nothing is allocated.
    ///
    /// # Errors
    ///
    /// [`CountRefusal::SizeOverflow`] when `raw_coord` is outside the admitted `u32` domain
    /// (`2^32` and above).
    pub fn get(&self, sigma: Sigma, raw_coord: u64) -> Result<i64, CountRefusal> {
        let coord = admit_coord(raw_coord)?;
        match self.probe(C::encode(sigma, coord)) {
            Probe::Matched(index) => Ok(self.slots[index].count),
            Probe::Vacant(_) | Probe::Absent => Ok(COUNT_DEFAULT),
        }
    }

    /// New-first `insertWith` at one coordinate: merges `new_count` into the committed count as
    /// `f(new, old) = new + old`, in the argument order of `Apeiron.Table.insertWith`
    /// (`insertWith f k v m = fun x => if x = k then f v (m x) else m x`,
    /// `../sanguine-apeiron/proof/Apeiron/Table/Basic.lean:80-81`). Returns the committed count
    /// after the merge.
    ///
    /// Order: locate → merge-checked-plan → reserve → relocate without merge → merge once → commit.
    /// The merge is the only fallible step on the matching arm and it runs before any write, and
    /// growth relocates live words and counts without invoking a merge, so a refusal on either arm
    /// leaves the committed state unchanged.
    ///
    /// # Complexity
    ///
    /// Temporal `O(1)` in the region width on the merge and in-place-write arms, counted in slot
    /// comparisons: one pass of at most one comparison per slot, as [`Self::get`]. On the growth arm
    /// one further pass relocates the `live` live slots into a region of `2 * slots.len()`, so the
    /// worst single call is `O(live)` comparisons and the amortized cost over a run that fills the
    /// region is `O(1)` per insert, because the widths double and the sum of `1 + 2 + 4 + … + n`
    /// is `2n - 1` for `n` inserts. Spatial `O(region_slots)` slots, unchanged by relocation
    /// because the fresh region is reserved before the old one is released.
    ///
    /// # Errors
    ///
    /// [`CountRefusal::SizeOverflow`] when `raw_coord` is outside the admitted `u32` domain, or
    /// when the region is full and already at [`MAX_REGION_SLOTS`];
    /// [`CountRefusal::AllocationFailed`] when the growth reservation fails; and
    /// [`CountRefusal::CountOverflow`] when `new_count + committed` leaves `i64`. On every arm the
    /// committed region, its counts and its live count are unchanged.
    pub fn insert_with(
        &mut self,
        sigma: Sigma,
        raw_coord: u64,
        new_count: i64,
    ) -> Result<i64, CountRefusal> {
        let coord = admit_coord(raw_coord)?;
        let word = C::encode(sigma, coord);
        match self.probe(word) {
            Probe::Matched(index) => {
                let merged = new_count
                    .checked_add(self.slots[index].count)
                    .ok_or(CountRefusal::CountOverflow)?;
                self.slots[index].count = merged;
                Ok(merged)
            }
            Probe::Vacant(index) => {
                self.slots[index] = CountSlot {
                    tag: TAG_LIVE,
                    word,
                    count: new_count,
                };
                self.live += 1;
                Ok(new_count)
            }
            Probe::Absent => self.insert_absent(word, new_count),
        }
    }

    /// The growth arm: the region is full, so reserve a doubled region, relocate into it without
    /// invoking a merge, then merge the absent coordinate exactly once.
    fn insert_absent(&mut self, word: u64, new_count: i64) -> Result<i64, CountRefusal> {
        if self.slots.len() >= MAX_REGION_SLOTS {
            return Err(CountRefusal::SizeOverflow);
        }
        let next = self
            .slots
            .len()
            .checked_mul(2)
            .ok_or(CountRefusal::SizeOverflow)?;
        self.relocate_into(next)?;
        match self.probe(word) {
            Probe::Vacant(index) => {
                self.slots[index] = CountSlot {
                    tag: TAG_LIVE,
                    word,
                    count: new_count,
                };
                self.live += 1;
                Ok(new_count)
            }
            // Unreachable: `next >= 2 * slots.len() > live`, so the probe of a doubled region finds
            // a vacancy. The arm is a refusal rather than an unwrap so the fail-closed shape holds.
            Probe::Matched(_) | Probe::Absent => Err(CountRefusal::SizeOverflow),
        }
    }

    /// Reserves a region of `next_len` slots and relocates every live word and count into it.
    ///
    /// Relocation invokes no merge: each live slot's word is distinct and is placed once into a
    /// region holding at most `live < next_len` occupied slots. The commit is the single assignment
    /// of the last line, so the refusal arm leaves `self.slots` untouched.
    ///
    /// # Complexity
    ///
    /// Temporal `O(live + next_len)` slot comparisons and writes: one pass over the old region and
    /// one relocation probe of at most `next_len` comparisons per live slot, over a bounded number
    /// of doubling steps. Spatial `O(next_len)` slots, reserved before the old region is released.
    fn relocate_into(&mut self, next_len: usize) -> Result<(), CountRefusal> {
        let mut fresh: Vec<CountSlot> = Vec::new();
        try_reserve_exact(&mut fresh, next_len).map_err(CountRefusal::from)?;
        fresh.resize(next_len, CountSlot::VACANT);
        for slot in &self.slots {
            if !slot.is_live() {
                continue;
            }
            match probe_in(&fresh, slot.word) {
                Probe::Vacant(index) => fresh[index] = *slot,
                // Unreachable for the checked layout: the live words are pairwise distinct and
                // fewer than `next_len` slots are occupied, so a forward probe cannot wrap.
                Probe::Matched(_) | Probe::Absent => return Err(CountRefusal::SizeOverflow),
            }
        }
        self.slots = fresh;
        Ok(())
    }

    /// One forward probe of the committed region at `word`.
    #[inline]
    fn probe(&self, word: u64) -> Probe {
        probe_in(&self.slots, word)
    }
}

/// The start index of the probe at `word` in a region of `width` slots.
///
/// The packed word is spread by the same odd constant `crate::coord::PHI` that
/// `src/table/state.rs` spreads `place` with, and the high half is reduced modulo the width, so
/// both the face half and the coordinate high bits reach the start index.
#[inline]
fn probe_start(word: u64, width: usize) -> usize {
    let spread = word.wrapping_mul(crate::coord::PHI);
    ((spread >> COORD_BITS) % width as u64) as usize
}

/// One forward linear probe of `slots` at `word`, as [`Probe`].
///
/// # Complexity
///
/// Temporal `O(width)` slot comparisons: the loop is `for step in 0..width`, so it performs at
/// most one comparison per slot, exits at the first match or vacancy, and closes at `width`
/// iterations when neither is found. Spatial `O(1)`.
fn probe_in(slots: &[CountSlot], word: u64) -> Probe {
    let width = slots.len();
    if width < MIN_REGION_SLOTS {
        return Probe::Absent;
    }
    let start = probe_start(word, width);
    for step in 0..width {
        let index = (start + step) % width;
        let slot = slots[index];
        if !slot.is_live() {
            return Probe::Vacant(index);
        }
        if slot.word == word {
            return Probe::Matched(index);
        }
    }
    Probe::Absent
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        admit_coord, probe_in, probe_start, CountKeyCodec, CountRefusal, CountSlot, CountingTable,
        Probe, U32CoordWord, COORD_BITS, COORD_MASK, COUNT_DEFAULT, MAX_REGION_SLOTS,
        MIN_REGION_SLOTS, TAG_LIVE,
    };
    use crate::cell::Sigma;

    const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

    /// The admitted sentinel and boundary population: the zero coordinate, the `2^31` sign-change
    /// coordinate, the `2^32 - 1` admitted-domain edge, and coordinates either side of a power of
    /// two so a low-bits-only start index would collide them all.
    const SENTINEL_COORDS: [u32; 10] = [
        0,
        1,
        2,
        0x0000_FFFF,
        0x0001_0000,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_0000,
        0xFFFF_FFFE,
        0xFFFF_FFFF,
    ];

    /// The first refused coordinate, `2^32`.
    const FIRST_REFUSED: u64 = 1 << COORD_BITS;

    /// The oracle's own face rank, written without `Sigma::value()` and without the container's
    /// codec, so the oracle is independent of the encoding it checks.
    fn oracle_face(sigma: Sigma) -> u8 {
        match sigma {
            Sigma::Elliptic => 0,
            Sigma::Parabolic => 1,
            Sigma::Hyperbolic => 2,
        }
    }

    /// The oracle's inverse face rank.
    fn sigma_of(face: u8) -> Sigma {
        match face {
            0 => Sigma::Elliptic,
            1 => Sigma::Parabolic,
            _ => Sigma::Hyperbolic,
        }
    }

    /// The independent full-key oracle: an ordered map over the whole `(face, coordinate)` key.
    ///
    /// It is not the container's readback and it is not a hash — it never calls
    /// [`CountingTable::get`], never calls the codec, and orders its keys by comparison.
    type Oracle = BTreeMap<(u8, u32), i64>;

    fn oracle_merge(oracle: &mut Oracle, sigma: Sigma, coord: u32, new_count: i64) {
        let entry = oracle.entry((oracle_face(sigma), coord)).or_insert(0);
        *entry = entry.wrapping_add(new_count);
    }

    fn oracle_expect(oracle: &Oracle, sigma: Sigma, coord: u32) -> i64 {
        oracle
            .get(&(oracle_face(sigma), coord))
            .copied()
            .unwrap_or(COUNT_DEFAULT)
    }

    fn table_of(region_slots: usize) -> CountingTable<U32CoordWord> {
        CountingTable::<U32CoordWord>::new(region_slots).unwrap()
    }

    #[test]
    fn counting_words_are_injective_across_faces() {
        for sigma in FACES {
            let face_half = sigma.value() + 1;
            assert_eq!(
                face_half,
                i64::from(oracle_face(sigma)),
                "the codec's face tag is not the injective rank of Sigma::value() + 1"
            );
            for coord in SENTINEL_COORDS {
                let word = U32CoordWord::encode(sigma, coord);
                assert_eq!(
                    i64::try_from(word >> COORD_BITS).unwrap(),
                    face_half,
                    "the face half of encode({sigma:?}, {coord}) is not sigma.value() + 1"
                );
                assert_eq!(
                    word & COORD_MASK,
                    u64::from(coord),
                    "the coordinate half of encode({sigma:?}, {coord}) is not c"
                );
                assert_eq!(
                    U32CoordWord::decode(sigma, word),
                    coord,
                    "law decode(s, encode(s, c)) = c failed at ({sigma:?}, {coord})"
                );
                assert!(
                    U32CoordWord::matches(sigma, word, coord),
                    "matches refused the word encode just produced"
                );
                assert!(
                    word <= 0x2_FFFF_FFFF,
                    "encode({sigma:?}, {coord}) = {word} left the image: bits 34-63 are outside it"
                );
            }
        }

        // Two admitted keys may never share a stored word.
        let mut words: Vec<u64> = Vec::new();
        for sigma in FACES {
            for coord in SENTINEL_COORDS {
                words.push(U32CoordWord::encode(sigma, coord));
            }
        }
        let population = words.len();
        words.sort_unstable();
        words.dedup();
        assert_eq!(
            words.len(),
            population,
            "two admitted keys share a stored word: the encoding is not injective"
        );
        assert_eq!(population, 30, "the injectivity population shrank");
    }

    #[test]
    fn zero_word_is_a_live_coordinate_not_a_vacancy() {
        assert_eq!(
            U32CoordWord::encode(Sigma::Elliptic, 0),
            0,
            "the zero sentinel is no longer the stored word of (Elliptic, 0)"
        );
        let mut table = table_of(8);
        assert_eq!(table.insert_with(Sigma::Elliptic, 0, 5).unwrap(), 5);
        // New-first merge f(new, old) = new + old: 3 + 5, not 5 + 3 read as a difference.
        assert_eq!(table.insert_with(Sigma::Elliptic, 0, 3).unwrap(), 8);
        assert_eq!(
            table.get(Sigma::Elliptic, 0).unwrap(),
            8,
            "the live zero word was read as a vacancy"
        );
        assert_eq!(
            table.live(),
            1,
            "the zero coordinate did not occupy exactly one slot"
        );

        // Occupancy is the tag: the live slot's word is 0 and the seven vacant slots also carry 0.
        assert_eq!(
            table.slots.iter().filter(|slot| slot.is_live()).count(),
            1,
            "a slot was read as occupied that the tag says is vacant"
        );
        assert!(
            table
                .slots
                .iter()
                .any(|slot| slot.is_live() && slot.word == 0 && slot.tag == TAG_LIVE),
            "no live slot holds the zero word"
        );
        assert_eq!(
            table.slots.iter().filter(|slot| !slot.is_live()).count(),
            7,
            "the vacant count moved"
        );

        // A later write at a neighbouring coordinate must not consume the zero coordinate.
        assert_eq!(table.insert_with(Sigma::Elliptic, 1, 7).unwrap(), 7);
        assert_eq!(table.insert_with(Sigma::Parabolic, 0, 4).unwrap(), 4);
        assert_eq!(
            table.get(Sigma::Elliptic, 0).unwrap(),
            8,
            "a later write lost the live zero coordinate"
        );
        assert_eq!(table.get(Sigma::Parabolic, 0).unwrap(), 4);
        assert_eq!(table.get(Sigma::Elliptic, 1).unwrap(), 7);
        assert_eq!(table.live(), 3);
    }

    #[test]
    fn sigma_separation_update_leaves_the_other_face_unchanged() {
        let mut table = table_of(8);
        let coord = 1u64 << 31;
        for sigma in FACES {
            assert_eq!(table.insert_with(sigma, coord, 1).unwrap(), 1);
        }
        assert_eq!(
            table.live(),
            3,
            "three faces at one coordinate did not occupy three slots: the stored word carries no \
             face half"
        );

        assert_eq!(table.insert_with(Sigma::Elliptic, coord, 10).unwrap(), 11);
        assert_eq!(table.get(Sigma::Elliptic, coord).unwrap(), 11);
        assert_eq!(
            table.get(Sigma::Parabolic, coord).unwrap(),
            1,
            "the Elliptic update reached the Parabolic face"
        );
        assert_eq!(
            table.get(Sigma::Hyperbolic, coord).unwrap(),
            1,
            "the Elliptic update reached the Hyperbolic face"
        );

        // The three stored words are pairwise distinct at the same coordinate.
        let mut words: Vec<u64> = table
            .slots
            .iter()
            .filter(|slot| slot.is_live())
            .map(|slot| slot.word)
            .collect();
        words.sort_unstable();
        words.dedup();
        assert_eq!(
            words.len(),
            3,
            "two faces at one coordinate share a stored word"
        );
        for sigma in FACES {
            assert!(
                words.contains(&U32CoordWord::encode(sigma, coord as u32)),
                "no live slot holds the word of {sigma:?} at 2^31"
            );
        }
    }

    #[test]
    fn full_key_oracle_agrees_with_the_counting_container() {
        let mut table = table_of(MIN_REGION_SLOTS);
        let mut oracle = Oracle::new();

        // Two rounds, so the new-first merge is checked against the oracle's own accumulation.
        for round in 1..=2i64 {
            for sigma in FACES {
                for coord in SENTINEL_COORDS {
                    table.insert_with(sigma, u64::from(coord), round).unwrap();
                    oracle_merge(&mut oracle, sigma, coord, round);
                }
            }
        }

        assert_eq!(
            table.live(),
            oracle.len(),
            "live coordinates and oracle keys disagree"
        );
        for ((face, coord), expected) in &oracle {
            assert_eq!(
                table.get(sigma_of(*face), u64::from(*coord)).unwrap(),
                *expected,
                "the container disagrees with the full-key oracle at face {face} coordinate {coord}"
            );
        }

        // An absent coordinate answers the default on both sides.
        for coord in [123_456_789u64, u64::from(u32::MAX)] {
            for sigma in FACES {
                assert_eq!(
                    table.get(sigma, coord).unwrap(),
                    oracle_expect(&oracle, sigma, coord as u32),
                    "default at ({sigma:?}, {coord}) disagrees with the oracle"
                );
            }
        }
        // The first refused coordinate is refused on the container side and has no oracle key at
        // all: it is not an absent coordinate, it is outside the domain.
        assert!(!oracle
            .keys()
            .any(|(_, coord)| u64::from(*coord) >= FIRST_REFUSED));
    }

    #[test]
    fn first_refused_coordinate_is_not_truncated_or_wrapped() {
        assert_eq!(
            admit_coord(u64::from(u32::MAX)),
            Ok(u32::MAX),
            "2^32 - 1 is admitted"
        );
        assert_eq!(admit_coord(1 << 31), Ok(1 << 31), "2^31 is admitted");
        assert_eq!(admit_coord(0), Ok(0), "0 is admitted");
        assert_eq!(
            admit_coord(FIRST_REFUSED),
            Err(CountRefusal::SizeOverflow),
            "2^32 is the first refused coordinate and must not truncate to 0"
        );
        assert_eq!(
            admit_coord(u64::MAX),
            Err(CountRefusal::SizeOverflow),
            "u64::MAX is outside the admitted domain"
        );

        let mut table = table_of(8);
        assert_eq!(
            table.insert_with(Sigma::Parabolic, FIRST_REFUSED, 9),
            Err(CountRefusal::SizeOverflow),
            "insert_with admitted 2^32"
        );
        assert_eq!(table.live(), 0, "the refused coordinate wrote a slot");
        assert_eq!(table.get(Sigma::Parabolic, 0).unwrap(), COUNT_DEFAULT);
        assert_eq!(
            table.get(Sigma::Parabolic, u64::from(u32::MAX)).unwrap(),
            COUNT_DEFAULT
        );
        assert_eq!(
            table.get(Sigma::Parabolic, FIRST_REFUSED),
            Err(CountRefusal::SizeOverflow),
            "get truncated the refused coordinate where insert_with refused it"
        );
    }

    #[test]
    fn count_overflow_is_refused_without_committing() {
        let mut table = table_of(8);
        table.insert_with(Sigma::Hyperbolic, 4, i64::MAX).unwrap();
        assert_eq!(
            table.insert_with(Sigma::Hyperbolic, 4, 1),
            Err(CountRefusal::CountOverflow),
            "the merge wrapped instead of refusing"
        );
        assert_eq!(
            table.get(Sigma::Hyperbolic, 4).unwrap(),
            i64::MAX,
            "the refused merge committed a count"
        );
        assert_eq!(table.live(), 1, "the refused merge changed the live count");

        // The refusal is not sticky: a merge that fits still commits.
        assert_eq!(
            table.insert_with(Sigma::Hyperbolic, 4, -1).unwrap(),
            i64::MAX - 1
        );
    }

    #[test]
    fn region_growth_preserves_every_committed_count() {
        let mut table = table_of(MIN_REGION_SLOTS);
        let mut oracle = Oracle::new();
        let population = 300usize;
        for round in 1..=population {
            let coord = (round * 2_654_435_761) % 4096;
            let sigma = FACES[round % FACES.len()];
            let value = i64::try_from(round % 7).unwrap() + 1;
            table
                .insert_with(sigma, u64::from(coord as u32), value)
                .unwrap();
            oracle_merge(&mut oracle, sigma, coord as u32, value);
            assert_eq!(
                table.live(),
                oracle.len(),
                "live count diverged at round {round}"
            );
        }
        assert!(
            table.slots.len() > MIN_REGION_SLOTS,
            "the region never grew, so relocation was never exercised"
        );
        for ((face, coord), expected) in &oracle {
            assert_eq!(
                table.get(sigma_of(*face), u64::from(*coord)).unwrap(),
                *expected,
                "relocation lost or merged a count at face {face} coordinate {coord}"
            );
        }
    }

    #[test]
    fn construction_refuses_an_unchecked_width() {
        assert_eq!(
            CountingTable::<U32CoordWord>::new(MIN_REGION_SLOTS - 1).err(),
            Some(CountRefusal::SizeOverflow),
            "a region narrower than MIN_REGION_SLOTS was accepted"
        );
        assert_eq!(
            CountingTable::<U32CoordWord>::new(MAX_REGION_SLOTS + 1).err(),
            Some(CountRefusal::SizeOverflow),
            "a region wider than MAX_REGION_SLOTS was accepted"
        );
        assert_eq!(
            CountingTable::<U32CoordWord>::new(0).err(),
            Some(CountRefusal::SizeOverflow)
        );
    }

    #[test]
    fn probe_is_bounded_by_the_region_width() {
        let table = table_of(MIN_REGION_SLOTS);
        // An empty region answers Vacant at the start index, never a wrap: one comparison.
        assert_eq!(
            probe_in(&table.slots, 7),
            Probe::Vacant(probe_start(7, MIN_REGION_SLOTS))
        );
        // A region with every slot occupied by a distinct word answers Absent after one full pass.
        let mut full = vec![CountSlot::VACANT; MIN_REGION_SLOTS];
        for (index, slot) in full.iter_mut().enumerate() {
            *slot = CountSlot {
                tag: TAG_LIVE,
                word: u64::try_from(index).unwrap() + 1,
                count: 1,
            };
        }
        assert_eq!(probe_in(&full, 999), Probe::Absent);
        assert_eq!(probe_in(&full, 1), Probe::Matched(0));
        // A width below MIN_REGION_SLOTS is refused rather than divided by.
        assert_eq!(probe_in(&[], 7), Probe::Absent);
    }

    #[test]
    fn sentinels_table_is_the_executed_one() {
        // The frozen table is executable, not prose: each row's admitted flag is the result of
        // admit_coord, and each admitted row round-trips.
        for (sigma, raw, admitted) in super::SENTINELS {
            match admit_coord(raw) {
                Ok(coord) => {
                    assert!(
                        admitted,
                        "{sigma:?} {raw} is refused but the table admits it"
                    );
                    assert_eq!(
                        U32CoordWord::decode(sigma, U32CoordWord::encode(sigma, coord)),
                        coord
                    );
                    assert_eq!(
                        raw & COORD_MASK,
                        raw,
                        "an admitted sentinel sits above the domain"
                    );
                }
                Err(error) => {
                    assert!(
                        !admitted,
                        "{sigma:?} {raw} is refused as {error:?} but the table admits it"
                    );
                    assert_eq!(
                        raw, FIRST_REFUSED,
                        "only 2^32 is refused in the frozen table"
                    );
                }
            }
        }
    }
}
