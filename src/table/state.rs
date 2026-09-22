// Region, summary, decision, scalar-geometry and rebuild-plan state for the growable table.
// Included textually by `include!("table/state.rs")` from `src/table.rs`, so these items stay
// definitions of the `table` module and keep their generated type identities. Bodies, layout,
// field privacy, signatures and visibility are unchanged; this is a physical-file cut only.

/// Bits of the per-region occupancy summary reserved per slot.
const SUMMARY_BITS_PER_SLOT: usize = 32;
/// Minimum summary width, so a small region still filters.
const SUMMARY_MIN_BITS: usize = 256;
/// Assumed upper bound, in bytes, of one flat slot, used only for the byte-budget heuristic.
///
/// The executing slots are fixed-width words; `core::mem::size_of` is an unmodeled Aeneas external,
/// so the budget uses this bound. A larger element type is capped to correspondingly fewer lanes.
const ASSUMED_SLOT_BYTES: usize = 32;

/// The number of trailing zero bits of `value`, or `usize::BITS` at zero.
///
/// A plain shift-and-count loop rather than the `usize::trailing_zeros` intrinsic, which the
/// extraction leaves unmodeled. It is total, agrees with `usize::trailing_zeros` on every `usize`
/// including zero, and executes at most `usize::BITS` iterations: temporal `O(usize::BITS)` counted
/// in loop steps at a fixed word width, spatial `O(1)`.
fn summary_trailing_zeros(mut value: usize) -> u32 {
    let mut zeros = 0;
    while zeros < usize::BITS && value & 1 == 0 {
        zeros += 1;
        value >>= 1;
    }
    zeros
}

/// The number of `u64` words in a region summary sized for `slots` reserved lanes, or
/// [`StorageError::SizeOverflow`] when the checked bit, round-up or power-of-two arithmetic
/// overflows, or when the resulting width would not fit the shift.
///
/// The summary carries [`SUMMARY_BITS_PER_SLOT`] Bloom bits per reserved slot, rounded up to a
/// power-of-two word count. It is at least [`SUMMARY_MIN_BITS`] wide so a single group still
/// filters. An empty region has no summary storage. The bit count is `trailingZeros(wordCount) + 6`
/// and must stay below `u64::BITS`, so the computed word count is bounded below `2^58`.
fn summary_words(slots: usize) -> Result<usize, StorageError> {
    if slots == 0 {
        return Ok(0);
    }
    let bits = slots
        .checked_mul(SUMMARY_BITS_PER_SLOT)
        .ok_or(StorageError::SizeOverflow)?
        .max(SUMMARY_MIN_BITS);
    // Explicit ceiling division and next-power-of-two: `div_ceil`/`checked_next_power_of_two` are
    // opaque to the extractor, and this keeps the two operations as scalar arithmetic.
    let mut words = bits / 64;
    if bits % 64 != 0 {
        words += 1;
    }
    if words == 0 {
        words = 1;
    }
    let mut power = 1usize;
    let mut overflow = false;
    while power < words && !overflow {
        match power.checked_mul(2) {
            Some(next) => power = next,
            None => overflow = true,
        }
    }
    if overflow {
        return Err(StorageError::SizeOverflow);
    }
    let words = power;
    if summary_trailing_zeros(words) + 6 >= u64::BITS {
        return Err(StorageError::SizeOverflow);
    }
    Ok(words)
}

/// Test-only record of the spill publication steps, so the contract order can be checked on the
/// real write path. Inert outside `cfg(test)`; it is not a production or reference path.
#[cfg(test)]
mod publication_trace {
    use std::cell::RefCell;

    /// The payload write of one fresh live spill slot.
    pub(super) const PAYLOAD: u8 = 0;
    /// The summary-bit recording of one fresh live spill slot.
    pub(super) const SUMMARY: u8 = 1;
    /// The nonzero tag publication of one fresh live spill slot.
    pub(super) const TAG: u8 = 2;

    thread_local! {
        static EVENTS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    /// Clears the recorded steps on this thread.
    pub(super) fn clear() {
        EVENTS.with(|events| events.borrow_mut().clear());
    }

    /// Records one publication step on this thread.
    pub(super) fn record(step: u8) {
        EVENTS.with(|events| events.borrow_mut().push(step));
    }

    /// Returns the steps recorded on this thread since the last [`clear`].
    pub(super) fn events() -> Vec<u8> {
        EVENTS.with(|events| events.borrow().clone())
    }
}

pub(crate) struct Region<R, V> {
    tags: Vec<u8>,
    slots: Vec<(Sigma, R, V)>,
    /// Bloom bits over `place >> 8` of the live slots.
    ///
    /// A live slot's placement word is recorded before its tag becomes nonzero, so a query whose
    /// placement word is absent from the summary cannot match any live slot. A collision is a false
    /// positive that costs the full sweep; it is never a false absence. Fresh writes publish the
    /// payload, then the summary bit, then the nonzero tag, so no search-visible state has a live
    /// tag without its recorded bit.
    summary: Vec<u64>,
}
impl<R, V> Region<R, V> {
    fn empty() -> Self {
        Self {
            tags: vec![],
            slots: vec![],
            summary: vec![],
        }
    }

    /// The summary bit index of `place` for a summary of `len_words` `u64` words, or `None` when
    /// no bit index exists.
    ///
    /// The representation is `Option<usize>`: `None` means "no usable summary word", so a caller
    /// must not index. The bit count is `trailingZeros(len_words) + 6`. A zero length yields
    /// `64 + 6`; a length whose trailing-zero count reaches `58` yields `>= 64`. Both are rejected
    /// here rather than shifted, so no call shifts a `u64` by `>= 64` and every returned index is in
    /// bounds: for any accepted length, `2^bits <= 64 * len_words`, hence `index / 64 < len_words`.
    /// A valid region's length is a power of two in `4..=2^57`, so this is `Some` and byte-identical
    /// to the pre-totalization index.
    ///
    /// A valid summary length is a power of two, so the general trailing-zero loop is replaced on
    /// that hot case by the exact mask `(len_words << 6) - 1`: for `len_words = 2^t` the bit count
    /// is `t + 6`, hence `2^bits == 64 * len_words` and the two masks are equal. The guard keeps
    /// the shift inside `usize` and the implied bit count below `u64::BITS`; every other length,
    /// including the out-of-range power of two, falls through to the trailing-zero loop unchanged.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the low-`bits` mask bounds the mixed value below 2^bits, and bits fits usize"
    )]
    #[inline]
    fn summary_index(place: u64, len_words: usize) -> Option<usize> {
        if len_words == 0 {
            return None;
        }
        let mask = if len_words.is_power_of_two() && len_words <= (usize::MAX >> 6) {
            ((len_words as u64) << 6) - 1
        } else {
            let bits = summary_trailing_zeros(len_words) + 6;
            if bits >= u64::BITS {
                return None;
            }
            (1u64 << bits) - 1
        };
        let mixed = (place >> 8).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mixed = mixed ^ (mixed >> 29);
        Some((mixed & mask) as usize)
    }

    /// Records `place` in the summary; a no-op when the region has no usable summary word.
    #[inline]
    fn summary_set(&mut self, place: u64) {
        let Some(index) = Self::summary_index(place, self.summary.len()) else {
            return;
        };
        self.summary[index / 64] |= 1u64 << (index % 64);
    }

    /// Reports whether `place` may be present.
    ///
    /// A summary with no usable word (`len_words == 0`, or a length whose bit count is `>= 64`)
    /// cannot certify absence, so filtering is disabled and the result is `true`; the caller's sweep
    /// then runs and can still find a match. This preserves the `SpillSummaryShape` empty-summary
    /// case and `SpillSummaryMissSound`, and is exactly the pre-totalization empty guard.
    #[inline]
    fn summary_maybe(&self, place: u64) -> bool {
        let Some(index) = Self::summary_index(place, self.summary.len()) else {
            return true;
        };
        self.summary[index / 64] & (1u64 << (index % 64)) != 0
    }

    /// Publishes one fresh live spill slot in the contract order: payload, summary bit, then the
    /// nonzero tag.
    ///
    /// The tag write is the occupancy publication point. No fallible operation and no search-visible
    /// handoff runs between the summary recording and the tag publication, so every search-visible
    /// state with a live tag has its summary bit recorded.
    #[inline]
    fn publish_live(&mut self, index: usize, payload: (Sigma, R, V), place: u64) {
        self.slots[index] = payload;
        #[cfg(test)]
        publication_trace::record(publication_trace::PAYLOAD);
        self.summary_set(place);
        #[cfg(test)]
        publication_trace::record(publication_trace::SUMMARY);
        self.tags[index] = tag(place);
        #[cfg(test)]
        publication_trace::record(publication_trace::TAG);
    }
}
impl<R: Clone, V: Clone> Region<R, V> {
    fn reserve_empty(n: usize) -> Result<Self, StorageError> {
        // Compute the complete checked layout before any reservation: a sizing overflow refuses
        // before a tag, payload or summary allocation is attempted.
        let words = summary_words(n)?;
        let mut tags = Vec::new();
        let mut slots = Vec::new();
        let mut summary = Vec::new();
        try_reserve_exact(&mut tags, n)?;
        try_reserve_exact(&mut slots, n)?;
        try_reserve_exact(&mut summary, words)?;
        tags.resize(n, 0);
        summary.resize(words, 0);
        Ok(Self {
            tags,
            slots,
            summary,
        })
    }

    fn initialize(&mut self, sigma: Sigma, held: &R, default: &V) {
        while self.slots.len() < self.tags.len() {
            self.slots.push((sigma, held.clone(), default.clone()));
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpillSite {
    Found(usize),
    Vacant(usize),
    Grow,
}
/// Classification returned by the unextracted composite probe walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchClass {
    /// The selected location holds the queried full logical key.
    Found,
    /// The selected location is the first vacant lane in probe order.
    Vacant,
    /// No indexed location is available; prepared spill growth is required.
    Exhausted,
}

/// Region and index selected by a [`SearchDecision`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchLocation {
    /// A shard-major resident-slot index.
    Flat(usize),
    /// A spill shard and its slot index.
    Spill(usize, usize),
    /// No index accompanies exhaustion.
    NoIndex,
}

/// A value-only result of the composite direct, window, and coverage-sweep probe walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchDecision {
    /// The probe classification.
    pub class: SearchClass,
    /// The location paired with [`Self::class`].
    pub location: SearchLocation,
}

impl SearchDecision {
    fn found_flat(index: usize) -> Self {
        Self {
            class: SearchClass::Found,
            location: SearchLocation::Flat(index),
        }
    }
    fn vacant_flat(index: usize) -> Self {
        Self {
            class: SearchClass::Vacant,
            location: SearchLocation::Flat(index),
        }
    }
    fn from_spill(shard: usize, site: SpillSite) -> Self {
        match site {
            SpillSite::Found(index) => Self {
                class: SearchClass::Found,
                location: SearchLocation::Spill(shard, index),
            },
            SpillSite::Vacant(index) => Self {
                class: SearchClass::Vacant,
                location: SearchLocation::Spill(shard, index),
            },
            SpillSite::Grow => Self {
                class: SearchClass::Exhausted,
                location: SearchLocation::NoIndex,
            },
        }
    }
}
/// Straight-line result of [`Table::apply_insert`].
///
/// `Committed` records an update or initialization of the indexed real slot.  `NeedsGrowth` returns
/// the pending entry unchanged so the caller can perform the resource step; neither variant
/// allocates, reserves or runs a probe loop.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ApplyOutcome<K, V> {
    /// The indexed real slot was updated or initialized.
    Committed,
    /// No indexed location exists; the caller must prepare growth for the returned pending entry.
    NeedsGrowth(K, V),
}

#[inline]
fn valid_n(n: usize) -> bool {
    n != 0 && n <= MAX_GROUP && n.is_power_of_two()
}
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "only the low shard byte is retained"
)]
pub(crate) fn shard(address: &TableAddress) -> usize {
    match address.direct_word {
        Some(word) => (word as usize) & SHARD_MASK,
        None => (address.place as usize) & SHARD_MASK,
    }
}
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the fingerprint mask retains only the low seven bits, so the u8 cast drops no \
              retained bit"
)]
fn tag(place: u64) -> u8 {
    TAG_LIVE | (((place >> TAG_FINGERPRINT_SHIFT) as u8) & TAG_FINGERPRINT_MASK)
}
#[inline]
fn matches<K, H, R: KeyRep<K, H>, V>(
    slot: &(Sigma, R, V),
    sigma: Sigma,
    key: &K,
    address: &TableAddress,
) -> bool {
    slot.0 == sigma && R::matches(sigma, &slot.1, key, address)
}

fn slots_for(load: usize, n: usize) -> Result<usize, StorageError> {
    if load == 0 {
        return Ok(0);
    }
    let twice = load.checked_mul(2).ok_or(StorageError::SizeOverflow)?;
    let groups_needed = twice.checked_add(n - 1).ok_or(StorageError::SizeOverflow)? / n;
    let mut groups = 2usize;
    while groups < groups_needed {
        groups = groups.checked_mul(2).ok_or(StorageError::SizeOverflow)?;
    }
    groups.checked_mul(n).ok_or(StorageError::SizeOverflow)
}

/// Builds a [`CapacityBound`] from a checked target and the capacity the reservation observed.
///
/// The production scalar comparison runs here.  A successful reservation always meets its checked
/// target, so a violated bound is an internal invariant failure, not a new refusal case or a late
/// `Err` after cloning.
fn capacity_bound(required_len: usize, observed_capacity: usize) -> CapacityBound {
    assert!(
        capacity_bound_holds(required_len, observed_capacity),
        "reserved capacity {observed_capacity} is below the checked target {required_len}"
    );
    CapacityBound {
        required_len,
        observed_capacity,
    }
}
#[inline]
fn group(place: u64, step: usize, mask: usize) -> usize {
    let d = (place >> 8) & DESCENT_MASK;
    if step < WINDOWS {
        (((d >> (49 - 7 * (step + 1))) & WINDOW_MASK) as usize) & mask
    } else {
        ((d as usize) & mask).wrapping_add(step - WINDOWS + 1) & mask
    }
}

/// Returns the least significant set-bit index for a nonzero word.
#[inline]
fn least_set_lane(bits: u64) -> usize {
    let mut lane = 0;
    while lane < u64::BITS as usize && ((bits >> lane) & 1) == 0 {
        lane += 1;
    }
    lane
}

/// Marks a certified direct word in the density bitmap and increments `count` on its first sight.
///
/// The word is ignored at or above `limit`; only words that could select a resident slot below
/// `limit` participate in the density count. Temporal O(1); spatial O(1).
#[allow(
    clippy::cast_possible_truncation,
    reason = "the `word < limit` guard bounds the index by a lane count that fits usize"
)]
fn mark_direct_word(seen: &mut [u64], count: &mut usize, word: u32, limit: u64) {
    if u64::from(word) >= limit {
        return;
    }
    let bit = word as usize;
    let slot = &mut seen[bit / 64];
    let mask = 1u64 << (bit % 64);
    if *slot & mask == 0 {
        *slot |= mask;
        *count += 1;
    }
}
/// A total-default table with flat certified resident cells and grouped spill regions.
struct FlatSlot<R, V> {
    live: bool,
    sigma: Sigma,
    held: R,
    value: V,
}

struct Flat<R, V> {
    lanes: usize,
    slots: Vec<FlatSlot<R, V>>,
}

#[derive(Clone, Copy)]
enum RebuildSource {
    Resident(usize),
    Spill(usize, usize),
}

struct RebuildPlanEntry {
    source: RebuildSource,
    flat: Option<usize>,
    spill: usize,
}

pub(crate) struct RebuildPlan {
    lanes: usize,
    entries: Vec<RebuildPlanEntry>,
    spill_loads: Vec<usize>,
}

pub(crate) struct RebuildReserved<R, V> {
    resident: Flat<R, V>,
    regions: Vec<Region<R, V>>,
}

impl<R, V> Flat<R, V> {
    fn empty() -> Self {
        Self {
            lanes: 0,
            slots: Vec::new(),
        }
    }
    #[inline]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the lane is below self.lanes as u64, so it fits usize; the shard is at most \
                  SHARD_MASK"
    )]
    fn index(&self, word: u32) -> Option<usize> {
        let lane_word = u64::from(word >> SHARD_BITS);
        if lane_word >= self.lanes as u64 {
            return None;
        }
        let lane = lane_word as usize;
        let shard = (word as usize) & SHARD_MASK;
        match shard.checked_mul(self.lanes) {
            Some(base) => base.checked_add(lane),
            None => None,
        }
    }
    fn byte_len_for(lanes: usize) -> Option<usize> {
        lanes.checked_mul(SHARDS)?.checked_mul(ASSUMED_SLOT_BYTES)
    }
}

impl<R: Clone, V: Clone> Flat<R, V> {
    fn reserve(lanes: usize) -> Result<Self, StorageError> {
        let count = lanes.checked_mul(SHARDS).ok_or(StorageError::SizeOverflow)?;
        let mut slots = Vec::new();
        try_reserve_exact(&mut slots, count)?;
        Ok(Self { lanes, slots })
    }

    fn initialize(&mut self, sigma: Sigma, held: &R, value: &V) {
        let count = self.lanes * SHARDS;
        while self.slots.len() < count {
            self.slots.push(FlatSlot {
                live: false,
                sigma,
                held: held.clone(),
                value: value.clone(),
            });
        }
    }
}

/// Where one linked representative's payload lives in the table's own storage.
///
/// This names a location; it is not a second store. The key and value remain in the named flat or
/// spill slot, so the directory still owns no key and no value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PayloadLocation {
    /// A live resident flat slot.
    Resident(usize),
    /// A live slot `(shard, index)` in a spill region.
    Spill(usize, usize),
}

/// The handle-only directory state attached to [`Table`].
///
/// The shared [`Directory`] owns no key and no value: terminal handle `i` names the payload at
/// `links[i].1`, and `links[i].0` is the terminal identity node that handle must occupy. The state
/// therefore records reverse links, not a duplicate key/value map. `staged` is the private staged
/// destination of an in-flight write; while it is `Some`, no coverage certificate may be issued.
pub(crate) struct TableLocator {
    /// The sealed hyperbolic profile used to rebuild a hyperbolic payload's residue path.
    profile: Option<H3_40>,
    /// The shared directory engine, or `None` for the allocation-free empty state.
    directory: Option<Directory>,
    /// Reverse links `(identity_node, payload_location)` indexed by terminal handle.
    links: Vec<(u32, PayloadLocation)>,
    /// The staged destination of a pending write, if any.
    staged: Option<(u32, StagedPath)>,
}

impl TableLocator {
    /// The empty state: no directory, no links, no staging and no allocation.
    pub(crate) const fn new() -> Self {
        Self {
            profile: None,
            directory: None,
            links: Vec::new(),
            staged: None,
        }
    }
}
