// Region, decision, scalar-geometry and rebuild-plan state for the growable table.
// Included textually by `include!("table/state.rs")` from `src/table.rs`, so these items stay
// definitions of the `table` module and keep their generated type identities. Bodies, layout,
// field privacy, signatures and visibility are unchanged; this is a physical-file cut only.

/// Assumed upper bound, in bytes, of one flat slot, used only for the byte-budget heuristic.
///
/// The executing slots are fixed-width words; `core::mem::size_of` is an unmodeled Aeneas external,
/// so the budget uses this bound. A larger element type is capped to correspondingly fewer lanes.
const ASSUMED_SLOT_BYTES: usize = 32;

/// The vacant-slot marker of a region's coordinate array.
///
/// Ported from `fold hashtrinity/src/lib.rs:298` (`const EMPTY: u64 = u64::MAX`). A live coordinate
/// never equals it: [`Region::coord_of`] nudges the one colliding value aside, so
/// `coord == COORD_EMPTY` unambiguously means "vacant" and nothing else. This is the word that
/// makes an absent read `O(1)`: one load, one compare, `ret` (`full.asm` `probe_get`,
/// `cmp $0xffffffffffffffff`).
const COORD_EMPTY: u64 = u64::MAX;

/// Test-only record of the spill publication steps, so the contract order can be checked on the
/// real write path. Inert outside `cfg(test)`; it is not a production or reference path.
#[cfg(test)]
mod publication_trace {
    use std::cell::RefCell;

    /// The payload write of one fresh live spill slot.
    pub(super) const PAYLOAD: u8 = 0;
    /// The nonzero tag publication of one fresh live spill slot.
    pub(super) const TAG: u8 = 1;

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

/// One spill slot: the occupancy marker, the face, the held carrier and the payload.
///
/// `tag` is the occupancy marker and fingerprint in one `u8`: `0` is a vacancy, and any nonzero
/// value is the live marker [`TAG_LIVE`] OR the query fingerprint of the stored key's place. The
/// tag is a fingerprint only — a matching tag still requires the full `(Sigma, K)` compare — and it
/// is derived, never stored twice.
pub(crate) struct SpillSlot<R, V> {
    /// `0` while vacant; the nonzero occupancy marker after publication.
    pub(crate) tag: u8,
    /// The face this slot's payload belongs to.
    pub(crate) sigma: Sigma,
    /// The held carrier; the key material the full compare runs against.
    pub(crate) held: R,
    /// The merged value.
    pub(crate) value: V,
}

impl<R, V> SpillSlot<R, V> {
    /// Reports whether this slot is occupied: the nonzero-tag test.
    #[inline]
    pub(crate) fn live(&self) -> bool {
        self.tag != 0
    }
}

/// One open-addressed spill region per shard: a single flat slot array with linear probing.
///
/// **Invariants.** `slots.len()` is the capacity and is either `0` or a power of two, so the probe
/// mask is `slots.len() - 1`; `live` is the exact number of occupied slots, so
/// `live <= slots.len()` always. A slot is occupied exactly when its `tag` is nonzero.
///
/// **Probe sequence.** Slot `i` of a query with placement `place` is visited at
/// `((home(place) + step) & mask)` for `step = 0, 1, ..., capacity - 1` — linear probing over one
/// cycle, in increasing index with wraparound. The **bound is the capacity**: after `capacity`
/// steps every slot has been visited exactly once, so a cycle that finds no vacancy has observed
/// every slot occupied. That is the absence certificate the read path relies on, and it is what
/// `SpillProbeBoundAbsentSound` states: no key can be stored in a slot of a completed cycle without
/// that cycle having visited it.
///
/// **No deletion.** Nothing in the table removes a spill entry; a repeated key merges in place.
/// There are therefore no tombstones, and a vacant coordinate terminates a probe correctly: no live
/// slot can sit beyond a vacancy, because a linear insert only ever places at the first vacancy at
/// or after its home. (A future deletion would have to introduce a tombstone and would void this
/// argument.)
///
/// **Publication order.** A fresh write stores `sigma`, `held` and `value` first, then writes the
/// nonzero `tag`, then writes the coordinate word. The coordinate write is the occupancy
/// publication point, so no search-visible state has a live coordinate over an unpublished payload.
///
/// **The resident kernel is the incumbent's.** `coords` is the contiguous coordinate array of
/// `fold hashtrinity/src/lib.rs:409` (`coords: Box<[u64]>`): one `u64` per slot holding the placed
/// key's own coordinate word, or [`COORD_EMPTY`] while vacant. One word carries occupancy, identity
/// and the absence certificate at once, so an absent read is one load and one compare
/// (`full.asm` `probe_get`: `mov (%rcx,%rax,8),%rcx` then `cmp $0xffffffffffffffff`). The per-slot
/// `tag` stays as the published occupancy marker the relocation and coverage passes read; it is not
/// on the probe path.
pub(crate) struct Region<R, V> {
    /// The open-addressed slot array; length is the capacity, a power of two or zero.
    slots: Vec<SpillSlot<R, V>>,
    /// The contiguous coordinate array: one placed coordinate word per slot index, or
    /// [`COORD_EMPTY`] while vacant. Its length equals [`Self::slots`].
    coords: Vec<u64>,
    /// The incumbent's `bits` for this capacity: `log2(slots.len())`, or `0` while there is no
    /// storage. Stored rather than recomputed, because `usize::trailing_zeros` has no field in the
    /// pinned Aeneas Lean model and the fail-closed audit rejects the external (see
    /// `density_below`'s note on the same rule).
    bits: u32,
    /// The number of occupied slots.
    live: usize,
}
impl<R, V> Region<R, V> {
    fn empty() -> Self {
        Self {
            slots: Vec::new(),
            coords: Vec::new(),
            bits: 0,
            live: 0,
        }
    }

    /// The probe bound of this region: its capacity, or `0` when it has no storage.
    ///
    /// This is the number of distinct slots one bounded linear-probe cycle visits, and therefore the
    /// absence certificate's width. `O(1)`; spatial `O(1)`.
    #[inline]
    fn bound(&self) -> usize {
        self.slots.len()
    }

    /// A placement's coordinate word: `place` with the low shard byte removed, and the reserved
    /// vacant marker nudged aside.
    ///
    /// The low [`SHARD_BITS`] bits of `place` select the region (see [`shard`]), so they are
    /// constant across one region's whole population and carry no information inside it; the
    /// coordinate word is what remains. The nudge is `Table::coord_of` of
    /// `fold hashtrinity/src/lib.rs:548-561`, which maps the one colliding value aside so a live
    /// coordinate is never [`COORD_EMPTY`]. That nudge aliases the top of the range, the same
    /// 2⁻⁶⁴ identity risk the coordinate already trusts and unreachable for the coordinate words
    /// this table homes.
    #[inline]
    fn coord_of(place: u64) -> u64 {
        let c = place >> SHARD_BITS;
        if c == COORD_EMPTY {
            COORD_EMPTY - 1
        } else {
            c
        }
    }

    /// The home slot of `place`: the incumbent's home, `fold hashtrinity/src/lib.rs:565-572`.
    ///
    /// A coordinate below the region capacity indexes directly, with no hash; a larger one takes the
    /// top `bits` of `coord · PHI`, the Fibonacci multiply-shift whose multiply is on the
    /// fall-through branch in `full.asm` `probe_get` (`1822b`/`18235`) while the direct index is the
    /// taken branch (`18225`/`18229`). `PHI` is odd, so the multiply-shift is a bijection on the
    /// word and loses no key. `O(1)`; spatial `O(1)`.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the direct branch proves the value is below capacity and the shift leaves at most                   `bits` bits, so both fit usize"
    )]
    #[inline]
    fn home(&self, place: u64) -> usize {
        let c = Self::coord_of(place);
        let len = self.coords.len();
        // A region with no storage never reaches a probe; the guard keeps the shift below 64.
        if len == 0 {
            return 0;
        }
        if c < len as u64 {
            c as usize
        } else {
            (c.wrapping_mul(TAG_SPREAD) >> (64 - self.bits)) as usize
        }
    }

    /// Publishes one fresh live spill slot in the contract order: payload, then the nonzero tag,
    /// then the coordinate word.
    ///
    /// The coordinate write is the occupancy publication point; no fallible operation and no
    /// search-visible handoff runs between the payload store and the coordinate store, so every
    /// search-visible state with a live coordinate has its published payload.
    #[inline]
    fn publish_live(&mut self, index: usize, payload: (Sigma, R, V), place: u64) {
        self.slots[index].sigma = payload.0;
        self.slots[index].held = payload.1;
        self.slots[index].value = payload.2;
        #[cfg(test)]
        publication_trace::record(publication_trace::PAYLOAD);
        self.slots[index].tag = tag(place);
        self.coords[index] = Self::coord_of(place);
        self.live += 1;
        #[cfg(test)]
        publication_trace::record(publication_trace::TAG);
    }
}
impl<R, V> Region<R, V> {
    /// The occupancy marker of every slot, for a before/after no-mutation assertion.
    ///
    /// Test-only: the read path never materializes this vector.
    #[cfg(test)]
    fn markers(&self) -> Vec<u8> {
        self.slots.iter().map(|slot| slot.tag).collect()
    }
}
impl<R: Clone, V: Clone> Region<R, V> {
    /// Reserves `capacity` initialized-length slots of storage without writing any slot.
    ///
    /// The capacity is checked before any reservation, so a sizing overflow refuses without a
    /// partial allocation. Both arrays are checked, so a refused second reservation commits neither.
    /// `O(capacity)`; spatial `O(capacity)` slots and `O(capacity)` coordinate words.
    fn reserve_empty(capacity: usize) -> Result<Self, StorageError> {
        let mut slots = Vec::new();
        try_reserve_exact(&mut slots, capacity)?;
        let mut coords = Vec::new();
        try_reserve_exact(&mut coords, capacity)?;
        Ok(Self {
            slots,
            coords,
            bits: 0,
            live: 0,
        })
    }

    /// Writes the vacant marker into every slot, filling both arrays to their reserved capacity.
    ///
    /// Temporal `O(capacity)` clones; spatial `O(capacity)`.
    fn initialize(&mut self, sigma: Sigma, held: &R, default: &V) {
        let capacity = self.slots.capacity();
        while self.slots.len() < capacity {
            self.slots.push(SpillSlot {
                tag: 0,
                sigma,
                held: held.clone(),
                value: default.clone(),
            });
        }
        while self.coords.len() < capacity {
            self.coords.push(COORD_EMPTY);
        }
        // The capacity is a power of two, so this counts the doublings that produce it: `bits` is
        // `log2(capacity)`, the shift width the multiply-shift home needs.
        let mut width = capacity;
        let mut bits = 0u32;
        while width > 1 {
            width /= 2;
            bits += 1;
        }
        self.bits = bits;
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
    reason = "the fingerprint shift leaves the top seven bits, so the u8 cast drops no retained \
              bit"
)]
fn tag(place: u64) -> u8 {
    let spread = place.wrapping_mul(TAG_SPREAD);
    TAG_LIVE | (((spread >> TAG_FINGERPRINT_SHIFT) as u8) & TAG_FINGERPRINT_MASK)
}
#[inline]
fn matches<K, H, R: KeyRep<K, H>, V>(
    slot: &SpillSlot<R, V>,
    sigma: Sigma,
    key: &K,
    address: &TableAddress,
) -> bool {
    slot.sigma == sigma && R::matches(sigma, &slot.held, key, address)
}

/// The capacity a region holding `load` live entries reserves, or `Ok(0)` for an empty region.
///
/// One flat open-addressed region needs a power-of-two capacity so the probe mask is
/// `capacity - 1`, and it needs at least one vacant slot, so the capacity is the least power of two
/// `>= 2 * load` and at least `2 * n`.  Every reserved region therefore satisfies
/// `capacity >= 2 * load`, and a region the insert path builds itself is reserved for the whole
/// group width, so it holds a vacancy at every load up to `n`: the load factor stays at or below
/// `1/2`, which is the classic linear-probing bound, under which an absent read stops at the first
/// vacancy after `1 / (1 - alpha)` slots on average instead of completing a whole cycle.
///
/// The floor is `2 * n` rather than `n` because `n` is a load a shard fills, not a capacity.  A
/// floor of `n` reserved `n` slots for the region's first entry, and the insert path then filled
/// those `n` slots: at load factor `1` the bounded read cycle has no vacancy to stop on, so every
/// absent read cost the region's whole capacity in tag compares, and the completed-cycle absence
/// certificate became reachable through [`Table::insert_with`] rather than only through a
/// deliberately filled region.
///
/// Temporal `O(log(load))` doublings; spatial `O(1)` words beyond the reserved region.
fn slots_for(load: usize, n: usize) -> Result<usize, StorageError> {
    if load == 0 {
        return Ok(0);
    }
    let twice = load.checked_mul(2).ok_or(StorageError::SizeOverflow)?;
    // `n.max(2)` keeps the zero-width floor of the original rule; `valid_n` admits `n = 1`, so
    // `2 * n` is a power of two above the smallest group the table accepts.
    let floor = n
        .max(2)
        .checked_mul(2)
        .ok_or(StorageError::SizeOverflow)?;
    let mut capacity = floor;
    while capacity < twice {
        capacity = capacity.checked_mul(2).ok_or(StorageError::SizeOverflow)?;
    }
    Ok(capacity)
}

/// Builds a [`CapacityBound`] from a checked target and the capacity the reservation observed.
///
/// The production scalar comparison runs here.  A successful reservation always meets its checked
/// target, so a violated bound is an internal invariant failure, not a new refusal case or a late
/// `Err` after cloning.
fn capacity_bound(required_len: usize, observed_capacity: usize) -> CapacityBound {
    assert!(
        capacity_bound_holds(required_len, observed_capacity),
        "reserved capacity is below the checked target"
    );
    CapacityBound {
        required_len,
        observed_capacity,
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
        let count = lanes
            .checked_mul(SHARDS)
            .ok_or(StorageError::SizeOverflow)?;
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
    ///
    /// The registry is built by the cold [`Table::assemble_directory`] pass from the table's own
    /// live payloads; a merge does not push a link. It stays empty until that pass runs, so a fresh
    /// insert writes only the payload stores.
    links: Vec<(u32, PayloadLocation)>,
    /// The number of representatives admitted against the reserved link extent.
    ///
    /// This is [`Self::links`]'s admitted count while the registry is not materialized: a merge
    /// increments it into the extent [`TableLocator::reserve_batch_paths`] reserved, and refuses
    /// when the extent is exhausted. [`Table::assemble_directory`] resets it to the rebuilt
    /// registry length.
    admitted: usize,
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
            admitted: 0,
            staged: None,
        }
    }
}
