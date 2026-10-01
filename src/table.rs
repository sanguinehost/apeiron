#![allow(
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    clippy::manual_map,
    clippy::manual_filter,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::len_zero,
    clippy::bool_to_int_with_if,
    clippy::question_mark,
    reason = "extraction-facing scalar forms: the pinned extractor emits an `axiom` for \
              `is_empty`, `div_ceil`, `Option::map`, `let-else`, integer `TryFrom` and the \
              narrowing casts, so the explicit forms are kept and these lints stand"
)]
//! The growable table with a flat certified-coordinate resident area and grouped spill regions.
//!
//! Composite invariant: every logical `(Sigma, K)` has exactly one representative.  When its
//! certified coordinate is admitted by the resident extent, that flat cell is live, even if a
//! colliding key's unique representative is in spill. Removal is deliberately out of scope.

use crate::cell::Sigma;
use crate::coord::address::{
    EllipticPacking, FaceCoord, FaceDomain, FaceHome, FaceResidueCapability, ParabolicDual,
    Residue, WordFaceHome, H3_40,
};
use crate::coord::locator::locator_core::{Directory, ResidueIndex, StagedPath, Walk};
use crate::coord::locator::{ExactIdentity, Refusal};
use core::marker::PhantomData;

// These explicit paths preserve normal crate resolution and also make this file's children resolve
// from `src/table/` when Charon includes `src/table.rs` through `proof/extract/lib.rs`.
#[path = "table/keyrep.rs"]
pub mod keyrep;
#[path = "table/parallel.rs"]
pub mod parallel;
#[path = "table/partition.rs"]
pub(crate) mod partition;
#[path = "table/probe.rs"]
mod probe;
#[path = "table/resources.rs"]
pub(crate) mod resources;
#[path = "table/scan.rs"]
pub(crate) mod scan;

use keyrep::{KeyRep, Stored};
#[cfg(feature = "read-stats")]
pub use probe::read_stats;
#[cfg(test)]
use probe::read_trace;
use probe::{spill_locate, spill_locate_held, spill_read};
use resources::{
    capacity_bound_holds, publish_prepared, try_reserve_exact, CapacityBound,
    LocatorReservationPlan, LocatorReservationReceipt, PreparedEnvelope, PreparedOutcome,
    PreparedResult, RegionReservation, ReservationReceipt,
};

const DEFAULT_BUDGET_BYTES: usize = SHARDS * 1024;
const MAX_LANE_BITS: u32 = 32 - SHARD_BITS;

const SHARDS: usize = partition::SHARDS;
const SHARD_BITS: u32 = partition::SHARD_BITS;
const SHARD_MASK: usize = partition::SHARD_MASK;
const _: () = {
    assert!(DEFAULT_BUDGET_BYTES == SHARDS * 1024);
    assert!(MAX_LANE_BITS == 32 - SHARD_BITS);
};
/// Inputs shorter than this use the original ordered fold without partitioning.
const BUILD_PARTITION_CROSSOVER: usize = 4096;
const PARTITION_CHUNK: usize = 4096;
const MAX_GROUP: usize = 64;
/// The live bit of a spill tag; every written tag is nonzero.
const TAG_LIVE: u8 = 0x80;
/// Number of fingerprint bits carried in a tag below the live bit.
const TAG_FINGERPRINT_BITS: u32 = 7;
/// Shift that lifts the top fingerprint bits of the spread place word into the tag byte.
///
/// A fingerprint read as a raw window of `place` is a function of bits the same `Home` word already
/// spent on the shard selector ([`shard`], the low eight bits) and on the descent windows ([`group`],
/// bits 8..56), and a `Home` is caller-supplied.  A `Home` may therefore hold any one window of
/// `place` constant across a whole key population while still returning distinct places: the
/// identity home over keys `j << 16` leaves `place & 0xFFFF == 0` for every key, so a raw
/// `(place >> 8) & 0x7f` is one constant over the population and every lane of every visited group
/// carries the query's fingerprint.  Reading the *top* bits of a spread word closes that route: for
/// any `k`, the population `place = j << k` has `place * TAG_SPREAD` with distinct top
/// [`TAG_FINGERPRINT_BITS`] over distinct `j`, because equality of those bits makes
/// `j * TAG_SPREAD` agree modulo `2^(64 - k)` and a multiply by an odd number is injective on
/// `Z/2^64`.  This is the placement of `hashbrown`'s `h2`, taken from the high end of the hash
/// rather than from the bits `h1` already consumed.
const TAG_FINGERPRINT_SHIFT: u32 = 64 - TAG_FINGERPRINT_BITS;
/// The odd spread multiplier applied to the place word before the fingerprint bits are read.
///
/// `crate::coord::PHI`, the same odd constant [`summary_index`](table::summary_index) mixes with and
/// [`WordHome`](table::WordHome) homes by.  One 64-bit multiply per probe, not per lane.
const TAG_SPREAD: u64 = crate::coord::PHI;
/// Mask of the fingerprint bits below the live bit.
const TAG_FINGERPRINT_MASK: u8 = (1 << TAG_FINGERPRINT_BITS) - 1;
const DESCENT_MASK: u64 = (1 << 49) - 1;
const WINDOW_MASK: u64 = 0x7f;
const WINDOWS: usize = 7;

/// Resource and checked-sizing outcomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// A checked layout calculation did not fit `usize` or the group width is invalid.
    SizeOverflow,
    /// A checked reservation could not be obtained; committed state is unchanged.
    AllocationFailed,
    /// Compatibility-only fixed-capacity error; this growable table does not return it.
    ShardFull,
}

/// Placement plus an optional exact, certified coordinate word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableAddress {
    /// Placement used by grouped spill probing.
    pub place: u64,
    /// Exact coordinate used only by the resident admission rule.
    pub direct_word: Option<u32>,
}

/// Supplies stable physical placement and, optionally, an exact coordinate certificate.
///
/// `Home` is the physical storage adapter during the flat/spill migration. It also inherits the
/// semantic [`FaceHome`] capability, so every home declares its face encodings; the physical
/// [`Home::address`] result is never a semantic address or an absence certificate.
///
/// There is deliberately no blanket implementation for placement-only closures.
///
/// ```compile_fail
/// use apeiron::table::Table;
///
/// // Rejected: the blanket `Fn(&K) -> u64` Home implementation is removed, so a closure is not
/// // a `Home`.
/// let home = |key: &i64| *key as u64;
/// let _ = Table::<i64, i64, _, 16>::new(0, home);
/// ```
pub trait Home<K>: FaceHome<K> {
    /// Returns the placement word.
    fn home(&self, key: &K) -> u64;
    /// Returns the `Sigma`-indexed physical address.
    ///
    /// Call as `Home::address` to disambiguate it from the inherited [`FaceHome::address`].
    fn address(&self, sigma: Sigma, key: &K) -> TableAddress;
}

/// Combines a new value with the prior value.
pub trait Merge<V> {
    /// Applies the new-first merge.
    fn merge(&self, new: V, old: V) -> V;
}
impl<V, F: Fn(V, V) -> V> Merge<V> for F {
    fn merge(&self, new: V, old: V) -> V {
        self(new, old)
    }
}

/// Word-coordinate placement and certified sub-`2^32` direct words.
///
/// The physical adapter is [`Home::home`]/[`Home::address`]. Its semantic capability is the same
/// all-face word map as [`WordFaceHome`], delegated rather than re-encoded.
pub struct WordHome;
impl Home<i64> for WordHome {
    fn home(&self, key: &i64) -> u64 {
        crate::coord::spread((*key).cast_unsigned()).place
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::checked_conversions,
        reason = "the preceding explicit bound proves the u64 value fits u32"
    )]
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        let word = (*key).cast_unsigned();
        TableAddress {
            place: crate::coord::spread(word).place,
            direct_word: if word <= u64::from(u32::MAX) {
                Some(word as u32)
            } else {
                None
            },
        }
    }
}
impl FaceResidueCapability<i64> for WordHome {
    fn parabolic_dual(&self, key: &i64) -> ParabolicDual {
        WordFaceHome.parabolic_dual(key)
    }
    fn hyperbolic_scalar(&self, profile: H3_40, key: &i64) -> Residue {
        WordFaceHome.hyperbolic_scalar(profile, key)
    }
}
impl EllipticPacking<i64> for WordHome {
    fn elliptic_pack(&self, key: &i64) -> u64 {
        WordFaceHome.elliptic_pack(key)
    }
    fn elliptic_unpack(&self, word: u64) -> Option<i64> {
        WordFaceHome.elliptic_unpack(word)
    }
}
impl FaceCoord<i64> for WordHome {
    type Faces = Sigma;
}
impl FaceHome<i64> for WordHome {}
/// Wrapping word addition.
pub struct WordMerge;
impl Merge<i64> for WordMerge {
    fn merge(&self, new: i64, old: i64) -> i64 {
        new.wrapping_add(old)
    }
}
#[derive(Clone, Copy, Debug, Default)]
/// New-first replacement merge.
pub struct ReplacementMerge;
impl Merge<i64> for ReplacementMerge {
    fn merge(&self, new: i64, _: i64) -> i64 {
        new
    }
}

// The region/summary/decision/scalar-geometry and rebuild-state declarations are included
// textually so they keep the `table` module namespace and generated type identities; the
// moved bodies live in `src/table/state.rs` unchanged. See the plan's exact size cut.
include!("table/state.rs");
// The handle-only directory state, its reverse links and the storage-coverage checker are included
// the same way: they are definitions of the `table` module and read the table's private storage.
include!("table/locator.rs");

// Textual inclusion preserves the extracted table namespace and free-function identities.
include!("table/prepared.rs");

/// A total-default table whose compile-time `R` policy selects carried or regenerated keys.
pub struct Table<K, V, H, const N: usize = 16, R = Stored<K>>
where
    R: KeyRep<K, H>,
{
    default: V,
    home: H,
    resident: Flat<R, V>,
    shards: Vec<Region<R, V>>,
    len: usize,
    key: PhantomData<K>,
    /// The handle-only directory, its reverse links and its private staging state.
    locator: TableLocator,
}
impl<K, V, H: Home<K>, const N: usize, R: KeyRep<K, H>> Table<K, V, H, N, R> {
    /// Creates an empty table without allocating resident slots.
    #[must_use]
    pub fn new(default: V, home: H) -> Self {
        let mut shards = Vec::new();
        let mut shard_at = 0;
        while shard_at < SHARDS {
            shards.push(Region::empty());
            shard_at += 1;
        }
        Self {
            default,
            home,
            resident: Flat::empty(),
            shards,
            len: 0,
            key: PhantomData,
            locator: TableLocator::new(),
        }
    }
}
impl<K, V, H: Home<K>, const N: usize, R: KeyRep<K, H>> Table<K, V, H, N, R> {
    /// Walks the direct candidate, seven descent windows, and cyclic coverage sweep.
    ///
    /// This classifier deliberately carries no borrowed payload or value.  Application of the
    /// selected real storage access is isolated in [`Self::apply_get`] and
    /// [`Self::apply_insert`].
    #[must_use]
    pub fn search_decision_at(&self, face: H::Faces, key: &K) -> SearchDecision {
        // Erasure boundary: the typed token becomes the stored `Sigma` here.
        let sigma = face.sigma();
        let address = Home::address(&self.home, sigma, key);
        self.search_decision(sigma, key, &address)
    }

    fn search_decision(&self, sigma: Sigma, key: &K, address: &TableAddress) -> SearchDecision {
        if let Some(word) = address.direct_word {
            if let Some(i) = self.resident.index(word) {
                let flat = &self.resident.slots[i];
                if flat.live && flat.sigma == sigma && R::matches(sigma, &flat.held, key, address) {
                    return SearchDecision::found_flat(i);
                }
                if !flat.live {
                    return SearchDecision::vacant_flat(i);
                }
            }
        }
        let shard_at = shard(address);
        SearchDecision::from_spill(
            shard_at,
            spill_locate::<K, H, R, V>(&self.shards[shard_at], sigma, address, key, N),
        )
    }

    /// Applies one already-classified read to the selected real storage location.
    ///
    /// The decision walk is separate from this straight-line access so an extraction root need not
    /// contain the probe loop's borrowed generic carrier.  No indexed location returns the default.
    ///
    /// This is the extraction root's straight-line apply; [`Self::get_at`] inlines the same
    /// resident classification and spill indexing so the production read builds no insertion
    /// decision.
    #[allow(
        dead_code,
        reason = "the Aeneas-only extraction wrapper calls this method; the production read inlines it"
    )]
    pub(crate) fn apply_get(&self, decision: SearchDecision) -> &V {
        match decision.location {
            SearchLocation::Flat(index) if decision.class == SearchClass::Found => {
                &self.resident.slots[index].value
            }
            SearchLocation::Spill(shard_at, index) if decision.class == SearchClass::Found => {
                &self.shards[shard_at].slots[index].2
            }
            _ => &self.default,
        }
    }
    /// Reads a face-qualified key, returning the default when absent.
    ///
    /// The read inlines resident classification, flat indexing and read application, then routes
    /// every occupied non-match or out-of-extent coordinate to the read-only spill matcher.  It
    /// performs no insertion vacancy search: absence is the direct-empty `DirectCover` certificate,
    /// a routed-spill summary miss after direct exclusion, or complete finite match coverage.
    ///
    /// The face argument is the home's own [`FaceDomain`] token, so a locator-only home cannot be
    /// passed the elliptic face: the restriction is a type, not a runtime branch. The token is
    /// erased to the stored `Sigma` at this boundary.
    ///
    /// ```compile_fail
    /// use apeiron::coord::address::{
    ///     FaceCoord, FaceHome, FaceResidueCapability, LocatorSigma, ParabolicDual, Residue, H3_40,
    /// };
    /// use apeiron::coord::spread;
    /// use apeiron::table::{Home, Table, TableAddress};
    /// use apeiron::Sigma;
    ///
    /// struct LocatorOnlyHome;
    ///
    /// impl FaceResidueCapability<i64> for LocatorOnlyHome {
    ///     fn parabolic_dual(&self, key: &i64) -> ParabolicDual {
    ///         ParabolicDual::new(spread(key.cast_unsigned()).place, 0)
    ///     }
    ///     fn hyperbolic_scalar(&self, profile: H3_40, key: &i64) -> Residue {
    ///         Residue::reduce(profile, key.cast_unsigned())
    ///     }
    /// }
    ///
    /// impl FaceCoord<i64> for LocatorOnlyHome {
    ///     type Faces = LocatorSigma;
    /// }
    ///
    /// impl FaceHome<i64> for LocatorOnlyHome {}
    ///
    /// impl Home<i64> for LocatorOnlyHome {
    ///     fn home(&self, key: &i64) -> u64 {
    ///         key.cast_unsigned()
    ///     }
    ///     fn address(&self, _: Sigma, key: &i64) -> TableAddress {
    ///         TableAddress { place: self.home(key), direct_word: None }
    ///     }
    /// }
    ///
    /// let table = Table::<i64, i64, LocatorOnlyHome, 16>::new(0, LocatorOnlyHome);
    /// // Rejected: `LocatorOnlyHome::Faces` is `LocatorSigma`, so `get_at` expects a
    /// // `LocatorSigma` token; `Sigma::Elliptic` is not one. There is no runtime
    /// // `UnsupportedFace` branch to reach.
    /// let _ = table.get_at(Sigma::Elliptic, &7);
    /// ```
    #[must_use]
    pub fn get_at(&self, face: H::Faces, key: &K) -> &V {
        let sigma = face.sigma();
        let address = Home::address(&self.home, sigma, key);
        if let Some(word) = address.direct_word {
            if let Some(index) = self.resident.index(word) {
                let flat = &self.resident.slots[index];
                if flat.live {
                    if flat.sigma == sigma && R::matches(sigma, &flat.held, key, &address) {
                        return &flat.value;
                    }
                } else {
                    return &self.default;
                }
            }
        }
        let shard_at = shard(&address);
        match spill_read::<K, H, R, V>(&self.shards[shard_at], sigma, &address, key, N) {
            Some(index) => &self.shards[shard_at].slots[index].2,
            None => &self.default,
        }
    }
    /// Reads the routed spill region under the prepared-region precondition.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the Aeneas-only extraction wrapper calls this method"
    )]
    pub(crate) fn get_prepared_shard_at(&self, face: H::Faces, key: &K) -> &V {
        let sigma = face.sigma();
        let address = Home::address(&self.home, sigma, key);
        get_prepared_shard::<K, H, R, V>(&self.shards[shard(&address)], face, &address, key, N)
            .unwrap_or(&self.default)
    }
    /// Reads at the home's default face ([`FaceDomain::DEFAULT`], parabolic for both admitted
    /// domains).
    #[must_use]
    pub fn get(&self, key: &K) -> &V {
        self.get_at(<H::Faces as FaceDomain<K, H>>::DEFAULT, key)
    }
}
impl<K: Clone, V: Clone, H: Home<K>, const N: usize, R: KeyRep<K, H>> Table<K, V, H, N, R> {
    /// Returns the number of logical overrides.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }
    /// Reports whether no logical override exists.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Counts distinct certified direct words below `lanes * 256` across the resident and spill
    /// areas, plus the pending `extra` word.
    ///
    /// The dedup is a direct-word bitmap rather than a linear search over a growing list: the
    /// candidate lanes only ever reach the byte-budget ceiling, so `limit` is at most a few
    /// thousand words and a bitmap of that many bits is exact and O(live). The former list-plus-
    /// scan made one `selected_lanes` call O(live^2), which the growth-insert trace paid on every
    /// extent decision. Temporal O(live) counted in live slots; spatial O(lanes) bits.
    ///
    /// The bitmap is part of the reserve-all path, so its sizing and reservation are checked: a
    /// `lanes * 256` overflow is [`StorageError::SizeOverflow`] and a refused reservation is
    /// [`StorageError::AllocationFailed`]. A refusal leaves `self` unchanged and runs no hold,
    /// clone, merge or relocation.
    fn density_at(&self, lanes: usize, extra: Option<u32>) -> Result<usize, StorageError> {
        let limit = (lanes as u64)
            .checked_mul(SHARDS as u64)
            .ok_or(StorageError::SizeOverflow)?;
        if limit > usize::MAX as u64 {
            return Err(StorageError::SizeOverflow);
        }
        let bits = limit as usize;
        let mut words = bits / 64;
        if bits % 64 != 0 {
            words += 1;
        }
        if words == 0 {
            words = 1;
        }
        let mut seen: Vec<u64> = Vec::new();
        try_reserve_exact(&mut seen, words)?;
        seen.resize(words, 0);
        let mut count = 0usize;
        let mut i = 0;
        while i < self.resident.slots.len() {
            let direct = if self.resident.slots[i].live {
                let sigma = self.resident.slots[i].sigma;
                R::address_held(&self.home, sigma, &self.resident.slots[i].held).direct_word
            } else {
                None
            };
            if let Some(word) = direct {
                mark_direct_word(&mut seen, &mut count, word, limit);
            }
            i += 1;
        }
        let mut shard_at = 0;
        while shard_at < self.shards.len() {
            let mut j = 0;
            while j < self.shards[shard_at].slots.len() {
                let direct = if self.shards[shard_at].tags[j] != 0 {
                    let sigma = self.shards[shard_at].slots[j].0;
                    R::address_held(&self.home, sigma, &self.shards[shard_at].slots[j].1)
                        .direct_word
                } else {
                    None
                };
                if let Some(word) = direct {
                    mark_direct_word(&mut seen, &mut count, word, limit);
                }
                j += 1;
            }
            shard_at += 1;
        }
        if let Some(word) = extra {
            mark_direct_word(&mut seen, &mut count, word, limit);
        }
        Ok(count)
    }
    fn selected_lanes(
        &self,
        required_lane: usize,
        extra: u32,
    ) -> Result<Option<usize>, StorageError> {
        // A coordinate beyond the byte-budget ceiling cannot become resident.  Reject it before
        // counting the table-wide density, so distant coordinates do not make insertion scale with
        // their value or with the current spill population.
        let mut error = None;
        let mut budget_lanes = 1usize;
        let mut budget_reached = false;
        while budget_lanes <= (1usize << MAX_LANE_BITS) && !budget_reached && error.is_none() {
            match Flat::<R, V>::byte_len_for(budget_lanes) {
                Some(bytes) if bytes > DEFAULT_BUDGET_BYTES => budget_reached = true,
                Some(_) => match budget_lanes.checked_mul(2) {
                    Some(next) => budget_lanes = next,
                    None => error = Some(StorageError::SizeOverflow),
                },
                None => error = Some(StorageError::SizeOverflow),
            }
        }
        let mut result = None;
        if required_lane < budget_lanes / 2 && error.is_none() {
            let mut lane = 1usize;
            let mut best = None;
            let mut budget_exceeded = false;
            while lane <= (1usize << MAX_LANE_BITS) && !budget_exceeded && error.is_none() {
                match Flat::<R, V>::byte_len_for(lane) {
                    Some(bytes) if bytes > DEFAULT_BUDGET_BYTES => budget_exceeded = true,
                    Some(_) => match lane.checked_mul(SHARDS) {
                        Some(capacity) => match self.density_at(lane, Some(extra)) {
                            Ok(density) => {
                                let doubled = match density.checked_mul(2) {
                                    Some(value) => value,
                                    None => usize::MAX,
                                };
                                if capacity <= SHARDS.max(doubled) {
                                    best = Some(lane);
                                }
                                match lane.checked_mul(2) {
                                    Some(next) => lane = next,
                                    None => error = Some(StorageError::SizeOverflow),
                                }
                            }
                            Err(inner) => error = Some(inner),
                        },
                        None => error = Some(StorageError::SizeOverflow),
                    },
                    None => error = Some(StorageError::SizeOverflow),
                }
            }
            if let Some(lanes) = best {
                if required_lane < lanes {
                    result = Some(lanes);
                }
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(result),
        }
    }
    #[allow(
        clippy::many_single_char_names,
        reason = "index and payload names mirror the bounded relocation loop"
    )]
    fn grow_spill<F: Merge<V>>(
        &mut self,
        s: usize,
        merge: &F,
        sigma: Sigma,
        key: K,
        value: V,
        address: &TableAddress,
    ) -> Result<(), StorageError> {
        let mut load = 0;
        let mut tag_at = 0;
        while tag_at < self.shards[s].tags.len() {
            load += usize::from(self.shards[s].tags[tag_at] != 0);
            tag_at += 1;
        }
        let minimum = slots_for(load.max(1), N)?;
        let doubled = self.shards[s]
            .slots
            .len()
            .checked_mul(2)
            .ok_or(StorageError::SizeOverflow)?;
        let target = minimum.max(doubled);
        let mut fresh = Region::reserve_empty(target)?;
        let held = R::hold(sigma, &key, address);
        fresh.initialize(sigma, &held, &self.default);
        let mut i = 0;
        let mut relocation_error = None;
        while i < self.shards[s].slots.len() && relocation_error.is_none() {
            if self.shards[s].tags[i] != 0 {
                let (p_sigma, p_held, p_value) = self.shards[s].slots[i].clone();
                let a = R::address_held(&self.home, p_sigma, &p_held);
                match spill_locate_held::<K, H, R, V>(&fresh, p_sigma, &a, &p_held, N) {
                    SpillSite::Vacant(x) => {
                        // Relocation retargets the moved representative's reverse link without a
                        // merge; the terminal handle is unchanged, only its location moves.
                        self.locator.retarget_held::<K, H, R>(
                            &self.home,
                            p_sigma,
                            &p_held,
                            PayloadLocation::Spill(s, i),
                            PayloadLocation::Spill(s, x),
                        );
                        fresh.publish_live(x, (p_sigma, p_held, p_value), a.place);
                    }
                    _ => relocation_error = Some(StorageError::SizeOverflow),
                }
            }
            i += 1;
        }
        if let Some(error) = relocation_error {
            return Err(error);
        }
        let SpillSite::Vacant(pending) =
            spill_locate_held::<K, H, R, V>(&fresh, sigma, address, &held, N)
        else {
            unreachable!("checked spill growth has a pending vacancy")
        };
        let merged = merge.merge(value, self.default.clone());
        fresh.publish_live(pending, (sigma, held, merged), address.place);
        self.shards[s] = fresh;
        // The pending path was staged before this effect; publish its reverse link now.
        self.locator.publish_for_query::<K, H, R>(
            &self.home,
            sigma,
            &key,
            PayloadLocation::Spill(s, pending),
        );
        self.len += 1;
        Ok(())
    }
    fn plan_rebuild(
        &self,
        lanes: usize,
        _sigma: Sigma,
        address: &TableAddress,
    ) -> Result<RebuildPlan, StorageError> {
        let flat_len = lanes
            .checked_mul(SHARDS)
            .ok_or(StorageError::SizeOverflow)?;
        let mut occupied: Vec<bool> = Vec::new();
        try_reserve_exact(&mut occupied, flat_len)?;
        occupied.resize(flat_len, false);
        let mut entries: Vec<RebuildPlanEntry> = Vec::new();
        try_reserve_exact(&mut entries, self.len)?;
        let mut spill_loads: Vec<usize> = Vec::new();
        try_reserve_exact(&mut spill_loads, SHARDS)?;
        spill_loads.resize(SHARDS, 0);
        let mut planning_error = None;
        let mut resident_at = 0;
        while resident_at < self.resident.slots.len() && planning_error.is_none() {
            let x = &self.resident.slots[resident_at];
            if x.live {
                let entry_address = R::address_held(&self.home, x.sigma, &x.held);
                let flat = Self::flat_index_for(lanes, entry_address.direct_word);
                let selected = Self::keep_unoccupied(flat, &occupied);
                mark_selected_occupied(&mut occupied, selected);
                let spill = shard(&entry_address);
                planning_error = account_spill_load(&mut spill_loads, spill, selected);
                entries.push(RebuildPlanEntry {
                    source: RebuildSource::Resident(resident_at),
                    flat: selected,
                    spill,
                });
            }
            resident_at += 1;
        }
        let mut shard_at = 0;
        while shard_at < self.shards.len() && planning_error.is_none() {
            let mut slot_at = 0;
            while slot_at < self.shards[shard_at].slots.len() && planning_error.is_none() {
                let source_live = self.shards[shard_at].tags[slot_at] != 0;
                if source_live {
                    let entry = &self.shards[shard_at].slots[slot_at];
                    let entry_address = R::address_held(&self.home, entry.0, &entry.1);
                    let flat = Self::flat_index_for(lanes, entry_address.direct_word);
                    let selected = Self::keep_unoccupied(flat, &occupied);
                    mark_selected_occupied(&mut occupied, selected);
                    let spill = shard(&entry_address);
                    planning_error = account_spill_load(&mut spill_loads, spill, selected);
                    entries.push(RebuildPlanEntry {
                        source: RebuildSource::Spill(shard_at, slot_at),
                        flat: selected,
                        spill,
                    });
                }
                slot_at += 1;
            }
            shard_at += 1;
        }
        let pending_flat = Self::flat_index_for(lanes, address.direct_word);
        let pending_occupied = match pending_flat {
            Some(index) => occupied[index],
            None => true,
        };
        if pending_occupied && planning_error.is_none() {
            let pending_shard = shard(address);
            planning_error = increment_spill_load(&mut spill_loads, pending_shard);
        }
        match planning_error {
            Some(error) => Err(error),
            None => Ok(RebuildPlan {
                lanes,
                entries,
                spill_loads,
            }),
        }
    }

    fn reserve_rebuild(plan: &RebuildPlan) -> Result<RebuildReserved<R, V>, StorageError> {
        let resident = Flat::reserve(plan.lanes)?;
        let regions = reserve_plan_regions::<R, V, N>(&plan.spill_loads)?;
        Ok(RebuildReserved { resident, regions })
    }

    /// Observes the reserved capacities and checks every receipt bound.
    ///
    /// `required_len` is the checked target vector length; `observed_capacity` is the real
    /// `Vec::capacity()` read after reserve-all and before initialization.  The flat target is
    /// checked `256 * plan.lanes`, the directory target is `SHARDS`, and each region's tag/payload
    /// target is the checked `slots_for(plan.spill_loads[shard], N)` or zero for zero load.  The
    /// summary target is the checked `summary_words` of that slot target, or zero for an empty
    /// region.
    fn observe_receipt(
        plan: &RebuildPlan,
        storage: &RebuildReserved<R, V>,
    ) -> Result<ReservationReceipt, StorageError> {
        let flat_required = plan
            .lanes
            .checked_mul(SHARDS)
            .ok_or(StorageError::SizeOverflow)?;
        let flat = capacity_bound(flat_required, storage.resident.slots.capacity());
        let directory = capacity_bound(SHARDS, storage.regions.capacity());
        let mut regions = [const {
            RegionReservation {
                tags: CapacityBound {
                    required_len: 0,
                    observed_capacity: 0,
                },
                payloads: CapacityBound {
                    required_len: 0,
                    observed_capacity: 0,
                },
                summary: CapacityBound {
                    required_len: 0,
                    observed_capacity: 0,
                },
            }
        }; SHARDS];
        let mut shard_at = 0;
        let mut error = None;
        while shard_at < SHARDS && error.is_none() {
            match slots_for(plan.spill_loads[shard_at], N) {
                Ok(required) => match summary_words(required) {
                    Ok(summary_required) => {
                        regions[shard_at] = RegionReservation {
                            tags: capacity_bound(
                                required,
                                storage.regions[shard_at].tags.capacity(),
                            ),
                            payloads: capacity_bound(
                                required,
                                storage.regions[shard_at].slots.capacity(),
                            ),
                            summary: capacity_bound(
                                summary_required,
                                storage.regions[shard_at].summary.capacity(),
                            ),
                        };
                        shard_at += 1;
                    }
                    Err(inner) => error = Some(inner),
                },
                Err(inner) => error = Some(inner),
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(ReservationReceipt {
                flat,
                directory,
                regions,
            }),
        }
    }

    /// Holds the pending carrier and materializes every reserved vector, then pairs the storage with
    /// the already-checked capacity receipt and the held carrier.
    ///
    /// This is the adapter's initialization phase: it runs after reserve-all, the receipt check and
    /// the locator-path reservation, uses the real pending held carrier and table default, replaces
    /// no reserved vector, allocates nothing beyond the checked capacity and has no failure path.
    #[allow(
        clippy::too_many_arguments,
        reason = "the adapter's initialization phase is explicit at the transaction boundary"
    )]
    fn materialize_rebuild(
        &self,
        plan: RebuildPlan,
        mut storage: RebuildReserved<R, V>,
        reservation: ReservationReceipt,
        locator: LocatorReservationReceipt,
        sigma: Sigma,
        key: &K,
        address: &TableAddress,
    ) -> PreparedEnvelope<R, V> {
        let pending_held = R::hold(sigma, key, address);
        storage
            .resident
            .initialize(sigma, &pending_held, &self.default);
        initialize_reserved_regions::<R, V>(
            &mut storage.regions,
            sigma,
            &pending_held,
            &self.default,
        );
        (
            plan,
            storage,
            PreparedOutcome {
                reservation,
                locator,
                pending_held,
            },
        )
    }

    /// Runs the unextracted resource adapter for an already checked plan.
    ///
    /// The only admitted order is reserve every flat/directory/region allocation, observe and check
    /// the table capacity receipt, reserve and stage the locator path and check its receipt, hold the
    /// pending carrier, materialize the flat and region payloads, then publish the exact success
    /// tuple. A refusal at any step exposes no plan, storage, held carrier or staged link, and the
    /// locator path is reserved before the holds and the table commit, so no locator failure can
    /// occur after a table merge.
    fn prepare_from_plan(
        &mut self,
        plan: RebuildPlan,
        sigma: Sigma,
        key: &K,
        address: &TableAddress,
    ) -> PreparedResult<R, V> {
        let storage = match Self::reserve_rebuild(&plan) {
            Ok(storage) => storage,
            Err(error) => return publish_prepared(Err(error)),
        };
        let reservation = match Self::observe_receipt(&plan, &storage) {
            Ok(reservation) => reservation,
            Err(error) => return publish_prepared(Err(error)),
        };
        // Reserve the locator path last, after every table reservation and receipt check, so a
        // locator refusal drops the uncommitted table storage and stages nothing. The holds below
        // run only after both reservations succeed, so no locator failure occurs after a hold.
        let locator = match self.reserve_locator_path(sigma, key) {
            Ok(receipt) => receipt,
            Err(error) => return publish_prepared(Err(error)),
        };
        publish_prepared(Ok(self.materialize_rebuild(
            plan,
            storage,
            reservation,
            locator,
            sigma,
            key,
            address,
        )))
    }

    /// Runs the checked plan followed by the resource adapter, publishing the joint envelope.
    fn prepare_rebuild(
        &mut self,
        lanes: usize,
        sigma: Sigma,
        key: &K,
        address: &TableAddress,
    ) -> PreparedResult<R, V> {
        match self.plan_rebuild(lanes, sigma, address) {
            Ok(plan) => self.prepare_from_plan(plan, sigma, key, address),
            Err(error) => publish_prepared(Err(error)),
        }
    }

    /// `Some(index)` when the optional flat index is not yet occupied, else `None`.
    ///
    /// A named call, not an in-loop closure or guarded match: the extractor joins loop contexts,
    /// and an opaque call keeps the body out of the joined context.
    fn keep_unoccupied(index: Option<usize>, occupied: &[bool]) -> Option<usize> {
        match index {
            Some(index) => {
                if occupied[index] {
                    None
                } else {
                    Some(index)
                }
            }
            None => None,
        }
    }

    fn flat_index_for(lanes: usize, word: Option<u32>) -> Option<usize> {
        let word = match word {
            Some(word) => word,
            None => return None,
        };
        let lane = (word >> SHARD_BITS) as usize;
        if lane >= lanes {
            return None;
        }
        let slot = (word as usize) & SHARD_MASK;
        let scaled = match slot.checked_mul(lanes) {
            Some(value) => value,
            None => return None,
        };
        scaled.checked_add(lane)
    }

    /// Classifies the pending key in the already-relocated reserved storage.
    ///
    /// This is the rebuilt-state counterpart of [`Self::search_decision`]: it walks the direct
    /// candidate, then the routed spill region, and returns the same value-only
    /// [`SearchDecision`].  [`Self::apply_pending_and_commit`] consumes the returned
    /// class/location without re-probing, so this function and [`Flat::index`] stay outside the
    /// extracted commit cone.
    fn locate_rebuilt(
        resident: &Flat<R, V>,
        regions: &[Region<R, V>],
        sigma: Sigma,
        key: &K,
        address: &TableAddress,
    ) -> SearchDecision {
        if let Some(word) = address.direct_word {
            if let Some(i) = resident.index(word) {
                let flat = &resident.slots[i];
                if flat.live && flat.sigma == sigma && R::matches(sigma, &flat.held, key, address) {
                    return SearchDecision::found_flat(i);
                }
                if !flat.live {
                    return SearchDecision::vacant_flat(i);
                }
            }
        }
        SearchDecision::from_spill(
            shard(address),
            spill_locate::<K, H, R, V>(&regions[shard(address)], sigma, address, key, N),
        )
    }

    /// Moves one checked-plan entry to its reserved destination.
    ///
    /// This is the straight-line relocation core.  It clones the selected live source payload from
    /// the old resident area or old spill region, writes the planned flat destination or the
    /// plan-certified spill vacancy, and sets the spill occupancy tag.  It does not initialize a
    /// vector, allocate, merge, insert the pending value, change `self.len` or commit `reserved`
    /// into `self`; the production rebuild path materializes payloads before calling it and applies
    /// the pending merge after the relocation loop.
    ///
    /// # Panics
    /// Panics when the checked plan's spill destination is not a vacancy, which the checked-plan
    /// premise excludes.
    pub(crate) fn relocate_entry(
        &mut self,
        plan: &RebuildPlan,
        reserved: &mut RebuildReserved<R, V>,
        entry_at: usize,
    ) {
        let entry = &plan.entries[entry_at];
        let source = entry.source;
        let (entry_sigma, entry_held, entry_value) = match source {
            RebuildSource::Resident(index) => {
                let source = &self.resident.slots[index];
                (source.sigma, source.held.clone(), source.value.clone())
            }
            RebuildSource::Spill(shard, index) => {
                let source = &self.shards[shard].slots[index];
                (source.0, source.1.clone(), source.2.clone())
            }
        };
        let old_location = match source {
            RebuildSource::Resident(index) => PayloadLocation::Resident(index),
            RebuildSource::Spill(shard, index) => PayloadLocation::Spill(shard, index),
        };
        if let Some(index) = entry.flat {
            // Retarget before the move: the directory is unchanged, so the terminal handle is
            // recovered from the payload identity and only its recorded location changes.
            self.locator.retarget_held::<K, H, R>(
                &self.home,
                entry_sigma,
                &entry_held,
                old_location,
                PayloadLocation::Resident(index),
            );
            reserved.resident.slots[index] = FlatSlot {
                live: true,
                sigma: entry_sigma,
                held: entry_held,
                value: entry_value,
            };
        } else {
            let entry_address = R::address_held(&self.home, entry_sigma, &entry_held);
            let SpillSite::Vacant(index) = spill_locate_held::<K, H, R, V>(
                &reserved.regions[entry.spill],
                entry_sigma,
                &entry_address,
                &entry_held,
                N,
            ) else {
                unreachable!("checked rebuild plan has a spill vacancy")
            };
            self.locator.retarget_held::<K, H, R>(
                &self.home,
                entry_sigma,
                &entry_held,
                old_location,
                PayloadLocation::Spill(entry.spill, index),
            );
            reserved.regions[entry.spill].publish_live(
                index,
                (entry_sigma, entry_held, entry_value),
                entry_address.place,
            );
        }
    }

    /// Relocates every checked-plan entry, in plan order, through [`Self::relocate_entry`].
    ///
    /// This is the straight-line relocation traversal.  It initializes no vector, allocates, merges,
    /// inserts the pending value, changes `self.len` or commits `reserved` into `self`; the
    /// production rebuild path materializes payloads before calling it and applies the pending merge
    /// exactly once after it returns.
    ///
    /// Complexity: `O(plan.entries.len())` entry steps, each `O(1)` plus one payload clone and one
    /// spill-locate probe, counted in moved entries.
    pub(crate) fn relocate_all(
        &mut self,
        plan: &RebuildPlan,
        reserved: &mut RebuildReserved<R, V>,
    ) {
        let mut entry_at = 0;
        while entry_at < plan.entries.len() {
            self.relocate_entry(plan, reserved, entry_at);
            entry_at += 1;
        }
    }

    /// Returns an unused staged path's scratch buffer to the directory's idle slot.
    ///
    /// A found update keeps its existing reverse link, so its staged path is never published; the
    /// buffer is still returned through [`Directory::recycle_staged`], the entry point the
    /// publication and refusal paths use, so the next reservation hands out the same allocation
    /// instead of allocating a fresh one. Temporal `O(1)`; spatial `O(1)`: the buffer is moved
    /// into the idle slot, never copied.
    fn recycle_unused_staged_path(&mut self) {
        let staged = match self.locator.staged.take() {
            Some((_, staged)) => staged,
            None => return,
        };
        // Staging materializes the directory, so the `None` arm is the same unreachable defensive
        // shape as in `TableLocator::publish_for_query`.
        if let Some(directory) = &mut self.locator.directory {
            directory.recycle_staged(staged);
        }
    }

    /// Applies the supplied pending classification to the relocated reserved storage, then commits.
    ///
    /// This is the straight-line pending-insertion and commit core.  It consumes the precomputed
    /// [`SearchDecision`] for the pending `(sigma, key)` in the already-relocated reserved storage,
    /// merges `value` through `merge` exactly once, increments `self.len` exactly once for a fresh
    /// key, and moves `reserved.resident` and `reserved.regions` into `self`.  It does not probe,
    /// initialize payloads, relocate entries, or allocate, and it has no error path, so no clone
    /// runs on a refusal.
    ///
    /// Complexity: one merge call and one payload clone on a found path, plus the two reserved
    /// vector moves; counted in merges, clones and moved entries.
    #[allow(
        clippy::too_many_arguments,
        reason = "the pending insertion is explicit at the transaction commit boundary"
    )]
    pub(crate) fn apply_pending_and_commit<F: Merge<V>>(
        &mut self,
        mut reserved: RebuildReserved<R, V>,
        decision: SearchDecision,
        merge: &F,
        sigma: Sigma,
        value: V,
        held: R,
        address: &TableAddress,
    ) {
        match (decision.class, decision.location) {
            (SearchClass::Found, SearchLocation::Flat(index)) => {
                // A found update keeps its existing reverse link; the staged path is unused and
                // its scratch returns to the directory's idle slot.
                self.recycle_unused_staged_path();
                let old = reserved.resident.slots[index].value.clone();
                reserved.resident.slots[index].value = merge.merge(value, old);
            }
            (SearchClass::Found, SearchLocation::Spill(spill, index)) => {
                // As on the flat arm: the unused staged path recycles, never drops.
                self.recycle_unused_staged_path();
                let old = reserved.regions[spill].slots[index].2.clone();
                reserved.regions[spill].slots[index].2 = merge.merge(value, old);
            }
            (SearchClass::Vacant, SearchLocation::Flat(index)) => {
                let merged = merge.merge(value, self.default.clone());
                self.locator.publish_for_held::<K, H, R>(
                    &self.home,
                    sigma,
                    &held,
                    PayloadLocation::Resident(index),
                );
                reserved.resident.slots[index] = FlatSlot {
                    live: true,
                    sigma,
                    held,
                    value: merged,
                };
                self.len += 1;
            }
            (SearchClass::Vacant, SearchLocation::Spill(spill, index)) => {
                let merged = merge.merge(value, self.default.clone());
                self.locator.publish_for_held::<K, H, R>(
                    &self.home,
                    sigma,
                    &held,
                    PayloadLocation::Spill(spill, index),
                );
                reserved.regions[spill].publish_live(index, (sigma, held, merged), address.place);
                self.len += 1;
            }
            (SearchClass::Exhausted, SearchLocation::NoIndex) => {
                unreachable!("checked rebuild plan reserves the pending spill")
            }
            _ => unreachable!("the probe walk produces only well-formed decisions"),
        }
        self.resident = reserved.resident;
        self.shards = reserved.regions;
    }

    /// Consumes the published prepared envelope, or propagates its refusal.
    ///
    /// On `Prepared` it runs the same order as before: relocate every planned entry through
    /// [`Self::relocate_all`], compute the pending [`SearchDecision`] in the relocated storage
    /// through [`Self::locate_rebuilt`], apply the pending insertion once through the unchanged
    /// decision-consuming [`Self::apply_pending_and_commit`], and commit.  On `Refused` it returns
    /// the error without touching `self`, so no hold, clone, merge or relocation runs.
    pub(crate) fn apply_prepared_rebuild<F: Merge<V>>(
        &mut self,
        prepared: PreparedResult<R, V>,
        merge: &F,
        sigma: Sigma,
        key: K,
        value: V,
        address: &TableAddress,
    ) -> Result<(), StorageError> {
        match prepared {
            PreparedResult::Prepared {
                plan,
                mut storage,
                outcome,
            } => {
                // The receipt's bounds were checked in the adapter before initialization; the
                // pending carrier is the one the adapter held and materialized.
                let PreparedOutcome {
                    reservation: _reservation,
                    locator: _locator,
                    pending_held,
                } = outcome;
                // Phase: relocate every old representative without merge.
                self.relocate_all(&plan, &mut storage);
                // Phase: classify the pending key in the relocated reserved storage, then apply the
                // pending insertion exactly once and commit.
                let decision =
                    Self::locate_rebuilt(&storage.resident, &storage.regions, sigma, &key, address);
                self.apply_pending_and_commit(
                    storage,
                    decision,
                    merge,
                    sigma,
                    value,
                    pending_held,
                    address,
                );
                Ok(())
            }
            PreparedResult::Refused(error) => Err(error),
        }
    }

    /// Performs the checked plan and the resource adapter, then the pending insertion as one
    /// transaction.
    fn rebuild_resident<F: Merge<V>>(
        &mut self,
        lanes: usize,
        merge: &F,
        sigma: Sigma,
        key: K,
        value: V,
        address: &TableAddress,
    ) -> Result<(), StorageError> {
        let prepared = self.prepare_rebuild(lanes, sigma, &key, address);
        self.apply_prepared_rebuild(prepared, merge, sigma, key, value, address)
    }
    /// Applies one already-classified insertion to the selected real storage location.
    ///
    /// The decision walk is separate from this straight-line storage update so an extraction root
    /// need not contain the probe loop's borrowed generic carrier.  Exhaustion is not applied here:
    /// it returns the pending entry as [`ApplyOutcome::NeedsGrowth`] and never allocates.
    pub(crate) fn apply_insert<F: Merge<V>>(
        &mut self,
        decision: SearchDecision,
        merge: &F,
        sigma: Sigma,
        key: K,
        value: V,
        address: &TableAddress,
    ) -> ApplyOutcome<K, V> {
        match (decision.class, decision.location) {
            (SearchClass::Found, SearchLocation::Flat(index)) => {
                let old = self.resident.slots[index].value.clone();
                self.resident.slots[index].value = merge.merge(value, old);
                ApplyOutcome::Committed
            }
            (SearchClass::Found, SearchLocation::Spill(shard_at, index)) => {
                let old = self.shards[shard_at].slots[index].2.clone();
                self.shards[shard_at].slots[index].2 = merge.merge(value, old);
                ApplyOutcome::Committed
            }
            (SearchClass::Vacant, SearchLocation::Flat(index)) => {
                let merged = merge.merge(value, self.default.clone());
                self.locator.publish_for_query::<K, H, R>(
                    &self.home,
                    sigma,
                    &key,
                    PayloadLocation::Resident(index),
                );
                self.resident.slots[index] = FlatSlot {
                    live: true,
                    sigma,
                    held: R::hold(sigma, &key, address),
                    value: merged,
                };
                self.len += 1;
                ApplyOutcome::Committed
            }
            (SearchClass::Vacant, SearchLocation::Spill(shard_at, index)) => {
                let merged = merge.merge(value, self.default.clone());
                self.locator.publish_for_query::<K, H, R>(
                    &self.home,
                    sigma,
                    &key,
                    PayloadLocation::Spill(shard_at, index),
                );
                self.shards[shard_at].publish_live(
                    index,
                    (sigma, R::hold(sigma, &key, address), merged),
                    address.place,
                );
                self.len += 1;
                ApplyOutcome::Committed
            }
            (SearchClass::Exhausted, SearchLocation::NoIndex) => {
                ApplyOutcome::NeedsGrowth(key, value)
            }
            _ => unreachable!("the probe walk produces only well-formed decisions"),
        }
    }
    /// Inserts at a face, applying `merge` exactly once after any required relocation.
    ///
    /// # Errors
    /// Returns checked sizing or reservation errors without committing a partial relocation.
    pub fn insert_with_at<F: Merge<V>>(
        &mut self,
        merge: &F,
        face: H::Faces,
        key: K,
        value: V,
    ) -> Result<(), StorageError> {
        if !valid_n(N) {
            return Err(StorageError::SizeOverflow);
        }
        // Erasure boundary: the typed token becomes the stored `Sigma` here.
        let sigma = face.sigma();
        let address = Home::address(&self.home, sigma, &key);
        if let Some(word) = address.direct_word {
            let lane = (word >> SHARD_BITS) as usize;
            if lane >= self.resident.lanes {
                if let Some(next) = self.selected_lanes(lane, word)? {
                    return self.rebuild_resident(next, merge, sigma, key, value, &address);
                }
            }
        }
        let decision = self.search_decision(sigma, &key, &address);
        // Stage the pending directory path before any payload effect for every path that can create
        // a fresh representative. A found update keeps its existing link, so it stages nothing.
        if matches!(decision.class, SearchClass::Vacant | SearchClass::Exhausted) {
            self.reserve_locator_path(sigma, &key)?;
        }
        match self.apply_insert(decision, merge, sigma, key, value, &address) {
            ApplyOutcome::Committed => Ok(()),
            ApplyOutcome::NeedsGrowth(key, value) => {
                self.grow_spill(shard(&address), merge, sigma, key, value, &address)
            }
        }
    }
    /// Applies an insertion to a region whose adapter has already reserved its destination.
    ///
    /// The region core performs no fallible allocation. This table wrapper stages the fresh
    /// representative's directory path before the region effect and publishes its reverse link after,
    /// so the serial prepared path maintains coverage `C`. Its caller supplies the
    /// `PreparedResources` vacancy/capacity premise.
    #[allow(
        dead_code,
        reason = "the Aeneas-only extraction wrapper calls this method"
    )]
    pub(crate) fn apply_prepared_shard_at<F: Merge<V>>(
        &mut self,
        merge: &F,
        face: H::Faces,
        key: &K,
        value: V,
    ) -> Result<(), StorageError> {
        let sigma = face.sigma();
        let address = Home::address(&self.home, sigma, key);
        let shard_at = shard(&address);
        let site = spill_locate::<K, H, R, V>(&self.shards[shard_at], sigma, &address, key, N);
        if matches!(site, SpillSite::Vacant(_)) {
            self.reserve_locator_path(sigma, key)?;
        }
        let inserted = apply_prepared_shard::<K, V, H, R, F, N>(
            &mut self.shards[shard_at],
            merge,
            &self.default,
            face,
            key,
            value,
            &address,
        )?;
        if inserted {
            let SpillSite::Vacant(index) = site else {
                unreachable!("a fresh prepared apply has a vacancy")
            };
            self.locator.publish_for_query::<K, H, R>(
                &self.home,
                sigma,
                key,
                PayloadLocation::Spill(shard_at, index),
            );
            self.len += 1;
        }
        Ok(())
    }
    /// Inserts at the default face.
    ///
    /// # Errors
    /// Propagates the checked sizing or reservation error from [`Self::insert_with_at`].
    pub fn insert_with<F: Merge<V>>(
        &mut self,
        merge: &F,
        key: K,
        value: V,
    ) -> Result<(), StorageError> {
        self.insert_with_at(merge, <H::Faces as FaceDomain<K, H>>::DEFAULT, key, value)
    }
    /// Folds a face-qualified stream in input order.
    ///
    /// Every tuple carries its own [`FaceDomain`] token; the default entry point [`Self::build`]
    /// supplies `<H::Faces as FaceDomain<K,H>>::DEFAULT`.
    ///
    /// # Errors
    /// Returns the first checked sizing or reservation error and returns no partial table.
    pub fn build_at<F: Merge<V>>(
        merge: &F,
        default: V,
        kvs: &[((H::Faces, K), V)],
        home: H,
    ) -> Result<Self, StorageError> {
        if kvs.len() < BUILD_PARTITION_CROSSOVER {
            Self::build_at_sequential(merge, default, kvs, home)
        } else if !valid_n(N) {
            Err(StorageError::SizeOverflow)
        } else {
            let workers = (kvs.len() + PARTITION_CHUNK - 1) / PARTITION_CHUNK;
            let workers = if workers < 1 { 1 } else { workers };
            let parts = partition::Parts::of(&home, kvs, workers);
            let loads = parts.loads();
            let schedule = partition::Schedule::lpt(&loads, workers);
            if !parts.is_intact(&home, kvs) || !schedule.is_complete() {
                Err(StorageError::SizeOverflow)
            } else {
                let mut t = Self::new(default, home);
                // Every nonempty spill region is reserved from its actual routed load before a merge runs.
                let mut s = 0;
                let mut reservation_error = None;
                while s < SHARDS && reservation_error.is_none() {
                    if loads[s] != 0 {
                        let mut first = kvs.len();
                        if let Some(index) = parts.first(s) {
                            first = index;
                        }
                        let reserved: Result<Region<R, V>, StorageError> = if first < kvs.len() {
                            match slots_for(loads[s], N) {
                                Ok(slots) if slots != 0 => Region::reserve_empty(slots),
                                Ok(_) => Err(StorageError::SizeOverflow),
                                Err(error) => Err(error),
                            }
                        } else {
                            Err(StorageError::SizeOverflow)
                        };
                        match reserved {
                            Ok(region) => t.shards[s] = region,
                            Err(error) => reservation_error = Some(error),
                        }
                    }
                    s += 1;
                }
                if let Some(error) = reservation_error {
                    Err(error)
                } else {
                    let mut initialized = 0;
                    while initialized < SHARDS {
                        if loads[initialized] != 0 {
                            if let Some(first) = parts.first(initialized) {
                                let face = kvs[first].0 .0;
                                let sigma = face.sigma();
                                let address = Home::address(&t.home, sigma, &kvs[first].0 .1);
                                let held = R::hold(sigma, &kvs[first].0 .1, &address);
                                t.shards[initialized].initialize(sigma, &held, &t.default);
                            }
                        }
                        initialized += 1;
                    }
                    // The LPT schedule orders shard work only. Runs remain in worker/chunk order.
                    let mut assembly_error = None;
                    let mut worker = 0;
                    while worker < schedule.len() && assembly_error.is_none() {
                        let bin = schedule.bin(worker);
                        let mut bin_at = 0;
                        while bin_at < bin.len() && assembly_error.is_none() {
                            let shard = bin[bin_at];
                            let mut chunk = 0;
                            while chunk < parts.len() && assembly_error.is_none() {
                                let run = parts.run(chunk, shard);
                                let mut run_at = 0;
                                while run_at < run.len() && assembly_error.is_none() {
                                    let ((face, key), value) = &kvs[run[run_at]];
                                    match t.insert_with_at(merge, *face, key.clone(), value.clone())
                                    {
                                        Ok(()) => {}
                                        Err(error) => assembly_error = Some(error),
                                    }
                                    run_at += 1;
                                }
                                chunk += 1;
                            }
                            bin_at += 1;
                        }
                        worker += 1;
                    }
                    match assembly_error {
                        Some(error) => Err(error),
                        None => Ok(t),
                    }
                }
            }
        }
    }
    fn build_at_sequential<F: Merge<V>>(
        merge: &F,
        default: V,
        kvs: &[((H::Faces, K), V)],
        home: H,
    ) -> Result<Self, StorageError> {
        let mut t = Self::new(default, home);
        for ((face, k), v) in kvs {
            t.insert_with_at(merge, *face, k.clone(), v.clone())?;
        }
        Ok(t)
    }
    /// Folds a default-face stream in input order.
    ///
    /// Every entry takes `<H::Faces as FaceDomain<K,H>>::DEFAULT`; the qualification carries the
    /// token in the tuple, not a second key clone.
    ///
    /// # Errors
    /// Returns the first checked sizing or reservation error and returns no partial table.
    pub fn build<F: Merge<V>>(
        merge: &F,
        default: V,
        kvs: &[(K, V)],
        home: H,
    ) -> Result<Self, StorageError> {
        if kvs.len() < BUILD_PARTITION_CROSSOVER {
            let mut t = Self::new(default, home);
            for (k, v) in kvs {
                t.insert_with(merge, k.clone(), v.clone())?;
            }
            return Ok(t);
        }
        let mut qualified = Vec::new();
        let mut i = 0;
        while i < kvs.len() {
            let (key, value) = &kvs[i];
            qualified.push((
                (<H::Faces as FaceDomain<K, H>>::DEFAULT, key.clone()),
                value.clone(),
            ));
            i += 1;
        }
        Self::build_at(merge, default, &qualified, home)
    }
}

#[cfg(test)]
#[path = "table/tests.rs"]
mod tests;
