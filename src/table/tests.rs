//! Unit tests for the growable table, factored out of `src/table.rs` without behavior change.

use super::keyrep::{I64WordCodec, Regen};
use super::{
    build_prepared_shard, capacity_bound_holds, keyrep::KeyRep, publication_trace, read_trace,
    shard, slots_for, spill_locate, spill_read, tag, ApplyOutcome, Flat, FlatSlot, Home,
    PreparedResult, RebuildPlan, RebuildReserved, Region, ReplacementMerge, SearchClass,
    SearchDecision, SearchLocation, SpillSite, StorageError, Stored, Table, TableAddress, WordHome,
    SHARDS,
};
use crate::cell::Sigma;
use crate::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, LocatorSigma, ParabolicDual,
    Residue, H3_40,
};
use crate::coord::locator::{ExactIdentity, Refusal};
use crate::coord::{spread, Packed};
use std::cell::Cell;
use std::rc::Rc;

/// The all-face semantic capability for a word key: `Faces = Sigma`, with the exact `Packed` word.
///
/// The map is derived from the key, never from a home's physical placement, so a collision-only
/// `Home` cannot leak into the face encoding.
macro_rules! sigma_word_face {
    ($home:ty, $key:ty) => {
        impl FaceResidueCapability<$key> for $home {
            fn parabolic_dual(&self, key: &$key) -> ParabolicDual {
                ParabolicDual::new(spread(<$key as Packed>::packed(key)).place, 0)
            }

            fn hyperbolic_scalar(&self, profile: H3_40, key: &$key) -> Residue {
                Residue::reduce(profile, <$key as Packed>::packed(key))
            }
        }

        impl EllipticPacking<$key> for $home {
            fn elliptic_pack(&self, key: &$key) -> u64 {
                <$key as Packed>::packed(key)
            }

            fn elliptic_unpack(&self, word: u64) -> Option<$key> {
                Some(<$key as Packed>::unpack(word))
            }
        }

        impl FaceCoord<$key> for $home {
            type Faces = Sigma;
        }

        impl FaceHome<$key> for $home {}
    };
}

/// The locator-only semantic capability for a key with exact bytes but no word packing.
macro_rules! locator_face {
    ($home:ty, $key:ty, $word:path) => {
        impl FaceResidueCapability<$key> for $home {
            fn parabolic_dual(&self, key: &$key) -> ParabolicDual {
                ParabolicDual::new(spread($word(key)).place, 0)
            }

            fn hyperbolic_scalar(&self, profile: H3_40, key: &$key) -> Residue {
                Residue::reduce(profile, $word(key))
            }
        }

        impl FaceCoord<$key> for $home {
            type Faces = LocatorSigma;
        }

        impl FaceHome<$key> for $home {}
    };
}

/// A selected reservation failure is mapped to the transactional allocation error.
#[test]
fn selected_resident_reservation_failure_leaves_table_unchanged() {
    assert!(Flat::<i64, i64>::reserve(usize::MAX / 256).is_err());
    let mut table = Table::<i64, i64, _, 16>::new(0, WordHome);
    assert_eq!(table.insert_with(&ReplacementMerge, 1, 2), Ok(()));
    assert_eq!(*table.get(&1), 2);
}

/// A home with certified direct words for the clone-counting key, so the density fixture can
/// assert zero clones on the refusal path.
#[derive(Clone, Copy)]
struct CountWordHome;

impl Home<Counted> for CountWordHome {
    fn home(&self, key: &Counted) -> u64 {
        key.value.cast_unsigned()
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::checked_conversions,
        reason = "the preceding explicit bound proves the u64 value fits u32"
    )]
    fn address(&self, _sigma: Sigma, key: &Counted) -> TableAddress {
        let word = key.value.cast_unsigned();
        TableAddress {
            place: word,
            direct_word: if word <= u64::from(u32::MAX) {
                Some(word as u32)
            } else {
                None
            },
        }
    }
}

// The semantic map is derived from `Counted::value`, never from the physical `home`/`address`
// route, so the collision-only placement cannot leak into a face encoding.
locator_face!(CountWordHome, Counted, counted_word);

/// The flat density planner is fallible and part of the reserve-all path: an oversized bitmap
/// sizing or a refused reservation returns `Err` with the committed table unchanged and no
/// hold, clone, merge or relocation. The admitted path returns the exact count and the extent
/// decision on a fixed fixture is the one the former infallible planner selected.
#[test]
fn density_plan_refusal_is_transactional_and_the_success_extent_is_unchanged() {
    let sigma = Sigma::Parabolic;
    let clones = Rc::new(Cell::new(0usize));
    let mk = |value: i64| Counted {
        value,
        clones: Rc::clone(&clones),
    };

    // Fixed fixture: 1000 live resident direct words across four lanes.
    let mut resident = Flat::<Stored<Counted>, Counted>::reserve(4).expect("four lanes");
    let seed = <Stored<Counted> as KeyRep<Counted, CountWordHome>>::hold(
        sigma,
        &mk(0),
        &Home::address(&CountWordHome, sigma, &mk(0)),
    );
    resident.initialize(sigma, &seed, &mk(-1));
    let mut key = 0i64;
    while key < 1000 {
        let held = <Stored<Counted> as KeyRep<Counted, CountWordHome>>::hold(
            sigma,
            &mk(key),
            &Home::address(&CountWordHome, sigma, &mk(key)),
        );
        let word = Home::address(&CountWordHome, sigma, &mk(key))
            .direct_word
            .expect("word key");
        let index = resident.index(word).expect("admitted lane");
        resident.slots[index] = FlatSlot {
            live: true,
            sigma,
            held,
            value: mk(key),
        };
        key += 1;
    }
    let mut table = Table::<Counted, Counted, CountWordHome, 16>::new(mk(-1), CountWordHome);
    table.resident = resident;

    // The admitted bitmap returns the exact distinct count and the extent is unchanged.
    assert_eq!(
        table.density_bitmap(4, Some(1000)).map(|(_, n)| n),
        Ok(1001)
    );
    assert_eq!(table.selected_lanes(0, 1000), Ok(Some(4)));

    // Snapshot the committed state; a refusal must not touch it.
    let before_len = table.len();
    let before_lanes = table.resident.lanes;
    let before_resident_len = table.resident.slots.len();
    let before_tags: Vec<u8> = table.shards[0].markers();
    let before_slots = table.shards[0].slots.len();
    clones.set(0);

    // A bitmap whose checked size overflows `usize` refuses before any allocation attempt.
    assert_eq!(
        table.density_bitmap(usize::MAX, Some(0)).map(|(_, n)| n),
        Err(StorageError::SizeOverflow)
    );
    // A bitmap sizing that fits but whose reservation the allocator refuses is a refusal too.
    assert_eq!(
        table
            .density_bitmap(usize::MAX / 256, Some(0))
            .map(|(_, n)| n),
        Err(StorageError::AllocationFailed)
    );

    assert_eq!(table.len(), before_len);
    assert_eq!(table.resident.lanes, before_lanes);
    assert_eq!(table.resident.slots.len(), before_resident_len);
    assert_eq!(table.shards[0].markers(), before_tags);
    assert_eq!(table.shards[0].slots.len(), before_slots);
    assert_eq!(clones.get(), 0, "a density refusal must not clone");
}

/// The bounded linear cycle finds a match stored past every same-fingerprint reject.
#[test]
fn spill_cycle_finds_a_match_past_same_fingerprint_rejects() {
    let address = TableAddress {
        place: 0,
        direct_word: None,
    };
    let held = <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &7, &address);
    let mut region = Region::reserve_empty(32).expect("one flat region");
    region.initialize(Sigma::Parabolic, &held, &0);
    // Placement 0 homes every query at slot 0, and linear probing places each insert at the first
    // vacancy, so slots 0..=15 must all be occupied before slot 16 can hold the matching key.
    let mut filler: i64 = 100;
    while filler < 116 {
        let index =
            usize::try_from(filler - 100).expect("the filler range is 0..16 and fits usize");
        region.publish_live(
            index,
            (
                Sigma::Parabolic,
                <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &filler, &address),
                41,
            ),
            address.place,
        );
        filler += 1;
    }
    region.publish_live(16, (Sigma::Parabolic, held, 41), address.place);

    assert_eq!(
        spill_locate::<i64, WordHome, Stored<i64>, i64>(
            &region,
            Sigma::Parabolic,
            &address,
            &7,
            16,
        ),
        SpillSite::Found(16)
    );
}

/// A spill vacancy selects the first free slot at or after the query's home.
#[test]
fn spill_decision_selects_the_first_vacant_slot() {
    let address = TableAddress {
        place: 0,
        direct_word: None,
    };
    let held = <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &8, &address);
    let mut region = Region::reserve_empty(32).expect("one flat region");
    region.initialize(Sigma::Parabolic, &held, &0);
    region.publish_live(0, (Sigma::Parabolic, held, 41), address.place);

    assert_eq!(
        spill_locate::<i64, WordHome, Stored<i64>, i64>(
            &region,
            Sigma::Parabolic,
            &address,
            &7,
            16,
        ),
        SpillSite::Vacant(1),
        "slot 0 is occupied by a different key, so the first vacancy is slot 1"
    );
}

/// An exhausted decision returns the prepared growth outcome without allocating or committing.
#[test]
fn exhausted_apply_returns_the_prepared_growth_outcome_without_allocating() {
    let mut table = Table::<i64, i64, _, 16>::new(0, WordHome);
    let address = Home::address(&WordHome, Sigma::Parabolic, &7);
    let decision = SearchDecision {
        class: SearchClass::Exhausted,
        location: SearchLocation::NoIndex,
    };
    let outcome = table.apply_insert(
        decision,
        &ReplacementMerge,
        Sigma::Parabolic,
        7,
        41,
        &address,
    );
    assert_eq!(outcome, ApplyOutcome::NeedsGrowth(7, 41));
    assert_eq!(table.len(), 0, "exhaustion commits nothing");
    assert!(
        table.resident.slots.is_empty(),
        "the apply must not allocate resident slots"
    );
    assert!(
        table.shards[shard(&address)].slots.is_empty(),
        "the apply must not allocate a spill region"
    );
}

/// The straight-line apply commits a flat first-fit vacancy, then updates that flat slot.
#[test]
fn apply_insert_commits_a_flat_vacancy_then_updates_a_flat_found_slot() {
    let mut table = Table::<i64, i64, _, 16>::new(-1, WordHome);
    // The first write selects the resident extent; the second key's direct lane is then vacant.
    table.insert_with(&ReplacementMerge, 0, 10).unwrap();
    let address = Home::address(&WordHome, Sigma::Parabolic, &1);
    let vacant = table.search_decision_at(Sigma::Parabolic, &1);
    assert_eq!(vacant.class, SearchClass::Vacant);
    assert!(matches!(vacant.location, SearchLocation::Flat(_)));
    assert_eq!(
        table.apply_insert(vacant, &ReplacementMerge, Sigma::Parabolic, 1, 11, &address),
        ApplyOutcome::Committed
    );
    assert_eq!(table.len(), 2);
    let found = table.search_decision_at(Sigma::Parabolic, &1);
    assert_eq!(found.class, SearchClass::Found);
    assert_eq!(
        table.apply_insert(found, &ReplacementMerge, Sigma::Parabolic, 1, 99, &address),
        ApplyOutcome::Committed
    );
    assert_eq!(
        table.len(),
        2,
        "an update does not change the logical length"
    );
    assert_eq!(
        *table.apply_get(table.search_decision_at(Sigma::Parabolic, &1)),
        99
    );
}

/// The straight-line apply updates a found spill slot on the placement-only route.
#[test]
fn apply_insert_updates_a_found_spill_slot() {
    let home = WordKeyHome;
    let address = Home::address(&home, Sigma::Parabolic, &7);
    let mut table = Table::<i64, i64, _, 16>::new(0, home);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    let found = table.search_decision_at(Sigma::Parabolic, &7);
    assert_eq!(found.class, SearchClass::Found);
    assert!(matches!(found.location, SearchLocation::Spill(_, _)));
    assert_eq!(
        table.apply_insert(found, &ReplacementMerge, Sigma::Parabolic, 7, 99, &address),
        ApplyOutcome::Committed
    );
    assert_eq!(table.len(), 1);
    assert_eq!(*table.get(&7), 99);
}

/// The commit body consumes the supplied classification: the same reserved flat slot receives a
/// fresh insert under a precomputed `Vacant` decision and a merge under a precomputed `Found`
/// decision, so the body never re-probes and never reclassifies.
#[test]
fn apply_pending_and_commit_consumes_the_supplied_decision() {
    let sigma = Sigma::Parabolic;
    let key = 5_i64;
    let address = Home::address(&WordHome, sigma, &key);
    let held = <Stored<i64> as KeyRep<i64, WordHome>>::hold(sigma, &key, &address);

    let reserved_for = |slot: Option<i64>| {
        let mut resident = Flat::<Stored<i64>, i64>::reserve(1).expect("one lane");
        resident.initialize(sigma, &held, &-1);
        if let Some(value) = slot {
            resident.slots[5] = FlatSlot {
                live: true,
                sigma,
                held: held.clone(),
                value,
            };
        }
        let regions: Vec<Region<Stored<i64>, i64>> = (0..SHARDS).map(|_| Region::empty()).collect();
        RebuildReserved { resident, regions }
    };

    // A precomputed vacancy writes the pending value and grows the logical length once.
    let mut vacant = Table::<i64, i64, _, 16>::new(-1, WordHome);
    vacant.apply_pending_and_commit(
        reserved_for(None),
        SearchDecision::vacant_flat(5),
        &ReplacementMerge,
        sigma,
        41,
        held.clone(),
        &address,
    );
    assert_eq!(vacant.len(), 1, "a vacant commit inserts one logical key");
    assert_eq!(*vacant.get_at(sigma, &key), 41);

    // A precomputed found decision merges into the existing slot without changing the length.
    let mut found = Table::<i64, i64, _, 16>::new(-1, WordHome);
    found.apply_pending_and_commit(
        reserved_for(Some(10)),
        SearchDecision::found_flat(5),
        &ReplacementMerge,
        sigma,
        7,
        held.clone(),
        &address,
    );
    assert_eq!(found.len(), 0, "a found commit updates no logical length");
    assert_eq!(*found.get_at(sigma, &key), 7);
}

/// An exact-identity key whose encoding is `len` repetitions of one byte.
///
/// The per-instance encoding length is the instrument: one table can stage a long path and then a
/// short one, so the staged-scratch observations below separate a recycled buffer (same address,
/// retained capacity covering the long path) from a fresh allocation sized for the short path.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Repeated {
    byte: u8,
    len: u8,
}

impl ExactIdentity for Repeated {
    type Retained = (u8, u8);
    type Reserved = ();

    fn retain(&self) -> (u8, u8) {
        (self.byte, self.len)
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> (u8, u8) {
        self.retain()
    }

    fn matches_retained(&self, retained: &(u8, u8)) -> bool {
        self.retain() == *retained
    }

    fn encoding_len(&self) -> usize {
        usize::from(self.len)
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        assert!(
            index < usize::from(self.len),
            "byte index within the encoding"
        );
        self.byte
    }

    fn retained_encoding_len(retained: &(u8, u8)) -> usize {
        usize::from(retained.1)
    }

    fn retained_encoding_byte(retained: &(u8, u8), index: usize) -> u8 {
        assert!(
            index < usize::from(retained.1),
            "byte index within the encoding"
        );
        retained.0
    }
}

/// The packed placement word of a [`Repeated`] key: its length and byte.
fn repeated_word(key: &Repeated) -> u64 {
    (u64::from(key.len) << 8) | u64::from(key.byte)
}

/// A placement-only home for [`Repeated`] keys on the locator faces.
#[derive(Clone, Copy)]
struct RepeatedHome;

impl Home<Repeated> for RepeatedHome {
    fn home(&self, key: &Repeated) -> u64 {
        repeated_word(key)
    }

    fn address(&self, _sigma: Sigma, key: &Repeated) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

locator_face!(RepeatedHome, Repeated, repeated_word);

/// The address and capacity of the outstanding staged path's identity-path scratch buffer.
fn scratch_marker<K, V, H, const N: usize, R>(table: &Table<K, V, H, N, R>) -> (*const u32, usize)
where
    R: KeyRep<K, H>,
{
    let Some((_, staged)) = &table.locator.staged else {
        panic!("a staged path is outstanding");
    };
    (
        staged.identity_path.as_ptr(),
        staged.identity_path.capacity(),
    )
}

/// The flat found-update commit recycles the staged scratch buffer instead of dropping it.
///
/// The pending key's staged path is unused on a found update, but its identity-path buffer returns
/// to the directory's idle slot through `Directory::recycle_staged`, so the next reservation hands
/// out the same allocation: same address, and capacity still covering the long path. Dropping the
/// buffer frees it, and the shorter reservation would then show a fresh, smaller allocation. The
/// checked receipt observes the same capacity through its `staging` bound.
#[test]
fn flat_found_update_recycles_the_staged_scratch() {
    let sigma = Sigma::Parabolic;
    let home = RepeatedHome;
    let long_key = Repeated {
        byte: 0xA5,
        len: 64,
    };
    let short_key = Repeated { byte: 0x5A, len: 1 };
    let mut table = Table::<Repeated, i64, RepeatedHome, 16>::new(-1, home);
    let address = Home::address(&home, sigma, &long_key);
    let held =
        <Stored<Repeated> as KeyRep<Repeated, RepeatedHome>>::hold(sigma, &long_key, &address);

    // Stage the pending key's path exactly as the rebuild reservation does.
    let receipt = table
        .reserve_locator_path(sigma, &long_key)
        .expect("the long path reserves");
    let first = scratch_marker(&table);
    assert_eq!(
        first.1, receipt.staging.observed_capacity,
        "the checked receipt observes the scratch capacity the staged path holds"
    );
    assert!(
        first.1 > long_key.encoding_len(),
        "the scratch is sized for the long path: {} words",
        first.1
    );

    // The rebuilt storage the commit consumes: one found flat slot holding the pending key.
    let mut resident = Flat::<Stored<Repeated>, i64>::reserve(1).expect("one lane");
    resident.initialize(sigma, &held, &-1);
    resident.slots[5] = FlatSlot {
        live: true,
        sigma,
        held: held.clone(),
        value: 10,
    };
    let regions: Vec<Region<Stored<Repeated>, i64>> =
        (0..SHARDS).map(|_| Region::empty()).collect();
    table.apply_pending_and_commit(
        RebuildReserved { resident, regions },
        SearchDecision::found_flat(5),
        &ReplacementMerge,
        sigma,
        99,
        held,
        &address,
    );
    assert_eq!(table.len(), 0, "a found update changes no logical length");
    assert_eq!(
        table.resident.slots[5].value, 99,
        "the found update merges into its existing slot"
    );

    // The next reservation takes the recycled buffer: same address and retained capacity, so the
    // short path allocates no fresh scratch.
    table
        .reserve_locator_path(sigma, &short_key)
        .expect("the short path reserves");
    let second = scratch_marker(&table);
    assert_eq!(
        second, first,
        "the found-update apply must recycle the staged scratch: the next reservation takes the \
         same buffer at the same address with its retained capacity, not a fresh allocation sized \
         for the short path"
    );
}

/// The spill found-update commit recycles the staged scratch buffer instead of dropping it.
///
/// The same observation as the flat arm, on the spill arm of the commit body: the unused staged
/// path's buffer returns to the idle slot, so the next reservation reuses it without allocating.
#[test]
fn spill_found_update_recycles_the_staged_scratch() {
    let sigma = Sigma::Parabolic;
    let home = RepeatedHome;
    let long_key = Repeated {
        byte: 0xC3,
        len: 64,
    };
    let short_key = Repeated { byte: 0x3C, len: 1 };
    let mut table = Table::<Repeated, i64, RepeatedHome, 16>::new(-1, home);
    let address = Home::address(&home, sigma, &long_key);
    let held =
        <Stored<Repeated> as KeyRep<Repeated, RepeatedHome>>::hold(sigma, &long_key, &address);

    table
        .reserve_locator_path(sigma, &long_key)
        .expect("the long path reserves");
    let first = scratch_marker(&table);

    // The rebuilt storage the commit consumes: one found spill slot holding the pending key.
    let resident = Flat::<Stored<Repeated>, i64>::reserve(1).expect("one lane");
    let mut region =
        Region::<Stored<Repeated>, i64>::reserve_empty(32).expect("two initialized groups");
    region.initialize(sigma, &held, &-1);
    region.publish_live(3, (sigma, held.clone(), 10), address.place);
    let mut regions: Vec<Region<Stored<Repeated>, i64>> =
        (0..SHARDS).map(|_| Region::empty()).collect();
    regions[0] = region;
    table.apply_pending_and_commit(
        RebuildReserved { resident, regions },
        SearchDecision {
            class: SearchClass::Found,
            location: SearchLocation::Spill(0, 3),
        },
        &ReplacementMerge,
        sigma,
        99,
        held,
        &address,
    );
    assert_eq!(table.len(), 0, "a found update changes no logical length");
    assert_eq!(
        table.shards[0].slots[3].value, 99,
        "the found update merges into its existing slot"
    );

    table
        .reserve_locator_path(sigma, &short_key)
        .expect("the short path reserves");
    let second = scratch_marker(&table);
    assert_eq!(
        second, first,
        "the spill found-update apply must recycle the staged scratch: the next reservation takes \
         the same buffer at the same address with its retained capacity, not a fresh allocation \
         sized for the short path"
    );
}

/// A word value that counts every clone, so a refusal path can be shown to clone nothing.
///
/// `K` and `V` share one counter through the `Rc`, so a `Stored::hold` key clone, a payload
/// clone or an initialization clone is observable as one increment.
#[derive(Debug)]
struct Counted {
    value: i64,
    clones: Rc<Cell<usize>>,
}

impl Clone for Counted {
    fn clone(&self) -> Self {
        self.clones.set(self.clones.get() + 1);
        Self {
            value: self.value,
            clones: Rc::clone(&self.clones),
        }
    }
}

impl PartialEq for Counted {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for Counted {}

/// The exact identity word of a [`Counted`] key: its `value`, never its clone counter.
fn counted_word(key: &Counted) -> u64 {
    key.value.cast_unsigned()
}

/// The frozen `H3_40` token the semantic face maps reduce against.
fn profile() -> H3_40 {
    H3_40::new().expect("the frozen H3_40 constants validate")
}

/// The exact-identity bytes of a key, collected for a direct byte-for-byte comparison.
fn identity_bytes<K: ExactIdentity>(key: &K) -> Vec<u8> {
    (0..key.encoding_len())
        .map(|index| key.encoding_byte(index))
        .collect()
}

/// The exact identity of [`Counted`] is its `value`; the clone counter is not equality and so is
/// not identity. The retained word is the two's-complement bit pattern of `value`.
impl ExactIdentity for Counted {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        counted_word(self)
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        counted_word(self)
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        counted_word(self) == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.value.cast_unsigned().to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        retained.to_be_bytes()[index]
    }
}

/// `Counted`'s exact identity is its `value` field, never its clone counter.
///
/// Two distinct equality keys must not share an encoding, and two equal keys with different
/// counters must share one. The rejecting control encodes by the clone counter instead; the two
/// distinct keys below then share a counter and the `assert_ne!` rejects before any locator runs.
#[test]
fn counted_identity_is_the_value_not_the_clone_counter() {
    let shared = Rc::new(Cell::new(0usize));
    let other = Rc::new(Cell::new(0usize));
    let seven = Counted {
        value: 7,
        clones: Rc::clone(&shared),
    };
    let eight = Counted {
        value: 8,
        clones: Rc::clone(&shared),
    };
    let seven_again = Counted {
        value: 7,
        clones: Rc::clone(&other),
    };

    assert_ne!(seven, eight, "distinct values are distinct keys");
    assert_ne!(
        identity_bytes(&seven),
        identity_bytes(&eight),
        "distinct equality keys must not share an exact encoding"
    );
    assert!(!seven.matches_retained(&eight.retain()));

    assert_eq!(seven, seven_again, "the clone counter is not equality");
    assert_eq!(
        identity_bytes(&seven),
        identity_bytes(&seven_again),
        "the clone counter must not enter the exact encoding"
    );
    assert!(seven.matches_retained(&seven_again.retain()));
}

/// The `Counted` face map separates keys that the collision-only physical route collapses.
#[test]
fn counted_semantic_map_is_independent_of_the_collision_only_route() {
    let p = profile();
    let seven = Counted {
        value: 7,
        clones: Rc::new(Cell::new(0usize)),
    };
    let eight = Counted {
        value: 8,
        clones: Rc::new(Cell::new(0usize)),
    };
    // The physical route collapses both keys onto one placement and one shard.
    assert_eq!(CountHome.home(&seven), CountHome.home(&eight));

    // The semantic map separates them on both filtered faces.
    let parabolic_seven = FaceCoord::coord(&CountHome, p, LocatorSigma::Parabolic, &seven);
    let parabolic_eight = FaceCoord::coord(&CountHome, p, LocatorSigma::Parabolic, &eight);
    assert_ne!(parabolic_seven, parabolic_eight);
    let hyperbolic_seven = FaceCoord::coord(&CountHome, p, LocatorSigma::Hyperbolic, &seven);
    let hyperbolic_eight = FaceCoord::coord(&CountHome, p, LocatorSigma::Hyperbolic, &eight);
    assert_ne!(hyperbolic_seven, hyperbolic_eight);
}

/// A placement-only home for [`Counted`], so a pending key routes to spill.
#[derive(Clone, Copy)]
struct CountHome;

impl Home<Counted> for CountHome {
    fn home(&self, _key: &Counted) -> u64 {
        0
    }

    /// Placement-only: reproduces the former default (`home`-derived place, no direct word).
    fn address(&self, _sigma: Sigma, key: &Counted) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

// The semantic map is derived from `Counted::value`, independent of the constant physical route.
locator_face!(CountHome, Counted, counted_word);

/// A named placement-only word home, so a test can call `address_held` at a concrete `H`.
#[derive(Clone, Copy)]
struct WordKeyHome;

impl Home<i64> for WordKeyHome {
    fn home(&self, key: &i64) -> u64 {
        key.cast_unsigned()
    }

    /// Placement-only: reproduces the former default (`home`-derived place, no direct word).
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

sigma_word_face!(WordKeyHome, i64);

/// A refusal at the final reservation step exposes no storage and runs no clone, hold, merge or
/// relocation.
///
/// The synthetic plan routes an overflowing checked target to the last shard, so the flat,
/// directory and first 255 region reservations succeed before the final step fails.  The plan is
/// the adapter's real input shape; `prepare_from_plan` is the real adapter minus planning.
#[test]
fn final_reservation_refusal_leaves_the_table_unchanged_with_no_clone_or_merge() {
    let clones = Rc::new(Cell::new(0usize));
    let merge_calls = Rc::new(Cell::new(0usize));
    let mk = |value: i64| Counted {
        value,
        clones: Rc::clone(&clones),
    };
    let merge = {
        let calls = Rc::clone(&merge_calls);
        move |new: Counted, _old: Counted| {
            calls.set(calls.get() + 1);
            new
        }
    };
    let mut table = Table::<Counted, Counted, CountHome, 16>::new(mk(-1), CountHome);
    table.insert_with(&merge, mk(7), mk(41)).unwrap();
    table.insert_with(&merge, mk(8), mk(42)).unwrap();
    assert_eq!(table.len(), 2);

    // Snapshot the committed state and reset the counters before the refusal.
    let before_len = table.len();
    let before_resident_lanes = table.resident.lanes;
    let before_resident_len = table.resident.slots.len();
    let before_tags: Vec<u8> = table.shards[0].markers();
    let before_slots = table.shards[0].slots.len();
    clones.set(0);
    merge_calls.set(0);

    // Inject the failure at the final reservation step: shard 255's checked target overflows.
    let mut spill_loads = vec![0usize; SHARDS];
    spill_loads[SHARDS - 1] = usize::MAX;
    let plan = RebuildPlan {
        lanes: 1,
        entries: Vec::new(),
        spill_loads,
    };
    let pending = mk(9);
    let address = Home::address(&CountHome, Sigma::Parabolic, &pending);
    let refused = table.prepare_from_plan(plan, Sigma::Parabolic, &pending, &address);
    assert!(matches!(
        refused,
        PreparedResult::Refused(StorageError::SizeOverflow)
    ));

    let outcome =
        table.apply_prepared_rebuild(refused, &merge, Sigma::Parabolic, pending, mk(99), &address);
    assert_eq!(outcome, Err(StorageError::SizeOverflow));

    // Byte-for-observation unchanged: length, values and both storage vectors.
    assert_eq!(table.len(), before_len);
    assert_eq!(table.resident.lanes, before_resident_lanes);
    assert_eq!(table.resident.slots.len(), before_resident_len);
    assert_eq!(table.shards[0].markers(), before_tags);
    assert_eq!(table.shards[0].slots.len(), before_slots);
    assert_eq!(table.get_at(LocatorSigma::Parabolic, &mk(7)).value, 41);
    assert_eq!(table.get_at(LocatorSigma::Parabolic, &mk(8)).value, 42);
    assert_eq!(table.get_at(LocatorSigma::Parabolic, &mk(9)).value, -1);

    // No hold, clone, merge or relocation ran before the refusal.
    assert_eq!(
        clones.get(),
        0,
        "a refusal must not clone a held carrier or payload"
    );
    assert_eq!(merge_calls.get(), 0, "a refusal must not merge");
}

/// The prepared adapter materializes vacant storage and carries the pending held carrier.
#[test]
fn prepared_adapter_materializes_vacant_storage_and_carries_the_pending_held() {
    let clones = Rc::new(Cell::new(0usize));
    let mk = |value: i64| Counted {
        value,
        clones: Rc::clone(&clones),
    };
    let mut table = Table::<Counted, Counted, CountHome, 16>::new(mk(-1), CountHome);
    let sigma = Sigma::Parabolic;
    let key = mk(7);
    let address = Home::address(&CountHome, sigma, &key);

    let prepared = table.prepare_rebuild(1, sigma, &key, &address);
    let PreparedResult::Prepared {
        plan,
        storage,
        outcome,
    } = prepared
    else {
        panic!("the fresh adapter must prepare the rebuild");
    };

    // The placement-only pending key routes one unit of load to shard 0.
    assert_eq!(plan.lanes, 1);
    assert_eq!(plan.spill_loads[0], 1);

    // Materialized vacant storage: flat length is its target, every live bit is false.
    assert_eq!(storage.resident.lanes, 1);
    assert_eq!(storage.resident.slots.len(), 256);
    assert!(storage.resident.slots.iter().all(|slot| !slot.live));
    assert_eq!(storage.regions.len(), SHARDS);
    let target = slots_for(1, 16).expect("checked region target");
    assert_eq!(storage.regions[0].slots.len(), target);
    assert!(storage.regions[0]
        .markers()
        .iter()
        .all(|&tag_at| tag_at == 0));
    assert_eq!(storage.regions[0].slots[0].sigma, sigma);

    // The receipt records the checked targets and the observed capacities meet them.
    let receipt = &outcome.reservation;
    assert_eq!(receipt.flat.required_len, 256);
    assert_eq!(receipt.directory.required_len, SHARDS);
    assert_eq!(receipt.regions[0].slots.required_len, target);
    assert!(capacity_bound_holds(
        receipt.flat.required_len,
        receipt.flat.observed_capacity
    ));
    assert!(capacity_bound_holds(
        receipt.directory.required_len,
        receipt.directory.observed_capacity
    ));
    assert!(capacity_bound_holds(
        receipt.regions[0].slots.required_len,
        receipt.regions[0].slots.observed_capacity
    ));

    // The carried pending carrier is the pending key's held form.
    assert!(<Stored<Counted> as KeyRep<Counted, CountHome>>::matches(
        sigma,
        &outcome.pending_held,
        &key,
        &address
    ));
    assert!(
        clones.get() > 0,
        "materialization must clone the held carrier and default"
    );
}

/// `build_prepared_shard` reserves its complete checked region before the first hold, so a
/// refusal runs no clone. This is the per-shard ordering a parallel worker uses.
///
/// The flat region has two refusals, and each is reached with the checked capacity rule of
/// [`slots_for`] as the only sizing authority: the doubling `checked_mul` refuses a load that
/// cannot be doubled, and the reservation refuses a checked target that cannot be obtained.
/// `1 << 63` is the largest load whose doubling overflows `usize`; `1 << 58` is a load the rule
/// *accepts* (`Ok(1 << 59)`), whose refusal therefore comes from the reservation itself.
#[test]
fn prepared_shard_reserves_before_hold_on_sizing_refusal() {
    let clones = Rc::new(Cell::new(0usize));
    let mk = |value: i64| Counted {
        value,
        clones: Rc::clone(&clones),
    };
    let merge = |new: Counted, _old: Counted| new;
    let input = vec![((LocatorSigma::Parabolic, mk(7)), mk(41))];
    let parts = super::partition::Parts::of(&CountHome, &input, 1);
    let run = |load: usize, clones: &Rc<Cell<usize>>| {
        clones.set(0);
        build_prepared_shard::<Counted, Counted, CountHome, Stored<Counted>, _, 16>(
            &CountHome,
            &merge,
            &mk(-1),
            &input,
            &parts,
            0,
            load,
        )
    };

    // The checked capacity rule refuses the undoubleable load.
    assert_eq!(slots_for(1usize << 63, 16), Err(StorageError::SizeOverflow));
    assert!(matches!(
        run(1usize << 63, &clones),
        Err(StorageError::SizeOverflow)
    ));
    assert_eq!(clones.get(), 0, "reservation must precede the first hold");

    // The rule accepts this target and the reservation refuses it; the refusal is still before the
    // first hold, so no clone and no callback runs on either arm.
    assert_eq!(slots_for(1usize << 58, 16), Ok(1usize << 59));
    assert!(matches!(
        run(1usize << 58, &clones),
        Err(StorageError::AllocationFailed)
    ));
    assert_eq!(clones.get(), 0, "a refused reservation runs no clone");
}

/// A real prepared-shard build whose load the capacity rule accepts is not a refusal path, so the
/// fixture above cannot pass vacuously: the same call at a real load reserves and initializes the
/// checked region it asked for.
#[test]
fn prepared_shard_reserves_the_checked_flat_capacity_at_a_real_load() {
    let clones = Rc::new(Cell::new(0usize));
    let mk = |value: i64| Counted {
        value,
        clones: Rc::clone(&clones),
    };
    let merge = |new: Counted, _old: Counted| new;
    let input = vec![((LocatorSigma::Parabolic, mk(7)), mk(41))];
    let parts = super::partition::Parts::of(&CountHome, &input, 1);
    let region = build_prepared_shard::<Counted, Counted, CountHome, Stored<Counted>, _, 16>(
        &CountHome,
        &merge,
        &mk(-1),
        &input,
        &parts,
        0,
        1,
    )
    .expect("the checked capacity of one live entry is reservable");
    assert_eq!(
        region.slots.len(),
        32,
        "the rule reserves the least power of two at or above max(2 * n, 2 * load)"
    );
    assert_eq!(region.live, 1, "one live entry is published");
    assert!(
        clones.get() > 0,
        "a successful reservation initializes every slot"
    );
}

/// The whole-table read path certifies absence on a completed spill cycle, not only on a zero tag.
///
/// The capacity rule of [`slots_for`] reserves at least twice the load and at least twice the group
/// width, so a region the insert path builds itself keeps a vacancy at every load up to `n` and the
/// completed-cycle certificate is unreachable through `insert_with` there.  Above `n` the region
/// grows only when a cycle finds it full, so the certificate becomes reachable once its load passes
/// `n`.  The fixture therefore fills one shard's region to its exact capacity through the
/// production locator and reads through `Table::get`, and attributes which certificate decided each
/// read: the absent key costs exactly one cycle of `capacity` steps with no zero-tag certificate,
/// the stored key costs fewer, so the two certificates are not conflated.
#[test]
fn table_get_certifies_absence_on_a_completed_spill_cycle() {
    let mut table = Table::<i64, i64, WordKeyHome, 16>::new(0, WordKeyHome);
    table.insert_with(&ReplacementMerge, 0, 1).unwrap();
    let capacity = table.shards[0].slots.len();
    assert!(
        capacity >= 8,
        "the checked capacity of one live entry is a power of two"
    );

    // Fill shard zero to its exact capacity. Every key below has shard byte zero, so each is a real
    // `WordKeyHome` key for shard zero and each is placed where the production locator selects.
    let mut stored = vec![0i64];
    for index in 1..capacity {
        let key = 256 * i64::try_from(index).expect("the index fits i64");
        let address = Home::address(&WordKeyHome, Sigma::Parabolic, &key);
        let slot = match spill_locate::<i64, WordKeyHome, Stored<i64>, i64>(
            &table.shards[0],
            Sigma::Parabolic,
            &address,
            &key,
            16,
        ) {
            SpillSite::Vacant(slot) => slot,
            other => panic!("a fresh key selects a vacancy, not {other:?}"),
        };
        let held =
            <Stored<i64> as KeyRep<i64, WordKeyHome>>::hold(Sigma::Parabolic, &key, &address);
        table.shards[0].publish_live(slot, (Sigma::Parabolic, held, 1_000 + key), address.place);
        stored.push(key);
    }
    assert_eq!(
        table.shards[0].live, capacity,
        "the region is full, so no vacancy remains"
    );
    assert!(table.shards[0].markers().iter().all(|&marker| marker != 0));

    // The absent key shares the full region and is not stored in it.
    let absent = 256 * i64::try_from(capacity).expect("the index fits i64");
    assert!(!stored.contains(&absent));
    assert_eq!(
        shard(&Home::address(&WordKeyHome, Sigma::Parabolic, &absent)),
        0
    );

    // A stored key is found before the cycle completes.
    read_trace::clear();
    assert_eq!(
        table.get(&256),
        &(1_000 + 256),
        "a stored key is read from its slot"
    );
    let hit_steps = read_trace::sweep_steps();
    assert!(
        hit_steps < capacity,
        "a present read stops inside the cycle"
    );
    assert_eq!(
        read_trace::absence_certificates(),
        0,
        "a present read certifies nothing"
    );

    // The absent key exhausts the bound: exactly one cycle, and no zero tag anywhere in it.
    read_trace::clear();
    assert_eq!(
        table.get(&absent),
        &table.default,
        "a completed cycle with no match reads the default"
    );
    assert_eq!(
        read_trace::sweep_steps(),
        capacity,
        "the absence certificate is one completed cycle over the whole region"
    );
    assert_eq!(
        read_trace::absence_certificates(),
        0,
        "a completed cycle is certified by the bound, not by a zero tag"
    );

    // The control: a vacancy present makes the same absent key cheap, so the cost above is the
    // cycle and not the key.
    let absent_address = Home::address(&WordKeyHome, Sigma::Parabolic, &absent);
    let home = table.shards[0].home(absent_address.place);
    table.shards[0].slots[home].tag = 0;
    table.shards[0].live -= 1;
    read_trace::clear();
    assert_eq!(table.get(&absent), &table.default);
    assert_eq!(
        read_trace::sweep_steps(),
        1,
        "with a vacancy at the home slot the same absent key stops on the first zero tag"
    );
    assert_eq!(
        read_trace::absence_certificates(),
        1,
        "the vacancy is the certificate now"
    );
}

/// The flat region carries exactly one live slot per key, its live count agrees with its nonzero
/// tags, and every live tag's slot holds the inserted key.
#[test]
fn every_live_spill_slot_is_tagged_and_counted_on_the_real_insertion_path() {
    let mut table = Table::<i64, i64, WordKeyHome, 16>::new(0, WordKeyHome);
    for key in 0..i64::try_from(SHARDS).expect("SHARDS = 256 fits i64") {
        table.insert_with(&ReplacementMerge, key, key + 1).unwrap();
    }
    let mut live = 0usize;
    for shard_at in 0..SHARDS {
        let region = &table.shards[shard_at];
        let mut counted = 0usize;
        for index in 0..region.slots.len() {
            if region.slots[index].tag != 0 {
                counted += 1;
                live += 1;
                let slot = &region.slots[index];
                let address = <Stored<i64> as KeyRep<i64, WordKeyHome>>::address_held(
                    &table.home,
                    slot.sigma,
                    &slot.held,
                );
                assert_eq!(
                    slot.tag,
                    tag(address.place),
                    "the live marker in shard {shard_at} is the published placement's fingerprint"
                );
            }
        }
        assert_eq!(
            counted, region.live,
            "the live count matches the nonzero tags"
        );
        assert!(region.live < region.slots.len(), "a region keeps a vacancy");
    }
    assert_eq!(live, SHARDS, "one live spill entry per shard");
}

/// A fresh spill write publishes the payload, then the nonzero tag.
#[test]
fn spill_publication_order_is_payload_then_tag() {
    publication_trace::clear();
    let mut table = Table::<i64, i64, WordKeyHome, 16>::new(0, WordKeyHome);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    assert_eq!(
        publication_trace::events(),
        vec![publication_trace::PAYLOAD, publication_trace::TAG],
        "a fresh spill write publishes payload, then the nonzero tag"
    );
}

/// A placement-only home that routes every key to shard zero at place zero, so a spill slot can be
/// placed in a later window than the first.
#[derive(Clone, Copy)]
struct ZeroHome;

impl Home<i64> for ZeroHome {
    fn home(&self, _key: &i64) -> u64 {
        0
    }

    /// Placement-only: reproduces the former default (`home`-derived place, no direct word).
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

sigma_word_face!(ZeroHome, i64);

/// `grow_spill` sizes the successor from the region's live count, and the successor keeps a vacancy
/// above the relocated entries.
///
/// `ZeroHome` routes every key to shard zero at placement zero, so one region's occupancy is driven
/// by successive insertions on the real path. [`SpillSite::Grow`] is raised only by a completed probe
/// cycle that found every slot occupied, so at each growth the live count equals the capacity: the
/// successor is exactly `slots_for(live_before, N)`, which for a power-of-two capacity at or above `N`
/// and a full region is twice the old capacity. The fixture asserts that identity, the retained
/// vacancy, and that every inserted key is still readable afterwards. A successor sized from any
/// other count leaves the region full or over-allocated and fails one of the three assertions.
#[test]
fn grow_spill_sizes_the_successor_from_the_live_count() {
    const N: usize = 16;
    let mut table = Table::<i64, i64, ZeroHome, N>::new(0, ZeroHome);
    let mut growths: Vec<(usize, usize, usize)> = Vec::new();
    let mut key = 0i64;
    // Bounded by the fixture's own key budget. Every key lands in shard zero, so the region grows
    // once per doubling and this loop is finite and deterministic.
    while key < 256 {
        let live_before = table.shards[0].live;
        let capacity_before = table.shards[0].slots.len();
        table.insert_with(&ReplacementMerge, key, key + 1).unwrap();
        let capacity_after = table.shards[0].slots.len();
        if capacity_after != capacity_before {
            growths.push((live_before, capacity_before, capacity_after));
        }
        key += 1;
    }
    assert!(
        !growths.is_empty(),
        "the fixture must observe at least one relocation, or it asserts nothing"
    );
    for (live_before, capacity_before, capacity_after) in &growths {
        if *capacity_before > 0 {
            assert_eq!(
                live_before, capacity_before,
                "`Grow` is raised only by a full region, so the live count is the capacity"
            );
            assert_eq!(
                *capacity_after,
                *capacity_before * 2,
                "the successor doubles a full region's capacity, which is slots_for(live, N) for a \
                 power-of-two capacity at or above N"
            );
        }
        assert_eq!(
            *capacity_after,
            slots_for((*live_before).max(1), N).expect("a checked successor target"),
            "the successor is the checked slots_for target of the live count"
        );
        // Read at the growth point, not at the end of the fill: the relocation leaves exactly
        // `live_before` entries in a `capacity_after`-slot region, and the keys inserted after it
        // legitimately fill that region again.
        assert!(
            live_before < capacity_after,
            "the successor keeps a vacancy above the relocated entries"
        );
    }
    for read in 0..key {
        assert_eq!(
            table.get(&read),
            &(read + 1),
            "every inserted key is still readable after the relocations"
        );
    }
}

// A test-only allocation counter is not declared here: the lib test binary already installs one
// `#[global_allocator]` (`src/table/partition.rs`, `partition::tests::counting`), and a second is
// rejected by the compiler. Read allocation-freedom is therefore checked by the clone/merge
// counters below plus an unchanged-capacity snapshot; see the task residual for the shared
// allocator hook.

/// A placement-only home for one shard: distinct placements, shard byte zero, no direct word.
///
/// `place = key << 8` leaves the shard selector — the low byte of the placement — at zero while
/// keeping every placement word distinct, so all keys reach shard zero's one region and the
/// region's own home mixer spreads them over its slots instead of stacking them at slot zero.
struct OneShardHome;

impl Home<i64> for OneShardHome {
    fn home(&self, key: &i64) -> u64 {
        (*key).cast_unsigned() << 8
    }

    /// Placement-only: no certified direct word, so every key reaches the spill region.
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

sigma_word_face!(OneShardHome, i64);

/// A shard region at the compile-time group width keeps a vacancy, so an absent read is certified by
/// the zero tag instead of by a completed cycle.
///
/// This is the read-path cost the region capacity rule decides. [`slots_for`] reserves `2 * n`
/// slots for a region's first entry, so after `n` real `insert_with` calls the region holds `n` live
/// entries in `2 * n` slots: load factor `1 / 2`, where a bounded read cycle meets a vacancy after
/// `1 / (1 - alpha) = 2` slots on average. A floor of `n` reserved `n` slots for the first entry and
/// let the insert path reach load factor `1`, where no vacancy exists and every absent read walks the
/// region's whole cycle before the bound certifies it. The assertions read the table's own state and
/// its own read counters: `n` live entries in `2 * n` slots, an absent read carrying one zero-tag
/// certificate and strictly fewer steps than the cycle, and every inserted key still readable.
#[test]
fn a_group_width_shard_region_keeps_a_vacancy_for_an_absent_read() {
    const N: usize = 16;
    let group = i64::try_from(N).expect("the group width fits i64");
    let mut table = Table::<i64, i64, OneShardHome, N>::new(0, OneShardHome);
    for key in 1..=group {
        table.insert_with(&ReplacementMerge, key, key + 1).unwrap();
    }

    let region = &table.shards[0];
    assert_eq!(table.len(), N, "every key is admitted exactly once");
    assert_eq!(region.live, N, "every key reaches shard zero's one region");
    assert_eq!(
        region.slots.len(),
        2 * N,
        "the reserved capacity is twice the group width, so the load factor is at most one half"
    );

    // The absent key shares shard zero with a half-full region and was never stored.
    let absent = group + 1;
    read_trace::clear();
    assert_eq!(
        table.get(&absent),
        &table.default,
        "a key the region never stored reads the default"
    );
    let steps = read_trace::sweep_steps();
    assert_eq!(
        read_trace::absence_certificates(),
        1,
        "a half-full region has a vacancy, so the zero tag certifies absence"
    );
    assert!(
        steps < region.slots.len(),
        "the read stops at the first vacancy after {steps} steps instead of completing the region's \
         whole cycle of {} steps",
        region.slots.len()
    );

    for key in 1..=group {
        assert_eq!(
            table.get(&key),
            &(key + 1),
            "every inserted key is still readable after the reservations"
        );
    }
}

/// The read helper walks the bounded cycle past same-fingerprint rejects to a late match.
#[test]
fn read_helper_finds_a_late_match_after_same_fingerprint_rejects() {
    let address = TableAddress {
        place: 0,
        direct_word: None,
    };
    let held = <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &7, &address);
    let mut region = Region::reserve_empty(32).expect("one flat region");
    region.initialize(Sigma::Parabolic, &held, &0);
    let mut filler: i64 = 100;
    while filler < 116 {
        let index =
            usize::try_from(filler - 100).expect("the filler range is 0..16 and fits usize");
        region.publish_live(
            index,
            (
                Sigma::Parabolic,
                <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &filler, &address),
                41,
            ),
            address.place,
        );
        filler += 1;
    }
    region.publish_live(16, (Sigma::Parabolic, held, 41), address.place);

    read_trace::clear();
    assert_eq!(
        spill_read::<i64, WordHome, Stored<i64>, i64>(&region, Sigma::Parabolic, &address, &7, 16,),
        Some(16)
    );
    assert_eq!(
        read_trace::sweep_steps(),
        17,
        "the cycle visits slots 0..=16 and stops at the match"
    );
}

/// Found-key identity is unchanged for the specialized read on both carriers and both storage
/// areas: a direct resident hit, a placement-only spill hit, and an unequal-`Sigma` miss.
#[test]
fn read_found_identity_is_unchanged_for_both_carriers_and_areas() {
    let sigma = Sigma::Parabolic;
    // Stored carrier, direct resident hit.
    let mut stored = Table::<i64, i64, WordHome, 16>::new(-1, WordHome);
    stored
        .insert_with_at(&ReplacementMerge, sigma, 7, 41)
        .unwrap();
    assert_eq!(*stored.get_at(sigma, &7), 41);

    // Regen carrier, direct resident hit.
    let mut regen = Table::<i64, i64, WordHome, 16, Regen<I64WordCodec>>::new(-1, WordHome);
    regen
        .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 43)
        .unwrap();
    assert_eq!(*regen.get_at(Sigma::Elliptic, &7), 43);

    // Stored carrier, placement-only spill hit.
    let home = WordKeyHome;
    let mut stored_spill = Table::<i64, i64, _, 16>::new(-1, home);
    stored_spill
        .insert_with_at(&ReplacementMerge, sigma, 7, 47)
        .unwrap();
    assert!(
        stored_spill.resident.slots.is_empty(),
        "a placement-only key admits no resident lane"
    );
    assert_eq!(*stored_spill.get_at(sigma, &7), 47);

    // Regen carrier, placement-only spill hit.
    let mut regen_spill = Table::<i64, i64, _, 16, Regen<I64WordCodec>>::new(-1, home);
    regen_spill
        .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 53)
        .unwrap();
    assert_eq!(*regen_spill.get_at(Sigma::Elliptic, &7), 53);

    // Unequal `Sigma` on the same word does not alias the resident or spill value.
    assert_eq!(*stored.get_at(Sigma::Hyperbolic, &7), -1);
    assert_eq!(*stored_spill.get_at(Sigma::Elliptic, &7), -1);
}

/// Absence is exact on the direct-empty certificate: an admitted resident lane that is not live
/// returns the default under composite `DirectCover`.
#[test]
fn read_absence_direct_empty_certificate_returns_default() {
    let sigma = Sigma::Parabolic;
    let mut table = Table::<i64, i64, WordHome, 16>::new(-1, WordHome);
    table
        .insert_with_at(&ReplacementMerge, sigma, 0, 10)
        .unwrap();
    let word = Home::address(&WordHome, sigma, &1)
        .direct_word
        .expect("a word key is certified");
    let index = table.resident.index(word).expect("lane one is admitted");
    assert!(
        !table.resident.slots[index].live,
        "the direct lane must be empty"
    );
    assert_eq!(*table.get_at(sigma, &1), -1);
}

/// A read is effect-free: it clones nothing, merges nothing and reallocates no committed storage on
/// both carriers and both storage areas. Allocation-freedom is additionally structural: `get_at`
/// takes `&self` and returns a borrow. See the residual for the shared allocation-counter hook.
#[test]
fn read_clones_and_merges_nothing_and_leaves_storage_unchanged() {
    let sigma = LocatorSigma::Parabolic;
    let clones = Rc::new(Cell::new(0usize));
    let merges = Rc::new(Cell::new(0usize));
    let mk = |value: i64| Counted {
        value,
        clones: Rc::clone(&clones),
    };
    let merge = {
        let calls = Rc::clone(&merges);
        move |new: Counted, _old: Counted| {
            calls.set(calls.get() + 1);
            new
        }
    };

    // Resident carrier: `CountWordHome` certifies direct words.
    let mut resident = Table::<Counted, Counted, CountWordHome, 16>::new(mk(-1), CountWordHome);
    resident.insert_with(&merge, mk(7), mk(41)).unwrap();
    // Spill carrier: `CountHome` is placement-only.
    let mut spill = Table::<Counted, Counted, CountHome, 16>::new(mk(-1), CountHome);
    spill.insert_with(&merge, mk(7), mk(47)).unwrap();

    let query_hit = mk(7);
    let query_absent = mk(1000);
    clones.set(0);
    merges.set(0);
    let resident_len = resident.resident.slots.len();
    let resident_cap = resident.resident.slots.capacity();
    let resident_lanes = resident.resident.lanes;
    let spill_slots = spill.shards[0].slots.capacity();
    let spill_len = spill.shards[0].slots.len();
    let resident_hit = resident.get_at(sigma, &query_hit).value;
    let resident_sigma_miss = resident.get_at(LocatorSigma::Hyperbolic, &query_hit).value;
    let spill_hit = spill.get_at(sigma, &query_hit).value;
    let spill_absent = spill.get_at(sigma, &query_absent).value;
    let resident_absent = resident.get_at(sigma, &query_absent).value;

    assert_eq!(
        (
            resident_hit,
            resident_sigma_miss,
            spill_hit,
            spill_absent,
            resident_absent
        ),
        (41, -1, 47, -1, -1)
    );
    assert_eq!(clones.get(), 0, "a read must not clone");
    assert_eq!(merges.get(), 0, "a read must not merge");
    // No read mutates or reallocates the committed storage it borrows.
    assert_eq!(resident.resident.slots.len(), resident_len);
    assert_eq!(resident.resident.slots.capacity(), resident_cap);
    assert_eq!(resident.resident.lanes, resident_lanes);
    assert_eq!(spill.shards[0].slots.capacity(), spill_slots);
    assert_eq!(spill.shards[0].slots.len(), spill_len);
}

/// Timing diagnostic, ignored by default: the old decision-then-apply read versus the specialized
/// `get_at`, on a resident hit and an absence-negative spill miss. Run with
/// `cargo test --release --lib read_path_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing diagnostic; run with --ignored --nocapture"]
#[allow(
    clippy::cast_precision_loss,
    reason = "the diagnostic prints a human-readable ns/op ratio; no assertion depends on the rounding"
)]
fn read_path_timing_old_decision_versus_specialized() {
    use std::time::Instant;
    const ITERS: usize = 2_000_000;
    let sigma = Sigma::Parabolic;
    let mut table = Table::<i64, i64, WordHome, 16>::new(-1, WordHome);
    let keys: Vec<i64> = (0..64).collect();
    for &key in &keys {
        table.insert_with(&ReplacementMerge, key, key + 1).unwrap();
    }
    let mut acc = 0i64;
    let old = Instant::now();
    for i in 0..ITERS {
        let key = keys[i % keys.len()];
        acc = acc.wrapping_add(*table.apply_get(table.search_decision_at(sigma, &key)));
    }
    let old = old.elapsed();
    let new = Instant::now();
    for i in 0..ITERS {
        let key = keys[i % keys.len()];
        acc = acc.wrapping_add(*table.get_at(sigma, &key));
    }
    let new = new.elapsed();
    println!(
        "resident hit: old {:.3} ns/op, specialized {:.3} ns/op (acc {acc})",
        old.as_nanos() as f64 / ITERS as f64,
        new.as_nanos() as f64 / ITERS as f64
    );

    // An absence-negative miss on a placement-only spill region: the lever this slice targets.
    let home = WordKeyHome;
    let mut spill = Table::<i64, i64, _, 16>::new(-1, home);
    spill.insert_with(&ReplacementMerge, 7, 41).unwrap();
    let shard_at = shard(&Home::address(&home, sigma, &7));
    let place = |key: i64| Home::address(&home, sigma, &key).place;
    // An absent key: the bounded cycle terminates on a zero tag, the flat port's absence
    // certificate.
    let mut miss = 7i64 + 256;
    while spill.shards[shard_at]
        .slots
        .iter()
        .any(|slot| slot.tag == tag(place(miss)))
    {
        miss += 256;
    }
    let old = Instant::now();
    for _ in 0..ITERS {
        acc = acc.wrapping_add(*spill.apply_get(spill.search_decision_at(sigma, &miss)));
    }
    let old = old.elapsed();
    let new = Instant::now();
    for _ in 0..ITERS {
        acc = acc.wrapping_add(*spill.get_at(sigma, &miss));
    }
    let new = new.elapsed();
    println!(
        "absence-negative spill miss: old {:.3} ns/op, specialized {:.3} ns/op (acc {acc})",
        old.as_nanos() as f64 / ITERS as f64,
        new.as_nanos() as f64 / ITERS as f64
    );
}

/// Both serial build branches maintain coverage `C`: the ordered short branch and the partitioned
/// long branch (above `BUILD_PARTITION_CROSSOVER`) each leave a covered table with the exact live
/// length, and a repeated key keeps one representative and one reverse link.
#[test]
fn serial_build_branches_maintain_storage_coverage() {
    // Short branch: below the partition crossover, the ordered fold.
    let short: Vec<(i64, i64)> = (0..64).map(|key| (key, key + 1)).collect();
    let table =
        Table::<i64, i64, WordHome, 16>::build(&ReplacementMerge, 0, &short, WordHome).unwrap();
    assert_eq!(table.len(), 64);
    assert!(table.check_storage_coverage(), "short serial build branch");

    // Long branch: above the crossover, the partitioned path runs.
    let long: Vec<(i64, i64)> = (0..5000).map(|key| (key, key + 1)).collect();
    let table =
        Table::<i64, i64, WordHome, 16>::build(&ReplacementMerge, 0, &long, WordHome).unwrap();
    assert_eq!(table.len(), 5000);
    assert!(
        table.check_storage_coverage(),
        "partitioned serial build branch"
    );

    // A repeated key is one logical representative with one reverse link.
    let repeated: Vec<(i64, i64)> = vec![(7, 1), (7, 2)];
    let table =
        Table::<i64, i64, WordHome, 16>::build(&ReplacementMerge, 0, &repeated, WordHome).unwrap();
    assert_eq!(table.len(), 1);
    assert_eq!(table.locator.links.len(), 1);
    assert!(table.check_storage_coverage());
}
