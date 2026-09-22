//! Unit tests for the growable table, factored out of `src/table.rs` without behavior change.

use super::keyrep::{I64WordCodec, Regen};
use super::{
    build_prepared_shard, capacity_bound_holds, keyrep::KeyRep, publication_trace, read_trace,
    shard, slots_for, spill_locate, spill_read, summary_trailing_zeros, summary_words, tag,
    ApplyOutcome, Flat, FlatSlot, Home, PreparedResult, RebuildPlan, RebuildReserved, Region,
    ReplacementMerge, SearchClass, SearchDecision, SearchLocation, SpillSite, StorageError, Stored,
    Table, TableAddress, WordHome, SHARDS,
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
    assert_eq!(table.density_at(4, Some(1000)), Ok(1001));
    assert_eq!(table.selected_lanes(0, 1000), Ok(Some(4)));

    // Snapshot the committed state; a refusal must not touch it.
    let before_len = table.len();
    let before_lanes = table.resident.lanes;
    let before_resident_len = table.resident.slots.len();
    let before_tags: Vec<u8> = table.shards[0].tags.clone();
    let before_slots = table.shards[0].slots.len();
    clones.set(0);

    // A bitmap whose checked size overflows `usize` refuses before any allocation attempt.
    assert_eq!(
        table.density_at(usize::MAX, Some(0)),
        Err(StorageError::SizeOverflow)
    );
    // A bitmap sizing that fits but whose reservation the allocator refuses is a refusal too.
    assert_eq!(
        table.density_at(usize::MAX / 256, Some(0)),
        Err(StorageError::AllocationFailed)
    );

    assert_eq!(table.len(), before_len);
    assert_eq!(table.resident.lanes, before_lanes);
    assert_eq!(table.resident.slots.len(), before_resident_len);
    assert_eq!(table.shards[0].tags, before_tags);
    assert_eq!(table.shards[0].slots.len(), before_slots);
    assert_eq!(clones.get(), 0, "a density refusal must not clone");
}

/// A recorded prefix vacancy does not stop the mandatory coverage sweep's match search.
#[test]
fn spill_sweep_finds_a_late_match_after_a_window_vacancy() {
    let address = TableAddress {
        place: 0,
        direct_word: None,
    };
    let held = <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &7, &address);
    let mut region = Region::reserve_empty(32).expect("two initialized groups");
    region.initialize(Sigma::Parabolic, &held, &0);
    region.slots[16] = (Sigma::Parabolic, held, 41);
    region.tags[16] = tag(address.place);
    region.summary_set(address.place);

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

/// A spill vacancy selects the least empty lane in the first probe group.
#[test]
fn spill_decision_selects_the_first_vacant_lane() {
    let address = TableAddress {
        place: 0,
        direct_word: None,
    };
    let held = <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &7, &address);
    let mut region = Region::reserve_empty(32).expect("two initialized groups");
    region.initialize(Sigma::Parabolic, &held, &0);
    region.tags[0] = tag(address.place);
    region.slots[0] = (Sigma::Parabolic, held, 41);
    region.summary_set(address.place);

    assert_eq!(
        spill_locate::<i64, WordHome, Stored<i64>, i64>(
            &region,
            Sigma::Parabolic,
            &address,
            &8,
            16,
        ),
        SpillSite::Vacant(1)
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
    let before_tags: Vec<u8> = table.shards[0].tags.clone();
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
    assert_eq!(table.shards[0].tags, before_tags);
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
    assert_eq!(storage.regions[0].tags.len(), target);
    assert_eq!(storage.regions[0].slots.len(), target);
    assert!(storage.regions[0].tags.iter().all(|&tag_at| tag_at == 0));
    assert_eq!(storage.regions[0].slots[0].0, sigma);

    // The receipt records the checked targets and the observed capacities meet them.
    let receipt = &outcome.reservation;
    assert_eq!(receipt.flat.required_len, 256);
    assert_eq!(receipt.directory.required_len, SHARDS);
    assert_eq!(receipt.regions[0].tags.required_len, target);
    assert_eq!(receipt.regions[0].payloads.required_len, target);
    assert_eq!(
        receipt.regions[0].summary.required_len,
        summary_words(target).expect("checked summary target")
    );
    assert!(capacity_bound_holds(
        receipt.flat.required_len,
        receipt.flat.observed_capacity
    ));
    assert!(capacity_bound_holds(
        receipt.directory.required_len,
        receipt.directory.observed_capacity
    ));
    assert!(capacity_bound_holds(
        receipt.regions[0].tags.required_len,
        receipt.regions[0].tags.observed_capacity
    ));
    assert!(capacity_bound_holds(
        receipt.regions[0].payloads.required_len,
        receipt.regions[0].payloads.observed_capacity
    ));
    assert!(capacity_bound_holds(
        receipt.regions[0].summary.required_len,
        receipt.regions[0].summary.observed_capacity
    ));
    assert_eq!(
        storage.regions[0].summary.len(),
        receipt.regions[0].summary.required_len
    );

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

/// A summary-sizing overflow is a transactional refusal: no hold, clone, merge or relocation
/// runs, and the committed table is unchanged.
///
/// The synthetic plan routes a checked slot target whose summary bit count overflows `usize` to
/// shard 0, so `reserve_rebuild` refuses at the checked summary sizing before any allocation or
/// hold. This is the deterministic summary-refusal injection: a saturating or unchecked summary
/// sizing does not return `SizeOverflow` here, so this test is red under that mutation.
#[test]
fn summary_sizing_overflow_refuses_before_any_hold_or_clone() {
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

    let before_len = table.len();
    let before_resident_len = table.resident.slots.len();
    let before_tags: Vec<u8> = table.shards[0].tags.clone();
    let before_slots = table.shards[0].slots.len();
    clones.set(0);
    merge_calls.set(0);

    // `slots_for(2^58, 16) = 2^59`; its summary bit count is `2^59 * 32 = 2^64`, an overflow.
    let mut spill_loads = vec![0usize; SHARDS];
    spill_loads[0] = 1usize << 58;
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

    assert_eq!(table.len(), before_len);
    assert_eq!(table.resident.slots.len(), before_resident_len);
    assert_eq!(table.shards[0].tags, before_tags);
    assert_eq!(table.shards[0].slots.len(), before_slots);
    assert_eq!(clones.get(), 0, "a summary refusal must not clone");
    assert_eq!(merge_calls.get(), 0, "a summary refusal must not merge");
}

/// `build_prepared_shard` reserves its complete checked region before the first hold, so a
/// sizing refusal runs no clone. This is the per-shard ordering a parallel worker uses.
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
    clones.set(0);
    let result = build_prepared_shard::<Counted, Counted, CountHome, Stored<Counted>, _, 16>(
        &CountHome,
        &merge,
        &mk(-1),
        &input,
        &parts,
        0,
        1usize << 58,
    );
    assert!(matches!(result, Err(StorageError::SizeOverflow)));
    assert_eq!(clones.get(), 0, "reservation must precede the first hold");
}

/// The plain shift loop agrees with `usize::trailing_zeros` at zero, odd, even, power-of-two and
/// maximal values, so replacing the intrinsic preserves the summary bit count.
#[test]
fn summary_trailing_zeros_matches_the_standard_intrinsic() {
    let mut samples = vec![0usize, 1, 2, 3, 6, 7, 8, usize::MAX];
    let mut bit = 0u32;
    while bit < usize::BITS {
        samples.push(1usize << bit);
        bit += 1;
    }
    for &value in &samples {
        assert_eq!(
            summary_trailing_zeros(value),
            value.trailing_zeros(),
            "summary_trailing_zeros({value})"
        );
    }
}

/// A zero-length summary is total: no shift is evaluated, `summary_set` is a no-op, and
/// `summary_maybe` reports "maybe" so filtering is disabled. `summary_index` returns `None` for the
/// zero length rather than shifting by `64 + 6`.
#[test]
fn zero_length_summary_is_total_and_disables_filtering() {
    let mut region = Region::<Stored<i64>, i64>::empty();
    assert_eq!(region.summary.len(), 0);
    assert_eq!(region.slots.len(), 0);
    assert_eq!(
        Region::<Stored<i64>, i64>::summary_index(0xDEAD_BEEF, 0),
        None,
        "a zero-length summary has no bit index"
    );

    region.summary_set(0xDEAD_BEEF);
    assert!(
        region.summary.is_empty(),
        "an empty summary records nothing"
    );

    assert!(
        region.summary_maybe(0xDEAD_BEEF),
        "an empty summary cannot certify absence, so the sweep must run"
    );
}

/// A malformed or oversized summary length is total: the shift is guarded, no call panics and the
/// result is defined. Lengths whose bit count reaches `u64::BITS` yield no index without
/// allocating; a small non-power-of-two length still yields an in-bounds index.
#[test]
fn malformed_summary_lengths_are_total() {
    // `trailingZeros(len) + 6 >= 64`: no index, so no `u64` shift by `>= 64` and no allocation.
    assert_eq!(
        Region::<Stored<i64>, i64>::summary_index(7, 1usize << 58),
        None
    );
    assert_eq!(
        Region::<Stored<i64>, i64>::summary_index(7, 3usize << 58),
        None
    );
    assert_eq!(
        Region::<Stored<i64>, i64>::summary_index(7, 1usize << 63),
        None
    );
    // The boundary below the guard is still usable: `trailingZeros(2^57) + 6 = 63`.
    assert!(
        Region::<Stored<i64>, i64>::summary_index(7, 1usize << 57).is_some(),
        "a bit count of 63 is still indexable"
    );

    // A small non-power-of-two summary stays in bounds: `bits = trailingZeros(5) + 6 = 6` and every
    // returned index is below `64 * 5`.
    let mut region = Region::<Stored<i64>, i64>::empty();
    region.summary = vec![0u64; 5];
    let index = Region::<Stored<i64>, i64>::summary_index(0x1234, region.summary.len())
        .expect("a small malformed summary still yields an index");
    assert!(index < 64 * region.summary.len());
    region.summary_set(0x1234);
    let _ = region.summary_maybe(0x1234);
    assert_eq!(region.summary.len(), 5, "no panic and no resize");
}

/// Every live spill tag has its placement's summary bit recorded on the real insertion path.
#[test]
fn every_live_spill_tag_has_its_summary_bit_recorded() {
    let mut table = Table::<i64, i64, WordKeyHome, 16>::new(0, WordKeyHome);
    for key in 0..i64::try_from(SHARDS).expect("SHARDS = 256 fits i64") {
        table.insert_with(&ReplacementMerge, key, key + 1).unwrap();
    }
    let mut live = 0usize;
    for shard_at in 0..SHARDS {
        let region = &table.shards[shard_at];
        assert!(
            !region.summary.is_empty(),
            "a nonempty region carries a summary"
        );
        for index in 0..region.tags.len() {
            if region.tags[index] != 0 {
                live += 1;
                let (sigma, held, _) = &region.slots[index];
                let address = <Stored<i64> as KeyRep<i64, WordKeyHome>>::address_held(
                    &table.home,
                    *sigma,
                    held,
                );
                let bit =
                    Region::<Stored<i64>, i64>::summary_index(address.place, region.summary.len())
                        .expect("a live nonempty region has a usable summary word");
                assert!(
                    region.summary[bit / 64] & (1u64 << (bit % 64)) != 0,
                    "live tag in shard {shard_at} has no recorded summary bit"
                );
            }
        }
    }
    assert_eq!(live, SHARDS, "one live spill entry per shard");
}

/// A fresh spill write publishes the payload, then the summary bit, then the nonzero tag.
#[test]
fn spill_publication_order_is_payload_then_summary_then_tag() {
    publication_trace::clear();
    let mut table = Table::<i64, i64, WordKeyHome, 16>::new(0, WordKeyHome);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    assert_eq!(
        publication_trace::events(),
        vec![
            publication_trace::PAYLOAD,
            publication_trace::SUMMARY,
            publication_trace::TAG
        ],
        "a fresh spill write publishes payload, then summary bit, then tag"
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

// A test-only allocation counter is not declared here: the lib test binary already installs one
// `#[global_allocator]` (`src/table/partition.rs`, `partition::tests::counting`), and a second is
// rejected by the compiler. Read allocation-freedom is therefore checked by the clone/merge
// counters below plus an unchanged-capacity snapshot; see the task residual for the shared
// allocator hook.

/// The read-only spill matcher finds a late match after a window vacancy, without any vacancy
/// bookkeeping.
#[test]
fn read_helper_finds_a_late_match_after_a_window_vacancy() {
    let address = TableAddress {
        place: 0,
        direct_word: None,
    };
    let held = <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &7, &address);
    let mut region = Region::reserve_empty(32).expect("two initialized groups");
    region.initialize(Sigma::Parabolic, &held, &0);
    region.slots[16] = (Sigma::Parabolic, held, 41);
    region.tags[16] = tag(address.place);
    region.summary_set(address.place);

    read_trace::clear();
    assert_eq!(
        spill_read::<i64, WordHome, Stored<i64>, i64>(&region, Sigma::Parabolic, &address, &7, 16,),
        Some(16)
    );
    assert_eq!(
        read_trace::sweep_steps(),
        super::WINDOWS + 1,
        "group zero is empty and the sweep still reaches the late group"
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

/// Absence on the summary-miss certificate: after direct exclusion a negative summary returns the
/// default and the read runs no coverage sweep.
#[test]
fn read_absence_summary_miss_certificate_skips_the_sweep() {
    let sigma = Sigma::Parabolic;
    let home = WordKeyHome;
    let mut table = Table::<i64, i64, _, 16>::new(-1, home);
    table
        .insert_with_at(&ReplacementMerge, sigma, 7, 41)
        .unwrap();
    let shard_at = shard(&Home::address(&home, sigma, &7));
    let region = &table.shards[shard_at];
    assert!(
        !region.summary.is_empty(),
        "the live region carries a summary"
    );
    let place = |key: i64| Home::address(&home, sigma, &key).place;
    let mut query = 7i64 + 256;
    let mut probes = 0usize;
    while region.summary_maybe(place(query)) && probes < 4096 {
        query += 256;
        probes += 1;
    }
    assert!(
        !region.summary_maybe(place(query)),
        "a same-shard summary-negative query must exist"
    );

    read_trace::clear();
    assert_eq!(*table.get_at(sigma, &query), -1);
    assert_eq!(
        read_trace::summary_misses(),
        1,
        "absence must be the summary-miss certificate"
    );
    assert_eq!(
        read_trace::sweep_steps(),
        0,
        "a summary miss must not sweep"
    );
    // The live key is still found.
    assert_eq!(*table.get_at(sigma, &7), 41);
}

/// Absence on the coverage certificate: a forced summary collision makes the summary positive for
/// an absent query, so the read must run complete finite match coverage and still return default.
#[test]
fn read_absence_coverage_certificate_forces_complete_sweep() {
    let sigma = Sigma::Parabolic;
    let home = WordKeyHome;
    let mut table = Table::<i64, i64, _, 16>::new(-1, home);
    table
        .insert_with_at(&ReplacementMerge, sigma, 7, 41)
        .unwrap();
    let shard_at = shard(&Home::address(&home, sigma, &7));
    let query = 7i64 + 256;
    let query_place = Home::address(&home, sigma, &query).place;
    let groups = table.shards[shard_at].slots.len() / 16;
    // Force the summary to report "maybe" for an absent key, exactly a filter false positive.
    table.shards[shard_at].summary_set(query_place);
    assert!(
        table.shards[shard_at].summary_maybe(query_place),
        "the forced collision makes the summary positive"
    );

    read_trace::clear();
    assert_eq!(*table.get_at(sigma, &query), -1);
    assert_eq!(
        read_trace::summary_misses(),
        0,
        "the summary reported maybe, so no miss"
    );
    assert_eq!(
        read_trace::sweep_steps(),
        super::WINDOWS + groups,
        "coverage must visit every group"
    );
    // The live key whose summary bit collided is still found: no false negative.
    assert_eq!(*table.get_at(sigma, &7), 41);
}

/// A forced summary collision never hides a live key on the real read path.
#[test]
fn read_forced_summary_collision_has_no_false_negative() {
    let sigma = Sigma::Parabolic;
    let home = WordKeyHome;
    let mut table = Table::<i64, i64, _, 16>::new(-1, home);
    table
        .insert_with_at(&ReplacementMerge, sigma, 7, 41)
        .unwrap();
    table
        .insert_with_at(&ReplacementMerge, sigma, 7 + 256, 42)
        .unwrap();
    let shard_at = shard(&Home::address(&home, sigma, &7));
    let absent = 7i64 + 512;
    let absent_place = Home::address(&home, sigma, &absent).place;
    table.shards[shard_at].summary_set(absent_place);
    assert!(
        table.shards[shard_at].summary_maybe(absent_place),
        "the forced collision makes the summary positive"
    );

    assert_eq!(*table.get_at(sigma, &7), 41, "first live key");
    assert_eq!(*table.get_at(sigma, &(7 + 256)), 42, "second live key");
    assert_eq!(*table.get_at(sigma, &absent), -1, "absent same-bit query");
}

/// A late spill hit after an earlier window vacancy is still found by the specialized read: the
/// only live slot sits in group one, group zero is empty, and the sweep must pass group zero.
#[test]
fn read_finds_a_late_spill_hit_after_a_window_vacancy() {
    let sigma = Sigma::Parabolic;
    let mut table = Table::<i64, i64, ZeroHome, 16>::new(-1, ZeroHome);
    table
        .insert_with_at(&ReplacementMerge, sigma, 7, 41)
        .unwrap();
    let address = Home::address(&ZeroHome, sigma, &7);
    let held = <Stored<i64> as KeyRep<i64, ZeroHome>>::hold(sigma, &7, &address);
    let shard_at = shard(&address);
    {
        let region = &mut table.shards[shard_at];
        for tag_at in &mut region.tags {
            *tag_at = 0;
        }
        for word in &mut region.summary {
            *word = 0;
        }
        region.slots[16] = (sigma, held, 41);
        region.tags[16] = tag(address.place);
        region.summary_set(address.place);
    }

    read_trace::clear();
    assert_eq!(*table.get_at(sigma, &7), 41);
    assert_eq!(read_trace::summary_misses(), 0);
    assert_eq!(
        read_trace::sweep_steps(),
        super::WINDOWS + 1,
        "the sweep must pass the seven empty first-window steps and reach the late group"
    );
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
    let spill_tags = spill.shards[0].tags.capacity();
    let spill_slots = spill.shards[0].slots.capacity();
    let spill_summary = spill.shards[0].summary.capacity();
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
    assert_eq!(spill.shards[0].tags.capacity(), spill_tags);
    assert_eq!(spill.shards[0].slots.capacity(), spill_slots);
    assert_eq!(spill.shards[0].summary.capacity(), spill_summary);
    assert_eq!(spill.shards[0].slots.len(), spill_len);
}

/// Timing diagnostic, ignored by default: the old decision-then-apply read versus the specialized
/// `get_at`, on a resident hit and a summary-negative spill miss. Run with
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

    // A summary-negative miss on a placement-only spill region: the lever this slice targets.
    let home = WordKeyHome;
    let mut spill = Table::<i64, i64, _, 16>::new(-1, home);
    spill.insert_with(&ReplacementMerge, 7, 41).unwrap();
    let shard_at = shard(&Home::address(&home, sigma, &7));
    let place = |key: i64| Home::address(&home, sigma, &key).place;
    let mut miss = 7i64 + 256;
    while spill.shards[shard_at].summary_maybe(place(miss)) {
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
        "summary-negative spill miss: old {:.3} ns/op, specialized {:.3} ns/op (acc {acc})",
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
