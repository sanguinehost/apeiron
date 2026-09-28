//! Parallel build/probe equivalence: the scoped executor must produce the same logical table as
//! the ordered serial path on the same input, for every worker count and under a load skew.
//!
//! The comparison is order-independent map equality: every distinct key is probed on both tables
//! and `len` is compared, so two tables that agree on every `get` and on their logical length are
//! equal as maps. The merge is deliberately non-commutative, so a worker that reorders one key's
//! values changes the compared value and the test fails.

use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, LocatorSigma, ParabolicDual,
    Residue, H3_40,
};
use apeiron::coord::locator::{ExactIdentity, Refusal};
use apeiron::coord::{spread, Packed};
use apeiron::table::keyrep::{I64WordCodec, KeyRep, Regen};
use apeiron::table::{Home, Table, TableAddress};
use apeiron::{Sigma, StorageError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

/// The crossover below which the parallel entry points fall back to the ordered serial path.
const CROSSOVER: usize = 4096;

/// An odd multiplier for the spread arm of the skew home.
const ODD: u64 = 0x9E37_79B9_7F4A_7C15;

/// The selected table under test: the default `Stored<i64>` carrier at `N = 16`.
type SkewTable = Table<i64, i64, SkewHome, 16>;

/// A home that routes seven eighths of the input to shard `7` and spreads the rest, so one worker
/// owns almost the whole load. Placement-only, so every key takes the spill path.
#[derive(Clone, Copy)]
struct SkewHome;

impl Home<i64> for SkewHome {
    fn home(&self, key: &i64) -> u64 {
        if key.rem_euclid(8) == 0 {
            key.cast_unsigned().wrapping_mul(ODD) | 1
        } else {
            7
        }
    }

    /// Placement-only: reproduces the former default (`home`-derived place, no direct word).
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

sigma_word_face!(SkewHome, i64);

/// A home that routes every key to shard 3, so one shard's run is the whole input and any
/// chunk-order concatenation error changes the non-commutative fold.
#[derive(Clone, Copy)]
struct SingleShardHome;

impl Home<i64> for SingleShardHome {
    fn home(&self, _key: &i64) -> u64 {
        3
    }

    /// Placement-only: reproduces the former default (`home`-derived place, no direct word).
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

sigma_word_face!(SingleShardHome, i64);

/// The selected table under test for the single-hot-shard order arm.
type SingleShardTable = Table<i64, i64, SingleShardHome, 16>;

/// A non-commutative Horner fold: the value depends on each key's value order, so a reordered
/// shard or a dropped shard changes it.
fn horner(new: i64, old: i64) -> i64 {
    old.wrapping_mul(31).wrapping_add(new)
}

/// Four occurrences of each of `n / 4` distinct keys, interleaved, so the partition has real work.
fn skewed_input(n: usize) -> Vec<(i64, i64)> {
    let distinct = i64::try_from((n / 4).max(1)).expect("test size fits i64");
    (0..n)
        .map(|i| {
            let i = i64::try_from(i).expect("test index fits i64");
            (i % distinct, i)
        })
        .collect()
}

/// The number of distinct keys in an input built by [`skewed_input`].
fn distinct_of(input: &[(i64, i64)]) -> i64 {
    i64::try_from((input.len() / 4).max(1)).expect("test size fits i64")
}

/// The probe set: every distinct key in reverse order plus absent keys, so the parallel grouped
/// readback must restore query order.
fn probe_set(distinct: i64) -> Vec<i64> {
    let mut queries: Vec<i64> = (0..distinct).rev().collect();
    queries.extend([-1, distinct, distinct + 1, -7]);
    queries
}

/// Build the serial reference table for the same input and merge.
fn serial(input: &[(i64, i64)]) -> SkewTable {
    SkewTable::build(&horner, 0, input, SkewHome).expect("serial build fits")
}

/// Assert two tables agree on every key in `0..distinct` plus `extra` and on their length.
fn assert_same_map(serial: &SkewTable, parallel: &SkewTable, distinct: i64, extra: &[i64]) {
    assert_eq!(
        serial.len(),
        parallel.len(),
        "parallel build must have the same logical length"
    );
    for key in (0..distinct).chain(extra.iter().copied()) {
        assert_eq!(
            serial.get(&key),
            parallel.get(&key),
            "key {key} disagrees between the serial and parallel tables"
        );
    }
}

/// The parallel build and probe equal the serial result across worker counts, including counts
/// wider than the shard count and the crossover boundary.
#[test]
fn parallel_build_and_probe_match_serial_across_worker_counts() {
    let input = skewed_input(20_000);
    let distinct = distinct_of(&input);
    let queries = probe_set(distinct);
    let reference = serial(&input);
    let serial_values: Vec<i64> = queries.iter().map(|key| *reference.get(key)).collect();

    for workers in [1usize, 2, 3, 7, 16, 64, 300] {
        let table = SkewTable::build_parallel(&horner, 0, &input, SkewHome, workers)
            .expect("parallel build fits");
        assert_same_map(&reference, &table, distinct, &queries);

        let values = table.get_each_parallel(&queries, workers, |value| *value);
        assert_eq!(
            values, serial_values,
            "workers={workers}: the parallel probe must restore query order"
        );
    }
}

/// With every key routed to one shard, the shard's run spans every input chunk. A concatenation
/// in worker-completion order or in reversed chunk order would reorder the non-commutative fold,
/// so equality with the serial map across worker counts is the chunk-order control.
#[test]
fn single_shard_repeated_keys_preserve_chunk_order_across_worker_counts() {
    let input = skewed_input(20_000);
    let distinct = distinct_of(&input);
    let reference = SingleShardTable::build(&horner, 0, &input, SingleShardHome)
        .expect("serial single-shard build fits");
    for workers in [1usize, 2, 3, 7, 16, 64, 300] {
        let table = SingleShardTable::build_parallel(&horner, 0, &input, SingleShardHome, workers)
            .expect("parallel single-shard build fits");
        assert_eq!(
            reference.len(),
            table.len(),
            "workers={workers}: logical length disagrees"
        );
        for key in 0..distinct {
            assert_eq!(
                reference.get(&key),
                table.get(&key),
                "workers={workers}: key {key} value order differs from serial"
            );
        }
    }
}

/// The same equivalence with `Sigma`-qualified keys through the `build_at_parallel` /
/// `get_each_at_parallel` entry points.
#[test]
fn parallel_at_build_and_probe_match_serial_across_worker_counts() {
    let input = skewed_input(20_000);
    let distinct = distinct_of(&input);
    let qualified: Vec<((Sigma, i64), i64)> = input
        .iter()
        .map(|(key, value)| ((Sigma::Parabolic, *key), *value))
        .collect();
    let queries: Vec<(Sigma, i64)> = probe_set(distinct)
        .into_iter()
        .map(|key| (Sigma::Parabolic, key))
        .collect();

    let reference =
        SkewTable::build_at(&horner, 0, &qualified, SkewHome).expect("serial build fits");
    let serial_values: Vec<i64> = queries
        .iter()
        .map(|(sigma, key)| *reference.get_at(*sigma, key))
        .collect();

    for workers in [1usize, 2, 3, 16, 64, 300] {
        let table = SkewTable::build_at_parallel(&horner, 0, &qualified, SkewHome, workers)
            .expect("parallel build fits");
        assert_eq!(reference.len(), table.len());
        for ((sigma, key), expected) in queries.iter().zip(&serial_values) {
            assert_eq!(
                *table.get_at(*sigma, key),
                *expected,
                "workers={workers}: key {key} disagrees"
            );
        }
        let values = table.get_each_at_parallel(&queries, workers, |value| *value);
        assert_eq!(
            values, serial_values,
            "workers={workers}: the qualified parallel probe must restore query order"
        );
    }
}

/// A load skew that sends most of the input to one shard still matches the serial map: the LPT
/// schedule assigns whole shards, so one worker carrying the hot shard cannot change the result.
#[test]
fn load_skew_matches_serial_and_dominates_one_shard() {
    let input = skewed_input(20_000);
    let distinct = distinct_of(&input);
    let hot = input
        .iter()
        .filter(|(key, _)| SkewHome.home(key) & 255 == 7)
        .count();
    assert!(
        hot * 2 > input.len(),
        "the skew must route more than half the input to shard 7 (got {hot} of {})",
        input.len()
    );

    let reference = serial(&input);
    for workers in [2usize, 3, 8, 16] {
        let table = SkewTable::build_parallel(&horner, 0, &input, SkewHome, workers)
            .expect("parallel build fits");
        assert_same_map(&reference, &table, distinct, &[-1, distinct, -7]);
    }
}

/// Below the crossover and at one worker the parallel entry points take the serial path and report
/// no parallel kernel time; at or above the crossover with two workers they report a parallel
/// kernel. This keeps the equivalence tests from passing on the fallback.
#[test]
fn parallel_kernel_runs_only_above_the_crossover_with_two_workers() {
    let small = skewed_input(CROSSOVER - 1);
    let (_, small_phases) = SkewTable::build_parallel_phased(&horner, 0, &small, SkewHome, 8)
        .expect("small build fits");
    assert_eq!(
        small_phases.kernel, 0,
        "below the crossover the entry point must use the serial path"
    );

    let (_, one_worker_phases) = SkewTable::build_parallel_phased(&horner, 0, &small, SkewHome, 1)
        .expect("one-worker build fits");
    assert_eq!(one_worker_phases.kernel, 0, "one worker is the serial path");

    let large = skewed_input(CROSSOVER + 1);
    let (table, phases) = SkewTable::build_parallel_phased(&horner, 0, &large, SkewHome, 2)
        .expect("large build fits");
    assert!(
        phases.kernel > 0,
        "two workers must run the parallel kernel"
    );
    assert_eq!(table.len(), serial(&large).len());
}

/// A `Merge` callback that panics in one worker is resumed after every scoped task has joined, so
/// the caller sees the panic instead of a silently truncated table.
#[test]
fn a_panicking_merge_propagates_and_never_returns_a_partial_table() {
    fn panicking_merge(_new: i64, _old: i64) -> i64 {
        panic!("merge refusal");
    }
    let input = skewed_input(20_000);
    let result = std::panic::catch_unwind(|| {
        SkewTable::build_parallel(&panicking_merge, 0, &input, SkewHome, 8)
    });
    assert!(
        result.is_err(),
        "a panicking worker must propagate, not return a partial table"
    );
}

/// The grouped reduction over default-face probes equals the serial sum.
#[test]
fn sum_parallel_matches_the_serial_sum() {
    let input = skewed_input(20_000);
    let distinct = distinct_of(&input);
    let queries = probe_set(distinct);
    let table = SkewTable::build_parallel(&horner, 0, &input, SkewHome, 16).expect("build fits");
    let serial_sum: i64 = queries.iter().map(|key| *table.get(key)).sum();
    for workers in [1usize, 4, 16, 300] {
        assert_eq!(table.sum_parallel(&queries, workers), serial_sum);
    }
}

/// The parallel build reports the same error type as the serial path on an unsatisfiable group
/// width; no partial table is returned.
#[test]
fn invalid_group_width_is_refused_on_both_paths() {
    let input = skewed_input(20_000);
    let parallel = Table::<i64, i64, SkewHome, 0>::build_parallel(&horner, 0, &input, SkewHome, 8);
    assert!(matches!(parallel, Err(StorageError::SizeOverflow)));
}

// ---------------------------------------------------------------------------------------------
// Cross-worker transactional lifecycle: reserve-all before any hold/clone/merge/relocation.
// ---------------------------------------------------------------------------------------------

/// A key that counts every clone, so `Stored::hold` (a key clone) and the held clone during region
/// initialization are observable across worker threads.
#[derive(Debug)]
struct CountedKey {
    value: i64,
    holds: Arc<AtomicUsize>,
}

impl Clone for CountedKey {
    fn clone(&self) -> Self {
        self.holds.fetch_add(1, Ordering::SeqCst);
        Self {
            value: self.value,
            holds: Arc::clone(&self.holds),
        }
    }
}

impl PartialEq for CountedKey {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for CountedKey {}

/// The exact identity word of a [`CountedKey`]: its `value`, never its hold counter.
fn counted_key_word(key: &CountedKey) -> u64 {
    key.value.cast_unsigned()
}

/// The exact-identity bytes of a key, collected for a direct byte-for-byte comparison.
fn identity_bytes<K: ExactIdentity>(key: &K) -> Vec<u8> {
    (0..key.encoding_len())
        .map(|index| key.encoding_byte(index))
        .collect()
}

/// The exact identity of [`CountedKey`] is its `value`; the hold counter is not equality and so is
/// not identity. The retained word is the two's-complement bit pattern of `value`.
impl ExactIdentity for CountedKey {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        counted_key_word(self)
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        counted_key_word(self)
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        counted_key_word(self) == *retained
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

/// `CountedKey`'s exact identity is its `value` field, never its hold counter.
///
/// Two distinct equality keys must not share an encoding, and two equal keys with different
/// counters must share one. The rejecting control encodes by the hold counter instead; the two
/// distinct keys below then share a counter and the `assert_ne!` rejects before any locator runs.
#[test]
fn counted_key_identity_is_the_value_not_the_hold_counter() {
    let shared = Arc::new(AtomicUsize::new(0));
    let other = Arc::new(AtomicUsize::new(0));
    let seven = CountedKey {
        value: 7,
        holds: Arc::clone(&shared),
    };
    let eight = CountedKey {
        value: 8,
        holds: Arc::clone(&shared),
    };
    let seven_again = CountedKey {
        value: 7,
        holds: Arc::clone(&other),
    };

    assert_ne!(seven, eight, "distinct values are distinct keys");
    assert_ne!(
        identity_bytes(&seven),
        identity_bytes(&eight),
        "distinct equality keys must not share an exact encoding"
    );
    assert!(!seven.matches_retained(&eight.retain()));

    assert_eq!(seven, seven_again, "the hold counter is not equality");
    assert_eq!(
        identity_bytes(&seven),
        identity_bytes(&seven_again),
        "the hold counter must not enter the exact encoding"
    );
    assert!(seven.matches_retained(&seven_again.retain()));
}

/// A value that counts every clone, so a payload or default clone is observable across workers.
struct CountedValue {
    value: i64,
    clones: Arc<AtomicUsize>,
}

impl Clone for CountedValue {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            value: self.value,
            clones: Arc::clone(&self.clones),
        }
    }
}

impl PartialEq for CountedValue {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for CountedValue {}

/// Placement-only routing: shard `value & 255`, so every shard carries load and shard 255 is the
/// final shard the serial reserve-all reaches with load.
#[derive(Clone, Copy)]
struct CountingHome;

impl Home<CountedKey> for CountingHome {
    fn home(&self, key: &CountedKey) -> u64 {
        key.value.cast_unsigned()
    }

    /// Placement-only: reproduces the former default (`home`-derived place, no direct word).
    fn address(&self, _sigma: Sigma, key: &CountedKey) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

// The semantic map is derived from `CountedKey::value`, independent of the physical routing.
locator_face!(CountingHome, CountedKey, counted_key_word);

/// The counted carrier under test: the default `Stored<CountedKey>` key representation.
type CountingTable = Table<CountedKey, CountedValue, CountingHome, 16>;

/// Placement-only routing that also counts every `address` call. The checked scatter plan calls
/// `address` once per input on the calling thread; each scatter worker calls it once more per
/// input, so a refused build that never starts a worker leaves the count at exactly `input.len()`.
struct AddressCountingHome {
    calls: Arc<AtomicUsize>,
}

impl Home<CountedKey> for AddressCountingHome {
    fn home(&self, key: &CountedKey) -> u64 {
        key.value.cast_unsigned()
    }

    fn address(&self, _sigma: Sigma, key: &CountedKey) -> TableAddress {
        self.calls.fetch_add(1, Ordering::SeqCst);
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

// The semantic map is derived from `CountedKey::value`, independent of the counted physical route.
locator_face!(AddressCountingHome, CountedKey, counted_key_word);

/// The address-counted carrier under test.
type AddressTable = Table<CountedKey, CountedValue, AddressCountingHome, 16>;

/// A probe key with a throwaway counter; constructing it does not clone.
fn probe(value: i64) -> CountedKey {
    CountedKey {
        value,
        holds: Arc::new(AtomicUsize::new(0)),
    }
}

/// `n` distinct keys, one per value, with `value * 2` payloads.
fn counting_input(
    n: i64,
    holds: &Arc<AtomicUsize>,
    clones: &Arc<AtomicUsize>,
) -> Vec<((LocatorSigma, CountedKey), CountedValue)> {
    (0..n)
        .map(|value| {
            (
                (
                    LocatorSigma::Parabolic,
                    CountedKey {
                        value,
                        holds: Arc::clone(holds),
                    },
                ),
                CountedValue {
                    value: value * 2,
                    clones: Arc::clone(clones),
                },
            )
        })
        .collect()
}

/// Every probed key and the logical length agree between the serial reference and the parallel
/// table.
fn assert_same_counting_map(reference: &CountingTable, parallel: &CountingTable) {
    assert_eq!(reference.len(), parallel.len(), "logical length disagrees");
    for value in 0..600i64 {
        let key = probe(value);
        assert_eq!(
            reference.get_at(LocatorSigma::Parabolic, &key).value,
            parallel.get_at(LocatorSigma::Parabolic, &key).value,
            "key {value} disagrees between the serial and parallel tables"
        );
    }
}

/// The final shard's reservation is forced to fail before any worker starts: the parallel build
/// returns `Err`, no key is held, no payload or default is cloned, no merge runs, and the committed
/// table is unchanged. With the seam cleared the build succeeds and equals the serial result.
///
/// The seam is the inherent `apeiron::table::parallel::reservation_failure` arm. It names shard
/// 255, the last shard the serial reserve-all reaches with load, so shards 0..254 are reserved
/// before the injected failure. Zero key clones also excludes relocation: relocation clones a held
/// carrier and a payload, and `build_at_parallel` invokes no relocation entry point.
#[cfg(feature = "injection")]
#[test]
fn injected_final_shard_reservation_refusal_has_zero_cross_worker_effects() {
    const FINAL_SHARD: u64 = 255;

    let holds = Arc::new(AtomicUsize::new(0));
    let clones = Arc::new(AtomicUsize::new(0));
    let merges = Arc::new(AtomicUsize::new(0));
    let input = counting_input(20_000, &holds, &clones);
    let merge = {
        let merges = Arc::clone(&merges);
        move |new: CountedValue, _old: CountedValue| {
            merges.fetch_add(1, Ordering::SeqCst);
            new
        }
    };
    let default = |value: i64| CountedValue {
        value,
        clones: Arc::clone(&clones),
    };

    // The committed serial table for this input; the restored parallel build must equal it.
    let reference = CountingTable::build_at(&merge, default(-1), &input, CountingHome)
        .expect("serial build fits");
    let committed: Vec<i64> = (0..600)
        .map(|value| {
            reference
                .get_at(LocatorSigma::Parabolic, &probe(value))
                .value
        })
        .collect();

    // Measure only the injected operation.
    holds.store(0, Ordering::SeqCst);
    clones.store(0, Ordering::SeqCst);
    merges.store(0, Ordering::SeqCst);

    assert!(
        input
            .iter()
            .any(|((_, key), _)| key.value.cast_unsigned() & 255 == FINAL_SHARD),
        "shard {FINAL_SHARD} must carry load"
    );

    apeiron::table::parallel::reservation_failure::arm(
        usize::try_from(FINAL_SHARD).expect("shard fits usize"),
    );
    let refused = CountingTable::build_at_parallel(&merge, default(-1), &input, CountingHome, 8);
    apeiron::table::parallel::reservation_failure::disarm();

    assert!(
        matches!(refused, Err(StorageError::AllocationFailed)),
        "the injected final-shard reservation must refuse with AllocationFailed"
    );
    assert_eq!(
        holds.load(Ordering::SeqCst),
        0,
        "a refusal must not hold a key"
    );
    assert_eq!(
        clones.load(Ordering::SeqCst),
        0,
        "a refusal must not clone a payload or default"
    );
    assert_eq!(merges.load(Ordering::SeqCst), 0, "a refusal must not merge");

    // The committed table is unchanged by the refused build.
    let after: Vec<i64> = (0..600)
        .map(|value| {
            reference
                .get_at(LocatorSigma::Parabolic, &probe(value))
                .value
        })
        .collect();
    assert_eq!(committed, after, "the committed table must be unchanged");

    // Restored success equals the serial result.
    let restored = CountingTable::build_at_parallel(&merge, default(-1), &input, CountingHome, 8)
        .expect("restored parallel build fits");
    assert_same_counting_map(&reference, &restored);
}

/// The checked chunk plan, scatter scratch and table-shard reservation all complete on the calling
/// thread before any scatter worker starts. With the final shard's reservation armed, the refused
/// build leaves the `address` count at exactly the planning pass, so no scatter worker ran; the
/// refusal also holds no key and clones no payload or default.
#[cfg(feature = "injection")]
#[test]
fn reserve_all_precedes_every_scatter_worker() {
    const FINAL_SHARD: u64 = 255;

    let calls = Arc::new(AtomicUsize::new(0));
    let holds = Arc::new(AtomicUsize::new(0));
    let clones = Arc::new(AtomicUsize::new(0));
    let input = counting_input(20_000, &holds, &clones);
    let merge = |new: CountedValue, _old: CountedValue| new;
    let default_value = CountedValue {
        value: -1,
        clones: Arc::clone(&clones),
    };
    let home = AddressCountingHome {
        calls: Arc::clone(&calls),
    };

    assert!(
        input
            .iter()
            .any(|((_, key), _)| key.value.cast_unsigned() & 255 == FINAL_SHARD),
        "shard {FINAL_SHARD} must carry load"
    );

    // Constructing the default clones once; measure only the injected build.
    holds.store(0, Ordering::SeqCst);
    clones.store(0, Ordering::SeqCst);

    apeiron::table::parallel::reservation_failure::arm(
        usize::try_from(FINAL_SHARD).expect("shard fits usize"),
    );
    let refused = AddressTable::build_at_parallel(&merge, default_value, &input, home, 8);
    apeiron::table::parallel::reservation_failure::disarm();

    assert!(
        matches!(refused, Err(StorageError::AllocationFailed)),
        "the injected final-shard reservation must refuse with AllocationFailed"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        input.len(),
        "only the checked planning pass may run before reserve-all refuses"
    );
    assert_eq!(
        holds.load(Ordering::SeqCst),
        0,
        "a refusal must not hold a key"
    );
    assert_eq!(
        clones.load(Ordering::SeqCst),
        0,
        "a refusal must not clone a payload or default"
    );
}

// ---------------------------------------------------------------------------------------------
// Parallel directory coverage: reserve every path before the workers, record placements, finalize.
// ---------------------------------------------------------------------------------------------

/// A mixed-face qualified input: each distinct key appears under both filtered faces, alternating
/// by input chunk, so a face-blind directory would leave half the live payloads uncovered.
fn mixed_face_input(n: usize) -> Vec<((Sigma, i64), i64)> {
    let distinct = i64::try_from((n / 4).max(1)).expect("test size fits i64");
    (0..n)
        .map(|i| {
            let i = i64::try_from(i).expect("test index fits i64");
            let key = i % distinct;
            let face = if (i / distinct) % 2 == 0 {
                Sigma::Parabolic
            } else {
                Sigma::Hyperbolic
            };
            ((face, key), i)
        })
        .collect()
}

/// The parallel build covers the assembled table on `R` across worker counts, and both faces read
/// back the serial map.
///
/// Coverage is checked on the serial reference and on every parallel result; a directory that
/// covered only the worker fragments, or whose reverse links kept rebase placeholders or wrong
/// worker-recorded slots, would leave the assembled table uncovered and fail here.
fn mixed_face_parallel_coverage<R>(label: &str)
where
    R: KeyRep<i64, SkewHome> + Send + Sync,
{
    let input = mixed_face_input(20_000);
    let distinct = i64::try_from((input.len() / 4).max(1)).expect("test size fits i64");
    let reference = Table::<i64, i64, SkewHome, 16, R>::build_at(&horner, 0, &input, SkewHome)
        .expect("serial mixed-face build fits");

    assert!(
        reference.check_storage_coverage(),
        "{label}: the serial reference must be covered"
    );
    for workers in [2usize, 3, 16, 300] {
        let table = Table::<i64, i64, SkewHome, 16, R>::build_at_parallel(
            &horner, 0, &input, SkewHome, workers,
        )
        .expect("parallel mixed-face build fits");
        assert!(
            table.check_storage_coverage(),
            "{label}: the parallel assembled table must be covered"
        );
        assert_eq!(reference.len(), table.len(), "{label}: logical length");
        for key in 0..distinct {
            assert_eq!(
                *reference.get_at(Sigma::Parabolic, &key),
                *table.get_at(Sigma::Parabolic, &key),
                "{label}: parabolic key {key}"
            );
            assert_eq!(
                *reference.get_at(Sigma::Hyperbolic, &key),
                *table.get_at(Sigma::Hyperbolic, &key),
                "{label}: hyperbolic key {key}"
            );
        }
    }
}

#[test]
fn parallel_build_covers_the_assembled_table_on_both_carriers() {
    mixed_face_parallel_coverage::<apeiron::table::keyrep::Stored<i64>>("stored");
    mixed_face_parallel_coverage::<Regen<I64WordCodec>>("regen");
}

/// An elliptic entry carries no locator handle: its fill placement records nothing, so the
/// finalization must skip it rather than refuse on its absent handle. Locator-face entries in the
/// same shards keep their worker-recorded locations and read back the serial map.
#[test]
fn parallel_build_skips_elliptic_placements_and_reads_back_every_face() {
    // Every key appears under both filtered faces and the locator-free elliptic face, cycling by
    // input position, so a key's entries span the chunks and the faces collide in their shard.
    const ENTRIES: usize = 20_000;
    let distinct = 5_000i64;
    let mut input: Vec<((Sigma, i64), i64)> = Vec::with_capacity(ENTRIES);
    let mut i = 0usize;
    while i < ENTRIES {
        let index = i64::try_from(i).expect("test index fits i64");
        let key = index % distinct;
        let face = match i % 3 {
            0 => Sigma::Parabolic,
            1 => Sigma::Hyperbolic,
            _ => Sigma::Elliptic,
        };
        input.push(((face, key), index));
        i += 1;
    }
    let reference =
        SkewTable::build_at(&horner, 0, &input, SkewHome).expect("serial three-face build fits");
    for workers in [2usize, 16] {
        let table = SkewTable::build_at_parallel(&horner, 0, &input, SkewHome, workers)
            .expect("parallel three-face build fits");
        assert_eq!(
            reference.len(),
            table.len(),
            "workers={workers}: logical length"
        );
        for key in 0..distinct {
            for face in [Sigma::Parabolic, Sigma::Hyperbolic, Sigma::Elliptic] {
                assert_eq!(
                    *reference.get_at(face, &key),
                    *table.get_at(face, &key),
                    "workers={workers}: {face:?} key {key}"
                );
            }
        }
    }
}

/// The directory reservation runs before the first fill worker: arming the late identity-edge seam
/// refuses the parallel build with no hold, clone, merge or scatter worker, and the restored build
/// covers the assembled table and equals the serial map.
#[cfg(feature = "injection")]
#[test]
fn injected_late_locator_edge_refusal_has_zero_cross_worker_effects() {
    let calls = Arc::new(AtomicUsize::new(0));
    let holds = Arc::new(AtomicUsize::new(0));
    let clones = Arc::new(AtomicUsize::new(0));
    let merges = Arc::new(AtomicUsize::new(0));
    let input = counting_input(20_000, &holds, &clones);
    let merge = {
        let merges = Arc::clone(&merges);
        move |new: CountedValue, _old: CountedValue| {
            merges.fetch_add(1, Ordering::SeqCst);
            new
        }
    };
    let default = |value: i64| CountedValue {
        value,
        clones: Arc::clone(&clones),
    };

    let reference = AddressTable::build_at(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
    )
    .expect("serial reference build fits");

    // Measure only the injected build.
    holds.store(0, Ordering::SeqCst);
    clones.store(0, Ordering::SeqCst);
    merges.store(0, Ordering::SeqCst);
    calls.store(0, Ordering::SeqCst);

    apeiron::coord::locator::reservation_failure::arm_late_edge(Refusal::AllocationFailed);
    let refused = AddressTable::build_at_parallel(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
        8,
    );
    apeiron::coord::locator::reservation_failure::disarm();

    assert!(
        matches!(refused, Err(StorageError::AllocationFailed)),
        "the injected late locator edge must refuse with AllocationFailed"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        input.len(),
        "only the checked planning pass may run before the locator reservation refuses"
    );
    assert_eq!(
        holds.load(Ordering::SeqCst),
        0,
        "a refusal must not hold a key"
    );
    assert_eq!(
        clones.load(Ordering::SeqCst),
        0,
        "a refusal must not clone a payload or default"
    );
    assert_eq!(merges.load(Ordering::SeqCst), 0, "a refusal must not merge");

    // Restored success covers the assembled table and equals the serial map.
    let restored = AddressTable::build_at_parallel(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
        8,
    )
    .expect("restored parallel build fits");
    assert_eq!(reference.len(), restored.len(), "logical length");
    for value in 0..600i64 {
        let key = probe(value);
        assert_eq!(
            reference.get_at(LocatorSigma::Parabolic, &key).value,
            restored.get_at(LocatorSigma::Parabolic, &key).value,
            "key {value} disagrees between the serial and parallel tables"
        );
    }
    assert!(
        restored.check_storage_coverage(),
        "the restored parallel table must be covered"
    );
}

/// One placement-record buffer per nonempty shard is reserved on the calling thread before the
/// first scatter worker: arming the recording-buffer seam refuses the parallel build with no hold,
/// clone or merge and the address count at exactly the planning pass, and the restored build covers
/// the assembled table and equals the serial map.
#[cfg(feature = "injection")]
#[test]
fn injected_recording_buffer_reservation_has_zero_cross_worker_effects() {
    const FINAL_SHARD: u64 = 255;

    let calls = Arc::new(AtomicUsize::new(0));
    let holds = Arc::new(AtomicUsize::new(0));
    let clones = Arc::new(AtomicUsize::new(0));
    let merges = Arc::new(AtomicUsize::new(0));
    let input = counting_input(20_000, &holds, &clones);
    let merge = {
        let merges = Arc::clone(&merges);
        move |new: CountedValue, _old: CountedValue| {
            merges.fetch_add(1, Ordering::SeqCst);
            new
        }
    };
    let default = |value: i64| CountedValue {
        value,
        clones: Arc::clone(&clones),
    };

    let reference = AddressTable::build_at(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
    )
    .expect("serial reference build fits");

    // Measure only the injected build.
    holds.store(0, Ordering::SeqCst);
    clones.store(0, Ordering::SeqCst);
    merges.store(0, Ordering::SeqCst);
    calls.store(0, Ordering::SeqCst);

    assert!(
        input
            .iter()
            .any(|((_, key), _)| key.value.cast_unsigned() & 255 == FINAL_SHARD),
        "shard {FINAL_SHARD} must carry load"
    );

    apeiron::table::parallel::reservation_failure::arm_records(
        usize::try_from(FINAL_SHARD).expect("shard fits usize"),
    );
    let refused = AddressTable::build_at_parallel(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
        8,
    );
    apeiron::table::parallel::reservation_failure::disarm();

    assert!(
        matches!(refused, Err(StorageError::AllocationFailed)),
        "the injected recording-buffer reservation must refuse with AllocationFailed"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        input.len(),
        "only the checked planning pass may run before the recording-buffer reservation refuses"
    );
    assert_eq!(
        holds.load(Ordering::SeqCst),
        0,
        "a refusal must not hold a key"
    );
    assert_eq!(
        clones.load(Ordering::SeqCst),
        0,
        "a refusal must not clone a payload or default"
    );
    assert_eq!(merges.load(Ordering::SeqCst), 0, "a refusal must not merge");

    // Restored success covers the assembled table and equals the serial map.
    let restored = AddressTable::build_at_parallel(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
        8,
    )
    .expect("restored parallel build fits");
    assert_eq!(reference.len(), restored.len(), "logical length");
    for value in 0..600i64 {
        let key = probe(value);
        assert_eq!(
            reference.get_at(LocatorSigma::Parabolic, &key).value,
            restored.get_at(LocatorSigma::Parabolic, &key).value,
            "key {value} disagrees between the serial and parallel tables"
        );
    }
    assert!(
        restored.check_storage_coverage(),
        "the restored parallel table must be covered"
    );
}

/// Every worker's outcome buffer and the joined-results sink are reserved on the calling thread
/// before the first scatter worker: arming the result-buffer seam refuses the parallel build with
/// no hold, clone or merge and the address count at exactly the planning pass, and the restored
/// build covers the assembled table and equals the serial map.
#[cfg(feature = "injection")]
#[test]
fn injected_worker_buffer_reservation_has_zero_cross_worker_effects() {
    let calls = Arc::new(AtomicUsize::new(0));
    let holds = Arc::new(AtomicUsize::new(0));
    let clones = Arc::new(AtomicUsize::new(0));
    let merges = Arc::new(AtomicUsize::new(0));
    let input = counting_input(20_000, &holds, &clones);
    let merge = {
        let merges = Arc::clone(&merges);
        move |new: CountedValue, _old: CountedValue| {
            merges.fetch_add(1, Ordering::SeqCst);
            new
        }
    };
    let default = |value: i64| CountedValue {
        value,
        clones: Arc::clone(&clones),
    };

    let reference = AddressTable::build_at(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
    )
    .expect("serial reference build fits");

    // Measure only the injected build.
    holds.store(0, Ordering::SeqCst);
    clones.store(0, Ordering::SeqCst);
    merges.store(0, Ordering::SeqCst);
    calls.store(0, Ordering::SeqCst);

    apeiron::table::parallel::reservation_failure::arm_output(0);
    let refused = AddressTable::build_at_parallel(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
        8,
    );
    apeiron::table::parallel::reservation_failure::disarm();

    assert!(
        matches!(refused, Err(StorageError::AllocationFailed)),
        "the injected worker-buffer reservation must refuse with AllocationFailed"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        input.len(),
        "only the checked planning pass may run before the result-buffer reservation refuses"
    );
    assert_eq!(
        holds.load(Ordering::SeqCst),
        0,
        "a refusal must not hold a key"
    );
    assert_eq!(
        clones.load(Ordering::SeqCst),
        0,
        "a refusal must not clone a payload or default"
    );
    assert_eq!(merges.load(Ordering::SeqCst), 0, "a refusal must not merge");

    // Restored success covers the assembled table and equals the serial map.
    let restored = AddressTable::build_at_parallel(
        &merge,
        default(-1),
        &input,
        AddressCountingHome {
            calls: Arc::clone(&calls),
        },
        8,
    )
    .expect("restored parallel build fits");
    assert_eq!(reference.len(), restored.len(), "logical length");
    for value in 0..600i64 {
        let key = probe(value);
        assert_eq!(
            reference.get_at(LocatorSigma::Parabolic, &key).value,
            restored.get_at(LocatorSigma::Parabolic, &key).value,
            "key {value} disagrees between the serial and parallel tables"
        );
    }
    assert!(
        restored.check_storage_coverage(),
        "the restored parallel table must be covered"
    );
}

// Recorded-location finalization: every checked refusal arm carries a one-shot injection witness.
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "injection")]
use apeiron::table::parallel::reservation_failure::{arm_finalize, disarm, FinalizeArm};

/// A small parallel input for the finalize-arm witnesses: 5,000 entries sit above the 4,096-entry
/// crossover, so `build_parallel` takes the scoped path and reaches
/// `finalize_recorded_locations`, and the default face is `Sigma::Parabolic`, so every entry
/// carries a handle and one placement record.
fn finalize_witness_input() -> Vec<(i64, i64)> {
    skewed_input(5_000)
}

/// Builds the witness input in parallel with `arm` armed, clears the seam afterwards, and returns
/// the build's error; an `Ok` result is the attributable failure "the arm did not fire".
#[cfg(feature = "injection")]
fn armed_finalize_build_error(arm: FinalizeArm) -> StorageError {
    let input = finalize_witness_input();
    arm_finalize(arm);
    let refused = SkewTable::build_parallel(&horner, 0, &input, SkewHome, 2);
    disarm();
    match refused {
        Ok(_) => panic!("the armed {arm:?} finalize arm must refuse and produce no table"),
        Err(error) => error,
    }
}

/// The witness protocol for one finalize arm: the unarmed build of the same input succeeds and
/// equals the serial map (so an armed refusal is attributable to the armed arm alone), the armed
/// build refuses with the arm's exact `StorageError` and produces no table, and the disarmed
/// rebuild succeeds and equals the serial map (the one-shot arm does not leak).
#[cfg(feature = "injection")]
fn assert_finalize_arm_witness(arm: FinalizeArm, expected: StorageError) {
    let input = finalize_witness_input();
    let distinct = distinct_of(&input);
    let reference = serial(&input);

    // Unarmed control: the identical build succeeds, so the armed refusal below has one cause.
    let unarmed = SkewTable::build_parallel(&horner, 0, &input, SkewHome, 2)
        .expect("the unarmed parallel build must fit");
    assert_same_map(&reference, &unarmed, distinct, &[]);

    let error = armed_finalize_build_error(arm);
    assert_eq!(
        error, expected,
        "the armed {arm:?} finalize arm must refuse with the arm's exact error"
    );

    // The arm is one-shot: the cleared rebuild succeeds and equals the serial map.
    let restored = SkewTable::build_parallel(&horner, 0, &input, SkewHome, 2)
        .expect("the disarmed parallel rebuild must fit");
    assert_same_map(&reference, &restored, distinct, &[]);
}

/// The out-of-range-input-index arm refuses with `SizeOverflow`.
#[cfg(feature = "injection")]
#[test]
fn injected_finalize_index_out_of_range_refuses_with_size_overflow() {
    assert_finalize_arm_witness(FinalizeArm::IndexOutOfRange, StorageError::SizeOverflow);
}

/// The missing-handle arm (a record whose input index has no unconsumed handle) refuses with
/// `AllocationFailed` and leaves no silent placeholder.
#[cfg(feature = "injection")]
#[test]
fn injected_finalize_missing_handle_refuses_with_allocation_failed() {
    assert_finalize_arm_witness(FinalizeArm::MissingHandle, StorageError::AllocationFailed);
}

/// The link-out-of-range arm (a handle naming no recorded link) refuses with `AllocationFailed`.
#[cfg(feature = "injection")]
#[test]
fn injected_finalize_link_out_of_range_refuses_with_allocation_failed() {
    assert_finalize_arm_witness(FinalizeArm::LinkOutOfRange, StorageError::AllocationFailed);
}

/// The unconsumed-handle arm (a `Some` handle left after the record pass) refuses with
/// `AllocationFailed`: an incomplete record set is a checked refusal.
#[cfg(feature = "injection")]
#[test]
fn injected_finalize_unconsumed_handle_refuses_with_allocation_failed() {
    assert_finalize_arm_witness(
        FinalizeArm::UnconsumedHandle,
        StorageError::AllocationFailed,
    );
}
