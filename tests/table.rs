use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue, H3_40,
};
use apeiron::coord::{spread, Packed};
use apeiron::table::keyrep::{I64WordCodec, KeyRep, Regen, Stored};
use apeiron::table::{
    AffineFoldStep, AffineMixStep, AffineStep, Home, Reduce, SearchClass, SearchLocation,
    TableAddress, WordHome,
};
use apeiron::{merge, table::ReplacementMerge, Cell, Sigma, Table};

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

/// Placement-only home for `i64` keys: one placement word with no direct lane.
#[derive(Clone, Copy)]
struct PlacementOnlyI64Home;

impl Home<i64> for PlacementOnlyI64Home {
    fn home(&self, key: &i64) -> u64 {
        0x1F00u64 ^ key.cast_unsigned()
    }

    /// Placement-only: `home`-derived place, no direct word.
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

/// Zero-placement home for `i32` keys.
#[derive(Clone, Copy)]
struct ZeroI32Home;

impl Home<i32> for ZeroI32Home {
    fn home(&self, _key: &i32) -> u64 {
        0
    }

    /// Placement-only: `home`-derived place, no direct word.
    fn address(&self, _sigma: Sigma, key: &i32) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

/// Identity-placement home for `i32` keys.
#[derive(Clone, Copy)]
struct IdentityI32Home;

impl Home<i32> for IdentityI32Home {
    fn home(&self, key: &i32) -> u64 {
        u64::from(key.cast_unsigned())
    }

    /// Placement-only: `home`-derived place, no direct word.
    fn address(&self, _sigma: Sigma, key: &i32) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

/// Zero-placement home for `i64` keys.
#[derive(Clone, Copy)]
struct ZeroI64Home;

impl Home<i64> for ZeroI64Home {
    fn home(&self, _key: &i64) -> u64 {
        0
    }

    /// Placement-only: `home`-derived place, no direct word.
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

/// Identity-placement home for `i64` keys.
#[derive(Clone, Copy)]
struct IdentityI64Home;

impl Home<i64> for IdentityI64Home {
    fn home(&self, key: &i64) -> u64 {
        key.cast_unsigned()
    }

    /// Placement-only: `home`-derived place, no direct word.
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

/// Shifted-placement home for `i64` keys: the merge-count fixture's placement.
#[derive(Clone, Copy)]
struct ShiftI64Home;

impl Home<i64> for ShiftI64Home {
    fn home(&self, key: &i64) -> u64 {
        7 + (key.cast_unsigned() << 8)
    }

    /// Placement-only: `home`-derived place, no direct word.
    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

/// Identity-placement home for `u64` keys.
#[derive(Clone, Copy)]
struct IdentityU64Home;

impl Home<u64> for IdentityU64Home {
    fn home(&self, key: &u64) -> u64 {
        *key
    }

    /// Placement-only: `home`-derived place, no direct word.
    fn address(&self, _sigma: Sigma, key: &u64) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

sigma_word_face!(PlacementOnlyI64Home, i64);
sigma_word_face!(ZeroI32Home, i32);
sigma_word_face!(IdentityI32Home, i32);
sigma_word_face!(ZeroI64Home, i64);
sigma_word_face!(IdentityI64Home, i64);
sigma_word_face!(ShiftI64Home, i64);
sigma_word_face!(IdentityU64Home, u64);

/// A named placement-only home, which admits no direct lane.
///
/// Control 1: the physical address has `direct_word: None`, so the table never fabricates a direct
/// candidate from `place`.
#[test]
fn placement_only_home_returns_no_direct_word() {
    let address = Home::address(&PlacementOnlyI64Home, Sigma::Parabolic, &7);
    assert_eq!(address.place, 0x1F00 ^ 7);
    assert_eq!(address.direct_word, None);
}

/// Control 2: `WordHome` certifies the exact word only below `2^32`.
#[test]
fn word_home_certifies_the_word_only_below_two_to_the_thirty_two() {
    let inside = Home::address(&WordHome, Sigma::Elliptic, &((1i64 << 32) - 1));
    assert_eq!(inside.direct_word, Some(u32::MAX));
    let edge = Home::address(&WordHome, Sigma::Hyperbolic, &(1i64 << 32));
    assert_eq!(edge.direct_word, None);
}

/// Both carriers retain the exact original-bit identity, never a reduced face residue.
///
/// Keys `0` and `q = 3^40` are distinct but share the hyperbolic residue `0`, so a residue-derived
/// `Regen` identity would collide. The held bytes of both carriers must agree with the query's
/// canonical bytes and separate the two keys.
#[test]
fn both_carriers_agree_on_exact_identity_bytes() {
    fn query_bytes<R: KeyRep<i64, WordHome>>(key: i64) -> Vec<u8> {
        let mut out = Vec::new();
        let mut index = 0;
        while index < R::query_encoding_len(&key) {
            out.push(R::query_encoding_byte(&key, index));
            index += 1;
        }
        out
    }
    fn held_bytes<R: KeyRep<i64, WordHome>>(carrier: &R) -> Vec<u8> {
        let mut out = Vec::new();
        let mut index = 0;
        while index < R::held_encoding_len(carrier) {
            out.push(R::held_encoding_byte(carrier, index));
            index += 1;
        }
        out
    }

    let profile = H3_40::new().expect("the frozen H3_40 constants validate");
    let sigma = Sigma::Hyperbolic;
    let zero = 0i64;
    let q_key = H3_40::Q.cast_signed();
    assert_ne!(zero, q_key, "0 and q are distinct keys");
    assert_eq!(
        Residue::reduce(profile, zero.cast_unsigned()),
        Residue::reduce(profile, q_key.cast_unsigned()),
        "0 and q share the hyperbolic residue 0"
    );

    let stored_zero = <Stored<i64> as KeyRep<i64, WordHome>>::hold(
        sigma,
        &zero,
        &Home::address(&WordHome, sigma, &zero),
    );
    let stored_q = <Stored<i64> as KeyRep<i64, WordHome>>::hold(
        sigma,
        &q_key,
        &Home::address(&WordHome, sigma, &q_key),
    );
    let regen_zero = <Regen<I64WordCodec> as KeyRep<i64, WordHome>>::hold(
        sigma,
        &zero,
        &Home::address(&WordHome, sigma, &zero),
    );
    let regen_q = <Regen<I64WordCodec> as KeyRep<i64, WordHome>>::hold(
        sigma,
        &q_key,
        &Home::address(&WordHome, sigma, &q_key),
    );

    // Query and held bytes agree for both carriers.
    assert_eq!(query_bytes::<Stored<i64>>(zero), held_bytes(&stored_zero));
    assert_eq!(query_bytes::<Stored<i64>>(q_key), held_bytes(&stored_q));
    assert_eq!(
        query_bytes::<Regen<I64WordCodec>>(zero),
        held_bytes(&regen_zero)
    );
    assert_eq!(
        query_bytes::<Regen<I64WordCodec>>(q_key),
        held_bytes(&regen_q)
    );
    // The exact identity separates keys that share a residue.
    assert_ne!(held_bytes(&stored_zero), held_bytes(&stored_q));
    assert_ne!(held_bytes(&regen_zero), held_bytes(&regen_q));

    // The semantic face address agrees with the home's capability for both carriers.
    assert_eq!(
        <Stored<i64> as KeyRep<i64, WordHome>>::face_address_held(
            &WordHome,
            profile,
            sigma,
            &stored_zero
        ),
        FaceHome::address(&WordHome, profile, sigma, &zero)
    );
    assert_eq!(
        <Regen<I64WordCodec> as KeyRep<i64, WordHome>>::face_address_held(
            &WordHome,
            profile,
            sigma,
            &regen_zero
        ),
        FaceHome::address(&WordHome, profile, sigma, &zero)
    );
}

/// Control 3: the direct lane is the coordinate high part, not `place >> 8`.
#[test]
fn the_direct_lane_is_the_coordinate_not_the_placement() {
    let key = 0x1234_5678i64;
    let address = Home::address(&WordHome, Sigma::Parabolic, &key);
    let word = key.cast_unsigned();
    assert_eq!(
        address.direct_word,
        Some(u32::try_from(word).expect("0x1234_5678 fits u32"))
    );
    let lane = u64::from(address.direct_word.expect("certified below 2^32")) >> 8;
    assert_eq!(lane, word >> 8);
    assert_ne!(
        lane,
        address.place >> 8,
        "the placement high part must not be the direct lane"
    );
}

/// A home that certifies one direct lane for every key, so the direct candidate is occupied by a
/// different key and the search must fall through to the window/sweep.
#[derive(Clone, Copy)]
struct SameLaneHome;

impl Home<i64> for SameLaneHome {
    fn home(&self, _key: &i64) -> u64 {
        0
    }

    fn address(&self, _sigma: Sigma, _key: &i64) -> TableAddress {
        TableAddress {
            place: 0,
            direct_word: Some(0),
        }
    }
}

sigma_word_face!(SameLaneHome, i64);

/// A home that certifies direct lane `k` for key `k`, all in one region.
#[derive(Clone, Copy)]
struct DirectHome;

impl Home<i64> for DirectHome {
    fn home(&self, _key: &i64) -> u64 {
        0
    }

    fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
        TableAddress {
            place: 0,
            direct_word: u32::try_from(key.cast_unsigned() << 8).ok(),
        }
    }
}

sigma_word_face!(DirectHome, i64);

#[test]
fn default_read_is_total_before_any_write() {
    let table = Table::<i32, _, _>::new(String::from("empty"), ZeroI32Home);
    assert_eq!(table.get(&42), "empty");
}

#[test]
fn insert_is_new_first_and_replaces_an_existing_override() {
    let mut table = Table::<i32, _, _>::new(10_i32, ZeroI32Home);
    table.insert_with(&|new, old| new - old, 7, 3).unwrap();
    assert_eq!(*table.get(&7), -7);
    table.insert_with(&|new, old| new - old, 7, 5).unwrap();
    assert_eq!(*table.get(&7), 12);
}

#[test]
fn replacement_merge_fresh_write_stores_new_value() {
    let mut table = Table::<i64, _, _, 1>::new(-17, ZeroI64Home);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    assert_eq!(*table.get(&7), 41);
}

#[test]
fn replacement_merge_present_write_stores_new_value() {
    let mut table = Table::<i64, _, _, 1>::new(-17, ZeroI64Home);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    table.insert_with(&ReplacementMerge, 7, -9).unwrap();
    assert_eq!(*table.get(&7), -9);
}

#[test]
fn growth_admits_far_beyond_the_old_fixed_ceiling() {
    let mut table = Table::<u64, u64, _, 16>::new(0, IdentityU64Home);
    for k in 0..100_000u64 {
        table
            .insert_with(&|new, _old| new, k, k.wrapping_mul(2).wrapping_add(1))
            .unwrap();
    }
    assert_eq!(table.len(), 100_000);
    for k in 0..100_000u64 {
        assert_eq!(*table.get(&k), k.wrapping_mul(2).wrapping_add(1));
    }
    assert_eq!(*table.get(&100_000), 0);
}

#[test]
fn build_preserves_each_keys_stream_order() {
    let table = Table::<i32, _, _>::build(
        &merge::link,
        Vec::new(),
        &[(1, vec![1]), (2, vec![9]), (1, vec![2])],
        ZeroI32Home,
    )
    .unwrap();
    assert_eq!(table.get(&1), &[2, 1]);
    assert_eq!(table.get(&2), &[9]);
}

/// The partitioned build consumes the worker chunk-runs in their original order.
/// Reversing those runs changes this non-commutative merge's result.
#[test]
fn partitioned_build_matches_the_sequential_non_commutative_oracle() {
    let mut input = Vec::new();
    for i in 0..5000_i32 {
        input.push((i % 2, vec![i]));
    }
    let partitioned = Table::<i32, _, _>::build(&merge::link, Vec::new(), &input, IdentityI32Home)
        .expect("partitioned build");
    let mut oracle = Table::<i32, _, _>::new(Vec::new(), IdentityI32Home);
    for (key, value) in &input {
        oracle
            .insert_with(&merge::link, *key, value.clone())
            .unwrap();
    }
    assert_eq!(partitioned.get(&0), oracle.get(&0));
    assert_eq!(partitioned.get(&1), oracle.get(&1));
    let mut reversed_runs = Table::<i32, _, _>::new(Vec::new(), IdentityI32Home);
    for chunk in input.chunks(2500).rev() {
        for (key, value) in chunk {
            reversed_runs
                .insert_with(&merge::link, *key, value.clone())
                .unwrap();
        }
    }
    assert_ne!(partitioned.get(&0), reversed_runs.get(&0));
}

#[test]
fn scoped_parallel_build_matches_one_worker_for_a_non_commutative_merge() {
    let input: Vec<_> = (0..6000_i32).map(|i| (i % 7, vec![i])).collect();
    let home = IdentityI32Home;
    let one = Table::<i32, _, _, 16>::build_parallel(&merge::link, Vec::new(), &input, home, 1)
        .expect("one-worker build");
    let sixteen = Table::<i32, _, _, 16>::build_parallel(
        &merge::link,
        Vec::new(),
        &input,
        IdentityI32Home,
        16,
    )
    .expect("sixteen-worker build");
    for key in 0..7 {
        assert_eq!(one.get(&key), sixteen.get(&key));
    }
}

#[test]
fn scoped_parallel_build_propagates_a_worker_panic_without_a_table() {
    let input: Vec<_> = (0..6000_i64).map(|i| (i % 3, i)).collect();
    let result = std::panic::catch_unwind(|| {
        let _ = Table::<i64, i64, _, 16>::build_parallel(
            &|new: i64, old: i64| {
                assert_ne!(new, 13, "worker merge panic");
                new.wrapping_add(old)
            },
            0,
            &input,
            IdentityI64Home,
            16,
        );
    });
    assert!(
        result.is_err(),
        "a panicking worker cannot return a partial table"
    );
}

#[test]
fn grouped_probe_restores_query_order_before_a_non_commutative_callback() {
    let input: Vec<_> = (0..32_i64).map(|key| (key, key + 1)).collect();
    let table =
        Table::<i64, i64, _, 16>::build_parallel(&ReplacementMerge, 0, &input, IdentityI64Home, 16)
            .unwrap();
    let queries: Vec<_> = (0..6000_i64).map(|i| (i * 11) % 32).collect();
    let mut callback_fold = 0_i64;
    let values = table.get_each_parallel(&queries, 16, |value| {
        callback_fold = callback_fold.wrapping_mul(31).wrapping_add(*value);
        *value
    });
    assert_eq!(
        values,
        queries.iter().map(|key| key + 1).collect::<Vec<_>>()
    );
    let expected = queries
        .iter()
        .fold(0_i64, |acc, key| acc.wrapping_mul(31).wrapping_add(key + 1));
    assert_eq!(
        callback_fold, expected,
        "reversed grouped callbacks change this fold"
    );
}

#[test]
fn skewed_parallel_build_is_independent_of_worker_count() {
    let input: Vec<_> = (0..6000_i64).map(|key| (key, key + 1)).collect();
    let one =
        Table::<i64, i64, _, 16>::build_parallel(&ReplacementMerge, 0, &input, ZeroI64Home, 1)
            .unwrap();
    let sixteen =
        Table::<i64, i64, _, 16>::build_parallel(&ReplacementMerge, 0, &input, ZeroI64Home, 16)
            .unwrap();
    for key in [0, 17, 5999] {
        assert_eq!(one.get(&key), sixteen.get(&key));
    }
}

/// Control 2: two unequal keys with the same home are distinguished by full-key equality.
///
/// The home is constant, so every key has the same placement word, tag and shard. A table that
/// decided identity from the tag or the placement word would let one write overwrite the other.
#[test]
fn same_home_unequal_keys_are_distinguished_by_full_key_equality() {
    let mut table = Table::<i32, i32, _>::new(0, ZeroI32Home);
    table.insert_with(&|new, _old| new, 7, 41).unwrap();
    table.insert_with(&|new, _old| new, 8, 42).unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!(*table.get(&7), 41);
    assert_eq!(*table.get(&8), 42);
    assert_eq!(*table.get(&9), 0);
}

/// Control 4 on the public surface: a non-power-of-two group width is refused, unchanged.
#[test]
fn non_power_of_two_group_width_is_refused_publicly() {
    let mut table = Table::<i64, i64, _, 3>::new(11, IdentityI64Home);
    assert_eq!(
        table.insert_with(&ReplacementMerge, 5, 1),
        Err(apeiron::StorageError::SizeOverflow)
    );
    assert!(table.is_empty());
    assert_eq!(*table.get(&5), 11);
}

/// Control 5 on the public surface: the zero-width refusal changes no state.
#[test]
fn zero_group_width_refusal_leaves_the_table_unchanged() {
    let mut table = Table::<i64, i64, _, 0>::new(13, IdentityI64Home);
    assert_eq!(
        table.insert_with(&|_, _| panic!("merge on refusal"), 5, 1),
        Err(apeiron::StorageError::SizeOverflow)
    );
    assert_eq!(table.len(), 0);
    assert!(table.is_empty());
    assert_eq!(*table.get(&5), 13);
}

/// A group width below one word drives the scan's scalar tail on the real path.
#[test]
fn a_sub_word_group_width_round_trips_through_the_scalar_tail() {
    let mut table = Table::<u64, u64, _, 4>::new(0, IdentityU64Home);
    for k in 0..4096u64 {
        table
            .insert_with(&|new, _old| new, k, k.wrapping_add(7))
            .unwrap();
    }
    assert_eq!(table.len(), 4096);
    for k in 0..4096u64 {
        assert_eq!(*table.get(&k), k.wrapping_add(7));
    }
    assert_eq!(*table.get(&4096), 0);
}

/// Control: a direct candidate holding a different key falls through to the window/sweep.
///
/// Every key has placement `0` and direct candidate lane `0`, so all share one region and one tag.
/// The first key owns that lane; the others are placed past it and must be found by full-key
/// equality in the window/sweep, never by the candidate's tag.
#[test]
fn a_direct_candidate_holding_another_key_falls_through() {
    let mut table = Table::<i64, i64, _, 16>::new(0, SameLaneHome);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    table.insert_with(&ReplacementMerge, 8, 42).unwrap();
    table.insert_with(&ReplacementMerge, 9, 43).unwrap();
    assert_eq!(table.len(), 3);
    assert_eq!(*table.get(&7), 41);
    assert_eq!(*table.get(&8), 42);
    assert_eq!(*table.get(&9), 43);
    assert_eq!(*table.get(&10), 0);
}

/// Control: a growth relocation keeps every direct-resident key findable.
///
/// [`DirectHome`] certifies direct lane `k` for key `k`, all in one region. Keys `0..=32` cross the
/// 32-to-64 slot growth; a key that was direct before the growth and a key inserted only after it
/// must both read back.
#[test]
fn growth_keeps_direct_resident_keys_findable() {
    let mut table = Table::<i64, i64, _, 16>::new(0, DirectHome);
    for k in 0..=32i64 {
        table.insert_with(&ReplacementMerge, k, k + 100).unwrap();
    }
    assert_eq!(table.len(), 33);
    for k in 0..=32i64 {
        assert_eq!(*table.get(&k), k + 100, "key {k} lost by growth");
    }
    assert_eq!(*table.get(&33), 0);
}

/// Control: the merge runs exactly once per logical insert, including a growth insert.
#[test]
fn merge_runs_exactly_once_per_logical_insert() {
    let calls = std::cell::Cell::new(0usize);
    let merge = |new: i64, old: i64| {
        calls.set(calls.get() + 1);
        new.wrapping_add(old)
    };
    let mut table = Table::<i64, i64, _, 16>::new(0, ShiftI64Home);
    for k in 0..40i64 {
        table.insert_with(&merge, k, 1).unwrap();
    }
    assert_eq!(calls.get(), 40, "one merge per insert, including growth");
    table.insert_with(&merge, 5, 2).unwrap();
    assert_eq!(calls.get(), 41, "one merge for an existing-key update");
    table.insert_with(&merge, 40, 3).unwrap();
    assert_eq!(calls.get(), 42, "one merge for a fresh insert after growth");
}

/// Control 4: two distinct `Sigma` values with an identical `K` are two logical keys.
///
/// The placement-only home gives both writes the same address, so the separation must come from the
/// full `(Sigma, K)` payload equality, never from the address. A `Sigma`-free interface would fold
/// them into one key.
#[test]
fn distinct_sigma_same_key_stay_distinct() {
    let mut table = Table::<i64, i64, _, 16>::new(-1, ZeroI64Home);
    table
        .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 41)
        .unwrap();
    table
        .insert_with_at(&ReplacementMerge, Sigma::Hyperbolic, 7, 42)
        .unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!(*table.get_at(Sigma::Elliptic, &7), 41);
    assert_eq!(*table.get_at(Sigma::Hyperbolic, &7), 42);
    assert_eq!(*table.get_at(Sigma::Parabolic, &7), -1);

    // The two writes really do share one address; the payload key separates them.
    let elliptic = Home::address(&ZeroI64Home, Sigma::Elliptic, &7);
    let hyperbolic = Home::address(&ZeroI64Home, Sigma::Hyperbolic, &7);
    assert_eq!(elliptic.place, hyperbolic.place);
    assert_eq!(elliptic.direct_word, hyperbolic.direct_word);
}

/// The `Sigma`-keyed build consumes `&[((Sigma, K), V)]` and keeps each face's stream order.
#[test]
fn build_at_separates_sigma_and_preserves_stream_order() {
    let table = Table::<i64, _, _>::build_at(
        &merge::link,
        Vec::new(),
        &[
            ((Sigma::Parabolic, 1), vec![1]),
            ((Sigma::Hyperbolic, 1), vec![9]),
            ((Sigma::Parabolic, 1), vec![2]),
        ],
        ZeroI64Home,
    )
    .unwrap();
    assert_eq!(table.get_at(Sigma::Parabolic, &1), &[2, 1]);
    assert_eq!(table.get_at(Sigma::Hyperbolic, &1), &[9]);
    assert!(table.get_at(Sigma::Elliptic, &1).is_empty());
}

/// Control 1: a certified direct-index key routes by its coordinate and round-trips across growth.
///
/// `WordHome` certifies `x < 2^32`; for `x = shard + 256 * slot` the coordinate route is `shard`
/// and the direct lane is `slot`. Seventeen slots per shard force the region past its first
/// 32-slot layout, so a routing or relocation error loses a key.
#[test]
fn certified_direct_index_keys_round_trip_by_coordinate() {
    let mut table = Table::<i64, i64, _, 16>::new(-1, WordHome);
    for slot in 0..17i64 {
        for shard in 0..256i64 {
            let key = shard + 256 * slot;
            table
                .insert_with(&ReplacementMerge, key, key + 1000)
                .unwrap();
        }
    }
    for slot in 0..17i64 {
        for shard in 0..256i64 {
            let key = shard + 256 * slot;
            assert_eq!(*table.get(&key), key + 1000, "key {key}");
        }
    }
    assert_eq!(*table.get(&(256 * 100)), -1);
}

/// Control 2: a direct candidate occupied by a different key falls through and is found after
/// growth.
///
/// Every key certifies direct lane 0, so the first key owns the candidate and every later key must
/// be found by full-key equality in the window/sweep. Forty keys cross the 32-slot growth.
#[test]
fn a_colliding_direct_candidate_falls_through_after_growth() {
    let mut table = Table::<i64, i64, _, 16>::new(0, SameLaneHome);
    for k in 0..40i64 {
        table.insert_with(&ReplacementMerge, k, k + 7).unwrap();
    }
    assert_eq!(table.len(), 40);
    for k in 0..40i64 {
        assert_eq!(
            *table.get(&k),
            k + 7,
            "key {k} lost by the colliding candidate"
        );
    }
    assert_eq!(*table.get(&40), 0);
}

/// Control 4: an absent direct-resident key reads the default under `DirectCover`.
///
/// Each shard holds slots `0..8`; slot 9's direct lane is empty for every shard, so the empty
/// candidate is the `DirectCover` absence certificate. A probe that ignored the certificate would
/// still return the default, but a table that misplaced a live key into slot 9's lane would not.
#[test]
fn an_absent_direct_resident_key_reads_the_default_under_direct_cover() {
    let mut table = Table::<i64, i64, _, 16>::new(-1, WordHome);
    for slot in 0..8i64 {
        for shard in 0..256i64 {
            table
                .insert_with(&ReplacementMerge, shard + 256 * slot, shard)
                .unwrap();
        }
    }
    for shard in 0..256i64 {
        assert_eq!(*table.get(&(shard + 256 * 9)), -1, "shard {shard}");
    }
}

/// An out-of-extent certified coordinate uses grouped spill, not a numeric-sized allocation.
#[test]
fn out_of_range_direct_index_spills_and_is_found() {
    let mut table = Table::<i64, i64, _, 16>::new(-1, WordHome);
    let key = 0x0010_0000_i64; // certified, but lane 4096 cannot pass the 256 KiB density budget.
    table.insert_with(&ReplacementMerge, key, 44).unwrap();
    assert_eq!(table.len(), 1);
    assert_eq!(*table.get(&key), 44);
}

/// A direct slot is not an identity key: unequal faces with the same certified word coexist.
#[test]
fn unequal_sigma_same_word_do_not_alias_a_flat_slot() {
    let mut table = Table::<i64, i64, _, 16>::new(0, WordHome);
    table
        .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 11)
        .unwrap();
    table
        .insert_with_at(&ReplacementMerge, Sigma::Hyperbolic, 7, 12)
        .unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!(*table.get_at(Sigma::Elliptic, &7), 11);
    assert_eq!(*table.get_at(Sigma::Hyperbolic, &7), 12);
}

/// A live direct cell containing another key must continue into the spill search.
#[test]
fn live_nonmatching_flat_slot_falls_through_to_spill() {
    let mut table = Table::<i64, i64, _, 16>::new(0, SameLaneHome);
    table.insert_with(&ReplacementMerge, 1, 10).unwrap();
    table.insert_with(&ReplacementMerge, 2, 20).unwrap();
    assert_eq!(*table.get(&2), 20);
}

/// The decision exposes the direct first-fit vacancy before its real apply writes the flat slot.
#[test]
fn search_decision_reports_direct_first_fit_and_apply_commits_it() {
    let mut table = Table::<i64, i64, _, 16>::new(-1, WordHome);
    table.insert_with(&ReplacementMerge, 0, 10).unwrap();
    let vacant = table.search_decision_at(Sigma::Parabolic, &1);
    assert_eq!(vacant.class, SearchClass::Vacant);
    assert!(matches!(vacant.location, SearchLocation::Flat(_)));

    table.insert_with(&ReplacementMerge, 1, 11).unwrap();
    let found = table.search_decision_at(Sigma::Parabolic, &1);
    assert_eq!(found.class, SearchClass::Found);
    assert!(matches!(found.location, SearchLocation::Flat(_)));
    assert_eq!(*table.get(&1), 11);
}

/// An occupied nonmatching direct candidate continues to the spill decision and its real apply.
#[test]
fn search_decision_finds_a_spill_key_after_a_direct_collision() {
    let mut table = Table::<i64, i64, _, 16>::new(0, SameLaneHome);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    table.insert_with(&ReplacementMerge, 8, 42).unwrap();
    let decision = table.search_decision_at(Sigma::Parabolic, &8);
    assert_eq!(decision.class, SearchClass::Found);
    assert!(matches!(decision.location, SearchLocation::Spill(0, _)));
    assert_eq!(*table.get(&8), 42);
}

/// An empty spill has no indexed candidate, so its decision requests prepared spill growth.
#[test]
fn search_decision_reports_exhaustion_without_an_index() {
    let table = Table::<i64, i64, _>::new(0, ZeroI64Home);
    let decision = table.search_decision_at(Sigma::Parabolic, &7);
    assert_eq!(decision.class, SearchClass::Exhausted);
    assert_eq!(decision.location, SearchLocation::NoIndex);
}

/// The spill region capacity this crate's sizing rule reserves for `load` live entries.
///
/// `src/table/state.rs`'s `slots_for` reserves the least power of two at or above twice the live load
/// and at or above twice the group width; `Table::grow_spill` then takes the double of a full region,
/// so the same geometric doubling continues above `2 * N`. The geometry is written down here rather
/// than read back from the table, and its two documented invariants are asserted by the caller, so a
/// floor of `N` instead of `2 * N` reddens the tests that use it rather than passing silently.
///
/// Temporal `O(log(load))` doublings; spatial `O(1)` words.
fn region_capacity(load: usize, group_width: usize) -> usize {
    if load == 0 {
        return 0;
    }
    let mut capacity = group_width.max(2) * 2;
    while capacity < load * 2 {
        capacity *= 2;
    }
    capacity
}

/// Exhausts the real placement-only decision/apply path and checks the absence certificate at every
/// live count of one shard's first spill region.
///
/// With `ZeroI64Home` every key places at `0`, so shard `0` is the only region, every key probes
/// from lane `0`, and the `live`-th distinct key lands in lane `live`. The absent probe is a key
/// outside the inserted family: it walks from lane `0` and stops at the first vacancy, so it is
/// `Vacant` at lane `live` exactly while the region still holds a vacancy and `Exhausted` with no
/// index only once every lane is live. Growth is then forced by one more insert and the certificate
/// is checked again, so the exhaustion boundary is pinned by the table's own behaviour.
macro_rules! absence_certificate {
    ($group_width:literal) => {{
        let group_width = $group_width;
        let capacity = region_capacity(1, group_width);
        assert!(
            capacity.is_power_of_two(),
            "a region reserves a power-of-two capacity: {capacity} at N = {group_width}"
        );
        assert!(
            capacity >= 2 * group_width,
            "the 2 * N floor: {capacity} at N = {group_width}"
        );

        let mut table = Table::<i64, i64, _, $group_width>::new(-1, ZeroI64Home);
        let resident = |live: usize| i64::try_from(live).expect("a region live count fits i64");
        let absent = |live: usize| resident(live) + 1_000_000;

        for live in 0..=capacity {
            if live > 0 {
                table
                    .insert_with_at(
                        &ReplacementMerge,
                        Sigma::Parabolic,
                        resident(live - 1),
                        stored(live - 1),
                    )
                    .unwrap();
            }
            let decision = table.search_decision_at(Sigma::Parabolic, &absent(live));
            // `region_capacity(0, _) = 0`: an empty region has no indexed lane, and a full one has
            // no vacancy to stop on.  `Vacant` needs a reserved region and at least one free lane.
            if live > 0 && live < capacity {
                assert_eq!(
                    decision.class,
                    SearchClass::Vacant,
                    "a vacancy remains at live {live} of {capacity}, N = {group_width}"
                );
                assert_eq!(decision.location, SearchLocation::Spill(0, live));
            } else {
                assert_eq!(
                    decision.class,
                    SearchClass::Exhausted,
                    "no vacancy remains at live {live} of {capacity}, N = {group_width}"
                );
                assert_eq!(decision.location, SearchLocation::NoIndex);
            }
            assert_eq!(*table.get_at(Sigma::Parabolic, &absent(live)), -1);

            for placed in 0..live {
                let found = table.search_decision_at(Sigma::Parabolic, &resident(placed));
                assert_eq!(
                    found.class,
                    SearchClass::Found,
                    "placed {placed} of {live}, N = {group_width}"
                );
                assert_eq!(found.location, SearchLocation::Spill(0, placed));
                assert_eq!(
                    *table.get_at(Sigma::Parabolic, &resident(placed)),
                    stored(placed)
                );
            }
        }

        table
            .insert_with_at(
                &ReplacementMerge,
                Sigma::Parabolic,
                resident(capacity),
                stored(capacity),
            )
            .unwrap();
        // Growth relocates the full region into a successor of twice its capacity, so lane
        // `capacity` holds the entry that forced growth and the absence certificate moves to the
        // first lane after it.
        let decision = table.search_decision_at(Sigma::Parabolic, &absent(capacity + 1));
        assert_eq!(
            decision.class,
            SearchClass::Vacant,
            "growth restores the vacancy at N = {group_width}"
        );
        assert_eq!(decision.location, SearchLocation::Spill(0, capacity + 1));
        assert_eq!(table.len(), capacity + 1);
    }};
}

/// Exhaustive small model for the real placement-only decision/apply path.
///
/// Every ordered prefix of three full logical keys is built at `N = 1` and every key is queried at
/// every prefix, so direct spill found, the exact insertion lane, first-fit vacancy, exhaustion, and
/// default readback are all checked against [`region_capacity`]: an absent key is `Vacant` at lane
/// `live` exactly while the region holds a vacancy, and `Exhausted` with no index otherwise. The
/// three logical keys include equal payload words at unequal `Sigma`, so identity is the full
/// logical key. The absence certificate's boundary is then checked at every live count from zero
/// past the first region for `N = 1` and for a larger `N`, which pins the `2 * N` floor by behaviour
/// rather than by a hand-picked constant.
#[test]
fn search_decision_matches_the_two_lane_spill_oracle_for_every_small_prefix() {
    let logical = [
        (Sigma::Elliptic, 7_i64),
        (Sigma::Parabolic, 7_i64),
        (Sigma::Parabolic, 8_i64),
    ];
    let capacity = region_capacity(1, 1);
    assert!(
        logical.len() < capacity,
        "the prefix family is shorter than the first region, so the full-region case is covered by \
         the absence certificate below rather than by a prefix"
    );

    for order in ordered_prefixes(logical.len()) {
        let mut table = Table::<i64, i64, _, 1>::new(-1, ZeroI64Home);
        for (rank, &index) in order.iter().enumerate() {
            let (sigma, key) = logical[index];
            table
                .insert_with_at(&ReplacementMerge, sigma, key, stored(rank))
                .unwrap();
        }

        for (query, &(sigma, key)) in logical.iter().enumerate() {
            let decision = table.search_decision_at(sigma, &key);
            match order.iter().position(|&index| index == query) {
                Some(rank) => {
                    assert_eq!(decision.class, SearchClass::Found, "key {key} in {order:?}");
                    assert_eq!(decision.location, SearchLocation::Spill(0, rank));
                    assert_eq!(*table.get_at(sigma, &key), stored(rank));
                }
                None if order.is_empty() => {
                    // No region exists yet, so there is no indexed lane at all.
                    assert_eq!(decision.class, SearchClass::Exhausted, "empty in {order:?}");
                    assert_eq!(decision.location, SearchLocation::NoIndex);
                    assert_eq!(*table.get_at(sigma, &key), -1);
                }
                None => {
                    assert_eq!(decision.class, SearchClass::Vacant, "absent in {order:?}");
                    assert_eq!(decision.location, SearchLocation::Spill(0, order.len()));
                    assert_eq!(*table.get_at(sigma, &key), -1);
                }
            }
        }
    }

    absence_certificate!(1);
    absence_certificate!(16);
}

/// Every ordered sequence of distinct indices from `0..n`, including the empty sequence.
///
/// The decision oracles below use it to enumerate the reachable prefixes of a fixed finite key
/// family, so they cover every insertion order rather than one hand-picked order.
fn ordered_prefixes(n: usize) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new()];
    let mut frontier = vec![Vec::new()];
    while let Some(prefix) = frontier.pop() {
        for next in 0..n {
            if prefix.contains(&next) {
                continue;
            }
            let mut extended = prefix.clone();
            extended.push(next);
            result.push(extended.clone());
            frontier.push(extended);
        }
    }
    result
}

/// The stored value for the key inserted at `index` in a decision-test prefix.
///
/// The explicit conversion keeps the test value an `i64` without the wrapping `as` cast.
fn stored(index: usize) -> i64 {
    100 + i64::try_from(index).expect("a three-key prefix index fits i64")
}

/// Exhaustive small-model oracle for the real decision on the direct-resident face.
///
/// `WordHome` certifies `direct_word = Some(k)` for each `k < 2^32`, and the first insertion selects
/// a one-lane resident extent, so keys `0`, `1`, `2` own flat slots `0`, `1`, `2`. The model is
/// written from that geometry, not read back from the table: a present key is `Found` at its unique
/// flat slot and an absent key is the first-fit `Vacant` flat slot. Every ordered prefix of the
/// three keys is built and every key is queried at every prefix, so direct found, direct first-fit
/// vacancy, the exact flat slot index, well-formedness and default readback are checked against the
/// model. A fresh table (the empty prefix) is the only `Exhausted/NoIndex` case; once the one-lane
/// extent exists no spill is reachable in this family.
#[test]
fn search_decision_matches_the_direct_resident_oracle_for_every_small_prefix() {
    let keys = [0_i64, 1, 2];
    let default = -1_i64;

    for order in ordered_prefixes(keys.len()) {
        let mut table = Table::<i64, i64, _, 16>::new(default, WordHome);
        for (rank, &index) in order.iter().enumerate() {
            table
                .insert_with_at(
                    &ReplacementMerge,
                    Sigma::Parabolic,
                    keys[index],
                    stored(rank),
                )
                .unwrap();
        }

        for (query, &key) in keys.iter().enumerate() {
            let decision = table.search_decision_at(Sigma::Parabolic, &key);
            if order.is_empty() {
                // No resident extent exists yet, so there is no indexed lane at all.
                assert_eq!(
                    decision.class,
                    SearchClass::Exhausted,
                    "empty table, key {key}"
                );
                assert_eq!(decision.location, SearchLocation::NoIndex);
                assert_eq!(*table.get_at(Sigma::Parabolic, &key), default);
                continue;
            }
            assert!(matches!(
                (decision.class, decision.location),
                (
                    SearchClass::Found | SearchClass::Vacant,
                    SearchLocation::Flat(_)
                )
            ));
            if let Some(rank) = order.iter().position(|&index| index == query) {
                assert_eq!(decision.class, SearchClass::Found, "key {key} in {order:?}");
                assert_eq!(decision.location, SearchLocation::Flat(query));
                assert_eq!(*table.get_at(Sigma::Parabolic, &key), stored(rank));
            } else {
                assert_eq!(
                    decision.class,
                    SearchClass::Vacant,
                    "key {key} in {order:?}"
                );
                assert_eq!(decision.location, SearchLocation::Flat(query));
                assert_eq!(*table.get_at(Sigma::Parabolic, &key), default);
            }
        }
    }
}

/// Exhaustive small-model oracle for the real decision on a direct-certified, single-shard face.
///
/// `SameLaneHome` certifies direct lane `0` and shard `0` for every key, so the first insertion
/// rebuilds a one-lane flat extent holding that key in flat slot `0` and every later key spills.
/// At `N = 1` the grown region has exactly two lanes. The model is written from that geometry: the
/// first inserted key is `Found` at `Flat(0)`; the `j`-th later key is `Found` at `Spill(0, j-1)`;
/// an absent query is `Exhausted` while the spill is empty, otherwise `Vacant` at the first free
/// spill lane, and `Exhausted` once both lanes are live. Every ordered prefix of three distinct
/// logical keys is built and every key is queried at every prefix, so direct found, spill found,
/// first-fit spill vacancy, exhaustion, unequal-`Sigma` identity and default readback are checked
/// against the model.
#[test]
fn search_decision_matches_the_direct_collision_spill_oracle_for_every_small_prefix() {
    let keys = [
        (Sigma::Elliptic, 7_i64),
        (Sigma::Parabolic, 7_i64),
        (Sigma::Parabolic, 8_i64),
    ];
    let default = -1_i64;

    for order in ordered_prefixes(keys.len()) {
        let mut table = Table::<i64, i64, _, 1>::new(default, SameLaneHome);
        for (rank, &index) in order.iter().enumerate() {
            let (sigma, key) = keys[index];
            table
                .insert_with_at(&ReplacementMerge, sigma, key, stored(rank))
                .unwrap();
        }

        for (query, &(sigma, key)) in keys.iter().enumerate() {
            let decision = table.search_decision_at(sigma, &key);
            assert!(matches!(
                (decision.class, decision.location),
                (
                    SearchClass::Found | SearchClass::Vacant,
                    SearchLocation::Flat(_) | SearchLocation::Spill(_, _)
                ) | (SearchClass::Exhausted, SearchLocation::NoIndex)
            ));
            match order.iter().position(|&index| index == query) {
                Some(0) => {
                    assert_eq!(decision.class, SearchClass::Found, "first key in {order:?}");
                    assert_eq!(decision.location, SearchLocation::Flat(0));
                    assert_eq!(*table.get_at(sigma, &key), 100);
                }
                Some(rank) => {
                    assert_eq!(
                        decision.class,
                        SearchClass::Found,
                        "spilled key in {order:?}"
                    );
                    assert_eq!(decision.location, SearchLocation::Spill(0, rank - 1));
                    assert_eq!(*table.get_at(sigma, &key), stored(rank));
                }
                None if order.len() <= 1 => {
                    assert_eq!(
                        decision.class,
                        SearchClass::Exhausted,
                        "absent in {order:?}"
                    );
                    assert_eq!(decision.location, SearchLocation::NoIndex);
                    assert_eq!(*table.get_at(sigma, &key), default);
                }
                None if order.len() < keys.len() => {
                    assert_eq!(decision.class, SearchClass::Vacant, "absent in {order:?}");
                    assert_eq!(decision.location, SearchLocation::Spill(0, order.len() - 1));
                    assert_eq!(*table.get_at(sigma, &key), default);
                }
                None => {
                    assert_eq!(decision.class, SearchClass::Exhausted, "full in {order:?}");
                    assert_eq!(decision.location, SearchLocation::NoIndex);
                    assert_eq!(*table.get_at(sigma, &key), default);
                }
            }
        }
    }
}

/// Extent doubling rebuilds the composite layout and moves a formerly spilled certified key.
#[test]
fn doubling_resident_extent_reclassifies_newly_eligible_spill_key() {
    let mut table = Table::<i64, i64, _, 16>::new(-1, WordHome);
    table.insert_with(&ReplacementMerge, 256, 900).unwrap(); // lane one initially spills.
    for word in 0..256_i64 {
        table.insert_with(&ReplacementMerge, word, word).unwrap();
    }
    assert_eq!(
        table.len(),
        257,
        "reclassification must not duplicate the key"
    );
    assert_eq!(*table.get(&256), 900);
    for word in 0..256_i64 {
        assert_eq!(*table.get(&word), word);
    }
}

/// A sparse coordinate fails the density policy but is still a valid spill insertion.
#[test]
fn density_failure_spills_instead_of_refusing() {
    let mut table = Table::<i64, i64, _, 16>::new(3, WordHome);
    table.insert_with(&ReplacementMerge, 256 * 1024, 8).unwrap();
    assert_eq!(table.len(), 1);
    assert_eq!(*table.get(&(256 * 1024)), 8);
}

/// Elliptic storage keeps only `Addr.id`; `I64WordCodec` recovers the word
/// through `unspread`, never through its placement word.
#[test]
fn regen_recovers_the_key_exactly_from_its_identity_coordinate() {
    let mut table = Table::<i64, i64, WordHome, 16, Regen<I64WordCodec>>::new(-1, WordHome);
    for key in [0, 1, -1, i64::MIN, i64::MAX, 0x1234_5678] {
        table
            .insert_with_at(&ReplacementMerge, Sigma::Elliptic, key, key.wrapping_add(9))
            .unwrap();
        assert_eq!(*table.get_at(Sigma::Elliptic, &key), key.wrapping_add(9));
    }
}

/// Equal `place` does not decide elliptic identity: `id` is the held word.
#[test]
fn regen_keys_with_one_place_and_distinct_ids_do_not_alias() {
    let mut table = Table::<i64, i64, SameLaneHome, 16, Regen<I64WordCodec>>::new(0, SameLaneHome);
    table
        .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 41)
        .unwrap();
    table
        .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 8, 42)
        .unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!(*table.get_at(Sigma::Elliptic, &7), 41);
    assert_eq!(*table.get_at(Sigma::Elliptic, &8), 42);
}

/// `Sigma` remains outside `Held` for regenerated slots as it does for stored slots.
#[test]
fn regen_keeps_unequal_sigma_with_an_equal_key_distinct() {
    let mut table = Table::<i64, i64, WordHome, 16, Regen<I64WordCodec>>::new(0, WordHome);
    table
        .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 11)
        .unwrap();
    table
        .insert_with_at(&ReplacementMerge, Sigma::Hyperbolic, 7, 12)
        .unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!(*table.get_at(Sigma::Elliptic, &7), 11);
    assert_eq!(*table.get_at(Sigma::Hyperbolic, &7), 12);
}

/// An exhausted search routes through the prepared outcome to `insert_with_at`'s spill growth,
/// which commits the pending entry exactly once and finds it on the next read.
#[test]
fn spill_growth_after_exhaustion_commits_the_pending_entry_exactly_once() {
    let calls = std::cell::Cell::new(0usize);
    let merge = |new: i64, old: i64| {
        calls.set(calls.get() + 1);
        new.wrapping_add(old)
    };
    let mut table = Table::<i64, i64, _, 16>::new(0, IdentityI64Home);
    // The empty region has no indexed candidate, so the first insertion is the exhaustion route.
    assert_eq!(
        table.search_decision_at(Sigma::Parabolic, &7).class,
        SearchClass::Exhausted
    );
    table.insert_with(&merge, 7, 41).unwrap();
    assert_eq!(calls.get(), 1, "one merge for the grown pending entry");
    assert_eq!(table.len(), 1);
    assert_eq!(*table.get(&7), 41);
    assert_eq!(
        table.search_decision_at(Sigma::Parabolic, &7).class,
        SearchClass::Found
    );
}

/// Asserts that the committed table, not the reserved staging storage, serves later reads: a
/// re-insertion of the pending key merges with its committed value exactly once and does not
/// change the logical length.
fn assert_committed_pending_is_live<R>(
    table: &mut Table<i64, i64, WordHome, 16, R>,
    merge: &impl Fn(i64, i64) -> i64,
    calls: &std::cell::Cell<usize>,
    label: &str,
) where
    R: KeyRep<i64, WordHome>,
{
    let after_commit = calls.get();
    table
        .insert_with_at(merge, Sigma::Elliptic, 256, 5)
        .unwrap();
    assert_eq!(
        calls.get() - after_commit,
        1,
        "{label}: the committed pending key merges exactly once"
    );
    assert_eq!(
        table.len(),
        258,
        "{label}: an update of the committed key does not change the length"
    );
    assert_eq!(
        *table.get_at(Sigma::Elliptic, &256),
        13,
        "{label}: the committed pending value is the merge base"
    );
}

/// Drives one carrier through a resident-extent rebuild that relocates a full resident lane and a
/// spilled entry, then checks value, identity and occupancy preservation and merge isolation.
///
/// The first 256 certified lane-0 keys fill one resident lane and its density.  A coordinate outside
/// the certified direct domain is placement-only and spills.  The certified lane-1 key then doubles
/// the resident extent: `apply_prepared_rebuild` materializes the new payloads and the relocation
/// core moves every old resident to flat index `2 * key` and the spill entry to its new region.  A
/// counting merge shows the callback runs once for the pending insert and never during relocation.
fn relocation_rebuild_moves_entries<R>(label: &str)
where
    R: KeyRep<i64, WordHome>,
{
    // A zero default keeps the total-default merge value-preserving for the fresh inserts below.
    let default = 0_i64;
    let calls = std::cell::Cell::new(0usize);
    let merge = |new: i64, old: i64| {
        calls.set(calls.get() + 1);
        new.wrapping_add(old)
    };
    let mut table = Table::<i64, i64, WordHome, 16, R>::new(default, WordHome);

    // One certified lane-0 key per shard fills the first resident extent and its density.
    for key in 0..256_i64 {
        table
            .insert_with_at(&merge, Sigma::Elliptic, key, key + 1000)
            .unwrap();
    }
    // `2^32` is outside the certified direct domain, so it is placement-only and spills.
    let spilled = 1_i64 << 32;
    table
        .insert_with_at(&merge, Sigma::Elliptic, spilled, 7)
        .unwrap();
    assert_eq!(table.len(), 257, "{label}: resident plus spill");

    // The lane-1 key doubles the extent, so the rebuild relocates every old resident and the spill.
    let before = calls.get();
    table
        .insert_with_at(&merge, Sigma::Elliptic, 256, 8)
        .unwrap();
    assert_eq!(
        calls.get() - before,
        1,
        "{label}: relocation invoked the callback merge; only the pending insert may"
    );
    assert_eq!(table.len(), 258, "{label}: relocation must not duplicate");

    // Every resident moved to its new lane and kept its value and identity.
    for key in 0..256_i64 {
        let index = usize::try_from(key).expect("0..256 fits usize");
        let decision = table.search_decision_at(Sigma::Elliptic, &key);
        assert_eq!(decision.class, SearchClass::Found, "{label}: key {key}");
        assert_eq!(
            decision.location,
            SearchLocation::Flat(2 * index),
            "{label}: key {key} did not move to the new destination"
        );
        assert_eq!(
            *table.get_at(Sigma::Elliptic, &key),
            key + 1000,
            "{label}: key {key} value"
        );
    }

    // The spill entry moved to its new region and kept its value and identity.
    let spill_decision = table.search_decision_at(Sigma::Elliptic, &spilled);
    assert_eq!(
        spill_decision.class,
        SearchClass::Found,
        "{label}: spill key"
    );
    assert!(
        matches!(spill_decision.location, SearchLocation::Spill(_, _)),
        "{label}: spill key location"
    );
    assert_eq!(
        *table.get_at(Sigma::Elliptic, &spilled),
        7,
        "{label}: spill value"
    );

    // The pending insert committed once at its certified lane and absent reads keep the default.
    let pending = table.search_decision_at(Sigma::Elliptic, &256);
    assert_eq!(pending.class, SearchClass::Found, "{label}: pending key");
    assert_eq!(
        pending.location,
        SearchLocation::Flat(1),
        "{label}: pending destination"
    );
    assert_eq!(
        *table.get_at(Sigma::Elliptic, &256),
        8,
        "{label}: pending value"
    );
    assert_eq!(
        *table.get_at(Sigma::Elliptic, &257),
        default,
        "{label}: absent default"
    );

    // The committed state, not the reserved staging storage, serves later reads and a
    // read-modify-write.
    assert_committed_pending_is_live(&mut table, &merge, &calls, label);

    // The traversal must place every planned entry: 256 relocated residents, the relocated spill,
    // and the pending insert.  A skipped relocation loses a key; a duplicated one collapses two
    // keys onto a single destination.  Finding every key and distinctness of the destinations
    // together witness full, in-order coverage.
    let mut destinations = Vec::with_capacity(table.len());
    for key in 0..=256_i64 {
        let decision = table.search_decision_at(Sigma::Elliptic, &key);
        assert_eq!(
            decision.class,
            SearchClass::Found,
            "{label}: relocated key {key} is missing"
        );
        destinations.push(decision.location);
    }
    destinations.push(spill_decision.location);
    assert_eq!(
        destinations.len(),
        table.len(),
        "{label}: every planned entry is present exactly once"
    );
    for i in 0..destinations.len() {
        for j in (i + 1)..destinations.len() {
            assert_ne!(
                destinations[i], destinations[j],
                "{label}: two planned entries share one destination"
            );
        }
    }
}

/// The prepared rebuild materializes payloads before the core and relocates stored and spill
/// entries without invoking the callback merge, on both carrier faces.
#[test]
fn prepared_rebuild_uses_materialized_payloads_and_relocates_without_merge() {
    relocation_rebuild_moves_entries::<Stored<i64>>("stored");
    relocation_rebuild_moves_entries::<Regen<I64WordCodec>>("regen");
}

// ---------------------------------------------------------------------------
// : the whole-table reduce, ported from incumbent
// `fold hashtrinity/src/lib.rs:891` (`Table::encircle`).
// ---------------------------------------------------------------------------

/// Builds a `Cell<i64>`-valued table over `IdentityI64Home`, writing key `i` on `sigma`.
fn cell_table(
    cell_keys: &[(i64, Sigma)],
    width: usize,
) -> Table<i64, Cell<i64>, IdentityI64Home, 16> {
    let merge = |new: Cell<i64>, _old: Cell<i64>| new;
    let mut table =
        Table::<i64, Cell<i64>, IdentityI64Home, 16>::new(Cell::zero(), IdentityI64Home);
    let mut index = 0_i64;
    while usize::try_from(index).expect("the index fits a usize") < cell_keys.len() {
        let (key, sigma) = cell_keys[usize::try_from(index).expect("the index fits a usize")];
        table
            .insert_with_at(&merge, sigma, key, Cell::<i64>::of(index + 1, -index - 2))
            .expect("the insertion reserves and commits");
        index += 1;
    }
    assert_eq!(table.len(), cell_keys.len(), "{width} keys are live");
    table
}

#[test]
fn encircle_reduce_all_of_an_empty_table_is_the_identity() {
    let table = Table::<i64, Cell<i64>, IdentityI64Home, 16>::new(Cell::zero(), IdentityI64Home);
    let witness = AffineFoldStep {
        sigma: Sigma::Parabolic,
    };
    assert_eq!(table.reduce_all(&witness), AffineStep::id());
    assert!(table.values_in_order().is_empty());
}

#[test]
fn encircle_reduce_all_is_the_corpus_fold_over_storage_order() {
    let keys: Vec<(i64, Sigma)> = (0..40_i64)
        .map(|i| (i * 7 % 40, Sigma::Parabolic))
        .collect();
    let table = cell_table(&keys, 16);
    let witness = AffineFoldStep {
        sigma: Sigma::Parabolic,
    };
    let steps: Vec<AffineStep<i64>> = table
        .values_in_order()
        .iter()
        .map(|value| witness.residual(value))
        .collect();
    assert_eq!(steps.len(), table.len(), "one step per live slot");
    assert_eq!(
        table.reduce_all(&witness),
        witness.fold(&steps),
        "the table reduce is the corpus left fold over the live slots in storage order"
    );
}

#[test]
fn encircle_chunked_merge_reproduces_the_sequential_reduce() {
    let keys: Vec<(i64, Sigma)> = (0..5000_i64)
        .map(|i| (i * 7 % 5000, Sigma::Parabolic))
        .collect();
    let table = cell_table(&keys, 16);
    let witness = AffineFoldStep {
        sigma: Sigma::Parabolic,
    };
    let values = table.values_in_order();
    assert!(
        values.len() > 4096,
        "the table grew past its resident extent: {} live slots",
        values.len()
    );
    // The incumbent's parallel branch: one partial per task, folded over a contiguous chunk of the
    // live stream, then merged in task order.
    let tasks = 4_usize;
    let chunk = values.len().div_ceil(tasks);
    let mut partials: Vec<AffineStep<i64>> = Vec::new();
    let mut start = 0;
    while start < values.len() {
        let end = (start + chunk).min(values.len());
        let steps: Vec<AffineStep<i64>> = values[start..end]
            .iter()
            .map(|value| witness.residual(value))
            .collect();
        partials.push(witness.fold(&steps));
        start = end;
    }
    let merged = partials.into_iter().fold(AffineStep::id(), |acc, part| {
        part.compose(Sigma::Parabolic, acc)
    });
    assert_eq!(
        merged,
        table.reduce_all(&witness),
        "composing the chunk partials recovers the sequential reduce"
    );
}

#[test]
fn encircle_reduce_all_runs_at_every_face_without_a_sign_hypothesis() {
    let keys: Vec<(i64, Sigma)> = (0..24_i64)
        .map(|i| {
            let sigma = match i % 3 {
                0 => Sigma::Elliptic,
                1 => Sigma::Parabolic,
                _ => Sigma::Hyperbolic,
            };
            (i, sigma)
        })
        .collect();
    let table = cell_table(&keys, 16);
    let values = table.values_in_order();
    for sigma in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
        let witness = AffineFoldStep { sigma };
        let steps: Vec<AffineStep<i64>> =
            values.iter().map(|value| witness.residual(value)).collect();
        assert_eq!(
            table.reduce_all(&witness),
            witness.fold(&steps),
            "the fold law holds at sigma = {}",
            sigma.value()
        );
    }
}

#[test]
fn encircle_compose_is_the_corpus_compose_on_a_witnessed_pair() {
    // `ringAffineCompose` term for term: the composed injection carries the inner injection
    // forward by the outer coefficient.
    let outer = AffineStep {
        coefficient: Cell::<i64>::of(3, 1),
        injection: Cell::<i64>::of(2, -5),
    };
    let inner = AffineStep {
        coefficient: Cell::<i64>::of(-7, 4),
        injection: Cell::<i64>::of(11, 6),
    };
    for sigma in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
        let composed = outer.compose(sigma, inner);
        assert_eq!(
            composed.coefficient,
            outer.coefficient.mul(sigma, inner.coefficient),
            "sigma = {}: coefficient is outer * inner",
            sigma.value()
        );
        assert_eq!(
            composed.injection,
            outer
                .coefficient
                .mul(sigma, inner.injection)
                .add(outer.injection),
            "sigma = {}: injection carries inner forward",
            sigma.value()
        );
        assert_eq!(
            composed.apply(sigma, Cell::<i64>::of(9, -3)),
            outer.apply(sigma, inner.apply(sigma, Cell::<i64>::of(9, -3))),
            "sigma = {}: apply(outer, apply(inner, z)) == apply(compose, z)",
            sigma.value()
        );
    }
}

#[test]
fn encircle_identity_step_is_neutral_under_compose() {
    let step = AffineStep {
        coefficient: Cell::<i64>::of(5, -2),
        injection: Cell::<i64>::of(-3, 8),
    };
    for sigma in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
        assert_eq!(step.compose(sigma, AffineStep::id()), step);
        assert_eq!(AffineStep::id().compose(sigma, step), step);
        assert_eq!(
            AffineStep::id().apply(sigma, Cell::<i64>::of(4, 6)),
            Cell::<i64>::of(4, 6)
        );
    }
}

#[test]
fn encircle_reduce_all_at_the_mixing_witness_is_the_corpus_fold_over_storage_order() {
    let keys: Vec<(i64, Sigma)> = (0..40_i64)
        .map(|i| (i * 7 % 40, Sigma::Parabolic))
        .collect();
    let table = cell_table(&keys, 16);
    let witness = AffineMixStep {
        sigma: Sigma::Parabolic,
    };
    let steps: Vec<AffineStep<i64>> = table
        .values_in_order()
        .iter()
        .map(|value| witness.residual(value))
        .collect();
    assert_eq!(steps.len(), table.len());
    assert_eq!(
        table.reduce_all(&witness),
        witness.fold(&steps),
        "the table reduce is the corpus left fold at the non-commutative witness"
    );
}

#[test]
fn encircle_chunked_merge_at_the_mixing_witness_reproduces_the_sequential_reduce() {
    let keys: Vec<(i64, Sigma)> = (0..5000_i64)
        .map(|i| (i * 7 % 5000, Sigma::Parabolic))
        .collect();
    let table = cell_table(&keys, 16);
    let witness = AffineMixStep {
        sigma: Sigma::Parabolic,
    };
    let values = table.values_in_order();
    assert!(
        values.len() > 4096,
        "the table grew past its resident extent: {} live slots",
        values.len()
    );
    let tasks = 4_usize;
    let chunk = values.len().div_ceil(tasks);
    let mut partials: Vec<AffineStep<i64>> = Vec::new();
    let mut start = 0;
    while start < values.len() {
        let end = (start + chunk).min(values.len());
        let steps: Vec<AffineStep<i64>> = values[start..end]
            .iter()
            .map(|value| witness.residual(value))
            .collect();
        partials.push(witness.fold(&steps));
        start = end;
    }
    let merged = partials.into_iter().fold(AffineStep::id(), |acc, part| {
        part.compose(Sigma::Parabolic, acc)
    });
    assert_eq!(
        merged,
        table.reduce_all(&witness),
        "composing the chunk partials recovers the sequential reduce at the non-commutative witness"
    );
}

#[test]
fn encircle_two_witnesses_disagree_so_neither_test_is_vacuous() {
    // The control for the two witnesses above: if `AffineFoldStep` and `AffineMixStep` produced the
    // same accumulator, swapping the `combine` arguments would be invisible to every test here.
    let keys: Vec<(i64, Sigma)> = (0..12_i64)
        .map(|i| (i * 7 % 12, Sigma::Parabolic))
        .collect();
    let table = cell_table(&keys, 16);
    let translate = AffineFoldStep {
        sigma: Sigma::Parabolic,
    };
    let mixing = AffineMixStep {
        sigma: Sigma::Parabolic,
    };
    let values = table.values_in_order();
    let translate_steps: Vec<AffineStep<i64>> = values
        .iter()
        .map(|value| translate.residual(value))
        .collect();
    let mixing_steps: Vec<AffineStep<i64>> =
        values.iter().map(|value| mixing.residual(value)).collect();
    assert_ne!(
        translate.fold(&translate_steps),
        mixing.fold(&mixing_steps),
        "the two witnesses are distinguishable, so a reversed combine is observable"
    );
    assert_ne!(
        mixing_steps[0].compose(Sigma::Parabolic, mixing_steps[1]),
        mixing_steps[1].compose(Sigma::Parabolic, mixing_steps[0]),
        "compose is not commutative at the mixing witness, so argument order is observable"
    );
}
