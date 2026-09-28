//! W4-F0 locator tests: per-face residue indexing, the shared exact-identity path, coverage,
//! refusal and the reserve-before-retain ordering.
//!
//! These exercise the standalone `src/coord/locator.rs` bodies. They do not migrate a table read,
//! write, carrier or storage path.

#[cfg(feature = "injection")]
use std::cell::Cell;

use apeiron::coord::address::{face_index_policy, FaceIndexPolicy, ParabolicDual, Residue, H3_40};
#[cfg(feature = "injection")]
use apeiron::coord::locator::reservation_failure;
use apeiron::coord::locator::{
    coefficient_bits, residue_digits, Absence, ExactIdentity, FaceIndex, Inserted, Locator, Lookup,
    Refusal,
};
use apeiron::table::keyrep::{ExactCodec, I64WordCodec};
#[cfg(feature = "injection")]
use apeiron::table::Table;
use apeiron::table::{TableAddress, WordHome};
use apeiron::Sigma;

fn profile() -> H3_40 {
    H3_40::new().expect("the frozen H3_40 constants validate")
}

fn residue_of(word: u64) -> Residue {
    Residue::reduce(profile(), word)
}

fn hyperbolic(word: u64) -> FaceIndex {
    FaceIndex::Hyperbolic(residue_of(word))
}

fn parabolic(scalar: u64, nilpotent: u64) -> FaceIndex {
    FaceIndex::Parabolic(ParabolicDual::new(scalar, nilpotent))
}

/// A key encoding both dual coefficients, so two keys can share the scalar `a` and differ in the
/// nilpotent `b`.
fn dual_key(scalar: u64, nilpotent: u64, tag: u8) -> Vec<u8> {
    let mut key = scalar.to_be_bytes().to_vec();
    key.extend_from_slice(&nilpotent.to_be_bytes());
    key.push(tag);
    key
}

fn assert_present<K, V>(locator: &Locator<K, V>, index: FaceIndex, key: &K, expected: &V)
where
    K: ExactIdentity,
    V: PartialEq + core::fmt::Debug,
{
    match locator.get(index, key) {
        Lookup::Present(value) => assert_eq!(value, expected),
        other => panic!("expected a present value, got {other:?}"),
    }
}

fn assert_absent<K, V>(locator: &Locator<K, V>, index: FaceIndex, key: &K)
where
    K: ExactIdentity,
{
    match locator.lookup_absent(index, key) {
        Absence::Absent(scope) => assert_eq!(scope.index(), index),
        Absence::Unknown => panic!("expected a certified absence"),
    }
}

/// The committed-state tuple a refusal must leave unchanged: live payloads, hyperbolic nodes,
/// parabolic nodes, identity nodes, identity edges and the coverage assertion.
#[cfg(feature = "injection")]
fn committed_state<K: ExactIdentity, V>(
    locator: &Locator<K, V>,
) -> (usize, usize, usize, usize, usize, bool) {
    (
        locator.len(),
        locator.residue_node_count(),
        locator.parabolic_node_count(),
        locator.identity_node_count(),
        locator.edge_count(),
        locator.coverage_established(),
    )
}

#[test]
fn face_index_policy_matches_the_contract() {
    assert_eq!(
        face_index_policy(Sigma::Elliptic),
        FaceIndexPolicy::ExactPackNoLocator
    );
    assert_eq!(
        face_index_policy(Sigma::Parabolic),
        FaceIndexPolicy::Base2ResidueTrie
    );
    assert_eq!(
        face_index_policy(Sigma::Hyperbolic),
        FaceIndexPolicy::Base3ResidueTrie
    );
    assert_eq!(FaceIndex::Elliptic.face(), Sigma::Elliptic);
    assert_eq!(parabolic(1, 0).face(), Sigma::Parabolic);
    assert_eq!(hyperbolic(1).face(), Sigma::Hyperbolic);
}

#[test]
fn residue_path_is_forty_ternary_levels() {
    let p = profile();
    // Zero and every leading zero digit are ordinary data.
    assert_eq!(residue_digits(Residue::new(p, 0).unwrap()), [0u8; 40]);
    // `q - 1` is `222...2` in base 3.
    assert_eq!(
        residue_digits(Residue::new(p, H3_40::Q - 1).unwrap()),
        [2u8; 40]
    );
    // Least-significant digit first: `3^39` is the digit 1 at level 39.
    let mut high = [0u8; 40];
    high[39] = 1;
    assert_eq!(residue_digits(Residue::new(p, 3u64.pow(39)).unwrap()), high);
    // `3 = 1*3^1`, so level 0 is zero and level 1 is one.
    let mut three = [0u8; 40];
    three[1] = 1;
    assert_eq!(residue_digits(Residue::new(p, 3).unwrap()), three);
}

#[test]
fn coefficient_path_is_sixty_four_binary_levels() {
    // Zero and every leading zero bit are ordinary data.
    assert_eq!(coefficient_bits(0), [0u8; 64]);
    assert_eq!(coefficient_bits(u64::MAX), [1u8; 64]);
    // Least-significant bit first: `2` is bit 1.
    let mut two = [0u8; 64];
    two[1] = 1;
    assert_eq!(coefficient_bits(2), two);
    let mut high = [0u8; 64];
    high[63] = 1;
    assert_eq!(coefficient_bits(1u64 << 63), high);
}

#[test]
fn insert_get_and_absent_on_a_word() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    assert!(locator.is_empty());
    assert_eq!(locator.profile(), p);
    assert!(locator.coverage_established());

    assert_eq!(locator.insert(hyperbolic(7), &7, 70), Ok(Inserted::Fresh));
    assert_eq!(locator.len(), 1);
    assert_present(&locator, hyperbolic(7), &7, &70);
    assert_absent(&locator, hyperbolic(8), &8);

    // A found key updates in place.
    assert_eq!(locator.insert(hyperbolic(7), &7, 71), Ok(Inserted::Updated));
    assert_present(&locator, hyperbolic(7), &7, &71);
    assert_eq!(locator.len(), 1);
    assert!(locator.check_coverage());
}

#[test]
fn many_to_one_residues_never_decide_identity() {
    let p = profile();
    let mut locator = Locator::<u64, i32>::new(p);
    let zero = Residue::reduce(p, 0);
    let wrapped = Residue::reduce(p, H3_40::Q);
    assert_eq!(zero, wrapped, "0 and q share one hyperbolic residue");

    assert_eq!(
        locator.insert(FaceIndex::Hyperbolic(zero), &0, 10),
        Ok(Inserted::Fresh)
    );
    assert_eq!(
        locator.insert(FaceIndex::Hyperbolic(wrapped), &H3_40::Q, 20),
        Ok(Inserted::Fresh)
    );
    assert_eq!(locator.len(), 2);
    assert_present(&locator, FaceIndex::Hyperbolic(zero), &0, &10);
    assert_present(&locator, FaceIndex::Hyperbolic(wrapped), &H3_40::Q, &20);

    // A third key with the same residue is absent: the residue never decides identity.
    assert_absent(&locator, FaceIndex::Hyperbolic(zero), &1);
    assert!(locator.check_coverage());
}

#[test]
fn parabolic_scalar_indexes_presence_and_absence() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    let scalar = 0xDEAD_BEEF_0000_0001;

    assert_eq!(
        locator.insert(parabolic(scalar, 0), &7, 70),
        Ok(Inserted::Fresh)
    );
    assert_present(&locator, parabolic(scalar, 0), &7, &70);
    // The scalar alone decides the fiber: the same key is found under any nilpotent part.
    assert_present(&locator, parabolic(scalar, 0xFFFF_FFFF), &7, &70);
    // A different key in the same fiber is absent.
    assert_absent(&locator, parabolic(scalar, 0), &8);
    // A different scalar is a different fiber and is absent even for the stored key.
    assert_absent(&locator, parabolic(scalar ^ 1, 0), &7);

    // The certificate reports the parabolic face, the dual coefficient and the 64-level path.
    match locator.lookup_absent(parabolic(scalar, 0), &8) {
        Absence::Absent(scope) => {
            assert_eq!(scope.face(), Sigma::Parabolic);
            assert_eq!(scope.coefficient(), Some(ParabolicDual::new(scalar, 0)));
            assert_eq!(scope.residue(), None);
            assert_eq!(scope.residue_levels(), 64);
        }
        Absence::Unknown => panic!("a complete coefficient path must certify absence"),
    }
    assert!(locator.parabolic_node_count() <= 1 + 64 * locator.len());
    assert!(locator.check_coverage());
}

/// Two keys with the same scalar coefficient `a` and different nilpotent `b` must not alias and
/// must not be falsely absent. The fiber is the scalar `a = fstHom`; `b` never decides identity.
#[test]
fn parabolic_scalar_collision_separates_by_identity() {
    let p = profile();
    let mut locator = Locator::<Vec<u8>, i32>::new(p);
    let scalar = 0x0123_4567_89AB_CDEF;
    let key_one = dual_key(scalar, 0x1111_1111, 0xA1);
    let key_two = dual_key(scalar, 0x2222_2222, 0xB2);

    // Same scalar `a`, different nilpotent `b`: one fiber, two keys.
    assert_eq!(
        locator.insert(parabolic(scalar, 0x1111_1111), &key_one, 1),
        Ok(Inserted::Fresh)
    );
    assert_eq!(
        locator.insert(parabolic(scalar, 0x2222_2222), &key_two, 2),
        Ok(Inserted::Fresh)
    );
    assert_eq!(locator.len(), 2);
    assert_present(&locator, parabolic(scalar, 0x1111_1111), &key_one, &1);
    assert_present(&locator, parabolic(scalar, 0x2222_2222), &key_two, &2);

    // `b` never decides identity or absence: each stored key is found under a different nilpotent.
    assert_present(&locator, parabolic(scalar, 0x9999_9999), &key_one, &1);
    assert_present(&locator, parabolic(scalar, 0x9999_9999), &key_two, &2);

    // A third key with the same `a` is absent even though its `b` matches a stored key's.
    let absent = dual_key(scalar, 0x2222_2222, 0xC3);
    assert_absent(&locator, parabolic(scalar, 0x2222_2222), &absent);
    assert!(locator.check_coverage());
}

/// Equal residue-path values across faces never alias keys: the shared identity trie is entered
/// from a face-specific residue root, and the terminal checks the stored face.
#[test]
fn parabolic_and_hyperbolic_faces_never_alias() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    // The same key at the same numeric index value on both faces.
    assert_eq!(
        locator.insert(parabolic(7, 0), &7, 700),
        Ok(Inserted::Fresh)
    );
    assert_eq!(locator.insert(hyperbolic(7), &7, 70), Ok(Inserted::Fresh));
    assert_eq!(locator.len(), 2);

    assert_present(&locator, parabolic(7, 0), &7, &700);
    assert_present(&locator, hyperbolic(7), &7, &70);

    // A key present on one face is absent on the other face's distinct index.
    assert_absent(&locator, hyperbolic(8), &8);
    assert_absent(&locator, parabolic(8, 0), &8);
    assert!(locator.check_coverage());
}

#[test]
fn absent_requires_the_coverage_invariant() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    locator.insert(hyperbolic(7), &7, 70).unwrap();

    // Established coverage certifies absence.
    assert_absent(&locator, hyperbolic(8), &8);

    // With the invariant withdrawn the same query is Unknown, never a false absence.
    locator.invalidate_coverage();
    assert!(!locator.coverage_established());
    assert_eq!(locator.lookup_absent(hyperbolic(8), &8), Absence::Unknown);
    assert_eq!(locator.get(hyperbolic(8), &8), Lookup::Unknown);

    // Presence is exact and does not depend on coverage.
    assert_present(&locator, hyperbolic(7), &7, &70);

    // The structure itself is intact; only the certificate is withheld.
    assert!(locator.check_coverage());
}

#[test]
fn capacity_refusal_leaves_state_intact() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::with_max_reps(p, 1);
    locator.insert(hyperbolic(11), &11, 110).unwrap();
    let before = (
        locator.residue_node_count(),
        locator.parabolic_node_count(),
        locator.identity_node_count(),
        locator.edge_count(),
    );

    assert_eq!(
        locator.insert(hyperbolic(12), &12, 120),
        Err(Refusal::Capacity)
    );
    assert_eq!(locator.len(), 1);
    assert_eq!(
        (
            locator.residue_node_count(),
            locator.parabolic_node_count(),
            locator.identity_node_count(),
            locator.edge_count(),
        ),
        before
    );
    assert_present(&locator, hyperbolic(11), &11, &110);
    assert_absent(&locator, hyperbolic(12), &12);
    assert!(locator.check_coverage());

    // A found key updates in place and is not refused.
    assert_eq!(
        locator.insert(hyperbolic(11), &11, 111),
        Ok(Inserted::Updated)
    );
    assert_present(&locator, hyperbolic(11), &11, &111);
}

#[test]
fn elliptic_face_has_no_locator() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    assert_eq!(
        locator.insert(FaceIndex::Elliptic, &3, 30),
        Err(Refusal::WrongFace)
    );
    assert!(locator.is_empty());
    assert_eq!(locator.get(FaceIndex::Elliptic, &3), Lookup::Unknown);
    assert_eq!(
        locator.lookup_absent(FaceIndex::Elliptic, &3),
        Absence::Unknown
    );
}

#[test]
fn word_identity_reuses_the_i64wordcodec() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    for (index, key) in [0i64, 1, -1, 7, i64::MIN, i64::MAX, (1i64 << 32) + 5]
        .into_iter()
        .enumerate()
    {
        // The retained identity is exactly the reviewed codec's identity word.
        let retained = key.retain();
        let codec = <I64WordCodec as ExactCodec<i64, WordHome>>::encode(
            Sigma::Hyperbolic,
            &key,
            &TableAddress {
                place: 0,
                direct_word: None,
            },
        );
        assert_eq!(retained, codec);
        let decoded =
            <I64WordCodec as ExactCodec<i64, WordHome>>::decode(Sigma::Hyperbolic, retained);
        assert_eq!(decoded, key);

        // The identity path is the eight big-endian bytes of the original bit pattern.
        assert_eq!(key.encoding_len(), 8);
        for (byte_index, byte) in key.cast_unsigned().to_be_bytes().iter().enumerate() {
            assert_eq!(key.encoding_byte(byte_index), *byte);
        }

        let value = i32::try_from(index).unwrap();
        let residue = residue_of(key.cast_unsigned());
        locator
            .insert(FaceIndex::Hyperbolic(residue), &key, value)
            .unwrap();
        assert_present(&locator, FaceIndex::Hyperbolic(residue), &key, &value);
    }
    assert_eq!(locator.len(), 7);
    assert!(locator.check_coverage());
}

/// The exact-identity byte law for a small integer key: `encoding_len` is the full fixed width and
/// `encoding_byte(i)` is byte `i` of the original-bit big-endian pattern, with an exact retained
/// identity. A control that drops the top identity byte or shortens the width mismatches here.
fn assert_integer_identity_bytes<K>(key: &K, expected: &[u8])
where
    K: ExactIdentity,
{
    assert_eq!(key.encoding_len(), expected.len());
    for (index, byte) in expected.iter().enumerate() {
        assert_eq!(key.encoding_byte(index), *byte);
    }
    let retained = key.retain();
    assert!(key.matches_retained(&retained));
    assert_eq!(K::retained_encoding_len(&retained), expected.len());
    for (index, byte) in expected.iter().enumerate() {
        assert_eq!(K::retained_encoding_byte(&retained, index), *byte);
    }
}

#[test]
fn small_integer_identity_is_the_original_bit_pattern() {
    assert_integer_identity_bytes(&0i8, &[0x00]);
    assert_integer_identity_bytes(&1i8, &[0x01]);
    assert_integer_identity_bytes(&-1i8, &[0xFF]);
    assert_integer_identity_bytes(&i8::MIN, &[0x80]);
    assert_integer_identity_bytes(&i8::MAX, &[0x7F]);

    assert_integer_identity_bytes(&0u8, &[0x00]);
    assert_integer_identity_bytes(&0x80u8, &[0x80]);
    assert_integer_identity_bytes(&u8::MAX, &[0xFF]);

    assert_integer_identity_bytes(&0i32, &[0x00, 0x00, 0x00, 0x00]);
    assert_integer_identity_bytes(&1i32, &[0x00, 0x00, 0x00, 0x01]);
    assert_integer_identity_bytes(&-1i32, &[0xFF, 0xFF, 0xFF, 0xFF]);
    assert_integer_identity_bytes(&i32::MIN, &[0x80, 0x00, 0x00, 0x00]);
    assert_integer_identity_bytes(&i32::MAX, &[0x7F, 0xFF, 0xFF, 0xFF]);

    assert_integer_identity_bytes(&0u32, &[0x00, 0x00, 0x00, 0x00]);
    assert_integer_identity_bytes(&0x8000_0000u32, &[0x80, 0x00, 0x00, 0x00]);
    assert_integer_identity_bytes(&u32::MAX, &[0xFF, 0xFF, 0xFF, 0xFF]);
}

/// Two distinct keys that share one parabolic scalar fiber and differ only in their top identity
/// byte must not alias: the scalar chooses the fiber, the exact identity alone separates the keys.
/// A control that drops the top identity byte makes `high` reach `low`'s terminal, so the second
/// insert is refused and the presence assertions fail.
fn assert_high_byte_separates<K: ExactIdentity>(low: &K, high: &K, absent: &K) {
    let p = profile();
    let mut locator = Locator::<K, i32>::new(p);
    let index = parabolic(0x0123_4567_89AB_CDEF, 0);
    assert_eq!(locator.insert(index, low, 10), Ok(Inserted::Fresh));
    assert_eq!(locator.insert(index, high, 20), Ok(Inserted::Fresh));
    assert_eq!(locator.len(), 2);
    assert_present(&locator, index, low, &10);
    assert_present(&locator, index, high, &20);
    assert_absent(&locator, index, absent);
    assert!(locator.check_coverage());
}

#[test]
fn small_integer_high_byte_distinguishes_identity() {
    assert_high_byte_separates(&0x0000_0001i32, &0x0100_0001i32, &0x0200_0001i32);
    assert_high_byte_separates(&0x01i8, &-127i8, &0x41i8);
    assert_high_byte_separates(&0x0000_0001u32, &0x0100_0001u32, &0x0200_0001u32);
    assert_high_byte_separates(&0x01u8, &0x81u8, &0x41u8);
}

#[test]
fn byte_identity_paths_handle_long_common_prefixes() {
    let p = profile();
    let mut locator = Locator::<Vec<u8>, i32>::new(p);
    let index = parabolic(12_345, 0);
    let prefix = vec![0xABu8; 64];

    for tail in 0u8..40 {
        let mut key = prefix.clone();
        key.push(tail);
        locator.insert(index, &key, i32::from(tail)).unwrap();
    }
    assert_eq!(locator.len(), 40);
    for tail in 0u8..40 {
        let mut key = prefix.clone();
        key.push(tail);
        assert_present(&locator, index, &key, &i32::from(tail));
    }

    let mut absent = prefix.clone();
    absent.push(200);
    assert_absent(&locator, index, &absent);
    assert!(locator.check_coverage());
}

#[test]
fn identity_node_fanout_is_bounded_by_256() {
    let p = profile();
    let mut locator = Locator::<Vec<u8>, i32>::new(p);
    let index = hyperbolic(9);
    for byte in 0u16..=255 {
        let tail = u8::try_from(byte).unwrap();
        let key = vec![0x00, tail];
        locator.insert(index, &key, i32::from(tail)).unwrap();
    }
    assert_eq!(locator.len(), 256);
    assert_eq!(locator.max_identity_fanout(), 256);
    assert_eq!(locator.identity_node_count(), 258); // root + branch + 256 leaves
    assert_eq!(locator.edge_count(), 257); // 1 root edge + 256 branch edges
    assert!(locator.check_coverage());

    for byte in 0u16..=255 {
        let tail = u8::try_from(byte).unwrap();
        let key = vec![0x00, tail];
        assert_present(&locator, index, &key, &i32::from(tail));
    }
}

#[test]
fn occupied_nodes_respect_the_design_bound() {
    let p = profile();
    let mut locator = Locator::<Vec<u8>, i32>::new(p);
    let mut identity_bytes = 0usize;
    for index in 0u32..100 {
        let key = index.to_be_bytes().to_vec();
        identity_bytes += key.len();
        locator
            .insert(
                FaceIndex::Hyperbolic(Residue::reduce(p, u64::from(index))),
                &key,
                i32::try_from(index).unwrap(),
            )
            .unwrap();
    }
    let n = locator.len();
    assert!(locator.residue_node_count() <= 1 + 40 * n);
    assert!(locator.parabolic_node_count() <= 1 + 64 * n);
    assert!(locator.identity_node_count() <= n + identity_bytes);
    assert!(locator.edge_count() <= n + identity_bytes);
    assert!(locator.max_identity_fanout() <= 256);
    assert!(locator.check_coverage());
}

/// A deliberately broken identity: the top byte is dropped from the path, so two distinct keys
/// share one byte path while their retained identities differ. The [`ExactIdentity`] law forbids
/// this; the locator must fail closed (`Unknown`), not answer from the stale terminal.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DroppedByte(u64);

impl ExactIdentity for DroppedByte {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        self.0
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        self.0
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        self.0 == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        if index == 0 {
            0
        } else {
            self.0.to_be_bytes()[index]
        }
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        if index == 0 {
            0
        } else {
            retained.to_be_bytes()[index]
        }
    }
}

#[test]
fn stale_identity_is_unknown_not_absent() {
    let p = profile();
    let mut locator = Locator::<DroppedByte, i32>::new(p);
    let index = hyperbolic(77);
    let stored = DroppedByte(0x0100_0000_0000_0001);
    let colliding = DroppedByte(0x0200_0000_0000_0001);
    locator.insert(index, &stored, 1).unwrap();
    assert!(locator.check_coverage());

    // `colliding` reaches the same terminal as `stored` but is a different key. The locator must
    // neither return the stored value nor certify absence.
    assert_eq!(locator.get(index, &colliding), Lookup::Unknown);
    assert_eq!(locator.lookup_absent(index, &colliding), Absence::Unknown);

    // The identity mismatch is confined to the colliding query: the stored key stays exact.
    assert_present(&locator, index, &stored, &1);
}

/// The reservation seam forces `Refusal::AllocationFailed` before any node, edge or payload is
/// written: the committed counts, coverage assertion and links are unchanged, the refused key is
/// still absent, and the committed key still reads.
#[cfg(feature = "injection")]
#[test]
fn forced_allocation_refusal_stutters_the_committed_state() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    locator.insert(hyperbolic(21), &21, 210).unwrap();
    let before = committed_state(&locator);
    let new_key = 22i64;

    reservation_failure::arm(Refusal::AllocationFailed);
    let refused = locator.insert(hyperbolic(new_key.cast_unsigned()), &new_key, 220);
    reservation_failure::disarm();

    assert_eq!(refused, Err(Refusal::AllocationFailed));
    assert_eq!(
        committed_state(&locator),
        before,
        "a forced allocation refusal must leave every committed count and coverage unchanged"
    );
    // No partial publication: the refused key is not present, and the committed key is untouched.
    assert_absent(&locator, hyperbolic(new_key.cast_unsigned()), &new_key);
    assert_present(&locator, hyperbolic(21), &21, &210);
    assert!(locator.check_coverage());
}

/// The same seam on the parabolic path: reserve-all is refused before any coefficient node,
/// identity node, edge or payload is written.
#[cfg(feature = "injection")]
#[test]
fn forced_allocation_refusal_stutters_the_parabolic_state() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    locator.insert(parabolic(21, 0), &21, 210).unwrap();
    let before = committed_state(&locator);

    reservation_failure::arm(Refusal::AllocationFailed);
    let refused = locator.insert(parabolic(22, 0), &22, 220);
    reservation_failure::disarm();

    assert_eq!(refused, Err(Refusal::AllocationFailed));
    assert_eq!(committed_state(&locator), before);
    assert_absent(&locator, parabolic(22, 0), &22);
    assert_present(&locator, parabolic(21, 0), &21, &210);
    assert!(locator.check_coverage());
}

/// The same seam forces `Refusal::SizeOverflow`, exercising the other refusal branch with the same
/// zero-effects contract.
#[cfg(feature = "injection")]
#[test]
fn forced_size_overflow_refusal_stutters_the_committed_state() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    locator.insert(hyperbolic(31), &31, 310).unwrap();
    let before = committed_state(&locator);
    let new_key = 32i64;

    reservation_failure::arm(Refusal::SizeOverflow);
    let refused = locator.insert(hyperbolic(new_key.cast_unsigned()), &new_key, 320);
    reservation_failure::disarm();

    assert_eq!(refused, Err(Refusal::SizeOverflow));
    assert_eq!(committed_state(&locator), before);
    assert_absent(&locator, hyperbolic(new_key.cast_unsigned()), &new_key);
    assert_present(&locator, hyperbolic(31), &31, &310);
    assert!(locator.check_coverage());
}

/// The arm is one-shot and consumed by the refused insert: the very next insert of the same key
/// succeeds and publishes normally, with no residual arm to leak into a later operation.
#[cfg(feature = "injection")]
#[test]
fn forced_refusal_is_one_shot_and_the_next_insert_succeeds() {
    let p = profile();
    let mut locator = Locator::<i64, i32>::new(p);
    let key = 41i64;

    reservation_failure::arm(Refusal::AllocationFailed);
    assert_eq!(
        locator.insert(parabolic(key.cast_unsigned(), 0), &key, 410),
        Err(Refusal::AllocationFailed)
    );
    assert!(locator.is_empty());

    // No re-arm: the consumed arm does not fire again.
    assert_eq!(
        locator.insert(parabolic(key.cast_unsigned(), 0), &key, 410),
        Ok(Inserted::Fresh)
    );
    assert_present(&locator, parabolic(key.cast_unsigned(), 0), &key, &410);
    assert!(locator.check_coverage());
}

#[cfg(feature = "injection")]
thread_local! {
    /// How many times [`CountedKey::retain_reserved`] has been entered on this thread.
    static RETAIN_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// A byte key whose retention is observable, so a late reservation failure can be shown to occur
/// before the retained identity is filled rather than merely before the node lengths change.
#[cfg(feature = "injection")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct CountedKey(Vec<u8>);

#[cfg(feature = "injection")]
#[allow(
    clippy::ptr_arg,
    reason = "the retained identity is the byte vector itself; the trait fixes `&Self::Retained`"
)]
impl ExactIdentity for CountedKey {
    type Retained = Vec<u8>;
    type Reserved = Vec<u8>;

    fn retain(&self) -> Vec<u8> {
        self.0.clone()
    }

    fn reserve_retain(&self) -> Result<Vec<u8>, Refusal> {
        let mut reserved = Vec::new();
        reserved
            .try_reserve_exact(self.0.len())
            .map_err(|_| Refusal::AllocationFailed)?;
        Ok(reserved)
    }

    fn retain_reserved(&self, mut reserved: Vec<u8>) -> Vec<u8> {
        RETAIN_CALLS.with(|calls| calls.set(calls.get() + 1));
        reserved.extend_from_slice(&self.0);
        reserved
    }

    fn matches_retained(&self, retained: &Vec<u8>) -> bool {
        &self.0 == retained
    }

    fn encoding_len(&self) -> usize {
        self.0.len()
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.0[index]
    }

    fn retained_encoding_len(retained: &Vec<u8>) -> usize {
        retained.len()
    }

    fn retained_encoding_byte(retained: &Vec<u8>, index: usize) -> u8 {
        retained[index]
    }
}

/// The late-edge seam fires at the real identity-edge reservation, before the retained identity is
/// filled. A long-prefix insertion reserves a whole chain of identity nodes and edges; the forced
/// failure must leave the committed state, counts, coverage assertion and every reverse link
/// unchanged, publish nothing, keep the refused key `Absent`, and not disturb a later successful
/// insert. The rejecting control moves one edge reservation after `retain_reserved`; the
/// zero-retain assertion below then goes RED.
#[cfg(feature = "injection")]
#[test]
fn forced_late_edge_refusal_precedes_retention_and_publishes_nothing() {
    let p = profile();
    let mut locator = Locator::<CountedKey, i32>::new(p);
    let committed = CountedKey(vec![0x11, 0x22, 0x33, 0x44]);
    locator.insert(parabolic(0x1111, 0), &committed, 1).unwrap();
    let before = committed_state(&locator);
    assert!(locator.check_coverage());

    // A long-prefix key: the fresh insert reserves a chain of identity nodes and edges.
    let fresh = CountedKey(vec![0xAB; 96]);
    RETAIN_CALLS.with(|calls| calls.set(0));
    reservation_failure::arm_late_edge(Refusal::AllocationFailed);
    let refused = locator.insert(parabolic(0xABAB, 0), &fresh, 2);
    reservation_failure::disarm();

    assert_eq!(refused, Err(Refusal::AllocationFailed));
    assert_eq!(
        RETAIN_CALLS.with(Cell::get),
        0,
        "a late-edge refusal must occur before the retained identity is filled"
    );
    assert_eq!(
        committed_state(&locator),
        before,
        "a forced late-edge refusal must leave every committed count and coverage unchanged"
    );
    assert!(
        locator.check_coverage(),
        "every committed reverse link must be unchanged"
    );
    assert_absent(&locator, parabolic(0xABAB, 0), &fresh);
    assert_present(&locator, parabolic(0x1111, 0), &committed, &1);

    // The consumed one-shot arm does not disturb a later successful insert of the same key.
    assert_eq!(
        locator.insert(parabolic(0xABAB, 0), &fresh, 2),
        Ok(Inserted::Fresh)
    );
    assert_present(&locator, parabolic(0xABAB, 0), &fresh, &2);
    assert!(locator.check_coverage());
}

/// A same-residue identity miss is certified only because the canonical 40-level residue path
/// completed. The rejecting control shortens the reported residue path on an identity miss (skips a
/// level); the `residue_levels != levels` guard then withholds the certificate and this test is RED.
#[test]
fn identity_miss_requires_the_complete_residue_path() {
    let p = profile();
    let mut locator = Locator::<Vec<u8>, i32>::new(p);
    let index = hyperbolic(4_242);
    let stored = vec![0x00, 0x00];
    locator.insert(index, &stored, 1).unwrap();

    // Same residue and same first identity byte, a different second byte: a genuine identity miss
    // under a complete residue path.
    let absent = vec![0x00, 0x01];
    match locator.lookup_absent(index, &absent) {
        Absence::Absent(scope) => {
            assert_eq!(scope.index(), index);
            assert_eq!(scope.residue(), Some(residue_of(4_242)));
            assert_eq!(scope.residue_levels(), 40);
            assert_eq!(scope.identity_bytes(), 1);
        }
        Absence::Unknown => panic!("a complete residue path must certify absence"),
    }
    assert!(locator.check_coverage());
}

/// The found-or-linked reservation seam on the parallel assembly path: the second occurrence of a
/// key returns the first occurrence's terminal handle and adds no reverse link, residue node,
/// identity node or edge, while the elliptic face records no handle at all. The positive tail shows
/// the counters move when a fresh key really stages, so the count comparison is not vacuous. The
/// rejecting control makes the found arm never fire (always stage and publish); the repeat then
/// returns a fresh handle, grows the reverse links and both assertions below go RED.
#[cfg(feature = "injection")]
#[test]
fn preassemble_repeat_key_returns_the_first_handle_and_adds_no_link_node_or_edge() {
    let mut table = Table::<i64, i64, WordHome, 16>::new(0, WordHome);
    let mixed = [
        ((Sigma::Parabolic, 7i64), 41i64),
        ((Sigma::Hyperbolic, 7i64), 42),
        ((Sigma::Elliptic, 7i64), 43),
    ];
    let first = table
        .preassemble_handles(&mixed)
        .expect("the first occurrence reserves and links every face");
    assert_eq!(first[2], None, "the elliptic face has no locator handle");
    let parabolic = first[0].expect("the parabolic face records a handle");
    let hyperbolic = first[1].expect("the hyperbolic face records a handle");
    assert_ne!(
        parabolic, hyperbolic,
        "the two faces never alias one terminal"
    );
    let before = table.locator_counts();

    let again = table
        .preassemble_handles(&mixed)
        .expect("a repeat key is found, never reserved");
    assert_eq!(
        again[0],
        Some(parabolic),
        "the repeated parabolic key returns the first handle"
    );
    assert_eq!(
        again[1],
        Some(hyperbolic),
        "the repeated hyperbolic key returns the first handle"
    );
    assert_eq!(again[2], None, "the elliptic entry stays handle-free");
    assert_eq!(
        table.locator_counts(),
        before,
        "a repeat key adds no reverse link, residue node, identity node or edge"
    );

    // The counters are live: a genuinely fresh key gets its own handle and moves them.
    let fresh = table
        .preassemble_handles(&[((Sigma::Parabolic, 8i64), 44i64)])
        .expect("a fresh key stages and links");
    assert_ne!(fresh[0], Some(parabolic), "a fresh key gets its own handle");
    assert_ne!(
        table.locator_counts(),
        before,
        "the countable directory state moves when nodes, edges and links are added"
    );
}
