//! Evidence for "three storage faces, one merge".
//!
//! The thesis is that hashtrinity's three map types and its single parameterised merge are not
//! three-and-one by design but three-and-one because the algebra is. These tests are the
//! empirical half; the proof, if it lands, belongs in sanguine.
//!
//! The argument has two legs and they pull in opposite directions, which is the whole point:
//!
//! - **Three, because the discard varies.** What a table can do to a key -- regenerate it, carry
//!   it, or destroy it -- is decided by the discriminant, and `classification` says there are
//!   exactly three discriminants up to sign.
//! - **One, because commutativity does not vary.** Order-free accumulation needs `mul_comm` and
//!   `mul_assoc`, and those hold on every face with no hypothesis. So there is nothing for a
//!   second merge to be: a merge cannot be licensed on one face and refused on another.

use apeiron::cell::{fold, Cell, Sigma};
use apeiron::keystore::{after_storing, key_cell, Keystore};
use apeiron::scalar::Ring;

const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

/// Three faces, three keystores, and the map between them is a bijection.
///
/// If a fourth storage face were wanted it would need a fourth discriminant, and the
/// classification theorem says there is not one.
#[test]
fn there_are_exactly_three_storage_faces_because_there_are_three_discriminants() {
    let stores: Vec<Keystore> = FACES.iter().copied().map(Keystore::of).collect();
    assert_eq!(stores.len(), 3);

    // Injective: no two faces give the same keystore.
    for (i, a) in stores.iter().enumerate() {
        for b in &stores[i + 1..] {
            assert_ne!(a, b, "two discriminants collapsed to one storage face");
        }
    }
    // And it round-trips, so the keystore carries the discriminant and nothing is invented.
    for sigma in FACES {
        assert_eq!(Keystore::of(sigma).sigma(), sigma);
    }
}

/// Each keystore does exactly what its face licenses, and nothing it does not.
#[test]
fn what_survives_a_store_is_what_the_face_permits() {
    // Being: the address determines the key, so nothing needs storing and nothing is lost.
    assert!(!Keystore::Being.stores_the_key());
    assert!(Keystore::Being.key_survives());

    // B/U: the middle face carries the key, because it can neither regenerate nor discard it.
    assert!(Keystore::BeingOverUnbeing.stores_the_key());
    assert!(Keystore::BeingOverUnbeing.key_survives());

    // NB: the key does not survive. This is available only where a discard is.
    assert!(!Keystore::NonBeing.key_survives());
    assert!(Keystore::NonBeing.sigma().discards());

    for face in [Keystore::Being, Keystore::BeingOverUnbeing] {
        assert!(
            !face.sigma().discards(),
            "{face:?} sits on a face with no discard, so it cannot lose the key"
        );
    }
}

/// The lossy face loses, the other two do not -- checked on the cells, not on the labels.
#[test]
fn only_the_non_being_face_destroys_information() {
    let keys: Vec<Cell<i64>> = (-4..5)
        .flat_map(|addr| (-4..5).map(move |res| key_cell(addr, res)))
        .collect();

    for face in [Keystore::Being, Keystore::BeingOverUnbeing] {
        for &k in &keys {
            assert_eq!(after_storing(face, k), k, "{face:?} altered a key");
        }
    }

    // NonBeing collapses: distinct keys become the same stored cell, which is precisely what
    // makes it lossy and what a 2^-64 false positive rate is a bound on.
    let stored: Vec<Cell<i64>> = keys
        .iter()
        .map(|&k| after_storing(Keystore::NonBeing, k))
        .collect();
    let distinct: std::collections::HashSet<_> = stored.iter().map(|c| (c.a, c.b)).collect();
    assert!(
        distinct.len() < keys.len(),
        "NonBeing must collapse distinct keys; {} keys gave {} cells",
        keys.len(),
        distinct.len()
    );

    // And what it destroyed is a whole channel: every stored cell is degenerate.
    for c in &stored {
        assert_eq!(c.norm_form(Sigma::Hyperbolic), 0, "the survivor lies on the null cone");
        assert_eq!(c.lower(Sigma::Hyperbolic), Some(0), "the lower lane is gone");
    }
}

/// The other leg: there is nothing for a second merge to be.
///
/// A merge that were licensed on one face and refused on another would make the merge vary with
/// the dial, which is what "more than one merge" would have to mean. It does not: the fold is
/// order-free on all three, so the same merge is legal everywhere.
#[test]
fn the_merge_does_not_vary_with_the_dial() {
    let cells: Vec<Cell<i64>> = vec![
        Cell::of(1, 1),
        Cell::of(2, -1),
        Cell::of(0, 3),
        Cell::of(-2, 1),
        Cell::of(3, 2),
    ];

    for sigma in FACES {
        let base = fold(sigma, &cells);
        // Every permutation of five, on every face.
        let mut idx = [0usize, 1, 2, 3, 4];
        let mut counter = [0usize; 5];
        let (mut i, mut seen) = (0, 0);
        while i < 5 {
            if counter[i] < i {
                idx.swap(if i % 2 == 0 { 0 } else { counter[i] }, i);
                let permuted: Vec<Cell<i64>> = idx.iter().map(|&k| cells[k]).collect();
                assert_eq!(fold(sigma, &permuted), base, "{sigma:?} order-dependent");
                seen += 1;
                counter[i] += 1;
                i = 0;
            } else {
                counter[i] = 0;
                i += 1;
            }
        }
        assert_eq!(seen, 119, "all permutations but the identity");
    }
}

/// The asymmetry, stated as one assertion: the keystore is a function of the dial and the merge
/// is not.
#[test]
fn three_keystores_vary_with_the_dial_and_the_one_merge_does_not() {
    let keystores: std::collections::HashSet<Keystore> =
        FACES.iter().copied().map(Keystore::of).collect();
    assert_eq!(keystores.len(), 3, "the keystore varies with the dial");

    let order_free: std::collections::HashSet<bool> = FACES
        .iter()
        .map(|&s| apeiron::cell::fold_is_order_free(s))
        .collect();
    assert_eq!(
        order_free,
        std::collections::HashSet::from([true]),
        "the merge's licence does not vary with the dial"
    );
}
