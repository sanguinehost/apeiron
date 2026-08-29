//! The one equation the borrowed-key surface rests on.
//!
//! > `q.coord_at(pepper) == owned.coord_with(pepper)` whenever `q` names `owned`
//!
//! Everything else about `get_at`, `entry_at` and `insert_at` follows from it: if a view addressed
//! its key differently from the key, the two would probe different slots and a table filled through
//! one surface would be empty through the other. The failure is silent, and it is not the kind of
//! thing a doc comment can hold up on its own.
//!
//! It is checked here rather than proved because the interesting implementations bottom out in
//! `fold_bytes` over a slice, which is a loop over an unbounded length -- exactly the shape a
//! bounded model checker unrolls forever. The keys below cover every branch of the length ladder,
//! which is where an implementation would actually diverge.

use apeiron::coord::Coord;
use apeiron::map::{Locate, Map};

fn corpus() -> Vec<String> {
    let mut v = vec![
        String::new(),
        "a".into(),
        "ab".into(),
        "abc".into(),
        "abcd".into(),
        "abcdefg".into(),
        "abcdefgh".into(),
        "0123456789abcdef".into(),
        "0123456789abcdefg".into(),
        "the quick brown fox jumps over the lazy dog".into(),
    ];
    for i in 0..200u32 {
        v.push(format!("{}{}", "nw_wswmg".repeat((i % 5) as usize + 1), i));
    }
    v
}

#[test]
fn a_str_addresses_the_string_it_names() {
    for pepper in [0u64, 1, 0xDEAD_BEEF_CAFE_F00D, u64::MAX] {
        for s in corpus() {
            assert_eq!(
                Locate::<String>::coord_at(&s.as_str(), pepper),
                s.coord_with(pepper),
                "{s:?} at pepper {pepper:#x}"
            );
        }
    }
}

#[test]
fn a_byte_slice_addresses_the_vec_it_names() {
    for pepper in [0u64, 7, u64::MAX] {
        for s in corpus() {
            let owned = s.clone().into_bytes();
            assert_eq!(
                Locate::<Vec<u8>>::coord_at(&owned.as_slice(), pepper),
                owned.coord_with(pepper),
                "{s:?}"
            );
        }
    }
}

/// The composite case, which is the one that pays: a `(String, String)` key probed with
/// `(&str, &str)` allocates nothing where the mechanical port allocates twice.
#[test]
fn a_pair_of_views_addresses_the_pair_of_keys_it_names() {
    for pepper in [0u64, 0x5EED] {
        for a in corpus().iter().take(40) {
            for b in corpus().iter().rev().take(40) {
                let owned = (a.clone(), b.clone());
                assert_eq!(
                    Locate::<(String, String)>::coord_at(&(a.as_str(), b.as_str()), pepper),
                    owned.coord_with(pepper),
                    "({a:?}, {b:?})"
                );
            }
        }
    }
}

/// The equation, exercised through the table rather than through the fold: a map filled by one
/// surface must be readable through the other, in both directions.
#[test]
fn the_two_surfaces_see_one_table() {
    for pepper in [0u64, 0xABCD_1234_5678_9EF0] {
        let mut m: Map<String, u32> = Map::with_pepper(pepper);
        for (i, s) in corpus().into_iter().enumerate() {
            // Half through the owned surface, half through the borrowed one.
            if i % 2 == 0 {
                m.insert(s.clone(), i as u32);
            } else {
                m.insert_at(s.as_str(), i as u32);
            }
        }
        for (i, s) in corpus().into_iter().enumerate() {
            assert_eq!(m.get(&s), Some(&(i as u32)), "owned read of {s:?}");
            assert_eq!(m.get_at(s.as_str()), Some(&(i as u32)), "borrowed read of {s:?}");
            assert!(m.contains_at(s.as_str()));
        }
        // And removal, which has to find the same slot the insert took.
        for (i, s) in corpus().into_iter().enumerate() {
            let got = if i % 2 == 0 {
                m.remove(&s)
            } else {
                m.remove_at(s.as_str())
            };
            assert_eq!(got, Some(i as u32), "{s:?}");
        }
        assert!(m.is_empty());
    }
}

#[test]
fn a_pair_table_is_probed_without_allocating() {
    let mut m: Map<(String, String), i64> = Map::new();
    m.insert(("area".into(), "tile".into()), 1);
    *m.entry_at(("area", "tile")).or_insert(0) += 41;
    assert_eq!(m.get_at(("area", "tile")), Some(&42));
    assert_eq!(m.get(&("area".into(), "tile".into())), Some(&42));
    assert_eq!(m.get_at(("area", "other")), None);
    assert_eq!(m.len(), 1);
}

/// A pepper changes addresses, so it must change them *consistently*: two maps with different
/// peppers hold the same entries and answer the same questions, and only their layout differs.
#[test]
fn a_pepper_changes_the_layout_and_nothing_else() {
    let build = |pepper: u64| {
        let mut m: Map<String, u32> = Map::with_pepper(pepper);
        for (i, s) in corpus().into_iter().enumerate() {
            m.insert(s, i as u32);
        }
        m
    };
    let plain = build(0);
    let salted = build(0x9999_1111_2222_3333);
    assert_eq!(plain.len(), salted.len());
    for (i, s) in corpus().into_iter().enumerate() {
        assert_eq!(plain.get(&s), Some(&(i as u32)));
        assert_eq!(salted.get(&s), Some(&(i as u32)));
    }
    let order = |m: &Map<String, u32>| m.keys().cloned().collect::<Vec<_>>();
    assert_ne!(order(&plain), order(&salted), "the pepper did not move anything");
}
