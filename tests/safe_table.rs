//! The safe finite table and the optimized map agree on their shared transition surface.

use apeiron::map::Map;
use apeiron::safe_table::{SafeTable, TableFull};

#[test]
fn lookup_replace_and_remove_have_one_meaning() {
    let mut table: SafeTable<u64, i64, 2> = SafeTable::new();

    assert_eq!(table.try_insert(7, 10), Ok(None));
    assert_eq!(table.try_insert(9, 20), Ok(None));
    assert_eq!(table.try_insert(7, 30), Ok(Some(10)));
    assert_eq!(table.get(&7), Some(&30));
    assert_eq!(table.get(&9), Some(&20));
    assert_eq!(table.remove(&7), Some(30));
    assert_eq!(table.remove(&7), None);
    assert_eq!(table.get(&7), None);
    assert_eq!(table.len(), 1);
}

#[test]
fn replacement_succeeds_at_capacity_and_fresh_insertion_refuses() {
    let mut table: SafeTable<String, u64, 2> = SafeTable::new();
    table.try_insert("a".into(), 1).unwrap();
    table.try_insert("b".into(), 2).unwrap();

    assert_eq!(table.try_insert("a".into(), 3), Ok(Some(1)));
    assert_eq!(
        table.try_insert("c".into(), 4),
        Err(TableFull {
            key: "c".into(),
            value: 4,
        })
    );
    assert_eq!(table.len(), 2);
    assert_eq!(table.get(&"a".into()), Some(&3));
    assert_eq!(table.get(&"b".into()), Some(&2));
    assert_eq!(table.get(&"c".into()), None);
}

#[test]
fn zero_capacity_has_an_inhabited_refusal() {
    let mut table: SafeTable<u64, u64, 0> = SafeTable::new();
    assert_eq!(table.capacity(), 0);
    assert!(table.is_empty());
    assert_eq!(table.try_insert(1, 2), Err(TableFull { key: 1, value: 2 }));
}

#[derive(Clone, Copy)]
enum Op {
    Insert(u32, u32),
    Get(u32),
    Remove(u32),
}

fn operations() -> impl Iterator<Item = Op> {
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    (0..10_000).map(move |_| {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15).rotate_left(17);
        let key = (state % 32) as u32;
        let value = (state >> 32) as u32;
        match state % 3 {
            0 => Op::Insert(key, value),
            1 => Op::Get(key),
            _ => Op::Remove(key),
        }
    })
}

#[test]
fn production_map_matches_every_bounded_reference_transition() {
    let mut reference: SafeTable<u32, u32, 32> = SafeTable::new();
    let mut production: Map<u32, u32> = Map::with_capacity(32);

    for (step, op) in operations().enumerate() {
        match op {
            Op::Insert(key, value) => {
                assert_eq!(
                    reference.try_insert(key, value).unwrap(),
                    production.insert(key, value),
                    "insert at step {step}"
                );
            }
            Op::Get(key) => assert_eq!(
                reference.get(&key),
                production.get(&key),
                "get at step {step}"
            ),
            Op::Remove(key) => assert_eq!(
                reference.remove(&key),
                production.remove(&key),
                "remove at step {step}"
            ),
        }

        assert_eq!(reference.len(), production.len(), "length at step {step}");
        for key in 0..32 {
            assert_eq!(
                reference.get(&key),
                production.get(&key),
                "body at step {step}, key {key}"
            );
        }
    }
}
