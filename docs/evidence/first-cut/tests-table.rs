use apeiron::{merge, table::ReplacementMerge, Table};

#[test]
fn default_read_is_total_before_any_write() {
    let table = Table::<i32, _, _>::new(String::from("empty"), |_: &i32| 0);
    assert_eq!(table.get(&42), "empty");
}

#[test]
fn insert_is_new_first_and_replaces_an_existing_override() {
    let mut table = Table::<i32, _, _>::new(10_i32, |_: &i32| 0);
    table.insert_with(&|new, old| new - old, 7, 3).unwrap();
    assert_eq!(*table.get(&7), -7);
    table.insert_with(&|new, old| new - old, 7, 5).unwrap();
    assert_eq!(*table.get(&7), 12);
}

#[test]
fn replacement_merge_fresh_write_stores_new_value() {
    let mut table = Table::<i64, _, _, 1>::new(-17, |_: &i64| 0);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    assert_eq!(*table.get(&7), 41);
}

#[test]
fn replacement_merge_present_write_stores_new_value() {
    let mut table = Table::<i64, _, _, 1>::new(-17, |_: &i64| 0);
    table.insert_with(&ReplacementMerge, 7, 41).unwrap();
    table.insert_with(&ReplacementMerge, 7, -9).unwrap();
    assert_eq!(*table.get(&7), -9);
}

#[test]
fn growth_admits_far_beyond_the_old_fixed_ceiling() {
    let mut table = Table::<u64, u64, _, 16>::new(0, |k: &u64| *k);
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
        |_: &i32| 0,
    )
    .unwrap();
    assert_eq!(table.get(&1), &[2, 1]);
    assert_eq!(table.get(&2), &[9]);
}
