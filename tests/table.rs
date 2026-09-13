use apeiron::{merge, Table};

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
