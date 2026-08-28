//! The table counts, exactly, whatever the stream and however it is scheduled.
//!
//! What is at stake is the elliptic guarantee reaching the table's door. `finish` is a bijection,
//! so carrying `finish(a)` in the slot instead of `a` is supposed to change *where* a key lands
//! and never *whether* two keys are the same key. That is only true if nothing downstream
//! confuses a finished address with a raw one, so it is asserted here against a reference count
//! rather than argued.
//!
//! `Table.build_eq_of_key_streams_eq` is the licence for the sharding and for the per-worker
//! chunking: a build depends on the stream only through each key's values in their stream order,
//! so cutting the stream into contiguous chunks and regrouping by key changes nothing.

use std::collections::HashMap;

use apeiron::coord::{i21, Packed};
use apeiron::table::CountTable;

fn reference(addrs: &[u64]) -> HashMap<u64, i64> {
    let mut m = HashMap::new();
    for &a in addrs {
        *m.entry(a).or_insert(0) += 1;
    }
    m
}

fn check(name: &str, addrs: &[u64]) {
    let want = reference(addrs);
    let mut probe: Vec<u64> = want.keys().copied().collect();
    probe.sort_unstable();

    for threads in [1usize, 2, 3, 7, 16] {
        let t = CountTable::build_parallel(addrs, 16, threads);
        for &a in &probe {
            assert_eq!(t.get(a), want[&a], "{name}: key {a:#x} at {threads} threads");
        }
        // A key that was never inserted counts zero, not a neighbour's count.
        for &a in &[u64::MAX, 0xDEAD_BEEF_DEAD_BEEF, 1] {
            if !want.contains_key(&a) {
                assert_eq!(t.get(a), 0, "{name}: absent key {a:#x} at {threads} threads");
            }
        }
        let total: i64 = t.sum_parallel(addrs, threads);
        let expect: i64 = addrs.iter().map(|a| want[a]).sum();
        assert_eq!(total, expect, "{name}: sum_parallel at {threads} threads");
    }

    let serial = CountTable::build(addrs, 16);
    for &a in &probe {
        assert_eq!(serial.get(a), want[&a], "{name}: serial, key {a:#x}");
    }
}

fn rng() -> impl FnMut() -> u64 {
    let mut r = 0x243F_6A88_85A3_08D3u64;
    move || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    }
}

#[test]
fn an_empty_stream_builds_a_table_that_answers_zero() {
    let t = CountTable::build(&[], 16);
    assert_eq!(t.get(0), 0);
    assert_eq!(t.get(u64::MAX), 0);
    let t = CountTable::build_parallel(&[], 16, 16);
    assert_eq!(t.get(12345), 0);
    assert_eq!(t.sum_parallel(&[1, 2, 3], 16), 0);
}

#[test]
fn a_single_key_repeated_counts_once_per_occurrence() {
    for n in [1usize, 2, 17, 1000] {
        check(&format!("one key x{n}"), &vec![0x1234_5678_9ABC_DEF0u64; n]);
    }
}

/// Includes the sentinel a slot used to be initialised with, and zero.
#[test]
fn the_extremes_of_the_word_are_ordinary_keys() {
    check("extremes", &[0, u64::MAX, 1, u64::MAX - 1, 0, u64::MAX, 1 << 63]);
}

#[test]
fn a_spatial_scene_counts_exactly() {
    let mut next = rng();
    let addrs: Vec<u64> = (0..200_000)
        .map(|_| {
            let v = next();
            let c = |s: u32| ((v >> s) as i32 % 300) - 150;
            (
                i21::new(c(0)).unwrap(),
                i21::new(c(20)).unwrap(),
                i21::new(c(40)).unwrap(),
            )
                .packed()
        })
        .collect();
    check("spatial scene", &addrs);
}

/// Every key in one shard, which is the case a fixed share of a fixed table cannot survive.
#[test]
fn a_stream_aimed_at_one_shard_still_counts_exactly() {
    let mut next = rng();
    let mut addrs = Vec::new();
    while addrs.len() < 40_000 {
        let a = next();
        if apeiron::table::shard_of(a) == 0 {
            addrs.push(a);
        }
    }
    assert!(addrs.iter().all(|&a| apeiron::table::shard_of(a) == 0));
    check("one shard", &addrs);
}

#[test]
fn counts_do_not_depend_on_the_worker_count() {
    let mut next = rng();
    let addrs: Vec<u64> = (0..100_000).map(|_| next() % 5000).collect();
    let one = CountTable::build(&addrs, 16);
    for threads in [2usize, 5, 16, 31] {
        let many = CountTable::build_parallel(&addrs, 16, threads);
        for &a in &addrs {
            assert_eq!(one.get(a), many.get(a), "key {a} at {threads} threads");
        }
        assert_eq!(one.slots(), many.slots(), "slot count at {threads} threads");
    }
}
