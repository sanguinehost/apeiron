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

#![expect(
    clippy::cast_possible_truncation,
    reason = "this is a test, not library code, and each lint below is listed because this file actually \
              triggers it: sizes and indices narrowed to build keys, bounded by the loop that produced them. \
              Listed individually rather than suppressed wholesale, so a pedantic lint this file does not \
              yet trigger still surfaces"
)]

use std::collections::HashMap;

use apeiron::coord::{i21, Packed};
use apeiron::table::CountTable;

/// Miri interprets rather than runs, so two hundred thousand keys is an hour. The shapes that find
/// bugs -- a shard that overflows, a worker split that reorders, a key at the extremes of the word
/// -- all happen well below that, and a native run keeps the real scale.
#[cfg(miri)]
const SCALE: usize = 100;
#[cfg(not(miri))]
const SCALE: usize = 1;

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
            assert_eq!(
                t.get(a),
                want[&a],
                "{name}: key {a:#x} at {threads} threads"
            );
        }
        // A key that was never inserted counts zero, not a neighbour's count.
        for &a in &[u64::MAX, 0xDEAD_BEEF_DEAD_BEEF, 1] {
            if !want.contains_key(&a) {
                assert_eq!(
                    t.get(a),
                    0,
                    "{name}: absent key {a:#x} at {threads} threads"
                );
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
    for n in [1usize, 2, 17, 1000 / SCALE] {
        check(&format!("one key x{n}"), &vec![0x1234_5678_9ABC_DEF0u64; n]);
    }
}

/// Includes the sentinel a slot used to be initialised with, and zero.
#[test]
fn the_extremes_of_the_word_are_ordinary_keys() {
    check(
        "extremes",
        &[0, u64::MAX, 1, u64::MAX - 1, 0, u64::MAX, 1 << 63],
    );
}

#[test]
fn a_spatial_scene_counts_exactly() {
    let mut next = rng();
    let addrs: Vec<u64> = (0..200_000 / SCALE)
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
    while addrs.len() < 40_000 / SCALE {
        let a = next();
        if apeiron::table::shard_of(apeiron::map::Salt::default(), a) == 0 {
            addrs.push(a);
        }
    }
    assert!(addrs
        .iter()
        .all(|&a| apeiron::table::shard_of(apeiron::map::Salt::default(), a) == 0));
    check("one shard", &addrs);
}

#[test]
fn counts_do_not_depend_on_the_worker_count() {
    let mut next = rng();
    let addrs: Vec<u64> = (0..100_000 / SCALE).map(|_| next() % 5000).collect();
    let one = CountTable::build(&addrs, 16);
    for threads in [2usize, 5, 16, 31] {
        let many = CountTable::build_parallel(&addrs, 16, threads);
        for &a in &addrs {
            assert_eq!(one.get(a), many.get(a), "key {a} at {threads} threads");
        }
        assert_eq!(one.slots(), many.slots(), "slot count at {threads} threads");
    }
}

/// `build_parallel` builds serially below `PARALLEL_FLOOR`, and that must be invisible.
///
/// Not run under Miri. The boundary this checks is at 65,536 keys and there is no smaller version
/// of it -- a scaled-down floor would be testing a threshold the code does not have. Miri cannot
/// interpret three tables that size, so the choice is this test or a finishing Miri run.
///
/// The fallback is a performance decision -- the partition costs more than the threads save at
/// small `n` -- so the one thing it may not do is change an answer. Checked either side of the
/// boundary, because a threshold is exactly the kind of thing that is right in the middle of a
/// range and wrong at its edge.
#[cfg_attr(
    miri,
    ignore = "the boundary is at 65,536 keys and does not scale down"
)]
#[test]
fn the_parallel_floor_does_not_change_any_count() {
    let mut next = rng();
    let floor = CountTable::PARALLEL_FLOOR;
    for n in [floor - 1, floor, floor + 1] {
        let addrs: Vec<u64> = (0..n).map(|_| next() % 5000).collect();
        let one = CountTable::build(&addrs, 16);
        for threads in [1usize, 4, 16] {
            let many = CountTable::build_parallel(&addrs, 16, threads);
            assert_eq!(
                one.slots(),
                many.slots(),
                "n {n}, {threads} threads: slot count"
            );
            for &a in &addrs {
                assert_eq!(one.get(a), many.get(a), "n {n}, {threads} threads, key {a}");
            }
        }
    }
}

/// `CountTable::FACE` is elliptic, and the claim is that a slot's identity word recovers the key.
#[test]
fn the_counting_table_regenerates_its_key_from_the_slot() {
    use apeiron::map::Salt;
    assert!(CountTable::FACE.key_survives());
    assert!(!CountTable::FACE.stores_the_key());
    for pepper in [0u64, 1, 0x5EED_1234_ABCD_9876] {
        let salt = Salt::new(pepper);
        let mut next = rng();
        for _ in 0..2_000 {
            let a = next();
            // The slot holds `id`; the address comes back off it exactly, which is what makes the
            // face elliptic rather than a discard.
            assert_eq!(
                salt.unspread(salt.spread(a).id),
                a,
                "pepper {pepper:#x}, address {a:#x}"
            );
        }
    }
}
