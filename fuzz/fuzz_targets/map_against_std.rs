//! `Map` against `std::collections::HashMap`, on programs a fuzzer writes.
//!
//! `tests/a_map_is_a_map.rs` already runs a thousand random programs through both, but it draws
//! them from a PRNG this crate wrote, and a PRNG only generates the shapes its author thought of.
//! A coverage-guided fuzzer writes the program instead, and steers it toward the branches the
//! tests do not reach -- the grow that lands exactly on a tombstone, the drain dropped mid-flight,
//! the extract that empties a region.
//!
//! std is the oracle for *behaviour*, never for layout: iteration order and capacity growth are
//! deliberately different and are not compared. What is compared is everything a caller can
//! observe about content.
//!
//! `cargo +nightly fuzz run map_against_std`
#![no_main]

use apeiron::map::Map;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

/// Keys are `u16` so the fuzzer collides them constantly. A fuzzer that has to guess 64-bit keys
/// never fills a group; one working in a 65,536-key space hits the fibers, the tombstones and the
/// growth boundaries within a few hundred operations.
type K = u16;

#[derive(Arbitrary, Debug)]
enum Op {
    Insert(K, u32),
    InsertWith(K, u32),
    Get(K),
    GetMut(K),
    ContainsKey(K),
    Remove(K),
    RemoveEntry(K),
    Entry(K, u32),
    EntryOrDefault(K),
    Retain(u8),
    ExtractIf(u8),
    Reserve(u8),
    ShrinkToFit,
    Clear,
    DrainAll,
    Len,
    IterSum,
    GetEach(Vec<K>),
    Encircle,
}

fuzz_target!(|program: Vec<Op>| {
    // A bounded program: the point is many short programs over many shapes, not one long one, and
    // an unbounded length just makes each case slower to minimise.
    if program.len() > 512 {
        return;
    }
    let mut ours: Map<K, u32> = Map::new();
    let mut theirs: HashMap<K, u32> = HashMap::new();

    for op in program {
        match op {
            Op::Insert(k, v) => {
                assert_eq!(ours.insert(k, v), theirs.insert(k, v), "insert({k}, {v})");
            }
            Op::InsertWith(k, v) => {
                // The in-place merge, which std spells as an entry dance.
                ours.insert_with(k, v, |slot, new| *slot = slot.wrapping_add(new));
                theirs
                    .entry(k)
                    .and_modify(|slot| *slot = slot.wrapping_add(v))
                    .or_insert(v);
            }
            Op::Get(k) => assert_eq!(ours.get(&k), theirs.get(&k), "get({k})"),
            Op::GetMut(k) => {
                // Mutating through the handle must land in the table, not in a copy.
                if let Some(v) = ours.get_mut(&k) {
                    *v = v.wrapping_add(1);
                }
                if let Some(v) = theirs.get_mut(&k) {
                    *v = v.wrapping_add(1);
                }
            }
            Op::ContainsKey(k) => {
                assert_eq!(ours.contains_key(&k), theirs.contains_key(&k), "contains({k})");
            }
            Op::Remove(k) => assert_eq!(ours.remove(&k), theirs.remove(&k), "remove({k})"),
            Op::RemoveEntry(k) => {
                assert_eq!(ours.remove_entry(&k), theirs.remove_entry(&k), "remove_entry({k})");
            }
            Op::Entry(k, v) => {
                *ours.entry(k).or_insert(v) = v;
                *theirs.entry(k).or_insert(v) = v;
            }
            Op::EntryOrDefault(k) => {
                assert_eq!(*ours.entry(k).or_default(), *theirs.entry(k).or_default());
            }
            Op::Retain(m) => {
                let m = u32::from(m).max(1);
                ours.retain(|k, _| u32::from(*k) % m == 0);
                theirs.retain(|k, _| u32::from(*k) % m == 0);
            }
            Op::ExtractIf(m) => {
                let m = u32::from(m).max(1);
                let mut a: Vec<(K, u32)> =
                    ours.extract_if(|k, _| u32::from(*k) % m == 0).collect();
                // Spelled out: `HashMap::extract_if` is newer than this crate's MSRV, and the
                // definition is the better oracle anyway.
                let mut b: Vec<(K, u32)> = Vec::new();
                theirs.retain(|&k, &mut v| {
                    if u32::from(k) % m == 0 {
                        b.push((k, v));
                        false
                    } else {
                        true
                    }
                });
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b, "extract_if(% {m})");
            }
            Op::Reserve(n) => {
                ours.reserve(n as usize);
                theirs.reserve(n as usize);
            }
            Op::ShrinkToFit => {
                ours.shrink_to_fit();
                theirs.shrink_to_fit();
            }
            Op::Clear => {
                ours.clear();
                theirs.clear();
            }
            Op::DrainAll => {
                let mut a: Vec<(K, u32)> = ours.drain().collect();
                let mut b: Vec<(K, u32)> = theirs.drain().collect();
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b, "drain");
            }
            Op::Len => assert_eq!(ours.len(), theirs.len(), "len"),
            Op::IterSum => {
                // Iteration *order* is deliberately not std's; the multiset of entries is.
                let mut a: Vec<(K, u32)> = ours.iter().map(|(k, v)| (*k, *v)).collect();
                let mut b: Vec<(K, u32)> = theirs.iter().map(|(k, v)| (*k, *v)).collect();
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b, "iter");
            }
            Op::GetEach(ref keys) => {
                if keys.len() <= 256 {
                    let got = ours.get_all(keys);
                    let want: Vec<Option<&u32>> = keys.iter().map(|k| theirs.get(k)).collect();
                    assert_eq!(got, want, "get_all");
                }
            }
            Op::Encircle => {
                let a = ours.encircle(0u64, |v| u64::from(*v), u64::wrapping_add);
                let b: u64 = theirs.values().map(|v| u64::from(*v)).sum();
                assert_eq!(a, b, "encircle");
            }
        }
        // The cheap invariants, after every single operation: this is what turns a crash-finder
        // into a correctness-finder.
        assert_eq!(ours.len(), theirs.len(), "length diverged after {op:?}");
        assert_eq!(ours.is_empty(), theirs.is_empty());
    }

    // And the full comparison once at the end.
    let mut a: Vec<(K, u32)> = ours.iter().map(|(k, v)| (*k, *v)).collect();
    let mut b: Vec<(K, u32)> = theirs.into_iter().collect();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
});
