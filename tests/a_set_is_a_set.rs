//! The set's specification, written as `std::collections::HashSet`.
//!
//! Same shape as `tests/a_map_is_a_map.rs`: a random program of inserts, removals, retains, drains
//! and set algebra runs against both, and after **every** operation the two must agree on the
//! return value, the length, and the whole contents. A set is a thin wrapper over the map, which
//! means it inherits the map's correctness -- and also every way the wrapper itself can be wrong,
//! which is what this is for. `insert` returning the wrong boolean is invisible to the map's tests.

#![expect(
    clippy::cast_possible_truncation,
    reason = "this is a test, not library code, and each lint below is listed because this file actually \
              triggers it: sizes and indices narrowed to build keys, bounded by the loop that produced them. \
              Listed individually rather than suppressed wholesale, so a pedantic lint this file does not \
              yet trigger still surfaces"
)]

use apeiron::set::{Set, WordSet};
use std::collections::HashSet;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[cfg(miri)]
const SCALE: usize = 40;
#[cfg(not(miri))]
const SCALE: usize = 1;

#[test]
fn random_programs_agree_with_std() {
    for seed in 0..(400 / SCALE as u64).max(4) {
        let mut r = Rng(seed);
        let mut ours: Set<u32> = Set::new();
        let mut theirs: HashSet<u32> = HashSet::new();

        for step in 0..(400 / SCALE) {
            let k = r.below(64) as u32;
            match r.below(100) {
                0..=39 => assert_eq!(ours.insert(k), theirs.insert(k), "insert at {seed}/{step}"),
                40..=59 => assert_eq!(ours.remove(&k), theirs.remove(&k), "remove at {seed}/{step}"),
                60..=79 => assert_eq!(ours.contains(&k), theirs.contains(&k), "contains"),
                80..=87 => {
                    assert_eq!(ours.take(&k), theirs.take(&k), "take at {seed}/{step}");
                }
                88..=93 => {
                    let m = r.below(7).max(1) as u32;
                    ours.retain(|k| k % m != 0);
                    theirs.retain(|k| k % m != 0);
                }
                94..=96 => {
                    let mut a: Vec<u32> = ours.drain().collect();
                    let mut b: Vec<u32> = theirs.drain().collect();
                    a.sort_unstable();
                    b.sort_unstable();
                    assert_eq!(a, b, "drain at {seed}/{step}");
                }
                97..=98 => {
                    ours.clear();
                    theirs.clear();
                }
                _ => {
                    ours.shrink_to_fit();
                    theirs.shrink_to_fit();
                }
            }
            assert_eq!(ours.len(), theirs.len(), "len at {seed}/{step}");
            assert!(
                theirs.iter().all(|k| ours.contains(k)) && ours.iter().all(|k| theirs.contains(k)),
                "contents at {seed}/{step}"
            );
        }
    }
}

/// The algebra, against std's, on sets built to overlap partially.
#[test]
fn the_algebra_agrees_with_std() {
    let sorted = |s: &Set<u32>| {
        let mut v: Vec<u32> = s.iter().copied().collect();
        v.sort_unstable();
        v
    };
    let sorted_std = |s: &HashSet<u32>| {
        let mut v: Vec<u32> = s.iter().copied().collect();
        v.sort_unstable();
        v
    };

    for (an, bn, stride) in [(0u32, 0u32, 1u32), (50, 50, 1), (200, 30, 3), (17, 400, 5)] {
        let a: Set<u32> = (0..an).collect();
        let b: Set<u32> = (0..bn).map(|i| i * stride).collect();
        let sa: HashSet<u32> = (0..an).collect();
        let sb: HashSet<u32> = (0..bn).map(|i| i * stride).collect();

        assert_eq!(sorted(&a.union(&b)), sorted_std(&sa.union(&sb).copied().collect()));
        assert_eq!(
            sorted(&a.intersection(&b)),
            sorted_std(&sa.intersection(&sb).copied().collect())
        );
        assert_eq!(
            sorted(&a.difference(&b)),
            sorted_std(&sa.difference(&sb).copied().collect())
        );
        assert_eq!(
            sorted(&a.symmetric_difference(&b)),
            sorted_std(&sa.symmetric_difference(&sb).copied().collect())
        );
        assert_eq!(a.is_subset(&b), sa.is_subset(&sb), "is_subset {an}/{bn}/{stride}");
        assert_eq!(a.is_superset(&b), sa.is_superset(&sb));
        assert_eq!(a.is_disjoint(&b), sa.is_disjoint(&sb));

        // And the operators are the methods.
        assert_eq!(sorted(&(&a | &b)), sorted(&a.union(&b)));
        assert_eq!(sorted(&(&a & &b)), sorted(&a.intersection(&b)));
        assert_eq!(sorted(&(&a ^ &b)), sorted(&a.symmetric_difference(&b)));
        assert_eq!(sorted(&(&a - &b)), sorted(&a.difference(&b)));
    }
}

/// String elements through the borrowed surface -- the path a port takes, and the one that would
/// otherwise allocate on every membership test.
#[test]
fn string_elements_work_borrowed() {
    let mut ours: Set<String> = Set::new();
    let mut theirs: HashSet<String> = HashSet::new();
    for i in 0..if cfg!(miri) { 120 } else { 3000u32 } {
        let k = format!("nw_wswmg{:03}", i % 900);
        assert_eq!(ours.insert_at(k.as_str()), theirs.insert(k.clone()), "at {i}");
        assert_eq!(ours.contains_at(k.as_str()), theirs.contains(k.as_str()));
    }
    for i in 0..900u32 {
        let k = format!("nw_wswmg{i:03}");
        assert_eq!(ours.remove_at(k.as_str()), theirs.remove(k.as_str()));
    }
    assert!(ours.is_empty() && theirs.is_empty());
}

/// The elliptic face stores no element at all -- only the address it came from.
#[test]
fn the_word_face_stores_nothing_but_addresses() {
    // Miri interprets every one of these inserts, so the native count turns a fast test into a
    // ten-minute one. The property is about the face, not the size, and it is as visible at 200.
    const N: u32 = if cfg!(miri) { 200 } else { 10_000 };
    let mut s: WordSet<u32> = WordSet::new();
    for i in 0..N {
        assert!(s.insert(i));
    }
    assert!(!s.insert(N / 2));
    assert_eq!(s.len(), N as usize);
    assert_eq!(s.face(), apeiron::Sigma::Elliptic);
    assert!(s.contains(&(N - 1)) && !s.contains(&N));

    // The elements come back off the addresses alone.
    let mut all: Vec<u32> = s.iter().collect();
    all.sort_unstable();
    assert_eq!(all, (0..N).collect::<Vec<u32>>());

    for i in 0..N {
        assert!(s.remove(&i));
    }
    assert!(s.is_empty());
}

/// Iteration order is stable across runs, which is what a `BTreeSet` chosen for determinism was
/// actually paying for.
#[test]
fn iteration_order_is_reproducible() {
    let build = || (0..500u32).map(|i| i * 7).collect::<WordSet<u32>>().into_iter().collect::<Vec<_>>();
    assert_eq!(build(), build());
    assert_eq!(build().len(), 500);
}

#[test]
fn elements_are_dropped_exactly_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static MADE: AtomicUsize = AtomicUsize::new(0);
    static GONE: AtomicUsize = AtomicUsize::new(0);

    #[derive(PartialEq, Eq, Clone)]
    struct Counted(String);
    impl apeiron::coord::Coord for Counted {
        fn coord(&self) -> u64 {
            self.0.coord()
        }
        fn coord_with(&self, p: u64) -> u64 {
            self.0.coord_with(p)
        }
    }
    impl Counted {
        fn new(s: String) -> Self {
            MADE.fetch_add(1, Ordering::Relaxed);
            Self(s)
        }
    }
    impl Drop for Counted {
        fn drop(&mut self) {
            GONE.fetch_add(1, Ordering::Relaxed);
        }
    }

    {
        let mut s: Set<Counted> = Set::new();
        let n = 3000 / SCALE as u32;
        for i in 0..n {
            s.insert(Counted::new(format!("k{i}")));
        }
        for i in (0..n).step_by(3) {
            s.remove(&Counted::new(format!("k{i}")));
        }
        s.retain(|k| !k.0.ends_with('7'));
        let taken: Vec<_> = s.drain().collect();
        drop(taken);
        for i in 0..n / 3 {
            s.insert(Counted::new(format!("z{i}")));
        }
        s.shrink_to_fit();
    }
    assert_eq!(MADE.load(Ordering::Relaxed), GONE.load(Ordering::Relaxed));
}

/// `get_at` hands back the stored element; the batch readers agree with `contains`.
#[test]
fn the_borrowed_and_batch_readers_agree_with_contains() {
    let names: Vec<String> = (0..300).map(|i| format!("nw_wswmg{i:05}")).collect();
    let s: Set<String> = names.iter().take(200).cloned().collect();

    assert_eq!(s.get_at(names[7].as_str()), Some(&names[7]));
    assert_eq!(s.get_at("absent"), None);

    let mut seen = Vec::new();
    s.contains_each(&names, |k, present| seen.push((k.clone(), present)));
    assert_eq!(seen.len(), names.len());
    for (k, present) in &seen {
        assert_eq!(*present, s.contains(k), "contains_each disagreed on {k}");
    }
    assert_eq!(s.count_present(&names), 200);
}
