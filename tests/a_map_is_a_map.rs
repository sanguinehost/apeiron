//! The map's specification, written as `std::collections::HashMap`.
//!
//! Not a list of things that happen to work. A random program of inserts, merges, removals,
//! retains and clears is run against both, and after **every** operation the two must agree on the
//! return value, on the length, and on the contents. A map that is right on the operations someone
//! thought to test and wrong on the fourth removal after a growth is the failure this is written
//! to catch, and the shrinking loop below reports the shortest program that separates them.
//!
//! Removal is the reason this exists. A tombstone that a probe treats as empty loses every key
//! placed past it while it was live -- and the table still answers, still has the right length, and
//! is wrong only for the keys that happened to sit beyond it. No hand-written test finds that.

use apeiron::map::{Map, Words};
use std::collections::HashMap;

/// Miri interprets rather than runs, so the same programs at the same sizes take hours. The shapes
/// are what find bugs -- growth, tombstone reuse, drop -- and those all happen well before a
/// thousand seeds. Native keeps the full sweep; Miri gets enough of it to exercise every path.
#[cfg(miri)]
const SCALE: usize = 40;
#[cfg(not(miri))]
const SCALE: usize = 1;

/// A deterministic generator. `SplitMix64`, so a failing seed is a reproducible program.
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

#[derive(Debug, Clone)]
enum Op {
    Insert(u32, u32),
    Merge(u32, u32),
    Get(u32),
    Remove(u32),
    Retain(u32),
    Clear,
    Reserve(u32),
}

fn program(seed: u64, len: usize, spread: u64) -> Vec<Op> {
    let mut r = Rng(seed);
    (0..len)
        .map(|_| {
            let k = r.below(spread) as u32;
            let v = r.next() as u32;
            // Weighted so the table actually fills and churns: removal has to be common enough to
            // lay tombstones across probe order, and clears rare enough not to hide anything.
            match r.below(100) {
                0..=39 => Op::Insert(k, v),
                40..=54 => Op::Merge(k, v),
                55..=74 => Op::Get(k),
                75..=91 => Op::Remove(k),
                92..=95 => Op::Retain(r.below(8) as u32),
                96..=97 => Op::Reserve(r.below(64) as u32),
                _ => Op::Clear,
            }
        })
        .collect()
}

/// Run a program against both and return the first operation index at which they disagree.
fn diverges(ops: &[Op]) -> Option<usize> {
    let mut ours: Map<u32, u32> = Map::new();
    let mut theirs: HashMap<u32, u32> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        let agree = match *op {
            Op::Insert(k, v) => ours.insert(k, v) == theirs.insert(k, v),
            Op::Merge(k, v) => {
                ours.insert_with(k, v, |slot, new| *slot = new.wrapping_add(*slot));
                let e = theirs.entry(k).or_insert(0);
                *e = v.wrapping_add(*e);
                true
            }
            Op::Get(k) => ours.get(&k) == theirs.get(&k),
            Op::Remove(k) => ours.remove(&k) == theirs.remove(&k),
            Op::Retain(m) => {
                let m = m.max(1);
                ours.retain(|k, _| k % m != 0);
                theirs.retain(|k, _| k % m != 0);
                true
            }
            Op::Reserve(n) => {
                ours.reserve(n as usize);
                theirs.reserve(n as usize);
                true
            }
            Op::Clear => {
                ours.clear();
                theirs.clear();
                true
            }
        };
        let same_len = ours.len() == theirs.len();
        // The whole contents, every step -- not just the key the operation touched. A probe that
        // has lost a key answers correctly about every other one.
        let same_body = theirs.iter().all(|(k, v)| ours.get(k) == Some(v))
            && ours.iter().all(|(k, v)| theirs.get(k) == Some(v));
        if !(agree && same_len && same_body) {
            return Some(i);
        }
    }
    None
}

/// The shortest prefix that still diverges, so a failure reports a program a person can read.
fn shrink(ops: &[Op]) -> Vec<Op> {
    let mut best = ops.to_vec();
    let mut i = 0;
    while i < best.len() {
        let mut without = best.clone();
        without.remove(i);
        if diverges(&without).is_some() {
            best = without;
        } else {
            i += 1;
        }
    }
    best
}

#[test]
fn a_thousand_random_programs_agree_with_std() {
    for seed in 0..(1000 / SCALE as u64).max(4) {
        // A narrow key space so collisions, reuse and tombstone reuse all actually happen.
        let ops = program(seed, 400 / SCALE.max(1), 64);
        if let Some(at) = diverges(&ops) {
            panic!("seed {seed} diverged at op {at}\nshortest: {:?}", shrink(&ops));
        }
    }
}

/// The same, wide enough that the table grows several times.
#[test]
fn programs_that_force_growth_agree_with_std() {
    for seed in 5000..(5000 + (40 / SCALE).max(2) as u64) {
        let ops = program(seed, 4000 / SCALE, 2000);
        if let Some(at) = diverges(&ops) {
            panic!("seed {seed} diverged at op {at}\nshortest: {:?}", shrink(&ops));
        }
    }
}

/// The elliptic face runs the same program, so the two faces are held to one specification.
#[test]
fn the_elliptic_face_agrees_with_std_too() {
    for seed in 900..(900 + (60 / SCALE).max(2) as u64) {
        let ops = program(seed, 600 / SCALE, 128);
        let mut ours: Words<u64, u32> = Words::new();
        let mut theirs: HashMap<u64, u32> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            match *op {
                Op::Insert(k, v) => {
                    assert_eq!(ours.insert(u64::from(k), v), theirs.insert(u64::from(k), v), "seed {seed} op {i}");
                }
                Op::Merge(k, v) => {
                    ours.insert_with(u64::from(k), v, |slot, n| *slot = n.wrapping_add(*slot));
                    let e = theirs.entry(u64::from(k)).or_insert(0);
                    *e = v.wrapping_add(*e);
                }
                Op::Get(k) => assert_eq!(ours.get(&u64::from(k)), theirs.get(&u64::from(k)), "seed {seed} op {i}"),
                Op::Remove(k) => {
                    assert_eq!(ours.remove(&u64::from(k)), theirs.remove(&u64::from(k)), "seed {seed} op {i}");
                }
                Op::Retain(m) => {
                    let m = u64::from(m.max(1));
                    ours.retain(|k, _| k % m != 0);
                    theirs.retain(|k, _| k % m != 0);
                }
                Op::Reserve(n) => {
                    ours.reserve(n as usize);
                    theirs.reserve(n as usize);
                }
                Op::Clear => {
                    ours.clear();
                    theirs.clear();
                }
            }
            assert_eq!(ours.len(), theirs.len(), "seed {seed} op {i}");
            assert!(theirs.iter().all(|(k, v)| ours.get(k) == Some(v)), "seed {seed} op {i}");
        }
    }
}

/// String keys through the borrowed surface, which is the path a real port takes.
#[test]
fn string_keys_agree_with_std_through_the_borrowed_surface() {
    let mut r = Rng(77);
    let mut ours: Map<String, u32> = Map::new();
    let mut theirs: HashMap<String, u32> = HashMap::new();
    for i in 0..(20_000 / SCALE as u32) {
        let k = format!("res{}_{}", r.below(3000), "x".repeat((i % 30) as usize));
        match r.below(4) {
            0 => {
                assert_eq!(ours.remove_at(k.as_str()), theirs.remove(k.as_str()));
            }
            1 => {
                assert_eq!(ours.get_at(k.as_str()), theirs.get(k.as_str()));
            }
            2 => {
                *ours.entry_at(k.as_str()).or_insert(0) += 1;
                *theirs.entry(k.clone()).or_insert(0) += 1;
            }
            _ => {
                assert_eq!(ours.insert_at(k.as_str(), i), theirs.insert(k.clone(), i));
            }
        }
        assert_eq!(ours.len(), theirs.len(), "at {i} on {k:?}");
    }
    assert!(theirs.iter().all(|(k, v)| ours.get_at(k.as_str()) == Some(v)));
    assert!(ours.iter().all(|(k, v)| theirs.get(k) == Some(v)));
}

/// Tombstones must be reused, or a table that is filled and drained forever grows forever. This is
/// the property that makes removal correct *and* bounded; correctness alone is satisfied by never
/// reusing one.
#[test]
fn filling_and_draining_forever_does_not_grow_the_table() {
    let mut m: Map<u64, u64> = Map::new();
    for i in 0..200u64 {
        m.insert(i, i);
    }
    assert_eq!(m.len(), 200);
    let settled = m.slots();
    // Fresh keys every round, so every insert is a genuine placement rather than an overwrite --
    // otherwise the tombstones are never in the way of anything and the test proves nothing.
    for round in 1..(500 / SCALE as u64).max(3) {
        for i in 0..200u64 {
            assert!(m.remove(&(i + (round - 1) * 1000)).is_some());
        }
        assert_eq!(m.len(), 0);
        for i in 0..200u64 {
            m.insert(i + round * 1000, i);
        }
        assert_eq!(m.len(), 200);
    }
    assert_eq!(m.slots(), settled, "the table grew on churn alone");
}

/// Values must be dropped exactly once -- not twice on removal, and not leaked on growth.
#[test]
fn every_value_is_dropped_exactly_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static MADE: AtomicUsize = AtomicUsize::new(0);
    static GONE: AtomicUsize = AtomicUsize::new(0);

    struct Counted(#[allow(dead_code)] String);
    impl Counted {
        fn new(s: &str) -> Self {
            MADE.fetch_add(1, Ordering::Relaxed);
            Self(s.to_owned())
        }
    }
    impl Drop for Counted {
        fn drop(&mut self) {
            GONE.fetch_add(1, Ordering::Relaxed);
        }
    }

    {
        let mut m: Map<String, Counted> = Map::new();
        for i in 0..(5000 / SCALE as u32) {
            m.insert(format!("k{i}"), Counted::new("v"));
        }
        for i in (0..(5000 / SCALE as u32)).step_by(3) {
            drop(m.remove(&format!("k{i}")));
        }
        for i in 0..(2000 / SCALE as u32) {
            m.insert(format!("k{i}"), Counted::new("w"));
        }
        m.retain(|k, _| !k.ends_with('7'));
        let taken = m.drain();
        assert!(!taken.is_empty());
        drop(taken);
        for i in 0..(1000 / SCALE as u32) {
            m.insert(format!("z{i}"), Counted::new("z"));
        }
    }
    assert_eq!(MADE.load(Ordering::Relaxed), GONE.load(Ordering::Relaxed));
    assert!(MADE.load(Ordering::Relaxed) > 5000 / SCALE);
}

/// The same keys give the same iteration order, on every run. Determinism is the property the
/// engine this is for actually needs, and it is the one a randomly-seeded hasher denies.
#[test]
fn iteration_order_is_the_same_every_time() {
    let build = || {
        let mut m: Map<String, u32> = Map::new();
        for i in 0..(500 / SCALE as u32) {
            m.insert(format!("entry-{i}"), i);
        }
        m.remove_at("entry-7");
        m.insert("entry-7000".into(), 7);
        m.keys().cloned().collect::<Vec<_>>()
    };
    let a = build();
    let b = build();
    assert_eq!(a, b);
    assert_eq!(a.len(), 500 / SCALE);
}

/// Equality is over the entries, so churn history must not show through it.
#[test]
fn a_churned_map_equals_a_fresh_one_with_the_same_entries() {
    let mut churned: Map<String, u32> = Map::new();
    for i in 0..300u32 {
        churned.insert(format!("k{i}"), i);
    }
    for i in 100..300u32 {
        churned.remove(&format!("k{i}"));
    }
    let fresh: Map<String, u32> = (0..100u32).map(|i| (format!("k{i}"), i)).collect();
    assert!(churned.tombstones() > 0);
    assert_eq!(churned, fresh);
    assert_eq!(churned.clone(), fresh);
}

/// A merge closure that panics must leave the map exactly as it was.
///
/// This is the whole point of `insert_with` taking `&mut V` rather than `V`. The earlier form had
/// to move the old value out of the slot to call the closure, so an unwind left the slot
/// initialised-looking and holding nothing; the entry was tombstoned and lost. Now the value never
/// leaves the slot, so there is no window -- and this asserts that from the outside: the entry is
/// still there, still readable, still counted, and the map still answers about every other key.
#[test]
fn a_merge_that_panics_leaves_the_map_intact() {
    let mut m: Map<String, u64> = Map::new();
    for i in 0..500u64 {
        m.insert(format!("k{i}"), i);
    }
    let before = m.len();

    for i in (0..500u64).step_by(7) {
        let key = format!("k{i}");
        let boom = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m.insert_with(key.clone(), 1, |slot, _new| {
                *slot += 1;
                panic!("merge failed halfway");
            });
        }));
        assert!(boom.is_err(), "the closure was supposed to panic");

        // The entry survives, and it survives with whatever the closure managed to write.
        assert_eq!(m.get(&key), Some(&(i + 1)), "entry lost on {key}");
        assert_eq!(m.len(), before, "length moved");
        assert_eq!(m.tombstones(), 0, "a failed merge left a tombstone");
    }

    // And nothing else was disturbed.
    for i in 0..500u64 {
        let want = if i % 7 == 0 { i + 1 } else { i };
        assert_eq!(m.get(&format!("k{i}")), Some(&want));
    }
    assert_eq!(m.len(), 500);
}

/// A value whose `Drop` runs must still run exactly once when a merge panics.
#[test]
fn a_panicking_merge_does_not_leak_or_double_drop() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static MADE: AtomicUsize = AtomicUsize::new(0);
    static GONE: AtomicUsize = AtomicUsize::new(0);

    struct Counted(#[allow(dead_code)] String);
    impl Counted {
        fn new(s: &str) -> Self {
            MADE.fetch_add(1, Ordering::Relaxed);
            Self(s.to_owned())
        }
    }
    impl Drop for Counted {
        fn drop(&mut self) {
            GONE.fetch_add(1, Ordering::Relaxed);
        }
    }

    {
        let mut m: Map<u32, Counted> = Map::new();
        for i in 0..200u32 {
            m.insert(i, Counted::new("v"));
        }
        for i in 0..200u32 {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // The closure takes ownership of `new` and panics; `new` must be dropped by the
                // unwind, and the slot's own value must not be.
                m.insert_with(i, Counted::new("w"), |_slot, _new| panic!("no"));
            }));
        }
        assert_eq!(m.len(), 200);
    }
    assert_eq!(
        MADE.load(Ordering::Relaxed),
        GONE.load(Ordering::Relaxed),
        "made {} but dropped {}",
        MADE.load(Ordering::Relaxed),
        GONE.load(Ordering::Relaxed)
    );
}
