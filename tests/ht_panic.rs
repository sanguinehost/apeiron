// Row: `apeiron-ht-sound-HuFJWskE6qumuLwAlSJ0xi` (APEIRON-HT-SOUND).
//! Panic safety and churn cost of the map table (`apeiron::freq::Table`).
//!
//! A merge that panics removes its key: `len` drops by one, a main slot reads `TOMB` (never `EMPTY`,
//! so other keys' probe chains survive), a side entry is swap-removed, an overflow entry is deleted,
//! and every value is dropped exactly once. A same-size rebuild runs only when `len + 1 ≤ cap/2`;
//! otherwise the table grows, so a steady remove+insert churn rebuilds O(1) times, not every round.
//!
//! The reproductions are the independent review's (F1–F5), turned into assertions. A counting
//! global allocator (per thread) observes leaks; a counting value type observes double drops.

use apeiron::freq::{Coord, Table};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

// ── Instruments ────────────────────────────────────────────────────────────────────────────────

/// Net bytes allocated by this thread. Each test runs on its own harness thread, so the count is
/// not disturbed by the tests running beside it.
struct Counting;

thread_local! {
    static NET: Cell<isize> = const { Cell::new(0) };
}

fn net_add(d: isize) {
    let _ = NET.try_with(|n| n.set(n.get() + d));
}

// SAFETY: every call is forwarded to `System` unchanged; only a thread-local counter is updated.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            net_add(l.size().cast_signed());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            net_add(l.size().cast_signed());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        net_add(-l.size().cast_signed());
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            net_add(new.cast_signed() - l.size().cast_signed());
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn net() -> isize {
    NET.with(Cell::get)
}

/// Silence the deliberate panics (`"merge"`, `"warm-up"`): the harness captures a printed panic
/// message into a per-thread buffer, which the allocation count would read as a leak. Every other
/// panic goes to the default hook unchanged.
fn quiet_expected_panics() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info.payload().downcast_ref::<&str>().copied();
            if !matches!(msg, Some("merge" | "warm-up")) {
                default(info);
            }
        }));
    });
}

/// Net bytes `f` leaves allocated on this thread. One panic is caught first so the panic runtime's
/// one-time per-thread state is not counted against `f`.
fn leaked_by(f: impl FnOnce()) -> isize {
    quiet_expected_panics();
    let _ = catch_unwind(|| panic!("warm-up"));
    let before = net();
    f();
    net() - before
}

/// A value that counts its constructions and drops against one test's own pair of counters.
struct D {
    dropped: &'static AtomicUsize,
}

impl D {
    fn new(made: &'static AtomicUsize, dropped: &'static AtomicUsize) -> D {
        made.fetch_add(1, Ordering::SeqCst);
        D { dropped }
    }
}

impl Drop for D {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

fn boom<V>(_new: V, _old: V) -> V {
    panic!("merge")
}

// ── F1: side store (reserved-coordinate keys) ──────────────────────────────────────────────────

/// A panicking merge on a side-store key removes the key; nothing is dropped twice.
#[test]
fn f1_side_merge_panic_removes_key_no_double_drop() {
    static MADE: AtomicUsize = AtomicUsize::new(0);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    let d = || D::new(&MADE, &DROPPED);
    // `u64::MAX` is `EMPTY`'s coordinate, `u64::MAX - 1` is `TOMB`'s: both live in the side store.
    for key in [u64::MAX, u64::MAX - 1] {
        let mut t: Table<u64, D> = Table::new();
        t.put(7, d());
        t.put(key, d());
        assert_eq!(t.len(), 2);
        let r = catch_unwind(AssertUnwindSafe(|| t.insert_with(key, d(), boom)));
        assert!(r.is_err());
        assert_eq!(
            DROPPED.load(Ordering::SeqCst),
            MADE.load(Ordering::SeqCst) - 1,
            "the unwind drops the old and new values once each; key 7's value stays"
        );
        assert!(t.get(&key).is_none(), "a panicking merge removes the key");
        assert!(t.get_mut(&key).is_none());
        assert!(!t.contains_key(&key));
        assert_eq!(t.len(), 1);
        assert_eq!(t.iter().count(), 1);
        assert!(t.remove(&key).is_none());
        // Re-insertion of the removed key works and is counted once.
        t.put(key, d());
        assert!(t.get(&key).is_some());
        assert_eq!(t.len(), 2);
        // The same panic through the borrowed-key face.
        let r = catch_unwind(AssertUnwindSafe(|| {
            t.insert_fetch_at(&key, d(), boom);
        }));
        assert!(r.is_err());
        assert!(t.get(&key).is_none());
        assert_eq!(t.len(), 1);
        drop(t);
        assert_eq!(
            DROPPED.load(Ordering::SeqCst),
            MADE.load(Ordering::SeqCst),
            "every value dropped exactly once"
        );
    }
}

/// The same with `String` values (a double drop is a double free) and keys, and no leak.
#[test]
fn f1_side_merge_panic_string_no_double_free_no_leak() {
    let leaked = leaked_by(|| {
        let mut t: Table<u64, String> = Table::new();
        t.put(u64::MAX, "old".to_owned());
        let r = catch_unwind(AssertUnwindSafe(|| {
            t.insert_with(u64::MAX, "new".to_owned(), boom);
        }));
        assert!(r.is_err());
        assert_eq!(t.get(&u64::MAX), None);
        assert_eq!(t.len(), 0);
        assert_eq!(t.remove(&u64::MAX), None);
        t.put(u64::MAX, "again".to_owned());
        assert_eq!(t.get(&u64::MAX).map(String::as_str), Some("again"));
        let mut s: Table<String, String> = Table::new();
        s.put("k".to_owned(), "v".to_owned());
        let r = catch_unwind(AssertUnwindSafe(|| {
            s.insert_with("k".to_owned(), "w".to_owned(), boom);
        }));
        assert!(r.is_err());
        assert_eq!(s.get(&"k".to_owned()), None);
        assert_eq!(s.len(), 0);
    });
    assert_eq!(leaked, 0, "bytes left allocated after the tables dropped");
}

// ── F2: main slot ──────────────────────────────────────────────────────────────────────────────

/// A panicking merge on a main-slot key leaves `TOMB`, so a key probed past it stays reachable.
#[test]
fn f2_main_merge_panic_keeps_chain() {
    let leaked = leaked_by(|| {
        let mut t: Table<u64, String> = Table::with_capacity(100); // 256 slots, bits 8
        assert_eq!(t.capacity(), 256);
        let k2 = (256u64..u64::MAX)
            .find(|c| (c.wrapping_mul(PHI) >> 56) == 5)
            .unwrap();
        t.put(5, "a".into()); // home 5 (identity)
        t.put(k2, "b".into()); // home 5 occupied: linear step to 6
        assert_eq!(t.get(&k2).map(String::as_str), Some("b"));
        let r = catch_unwind(AssertUnwindSafe(|| t.insert_with(5, "x".into(), boom)));
        assert!(r.is_err());
        assert_eq!(t.get(&5), None, "a panicking merge removes the key");
        assert_eq!(
            t.get(&k2).map(String::as_str),
            Some("b"),
            "k2 lost behind a hole"
        );
        assert_eq!(t.len(), 1);
        t.put(k2, "c".into());
        assert_eq!(
            t.iter().filter(|(k, _)| **k == k2).count(),
            1,
            "k2 duplicated"
        );
        assert_eq!(t.len(), 1);
        assert_eq!(t.get(&k2).map(String::as_str), Some("c"));
        t.put(5, "d".into());
        assert_eq!(t.len(), 2);
        // The borrowed-key face's main-slot merge.
        let r = catch_unwind(AssertUnwindSafe(|| {
            t.insert_fetch_at(&5, "y".into(), boom);
        }));
        assert!(r.is_err());
        assert_eq!(t.get(&5), None);
        assert_eq!(t.get(&k2).map(String::as_str), Some("c"));
        assert_eq!(t.len(), 1);
        t.put(5, "e".into());
        assert_eq!(t.iter().count(), 2);
    });
    assert_eq!(leaked, 0, "bytes left allocated after the table dropped");
}

// ── F3: overflow ───────────────────────────────────────────────────────────────────────────────

fn phi_inv() -> u64 {
    let mut x = 1u64;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(PHI.wrapping_mul(x)));
    }
    assert_eq!(PHI.wrapping_mul(x), 1);
    x
}

/// A `u64` key whose whole capped prefix lies in slots [0, 64) of a 1024-slot table.
fn blocked_key(seed: u64) -> u64 {
    // bits = 10: home = top 10 bits; radix windows jj = 1..5 are the next 10-bit fields.
    let w = |i: u64| (seed.wrapping_mul(0x9E37) >> (i * 6)) & 0x3F; // each field in [0, 64)
    let h = 8 + (seed % 40); // home in [8, 48), so home..home+7 < 64
    let p = (h << 54)
        | (w(1) << 44)
        | (w(2) << 34)
        | (w(3) << 24)
        | (w(4) << 14)
        | (w(5) << 4)
        | (seed & 0xF);
    p.wrapping_mul(phi_inv())
}

fn blocked_table() -> (Table<u64, String>, u64) {
    let mut t: Table<u64, String> = Table::with_capacity(700);
    assert_eq!(t.capacity(), 1024);
    for k in 0..64u64 {
        t.put(k, format!("{k}"));
    }
    let k = blocked_key(12345);
    assert!((1024..u64::MAX - 1).contains(&k));
    t.put(k, "v".into());
    assert_eq!(t.overflow_len(), 1, "crafted key must be in the overflow");
    (t, k)
}

/// A panicking merge on an overflow key deletes the entry; the key reads absent, is not counted,
/// and a later insert of it succeeds.
#[test]
fn f3_overflow_merge_panic_deletes_entry() {
    let leaked = leaked_by(|| {
        let (mut t, k) = blocked_table();
        let r = catch_unwind(AssertUnwindSafe(|| t.insert_with(k, "w".into(), boom)));
        assert!(r.is_err());
        assert_eq!(t.get(&k), None);
        assert_eq!(t.len(), 64);
        assert_eq!(t.overflow_len(), 0);
        assert_eq!(t.iter().count(), 64);
        let r2 = catch_unwind(AssertUnwindSafe(|| t.put(k, "z".into())));
        assert!(r2.is_ok(), "key poisoned: a later insert panics");
        assert_eq!(t.get(&k).map(String::as_str), Some("z"));
        assert_eq!(t.len(), 65);
        // The overflow upsert (the borrowed-key face's blocked path).
        let r3 = catch_unwind(AssertUnwindSafe(|| {
            t.insert_fetch_at(&k, "y".into(), boom);
        }));
        assert!(r3.is_err());
        assert_eq!(t.get(&k), None);
        assert_eq!(t.len(), 64);
        assert_eq!(t.overflow_len(), 0);
        t.insert_fetch_at(&k, "q".into(), |n, _o| n);
        assert_eq!(t.get(&k).map(String::as_str), Some("q"));
        assert_eq!(t.len(), 65);
    });
    assert_eq!(leaked, 0, "bytes left allocated after the table dropped");
}

// ── F4: tombstone churn ────────────────────────────────────────────────────────────────────────

/// 2^14 slots: `with_capacity(12287)` sizes exactly `16384`; the grow threshold is
/// `T = 3/4 · 16384 = 12288` keys, so `T − 1` is the steady size the review measured. The keys are
/// the review's `0..fill` (each its own home slot, so none reaches the overflow and `len` is exact).
const CHURN_CAP: usize = 12_287;
const ROUNDS: u32 = 2_000;

/// `rounds` remove+insert pairs at steady size `fill` (the insert through `insert`), returning the
/// time of rounds `1..rounds` and the number of rebuilds over all of them. A rebuild is counted when
/// the insert changes the capacity or lowers the tombstone count: only a rebuild reclaims
/// tombstones, and an insert never adds or reuses one. Round 0 is untimed: at `T − 1` it carries
/// the one permitted grow, which the rebuild count already bounds, so the timed rounds compare
/// steady-state churn cost.
fn churn<V>(fill: u64, rounds: u32, insert: impl Fn(&mut Table<u64, V>, u64)) -> (Duration, usize) {
    let mut t: Table<u64, V> = Table::with_capacity(CHURN_CAP);
    assert_eq!(t.capacity(), 1 << 14);
    for i in 0..fill {
        insert(&mut t, i);
    }
    let mut rebuilds = 0;
    let mut t0 = Instant::now();
    for r in 0..rounds {
        if r == 1 {
            t0 = Instant::now();
        }
        let k = u64::from(r);
        assert!(t.remove(&k).is_some());
        let (cap, tombs) = (t.capacity(), t.tombstones());
        insert(&mut t, k);
        if t.capacity() != cap || t.tombstones() < tombs {
            rebuilds += 1;
        }
    }
    let dt = t0.elapsed();
    assert_eq!(t.len() as u64, fill, "steady size");
    (dt, rebuilds)
}

fn put(t: &mut Table<u64, u64>, k: u64) {
    t.put(k, k);
}

/// At steady size `T − 1`, 2000 remove+put rounds rebuild O(1) times (one grow), and each
/// steady-state round costs within 20× of the same churn at half load.
#[test]
fn f4_churn_at_threshold_rebuilds_o1_times() {
    let thr = (1u64 << 14) * 3 / 4; // 12288
    let (mut slow, mut fast) = (Duration::ZERO, Duration::ZERO);
    for _ in 0..3 {
        let (s, rebuilds) = churn(thr - 1, ROUNDS, put);
        assert!(
            rebuilds <= 2,
            "{rebuilds} rebuilds in {ROUNDS} rounds at size T-1"
        );
        let (f, control) = churn(thr / 2, ROUNDS, put);
        assert_eq!(control, 0, "the half-load control never rebuilds");
        slow += s;
        fast += f;
    }
    println!(
        "T-1: {:?}/round; T/2: {:?}/round; ratio {:.2}",
        slow / (3 * (ROUNDS - 1)),
        fast / (3 * (ROUNDS - 1)),
        slow.as_secs_f64() / fast.as_secs_f64()
    );
    assert!(
        slow.as_secs_f64() < 20.0 * fast.as_secs_f64(),
        "per-round cost at T-1 is {:.0}x the half-load control",
        slow.as_secs_f64() / fast.as_secs_f64()
    );
}

/// The same rule on the two other insert faces: `insert` (the `entry` path) and `add` (the count
/// face's in-place path).
#[test]
fn f4_churn_rule_holds_on_entry_and_count_faces() {
    let thr = (1u64 << 14) * 3 / 4;
    let (_, rebuilds) = churn(thr - 1, ROUNDS, |t: &mut Table<u64, u64>, k| {
        assert!(t.insert(k, k).is_none());
    });
    assert!(rebuilds <= 2, "entry face: {rebuilds} rebuilds");
    let (_, rebuilds) = churn(thr - 1, ROUNDS, |t: &mut Table<u64, i64>, k| t.add(k, 1));
    assert!(rebuilds <= 2, "count face: {rebuilds} rebuilds");
}

// ── F5: consuming iteration frees every buffer ─────────────────────────────────────────────────

/// `for (k, v) in table` frees the side store's buffers and its key box: nothing leaks, with side,
/// main and overflow entries present, whether the iterator is run out or dropped part-way.
#[test]
fn f5_into_iter_frees_side_store() {
    for take in [usize::MAX, 1] {
        let leaked = leaked_by(|| {
            let mut keyed: Table<String, String> = Table::new();
            keyed.put("main".to_owned(), "m".to_owned());
            let mut side: Table<u64, String> = Table::new();
            side.put(u64::MAX, "side".to_owned());
            side.put(u64::MAX - 1, "side2".to_owned());
            side.put(3, "main".to_owned());
            assert!(side.into_iter().take(take).count() >= 1);
            assert!(keyed.into_iter().take(take).count() >= 1);
            let (blocked, _) = blocked_table();
            assert!(blocked.into_iter().take(take).count() >= 1);
        });
        assert_eq!(
            leaked, 0,
            "bytes left allocated after into_iter (take {take})"
        );
    }
}

// ── G1–G3: unwind ordering (independent review round 2) ─────────────────────────────────────────

/// A value that counts its drops into its own counter and panics on the second drop of its test's
/// shared `total`. The panic cuts an unwinding `SideStore::clear` short mid-loop; a later `clear`
/// that re-drops entry 0 is then visible as `entry 0 dropped twice`.
struct Pv {
    drops: &'static AtomicUsize,
    total: &'static AtomicUsize,
}

impl Drop for Pv {
    fn drop(&mut self) {
        let n = self.total.fetch_add(1, Ordering::SeqCst) + 1;
        self.drops.fetch_add(1, Ordering::SeqCst);
        assert!(n != 2, "side-drop");
    }
}

/// T1: `clear` on a table with two side-store entries whose `V::drop` panics on the second drop. The
/// entry tables are detached before any destructor runs, so the later `Drop for Table` reaches
/// neither entry: each value is dropped at most once.
#[test]
fn g1_side_clear_panic_no_double_drop() {
    static A: AtomicUsize = AtomicUsize::new(0);
    static B: AtomicUsize = AtomicUsize::new(0);
    static TOTAL: AtomicUsize = AtomicUsize::new(0);
    // `u64::MAX` is `EMPTY`'s coordinate, `u64::MAX - 1` is `TOMB`'s: both live in the side store.
    let mut t: Table<u64, Pv> = Table::new();
    t.put(
        u64::MAX,
        Pv {
            drops: &A,
            total: &TOTAL,
        },
    );
    t.put(
        u64::MAX - 1,
        Pv {
            drops: &B,
            total: &TOTAL,
        },
    );
    assert_eq!(t.len(), 2);
    let r = catch_unwind(AssertUnwindSafe(|| t.clear()));
    assert!(r.is_err(), "the second value drop must panic");
    drop(t);
    assert!(A.load(Ordering::SeqCst) <= 1, "entry 0 dropped twice");
    assert!(B.load(Ordering::SeqCst) <= 1, "entry 1 dropped twice");
}

/// A key whose `Drop` panics when its own `panic` flag is set, so the main-slot removal faces can be
/// interrupted between moving the value out and tombstoning the slot. The flag rides the key, so the
/// reproduction is independent of other tests running beside it.
struct Pk {
    c: u64,
    panic: bool,
}

impl PartialEq for Pk {
    fn eq(&self, other: &Self) -> bool {
        self.c == other.c
    }
}
impl Eq for Pk {}
impl Coord for Pk {
    const INJECTIVE: bool = true; // `coord` is the field itself
    #[inline]
    fn coord(&self) -> u64 {
        self.c
    }
}

impl Drop for Pk {
    fn drop(&mut self) {
        assert!(!self.panic, "key-drop");
    }
}

/// T2 body: one main-slot key removed through the given face with a panicking `K::drop`. The fix
/// commits `TOMB`/`len`/`tombs` first, so the caught panic leaves the slot absent, `len`
/// decremented, and the value dropped exactly once.
fn g2_remove_face(
    made: &'static AtomicUsize,
    dropped: &'static AtomicUsize,
    remove: impl Fn(&mut Table<Pk, D>, &Pk) -> Option<D>,
) {
    let mut t: Table<Pk, D> = Table::new();
    let probe = Pk { c: 1, panic: false };
    t.put(Pk { c: 1, panic: true }, D::new(made, dropped));
    assert_eq!(t.len(), 1);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = remove(&mut t, &probe);
    }));
    assert!(r.is_err(), "the key drop must panic");
    assert!(t.get(&probe).is_none(), "a removed slot reads as absent");
    assert_eq!(t.len(), 0, "the removal is counted before the key drop");
    drop(t);
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        made.load(Ordering::SeqCst),
        "the value is dropped exactly once"
    );
}

/// The two removal faces as named callbacks: a fn item carries the exact monomorphic type (a
/// closure over the generic `remove_at::<&Pk>` is not higher-ranked enough for `Fn`).
fn remove_face(t: &mut Table<Pk, D>, k: &Pk) -> Option<D> {
    t.remove(k)
}

fn remove_at_face(t: &mut Table<Pk, D>, k: &Pk) -> Option<D> {
    t.remove_at(k)
}

#[test]
fn g2_remove_panic_commits_slot_first() {
    static MADE: AtomicUsize = AtomicUsize::new(0);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    g2_remove_face(&MADE, &DROPPED, remove_face);
}

#[test]
fn g2_remove_at_panic_commits_slot_first() {
    static MADE: AtomicUsize = AtomicUsize::new(0);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    g2_remove_face(&MADE, &DROPPED, remove_at_face);
}

/// A key whose `Clone` panics when its own flag is set: the `Absorb` impl for `&K` calls it on the
/// vacant path, so it panics exactly where `insert_fetch_at` materializes the owned key.
struct Ck {
    c: u64,
    clone_panics: bool,
}

impl PartialEq for Ck {
    fn eq(&self, other: &Self) -> bool {
        self.c == other.c
    }
}
impl Eq for Ck {}
impl Coord for Ck {
    const INJECTIVE: bool = true;
    #[inline]
    fn coord(&self) -> u64 {
        self.c
    }
}

impl Clone for Ck {
    fn clone(&self) -> Self {
        assert!(!self.clone_panics, "absorb");
        Ck {
            c: self.c,
            clone_panics: self.clone_panics,
        }
    }
}

/// T3: a panicking `Absorb` on the vacant path. The key is materialized before any slot is marked
/// live, so the caught panic leaves the slot `EMPTY` — `len` unchanged, `get` absent, and the value
/// dropped exactly once.
#[test]
fn g3_insert_fetch_vacant_absorb_panic_leaves_slot_empty() {
    static MADE: AtomicUsize = AtomicUsize::new(0);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    let mut t: Table<Ck, D> = Table::new();
    let key = Ck {
        c: 1,
        clone_panics: true,
    };
    let r = catch_unwind(AssertUnwindSafe(|| {
        t.insert_fetch_at(&key, D::new(&MADE, &DROPPED), |n, _o| n);
    }));
    assert!(r.is_err(), "absorb must panic");
    assert_eq!(t.len(), 0, "no slot is committed before the key exists");
    assert!(t.get(&key).is_none(), "the vacant slot was left live");
    drop(t);
    assert_eq!(
        DROPPED.load(Ordering::SeqCst),
        MADE.load(Ordering::SeqCst),
        "the value is dropped exactly once"
    );
}

// ── G4–G8: the five remaining same-class sites (independent review round 3) ─────────────────────

/// G4 (site 1, `OccupiedEntry::remove_entry`, every `Store` arm): `remove_entry` moves the key and
/// value out and RETURNS them; it runs no user destructor of its own, so there is no unwind to
/// reorder against — this pins the invariant. A stored key whose `Drop` panics, dropped by the
/// caller after the call returns, leaves the slot absent, `len` decremented and the value dropped
/// exactly once. The main store and the reserved-coordinate side store are both exercised.
#[test]
fn g4_remove_entry_returns_before_any_key_drop() {
    static MADE: AtomicUsize = AtomicUsize::new(0);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    // `7` is a main slot; `u64::MAX` / `u64::MAX - 1` are the reserved (`EMPTY`/`TOMB`)
    // coordinates held in the side store.
    for coord in [7u64, u64::MAX, u64::MAX - 1] {
        let mut t: Table<Pk, D> = Table::new();
        t.put(
            Pk {
                c: coord,
                panic: true,
            },
            D::new(&MADE, &DROPPED),
        );
        assert_eq!(t.len(), 1, "one entry at {coord:#x}");
        let probe = Pk {
            c: coord,
            panic: false,
        };
        let (k, v) = t.remove_entry(&probe).expect("present");
        // The structural commit happened before `remove_entry` returned.
        assert!(t.get(&probe).is_none(), "the removed slot reads as absent");
        assert_eq!(t.len(), 0, "the removal is counted");
        // The key's destructor runs here, after the commit; its panic cannot corrupt the table.
        let r = catch_unwind(AssertUnwindSafe(move || drop(k)));
        assert!(r.is_err(), "the returned key's drop panics");
        drop(v);
        drop(t);
    }
    assert_eq!(
        DROPPED.load(Ordering::SeqCst),
        MADE.load(Ordering::SeqCst),
        "every value is dropped exactly once"
    );
}

/// G5 (site 2, `Table::clear`): a `V::drop` that panics mid-pass must leave a table that is empty
/// or consistent, the values not yet dropped leaked rather than double-dropped on the next
/// `clear`/`Drop`. The clearing loop commits each slot `EMPTY` (and its `len`/`tombs` share)
/// before running that slot's destructor.
#[test]
fn g5_clear_panic_no_double_drop() {
    static A: AtomicUsize = AtomicUsize::new(0);
    static B: AtomicUsize = AtomicUsize::new(0);
    static TOTAL: AtomicUsize = AtomicUsize::new(0);
    let mut t: Table<u64, Pv> = Table::new();
    t.put(
        1,
        Pv {
            drops: &A,
            total: &TOTAL,
        },
    );
    t.put(
        2,
        Pv {
            drops: &B,
            total: &TOTAL,
        },
    );
    assert_eq!(t.len(), 2);
    let r = catch_unwind(AssertUnwindSafe(|| t.clear()));
    assert!(r.is_err(), "the second value drop must panic");
    drop(t);
    assert!(A.load(Ordering::SeqCst) <= 1, "entry 0 dropped twice");
    assert!(B.load(Ordering::SeqCst) <= 1, "entry 1 dropped twice");
}

/// G6 (site 3, `Table::retain`): a panicking drop of a removed value must not leave a live slot
/// whose value was moved out. The main-slot removal commits `TOMB`/`len`/`tombs` before the value
/// and the key are dropped.
#[test]
fn g6_retain_removal_panic_commits_slot_first() {
    static A: AtomicUsize = AtomicUsize::new(0);
    static B: AtomicUsize = AtomicUsize::new(0);
    static TOTAL: AtomicUsize = AtomicUsize::new(0);
    let mut t: Table<u64, Pv> = Table::new();
    t.put(
        1,
        Pv {
            drops: &A,
            total: &TOTAL,
        },
    );
    t.put(
        2,
        Pv {
            drops: &B,
            total: &TOTAL,
        },
    );
    let r = catch_unwind(AssertUnwindSafe(|| t.retain(|_k, _v| false)));
    assert!(r.is_err(), "the second removed value's drop must panic");
    drop(t);
    assert!(A.load(Ordering::SeqCst) <= 1, "entry 0 dropped twice");
    assert!(B.load(Ordering::SeqCst) <= 1, "entry 1 dropped twice");
}

/// G7 (site 4, `IntoIter::next`): `next` moves the value and key out and blanks the lane before
/// returning, so a consumer panic between calls — or a panicking drop of a yielded value — does not
/// re-yield the slot or drop it again on `IntoIter`'s own `Drop`.
#[test]
fn g7_into_iter_drop_does_not_double_drop_a_yielded_item() {
    static A: AtomicUsize = AtomicUsize::new(0);
    static B: AtomicUsize = AtomicUsize::new(0);
    static TOTAL: AtomicUsize = AtomicUsize::new(0);
    let mut t: Table<u64, Pv> = Table::new();
    t.put(
        1,
        Pv {
            drops: &A,
            total: &TOTAL,
        },
    );
    t.put(
        2,
        Pv {
            drops: &B,
            total: &TOTAL,
        },
    );
    let mut it = t.into_iter();
    let (_k, v) = it.next().expect("one live lane");
    drop(v); // total 1: the yielded value is consumed by the consumer
             // The remaining live lane's drop panics; the iterator must not have re-dropped the yielded one.
    let r = catch_unwind(AssertUnwindSafe(move || drop(it)));
    assert!(r.is_err(), "the remaining value's drop must panic");
    assert!(
        A.load(Ordering::SeqCst) <= 1,
        "the yielded value was dropped twice"
    );
    assert!(
        B.load(Ordering::SeqCst) <= 1,
        "the remaining value was dropped twice"
    );
}

/// G8 (site 5, `place_capped`): a rebuild re-places every carried entry through the capped
/// placement. `probe_capped_keyed` confirms a coordinate match against the stored key, so with an
/// injective coordinate two distinct keys cannot share one and a rebuild never takes the defensive
/// `Hit` arm (the one that drops the old value before writing the new); no reorder is needed there.
/// The rebuild must still move every value exactly once — no double drop, no leak.
#[test]
fn g8_place_capped_rebuild_moves_each_value_once() {
    let leaked = leaked_by(|| {
        let mut t: Table<u64, String> = Table::with_capacity(4);
        let cap0 = t.capacity();
        for k in 0..2048u64 {
            t.put(k, format!("v{k}"));
        }
        assert!(t.capacity() > cap0, "expected at least one rebuild");
        assert_eq!(t.len(), 2048);
        for k in [0u64, 1, 7, 100, 1000, 2047] {
            let want = format!("v{k}");
            assert_eq!(
                t.get(&k).map(String::as_str),
                Some(want.as_str()),
                "key {k} lost across a rebuild"
            );
        }
        drop(t);
    });
    assert_eq!(leaked, 0, "every value freed exactly once across rebuilds");
}
