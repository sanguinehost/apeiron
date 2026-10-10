//! Logical-key identity for the vendored `Table`.
//!
//! Two identity faults are pinned here. First, the coordinates reserved as occupancy markers
//! (`EMPTY = u64::MAX`, `TOMB = u64::MAX - 1`) must not alias keys that *are* those coordinates:
//! `u64::MAX`, `u64::MAX - 1`, `u64::MAX - 2` and `0` are four distinct keys. Second, for a key type
//! whose `Coord::coord` is not injective, a coordinate match must be confirmed against the stored
//! key, so two distinct keys with equal coordinates are two entries.
//!
//! `apeiron` re-exports only the `Stored`-key `Table` publicly (`apeiron::freq::Table`), so both
//! cases exercise the full-key-confirmation path.

use apeiron::freq::{Coord, Table};
use std::collections::HashMap;

/// A key whose coordinate is deliberately non-injective: `2` and `3` both map to `1`. `Eq` compares
/// the full value, which is the identity the table must preserve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Collider(u64);

impl Coord for Collider {
    #[inline]
    fn coord(&self) -> u64 {
        self.0 >> 1
    }
}

/// `u64::MAX`, `u64::MAX - 1`, `u64::MAX - 2` and `0`, four distinct keys: four entries, each read
/// back, one removed leaves the other three, and the table survives a resize.
#[test]
fn reserved_coordinates_are_four_distinct_keys() {
    let mut t: Table<u64, i64> = Table::new();
    let keys = [u64::MAX, u64::MAX - 1, u64::MAX - 2, 0];
    for (i, &k) in keys.iter().enumerate() {
        t.put(k, i64::try_from(i).unwrap() + 1);
    }

    assert_eq!(t.len(), 4, "four distinct keys ⇒ four entries");
    for (i, &k) in keys.iter().enumerate() {
        assert_eq!(
            t.get(&k),
            Some(&(i64::try_from(i).unwrap() + 1)),
            "key {k:#x} lost"
        );
    }

    // insert_with on one key does not touch another.
    t.insert_with(u64::MAX - 1, 100, |new, old| new + old);
    assert_eq!(t.get(&(u64::MAX - 1)), Some(&102));
    assert_eq!(t.get(&u64::MAX), Some(&1));
    assert_eq!(t.get(&(u64::MAX - 2)), Some(&3));
    assert_eq!(t.get(&0), Some(&4));

    // remove of one leaves the other three.
    assert_eq!(t.remove(&(u64::MAX - 2)), Some(3));
    assert_eq!(t.len(), 3);
    assert_eq!(t.get(&(u64::MAX - 2)), None);
    assert_eq!(t.get(&u64::MAX), Some(&1));
    assert_eq!(t.get(&(u64::MAX - 1)), Some(&102));
    assert_eq!(t.get(&0), Some(&4));

    // Force a resize with enough other keys: the reserved-coordinate entries must survive it.
    for k in 1..64u64 {
        t.put(k, -1);
    }
    assert!(t.capacity() > 8, "expected a resize");

    assert_eq!(
        t.get(&u64::MAX),
        Some(&1),
        "reserved key lost across resize"
    );
    assert_eq!(t.get(&(u64::MAX - 1)), Some(&102));
    assert_eq!(t.get(&0), Some(&4));
    assert_eq!(t.get(&(u64::MAX - 2)), None);

    // iter yields exactly the live keys (the three survivors plus the resize fillers).
    let mut got: Vec<u64> = t.iter().map(|(k, _)| *k).collect();
    got.sort_unstable();
    let mut want: Vec<u64> = vec![u64::MAX, u64::MAX - 1, 0];
    want.extend(1..64u64);
    want.sort_unstable();
    assert_eq!(got, want);
    assert_eq!(t.len(), got.len());
}

/// Keys `2` and `3` share a coordinate but are distinct. Both insert, both read back, `insert_with`
/// on one leaves the other, `remove` of one keeps the other, and this holds across a resize.
#[test]
fn equal_coordinates_are_distinct_keys() {
    let mut t: Table<Collider, i64> = Table::new();
    t.put(Collider(2), 20);
    t.put(Collider(3), 30);

    assert_eq!(t.len(), 2);
    assert_eq!(t.get(&Collider(2)), Some(&20));
    assert_eq!(t.get(&Collider(3)), Some(&30));

    // insert_with on one does not touch the other.
    t.insert_with(Collider(2), 5, |new, old| new + old);
    assert_eq!(t.get(&Collider(2)), Some(&25));
    assert_eq!(t.get(&Collider(3)), Some(&30));

    // remove of one keeps the other.
    assert_eq!(t.remove(&Collider(2)), Some(25));
    assert_eq!(t.get(&Collider(2)), None);
    assert_eq!(t.get(&Collider(3)), Some(&30));

    // Two fresh colliding keys both survive a resize.
    let mut t2: Table<Collider, i64> = Table::with_capacity(4);
    for i in 0..64u64 {
        t2.put(Collider(i * 4 + 100), i64::try_from(i).unwrap());
    }
    assert!(t2.capacity() > 8, "expected a resize");
    t2.put(Collider(2), 2);
    t2.put(Collider(3), 3);
    assert_eq!(t2.get(&Collider(2)), Some(&2));
    assert_eq!(t2.get(&Collider(3)), Some(&3));
    assert_eq!(t2.len(), 66);

    // A resized table keeps the collision-resolved pair distinct.
    for i in 0..64u64 {
        t.put(Collider(i * 4 + 1000), i64::try_from(i).unwrap());
    }
    assert!(t.capacity() > 8);
    assert_eq!(
        t.get(&Collider(3)),
        Some(&30),
        "colliding key lost across resize"
    );
    assert_eq!(t.get(&Collider(2)), None);
}

/// The count face (`add` / `count_each` / `count`) with reserved-coordinate keys (`u64::MAX` =
/// `EMPTY`, `u64::MAX - 1` = `TOMB`, `u64::MAX - 2` = `TOMB - 1`) and a colliding-coordinate pair:
/// every key is counted, the counts are exact, and the distinct keys stay distinct.
#[test]
fn reserved_coordinate_counts_are_exact() {
    let mut t: Table<u64, i64> = Table::new();
    for &k in &[u64::MAX, u64::MAX - 1, u64::MAX - 2, 0] {
        t.add(k, 1);
    }
    t.add(u64::MAX, 4);
    assert_eq!(t.count(&u64::MAX), 5, "reserved EMPTY key dropped");
    assert_eq!(t.count(&(u64::MAX - 1)), 1);
    assert_eq!(t.count(&(u64::MAX - 2)), 1);
    assert_eq!(t.count(&0), 1);
    assert_eq!(t.len(), 4, "four distinct keys ⇒ four entries");

    // Colliding coordinates are distinct logical keys, each counted exactly.
    let mut c: Table<Collider, i64> = Table::new();
    c.add(Collider(2), 3);
    c.add(Collider(3), 4);
    c.add(Collider(2), 1);
    assert_eq!(c.count(&Collider(2)), 4);
    assert_eq!(c.count(&Collider(3)), 4);
    assert_eq!(c.len(), 2);

    // The plain (small) `count_each` path.
    let mut e: Table<u64, i64> = Table::new();
    let mut stream = vec![u64::MAX; 5];
    stream.extend(std::iter::repeat_n(u64::MAX - 1, 3));
    stream.extend(std::iter::repeat_n(u64::MAX - 2, 2));
    stream.push(0);
    e.count_each(&stream);
    assert_eq!(e.count(&u64::MAX), 5);
    assert_eq!(e.count(&(u64::MAX - 1)), 3);
    assert_eq!(e.count(&(u64::MAX - 2)), 2);
    assert_eq!(e.count(&0), 1);
    assert_eq!(e.len(), 4);

    // The pipelined `count_each` path (`n > P` and a footprint over `1 << 19`): precomputed
    // coordinates reach `add_with_coord` directly, so the reserved routing must hold there too.
    let mut p: Table<u64, i64> = Table::with_capacity(40_000);
    let mut stream2 = vec![u64::MAX; 25];
    stream2.extend(std::iter::repeat_n(u64::MAX - 1, 15));
    stream2.extend(std::iter::repeat_n(0, 10));
    p.count_each(&stream2);
    assert_eq!(p.count(&u64::MAX), 25);
    assert_eq!(p.count(&(u64::MAX - 1)), 15);
    assert_eq!(p.count(&0), 10);
    assert_eq!(p.len(), 3);
}

/// `fuse` of two tables that each hold reserved-coordinate keys sums both side stores; a `fuse`
/// that walked only the main `coords` would drop one side's counts.
#[test]
fn fuse_sums_reserved_coordinate_counts() {
    let mut a: Table<u64, i64> = Table::new();
    a.add(u64::MAX, 2);
    a.add(u64::MAX - 1, 3);
    a.add(7, 1);
    let mut b: Table<u64, i64> = Table::new();
    b.add(u64::MAX, 10);
    b.add(u64::MAX - 1, 4);
    b.add(0, 7);
    a.fuse(b, |new, old| new + old);
    assert_eq!(a.count(&u64::MAX), 12, "LEFT side store lost in fuse");
    assert_eq!(a.count(&(u64::MAX - 1)), 7, "RIGHT side store lost in fuse");
    assert_eq!(a.count(&7), 1);
    assert_eq!(a.count(&0), 7);
    assert_eq!(a.len(), 4);
}

/// Last-write on an existing **main slot** leaves the new value. `put` is the map face and its body
/// is `insert_merge_last_write`, whose hit path overwrites the value in place instead of blanking
/// and re-occupying the coordinate; this pins the value the probe leaves behind after a second write
/// to the same key. Keys `0..8` are smaller than the slot count, so `home(c) = c` addresses a main
/// slot directly (not a collision, the side store, or the overflow).
// The in-place last-write overwrite.
#[test]
fn put_overwrites_an_existing_main_slot() {
    let mut t: Table<u64, i64> = Table::with_capacity(16);
    assert!(t.capacity() >= 8, "keys 0..8 must address main slots");
    for k in 0..8u64 {
        t.put(k, 10 + i64::try_from(k).unwrap());
    }
    for k in 0..8u64 {
        t.put(k, 100 + i64::try_from(k).unwrap());
    }
    for k in 0..8u64 {
        assert_eq!(
            t.get(&k),
            Some(&(100 + i64::try_from(k).unwrap())),
            "put must overwrite main slot {k}"
        );
    }
    assert_eq!(t.len(), 8);
}

/// Two tables differing only in a reserved-coordinate entry compare unequal; equal ones compare
/// equal. `PartialEq` used to compare side entries only through `len()`.
#[test]
fn side_entries_participate_in_equality() {
    let mut base: Table<u64, i64> = Table::new();
    base.put(0, 1);
    base.put(u64::MAX, 5);

    let same = base.clone();
    assert_eq!(base, same, "clones with the same side entries are equal");

    let mut changed_value = base.clone();
    changed_value.put(u64::MAX, 6); // same reserved key, different value
    assert_ne!(
        base, changed_value,
        "different side value must compare unequal"
    );

    let mut other_key: Table<u64, i64> = Table::new();
    other_key.put(0, 1);
    other_key.put(u64::MAX - 1, 5); // a different reserved key, same value
    assert_ne!(
        base, other_key,
        "different reserved key must compare unequal"
    );

    let mut removed = base.clone();
    removed.remove(&u64::MAX);
    assert_ne!(base, removed, "removing a side entry must compare unequal");
}

/// Run one fixed operation sequence over every `face.rs` operation against a `HashMap` oracle, so
/// the observable result of each is compared to the oracle after the same op. `keys` is either a set
/// of reserved-coordinate keys or a set with colliding coordinates.
// One long fixed op sequence reads as one story; splitting it would scatter the oracle pairing.
#[allow(clippy::too_many_lines)]
fn run_face_sequence<K>(keys: &[K])
where
    K: Copy + Clone + Eq + std::hash::Hash + Ord + std::fmt::Debug + Coord + 'static,
{
    let mut t: Table<K, i64> = Table::new();
    let mut m: HashMap<K, i64> = HashMap::new();

    // entry().or_insert(0) += 1 — the one-probe RMW.
    for &k in keys {
        *t.entry(k).or_insert(0) += 1;
        *m.entry(k).or_insert(0) += 1;
    }
    // entry_at().or_insert(0) += 2 — the borrowed-key form.
    for &k in keys {
        *t.entry_at(&k).or_insert(0) += 2;
        *m.entry(k).or_insert(0) += 2;
    }
    // get_mut.
    for &k in keys {
        if let Some(v) = t.get_mut(&k) {
            *v += 10;
        }
        if let Some(v) = m.get_mut(&k) {
            *v += 10;
        }
        assert_eq!(t.get(&k), m.get(&k), "get_mut {k:?}");
    }
    // insert (replace, returns the old value).
    for &k in keys {
        let want = *m.get(&k).unwrap() + 1;
        assert_eq!(t.insert(k, want), m.insert(k, want), "insert {k:?}");
    }
    // insert_at (borrowed replace).
    for &k in keys {
        let want = *m.get(&k).unwrap() + 1;
        assert_eq!(t.insert_at(&k, want), m.insert(k, want), "insert_at {k:?}");
    }
    // insert_fetch_at (merge, returns the merged value).
    for &k in keys {
        let want = *m.get(&k).unwrap() + 5;
        assert_eq!(
            *t.insert_fetch_at(&k, 5, |new, old| new + old),
            want,
            "insert_fetch_at {k:?}"
        );
        m.insert(k, want);
    }
    // Reads: get / get_at / get_exact_at / get_mut_at / get_key_value / contains_key / contains_at.
    for &k in keys {
        assert_eq!(t.get(&k), m.get(&k), "get {k:?}");
        assert_eq!(t.get_at(&k), m.get(&k), "get_at {k:?}");
        assert_eq!(t.get_exact_at(&k), m.get(&k), "get_exact_at {k:?}");
        assert_eq!(
            t.get_mut_at(&k).copied(),
            m.get(&k).copied(),
            "get_mut_at {k:?}"
        );
        assert_eq!(
            t.get_key_value(&k).map(|(_, v)| *v),
            m.get(&k).copied(),
            "get_key_value {k:?}"
        );
        assert_eq!(t.contains_key(&k), m.contains_key(&k), "contains_key {k:?}");
        assert_eq!(t.contains_at(&k), m.contains_key(&k), "contains_at {k:?}");
    }
    // keys / values.
    let mut tk: Vec<K> = t.keys().copied().collect();
    let mut mk: Vec<K> = m.keys().copied().collect();
    tk.sort_unstable();
    mk.sort_unstable();
    assert_eq!(tk, mk, "keys()");
    let mut tv: Vec<i64> = t.values().copied().collect();
    let mut mv: Vec<i64> = m.values().copied().collect();
    tv.sort_unstable();
    mv.sort_unstable();
    assert_eq!(tv, mv, "values()");
    // iter_mut / values_mut.
    for (k, v) in t.iter_mut() {
        *v += 1;
        *m.get_mut(k).unwrap() += 1;
    }
    for v in t.values_mut() {
        *v += 1;
    }
    for v in m.values_mut() {
        *v += 1;
    }
    // retain — keep only the even values.
    t.retain(|_k, v| *v % 2 == 0);
    m.retain(|_k, v| *v % 2 == 0);
    let mut tk: Vec<K> = t.keys().copied().collect();
    let mut mk: Vec<K> = m.keys().copied().collect();
    tk.sort_unstable();
    mk.sort_unstable();
    assert_eq!(tk, mk, "retain");
    // reserve.
    t.reserve(64);
    assert_eq!(t.len(), m.len(), "reserve must not change the entries");
    // remove_at / remove.
    let k0 = keys[0];
    assert_eq!(t.remove_at(&k0), m.remove(&k0), "remove_at {k0:?}");
    let k1 = keys[keys.len() - 1];
    assert_eq!(t.remove(&k1), m.remove(&k1), "remove {k1:?}");
    // drain.
    let mut td: Vec<(K, i64)> = t.drain().collect();
    let mut md: Vec<(K, i64)> = m.drain().collect();
    td.sort_unstable();
    md.sort_unstable();
    assert_eq!(td, md, "drain");
    assert_eq!(t.len(), 0);
    // remove_entry + IntoIterator + clear on a fresh table.
    let mut t2: Table<K, i64> = Table::new();
    for (i, &k) in keys.iter().enumerate() {
        t2.put(k, i64::try_from(i).unwrap());
    }
    assert_eq!(t2.remove_entry(&k0), Some((k0, 0)), "remove_entry {k0:?}");
    let mut owned: Vec<(K, i64)> = t2.clone().into_iter().collect();
    owned.sort_unstable();
    let mut want: Vec<(K, i64)> = keys
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, &k)| (k, i64::try_from(i).unwrap()))
        .collect();
    want.sort_unstable();
    assert_eq!(owned, want, "IntoIterator");
    t2.clear();
    assert_eq!(t2.len(), 0, "clear");
    assert!(t2.is_empty());
    assert_eq!(t2.into_iter().count(), 0);
}

/// Every listed `face.rs` operation, over reserved-coordinate keys, matches the `HashMap` oracle.
#[test]
fn reserved_face_ops_match_hashmap() {
    run_face_sequence(&[u64::MAX, u64::MAX - 1, u64::MAX - 2, 0, 7]);
}

/// Every listed `face.rs` operation, over a colliding-coordinate key set, matches the oracle.
#[test]
fn colliding_face_ops_match_hashmap() {
    run_face_sequence(&[
        Collider(2),
        Collider(3),
        Collider(4),
        Collider(5),
        Collider(12),
    ]);
}
