//! Red team for the capped primary probe and its deterministic overflow — the HashDOS answer.
//!
//! `Table` examines at most [`PROBE_CAP`] primary slots per lookup, insert, update or remove; the
//! terminal sweep is removed. A key whose capped prefix was fully occupied is stored in a
//! table-wide AVL overflow tree, and every capped miss consults it. These tests are the executable
//! bounds: the crafted block that used to push a victim into the sweep now stays capped; a flood at
//! one home keeps the primary probe bounded and the overflow balanced; a dense miss no longer walks
//! the occupied block. The HashDOS claim covers keys whose coordinate is injective (`u64` here).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::many_single_char_names
)]

use apeiron::freq::{Table, PROBE_CAP};

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;
const LINEAR_G: u32 = 8; // src/hashtrinity/mod.rs

/// `PHI⁻¹ mod 2⁶⁴` (Newton iteration; `PHI` is odd ⇒ a unit). Lets a test name the product
/// `k·PHI` exactly, so it can aim a key at a chosen home slot.
fn phi_inv() -> u64 {
    let mut x = 1u64;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(PHI.wrapping_mul(x)));
    }
    assert_eq!(PHI.wrapping_mul(x), 1, "PHI⁻¹");
    x
}

/// The capped probe's address for step `j`, replicated from `Table::descend` for a `bits`-wide table
/// with the victim's home slot `home`. Steps `j >= LINEAR_G` are the fresh radix windows; all callers
/// here stay within the window set (the removed sweep is never named).
fn addr(bits: u32, home: usize, p: u64, j: u32) -> usize {
    let mask = (1usize << bits) - 1;
    if j < LINEAR_G {
        home.wrapping_add(j as usize) & mask
    } else {
        let used = bits * (j - LINEAR_G + 2);
        assert!(used <= 64, "step {j} is the removed terminal sweep");
        (p >> (64 - used)) as usize & mask
    }
}

/// The overflow-height bound: an AVL tree of `n` entries has height `2·log2(n + 1) + 1` or less.
fn avl_bound(n: usize) -> f64 {
    2.0 * ((n + 1) as f64).log2() + 1.0
}

/// Coordinate keys are HashDOS-immune by construction: `home(c) = c`, so a key's probe length is 1
/// for any adversarial pattern. The cap changes nothing here.
#[test]
fn coordinate_keys_probe_one() {
    let n = 4000u64;
    let patterns: [Vec<u64>; 4] = [
        (0..n).collect(),                    // sequential
        (0..n).map(|i| i * 2).collect(),     // strided (floods a low-bit mask)
        (0..n).rev().collect(),              // reversed
        (0..n).map(|i| i ^ 0xA5A).collect(), // xor-scrambled, dense and distinct
    ];
    for keys in patterns {
        let mut t: Table<u64, i64> = Table::new();
        for &k in &keys {
            t.add(k, 1);
        }
        assert_eq!(t.len(), n as usize);
        assert_eq!(
            t.longest_probe(),
            1,
            "coordinate keys stay probe-1 regardless of the (adversarial) pattern"
        );
        assert_eq!(
            t.overflow_len(),
            0,
            "coordinate keys never need the overflow"
        );
    }
}

/// The classic HashDOS payload — keys that are all multiples of a power of two — collides in every
/// `hash & mask` table. The substrate folds by the high bits of the golden-ratio product, and the
/// capped probe keeps the worst case at [`PROBE_CAP`].
#[test]
fn multiples_attack_stays_capped() {
    let n = 4000usize;
    let stride = 1u64 << 20;
    let keys: Vec<u64> = (1..=n as u64).map(|i| i * stride).collect();
    let mut t: Table<u64, i64> = Table::new();
    for &k in &keys {
        t.add(k, 1);
    }
    assert_eq!(t.len(), n);
    assert!(
        t.longest_probe() <= PROBE_CAP as usize,
        "capped probe: worst {} > PROBE_CAP {}",
        t.longest_probe(),
        PROBE_CAP
    );
    for &k in &keys {
        assert_eq!(t.get(&k), Some(&1), "key {k} lost");
    }
}

/// The substrate-aware flood: every key homes to window 0 — a linear-probe table would chain all
/// `n` into one slot. The capped probe places at most [`PROBE_CAP`] in the primary array and routes
/// the rest to the overflow; every key still resolves.
#[test]
fn fold_flood_stays_capped() {
    let n = 12_000usize;
    let bits = 14u32; // home window the flood pins
    let mut keys = Vec::with_capacity(n);
    let mut k = 1u64 << 40;
    while keys.len() < n {
        if (k.wrapping_mul(PHI) >> (64 - bits)) == 0 {
            keys.push(k);
        }
        k += 1;
    }

    let mut t: Table<u64, i64> = Table::with_capacity(n);
    for &k in &keys {
        t.add(k, 1);
    }
    assert_eq!(t.len(), n);
    assert!(
        t.longest_probe() <= PROBE_CAP as usize,
        "flood must stay capped, got worst {}",
        t.longest_probe()
    );
    assert!(
        t.overflow_len() > 0,
        "a flood past the cap uses the overflow"
    );
    for &k in &keys {
        assert_eq!(t.get(&k), Some(&1), "key {k:#x} lost under the flood");
        assert!(
            t.primary_examinations(&k) <= PROBE_CAP,
            "key {k:#x}: {} primary examinations > PROBE_CAP {PROBE_CAP}",
            t.primary_examinations(&k)
        );
    }
}

/// The construction of `HashDos.descent_can_be_wholly_blocked`, executed against the shipped capped
/// table: `PHI⁻¹` turns each of a victim's first [`PROBE_CAP`] window slots into a key that homes
/// there. With every capped slot occupied, the victim is *blocked* and lands in the overflow; its
/// lookup resolves the correct value and stays within the cap. Unaimed keys do not block it.
#[test]
fn crafted_keys_block_a_victims_whole_capped_prefix() {
    // The shipped cap is pinned as a literal: this test is about the cap *value*, and it is the
    // probe-bound control — restoring the uncapped sweep pushes the victim past it.
    const CAP: usize = 16;
    let bits = 5u32; // 32 slots: the window set (8 + 64/5 − 1 = 19) covers CAP
    let len = 1usize << bits;
    let phiinv = phi_inv();

    let victim: u64 = 0xDEAD_BEEF_0000_0001;
    let p = victim.wrapping_mul(PHI);
    let home = (p >> (64 - bits)) as usize & (len - 1);

    // The victim's first CAP probe addresses, de-duplicated — one blocker per distinct slot.
    let mut targets: Vec<usize> = (0..CAP as u32).map(|j| addr(bits, home, p, j)).collect();
    targets.sort_unstable();
    targets.dedup();
    assert!(
        targets.len() >= 8,
        "the capped prefix must be non-trivial, got {} distinct slots",
        targets.len()
    );

    let blockers: Vec<u64> = targets
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            // Product top `bits` = s (so `home(k) = s`); the low bits keep the key large and distinct.
            let low = (i as u64) + 1;
            let product = ((s as u64) << (64 - bits)) | low;
            product.wrapping_mul(phiinv)
        })
        .collect();
    for &b in &blockers {
        assert_ne!(b, victim, "a blocker must not be the victim");
        assert!(b as usize >= len, "a blocker must take the fold path");
    }

    let mut t: Table<u64, i64> = Table::with_capacity(16);
    assert_eq!(t.capacity(), len, "bits_for(16) must be 5");
    for &b in &blockers {
        t.add(b, 1);
    }
    t.add(victim, 42);

    assert_eq!(
        t.get(&victim),
        Some(&42),
        "the blocked victim must be found"
    );
    // The probe bound first: restoring the uncapped sweep places the victim at step CAP + 1 in the
    // primary array, so this is the assertion that reddens.
    assert!(
        t.longest_probe() <= CAP,
        "the capped probe bound: {} > cap {CAP}",
        t.longest_probe()
    );
    assert!(
        t.overflow_len() >= 1,
        "an exhausted capped prefix routes the victim to the overflow"
    );
    assert!(t.primary_examinations(&victim) <= CAP as u32);

    // The control: the same COUNT of unaimed keys does not block the victim — it stays in primary.
    let mut c: Table<u64, i64> = Table::with_capacity(16);
    for i in 0..blockers.len() as u64 {
        c.add(0x1234_5678_9ABC_0000 ^ (i.wrapping_mul(0x9E37_79B9)), 1);
    }
    c.add(victim, 42);
    assert_eq!(c.get(&victim), Some(&42));
    assert_eq!(
        c.overflow_len(),
        0,
        "unaimed keys must not force the overflow"
    );
}

/// Flood: 2¹⁶ keys aimed at one home. Every primary probe examines at most [`PROBE_CAP`] slots; the
/// overflow stays AVL-balanced; every value is correct; removing half then re-querying holds.
#[test]
fn overflow_flood_is_capped_and_balanced() {
    const N: usize = 1 << 16;
    // Presize to a 10-bit table (1024 slots) so `home` is pinned while the flood is inserted.
    let mut t: Table<u64, i64> = Table::with_capacity(600);
    let bits = t.capacity().trailing_zeros();
    assert_eq!(t.capacity(), 1 << 10, "bits_for(600) must be 10");

    // Keys with the top `bits` bits of `k·PHI` zero home to slot 0, found by scanning upward — so the
    // keys are strictly increasing. Random order keeps an un-rebalanced BST shallow, so the height
    // control needs this monotone sequence (an AVL tree is balanced either way); the cap keeps the
    // primary probe bounded and the rest live in the overflow.
    let mut keys: Vec<u64> = Vec::with_capacity(N);
    let mut k = 1u64 << 20;
    while keys.len() < N {
        if (k.wrapping_mul(PHI) >> (64 - bits)) == 0 {
            keys.push(k);
        }
        k += 1;
    }
    let value = |k: u64| (k as i64).wrapping_add(7);

    for &key in &keys {
        assert!(
            key as usize >= t.capacity(),
            "flood key must take the fold path"
        );
        t.insert_merge_last_write(key, value(key));
    }
    assert_eq!(t.len(), N, "every distinct key is stored");

    // The primary probe bound, per key.
    for &k in &keys {
        assert_eq!(t.get(&k), Some(&value(k)), "key {k} lost");
        assert!(
            t.primary_examinations(&k) <= PROBE_CAP,
            "key {k}: {} > PROBE_CAP {PROBE_CAP}",
            t.primary_examinations(&k)
        );
    }
    assert!(
        t.overflow_len() > 0,
        "a flood past the cap must populate the overflow"
    );
    assert!(
        f64::from(t.overflow_height()) <= avl_bound(t.overflow_len()),
        "overflow height {} exceeds the AVL bound {:.2} for {} entries",
        t.overflow_height(),
        avl_bound(t.overflow_len()),
        t.overflow_len()
    );
    assert!(
        t.longest_probe() <= PROBE_CAP as usize,
        "primary probe bound: {} > PROBE_CAP {PROBE_CAP}",
        t.longest_probe()
    );

    // Remove half (the even keys) and re-query: removed read absent, the rest read their value.
    for &k in keys.iter().step_by(2) {
        assert_eq!(t.remove(&k), Some(value(k)), "remove key {k}");
    }
    assert_eq!(t.len(), N / 2);
    for (i, &k) in keys.iter().enumerate() {
        if i % 2 == 0 {
            assert_eq!(t.get(&k), None, "removed key {k} still present");
        } else {
            assert_eq!(t.get(&k), Some(&value(k)), "kept key {k} lost");
        }
    }
    assert!(
        f64::from(t.overflow_height()) <= avl_bound(t.overflow_len()),
        "height {} exceeds the AVL bound {:.2} after removals",
        t.overflow_height(),
        avl_bound(t.overflow_len())
    );
}

/// A dense miss no longer sweeps the occupied block. `n` dense keys index their own slots and form
/// one contiguous occupied block; absent keys that fold into that block are rejected within the cap
/// — the cap replaced a walk of the whole block (measured at ~22300 ns on the bench).
#[test]
fn dense_miss_is_capped() {
    let n = 100_000u64;
    let mut t: Table<u64, i64> = Table::with_capacity(n as usize);
    for k in 0..n {
        t.insert_merge_last_write(k, (k as i64).wrapping_add(1));
    }
    let width = t.capacity() as u64;
    assert!(width >= n, "the dense build must fit one table");

    // Absent keys at or above the slot count: their homes fold into the occupied block.
    let mut examined_bound = 0u32;
    let mut walked = 0usize;
    for k in width..width + 10_000 {
        assert_eq!(t.get(&k), None, "absent key {k} must read None");
        let e = t.primary_examinations(&k);
        assert!(
            e <= PROBE_CAP,
            "absent key {k}: {e} primary examinations > PROBE_CAP {PROBE_CAP}"
        );
        examined_bound = examined_bound.max(e);
        if e > 1 {
            walked += 1;
        }
    }
    assert!(
        walked > 0,
        "the test must exercise a real collision walk, not only home misses"
    );
    assert!(examined_bound <= PROBE_CAP);
    assert!(
        t.longest_probe() <= PROBE_CAP as usize,
        "present-key probe bound: {} > PROBE_CAP {PROBE_CAP}",
        t.longest_probe()
    );
}

/// Deterministic by construction: no seed, so the same keys give the same layout and reads.
#[test]
fn deterministic_layout_and_reads() {
    let keys: Vec<u64> = (0..5000u64).map(|i| i.wrapping_mul(PHI)).collect();
    let build = || {
        let mut t: Table<u64, i64> = Table::with_capacity(keys.len());
        for &k in &keys {
            t.insert_merge_last_write(k, 1);
        }
        t
    };
    let a = build();
    let b = build();
    assert_eq!(a.longest_probe(), b.longest_probe());
    assert_eq!(a.capacity(), b.capacity());
    assert_eq!(a.overflow_len(), b.overflow_len());
    for &k in &keys {
        assert_eq!(a.get(&k), b.get(&k));
    }
    let mut va: Vec<u64> = a.iter().map(|(k, _)| *k).collect();
    let mut vb: Vec<u64> = b.iter().map(|(k, _)| *k).collect();
    va.sort_unstable();
    vb.sort_unstable();
    assert_eq!(va, vb);
}
