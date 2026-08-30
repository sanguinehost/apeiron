//! Where a table's instructions go, split three ways: **moving** bytes into an address, **storing**
//! them, and **retrieving** them.
//!
//! `THREE_ARM=<arm> cargo run --release --example three_costs`, under
//! `perf stat -e instructions,cycles`, divided by `694 * 20_000`. Instructions are the estimator
//! because they do not move with machine load; cycles are printed alongside and are only
//! comparable within one quiet run.
//!
//! Every arm is one `#[inline(never)]` loop, and every arm that builds a table builds it in a
//! *different* frame from the loop being measured, for the reason `probe_cost.rs` documents at
//! length: sharing a frame with `insert` costs a lookup six callee-saved registers and makes the
//! register allocator rematerialise the fold's seed on every key.
//!
//! # The three costs are not measured the same way, and cannot be
//!
//! **Moving** is direct: the address function alone, no table.
//!
//! **Storing** is a difference. An insert loop has to start from an empty table, so the loop being
//! timed contains a `clear` (or an allocation) that is not an insert. Each store arm therefore has
//! a matching `*_only` arm running just that preamble, and the store cost is the subtraction. The
//! `_only` arms are listed in the table so the subtraction is visible rather than asserted.
//!
//! **Retrieving** is direct again, and is the number `probe_cost.rs` reports.
//!
//! One caveat on subtracting *moving* out of the other two: measured alone, the address loop
//! unrolls, and inside a table it cannot. `move_ap_u64` reads 6.54 for a body that is nine
//! instructions written out. So "table = total - moving" is an **upper** bound on the table's
//! share for word keys. The string arms do not have this problem in any size that matters -- a
//! 30-instruction fold amortises its loop overhead already, and `move_ap_str` agrees with
//! `fold_instr.rs` to 0.01.
//!
//! # What it found
//!
//! Instructions per key, 694 keys, 13-byte resrefs, against `hashbrown` + `foldhash`:
//!
//! | | apeiron | hb + foldhash |
//! |---|---|---|
//! | **move** `u64` | **6.5** | 10.1 |
//! | **move** 13-byte key | 30.1 | **22.1** |
//! | **move** the key itself (`String::clone` + drop) | 376.9 | 376.9 |
//! | **store** `u64`, pre-sized | 66.1 | **64.7** |
//! | **store** `u64`, growing | 141.5 | **123.5** |
//! | **store** `String`, pre-sized | 122.1 | **105.0** |
//! | **store** `String`, growing | **205.9** | 206.0 |
//! | **retrieve** `u64` | 33.9 | **33.2** |
//! | **retrieve** `String` | 97.6 | **80.0** |
//!
//! The row that matters most is the third. Copying a 13-byte `String` into the table costs
//! **376.9 instructions**, three times the whole insert, and it is the allocator rather than the
//! copy: the same arm reads 375.6 at 42 bytes and 399.4 at 200, flat across a 3x range in length.
//! Both libraries pay it identically, so the 17 instructions of fold and table that separate them
//! on a `String` insert are **3% of what a `Map<String, V>` actually costs**. A caller who cares
//! about that number should stop copying the key -- intern it, or key on a `u32` handle -- and no
//! hash table can do it for them.
use apeiron::coord::Coord;
use apeiron::fold::fold_bytes_with;
use apeiron::map::{Map, Salt, Words};
use std::hash::{BuildHasher, Hash};
use std::hint::black_box;

const N: usize = 694;
const REPS: usize = 20_000;
/// Keeps a `u64` accumulator inside `usize` on a 32-bit target without a lossy cast.
const MASK: u64 = 0xFFFF_FFFF;

type Fold = foldhash::fast::RandomState;
type HbMap<K, V> = hashbrown::HashMap<K, V, Fold>;

fn words() -> Vec<u64> {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    (0..N)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s & 0xFFFF_FFFF
        })
        .collect()
}

/// `THREE_KEYS=resref|path|long` -- 13 bytes, ~42, or ~200. Three lengths across one `malloc`
/// size class boundary and one `memcpy` regime, which is how the key-copy cost gets attributed to
/// the allocator or to the copy.
fn strings() -> Vec<String> {
    match std::env::var("THREE_KEYS").unwrap_or_else(|_| "resref".into()).as_str() {
        "path" => (0..N)
            .map(|i| format!("modules/neverwinter/areas/nw_wswmg{i:05}.are"))
            .collect(),
        "long" => (0..N).map(|i| format!("{i:05}{}", "x".repeat(195))).collect(),
        _ => (0..N).map(|i| format!("nw_wswmg{i:05}")).collect(),
    }
}

// -- moving: bytes to an address, with no table underneath ---------------------------------------

#[inline(never)]
fn move_ap_u64(ks: &[u64], salt: Salt) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for &k in ks {
            let a = salt.spread(k.coord_with(salt.pepper()));
            acc = acc.wrapping_add(a.place ^ a.id);
        }
    }
    acc
}

#[inline(never)]
fn move_hb_u64(ks: &[u64], h: Fold) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for &k in ks {
            acc = acc.wrapping_add(h.hash_one(k));
        }
    }
    acc
}

#[inline(never)]
fn move_ap_str(ks: &[String], pepper: u64) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for s in ks {
            acc = acc.wrapping_add(fold_bytes_with(pepper, s.as_bytes()));
        }
    }
    acc
}

#[inline(never)]
fn move_hb_str(ks: &[String], h: Fold) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for s in ks {
            acc = acc.wrapping_add(h.hash_one(s.as_str()));
        }
    }
    acc
}

// -- storing: the preamble alone, then the preamble plus the inserts -----------------------------
//
// A build loop cannot start from a full table, so what it times is an allocation plus N inserts.
// Each `*_alloc` arm runs the allocation and the drop with **no** inserts, and the store cost is
// the subtraction. For a `String` key a third arm is needed: `clone_only` times N `String::clone`s
// into a `Vec`, which is the key's own `malloc` and `memcpy` and belongs under *moving*, not
// storing. All three are divided by the same N, so they subtract directly.

#[inline(never)]
fn alloc_ap<K: Coord + Eq, V>(reserve: bool) -> usize {
    let mut n = 0;
    for _ in 0..REPS {
        let m: Map<K, V> = if reserve { Map::with_capacity(N) } else { Map::new() };
        n += m.capacity();
    }
    n
}

#[inline(never)]
fn alloc_hb<K: Hash + Eq, V>(reserve: bool) -> usize {
    let mut n = 0;
    for _ in 0..REPS {
        let m: HbMap<K, V> = if reserve {
            HbMap::with_capacity_and_hasher(N, Fold::default())
        } else {
            HbMap::default()
        };
        n += m.capacity();
    }
    n
}

/// N `String::clone`s and their drops, with no table anywhere: the key's own allocation and copy.
#[inline(never)]
fn clone_only(ks: &[String]) -> usize {
    let mut n = 0;
    for _ in 0..REPS {
        let v: Vec<String> = ks.to_vec();
        n += v.len();
    }
    n
}

/// The same shape for a word key, so the `Vec` machinery cancels out of the string subtraction.
#[inline(never)]
fn clone_only_u64(ks: &[u64]) -> usize {
    let mut n = 0;
    for _ in 0..REPS {
        let v: Vec<u64> = ks.to_vec();
        n += v.len();
    }
    n
}

/// The elliptic face, whose slot is `(u64, V)` and not `(u64, K, V)`: 16 bytes against 24, which
/// is the same width `hashbrown` moves. Growth moves every live entry, so if slot width is what
/// makes our rehome dearer this arm is where it shows.
#[inline(never)]
fn fresh_ap_words(ks: &[u64], reserve: bool) -> usize {
    let mut n = 0;
    for _ in 0..REPS {
        let mut m: Words<u64, u64> = if reserve { Words::with_capacity(N) } else { Words::new() };
        for &k in ks {
            m.insert(k, k);
        }
        n += m.len();
    }
    n
}

/// A fresh table every round. `reserve` picks whether the growth path is in the measurement.
#[inline(never)]
fn fresh_ap<K: Coord + Eq + Clone, V: Clone>(ks: &[K], v: V, reserve: bool) -> usize {
    let mut n = 0;
    for _ in 0..REPS {
        let mut m: Map<K, V> = if reserve { Map::with_capacity(N) } else { Map::new() };
        for k in ks {
            m.insert(k.clone(), v.clone());
        }
        n += m.len();
    }
    n
}

#[inline(never)]
fn fresh_hb<K: Hash + Eq + Clone, V: Clone>(ks: &[K], v: V, reserve: bool) -> usize {
    let mut n = 0;
    for _ in 0..REPS {
        let mut m: HbMap<K, V> = if reserve {
            HbMap::with_capacity_and_hasher(N, Fold::default())
        } else {
            HbMap::default()
        };
        for k in ks {
            m.insert(k.clone(), v.clone());
        }
        n += m.len();
    }
    n
}

/// The same 694 keys as `[u8; 16]`, which is what an NWN `ResRef` actually is: at most sixteen
/// bytes, and this crate already implements `Coord` for the array. No `String`, so no `malloc`,
/// no `free`, and no `bcmp` call -- the key lives in the slot.
fn arrays() -> Vec<[u8; 16]> {
    strings()
        .iter()
        .map(|s| {
            let mut a = [0u8; 16];
            let b = s.as_bytes();
            let n = b.len().min(16);
            a[..n].copy_from_slice(&b[..n]);
            a
        })
        .collect()
}

#[inline(never)]
fn clone_only_arr(ks: &[[u8; 16]]) -> usize {
    let mut n = 0;
    for _ in 0..REPS {
        let v: Vec<[u8; 16]> = ks.to_vec();
        n += v.len();
    }
    n
}

#[inline(never)]
fn move_ap_arr(ks: &[[u8; 16]], pepper: u64) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for k in ks {
            acc = acc.wrapping_add(fold_bytes_with(pepper, k));
        }
    }
    acc
}

/// Eight-byte keys: our slot is `{id, [u8; 8], V}` = 24 bytes against `hashbrown`'s 16, the same
/// 8-byte difference as at sixteen. If the store gap is the identity word rather than the key
/// width, it is the same at both.
fn arrays8() -> Vec<[u8; 8]> {
    strings()
        .iter()
        .map(|s| {
            let mut a = [0u8; 8];
            let b = s.as_bytes();
            let n = b.len().min(8);
            a[..n].copy_from_slice(&b[..n]);
            a
        })
        .collect()
}

#[inline(never)]
fn move_hb_arr(ks: &[[u8; 16]], h: Fold) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for k in ks {
            acc = acc.wrapping_add(h.hash_one(k));
        }
    }
    acc
}

#[inline(never)]
fn get_ap_arr(m: &Map<[u8; 16], u64>, ks: &[[u8; 16]]) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for k in ks {
            acc = acc.wrapping_add(m.get(k).copied().unwrap_or(0));
        }
    }
    acc
}

#[inline(never)]
fn get_hb_arr(m: &HbMap<[u8; 16], u64>, ks: &[[u8; 16]]) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for k in ks {
            acc = acc.wrapping_add(m.get(k).copied().unwrap_or(0));
        }
    }
    acc
}

// -- retrieving ----------------------------------------------------------------------------------

#[inline(never)]
fn get_ap_u64(m: &Words<u64, u64>, ks: &[u64]) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for &k in ks {
            acc = acc.wrapping_add(m.get(&k).copied().unwrap_or(0));
        }
    }
    acc
}

#[inline(never)]
fn get_hb_u64(m: &HbMap<u64, u64>, ks: &[u64]) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for &k in ks {
            acc = acc.wrapping_add(m.get(&k).copied().unwrap_or(0));
        }
    }
    acc
}

#[inline(never)]
fn get_ap_str(m: &Map<String, u64>, ks: &[String]) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for k in ks {
            acc = acc.wrapping_add(m.get_at(k.as_str()).copied().unwrap_or(0));
        }
    }
    acc
}

#[inline(never)]
fn get_hb_str(m: &HbMap<String, u64>, ks: &[String]) -> u64 {
    let mut acc = 0u64;
    for _ in 0..REPS {
        for k in ks {
            acc = acc.wrapping_add(m.get(k.as_str()).copied().unwrap_or(0));
        }
    }
    acc
}

fn main() {
    let arm = std::env::var("THREE_ARM").unwrap_or_else(|_| "list".into());
    let out: usize = match arm.as_str() {
        // moving
        "move_ap_u64" => usize::try_from(move_ap_u64(&words(), Salt::new(0x5EED)) & MASK).unwrap_or(0),
        "move_hb_u64" => usize::try_from(move_hb_u64(&words(), Fold::default()) & MASK).unwrap_or(0),
        "move_ap_str" => usize::try_from(move_ap_str(&strings(), black_box(0x5EED)) & MASK).unwrap_or(0),
        "move_hb_str" => usize::try_from(move_hb_str(&strings(), Fold::default()) & MASK).unwrap_or(0),

        // storing, word keys
        "alloc_ap_u64" => alloc_ap::<u64, u64>(true),
        "alloc_hb_u64" => alloc_hb::<u64, u64>(true),
        "clone_u64" => clone_only_u64(&words()),
        "fresh_ap_u64" => fresh_ap(&words(), 1u64, true),
        "fresh_hb_u64" => fresh_hb(&words(), 1u64, true),
        "grow_ap_u64" => fresh_ap(&words(), 1u64, false),
        "fresh_ap_words" => fresh_ap_words(&words(), true),
        "grow_ap_words" => fresh_ap_words(&words(), false),
        "grow_hb_u64" => fresh_hb(&words(), 1u64, false),

        // storing, string keys
        "alloc_ap_str" => alloc_ap::<String, u64>(true),
        "alloc_hb_str" => alloc_hb::<String, u64>(true),
        "clone_str" => clone_only(&strings()),
        "fresh_ap_str" => fresh_ap(&strings(), 1u64, true),
        "fresh_hb_str" => fresh_hb(&strings(), 1u64, true),
        "grow_ap_str" => fresh_ap(&strings(), 1u64, false),
        "grow_hb_str" => fresh_hb(&strings(), 1u64, false),

        "clone_arr" => clone_only_arr(&arrays()),
        "move_ap_arr" => usize::try_from(move_ap_arr(&arrays(), black_box(0x5EED)) & MASK).unwrap_or(0),
        "fresh_ap_arr8" => fresh_ap(&arrays8(), 1u64, true),
        "fresh_hb_arr8" => fresh_hb(&arrays8(), 1u64, true),
        "fresh_ap_arrz" => fresh_ap(&arrays(), (), true),
        "fresh_hb_arrz" => fresh_hb(&arrays(), (), true),
        "move_hb_arr" => usize::try_from(move_hb_arr(&arrays(), Fold::default()) & MASK).unwrap_or(0),
        "alloc_ap_arr" => alloc_ap::<[u8; 16], u64>(true),
        "alloc_hb_arr" => alloc_hb::<[u8; 16], u64>(true),
        "fresh_ap_arr" => fresh_ap(&arrays(), 1u64, true),
        "fresh_hb_arr" => fresh_hb(&arrays(), 1u64, true),
        "grow_ap_arr" => fresh_ap(&arrays(), 1u64, false),
        "grow_hb_arr" => fresh_hb(&arrays(), 1u64, false),
        "get_ap_arr" => {
            let ks = arrays();
            let mut m: Map<[u8; 16], u64> = Map::with_capacity(ks.len());
            for k in &ks { m.insert(*k, 1); }
            usize::try_from(get_ap_arr(&m, &ks) & MASK).unwrap_or(0)
        }
        "get_hb_arr" => {
            let ks = arrays();
            let mut m: HbMap<[u8; 16], u64> = HbMap::with_capacity_and_hasher(ks.len(), Fold::default());
            for k in &ks { m.insert(*k, 1); }
            usize::try_from(get_hb_arr(&m, &ks) & MASK).unwrap_or(0)
        }

        // retrieving
        "get_ap_u64" => {
            let ks = words();
            let m: Words<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
            usize::try_from(get_ap_u64(&m, &ks) & MASK).unwrap_or(0)
        }
        "get_hb_u64" => {
            let ks = words();
            let m: HbMap<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
            usize::try_from(get_hb_u64(&m, &ks) & MASK).unwrap_or(0)
        }
        "get_ap_str" => {
            let ks = strings();
            let mut m: Map<String, u64> = Map::with_capacity(N);
            for k in &ks {
                m.insert(k.clone(), 1);
            }
            usize::try_from(get_ap_str(&m, &ks) & MASK).unwrap_or(0)
        }
        "get_hb_str" => {
            let ks = strings();
            let mut m: HbMap<String, u64> = HbMap::with_capacity_and_hasher(N, Fold::default());
            for k in &ks {
                m.insert(k.clone(), 1);
            }
            usize::try_from(get_hb_str(&m, &ks) & MASK).unwrap_or(0)
        }
        _ => {
            println!("THREE_ARM=move_{{ap,hb}}_{{u64,str}} | store_* | fresh_* | grow_* | get_*");
            0
        }
    };
    println!("{arm}: {}", black_box(out));
}
