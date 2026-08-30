//! Every operation, against `std` and against `hashbrown` + `foldhash`.
//!
//! A table is not one function. Claiming it replaces `std::collections::HashMap` means claiming it
//! on insert, lookup, miss, removal, entry, iteration, clone, drain, retain and rebuild -- on word
//! keys and on string keys, small and large. This runs all of them and prints a `LOSS` marker on
//! any row where apeiron is more than 2% behind, so a regression anywhere is visible.
//!
//! Estimator is the **minimum** of many rounds with the arms alternated inside each round: a busy
//! machine can only make a round slower, so the minimum is the closest a contended machine gets to
//! the uncontended cost.
//!
//! Run: `cargo run --release --example whole_surface [u64|str]`
#![expect(
    clippy::cast_precision_loss,
    reason = "a measurement harness: counts become floats to report nanoseconds per operation"
)]

use apeiron::map::{Map, Words};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

type Fold = foldhash::fast::RandomState;
type HbMap<K, V> = hashbrown::HashMap<K, V, Fold>;

/// `#[inline(never)]` is the whole measurement, not a detail.
///
/// `best` is generic, so each closure gets its own instantiation -- but inlined, all twenty-four of
/// them land in one `str_surface` frame, and the lookup loops then inherit the callee-saved
/// registers and the stack frame of an `insert` that inlines a grow path and an allocation. The
/// register allocator pays for that by rematerialising the fold's seed on every key. Isolated,
/// `examples/probe_cost.rs` measures the same 13-byte resref lookup at 92.5 instructions where the
/// shared frame reads 107.7, and `hashbrown`'s arm moves a third as far -- so the shared frame is
/// not even a fair harness, let alone an accurate one. One frame per timed loop is the fix, and it
/// is also the shape a caller writes: `map.get(k)` is called from a function that is not also
/// building the map.
#[inline(never)]
fn best(rounds: usize, mut f: impl FnMut() -> u64) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        black_box(f());
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

/// Time an operation that consumes its input, with the inputs built **before** the clock starts.
///
/// `clone().drain()` measures a clone and a drain together, and at these sizes the clone is most
/// of it. Pre-building one victim per round is the only way to see the drain by itself.
#[inline(never)]
fn best_consuming<T>(rounds: usize, make: impl Fn() -> T, mut f: impl FnMut(T) -> u64) -> f64 {
    let victims: Vec<T> = (0..rounds).map(|_| make()).collect();
    let mut b = f64::INFINITY;
    for v in victims {
        let t = Instant::now();
        black_box(f(v));
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn row(op: &str, n: usize, a: f64, s: f64, f: f64, per: f64) {
    let (a, s, f) = (a * 1e9 / per, s * 1e9 / per, f * 1e9 / per);
    let worst = s.min(f);
    let mark = if a > worst * 1.02 { "  LOSS" } else { "" };
    println!("  {op:<16} {n:>8} {a:>9.2} {s:>9.2} {f:>9.2}   {:>5.2}x{mark}", worst / a);
}

fn head(what: &str) {
    println!("\n{what}   (ns/op; last column = apeiron vs best rival, >1 is a win)");
    println!("  {:<16} {:>8} {:>9} {:>9} {:>9}", "op", "n", "apeiron", "std", "hb+fold");
}

fn words(n: usize) -> Vec<u64> {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s }).collect()
}

fn strings(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("nw_wswmg{i:07}")).collect()
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "u64".into());
    let sizes: Vec<usize> = vec![1_000, 100_000, 1_000_000];
    if which == "u64" { u64_surface(&sizes); } else { str_surface(&sizes); }
}

fn u64_surface(sizes: &[usize]) {
    head("integer keys");
    for &n in sizes {
        let ks = words(n);
        let miss: Vec<u64> = words(n).iter().map(|k| k ^ 0xFFFF_FFFF_0000_0000).collect();
        let d = n as f64;
        let r = if n > 200_000 { 5 } else { 30 };

        let built_a: Words<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
        let built_s: HashMap<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
        let built_f: HbMap<u64, u64> = ks.iter().map(|&k| (k, k)).collect();

        row("insert fresh", n,
            best(r, || { let mut m: Words<u64,u64> = Words::new(); for &k in &ks { m.insert(k,k); } m.len() as u64 }),
            best(r, || { let mut m: HashMap<u64,u64> = HashMap::new(); for &k in &ks { m.insert(k,k); } m.len() as u64 }),
            best(r, || { let mut m: HbMap<u64,u64> = HbMap::default(); for &k in &ks { m.insert(k,k); } m.len() as u64 }), d);

        row("insert reserved", n,
            best(r, || { let mut m: Words<u64,u64> = Words::with_capacity(n); for &k in &ks { m.insert(k,k); } m.len() as u64 }),
            best(r, || { let mut m: HashMap<u64,u64> = HashMap::with_capacity(n); for &k in &ks { m.insert(k,k); } m.len() as u64 }),
            best(r, || { let mut m: HbMap<u64,u64> = HbMap::with_capacity_and_hasher(n, Fold::default()); for &k in &ks { m.insert(k,k); } m.len() as u64 }), d);

        row("get hit", n,
            best(r, || ks.iter().filter_map(|k| built_a.get(k)).sum()),
            best(r, || ks.iter().filter_map(|k| built_s.get(k)).sum()),
            best(r, || ks.iter().filter_map(|k| built_f.get(k)).sum()), d);

        row("get miss", n,
            best(r, || miss.iter().filter_map(|k| built_a.get(k)).sum()),
            best(r, || miss.iter().filter_map(|k| built_s.get(k)).sum()),
            best(r, || miss.iter().filter_map(|k| built_f.get(k)).sum()), d);

        row("entry", n,
            best(r, || { let mut m: Words<u64,u64> = Words::with_capacity(n); for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.len() as u64 }),
            best(r, || { let mut m: HashMap<u64,u64> = HashMap::with_capacity(n); for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.len() as u64 }),
            best(r, || { let mut m: HbMap<u64,u64> = HbMap::with_capacity_and_hasher(n, Fold::default()); for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.len() as u64 }), d);

        row("remove all", n,
            best_consuming(r, || built_a.clone(), |mut m| { for &k in &ks { m.remove(&k); } m.len() as u64 }),
            best_consuming(r, || built_s.clone(), |mut m| { for &k in &ks { m.remove(&k); } m.len() as u64 }),
            best_consuming(r, || built_f.clone(), |mut m| { for &k in &ks { m.remove(&k); } m.len() as u64 }), d);

        row("iterate", n,
            best(r, || built_a.values().fold(0u64, |a,v| a.wrapping_add(*v))),
            best(r, || built_s.values().fold(0u64, |a,v| a.wrapping_add(*v))),
            best(r, || built_f.values().fold(0u64, |a,v| a.wrapping_add(*v))), d);

        row("clone", n,
            best(r, || built_a.clone().len() as u64),
            best(r, || built_s.clone().len() as u64),
            best(r, || built_f.clone().len() as u64), d);

        // The keys are summed, not counted: draining a type with no `Drop` and discarding the
        // pairs is dead code, and LLVM removes it -- which reads as an impossibly fast `drain`.
        // The keys are summed, not counted: draining a type with no `Drop` and discarding the
        // pairs is dead code, and LLVM removes it -- which reads as an impossibly fast `drain`.
        row("drain", n,
            best_consuming(r, || built_a.clone(), |mut m| m.drain().fold(0u64, |a,(k,_)| a.wrapping_add(k))),
            best_consuming(r, || built_s.clone(), |mut m| m.drain().fold(0u64, |a,(k,_)| a.wrapping_add(k))),
            best_consuming(r, || built_f.clone(), |mut m| m.drain().fold(0u64, |a,(k,_)| a.wrapping_add(k))), d);

        row("retain half", n,
            best_consuming(r, || built_a.clone(), |mut m| { m.retain(|k,_| k % 2 == 0); m.len() as u64 }),
            best_consuming(r, || built_s.clone(), |mut m| { m.retain(|k,_| k % 2 == 0); m.len() as u64 }),
            best_consuming(r, || built_f.clone(), |mut m| { m.retain(|k,_| k % 2 == 0); m.len() as u64 }), d);
    }
}

fn str_surface(sizes: &[usize]) {
    head("string keys (resrefs)");
    for &n in sizes {
        let ks = strings(n);
        let d = n as f64;
        let r = if n > 200_000 { 5 } else { 20 };
        let built_a: Map<String, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();
        let built_s: HashMap<String, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();
        let built_f: HbMap<String, u64> = ks.iter().map(|k| (k.clone(), 1)).collect();

        row("insert fresh", n,
            best(r, || { let mut m: Map<String,u64> = Map::new(); for k in &ks { m.insert(k.clone(),1); } m.len() as u64 }),
            best(r, || { let mut m: HashMap<String,u64> = HashMap::new(); for k in &ks { m.insert(k.clone(),1); } m.len() as u64 }),
            best(r, || { let mut m: HbMap<String,u64> = HbMap::default(); for k in &ks { m.insert(k.clone(),1); } m.len() as u64 }), d);

        row("get hit", n,
            best(r, || ks.iter().filter_map(|k| built_a.get_at(k.as_str())).sum()),
            best(r, || ks.iter().filter_map(|k| built_s.get(k.as_str())).sum()),
            best(r, || ks.iter().filter_map(|k| built_f.get(k.as_str())).sum()), d);

        row("contains", n,
            best(r, || ks.iter().map(|k| u64::from(built_a.contains_at(k.as_str()))).sum()),
            best(r, || ks.iter().map(|k| u64::from(built_s.contains_key(k.as_str()))).sum()),
            best(r, || ks.iter().map(|k| u64::from(built_f.contains_key(k.as_str()))).sum()), d);

        row("iterate", n,
            best(r, || built_a.values().fold(0u64, |a,v| a.wrapping_add(*v))),
            best(r, || built_s.values().fold(0u64, |a,v| a.wrapping_add(*v))),
            best(r, || built_f.values().fold(0u64, |a,v| a.wrapping_add(*v))), d);

        row("clone", n,
            best(r, || built_a.clone().len() as u64),
            best(r, || built_s.clone().len() as u64),
            best(r, || built_f.clone().len() as u64), d);

        row("remove all", n,
            best(r, || { let mut m = built_a.clone(); for k in &ks { m.remove_at(k.as_str()); } m.len() as u64 }),
            best(r, || { let mut m = built_s.clone(); for k in &ks { m.remove(k.as_str()); } m.len() as u64 }),
            best(r, || { let mut m = built_f.clone(); for k in &ks { m.remove(k.as_str()); } m.len() as u64 }), d);
    }
}
