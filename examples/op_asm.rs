//! One operation per out-of-line function, so `objdump` can be pointed at the loop that costs.
//!
//! Builds a map per arm, runs the named operation over every key in an `#[inline(never)]` frame,
//! and prints instructions and cycles per operation from the in-process counter. Takes `OP` and
//! prints one line per arm; the point of the file is that `objdump -d --disassemble='...'` on the
//! resulting binary shows exactly the instructions the counter just priced.
//!
//! Run: `taskset -c 2 cargo run --release --example op_asm remove`

#![expect(clippy::cast_precision_loss, reason = "a harness printing per-operation rates")]

#[path = "support/counter.rs"]
mod counter;

use apeiron::map::Words;
use counter::Counter;
use std::hint::black_box;

type Fold = foldhash::fast::RandomState;
type HbMap<K, V> = hashbrown::HashMap<K, V, Fold>;

const N: usize = 20_000;

#[inline(never)]
fn ap_remove(m: &mut Words<u64, u64>, ks: &[u64]) -> u64 {
    let mut s = 0u64;
    for &k in ks {
        s = s.wrapping_add(m.remove(&k).unwrap_or(0));
    }
    s
}

#[inline(never)]
fn hb_remove(m: &mut HbMap<u64, u64>, ks: &[u64]) -> u64 {
    let mut s = 0u64;
    for &k in ks {
        s = s.wrapping_add(m.remove(&k).unwrap_or(0));
    }
    s
}

#[inline(never)]
fn ap_get(m: &Words<u64, u64>, ks: &[u64]) -> u64 {
    let mut s = 0u64;
    for &k in ks {
        s = s.wrapping_add(m.get(&k).copied().unwrap_or(0));
    }
    s
}

#[inline(never)]
fn hb_get(m: &HbMap<u64, u64>, ks: &[u64]) -> u64 {
    let mut s = 0u64;
    for &k in ks {
        s = s.wrapping_add(m.get(&k).copied().unwrap_or(0));
    }
    s
}

fn keys() -> Vec<u64> {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    (0..N)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
        .collect()
}

/// Counts a consumer over a victim built **outside** the counted region.
///
/// The first version of this file wrote `c.count(|| ap_remove(&mut built.clone(), &ks))`, which
/// put a twenty-thousand-entry clone and its allocation inside the count. It priced `hashbrown`'s
/// remove at 76.2 instructions where `examples/whole_surface.rs` reads 55.0 for the same
/// operation, and the 21 instructions of daylight between two of this crate's own harnesses is
/// what said one of them was wrong.
#[expect(clippy::many_single_char_names, reason = "`i` and `y` are the instruction and cycle minima, `k` the counts, `v` the value under test")]
fn report<T>(c: &Counter, name: &str, make: impl Fn() -> T, mut f: impl FnMut(T) -> u64) {
    let (mut i, mut y) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..5 {
        let v = make();
        let (v, k) = c.count(|| f(v));
        black_box(v);
        i = i.min(k.instructions as f64 / N as f64);
        y = y.min(k.cycles as f64 / N as f64);
    }
    println!("  {name:<12} {i:>7.2} instructions  {y:>7.2} cycles");
}

fn main() {
    let op = std::env::args().nth(1).unwrap_or_else(|| "remove".into());
    let Some(c) = Counter::open() else {
        println!("counters unavailable");
        return;
    };
    let ks = keys();
    let built_a: Words<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
    let built_f: HbMap<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
    println!("{op}, n={N}");
    if op == "remove" {
        report(&c, "apeiron", || built_a.clone(), |mut m| ap_remove(&mut m, &ks));
        report(&c, "hb+fold", || built_f.clone(), |mut m| hb_remove(&mut m, &ks));
    } else {
        report(&c, "apeiron", || (), |()| ap_get(&built_a, &ks));
        report(&c, "hb+fold", || (), |()| hb_get(&built_f, &ks));
    }
}
