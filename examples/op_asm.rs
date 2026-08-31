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

/// Override with `AP_N`. The surface harness disagrees with this file about `clone` at 100,000
/// and agrees at 20,000, so the size has to be a knob rather than a constant.
static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("AP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000)
});

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

/// The shape `examples/whole_surface.rs` uses for its `get hit` row.
#[inline(never)]
fn ap_get_fm(m: &Words<u64, u64>, ks: &[u64]) -> u64 {
    ks.iter().filter_map(|k| m.get(k)).sum()
}

#[inline(never)]
fn hb_get_fm(m: &HbMap<u64, u64>, ks: &[u64]) -> u64 {
    ks.iter().filter_map(|k| m.get(k)).sum()
}

#[inline(never)]
fn ap_iter(m: &Words<u64, u64>) -> u64 {
    m.values().fold(0u64, |a, v| a.wrapping_add(*v))
}

#[inline(never)]
fn hb_iter(m: &HbMap<u64, u64>) -> u64 {
    m.values().fold(0u64, |a, v| a.wrapping_add(*v))
}

/// `black_box(&c)` and not `m.clone().len()`: `len()` of a clone is known without cloning and a
/// `u64` has no `Drop`, so the expression is elidable in principle. It is not in fact elided here
/// -- see below -- but a benchmark should not depend on that.
///
/// # What the counter found, and what it was not
///
/// `std::collections::HashMap<u64, u64>::clone` reads **0.00 instructions an entry** against
/// `hashbrown` 0.15's 14.66 and this crate's 9.91. That is not a deleted arm, which is what it
/// looks like and what it was first taken for. `objdump` on
/// `hashbrown::raw::RawTable<(u64, u64)>::clone` inside `std` shows two `call *%r14` to
/// `memcpy@GLIBC_2.14` and no per-element loop at all: one memcpy for the control bytes, one for
/// the buckets. glibc's memcpy retires a handful of instructions for 1.8 MB, so ~0 instructions an
/// entry is the truth rather than an artifact.
///
/// That is `hashbrown`'s `T: Copy` clone specialisation, which needs `feature(specialization)`.
/// `std` vendors `hashbrown` and builds it with the `nightly` feature; **no crate on stable can
/// have it**, `hashbrown` as published on crates.io included -- which is why the crates.io build
/// takes 14.66 instructions an entry for the same operation on the same type. It is the reason the
/// `clone+drop` row loses to `std` and beats `hashbrown` at once, and it is not a defect in this
/// crate's clone.
///
#[inline(never)]
fn ap_clone(m: &Words<u64, u64>) -> u64 {
    let c = m.clone();
    black_box(&c);
    c.len() as u64
}

#[inline(never)]
fn hb_clone(m: &HbMap<u64, u64>) -> u64 {
    let c = m.clone();
    black_box(&c);
    c.len() as u64
}

#[inline(never)]
fn std_clone(m: &std::collections::HashMap<u64, u64>) -> u64 {
    let c = m.clone();
    black_box(&c);
    c.len() as u64
}

fn keys() -> Vec<u64> {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    (0..*N)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
        .collect()
}

/// Counts two arms over victims built **outside** the counted region, alternating them.
///
/// Two mistakes this file made, both caught by disagreeing with `examples/whole_surface.rs`
/// rather than by inspection:
///
/// The first version wrote `c.count(|| ap_remove(&mut built.clone(), &ks))`, which put a
/// twenty-thousand-entry clone and its allocation inside the count. It priced `hashbrown`'s remove
/// at 76.2 instructions where the surface harness reads 55.0.
///
/// The second ran every round of one arm before the first round of the other, so the machine's
/// frequency and cache state drifted across an arm rather than across a round. That is the same
/// defect the surface harness had, and it showed there as a row reporting apeiron slower in
/// nanoseconds and faster in cycles at once.
fn arms<A, B>(
    c: &Counter,
    ma: impl Fn() -> A,
    mut fa: impl FnMut(A) -> u64,
    mb: impl Fn() -> B,
    mut fb: impl FnMut(B) -> u64,
) {
    let (mut ai, mut ay, mut bi, mut by) = (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for r in 0..8 {
        // The order swaps every round. Alternating the arms bounds the machine's drift to one
        // round, but running the same arm first in every round hands that arm every cold start:
        // measured on `get`, apeiron-always-first read 0.83x where apeiron-always-second read
        // 0.96x, and neither of those is the number. Swapping makes the position average out.
        let (va, vb) = (ma(), mb());
        if r % 2 == 0 {
            let (v, k) = c.count(|| fa(va));
            black_box(v);
            ai = ai.min(k.instructions as f64 / *N as f64);
            ay = ay.min(k.cycles as f64 / *N as f64);
            let (v, k) = c.count(|| fb(vb));
            black_box(v);
            bi = bi.min(k.instructions as f64 / *N as f64);
            by = by.min(k.cycles as f64 / *N as f64);
        } else {
            let (v, k) = c.count(|| fb(vb));
            black_box(v);
            bi = bi.min(k.instructions as f64 / *N as f64);
            by = by.min(k.cycles as f64 / *N as f64);
            let (v, k) = c.count(|| fa(va));
            black_box(v);
            ai = ai.min(k.instructions as f64 / *N as f64);
            ay = ay.min(k.cycles as f64 / *N as f64);
        }
    }
    println!("  apeiron      {ai:>7.2} instructions  {ay:>7.2} cycles");
    println!("  hb+fold      {bi:>7.2} instructions  {by:>7.2} cycles");
    println!("  ratio        {:>7.2}x              {:>7.2}x   (>1 is a win)", bi / ai, by / ay);
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
    let built_s: std::collections::HashMap<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
    println!("  capacity: apeiron {} hb {} std {}", built_a.capacity(), built_f.capacity(), built_s.capacity());
    println!("{op}, n={}", *N);
    match op.as_str() {
        "remove" => arms(&c,
            || built_a.clone(), |mut m| ap_remove(&mut m, &ks),
            || built_f.clone(), |mut m| hb_remove(&mut m, &ks)),
        "getfm" => arms(&c,
            || (), |()| ap_get_fm(&built_a, &ks),
            || (), |()| hb_get_fm(&built_f, &ks)),
        "clone" => arms(&c,
            || (), |()| ap_clone(&built_a),
            || (), |()| hb_clone(&built_f)),
        "clonestd" => arms(&c,
            || (), |()| std_clone(&built_s),
            || (), |()| hb_clone(&built_f)),
        "iter" => arms(&c,
            || (), |()| ap_iter(&built_a),
            || (), |()| hb_iter(&built_f)),
        _ => arms(&c,
            || (), |()| ap_get(&built_a, &ks),
            || (), |()| hb_get(&built_f, &ks)),
    }
}
