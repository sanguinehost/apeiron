//! apeiron against `std` and `hashbrown`+`foldhash`, on `hashbrown`'s own bench methodology.
//!
//! Three things taken from `rust-lang/hashbrown/benches/general_ops.rs`, each of which this
//! crate's earlier harnesses got wrong:
//!
//! - **Three key distributions per operation**, not one. `serial` is `0..`, `highbits` is
//!   `0..` under `swap_bytes`, `random` is a multiplicative PRNG. `highbits` is the distribution
//!   that has broken this crate's address twice.
//! - **Values that implement `Drop`** with an observed side effect. A `u64` value lets LLVM
//!   delete a drain or a drop entirely, which reads as an impossibly fast operation; it hid a
//!   real `drain` defect here for several measurements.
//! - **`clone_from` measured apart from `clone`**, because reusing an allocation is a different
//!   operation from making one.

use apeiron::map::{Map, Words};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

const SIZE: usize = 10_000;

type Fold = foldhash::fast::RandomState;
type HbMap<K, V> = hashbrown::HashMap<K, V, Fold>;

/// Observed on drop, so the compiler cannot delete the drop paths.
static SIDE_EFFECT: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct Dropper(usize);

impl Drop for Dropper {
    fn drop(&mut self) {
        SIDE_EFFECT.fetch_add(self.0, Ordering::Relaxed);
    }
}

fn observe() {
    black_box(SIDE_EFFECT.load(Ordering::Relaxed));
}

fn keys(which: &str, n: usize) -> Vec<u64> {
    match which {
        "serial" => (0..n as u64).collect(),
        "highbits" => (0..n as u64).map(u64::swap_bytes).collect(),
        _ => {
            let mut s = 0u64;
            (0..n).map(|_| { s = s.wrapping_add(1).wrapping_mul(3_787_392_781); s }).collect()
        }
    }
}

/// One operation, three implementations, three distributions.
macro_rules! across {
    ($c:expr, $name:literal, $ks:ident, $ap:expr, $st:expr, $hb:expr) => {
        for dist in ["serial", "highbits", "random"] {
            let $ks = keys(dist, SIZE);
            let mut g = $c.benchmark_group(concat!($name, "/"));
            g.bench_function(format!("{dist}/apeiron"), |b| b.iter(|| $ap));
            g.bench_function(format!("{dist}/std"), |b| b.iter(|| $st));
            g.bench_function(format!("{dist}/hb+foldhash"), |b| b.iter(|| $hb));
            g.finish();
        }
    };
}

fn insert(c: &mut Criterion) {
    across!(c, "insert", ks,
        { let mut m: Words<u64, Dropper> = Words::with_capacity(SIZE);
          for (i, &k) in ks.iter().enumerate() { m.insert(k, Dropper(i)); } observe(); m.len() },
        { let mut m: HashMap<u64, Dropper> = HashMap::with_capacity(SIZE);
          for (i, &k) in ks.iter().enumerate() { m.insert(k, Dropper(i)); } observe(); m.len() },
        { let mut m: HbMap<u64, Dropper> = HbMap::with_capacity_and_hasher(SIZE, Fold::default());
          for (i, &k) in ks.iter().enumerate() { m.insert(k, Dropper(i)); } observe(); m.len() });
}

fn entry(c: &mut Criterion) {
    across!(c, "entry", ks,
        { let mut m: Words<u64, usize> = Words::with_capacity(SIZE);
          for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.len() },
        { let mut m: HashMap<u64, usize> = HashMap::with_capacity(SIZE);
          for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.len() },
        { let mut m: HbMap<u64, usize> = HbMap::with_capacity_and_hasher(SIZE, Fold::default());
          for &k in &ks { *m.entry(k).or_insert(0) += 1; } m.len() });
}

fn lookup(c: &mut Criterion) {
    for dist in ["serial", "highbits", "random"] {
        let ks = keys(dist, SIZE);
        let ap: Words<u64, usize> = ks.iter().map(|&k| (k, 1)).collect();
        let st: HashMap<u64, usize> = ks.iter().map(|&k| (k, 1)).collect();
        let hb: HbMap<u64, usize> = ks.iter().map(|&k| (k, 1)).collect();
        let mut g = c.benchmark_group("lookup_hit/");
        g.bench_function(format!("{dist}/apeiron"), |b| b.iter(|| ks.iter().filter_map(|k| ap.get(k)).count()));
        g.bench_function(format!("{dist}/std"), |b| b.iter(|| ks.iter().filter_map(|k| st.get(k)).count()));
        g.bench_function(format!("{dist}/hb+foldhash"), |b| b.iter(|| ks.iter().filter_map(|k| hb.get(k)).count()));
        g.finish();
    }
}

fn clone_ops(c: &mut Criterion) {
    for dist in ["serial", "highbits", "random"] {
        let ks = keys(dist, SIZE);
        let ap: Words<u64, Dropper> = ks.iter().enumerate().map(|(i, &k)| (k, Dropper(i))).collect();
        let st: HashMap<u64, Dropper> = ks.iter().enumerate().map(|(i, &k)| (k, Dropper(i))).collect();
        let hb: HbMap<u64, Dropper> = ks.iter().enumerate().map(|(i, &k)| (k, Dropper(i))).collect();

        let mut g = c.benchmark_group("clone/");
        g.bench_function(format!("{dist}/apeiron"), |b| b.iter(|| { let m = ap.clone(); observe(); m.len() }));
        g.bench_function(format!("{dist}/std"), |b| b.iter(|| { let m = st.clone(); observe(); m.len() }));
        g.bench_function(format!("{dist}/hb+foldhash"), |b| b.iter(|| { let m = hb.clone(); observe(); m.len() }));
        g.finish();

        // The allocation already exists; only the contents are replaced.
        let mut g = c.benchmark_group("clone_from/");
        g.bench_function(format!("{dist}/apeiron"), |b| {
            b.iter_batched_ref(|| ap.clone(), |d| { d.clone_from(&ap); observe(); }, BatchSize::SmallInput)
        });
        g.bench_function(format!("{dist}/std"), |b| {
            b.iter_batched_ref(|| st.clone(), |d| { d.clone_from(&st); observe(); }, BatchSize::SmallInput)
        });
        g.bench_function(format!("{dist}/hb+foldhash"), |b| {
            b.iter_batched_ref(|| hb.clone(), |d| { d.clone_from(&hb); observe(); }, BatchSize::SmallInput)
        });
        g.finish();
    }
}

fn strings(c: &mut Criterion) {
    let ks: Vec<String> = (0..SIZE).map(|i| format!("nw_wswmg{i:07}")).collect();
    let ap: Map<String, usize> = ks.iter().map(|k| (k.clone(), 1)).collect();
    let st: HashMap<String, usize> = ks.iter().map(|k| (k.clone(), 1)).collect();
    let hb: HbMap<String, usize> = ks.iter().map(|k| (k.clone(), 1)).collect();
    let mut g = c.benchmark_group("string_lookup/");
    g.bench_function("apeiron", |b| b.iter(|| ks.iter().filter_map(|k| ap.get_at(k.as_str())).count()));
    g.bench_function("std", |b| b.iter(|| ks.iter().filter_map(|k| st.get(k.as_str())).count()));
    g.bench_function("hb+foldhash", |b| b.iter(|| ks.iter().filter_map(|k| hb.get(k.as_str())).count()));
    g.finish();
}

criterion_group!(field, insert, entry, lookup, clone_ops, strings);
criterion_main!(field);
