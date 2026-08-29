//! The flatness gate, applied to the competition.
//!
//! `probe_cost` says hashbrown+fxhash is the fastest arm. That is only a target worth chasing if it
//! is doing the same job -- a table is not faster for being fast on one key distribution and
//! quadratic on another. This runs the same distributions `tests/the_probe_stays_flat.rs` holds
//! apeiron to, against each competitor, and reports lookup time rather than probe length because
//! the competitors' internals are not ours to read.
#![expect(
    clippy::cast_precision_loss,
    reason = "this is a measurement harness, not library code, and each lint below is listed because this \
              file actually triggers it: counts converted to floats to report rates; a harness would need \
              2^53 samples to lose a bit. Listed individually rather than suppressed wholesale, so a \
              pedantic lint this file does not yet trigger still surfaces"
)]

use apeiron::map::Words;
use hashbrown::HashMap as HbMap;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;
use std::time::Instant;

type Fx = BuildHasherDefault<rustc_hash::FxHasher>;
type Fold = foldhash::fast::RandomState;

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..reps.clamp(1, 3) { f(); }
    let mut v: Vec<f64> = (0..7).map(|_| {
        let t = Instant::now();
        for _ in 0..reps { f(); }
        t.elapsed().as_secs_f64() * 1e9 / reps as f64
    }).collect();
    v.sort_by(f64::total_cmp);
    v[3]
}

fn main() {
    type Dist = (&'static str, fn(usize) -> u64);
    let dists: [Dist; 6] = [
        ("serial", |i| i as u64),
        ("highbits", |i| (i as u64).swap_bytes()),
        ("i21hi", |i| ((i as u64) & 0x1F_FFFF) << 42),
        ("aligned*4096", |i| (i as u64) * 4096),
        ("pairs", |i| (((i % 80) as u64) << 32) | (i / 80) as u64),
        ("random", |i| (i as u64).wrapping_add(1).wrapping_mul(3_787_392_781)),
    ];
    for &n in &[5_000usize, 200_000] {
        println!("\n  n = {n}   ns per lookup (hit), median of 7");
        println!("  {:<14} {:>10} {:>10} {:>10} {:>10}", "distribution", "apeiron", "hb+fx", "hb+fold", "std");
        let mut worst = [0f64; 4];
        for (name, f) in dists {
            let ks: Vec<u64> = { let mut v: Vec<u64> = (0..n).map(f).collect(); v.sort_unstable(); v.dedup(); v };
            let reps = (2_000_000 / ks.len()).clamp(3, 400);
            let mut ap: Words<u64, u64> = Words::with_capacity(ks.len());
            let mut hx: HbMap<u64, u64, Fx> = HbMap::with_capacity_and_hasher(ks.len(), Fx::default());
            let mut hf: HbMap<u64, u64, Fold> = HbMap::with_capacity_and_hasher(ks.len(), Fold::default());
            let mut st: HashMap<u64, u64> = HashMap::with_capacity(ks.len());
            for &k in &ks { ap.insert(k, k); hx.insert(k, k); hf.insert(k, k); st.insert(k, k); }
            let m = ks.len() as f64;
            let r = [
                time(reps, || { black_box(ks.iter().filter(|&&k| ap.get(&k).is_some()).count()); }) / m,
                time(reps, || { black_box(ks.iter().filter(|&&k| hx.contains_key(&k)).count()); }) / m,
                time(reps, || { black_box(ks.iter().filter(|&&k| hf.contains_key(&k)).count()); }) / m,
                time(reps, || { black_box(ks.iter().filter(|&&k| st.contains_key(&k)).count()); }) / m,
            ];
            for i in 0..4 { worst[i] = worst[i].max(r[i]); }
            println!("  {name:<14} {:>10.2} {:>10.2} {:>10.2} {:>10.2}", r[0], r[1], r[2], r[3]);
        }
        println!("  {:<14} {:>10.2} {:>10.2} {:>10.2} {:>10.2}   <- what a table is judged on",
                 "WORST", worst[0], worst[1], worst[2], worst[3]);
    }
}
