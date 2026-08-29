//! Where apeiron starts paying, and where it costs.
//!
//! The headline benchmark is a million keys. A Neverwinter area holds a few hundred entities and a
//! large persistent world a few thousand, so the number that decides whether this belongs in a
//! game engine is measured three orders of magnitude below the one that sells it.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "this is a measurement harness, not library code, and each lint below is listed because this \
              file actually triggers it: sizes and indices narrowed to build keys, bounded by the loop that \
              produced them; counts converted to floats to report rates; a harness would need 2^53 samples \
              to lose a bit. Listed individually rather than suppressed wholesale, so a pedantic lint this \
              file does not yet trigger still surfaces"
)]

use apeiron::coord::{i21, Packed};
use apeiron::table::CountTable;
use std::collections::HashMap;
use std::time::Instant;

fn keys(n: usize) -> Vec<u64> {
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r };
    (0..n).map(|_| {
        let v = next();
        let c = |s: u32| ((v >> s) as i32 % 40) - 20;
        (i21::new(c(0)).unwrap(), i21::new(c(20)).unwrap(), i21::new(c(40)).unwrap()).packed()
    }).collect()
}

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    // Warm, then the median of five, in microseconds.
    for _ in 0..reps { f(); }
    let mut runs: Vec<f64> = (0..5).map(|_| {
        let t = Instant::now();
        for _ in 0..reps { f(); }
        t.elapsed().as_secs_f64() * 1e6 / reps as f64
    }).collect();
    runs.sort_by(f64::total_cmp);
    runs[2]
}

fn main() {
    println!("  {:>8} {:>12} {:>12} {:>12} {:>12}", "n", "std us", "apeiron us", "ap-par16 us", "verdict");
    for &n in &[100usize, 694, 5_000, 50_000, 500_000] {
        let ks = keys(n);
        let mut occ = ks.clone(); occ.sort_unstable(); occ.dedup();
        let bits = (occ.len() * 2).max(64).next_power_of_two().trailing_zeros();
        let reps = if n <= 5_000 { 200 } else { 5 };

        let std_us = time(reps, || {
            let mut m: HashMap<u64, i64> = HashMap::new();
            for &k in &ks { *m.entry(k).or_insert(0) += 1; }
            let s: i64 = occ.iter().map(|k| m.get(k).copied().unwrap_or(0)).sum();
            std::hint::black_box(s);
        });
        let ap_us = time(reps, || {
            let t = CountTable::build(&ks, bits);
            std::hint::black_box(occ.iter().map(|&a| t.get(a)).sum::<i64>());
        });
        let par_us = time(reps.min(20), || {
            let t = CountTable::build_parallel(&ks, bits, 16);
            std::hint::black_box(t.sum_parallel(&occ, 16));
        });
        let best = if ap_us.min(par_us) < std_us { "apeiron" } else { "std" };
        println!("  {n:>8} {std_us:>12.1} {ap_us:>12.1} {par_us:>12.1} {best:>12}");
    }
}
