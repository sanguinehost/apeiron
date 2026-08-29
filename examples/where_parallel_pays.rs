//! Where `CountTable::build_parallel` starts paying, measured against its own serial build.
//!
//! `build_parallel` partitions at every size. The partition is a fixed cost, so below some `n` it
//! is pure overhead -- the criterion run has the serial build beating *every* parallel arm at
//! 100,000 keys, its own included, and losing to its own by 3.7x at a million. This program finds
//! where the two cross.
//!
//! The estimator is the **minimum** of many rounds, not the mean. A loaded machine can only make a
//! round slower, never faster, so the minimum is the closest thing to the uncontended cost that a
//! contended machine can report; a mean mixes in whatever else was running. The two arms are
//! alternated inside each round rather than run in blocks, so a load spike lands on both.
//!
//! Run: `cargo run --release --example where_parallel_pays`

use apeiron::coord::{i21, Packed};
use apeiron::table::CountTable;
use std::hint::black_box;
use std::time::Instant;

const CELL: f32 = 0.25;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 * 2000.0 - 1000.0
    }
}

/// The benchmark's scene, verbatim, so this reads the same workload `spatial_binning` does.
fn scene(n: usize) -> Vec<(i32, i32, i32)> {
    let mut r = Rng(0x243F_6A88_85A3_08D3);
    let clusters: Vec<(f32, f32, f32)> = (0..32).map(|_| (r.f(), r.f() * 0.1, r.f())).collect();
    (0..n)
        .map(|i| {
            let c = clusters[i % clusters.len()];
            let j = |v: f32, r: &mut Rng| v + ((r.next() >> 40) as f32 / 16777.0 - 50.0);
            (
                (j(c.0, &mut r) * CELL) as i32,
                (j(c.1, &mut r) * CELL) as i32,
                (j(c.2, &mut r) * CELL) as i32,
            )
        })
        .collect()
}

fn pack(k: (i32, i32, i32)) -> u64 {
    (
        i21::new(k.0.clamp(i21::MIN, i21::MAX)).unwrap(),
        i21::new(k.1.clamp(i21::MIN, i21::MAX)).unwrap(),
        i21::new(k.2.clamp(i21::MIN, i21::MAX)).unwrap(),
    )
        .packed()
}

fn main() {
    let threads: usize = std::env::var("THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    // `ARM=serial` / `ARM=par` runs one arm alone, so the other's allocation pattern cannot
    // disturb it. Interleaved in one process the parallel arm's per-thread regions churn the
    // free list under the serial arm, and the two designs disagree about the crossover because
    // of it -- which is the whole reason this switch exists.
    let arm = std::env::var("ARM").unwrap_or_else(|_| "both".into());
    println!("build and probe, {threads} threads, best of {rounds} interleaved rounds\n");
    println!("  {:>9} {:>11} {:>11} {:>9}  verdict", "n", "serial ms", "par ms", "speedup");

    for n in [25_000usize, 50_000, 100_000, 150_000, 200_000, 300_000, 400_000, 600_000, 1_000_000]
    {
        let keys = scene(n);
        let mut occupied = keys.clone();
        occupied.sort_unstable();
        occupied.dedup();
        let packed: Vec<u64> = keys.iter().map(|&k| pack(k)).collect();
        let probe: Vec<u64> = occupied.iter().map(|&k| pack(k)).collect();
        let bits = (occupied.len() * 2).next_power_of_two().trailing_zeros();

        let (mut best_s, mut best_p) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..rounds {
            if arm != "par" {
                let t = Instant::now();
                let tab = CountTable::build(&packed, bits);
                black_box(probe.iter().map(|&a| tab.get(a)).sum::<i64>());
                best_s = best_s.min(t.elapsed().as_secs_f64() * 1e3);
            }
            if arm != "serial" {
                let t = Instant::now();
                let tab = CountTable::build_parallel(&packed, bits, threads);
                black_box(tab.sum_parallel(&probe, threads));
                best_p = best_p.min(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        match arm.as_str() {
            "serial" => println!("  {n:>9} {best_s:>11.3}"),
            "par" => println!("  {n:>9} {best_p:>11.3}"),
            _ => {
                let sp = best_s / best_p;
                let v = if sp > 1.05 { "parallel" } else if sp < 0.95 { "SERIAL" } else { "tie" };
                println!("  {n:>9} {best_s:>11.3} {best_p:>11.3} {sp:>8.2}x  {v}");
            }
        }
    }
}
