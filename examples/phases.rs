//! Where `build_parallel` + `sum_parallel` actually spend the 10 ms.
//!
//! Same scene as `benches/spatial_binning.rs`, so the numbers are comparable to the headline.

use apeiron::coord::{finish, i21, Packed};
use apeiron::table::{shard_of, CountTable};
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
    (i21::new(k.0).unwrap(), i21::new(k.1).unwrap(), i21::new(k.2).unwrap()).packed()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn best(runs: usize, mut f: impl FnMut() -> f64) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..runs {
        b = b.min(f());
    }
    b
}

fn main() {
    let n = 1_000_000usize;
    let keys = scene(n);
    let mut occ = keys.clone();
    occ.sort_unstable();
    occ.dedup();
    let packed: Vec<u64> = keys.iter().map(|&k| pack(k)).collect();
    let probe: Vec<u64> = occ.iter().map(|&k| pack(k)).collect();
    let bits = (occ.len() * 2).next_power_of_two().trailing_zeros();
    println!("n={n}  distinct={}  distinct/total={:.3}", occ.len(), occ.len() as f64 / n as f64);

    // Per-shard load and distinct load, which is what `region_for` is guessing at.
    let mut load = [0usize; 256];
    for &a in &packed {
        load[shard_of(a)] += 1;
    }
    let mut dist = [0usize; 256];
    for &a in &probe {
        dist[shard_of(a)] += 1;
    }
    let (lmax, dmax) = (*load.iter().max().unwrap(), *dist.iter().max().unwrap());
    let slots: usize = load.iter().map(|&l| if l == 0 { 0 } else { (l * 2).max(32).next_power_of_two() }).sum();
    println!("  shard load  max {lmax:>7}  mean {:>7.0}", n as f64 / 256.0);
    println!("  shard distinct max {dmax:>7}  mean {:>7.0}", occ.len() as f64 / 256.0);
    println!("  slots allocated {slots} = {:.1} MB, occupancy {:.1}%",
             slots as f64 * 16.0 / 1e6, occ.len() as f64 / slots as f64 * 100.0);
    println!("  mean region bytes {:.0} KB", slots as f64 * 16.0 / 256.0 / 1024.0);

    println!("\n  phase (best of 7, ms)");
    let t = best(7, || { let t = Instant::now(); let v: Vec<u64> = packed.iter().map(|&a| finish(a)).collect(); let m = ms(t); std::hint::black_box(v); m });
    println!("    finish over 1M keys              {t:>7.2}");

    let t = best(7, || {
        let t = Instant::now();
        let mut counts = [0usize; 256];
        for &a in &packed { counts[shard_of(a)] += 1; }
        let m = ms(t); std::hint::black_box(counts); m
    });
    println!("    partition counting pass          {t:>7.2}");

    let t = best(7, || {
        let t = Instant::now();
        let mut counts = [0usize; 256];
        for &a in &packed { counts[shard_of(a)] += 1; }
        let mut sh: Vec<Vec<u64>> = counts.iter().map(|&c| Vec::with_capacity(c)).collect();
        for &a in &packed { sh[shard_of(a)].push(a); }
        let m = ms(t); std::hint::black_box(sh); m
    });
    println!("    full partition (count+alloc+scatter) {t:>3.2}");

    let t = best(7, || { let t = Instant::now(); let tb = CountTable::build_parallel(&packed, bits, 16); let m = ms(t); std::hint::black_box(tb); m });
    println!("    build_parallel(16)               {t:>7.2}");
    let tb = CountTable::build_parallel(&packed, bits, 16);
    let t = best(7, || { let t = Instant::now(); let s = tb.sum_parallel(&probe, 16); let m = ms(t); std::hint::black_box(s); m });
    println!("    sum_parallel(16)                 {t:>7.2}");
    let t = best(7, || { let t = Instant::now(); let tb = CountTable::build_parallel(&packed, bits, 16); let s = tb.sum_parallel(&probe, 16); let m = ms(t); std::hint::black_box(s); m });
    println!("    both                             {t:>7.2}");
}
