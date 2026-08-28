//! Spatial binning, head to head.
//!
//! The workload the whole exercise is about: entities at world positions, bucketed into cells,
//! counted per cell, then probed for "how many entities can this observer perceive". That is
//! broadphase, interest management and threat aggregation, which is where a game engine spends a
//! table.
//!
//! Six arms. The keys are **signed** and centred on the origin, because that is what a world
//! looks like and it is the case the folds disagree on. Every arm is checked to produce the same
//! answer outside the timed region.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use apeiron::coord::{bucket, i21, Packed};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hashtrinity::FreqMap;

const CELL: f32 = 0.25; // 4-unit cells

/// `FreqMap::count_par`'s home bucket. Mirrors its private `BBITS`, as its own benchmark does.
const HT_BBITS: u32 = 8;

#[inline]
fn ht_home(k: u64) -> usize {
    (k.wrapping_mul(hashtrinity::PHI) >> (64 - HT_BBITS)) as usize
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        // Centred on the origin: half the world is negative.
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 * 2000.0 - 1000.0
    }
}

/// A clustered scene, as entities actually sit: crowds, not a uniform spray.
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

/// apeiron: a bijective address, then an avalanched bucket. Open addressing, linear probe.
fn apeiron_table(keys: &[(i32, i32, i32)], probe: &[(i32, i32, i32)], bits: u32) -> i64 {
    let n = 1usize << bits;
    let mut addr = vec![u64::MAX; n];
    let mut count = vec![0i64; n];
    let narrow = |k: &(i32, i32, i32)| {
        (
            i21::new(k.0).unwrap(),
            i21::new(k.1).unwrap(),
            i21::new(k.2).unwrap(),
        )
    };
    for k in keys {
        let a = narrow(k).packed();
        let mut i = bucket(a, bits) as usize;
        loop {
            if addr[i] == u64::MAX {
                addr[i] = a;
                count[i] = 1;
                break;
            } else if addr[i] == a {
                count[i] += 1;
                break;
            }
            i = (i + 1) & (n - 1);
        }
    }
    let mut total = 0;
    for k in probe {
        let a = narrow(k).packed();
        let mut i = bucket(a, bits) as usize;
        loop {
            if addr[i] == u64::MAX {
                break;
            } else if addr[i] == a {
                total += count[i];
                break;
            }
            i = (i + 1) & (n - 1);
        }
    }
    total
}

/// hashtrinity's own path, its way: the packed key its benchmark uses, through `count_par`.
///
/// `count_par` returns one map per home bucket, holding disjoint keys, so a probe is the sum over
/// the maps -- that disjointness is the reason it needs no merge step and is the thing it wins on.
fn hashtrinity_packed(keys: &[u64], probe: &[u64]) -> i64 {
    let maps = FreqMap::<u64, i64>::count_par(keys);
    let mut acc = 0i64;
    if maps.len() == 1 {
        maps[0].get_each(probe, |_, v| acc += v.copied().unwrap_or(0));
        return acc;
    }
    // Grouped by home bucket and batched, the way its own benchmark does it. Probing every map
    // per key measures the caller's mistake and reads as a 40x loss that is not there.
    let mut by_bucket: Vec<Vec<u64>> = vec![Vec::new(); maps.len()];
    for &k in probe {
        by_bucket[ht_home(k)].push(k);
    }
    for (m, ks) in maps.iter().zip(by_bucket.iter()) {
        m.get_each(ks, |_, v| acc += v.copied().unwrap_or(0));
    }
    acc
}

/// hashtrinity's tuple fold, which is the path a caller reaches for without knowing better --
/// and the one whose address collides on signed coordinates.
fn hashtrinity_tuple(keys: &[(i32, i32, i32)], probe: &[(i32, i32, i32)]) -> i64 {
    use hashtrinity::Coord;
    let maps = FreqMap::<(i32, i32, i32), i64>::count_par(keys);
    let mut acc = 0i64;
    if maps.len() == 1 {
        maps[0].get_each(probe, |_, v| acc += v.copied().unwrap_or(0));
        return acc;
    }
    let mut by_bucket: Vec<Vec<(i32, i32, i32)>> = vec![Vec::new(); maps.len()];
    for &k in probe {
        by_bucket[ht_home(k.coord())].push(k);
    }
    for (m, ks) in maps.iter().zip(by_bucket.iter()) {
        m.get_each(ks, |_, v| acc += v.copied().unwrap_or(0));
    }
    acc
}

fn with_hasher<S: std::hash::BuildHasher + Default>(
    keys: &[(i32, i32, i32)],
    probe: &[(i32, i32, i32)],
) -> i64 {
    let mut m: HashMap<(i32, i32, i32), i64, S> = HashMap::default();
    for k in keys {
        *m.entry(*k).or_insert(0) += 1;
    }
    probe.iter().map(|p| m.get(p).copied().unwrap_or(0)).sum()
}

fn hashbrown_with<S: std::hash::BuildHasher + Default>(
    keys: &[(i32, i32, i32)],
    probe: &[(i32, i32, i32)],
) -> i64 {
    let mut m: hashbrown::HashMap<(i32, i32, i32), i64, S> = hashbrown::HashMap::default();
    for k in keys {
        *m.entry(*k).or_insert(0) += 1;
    }
    probe.iter().map(|p| m.get(p).copied().unwrap_or(0)).sum()
}

fn pack(k: (i32, i32, i32)) -> u64 {
    (i21::new(k.0).unwrap(), i21::new(k.1).unwrap(), i21::new(k.2).unwrap()).packed()
}

fn bench(c: &mut Criterion) {
    for &n in &[100_000usize, 1_000_000] {
        let keys = scene(n);
        let mut occupied: Vec<(i32, i32, i32)> = keys.clone();
        occupied.sort_unstable();
        occupied.dedup();
        let packed: Vec<u64> = keys.iter().map(|&k| pack(k)).collect();
        let probe_packed: Vec<u64> = occupied.iter().map(|&k| pack(k)).collect();
        let bits = (occupied.len() * 2).next_power_of_two().trailing_zeros();

        // Correctness before speed, outside the timed region. An arm that returns the wrong
        // count is not competing; it is reported and then benchmarked anyway, because "wrong and
        // fast" is exactly the trade this whole exercise exists to make visible.
        let truth = with_hasher::<std::collections::hash_map::RandomState>(&keys, &occupied);
        let report = |name: &str, got: i64| {
            let mark = if got == truth { "ok  " } else { "WRONG" };
            println!(
                "  correctness n={n:<9} {name:<20} {mark}  got {got:>12}  want {truth:>12}  \
                 off by {:>10}",
                (got - truth).abs()
            );
        };
        report("apeiron", apeiron_table(&keys, &occupied, bits));
        report("hashtrinity_packed", hashtrinity_packed(&packed, &probe_packed));
        report("hashtrinity_tuple", hashtrinity_tuple(&keys, &occupied));
        report(
            "hashbrown_foldhash",
            hashbrown_with::<foldhash::fast::RandomState>(&keys, &occupied),
        );
        report(
            "hashbrown_fxhash",
            hashbrown_with::<BuildHasherDefault<rustc_hash::FxHasher>>(&keys, &occupied),
        );
        assert_eq!(apeiron_table(&keys, &occupied, bits), truth, "apeiron must be exact");

        let mut g = c.benchmark_group(format!("bin_and_probe/{n}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("apeiron", n), &n, |b, _| {
            b.iter(|| black_box(apeiron_table(&keys, &occupied, bits)));
        });
        g.bench_with_input(BenchmarkId::new("hashtrinity_packed", n), &n, |b, _| {
            b.iter(|| black_box(hashtrinity_packed(&packed, &probe_packed)));
        });
        g.bench_with_input(BenchmarkId::new("hashtrinity_tuple", n), &n, |b, _| {
            b.iter(|| black_box(hashtrinity_tuple(&keys, &occupied)));
        });
        g.bench_with_input(BenchmarkId::new("std_siphash", n), &n, |b, _| {
            b.iter(|| {
                black_box(with_hasher::<std::collections::hash_map::RandomState>(
                    &keys, &occupied,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("hashbrown_foldhash", n), &n, |b, _| {
            b.iter(|| {
                black_box(hashbrown_with::<foldhash::fast::RandomState>(
                    &keys, &occupied,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("hashbrown_fxhash", n), &n, |b, _| {
            b.iter(|| {
                black_box(hashbrown_with::<BuildHasherDefault<rustc_hash::FxHasher>>(
                    &keys, &occupied,
                ))
            });
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
