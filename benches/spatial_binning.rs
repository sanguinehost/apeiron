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

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "this is a benchmark, not library code, and each lint below is listed because this file \
              actually triggers it: sizes and indices narrowed to build keys, bounded by the loop that \
              produced them; counts converted to floats to report rates; a harness would need 2^53 samples \
              to lose a bit. Listed individually rather than suppressed wholesale, so a pedantic lint this \
              file does not yet trigger still surfaces"
)]

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use apeiron::coord::{i21, Packed};
use apeiron::table::CountTable;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const CELL: f32 = 0.25; // 4-unit cells

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

/// apeiron, one thread: a bijective address, an avalanched bucket, a partitioned table.
fn apeiron_serial(addrs: &[u64], probe: &[u64], bits: u32) -> i64 {
    let t = CountTable::build(addrs, bits);
    probe.iter().map(|&a| t.get(a)).sum()
}

/// apeiron, parallel. Shards hold disjoint keys, so workers own contiguous regions outright:
/// no merge, no synchronisation on the hot path, and the probe is grouped the same way the build
/// was -- a key is in one shard and visiting the others measures nothing.
fn apeiron_parallel(addrs: &[u64], probe: &[u64], bits: u32, threads: usize) -> i64 {
    let t = CountTable::build_parallel(addrs, bits, threads);
    t.sum_parallel(probe, threads)
}

const SHARDS: usize = 256;

/// A stream cut into shards the way `apeiron::table` cuts it: each worker partitions a contiguous
/// chunk into one buffer it owns, its shard runs laid end to end, and shard `s`'s stream is the
/// workers' runs in worker order.
///
/// Here so the reference arms pay what apeiron pays. apeiron's partition is threaded; leaving
/// this one serial would make the comparison a parallel arm against a serial one, which is the
/// same mistake these arms exist to stop making in the other direction.
struct Runs<K> {
    bufs: Vec<Vec<K>>,
    ends: Vec<Vec<usize>>,
}

impl<K: Copy + Default + Send + Sync> Runs<K> {
    fn of(xs: &[K], shard: &(impl Fn(&K) -> usize + Sync), threads: usize) -> Self {
        let workers = threads.max(1);
        if xs.is_empty() {
            return Self {
                bufs: Vec::new(),
                ends: Vec::new(),
            };
        }
        let chunk = xs.len().div_ceil(workers).max(1);
        let done: Vec<(Vec<K>, Vec<usize>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = xs
                .chunks(chunk)
                .map(|c| {
                    scope.spawn(move || {
                        let mut counts = vec![0usize; SHARDS];
                        for k in c {
                            counts[shard(k)] += 1;
                        }
                        let mut ends = vec![0usize; SHARDS + 1];
                        let mut cursor = vec![0usize; SHARDS];
                        let mut acc = 0usize;
                        for s in 0..SHARDS {
                            ends[s] = acc;
                            cursor[s] = acc;
                            acc += counts[s];
                        }
                        ends[SHARDS] = acc;
                        let mut buf = vec![K::default(); c.len()];
                        for k in c {
                            let s = shard(k);
                            buf[cursor[s]] = *k;
                            cursor[s] += 1;
                        }
                        (buf, ends)
                    })
                })
                .collect();
            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });
        let (bufs, ends) = done.into_iter().unzip();
        Self { bufs, ends }
    }

    fn runs(&self, s: usize) -> impl Iterator<Item = &[K]> {
        self.bufs
            .iter()
            .zip(self.ends.iter())
            .map(move |(b, e)| &b[e[s]..e[s + 1]])
    }
}

/// Any `HashMap` sharded and threaded exactly as apeiron's table is: keys partitioned in parallel
/// by the same shard function, one map per shard, workers owning disjoint shards, probes grouped
/// the same way. Without this the comparison is a parallel arm against a serial one, which
/// measures nothing.
fn sharded_parallel<S: std::hash::BuildHasher + Default + Send + Sync>(
    keys: &[(i32, i32, i32)],
    probe: &[(i32, i32, i32)],
    threads: usize,
) -> i64 {
    use apeiron::coord::Packed;
    let shard = |k: &(i32, i32, i32)| {
        apeiron::table::shard_of(
            apeiron::map::Salt::default(),
            (
                i21::new(k.0).unwrap(),
                i21::new(k.1).unwrap(),
                i21::new(k.2).unwrap(),
            )
                .packed(),
        )
    };
    let kb = Runs::of(keys, &shard, threads);
    let pb = Runs::of(probe, &shard, threads);
    let chunk = SHARDS.div_ceil(threads.max(1));
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let (kb, pb) = (&kb, &pb);
        for g in (0..SHARDS).step_by(chunk) {
            let hi = (g + chunk).min(SHARDS);
            handles.push(scope.spawn(move || {
                let mut acc = 0i64;
                for s in g..hi {
                    let mut m: HashMap<(i32, i32, i32), i64, S> = HashMap::default();
                    for run in kb.runs(s) {
                        for k in run {
                            *m.entry(*k).or_insert(0) += 1;
                        }
                    }
                    for run in pb.runs(s) {
                        for k in run {
                            acc += m.get(k).copied().unwrap_or(0);
                        }
                    }
                }
                acc
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
    })
}

/// The lossy tuple fold: three `i21`s packed into one word, which is the shape a spatial key
/// actually has. `hashtrinity` had an arm here until its dependency was dropped; its tuple fold
/// collided on signed coordinates, and `tests/coord_is_injective.rs` is what keeps this one from
/// doing the same.
fn apeiron_tuple(
    keys: &[(i32, i32, i32)],
    probe: &[(i32, i32, i32)],
    bits: u32,
    threads: usize,
) -> i64 {
    use apeiron::coord::Coord;
    let addrs: Vec<u64> = keys.iter().map(Coord::coord).collect();
    let probes: Vec<u64> = probe.iter().map(Coord::coord).collect();
    let t = CountTable::build_parallel(&addrs, bits, threads);
    t.sum_parallel(&probes, threads)
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
    (
        i21::new(k.0).unwrap(),
        i21::new(k.1).unwrap(),
        i21::new(k.2).unwrap(),
    )
        .packed()
}

#[expect(
    clippy::too_many_lines,
    reason = "one criterion group measures the whole spatial surface; the shared scene is the function"
)]
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
        report(
            "apeiron_serial",
            apeiron_serial(&packed, &probe_packed, bits),
        );
        report(
            "apeiron_parallel",
            apeiron_parallel(&packed, &probe_packed, bits, 16),
        );
        report(
            "hashbrown_foldhash",
            hashbrown_with::<foldhash::fast::RandomState>(&keys, &occupied),
        );
        report(
            "hashbrown_fxhash",
            hashbrown_with::<BuildHasherDefault<rustc_hash::FxHasher>>(&keys, &occupied),
        );
        report(
            "apeiron_tuple_par16",
            apeiron_tuple(&keys, &occupied, bits, 16),
        );
        report(
            "std_siphash_par16",
            sharded_parallel::<std::collections::hash_map::RandomState>(&keys, &occupied, 16),
        );
        report(
            "foldhash_par16",
            sharded_parallel::<foldhash::fast::RandomState>(&keys, &occupied, 16),
        );
        report(
            "fxhash_par16",
            sharded_parallel::<BuildHasherDefault<rustc_hash::FxHasher>>(&keys, &occupied, 16),
        );
        assert_eq!(
            apeiron_serial(&packed, &probe_packed, bits),
            truth,
            "apeiron must be exact"
        );
        assert_eq!(
            apeiron_parallel(&packed, &probe_packed, bits, 16),
            truth,
            "apeiron parallel must be exact"
        );

        let mut g = c.benchmark_group(format!("bin_and_probe/{n}"));
        g.sample_size(20);
        g.bench_with_input(BenchmarkId::new("apeiron_serial", n), &n, |b, _| {
            b.iter(|| black_box(apeiron_serial(&packed, &probe_packed, bits)));
        });
        g.bench_with_input(BenchmarkId::new("apeiron_par16", n), &n, |b, _| {
            b.iter(|| black_box(apeiron_parallel(&packed, &probe_packed, bits, 16)));
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
        g.bench_with_input(BenchmarkId::new("apeiron_tuple_par16", n), &n, |b, _| {
            b.iter(|| black_box(apeiron_tuple(&keys, &occupied, bits, 16)));
        });
        g.bench_with_input(BenchmarkId::new("std_siphash_par16", n), &n, |b, _| {
            b.iter(|| {
                black_box(sharded_parallel::<std::collections::hash_map::RandomState>(
                    &keys, &occupied, 16,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("foldhash_par16", n), &n, |b, _| {
            b.iter(|| {
                black_box(sharded_parallel::<foldhash::fast::RandomState>(
                    &keys, &occupied, 16,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("fxhash_par16", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    sharded_parallel::<BuildHasherDefault<rustc_hash::FxHasher>>(
                        &keys, &occupied, 16,
                    ),
                )
            });
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
