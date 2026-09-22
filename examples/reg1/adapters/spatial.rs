//! REG-1 spatial/parallel incumbent adapters on the packed `71942db` scene.
//!
//! This module measures the parallel/spatial face that the design notes names as
//! load-bearing: `hashtrinity::FreqMap::count_par` + `get_each`, `hashbrown` + `rayon`, and the
//! 16-worker sharded `HashMap` arms with `foldhash` and `fxhash`. It reproduces the registered
//! `historical_spatial_parallel` workload exactly — the `71942db` `scene(n)` generator, the
//! `(i21,i21,i21)::packed()` packing and the build+probe timed region — so the incumbents are
//! measured on the same key population and the same timed region as the registered spatial cells.
//!
//! The selected implementation is the current executing `apeiron::Table` used as a counter (one
//! increment per entity, then a probe of the occupied cells). The independent oracle is a
//! `std::collections::HashMap` count folded over the probe set **in query order**; the selected
//! implementation and every adapter must reproduce that digest before timing is accepted. The
//! grouped readbacks (`count_par` returns one map per home bucket, the 16-worker arms own shards)
//! restore the original query order before the order-sensitive FNV fold; only a commutative
//! reduction could omit that step, and the digest here is not commutative.
//!
//! Below `count_par`'s `1 << 15` crossover the hashtrinity face falls back to one serial
//! aggregation. The reported [`WorkerPolicy`] records the actual pool width and the fallback flag
//! rather than presenting a serial run as parallel.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the REG-1 spatial adapter campaign and its outcome types"
)]

use std::collections::HashMap as StdHashMap;
use std::hash::BuildHasher;
use std::hint::black_box;
use std::time::Instant;

use apeiron::coord::{i21, Packed};
use apeiron::table::Home;

use crate::matrix::{cpu_model, maximum, median, minimum, ratio, Profile};
use crate::strata::{fold, nanos_since, StrataHome, StrataKey, StrataWitness, FNV_OFFSET, SHARDS};

/// The registered packed spatial sizes: `matrix.json` `historical_spatial_parallel` `n100000` and
/// `n1000000`.
pub(crate) const SPATIAL_SIZES: [usize; 2] = [100_000, 1_000_000];
/// The `71942db` scene seed and cell size.
const SCENE_SEED: u64 = 0x243F_6A88_85A3_08D3;
const CELL: f32 = 0.25;
/// `FreqMap::count_par`'s private home-bucket width, mirrored so a grouped probe lands in the
/// bucket the scatter assigned.
const HT_BBITS: u32 = 8;
/// The 16-worker bulk arms.
const PAR16_WORKERS: usize = 16;
/// The reduced spatial size the rejecting controls run, above `count_par`'s `1 << 15` crossover so
/// the grouped path is exercised.
const CONTROL_SIZE: usize = 40_000;
/// The selected `apeiron::Table` per-shard override width.
const APEIRON_N: usize = 16;

/// The selected table type on the packed spatial workload.
type SpatialTable = apeiron::Table<StrataKey, i64, StrataHome, APEIRON_N>;

/// The three rejecting controls.
pub(crate) const CONTROLS: [&str; 3] = [
    "spatial-wrong-packed-input",
    "spatial-wrong-bucket-routing",
    "spatial-wrong-worker-width",
];

/// Whether `name` is a spatial control.
pub(crate) fn is_control(name: &str) -> bool {
    CONTROLS.contains(&name)
}

/// The actual worker policy an arm ran under. Reported, never inferred: a serial fallback is named.
#[derive(Clone)]
pub(crate) struct WorkerPolicy {
    /// The worker count the arm actually used (`1` for a serial fallback or a single-core probe).
    pub(crate) workers: usize,
    /// Whether the arm took hashtrinity's documented serial fallback below its crossover.
    pub(crate) serial_fallback: bool,
    /// A human-readable statement of the mechanism.
    pub(crate) note: String,
}

/// One adapter's campaign on one spatial cell.
pub(crate) struct SpatialOutcome {
    /// The registered incumbent id.
    pub(crate) id: &'static str,
    /// The pooled per-sample nanoseconds across every campaign.
    pub(crate) samples: Vec<u64>,
    /// The selected samples measured at this incumbent's actual worker width.
    pub(crate) selected_samples: Vec<u64>,
    /// The per-campaign `selected/incumbent` ratios.
    pub(crate) campaign_ratios: Vec<f64>,
    /// Whether every sample's digest equalled the spatial oracle digest.
    pub(crate) digest_ok: bool,
    /// Whether the width-matched selected samples equalled the same oracle digest.
    pub(crate) selected_digest_ok: bool,
    /// The selected execution policy for this pair, measured from the width passed to its API.
    pub(crate) selected_policy: WorkerPolicy,
    /// The worker policy the arm actually ran under.
    pub(crate) policy: WorkerPolicy,
}

/// One spatial cell: the selected implementation and every adapter on one packed scene size.
pub(crate) struct SpatialCell {
    /// The stable cell identifier, e.g. `spatial/n100000`.
    pub(crate) cell_id: String,
    /// The packed scene size.
    pub(crate) population: usize,
    /// The probe count (the occupied cells).
    pub(crate) ops: usize,
    /// The independent count oracle digest folded in query order.
    pub(crate) oracle_digest: u64,
    /// Every adapter's outcome on this cell.
    pub(crate) outcomes: Vec<SpatialOutcome>,
}

/// The complete spatial enumeration.
pub(crate) struct SpatialWitness {
    /// One cell per registered spatial size.
    pub(crate) cells: Vec<SpatialCell>,
    /// Campaigns, warmup traces and samples per campaign (shared with the strata protocol).
    pub(crate) campaigns: u32,
    pub(crate) warmup: u32,
    pub(crate) samples: u32,
}

/// The adapter functions in report order: `(id, run)`.
const ARMS: [(&str, ArmFn); 5] = [
    ("hashtrinity_count_par", arm_count_par),
    ("hashtrinity_get_each", arm_get_each),
    ("hashbrown_rayon", arm_hashbrown_rayon),
    ("foldhash_par16", arm_foldhash_par16),
    ("fxhash_par16", arm_fxhash_par16),
];

/// One arm's run result.
struct ArmResult {
    digest: u64,
    nanos: u64,
    policy: WorkerPolicy,
}

/// The run signature shared by every adapter.
type ArmFn = fn(&[u64], &[u64], Corruption) -> ArmResult;

/// The in-memory corruption a control applies. `None` is the real run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Corruption {
    None,
    /// The adapters see a corrupted packed input while the oracle sees the real one.
    Input,
    /// The grouped hashtrinity probe routes by the wrong bucket function.
    Routing,
    /// The reported worker width is wrong.
    Width,
}

/// A deterministic xorshift scene generator, copied from `71942db:benches/spatial_binning.rs`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "the scene is the registered historical float workload; the cast is the workload"
    )]
    fn f(&mut self) -> f32 {
        // Centred on the origin: half the world is negative.
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 * 2000.0 - 1000.0
    }
}

/// A clustered scene, as entities actually sit: crowds, not a uniform spray. Reproduces the
/// registered `71942db` `scene(n)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the registered spatial workload is this exact float scene; the casts are its definition"
)]
fn scene(n: usize) -> Vec<(i32, i32, i32)> {
    let mut r = Rng(SCENE_SEED);
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

/// The registered packed key: three `i21` world cells packed into one word.
fn pack(k: (i32, i32, i32)) -> u64 {
    (
        i21::new(k.0).expect("scene coordinate fits i21"),
        i21::new(k.1).expect("scene coordinate fits i21"),
        i21::new(k.2).expect("scene coordinate fits i21"),
    )
        .packed()
}

/// The packed scene keys and the packed occupied-cell probe set.
fn spatial_input(n: usize) -> (Vec<u64>, Vec<u64>) {
    let keys = scene(n);
    let mut occupied = keys.clone();
    occupied.sort_unstable();
    occupied.dedup();
    (
        keys.iter().map(|&k| pack(k)).collect(),
        occupied.iter().map(|&k| pack(k)).collect(),
    )
}

/// `count_par`'s home bucket, mirroring its private `BBITS`.
fn ht_home(k: u64) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the shift leaves at most 8 bits, so the result fits usize"
    )]
    {
        (k.wrapping_mul(hashtrinity::PHI) >> (64 - HT_BBITS)) as usize
    }
}

/// The deliberately wrong home used by the routing control: the same shape with a wrong window.
fn ht_home_wrong(k: u64) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the shift leaves at most 7 bits, so the result fits usize"
    )]
    {
        (k.wrapping_mul(hashtrinity::PHI) >> (64 - (HT_BBITS - 1))) as usize
    }
}

/// The independent count oracle: count every packed key, then fold the probe counts in query order.
fn oracle_digest(keys: &[u64], probe: &[u64]) -> u64 {
    let mut counts: StdHashMap<u64, i64> = StdHashMap::with_capacity(keys.len());
    for &k in keys {
        *counts.entry(k).or_insert(0) += 1;
    }
    let mut hash = FNV_OFFSET;
    for &k in probe {
        fold(&mut hash, counts.get(&k).copied().unwrap_or(0));
    }
    hash
}

/// The selected implementation: the current executing `apeiron::Table` used as a counter. Builds
/// one increment per entity, probes the occupied cells and folds the counts in query order.
fn run_selected(keys: &[u64], probe: &[u64]) -> (u64, u64) {
    let merge: crate::strata::MergeFn = |new: i64, old: i64| new.wrapping_add(old);
    let mut table = SpatialTable::new(0, StrataHome);
    let mut hash = FNV_OFFSET;
    let start = Instant::now();
    for &k in keys {
        let _ = table.insert_with(&merge, StrataKey { tag: k }, 1);
    }
    for &k in probe {
        fold(&mut hash, *table.get(black_box(&StrataKey { tag: k })));
    }
    (hash, nanos_since(start))
}

/// The selected parallel face. The public APIs restore probe results to query order before this
/// order-sensitive fold, so its digest is directly comparable with the serial selected face.
fn run_selected_parallel(keys: &[u64], probe: &[u64], requested_workers: usize) -> ArmResult {
    let merge: crate::strata::MergeFn = |new: i64, old: i64| new.wrapping_add(old);
    let input: Vec<(StrataKey, i64)> = keys
        .iter()
        .copied()
        .map(|tag| (StrataKey { tag }, 1))
        .collect();
    let start = Instant::now();
    let table = SpatialTable::build_parallel(&merge, 0, &input, StrataHome, requested_workers)
        .expect("spatial selected parallel build must fit");
    let queries: Vec<StrataKey> = probe.iter().copied().map(|tag| StrataKey { tag }).collect();
    let values = table.get_each_parallel(&queries, requested_workers, |value| *value);
    let mut hash = FNV_OFFSET;
    for value in values {
        fold(&mut hash, value);
    }
    // Both public parallel APIs clamp the requested width to the input size. All registered and
    // control scenes exceed that size, so this is the task count those APIs can actually schedule.
    let workers = requested_workers.max(1).min(keys.len());
    ArmResult {
        digest: hash,
        nanos: nanos_since(start),
        policy: WorkerPolicy {
            workers,
            serial_fallback: false,
            note: format!(
                "apeiron::Table::build_parallel + get_each_parallel requested_workers={requested_workers}; \
                 actual_tasks=min(requested_workers, keys.len())={workers}"
            ),
        },
    }
}

/// Run the selected face that matches an observed incumbent policy. Width one retains the existing
/// serial implementation; every wider run uses the table's parallel build and parallel probe APIs.
fn run_selected_at_width(keys: &[u64], probe: &[u64], workers: usize) -> ArmResult {
    if workers <= 1 {
        let (digest, nanos) = run_selected(keys, probe);
        ArmResult {
            digest,
            nanos,
            policy: WorkerPolicy {
                workers: 1,
                serial_fallback: false,
                note: "apeiron serial insert_with + get; requested_workers=1".to_string(),
            },
        }
    } else {
        run_selected_parallel(keys, probe, workers)
    }
}

/// The hashtrinity count face: `count_par` partitions and counts, then a grouped `get_each` probes
/// one map per home bucket. The grouped readback restores the original query order before the
/// order-sensitive fold.
fn run_count_par(keys: &[u64], probe: &[u64], corruption: Corruption) -> ArmResult {
    let pool = hashtrinity::par::Pool::global();
    let width = pool.width();
    let start = Instant::now();
    let maps = hashtrinity::FreqMap::<u64, i64>::count_par(keys);
    let mut out = vec![0i64; probe.len()];
    if maps.len() == 1 {
        let mut i = 0usize;
        maps[0].get_each(probe, |_, v| {
            out[i] = v.copied().unwrap_or(0);
            i += 1;
        });
    } else {
        let route = |k: u64| {
            if corruption == Corruption::Routing {
                ht_home_wrong(k)
            } else {
                ht_home(k)
            }
        };
        let mut by_bucket: Vec<Vec<usize>> = vec![Vec::new(); maps.len()];
        for (i, &k) in probe.iter().enumerate() {
            by_bucket[route(k)].push(i);
        }
        for (m, idxs) in maps.iter().zip(by_bucket.iter()) {
            let ks: Vec<u64> = idxs.iter().map(|&i| probe[i]).collect();
            let mut j = 0usize;
            m.get_each(&ks, |_, v| {
                out[idxs[j]] = v.copied().unwrap_or(0);
                j += 1;
            });
        }
    }
    let mut hash = FNV_OFFSET;
    for &v in &out {
        fold(&mut hash, v);
    }
    let serial_fallback = width <= 1 || keys.len() < (1 << 15);
    ArmResult {
        digest: hash,
        nanos: nanos_since(start),
        policy: WorkerPolicy {
            workers: if serial_fallback { 1 } else { width },
            serial_fallback,
            note: format!(
                "hashtrinity par::Pool::global().width()={width}; count_par serial fallback below \
                 2^15={serial_fallback}"
            ),
        },
    }
}

/// `count_par` is the build/partition component of the same combined arm; the reported policy is
/// the build's pool.
fn arm_count_par(keys: &[u64], probe: &[u64], corruption: Corruption) -> ArmResult {
    run_count_par(keys, probe, corruption)
}

/// `get_each` is the probe component of the same combined arm; the reported policy is the
/// single-core pipelined probe.
fn arm_get_each(keys: &[u64], probe: &[u64], corruption: Corruption) -> ArmResult {
    let mut result = run_count_par(keys, probe, corruption);
    result.policy = WorkerPolicy {
        workers: 1,
        serial_fallback: false,
        note: "hashtrinity get_each single-core pipelined probe (P=16)".to_string(),
    };
    result
}

/// `hashbrown` + `rayon`: the parallel fold/reduce build, then a serial probe in query order.
fn arm_hashbrown_rayon(keys: &[u64], probe: &[u64], _corruption: Corruption) -> ArmResult {
    use rayon::prelude::*;
    let start = Instant::now();
    let map = keys
        .par_iter()
        .fold(hashbrown::HashMap::<u64, i64>::new, |mut acc, &k| {
            *acc.entry(k).or_insert(0) += 1;
            acc
        })
        .reduce(hashbrown::HashMap::<u64, i64>::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        });
    let mut hash = FNV_OFFSET;
    for &k in probe {
        fold(&mut hash, map.get(&k).copied().unwrap_or(0));
    }
    ArmResult {
        digest: hash,
        nanos: nanos_since(start),
        policy: WorkerPolicy {
            workers: rayon::current_num_threads(),
            serial_fallback: false,
            note: format!(
                "rayon global pool, current_num_threads()={}",
                rayon::current_num_threads()
            ),
        },
    }
}

/// A stream cut into shards the way the historical `sharded_parallel` arm cuts it: each worker
/// partitions a contiguous chunk into one buffer it owns, and shard `s`'s stream is the workers'
/// runs in worker order.
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

/// Any `HashMap` sharded and threaded exactly as the historical arm is: keys partitioned in
/// parallel by the apeiron home shard, one map per shard, workers owning disjoint shards, probes
/// grouped the same way. The grouped readback restores the original query order before the fold.
fn sharded_16<S: BuildHasher + Default + Send + Sync>(keys: &[u64], probe: &[u64]) -> u64 {
    let shard_of = |k: &u64| -> usize {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the residue is below 256, so the result fits usize"
        )]
        {
            (Home::home(&StrataHome, &StrataKey { tag: *k }) % SHARDS as u64) as usize
        }
    };
    let indexed: Vec<(usize, u64)> = probe.iter().copied().enumerate().collect();
    let shard_of_indexed = |p: &(usize, u64)| -> usize {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the residue is below 256, so the result fits usize"
        )]
        {
            (Home::home(&StrataHome, &StrataKey { tag: p.1 }) % SHARDS as u64) as usize
        }
    };
    let kb = Runs::of(keys, &shard_of, PAR16_WORKERS);
    let pb = Runs::of(&indexed, &shard_of_indexed, PAR16_WORKERS);
    let chunk = SHARDS.div_ceil(PAR16_WORKERS);
    let partials: Vec<Vec<(usize, i64)>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for g in (0..SHARDS).step_by(chunk) {
            let hi = (g + chunk).min(SHARDS);
            let (kb, pb) = (&kb, &pb);
            handles.push(scope.spawn(move || {
                let mut local: Vec<(usize, i64)> = Vec::new();
                for s in g..hi {
                    let mut m: hashbrown::HashMap<u64, i64, S> = hashbrown::HashMap::default();
                    for run in kb.runs(s) {
                        for k in run {
                            *m.entry(*k).or_insert(0) += 1;
                        }
                    }
                    for run in pb.runs(s) {
                        for &(idx, k) in run {
                            local.push((idx, m.get(&k).copied().unwrap_or(0)));
                        }
                    }
                }
                local
            }));
        }
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mut out = vec![0i64; probe.len()];
    for part in partials {
        for (i, c) in part {
            out[i] = c;
        }
    }
    let mut hash = FNV_OFFSET;
    for &v in &out {
        fold(&mut hash, v);
    }
    hash
}

/// The 16-worker `foldhash` bulk arm.
fn arm_foldhash_par16(keys: &[u64], probe: &[u64], _corruption: Corruption) -> ArmResult {
    let start = Instant::now();
    let digest = sharded_16::<foldhash::fast::RandomState>(keys, probe);
    ArmResult {
        digest,
        nanos: nanos_since(start),
        policy: WorkerPolicy {
            workers: PAR16_WORKERS,
            serial_fallback: false,
            note: format!("{PAR16_WORKERS} std::thread::scope workers, foldhash sharded"),
        },
    }
}

/// The 16-worker `fxhash` bulk arm.
fn arm_fxhash_par16(keys: &[u64], probe: &[u64], _corruption: Corruption) -> ArmResult {
    let start = Instant::now();
    let digest = sharded_16::<fxhash::FxBuildHasher>(keys, probe);
    ArmResult {
        digest,
        nanos: nanos_since(start),
        policy: WorkerPolicy {
            workers: PAR16_WORKERS,
            serial_fallback: false,
            note: format!("{PAR16_WORKERS} std::thread::scope workers, fxhash sharded"),
        },
    }
}

/// The input the control runs: the real packed keys, corrupted when the control asks.
fn corrupted_input(keys: &[u64], probe: &[u64], corruption: Corruption) -> (Vec<u64>, Vec<u64>) {
    if corruption == Corruption::Input {
        // Drop one packed entity: the selected and every adapter see a different count.
        let mut work = keys.to_vec();
        if !work.is_empty() {
            work.pop();
        }
        (work, probe.to_vec())
    } else {
        (keys.to_vec(), probe.to_vec())
    }
}

/// Run one spatial cell: warm up and sample each incumbent with a selected run at that incumbent's
/// reported width. Every selected and incumbent digest must equal the independent oracle.
#[allow(
    clippy::too_many_lines,
    reason = "one cell's warmup, sampling and per-arm digest verdict are one unit"
)]
fn run_cell(
    size: usize,
    campaigns: u32,
    warmup: u32,
    samples: u32,
    corruption: Corruption,
) -> SpatialCell {
    let (keys, probe) = spatial_input(size);
    let oracle = oracle_digest(&keys, &probe);
    let (work_keys, work_probe) = corrupted_input(&keys, &probe, corruption);

    let mut arm_samples: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
    let mut selected_samples: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
    let mut arm_ratios: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
    let mut arm_digest_ok: Vec<bool> = vec![true; ARMS.len()];
    let mut selected_digest_ok: Vec<bool> = vec![true; ARMS.len()];
    let mut arm_policy: Vec<Option<WorkerPolicy>> = vec![None; ARMS.len()];
    let mut selected_policy: Vec<Option<WorkerPolicy>> = vec![None; ARMS.len()];

    for _ in 0..campaigns {
        for _ in 0..warmup {
            for (_, run) in ARMS {
                let incumbent = run(&work_keys, &work_probe, corruption);
                let _ = run_selected_at_width(&work_keys, &work_probe, incumbent.policy.workers);
            }
        }
        let mut arm_campaign: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
        let mut selected_campaign: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
        for _ in 0..samples {
            for (i, (_, run)) in ARMS.iter().enumerate() {
                let result = run(&work_keys, &work_probe, corruption);
                arm_digest_ok[i] &= result.digest == oracle;
                let selected =
                    run_selected_at_width(&work_keys, &work_probe, result.policy.workers);
                selected_digest_ok[i] &= selected.digest == oracle;
                arm_policy[i] = Some(result.policy);
                selected_policy[i] = Some(selected.policy);
                arm_campaign[i].push(result.nanos);
                selected_campaign[i].push(selected.nanos);
            }
        }
        for i in 0..ARMS.len() {
            arm_ratios[i].push(ratio(
                median(&selected_campaign[i]),
                median(&arm_campaign[i]),
            ));
            arm_samples[i].extend_from_slice(&arm_campaign[i]);
            selected_samples[i].extend_from_slice(&selected_campaign[i]);
        }
    }

    let mut outcomes: Vec<SpatialOutcome> = Vec::with_capacity(ARMS.len());
    for (i, (id, _)) in ARMS.iter().enumerate() {
        let mut policy = arm_policy[i].clone().unwrap_or_else(|| WorkerPolicy {
            workers: 0,
            serial_fallback: false,
            note: "unmeasured".to_string(),
        });
        if corruption == Corruption::Width && *id == "foldhash_par16" {
            policy.workers = PAR16_WORKERS - 1;
        }
        outcomes.push(SpatialOutcome {
            id,
            samples: std::mem::take(&mut arm_samples[i]),
            selected_samples: std::mem::take(&mut selected_samples[i]),
            campaign_ratios: std::mem::take(&mut arm_ratios[i]),
            digest_ok: arm_digest_ok[i],
            selected_digest_ok: selected_digest_ok[i],
            selected_policy: selected_policy[i].clone().unwrap_or_else(|| WorkerPolicy {
                workers: 0,
                serial_fallback: false,
                note: "unmeasured".to_string(),
            }),
            policy,
        });
    }

    SpatialCell {
        cell_id: format!("spatial/n{size}"),
        population: keys.len(),
        ops: probe.len(),
        oracle_digest: oracle,
        outcomes,
    }
}

/// Run the registered packed spatial workload at the profile's campaign/sample counts.
pub(crate) fn measure(strata: &StrataWitness) -> SpatialWitness {
    let cells = SPATIAL_SIZES
        .iter()
        .map(|&size| {
            run_cell(
                size,
                strata.campaigns,
                strata.warmup,
                strata.samples,
                Corruption::None,
            )
        })
        .collect();
    SpatialWitness {
        cells,
        campaigns: strata.campaigns,
        warmup: strata.warmup,
        samples: strata.samples,
    }
}

/// Validate the spatial witness: the selected implementation and every adapter agree with the
/// oracle, every sample set is non-empty and non-zero, every ratio interval is finite, and every
/// reported worker policy matches the arm's actual mechanism.
///
/// # Errors
/// Returns a description of the first missing digest, empty/zero sample, non-finite ratio or wrong
/// worker policy. A caller that observes `Ok(())` has an oracle-checked spatial measurement.
pub(crate) fn validate(w: &SpatialWitness) -> Result<(), String> {
    if w.cells.is_empty() {
        return Err("spatial witness has no cells".to_string());
    }
    if w.campaigns == 0 || w.warmup == 0 || w.samples == 0 {
        return Err("spatial protocol has a zero count".to_string());
    }
    let expected =
        usize::try_from(w.campaigns).unwrap_or(0) * usize::try_from(w.samples).unwrap_or(0);
    for cell in &w.cells {
        if cell.population == 0 || cell.ops == 0 {
            return Err(format!(
                "spatial cell {} has a zero population/ops",
                cell.cell_id
            ));
        }
        if cell.outcomes.len() != ARMS.len() {
            return Err(format!(
                "spatial cell {} has {} adapters, expected {}",
                cell.cell_id,
                cell.outcomes.len(),
                ARMS.len()
            ));
        }
        for outcome in &cell.outcomes {
            if !outcome.digest_ok {
                return Err(format!(
                    "spatial cell {} adapter {} disagrees with the oracle",
                    cell.cell_id, outcome.id
                ));
            }
            if !outcome.selected_digest_ok {
                return Err(format!(
                    "spatial cell {} selected {} implementation disagrees with the oracle",
                    cell.cell_id, outcome.id
                ));
            }
            if outcome.selected_samples.len() != expected || outcome.selected_samples.contains(&0) {
                return Err(format!(
                    "spatial cell {} selected {} samples invalid ({} != {expected})",
                    cell.cell_id,
                    outcome.id,
                    outcome.selected_samples.len()
                ));
            }
            if outcome.samples.len() != expected || outcome.samples.contains(&0) {
                return Err(format!(
                    "spatial cell {} adapter {} samples invalid ({} != {expected})",
                    cell.cell_id,
                    outcome.id,
                    outcome.samples.len()
                ));
            }
            if outcome.campaign_ratios.len() != usize::try_from(w.campaigns).unwrap_or(0)
                || outcome.campaign_ratios.iter().any(|r| !r.is_finite())
            {
                return Err(format!(
                    "spatial cell {} adapter {} ratio interval invalid",
                    cell.cell_id, outcome.id
                ));
            }
            validate_policy(outcome.id, &outcome.policy)?;
            if outcome.selected_policy.workers != outcome.policy.workers {
                return Err(format!(
                    "spatial cell {} selected {} workers {} != incumbent workers {}",
                    cell.cell_id,
                    outcome.id,
                    outcome.selected_policy.workers,
                    outcome.policy.workers
                ));
            }
        }
    }
    Ok(())
}

/// Check one reported worker policy against the arm's actual mechanism. A wrong reported width
/// rejects: the policy is evidence, not a label.
fn validate_policy(id: &str, policy: &WorkerPolicy) -> Result<(), String> {
    let expected = match id {
        "foldhash_par16" | "fxhash_par16" => PAR16_WORKERS,
        "hashbrown_rayon" => rayon::current_num_threads(),
        "hashtrinity_get_each" => 1,
        "hashtrinity_count_par" => {
            let width = hashtrinity::par::Pool::global().width();
            if width <= 1 || policy.serial_fallback {
                1
            } else {
                width
            }
        }
        other => return Err(format!("unknown spatial arm {other}")),
    };
    if policy.workers != expected {
        return Err(format!(
            "spatial arm {id} reports {} workers, expected {expected}",
            policy.workers
        ));
    }
    Ok(())
}

/// The minimum and maximum of a value set, or `(NAN, NAN)` when empty.
fn interval(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// Print the spatial cells, their worker policies and their `selected/incumbent` ratios. Machine,
/// metric and interval are printed with the numbers; the worker policy is the actual mechanism.
pub(crate) fn print(w: &SpatialWitness) {
    println!(
        "spatial_protocol workload=\"71942db scene(n) packed u64, build+probe\" sizes={:?} \
         campaigns={} warmup={} samples={} metric=elapsed_ns_median cpu=\"{}\" \
         oracle=std::HashMap count folded in query order",
        SPATIAL_SIZES,
        w.campaigns,
        w.warmup,
        w.samples,
        cpu_model()
    );
    for cell in &w.cells {
        for outcome in &cell.outcomes {
            let sel_median = median(&outcome.selected_samples);
            let med = median(&outcome.samples);
            let (r_lo, r_hi) = interval(&outcome.campaign_ratios);
            println!(
                "spatial cell={} population={} ops={} oracle_digest={:016x} incumbent={} selected_median_ns={sel_median} \
                  incumbent_median_ns={med} incumbent_interval_ns=[{},{}] ratio={:.4} \
                  ratio_interval=[{r_lo:.4},{r_hi:.4}] selected_digest_ok={} digest_ok={} sel_workers={} \
                  inc_workers={} sample_basis={} policy=\"{}\"",
                cell.cell_id,
                cell.population,
                cell.ops,
                cell.oracle_digest,
                outcome.id,
                minimum(&outcome.samples),
                maximum(&outcome.samples),
                ratio(sel_median, med),
                outcome.selected_digest_ok,
                outcome.digest_ok,
                outcome.selected_policy.workers,
                outcome.policy.workers,
                if outcome.selected_policy.workers == 1 { "serial" } else { "parallel" },
                outcome.policy.note
            );
        }
    }
}

/// Build the control witness for `name`, applying exactly one corruption.
fn control_witness(name: &str) -> Result<SpatialWitness, String> {
    let corruption = match name {
        "spatial-wrong-packed-input" => Corruption::Input,
        "spatial-wrong-bucket-routing" => Corruption::Routing,
        "spatial-wrong-worker-width" => Corruption::Width,
        _ => return Err(format!("unknown spatial control {name}")),
    };
    Ok(SpatialWitness {
        cells: vec![run_cell(CONTROL_SIZE, 1, 1, 2, corruption)],
        campaigns: 1,
        warmup: 1,
        samples: 2,
    })
}

/// The control child: build the corrupted spatial witness and require the same validator to reject
/// it. Exits 2 with a `rejected` marker on success, 1 on a wrong verdict.
pub(crate) fn run_control_child(name: &str, _profile: Profile) -> i32 {
    let witness = match control_witness(name) {
        Ok(witness) => witness,
        Err(e) => {
            eprintln!("performance-control {name} NOT rejected: {e}");
            return 1;
        }
    };
    match validate(&witness) {
        Ok(()) => {
            eprintln!("performance-control {name} NOT rejected: validator accepted corruption");
            1
        }
        Err(e) => {
            eprintln!("performance-control {name} rejected: {e}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{StrataHome, StrataKey};
    use apeiron::coord::address::{FaceHome, H3_40};
    use apeiron::coord::spread;
    use apeiron::table::Home;
    use apeiron::Sigma;

    /// Audit: the spatial adapter routes by the physical `Home` placement, and the semantic face
    /// map is a separate, face-tagged address.
    ///
    /// `Home::address` ignores `Sigma` and returns the placement-only `TableAddress`;
    /// `FaceHome::address` reports the face and its exact/specified value. The routing call sites
    /// in `sharded_16` use `Home::home`, so they never read the semantic map.
    #[test]
    fn spatial_home_physical_and_semantic_are_separate() {
        let key = StrataKey { tag: 7 };
        let profile = H3_40::new().expect("the frozen H3_40 token validates");

        // The physical placement is the same at every dial and carries no face certificate.
        let physical = Home::address(&StrataHome, Sigma::Elliptic, &key);
        assert_eq!(physical.place, spread(key.tag).place);
        assert_eq!(
            physical,
            Home::address(&StrataHome, Sigma::Hyperbolic, &key)
        );

        // The semantic map is face-tagged; elliptic is the exact word, not the placement.
        let semantic = FaceHome::address(&StrataHome, profile, Sigma::Elliptic, &key);
        assert_eq!(semantic.face(), Sigma::Elliptic);
        assert_eq!(semantic.value().elliptic_word(), Some(key.tag));
        assert_ne!(semantic.value().elliptic_word(), Some(physical.place));
    }
}
