// Part 2: the count
// face instrument. Measures `apeiron::freq::FreqMap<u64, i64>` (the parallel bucketed group-by
// `count_par`, and the sequential `add` loop) against `hashbrown`+FxHasher and `hashbrown`+XXH3.
// Two columns: `count` (build the counts over n keys with d distinct) and `lookup` (read back every
// distinct key's count). Correctness oracle first (exit 2 on any arm mismatch), then interleaved
// rounds rotating the arm order, mean + min-max per cell, `--check`/`--check-seq` acceptance.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::format_push_string,
    clippy::map_unwrap_or
)]

use std::hash::{BuildHasher, Hash, Hasher};
use std::hint::black_box;
use std::time::{Duration, Instant};

use apeiron::freq::FreqMap;
use hashbrown::HashMap;
use rustc_hash::{FxBuildHasher, FxHasher};
use sha2::Digest;

/// The parallel count face: `count_par` on the FB/NB group-probe engine.
type ApFreq = FreqMap<u64, i64>;
/// The `FxHash` comparator arm (no entry API on the apeiron side; `add` is the count analogue).
type HbFx = HashMap<u64, i64, FxBuildHasher>;
/// The `XxHash3` comparator arm.
type HbXxh3 = HashMap<u64, i64, Xxh3BuildHasher>;

#[derive(Clone, Copy, Default)]
pub struct Xxh3BuildHasher;

#[derive(Default)]
pub struct Xxh3Hasher {
    bytes: [u8; 8],
    len: usize,
    overflow: Vec<u8>,
}

impl core::hash::Hasher for Xxh3Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        if self.overflow.is_empty() {
            twox_hash::XxHash3_64::oneshot(&self.bytes[..self.len])
        } else {
            twox_hash::XxHash3_64::oneshot(&self.overflow)
        }
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        if self.overflow.is_empty() && self.len + bytes.len() <= 8 {
            self.bytes[self.len..self.len + bytes.len()].copy_from_slice(bytes);
            self.len += bytes.len();
        } else {
            if self.overflow.is_empty() {
                self.overflow.extend_from_slice(&self.bytes[..self.len]);
            }
            self.overflow.extend_from_slice(bytes);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        if self.overflow.is_empty() && self.len == 0 {
            self.bytes = i.to_ne_bytes();
            self.len = 8;
        } else {
            self.write(&i.to_ne_bytes());
        }
    }
}

impl core::hash::BuildHasher for Xxh3BuildHasher {
    type Hasher = Xxh3Hasher;
    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        Xxh3Hasher::default()
    }
}

// ── The measured arms ────────────────────────────────────────────────────────────────────────

/// The five arms in the base order the per-round rotation starts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    /// `FreqMap::<u64, i64>::count_par(&keys)` — the parallel bucketed group-by.
    ApPar,
    /// A sequential `FreqMap::add` loop.
    ApSeq,
    /// `hashbrown` + `FxHasher`, `*entry(k).or_insert(0) += 1`.
    HbFx,
    /// `hashbrown` + `XXH3`, `*entry(k).or_insert(0) += 1`.
    HbXxh3,
    /// Rayon-free parallel baseline: Fx-hash-shard into per-thread `hashbrown`s, then merge.
    HbFxPar,
}

const AP_PAR: usize = 0;
const AP_SEQ: usize = 1;
const HB_FX: usize = 2;
const HB_XXH3: usize = 3;
const HB_FX_PAR: usize = 4;

impl Arm {
    const ALL: [Self; 5] = [
        Self::ApPar,
        Self::ApSeq,
        Self::HbFx,
        Self::HbXxh3,
        Self::HbFxPar,
    ];

    fn index(self) -> usize {
        match self {
            Self::ApPar => AP_PAR,
            Self::ApSeq => AP_SEQ,
            Self::HbFx => HB_FX,
            Self::HbXxh3 => HB_XXH3,
            Self::HbFxPar => HB_FX_PAR,
        }
    }
}

const ARM_NAMES: [&str; 5] = ["ap_par", "ap_seq", "hb_fx", "hb_xxh3", "hb_fx_par"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Column {
    Count,
    Lookup,
}

impl Column {
    const ALL: [Self; 2] = [Self::Count, Self::Lookup];

    fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Lookup => "lookup",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Distribution {
    Dense,
    Random,
    HighBits,
}

impl Distribution {
    const ALL: [Self; 3] = [Self::Dense, Self::Random, Self::HighBits];

    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Random => "random",
            Self::HighBits => "high-bits",
        }
    }
}

/// The distinct-key ratio: how many of the `n` keys are distinct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Distinct {
    All,
    Ten,
    One,
}

impl Distinct {
    const ALL: [Self; 3] = [Self::All, Self::Ten, Self::One];

    fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Ten => "10%",
            Self::One => "1%",
        }
    }

    fn count(self, n: usize) -> usize {
        match self {
            Self::All => n,
            Self::Ten => (n / 10).max(1),
            Self::One => (n / 100).max(1),
        }
    }
}

// ── Key generation ───────────────────────────────────────────────────────────────────────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// `n` keys over a pool of exactly `d` distinct values (distribution-shaped), shuffled so the
/// access order is not the generation order. Returns `(keys, pool)`; every pool value appears.
fn generate_keys(dist: Distribution, n: usize, d: usize) -> (Vec<u64>, Vec<u64>) {
    let d = d.clamp(1, n);
    let mut pool = Vec::with_capacity(d);
    match dist {
        Distribution::Dense => {
            for i in 0..d {
                pool.push(i as u64);
            }
        }
        Distribution::HighBits => {
            for i in 0..d {
                pool.push((i as u64) << 32);
            }
        }
        Distribution::Random => {
            // SplitMix64's finalizer is a bijection, so distinct states give distinct outputs.
            let mut rng = SplitMix64::new(0x1234_5678_9abc_def0);
            for _ in 0..d {
                pool.push(rng.next_u64() & !(1u64 << 63));
            }
        }
    }
    let mut keys: Vec<u64> = (0..n).map(|i| pool[i % d]).collect();
    let mut rng = SplitMix64::new(0xdead_beef_cafe_f00d);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        keys.swap(i, j);
    }
    (keys, pool)
}

// ── The arm builders ─────────────────────────────────────────────────────────────────────────

fn ap_seq_build(keys: &[u64]) -> ApFreq {
    let mut m = ApFreq::with_capacity(keys.len());
    for &k in keys {
        m.add(&k, 1);
    }
    m
}

fn hb_fx_build(keys: &[u64]) -> HbFx {
    let mut m = HbFx::with_capacity_and_hasher(keys.len(), FxBuildHasher);
    for &k in keys {
        *m.entry(k).or_insert(0) += 1;
    }
    m
}

fn hb_xxh3_build(keys: &[u64]) -> HbXxh3 {
    let mut m = HbXxh3::with_capacity_and_hasher(keys.len(), Xxh3BuildHasher);
    for &k in keys {
        *m.entry(k).or_insert(0) += 1;
    }
    m
}

fn hb_threads() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(32)
}

/// Rayon-free parallel baseline: Fx-hash-shard every key to one of `t` threads, count the shard
/// into a presized `hashbrown`, then merge the per-thread maps into one. Time `O(n)` plus the
/// merge; space `O(n)` for the shards and `O(distinct)` for the maps.
fn hb_fx_par_build(keys: &[u64]) -> HbFx {
    let t = hb_threads();
    if t <= 1 || keys.len() < (1usize << 15) {
        return hb_fx_build(keys);
    }
    let mut shards: Vec<Vec<u64>> = (0..t)
        .map(|_| Vec::with_capacity(keys.len() / t + 1))
        .collect();
    for &k in keys {
        let mut h = FxHasher::default();
        k.hash(&mut h);
        shards[(h.finish() as usize) % t].push(k);
    }
    let parts: Vec<HbFx> = std::thread::scope(|s| {
        let handles: Vec<_> = shards
            .into_iter()
            .map(|shard| {
                s.spawn(move || {
                    let mut m = HbFx::with_capacity_and_hasher(shard.len(), FxBuildHasher);
                    for k in shard {
                        *m.entry(k).or_insert(0) += 1;
                    }
                    m
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("hb_fx_par shard panicked"))
            .collect()
    });
    let mut total = HbFx::with_capacity_and_hasher(keys.len(), FxBuildHasher);
    for part in parts {
        for (k, v) in part {
            *total.entry(k).or_insert(0) += v;
        }
    }
    total
}

// ── Reading a count back from an arm ──────────────────────────────────────────────────────────

const SAMPLE: usize = 64;

trait CountRead {
    fn count_of(&self, k: u64) -> i64;
}

impl CountRead for ApFreq {
    fn count_of(&self, k: u64) -> i64 {
        self.get(&k).copied().unwrap_or(0)
    }
}

impl<S: BuildHasher> CountRead for HashMap<u64, i64, S> {
    fn count_of(&self, k: u64) -> i64 {
        self.get(&k).copied().unwrap_or(0)
    }
}

/// The O(1) fused read of `count_par`'s result: `FreqMap::count_get` forms `p = k·PHI` once and
/// feeds the bucket window `[40,48)` and the probe's home/tag windows from the same product (they
/// are disjoint bits of `p`). Buckets hold disjoint keys, so a key's count lives in exactly its
/// home-bucket shard. The serial fallback below the crossover returns one map (`maps.len() == 1`),
/// which has no bucket to select and is read directly.
fn ap_par_count(maps: &[ApFreq], k: u64) -> i64 {
    if maps.len() == 1 {
        return maps[0].get(&k).copied().unwrap_or(0);
    }
    ApFreq::count_get(maps, &k).copied().unwrap_or(0)
}

/// The literal collapse `∑ maps[b].get(k)` — the oracle's independent read of the same result.
fn ap_par_collapse(maps: &[ApFreq], k: u64) -> i64 {
    let mut c = 0i64;
    for m in maps {
        c = c.wrapping_add(m.get(&k).copied().unwrap_or(0));
    }
    c
}

// ── The correctness oracle ───────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq, Clone)]
struct CountObs {
    total: i64,
    distinct: usize,
    sample: Vec<(u64, i64)>,
}

fn obs_single<M: CountRead>(m: &M, pool: &[u64]) -> CountObs {
    let mut total = 0i64;
    let mut distinct = 0usize;
    let mut sample = Vec::with_capacity(SAMPLE.min(pool.len()));
    for (i, &k) in pool.iter().enumerate() {
        let c = m.count_of(k);
        total = total.wrapping_add(c);
        if c > 0 {
            distinct += 1;
        }
        if i < SAMPLE {
            sample.push((k, c));
        }
    }
    CountObs {
        total,
        distinct,
        sample,
    }
}

fn obs_ap_par(maps: &[ApFreq], pool: &[u64]) -> CountObs {
    let mut total = 0i64;
    let mut distinct = 0usize;
    let mut sample = Vec::with_capacity(SAMPLE.min(pool.len()));
    for (i, &k) in pool.iter().enumerate() {
        let c = ap_par_collapse(maps, k);
        if maps.len() > 1 {
            let home = ap_par_count(maps, k);
            if home != c {
                eprintln!(
                    "COUNT ROUTE MISMATCH: home-bucket read {home} != shard-sum {c} for key {k}"
                );
                std::process::exit(2);
            }
        }
        total = total.wrapping_add(c);
        if c > 0 {
            distinct += 1;
        }
        if i < SAMPLE {
            sample.push((k, c));
        }
    }
    CountObs {
        total,
        distinct,
        sample,
    }
}

fn oracle_arm(arm: Arm, keys: &[u64], pool: &[u64]) -> CountObs {
    match arm {
        Arm::ApPar => obs_ap_par(&ApFreq::count_par(keys), pool),
        Arm::ApSeq => obs_single(&ap_seq_build(keys), pool),
        Arm::HbFx => obs_single(&hb_fx_build(keys), pool),
        Arm::HbXxh3 => obs_single(&hb_xxh3_build(keys), pool),
        Arm::HbFxPar => obs_single(&hb_fx_par_build(keys), pool),
    }
}

// ── Timing ───────────────────────────────────────────────────────────────────────────────────

fn time_lookup<M: CountRead>(m: &M, pool: &[u64]) -> Duration {
    let start = Instant::now();
    let mut acc = 0i64;
    for &k in pool {
        acc = acc.wrapping_add(m.count_of(k));
    }
    let dur = start.elapsed();
    black_box(acc);
    dur
}

/// Build the arm's count, timing the build (`count`) and then reading back every distinct key's
/// count (`lookup`). Returns `(count_duration, lookup_duration)`.
fn run_arm(arm: Arm, keys: &[u64], pool: &[u64]) -> (Duration, Duration) {
    match arm {
        Arm::ApPar => {
            let start = Instant::now();
            let maps = ApFreq::count_par(keys);
            let count_d = start.elapsed();
            let start = Instant::now();
            let mut acc = 0i64;
            // Full-bucket shape: `maps.len() == COUNT_BUCKETS` is in the type, so the masked bucket
            // index needs no per-lookup bounds check (the length test is once per batch). The
            // sub-crossover serial shape (`maps.len() == 1`) reads the single map.
            if let Ok(arr) = <&[ApFreq; ApFreq::COUNT_BUCKETS]>::try_from(maps.as_slice()) {
                for &k in pool {
                    acc = acc.wrapping_add(ApFreq::count_get_arr(arr, &k).copied().unwrap_or(0));
                }
            } else {
                for &k in pool {
                    acc = acc.wrapping_add(maps[0].get(&k).copied().unwrap_or(0));
                }
            }
            let lookup_d = start.elapsed();
            black_box(acc);
            (count_d, lookup_d)
        }
        Arm::ApSeq => {
            let start = Instant::now();
            let m = ap_seq_build(keys);
            let count_d = start.elapsed();
            (count_d, time_lookup(&m, pool))
        }
        Arm::HbFx => {
            let start = Instant::now();
            let m = hb_fx_build(keys);
            let count_d = start.elapsed();
            (count_d, time_lookup(&m, pool))
        }
        Arm::HbXxh3 => {
            let start = Instant::now();
            let m = hb_xxh3_build(keys);
            let count_d = start.elapsed();
            (count_d, time_lookup(&m, pool))
        }
        Arm::HbFxPar => {
            let start = Instant::now();
            let m = hb_fx_par_build(keys);
            let count_d = start.elapsed();
            (count_d, time_lookup(&m, pool))
        }
    }
}

// ── Statistics and reporting ─────────────────────────────────────────────────────────────────

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / (values.len() as f64)
}

/// `[mean, min, max]` of the per-round ratios `numer[i] / denom[i]`.
fn ratio_stats(numer: &[f64], denom: &[f64]) -> [f64; 3] {
    let ratios: Vec<f64> = numer
        .iter()
        .zip(denom.iter())
        .map(|(a, b)| *a / *b)
        .collect();
    let m = mean(&ratios);
    let lo = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    [m, lo, hi]
}

const RATIO_NAMES: [&str; 4] = [
    "ap_par/hb_fx",
    "ap_par/hb_xxh3",
    "ap_seq/hb_fx",
    "ap_seq/hb_xxh3",
];

struct CellResult {
    column: &'static str,
    n: usize,
    distinct: &'static str,
    distribution: &'static str,
    ns: [f64; 5],
    /// `[mean, min, max]` for each of [`RATIO_NAMES`].
    ratios: [[f64; 3]; 4],
}

fn compute_file_sha256(path: &str) -> String {
    if let Ok(bytes) = std::fs::read(path) {
        let mut hasher = sha2::Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    } else {
        "file-read-error".to_string()
    }
}

fn read_cpu_model() -> String {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if line.starts_with("model name") {
                if let Some((_, model)) = line.split_once(':') {
                    return model.trim().to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn get_header_line() -> String {
    let rev = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    let sha_freq = compute_file_sha256("src/hashtrinity/freq.rs");
    let sha_par = compute_file_sha256("src/hashtrinity/par.rs");
    let rustc = std::process::Command::new("rustc")
        .arg("-V")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let cpu = read_cpu_model();
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    format!(
        "git: {} dirty: {} sha256(src/hashtrinity/freq.rs): {} sha256(src/hashtrinity/par.rs): {} rustc: {} cpu: {} target_dir: {}",
        rev, dirty, sha_freq, sha_par, rustc, cpu, target_dir
    )
}

fn get_face_line() -> String {
    format!(
        "faces: ap_par={} (count_par, bucketed group-by) | ap_seq={} (add loop) | hb_fx={} | hb_xxh3={} | hb_fx_par={} (thread::scope shards, merged)",
        std::any::type_name::<ApFreq>(),
        std::any::type_name::<ApFreq>(),
        std::any::type_name::<HbFx>(),
        std::any::type_name::<HbXxh3>(),
        std::any::type_name::<HbFx>(),
    )
}

fn main() {
    let mut check = false;
    let mut check_seq = false;
    let mut quick = false;
    let mut json_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--check-seq" => check_seq = true,
            "--quick" => quick = true,
            "--json" => {
                if let Some(path) = args.next() {
                    json_path = Some(path);
                }
            }
            _ => {}
        }
    }

    let rounds = if quick { 3 } else { 7 };
    let sizes: Vec<usize> = if quick {
        vec![1_000, 100_000]
    } else {
        vec![100_000, 1_000_000, 10_000_000]
    };

    let mut results: Vec<CellResult> = Vec::new();

    for &n in &sizes {
        for &distinct in &Distinct::ALL {
            for &dist in &Distribution::ALL {
                let d = distinct.count(n);
                let (keys, pool) = generate_keys(dist, n, d);

                // Correctness oracle: every arm's per-key counts must agree (total, distinct count
                // and a sampled set), before any timing is used. A mismatch prints the cell and
                // exits 2.
                let obs: Vec<CountObs> = Arm::ALL
                    .iter()
                    .map(|&arm| oracle_arm(arm, &keys, &pool))
                    .collect();
                for (i, o) in obs.iter().enumerate().skip(1) {
                    if *o != obs[0] {
                        eprintln!(
                            "CORRECTNESS MISMATCH in cell [n: {}, distinct: {}, distribution: {}]:\n  {}: {:?}\n  {}: {:?}",
                            n,
                            distinct.name(),
                            dist.name(),
                            ARM_NAMES[0],
                            obs[0],
                            ARM_NAMES[i],
                            o
                        );
                        std::process::exit(2);
                    }
                }

                // Protocol: `rounds` per cell; in each round run all five arms once each, rotating
                // their order so no arm holds a fixed position. Each `run_arm` yields the build
                // (`count`) and the read-back (`lookup`) durations from one build. ns are per
                // operation: `count` per input key, `lookup` per distinct key.
                let mut count_ns: [Vec<f64>; 5] =
                    std::array::from_fn(|_| Vec::with_capacity(rounds));
                let mut lookup_ns: [Vec<f64>; 5] =
                    std::array::from_fn(|_| Vec::with_capacity(rounds));

                for r in 0..rounds {
                    let rotated: [Arm; 5] =
                        std::array::from_fn(|i| Arm::ALL[(i + r) % Arm::ALL.len()]);
                    for arm in rotated {
                        let (count_d, lookup_d) = run_arm(arm, &keys, &pool);
                        count_ns[arm.index()].push(count_d.as_nanos() as f64 / n as f64);
                        lookup_ns[arm.index()].push(lookup_d.as_nanos() as f64 / d as f64);
                    }
                }

                for &column in &Column::ALL {
                    let series: &[Vec<f64>; 5] = match column {
                        Column::Count => &count_ns,
                        Column::Lookup => &lookup_ns,
                    };
                    let mut ns = [0f64; 5];
                    for (slot, s) in ns.iter_mut().zip(series) {
                        *slot = mean(s);
                    }
                    let ratios = [
                        ratio_stats(&series[AP_PAR], &series[HB_FX]),
                        ratio_stats(&series[AP_PAR], &series[HB_XXH3]),
                        ratio_stats(&series[AP_SEQ], &series[HB_FX]),
                        ratio_stats(&series[AP_SEQ], &series[HB_XXH3]),
                    ];
                    results.push(CellResult {
                        column: column.name(),
                        n,
                        distinct: distinct.name(),
                        distribution: dist.name(),
                        ns,
                        ratios,
                    });
                }
            }
        }
    }

    println!("{}", get_header_line());
    println!("{}", get_face_line());
    println!();

    println!(
        "| {:<6} | {:>9} | {:<4} | {:<9} | {:>11} | {:>11} | {:>11} | {:>11} | {:>11} | {:>22} | {:>22} | {:>22} | {:>22} |",
        "column",
        "n",
        "dist%",
        "dist",
        "ap_par ns",
        "ap_seq ns",
        "hb_fx ns",
        "hb_xxh3 ns",
        "hb_fx_par ns",
        "ap_par/hb_fx [min,max]",
        "ap_par/hb_xxh3 [min,max]",
        "ap_seq/hb_fx [min,max]",
        "ap_seq/hb_xxh3 [min,max]"
    );
    println!(
        "|{:-<8}|{:-<11}|{:-<6}|{:-<11}|{:-<13}|{:-<13}|{:-<13}|{:-<13}|{:-<13}|{:-<24}|{:-<24}|{:-<24}|{:-<24}|",
        "", "", "", "", "", "", "", "", "", "", "", "", ""
    );

    for cell in &results {
        let mut line = String::new();
        line.push_str(&format!(
            "| {:<6} | {:>9} | {:<4} | {:<9} |",
            cell.column, cell.n, cell.distinct, cell.distribution
        ));
        for value in cell.ns {
            line.push_str(&format!(" {:>11.3} |", value));
        }
        for [m, lo, hi] in cell.ratios {
            line.push_str(&format!(" {:>7.3} [{:>6.3}, {:>6.3}] |", m, lo, hi));
        }
        println!("{}", line);
    }

    if let Some(path) = &json_path {
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut json_lines = String::new();
        for cell in &results {
            json_lines.push_str(&format!(
                "{{\"column\":\"{}\",\"n\":{},\"distinct\":\"{}\",\"distribution\":\"{}\",\"ap_par_ns\":{:.3},\"ap_seq_ns\":{:.3},\"hb_fx_ns\":{:.3},\"hb_xxh3_ns\":{:.3},\"hb_fx_par_ns\":{:.3},\"ap_par_hb_fx\":{:.4},\"ap_par_hb_fx_min\":{:.4},\"ap_par_hb_fx_max\":{:.4},\"ap_par_hb_xxh3\":{:.4},\"ap_par_hb_xxh3_min\":{:.4},\"ap_par_hb_xxh3_max\":{:.4},\"ap_seq_hb_fx\":{:.4},\"ap_seq_hb_fx_min\":{:.4},\"ap_seq_hb_fx_max\":{:.4},\"ap_seq_hb_xxh3\":{:.4},\"ap_seq_hb_xxh3_min\":{:.4},\"ap_seq_hb_xxh3_max\":{:.4}}}\n",
                cell.column,
                cell.n,
                cell.distinct,
                cell.distribution,
                cell.ns[AP_PAR],
                cell.ns[AP_SEQ],
                cell.ns[HB_FX],
                cell.ns[HB_XXH3],
                cell.ns[HB_FX_PAR],
                cell.ratios[0][0],
                cell.ratios[0][1],
                cell.ratios[0][2],
                cell.ratios[1][0],
                cell.ratios[1][1],
                cell.ratios[1][2],
                cell.ratios[2][0],
                cell.ratios[2][1],
                cell.ratios[2][2],
                cell.ratios[3][0],
                cell.ratios[3][1],
                cell.ratios[3][2],
            ));
        }
        if let Err(e) = std::fs::write(path, json_lines) {
            eprintln!("Failed to write JSON lines to {}: {}", path, e);
        }
    }

    if check || check_seq {
        let mut failures: Vec<String> = Vec::new();
        for cell in &results {
            // `--check` reads the parallel `ap_par` face; `--check-seq` also reads the sequential
            // `ap_seq` face. A ratio above 1.0 against either hashbrown comparator fails the cell.
            let mut gated: Vec<(usize, &str)> = vec![(0, "ap_par"), (1, "ap_par")];
            if check_seq {
                gated.push((2, "ap_seq"));
                gated.push((3, "ap_seq"));
            }
            for (idx, name) in gated {
                let ratio = cell.ratios[idx];
                if ratio[0] > 1.0 {
                    failures.push(format!(
                        "FAIL({}): {} n={} distinct={} dist={}: {} = {:.3} > 1.0",
                        name,
                        cell.column,
                        cell.n,
                        cell.distinct,
                        cell.distribution,
                        RATIO_NAMES[idx],
                        ratio[0]
                    ));
                }
            }
        }
        if !failures.is_empty() {
            println!();
            for f in &failures {
                println!("{}", f);
            }
            eprintln!(
                "\nCheck failed: {} cell-ratio threshold(s) exceeded:",
                failures.len()
            );
            for f in &failures {
                eprintln!("  {}", f);
            }
            std::process::exit(1);
        }
    }
}
