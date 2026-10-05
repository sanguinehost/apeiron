// Row: `apeiron-bench-restore-EaEUYoDjUS4nFW9F4W7iUP` (APEIRON-BENCH-RESTORE): the L3 instrument.
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

use std::hint::black_box;
use std::time::Instant;

use apeiron::freq::Table as ApeironTable;
use hashbrown::HashMap;
use rustc_hash::FxBuildHasher;
use sha2::Digest;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Column {
    PresentHit,
    AbsentMiss,
    InsertFresh,
    Build,
    Update,
    RemovePresent,
}

impl Column {
    const ALL: [Self; 6] = [
        Self::PresentHit,
        Self::AbsentMiss,
        Self::InsertFresh,
        Self::Build,
        Self::Update,
        Self::RemovePresent,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::PresentHit => "present-hit",
            Self::AbsentMiss => "absent-miss",
            Self::InsertFresh => "insert-fresh",
            Self::Build => "build",
            Self::Update => "update",
            Self::RemovePresent => "remove-present",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Distribution {
    Dense,
    Random,
    HighBits,
    LowByte,
}

impl Distribution {
    const ALL: [Self; 4] = [Self::Dense, Self::Random, Self::HighBits, Self::LowByte];

    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Random => "random",
            Self::HighBits => "high-bits",
            Self::LowByte => "low-byte",
        }
    }
}

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

fn generate_keys(dist: Distribution, n: usize) -> (Vec<u64>, Vec<u64>) {
    let effective_n = if dist == Distribution::LowByte {
        n.min(256)
    } else {
        n
    };
    let mut present = Vec::with_capacity(effective_n);
    let mut absent = Vec::with_capacity(effective_n);

    match dist {
        Distribution::Dense => {
            for i in 0..effective_n {
                present.push(i as u64);
                absent.push((effective_n + i) as u64);
            }
        }
        Distribution::HighBits => {
            for i in 0..effective_n {
                present.push((i as u64) << 32);
                absent.push(((effective_n + i) as u64) << 32);
            }
        }
        Distribution::LowByte => {
            let high_present: u64 = 0xdead_beef_0000_0000;
            let high_absent: u64 = 0xcafe_babe_0000_0000;
            for i in 0..effective_n {
                present.push(high_present | ((i as u64) & 0xff));
                absent.push(high_absent | ((i as u64) & 0xff));
            }
        }
        Distribution::Random => {
            let mut rng_present = SplitMix64::new(0x1234_5678_9abc_def0);
            let mut rng_absent = SplitMix64::new(0x0fed_cba9_8765_4321);
            for _ in 0..effective_n {
                present.push(rng_present.next_u64() & !(1u64 << 63));
                absent.push(rng_absent.next_u64() | (1u64 << 63));
            }
        }
    }

    (present, absent)
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct ObservableResult {
    sum_values: i64,
    hit_count: usize,
    final_len: usize,
}

fn oracle_apeiron(
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
) -> ObservableResult {
    let n = present_keys.len();
    match col {
        Column::PresentHit => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in present_keys {
                if let Some(&v) = table.get(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: table.len(),
            }
        }
        Column::AbsentMiss => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in absent_keys {
                if let Some(&v) = table.get(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: table.len(),
            }
        }
        Column::InsertFresh => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: table.len(),
            }
        }
        Column::Build => {
            let mut table = ApeironTable::new();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: table.len(),
            }
        }
        Column::Update => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            for (&k, &v) in present_keys.iter().zip(update_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: table.len(),
            }
        }
        Column::RemovePresent => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in present_keys {
                if let Some(v) = table.remove(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: table.len(),
            }
        }
    }
}

fn oracle_hb_fx(
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
) -> ObservableResult {
    let n = present_keys.len();
    match col {
        Column::PresentHit => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in present_keys {
                if let Some(&v) = map.get(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: map.len(),
            }
        }
        Column::AbsentMiss => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in absent_keys {
                if let Some(&v) = map.get(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: map.len(),
            }
        }
        Column::InsertFresh => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: map.len(),
            }
        }
        Column::Build => {
            let mut map = HashMap::with_hasher(FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: map.len(),
            }
        }
        Column::Update => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            for (&k, &v) in present_keys.iter().zip(update_vals.iter()) {
                map.insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: map.len(),
            }
        }
        Column::RemovePresent => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in present_keys {
                if let Some(v) = map.remove(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: map.len(),
            }
        }
    }
}

fn oracle_hb_xxh3(
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
) -> ObservableResult {
    let n = present_keys.len();
    match col {
        Column::PresentHit => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in present_keys {
                if let Some(&v) = map.get(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: map.len(),
            }
        }
        Column::AbsentMiss => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in absent_keys {
                if let Some(&v) = map.get(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: map.len(),
            }
        }
        Column::InsertFresh => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: map.len(),
            }
        }
        Column::Build => {
            let mut map = HashMap::with_hasher(Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: map.len(),
            }
        }
        Column::Update => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            for (&k, &v) in present_keys.iter().zip(update_vals.iter()) {
                map.insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: map.len(),
            }
        }
        Column::RemovePresent => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in present_keys {
                if let Some(v) = map.remove(&k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: map.len(),
            }
        }
    }
}

fn time_apeiron(
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
) -> std::time::Duration {
    let n = present_keys.len();
    match col {
        Column::PresentHit => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let start = Instant::now();
            let mut hits = 0usize;
            for &k in present_keys {
                if let Some(v) = table.get(&k) {
                    black_box(v);
                    hits += 1;
                }
            }
            let dur = start.elapsed();
            black_box(hits);
            dur
        }
        Column::AbsentMiss => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let start = Instant::now();
            let mut misses = 0usize;
            for &k in absent_keys {
                if let Some(v) = table.get(&k) {
                    black_box(v);
                } else {
                    misses += 1;
                }
            }
            let dur = start.elapsed();
            black_box(misses);
            dur
        }
        Column::InsertFresh => {
            let mut table = ApeironTable::with_capacity(n);
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let dur = start.elapsed();
            black_box(table.len());
            dur
        }
        Column::Build => {
            let mut table = ApeironTable::new();
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let dur = start.elapsed();
            black_box(table.len());
            dur
        }
        Column::Update => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(update_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let dur = start.elapsed();
            black_box(table.len());
            dur
        }
        Column::RemovePresent => {
            let mut table = ApeironTable::with_capacity(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                table.insert_merge_last_write(k, v);
            }
            let start = Instant::now();
            let mut count = 0usize;
            for &k in present_keys {
                if let Some(v) = table.remove(&k) {
                    black_box(v);
                    count += 1;
                }
            }
            let dur = start.elapsed();
            black_box(count);
            dur
        }
    }
}

fn time_hb_fx(
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
) -> std::time::Duration {
    let n = present_keys.len();
    match col {
        Column::PresentHit => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let start = Instant::now();
            let mut hits = 0usize;
            for &k in present_keys {
                if let Some(v) = map.get(&k) {
                    black_box(v);
                    hits += 1;
                }
            }
            let dur = start.elapsed();
            black_box(hits);
            dur
        }
        Column::AbsentMiss => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let start = Instant::now();
            let mut misses = 0usize;
            for &k in absent_keys {
                if let Some(v) = map.get(&k) {
                    black_box(v);
                } else {
                    misses += 1;
                }
            }
            let dur = start.elapsed();
            black_box(misses);
            dur
        }
        Column::InsertFresh => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let dur = start.elapsed();
            black_box(map.len());
            dur
        }
        Column::Build => {
            let mut map = HashMap::with_hasher(FxBuildHasher);
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let dur = start.elapsed();
            black_box(map.len());
            dur
        }
        Column::Update => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(update_vals.iter()) {
                map.insert(k, v);
            }
            let dur = start.elapsed();
            black_box(map.len());
            dur
        }
        Column::RemovePresent => {
            let mut map = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let start = Instant::now();
            let mut count = 0usize;
            for &k in present_keys {
                if let Some(v) = map.remove(&k) {
                    black_box(v);
                    count += 1;
                }
            }
            let dur = start.elapsed();
            black_box(count);
            dur
        }
    }
}

fn time_hb_xxh3(
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
) -> std::time::Duration {
    let n = present_keys.len();
    match col {
        Column::PresentHit => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let start = Instant::now();
            let mut hits = 0usize;
            for &k in present_keys {
                if let Some(v) = map.get(&k) {
                    black_box(v);
                    hits += 1;
                }
            }
            let dur = start.elapsed();
            black_box(hits);
            dur
        }
        Column::AbsentMiss => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let start = Instant::now();
            let mut misses = 0usize;
            for &k in absent_keys {
                if let Some(v) = map.get(&k) {
                    black_box(v);
                } else {
                    misses += 1;
                }
            }
            let dur = start.elapsed();
            black_box(misses);
            dur
        }
        Column::InsertFresh => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let dur = start.elapsed();
            black_box(map.len());
            dur
        }
        Column::Build => {
            let mut map = HashMap::with_hasher(Xxh3BuildHasher);
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let dur = start.elapsed();
            black_box(map.len());
            dur
        }
        Column::Update => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(update_vals.iter()) {
                map.insert(k, v);
            }
            let dur = start.elapsed();
            black_box(map.len());
            dur
        }
        Column::RemovePresent => {
            let mut map = HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                map.insert(k, v);
            }
            let start = Instant::now();
            let mut count = 0usize;
            for &k in present_keys {
                if let Some(v) = map.remove(&k) {
                    black_box(v);
                    count += 1;
                }
            }
            let dur = start.elapsed();
            black_box(count);
            dur
        }
    }
}

#[derive(Debug, Clone)]
struct CellResult {
    column: &'static str,
    n: usize,
    distribution: &'static str,
    apeiron_ns: f64,
    hb_fx_ns: f64,
    ratio_hb_fx: f64,
    ratio_hb_fx_min: f64,
    ratio_hb_fx_max: f64,
    hb_xxh3_ns: f64,
    ratio_hb_xxh3: f64,
    ratio_hb_xxh3_min: f64,
    ratio_hb_xxh3_max: f64,
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

    let sha_ht = compute_file_sha256("src/hashtrinity/mod.rs");
    let sha_freq = compute_file_sha256("src/freq.rs");

    let rustc = std::process::Command::new("rustc")
        .arg("-V")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let cpu = read_cpu_model();
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());

    format!(
        "git: {} dirty: {} sha256(src/hashtrinity/mod.rs): {} sha256(src/freq.rs): {} rustc: {} cpu: {} target_dir: {}",
        rev, dirty, sha_ht, sha_freq, rustc, cpu, target_dir
    )
}

fn main() {
    let mut check = false;
    let mut quick = false;
    let mut json_path = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
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
    let sizes = if quick {
        vec![1_000, 100_000]
    } else {
        vec![1_000, 100_000, 1_000_000]
    };

    let mut results = Vec::new();

    for &col in &Column::ALL {
        for &dist in &Distribution::ALL {
            let dist_sizes: Vec<usize> = if dist == Distribution::LowByte {
                vec![256]
            } else {
                sizes.clone()
            };

            for &n in &dist_sizes {
                let (present_keys, absent_keys) = generate_keys(dist, n);
                let initial_vals: Vec<i64> = present_keys
                    .iter()
                    .map(|&k| (k as i64).wrapping_add(1))
                    .collect();
                let update_vals: Vec<i64> = present_keys
                    .iter()
                    .map(|&k| (k as i64).wrapping_add(1001))
                    .collect();

                // 7. Correctness oracle: per cell, the three arms must return identical observable results
                // (sum of found values, count of hits, final len); a mismatch prints the cell and exits 2
                // before any timing is used.
                let res_apeiron = oracle_apeiron(
                    col,
                    &present_keys,
                    &absent_keys,
                    &initial_vals,
                    &update_vals,
                );
                let res_hb_fx = oracle_hb_fx(
                    col,
                    &present_keys,
                    &absent_keys,
                    &initial_vals,
                    &update_vals,
                );
                let res_hb_xxh3 = oracle_hb_xxh3(
                    col,
                    &present_keys,
                    &absent_keys,
                    &initial_vals,
                    &update_vals,
                );

                if res_apeiron != res_hb_fx || res_apeiron != res_hb_xxh3 {
                    eprintln!(
                        "CORRECTNESS MISMATCH in cell [column: {}, n: {}, distribution: {}]:\n  apeiron: {:?}\n  hb_fx:   {:?}\n  hb_xxh3: {:?}",
                        col.name(),
                        n,
                        dist.name(),
                        res_apeiron,
                        res_hb_fx,
                        res_hb_xxh3
                    );
                    std::process::exit(2);
                }

                // 6. Protocol: R rounds per (column, n, distribution).
                // In each round run the three arms once each, rotating their order per round (interleaved).
                let mut ap_ns_rounds = Vec::with_capacity(rounds);
                let mut fx_ns_rounds = Vec::with_capacity(rounds);
                let mut xxh3_ns_rounds = Vec::with_capacity(rounds);
                let mut ratio_fx_rounds = Vec::with_capacity(rounds);
                let mut ratio_xxh3_rounds = Vec::with_capacity(rounds);

                for r in 0..rounds {
                    let (ap_dur, fx_dur, xxh3_dur) = match r % 3 {
                        0 => {
                            let ap = time_apeiron(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            let fx = time_hb_fx(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            let xxh3 = time_hb_xxh3(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            (ap, fx, xxh3)
                        }
                        1 => {
                            let fx = time_hb_fx(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            let xxh3 = time_hb_xxh3(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            let ap = time_apeiron(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            (ap, fx, xxh3)
                        }
                        _ => {
                            let xxh3 = time_hb_xxh3(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            let ap = time_apeiron(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            let fx = time_hb_fx(
                                col,
                                &present_keys,
                                &absent_keys,
                                &initial_vals,
                                &update_vals,
                            );
                            (ap, fx, xxh3)
                        }
                    };

                    let ap_ns = ap_dur.as_nanos() as f64 / n as f64;
                    let fx_ns = fx_dur.as_nanos() as f64 / n as f64;
                    let xxh3_ns = xxh3_dur.as_nanos() as f64 / n as f64;

                    ap_ns_rounds.push(ap_ns);
                    fx_ns_rounds.push(fx_ns);
                    xxh3_ns_rounds.push(xxh3_ns);

                    ratio_fx_rounds.push(ap_ns / fx_ns);
                    ratio_xxh3_rounds.push(ap_ns / xxh3_ns);
                }

                let mean_ap = ap_ns_rounds.iter().sum::<f64>() / (rounds as f64);
                let mean_fx = fx_ns_rounds.iter().sum::<f64>() / (rounds as f64);
                let mean_xxh3 = xxh3_ns_rounds.iter().sum::<f64>() / (rounds as f64);

                let mean_ratio_fx = ratio_fx_rounds.iter().sum::<f64>() / (rounds as f64);
                let min_ratio_fx = ratio_fx_rounds
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let max_ratio_fx = ratio_fx_rounds
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);

                let mean_ratio_xxh3 = ratio_xxh3_rounds.iter().sum::<f64>() / (rounds as f64);
                let min_ratio_xxh3 = ratio_xxh3_rounds
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let max_ratio_xxh3 = ratio_xxh3_rounds
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);

                results.push(CellResult {
                    column: col.name(),
                    n,
                    distribution: dist.name(),
                    apeiron_ns: mean_ap,
                    hb_fx_ns: mean_fx,
                    ratio_hb_fx: mean_ratio_fx,
                    ratio_hb_fx_min: min_ratio_fx,
                    ratio_hb_fx_max: max_ratio_fx,
                    hb_xxh3_ns: mean_xxh3,
                    ratio_hb_xxh3: mean_ratio_xxh3,
                    ratio_hb_xxh3_min: min_ratio_xxh3,
                    ratio_hb_xxh3_max: max_ratio_xxh3,
                });
            }
        }
    }

    // 8. Header line before results: git rev-parse HEAD, dirty flag, sha256 of src/hashtrinity/mod.rs
    // and src/freq.rs, rustc -V, the CPU model from /proc/cpuinfo, and CARGO_TARGET_DIR (or default).
    let header = get_header_line();
    println!("{}", header);
    println!();

    println!(
        "| {:<14} | {:>7} | {:<9} | {:>10} | {:>10} | {:>22} | {:>10} | {:>22} |",
        "column",
        "n",
        "dist",
        "apeiron ns",
        "hb_fx ns",
        "ap/fx [min, max]",
        "hb_xxh3 ns",
        "ap/xxh3 [min, max]"
    );
    println!(
        "|{:-<16}|{:-<9}|{:-<11}|{:-<12}|{:-<12}|{:-<24}|{:-<12}|{:-<24}|",
        "", "", "", "", "", "", "", ""
    );

    for cell in &results {
        println!(
            "| {:<14} | {:>7} | {:<9} | {:>10.2} | {:>10.2} | {:>6.3} [{:>5.3}, {:>5.3}] | {:>10.2} | {:>6.3} [{:>5.3}, {:>5.3}] |",
            cell.column,
            cell.n,
            cell.distribution,
            cell.apeiron_ns,
            cell.hb_fx_ns,
            cell.ratio_hb_fx,
            cell.ratio_hb_fx_min,
            cell.ratio_hb_fx_max,
            cell.hb_xxh3_ns,
            cell.ratio_hb_xxh3,
            cell.ratio_hb_xxh3_min,
            cell.ratio_hb_xxh3_max,
        );
    }

    if let Some(path) = &json_path {
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut json_lines = String::new();
        for cell in &results {
            json_lines.push_str(&format!(
                "{{\"column\":\"{}\",\"n\":{},\"distribution\":\"{}\",\"apeiron_ns\":{:.3},\"hb_fx_ns\":{:.3},\"ratio_hb_fx\":{:.4},\"ratio_hb_fx_min\":{:.4},\"ratio_hb_fx_max\":{:.4},\"hb_xxh3_ns\":{:.3},\"ratio_hb_xxh3\":{:.4},\"ratio_hb_xxh3_min\":{:.4},\"ratio_hb_xxh3_max\":{:.4}}}\n",
                cell.column,
                cell.n,
                cell.distribution,
                cell.apeiron_ns,
                cell.hb_fx_ns,
                cell.ratio_hb_fx,
                cell.ratio_hb_fx_min,
                cell.ratio_hb_fx_max,
                cell.hb_xxh3_ns,
                cell.ratio_hb_xxh3,
                cell.ratio_hb_xxh3_min,
                cell.ratio_hb_xxh3_max,
            ));
        }
        if let Err(e) = std::fs::write(path, json_lines) {
            eprintln!("Failed to write JSON lines to {}: {}", path, e);
        }
    }

    if check {
        let mut failures = Vec::new();
        for cell in &results {
            if cell.ratio_hb_fx > 1.0 {
                failures.push(format!(
                    "FAIL: {} n={} dist={}: ratio vs hb_fx = {:.3} > 1.0",
                    cell.column, cell.n, cell.distribution, cell.ratio_hb_fx
                ));
            }
            if cell.ratio_hb_xxh3 > 1.0 {
                failures.push(format!(
                    "FAIL: {} n={} dist={}: ratio vs hb_xxh3 = {:.3} > 1.0",
                    cell.column, cell.n, cell.distribution, cell.ratio_hb_xxh3
                ));
            }
        }
        if !failures.is_empty() {
            eprintln!(
                "\nCheck failed: {} cell-ratio thresholds exceeded:",
                failures.len()
            );
            for f in &failures {
                eprintln!("  {}", f);
            }
            std::process::exit(1);
        }
    }
}
