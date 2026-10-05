// Row: `apeiron-bench-restore-EaEUYoDjUS4nFW9F4W7iUP` (APEIRON-BENCH-RESTORE): the L3 instrument.
// Row: `apeiron-beat-Bu3jZusxHvuMEJ8FOQ3C9O` (APEIRON-BEAT): both apeiron faces measured side by side.
// The stored-key `Table` and the keyless `CoordMap` are two faces of one table; the bench prints both,
// and the acceptance `--check` reads the `CoordMap` (the recommended face for injective coordinates).
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

use apeiron::freq::{CoordMap, Table};
use hashbrown::HashMap;
use rustc_hash::FxBuildHasher;
use sha2::Digest;

/// The stored-key apeiron face: keys live in the map (`Stored`).
type ApTable = Table<u64, i64>;
/// The keyless apeiron face: keys regenerate from the slot (`Regen`), recommended for injective coordinates.
type ApCoord = CoordMap<u64, i64>;
/// The `FxHash` comparator arm.
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

/// The operations the oracle and the timer drive on the four measured arms. One impl per arm lets a
/// single generic body measure all four side by side, so the type under measurement is never swapped.
trait FaceOps {
    fn face_insert(&mut self, k: u64, v: i64);
    fn face_get(&self, k: u64) -> Option<&i64>;
    fn face_remove(&mut self, k: u64) -> Option<i64>;
    fn face_len(&self) -> usize;
}

impl FaceOps for ApTable {
    fn face_insert(&mut self, k: u64, v: i64) {
        self.insert_merge_last_write(k, v);
    }
    fn face_get(&self, k: u64) -> Option<&i64> {
        self.get(&k)
    }
    fn face_remove(&mut self, k: u64) -> Option<i64> {
        self.remove(&k)
    }
    fn face_len(&self) -> usize {
        self.len()
    }
}

impl FaceOps for ApCoord {
    fn face_insert(&mut self, k: u64, v: i64) {
        self.insert_merge_last_write(k, v);
    }
    fn face_get(&self, k: u64) -> Option<&i64> {
        self.get(&k)
    }
    fn face_remove(&mut self, k: u64) -> Option<i64> {
        self.remove(&k)
    }
    fn face_len(&self) -> usize {
        self.len()
    }
}

impl FaceOps for HbFx {
    fn face_insert(&mut self, k: u64, v: i64) {
        self.insert(k, v);
    }
    fn face_get(&self, k: u64) -> Option<&i64> {
        self.get(&k)
    }
    fn face_remove(&mut self, k: u64) -> Option<i64> {
        self.remove(&k)
    }
    fn face_len(&self) -> usize {
        self.len()
    }
}

impl FaceOps for HbXxh3 {
    fn face_insert(&mut self, k: u64, v: i64) {
        self.insert(k, v);
    }
    fn face_get(&self, k: u64) -> Option<&i64> {
        self.get(&k)
    }
    fn face_remove(&mut self, k: u64) -> Option<i64> {
        self.remove(&k)
    }
    fn face_len(&self) -> usize {
        self.len()
    }
}

// Concrete constructors: `sized(n)` presizes for `n` keys, `empty()` is the grow-as-needed build.
fn ap_table_sized(n: usize) -> ApTable {
    ApTable::with_capacity(n)
}
fn ap_table_empty() -> ApTable {
    ApTable::new()
}
fn ap_coord_sized(n: usize) -> ApCoord {
    ApCoord::coordinate_with_capacity(n)
}
fn ap_coord_empty() -> ApCoord {
    ApCoord::coordinate()
}
fn hb_fx_sized(n: usize) -> HbFx {
    HashMap::with_capacity_and_hasher(n, FxBuildHasher)
}
fn hb_fx_empty() -> HbFx {
    HashMap::with_hasher(FxBuildHasher)
}
fn hb_xxh3_sized(n: usize) -> HbXxh3 {
    HashMap::with_capacity_and_hasher(n, Xxh3BuildHasher)
}
fn hb_xxh3_empty() -> HbXxh3 {
    HashMap::with_hasher(Xxh3BuildHasher)
}

/// The four measured arms, in the base order the per-round rotation starts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    ApTable,
    ApCoord,
    HbFx,
    HbXxh3,
}

const AP_TABLE: usize = 0;
const AP_COORD: usize = 1;
const HB_FX: usize = 2;
const HB_XXH3: usize = 3;

impl Arm {
    const ALL: [Self; 4] = [Self::ApTable, Self::ApCoord, Self::HbFx, Self::HbXxh3];

    fn index(self) -> usize {
        match self {
            Self::ApTable => AP_TABLE,
            Self::ApCoord => AP_COORD,
            Self::HbFx => HB_FX,
            Self::HbXxh3 => HB_XXH3,
        }
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

fn oracle_face<T: FaceOps>(
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
    sized: fn(usize) -> T,
    empty: fn() -> T,
) -> ObservableResult {
    let n = present_keys.len();
    match col {
        Column::PresentHit => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in present_keys {
                if let Some(&v) = face.face_get(k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: face.face_len(),
            }
        }
        Column::AbsentMiss => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in absent_keys {
                if let Some(&v) = face.face_get(k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: face.face_len(),
            }
        }
        Column::InsertFresh => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: face.face_len(),
            }
        }
        Column::Build => {
            let mut face = empty();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: face.face_len(),
            }
        }
        Column::Update => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            for (&k, &v) in present_keys.iter().zip(update_vals.iter()) {
                face.face_insert(k, v);
            }
            ObservableResult {
                sum_values: 0,
                hit_count: 0,
                final_len: face.face_len(),
            }
        }
        Column::RemovePresent => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let mut sum_values = 0i64;
            let mut hit_count = 0usize;
            for &k in present_keys {
                if let Some(v) = face.face_remove(k) {
                    hit_count += 1;
                    sum_values = sum_values.wrapping_add(v);
                }
            }
            ObservableResult {
                sum_values,
                hit_count,
                final_len: face.face_len(),
            }
        }
    }
}

fn oracle_arm(
    arm: Arm,
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
) -> ObservableResult {
    match arm {
        Arm::ApTable => oracle_face(
            col,
            present_keys,
            absent_keys,
            initial_vals,
            update_vals,
            ap_table_sized,
            ap_table_empty,
        ),
        Arm::ApCoord => oracle_face(
            col,
            present_keys,
            absent_keys,
            initial_vals,
            update_vals,
            ap_coord_sized,
            ap_coord_empty,
        ),
        Arm::HbFx => oracle_face(
            col,
            present_keys,
            absent_keys,
            initial_vals,
            update_vals,
            hb_fx_sized,
            hb_fx_empty,
        ),
        Arm::HbXxh3 => oracle_face(
            col,
            present_keys,
            absent_keys,
            initial_vals,
            update_vals,
            hb_xxh3_sized,
            hb_xxh3_empty,
        ),
    }
}

fn time_face<T: FaceOps>(
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
    sized: fn(usize) -> T,
    empty: fn() -> T,
) -> std::time::Duration {
    let n = present_keys.len();
    match col {
        Column::PresentHit => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let start = Instant::now();
            let mut hits = 0usize;
            for &k in present_keys {
                if let Some(v) = face.face_get(k) {
                    black_box(v);
                    hits += 1;
                }
            }
            let dur = start.elapsed();
            black_box(hits);
            dur
        }
        Column::AbsentMiss => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let start = Instant::now();
            let mut misses = 0usize;
            for &k in absent_keys {
                if let Some(v) = face.face_get(k) {
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
            let mut face = sized(n);
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let dur = start.elapsed();
            black_box(face.face_len());
            dur
        }
        Column::Build => {
            let mut face = empty();
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let dur = start.elapsed();
            black_box(face.face_len());
            dur
        }
        Column::Update => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let start = Instant::now();
            for (&k, &v) in present_keys.iter().zip(update_vals.iter()) {
                face.face_insert(k, v);
            }
            let dur = start.elapsed();
            black_box(face.face_len());
            dur
        }
        Column::RemovePresent => {
            let mut face = sized(n);
            for (&k, &v) in present_keys.iter().zip(initial_vals.iter()) {
                face.face_insert(k, v);
            }
            let start = Instant::now();
            let mut count = 0usize;
            for &k in present_keys {
                if let Some(v) = face.face_remove(k) {
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

fn time_arm(
    arm: Arm,
    col: Column,
    present_keys: &[u64],
    absent_keys: &[u64],
    initial_vals: &[i64],
    update_vals: &[i64],
) -> std::time::Duration {
    match arm {
        Arm::ApTable => time_face(
            col,
            present_keys,
            absent_keys,
            initial_vals,
            update_vals,
            ap_table_sized,
            ap_table_empty,
        ),
        Arm::ApCoord => time_face(
            col,
            present_keys,
            absent_keys,
            initial_vals,
            update_vals,
            ap_coord_sized,
            ap_coord_empty,
        ),
        Arm::HbFx => time_face(
            col,
            present_keys,
            absent_keys,
            initial_vals,
            update_vals,
            hb_fx_sized,
            hb_fx_empty,
        ),
        Arm::HbXxh3 => time_face(
            col,
            present_keys,
            absent_keys,
            initial_vals,
            update_vals,
            hb_xxh3_sized,
            hb_xxh3_empty,
        ),
    }
}

#[derive(Debug, Clone)]
struct CellResult {
    column: &'static str,
    n: usize,
    distribution: &'static str,
    ap_table_ns: f64,
    ap_coord_ns: f64,
    hb_fx_ns: f64,
    hb_xxh3_ns: f64,
    ratio_ap_coord_hb_fx: f64,
    ratio_ap_coord_hb_fx_min: f64,
    ratio_ap_coord_hb_fx_max: f64,
    ratio_ap_coord_hb_xxh3: f64,
    ratio_ap_coord_hb_xxh3_min: f64,
    ratio_ap_coord_hb_xxh3_max: f64,
    ratio_ap_table_hb_fx: f64,
    ratio_ap_table_hb_fx_min: f64,
    ratio_ap_table_hb_fx_max: f64,
    ratio_ap_table_hb_xxh3: f64,
    ratio_ap_table_hb_xxh3_min: f64,
    ratio_ap_table_hb_xxh3_max: f64,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / (values.len() as f64)
}

fn ratio_stats(values: &[f64]) -> (f64, f64, f64) {
    let mean = mean(values);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
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

/// The four face names, resolved from the types rather than restated.
fn get_face_line() -> String {
    format!(
        "faces: ap_table={} (stored keys, with_capacity) | ap_coord={} (keyless, coordinate_with_capacity) | hb_fx={} | hb_xxh3={}",
        std::any::type_name::<ApTable>(),
        std::any::type_name::<ApCoord>(),
        std::any::type_name::<HbFx>(),
        std::any::type_name::<HbXxh3>(),
    )
}

fn main() {
    let mut check = false;
    let mut check_table = false;
    let mut quick = false;
    let mut json_path = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--check-table" => check_table = true,
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

                // Correctness oracle: per cell, all four arms must return identical observable results
                // (sum of found values, count of hits, final len); a mismatch prints the cell and exits 2
                // before any timing is used.
                let res_ap_table = oracle_arm(
                    Arm::ApTable,
                    col,
                    &present_keys,
                    &absent_keys,
                    &initial_vals,
                    &update_vals,
                );
                let res_ap_coord = oracle_arm(
                    Arm::ApCoord,
                    col,
                    &present_keys,
                    &absent_keys,
                    &initial_vals,
                    &update_vals,
                );
                let res_hb_fx = oracle_arm(
                    Arm::HbFx,
                    col,
                    &present_keys,
                    &absent_keys,
                    &initial_vals,
                    &update_vals,
                );
                let res_hb_xxh3 = oracle_arm(
                    Arm::HbXxh3,
                    col,
                    &present_keys,
                    &absent_keys,
                    &initial_vals,
                    &update_vals,
                );

                if res_ap_table != res_ap_coord
                    || res_ap_table != res_hb_fx
                    || res_ap_table != res_hb_xxh3
                {
                    eprintln!(
                        "CORRECTNESS MISMATCH in cell [column: {}, n: {}, distribution: {}]:\n  ap_table: {:?}\n  ap_coord: {:?}\n  hb_fx:    {:?}\n  hb_xxh3:  {:?}",
                        col.name(),
                        n,
                        dist.name(),
                        res_ap_table,
                        res_ap_coord,
                        res_hb_fx,
                        res_hb_xxh3
                    );
                    std::process::exit(2);
                }

                // Protocol: R rounds per (column, n, distribution). In each round run all four arms once
                // each, rotating their order per round (interleaved), so no arm holds a fixed position.
                let mut ns_rounds: [Vec<f64>; 4] =
                    std::array::from_fn(|_| Vec::with_capacity(rounds));
                let mut ratio_coord_fx = Vec::with_capacity(rounds);
                let mut ratio_coord_xxh3 = Vec::with_capacity(rounds);
                let mut ratio_table_fx = Vec::with_capacity(rounds);
                let mut ratio_table_xxh3 = Vec::with_capacity(rounds);

                for r in 0..rounds {
                    let rotated: [Arm; 4] =
                        std::array::from_fn(|i| Arm::ALL[(i + r) % Arm::ALL.len()]);
                    let mut durs = [std::time::Duration::ZERO; 4];
                    for arm in rotated {
                        durs[arm.index()] = time_arm(
                            arm,
                            col,
                            &present_keys,
                            &absent_keys,
                            &initial_vals,
                            &update_vals,
                        );
                    }

                    let ns: [f64; 4] =
                        std::array::from_fn(|i| durs[i].as_nanos() as f64 / n as f64);
                    for (slot, value) in ns_rounds.iter_mut().zip(ns) {
                        slot.push(value);
                    }

                    ratio_coord_fx.push(ns[AP_COORD] / ns[HB_FX]);
                    ratio_coord_xxh3.push(ns[AP_COORD] / ns[HB_XXH3]);
                    ratio_table_fx.push(ns[AP_TABLE] / ns[HB_FX]);
                    ratio_table_xxh3.push(ns[AP_TABLE] / ns[HB_XXH3]);
                }

                let (mean_ratio_coord_fx, min_ratio_coord_fx, max_ratio_coord_fx) =
                    ratio_stats(&ratio_coord_fx);
                let (mean_ratio_coord_xxh3, min_ratio_coord_xxh3, max_ratio_coord_xxh3) =
                    ratio_stats(&ratio_coord_xxh3);
                let (mean_ratio_table_fx, min_ratio_table_fx, max_ratio_table_fx) =
                    ratio_stats(&ratio_table_fx);
                let (mean_ratio_table_xxh3, min_ratio_table_xxh3, max_ratio_table_xxh3) =
                    ratio_stats(&ratio_table_xxh3);

                results.push(CellResult {
                    column: col.name(),
                    n,
                    distribution: dist.name(),
                    ap_table_ns: mean(&ns_rounds[AP_TABLE]),
                    ap_coord_ns: mean(&ns_rounds[AP_COORD]),
                    hb_fx_ns: mean(&ns_rounds[HB_FX]),
                    hb_xxh3_ns: mean(&ns_rounds[HB_XXH3]),
                    ratio_ap_coord_hb_fx: mean_ratio_coord_fx,
                    ratio_ap_coord_hb_fx_min: min_ratio_coord_fx,
                    ratio_ap_coord_hb_fx_max: max_ratio_coord_fx,
                    ratio_ap_coord_hb_xxh3: mean_ratio_coord_xxh3,
                    ratio_ap_coord_hb_xxh3_min: min_ratio_coord_xxh3,
                    ratio_ap_coord_hb_xxh3_max: max_ratio_coord_xxh3,
                    ratio_ap_table_hb_fx: mean_ratio_table_fx,
                    ratio_ap_table_hb_fx_min: min_ratio_table_fx,
                    ratio_ap_table_hb_fx_max: max_ratio_table_fx,
                    ratio_ap_table_hb_xxh3: mean_ratio_table_xxh3,
                    ratio_ap_table_hb_xxh3_min: min_ratio_table_xxh3,
                    ratio_ap_table_hb_xxh3_max: max_ratio_table_xxh3,
                });
            }
        }
    }

    // Header line before results: git rev-parse HEAD, dirty flag, sha256 of src/hashtrinity/mod.rs
    // and src/freq.rs, rustc -V, the CPU model from /proc/cpuinfo, and CARGO_TARGET_DIR (or default).
    // The face line names the four measured types; the identity line above it is unchanged.
    let header = get_header_line();
    println!("{}", header);
    println!("{}", get_face_line());
    println!();

    println!(
        "| {:<14} | {:>7} | {:<9} | {:>12} | {:>12} | {:>12} | {:>12} | {:>24} | {:>24} | {:>24} | {:>24} |",
        "column",
        "n",
        "dist",
        "ap_table ns",
        "ap_coord ns",
        "hb_fx ns",
        "hb_xxh3 ns",
        "ap_coord/hb_fx [min,max]",
        "ap_coord/hb_xxh3 [min,max]",
        "ap_table/hb_fx [min,max]",
        "ap_table/hb_xxh3 [min,max]"
    );
    println!(
        "|{:-<16}|{:-<9}|{:-<11}|{:-<14}|{:-<14}|{:-<14}|{:-<14}|{:-<26}|{:-<26}|{:-<26}|{:-<26}|",
        "", "", "", "", "", "", "", "", "", "", ""
    );

    for cell in &results {
        println!(
            "| {:<14} | {:>7} | {:<9} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.2} | {:>7.3} [{:>6.3}, {:>6.3}] | {:>7.3} [{:>6.3}, {:>6.3}] | {:>7.3} [{:>6.3}, {:>6.3}] | {:>7.3} [{:>6.3}, {:>6.3}] |",
            cell.column,
            cell.n,
            cell.distribution,
            cell.ap_table_ns,
            cell.ap_coord_ns,
            cell.hb_fx_ns,
            cell.hb_xxh3_ns,
            cell.ratio_ap_coord_hb_fx,
            cell.ratio_ap_coord_hb_fx_min,
            cell.ratio_ap_coord_hb_fx_max,
            cell.ratio_ap_coord_hb_xxh3,
            cell.ratio_ap_coord_hb_xxh3_min,
            cell.ratio_ap_coord_hb_xxh3_max,
            cell.ratio_ap_table_hb_fx,
            cell.ratio_ap_table_hb_fx_min,
            cell.ratio_ap_table_hb_fx_max,
            cell.ratio_ap_table_hb_xxh3,
            cell.ratio_ap_table_hb_xxh3_min,
            cell.ratio_ap_table_hb_xxh3_max,
        );
    }

    if let Some(path) = &json_path {
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut json_lines = String::new();
        for cell in &results {
            json_lines.push_str(&format!(
                "{{\"column\":\"{}\",\"n\":{},\"distribution\":\"{}\",\"ap_table_ns\":{:.3},\"ap_coord_ns\":{:.3},\"hb_fx_ns\":{:.3},\"hb_xxh3_ns\":{:.3},\"ratio_ap_coord_hb_fx\":{:.4},\"ratio_ap_coord_hb_fx_min\":{:.4},\"ratio_ap_coord_hb_fx_max\":{:.4},\"ratio_ap_coord_hb_xxh3\":{:.4},\"ratio_ap_coord_hb_xxh3_min\":{:.4},\"ratio_ap_coord_hb_xxh3_max\":{:.4},\"ratio_ap_table_hb_fx\":{:.4},\"ratio_ap_table_hb_fx_min\":{:.4},\"ratio_ap_table_hb_fx_max\":{:.4},\"ratio_ap_table_hb_xxh3\":{:.4},\"ratio_ap_table_hb_xxh3_min\":{:.4},\"ratio_ap_table_hb_xxh3_max\":{:.4}}}\n",
                cell.column,
                cell.n,
                cell.distribution,
                cell.ap_table_ns,
                cell.ap_coord_ns,
                cell.hb_fx_ns,
                cell.hb_xxh3_ns,
                cell.ratio_ap_coord_hb_fx,
                cell.ratio_ap_coord_hb_fx_min,
                cell.ratio_ap_coord_hb_fx_max,
                cell.ratio_ap_coord_hb_xxh3,
                cell.ratio_ap_coord_hb_xxh3_min,
                cell.ratio_ap_coord_hb_xxh3_max,
                cell.ratio_ap_table_hb_fx,
                cell.ratio_ap_table_hb_fx_min,
                cell.ratio_ap_table_hb_fx_max,
                cell.ratio_ap_table_hb_xxh3,
                cell.ratio_ap_table_hb_xxh3_min,
                cell.ratio_ap_table_hb_xxh3_max,
            ));
        }
        if let Err(e) = std::fs::write(path, json_lines) {
            eprintln!("Failed to write JSON lines to {}: {}", path, e);
        }
    }

    if check || check_table {
        let mut failures: Vec<String> = Vec::new();
        for cell in &results {
            // The acceptance ratios: the keyless `ap_coord` face against each hashbrown comparator.
            for (opponent, ratio) in [
                ("hb_fx", cell.ratio_ap_coord_hb_fx),
                ("hb_xxh3", cell.ratio_ap_coord_hb_xxh3),
            ] {
                if ratio > 1.0 {
                    failures.push(format!(
                        "FAIL(ap_coord): {} n={} dist={}: ratio vs {} = {:.3} > 1.0",
                        cell.column, cell.n, cell.distribution, opponent, ratio
                    ));
                }
            }
            // `--check-table` additionally reads the stored-key `ap_table` face.
            if check_table {
                for (opponent, ratio) in [
                    ("hb_fx", cell.ratio_ap_table_hb_fx),
                    ("hb_xxh3", cell.ratio_ap_table_hb_xxh3),
                ] {
                    if ratio > 1.0 {
                        failures.push(format!(
                            "FAIL(ap_table): {} n={} dist={}: ratio vs {} = {:.3} > 1.0",
                            cell.column, cell.n, cell.distribution, opponent, ratio
                        ));
                    }
                }
            }
        }
        if !failures.is_empty() {
            println!();
            for failure in &failures {
                println!("{}", failure);
            }
            eprintln!(
                "\nCheck failed: {} cell-ratio threshold(s) exceeded:",
                failures.len()
            );
            for failure in &failures {
                eprintln!("  {}", failure);
            }
            std::process::exit(1);
        }
    }
}
