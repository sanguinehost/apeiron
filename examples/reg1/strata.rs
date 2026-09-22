//! REG-1 current strata enumeration over the current `apeiron::Table` and the pinned
//! `hashtrinity::Table`.
//!
//! The existing fixed-16 measurement conflates three axes in one cell: occupancy 16 per Apeiron
//! shard, a direct-index-resident coordinate set, and a present-hit/update operation mix. This
//! module separates them:
//!
//! * occupancy per Apeiron shard: 1, 2, 4, 8, 16 over 256 shards;
//! * coordinate distribution: direct-index-resident, offset beyond hashtrinity's direct range,
//!   deterministic seeded random, and clustered/skewed (distinct coordinates that share a
//!   hashtrinity home);
//! * operation: present hit, absent miss, existing update, admitted fresh insert, growth insert.
//!
//! Each cell carries its own deterministic seed, simultaneous population, an oracle digest computed
//! before timing, the warmup/sample protocol, the raw sample interval, the median and the same-run
//! ratio. A cell whose coordinate set is not distinct, or whose implementation digest disagrees with
//! the oracle, is rejected before any sample is accepted and never enters a speed cell.
//!
//! Observable metrics only: `hashtrinity::Table::{capacity, longest_probe}` are public, so the slot
//! count and the worst probe depth are read from the real table. The production `apeiron::Table`
//! exposes no comparison counter, so `apeiron_key_compares` is reported `MISSING` rather than
//! inferred from occupancy.
//!
//! The owner acceptance threshold `apeiron/hashtrinity <= 1.0` (match or beat the applicable
//! incumbent) is applied per applicable cell: a cell passes only when its measured ratio is at
//! most 1.0. The retired fixed-shard `refusal` cell has no hashtrinity counterpart; the growable
//! contract replaces it with a `growth-insert` cell (a fresh insert into a 16-key region, which
//! grows the region) that does have a corresponding hashtrinity insert path and is measured
//! normally. There is no envelope tolerance.
//!
//! Genuine resource cells of the growable contract (zero group width, sizing overflow, injected
//! reservation failure) are enumerated beside the speed cells. Zero group width is measured: every
//! insert returns `SizeOverflow`, the state is unchanged and neither merge nor clone runs. Sizing
//! overflow and injected reservation failure are not forceable in this harness and are registered
//! `missing`, not fabricated.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the REG-1 strata enumeration"
)]

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::Instant;

use apeiron::table::Table;
use hashtrinity::Table as HtTable;

use crate::matrix::{
    cpu_model, join, maximum, median, minimum, ns_per_op, ratio, MergeMode, Profile, ResourceCell,
};

/// The fixed shard count of `apeiron::Table`.
pub(crate) const SHARDS: usize = 256;
/// The current `apeiron::Table` per-shard override capacity.
pub(crate) const APEIRON_N: usize = 16;
/// The occupancy strata per Apeiron shard.
const OCCUPANCIES: [usize; 5] = [1, 2, 4, 8, 16];
/// The owner acceptance threshold (elapsed-median, selected/incumbent): match or beat the
/// applicable incumbent, so `selected/incumbent <= 1.0`. No envelope tolerance.
const RATIO_LIMIT: f64 = 1.0;

/// Warmup traces per implementation per campaign.
pub(crate) const STRATA_WARMUP: u32 = 2;
/// Timed samples per implementation per campaign.
pub(crate) const STRATA_SAMPLES: u32 = 15;
/// Scheduled operations in one read or update trace.
pub(crate) const STRATA_OPS: usize = 4096;
/// Campaigns for the full profile.
pub(crate) const FULL_CAMPAIGNS: u32 = 3;
/// Campaigns for the reduced smoke profile.
pub(crate) const SMOKE_CAMPAIGNS: u32 = 1;

/// The deterministic workload seed, printed with every cell.
const STRATA_SEED: u64 = 0x5EED_1234_ABCD_0001;

/// FNV-1a 64-bit offset basis.
pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The base of the offset-beyond-direct coordinate band (a multiple of 256, far above any
/// hashtrinity direct range at these populations).
const OFFSET_BASE: u64 = 1 << 24;
/// The base of the seeded-random coordinate band.
const RAND_BASE: u64 = 1 << 25;
/// The 23-bit mask of the seeded-random displacement.
const RAND_MASK: u64 = (1 << 23) - 1;
const RAND_ODD: u64 = 0x9E37_79B9_7F4A_7C15;

// The key and home definitions moved into this child module to keep `strata.rs` below the
// source-size cap; the re-export preserves the `crate::strata::{..}` access paths.
#[path = "adapters/native_strata.rs"]
mod native_strata;

pub(crate) use native_strata::{RawStrataHome, StrataHome, StrataKey};

type Apeiron = Table<StrataKey, i64, RawStrataHome, APEIRON_N>;
/// The pinned hashtrinity baseline under measurement (stored keys, the existing baseline).
type Ht = HtTable<StrataKey, i64>;

/// A total merge usable as both Apeiron `Merge<i64>` and hashtrinity's `FnOnce(i64, i64) -> i64`.
pub(crate) type MergeFn = fn(i64, i64) -> i64;

fn replacement_merge(new: i64, _old: i64) -> i64 {
    new
}

/// Wrapping addition: the new contribution is added to the old.
fn wrapping_add_merge(new: i64, old: i64) -> i64 {
    new.wrapping_add(old)
}

pub(crate) fn merge_fn(mode: MergeMode) -> MergeFn {
    match mode {
        MergeMode::Replacement => replacement_merge,
        MergeMode::WrappingAdd => wrapping_add_merge,
    }
}

/// FNV-1a fold of one canonical value, the same for both implementations and the oracle.
#[inline]
pub(crate) fn fold(hash: &mut u64, value: i64) {
    *hash = (*hash ^ value.cast_unsigned()).wrapping_mul(FNV_PRIME);
}

/// Nanoseconds elapsed since `start`, saturating rather than wrapping.
pub(crate) fn nanos_since(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Distribution {
    /// Small coordinates that hashtrinity indexes directly (`coord < capacity`).
    DirectIndexResident,
    /// Coordinates above hashtrinity's direct range, clustered in one band.
    OffsetBeyondDirect,
    /// A deterministic seeded pseudo-random spread above the direct range.
    SeededRandom,
    /// Distinct coordinates that all share one hashtrinity home per Apeiron shard.
    ClusteredSkewed,
}

impl Distribution {
    const ALL: [Distribution; 4] = [
        Distribution::DirectIndexResident,
        Distribution::OffsetBeyondDirect,
        Distribution::SeededRandom,
        Distribution::ClusteredSkewed,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Distribution::DirectIndexResident => "direct-index-resident",
            Distribution::OffsetBeyondDirect => "offset-beyond-direct",
            Distribution::SeededRandom => "seeded-random",
            Distribution::ClusteredSkewed => "clustered-skewed",
        }
    }

    pub(crate) fn is_direct(self) -> bool {
        matches!(self, Distribution::DirectIndexResident)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Operation {
    /// A lookup of an admitted key.
    PresentHit,
    /// A lookup of a key absent from both tables.
    AbsentMiss,
    /// An `insert_with` on an admitted key (the merge path).
    ExistingUpdate,
    /// An `insert_with` of a fresh key into a shard with a vacant slot.
    AdmittedFreshInsert,
    /// An `insert_with` of a fresh key into a region already holding `occupancy` keys; at occupancy
    /// 16 the region has no vacant lane, so the growable table grows it instead of refusing.
    GrowthInsert,
}

impl Operation {
    const ALL: [Operation; 5] = [
        Operation::PresentHit,
        Operation::AbsentMiss,
        Operation::ExistingUpdate,
        Operation::AdmittedFreshInsert,
        Operation::GrowthInsert,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Operation::PresentHit => "present-hit",
            Operation::AbsentMiss => "absent-miss",
            Operation::ExistingUpdate => "existing-update",
            Operation::AdmittedFreshInsert => "admitted-fresh-insert",
            Operation::GrowthInsert => "growth-insert",
        }
    }

    fn is_read(self) -> bool {
        matches!(self, Operation::PresentHit | Operation::AbsentMiss)
    }
}

/// One strata cell identity.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellSpec {
    pub(crate) occupancy: usize,
    pub(crate) distribution: Distribution,
    pub(crate) operation: Operation,
    pub(crate) merge: MergeMode,
}

impl CellSpec {
    /// The stable cell identifier.
    pub(crate) fn id(self) -> String {
        format!(
            "occ{}-{}-{}-{}",
            self.occupancy,
            self.distribution.name(),
            self.operation.name(),
            self.merge.name()
        )
    }
}

/// A cell's measured status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CellStatus {
    /// Measured and at or below the owner threshold (`selected/incumbent <= 1.0`).
    Pass,
    /// Measured and above the owner threshold (`selected/incumbent > 1.0`).
    Fail,
    /// Measured, but the incumbent has no corresponding path. Retained in the status vocabulary;
    /// the growable contract's resource cells report `not-applicable` as a string, and no speed
    /// cell is not-applicable.
    #[allow(
        dead_code,
        reason = "retained status vocabulary; no growable speed cell is not-applicable"
    )]
    NotApplicable,
    /// Not measurable under this protocol; stays missing.
    Missing,
    /// The workload identity or oracle digest failed before timing.
    Rejected,
}

impl CellStatus {
    /// The status's report name.
    fn name(self) -> &'static str {
        match self {
            CellStatus::Pass => "pass",
            CellStatus::Fail => "fail",
            CellStatus::NotApplicable => "not-applicable",
            CellStatus::Missing => "missing",
            CellStatus::Rejected => "rejected",
        }
    }
}

/// One cell's identity, precheck record, samples and verdict.
pub(crate) struct CellSamples {
    pub(crate) cell_id: String,
    pub(crate) occupancy: usize,
    pub(crate) distribution: Distribution,
    pub(crate) operation: Operation,
    pub(crate) merge: MergeMode,
    pub(crate) population: usize,
    pub(crate) ops: usize,
    pub(crate) hit_ratio: f64,
    pub(crate) oracle_digest: u64,
    pub(crate) apeiron_digest_ok: bool,
    pub(crate) ht_digest_ok: bool,
    pub(crate) apeiron_admitted: usize,
    pub(crate) ht_len: usize,
    pub(crate) oracle_len: usize,
    pub(crate) ht_capacity: usize,
    pub(crate) ht_longest_probe: usize,
    pub(crate) apeiron_samples: Vec<u64>,
    pub(crate) ht_samples: Vec<u64>,
    pub(crate) campaign_ratios: Vec<f64>,
    pub(crate) status: CellStatus,
    pub(crate) reason: String,
}

/// The complete strata enumeration.
pub(crate) struct StrataWitness {
    pub(crate) cells: Vec<CellSamples>,
    /// The growable contract's genuine resource cells, separate from the speed cells.
    pub(crate) resources: Vec<ResourceCell>,
    pub(crate) profile: String,
    pub(crate) campaigns: u32,
    pub(crate) warmup: u32,
    pub(crate) samples: u32,
    pub(crate) ops: usize,
    pub(crate) measured: usize,
    pub(crate) missing: usize,
    pub(crate) rejected: usize,
    pub(crate) passed: usize,
    pub(crate) failed: usize,
    pub(crate) not_applicable: usize,
    pub(crate) ht_rev: String,
    pub(crate) rustc: String,
    pub(crate) cpu: String,
}

/// The generated key sets for one cell. Classes are disjoint; only the class the operation names
/// enters the timed trace.
pub(crate) struct KeyPlan {
    pub(crate) present: Vec<StrataKey>,
    pub(crate) absent: Vec<StrataKey>,
    pub(crate) fresh: Vec<StrataKey>,
}

/// A built operation trace.
pub(crate) struct Trace {
    pub(crate) lookups: Vec<StrataKey>,
    pub(crate) inserts: Vec<(StrataKey, i64)>,
    pub(crate) hit_ratio: f64,
}

/// One implementation's timed trace outcome.
struct RunOutcome {
    digest: u64,
    lookup_nanos: u64,
    insert_nanos: u64,
}

impl RunOutcome {
    /// The scheduled-op time: lookups plus inserts.
    fn total_nanos(&self) -> u64 {
        self.lookup_nanos.saturating_add(self.insert_nanos)
    }
}

/// The inverse of the odd hashtrinity multiplier modulo 2^64.
fn phi_inv() -> u64 {
    let mut inv = hashtrinity::PHI;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(hashtrinity::PHI.wrapping_mul(inv)));
    }
    inv
}

/// The hashtrinity home slot for `coord` at `bits`, mirroring the documented placement so the
/// clustered generator can aim keys at one home. The real table's `longest_probe` confirms it.
fn ht_home(coord: u64, capacity: usize, bits: u32) -> usize {
    if usize::try_from(coord).is_ok_and(|c| c < capacity) {
        usize::try_from(coord).expect("a coordinate below capacity fits usize")
    } else {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the shift leaves at most `bits` bits, and bits <= 63"
        )]
        {
            (coord.wrapping_mul(hashtrinity::PHI) >> (64 - bits)) as usize
        }
    }
}

/// The hashtrinity slot capacity for a population of `total` keys, and its bit width.
pub(crate) fn probe_capacity(total: usize) -> (usize, u32) {
    let table: Ht = Ht::with_capacity(total);
    let capacity = table.capacity();
    (capacity, capacity.trailing_zeros())
}

/// A direct-index coordinate: `shard` is the residue and `slot` is the direct slot.
fn direct_key(shard: usize, slot: usize) -> StrataKey {
    let shard = u64::try_from(shard).expect("a shard index below 256 fits u64");
    let slot = u64::try_from(slot).expect("a slot index below 64 fits u64");
    StrataKey {
        tag: shard + 256 * slot,
    }
}

/// An offset coordinate beyond any hashtrinity direct range at these populations.
fn offset_key(shard: usize, slot: usize) -> StrataKey {
    let shard = u64::try_from(shard).expect("a shard index below 256 fits u64");
    let slot = u64::try_from(slot).expect("a slot index below 64 fits u64");
    StrataKey {
        tag: OFFSET_BASE + shard + 256 * slot,
    }
}

/// A seeded-random coordinate beyond the direct range; distinct `index` values are distinct tags.
fn random_key(shard: usize, index: u64) -> StrataKey {
    let shard = u64::try_from(shard).expect("a shard index below 256 fits u64");
    let word = index
        .wrapping_add(STRATA_SEED)
        .wrapping_mul(RAND_ODD)
        .wrapping_add(shard)
        & RAND_MASK;
    StrataKey {
        tag: RAND_BASE + shard + 256 * word,
    }
}

/// A clustered coordinate with `shard`'s target hashtrinity home and residue `shard`.
///
/// `x = (home << (64 - bits)) | low` has top `bits` equal to the target home; `tag = x · PHI⁻¹`
/// then satisfies `tag · PHI ≡ x (mod 2^64)`, so hashtrinity's `home(tag)` is the target. The low
/// byte of `low` is `shard · PHI`, so `tag ≡ shard (mod 256)`. `mid` is nudged off zero so the
/// `shard = 0` / `mid = 0` case cannot produce the zero coordinate.
fn clustered_key(shard: usize, mid: u64, bits: u32) -> StrataKey {
    let shard64 = u64::try_from(shard).expect("a shard index below 256 fits u64");
    let home = shard64;
    let x_low = shard64.wrapping_mul(hashtrinity::PHI) & 0xFF;
    let free_bits = 64 - bits - 8;
    let free_mask = (1u64 << free_bits) - 1;
    let mid_masked = mid.wrapping_add(1).wrapping_mul(RAND_ODD) & free_mask;
    let low = x_low | (mid_masked << 8);
    let x = (home << (64 - bits)) | low;
    StrataKey {
        tag: x.wrapping_mul(phi_inv()),
    }
}

/// The class offset that keeps the three key classes disjoint for the random and clustered
/// distributions.
fn class_offset(class: u8) -> u64 {
    u64::from(class) * 1_000_000
}

/// A key for `distribution` at `shard`/`slot`, in key class `class` (0 present, 1 absent, 2 fresh).
fn key_for(
    distribution: Distribution,
    shard: usize,
    slot: usize,
    class: u8,
    bits: u32,
) -> StrataKey {
    match distribution {
        Distribution::DirectIndexResident => direct_key(shard, slot),
        Distribution::OffsetBeyondDirect => offset_key(shard, slot),
        Distribution::SeededRandom => {
            let index = class_offset(class)
                + u64::try_from(shard * 16 + slot).expect("a per-shard slot fits u64");
            random_key(shard, index)
        }
        Distribution::ClusteredSkewed => {
            let mid = class_offset(class)
                + u64::try_from(shard * 16 + slot).expect("a per-shard slot fits u64");
            clustered_key(shard, mid, bits)
        }
    }
}

/// Generate one key class, `per_shard` keys per shard, in `distribution`.
fn key_class(
    distribution: Distribution,
    occupancy: usize,
    per_shard: usize,
    class: u8,
    bits: u32,
    capacity: usize,
) -> Result<Vec<StrataKey>, String> {
    let mut keys = Vec::with_capacity(SHARDS * per_shard);
    for shard in 0..SHARDS {
        for slot in 0..per_shard {
            let key = key_for(distribution, shard, occupancy + slot, class, bits);
            if distribution.is_direct() {
                if key.tag >= u64::try_from(capacity).expect("a capacity fits u64") {
                    return Err(format!(
                        "direct key {} leaves the direct range below {capacity}",
                        key.tag
                    ));
                }
            } else if key.tag < u64::try_from(capacity).expect("a capacity fits u64") {
                return Err(format!(
                    "probing key {} falls inside the direct range below {capacity}",
                    key.tag
                ));
            }
            if key.tag >= u64::MAX - 1 {
                return Err(format!("key {} is the reserved sentinel range", key.tag));
            }
            if distribution == Distribution::ClusteredSkewed {
                let observed = ht_home(key.tag, capacity, bits);
                if observed != shard {
                    return Err(format!(
                        "clustered key {} homes to {observed}, not shard {shard}",
                        key.tag
                    ));
                }
            }
            keys.push(key);
        }
    }
    Ok(keys)
}

/// Build the present/absent/fresh key sets for one cell.
pub(crate) fn make_plan(spec: CellSpec, capacity: usize, bits: u32) -> Result<KeyPlan, String> {
    let occupancy = spec.occupancy;
    let direct_slots = capacity / SHARDS;
    if spec.distribution.is_direct() && occupancy > direct_slots {
        return Err(format!(
            "occupancy {occupancy} exceeds the {direct_slots} direct slots per shard"
        ));
    }
    let absent_per_shard = if spec.distribution.is_direct() {
        direct_slots - occupancy
    } else {
        APEIRON_N
    };
    let fresh_per_shard = if occupancy == APEIRON_N {
        if spec.distribution.is_direct() {
            direct_slots - occupancy
        } else {
            APEIRON_N
        }
    } else if spec.distribution.is_direct() {
        (APEIRON_N - occupancy).min(direct_slots - occupancy)
    } else {
        APEIRON_N - occupancy
    };
    let present = key_class(spec.distribution, 0, occupancy, 0, bits, capacity)?;
    let absent = key_class(
        spec.distribution,
        occupancy,
        absent_per_shard,
        1,
        bits,
        capacity,
    )?;
    let fresh = key_class(
        spec.distribution,
        occupancy,
        fresh_per_shard,
        2,
        bits,
        capacity,
    )?;
    Ok(KeyPlan {
        present,
        absent,
        fresh,
    })
}

/// Cycle `items` to `n` entries. `items` must be non-empty.
fn cycle<T: Copy>(items: &[T], n: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(items[i % items.len()]);
    }
    out
}

/// The deterministic update value for a schedule index.
fn update_value(index: usize) -> i64 {
    i64::try_from(index)
        .unwrap_or(i64::MAX)
        .wrapping_mul(3)
        .wrapping_add(7)
}

/// Build the timed trace for a cell.
pub(crate) fn build_trace(spec: CellSpec, plan: &KeyPlan) -> Trace {
    let mut lookups = Vec::new();
    let mut inserts = Vec::new();
    let hit_ratio = match spec.operation {
        Operation::PresentHit => {
            lookups = cycle(&plan.present, STRATA_OPS);
            1.0
        }
        Operation::AbsentMiss => {
            lookups = cycle(&plan.absent, STRATA_OPS);
            0.0
        }
        Operation::ExistingUpdate => {
            inserts = cycle(&plan.present, STRATA_OPS)
                .into_iter()
                .enumerate()
                .map(|(i, key)| (key, update_value(i)))
                .collect();
            1.0
        }
        Operation::AdmittedFreshInsert | Operation::GrowthInsert => {
            inserts = plan
                .fresh
                .iter()
                .enumerate()
                .map(|(i, &key)| (key, update_value(i)))
                .collect();
            0.0
        }
    };
    Trace {
        lookups,
        inserts,
        hit_ratio,
    }
}

/// The initial value of an admitted key. Nonzero, so an absent default is distinguishable.
pub(crate) fn initial_value(key: StrataKey) -> i64 {
    key.tag.cast_signed().wrapping_add(1)
}

/// Build an Apeiron table holding the admitted key set, returning the admitted count.
fn build_apeiron(keys: &[StrataKey], merge: MergeFn) -> (Apeiron, usize) {
    let mut table = Apeiron::new(0, RawStrataHome);
    let mut admitted = 0usize;
    for key in keys {
        if table.insert_with(&merge, *key, initial_value(*key)).is_ok() {
            admitted += 1;
        }
    }
    (table, admitted)
}

/// Build a hashtrinity table holding the admitted key set, returning its live length.
fn build_ht(keys: &[StrataKey], merge: MergeFn) -> (Ht, usize) {
    let mut table = Ht::with_capacity(keys.len());
    for key in keys {
        table.insert_with(*key, initial_value(*key), merge);
    }
    let len = table.len();
    (table, len)
}

/// Touch every admitted key so the table is resident before a timed trace.
fn touch_apeiron(table: &Apeiron, keys: &[StrataKey]) {
    let mut acc = 0i64;
    for key in keys {
        acc = acc.wrapping_add(*table.get(key));
    }
    black_box(acc);
}

/// Touch every admitted key so the table is resident before a timed trace.
fn touch_ht(table: &Ht, keys: &[StrataKey]) {
    let mut acc = 0i64;
    for key in keys {
        acc = acc.wrapping_add(table.get(key).copied().unwrap_or(0));
    }
    black_box(acc);
}

/// Run the timed schedule against Apeiron, then the untimed canonical readback.
fn measure_apeiron(
    table: &mut Apeiron,
    trace: &Trace,
    keys: &[StrataKey],
    merge: MergeFn,
) -> RunOutcome {
    let table = black_box(table);
    let mut hash = FNV_OFFSET;
    let start = Instant::now();
    for key in &trace.lookups {
        fold(&mut hash, *table.get(black_box(key)));
    }
    let lookup_nanos = nanos_since(start);
    let start = Instant::now();
    for (key, value) in &trace.inserts {
        let refused = table.insert_with(&merge, black_box(*key), black_box(*value));
        fold(&mut hash, i64::from(u8::from(refused.is_err())));
        fold(&mut hash, *value);
    }
    let insert_nanos = nanos_since(start);
    for key in keys {
        fold(&mut hash, *table.get(black_box(key)));
    }
    RunOutcome {
        digest: hash,
        lookup_nanos,
        insert_nanos,
    }
}

/// Run the timed schedule against hashtrinity, then the untimed canonical readback.
fn measure_ht(table: &mut Ht, trace: &Trace, keys: &[StrataKey], merge: MergeFn) -> RunOutcome {
    let table = black_box(table);
    let mut hash = FNV_OFFSET;
    let start = Instant::now();
    for key in &trace.lookups {
        fold(&mut hash, table.get(black_box(key)).copied().unwrap_or(0));
    }
    let lookup_nanos = nanos_since(start);
    let start = Instant::now();
    for (key, value) in &trace.inserts {
        table.insert_with(black_box(*key), black_box(*value), merge);
        fold(&mut hash, 0);
        fold(&mut hash, *value);
    }
    let insert_nanos = nanos_since(start);
    for key in keys {
        fold(&mut hash, table.get(black_box(key)).copied().unwrap_or(0));
    }
    RunOutcome {
        digest: hash,
        lookup_nanos,
        insert_nanos,
    }
}

/// The independent oracle digest for the growable contract, computed before any timing. The oracle
/// starts from the same populated state the implementations start from: every admitted key holds
/// its initial value in its home region. An absent insert always stores, matching both the growable
/// `Table` (which grows the region) and hashtrinity; there is no refusal branch.
fn oracle_digest(
    lookups: &[StrataKey],
    inserts: &[(StrataKey, i64)],
    keys: &[StrataKey],
    merge: MergeFn,
) -> u64 {
    let mut values: HashMap<StrataKey, i64> = HashMap::new();
    for key in keys {
        values.insert(*key, merge(initial_value(*key), 0));
    }
    let mut hash = FNV_OFFSET;
    for key in lookups {
        fold(&mut hash, values.get(key).copied().unwrap_or(0));
    }
    for (key, value) in inserts {
        let old = values.get(key).copied().unwrap_or(0);
        values.insert(*key, merge(*value, old));
        fold(&mut hash, 0);
        fold(&mut hash, *value);
    }
    for key in keys {
        fold(&mut hash, values.get(key).copied().unwrap_or(0));
    }
    hash
}

/// A cell that could not be measured; it stays missing.
fn missing_cell(spec: CellSpec, capacity: usize, reason: String) -> CellSamples {
    CellSamples {
        cell_id: spec.id(),
        occupancy: spec.occupancy,
        distribution: spec.distribution,
        operation: spec.operation,
        merge: spec.merge,
        population: 0,
        ops: 0,
        hit_ratio: 0.0,
        oracle_digest: 0,
        apeiron_digest_ok: false,
        ht_digest_ok: false,
        apeiron_admitted: 0,
        ht_len: 0,
        oracle_len: 0,
        ht_capacity: capacity,
        ht_longest_probe: 0,
        apeiron_samples: Vec::new(),
        ht_samples: Vec::new(),
        campaign_ratios: Vec::new(),
        status: CellStatus::Missing,
        reason,
    }
}

/// A cell rejected before timing: its workload identity or oracle digest failed.
fn rejected_cell(spec: CellSpec, plan: &KeyPlan, capacity: usize, reason: String) -> CellSamples {
    CellSamples {
        cell_id: spec.id(),
        occupancy: spec.occupancy,
        distribution: spec.distribution,
        operation: spec.operation,
        merge: spec.merge,
        population: plan.present.len(),
        ops: 0,
        hit_ratio: 0.0,
        oracle_digest: 0,
        apeiron_digest_ok: false,
        ht_digest_ok: false,
        apeiron_admitted: 0,
        ht_len: 0,
        oracle_len: 0,
        ht_capacity: capacity,
        ht_longest_probe: 0,
        apeiron_samples: Vec::new(),
        ht_samples: Vec::new(),
        campaign_ratios: Vec::new(),
        status: CellStatus::Rejected,
        reason,
    }
}

/// Run one strata cell: identity and oracle precheck, then the campaign samples.
#[allow(
    clippy::too_many_lines,
    reason = "one cell's identity, precheck, warmup, sampling and verdict are one unit"
)]
fn run_cell(spec: CellSpec, campaigns: u32) -> CellSamples {
    let total = SHARDS * spec.occupancy;
    let (capacity, bits) = probe_capacity(total);
    let plan = match make_plan(spec, capacity, bits) {
        Ok(plan) => plan,
        Err(reason) => return missing_cell(spec, capacity, reason),
    };
    if spec.operation == Operation::GrowthInsert && plan.fresh.is_empty() {
        return missing_cell(
            spec,
            capacity,
            "no fresh key remains in the direct range at occupancy 16".to_string(),
        );
    }
    let merge = merge_fn(spec.merge);

    // Identity: every present coordinate is distinct and the oracle admits exactly that set.
    let distinct = plan
        .present
        .iter()
        .map(|key| key.tag)
        .collect::<HashSet<u64>>();
    if distinct.len() != plan.present.len() {
        return rejected_cell(
            spec,
            &plan,
            capacity,
            format!(
                "present coordinates {} are not distinct ({} unique)",
                plan.present.len(),
                distinct.len()
            ),
        );
    }
    let oracle_len = distinct.len();
    let trace = build_trace(spec, &plan);

    // The oracle digest is computed before any timed sample. The growable table and hashtrinity
    // share the total insert semantics on these traces, so one oracle admits both.
    let oracle = oracle_digest(&trace.lookups, &trace.inserts, &plan.present, merge);

    // Precheck: build the real tables, run the trace once, and require the oracle digest.
    let (mut apeiron, admitted) = build_apeiron(&plan.present, merge);
    let (mut ht, ht_len) = build_ht(&plan.present, merge);
    let ht_capacity = ht.capacity();
    let ht_longest_probe = ht.longest_probe();
    if admitted != plan.present.len() {
        return rejected_cell(
            spec,
            &plan,
            capacity,
            format!("apeiron admitted {admitted} != {}", plan.present.len()),
        );
    }
    if ht_len != plan.present.len() {
        return rejected_cell(
            spec,
            &plan,
            capacity,
            format!("hashtrinity live length {ht_len} != {}", plan.present.len()),
        );
    }
    let pre_a = measure_apeiron(&mut apeiron, &trace, &plan.present, merge);
    let pre_h = measure_ht(&mut ht, &trace, &plan.present, merge);
    if pre_a.digest != oracle {
        return rejected_cell(
            spec,
            &plan,
            capacity,
            format!(
                "apeiron digest {:016x} != oracle {oracle:016x}",
                pre_a.digest
            ),
        );
    }
    if pre_h.digest != oracle {
        return rejected_cell(
            spec,
            &plan,
            capacity,
            format!(
                "hashtrinity digest {:016x} != oracle {oracle:016x}",
                pre_h.digest
            ),
        );
    }
    if spec.distribution.is_direct() && ht_longest_probe != 1 {
        return rejected_cell(
            spec,
            &plan,
            capacity,
            format!("direct-index cell has hashtrinity probe depth {ht_longest_probe} != 1"),
        );
    }

    let mut apeiron_samples = Vec::new();
    let mut ht_samples = Vec::new();
    let mut campaign_ratios = Vec::new();
    let mut apeiron_digest_ok = true;
    let mut ht_digest_ok = true;

    for _ in 0..campaigns {
        for _ in 0..STRATA_WARMUP {
            let (mut warm_ap, _) = build_apeiron(&plan.present, merge);
            let (mut warm_ht, _) = build_ht(&plan.present, merge);
            touch_apeiron(&warm_ap, &plan.present);
            touch_ht(&warm_ht, &plan.present);
            let _ = measure_apeiron(&mut warm_ap, &trace, &plan.present, merge);
            let _ = measure_ht(&mut warm_ht, &trace, &plan.present, merge);
        }
        let mut campaign_a = Vec::with_capacity(STRATA_SAMPLES as usize);
        let mut campaign_h = Vec::with_capacity(STRATA_SAMPLES as usize);
        for sample in 0..STRATA_SAMPLES {
            let (mut sample_ap, _) = build_apeiron(&plan.present, merge);
            let (mut sample_ht, _) = build_ht(&plan.present, merge);
            touch_apeiron(&sample_ap, &plan.present);
            touch_ht(&sample_ht, &plan.present);
            let (a_out, h_out) = if sample % 2 == 0 {
                let a = measure_apeiron(&mut sample_ap, &trace, &plan.present, merge);
                let h = measure_ht(&mut sample_ht, &trace, &plan.present, merge);
                (a, h)
            } else {
                let h = measure_ht(&mut sample_ht, &trace, &plan.present, merge);
                let a = measure_apeiron(&mut sample_ap, &trace, &plan.present, merge);
                (a, h)
            };
            apeiron_digest_ok &= a_out.digest == oracle;
            ht_digest_ok &= h_out.digest == oracle;
            campaign_a.push(a_out.total_nanos());
            campaign_h.push(h_out.total_nanos());
        }
        campaign_ratios.push(ratio(median(&campaign_a), median(&campaign_h)));
        apeiron_samples.extend_from_slice(&campaign_a);
        ht_samples.extend_from_slice(&campaign_h);
    }

    let pooled_ratio = ratio(median(&apeiron_samples), median(&ht_samples));
    let (status, reason) = if !apeiron_digest_ok || !ht_digest_ok {
        (
            CellStatus::Rejected,
            "a campaign sample digest disagreed with the precheck oracle".to_string(),
        )
    } else if pooled_ratio <= RATIO_LIMIT {
        (
            CellStatus::Pass,
            format!("ratio {pooled_ratio:.4} <= {RATIO_LIMIT:.2}"),
        )
    } else {
        (
            CellStatus::Fail,
            format!("ratio {pooled_ratio:.4} > {RATIO_LIMIT:.2}"),
        )
    };

    CellSamples {
        cell_id: spec.id(),
        occupancy: spec.occupancy,
        distribution: spec.distribution,
        operation: spec.operation,
        merge: spec.merge,
        population: plan.present.len(),
        ops: trace.lookups.len() + trace.inserts.len(),
        hit_ratio: trace.hit_ratio,
        oracle_digest: oracle,
        apeiron_digest_ok,
        ht_digest_ok,
        apeiron_admitted: admitted,
        ht_len,
        oracle_len,
        ht_capacity,
        ht_longest_probe,
        apeiron_samples,
        ht_samples,
        campaign_ratios,
        status,
        reason,
    }
}

/// Run the whole strata enumeration for `profile`.
pub(crate) fn run_strata(profile: Profile) -> StrataWitness {
    let campaigns = match profile {
        Profile::Full => FULL_CAMPAIGNS,
        Profile::Smoke => SMOKE_CAMPAIGNS,
    };
    let mut cells = Vec::new();
    for operation in Operation::ALL {
        for occupancy in OCCUPANCIES {
            if operation == Operation::AdmittedFreshInsert && occupancy == APEIRON_N {
                continue;
            }
            if operation == Operation::GrowthInsert && occupancy != APEIRON_N {
                continue;
            }
            for distribution in Distribution::ALL {
                for merge in MergeMode::ALL {
                    cells.push(run_cell(
                        CellSpec {
                            occupancy,
                            distribution,
                            operation,
                            merge,
                        },
                        campaigns,
                    ));
                }
            }
        }
    }
    let mut witness = StrataWitness {
        cells,
        resources: crate::matrix::run_resource_cells(),
        profile: profile.name().to_string(),
        campaigns,
        warmup: STRATA_WARMUP,
        samples: STRATA_SAMPLES,
        ops: STRATA_OPS,
        measured: 0,
        missing: 0,
        rejected: 0,
        passed: 0,
        failed: 0,
        not_applicable: 0,
        ht_rev: crate::PINNED_HASHTRINITY_REV.to_string(),
        rustc: crate::tool_version("rustc"),
        cpu: cpu_model(),
    };
    for cell in &witness.cells {
        match cell.status {
            CellStatus::Missing => witness.missing += 1,
            CellStatus::Rejected => witness.rejected += 1,
            CellStatus::Pass => {
                witness.measured += 1;
                witness.passed += 1;
            }
            CellStatus::Fail => {
                witness.measured += 1;
                witness.failed += 1;
            }
            CellStatus::NotApplicable => {
                witness.measured += 1;
                witness.not_applicable += 1;
            }
        }
    }
    witness
}

/// Print the whole strata enumeration.
#[allow(
    clippy::too_many_lines,
    reason = "one line per cell plus the shared protocol and the dimension summaries"
)]
pub(crate) fn print_strata(w: &StrataWitness) {
    let (head, dirty) = crate::git_head();
    println!(
        "strata_protocol profile={} apeiron_head={head} dirty={} hashtrinity_rev={} rustc=\"{}\" \
         cpu=\"{}\" campaigns={} warmup={} samples={} ops={} seed={:#018x} \
         oracle_timing=before_timing note=\"every strata cell line shares this protocol\"",
        w.profile,
        if dirty { "yes" } else { "no" },
        w.ht_rev,
        w.rustc,
        w.cpu,
        w.campaigns,
        w.warmup,
        w.samples,
        w.ops,
        STRATA_SEED
    );
    for cell in &w.cells {
        let ops = u32::try_from(cell.ops).unwrap_or(u32::MAX);
        if cell.status == CellStatus::Missing {
            println!(
                "strata cell={} occupancy={} distribution={} operation={} merge={} \
                 status=missing reason=\"{}\"",
                cell.cell_id,
                cell.occupancy,
                cell.distribution.name(),
                cell.operation.name(),
                cell.merge.name(),
                cell.reason
            );
            continue;
        }
        let a_med = median(&cell.apeiron_samples);
        let h_med = median(&cell.ht_samples);
        let campaign_ratios = cell
            .campaign_ratios
            .iter()
            .map(|r| format!("{r:.4}"))
            .collect::<Vec<_>>()
            .join(",");
        let (r_min, r_max) = interval(&cell.campaign_ratios);
        println!(
            "strata cell={} occupancy={} distribution={} operation={} merge={} seed={:#018x} \
             population={} simultaneous_population=true ops={} hit_ratio={:.3} oracle_digest={:016x} \
             apeiron_digest_ok={} ht_digest_ok={} apeiron_admitted={} ht_len={} oracle_len={} \
             ht_capacity={} ht_longest_probe={} apeiron_key_compares=MISSING \
             apeiron_median_ns={a_med} ht_median_ns={h_med} ratio={:.4} campaign_ratios=[{}] \
             ratio_interval=[{r_min:.4},{r_max:.4}] apeiron_samples_interval=[{},{}] \
             ht_samples_interval=[{},{}] apeiron_ns_per_op={:.4} ht_ns_per_op={:.4} \
             status={} reason=\"{}\"",
            cell.cell_id,
            cell.occupancy,
            cell.distribution.name(),
            cell.operation.name(),
            cell.merge.name(),
            STRATA_SEED,
            cell.population,
            cell.ops,
            cell.hit_ratio,
            cell.oracle_digest,
            cell.apeiron_digest_ok,
            cell.ht_digest_ok,
            cell.apeiron_admitted,
            cell.ht_len,
            cell.oracle_len,
            cell.ht_capacity,
            cell.ht_longest_probe,
            ratio(a_med, h_med),
            campaign_ratios,
            minimum(&cell.apeiron_samples),
            maximum(&cell.apeiron_samples),
            minimum(&cell.ht_samples),
            maximum(&cell.ht_samples),
            ns_per_op(a_med, ops),
            ns_per_op(h_med, ops),
            cell.status.name(),
            cell.reason
        );
        if cell.operation.is_read() {
            println!(
                "strata cell={} raw_apeiron_samples_ns={}",
                cell.cell_id,
                join(&cell.apeiron_samples)
            );
            println!(
                "strata cell={} raw_ht_samples_ns={}",
                cell.cell_id,
                join(&cell.ht_samples)
            );
        }
    }
    for cell in &w.resources {
        println!(
            "strata_resource cell={} operation={} status={} population={} ops={} observed=\"{}\" \
             reason=\"{}\"",
            cell.id,
            cell.operation,
            cell.status,
            cell.population,
            cell.ops,
            cell.observed,
            cell.reason
        );
    }
    let resource_na = w
        .resources
        .iter()
        .filter(|c| c.status == "not-applicable")
        .count();
    let resource_missing = w.resources.iter().filter(|c| c.status == "missing").count();
    println!(
        "strata_summary cells={} measured={} pass={} fail={} not_applicable={} \
         missing={} rejected={} resource_cells={} resource_not_applicable={resource_na} \
         resource_missing={resource_missing} campaigns={} warmup={} samples={} ops={} ht_rev={} \
         rustc=\"{}\" cpu=\"{}\"",
        w.cells.len(),
        w.measured,
        w.passed,
        w.failed,
        w.not_applicable,
        w.missing,
        w.rejected,
        w.resources.len(),
        w.campaigns,
        w.warmup,
        w.samples,
        w.ops,
        w.ht_rev,
        w.rustc,
        w.cpu
    );
    print_family(w, "distribution", |c| c.distribution.name());
    print_family(w, "operation", |c| c.operation.name());
    print_family(w, "occupancy", |c| occupancy_label(c.occupancy));
    print_family(w, "merge", |c| c.merge.name());
}

/// The occupancy's report label, borrowed from the cell's own copy.
fn occupancy_label(occupancy: usize) -> &'static str {
    match occupancy {
        1 => "1",
        2 => "2",
        4 => "4",
        8 => "8",
        _ => "16",
    }
}

/// The minimum and maximum of a ratio set.
fn interval(ratios: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &r in ratios {
        if r < lo {
            lo = r;
        }
        if r > hi {
            hi = r;
        }
    }
    if ratios.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        (lo, hi)
    }
}

/// Print, for every value of one dimension, the per-cell pass/fail tally against `<= 1.0` and the
/// worst applicable ratio.
fn print_family(w: &StrataWitness, dimension: &str, key_of: impl Fn(&CellSamples) -> &'static str) {
    let mut values: Vec<&'static str> = Vec::new();
    for cell in &w.cells {
        let value = key_of(cell);
        if !values.contains(&value) {
            values.push(value);
        }
    }
    for value in values {
        let mut worst: Option<(&CellSamples, f64)> = None;
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut not_applicable = 0usize;
        for cell in &w.cells {
            if key_of(cell) != value {
                continue;
            }
            match cell.status {
                CellStatus::Pass => passed += 1,
                CellStatus::Fail => failed += 1,
                CellStatus::NotApplicable => {
                    not_applicable += 1;
                    continue;
                }
                CellStatus::Missing | CellStatus::Rejected => continue,
            }
            let r = ratio(median(&cell.apeiron_samples), median(&cell.ht_samples));
            if worst.is_none_or(|(_, w)| r > w) {
                worst = Some((cell, r));
            }
        }
        match worst {
            Some((cell, r)) => println!(
                "strata_family dimension={dimension} value={value} pass={passed} fail={failed} \
                 not_applicable={not_applicable} worst_cell={} worst_ratio={r:.4}",
                cell.cell_id
            ),
            None => println!(
                "strata_family dimension={dimension} value={value} pass={passed} fail={failed} \
                 not_applicable={not_applicable} worst_cell=NONE"
            ),
        }
    }
}

/// The one strata validator. It fails closed on the cell schema, the sample sets, the precheck
/// record, the identity counts and the derived status.
///
/// # Errors
/// Returns a description of the first observation, count, sample or status that violates the
/// strata contract. A caller that observes `Ok(())` has an accepted enumeration, not REG-1
/// acceptance.
pub(crate) fn validate_strata(w: &StrataWitness) -> Result<(), String> {
    if w.cells.is_empty() {
        return Err("strata enumeration has no cells".to_string());
    }
    if w.campaigns == 0 || w.warmup == 0 || w.samples == 0 || w.ops == 0 {
        return Err("strata protocol has a zero count".to_string());
    }
    let mut measured = 0usize;
    let mut missing = 0usize;
    let rejected = 0usize;
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut not_applicable = 0usize;
    for cell in &w.cells {
        match cell.status {
            CellStatus::Missing => missing += 1,
            CellStatus::Rejected => {
                return Err(format!(
                    "strata cell {} rejected: {}",
                    cell.cell_id, cell.reason
                ));
            }
            CellStatus::Pass | CellStatus::Fail | CellStatus::NotApplicable => {
                measured += 1;
                validate_cell(cell, w)?;
                match cell.status {
                    CellStatus::Pass => passed += 1,
                    CellStatus::Fail => failed += 1,
                    _ => not_applicable += 1,
                }
            }
        }
    }
    if measured == 0 {
        return Err("strata enumeration measured no cell".to_string());
    }
    if w.measured != measured
        || w.missing != missing
        || w.rejected != rejected
        || w.passed != passed
        || w.failed != failed
        || w.not_applicable != not_applicable
    {
        return Err("strata summary counts do not match the cells".to_string());
    }
    let mut ids = HashSet::new();
    for cell in &w.resources {
        if cell.id.is_empty() || !ids.insert(cell.id.as_str()) {
            return Err(format!(
                "resource cell id {} is empty or duplicated",
                cell.id
            ));
        }
        if !matches!(cell.status.as_str(), "not-applicable" | "missing") {
            return Err(format!(
                "resource cell {} status {} outside not-applicable/missing",
                cell.id, cell.status
            ));
        }
        if cell.status == "not-applicable" && cell.population != "0" {
            return Err(format!(
                "resource cell {} is not-applicable but population {} != 0",
                cell.id, cell.population
            ));
        }
    }
    Ok(())
}

/// Validate one measured cell's protocol, precheck record and status.
fn validate_cell(cell: &CellSamples, w: &StrataWitness) -> Result<(), String> {
    let expected_samples = usize::try_from(w.campaigns)
        .unwrap_or(0)
        .saturating_mul(usize::try_from(w.samples).unwrap_or(0));
    if cell.apeiron_samples.len() != expected_samples || cell.ht_samples.len() != expected_samples {
        return Err(format!(
            "cell {} sample counts {}/{} != {expected_samples}",
            cell.cell_id,
            cell.apeiron_samples.len(),
            cell.ht_samples.len()
        ));
    }
    if cell.apeiron_samples.contains(&0) || cell.ht_samples.contains(&0) {
        return Err(format!(
            "cell {} has a zero-nanosecond sample",
            cell.cell_id
        ));
    }
    if !cell.apeiron_digest_ok || !cell.ht_digest_ok {
        return Err(format!("cell {} digest check failed", cell.cell_id));
    }
    if cell.ht_len != cell.population || cell.oracle_len != cell.population {
        return Err(format!(
            "cell {} identity lengths ht={} oracle={} population={}",
            cell.cell_id, cell.ht_len, cell.oracle_len, cell.population
        ));
    }
    if cell.apeiron_admitted != cell.population {
        return Err(format!(
            "cell {} apeiron admitted {} != population {}",
            cell.cell_id, cell.apeiron_admitted, cell.population
        ));
    }
    if cell.population == 0 || cell.ops == 0 {
        return Err(format!(
            "cell {} has a zero population or op count",
            cell.cell_id
        ));
    }
    if cell.ht_capacity < cell.population || !cell.ht_capacity.is_power_of_two() {
        return Err(format!(
            "cell {} hashtrinity capacity {} is not a power of two above population {}",
            cell.cell_id, cell.ht_capacity, cell.population
        ));
    }
    if cell.distribution.is_direct() && cell.ht_longest_probe != 1 {
        return Err(format!(
            "cell {} direct distribution has probe depth {} != 1",
            cell.cell_id, cell.ht_longest_probe
        ));
    }
    if cell.campaign_ratios.len() != usize::try_from(w.campaigns).unwrap_or(0)
        || cell.campaign_ratios.iter().any(|r| !r.is_finite())
    {
        return Err(format!(
            "cell {} campaign ratio count/values invalid",
            cell.cell_id
        ));
    }
    let a_med = median(&cell.apeiron_samples);
    let h_med = median(&cell.ht_samples);
    if a_med == 0 || h_med == 0 {
        return Err(format!("cell {} has a zero median", cell.cell_id));
    }
    if !(minimum(&cell.apeiron_samples) <= a_med && a_med <= maximum(&cell.apeiron_samples)) {
        return Err(format!(
            "cell {} apeiron min/median/max unordered",
            cell.cell_id
        ));
    }
    if !(minimum(&cell.ht_samples) <= h_med && h_med <= maximum(&cell.ht_samples)) {
        return Err(format!("cell {} ht min/median/max unordered", cell.cell_id));
    }
    let r = ratio(a_med, h_med);
    if !r.is_finite() {
        return Err(format!("cell {} ratio is not finite", cell.cell_id));
    }
    match cell.status {
        CellStatus::Pass if r > RATIO_LIMIT => Err(format!(
            "cell {} status pass but ratio {r:.4} > {RATIO_LIMIT:.2}",
            cell.cell_id
        )),
        CellStatus::Fail if r <= RATIO_LIMIT => Err(format!(
            "cell {} status fail but ratio {r:.4} <= {RATIO_LIMIT:.2}",
            cell.cell_id
        )),
        _ => Ok(()),
    }
}

/// The strata summary clause appended to the performance axis evidence.
pub(crate) fn report_clause(w: &StrataWitness) -> String {
    let worst = w
        .cells
        .iter()
        .filter(|c| matches!(c.status, CellStatus::Pass | CellStatus::Fail))
        .map(|c| {
            (
                c.cell_id.clone(),
                ratio(median(&c.apeiron_samples), median(&c.ht_samples)),
            )
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    match worst {
        Some((id, r)) => format!(
            "strata: cells={} measured={} pass={} fail={} not_applicable={} \
             missing={} rejected={} resource_cells={} campaigns={} samples={} ops={} ht_rev={} \
             rustc=\"{}\" worst_cell={id} worst_ratio={r:.4}",
            w.cells.len(),
            w.measured,
            w.passed,
            w.failed,
            w.not_applicable,
            w.missing,
            w.rejected,
            w.resources.len(),
            w.campaigns,
            w.samples,
            w.ops,
            w.ht_rev,
            w.rustc
        ),
        None => format!(
            "strata: cells={} measured={} missing={} rejected={} resource_cells={} \
             worst_cell=NONE",
            w.cells.len(),
            w.measured,
            w.missing,
            w.rejected,
            w.resources.len()
        ),
    }
}
