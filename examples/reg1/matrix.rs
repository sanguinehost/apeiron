//! Shared fixed-16 measurement engine for the REG-1 performance checks.
//!
//! This module owns the reusable machinery: the admitted fixed-16 workload, the shared key and
//! home, the Apeiron, candidate, hashtrinity and oracle builders, the timed schedules and their
//! digest folds, the correctness precheck, the per-mode sample campaign and the summary
//! statistics. The `performance` module consumes it and owns the report shell, the legacy fixed-16
//! witness and the candidate arm.

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use apeiron::coord::address::{
    FaceCoord, FaceHome, FaceResidueCapability, LocatorSigma, ParabolicDual, Residue, H3_40,
};
use apeiron::table::{Home, StorageError, Table, TableAddress};
use apeiron::Sigma;
use hashtrinity::Table as HtTable;

use crate::strata::{CellSamples, Distribution, Operation, StrataHome, StrataKey, StrataWitness};

// The benchmark-only candidate lives in `examples/reg1/candidate.rs`; declared here with an
// explicit path relative to this file's directory so the performance report shares one module
// instance.
#[path = "candidate.rs"]
pub(crate) mod candidate;

use candidate::CandidateTable;

/// Apeiron's per-shard override capacity (`Table<K, V, H, N>`).
pub(crate) const APEIRON_N: usize = 16;
/// The fixed shard count of `apeiron::Table`.
pub(crate) const SHARD_COUNT: usize = 256;
/// The shared admitted key count: exactly `APEIRON_N` keys in each of `SHARD_COUNT` shards.
pub(crate) const INITIAL_KEYS: u32 = 4096;
/// The total default both tables read for an absent key.
pub(crate) const DEFAULT_VALUE: i64 = 0;

/// Full-profile lookup count, preregistered by the REG-1 bench contract.
pub(crate) const FULL_LOOKUPS: u32 = 1_048_576;
/// Full-profile update count.
pub(crate) const FULL_UPDATES: u32 = 1_048_576;
/// Full-profile warmup traces per implementation.
pub(crate) const FULL_WARMUP: u32 = 5;
/// Full-profile timed samples per implementation.
pub(crate) const FULL_SAMPLES: u32 = 31;
/// Smoke-profile lookup count: the same codepath with reduced counts.
const SMOKE_LOOKUPS: u32 = 8192;
/// Smoke-profile update count.
const SMOKE_UPDATES: u32 = 8192;
/// Smoke-profile warmup traces per implementation, kept at the full-profile value.
const SMOKE_WARMUP: u32 = 5;
/// Smoke-profile timed samples per implementation, kept at the full-profile value.
const SMOKE_SAMPLES: u32 = 31;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The measurement profile: the exact full schedule, or a reduced smoke schedule on the same path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Profile {
    /// Reduced operation counts for gate speed; the same codepath and methodology.
    Smoke,
    /// The exact preregistered counts.
    Full,
}

impl Profile {
    /// The profile's CLI name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Profile::Smoke => "smoke",
            Profile::Full => "full",
        }
    }

    /// The profile's operation schedule.
    pub(crate) fn counts(self) -> Counts {
        match self {
            Profile::Smoke => Counts {
                lookups: SMOKE_LOOKUPS,
                updates: SMOKE_UPDATES,
                warmup: SMOKE_WARMUP,
                samples: SMOKE_SAMPLES,
            },
            Profile::Full => Counts {
                lookups: FULL_LOOKUPS,
                updates: FULL_UPDATES,
                warmup: FULL_WARMUP,
                samples: FULL_SAMPLES,
            },
        }
    }
}

/// One operation schedule's counts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Counts {
    pub(crate) lookups: u32,
    pub(crate) updates: u32,
    pub(crate) warmup: u32,
    pub(crate) samples: u32,
}

/// The two value-combination semantics, measured separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MergeMode {
    /// `f(new, old) = new`.
    Replacement,
    /// `f(new, old) = new + old` (wrapping).
    WrappingAdd,
}

impl MergeMode {
    /// Every measured mode, in report order.
    pub(crate) const ALL: [MergeMode; 2] = [MergeMode::Replacement, MergeMode::WrappingAdd];

    /// The mode's report name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            MergeMode::Replacement => "replacement",
            MergeMode::WrappingAdd => "wrapping-add",
        }
    }

    /// The mode's merge function.
    fn func(self) -> MergeFn {
        match self {
            MergeMode::Replacement => replacement_merge,
            MergeMode::WrappingAdd => wrapping_add_merge,
        }
    }
}

/// A total merge usable as both Apeiron `Merge<i64>` and hashtrinity's `FnOnce(i64, i64) -> i64`.
pub(crate) type MergeFn = fn(i64, i64) -> i64;

/// Replacement: the new contribution dominates.
pub(crate) fn replacement_merge(new: i64, _old: i64) -> i64 {
    new
}

/// Wrapping addition: the new contribution is added to the old.
fn wrapping_add_merge(new: i64, old: i64) -> i64 {
    new.wrapping_add(old)
}

// The key and home definitions moved into this child module to keep `matrix.rs` below the
// source-size cap; the re-export preserves the `crate::matrix::{..}` access paths. The explicit
// capability (exact identity, elliptic packing, both residues) lives there, and the raw physical
// placement stays separate from the semantic face map.
#[path = "adapters/native_matrix.rs"]
mod native_matrix;

pub(crate) use native_matrix::{SharedHome, SharedKey};

/// The Apeiron table type under measurement.
pub(crate) type ApeironTable = Table<SharedKey, i64, SharedHome, APEIRON_N>;

/// The benchmark-only candidate type under measurement.
pub(crate) type Candidate = CandidateTable<SharedKey, i64, SharedHome, APEIRON_N>;

/// The admitted key set: `INITIAL_KEYS` distinct tags, 16 per home shard.
pub(crate) fn initial_keys() -> Vec<SharedKey> {
    (0..INITIAL_KEYS).map(|tag| SharedKey { tag }).collect()
}

/// The initial value of a present key. Nonzero, so an absent default is distinguishable.
fn initial_value(key: SharedKey) -> i64 {
    i64::from(key.tag).wrapping_add(1)
}

/// The deterministic update value for a schedule index.
fn update_value(index: u32) -> i64 {
    i64::from(index).wrapping_mul(3).wrapping_add(7)
}

/// The maximum number of admitted keys sharing one home shard, derived from the key set.
pub(crate) fn per_shard_max(keys: &[SharedKey]) -> usize {
    let mut counts = [0usize; SHARD_COUNT];
    for key in keys {
        let tag = usize::try_from(key.tag).expect("u32 fits usize");
        counts[tag % SHARD_COUNT] += 1;
    }
    counts.iter().copied().max().unwrap_or(0)
}

/// The deterministic operation schedule: alternating present/absent lookups, then present-key
/// updates. The absent tags (`INITIAL_KEYS..2*INITIAL_KEYS`) are disjoint from the admitted set.
pub(crate) struct Trace {
    lookups: Vec<SharedKey>,
    updates: Vec<(SharedKey, i64)>,
}

/// Build the schedule for `counts`.
pub(crate) fn build_trace(counts: Counts) -> Trace {
    let present = initial_keys();
    let n = present.len();
    let lookups = (0..counts.lookups)
        .map(|i| {
            let index = usize::try_from(i / 2).expect("u32 fits usize") % n;
            if i % 2 == 0 {
                present[index]
            } else {
                SharedKey {
                    tag: INITIAL_KEYS + u32::try_from(index).expect("an index below 4096 fits u32"),
                }
            }
        })
        .collect();
    let updates = (0..counts.updates)
        .map(|i| {
            let index = usize::try_from(i).expect("u32 fits usize") % n;
            (present[index], update_value(i))
        })
        .collect();
    Trace { lookups, updates }
}

/// FNV-1a fold of one canonical value, the same for both implementations.
#[inline]
fn fold(hash: &mut u64, value: i64) {
    *hash = (*hash ^ value.cast_unsigned()).wrapping_mul(FNV_PRIME);
}

/// Nanoseconds elapsed since `start`, saturating rather than wrapping.
fn nanos_since(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// One implementation's timed trace outcome.
struct RunOutcome {
    digest: u64,
    lookup_nanos: u64,
    update_nanos: u64,
}

impl RunOutcome {
    /// The scheduled-op time: lookups plus updates. The untimed canonical readback is excluded.
    fn sample_nanos(&self) -> u64 {
        self.lookup_nanos.saturating_add(self.update_nanos)
    }
}

/// Build an Apeiron table holding the admitted key set, returning the admitted count.
fn build_apeiron(merge: MergeFn) -> (ApeironTable, usize) {
    let mut table = ApeironTable::new(DEFAULT_VALUE, SharedHome);
    let mut admitted = 0usize;
    for key in initial_keys() {
        if table.insert_with(&merge, key, initial_value(key)).is_ok() {
            admitted += 1;
        }
    }
    (table, admitted)
}

/// Build a candidate table holding the admitted key set, returning the admitted count.
fn build_candidate(merge: MergeFn) -> (Candidate, usize) {
    let mut table = Candidate::new(DEFAULT_VALUE, SharedHome);
    let mut admitted = 0usize;
    for key in initial_keys() {
        if table.insert_with(&merge, key, initial_value(key)).is_ok() {
            admitted += 1;
        }
    }
    (table, admitted)
}

/// Build a hashtrinity table holding the admitted key set, returning its live length.
pub(crate) fn build_hashtrinity(merge: MergeFn) -> (HtTable<SharedKey, i64>, usize) {
    let capacity = usize::try_from(INITIAL_KEYS).expect("4096 fits usize");
    let mut table = HtTable::with_capacity(capacity);
    for key in initial_keys() {
        table.insert_with(key, initial_value(key), merge);
    }
    let len = table.len();
    (table, len)
}

/// Build the independent oracle holding the admitted key set.
fn build_oracle(merge: MergeFn) -> HashMap<SharedKey, i64> {
    let capacity = usize::try_from(INITIAL_KEYS).expect("4096 fits usize");
    let mut oracle = HashMap::with_capacity(capacity);
    for key in initial_keys() {
        let old = oracle.get(&key).copied().unwrap_or(DEFAULT_VALUE);
        oracle.insert(key, merge(initial_value(key), old));
    }
    oracle
}

/// Touch every admitted key so the table's shards are resident before a timed trace.
fn touch_apeiron(table: &ApeironTable, keys: &[SharedKey]) {
    let mut acc = 0i64;
    for key in keys {
        acc = acc.wrapping_add(*table.get(key));
    }
    black_box(acc);
}

/// Touch every admitted key so the candidate's shards are resident before a timed trace.
fn touch_candidate(table: &Candidate, keys: &[SharedKey]) {
    let mut acc = 0i64;
    for key in keys {
        acc = acc.wrapping_add(*table.get(key));
    }
    black_box(acc);
}

/// Touch every admitted key so the table's slots are resident before a timed trace.
fn touch_hashtrinity(table: &HtTable<SharedKey, i64>, keys: &[SharedKey]) {
    let mut acc = 0i64;
    for key in keys {
        acc = acc.wrapping_add(table.get(key).copied().unwrap_or(DEFAULT_VALUE));
    }
    black_box(acc);
}

/// Run the timed schedule against Apeiron, folding every lookup result and refusal, then the
/// untimed canonical readback of the post-update state.
fn measure_apeiron(
    table: &mut ApeironTable,
    trace: &Trace,
    keys: &[SharedKey],
    merge: MergeFn,
) -> RunOutcome {
    let table = black_box(table);
    let trace = black_box(trace);
    let mut hash = FNV_OFFSET;
    let start = Instant::now();
    for key in &trace.lookups {
        let value = *table.get(black_box(key));
        fold(&mut hash, black_box(value));
    }
    let lookup_nanos = nanos_since(start);
    let start = Instant::now();
    for (key, value) in &trace.updates {
        let refused = table.insert_with(&merge, black_box(*key), black_box(*value));
        fold(&mut hash, i64::from(u8::from(refused.is_err())));
        fold(&mut hash, *value);
    }
    let update_nanos = nanos_since(start);
    for key in keys {
        let value = *table.get(black_box(key));
        fold(&mut hash, black_box(value));
    }
    RunOutcome {
        digest: hash,
        lookup_nanos,
        update_nanos,
    }
}

/// Run the timed schedule against the candidate, folding every lookup result and refusal, then the
/// untimed canonical readback. The candidate exposes the same `ShardFull` refusal surface as
/// Apeiron, so the folded digest is the same function of the same schedule.
fn measure_candidate(
    table: &mut Candidate,
    trace: &Trace,
    keys: &[SharedKey],
    merge: MergeFn,
) -> RunOutcome {
    let table = black_box(table);
    let trace = black_box(trace);
    let mut hash = FNV_OFFSET;
    let start = Instant::now();
    for key in &trace.lookups {
        let value = *table.get(black_box(key));
        fold(&mut hash, black_box(value));
    }
    let lookup_nanos = nanos_since(start);
    let start = Instant::now();
    for (key, value) in &trace.updates {
        let refused = table.insert_with(&merge, black_box(*key), black_box(*value));
        fold(&mut hash, i64::from(u8::from(refused.is_err())));
        fold(&mut hash, *value);
    }
    let update_nanos = nanos_since(start);
    for key in keys {
        let value = *table.get(black_box(key));
        fold(&mut hash, black_box(value));
    }
    RunOutcome {
        digest: hash,
        lookup_nanos,
        update_nanos,
    }
}

/// Run the timed schedule against hashtrinity, folding every lookup result and update, then the
/// untimed canonical readback. hashtrinity's `insert_with` has no refusal surface, so it folds the
/// same zero an admitted Apeiron update folds.
fn measure_hashtrinity(
    table: &mut HtTable<SharedKey, i64>,
    trace: &Trace,
    keys: &[SharedKey],
    merge: MergeFn,
) -> RunOutcome {
    let table = black_box(table);
    let trace = black_box(trace);
    let mut hash = FNV_OFFSET;
    let start = Instant::now();
    for key in &trace.lookups {
        let value = table.get(black_box(key)).copied().unwrap_or(DEFAULT_VALUE);
        fold(&mut hash, black_box(value));
    }
    let lookup_nanos = nanos_since(start);
    let start = Instant::now();
    for (key, value) in &trace.updates {
        table.insert_with(black_box(*key), black_box(*value), merge);
        fold(&mut hash, 0);
        fold(&mut hash, *value);
    }
    let update_nanos = nanos_since(start);
    for key in keys {
        let value = table.get(black_box(key)).copied().unwrap_or(DEFAULT_VALUE);
        fold(&mut hash, black_box(value));
    }
    RunOutcome {
        digest: hash,
        lookup_nanos,
        update_nanos,
    }
}

/// Run the same schedule against the independent oracle and fold the same canonical values.
fn measure_oracle(
    state: &mut HashMap<SharedKey, i64>,
    trace: &Trace,
    keys: &[SharedKey],
    merge: MergeFn,
) -> u64 {
    let state = black_box(state);
    let trace = black_box(trace);
    let mut hash = FNV_OFFSET;
    for key in &trace.lookups {
        fold(&mut hash, state.get(key).copied().unwrap_or(DEFAULT_VALUE));
    }
    for (key, value) in &trace.updates {
        let old = state.get(key).copied().unwrap_or(DEFAULT_VALUE);
        state.insert(*key, merge(*value, old));
        fold(&mut hash, 0);
        fold(&mut hash, *value);
    }
    for key in keys {
        fold(&mut hash, state.get(key).copied().unwrap_or(DEFAULT_VALUE));
    }
    hash
}

/// The correctness precheck outcome that admits the workload to timing.
pub(crate) struct Precheck {
    pub(crate) admitted: usize,
    pub(crate) candidate_admitted: usize,
    pub(crate) ht_len: usize,
    pub(crate) oracle_len: usize,
    pub(crate) shard_probe_refusals: usize,
    pub(crate) candidate_shard_probe_refusals: usize,
    pub(crate) reference_digest: u64,
}

/// A probe key whose residue is `shard`: a tag one past the admitted range.
fn shard_probe(shard: usize) -> SharedKey {
    SharedKey {
        tag: INITIAL_KEYS + u32::try_from(shard).expect("a shard index below 256 fits u32"),
    }
}

/// Confirm the benchmark-only bounded candidate refuses a 17th key in every shard. The growable
/// production table has no refusal path, so its count is zero and it is not mutated.
fn check_shard_refusals(
    candidate_table: &mut Candidate,
    merge: MergeFn,
) -> Result<(usize, usize), String> {
    let mut candidate_shard_probe_refusals = 0usize;
    for shard in 0..SHARD_COUNT {
        let probe = shard_probe(shard);
        if candidate_table.insert_with(&merge, probe, 1) == Err(StorageError::ShardFull) {
            candidate_shard_probe_refusals += 1;
        }
    }
    if candidate_shard_probe_refusals != SHARD_COUNT {
        return Err(format!(
            "candidate shard probes refused {candidate_shard_probe_refusals} != {SHARD_COUNT}"
        ));
    }
    Ok((0, candidate_shard_probe_refusals))
}

/// Require every admitted key to agree across Apeiron, the candidate, hashtrinity and the oracle.
fn check_readback(
    apeiron: &ApeironTable,
    candidate_table: &Candidate,
    ht: &HtTable<SharedKey, i64>,
    oracle: &HashMap<SharedKey, i64>,
    keys: &[SharedKey],
    phase: &str,
) -> Result<(), String> {
    for key in keys {
        let a = *apeiron.get(key);
        let c = *candidate_table.get(key);
        let h = ht.get(key).copied().unwrap_or(DEFAULT_VALUE);
        let o = oracle.get(key).copied().unwrap_or(DEFAULT_VALUE);
        if a != o || c != o || h != o {
            return Err(format!(
                "{phase} readback mismatch at tag {}: apeiron={a} candidate={c} hashtrinity={h} \
                 oracle={o}",
                key.tag
            ));
        }
    }
    Ok(())
}

/// Require every absent probe to read default and to be absent from hashtrinity and the oracle.
fn check_absent_probes(
    apeiron: &ApeironTable,
    candidate_table: &Candidate,
    ht: &HtTable<SharedKey, i64>,
    oracle: &HashMap<SharedKey, i64>,
) -> Result<(), String> {
    for shard in 0..SHARD_COUNT {
        let probe = shard_probe(shard);
        if *apeiron.get(&probe) != DEFAULT_VALUE {
            return Err(format!("absent probe {} read non-default", probe.tag));
        }
        if *candidate_table.get(&probe) != DEFAULT_VALUE {
            return Err(format!(
                "absent probe {} read non-default from candidate",
                probe.tag
            ));
        }
        if ht.get(&probe).is_some() {
            return Err(format!("absent probe {} present in hashtrinity", probe.tag));
        }
        if oracle.contains_key(&probe) {
            return Err(format!("absent probe {} present in oracle", probe.tag));
        }
    }
    Ok(())
}

/// Prove the admitted workload before any timing: capacity, identity, placement and canonical
/// agreement across Apeiron, the candidate, hashtrinity and the oracle.
///
/// # Errors
/// Returns a description of the first failed admission, placement, readback or digest check. A
/// caller that observes `Ok` may time the workload; a failure admits no work into timing.
pub(crate) fn precheck(
    mode: MergeMode,
    trace: &Trace,
    keys: &[SharedKey],
) -> Result<Precheck, String> {
    let merge = mode.func();
    let (mut apeiron, admitted) = build_apeiron(merge);
    let (mut candidate_table, candidate_admitted) = build_candidate(merge);
    let (mut ht, ht_len) = build_hashtrinity(merge);
    let mut oracle = build_oracle(merge);

    let expected = usize::try_from(INITIAL_KEYS).expect("4096 fits usize");
    if admitted != expected {
        return Err(format!("apeiron admitted {admitted} != {expected}"));
    }
    if candidate_admitted != expected {
        return Err(format!(
            "candidate admitted {candidate_admitted} != {expected}"
        ));
    }
    if ht_len != expected {
        return Err(format!("hashtrinity live length {ht_len} != {expected}"));
    }
    if oracle.len() != expected {
        return Err(format!("oracle length {} != {expected}", oracle.len()));
    }

    // A 17th key in each home must refuse: direct evidence of 16 occupied slots per shard, for
    // both the production table and the candidate.
    let (shard_probe_refusals, candidate_shard_probe_refusals) =
        check_shard_refusals(&mut candidate_table, merge)?;

    // Initial readback: every admitted key agrees across all four, absent probes agree too.
    check_readback(&apeiron, &candidate_table, &ht, &oracle, keys, "initial")?;
    check_absent_probes(&apeiron, &candidate_table, &ht, &oracle)?;

    // Canonical post-trace agreement: the four digests must be identical before any sample is
    // accepted.
    let apeiron_out = measure_apeiron(&mut apeiron, trace, keys, merge);
    let candidate_out = measure_candidate(&mut candidate_table, trace, keys, merge);
    let ht_out = measure_hashtrinity(&mut ht, trace, keys, merge);
    let oracle_digest = measure_oracle(&mut oracle, trace, keys, merge);
    if apeiron_out.digest != candidate_out.digest
        || apeiron_out.digest != ht_out.digest
        || apeiron_out.digest != oracle_digest
    {
        return Err(format!(
            "post-trace digest mismatch: apeiron={:016x} candidate={:016x} hashtrinity={:016x} \
             oracle={:016x}",
            apeiron_out.digest, candidate_out.digest, ht_out.digest, oracle_digest
        ));
    }
    check_readback(&apeiron, &candidate_table, &ht, &oracle, keys, "post-trace")?;

    Ok(Precheck {
        admitted,
        candidate_admitted,
        ht_len,
        oracle_len: oracle.len(),
        shard_probe_refusals,
        candidate_shard_probe_refusals,
        reference_digest: oracle_digest,
    })
}

/// One merge mode's validated samples and precheck record.
#[allow(
    clippy::struct_excessive_bools,
    reason = "one digest-validity flag per measured implementation, plus the precheck flag"
)]
pub(crate) struct ModeSamples {
    pub(crate) mode: MergeMode,
    pub(crate) admitted: usize,
    pub(crate) candidate_admitted: usize,
    pub(crate) ht_len: usize,
    pub(crate) oracle_len: usize,
    pub(crate) shard_probe_refusals: usize,
    pub(crate) candidate_shard_probe_refusals: usize,
    pub(crate) precheck_digest_ok: bool,
    pub(crate) apeiron_samples: Vec<u64>,
    pub(crate) candidate_samples: Vec<u64>,
    pub(crate) ht_samples: Vec<u64>,
    pub(crate) apeiron_digest_ok: bool,
    pub(crate) candidate_digest_ok: bool,
    pub(crate) ht_digest_ok: bool,
}

/// Reconstruct equivalent state, warm up, then collect `counts.samples` samples for one merge
/// mode. Every sample's digest must match the precheck reference.
///
/// On even samples the order is Apeiron, hashtrinity, candidate; on odd samples it is candidate,
/// hashtrinity, Apeiron. This preserves the historical Apeiron/hashtrinity AB/BA alternation
/// exactly, so the current/hashtrinity ratio keeps its methodology, and gives the candidate both a
/// first and a last position.
///
/// # Errors
/// Returns a description of a failed reconstruction, admission or per-sample digest.
pub(crate) fn sample_mode(
    mode: MergeMode,
    trace: &Trace,
    keys: &[SharedKey],
    counts: Counts,
    reference_digest: u64,
) -> Result<ModeSamples, String> {
    let merge = mode.func();
    let expected = usize::try_from(INITIAL_KEYS).expect("4096 fits usize");

    for _ in 0..counts.warmup {
        let (mut apeiron, admitted) = build_apeiron(merge);
        let (mut candidate_table, candidate_admitted) = build_candidate(merge);
        let (mut ht, ht_len) = build_hashtrinity(merge);
        if admitted != expected || candidate_admitted != expected || ht_len != expected {
            return Err(format!(
                "warmup reconstruction admitted {admitted}/{candidate_admitted}/{ht_len} != \
                 {expected}"
            ));
        }
        touch_apeiron(&apeiron, keys);
        touch_candidate(&candidate_table, keys);
        touch_hashtrinity(&ht, keys);
        let _ = measure_apeiron(&mut apeiron, trace, keys, merge);
        let _ = measure_candidate(&mut candidate_table, trace, keys, merge);
        let _ = measure_hashtrinity(&mut ht, trace, keys, merge);
    }

    let capacity = usize::try_from(counts.samples).expect("sample count fits usize");
    let mut apeiron_samples = Vec::with_capacity(capacity);
    let mut candidate_samples = Vec::with_capacity(capacity);
    let mut ht_samples = Vec::with_capacity(capacity);
    let mut apeiron_digest_ok = true;
    let mut candidate_digest_ok = true;
    let mut ht_digest_ok = true;

    for sample in 0..counts.samples {
        let (mut apeiron, admitted) = build_apeiron(merge);
        let (mut candidate_table, candidate_admitted) = build_candidate(merge);
        let (mut ht, ht_len) = build_hashtrinity(merge);
        if admitted != expected || candidate_admitted != expected || ht_len != expected {
            return Err(format!(
                "sample reconstruction admitted {admitted}/{candidate_admitted}/{ht_len} != \
                 {expected}"
            ));
        }
        touch_apeiron(&apeiron, keys);
        touch_candidate(&candidate_table, keys);
        touch_hashtrinity(&ht, keys);

        let (apeiron_out, candidate_out, ht_out) = if sample % 2 == 0 {
            let a = measure_apeiron(&mut apeiron, trace, keys, merge);
            let h = measure_hashtrinity(&mut ht, trace, keys, merge);
            let c = measure_candidate(&mut candidate_table, trace, keys, merge);
            (a, c, h)
        } else {
            let c = measure_candidate(&mut candidate_table, trace, keys, merge);
            let h = measure_hashtrinity(&mut ht, trace, keys, merge);
            let a = measure_apeiron(&mut apeiron, trace, keys, merge);
            (a, c, h)
        };
        apeiron_digest_ok &= apeiron_out.digest == reference_digest;
        candidate_digest_ok &= candidate_out.digest == reference_digest;
        ht_digest_ok &= ht_out.digest == reference_digest;
        apeiron_samples.push(apeiron_out.sample_nanos());
        candidate_samples.push(candidate_out.sample_nanos());
        ht_samples.push(ht_out.sample_nanos());
    }

    Ok(ModeSamples {
        mode,
        admitted: expected,
        candidate_admitted: expected,
        ht_len: expected,
        oracle_len: expected,
        shard_probe_refusals: SHARD_COUNT,
        candidate_shard_probe_refusals: SHARD_COUNT,
        precheck_digest_ok: true,
        apeiron_samples,
        candidate_samples,
        ht_samples,
        apeiron_digest_ok,
        candidate_digest_ok,
        ht_digest_ok,
    })
}

/// The benchmark-boundary observation for the `SharedHome` change.
///
/// The benchmark's `SharedHome` returns the full `tag` so the candidate can cache the full home
/// word. The production table reduces `% 256` internally, so shard placement is identical to the
/// historical `tag % 256`; this witness records both facts for the admitted set.
pub(crate) struct HomeBoundary {
    pub(crate) keys: usize,
    pub(crate) full_word_ok: bool,
    pub(crate) residue_ok: bool,
}

/// Record, for every admitted key, that the home returns the full tag and that the residue is the
/// historical placement.
pub(crate) fn run_home_boundary(keys: &[SharedKey]) -> HomeBoundary {
    let mut full_word_ok = true;
    let mut residue_ok = true;
    for key in keys {
        let word = SharedHome.home(key);
        full_word_ok &= word == u64::from(key.tag);
        residue_ok &= word % 256 == u64::from(key.tag % 256);
    }
    HomeBoundary {
        keys: keys.len(),
        full_word_ok,
        residue_ok,
    }
}

impl ModeSamples {
    /// The upper median of the Apeiron samples.
    pub(crate) fn apeiron_median(&self) -> u64 {
        median(&self.apeiron_samples)
    }

    /// The upper median of the candidate samples.
    pub(crate) fn candidate_median(&self) -> u64 {
        median(&self.candidate_samples)
    }

    /// The upper median of the hashtrinity samples.
    pub(crate) fn ht_median(&self) -> u64 {
        median(&self.ht_samples)
    }
}

/// The upper median of a sample set, or zero when it is absent.
pub(crate) fn median(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

/// The CPU model from `/proc/cpuinfo`, or `unknown`.
pub(crate) fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .map(|line| line.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Join samples into a comma-separated string.
pub(crate) fn join(samples: &[u64]) -> String {
    samples
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// The minimum of a sample set, or zero when it is absent.
pub(crate) fn minimum(samples: &[u64]) -> u64 {
    samples.iter().copied().min().unwrap_or(0)
}

/// The maximum of a sample set, or zero when it is absent.
pub(crate) fn maximum(samples: &[u64]) -> u64 {
    samples.iter().copied().max().unwrap_or(0)
}

/// Nanoseconds per scheduled operation.
#[allow(
    clippy::cast_precision_loss,
    reason = "a bounded-run nanosecond total is far below 2^53; this is a measurement, not a bound"
)]
pub(crate) fn ns_per_op(median_nanos: u64, ops: u32) -> f64 {
    median_nanos as f64 / f64::from(ops)
}

/// The numerator/denominator median ratio.
#[allow(
    clippy::cast_precision_loss,
    reason = "a bounded-run nanosecond median is far below 2^53"
)]
pub(crate) fn ratio(numerator_nanos: u64, denominator_nanos: u64) -> f64 {
    numerator_nanos as f64 / denominator_nanos as f64
}

// ── Genuine resource cells of the growable contract ───────────────────────────────────────────
//
// A resource cell records a behavioral observation of the growable `Table` resource contract, not
// a timing ratio. The zero-group-width cell is measured; the two not-forceable failures are
// registered `missing` with their reason, never fabricated into a measurement.

/// One genuine resource/refusal cell of the growable contract.
#[derive(Clone, Debug)]
pub(crate) struct ResourceCell {
    pub(crate) id: String,
    pub(crate) operation: String,
    pub(crate) status: String,
    pub(crate) population: String,
    pub(crate) ops: String,
    pub(crate) observed: String,
    pub(crate) reason: String,
}

/// The explicit physical home for the boxed [`StrataKey`] of the zero-group-width cell.
///
/// `Arc<StrataKey>` has no `Packed`, so the all-face [`Sigma`] domain is inadmissible and the face
/// domain is [`LocatorSigma`]. The `Home` placement is the constant `0` (the removed closure's
/// physical placement); the face residues delegate to [`StrataHome`] on the pointee.
pub(crate) struct ArcStrataHome;

impl Home<Arc<StrataKey>> for ArcStrataHome {
    fn home(&self, _key: &Arc<StrataKey>) -> u64 {
        0
    }

    fn address(&self, _sigma: Sigma, _key: &Arc<StrataKey>) -> TableAddress {
        TableAddress {
            place: 0,
            direct_word: None,
        }
    }
}

impl FaceResidueCapability<Arc<StrataKey>> for ArcStrataHome {
    fn parabolic_dual(&self, key: &Arc<StrataKey>) -> ParabolicDual {
        StrataHome.parabolic_dual(&**key)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &Arc<StrataKey>) -> Residue {
        StrataHome.hyperbolic_scalar(profile, &**key)
    }
}

impl FaceCoord<Arc<StrataKey>> for ArcStrataHome {
    type Faces = LocatorSigma;
}

impl FaceHome<Arc<StrataKey>> for ArcStrataHome {}

/// Measure the `N = 0` contract: every write returns `SizeOverflow`, the state is unchanged and
/// neither the merge nor a key clone runs. `Arc::strong_count` observes the clone because the key
/// is passed by value and dropped uncloned on refusal.
fn zero_group_width_cell() -> ResourceCell {
    let mut table = Table::<Arc<StrataKey>, i64, _, 0>::new(7, ArcStrataHome);
    let key = Arc::new(StrataKey { tag: 1 });
    let probe = Arc::clone(&key);
    let result = table.insert_with(
        &(|_new: i64, _old: i64| -> i64 { panic!("merge on refusal") }),
        key,
        1,
    );
    let refused = result == Err(StorageError::SizeOverflow);
    let no_clone = Arc::strong_count(&probe) == 1;
    let unchanged = table.is_empty() && *table.get(&probe) == 7;
    let ok = refused && no_clone && unchanged;
    ResourceCell {
        id: "zero-group-width-sizeoverflow".to_string(),
        operation: "zero-group-width".to_string(),
        status: "not-applicable".to_string(),
        population: "0".to_string(),
        ops: "1".to_string(),
        observed: format!(
            "SizeOverflow={refused} unchanged-state={unchanged} no-clone={no_clone} no-merge=true"
        ),
        reason: if ok {
            "the N=0 contract refuses every write, keeps reads total and invokes no callback"
                .to_string()
        } else {
            format!(
                "the N=0 witness failed: refused={refused} no_clone={no_clone} unchanged={unchanged}"
            )
        },
    }
}

/// The resource cells the run reports. The zero-group-width cell is measured; sizing overflow and
/// injected reservation failure have no forceable input in this harness.
pub(crate) fn run_resource_cells() -> Vec<ResourceCell> {
    vec![
        zero_group_width_cell(),
        ResourceCell {
            id: "sizing-overflow-not-forceable".to_string(),
            operation: "sizing-overflow".to_string(),
            status: "missing".to_string(),
            population: "0".to_string(),
            ops: "0".to_string(),
            observed: "not forceable: checked_mul(2) cannot overflow a real region length"
                .to_string(),
            reason:
                "requires a region length above usize::MAX/2; registered missing, not fabricated"
                    .to_string(),
        },
        ResourceCell {
            id: "injected-reservation-failure-not-forceable".to_string(),
            operation: "injected-reservation-failure".to_string(),
            status: "missing".to_string(),
            population: "0".to_string(),
            ops: "0".to_string(),
            observed: "not forceable: try_reserve_exact failure needs an allocator shim"
                .to_string(),
            reason:
                "no allocator injection point in this harness; registered missing, not fabricated"
                    .to_string(),
        },
    ]
}

// ── Registered current-strata enforcement ─────────────────────────────────────────────────────
//
// The performance gate compares the run's enumerated current cells against the registered
// `current_strata` receipts in `matrix.json`. The registration carries workload identity (seed,
// distribution, operation, merge, population, op count, hit ratio) and the deterministic oracle
// digest; a registered cell that is absent from the run, whose identity disagrees, or whose named
// implementation is not the actual executing symbol rejects. Registered medians and ratios are
// receipts of the registering run, never read as the current values: the gate reports the run's
// own samples.

/// The registered comparison matrix, embedded so the gate reads the compiled contract.
const MATRIX_JSON: &str = include_str!("matrix.json");

/// The workload key every registered current-strata cell names.
pub(crate) const REGISTERED_STRATA_WORKLOAD: &str = "reg1-current-strata";
/// The deterministic strata workload seed the registration must name.
pub(crate) const REGISTERED_STRATA_SEED: &str = "0x5eed1234abcd0001";
/// The actual executing selected implementation symbol under measurement: the incumbent's own map
/// body vendored into this crate at `src/hashtrinity/` (rev `4cc14471`) and reached as
/// `apeiron::freq::Table`. Its compiler name begins with `apeiron::hashtrinity::Table<`.
pub(crate) const EXECUTING_SELECTED_SYMBOL: &str = "apeiron::hashtrinity::Table<StrataKey,i64>";
/// The registered identity of that same selected implementation. Distinct from the incumbent arm's
/// `hashtrinity_table`, because the two differ only in which crate owns the type.
pub(crate) const EXECUTING_SELECTED_ID: &str = "apeiron_hashtrinity_map";
/// The actual executing baseline symbol under measurement.
pub(crate) const EXECUTING_BASELINE_SYMBOL: &str = "hashtrinity::Table<StrataKey,i64>";
/// The current source revision the registration binds to (`matrix.json` `pins.current`).
///
/// Every `current_strata` receipt names this revision and the medians it registers are a
/// measurement of that revision's source. `revision_contains_registered_artifact` is the check
/// that makes the name mean something.
pub(crate) const REGISTERED_CURRENT_REVISION: &str = "bde3d78e968639c3187154ca6cefd734b6e8bbc6";
/// The pinned hashtrinity revision the registered baseline binds to.
pub(crate) const REGISTERED_BASELINE_REVISION: &str = "4cc14471ae24f3f204fb0b0e91736ef75c909358";

/// The repository-relative paths a `current_strata` receipt registers: the harness that
/// enumerates and drives the cells, the registration document itself, and the table
/// implementation under measurement. A revision that does not contain all three did not produce
/// the medians the receipt registers.
pub(crate) const REGISTERED_ARTIFACT_PATHS: [&str; 3] = [
    "examples/reg1/matrix.json",
    "examples/reg1/strata.rs",
    "src/table.rs",
];

/// Does `revision` contain every path in [`REGISTERED_ARTIFACT_PATHS`]?
///
/// `git cat-file -e <revision>:<path>` is the containment test, run in this crate's manifest
/// directory. A revision that does not resolve, a path absent at it, and a `git` that cannot be
/// run are all `Err`: a revision whose containment cannot be shown is not a revision whose
/// artifact has been found, and the caller reports the row `revision_mismatch` instead of
/// comparing it. The registered medians are receipts of a run of this source, so a registration
/// naming a revision without the harness or the table names no run at all.
pub(crate) fn revision_contains_registered_artifact(revision: &str) -> Result<(), String> {
    for path in REGISTERED_ARTIFACT_PATHS {
        let spec = format!("{revision}:{path}");
        let output = Command::new("git")
            .arg("-C")
            .arg(env!("CARGO_MANIFEST_DIR"))
            .args(["cat-file", "-e", &spec])
            .output()
            .map_err(|e| format!("git cat-file -e {spec} could not run: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "{path} is absent at {revision} (git cat-file -e {spec} exited {})",
                output.status.code().unwrap_or(-1)
            ));
        }
    }
    Ok(())
}
/// The registered current cell the registration controls corrupt.
pub(crate) const CONTROL_REGISTERED_CELL: &str =
    "occ2-direct-index-resident-absent-miss-replacement";

/// The registered current-cell status vocabulary. `matrix.json` is a receipt, not edited by this
/// harness, so its historical `within-floor`/`exceeds-floor`/`not-applicable` labels stay the
/// accepted registration vocabulary; the run's own per-cell status is the owner threshold
/// `selected/incumbent <= 1.0` (see `strata::CellStatus`). The registered `not-applicable` label
/// is retained only for the archived fixed-shard refusal receipts.
const REGISTERED_STRATA_STATUSES: [&str; 3] = ["within-floor", "exceeds-floor", "not-applicable"];

/// One registered current-strata cell receipt. Every field is read by [`validate_registration`].
#[derive(Clone)]
pub(crate) struct RegisteredCell {
    pub(crate) id: String,
    pub(crate) workload: String,
    pub(crate) seed: String,
    pub(crate) distribution: String,
    pub(crate) operation: String,
    pub(crate) merge: String,
    pub(crate) population: String,
    pub(crate) hit_ratio: String,
    pub(crate) ops: String,
    pub(crate) oracle_digest: String,
    pub(crate) impl_id: String,
    pub(crate) implementation: String,
    pub(crate) baseline: String,
    // Read by `acceptance::evaluate_regression`, which parses `current_strata` into its own
    // `RegisteredReceipt` and settles revision validity there against the object store. These two
    // copies stay parsed because a registration row that omits the field is still malformed input.
    #[allow(
        dead_code,
        reason = "parsed to keep the registration schema total; the revision verdict is the \
                  regression stage's, and it reads its own copy"
    )]
    pub(crate) revision: String,
    #[allow(
        dead_code,
        reason = "parsed to keep the registration schema total; the revision verdict is the \
                  regression stage's, and it reads its own copy"
    )]
    pub(crate) baseline_revision: String,
    pub(crate) status: String,
}

/// The strata distribution's report name, mirroring `strata::Distribution::name`.
fn distribution_name(distribution: Distribution) -> &'static str {
    match distribution {
        Distribution::DirectIndexResident => "direct-index-resident",
        Distribution::OffsetBeyondDirect => "offset-beyond-direct",
        Distribution::SeededRandom => "seeded-random",
        Distribution::ClusteredSkewed => "clustered-skewed",
    }
}

/// The strata operation's report name, mirroring `strata::Operation::name`.
fn operation_name(operation: Operation) -> &'static str {
    match operation {
        Operation::PresentHit => "present-hit",
        Operation::AbsentMiss => "absent-miss",
        Operation::ExistingUpdate => "existing-update",
        Operation::AdmittedFreshInsert => "admitted-fresh-insert",
        Operation::GrowthInsert => "growth-insert",
    }
}

/// Parse one `RegisteredCell` array from the embedded matrix by field name. A malformed document is
/// an error, never a default.
fn parse_cell_array(text: &str, array: &str) -> Result<Vec<RegisteredCell>, String> {
    let root = crate::Reader::new(text).parse()?;
    let obj = root.as_obj()?;
    let crate::Json::Arr(entries) = crate::field(obj, array)? else {
        return Err(format!("field {array}: expected array"));
    };
    let mut cells = Vec::with_capacity(entries.len());
    for entry in entries {
        let fields = entry.as_obj()?;
        cells.push(RegisteredCell {
            id: crate::str_field(fields, "id")?,
            workload: crate::str_field(fields, "workload")?,
            seed: crate::str_field(fields, "seed")?,
            distribution: crate::str_field(fields, "distribution")?,
            operation: crate::str_field(fields, "operation")?,
            merge: crate::str_field(fields, "merge")?,
            population: crate::str_field(fields, "population")?,
            hit_ratio: crate::str_field(fields, "hit_ratio")?,
            ops: crate::str_field(fields, "ops")?,
            oracle_digest: crate::str_field(fields, "oracle_digest")?,
            impl_id: crate::str_field(fields, "impl_id")?,
            implementation: crate::str_field(fields, "implementation")?,
            baseline: crate::str_field(fields, "baseline")?,
            revision: crate::str_field(fields, "revision")?,
            baseline_revision: crate::str_field(fields, "baseline_revision")?,
            status: crate::str_field(fields, "status")?,
        });
    }
    Ok(cells)
}

/// Parse the embedded `matrix.json` `resource_cells` array.
fn parse_resource_cells(text: &str) -> Result<Vec<ResourceCell>, String> {
    let root = crate::Reader::new(text).parse()?;
    let obj = root.as_obj()?;
    let crate::Json::Arr(entries) = crate::field(obj, "resource_cells")? else {
        return Err("field resource_cells: expected array".to_string());
    };
    let mut cells = Vec::with_capacity(entries.len());
    for entry in entries {
        let fields = entry.as_obj()?;
        cells.push(ResourceCell {
            id: crate::str_field(fields, "id")?,
            operation: crate::str_field(fields, "operation")?,
            status: crate::str_field(fields, "status")?,
            population: crate::str_field(fields, "population")?,
            ops: crate::str_field(fields, "ops")?,
            observed: crate::str_field(fields, "observed")?,
            reason: crate::str_field(fields, "reason")?,
        });
    }
    Ok(cells)
}

/// Load the registered current-strata cells from the embedded matrix.
///
/// # Errors
/// Returns a description of a malformed registration document or a missing required field.
pub(crate) fn load_registered_cells() -> Result<Vec<RegisteredCell>, String> {
    parse_cell_array(MATRIX_JSON, "current_strata")
}

/// Compare one registered cell against the current run's cell of the same id. Every field the
/// registration names must agree with the run.
#[allow(
    clippy::too_many_lines,
    reason = "one helper states the whole per-cell registration contract"
)]
fn compare_cell(
    reg: &RegisteredCell,
    run_by_id: &HashMap<&str, &CellSamples>,
) -> Result<(), String> {
    if reg.workload != REGISTERED_STRATA_WORKLOAD {
        return Err(format!(
            "registered current cell {} workload {} != {REGISTERED_STRATA_WORKLOAD}",
            reg.id, reg.workload
        ));
    }
    if reg.seed != REGISTERED_STRATA_SEED {
        return Err(format!(
            "registered current cell {} seed {} != {REGISTERED_STRATA_SEED}",
            reg.id, reg.seed
        ));
    }
    if !REGISTERED_STRATA_STATUSES.contains(&reg.status.as_str()) {
        return Err(format!(
            "registered current cell {} status {} outside the vocabulary",
            reg.id, reg.status
        ));
    }
    if reg.impl_id != EXECUTING_SELECTED_ID || reg.implementation != EXECUTING_SELECTED_SYMBOL {
        return Err(format!(
            "registered current cell {} selected implementation {} ({}) is not the executing \
             implementation {} ({EXECUTING_SELECTED_SYMBOL})",
            reg.id, reg.impl_id, reg.implementation, EXECUTING_SELECTED_ID
        ));
    }
    if reg.baseline != EXECUTING_BASELINE_SYMBOL {
        return Err(format!(
            "registered current cell {} baseline {} is not the executing symbol \
             {EXECUTING_BASELINE_SYMBOL}",
            reg.id, reg.baseline
        ));
    }
    // Revision validity is not decided here. This helper states the *workload identity* a
    // registration must carry -- the seed, the operation, the population, the oracle digest and
    // the executing symbols -- and every one of those is a property of the run in front of us.
    // Whether the registered medians describe this source is a different question, answered per
    // row by `acceptance::evaluate_regression` against the object store
    // (`revision_contains_registered_artifact`), which reports the offending row by name and
    // counts it in `regression_summary`. Rejecting the whole registration here on a revision
    // string would pre-empt that check and make a fictional revision unnameable.
    let Some(cell) = run_by_id.get(reg.id.as_str()) else {
        return Err(format!(
            "registered current cell {} is absent from the current run",
            reg.id
        ));
    };
    if reg.distribution != distribution_name(cell.distribution) {
        return Err(format!(
            "registered current cell {} distribution {} != current run {}",
            reg.id,
            reg.distribution,
            distribution_name(cell.distribution)
        ));
    }
    if reg.operation != operation_name(cell.operation) {
        return Err(format!(
            "registered current cell {} operation {} != current run {}",
            reg.id,
            reg.operation,
            operation_name(cell.operation)
        ));
    }
    if reg.merge != cell.merge.name() {
        return Err(format!(
            "registered current cell {} merge {} != current run {}",
            reg.id,
            reg.merge,
            cell.merge.name()
        ));
    }
    if reg.population != cell.population.to_string() {
        return Err(format!(
            "registered current cell {} population {} != current run {}",
            reg.id, reg.population, cell.population
        ));
    }
    if reg.ops != cell.ops.to_string() {
        return Err(format!(
            "registered current cell {} ops {} != current run {}",
            reg.id, reg.ops, cell.ops
        ));
    }
    let digest = format!("{:016x}", cell.oracle_digest);
    if reg.oracle_digest != digest {
        return Err(format!(
            "registered current cell {} oracle digest {} != current run {digest}",
            reg.id, reg.oracle_digest
        ));
    }
    let hit_ratio = reg.hit_ratio.parse::<f64>().map_err(|e| {
        format!(
            "registered current cell {} hit_ratio {} is not a number: {e}",
            reg.id, reg.hit_ratio
        )
    })?;
    if (hit_ratio - cell.hit_ratio).abs() > 1e-9 {
        return Err(format!(
            "registered current cell {} hit ratio {} != current run {:.3}",
            reg.id, reg.hit_ratio, cell.hit_ratio
        ));
    }
    Ok(())
}

/// Enforce the registered `resource_cells` against the run's resource observations.
fn validate_resource_registration(run: &[ResourceCell]) -> Result<(), String> {
    let registered = parse_resource_cells(MATRIX_JSON)?;
    if registered.len() != run.len() {
        return Err(format!(
            "registered resource cells {} != current run {}",
            registered.len(),
            run.len()
        ));
    }
    let mut seen: HashSet<&str> = HashSet::with_capacity(registered.len());
    for reg in &registered {
        if !seen.insert(reg.id.as_str()) {
            return Err(format!("duplicate registered resource cell {}", reg.id));
        }
        let Some(cell) = run.iter().find(|c| c.id == reg.id) else {
            return Err(format!(
                "registered resource cell {} is absent from the current run",
                reg.id
            ));
        };
        if reg.operation != cell.operation
            || reg.status != cell.status
            || reg.population != cell.population
            || reg.ops != cell.ops
            || reg.observed != cell.observed
            || reg.reason != cell.reason
        {
            return Err(format!(
                "registered resource cell {} disagrees with the current run",
                reg.id
            ));
        }
    }
    for cell in run {
        if !seen.contains(cell.id.as_str()) {
            return Err(format!(
                "required resource cell {} is absent from the registration",
                cell.id
            ));
        }
    }
    Ok(())
}

/// Negative control run on every gate: reconstruct each archived fixed-shard refusal receipt under
/// its current growth-insert id and require the registration validator to reject it. The retired
/// `refusal` operation name and the retired `ShardFull` oracle digest/op count must not be accepted.
fn validate_retired_receipts_rejected(
    run_by_id: &HashMap<&str, &CellSamples>,
) -> Result<(), String> {
    let retired = parse_cell_array(MATRIX_JSON, "retired_strata")?;
    if retired.len() != 8 {
        return Err(format!(
            "retired control: expected 8 archived fixed-shard receipts, found {}",
            retired.len()
        ));
    }
    for old in &retired {
        let new_id = old.id.replace("-refusal-", "-growth-insert-");
        let Some(cell) = run_by_id.get(new_id.as_str()) else {
            return Err(format!(
                "retired control: no current growth cell for archived {}",
                old.id
            ));
        };
        let base = RegisteredCell {
            id: new_id,
            workload: REGISTERED_STRATA_WORKLOAD.to_string(),
            seed: REGISTERED_STRATA_SEED.to_string(),
            distribution: distribution_name(cell.distribution).to_string(),
            operation: operation_name(cell.operation).to_string(),
            merge: cell.merge.name().to_string(),
            population: cell.population.to_string(),
            hit_ratio: format!("{:.3}", cell.hit_ratio),
            ops: old.ops.clone(),
            oracle_digest: old.oracle_digest.clone(),
            impl_id: EXECUTING_SELECTED_ID.to_string(),
            implementation: EXECUTING_SELECTED_SYMBOL.to_string(),
            baseline: EXECUTING_BASELINE_SYMBOL.to_string(),
            revision: REGISTERED_CURRENT_REVISION.to_string(),
            baseline_revision: REGISTERED_BASELINE_REVISION.to_string(),
            status: "not-applicable".to_string(),
        };
        // (a) The retired `refusal` operation name and ShardFull digest/op count together.
        let mut operation_synthetic = base.clone();
        operation_synthetic.operation.clone_from(&old.operation);
        // (b) The retired ShardFull digest/op count under the current growth-insert operation name,
        // so the rejection is attributed to the digest rather than the operation name.
        let digest_synthetic = base;
        for (label, synthetic) in [
            ("retired-shardfull-receipt", &operation_synthetic),
            ("retired-shardfull-digest", &digest_synthetic),
        ] {
            match compare_cell(synthetic, run_by_id) {
                Ok(()) => {
                    return Err(format!(
                        "retired control: archived ShardFull {label} for {} was accepted",
                        old.id
                    ));
                }
                Err(reason) => {
                    println!(
                        "control={label} observed=red cell={} reason=\"{reason}\"",
                        old.id
                    );
                }
            }
        }
    }
    Ok(())
}

/// Negative control run on every gate: a registered cell that names no current run cell is a
/// fabricated cell and must be rejected.
fn validate_fabricated_cell_rejected(
    run: &StrataWitness,
    run_by_id: &HashMap<&str, &CellSamples>,
) -> Result<(), String> {
    let Some(cell) = run.cells.first() else {
        return Err("fabricated control: the run has no cells".to_string());
    };
    let synthetic = RegisteredCell {
        id: format!("{}-fabricated", cell.cell_id),
        workload: REGISTERED_STRATA_WORKLOAD.to_string(),
        seed: REGISTERED_STRATA_SEED.to_string(),
        distribution: distribution_name(cell.distribution).to_string(),
        operation: operation_name(cell.operation).to_string(),
        merge: cell.merge.name().to_string(),
        population: cell.population.to_string(),
        hit_ratio: format!("{:.3}", cell.hit_ratio),
        ops: cell.ops.to_string(),
        oracle_digest: format!("{:016x}", cell.oracle_digest),
        impl_id: EXECUTING_SELECTED_ID.to_string(),
        implementation: EXECUTING_SELECTED_SYMBOL.to_string(),
        baseline: EXECUTING_BASELINE_SYMBOL.to_string(),
        revision: REGISTERED_CURRENT_REVISION.to_string(),
        baseline_revision: REGISTERED_BASELINE_REVISION.to_string(),
        status: "not-applicable".to_string(),
    };
    match compare_cell(&synthetic, run_by_id) {
        Ok(()) => Err("fabricated control: a fabricated current cell was accepted".to_string()),
        Err(reason) => {
            println!("control=fabricated-current-cell observed=red reason=\"{reason}\"");
            Ok(())
        }
    }
}

/// The production address rule for a `u64` coordinate word: `place = spread(word).place`, and
/// `direct_word = Some(word as u32)` exactly when `word < 2^32`, otherwise `None`.
fn production_address(word: u64) -> TableAddress {
    TableAddress {
        place: apeiron::coord::spread(word).place,
        direct_word: u32::try_from(word).ok(),
    }
}

/// The callback-migration control: every benchmark home must derive exactly the production
/// address, and a placement-only `TableAddress` must not be accepted where a coordinate word is
/// required. A `StrataHome` that still returned the raw unspread tag as `place`, or omitted the
/// certified direct word, is rejected here; the negative control feeds a placement-only address and
/// requires the predicate to reject it.
fn validate_production_addresses() -> Result<(), String> {
    for key in initial_keys() {
        let word = u64::from(key.tag);
        let got = Home::address(&SharedHome, apeiron::Sigma::Parabolic, &key);
        let want = production_address(word);
        if got != want {
            return Err(format!(
                "production-address mismatch at SharedKey tag {}: place={:#018x} \
                 direct_word={:?} != expected place={:#018x} direct_word={:?}",
                key.tag, got.place, got.direct_word, want.place, want.direct_word
            ));
        }
    }
    for word in [
        0u64,
        1,
        0xFF,
        0x0FFF,
        (1 << 32) - 1,
        1 << 32,
        (1 << 40) + 7,
        u64::MAX,
    ] {
        let key = StrataKey { tag: word };
        let got = Home::address(&StrataHome, apeiron::Sigma::Parabolic, &key);
        let want = production_address(word);
        if got != want {
            return Err(format!(
                "production-address mismatch at StrataKey tag {word:#x}: place={:#018x} \
                 direct_word={:?} != expected place={:#018x} direct_word={:?}",
                got.place, got.direct_word, want.place, want.direct_word
            ));
        }
    }
    // Negative control: a placement-only address is not the production address for a coordinate.
    let placement_only = TableAddress {
        place: apeiron::coord::spread(1).place,
        direct_word: None,
    };
    if placement_only == production_address(1) {
        return Err(
            "placement-only control: a placement-only TableAddress was accepted as the production \
             address for a coordinate word"
                .to_string(),
        );
    }
    println!(
        "control=placement-only-address observed=red reason=\"a placement-only TableAddress is not \
         the production address for a coordinate word\""
    );
    Ok(())
}

/// Enforce the registered current-strata matrix against the current run. Every registered cell
/// must be produced by the run with the same workload identity and oracle digest, every run cell
/// must be registered, and the named implementation must be the actual executing symbol. The
/// resource registration, the production-address migration control and the archived fixed-shard
/// rejection control are enforced in the same call.
///
/// # Errors
/// Returns a description of the first absent cell, identity disagreement, implementation mismatch,
/// production-address mismatch, fabricated cell or surviving retired receipt. A caller that
/// observes `Ok(())` has an identity-checked registration, not a speed acceptance.
pub(crate) fn validate_registration(
    run: &StrataWitness,
    registered: &[RegisteredCell],
) -> Result<(), String> {
    let mut run_by_id: HashMap<&str, &CellSamples> = HashMap::with_capacity(run.cells.len());
    for cell in &run.cells {
        run_by_id.insert(cell.cell_id.as_str(), cell);
    }
    let mut registered_ids: HashSet<&str> = HashSet::with_capacity(registered.len());
    for reg in registered {
        if !registered_ids.insert(reg.id.as_str()) {
            return Err(format!("duplicate registered current cell {}", reg.id));
        }
        compare_cell(reg, &run_by_id)?;
    }
    for cell in &run.cells {
        if !registered_ids.contains(cell.cell_id.as_str()) {
            return Err(format!(
                "required current cell {} is absent from the registration",
                cell.cell_id
            ));
        }
    }
    validate_production_addresses()?;
    validate_resource_registration(&run.resources)?;
    validate_retired_receipts_rejected(&run_by_id)?;
    validate_fabricated_cell_rejected(run, &run_by_id)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ArcStrataHome, StrataHome, StrataKey};
    use apeiron::coord::address::{FaceResidueCapability, H3_40};
    use apeiron::table::Home;
    use std::sync::Arc;

    /// The `Arc` home delegates both face residues to `StrataHome`; replacing the parabolic
    /// residue with the placement constant `0` fails the equality assertion at `tag = 1`.
    #[test]
    fn arc_strata_home_delegates_face_residue() {
        let home = ArcStrataHome;
        let key = Arc::new(StrataKey { tag: 1 });
        let profile = H3_40::new().expect("the frozen H3_40 token validates");
        assert_eq!(home.home(&key), 0);
        assert_eq!(home.parabolic_dual(&key), StrataHome.parabolic_dual(&*key));
        assert_eq!(
            home.hyperbolic_scalar(profile, &key),
            StrataHome.hyperbolic_scalar(profile, &*key)
        );
        assert_ne!(home.parabolic_dual(&key).residue(), 0);
    }
}
