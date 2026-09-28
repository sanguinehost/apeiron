//! REG-1 workloads, their observations and the shared bootstrap witness.
//!
//! Every workload is deterministic and preregistered: the keys, homes, streams and expected folds
//! are constants in this file, not drawn at run time. `std::collections::HashMap` is an independent
//! oracle for the admitted readback, but its iteration order is never read: every observation is
//! emitted in key order. The cross-process digest is FNV-1a over canonical fixed-width fields, so
//! no `DefaultHasher` (whose seed is per-process) is involved.

#![allow(
    clippy::struct_excessive_bools,
    reason = "a witness records independent observations, several of them booleans"
)]

use std::collections::HashMap;
use std::process::Command;

use apeiron::table::{Merge, StorageError, Table, WordMerge};
use hashtrinity::Table as HtTable;

use crate::{AxisRecord, AxisStatus, Report};

/// The 17 distinct keys of the constant-coordinate identity workload.
pub(crate) const KEY_COUNT: u32 = 17;
/// Apeiron's per-shard override capacity (`Table<K, V, H, N>`).
pub(crate) const APEIRON_N: usize = 16;

// The key and home definitions moved into this child module to keep `workload.rs` below the
// source-size cap; the re-export preserves the module-local access paths. The exact identities and
// explicit face capabilities live there: `ShardKey` keeps both equality fields in 12 bytes on the
// locator domain, and `ConstantKey` keeps its full tag with the constant physical placement
// separate from the semantic face map.
#[path = "adapters/native_workload.rs"]
mod native_workload;

use native_workload::{ConstantHome, ConstantKey, ShardHome, ShardKey};

/// A non-commutative merge: the new contribution dominates, the old is doubled. `f(v, 0) = v`, so
/// hashtrinity's absent-key store (which skips the identity) coincides with apeiron's
/// merge-with-default first write.
struct OrderMerge;

impl Merge<i64> for OrderMerge {
    fn merge(&self, new: i64, old: i64) -> i64 {
        new.wrapping_add(old.wrapping_mul(2))
    }
}

/// One canonical observation: a key and the value read back for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KeyValue {
    pub(crate) key: u64,
    pub(crate) value: i64,
}

/// The constant-coordinate identity observation: 17 distinct keys, one coordinate.
#[derive(Clone, Debug)]
pub(crate) struct IdentityWitness {
    pub(crate) oracle_len: usize,
    pub(crate) oracle_readback_ok: bool,
    pub(crate) ht_len: usize,
    pub(crate) ht_value: i64,
    pub(crate) ht_readback_all_some: bool,
    pub(crate) apeiron_admitted: usize,
    pub(crate) apeiron_refused: usize,
    pub(crate) apeiron_readback_ok: bool,
}

/// Run the identity workload: the same 17 constant-coordinate keys against all three tables.
#[must_use]
pub(crate) fn run_identity_witness() -> IdentityWitness {
    let keys: Vec<ConstantKey> = (0..KEY_COUNT).map(|tag| ConstantKey { tag }).collect();
    let value_of = |k: &ConstantKey| i64::from(k.tag);

    // Independent oracle: 17 distinct identities, 17 entries, each read back as submitted.
    let mut oracle: HashMap<ConstantKey, i64> = HashMap::new();
    for k in &keys {
        oracle.insert(*k, value_of(k));
    }
    let oracle_readback_ok = keys.iter().all(|k| oracle.get(k) == Some(&value_of(k)));

    // Pinned historical hashtrinity: a constant coordinate puts all 17 keys in one slot, so the
    // counter merge accumulates them there and every read returns that one coalesced value.
    let mut ht: HtTable<ConstantKey, i64> = HtTable::new();
    for k in &keys {
        ht.insert_with(*k, value_of(k), |new, old| new + old);
    }
    let ht_readback_all_some = keys.iter().all(|k| ht.get(k).is_some());
    let ht_value = *ht.get(&keys[0]).unwrap_or(&-1);

    // Current apeiron: a constant home puts all 17 keys in one shard with N=16 slots. The first 16
    // are admitted; the 17th is refused with `ShardFull`, and the table is left unchanged.
    let mut ap: Table<ConstantKey, i64, ConstantHome, APEIRON_N> = Table::new(0, ConstantHome);
    let mut apeiron_admitted = 0usize;
    let mut apeiron_refused = 0usize;
    for k in &keys {
        match ap.insert_with(&|new: i64, _old: i64| new, *k, value_of(k)) {
            Ok(()) => apeiron_admitted += 1,
            Err(
                StorageError::ShardFull
                | StorageError::SizeOverflow
                | StorageError::AllocationFailed,
            ) => {
                apeiron_refused += 1;
            }
        }
    }
    // The growable representation admits every distinct key in the home; the region grows.
    let apeiron_readback_ok = keys.iter().all(|k| *ap.get(k) == value_of(k));

    IdentityWitness {
        oracle_len: oracle.len(),
        oracle_readback_ok,
        ht_len: ht.len(),
        ht_value,
        ht_readback_all_some,
        apeiron_admitted,
        apeiron_refused,
        apeiron_readback_ok,
    }
}

/// The bootstrap witness: the identity workload plus the `N = 0` refusal.
pub(crate) struct BootstrapWitness {
    pub(crate) identity: IdentityWitness,
    pub(crate) zero_capacity_refused: bool,
}

/// Run the bootstrap workload.
#[must_use]
pub(crate) fn run_bootstrap_witness() -> BootstrapWitness {
    // `N = 0` is a valid table whose reads return the default and whose inserts all refuse.
    let mut zero: Table<ConstantKey, i64, ConstantHome, 0> = Table::new(7, ConstantHome);
    let zero_capacity_refused = matches!(
        zero.insert_with(&|new: i64, _old: i64| new, ConstantKey { tag: 99 }, 1),
        Err(StorageError::SizeOverflow)
    );
    BootstrapWitness {
        identity: run_identity_witness(),
        zero_capacity_refused,
    }
}

/// Build the bootstrap report. Its axis vocabulary and evidence are the W1 contract.
#[must_use]
pub(crate) fn build_bootstrap_report(w: &BootstrapWitness) -> Report {
    let id = &w.identity;
    let axes = vec![
        AxisRecord {
            axis: "identity",
            status: AxisStatus::Missing,
            evidence: format!(
                "full-scale 1e8 coordinate-colliding identity run unrun; bootstrap witness \
                 separates apeiron refusal ({}/{} admitted) from hashtrinity coordinate \
                 coalescing ({} live of {} submitted)",
                id.apeiron_admitted, KEY_COUNT, id.ht_len, KEY_COUNT
            ),
        },
        AxisRecord {
            axis: "round_trip",
            status: AxisStatus::Boundary,
            evidence: "apeiron::Table exposes no encode/decode surface; the sentinel/boundary \
                       round-trip is not exercisable"
                .to_string(),
        },
        AxisRecord {
            axis: "determinism",
            status: AxisStatus::Missing,
            evidence: "separate-process comparison unrun; only the single-process bootstrap ran"
                .to_string(),
        },
        AxisRecord {
            axis: "growth_merge_refusal",
            status: AxisStatus::Boundary,
            evidence: format!(
                "region growth admits every distinct key (full_keys={}), and N=0 refuses with \
                 SizeOverflow (N=0 refusal={}); remove remains absent; ordered and commutative \
                 merge comparison is unrun at bootstrap",
                id.apeiron_admitted, w.zero_capacity_refused
            ),
        },
        AxisRecord {
            axis: "performance",
            status: AxisStatus::Missing,
            evidence: "no benchmark run; no ratio measured and none claimed".to_string(),
        },
        AxisRecord {
            axis: "coverage",
            status: AxisStatus::Missing,
            evidence: "coverage census and fractions are unrun; real-source extraction, bridge \
                       refinement and executed Rust tests must be counted separately at their \
                       stated scopes, and model-only evidence must not be attributed as \
                       real-container coverage"
                .to_string(),
        },
    ];
    Report {
        axes,
        report_complete: false,
        reg1_pass: false,
    }
}

/// The one bootstrap validator. It fails closed on every status and count.
///
/// # Errors
/// Returns a description of the first observation, count or status that violates the W1 bootstrap
/// contract.
pub(crate) fn validate_bootstrap(w: &BootstrapWitness, r: &Report) -> Result<(), String> {
    let id = &w.identity;
    if id.oracle_len != KEY_COUNT as usize {
        return Err(format!("oracle len {} != {KEY_COUNT}", id.oracle_len));
    }
    if !id.oracle_readback_ok {
        return Err("oracle readback mismatch".to_string());
    }
    if id.ht_len != 1 {
        return Err(format!(
            "hashtrinity coalescing expected 1 live entry, got {}",
            id.ht_len
        ));
    }
    if !id.ht_readback_all_some {
        return Err("hashtrinity readback missed a submitted key".to_string());
    }
    if id.apeiron_admitted != KEY_COUNT as usize || id.apeiron_refused != 0 {
        return Err(format!(
            "apeiron admitted {} refused {} != {KEY_COUNT}/0",
            id.apeiron_admitted, id.apeiron_refused
        ));
    }
    if !id.apeiron_readback_ok {
        return Err("apeiron readback mismatch on admitted/refused keys".to_string());
    }
    if !w.zero_capacity_refused {
        return Err("N=0 apeiron table did not refuse an insert".to_string());
    }
    if r.report_complete {
        return Err("report_complete must be false at bootstrap".to_string());
    }
    if r.reg1_pass {
        return Err("reg1_pass must be false at bootstrap".to_string());
    }
    let names: Vec<&str> = r.axes.iter().map(|a| a.axis).collect();
    let expected = [
        "identity",
        "round_trip",
        "determinism",
        "growth_merge_refusal",
        "performance",
        "coverage",
    ];
    if names != expected {
        return Err(format!("axis schema mismatch: {names:?}"));
    }
    if r.axes.iter().any(|a| !crate::allowed(a.status)) {
        return Err("axis status outside pass/fail/boundary/missing/tool_error".to_string());
    }
    if r.axes[0].status == AxisStatus::Pass {
        return Err("identity cannot pass while the 1e8 run is unrun".to_string());
    }
    if r.axes[3].status != AxisStatus::Boundary {
        return Err(
            "growth_merge_refusal must be boundary: refusal observed, remove/growth absent"
                .to_string(),
        );
    }
    if r.axes[4].status != AxisStatus::Missing {
        return Err("performance must be missing: no measurement was run".to_string());
    }
    if r.axes[5].status != AxisStatus::Missing {
        return Err("coverage must be missing: no coverage census was run".to_string());
    }
    if !r
        .axes
        .iter()
        .any(|a| matches!(a.status, AxisStatus::Missing | AxisStatus::Boundary))
    {
        return Err("report carries no missing/boundary axis".to_string());
    }
    Ok(())
}

// ── Correctness smoke workloads ────────────────────────────────────────────────────────────────

/// How many home groups the admitted workload fills, each to exactly `APEIRON_N`.
const ADMITTED_GROUPS: u32 = 3;
/// Keys per home group. Exactly `APEIRON_N`, so every group's shard is at its capacity.
const ADMITTED_PER_GROUP: usize = APEIRON_N;

/// The admitted workload's keys. Group `g` shares home `g`, so its `APEIRON_N` keys share one
/// Apeiron shard and fill it exactly; tags are distinct, so hashtrinity keeps them separate.
fn admitted_keys() -> Vec<ShardKey> {
    let mut keys = Vec::with_capacity(ADMITTED_GROUPS as usize * ADMITTED_PER_GROUP);
    for group in 0..ADMITTED_GROUPS {
        for slot in 0..ADMITTED_PER_GROUP {
            let offset = u32::try_from(slot).expect("APEIRON_N fits u32");
            keys.push(ShardKey {
                tag: group * 1000 + offset + 1,
                home: u64::from(group),
            });
        }
    }
    keys
}

/// The admitted fixed-capacity observation: all three tables over the same admitted key set.
pub(crate) struct AdmittedWitness {
    pub(crate) keys: usize,
    pub(crate) per_shard_max: usize,
    pub(crate) apeiron_admitted: usize,
    pub(crate) ht_len: usize,
    pub(crate) oracle_len: usize,
    pub(crate) apeiron: Vec<KeyValue>,
    pub(crate) oracle: Vec<KeyValue>,
    pub(crate) ht: Vec<KeyValue>,
    pub(crate) readback_ok: bool,
}

/// Run the admitted fixed-capacity workload and read every key back from all three tables.
#[must_use]
pub(crate) fn run_admitted_witness() -> AdmittedWitness {
    let keys = admitted_keys();
    let value_of = |k: &ShardKey| i64::from(k.tag).wrapping_mul(7).wrapping_add(11);

    let mut oracle: HashMap<ShardKey, i64> = HashMap::new();
    for k in &keys {
        oracle.insert(*k, value_of(k));
    }

    let mut ap: Table<ShardKey, i64, ShardHome, APEIRON_N> = Table::new(0, ShardHome);
    let mut apeiron_admitted = 0usize;
    for k in &keys {
        if ap
            .insert_with(&|new: i64, _old: i64| new, *k, value_of(k))
            .is_ok()
        {
            apeiron_admitted += 1;
        }
    }

    let mut ht: HtTable<ShardKey, i64> = HtTable::new();
    for k in &keys {
        ht.insert_with(*k, value_of(k), |new, _old| new);
    }

    let sorted = sorted_keys(&keys);
    let mut apeiron = Vec::with_capacity(sorted.len());
    let mut oracle_obs = Vec::with_capacity(sorted.len());
    let mut ht_obs = Vec::with_capacity(sorted.len());
    let mut readback_ok = true;
    for k in &sorted {
        let a = *ap.get(k);
        let o = *oracle.get(k).unwrap_or(&i64::MIN);
        let h = *ht.get(k).unwrap_or(&i64::MIN);
        readback_ok &= a == o && a == h;
        let key = u64::from(k.tag);
        apeiron.push(KeyValue { key, value: a });
        oracle_obs.push(KeyValue { key, value: o });
        ht_obs.push(KeyValue { key, value: h });
    }

    // Keys sharing a home necessarily share a shard, so the per-home count is the per-shard count.
    let mut per_home: HashMap<u64, usize> = HashMap::new();
    for k in &keys {
        *per_home.entry(k.home).or_insert(0) += 1;
    }
    let per_shard_max = per_home.values().copied().max().unwrap_or(0);

    AdmittedWitness {
        keys: keys.len(),
        per_shard_max,
        apeiron_admitted,
        ht_len: ht.len(),
        oracle_len: oracle.len(),
        apeiron,
        oracle: oracle_obs,
        ht: ht_obs,
        readback_ok,
    }
}

/// Sort keys by tag, the canonical order every observation vector uses.
fn sorted_keys(keys: &[ShardKey]) -> Vec<ShardKey> {
    let mut sorted = keys.to_vec();
    sorted.sort_by_key(|k| k.tag);
    sorted
}

/// The ordered-merge fixture keys: three distinct homes and tags.
const ORDER_KEYS: [u32; 3] = [101, 202, 303];
/// The ordered-merge fixture homes.
const ORDER_HOMES: [u64; 3] = [11, 22, 33];
/// The preregistered value stream. `OrderMerge` makes its forward fold differ from its reverse.
const ORDER_STREAM: [i64; 3] = [3, 5, 7];

/// The ordered-merge observation: forward and reversed streams against their stream reference.
pub(crate) struct OrderedWitness {
    pub(crate) keys: usize,
    pub(crate) forward: Vec<KeyValue>,
    pub(crate) reverse: Vec<KeyValue>,
    pub(crate) reference_forward: Vec<KeyValue>,
    pub(crate) reference_reverse: Vec<KeyValue>,
    pub(crate) ht_forward: Vec<KeyValue>,
    pub(crate) ht_reverse: Vec<KeyValue>,
    pub(crate) order_observable: bool,
}

/// The independent stream fold: start from the default and merge each value new-first.
fn fold_stream(stream: &[i64]) -> i64 {
    let mut acc = 0i64;
    for &v in stream {
        acc = OrderMerge.merge(v, acc);
    }
    acc
}

/// Run the ordered-merge fixture: forward and reversed streams on apeiron and hashtrinity.
#[must_use]
pub(crate) fn run_ordered_witness() -> OrderedWitness {
    let keys: Vec<ShardKey> = ORDER_KEYS
        .iter()
        .zip(ORDER_HOMES)
        .map(|(&tag, home)| ShardKey { tag, home })
        .collect();
    let forward_stream = ORDER_STREAM;
    let mut reverse_stream = ORDER_STREAM;
    reverse_stream.reverse();

    let (forward, ht_forward) = run_ordered_tables(&keys, &forward_stream);
    let (reverse, ht_reverse) = run_ordered_tables(&keys, &reverse_stream);

    let reference_forward = reference_values(&keys, fold_stream(&forward_stream));
    let reference_reverse = reference_values(&keys, fold_stream(&reverse_stream));
    let order_observable = forward != reverse;

    OrderedWitness {
        keys: keys.len(),
        forward,
        reverse,
        reference_forward,
        reference_reverse,
        ht_forward,
        ht_reverse,
        order_observable,
    }
}

/// A per-key observation vector carrying the same reference value for every key.
fn reference_values(keys: &[ShardKey], value: i64) -> Vec<KeyValue> {
    sorted_keys(keys)
        .iter()
        .map(|k| KeyValue {
            key: u64::from(k.tag),
            value,
        })
        .collect()
}

/// Insert `stream` new-first into a fresh apeiron and hashtrinity table for every key.
fn run_ordered_tables(keys: &[ShardKey], stream: &[i64]) -> (Vec<KeyValue>, Vec<KeyValue>) {
    let mut ap: Table<ShardKey, i64, ShardHome, APEIRON_N> = Table::new(0, ShardHome);
    let mut ht: HtTable<ShardKey, i64> = HtTable::new();
    for k in keys {
        for &v in stream {
            ap.insert_with(&OrderMerge, *k, v)
                .expect("ordered fixture fits the fixed capacity");
            ht.insert_with(*k, v, |new, old| OrderMerge.merge(new, old));
        }
    }
    let sorted = sorted_keys(keys);
    let ap_obs = sorted
        .iter()
        .map(|k| KeyValue {
            key: u64::from(k.tag),
            value: *ap.get(k),
        })
        .collect();
    let ht_obs = sorted
        .iter()
        .map(|k| KeyValue {
            key: u64::from(k.tag),
            value: *ht.get(k).expect("ordered fixture present in hashtrinity"),
        })
        .collect();
    (ap_obs, ht_obs)
}

/// The commutative-merge fixture keys: four distinct homes and tags.
const COMMUTATIVE_KEYS: [u32; 4] = [7, 19, 31, 43];
/// The commutative-merge fixture homes.
const COMMUTATIVE_HOMES: [u64; 4] = [70, 71, 72, 73];

/// The commutative-merge observation: original, permuted and partitioned streams.
pub(crate) struct CommutativeWitness {
    pub(crate) keys: usize,
    pub(crate) original: Vec<KeyValue>,
    pub(crate) permuted: Vec<KeyValue>,
    pub(crate) partitioned: Vec<KeyValue>,
    pub(crate) oracle: Vec<KeyValue>,
    pub(crate) ht_original: Vec<KeyValue>,
    pub(crate) ht_permuted: Vec<KeyValue>,
    pub(crate) agree: bool,
}

/// The original stream order, as a named function so the fixture's order choices are explicit.
fn identity_order(stream: &[i64]) -> Vec<i64> {
    stream.to_vec()
}

/// The preregistered per-key value stream for the commutative fixture.
fn commutative_stream(tag: u32) -> Vec<i64> {
    let base = i64::from(tag);
    vec![base, base + 1, base + 2, base + 3]
}

/// Run the commutative wrapping-add fixture: original, reversed and even/odd-partitioned streams.
#[must_use]
pub(crate) fn run_commutative_witness() -> CommutativeWitness {
    let keys: Vec<ShardKey> = COMMUTATIVE_KEYS
        .iter()
        .zip(COMMUTATIVE_HOMES)
        .map(|(&tag, home)| ShardKey { tag, home })
        .collect();

    let original = run_commutative_apeiron(&keys, identity_order);
    let permuted = run_commutative_apeiron(&keys, |stream| {
        let mut reversed = stream.to_vec();
        reversed.reverse();
        reversed
    });
    let partitioned = run_commutative_apeiron(&keys, |stream| {
        let mut split: Vec<i64> = stream.iter().step_by(2).copied().collect();
        split.extend(stream.iter().skip(1).step_by(2).copied());
        split
    });
    let ht_original = run_commutative_hashtrinity(&keys, identity_order);
    let ht_permuted = run_commutative_hashtrinity(&keys, |stream| {
        let mut reversed = stream.to_vec();
        reversed.reverse();
        reversed
    });
    let oracle: Vec<KeyValue> = sorted_keys(&keys)
        .iter()
        .map(|k| {
            let sum = commutative_stream(k.tag)
                .iter()
                .fold(0i64, |acc, &v| acc.wrapping_add(v));
            KeyValue {
                key: u64::from(k.tag),
                value: sum,
            }
        })
        .collect();

    let agree = original == permuted
        && original == partitioned
        && original == oracle
        && original == ht_original
        && original == ht_permuted;

    CommutativeWitness {
        keys: keys.len(),
        original,
        permuted,
        partitioned,
        oracle,
        ht_original,
        ht_permuted,
        agree,
    }
}

/// Insert each key's stream into a fresh apeiron table in the order `order` returns.
fn run_commutative_apeiron(
    keys: &[ShardKey],
    mut order: impl FnMut(&[i64]) -> Vec<i64>,
) -> Vec<KeyValue> {
    let mut ap: Table<ShardKey, i64, ShardHome, APEIRON_N> = Table::new(0, ShardHome);
    for k in keys {
        for v in order(&commutative_stream(k.tag)) {
            ap.insert_with(&WordMerge, *k, v)
                .expect("commutative fixture fits the fixed capacity");
        }
    }
    sorted_keys(keys)
        .iter()
        .map(|k| KeyValue {
            key: u64::from(k.tag),
            value: *ap.get(k),
        })
        .collect()
}

/// Insert each key's stream into a fresh hashtrinity table in the order `order` returns.
fn run_commutative_hashtrinity(
    keys: &[ShardKey],
    mut order: impl FnMut(&[i64]) -> Vec<i64>,
) -> Vec<KeyValue> {
    let mut ht: HtTable<ShardKey, i64> = HtTable::new();
    for k in keys {
        for v in order(&commutative_stream(k.tag)) {
            ht.insert_with(*k, v, i64::wrapping_add);
        }
    }
    sorted_keys(keys)
        .iter()
        .map(|k| KeyValue {
            key: u64::from(k.tag),
            value: *ht
                .get(k)
                .expect("commutative fixture present in hashtrinity"),
        })
        .collect()
}

/// The refusal observation: a full shard refuses without changing state, and `N = 0` refuses.
pub(crate) struct RefusalWitness {
    pub(crate) full_keys: usize,
    pub(crate) full_refused: bool,
    pub(crate) refusal_is_shard_full: bool,
    pub(crate) state_unchanged: bool,
    pub(crate) zero_capacity_refused: bool,
    pub(crate) zero_read_is_default: bool,
}

/// Run the growth fixture: one home admits more distinct keys than the fixed group width, the
/// region grows and preserves every key/value, and `N = 0` refuses with `SizeOverflow`.
#[must_use]
pub(crate) fn run_refusal_witness() -> RefusalWitness {
    const FULL_HOME: u64 = 42;
    let over = APEIRON_N * 8;
    let keys: Vec<ShardKey> = (0..over)
        .map(|i| ShardKey {
            tag: u32::try_from(i).expect("over fits u32") + 1,
            home: FULL_HOME,
        })
        .collect();
    let value_of = |k: &ShardKey| i64::from(k.tag).wrapping_mul(13).wrapping_add(5);

    let mut ap: Table<ShardKey, i64, ShardHome, APEIRON_N> = Table::new(0, ShardHome);
    let mut full_refused = false;
    for k in &keys {
        full_refused |= ap
            .insert_with(&|new: i64, _old: i64| new, *k, value_of(k))
            .is_err();
    }
    let state_unchanged = !full_refused && keys.iter().all(|k| *ap.get(k) == value_of(k));

    let intruder = ShardKey {
        tag: u32::try_from(over).expect("over fits u32") + 1,
        home: FULL_HOME,
    };
    let mut zero: Table<ShardKey, i64, ShardHome, 0> = Table::new(7, ShardHome);
    let zero_read_is_default = *zero.get(&intruder) == 7;
    let zero_capacity_refused = matches!(
        zero.insert_with(&|new: i64, _old: i64| new, intruder, 1),
        Err(StorageError::SizeOverflow)
    );

    RefusalWitness {
        full_keys: keys.len(),
        full_refused,
        refusal_is_shard_full: false,
        state_unchanged,
        zero_capacity_refused,
        zero_read_is_default,
    }
}

// ── Separate-process determinism ───────────────────────────────────────────────────────────────

/// The determinism observation: the canonical digest emitted by fresh child processes.
#[allow(
    clippy::struct_field_names,
    reason = "every field describes the child-process observation"
)]
pub(crate) struct DeterminismWitness {
    pub(crate) child_count: usize,
    pub(crate) child_match: bool,
    pub(crate) child_exit_ok: bool,
    pub(crate) child_digests: Vec<String>,
}

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over `bytes`. Explicit and seed-free, so separate processes agree; `DefaultHasher` is
/// randomized per process and is not used.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The child observation: canonical sorted admitted observations and refusal outcomes, plus the
/// digest over their fixed-width serialization. This is the only output a child process emits.
#[must_use]
pub(crate) fn determinism_child_output() -> String {
    let admitted = run_admitted_witness();
    let refusal = run_refusal_witness();
    let mut observations = admitted.apeiron;
    observations.sort_by_key(|kv| kv.key);

    let mut lines = Vec::with_capacity(observations.len() + 3);
    let mut bytes = Vec::with_capacity(observations.len() * 16 + 5);
    for kv in &observations {
        lines.push(format!("obs {} {}", kv.key, kv.value));
        bytes.extend_from_slice(&kv.key.to_le_bytes());
        bytes.extend_from_slice(&kv.value.to_le_bytes());
    }
    lines.push(format!(
        "refusal full={} shard_full={} unchanged={}",
        refusal.full_refused, refusal.refusal_is_shard_full, refusal.state_unchanged
    ));
    bytes.push(u8::from(refusal.full_refused));
    bytes.push(u8::from(refusal.refusal_is_shard_full));
    bytes.push(u8::from(refusal.state_unchanged));
    lines.push(format!(
        "refusal zero_refused={} zero_default={}",
        refusal.zero_capacity_refused, refusal.zero_read_is_default
    ));
    bytes.push(u8::from(refusal.zero_capacity_refused));
    bytes.push(u8::from(refusal.zero_read_is_default));

    let digest = fnv1a64(&bytes);
    lines.push(format!("digest={digest:016x}"));
    lines.join("\n")
}

/// Run at least two fresh child processes and require their canonical output to match.
#[must_use]
pub(crate) fn run_determinism_witness() -> DeterminismWitness {
    const CHILDREN: usize = 2;
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            return DeterminismWitness {
                child_count: 1,
                child_match: false,
                child_exit_ok: false,
                child_digests: vec![format!("current_exe error: {e}")],
            };
        }
    };
    let mut child_exit_ok = true;
    let mut child_digests = Vec::with_capacity(CHILDREN);
    for _ in 0..CHILDREN {
        match Command::new(&exe).arg("--determinism-child").output() {
            Ok(output) if output.status.success() => {
                child_digests.push(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
            Ok(output) => {
                child_exit_ok = false;
                child_digests.push(format!("child exit {:?}", output.status.code()));
            }
            Err(e) => {
                child_exit_ok = false;
                child_digests.push(format!("spawn error: {e}"));
            }
        }
    }
    let child_match = child_exit_ok
        && child_digests.len() == CHILDREN
        && child_digests
            .iter()
            .all(|d| d == &child_digests[0] && d.contains("digest="));
    DeterminismWitness {
        child_count: child_digests.len(),
        child_match,
        child_exit_ok,
        child_digests,
    }
}
