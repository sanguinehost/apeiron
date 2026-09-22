//! REG-1 serial table adapters over the current-strata `StrataKey` population.
//!
//! Each adapter re-runs the same admitted key set, lookup trace, update trace and canonical readback
//! that `strata.rs` measures for the selected `apeiron::Table` and the pinned `hashtrinity::Table`.
//! The selected serial samples use `RawStrataHome`: `place = StrataKey::tag` and
//! `direct_word = Some(tag)` exactly when that raw coordinate fits `u32`.  Thus each current
//! serial incumbent receives the same raw key coordinate; the paired acceptance row records that
//! face and width. The warmup/sample/campaign protocol, merge function and oracle digest stay
//! shared. The adapter's digest must equal the cell's pre-timing oracle digest; a mismatch is
//! carried as `digest_ok = false` and the pair rejects rather than passing.
//!
//! Load-bearing incumbents (`hashbrown_foldhash`, `hashbrown_fxhash`, `std_siphash`) use the real
//! crate APIs: `hashbrown` with `foldhash::fast::RandomState` and `fxhash::FxBuildHasher`, and
//! `std::collections::HashMap` (SipHash-1-3). The informational comparators use
//! `hashbrown::HashMap<_, _, BuildHasherDefault<H>>` with `twox_hash::{XxHash3_64, XxHash64}`,
//! `wyhash::WyHash`, `ahash::AHasher` and `fnv::FnvHasher`. `blake3` and `sha2::Sha256` are not
//! `std::hash::Hasher` implementations, so each is wrapped by a local `Hasher` that feeds the same
//! key bytes into the real crate's digest and returns its first eight bytes; the map still stores
//! the full key, so collisions are resolved by key equality and the canonical result is unchanged.
//!
//! The workload is reconstructed from the strata witness with the same generators
//! (`make_plan`, `build_trace`, `initial_value`, `merge_fn`), so the adapter sees the identical
//! `StrataKey` values; the reference digest is the cell's own oracle digest, never a value the
//! adapter computes.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the serial adapter campaign and its outcome types"
)]

use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
use std::hint::black_box;
use std::time::Instant;

use crate::matrix::{median, ratio};
use crate::strata::{
    build_trace, fold, initial_value, make_plan, merge_fn, nanos_since, probe_capacity,
    CellSamples, CellSpec, MergeFn, StrataKey, StrataWitness, FNV_OFFSET, STRATA_WARMUP,
};

/// The load-bearing `hashbrown` + `foldhash` serial map.
type FoldhashMap = hashbrown::HashMap<StrataKey, i64, foldhash::fast::RandomState>;
/// The load-bearing `hashbrown` + `fxhash` serial map.
type FxhashMap = hashbrown::HashMap<StrataKey, i64, fxhash::FxBuildHasher>;
/// The load-bearing `std::collections::HashMap` (SipHash-1-3) serial map.
type SiphashMap = std::collections::HashMap<StrataKey, i64>;
/// The informational XXH3-64 map.
type Xxh3Map = hashbrown::HashMap<StrataKey, i64, BuildHasherDefault<Xxh3Hasher>>;
/// The informational XXH64 map.
type Xxh64Map = hashbrown::HashMap<StrataKey, i64, BuildHasherDefault<twox_hash::XxHash64>>;
/// The informational wyhash map.
type WyhashMap = hashbrown::HashMap<StrataKey, i64, BuildHasherDefault<wyhash::WyHash>>;
/// The informational ahash map.
type AHashMap = hashbrown::HashMap<StrataKey, i64, BuildHasherDefault<ahash::AHasher>>;
/// The informational FNV-1a map.
type FnvMap = hashbrown::HashMap<StrataKey, i64, BuildHasherDefault<fnv::FnvHasher>>;
/// The informational BLAKE3 map.
type Blake3Map = hashbrown::HashMap<StrataKey, i64, BuildHasherDefault<Blake3Hasher>>;
/// The informational SHA-256 map.
type Sha256Map = hashbrown::HashMap<StrataKey, i64, BuildHasherDefault<Sha256Hasher>>;

/// One adapter's campaign result for one cell.
pub(crate) struct SerialOutcome {
    /// The registered incumbent id.
    pub(crate) id: &'static str,
    /// The pooled per-sample nanoseconds across every campaign.
    pub(crate) samples: Vec<u64>,
    /// The per-campaign `selected/incumbent` ratios.
    pub(crate) campaign_ratios: Vec<f64>,
    /// Whether every sample's digest equalled the cell's oracle digest.
    pub(crate) digest_ok: bool,
}

/// One cell's serial adapter outcomes.
pub(crate) struct SerialCellOutcome {
    /// The strata cell id.
    pub(crate) cell_id: String,
    /// The adapters measured on this cell, or empty when the cell was not measured.
    pub(crate) outcomes: Vec<SerialOutcome>,
}

/// The shared inputs of one cell's adapter campaign.
struct Campaign<'a> {
    keys: &'a [StrataKey],
    lookups: &'a [StrataKey],
    inserts: &'a [(StrataKey, i64)],
    merge: MergeFn,
    oracle: u64,
    campaigns: u32,
    samples: u32,
    /// The selected implementation's per-campaign medians, one per campaign.
    selected_medians: &'a [u64],
}

/// One adapter sample's digest and elapsed nanoseconds.
struct SampleOutcome {
    digest: u64,
    nanos: u64,
}

/// A minimal map surface shared by `hashbrown::HashMap` and `std::collections::HashMap`.
trait SerialMap: Sized {
    /// A map with capacity for `capacity` entries and the adapter's hasher.
    fn with_capacity(capacity: usize) -> Self;
    /// The value stored at `key`, or `None`.
    fn get_value(&self, key: StrataKey) -> Option<i64>;
    /// Store `value` at `key`.
    fn put_value(&mut self, key: StrataKey, value: i64);
}

impl<S: BuildHasher + Default> SerialMap for hashbrown::HashMap<StrataKey, i64, S> {
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }

    fn get_value(&self, key: StrataKey) -> Option<i64> {
        self.get(&key).copied()
    }

    fn put_value(&mut self, key: StrataKey, value: i64) {
        self.insert(key, value);
    }
}

impl SerialMap for std::collections::HashMap<StrataKey, i64> {
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    fn get_value(&self, key: StrataKey) -> Option<i64> {
        self.get(&key).copied()
    }

    fn put_value(&mut self, key: StrataKey, value: i64) {
        self.insert(key, value);
    }
}

/// A `std::hash::Hasher` over XXH3-64: `write` accumulates the key bytes and `finish` hashes them
/// with `twox_hash::XxHash3_64::oneshot`. The crate's streaming XXH3 hasher is behind its `alloc`
/// feature, which the pinned dependency does not enable.
#[derive(Default)]
struct Xxh3Hasher(Vec<u8>);

impl Hasher for Xxh3Hasher {
    fn finish(&self) -> u64 {
        twox_hash::XxHash3_64::oneshot(&self.0)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
}

/// A `std::hash::Hasher` over BLAKE3: `write` feeds the key bytes to `blake3::Hasher`, `finish`
/// returns the first eight bytes of the digest.
#[derive(Default)]
struct Blake3Hasher(blake3::Hasher);

impl Hasher for Blake3Hasher {
    fn finish(&self) -> u64 {
        let digest = self.0.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest.as_bytes()[..8]);
        u64::from_le_bytes(bytes)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}

/// A `std::hash::Hasher` over SHA-256: `write` feeds the key bytes to `sha2::Sha256`, `finish`
/// returns the first eight bytes of the digest.
#[derive(Default)]
struct Sha256Hasher(sha2::Sha256);

impl Hasher for Sha256Hasher {
    fn finish(&self) -> u64 {
        use sha2::Digest;
        let digest = self.0.clone().finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[..8]);
        u64::from_le_bytes(bytes)
    }

    fn write(&mut self, bytes: &[u8]) {
        use sha2::Digest;
        self.0.update(bytes);
    }
}

/// Measure every serial adapter over the strata witness's admitted cells. A cell that the selected
/// implementation did not measure yields no adapter outcomes, so its pairs stay invalid and reject.
pub(crate) fn measure(witness: &StrataWitness) -> Vec<SerialCellOutcome> {
    let mut cells = Vec::with_capacity(witness.cells.len());
    for cell in &witness.cells {
        cells.push(measure_cell(cell, witness));
    }
    cells
}

/// An outcome with no adapters, for a cell that was not measured.
fn empty_cell(cell: &CellSamples) -> SerialCellOutcome {
    SerialCellOutcome {
        cell_id: cell.cell_id.clone(),
        outcomes: Vec::new(),
    }
}

/// Reconstruct one cell's workload and run every adapter on it.
fn measure_cell(cell: &CellSamples, witness: &StrataWitness) -> SerialCellOutcome {
    if cell.apeiron_samples.is_empty() || cell.population == 0 {
        return empty_cell(cell);
    }
    let spec = CellSpec {
        occupancy: cell.occupancy,
        distribution: cell.distribution,
        operation: cell.operation,
        merge: cell.merge,
    };
    let (capacity, bits) = probe_capacity(cell.population);
    let Ok(plan) = make_plan(spec, capacity, bits) else {
        return empty_cell(cell);
    };
    let trace = build_trace(spec, &plan);
    let merge = merge_fn(spec.merge);
    let selected_medians =
        campaign_medians(&cell.apeiron_samples, witness.campaigns, witness.samples);
    let ctx = Campaign {
        keys: &plan.present,
        lookups: &trace.lookups,
        inserts: &trace.inserts,
        merge,
        oracle: cell.oracle_digest,
        campaigns: witness.campaigns,
        samples: witness.samples,
        selected_medians: &selected_medians,
    };
    SerialCellOutcome {
        cell_id: cell.cell_id.clone(),
        outcomes: run_adapters(&ctx),
    }
}

/// Split pooled samples into per-campaign medians.
fn campaign_medians(pooled: &[u64], campaigns: u32, samples: u32) -> Vec<u64> {
    let per = usize::try_from(samples).unwrap_or(0);
    let count = usize::try_from(campaigns).unwrap_or(0);
    let mut medians = Vec::with_capacity(count);
    for campaign in 0..count {
        let start = campaign.saturating_mul(per);
        let end = start.saturating_add(per).min(pooled.len());
        medians.push(if start >= pooled.len() {
            0
        } else {
            median(&pooled[start..end])
        });
    }
    medians
}

/// Every registered serial adapter, in report order.
fn run_adapters(ctx: &Campaign<'_>) -> Vec<SerialOutcome> {
    vec![
        run_adapter::<FoldhashMap>("hashbrown_foldhash", ctx),
        run_adapter::<FxhashMap>("hashbrown_fxhash", ctx),
        run_adapter::<SiphashMap>("std_siphash", ctx),
        run_adapter::<Xxh3Map>("xxhash_xxh3", ctx),
        run_adapter::<Xxh64Map>("xxhash_xxh64", ctx),
        run_adapter::<WyhashMap>("wyhash", ctx),
        run_adapter::<AHashMap>("ahash", ctx),
        run_adapter::<FnvMap>("fnv", ctx),
        run_adapter::<Blake3Map>("blake3", ctx),
        run_adapter::<Sha256Map>("sha256", ctx),
    ]
}

/// Run one adapter's warmup and sample campaign, checking every sample's digest against the cell's
/// oracle digest.
fn run_adapter<M: SerialMap>(id: &'static str, ctx: &Campaign<'_>) -> SerialOutcome {
    let per_campaign = usize::try_from(ctx.samples).unwrap_or(0);
    let campaign_count = usize::try_from(ctx.campaigns).unwrap_or(0);
    let mut samples = Vec::with_capacity(per_campaign.saturating_mul(campaign_count));
    let mut campaign_ratios = Vec::with_capacity(campaign_count);
    let mut digest_ok = true;
    for campaign in 0..ctx.campaigns {
        for _ in 0..STRATA_WARMUP {
            let mut map = build_map::<M>(ctx);
            touch(&map, ctx.keys);
            let _ = run_sample::<M>(&mut map, ctx);
        }
        let mut campaign_samples = Vec::with_capacity(per_campaign);
        for _ in 0..ctx.samples {
            let mut map = build_map::<M>(ctx);
            touch(&map, ctx.keys);
            let outcome = run_sample::<M>(&mut map, ctx);
            digest_ok &= outcome.digest == ctx.oracle;
            campaign_samples.push(outcome.nanos);
        }
        let selected = ctx
            .selected_medians
            .get(usize::try_from(campaign).unwrap_or(usize::MAX))
            .copied()
            .unwrap_or(0);
        campaign_ratios.push(ratio(selected, median(&campaign_samples)));
        samples.extend_from_slice(&campaign_samples);
    }
    SerialOutcome {
        id,
        samples,
        campaign_ratios,
        digest_ok,
    }
}

/// Build the adapter's map over the admitted key set with the oracle's initial values.
fn build_map<M: SerialMap>(ctx: &Campaign<'_>) -> M {
    let mut map = M::with_capacity(ctx.keys.len());
    for key in ctx.keys {
        map.put_value(*key, (ctx.merge)(initial_value(*key), 0));
    }
    map
}

/// Touch every admitted key so the map is resident before a timed trace.
fn touch<M: SerialMap>(map: &M, keys: &[StrataKey]) {
    let mut acc = 0i64;
    for key in keys {
        acc = acc.wrapping_add(map.get_value(*key).unwrap_or(0));
    }
    black_box(acc);
}

/// Run one timed schedule against the adapter, then the untimed canonical readback. A serial map
/// never refuses, so the refusal fold is the same zero an admitted update folds.
fn run_sample<M: SerialMap>(map: &mut M, ctx: &Campaign<'_>) -> SampleOutcome {
    let mut hash = FNV_OFFSET;
    let start = Instant::now();
    for key in ctx.lookups {
        fold(&mut hash, map.get_value(*key).unwrap_or(0));
    }
    let lookup_nanos = nanos_since(start);
    let start = Instant::now();
    for (key, value) in ctx.inserts {
        let old = map.get_value(*key).unwrap_or(0);
        map.put_value(*key, (ctx.merge)(*value, old));
        fold(&mut hash, 0);
        fold(&mut hash, *value);
    }
    let insert_nanos = nanos_since(start);
    for key in ctx.keys {
        fold(&mut hash, map.get_value(*key).unwrap_or(0));
    }
    SampleOutcome {
        digest: hash,
        nanos: lookup_nanos.saturating_add(insert_nanos),
    }
}
