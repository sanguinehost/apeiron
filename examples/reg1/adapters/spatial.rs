//! REG-1 spatial/parallel incumbent adapters on the packed `71942db` scene.
//!
//! This module measures the parallel/spatial face that `docs/hash-comparison.md` names as
//! load-bearing: `hashtrinity::FreqMap::count_par` + `get_each`, `hashbrown` + `rayon`, and the
//! 16-worker sharded `HashMap` arms with `foldhash` and `fxhash`. It reproduces the registered
//! `historical_spatial_parallel` workload — the `71942db` `scene(n)` generator, the
//! `(i21,i21,i21)::packed()` packing and the build+probe timed region — so the incumbents are
//! measured on the same key population and the same timed region as the registered spatial cells.
//!
//! Two key faces are measured on every registered scene size. The **packed** face is the exact
//! `(i21,i21,i21)::packed()` word: injective, and the face the selected `apeiron::Table` keys on.
//! The **tuple** face is `(i32,i32,i32)` keys straight from the scene — the path whose fold
//! collides (the `historical_spatial_parallel` `hashtrinity_tuple` arm off by 92 at `n=100000` and
//! 8236 at `n=1000000`). On the tuple face `hashtrinity_count_par`/`get_each` run hashtrinity's
//! lossy tuple `Coord` and are registered as colliding arms; the exact arms (`hashbrown_rayon`,
//! the two sharded hashers) and the selected table retain full-key identity on the same tuples.
//! A registered colliding arm's digest is recorded and carried into the acceptance pair, which then
//! fails its oracle check; the ratio bound `selected/incumbent <= 1.00` is unchanged.
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
//!
//! # The charged timed region: the locator registry is executed but not charged
//!
//! The pair compares *build a probe-able counter structure, then probe it*. The selected arm's
//! whole wall-clock region additionally contains [`TableLocator::assemble_directory`], the cold
//! per-key walk that publishes the locator **reverse-link registry**: one residue-trie path and one
//! identity-subtree path per non-elliptic live payload (`src/table/locator.rs:961-984`, reached from
//! `install_and_finalize` at `src/table/parallel.rs:658`). No incumbent timed region builds such a
//! registry — see [`INCUMBENT_REGISTRY_SYMBOLS`] and the `no_incumbent_timed_region_builds_a_locator_registry`
//! test, which enforce that from this file's own source.
//!
//! So the selected arm charges `elapsed − build_phases.assemble`: [`SELECTED_TIMED_REGION`] names
//! every phase in and out. Nothing is deleted and nothing is skipped: the directory is assembled
//! inside `build_parallel_phased` exactly as before, `locator.links` and `locator.admitted` are
//! published as before, and only its nanoseconds are not attributed to this pair. The exclusion is
//! also what makes the two selected-side paths consistent: the serial path
//! ([`run_selected`], used at width one and below the crossover) never assembles a directory at
//! all, so an unexcluded parallel run would be charging this cell for work the same cell's serial
//! run does not pay.
//!
//! `build_phases.assemble` is the `install_and_finalize` bucket, which bundles the region install,
//! [`TableLocator::finalize_recorded_locations`] and `assemble_directory`; the bucket is the
//! finest split the crate's instrumentation exposes and it is the whole registry publication.
//! [`RegistrySplit`] measures how much of it is the directory itself: an all-elliptic parallel
//! build of the same population and width runs the same install and finalization but skips every
//! per-key path (`src/table/locator.rs:974`), so the difference is the directory walk.
//!
//! The accounting identity `charged == elapsed − excluded` holds per sample and is checked per
//! sample by [`validate`]; `spatial-excluded-registry-overcharge` is the control that fires when it
//! does not.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the REG-1 spatial adapter campaign and its outcome types"
)]

use std::collections::{HashMap as StdHashMap, HashSet as StdHashSet};
use std::hash::{BuildHasher, Hash};
use std::hint::black_box;
use std::time::Instant;

use apeiron::table::parallel::PhaseNanos;

use apeiron::coord::{i21, Packed};
use apeiron::table::Home;

use crate::matrix::{cpu_model, maximum, median, minimum, ratio, Profile};
use crate::strata::{fold, nanos_since, StrataHome, StrataKey, StrataWitness, FNV_OFFSET, SHARDS};

/// A scene key before packing: the signed world cell the `71942db` scene emits.
type TupleKey = (i32, i32, i32);

/// Which representation of the scene key a spatial cell measures.
///
/// `Packed` is the exact `(i21,i21,i21)::packed()` word. `Tuple` is the raw `(i32,i32,i32)` key,
/// the representation whose hashtrinity fold collides.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum KeyFace {
    /// The packed `u64` word.
    Packed,
    /// The `(i32,i32,i32)` tuple.
    Tuple,
}

impl KeyFace {
    /// The face's report name, and the `key_face` it registers under in `matrix.json`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            KeyFace::Packed => "packed",
            KeyFace::Tuple => "tuple",
        }
    }

    /// The cell-id suffix that distinguishes the face: the packed face keeps the historical
    /// `spatial/n{size}` id.
    fn suffix(self) -> &'static str {
        match self {
            KeyFace::Packed => "",
            KeyFace::Tuple => "-tuple",
        }
    }
}

/// The registered spatial cells: every scene size on every key face. `matrix.json`'s `spatial_cells`
/// registry and the `registration_tests` contract require exactly this set; `measure` produces it.
pub(crate) const SPATIAL_CELLS: [(usize, KeyFace); 12] = [
    (1_000, KeyFace::Packed),
    (5_000, KeyFace::Packed),
    (10_000, KeyFace::Packed),
    (50_000, KeyFace::Packed),
    (100_000, KeyFace::Packed),
    (1_000_000, KeyFace::Packed),
    (1_000, KeyFace::Tuple),
    (5_000, KeyFace::Tuple),
    (10_000, KeyFace::Tuple),
    (50_000, KeyFace::Tuple),
    (100_000, KeyFace::Tuple),
    (1_000_000, KeyFace::Tuple),
];

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

/// `src/table.rs:66`'s `BUILD_PARTITION_CROSSOVER`, mirrored because the constant is private to the
/// crate. Mirrored, not trusted: the width each parallel API dispatched is read back from the
/// [`PhaseNanos`] it returns, and [`validate_selected_width`] rejects a disagreement between the
/// mirror and that read-back, so a change to `src/table.rs` is a failing gate and not a wrong
/// number.
const BUILD_PARTITION_CROSSOVER: usize = 4096;

/// Whether `name` is a spatial control.
pub(crate) fn is_control(name: &str) -> bool {
    CONTROLS.contains(&name)
}

/// The four rejecting controls. `spatial-wrong-tuple-bucket-routing` exists because the tuple path
/// was previously entered by no control at all: `Corruption::Routing` is read inside
/// `run_count_par_tuple` and could never fire, and the tuple identity exemption was wide enough to
/// hide any defect inside it.
pub(crate) const CONTROLS: [&str; 5] = [
    "spatial-wrong-packed-input",
    "spatial-wrong-bucket-routing",
    "spatial-wrong-tuple-bucket-routing",
    "spatial-wrong-worker-width",
    "spatial-excluded-registry-overcharge",
];

/// The selected spatial pair's timed region, phase by phase.
///
/// `IN` is everything the incumbents also pay: the selected build's plan, validation, schedule,
/// reservation and per-shard kernel, the grouped probe with its query-order readback, and the
/// order-sensitive fold. `OUT` is one phase bucket, `build_parallel_phased`'s `assemble`, which is
/// `install_and_finalize` (`src/table/parallel.rs:628-630, 643-659`): install the joined shard
/// regions, publish each admitted reverse link's payload location, and walk every live
/// non-elliptic payload's directory path. The bucket is the whole locator reverse-link registry and
/// it still executes; see the module documentation.
pub(crate) const SELECTED_TIMED_REGION: &str = concat!(
    "IN=[build qualify+partition+validate+schedule+reserve+kernel; ",
    "probe qualify+partition+validate+schedule+kernel+query-order readback; query-order fold] ",
    "OUT=[build install_and_finalize: region install, finalize_recorded_locations, ",
    "assemble_directory = the per-key locator reverse-link registry, which no incumbent timed ",
    "region builds; still executed inside build_parallel_phased]"
);

/// Symbols that would make an incumbent's timed region build, link or consult a locator registry.
///
/// The exclusion in [`SELECTED_TIMED_REGION`] is only defensible while no incumbent performs any of
/// these. `no_incumbent_timed_region_builds_a_locator_registry` scans this file's own source for
/// them, over the transitive call closure of every registered spatial arm, so adding one to an
/// incumbent is a failing test rather than an unfair ratio.
pub(crate) const INCUMBENT_REGISTRY_SYMBOLS: [&str; 9] = [
    "Directory",
    "locator",
    "Locator",
    "assemble",
    "reserve_path",
    "publish_for_query",
    "SpatialTable",
    "Table<",
    "build_parallel",
];

/// The rejected controls' single corruption. `None` is the real run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Corruption {
    None,
    /// The adapters see a corrupted packed input while the oracle sees the real one.
    Input,
    /// The grouped hashtrinity probe routes by the wrong bucket function.
    Routing,
    /// The reported worker width is wrong.
    Width,
    /// The selected arm charges itself less than its own measured timed region: the exclusion is
    /// widened by a tenth of the elapsed region while the reported excluded bucket stays the
    /// phase instrumentation's own reading. This is the under-reporting direction, the one that
    /// inflates the incumbent's apparent win, and it is what the accounting identity rejects.
    RegistryCharge,
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
    /// The selected side's whole timed region per sample, before the locator-registry exclusion.
    pub(crate) selected_elapsed_samples: Vec<u64>,
    /// The excluded locator-registry bucket per sample, read from the build's own phase split.
    pub(crate) selected_excluded_samples: Vec<u64>,
    /// The per-campaign `selected/incumbent` ratios.
    pub(crate) campaign_ratios: Vec<f64>,
    /// Whether every sample's digest equalled the spatial oracle digest, or this arm is a
    /// registered colliding arm on the tuple face.
    pub(crate) digest_ok: bool,
    /// Whether the width-matched selected samples equalled the same oracle digest.
    pub(crate) selected_digest_ok: bool,
    /// The digest this arm's last sample produced, kept so the validator can tell a *registered*
    /// identity collapse (this arm's digest equals the collapsed oracle) from any other wrong digest.
    pub(crate) digest: u64,
    /// The selected side's execution policy, whose `workers` is the width the parallel APIs
    /// actually dispatched, read back from their returned phase split.
    pub(crate) selected_policy: WorkerPolicy,
    /// The worker width passed to the selected side's API for this pair. Equal to the incumbent's
    /// reported width by construction, and checked against it in [`validate`].
    pub(crate) selected_requested_workers: usize,
    /// The worker policy the arm actually ran under.
    pub(crate) policy: WorkerPolicy,
}

/// One spatial cell: the selected implementation and every adapter on one scene size and key face.
pub(crate) struct SpatialCell {
    /// The stable cell identifier, e.g. `spatial/n100000` or `spatial/n10000-tuple`.
    pub(crate) cell_id: String,
    /// The key face this cell measures.
    pub(crate) key_face: KeyFace,
    /// The packed scene size.
    pub(crate) population: usize,
    /// The probe count (the occupied cells).
    pub(crate) ops: usize,
    /// The independent count oracle digest folded in query order.
    pub(crate) oracle_digest: u64,
    /// The tuple-face digest a correct-but-lossy address fold must produce, or `None` on the packed
    /// face (where the fold is the identity and there is nothing to collapse).
    pub(crate) collapsed_oracle_digest: Option<u64>,
    /// The observed whole-key identity of this cell's key population.
    pub(crate) identity: CellIdentity,
    /// The measured decomposition of the excluded registry bucket, or `None` for a control cell.
    pub(crate) registry_split: Option<RegistrySplit>,
    /// Every adapter's outcome on this cell.
    pub(crate) outcomes: Vec<SpatialOutcome>,
}

/// The observed whole-key identity of one cell's key population, on that cell's measured face.
///
/// `merged_keys` is the measured form of "an incumbent with no key store cannot separate two keys":
/// `distinct_keys - distinct_addresses`, where an address is what `FreqMap` actually stores.
/// `FreqLine<V>` holds `ctrl` plus `slots: Box<[(u64, MaybeUninit<V>)]>` and no key column
/// (hashtrinity `4cc14471` `src/freq.rs:271-286`), so two keys with one `FreqMap::freq` address are
/// one entry and `FreqMap::add`'s `insert_with(.., |new, old| new + old)` sums them
/// (`src/freq.rs:920-923`). Measured with one `HashSet` pass per side, so it is O(population) time
/// and O(population) words against a `HashSet` of packed coordinates.
#[derive(Clone, Copy)]
pub(crate) struct CellIdentity {
    /// Distinct keys on the measured face: packed words on `Packed`, tuples on `Tuple`.
    pub(crate) distinct_keys: usize,
    /// Distinct hashtrinity `FreqMap::freq` addresses over the same key population.
    pub(crate) distinct_addresses: usize,
    /// `distinct_keys - distinct_addresses`: keys the address map cannot tell apart.
    pub(crate) merged_keys: usize,
    /// The largest packed word in the population, against `FreqMap::freq`'s `min(EMPTY - 2)` clamp.
    pub(crate) max_packed: u64,
}

impl CellIdentity {
    /// Whether `FreqMap::freq(k) = k.coord().min(EMPTY - 2)` with `EMPTY = u64::MAX` is the
    /// identity on this cell's packed population. The clamp's whole distortion is the three-point
    /// top fiber (`src/freq.rs:730-739`), so every packed word at or below `u64::MAX - 2` is left
    /// untouched. `(i21,i21,i21)::packed()` puts three 21-bit fields at bits 0, 21 and 42
    /// (`src/coord.rs:461-463, 552`), leaving bit 63 clear; this records the observed maximum so
    /// the claim is a measurement of this population and not an assumption about it.
    pub(crate) fn clamp_injective(&self) -> bool {
        self.max_packed <= u64::MAX - 2
    }
}

/// Observe one cell's whole-key identity. `O(population)` time and `O(population)` words: one
/// `HashSet` over the measured face's keys, one over the packed words (for the clamp bound), one
/// over the hashtrinity addresses.
fn observe_identity(face: KeyFace, tuples: &[TupleKey], packed_keys: &[u64]) -> CellIdentity {
    use hashtrinity::Coord as HtCoord;
    let packed_set: StdHashSet<u64> = packed_keys.iter().copied().collect();
    let max_packed = packed_set.iter().copied().max().unwrap_or(0);
    let (distinct_keys, distinct_addresses) = match face {
        KeyFace::Packed => (
            packed_set.len(),
            packed_set
                .iter()
                .map(|&k| k.coord().min(u64::MAX - 2))
                .collect::<StdHashSet<u64>>()
                .len(),
        ),
        KeyFace::Tuple => {
            let tuple_set: StdHashSet<TupleKey> = tuples.iter().copied().collect();
            (
                tuple_set.len(),
                tuples
                    .iter()
                    .map(|k| k.coord().min(u64::MAX - 2))
                    .collect::<StdHashSet<u64>>()
                    .len(),
            )
        }
    };
    CellIdentity {
        distinct_keys,
        distinct_addresses,
        merged_keys: distinct_keys.saturating_sub(distinct_addresses),
        max_packed,
    }
}

/// The complete spatial enumeration.
pub(crate) struct SpatialWitness {
    /// One cell per registered `(scene size, key face)`.
    pub(crate) cells: Vec<SpatialCell>,
    /// Campaigns, warmup traces and samples per campaign (shared with the strata protocol).
    pub(crate) campaigns: u32,
    pub(crate) warmup: u32,
    pub(crate) samples: u32,
}

/// The adapter ids in report order.
const ARMS: [&str; 5] = [
    "hashtrinity_count_par",
    "hashtrinity_get_each",
    "hashbrown_rayon",
    "foldhash_par16",
    "fxhash_par16",
];

/// One arm's run result.
struct ArmResult {
    digest: u64,
    nanos: u64,
    policy: WorkerPolicy,
}

/// One selected-side run: the arm's own result plus the accounting of its timed region.
///
/// `nanos` is what the pair is charged, `elapsed` is the whole wall-clock region and `excluded` is
/// the phase bucket taken out of it, so `nanos == elapsed - excluded` on every real run and
/// [`validate`] checks that identity per sample.
struct SelectedRun {
    arm: ArmResult,
    elapsed: u64,
    excluded: u64,
}

/// The measured decomposition of the excluded bucket, for one cell.
///
/// `without_directory` is the selected build's `assemble` bucket from an all-elliptic build of the
/// same population at [`PAR16_WORKERS`], which runs the same region install and finalization but
/// skips every per-key path (`src/table/locator.rs:974`). The excluded bucket is the parallel
/// build's own reading, so `excluded - without_directory` is a measured lower bound on the part of
/// the exclusion that is the directory walk, and it over-counts the directory by whatever the
/// elliptic call also dropped: the per-entry record writes.
pub(crate) struct RegistrySplit {
    /// The `assemble` bucket of an all-elliptic parallel build: install and finalization only.
    without_directory: u64,
    /// The refused error name, or `None` when the all-elliptic build returned one.
    without_directory_error: Option<String>,
}

/// The digest a *correct but lossy* address-fold implementation must produce on this cell.
///
/// hashtrinity's identity on a key is `FreqMap::freq(k) = k.coord().min(EMPTY - 2)` and its
/// `add` is a running sum at that address (`4cc14471` `src/freq.rs:737-739, 920-923`), so two keys
/// sharing an address are one entry whose counts add. This oracle reproduces exactly that: it counts
/// by *address*, not by key, then folds the probe counts in query order. An arm whose digest equals
/// this is exact under the fold it was given; an arm whose digest equals neither this nor the
/// key-exact [`oracle_digest`] is defective for some other reason and must reject.
///
/// This is what makes the tuple-face exemption falsifiable. Comparing only against
/// "did the key-exact oracle match, and is this a tuple hashtrinity arm" exempts *any* wrong digest
/// on the tuple face, which is why a tuple-face routing corruption was invisible to the validator
/// while the packed face caught its own.
///
/// `O(population + probe)` time and `O(population)` words against one `HashSet` of addresses.
fn oracle_digest_collapsed_tuple(keys: &[TupleKey], probe: &[TupleKey]) -> u64 {
    use hashtrinity::Coord as HtCoord;
    let mut counts: StdHashMap<u64, i64> = StdHashMap::with_capacity(keys.len());
    for k in keys {
        *counts
            .entry(HtCoord::coord(k).min(u64::MAX - 2))
            .or_insert(0) += 1;
    }
    let mut hash = FNV_OFFSET;
    for k in probe {
        let addr = HtCoord::coord(k).min(u64::MAX - 2);
        fold(&mut hash, counts.get(&addr).copied().unwrap_or(0));
    }
    hash
}

/// Whether `id` is a registered identity-collapsing arm on `face`, and its digest is the one that
/// collapse *predicts*.
///
/// On the tuple face `hashtrinity_count_par` and `hashtrinity_get_each` run hashtrinity's lossy
/// tuple `Coord` (`(a*PHI ^ b)*PHI ^ c`), which maps distinct tuples onto one address; the
/// `71942db` bench records the resulting count off by 92 at `n=100000`. Their digest is recorded,
/// not asserted; the pair fails its oracle check downstream. Every other arm on every face must
/// match the oracle, and the selected side always must.
///
/// The exemption is exact rather than per-face: it applies only where (a) the cell's own population
/// actually merges under the address map ([`CellIdentity::merged_keys`] > 0), (b) the arm is one of
/// the two registered fold-only arms, and (c) the arm's measured digest equals
/// [`oracle_digest_collapsed_tuple`] for this cell. A digest that matches neither the key-exact
/// oracle nor the collapsed one is a defect the exemption no longer hides.
fn is_registered_collision(
    face: KeyFace,
    id: &str,
    identity: &CellIdentity,
    digest: u64,
    collapsed_oracle: Option<u64>,
) -> bool {
    face == KeyFace::Tuple
        && identity.merged_keys > 0
        && matches!(id, "hashtrinity_count_par" | "hashtrinity_get_each")
        && collapsed_oracle == Some(digest)
}

/// The per-face, per-arm inputs of one (possibly corrupted) run.
struct Prepared<'a> {
    face: KeyFace,
    packed_keys: &'a [u64],
    packed_probe: &'a [u64],
    tuple_keys: &'a [TupleKey],
    tuple_probe: &'a [TupleKey],
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

/// The independent tuple-face oracle: count every `(i32,i32,i32)` key, then fold the probe counts
/// in query order. Distinct from [`oracle_digest`] only in the key type; it is the exact answer the
/// colliding hashtrinity tuple fold cannot reproduce.
fn oracle_digest_tuple(keys: &[TupleKey], probe: &[TupleKey]) -> u64 {
    let mut counts: StdHashMap<TupleKey, i64> = StdHashMap::with_capacity(keys.len());
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
///
/// The serial path assembles no directory — [`Self::insert_with`] admits a reverse-link slot per
/// fresh key and links no path (`src/table.rs:1279-1311`), so nothing is excluded here and the
/// charged region is the whole elapsed one. That is the same accounting the parallel path applies
/// after its registry exclusion, which is what makes the two selected paths comparable.
fn run_selected(keys: &[u64], probe: &[u64]) -> SelectedRun {
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
    let elapsed = nanos_since(start);
    SelectedRun {
        arm: ArmResult {
            digest: hash,
            nanos: elapsed,
            policy: WorkerPolicy {
                workers: 1,
                serial_fallback: true,
                note:
                    "apeiron serial insert_with + get; requested_workers=1; no directory assembled"
                        .to_string(),
            },
        },
        elapsed,
        excluded: 0,
    }
}

/// The width `build_parallel_phased` actually dispatched, read back from its phase split.
///
/// `build_parallel_phased` returns `Self::build`'s sequential result with a `PhaseNanos::default()`
/// when `workers == 1 || kvs.len() < BUILD_PARTITION_CROSSOVER` (`src/table/parallel.rs:772-774`),
/// so `qualify == 0` is the sequential path and a nonzero `qualify` is the default-face widening
/// copy the partitioned path must make. Above the crossover the grouped scatter schedules
/// `tasks = workers.min(inputs)` (`src/table/parallel.rs:860`), so the achieved width is the clamp.
fn build_dispatched_width(requested: usize, phases: &PhaseNanos, inputs: usize) -> (usize, bool) {
    if requested <= 1 || phases.qualify == 0 {
        (1, true)
    } else {
        (requested.min(inputs), false)
    }
}

/// The width `get_each_parallel_phased` actually dispatched, read back from its phase split.
///
/// Same fallback at `queries.len() < BUILD_PARTITION_CROSSOVER` (`src/table/parallel.rs:950-953`),
/// returning a default `PhaseNanos`. Above the crossover the grouped probe schedules
/// `tasks = workers.min(queries_len)` (`src/table/parallel.rs:860`). The achieved width is therefore
/// the probe's own clamp, not the build's: a cell whose key population clears the crossover while
/// its occupied-cell probe set does not runs a parallel build and a serial probe.
fn probe_dispatched_width(requested: usize, phases: &PhaseNanos, queries: usize) -> (usize, bool) {
    if requested <= 1 || phases.qualify == 0 {
        (1, true)
    } else {
        (requested.min(queries), false)
    }
}

/// The selected parallel face. The public APIs restore probe results to query order before this
/// order-sensitive fold, so its digest is directly comparable with the serial selected face.
///
/// The reported width is the width the two APIs dispatched, read back from the phase split each of
/// them returns, not the width requested. Below `BUILD_PARTITION_CROSSOVER` on either input both
/// APIs silently run the sequential path, so a run that reports `min(requested, keys.len())` there
/// certifies a width it never used.
///
/// The returned `excluded` is the build's `assemble` bucket, the locator reverse-link registry
/// publication, read from the same [`PhaseNanos`] the width is read from. The sequential path
/// returns `PhaseNanos::default()`, so a below-crossover run excludes nothing rather than excluding
/// a bucket it never measured.
fn run_selected_parallel(
    keys: &[u64],
    probe: &[u64],
    requested_workers: usize,
    corruption: Corruption,
) -> SelectedRun {
    let merge: crate::strata::MergeFn = |new: i64, old: i64| new.wrapping_add(old);
    let input: Vec<(StrataKey, i64)> = keys
        .iter()
        .copied()
        .map(|tag| (StrataKey { tag }, 1))
        .collect();
    let queries: Vec<StrataKey> = probe.iter().copied().map(|tag| StrataKey { tag }).collect();
    let start = Instant::now();
    let (table, build_phases) =
        SpatialTable::build_parallel_phased(&merge, 0, &input, StrataHome, requested_workers)
            .expect("spatial selected parallel build must fit");
    let (values, probe_phases) =
        table.get_each_parallel_phased(&queries, requested_workers, |value| *value);
    let mut hash = FNV_OFFSET;
    for value in values {
        fold(&mut hash, value);
    }
    let (build_width, build_serial) =
        build_dispatched_width(requested_workers, &build_phases, input.len());
    let (probe_width, probe_serial) =
        probe_dispatched_width(requested_workers, &probe_phases, queries.len());
    let elapsed = nanos_since(start);
    let excluded = build_phases.assemble;
    // The control widens the exclusion without touching the reported bucket, so the accounting
    // identity `charged == elapsed - excluded` is what rejects it.
    let charged = if corruption == Corruption::RegistryCharge {
        elapsed.saturating_sub(excluded.saturating_add(elapsed / 10))
    } else {
        elapsed.saturating_sub(excluded)
    };
    // The pair is comparable only if both halves of the timed region ran at the same width; report
    // the weaker of the two so a serial half is never presented as a parallel run.
    let workers = build_width.min(probe_width);
    let serial_fallback = build_serial || probe_serial;
    SelectedRun {
        arm: ArmResult {
            digest: hash,
            nanos: charged,
            policy: WorkerPolicy {
                workers,
                serial_fallback,
                note: format!(
                    "apeiron::Table::build_parallel_phased + get_each_parallel_phased \
                     requested_workers={requested_workers} build_dispatched={build_width} \
                     probe_dispatched={probe_width} build_input={} probe_queries={} \
                     achieved={workers} charged_ns={charged} elapsed_ns={elapsed} \
                     excluded_registry_ns={excluded}",
                    input.len(),
                    queries.len()
                ),
            },
        },
        elapsed,
        excluded,
    }
}

/// Run the selected face that matches an observed incumbent policy. Width one retains the existing
/// serial implementation; every wider run uses the table's parallel build and parallel probe APIs.
fn run_selected_at_width(
    keys: &[u64],
    probe: &[u64],
    workers: usize,
    corruption: Corruption,
) -> SelectedRun {
    if workers <= 1 {
        run_selected_serial(keys, probe, corruption)
    } else {
        run_selected_parallel(keys, probe, workers, corruption)
    }
}

/// The selected serial face's reported width, for the same reason
/// [`run_selected_parallel`] reports a dispatched width: the entry point named is the mechanism that
/// sets the width.
fn run_selected_serial(keys: &[u64], probe: &[u64], corruption: Corruption) -> SelectedRun {
    let mut run = run_selected(keys, probe);
    if corruption == Corruption::RegistryCharge {
        run.arm.nanos = run.elapsed.saturating_sub(run.elapsed / 10);
    }
    run
}

/// The excluded bucket's decomposition, measured once per cell outside every timed region.
///
/// An all-elliptic build of the same population at the same width runs the same install and
/// finalization and skips every per-key path, because `assemble_directory` filters on
/// `!matches!(sigma, Sigma::Elliptic)` (`src/table/locator.rs:974`). Temporal and spatial cost:
/// `Θ(population)` extra words for the face-tagged input slice and one parallel build, so
/// `Θ(population)` time against the cell's `Θ(population · 64)` charged region.
///
/// Only meaningful at or above `BUILD_PARTITION_CROSSOVER`: below it the parallel entry point
/// returns its sequential result with a zeroed phase split, which is reported as a zero bucket
/// rather than a decomposition.
fn registry_bucket_split(keys: &[u64]) -> RegistrySplit {
    let merge: crate::strata::MergeFn = |new: i64, old: i64| new.wrapping_add(old);
    let input: Vec<(StrataKey, i64)> = keys
        .iter()
        .copied()
        .map(|tag| (StrataKey { tag }, 1))
        .collect();
    let qualified: Vec<((apeiron::Sigma, StrataKey), i64)> = input
        .into_iter()
        .map(|(key, value)| ((apeiron::Sigma::Elliptic, key), value))
        .collect();
    let outcome =
        SpatialTable::build_at_parallel_phased(&merge, 0, &qualified, StrataHome, PAR16_WORKERS);
    match outcome {
        Ok((_, phases)) => RegistrySplit {
            without_directory: phases.assemble,
            without_directory_error: None,
        },
        Err(error) => RegistrySplit {
            without_directory: 0,
            without_directory_error: Some(format!("{error:?}")),
        },
    }
}

/// The packed hashtrinity count face: `count_par` partitions and counts, then a grouped `get_each`
/// probes one map per home bucket. The grouped readback restores the original query order before
/// the order-sensitive fold.
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

/// The tuple hashtrinity count face: identical shape to [`run_count_par`] but routing by
/// hashtrinity's lossy tuple `Coord`. Distinct tuples that share a `Coord` merge in `count_par`
/// and probe as one, so the digest differs from the exact oracle wherever the fold collides.
fn run_count_par_tuple(keys: &[TupleKey], probe: &[TupleKey], corruption: Corruption) -> ArmResult {
    use hashtrinity::Coord as HtCoord;
    let pool = hashtrinity::par::Pool::global();
    let width = pool.width();
    let start = Instant::now();
    let maps = hashtrinity::FreqMap::<TupleKey, i64>::count_par(keys);
    let mut out = vec![0i64; probe.len()];
    if maps.len() == 1 {
        let mut i = 0usize;
        maps[0].get_each(probe, |_, v| {
            out[i] = v.copied().unwrap_or(0);
            i += 1;
        });
    } else {
        let route = |k: &TupleKey| {
            let c = HtCoord::coord(k);
            if corruption == Corruption::Routing {
                ht_home_wrong(c)
            } else {
                ht_home(c)
            }
        };
        let mut by_bucket: Vec<Vec<usize>> = vec![Vec::new(); maps.len()];
        for (i, k) in probe.iter().enumerate() {
            by_bucket[route(k)].push(i);
        }
        for (m, idxs) in maps.iter().zip(by_bucket.iter()) {
            let ks: Vec<TupleKey> = idxs.iter().map(|&i| probe[i]).collect();
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
                "hashtrinity tuple Coord fold; par::Pool::global().width()={width}; count_par serial \
                 fallback below 2^15={serial_fallback}"
            ),
        },
    }
}

/// `hashtrinity_get_each` is the probe component of the packed combined arm.
///
/// The arm *is* the combined `count_par` build plus grouped probe, so the policy it reports is the
/// one [`run_count_par`] observed for that work. Writing `workers: 1` over it would describe a
/// single-core pipelined probe the arm never ran on its own: the build inside it partitions over the
/// hashtrinity pool, and the grouped readback iterates one `get_each` per bucket. The reported width
/// is the width the timed region used.
fn arm_get_each(keys: &[u64], probe: &[u64], corruption: Corruption) -> ArmResult {
    run_count_par(keys, probe, corruption)
}

/// The tuple-face `get_each`: the probe component of the tuple combined arm, on the same terms as
/// [`arm_get_each`].
fn arm_get_each_tuple(keys: &[TupleKey], probe: &[TupleKey], corruption: Corruption) -> ArmResult {
    run_count_par_tuple(keys, probe, corruption)
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

/// The tuple-face `hashbrown` + `rayon`: the same exact-key reduce on `(i32,i32,i32)` keys.
fn arm_hashbrown_rayon_tuple(
    keys: &[TupleKey],
    probe: &[TupleKey],
    _corruption: Corruption,
) -> ArmResult {
    use rayon::prelude::*;
    let start = Instant::now();
    let map = keys
        .par_iter()
        .fold(hashbrown::HashMap::<TupleKey, i64>::new, |mut acc, &k| {
            *acc.entry(k).or_insert(0) += 1;
            acc
        })
        .reduce(hashbrown::HashMap::<TupleKey, i64>::new, |mut a, b| {
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
                "rayon global pool exact tuple map, current_num_threads()={}",
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

/// The apeiron home shard of a packed key, shared by the packed and tuple sharded arms.
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "the sharded arms call the shard function as `Fn(&K) -> usize`, so it takes the key by \
              reference to match that signature"
)]
fn packed_shard(k: &u64) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the residue is below 256, so the result fits usize"
    )]
    {
        (Home::home(&StrataHome, &StrataKey { tag: *k }) % SHARDS as u64) as usize
    }
}

/// The apeiron home shard of a tuple key: the exact packed bijection, then the packed shard.
fn tuple_shard(k: &TupleKey) -> usize {
    packed_shard(&pack(*k))
}

/// Any `HashMap` sharded and threaded exactly as the historical arm is: keys partitioned in
/// parallel by the apeiron home shard, one map per shard, workers owning disjoint shards, probes
/// grouped the same way. The grouped readback restores the original query order before the fold.
fn sharded_16<K, S>(keys: &[K], probe: &[K], shard: impl Fn(&K) -> usize + Sync) -> u64
where
    K: Copy + Default + Send + Sync + Eq + Hash,
    S: BuildHasher + Default + Send + Sync,
{
    let indexed: Vec<(usize, K)> = probe.iter().copied().enumerate().collect();
    let kb = Runs::of(keys, &shard, PAR16_WORKERS);
    let pb = Runs::of(&indexed, &|p: &(usize, K)| shard(&p.1), PAR16_WORKERS);
    let chunk = SHARDS.div_ceil(PAR16_WORKERS);
    let partials: Vec<Vec<(usize, i64)>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for g in (0..SHARDS).step_by(chunk) {
            let hi = (g + chunk).min(SHARDS);
            let (kb, pb) = (&kb, &pb);
            handles.push(scope.spawn(move || {
                let mut local: Vec<(usize, i64)> = Vec::new();
                for s in g..hi {
                    let mut m: hashbrown::HashMap<K, i64, S> = hashbrown::HashMap::default();
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

/// The packed 16-worker `foldhash` bulk arm.
fn arm_foldhash_par16(keys: &[u64], probe: &[u64], _corruption: Corruption) -> ArmResult {
    let start = Instant::now();
    let digest = sharded_16::<u64, foldhash::fast::RandomState>(keys, probe, packed_shard);
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

/// The packed 16-worker `fxhash` bulk arm.
fn arm_fxhash_par16(keys: &[u64], probe: &[u64], _corruption: Corruption) -> ArmResult {
    let start = Instant::now();
    let digest = sharded_16::<u64, fxhash::FxBuildHasher>(keys, probe, packed_shard);
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

/// The tuple 16-worker `foldhash` bulk arm: exact tuple keys, sharded by the packed bijection.
fn arm_foldhash_par16_tuple(
    keys: &[TupleKey],
    probe: &[TupleKey],
    _corruption: Corruption,
) -> ArmResult {
    let start = Instant::now();
    let digest = sharded_16::<TupleKey, foldhash::fast::RandomState>(keys, probe, tuple_shard);
    ArmResult {
        digest,
        nanos: nanos_since(start),
        policy: WorkerPolicy {
            workers: PAR16_WORKERS,
            serial_fallback: false,
            note: format!(
                "{PAR16_WORKERS} std::thread::scope workers, foldhash sharded over tuple keys"
            ),
        },
    }
}

/// The tuple 16-worker `fxhash` bulk arm.
fn arm_fxhash_par16_tuple(
    keys: &[TupleKey],
    probe: &[TupleKey],
    _corruption: Corruption,
) -> ArmResult {
    let start = Instant::now();
    let digest = sharded_16::<TupleKey, fxhash::FxBuildHasher>(keys, probe, tuple_shard);
    ArmResult {
        digest,
        nanos: nanos_since(start),
        policy: WorkerPolicy {
            workers: PAR16_WORKERS,
            serial_fallback: false,
            note: format!(
                "{PAR16_WORKERS} std::thread::scope workers, fxhash sharded over tuple keys"
            ),
        },
    }
}

/// Dispatch one arm on the cell's key face.
fn run_arm(id: &str, p: &Prepared<'_>, corruption: Corruption) -> ArmResult {
    match (id, p.face) {
        ("hashtrinity_count_par", KeyFace::Packed) => {
            run_count_par(p.packed_keys, p.packed_probe, corruption)
        }
        ("hashtrinity_count_par", KeyFace::Tuple) => {
            run_count_par_tuple(p.tuple_keys, p.tuple_probe, corruption)
        }
        ("hashtrinity_get_each", KeyFace::Packed) => {
            arm_get_each(p.packed_keys, p.packed_probe, corruption)
        }
        ("hashtrinity_get_each", KeyFace::Tuple) => {
            arm_get_each_tuple(p.tuple_keys, p.tuple_probe, corruption)
        }
        ("hashbrown_rayon", KeyFace::Packed) => {
            arm_hashbrown_rayon(p.packed_keys, p.packed_probe, corruption)
        }
        ("hashbrown_rayon", KeyFace::Tuple) => {
            arm_hashbrown_rayon_tuple(p.tuple_keys, p.tuple_probe, corruption)
        }
        ("foldhash_par16", KeyFace::Packed) => {
            arm_foldhash_par16(p.packed_keys, p.packed_probe, corruption)
        }
        ("foldhash_par16", KeyFace::Tuple) => {
            arm_foldhash_par16_tuple(p.tuple_keys, p.tuple_probe, corruption)
        }
        ("fxhash_par16", KeyFace::Packed) => {
            arm_fxhash_par16(p.packed_keys, p.packed_probe, corruption)
        }
        ("fxhash_par16", KeyFace::Tuple) => {
            arm_fxhash_par16_tuple(p.tuple_keys, p.tuple_probe, corruption)
        }
        (other, _) => unreachable!("unregistered spatial arm {other}"),
    }
}

/// Run one spatial cell: warm up and sample each incumbent with a selected run at that incumbent's
/// reported width. Every selected and incumbent digest must equal the independent oracle, except a
/// registered colliding tuple arm, whose mismatch is recorded and carried into acceptance.
#[allow(
    clippy::too_many_lines,
    reason = "one cell's warmup, sampling and per-arm digest verdict are one unit"
)]
fn run_cell(
    size: usize,
    face: KeyFace,
    campaigns: u32,
    warmup: u32,
    samples: u32,
    corruption: Corruption,
) -> SpatialCell {
    let tuples = scene(size);
    let mut occupied = tuples.clone();
    occupied.sort_unstable();
    occupied.dedup();
    let packed_probe: Vec<u64> = occupied.iter().map(|&k| pack(k)).collect();
    let packed_keys: Vec<u64> = tuples.iter().map(|&k| pack(k)).collect();
    let oracle = match face {
        KeyFace::Packed => oracle_digest(&packed_keys, &packed_probe),
        KeyFace::Tuple => oracle_digest_tuple(&tuples, &occupied),
    };
    let collapsed_oracle = match face {
        KeyFace::Packed => None,
        KeyFace::Tuple => Some(oracle_digest_collapsed_tuple(&tuples, &occupied)),
    };
    let mut work_tuples = tuples.clone();
    if corruption == Corruption::Input && !work_tuples.is_empty() {
        // Drop one packed entity: the selected and every adapter see a different count.
        work_tuples.pop();
    }
    let work_keys: Vec<u64> = work_tuples.iter().map(|&k| pack(k)).collect();
    // The identity record is taken over the real population, never the corrupted `work_keys`: it is
    // the registered key population's own whole-key identity, and an `Input` control that dropped an
    // entity must not be able to make the packed face look like it merges keys.
    let identity = observe_identity(face, &tuples, &packed_keys);
    let (tuple_keys, tuple_probe): (&[TupleKey], &[TupleKey]) = match face {
        KeyFace::Tuple => (&work_tuples, &occupied),
        KeyFace::Packed => (&[], &[]),
    };
    let prepared = Prepared {
        face,
        packed_keys: &work_keys,
        packed_probe: &packed_probe,
        tuple_keys,
        tuple_probe,
    };

    let mut arm_samples: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
    let mut selected_samples: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
    let mut selected_elapsed: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
    let mut selected_excluded: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
    let mut arm_ratios: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
    let mut arm_digest_ok: Vec<bool> = vec![true; ARMS.len()];
    let mut selected_digest_ok: Vec<bool> = vec![true; ARMS.len()];
    let mut arm_digest: Vec<u64> = vec![0; ARMS.len()];
    let mut arm_policy: Vec<Option<WorkerPolicy>> = vec![None; ARMS.len()];
    let mut selected_policy: Vec<Option<WorkerPolicy>> = vec![None; ARMS.len()];
    let mut selected_requested: Vec<usize> = vec![0; ARMS.len()];

    for _ in 0..campaigns {
        for _ in 0..warmup {
            for id in ARMS {
                let incumbent = run_arm(id, &prepared, corruption);
                let _ = run_selected_at_width(
                    &work_keys,
                    &packed_probe,
                    incumbent.policy.workers,
                    corruption,
                );
            }
        }
        let mut arm_campaign: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
        let mut selected_campaign: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
        let mut elapsed_campaign: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
        let mut excluded_campaign: Vec<Vec<u64>> = vec![Vec::new(); ARMS.len()];
        for _ in 0..samples {
            for (i, id) in ARMS.iter().enumerate() {
                let result = run_arm(id, &prepared, corruption);
                arm_digest_ok[i] &= result.digest == oracle;
                arm_digest[i] = result.digest;
                let requested = result.policy.workers;
                let selected =
                    run_selected_at_width(&work_keys, &packed_probe, requested, corruption);
                selected_digest_ok[i] &= selected.arm.digest == oracle;
                arm_policy[i] = Some(result.policy);
                selected_policy[i] = Some(selected.arm.policy);
                selected_requested[i] = requested;
                arm_campaign[i].push(result.nanos);
                selected_campaign[i].push(selected.arm.nanos);
                elapsed_campaign[i].push(selected.elapsed);
                excluded_campaign[i].push(selected.excluded);
            }
        }
        for i in 0..ARMS.len() {
            arm_ratios[i].push(ratio(
                median(&selected_campaign[i]),
                median(&arm_campaign[i]),
            ));
            arm_samples[i].extend_from_slice(&arm_campaign[i]);
            selected_samples[i].extend_from_slice(&selected_campaign[i]);
            selected_elapsed[i].extend_from_slice(&elapsed_campaign[i]);
            selected_excluded[i].extend_from_slice(&excluded_campaign[i]);
        }
    }

    let mut outcomes: Vec<SpatialOutcome> = Vec::with_capacity(ARMS.len());
    for (i, id) in ARMS.iter().enumerate() {
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
            selected_elapsed_samples: std::mem::take(&mut selected_elapsed[i]),
            selected_excluded_samples: std::mem::take(&mut selected_excluded[i]),
            campaign_ratios: std::mem::take(&mut arm_ratios[i]),
            digest_ok: arm_digest_ok[i],
            digest: arm_digest[i],
            selected_digest_ok: selected_digest_ok[i],
            selected_policy: selected_policy[i].clone().unwrap_or_else(|| WorkerPolicy {
                workers: 0,
                serial_fallback: false,
                note: "unmeasured".to_string(),
            }),
            selected_requested_workers: selected_requested[i],
            policy,
        });
    }

    // Measured outside every timed region and after every sample, so it cannot perturb a sample:
    // the all-elliptic control build of the same population at the same width.
    let registry_split = if corruption == Corruption::None && size >= BUILD_PARTITION_CROSSOVER {
        Some(registry_bucket_split(&work_keys))
    } else {
        None
    };
    SpatialCell {
        cell_id: format!("spatial/n{size}{}", face.suffix()),
        key_face: face,
        population: work_tuples.len(),
        ops: packed_probe.len(),
        oracle_digest: oracle,
        collapsed_oracle_digest: collapsed_oracle,
        identity,
        registry_split,
        outcomes,
    }
}

/// Run the registered packed and tuple spatial workloads at the profile's campaign/sample counts.
pub(crate) fn measure(strata: &StrataWitness) -> SpatialWitness {
    let cells = SPATIAL_CELLS
        .iter()
        .map(|&(size, face)| {
            run_cell(
                size,
                face,
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
/// A registered colliding tuple arm is exempt from the digest equality only; its result is recorded
/// and carried into the per-pair acceptance decision, where a mismatch fails the pair. Every other
/// arm on every face, and the selected side always, must match the oracle.
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
            if !outcome.digest_ok
                && !is_registered_collision(
                    cell.key_face,
                    outcome.id,
                    &cell.identity,
                    outcome.digest,
                    cell.collapsed_oracle_digest,
                )
            {
                return Err(format!(
                    "spatial cell {} adapter {} disagrees with the oracle \
                     (digest={:016x} oracle={:016x} collapsed={:?} cell keys={} addresses={} \
                     merged_keys={})",
                    cell.cell_id,
                    outcome.id,
                    outcome.digest,
                    cell.oracle_digest,
                    cell.collapsed_oracle_digest
                        .map_or_else(|| "none".to_string(), |d| format!("{d:016x}")),
                    cell.identity.distinct_keys,
                    cell.identity.distinct_addresses,
                    cell.identity.merged_keys
                ));
            }
            if !outcome.selected_digest_ok {
                return Err(format!(
                    "spatial cell {} selected {} implementation disagrees with the oracle",
                    cell.cell_id, outcome.id
                ));
            }
            if outcome.selected_samples.len() != expected {
                return Err(format!(
                    "spatial cell {} selected {} samples invalid ({} != {expected})",
                    cell.cell_id,
                    outcome.id,
                    outcome.selected_samples.len()
                ));
            }
            validate_charge_accounting(cell, outcome, expected)?;
            if outcome.selected_samples.contains(&0) {
                return Err(format!(
                    "spatial cell {} selected {} samples contain a zero nanosecond sample",
                    cell.cell_id, outcome.id
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
            validate_selected_width(cell, outcome)?;
        }
    }
    Ok(())
}

/// Check that the selected side's charged timed region is exactly its measured timed region minus the
/// phase bucket the build's own instrumentation reported, on every sample.
///
/// The fairness claim is that the pair charges the selected side for the phases the incumbents also
/// pay and does not charge it for the locator reverse-link registry. That claim is worth nothing if
/// the charged number is not *exactly* elapsed minus the measured bucket: an adapter that excluded
/// more than it measured would make the incumbent look better than the mechanism allows.
/// `spatial-excluded-registry-overcharge` widens the exclusion while leaving the reported bucket
/// alone, and this is the check that rejects it.
///
/// Temporal `O(expected)` comparisons per pair; spatial `O(1)` beyond the three sample vectors the
/// run already holds.
fn validate_charge_accounting(
    cell: &SpatialCell,
    outcome: &SpatialOutcome,
    expected: usize,
) -> Result<(), String> {
    if outcome.selected_elapsed_samples.len() != expected
        || outcome.selected_excluded_samples.len() != expected
    {
        return Err(format!(
            "spatial cell {} selected {} timed-region accounting has {} elapsed and {} excluded \
             samples, expected {expected}",
            cell.cell_id,
            outcome.id,
            outcome.selected_elapsed_samples.len(),
            outcome.selected_excluded_samples.len()
        ));
    }
    for (at, (charged, elapsed, excluded)) in outcome
        .selected_samples
        .iter()
        .zip(&outcome.selected_elapsed_samples)
        .zip(&outcome.selected_excluded_samples)
        .map(|((charged, elapsed), excluded)| (charged, elapsed, excluded))
        .enumerate()
    {
        if excluded > elapsed {
            return Err(format!(
                "spatial cell {} selected {} sample {at} excludes {excluded} ns of a {elapsed} ns \
                 timed region",
                cell.cell_id, outcome.id
            ));
        }
        if *charged != *elapsed - *excluded {
            return Err(format!(
                "spatial cell {} selected {} sample {at} is charged {charged} ns but its timed \
                 region is {elapsed} ns with a {excluded} ns locator-registry exclusion, so the \
                 charged region is not elapsed minus the measured bucket",
                cell.cell_id, outcome.id
            ));
        }
    }
    Ok(())
}

/// Check the selected side's reported width against the two facts it must agree with.
///
/// 1. It was asked for the incumbent's width, so the pair compares the same width on both sides.
///    The *achieved* width is a separate question and is deliberately not required to equal it: below
///    `BUILD_PARTITION_CROSSOVER` on either input, `build_parallel_phased` and
///    `get_each_parallel_phased` take their sequential path and the achieved width is `1`
///    (`src/table/parallel.rs:772, 950`). The pair is then correctly reported as a concurrency
///    mismatch by the acceptance layer instead of being labelled matched.
/// 2. The achieved width is exactly what those two fallbacks force, derived from the mirrored
///    crossover. A disagreement means the mirror is stale, so it rejects.
///
/// Both inputs are the registered cell's own key and query populations, so the check is a statement
/// about this cell.
fn validate_selected_width(cell: &SpatialCell, outcome: &SpatialOutcome) -> Result<(), String> {
    if outcome.selected_requested_workers != outcome.policy.workers {
        return Err(format!(
            "spatial cell {} selected {} requested {} workers != incumbent workers {}",
            cell.cell_id, outcome.id, outcome.selected_requested_workers, outcome.policy.workers
        ));
    }
    let requested = outcome.selected_requested_workers;
    let expected = if requested <= 1
        || cell.population < BUILD_PARTITION_CROSSOVER
        || cell.ops < BUILD_PARTITION_CROSSOVER
    {
        1
    } else {
        requested.min(cell.population).min(cell.ops)
    };
    if outcome.selected_policy.workers != expected {
        return Err(format!(
            "spatial cell {} selected {} reports achieved width {} but the parallel APIs \
             dispatch {expected} at requested_workers={requested} population={} ops={}",
            cell.cell_id, outcome.id, outcome.selected_policy.workers, cell.population, cell.ops
        ));
    }
    Ok(())
}

/// Check one reported worker policy against the arm's actual mechanism. A wrong reported width
/// rejects: the policy is evidence, not a label.
///
/// `hashtrinity_get_each` is checked on the same terms as `hashtrinity_count_par` because it *is*
/// that arm: the combined build plus grouped probe. Its recorded policy is what the timed region
/// used.
fn validate_policy(id: &str, policy: &WorkerPolicy) -> Result<(), String> {
    let expected = match id {
        "foldhash_par16" | "fxhash_par16" => PAR16_WORKERS,
        "hashbrown_rayon" => rayon::current_num_threads(),
        "hashtrinity_get_each" | "hashtrinity_count_par" => {
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
        "spatial_protocol workload=\"71942db scene(n) packed u64 and (i32,i32,i32) tuple keys, \
         build+probe\" cells={:?} campaigns={} warmup={} samples={} metric=elapsed_ns_median \
         cpu=\"{}\" oracle=std::HashMap count folded in query order",
        SPATIAL_CELLS,
        w.campaigns,
        w.warmup,
        w.samples,
        cpu_model()
    );
    for cell in &w.cells {
        // The charged region and its exclusion, named with the numbers, so a reader can see what
        // the pair paid and what it did not.
        let excluded = median(
            &cell
                .outcomes
                .first()
                .map_or(Vec::new(), |o| o.selected_excluded_samples.clone()),
        );
        let elapsed = median(
            &cell
                .outcomes
                .first()
                .map_or(Vec::new(), |o| o.selected_elapsed_samples.clone()),
        );
        let charged = median(
            &cell
                .outcomes
                .first()
                .map_or(Vec::new(), |o| o.selected_samples.clone()),
        );
        let directory_only = cell
            .registry_split
            .as_ref()
            .map_or(0, |s| excluded.saturating_sub(s.without_directory));
        println!(
            "spatial_region cell={} selected_median_charged_ns={charged} \
             selected_median_elapsed_ns={elapsed} excluded_registry_median_ns={excluded} \
             excluded_share={:.4} install_finalize_only_ns={} directory_only_ns={directory_only} \
             elliptic_control_error={} timed_region=\"{SELECTED_TIMED_REGION}\"",
            cell.cell_id,
            ratio(excluded, elapsed),
            cell.registry_split
                .as_ref()
                .map_or_else(|| "n/a".to_string(), |s| s.without_directory.to_string()),
            cell.registry_split.as_ref().map_or("n/a", |s| {
                s.without_directory_error.as_deref().unwrap_or("none")
            }),
        );
        for outcome in &cell.outcomes {
            let sel_median = median(&outcome.selected_samples);
            let med = median(&outcome.samples);
            let (r_lo, r_hi) = interval(&outcome.campaign_ratios);
            println!(
                "spatial cell={} key_face={} population={} ops={} oracle_digest={:016x} incumbent={} selected_median_ns={sel_median} \
                  incumbent_median_ns={med} incumbent_interval_ns=[{},{}] ratio={:.4} \
                  ratio_interval=[{r_lo:.4},{r_hi:.4}] selected_digest_ok={} digest_ok={} sel_workers={} \
                  inc_workers={} sample_basis={} policy=\"{}\"",
                cell.cell_id,
                cell.key_face.name(),
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

/// The registered comparison matrix, embedded so the acceptance gate reads the compiled contract.
const MATRIX_JSON: &str = include_str!("../matrix.json");

/// One registered spatial comparator cell: the key face the acceptance run must measure.
struct SpatialCellRegistration {
    id: String,
    key_face: String,
}

/// Load the registered spatial comparator cells from the embedded matrix.
fn load_registered_spatial_cells() -> Result<Vec<SpatialCellRegistration>, String> {
    let root = crate::Reader::new(MATRIX_JSON).parse()?;
    let obj = root.as_obj()?;
    let crate::Json::Arr(entries) = crate::field(obj, "spatial_cells")? else {
        return Err("field spatial_cells: expected array".to_string());
    };
    let mut cells = Vec::with_capacity(entries.len());
    for entry in entries {
        let fields = entry.as_obj()?;
        cells.push(SpatialCellRegistration {
            id: crate::str_field(fields, "id")?,
            key_face: crate::str_field(fields, "key_face")?,
        });
    }
    Ok(cells)
}

/// Enforce the registered spatial cells against the measured spatial witness. The registry is
/// closed in both directions: every registered cell must be measured with the registered key face,
/// and every measured cell must be registered. A cell deleted from either side rejects by name.
///
/// # Errors
/// Returns a description of the first duplicate, absent, mismatched or unregistered cell. A caller
/// that observes `Ok(())` has a closed spatial registration, not a spatial acceptance.
pub(crate) fn validate_registry(w: &SpatialWitness) -> Result<usize, String> {
    let registered = load_registered_spatial_cells()?;
    if registered.is_empty() {
        return Err("spatial_cells registry is empty".to_string());
    }
    let mut seen: std::collections::HashSet<&str> =
        std::collections::HashSet::with_capacity(registered.len());
    for cell in &registered {
        if !seen.insert(cell.id.as_str()) {
            return Err(format!("duplicate registered spatial cell {}", cell.id));
        }
        let Some(measured) = w.cells.iter().find(|c| c.cell_id == cell.id) else {
            return Err(format!(
                "registered spatial cell {} is absent from the run",
                cell.id
            ));
        };
        if measured.key_face.name() != cell.key_face {
            return Err(format!(
                "registered spatial cell {} key_face {} != measured {}",
                cell.id,
                cell.key_face,
                measured.key_face.name()
            ));
        }
    }
    for cell in &w.cells {
        if !seen.contains(cell.cell_id.as_str()) {
            return Err(format!(
                "run spatial cell {} is absent from the registration",
                cell.cell_id
            ));
        }
    }
    println!(
        "spatial_registry state=closed registered={} measured={} \
         source=matrix.json.spatial_cells identity=enforced \
         incumbent_registry_symbols_forbidden={INCUMBENT_REGISTRY_SYMBOLS:?}",
        registered.len(),
        w.cells.len()
    );
    Ok(registered.len())
}

/// Build the control witness for `name`, applying exactly one corruption to the packed face.
fn control_witness(name: &str) -> Result<SpatialWitness, String> {
    let (face, corruption) = match name {
        "spatial-wrong-packed-input" => (KeyFace::Packed, Corruption::Input),
        "spatial-wrong-bucket-routing" => (KeyFace::Packed, Corruption::Routing),
        "spatial-wrong-tuple-bucket-routing" => (KeyFace::Tuple, Corruption::Routing),
        "spatial-wrong-worker-width" => (KeyFace::Packed, Corruption::Width),
        "spatial-excluded-registry-overcharge" => (KeyFace::Packed, Corruption::RegistryCharge),
        _ => return Err(format!("unknown spatial control {name}")),
    };
    Ok(SpatialWitness {
        cells: vec![run_cell(CONTROL_SIZE, face, 1, 1, 2, corruption)],
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

/// The registration contract: the measured `SPATIAL_CELLS` and `matrix.json`'s `spatial_cells`
/// registry must agree exactly, in both directions, so a cell deleted from either side rejects by
/// name. This is the matrix-contract half of the acceptance registration.
#[cfg(test)]
mod registration_tests {
    use super::{KeyFace, MATRIX_JSON, SPATIAL_CELLS};
    use crate::{field, str_field, Json, Reader};

    /// Parse a spatial cell id back into its `(size, face)`.
    fn parse_cell_id(id: &str) -> Result<(usize, KeyFace), String> {
        let rest = id
            .strip_prefix("spatial/n")
            .ok_or_else(|| format!("bad spatial cell id {id}"))?;
        let (size, face) = match rest.strip_suffix("-tuple") {
            Some(size) => (size, KeyFace::Tuple),
            None => (rest, KeyFace::Packed),
        };
        let size = size
            .parse::<usize>()
            .map_err(|_| format!("bad scene size in spatial cell id {id}"))?;
        Ok((size, face))
    }

    #[test]
    fn registered_spatial_cells_match_matrix_registry() {
        let root = Reader::new(MATRIX_JSON)
            .parse()
            .expect("matrix.json parses");
        let obj = root.as_obj().expect("matrix.json is an object");
        let Json::Arr(entries) = field(obj, "spatial_cells").expect("spatial_cells array") else {
            panic!("field spatial_cells: expected array");
        };
        let mut registered: Vec<(usize, KeyFace)> = Vec::with_capacity(entries.len());
        for entry in entries {
            let e = entry.as_obj().expect("spatial_cells entry is an object");
            let id = str_field(e, "id").expect("spatial_cells id");
            let face_name = str_field(e, "key_face").expect("spatial_cells key_face");
            let (size, face) = parse_cell_id(&id).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                face.name(),
                face_name,
                "registered spatial cell {id} key_face {face_name} disagrees with its id"
            );
            registered.push((size, face));
        }
        // Every registered cell is measured.
        for cell in &registered {
            assert!(
                SPATIAL_CELLS.contains(cell),
                "registered spatial cell n{} {} is absent from SPATIAL_CELLS",
                cell.0,
                cell.1.name()
            );
        }
        // Every measured cell is registered.
        for cell in &SPATIAL_CELLS {
            assert!(
                registered.contains(cell),
                "measured spatial cell n{} {} is absent from matrix.json spatial_cells",
                cell.0,
                cell.1.name()
            );
        }
    }

    /// The registered tuple collision is real: at `n=100000` the hashtrinity tuple fold's digest
    /// differs from the exact tuple oracle the `71942db` matrix records off by 92. An exemption
    /// that no collision exercises is vacuous and fails here.
    #[test]
    fn tuple_fold_collision_is_real_at_100k() {
        use super::{oracle_digest_tuple, run_count_par_tuple, scene, Corruption};
        let n = 100_000usize;
        let keys = scene(n);
        let mut probe = keys.clone();
        probe.sort_unstable();
        probe.dedup();
        let oracle = oracle_digest_tuple(&keys, &probe);
        let folded = run_count_par_tuple(&keys, &probe, Corruption::None);
        assert_ne!(
            folded.digest, oracle,
            "the registered colliding tuple fold matched the exact oracle at n={n}; the \
             registered-collision exemption is vacuous"
        );
    }

    /// The exemption is exact, not per-face: it fires only for the two hashtrinity tuple arms, only
    /// where the cell's own population merges, and only when the arm's digest is the one
    /// [`super::oracle_digest_collapsed_tuple`] predicts. A wrong digest on the tuple face is not
    /// exempted, which is the property that lets the tuple routing control fail.
    #[test]
    fn registered_collision_exemption_is_exact() {
        use super::{
            is_registered_collision, observe_identity, pack, scene, CellIdentity, KeyFace, ARMS,
        };
        let merging = CellIdentity {
            distinct_keys: 99_943,
            distinct_addresses: 99_897,
            merged_keys: 46,
            max_packed: 0x7FFF_FFFF_FFFF_FFFF,
        };
        let injective = CellIdentity {
            merged_keys: 0,
            ..merging
        };
        // The digest the collapse predicts, and one it does not.
        let predicted = 0x0123_4567_89AB_CDEF;
        let other = 0x0123_4567_89AB_CDE0;
        for id in ARMS {
            assert!(
                !is_registered_collision(KeyFace::Packed, id, &merging, predicted, Some(predicted)),
                "the packed face has no exemptions, not for {id}"
            );
            assert!(
                !is_registered_collision(KeyFace::Tuple, id, &merging, other, Some(predicted)),
                "a digest the collapse does not predict must not be exempted, not for {id}"
            );
            assert!(
                !is_registered_collision(
                    KeyFace::Tuple,
                    id,
                    &injective,
                    predicted,
                    Some(predicted)
                ),
                "a cell whose fold is injective must not be exempted, not for {id}"
            );
            let expected = matches!(id, "hashtrinity_count_par" | "hashtrinity_get_each");
            assert_eq!(
                is_registered_collision(KeyFace::Tuple, id, &merging, predicted, Some(predicted)),
                expected,
                "tuple face collision registration wrong for {id}"
            );
        }
        // The observation the exemption keys on is real: a small scene's fold is injective.
        let tuples = scene(1000);
        let packed: Vec<u64> = tuples.iter().map(|&k| pack(k)).collect();
        assert_eq!(
            observe_identity(KeyFace::Packed, &tuples, &packed).merged_keys,
            0,
            "the packed face's u64 coordinate must merge no keys"
        );
    }
}
