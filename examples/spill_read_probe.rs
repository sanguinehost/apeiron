#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    reason = "a measurement harness: fixed-width key and slot arithmetic is the subject, and the \
              report formats per-key nanosecond and per-key counter quotients as f64 at the point \
              of printing"
)]
//! Present-stream read isolation: the canonical spill read against the resident read.
//!
//! This is the committed gate of record for the E6.W2 spill-read port. Every later slice of that
//! port is graded by running this example, not by a scratch script, because the prototype harness
//! that produced the 6x figure was scratch-only and is gone.
//!
//! The graded acceptance is the matched ratio `spill median / resident median` from one
//! interleaved run (`<= PARITY_RATIO_MAX`, defined below); the per-arm band is a reference print.
//! A correct but slow spill read fails the ratio; the resident arm is the comparator, not the
//! target.
//!
//! ## What it isolates
//!
//! One process, one key type, one home, one build path, two arms over the **present stream only**.
//! There is no absent stream and no insert timing: `examples/table_cost_probe.rs` mixes insert,
//! present and absent streams into one process, so its `present_ns_per_key` is a median over a
//! warmed allocator and three adjacent loops rather than a clean isolation of the read. Here each
//! arm is one table, one loop over the present keys, and nothing else.
//!
//! Geometry matches the `distinct` population of `table_cost_probe.rs`: `N = 16`, `KEYS = 192`,
//! `place = key`, `coord = key`, and the same odd-multiply spread, so the two arms differ only in
//! the address the home returns. The arms are:
//!
//! * `canonical_spill` — `direct_word: None`. Every read routes to a spill region and runs
//!   `spill_read`.
//! * `canonical_resident` — `direct_word: Some` on a *conforming* population: the direct words are
//!   in the certified range (`word < 2^32`) **and inside the shipped resident extent**
//!   (`word < 256`, one lane), so every read is answered from the flat array and `spill_read` never
//!   runs. The second condition is not a detail: a certified direct word whose lane is beyond
//!   `self.resident.lanes` falls through to `spill_read` (`src/table.rs` `get_at`), so a resident
//!   arm built from larger words measures the spill path and reports a non-zero
//!   `sweep_steps_per_key`. The word of key `j` is the *same* shard byte the spill arm's `place`
//!   realises, and `SPREAD` is odd, so the two arms have the identical shard distribution over 192
//!   distinct words.
//!
//! `Home::home` is the identity in both arms, so the home function is not a variable.
//!
//! ## Timing runs are built WITHOUT `read-stats`
//!
//! Two commands, two builds, and the numbers are not interchangeable:
//!
//! ```sh
//! cargo run --release --example spill_read_probe                          # timing
//! cargo run --release --features read-stats --example spill_read_probe    # attribution
//! ```
//!
//! The `read-stats` counters are `thread_local` increments on the canonical read path, so the
//! instrumented build is a *different* build of the read, not the same read plus a cost. Measured
//! on the live tree after a forced rebuild, two runs each: un-instrumented canonical
//! `median_ns_per_key = 9.26, 9.48`; instrumented `11.73, 11.97`; resident `1.42, 1.44` on both
//! builds, because it never reaches a counter. The counters cost about 2.4 ns per spill read.
//!
//! **A timed run binds to a source identity.** An earlier revision of this section reported the
//! instrumented arm at about *half* the un-instrumented one, and a run of the unported tree once
//! reported spill at 2.02 ns/key and "passed": both were builds that shared `target/` with a
//! sibling worktree of the same package (`CARGO_TARGET_DIR` pointed at the live tree by the
//! porting task) and linked its objects. The cached warnings name that tree - `src/table.rs:84`
//! against the live line 97. Force the rebuild (`touch src/lib.rs`) or use a private target
//! directory before quoting a figure, and check the cached warnings' line numbers against the
//! tree you think you built. Two consequences, and both are the reason the numbers are kept in
//! separate builds:
//!
//! * An instrumented run cannot supply a timing figure in either direction. It prints
//!   `mode=instrumented`, its acceptance verdict reads `not-graded`, and the ratio is not
//!   enforced.
//! * The un-instrumented release build is the timed path and the only one a timing claim may
//!   quote. `examples/table_cost_probe.rs` prints its nanosecond columns under both builds for the
//!   same reason, so its `present_ns_per_key` is a per-build figure too.
//!
//! `sweep_steps`, `tag_matches` and `full_compares` are the three per-read stages and are the
//! columns a port slice is graded against. `summary_hits` and `summary_misses` read `0` on this
//! present-hit path while `summary_tests` reads one per key, so they are not the hit/miss oracle
//! here; the wrong-answer counters are.
//!
//! ## Instrument
//!
//! Each timed run is `PASSES_PER_RUN` passes over the 192 present keys, `black_box` on every key
//! and every returned value, and the per-key figure is `elapsed_ns / (PASSES_PER_RUN * KEYS)`.
//! Per-key `Instant::now()` is not used: one clock read is on the order of 20 ns on this host,
//! three times the whole canonical read it would bracket, so a per-key clock would measure the
//! clock. The run is instead long enough (about 100 microseconds of canonical reads) that one clock
//! read per run is noise. The two arms are timed *interleaved*, one run each per iteration with
//! the order alternating, so a preemption or a frequency excursion on a shared box lands on both
//! arms instead of on whichever arm was running; timing one arm to completion and then the other
//! measured a load drift between the arms and a ratio that moved with the box's load.
//! `median_ns_per_key` is the median over `RUNS` runs, `min_ns_per_key` the minimum (the
//! load-free estimate) and `max_ns_per_key` the worst, with `spread_max_over_median` printed so a
//! reader can see whether the box was contended when the numbers were taken.
//!
//! Correctness is checked *after* the timed region, never inside it: every present key must return
//! its own value, and a same-shape absent key one band above must return the table's sentinel
//! default, or the run fails closed. Nothing here is a `lake build` result; the wall clocks are
//! `measured` and the counters are `measured` for the canonical table's own instrumented stages.

use std::hint::black_box;
use std::time::Instant;

use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue,
    WordFaceHome, H3_40,
};
use apeiron::coord::locator::{ExactIdentity, Refusal};
use apeiron::table::{Home, Table, TableAddress};
use apeiron::Sigma;

/// The compile-time group width under measurement: the `Table` default, and the same `N` as
/// `examples/table_cost_probe.rs`.
const N: usize = 16;
/// Keys per arm: the `distinct` population size of `examples/table_cost_probe.rs`.
const KEYS: usize = 192;
/// Untimed runs before sampling.
const WARMUP_RUNS: u32 = 5;
/// Timed runs per arm.
const RUNS: u32 = 31;
/// Present-stream passes inside one timed run, so one clock read brackets about 100 microseconds of
/// canonical reads and about 17 of resident reads.
const PASSES_PER_RUN: usize = 64;
/// The odd multiplier that spreads the population, transcribed from `examples/table_cost_probe.rs`
/// `SPREAD`.
const SPREAD: u64 = 0x9E37_79B9_7F4A_7C15;
/// The odd offset that separates this harness's population from the raw multiply, transcribed from
/// `examples/table_cost_probe.rs` `distinct_population`.
const SEED: u64 = 0x1234_5678_9ABC_DEF0;
/// The width of one lane in direct words, `SHARD_BITS = 8`. A certified direct word is inside the
/// shipped resident extent only when `word >> 8` is below `self.resident.lanes`, and for a
/// 192-key population that extent is *one* lane: `selected_lanes` keeps the smallest power-of-two
/// lane count whose capacity `SHARDS * lanes` satisfies `lanes * SHARDS <= max(SHARDS, 2 * density)`
/// (`src/table.rs`), which is 256 for any population of at most 256 live direct words. So the
/// resident arm's words must satisfy `word < LANE_WORDS`, and a word at or above it is answered by
/// `spill_read` even though the home certified it. This harness measures that rather than assuming
/// it: the attributed run reports `sweep_steps_per_key`, which must be `0.0000` for the resident
/// arm.
const LANE_WORDS: u64 = 256;
/// The stride of the absent control family, in units of `SPREAD`. The absent word of key `j` is
/// `place_j + SPREAD * ABSENT_STRIDE`, so the family is `SPREAD * (j + ABSENT_STRIDE) + SEED`; with
/// `ABSENT_STRIDE > KEYS` it lies outside the present index range, and multiplication by odd
/// `SPREAD` is a bijection on `Z/2^64`, so no other difference in the family can coincide with it.
/// A plain `place | band` does not have that property: three quarters of the spread words already
/// carry bit 40, so the control word would *be* a present key.
const ABSENT_STRIDE: u64 = 4096;
/// The reference band on either arm's `median_ns_per_key`, in nanoseconds per key. It is printed
/// per arm so a reader can place each median; the acceptance below is a matched ratio, not
/// membership in this band, because the box inflates both arms together and an absolute band
/// grades the box's load as much as the code.
///
/// Measured on the host named in the header, un-instrumented build. Quiet: resident
/// `median = 1.34`, `min = 1.26`, canonical spill `median = 8.81`, ratio `6.56`. Contended (load
/// average 16 on a shared box): resident `median = 1.97`, canonical spill `median = 13.07`, ratio
/// `5.83`; one of those runs carried a 124 microsecond outlier, which is why the minimum and the
/// spread are printed beside every median. The band upper bound is about 2.2x the quiet resident
/// figure, the same factor the acceptance admits.
const PARITY_BAND_NS_PER_KEY: (f64, f64) = (0.90, 3.00);
/// The graded acceptance for the E6.W2 port: the matched ratio `spill median / resident median`,
/// measured in one interleaved run, must be at most this. Both arms are timed run-by-run in the
/// same process, so the ratio is load-invariant and it is the criterion that can reject a spill
/// read that is correct but slow (the pre-port ratio is 5.75..6.60). The port target is within
/// about 2x of the resident read, ideally parity; the stricter `<=1.0` criterion is G1's
/// production-vs-predecessor comparison, not this probe's. Enforced only on the un-instrumented
/// build, where both terms are un-instrumented.
const PARITY_RATIO_MAX: f64 = 2.2;
/// The table's default, and therefore the value a miss returns. `Table::get` returns `&V` and hands
/// back the default on absence (`src/table.rs` `get_at`), so a miss is detected by this sentinel
/// rather than by an `Option`. No key's value may equal it, which `measure` checks before it reads
/// anything.
const DEFAULT_SENTINEL: i64 = i64::MIN;

// ── The key and the one identity home ───────────────────────────────────────────────────────────

/// One key. The word *is* the home word: identity is the tag, and the placement is the tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    tag: u64,
}

/// The identity home, `place = key`, in both arms.
///
/// `direct` is the only difference between the arms. It is an explicit field rather than a
/// consequence of the word's magnitude, so arm (a) is *exactly* `direct_word: None` for every key
/// and arm (b) is *exactly* `direct_word: Some` for every key of a conforming population, with no
/// word-by-word fallback that could route one key of the resident arm into spill.
#[derive(Clone, Copy)]
struct IdentityHome {
    direct: bool,
}

impl IdentityHome {
    /// Arm (a): no certified direct word, so every read and every insert routes to a spill region.
    const SPILL: Self = Self { direct: false };
    /// Arm (b): a certified direct word, so a word inside the resident extent is answered from the
    /// flat array.
    const RESIDENT: Self = Self { direct: true };
}

impl Home<Key> for IdentityHome {
    fn home(&self, key: &Key) -> u64 {
        key.tag
    }

    fn address(&self, _sigma: Sigma, key: &Key) -> TableAddress {
        TableAddress {
            place: key.tag,
            direct_word: self.direct.then(|| u32::try_from(key.tag).ok()).flatten(),
        }
    }
}

/// The exact identity is the whole 64-bit tag as eight big-endian bytes: the home word is the key,
/// so nothing may reduce, truncate or hash it.
impl ExactIdentity for Key {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        self.tag
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        self.tag
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        self.tag == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.tag.to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        retained.to_be_bytes()[index]
    }
}

impl FaceResidueCapability<Key> for IdentityHome {
    fn parabolic_dual(&self, key: &Key) -> ParabolicDual {
        WordFaceHome.parabolic_dual(&key.tag)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &Key) -> Residue {
        WordFaceHome.hyperbolic_scalar(profile, &key.tag)
    }
}

impl EllipticPacking<Key> for IdentityHome {
    fn elliptic_pack(&self, key: &Key) -> u64 {
        WordFaceHome.elliptic_pack(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<Key> {
        WordFaceHome.elliptic_unpack(word).map(|tag| Key { tag })
    }
}

impl FaceCoord<Key> for IdentityHome {
    type Faces = Sigma;
}

impl FaceHome<Key> for IdentityHome {}

// ── The two populations, one shared geometry ───────────────────────────────────────────────────

/// Arm (a)'s population: the `distinct` population of `examples/table_cost_probe.rs`, 192 distinct
/// words spread over the full width by one fixed odd multiply. Every word is above `2^32`, so
/// `direct_word` is `None` under any conforming reading of the address.
fn spill_keys() -> Vec<Key> {
    (0..KEYS as u64)
        .map(|j| Key {
            tag: j.wrapping_mul(SPREAD).wrapping_add(SEED),
        })
        .collect()
}

/// Arm (b)'s population: the *same* 192 keys, each reduced to a certified direct word that keeps
/// the shard byte its spill twin realises.
///
/// The resident flat index is `shard * lanes + lane`, with `shard = word & 0xff` and
/// `lane = word >> 8` (`src/table/state.rs` `Flat::index`), and the shipped extent is one lane
/// (`LANE_WORDS`), so a resident word for this population *is* its shard byte. `SPREAD` is odd, so
/// `place & 0xff` takes 192 distinct values over `j < 192`: the two arms have the identical shard
/// distribution, every resident key is inside the extent, and the flat array is the 256-slot array
/// one lane names.
fn resident_keys(spill: &[Key]) -> Vec<Key> {
    spill
        .iter()
        .map(|key| Key {
            tag: key.tag & 0xff,
        })
        .collect()
}

/// The absent control family: the same words, displaced by `SPREAD * ABSENT_STRIDE`, so no member
/// is a present key. Not timed and not read by either arm; it exists so a present loop that reads a
/// constant or a stale slot is caught.
fn absent_keys(keys: &[Key]) -> Vec<Key> {
    let stride = SPREAD.wrapping_mul(ABSENT_STRIDE);
    keys.iter()
        .map(|key| Key {
            tag: key.tag.wrapping_add(stride),
        })
        .collect()
}

// ── One arm ─────────────────────────────────────────────────────────────────────────────────────

/// The read-path stage counts of one attributed pass. Feature `read-stats` only.
#[cfg(feature = "read-stats")]
#[derive(Clone, Copy, Default)]
struct ReadStages {
    summary_tests: u64,
    summary_hits: u64,
    summary_misses: u64,
    sweep_steps: u64,
    tag_matches: u64,
    full_compares: u64,
}

/// What one arm reports.
struct Arm {
    name: &'static str,
    admitted: usize,
    len: usize,
    runs: u32,
    /// Per-run nanoseconds per key, in run order; `median_ns_per_key` and `min_ns_per_key` read
    /// this.
    ns_per_key: Vec<f64>,
    /// How many keys of the arm returned the wrong value or no value, checked after the timed
    /// region.
    wrong_present: u64,
    missing_present: u64,
    /// The non-vacuity control: how many absent control keys did not return the table default.
    leaked_absent: u64,
    #[cfg(feature = "read-stats")]
    stages: ReadStages,
    /// The realised shard and lane geometry, so a reader can see that the two arms share a shard
    /// distribution.
    distinct_shards: usize,
    peak_shard: usize,
    distinct_lanes: usize,
    max_lane: u64,
    max_word: u64,
    in_extent: usize,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn replacement(new: i64, _old: i64) -> i64 {
    new
}

fn value_of(key: Key) -> i64 {
    key.tag.cast_signed()
}

/// Builds the arm's table and returns it with the admitted count. Not timed: the port's read is
/// the object, and an insert path that rebuilds the resident extent must not sit inside a read
/// measurement.
fn build(keys: &[Key], home: IdentityHome) -> (Table<Key, i64, IdentityHome, N>, usize) {
    let mut table = Table::new(DEFAULT_SENTINEL, home);
    let mut admitted = 0usize;
    for key in keys {
        if table
            .insert_with(&replacement, *key, value_of(*key))
            .is_ok()
        {
            admitted += 1;
        }
    }
    (table, admitted)
}

/// The present-stream read, `black_box` on the key and on the value.
fn read_pass(table: &Table<Key, i64, IdentityHome, N>, keys: &[Key], passes: usize) -> u64 {
    let mut acc = 0i64;
    for _ in 0..passes {
        for key in keys {
            acc = acc.wrapping_add(*table.get(black_box(key)));
        }
    }
    black_box(acc);
    u64::try_from(acc).unwrap_or(0)
}

/// The peak multiplicity of one key of `values` under `extract`, and the number of distinct
/// values it realises.
fn peak_and_distinct(values: &[u64], extract: impl Fn(u64) -> u64) -> (usize, usize) {
    let mut counts: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();
    for value in values {
        *counts.entry(extract(*value)).or_insert(0) += 1;
    }
    (counts.len(), counts.values().copied().max().unwrap_or(0))
}

/// The geometry and correctness facts of one arm, collected before any timed run so the two arms
/// are built and checked in the same state.
struct ArmSetup {
    name: &'static str,
    keys: Vec<Key>,
    table: Table<Key, i64, IdentityHome, N>,
    admitted: usize,
    distinct_shards: usize,
    peak_shard: usize,
    distinct_lanes: usize,
    max_lane: u64,
    max_word: u64,
    in_extent: usize,
}

impl ArmSetup {
    /// Builds the arm's table and records its geometry. Not timed: the port's read is the object,
    /// and an insert path that rebuilds the resident extent must not sit inside a read measurement.
    fn new(name: &'static str, home: IdentityHome, keys: Vec<Key>) -> Self {
        let (table, admitted) = build(&keys, home);
        let words: Vec<u64> = keys.iter().map(|k| k.tag).collect();
        let (distinct_shards, peak_shard) = peak_and_distinct(&words, |w| w & 0xff);
        let (distinct_lanes, _) = peak_and_distinct(&words, |w| w >> 8);
        let max_lane = words.iter().map(|w| w >> 8).max().unwrap_or(0);
        let max_word = words.iter().copied().max().unwrap_or(0);
        // How many keys the shipped one-lane extent can answer. `sweep_steps_per_key` is the
        // measured statement of the same fact; this is the arithmetic one.
        let in_extent = words.iter().filter(|w| **w < LANE_WORDS).count();
        Self {
            name,
            keys,
            table,
            admitted,
            distinct_shards,
            peak_shard,
            distinct_lanes,
            max_lane,
            max_word,
            in_extent,
        }
    }
}

/// One timed pass over an arm's present stream, in nanoseconds per key.
fn timed_pass(table: &Table<Key, i64, IdentityHome, N>, keys: &[Key]) -> f64 {
    let reads = PASSES_PER_RUN as u64 * keys.len() as u64;
    let start = Instant::now();
    let checksum = read_pass(table, keys, PASSES_PER_RUN);
    let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    black_box(checksum);
    elapsed as f64 / reads as f64
}

/// Interleaved timing: one run of arm A, then one of arm B, alternating which goes first.
///
/// The two arms are timed in the same process, back to back, run by run, so a preemption or a
/// frequency excursion on a shared box lands on both arms instead of on whichever arm was running
/// at the time. Timing arm A over all `RUNS` and then arm B over all `RUNS` measured a
/// load *drift* between the arms, not a load *offset*, and reported a ratio that moved with the
/// box's load rather than with the code.
fn timed_runs(arms: &mut [&mut ArmSetup; 2]) -> [Vec<f64>; 2] {
    for _ in 0..WARMUP_RUNS {
        for arm in arms.iter() {
            read_pass(&arm.table, &arm.keys, PASSES_PER_RUN);
        }
    }
    let mut samples = [
        Vec::with_capacity(RUNS as usize),
        Vec::with_capacity(RUNS as usize),
    ];
    for run in 0..RUNS {
        // Alternate the order so neither arm is systematically measured on a cold branch
        // predictor or a cold cache right after the other arm's run.
        let order = if run % 2 == 0 { [0, 1] } else { [1, 0] };
        for slot in order {
            let arm = &mut *arms[slot];
            samples[slot].push(timed_pass(&arm.table, &arm.keys));
        }
    }
    samples
}

/// Collects the reported facts of one arm after its timed runs: the attributed pass and the
/// correctness checks, both outside the timed region.
fn finish(arm: &ArmSetup, ns_per_key: Vec<f64>) -> Arm {
    let table = &arm.table;
    let keys = &arm.keys;

    // Attribution: one pass with the counters cleared. Outside the timed region, and only present
    // under the feature, so the timed build has no counter on its read path.
    #[cfg(feature = "read-stats")]
    let stages = {
        apeiron::table::read_stats::clear();
        let checksum = read_pass(table, keys, 1);
        black_box(checksum);
        let s = apeiron::table::read_stats::snapshot();
        ReadStages {
            summary_tests: s.summary_tests,
            summary_hits: s.summary_hits,
            summary_misses: s.summary_misses,
            sweep_steps: s.sweep_steps,
            tag_matches: s.tag_matches,
            full_compares: s.full_compares,
        }
    };

    // Correctness, after the timed region: every present key returns its own value, and the
    // sentinel default is unambiguous for the population.
    assert!(
        !keys.iter().any(|k| value_of(*k) == DEFAULT_SENTINEL),
        "spill_read_probe: arm {} carries a key whose value is the sentinel default",
        arm.name
    );
    let mut wrong_present = 0u64;
    let mut missing_present = 0u64;
    for key in keys {
        let got = *table.get(black_box(key));
        if got == DEFAULT_SENTINEL {
            missing_present += 1;
        } else if got != value_of(*key) {
            wrong_present += 1;
        }
    }
    // The non-vacuity control: an absent word of the same shape returns the default. If the present
    // loop were reading a constant or a stale slot, this would find it.
    let mut leaked_absent = 0u64;
    for key in absent_keys(keys) {
        if *table.get(black_box(&key)) != DEFAULT_SENTINEL {
            leaked_absent += 1;
        }
    }

    Arm {
        name: arm.name,
        admitted: arm.admitted,
        len: table.len(),
        runs: RUNS,
        ns_per_key,
        wrong_present,
        missing_present,
        leaked_absent,
        #[cfg(feature = "read-stats")]
        stages,
        distinct_shards: arm.distinct_shards,
        peak_shard: arm.peak_shard,
        distinct_lanes: arm.distinct_lanes,
        max_lane: arm.max_lane,
        max_word: arm.max_word,
        in_extent: arm.in_extent,
    }
}

// ── The report ───────────────────────────────────────────────────────────────────────────────────

/// True when `ns_per_key` lands in the reference band. Printed per arm; not the acceptance.
fn in_parity_band(ns_per_key: f64) -> bool {
    (PARITY_BAND_NS_PER_KEY.0..=PARITY_BAND_NS_PER_KEY.1).contains(&ns_per_key)
}

/// The graded acceptance: true when the matched spill/resident ratio is at most
/// `PARITY_RATIO_MAX`. Only the un-instrumented build carries two comparable terms; under
/// `read-stats` the ratio is reported `not-graded` and not enforced.
fn in_parity_ratio(ratio: f64) -> bool {
    ratio <= PARITY_RATIO_MAX
}

/// A per-arm reference label: `pass`/`FAIL` against the band, `not-graded` under `read-stats`.
fn parity_label(ns_per_key: f64) -> &'static str {
    if cfg!(feature = "read-stats") {
        "not-graded"
    } else if in_parity_band(ns_per_key) {
        "pass"
    } else {
        "FAIL"
    }
}

/// The graded verdict label for the matched ratio: `pass`, `FAIL`, or `not-graded`.
fn ratio_label(ratio: f64) -> &'static str {
    if cfg!(feature = "read-stats") {
        "not-graded"
    } else if in_parity_ratio(ratio) {
        "pass"
    } else {
        "FAIL"
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the per-key quotients are a diagnostic display of integer nanosecond medians"
)]
fn main() {
    println!(
        "# spill_read_probe: present-stream read isolation, canonical spill against resident \
         (gate of record for the E6.W2 spill-read port)"
    );
    println!(
        "# build: apeiron={} keys={KEYS} group_width={N} runs={RUNS} passes_per_run=\
         {PASSES_PER_RUN} warmup_runs={WARMUP_RUNS} read_stats={} mode={}",
        env!("CARGO_PKG_VERSION"),
        cfg!(feature = "read-stats"),
        if cfg!(feature = "read-stats") {
            "instrumented-TIMES-ARE-INFLATED"
        } else {
            "timing"
        }
    );
    println!(
        "# acceptance: matched ratio spill/resident <= {:.2}; band_ns_per_key=({:.2},{:.2}) is \
         the per-arm reference scale, not the graded criterion",
        PARITY_RATIO_MAX, PARITY_BAND_NS_PER_KEY.0, PARITY_BAND_NS_PER_KEY.1
    );

    let spill = spill_keys();
    let resident = resident_keys(&spill);

    let mut spill_arm = ArmSetup::new("canonical_spill", IdentityHome::SPILL, spill);
    let mut resident_arm = ArmSetup::new("canonical_resident", IdentityHome::RESIDENT, resident);
    let samples = timed_runs(&mut [&mut spill_arm, &mut resident_arm]);
    let spill_arm = finish(&spill_arm, samples[0].clone());
    let resident_arm = finish(&resident_arm, samples[1].clone());
    print_arm(&spill_arm);
    print_arm(&resident_arm);

    let spill_median = median_arm(&spill_arm);
    let resident_median = median_arm(&resident_arm);
    let ratio = spill_median / resident_median;
    let parity = in_parity_ratio(ratio);
    println!(
        "ratio={ratio:.4} arms=canonical_spill/canonical_resident median_ns_per_key={spill_median:.4}/{resident_median:.4}"
    );

    let clean = spill_arm.wrong_present == 0
        && spill_arm.missing_present == 0
        && spill_arm.leaked_absent == 0
        && resident_arm.wrong_present == 0
        && resident_arm.missing_present == 0
        && resident_arm.leaked_absent == 0
        && spill_arm.admitted == KEYS
        && resident_arm.admitted == KEYS;
    println!(
        "verdict parity={} ratio={:.4} spill_median={:.4} resident_median={:.4} correctness={}",
        ratio_label(ratio),
        ratio,
        spill_median,
        resident_median,
        if clean { "pass" } else { "FAIL" }
    );
    println!(
        "gate command: cargo run --release --example spill_read_probe (timing), \
         cargo run --release --features read-stats --example spill_read_probe (counters)"
    );

    assert!(
        clean,
        "spill_read_probe correctness: wrong_present={} missing_present={} leaked_absent={} \
         admitted={}/{}",
        spill_arm.wrong_present + resident_arm.wrong_present,
        spill_arm.missing_present + resident_arm.missing_present,
        spill_arm.leaked_absent + resident_arm.leaked_absent,
        spill_arm.admitted,
        resident_arm.admitted
    );
    // Only the un-instrumented build carries two comparable timings. Under `read-stats` the
    // counters sit on the spill read path, so the ratio is reported as not graded and not
    // enforced.
    assert!(
        parity || cfg!(feature = "read-stats"),
        "spill_read_probe parity: spill/resident ratio {ratio:.4} exceeds {PARITY_RATIO_MAX:.2} \
         (spill median {spill_median:.4} ns/key against resident {resident_median:.4} ns/key)"
    );
}

fn median_arm(arm: &Arm) -> f64 {
    let mut values = arm.ns_per_key.clone();
    median(&mut values)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the per-key quotients are a diagnostic display of integer nanosecond medians"
)]
fn print_arm(arm: &Arm) {
    let mut values = arm.ns_per_key.clone();
    let med = median(&mut values);
    let min = values.first().copied().unwrap_or(f64::NAN);
    let max = values.last().copied().unwrap_or(f64::NAN);
    println!(
        "arm={} direct_word={} admitted={} len={} runs={} passes_per_run={} \
         median_ns_per_key={med:.4} min_ns_per_key={min:.4} max_ns_per_key={max:.4} \
         spread_max_over_median={:.2} band_ns_per_key=({:.2},{:.2}) parity={}",
        arm.name,
        if arm.name == "canonical_spill" {
            "None"
        } else {
            "Some"
        },
        arm.admitted,
        arm.len,
        arm.runs,
        PASSES_PER_RUN,
        max / med,
        PARITY_BAND_NS_PER_KEY.0,
        PARITY_BAND_NS_PER_KEY.1,
        parity_label(med)
    );
    println!(
        "arm={} geometry distinct_shards={} peak_shard={} distinct_lanes={} max_lane={} \
         max_word={} lane_words={} in_extent={}/{}",
        arm.name,
        arm.distinct_shards,
        arm.peak_shard,
        arm.distinct_lanes,
        arm.max_lane,
        arm.max_word,
        LANE_WORDS,
        arm.in_extent,
        KEYS
    );
    #[cfg(feature = "read-stats")]
    {
        let per = |v: u64| v as f64 / KEYS as f64;
        println!(
            "arm={} counters sweep_steps={} sweep_steps_per_key={:.4} tag_matches={} \
             tag_matches_per_key={:.4} full_compares={} full_compares_per_key={:.4} \
             summary_tests={} summary_hits={} summary_misses={}",
            arm.name,
            arm.stages.sweep_steps,
            per(arm.stages.sweep_steps),
            arm.stages.tag_matches,
            per(arm.stages.tag_matches),
            arm.stages.full_compares,
            per(arm.stages.full_compares),
            arm.stages.summary_tests,
            arm.stages.summary_hits,
            arm.stages.summary_misses
        );
    }
    println!(
        "arm={} wrong_present={} missing_present={} absent_control_reads={} leaked_absent={}",
        arm.name, arm.wrong_present, arm.missing_present, KEYS, arm.leaked_absent
    );
}

#[cfg(test)]
mod tests {
    use super::{absent_keys, resident_keys, spill_keys, IdentityHome, KEYS, LANE_WORDS};
    use apeiron::coord::locator::ExactIdentity;
    use apeiron::table::Home;
    use apeiron::Sigma;

    fn distinct(tags: &[u64]) -> usize {
        let mut sorted = tags.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len()
    }

    /// Both arms are 192 distinct keys, so a per-key loop over either population cannot read one
    /// key twice or a stale slot.
    #[test]
    fn both_populations_are_192_distinct_keys() {
        let spill = spill_keys();
        for keys in [&spill[..], &resident_keys(&spill)[..]] {
            assert_eq!(keys.len(), KEYS);
            let tags: Vec<u64> = keys.iter().map(|k| k.tag).collect();
            assert_eq!(distinct(&tags), KEYS);
        }
    }

    /// The two arms share a shard distribution by construction, and the spill arm's words are all
    /// outside the certified direct range.
    #[test]
    fn the_arms_share_shards_and_the_spill_arm_has_no_direct_word() {
        let spill = spill_keys();
        let resident = resident_keys(&spill);
        for (a, b) in spill.iter().zip(&resident) {
            assert_eq!(a.tag & 0xff, b.tag & 0xff);
            assert_eq!(
                IdentityHome::SPILL.address(Sigma::Elliptic, a).direct_word,
                None,
                "a spill-arm key certified a direct word"
            );
            assert!(
                IdentityHome::RESIDENT
                    .address(Sigma::Elliptic, b)
                    .direct_word
                    .is_some(),
                "a resident-arm key failed to certify a direct word"
            );
        }
    }

    /// Every resident word is inside the shipped one-lane extent, so the resident arm cannot
    /// silently route a key into `spill_read`. Without this the arm times the spill path and
    /// reports a wrong ratio; `sweep_steps_per_key = 0` on the attributed run is the measured
    /// form of the same statement.
    #[test]
    fn every_resident_word_is_in_extent() {
        for key in resident_keys(&spill_keys()) {
            assert!(key.tag < LANE_WORDS);
            assert!(key.tag < u64::from(u32::MAX));
            let word = IdentityHome::RESIDENT
                .address(Sigma::Elliptic, &key)
                .direct_word
                .expect("the resident arm certifies every key's direct word");
            assert_eq!(u64::from(word), key.tag);
        }
    }

    /// The absent control family is disjoint from the present family of *both* arms. Without this
    /// the correctness control reads a present key and reports a false leak, which is how a
    /// `place | 1 << 40` control fails on a population whose spread words already carry bit 40.
    #[test]
    fn the_absent_control_family_is_disjoint_from_both_present_families() {
        let spill = spill_keys();
        let resident = resident_keys(&spill);
        for present in [&spill[..], &resident[..]] {
            let present_tags: Vec<u64> = present.iter().map(|k| k.tag).collect();
            let absent_tags: Vec<u64> = absent_keys(present).iter().map(|k| k.tag).collect();
            assert_eq!(distinct(&absent_tags), KEYS);
            let overlap = absent_tags
                .iter()
                .filter(|t| present_tags.contains(t))
                .count();
            assert_eq!(overlap, 0, "an absent control word is a present key");
            // An absent word inside the one-lane extent would be answerable from the flat array.
            assert!(absent_tags.iter().all(|t| *t >= LANE_WORDS));
        }
    }

    /// The key's identity encoding is the tag's big-endian bytes, so distinct tags have distinct
    /// encodings and a present read that returns a value did so by matching the exact identity.
    #[test]
    fn key_identity_is_the_exact_tag() {
        let spill = spill_keys();
        let resident = resident_keys(&spill);
        for key in spill.iter().take(4).chain(&resident) {
            let bytes: Vec<u8> = (0..key.encoding_len())
                .map(|index| key.encoding_byte(index))
                .collect();
            assert_eq!(bytes, key.tag.to_be_bytes().to_vec());
            assert!(key.matches_retained(&key.retain()));
        }
    }

    /// The graded acceptance must reject the measured pre-port ratios and a deliberately slowed
    /// spill arm, and accept the port target. This is the logic-level form of the reviewer's finding: an
    /// acceptance that graded only the resident median could never reject a slow spill read,
    /// because the resident arm is the comparator and does not change.
    #[test]
    fn the_acceptance_rejects_a_slow_spill_read() {
        // Measured pre-port ratios on the host named in the header: 5.75..6.60.
        assert!(!super::in_parity_ratio(6.56));
        assert!(!super::in_parity_ratio(5.75));
        // A spill arm deliberately slowed past the bound fails by construction.
        assert!(!super::in_parity_ratio(2.2001));
        // The port target: within about 2x of the resident read, ideally parity.
        assert!(super::in_parity_ratio(2.2));
        assert!(super::in_parity_ratio(1.05));
    }
}
