//! REG-1 shared fixed-16 performance measurement: the current `apeiron::Table`, a benchmark-only
//! candidate representation, and the pinned historical `hashtrinity::Table`, in one release
//! binary, over one deterministic operation schedule.
//!
//! The admitted workload is 4096 keys placed exactly 16 per Apeiron shard (256 shards), proven
//! before timing: every key is admitted, each of the 256 shards refuses a 17th key with
//! `ShardFull`, and Apeiron, the candidate, hashtrinity and an independent `HashMap` oracle agree
//! on the initial and post-trace canonical readback. `Coord` is the injective key tag, so
//! hashtrinity preserves identity for the same keys.
//!
//! **Benchmark boundary change.** `SharedHome::home` now returns the full `tag` rather than
//! `tag % 256`, so the candidate can cache the full home word. The production `apeiron::Table`
//! reduces `% 256` internally, so shard placement is identical to the historical residue; the
//! `home_boundary` witness records both the full word and the unchanged residue for the admitted
//! set, and the precheck requires the Apeiron, candidate, hashtrinity and oracle digests to agree
//! before any sample is accepted.
//!
//! Two merge semantics are measured separately (`replacement`, `wrapping-add`), each with its own
//! warmup traces and samples; no two different merge semantics are compared.
//!
//! The REG-1 performance `pass`/`boundary` verdict remains the current-Apeiron/hashtrinity ratio.
//! The candidate/hashtrinity and candidate/current ratios are reported as experimental evidence
//! only: the candidate is not the production table and is not proof-covered.
//!
//! **Strict acceptance.** `run_performance(profile, controls, accept)` with `accept` (CLI
//! `--accept`) is the acceptance command: the performance axis is `pass` only when every applicable
//! strata cell passes and no applicable cell's campaign-ratio interval crosses the owner threshold
//! `selected/incumbent <= 1.0`; otherwise the axis is `fail` and the process exits nonzero. A
//! `pass` cell whose interval crosses the threshold is inconclusive and fails. Not-applicable cells
//! (no incumbent counterpart, e.g. the resource cells) do not fail it. Without `accept` the same run
//! is the diagnostic mode and reports the axis `boundary`.
//!
//! The reusable measurement engine (workload, builders, timed schedules, precheck, sample campaign
//! and statistics) lives in `examples/reg1/matrix.rs`; this module owns the report shell, the
//! legacy fixed-16 witness, the candidate arm and the validator controls. The separated
//! occupancy/distribution/operation enumeration lives in `examples/reg1/strata.rs` and is called
//! from this same gate; it appends evidence to the performance axis and, in strict mode, determines
//! the axis verdict.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the REG-1 performance check and its witnesses"
)]

use std::process::Command;

use crate::matrix::candidate::{self, CollisionWitness};
use crate::matrix::{
    build_hashtrinity, build_trace, initial_keys, load_registered_cells, maximum, median, minimum,
    ns_per_op, per_shard_max, precheck, ratio, replacement_merge, run_home_boundary, sample_mode,
    validate_registration, ApeironTable, Candidate, Counts, HomeBoundary, MergeMode, ModeSamples,
    RegisteredCell, APEIRON_N, CONTROL_REGISTERED_CELL, FULL_LOOKUPS, FULL_SAMPLES, FULL_UPDATES,
    FULL_WARMUP, INITIAL_KEYS, SHARD_COUNT,
};
// Re-exported so the aggregate binary keeps its `performance::Profile` CLI surface.
pub(crate) use crate::matrix::Profile;
use crate::strata::{self, CellSamples, CellStatus, Distribution, Operation, StrataWitness};

// The spatial/parallel incumbent adapters live in `examples/reg1/adapters/spatial.rs`; declared
// here so this task owns the module without touching `main.rs`.
#[path = "adapters/spatial.rs"]
pub(crate) mod spatial;
use crate::workload::{run_identity_witness, IdentityWitness, KEY_COUNT};
use crate::{print_identity, print_report, AxisRecord, AxisStatus, Report};

/// The owner acceptance threshold for the fixed-16 family: match or beat the applicable incumbent
/// on every applicable cell, `selected/incumbent <= 1.0`. No envelope tolerance.
const RATIO_LIMIT: f64 = 1.0;

/// The complete performance witness: workload identity, the boundary and collision controls, and
/// per-mode samples.
struct PerfWitness {
    identity: IdentityWitness,
    collision: CollisionWitness,
    home_boundary: HomeBoundary,
    profile: Profile,
    counts: Counts,
    initial_keys: usize,
    per_shard_max: usize,
    modes: Vec<ModeSamples>,
}

/// Run the correctness precheck and the full sample campaign for every mode.
///
/// # Errors
/// Returns a description of the first failed precheck or sample.
fn run_witness(profile: Profile) -> Result<PerfWitness, String> {
    let counts = profile.counts();
    let keys = initial_keys();
    let trace = build_trace(counts);
    let identity = run_identity_witness();
    let collision = candidate::run_collision_witness();
    let home_boundary = run_home_boundary(&keys);
    let mut modes = Vec::with_capacity(MergeMode::ALL.len());
    for mode in MergeMode::ALL {
        let pre = precheck(mode, &trace, &keys)?;
        let mut samples = sample_mode(mode, &trace, &keys, counts, pre.reference_digest)?;
        samples.admitted = pre.admitted;
        samples.candidate_admitted = pre.candidate_admitted;
        samples.ht_len = pre.ht_len;
        samples.oracle_len = pre.oracle_len;
        samples.shard_probe_refusals = pre.shard_probe_refusals;
        samples.candidate_shard_probe_refusals = pre.candidate_shard_probe_refusals;
        modes.push(samples);
    }
    Ok(PerfWitness {
        identity,
        collision,
        home_boundary,
        profile,
        counts,
        initial_keys: keys.len(),
        per_shard_max: per_shard_max(&keys),
        modes,
    })
}

/// The named performance boundary for a ratio above the limit.
fn boundary_name(ratio_value: f64) -> String {
    format!("REG1-PERF-SHARED-FIXED16={ratio_value:.3}")
}

/// The worst per-mode value, infinite if any value is non-finite.
fn worst_over_modes(w: &PerfWitness, value_of: impl Fn(&ModeSamples) -> f64) -> f64 {
    let mut worst = 0.0f64;
    for mode in &w.modes {
        let value = value_of(mode);
        if !value.is_finite() {
            return f64::INFINITY;
        }
        if value > worst {
            worst = value;
        }
    }
    worst
}

/// The worst per-mode Apeiron/hashtrinity ratio, the REG-1 performance axis.
fn worst_ratio(w: &PerfWitness) -> f64 {
    worst_over_modes(w, |mode| ratio(mode.apeiron_median(), mode.ht_median()))
}

/// The worst per-mode candidate/hashtrinity ratio. Experimental evidence only.
fn worst_candidate_ht_ratio(w: &PerfWitness) -> f64 {
    worst_over_modes(w, |mode| ratio(mode.candidate_median(), mode.ht_median()))
}

/// The worst per-mode candidate/current-Apeiron ratio. Experimental evidence only.
fn worst_candidate_current_ratio(w: &PerfWitness) -> f64 {
    worst_over_modes(w, |mode| {
        ratio(mode.candidate_median(), mode.apeiron_median())
    })
}

// ── Strict acceptance over the current-strata cells ───────────────────────────────────────────
//
// `--accept` is the acceptance command. The diagnostic mode reports a `boundary` axis and exits 0;
// strict mode exits nonzero unless every applicable strata speed cell passes with its whole
// campaign-ratio interval at or below `selected/incumbent <= 1.0`. A `pass` cell whose interval
// crosses the threshold is inconclusive, not a pass. Not-applicable cells (no incumbent
// counterpart) and the separate resource cells are excluded from the applicable population.

/// The strict acceptance outcome: the applicable cell count, the failing cell descriptions and the
/// worst failing cell with its ratio.
pub(crate) struct StrictAcceptance {
    pub(crate) applicable: usize,
    pub(crate) failing: Vec<String>,
    pub(crate) worst: Option<(String, f64)>,
}

impl StrictAcceptance {
    /// Whether every applicable cell passed.
    pub(crate) fn passed(&self) -> bool {
        self.failing.is_empty()
    }
}

/// The report name of a cell status, mirroring `strata::CellStatus::name`.
fn cell_status_name(status: CellStatus) -> &'static str {
    match status {
        CellStatus::Pass => "pass",
        CellStatus::Fail => "fail",
        CellStatus::NotApplicable => "not-applicable",
        CellStatus::Missing => "missing",
        CellStatus::Rejected => "rejected",
    }
}

/// The minimum and maximum of a cell's campaign-ratio set. Empty is `(inf, -inf)`, which fails the
/// finite check below.
fn ratio_interval(ratios: &[f64]) -> (f64, f64) {
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
    (lo, hi)
}

/// The pooled Apeiron/hashtrinity ratio of a measured cell, or `None` when it has no samples.
fn cell_ratio(cell: &CellSamples) -> Option<f64> {
    if cell.apeiron_samples.is_empty() || cell.ht_samples.is_empty() {
        return None;
    }
    let r = ratio(median(&cell.apeiron_samples), median(&cell.ht_samples));
    r.is_finite().then_some(r)
}

/// Track the largest-ratio failing cell seen so far.
fn update_worst(worst: &mut Option<(String, f64)>, id: &str, r: f64) {
    if worst.as_ref().is_none_or(|(_, w)| r > *w) {
        *worst = Some((id.to_string(), r));
    }
}

/// Evaluate strict acceptance. Every applicable speed cell must be `pass` and its campaign-ratio
/// interval must lie at or below `RATIO_LIMIT`; `fail`/`missing`/`rejected` cells and `pass` cells
/// whose interval crosses the threshold are failures. `not-applicable` cells are excluded.
fn strict_acceptance(strata: &StrataWitness) -> StrictAcceptance {
    let mut applicable = 0usize;
    let mut failing = Vec::new();
    let mut worst: Option<(String, f64)> = None;
    for cell in &strata.cells {
        match cell.status {
            CellStatus::NotApplicable => {}
            CellStatus::Missing | CellStatus::Rejected | CellStatus::Fail => {
                applicable += 1;
                if let Some(r) = cell_ratio(cell) {
                    update_worst(&mut worst, &cell.cell_id, r);
                    failing.push(format!(
                        "{} status={} ratio={r:.4} reason=\"{}\"",
                        cell.cell_id,
                        cell_status_name(cell.status),
                        cell.reason
                    ));
                } else {
                    failing.push(format!(
                        "{} status={} reason=\"{}\"",
                        cell.cell_id,
                        cell_status_name(cell.status),
                        cell.reason
                    ));
                }
            }
            CellStatus::Pass => {
                applicable += 1;
                let (r_min, r_max) = ratio_interval(&cell.campaign_ratios);
                let finite = !cell.campaign_ratios.is_empty()
                    && cell.campaign_ratios.iter().all(|r| r.is_finite());
                if !finite || r_max > RATIO_LIMIT {
                    update_worst(&mut worst, &cell.cell_id, r_max);
                    failing.push(format!(
                        "{} status=pass but ratio_interval=[{r_min:.4},{r_max:.4}] is \
                         inconclusive above {RATIO_LIMIT:.2}",
                        cell.cell_id
                    ));
                }
            }
        }
    }
    StrictAcceptance {
        applicable,
        failing,
        worst,
    }
}

/// The performance axis status. Strict mode is `pass` only when the strict outcome passes and
/// `fail` otherwise; diagnostic mode is `pass`/`boundary` from the fixed-16 mode ratio.
fn performance_axis_status(
    accept: bool,
    worst: f64,
    strict: Option<&StrictAcceptance>,
) -> AxisStatus {
    if accept {
        match strict {
            Some(outcome) if outcome.failing.is_empty() => AxisStatus::Pass,
            _ => AxisStatus::Fail,
        }
    } else if worst <= RATIO_LIMIT {
        AxisStatus::Pass
    } else {
        AxisStatus::Boundary
    }
}

/// The strict-acceptance evidence clause for the performance axis.
fn strict_evidence_clause(strict: Option<&StrictAcceptance>) -> String {
    match strict {
        Some(outcome) if outcome.failing.is_empty() => format!(
            "; strict_acceptance=pass applicable={} failing=0",
            outcome.applicable
        ),
        Some(outcome) => {
            let worst_cell = outcome.worst.as_ref().map_or_else(
                || "NONE".to_string(),
                |(id, r)| format!("{id} ratio={r:.4}"),
            );
            format!(
                "; strict_acceptance=fail applicable={} failing={} worst_cell={worst_cell}",
                outcome.applicable,
                outcome.failing.len()
            )
        }
        None => "; strict_acceptance=fail applicable=0 failing=0 strata=MISSING".to_string(),
    }
}

/// A minimal, syntactically valid cell for the strict-acceptance control.
fn synthetic_cell(id: &str, status: CellStatus, campaign_ratios: &[f64]) -> CellSamples {
    CellSamples {
        cell_id: id.to_string(),
        occupancy: 1,
        distribution: Distribution::DirectIndexResident,
        operation: Operation::PresentHit,
        merge: MergeMode::Replacement,
        population: 256,
        ops: 4096,
        hit_ratio: 1.0,
        oracle_digest: 0,
        apeiron_digest_ok: true,
        ht_digest_ok: true,
        apeiron_admitted: 256,
        ht_len: 256,
        oracle_len: 256,
        ht_capacity: 512,
        ht_longest_probe: 1,
        apeiron_samples: vec![100; 15],
        ht_samples: vec![100; 15],
        campaign_ratios: campaign_ratios.to_vec(),
        selected_impl_symbol: crate::strata::VENDORED_MAP_SYMBOL,
        status,
        reason: "synthetic strict-acceptance control cell".to_string(),
    }
}

/// A one-cell witness wrapping a synthetic cell, so the strict decision runs on the same type as
/// the real enumeration.
fn synthetic_witness(cell: CellSamples) -> StrataWitness {
    let (passed, failed) = match cell.status {
        CellStatus::Pass => (1, 0),
        CellStatus::Fail => (0, 1),
        _ => (0, 0),
    };
    StrataWitness {
        cells: vec![cell],
        resources: Vec::new(),
        profile: "control".to_string(),
        campaigns: 3,
        warmup: 2,
        samples: 15,
        ops: 4096,
        measured: 1,
        missing: 0,
        rejected: 0,
        passed,
        failed,
        not_applicable: 0,
        ht_rev: String::new(),
        rustc: String::new(),
        cpu: String::new(),
    }
}

/// The strict-acceptance control. It runs the same strict decision the `--accept` exit path uses:
/// one failing applicable cell must be rejected, a `pass` cell whose interval crosses the
/// threshold must be rejected as inconclusive, and the same cell at or below the threshold must be
/// accepted. It exits 2 with a `rejected` marker on success, matching the existing control
/// convention; any wrong verdict is a control failure.
fn run_strict_acceptance_control(name: &str) -> i32 {
    let failing = synthetic_witness(synthetic_cell(
        "synthetic-failing-applicable-cell",
        CellStatus::Fail,
        &[2.0, 2.0, 2.0],
    ));
    let passing = synthetic_witness(synthetic_cell(
        "synthetic-passing-applicable-cell",
        CellStatus::Pass,
        &[0.5, 0.5, 0.5],
    ));
    let crossing = synthetic_witness(synthetic_cell(
        "synthetic-crossing-applicable-cell",
        CellStatus::Pass,
        &[0.8, 1.3, 1.1],
    ));
    let failing_outcome = strict_acceptance(&failing);
    let passing_outcome = strict_acceptance(&passing);
    let crossing_outcome = strict_acceptance(&crossing);
    if failing_outcome.passed()
        || failing_outcome.failing.len() != 1
        || failing_outcome.applicable != 1
    {
        eprintln!(
            "performance-control {name} NOT rejected: failing cell accepted or mis-counted \
             (applicable={} failing={})",
            failing_outcome.applicable,
            failing_outcome.failing.len()
        );
        return 1;
    }
    if !passing_outcome.passed() {
        eprintln!(
            "performance-control {name} NOT rejected: at-or-below-threshold cell was rejected: {}",
            passing_outcome.failing.join("; ")
        );
        return 1;
    }
    if crossing_outcome.passed() || crossing_outcome.failing.len() != 1 {
        eprintln!(
            "performance-control {name} NOT rejected: crossing-interval cell was accepted \
             (failing={})",
            crossing_outcome.failing.len()
        );
        return 1;
    }
    eprintln!(
        "performance-control {name} rejected: strict acceptance rejected the failing applicable \
         cell and the crossing-interval cell and accepted the at-or-below-threshold cell"
    );
    2
}

/// Build the performance report. In diagnostic mode performance is `pass` only when every measured
/// mode is at or below the owner threshold `selected/incumbent <= 1.0`; otherwise it is `boundary`
/// named `REG1-PERF-SHARED-FIXED16=<ratio>`. In strict acceptance mode (`accept`) the axis is
/// `pass` only when every applicable strata cell passes and no applicable cell's ratio interval
/// crosses the threshold; otherwise it is `fail`. The candidate ratios are experimental evidence.
/// The identity, round-trip, growth/merge and coverage rows are not repaired here. When `strata` is
/// supplied, the separated occupancy/distribution/operation enumeration is appended to the
/// performance evidence.
fn build_performance_report(
    w: &PerfWitness,
    strata: Option<&StrataWitness>,
    accept: bool,
) -> Report {
    let worst = worst_ratio(w);
    let strict = strata.filter(|_| accept).map(strict_acceptance);
    let status = performance_axis_status(accept, worst, strict.as_ref());
    let modes: Vec<String> = w
        .modes
        .iter()
        .map(|mode| {
            format!(
                "{} ratio={:.4} candidate_ht_ratio={:.4} candidate_current_ratio={:.4}",
                mode.mode.name(),
                ratio(mode.apeiron_median(), mode.ht_median()),
                ratio(mode.candidate_median(), mode.ht_median()),
                ratio(mode.candidate_median(), mode.apeiron_median())
            )
        })
        .collect();
    let boundary = if status == AxisStatus::Boundary {
        format!("; {}", boundary_name(worst))
    } else {
        String::new()
    };
    let candidate_ht = worst_candidate_ht_ratio(w);
    let candidate_current = worst_candidate_current_ratio(w);
    let strata_clause =
        strata.map_or_else(String::new, |s| format!("; {}", strata::report_clause(s)));
    let strict_clause = if accept {
        strict_evidence_clause(strict.as_ref())
    } else {
        String::new()
    };
    let performance_evidence = format!(
        "shared fixed-16 workload: initial_keys={} ({} per shard x {} shards), lookups={}, \
         updates={}, warmup={}, samples={}; {}; ratio_limit={:.2}; profile={}; \
         candidate/hashtrinity={candidate_ht:.4} candidate/current={candidate_current:.4} \
         (experimental, not the REG-1 axis){boundary}{strata_clause}{strict_clause}",
        w.initial_keys,
        APEIRON_N,
        SHARD_COUNT,
        w.counts.lookups,
        w.counts.updates,
        w.counts.warmup,
        w.counts.samples,
        modes.join(" "),
        RATIO_LIMIT,
        w.profile.name()
    );
    let id = &w.identity;
    let axes = vec![
        AxisRecord {
            axis: "identity",
            status: AxisStatus::Pass,
            evidence: format!(
                "identity row is unchanged by this measurement: the {KEY_COUNT}-key \
                 constant-coordinate witness admits all {}/{} distinct keys with oracle agreement \
                 and hashtrinity coalesces {} submitted onto {} live; the full-scale 1e8 \
                 coordinate-colliding run is unrun",
                id.apeiron_admitted, KEY_COUNT, KEY_COUNT, id.ht_len
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
            evidence:
                "this measurement does not rerun the separate-process determinism check; see \
                       --check correctness"
                    .to_string(),
        },
        AxisRecord {
            axis: "growth_merge_refusal",
            status: AxisStatus::Pass,
            evidence: "region growth and the N=0 SizeOverflow refusal are observed in the \
                       correctness smoke; remove remains absent, so a performance measurement does \
                       not repair the remove row"
                .to_string(),
        },
        AxisRecord {
            axis: "performance",
            status,
            evidence: performance_evidence,
        },
        AxisRecord {
            axis: "coverage",
            status: AxisStatus::Missing,
            evidence: "coverage census and fractions are unrun".to_string(),
        },
    ];
    Report {
        axes,
        report_complete: false,
        reg1_pass: false,
    }
}

/// The one performance validator. It fails closed on the workload identity, the sample sets, the
/// derived ratio and every report status.
///
/// # Errors
/// Returns a description of the first observation, count, sample or status that violates the REG-1
/// performance contract. A caller that observes `Ok(())` has an accepted measurement, not REG-1
/// acceptance.
fn validate_performance(w: &PerfWitness, r: &Report) -> Result<(), String> {
    validate_workload(w)?;
    validate_report(w, r, None, false)
}

/// The full performance gate validator: the measurement contract plus the registered current-strata
/// matrix. Every registered cell must be produced by the current run with the same workload identity
/// and oracle digest, every run cell must be registered, and the selected implementation must be the
/// actual executing symbol.
///
/// # Errors
/// Returns the first measurement or registration violation. A caller that observes `Ok(())` has an
/// identity-checked registration; the registered medians are receipts, never the current values.
fn validate_all(
    w: &PerfWitness,
    r: &Report,
    strata: &StrataWitness,
    registered: &[RegisteredCell],
    accept: bool,
) -> Result<(), String> {
    validate_workload(w)?;
    validate_report(w, r, Some(strata), accept)?;
    validate_registration(strata, registered)
}

/// Validate the workload identity and every per-mode sample set.
fn validate_workload(w: &PerfWitness) -> Result<(), String> {
    let expected = usize::try_from(INITIAL_KEYS).expect("4096 fits usize");
    if w.initial_keys != expected {
        return Err(format!(
            "workload identity: initial_keys {} != {expected}",
            w.initial_keys
        ));
    }
    if w.per_shard_max != APEIRON_N {
        return Err(format!(
            "workload identity: per_shard_max {} != {APEIRON_N}",
            w.per_shard_max
        ));
    }
    if w.counts.lookups == 0 || w.counts.updates == 0 || w.counts.samples == 0 {
        return Err("workload schedule has a zero count".to_string());
    }
    if w.modes.len() != MergeMode::ALL.len() {
        return Err(format!(
            "mode count {} != {}",
            w.modes.len(),
            MergeMode::ALL.len()
        ));
    }
    if w.home_boundary.keys != expected
        || !w.home_boundary.full_word_ok
        || !w.home_boundary.residue_ok
    {
        return Err("benchmark boundary: SharedHome full-word/residue witness failed".to_string());
    }
    candidate::validate_collision_witness(&w.collision)?;
    let sample_count = usize::try_from(w.counts.samples).expect("sample count fits usize");
    for mode in &w.modes {
        validate_mode(mode, expected, sample_count)?;
    }

    Ok(())
}

/// Validate one merge mode's precheck record and sample sets.
fn validate_mode(mode: &ModeSamples, expected: usize, sample_count: usize) -> Result<(), String> {
    if mode.admitted != expected {
        return Err(format!(
            "mode {} admitted {} != {expected}",
            mode.mode.name(),
            mode.admitted
        ));
    }
    if mode.candidate_admitted != expected {
        return Err(format!(
            "mode {} candidate admitted {} != {expected}",
            mode.mode.name(),
            mode.candidate_admitted
        ));
    }
    if mode.ht_len != expected || mode.oracle_len != expected {
        return Err(format!(
            "mode {} lengths hashtrinity={} oracle={} != {expected}",
            mode.mode.name(),
            mode.ht_len,
            mode.oracle_len
        ));
    }
    if mode.shard_probe_refusals != 0 {
        return Err(format!(
            "mode {} growable shard probe refusals {} != 0",
            mode.mode.name(),
            mode.shard_probe_refusals
        ));
    }
    if mode.candidate_shard_probe_refusals != SHARD_COUNT {
        return Err(format!(
            "mode {} candidate shard probe refusals {} != {SHARD_COUNT}",
            mode.mode.name(),
            mode.candidate_shard_probe_refusals
        ));
    }
    if !mode.precheck_digest_ok {
        return Err(format!("mode {} precheck digest not ok", mode.mode.name()));
    }
    if mode.apeiron_samples.len() != sample_count
        || mode.candidate_samples.len() != sample_count
        || mode.ht_samples.len() != sample_count
    {
        return Err(format!(
            "mode {} sample counts {}/{}/{} != {sample_count}",
            mode.mode.name(),
            mode.apeiron_samples.len(),
            mode.candidate_samples.len(),
            mode.ht_samples.len()
        ));
    }
    if mode.apeiron_samples.contains(&0)
        || mode.candidate_samples.contains(&0)
        || mode.ht_samples.contains(&0)
    {
        return Err(format!(
            "mode {} has a zero-nanosecond sample",
            mode.mode.name()
        ));
    }
    if !mode.apeiron_digest_ok || !mode.candidate_digest_ok || !mode.ht_digest_ok {
        return Err(format!(
            "mode {} digest mismatch against the precheck reference",
            mode.mode.name()
        ));
    }
    let a_med = mode.apeiron_median();
    let c_med = mode.candidate_median();
    let h_med = mode.ht_median();
    if h_med == 0 || a_med == 0 || c_med == 0 {
        return Err(format!("mode {} has a zero median", mode.mode.name()));
    }
    if !(minimum(&mode.apeiron_samples) <= a_med && a_med <= maximum(&mode.apeiron_samples)) {
        return Err(format!(
            "mode {} apeiron min/median/max unordered",
            mode.mode.name()
        ));
    }
    if !(minimum(&mode.candidate_samples) <= c_med && c_med <= maximum(&mode.candidate_samples)) {
        return Err(format!(
            "mode {} candidate min/median/max unordered",
            mode.mode.name()
        ));
    }
    if !(minimum(&mode.ht_samples) <= h_med && h_med <= maximum(&mode.ht_samples)) {
        return Err(format!(
            "mode {} hashtrinity min/median/max unordered",
            mode.mode.name()
        ));
    }
    Ok(())
}

/// Validate the report schema, the ratio-derived status and the fail-closed flags. In strict mode
/// the performance axis must be `pass` exactly when every applicable strata cell passes; in
/// diagnostic mode it must be `pass`/`boundary` from the fixed-16 mode ratio.
fn validate_report(
    w: &PerfWitness,
    r: &Report,
    strata: Option<&StrataWitness>,
    accept: bool,
) -> Result<(), String> {
    let names: Vec<&str> = r.axes.iter().map(|a| a.axis).collect();
    let expected_names = [
        "identity",
        "round_trip",
        "determinism",
        "growth_merge_refusal",
        "performance",
        "coverage",
    ];
    if names != expected_names {
        return Err(format!("axis schema mismatch: {names:?}"));
    }
    if r.axes.iter().any(|a| !crate::allowed(a.status)) {
        return Err("axis status outside pass/fail/boundary/missing/tool_error".to_string());
    }
    if !matches!(r.axes[0].status, AxisStatus::Pass | AxisStatus::Boundary) {
        return Err("identity must be pass or boundary for the growable table".to_string());
    }
    if r.axes[1].status != AxisStatus::Boundary {
        return Err("round_trip must stay boundary: no codec surface".to_string());
    }
    if !matches!(r.axes[3].status, AxisStatus::Pass | AxisStatus::Boundary) {
        return Err("growth_merge_refusal must be pass or boundary".to_string());
    }
    if r.axes[5].status != AxisStatus::Missing {
        return Err("coverage must stay missing: no census was run".to_string());
    }
    let worst = worst_ratio(w);
    let expected_status = if accept {
        match strata.map(strict_acceptance) {
            Some(outcome) if outcome.failing.is_empty() => AxisStatus::Pass,
            _ => AxisStatus::Fail,
        }
    } else if worst <= RATIO_LIMIT {
        AxisStatus::Pass
    } else {
        AxisStatus::Boundary
    };
    if r.axes[4].status != expected_status {
        return Err(format!(
            "performance status {} does not match the {} verdict",
            r.axes[4].status.as_str(),
            if accept { "strict" } else { "diagnostic" }
        ));
    }
    if accept {
        let marker = if expected_status == AxisStatus::Pass {
            "strict_acceptance=pass"
        } else {
            "strict_acceptance=fail"
        };
        if !r.axes[4].evidence.contains(marker) {
            return Err(format!("performance strict evidence omits {marker}"));
        }
    } else if expected_status == AxisStatus::Boundary
        && !r.axes[4].evidence.contains(&boundary_name(worst))
    {
        return Err(format!(
            "performance boundary evidence omits {}",
            boundary_name(worst)
        ));
    }
    if r.report_complete {
        return Err("report_complete must be false".to_string());
    }
    if r.reg1_pass {
        return Err("reg1_pass must be false: REG-1/G1 stays OPEN".to_string());
    }
    Ok(())
}

/// The performance validator controls. Each corrupts the in-memory witness or the in-memory
/// parsed registration; no source is mutated.
///
/// `cached-home-only-collision` replaces the correct collision witness with the deliberately
/// broken word-only variant, whose cached-home acceptance coalesces two distinct keys. The three
/// registration controls corrupt the `current_strata` registration: a wrong oracle digest, a
/// deleted required current cell, and a selected implementation that is not the executing symbol.
/// `strict-acceptance-single-failing-cell` runs the strict decision on synthetic one-cell reports:
/// a failing applicable cell and a crossing-interval cell must be rejected, an at-or-below cell
/// accepted.
pub(crate) const CONTROLS: [&str; 11] = [
    "wrong-workload-identity",
    "invalid-samples",
    "absent-samples",
    "cached-home-only-collision",
    "wrong-oracle-digest",
    "deleted-required-current-cell",
    "selected-implementation-mismatch",
    "strict-acceptance-single-failing-cell",
    "spatial-wrong-packed-input",
    "spatial-wrong-bucket-routing",
    "spatial-wrong-worker-width",
];

/// The strict-acceptance control name.
const STRICT_ACCEPTANCE_CONTROL: &str = "strict-acceptance-single-failing-cell";

/// The controls that corrupt the parsed current-strata registration.
const REGISTRATION_CONTROLS: [&str; 3] = [
    "wrong-oracle-digest",
    "deleted-required-current-cell",
    "selected-implementation-mismatch",
];

/// Whether `name` is a registration control.
fn is_registration_control(name: &str) -> bool {
    REGISTRATION_CONTROLS.contains(&name)
}

/// Apply one registration control's in-memory corruption to the parsed registration. Returns
/// whether the named mutation was applied.
fn corrupt_registration(name: &str, cells: &mut Vec<RegisteredCell>) -> bool {
    match name {
        "wrong-oracle-digest" => match cells.iter_mut().find(|c| c.id == CONTROL_REGISTERED_CELL) {
            Some(cell) => {
                cell.oracle_digest = "0000000000000000".to_string();
                true
            }
            None => false,
        },
        "deleted-required-current-cell" => {
            let before = cells.len();
            cells.retain(|c| c.id != CONTROL_REGISTERED_CELL);
            cells.len() != before
        }
        "selected-implementation-mismatch" => {
            match cells.iter_mut().find(|c| c.id == CONTROL_REGISTERED_CELL) {
                Some(cell) => {
                    cell.implementation = "some::other::Table<K,V,H,N>".to_string();
                    true
                }
                None => false,
            }
        }
        _ => false,
    }
}

/// Spawn one control child and require it to be rejected (nonzero exit and a `rejected` marker).
fn run_control(name: &str, profile: Profile) -> Result<i32, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let output = Command::new(exe)
        .args(["--performance-control", name, "--profile", profile.name()])
        .output()
        .map_err(|e| format!("spawn control {name}: {e}"))?;
    if output.status.success() {
        return Err(format!(
            "control {name} exited 0; corrupted observations were accepted"
        ));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let marker = format!("{name} rejected:");
    if !stderr.contains(&marker) {
        return Err(format!(
            "control {name} failed for an unexpected reason: {}",
            stderr.trim()
        ));
    }
    Ok(output.status.code().unwrap_or(-1))
}

/// The control child: build the real witness, corrupt it in memory, and require the same validator
/// to reject it.
pub(crate) fn run_control_child(name: &str, profile: Profile) -> i32 {
    if spatial::is_control(name) {
        return spatial::run_control_child(name, profile);
    }
    if name == STRICT_ACCEPTANCE_CONTROL {
        return run_strict_acceptance_control(name);
    }
    let mut witness = match run_witness(profile) {
        Ok(witness) => witness,
        Err(e) => {
            eprintln!("performance-control witness failed: {e}");
            return 1;
        }
    };
    let report = build_performance_report(&witness, None, false);

    if is_registration_control(name) {
        let mut registered = match load_registered_cells() {
            Ok(cells) => cells,
            Err(e) => {
                eprintln!("performance-control {name} registration load failed: {e}");
                return 1;
            }
        };
        if !corrupt_registration(name, &mut registered) {
            eprintln!("performance-control {name} could not be applied to the registration");
            return 1;
        }
        let strata = strata::run_strata(Profile::Smoke);
        return match validate_all(&witness, &report, &strata, &registered, false) {
            Ok(()) => {
                eprintln!("performance-control {name} NOT rejected: validator accepted corruption");
                1
            }
            Err(e) => {
                eprintln!("performance-control {name} rejected: {e}");
                2
            }
        };
    }

    match name {
        "wrong-workload-identity" => {
            witness.per_shard_max = witness.per_shard_max.saturating_add(1);
        }
        "invalid-samples" => {
            if let Some(mode) = witness.modes.first_mut() {
                if let Some(first) = mode.apeiron_samples.first_mut() {
                    *first = 0;
                }
            }
        }
        "absent-samples" => {
            if let Some(mode) = witness.modes.first_mut() {
                mode.ht_samples.clear();
            }
        }
        "cached-home-only-collision" => {
            witness.collision = candidate::run_word_only_collision_witness();
        }
        _ => {
            eprintln!("unknown performance control: {name}");
            return 2;
        }
    }
    match validate_performance(&witness, &report) {
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

/// The report presentation for [`run_performance`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Output {
    /// The human default: the fixed-width `(cell, incumbent)` pair table plus the HEADLINE.
    Table,
    /// The existing machine-line dump.
    Json,
    /// Validation with only the identity line; the acceptance path prints its own table.
    Silent,
}

/// Run the performance check, validate it, and (when requested) require every control rejected.
/// `accept` selects the strict acceptance verdict: the process exits nonzero unless every
/// applicable strata cell passes and no applicable interval crosses the threshold. Without
/// `accept` the run is diagnostic and exits 0 on a valid boundary report. `output` selects the
/// human pair table (the default), the machine-line dump, or silent validation.
/// Print a wall-clock marker for attribution runs. Stderr only, and outside every timed sample
/// region, so the registered counts, statistics and output parsing are unchanged.
pub(crate) fn phase_mark(name: &str, total: std::time::Instant, last: &mut std::time::Instant) {
    let now = std::time::Instant::now();
    eprintln!(
        "phase={name} elapsed_s={:.3} total_s={:.3}",
        (now - *last).as_secs_f64(),
        total.elapsed().as_secs_f64()
    );
    *last = now;
}

#[allow(
    clippy::too_many_lines,
    reason = "one function states the performance campaign's phase sequence"
)]
pub(crate) fn run_performance(
    profile: Profile,
    controls: bool,
    accept: bool,
    output: Output,
) -> i32 {
    let t_total = std::time::Instant::now();
    let mut t_last = t_total;
    let witness = match run_witness(profile) {
        Ok(witness) => witness,
        Err(e) => {
            eprintln!("performance witness failed: {e}");
            return 1;
        }
    };
    phase_mark("perf.witness", t_total, &mut t_last);
    let strata = strata::run_strata(profile);
    phase_mark("perf.strata", t_total, &mut t_last);
    let registered = match load_registered_cells() {
        Ok(cells) => cells,
        Err(e) => {
            eprintln!("performance registration load failed: {e}");
            return 1;
        }
    };
    phase_mark("perf.registered", t_total, &mut t_last);
    let report = build_performance_report(&witness, Some(&strata), accept);
    phase_mark("perf.report", t_total, &mut t_last);
    print_identity(&format!(
        "reg1-performance-{}-shared-fixed16",
        profile.name()
    ));
    match output {
        Output::Json => {
            print_machine(&witness);
            print_counts(&witness);
            print_measurements(&witness);
            strata::print_strata(&strata);
            println!(
                "performance_registration registered={} run_cells={} \
                 source=matrix.json.current_strata identity=enforced accept={accept}",
                registered.len(),
                strata.cells.len()
            );
            print_report(&report);
        }
        Output::Table => {
            phase_mark("perf.spatial.begin", t_total, &mut t_last);
            let spatial = spatial::measure(&strata);
            phase_mark("perf.spatial", t_total, &mut t_last);
            if let Err(e) = spatial::validate(&spatial) {
                eprintln!("performance spatial validation failed: {e}");
                return 1;
            }
            let campaign = crate::acceptance::build_campaign(&strata, &registered, &spatial);
            phase_mark("perf.campaign", t_total, &mut t_last);
            let outcome = crate::acceptance::evaluate(&campaign, profile == Profile::Full);
            phase_mark("perf.evaluate", t_total, &mut t_last);
            crate::acceptance::print_pair_table(&campaign, &outcome);
            crate::acceptance::print_headline(&outcome);
        }
        Output::Silent => {}
    }
    phase_mark("perf.output", t_total, &mut t_last);
    if let Err(e) = validate_all(&witness, &report, &strata, &registered, accept) {
        eprintln!("performance validation failed: {e}");
        return 1;
    }
    phase_mark("perf.validate_all", t_total, &mut t_last);
    if let Err(e) = strata::validate_strata(&strata) {
        eprintln!("strata validation failed: {e}");
        return 1;
    }
    phase_mark("perf.validate_strata", t_total, &mut t_last);
    if controls {
        for name in CONTROLS {
            match run_control(name, profile) {
                Ok(code) => println!("control={name} observed=red child_exit={code}"),
                Err(e) => {
                    eprintln!("control {name} failed: {e}");
                    return 1;
                }
            }
        }
    }
    phase_mark("perf.controls", t_total, &mut t_last);
    if accept {
        let outcome = strict_acceptance(&strata);
        if outcome.passed() {
            println!(
                "strict_acceptance=pass applicable={} failing=0",
                outcome.applicable
            );
        } else {
            let worst = outcome.worst.as_ref().map_or_else(
                || "NONE".to_string(),
                |(id, r)| format!("{id} ratio={r:.4}"),
            );
            eprintln!(
                "performance strict acceptance failed: {}/{} applicable cells fail; \
                 worst_cell={worst}; first=\"{}\"",
                outcome.failing.len(),
                outcome.applicable,
                outcome.failing[0]
            );
            return 1;
        }
    }
    0
}

/// Print the machine, cache and thread context of the measurement.
fn print_machine(w: &PerfWitness) {
    let (ht, _) = build_hashtrinity(replacement_merge);
    println!("cpu={}", cpu_model());
    println!("kernel={}", uname_r());
    println!("threads=1 single_thread=true");
    println!(
        "cache_condition=warmed by {} warmup traces; tables rebuilt and touched before every sample",
        w.counts.warmup
    );
    println!(
        "apeiron_table_bytes={} candidate_table_bytes={} hashtrinity_capacity={} scheduled_ops={} \
         modes={}",
        std::mem::size_of::<ApeironTable>(),
        std::mem::size_of::<Candidate>(),
        ht.capacity(),
        w.counts.lookups + w.counts.updates,
        w.modes.len()
    );
    println!(
        "home_boundary_keys={} full_word_ok={} residue_ok={}",
        w.home_boundary.keys, w.home_boundary.full_word_ok, w.home_boundary.residue_ok
    );
    println!(
        "collision_keys={} word_equal={} occupancy={} capacity={} first_ok={} second_ok={} \
         zero_refused={} zero_default={}",
        w.collision.keys,
        w.collision.word_equal,
        w.collision.occupancy,
        w.collision.capacity,
        w.collision.first_read_ok,
        w.collision.second_read_ok,
        w.collision.zero_capacity_refused,
        w.collision.zero_read_is_default
    );
}

/// Print the active and full-profile operation counts.
fn print_counts(w: &PerfWitness) {
    println!(
        "profile={} lookups={} updates={} warmup={} samples={} modes={}",
        w.profile.name(),
        w.counts.lookups,
        w.counts.updates,
        w.counts.warmup,
        w.counts.samples,
        w.modes.len()
    );
    println!(
        "full_profile lookups={FULL_LOOKUPS} updates={FULL_UPDATES} warmup={FULL_WARMUP} \
         samples={FULL_SAMPLES}"
    );
}

/// Print the raw samples, the min/median/max summary, ns/op, ratio and the full-profile projection.
fn print_measurements(w: &PerfWitness) {
    let ops = w.counts.lookups + w.counts.updates;
    for mode in &w.modes {
        let a_med = mode.apeiron_median();
        let c_med = mode.candidate_median();
        let h_med = mode.ht_median();
        println!(
            "mode={} apeiron_min_ns={} apeiron_median_ns={a_med} apeiron_max_ns={} \
             candidate_min_ns={} candidate_median_ns={c_med} candidate_max_ns={} ht_min_ns={} \
             ht_median_ns={h_med} ht_max_ns={} ratio={:.4} candidate_ht_ratio={:.4} \
             candidate_current_ratio={:.4} apeiron_ns_per_op={:.3} candidate_ns_per_op={:.3} \
             ht_ns_per_op={:.3}",
            mode.mode.name(),
            minimum(&mode.apeiron_samples),
            maximum(&mode.apeiron_samples),
            minimum(&mode.candidate_samples),
            maximum(&mode.candidate_samples),
            minimum(&mode.ht_samples),
            maximum(&mode.ht_samples),
            ratio(a_med, h_med),
            ratio(c_med, h_med),
            ratio(c_med, a_med),
            ns_per_op(a_med, ops),
            ns_per_op(c_med, ops),
            ns_per_op(h_med, ops)
        );
        println!(
            "mode={} apeiron_samples_ns={}",
            mode.mode.name(),
            join(&mode.apeiron_samples)
        );
        println!(
            "mode={} candidate_samples_ns={}",
            mode.mode.name(),
            join(&mode.candidate_samples)
        );
        println!(
            "mode={} ht_samples_ns={}",
            mode.mode.name(),
            join(&mode.ht_samples)
        );
    }
    let projected = full_projection_seconds(w);
    let reproduces = (73.0 * 0.75..=73.0 * 1.25).contains(&projected);
    println!(
        "full_projection_seconds={projected:.2} reference_seconds=73 \
         reproduces_reference={reproduces} note=\"projected from smoke medians, not a measurement; \
         band +/-25%\""
    );
}

/// Join samples into a comma-separated string.
fn join(samples: &[u64]) -> String {
    samples
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// Project the full-profile wall time from the measured smoke medians. A projection, not a
/// measurement: it assumes the same per-op cost at the full counts.
fn full_projection_seconds(w: &PerfWitness) -> f64 {
    let full_ops = f64::from(FULL_LOOKUPS + FULL_UPDATES);
    let traces = f64::from(FULL_WARMUP + FULL_SAMPLES);
    let mode_count = f64::from(u32::try_from(w.modes.len()).unwrap_or(0));
    let ops = w.counts.lookups + w.counts.updates;
    let mut per_trace = 0.0f64;
    for mode in &w.modes {
        per_trace += ns_per_op(mode.apeiron_median(), ops) + ns_per_op(mode.ht_median(), ops);
    }
    if w.modes.is_empty() {
        return 0.0;
    }
    let mean = per_trace / f64::from(u32::try_from(w.modes.len()).unwrap_or(1));
    mean * full_ops * traces * mode_count / 1.0e9
}

/// The CPU model from `/proc/cpuinfo`.
fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .map(|line| line.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// The running kernel release.
fn uname_r() -> String {
    Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}
