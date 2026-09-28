//! REG-1 correctness smoke: actual Apeiron `Table`, the pinned hashtrinity `Table` and an
//! independent `std::HashMap` oracle over deterministic, preregistered small workloads.
//!
//! The smoke never claims REG-1: identity `pass` is the preregistered small witness (the full-scale
//! `1e8` coordinate-colliding run is unrun), round-trip stays `boundary` (no codec), growth/merge
//! `pass` covers region growth and the `N = 0` `SizeOverflow` refusal, and performance and coverage
//! stay `missing`. `report_complete` and `reg1_pass` are false.
//!
//! The controls are **validator controls, not source-mutation evidence**: each one runs the same
//! workload, corrupts the observation in memory and requires the same validator to reject it. No
//! source is mutated in this wave.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the correctness check and its witnesses"
)]

use std::process::Command;

use crate::workload::{
    run_admitted_witness, run_commutative_witness, run_determinism_witness, run_identity_witness,
    run_ordered_witness, run_refusal_witness, AdmittedWitness, CommutativeWitness,
    DeterminismWitness, IdentityWitness, KeyValue, OrderedWitness, RefusalWitness, APEIRON_N,
    KEY_COUNT,
};
use crate::{print_identity, print_report, AxisRecord, AxisStatus, Report};

/// The workload label the correctness gate prints.
pub(crate) const WORKLOAD: &str = "reg1-correctness-smoke";

/// Every observation the correctness smoke validates.
pub(crate) struct CorrectnessWitness {
    pub(crate) identity: IdentityWitness,
    pub(crate) admitted: AdmittedWitness,
    pub(crate) ordered: OrderedWitness,
    pub(crate) commutative: CommutativeWitness,
    pub(crate) refusal: RefusalWitness,
    pub(crate) determinism: DeterminismWitness,
}

/// Run every correctness workload once, including the separate-process determinism children.
#[must_use]
pub(crate) fn run_correctness_witness() -> CorrectnessWitness {
    CorrectnessWitness {
        identity: run_identity_witness(),
        admitted: run_admitted_witness(),
        ordered: run_ordered_witness(),
        commutative: run_commutative_witness(),
        refusal: run_refusal_witness(),
        determinism: run_determinism_witness(),
    }
}

/// Build the correctness report. The axis schema is the W1 schema; the statuses are the smoke's.
#[must_use]
pub(crate) fn build_correctness_report(w: &CorrectnessWitness) -> Report {
    let id = &w.identity;
    let det = &w.determinism;
    let det_status = if det.child_match {
        AxisStatus::Pass
    } else {
        AxisStatus::Fail
    };
    let axes = vec![
        AxisRecord {
            axis: "identity",
            status: AxisStatus::Pass,
            evidence: format!(
                "preregistered {KEY_COUNT}-key constant-coordinate witness: apeiron admits all \
                 {}/{} distinct keys with oracle agreement; hashtrinity coalesces {} submitted keys \
                 onto {} live coordinate; the full-scale 1e8 coordinate-colliding run remains unrun",
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
            status: det_status,
            evidence: format!(
                "separate-process comparison of the canonical sorted observation/refusal digest: \
                 child_count={}, all_match={}, all_exit_ok={}",
                det.child_count, det.child_match, det.child_exit_ok
            ),
        },
        AxisRecord {
            axis: "growth_merge_refusal",
            status: AxisStatus::Pass,
            evidence: format!(
                "region growth admits every distinct key in one home (full_keys={}, refused={}, \
                 preserved={}) and N=0 refuses with SizeOverflow (zero_refused={}); ordered=pass \
                 commutative=pass growth=pass; remove remains absent",
                w.refusal.full_keys,
                w.refusal.full_refused,
                w.refusal.state_unchanged,
                w.refusal.zero_capacity_refused
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

/// The one correctness validator. It fails closed on every observation, count and status.
///
/// # Errors
/// Returns a description of the first observation, count or status that violates the smoke
/// contract. A caller that observes `Ok(())` has a fail-closed report, not REG-1 acceptance.
pub(crate) fn validate_correctness(w: &CorrectnessWitness, r: &Report) -> Result<(), String> {
    validate_identity(&w.identity)?;
    validate_admitted(&w.admitted)?;
    validate_ordered(&w.ordered)?;
    validate_commutative(&w.commutative)?;
    validate_refusal(&w.refusal)?;
    validate_determinism(&w.determinism)?;
    validate_report(w, r)
}

/// Validate the identity observation and its counts.
fn validate_identity(w: &IdentityWitness) -> Result<(), String> {
    if w.oracle_len != KEY_COUNT as usize {
        return Err(format!(
            "identity oracle len {} != {KEY_COUNT}",
            w.oracle_len
        ));
    }
    if !w.oracle_readback_ok {
        return Err("identity oracle readback mismatch".to_string());
    }
    if w.ht_len != 1 {
        return Err(format!(
            "identity hashtrinity live entries {} != 1",
            w.ht_len
        ));
    }
    if !w.ht_readback_all_some {
        return Err("identity hashtrinity readback missed a submitted key".to_string());
    }
    if w.apeiron_admitted != KEY_COUNT as usize || w.apeiron_refused != 0 {
        return Err(format!(
            "identity apeiron admitted {} refused {} != {KEY_COUNT}/0",
            w.apeiron_admitted, w.apeiron_refused
        ));
    }
    if !w.apeiron_readback_ok {
        return Err("identity apeiron readback mismatch".to_string());
    }
    Ok(())
}

/// Validate the admitted fixed-capacity observation: counts, occupancy and per-key readback.
fn validate_admitted(w: &AdmittedWitness) -> Result<(), String> {
    let expected = w.keys;
    if expected == 0 {
        return Err("admitted workload is empty".to_string());
    }
    if w.per_shard_max == 0 || w.per_shard_max > APEIRON_N {
        return Err(format!(
            "admitted per-shard occupancy {} outside 1..={APEIRON_N}",
            w.per_shard_max
        ));
    }
    if w.apeiron_admitted != expected {
        return Err(format!(
            "admitted apeiron admitted {} != {expected}",
            w.apeiron_admitted
        ));
    }
    if w.ht_len != expected || w.oracle_len != expected {
        return Err(format!(
            "admitted lengths hashtrinity={} oracle={} != {expected}",
            w.ht_len, w.oracle_len
        ));
    }
    if w.apeiron.len() != expected || w.oracle.len() != expected || w.ht.len() != expected {
        return Err("admitted observation vector length mismatch".to_string());
    }
    if !w.readback_ok {
        return Err("admitted readback flag is false".to_string());
    }
    for i in 0..expected {
        let a = w.apeiron[i];
        let o = w.oracle[i];
        let h = w.ht[i];
        if a.key != o.key || a.key != h.key {
            return Err(format!("admitted key-set mismatch at index {i}"));
        }
        if a != o || a != h {
            return Err(format!(
                "admitted readback mismatch at key {}: apeiron={} oracle={} hashtrinity={}",
                a.key, a.value, o.value, h.value
            ));
        }
    }
    if !is_sorted(&w.apeiron) || !is_sorted(&w.oracle) || !is_sorted(&w.ht) {
        return Err("admitted observations are not in canonical key order".to_string());
    }
    Ok(())
}

/// Whether observations are strictly increasing by key.
fn is_sorted(obs: &[KeyValue]) -> bool {
    obs.windows(2).all(|pair| pair[0].key < pair[1].key)
}

/// Validate the ordered-merge fixture against its stream reference.
fn validate_ordered(w: &OrderedWitness) -> Result<(), String> {
    if w.keys == 0 {
        return Err("ordered fixture is empty".to_string());
    }
    let n = w.keys;
    if w.forward.len() != n
        || w.reverse.len() != n
        || w.reference_forward.len() != n
        || w.reference_reverse.len() != n
        || w.ht_forward.len() != n
        || w.ht_reverse.len() != n
    {
        return Err("ordered fixture vector length mismatch".to_string());
    }
    if w.forward != w.reference_forward {
        return Err("ordered forward fold disagrees with the stream reference".to_string());
    }
    if w.reverse != w.reference_reverse {
        return Err("ordered reversed fold disagrees with the stream reference".to_string());
    }
    if w.ht_forward != w.reference_forward {
        return Err(
            "ordered hashtrinity forward fold disagrees with the stream reference".to_string(),
        );
    }
    if w.ht_reverse != w.reference_reverse {
        return Err(
            "ordered hashtrinity reversed fold disagrees with the stream reference".to_string(),
        );
    }
    if !w.order_observable || w.forward == w.reverse {
        return Err("ordered fixture does not expose new-old order".to_string());
    }
    Ok(())
}

/// Validate the commutative fixture across original, permuted and partitioned streams.
fn validate_commutative(w: &CommutativeWitness) -> Result<(), String> {
    if w.keys == 0 {
        return Err("commutative fixture is empty".to_string());
    }
    let n = w.keys;
    if w.original.len() != n
        || w.permuted.len() != n
        || w.partitioned.len() != n
        || w.oracle.len() != n
        || w.ht_original.len() != n
        || w.ht_permuted.len() != n
    {
        return Err("commutative fixture vector length mismatch".to_string());
    }
    if w.original != w.permuted {
        return Err("commutative original and permuted sums disagree".to_string());
    }
    if w.original != w.partitioned {
        return Err("commutative original and partitioned sums disagree".to_string());
    }
    if w.original != w.oracle {
        return Err("commutative apeiron and oracle sums disagree".to_string());
    }
    if w.original != w.ht_original || w.original != w.ht_permuted {
        return Err("commutative hashtrinity sums disagree".to_string());
    }
    if !w.agree {
        return Err("commutative agreement flag is false".to_string());
    }
    Ok(())
}

/// Validate the growth fixture.
fn validate_refusal(w: &RefusalWitness) -> Result<(), String> {
    if w.full_keys <= APEIRON_N {
        return Err(format!(
            "growth full_keys {} not above the fixed group width {APEIRON_N}",
            w.full_keys
        ));
    }
    if w.full_refused {
        return Err("growth: a region refused an absent key instead of growing".to_string());
    }
    if w.refusal_is_shard_full {
        return Err(
            "growth: ShardFull must not be produced by the growable representation".to_string(),
        );
    }
    if !w.state_unchanged {
        return Err("growth: readback after growth lost a key or value".to_string());
    }
    if !w.zero_capacity_refused {
        return Err("growth: N=0 did not refuse with SizeOverflow".to_string());
    }
    if !w.zero_read_is_default {
        return Err("growth: N=0 read is not the default".to_string());
    }
    Ok(())
}

/// Validate the separate-process determinism observation.
fn validate_determinism(w: &DeterminismWitness) -> Result<(), String> {
    if w.child_count < 2 {
        return Err(format!(
            "determinism requires >= 2 children, got {}",
            w.child_count
        ));
    }
    if w.child_digests.len() != w.child_count {
        return Err("determinism digest vector length mismatch".to_string());
    }
    if !w.child_exit_ok {
        return Err("determinism: a child process did not exit 0".to_string());
    }
    let first = &w.child_digests[0];
    if first.is_empty() || !first.contains("digest=") {
        return Err("determinism child emitted no digest".to_string());
    }
    if !w.child_digests.iter().all(|d| d == first) {
        return Err("determinism child outputs differ".to_string());
    }
    if !w.child_match {
        return Err("determinism match flag is false".to_string());
    }
    Ok(())
}

/// Validate the report schema and every axis status against the observations.
fn validate_report(w: &CorrectnessWitness, r: &Report) -> Result<(), String> {
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
    if !matches!(
        r.axes[0].status,
        AxisStatus::Pass | AxisStatus::Boundary | AxisStatus::Missing | AxisStatus::Fail
    ) {
        return Err(format!(
            "identity status {} is not one of pass/boundary/missing/fail",
            r.axes[0].status.as_str()
        ));
    }
    if r.axes[0].status == AxisStatus::Pass && !w.identity.apeiron_readback_ok {
        return Err("identity passed without readback agreement".to_string());
    }
    if r.axes[1].status != AxisStatus::Boundary {
        return Err("round_trip must be boundary: no codec surface".to_string());
    }
    if w.determinism.child_match {
        if r.axes[2].status != AxisStatus::Pass {
            return Err("determinism matched but status is not pass".to_string());
        }
    } else if r.axes[2].status != AxisStatus::Fail {
        return Err("determinism did not match but status is not fail".to_string());
    }
    if !matches!(r.axes[3].status, AxisStatus::Boundary | AxisStatus::Pass) {
        return Err("growth_merge_refusal must be boundary or pass".to_string());
    }
    if r.axes[4].status != AxisStatus::Missing {
        return Err("performance must be missing: no measurement was run".to_string());
    }
    if r.axes[5].status != AxisStatus::Missing {
        return Err("coverage must be missing: no coverage census was run".to_string());
    }
    if r.report_complete {
        return Err("report_complete must be false".to_string());
    }
    if r.reg1_pass {
        return Err("reg1_pass must be false".to_string());
    }
    let growth = &r.axes[3].evidence;
    for token in ["ordered=pass", "commutative=pass", "growth=pass"] {
        if !growth.contains(token) {
            return Err(format!("growth evidence omits {token}"));
        }
    }
    Ok(())
}

// ── Gate and controls ──────────────────────────────────────────────────────────────────────────

/// The validator controls. Each corrupts an observation in memory; no source is mutated.
pub(crate) const CONTROLS: [&str; 3] = [
    "wrong-key-substitution",
    "reversed-ordered-merge",
    "altered-child-digest",
];

/// Run the correctness smoke, validate it, and (when requested) require every control rejected.
pub(crate) fn run_correctness(controls: bool) -> i32 {
    let witness = run_correctness_witness();
    let report = build_correctness_report(&witness);
    print_identity(WORKLOAD);
    print_report(&report);
    print_observations(&witness);
    if let Err(e) = validate_correctness(&witness, &report) {
        eprintln!("correctness validation failed: {e}");
        return 1;
    }
    if controls {
        for name in CONTROLS {
            match run_control(name) {
                Ok(code) => println!("control={name} observed=red child_exit={code}"),
                Err(e) => {
                    eprintln!("control {name} failed: {e}");
                    return 1;
                }
            }
        }
    }
    0
}

/// Print the scalar observation lines that accompany the six axis records.
fn print_observations(w: &CorrectnessWitness) {
    let id = &w.identity;
    println!(
        "identity_apeiron_admitted={} identity_apeiron_refused={} identity_ht_live={} identity_oracle_len={}",
        id.apeiron_admitted, id.apeiron_refused, id.ht_len, id.oracle_len
    );
    println!(
        "admitted_keys={} admitted_per_shard_max={} admitted_readback_ok={} ht_len={} oracle_len={}",
        w.admitted.keys,
        w.admitted.per_shard_max,
        w.admitted.readback_ok,
        w.admitted.ht_len,
        w.admitted.oracle_len
    );
    let forward = w.ordered.forward.first().map_or(i64::MIN, |kv| kv.value);
    let reverse = w.ordered.reverse.first().map_or(i64::MIN, |kv| kv.value);
    println!(
        "ordered_forward_first={forward} ordered_reverse_first={reverse} order_observable={}",
        w.ordered.order_observable
    );
    println!(
        "commutative_keys={} commutative_agree={} refusal_full_keys={} refusal_unchanged={} zero_refused={}",
        w.commutative.keys,
        w.commutative.agree,
        w.refusal.full_keys,
        w.refusal.state_unchanged,
        w.refusal.zero_capacity_refused
    );
    println!(
        "determinism_children={} determinism_match={} determinism_exit_ok={}",
        w.determinism.child_count, w.determinism.child_match, w.determinism.child_exit_ok
    );
}

/// Spawn one control child and require it to be rejected (nonzero exit and a `rejected` marker).
fn run_control(name: &str) -> Result<i32, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let output = Command::new(exe)
        .args(["--correctness-control", name])
        .output()
        .map_err(|e| format!("spawn control {name}: {e}"))?;
    if output.status.success() {
        return Err(format!(
            "control {name} exited 0; corrupted observations were accepted"
        ));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains("rejected") {
        return Err(format!(
            "control {name} failed for an unexpected reason: {}",
            stderr.trim()
        ));
    }
    Ok(output.status.code().unwrap_or(-1))
}

/// The control child: run the workload, corrupt one observation, and require rejection.
pub(crate) fn run_control_child(name: &str) -> i32 {
    let mut witness = run_correctness_witness();
    let report = build_correctness_report(&witness);
    match name {
        "wrong-key-substitution" => substitute_wrong_key(&mut witness),
        "reversed-ordered-merge" => reverse_ordered_result(&mut witness),
        "altered-child-digest" => alter_child_digest(&mut witness),
        _ => {
            eprintln!("unknown correctness control: {name}");
            return 2;
        }
    }
    match validate_correctness(&witness, &report) {
        Ok(()) => {
            eprintln!("correctness-control {name} NOT rejected: validator accepted corruption");
            1
        }
        Err(e) => {
            eprintln!("correctness-control {name} rejected: {e}");
            2
        }
    }
}

/// Replace one admitted key's readback value with a different key's value.
fn substitute_wrong_key(w: &mut CorrectnessWitness) {
    if w.admitted.apeiron.len() >= 2 {
        let other = w.admitted.apeiron[1].value;
        w.admitted.apeiron[0].value = other;
    } else if let Some(first) = w.admitted.apeiron.first_mut() {
        first.value = first.value.wrapping_add(1);
    }
}

/// Report the forward fold as the reversed fold, erasing the observable order.
fn reverse_ordered_result(w: &mut CorrectnessWitness) {
    w.ordered.reverse = w.ordered.forward.clone();
    w.ordered.order_observable = false;
}

/// Alter one child digest while leaving the match flag as reported.
fn alter_child_digest(w: &mut CorrectnessWitness) {
    if let Some(last) = w.determinism.child_digests.last_mut() {
        last.push('x');
    }
}
