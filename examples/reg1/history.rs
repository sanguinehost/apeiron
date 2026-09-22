//! REG-1 history mode: matrix inventory validation and isolated historical reproduction.
//!
//! `--check history` validates the hardcoded `matrix.json` contract (unique ids, the six mandatory
//! families, pinned revisions, the status vocabulary, registered workloads, and the rule that a
//! `historical` value is never accepted as fresh). It reproduces both historical campaigns in
//! isolated worktrees: the `71942db` spatial benchmark (apeiron at `71942db`, a Fold worktree at
//! `4cc14471`) and the `3a95bb5` serial Map campaign (`map_sweep` ns/key, `probe_cost`
//! instructions/lookup under `perf stat` on one core). Correctness precedes timing; every worktree
//! is removed on success and failure. Exit 0 means the framework is valid, not that G1 closed.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the REG-1 history check and its matrix"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{
    field, field_opt, opt_str_field, print_identity, print_report, str_field, string_map,
    AxisRecord, AxisStatus, Json, Reader, Report,
};

const MATRIX: &str = include_str!("matrix.json");
const LOCK: &str = include_str!("../../Cargo.lock");
const WORKLOAD: &str = "reg1-history-matrix-and-spatial-reproduction";
const SCHEMA: &str = "apeiron.reg1.history-matrix.v1";
const SPATIAL_REV: &str = "71942db30f8cf2970c7d3ceb7b5d7389d66fd0a4";
const SERIAL_REV: &str = "3a95bb512d59e56456d5a2edb1324639f63922b0";
const FOLD_REV: &str = "4cc14471ae24f3f204fb0b0e91736ef75c909358";
const CURRENT_REV: &str = "c450c222905c050f793a23332708c230c658f82c";

const SPATIAL_BENCH: &str = "fb905f3a93c70890cdb121a641240d041d50ac824a9e90ad277a75bfa42e6e66";
const SPATIAL_TOML: &str = "5094c209a5da52c479c39d11bd3e30d87329a1cafdb7281f47368248ab80d943";
const SPATIAL_LOCK: &str = "c3a3f09a44300b725e4249c149ce71ae4677f4b82f390adcea8f8bf2a9157f83";
const SPATIAL_TABLE: &str = "c345b63bca8d9a6fb46cdf4cc57f4433d942acbd414d5e5ea07e74307978e1b9";
const SPATIAL_COORD: &str = "170d4da5d8658b7a1f039091e7ed87d4a9ae5da943abb62b05b431c1f3acf504";
const FOLD_TOML: &str = "f78f93835849030722b5b90b87efb655b9c4a8a833d989a9d378c0278731194c";
const FOLD_LIB: &str = "f8c8fd12fc2b7a69c0a045611d70e3a02fc43310e56c69de75e163341ed19e87";
const FOLD_FREQ: &str = "02f8bfdbdd47e187f2b26aa5b233ef83baf9f41babb4e06f501ddeb7f6358898";

/// The `3a95bb5` serial campaign source digests: the three examples, the two implementation modules
/// they exercise and the manifest/lock they build against.
const SERIAL_MAP: &str = "8ba4f4785c2a79abaa220bd21a22f1694f08921c275a8f49616d77bb109f9e6e";
const SERIAL_PROBE: &str = "b7bae57d6f3fcad1cbee903351d82a9fbe6dbd02d1878d42a6545b681e4ba1e2";
const SERIAL_FOLD: &str = "b3aa896943d77e276a3600b9090c32af5c146d07a49586761a9cfc0690eea93d";
const SERIAL_RS: &str = "84a9f26771dc0508cd5ea10e48882c8e4ef8caf516b5076afb34116e0fa3b22c";
const SERIAL_FOLD_RS: &str = "f046da964af81e3d4e3ff7b8a288998f94b15698106113c369769e6baa37a87a";
const SERIAL_TOML: &str = "3298d1bb423f0cdfcf34201e1d48bf6debb7f237165e4de1d18edf0d1cd32207";
const SERIAL_LOCK: &str = "e2c7c27cc017f31f9a06565414f85909dc26101abc3e6c0a9dc77c34f10427e5";

/// The `3a95bb5` build and run commands. `perf stat` does not inherit into a child, so the counter
/// arms run the built example binary directly, pinned to one core so a hybrid CPU reports a single
/// instruction PMU instead of a split E-core/P-core count.
const SERIAL_BUILD_CMD: &str =
    "cargo build --release --offline --example map_sweep --example probe_cost";
const SERIAL_MAP_CMD: &str = "cargo run --release --offline --example map_sweep";
const SERIAL_PROBE_CMD: &str =
    "taskset -c 2 perf stat -e instructions:u -- target/release/examples/probe_cost";
const PERF_CPU: &str = "2";
const PERF_REPS: u32 = 3;
/// The `probe_cost` presence oracle: `694` keys, `200_000` repetitions, every lookup a hit.
const PROBE_LOOKUPS: u64 = 138_800_000;
const PROBE_LOOKUPS_F: f64 = 138_800_000.0;
/// The `probe_cost` arms: (matrix `impl_id`, matrix dimension, `LOOKUP_ARM`).
#[rustfmt::skip]
const PROBE_ARMS: [(&str, &str, &str); 7] = [
    ("apeiron_map", "instr_u64_lookup", "apeiron"), ("std_hashmap", "instr_u64_lookup", "std"),
    ("hashbrown_foldhash", "instr_u64_lookup", "fold"), ("hashbrown_fxhash", "instr_u64_lookup", "fx"),
    ("apeiron_map", "instr_string_lookup", "apstr"), ("hashbrown_foldhash", "instr_string_lookup", "foldstr"),
    ("hashbrown_fxhash", "instr_string_lookup", "fxstr"),
];
/// The original historical spatial command, without the harness's bounded timeout wrapper.
const SPATIAL_CMD: &str = "cargo bench --bench spatial_binning --offline";
const BENCH_TIMEOUT_SECS: u64 = 900;
const BENCH_SIZES: [u32; 2] = [100_000, 1_000_000];
/// The correctness arms that must match the oracle at both sizes before timing is accepted.
const REQUIRED_OK: [&str; 3] = ["apeiron_serial", "apeiron_parallel", "hashtrinity_packed"];
/// The reproduced spatial cells: (matrix `impl_id`, criterion arm name). `hashtrinity_tuple` is
/// included so its exact failure at both sizes is observed, but it never promotes.
#[rustfmt::skip]
const RERUN_ARMS: [(&str, &str); 10] = [
    ("apeiron_parallel", "apeiron_par16"), ("hashtrinity_packed", "hashtrinity_packed"),
    ("hashtrinity_tuple", "hashtrinity_tuple"), ("hashbrown_foldhash", "hashbrown_foldhash"),
    ("hashbrown_fxhash", "hashbrown_fxhash"), ("std_siphash_par16", "std_siphash_par16"),
    ("foldhash_par16", "foldhash_par16"), ("fxhash_par16", "fxhash_par16"),
    ("apeiron_serial", "apeiron_serial"), ("apeiron_tuple_par16", "apeiron_tuple_par16"),
];

#[rustfmt::skip]
const STATUSES: [&str; 6] = ["measured", "historical", "missing", "boundary", "fail", "tool_error"];

/// The validator controls. Each corrupts the parsed matrix in memory and must be rejected by
/// `validate_matrix` for the intended reason.
const CONTROLS: [&str; 7] = [
    "wrong-oracle-status",
    "wrong-source-digest",
    "wrong-revision",
    "wrong-workload-identity",
    "required-cell-deleted",
    "promoted-oracle-flipped",
    "promoted-digest-cleared",
];

/// One mandatory matrix family. `allowed` is the `|`-joined implementation ids a cell may name.
/// `required` lists the mandatory dimensions and their `|`-joined implementation ids.
struct FamilySpec {
    id: &'static str,
    revision: &'static str,
    source_sha256: Option<&'static str>,
    allowed: &'static str,
    required: &'static [(&'static str, &'static str)],
}

/// The shared `|`-joined implementation lists, so the mandatory contract reads once.
const SPATIAL_REQ: &str =
    "apeiron_parallel|hashtrinity_packed|hashbrown_foldhash|hashbrown_fxhash|std_siphash_par16";
const SERIAL_REQ: &str = "apeiron_map|std_hashmap|hashbrown_foldhash|hashbrown_fxhash";
const CURRENT_REQ: &str = "apeiron_table|hashtrinity_table";
const APEIRON_REQ: &str = "apeiron_table";

/// The mandatory matrix contract. A cell outside these families, dimensions or implementations is
/// an error, and every required triple must be present.
#[rustfmt::skip]
const FAMILIES: &[FamilySpec] = &[
    FamilySpec { id: "historical_spatial_parallel", revision: SPATIAL_REV, source_sha256: Some(SPATIAL_BENCH), allowed: "apeiron_parallel|hashtrinity_packed|hashbrown_foldhash|hashbrown_fxhash|std_siphash_par16|foldhash_par16|fxhash_par16|apeiron_serial|hashtrinity_tuple|apeiron_tuple_par16", required: &[("n100000", SPATIAL_REQ), ("n1000000", SPATIAL_REQ)] },
    FamilySpec { id: "historical_serial_map", revision: SERIAL_REV, source_sha256: None, allowed: SERIAL_REQ, required: &[("int_build_get", SERIAL_REQ), ("int_get", SERIAL_REQ), ("string_build_get", SERIAL_REQ), ("string_get", SERIAL_REQ), ("instr_u64_lookup", SERIAL_REQ), ("instr_string_lookup", "apeiron_map|hashbrown_foldhash|hashbrown_fxhash"), ("fold_instr", "apeiron_map|hashbrown_foldhash")] },
    FamilySpec { id: "current_occupancy", revision: CURRENT_REV, source_sha256: None, allowed: CURRENT_REQ, required: &[("occupancy_1", CURRENT_REQ), ("occupancy_2", CURRENT_REQ), ("occupancy_4", CURRENT_REQ), ("occupancy_8", CURRENT_REQ), ("occupancy_16", CURRENT_REQ)] },
    FamilySpec { id: "current_coordinate", revision: CURRENT_REV, source_sha256: None, allowed: CURRENT_REQ, required: &[("direct_index_resident", CURRENT_REQ), ("offset", CURRENT_REQ), ("random", CURRENT_REQ), ("clustered_skewed", CURRENT_REQ)] },
    FamilySpec { id: "operation_strata", revision: CURRENT_REV, source_sha256: None, allowed: APEIRON_REQ, required: &[("hit", APEIRON_REQ), ("miss", APEIRON_REQ), ("existing_update", APEIRON_REQ), ("admitted_insert", APEIRON_REQ), ("refusal", APEIRON_REQ)] },
    FamilySpec { id: "correctness_rows", revision: CURRENT_REV, source_sha256: None, allowed: APEIRON_REQ, required: &[("identity", APEIRON_REQ), ("round_trip", APEIRON_REQ), ("determinism", APEIRON_REQ), ("growth_merge", APEIRON_REQ), ("coverage", APEIRON_REQ)] },
];

/// The workload identities every registered matrix must declare.
#[rustfmt::skip]
const MANDATORY_WORKLOADS: [&str; 12] = [
    "spatial-scene-clustered-cells-packed", "spatial-scene-clustered-cells-tuple",
    "serial-map-integer-keys", "serial-map-resref-keys", "serial-probe-instructions",
    "reg1-shared-fixed16-schedule", "reg1-direct-index-resident", "reg1-offset-coordinate",
    "reg1-random-coordinate", "reg1-clustered-skewed", "reg1-operation-strata",
    "reg1-correctness-rows",
];

/// One fresh measurement receipt. A `measured` cell must carry one whose `oracle_ok` is true.
struct Fresh {
    command: String,
    revision: String,
    output_sha256: String,
    oracle_ok: bool,
    result: String,
    wall_ms: String,
}

struct Cell {
    id: String,
    family: String,
    dimension: String,
    workload: String,
    impl_id: String,
    implementation: String,
    baseline: String,
    revision: String,
    command: String,
    metric: String,
    oracle: String,
    control: String,
    status: String,
    source_sha256: Option<String>,
    historical_value: Option<String>,
    evidence: Option<String>,
    fresh: Option<Fresh>,
}

struct Matrix {
    schema: String,
    pins: BTreeMap<String, String>,
    workloads: BTreeMap<String, String>,
    cells: Vec<Cell>,
}

fn parse_matrix(text: &str) -> Result<Matrix, String> {
    let root = Reader::new(text).parse()?;
    let obj = root.as_obj()?;
    let Json::Arr(entries) = field(obj, "cells")? else {
        return Err("field cells: expected array".to_string());
    };
    let mut cells = Vec::with_capacity(entries.len());
    for entry in entries {
        cells.push(parse_cell(entry.as_obj()?)?);
    }
    Ok(Matrix {
        schema: str_field(obj, "schema")?,
        pins: string_map(obj, "pins")?,
        workloads: string_map(obj, "workloads")?,
        cells,
    })
}

fn parse_cell(obj: &[(String, Json)]) -> Result<Cell, String> {
    Ok(Cell {
        id: str_field(obj, "id")?,
        family: str_field(obj, "family")?,
        dimension: str_field(obj, "dimension")?,
        workload: str_field(obj, "workload")?,
        impl_id: str_field(obj, "impl_id")?,
        implementation: str_field(obj, "implementation")?,
        baseline: str_field(obj, "baseline")?,
        revision: str_field(obj, "revision")?,
        command: str_field(obj, "command")?,
        metric: str_field(obj, "metric")?,
        oracle: str_field(obj, "oracle")?,
        control: str_field(obj, "control")?,
        status: str_field(obj, "status")?,
        source_sha256: opt_str_field(obj, "source_sha256")?,
        historical_value: opt_str_field(obj, "historical_value")?,
        evidence: opt_str_field(obj, "evidence")?,
        fresh: parse_fresh(obj)?,
    })
}

/// Parse an optional `fresh` receipt. The matrix JSON stores `oracle_ok` as the string `"true"`.
fn parse_fresh(obj: &[(String, Json)]) -> Result<Option<Fresh>, String> {
    match field_opt(obj, "fresh") {
        None => Ok(None),
        Some(Json::Obj(f)) => Ok(Some(Fresh {
            command: str_field(f, "command")?,
            revision: str_field(f, "revision")?,
            output_sha256: str_field(f, "output_sha256")?,
            oracle_ok: str_field(f, "oracle_ok")? == "true",
            result: str_field(f, "result")?,
            wall_ms: str_field(f, "wall_ms")?,
        })),
        Some(_) => Err("field fresh: expected object".to_string()),
    }
}

/// A 64-character lowercase hex digest.
fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Require a cell's source digest to match the family's registered digest when required.
fn source_ok(cell: &Cell, family: &FamilySpec, required: bool) -> Result<(), String> {
    let sha = match &cell.source_sha256 {
        Some(sha) => sha,
        None if required => return Err(format!("cell {} has no source_sha256", cell.id)),
        None => return Ok(()),
    };
    if !is_sha256(sha) || family.source_sha256.is_some_and(|e| sha != e) {
        return Err(format!("cell {} source digest", cell.id));
    }
    Ok(())
}

/// Validate the parsed matrix against the hardcoded contract. This is the single validator the
/// controls corrupt and re-run.
#[allow(
    clippy::too_many_lines,
    reason = "the whole matrix contract is stated once"
)]
fn validate_matrix(m: &Matrix) -> Result<(), String> {
    if m.schema != SCHEMA {
        return Err(format!("schema {} != {SCHEMA}", m.schema));
    }
    for (key, expected) in [
        ("historical_spatial_parallel", SPATIAL_REV),
        ("historical_serial_map", SERIAL_REV),
        ("fold", FOLD_REV),
        ("current", CURRENT_REV),
    ] {
        match m.pins.get(key) {
            Some(a) if a == expected => (),
            Some(a) => return Err(format!("pin {key}: {a} != {expected}")),
            None => return Err(format!("pin {key} missing")),
        }
    }
    for w in MANDATORY_WORKLOADS {
        if !m.workloads.contains_key(w) {
            return Err(format!("workload {w} missing"));
        }
    }
    let mut seen = BTreeSet::new();
    for cell in &m.cells {
        if !seen.insert(cell.id.clone()) {
            return Err(format!("duplicate id {}", cell.id));
        }
        let family = FAMILIES
            .iter()
            .find(|f| f.id == cell.family.as_str())
            .ok_or_else(|| format!("cell {} unknown family", cell.id))?;
        if !family
            .required
            .iter()
            .any(|(d, _)| *d == cell.dimension.as_str())
        {
            return Err(format!("cell {} bad dimension", cell.id));
        }
        if !family.allowed.split('|').any(|c| c == cell.impl_id) {
            return Err(format!("cell {} bad impl_id", cell.id));
        }
        if cell.revision != family.revision {
            return Err(format!("cell {} bad revision", cell.id));
        }
        if !STATUSES.contains(&cell.status.as_str()) {
            return Err(format!("cell {} bad status", cell.id));
        }
        for (v, what) in [
            (&cell.workload, "workload"),
            (&cell.implementation, "implementation"),
            (&cell.baseline, "baseline"),
            (&cell.command, "command"),
            (&cell.metric, "metric"),
            (&cell.oracle, "oracle"),
            (&cell.control, "control"),
        ] {
            if v.trim().is_empty() {
                return Err(format!("cell {} empty {what}", cell.id));
            }
        }
        if !m.workloads.contains_key(&cell.workload) {
            return Err(format!("cell {} bad workload", cell.id));
        }
        if cell
            .evidence
            .as_deref()
            .is_some_and(|e| e.trim().is_empty())
        {
            return Err(format!("cell {} empty evidence", cell.id));
        }
        // The serial Map campaign is a Map comparison; naming hashtrinity would mislabel it.
        if cell.family == "historical_serial_map"
            && (cell.impl_id.contains("hashtrinity")
                || cell.baseline.contains("hashtrinity")
                || cell.implementation.contains("hashtrinity"))
        {
            return Err(format!("cell {} names hashtrinity", cell.id));
        }
        match cell.status.as_str() {
            "measured" => {
                let fresh = cell
                    .fresh
                    .as_ref()
                    .ok_or_else(|| format!("cell {} no fresh", cell.id))?;
                if fresh.command.trim().is_empty()
                    || fresh.result.trim().is_empty()
                    || fresh.wall_ms.trim().is_empty()
                {
                    return Err(format!("cell {} incomplete fresh", cell.id));
                }
                if fresh.revision != family.revision {
                    return Err(format!("cell {} fresh revision", cell.id));
                }
                if !is_sha256(&fresh.output_sha256) {
                    return Err(format!("cell {} fresh digest", cell.id));
                }
                if !fresh.oracle_ok {
                    return Err(format!("cell {} oracle_ok=false", cell.id));
                }
                if cell.historical_value.is_some() {
                    return Err(format!("cell {} measured+historical", cell.id));
                }
                source_ok(cell, family, false)?;
            }
            "historical" => {
                let value = cell
                    .historical_value
                    .as_deref()
                    .ok_or_else(|| format!("cell {} no historical_value", cell.id))?;
                if value.trim().is_empty() {
                    return Err(format!("cell {} empty historical_value", cell.id));
                }
                if cell.fresh.is_some() {
                    return Err(format!("cell {} historical+fresh", cell.id));
                }
                source_ok(cell, family, true)?;
            }
            _ => {
                if cell.fresh.is_some() {
                    return Err(format!("cell {} non-measured+fresh", cell.id));
                }
            }
        }
    }
    for family in FAMILIES {
        for (dimension, impls) in family.required {
            for imp in impls.split('|') {
                if !m
                    .cells
                    .iter()
                    .any(|c| c.family == family.id && c.dimension == *dimension && c.impl_id == imp)
                {
                    return Err(format!("missing {} {dimension} {imp}", family.id));
                }
            }
        }
    }
    Ok(())
}

/// The spatial reproduction receipt.
struct Repro {
    root: String,
    original_command: String,
    executed_command: String,
    apeiron_worktree: String,
    fold_worktree: String,
    apeiron_head: String,
    fold_head: String,
    wall_ms: u128,
    output_sha256: String,
    correctness: Vec<(u32, String, bool, String)>,
    /// The sha256 of every passing arm's exact correctness line, by `(n, arm)`.
    oracle_digests: BTreeMap<(u32, String), String>,
    times: Vec<(u32, String, f64)>,
}

/// One `probe_cost` arm's counter receipt.
struct ProbeRow {
    impl_id: String,
    dimension: String,
    arm: String,
    hits: u64,
    lo: f64,
    median: f64,
    hi: f64,
    line_digest: String,
}

/// The `3a95bb5` serial Map campaign receipt.
struct SerialRepro {
    root: String,
    worktree: String,
    head: String,
    map_output_sha256: String,
    map_wall_ms: u128,
    wall_ms: u128,
    probe: Vec<ProbeRow>,
}

/// The temporary root: `/tmp/apeiron/reg1-history-<pid>` unless overridden.
fn history_root() -> PathBuf {
    std::env::var("APEIRON_REG1_HISTORY_ROOT").map_or_else(
        |_| std::env::temp_dir().join(format!("opencode/reg1-history-{}", std::process::id())),
        PathBuf::from,
    )
}

/// The temporary root, its three worktrees and their repositories, removed on drop.
struct Isolated {
    root: PathBuf,
    apeiron_repo: PathBuf,
    fold_repo: PathBuf,
    apeiron_worktree: PathBuf,
    serial_worktree: PathBuf,
    fold_worktree: PathBuf,
    removed: bool,
}

impl Isolated {
    /// Remove every worktree, prune the records and delete the temporary root. Idempotent.
    fn cleanup(&mut self) -> Result<(), String> {
        if self.removed {
            return Ok(());
        }
        self.removed = true;
        let mut errors = Vec::new();
        for (repo, worktree) in [
            (&self.apeiron_repo, &self.apeiron_worktree),
            (&self.apeiron_repo, &self.serial_worktree),
            (&self.fold_repo, &self.fold_worktree),
        ] {
            if !worktree.exists() {
                continue;
            }
            match Command::new("git")
                .arg("-C")
                .arg(repo)
                .args(["worktree", "remove", "--force"])
                .arg(worktree)
                .output()
            {
                Ok(out) if out.status.success() => (),
                Ok(out) => errors.push(format!(
                    "worktree remove {}: {}",
                    worktree.display(),
                    String::from_utf8_lossy(&out.stderr).trim()
                )),
                Err(e) => errors.push(format!("worktree remove {}: {e}", worktree.display())),
            }
        }
        for repo in [&self.apeiron_repo, &self.fold_repo] {
            let _ = Command::new("git")
                .arg("-C")
                .arg(repo)
                .args(["worktree", "prune"])
                .output();
        }
        if self.root.exists() {
            if let Err(e) = fs::remove_dir_all(&self.root) {
                errors.push(format!("remove {}: {e}", self.root.display()));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

impl Drop for Isolated {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// Resolve the hashtrinity dependency path and revision from the embedded lockfile.
fn lock_hashtrinity_source() -> Result<(PathBuf, String), String> {
    let mut in_block = false;
    for line in LOCK.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            in_block = false;
        } else if line == "name = \"hashtrinity\"" {
            in_block = true;
        } else if in_block && line.starts_with("source = ") {
            let source = line
                .trim_start_matches("source = ")
                .trim_matches('"')
                .to_string();
            let rest = source.strip_prefix("git+").unwrap_or(&source);
            let (url, tail) = rest
                .split_once("?rev=")
                .ok_or("hashtrinity source has no ?rev=")?;
            let rev = tail.split('#').next().unwrap_or("").to_string();
            let path = url
                .strip_prefix("file://")
                .ok_or("hashtrinity is not a file:// dependency; cannot isolate it")?;
            return Ok((PathBuf::from(path), rev));
        }
    }
    Err("Cargo.lock has no hashtrinity git source".to_string())
}

fn git_rev_parse(repo: &Path, rev: &str) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--verify"])
        .arg(format!("{rev}^{{commit}}"))
        .output()
        .map_err(|e| format!("git rev-parse in {}: {e}", repo.display()))?;
    if !output.status.success() {
        return Err(format!(
            "git rev-parse {rev} in {}: {}",
            repo.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let resolved = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if resolved.is_empty() {
        Err(format!("git rev-parse {rev} produced no revision"))
    } else {
        Ok(resolved)
    }
}

fn worktree_add(repo: &Path, worktree: &Path, rev: &str) -> Result<(), String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["worktree", "add", "--detach"])
        .arg(worktree)
        .arg(rev)
        .output()
        .map_err(|e| format!("git worktree add in {}: {e}", repo.display()))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "git worktree add {} {rev}: {}",
            worktree.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|e| format!("sha256sum {}: {e}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "sha256sum {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let digest = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    if !is_sha256(&digest) {
        return Err(format!("sha256sum {} gave a non-digest", path.display()));
    }
    Ok(digest)
}

/// The sha256 of one text line, written to `path` so the file digest path is reused unchanged.
fn sha256_text(text: &str, path: &Path) -> Result<String, String> {
    fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
    sha256_file(path)
}

/// Run a command with stdout and stderr redirected to files, bounded by a wall-clock deadline.
fn run_capture(
    command: &mut Command,
    timeout: Duration,
    stdout_path: &Path,
    stderr_path: &Path,
) -> Result<(i32, u128), String> {
    let stdout = fs::File::create(stdout_path)
        .map_err(|e| format!("create {}: {e}", stdout_path.display()))?;
    let stderr = fs::File::create(stderr_path)
        .map_err(|e| format!("create {}: {e}", stderr_path.display()))?;
    let start = Instant::now();
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|e| format!("spawn {command:?}: {e}"))?;
    loop {
        if let Some(status) = child.try_wait().map_err(|e| format!("wait: {e}"))? {
            return Ok((status.code().unwrap_or(-1), start.elapsed().as_millis()));
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("timed out after {}s", timeout.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Parse the `correctness n=<n> <arm> ok|WRONG ...` lines the historical bench prints, keeping the
/// exact trimmed line so its digest can be registered as the cell's oracle receipt.
fn parse_correctness(stdout: &str) -> Result<Vec<(u32, String, bool, String)>, String> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() >= 4 && tokens[0] == "correctness" && tokens[1].starts_with("n=") {
            let n: u32 = tokens[1][2..]
                .parse()
                .map_err(|e| format!("correctness line {line}: {e}"))?;
            let arm = tokens[2].to_string();
            let ok = match tokens[3] {
                "ok" => true,
                "WRONG" => false,
                other => return Err(format!("unknown correctness mark {other} in {line}")),
            };
            out.push((n, arm, ok, line.trim().to_string()));
        }
    }
    Ok(out)
}

/// The median of a criterion `time: [lo mid hi unit]` line, in milliseconds.
fn parse_time_median(line: &str) -> Option<f64> {
    let open = line.find('[')?;
    let close = line.find(']')?;
    let tokens: Vec<&str> = line[open + 1..close].split_whitespace().collect();
    if tokens.len() < 4 {
        return None;
    }
    let value: f64 = tokens[2].parse().ok()?;
    match tokens[3] {
        "ns" => Some(value / 1e6),
        "us" | "µs" => Some(value / 1e3),
        "ms" => Some(value),
        "s" => Some(value * 1e3),
        _ => None,
    }
}

/// Parse the `bin_and_probe/<n>/<arm>/<n>` header and its following `time:` line.
fn parse_times(stdout: &str) -> Result<Vec<(u32, String, f64)>, String> {
    let lines: Vec<&str> = stdout.lines().collect();
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(rest) = line.trim().strip_prefix("bin_and_probe/") else {
            continue;
        };
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() < 2 {
            continue;
        }
        let n: u32 = parts[0]
            .parse()
            .map_err(|e| format!("bench header {}: {e}", line.trim()))?;
        let arm = parts[1].to_string();
        if let Some(median) = lines[index + 1..]
            .iter()
            .find(|l| l.contains("time:"))
            .and_then(|l| parse_time_median(l))
        {
            out.push((n, arm, median));
        }
    }
    Ok(out)
}

fn tail(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= limit {
        trimmed.to_string()
    } else {
        trimmed[trimmed.len() - limit..].to_string()
    }
}

/// Create and pin every worktree, then verify each registered source digest.
fn prepare_worktrees(repo: &Path) -> Result<Isolated, String> {
    let (fold_repo, lock_rev) = lock_hashtrinity_source()?;
    if lock_rev != FOLD_REV {
        return Err(format!(
            "Cargo.lock resolves hashtrinity {lock_rev} != pinned {FOLD_REV}"
        ));
    }
    for (rev, what) in [
        (SPATIAL_REV, "apeiron"),
        (SERIAL_REV, "apeiron"),
        (FOLD_REV, "fold"),
    ] {
        let observed = if what == "fold" {
            git_rev_parse(&fold_repo, rev)?
        } else {
            git_rev_parse(repo, rev)?
        };
        if observed != rev {
            return Err(format!("{what} {rev} resolved to {observed}"));
        }
    }
    let root = history_root();
    if root.exists() {
        fs::remove_dir_all(&root).map_err(|e| format!("clear {}: {e}", root.display()))?;
    }
    fs::create_dir_all(&root).map_err(|e| format!("create {}: {e}", root.display()))?;
    let isolated = Isolated {
        root: root.clone(),
        apeiron_repo: repo.to_path_buf(),
        fold_repo: fold_repo.clone(),
        apeiron_worktree: root.join("apeiron"),
        serial_worktree: root.join("serial"),
        fold_worktree: root.join("fold"),
        removed: false,
    };
    worktree_add(repo, &isolated.apeiron_worktree, SPATIAL_REV)?;
    worktree_add(repo, &isolated.serial_worktree, SERIAL_REV)?;
    worktree_add(&fold_repo, &isolated.fold_worktree, FOLD_REV)?;
    for (worktree, rev, what) in [
        (&isolated.apeiron_worktree, SPATIAL_REV, "apeiron"),
        (&isolated.serial_worktree, SERIAL_REV, "serial"),
        (&isolated.fold_worktree, FOLD_REV, "fold"),
    ] {
        if git_rev_parse(worktree, "HEAD")? != rev {
            return Err(format!("{what} worktree HEAD is not the pinned revision"));
        }
    }
    let (a, s, f) = (
        &isolated.apeiron_worktree,
        &isolated.serial_worktree,
        &isolated.fold_worktree,
    );
    let registered: &[(&str, &str, &Path)] = &[
        ("benches/spatial_binning.rs", SPATIAL_BENCH, a),
        ("Cargo.toml", SPATIAL_TOML, a),
        ("Cargo.lock", SPATIAL_LOCK, a),
        ("src/table.rs", SPATIAL_TABLE, a),
        ("src/coord.rs", SPATIAL_COORD, a),
        ("hashtrinity/Cargo.toml", FOLD_TOML, f),
        ("hashtrinity/src/lib.rs", FOLD_LIB, f),
        ("hashtrinity/src/freq.rs", FOLD_FREQ, f),
        ("examples/map_sweep.rs", SERIAL_MAP, s),
        ("examples/probe_cost.rs", SERIAL_PROBE, s),
        ("examples/fold_instr.rs", SERIAL_FOLD, s),
        ("src/map.rs", SERIAL_RS, s),
        ("src/fold.rs", SERIAL_FOLD_RS, s),
        ("Cargo.toml", SERIAL_TOML, s),
        ("Cargo.lock", SERIAL_LOCK, s),
    ];
    for (relative, expected, base) in registered.iter().copied() {
        let observed = sha256_file(&base.join(relative))?;
        if observed != expected {
            return Err(format!(
                "source {relative} digest {observed} != registered {expected}"
            ));
        }
        println!(
            "history_source path={relative} sha256={observed} registered={expected} match=yes"
        );
    }
    Ok(isolated)
}

/// Run the original spatial bench in the pinned worktrees and build the receipt.
fn reproduce(isolated: &Isolated, repo: &Path) -> Result<Repro, String> {
    let target = repo.join("target/reg1-history/target");
    fs::create_dir_all(&target).map_err(|e| format!("create {}: {e}", target.display()))?;
    let stdout_path = isolated.root.join("bench.out");
    let stderr_path = isolated.root.join("bench.err");
    let executed_command = format!("timeout {BENCH_TIMEOUT_SECS}s {SPATIAL_CMD}");
    let timeout_arg = format!("{BENCH_TIMEOUT_SECS}s");
    let mut command = Command::new("timeout");
    command
        .args([
            timeout_arg.as_str(),
            "cargo",
            "bench",
            "--bench",
            "spatial_binning",
            "--offline",
        ])
        .current_dir(&isolated.apeiron_worktree)
        .env("CARGO_TARGET_DIR", &target)
        .env("CARGO_NET_OFFLINE", "true")
        .env("CARGO_TERM_COLOR", "never")
        .env("RUST_BACKTRACE", "0");
    let (code, wall_ms) = run_capture(
        &mut command,
        Duration::from_secs(BENCH_TIMEOUT_SECS + 60),
        &stdout_path,
        &stderr_path,
    )
    .map_err(|e| format!("historical bench {executed_command}: {e}"))?;
    let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
    if code != 0 {
        return Err(format!(
            "historical bench {executed_command} exited {code}: {}",
            tail(&stderr, 800)
        ));
    }
    let output_sha256 = sha256_file(&stdout_path)?;
    let correctness = parse_correctness(&stdout)?;
    let times = parse_times(&stdout)?;
    for n in BENCH_SIZES {
        for arm in REQUIRED_OK {
            match correctness
                .iter()
                .find(|(size, name, _, _)| *size == n && name == arm)
            {
                Some((_, _, true, _)) => (),
                Some((_, _, false, _)) => {
                    return Err(format!("correctness mismatch: n={n} arm={arm}"))
                }
                None => return Err(format!("correctness line missing: n={n} arm={arm}")),
            }
        }
        for (impl_id, criterion_arm) in RERUN_ARMS {
            if !times
                .iter()
                .any(|(size, arm, _)| *size == n && arm == criterion_arm)
            {
                return Err(format!(
                    "timing line missing for {impl_id}: n={n} arm={criterion_arm}"
                ));
            }
        }
    }
    let mut oracle_digests = BTreeMap::new();
    for (n, arm, ok, line) in &correctness {
        if *ok {
            let digest = sha256_text(line, &isolated.root.join("oracle-line.txt"))?;
            oracle_digests.insert((*n, arm.clone()), digest);
        }
    }
    // The bench's correctness arm for the parallel table is `apeiron_parallel`, while its criterion
    // arm is `apeiron_par16`; alias the digest so the promoted cell resolves.
    for n in BENCH_SIZES {
        if let Some(digest) = oracle_digests
            .get(&(n, "apeiron_parallel".to_string()))
            .cloned()
        {
            oracle_digests.insert((n, "apeiron_par16".to_string()), digest);
        }
    }
    Ok(Repro {
        root: isolated.root.display().to_string(),
        original_command: SPATIAL_CMD.to_string(),
        executed_command,
        apeiron_worktree: isolated.apeiron_worktree.display().to_string(),
        fold_worktree: isolated.fold_worktree.display().to_string(),
        apeiron_head: SPATIAL_REV.to_string(),
        fold_head: FOLD_REV.to_string(),
        wall_ms,
        output_sha256,
        correctness,
        oracle_digests,
        times,
    })
}

/// Prepare, reproduce both historical campaigns and always clean up every worktree.
fn run_reproduction(manifest_dir: &Path) -> Result<(Repro, SerialRepro), String> {
    let mut isolated = prepare_worktrees(manifest_dir)?;
    let outcome = reproduce(&isolated, manifest_dir)
        .and_then(|r| reproduce_serial(&isolated, manifest_dir).map(|s| (r, s)));
    let cleanup = isolated.cleanup();
    match (outcome, cleanup) {
        (Ok(receipts), Ok(())) => Ok(receipts),
        (Ok(_), Err(e)) => Err(format!("worktree cleanup failed: {e}")),
        (Err(e), _) => Err(e),
    }
}

/// Sum the `instructions/u` counts of a plain `perf stat` report, skipping an unscheduled PMU.
fn parse_counter(stderr: &str) -> u64 {
    stderr
        .lines()
        .filter(|l| l.contains("instructions/u") && !l.contains("not counted"))
        .filter_map(|l| {
            l.split_whitespace()
                .next()?
                .replace(',', "")
                .parse::<u64>()
                .ok()
        })
        .sum()
}

/// Run one counter arm `PERF_REPS` times, pinned to one core, and return its printed count, its
/// min/median/max instructions per unit and the exact printed line.
#[allow(
    clippy::cast_precision_loss,
    reason = "a bounded instruction total is far below 2^53; this is a measurement"
)]
fn run_counter(
    bin: &Path,
    var: &str,
    arm: &str,
    base: f64,
    root: &Path,
) -> Result<(u64, f64, f64, f64, String), String> {
    let mut vals = Vec::with_capacity(PERF_REPS as usize);
    let mut line = String::new();
    let mut hits = 0u64;
    for rep in 0..PERF_REPS {
        let out = root.join(format!("counter-{var}-{arm}-{rep}.out"));
        let err = root.join(format!("counter-{var}-{arm}-{rep}.err"));
        let mut command = Command::new("taskset");
        command
            .args(["-c", PERF_CPU, "perf", "stat", "-e", "instructions:u", "--"])
            .arg(bin)
            .env(var, arm);
        let (code, _) = run_capture(&mut command, Duration::from_secs(300), &out, &err)?;
        if code != 0 {
            return Err(format!("counter {var}={arm} exited {code}"));
        }
        line = fs::read_to_string(&out)
            .unwrap_or_default()
            .trim()
            .to_string();
        hits = line
            .split_whitespace()
            .nth(1)
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| format!("counter {var}={arm} printed {line:?} without a count"))?;
        let total = parse_counter(&fs::read_to_string(&err).unwrap_or_default());
        if total == 0 {
            return Err(format!("counter {var}={arm} reported no instructions"));
        }
        vals.push(total as f64 / base);
    }
    vals.sort_by(f64::total_cmp);
    Ok((
        hits,
        vals[0],
        vals[vals.len() / 2],
        vals[vals.len() - 1],
        line,
    ))
}

/// Run the `3a95bb5` serial campaign: build the two examples, run `map_sweep`, then the
/// instructions-per-lookup probe arms.
fn reproduce_serial(isolated: &Isolated, repo: &Path) -> Result<SerialRepro, String> {
    let start = Instant::now();
    let target = repo.join("target/reg1-history/serial-target");
    fs::create_dir_all(&target).map_err(|e| format!("create {}: {e}", target.display()))?;
    let root = &isolated.root;
    let cargo = |args: &[&str], out: &Path, err: &Path| {
        let mut command = Command::new("cargo");
        command
            .args(args)
            .current_dir(&isolated.serial_worktree)
            .env("CARGO_TARGET_DIR", &target)
            .env("CARGO_NET_OFFLINE", "true")
            .env("CARGO_TERM_COLOR", "never");
        run_capture(
            &mut command,
            Duration::from_secs(BENCH_TIMEOUT_SECS),
            out,
            err,
        )
    };
    let build_err = root.join("serial-build.err");
    let (code, _) = cargo(
        &[
            "build",
            "--release",
            "--offline",
            "--example",
            "map_sweep",
            "--example",
            "probe_cost",
        ],
        &root.join("serial-build.out"),
        &build_err,
    )?;
    if code != 0 {
        return Err(format!(
            "serial build exited {code}: {}",
            tail(&fs::read_to_string(&build_err).unwrap_or_default(), 800)
        ));
    }
    let map_out = root.join("map.out");
    let map_err = root.join("map.err");
    let (code, map_wall_ms) = cargo(
        &["run", "--release", "--offline", "--example", "map_sweep"],
        &map_out,
        &map_err,
    )?;
    if code != 0 {
        return Err(format!(
            "serial map_sweep exited {code}: {}",
            tail(&fs::read_to_string(&map_err).unwrap_or_default(), 800)
        ));
    }
    let map_output_sha256 = sha256_file(&map_out)?;
    let mut probe = Vec::with_capacity(PROBE_ARMS.len());
    for (impl_id, dimension, arm) in PROBE_ARMS {
        let (hits, lo, median, hi, line) = run_counter(
            &target.join("release/examples/probe_cost"),
            "LOOKUP_ARM",
            arm,
            PROBE_LOOKUPS_F,
            root,
        )?;
        probe.push(ProbeRow {
            impl_id: impl_id.to_string(),
            dimension: dimension.to_string(),
            arm: arm.to_string(),
            hits,
            lo,
            median,
            hi,
            line_digest: sha256_text(&line, &root.join("probe-line.txt"))?,
        });
    }
    Ok(SerialRepro {
        root: root.display().to_string(),
        worktree: isolated.serial_worktree.display().to_string(),
        head: SERIAL_REV.to_string(),
        map_output_sha256,
        map_wall_ms,
        wall_ms: start.elapsed().as_millis(),
        probe,
    })
}

/// Build the fail-closed history report. It never claims G1 or REG-1.
fn build_report(repro: &Repro, serial: &SerialRepro, cells: usize) -> Report {
    let axes = vec![
        (
            "history_matrix",
            AxisStatus::Pass,
            format!(
                "{cells} registered cells validate against the six mandatory families, the pinned \
                 revisions, the status vocabulary and the no-historical-as-fresh rule"
            ),
        ),
        (
            "historical_reproduction",
            AxisStatus::Pass,
            format!(
                "{} wall_ms={} output_sha256={} with exact oracle correctness on {} arms",
                repro.executed_command,
                repro.wall_ms,
                repro.output_sha256,
                REQUIRED_OK.len() * BENCH_SIZES.len()
            ),
        ),
        (
            "identity",
            AxisStatus::Missing,
            "full-scale 1e8 coordinate-colliding identity run unrun".to_string(),
        ),
        (
            "round_trip",
            AxisStatus::Boundary,
            "apeiron::Table exposes no encode/decode surface".to_string(),
        ),
        (
            "determinism",
            AxisStatus::Missing,
            "separate-process determinism is covered by --check correctness, not here".to_string(),
        ),
        (
            "growth_merge_refusal",
            AxisStatus::Boundary,
            "ShardFull and N=0 refusal observed; remove/growth are absent operations".to_string(),
        ),
        (
            "performance",
            AxisStatus::Missing,
            "historical direction reproduced; current G1 ratio cells remain missing".to_string(),
        ),
        (
            "coverage",
            AxisStatus::Missing,
            "Aeneas/Kani census is reported by --check coverage, not here".to_string(),
        ),
        (
            "historical_serial",
            AxisStatus::Pass,
            format!(
                "{SERIAL_MAP_CMD} wall_ms={} map_output_sha256={}; {} probe arms on one core, no \
                 hashtrinity comparison",
                serial.map_wall_ms,
                serial.map_output_sha256,
                serial.probe.len()
            ),
        ),
    ]
    .into_iter()
    .map(|(axis, status, evidence)| AxisRecord {
        axis,
        status,
        evidence,
    })
    .collect();
    Report {
        axes,
        report_complete: false,
        reg1_pass: false,
    }
}

/// Validate the printed report: it must stay fail-closed and name the G1 axes.
fn validate_report(report: &Report, repro: &Repro) -> Result<(), String> {
    if report.report_complete {
        return Err(
            "report_complete must be false; the history gate does not close G1".to_string(),
        );
    }
    if report.reg1_pass {
        return Err("reg1_pass must be false; the history gate does not close REG-1".to_string());
    }
    let names: Vec<&str> = report.axes.iter().map(|a| a.axis).collect();
    let expected = [
        "history_matrix",
        "historical_reproduction",
        "identity",
        "round_trip",
        "determinism",
        "growth_merge_refusal",
        "performance",
        "coverage",
        "historical_serial",
    ];
    if names != expected {
        return Err(format!("report axis schema mismatch: {names:?}"));
    }
    if report.axes.iter().any(|a| !crate::allowed(a.status)) {
        return Err("axis status outside the vocabulary".to_string());
    }
    if report.axes[1].status != AxisStatus::Pass {
        return Err("historical_reproduction must pass when the reproduction returned".to_string());
    }
    if report.axes[6].status != AxisStatus::Missing || report.axes[7].status != AxisStatus::Missing
    {
        return Err("performance and coverage must stay missing".to_string());
    }
    if repro.output_sha256.len() != 64 {
        return Err("reproduction output digest is not recorded".to_string());
    }
    Ok(())
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

fn print_matrix(matrix: &Matrix) {
    let mut by_family: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_status: BTreeMap<&str, usize> = BTreeMap::new();
    for cell in &matrix.cells {
        *by_family.entry(cell.family.as_str()).or_insert(0) += 1;
        *by_status.entry(cell.status.as_str()).or_insert(0) += 1;
    }
    let statuses = STATUSES
        .iter()
        .map(|status| format!("{status}={}", by_status.get(status).copied().unwrap_or(0)))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "history_matrix schema={} cells={} {statuses}",
        matrix.schema,
        matrix.cells.len()
    );
    for (family, count) in &by_family {
        println!("history_family={family} cells={count}");
    }
}

fn print_reproduction(repro: &Repro) {
    let median = |n: u32, arm: &str| -> Option<f64> {
        repro
            .times
            .iter()
            .find(|(size, name, _)| *size == n && name == arm)
            .map(|(_, _, v)| *v)
    };
    println!("history_root={}", repro.root);
    println!("history_original_command={}", repro.original_command);
    println!("history_executed_command={}", repro.executed_command);
    println!(
        "history_worktree apeiron={} head={}",
        repro.apeiron_worktree, repro.apeiron_head
    );
    println!(
        "history_worktree fold={} head={}",
        repro.fold_worktree, repro.fold_head
    );
    println!("history_output_sha256={}", repro.output_sha256);
    println!("history_wall_ms={}", repro.wall_ms);
    for (n, arm, ok, _) in &repro.correctness {
        println!(
            "history_correctness n={n} arm={arm} status={}",
            if *ok { "ok" } else { "wrong" }
        );
    }
    for n in BENCH_SIZES {
        for (impl_id, criterion_arm) in RERUN_ARMS {
            if let Some(median) = median(n, criterion_arm) {
                println!("history_measured n={n} impl_id={impl_id} median_ms={median:.4}");
                println!(
                    "history_rerun family=historical_spatial_parallel dimension=n{n} \
                     impl_id={impl_id} status=measured oracle_ok=true"
                );
                if let Some(digest) = repro.oracle_digests.get(&(n, criterion_arm.to_string())) {
                    println!(
                        "history_promote family=historical_spatial_parallel dimension=n{n} \
                         impl_id={impl_id} output_sha256={digest} result=\"{median:.4} ms\" \
                         wall_ms={}",
                        repro.wall_ms
                    );
                }
            }
        }
        if let (Some(selected), Some(baseline)) =
            (median(n, "apeiron_par16"), median(n, "hashtrinity_packed"))
        {
            let ratio = selected / baseline;
            println!(
                "history_ratio n={n} selected_over_hashtrinity={ratio:.4} direction={}",
                if selected < baseline {
                    "apeiron_faster"
                } else {
                    "apeiron_not_faster"
                }
            );
        }
    }
}

/// Print the `3a95bb5` serial campaign receipts. Nothing here is a hashtrinity comparison.
fn print_serial(serial: &SerialRepro) {
    println!("history_serial_root={}", serial.root);
    println!(
        "history_serial_worktree={} head={}",
        serial.worktree, serial.head
    );
    println!("history_serial_build_command={SERIAL_BUILD_CMD}");
    println!("history_serial_map_command={SERIAL_MAP_CMD}");
    println!(
        "history_serial_map_output_sha256={} wall_ms={}",
        serial.map_output_sha256, serial.map_wall_ms
    );
    println!("history_serial_wall_ms={}", serial.wall_ms);
    println!("history_serial_probe_command={SERIAL_PROBE_CMD}");
    for row in &serial.probe {
        let ok = row.hits == PROBE_LOOKUPS;
        println!(
            "history_serial_probe arm={} impl_id={} dimension={} hits={} expected={} \
             instr_per_lookup={:.3} interval=[{:.3},{:.3}] oracle_ok={} output_sha256={}",
            row.arm,
            row.impl_id,
            row.dimension,
            row.hits,
            PROBE_LOOKUPS,
            row.median,
            row.lo,
            row.hi,
            ok,
            row.line_digest
        );
        if ok {
            println!(
                "history_promote family=historical_serial_map dimension={} impl_id={} \
                 output_sha256={} result=\"{:.3} [{:.3},{:.3}]\" wall_ms={}",
                row.dimension,
                row.impl_id,
                row.line_digest,
                row.median,
                row.lo,
                row.hi,
                serial.wall_ms
            );
        }
    }
}

/// Run the history check, and (when requested) require every validator control rejected.
pub(crate) fn run_history(controls: bool) -> i32 {
    let matrix = match parse_matrix(MATRIX) {
        Ok(matrix) => matrix,
        Err(e) => {
            eprintln!("history matrix parse failed: {e}");
            return 1;
        }
    };
    if let Err(e) = validate_matrix(&matrix) {
        eprintln!("history matrix validation failed: {e}");
        return 1;
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let head = match git_rev_parse(&manifest_dir, "HEAD") {
        Ok(head) => head,
        Err(e) => {
            eprintln!("history HEAD resolution failed: {e}");
            return 1;
        }
    };
    if head != CURRENT_REV {
        eprintln!(
            "history matrix current pin {CURRENT_REV} != observed HEAD {head}; the matrix is stale"
        );
        return 1;
    }
    print_identity(WORKLOAD);
    print_matrix(&matrix);
    println!("history_matrix_pin current={CURRENT_REV} observed_head={head}");

    let (repro, serial) = match run_reproduction(&manifest_dir) {
        Ok(receipts) => receipts,
        Err(e) => {
            println!(
                "history_reproduction status=tool_error reason=\"{}\"",
                escape(&e)
            );
            eprintln!("history reproduction blocked: {e}");
            return 1;
        }
    };
    print_reproduction(&repro);
    print_serial(&serial);
    let report = build_report(&repro, &serial, matrix.cells.len());
    print_report(&report);
    if let Err(e) = validate_report(&report, &repro) {
        eprintln!("history report validation failed: {e}");
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

/// The cell the first four controls corrupt.
const CONTROL_CELL: &str = "hist-spatial/n1000000/apeiron_parallel";
/// The cell the deletion control removes.
const DELETED_CELL: &str = "current-occupancy/occupancy_8/apeiron_table";
/// The promoted spatial cell the promotion controls corrupt.
const PROMOTED_CELL: &str = "hist-spatial/n1000000/apeiron_parallel";

/// Spawn one control child and require it to be rejected (nonzero exit and a `rejected` marker).
fn run_control(name: &str) -> Result<i32, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let output = Command::new(exe)
        .args(["--history-control", name])
        .output()
        .map_err(|e| format!("spawn control {name}: {e}"))?;
    if output.status.success() {
        return Err(format!(
            "control {name} exited 0; corrupted matrix was accepted"
        ));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains("REJECTED") {
        return Err(format!(
            "control {name} failed for an unexpected reason: {}",
            stderr.trim()
        ));
    }
    Ok(output.status.code().unwrap_or(-1))
}

fn find_cell<'a>(m: &'a mut Matrix, id: &str) -> Option<&'a mut Cell> {
    m.cells.iter_mut().find(|c| c.id == id)
}

/// The control child: parse the real matrix, corrupt one property in memory, and require the same
/// validator to reject it.
pub(crate) fn run_control_child(name: &str) -> i32 {
    let mut matrix = match parse_matrix(MATRIX) {
        Ok(matrix) => matrix,
        Err(e) => {
            eprintln!("history-control matrix parse failed: {e}");
            return 1;
        }
    };
    let applied = match name {
        "wrong-oracle-status" => find_cell(&mut matrix, CONTROL_CELL).map(|cell| {
            cell.status = "measured".to_string();
            cell.historical_value = None;
            cell.fresh = Some(Fresh {
                command: SPATIAL_CMD.to_string(),
                revision: SPATIAL_REV.to_string(),
                output_sha256: "0".repeat(64),
                oracle_ok: false,
                result: "2.49 ms".to_string(),
                wall_ms: "1".to_string(),
            });
        }),
        "wrong-source-digest" => find_cell(&mut matrix, CONTROL_CELL)
            .map(|cell| cell.source_sha256 = Some("0".repeat(64))),
        "wrong-revision" => {
            find_cell(&mut matrix, CONTROL_CELL).map(|cell| cell.revision = "0".repeat(40))
        }
        "wrong-workload-identity" => find_cell(&mut matrix, CONTROL_CELL)
            .map(|cell| cell.workload = "unregistered-workload".to_string()),
        "required-cell-deleted" => {
            let before = matrix.cells.len();
            matrix.cells.retain(|c| c.id != DELETED_CELL);
            (matrix.cells.len() != before).then_some(())
        }
        "promoted-oracle-flipped" => find_cell(&mut matrix, PROMOTED_CELL)
            .and_then(|cell| cell.fresh.as_mut())
            .map(|fresh| fresh.oracle_ok = false),
        "promoted-digest-cleared" => find_cell(&mut matrix, PROMOTED_CELL)
            .and_then(|cell| cell.fresh.as_mut())
            .map(|fresh| fresh.output_sha256 = "not-a-digest".to_string()),
        _ => {
            eprintln!("unknown history control: {name}");
            return 2;
        }
    };
    if applied.is_none() {
        eprintln!("history-control {name} could not be applied to the matrix");
        return 1;
    }
    match validate_matrix(&matrix) {
        Ok(()) => {
            eprintln!("history-control {name} NOT accepted: validator did not reject");
            1
        }
        Err(e) => {
            eprintln!("history-control {name} REJECTED: {e}");
            2
        }
    }
}
