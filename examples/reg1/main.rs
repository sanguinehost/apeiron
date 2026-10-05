//! REG-1 gate: a fail-closed aggregate comparison of the current `apeiron::Table` against the
//! pinned historical `hashtrinity::Table`, with an independent `std::collections::HashMap` oracle.
//!
//! Several checks share this binary:
//!
//! - `--check bootstrap`: the W1 witness. It exits 0 when the small witness is correct and the
//!   report is fail-closed (`report_complete=false`, `reg1_pass=false`).
//! - `--check correctness --profile smoke`: the W2 smoke. Actual Apeiron, pinned hashtrinity and
//!   the oracle run on deterministic preregistered workloads, with fail-closed validators and
//!   validator controls.
//! - `--check performance --profile smoke|full [--accept] [--json]`: the W3 measurement. Both real
//!   tables run one deterministic shared fixed-16 schedule; the smoke profile is the focused gate
//!   and the full profile is the exact preregistered counts. Before the first sample the run takes
//!   an exclusive `flock` on `/tmp/apeiron/reg1.lock` (override with `REG1_MEASURE_LOCK`, queue
//!   behind a held lock with `REG1_MEASURE_LOCK_WAIT_S`, default `0` = refuse immediately) and
//!   reads the quiet-box instruments; a contended box **refuses** with
//!   `measurement_valid=false reason=<lock_contended|load1_above_limit|contending_processes|
//!   host_busy_above_limit|precondition_error>` on stdout and exit 3. A refused run prints no pass
//!   tally, no HEADLINE and no `strict_acceptance` line: it is invalid for measurement, not a
//!   verdict. See `measure_guard` for the limits and their overrides. The default human output is a
//!   fixed-width `(cell, incumbent)` pair table with the selected and incumbent worker counts, a
//!   `concurrency` column (`matched`/`MISMATCH`), an `address_face` column and a single HEADLINE;
//!   `--json` restores the machine-line dump. `--accept` is the strict acceptance mode: it applies
//!   the registered per-pair rule `selected/incumbent <= 1.0` on every applicable `(cell, incumbent)`
//!   pair, rejects a pair whose two sides ran at different worker widths (`concurrency=MISMATCH`),
//!   and the process exits nonzero when any applicable pair fails, any registered load-bearing
//!   adapter is missing, or any `not-applicable` is unauthorized. Without `--accept` the same run is
//!   diagnostic and reports the axis `boundary`. `--accept --profile smoke` applies the same rule on
//!   the reduced schedule but cannot close G1.
//!
//!   Every `--accept` run also prints the averaged block, by default and with no flag: the pass
//!   tally's mean, minimum, maximum, spread and standard deviation over the profile's campaigns, the
//!   same for each incumbent's ratios, and each decided `(cell, incumbent)` pair's ratio mean,
//!   interval and spread (`acceptance_repeats`, `acceptance_repeats_campaign`,
//!   `acceptance_repeats_incumbent`, `acceptance_cell_ratio`). The gate is unchanged: it remains the
//!   pooled campaign interval max, and the block's `rule=` field says so. The repetition axis is the
//!   campaign axis the profile already runs, so the block costs no additional campaign and no
//!   additional wall-clock time. The reported `spread_bound` is never narrower than the registered
//!   cross-invocation pass-count spread, so a reader does not need a second invocation to know that
//!   a movement is inside the noise already measured on unchanged source.
//! - `--check coverage`: the v2 census. It parses and validates the operation/evidence registry in
//!   `coverage.json`, reports real-source extraction, bridge refinement and executed Rust tests at
//!   their separate scopes, prints exclusions and freshness, and requires rejecting controls.
//! - `--check history --controls`: the W1 performance-restoration framework. It validates the
//!   registered comparison matrix in `matrix.json` and reproduces the original `71942db` spatial
//!   benchmark in isolated revision-pinned child worktrees, with exact source digests, oracle
//!   correctness before timing, and five rejecting matrix controls. Exit 0 means the framework is
//!   valid; the matrix's unrun cells stay `missing`, the historical rows stay `historical` until
//!   rerun, and G1 stays open.
//!
//! No check claims REG-1: the 10^8 identity run, separate-process determinism, the round-trip
//! codec, growth/remove and coverage remain unrun or unsupported and are reported as such. The
//! performance ratio is a measurement, not a pass of the identity or growth rows. The coverage
//! fractions count existing evidence at its stated scope; freshness is reported separately for
//! extraction, refinement and executed Rust tests. REG-1/G1 stays OPEN.

mod acceptance;
mod attrib;
mod correctness;
mod coverage;
mod history;
mod matrix;
mod measure_guard;
mod performance;
mod strata;
mod workload;

use std::collections::BTreeMap;
use std::process::Command;

use crate::workload::KEY_COUNT;

// ── Dependency-free JSON reader shared by the history matrix validator ────────────────────────
//
// The reader covers the JSON subset `matrix.json` uses: strings, arrays and objects. A
// malformed document is an error, never a default.

#[derive(Debug, Clone)]
pub(crate) enum Json {
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn as_str(&self) -> Result<&str, String> {
        match self {
            Json::Str(s) => Ok(s),
            _ => Err("expected string".to_string()),
        }
    }

    pub(crate) fn as_obj(&self) -> Result<&[(String, Json)], String> {
        match self {
            Json::Obj(v) => Ok(v),
            _ => Err("expected object".to_string()),
        }
    }
}

/// A minimal recursive-descent reader for the JSON subset the matrix uses. A malformed document is
/// an error, never a default.
pub(crate) struct Reader<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(text: &'a str) -> Self {
        Self {
            chars: text.chars().peekable(),
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.chars.peek(), Some(c) if c.is_whitespace()) {
            self.chars.next();
        }
    }

    pub(crate) fn parse(mut self) -> Result<Json, String> {
        self.skip_ws();
        let value = self.value()?;
        self.skip_ws();
        if self.chars.peek().is_some() {
            return Err("trailing characters after the top-level value".to_string());
        }
        Ok(value)
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.chars.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some(other) => Err(format!("unexpected character {other}")),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.chars.next();
        let mut fields = Vec::new();
        self.skip_ws();
        if self.chars.peek() == Some(&'}') {
            self.chars.next();
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            if self.chars.next() != Some(':') {
                return Err("expected ':' after an object key".to_string());
            }
            let value = self.value()?;
            fields.push((key, value));
            self.skip_ws();
            match self.chars.next() {
                Some(',') => (),
                Some('}') => break,
                _ => return Err("expected ',' or '}' in object".to_string()),
            }
        }
        Ok(Json::Obj(fields))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.chars.next();
        let mut items = Vec::new();
        self.skip_ws();
        if self.chars.peek() == Some(&']') {
            self.chars.next();
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.chars.next() {
                Some(',') => (),
                Some(']') => break,
                _ => return Err("expected ',' or ']' in array".to_string()),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, String> {
        if self.chars.next() != Some('"') {
            return Err("expected a string".to_string());
        }
        let mut out = String::new();
        loop {
            match self.chars.next() {
                None => return Err("unterminated string".to_string()),
                Some('"') => break,
                Some('\\') => match self.chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('b') => out.push('\u{0008}'),
                    Some('f') => out.push('\u{000C}'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let digit = self.chars.next().ok_or("short unicode escape")?;
                            let value = digit.to_digit(16).ok_or("bad unicode escape")?;
                            code = code * 16 + value;
                        }
                        out.push(char::from_u32(code).ok_or("bad code point")?);
                    }
                    _ => return Err("unknown escape".to_string()),
                },
                Some(c) => out.push(c),
            }
        }
        Ok(out)
    }
}

pub(crate) fn field<'a>(obj: &'a [(String, Json)], key: &str) -> Result<&'a Json, String> {
    field_opt(obj, key).ok_or_else(|| format!("missing field {key}"))
}

pub(crate) fn field_opt<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

pub(crate) fn str_field(obj: &[(String, Json)], key: &str) -> Result<String, String> {
    field(obj, key)?
        .as_str()
        .map(str::to_string)
        .map_err(|e| format!("field {key}: {e}"))
}

pub(crate) fn opt_str_field(obj: &[(String, Json)], key: &str) -> Result<Option<String>, String> {
    match field_opt(obj, key) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .map(|s| Some(s.to_string()))
            .map_err(|e| format!("field {key}: {e}")),
    }
}

pub(crate) fn string_map(
    obj: &[(String, Json)],
    key: &str,
) -> Result<BTreeMap<String, String>, String> {
    let entries = field(obj, key)?
        .as_obj()
        .map_err(|e| format!("field {key}: {e}"))?;
    let mut out = BTreeMap::new();
    for (name, value) in entries {
        out.insert(
            name.clone(),
            value
                .as_str()
                .map_err(|e| format!("field {key}.{name}: {e}"))?
                .to_string(),
        );
    }
    Ok(out)
}

/// The revision the Git dependency must resolve to. Bound to `Cargo.lock`, not to a memory.
const PINNED_HASHTRINITY_REV: &str = "4cc14471ae24f3f204fb0b0e91736ef75c909358";
/// The committed lockfile, embedded so the baseline check reads the resolved identity, not `HEAD`.
const LOCK: &str = include_str!("../../Cargo.lock");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(
    dead_code,
    reason = "the full status vocabulary is part of the fail-closed report schema"
)]
pub(crate) enum AxisStatus {
    Pass,
    Fail,
    Boundary,
    Missing,
    ToolError,
}

impl AxisStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            AxisStatus::Pass => "pass",
            AxisStatus::Fail => "fail",
            AxisStatus::Boundary => "boundary",
            AxisStatus::Missing => "missing",
            AxisStatus::ToolError => "tool_error",
        }
    }
}

pub(crate) struct AxisRecord {
    pub(crate) axis: &'static str,
    pub(crate) status: AxisStatus,
    pub(crate) evidence: String,
}

#[allow(
    clippy::struct_field_names,
    reason = "the field names are the printed report keys"
)]
pub(crate) struct Report {
    pub(crate) axes: Vec<AxisRecord>,
    pub(crate) report_complete: bool,
    pub(crate) reg1_pass: bool,
}

/// The allowed status vocabulary.
pub(crate) fn allowed(s: AxisStatus) -> bool {
    matches!(
        s,
        AxisStatus::Pass
            | AxisStatus::Fail
            | AxisStatus::Boundary
            | AxisStatus::Missing
            | AxisStatus::ToolError
    )
}

/// Extract the resolved hashtrinity revision from the embedded `Cargo.lock`.
fn lock_hashtrinity_rev() -> Result<String, String> {
    let mut in_block = false;
    for line in LOCK.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            in_block = false;
        } else if line == "name = \"hashtrinity\"" {
            in_block = true;
        } else if in_block && line.starts_with("source = ") {
            if let Some(rest) = line.split("rev=").nth(1) {
                let rev = rest.split(['#', '"']).next().unwrap_or("");
                if !rev.is_empty() {
                    return Ok(rev.to_string());
                }
            }
        }
    }
    Err("Cargo.lock has no hashtrinity git source".to_string())
}

/// The one baseline-identity validator. It fails closed on any mismatch between the expected
/// revision and the revision the committed lockfile resolves.
fn baseline_identity(expected_rev: &str) -> Result<(), String> {
    let observed = lock_hashtrinity_rev()?;
    if observed == expected_rev {
        Ok(())
    } else {
        Err(format!(
            "baseline revision mismatch: expected {expected_rev}, Cargo.lock resolves {observed}"
        ))
    }
}

fn tool_version(tool: &str) -> String {
    Command::new(tool)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unavailable".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

fn git_head() -> (String, bool) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let head = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        );
    let dirty = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .is_some_and(|o| !o.stdout.is_empty());
    (head, dirty)
}

pub(crate) fn escape(s: &str) -> String {
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

pub(crate) fn print_identity(workload: &str) {
    let (head, dirty) = git_head();
    println!(
        "apeiron_head={head} dirty={}",
        if dirty { "yes" } else { "no" }
    );
    println!("hashtrinity_rev={PINNED_HASHTRINITY_REV}");
    println!("rustc={}", tool_version("rustc"));
    println!("cargo={}", tool_version("cargo"));
    println!(
        "target={}-{} profile={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!("workload={workload}");
}

pub(crate) fn print_report(r: &Report) {
    for a in &r.axes {
        println!(
            "axis={} status={} evidence=\"{}\"",
            a.axis,
            a.status.as_str(),
            escape(&a.evidence)
        );
    }
    println!("report_complete={}", r.report_complete);
    println!("reg1_pass={}", r.reg1_pass);
}

/// Negative control: re-invoke this same binary with an intentionally wrong expected revision and
/// require the baseline-identity validator to reject it. No source mutation is needed.
fn run_baseline_control() -> Result<i32, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let wrong = "0000000000000000000000000000000000000000";
    let output = Command::new(exe)
        .args(["--baseline-rev-check", wrong])
        .output()
        .map_err(|e| format!("spawn control: {e}"))?;
    if output.status.success() {
        return Err("control accepted an incorrect expected revision".to_string());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains("mismatch") {
        return Err(format!(
            "control rejected for an unexpected reason: {}",
            stderr.trim()
        ));
    }
    Ok(output.status.code().unwrap_or(-1))
}

/// Run the W1 bootstrap check and preserve its output semantics.
fn run_bootstrap(controls: bool) -> i32 {
    let witness = workload::run_bootstrap_witness();
    let report = workload::build_bootstrap_report(&witness);
    print_identity(&format!(
        "reg1-bootstrap-{KEY_COUNT}-constant-coordinate-keys"
    ));
    print_report(&report);

    if let Err(e) = workload::validate_bootstrap(&witness, &report) {
        eprintln!("bootstrap validation failed: {e}");
        return 1;
    }
    if let Err(e) = baseline_identity(PINNED_HASHTRINITY_REV) {
        eprintln!("baseline-identity check failed: {e}");
        return 1;
    }
    println!(
        "ht_coalesced_value={} oracle_len={} apeiron_admitted={} apeiron_refused={}",
        witness.identity.ht_value,
        witness.identity.oracle_len,
        witness.identity.apeiron_admitted,
        witness.identity.apeiron_refused
    );

    if controls {
        match run_baseline_control() {
            Ok(code) => {
                println!("control=baseline-revision-rejection observed=red child_exit={code}");
            }
            Err(e) => {
                eprintln!("control failed: {e}");
                return 1;
            }
        }
    }
    0
}

#[allow(
    clippy::too_many_lines,
    reason = "one function states the CLI dispatch table"
)]
fn run(args: &[String]) -> i32 {
    if let Some(pos) = args.iter().position(|a| a == "--baseline-rev-check") {
        let Some(rev) = args.get(pos + 1) else {
            eprintln!("--baseline-rev-check requires a revision");
            return 2;
        };
        return match baseline_identity(rev) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("baseline-identity check rejected: {e}");
                2
            }
        };
    }

    if args.iter().any(|a| a == "--determinism-child") {
        println!("{}", workload::determinism_child_output());
        return 0;
    }

    if let Some(pos) = args.iter().position(|a| a == "--correctness-control") {
        let Some(name) = args.get(pos + 1) else {
            eprintln!("--correctness-control requires a name");
            return 2;
        };
        return correctness::run_control_child(name);
    }

    if let Some(pos) = args.iter().position(|a| a == "--performance-control") {
        let Some(name) = args.get(pos + 1) else {
            eprintln!("--performance-control requires a name");
            return 2;
        };
        let profile = parse_profile(args);
        let Some(profile) = profile else {
            eprintln!("--performance-control requires --profile smoke|full");
            return 2;
        };
        if acceptance::is_control(name) {
            return acceptance::run_control_child(name);
        }
        return performance::run_control_child(name, profile);
    }

    if let Some(pos) = args.iter().position(|a| a == "--coverage-control") {
        let Some(name) = args.get(pos + 1) else {
            eprintln!("--coverage-control requires a name");
            return 2;
        };
        return coverage::run_control_child(name);
    }

    if let Some(pos) = args.iter().position(|a| a == "--history-control") {
        let Some(name) = args.get(pos + 1) else {
            eprintln!("--history-control requires a name");
            return 2;
        };
        return history::run_control_child(name);
    }

    let check = args.windows(2).find_map(|w| {
        if w[0] == "--check" {
            Some(w[1].as_str())
        } else {
            None
        }
    });
    let controls = args.iter().any(|a| a == "--controls");
    let accept = args.iter().any(|a| a == "--accept");
    let json = args.iter().any(|a| a == "--json");
    match check {
        Some("bootstrap") => run_bootstrap(controls),
        Some("correctness") => {
            let profile = args.windows(2).find_map(|w| {
                if w[0] == "--profile" {
                    Some(w[1].as_str())
                } else {
                    None
                }
            });
            if profile != Some("smoke") {
                eprintln!("usage: reg1 --check correctness --profile smoke [--controls]");
                return 64;
            }
            correctness::run_correctness(controls)
        }
        Some("performance") => {
            let Some(profile) = parse_profile(args) else {
                eprintln!(
                    "usage: reg1 --check performance --profile smoke|full [--accept] [--controls] \
                     [--json]"
                );
                return 64;
            };
            // The measurement precondition. A timing claim on a contended box is a different
            // quantity, so the run refuses by name instead of reporting; see
            // `measure_guard`. The lock is held for the whole measurement and released on exit.
            let guard = match measure_guard::acquire(profile, accept) {
                Ok(guard) => guard,
                Err(refusal) => return refusal.report(),
            };
            let code = if accept {
                acceptance::run_accept(profile, controls, json)
            } else {
                let output = if json {
                    performance::Output::Json
                } else {
                    performance::Output::Table
                };
                performance::run_performance(profile, controls, false, output)
            };
            guard.release();
            code
        }
        Some("absent-attrib") => attrib::run(args),
        Some("coverage") => coverage::run_coverage(controls),
        Some("history") => history::run_history(controls),
        _ => {
            eprintln!(
                "usage: reg1 --check bootstrap [--controls] | --check correctness --profile smoke \
                 [--controls] | --check performance --profile smoke|full [--accept] [--controls] \
                 [--json] | --check absent-attrib --cell <id> --arm apeiron|ht [--reps N] \
                 [--window CTL:ACK] [--window-group SPEC] | --check coverage [--controls] | --check history [--controls]"
            );
            64
        }
    }
}

/// Parse the `--profile` argument into a performance profile.
fn parse_profile(args: &[String]) -> Option<performance::Profile> {
    let profile = args.windows(2).find_map(|w| {
        if w[0] == "--profile" {
            Some(w[1].as_str())
        } else {
            None
        }
    });
    match profile {
        Some("smoke") => Some(performance::Profile::Smoke),
        Some("full") => Some(performance::Profile::Full),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&args));
}
