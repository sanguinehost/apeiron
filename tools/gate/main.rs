//! `tools/gate` adapter: the thin IO boundary around the pure decisions in `core.rs`.
//!
//! This file owns filesystem, process and environment access; `core.rs` owns the pure decision, so
//! the checking bodies stay extractable through Charon/Aeneas for refinement. The adapter boundary
//! is the trust boundary: it reads the environment and reports what it found, and it decides nothing
//! about admission.
//!
//! `gate hold` is the hold preflight required by the owner-layer authority
//! (`~/Workspace/AGENTS.md`, "Authority, hold, momentum"): it walks up from the current directory
//! looking for a `HOLD` marker, prints every marker it finds, and exits non-zero while any is in
//! force. Run it before any write or dispatch. Absence of a marker is release. It is deliberately
//! not a silent success: a hold in force prints its own reason and scope.
//!
//! `gate docs check` loads the artefact registry in `docs/contracts/artifacts.toml` through
//! `adapter.rs` and reports what the pure core decides about it: the rows it loaded, a repeated id,
//! and a recorded `sha256` that no longer matches the file on disk. The subcommand decides nothing
//! itself; every verdict is a `core::Refusal`.
//!
//! `gate run <row-id>` executes one registry row. It resolves the row, refuses it through
//! `core::admit` unless the recorded `sha256` is the identity on disk *right now*, and only then
//! runs the row's `gate` command through `adapter::run_captured_with_tail` under that row's
//! `budget_s`, from the repository root the manifest resolves to. It reads no `control` column: a
//! control mutates the real source and belongs to the task that owns it. The child streams are
//! relayed by the adapter as it runs, so `gate run` adds a verdict block and, on a refusal, the
//! tail of what the child printed; the verdict block is `key=value` lines in a fixed order with
//! every value quoted, so a receipt parses it without a JSON reader.
//!
//! The row's green marker is not a registry column. It is read from the frozen record
//! `tools/gate/conformance.json`, which keys `gates[].green_marker` by each gate row's `path`
//! (`docs/epics/gate-driver.md:20-23`: the driver must reproduce "the same green markers"). A row
//! the record carries no marker for is reported as `marker_state=unrecorded` and judged on its
//! exit status alone; the verdict says so rather than implying a marker check that did not happen.
//!
//! **`gate docs check --controls` — the aggregate.** `--controls` keeps the identity check first
//! and unchanged, and then runs each selected row's recorded `control` command **in a throwaway
//! copy** of the repository (`adapter::ControlCopy`), requiring two things of it and reporting both
//! by name:
//!
//! 1. **the control fired** — the command exited 0 inside the row's own `budget_s`. Per
//!    `docs/contracts/artifacts.toml`, a `control` "is a command that must itself require a
//!    refusal: the `*_gate.py` controls mutate the real source, require the proof to go red with an
//!    attributable diagnostic, restore it, and exit 0 only if all three happened". The red's
//!    attribution is therefore that command's own contract, and **this aggregate does not check it**:
//!    the registry declares no control marker, so inventing one here would be a second spelling of a
//!    column that does not exist. What the aggregate checks for itself is the restoration:
//! 2. **the copy carries the registered identity** — the row's recorded `sha256` is re-read from
//!    the copy *before* the control runs (a copy that does not carry it runs no control at all) and
//!    *again after*, and a control that exited 0 having left the identity moved is refused. The
//!    restoration is observed here rather than taken on the command's word.
//!
//! Exit 0 requires **every selected row** to have been attempted and every attempted row's control
//! to have fired. Anything else is a non-zero exit and is named:
//!
//! | exit | meaning |
//! |---|---|
//! | 0 | every selected row attempted, every control fired, every identity restored |
//! | 1 | a control ran and did not fire (non-zero or absent exit, or the identity left moved) |
//! | 2 | the registry, the lock, or the throwaway copy could not be used |
//! | 3 | **incomplete** — a row was blocked, or the invocation budget ran out before a row was reached |
//!
//! A row that cannot run, or that overruns its budget, is `blocked` and is never counted as success,
//! so a batch over a registry whose controls are hours of Lean reports how far it got and exits 3.
//! `gate docs check` without `--controls` runs no control and its output is byte for byte what it
//! was.
//!
//! **Batching.** `--budget-s` (default `adapter::MAX_BUDGET_S` = 600, admissible range `1..=600`)
//! bounds the aggregate, and its clock starts **before** the throwaway copy: the copy, the
//! per-row digests and the children are all paid out of one budget, because starting the clock after
//! the copy would be the budget reset `AGENTS.md` §10 forbids. A row is attempted only when the
//! remaining budget covers that row's own `budget_s`, so every attempted child gets its full
//! declared budget and the invocation overruns by at most one `POLL_MS` per attempted row; rows past
//! that point are `unattempted`, never assumed.
//!
//! **The default is the ceiling, because the registry's rows are.** A row declaring `budget_s = 540`
//! needs 540 s of the *remaining* budget and the setup has already spent part of it, so a default
//! below the largest declared row budget makes that row unattempted by construction: at 540 s the
//! `tools/gate` row's own control was never attempted, and the aggregate reported
//! `unattempted:invocation-budget` on a row whose control had never been asked whether it fires.
//! The headroom `MAX_BUDGET_S − row.budget_s` is what pays the copy — 60 s against a copy of this
//! tree measured at 188 ms warm and 43 s after a reboot. A copy slower than its headroom is
//! reported honestly: the row is `unattempted` by name, the aggregate exits 3, and the reason says
//! how much the copy spent.
//!
//! **Trust boundary.** `--copy-root` defaults to `std::env::temp_dir()`, so a copy lands on the
//! `/tmp` tmpfs unless `TMPDIR` says otherwise; the failure register puts disposable copies off the
//! tmpfs, so the aggregate prints that advice on the invocation that used the default and
//! `--copy-root /home/socol/oc-tmp` is the way to take it. One copy per invocation serves every
//! selected row — a copy per row would multiply the cost by the batch — and the copy is removed
//! when the aggregate drops it. A copy excludes what a directory digest excludes (dot components
//! and `target`), so a row whose control needs an excluded build product is reported
//! `blocked:unrunnable` by name rather than run against a tree missing what it needs. The
//! `*_gate.py` rows resolve their toolchain with
//! `TOOLS = ROOT if (ROOT/'target/aeneas/aeneas').is_file() else Path('/home/socol/Workspace/apeiron')`,
//! so a control run from a source-only copy reads its Aeneas tools from the **live** tree: their
//! reads are outside the copy and that is stated here rather than claimed confined.

mod adapter;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use adapter::Row;
use gate::{validate, Digest, Manifest, Refusal, Registered, Submitted};

/// Where the registry lives when no manifest path is given. A document a gate parses belongs in
/// `docs/contracts/` (`AGENTS.md` §2), so this is not `docs/artifacts.toml`.
const DEFAULT_MANIFEST: &str = "docs/contracts/artifacts.toml";

/// Levels above the current directory to inspect. Bounded so a stray marker at the filesystem root
/// is not reached by accident.
const MAX_LEVELS: usize = 8;

/// How long `docs check` waits for another check over the same tree to finish before reporting the
/// lock as a blocker. Well under a gate budget: a check that cannot start is a blocker, not a wait.
const LOCK_WAIT_S: u64 = 30;

/// The registry lock. Off the repository so a check never writes into the tree it is hashing, and
/// beside the lane locks the `proof/resource` gates already take.
fn lock_path() -> PathBuf {
    PathBuf::from("/tmp/opencode/gate-docs-check.lock")
}

/// Every `HOLD` marker at or above `start`, nearest first.
fn holds_from(start: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut current = Some(start);
    let mut level = 0usize;
    while let Some(dir) = current {
        if level > MAX_LEVELS {
            break;
        }
        let marker = dir.join("HOLD");
        if marker.is_file() {
            found.push(marker);
        }
        current = dir.parent();
        level += 1;
    }
    found
}

/// The `hold` preflight. Exit 0 when no marker is in force; exit 1 and print each marker's contents
/// while one is; exit 2 when the environment cannot be read.
fn hold() -> ExitCode {
    let cwd = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(error) => {
            eprintln!("gate hold: cannot read the current directory: {error}");
            return ExitCode::from(2);
        }
    };

    let markers = holds_from(&cwd);
    if markers.is_empty() {
        println!("gate hold: no hold in force");
        return ExitCode::SUCCESS;
    }

    for marker in &markers {
        println!("gate hold: HOLD in force at {}", marker.display());
        match std::fs::read_to_string(marker) {
            Ok(text) => print!("{text}"),
            Err(error) => println!("  (unreadable: {error})"),
        }
        println!();
    }
    ExitCode::from(1)
}

/// The repository root above `manifest_path`: `docs/contracts/artifacts.toml` resolves its rows
/// against the directory three levels up, so the root is derived from the manifest rather than
/// guessed from the working directory.
///
/// `base` is the directory a relative manifest path is resolved against, and the result is always
/// absolute. Three `parent()` steps on the *relative* `docs/contracts/artifacts.toml` overshoot the
/// front of the path and return `""`: `docs check` survives that only because `root.join(path)` then
/// yields a cwd-relative path, so it silently depends on the process cwd, while `gate run` hands
/// the root to `Command::current_dir` and `""` fails to spawn. A root that is only sometimes a
/// directory is the defect; this returns one that always is, and a manifest too shallow to have
/// three parents falls back to `base` rather than to an empty path.
fn root_for(manifest_path: &Path, base: &Path) -> PathBuf {
    let resolved = if manifest_path.is_absolute() {
        manifest_path.to_path_buf()
    } else {
        base.join(manifest_path)
    };
    resolved
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map_or_else(|| base.to_path_buf(), Path::to_path_buf)
}

/// The process's own directory, the base a relative manifest path resolves against. A caller that
/// cannot read it has no root to spawn in.
fn current_dir() -> Result<PathBuf, String> {
    std::env::current_dir().map_err(|error| error.to_string())
}

/// The core's view of a parsed registry. `refs` is empty because the registry schema declares no
/// dependency column; the columns it does declare are the payload identity, which the core checks.
fn to_manifest(rows: &[Row]) -> Manifest {
    Manifest {
        artifacts: rows
            .iter()
            .map(|row| Registered {
                id: row.id.clone(),
                digest: Digest(format!("sha256:{}", row.sha256)),
                refs: Vec::new(),
            })
            .collect(),
    }
}

/// What `docs check` found, split by the cause so a report cannot merge them: a repeated id and a
/// drifted digest are different failures with different fixes.
#[derive(Debug, Default, PartialEq, Eq)]
struct Report {
    loaded: usize,
    duplicates: Vec<Refusal>,
    mismatches: Vec<Refusal>,
    /// Paths the adapter could not hash. Not a core verdict: there is no identity to compare.
    unreadable: Vec<String>,
}

impl Report {
    /// Clean when nothing was refused. A load of zero rows is *not* clean: an empty registry is a
    /// registry that has lost its records, and reporting it green would be the quiet success the
    /// failure register records.
    fn accepted(&self) -> bool {
        self.loaded > 0
            && self.duplicates.is_empty()
            && self.mismatches.is_empty()
            && self.unreadable.is_empty()
    }
}

/// Check every row against the tree at `root`, through the adapter, and let `core` decide. Does no
/// IO beyond hashing the registered paths: it runs no gate and reads no marker, so it says whether
/// the registry still describes the tree, not whether a gate passed.
fn check(root: &Path, rows: &[Row]) -> Report {
    let manifest = to_manifest(rows);
    let mut report = Report {
        loaded: rows.len(),
        ..Report::default()
    };
    report.duplicates = validate(&manifest);
    if !report.duplicates.is_empty() {
        // The core refuses everything while an id is registered twice, so no identity in the
        // manifest is authoritative to compare against.
        return report;
    }

    for row in rows {
        let path = root.join(&row.path);
        match adapter::sha256_path(&path, root) {
            Ok(hex) => {
                let submitted = Submitted {
                    id: row.id.clone(),
                    digest: Digest(format!("sha256:{hex}")),
                    refs: Vec::new(),
                };
                report.mismatches.extend(gate::admit(&manifest, &submitted));
            }
            Err(error) => report
                .unreadable
                .push(unreadable_text(&row.path, &error.to_string())),
        }
    }
    report
}

/// A path the adapter could not hash, as a report line naming the row and the reason. Not a core
/// verdict: there is no identity on disk to compare against, and reporting a row as drifted would
/// name an identity that does not exist. The line carries the row path and the OS reason so a
/// refusal says which row and why.
fn unreadable_text(path: &str, error: &str) -> String {
    format!("unreadable: {path}: {error}")
}

/// One refusal as the report prints it. Every variant names its id or its pair of ids, so a
/// diagnostic can be attributed to a row without re-reading the manifest.
fn refusal_text(refusal: &Refusal) -> String {
    match refusal {
        Refusal::DuplicateRegistration(id) => format!("  duplicate id: {id}"),
        Refusal::IdentityMismatch {
            id,
            expected,
            actual,
        } => format!("  digest mismatch: {id} recorded {expected}, on disk {actual}"),
        Refusal::MissingDependency { from, to } => {
            format!("  missing dependency: {from} requires {to}")
        }
        Refusal::Unregistered(id) => format!("  unregistered: {id}"),
        Refusal::DanglingReference { from, to } => format!("  dangling reference: {from} -> {to}"),
    }
}

fn print_refusal(refusal: &Refusal) {
    println!("{}", refusal_text(refusal));
}

/// `gate docs check [manifest]`. Exit 0 when every loaded row still names the file it names; exit 1
/// on a refusal; exit 2 when the registry cannot be read, parsed or locked.
///
/// The lock serialises concurrent checks over one tree: two checks hashing the same files while a
/// third edits them would each report a digest the other never saw. A lock still held after
/// [`LOCK_WAIT_S`] is reported as a blocker, never broken.
///
/// `options.controls` off is the whole of the pre-`--controls` behaviour: the identity check, its
/// report and its two exits, unchanged and in that order. With `--controls` the identity check
/// still runs first and still decides — a refused identity runs no control, because a control
/// exercised against a tree the registry no longer describes would redden for the wrong reason —
/// and the aggregate takes over only after the identity check has accepted.
fn docs_check(manifest_path: &Path, options: &CheckOptions) -> ExitCode {
    let path = lock_path();
    let lock = match adapter::Lock::acquire(&path, LOCK_WAIT_S) {
        Ok(lock) => lock,
        Err(error) => {
            eprintln!("gate docs check: {}: {error}", path.display());
            return ExitCode::from(2);
        }
    };

    let rows = match adapter::load_manifest(manifest_path) {
        Ok(rows) => rows,
        Err(error) => {
            eprintln!("gate docs check: {error}");
            return ExitCode::from(2);
        }
    };

    let base = match current_dir() {
        Ok(base) => base,
        Err(error) => {
            eprintln!("gate docs check: cannot read the current directory: {error}");
            return ExitCode::from(2);
        }
    };
    let root = root_for(manifest_path, &base);
    let report = check(&root, &rows);
    println!(
        "gate docs check: {} rows loaded; {} duplicate-id; {} digest-mismatch; {} unreadable",
        report.loaded,
        report.duplicates.len(),
        report.mismatches.len(),
        report.unreadable.len()
    );
    for refusal in report.duplicates.iter().chain(&report.mismatches) {
        print_refusal(refusal);
    }
    for path in &report.unreadable {
        println!("  {path}");
    }
    if !report.accepted() {
        drop(lock);
        return ExitCode::from(1);
    }
    if !options.controls {
        drop(lock);
        return ExitCode::SUCCESS;
    }
    let code = run_controls(&root, &rows, options);
    drop(lock);
    code
}

// -- gate run --------------------------------------------------------------------------

/// Exit codes `gate run` returns. `docs/epics/gate-driver.md:52-54`: "The driver exits non-zero on
/// any stage failure with the stage and the tool named"; "a mutation that leaves the proof green is
/// exit 1 (the check could not fail)"; "Budget expiry is exit 124 with elapsed time and the
/// incomplete stage". `2` is the environment verdict `docs check` already returns — the registry
/// cannot be read or parsed, or the child could not be started at all — and is not a refusal of the
/// row's gate.
const EXIT_ACCEPTED: u8 = 0;
const EXIT_REFUSED: u8 = 1;
const EXIT_ENVIRONMENT: u8 = 2;
const EXIT_BUDGET: u8 = 124;

/// Exit `gate docs check --controls` returns when it could not finish the selection: a row was
/// blocked, or the invocation budget ran out before a row was reached. Distinct from
/// [`EXIT_REFUSED`] because the two need different readers — a refusal says a control ran and did
/// not fire, which is a defect to fix; `3` says the aggregate did not get to that row, which is a
/// batch that is not yet an answer. Both are non-zero, and neither is ever reported as success.
const EXIT_INCOMPLETE: u8 = 3;

/// Bytes of the child's own output kept for a refusal tail. The streams are unbounded; the excerpt
/// is not, and a refusal that grows with its child's output is not a diagnostic.
const TAIL_BYTES: usize = 4096;

/// The frozen marker record, relative to the root the manifest resolves to.
/// `tools/gate/conformance.json` keys each gate row's `green_marker` by that row's `path`.
const CONFORMANCE: &str = "tools/gate/conformance.json";

/// One `key=value` line per field, in insertion order, every string value quoted and escaped. Fixed
/// order and quoting are what make the block parseable by a receipt that has no JSON reader, so the
/// block is built by one writer rather than by `println!` at each decision point.
#[derive(Debug, Default)]
struct Verdict {
    fields: Vec<(String, String)>,
}

impl Verdict {
    /// A `key=value` string value: quotes, backslashes and line breaks escaped, everything else
    /// verbatim. A command string carries spaces and a refusal carries a semicolon, so an unquoted
    /// value would not survive one line.
    fn quoted(key: &str, value: &str) -> (String, String) {
        let mut out = String::with_capacity(value.len() + 2);
        out.push('"');
        for ch in value.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                other => out.push(other),
            }
        }
        out.push('"');
        (key.to_string(), out)
    }

    /// A `key=value` numeric field, unquoted: a number is read as a number.
    fn number(key: &str, value: impl std::fmt::Display) -> (String, String) {
        (key.to_string(), value.to_string())
    }

    /// The block, one line per field, each line newline-terminated.
    fn render(&self) -> String {
        let mut out = String::new();
        for (key, value) in &self.fields {
            out.push_str(key);
            out.push('=');
            out.push_str(value);
            out.push('\n');
        }
        out
    }
}

/// What `gate run` decided. `verdict` goes to stdout as the machine-readable block; `report` goes to
/// stderr as the human refusals and the child's tail, so a piped stdout carries the block alone.
#[derive(Debug)]
struct Outcome {
    code: ExitCode,
    verdict: String,
    report: Vec<String>,
}

impl Outcome {
    /// `gate run` accepted the row: the command exited 0 inside its budget and printed the marker
    /// the frozen record requires, or carries no marker at all (stated in the block).
    fn accepted(verdict: String) -> Outcome {
        Outcome {
            code: ExitCode::from(EXIT_ACCEPTED),
            verdict,
            report: Vec::new(),
        }
    }

    /// Every other verdict: the exit code, the block, and what a reader has to see to attribute it.
    fn refused(code: u8, verdict: String, report: Vec<String>) -> Outcome {
        Outcome {
            code: ExitCode::from(code),
            verdict,
            report,
        }
    }

    /// A block with `id`, `manifest`, `cwd` and `command` already bound: the fields every verdict
    /// names, whether it accepted or refused, so a refusal says which row and which command.
    fn block(row_id: &str, manifest: &Path, root: &Path, command: &str, budget_s: u64) -> Verdict {
        let mut verdict = Verdict::default();
        verdict.fields.push(Verdict::quoted("id", row_id));
        verdict
            .fields
            .push(Verdict::quoted("manifest", &manifest.display().to_string()));
        verdict
            .fields
            .push(Verdict::quoted("cwd", &root.display().to_string()));
        verdict.fields.push(Verdict::quoted("command", command));
        verdict.fields.push(Verdict::number("budget_s", budget_s));
        verdict
    }
}

/// The `green_marker` the frozen record registers for `path`, or `None` when it registers none.
///
/// Reads the record as text rather than parsing it: this crate carries no dependency (see the
/// header of `adapter.rs`), and this reads one string out of one known-shaped object. The window is
/// cut at the next `"path"` key, so a marker is never taken from the neighbouring row, and a value
/// that is not a bare identifier is refused rather than returned as a marker the run could not
/// plausibly have printed — a marker that mis-parsed here would be reported as *missing* by every
/// green row, which is a check that fails on its own population.
///
/// Time O(record bytes): two substring searches over text already in memory. Space O(1) beyond the
/// record: every slice borrows it.
fn green_marker(record: &str, path: &str) -> Option<String> {
    let anchor = format!("\"path\": \"{path}\"");
    let start = record.find(&anchor)?;
    let rest = &record[start + anchor.len()..];
    let window = &rest[..rest.find("\"path\":").unwrap_or(rest.len())];
    let key = "\"green_marker\": \"";
    let tail = &window[window.find(key)? + key.len()..];
    let marker = &tail[..tail.find('"')?];
    let plain = !marker.is_empty()
        && marker
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    plain.then(|| marker.to_string())
}

/// Whether the row's green marker was printed.
#[derive(Debug, PartialEq, Eq)]
enum Marker {
    /// Printed by the child.
    Present(String),
    /// Required by the frozen record, absent from what the child printed. A row whose gate exits 0
    /// without its marker has not passed: it printed no evidence that it ran the thing.
    Missing(String),
    /// The record registers no marker for this row, so only its exit status was read. Reported
    /// rather than silently folded into `Present`.
    Unrecorded,
}

impl Marker {
    fn state(&self) -> &'static str {
        match self {
            Marker::Present(_) => "present",
            Marker::Missing(_) => "missing",
            Marker::Unrecorded => "unrecorded",
        }
    }

    /// The marker the record requires, or an empty value when it registers none.
    fn required(&self) -> &str {
        match self {
            Marker::Present(marker) | Marker::Missing(marker) => marker,
            Marker::Unrecorded => "",
        }
    }
}

/// `gate run <row-id> [--manifest <toml>]`. Exit 0 when the row's gate exited 0 inside its budget
/// and its recorded marker was printed; exit 1 on a refusal (unregistered id, repeated id, drifted
/// identity, unreadable path, non-zero exit, missing marker, no exit status); exit 2 when the
/// registry cannot be read or parsed or the child cannot be started; exit 124 when the child
/// outlives `budget_s` and is killed at it.
///
/// The identity check comes first and is the core's: `core::admit` refuses a row whose recorded
/// `sha256` is not the digest of the file on disk now, so a run can never execute a gate against a
/// source the registry no longer describes. No execution happens on that path, which is what the
/// digest control observes.
///
/// Complexity: one manifest load, one `sha256_path` over the row's path — O(bytes) time, O(files)
/// space for a directory — one conformance record read and scan, and one budgeted child. The child
/// dominates; wall time is bounded by `budget_s + POLL_MS` plus what the child itself printed.
fn run_row(row_id: &str, manifest_path: &Path) -> Outcome {
    let rows = match adapter::load_manifest(manifest_path) {
        Ok(rows) => rows,
        Err(error) => {
            return Outcome::refused(
                EXIT_ENVIRONMENT,
                Verdict {
                    fields: vec![
                        Verdict::quoted("id", row_id),
                        Verdict::quoted("manifest", &manifest_path.display().to_string()),
                        Verdict::quoted("verdict", "refused"),
                        Verdict::quoted("stage", "load-manifest"),
                        Verdict::quoted("reason", &error),
                        Verdict::number("exit", EXIT_ENVIRONMENT),
                    ],
                }
                .render(),
                vec![format!("gate run: {error}")],
            );
        }
    };

    let base = match current_dir() {
        Ok(base) => base,
        Err(error) => {
            return Outcome::refused(
                EXIT_ENVIRONMENT,
                Verdict {
                    fields: vec![
                        Verdict::quoted("id", row_id),
                        Verdict::quoted("manifest", &manifest_path.display().to_string()),
                        Verdict::quoted("verdict", "refused"),
                        Verdict::quoted("stage", "resolve-root"),
                        Verdict::quoted("reason", &error),
                        Verdict::number("exit", EXIT_ENVIRONMENT),
                    ],
                }
                .render(),
                vec![format!(
                    "gate run: cannot read the current directory to resolve the manifest root: {error}"
                )],
            );
        }
    };
    let root = root_for(manifest_path, &base);
    let manifest = to_manifest(&rows);

    // The core owns the identity verdict. A row that is not registered is still submitted, with an
    // empty digest: `core::admit` returns `Unregistered` for an id it cannot look up and never
    // compares that digest (`core.rs:132-136`), so the refusal is the core's and the path needs no
    // second spelling of "no such row".
    let found = rows.iter().find(|row| row.id == row_id);
    let refusals = match found {
        None => gate::admit(
            &manifest,
            &Submitted {
                id: row_id.to_string(),
                digest: Digest(String::new()),
                refs: Vec::new(),
            },
        ),
        Some(row) => match adapter::sha256_path(&root.join(&row.path), &root) {
            Err(error) => {
                let text = unreadable_text(&row.path, &error.to_string());
                return Outcome::refused(
                    EXIT_REFUSED,
                    Verdict {
                        fields: vec![
                            Verdict::quoted("id", row_id),
                            Verdict::quoted("manifest", &manifest_path.display().to_string()),
                            Verdict::quoted("cwd", &root.display().to_string()),
                            Verdict::quoted("command", &row.gate),
                            Verdict::number("budget_s", row.budget_s),
                            Verdict::quoted("verdict", "refused"),
                            Verdict::quoted("stage", "hash-path"),
                            Verdict::quoted("reason", &text),
                            Verdict::number("exit", EXIT_REFUSED),
                        ],
                    }
                    .render(),
                    vec![format!("  {text}")],
                );
            }
            Ok(hex) => gate::admit(
                &manifest,
                &Submitted {
                    id: row.id.clone(),
                    digest: Digest(format!("sha256:{hex}")),
                    refs: Vec::new(),
                },
            ),
        },
    };

    if !refusals.is_empty() {
        // Refused before execution: the identity the gate would run against is not the one this
        // registry records. An unregistered id has no row, so the block carries an empty command
        // rather than a command invented for it. Each refusal is also a field, so the block alone
        // names the cause and a receipt need not keep the human report.
        let (command, budget_s) =
            found.map_or((String::new(), 0), |row| (row.gate.clone(), row.budget_s));
        let report: Vec<String> = refusals.iter().map(refusal_text).collect();
        let mut fields = vec![
            Verdict::quoted("id", row_id),
            Verdict::quoted("manifest", &manifest_path.display().to_string()),
            Verdict::quoted("cwd", &root.display().to_string()),
            Verdict::quoted("command", &command),
            Verdict::number("budget_s", budget_s),
            Verdict::quoted("verdict", "refused"),
            Verdict::quoted("stage", "identity"),
            Verdict::number("refusals", refusals.len()),
        ];
        fields.extend(
            report
                .iter()
                .enumerate()
                .map(|(index, text)| Verdict::quoted(&format!("refusal[{index}]"), text.trim())),
        );
        fields.push(Verdict::number("exit", EXIT_REFUSED));
        return Outcome::refused(EXIT_REFUSED, Verdict { fields }.render(), report);
    }
    let row = found.expect("an accepted id resolves to its row");

    let required = adapter::read_text(&root.join(CONFORMANCE))
        .ok()
        .and_then(|record| green_marker(&record, &row.path))
        .map_or(Marker::Unrecorded, Marker::Missing);

    let (record, tail) =
        match adapter::run_captured_with_tail(&row.gate, &root, row.budget_s, TAIL_BYTES) {
            Ok(captured) => captured,
            Err(refusal) => {
                // `run_captured_with_tail` returns `Err` only when the child never ran: a child killed
                // at its budget comes back as a record with `timed_out` set, and is mapped below from
                // that record. Both variants are mapped here anyway, because a budget expiry that
                // arrived as a refusal must not become exit 2.
                let (stage, exit) = match &refusal {
                    adapter::RunRefusal::BudgetExceeded { .. } => ("budget", EXIT_BUDGET),
                    adapter::RunRefusal::Unrunnable { .. } => ("spawn", EXIT_ENVIRONMENT),
                };
                return Outcome::refused(
                    exit,
                    Verdict {
                        fields: vec![
                            Verdict::quoted("id", row_id),
                            Verdict::quoted("manifest", &manifest_path.display().to_string()),
                            Verdict::quoted("cwd", &root.display().to_string()),
                            Verdict::quoted("command", &row.gate),
                            Verdict::number("budget_s", row.budget_s),
                            Verdict::quoted("verdict", "refused"),
                            Verdict::quoted("stage", stage),
                            Verdict::quoted("reason", &refusal.to_string()),
                            Verdict::number("exit", exit),
                        ],
                    }
                    .render(),
                    vec![format!("gate run: {refusal}")],
                );
            }
        };

    // A child that exited 0 has proved nothing on its own: the marker the frozen record requires is
    // what says the gate ran the thing rather than printing nothing.
    let marker = match (&required, record.accepted()) {
        (Marker::Missing(marker), true) if tail_printed(&tail, marker) => {
            Marker::Present(marker.clone())
        }
        _ => required,
    };

    let mut verdict = Outcome::block(row_id, manifest_path, &root, &row.gate, row.budget_s);
    verdict
        .fields
        .push(Verdict::number("elapsed_ms", record.elapsed_ms));
    // The child's status and the driver's exit are different numbers under different names: a block
    // carrying two `exit` keys would leave a reader to guess which one it parsed.
    verdict.fields.push(match record.code {
        Some(code) => Verdict::number("child_exit", code),
        None => Verdict::quoted("child_exit", "none"),
    });
    verdict
        .fields
        .push(Verdict::quoted("timed_out", &record.timed_out.to_string()));
    verdict
        .fields
        .push(Verdict::quoted("stdout_sha256", &record.stdout_sha256));
    verdict
        .fields
        .push(Verdict::quoted("stderr_sha256", &record.stderr_sha256));
    verdict
        .fields
        .push(Verdict::quoted("marker", marker.required()));
    verdict
        .fields
        .push(Verdict::quoted("marker_state", marker.state()));
    verdict
        .fields
        .push(Verdict::number("tail_bytes", tail.len()));
    verdict
        .fields
        .push(Verdict::quoted("control", &row.control));

    // The tail is quoted on a refusal only: the adapter has already relayed the child's bytes to this
    // process's streams, so echoing them again on a pass would be the same output twice.
    let tail_report = |extra: String| {
        let mut report = vec![format!("--- {row_id} tail ---")];
        report.push(if tail.is_empty() {
            format!("gate run: {row_id} printed nothing on either stream")
        } else {
            tail.trim_end().to_string()
        });
        report.push(extra);
        report
    };

    match record.accepted() {
        // A child killed at its budget reported nothing, so reading its silence as a pass — or as a
        // missing marker — is the quiet success `AGENTS.md` §10 records. `docs/epics/gate-driver.md:54`
        // gives this exit: "Budget expiry is exit 124 with elapsed time and the incomplete stage".
        false if record.timed_out => {
            let refusal = record
                .refusal()
                .expect("a killed child carries its refusal");
            verdict.fields.push(Verdict::quoted("verdict", "refused"));
            verdict.fields.push(Verdict::quoted("stage", "budget"));
            verdict
                .fields
                .push(Verdict::quoted("reason", &refusal.to_string()));
            verdict.fields.push(Verdict::number("exit", EXIT_BUDGET));
            Outcome::refused(
                EXIT_BUDGET,
                verdict.render(),
                tail_report(format!("gate run: {refusal}")),
            )
        }
        false => {
            let (status, note) = match record.code {
                Some(code) => (
                    format!("exit {code}"),
                    format!(
                        "gate run: {row_id} refused: {} reported exit {code} in {}ms",
                        row.gate, record.elapsed_ms
                    ),
                ),
                None => (
                    "no exit status (killed by a signal)".to_string(),
                    format!(
                        "gate run: {row_id} refused: {} reported no exit status in {}ms; \
                         a signal death is not a verdict",
                        row.gate, record.elapsed_ms
                    ),
                ),
            };
            verdict.fields.push(Verdict::quoted("verdict", "refused"));
            verdict.fields.push(Verdict::quoted("stage", "gate-exit"));
            verdict.fields.push(Verdict::quoted("reason", &status));
            verdict.fields.push(Verdict::number("exit", EXIT_REFUSED));
            Outcome::refused(EXIT_REFUSED, verdict.render(), tail_report(note))
        }
        // A child that exited 0 has proved nothing on its own: the marker the frozen record
        // requires is what says the gate ran the thing rather than printing nothing.
        true => match &marker {
            Marker::Missing(marker) => {
                let note = format!(
                    "gate run: {row_id} exited 0 but did not print its recorded marker `{marker}`"
                );
                verdict.fields.push(Verdict::quoted("verdict", "refused"));
                verdict
                    .fields
                    .push(Verdict::quoted("stage", "green-marker"));
                verdict.fields.push(Verdict::number("exit", EXIT_REFUSED));
                Outcome::refused(EXIT_REFUSED, verdict.render(), tail_report(note))
            }
            Marker::Present(_) | Marker::Unrecorded => {
                verdict.fields.push(Verdict::quoted("verdict", "accepted"));
                verdict.fields.push(Verdict::number("exit", EXIT_ACCEPTED));
                Outcome::accepted(verdict.render())
            }
        },
    }
}

/// Whether `marker` is in the captured tail. The tail is the excerpt a refusal shows; it is also
/// what the marker is read from, because the adapter relays the child's bytes to this process's
/// streams as it runs and keeps no other copy.
fn tail_printed(tail: &str, marker: &str) -> bool {
    !marker.is_empty() && tail.contains(marker)
}

/// `gate run <row-id> [--manifest <path>]`. `--manifest` may come before or after the row id; the
/// default is the registry `docs check` reads. Two row ids is a usage error rather than a silent
/// choice of one.
fn run_command(args: &[String]) -> ExitCode {
    let mut manifest = PathBuf::from(DEFAULT_MANIFEST);
    let mut row_id: Option<&str> = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--manifest" => match args.get(index + 1) {
                Some(path) => {
                    manifest = PathBuf::from(path);
                    index += 1;
                }
                None => {
                    eprintln!("gate run: --manifest needs a path");
                    return ExitCode::from(EXIT_ENVIRONMENT);
                }
            },
            other if other.starts_with('-') => {
                eprintln!("gate run: unknown flag `{other}`");
                return ExitCode::from(EXIT_ENVIRONMENT);
            }
            other => {
                if row_id.replace(other).is_some() {
                    eprintln!("usage: gate run <row-id> [--manifest <toml>]");
                    return ExitCode::from(EXIT_ENVIRONMENT);
                }
            }
        }
        index += 1;
    }

    let Some(row_id) = row_id else {
        eprintln!("usage: gate run <row-id> [--manifest <toml>]\n  manifest defaults to {DEFAULT_MANIFEST}");
        return ExitCode::from(EXIT_ENVIRONMENT);
    };

    let outcome = run_row(row_id, &manifest);
    eprint!("{}", outcome.report.join("\n"));
    print!("{}", outcome.verdict);
    outcome.code
}

// -- gate docs check --controls ---------------------------------------------------------

/// Wall-clock seconds a `--controls` invocation may take when the caller declares none.
///
/// The default is `adapter::MAX_BUDGET_S` = 600, the registry's own per-row ceiling and the
/// `AGENTS.md` §10 failure boundary, because the default has to leave room for everything the
/// invocation spends before its first row. The aggregate makes its throwaway copy inside the same
/// budget — the copy of this tree measured 7426 files in 188 ms warm and in 43 s after a reboot —
/// and a row is attempted only when what is left covers that row's own `budget_s`. A default below
/// the largest declared row budget therefore cannot attempt it: `budget_s = 540` for a 540 s row
/// leaves `540 − copy` seconds for a 540 s child, which is never enough, and the row is reported
/// `unattempted:invocation-budget` without its control ever running. At the ceiling the headroom is
/// `MAX_BUDGET_S − row.budget_s`, which is what pays the copy. The admissible range stays
/// `1..=adapter::MAX_BUDGET_S`, so a declared budget is one that could be met.
const DEFAULT_CONTROL_BUDGET_S: u64 = adapter::MAX_BUDGET_S;

/// One per-row control state, as the aggregate prints it. Every state is a machine-readable token
/// and none of them reads as success except [`FIRED`]: a reader that greps for `state=` gets a
/// verdict, not a count.
const FIRED: &str = "fired";
const DID_NOT_FIRE: &str = "did-not-fire";
const BLOCKED_NO_CONTROL: &str = "blocked:no-control";
const BLOCKED_IDENTITY: &str = "blocked:identity";
const BLOCKED_UNRUNNABLE: &str = "blocked:unrunnable";
const BLOCKED_ROW_BUDGET: &str = "blocked:row-budget";
const UNATTEMPTED: &str = "unattempted:invocation-budget";

/// The parsed `gate docs check` invocation.
///
/// `controls` is off by default, and every other field is refused without it, so `gate docs check`
/// and `gate docs check <manifest>` reach the identity check by exactly the path they reached it
/// before this flag existed. `budget_given` distinguishes an explicit `--budget-s` from the default,
/// because `--budget-s 540` is an error without `--controls` while the default is not a request.
#[derive(Debug)]
struct CheckOptions {
    controls: bool,
    /// Row ids to attempt, in the registry's own order. Empty means every row.
    rows: Vec<String>,
    budget_s: u64,
    budget_given: bool,
    copy_root: Option<PathBuf>,
    keep_copy: bool,
}

impl Default for CheckOptions {
    fn default() -> CheckOptions {
        CheckOptions {
            controls: false,
            rows: Vec::new(),
            budget_s: DEFAULT_CONTROL_BUDGET_S,
            budget_given: false,
            copy_root: None,
            keep_copy: false,
        }
    }
}

impl CheckOptions {
    /// Usage error, on stderr, exit `2`. An invocation the tool cannot act on is an environment
    /// verdict, not a refusal of a row: nothing was run, so nothing was disproved.
    fn usage(&self, message: &str) -> ExitCode {
        eprintln!("gate docs check: {message}");
        eprintln!("{USAGE}");
        ExitCode::from(EXIT_ENVIRONMENT)
    }
}

const USAGE: &str = "usage: gate docs check [manifest] [--controls] [--row <id>]... [--budget-s <n>]\n\
  \x20                     [--copy-root <dir>] [--keep-copy]\n  \
manifest defaults to docs/contracts/artifacts.toml\n  \
--controls      run each selected row's control in a throwaway copy; exit 0 only if every\n                \
                selected row was attempted and every control fired and restored\n  \
--row <id>      attempt only this row; repeatable, deduplicated, applied in registry order\n  \
--budget-s <n>  whole-invocation wall-clock ceiling, 1..=600 (default 600)\n  \
--copy-root <d> where the throwaway copy is made (default $TMPDIR; pass /home/socol/oc-tmp on this\n                \
                host: /tmp is a 32G tmpfs and one copy of this tree is ~77 MB)\n  \
--keep-copy     keep the throwaway copy instead of removing it";

/// `gate docs check …`: parse the invocation, then run the identity check and, when asked, the
/// aggregate over the selected rows' controls. The two are separate phases rather than one pass so
/// the identity verdict is reached on its own evidence before anything is executed.
fn docs_command(args: &[String]) -> ExitCode {
    let mut manifest: Option<PathBuf> = None;
    let mut options = CheckOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--controls" => options.controls = true,
            "--row" => match args.get(index + 1) {
                Some(id) => {
                    options.rows.push(id.clone());
                    index += 1;
                }
                None => return options.usage("--row needs a row id"),
            },
            "--budget-s" => match args.get(index + 1).map(|n| n.parse::<u64>()) {
                Some(Ok(seconds)) if (1..=adapter::MAX_BUDGET_S).contains(&seconds) => {
                    options.budget_s = seconds;
                    options.budget_given = true;
                    index += 1;
                }
                Some(Ok(seconds)) => {
                    return options.usage(&format!(
                        "--budget-s {seconds} is outside 1..={}",
                        adapter::MAX_BUDGET_S
                    ))
                }
                Some(Err(_)) => return options.usage("--budget-s is not a whole number of seconds"),
                None => return options.usage("--budget-s needs a number of seconds"),
            },
            "--copy-root" => match args.get(index + 1) {
                Some(dir) => {
                    options.copy_root = Some(PathBuf::from(dir));
                    index += 1;
                }
                None => return options.usage("--copy-root needs a directory"),
            },
            "--keep-copy" => options.keep_copy = true,
            other if other.starts_with('-') => {
                return options.usage(&format!("unknown flag `{other}`"))
            }
            other => {
                if manifest.replace(PathBuf::from(other)).is_some() {
                    return options.usage("two manifest paths; `gate docs check` takes one");
                }
            }
        }
        index += 1;
    }

    if !options.controls {
        // A flag that means nothing without `--controls` is a usage error rather than a silent
        // no-op: a caller who asked for a batch of three rows and got an identity check has not been
        // answered.
        if !options.rows.is_empty() {
            return options.usage("--row needs --controls");
        }
        if options.budget_given {
            return options.usage("--budget-s needs --controls");
        }
        if options.copy_root.is_some() {
            return options.usage("--copy-root needs --controls");
        }
        if options.keep_copy {
            return options.usage("--keep-copy needs --controls");
        }
    }

    docs_check(
        &manifest.unwrap_or_else(|| PathBuf::from(DEFAULT_MANIFEST)),
        &options,
    )
}

/// The rows to attempt, in registry order.
///
/// With no `--row`, every row. With `--row`, the named rows in the order the registry holds them,
/// so the selection does not depend on the order the flags were typed in, and a repeated id is
/// attempted once: running a control twice would double-count it in the summary and burn a row's
/// budget twice for one verdict.
///
/// An id the registry does not register is returned as a refusal rather than skipped: a selection
/// naming a row that does not exist is a question about this registry, and answering it by running
/// the other rows would leave the reader to guess which row was dropped.
///
/// Complexity: time O(comparisons) — one linear scan of the row list per requested id — and space
/// O(selected) for the rows kept. Each scan walks the list and stops at the first id match, and the
/// loop is over the requested ids, so the count closes over `rows × requested` comparisons and the
/// selection itself never allocates per registry row.
fn select_rows<'a>(rows: &'a [Row], requested: &[String]) -> Result<Vec<&'a Row>, Vec<String>> {
    if requested.is_empty() {
        return Ok(rows.iter().collect());
    }
    let mut selected: Vec<&Row> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    for id in requested {
        match rows.iter().find(|row| &row.id == id) {
            None => unknown.push(id.clone()),
            Some(row) => {
                if !selected.iter().any(|kept| kept.id == row.id) {
                    selected.push(row);
                }
            }
        }
    }
    if unknown.is_empty() {
        selected.sort_by_key(|row| {
            rows.iter()
                .position(|earlier| earlier.id == row.id)
                .unwrap_or(usize::MAX)
        });
        Ok(selected)
    } else {
        Err(unknown)
    }
}

/// One row's control verdict, and the observations it was decided from. Every row produces one of
/// these, whatever happened to it, so the summary counts and the per-row blocks are the same set.
struct ControlOutcome {
    id: String,
    control: String,
    budget_s: u64,
    state: &'static str,
    /// Whether the copy carried the recorded identity before the control ran.
    identity_before: bool,
    /// Whether the copy carried it afterwards; `None` when no control ran, so `none` never reads as
    /// "restored".
    identity_after: Option<bool>,
    code: Option<i32>,
    timed_out: bool,
    elapsed_ms: u128,
    stdout_sha256: String,
    stderr_sha256: String,
    reason: String,
}

impl ControlOutcome {
    /// A row the aggregate did not attempt, or refused before running anything.
    fn without_run(id: &str, control: &str, budget_s: u64, state: &'static str, reason: String) -> Self {
        ControlOutcome {
            id: id.to_string(),
            control: control.to_string(),
            budget_s,
            state,
            identity_before: false,
            identity_after: None,
            code: None,
            timed_out: false,
            elapsed_ms: 0,
            stdout_sha256: NONE.to_string(),
            stderr_sha256: NONE.to_string(),
            reason,
        }
    }

    /// Whether this row's control ran to an exit status and the copy still carries the registered
    /// identity. The only shape `fired` has.
    fn fired(&self) -> bool {
        self.state == FIRED
    }

    /// The row's `key=value` block, in one fixed field order with every value quoted or a number, so
    /// a receipt parses it without a JSON reader. Absent values are spelled (`"none"`, `"false"`)
    /// rather than omitted: a field that is missing is a field the aggregate did not report, and a
    /// reader must not have to tell that from a field reported as absent.
    fn verdict(&self, copy_root: &Path) -> String {
        let mut verdict = Verdict::default();
        verdict.fields.push(Verdict::quoted("id", &self.id));
        verdict.fields.push(Verdict::quoted("state", self.state));
        verdict.fields.push(Verdict::quoted("control", &self.control));
        verdict.fields.push(Verdict::number("budget_s", self.budget_s));
        verdict
            .fields
            .push(Verdict::quoted("copy_root", &copy_root.display().to_string()));
        verdict
            .fields
            .push(Verdict::quoted("identity_before", &self.identity_before.to_string()));
        verdict.fields.push(Verdict::quoted(
            "identity_after",
            match self.identity_after {
                None => NONE,
                Some(restored) => {
                    if restored {
                        "true"
                    } else {
                        "false"
                    }
                }
            },
        ));
        verdict.fields.push(match self.code {
            Some(code) => Verdict::number("child_exit", code),
            None => Verdict::quoted("child_exit", NONE),
        });
        verdict
            .fields
            .push(Verdict::quoted("timed_out", &self.timed_out.to_string()));
        verdict.fields.push(Verdict::number("elapsed_ms", self.elapsed_ms));
        verdict
            .fields
            .push(Verdict::quoted("stdout_sha256", &self.stdout_sha256));
        verdict
            .fields
            .push(Verdict::quoted("stderr_sha256", &self.stderr_sha256));
        verdict.fields.push(Verdict::quoted("reason", &self.reason));
        verdict.fields.push(Verdict::number("exit", self.exit_code()));
        verdict.render()
    }

    /// The exit this row would produce on its own. `fired` alone is 0; a control that ran and did
    /// not fire is 1; a row the aggregate could not run, or did not reach, is 3.
    fn exit_code(&self) -> u8 {
        match self.state {
            FIRED => EXIT_ACCEPTED,
            DID_NOT_FIRE => EXIT_REFUSED,
            _ => EXIT_INCOMPLETE,
        }
    }
}

/// The value a field takes when the aggregate has nothing to report for it. A string, not an empty
/// field, so `identity_after=none` is distinguishable from `identity_after=false`.
const NONE: &str = "none";

/// One row's control, run in `copy_root`.
///
/// Three refusals are possible before or instead of the run, and each names its cause: no control
/// recorded, a copy that does not carry the registered identity, a control that cannot be started.
/// Two are possible after it: a child killed at the row's own `budget_s`, and a child that exited 0
/// having left the registered identity moved — a control that mutated the copy and did not restore
/// it has not done the third thing `docs/contracts/artifacts.toml` requires of it, and its exit 0
/// is not evidence that it did.
///
/// Complexity: two `sha256_path` digests over the row's registered path — O(bytes) time on a file,
/// O(files + bytes) over a directory walk — and one budgeted child bounded by `row.budget_s` plus
/// the adapter's `POLL_MS`. Space O(TAIL_BYTES) for the refusal tail and O(COPY_CHUNK) for the
/// relay; the child dominates, and it is bounded.
fn run_one_control(copy_root: &Path, row: &Row) -> ControlOutcome {
    if row.control.trim().is_empty() {
        return ControlOutcome::without_run(
            &row.id,
            &row.control,
            row.budget_s,
            BLOCKED_NO_CONTROL,
            "the registry records no control for this row, so there is nothing that could have reddened"
                .to_string(),
        );
    }

    // The control runs against the copy, so the copy must carry the identity the registry records.
    // A control handed a payload that has already moved would redden for a reason that has nothing
    // to do with the control, and the red would be reported as the control's evidence.
    let path = copy_root.join(&row.path);
    let before = match adapter::sha256_path(&path, copy_root) {
        Err(error) => {
            return ControlOutcome::without_run(
                &row.id,
                &row.control,
                row.budget_s,
                BLOCKED_UNRUNNABLE,
                format!("the copy has no {}: {error}", row.path),
            )
        }
        Ok(hex) if hex != row.sha256 => {
            return ControlOutcome::without_run(
                &row.id,
                &row.control,
                row.budget_s,
                BLOCKED_IDENTITY,
                format!(
                    "the copy carries {} for {}, the registry records {}; no control ran",
                    hex, row.path, row.sha256
                ),
            )
        }
        Ok(_) => true,
    };

    let record = match adapter::run_captured_with_tail(&row.control, copy_root, row.budget_s, TAIL_BYTES)
    {
        Ok((record, _tail)) => record,
        Err(refusal) => {
            // `run_captured_with_tail` returns `Err` only when the child never ran; a child killed at
            // its budget comes back as a record with `timed_out` set and is classified below.
            let state = match &refusal {
                adapter::RunRefusal::BudgetExceeded { .. } => BLOCKED_ROW_BUDGET,
                adapter::RunRefusal::Unrunnable { .. } => BLOCKED_UNRUNNABLE,
            };
            return ControlOutcome::without_run(
                &row.id,
                &row.control,
                row.budget_s,
                state,
                refusal.to_string(),
            );
        }
    };

    // The restoration is read from the copy, not taken from the control's exit status: a control
    // that exited 0 having deleted or rewritten the payload it was given has not restored it, and
    // that is a refusal rather than a pass.
    let restored = adapter::sha256_path(&path, copy_root)
        .map(|hex| hex == row.sha256)
        .map_err(|error| format!("the copy no longer has {} to compare: {error}", row.path));
    let mut outcome = ControlOutcome {
        id: row.id.clone(),
        control: row.control.clone(),
        budget_s: row.budget_s,
        state: FIRED,
        identity_before: before,
        identity_after: restored.as_ref().ok().copied(),
        code: record.code,
        timed_out: record.timed_out,
        elapsed_ms: record.elapsed_ms,
        stdout_sha256: record.stdout_sha256.clone(),
        stderr_sha256: record.stderr_sha256.clone(),
        reason: String::new(),
    };

    if record.timed_out {
        outcome.state = BLOCKED_ROW_BUDGET;
        outcome.reason = record
            .refusal()
            .map_or_else(|| format!("killed at {}s", row.budget_s), |r| r.to_string());
    } else if record.code != Some(0) {
        outcome.state = DID_NOT_FIRE;
        outcome.reason = match record.code {
            Some(code) => format!(
                "the control reported exit {code} in {}ms; it did not require the refusal it exists for",
                record.elapsed_ms
            ),
            None => format!(
                "the control reported no exit status in {}ms (killed by a signal); a signal death is not a verdict",
                record.elapsed_ms
            ),
        };
    } else {
        // Exited 0. `fired` needs the copy to still carry the registered identity, so a control that
        // rewrote or removed what it was given is refused by name and the two causes are told apart.
        outcome.state = DID_NOT_FIRE;
        outcome.reason = match &restored {
            Ok(true) => {
                outcome.state = FIRED;
                outcome.reason = String::new();
                outcome.reason.clone()
            }
            Ok(false) => format!(
                "the control exited 0 but left the registered identity of {} moved; it mutated the copy and did not restore it",
                row.path
            ),
            Err(reason) => format!("the control exited 0 but {reason}"),
        };
    }
    outcome
}

/// Milliseconds of the invocation budget still unspent at `elapsed_ms`, saturating at zero.
///
/// The budget is whole seconds and the clock is milliseconds, so a `u128` elapsed past the `u64`
/// range is clamped rather than wrapped: a wrapped subtraction would report a full budget again
/// after the overrun, and a row the budget had already spent would be attempted.
///
/// Complexity: O(1) time and O(1) space — two saturating arithmetic operations.
fn remaining_ms(budget_s: u64, elapsed_ms: u128) -> u64 {
    budget_s
        .saturating_mul(1000)
        .saturating_sub(elapsed_ms.min(u64::MAX as u128) as u64)
}

/// Why a row was not attempted: the budget arithmetic, and what the setup spent.
///
/// The copy is inside the invocation budget, so the gap between the two numbers is where the
/// aggregate's own setup went. A reader told only `497s remain of the 540s invocation budget` cannot
/// tell a 43 s copy from 43 s of other rows' controls, and the two need different remedies: one is
/// this row's chance, the other is the batch's.
///
/// Complexity: O(1) time and O(1) space; one `format!` over four numbers.
fn unattempted_reason(
    row_budget_s: u64,
    budget_s: u64,
    remaining_ms: u64,
    setup_ms: u128,
) -> String {
    format!(
        "needs {row_budget_s}s of control budget, {}s remain of the {budget_s}s invocation budget \
         ({setup_ms}ms spent making the copy)",
        remaining_ms / 1000
    )
}

/// Where the throwaway copy goes, and whether the caller should be advised about it.
///
/// A declared `--copy-root` is used verbatim and nothing is advised: the caller said where, and
/// repeating the advice back at them is noise. Without one the copy goes to `TMPDIR`, and the second
/// value is `true` because that is the case the failure register's tmpfs-exhaustion row is about —
/// `/tmp` here is a 32 GB tmpfs that filled with finished scratch trees — so the aggregate says so
/// once, naming the flag and this box's answer, rather than changing the default behind the caller's
/// back. Kept separate from the copy itself so the decision is testable without making a copy.
///
/// Complexity: O(1) time and O(1) space; a `clone` of the declared path and one `temp_dir` read.
fn copy_base(declared: Option<&Path>) -> (PathBuf, bool) {
    match declared {
        Some(dir) => (dir.to_path_buf(), false),
        None => (std::env::temp_dir(), true),
    }
}

/// The aggregate: the throwaway copy, then one control per selected row inside it.
///
/// The copy is made once and every control runs in it, so the controls share one tree the way the
/// repository's own gates share one working tree, and one copy's cost is paid once. The registry
/// lock is held by the caller across this, so a check cannot overlap another check over the same
/// tree while a control is mutating the copy.
///
/// Exit 0 requires every selected row attempted and every control fired and restored. A refusal
/// (exit 1) outranks incompleteness (exit 3) in the returned code and both outrank success, so a run
/// with one broken control and thirty unreached rows cannot be read as a pass.
///
/// Complexity: one directory walk for the copy — O(files) writes, O(bytes) copied — then one
/// budgeted child per attempted row. Wall time is bounded by `options.budget_s`, whose clock starts
/// before the copy: a row is attempted only when the remaining budget covers its own `budget_s`, and
/// each child is killed at `row.budget_s`, so the invocation overruns by at most one `POLL_MS` per
/// attempted row. Space is O(outcomes) for the per-row records.
fn run_controls(root: &Path, rows: &[Row], options: &CheckOptions) -> ExitCode {
    let selected = match select_rows(rows, &options.rows) {
        Ok(selected) => selected,
        Err(unknown) => {
            for id in &unknown {
                eprintln!("  unregistered row: {id}");
            }
            eprintln!(
                "gate docs check --controls: {} of {} requested row id(s) are not in this registry; no control ran",
                unknown.len(),
                options.rows.len()
            );
            return ExitCode::from(EXIT_REFUSED);
        }
    };

    let (base, advise_tmpfs) = copy_base(options.copy_root.as_deref());
    if advise_tmpfs {
        // Advice on the invocation that took the default, because the default is the temporary
        // directory, which on this host is the 32 GB tmpfs the failure register records filling with
        // finished scratch trees. It is advice and not a change of default: `TMPDIR` is the portable
        // answer, this box's is not, and the caller passes `--copy-root` to say where. One copy
        // serves every selected row and it is removed when this function returns.
        println!(
            "gate docs check --controls: copy root {} is the default; pass --copy-root \
             /home/socol/oc-tmp to keep a ~77 MB copy off the /tmp tmpfs on this host",
            base.display()
        );
    }
    let start = std::time::Instant::now();
    let (copy, copied) = match adapter::ControlCopy::create(&base, root, options.keep_copy) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!(
                "gate docs check --controls: throwaway copy of {} under {}: {error}",
                root.display(),
                base.display()
            );
            return ExitCode::from(EXIT_ENVIRONMENT);
        }
    };
    let copy_root = copy.root().to_path_buf();
    let copy_ms = start.elapsed().as_millis();
    println!(
        "gate docs check --controls: copy {} — {} files, {} bytes, {} skipped, {} symlinks, {}ms",
        copy_root.display(),
        copied.files,
        copied.bytes,
        copied.skipped,
        copied.symlinks,
        copy_ms
    );

    let mut outcomes: Vec<ControlOutcome> = Vec::new();
    for row in &selected {
        let remaining_ms = remaining_ms(options.budget_s, start.elapsed().as_millis());
        if remaining_ms < row.budget_s.saturating_mul(1000) {
            outcomes.push(ControlOutcome::without_run(
                &row.id,
                &row.control,
                row.budget_s,
                UNATTEMPTED,
                unattempted_reason(row.budget_s, options.budget_s, remaining_ms, copy_ms),
            ));
            continue;
        }
        outcomes.push(run_one_control(&copy_root, row));
    }

    let count = |state: &str| outcomes.iter().filter(|row| row.state == state).count();
    let fired = outcomes.iter().filter(|row| row.fired()).count();
    let did_not_fire = count(DID_NOT_FIRE);
    let blocked = outcomes.len() - fired - did_not_fire;
    let unattempted = count(UNATTEMPTED);
    println!(
        "gate docs check --controls: {} selected; {} fired; {} did-not-fire; {} blocked; {} unattempted; budget {}s, elapsed {}ms",
        selected.len(),
        fired,
        did_not_fire,
        blocked,
        unattempted,
        options.budget_s,
        start.elapsed().as_millis()
    );

    let mut code = ExitCode::SUCCESS;
    for outcome in &outcomes {
        print!("{}", outcome.verdict(&copy_root));
        if outcome.state != FIRED {
            eprintln!("  {}: {}: {}", outcome.state, outcome.id, outcome.reason);
            if code == ExitCode::SUCCESS {
                code = ExitCode::from(outcome.exit_code());
            }
        }
    }
    if did_not_fire > 0 {
        code = ExitCode::from(EXIT_REFUSED);
    } else if blocked > 0 || unattempted > 0 {
        code = ExitCode::from(EXIT_INCOMPLETE);
    }
    drop(copy);
    code
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hold") => hold(),
        Some("docs") => match args.get(1).map(String::as_str) {
            Some("check") => docs_command(&args[2..]),
            _ => {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        },
        Some("run") => run_command(&args[1..]),
        Some(other) => {
            eprintln!("gate: unknown subcommand `{other}`; known: docs, hold, run");
            ExitCode::from(2)
        }
        None => {
            eprintln!(
                "usage: gate <subcommand>\n  docs check   load the artefact registry, report refusals, and with\n                \
                 --controls run each selected row's control in a throwaway copy\n  \
                 run <id>     execute one registry row's gate command under its budget\n  \
                 hold         refuse while a HOLD marker is in force"
            );
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod controls {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A scratch tree off the tmpfs-filling `/tmp` when `TMPDIR` says so; removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let path =
                std::env::temp_dir().join(format!("gate-{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(path.join("docs/contracts")).expect("scratch tree");
            Scratch(path)
        }

        fn write(&self, relative: &str, contents: &str) {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().expect("a parent")).expect("parent dir");
            std::fs::write(path, contents).expect("scratch file");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// One registry row as text. `sha256` is passed through verbatim, so a test can record an
    /// identity the file does not have.
    fn row_text(id: &str, path: &str, sha256: &str) -> String {
        format!(
            "[[artifact]]\nid = \"{id}\"\nkind = \"rust\"\npath = \"{path}\"\n\
             sha256 = \"{sha256}\"\ngate = \"cargo test\"\ncontrol = \"\"\nbudget_s = 540\n\n"
        )
    }

    /// The recorded identity of a scratch file, from the adapter — the same code path the registry
    /// is built with, so a test never hard-codes a digest.
    fn digest_of(root: &Path, relative: &str) -> String {
        adapter::sha256_path(&root.join(relative), root).expect("scratch digest")
    }

    /// Rejecting control: a registry that registers one id twice. The repeated id is refused
    /// before any payload is compared, because no identity in that manifest is authoritative, and
    /// the refusal names the id rather than counting it.
    #[test]
    fn repeated_id_is_refused() {
        let scratch = Scratch::new("dup");
        scratch.write("a.rs", "fn a() {}\n");
        let root = scratch.0.clone();
        let hex = digest_of(&root, "a.rs");
        let manifest_text =
            row_text("a.rs", "a.rs", &hex) + &row_text("a.rs", "a.rs", "0".repeat(64).as_str());
        let rows = adapter::parse(&manifest_text).expect("syntax is valid; the id is not");

        let report = check(&root, &rows);
        assert_eq!(report.loaded, 2);
        assert_eq!(
            report.duplicates,
            vec![Refusal::DuplicateRegistration("a.rs".into())]
        );
        assert!(report.mismatches.is_empty(), "no row is compared");
        assert!(!report.accepted());
    }

    /// Rejecting control: a row whose recorded identity differs from the file on disk. The verdict
    /// is the core's, and it carries both identities so the report says which one moved.
    #[test]
    fn digest_mismatch_is_refused() {
        let scratch = Scratch::new("drift");
        scratch.write("a.rs", "fn a() {}\n");
        let root = scratch.0.clone();
        let recorded = "0".repeat(64);
        let manifest_text = row_text("a.rs", "a.rs", &recorded);
        let rows = adapter::parse(&manifest_text).expect("syntax is valid");

        let report = check(&root, &rows);
        assert!(report.duplicates.is_empty());
        assert_eq!(
            report.mismatches,
            vec![Refusal::IdentityMismatch {
                id: "a.rs".into(),
                expected: format!("sha256:{recorded}"),
                actual: format!("sha256:{}", digest_of(&root, "a.rs")),
            }]
        );
        assert!(!report.accepted());
    }

    /// The positive the two controls are measured against: the same two rows with the identity the
    /// file actually has load clean. Without this, a check that refused everything would pass both
    /// controls.
    #[test]
    fn matching_digest_passes() {
        let scratch = Scratch::new("match");
        scratch.write("a.rs", "fn a() {}\n");
        let root = scratch.0.clone();
        let hex = digest_of(&root, "a.rs");
        let manifest_text = row_text("a.rs", "a.rs", &hex) + &row_text("b.rs", "a.rs", &hex);
        let rows = adapter::parse(&manifest_text).expect("syntax is valid");

        let report = check(&root, &rows);
        assert!(report.accepted(), "{report:?}");
        assert_eq!(report.loaded, 2);
    }

    /// An empty registry is refused, not accepted: a loader that reports green for zero records
    /// cannot fail, and a check that cannot fail has no weight.
    #[test]
    fn empty_registry_is_refused() {
        let scratch = Scratch::new("empty");
        let root = scratch.0.clone();
        let report = check(
            &root,
            &adapter::parse("").expect("empty text is valid TOML"),
        );
        assert_eq!(report.loaded, 0);
        assert!(!report.accepted());
    }

    /// A digest-mismatch diagnostic must be attributable: it names the row and both identities, so a
    /// reader can tell which recorded digest moved and which one is on disk without re-running the
    /// check. A refusal that says only "mismatch" would leave the reader to re-derive it.
    #[test]
    fn digest_mismatch_diagnostic_names_row_and_both_identities() {
        let scratch = Scratch::new("attrib");
        scratch.write("a.rs", "fn a() {}\n");
        let root = scratch.0.clone();
        let recorded = "0".repeat(64);
        let rows = adapter::parse(&row_text("proof/a.rs", "a.rs", &recorded)).expect("syntax");
        let report = check(&root, &rows);
        assert_eq!(report.mismatches.len(), 1, "{report:?}");

        let text = refusal_text(&report.mismatches[0]);
        assert!(text.contains("proof/a.rs"), "{text}");
        assert!(text.contains(&recorded), "{text}");
        assert!(text.contains(&digest_of(&root, "a.rs")), "{text}");
    }

    /// Rejecting control: a row whose `path` does not resolve. The refusal names the row and the OS
    /// reason, and restoring the file at the recorded identity turns it green — so the refusal came
    /// from the missing path and not from the loader refusing everything.
    #[test]
    fn missing_path_is_refused() {
        let scratch = Scratch::new("missing");
        scratch.write("a.rs", "fn a() {}\n");
        let root = scratch.0.clone();
        let hex = digest_of(&root, "a.rs");
        let rows = adapter::parse(&row_text("a.rs", "a.rs", &hex)).expect("syntax");

        // Green first: the row resolves and the identity matches.
        assert!(check(&root, &rows).accepted());

        // The path goes missing on the throwaway tree.
        std::fs::remove_file(root.join("a.rs")).expect("remove");
        let report = check(&root, &rows);
        assert_eq!(report.loaded, 1, "the row still loaded");
        assert_eq!(
            report.mismatches,
            Vec::new(),
            "no identity exists to compare"
        );
        assert_eq!(report.unreadable.len(), 1, "{report:?}");
        let text = &report.unreadable[0];
        assert!(text.starts_with("unreadable: a.rs: "), "{text}");
        assert!(
            text.len() > "unreadable: a.rs: ".len(),
            "the reason is named: {text}"
        );
        assert!(!report.accepted());

        // Restore: the same bytes at the same identity load clean again.
        scratch.write("a.rs", "fn a() {}\n");
        assert!(check(&root, &rows).accepted());
    }

    // -- gate run --------------------------------------------------------------------------

    /// A registry row for `gate run`, in the seven keys `adapter::parse` admits. The recorded
    /// identity is the identity of the payload, or `recorded` when the test is about a drifted
    /// digest.
    ///
    /// The command carries no `\n`: the parser's escape vocabulary is `\"` and `\\` only
    /// (`adapter.rs:116-118`), so a command with a line break in it is refused at the syntax layer
    /// rather than silently repaired. `gate run` passes a row's command to `/bin/sh -c` exactly as
    /// recorded, so the registry's escape rule is also the command's.
    fn run_row_text(id: &str, path: &str, sha256: &str, gate: &str, budget_s: u64) -> String {
        format!(
            "[[artifact]]\nid = \"{id}\"\nkind = \"gate\"\npath = \"{path}\"\n\
             sha256 = \"{sha256}\"\ngate = \"{gate}\"\ncontrol = \"\"\nbudget_s = {budget_s}\n\n"
        )
    }

    /// The frozen marker record `gate run` reads for a row's green marker, with one entry.
    fn conformance_text(path: &str, marker: &str) -> String {
        format!(
            "{{\n  \"record\": \"test\",\n  \"gates\": [\n    {{\n      \"id\": \"Row\",\n      \
             \"path\": \"{path}\",\n      \"green_marker\": \"{marker}\"\n    }}\n  ]\n}}\n"
        )
    }

    /// A throwaway tree carrying one runnable row: the payload the row's digest names, the registry
    /// that names it, and the marker record that names its marker. The gate command runs from the
    /// root this resolves, so the row sees the same tree the digest was taken over.
    fn runnable(tag: &str, gate: &str, budget_s: u64) -> (Scratch, PathBuf) {
        let scratch = Scratch::new(tag);
        // The payload first: the recorded identity is taken over the file the row names, so it must
        // exist before the registry that names its digest is written.
        scratch.write("proof/resource/Row_gate.py", "# the retired gate script\n");
        let root = scratch.0.clone();
        let hex = digest_of(&root, "proof/resource/Row_gate.py");
        scratch.write(
            "docs/contracts/artifacts.toml",
            &run_row_text(
                "proof/resource/Row_gate.py",
                "proof/resource/Row_gate.py",
                &hex,
                gate,
                budget_s,
            ),
        );
        scratch.write(
            CONFORMANCE,
            &conformance_text("proof/resource/Row_gate.py", "ROW_SOURCE_RESOURCE_OK"),
        );
        (scratch, root)
    }

    /// The manifest of a throwaway tree, as the row's registry.
    fn manifest_of(root: &Path) -> PathBuf {
        root.join("docs/contracts/artifacts.toml")
    }

    /// Every `run` control takes the adapter's scratch census: `adapter::controls::scratch_dirs`
    /// counts directories under this process's id, so a run in flight here would fail the leak
    /// check that module makes.
    fn serialise_scratch() -> std::sync::MutexGuard<'static, ()> {
        adapter::controls::serialise_scratch()
    }

    /// `gate run` must execute a recorded row and accept it on its own evidence: exit 0, the
    /// child's exit status carried through, and the marker the frozen record requires reported as
    /// printed. Without this the three controls below would also pass a runner that refuses
    /// everything.
    #[test]
    fn passing_row_exits_zero_with_its_marker() {
        let _serial = serialise_scratch();
        let (_scratch, root) = runnable("run-pass", "printf 'ROW_SOURCE_RESOURCE_OK'", 60);
        let outcome = run_row("proof/resource/Row_gate.py", &manifest_of(&root));
        assert_eq!(outcome.code, ExitCode::SUCCESS, "{}", outcome.verdict);
        assert!(outcome.report.is_empty(), "{:?}", outcome.report);
        for line in [
            "verdict=\"accepted\"",
            "exit=0\n",
            "timed_out=\"false\"",
            "marker=\"ROW_SOURCE_RESOURCE_OK\"",
            "marker_state=\"present\"",
        ] {
            assert!(
                outcome.verdict.contains(line),
                "{line} absent from:\n{}",
                outcome.verdict
            );
        }
    }

    /// Rejecting control: a row whose command exits non-zero. `gate run` must refuse it with exit 1,
    /// carry the child's own status in the verdict, and quote the tail the child printed — the
    /// attribution a receipt needs to say which gate went red and why.
    #[test]
    fn failing_row_exits_one_with_the_captured_tail() {
        let _serial = serialise_scratch();
        let (_scratch, root) = runnable(
            "run-fail",
            "printf 'starting'; printf 'ROW_EXTRACTION_DRIFT' >&2; exit 3",
            60,
        );
        let outcome = run_row("proof/resource/Row_gate.py", &manifest_of(&root));
        assert_eq!(outcome.code, ExitCode::from(1), "{}", outcome.verdict);
        for line in [
            "verdict=\"refused\"",
            "stage=\"gate-exit\"",
            "child_exit=3\n",
            "reason=\"exit 3\"",
            "exit=1\n",
        ] {
            assert!(
                outcome.verdict.contains(line),
                "{line} absent from:\n{}",
                outcome.verdict
            );
        }
        let report = outcome.report.join("\n");
        assert!(report.contains("ROW_EXTRACTION_DRIFT"), "{report}");
        assert!(report.contains("reported exit 3"), "{report}");
    }

    /// Rejecting control: a recorded identity that is not the identity on disk. The refusal must
    /// arrive *before* execution, and this is what shows it: the row's command would create the
    /// sentinel, and the sentinel is absent after the refusal.
    #[test]
    fn drifted_digest_is_refused_before_execution() {
        let _serial = serialise_scratch();
        let scratch = Scratch::new("run-drift");
        scratch.write("proof/resource/Row_gate.py", "# the retired gate script\n");
        let root = scratch.0.clone();
        scratch.write(
            "docs/contracts/artifacts.toml",
            &run_row_text(
                "proof/resource/Row_gate.py",
                "proof/resource/Row_gate.py",
                &"f".repeat(64),
                "touch ran.sentinel",
                60,
            ),
        );
        let outcome = run_row("proof/resource/Row_gate.py", &manifest_of(&root));
        assert_eq!(outcome.code, ExitCode::from(1), "{}", outcome.verdict);
        assert!(
            outcome.verdict.contains("stage=\"identity\"")
                && outcome
                    .verdict
                    .contains("digest mismatch: proof/resource/Row_gate.py"),
            "{}",
            outcome.verdict
        );
        assert!(
            !root.join("ran.sentinel").exists(),
            "the row's command ran against a source the registry no longer describes"
        );

        // Restore: the digest the file actually has runs the row, so the refusal came from the drift
        // and not from a runner that never executes anything.
        let hex = digest_of(&root, "proof/resource/Row_gate.py");
        scratch.write(
            "docs/contracts/artifacts.toml",
            &run_row_text(
                "proof/resource/Row_gate.py",
                "proof/resource/Row_gate.py",
                &hex,
                "touch ran.sentinel",
                60,
            ),
        );
        let restored = run_row("proof/resource/Row_gate.py", &manifest_of(&root));
        assert_eq!(restored.code, ExitCode::SUCCESS, "{}", restored.verdict);
        assert!(root.join("ran.sentinel").exists());
    }

    /// A gate that exits 0 without printing the marker the frozen record requires has printed no
    /// evidence that it ran the thing, so it is refused at exit 1. Without this control, exit status
    /// alone would decide every row and the marker column would be decoration.
    #[test]
    fn exit_zero_without_the_recorded_marker_is_refused() {
        let _serial = serialise_scratch();
        let (_scratch, root) = runnable("run-nomarker", "printf 'nothing to declare'", 60);
        let outcome = run_row("proof/resource/Row_gate.py", &manifest_of(&root));
        assert_eq!(outcome.code, ExitCode::from(1), "{}", outcome.verdict);
        assert!(
            outcome.verdict.contains("stage=\"green-marker\"")
                && outcome.verdict.contains("marker_state=\"missing\"")
                && outcome.verdict.contains("child_exit=0\n"),
            "{}",
            outcome.verdict
        );
        assert!(
            outcome.report.join("\n").contains("ROW_SOURCE_RESOURCE_OK"),
            "the refusal names the marker it wanted: {:?}",
            outcome.report
        );
    }

    /// The budget mapping `docs/epics/gate-driver.md:54` names: a child outliving `budget_s` is
    /// killed and exits 124, never 0 and never a missing-marker refusal. A runner that read a
    /// killed child as a pass would fail here and nowhere else.
    #[test]
    fn budget_expiry_exits_124() {
        let _serial = serialise_scratch();
        let (_scratch, root) = runnable("run-budget", "sleep 30", 1);
        let outcome = run_row("proof/resource/Row_gate.py", &manifest_of(&root));
        assert_eq!(outcome.code, ExitCode::from(124), "{}", outcome.verdict);
        assert!(
            outcome.verdict.contains("stage=\"budget\"")
                && outcome.verdict.contains("timed_out=\"true\"")
                && outcome.verdict.contains("budget of 1s")
                && outcome.verdict.contains("exit=124"),
            "{}",
            outcome.verdict
        );
    }

    /// An id the registry does not register is refused by the core's own variant, and a row whose
    /// path cannot be hashed is refused before execution too: a row with no identity has nothing to
    /// run against. Both are attributed, so a refusal names its cause rather than counting.
    #[test]
    fn unresolvable_rows_are_refused_by_cause() {
        let _serial = serialise_scratch();
        let scratch = Scratch::new("run-unknown");
        let root = scratch.0.clone();
        // An empty registry parses; the id asked for is absent from it.
        scratch.write("docs/contracts/artifacts.toml", "");

        let unknown = run_row("proof/resource/Ghost_gate.py", &manifest_of(&root));
        assert_eq!(unknown.code, ExitCode::from(1), "{}", unknown.verdict);
        assert!(
            unknown.verdict.contains("stage=\"identity\"")
                && unknown
                    .report
                    .join("\n")
                    .contains("unregistered: proof/resource/Ghost_gate.py"),
            "{}\n{:?}",
            unknown.verdict,
            unknown.report
        );

        scratch.write(
            "docs/contracts/artifacts.toml",
            &run_row_text(
                "proof/resource/Gone_gate.py",
                "proof/resource/Gone_gate.py",
                &"a".repeat(64),
                "true",
                60,
            ),
        );
        let unreadable = run_row("proof/resource/Gone_gate.py", &manifest_of(&root));
        assert_eq!(unreadable.code, ExitCode::from(1), "{}", unreadable.verdict);
        assert!(
            unreadable.verdict.contains("stage=\"hash-path\"")
                && unreadable
                    .report
                    .join("\n")
                    .contains("unreadable: proof/resource/Gone_gate.py"),
            "{}\n{:?}",
            unreadable.verdict,
            unreadable.report
        );
    }

    /// A row the marker record does not cover is judged on its exit status and says so. The verdict
    /// must name `unrecorded` rather than `present`: a runner that reported an unchecked marker as
    /// checked would claim a verification it never performed.
    #[test]
    fn row_without_a_recorded_marker_is_reported_unrecorded() {
        let _serial = serialise_scratch();
        let scratch = Scratch::new("run-norecord");
        scratch.write("gate.sh", "placeholder\n");
        let root = scratch.0.clone();
        let hex = digest_of(&root, "gate.sh");
        scratch.write(
            "docs/contracts/artifacts.toml",
            &run_row_text("gate.sh", "gate.sh", &hex, "true", 60),
        );
        let outcome = run_row("gate.sh", &manifest_of(&root));
        assert_eq!(outcome.code, ExitCode::SUCCESS, "{}", outcome.verdict);
        assert!(
            outcome.verdict.contains("marker_state=\"unrecorded\"")
                && outcome.verdict.contains("marker=\"\""),
            "{}",
            outcome.verdict
        );
    }

    /// The marker lookup reads one entry per row out of the record and never a neighbour's: with two
    /// rows recorded, each row is asked for its own marker and a row asking for a third name gets
    /// none. A lookup that took the first marker in the file would pass both greps below and fail
    /// every real row.
    #[test]
    fn green_marker_is_read_per_row() {
        let record = "{\n  \"gates\": [\n    {\n      \"id\": \"A\",\n      \
                      \"path\": \"proof/resource/A_gate.py\",\n      \
                      \"green_marker\": \"A_SOURCE_RESOURCE_OK\"\n    },\n    {\n      \
                      \"id\": \"B\",\n      \"path\": \"proof/resource/B_gate.py\",\n      \
                      \"green_marker\": \"B_SOURCE_RESOURCE_OK\"\n    }\n  ]\n}\n";
        assert_eq!(
            green_marker(record, "proof/resource/A_gate.py").as_deref(),
            Some("A_SOURCE_RESOURCE_OK")
        );
        assert_eq!(
            green_marker(record, "proof/resource/B_gate.py").as_deref(),
            Some("B_SOURCE_RESOURCE_OK")
        );
        assert_eq!(green_marker(record, "proof/resource/C_gate.py"), None);
    }

    /// The child's working directory is the manifest's root, and the child says so itself. The row's
    /// command is recorded relative to the repository root (`docs/contracts/artifacts.toml`, the
    /// `gate` column: "the exact command that checks this artefact, run from the repository root"), so
    /// a runner that spawns anywhere else runs a command that names nothing. The command is `pwd`
    /// plus a non-zero exit, so the refusal quotes the directory the child actually ran in.
    #[test]
    fn child_runs_in_the_manifest_root() {
        let _serial = serialise_scratch();
        let (_scratch, root) = runnable("run-cwd", "pwd; exit 3", 60);
        let outcome = run_row("proof/resource/Row_gate.py", &manifest_of(&root));
        assert_eq!(outcome.code, ExitCode::from(1), "{}", outcome.verdict);
        assert!(
            outcome
                .verdict
                .contains(&format!("cwd=\"{}\"", root.display())),
            "the block does not name the root it spawned in:\n{}",
            outcome.verdict
        );
        assert!(
            outcome
                .report
                .join("\n")
                .contains(&root.display().to_string()),
            "the child's own `pwd` is not the manifest root: {:?}",
            outcome.report
        );
    }

    /// The root of a *relative* manifest path resolves against the process directory and stays
    /// absolute. Three `parent()` steps on the relative `docs/contracts/artifacts.toml` overshoot the
    /// front of the path and return `""`, which `Command::current_dir` cannot spawn in — the defect
    /// this control pins. The absolute case and the too-shallow case are in the same table because
    /// they are the same function.
    #[test]    fn manifest_root_is_always_absolute() {
        let base = Path::new("/repo");
        let cases = [
            (
                Path::new("docs/contracts/artifacts.toml"),
                Path::new("/repo"),
            ),
            (
                Path::new("/elsewhere/docs/contracts/artifacts.toml"),
                Path::new("/elsewhere"),
            ),
            // Too shallow to have three parents: `base`, never the empty path.
            (Path::new("artifacts.toml"), Path::new("/repo")),
        ];
        for (manifest, expected) in cases {
            let root = root_for(manifest, base);
            assert_eq!(root, expected, "manifest {manifest:?}");
            assert!(root.is_absolute(), "manifest {manifest:?} -> {root:?}");
            assert!(!root.as_os_str().is_empty());
        }
    }

    // -- gate docs check --controls -----------------------------------------------------

    /// A registry row for the aggregate, in the seven keys `adapter::parse` admits. The recorded
    /// identity is the identity of the payload, which must exist before the manifest is written.
    fn control_row_text(id: &str, path: &str, sha256: &str, control: &str, budget_s: u64) -> String {
        format!(
            "[[artifact]]\nid = \"{id}\"\nkind = \"gate\"\npath = \"{path}\"\n\
             sha256 = \"{sha256}\"\ngate = \"printf GATE_RAN\"\ncontrol = \"{control}\"\n\
             budget_s = {budget_s}\n\n"
        )
    }

    /// A throwaway tree carrying the named rows, each with a payload the manifest names at the
    /// digest that payload has. Returns the scratch guard (which removes the tree) and the manifest
    /// path, so a test reads exactly what `run_controls` resolves its root from.
    ///
    /// One payload file is shared by every row, so a control that mutates it moves every row's
    /// registered identity at once — which is why the restore controls below are checked after the
    /// mutation is undone, not while it stands.
    fn registry(tag: &str, rows: &[(&str, &str, u64)]) -> (Scratch, PathBuf) {
        let scratch = Scratch::new(tag);
        scratch.write("proof/resource/Row_gate.py", "# the retired gate script\n");
        let root = scratch.0.clone();
        let hex = digest_of(&root, "proof/resource/Row_gate.py");
        let manifest: String = rows
            .iter()
            .map(|(id, control, budget_s)| {
                control_row_text(id, "proof/resource/Row_gate.py", &hex, control, *budget_s)
            })
            .collect();
        scratch.write("docs/contracts/artifacts.toml", &manifest);
        (scratch, manifest_of(&root))
    }

    /// The default `--controls` options over `manifest`: every row, the default invocation budget,
    /// and the throwaway copy under the process's own temporary directory.
    fn control_options() -> CheckOptions {
        CheckOptions {
            controls: true,
            ..CheckOptions::default()
        }
    }

    /// Only this row, with an explicit budget. `budget_s` 0 keeps the default.
    fn only(rows: &[&str], budget_s: u64) -> CheckOptions {
        let mut options = control_options();
        options.rows = rows.iter().map(|id| (*id).to_string()).collect();
        if budget_s > 0 {
            options.budget_s = budget_s;
            options.budget_given = true;
        }
        options
    }

    /// The positive every rejecting control below is measured against: a control that exits 0 and
    /// restores what it was given is `fired`, the summary says `1 fired`, and the aggregate exits 0.
    /// Without it, an aggregate that refused every row would pass the eight controls after it.
    #[test]
    fn a_control_that_exits_zero_and_restores_is_fired() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry("ctrl-pass", &[("proof/resource/Row_gate.py", "true", 60)]);
        let code = run_controls(&root_for(&manifest, &cwd()), &load(&manifest), &only(&["proof/resource/Row_gate.py"], 0));
        assert_eq!(code, ExitCode::SUCCESS);
    }

    /// Load a registry through the adapter, the way `docs check` does.
    fn load(manifest: &Path) -> Vec<Row> {
        adapter::load_manifest(manifest).expect("the scratch registry parses")
    }

    /// The directory a relative manifest path resolves against. Every scratch manifest here is
    /// absolute, so this only supplies the base `root_for` falls back to.
    fn cwd() -> PathBuf {
        std::env::temp_dir()
    }

    /// Rejecting control: a control command that cannot redden — it exits non-zero without requiring
    /// the refusal it exists for. The aggregate must refuse it, exit 1, and name the row and the
    /// child's own status. The sibling positives in the table below show this is the control's
    /// failure and not the runner's: the same row with a control that exits 0 is `fired`.
    #[test]
    fn a_control_that_cannot_redden_is_refused_by_name() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry(
            "ctrl-red",
            &[("proof/resource/Row_gate.py", "exit 7", 60)],
        );
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Row_gate.py"], 0),
        );
        assert_eq!(code, ExitCode::from(EXIT_REFUSED));
    }

    /// Rejecting control: a control that reports no exit status at all, because it was killed by a
    /// signal. Reading that as a fire would be the quiet success `AGENTS.md` §10 records, so it is
    /// refused with the same cause named.
    #[test]
    fn a_control_killed_by_a_signal_is_refused_by_name() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry(
            "ctrl-signal",
            &[("proof/resource/Row_gate.py", "kill -9 $$", 60)],
        );
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Row_gate.py"], 0),
        );
        assert_eq!(code, ExitCode::from(EXIT_REFUSED));
    }

    /// Rejecting control: a control that exits 0 having mutated the payload it was given and left it
    /// moved. The registry requires a control to mutate, redden and restore; this one skipped the
    /// third, so the aggregate's own restore check refuses it — the verdict that `fired` cannot be
    /// reached by exit status alone.
    #[test]
    fn a_control_that_does_not_restore_is_refused_by_name() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry(
            "ctrl-norestore",
            &[(
                "proof/resource/Row_gate.py",
                "printf 'x' >> proof/resource/Row_gate.py",
                60,
            )],
        );
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Row_gate.py"], 0),
        );
        assert_eq!(code, ExitCode::from(EXIT_REFUSED));
    }

    /// Rejecting control: a control naming a program the host does not have. `/bin/sh` resolves the
    /// command itself and reports 127, so this is a control that **ran and reported a non-zero
    /// status** — a refusal at exit 1, not a block: something was run and it did not fire. Pinning
    /// it here is what separates the two states; a runner that folded "not zero" into "could not
    /// run" would report a real refusal as an environment problem.
    #[test]
    fn a_control_naming_a_missing_program_is_refused_not_blocked() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry(
            "ctrl-missingprog",
            &[(
                "proof/resource/Row_gate.py",
                "/nonexistent-gate-tool --controls",
                60,
            )],
        );
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Row_gate.py"], 0),
        );
        assert_eq!(code, ExitCode::from(EXIT_REFUSED));
    }

    /// Rejecting control: a row whose registered `path` the copy does not carry at all — a payload
    /// under an excluded component, which is where a control that needs a build product lands. There
    /// is no identity to run against and none to restore, so the row is `blocked:unrunnable` by
    /// name and the aggregate is incomplete. It is never reported as a fire and never as a pass.
    #[test]
    fn a_row_the_copy_cannot_carry_is_blocked_by_name() {
        let _serial = serialise_scratch();
        let scratch = Scratch::new("ctrl-nopath");
        scratch.write("proof/resource/Row_gate.py", "# the retired gate script\n");
        let root = scratch.0.clone();
        // A payload inside `target/`, which the copy's exclusion rule leaves out — the same rule a
        // directory digest uses, so this row's recorded identity could never be re-read from a copy.
        let hex = digest_of(&root, "proof/resource/Row_gate.py");
        scratch.write(
            "docs/contracts/artifacts.toml",
            &control_row_text(
                "target/build/Row_gate.py",
                "target/build/Row_gate.py",
                &hex,
                "true",
                60,
            ),
        );
        let code = run_controls(&root, &load(&manifest_of(&root)), &only(&["target/build/Row_gate.py"], 0));
        assert_eq!(code, ExitCode::from(EXIT_INCOMPLETE));
    }

    /// Rejecting control: a control that outlives its own row budget. It is killed at the budget,
    /// reported `blocked:row-budget` with the elapsed time, and never counted as a fire: a killed
    /// child reported nothing, and its silence is not a verdict.
    #[test]
    fn a_control_over_its_row_budget_is_blocked_by_name() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry(
            "ctrl-rowbudget",
            &[("proof/resource/Row_gate.py", "sleep 30", 1)],
        );
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Row_gate.py"], 0),
        );
        assert_eq!(code, ExitCode::from(EXIT_INCOMPLETE));
    }

    /// Rejecting control: a row whose `control` column is empty. There is no command that could
    /// have reddened, so the row is `blocked:no-control` and the aggregate is incomplete. A row
    /// with no control is not a row with a passing control.
    #[test]
    fn a_row_without_a_control_is_blocked_by_name() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry("ctrl-none", &[("proof/resource/Row_gate.py", "  ", 60)]);
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Row_gate.py"], 0),
        );
        assert_eq!(code, ExitCode::from(EXIT_INCOMPLETE));
    }

    /// A row whose payload the copy does not carry at the recorded identity runs no control at all.
    /// The control would create a sentinel, so the sentinel's absence after the aggregate is what
    /// shows the refusal arrived before execution — and the reason names both identities, because a
    /// control run against a tree the registry no longer describes would redden for the wrong cause.
    #[test]
    fn a_copy_without_the_registered_identity_runs_no_control() {
        let _serial = serialise_scratch();
        let scratch = Scratch::new("ctrl-copydrift");
        scratch.write("proof/resource/Row_gate.py", "# the retired gate script\n");
        let root = scratch.0.clone();
        scratch.write(
            "docs/contracts/artifacts.toml",
            &control_row_text(
                "proof/resource/Row_gate.py",
                "proof/resource/Row_gate.py",
                &"f".repeat(64),
                "touch ran.sentinel",
                60,
            ),
        );
        // The manifest records an identity the payload does not have, so `run_controls` is reached
        // with the identity check bypassed on purpose: this control is about the copy's own re-read,
        // and the identity check's refusal of the same drift is `drifted_digest_is_refused_before_execution`.
        let rows = load(&manifest_of(&root));
        let code = run_controls(&root, &rows, &only(&["proof/resource/Row_gate.py"], 0));
        assert_eq!(code, ExitCode::from(EXIT_INCOMPLETE));
        assert!(
            !root.join("ran.sentinel").exists(),
            "the control ran against a payload the copy does not carry at the recorded identity"
        );
    }

    /// The batch bound. With an invocation budget smaller than the second row's own budget, the
    /// first row is attempted and the second is `unattempted:invocation-budget` — reported by name,
    /// counted in the summary, and never counted as a fire. Exit 3, so a one-of-two batch is not a
    /// pass over the registry.
    #[test]
    fn rows_the_invocation_budget_cannot_reach_are_unattempted_not_success() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry(
            "ctrl-batch",
            &[
                ("proof/resource/A_gate.py", "true", 1),
                ("proof/resource/B_gate.py", "true", 60),
            ],
        );
        // The rows share one payload path, so the manifest names `Row_gate.py`'s payload under two
        // ids; the point here is the budget arithmetic, which is per row and independent of the path.
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/A_gate.py", "proof/resource/B_gate.py"], 30),
        );
        assert_eq!(code, ExitCode::from(EXIT_INCOMPLETE));
    }

    /// A row declaring the registry's own per-row ceiling is reached under the default budget.
    ///
    /// The defect this answers: the throwaway copy is paid out of the invocation clock, so a default
    /// equal to the row's own `budget_s` left nothing for the copy — `540 s − 188 ms` on a warm page
    /// cache and `540 s − 43 s` on the cold one after a reboot, both `unattempted:invocation-budget`
    /// — and that row was the aggregate's own, so no batch could ever run its control.
    ///
    /// The row budget is written `MAX_BUDGET_S - 60` so the arithmetic, not the number, is the claim.
    ///
    /// Scope, stated because it bounds what this test can see: a scratch tree is a handful of files,
    /// so its copy costs under a millisecond and the run passes whenever the remaining budget is *at*
    /// the row's budget. The nonzero setup that made the defect observable is asserted separately and
    /// deterministically, by `the_default_budget_outlives_the_largest_registered_row_budget`, which
    /// holds the default strictly above every registered `budget_s`. Without that second test this one
    /// would pass under the defect it is written for.
    #[test]
    fn the_default_budget_attempts_a_row_at_the_maximum_row_budget() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry(
            "ctrl-maxrow",
            &[(
                "proof/resource/Row_gate.py",
                "true",
                adapter::MAX_BUDGET_S - 60,
            )],
        );
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Row_gate.py"], 0),
        );
        assert_eq!(
            code,
            ExitCode::SUCCESS,
            "a {}s row under the {}s default must be attempted, not unattempted",
            adapter::MAX_BUDGET_S - 60,
            DEFAULT_CONTROL_BUDGET_S
        );
    }

    /// The headroom, as an inequality over the registry that actually ships. The default must be
    /// *strictly* greater than every registered row's `budget_s`, because the copy's cost is nonzero
    /// and is charged first; equality is the defect, not a tie. This is the assertion that fails
    /// when the default is put back to the registry's own row ceiling, and it reads the real
    /// manifest rather than a fixture, so raising a row's `budget_s` cannot silently make it
    /// unreachable.
    ///
    /// Read through `CARGO_MANIFEST_DIR` rather than the working directory, so it is the same file
    /// whether `cargo test` is run from the workspace root or from this crate.
    #[test]
    fn the_default_budget_outlives_the_largest_registered_row_budget() {
        let manifest =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/contracts/artifacts.toml");
        let rows = adapter::load_manifest(&manifest).expect("the shipped registry parses");
        assert!(!rows.is_empty(), "{} registers no row", manifest.display());
        let largest = rows.iter().map(|row| row.budget_s).max().expect("a row");
        assert_eq!(
            largest, 540,
            "the registry's per-row ceiling moved; restate the headroom"
        );
        assert!(
            DEFAULT_CONTROL_BUDGET_S > largest,
            "the default {}s leaves no headroom above the largest registered row budget {}s, so the \
             copy spends the row's budget and the row is unattempted",
            DEFAULT_CONTROL_BUDGET_S,
            largest
        );
    }

    /// The same row under an explicit `--budget-s` below it stays `unattempted` and exits 3. The
    /// default was raised to make the ceiling reachable; it did not remove the bound, and a caller
    /// who declares 180 s still gets an honest incomplete rather than a child killed at 180 s and
    /// reported as a fire.
    #[test]
    fn an_explicit_budget_below_a_row_budget_still_refuses_to_attempt_it() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry(
            "ctrl-explicit",
            &[(
                "proof/resource/Row_gate.py",
                "true",
                adapter::MAX_BUDGET_S - 60,
            )],
        );
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Row_gate.py"], 180),
        );
        assert_eq!(code, ExitCode::from(EXIT_INCOMPLETE));
    }

    /// The default is derived from [`adapter::MAX_BUDGET_S`], so it cannot drift below the ceiling it
    /// exists to reach. Asserted as the relation rather than as two numbers, because the relation is
    /// the contract: a row may declare at most the maximum, and the default must cover the maximum.
    #[test]
    fn the_default_budget_is_the_admissible_maximum() {
        assert_eq!(
            DEFAULT_CONTROL_BUDGET_S,
            adapter::MAX_BUDGET_S,
            "a default below the ceiling cannot attempt a row declaring the ceiling"
        );
        assert_eq!(CheckOptions::default().budget_s, adapter::MAX_BUDGET_S);
    }

    /// The `unattempted` reason names the budget arithmetic *and* what the copy spent, because the
    /// two call for different remedies: a copy that ate the headroom is one row's chance, while 43 s
    /// of other rows' controls is the batch's. `remaining_ms` and `unattempted_reason` are the pure
    /// parts of the loop, separated so the arithmetic is checkable without a 540 s run.
    #[test]
    fn the_unattempted_reason_names_the_budget_and_the_copy() {
        assert_eq!(remaining_ms(600, 0), 600_000);
        assert_eq!(remaining_ms(600, 188), 599_812);
        assert_eq!(remaining_ms(540, 43_000), 497_000);
        // Elapsed past the budget saturates at zero rather than wrapping to a full budget again.
        assert_eq!(remaining_ms(540, u128::MAX), 0);
        assert_eq!(
            unattempted_reason(540, 540, remaining_ms(540, 43_000), 43_000),
            "needs 540s of control budget, 497s remain of the 540s invocation budget \
             (43000ms spent making the copy)"
        );
    }

    /// The copy root. A `--copy-root` is obeyed and not advised about — the caller named it, and
    /// telling them to name it is the noise a gate must not produce. Without one the copy goes to
    /// `TMPDIR` and the aggregate says so once, because that default is the tmpfs this box's
    /// failure register records filling with finished scratch trees.
    #[test]
    fn the_copy_root_is_the_declared_one_and_the_default_is_advised_about() {
        let declared = Path::new("/home/socol/oc-tmp");
        assert_eq!(
            copy_base(Some(declared)),
            (declared.to_path_buf(), false),
            "a declared copy root is used verbatim, with no advice"
        );
        let (base, advise) = copy_base(None);
        assert_eq!(base, std::env::temp_dir(), "the default is TMPDIR");
        assert!(
            advise,
            "the default is the tmpfs the failure register is about"
        );
    }

    /// A selection naming an id the registry does not register is refused by name, and no control
    /// and no copy is made: the aggregate cannot answer a question about a row that is not in this
    /// registry by running the rows that are.
    #[test]
    fn an_unregistered_row_id_is_refused_by_name() {
        let _serial = serialise_scratch();
        let (_scratch, manifest) = registry("ctrl-unknown", &[("proof/resource/Row_gate.py", "true", 60)]);
        let code = run_controls(
            &root_for(&manifest, &cwd()),
            &load(&manifest),
            &only(&["proof/resource/Ghost_gate.py"], 0),
        );
        assert_eq!(code, ExitCode::from(EXIT_REFUSED));
    }

    /// `--controls` is additive. The flags that only mean something with it are refused without it,
    /// so no invocation can ask for a batch of rows and silently receive an identity check; and a
    /// bare `gate docs check <manifest>` runs **no control at all**: the row's control would create
    /// a sentinel, and the sentinel's absence after the identity check is what shows the aggregate
    /// is off. The same manifest with `--controls` creates it, which is the positive that makes the
    /// absence evidence rather than a check that cannot fire.
    #[test]
    fn control_flags_are_additive_and_the_identity_path_runs_no_control() {
        for args in [
            vec!["--row".to_string(), "x".to_string()],
            vec!["--budget-s".to_string(), "10".to_string()],
            vec!["--copy-root".to_string(), "/tmp".to_string()],
            vec!["--keep-copy".to_string()],
        ] {
            assert_eq!(
                docs_command(&args),
                ExitCode::from(EXIT_ENVIRONMENT),
                "{args:?}"
            );
        }
        for args in [
            vec!["--controls".to_string(), "--row".to_string()],
            vec!["--controls".to_string(), "--budget-s".to_string()],
            vec![
                "--budget-s".to_string(),
                "601".to_string(),
                "--controls".to_string(),
            ],
            vec![
                "--budget-s".to_string(),
                "0".to_string(),
                "--controls".to_string(),
            ],
            vec!["--nope".to_string(), "--controls".to_string()],
        ] {
            assert_eq!(
                docs_command(&args),
                ExitCode::from(EXIT_ENVIRONMENT),
                "{args:?}"
            );
        }

        let _serial = serialise_scratch();
        let scratch = Scratch::new("ctrl-additive");
        scratch.write("proof/resource/Row_gate.py", "# the retired gate script\n");
        let root = scratch.0.clone();
        let hex = digest_of(&root, "proof/resource/Row_gate.py");
        // The sentinel is an absolute path **outside** the copy, because the control runs in the
        // copy and a sentinel written inside it would be removed with it before the test could look.
        let sentinel = std::env::temp_dir().join(format!("gate-ctrl-sentinel-{}", std::process::id()));
        scratch.write(
            "docs/contracts/artifacts.toml",
            &control_row_text(
                "proof/resource/Row_gate.py",
                "proof/resource/Row_gate.py",
                &hex,
                &format!("touch {}", sentinel.display()),
                60,
            ),
        );
        let manifest = manifest_of(&root).display().to_string();

        assert_eq!(docs_command(&[manifest.clone()]), ExitCode::SUCCESS);
        assert!(
            !sentinel.exists(),
            "the identity check ran the row's control"
        );
        assert_eq!(
            docs_command(&[manifest.clone(), "--controls".to_string()]),
            ExitCode::SUCCESS
        );
        assert!(
            sentinel.exists(),
            "--controls ran the control, so the absence above was the flag and not a runner that runs nothing"
        );
        let _ = std::fs::remove_file(&sentinel);
    }

    /// The identity check comes first and decides: a drifted digest refuses before any control runs,
    /// and `--controls` does not get past it. The row's control would create a sentinel.
    #[test]
    fn a_drifted_identity_refuses_the_aggregate_before_any_control_runs() {
        let _serial = serialise_scratch();
        let scratch = Scratch::new("ctrl-aggdrift");
        scratch.write("proof/resource/Row_gate.py", "# the retired gate script\n");
        let root = scratch.0.clone();
        scratch.write(
            "docs/contracts/artifacts.toml",
            &control_row_text(
                "proof/resource/Row_gate.py",
                "proof/resource/Row_gate.py",
                &"f".repeat(64),
                "touch ran.sentinel",
                60,
            ),
        );
        let manifest = manifest_of(&root).display().to_string();
        assert_eq!(
            docs_command(&[manifest, "--controls".to_string()]),
            ExitCode::from(EXIT_REFUSED)
        );
        assert!(
            !root.join("ran.sentinel").exists(),
            "a control ran against a source the registry no longer describes"
        );
    }

    /// The selection is applied in registry order and deduplicated, so the summary counts a row once
    /// however the flags were typed, and `--row` with no value is a usage error rather than a row
    /// named by the next flag.
    #[test]
    fn selection_is_registry_ordered_and_deduplicated() {
        let rows = load_registry_text(&["a", "b", "c"]);
        let requested: Vec<String> = ["c".to_string(), "a".to_string(), "c".to_string()].to_vec();
        let selected = select_rows(&rows, &requested).expect("all three are registered");
        assert_eq!(
            selected.iter().map(|row| row.id.as_str()).collect::<Vec<_>>(),
            vec!["a", "c"]
        );
        assert_eq!(select_rows(&rows, &[]).expect("every row").len(), 3);
        assert_eq!(
            select_rows(&rows, &["z".to_string()]).expect_err("unregistered"),
            vec!["z".to_string()]
        );
    }

    /// Three registry rows parsed from text, for the selection arithmetic alone.
    fn load_registry_text(ids: &[&str]) -> Vec<Row> {
        let hex = "0".repeat(64);
        let text: String = ids
            .iter()
            .map(|id| {
                format!(
                    "[[artifact]]\nid = \"{id}\"\nkind = \"gate\"\npath = \"p\"\n\
                     sha256 = \"{hex}\"\ngate = \"true\"\ncontrol = \"true\"\nbudget_s = 60\n\n"
                )
            })
            .collect();
        adapter::parse(&text).expect("syntax is valid")
    }
}
