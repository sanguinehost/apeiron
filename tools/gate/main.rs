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
fn docs_check(manifest_path: &Path) -> ExitCode {
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
    drop(lock);
    if report.accepted() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("hold") => hold(),
        Some("docs") => {
            match args.get(1).map(String::as_str) {
                Some("check") => {
                    let manifest = match args.get(2) {
                        Some(path) => PathBuf::from(path),
                        None => PathBuf::from(DEFAULT_MANIFEST),
                    };
                    docs_check(&manifest)
                }
                _ => {
                    eprintln!("usage: gate docs check [manifest]\n  manifest defaults to {DEFAULT_MANIFEST}");
                    ExitCode::from(2)
                }
            }
        }
        Some("run") => run_command(&args[1..]),
        Some(other) => {
            eprintln!("gate: unknown subcommand `{other}`; known: docs, hold, run");
            ExitCode::from(2)
        }
        None => {
            eprintln!(
                "usage: gate <subcommand>\n  docs check   load the artefact registry and report refusals\n  \
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
    #[test]
    fn manifest_root_is_always_absolute() {
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
}
