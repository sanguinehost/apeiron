//! `gate run <row-id>`: execute one registry row under its own budget.
//!
//! It resolves the row, refuses it through `core::admit` unless the recorded `sha256` is the
//! identity on disk *right now*, and only then runs the row's `gate` command through
//! `adapter::run_captured_with_tail` under that row's `budget_s`, from the repository root the
//! manifest resolves to. It reads no `control` column: a control mutates the real source and
//! belongs to the task that owns it. The child streams are relayed by the adapter as it runs, so
//! `gate run` adds a verdict block and, on a refusal, the tail of what the child printed; the
//! verdict block is `key=value` lines in a fixed order with every value quoted, so a receipt parses
//! it without a JSON reader.
//!
//! The row's green marker is not a registry column. It is read from the frozen record
//! `tools/gate/conformance.json`, which keys `gates[].green_marker` by each gate row's `path`
//! (`docs/epics/gate-driver.md:20-23`: the driver must reproduce "the same green markers"). A row
//! the record carries no marker for is reported as `marker_state=unrecorded` and judged on its
//! exit status alone; the verdict says so rather than implying a marker check that did not happen.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::adapter;
use crate::cli::{Verdict, EXIT_ACCEPTED, EXIT_BUDGET, EXIT_ENVIRONMENT, EXIT_REFUSED, TAIL_BYTES};
use crate::registry::{current_dir, root_for, to_manifest, DEFAULT_MANIFEST};
use crate::report::{refusal_text, unreadable_text};
use gate::{Digest, Submitted};

/// The frozen marker record, relative to the root the manifest resolves to.
/// `tools/gate/conformance.json` keys each gate row's `green_marker` by that row's `path`.
pub(crate) const CONFORMANCE: &str = "tools/gate/conformance.json";
/// What `gate run` decided. `verdict` goes to stdout as the machine-readable block; `report` goes to
/// stderr as the human refusals and the child's tail, so a piped stdout carries the block alone.
#[derive(Debug)]
pub(crate) struct Outcome {
    pub(crate) code: ExitCode,
    pub(crate) verdict: String,
    pub(crate) report: Vec<String>,
}

impl Outcome {
    /// `gate run` accepted the row: the command exited 0 inside its budget and printed the marker
    /// the frozen record requires, or carries no marker at all (stated in the block).
    pub(crate) fn accepted(verdict: String) -> Outcome {
        Outcome {
            code: ExitCode::from(EXIT_ACCEPTED),
            verdict,
            report: Vec::new(),
        }
    }

    /// Every other verdict: the exit code, the block, and what a reader has to see to attribute it.
    pub(crate) fn refused(code: u8, verdict: String, report: Vec<String>) -> Outcome {
        Outcome {
            code: ExitCode::from(code),
            verdict,
            report,
        }
    }

    /// A block with `id`, `manifest`, `cwd` and `command` already bound: the fields every verdict
    /// names, whether it accepted or refused, so a refusal says which row and which command.
    pub(crate) fn block(
        row_id: &str,
        manifest: &Path,
        root: &Path,
        command: &str,
        budget_s: u64,
    ) -> Verdict {
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
pub(crate) fn green_marker(record: &str, path: &str) -> Option<String> {
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
pub(crate) enum Marker {
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
    pub(crate) fn state(&self) -> &'static str {
        match self {
            Marker::Present(_) => "present",
            Marker::Missing(_) => "missing",
            Marker::Unrecorded => "unrecorded",
        }
    }

    /// The marker the record requires, or an empty value when it registers none.
    pub(crate) fn required(&self) -> &str {
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
pub(crate) fn run_row(row_id: &str, manifest_path: &Path) -> Outcome {
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
pub(crate) fn tail_printed(tail: &str, marker: &str) -> bool {
    !marker.is_empty() && tail.contains(marker)
}

/// `gate run <row-id> [--manifest <path>]`. `--manifest` may come before or after the row id; the
/// default is the registry `docs check` reads. Two row ids is a usage error rather than a silent
/// choice of one.
pub(crate) fn run_command(args: &[String]) -> ExitCode {
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
