//! `gate docs check --controls`: the aggregate over the selected rows' recorded controls.
//!
//! `--controls` keeps the identity check first and unchanged, and then runs each selected row's
//! recorded `control` command **in a throwaway copy** of the repository
//! (`adapter::ControlCopy`), requiring two things of it and reporting both by name:
//!
//! 1. **the control fired** — the command exited 0 inside the row's own `budget_s`. Per
//!    `docs/contracts/artifacts.toml`, a `control` "is a command that must itself require a
//!    refusal: the `*_gate.py` controls mutate the real source, require the proof to go red with an
//!    attributable diagnostic, restore it, and exit 0 only if all three happened". The red's
//!    attribution is therefore that command's own contract, and **this aggregate does not check it**:
//!    the registry declares no control marker, so inventing one here would be a second spelling of a
//!    column that does not exist. What the aggregate checks for itself is the restoration:
//! 2. **the copy carries the registered identity** — the row's recorded `sha256` is re-read from the
//!    copy *before* the control runs (a copy that does not carry it runs no control at all) and
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

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::adapter::{self, Row};
use crate::cli::{
    Verdict, EXIT_ACCEPTED, EXIT_ENVIRONMENT, EXIT_INCOMPLETE, EXIT_REFUSED, TAIL_BYTES,
};
use crate::docs::CheckOptions;

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
pub(crate) const DEFAULT_CONTROL_BUDGET_S: u64 = adapter::MAX_BUDGET_S;

/// One per-row control state, as the aggregate prints it. Every state is a machine-readable token
/// and none of them reads as success except [`FIRED`]: a reader that greps for `state=` gets a
/// verdict, not a count.
pub(crate) const FIRED: &str = "fired";
pub(crate) const DID_NOT_FIRE: &str = "did-not-fire";
pub(crate) const BLOCKED_NO_CONTROL: &str = "blocked:no-control";
pub(crate) const BLOCKED_IDENTITY: &str = "blocked:identity";
pub(crate) const BLOCKED_UNRUNNABLE: &str = "blocked:unrunnable";
pub(crate) const BLOCKED_ROW_BUDGET: &str = "blocked:row-budget";
pub(crate) const UNATTEMPTED: &str = "unattempted:invocation-budget";
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
pub(crate) fn select_rows<'a>(
    rows: &'a [Row],
    requested: &[String],
) -> Result<Vec<&'a Row>, Vec<String>> {
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
pub(crate) struct ControlOutcome {
    pub(crate) id: String,
    pub(crate) control: String,
    pub(crate) budget_s: u64,
    pub(crate) state: &'static str,
    /// Whether the copy carried the recorded identity before the control ran.
    pub(crate) identity_before: bool,
    /// Whether the copy carried it afterwards; `None` when no control ran, so `none` never reads as
    /// "restored".
    pub(crate) identity_after: Option<bool>,
    pub(crate) code: Option<i32>,
    pub(crate) timed_out: bool,
    pub(crate) elapsed_ms: u128,
    pub(crate) stdout_sha256: String,
    pub(crate) stderr_sha256: String,
    pub(crate) reason: String,
}

impl ControlOutcome {
    /// A row the aggregate did not attempt, or refused before running anything.
    pub(crate) fn without_run(
        id: &str,
        control: &str,
        budget_s: u64,
        state: &'static str,
        reason: String,
    ) -> Self {
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
    pub(crate) fn fired(&self) -> bool {
        self.state == FIRED
    }

    /// The row's `key=value` block, in one fixed field order with every value quoted or a number, so
    /// a receipt parses it without a JSON reader. Absent values are spelled (`"none"`, `"false"`)
    /// rather than omitted: a field that is missing is a field the aggregate did not report, and a
    /// reader must not have to tell that from a field reported as absent.
    pub(crate) fn verdict(&self, copy_root: &Path) -> String {
        let mut verdict = Verdict::default();
        verdict.fields.push(Verdict::quoted("id", &self.id));
        verdict.fields.push(Verdict::quoted("state", self.state));
        verdict
            .fields
            .push(Verdict::quoted("control", &self.control));
        verdict
            .fields
            .push(Verdict::number("budget_s", self.budget_s));
        verdict.fields.push(Verdict::quoted(
            "copy_root",
            &copy_root.display().to_string(),
        ));
        verdict.fields.push(Verdict::quoted(
            "identity_before",
            &self.identity_before.to_string(),
        ));
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
        verdict
            .fields
            .push(Verdict::number("elapsed_ms", self.elapsed_ms));
        verdict
            .fields
            .push(Verdict::quoted("stdout_sha256", &self.stdout_sha256));
        verdict
            .fields
            .push(Verdict::quoted("stderr_sha256", &self.stderr_sha256));
        verdict.fields.push(Verdict::quoted("reason", &self.reason));
        verdict
            .fields
            .push(Verdict::number("exit", self.exit_code()));
        verdict.render()
    }

    /// The exit this row would produce on its own. `fired` alone is 0; a control that ran and did
    /// not fire is 1; a row the aggregate could not run, or did not reach, is 3.
    pub(crate) fn exit_code(&self) -> u8 {
        match self.state {
            FIRED => EXIT_ACCEPTED,
            DID_NOT_FIRE => EXIT_REFUSED,
            _ => EXIT_INCOMPLETE,
        }
    }
}

/// The value a field takes when the aggregate has nothing to report for it. A string, not an empty
/// field, so `identity_after=none` is distinguishable from `identity_after=false`.
pub(crate) const NONE: &str = "none";

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
pub(crate) fn run_one_control(copy_root: &Path, row: &Row) -> ControlOutcome {
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

    let record =
        match adapter::run_captured_with_tail(&row.control, copy_root, row.budget_s, TAIL_BYTES) {
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
pub(crate) fn remaining_ms(budget_s: u64, elapsed_ms: u128) -> u64 {
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
pub(crate) fn unattempted_reason(
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
pub(crate) fn copy_base(declared: Option<&Path>) -> (PathBuf, bool) {
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
pub(crate) fn run_controls(root: &Path, rows: &[Row], options: &CheckOptions) -> ExitCode {
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
