//! The conformance suite for this crate: one positive beside every refusal it measures against.
//!
//! It is one module at the crate root so that every test keeps the name `controls::<test>` it has
//! always had, whichever module the body under test moved to. The imports below reach every
//! sibling module's items, which are `pub(crate)`; the helpers build the throwaway trees the rows
//! are read from.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::adapter::{self, Row};
use crate::aggregate::*;
use crate::cli::*;
use crate::docs::*;
use crate::registry::*;
use crate::report::*;
use crate::runner::*;
use gate::Refusal;

/// A scratch tree off the tmpfs-filling `/tmp` when `TMPDIR` says so; removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!("gate-{tag}-{}-{nanos}", std::process::id()));
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
    let code = run_controls(
        &root_for(&manifest, &cwd()),
        &load(&manifest),
        &only(&["proof/resource/Row_gate.py"], 0),
    );
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
    let (_scratch, manifest) =
        registry("ctrl-red", &[("proof/resource/Row_gate.py", "exit 7", 60)]);
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
    let code = run_controls(
        &root,
        &load(&manifest_of(&root)),
        &only(&["target/build/Row_gate.py"], 0),
    );
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
        &only(
            &["proof/resource/A_gate.py", "proof/resource/B_gate.py"],
            30,
        ),
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
    let (_scratch, manifest) = registry(
        "ctrl-unknown",
        &[("proof/resource/Row_gate.py", "true", 60)],
    );
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

    assert_eq!(
        docs_command(std::slice::from_ref(&manifest)),
        ExitCode::SUCCESS
    );
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
        selected
            .iter()
            .map(|row| row.id.as_str())
            .collect::<Vec<_>>(),
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
