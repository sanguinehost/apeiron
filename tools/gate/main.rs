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
fn root_for(manifest_path: &Path) -> PathBuf {
    manifest_path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
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
            Err(error) => report.unreadable.push(format!("{}: {error}", row.path)),
        }
    }
    report
}

fn print_refusal(refusal: &Refusal) {
    match refusal {
        Refusal::DuplicateRegistration(id) => println!("  duplicate id: {id}"),
        Refusal::IdentityMismatch {
            id,
            expected,
            actual,
        } => println!("  digest mismatch: {id} recorded {expected}, on disk {actual}"),
        Refusal::MissingDependency { from, to } => {
            println!("  missing dependency: {from} requires {to}");
        }
        Refusal::Unregistered(id) => println!("  unregistered: {id}"),
        Refusal::DanglingReference { from, to } => {
            println!("  dangling reference: {from} -> {to}");
        }
    }
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

    let root = root_for(manifest_path);
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
        println!("  unreadable: {path}");
    }
    drop(lock);
    if report.accepted() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
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
        Some(other) => {
            eprintln!("gate: unknown subcommand `{other}`; known: docs, hold");
            ExitCode::from(2)
        }
        None => {
            eprintln!(
                "usage: gate <subcommand>\n  docs check   load the artefact registry and report refusals\n  hold         refuse while a HOLD marker is in force"
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
}
