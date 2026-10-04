//! `gate docs check`: the invocation, and the identity check that decides it.
//!
//! The lock serialises concurrent checks over one tree: two checks hashing the same files while a
//! third edits them would each report a digest the other never saw. A lock still held after
//! [`crate::registry::LOCK_WAIT_S`] is reported as a blocker, never broken.
//!
//! `options.controls` off is the whole of the pre-`--controls` behaviour: the identity check, its
//! report and its two exits, unchanged and in that order. With `--controls` the identity check
//! still runs first and still decides — a refused identity runs no control, because a control
//! exercised against a tree the registry no longer describes would redden for the wrong reason —
//! and the aggregate in `crate::aggregate` takes over only after the identity check has accepted.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::adapter;
use crate::aggregate::{run_controls, DEFAULT_CONTROL_BUDGET_S};
use crate::cli::EXIT_ENVIRONMENT;
use crate::registry::{current_dir, lock_path, root_for, DEFAULT_MANIFEST, LOCK_WAIT_S};
use crate::report::{check, print_refusal};

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
pub(crate) fn docs_check(manifest_path: &Path, options: &CheckOptions) -> ExitCode {
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
/// The parsed `gate docs check` invocation.
///
/// `controls` is off by default, and every other field is refused without it, so `gate docs check`
/// and `gate docs check <manifest>` reach the identity check by exactly the path they reached it
/// before this flag existed. `budget_given` distinguishes an explicit `--budget-s` from the default,
/// because `--budget-s 540` is an error without `--controls` while the default is not a request.
#[derive(Debug)]
pub(crate) struct CheckOptions {
    pub(crate) controls: bool,
    /// Row ids to attempt, in the registry's own order. Empty means every row.
    pub(crate) rows: Vec<String>,
    pub(crate) budget_s: u64,
    pub(crate) budget_given: bool,
    pub(crate) copy_root: Option<PathBuf>,
    pub(crate) keep_copy: bool,
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
    pub(crate) fn usage(&self, message: &str) -> ExitCode {
        eprintln!("gate docs check: {message}");
        eprintln!("{USAGE}");
        ExitCode::from(EXIT_ENVIRONMENT)
    }
}

pub(crate) const USAGE: &str = "usage: gate docs check [manifest] [--controls] [--row <id>]... [--budget-s <n>]\n\
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
pub(crate) fn docs_command(args: &[String]) -> ExitCode {
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
                Some(Err(_)) => {
                    return options.usage("--budget-s is not a whole number of seconds")
                }
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
