//! `gate hold`: the hold preflight required by the owner-layer authority
//! (`~/Workspace/AGENTS.md`, "Authority, hold, momentum").
//!
//! It walks up from the current directory looking for a `HOLD` marker, prints every marker it
//! finds, and exits non-zero while any is in force. Run it before any write or dispatch. Absence
//! of a marker is release. It is deliberately not a silent success: a hold in force prints its own
//! reason and scope.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Levels above the current directory to inspect. Bounded so a stray marker at the filesystem root
/// is not reached by accident.
pub(crate) const MAX_LEVELS: usize = 8;
/// Every `HOLD` marker at or above `start`, nearest first.
pub(crate) fn holds_from(start: &Path) -> Vec<PathBuf> {
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
pub(crate) fn hold() -> ExitCode {
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
