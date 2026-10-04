//! The identity report: what `docs check` found, split by the cause.
//!
//! A repeated id and a drifted digest are different failures with different fixes, so they are
//! counted and printed separately. This module does no IO beyond hashing the registered paths: it
//! runs no gate and reads no marker, so it says whether the registry still describes the tree, not
//! whether a gate passed. Every verdict in it is a `gate::Refusal` from the pure core.

use std::path::Path;

use crate::adapter::{self, Row};
use crate::registry::to_manifest;
use gate::{validate, Digest, Refusal, Submitted};

/// What `docs check` found, split by the cause so a report cannot merge them: a repeated id and a
/// drifted digest are different failures with different fixes.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Report {
    pub(crate) loaded: usize,
    pub(crate) duplicates: Vec<Refusal>,
    pub(crate) mismatches: Vec<Refusal>,
    /// Paths the adapter could not hash. Not a core verdict: there is no identity to compare.
    pub(crate) unreadable: Vec<String>,
}

impl Report {
    /// Clean when nothing was refused. A load of zero rows is *not* clean: an empty registry is a
    /// registry that has lost its records, and reporting it green would be the quiet success the
    /// failure register records.
    pub(crate) fn accepted(&self) -> bool {
        self.loaded > 0
            && self.duplicates.is_empty()
            && self.mismatches.is_empty()
            && self.unreadable.is_empty()
    }
}

/// Check every row against the tree at `root`, through the adapter, and let `core` decide. Does no
/// IO beyond hashing the registered paths: it runs no gate and reads no marker, so it says whether
/// the registry still describes the tree, not whether a gate passed.
pub(crate) fn check(root: &Path, rows: &[Row]) -> Report {
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
pub(crate) fn unreadable_text(path: &str, error: &str) -> String {
    format!("unreadable: {path}: {error}")
}

/// One refusal as the report prints it. Every variant names its id or its pair of ids, so a
/// diagnostic can be attributed to a row without re-reading the manifest.
pub(crate) fn refusal_text(refusal: &Refusal) -> String {
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

pub(crate) fn print_refusal(refusal: &Refusal) {
    println!("{}", refusal_text(refusal));
}
