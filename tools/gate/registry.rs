//! Where the artefact registry lives and which tree its rows are read against.
//!
//! A document a gate parses belongs in `docs/contracts/` (`AGENTS.md` §2), so the default manifest
//! is `docs/contracts/artifacts.toml`, not `docs/artifacts.toml`. The root a row's paths resolve
//! against is derived from the manifest path rather than from the working directory, and the lock
//! is taken off the repository so a check never writes into the tree it is hashing.

use std::path::{Path, PathBuf};

use crate::adapter::Row;
use gate::{Digest, Manifest, Registered};

/// Where the registry lives when no manifest path is given. A document a gate parses belongs in
/// `docs/contracts/` (`AGENTS.md` §2), so this is not `docs/artifacts.toml`.
pub(crate) const DEFAULT_MANIFEST: &str = "docs/contracts/artifacts.toml";
/// How long `docs check` waits for another check over the same tree to finish before reporting the
/// lock as a blocker. Well under a gate budget: a check that cannot start is a blocker, not a wait.
pub(crate) const LOCK_WAIT_S: u64 = 30;

/// The registry lock. Off the repository so a check never writes into the tree it is hashing, and
/// beside the lane locks the `proof/resource` gates already take.
pub(crate) fn lock_path() -> PathBuf {
    PathBuf::from("/tmp/opencode/gate-docs-check.lock")
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
pub(crate) fn root_for(manifest_path: &Path, base: &Path) -> PathBuf {
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
pub(crate) fn current_dir() -> Result<PathBuf, String> {
    std::env::current_dir().map_err(|error| error.to_string())
}

/// The core's view of a parsed registry. `refs` is empty because the registry schema declares no
/// dependency column; the columns it does declare are the payload identity, which the core checks.
pub(crate) fn to_manifest(rows: &[Row]) -> Manifest {
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
