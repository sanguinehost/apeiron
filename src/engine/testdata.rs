//! Portable resolution of the model and fixture files the tests read.
//!
//! The GGUF checkpoints, the tokenizer and the prompt-id files live under one data root (the parent
//! of `models/`). `$APEIRON_MODELS` names that root; when it is unset, `$HOME` is used. A file that
//! does not resolve yields `None`, so a machine without the fixture skips the test instead of
//! failing.

use std::path::PathBuf;

/// Resolve a fixture path relative to the data root, or `None` when the file is absent.
///
/// `relative` (for example `models/gguf/qwen05b-Q4_K_M.gguf`) is joined onto `$APEIRON_MODELS`
/// when that variable is set, otherwise onto `$HOME`. The path is returned as an owned
/// `String` because the GGUF, engine and BPE loaders take `&str`. `None` on absence keeps each
/// caller's skip-on-absent behavior.
#[doc(hidden)]
#[must_use]
pub fn model_path(relative: &str) -> Option<String> {
    let root =
        PathBuf::from(std::env::var_os("APEIRON_MODELS").or_else(|| std::env::var_os("HOME"))?);
    let candidate = root.join(relative);
    candidate
        .exists()
        .then(|| candidate.to_string_lossy().into_owned())
}
