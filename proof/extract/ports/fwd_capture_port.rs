//! Extraction unit for the engine-sourced activation-Gram capture
//! (`src/learn/fold/stats/capture.rs`).
//!
//! # Merge (`proof/extract/lib.rs`)
//!
//! ```text
//! #[path = "ports/fwd_capture_port.rs"]
//! pub mod fwd_capture_port;
//! ```
//!
//! `src/learn/fold/stats/mod.rs` is already declared in this unit by `fold_port.rs`
//! (`#[path = "../../src/learn/fold/stats/mod.rs"] pub mod stats;`), so no `#[path]` declaration is
//! needed here and the extracted module path is `fold_port.stats.capture`. The same shape is
//! `ports/limb_port.rs`.
//!
//! The roots reach the production engine path ([`Grams::accumulate_tap`], its `add_gram_stage`
//! callee and therefore the limb product), the constructor and the finishing pass. The exact
//! reference path [`Grams::accumulate`] is not rooted: its body drives `FoldRows` through the
//! thread pool (`crate::engine::chart::pool`, opaque in this unit) and its `fold_upper_rows` is a
//! three-nested-loop scalar product the pinned translator rejects; it is the test-only oracle the
//! engine path is measured against (`tests/learn_fold_capture_engine.rs`).
//!
//! # Start-from roots the merge adds to `proof/regen.sh`
//!
//! ```text
//! --start-from crate::fwd_capture_port::word_stats_capture_new \
//! --start-from crate::fwd_capture_port::word_stats_capture_accumulate_tap \
//! --start-from crate::fwd_capture_port::word_stats_capture_finish \
//! ```

use crate::fold_port::stats::capture::{CaptureError, Grams, LayerGrams, Normalisation};
use crate::learn::grad::model::Config;

/// [`Grams::new`] at the concrete `Config`.
pub fn word_stats_capture_new(cfg: &Config) -> Result<Grams, CaptureError> {
    Grams::new(cfg)
}

/// [`Grams::accumulate_tap`] over the engine's `[layer][stage][column][width]` rows.
pub fn word_stats_capture_accumulate_tap(
    grams: &mut Grams,
    rows: &[Vec<Vec<Vec<i64>>>],
) -> Result<(), CaptureError> {
    grams.accumulate_tap(rows)
}

/// [`Grams::finish`] at the concrete normalisation.
pub fn word_stats_capture_finish(
    grams: &Grams,
    normalisation: Normalisation,
) -> Result<Vec<LayerGrams>, CaptureError> {
    grams.finish(normalisation)
}
