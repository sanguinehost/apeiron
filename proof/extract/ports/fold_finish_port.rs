//! Extraction unit for the run glue's `convert` (`src/learn/fold/run/convert.rs`), reached by
//! `#[path]` so the extracted bodies are the real source; each `word_*` wrapper fixes the
//! monomorphic root and calls the real function, restating no formula.
//!
//! The `learn::fold` mirror in `fold_port.rs` re-exports `convert` as `learn::fold::run::convert`,
//! the absolute path the sources name.

/// The real `src/learn/fold/run/convert.rs`: the engine-weight conversion and the `GGUF` scalar
/// encoders.
#[path = "../../../src/learn/fold/run/convert.rs"]
pub mod convert;

use convert::ConvertError;

// ── `learn::fold::run::convert` ───────────────────────────────────────────────────────────────

/// Whole little-endian `f16`s as Q32.32 words.
pub fn word_run_convert_decode_f16(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    convert::decode_f16(raw)
}

/// Whole little-endian `bf16`s as Q32.32 words.
pub fn word_run_convert_decode_bf16(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    convert::decode_bf16(raw)
}

/// Whole little-endian `f32`s as Q32.32 words.
pub fn word_run_convert_decode_f32(raw: &[u8]) -> Result<Vec<i64>, ConvertError> {
    convert::decode_f32(raw)
}
