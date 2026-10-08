//! Learning track: training, finetuning and quantization ported from `../spinor-ttl` and `../fold`
//! (`docs/epics/neural-rebase.md`, stage 3). One submodule per mechanism; each carries its focused
//! test `tests/learn_<x>.rs`.

/// GGUF v3 container writer: [`crate::format::gguf`]'s header, metadata and tensor types serialized
/// back to disk.
pub mod gguf;
