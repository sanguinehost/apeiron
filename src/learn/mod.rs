//! Learning track: training, finetuning and quantization ported from `../spinor-ttl` and `../fold`
//!. One submodule per mechanism; each carries its focused
//! test `tests/learn_<x>.rs`.

/// GGUF v3 container writer: [`crate::format::gguf`]'s header, metadata and tensor types serialized
/// back to disk.
pub mod gguf;

/// Activation-weighted `Q4_K` block quantization: the seal core emitting bytes
/// [`crate::format::quant::dequant_q4_k_block`] reads unchanged.
pub mod seal;
