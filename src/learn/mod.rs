//! Learning track: training, finetuning and quantization ported from `../spinor-ttl` and `../fold`
//!. One submodule per mechanism; each carries its focused
//! test `tests/learn_<x>.rs`.

/// GGUF v3 container writer: [`crate::format::gguf`]'s header, metadata and tensor types serialized
/// back to disk.
pub mod gguf;

/// llama.cpp imatrix `GGUF` reader: per-column activation importance vectors.
pub mod imatrix;

/// Activation-weighted `Q4_K` block quantization: the seal core emitting bytes
/// [`crate::format::quant::dequant_q4_k_block`] reads unchanged.
pub mod seal;

/// Kneser–Ney byte count language model over the vendored counting face [`crate::freq::FreqMap`].
pub mod countlm;

/// Streaming Gram accumulation and direct SPD ridge solve: the closed-form linear head.
pub mod ridge;
