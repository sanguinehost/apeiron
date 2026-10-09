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

/// Kneser–Ney byte count language model over the ported counting face [`crate::freq::FreqMap`].
pub mod countlm;

/// Forward-only finetuning adapter over token contexts via Kneser–Ney count refinement.
pub mod foldadapt;

/// GPTQ error-feedback quantization to balanced-trit planes and affine carriers.
pub mod gptq;

/// One-bit arms on the calibration Gram: `Q1_0` sign planes, sigma-delta, GPTQ and Hadamard.
pub mod onebit;

/// Exact-integer first-order sigma-delta: the executed body of the `sd1` arms, refined to
/// `Numeric.SigmaDelta` (the ternary-hadamard note Part II, W8).
pub mod sigma_delta;

/// Streaming Gram accumulation and direct SPD ridge solve: the closed-form linear head.
pub mod ridge;

/// Exact-integer Gram fold: `g += XᵀX` over a batch of `i8` rows, checked `i64`.
pub mod gram;
