//! Learning track: training, finetuning and quantization ported from `../spinor-ttl` and `../fold`
//! One submodule per mechanism; each carries its focused
//! test `tests/learn_<x>.rs`.

/// GGUF v3 container writer: [`crate::format::gguf`]'s header, metadata and tensor types serialized
/// back to disk.
pub mod gguf;

/// The NanoQuant binary low-rank factorisation (`SamsungLabs/NanoQuant` @ `a9e0a43`): ADMM to
/// `diag(s_post)·U·V·diag(s_pre)` with sign factors.
pub mod nanoquant;

/// Backward passes of the ported model: dense adjoints, attention with `RoPE`, KL/TV heads.
pub mod grad;

/// Calibration Grams, held-out evaluation and the student-GGUF conversions NanoQuant uses.
pub mod fold;

/// Q32.32 SPD solve, inverse factor and weighted Gram fold shared by NanoQuant and the solvers.
pub mod solve;
