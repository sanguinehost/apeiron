//! Calibration and evaluation statistics the NanoQuant path consumes (`stats`), the engine-weight
//! conversions it writes students through (`run`), and the symmetric eigendecomposition and scale
//! quantisers the solvers share (`fit`). The E8 fold fit that gave the module its name is removed.

/// Symmetric eigendecomposition (cyclic Jacobi), the low-rank corrector it backs, and the binary16
/// and hierarchical scale quantisers.
pub mod fit;
/// Conversions between the engine-loaded `Q32.32` weights and student `GGUF` bytes.
pub mod run;
/// Activation Grams, the output-side Hessian sketch, held-out KL and the `--kl-divergence-base` reader.
pub mod stats;
