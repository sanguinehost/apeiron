//! Fold-transfer pipeline on the crate build path: activation Grams, the exact corrector, the
//! held-out evaluator and the byte budget.

/// Exact 27B byte accounting at a target bits-per-weight, over the reference class table.
pub mod budget;
/// Per-layer activation Grams of the four linears, accumulated from the forward tape.
pub mod capture;
/// Symmetric eigendecomposition (cyclic Jacobi) and the exact rank-`rank` corrector in the `Hl` metric.
pub mod eigh;
/// Held-out KL, TV, top-1 and NLL of a student against a teacher on the ported model.
pub mod heldout;
/// Bit-exact binary16 rounding and the hierarchical row/column/diagonal scale quantisers.
pub mod quant;
