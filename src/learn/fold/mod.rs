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
/// Per-block nearest-codeword ICM passes over the E8 codes of one and two planes, and the two-plane `r` refit.
pub mod icm;
/// Bit-exact binary16 rounding and the hierarchical row/column/diagonal scale quantisers.
pub mod quant;
/// `fwht128`, `compose`, the refit objective and the closed-form `c`, `r`, `d` steps.
pub mod steps;
