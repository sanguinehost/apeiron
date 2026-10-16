//! Direct SPD solve and weighted Gram fold in Q32.32 `i64`: the primitives the weight-fit steps share.

/// Diagonal-equilibrated blocked `Cholesky` SPD solve in Q32.32 `i64` with one exact-residual
/// refinement step and a threaded trailing update.
pub mod chol;

/// Weighted exact-integer SYRK (`G = Σ_o w_o·x_o x_oᵀ` in Q32.32) and the `A ∘ hl` hadamard
/// factor — the blocked, threaded `A`-formation of the `c_step` metric.
pub mod syrk;
