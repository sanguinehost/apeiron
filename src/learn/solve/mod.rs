//! Direct SPD solve and weighted Gram fold in Q32.32 `i64`: the primitives the weight-fit steps share.

/// Diagonal-equilibrated blocked `Cholesky` SPD solve in Q32.32 `i64` with one exact-residual
/// refinement step and a pooled trailing update.
pub mod chol;

/// Weighted exact-integer SYRK (`G = Σ_o w_o·x_o x_oᵀ` in Q32.32) and the `A ∘ hl` hadamard
/// factor — the blocked, pooled `A`-formation of the `c_step` metric.
pub mod syrk;

/// Blocked, pooled, exact-accumulation Q32.32 matrix products `C = A·B` ([`gemm::gemm_nn`]) and
/// `C = A·Bᵀ` ([`gemm::gemm_nt`]) — the one GEMM the low-rank corrector's `w1 = ΔW·Hl` and
/// `m = w1·ΔWᵀ` need, with a single round-to-nearest per entry.
pub mod gemm;

/// The `f64` ↔ Q32.32 boundary shared by every caller of the primitives above: the range-checked
/// rounding, the `wsyrk` panel shift and the power-of-two fill scale.
pub mod q32;

/// Top-`r` eigenpairs of a symmetric PSD Q32.32 matrix ([`eig::top_eigh`]) by block subspace
/// iteration with a Rayleigh–Ritz acceleration and an integer Jacobi on the small block — the
/// corrector's replacement for the full `eigh`.
pub mod eig;

/// The upper-triangular inverse feedback factor `U` with `UᵀU = H⁻¹` ([`inv::inv_factor_upper`]):
/// diagonal equilibration, the blocked `Cholesky` of [`chol`] on the reversed matrix, and a blocked,
/// pooled triangular inverse — the primitive behind the GPTQ error feedback's `feedback_u`.
pub mod inv;
