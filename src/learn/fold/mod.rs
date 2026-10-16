//! Fold-transfer pipeline on the crate build path: the weight-fit steps (`fit`) and the activation,
//! evaluation and accounting statistics they consume (`stats`).

/// Weight-fit steps: Jacobi eigendecomposition, scale quantisers, closed-form steps, ICM passes and
/// the low-rank corrector.
pub mod fit;
/// Statistics: activation Grams, held-out KL, layer-wise intensity and the exact byte budget.
pub mod stats;
