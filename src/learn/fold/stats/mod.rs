//! Statistics the fold transfer consumes: activation Grams, held-out evaluation, shrinkage
//! intensity and byte accounting.

/// Exact 27B byte accounting at a target bits-per-weight, over the reference class table.
pub mod budget;
/// Per-layer activation Grams of the four linears, accumulated from the forward tape.
pub mod capture;
/// Held-out KL, TV, top-1 and NLL of a student against a teacher on the ported model.
pub mod heldout;
/// Output-side Hessian sketch `HO = Σ w_t g_t g_tᵀ` from sampled-label gradients on the ported backward pass.
pub mod hocapture;
/// Analytic shrinkage intensities (Schäfer–Strimmer, Ledoit–Wolf) of the output-side Hessian sketch.
pub mod lw;
