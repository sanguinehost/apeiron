//! Statistics the NanoQuant path consumes: activation Grams, the output-side Hessian sketch and
//! held-out evaluation.

/// Per-layer activation Grams of the four linears, accumulated from the forward tape.
pub mod capture;
/// Softmax, sampled-label cross entropy and target draw of the held-out capture.
pub mod ce;
/// Held-out KL, TV, top-1 and NLL of a student against a teacher on the ported model.
pub mod heldout;
/// Output-side Hessian sketch `HO = Σ w_t g_t g_tᵀ` from sampled-label gradients on the ported backward pass.
pub mod hocapture;
/// Reader of llama.cpp's `--kl-divergence-base` log-probability file, one chunk at a time.
pub mod kldbase;
/// `f64` CPU forward of one `Qwen2` decoder layer, returning every intermediate the corrector reads.
pub mod parts;
