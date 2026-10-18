//! Extraction roots for the closed-form weight fit (`src/learn/fold/fit/`) beyond the corrector,
//! the eigendecomposition and the quantisers already rooted in `fold_port.rs`.
//!
//! Each `word_*` wrapper fixes the monomorphic root and calls the real
//! `crate::fold_port::fit::<module>::<fn>` body; it restates no formula. The module is reached
//! through the real `mod.rs` declared in `fold_port.rs`, so the extracted bodies are the sources
//! under `src/learn/fold/fit/`.
//!
//! The remaining entry points are not roots: their call graphs reach `kern::limb` (the `c`/`r`/`d`
//! steps), `fit::icm` (`fold_fit`, `nb_refine`) or `stats::hocapture` (`fnorm_fit`), and those
//! bodies carry constructs the pinned Aeneas refuses. `gptq_e8` is reached from `feedback_u` only
//! as far as the carrier boundary; its own body is a rewrite lane.

use crate::fold_port::fit;

/// The normalised 128-block Hadamard transform on words.
pub fn word_fit_steps_fwht_words(x: &mut [i64]) -> Result<(), fit::steps::StepsError> {
    fit::steps::fwht_words(x)
}

/// The block-diagonal Hadamard Gram `R·H·R`.
pub fn word_fit_steps_had_gram(
    h: &[i64],
    n: usize,
) -> Result<Vec<i64>, fit::steps::StepsError> {
    fit::steps::had_gram(h, n)
}

/// The reconstruction `S[o,i]·r[o,i/128]·c[i]`.
pub fn word_fit_steps_compose(
    s: &[i64],
    r: &[i64],
    c: &[i64],
    o: usize,
    n: usize,
) -> Result<Vec<i64>, fit::steps::StepsError> {
    fit::steps::compose(s, r, c, o, n)
}

/// The inverse feedback factor `U` (`UᵀU = H⁻¹`) of the damped Hessian.
pub fn word_fit_gptq_e8_carrier_feedback_u(
    h: &[i64],
    n: usize,
    damp: i64,
) -> Result<fit::gptq_e8_carrier::UFac, fit::gptq_e8::GptqE8Error> {
    fit::gptq_e8_carrier::feedback_u(h, n, damp)
}
