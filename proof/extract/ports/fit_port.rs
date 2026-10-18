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

// ── `learn::fold::fit::icm` ──────────────────────────────────────────────────────────────────

/// The 256 `E8 ∪ O8` codewords as Q32.32 words.
pub fn word_fit_icm_codebook() -> [[i64; fit::icm::BLOCK]; fit::icm::CODE_LEN] {
    fit::icm::codebook()
}

/// Euclidean nearest codebook row of a grid 8-vector.
pub fn word_fit_icm_nearest_index(
    code: &[i64; fit::icm::BLOCK],
) -> Result<usize, fit::icm::IcmError> {
    fit::icm::nearest_index(code)
}

/// The metric the ICM cost is minimised in.
pub fn word_fit_icm_ridged_metric(
    hl: &[i64],
    n: usize,
    icm_ridge: i64,
) -> Result<Vec<i64>, fit::icm::IcmError> {
    fit::icm::ridged_metric(hl, n, icm_ridge)
}

/// `plane_w(P) = fwht128(s·d) · r_rep · c`.
pub fn word_fit_icm_plane_w(
    p: fit::icm::PlaneRef<'_>,
    o: usize,
    n: usize,
) -> Result<Vec<i64>, fit::icm::IcmError> {
    fit::icm::plane_w(p, o, n)
}

/// `J(Ws, W, Hl) = Σ_o g_row[o] · (D_o Hl · D_o)`.
pub fn word_fit_icm_j_rows(
    ws: &[i64],
    w: &[i64],
    hl: &[i64],
    o: usize,
    n: usize,
    g_row: Option<&[i64]>,
) -> Result<i64, fit::icm::IcmError> {
    fit::icm::j_rows(ws, w, hl, o, n, g_row)
}

/// One plane's iterated conditional modes over the block codebook.
pub fn word_fit_icm_e8_icm_step(
    ws: &[i64],
    hl: &[i64],
    p: fit::icm::PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: i64,
) -> Result<fit::icm::Plane, fit::icm::IcmError> {
    fit::icm::e8_icm_step(ws, hl, p, o, n, sweeps, icm_ridge)
}

/// Joint block ICM over two planes.
pub fn word_fit_icm_e8_joint_icm_2plane(
    ws: &[i64],
    hl: &[i64],
    p1: fit::icm::PlaneRef<'_>,
    p2: fit::icm::PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    icm_ridge: i64,
    g_row: Option<&[i64]>,
    row_chunk: usize,
    blocks: Option<&[usize]>,
) -> Result<(fit::icm::Plane, fit::icm::Plane, fit::icm::JointStats), fit::icm::IcmError> {
    fit::icm::e8_joint_icm_2plane(
        ws, hl, p1, p2, o, n, sweeps, icm_ridge, g_row, row_chunk, blocks,
    )
}

/// Joint least-squares solve for both planes' group scales `(r1, r2)`.
pub fn word_fit_icm_r_step2(
    ws: &[i64],
    hl: &[i64],
    p1: fit::icm::PlaneRef<'_>,
    p2: fit::icm::PlaneRef<'_>,
    o: usize,
    n: usize,
) -> Result<(Vec<i64>, Vec<i64>), fit::icm::IcmError> {
    fit::icm::r_step2(ws, hl, p1, p2, o, n)
}

/// Row-major `[rows, inner] × [inner, cols]` product on the shared `Q32.32` path.
pub fn word_fit_icm_matmul(
    a: &[i64],
    b: &[i64],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<i64>, fit::icm::IcmError> {
    fit::icm::matmul(a, b, rows, inner, cols)
}
