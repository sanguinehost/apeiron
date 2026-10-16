//! Extraction unit for the Q32.32 linear-algebra primitives (`src/learn/solve/`). Each module is
//! reached by `#[path]` (relative to this file's directory), so the extracted bodies are the real
//! source; each `word_solve_<fn>` wrapper fixes the monomorphic root and calls the real function,
//! restating no formula. The thread drivers (`block_rows_par`, `trailing_update_par`,
//! `inverse_upper_par`) are opaque in `proof/regen.sh`; their boundary axioms are in
//! `proof/boundary-axioms.txt`.

#[path = "../../src/learn/solve/chol.rs"]
pub mod chol;
#[path = "../../src/learn/solve/eig.rs"]
pub mod eig;
#[path = "../../src/learn/solve/gemm.rs"]
pub mod gemm;
#[path = "../../src/learn/solve/inv.rs"]
pub mod inv;
#[path = "../../src/learn/solve/q32.rs"]
pub mod q32;
#[path = "../../src/learn/solve/syrk.rs"]
pub mod syrk;

/// The real `A·B` body (`C = round32(Σ a·b)`), rooted at the concrete Q32.32 `i64` word.
pub fn word_solve_gemm_nn(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<i64>, gemm::GemmError> {
    gemm::gemm_nn(a, b, m, k, n, threads)
}

/// The real `A·Bᵀ` body, rooted at the concrete Q32.32 `i64` word.
pub fn word_solve_gemm_nt(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<i64>, gemm::GemmError> {
    gemm::gemm_nt(a, b, m, k, n, threads)
}

/// The real weighted-Gram body `Xᵀ diag(w) X`, rooted at the concrete Q32.32 `i64` word.
pub fn word_solve_wsyrk(
    x: &[i64],
    w: Option<&[i64]>,
    o: usize,
    n: usize,
    threads: usize,
) -> Result<Vec<i64>, syrk::SyrkError> {
    syrk::wsyrk(x, w, o, n, threads)
}

/// The real elementwise Hadamard product body, rooted at the concrete Q32.32 `i64` word.
pub fn word_solve_hadamard_into(a: &mut [i64], h: &[i64]) -> Result<(), syrk::SyrkError> {
    syrk::hadamard_into(a, h)
}

/// The real SPD solve body, rooted at the concrete Q32.32 `i64` word.
pub fn word_solve_chol_solve(
    a: &[i64],
    b: &[i64],
    n: usize,
    k: usize,
    threads: usize,
) -> Result<chol::Solved, chol::SolveError> {
    chol::chol_solve(a, b, n, k, threads)
}

/// The real upper inverse-Cholesky factor body, rooted at the concrete Q32.32 `i64` word.
pub fn word_solve_inv_factor_upper(
    h: &[i64],
    n: usize,
    threads: usize,
) -> Result<Vec<i64>, chol::SolveError> {
    inv::inv_factor_upper(h, n, threads)
}

/// The real top-`r` symmetric eigen-subspace body, rooted at the concrete Q32.32 `i64` word.
pub fn word_solve_top_eigh(
    m: &[i64],
    n: usize,
    r: usize,
    threads: usize,
) -> Result<eig::TopEig, eig::EigError> {
    eig::top_eigh(m, n, r, threads)
}

/// The real floor-log2 body, rooted at the concrete `u128` word.
pub fn word_solve_q32_ilog2_pos(x: u128) -> Option<u32> {
    q32::ilog2_pos(x)
}

/// The real round-to-nearest-even right shift, rooted at the concrete `i128` value.
pub fn word_solve_q32_rshift_rne(v: i128, s: u32) -> Option<i128> {
    q32::rshift_rne(v, s)
}

/// The real Q64.64 → Q32.32 round-to-nearest-even narrowing, rooted at the concrete `i128`.
pub fn word_solve_q32_round_even(v: i128) -> Result<i64, q32::Q32Error> {
    q32::round_q32_even(v)
}

/// The real [`q32::Fx::from_i128`] body, rooted at the concrete `i128`/`i32`.
pub fn word_solve_q32_fx_from_i128(v: i128, e: i32) -> Result<q32::Fx, q32::Q32Error> {
    q32::Fx::from_i128(v, e)
}

/// The real [`q32::Bfp::from_i128`] body, rooted at the concrete `&[i128]`.
pub fn word_solve_q32_bfp_from_i128(v: &[i128], e: i32) -> Result<q32::Bfp, q32::Q32Error> {
    q32::Bfp::from_i128(v, e)
}

/// The real [`q32::SpdQ::from_q32`] body, rooted at the concrete Q32.32 `i64` word.
pub fn word_solve_q32_spd_from_q32(h: &[i64], n: usize) -> Result<q32::SpdQ, q32::Q32Error> {
    q32::SpdQ::from_q32(h, n)
}

/// The real binary16 round-to-nearest-even body, rooted at the concrete [`q32::Fx`].
pub fn word_solve_q32_round_f16(x: q32::Fx) -> Result<q32::Fx, q32::Q32Error> {
    q32::round_f16(x)
}
