//! Extraction unit for the shared Q32.32 limb product (`src/engine/kern/limb.rs`).
//!
//! `src/engine/kern/mod.rs` already declares the module in this unit
//! (`pub(crate) mod limb;`), so no `#[path]` declaration is needed here. Each wrapper below fixes
//! the concrete word monomorphization and calls a real public entry point; no wrapper restates a
//! formula, a placement or a digit.
//!
//! # Boundary
//!
//! The pool dispatch (`crate::engine::chart::pool`) is opaque to the extraction with the rest of
//! that module (thread boundary), so the serial fallback bodies of the chunk tasks are the ones the
//! roots reach: `Plane::product_rows` and its serial `Layout::placement`, `Layout::weight_payload`,
//! `Layout::columns` and `transpose_rows` call each task's `ChunkTask::run` directly below
//! `POOL_MIN`. The kernel calls inside those bodies (`gemm_q8_0`, `matmat_q8act`) are separately
//! opaque/twin-tested; the digit, placement and transpose arithmetic is the real source.

use crate::kern::limb::{gemm_q32, gemv_q32, rowdot_q32, syrk_q32, LimbError};

/// `gemm_q32`: `C = A·Bᵀ` in Q32.32 on the limb planes.
pub fn word_limb_gemm_q32(
    a: &[i64],
    m: usize,
    b: &[i64],
    p: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    gemm_q32(a, m, b, p, n, limbs)
}

/// `syrk_q32`: the mirrored upper triangle of `A·Aᵀ`.
pub fn word_limb_syrk_q32(
    a: &[i64],
    m: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    syrk_q32(a, m, n, limbs)
}

/// `gemv_q32`: one row against many.
pub fn word_limb_gemv_q32(
    x: &[i64],
    b: &[i64],
    p: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    gemv_q32(x, b, p, n, limbs)
}

/// `rowdot_q32`: the batched diagonal of the same planes.
pub fn word_limb_rowdot_q32(
    a: &[i64],
    b: &[i64],
    m: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    rowdot_q32(a, b, m, n, limbs)
}

