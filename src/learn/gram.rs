//! Exact-integer Gram accumulation: fold a batch of `i8` rows into an `i64` accumulator.
//!
//! [`gram_fold_i8`] performs `g[i*d+j] += Σ_r x[r*d+i]·x[r*d+j]` in checked `i64` arithmetic. It
//! is the exact integer Gram fold that `Sanguine.TrainingCollapse.gram_stack`
//! (`proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean`) states over `ℝ`: folding `X₁`
//! then `X₂` from the zero accumulator gives `(stack X₁ X₂)ᵀ (stack X₁ X₂)`, because the row sum is
//! additive and order-free. No rounding enters the path.
//!
//! This module is included by `#[path]` from the Aeneas extraction unit `proof/extract/lib.rs`, which
//! also includes `src/engine/kern/mod.rs`, so the kernel sibling [`gram_fold_i8_kern`] type-checks there.
//!
//! The three loops are split into three functions (`rows_loop` calls `row_loop` calls
//! `row_pair_loop`), one `while` loop each, with a `bool` failure flag and an `Option<GramError>`
//! accumulator: the pinned Charon/Aeneas fragment does not compute a loop fixed point for a nested
//! loop that carries an index re-read from a mutated `Vec`. No `?` and no early `return` appears
//! inside a loop; overflow is `checked_*` only.

#![allow(
    clippy::many_single_char_names,
    reason = "the flat numeric kernel indexes row-major `[rows, d]` buffers by `r/i/j/a/b`, which \
              reads closer to the reference Gram arithmetic than spelled-out names; the pedantic \
              lints stay on for the rest of the crate."
)]

/// A refused exact-integer Gram fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GramError {
    /// `g.len() != d*d`, or `x.len() != rows*d`.
    ShapeMismatch,
    /// A checked `i64` product or accumulation left its range.
    Overflow,
}

/// Innermost loop over `j`: add `a * (x[r*d+j] as i64)` to `g[i*d+j]` for every `j < d`, where
/// `a = (x[r*d+i] as i64)`. Returns the updated accumulator and the first refusal, or `None`.
///
/// # Complexity
/// `O(d)` checked `i64` multiply-adds and `O(1)` extra words; one element write per `j`.
fn row_pair_loop(
    mut g: Vec<i64>,
    x: &[i8],
    d: usize,
    r: usize,
    i: usize,
    a: i64,
) -> (Vec<i64>, Option<GramError>) {
    let mut j = 0usize;
    let mut ok = true;
    let mut err: Option<GramError> = None;
    while j < d && ok {
        let b = i64::from(x[r * d + j]);
        let idx = i * d + j;
        if let Some(prod) = a.checked_mul(b) {
            if let Some(sum) = g[idx].checked_add(prod) {
                g[idx] = sum;
            } else {
                err = Some(GramError::Overflow);
                ok = false;
            }
        } else {
            err = Some(GramError::Overflow);
            ok = false;
        }
        j += 1;
    }
    (g, err)
}

/// Middle loop over `i`: fold row `r` of `x` into `g` by calling [`row_pair_loop`] for every `i`.
/// Returns the updated accumulator and the first refusal, or `None`.
///
/// # Complexity
/// `O(d²)` checked `i64` multiply-adds and `O(1)` extra words.
fn row_loop(mut g: Vec<i64>, x: &[i8], d: usize, r: usize) -> (Vec<i64>, Option<GramError>) {
    let mut i = 0usize;
    let mut ok = true;
    let mut err: Option<GramError> = None;
    while i < d && ok {
        let a = i64::from(x[r * d + i]);
        let (g1, e1) = row_pair_loop(g, x, d, r, i, a);
        g = g1;
        if let Some(e) = e1 {
            err = Some(e);
            ok = false;
        }
        i += 1;
    }
    (g, err)
}

/// Outer loop over `r`: fold every row of `x` into `g` by calling [`row_loop`] for each `r`.
/// Returns the updated accumulator and the first refusal, or `None`.
///
/// # Complexity
/// `O(rows·d²)` checked `i64` multiply-adds and `O(1)` extra words.
fn rows_loop(mut g: Vec<i64>, x: &[i8], d: usize, rows: usize) -> (Vec<i64>, Option<GramError>) {
    let mut r = 0usize;
    let mut ok = true;
    let mut err: Option<GramError> = None;
    while r < rows && ok {
        let (g1, e1) = row_loop(g, x, d, r);
        g = g1;
        if let Some(e) = e1 {
            err = Some(e);
            ok = false;
        }
        r += 1;
    }
    (g, err)
}

/// Fold the `rows` rows of `x` (row-major `[rows, d]`, entries in `[-128, 127]`) into the exact
/// Gram accumulator `g` (row-major `[d, d]`): `g[i*d+j] += Σ_{r<rows} x[r*d+i]·x[r*d+j]`, in
/// checked `i64` arithmetic.
///
/// # Errors
/// [`GramError::ShapeMismatch`] if `g.len() != d*d` or `x.len() != rows*d`;
/// [`GramError::Overflow`] if a checked `i64` product or accumulation leaves its range, in which
/// case the returned accumulator is unspecified (the shape was valid but a partial fold ran).
///
/// # Complexity
/// `O(rows·d²)` checked `i64` multiply-adds and `O(1)` extra words beyond `g`; no allocation.
pub fn gram_fold_i8(g: Vec<i64>, x: &[i8], rows: usize, d: usize) -> Result<Vec<i64>, GramError> {
    if g.len() != d * d || x.len() != rows * d {
        return Err(GramError::ShapeMismatch);
    }
    let (g, err) = rows_loop(g, x, d, rows);
    if let Some(e) = err {
        return Err(e);
    }
    Ok(g)
}

// Imports of the kernel route below. They follow the extracted bodies so the source lines the
// generated `ApeironWord.lean` cites do not move.
use crate::engine::chart::QuantAct;
use crate::engine::kern::gemm::gemm_q8_0;
use crate::format::quant::{GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES, Q8_0_QK};

/// The `f16` word of the weight scale `1.0`: `pow2::f16_split(0x3C00) = (1024, 22)` and
/// `1024·2^22 = 2^32`, the Q32.32 unit, so a `Q8_0` row carrying this scale decodes each signed
/// code unchanged.
const Q8_0_UNIT: u16 = 0x3C00;

/// The largest magnitude of one product of two `i8` entries: `128·128`.
const I8_PRODUCT_MAX: u64 = 16_384;

/// The widest contraction the kernel takes, in activation blocks (`n_in = blocks·32`).
const KERNEL_MAX_BLOCKS: usize = 512;

/// Whether every accumulator entry is far enough from `i64`'s ends that no prefix sum of the fold
/// can leave range: `|g[idx]| ≤ i64::MAX − rows·16_384`, the largest possible deviation of a
/// partial row sum from `0`.
///
/// # Complexity
///
/// `O(d²)` compares and `O(1)` extra space.
pub(crate) fn gram_accumulator_clear(g: &[i64], rows: usize) -> bool {
    let Some(slack) = u64::try_from(rows)
        .ok()
        .and_then(|r| r.checked_mul(I8_PRODUCT_MAX))
    else {
        return false;
    };
    let Some(limit) = i64::MAX.unsigned_abs().checked_sub(slack) else {
        return false;
    };
    let mut idx = 0usize;
    while idx < g.len() {
        if g[idx].unsigned_abs() > limit {
            return false;
        }
        idx += 1;
    }
    true
}

/// The kernel columns of `XᵀX`, or `None` when the kernel refuses the call.
///
/// Weight row `i` is column `i` of `x` as `Q8_0` codes under the `f16` scale `1.0` (`d = 2^32` in
/// Q32.32); activation column `j` is column `j` of `x` as `QuantAct` codes under the scale `1`.
/// Column `j` of the result therefore holds `Σ_r x[r*d+i]·x[r*d+j]` for `i < d` — the integer Gram
/// entry `g[i*d+j]` — with no limb split and no rounding: the contract
/// `round_shift_32(Σ_b d_b·S_b·D_b)` is `round_shift_32(2^32·Σ_b D_b)` over integer `D_b`, which is
/// `Σ_b D_b` exactly (the final `+2^9` of `finish(…, 22)` is below one unit of the integer sum).
///
/// `-128` is admitted on both sides: the kernel's fast bound is stated for `|xq_j| ≤ 128`
/// (`chart::q8act_fast_bound_ok` with `top = 128`) and the packing adds `128` to the signed code.
/// The contraction `rows` is zero-padded to a whole 32-block, which contributes `0·0`.
///
/// # Complexity
///
/// Temporal `O(d·ceil(rows/32)·34 + d·rows)` byte writes to build the operands plus the kernel's
/// `O(rows·d²)` byte multiply-adds (32 per `vpdpbusd`); spatial `O(d·ceil(rows/32)·34)` weight bytes
/// and `O(d·rows)` activation bytes.
fn gram_columns(x: &[i8], rows: usize, d: usize) -> Option<Vec<Vec<i64>>> {
    let blocks = rows.div_ceil(Q8_0_QK);
    if blocks > KERNEL_MAX_BLOCKS {
        return None;
    }
    let n_in = blocks * Q8_0_QK;
    let mut raw: Vec<u8> = Vec::with_capacity(d * blocks * Q8_0_BLOCK_BYTES);
    let mut i = 0usize;
    while i < d {
        let mut b = 0usize;
        while b < blocks {
            raw.extend_from_slice(&Q8_0_UNIT.to_le_bytes());
            let mut l = 0usize;
            while l < Q8_0_QK {
                let r = b * Q8_0_QK + l;
                raw.push(if r < rows {
                    x[r * d + i].cast_unsigned()
                } else {
                    0
                });
                l += 1;
            }
            b += 1;
        }
        i += 1;
    }
    let mut qs: Vec<QuantAct> = Vec::with_capacity(d);
    let mut j = 0usize;
    while j < d {
        let mut codes: Vec<i8> = Vec::with_capacity(n_in);
        let mut r = 0usize;
        while r < n_in {
            codes.push(if r < rows { x[r * d + j] } else { 0 });
            r += 1;
        }
        qs.push(QuantAct::from_parts(vec![1i64; blocks], codes));
        j += 1;
    }
    gemm_q8_0(GGML_TYPE_Q8_0, &raw, d, n_in, &qs)
}

/// Add the kernel columns into the accumulator: `g[i*d+j] += cols[j][i]`.
///
/// # Errors
/// [`GramError::Overflow`] when an entry's checked `i64` addition leaves range; unreachable on the
/// caller's admissible band (`gram_accumulator_clear`), kept as the checked mapping.
///
/// # Complexity
///
/// `O(d²)` checked `i64` additions and `O(1)` extra words.
pub(crate) fn gram_accumulate(
    mut g: Vec<i64>,
    cols: &[Vec<i64>],
    d: usize,
) -> Result<Vec<i64>, GramError> {
    debug_assert_eq!(cols.len(), d, "one kernel column per Gram row");
    // Flag loops: a `return` inside these nested sweeps has no fixed point in the pinned
    // Charon/Aeneas fragment, so the first refusal is carried in `err`, both loop conditions are
    // guarded by `err.is_none()`, and one `return` follows the outer sweep. The accumulation order
    // and the first-refusal order are unchanged.
    let mut err: Option<GramError> = None;
    let mut j = 0usize;
    while j < d && err.is_none() {
        let col = &cols[j];
        let mut i = 0usize;
        while i < d && err.is_none() {
            let idx = i * d + j;
            match g[idx].checked_add(col[i]) {
                Some(sum) => g[idx] = sum,
                None => err = Some(GramError::Overflow),
            }
            i += 1;
        }
        j += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(g)
}

/// The kernel route of [`gram_fold_i8`]: the same exact `g[i*d+j] += Σ_r x[r*d+i]·x[r*d+j]`, with
/// the `rows·d²` products on `gemm_q8_0` (AVX-VNNI `vpdpbusd`) instead
/// of the checked scalar loop. [`gram_columns`] carries the operand construction; both sides are
/// `i8`, so the `Q8_0` kernel is exact with no limb split and no re-pin. This function is not an
/// extraction root and takes `crate::` paths; `proof/extract/lib.rs` roots at [`gram_fold_i8`].
///
/// Every call the kernel accepts accumulates exactly the integer total the extracted body
/// accumulates, on the admissible band where no partial sum can leave `i64`. A call whose
/// accumulator is nearer an `i64` end than `rows·16_384`, or that the kernel refuses (a host
/// without AVX-VNNI, a contraction past 512 blocks of 32), is handed to [`gram_fold_i8`] itself
/// unchanged, so the refusal — including the scalar fold's first-overflowing-prefix rule — is the
/// extracted body's.
///
/// # Errors
/// [`GramError::ShapeMismatch`] on the same shape test as [`gram_fold_i8`];
/// [`GramError::Overflow`] when a checked `i64` addition of the exact totals leaves range.
///
/// # Complexity
///
/// Temporal `O(rows·d²)` kernel byte multiply-adds plus `O(d²)` checked adds on the kernel route;
/// spatial `O(d²)` for the columns, `O(d·rows)` for the operands, `O(1)` beyond `g`.
pub fn gram_fold_i8_kern(
    g: Vec<i64>,
    x: &[i8],
    rows: usize,
    d: usize,
) -> Result<Vec<i64>, GramError> {
    if g.len() != d * d || x.len() != rows * d {
        return Err(GramError::ShapeMismatch);
    }
    if rows == 0 || d == 0 {
        return Ok(g);
    }
    if gram_accumulator_clear(&g, rows) {
        if let Some(cols) = gram_columns(x, rows, d) {
            return gram_accumulate(g, &cols, d);
        }
    }
    gram_fold_i8(g, x, rows, d)
}
