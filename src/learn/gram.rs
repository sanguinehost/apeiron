//! Exact-integer Gram accumulation: fold a batch of `i8` rows into an `i64` accumulator.
//!
//! [`gram_fold_i8`] performs `g[i*d+j] += Σ_r x[r*d+i]·x[r*d+j]` in checked `i64` arithmetic. It
//! is the exact integer Gram fold that `Sanguine.TrainingCollapse.gram_stack`
//! (`proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean`) states over `ℝ`: folding `X₁`
//! then `X₂` from the zero accumulator gives `(stack X₁ X₂)ᵀ (stack X₁ X₂)`, because the row sum is
//! additive and order-free. No rounding enters the path.
//!
//! This module is dependency-free (no `crate::` imports) so it can be included by `#[path]` from
//! the Aeneas extraction unit `proof/extract/lib.rs` and the extracted body is the real source.
//!
//! The three loops are split into three functions (`rows_loop` calls `row_loop` calls
//! `row_pair_loop`), one `while` loop each, with a `bool` failure flag and an `Option<GramError>`
//! accumulator: the pinned Charon/Aeneas fragment does not compute a loop fixed point for a nested
//! loop that carries an index re-read from a mutated `Vec`. No `?` and no early `return` appears inside a
//! loop; overflow is `checked_*` only.

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
