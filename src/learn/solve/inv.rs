//! The upper-triangular inverse feedback factor `U` of an SPD matrix, `UᵀU = H⁻¹`, in Q32.32 `i64`:
//! [`inv_factor_upper`] runs the diagonal-equilibrated `Cholesky` of [`super::chol`] and inverts its
//! upper factor with a blocked, pooled triangular inverse.
//!
//! # Object
//!
//! With `D = diag(d_i)`, `d_i = sqrt(H_ii)` (the same equilibrating scale [`super::chol`] uses),
//! `Ĥ = D⁻¹ H D⁻¹` is SPD with a unit diagonal, so `Ĥ = V̂ V̂ᵀ` with `V̂` upper-triangular and a
//! positive diagonal. Then `H = D^{1/2} Ĥ D^{1/2} = (D^{1/2} V̂)(D^{1/2} V̂)ᵀ` and
//!
//! ```text
//! U = V̂⁻¹ D^{-1/2},   Uᵀ U = D^{-1/2} V̂⁻ᵀ V̂⁻¹ D^{-1/2} = D^{-1/2} Ĥ⁻¹ D^{-1/2} = H⁻¹,
//! ```
//!
//! with `U` upper-triangular and a positive diagonal (the positive-diagonal upper factor of `H⁻¹`,
//! i.e. `U = cholesky_upper(H⁻¹)` by uniqueness). The returned matrix is `U`, row-major `[n, n]`.
//!
//! # Method
//!
//! 1. [`super::chol::equilibrate`] gives `Ĥ` and `d`; a non-positive `H_ii` is
//!    [`SolveError::NonPositiveDiagonal`].
//! 2. `V̂` from the *lower* factor of [`super::chol::factor_scaled`] by the reversal permutation `P`
//!    (rows and columns reversed): `P Ĥ P = (P L P)(P L P)ᵀ` with `P L P` lower, and
//!    `V̂ = P (P L P) P` is upper with `V̂ V̂ᵀ = P (P Ĥ P) P = Ĥ`. One `O(n²)` reversal in, one out.
//! 3. `V̂⁻¹` by a blocked back-substitution, one round-to-nearest per stored entry: the columns of
//!    `U` are independent (`U[r, c]` for `r < c` reads only column `c`), so the `INV_BLOCK`-wide
//!    column blocks are independent and are handed to the pool, one contiguous column range per
//!    participant; the result is bit-identical to the serial body.
//! 4. `U = V̂⁻¹ D^{-1/2}`, one rounded `i128` quotient per entry.
//!
//! # Partiality
//!
//! Every dimension is checked ([`SolveError::ShapeMismatch`]), `n == 0` is [`SolveError::Empty`], a
//! non-positive pivot is [`SolveError::NotPositiveDefinite`], and every operation is checked, so a
//! factor that leaves the `i128` range is [`SolveError::Overflow`]. No function here panics on
//! caller input.
//!
//! # Cost
//!
//! `O(n³/6)` checked `i128` multiply-subtracts for the inverse on top of [`super::chol`]'s
//! `O(n³/3)` factor, `O(n²)` `i64` words for the reversed matrix, the transposed inverse and the
//! result.
//!
//! Grade: `tested` (`tests/learn_solve_inv.rs`); [`invert_column_blocks`] extracts under the pinned
//! Charon/Aeneas fragment, and no body here carries a refinement theorem, so nothing here is
//! `proved`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the recurrence's own index symbols are `r/c/t` on a row-major `[n, n]` Q32.32 buffer, \
              narrower than spelled-out names; the pedantic lints stay on for the rest of the crate."
)]

use super::chol::{div_to_i64, equilibrate, factor_scaled, SolveError};
use crate::engine::chart::pool::{chunks_mut_global, width_global, FirstErr};
use crate::engine::chart::task::ChunkTask;

/// Column-block width of the blocked triangular inverse; the inverse is column-parallel, so the
/// block is only a locality unit (one `64 × 64` diagonal block and the rows below it stay resident
/// while a block column is swept). Matches [`super::chol`]'s factor block width.
const INV_BLOCK: usize = 64;

/// One stored entry of `V̂⁻¹`: `round(−Σ_{t ∈ [t0, t1)} v[r·d + t]·urow[t] / v[r·d + r])`, or `None`
/// if an `i128` accumulation or the rounding leaves its range. `v[r·d + r]` must be positive; the
/// caller checks it. Entries of `U` above the diagonal are zero, so a contraction may run past the
/// entry's own column to the end of its block column without changing the sum.
///
/// # Complexity
///
/// `t1 − t0` checked `i128` multiply-adds, `O(1)` space.
fn inv_row_entry(v: &[i64], d: usize, r: usize, t0: usize, t1: usize, urow: &[i64]) -> Option<i64> {
    let mut acc: i128 = 0;
    let mut t = t0;
    let mut ok = true;
    while t < t1 && ok {
        match i128::from(v[r * d + t]).checked_mul(i128::from(urow[t])) {
            Some(pr) => match acc.checked_add(pr) {
                Some(s) => acc = s,
                None => ok = false,
            },
            None => ok = false,
        }
        t += 1;
    }
    if ok {
        div_to_i64(-acc, i128::from(v[r * d + r]))
    } else {
        None
    }
}

/// Invert the columns `[col0, col_end)` of an upper-triangular `v` (row-major `[n, n]`, positive
/// diagonal) into the transposed buffer `ut`, where `ut[(c − col0)·n + r]` is `U[r, c]`: first the
/// `INV_BLOCK`-wide diagonal block `V[c0, c0..c1]` by an in-block back-substitution, then every row
/// `r < c0` by one back-substitution against the computed entries of the same block column. The
/// column blocks are processed in order and each is independent of the ones after it.
///
/// # Errors
///
/// [`SolveError::NotPositiveDefinite`] if a diagonal entry of `v` is non-positive;
/// [`SolveError::Overflow`] if an `i128` accumulation or the rounded quotient leaves its range.
///
/// # Complexity
///
/// `O((col_end − col0)·n·INV_BLOCK + Σ_c (c − col0)²/2)` checked multiply-adds derived from the two
/// sweeps (in-block `Σ_{c} (c − c0)²/2` per block, rows `Σ_{r<c0} (c1 − r)·sj` per block), `O(1)`
/// space beyond the caller's `ut`.
fn invert_column_blocks(
    v: &[i64],
    n: usize,
    ut: &mut [i64],
    col0: usize,
    col_end: usize,
) -> Result<(), SolveError> {
    // Flag loop: a `return`/`?` inside the two-nested sweeps is not modelled by the pinned
    // Charon/Aeneas fragment, so the first refusal is carried in `err`, every enclosing loop
    // condition is guarded by `err.is_none()`, and the single `return` sits after the outermost
    // loop. The write order, the contraction order and every rounding are unchanged.
    let mut err: Option<SolveError> = None;
    let mut c0 = col0;
    while c0 < col_end && err.is_none() {
        let c1 = (c0 + INV_BLOCK).min(col_end);
        let base = (c0 - col0) * n;
        // (1) The diagonal block `V[c0..c1, c0..c1]`.
        let mut c = c0;
        while c < c1 && err.is_none() {
            let vcc = v[c * n + c];
            if vcc <= 0 {
                err = Some(SolveError::NotPositiveDefinite);
            } else {
                let row = &mut ut[base + (c - c0) * n..base + (c - c0) * n + n];
                match div_to_i64(1_i128 << 64, i128::from(vcc)) {
                    Some(w) => row[c] = w,
                    None => err = Some(SolveError::Overflow),
                }
                let mut r = c;
                while r > c0 && err.is_none() {
                    r -= 1;
                    let vrr = v[r * n + r];
                    if vrr <= 0 {
                        err = Some(SolveError::NotPositiveDefinite);
                    } else {
                        match inv_row_entry(v, n, r, r + 1, c1, row) {
                            Some(w) => row[r] = w,
                            None => err = Some(SolveError::Overflow),
                        }
                    }
                }
            }
            c += 1;
        }
        // (2) Every row above the block, against the whole block column.
        let mut r = c0;
        while r > 0 && err.is_none() {
            r -= 1;
            let vrr = v[r * n + r];
            if vrr <= 0 {
                err = Some(SolveError::NotPositiveDefinite);
            } else {
                let mut k = 0usize;
                while k < c1 - c0 && err.is_none() {
                    let row = &mut ut[base + k * n..base + k * n + n];
                    match inv_row_entry(v, n, r, r + 1, c1, row) {
                        Some(w) => row[r] = w,
                        None => err = Some(SolveError::Overflow),
                    }
                    k += 1;
                }
            }
        }
        c0 = c1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// The pool task context of [`invert_columns_pool`]: the factor, the row stride and the refusal
/// cell.
struct InvertColumns<'a> {
    /// The upper-triangular factor `V̂`, row-major `[n, n]`.
    v: &'a [i64],
    /// Row and column stride.
    n: usize,
    /// Lowest-offset refusal recorded by the tasks.
    first: &'a FirstErr<usize, SolveError>,
}

impl ChunkTask<i64> for InvertColumns<'_> {
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        let c0 = start / self.n;
        let c1 = c0 + piece.len() / self.n;
        if let Err(e) = invert_column_blocks(self.v, self.n, piece, c0, c1) {
            self.first.record(start, e);
        }
    }
}

/// Run [`invert_column_blocks`] over the `m = ⌈n/INV_BLOCK⌉` column blocks through the engine pool,
/// one contiguous range of column blocks per `width_global()` participant (a contiguous row range of
/// the transposed buffer `ut`). Each entry is written by exactly one task in a fixed order, so the
/// result is bit-identical to the serial body.
///
/// # Errors
///
/// The lowest-offset [`SolveError`].
///
/// # Complexity
///
/// As [`inverse_upper_par`]: `O(n³/6)` multiply-adds split across the pool, `O(1)` extra space beyond
/// the buffer.
fn invert_columns_pool(v: &[i64], n: usize, ut: &mut [i64]) -> Result<(), SolveError> {
    let blocks = n.div_ceil(INV_BLOCK);
    let per_task = blocks.div_ceil(width_global()).max(1);
    let first: FirstErr<usize, SolveError> = FirstErr::new();
    chunks_mut_global(
        ut,
        per_task * INV_BLOCK * n,
        &InvertColumns {
            v,
            n,
            first: &first,
        },
    );
    match first.take() {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// `V̂⁻¹` of an upper-triangular `v` (row-major `[n, n]` Q32.32, positive diagonal), blocked over
/// `INV_BLOCK` column blocks and distributed over the engine pool, one contiguous range of column
/// blocks per `width_global()` participant. Every entry is written by exactly one task with a fixed
/// contraction order, so the result is bit-identical to the serial body.
///
/// # Errors
///
/// [`SolveError::ShapeMismatch`] if `v.len() != n·n`; [`SolveError::NotPositiveDefinite`] if a
/// diagonal entry is non-positive; [`SolveError::Overflow`] from any worker, or from a worker panic
/// (the body does not panic on caller input).
///
/// # Complexity
///
/// `O(n³/6 + 64·n²)` checked multiply-adds, split across the pool; `O(n²)` `i64` words for the transposed buffer plus the `O(n²)` result.
/// Derivation: [`invert_column_blocks`] over the whole `n` columns, plus the `O(n²)` transpose.
fn inverse_upper_par(v: &[i64], n: usize) -> Result<Vec<i64>, SolveError> {
    let nn = n.checked_mul(n).ok_or(SolveError::Overflow)?;
    if v.len() != nn {
        return Err(SolveError::ShapeMismatch);
    }
    // `ut[c·n + r] = U[r, c]`, so a column range of `U` is a contiguous row range of `ut`.
    let mut ut = vec![0i64; nn];
    invert_columns_pool(v, n, &mut ut)?;
    let mut u = vec![0i64; nn];
    for c in 0..n {
        for r in 0..=c {
            u[r * n + c] = ut[c * n + r];
        }
    }
    Ok(u)
}

/// Rotate a row-major `[n, n]` buffer by 180° in place (`x[i, j] ← x[n−1−i, n−1−j]`), the reversal
/// that turns the lower factor `P L P` of the reversed matrix back into the upper factor `V̂ = P L P`
/// of `Ĥ`.
///
/// # Complexity
///
/// `⌊n²/2⌋` `i64` swaps, `O(1)` space. Derivation: `⌊n/2⌋·n` row pairs plus the odd middle row's
/// `⌊n/2⌋` column pairs.
fn rotate_180(x: &mut [i64], n: usize) {
    for i in 0..n / 2 {
        for j in 0..n {
            x.swap(i * n + j, (n - 1 - i) * n + (n - 1 - j));
        }
    }
    if n % 2 == 1 {
        let i = n / 2;
        for j in 0..n / 2 {
            x.swap(i * n + j, i * n + (n - 1 - j));
        }
    }
}

/// The upper-triangular `U` with `UᵀU = H⁻¹` of an SPD `h` (row-major `[n, n]` Q32.32): equilibrate
/// `Ĥ = D⁻¹HD⁻¹`, factor `Ĥ = V̂V̂ᵀ` through [`super::chol::factor_scaled`] on the reversed matrix,
/// invert `V̂` ([`inverse_upper_par`], blocked and pooled) and scale the columns by `D^{-1/2}`.
/// The result is bit-identical to the serial body.
///
/// # Errors
///
/// [`SolveError::Empty`] if `n == 0`; [`SolveError::ShapeMismatch`] if `h.len() != n·n`;
/// [`SolveError::NonPositiveDiagonal`] if some `h_ii <= 0`; [`SolveError::NotPositiveDefinite`] if a
/// `Cholesky` pivot of `Ĥ` is non-positive, so `h` is not positive-definite;
/// [`SolveError::Overflow`] on any checked `i128` operation or `i64` narrowing, or if `n·n` leaves
/// `usize`.
///
/// # Complexity
///
/// `O(n³/3 + n³/6)` checked multiply-adds (`O(n³/3)` for the factor, `O(n³/6)` for the inverse) and
/// `O(n²)` `i64` words: the reversed matrix, the factor, the transposed inverse and the result.
/// Derivation: [`super::chol::equilibrate`] `O(n²)`, [`super::chol::factor_scaled`] `O(n³/3)`,
/// [`inverse_upper_par`] `O(n³/6)`, the `O(n²)` reversal, rotation and column scaling.
pub fn inv_factor_upper(h: &[i64], n: usize) -> Result<Vec<i64>, SolveError> {
    if n == 0 {
        return Err(SolveError::Empty);
    }
    let nn = n.checked_mul(n).ok_or(SolveError::Overflow)?;
    if h.len() != nn {
        return Err(SolveError::ShapeMismatch);
    }
    let (a_hat, d) = equilibrate(h, n)?;
    // `P Ĥ P = P L P·(P L P)ᵀ`; feed the reversed copy to the same blocked lower factor.
    let mut rev = vec![0i64; nn];
    for i in 0..n {
        for j in 0..n {
            rev[i * n + j] = a_hat[(n - 1 - i) * n + (n - 1 - j)];
        }
    }
    drop(a_hat);
    let mut v_hat = factor_scaled(&rev, n)?;
    drop(rev);
    // `V̂ = P·(P L P)·P`, the 180° rotation of the reversed lower factor.
    rotate_180(&mut v_hat, n);
    let binv = inverse_upper_par(&v_hat, n)?;
    drop(v_hat);
    // `U = V̂⁻¹·D^{-1/2}`: column `j` of `V̂⁻¹` divided by `d_j`, one rounded quotient per entry.
    // The checked quotient reports through `err` rather than `?` so no return sits inside the
    // nested loop; the iteration order and the per-entry rounding are unchanged.
    let mut u = vec![0i64; nn];
    let mut err: Option<SolveError> = None;
    let mut i = 0;
    while i < n && err.is_none() {
        let mut j = i;
        while j < n && err.is_none() {
            match div_to_i64(i128::from(binv[i * n + j]) << 32, i128::from(d[j])) {
                Some(w) => u[i * n + j] = w,
                None => err = Some(SolveError::Overflow),
            }
            j += 1;
        }
        i += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(u)
}
