//! Fold-transfer refit steps over the 128-block Hadamard basis.
//!
//! The one-bit transfer refits a quantized weight `Wq = compose(S, r, c)` by alternating exact
//! continuous updates with a discrete one. This module ports the continuous steps and the
//! transforms they run on.
//!
//! # Port
//!
//! Ports the refit steps of the frozen one-bit transfer script `fold_transfer.py` (read-only,
//! `spec/fold_transfer.py:426-490`): the objective `J` (434), `compose` (438), `fwht128` (441),
//! `had_gram` (449), `c_step` (453), `r_step` (463) and `d_step` (478). The d step's forward model
//! is the one `run_variant` refits (`spec/fold_transfer.py:886`), `S = fwht128(S_rot * d[None, :])`,
//! i.e. `recon(d) = compose(fwht128(S_rot ⊙ d), r, c)`.
//!
//! # The steps
//!
//! `W`, `Ws` are `O × n` (`O` out rows, `n` in columns, row-major); `n = G·128` for the Hadamard
//! steps, `G = n / 128`. `Ws`, `S` and the metric `Hl` (`n × n`, symmetric positive-definite) are
//! the calibration data; `S` is a sign matrix, `r` an `O × G` per-group scale and `c` a length-`n`
//! per-column scale.
//!
//! - `compose(S, r, c)[o,i] = S[o,i]·r[o, i/128]·c[i]`.
//! - `objective(Ws, Wp, Hl) = tr((Ws−Wp)·Hl·(Ws−Wp)ᵀ)`.
//! - `c_step` solves the `n × n` normal system `A·c = b`, `A[i,j] = Hl[i,j]·(Shᵀ·Sh_g)[i,j]`,
//!   `Sh = S⊙r_rep`, `Sh_g = Sh⊙g_row`, `b[i] = Σ_o (Ws·Hl)[o,i]·Sh_g[o,i]`. The `A`-formation is
//!   the shared primitive [`crate::learn::solve::syrk::wsyrk`] (the weighted Gram) followed by
//!   [`crate::learn::solve::syrk::hadamard_into`] (the `hl` factor); the solve is
//!   [`crate::learn::solve::chol::chol_solve`]; the metric product `Ws·Hl` is [`mul_metric`] over
//!   the shared `Q32.32` product [`crate::learn::solve::gemm::gemm_nn`]. The step stays `f64` and
//!   crosses into the `Q32.32` grid only at the primitive boundaries: `Sh`, the row weight, `Hl` and
//!   `b` are rounded to `round(x·2³²)`, and the returned `c` is read back with `v/2³²`.
//! - `r_step` solves one `G × G` SPD system per out row (`M[o]·r[o] = t[o]`).
//! - `d_step` solves the `n × n` normal system of the Hadamard-domain column scale, with the rotated
//!   metric `H_rot = had_gram(Hl ⊙ c cᵀ)` and `G_rot = fwht128((Ws·Hl) ⊙ c)`. Its `A`-formation is
//!   the same shared primitive pair as `c_step` ([`crate::learn::solve::syrk::wsyrk`] then
//!   [`crate::learn::solve::syrk::hadamard_into`]) with `H_rot` as the elementwise factor, and its
//!   solve is [`crate::learn::solve::chol::chol_solve`]; the boundary rounding is identical to
//!   `c_step`'s, and the `f64` `A` loop and the [`crate::learn::ridge::solve_spd`] call are gone with
//!   no fallback.
//!
//! Every step adds the same ridge the script does, `1e-12·mean(diag)` (`d_step` clamps the mean
//! below at `1e-12`), so a recovery of an exact planted parameter is exact only up to the ridge; the
//! tests state that bound. `c_step` adds it to the `Q32.32` diagonal, where the grid's own quantum
//! `2⁻³²` floors it: a ridge below one raw unit is inert (see [`c_step`]).
//!
//! # Partiality
//!
//! Every function refuses a shape that disagrees with the declared dims, and a Hadamard function
//! refuses an `n` that is not a whole number of 128-blocks, with [`StepsError::ShapeMismatch`]. A
//! non-SPD `Hl` reaches the `Cholesky` pivot and is refused with [`StepsError::NotPositiveDefinite`].
//! No function here panics on caller input.
//!
//! # Trust boundary
//!
//! The bodies run in `f64` and are `tested` (`tests/learn_fold_steps.rs`); they are not extracted and
//! no refinement theorem links them, so they are not `proved`. The `c_step` and `d_step` Gram and
//! solve are the `Q32.32` primitives `wsyrk`, `hadamard_into` and `chol_solve`, the metric product
//! `Ws·Hl` is [`mul_metric`] over `gemm_nn`, and all of those are themselves `tested` and not
//! extracted; the `f64` work around them (`G_rot`, `H_rot`, the right-hand side, the scale rule's
//! operand scans and read-back) and `r_step`'s `f64` systems follow the script's arithmetic order;
//! no hardware or performance claim is made.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the flat-slice kernels index row-major `[o, n]` buffers by `o/i/j/gg/k/m`, matching \
              the reference arithmetic's vocabulary, and the `usize`-to-`f64` casts are the shape \
              means the reference forms (`mean(diag)`, `1/sqrt(128)`). The `f64 as i64` of \
              the ridge raw word is the Q32.32 boundary rounding of a ridge the step range-checks."
)]

use crate::learn::ridge::{solve_spd, RidgeError};
use crate::learn::solve::chol::{chol_solve, SolveError};
use crate::learn::solve::gemm::{gemm_nn_f64, GemmError};
use crate::learn::solve::q32::{self, from_grid, panel_shift, GRID_BOUND, GRID_SCALE, INPUT_BOUND};
use crate::learn::solve::syrk::{hadamard_into, wsyrk, SyrkError};

/// Hadamard block width: one normalised `H₁₂₈/√128` block per 128 columns.
pub const GROUP: usize = 128;

/// Ridge added to a normal-equation diagonal for `f64` (`eps` in the script).
const EPS: f64 = 1e-12;

/// The worker count handed to the two solve primitives by [`c_step`] and [`d_step`]. Neither step
/// takes a thread parameter and `c_step`'s sole production caller
/// ([`super::fold_fit::fold_fit`]) is single-threaded, so both fan out at `8`; both primitives are
/// documented bit-identical across thread counts, so this changes only wall-clock, not the returned
/// vector.
const STEP_THREADS: usize = 8;

/// [`q32::to_grid`] with the step's [`StepsError::ShapeMismatch`] for a non-finite or out-of-bound
/// value (no new error variant: the crate maps a domain violation onto the step's existing shape
/// refusal, as [`super::fold_fit`] does for its own conversions).
///
/// # Complexity
///
/// `O(1)`.
fn to_grid(x: f64, bound: f64) -> Result<i64, StepsError> {
    q32::to_grid(x, bound).ok_or(StepsError::ShapeMismatch)
}

/// A refusal from a fold-transfer step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepsError {
    /// A slice length disagrees with the declared shape, or a dimension is not a multiple of
    /// [`GROUP`] where a Hadamard step needs one.
    ShapeMismatch,
    /// A reduced matrix was not positive-definite: a non-SPD metric `hl`, or a rank-deficient
    /// normal system.
    NotPositiveDefinite,
}

/// Map a [`RidgeError`] from [`solve_spd`] onto a [`StepsError`]; the caller shape-checks first, so
/// the shape arm is the already-validated `cholesky` mismatch, kept total.
fn map_ridge(e: RidgeError) -> StepsError {
    match e {
        RidgeError::NotPositiveDefinite => StepsError::NotPositiveDefinite,
        RidgeError::ShapeMismatch | RidgeError::IndexOutOfRange => StepsError::ShapeMismatch,
    }
}

/// Map a [`SyrkError`] from [`wsyrk`]/[`hadamard_into`] onto a [`StepsError`]: a shape or empty
/// dimension keeps the shape refusal, and an input-bound or checked-`i128` overflow refusal is the
/// step's domain refusal [`StepsError::ShapeMismatch`].
///
/// # Complexity
///
/// `O(1)`.
fn map_syrk(e: SyrkError) -> StepsError {
    match e {
        SyrkError::ShapeMismatch
        | SyrkError::ZeroN
        | SyrkError::ZeroO
        | SyrkError::InputBound
        | SyrkError::Overflow => StepsError::ShapeMismatch,
    }
}

/// Map a [`GemmError`] from [`gemm_nn`] onto a [`StepsError`]: every arm is the step's domain
/// refusal [`StepsError::ShapeMismatch`], as for the other primitives — an empty dimension, a
/// length disagreeing with the declared shape, an entry at the input bound, or a contraction or
/// rounded entry that leaves the grid range.
///
/// # Complexity
///
/// `O(1)`.
fn map_gemm(e: GemmError) -> StepsError {
    match e {
        GemmError::ZeroDim
        | GemmError::ShapeMismatch
        | GemmError::InputBound
        | GemmError::Overflow => StepsError::ShapeMismatch,
    }
}

/// Map a [`SolveError`] from [`chol_solve`] onto a [`StepsError`]: a non-positive equilibration
/// diagonal or `Cholesky` pivot is the step's non-SPD refusal, and a shape, empty or overflow
/// refusal is the shape refusal.
///
/// # Complexity
///
/// `O(1)`.
fn map_solve(e: SolveError) -> StepsError {
    match e {
        SolveError::NotPositiveDefinite | SolveError::NonPositiveDiagonal => {
            StepsError::NotPositiveDefinite
        }
        SolveError::ShapeMismatch | SolveError::Empty | SolveError::Overflow => {
            StepsError::ShapeMismatch
        }
    }
}

/// The in-place normalised 128-block Walsh–Hadamard transform: each contiguous 128-chunk of `x` is
/// replaced by its `H₁₂₈/√128` image. `x.len()` must be a multiple of [`GROUP`] (the caller checks).
///
/// # Complexity
///
/// Temporal: `O(x.len())` — `7` butterfly stages over each 128-chunk, `64` add/sub per stage per
/// chunk, so `448` add/sub per chunk. The derivation is the radix-2 recurrence
/// `T(2^k) = 2·T(2^{k-1}) + 2^k` on a chunk with `k = 7`. Spatial: `O(1)` (in place).
fn fwht_in_place(x: &mut [f64]) {
    for chunk in x.chunks_mut(GROUP) {
        let mut h = 1;
        while h < GROUP {
            let mut i = 0;
            while i < GROUP {
                for j in i..(i + h) {
                    let a = chunk[j];
                    let b = chunk[j + h];
                    chunk[j] = a + b;
                    chunk[j + h] = a - b;
                }
                i += 2 * h;
            }
            h *= 2;
        }
        let inv = 1.0 / (GROUP as f64).sqrt();
        for v in chunk.iter_mut() {
            *v *= inv;
        }
    }
}

/// The normalised 128-block Walsh–Hadamard transform of `x` (a fresh buffer), `H₁₂₈/√128` applied to
/// each contiguous 128-chunk. The transform is its own inverse and preserves the Euclidean norm
/// (`(H/√128)ᵀ(H/√128) = I`), so `fwht128(fwht128(x)) = x`.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if `x.len()` is not a multiple of [`GROUP`].
///
/// # Complexity
///
/// Temporal: `O(x.len())` (see [`fwht_in_place`]). Spatial: `O(x.len())` f64 words for the copy.
pub fn fwht128(x: &[f64]) -> Result<Vec<f64>, StepsError> {
    if !x.len().is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let mut y = x.to_vec();
    fwht_in_place(&mut y);
    Ok(y)
}

/// The transpose of a row-major `rows × cols` matrix. `O(rows·cols)` time and space.
fn transpose(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

/// The congruent Gram `R·H·R` where `R` is the block-diagonal `H₁₂₈/√128` (`h` an `n × n` matrix,
/// row-major, `n = G·128`): transform the columns of `h` (right multiply by `R`), then its rows
/// (left multiply by `R`). `R` orthogonal makes the congruence norm-preserving; the script's
/// `had_gram(H) = fwht128(fwht128(H).T).T` is this same matrix.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if `h.len() != n·n`, `n == 0`, or `n` is not a multiple of
/// [`GROUP`].
///
/// # Complexity
///
/// Temporal: `O(n²·log 128)` — `7` butterfly stages over each of the `n²/128` row-blocks and the
/// `n²/128` column-blocks. Spatial: `O(n²)` f64 words for the copy and the transpose.
pub fn had_gram(h: &[f64], n: usize) -> Result<Vec<f64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) || h.len() != n * n {
        return Err(StepsError::ShapeMismatch);
    }
    // Rows: h ← h·R.
    let mut t = h.to_vec();
    fwht_in_place(&mut t);
    // Columns: transpose, transform rows, transpose back → R·h·R.
    let mut u = transpose(&t, n, n);
    fwht_in_place(&mut u);
    Ok(transpose(&u, n, n))
}

/// The reconstruction `compose(S, r, c)[o,i] = S[o,i]·r[o, i/128]·c[i]`, `O × n` row-major. `S` is
/// `O × n`, `r` is `O × G` (`G = n / 128`) and `c` is length `n`.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if `n == 0`, `n` is not a multiple of [`GROUP`], or the slice
/// lengths disagree with `O × n`, `O × G`, `n`.
///
/// # Complexity
///
/// Temporal: `O(O·n)` multiply-adds. Spatial: `O(O·n)` f64 words for the output.
pub fn compose(
    s: &[f64],
    r: &[f64],
    c: &[f64],
    o: usize,
    n: usize,
) -> Result<Vec<f64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let g = n / GROUP;
    if s.len() != o * n || r.len() != o * g || c.len() != n {
        return Err(StepsError::ShapeMismatch);
    }
    let mut w = vec![0.0f64; o * n];
    for oi in 0..o {
        for i in 0..n {
            w[oi * n + i] = s[oi * n + i] * r[oi * g + i / GROUP] * c[i];
        }
    }
    Ok(w)
}

/// The metric objective `J(Ws, Wp, Hl) = Σ_{o,i,j} D[o,i]·Hl[i,j]·D[o,j] = tr(D·Hl·Dᵀ)` with
/// `D = Ws − Wp`. `ws`, `wp` are `O × n` row-major; `hl` is `n × n`.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if the slice lengths disagree with `O × n`, `O × n`, `n × n`.
///
/// # Complexity
///
/// Temporal: `O(O·n²)` multiply-adds — one `n`-term reduction per `(o, j)` entry of `D·Hl`, then one
/// multiply-add per `(o, j)`. Spatial: `O(1)` in a scalar accumulator.
pub fn objective(
    ws: &[f64],
    wp: &[f64],
    hl: &[f64],
    o: usize,
    n: usize,
) -> Result<f64, StepsError> {
    if ws.len() != o * n || wp.len() != o * n || hl.len() != n * n {
        return Err(StepsError::ShapeMismatch);
    }
    let mut j = 0.0f64;
    for oo in 0..o {
        for col in 0..n {
            let mut y = 0.0f64;
            for i in 0..n {
                y += (ws[oo * n + i] - wp[oo * n + i]) * hl[i * n + col];
            }
            j += (ws[oo * n + col] - wp[oo * n + col]) * y;
        }
    }
    Ok(j)
}

/// The per-group scale `B[o,i] = s[o,i]·r[o, i/128]` of an `O × n` sign matrix and an `O × G` scale.
/// `O(O·n)` time and space.
fn group_scale(s: &[f64], r: &[f64], o: usize, n: usize, g: usize) -> Vec<f64> {
    let mut b = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            b[oo * n + i] = s[oo * n + i] * r[oo * g + i / GROUP];
        }
    }
    b
}

/// `Ws·Hl`, `O × n` row-major (`ws` is `O × n`, `hl` is `n × n`), through the shared `Q32.32`
/// product [`gemm_nn_f64`]: the operand scale rule, the grid rounding and the exact undo are its.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if a length disagrees, a value is non-finite or beyond the
/// [`gemm_nn_f64`] bound, or the product leaves `f64` ([`map_gemm`]). Never a panic.
///
/// # Complexity
///
/// As [`gemm_nn_f64`] at `m = O`, `k = n`, `n = n`, split across [`STEP_THREADS`] workers.
pub fn mul_metric(ws: &[f64], hl: &[f64], o: usize, n: usize) -> Result<Vec<f64>, StepsError> {
    if ws.len() != o * n || hl.len() != n * n {
        return Err(StepsError::ShapeMismatch);
    }
    if o == 0 || n == 0 {
        return Ok(vec![0.0f64; o * n]);
    }
    gemm_nn_f64(ws, hl, o, n, n, STEP_THREADS).map_err(map_gemm)
}

/// The `c_step`: one exact update of the per-column scale against the metric `Hl`, holding `S` and
/// `r`. Solves `A·c = b` with `Sh = S⊙r_rep`, `Sh_g = Sh⊙g_row`,
/// `A[i,j] = Hl[i,j]·Σ_o Sh[o,i]·Sh_g[o,j]` and `b[i] = Σ_o (Ws·Hl)[o,i]·Sh_g[o,i]`, adding the
/// ridge `1e-12·mean(diag(A))` to the diagonal. `g_row[i]` is the row weight (`1` when `None`).
///
/// `A` is symmetric — `Σ_o g_row[o]·Sh[o,i]·Sh[o,j]` and `Hl` are — so the `Cholesky` solve is valid.
/// When `Ws = compose(S, r, c0)` the un-ridged system has `c0` as its exact solution; the ridge moves
/// the returned `c` off `c0` by `λ·(A+λI)⁻¹c0`.
///
/// # The `Q32.32` boundary
///
/// The metric product `Ws·Hl` is [`mul_metric`] over the shared product [`gemm_nn`]; the
/// `A`-formation and the solve are the shared primitives [`wsyrk`], [`hadamard_into`] and
/// [`chol_solve`], not a second copy of the same math. This step stays `f64`; the primitives are
/// `Q32.32`, so the boundary is: `Ws`, `Hl`, `Sh`, the row weight and `b` are rounded to
/// `round(x·2³²)` (`Ws` and `Hl` after [`mul_metric`]'s exact power-of-two operand scale, undone on
/// its read-back); `wsyrk` forms the weighted Gram (the `hl` factor applied by `hadamard_into`) and
/// `chol_solve` takes it with the grid right-hand side `b`; the returned `c` is read back with
/// `v/2³²`. The ridge `1e-12·mean(diag A)` is added on the grid diagonal, so a ridge below one raw
/// unit (`2⁻³² ≈ 2.3e-10`) is inert — the grid's resolution is the floor; a positive ridge is kept
/// at one raw unit so the diagonal stays positive. A non-finite or out-of-bound value is refused at
/// the boundary, never wrapped.
///
/// The `Sh` magnitudes set the grid scale: `wsyrk`'s input bound is `|x| < 2³⁶` raw, so the panel
/// is multiplied by the exact power of two `2⁻ᵏ`, `k` read from `max|Sh|` (the least `k` with
/// `max|Sh|·2⁻ᵏ ≤ 8`). The panel scaling propagates as `A → A·2⁻²ᵏ` and `b → b·2⁻ᵏ`, so the
/// solution is `c·2ᵏ` and is scaled back by `2⁻ᵏ` before it is returned; `k = 0` when the panel is
/// already within `8`, which leaves the tests and the production fixture unscaled. The primitives
/// are handed [`STEP_THREADS`] workers; the result is bit-identical for every thread count.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if `n == 0`, `n` is not a multiple of [`GROUP`], a slice length
/// disagrees, `g_row` is present with the wrong length, or a value is non-finite or beyond the
/// boundary bound (`Ws`/`Hl`/`Sh`/`g_row` at `2³⁶` raw, `b` at the grid `i64` range);
/// [`StepsError::NotPositiveDefinite`] if the diagonal metric reduces to a non-SPD `A` (a non-SPD
/// `hl`).
///
/// # Complexity
///
/// Temporal: `O(O·n²)` checked `i128` multiply-accumulates for `Ws·Hl` ([`mul_metric`] over
/// [`gemm_nn`], split across [`STEP_THREADS`] workers — the term that replaced an `f64` triple loop
/// at the same shape), `O(O·n²/2)` for the `wsyrk` weighted Gram `A` measured in checked `i128`
/// multiply-accumulates (the upper triangle, mirrored), `O(n²)` for the `hl` hadamard factor,
/// `O(O·n)` for `b`, and `O(n³/3 + n²)` for the `chol_solve` factor and refinement measured in
/// checked `i128` multiply-accumulates. Spatial: `O(n²)` for the grid `A` (plus the `chol_solve`
/// working copies and factor) and `O(O·n)` for `Sh`, `Ws·Hl` and `b`.
pub fn c_step(
    ws: &[f64],
    hl: &[f64],
    s: &[f64],
    r: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
) -> Result<Vec<f64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let g = n / GROUP;
    if ws.len() != o * n || hl.len() != n * n || s.len() != o * n || r.len() != o * g {
        return Err(StepsError::ShapeMismatch);
    }
    if let Some(gr) = g_row {
        if gr.len() != o {
            return Err(StepsError::ShapeMismatch);
        }
    }
    let mut b_sh = group_scale(s, r, o, n, g);
    let wshl = mul_metric(ws, hl, o, n)?;
    let weight = |oo: usize| g_row.map_or(1.0f64, |gr| gr[oo]);
    // Scale the panel by the exact power of two `2⁻ᵏ` so it meets the `wsyrk` input bound
    // `|x| < 2³⁶` raw; `k` is read from `max|Sh|` (see [`panel_shift`]). The scaling is undone on
    // the returned `c` (`A → A·2⁻²ᵏ`, `b → b·2⁻ᵏ`, so the solution is `c·2ᵏ`).
    let k = panel_shift(b_sh.iter().fold(0.0f64, |m, v| m.max(v.abs())));
    if k != 0 {
        let f = 2f64.powi(-k);
        for v in &mut b_sh {
            *v *= f;
        }
    }
    // f64 → Q32.32 grid at the boundary: the panel and the row weight at `wsyrk`'s input bound, the
    // metric and the right-hand side at the grid's `i64` range.
    let sh_grid: Vec<i64> = b_sh
        .iter()
        .map(|&x| to_grid(x, INPUT_BOUND))
        .collect::<Result<Vec<i64>, StepsError>>()?;
    let gr_grid: Option<Vec<i64>> = match g_row {
        Some(gr) => Some(
            gr.iter()
                .map(|&x| to_grid(x, INPUT_BOUND))
                .collect::<Result<Vec<i64>, StepsError>>()?,
        ),
        None => None,
    };
    let hl_grid: Vec<i64> = hl
        .iter()
        .map(|&x| to_grid(x, GRID_BOUND))
        .collect::<Result<Vec<i64>, StepsError>>()?;
    // A = hl ⊙ (Shᵀ diag(g_row) Sh): the weighted Gram by [`wsyrk`], the `hl` factor by
    // [`hadamard_into`].
    let mut a = wsyrk(&sh_grid, gr_grid.as_deref(), o, n, STEP_THREADS).map_err(map_syrk)?;
    hadamard_into(&mut a, &hl_grid).map_err(map_syrk)?;
    // The ridge `1e-12·mean(diag A)` in the scaled system (`mean(diag A)` shares the `2⁻²ᵏ`
    // factor, so it cancels), at the grid quantum; a positive ridge is floored at one raw unit so
    // the diagonal stays positive.
    let mut dmean = 0.0f64;
    for i in 0..n {
        dmean += from_grid(a[i * n + i]);
    }
    dmean /= n as f64;
    let ridge = EPS * dmean;
    let mut ridge_raw = (ridge * GRID_SCALE).round() as i64;
    if ridge > 0.0 && ridge_raw == 0 {
        ridge_raw = 1;
    }
    for i in 0..n {
        let diag = a[i * n + i];
        a[i * n + i] = diag
            .checked_add(ridge_raw)
            .ok_or(StepsError::ShapeMismatch)?;
    }
    // b[i] = Σ_o (Ws·Hl)[o,i]·Sh_g[o,i]·g_row[o], on the scaled panel; then to the grid at the
    // `i64` range.
    let mut rhs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for oo in 0..o {
            acc += wshl[oo * n + i] * b_sh[oo * n + i] * weight(oo);
        }
        rhs[i] = acc;
    }
    let b_grid: Vec<i64> = rhs
        .iter()
        .map(|&x| to_grid(x, GRID_BOUND))
        .collect::<Result<Vec<i64>, StepsError>>()?;
    let solved = chol_solve(&a, &b_grid, n, 1, STEP_THREADS).map_err(map_solve)?;
    // Undo the `2⁻ᵏ` panel scale on the solution.
    let back = if k == 0 { 1.0 } else { 2f64.powi(-k) };
    Ok(solved.x.iter().map(|&v| from_grid(v) * back).collect())
}

/// The `r_step`: one exact update of the per-group scale `r` (`O × G`), holding the un-rotated sign
/// matrix `S` and the column scale `c`. Per out row `o` it solves the `G × G` SPD system
/// `M[o]·r[o] = t[o]` with `P[o,g,k] = S[o,g·128+k]·c[g·128+k]`,
/// `M[o,g,g'] = Σ_{k,k'} P[o,g,k]·Hl[g·128+k, g'·128+k']·P[o,g',k']` and
/// `t[o,g] = Σ_k P[o,g,k]·(Ws·Hl)[o, g·128+k]`, adding the ridge `1e-12·mean(diag(M[o]))`.
///
/// When `Ws = compose(S, r0, c)` each `r0[o]` is the un-ridged solution of its system.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if `n == 0`, `n` is not a multiple of [`GROUP`], or a slice length
/// disagrees; [`StepsError::NotPositiveDefinite`] if a reduced `M[o]` is not SPD (a non-SPD `hl`).
///
/// # Complexity
///
/// Temporal: `O(O·G·128·n)` to form the `K` blocks plus `O(O·G²·128)` for `M` and `t`, and
/// `O(O·G³)` for the per-row solves — `O(O·n²)` in total. Spatial: `O(n)` scratch (`K`, `P1`,
/// `Ws·Hl` row) plus `O(G²)` per-row system and `O(O·G)` output.
pub fn r_step(
    ws: &[f64],
    hl: &[f64],
    s: &[f64],
    c: &[f64],
    o: usize,
    n: usize,
) -> Result<Vec<f64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let g = n / GROUP;
    if ws.len() != o * n || hl.len() != n * n || s.len() != o * n || c.len() != n {
        return Err(StepsError::ShapeMismatch);
    }
    let mut out = vec![0.0f64; o * g];
    let mut p1 = vec![0.0f64; GROUP];
    let mut kvec = vec![0.0f64; n];
    let mut m = vec![0.0f64; g * g];
    let mut t = vec![0.0f64; g];
    let wshl = mul_metric(ws, hl, o, n)?;
    for oo in 0..o {
        for gg in 0..g {
            let mut acc = 0.0f64;
            for k in 0..GROUP {
                let idx = gg * GROUP + k;
                acc += s[oo * n + idx] * c[idx] * wshl[oo * n + idx];
            }
            t[gg] = acc;
        }
        for gg in 0..g {
            for k in 0..GROUP {
                let idx = gg * GROUP + k;
                p1[k] = s[oo * n + idx] * c[idx];
            }
            // K[m] = Σ_k P1[gg,k]·Hl[gg·128+k, m].
            for mm in 0..n {
                let mut acc = 0.0f64;
                for k in 0..GROUP {
                    acc += p1[k] * hl[(gg * GROUP + k) * n + mm];
                }
                kvec[mm] = acc;
            }
            for gg2 in 0..g {
                let mut acc = 0.0f64;
                for k in 0..GROUP {
                    let idx = gg2 * GROUP + k;
                    acc += kvec[idx] * s[oo * n + idx] * c[idx];
                }
                m[gg * g + gg2] = acc;
            }
        }
        let mut dmean = 0.0f64;
        for gg in 0..g {
            dmean += m[gg * g + gg];
        }
        dmean /= g as f64;
        let ridge = EPS * dmean;
        for gg in 0..g {
            m[gg * g + gg] += ridge;
        }
        let sol = solve_spd(&m, &t, g, 1).map_err(map_ridge)?;
        for gg in 0..g {
            out[oo * g + gg] = sol[gg];
        }
    }
    Ok(out)
}

/// The `d_step`: the exact Hadamard-domain column-scale minimizer, holding `S_rot`, `r` and `c`.
/// Solves `A·d = b` with `B = S_rot⊙r_rep`, `B_g = B⊙g_row`, `G_rot = fwht128((Ws·Hl)⊙c)`,
/// `H_rot = had_gram(Hl⊙c cᵀ)`, `A[i,j] = H_rot[i,j]·Σ_o B[o,i]·B_g[o,j]` and
/// `b[i] = Σ_o B_g[o,i]·G_rot[o,i]`, adding the ridge `1e-12·max(mean(diag(A)), 1e-12)`.
///
/// The forward model it minimizes is `recon(d) = compose(fwht128(S_rot⊙d), r, c)` — the script's
/// `run_variant` refit (`spec/fold_transfer.py:886`). Because `r_rep` is constant on each 128-block
/// and `fwht128` is block-local, `r` commutes with the transform, so
/// `recon(d) = compose(fwht128(B⊙d), ones, c)` and `A`, `b` are the gradient of `J(Ws, recon(d), Hl)`.
///
/// # The `Q32.32` boundary
///
/// The `A`-formation and the solve are the shared primitives [`wsyrk`] (the weighted Gram
/// `Bᵀ diag(g_row) B`), [`hadamard_into`] (the rotated metric `H_rot` as the elementwise factor) and
/// [`chol_solve`], not a second copy of the same math. There is no `f64` `A` loop and no
/// [`solve_spd`] fallback: the step stays `f64` and crosses into the grid only at this boundary.
/// `B`, the row weight and `H_rot` are rounded to `round(x·2³²)`; `chol_solve` takes the grid `A`
/// with the grid right-hand side `b`; the returned `d` is read back with `v/2³²`. As in [`c_step`]
/// the panel is shifted by the exact power of two `2⁻ᵏ` read from `max|B|`, so it meets the `wsyrk`
/// input bound `|x| < 2³⁶` raw; the shift propagates as `A → A·2⁻²ᵏ` and `b → b·2⁻ᵏ`, so the
/// solution is `d·2ᵏ` and is scaled back by `2⁻ᵏ` before it is returned (`k = 0` when the panel is
/// already within `8`). The ridge is the script's `1e-12·max(mean(diag A), 1e-12)`, its clamp
/// evaluated on the unshifted `mean(diag A)`, and it is added on the grid diagonal; a positive ridge
/// is floored at one raw unit so the diagonal stays positive. A non-finite or out-of-bound value is
/// refused at the boundary, never wrapped.
///
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if `n == 0`, `n` is not a multiple of [`GROUP`], a slice length
/// disagrees, `g_row` is present with the wrong length, or a value is non-finite or beyond the
/// boundary bound (`Ws`/`Hl`/`B`/`g_row` at `2³⁶` raw, `H_rot`/`b` at the grid `i64` range);
/// [`StepsError::NotPositiveDefinite`] if the reduced `A` is not SPD.
///
/// # Complexity
///
/// Temporal: `O(O·n²)` checked `i128` multiply-accumulates for `Ws·Hl` ([`mul_metric`] over
/// [`gemm_nn`], split across [`STEP_THREADS`] workers), `O(O·n·log 128)` for the transform and
/// `O(n²·log 128)` for `H_rot` (f64), `O(O·n²/2)` for the `wsyrk` weighted Gram `A` measured in
/// checked `i128` multiply-accumulates (the upper triangle, mirrored), `O(n²)` for the
/// `hadamard_into` factor, `O(O·n)` for `b`, and `O(n³/3 + n²)` for the `chol_solve` factor and
/// refinement measured in checked `i128` multiply-accumulates. Spatial: `O(n²)` for `H_rot` and the
/// grid `A` (plus the `chol_solve` working copies and factor), `O(O·n)` for `B`, `Ws·Hl` and
/// `G_rot`.
#[allow(
    clippy::too_many_arguments,
    reason = "the step is handed the metric, the rotated sign matrix, the two scales, the row \
              weights and the geometry; bundling them would only relocate the fields."
)]
pub fn d_step(
    ws: &[f64],
    hl: &[f64],
    s_rot: &[f64],
    r: &[f64],
    c: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
) -> Result<Vec<f64>, StepsError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(StepsError::ShapeMismatch);
    }
    let g = n / GROUP;
    if ws.len() != o * n
        || hl.len() != n * n
        || s_rot.len() != o * n
        || r.len() != o * g
        || c.len() != n
    {
        return Err(StepsError::ShapeMismatch);
    }
    if let Some(gr) = g_row {
        if gr.len() != o {
            return Err(StepsError::ShapeMismatch);
        }
    }
    let wshl = mul_metric(ws, hl, o, n)?;
    let weight = |oo: usize| g_row.map_or(1.0f64, |gr| gr[oo]);
    // G_rot[o,i] = fwht128((Ws·Hl) ⊙ c)[o,i].
    let mut g_rot = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            g_rot[oo * n + i] = wshl[oo * n + i] * c[i];
        }
    }
    fwht_in_place(&mut g_rot);
    // H_rot = had_gram(Hl ⊙ c cᵀ).
    let mut hlc = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            hlc[i * n + j] = hl[i * n + j] * c[i] * c[j];
        }
    }
    let h_rot = had_gram(&hlc, n)?;
    let mut b_scaled = group_scale(s_rot, r, o, n, g);
    // Scale the panel by the exact power of two `2⁻ᵏ` so it meets the `wsyrk` input bound
    // `|x| < 2³⁶` raw; `k` is read from `max|B|` (see [`panel_shift`]). The scaling is undone on the
    // returned `d` (`A → A·2⁻²ᵏ`, `b → b·2⁻ᵏ`, so the solution is `d·2ᵏ`).
    let k = panel_shift(b_scaled.iter().fold(0.0f64, |m, v| m.max(v.abs())));
    if k != 0 {
        let f = 2f64.powi(-k);
        for v in &mut b_scaled {
            *v *= f;
        }
    }
    // f64 → Q32.32 grid at the boundary: the panel and the row weight at `wsyrk`'s input bound, the
    // rotated metric and the right-hand side at the grid's `i64` range.
    let panel_grid: Vec<i64> = b_scaled
        .iter()
        .map(|&x| to_grid(x, INPUT_BOUND))
        .collect::<Result<Vec<i64>, StepsError>>()?;
    let gr_grid: Option<Vec<i64>> = match g_row {
        Some(gr) => Some(
            gr.iter()
                .map(|&x| to_grid(x, INPUT_BOUND))
                .collect::<Result<Vec<i64>, StepsError>>()?,
        ),
        None => None,
    };
    let h_grid: Vec<i64> = h_rot
        .iter()
        .map(|&x| to_grid(x, GRID_BOUND))
        .collect::<Result<Vec<i64>, StepsError>>()?;
    // A = H_rot ⊙ (Bᵀ diag(g_row) B): the weighted Gram by [`wsyrk`], the rotated-metric factor by
    // [`hadamard_into`].
    let mut a = wsyrk(&panel_grid, gr_grid.as_deref(), o, n, STEP_THREADS).map_err(map_syrk)?;
    hadamard_into(&mut a, &h_grid).map_err(map_syrk)?;
    // The ridge `1e-12·max(mean(diag A), 1e-12)` at the grid quantum. The `2⁻²ᵏ` panel factor on `A`
    // cancels between the diagonal mean and the diagonal it is added to, so the clamp `1e-12` is
    // written in the scaled system (`1e-12·2⁻²ᵏ`), which is the script's clamp on the unshifted
    // `mean(diag A)`; a positive ridge is floored at one raw unit so the diagonal stays positive.
    let mut dmean = 0.0f64;
    for i in 0..n {
        dmean += from_grid(a[i * n + i]);
    }
    dmean /= n as f64;
    let ridge = EPS * dmean.max(EPS * 2f64.powi(-2 * k));
    let mut ridge_raw = (ridge * GRID_SCALE).round() as i64;
    if ridge > 0.0 && ridge_raw == 0 {
        ridge_raw = 1;
    }
    for i in 0..n {
        let diag = a[i * n + i];
        a[i * n + i] = diag
            .checked_add(ridge_raw)
            .ok_or(StepsError::ShapeMismatch)?;
    }
    // b[i] = Σ_o B[o,i]·g_row[o]·G_rot[o,i], on the scaled panel; then to the grid at the `i64`
    // range.
    let mut rhs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for oo in 0..o {
            acc += b_scaled[oo * n + i] * weight(oo) * g_rot[oo * n + i];
        }
        rhs[i] = acc;
    }
    let b_grid: Vec<i64> = rhs
        .iter()
        .map(|&x| to_grid(x, GRID_BOUND))
        .collect::<Result<Vec<i64>, StepsError>>()?;
    let solved = chol_solve(&a, &b_grid, n, 1, STEP_THREADS).map_err(map_solve)?;
    // Undo the `2⁻ᵏ` panel scale on the solution.
    let back = if k == 0 { 1.0 } else { 2f64.powi(-k) };
    Ok(solved.x.iter().map(|&v| from_grid(v) * back).collect())
}
