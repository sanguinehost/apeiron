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
//!   `Sh = S⊙r_rep`, `Sh_g = Sh⊙g_row`, `b[i] = Σ_o (Ws·Hl)[o,i]·Sh_g[o,i]`.
//! - `r_step` solves one `G × G` SPD system per out row (`M[o]·r[o] = t[o]`).
//! - `d_step` solves the `n × n` normal system of the Hadamard-domain column scale, with the rotated
//!   metric `H_rot = had_gram(Hl ⊙ c cᵀ)` and `G_rot = fwht128((Ws·Hl) ⊙ c)`.
//!
//! Every step adds the same f64 ridge the script does, `1e-12·mean(diag)` (`d_step` clamps the mean
//! below at `1e-12`), so a recovery of an exact planted parameter is exact only up to the ridge; the
//! tests state that bound.
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
//! no refinement theorem links them, so they are not `proved`. The `f64` arithmetic order is the
//! script's; no hardware or performance claim is made.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    reason = "the flat-slice kernels index row-major `[o, n]` buffers by `o/i/j/gg/k/m`, matching \
              the reference arithmetic's vocabulary, and the `usize`-to-`f64` casts are the shape \
              means the reference forms (`mean(diag)`, `1/sqrt(128)`)."
)]

use crate::learn::ridge::{solve_spd, RidgeError};

/// Hadamard block width: one normalised `H₁₂₈/√128` block per 128 columns.
pub const GROUP: usize = 128;

/// Ridge added to a normal-equation diagonal for `f64` (`eps` in the script).
const EPS: f64 = 1e-12;

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

/// `Ws·Hl`, `O × n` row-major (`ws` is `O × n`, `hl` is `n × n`). `O(O·n²)` time, `O(O·n)` space.
fn mul_metric(ws: &[f64], hl: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; o * n];
    for oo in 0..o {
        for j in 0..n {
            let mut acc = 0.0f64;
            for i in 0..n {
                acc += ws[oo * n + i] * hl[i * n + j];
            }
            y[oo * n + j] = acc;
        }
    }
    y
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
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if `n == 0`, `n` is not a multiple of [`GROUP`], a slice length
/// disagrees, or `g_row` is present with the wrong length; [`StepsError::NotPositiveDefinite`] if the
/// diagonal metric reduces to a non-SPD `A` (a non-SPD `hl`).
///
/// # Complexity
///
/// Temporal: `O(O·n²)` to form `A`, `O(O·n)` for `Ws·Hl` and `b`, and `O(n³)` for the `Cholesky`
/// solve. Spatial: `O(n²)` for `A` plus `O(O·n)` for `Ws·Hl`.
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
    let b_sh = group_scale(s, r, o, n, g);
    let wshl = mul_metric(ws, hl, o, n);
    let weight = |oo: usize| g_row.map_or(1.0f64, |gr| gr[oo]);
    // A[i,j] = hl[i,j] · Σ_o Sh[o,i]·Sh[o,j]·g_row[o].
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for oo in 0..o {
                acc += b_sh[oo * n + i] * b_sh[oo * n + j] * weight(oo);
            }
            a[i * n + j] = hl[i * n + j] * acc;
        }
    }
    let mut dmean = 0.0f64;
    for i in 0..n {
        dmean += a[i * n + i];
    }
    dmean /= n as f64;
    let ridge = EPS * dmean;
    for i in 0..n {
        a[i * n + i] += ridge;
    }
    // b[i] = Σ_o (Ws·Hl)[o,i]·Sh[o,i]·g_row[o].
    let mut rhs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for oo in 0..o {
            acc += wshl[oo * n + i] * b_sh[oo * n + i] * weight(oo);
        }
        rhs[i] = acc;
    }
    solve_spd(&a, &rhs, n, 1).map_err(map_ridge)
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
    let wshl = mul_metric(ws, hl, o, n);
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
/// # Errors
///
/// [`StepsError::ShapeMismatch`] if `n == 0`, `n` is not a multiple of [`GROUP`], a slice length
/// disagrees, or `g_row` is present with the wrong length; [`StepsError::NotPositiveDefinite`] if the
/// reduced `A` is not SPD.
///
/// # Complexity
///
/// Temporal: `O(O·n²)` to form `A` and `Ws·Hl`, plus `O(O·n·log 128)` for the transform and
/// `O(n²·log 128)` for `H_rot`, and `O(n³)` for the `Cholesky` solve. Spatial: `O(n²)` for `A` and
/// `H_rot`, plus `O(O·n)` for `Ws·Hl` and `G_rot`.
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
    let wshl = mul_metric(ws, hl, o, n);
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
    let b_scaled = group_scale(s_rot, r, o, n, g);
    // A[i,j] = H_rot[i,j] · Σ_o B[o,i]·B[o,j]·g_row[o].
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for oo in 0..o {
                acc += b_scaled[oo * n + i] * b_scaled[oo * n + j] * weight(oo);
            }
            a[i * n + j] = h_rot[i * n + j] * acc;
        }
    }
    let mut dmean = 0.0f64;
    for i in 0..n {
        dmean += a[i * n + i];
    }
    dmean /= n as f64;
    let ridge = EPS * dmean.max(1e-12);
    for i in 0..n {
        a[i * n + i] += ridge;
    }
    // b[i] = Σ_o B[o,i]·g_row[o]·G_rot[o,i].
    let mut rhs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for oo in 0..o {
            acc += b_scaled[oo * n + i] * weight(oo) * g_rot[oo * n + i];
        }
        rhs[i] = acc;
    }
    solve_spd(&a, &rhs, n, 1).map_err(map_ridge)
}
