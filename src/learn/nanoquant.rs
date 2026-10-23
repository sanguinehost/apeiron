//! The NanoQuant binary low-rank factorisation: Samsung Labs, arXiv 2602.06694, ported from
//! `SamsungLabs/NanoQuant` @ `a9e0a43`, `src/nanoquant/core/admm_nq.py` (`factorize_admm_nanoquant`)
//! and `src/nanoquant/utils/utils.py:128` (`calculate_ranks`).
//!
//! `Ŵ = diag(s_post)·U·V·diag(s_pre)`, `U ∈ {±1}^{out×r}`, `V ∈ {±1}^{r×in}`; the stored bits are
//! `r·(out + in) + 16·(out + in)` (two binary16 scale vectors). The factors come from the ADMM of
//! `admm_nq.py`: a ridge least-squares `X`-update, a sign-value-independent rank-one `Z`-update
//! (`sign(M) ⊙ u vᵀ`, `u σ vᵀ` the top singular triplet of `|M|` by power iteration) and the scaled
//! dual `U`-update, under the penalty schedule [`Schedule`] (the reference CLI default is linear,
//! `ρ = t/T`, `main.py:81`; `cubic`, `ρ = (t/T)³`, is the function signature's default).
//!
//! Every matrix product runs on the kernel through [`crate::engine::kern::limb`] (AGENTS.md §0 K2):
//! the Grams and the right-hand sides by prepared products, the solves by `gemm_q32`, the power
//! iteration by prepared one-column products and the norms by `rowdot_q32`. The `r × r` ridge
//! system is solved as `F = H⁻¹R` with `H⁻¹` from the block Schur-complement recursion
//! [`spd_inverse`], so the `O(r²·k)` substitution is one kernel product instead of a scalar
//! triangular solve.
//!
//! Deviations from the reference, each a stated input, not a silent change: the Gaussian draws
//! come from [`Rng`] (splitmix64 and Box–Muller), not `torch.randn`; the `B` power iteration runs
//! on the transposed frame with its start vector on the same (input) side as the reference.

#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "single letters are the reference's symbols (`o`, `n`, `r`, `k`, `w`); the casts are \
              dimensions and 53-bit draws far below 2^52; every loop is one row-major index map"
)]

use std::sync::atomic::{AtomicBool, Ordering};

use crate::engine::chart::pool::{width_global, Pool};
use crate::engine::kern::limb::{gemm_q32, prepare, rowdot_q32, syrk_q32, LimbError};
use crate::learn::solve::chol::SolveError;
use crate::learn::solve::inv::inv_factor_upper;
use crate::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// Limb count of every kernel call here: 3, the fp32-reference precision (AGENTS.md §0 K3).
const LIMBS: u32 = 3;

/// The reference's `eps` clamp on norms (`admm_nq.py`, `eps=1e-12`).
const EPS: f64 = 1e-12;

/// A refusal of [`factorize`] or [`NqFactor::weight`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NqError {
    /// A slice length disagrees with the declared shape, or a dimension or the rank is zero.
    Shape,
    /// A value left the Q32.32 grid or is not finite.
    Range,
    /// The limb kernel refused a product (shape or scale placement).
    Kernel,
    /// The ridge system's inverse factor was refused.
    Solve(SolveError),
}

impl From<LimbError> for NqError {
    fn from(_: LimbError) -> Self {
        Self::Kernel
    }
}

impl From<SolveError> for NqError {
    fn from(e: SolveError) -> Self {
        Self::Solve(e)
    }
}

/// The two-scale rank of an `a × b` matrix at `bits_ppm` millionths of a bit per weight
/// (`calculate_ranks`, `num_scales = 2`): `r = ⌊a·b·bits/(a + b)⌋ − 16`, floored to a multiple of
/// 32, and `min_rank` when that is zero or below.
#[must_use]
pub fn rank_for_bits(a: usize, b: usize, bits_ppm: u64, min_rank: usize) -> usize {
    let ab = (a as u128) * (b as u128) * u128::from(bits_ppm);
    let per = ab / (1_000_000u128 * (a as u128 + b as u128));
    let r = (per as usize).saturating_sub(16) / 32 * 32;
    r.max(min_rank)
}

/// Stored bits of a two-scale rank-`r` factor of an `a × b` matrix: `r·(a + b) + 16·(a + b)`.
#[must_use]
pub fn stored_bits(a: usize, b: usize, r: usize) -> u64 {
    ((r + 16) * (a + b)) as u64
}

/// ADMM settings; [`NqConfig::reference`] is the README's `--admm_outer_iters 400` run.
#[derive(Debug, Clone, Copy)]
pub struct NqConfig {
    /// The rank `r`.
    pub rank: usize,
    /// Outer ADMM iterations `T` (reference 400).
    pub outer_iters: usize,
    /// Power-iteration steps of each rank-one `Z`-update (reference 5).
    pub inner_iters: usize,
    /// The ridge term added to the normalised system's diagonal (reference `3e-2`).
    pub reg: f64,
    /// Seed of the Gaussian initialisation and the power-iteration start vectors.
    pub seed: u64,
    /// The penalty schedule `ρ(t/T)`.
    pub schedule: Schedule,
}

/// The ADMM penalty schedule `ρ(x)`, `x = t/T` (`admm_nq.py:255-277`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// `ρ = x`: the reference CLI and `NanoQuantConfig` default.
    Linear,
    /// `ρ = min(1, x)³`.
    Cubic,
}

impl Schedule {
    fn rho(self, x: f64) -> f64 {
        match self {
            Self::Linear => x,
            Self::Cubic => x.min(1.0).powi(3),
        }
    }
}

impl NqConfig {
    /// The reference settings at rank `rank`.
    #[must_use]
    pub fn reference(rank: usize) -> Self {
        Self {
            rank,
            outer_iters: 400,
            inner_iters: 5,
            reg: 3e-2,
            seed: 0,
            schedule: Schedule::Linear,
        }
    }
}

/// The deployed factor: `Ŵ[i][j] = s_post[i]·Σ_k U[i][k]·V[k][j]·s_pre[j]`.
#[derive(Debug, Clone)]
pub struct NqFactor {
    /// Output rows `out`.
    pub out: usize,
    /// Input columns `in`.
    pub inn: usize,
    /// Rank `r`.
    pub rank: usize,
    /// `U`, row-major `out × r`, entries `±1`.
    pub u: Vec<i8>,
    /// `V`, row-major `r × in`, entries `±1`.
    pub v: Vec<i8>,
    /// `s_pre`, length `in`.
    pub s_pre: Vec<f64>,
    /// `s_post`, length `out`.
    pub s_post: Vec<f64>,
    /// The ADMM's own normalised loss `‖W_n − (A_z/m)·B_z‖²/‖W_n‖²` at exit (`admm_nq.py`'s
    /// printed `Loss`, `m` the row norms of `B_z`).
    pub admm_loss: f64,
    /// `U`'s latent, row-major `out × r`: `(A_ls + A_u)/norm_o·bf` (`admm_nq.py:215-232`);
    /// its sign is `u` wherever the rank-one magnitude is nonzero. The reference's `tune_fact`
    /// trains it through the sign.
    pub u_latent: Vec<f64>,
    /// `V`'s latent, row-major `r × in`: `(B_ls + B_u)/norm_i/bf`, the same way.
    pub v_latent: Vec<f64>,
}

/// Deterministic Gaussian source: splitmix64 words through Box–Muller.
struct Rng(u64);

impl Rng {
    fn word(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn normal(&mut self) -> f64 {
        let u1 = ((self.word() >> 11) as f64 + 1.0) / 9_007_199_254_740_993.0;
        let u2 = (self.word() >> 11) as f64 / 9_007_199_254_740_992.0;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Below this many entries an elementwise pass runs on the caller: a pool dispatch costs more than
/// the pass.
const PAR_MIN: usize = 1 << 14;

/// `f(c, &mut out[c])` for every `c`, one contiguous chunk per pool worker (under [`PAR_MIN`]
/// entries on the caller). Each entry is computed alone, so the result is the sequential one.
/// `O(len)` calls of `f`.
fn par_fill<T: Send>(out: &mut [T], f: &(impl Fn(usize, &mut T) + Sync)) {
    if out.len() < PAR_MIN {
        for (c, o) in out.iter_mut().enumerate() {
            f(c, o);
        }
        return;
    }
    let chunk = out.len().div_ceil(width_global());
    Pool::global().chunks_mut(out, chunk, &|_, start, piece: &mut [T]| {
        for (k, o) in piece.iter_mut().enumerate() {
            f(start + k, o);
        }
    });
}

/// `to_grid(f(c))` for `c < len`, through [`par_fill`]; [`NqError::Range`] if any entry leaves the
/// grid. `O(len)` calls of `f`.
fn grid_map(len: usize, f: &(impl Fn(usize) -> f64 + Sync)) -> Result<Vec<i64>, NqError> {
    let bad = AtomicBool::new(false);
    let mut out = vec![0i64; len];
    par_fill(&mut out, &|c, o| match to_grid(f(c), GRID_BOUND) {
        Some(w) => *o = w,
        None => bad.store(true, Ordering::Relaxed),
    });
    if bad.into_inner() {
        Err(NqError::Range)
    } else {
        Ok(out)
    }
}

/// Every entry of `x` on the Q32.32 grid. `O(len)`.
fn grid(x: &[f64]) -> Result<Vec<i64>, NqError> {
    grid_map(x.len(), &|c| x[c])
}

/// Row-major `rows × cols` transposed to `cols × rows`. `O(rows·cols)` moves.
fn transpose<T: Copy + Default + Send + Sync>(m: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut t = vec![T::default(); m.len()];
    par_fill(&mut t, &|c, o| *o = m[(c % rows) * cols + c / rows]);
    t
}

/// `‖m‖_F²` of a row-major `rows × cols` matrix: one `rowdot_q32` of `m` with itself, summed.
/// `O(rows·cols·limbs²)` kernel byte products.
fn fro2(m: &[f64], rows: usize, cols: usize) -> Result<f64, NqError> {
    let q = grid(m)?;
    let d = rowdot_q32(&q, &q, rows, cols, LIMBS)?;
    Ok(d.iter().map(|&w| from_grid(w)).sum())
}

/// The reference's `rank1_approx`: `sign(M) ⊙ (σu)vᵀ` with `(u, σ, v)` the top singular triplet
/// of `|M|` after `iters` power steps from a Gaussian start. `start_rows` puts the start vector on
/// the row side (the reference's `B`, run here on `Bᵀ`), else on the column side.
///
/// `O(iters·rows·cols·limbs²)` kernel byte products through two prepared planes (`|M|` and
/// `|M|ᵀ`, `O(rows·cols·limbs)` bytes each), plus `O(rows·cols)` for the sign product.
fn rank1(
    m: &[f64],
    rows: usize,
    cols: usize,
    iters: usize,
    start_rows: bool,
    rng: &mut Rng,
) -> Result<Vec<f64>, NqError> {
    let abs: Vec<f64> = m.iter().map(|x| x.abs()).collect();
    let pa = prepare(&grid(&abs)?, rows, cols, LIMBS)?;
    let pt = prepare(&grid(&transpose(&abs, rows, cols))?, cols, rows, LIMBS)?;
    // `|M|·x` (x on the column side) and `|M|ᵀ·y` (y on the row side), each one prepared product.
    let mul = |x: &[f64]| -> Result<Vec<f64>, NqError> {
        Ok(pa
            .gemm(&grid(x)?, 1)?
            .iter()
            .map(|&w| from_grid(w))
            .collect())
    };
    let mul_t = |y: &[f64]| -> Result<Vec<f64>, NqError> {
        Ok(pt
            .gemm(&grid(y)?, 1)?
            .iter()
            .map(|&w| from_grid(w))
            .collect())
    };
    let unit = |x: Vec<f64>| -> Result<Vec<f64>, NqError> {
        let n = fro2(&x, 1, x.len())?.sqrt().max(EPS);
        Ok(x.into_iter().map(|e| e / n).collect())
    };
    let (u, v) = if start_rows {
        // Reference frame `Mᵀ` (`cols × rows`): its column-side start is our row side.
        let mut a = unit((0..rows).map(|_| rng.normal()).collect())?;
        for _ in 0..iters {
            let b = unit(mul_t(&a)?)?;
            a = unit(mul(&b)?)?;
        }
        let b_un = mul_t(&a)?;
        // The reference's `u` (σ folded in) is our column vector, its unit `v` our row vector.
        (a, b_un)
    } else {
        let mut v = unit((0..cols).map(|_| rng.normal()).collect())?;
        for _ in 0..iters {
            let u = unit(mul(&v)?)?;
            v = unit(mul_t(&u)?)?;
        }
        (mul(&v)?, v)
    };
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let s = if m[i * cols + j] < 0.0 { -1.0 } else { 1.0 };
            out[i * cols + j] = s * u[i] * v[j];
        }
    }
    Ok(out)
}

/// Order at or below which [`spd_inverse`] stops recursing and takes `UᵀU` from the exact inverse
/// factor. Measured at order 768 (`spd_inverse_base_wall`): 12.2 ms at 32 against 17.0 ms at 192;
/// residual at order 300 (`spd_inverse_inverts`) 7.7e-6 against 3.4e-5 without recursion.
const INV_BASE: usize = 32;

/// The `[r0, r1) × [c0, c1)` block of a row-major matrix with `cols` columns.
///
/// `O((r1 − r0)·(c1 − c0))` word copies and words.
fn block(m: &[i64], cols: usize, r0: usize, r1: usize, c0: usize, c1: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity((r1 - r0) * (c1 - c0));
    for i in r0..r1 {
        out.extend_from_slice(&m[i * cols + c0..i * cols + c1]);
    }
    out
}

/// `H⁻¹` of an SPD `h` (row-major `n × n`, Q32.32), symmetric, by the block Schur-complement
/// recursion on `H = [A B; Bᵀ C]` (`A` of order `p = ⌊n/2⌋`, `C` of order `q = n − p`):
///
/// ```text
/// X = A⁻¹B,  S = C − BᵀX,  Y = X·S⁻¹,  H⁻¹ = [A⁻¹ + Y·Xᵀ, −Y; −Yᵀ, S⁻¹],
/// ```
///
/// the four products on the kernel at [`LIMBS`] (the fp32-reference precision of the products
/// that consume `H⁻¹`), `A⁻¹` and `S⁻¹` by recursion; at order `≤ INV_BASE` the base is `UᵀU` with
/// `U` from [`inv_factor_upper`]. `S` is SPD whenever `H` is (Schur complement of an SPD matrix).
///
/// # Errors
///
/// [`NqError::Solve`] as [`inv_factor_upper`] refuses a base block (a computed `S` that is not SPD
/// lands here); [`NqError::Kernel`] as a product refuses; [`NqError::Range`] if a sum leaves `i64`.
///
/// # Complexity
///
/// Four products of half order per level, `p·p·q + q·q·p + p·q·q + p·p·q = n³/2` multiply-adds at
/// `p = q = n/2`, so `T(n) = n³/2 + 2·T(n/2)` closes at `2n³/3` kernel multiply-adds, plus `n/INV_BASE`
/// bases of `INV_BASE³/2` checked `i128` multiply-adds and an `INV_BASE³/2` kernel Gram each.
/// `O(n²)` words live per level.
fn spd_inverse(h: &[i64], n: usize) -> Result<Vec<i64>, NqError> {
    spd_inverse_at(h, n, INV_BASE)
}

/// [`spd_inverse`] with the recursion base order `base` as an argument.
fn spd_inverse_at(h: &[i64], n: usize, base: usize) -> Result<Vec<i64>, NqError> {
    if n <= base {
        let u = inv_factor_upper(h, n)?;
        return Ok(syrk_q32(&transpose(&u, n, n), n, n, LIMBS)?);
    }
    let (p, q) = (n / 2, n - n / 2);
    let ai = spd_inverse_at(&block(h, n, 0, p, 0, p), p, base)?;
    let bt = block(&transpose(h, n, n), n, p, n, 0, p);
    let x = gemm_q32(&ai, p, &bt, q, p, LIMBS)?;
    let btx = gemm_q32(&bt, q, &transpose(&x, p, q), q, p, LIMBS)?;
    let c = block(h, n, p, n, p, n);
    let mut s = vec![0i64; q * q];
    for i in 0..q {
        for j in 0..q {
            // `(BᵀX)ᵢⱼ` and `(BᵀX)ⱼᵢ` agree in exact arithmetic; the mean keeps `S` symmetric.
            let m = (btx[i * q + j] >> 1) + (btx[j * q + i] >> 1);
            s[i * q + j] = c[i * q + j].checked_sub(m).ok_or(NqError::Range)?;
        }
    }
    let si = spd_inverse_at(&s, q, base)?;
    let y = gemm_q32(&x, p, &si, q, q, LIMBS)?;
    let yx = gemm_q32(&y, p, &x, p, q, LIMBS)?;
    let mut out = vec![0i64; n * n];
    for i in 0..p {
        for j in 0..p {
            let m = (yx[i * p + j] >> 1) + (yx[j * p + i] >> 1);
            out[i * n + j] = ai[i * p + j].checked_add(m).ok_or(NqError::Range)?;
        }
        for j in 0..q {
            let v = 0i64.checked_sub(y[i * q + j]).ok_or(NqError::Range)?;
            out[i * n + p + j] = v;
            out[(p + j) * n + i] = v;
        }
    }
    for i in 0..q {
        out[(p + i) * n + p..(p + i) * n + n].copy_from_slice(&si[i * q..i * q + q]);
    }
    Ok(out)
}

/// One ridge `X`-update in the transposed frame: returns `Fᵀ` (`k × r`) with
/// `F = (XᵀX + (ρ·mean diag + reg)·I)⁻¹ (XᵀY + ρ(Z − U))`, `X = Z_x/‖·‖` column-normalised.
///
/// `zx` is the `n × r` normalising factor (`Bᵀ_z` or `A_z`), `wp` the prepared `k × n` target
/// (`W_n` for `A`, `W_nᵀ` for `B`), `z`/`du` the `k × r` consensus and dual.
///
/// `O(r²·n + r·n·k + 2r³/3 + r²·k)` multiply-adds: the full-square Gram, the right-hand side, the
/// inverse ([`spd_inverse`]) and `Rᵀ·H⁻¹`, all on the kernel but the inverse's order-32 bases.
#[allow(
    clippy::too_many_arguments,
    reason = "the ADMM step's operands are distinct matrices"
)]
fn x_update(
    zx: &[f64],
    n: usize,
    r: usize,
    wp: &crate::engine::kern::limb::Prepared,
    k: usize,
    z: &[f64],
    du: &[f64],
    rho: f64,
    reg: f64,
) -> Result<Vec<f64>, NqError> {
    // `Z_xᵀ` on the grid in one pass: entry `(i, j)` of the `r × n` result is `zx[j·r + i]`.
    let zt = grid_map(r * n, &|c| zx[(c % n) * r + c / n])?;
    // The full square in one product: `syrk_q32` tiles `SYRK_COLS = 16` columns per kernel call.
    let g = prepare(&zt, r, n, LIMBS)?.gemm(&zt, r)?;
    let d: Vec<f64> = (0..r)
        .map(|i| from_grid(g[i * r + i]).sqrt().max(EPS))
        .collect();
    let hn = |i: usize, j: usize| from_grid(g[i * r + j]) / (d[i] * d[j]);
    let mean_diag = (0..r).map(|i| hn(i, i)).sum::<f64>() / r as f64;
    let stab = (rho * mean_diag.abs() + reg).max(EPS);
    // The reference's symmetrisation `(H + Hᵀ)/2` off the diagonal (IEEE addition commutes, so
    // both triangles take the same word), then the stabiliser on the diagonal.
    let hw = grid_map(r * r, &|c| {
        let (i, j) = (c / r, c % r);
        if i == j {
            hn(i, i) + stab
        } else {
            0.5 * (hn(i, j) + hn(j, i))
        }
    })?;
    let hi = spd_inverse(&hw, r)?;
    // Rᵀ = Y_x·Z_x·D⁻¹ + ρ(Z − U), `k × r`.
    let wz = wp.gemm(&zt, r)?;
    let rt = grid_map(k * r, &|c| {
        from_grid(wz[c]) / d[c % r] + rho * (z[c] - du[c])
    })?;
    // Fᵀ = Rᵀ·H⁻¹: `gemm_q32(a, b) = a·bᵀ`, and `H⁻¹` is symmetric, so its rows serve.
    let f = gemm_q32(&rt, k, &hi, r, r, LIMBS)?;
    let mut out = vec![0.0f64; f.len()];
    par_fill(&mut out, &|c, o| *o = from_grid(f[c]));
    Ok(out)
}

/// `factorize_admm_nanoquant` on `w` (`out × inn`, Q32.32 row-major) under the input and output
/// statistics `i_norm` (length `inn`) and `o_norm` (length `out`), exported to the deployed
/// two-scale sign factor. A wide matrix (`out < inn`) runs on `wᵀ` with the statistics swapped
/// and is transposed back, as the reference's `is_transpose`.
///
/// # Errors
///
/// [`NqError::Shape`] on a mismatched length or a zero dimension or rank; [`NqError::Range`] on a
/// value off the Q32.32 grid; [`NqError::Kernel`] and [`NqError::Solve`] as the products and the
/// inverse factor refuse.
///
/// # Complexity
///
/// Per outer iteration (frame `o ≥ n`, rank `r`): two [`x_update`]s, `O(r²(o + n) + 2·r·o·n +
/// 4r³/3 + r²(o + n))` multiply-adds, and two [`rank1`]s, `O(inner·r·(o + n))`. Total
/// `T` times that; space `O(o·n + r·(o + n) + r²)` words.
pub fn factorize(
    w: &[i64],
    out: usize,
    inn: usize,
    i_norm: &[f64],
    o_norm: &[f64],
    cfg: &NqConfig,
) -> Result<NqFactor, NqError> {
    if out == 0 || inn == 0 || cfg.rank == 0 {
        return Err(NqError::Shape);
    }
    if w.len() != out * inn || i_norm.len() != inn || o_norm.len() != out {
        return Err(NqError::Shape);
    }
    if out < inn {
        let wt = transpose(w, out, inn);
        let f = factorize(&wt, inn, out, o_norm, i_norm, cfg)?;
        return Ok(NqFactor {
            out,
            inn,
            rank: f.rank,
            u: transpose(&f.v, f.rank, out),
            v: transpose(&f.u, inn, f.rank),
            s_pre: f.s_post,
            s_post: f.s_pre,
            admm_loss: f.admm_loss,
            u_latent: transpose(&f.v_latent, f.rank, out),
            v_latent: transpose(&f.u_latent, inn, f.rank),
        });
    }
    let (o, n, r) = (out, inn, cfg.rank);
    let ni: Vec<f64> = i_norm.iter().map(|x| x.sqrt().max(EPS)).collect();
    let no: Vec<f64> = o_norm.iter().map(|x| x.sqrt().max(EPS)).collect();
    let mut wn = vec![0.0f64; o * n];
    for i in 0..o {
        for j in 0..n {
            wn[i * n + j] = from_grid(w[i * n + j]) * ni[j] * no[i];
        }
    }
    let wp = prepare(&grid(&wn)?, o, n, LIMBS)?;
    let wtp = prepare(&grid(&transpose(&wn, o, n))?, n, o, LIMBS)?;
    let mut rng = Rng(cfg.seed);
    // A is `o × r`; B is held as Bᵀ, `n × r`.
    let mut a_ls: Vec<f64> = (0..o * r).map(|_| rng.normal()).collect();
    let b_ls_r: Vec<f64> = (0..r * n).map(|_| rng.normal()).collect();
    let mut b_ls = transpose(&b_ls_r, r, n);
    let (mut a_z, mut b_z) = if cfg.outer_iters > 0 {
        (
            rank1(&a_ls, o, r, cfg.inner_iters, false, &mut rng)?,
            rank1(&b_ls, n, r, cfg.inner_iters, true, &mut rng)?,
        )
    } else {
        (a_ls.clone(), b_ls.clone())
    };
    let mut a_u: Vec<f64> = a_ls.iter().zip(&a_z).map(|(x, z)| x - z).collect();
    let mut b_u: Vec<f64> = b_ls.iter().zip(&b_z).map(|(x, z)| x - z).collect();
    for t in 0..cfg.outer_iters {
        let x = t as f64 / cfg.outer_iters as f64;
        let rho = cfg.schedule.rho(x);
        a_ls = x_update(&b_z, n, r, &wp, o, &a_z, &a_u, rho, cfg.reg)?;
        b_ls = x_update(&a_z, o, r, &wtp, n, &b_z, &b_u, rho, cfg.reg)?;
        let ta: Vec<f64> = a_ls.iter().zip(&a_u).map(|(x, u)| x + u).collect();
        let tb: Vec<f64> = b_ls.iter().zip(&b_u).map(|(x, u)| x + u).collect();
        a_z = rank1(&ta, o, r, cfg.inner_iters, false, &mut rng)?;
        b_z = rank1(&tb, n, r, cfg.inner_iters, true, &mut rng)?;
        for (u, (l, z)) in a_u.iter_mut().zip(a_ls.iter().zip(&a_z)) {
            *u += l - z;
        }
        for (u, (l, z)) in b_u.iter_mut().zip(b_ls.iter().zip(&b_z)) {
            *u += l - z;
        }
    }
    let admm_loss = admm_loss(&wn, &a_z, &b_z, o, n, r)?;
    let a_lat: Vec<f64> = a_ls.iter().zip(&a_u).map(|(x, u)| x + u).collect();
    let b_lat: Vec<f64> = b_ls.iter().zip(&b_u).map(|(x, u)| x + u).collect();
    export(&[&a_z, &b_z, &a_lat, &b_lat], &ni, &no, o, n, r, admm_loss)
}

/// `‖W_n − (A_z/m)·B_z‖²/‖W_n‖²`, `m_k = ‖B_z row k‖` (the reference's printed loss).
/// `O(o·n·r)` on the kernel plus `O(o·n)`.
fn admm_loss(
    wn: &[f64],
    a_z: &[f64],
    b_z: &[f64],
    o: usize,
    n: usize,
    r: usize,
) -> Result<f64, NqError> {
    let bzr = transpose(b_z, n, r);
    let mid = rowdot_q32(&grid(&bzr)?, &grid(&bzr)?, r, n, LIMBS)?;
    let mut am = a_z.to_vec();
    for i in 0..o {
        for k in 0..r {
            am[i * r + k] /= from_grid(mid[k]).sqrt().max(EPS);
        }
    }
    let pred = gemm_q32(&grid(&am)?, o, &grid(b_z)?, n, r, LIMBS)?;
    let e: Vec<f64> = wn
        .iter()
        .zip(&pred)
        .map(|(w, &p)| w - from_grid(p))
        .collect();
    Ok(fro2(&e, o, n)? / fro2(wn, o, n)?.max(EPS))
}

/// The reference's final export: divide out the statistics, balance `A` and `B` to equal
/// Frobenius norm, scale `A`'s columns by `1/‖A_z col‖`, take `s_pre`/`s_post` as the mean
/// magnitudes and `U`/`V` as the signs (zero to `+1`). The latents `A_ls + A_u`, `B_ls + B_u`
/// take the same statistics and balance, without the column scale. `z = [A_z, B_zᵀ, A_lat,
/// B_latᵀ]`. `O(r·(o + n))` plus two `rowdot_q32`.
fn export(
    z: &[&[f64]; 4],
    ni: &[f64],
    no: &[f64],
    o: usize,
    n: usize,
    r: usize,
    admm_loss: f64,
) -> Result<NqFactor, NqError> {
    let [a_z, b_z, a_lat, b_lat] = *z;
    let mut a: Vec<f64> = a_z.to_vec();
    for i in 0..o {
        for k in 0..r {
            a[i * r + k] /= no[i];
        }
    }
    let mut b: Vec<f64> = b_z.to_vec();
    for j in 0..n {
        for k in 0..r {
            b[j * r + k] /= ni[j];
        }
    }
    let na = fro2(&a, o, r)?.sqrt().max(EPS);
    let nb = fro2(&b, n, r)?.sqrt().max(EPS);
    let bf = (nb / na).sqrt();
    let azt = transpose(a_z, o, r);
    let col = rowdot_q32(&grid(&azt)?, &grid(&azt)?, r, o, LIMBS)?;
    for i in 0..o {
        for k in 0..r {
            a[i * r + k] *= bf / from_grid(col[k]).sqrt().max(EPS);
        }
    }
    for x in &mut b {
        *x /= bf;
    }
    let mut ul = a_lat.to_vec();
    for i in 0..o {
        for k in 0..r {
            ul[i * r + k] *= bf / no[i];
        }
    }
    let mut vl = b_lat.to_vec();
    for j in 0..n {
        for k in 0..r {
            vl[j * r + k] /= ni[j] * bf;
        }
    }
    let sign = |x: f64| -> i8 {
        if x < 0.0 {
            -1
        } else {
            1
        }
    };
    let s_post: Vec<f64> = (0..o)
        .map(|i| a[i * r..(i + 1) * r].iter().map(|x| x.abs()).sum::<f64>() / r as f64)
        .collect();
    let s_pre: Vec<f64> = (0..n)
        .map(|j| b[j * r..(j + 1) * r].iter().map(|x| x.abs()).sum::<f64>() / r as f64)
        .collect();
    Ok(NqFactor {
        out: o,
        inn: n,
        rank: r,
        u: a.iter().map(|&x| sign(x)).collect(),
        v: transpose(&b.iter().map(|&x| sign(x)).collect::<Vec<i8>>(), n, r),
        s_pre,
        s_post,
        admm_loss,
        u_latent: ul,
        v_latent: transpose(&vl, n, r),
    })
}

/// Settings of [`tune`]: the reference `tune_fact` optimiser (`compress_block.py:160-201`,
/// `main.py:90-94`): `optimi.AdamW` with `β = (0.9, 0.99)`, `ε = 1e-6`, weight decay 0, cosine
/// annealing over `steps` to `1e-4·lr_scale`.
#[derive(Debug, Clone, Copy)]
pub struct TuneConfig {
    /// Optimiser steps (reference: 128 samples × 8 epochs at batch 1 = 1024).
    pub steps: usize,
    /// Learning rate of the latents (reference `fact_binary_lr` `1e-5`).
    pub lr_binary: f64,
    /// Learning rate of `s_pre`, `s_post` (reference `fact_scale_lr` `1e-5`).
    pub lr_scale: f64,
    /// Multiplier of the per-token Gram: the reference loss sums one 2048-token sample, so `2048`
    /// puts the gradient on the reference's scale against `ε`.
    pub h_scale: f64,
}

impl TuneConfig {
    /// The reference settings at `steps` steps.
    #[must_use]
    pub fn reference(steps: usize) -> Self {
        Self {
            steps,
            lr_binary: 1e-5,
            lr_scale: 1e-5,
            h_scale: 2048.0,
        }
    }
}

/// The record of one [`tune`].
#[derive(Debug, Clone)]
pub struct TuneTrace {
    /// `tr(EHEᵀ)/tr(WHWᵀ)` of the deployed factor before the first step and after the last.
    pub rel_before: f64,
    /// As `rel_before`, after the last step's signs and scales.
    pub rel_after: f64,
    /// The lowest `rel` seen at any step, and its step.
    pub rel_best: (f64, usize),
    /// Entries of `U` and `V` whose sign changed.
    pub flips: usize,
}

/// One `optimi` Adam step on `p` with gradient `g` (moments `m`, `v`, 1-based step `t`):
/// `β̂ = (βᵗ − β)/(βᵗ − 1)`, `m ← lerp(m, g, 1 − β̂₁)`, `v ← β̂₂v + (1 − β̂₂)g²`,
/// `p ← p − lr·m/(√v + ε)`. `O(len)`.
fn adam(p: &mut [f64], g: &[f64], m: &mut [f64], v: &mut [f64], t: usize, lr: f64) {
    const B1: f64 = 0.9;
    const B2: f64 = 0.99;
    const E: f64 = 1e-6;
    let debias = |b: f64| (b.powf(t as f64) - b) / (b.powf(t as f64) - 1.0);
    let (b1, b2) = (debias(B1), debias(B2));
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        p[i] -= lr * m[i] / (v[i].sqrt() + E);
    }
}

/// `±1` words of the signs of `x` (zero to `+1`).
fn sign_words(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&e| if e < 0.0 { -(1i64 << 32) } else { 1i64 << 32 })
        .collect()
}

/// The four gradients of [`tune`]'s loss from `(EH)ᵀ = pt` (`n × o`), `M` and the signs:
/// `[∂U (o × r), ∂Vᵀ (n × r), ∂s_post, ∂s_pre]`. Two `o·n·r` kernel products and two
/// `rowdot_q32`s of `o·n`, plus `O(o·n)` element maps and transposes.
fn grads(
    pt: &[f64],
    m: &[f64],
    us: &[i64],
    vst: &[i64],
    sp: &[f64],
    sq: &[f64],
    (o, n, r): (usize, usize, usize),
) -> Result<[Vec<f64>; 4], NqError> {
    // `Gᵀ = 2·H·Eᵀ` (`n × o`) and `G`.
    let gt: Vec<f64> = pt.iter().map(|x| 2.0 * x).collect();
    let mut gsq = transpose(&gt, n, o);
    for i in 0..o {
        for j in 0..n {
            gsq[i * n + j] *= sq[j];
        }
    }
    let mut gsp = gt.clone();
    for j in 0..n {
        for i in 0..o {
            gsp[j * o + i] *= sp[i];
        }
    }
    // ∂U = diag(sp)·(G·diag(sq))·Vᵀ: `a·bᵀ` with `b = V` (`r × n`).
    let vs = transpose(vst, n, r);
    let mut ga: Vec<f64> = gemm_q32(&grid(&gsq)?, o, &vs, r, n, LIMBS)?
        .iter()
        .map(|&x| from_grid(x))
        .collect();
    for i in 0..o {
        for k in 0..r {
            ga[i * r + k] *= sp[i];
        }
    }
    // ∂Vᵀ = diag(sq)·(Gᵀ·diag(sp))·U: `a·bᵀ` with `b = Uᵀ` (`r × o`).
    let ust = transpose(us, o, r);
    let mut gb: Vec<f64> = gemm_q32(&grid(&gsp)?, n, &ust, r, o, LIMBS)?
        .iter()
        .map(|&x| from_grid(x))
        .collect();
    for j in 0..n {
        for k in 0..r {
            gb[j * r + k] *= sq[j];
        }
    }
    // ∂sp = rows of `G·diag(sq)` dotted with `M`; ∂sq = rows of `Gᵀ·diag(sp)` with `Mᵀ`.
    let grad_post: Vec<f64> = rowdot_q32(&grid(&gsq)?, &grid(m)?, o, n, LIMBS)?
        .iter()
        .map(|&x| from_grid(x))
        .collect();
    let grad_pre: Vec<f64> = rowdot_q32(&grid(&gsp)?, &grid(&transpose(m, o, n))?, n, o, LIMBS)?
        .iter()
        .map(|&x| from_grid(x))
        .collect();
    Ok([ga, gb, grad_post, grad_pre])
}

/// The layer-scope analogue of the reference's `tune_fact`: Adam on the latents (through the
/// straight-through sign, `binary_ste`) and the two scales against
/// `L = tr((Ŵ − W)·H·(Ŵ − W)ᵀ)`, the squared output error of the layer over the calibration
/// tokens whose Gram is `h` (`in × in`, Q32.32, per token), then the signs of the latents.
///
/// With `G = 2(Ŵ − W)H`, `M = U·V`: `∂L/∂U = diag(s_post)·G·diag(s_pre)·Vᵀ`,
/// `∂L/∂V = Uᵀ·diag(s_post)·G·diag(s_pre)`, `∂L/∂s_post[i] = Σ_j G_ij·M_ij·s_pre[j]`,
/// `∂L/∂s_pre[j] = Σ_i G_ij·M_ij·s_post[i]`.
///
/// # Errors
///
/// [`NqError::Shape`] on a mismatched `w`, `h` or latent; [`NqError::Range`] and
/// [`NqError::Kernel`] as the products refuse.
///
/// # Complexity
///
/// Per step `o·n² + 3·o·n·r` multiply-adds on the kernel (`EH`, `UV` and the two latent
/// gradients) plus `O(o·n)` element maps and two `rowdot_q32`s; one more `EH` before the first
/// step, one after the last, and one `WH`. Space `O(o·n + n²)` words.
pub fn tune(
    f: &mut NqFactor,
    w: &[i64],
    h: &[i64],
    cfg: &TuneConfig,
) -> Result<TuneTrace, NqError> {
    let (o, n, r) = (f.out, f.inn, f.rank);
    if w.len() != o * n
        || h.len() != n * n
        || f.u_latent.len() != o * r
        || f.v_latent.len() != r * n
    {
        return Err(NqError::Shape);
    }
    let hs: Vec<f64> = h.iter().map(|&x| from_grid(x) * cfg.h_scale).collect();
    let hp = prepare(&grid(&hs)?, n, n, LIMBS)?;
    let wf: Vec<f64> = w.iter().map(|&x| from_grid(x)).collect();
    // `tr(XHXᵀ)` and `(XH)ᵀ` for an `o × n` matrix `X`: `H·Xᵀ` on the prepared plane, then the
    // trace as row dots of `Xᵀ` against it.
    let energy = |x: &[f64]| -> Result<(f64, Vec<f64>), NqError> {
        let pt = hp.gemm(&grid(x)?, o)?;
        let d = rowdot_q32(&grid(&transpose(x, o, n))?, &pt, n, o, LIMBS)?;
        Ok((
            d.iter().map(|&e| from_grid(e)).sum(),
            pt.iter().map(|&e| from_grid(e)).collect(),
        ))
    };
    let base = energy(&wf)?.0.max(EPS);
    let mut ul = f.u_latent.clone();
    let mut vlt = transpose(&f.v_latent, r, n);
    let (mut sp, mut sq) = (f.s_post.clone(), f.s_pre.clone());
    let (u0, v0) = (sign_words(&ul), sign_words(&vlt));
    let mut mom = [
        vec![0.0; o * r],
        vec![0.0; o * r],
        vec![0.0; n * r],
        vec![0.0; n * r],
        vec![0.0; o],
        vec![0.0; o],
        vec![0.0; n],
        vec![0.0; n],
    ];
    // `Ŵ = diag(sp)·M·diag(sq)`, `M = sign(ul)·sign(vlt)ᵀ` exact on the kernel.
    let forward = |ul: &[f64], vlt: &[f64], sp: &[f64], sq: &[f64]| {
        let (us, vst) = (sign_words(ul), sign_words(vlt));
        let m: Vec<f64> = gemm_q32(&us, o, &vst, n, r, LIMBS)?
            .iter()
            .map(|&e| from_grid(e))
            .collect();
        let mut e = vec![0.0f64; o * n];
        for i in 0..o {
            for j in 0..n {
                e[i * n + j] = sp[i] * m[i * n + j] * sq[j] - wf[i * n + j];
            }
        }
        Ok::<_, NqError>((us, vst, m, e))
    };
    let (_, _, _, e0) = forward(&ul, &vlt, &sp, &sq)?;
    let rel_before = energy(&e0)?.0 / base;
    let mut rel_best = (rel_before, 0);
    let eta_min = 1e-4 * cfg.lr_scale;
    for t in 0..cfg.steps {
        let (us, vst, m, e) = forward(&ul, &vlt, &sp, &sq)?;
        let (en, pt) = energy(&e)?;
        if en / base < rel_best.0 {
            rel_best = (en / base, t);
        }
        let [ga, gb, grad_post, grad_pre] = grads(&pt, &m, &us, &vst, &sp, &sq, (o, n, r))?;
        let cos = f64::midpoint(
            1.0,
            (std::f64::consts::PI * t as f64 / cfg.steps as f64).cos(),
        );
        let lr_b = eta_min + (cfg.lr_binary - eta_min) * cos;
        let lr_s = eta_min + (cfg.lr_scale - eta_min) * cos;
        let [m0, v0m, m1, v1m, m2, v2m, m3, v3m] = &mut mom;
        adam(&mut ul, &ga, m0, v0m, t + 1, lr_b);
        adam(&mut vlt, &gb, m1, v1m, t + 1, lr_b);
        adam(&mut sp, &grad_post, m2, v2m, t + 1, lr_s);
        adam(&mut sq, &grad_pre, m3, v3m, t + 1, lr_s);
    }
    let (us, vst, _, e) = forward(&ul, &vlt, &sp, &sq)?;
    let rel_after = energy(&e)?.0 / base;
    if rel_after < rel_best.0 {
        rel_best = (rel_after, cfg.steps);
    }
    let flips = us.iter().zip(&u0).filter(|(a, b)| a != b).count()
        + vst.iter().zip(&v0).filter(|(a, b)| a != b).count();
    let sgn = |x: i64| -> i8 {
        if x < 0 {
            -1
        } else {
            1
        }
    };
    f.u = us.iter().map(|&x| sgn(x)).collect();
    f.v = transpose(&vst.iter().map(|&x| sgn(x)).collect::<Vec<i8>>(), n, r);
    f.u_latent = ul;
    f.v_latent = transpose(&vlt, n, r);
    f.s_post = sp;
    f.s_pre = sq;
    Ok(TuneTrace {
        rel_before,
        rel_after,
        rel_best,
        flips,
    })
}

/// A rank-`s` sketch of the calibration Gram: `Q` (`n × s`, orthonormal columns) spanning `H`'s
/// dominant subspace and `H_s = QᵀHQ` (`s × s`), so `QH_sQᵀ = PHP` with `P = QQᵀ`.
#[derive(Debug, Clone)]
pub struct HSketch {
    /// The sketch width `s`.
    pub s: usize,
    /// `Q`, row-major `n × s`.
    pub q: Vec<f64>,
    /// `H_s = QᵀHQ`, row-major `s × s`.
    pub hs: Vec<f64>,
}

/// The columns of `y` (`n × s`) orthonormalised by Cholesky QR: unit columns, `G = YᵀY` by
/// `syrk_q32`, `G⁻¹ = UᵀU` by [`inv_factor_upper`], `Q = Y·Uᵀ` (`QᵀQ = U·G·Uᵀ = I`). Three passes,
/// the first on `G + 2⁻¹²·I` (shifted `CholeskyQR3`, Fukaya et al. 2020): the 3-limb `H·Q`
/// carries its tail columns to about `10⁻⁴` relative, and at `s = 512` that leaves `G` singular
/// unshifted. `O(3·(n·s² + s³))` multiply-adds.
fn orth(y: &[f64], n: usize, s: usize) -> Result<Vec<f64>, NqError> {
    let mut q = y.to_vec();
    for pass in 0..3 {
        let mut yt = transpose(&q, n, s);
        let norms = rowdot_q32(&grid(&yt)?, &grid(&yt)?, s, n, LIMBS)?;
        for k in 0..s {
            let d = from_grid(norms[k]).sqrt().max(EPS);
            for e in &mut yt[k * n..(k + 1) * n] {
                *e /= d;
            }
        }
        let mut g = syrk_q32(&grid(&yt)?, s, n, LIMBS)?;
        if pass == 0 {
            for k in 0..s {
                g[k * s + k] += 1 << 20;
            }
        }
        let ui = inv_factor_upper(&g, s)?;
        q = gemm_q32(&grid(&transpose(&yt, s, n))?, n, &ui, s, s, LIMBS)?
            .iter()
            .map(|&w| from_grid(w))
            .collect();
    }
    Ok(q)
}

/// [`HSketch`] of `h` (`n × n`, Q32.32) by subspace iteration: a Gaussian `n × s` start, then
/// `passes` steps `Q ← orth(H·Q)` on one prepared plane of `H`, then `H_s = Qᵀ(HQ)`.
///
/// # Errors
///
/// [`NqError::Shape`] if `h.len() != n·n` or `s` is zero or above `n`; [`NqError::Range`],
/// [`NqError::Kernel`] and [`NqError::Solve`] as the products and the factor refuse.
///
/// # Complexity
///
/// `(passes + 1)·n²·s` kernel multiply-adds for the `H·Q` products, `(passes + 1)·3·(n·s² + s³)`
/// for [`orth`], `n·s²` for `H_s`. Space `O(n² + n·s)` words (the prepared plane of `H`).
pub fn h_sketch(
    h: &[i64],
    n: usize,
    s: usize,
    passes: usize,
    seed: u64,
) -> Result<HSketch, NqError> {
    if h.len() != n * n || s == 0 || s > n {
        return Err(NqError::Shape);
    }
    let hp = prepare(h, n, n, LIMBS)?;
    let mut rng = Rng(seed);
    let omega: Vec<f64> = (0..n * s).map(|_| rng.normal()).collect();
    let mut q = orth(&omega, n, s)?;
    // `hp.gemm(Qᵀ, s) = H·Q` (`n × s`).
    let hq = |q: &[f64]| -> Result<Vec<f64>, NqError> {
        Ok(hp
            .gemm(&grid(&transpose(q, n, s))?, s)?
            .iter()
            .map(|&w| from_grid(w))
            .collect())
    };
    for _ in 0..passes {
        q = orth(&hq(&q)?, n, s)?;
    }
    let y = hq(&q)?;
    let hs = gemm_q32(
        &grid(&transpose(&q, n, s))?,
        s,
        &grid(&transpose(&y, n, s))?,
        s,
        n,
        LIMBS,
    )?
    .iter()
    .map(|&w| from_grid(w))
    .collect();
    Ok(HSketch { s, q, hs })
}

impl HSketch {
    /// `tr(X·QH_sQᵀ·Xᵀ)` of a row-major `o × n` matrix `x`: `XQ` (`o·n·s`), `(XQ)·H_s` (`o·s²`) and
    /// the row dots.
    ///
    /// # Errors
    ///
    /// [`NqError::Shape`] on a mismatched `x`; [`NqError::Range`] and [`NqError::Kernel`] as the
    /// products refuse.
    pub fn energy(&self, x: &[f64], o: usize) -> Result<f64, NqError> {
        let s = self.s;
        let n = self.q.len() / s;
        if x.len() != o * n {
            return Err(NqError::Shape);
        }
        let xq = gemm_q32(&grid(x)?, o, &grid(&transpose(&self.q, n, s))?, s, n, LIMBS)?;
        let xqh = gemm_q32(&xq, o, &grid(&self.hs)?, s, s, LIMBS)?;
        let d = rowdot_q32(&xq, &xqh, o, s, LIMBS)?;
        Ok(d.iter().map(|&e| from_grid(e)).sum())
    }
}

/// Words of `x` (row-major `rows × cols`) with column `j` scaled by `c[j]`. `O(rows·cols)`.
fn scale_cols(x: &[f64], rows: usize, cols: usize, c: &[f64]) -> Result<Vec<i64>, NqError> {
    let mut y = x.to_vec();
    for i in 0..rows {
        for j in 0..cols {
            y[i * cols + j] *= c[j];
        }
    }
    grid(&y)
}

/// Words of `x` (row-major `rows × cols`) with row `i` scaled by `c[i]`. `O(rows·cols)`.
fn scale_rows(x: &[f64], rows: usize, cols: usize, c: &[f64]) -> Result<Vec<i64>, NqError> {
    let mut y = x.to_vec();
    for i in 0..rows {
        for e in &mut y[i * cols..(i + 1) * cols] {
            *e *= c[i];
        }
    }
    grid(&y)
}

/// Kernel words to reals.
fn reals(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&w| from_grid(w)).collect()
}

/// The fixed `s`-frame of [`tune_sketch`]: `Qᵀ`, `Q`, `C = h_scale·H_s` as kernel words and `W·Q`.
struct SkFrame {
    dims: (usize, usize, usize, usize),
    qt: Vec<i64>,
    qw: Vec<i64>,
    c: Vec<i64>,
    wq: Vec<f64>,
}

/// One forward of [`tune_sketch`]: the signs, `P`, `U·P`, `E_Q·C` and the sketched loss `L_s`.
struct SkStep {
    us: Vec<i64>,
    vsf: Vec<f64>,
    p: Vec<i64>,
    up: Vec<f64>,
    eqc: Vec<i64>,
    ls: f64,
}

impl SkFrame {
    /// `P = V·diag(sq)·Q` (`r × s`), `UP = U·P` (`o × s`), `E_Q = diag(sp)·UP − W·Q`,
    /// `E_Q·C` and `L_s = tr(E_Q·C·E_Qᵀ)`. `r·s·(n + o) + o·s²` kernel multiply-adds.
    fn step(&self, ul: &[f64], vl: &[f64], sp: &[f64], sq: &[f64]) -> Result<SkStep, NqError> {
        let (o, n, r, s) = self.dims;
        let (us, vs) = (sign_words(ul), sign_words(vl));
        let vsf = reals(&vs);
        let p = gemm_q32(&scale_cols(&vsf, r, n, sq)?, r, &self.qt, s, n, LIMBS)?;
        let up = reals(&gemm_q32(
            &us,
            o,
            &grid(&transpose(&reals(&p), r, s))?,
            s,
            r,
            LIMBS,
        )?);
        let mut eq = vec![0.0f64; o * s];
        for i in 0..o {
            for k in 0..s {
                eq[i * s + k] = sp[i] * up[i * s + k] - self.wq[i * s + k];
            }
        }
        let eqw = grid(&eq)?;
        let eqc = gemm_q32(&eqw, o, &self.c, s, s, LIMBS)?;
        let ls = reals(&rowdot_q32(&eqw, &eqc, o, s, LIMBS)?).iter().sum();
        Ok(SkStep {
            us,
            vsf,
            p,
            up,
            eqc,
            ls,
        })
    }

    /// `[∂U, ∂V, ∂s_post, ∂s_pre]` at `st` (formulas at [`tune_sketch`]).
    /// `r·s·(2·o + 2·n)` kernel multiply-adds plus `O(s·(o + n))` row dots.
    fn grads(&self, st: &SkStep, sp: &[f64], sq: &[f64]) -> Result<[Vec<f64>; 4], NqError> {
        let (o, n, r, s) = self.dims;
        let g: Vec<f64> = reals(&st.eqc).iter().map(|x| 2.0 * x).collect();
        let gsp = scale_rows(&g, o, s, sp)?;
        let ga = reals(&gemm_q32(&gsp, o, &st.p, r, s, LIMBS)?);
        let a = gemm_q32(
            &grid(&transpose(&reals(&st.us), o, r))?,
            r,
            &grid(&transpose(&reals(&gsp), o, s))?,
            s,
            o,
            LIMBS,
        )?;
        let at = grid(&transpose(&reals(&a), r, s))?;
        // `∂V = A·Qᵀ·diag(sq)` (`r × n`), `VᵀA` (`n × s`).
        let gb = reals(&scale_cols(
            &reals(&gemm_q32(&a, r, &self.qw, n, s, LIMBS)?),
            r,
            n,
            sq,
        )?);
        let vta = gemm_q32(&grid(&transpose(&st.vsf, r, n))?, n, &at, s, r, LIMBS)?;
        let grad_post = reals(&rowdot_q32(&grid(&g)?, &grid(&st.up)?, o, s, LIMBS)?);
        let grad_pre = reals(&rowdot_q32(&self.qw, &vta, n, s, LIMBS)?);
        Ok([ga, gb, grad_post, grad_pre])
    }
}

/// [`tune`] on the sketched loss `L_s = tr(E·QH_sQᵀ·Eᵀ)` (`sk` from [`h_sketch`]): the same
/// optimiser, rates, schedule and STE, with every product taken in the `s`-dimensional frame.
/// With `P = V·diag(s_pre)·Q` (`r × s`), `E_Q = diag(s_post)·U·P − W·Q`, `G = 2·E_Q·C`
/// (`C = h_scale·H_s`): `∂U = diag(s_post)·G·Pᵀ`, `A = Uᵀ·diag(s_post)·G`,
/// `∂Vᵀ = diag(s_pre)·Q·Aᵀ`, `∂s_post[i] = ⟨G_i, (U·P)_i⟩`, `∂s_pre[j] = ⟨Q_j, (Vᵀ·A)_j⟩`.
/// `rel_before`/`rel_after` are the full `tr(EHEᵀ)/tr(WHWᵀ)` under `h`; `rel_best` tracks the
/// sketched ratio `L_s/tr(W·QH_sQᵀ·Wᵀ)`.
///
/// # Errors
///
/// As [`tune`]; [`NqError::Shape`] also when `sk` is not `n × s`.
///
/// # Complexity
///
/// Per step `r·s·(3·o + 3·n) + o·s²` kernel multiply-adds (`P`, `U·P`, `∂U`, `A`, `∂Vᵀ`, `Vᵀ·A`
/// and `G`) plus `O(r·(o + n))` element maps; against [`tune`]'s `o·n² + 3·o·n·r`. One `W·Q` and
/// the two full energies (`2·o·n² + o·n·r`) outside the loop. Space `O(r·(o + n) + n·s + n²)`.
pub fn tune_sketch(
    f: &mut NqFactor,
    w: &[i64],
    h: &[i64],
    sk: &HSketch,
    cfg: &TuneConfig,
) -> Result<TuneTrace, NqError> {
    let (o, n, r, s) = (f.out, f.inn, f.rank, sk.s);
    if w.len() != o * n
        || h.len() != n * n
        || sk.q.len() != n * s
        || sk.hs.len() != s * s
        || f.u_latent.len() != o * r
        || f.v_latent.len() != r * n
    {
        return Err(NqError::Shape);
    }
    let qt = grid(&transpose(&sk.q, n, s))?;
    let qw = grid(&sk.q)?;
    let c = grid(&sk.hs.iter().map(|x| x * cfg.h_scale).collect::<Vec<f64>>())?;
    let wq = reals(&gemm_q32(w, o, &qt, s, n, LIMBS)?);
    let wqc = gemm_q32(&grid(&wq)?, o, &c, s, s, LIMBS)?;
    let base_s: f64 = reals(&rowdot_q32(&grid(&wq)?, &wqc, o, s, LIMBS)?)
        .iter()
        .sum::<f64>()
        .max(EPS);
    let fr = SkFrame {
        dims: (o, n, r, s),
        qt,
        qw,
        c,
        wq,
    };
    let hp = prepare(h, n, n, LIMBS)?;
    let full = |x: &[i64]| -> Result<f64, NqError> {
        let pt = hp.gemm(x, o)?;
        let xt = grid(&transpose(&reals(x), o, n))?;
        Ok(reals(&rowdot_q32(&xt, &pt, n, o, LIMBS)?).iter().sum())
    };
    let base = full(w)?.max(EPS);
    let rel_full = |f: &NqFactor| -> Result<f64, NqError> {
        let e: Vec<i64> = f.weight()?.iter().zip(w).map(|(&a, &b)| a - b).collect();
        Ok(full(&e)? / base)
    };
    let rel_before = rel_full(f)?;
    let mut ul = f.u_latent.clone();
    let mut vl = f.v_latent.clone();
    let (mut sp, mut sq) = (f.s_post.clone(), f.s_pre.clone());
    let (u0, v0) = (sign_words(&ul), sign_words(&vl));
    let mut mom = [
        vec![0.0; o * r],
        vec![0.0; o * r],
        vec![0.0; r * n],
        vec![0.0; r * n],
        vec![0.0; o],
        vec![0.0; o],
        vec![0.0; n],
        vec![0.0; n],
    ];
    let mut rel_best = (f64::INFINITY, 0);
    let eta_min = 1e-4 * cfg.lr_scale;
    for t in 0..=cfg.steps {
        let st = fr.step(&ul, &vl, &sp, &sq)?;
        if st.ls / base_s < rel_best.0 {
            rel_best = (st.ls / base_s, t);
        }
        if t == cfg.steps {
            break;
        }
        let [ga, gb, grad_post, grad_pre] = fr.grads(&st, &sp, &sq)?;
        let cos = f64::midpoint(
            1.0,
            (std::f64::consts::PI * t as f64 / cfg.steps as f64).cos(),
        );
        let lr_b = eta_min + (cfg.lr_binary - eta_min) * cos;
        let lr_s = eta_min + (cfg.lr_scale - eta_min) * cos;
        let [m0, v0m, m1, v1m, m2, v2m, m3, v3m] = &mut mom;
        adam(&mut ul, &ga, m0, v0m, t + 1, lr_b);
        adam(&mut vl, &gb, m1, v1m, t + 1, lr_b);
        adam(&mut sp, &grad_post, m2, v2m, t + 1, lr_s);
        adam(&mut sq, &grad_pre, m3, v3m, t + 1, lr_s);
    }
    let (us, vs) = (sign_words(&ul), sign_words(&vl));
    let flips = us.iter().zip(&u0).filter(|(a, b)| a != b).count()
        + vs.iter().zip(&v0).filter(|(a, b)| a != b).count();
    let sgn = |x: i64| -> i8 {
        if x < 0 {
            -1
        } else {
            1
        }
    };
    f.u = us.iter().map(|&x| sgn(x)).collect();
    f.v = vs.iter().map(|&x| sgn(x)).collect();
    f.u_latent = ul;
    f.v_latent = vl;
    f.s_post = sp;
    f.s_pre = sq;
    let rel_after = rel_full(f)?;
    Ok(TuneTrace {
        rel_before,
        rel_after,
        rel_best,
        flips,
    })
}

impl NqFactor {
    /// The dense `Ŵ` (`out × in`, Q32.32): `U·V` on the kernel (exact: `±1` words), then the two
    /// diagonal scales. `O(out·in·r)` kernel products plus `O(out·in)`.
    ///
    /// # Errors
    ///
    /// [`NqError::Kernel`] if the product refuses; [`NqError::Range`] off the grid.
    pub fn weight(&self) -> Result<Vec<i64>, NqError> {
        let (o, n, r) = (self.out, self.inn, self.rank);
        let one = |s: i8| i64::from(s) << 32;
        let u: Vec<i64> = self.u.iter().map(|&s| one(s)).collect();
        let vt: Vec<i64> = transpose(&self.v, r, n).iter().map(|&s| one(s)).collect();
        let uv = gemm_q32(&u, o, &vt, n, r, LIMBS)?;
        let mut w = vec![0.0f64; o * n];
        for i in 0..o {
            for j in 0..n {
                w[i * n + j] = from_grid(uv[i * n + j]) * self.s_post[i] * self.s_pre[j];
            }
        }
        grid(&w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `H·spd_inverse(H) = I` within `1e-5` (`f64` oracle) on a well-conditioned `H = AAᵀ + I` of
    /// order 300, and no worse than the unrecursed base `UᵀU` on the same `H`.
    #[test]
    fn spd_inverse_inverts() {
        let n = 300usize;
        let mut rng = Rng(7);
        let scale = 1.0 / (n as f64).sqrt();
        let a = grid(&(0..n * n).map(|_| rng.normal() * scale).collect::<Vec<_>>()).unwrap();
        let aat = gemm_q32(&a, n, &a, n, n, LIMBS).unwrap();
        let h = grid_map(n * n, &|c| {
            from_grid(aat[c]) + if c / n == c % n { 1.0 } else { 0.0 }
        })
        .unwrap();
        // The oracle residual in `f64` (test-only scalar product, not on the build path).
        let hf: Vec<f64> = h.iter().map(|&w| from_grid(w)).collect();
        let mut res = Vec::new();
        for base in [n, INV_BASE] {
            let x = spd_inverse_at(&h, n, base).unwrap();
            let mut worst = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let d: f64 = (0..n)
                        .map(|t| hf[i * n + t] * from_grid(x[t * n + j]))
                        .sum();
                    worst = worst.max((d - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            println!("INV residual base {base} n {n} max |H·X − I| = {worst:e}");
            res.push(worst);
        }
        assert!(
            res[1] < 1e-5 && res[1] <= res[0],
            "max |H·X − I|: recursion {:e}, none {:e}",
            res[1],
            res[0]
        );
    }

    /// Wall of [`spd_inverse_at`] at the production order `r = 768` per base order (microbench;
    /// prints, asserts nothing). `APEIRON_NQ_INV_BASES` is a comma list, default `16,32,48,96,192`.
    #[test]
    #[ignore = "microbench"]
    fn spd_inverse_base_wall() {
        let n = 768usize;
        let mut rng = Rng(11);
        let scale = 1.0 / (n as f64).sqrt();
        let a = grid(&(0..n * n).map(|_| rng.normal() * scale).collect::<Vec<_>>()).unwrap();
        let aat = gemm_q32(&a, n, &a, n, n, LIMBS).unwrap();
        let h = grid_map(n * n, &|c| {
            from_grid(aat[c]) + if c / n == c % n { 1.0 } else { 0.0 }
        })
        .unwrap();
        let bases: Vec<usize> = std::env::var("APEIRON_NQ_INV_BASES")
            .unwrap_or_else(|_| String::from("16,32,48,96,192"))
            .split(',')
            .map(|s| s.parse().expect("base"))
            .collect();
        for &base in &bases {
            let mut ms = Vec::new();
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let x = spd_inverse_at(&h, n, base).unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(x);
            }
            let mean = ms.iter().sum::<f64>() / ms.len() as f64;
            let (lo, hi) = ms
                .iter()
                .fold((f64::MAX, 0.0f64), |(l, h), &m| (l.min(m), h.max(m)));
            println!("INV base {base} n {n} mean {mean:.2} ms min {lo:.2} max {hi:.2} runs 5");
        }
    }
}
