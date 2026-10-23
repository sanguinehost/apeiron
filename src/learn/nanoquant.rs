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
//! the Grams by `syrk_q32`, the right-hand sides and the solves by `gemm_q32`, the power iteration
//! by prepared one-column products and the norms by `rowdot_q32`. The `r × r` ridge system is
//! solved as `F = H⁻¹R` with `H⁻¹ = UᵀU` from [`crate::learn::solve::inv::inv_factor_upper`], so
//! the `O(r²·k)` substitution is two kernel products instead of a scalar triangular solve.
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

/// Every entry of `x` on the Q32.32 grid.
fn grid(x: &[f64]) -> Result<Vec<i64>, NqError> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).ok_or(NqError::Range))
        .collect()
}

/// Row-major `rows × cols` transposed to `cols × rows`. `O(rows·cols)` moves.
fn transpose<T: Copy + Default>(m: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut t = vec![T::default(); m.len()];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = m[i * cols + j];
        }
    }
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

/// One ridge `X`-update in the transposed frame: returns `Fᵀ` (`k × r`) with
/// `F = (XᵀX + (ρ·mean diag + reg)·I)⁻¹ (XᵀY + ρ(Z − U))`, `X = Z_x/‖·‖` column-normalised.
///
/// `zx` is the `n × r` normalising factor (`Bᵀ_z` or `A_z`), `wp` the prepared `k × n` target
/// (`W_n` for `A`, `W_nᵀ` for `B`), `z`/`du` the `k × r` consensus and dual.
///
/// `O(r²·n/2 + r·n·k + r³/2 + 2·r²·k)` multiply-adds: the Gram, the right-hand side, the inverse
/// factor and the two products `(RᵀUᵀ)·U`; all but the inverse factor on the kernel.
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
    let zt = grid(&transpose(zx, n, r))?;
    let g = syrk_q32(&zt, r, n, LIMBS)?;
    let d: Vec<f64> = (0..r)
        .map(|i| from_grid(g[i * r + i]).sqrt().max(EPS))
        .collect();
    let mut h = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            h[i * r + j] = from_grid(g[i * r + j]) / (d[i] * d[j]);
        }
    }
    // Symmetrise exactly as the reference, then the stabiliser.
    for i in 0..r {
        for j in (i + 1)..r {
            let s = 0.5 * (h[i * r + j] + h[j * r + i]);
            h[i * r + j] = s;
            h[j * r + i] = s;
        }
    }
    let mean_diag = (0..r).map(|i| h[i * r + i]).sum::<f64>() / r as f64;
    let stab = (rho * mean_diag.abs() + reg).max(EPS);
    for i in 0..r {
        h[i * r + i] += stab;
    }
    let ui = inv_factor_upper(&grid(&h)?, r)?;
    // Rᵀ = Y_x·Z_x·D⁻¹ + ρ(Z − U), `k × r`.
    let wz = wp.gemm(&zt, r)?;
    let mut rt = vec![0.0f64; k * r];
    for i in 0..k {
        for j in 0..r {
            let c = i * r + j;
            rt[c] = from_grid(wz[c]) / d[j] + rho * (z[c] - du[c]);
        }
    }
    // Fᵀ = Rᵀ·H⁻¹ = (Rᵀ·Uᵀ)·U: `gemm_q32(a, b) = a·bᵀ`, so the first takes `U`'s rows and the
    // second `Uᵀ`'s rows.
    let rtu = gemm_q32(&grid(&rt)?, k, &ui, r, r, LIMBS)?;
    let uit = transpose(&ui, r, r);
    let f = gemm_q32(&rtu, k, &uit, r, r, LIMBS)?;
    Ok(f.iter().map(|&w| from_grid(w)).collect())
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
/// Per outer iteration (frame `o ≥ n`, rank `r`): two [`x_update`]s, `O(r²(o + n)/2 + 2·r·o·n +
/// r³ + 2·r²(o + n))` multiply-adds, and two [`rank1`]s, `O(inner·r·(o + n))`. Total
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
