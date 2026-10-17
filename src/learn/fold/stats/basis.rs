//! Input-basis concentration for the 1.18 b/w fold: does a basis that concentrates the weight's
//! column energy beat the Hadamard basis that flattens it, once the coder may vary its rate per coefficient?
//!
//! # Model (grade `model`)
//!
//! For `W` (row-major `O × n`, Q32.32 dequantized to `f64`), a per-input orthogonal `Q` (block
//! diagonal in [`GROUP`] columns, like `fwht128`), the damped input Gram
//! `H + 0.01·mean(diag H)·I` (from [`super::capture::Grams`], [`super::capture::Normalisation::Mean`]) and `B = W·Q`:
//!
//! - `M = Qᵀ·H·Q`; the feedback weights `d_j` are the `LDLᵀ` pivots of `M` (`d_j = L_jj²` with
//!   `M = L·Lᵀ`), and `Π_j d_j = det H` for every orthogonal `Q`;
//! - the coefficient variances are [`Density::Separable`] `σ²_ij = r_i·c_j/m` (`r_i = Σ_j B_ij²`,
//!   `c_j = Σ_i B_ij²`, `m = Σ_ij B_ij²`; 16 side bits per row and column),
//!   [`Density::Tile`] the 16×16 tile mean of `B²` (8 side bits per tile), or
//!   [`Density::Oracle`] `σ²_ij = B_ij²` (no density side bits);
//! - reverse water-filling at `λ_ij = d_j·σ²_ij` has `D(θ) = Σ min(λ_ij, θ)` and
//!   `R(θ) = Σ max(0, ½·log₂(λ_ij/θ))`, with `θ` bisected in the log domain so that
//!   `R + side_bits = 1.18·O·n`; the fixed-rate baseline is `D_u = Σ λ_ij·2^{−2·(1.18 − side/(O·n))}`;
//! - the bases are the normalised 128-block Hadamard (`fwht128/√128`, [`hadamard_basis`]), the
//!   identity ([`identity_basis`]), the adapted butterfly ([`butterfly_basis`]: seven-stage radix-2
//!   butterfly, each node a Givens rotation by the Jacobi angle `θ = ½·atan2(2·G_ab, G_aa − G_bb)`
//!   of the current `G_c = BᵀB`, WHT stage order, then up to three further sweeps while `Σ log c_j`
//!   falls by more than `1e-6` relative; 8 angle bits per node, `7·64` nodes per block, shared
//!   across rows) and the per-block eigenbasis of `WᵀW` ([`eigen_basis`], basis side bits not
//!   counted: an upper reference). The output side is the identity in every arm; output rotations
//!   need residual-stream folding and are a stated non-goal.
//!
//! # Subsumption
//!
//! The empirical best for "rotation before low-bit quantisation" is a learned or folded rotation:
//! QuaRot (arXiv `2404.00456`), SpinQuant (arXiv `2405.16406`) and DuQuant (arXiv `2406.01721`).
//! `rg -l -i "spinquant|quarot|duquant" ~/Workspace` returns only citations (for example
//! `docs/epics/decode-dominance.md`, the L4 row) and no implementation, so this lane is
//! `not-subsumable` here: those methods FLATTEN the weight for a fixed-rate coder, while this lane
//! measures the opposite — CONCENTRATE the coefficient energy and let reverse water-filling spend
//! 3–4 bits on the occupied modes and 0 on the vacuum (Madelung reading: amplitude `R = |B_ij|`,
//! density `ρ = R²`; Hadamard sets `ρ` flat, the adapted basis keeps it concentrated).
//!
//! # Evidence grade
//!
//! `model`. The densities are a model of the coefficient magnitudes, the side-information counts
//! are the stated bit charges, and the distortion is the Gaussian rate-distortion model. No coder,
//! transfer or held-out KL is measured here; the synthetic tests are `tested`.
//!
//! # Trust boundary
//!
//! The activation Gram is the crate forward's Q32.32 tape projected to `f64` and averaged over
//! tokens; the weights are the engine's dequantized Q32.32 parameters. Not covered: the quantiser's
//! own accumulation, any output-side rotation, any real coder, and any quality effect of the modelled distortion.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the module's arithmetic is over f64 coefficient counts and bit budgets, so usize→f64 \
              casts are explicit and lossy by construction; the single-letter names are the \
              matrix symbols of the model (O, n, Q, H, B, M, d, c, r, λ, θ) and the doc names \
              upstream artefacts verbatim; the flat row-major loops are the reference layout and \
              are written explicitly so the complexity comment's multiply-add count is visible"
)]

use crate::learn::fold::fit::eigh::{eigh, EighError};
use crate::learn::fold::fit::steps::fwht128;
use crate::learn::ridge::{cholesky, RidgeError};

/// Side of one transform block. Every linear of the `Qwen2.5 0.5B` model has an input width that is
/// a multiple of this, matching `fit::steps::fwht128`'s own block width.
pub const GROUP: usize = 128;

/// Diagonal damping of the input Gram: `H + DAMP·mean(diag H)·I`.
pub const DAMP: f64 = 0.01;

/// Side of one coefficient-variance tile in [`Density::Tile`].
pub const TILE: usize = 16;

/// Reverse water-filling iterations and the rate-convergence threshold.
const BISECTION_ITERS: usize = 200;

/// The rate-convergence threshold per coefficient, `|ΔR| < RATE_TOL·N`.
const RATE_TOL: f64 = 1e-9;

/// A refused basis construction or distortion evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BasisError {
    /// A slice length or dimension disagrees with the declared geometry.
    Shape(String),
    /// A length product left `usize`.
    Overflow,
    /// A Cholesky pivot of `M` was not positive, so the damped metric is not positive definite.
    NotPositiveDefinite,
    /// The symmetric eigendecomposition did not converge.
    Eigh(String),
}

impl core::fmt::Display for BasisError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BasisError::Shape(m) => write!(f, "shape: {m}"),
            BasisError::Overflow => write!(f, "length product left usize"),
            BasisError::NotPositiveDefinite => write!(f, "metric is not positive definite"),
            BasisError::Eigh(m) => write!(f, "eigendecomposition: {m}"),
        }
    }
}

impl std::error::Error for BasisError {}

fn square(rows: usize, cols: usize) -> Result<usize, BasisError> {
    rows.checked_mul(cols).ok_or(BasisError::Overflow)
}

fn expect(len: usize, rows: usize, cols: usize, what: &str) -> Result<usize, BasisError> {
    match square(rows, cols)? {
        n if n == len => Ok(n),
        n => Err(BasisError::Shape(format!(
            "{what}: {len} != {rows}·{cols} = {n}"
        ))),
    }
}

fn expect_blocks(n: usize) -> Result<usize, BasisError> {
    if n == 0 || !n.is_multiple_of(GROUP) {
        return Err(BasisError::Shape(format!(
            "n = {n} is not a positive multiple of {GROUP}"
        )));
    }
    Ok(n / GROUP)
}

/// The damped input Gram `H + [`DAMP`]·mean(diag H)·I`, a fresh row-major `n × n` matrix.
///
/// The mean diagonal must be finite and positive: a zero Gram has no positive-definite damping and
/// is refused rather than silently left singular.
///
/// # Errors
///
/// [`BasisError::Shape`] when `h.len() != n·n`, `n == 0`, or the mean diagonal is not finite and
/// positive; [`BasisError::Overflow`] when `n·n` leaves `usize`.
///
/// # Complexity
/// Time `O(n²)` (one diagonal scan and one full copy); space `O(n²)` f64 words for the copy.
pub fn damped_gram(h: &[f64], n: usize) -> Result<Vec<f64>, BasisError> {
    expect(h.len(), n, n, "H")?;
    let mut mean = 0.0f64;
    for i in 0..n {
        mean += h[i * n + i];
    }
    mean /= n as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return Err(BasisError::Shape(format!(
            "mean(diag H) = {mean} is not finite and positive"
        )));
    }
    let delta = DAMP * mean;
    let mut out = h.to_vec();
    for i in 0..n {
        out[i * n + i] += delta;
    }
    Ok(out)
}

/// The `n × n` identity basis, row-major.
///
/// # Errors
///
/// [`BasisError::Overflow`] when `n·n` leaves `usize`.
///
/// # Complexity
/// Time and space `O(n²)`.
pub fn identity_basis(n: usize) -> Result<Vec<f64>, BasisError> {
    let n2 = square(n, n)?;
    let mut q = vec![0.0f64; n2];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    Ok(q)
}

/// The normalised 128-block Hadamard basis `H₁₂₈/√128`, block diagonal, row-major `n × n`.
///
/// Column `j` of the returned matrix is `fwht128(e_j)` for the block-local unit vector, so the
/// transform body is [`fwht128`], not a second implementation.
///
/// # Errors
///
/// [`BasisError::Shape`] when `n` is not a positive multiple of [`GROUP`] or a transform length
/// disagrees; [`BasisError::Overflow`] when `n·n` leaves `usize`.
///
/// # Complexity
/// Time `O(n·GROUP)` (one 128-point transform per column, `7·128` butterflies each); space `O(n²)`
/// for the basis plus `O(GROUP)` scratch.
pub fn hadamard_basis(n: usize) -> Result<Vec<f64>, BasisError> {
    let blocks = expect_blocks(n)?;
    let mut q = vec![0.0f64; square(n, n)?];
    let mut e = vec![0.0f64; GROUP];
    for b in 0..blocks {
        let off = b * GROUP;
        for j in 0..GROUP {
            e.fill(0.0);
            e[j] = 1.0;
            let col = fwht128(&e).map_err(|err| BasisError::Shape(format!("fwht128: {err:?}")))?;
            for i in 0..GROUP {
                q[(off + i) * n + off + j] = col[i];
            }
        }
    }
    Ok(q)
}

/// Every entry of `q` outside a [`GROUP`]×[`GROUP`] diagonal block is zero.
///
/// # Errors
///
/// [`BasisError::Shape`] on a non-block-diagonal basis; [`BasisError::Overflow`] when `n·n` leaves `usize`.
///
/// # Complexity
/// Time and space `O(n²)`.
fn expect_block_diagonal(q: &[f64], n: usize) -> Result<(), BasisError> {
    expect(q.len(), n, n, "Q")?;
    for i in 0..n {
        let bi = i / GROUP;
        for j in 0..n {
            if j / GROUP != bi && q[i * n + j] != 0.0 {
                return Err(BasisError::Shape(format!(
                    "Q[{i},{j}] = {} lies outside a {GROUP}-block",
                    q[i * n + j]
                )));
            }
        }
    }
    Ok(())
}

/// The coefficients `B = W·Q`, row-major `O × n`, for a block-diagonal `Q`.
///
/// # Errors
///
/// [`BasisError::Shape`] when `w.len() != O·n`, `Q` is not `n × n` block diagonal, or `n` is not a
/// positive multiple of [`GROUP`]; [`BasisError::Overflow`] on a length product.
///
/// # Complexity
/// Time `O(O·n·GROUP)`: per block, one `O × 128` by `128 × 128` product, `n/GROUP` blocks. Space
/// `O(O·n)` f64 words for the coefficients.
pub fn apply_basis(w: &[f64], o: usize, n: usize, q: &[f64]) -> Result<Vec<f64>, BasisError> {
    expect(w.len(), o, n, "W")?;
    let blocks = expect_blocks(n)?;
    expect_block_diagonal(q, n)?;
    let mut b = vec![0.0f64; square(o, n)?];
    for blk in 0..blocks {
        let off = blk * GROUP;
        for i in 0..o {
            for k in 0..GROUP {
                let a = w[i * n + off + k];
                if a == 0.0 {
                    continue;
                }
                for c in 0..GROUP {
                    b[i * n + off + c] += a * q[(off + k) * n + off + c];
                }
            }
        }
    }
    Ok(b)
}

/// The metric `M = Qᵀ·H·Q`, row-major `n × n`, for a block-diagonal `Q`.
///
/// `H` is the (already damped) input Gram. Only the upper block triangle is formed and the lower one
/// is its transpose, since `H` is symmetric.
///
/// # Errors
///
/// [`BasisError::Shape`] when `h.len() != n·n`, `Q` is not `n × n` block diagonal, or `n` is not a
/// positive multiple of [`GROUP`]; [`BasisError::Overflow`] on a length product.
///
/// # Complexity
/// Time `O(n²·GROUP)`: `n²/GROUP²` upper block pairs, each `128³` multiply-adds twice. Space `O(n²)`
/// for `M` plus `O(GROUP²)` scratch per block pair.
pub fn block_metric(q: &[f64], h: &[f64], n: usize) -> Result<Vec<f64>, BasisError> {
    expect(h.len(), n, n, "H")?;
    let blocks = expect_blocks(n)?;
    expect_block_diagonal(q, n)?;
    let mut m = vec![0.0f64; square(n, n)?];
    let mut t = vec![0.0f64; GROUP * GROUP];
    for b1 in 0..blocks {
        for b2 in b1..blocks {
            // t = H[b1,b2] · Q[b2], row-major 128 × 128.
            t.fill(0.0);
            for i in 0..GROUP {
                for k in 0..GROUP {
                    let a = h[(b1 * GROUP + i) * n + b2 * GROUP + k];
                    if a == 0.0 {
                        continue;
                    }
                    for c in 0..GROUP {
                        t[i * GROUP + c] += a * q[(b2 * GROUP + k) * n + b2 * GROUP + c];
                    }
                }
            }
            // m[b1,b2] = Q[b1]ᵀ · t.
            for c1 in 0..GROUP {
                for i in 0..GROUP {
                    let a = q[(b1 * GROUP + i) * n + b1 * GROUP + c1];
                    if a == 0.0 {
                        continue;
                    }
                    for c2 in 0..GROUP {
                        let v = a * t[i * GROUP + c2];
                        let at = (b1 * GROUP + c1) * n + b2 * GROUP + c2;
                        m[at] += v;
                        if b1 != b2 {
                            m[(b2 * GROUP + c2) * n + b1 * GROUP + c1] += v;
                        }
                    }
                }
            }
        }
    }
    Ok(m)
}

/// The successive-coding feedback weights `d_j = L_jj²` of the Cholesky factor of `m`.
///
/// # Errors
///
/// [`BasisError::Shape`] when `m.len() != n·n`; [`BasisError::NotPositiveDefinite`] when a pivot is non-positive or NaN.
///
/// # Complexity
/// Time `O(n³/6)` multiply-adds (the Cholesky factor's lower triangle); space `O(n²)` for `L`.
pub fn cholesky_pivots(m: &[f64], n: usize) -> Result<Vec<f64>, BasisError> {
    expect(m.len(), n, n, "M")?;
    let l = cholesky(m, n).map_err(|err| match err {
        RidgeError::NotPositiveDefinite => BasisError::NotPositiveDefinite,
        other => BasisError::Shape(format!("cholesky: {other:?}")),
    })?;
    let mut d = vec![0.0f64; n];
    for j in 0..n {
        let lj = l[j * n + j];
        d[j] = lj * lj;
    }
    Ok(d)
}

/// Row energies `r_i = Σ_j B_ij²`, length `O`.
///
/// # Errors
///
/// [`BasisError::Shape`] when `b.len() != O·n`.
///
/// # Complexity
/// Time and space `O(O·n)` for the scan and the output.
pub fn row_energies(b: &[f64], o: usize, n: usize) -> Result<Vec<f64>, BasisError> {
    expect(b.len(), o, n, "B")?;
    let mut r = vec![0.0f64; o];
    for i in 0..o {
        for j in 0..n {
            let v = b[i * n + j];
            r[i] += v * v;
        }
    }
    Ok(r)
}

/// Column energies `c_j = Σ_i B_ij²`, length `n`.
///
/// # Errors
///
/// [`BasisError::Shape`] when `b.len() != O·n`.
///
/// # Complexity
/// Time `O(O·n)`; space `O(n)`.
pub fn column_energies(b: &[f64], o: usize, n: usize) -> Result<Vec<f64>, BasisError> {
    expect(b.len(), o, n, "B")?;
    let mut c = vec![0.0f64; n];
    for i in 0..o {
        for j in 0..n {
            let v = b[i * n + j];
            c[j] += v * v;
        }
    }
    Ok(c)
}

/// `Σ_j log c_j` over the positive column energies; a zero energy is floored at `f64::MIN_POSITIVE`
/// so the sum stays finite and deterministic.
///
/// # Complexity
/// Time `O(n)`; space `O(1)`.
pub fn sum_log_column_energies(c: &[f64]) -> f64 {
    c.iter().map(|&v| v.max(f64::MIN_POSITIVE).ln()).sum()
}

/// The 128×128 Gram `W_blockᵀ·W_block` of one column block, row-major `128 × 128`, symmetric.
///
/// # Complexity
/// Time `O(O·GROUP²/2)` from the upper-triangle rank-one updates; space `O(GROUP²)`.
fn block_gram(w: &[f64], o: usize, n: usize, blk: usize) -> Vec<f64> {
    let off = blk * GROUP;
    let mut g = vec![0.0f64; GROUP * GROUP];
    for i in 0..GROUP {
        for r in 0..o {
            let a = w[r * n + off + i];
            if a == 0.0 {
                continue;
            }
            for j in i..GROUP {
                g[i * GROUP + j] += a * w[r * n + off + j];
            }
        }
    }
    for i in 0..GROUP {
        for j in i..GROUP {
            g[j * GROUP + i] = g[i * GROUP + j];
        }
    }
    g
}

/// Apply `Jᵀ·G·J` to the 128×128 Gram `g` for the Givens rotation `J = [[c, −s], [s, c]]` on columns `(a, b)`.
///
/// # Complexity
/// Time `O(GROUP)`; space `O(1)`.
fn rotate_gram(g: &mut [f64], a: usize, b: usize, c: f64, s: f64) {
    let (i, j) = (GROUP, GROUP);
    for r in 0..i {
        if r == a || r == b {
            continue;
        }
        let gia = g[r * j + a];
        let gib = g[r * j + b];
        let nia = c * gia + s * gib;
        let nib = -s * gia + c * gib;
        g[r * j + a] = nia;
        g[a * j + r] = nia;
        g[r * j + b] = nib;
        g[b * j + r] = nib;
    }
    let gaa = g[a * j + a];
    let gbb = g[b * j + b];
    let gab = g[a * j + b];
    g[a * j + a] = c * c * gaa + 2.0 * c * s * gab + s * s * gbb;
    g[b * j + b] = s * s * gaa - 2.0 * c * s * gab + c * c * gbb;
    let off = -c * s * gaa + (c * c - s * s) * gab + c * s * gbb;
    g[a * j + b] = off;
    g[b * j + a] = off;
}

/// Right-multiply the whole `n × n` basis `q` by the same Givens rotation on columns `(ca, cb)`.
///
/// # Complexity
/// Time `O(n)`; space `O(1)`.
fn rotate_columns(q: &mut [f64], n: usize, ca: usize, cb: usize, c: f64, s: f64) {
    for r in 0..n {
        let qa = q[r * n + ca];
        let qb = q[r * n + cb];
        q[r * n + ca] = c * qa + s * qb;
        q[r * n + cb] = -s * qa + c * qb;
    }
}

/// Which angle a butterfly node uses.
#[derive(Clone, Copy, Debug, PartialEq)]
enum AngleRule {
    /// The Jacobi angle that diagonalises the node's current 2×2 Gram block.
    Jacobi,
    /// A fixed angle; `π/4` reproduces `fwht128/√128` up to column signs.
    Fixed(f64),
}

/// One pass over the seven-stage butterfly graph of a block: `7·64` nodes in WHT order.
///
/// # Complexity
/// Time `O(7·64·n)` (each node rotates `GROUP` Gram entries and `n` basis rows); space `O(1)`.
fn sweep(q: &mut [f64], g: &mut [f64], n: usize, off: usize, rule: AngleRule) {
    let mut h = 1usize;
    while h < GROUP {
        let mut i = 0usize;
        while i < GROUP {
            for j in i..i + h {
                let (a, b) = (j, j + h);
                let theta = match rule {
                    AngleRule::Jacobi => {
                        let gab = g[a * GROUP + b];
                        let gaa = g[a * GROUP + a];
                        let gbb = g[b * GROUP + b];
                        0.5 * (2.0 * gab).atan2(gaa - gbb)
                    }
                    AngleRule::Fixed(t) => t,
                };
                let (s, c) = theta.sin_cos();
                rotate_gram(g, a, b, c, s);
                rotate_columns(q, n, off + a, off + b, c, s);
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// `Σ log(diag g)` with the [`sum_log_column_energies`] floor.
///
/// # Complexity
/// Time `O(GROUP)`; space `O(1)`.
fn sum_log_diag(g: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..GROUP {
        s += g[i * GROUP + i].max(f64::MIN_POSITIVE).ln();
    }
    s
}

/// The adapted butterfly basis: each node of the WHT butterfly is a Givens rotation by the Jacobi
/// angle of the current column Gram `G_c = BᵀB`, one mandatory pass plus up to three further sweeps
/// while `Σ log c_j` falls by more than `1e-6` relative.
///
/// # Errors
///
/// [`BasisError::Shape`] when `w.len() != O·n` or `n` is not a positive multiple of [`GROUP`];
/// [`BasisError::Overflow`] on a length product.
///
/// # Complexity
/// Time `O(n²·GROUP + n·GROUP·nodes·sweeps)`: block Grams `O(O·n·GROUP)`, then at most four passes
/// of `7·64` nodes, each rotating `GROUP` Gram entries and `n` basis rows. Space `O(n²)` for `Q`
/// plus `O(GROUP² + O·n)` scratch.
pub fn butterfly_basis(w: &[f64], o: usize, n: usize) -> Result<Vec<f64>, BasisError> {
    butterfly_build(w, o, n, AngleRule::Jacobi)
}

/// The butterfly basis with every node fixed at `angle` radians. With `angle = π/4` the result is
/// `fwht128/√128` up to column signs; the public seam exists so the test can assert that identity.
///
/// # Errors
///
/// As [`butterfly_basis`].
///
/// # Complexity
/// Time `O(O·n·GROUP + n·GROUP²)` (one pass, no extra sweeps); space `O(n²)`.
pub fn butterfly_basis_fixed(
    w: &[f64],
    o: usize,
    n: usize,
    angle: f64,
) -> Result<Vec<f64>, BasisError> {
    butterfly_build(w, o, n, AngleRule::Fixed(angle))
}

/// Shared body of [`butterfly_basis`] and [`butterfly_basis_fixed`].
///
/// # Complexity
/// See [`butterfly_basis`]; the adaptive rule performs at most three extra sweeps.
fn butterfly_build(w: &[f64], o: usize, n: usize, rule: AngleRule) -> Result<Vec<f64>, BasisError> {
    expect(w.len(), o, n, "W")?;
    let blocks = expect_blocks(n)?;
    let mut q = identity_basis(n)?;
    for blk in 0..blocks {
        let mut g = block_gram(w, o, n, blk);
        sweep(&mut q, &mut g, n, blk * GROUP, rule);
        if rule == AngleRule::Jacobi {
            let mut prev = sum_log_diag(&g);
            for _ in 0..3 {
                sweep(&mut q, &mut g, n, blk * GROUP, rule);
                let cur = sum_log_diag(&g);
                if prev - cur <= 1e-6 * prev.abs() {
                    break;
                }
                prev = cur;
            }
        }
    }
    Ok(q)
}

/// The angle side information of the adapted butterfly: 8 bits per node, `7·64` nodes per block,
/// shared across the block's rows. `28·n` bits for an `n`-column tensor.
///
/// # Complexity
/// Time and space `O(1)`.
pub fn butterfly_side_bits(n: usize) -> f64 {
    8.0 * (7.0 * (GROUP as f64) / 2.0) * (n / GROUP) as f64
}

/// The dense per-block eigenbasis of the 128×128 diagonal blocks of `WᵀW`, using
/// [`eigh`](crate::learn::fold::fit::eigh::eigh). The column order is ascending eigenvalue, as
/// `eigh` returns it. Basis side bits are **not** counted anywhere: this arm is an upper reference.
///
/// # Errors
///
/// [`BasisError::Shape`] when `w.len() != O·n` or `n` is not a positive multiple of [`GROUP`];
/// [`BasisError::Eigh`] when a block does not converge; [`BasisError::Overflow`] on a length product.
///
/// # Complexity
/// Time `O(O·n·GROUP + (n/GROUP)·GROUP³·sweeps)` — one Gram per block and one bounded cyclic Jacobi
/// decomposition of each 128×128 block; space `O(n²)` for the basis.
pub fn eigen_basis(w: &[f64], o: usize, n: usize) -> Result<Vec<f64>, BasisError> {
    expect(w.len(), o, n, "W")?;
    let blocks = expect_blocks(n)?;
    let mut q = vec![0.0f64; square(n, n)?];
    for blk in 0..blocks {
        let g = block_gram(w, o, n, blk);
        let (_, vecs) = eigh(&g, GROUP).map_err(|err| match err {
            EighError::NotConverged => BasisError::Eigh("block did not converge".to_string()),
            other => BasisError::Shape(format!("eigh: {other:?}")),
        })?;
        let off = blk * GROUP;
        for i in 0..GROUP {
            for j in 0..GROUP {
                q[(off + i) * n + off + j] = vecs[i * GROUP + j];
            }
        }
    }
    Ok(q)
}

/// One coefficient-variance density model and its measured side-information charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Density {
    /// Separable `σ²_ij = r_i·c_j/m`, 16 side bits per row and per column.
    Separable,
    /// Mean of `B²` over the 16×16 tile containing `(i,j)`, 8 side bits per tile.
    Tile,
    /// `σ²_ij = B_ij²`, no density side information: the attainable reference.
    Oracle,
}

impl Density {
    /// The model's own side-information charge in bits, counting rows/columns or tiles for the
    /// whole `O × n` coefficient matrix.
    ///
    /// # Complexity
    /// Time and space `O(1)`.
    pub fn side_bits(self, o: usize, n: usize) -> f64 {
        match self {
            Density::Separable => 16.0 * (o + n) as f64,
            Density::Tile => 8.0 * o.div_ceil(TILE) as f64 * n.div_ceil(TILE) as f64,
            Density::Oracle => 0.0,
        }
    }

    /// The `O × n` coefficient-variance matrix of `b = W·Q`.
    ///
    /// # Errors
    ///
    /// [`BasisError::Shape`] when `b.len() != O·n`; [`BasisError::Overflow`] on a length product.
    ///
    /// # Complexity
    /// Time and space `O(O·n)` for every variant: one scan of `B`, one tile accumulate for
    /// [`Density::Tile`], one output matrix.
    pub fn sigma(self, b: &[f64], o: usize, n: usize) -> Result<Vec<f64>, BasisError> {
        let len = expect(b.len(), o, n, "B")?;
        match self {
            Density::Oracle => Ok(b.iter().map(|v| v * v).collect()),
            Density::Separable => {
                let r = row_energies(b, o, n)?;
                let c = column_energies(b, o, n)?;
                let m: f64 = r.iter().sum();
                let mut out = vec![0.0f64; len];
                if m <= 0.0 {
                    return Ok(out);
                }
                for i in 0..o {
                    for j in 0..n {
                        out[i * n + j] = r[i] * c[j] / m;
                    }
                }
                Ok(out)
            }
            Density::Tile => {
                let nti = o.div_ceil(TILE);
                let ntj = n.div_ceil(TILE);
                let ntiles = nti * ntj;
                let mut sums = vec![0.0f64; ntiles];
                let mut counts = vec![0usize; ntiles];
                for i in 0..o {
                    for j in 0..n {
                        let k = (i / TILE) * ntj + j / TILE;
                        let v = b[i * n + j];
                        sums[k] += v * v;
                        counts[k] += 1;
                    }
                }
                for (s, &c) in sums.iter_mut().zip(&counts) {
                    if c > 0 {
                        *s /= c as f64;
                    }
                }
                let mut out = vec![0.0f64; len];
                for i in 0..o {
                    for j in 0..n {
                        out[i * n + j] = sums[(i / TILE) * ntj + j / TILE];
                    }
                }
                Ok(out)
            }
        }
    }
}

/// A reverse water-filling solution at the target rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFill {
    /// The water level `θ`.
    pub theta: f64,
    /// `D(θ) = Σ min(λ_ij, θ)`.
    pub distortion: f64,
    /// `R(θ) = Σ max(0, ½·log₂(λ_ij/θ))`.
    pub rate: f64,
}

/// `D(θ) = Σ min(λ, θ)`; `θ ≤ 0` gives zero.
///
/// # Complexity
/// Time `O(N)`; space `O(1)`.
pub fn distortion_at(lambda: &[f64], theta: f64) -> f64 {
    if theta <= 0.0 {
        return 0.0;
    }
    lambda.iter().map(|&l| l.min(theta)).sum()
}

/// `R(θ) = Σ max(0, ½·log₂(λ/θ))`; `θ ≤ 0` gives an infinite rate when any `λ > 0`.
///
/// # Complexity
/// Time `O(N)`; space `O(1)`.
pub fn rate_at(lambda: &[f64], theta: f64) -> f64 {
    if theta <= 0.0 {
        return if lambda.iter().any(|&l| l > 0.0) {
            f64::INFINITY
        } else {
            0.0
        };
    }
    let t = theta.log2();
    let mut r = 0.0f64;
    for &l in lambda {
        if l > theta {
            r += 0.5 * (l.log2() - t);
        }
    }
    r
}

/// The reverse water-filling solution with `R(θ) + side_bits = budget_bits`: `θ` bisected in the log
/// domain for at most 200 iterations or until `|R − target| < 1e-9·N`.
///
/// # Complexity
/// Time `O(N·BISECTION_ITERS)` worst case (a rate evaluation is one `N` scan); space `O(1)` beyond
/// the caller's `lambda`. The bracket walk is bounded by [`BISECTION_ITERS`] doublings.
pub fn solve_water_fill(lambda: &[f64], budget_bits: f64, side_bits: f64) -> WaterFill {
    let ncoef = lambda.len() as f64;
    let target = budget_bits - side_bits;
    let max = lambda.iter().copied().fold(0.0f64, f64::max);
    if lambda.is_empty() || max <= 0.0 {
        return WaterFill {
            theta: 0.0,
            distortion: 0.0,
            rate: 0.0,
        };
    }
    if target <= 0.0 {
        return WaterFill {
            theta: max,
            distortion: lambda.iter().sum(),
            rate: 0.0,
        };
    }
    let mut t_hi = max.log2() + 1.0;
    let min_pos = lambda
        .iter()
        .copied()
        .filter(|&l| l > 0.0)
        .fold(f64::INFINITY, f64::min);
    let mut t_lo = min_pos.log2() - 1.0;
    let mut gap = 1.0f64;
    for _ in 0..BISECTION_ITERS {
        if rate_at(lambda, 2.0f64.powf(t_lo)) >= target {
            break;
        }
        t_lo -= gap;
        gap *= 2.0;
    }
    let mut t = t_hi;
    for _ in 0..BISECTION_ITERS {
        t = 0.5 * (t_lo + t_hi);
        let r = rate_at(lambda, 2.0f64.powf(t));
        if (r - target).abs() <= RATE_TOL * ncoef {
            break;
        }
        if r > target {
            t_lo = t;
        } else {
            t_hi = t;
        }
    }
    let theta = 2.0f64.powf(t);
    WaterFill {
        theta,
        distortion: distortion_at(lambda, theta),
        rate: rate_at(lambda, theta),
    }
}

/// The fixed-rate distortion `D_u = Σ λ·2^{−2·(budget/N − side/N)}`.
///
/// # Complexity
/// Time `O(N)`; space `O(1)`.
pub fn uniform_distortion(lambda: &[f64], budget_bits: f64, side_bits: f64) -> f64 {
    if lambda.is_empty() {
        return 0.0;
    }
    let bits = (budget_bits - side_bits) / lambda.len() as f64;
    let factor = 2.0f64.powf(-2.0 * bits);
    lambda.iter().map(|&l| l * factor).sum()
}

/// The basis-independent part of one arm: coefficients `B = W·Q` and feedback weights `d`.
#[derive(Clone, Debug, PartialEq)]
pub struct Coefficients {
    /// `O`.
    pub o: usize,
    /// `n`.
    pub n: usize,
    /// `B = W·Q`, row-major `O × n`.
    pub b: Vec<f64>,
    /// The Cholesky pivots `d_j` of `QᵀHQ`, length `n`.
    pub d: Vec<f64>,
}

/// One arm's distortion at the registered rate: uniform (`D_u`) and reverse water-filling (`D_wf`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arm {
    /// `D_u = Σ λ·2^{−2·(1.18 − side/(O·n))}`, the fixed-rate coder.
    pub uniform: f64,
    /// `D_wf`, the reverse water-filling distortion.
    pub water_filled: f64,
    /// The achieved `R + side`, in bits, for the water-filling solution.
    pub rate: f64,
}

impl Coefficients {
    /// Form `B = W·Q`, `M = Qᵀ·H·Q` and the pivots `d` of `M` for one basis.
    ///
    /// # Errors
    ///
    /// The [`apply_basis`], [`block_metric`] and [`cholesky_pivots`] refusals.
    ///
    /// # Complexity
    /// Dominated by the metric `O(n²·GROUP)` and the Cholesky `O(n³/6)`; space `O(O·n + n²)`.
    pub fn from_basis(
        w: &[f64],
        o: usize,
        n: usize,
        h: &[f64],
        q: &[f64],
    ) -> Result<Coefficients, BasisError> {
        let b = apply_basis(w, o, n, q)?;
        let m = block_metric(q, h, n)?;
        let d = cholesky_pivots(&m, n)?;
        Ok(Coefficients { o, n, b, d })
    }

    /// Evaluate one density model at `budget_bits` (the pipeline's `1.18·O·n`) with the density's
    /// side information plus `basis_side_bits` charged against it.
    ///
    /// # Errors
    ///
    /// [`BasisError::Shape`] when the stored geometry is inconsistent; [`BasisError::Overflow`] on
    /// a length product.
    ///
    /// # Complexity
    /// Time `O(O·n·BISECTION_ITERS)` for the solve plus `O(O·n)` for the density and the uniform
    /// distortion; space `O(O·n)` for `σ²` and `λ`.
    pub fn evaluate(
        &self,
        density: Density,
        basis_side_bits: f64,
        budget_bits: f64,
    ) -> Result<Arm, BasisError> {
        let sigma = density.sigma(&self.b, self.o, self.n)?;
        let mut lambda = vec![0.0f64; square(self.o, self.n)?];
        for i in 0..self.o {
            for j in 0..self.n {
                lambda[i * self.n + j] = self.d[j] * sigma[i * self.n + j];
            }
        }
        let side = density.side_bits(self.o, self.n) + basis_side_bits;
        let wf = solve_water_fill(&lambda, budget_bits, side);
        Ok(Arm {
            uniform: uniform_distortion(&lambda, budget_bits, side),
            water_filled: wf.distortion,
            rate: wf.rate,
        })
    }
}
