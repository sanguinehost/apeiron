//! Symmetric eigendecomposition by cyclic Jacobi rotations, and the exact rank-`rank` corrector in
//! the metric `Hl` — the `round_f16=False` path of `lowrank_corrector` in the removed one-bit fold
//! transfer (`fold_transfer.py:560–583`, read-only spec outside this repository).
//!
//! The Python computes `M = (ΔW · Hl) · ΔWᵀ`, takes the `rank` largest eigenpairs of the symmetrised
//! `½(M + Mᵀ)` by `torch.linalg.eigh`, and returns `Corr = U · (Uᵀ ΔW)` with `U` the top-`rank`
//! eigenvectors (columns) plus `top_sum = Σ_k max(λ_k, 0)` over the top `rank` eigenvalues. The
//! stored factors `U`, `V = Uᵀ ΔW` are plain `f64` so a later task can round them (the fp16 and u8
//! variants of the same function), which is why the exact path returns them rather than `Corr` alone.
//!
//! # What is established
//!
//! The Jacobi rotation below carries one eigenpair at a time and the recurrence closes on the
//! off-diagonal Frobenius norm reaching the tolerance; that arithmetic is `tested`
//! (`tests/learn_fold_eigh.rs`), not `proved`. The trace-optimality of `Corr` — that
//! `diag(ΔW − Corr) = (I − U Uᵀ) ΔW` minimises `tr(R · Hl · Rᵀ)` over rank-`rank` projections — is the
//! classical Ky Fan / Eckart–Young result; the test compares it against 20 random rank-`rank`
//! projections rather than reproving it.
//!
//! # Deviations from the Python
//!
//! - `torch.linalg.eigh` is replaced by cyclic Jacobi. The spec's matrix is symmetric by
//!   construction and small (`o ≤ a few thousand` rows), so the `O(n³)`-per-sweep, bounded-sweep
//!   Jacobi is a faithful and dependency-free stand-in; the eigenvalues are returned ascending, as
//!   `torch.linalg.eigh` returns them.
//! - The Python's `rank == 0` (`evecs[:, -0:]` selects *all* columns) is refused as `RankOutOfRange`
//!   only when `rank > o`; `rank == 0` returns the empty corrector (`Corr = 0`, empty `U`, `V`),
//!   which is the reading the callers mean. No caller passes `rank == 0`.
//! - `Hl` is not required symmetric: the eigendecomposition symmetrises `M` as `½(M + Mᵀ)`, exactly
//!   as the Python does.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the Jacobi sweep, the Gram product and the `U V` accumulation index row-major `[r, c]` \
              buffers by `o/n/i/j/k/l/p/q`, which reads against the spec's `DeltaW[i, j]` arithmetic, \
              and name the rotation's `aip/aiq/vip/viq` pairs; `apq == 0.0` is the exact \
              `torch` zero-skip test on a symmetrised entry; the tolerance is \
              `f64::EPSILON · n · ‖A‖_F`, a deliberate lossy `usize as f64` of a dimension. The \
              pedantic lints stay on for the rest of the crate."
)]

/// A refusal from the symmetric eigendecomposition or the exact low-rank corrector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EighError {
    /// A slice length disagrees with the declared shape (`n·n`, or `o·n` for `ΔW`).
    ShapeMismatch,
    /// The cyclic Jacobi sweep bound ([`MAX_SWEEPS`]) was exhausted before the off-diagonal norm
    /// fell to the convergence tolerance.
    NotConverged,
    /// The requested `rank` exceeds the row count `o` of `ΔW`; the corrector cannot exceed the
    /// ambient dimension.
    RankOutOfRange,
}

/// Cyclic Jacobi sweep bound. Jacobi's off-diagonal norm is reduced by at least a constant factor per
/// sweep and converges quadratically near the solution, so `~10` sweeps reach machine precision for
/// every tested `n`; `60` is a bounded ceiling that refuses a pathological input instead of looping.
pub const MAX_SWEEPS: usize = 60;

/// Symmetric eigendecomposition of `a` (row-major `[n, n]`) by cyclic Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` where the eigenvalues are ascending and `eigenvectors` is
/// row-major `[n, n]` with column `k` the eigenvector for `eigenvalues[k]` (`Σ_i Q[i,k]²  = 1`,
/// `Qᵀ Q = I`). `a` need not be symmetric on entry: `½(a + aᵀ)` is decomposed, matching the Python's
/// `eigh(0.5 * (M + M.T))`.
///
/// # Complexity
///
/// Time `O(n³ · sweeps)`: one sweep visits the `n(n−1)/2` pairs and each Jacobi rotation updates
/// `O(n)` matrix and eigenvector entries. Space `O(n²)` for the working copy, the eigenvector basis
/// and the returned buffers. The off-diagonal norm (`O(n²)`) is recomputed once per sweep, a term
/// dominated by the `O(n³)` sweep. Sweeps are bounded by [`MAX_SWEEPS`]; convergence is declared
/// when the off-diagonal Frobenius norm falls to `f64::EPSILON · n · ‖½(a + aᵀ)‖_F`.
///
/// # Errors
///
/// [`EighError::ShapeMismatch`] if `a.len() != n·n`; [`EighError::NotConverged`] if the
/// off-diagonal norm has not reached the tolerance within [`MAX_SWEEPS`] sweeps.
pub fn eigh(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), EighError> {
    if a.len() != n * n {
        return Err(EighError::ShapeMismatch);
    }
    // Symmetrise: `m = ½(a + aᵀ)`.
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    // `v` accumulates the rotations; it starts as the identity (`v[i, i] = 1`).
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let frob = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    let tol = f64::EPSILON * (n as f64) * frob;
    let mut converged = false;
    for _ in 0..MAX_SWEEPS {
        // Off-diagonal Frobenius norm of the current matrix: the convergence measure.
        let mut off_sq = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off_sq += m[i * n + j] * m[i * n + j];
                }
            }
        }
        if off_sq.sqrt() <= tol {
            converged = true;
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                jacobi_rotate(&mut m, &mut v, n, p, q);
            }
        }
    }
    if !converged {
        return Err(EighError::NotConverged);
    }
    let mut evals: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    sort_ascending(&mut evals, &mut v, n);
    Ok((evals, v))
}

/// One cyclic Jacobi rotation on the symmetric pair `(p, q)` of `m` (row-major `[n, n]`, full
/// symmetric storage), applying the same rotation to the columns `p`, `q` of the basis `v`. The
/// rotation is chosen to annihilate `m[p, q]`; `t = sign(θ)/(|θ| + √(θ²+1))`, `c = 1/√(t²+1)`,
/// `s = t·c`, with `θ = (m[q,q] − m[p,p]) / (2·m[p,q])` (Numerical Recipes §11.1). `O(n)` time.
fn jacobi_rotate(m: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize) {
    let apq = m[p * n + q];
    if apq == 0.0 {
        return;
    }
    let app = m[p * n + p];
    let aqq = m[q * n + q];
    // θ = (a_qq − a_pp) / (2 a_pq); the sign is taken so |t| ≤ 1.
    let theta = (aqq - app) / (2.0 * apq);
    let t = if theta >= 0.0 {
        1.0 / (theta + theta.hypot(1.0))
    } else {
        -1.0 / (-theta + theta.hypot(1.0))
    };
    let c = 1.0 / t.hypot(1.0);
    let s = t * c;
    // Diagonal and the annihilated entry.
    m[p * n + p] = app - t * apq;
    m[q * n + q] = aqq + t * apq;
    m[p * n + q] = 0.0;
    m[q * n + p] = 0.0;
    // Off-diagonal rows/columns: the rotation on `(i, p)`, `(i, q)` for `i ∉ {p, q}`, mirrored.
    for i in 0..n {
        if i == p || i == q {
            continue;
        }
        let aip = m[i * n + p];
        let aiq = m[i * n + q];
        let nip = c * aip - s * aiq;
        let niq = s * aip + c * aiq;
        m[i * n + p] = nip;
        m[p * n + i] = nip;
        m[i * n + q] = niq;
        m[q * n + i] = niq;
    }
    // Eigenvector columns `p`, `q`.
    for i in 0..n {
        let vip = v[i * n + p];
        let viq = v[i * n + q];
        v[i * n + p] = c * vip - s * viq;
        v[i * n + q] = s * vip + c * viq;
    }
}

/// Sort `evals` ascending, carrying the matching columns of `v` (row-major `[n, n]`) through the same
/// swaps. Insertion sort: deterministic, `O(n²)` time and `O(1)` extra space — the eigenvalue count
/// `n` is a matrix dimension, not a batch size.
fn sort_ascending(evals: &mut [f64], v: &mut [f64], n: usize) {
    for i in 1..evals.len() {
        let mut j = i;
        while j > 0 && evals[j - 1] > evals[j] {
            evals.swap(j - 1, j);
            for r in 0..n {
                v.swap(r * n + (j - 1), r * n + j);
            }
            j -= 1;
        }
    }
}

/// The exact rank-`rank` corrector in the metric `Hl`: the stored factors and the corrected dense
/// block `Corr = U·V`.
#[derive(Clone, Debug)]
pub struct Corrector {
    /// The dense correction `U·V`, row-major `[o, n]`.
    pub corr: Vec<f64>,
    /// `Σ_k max(λ_k, 0)` over the top `rank` eigenvalues of `M = (ΔW·Hl)·ΔWᵀ` — the first-order drop
    /// in the quadratic objective the corrector attains.
    pub top_sum: f64,
    /// The top `rank` eigenvectors of `M`, row-major `[o, rank]`, column `k` the eigenvector for the
    /// `(rank−k)`-th largest eigenvalue (ascending order, so column `rank−1` is the largest).
    pub u: Vec<f64>,
    /// `V = Uᵀ·ΔW`, row-major `[rank, n]`, so `Corr = U·V`.
    pub v: Vec<f64>,
}

/// The exact rank-`rank` corrector of `ΔW` in the metric `Hl` — the `round_f16=False, u8_corr=False`
/// branch of `lowrank_corrector`.
///
/// `ΔW` is row-major `[o, n]`, `Hl` row-major `[n, n]`. Returns `Corr = U·(Uᵀ ΔW)` where `U` holds
/// the `rank` largest eigenvectors (columns) of `M = (ΔW·Hl)·ΔWᵀ`, its stored factors `U`, `V`, and
/// `top_sum = Σ max(λ_k, 0)` over the top `rank` eigenvalues. `Corr = (I − U Uᵀ) ΔW` is the
/// rank-`rank` projection residual of `ΔW` on the dominant eigenspace of `M`, and it minimises
/// `tr(R·Hl·Rᵀ)` over rank-`rank` projections: the Ky Fan / Eckart–Young argument.
///
/// # Complexity
///
/// Time `O(o·n·(n + rank) + o²·n + o³·sweeps)` — the `ΔW·Hl` product `O(o·n²)`, the `M` product
/// `O(o²·n)`, the eigendecomposition `O(o³·sweeps)` ([`eigh`]), `V` `O(rank·o·n)` and `Corr`
/// `O(o·n·rank)`. Space `O(o·n + o² + o·rank + rank·n)` for the working product, the matrix, the
/// basis and the returned buffers.
///
/// # Errors
///
/// [`EighError::ShapeMismatch`] if `delta_w.len() != o·n` or `hl.len() != n·n`;
/// [`EighError::RankOutOfRange`] if `rank > o`; otherwise the [`EighError`] of the underlying
/// [`eigh`].
pub fn lowrank_corrector_exact(
    delta_w: &[f64],
    hl: &[f64],
    o: usize,
    n: usize,
    rank: usize,
) -> Result<Corrector, EighError> {
    if delta_w.len() != o * n || hl.len() != n * n {
        return Err(EighError::ShapeMismatch);
    }
    if rank > o {
        return Err(EighError::RankOutOfRange);
    }
    // `w1 = ΔW · Hl` (o × n).
    let mut w1 = vec![0.0f64; o * n];
    for i in 0..o {
        for k in 0..n {
            let mut acc = 0.0f64;
            for j in 0..n {
                acc += delta_w[i * n + j] * hl[j * n + k];
            }
            w1[i * n + k] = acc;
        }
    }
    // `m = w1 · ΔWᵀ` (o × o); `eigh` symmetrises it as the Python does.
    let mut m = vec![0.0f64; o * o];
    for i in 0..o {
        for l in 0..o {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += w1[i * n + k] * delta_w[l * n + k];
            }
            m[i * o + l] = acc;
        }
    }
    let (evals, evecs) = eigh(&m, o)?;
    // `U` = the `rank` largest eigenvectors: columns `o − rank .. o` of the ascending-order basis.
    let mut u = vec![0.0f64; o * rank];
    for i in 0..o {
        for k in 0..rank {
            u[i * rank + k] = evecs[i * o + (o - rank + k)];
        }
    }
    let mut top_sum = 0.0f64;
    for k in 0..rank {
        let lam = evals[o - rank + k];
        if lam > 0.0 {
            top_sum += lam;
        }
    }
    // `V = Uᵀ · ΔW` (rank × n).
    let mut v = vec![0.0f64; rank * n];
    for k in 0..rank {
        for j in 0..n {
            let mut acc = 0.0f64;
            for i in 0..o {
                acc += u[i * rank + k] * delta_w[i * n + j];
            }
            v[k * n + j] = acc;
        }
    }
    // `Corr = U · V` (o × n).
    let mut corr = vec![0.0f64; o * n];
    for i in 0..o {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..rank {
                acc += u[i * rank + k] * v[k * n + j];
            }
            corr[i * n + j] = acc;
        }
    }
    Ok(Corrector {
        corr,
        top_sum,
        u,
        v,
    })
}
