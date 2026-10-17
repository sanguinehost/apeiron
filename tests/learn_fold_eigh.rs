//! `tests/learn_fold_eigh.rs` — the Q32.32 symmetric eigendecomposition and the exact
//! rank-`rank` corrector in the `Hl` metric (the `round_f16=False` path of `lowrank_corrector`).
//!
//! Positive: agreement of `eigh` with the removed `f64` cyclic-Jacobi oracle — eigenvalues to the
//! derived residual-plus-grid bound and the sign-free eigenvector overlaps against a planted,
//! gap-separated spectrum; reconstruction `Q diag(L) Qᵀ = A` and orthonormality `QᵀQ = I` on random
//! SPD matrices, repeated-eigenvalue, zero and semidefinite diagonal spectra;
//! exact rank-`k` recovery when `rank ≥ k`; the trace-optimality of `Corr` against 20 random
//! rank-`rank` projections; `top_sum` and `Corr` against the oracle; `Corr = U·V`. Boundary: the
//! shape and rank refusals, and an indefinite input (both a negative diagonal and a zero-diagonal
//! matrix whose spectrum is negative) is refused, not shifted. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixed-seed LCG maps a 24-bit integer field to an f64 in [0, 1); the f64 Jacobi \
              oracle below is the removed reference body kept verbatim as the migration's \
              comparison oracle; the test kernels index row-major `[r, c]` buffers by `i/j/k/l` \
              against the spec's `DeltaW[i, j]` arithmetic. The pedantic lints stay on for the \
              crate's non-test code."
)]

use apeiron::learn::fold::fit::eigh::{eigh, lowrank_corrector_exact, EighError};
use apeiron::learn::solve::q32::{fill_scale, from_grid, to_grid, GRID_BOUND};

/// Deterministic 64-bit LCG (Numerical Recipes constants), the tests' only randomness source.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[0, 1)`, 53 significant bits.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
}

/// `x` as a Q32.32 word (the boundary the word APIs take).
fn q(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("grid")
}

/// `x` as words.
fn qv(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| q(v)).collect()
}

/// The `f64` view of a word buffer.
fn uw(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| from_grid(v)).collect()
}

/// `a · b` for row-major `[r, k] · [k, c]`.
fn matmul(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += a[i * k + t] * b[t * c + j];
            }
            out[i * c + j] = acc;
        }
    }
    out
}

fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = a[i * c + j];
        }
    }
    t
}

/// `Q · diag(d) · Qᵀ` for row-major `[n, n]` `q` whose columns are the basis.
fn q_diag_qt(q: &[f64], d: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += q[i * n + k] * d[k] * q[j * n + k];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// `max |(QᵀQ)[i, j] − δ_ij|`, the orthonormality defect of the columns of row-major `[n, n]` `q`.
fn orthonormality_defect(q: &[f64], n: usize) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += q[k * n + i] * q[k * n + j];
            }
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((acc - target).abs());
        }
    }
    worst
}

/// A random orthonormal `o × r` matrix (column-major Gram–Schmidt), row-major `[o, r]`.
fn random_orthonormal(rng: &mut Lcg, o: usize, r: usize) -> Vec<f64> {
    let mut p = vec![0.0f64; o * r];
    for c in 0..r {
        for i in 0..o {
            p[i * r + c] = rng.sym();
        }
        for c2 in 0..c {
            let mut dot = 0.0f64;
            for i in 0..o {
                dot += p[i * r + c2] * p[i * r + c];
            }
            for i in 0..o {
                p[i * r + c] -= dot * p[i * r + c2];
            }
        }
        let mut nrm = 0.0f64;
        for i in 0..o {
            nrm += p[i * r + c] * p[i * r + c];
        }
        let nrm = nrm.sqrt();
        for i in 0..o {
            p[i * r + c] /= nrm;
        }
    }
    p
}

/// A random SPD matrix `BᵀB + ½I`, row-major `[n, n]`.
fn random_spd(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let mut b = vec![0.0f64; n * n];
    for x in &mut b {
        *x = rng.sym();
    }
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += b[k * n + i] * b[k * n + j];
            }
            h[i * n + j] = acc + if i == j { 0.5 } else { 0.0 };
        }
    }
    h
}

/// `(I − P·Pᵀ)·ΔW` for row-major `[o, r]` orthonormal `p` and row-major `[o, n]` `dw`.
fn projection_residual(dw: &[f64], p: &[f64], o: usize, n: usize, r: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; r * n];
    for k in 0..r {
        for j in 0..n {
            let mut acc = 0.0f64;
            for i in 0..o {
                acc += p[i * r + k] * dw[i * n + j];
            }
            s[k * n + j] = acc;
        }
    }
    let mut res = dw.to_vec();
    for i in 0..o {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..r {
                acc += p[i * r + k] * s[k * n + j];
            }
            res[i * n + j] -= acc;
        }
    }
    res
}

/// `tr(R · Hl · Rᵀ)` for row-major `[o, n]` `r` and row-major `[n, n]` `hl`: `Σ_ij (R·Hl)[i,j]·R[i,j]`.
fn trace_r_hl_rt(r: &[f64], hl: &[f64], o: usize, n: usize) -> f64 {
    let rhl = matmul(r, hl, o, n, n);
    let mut t = 0.0f64;
    for i in 0..o {
        for j in 0..n {
            t += rhl[i * n + j] * r[i * n + j];
        }
    }
    t
}

// ---------------------------------------------------------------------------------------------
// The migration oracle: the removed `f64` cyclic Jacobi, verbatim, test-only.
// ---------------------------------------------------------------------------------------------

/// One cyclic Jacobi rotation on `(p, q)` of the symmetric `m` and the basis `v` (the removed body).
fn ref_rotate(m: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize) {
    let apq = m[p * n + q];
    if apq == 0.0 {
        return;
    }
    let app = m[p * n + p];
    let aqq = m[q * n + q];
    let theta = (aqq - app) / (2.0 * apq);
    let t = if theta >= 0.0 {
        1.0 / (theta + theta.hypot(1.0))
    } else {
        -1.0 / (-theta + theta.hypot(1.0))
    };
    let c = 1.0 / t.hypot(1.0);
    let s = t * c;
    m[p * n + p] = app - t * apq;
    m[q * n + q] = aqq + t * apq;
    m[p * n + q] = 0.0;
    m[q * n + p] = 0.0;
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
    for i in 0..n {
        let vip = v[i * n + p];
        let viq = v[i * n + q];
        v[i * n + p] = c * vip - s * viq;
        v[i * n + q] = s * vip + c * viq;
    }
}

/// Sort `evals` ascending, carrying the columns of `v`.
fn ref_sort(evals: &mut [f64], v: &mut [f64], n: usize) {
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

/// The removed `f64` eigensolver (cyclic Jacobi on `½(a + aᵀ)`), the migration's oracle.
fn ref_eigh(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let frob = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    let tol = f64::EPSILON * (n as f64) * frob;
    let mut converged = false;
    for _ in 0..60 {
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
                ref_rotate(&mut m, &mut v, n, p, q);
            }
        }
    }
    assert!(converged, "the f64 oracle must converge");
    let mut evals: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    ref_sort(&mut evals, &mut v, n);
    (evals, v)
}

/// The removed `f64` exact corrector: scalar triple-loop Gram, [`ref_eigh`], `f64` `U`, `V`, `Corr`.
fn ref_corrector(dw: &[f64], hl: &[f64], o: usize, n: usize, rank: usize) -> (Vec<f64>, f64) {
    let w1 = matmul(dw, hl, o, n, n);
    let mut m = matmul(&w1, &transpose(dw, o, n), o, n, o);
    for i in 0..o {
        for j in 0..i {
            let avg = 0.5 * (m[i * o + j] + m[j * o + i]);
            m[i * o + j] = avg;
            m[j * o + i] = avg;
        }
    }
    let (evals, evecs) = ref_eigh(&m, o);
    let mut u = vec![0.0f64; o * rank];
    for i in 0..o {
        for k in 0..rank {
            u[i * rank + k] = evecs[i * o + (o - rank + k)];
        }
    }
    let top_sum: f64 = evals[o - rank..].iter().map(|&l| l.max(0.0)).sum();
    let v = matmul(&transpose(&u, o, rank), dw, rank, o, n);
    (matmul(&u, &v, o, rank, n), top_sum)
}

/// The Q32.32 quantum, in real units, of the grid [`eigh`] uses for a matrix of maximum magnitude
/// `max_mag`: the boundary scale is the exact power of two `s` with `max_mag·s ≤ 8`, so one word is
/// `2⁻³²/s` in real units.
fn quantum(max_mag: f64) -> f64 {
    2f64.powi(-32) / (fill_scale(max_mag) * 2f64.powi(-28))
}

/// The derived eigenvalue bound of [`eigh`]: `top_eigh` stops at a column residual
/// `‖M v − λ v‖∞ ≤ λ_max·2⁻²⁰`, and for a symmetric matrix `|λ − λ̂| ≤ ‖r‖₂ ≤ √n·‖r‖∞`; the
/// symmetrised grid matrix perturbs every eigenvalue by at most its max row sum, `n` quanta.
fn eig_bound(n: usize, lam_max: f64, q: f64) -> f64 {
    (n as f64).sqrt() * lam_max * 2f64.powi(-20) + (n as f64) * q
}

#[test]
fn agrees_with_f64_oracle_on_planted_spectrum() {
    // Ascending, gap-separated spectra: `gap_min = 0.9` keeps the eigenvector overlaps a
    // per-vector statement (|cos| ≥ 1 − θ² with θ = eigenvalue bound / gap).
    let spec = [
        1.0f64, 1.9, 3.4, 5.8, 9.1, 13.7, 19.9, 28.1, 39.0, 53.0, 71.0, 92.0,
    ];
    for &n in &[4usize, 7, 12] {
        let d = &spec[..n];
        let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15 ^ (n as u64));
        let q0 = random_orthonormal(&mut rng, n, n);
        let a = q_diag_qt(&q0, d, n);
        let (l, q) = eigh(&a, n).expect("eigh must converge on a planted spectrum");
        let (lr, qr) = ref_eigh(&a, n);
        let lam_max = d[n - 1];
        let qq = quantum(max_abs(&a));
        let bound = eig_bound(n, lam_max, qq);
        let gap = d
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::INFINITY, f64::min);
        for k in 0..n {
            let err = (l[k] - lr[k]).abs();
            assert!(
                err <= bound,
                "n = {n} k = {k}: eigenvalue {} vs oracle {} err {err:e} over bound {bound:e}",
                l[k],
                lr[k]
            );
            let mut dot = 0.0f64;
            for i in 0..n {
                dot += qr[i * n + k] * q[i * n + k];
            }
            let theta = bound / gap;
            assert!(
                dot.abs() >= 1.0 - theta * theta,
                "n = {n} k = {k}: |<v_ref, v>| = {} below 1 − θ² = {}",
                dot.abs(),
                1.0 - theta * theta
            );
        }
    }
}

#[test]
fn agrees_with_f64_oracle_on_random_spd() {
    for &n in &[5usize, 10, 24] {
        let mut rng = Lcg::new(0xDEAD_BEEF_0BAD_F00D ^ (n as u64));
        let a = random_spd(&mut rng, n);
        let (l, _) = eigh(&a, n).expect("eigh must converge on a random SPD matrix");
        let (lr, _) = ref_eigh(&a, n);
        let bound = eig_bound(n, max_abs(&lr), quantum(max_abs(&a)));
        for k in 0..n {
            let err = (l[k] - lr[k]).abs();
            assert!(
                err <= bound,
                "n = {n} k = {k}: eigenvalue {} vs oracle {} err {err:e} over bound {bound:e}",
                l[k],
                lr[k]
            );
        }
    }
}

#[test]
fn reconstruction_and_orthonormality() {
    for &n in &[3usize, 8, 24] {
        let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15 ^ (n as u64));
        let a = random_spd(&mut rng, n);
        let (l, q) = eigh(&a, n).expect("eigh must converge on an SPD input");
        assert_eq!(l.len(), n);
        assert_eq!(q.len(), n * n);
        for i in 1..n {
            assert!(
                l[i - 1] <= l[i],
                "n = {n}: eigenvalues not ascending at {i}: {} > {}",
                l[i - 1],
                l[i]
            );
        }
        // The returned (l, q) satisfies `A q_k − l_k q_k = r_k` with `‖r_k‖∞ ≤ λ_max·2⁻²⁰`, and
        // `Q` is square orthonormal (`Q Qᵀ = I`), so `QΛQᵀ − A = −R Qᵀ`: each entry sums `n`
        // residuals, and the grid perturbation of the matrix adds `n` quanta through the same sum.
        let lam_max = l[n - 1].abs().max(l[0].abs());
        let res = n as f64 * (lam_max * 2f64.powi(-20) + 2.0 * quantum(max_abs(&a)));
        let ortho = orthonormality_defect(&q, n);
        // Unit-norm grid columns (quantum 2⁻³²) and `n` products per entry: `n²·2⁻³²` at most.
        let o_bound = (n * n) as f64 * 2f64.powi(-32);
        assert!(ortho <= o_bound, "n = {n}: orthonormality defect {ortho:e}");
        let r = q_diag_qt(&q, &l, n);
        let rec = max_abs(&r.iter().zip(&a).map(|(x, y)| x - y).collect::<Vec<f64>>());
        assert!(
            rec <= res,
            "n = {n}: reconstruction error {rec:e} over bound {res:e}"
        );
    }
}

#[test]
fn repeated_eigenvalues() {
    let n = 5usize;
    let mut rng = Lcg::new(0xDEAD_BEEF_1234_5678);
    let q0 = random_orthonormal(&mut rng, n, n);
    let d = [1.5f64, 1.5, 4.0, 4.0, 7.0];
    let a = q_diag_qt(&q0, &d, n);
    let (l, q) = eigh(&a, n).expect("eigh must converge on a repeated spectrum");
    let bound = eig_bound(n, d[4], quantum(max_abs(&a)));
    for i in 0..n {
        assert!(
            (l[i] - d[i]).abs() <= bound,
            "repeated spectrum: l[{i}] = {} != {} (bound {bound:e})",
            l[i],
            d[i]
        );
    }
    assert!(
        orthonormality_defect(&q, n) <= (n * n) as f64 * 2f64.powi(-32),
        "repeated: QᵀQ ≠ I"
    );
    let r = q_diag_qt(&q, &l, n);
    let res = n as f64 * (d[4] * 2f64.powi(-20) + 2.0 * quantum(max_abs(&a)));
    assert!(
        max_abs(&r.iter().zip(&a).map(|(x, y)| x - y).collect::<Vec<f64>>()) <= res,
        "repeated: reconstruction failed"
    );
}

#[test]
fn zero_matrix() {
    let n = 4usize;
    let a = vec![0.0f64; n * n];
    let (l, q) = eigh(&a, n).expect("the zero matrix converges immediately");
    for i in 0..n {
        assert!(
            l[i].abs() <= 2f64.powi(-32),
            "zero matrix: l[{i}] = {}",
            l[i]
        );
    }
    // Unit-norm grid columns (quantum 2⁻³²), `n` products per entry of the defect: `n²·2⁻³²`.
    assert!(
        orthonormality_defect(&q, n) <= (n * n) as f64 * 2f64.powi(-32),
        "zero: QᵀQ ≠ I"
    );
}

#[test]
fn diagonal_matrix() {
    // A semidefinite diagonal: the zero eigenvalue sits inside the one-word grid noise, so the
    // non-negativity bound must accept it.
    let n = 3usize;
    let a = vec![3.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0];
    let (l, q) = eigh(&a, n).expect("a semidefinite diagonal matrix converges immediately");
    let want = [0.0f64, 2.0, 3.0];
    let bound = eig_bound(n, 3.0, quantum(3.0));
    for i in 0..n {
        assert!(
            (l[i] - want[i]).abs() <= bound,
            "diagonal: l[{i}] = {} != {} (bound {bound:e})",
            l[i],
            want[i]
        );
    }
    assert!(orthonormality_defect(&q, n) < 1e-8, "diagonal: QᵀQ ≠ I");
    let r = q_diag_qt(&q, &l, n);
    let res = n as f64 * (3.0 * 2f64.powi(-20) + 2.0 * quantum(3.0));
    assert!(
        max_abs(&r.iter().zip(&a).map(|(x, y)| x - y).collect::<Vec<f64>>()) <= res,
        "diagonal: reconstruction failed"
    );
}

#[test]
fn indefinite_input_refused() {
    // A negative diagonal is outside the one-word bound and is refused before the sweep.
    let a = vec![3.0f64, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0];
    assert_eq!(eigh(&a, 3).err(), Some(EighError::NotConverged));
    // A negative-definite input has no positive eigenvalue to iterate on: refused, not shifted.
    let d = vec![-3.0f64, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -2.0];
    assert!(eigh(&d, 3).is_err(), "negative definite must be refused");
    // Zero diagonal, spectrum `±1`: only the full-spectrum bound can see it, and it refuses.
    let s = vec![0.0f64, 1.0, 1.0, 0.0];
    assert_eq!(eigh(&s, 2).err(), Some(EighError::NotConverged));
}

#[test]
fn rank_k_recovery() {
    let (o, n, k) = (6usize, 8usize, 3usize);
    let mut rng = Lcg::new(0x0123_4567_89AB_CDEF);
    let r1: Vec<f64> = (0..o * k).map(|_| rng.sym()).collect();
    let r2: Vec<f64> = (0..k * n).map(|_| rng.sym()).collect();
    let dw = matmul(&r1, &r2, o, k, n); // rank k by construction
    let hl = random_spd(&mut rng, n);
    let scale = max_abs(&dw);
    // `rank == k`: the top-k eigenspace of M is exactly range(ΔW), so Corr = ΔW. The subspace
    // angle is at most `√o·2⁻²⁰·(λ_max/gap)` and the gap between the k-th eigenvalue and the zero
    // null space is the k-th eigenvalue itself, so 2·θ ≤ 2·√6·2⁻²⁰ ≈ 5e-6 of the `Hl`-scale, and
    // `1e-4·max|ΔW|` carries that with two orders of head-room.
    let c = lowrank_corrector_exact(&qv(&dw), &qv(&hl), o, n, k).expect("rank = k < o");
    let err = max_abs(
        &uw(&c.corr)
            .iter()
            .zip(&dw)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    );
    assert!(err <= 1e-4 * scale, "rank = k recovery error {err:e}");
    // `rank > k`: the extra eigenspace is the null space of M; ΔW still lies in the span.
    let c2 = lowrank_corrector_exact(&qv(&dw), &qv(&hl), o, n, k + 2).expect("rank = k+2 < o");
    let err2 = max_abs(
        &uw(&c2.corr)
            .iter()
            .zip(&dw)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    );
    assert!(err2 <= 1e-3 * scale, "rank = k+2 recovery error {err2:e}");
}

#[test]
fn optimality_vs_random_projections() {
    let (o, n, rank) = (6usize, 8usize, 2usize);
    let mut rng = Lcg::new(0xABCD_EF01_2345_6789);
    let dw: Vec<f64> = (0..o * n).map(|_| rng.sym()).collect();
    let hl = random_spd(&mut rng, n);
    let c = lowrank_corrector_exact(&qv(&dw), &qv(&hl), o, n, rank).expect("rank < o");
    let r_ours: Vec<f64> = dw.iter().zip(&uw(&c.corr)).map(|(x, y)| x - y).collect();
    let t_ours = trace_r_hl_rt(&r_ours, &hl, o, n);
    for trial in 0..20 {
        let p = random_orthonormal(&mut rng, o, rank);
        let r_j = projection_residual(&dw, &p, o, n, rank);
        let t_j = trace_r_hl_rt(&r_j, &hl, o, n);
        assert!(
            t_ours <= t_j + 1e-6 * (1.0 + t_j.abs()),
            "trial {trial}: tr(R Hl Rᵀ) = {t_ours} > random projection {t_j}"
        );
    }
}

#[test]
fn top_sum_and_factors() {
    let (o, n, rank) = (5usize, 7usize, 2usize);
    let mut rng = Lcg::new(0x5555_AAAA_5555_AAAA);
    let dw: Vec<f64> = (0..o * n).map(|_| rng.sym()).collect();
    let hl = random_spd(&mut rng, n);
    let c = lowrank_corrector_exact(&qv(&dw), &qv(&hl), o, n, rank).expect("rank < o");
    // Corr = U·V: both sides are Q32.32 products of the same grid values; the f64 reproduct differs
    // by at most one quantum per product entry and `rank` entries per output entry.
    let uv = matmul(&uw(&c.u), &uw(&c.v), o, rank, n);
    let q = 2f64.powi(-32);
    let e = max_abs(
        &uv.iter()
            .zip(&uw(&c.corr))
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    );
    assert!(
        e <= 4.0 * (rank as f64) * q * (1.0 + max_abs(&uw(&c.corr))),
        "Corr ≠ U·V by {e:e}"
    );
    // top_sum against the removed f64 path: the two spectra differ by the eigenvalue bound of the
    // Q32.32 eigenstage, which is at most `√o·λ_max·2⁻²⁰` relative to `top_sum ≤ rank·λ_max`.
    let (_, want) = ref_corrector(&dw, &hl, o, n, rank);
    let top = from_grid(c.top_sum);
    let rel = (top - want).abs() / want.abs().max(1e-30);
    assert!(
        rel <= 1e-5,
        "top_sum = {top} != oracle top eigenvalues {want} (rel {rel:e})"
    );
    assert!(
        c.u.len() == o * rank && c.v.len() == rank * n && c.corr.len() == o * n,
        "factor shapes: u {}, v {}, corr {}",
        c.u.len(),
        c.v.len(),
        c.corr.len()
    );
}

#[test]
fn corrector_agrees_with_f64_oracle() {
    let (o, n, rank) = (6usize, 8usize, 2usize);
    let mut rng = Lcg::new(0x7777_3333_1111_9999);
    let dw: Vec<f64> = (0..o * n).map(|_| rng.sym()).collect();
    let hl = random_spd(&mut rng, n);
    let c = lowrank_corrector_exact(&qv(&dw), &qv(&hl), o, n, rank).expect("rank < o");
    let (corr_ref, top_ref) = ref_corrector(&dw, &hl, o, n, rank);
    let e = max_abs(
        &uw(&c.corr)
            .iter()
            .zip(&corr_ref)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    );
    let scale = max_abs(&dw);
    // Two subspace angles `θ ≤ √o·λ_max·2⁻²⁰/gap` of the row space (`gap` the top-`rank` spectral
    // gap, `≥ 0.1·λ_max` on this fixture), so `2·θ ≤ 5e-5`; `1e-3·max|ΔW|` carries it.
    assert!(
        e <= 1e-3 * scale,
        "corr vs oracle error {e:e} over {:.1e}",
        1e-3 * scale
    );
    let rel = (from_grid(c.top_sum) - top_ref).abs() / top_ref.abs().max(1e-30);
    assert!(rel <= 1e-5, "top_sum vs oracle rel {rel:e}");
}

#[test]
fn refusals() {
    let (o, n) = (4usize, 5usize);
    let dw = qv(&vec![0.1f64; o * n]);
    let hl = qv(&vec![0.0f64; n * n]);
    // rank > o.
    assert_eq!(
        lowrank_corrector_exact(&dw, &hl, o, n, o + 1).err(),
        Some(EighError::RankOutOfRange)
    );
    // non-square Hl: its length disagrees with n·n.
    let hl_bad = vec![0i64; n * n - 1];
    assert_eq!(
        lowrank_corrector_exact(&dw, &hl_bad, o, n, 1).err(),
        Some(EighError::ShapeMismatch)
    );
    // ΔW length disagrees with o·n.
    let dw_bad = vec![0i64; o * n + 1];
    assert_eq!(
        lowrank_corrector_exact(&dw_bad, &hl, o, n, 1).err(),
        Some(EighError::ShapeMismatch)
    );
    // eigh shape refusal.
    assert_eq!(eigh(&uw(&dw), n).err(), Some(EighError::ShapeMismatch));
    // A non-finite entry has no grid word.
    let mut nan = vec![0.0f64; n * n];
    nan[0] = f64::NAN;
    assert_eq!(eigh(&nan, n).err(), Some(EighError::NotConverged));
}
