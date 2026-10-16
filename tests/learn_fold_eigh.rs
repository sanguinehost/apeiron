//! `tests/learn_fold_eigh.rs` — symmetric eigendecomposition (cyclic Jacobi) and the exact
//! rank-`rank` corrector in the `Hl` metric (the `round_f16=False` path of `lowrank_corrector`).
//!
//! Positive: reconstruction `Q diag(L) Qᵀ = A` and orthonormality `QᵀQ = I` on random symmetric
//! matrices (`n = 3, 8, 24`) and on repeated-eigenvalue, zero and diagonal spectra; exact rank-`k`
//! recovery when `rank ≥ k`; the trace-optimality of `Corr` against 20 random rank-`rank`
//! projections; `top_sum` against the top eigenvalues; `Corr = U·V`. Boundary: the shape and rank
//! refusals. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixed-seed LCG maps a 24-bit integer field to an f64 in [0, 1), and the test \
              kernels index row-major `[r, c]` buffers by `i/j/k/l` against the spec's `DeltaW[i, j]` \
              arithmetic; the pedantic lints stay on for the crate's non-test code."
)]

use apeiron::learn::fold::fit::eigh::{eigh, lowrank_corrector_exact, EighError};

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

/// A random symmetric matrix, row-major `[n, n]`, entries in `[-1, 1)`.
fn random_symmetric(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = rng.sym();
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
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

#[test]
fn reconstruction_and_orthonormality() {
    for &n in &[3usize, 8, 24] {
        let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15 ^ (n as u64));
        let a = random_symmetric(&mut rng, n);
        let (l, q) = eigh(&a, n).expect("eigh must converge on a symmetrised input");
        assert_eq!(l.len(), n);
        assert_eq!(q.len(), n * n);
        // Ascending order.
        for i in 1..n {
            assert!(
                l[i - 1] <= l[i] + 1e-12,
                "n = {n}: eigenvalues not ascending at {i}: {} > {}",
                l[i - 1],
                l[i]
            );
        }
        let ortho = orthonormality_defect(&q, n);
        assert!(ortho < 1e-9, "n = {n}: orthonormality defect {ortho}");
        let r = q_diag_qt(&q, &l, n);
        let rec = max_abs(&r.iter().zip(&a).map(|(x, y)| x - y).collect::<Vec<f64>>());
        assert!(rec < 1e-9, "n = {n}: reconstruction error {rec}");
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
    for i in 0..n {
        assert!(
            (l[i] - d[i]).abs() < 1e-9,
            "repeated spectrum: l[{i}] = {} != {}",
            l[i],
            d[i]
        );
    }
    assert!(orthonormality_defect(&q, n) < 1e-9, "repeated: QᵀQ ≠ I");
    let r = q_diag_qt(&q, &l, n);
    assert!(
        max_abs(&r.iter().zip(&a).map(|(x, y)| x - y).collect::<Vec<f64>>()) < 1e-9,
        "repeated: reconstruction failed"
    );
}

#[test]
fn zero_matrix() {
    let n = 4usize;
    let a = vec![0.0f64; n * n];
    let (l, q) = eigh(&a, n).expect("the zero matrix converges in zero sweeps");
    for i in 0..n {
        assert!(l[i].abs() < 1e-15, "zero matrix: l[{i}] = {}", l[i]);
    }
    assert!(orthonormality_defect(&q, n) < 1e-12, "zero: QᵀQ ≠ I");
}

#[test]
fn diagonal_matrix() {
    let n = 3usize;
    let a = vec![3.0f64, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0];
    let (l, q) = eigh(&a, n).expect("a diagonal matrix converges in zero sweeps");
    let want = [-1.0f64, 2.0, 3.0];
    for i in 0..n {
        assert!((l[i] - want[i]).abs() < 1e-12, "diagonal: l = {l:?}");
    }
    assert!(orthonormality_defect(&q, n) < 1e-12, "diagonal: QᵀQ ≠ I");
    let r = q_diag_qt(&q, &l, n);
    assert!(
        max_abs(&r.iter().zip(&a).map(|(x, y)| x - y).collect::<Vec<f64>>()) < 1e-12,
        "diagonal: reconstruction failed"
    );
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
    // rank == k: the top-k eigenspace of M is exactly range(ΔW), so Corr = ΔW.
    let c = lowrank_corrector_exact(&dw, &hl, o, n, k).expect("rank = k < o");
    let err = max_abs(
        &c.corr
            .iter()
            .zip(&dw)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    );
    assert!(err <= 1e-8 * scale, "rank = k recovery error {err}");
    // rank > k: the extra eigenspace is the numerical null space of M; ΔW still lies in it.
    let c2 = lowrank_corrector_exact(&dw, &hl, o, n, k + 2).expect("rank = k+2 < o");
    let err2 = max_abs(
        &c2.corr
            .iter()
            .zip(&dw)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    );
    assert!(err2 <= 1e-6 * scale, "rank = k+2 recovery error {err2}");
}

#[test]
fn optimality_vs_random_projections() {
    let (o, n, rank) = (6usize, 8usize, 2usize);
    let mut rng = Lcg::new(0xABCD_EF01_2345_6789);
    let dw: Vec<f64> = (0..o * n).map(|_| rng.sym()).collect();
    let hl = random_spd(&mut rng, n);
    let c = lowrank_corrector_exact(&dw, &hl, o, n, rank).expect("rank < o");
    let r_ours: Vec<f64> = dw.iter().zip(&c.corr).map(|(x, y)| x - y).collect();
    let t_ours = trace_r_hl_rt(&r_ours, &hl, o, n);
    for trial in 0..20 {
        let p = random_orthonormal(&mut rng, o, rank);
        let r_j = projection_residual(&dw, &p, o, n, rank);
        let t_j = trace_r_hl_rt(&r_j, &hl, o, n);
        assert!(
            t_ours <= t_j + 1e-9,
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
    let c = lowrank_corrector_exact(&dw, &hl, o, n, rank).expect("rank < o");
    // Corr = U·V.
    let uv = matmul(&c.u, &c.v, o, rank, n);
    let e = max_abs(
        &uv.iter()
            .zip(&c.corr)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>(),
    );
    assert!(e <= 1e-12, "Corr ≠ U·V by {e}");
    // top_sum == Σ max(λ, 0) over the top `rank` eigenvalues of M = (ΔW·Hl)·ΔWᵀ.
    let w1 = matmul(&dw, &hl, o, n, n);
    let m = matmul(&w1, &transpose(&dw, o, n), o, n, o);
    let (evals, _) = eigh(&m, o).expect("eigh of M");
    let want: f64 = evals[o - rank..].iter().map(|&lam| lam.max(0.0)).sum();
    assert!(
        (c.top_sum - want).abs() <= 1e-9 * (1.0 + want.abs()),
        "top_sum = {} != Σ top eigenvalues {}",
        c.top_sum,
        want
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
fn refusals() {
    let (o, n) = (4usize, 5usize);
    let dw = vec![0.1f64; o * n];
    let hl = vec![0.0f64; n * n];
    // rank > o.
    assert_eq!(
        lowrank_corrector_exact(&dw, &hl, o, n, o + 1).err(),
        Some(EighError::RankOutOfRange)
    );
    // non-square Hl: its length disagrees with n·n.
    let hl_bad = vec![0.0f64; n * n - 1];
    assert_eq!(
        lowrank_corrector_exact(&dw, &hl_bad, o, n, 1).err(),
        Some(EighError::ShapeMismatch)
    );
    // ΔW length disagrees with o·n.
    let dw_bad = vec![0.0f64; o * n + 1];
    assert_eq!(
        lowrank_corrector_exact(&dw_bad, &hl, o, n, 1).err(),
        Some(EighError::ShapeMismatch)
    );
    // eigh shape refusal.
    assert_eq!(eigh(&dw, n).err(), Some(EighError::ShapeMismatch));
}
