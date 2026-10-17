//! Gates for `src/learn/fold/stats/basis.rs`: the reverse water-filling closed form and its
//! side-information accounting, the basis-invariance of `Π d_j = det H`, the adapted butterfly's
//! orthogonality and its `π/4` identity with `fwht128/√128`, the concentration advantage on a
//! planted subspace, and the real `Qwen2.5 0.5B` 1.18 b/w table over 21 tensors.
//!
//! Grade: `tested` for [1]–[4] and `model` for the printed real-model numbers (they are a density
//! model of the coefficient magnitudes, not a coder measurement). All synthetic inputs are a fixed
//! inline LCG; the real-model test is `#[ignore]`d and prints one row per tensor.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the synthetic fixtures map integer LCG outputs and the real model's Q32.32 words to \
              f64 by exact power-of-two division; the single-letter names are the model's matrix \
              symbols; the equality checks are the water-filling closed form at a chosen θ, which \
              is exact by construction"
)]

use std::fs;
use std::time::Instant;

use apeiron::engine::Engine;
use apeiron::learn::fold::fit::steps::fwht128;
use apeiron::learn::fold::run::npy::read_npy_i4;
use apeiron::learn::fold::stats::basis::{
    apply_basis, block_metric, butterfly_basis, butterfly_basis_fixed, butterfly_side_bits,
    cholesky_pivots, column_energies, damped_gram, distortion_at, eigen_basis, hadamard_basis,
    identity_basis, rate_at, row_energies, solve_water_fill, sum_log_column_energies,
    uniform_distortion, Coefficients, Density, GROUP,
};
use apeiron::learn::fold::stats::capture::{Grams, Normalisation};
use apeiron::learn::grad::model::{forward, Config, Weights};

/// Deterministic xorshift64 LCG shared by every synthetic fixture.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// Standard normal (sum of twelve uniforms minus six).
    fn gauss(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// One grid quantum, `2⁻³²`: the bound the `Q32.32` carrier rounds to at every reported word.
const Q: f64 = 1.0 / 4_294_967_296.0;

/// The pre-migration `f64` body of [`damped_gram`], kept as the agreement oracle.
fn ref_damped_gram(h: &[f64], n: usize) -> Vec<f64> {
    let mean = h.iter().step_by(n + 1).sum::<f64>() / n as f64;
    let mut out = h.to_vec();
    for i in 0..n {
        out[i * n + i] += 0.01 * mean;
    }
    out
}

/// The pre-migration `f64` body of [`apply_basis`].
fn ref_apply_basis(w: &[f64], o: usize, n: usize, q: &[f64]) -> Vec<f64> {
    let mut b = vec![0.0f64; o * n];
    for blk in 0..n / GROUP {
        let off = blk * GROUP;
        for i in 0..o {
            for k in 0..GROUP {
                let a = w[i * n + off + k];
                for c in 0..GROUP {
                    b[i * n + off + c] += a * q[(off + k) * n + off + c];
                }
            }
        }
    }
    b
}

/// The pre-migration `f64` body of [`block_metric`]: `Qᵀ·(H·Q)`, `O(n³)`.
fn ref_block_metric(q: &[f64], h: &[f64], n: usize) -> Vec<f64> {
    let mut hq = vec![0.0f64; n * n];
    for i in 0..n {
        for c in 0..n {
            hq[i * n + c] = (0..n).map(|k| h[i * n + k] * q[k * n + c]).sum();
        }
    }
    let mut m = vec![0.0f64; n * n];
    for a in 0..n {
        for c in 0..n {
            m[a * n + c] = (0..n).map(|i| q[i * n + a] * hq[i * n + c]).sum();
        }
    }
    m
}

/// The pre-migration `f64` body of [`row_energies`].
fn ref_row_energies(b: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; o];
    for i in 0..o {
        for j in 0..n {
            r[i] += b[i * n + j] * b[i * n + j];
        }
    }
    r
}

/// The pre-migration `f64` body of [`column_energies`].
fn ref_column_energies(b: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n];
    for i in 0..o {
        for j in 0..n {
            c[j] += b[i * n + j] * b[i * n + j];
        }
    }
    c
}

/// The pre-migration `f64` body of [`sum_log_column_energies`].
fn ref_sum_log_column_energies(c: &[f64]) -> f64 {
    c.iter().map(|&v| v.max(f64::MIN_POSITIVE).ln()).sum()
}

/// The pre-migration `f64` body of [`cholesky_pivots`] (with the previous `ridge::cholesky`).
fn ref_cholesky_pivots(m: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                assert!(s > 0.0, "ref cholesky: pivot {s}");
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    (0..n).map(|j| l[j * n + j] * l[j * n + j]).collect()
}

/// The pre-migration `f64` body of [`Density::sigma`] for the separable model.
fn ref_separable(b: &[f64], o: usize, n: usize) -> Vec<f64> {
    let r = ref_row_energies(b, o, n);
    let c = ref_column_energies(b, o, n);
    let m: f64 = r.iter().sum();
    let mut out = vec![0.0f64; o * n];
    if m > 0.0 {
        for i in 0..o {
            for j in 0..n {
                out[i * n + j] = r[i] * c[j] / m;
            }
        }
    }
    out
}

/// The pre-migration `f64` body of [`Density::sigma`] for the tile model.
fn ref_tile(b: &[f64], o: usize, n: usize) -> Vec<f64> {
    let nti = o.div_ceil(16);
    let ntj = n.div_ceil(16);
    let mut mean = vec![0.0f64; nti * ntj];
    let mut count = vec![0usize; nti * ntj];
    for i in 0..o {
        for j in 0..n {
            let k = (i / 16) * ntj + j / 16;
            mean[k] += b[i * n + j] * b[i * n + j];
            count[k] += 1;
        }
    }
    for k in 0..mean.len() {
        if count[k] > 0 {
            mean[k] /= count[k] as f64;
        }
    }
    (0..o * n)
        .map(|k| mean[(k / n / 16) * ntj + (k % n) / 16])
        .collect()
}

/// The pre-migration `f64` body of [`distortion_at`].
fn ref_distortion(lambda: &[f64], theta: f64) -> f64 {
    if theta <= 0.0 {
        return 0.0;
    }
    lambda.iter().map(|&l| l.min(theta)).sum()
}

/// The pre-migration `f64` body of [`rate_at`].
fn ref_rate(lambda: &[f64], theta: f64) -> f64 {
    if theta <= 0.0 {
        return if lambda.iter().any(|&l| l > 0.0) {
            f64::INFINITY
        } else {
            0.0
        };
    }
    let t = theta.log2();
    lambda
        .iter()
        .filter(|&&l| l > theta)
        .map(|&l| 0.5 * (l.log2() - t))
        .sum()
}

/// The pre-migration `f64` body of [`solve_water_fill`].
fn ref_solve_water_fill(lambda: &[f64], budget: f64, side: f64) -> (f64, f64, f64) {
    let target = budget - side;
    let max = lambda.iter().copied().fold(0.0f64, f64::max);
    if lambda.is_empty() || max <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let mut t_hi = max.log2() + 1.0;
    let min_pos = lambda
        .iter()
        .copied()
        .filter(|&l| l > 0.0)
        .fold(f64::INFINITY, f64::min);
    let mut t_lo = min_pos.log2() - 1.0;
    let mut gap = 1.0f64;
    while ref_rate(lambda, 2.0f64.powf(t_lo)) < target {
        t_lo -= gap;
        gap *= 2.0;
    }
    let mut t = t_hi;
    for _ in 0..200 {
        t = 0.5 * (t_lo + t_hi);
        let r = ref_rate(lambda, 2.0f64.powf(t));
        if (r - target).abs() <= 1e-9 * lambda.len() as f64 {
            break;
        }
        if r > target {
            t_lo = t;
        } else {
            t_hi = t;
        }
    }
    let theta = 2.0f64.powf(t);
    (
        theta,
        ref_distortion(lambda, theta),
        ref_rate(lambda, theta),
    )
}

/// The pre-migration `f64` body of [`uniform_distortion`].
fn ref_uniform(lambda: &[f64], budget: f64, side: f64) -> f64 {
    let bits = (budget - side) / lambda.len() as f64;
    let factor = 2.0f64.powf(-2.0 * bits);
    lambda.iter().map(|&l| l * factor).sum()
}

/// Relative-plus-absolute agreement assertion with the bound stated at the call.
fn close(a: f64, b: f64, rel: f64, abs: f64, what: &str) {
    let d = (a - b).abs();
    assert!(
        d <= rel * b.abs() + abs,
        "{what}: {a} vs oracle {b} differs by {d} > {rel}·|{b}| + {abs}"
    );
}

/// (1) The closed-form `D(θ)` and `R(θ)` of a geometric `λ`, and the side-information budget.
#[test]
fn a_water_filling_closed_form_and_side_bits() {
    // Diagonal variances 4, 1, 0.25, 0.0625 = 4·4^-k. At θ = 1 the closed forms are
    // D = 1 + 1 + 0.25 + 0.0625 and R = ½·log₂ 4 = 1.
    let lam = [4.0f64, 1.0, 0.25, 0.0625];
    let theta = 1.0f64;
    let d = distortion_at(&lam, theta);
    let r = rate_at(&lam, theta);
    let want_d = 1.0 + 1.0 + 0.25 + 0.0625;
    // `D` sums exact grid minima and rounds the sum once (≤ ½q); `R` adds four `log₂` words, each
    // within 2q of the series bound, halves, then rounds: ≤ 16q.
    assert!((d - want_d).abs() < 4.0 * Q, "D = {d} != {want_d}");
    assert!((r - 1.0).abs() < 16.0 * Q, "R = {r} != 1");

    // Side information is charged against the rate budget: budget 1.18·4 = 4.72, side 1.0 bits
    // leaves 3.72 bits of coefficient rate. Dropping the subtraction would solve for 4.72.
    let budget = 1.18 * 4.0;
    let side = 1.0;
    let sol = solve_water_fill(&lam, budget, side);
    let target = budget - side;
    // The solve stops at `|R − target| ≤ 4q·N`; the four `log₂` words add ≤ 8q.
    assert!(
        (sol.rate - target).abs() < 16.0 * Q * lam.len() as f64,
        "rate {} != target {target}",
        sol.rate
    );
    assert!(
        (sol.distortion - distortion_at(&lam, sol.theta)).abs() < 16.0 * Q,
        "D {} != D(θ) {}",
        sol.distortion,
        distortion_at(&lam, sol.theta)
    );
    assert!(
        (sol.rate - 1.18 * 4.0).abs() > 0.1,
        "the side bits did not move the rate"
    );

    // The fixed-rate baseline is the closed form Σ λ·2^{-2·(1.18 - side/N)}.
    let want_u: f64 = lam
        .iter()
        .map(|l| l * 2.0f64.powf(-2.0 * (1.18 - side / 4.0)))
        .sum();
    let u = uniform_distortion(&lam, budget, side);
    // `bits` rounds once (≤ ½q), `2^x` adds ≤ 2q relative, and `Σ λ·factor` rounds once (≤ ½q·N).
    assert!(
        (u - want_u).abs() < 16.0 * Q * want_u + 4.0 * Q * lam.len() as f64,
        "D_u = {u} != {want_u}"
    );
}

/// A deterministic dense symmetric positive-definite Gram.
fn spd(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut a = vec![0.0f64; n * n];
    for v in &mut a {
        *v = rng.gauss();
    }
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += a[i * n + k] * a[j * n + k];
            }
            h[i * n + j] = s;
        }
    }
    for i in 0..n {
        h[i * n + i] += n as f64;
    }
    h
}

/// (2) `Π d_j = det H` for every orthogonal block basis: the log-pivot sums agree across the
/// identity and Hadamard bases, and with the Cholesky pivots of `H` itself.
#[test]
fn b_pivot_product_is_basis_invariant() {
    let n = GROUP;
    let h = damped_gram(&spd(n, 0x5eed), n).expect("damped H");
    let qi = identity_basis(n).expect("identity");
    let qh = hadamard_basis(n).expect("hadamard");
    let di = cholesky_pivots(&block_metric(&qi, &h, n).expect("M_i"), n).expect("pivots i");
    let dh = cholesky_pivots(&block_metric(&qh, &h, n).expect("M_h"), n).expect("pivots h");
    let s_i: f64 = di.iter().map(|d| d.ln()).sum();
    let s_h: f64 = dh.iter().map(|d| d.ln()).sum();
    let d_ref = cholesky_pivots(&h, n).expect("pivots H");
    let s_ref: f64 = d_ref.iter().map(|d| d.ln()).sum();
    // Each `d_j` is one exact `i128` sum rounded once, but the `L_jk` it subtracts carry their own
    // ≤½q rounding, so a pivot's relative error is bounded by `j·q`; the log sums over `n` pivots
    // are within `n²·q/2` of the f64 reference, and three such sums are compared.
    let tol = 2.0 * (n * n) as f64 * Q;
    assert!((s_i - s_h).abs() < tol, "log Πd_i {s_i} != log Πd_h {s_h}");
    assert!(
        (s_i - s_ref).abs() < tol,
        "log Πd_i {s_i} != log det H {s_ref}"
    );
}

/// Column `j` of `q`, length `n`.
fn column(q: &[f64], n: usize, j: usize) -> Vec<f64> {
    (0..n).map(|r| q[r * n + j]).collect()
}

/// (3) The adaptive butterfly is orthogonal, and with every angle `π/4` it is `fwht128/√128` up to
/// column signs.
#[test]
fn c_butterfly_orthogonal_and_fixed_angle_is_hadamard() {
    let n = 2 * GROUP;
    let o = 24;
    let mut rng = Lcg::new(0x00b0_77e2);
    let w: Vec<f64> = (0..o * n).map(|_| rng.gauss()).collect();

    let q = butterfly_basis(&w, o, n).expect("butterfly");
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for r in 0..n {
                s += q[r * n + i] * q[r * n + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((s - want).abs());
        }
    }
    // `Q` is a product of at most 28 Givens rotations (one mandatory pass plus three sweeps), each
    // with `c² + s² − 1 ≤ 4q` from its two rounded words and the two rounded products.
    assert!(worst < 128.0 * Q, "‖QᵀQ − I‖_max = {worst}");

    let qf = butterfly_basis_fixed(&w, o, n, std::f64::consts::FRAC_PI_4).expect("fixed");
    let qh = hadamard_basis(n).expect("hadamard");
    // Seven stages of `cos θ = sin θ = 2^{-1/2}` rounded to grid words: ≤ 7·(2q) per entry.
    let tol = 64.0 * Q;
    for j in 0..n {
        let a = column(&qf, n, j);
        let b = column(&qh, n, j);
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
        let mut diff = 0.0f64;
        for (x, y) in a.iter().zip(&b) {
            diff = diff.max((sign * x - y).abs());
        }
        assert!(diff < tol, "column {j} differs by {diff} beyond a sign");
    }

    // The seam is the WHT body: one fixed transform of a unit vector is the Hadamard column, and the
    // basis entry is that `f64` value quantised once (≤ ½q).
    let e = column(&identity_basis(GROUP).expect("I"), GROUP, 5);
    let f = fwht128(&e).expect("fwht");
    for i in 0..GROUP {
        assert!(
            (f[i] - qh[i * n + 5]).abs() < Q,
            "fwht128 column mismatch at {i}"
        );
    }
}

/// `t` orthonormalised standard-normal vectors of length `n`.
fn orthonormal(rng: &mut Lcg, n: usize, t: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(t);
    for _ in 0..t {
        let mut v: Vec<f64> = (0..n).map(|_| rng.gauss()).collect();
        for u in &out {
            let dot: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
            for (a, b) in v.iter_mut().zip(u) {
                *a -= dot * b;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for a in &mut v {
            *a /= norm;
        }
        out.push(v);
    }
    out
}

/// (4) On a synthetic `W` carrying a dominant two-dimensional column subspace, the adapted
/// butterfly concentrates the column energy and lowers the water-filling distortion below the
/// Hadamard basis.
#[test]
fn d_adapted_butterfly_concentrates_a_planted_subspace() {
    let n = GROUP;
    let o = 3 * GROUP / 2;
    let mut rng = Lcg::new(0x009a_17e5);
    let u = orthonormal(&mut rng, o, 2);
    let b = orthonormal(&mut rng, n, 2);
    let mut w = vec![0.0f64; o * n];
    for v in &mut w {
        *v = 0.01 * rng.gauss();
    }
    for i in 0..o {
        for j in 0..n {
            w[i * n + j] += 8.0 * u[0][i] * b[0][j] + 5.0 * u[1][i] * b[1][j];
        }
    }

    let h = damped_gram(&identity_basis(n).expect("I"), n).expect("damped I");
    let qj = butterfly_basis(&w, o, n).expect("butterfly");
    let qh = hadamard_basis(n).expect("hadamard");
    let cj = column_energies(&apply_basis(&w, o, n, &qj).expect("B_j"), o, n).expect("c_j");
    let ch = column_energies(&apply_basis(&w, o, n, &qh).expect("B_h"), o, n).expect("c_h");

    let mut top: Vec<f64> = cj.clone();
    top.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
    let total: f64 = cj.iter().sum();
    let top2 = (top[0] + top[1]) / total;
    assert!(
        top2 > 0.5,
        "adapted basis keeps only {top2} of the energy in its top two columns"
    );
    let sj = sum_log_column_energies(&cj);
    let sh = sum_log_column_energies(&ch);
    assert!(sj < sh, "Σ log c_j adapted {sj} >= hadamard {sh}");

    let budget = 1.18 * o as f64 * n as f64;
    let aj = Coefficients::from_basis(&w, o, n, &h, &qj)
        .expect("coeff j")
        .evaluate(Density::Separable, butterfly_side_bits(n), budget)
        .expect("arm j");
    let ah = Coefficients::from_basis(&w, o, n, &h, &qh)
        .expect("coeff h")
        .evaluate(Density::Separable, 0.0, budget)
        .expect("arm h");
    assert!(
        aj.water_filled < ah.water_filled,
        "D_wf(J) {} >= D_wf(H) {}",
        aj.water_filled,
        ah.water_filled
    );
}

/// (5) The `Q32.32` carrier agrees with the pre-migration `f64` bodies on a fixed LCG sample. Each
/// bound is the number of grid roundings the path performs, times `Q`; the `f64` oracle's own
/// rounding is `2⁻⁵³`-level and does not enter a bound above `Q`.
#[test]
#[allow(clippy::too_many_lines)]
fn e_q32_agrees_with_f64_oracle() {
    let mut rng = Lcg::new(0x00a9_9e11);

    // Damped Gram (n = 8): the mean diagonal is one rounding, `DAMP·mean` one more, and `DAMP_Q`
    // itself is within 4q of 0.01.
    let n0 = 8;
    let h0 = spd(n0, 0x1234_5678);
    let dg = damped_gram(&h0, n0).expect("damped");
    let rdg = ref_damped_gram(&h0, n0);
    for (k, (a, b)) in dg.iter().zip(&rdg).enumerate() {
        close(*a, *b, 8.0 * Q, Q, &format!("damped_gram[{k}]"));
    }

    // The pivot recurrence: `L_jk` carries its own ≤ ½q rounding and is subtracted from `m_jj`, so a
    // pivot's relative error is ≤ `j·q`; the factor is `n = 8`.
    let gp = cholesky_pivots(&dg, n0).expect("pivots");
    let rp = ref_cholesky_pivots(&dg, n0);
    for (j, (a, b)) in gp.iter().zip(&rp).enumerate() {
        close(*a, *b, 8.0 * n0 as f64 * Q, 8.0 * Q, &format!("pivot[{j}]"));
    }

    // `D` and `R` at a fixed θ over 64 geometric-ish rates: every input word rounds once (≤ ½q) and
    // each `log₂` adds ≤ 2q.
    let lam: Vec<f64> = (0..64).map(|_| 0.05 + 4.0 * rng.uniform()).collect();
    let nn = lam.len() as f64;
    let theta = 0.7;
    close(
        distortion_at(&lam, theta),
        ref_distortion(&lam, theta),
        4.0 * Q,
        2.0 * nn * Q,
        "distortion_at",
    );
    close(
        rate_at(&lam, theta),
        ref_rate(&lam, theta),
        4.0 * Q,
        4.0 * nn * Q,
        "rate_at",
    );

    // The solve stops at `|R − target| ≤ 4q·N`; with `dR/dt = −N/2` that is `|Δlog₂θ| ≤ 8q`, so
    // `|ΔD| ≤ N·θ·6q + N·q/2` and `|ΔR| ≤ 4q·N + 2q·N`.
    let budget = 1.18 * nn;
    let sol = solve_water_fill(&lam, budget, 1.0);
    let (_, rd, rr) = ref_solve_water_fill(&lam, budget, 1.0);
    close(
        sol.distortion,
        rd,
        8.0 * Q,
        32.0 * nn * Q,
        "solve_water_fill distortion",
    );
    close(
        sol.rate,
        rr,
        8.0 * Q,
        32.0 * nn * Q,
        "solve_water_fill rate",
    );

    // `bits` rounds once (≤ ½q), `2^x` adds ≤ 2q relative, `Σ λ·factor` rounds once.
    close(
        uniform_distortion(&lam, budget, 1.0),
        ref_uniform(&lam, budget, 1.0),
        8.0 * Q,
        4.0 * nn * Q,
        "uniform_distortion",
    );

    // Row and column energies: every `b` word rounds once (≤ ½q), the `i128` sum is exact, one
    // output rounding; `o = 6`, `n = 128`.
    let (o, nb) = (6, GROUP);
    let b: Vec<f64> = (0..o * nb).map(|_| rng.gauss()).collect();
    let r = row_energies(&b, o, nb).expect("rows");
    let rr_ = ref_row_energies(&b, o, nb);
    for (i, (a, w)) in r.iter().zip(&rr_).enumerate() {
        close(*a, *w, 4.0 * Q, 2.0 * nb as f64 * Q, &format!("row[{i}]"));
    }
    let c = column_energies(&b, o, nb).expect("cols");
    let rc = ref_column_energies(&b, o, nb);
    for (j, (a, w)) in c.iter().zip(&rc).enumerate() {
        close(*a, *w, 4.0 * Q, 2.0 * o as f64 * Q, &format!("col[{j}]"));
    }

    // `Σ ln c_j` over the quantised energies: one `log₂` and one `ln 2` product per column, ≤ 4q.
    close(
        sum_log_column_energies(&c),
        ref_sum_log_column_energies(&rc),
        8.0 * Q,
        4.0 * nb as f64 * Q,
        "sum_log_column_energies",
    );

    // The two density models over `6 × 128`: `r_i·c_j/m` and the 16×16 tile mean, one rounding per
    // output word.
    let sig = Density::Separable.sigma(&b, o, nb).expect("separable");
    let rsig = ref_separable(&b, o, nb);
    for (k, (a, w)) in sig.iter().zip(&rsig).enumerate() {
        close(
            *a,
            *w,
            16.0 * Q,
            4.0 * nb as f64 * Q,
            &format!("separable[{k}]"),
        );
    }
    let til = Density::Tile.sigma(&b, o, nb).expect("tile");
    let rtil = ref_tile(&b, o, nb);
    for (k, (a, w)) in til.iter().zip(&rtil).enumerate() {
        close(*a, *w, 16.0 * Q, 4.0 * nb as f64 * Q, &format!("tile[{k}]"));
    }

    // `B = W·Q` and `M = Qᵀ·H·Q` with the Hadamard basis: each input word rounds once, so the
    // relative error is ≤ `GROUP·q` per accumulation of `GROUP` terms.
    let qh = hadamard_basis(nb).expect("hadamard");
    let bq = apply_basis(&b, o, nb, &qh).expect("apply");
    let rbq = ref_apply_basis(&b, o, nb, &qh);
    for (k, (a, w)) in bq.iter().zip(&rbq).enumerate() {
        close(
            *a,
            *w,
            4.0 * nb as f64 * Q,
            4.0 * nb as f64 * Q,
            &format!("apply[{k}]"),
        );
    }
    let hh = damped_gram(&spd(nb, 0x0bad_5eed), nb).expect("damped 128");
    let m = block_metric(&qh, &hh, nb).expect("metric");
    let rm = ref_block_metric(&qh, &hh, nb);
    for (k, (a, w)) in m.iter().zip(&rm).enumerate() {
        close(
            *a,
            *w,
            8.0 * nb as f64 * Q,
            8.0 * nb as f64 * Q,
            &format!("metric[{k}]"),
        );
    }
}

/// The real-model gate's constants.
const MODEL: &str = "~/models/gguf/qwen05b-f16.gguf";
const TOKENS: &str = "~/lanes/w18/kd/c4data/train-c4.qwen.npy";
const LAYERS: [usize; 3] = [2, 11, 20];
const BITS: f64 = 1.18;
const PROJECT_SECONDS: f64 = 300.0;
const MIN_TOKENS: usize = 256;
const TAPE_CAP_BYTES: u128 = 6 * (1u128 << 30);

/// The `f64` copy of a Q32.32 matrix.
fn matrix_f64(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&v| v as f64 / 4_294_967_296.0).collect()
}

/// A conservative tape-size estimate at `t` tokens, from the forward's own buffer widths.
fn tape_bytes_at(cfg: &Config, t: usize) -> u128 {
    let e = cfg.n_embd as u128;
    let q = (cfg.n_head * cfg.head_dim) as u128;
    let kv = (cfg.n_head_kv * cfg.head_dim) as u128;
    let ff = cfg.n_ff as u128;
    let per_token = 4 * e + 2 * q + 2 * kv + 3 * ff;
    let per_layer = t as u128 * per_token * 8 + (t * t * cfg.n_head) as u128 * 8;
    cfg.n_layer as u128 * per_layer + t as u128 * (2 * e + cfg.vocab as u128) * 8
}

/// One tensor's seven arm values for one density model.
#[derive(Clone, Copy, Default)]
struct Row {
    du_h: f64,
    wf_h: f64,
    wf_i: f64,
    wf_j: f64,
    wf_v: f64,
    or_h: f64,
    or_j: f64,
}

impl std::ops::AddAssign for Row {
    fn add_assign(&mut self, other: Row) {
        self.du_h += other.du_h;
        self.wf_h += other.wf_h;
        self.wf_i += other.wf_i;
        self.wf_j += other.wf_j;
        self.wf_v += other.wf_v;
        self.or_h += other.or_h;
        self.or_j += other.or_j;
    }
}

/// Evaluate all four bases on one tensor, returning the `(separable, tile)` rows.
fn measure(w_i64: &[i64], o: usize, n: usize, h: &[f64], budget: f64) -> (Row, Row) {
    let w = matrix_f64(w_i64);
    let mut s = Row::default();
    let mut t = Row::default();

    let qh = hadamard_basis(n).expect("hadamard");
    let ch = Coefficients::from_basis(&w, o, n, h, &qh).expect("coeff H");
    let hs = ch.evaluate(Density::Separable, 0.0, budget).expect("H s");
    let ht = ch.evaluate(Density::Tile, 0.0, budget).expect("H t");
    let ohs = ch
        .evaluate(Density::Oracle, 0.0, budget)
        .expect("H oracle s");
    let oht = ch
        .evaluate(Density::Oracle, 0.0, budget)
        .expect("H oracle t");
    s.du_h = hs.uniform;
    s.wf_h = hs.water_filled;
    s.or_h = ohs.water_filled;
    t.du_h = ht.uniform;
    t.wf_h = ht.water_filled;
    t.or_h = oht.water_filled;

    let qi = identity_basis(n).expect("identity");
    let ci = Coefficients::from_basis(&w, o, n, h, &qi).expect("coeff I");
    s.wf_i = ci
        .evaluate(Density::Separable, 0.0, budget)
        .expect("I s")
        .water_filled;
    t.wf_i = ci
        .evaluate(Density::Tile, 0.0, budget)
        .expect("I t")
        .water_filled;

    let angle_bits = butterfly_side_bits(n);
    let qj = butterfly_basis(&w, o, n).expect("butterfly");
    let cj = Coefficients::from_basis(&w, o, n, h, &qj).expect("coeff J");
    s.wf_j = cj
        .evaluate(Density::Separable, angle_bits, budget)
        .expect("J s")
        .water_filled;
    t.wf_j = cj
        .evaluate(Density::Tile, angle_bits, budget)
        .expect("J t")
        .water_filled;
    s.or_j = cj
        .evaluate(Density::Oracle, angle_bits, budget)
        .expect("J oracle s")
        .water_filled;
    t.or_j = cj
        .evaluate(Density::Oracle, angle_bits, budget)
        .expect("J oracle t")
        .water_filled;

    let qv = eigen_basis(&w, o, n).expect("eigen");
    let cv = Coefficients::from_basis(&w, o, n, h, &qv).expect("coeff V");
    s.wf_v = cv
        .evaluate(Density::Separable, 0.0, budget)
        .expect("V s")
        .water_filled;
    t.wf_v = cv
        .evaluate(Density::Tile, 0.0, budget)
        .expect("V t")
        .water_filled;

    (s, t)
}

/// One density section: header, per-tensor rows, sums and ratios.
fn report(label: &str, rows: &[(String, usize, usize, Row)]) {
    println!("\n== density {label} ==  (target {BITS} b/w, D in the metric tr(ΔW·H·ΔWᵀ))");
    println!(
        "{:>8} {:>5} {:>5} | {:>11} {:>11} | {:>11} | {:>11} | {:>11} | {:>11} {:>11}",
        "tensor", "O", "n", "D_u(H)", "D_wf(H)", "D_wf(I)", "D_wf(J)", "D_wf(V)", "or(H)", "or(J)"
    );
    let mut sum = Row::default();
    for (name, o, n, r) in rows {
        sum += *r;
        println!(
            "{name:>8} {o:>5} {n:>5} | {:>11.5e} {:>11.5e} | {:>11.5e} | {:>11.5e} | {:>11.5e} | {:>11.5e} {:>11.5e}",
            r.du_h, r.wf_h, r.wf_i, r.wf_j, r.wf_v, r.or_h, r.or_j
        );
    }
    println!(
        "{:>8} {:>5} {:>5} | {:>11.5e} {:>11.5e} | {:>11.5e} | {:>11.5e} | {:>11.5e} | {:>11.5e} {:>11.5e}",
        "SUM", "", "", sum.du_h, sum.wf_h, sum.wf_i, sum.wf_j, sum.wf_v, sum.or_h, sum.or_j
    );
    println!(
        "ratio D_u(H)/D_wf(J) = {:.4}   D_wf(H)/D_wf(J) = {:.4}   D_wf(V)/D_wf(J) = {:.4}",
        sum.du_h / sum.wf_j,
        sum.wf_h / sum.wf_j,
        sum.wf_v / sum.wf_j
    );
}

/// The real `Qwen2.5 0.5B` 1.18 b/w table over layers 2, 11 and 20 and the seven linears.
#[test]
#[allow(clippy::type_complexity)]
#[ignore = "loads ~1 GB and runs the crate forward over the C4 calibration tokens; --release --ignored --nocapture"]
fn basis_rate_qwen05b() {
    let start = Instant::now();
    let bytes = fs::read(TOKENS).expect("read token npy");
    let ids = read_npy_i4(&bytes).expect("parse token npy");
    assert!(ids.len() >= MIN_TOKENS, "only {} tokens", ids.len());
    let engine = Engine::open(MODEL).expect("open gguf");
    let (weights, cfg) = Weights::from_engine(&engine).expect("load weights");
    eprintln!(
        "[basis] loaded model at {:.1}s",
        start.elapsed().as_secs_f64()
    );
    let e = cfg.n_embd;
    let q_out = cfg.q_out().expect("q_out");
    let kv_out = cfg.kv_out().expect("kv_out");
    let n_ff = cfg.n_ff;

    // Probe: 32 tokens, forward plus the full Gram accumulation, then project the wall linearly.
    let probe: Vec<u32> = ids[..32]
        .iter()
        .map(|&v| u32::try_from(v).expect("token"))
        .collect();
    let t0 = Instant::now();
    let (_, tape32) = forward(&weights, &cfg, &probe).expect("forward probe");
    let mut grams32 = Grams::new(&cfg).expect("grams");
    grams32.accumulate(&tape32, &cfg).expect("accumulate probe");
    let wall32 = t0.elapsed().as_secs_f64();
    eprintln!(
        "[basis] 32-token probe wall {wall32:.2}s at {:.1}s",
        start.elapsed().as_secs_f64()
    );
    let by_time = (PROJECT_SECONDS / wall32 * 32.0) as usize;

    let mut ntok = (by_time / GROUP) * GROUP;
    while ntok > MIN_TOKENS && tape_bytes_at(&cfg, ntok) > TAPE_CAP_BYTES {
        ntok -= GROUP;
    }
    ntok = ntok.max(MIN_TOKENS).min(ids.len());
    let tokens: Vec<u32> = ids[..ntok]
        .iter()
        .map(|&v| u32::try_from(v).expect("token"))
        .collect();

    let t1 = Instant::now();
    let (_, tape) = forward(&weights, &cfg, &tokens).expect("forward");
    let mut grams = Grams::new(&cfg).expect("grams");
    grams.accumulate(&tape, &cfg).expect("accumulate");
    let wall_capture = t1.elapsed().as_secs_f64();
    eprintln!(
        "[basis] {ntok}-token capture {wall_capture:.2}s at {:.1}s",
        start.elapsed().as_secs_f64()
    );
    drop(tape);
    let lg = grams.finish(Normalisation::Mean).expect("finish grams");
    let t2 = Instant::now();

    let mut sep: Vec<(String, usize, usize, Row)> = Vec::new();
    let mut tile: Vec<(String, usize, usize, Row)> = Vec::new();
    for &l in &LAYERS {
        let g = &lg[l];
        let h_qkv = damped_gram(&g.qkv, e).expect("damped qkv");
        let h_o = damped_gram(&g.o, q_out).expect("damped o");
        let h_gu = damped_gram(&g.gate_up, e).expect("damped gate_up");
        let h_dn = damped_gram(&g.down, n_ff).expect("damped down");
        let lw = &weights.layers[l];
        let tensors: [(&str, &[i64], usize, usize, &[f64]); 7] = [
            ("q", &lw.wq, q_out, e, &h_qkv),
            ("k", &lw.wk, kv_out, e, &h_qkv),
            ("v", &lw.wv, kv_out, e, &h_qkv),
            ("o", &lw.wo, e, q_out, &h_o),
            ("gate", &lw.w_gate, n_ff, e, &h_gu),
            ("up", &lw.w_up, n_ff, e, &h_gu),
            ("down", &lw.w_down, e, n_ff, &h_dn),
        ];
        for (name, w, o, n, h) in tensors {
            let tm = Instant::now();
            let budget = BITS * o as f64 * n as f64;
            let (s, t) = measure(w, o, n, h, budget);
            eprintln!(
                "[basis] L{l}.{name} {o}x{n} {:.2}s at {:.1}s",
                tm.elapsed().as_secs_f64(),
                start.elapsed().as_secs_f64()
            );
            sep.push((format!("L{l}.{name}"), o, n, s));
            tile.push((format!("L{l}.{name}"), o, n, t));
        }
    }
    report("s: separable r_i c_j / m", &sep);
    report("t: 16x16 tile mean", &tile);
    println!(
        "\nbasis_rate_qwen05b: tokens = {ntok} (32-token probe {wall32:.2}s, projected {by_time}), \
         capture {wall_capture:.2}s, analysis {:.2}s, total {:.2}s",
        t2.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64()
    );
}
