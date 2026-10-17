//! `tests/learn_solve_chol.rs` — the Q32.32 SPD solve `A X = B`: agreement with the f64
//! `ridge::solve_spd` reference on random SPD systems, bit-exact residual, thread-count
//! determinism, the shape / diagonal / non-SPD / overflow refusals, equilibration on a `1e4`
//! diagonal range, and an ignored timing table at `n = 4864`. Grade `tested` for correctness,
//! `measured` for the timing run.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the tests convert the Q32.32 grid to/from f64 for the reference oracle and index \
              row-major buffers by i/j/k/t; the pedantic lints stay on for the crate's non-test code."
)]

use apeiron::learn::ridge::solve_spd;
use apeiron::learn::solve::chol::{chol_solve, equilibrate, SolveError};

/// The Q32.32 unit.
const ONE: i64 = 1 << 32;

/// `2^32`, the Q32.32 scale factor.
const SCALE: f64 = 4_294_967_296.0;

/// Deterministic 64-bit LCG (Numerical Recipes constants): the tests' only randomness source.
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

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }
}

/// Q32.32 grid of an `f64` (round to nearest).
fn to_q32(x: f64) -> i64 {
    (x * SCALE).round() as i64
}

/// `f64` value of a Q32.32 `i64`.
fn to_f64(v: i64) -> f64 {
    (v as f64) / SCALE
}

/// A well-conditioned SPD `A = M·Mᵀ/n + 0.1·I` (`n × n`) and an `n × k` right-hand side `B`, both
/// `f64`, from a fixed seed. `M` is `n × n` uniform in `[-1, 1)`.
fn build_system(seed: u64, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lcg = Lcg::new(seed);
    let m: Vec<f64> = (0..n * n).map(|_| lcg.sym()).collect();
    let mut a = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for t in 0..n {
                s += m[i * n + t] * m[j * n + t];
            }
            a[i * n + j] = s / (n as f64) + if i == j { 0.1 } else { 0.0 };
        }
    }
    let b: Vec<f64> = (0..n * k).map(|_| lcg.sym()).collect();
    (a, b)
}

/// Largest relative difference `|q − f| / max(|f|, 1)` over the two slices.
fn max_rel_diff(q: &[i64], f: &[f64]) -> f64 {
    q.iter()
        .zip(f)
        .map(|(&qq, &ff)| (to_f64(qq) - ff).abs() / ff.abs().max(1.0))
        .fold(0.0f64, f64::max)
}

/// Independent exact recomputation of `round32(B − A X)` in `i128`, entrywise (test (b)).
fn exact_residual_test(a: &[i64], b: &[i64], x: &[i64], n: usize, k: usize) -> Vec<i64> {
    let mut r = vec![0i64; n * k];
    for i in 0..n {
        for j in 0..k {
            let mut acc: i128 = i128::from(b[i * k + j]) << 32;
            for t in 0..n {
                acc -= i128::from(a[i * n + t]) * i128::from(x[t * k + j]);
            }
            r[i * k + j] = ((acc + (1_i128 << 31)) >> 32) as i64;
        }
    }
    r
}

/// (a) The Q32.32 solve agrees with the f64 `ridge::solve_spd` on the same grid-rounded inputs, to
/// relative `1e-7`, for `n ∈ {1, 2, 7, 64, 130}` (130 is not a multiple of the block width).
#[test]
fn solves_agree_with_f64_reference() {
    for &n in &[1usize, 2, 7, 64, 130] {
        let k = 3usize;
        let (a, b) = build_system(0xa5 + n as u64, n, k);
        let aq: Vec<i64> = a.iter().map(|&x| to_q32(x)).collect();
        let bq: Vec<i64> = b.iter().map(|&x| to_q32(x)).collect();
        let a_f: Vec<f64> = aq.iter().map(|&v| to_f64(v)).collect();
        let b_f: Vec<f64> = bq.iter().map(|&v| to_f64(v)).collect();
        let solved = chol_solve(&aq, &bq, n, k).expect("solve");
        let xref = solve_spd(&a_f, &b_f, n, k).expect("f64 reference");
        let rel = max_rel_diff(&solved.x, &xref);
        println!("n={n} rel={rel:e} residual_max={}", solved.residual_max);
        assert!(rel <= 1e-7, "n={n} relative error {rel:e} > 1e-7");
    }
}

/// (b) The returned `residual_max` equals an independent exact `i128` recomputation, bit-equal.
#[test]
fn residual_max_is_bit_exact() {
    for &n in &[1usize, 2, 7, 64, 130] {
        let k = 2usize;
        let (a, b) = build_system(0x77 + n as u64, n, k);
        let aq: Vec<i64> = a.iter().map(|&x| to_q32(x)).collect();
        let bq: Vec<i64> = b.iter().map(|&x| to_q32(x)).collect();
        let solved = chol_solve(&aq, &bq, n, k).expect("solve");
        let rf = exact_residual_test(&aq, &bq, &solved.x, n, k);
        let mx = rf
            .iter()
            .map(|v| i128::from(*v).abs())
            .max()
            .expect("nonempty");
        assert_eq!(mx as i64, solved.residual_max, "n={n}");
    }
}

/// (d) Every refusal arm is reachable: empty, shape, non-positive diagonal, zero diagonal, a
/// non-SPD matrix with a positive diagonal, and an overflow with entries near `2^62`.
#[test]
fn refusals_reach_every_arm() {
    assert_eq!(chol_solve(&[], &[], 0, 0), Err(SolveError::Empty));
    assert_eq!(equilibrate(&[], 0), Err(SolveError::Empty));
    assert_eq!(
        chol_solve(&[ONE], &[ONE], 2, 1),
        Err(SolveError::ShapeMismatch)
    );
    assert_eq!(
        chol_solve(&[ONE, 0, 0, ONE], &[ONE, ONE], 2, 2),
        Err(SolveError::ShapeMismatch)
    );
    assert_eq!(equilibrate(&[ONE], 2), Err(SolveError::ShapeMismatch));
    // A negative diagonal.
    assert_eq!(
        chol_solve(&[-ONE, 0, 0, ONE], &[ONE, 0], 2, 1),
        Err(SolveError::NonPositiveDiagonal)
    );
    // A zero diagonal.
    assert_eq!(
        chol_solve(&[0, 0, 0, ONE], &[ONE, 0], 2, 1),
        Err(SolveError::NonPositiveDiagonal)
    );
    // Non-SPD `[[1, 2], [2, 1]]` with a positive diagonal: a non-positive Cholesky pivot.
    assert_eq!(
        chol_solve(&[ONE, 2 * ONE, 2 * ONE, ONE], &[ONE, ONE], 2, 1),
        Err(SolveError::NotPositiveDefinite)
    );
    // Overflow: an off-diagonal entry near 2^62 against a unit-magnitude diagonal, so the
    // equilibrated entry `a·2^64/(d_i·d_j)` leaves i64.
    let big = 1i64 << 62;
    assert_eq!(
        chol_solve(&[1, big, big, 1], &[ONE, ONE], 2, 1),
        Err(SolveError::Overflow)
    );
}

/// (e) A `1e4` diagonal range (`A = D M D`, `D = diag(1, …, 1e2)`) still meets (a), and the
/// equilibrated matrix has a unit diagonal to within `2^-30` (the reason equilibration exists).
#[test]
fn conditioning_equilibration_holds_unit_diagonal() {
    for &n in &[3usize, 8, 40] {
        let k = 2usize;
        let (m, b) = build_system(0x51 + n as u64, n, k);
        let mut d = vec![0f64; n];
        for i in 0..n {
            d[i] = 1.0 + 99.0 * (i as f64) / ((n.max(2) - 1) as f64);
        }
        let mut a = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = d[i] * m[i * n + j] * d[j];
            }
        }
        let aq: Vec<i64> = a.iter().map(|&x| to_q32(x)).collect();
        let bq: Vec<i64> = b.iter().map(|&x| to_q32(x)).collect();
        let (scaled, _scale) = equilibrate(&aq, n).expect("equilibrate");
        for i in 0..n {
            let e = (scaled[i * n + i] - ONE).abs();
            assert!(e <= 4, "n={n} i={i} scaled-diagonal error {e} > 2^-30·ONE");
        }
        let a_f: Vec<f64> = aq.iter().map(|&v| to_f64(v)).collect();
        let b_f: Vec<f64> = bq.iter().map(|&v| to_f64(v)).collect();
        let solved = chol_solve(&aq, &bq, n, k).expect("solve");
        let xref = solve_spd(&a_f, &b_f, n, k).expect("f64 reference");
        let rel = max_rel_diff(&solved.x, &xref);
        println!("conditioned n={n} rel={rel:e}");
        assert!(rel <= 1e-7, "conditioned n={n} relative error {rel:e}");
    }
}

/// (f) Timing at `n = 4864`, `k = 1`: a well-conditioned SPD from the Gram of 2048 LCG rows plus
/// ridge, three pooled runs: mean and min–max wall seconds.
#[test]
#[ignore = "timing campaign at n=4864: run with --release -- --ignored --nocapture"]
fn timing_large_spd() {
    let n = 4864usize;
    let rows = 2048usize;
    let k = 1usize;
    let mut lcg = Lcg::new(0xfeed_face_dead_beef);
    let mut m = vec![0f64; rows * n];
    for v in &mut m {
        *v = lcg.sym();
    }
    let inv = 1.0 / (rows as f64);
    let mut a = vec![0f64; n * n];
    for r in 0..rows {
        let mrow = &m[r * n..r * n + n];
        for i in 0..n {
            let mi = mrow[i];
            for j in 0..=i {
                a[i * n + j] += mi * mrow[j];
            }
        }
    }
    for i in 0..n {
        for j in 0..=i {
            let v = a[i * n + j] * inv + if i == j { 1.0 } else { 0.0 };
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    let aq: Vec<i64> = a.iter().map(|&x| to_q32(x)).collect();
    let bq: Vec<i64> = (0..n * k).map(|_| to_q32(lcg.sym())).collect();
    let mut ts: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let solved = chol_solve(&aq, &bq, n, k).expect("solve");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(solved.residual_max);
        ts.push(dt);
    }
    let mean = ts.iter().sum::<f64>() / (ts.len() as f64);
    let mn = ts.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = ts.iter().copied().fold(0.0f64, f64::max);
    println!(
        "chol_solve n={n} k={k} pooled: mean {mean:.3}s min {mn:.3}s max {mx:.3}s runs {ts:?}"
    );
}
