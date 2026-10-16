//! `tests/learn_fold_steps.rs` — the fold-transfer refit steps: `fwht128` involution and
//! norm-preservation, `compose` shapes, `c_step` / `r_step` planted-parameter recovery, the `d_step`
//! (Hadamard-domain column scale) forward-model stationarity by central differences and descent, and
//! the shape / non-SPD refusals. Grade `tested`; nothing here is `proved`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the fixed-seed LCG maps its 53 high bits to an f64 in [-1, 1) and the shape means \
              divide by a usize count; the short `o/i/j/k` indices are the reference arithmetic's \
              vocabulary. The pedantic lints stay on for the rest of the crate."
)]

use apeiron::learn::fold::fit::steps::{
    c_step, compose, d_step, fwht128, had_gram, objective, r_step, StepsError, GROUP,
};
use apeiron::learn::ridge::solve_spd;

/// Out rows of the fixtures.
const O: usize = 4;
/// In columns (a whole number of 128-blocks).
const N: usize = 256;
/// Group count along `n`.
const G: usize = N / GROUP;

/// Deterministic 64-bit linear congruential generator (fixed seed, no `rand` crate).
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

    /// Uniform in `[0, 1)`, from the 53 high bits (exactly representable in `f64`).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }

    /// Uniform in `[0.2, 1.0)`.
    fn pos(&mut self) -> f64 {
        0.2 + 0.8 * self.unit()
    }
}

fn sym_vec(rng: &mut Lcg, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.sym()).collect()
}

fn pos_vec(rng: &mut Lcg, len: usize) -> Vec<f64> {
    (0..len).map(|_| rng.pos()).collect()
}

fn sign_vec(rng: &mut Lcg, len: usize) -> Vec<f64> {
    (0..len)
        .map(|_| if rng.sym() < 0.0 { -1.0 } else { 1.0 })
        .collect()
}

/// A symmetric positive-definite `n × n` metric: `(1/n)·A·Aᵀ + I` with `A` uniform in `[-1, 1)`.
fn spd_metric(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let a = sym_vec(rng, n * n);
    let mut hl = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += a[i * n + k] * a[j * n + k];
            }
            hl[i * n + j] = acc / (n as f64) + if i == j { 1.0 } else { 0.0 };
        }
    }
    hl
}

fn max_abs(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn fwht128_is_involution_and_norm_preserving() {
    let mut rng = Lcg::new(1);
    let x = sym_vec(&mut rng, O * N);
    let y = fwht128(&x).unwrap();
    let z = fwht128(&y).unwrap();
    assert_eq!(z.len(), x.len());
    let err = max_abs_diff(&x, &z);
    assert!(err < 1e-12, "involution err {err}");
    let nx: f64 = x.iter().map(|v| v * v).sum();
    let ny: f64 = y.iter().map(|v| v * v).sum();
    assert!((nx - ny).abs() <= 1e-12 * nx, "norm {nx} vs {ny}");
    // One block, and the constant vector (equals its own DC coefficient times √128).
    let xb = sym_vec(&mut rng, GROUP);
    let zb = fwht128(&fwht128(&xb).unwrap()).unwrap();
    assert!(max_abs_diff(&xb, &zb) < 1e-12, "block involution");
    let ones = vec![1.0f64; GROUP];
    let yo = fwht128(&ones).unwrap();
    assert!((yo[0] - (GROUP as f64).sqrt()).abs() < 1e-12);
    assert!(max_abs(&yo[1..]) < 1e-12);
}

#[test]
fn compose_shapes_and_objective() {
    let mut rng = Lcg::new(2);
    let s = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let c = pos_vec(&mut rng, N);
    let w = compose(&s, &r, &c, O, N).unwrap();
    assert_eq!(w.len(), O * N);
    for oo in 0..O {
        for i in 0..N {
            let want = s[oo * N + i] * r[oo * G + i / GROUP] * c[i];
            assert!((w[oo * N + i] - want).abs() < 1e-15);
        }
    }
    let hl = spd_metric(&mut rng, N);
    // J(W, W, Hl) = 0 and J is non-negative on a distinct pair.
    assert!(objective(&w, &w, &hl, O, N).unwrap().abs() < 1e-15);
    let other = sym_vec(&mut rng, O * N);
    assert!(objective(&w, &other, &hl, O, N).unwrap() > 0.0);
}

#[test]
fn c_step_recovers_c0() {
    let mut rng = Lcg::new(3);
    let s = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let c0 = pos_vec(&mut rng, N);
    let hl = spd_metric(&mut rng, N);
    let ws = compose(&s, &r, &c0, O, N).unwrap();
    let c = c_step(&ws, &hl, &s, &r, O, N, None).unwrap();
    // The un-ridged normal system has c0 as its exact solution; the 1e-12·mean(diag) ridge moves the
    // result off c0 by λ·(A+λI)⁻¹c0, far below this bound.
    let err = max_abs_diff(&c, &c0);
    assert!(err <= 1e-6 * (1.0 + max_abs(&c0)), "c recovery err {err}");
    // An all-ones row weight is the unweighted step.
    let ones = vec![1.0f64; O];
    let c_w = c_step(&ws, &hl, &s, &r, O, N, Some(&ones)).unwrap();
    assert!(max_abs_diff(&c, &c_w) < 1e-12);
}

#[test]
fn r_step_recovers_r0() {
    let mut rng = Lcg::new(4);
    let s = sign_vec(&mut rng, O * N);
    let c = pos_vec(&mut rng, N);
    let r0 = pos_vec(&mut rng, O * G);
    let hl = spd_metric(&mut rng, N);
    let ws = compose(&s, &r0, &c, O, N).unwrap();
    let r = r_step(&ws, &hl, &s, &c, O, N).unwrap();
    assert_eq!(r.len(), O * G);
    let err = max_abs_diff(&r, &r0);
    assert!(err <= 1e-6 * (1.0 + max_abs(&r0)), "r recovery err {err}");
}

/// The forward model `run_variant` uses for the Hadamard-domain column scale:
/// `recon(d) = compose(fwht128(S_rot ⊙ d), r, c)`.
fn recon_d(s_rot: &[f64], r: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let mut scaled = vec![0.0f64; O * N];
    for oo in 0..O {
        for i in 0..N {
            scaled[oo * N + i] = s_rot[oo * N + i] * d[i];
        }
    }
    let t = fwht128(&scaled).unwrap();
    compose(&t, r, c, O, N).unwrap()
}

#[test]
fn d_step_is_stationary_and_descends() {
    let mut rng = Lcg::new(5);
    let ws = sym_vec(&mut rng, O * N);
    let hl = spd_metric(&mut rng, N);
    let s_rot = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let c = pos_vec(&mut rng, N);
    let d = d_step(&ws, &hl, &s_rot, &r, &c, O, N, None).unwrap();
    assert_eq!(d.len(), N);
    let j_at = |dd: &[f64]| objective(&ws, &recon_d(&s_rot, &r, &c, dd), &hl, O, N).unwrap();
    let j_ones = j_at(&vec![1.0f64; N]);
    let j_d = j_at(&d);
    assert!(j_d <= j_ones, "descent {j_d} vs {j_ones}");
    // Central finite differences of the exactly-quadratic objective at the solution, scaled by
    // J(d=1). The step is the minimizer up to the ridge, so the gradient is ~2λ|d|.
    let h = 1e-4f64;
    let mut max_grad = 0.0f64;
    for k in 0..N {
        let mut dp = d.clone();
        let mut dm = d.clone();
        dp[k] += h;
        dm[k] -= h;
        let grad = (j_at(&dp) - j_at(&dm)) / (2.0 * h);
        max_grad = max_grad.max(grad.abs());
    }
    assert!(
        max_grad <= 1e-6 * j_ones,
        "max |dJ/dd_k| {max_grad} vs J_scale {j_ones}"
    );
}

/// The deleted `f64` `c_step` body, kept here as the reference: `Sh = S⊙r_rep`, `Sh_g = Sh⊙g_row`,
/// `A[i,j] = Hl[i,j]·Σ_o Sh[o,i]·Sh_g[o,j]`, the ridge `1e-12·mean(diag A)`, the right-hand side
/// `b[i] = Σ_o (Ws·Hl)[o,i]·Sh_g[o,i]` and `ridge::solve_spd`, all in `f64` with no grid boundary.
fn old_c_step(
    ws: &[f64],
    hl: &[f64],
    s: &[f64],
    r: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
) -> Vec<f64> {
    let g = n / GROUP;
    let mut b_sh = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            b_sh[oo * n + i] = s[oo * n + i] * r[oo * g + i / GROUP];
        }
    }
    let weight = |oo: usize| g_row.map_or(1.0f64, |gr| gr[oo]);
    let mut wshl = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            let mut acc = 0.0f64;
            for j in 0..n {
                acc += ws[oo * n + j] * hl[j * n + i];
            }
            wshl[oo * n + i] = acc;
        }
    }
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
    let ridge = 1e-12 * dmean;
    for i in 0..n {
        a[i * n + i] += ridge;
    }
    let mut rhs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for oo in 0..o {
            acc += wshl[oo * n + i] * b_sh[oo * n + i] * weight(oo);
        }
        rhs[i] = acc;
    }
    solve_spd(&a, &rhs, n, 1).unwrap()
}

/// `max_i |new_i − old_i| / max(|old_i|, 1)`, the mixed absolute/relative agreement the `chol` test
/// uses.
fn max_rel_diff(new: &[f64], old: &[f64]) -> f64 {
    new.iter()
        .zip(old)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs() / b.abs().max(1.0)))
}

/// (b) `c_step` (grid primitives) against the test-local old `f64` formula, relative `1e-7`, with a
/// positive `g_row`. The brief's stated shapes `(4,6),(16,40),(64,130)` are inadmissible: `c_step`
/// requires `n` a whole number of 128-blocks, so the nearest admissible shapes are used. The last
/// case draws `r ∈ [10,16)` so the panel crosses `8` and the `2⁻ᵏ` panel scaling is exercised
/// (`k = 1`), still meeting `1e-7` after the scaling is undone.
#[test]
fn c_step_matches_old_f64_formula() {
    let cases: [(usize, usize, bool); 4] = [
        (4, 128, false),
        (16, 256, false),
        (64, 384, false),
        (8, 128, true),
    ];
    for &(o, n, big_r) in &cases {
        let mut rng = Lcg::new(0x51_0000 + (o * n) as u64);
        let s = sign_vec(&mut rng, o * n);
        let r: Vec<f64> = if big_r {
            (0..o * (n / GROUP))
                .map(|_| 10.0 + 6.0 * rng.unit())
                .collect()
        } else {
            pos_vec(&mut rng, o * (n / GROUP))
        };
        let hl = spd_metric(&mut rng, n);
        let ws = sym_vec(&mut rng, o * n);
        let gr = pos_vec(&mut rng, o);
        let new = c_step(&ws, &hl, &s, &r, o, n, Some(&gr)).unwrap();
        let old = old_c_step(&ws, &hl, &s, &r, o, n, Some(&gr));
        let rel = max_rel_diff(&new, &old);
        println!("c_step vs old f64 ({o},{n}) big_r={big_r}: max rel {rel:e}");
        assert!(rel <= 1e-7, "({o},{n}) relative error {rel:e} > 1e-7");
    }
}

/// (c) A non-finite input and an out-of-bound input return the step's error, no panic: a `NaN`
/// metric entry, an infinite weight, a row weight at the `2³⁶` raw input bound, and a weight large
/// enough that the right-hand side leaves the grid `i64` range.
#[test]
fn c_step_refuses_non_finite_and_out_of_bound() {
    let mut rng = Lcg::new(12);
    let s = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let hl = spd_metric(&mut rng, N);
    let ws = sym_vec(&mut rng, O * N);
    let nan_hl = {
        let mut h = hl.clone();
        h[5] = f64::NAN;
        h
    };
    assert_eq!(
        c_step(&ws, &nan_hl, &s, &r, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
    let inf_ws = {
        let mut w = ws.clone();
        w[3] = f64::INFINITY;
        w
    };
    assert_eq!(
        c_step(&inf_ws, &hl, &s, &r, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
    // A row weight at the primitive input bound `2³⁶` raw = `16`.
    let mut gr_bad = pos_vec(&mut rng, O);
    gr_bad[O - 1] = 16.0;
    assert_eq!(
        c_step(&ws, &hl, &s, &r, O, N, Some(&gr_bad)).unwrap_err(),
        StepsError::ShapeMismatch
    );
    // A weight whose product leaves the grid `i64` range.
    let big_ws = vec![1.0e30f64; O * N];
    assert_eq!(
        c_step(&big_ws, &hl, &s, &r, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
}

/// (d) Timing at the production shape `o = 896`, `n = 4864`, three runs of `c_step` against the
/// previous profile's single `150.6 s` for the old `f64` body at the same shape (threads: `c_step`
/// takes none and its caller is single-threaded, so the primitives run at `8`; see
/// `C_STEP_THREADS`). Run with `--release -- --ignored --nocapture`.
#[test]
#[ignore = "timing campaign at o=896 n=4864: run with --release -- --ignored --nocapture"]
fn timing_c_step_production_shape() {
    let (o, n) = (896usize, 4864usize);
    let mut rng = Lcg::new(0x5eed_0f0f_a5a5_1234);
    let s = sign_vec(&mut rng, o * n);
    let r = pos_vec(&mut rng, o * (n / GROUP));
    let ws = sym_vec(&mut rng, o * n);
    let gr = pos_vec(&mut rng, o);
    // A well-conditioned SPD metric: the Kac–Murdock–Szegő Toeplitz `ρ^|i−j|` with `ρ = 0.5`
    // (eigenvalues `(1−ρ²)/(1−2ρ cos θ+ρ²) ∈ [1/3, 3]`). `o < n`, so the Gram `G = Shᵀ diag(g) Sh`
    // is rank-`o`; the decaying metric keeps `A = Hl ⊙ G` diagonally dominant — a metric the
    // `f64` body also solves.
    let mut pw = vec![1.0f64; n];
    for k in 1..n {
        pw[k] = pw[k - 1] * 0.5;
    }
    let mut hl = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            hl[i * n + j] = pw[i.abs_diff(j)];
        }
    }
    let mut times: Vec<f64> = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let c = c_step(&ws, &hl, &s, &r, o, n, Some(&gr)).expect("c_step");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(c);
        times.push(dt);
    }
    let mean = times.iter().sum::<f64>() / (times.len() as f64);
    let mn = times.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = times.iter().copied().fold(0.0f64, f64::max);
    println!(
        "c_step (grid primitives) o={o} n={n} threads=8: mean {mean:.3}s min {mn:.3}s max {mx:.3}s \
         runs {times:?}; old f64 body at this shape (single run): 150.6 s"
    );
}

#[test]
fn refusals_are_reached() {
    // `n` not a whole number of 128-blocks.
    assert_eq!(
        fwht128(&vec![0.0f64; 200]).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        had_gram(&vec![0.0f64; 200 * 200], 200).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        compose(&[], &[], &[], 0, 200).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        c_step(&[], &[], &[], &[], 0, 200, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        r_step(&[], &[], &[], &[], 0, 200).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        d_step(&[], &[], &[], &[], &[], 0, 200, None).unwrap_err(),
        StepsError::ShapeMismatch
    );

    let mut rng = Lcg::new(9);
    let s = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let c = pos_vec(&mut rng, N);
    let ws = sym_vec(&mut rng, O * N);
    let hl = spd_metric(&mut rng, N);
    // Length mismatches.
    assert_eq!(
        c_step(&ws[1..], &hl, &s, &r, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        r_step(&ws, &hl, &s[1..], &c, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        d_step(&ws, &hl, &s, &r, &c, O, N, Some(&[1.0, 1.0, 1.0][..])).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        objective(&ws, &ws[1..], &hl, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        compose(&s, &r[1..], &c, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    // A non-SPD metric: the last diagonal entry is negative, so `A = hl ⊙ Gram` is indefinite and the
    // Cholesky pivot refuses.
    let mut hl_bad = vec![0.0f64; N * N];
    for i in 0..N {
        hl_bad[i * N + i] = 1.0;
    }
    hl_bad[(N - 1) * N + (N - 1)] = -1.0;
    assert_eq!(
        c_step(&ws, &hl_bad, &s, &r, O, N, None).unwrap_err(),
        StepsError::NotPositiveDefinite
    );
}
