//! `tests/learn_fold_steps.rs` — the fold-transfer refit steps: `fwht128` involution and
//! norm-preservation, `compose` shapes, `c_step` / `r_step` planted-parameter recovery, the `d_step`
//! (Hadamard-domain column scale) forward-model stationarity by central differences and descent, the
//! agreement of `c_step`, `d_step` and `r_step` with their deleted `f64` bodies, `mul_metric`'s
//! `Q32.32` scale rule, and the shape / non-SPD / non-finite refusals. Grade `tested`; nothing here
//! is `proved`.
//!
//! Mutation controls: flipping the `h_rot`/`hl` factor application in `c_step` or dropping it in
//! `d_step` turns `c_step_matches_old_f64_formula` / `d_step_matches_old_f64_formula` red; dropping
//! the exact undo power of two in `mul_metric` (`from_grid(v) * back` → `from_grid(v)`) turns
//! `mul_metric_matches_old_f64_formula` red at the `1e3` magnitude, and
//! `c_step_matches_old_f64_formula` / `d_step_matches_old_f64_formula` / `r_step_matches_old_f64_formula`
//! red at their `|Ws| ≈ 1e3` case.

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
    c_step, c_step_wshl, compose, d_step, d_step_wshl, fwht128, had_gram, mul_metric, objective,
    r_step, r_step_wshl, StepsError, GROUP,
};
use apeiron::learn::ridge::solve_spd;
use apeiron::learn::solve::chol::chol_solve;
use apeiron::learn::solve::q32;
use apeiron::learn::solve::syrk::{hadamard_into, wsyrk};

/// Out rows of the fixtures.
const O: usize = 4;
/// In columns (a whole number of 128-blocks).
const N: usize = 256;
/// Group count along `n`.
const G: usize = N / GROUP;

/// The run count of the `#[ignore]` timing campaign: `3`, or the value of `Q_MM_STEP_RUNS` when set
/// (the `580 s` gate budget decides; the brief allows `2` and requires it be said).
fn step_runs() -> usize {
    std::env::var("Q_MM_STEP_RUNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1)
}

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

/// The `fwht128` round-trip bound, derived. Each pass rounds the input to a raw word (`≤2⁻³³`
/// absolute) and each output entry once after the exact butterfly (`≤2⁻³³` of `|Hx|`); in the
/// `2`-norm that is `‖e‖₂ ≤ 2⁻³³(‖x‖₂ + 2√n) ≈ 1.0e-8` at `n = O·N = 1024` and `|x| < 1`, and the
/// orthonormal transform carries it to the second pass unchanged. The DC of the all-ones block is
/// `128·c` with `c = round(2³²/√128)/2³²`, i.e. `√128 ± 128·2⁻³³ ≈ √128 ± 1.5e-8`; its other
/// coefficients are exact zeros before the scaling, so they stay exactly `0`; the norm is preserved
/// to `2‖x‖₂‖e‖₂` absolute, `≈1e-9` relative.
#[test]
fn fwht128_is_involution_and_norm_preserving() {
    let mut rng = Lcg::new(1);
    let x = sym_vec(&mut rng, O * N);
    let y = fwht128(&x).unwrap();
    let z = fwht128(&y).unwrap();
    assert_eq!(z.len(), x.len());
    let err = max_abs_diff(&x, &z);
    println!("fwht128 involution: max |x−z| {err:e}");
    assert!(err < 2e-8, "involution err {err} above the 1e-8 grid bound");
    let nx: f64 = x.iter().map(|v| v * v).sum();
    let ny: f64 = y.iter().map(|v| v * v).sum();
    assert!((nx - ny).abs() <= 1e-8 * nx, "norm {nx} vs {ny}");
    // One block, and the constant vector (equals its own DC coefficient times √128).
    let xb = sym_vec(&mut rng, GROUP);
    let zb = fwht128(&fwht128(&xb).unwrap()).unwrap();
    assert!(max_abs_diff(&xb, &zb) < 2e-8, "block involution");
    let ones = vec![1.0f64; GROUP];
    let yo = fwht128(&ones).unwrap();
    assert!((yo[0] - (GROUP as f64).sqrt()).abs() < 1e-7);
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
            // Three operands each rounded to a raw word (`≤2⁻³³`) and two rounded products:
            // `|err| ≤ 2⁻³³(1 + |s·r| + |s·r·c|) ≤ 5e-10` for `|s|,|r|,|c| < 1`.
            assert!((w[oo * N + i] - want).abs() < 1e-9);
        }
    }
    let hl = spd_metric(&mut rng, N);
    // `J(W, W, Hl)` is the zero word difference, so exactly `0`; `J` is positive on a distinct pair.
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

/// The deleted `f64` 128-block transform: the butterfly per block and one `1/√128` factor per
/// entry, no grid boundary. The oracle for [`old_d_step`], so that body carries no grid arithmetic.
fn ref_fwht(x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for chunk in y.chunks_mut(GROUP) {
        let mut h = 1;
        while h < GROUP {
            let mut i = 0;
            while i < GROUP {
                for j in i..i + h {
                    let (a, b) = (chunk[j], chunk[j + h]);
                    chunk[j] = a + b;
                    chunk[j + h] = a - b;
                }
                i += 2 * h;
            }
            h *= 2;
        }
        for v in chunk.iter_mut() {
            *v /= (GROUP as f64).sqrt();
        }
    }
    y
}

/// The transpose of a row-major `rows × cols` `f64` matrix.
fn ref_transpose(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

/// The deleted `f64` `had_gram(H) = fwht128(fwht128(H).T).T`, no grid boundary.
fn ref_had_gram(h: &[f64], n: usize) -> Vec<f64> {
    let t = ref_transpose(&ref_fwht(h), n, n);
    ref_transpose(&ref_fwht(&t), n, n)
}

/// The deleted `f64` `mul_metric` body, kept here as the reference: `Ws·Hl`, `O × n` row-major,
/// one scalar accumulator per entry with no grid boundary and no scaling.
fn old_mul_metric(ws: &[f64], hl: &[f64], o: usize, n: usize) -> Vec<f64> {
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
    let wshl = old_mul_metric(ws, hl, o, n);
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

/// The deleted `f64` `d_step` body, kept here as the reference: `G_rot = fwht128((Ws·Hl)⊙c)`,
/// `H_rot = had_gram(Hl ⊙ c cᵀ)`, `B = S_rot⊙r_rep`, `A[i,j] = H_rot[i,j]·Σ_o B[o,i]·B[o,j]·g_row[o]`,
/// the ridge `1e-12·max(mean(diag A), 1e-12)`, `b[i] = Σ_o B[o,i]·g_row[o]·G_rot[o,i]` and
/// `ridge::solve_spd`, all in `f64`, with no grid boundary and no panel scaling.
#[allow(
    clippy::too_many_arguments,
    reason = "the reference body is handed exactly the eight arguments of the public `d_step` it \
              stands in for; bundling them would not be the same call."
)]
fn old_d_step(
    ws: &[f64],
    hl: &[f64],
    s_rot: &[f64],
    r: &[f64],
    c: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
) -> Vec<f64> {
    let g = n / GROUP;
    let weight = |oo: usize| g_row.map_or(1.0f64, |gr| gr[oo]);
    let wshl = old_mul_metric(ws, hl, o, n);
    let mut g_rot = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            g_rot[oo * n + i] = wshl[oo * n + i] * c[i];
        }
    }
    let g_rot = ref_fwht(&g_rot);
    let mut hlc = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            hlc[i * n + j] = hl[i * n + j] * c[i] * c[j];
        }
    }
    let h_rot = ref_had_gram(&hlc, n);
    let mut b_scaled = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            b_scaled[oo * n + i] = s_rot[oo * n + i] * r[oo * g + i / GROUP];
        }
    }
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
    let ridge = 1e-12 * dmean.max(1e-12);
    for i in 0..n {
        a[i * n + i] += ridge;
    }
    let mut rhs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for oo in 0..o {
            acc += b_scaled[oo * n + i] * weight(oo) * g_rot[oo * n + i];
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

/// The deleted `f64` `r_step` body, kept here as the reference: per out row `P[o,g,k] = S·c`,
/// `K[o,g,m] = Σ_k P·Hl`, `M[o,g,g'] = Σ_k K·P`, `t[o,g] = Σ_k P·(Ws·Hl)`, the ridge
/// `1e-12·mean(diag M[o])` and `ridge::solve_spd`, all in `f64` with no grid boundary and no
/// scaling.
fn old_r_step(ws: &[f64], hl: &[f64], s: &[f64], c: &[f64], o: usize, n: usize) -> Vec<f64> {
    let g = n / GROUP;
    let wshl = old_mul_metric(ws, hl, o, n);
    let mut out = vec![0.0f64; o * g];
    let mut p1 = vec![0.0f64; GROUP];
    let mut kvec = vec![0.0f64; n];
    let mut m = vec![0.0f64; g * g];
    let mut t = vec![0.0f64; g];
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
        let ridge = 1e-12 * dmean;
        for gg in 0..g {
            m[gg * g + gg] += ridge;
        }
        let sol = solve_spd(&m, &t, g, 1).unwrap();
        for gg in 0..g {
            out[oo * g + gg] = sol[gg];
        }
    }
    out
}

/// (b) `c_step` (grid primitives) against the test-local old `f64` formula, relative `1e-7`, with a
/// positive `g_row`. The brief's stated shapes `(4,6),(16,40),(64,130)` are inadmissible: `c_step`
/// requires `n` a whole number of 128-blocks, so the nearest admissible shapes are used. The last
/// case draws `r ∈ [10,16)` so the panel crosses `8` and the `2⁻ᵏ` panel scaling is exercised
/// (`k = 1`), still meeting `1e-7` after the scaling is undone; the fifth draws `|Ws| ≈ 1e2`, above
/// `mul_metric`'s `gemm_nn` input bound, so the operand scale rule and its exact undo are exercised
/// inside the step (dropping the undo turns this case red; `1e2` is chosen over `1e3` because the
/// grid quantum propagates to the solved output — `mul_metric_matches_old_f64_formula` states that
/// bound at `1e3`).
///
/// The agreement bound, derived: every operand enters as a raw word rounded at half a unit
/// (`2⁻³³` in the reals of its scaled space) and each product rounds once, so the relative input
/// perturbation is `≈2⁻³²`; the solved system is `κ`-conditioned, and these fixtures'
/// `spd_metric = AAᵀ/n + I` has eigenvalues in `[1, O(1)]` with `A = Hl ⊙ Gram` diagonally
/// dominant, so the propagated relative error is `≲ n·2⁻³³·κ ≈ 1e-7` at `n ≤ 384`. The observed
/// maximum is printed and stays below `5.4e-8`.
#[test]
fn c_step_matches_old_f64_formula() {
    let cases: [(usize, usize, bool, f64); 5] = [
        (4, 128, false, 1.0),
        (16, 256, false, 1.0),
        (64, 384, false, 1.0),
        (8, 128, true, 1.0),
        (8, 128, false, 1.0e2),
    ];
    for &(o, n, big_r, ws_mag) in &cases {
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
        let ws: Vec<f64> = (0..o * n).map(|_| ws_mag * rng.sym()).collect();
        let gr = pos_vec(&mut rng, o);
        let new = c_step(&ws, &hl, &s, &r, o, n, Some(&gr)).unwrap();
        let old = old_c_step(&ws, &hl, &s, &r, o, n, Some(&gr));
        let rel = max_rel_diff(&new, &old);
        println!("c_step vs old f64 ({o},{n}) big_r={big_r} |Ws|~{ws_mag:e}: max rel {rel:e}");
        assert!(
            rel <= 1e-7,
            "({o},{n}) |Ws|~{ws_mag:e} relative error {rel:e} > 1e-7"
        );
    }
}

/// (c) A non-finite input and an out-of-bound input return the step's error, no panic: a `NaN`
/// metric entry, an infinite weight, a non-finite row weight (the row weight itself is shifted into
/// the input bound, so only finiteness refuses), and a weight large enough that the right-hand side
/// leaves the grid `i64` range.
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
    // A non-finite row weight.
    let mut gr_bad = pos_vec(&mut rng, O);
    gr_bad[O - 1] = f64::INFINITY;
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

/// A `g_row` magnitude above the `wsyrk` input bound (`2³⁶` raw = `16`) is shifted into it by an
/// exact power of two, as the panel is, instead of refused: `c_step` and `d_step` accept `18.475`,
/// `1e3` and the former boundary `16.0`, and agree with the same call whose `g_row` is prescaled by
/// that power of two (`16` was the first value the unshifted boundary refused). The two calls round
/// `g`'s raw word independently, so they can differ by one unit (`2⁻³²` relative); on these systems
/// that propagates to `≤1e-8` relative in the solution. Dropping the row-weight factor from the
/// right-hand-side `weight` (or the `g_scale` clamp in `d_step`) turns this red.
#[test]
fn large_g_row_is_shifted_not_refused() {
    let mut rng = Lcg::new(0x6_9000_0001);
    let s = sign_vec(&mut rng, O * N);
    let s_rot = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let c = pos_vec(&mut rng, N);
    let hl = spd_metric(&mut rng, N);
    let ws = sym_vec(&mut rng, O * N);
    let base = pos_vec(&mut rng, O);
    for (big, down) in [(18.475f64, 4.0f64), (1.0e3, 128.0), (16.0, 2.0)] {
        let mut gr = base.clone();
        gr[0] = big;
        let gr_down: Vec<f64> = gr.iter().map(|&v| v / down).collect();
        let got = c_step(&ws, &hl, &s, &r, O, N, Some(&gr)).expect("shifted c_step");
        let want = c_step(&ws, &hl, &s, &r, O, N, Some(&gr_down)).expect("prescaled c_step");
        let rel = max_rel_diff(&got, &want);
        println!("c_step g_row {big} vs /{down}: max rel {rel:e}");
        assert!(rel <= 1e-8, "c_step g_row {big} vs /{down}: {rel:e}");
        let got_d = d_step(&ws, &hl, &s_rot, &r, &c, O, N, Some(&gr)).expect("shifted d_step");
        let want_d =
            d_step(&ws, &hl, &s_rot, &r, &c, O, N, Some(&gr_down)).expect("prescaled d_step");
        let rel_d = max_rel_diff(&got_d, &want_d);
        println!("d_step g_row {big} vs /{down}: max rel {rel_d:e}");
        assert!(rel_d <= 1e-8, "d_step g_row {big} vs /{down}: {rel_d:e}");
    }
}

/// (d) Timing at the production shape `o = 896`, `n = 4864`, three runs of `c_step` against the
/// previous profile's single `150.6 s` for the old `f64` body at the same shape (the primitives
/// run on the shared engine pool at `width_global()`; the output is invariant to that width). Run
/// with `--release -- --ignored --nocapture`.
#[test]
#[ignore = "timing campaign at o=896 n=4864: run with --release -- --ignored --nocapture"]
#[allow(
    clippy::too_many_lines,
    reason = "the timing campaign is one measurement unit: one fixture, one stage split and the \
              repeated runs have to report together or the split cannot be attributed."
)]
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
    // Stage split first, so a budget that only covers the split still reports where the time goes:
    // the shared primitives are timed directly on the step's own intermediates.
    let st = c_step_stages(&ws, &hl, &s, &r, &gr, o, n);
    let sum = st.iter().sum::<f64>();
    println!(
        "c_step stage split o={o} n={n} threads=8, once each: mul_metric(scale+gemm_nn) {:.3}s \
         boundary-conversion(f64→Q32.32 maps) {:.3}s wsyrk {:.3}s hadamard_into {:.3}s chol_solve \
         {:.3}s sum {sum:.3}s",
        st[0], st[1], st[2], st[3], st[4]
    );
    let runs = step_runs();
    let mut times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
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
        "c_step (mul_metric over gemm_nn) o={o} n={n} threads=8 runs={runs}: mean {mean:.3}s min \
         {mn:.3}s max {mx:.3}s runs {times:?}; stage sum {sum:.3}s, remainder (group_scale, panel \
         scan, ridge, right-hand side, read-back) {:.3}s; previous measurement of this step with the \
         scalar f64 mul_metric: mean 121.1 s (runs 120.904/121.263/120.982)",
        mean - sum
    );
}

/// The `c_step` body stage by stage on the timing fixture, each stage timed once: `[mul_metric,
/// boundary conversion, wsyrk, hadamard_into, chol_solve]`. The intermediates are `c_step`'s own
/// (`Sh = S⊙r_rep`, its `2⁻ᵏ` panel scale, the three f64→`Q32.32` maps, the ridge and the
/// right-hand side); `group_scale`, the panel scan, the ridge, the right-hand side and the
/// read-back are outside the timed regions and are the remainder reported against `c_step`'s mean.
fn c_step_stages(
    ws: &[f64],
    hl: &[f64],
    s: &[f64],
    r: &[f64],
    gr: &[f64],
    o: usize,
    n: usize,
) -> [f64; 5] {
    let g = n / GROUP;
    let mut b_sh = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            b_sh[oo * n + i] = s[oo * n + i] * r[oo * g + i / GROUP];
        }
    }
    let t0 = std::time::Instant::now();
    let wshl = mul_metric(ws, hl, o, n).expect("mul_metric");
    let staged_mul = t0.elapsed().as_secs_f64();
    let k = q32::panel_shift(max_abs(&b_sh));
    if k != 0 {
        let f = 2f64.powi(-k);
        for v in &mut b_sh {
            *v *= f;
        }
    }
    let t0 = std::time::Instant::now();
    let sh_grid: Vec<i64> = b_sh
        .iter()
        .map(|&x| q32::to_grid(x, q32::INPUT_BOUND).expect("Sh grid"))
        .collect();
    let gr_grid: Vec<i64> = gr
        .iter()
        .map(|&x| q32::to_grid(x, q32::INPUT_BOUND).expect("g_row grid"))
        .collect();
    let hl_grid: Vec<i64> = hl
        .iter()
        .map(|&x| q32::to_grid(x, q32::GRID_BOUND).expect("Hl grid"))
        .collect();
    let conv = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let mut a = wsyrk(&sh_grid, Some(&gr_grid), o, n).expect("wsyrk");
    let syrk = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    hadamard_into(&mut a, &hl_grid).expect("hadamard_into");
    let had = t0.elapsed().as_secs_f64();
    let mut dmean = 0.0f64;
    for i in 0..n {
        dmean += q32::from_grid(a[i * n + i]);
    }
    dmean /= n as f64;
    let mut ridge_raw = (1e-12 * dmean * q32::GRID_SCALE).round() as i64;
    if ridge_raw == 0 {
        ridge_raw = 1;
    }
    for i in 0..n {
        a[i * n + i] += ridge_raw;
    }
    let mut rhs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for oo in 0..o {
            acc += wshl[oo * n + i] * b_sh[oo * n + i] * gr[oo];
        }
        rhs[i] = acc;
    }
    let b_grid: Vec<i64> = rhs
        .iter()
        .map(|&x| q32::to_grid(x, q32::GRID_BOUND).expect("b grid"))
        .collect();
    let t0 = std::time::Instant::now();
    let solved = chol_solve(&a, &b_grid, n, 1).expect("chol_solve");
    let chol = t0.elapsed().as_secs_f64();
    std::hint::black_box(&solved);
    [staged_mul, conv, syrk, had, chol]
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

/// (b) `d_step` (grid primitives) against the test-local old `f64` formula, relative `1e-7`. Every
/// shape here is admissible to `d_step` (`n` a whole number of 128-blocks): the brief's `(4,128)`,
/// `(16,256)`, `(64,384)`, plus `(8,128)` with `r ∈ [10,16)` so the panel crosses `8` and the `2⁻ᵏ`
/// panel scaling is exercised (`k = 1`), still meeting `1e-7` after the scaling is undone, plus
/// `(8,128)` with `|Ws| ≈ 1e2` so `mul_metric`'s operand scale rule and its exact undo are
/// exercised inside the step (dropping the undo turns that case red).
///
/// The agreement bound is `c_step_matches_old_f64_formula`'s: the same `2⁻³²` input quantum and
/// the same conditioning of `A = H_rot ⊙ Gram`, with the rotated metric adding one transform that
/// is orthonormal up to `2⁻³³`; observed maximum below `5.4e-8`.
///
/// Mutation control: dropping the `H_rot`/`hadamard_into` rotation factor in `d_step` (passing the
/// weighted Gram as `A`) turns this test red at the first shape.
#[test]
fn d_step_matches_old_f64_formula() {
    let cases: [(usize, usize, bool, f64); 5] = [
        (4, 128, false, 1.0),
        (16, 256, false, 1.0),
        (64, 384, false, 1.0),
        (8, 128, true, 1.0),
        (8, 128, false, 1.0e2),
    ];
    for &(o, n, big_r, ws_mag) in &cases {
        let mut rng = Lcg::new(0xd1_0000 + (o * n) as u64);
        let ws: Vec<f64> = (0..o * n).map(|_| ws_mag * rng.sym()).collect();
        let hl = spd_metric(&mut rng, n);
        let s_rot = sign_vec(&mut rng, o * n);
        let r: Vec<f64> = if big_r {
            (0..o * (n / GROUP))
                .map(|_| 10.0 + 6.0 * rng.unit())
                .collect()
        } else {
            pos_vec(&mut rng, o * (n / GROUP))
        };
        let c = pos_vec(&mut rng, n);
        let gr = pos_vec(&mut rng, o);
        let new = d_step(&ws, &hl, &s_rot, &r, &c, o, n, Some(&gr)).unwrap();
        let old = old_d_step(&ws, &hl, &s_rot, &r, &c, o, n, Some(&gr));
        let rel = max_rel_diff(&new, &old);
        println!("d_step vs old f64 ({o},{n}) big_r={big_r} |Ws|~{ws_mag:e}: max rel {rel:e}");
        assert!(
            rel <= 1e-7,
            "({o},{n}) |Ws|~{ws_mag:e} relative error {rel:e} > 1e-7"
        );
    }
}

/// (b) `r_step` (which forms `Ws·Hl` through `mul_metric` and solves one `G × G` system per out
/// row) against the test-local old `f64` formula, relative `1e-7`, at the shapes the other step
/// comparisons use. The last case draws `|Ws| ≈ 1e3`, so `mul_metric`'s operand scale rule and its
/// exact undo are exercised inside `r_step`.
///
/// Mutation control: dropping the exact undo power of two in `mul_metric` turns the `|Ws| ≈ 1e3`
/// case red.
#[test]
fn r_step_matches_old_f64_formula() {
    let cases: [(usize, usize, f64); 4] = [
        (4, 128, 1.0),
        (16, 256, 1.0),
        (64, 384, 1.0),
        (8, 128, 1.0e3),
    ];
    for &(o, n, ws_mag) in &cases {
        let mut rng = Lcg::new(0x71_0000 + (o * n) as u64);
        let s = sign_vec(&mut rng, o * n);
        let c = pos_vec(&mut rng, n);
        let hl = spd_metric(&mut rng, n);
        let ws: Vec<f64> = (0..o * n).map(|_| ws_mag * rng.sym()).collect();
        let new = r_step(&ws, &hl, &s, &c, o, n).unwrap();
        let old = old_r_step(&ws, &hl, &s, &c, o, n);
        let rel = max_rel_diff(&new, &old);
        println!("r_step vs old f64 ({o},{n}) |Ws|~{ws_mag:e}: max rel {rel:e}");
        assert!(
            rel <= 1e-7,
            "({o},{n}) |Ws|~{ws_mag:e} relative error {rel:e} > 1e-7"
        );
    }
}

/// (b) `mul_metric` through its public path against the old `f64` loop at `(4,128)` and `(16,256)`
/// with `Ws` entries of magnitude `1e-3` and `1e3` — the scale rule. `1e3` is above `gemm_nn`'s
/// documented input bound `|v| < 2³⁶` raw (`16` in the reals), so [`panel_shift`] down-shifts the
/// panel by `2⁻⁷`; `1e-3` is already within `8` and is left unscaled. The product is read back with
/// the exact undo power of two, and the assertions are the two the rule makes:
///
/// 1. `panel_shift` of the pinned largest magnitude is `7` at `1e3` and `0` at `1e-3`, so the
///    scaled panel is at most `8` (raw `2³⁵`, inside the bound with a factor of two of head-room);
/// 2. the result is inside the boundary's propagated quantization bound. Each operand entry is
///    rounded to a raw `Q32.32` word, i.e. to half a raw unit `2⁻³³` in the reals of the *scaled*
///    space, so
///    `|ΔC[o,j]| ≤ 2⁻³³·(2^{k_ws}·Σ_l |hl[l,j]| + 2^{k_hl}·Σ_l |ws[o,l]|)`.
///    At `|Ws| ≈ 1e-3` the bound is `≈1.2e-9` and the crate's mixed absolute/relative
///    `max_rel_diff` (`|new−old|/max(|old|,1)`) is asserted at `1e-7`; at `|Ws| ≈ 1e3` the bound
///    itself is `≈1.6e-7` — the grid quantum `2⁻³²` is an *absolute* resolution, so a magnitude-`1e3`
///    entry carries a relative quantum `≈2.3e-10` that the `1e3`-wide column mass of `Hl` turns into
///    `≈1e-7` in the product, and the bound is the statement at that magnitude.
#[test]
fn mul_metric_matches_old_f64_formula() {
    for &(o, n) in &[(4usize, 128usize), (16, 256)] {
        for &mag in &[1.0e-3f64, 1.0e3f64] {
            let mut rng = Lcg::new(0x6d_0000 + (o * n) as u64);
            let mut ws: Vec<f64> = (0..o * n).map(|_| mag * rng.sym()).collect();
            // Pin the largest magnitude, so the scale rule's exponent is deterministic.
            ws[0] = mag;
            ws[1] = -mag;
            let hl = spd_metric(&mut rng, n);
            let k = q32::panel_shift(max_abs(&ws));
            let want_k = if mag > 1.0 { 7 } else { 0 };
            assert_eq!(
                k, want_k,
                "({o},{n}) |Ws|~{mag:e}: panel_shift {k} != {want_k}"
            );
            let scale = 2f64.powi(-k);
            let scaled_max = scale * max_abs(&ws);
            assert!(
                scaled_max <= 8.0,
                "({o},{n}) |Ws|~{mag:e}: scaled panel {scaled_max} above 8 (raw above 2³⁵)"
            );
            let k_hl = q32::panel_shift(max_abs(&hl));
            let col_mass = (0..n)
                .map(|j| (0..n).map(|l| hl[l * n + j].abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let row_mass = (0..o)
                .map(|oo| ws[oo * n..oo * n + n].iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let bound = 2f64.powi(-33) * (2f64.powi(k) * col_mass + 2f64.powi(k_hl) * row_mass);
            let new = mul_metric(&ws, &hl, o, n).unwrap();
            let old = old_mul_metric(&ws, &hl, o, n);
            assert_eq!(new.len(), o * n);
            let diff = max_abs_diff(&new, &old);
            let rel = max_rel_diff(&new, &old);
            println!(
                "mul_metric vs old f64 ({o},{n}) |Ws|~{mag:e} k_ws={k} k_hl={k_hl}: max|diff| \
                 {diff:e} ≤ bound {bound:e}, mixed rel {rel:e}, max|new| {:.3e}",
                max_abs(&new)
            );
            assert!(
                diff <= bound,
                "({o},{n}) |Ws|~{mag:e}: |diff| {diff:e} above the quantization bound {bound:e}"
            );
            if mag < 1.0 {
                assert!(
                    rel <= 1e-7,
                    "({o},{n}) |Ws|~{mag:e} mixed relative error {rel:e} > 1e-7"
                );
            }
            assert!(max_abs(&new) > 0.0, "({o},{n}) |Ws|~{mag:e}: empty product");
        }
    }
}

/// (c) `mul_metric` refuses a non-finite, an out-of-bound and an unrepresentable input with
/// [`StepsError::ShapeMismatch`] and does not panic: a `NaN` and an infinite `Ws` entry, the same in
/// `Hl`, a length that disagrees with the declared shape in either operand, and a pair whose exact
/// undo power of two leaves `f64`. A zero row count is the empty product and stays `Ok`.
#[test]
fn mul_metric_refuses_non_finite_and_out_of_bound() {
    let mut rng = Lcg::new(0x6d_1efe);
    let ws = sym_vec(&mut rng, O * N);
    let hl = spd_metric(&mut rng, N);
    let nan_ws = {
        let mut w = ws.clone();
        w[3] = f64::NAN;
        w
    };
    assert_eq!(
        mul_metric(&nan_ws, &hl, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    let inf_ws = {
        let mut w = ws.clone();
        w[3] = f64::INFINITY;
        w
    };
    assert_eq!(
        mul_metric(&inf_ws, &hl, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    let nan_hl = {
        let mut h = hl.clone();
        h[5] = f64::NAN;
        h
    };
    assert_eq!(
        mul_metric(&ws, &nan_hl, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    let inf_hl = {
        let mut h = hl.clone();
        h[5] = f64::INFINITY;
        h
    };
    assert_eq!(
        mul_metric(&ws, &inf_hl, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    // Shapes that disagree with the declared `O × n` / `n × n`.
    assert_eq!(
        mul_metric(&ws[1..], &hl, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        mul_metric(&ws, &hl[1..], O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    // A magnitude pair whose exact undo power of two leaves `f64` (`1e308·1e308` in reals).
    let huge = vec![1.0e308f64; O * N];
    let huge_hl = vec![1.0e308f64; N * N];
    assert_eq!(
        mul_metric(&huge, &huge_hl, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    // Zero out rows: the empty product, as the deleted loop returned it.
    assert!(mul_metric(&[], &hl, 0, N).unwrap().is_empty());
}

/// (c) A non-finite or out-of-bound input to `d_step` returns the step's error, no panic: a `NaN`
/// metric entry, an infinite `Ws`, a `NaN` rotated sign entry, a non-finite row weight (the row
/// weight itself is shifted into the input bound, so only finiteness refuses), and a `Ws` large
/// enough that the right-hand side leaves the grid `i64` range.
#[test]
fn d_step_refuses_non_finite_and_out_of_bound() {
    let mut rng = Lcg::new(13);
    let ws = sym_vec(&mut rng, O * N);
    let hl = spd_metric(&mut rng, N);
    let s_rot = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let c = pos_vec(&mut rng, N);
    let nan_hl = {
        let mut h = hl.clone();
        h[5] = f64::NAN;
        h
    };
    assert_eq!(
        d_step(&ws, &nan_hl, &s_rot, &r, &c, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
    let inf_ws = {
        let mut w = ws.clone();
        w[3] = f64::INFINITY;
        w
    };
    assert_eq!(
        d_step(&inf_ws, &hl, &s_rot, &r, &c, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
    let nan_s = {
        let mut s = s_rot.clone();
        s[7] = f64::NAN;
        s
    };
    assert_eq!(
        d_step(&ws, &hl, &nan_s, &r, &c, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
    // A non-finite row weight.
    let mut gr_bad = pos_vec(&mut rng, O);
    gr_bad[O - 1] = f64::INFINITY;
    assert_eq!(
        d_step(&ws, &hl, &s_rot, &r, &c, O, N, Some(&gr_bad)).unwrap_err(),
        StepsError::ShapeMismatch
    );
    // A `Ws` whose right-hand side leaves the grid `i64` range.
    let big_ws = vec![1.0e30f64; O * N];
    assert_eq!(
        d_step(&big_ws, &hl, &s_rot, &r, &c, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
}

/// (d) Timing at the production shape `o = 896`, `n = 4864`: `d_step`'s runs on the shared engine
/// pool — `d_step` takes no thread parameter — its stage split (the same shared
/// primitives as `c_step` plus the rotated-metric `had_gram`), and, only when `OLD_D_STEP` is set,
/// one run of the test-local old `f64` body (its `mul_metric` triple loop alone is ~65 s at this
/// shape, so it does not fit the budget beside the runs). Run with
/// `--release -- --ignored --nocapture`.
#[test]
#[ignore = "timing campaign at o=896 n=4864: run with --release -- --ignored --nocapture"]
#[allow(
    clippy::too_many_lines,
    reason = "the timing campaign is one measurement unit: one fixture, one stage split and the \
              repeated runs have to report together or the split cannot be attributed."
)]
fn timing_d_step_production_shape() {
    let (o, n) = (896usize, 4864usize);
    let mut rng = Lcg::new(0x5eed_1111_d0d0_1234);
    let s_rot = sign_vec(&mut rng, o * n);
    let r = pos_vec(&mut rng, o * (n / GROUP));
    let ws = sym_vec(&mut rng, o * n);
    let c = pos_vec(&mut rng, n);
    let gr = pos_vec(&mut rng, o);
    // The same well-conditioned Kac–Murdock–Szegő Toeplitz metric the `c_step` timing uses
    // (`ρ^|i−j|`, `ρ = 0.5`, eigenvalues in `[1/3, 3]`); `o < n`, so the weighted Gram is rank-`o`
    // and the decaying metric keeps `A` diagonally dominant — a metric the `f64` body also solves.
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
    let st = d_step_stages(&ws, &hl, &s_rot, &r, &c, &gr, o, n);
    let sum = st.iter().sum::<f64>();
    println!(
        "d_step stage split o={o} n={n} threads=8, once each: mul_metric(scale+gemm_nn) {:.3}s \
         boundary-conversion(f64→Q32.32 maps) {:.3}s wsyrk {:.3}s hadamard_into {:.3}s chol_solve \
         {:.3}s had_gram+hlc (H_rot) {:.3}s sum {sum:.3}s",
        st[0], st[1], st[2], st[3], st[4], st[5]
    );
    let runs = step_runs();
    let mut times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = std::time::Instant::now();
        let d = d_step(&ws, &hl, &s_rot, &r, &c, o, n, Some(&gr)).expect("d_step");
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(d);
        times.push(dt);
    }
    let mean = times.iter().sum::<f64>() / (times.len() as f64);
    let mn = times.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = times.iter().copied().fold(0.0f64, f64::max);
    // The old f64 body adds ~157 s (its `mul_metric` loop ~65 s and its `f64` `A` loop the rest);
    // it runs only when `OLD_D_STEP` is set, keeping the default run inside the 580 s budget.
    let old_dt = if std::env::var_os("OLD_D_STEP").is_some() {
        let t0 = std::time::Instant::now();
        let od = old_d_step(&ws, &hl, &s_rot, &r, &c, o, n, Some(&gr));
        std::hint::black_box(od);
        t0.elapsed().as_secs_f64()
    } else {
        f64::NAN
    };
    println!(
        "d_step (mul_metric over gemm_nn) o={o} n={n} threads=8 runs={runs}: mean {mean:.3}s min \
         {mn:.3}s max {mx:.3}s runs {times:?}; stage sum {sum:.3}s, remainder (group_scale, G_rot \
         fwht, panel scan, ridge, right-hand side, read-back) {:.3}s; previous measurement of this \
         step with the scalar f64 mul_metric: mean 123.5 s (runs 122.754/124.136/123.621); old f64 \
         body at this shape (single run, NaN = not run): {old_dt:.3}s",
        mean - sum
    );
}

/// The `d_step` body stage by stage on the timing fixture, each stage timed once: `[mul_metric,
/// boundary conversion, wsyrk, hadamard_into, chol_solve, had_gram]`. The intermediates are
/// `d_step`'s own (`B = S_rot⊙r_rep`, its `2⁻ᵏ` panel scale, `G_rot = fwht128((Ws·Hl)⊙c)`,
/// `H_rot = had_gram(Hl ⊙ c cᵀ)`, the three f64→`Q32.32` maps, the ridge and the right-hand side);
/// `group_scale`, the `G_rot` transform, the panel scan, the ridge, the right-hand side and the
/// read-back are outside the timed regions and are the remainder reported against `d_step`'s mean.
#[allow(
    clippy::too_many_arguments,
    reason = "the split is handed `d_step`'s own operands (`ws`, `hl`, `s_rot`, `r`, `c`, `g_row`) \
              plus the geometry; bundling them would not be the same call the step makes."
)]
fn d_step_stages(
    ws: &[f64],
    hl: &[f64],
    s_rot: &[f64],
    r: &[f64],
    c: &[f64],
    gr: &[f64],
    o: usize,
    n: usize,
) -> [f64; 6] {
    let g = n / GROUP;
    let t0 = std::time::Instant::now();
    let wshl = mul_metric(ws, hl, o, n).expect("mul_metric");
    let staged_mul = t0.elapsed().as_secs_f64();
    let mut g_rot = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            g_rot[oo * n + i] = wshl[oo * n + i] * c[i];
        }
    }
    let g_rot = fwht128(&g_rot).expect("G_rot");
    let t0 = std::time::Instant::now();
    let mut hlc = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            hlc[i * n + j] = hl[i * n + j] * c[i] * c[j];
        }
    }
    let h_rot = had_gram(&hlc, n).expect("H_rot");
    let hrot = t0.elapsed().as_secs_f64();
    let mut b_scaled = vec![0.0f64; o * n];
    for oo in 0..o {
        for i in 0..n {
            b_scaled[oo * n + i] = s_rot[oo * n + i] * r[oo * g + i / GROUP];
        }
    }
    let k = q32::panel_shift(max_abs(&b_scaled));
    if k != 0 {
        let f = 2f64.powi(-k);
        for v in &mut b_scaled {
            *v *= f;
        }
    }
    let t0 = std::time::Instant::now();
    let panel_grid: Vec<i64> = b_scaled
        .iter()
        .map(|&x| q32::to_grid(x, q32::INPUT_BOUND).expect("B grid"))
        .collect();
    let gr_grid: Vec<i64> = gr
        .iter()
        .map(|&x| q32::to_grid(x, q32::INPUT_BOUND).expect("g_row grid"))
        .collect();
    let h_grid: Vec<i64> = h_rot
        .iter()
        .map(|&x| q32::to_grid(x, q32::GRID_BOUND).expect("H_rot grid"))
        .collect();
    let conv = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let mut a = wsyrk(&panel_grid, Some(&gr_grid), o, n).expect("wsyrk");
    let syrk = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    hadamard_into(&mut a, &h_grid).expect("hadamard_into");
    let had = t0.elapsed().as_secs_f64();
    let mut dmean = 0.0f64;
    for i in 0..n {
        dmean += q32::from_grid(a[i * n + i]);
    }
    dmean /= n as f64;
    let ridge = 1e-12 * dmean.max(1e-12 * 2f64.powi(-2 * k));
    let mut ridge_raw = (ridge * q32::GRID_SCALE).round() as i64;
    if ridge_raw == 0 {
        ridge_raw = 1;
    }
    for i in 0..n {
        a[i * n + i] += ridge_raw;
    }
    let mut rhs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for oo in 0..o {
            acc += b_scaled[oo * n + i] * gr[oo] * g_rot[oo * n + i];
        }
        rhs[i] = acc;
    }
    let b_grid: Vec<i64> = rhs
        .iter()
        .map(|&x| q32::to_grid(x, q32::GRID_BOUND).expect("b grid"))
        .collect();
    let t0 = std::time::Instant::now();
    let solved = chol_solve(&a, &b_grid, n, 1).expect("chol_solve");
    let chol = t0.elapsed().as_secs_f64();
    std::hint::black_box(&solved);
    [staged_mul, conv, syrk, had, chol, hrot]
}

/// (b) The `_wshl` variants with a caller-supplied `wshl = mul_metric(ws, hl, o, n)` return the
/// same vectors as the wrapper steps that form the product internally, element for element under
/// `f64::to_bits`: `c_step_wshl` against `c_step` with `None` and with a row weight, `r_step_wshl`
/// against `r_step`, and `d_step_wshl` against `d_step` with `None` and with a row weight.
///
/// Mutation control: an off-by-one read in `r_step_wshl` (`wshl[oo·n + idx + 1]`) turns this test
/// (and `r_step_matches_old_f64_formula`) red.
#[test]
fn wshl_variants_are_bit_identical_to_wrappers() {
    let mut rng = Lcg::new(0x7750_0001);
    let s = sign_vec(&mut rng, O * N);
    let s_rot = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let c = pos_vec(&mut rng, N);
    let hl = spd_metric(&mut rng, N);
    let ws = sym_vec(&mut rng, O * N);
    let gr = pos_vec(&mut rng, O);
    let wshl = mul_metric(&ws, &hl, O, N).unwrap();

    let c_ref = c_step(&ws, &hl, &s, &r, O, N, None).unwrap();
    let c_new = c_step_wshl(&ws, &hl, &wshl, &s, &r, O, N, None).unwrap();
    assert_eq!(c_ref.len(), c_new.len());
    for k in 0..c_ref.len() {
        assert_eq!(
            c_ref[k].to_bits(),
            c_new[k].to_bits(),
            "c_step_wshl[{k}] differs from c_step"
        );
    }
    let c_w_ref = c_step(&ws, &hl, &s, &r, O, N, Some(&gr)).unwrap();
    let c_w_new = c_step_wshl(&ws, &hl, &wshl, &s, &r, O, N, Some(&gr)).unwrap();
    for k in 0..c_w_ref.len() {
        assert_eq!(
            c_w_ref[k].to_bits(),
            c_w_new[k].to_bits(),
            "c_step_wshl(weights)[{k}] differs from c_step"
        );
    }

    let r_ref = r_step(&ws, &hl, &s, &c, O, N).unwrap();
    let r_new = r_step_wshl(&ws, &hl, &wshl, &s, &c, O, N).unwrap();
    assert_eq!(r_ref.len(), r_new.len());
    for k in 0..r_ref.len() {
        assert_eq!(
            r_ref[k].to_bits(),
            r_new[k].to_bits(),
            "r_step_wshl[{k}] differs from r_step"
        );
    }

    let d_ref = d_step(&ws, &hl, &s_rot, &r, &c, O, N, None).unwrap();
    let d_new = d_step_wshl(&ws, &hl, &wshl, &s_rot, &r, &c, O, N, None).unwrap();
    assert_eq!(d_ref.len(), d_new.len());
    for k in 0..d_ref.len() {
        assert_eq!(
            d_ref[k].to_bits(),
            d_new[k].to_bits(),
            "d_step_wshl[{k}] differs from d_step"
        );
    }
    let d_w_ref = d_step(&ws, &hl, &s_rot, &r, &c, O, N, Some(&gr)).unwrap();
    let d_w_new = d_step_wshl(&ws, &hl, &wshl, &s_rot, &r, &c, O, N, Some(&gr)).unwrap();
    for k in 0..d_w_ref.len() {
        assert_eq!(
            d_w_ref[k].to_bits(),
            d_w_new[k].to_bits(),
            "d_step_wshl(weights)[{k}] differs from d_step"
        );
    }
}

/// (b) Each `_wshl` variant refuses a product one element short of `o·n` with
/// [`StepsError::ShapeMismatch`], before any work: `c_step_wshl`, `r_step_wshl` and `d_step_wshl`.
#[test]
fn wshl_variants_refuse_a_short_product() {
    let mut rng = Lcg::new(0x7750_0002);
    let s = sign_vec(&mut rng, O * N);
    let s_rot = sign_vec(&mut rng, O * N);
    let r = pos_vec(&mut rng, O * G);
    let c = pos_vec(&mut rng, N);
    let hl = spd_metric(&mut rng, N);
    let ws = sym_vec(&mut rng, O * N);
    let wshl = mul_metric(&ws, &hl, O, N).unwrap();
    let short = &wshl[..wshl.len() - 1];
    assert_eq!(short.len(), O * N - 1);
    assert_eq!(
        c_step_wshl(&ws, &hl, short, &s, &r, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        r_step_wshl(&ws, &hl, short, &s, &c, O, N).unwrap_err(),
        StepsError::ShapeMismatch
    );
    assert_eq!(
        d_step_wshl(&ws, &hl, short, &s_rot, &r, &c, O, N, None).unwrap_err(),
        StepsError::ShapeMismatch
    );
}
