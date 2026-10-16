//! `tests/learn_fold_stage.rs` — the per-stage sink-gated statistics and the `fold_gqa` split
//! intra/inter ridge solve (`src/learn/fold/layer/stage.rs`).
//!
//! Positive: two slabs fold to the same accumulator as one concatenated slab; `Hm_w = Hm` under
//! unit weights and `Hm_w = (Xw)ᵀ(Xw)` with the weight applied linearly; the even-count lower-median
//! rule; the `> thresh·r_med` token weight; a solve with `Cs = H_solve·W0ᵀ` leaves `W0` unchanged;
//! the solve satisfies the ridge normal equations; the sink clamp caps a row at `τ·‖W0_row‖`.
//! Refusals: shape mismatch, non-SPD, non-finite, empty median and missing `Cts`. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixed-seed LCG maps a 24-bit integer field to an f64, and the test kernels index \
              row-major `[r, c]` buffers by `i/j/k/o` against the spec's arithmetic; the exact float \
              comparisons are on values the LCG and the lowering produce bit-for-bit."
)]

use apeiron::learn::fold::layer::stage::{
    bias_update, metric, ridge_solve, sink_token_weights, solve_stage, SolveParams, StageAccum,
    StageError,
};
use apeiron::learn::fold::stats::hocapture::Kind;

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

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn rand_mat(lcg: &mut Lcg, rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|_| lcg.sym()).collect()
}

/// Row-major `[r, k] · [k, c]` product.
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

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// (a) Folding two slabs equals folding their concatenation, bitwise-close.
#[test]
fn two_slabs_equal_concatenation() {
    let n = 4;
    let outs = [3usize, 2];
    let with_ts = [true, false];
    let (r1, r2) = (3usize, 2usize);
    let mut lcg = Lcg::new(0x51ab_cd01);
    let x1 = rand_mat(&mut lcg, r1, n);
    let x2 = rand_mat(&mut lcg, r2, n);
    let w1 = rand_mat(&mut lcg, r1, 1);
    let w2 = rand_mat(&mut lcg, r2, 1);
    let y1a = rand_mat(&mut lcg, r1, outs[0]);
    let y2a = rand_mat(&mut lcg, r2, outs[0]);
    let y1b = rand_mat(&mut lcg, r1, outs[1]);
    let y2b = rand_mat(&mut lcg, r2, outs[1]);
    let t1 = rand_mat(&mut lcg, r1, outs[0]);
    let t2 = rand_mat(&mut lcg, r2, outs[0]);

    let mut split = StageAccum::new(n, &outs, &with_ts).unwrap();
    split
        .add_slab(
            &x1,
            &w1,
            &[y1a.as_slice(), y1b.as_slice()],
            &[Some(t1.as_slice()), None],
        )
        .unwrap();
    split
        .add_slab(
            &x2,
            &w2,
            &[y2a.as_slice(), y2b.as_slice()],
            &[Some(t2.as_slice()), None],
        )
        .unwrap();

    let x = [x1.as_slice(), x2.as_slice()].concat();
    let w = [w1.as_slice(), w2.as_slice()].concat();
    let ya = [y1a.as_slice(), y2a.as_slice()].concat();
    let yb = [y1b.as_slice(), y2b.as_slice()].concat();
    let ta = [t1.as_slice(), t2.as_slice()].concat();
    let mut one = StageAccum::new(n, &outs, &with_ts).unwrap();
    one.add_slab(
        &x,
        &w,
        &[ya.as_slice(), yb.as_slice()],
        &[Some(ta.as_slice()), None],
    )
    .unwrap();

    assert!(max_abs_diff(&split.hm, &one.hm) < 1e-12);
    assert!(max_abs_diff(&split.hm_w, &one.hm_w) < 1e-12);
    assert!(max_abs_diff(&split.cs[0], &one.cs[0]) < 1e-12);
    assert!(max_abs_diff(&split.cs[1], &one.cs[1]) < 1e-12);
    assert!(max_abs_diff(&split.cts[0], &one.cts[0]) < 1e-12);
    assert!(split.cts[1].is_empty() && one.cts[1].is_empty());
    assert!(max_abs_diff(&split.x_sum, &one.x_sum) < 1e-12);
    assert!(max_abs_diff(&split.y_sum[0], &one.y_sum[0]) < 1e-12);
    assert!(max_abs_diff(&split.y_sum[1], &one.y_sum[1]) < 1e-12);
    assert_eq!(split.n_toks, one.n_toks);
}

/// (b) With every `w_tok == 1`, `Hm_w` equals the unweighted `Hm`.
#[test]
fn hm_w_equals_hm_under_unit_weights() {
    let n = 5;
    let outs = [2usize];
    let with_ts = [false];
    let rows = 6;
    let mut lcg = Lcg::new(11);
    let x = rand_mat(&mut lcg, rows, n);
    let w = vec![1.0f64; rows];
    let y = rand_mat(&mut lcg, rows, outs[0]);
    let mut acc = StageAccum::new(n, &outs, &with_ts).unwrap();
    acc.add_slab(&x, &w, &[y.as_slice()], &[None]).unwrap();
    assert!(max_abs_diff(&acc.hm, &acc.hm_w) < 1e-12);
}

/// (b') `Hm_w = (Xw)ᵀ(Xw)` with `Xw = X·w` applied linearly (not `√w`) — the mutation-sensitive
/// check.
#[test]
fn hm_w_applies_weights_linearly() {
    let n = 5;
    let outs = [2usize];
    let with_ts = [false];
    let rows = 7;
    let mut lcg = Lcg::new(23);
    let x = rand_mat(&mut lcg, rows, n);
    let w: Vec<f64> = (0..rows).map(|r| 0.5 + (r as f64) * 0.75).collect();
    let y = rand_mat(&mut lcg, rows, outs[0]);
    let mut acc = StageAccum::new(n, &outs, &with_ts).unwrap();
    acc.add_slab(&x, &w, &[y.as_slice()], &[None]).unwrap();
    // Xw row r is X row r scaled by w[r].
    let mut xw = x.clone();
    for r in 0..rows {
        for i in 0..n {
            xw[r * n + i] *= w[r];
        }
    }
    let expect = matmul(&transpose(&xw, rows, n), &xw, n, rows, n);
    assert!(max_abs_diff(&acc.hm_w, &expect) < 1e-12);
    // Guard the check's own sensitivity: with √w the matrix differs.
    let mut xs = x.clone();
    for r in 0..rows {
        for i in 0..n {
            xs[r * n + i] *= w[r].sqrt();
        }
    }
    let sqrt_gram = matmul(&transpose(&xs, rows, n), &xs, n, rows, n);
    assert!(max_abs_diff(&acc.hm_w, &sqrt_gram) > 1e-6);
}

/// (c) The even-count median takes the lower middle element (sorted index `(k-1)/2`).
#[test]
fn lower_median_on_even_count() {
    let h_t = [0.0f64, 1.0, 2.0, 3.0, 4.0];
    let zeros = [0.0f64; 5];
    let w = sink_token_weights(&h_t, &zeros, &zeros, &zeros, None, 1, 5, 1, 0.5).unwrap();
    // r_t = positions {1,2,3,4}; lower median = 2; thresh·r_med = 1.
    let expect = [1.0f64, 1.0, 1.0, 2.0 / 3.0, 0.5];
    assert!(max_abs_diff(&w, &expect) < 1e-12);
}

/// (d) A token with `r_tok > thresh·r_med` gets `r_med / r_tok`; the rest get `1`; `r_s` includes
/// `p_in_mid` when given.
#[test]
fn token_weight_rule_and_mid_input() {
    let h_t = [1.0f64, 2.0, 10.0];
    let p_mid = [0.0f64; 3];
    let p_out = [0.0f64; 3];
    let h_s = [3.0f64, 0.0, 0.0];
    let p_in_mid = [0.0f64, 5.0, 0.0];
    let w = sink_token_weights(&h_t, &p_mid, &p_out, &h_s, Some(&p_in_mid), 1, 3, 1, 2.0).unwrap();
    // r_t = {1,2,10}, r_s = {3,5,0}, r_tok = {3,5,10}; median over positions >=1 of r_t = 2;
    // thresh·r_med = 4 -> {1, 0.4, 0.2}.
    let expect = [1.0f64, 0.4, 0.2];
    assert!(max_abs_diff(&w, &expect) < 1e-12);
}

/// (e) With `Cs = Cts = H_solve·W0ᵀ` the split residuals vanish and `Ws == W0`.
#[test]
fn solve_with_anchor_residual_is_identity() {
    let n = 4;
    let outs = [1usize];
    let with_ts = [true];
    let w0 = [0.5f64, -0.25, 0.1, 0.3];
    // X = I so Hm_w = I; Y = X·W0ᵀ = W0ᵀ, and Y_ts = Y.
    let mut x = vec![0.0f64; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0;
    }
    let y = w0.to_vec();
    let w = vec![1.0f64; n];
    let mut acc = StageAccum::new(n, &outs, &with_ts).unwrap();
    acc.add_slab(&x, &w, &[y.as_slice()], &[Some(y.as_slice())])
        .unwrap();
    let params = SolveParams::default();
    let ws = solve_stage(&acc, &[w0.as_slice()], &[Kind::O], &params).unwrap();
    assert!(max_abs_diff(&ws[0], &w0) < 1e-12);
}

/// (f) The solved `dW` satisfies the ridge normal equations
/// `(H + ridge·mean(diag H)·I)·dWᵀ = C`.
#[test]
fn solve_satisfies_normal_equations() {
    let (n, out, rows) = (6usize, 2usize, 12usize);
    let mut lcg = Lcg::new(7);
    let x = rand_mat(&mut lcg, rows, n);
    let h = matmul(&transpose(&x, rows, n), &x, n, rows, n);
    let c = rand_mat(&mut lcg, n, out);
    let ridge = 0.045 * (n as f64 / 896.0);
    let dw = ridge_solve(&h, &c, n, out, ridge).unwrap();
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let mut residual = 0.0f64;
    for i in 0..n {
        for o in 0..out {
            let mut s = 0.0f64;
            for k in 0..n {
                let hr = h[i * n + k] + if i == k { ridge * mean } else { 0.0 };
                s += hr * dw[o * n + k];
            }
            residual = residual.max((s - c[i * out + o]).abs());
        }
    }
    assert!(residual < 1e-9, "residual {residual}");
}

/// (g) The sink clamp caps a `dW` row at `τ·max(‖W0_row‖, 1e-6)`.
#[test]
fn sink_clamp_caps_row_ratio() {
    let n = 4;
    let outs = [1usize];
    let with_ts = [true];
    let w0 = [0.5f64, 0.0, 0.0, 0.0];
    let mut x = vec![0.0f64; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0;
    }
    // Y_ts = X·W0ᵀ + Z with Z = [10,0,0,0] -> Cts - C0 = Z; Cs = Cts so C_inter = 0.
    let y = [0.5f64 + 10.0, 0.0, 0.0, 0.0];
    let w = vec![1.0f64; n];
    let mut acc = StageAccum::new(n, &outs, &with_ts).unwrap();
    acc.add_slab(&x, &w, &[y.as_slice()], &[Some(y.as_slice())])
        .unwrap();
    let params = SolveParams::default();
    let ws = solve_stage(&acc, &[w0.as_slice()], &[Kind::O], &params).unwrap();
    let dw_row = [ws[0][0] - w0[0], ws[0][1], ws[0][2], ws[0][3]];
    let dw_rn = dw_row.iter().map(|v| v * v).sum::<f64>().sqrt();
    let w0_rn = w0.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (dw_rn - params.sink_tau * w0_rn).abs() < 1e-9,
        "dw_rn {dw_rn}"
    );
    assert!((dw_rn / w0_rn - params.sink_tau).abs() < 1e-9);
}

/// (h) Refusals: shape mismatch, empty median, non-SPD, non-finite and missing `Cts`.
#[test]
fn refusals() {
    let z = [0.0f64; 4];
    // shape mismatch: h_t too short for seqs=1, t=2, width=1.
    assert_eq!(
        sink_token_weights(&[1.0, 2.0], &[0.0], &z, &z, None, 1, 2, 1, 1.0),
        Err(StageError::Shape)
    );
    // empty median: t < 2.
    assert_eq!(
        sink_token_weights(&[1.0], &[0.0], &[0.0], &[0.0], None, 1, 1, 1, 1.0),
        Err(StageError::Empty)
    );
    // non-finite token input.
    assert_eq!(
        sink_token_weights(
            &[f64::NAN, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            None,
            1,
            2,
            1,
            1.0
        ),
        Err(StageError::NonFinite)
    );
    // add_slab shape mismatch.
    let mut acc = StageAccum::new(3, &[2], &[false]).unwrap();
    assert_eq!(
        acc.add_slab(&[1.0, 2.0], &[1.0], &[&[1.0, 2.0]], &[None]),
        Err(StageError::Shape)
    );
    // add_slab non-finite.
    assert_eq!(
        acc.add_slab(&[f64::INFINITY; 3], &[1.0], &[&[1.0, 2.0]], &[None]),
        Err(StageError::NonFinite)
    );
    // solve_stage shape mismatch (W0 length wrong).
    let mut acc2 = StageAccum::new(3, &[2], &[false]).unwrap();
    acc2.add_slab(&[1.0, 0.0, 0.0], &[1.0], &[&[1.0, 2.0]], &[None])
        .unwrap();
    let params = SolveParams::default();
    assert_eq!(
        solve_stage(&acc2, &[&[1.0, 2.0]], &[Kind::Q], &params),
        Err(StageError::Shape)
    );
    // missing Cts: an `o` output accumulated with with_ts = false.
    assert_eq!(
        solve_stage(
            &acc2,
            &[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            &[Kind::O],
            &params
        ),
        Err(StageError::MissingCts)
    );
    // non-SPD: the zero matrix has a zero pivot.
    assert_eq!(
        ridge_solve(&[0.0; 9], &[1.0; 3], 3, 1, 0.0),
        Err(StageError::NotPositiveDefinite)
    );
    // metric shape mismatch.
    assert_eq!(metric(&[1.0; 3], 3, 0.1), Err(StageError::Shape));
    // bias_update zero token count.
    assert_eq!(
        bias_update(&[1.0], &[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], 0.0),
        Err(StageError::Empty)
    );
}
