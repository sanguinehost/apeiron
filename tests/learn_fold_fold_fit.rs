//! `tests/learn_fold_fold_fit.rs` — the production `fold_fit`: the winning pack recomposes to `Wp`
//! bit-for-bit, `wp = plane_w(pack)`, `j_best` is the candidate minimum with the refit strictly
//! below the plain `gptq` candidate, the carriers `dh`/`ch` lie on their `quant_c_u8` grids and the
//! code plane's 8-blocks are exact `E8 ∪ O8` codewords, the refusals, the `g_row` dependence, and a
//! numeric regression pin. Grade `tested`.
//!
//! Mutation control (see the lane return): removing the 8-block sign parity fix in `fold_fit`, or
//! flipping the `<` in the `e8_icm` pack-candidate comparison to `>`, turns
//! `learn_fold_fold_fit_production_numbers_are_pinned` red.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::unreadable_literal,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps integers to `f64` by division and the flat row-major index \
              arithmetic is the reference payload format; the pack/recompose, grid and codebook \
              checks are exact float comparisons by construction, and the `o/n/m/k/g` names are the \
              reference's vocabulary"
)]

use apeiron::learn::code::e8::codeword;
use apeiron::learn::fold::fit::fold_fit::{
    fold_fit, FoldFit, FoldFitConfig, FoldFitError, FoldFitTag, BLOCK, GROUP,
};
use apeiron::learn::fold::fit::icm::{nearest_index, plane_w, PlaneRef};
use apeiron::learn::fold::fit::quant::{f16r, quant_r_hier};
use apeiron::learn::fold::fit::steps::{compose, fwht128, objective};

/// Deterministic xorshift64 LCG.
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

    /// Uniform in `[lo, hi)` from the top 53 bits.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        lo + (hi - lo) * u
    }
}

/// A positive-definite Toeplitz metric `H[i,j] = rho^|i-j|`, the calibration Gram stand-in.
fn toeplitz(n: usize, rho: f64) -> Vec<f64> {
    let mut h = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let d = i.abs_diff(j);
            h[i * n + j] = rho.powi(i32::try_from(d).expect("d < 2^31"));
        }
    }
    h
}

fn rand_mat(rows: usize, cols: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..rows * cols).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// The production call-site configuration (`fold_transfer.py:1733–1736`).
fn production_cfg() -> FoldFitConfig {
    FoldFitConfig {
        iters: 1,
        ridge: 1.0e-3,
        damp: 0.01,
        inner_iters: 3,
        e8_beam: 4,
        icm_ridge: 0.20,
        sweeps: 2,
    }
}

/// The fixed test matrix: `o = 4`, `n = 256`, a correlated SPD Toeplitz Gram.
fn fixed_inputs() -> (usize, usize, Vec<f64>, Vec<f64>) {
    let (o, n) = (4_usize, 256_usize);
    let h = toeplitz(n, 0.4);
    let ws = rand_mat(o, n, &mut Lcg::new(0x0F01DF17));
    (o, n, ws, h)
}

fn run() -> (usize, usize, Vec<f64>, Vec<f64>, FoldFit) {
    let (o, n, ws, h) = fixed_inputs();
    let fit = fold_fit(&ws, &h, o, n, None, &production_cfg()).expect("production fit");
    (o, n, ws, h, fit)
}

/// The `E8 ∪ O8` codeword `idx` as real values `codeword(idx)[k] / 2^32`.
fn codebook_row(idx: usize) -> [f64; BLOCK] {
    let one = (1_i64 << 32) as f64;
    let row = codeword(idx).expect("row in range");
    let mut out = [0.0_f64; BLOCK];
    for k in 0..BLOCK {
        out[k] = row[k] as f64 / one;
    }
    out
}

/// `x[:, i] *= v[i]` for a row-major `[o, n]` buffer, the test-side recomposition helper.
fn scale_columns(x: &mut [f64], o: usize, n: usize, v: &[f64]) {
    for row in 0..o {
        for col in 0..n {
            x[row * n + col] *= v[col];
        }
    }
}

/// `metric(H, ridge)`, the test-side objective metric.
fn metric(h: &[f64], n: usize, ridge: f64) -> Vec<f64> {
    let mut hl = h.to_vec();
    let mean = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        hl[i * n + i] += ridge * mean;
    }
    hl
}

/// `J(Ws, Wp, Hl)` of a returned `Wp`.
fn wp_objective(ws: &[f64], h: &[f64], o: usize, n: usize, wp: &[f64]) -> f64 {
    let hl = metric(h, n, production_cfg().ridge);
    objective(ws, wp, &hl, o, n).expect("objective")
}

/// A value is on the affine `bits`-grid `[f16r(min), f16r(max)]` with `2^bits − 1` levels: the
/// `quant_c_u8` grid of the carrier.
fn on_affine_grid(values: &[f64], bits: u32) -> bool {
    let lo = f16r(values.iter().copied().fold(f64::INFINITY, f64::min));
    let hi = f16r(values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let span = (hi - lo).max(1e-12);
    let levels = f64::from((1_u32 << bits) - 1);
    values.iter().all(|&v| {
        let q = (((v - lo) / span) * levels)
            .round_ties_even()
            .clamp(0.0, levels);
        // quant_c_u8 rounds each level once to the Q32.32 grid: one quantum 2^-32 plus f64 rounding of
        // this recomputation, far below the level spacing span/levels.
        (lo + (q / levels) * span - v).abs() <= 2.0_f64.powi(-31)
    })
}

#[test]
fn learn_fold_fold_fit_pack_recomposes_to_wp_bitwise() {
    let (o, n, _ws, _h, fit) = run();
    let mut rotated = fit.pack.s_rot.clone();
    scale_columns(&mut rotated, o, n, &fit.pack.dh);
    let s_dh = fwht128(&rotated).expect("fwht");
    let recomposed = compose(&s_dh, &fit.pack.rh, &fit.pack.ch, o, n).expect("compose");
    assert_eq!(recomposed.len(), o * n);
    for k in 0..o * n {
        assert_eq!(
            recomposed[k].to_bits(),
            fit.wp[k].to_bits(),
            "pack does not recompose to wp at {k}: {} != {}",
            recomposed[k],
            fit.wp[k]
        );
    }
}

#[test]
fn learn_fold_fold_fit_wp_equals_plane_w_of_pack() {
    let (o, n, _ws, _h, fit) = run();
    let idx = vec![0_u16; o * (n / BLOCK)];
    let w = plane_w(
        PlaneRef {
            s: &fit.pack.s_rot,
            idx: &idx,
            r: &fit.pack.rh,
            c: &fit.pack.ch,
            d: &fit.pack.dh,
        },
        o,
        n,
    )
    .expect("plane_w");
    // Both sides transform the same `s·d` through the same grid `fwht128`, so `w = s̃·r·c` differs
    // only in `compose`'s carrier work: `to_grid` rounds each of `s̃`, `r`, `c` (|δ| ≤ 2⁻³³) and
    // `mul_word` rounds the two products (|ε| ≤ 2⁻³³), giving
    //   |Δw| ≤ 2⁻³³·(1 + |c| + |s̃·c| + |s̃·r| + |r·c|)
    // at the fixture's operand maxima; `plane_w`'s two `f64` associations add `2⁻⁵²·|w|` each,
    // six orders below that term.
    let mut rotated = fit.pack.s_rot.clone();
    scale_columns(&mut rotated, o, n, &fit.pack.dh);
    let s_dh = fwht128(&rotated).expect("fwht");
    let max_abs = |x: &[f64]| x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let (s_max, r_max, c_max) = (max_abs(&s_dh), max_abs(&fit.pack.rh), max_abs(&fit.pack.ch));
    let tol = 2.0_f64.powi(-33) * (1.0 + c_max + s_max * c_max + s_max * r_max + r_max * c_max);
    for k in 0..o * n {
        assert!(
            (w[k] - fit.wp[k]).abs() <= tol,
            "wp != plane_w(pack) at {k}: {} != {} (tol {tol:e})",
            w[k],
            fit.wp[k]
        );
    }
}

#[test]
fn learn_fold_fold_fit_j_best_is_the_candidate_minimum() {
    let (o, n, ws, h, fit) = run();
    assert_eq!(fit.trace.len(), 2);
    let min = fit.trace.iter().copied().fold(f64::INFINITY, f64::min);
    assert_eq!(fit.j_best, min, "j_best is not the trace minimum");
    assert!(
        fit.j_best <= fit.trace[0],
        "j_best above the gptq candidate"
    );
    // The continuous refit strictly improves on the plain gptq candidate on this matrix.
    assert!(
        fit.trace[1] < fit.trace[0],
        "refit {} not below gptq {}",
        fit.trace[1],
        fit.trace[0]
    );
    assert_eq!(fit.tag, FoldFitTag::Refit0);
    assert_eq!(fit.tag.as_str(), "refit0");
    // The final quantised weight is no worse than the winning candidate's continuous fit.
    let jwp = wp_objective(&ws, &h, o, n, &fit.wp);
    assert!(jwp.is_finite(), "J(wp) is not finite");
    assert!(
        jwp <= fit.j_best + 1e-9 * (1.0 + fit.j_best.abs()),
        "J(wp) {jwp} above the candidate J {}",
        fit.j_best
    );
}

#[test]
fn learn_fold_fold_fit_pack_carriers_are_in_stored_format() {
    let (o, n, _ws, _h, fit) = run();
    let g = n / GROUP;
    let nb = n / BLOCK;
    // dh is on the 5-bit affine grid of quant_c_u8; ch on the 6-bit grid (lines 933, 939).
    assert_eq!(fit.pack.dh.len(), n);
    assert_eq!(fit.pack.ch.len(), n);
    assert!(on_affine_grid(&fit.pack.dh, 5), "dh off the 5-bit grid");
    assert!(on_affine_grid(&fit.pack.ch, 6), "ch off the 6-bit grid");
    // The bit budgets of the carriers: 4·o·g + 8·o + 64, 6·n + 32 and 5·n + 32.
    assert_eq!(fit.bits_r, (4 * o * g + 8 * o + 64) as u64);
    assert_eq!(fit.bits_c, (6 * n + 32) as u64);
    assert_eq!(fit.bits_d, (5 * n + 32) as u64);
    // rh is the hier carrier's `rq`: the stored-format decode of its own re-quantisation is exact.
    let h2 = quant_r_hier(&fit.pack.rh, o, g).expect("rh hier");
    assert_eq!(h2.bits, fit.bits_r);
    let decoded = h2.reconstruct();
    for k in 0..decoded.len() {
        assert_eq!(
            decoded[k].to_bits(),
            h2.rq[k].to_bits(),
            "rh hier decode mismatch at {k}"
        );
    }
    // The 8-block sign/parity structure: every block of the code plane is exactly an `E8 ∪ O8`
    // codeword, so `icm::nearest_index` returns its own row with distance 0 (`codeword` reproduces
    // the block bit-for-bit). The parity fix keeps the block inside the codebook across the sign
    // flip.
    for row in 0..o {
        for m in 0..nb {
            let mut block = [0.0_f64; BLOCK];
            for k in 0..BLOCK {
                block[k] = fit.pack.s_rot[row * n + m * BLOCK + k];
            }
            let idx = nearest_index(&block).expect("nearest");
            let cb = codebook_row(idx);
            for k in 0..BLOCK {
                assert_eq!(
                    cb[k].to_bits(),
                    block[k].to_bits(),
                    "row {row} block {m} coord {k} is not codeword {idx}"
                );
            }
        }
    }
}

#[test]
fn learn_fold_fold_fit_refuses_bad_configuration_and_shapes() {
    let (o, n, ws, h) = fixed_inputs();
    let cfg = production_cfg();
    let bad_iters = FoldFitConfig { iters: 2, ..cfg };
    assert_eq!(
        fold_fit(&ws, &h, o, n, None, &bad_iters).err(),
        Some(FoldFitError::Unsupported)
    );
    assert_eq!(
        fold_fit(&ws, &h, 0, n, None, &cfg).err(),
        Some(FoldFitError::ZeroDim)
    );

    // A short weight, Hessian or row weight is a shape refusal.
    assert_eq!(
        fold_fit(&ws[1..], &h, o, n, None, &cfg).err(),
        Some(FoldFitError::ShapeMismatch)
    );
    assert_eq!(
        fold_fit(&ws, &h[1..], o, n, None, &cfg).err(),
        Some(FoldFitError::ShapeMismatch)
    );
    assert_eq!(
        fold_fit(&ws, &h, o, n, Some(&[1.0, 1.0]), &cfg).err(),
        Some(FoldFitError::ShapeMismatch)
    );

    // `n` must be a positive multiple of GROUP.
    let bad_n = 64_usize;
    assert_eq!(
        fold_fit(
            &vec![0.0; o * bad_n],
            &vec![0.0; bad_n * bad_n],
            o,
            bad_n,
            None,
            &cfg
        )
        .err(),
        Some(FoldFitError::Width)
    );

    // A non-finite weight or Hessian is refused.
    let mut ws_nan = ws.clone();
    ws_nan[7] = f64::NAN;
    assert_eq!(
        fold_fit(&ws_nan, &h, o, n, None, &cfg).err(),
        Some(FoldFitError::NonFinite)
    );
    let mut h_inf = h.clone();
    h_inf[3] = f64::INFINITY;
    assert_eq!(
        fold_fit(&ws, &h_inf, o, n, None, &cfg).err(),
        Some(FoldFitError::NonFinite)
    );
}

#[test]
fn learn_fold_fold_fit_g_row_changes_the_result() {
    let (o, n, ws, h) = fixed_inputs();
    let cfg = production_cfg();
    let g_row: Vec<f64> = (0..o).map(|i| 0.3 + 0.4 * (i as f64)).collect();
    let with = fold_fit(&ws, &h, o, n, Some(&g_row), &cfg).expect("g_row fit");
    let without = fold_fit(&ws, &h, o, n, None, &cfg).expect("unweighted fit");

    // Both run and give a finite objective.
    assert!(with.j_best.is_finite() && without.j_best.is_finite());
    assert!(wp_objective(&ws, &h, o, n, &with.wp).is_finite());
    assert!(wp_objective(&ws, &h, o, n, &without.wp).is_finite());

    // The row weight enters `c_step` and `d_step`, so it moves the fit.
    let differs = with
        .wp
        .iter()
        .zip(&without.wp)
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        differs || with.j_best.to_bits() != without.j_best.to_bits(),
        "g_row changed nothing"
    );
}

/// The reference numbers of the production fit on the fixed matrix, pinned to catch a change in the
/// refit alternation or the pack-candidate rule (the mutation control's sensor). The tolerance is
/// `1e-6` relative; the recorded run is bit-exact on the gate box.
#[test]
fn learn_fold_fold_fit_production_numbers_are_pinned() {
    let (o, n, ws, h, fit) = run();
    let jwp = wp_objective(&ws, &h, o, n, &fit.wp);
    let close = |got: f64, want: f64| (got - want).abs() <= 1e-6 * (1.0 + want.abs());
    assert!(
        close(fit.trace[0], 9.553_694_787_010_723e1),
        "trace[0] = {}",
        fit.trace[0]
    );
    assert!(
        close(fit.trace[1], 4.359_351_242_410_637e1),
        "trace[1] = {}",
        fit.trace[1]
    );
    assert!(
        close(fit.j_best, 4.359_351_242_410_637e1),
        "j_best = {}",
        fit.j_best
    );
    assert!(close(jwp, 4.351_880_967_972_322e1), "J(wp) = {jwp}");
    let wsum: f64 = fit.wp.iter().map(|x| x.abs()).sum();
    assert!(close(wsum, 4.515_831_870_208_334e2), "|wp| sum = {wsum}");
}
