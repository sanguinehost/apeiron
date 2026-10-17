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
    fold_fit, FoldFit, FoldFitConfig, FoldFitError, FoldFitTag, PackOut, BLOCK, GROUP,
};
use apeiron::learn::fold::fit::gptq_e8::gptq_e8;
use apeiron::learn::fold::fit::icm::{e8_icm_step, nearest_index, plane_w, PlaneRef};
use apeiron::learn::fold::fit::quant::{f16r, quant_c_u8, quant_r_hier};
use apeiron::learn::fold::fit::steps::{
    c_step_wshl, compose, d_step_wshl, fwht128, had_gram, mul_metric, objective, r_step_wshl,
};
use apeiron::learn::solve::q32::{from_grid, round_q32_even, to_grid, GRID_BOUND};

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

/// The Q32.32 grid quantum `2⁻³²`.
const Q32_QUANTUM: f64 = 2.328_306_436_538_696_3e-10;

/// `x[:, i] *= v[i]` for a row-major `[o, n]` buffer in the carrier's arithmetic: each entry is the
/// Q32.32 word of the product, `round(x·v·2³²)` (one round-to-nearest, ties to even), which is the
/// `fold_fit::scale_columns` operation on the stored plane.
fn scale_columns(x: &mut [f64], o: usize, n: usize, v: &[f64]) {
    for row in 0..o {
        for col in 0..n {
            let a = i128::from(grid(x[row * n + col]));
            let b = i128::from(grid(v[col]));
            let w = round_q32_even(a * b).expect("product inside the grid");
            x[row * n + col] = from_grid(w);
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

/// A value on the Q32.32 grid: `round(x·2³²)` exists and is within the grid range.
fn grid(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("value inside the Q32.32 grid range")
}

/// The grid image of a value as the `f64` it denotes: `from_grid(to_grid(x))`.
fn from_grid_word(x: f64) -> f64 {
    from_grid(grid(x))
}

/// A value is on the affine `bits`-grid `[f16r(min), f16r(max)]` with `2^bits − 1` levels — the
/// `quant_c_u8` grid of the carrier — up to the Q32.32 boundary: `quant_c_u8` emits the level value
/// `lo + (q/levels)·span` in `f64`, and the pack stores its round-to-nearest grid image, so each
/// entry sits within half a grid quantum of its level. `tol` is that bound.
fn on_affine_grid(values: &[f64], bits: u32, tol: f64) -> bool {
    let lo = f16r(values.iter().copied().fold(f64::INFINITY, f64::min));
    let hi = f16r(values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let span = (hi - lo).max(1e-12);
    let levels = f64::from((1_u32 << bits) - 1);
    values.iter().all(|&v| {
        let q = (((v - lo) / span) * levels)
            .round_ties_even()
            .clamp(0.0, levels);
        (lo + (q / levels) * span - v).abs() <= tol * (1.0 + v.abs())
    })
}

#[test]
fn learn_fold_fold_fit_pack_recomposes_to_wp_bitwise() {
    let (o, n, _ws, _h, fit) = run();
    let mut rotated = fit.pack.s_rot.clone();
    scale_columns(&mut rotated, o, n, &fit.pack.dh);
    // Each helper boundary carries the Q32.32 stage image: `fold_fit` writes the `fwht` output back
    // to the grid before `compose` reads it, so the test does the same.
    let s_dh: Vec<f64> = fwht128(&rotated)
        .expect("fwht")
        .into_iter()
        .map(from_grid_word)
        .collect();
    let recomposed = compose(&s_dh, &fit.pack.rh, &fit.pack.ch, o, n).expect("compose");
    assert_eq!(recomposed.len(), o * n);
    // The pack is stored on the Q32.32 grid, so the recomposition check is word for word in that
    // domain: the same carrier operations run on both sides, so no tolerance is applied and the
    // quantum is the carrier's `2⁻³²`.
    for k in 0..o * n {
        assert_eq!(
            grid(recomposed[k]),
            grid(fit.wp[k]),
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
    // `plane_w` accumulates `(s·d → fwht → c) · r` in `f64` from the stored pack, while `Wp` is
    // the grid image of `compose` `(s·r)·c` taken from the *gridded* intermediate `s_dh`: the two
    // differ by (i) that one grid rounding, whose residue is scaled by `|rh·ch|` on the way out,
    // (ii) the `Wp` grid image itself, `½` quantum, and (iii) the association orders' `f64`
    // error. Bound: `½q·(1 + |rh·ch|) + 16·eps·(1 + |w|)`.
    let g = n / GROUP;
    let mut worst = 0.0_f64;
    for k in 0..o * n {
        let scale = fit.pack.rh[(k / n) * g + (k % n) / GROUP].abs() * fit.pack.ch[k % n].abs();
        let tol = 0.5 * Q32_QUANTUM * (1.0 + scale) + 16.0 * f64::EPSILON * (1.0 + w[k].abs());
        let err = (w[k] - fit.wp[k]).abs();
        worst = worst.max(err);
        assert!(
            err <= tol,
            "wp != plane_w(pack) at {k}: {} != {}",
            w[k],
            fit.wp[k]
        );
    }
    eprintln!("fold_fit Wp vs plane_w(pack): worst |Δ| = {worst:e}");
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
    // The carriers are the quantiser's level values stored as Q32.32 words: each sits within half a
    // grid quantum of its level, and `lo`/`hi` are themselves grid images, so the bound is four
    // quanta.
    let tol = 4.0 * Q32_QUANTUM;
    assert!(
        on_affine_grid(&fit.pack.dh, 5, tol),
        "dh off the 5-bit grid"
    );
    assert!(
        on_affine_grid(&fit.pack.ch, 6, tol),
        "ch off the 6-bit grid"
    );
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
/// refit alternation or the pack-candidate rule (the mutation control's sensor), and the migrated
/// Q32.32 fit's distance from them.
///
/// # The bound
///
/// The oracle (`ref_fold_fit`, the `f64` body at base `9f440192`) reproduces the recorded run to
/// `1e-9` relative. The migrated body is the oracle's Q32.32 image: at most `R = 64` carrier
/// roundings (`3` refit passes × ~12 plane writes, plus ~12 in the pack phase) each move an entry
/// by at most `2⁻³³` relative, the c/d normal systems square the Toeplitz Gram's conditioning
/// (≤ `2⁹` for `rho = 0.4`, `n = 256`), and the fit's `≤ 5` passes compose those: the stated bound
/// is `R·2⁻³³·2⁹·5 ≈ 1.9e-5` relative, taken as `2e-5·(1+|J|)`. The recorded pins are the
/// oracle's values, so the migration cannot move the sensor: a change in the alternation or the
/// pack rule moves `J` by `O(1)`, four orders above this bound.
#[test]
fn learn_fold_fold_fit_production_numbers_are_pinned() {
    let (o, n, ws, h, fit) = run();
    let oracle = ref_fold_fit(&ws, &h, o, n, None, &production_cfg()).expect("oracle fit");
    let jwp = wp_objective(&ws, &h, o, n, &fit.wp);
    let jwp_ref = wp_objective(&ws, &h, o, n, &oracle.wp);
    let close = |got: f64, want: f64| (got - want).abs() <= 1e-9 * (1.0 + want.abs());
    // The oracle is the recorded run.
    assert!(
        close(oracle.trace[0], 9.553_694_787_010_723e1),
        "oracle trace[0] = {}",
        oracle.trace[0]
    );
    assert!(
        close(oracle.trace[1], 4.359_351_242_410_637e1),
        "oracle trace[1] = {}",
        oracle.trace[1]
    );
    assert!(
        close(oracle.j_best, 4.359_351_242_410_637e1),
        "oracle j_best = {}",
        oracle.j_best
    );
    assert!(
        close(jwp_ref, 4.351_880_967_972_322e1),
        "oracle J(wp) = {jwp_ref}"
    );
    let wsum_ref: f64 = oracle.wp.iter().map(|x| x.abs()).sum();
    assert!(
        close(wsum_ref, 4.515_831_864_262e2),
        "oracle |wp| sum = {wsum_ref}"
    );
    // The migrated fit sits inside the derived grid bound of the oracle.
    let wsum: f64 = fit.wp.iter().map(|x| x.abs()).sum();
    let near = |got: f64, want: f64| (got - want).abs() <= 2e-5 * (1.0 + want.abs());
    for (got, want, what) in [
        (fit.trace[0], oracle.trace[0], "trace[0]"),
        (fit.trace[1], oracle.trace[1], "trace[1]"),
        (fit.j_best, oracle.j_best, "j_best"),
        (jwp, jwp_ref, "J(wp)"),
        (wsum, wsum_ref, "|wp| sum"),
    ] {
        assert!(
            near(got, want),
            "{what}: {got} vs oracle {want} beyond the derived grid bound"
        );
    }
}

/// The agreement oracle: the `f64` body of `fold_fit` at base `9f440192` (`fold_fit.rs` lines
/// 254–580), copied verbatim with the names prefixed. Test-only `f64` is allowed; nothing on the
/// crate's build path runs it.
const REF_EPS: f64 = 1e-12;

/// A candidate of the oracle's alternation.
struct RefCand {
    /// Candidate `J`.
    j: f64,
    /// Candidate tag.
    tag: FoldFitTag,
    /// Hadamard-domain code plane.
    s_rot: Vec<f64>,
    /// Per-group row scale.
    r: Vec<f64>,
    /// Per-column scale.
    c: Vec<f64>,
    /// Hadamard column scale.
    d: Vec<f64>,
}
fn ref_metric(h: &[f64], n: usize, ridge: f64) -> Vec<f64> {
    let mut hl = h.to_vec();
    let mut mean = 0.0_f64;
    for i in 0..n {
        mean += h[i * n + i];
    }
    mean /= n as f64;
    let add = ridge * mean;
    for i in 0..n {
        hl[i * n + i] += add;
    }
    hl
}

fn ref_scale_columns(x: &mut [f64], o: usize, n: usize, v: &[f64]) {
    for row in 0..o {
        for col in 0..n {
            x[row * n + col] *= v[col];
        }
    }
}

fn ref_parity_signs(d: &[f64], n: usize) -> Vec<f64> {
    let mut sign = vec![1.0_f64; n];
    for i in 0..n {
        if d[i] < 0.0 {
            sign[i] = -1.0;
        }
    }
    let nb = n / BLOCK;
    for m in 0..nb {
        let b0 = m * BLOCK;
        let mut negatives = 0_usize;
        for k in 0..BLOCK {
            if sign[b0 + k] < 0.0 {
                negatives += 1;
            }
        }
        if negatives % 2 == 1 {
            let mut min_i = 0_usize;
            let mut min_abs = f64::INFINITY;
            for k in 0..BLOCK {
                let a = d[b0 + k].abs();
                if a < min_abs {
                    min_abs = a;
                    min_i = k;
                }
            }
            sign[b0 + min_i] = -sign[b0 + min_i];
        }
    }
    sign
}

fn ref_fold_fit(
    ws: &[f64],
    h: &[f64],
    o: usize,
    n: usize,
    g_row: Option<&[f64]>,
    cfg: &FoldFitConfig,
) -> Result<FoldFit, FoldFitError> {
    if cfg.iters != 1 {
        return Err(FoldFitError::Unsupported);
    }
    if o == 0 || n == 0 {
        return Err(FoldFitError::ZeroDim);
    }
    if !n.is_multiple_of(GROUP) {
        return Err(FoldFitError::Width);
    }
    if ws.len() != o * n || h.len() != n * n {
        return Err(FoldFitError::ShapeMismatch);
    }
    if let Some(gr) = g_row {
        if gr.len() != o {
            return Err(FoldFitError::ShapeMismatch);
        }
    }
    if ws.iter().any(|x| !x.is_finite()) || h.iter().any(|x| !x.is_finite()) {
        return Err(FoldFitError::NonFinite);
    }
    if let Some(gr) = g_row {
        if gr.iter().any(|x| !x.is_finite()) {
            return Err(FoldFitError::NonFinite);
        }
    }

    let g = n / GROUP;
    let nb = n / BLOCK;
    let hl = ref_metric(h, n, cfg.ridge);
    // One metric product `Ws·Hl` for the whole fit: every `c`/`r`/`d` step below reuses it, so the
    // redundant per-step `mul_metric` calls collapse to one. Same refusal path as the first step's
    // internal product.
    let wshl = mul_metric(ws, &hl, o, n)?;
    let idx_zero = vec![0_u16; o * nb];

    // Step 2: the `gptq_e8` candidate (lines 843–860).
    let ws_rot = fwht128(ws)?;
    let h_gram = had_gram(h, n)?;
    let q = gptq_e8(&ws_rot, &h_gram, o, n, cfg.damp, true, true, cfg.e8_beam)?;
    let s_rot0 = q.s;
    let r0 = q.scale;
    let c0 = vec![1.0_f64; n];
    let d0 = vec![1.0_f64; n];
    let s0 = fwht128(&s_rot0)?;
    let wq0 = compose(&s0, &r0, &c0, o, n)?;
    let j0 = objective(ws, &wq0, &hl, o, n)?;

    let mut cands: Vec<RefCand> = vec![RefCand {
        j: j0,
        tag: FoldFitTag::Gptq,
        s_rot: s_rot0,
        r: r0,
        c: c0,
        d: d0,
    }];
    let mut trace = vec![j0];

    // Step 3: the single refit pass (`iters = 1`, lines 862–889). The inner loop works on owned
    // copies of the `gptq` candidate's plane; the candidate itself keeps the pristine values.
    let mut s_rot = cands[0].s_rot.clone();
    let mut s_cur = fwht128(&s_rot)?;
    let mut r_cur = cands[0].r.clone();
    let mut c_cur = cands[0].c.clone();
    let mut d_cur = cands[0].d.clone();
    for _ in 0..cfg.inner_iters {
        c_cur = c_step_wshl(ws, &hl, &wshl, &s_cur, &r_cur, o, n, g_row)?;
        let smean = c_cur.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
        for v in &mut c_cur {
            *v /= smean;
        }
        for v in &mut r_cur {
            *v *= smean;
        }
        r_cur = r_step_wshl(ws, &hl, &wshl, &s_cur, &c_cur, o, n)?;

        let d_new = d_step_wshl(ws, &hl, &wshl, &s_rot, &r_cur, &c_cur, o, n, g_row)?;
        let sign = ref_parity_signs(&d_new, n);
        ref_scale_columns(&mut s_rot, o, n, &sign);
        let mut group_max = vec![0.0_f64; g];
        for gi in 0..g {
            let mut mx = 0.0_f64;
            for k in 0..GROUP {
                let v = (d_new[gi * GROUP + k] * sign[gi * GROUP + k]).abs();
                if v > mx {
                    mx = v;
                }
            }
            group_max[gi] = mx.max(REF_EPS);
        }
        for i in 0..n {
            d_cur[i] = d_new[i] * sign[i] / group_max[i / GROUP];
        }
        for r in 0..o {
            for gi in 0..g {
                r_cur[r * g + gi] *= group_max[gi];
            }
        }

        let plane = PlaneRef {
            s: &s_rot,
            idx: &idx_zero,
            r: &r_cur,
            c: &c_cur,
            d: &d_cur,
        };
        let refined = e8_icm_step(ws, &hl, plane, o, n, cfg.sweeps, cfg.icm_ridge)?;
        s_rot = refined.s;
        let mut rotated = s_rot.clone();
        ref_scale_columns(&mut rotated, o, n, &d_cur);
        s_cur = fwht128(&rotated)?;
        r_cur = r_step_wshl(ws, &hl, &wshl, &s_cur, &c_cur, o, n)?;
    }
    let wq1 = compose(&s_cur, &r_cur, &c_cur, o, n)?;
    let j1 = objective(ws, &wq1, &hl, o, n)?;
    cands.push(RefCand {
        j: j1,
        tag: FoldFitTag::Refit0,
        s_rot,
        r: r_cur,
        c: c_cur,
        d: d_cur,
    });
    trace.push(j1);

    // Step 4: `best = argmin J`, first on a tie (line 910).
    let mut best = 0_usize;
    for i in 1..cands.len() {
        if cands[i].j < cands[best].j {
            best = i;
        }
    }
    let j_best = cands[best].j;
    let tag = cands[best].tag;
    let s_rot_best = cands[best].s_rot.clone();
    let c_best = cands[best].c.clone();
    let d_best = cands[best].d.clone();

    // The `bscale and had` arm (lines 932–968).
    let (dh, bits_d) = quant_c_u8(&d_best, 5)?;
    let mut rotated = s_rot_best.clone();
    ref_scale_columns(&mut rotated, o, n, &dh);
    let s_dh = fwht128(&rotated)?;
    let rh_hier = quant_r_hier(&r_step_wshl(ws, &hl, &wshl, &s_dh, &c_best, o, n)?, o, g)?;
    let bits_r = rh_hier.bits;
    let rh = rh_hier.rq;
    let c_ref = c_step_wshl(ws, &hl, &wshl, &s_dh, &rh, o, n, g_row)?;
    let (ch, bits_c) = quant_c_u8(&c_ref, 6)?;
    let mut wp = compose(&s_dh, &rh, &ch, o, n)?;
    let mut j_wp = objective(ws, &wp, &hl, o, n)?;
    let mut pack_s_rot = s_rot_best.clone();
    let mut pack_rh = rh;
    let mut pack_ch = ch;
    let mut pack_dh = dh;
    let mut s_rot_cur = s_rot_best;

    // The `e8_icm` re-quantisation candidate (lines 942–950).
    let plane = PlaneRef {
        s: &pack_s_rot,
        idx: &idx_zero,
        r: &pack_rh,
        c: &pack_ch,
        d: &pack_dh,
    };
    let s_rot_q = e8_icm_step(ws, &hl, plane, o, n, cfg.sweeps, cfg.icm_ridge)?.s;
    {
        let mut rotated = s_rot_q.clone();
        ref_scale_columns(&mut rotated, o, n, &pack_dh);
        let s_dh_q = fwht128(&rotated)?;
        let rh_q = quant_r_hier(&r_step_wshl(ws, &hl, &wshl, &s_dh_q, &pack_ch, o, n)?, o, g)?.rq;
        let ch_q = quant_c_u8(
            &c_step_wshl(ws, &hl, &wshl, &s_dh_q, &rh_q, o, n, g_row)?,
            6,
        )?
        .0;
        let wp_q = compose(&s_dh_q, &rh_q, &ch_q, o, n)?;
        let j_q = objective(ws, &wp_q, &hl, o, n)?;
        if j_q < j_wp {
            j_wp = j_q;
            wp = wp_q;
            s_rot_cur.clone_from(&s_rot_q);
            pack_s_rot = s_rot_q;
            pack_rh = rh_q;
            pack_ch = ch_q;
        }
    }

    // The un-`d` fallback candidate (lines 960–967).
    {
        let s_1 = fwht128(&s_rot_cur)?;
        let r_1 = r_step_wshl(ws, &hl, &wshl, &s_1, &c_best, o, n)?;
        let rh_1 = quant_r_hier(&r_1, o, g)?.rq;
        let ch_1 = quant_c_u8(&c_step_wshl(ws, &hl, &wshl, &s_1, &rh_1, o, n, g_row)?, 6)?.0;
        let wp_1 = compose(&s_1, &rh_1, &ch_1, o, n)?;
        if objective(ws, &wp_1, &hl, o, n)? < j_wp {
            wp = wp_1;
            pack_s_rot = s_rot_cur;
            pack_rh = rh_1;
            pack_ch = ch_1;
            pack_dh = vec![1.0_f64; n];
        }
    }

    // The winning pack must recompose to `Wp` exactly (the Python's `assert`, line 968).
    let mut rotated = pack_s_rot.clone();
    ref_scale_columns(&mut rotated, o, n, &pack_dh);
    let recomposed = compose(&fwht128(&rotated)?, &pack_rh, &pack_ch, o, n)?;
    for k in 0..o * n {
        if recomposed[k].to_bits() != wp[k].to_bits() {
            return Err(FoldFitError::PackMismatch);
        }
    }

    Ok(FoldFit {
        wp,
        pack: PackOut {
            s_rot: pack_s_rot,
            rh: pack_rh,
            ch: pack_ch,
            dh: pack_dh,
        },
        j_best,
        trace,
        bits_r,
        bits_c,
        bits_d,
        tag,
    })
}
