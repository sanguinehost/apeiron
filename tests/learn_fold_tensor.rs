//! `tests/learn_fold_tensor.rs` — the per-tensor assembly (`fit_tensor`, `fold_row_scale`,
//! `recon`): the stored planes recompose to `Wp`, the bit budget's split and the corrector bits, the
//! `-nb`/second-plane never worsening `J` on the `k` fixture, the `q`/`down` corrector's acceptance
//! predicate and its unconditional bit count, the `q`/`k`/`v` bias, the `q_mlp_trans` top rows, the
//! `-resq` row-scale fold and the shape/width/non-finite refusals. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the fixed-seed LCG maps integers to `f64` by division and the flat row-major index \
              arithmetic is the reference payload format; the equality checks are exact float \
              comparisons by construction (the same expressions the port evaluates), and the \
              `o/n/m/i/j` names are the reference's vocabulary"
)]

use apeiron::learn::fold::fit::corrector::{lowrank_corrector, Mode};
use apeiron::learn::fold::fit::fold_fit::{fold_fit, FoldFitConfig};
use apeiron::learn::fold::fit::steps::{compose, fwht128, objective};
use apeiron::learn::fold::layer::stage::metric;
use apeiron::learn::fold::layer::tensor::{
    fit_tensor, fold_row_scale, recon, BiasStats, SinkPlane, TensorError, TensorFit, TensorKind,
    TensorParams,
};

const N: usize = 256;
const O: usize = 4;
const MLP_O: usize = 8;
const K_ROWS: usize = 3;
const QUANT_RIDGE: f64 = 0.02;
/// Half a Q32.32 grid quantum, `2⁻³³`: the largest round-to-nearest error of one carrier step.
const GRID_HALF: f64 = 1.0 / 8_589_934_592.0;

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

/// `rg_q = quant_ridge · (n/896)`.
fn rg_q(n: usize) -> f64 {
    QUANT_RIDGE * (n as f64 / 896.0)
}

/// The production `FoldFitConfig` (`inner_iters` 3, `e8_beam` 4, `icm_ridge` 0.20, `iters` 1).
fn cfg(n: usize) -> FoldFitConfig {
    FoldFitConfig {
        iters: 1,
        ridge: rg_q(n),
        damp: 0.01,
        inner_iters: 3,
        e8_beam: 4,
        icm_ridge: 0.20,
        sweeps: 2,
    }
}

fn params(n: usize, qmlp_rows: usize) -> TensorParams {
    TensorParams {
        rg_q: rg_q(n),
        qmlp_rows,
        fit: cfg(n),
        nb_sweeps: 2,
        nb_rounds: 2,
    }
}

/// A correlated SPD Gram `Hm = XᵀX + I` (`X` is `64 × n` random), a stand-in for the calibration
/// Gram.
fn rand_spd(n: usize, rng: &mut Lcg) -> Vec<f64> {
    let r = 64_usize.min(n);
    let mut x = vec![0.0_f64; r * n];
    for v in &mut x {
        *v = rng.uniform(-1.0, 1.0);
    }
    let mut h = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0_f64;
            for k in 0..r {
                s += x[k * n + i] * x[k * n + j];
            }
            if i == j {
                s += 1.0;
            }
            h[i * n + j] = s;
            h[j * n + i] = s;
        }
    }
    h
}

fn rand_weight(o: usize, n: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..o * n).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// A weight whose row masses are exactly `masses`: `ws[i, j] = sqrt(masses[i]/n)`.
fn massed_weight(masses: &[f64], n: usize) -> Vec<f64> {
    let o = masses.len();
    let mut ws = vec![0.0_f64; o * n];
    for i in 0..o {
        let v = (masses[i] / n as f64).sqrt();
        for j in 0..n {
            ws[i * n + j] = v;
        }
    }
    ws
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// `max|recon − wp| / max|wp|`, and whether `recon == wp` bit-for-bit.
fn recon_deviation(fit: &TensorFit, o: usize, n: usize) -> (f64, bool) {
    let r = recon(&fit.planes, o, n).expect("planes reconstruct");
    let mut equal = true;
    let mut maxd = 0.0_f64;
    let mut maxw = 0.0_f64;
    for k in 0..fit.wp.len() {
        if r[k].to_bits() != fit.wp[k].to_bits() {
            equal = false;
        }
        maxd = maxd.max((r[k] - fit.wp[k]).abs());
        maxw = maxw.max(fit.wp[k].abs());
    }
    let rel = if maxw > 0.0 { maxd / maxw } else { maxd };
    (rel, equal)
}

/// The `_sink_store` rule: bit-for-bit, else within `1e-6` relative.
fn assert_recon_matches(fit: &TensorFit, o: usize, n: usize) -> bool {
    let (rel, equal) = recon_deviation(fit, o, n);
    assert!(
        equal || rel <= 1e-6,
        "recon != wp: relative {rel:e} (equal {equal}) for {:?}",
        fit.kind
    );
    equal
}

fn fit(
    kind: TensorKind,
    o: usize,
    n: usize,
    ws: &[f64],
    hm: &[f64],
    p: &TensorParams,
) -> TensorFit {
    fit_tensor(kind, ws, hm, o, n, None, p, None).expect("fit succeeds")
}

#[test]
fn every_kind_recomposes_to_wp() {
    let mut rng = Lcg::new(0x5eed_0001);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);
    let cases: [(TensorKind, usize); 7] = [
        (TensorKind::Q, O),
        (TensorKind::K, O),
        (TensorKind::V, O),
        (TensorKind::O, O),
        (TensorKind::Gate, MLP_O),
        (TensorKind::Up, MLP_O),
        (TensorKind::Down, O),
    ];
    for (kind, o) in cases {
        let ws = rand_weight(o, N, &mut rng);
        let f = fit(kind, o, N, &ws, &hm, &p);
        let equal = assert_recon_matches(&f, o, N);
        // The script's store check passes; report bitwise vs the `1e-6` fallback.
        let (rel, _) = recon_deviation(&f, o, N);
        assert!(equal || rel <= 1e-6);
    }
}

#[test]
fn bit_budget_split() {
    let mut rng = Lcg::new(0x5eed_0002);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);

    for kind in [TensorKind::K, TensorKind::V, TensorKind::O] {
        let ws = rand_weight(O, N, &mut rng);
        let f = fit(kind, O, N, &ws, &hm, &p);
        assert_eq!(f.bits.nsign, (2 * O * N) as u64, "{kind:?}");
        assert_eq!(f.bits.ncorr, 0, "{kind:?}");
        assert!(f.nb.is_some(), "{kind:?} has the -nb log");
        assert!(f.top_rows.is_none(), "{kind:?}");
    }

    for kind in [TensorKind::Gate, TensorKind::Up] {
        let ws = rand_weight(MLP_O, N, &mut rng);
        let f = fit(kind, MLP_O, N, &ws, &hm, &p);
        assert_eq!(f.bits.nsign, (MLP_O * N + K_ROWS * N) as u64, "{kind:?}");
        assert_eq!(f.bits.ncorr, 0, "{kind:?}");
        assert!(f.nb.is_none(), "{kind:?}");
    }

    for kind in [TensorKind::Q, TensorKind::Down] {
        let ws = rand_weight(O, N, &mut rng);
        let f = fit(kind, O, N, &ws, &hm, &p);
        assert_eq!(f.bits.nsign, (O * N) as u64, "{kind:?}");
        assert!(f.bits.ncorr > 0, "{kind:?} counts the corrector bits");
        assert!(f.nb.is_none(), "{kind:?}");
        assert!(f.top_rows.is_none(), "{kind:?}");
    }
}

#[test]
fn second_plane_never_worsens_j_on_k() {
    let mut rng = Lcg::new(0x5eed_0003);
    let hm = rand_spd(N, &mut rng);
    let ws = rand_weight(O, N, &mut rng);
    let p = params(N, K_ROWS);

    let f = fit(TensorKind::K, O, N, &ws, &hm, &p);
    let hl = metric(&hm, N, rg_q(N)).expect("metric");
    let wp1 = fold_fit(&ws, &hm, O, N, None, &cfg(N)).expect("Wp1 fit").wp;
    let j_two = objective(&ws, &f.wp, &hl, O, N).expect("J");
    let j_one = objective(&ws, &wp1, &hl, O, N).expect("J");
    assert!(
        j_two <= j_one,
        "second plane + nb worsened J: J(wp)={j_two:.6e} J(Wp1)={j_one:.6e}"
    );
}

#[test]
fn corrector_acceptance_predicate_and_bits() {
    let mut rng = Lcg::new(0x5eed_0004);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);
    let hl = metric(&hm, N, rg_q(N)).expect("metric");

    for (kind, rk) in [(TensorKind::Q, 2_usize), (TensorKind::Down, 1)] {
        let ws = rand_weight(O, N, &mut rng);
        let f = fit(kind, O, N, &ws, &hm, &p);

        // Recompute the pre-corrector fit and the corrector independently.
        let pre = fold_fit(&ws, &hm, O, N, None, &cfg(N)).expect("pre fit").wp;
        let delta: Vec<f64> = ws.iter().zip(&pre).map(|(a, b)| a - b).collect();
        let corr = lowrank_corrector(&delta, &hl, O, N, rk, Mode::U8, true).expect("corrector");
        let j_new = objective(&ws, &add(&pre, &corr.corr), &hl, O, N).expect("J");
        let j_old = objective(&ws, &pre, &hl, O, N).expect("J");
        let applied = f.planes.iter().any(|pl| pl.dense.is_some());

        assert_eq!(applied, j_new < j_old, "{kind:?}: acceptance predicate");
        // The corrector bits are counted whether or not it is applied.
        assert_eq!(f.bits.ncorr, corr.bits, "{kind:?}: ncorr");
        // The applied corrector never raises J.
        let j_wp = objective(&ws, &f.wp, &hl, O, N).expect("J");
        assert!(
            j_wp <= j_old,
            "{kind:?}: J(wp)={j_wp:.6e} > J(pre)={j_old:.6e}"
        );
        if applied {
            // The carrier's `Wp = reals(add_words(grids(Wp1), grids(Corr)))`: each entry is two
            // round-to-nearest grid conversions and one exact word add from the test's f64
            // `add(Wp1, Corr)`, so the derived bound is `2·2⁻³³ = 2⁻³²`.
            let want = add(&pre, &corr.corr);
            let mut max = 0.0_f64;
            for k in 0..f.wp.len() {
                max = max.max((f.wp[k] - want[k]).abs());
            }
            assert!(
                max <= 2.0 * GRID_HALF,
                "{kind:?}: |Wp - (Wp1+Corr)| {max:e} exceeds 2·2⁻³³"
            );
        }
    }
}

#[test]
fn down_corrector_rejected_still_counts_bits() {
    // `2e-4` is the carrier's fit floor: at `1e-12` every `ws` word rounds to `0` and `fold_fit`
    // refuses `NotPositiveDefinite`; below `1.26e-4` it still refuses, and at every legal scale the
    // U8 corrector lowers `J`. The unconditional count is therefore tested at the zero residual.
    let mut rng = Lcg::new(0x5eed_0005);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);
    let hl = metric(&hm, N, rg_q(N)).expect("metric");

    let mut r = Lcg::new(0x5eed_0005);
    let ws: Vec<f64> = (0..O * N).map(|_| 2e-4 * r.uniform(-1.0, 1.0)).collect();
    let f = fit_tensor(TensorKind::Down, &ws, &hm, O, N, None, &p, None).expect("fit");
    let applied = f.planes.iter().any(|pl| pl.dense.is_some());

    let pre = fold_fit(&ws, &hm, O, N, None, &cfg(N)).expect("pre").wp;
    let delta: Vec<f64> = ws.iter().zip(&pre).map(|(a, b)| a - b).collect();
    let corr = lowrank_corrector(&delta, &hl, O, N, 1, Mode::U8, true).expect("corrector");
    let j_new = objective(&ws, &add(&pre, &corr.corr), &hl, O, N).expect("J");
    let j_old = objective(&ws, &pre, &hl, O, N).expect("J");

    assert_eq!(applied, j_new < j_old, "acceptance predicate");
    assert!(corr.bits > 0, "the corrector spends stored-factor bits");
    // `ncorr += b_corr` sits outside the acceptance test (`:1884`), and `b_corr` is the stored
    // factors' structural count: a zero residual yields the zero `Corr` with the same bits.
    let zero = vec![0.0_f64; O * N];
    let z = lowrank_corrector(&zero, &hl, O, N, 1, Mode::U8, true).expect("zero corrector");
    assert_eq!(z.bits, corr.bits, "the bit count does not depend on Corr");
    assert_eq!(f.bits.ncorr, corr.bits, "ncorr counted even when rejected");
}

#[test]
fn bias_matches_hand_formula() {
    let mut rng = Lcg::new(0x5eed_0006);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);
    let n_toks = 37.0_f64;
    let x_sum: Vec<f64> = (0..N).map(|_| rng.uniform(-1.0, 1.0)).collect();

    for kind in [TensorKind::Q, TensorKind::K, TensorKind::V] {
        let o = O;
        let ws = rand_weight(o, N, &mut rng);
        let y_sum: Vec<f64> = (0..o).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let bias = BiasStats {
            x_sum: &x_sum,
            y_sum: &y_sum,
            n_toks,
        };
        let f = fit_tensor(kind, &ws, &hm, o, N, None, &p, Some(bias)).expect("fit");
        let db = f.db.expect("db");
        for r in 0..o {
            let (mut s, mut mag) = (0.0_f64, 0.0_f64);
            for i in 0..N {
                s += f.wp[r * N + i] * x_sum[i];
                mag += f.wp[r * N + i].abs() + x_sum[i].abs();
            }
            let hand = (y_sum[r] - s) / n_toks;
            // Each input reads onto the grid within GRID_HALF, the quotient rounds once more.
            let tol = GRID_HALF * (1.0 + mag) / n_toks + GRID_HALF + 2.0 * f64::EPSILON;
            assert!(
                (db[r] - hand).abs() <= tol,
                "{kind:?} row {r}: {} vs {hand}",
                db[r]
            );
        }
    }

    // A bias supplied for a kind without one is refused.
    let ws = rand_weight(O, N, &mut rng);
    let y_sum = vec![0.0_f64; O];
    let bad = fit_tensor(
        TensorKind::O,
        &ws,
        &hm,
        O,
        N,
        None,
        &p,
        Some(BiasStats {
            x_sum: &x_sum,
            y_sum: &y_sum,
            n_toks,
        }),
    );
    assert_eq!(bad.unwrap_err(), TensorError::BiasKind);
}

#[test]
fn qmlp_top_rows_and_boundaries() {
    let mut rng = Lcg::new(0x5eed_0007);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);

    // Ties at the top-k boundary break to the lower index: masses [1, 2, 2, 2, ...] -> [1, 2, 3].
    let masses = [1.0_f64, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5];
    let ws = massed_weight(&masses, N);
    let f = fit(TensorKind::Gate, MLP_O, N, &ws, &hm, &p);
    assert_eq!(f.top_rows.as_deref(), Some(&[1_usize, 2, 3][..]));
    assert_eq!(f.planes.len(), 2);
    assert_eq!(f.planes[1].rows.as_deref(), Some(&[1_usize, 2, 3][..]));

    // Descending order with the largest not first: [0.5, 3, 1, 2, ...] -> [1, 3, 2].
    let masses = [0.5_f64, 3.0, 1.0, 2.0, 0.1, 0.1, 0.1, 0.1];
    let ws = massed_weight(&masses, N);
    let f = fit(TensorKind::Up, MLP_O, N, &ws, &hm, &p);
    assert_eq!(f.top_rows.as_deref(), Some(&[1_usize, 3, 2][..]));

    // k_rows == 0: a single plane, no rows.
    let p0 = params(N, 0);
    let ws = rand_weight(MLP_O, N, &mut rng);
    let f = fit(TensorKind::Up, MLP_O, N, &ws, &hm, &p0);
    assert!(f.top_rows.is_none());
    assert_eq!(f.planes.len(), 1);
    assert_eq!(f.bits.nsign, (MLP_O * N) as u64);
    assert_recon_matches(&f, MLP_O, N);

    // k_rows > o is refused.
    let p9 = params(N, MLP_O + 1);
    assert_eq!(
        fit_tensor(TensorKind::Up, &ws, &hm, MLP_O, N, None, &p9, None).unwrap_err(),
        TensorError::QmlpRowsOutOfRange {
            k_rows: MLP_O + 1,
            o: MLP_O
        }
    );
}

#[test]
fn row_scale_fold() {
    let mut rng = Lcg::new(0x5eed_0008);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);
    let ws = rand_weight(MLP_O, N, &mut rng);
    let base = fit(TensorKind::Up, MLP_O, N, &ws, &hm, &p);

    let ones = vec![1.0_f64; MLP_O];
    let mut a = base.clone();
    fold_row_scale(&mut a, &ones, MLP_O, N).expect("ones fold");
    assert_eq!(recon(&a.planes, MLP_O, N).expect("recon"), a.wp);

    let mut var = vec![0.5_f64; MLP_O];
    for (i, v) in var.iter_mut().enumerate() {
        *v = 0.5 + 1.5 * (i as f64) / (MLP_O as f64 - 1.0);
    }
    let mut b = base.clone();
    fold_row_scale(&mut b, &var, MLP_O, N).expect("varying fold");
    assert_eq!(recon(&b.planes, MLP_O, N).expect("recon"), b.wp);
    assert_ne!(a.wp, b.wp, "a non-unit row scale must change Wp");

    // A non-`up` fit is refused.
    let fk = fit(TensorKind::K, O, N, &ws[..O * N], &hm, &p);
    let mut fk = fk;
    assert_eq!(
        fold_row_scale(&mut fk, &[1.0_f64; O], O, N).unwrap_err(),
        TensorError::NotUp
    );

    // A wrong-length `s_sol` is refused.
    let mut c = base.clone();
    assert_eq!(
        fold_row_scale(&mut c, &ones[..MLP_O - 1], MLP_O, N).unwrap_err(),
        TensorError::ShapeMismatch
    );
}

#[test]
fn refusals() {
    let mut rng = Lcg::new(0x5eed_0009);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);
    let ws = rand_weight(O, N, &mut rng);

    // Wrong `ws` length.
    assert_eq!(
        fit_tensor(TensorKind::Q, &ws[..O * N - 1], &hm, O, N, None, &p, None).unwrap_err(),
        TensorError::ShapeMismatch
    );
    // Wrong `hm` length.
    assert_eq!(
        fit_tensor(TensorKind::Q, &ws, &hm[..N * N - 1], O, N, None, &p, None).unwrap_err(),
        TensorError::ShapeMismatch
    );
    // Wrong `g_row` length.
    assert_eq!(
        fit_tensor(
            TensorKind::Q,
            &ws,
            &hm,
            O,
            N,
            Some(&[1.0_f64; O - 1]),
            &p,
            None
        )
        .unwrap_err(),
        TensorError::ShapeMismatch
    );
    // `n` not a multiple of 128.
    let hm100 = rand_spd(100, &mut rng);
    let ws100 = rand_weight(O, 100, &mut rng);
    assert_eq!(
        fit_tensor(TensorKind::Q, &ws100, &hm100, O, 100, None, &p, None).unwrap_err(),
        TensorError::Width
    );
    // Zero `o`.
    assert_eq!(
        fit_tensor(TensorKind::Q, &[], &hm, 0, N, None, &p, None).unwrap_err(),
        TensorError::ZeroDim
    );
    // A NaN weight.
    let mut nan_ws = ws.clone();
    nan_ws[0] = f64::NAN;
    assert_eq!(
        fit_tensor(TensorKind::Q, &nan_ws, &hm, O, N, None, &p, None).unwrap_err(),
        TensorError::NonFinite
    );
    // A NaN Gram.
    let mut nan_hm = hm.clone();
    nan_hm[0] = f64::INFINITY;
    assert_eq!(
        fit_tensor(TensorKind::Q, &ws, &nan_hm, O, N, None, &p, None).unwrap_err(),
        TensorError::NonFinite
    );
}

/// The reference `plane_weight` of the f64 implementation at `9f440192`, copied here as the oracle:
/// `rot = s_rot ⊙ dh`, the normalised Hadamard transform, `compose`, then a scatter-add at `rows`.
fn ref_plane_weight(p: &SinkPlane, o: usize, n: usize) -> Vec<f64> {
    if let Some(dense) = &p.dense {
        return dense.clone();
    }
    let (rlen, rows) = match &p.rows {
        Some(rs) => (rs.len(), Some(rs.as_slice())),
        None => (o, None),
    };
    let mut rot = p.s_rot.clone();
    for row in 0..rlen {
        for col in 0..n {
            rot[row * n + col] *= p.dh[col];
        }
    }
    let t = fwht128(&rot).expect("reference transform");
    let wi = compose(&t, &p.rh, &p.ch, rlen, n).expect("reference compose");
    match rows {
        None => wi,
        Some(rs) => {
            let mut full = vec![0.0_f64; o * n];
            for (i, &r) in rs.iter().enumerate() {
                for j in 0..n {
                    full[r * n + j] += wi[i * n + j];
                }
            }
            full
        }
    }
}

/// The reference `recon` of the same revision: `Σ_p plane_weight(p)`.
fn ref_recon(planes: &[SinkPlane], o: usize, n: usize) -> Vec<f64> {
    let mut w = vec![0.0_f64; o * n];
    for p in planes {
        let pw = ref_plane_weight(p, o, n);
        for k in 0..o * n {
            w[k] += pw[k];
        }
    }
    w
}

/// The derived per-entry bound between the carrier's `recon` and [`ref_recon`], from the plane
/// list's own magnitudes: a pack plane rounds `s_rot ⊙ dh` once per entry (`2⁻³³`), the normalised
/// Hadamard transform has `∞`-norm at most `√n` (it preserves the 2-norm), `compose` scales by at
/// most `max|rh|·max|ch|`, and the composed plane enters the grid once more (`2⁻³³`); a dense plane
/// is one conversion. [`GRID_HALF`] carries the two conversions of the summed result's operands.
fn recon_bound(planes: &[SinkPlane], n: usize) -> f64 {
    let mut b = 0.0_f64;
    for p in planes {
        let scale = |v: &[f64]| v.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        let amp = if p.dense.is_some() {
            1.0
        } else {
            (n as f64).sqrt() * scale(&p.rh) * scale(&p.ch)
        };
        b += GRID_HALF * (amp + 1.0);
    }
    b
}

/// (recon) the carrier's reconstruction against the f64 oracle on a two-plane `up` fit at the
/// derived [`recon_bound`].
#[test]
fn recon_matches_oracle_at_the_grid_bound() {
    let mut rng = Lcg::new(0x5eed_0009);
    let hm = rand_spd(N, &mut rng);
    let p = params(N, K_ROWS);
    let ws = rand_weight(MLP_O, N, &mut rng);
    let f = fit(TensorKind::Up, MLP_O, N, &ws, &hm, &p);
    assert_eq!(f.planes.len(), 2, "a full plane and a rows plane");

    let got = recon(&f.planes, MLP_O, N).expect("recon");
    let want = ref_recon(&f.planes, MLP_O, N);
    assert_eq!(got.len(), want.len());
    let bound = recon_bound(&f.planes, N);
    let mut max = 0.0_f64;
    for k in 0..got.len() {
        max = max.max((got[k] - want[k]).abs());
    }
    assert!(
        max <= bound,
        "recon max |carrier - oracle| {max:e} > {bound:e}"
    );
    eprintln!("(recon) max {max:e} (bound {bound:e})");
}
