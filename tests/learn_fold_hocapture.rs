//! Integration tests for the output-side output-gradient Gram capture
//! (`src/learn/fold/hocapture.rs`).
//!
//! Grade: `tested`. The positive assertions check symmetry and `Cholesky` positive-definiteness of
//! the finished `HO`, the per-linear scatter `Σ_t w_t g_t g_tᵀ` against an independent recomputation
//! from the tape and the taps, a central difference of the sampled-label cross-entropy for one
//! linear's output gradient, the additivity of the accumulator, the `‖x‖²` weighting law under an
//! exact scaling of the input, the identity of the tapped backward's weight gradients with
//! `grad::model::backward`'s, the softmax rule against `grad::head::kl_row`, the sampler's
//! determinism and support, and the refusals.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::needless_range_loop,
    reason = "test-only arithmetic on a small deterministic Q32.32 fixture and its f64 Gram; the \
              single-letter names are the geometry's symbols and the float comparisons assert the \
              exact f64 divisibility and the exact scaling law the accumulator implements"
)]

use apeiron::engine::ONE;
use apeiron::learn::fold::stats::hocapture::{
    accumulate_kind, backward_with_taps, cross_entropy_sum, logits_grad_ce, sample_targets,
    softmax_row, Hocapture, HocaptureError, Kind, KindStats, DEFAULT_EPS, N_KINDS,
};
use apeiron::learn::grad::head::kl_row;
use apeiron::learn::grad::model::{backward, forward, Config, LayerTape, LayerWeights, Weights};
use apeiron::learn::solve::q32::{from_grid, Fx};

#[path = "support/ridge_oracle.rs"]
mod ridge_oracle;

/// Residual width.
const E: usize = 4;
/// Query heads.
const NH: usize = 2;
/// Per-head width.
const HD: usize = 2;
/// `q_out`.
const QW: usize = NH * HD;
/// KV heads.
const NKV: usize = 1;
/// `kv_out`.
const KVW: usize = NKV * HD;
/// Feed-forward width.
const FF: usize = 3;
/// Vocabulary.
const V: usize = 8;
/// Trunk layers.
const L: usize = 2;

/// The tiny geometry every test uses, identical to `tests/learn_fold_capture.rs`.
fn cfg() -> Config {
    Config {
        n_layer: L,
        n_head: NH,
        n_head_kv: NKV,
        n_embd: E,
        n_ff: FF,
        vocab: V,
        head_dim: HD,
        eps: 42_950,
        rope_theta: 10_000.0,
    }
}

/// Deterministic Q32.32 values in `{-3, …, 3}` from a fixed `LCG`.
fn lcg(state: &mut u64) -> i64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let v = (*state >> 33) % 7;
    v as i64 - 3
}

/// `n` deterministic Q32.32 values.
fn buf(n: usize, state: &mut u64) -> Vec<i64> {
    (0..n).map(|_| lcg(state) * ONE).collect()
}

/// The tiny synthetic model.
fn model() -> Weights {
    let mut s = 12_345u64;
    let layer = |s: &mut u64| LayerWeights {
        attn_norm: buf(E, s),
        wq: buf(QW * E, s),
        bq: Some(buf(QW, s)),
        wk: buf(KVW * E, s),
        bk: Some(buf(KVW, s)),
        wv: buf(KVW * E, s),
        bv: Some(buf(KVW, s)),
        wo: buf(E * QW, s),
        ffn_norm: buf(E, s),
        w_gate: buf(FF * E, s),
        w_up: buf(FF * E, s),
        w_down: buf(E * FF, s),
    };
    Weights {
        layers: (0..L).map(|_| layer(&mut s)).collect(),
        output_norm: buf(E, &mut s),
        token_embd: buf(V * E, &mut s),
        lm_head: buf(V * E, &mut s),
    }
}

/// The sampled-label cross-entropy of `tokens` at the fixed `targets`, in Q32.32.
fn ce(w: &Weights, c: &Config, tokens: &[u32], targets: &[u32]) -> i64 {
    let (logits, _tape) = forward(w, c, tokens).unwrap();
    cross_entropy_sum(&logits, tokens.len(), c.vocab, targets).unwrap()
}

/// The reference's `GQ_IN_OF` (`fold_transfer.py:1098`): the linear input whose squared row norm
/// weights each kind's update, selected here directly from the tape rather than through the module.
fn input_of(layer: &LayerTape, kind: Kind) -> &[i64] {
    match kind {
        Kind::Q | Kind::K | Kind::V => &layer.ln1_out, // xa
        Kind::O => &layer.attn_out,                    // at
        Kind::Gate | Kind::Up => &layer.ln2_out,       // xf
        Kind::Down => &layer.m,                        // silu(g)*u
    }
}

/// The kind's output width, the reference's `chan[kind]`.
fn width_of(kind: Kind) -> usize {
    match kind {
        Kind::Q | Kind::O => QW,
        Kind::K | Kind::V => KVW,
        Kind::Gate | Kind::Up => FF,
        Kind::Down => E,
    }
}

/// The independent `Σ_t ‖x_t‖² g_t g_tᵀ`, `Σ_t ‖s_t‖⁴` and `Σ_t s_ti⁴` over a Q32.32 `t × w` input
/// and a Q32.32 `t × o` output gradient, in token-outermost order.
fn naive_stats(x: &[i64], g: &[i64], t: usize, w: usize, o: usize) -> (Vec<f64>, f64, Vec<f64>) {
    let mut ho = vec![0.0f64; o * o];
    let mut s4 = 0.0f64;
    let mut q = vec![0.0f64; o];
    for r in 0..t {
        let mut wt = 0.0f64;
        for i in 0..w {
            let v = x[r * w + i] as f64 / 4_294_967_296.0;
            wt += v * v;
        }
        let mut s2 = vec![0.0f64; o];
        for i in 0..o {
            let gi = g[r * o + i] as f64 / 4_294_967_296.0;
            s2[i] = wt * gi * gi;
        }
        for i in 0..o {
            for j in 0..o {
                let a = g[r * o + i] as f64 / 4_294_967_296.0;
                let b = g[r * o + j] as f64 / 4_294_967_296.0;
                ho[i * o + j] += wt * a * b;
            }
        }
        let n2: f64 = s2.iter().sum();
        s4 += n2 * n2;
        for i in 0..o {
            q[i] += s2[i] * s2[i];
        }
    }
    (ho, s4, q)
}

/// The `f64` view of Q32.32 words: the single conversion at the accumulator's boundary.
fn reals(w: &[i64]) -> Vec<f64> {
    w.iter().map(|&v| from_grid(v)).collect()
}

/// The symmetric mirror of the accumulator's upper-triangle-only `o × o` scatter, the form
/// [`Hocapture::finish`] writes.
fn mirror(upper: &[i64], o: usize) -> Vec<i64> {
    let mut out = vec![0i64; o * o];
    for i in 0..o {
        for j in i..o {
            out[i * o + j] = upper[i * o + j];
            out[j * o + i] = upper[i * o + j];
        }
    }
    out
}

/// The `f64` view of an [`Fx`] accumulator: `m·2^e`, the carrier's own single conversion.
fn from_fx(x: &Fx) -> f64 {
    x.m as f64 * 2f64.powi(x.e)
}

/// `2⁻⁶¹`, the relative quantum of the [`Fx`] carrier's 62-bit significand.
fn fx_quantum() -> f64 {
    2f64.powi(-61)
}

/// `b` equals `k·a` to within `steps` carrier roundings: the relative bound the [`Fx`] accumulation
/// carries (one 62-bit renormalisation per `from_i128` and one per addition).
fn fx_scaled(a: &Fx, b: &Fx, k: f64, steps: f64) -> bool {
    (from_fx(b) - k * from_fx(a)).abs() <= steps * fx_quantum() * from_fx(b).abs()
}

/// The per-term error bound of one `HO` entry `Σ_t round(round(w_t·g_i)·g_j)`: `w_t`'s own rounding
/// is scaled by `|g_i g_j|`, `round(w·g_i)`'s by `|g_j|`, and the entry's by `1`, each at most half a
/// grid quantum, so a term carries `0.5·quantum·(1 + gmax + gmax²)`.
fn ho_term_bound(gmax: f64) -> f64 {
    0.5 * quantum() * (1.0 + gmax + gmax * gmax)
}

/// The per-term error bound of one `s_ti² = round(round(w_t·g_ti)·g_ti)`: the same three roundings,
/// with no entry rounding to absorb them, so `quantum·(1 + gmax + gmax²)`.
fn s2_term_bound(gmax: f64) -> f64 {
    quantum() * (1.0 + gmax + gmax * gmax)
}

/// `2⁻³²`, the grid quantum of the Q32.32 carriers the accumulator stores.
fn quantum() -> f64 {
    2f64.powi(-32)
}

/// Element-wise closeness with a tolerance relative to the larger magnitude: the test's own
/// recomputation groups each product `w_t·g_i·g_j` differently from the accumulator's mirrored
/// write, which is one floating-point association apart (`(w g_i) g_j` against `(w g_j) g_i`), so
/// the comparison is a relative one and stays five orders below any change of formula.
fn close_rel(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length {} != {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(
            (x - y).abs() <= tol * scale,
            "index {i}: {x} != {y} (rel tol {tol})"
        );
    }
}

/// `HO` symmetric, `HO + εI` positive-definite, and the mean diagonal `1 + eps` after `finish`.
#[test]
fn ho_is_symmetric_psd_and_finishes_to_mean_diagonal_one() {
    let c = cfg();
    let w = model();
    let tokens = [1u32, 2, 3];
    let mut cap = Hocapture::new(&c, 2, 5).unwrap();
    cap.accumulate(&w, &c, &tokens).unwrap();
    let raw = cap.stats();
    assert_eq!(raw.len(), L);
    for ls in raw {
        for kind in Kind::ALL {
            let ks = ls.kind(kind);
            let o = width_of(kind);
            assert_eq!(ks.ho.len(), o * o);
            assert_eq!(ks.q.len(), o);
            assert_eq!(ks.samples, u64::try_from(2 * tokens.len()).unwrap());
            assert!(ks.s4.m >= 0);
            assert!(ks.q.iter().all(|v| v.m >= 0));
            // The raw accumulator carries the upper triangle only; `finish` mirrors it.
            for i in 0..o {
                for j in 0..i {
                    assert_eq!(ks.ho[i * o + j], 0, "raw lower triangle is not zero");
                }
            }
            // PSD: `HO + 1e-9·I` is positive-definite, so `Cholesky` succeeds.
            let mut a = reals(&mirror(&ks.ho, o));
            for i in 0..o {
                a[i * o + i] += 1e-9;
            }
            assert!(
                ridge_oracle::cholesky(&a, o).is_ok(),
                "{kind:?} raw HO + 1e-9 I is not positive-definite at layer o={o}"
            );
        }
    }
    let fin = cap.finish(DEFAULT_EPS).unwrap();
    for ls in &fin {
        for kind in Kind::ALL {
            let ks = ls.kind(kind);
            let o = width_of(kind);
            for i in 0..o {
                for j in 0..o {
                    assert_eq!(
                        ks.ho[i * o + j],
                        ks.ho[j * o + i],
                        "asymmetric after finish"
                    );
                }
            }
            let mut a = reals(&ks.ho);
            for i in 0..o {
                a[i * o + i] += 1e-9;
            }
            assert!(
                ridge_oracle::cholesky(&a, o).is_ok(),
                "{kind:?} finished HO is not PD"
            );
            let mut diag = 0.0f64;
            for i in 0..o {
                diag += from_grid(ks.ho[i * o + i]);
            }
            // The finish rounds the fixed-point division and the `eps` word: one quantum each.
            assert!(
                (diag / o as f64 - (1.0 + DEFAULT_EPS)).abs() < 2.0 * quantum(),
                "{kind:?} mean diagonal {}",
                diag / o as f64
            );
        }
    }
    // The raw scatter is left untouched by `finish`, so the caller can still form its own lambda.
    for (a, b) in raw.iter().zip(&fin) {
        for kind in Kind::ALL {
            assert_eq!(a.kind(kind).s4, b.kind(kind).s4);
            assert_eq!(a.kind(kind).q, b.kind(kind).q);
            assert_ne!(a.kind(kind).ho, b.kind(kind).ho);
        }
    }
}

/// K=1 with a fixed sampled token: the tapped `∂L/∂(down output)` equals the central difference of
/// the sampled-label cross-entropy through a perturbed `ffn_down` weight, equals the kernel's own
/// weight-gradient adjoint, and the accumulated `HO` is `w·g gᵀ`.
#[test]
fn down_tap_matches_finite_difference_and_outer_product() {
    let c = cfg();
    let w = model();
    let tokens = [3u32];
    let target = [2u32];
    // The step is `2^-7` real, chosen from the measured noise floor: the loss's `fixed_exp`/`fixed_ln`
    // path carries an absolute error of order `1e-8` in the cross-entropy, so a step of `2^-7` puts
    // the signal (`2h·∂CE/∂W ≈ 8e-3·0.026`) about four orders above that floor and four times above
    // the observed `±0.2 %` scatter of the quotient. It is `0.26 %` of a unit weight here.
    let h = ONE >> 7;
    let (logits, tape) = forward(&w, &c, &tokens).unwrap();
    let mut dlogits = vec![0i64; c.vocab];
    logits_grad_ce(&logits, &target, c.vocab, &mut dlogits).unwrap();
    let (grads, taps) = backward_with_taps(&w, &c, &tape, &dlogits).unwrap();

    let lt = &tape.layers[0];
    let g = taps.get(0, Kind::Down).unwrap();
    assert_eq!(g.len(), E);
    let mut checked = 0usize;
    let mut skipped = 0usize;
    for o in 0..E {
        for i in 0..FF {
            let idx = o * FF + i;
            let pred = (g[o] as f64) * (lt.m[i] as f64) / (ONE as f64) / (ONE as f64);
            // The same product in Q32.32 units is what the kernel's weight gradient rounds to, so
            // this pins every entry of the tap to the adjoint the kernel actually used.
            let kernel = grads.layers[0].w_down[idx] as f64;
            let pred_q32 = (g[o] as f64) * (lt.m[i] as f64) / (ONE as f64);
            assert!(
                (kernel - pred_q32).abs() <= 1.0,
                "down[{o}][{i}] kernel {kernel} != tap·m {pred_q32}"
            );
            // The quotient's noise floor is ~2e-5 here (the cross-entropy's own `fixed_exp`/
            // `fixed_ln` error divided by `2h`), so a pair whose product is below 1e-3 carries no
            // signal; the count of resolvable pairs is asserted below.
            if pred.abs() <= 1e-3 {
                skipped += 1;
                continue;
            }
            let base = w.layers[0].w_down[idx];
            let mut wp = w.clone();
            wp.layers[0].w_down[idx] = base + h;
            let mut wm = w.clone();
            wm.layers[0].w_down[idx] = base - h;
            let fd = (ce(&wp, &c, &tokens, &target) - ce(&wm, &c, &tokens, &target)) as f64
                / (2.0 * h as f64);
            assert!(
                (fd - pred).abs() <= 1e-2 * pred.abs() + 1e-6,
                "down[{o}][{i}] fd = {fd} tap·m = {pred}"
            );
            checked += 1;
        }
    }
    assert!(
        checked >= E,
        "only {checked} of {} pairs were above the loss's noise floor ({skipped} skipped)",
        E * FF
    );

    // HO for one token and one draw is exactly `w·g gᵀ` with `w = ‖m_t‖²`.
    let mut cap = Hocapture::new(&c, 1, 0).unwrap();
    cap.accumulate_with_targets(&w, &c, &tokens, &target)
        .unwrap();
    let ks = cap.stats()[0].kind(Kind::Down);
    let mut wt = 0.0f64;
    for i in 0..FF {
        let v = lt.m[i] as f64 / 4_294_967_296.0;
        wt += v * v;
    }
    let gf: Vec<f64> = g.iter().map(|&v| from_grid(v)).collect();
    let outer: Vec<f64> = (0..E * E).map(|k| wt * gf[k / E] * gf[k % E]).collect();
    // One token and one draw: `HO_ij` is `round(round(w·g_i)·g_j)`, so `w`'s rounding is scaled by
    // `|g_i g_j|`, the inner rounding by `|g_j|` and the entry's own by one, each at most half a
    // quantum.
    let gmax = gf.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    close_rel(&reals(&mirror(&ks.ho, E)), &outer, ho_term_bound(gmax));
    assert_eq!(ks.samples, 1);
}

/// The scatter equals an independent `Σ_t ‖x_t‖² g_t g_tᵀ` on the real path, and the accumulator adds.
#[test]
fn ho_matches_the_independent_scatter_and_adds_across_calls() {
    let c = cfg();
    let w = model();
    let tokens = [4u32, 5, 6, 7];
    let target = [1u32, 0, 3, 2];
    let (logits, tape) = forward(&w, &c, &tokens).unwrap();
    let mut dlogits = vec![0i64; tokens.len() * c.vocab];
    logits_grad_ce(&logits, &target, c.vocab, &mut dlogits).unwrap();
    let (_grads, taps) = backward_with_taps(&w, &c, &tape, &dlogits).unwrap();

    let mut cap = Hocapture::new(&c, 1, 0).unwrap();
    cap.accumulate_with_targets(&w, &c, &tokens, &target)
        .unwrap();
    for l in 0..L {
        for kind in Kind::ALL {
            let o = width_of(kind);
            let (ho, s4, q) = naive_stats(
                input_of(&tape.layers[l], kind),
                taps.get(l, kind).unwrap(),
                tokens.len(),
                input_of(&tape.layers[l], kind).len() / tokens.len(),
                o,
            );
            let ks = cap.stats()[l].kind(kind);
            // `HO_ij = Σ_t round(round(w·g_i)·g_j)`: `w_t`'s rounding is scaled by `|g_i g_j|`, the
            // inner rounding by `|g_j|` and the entry's own by `1`, each at most half a quantum.
            let gmax = taps
                .get(l, kind)
                .unwrap()
                .iter()
                .fold(0.0f64, |m, &v| m.max(from_grid(v).abs()));
            let t = tokens.len() as f64;
            close_rel(&reals(&mirror(&ks.ho, o)), &ho, t * ho_term_bound(gmax));
            // `s4 = Σ_t (Σ_i s_ti²)²` with `s_ti² = round(round(w·g_ti)·g_ti)`: the same three
            // roundings give `|s_ti² − w·g_ti²| ≤ δ = (1 + gmax + gmax²)` half-quanta, so
            // `|n2_t − N_t| ≤ q·δ` with `N_t = Σ_i w g_ti²` and `δs4 ≤ Σ_t (2·N_t + δ')·δ'` with
            // `δ' = q·δ`. Cauchy–Schwarz on the non-negative `N_t` bounds `Σ_t N_t ≤ √(t·Σ_t N_t²)`;
            // the `Fx` carrier adds one 62-bit renormalisation for the call's exact total and one for
            // the fold, relative `2·2⁻⁶¹`.
            let delta = o as f64 * s2_term_bound(gmax);
            let s4_bound = 2.0 * (t * s4.abs()).sqrt() * delta
                + t * delta * delta
                + 2.0 * fx_quantum() * s4.abs();
            assert!(
                (from_fx(&ks.s4) - s4).abs() <= s4_bound,
                "{kind:?} s4 {} vs {s4} (bound {s4_bound})",
                from_fx(&ks.s4)
            );
            // The same derivation per coordinate, one rounding of `s_ti²` at a time:
            // `δq_i ≤ 2·√(t·q_i)·δ + t·δ²` with `δ = (1 + gmax + gmax²)` half-quanta.
            let dq = s2_term_bound(gmax);
            for i in 0..o {
                let q_bound = 2.0 * (t * q[i].abs()).sqrt() * dq
                    + t * dq * dq
                    + 2.0 * fx_quantum() * q[i].abs();
                assert!(
                    (from_fx(&ks.q[i]) - q[i]).abs() <= q_bound,
                    "{kind:?} q[{i}] {} vs {} (bound {q_bound})",
                    from_fx(&ks.q[i]),
                    q[i]
                );
            }
        }
    }

    // Two identical calls in one accumulator are twice one call, up to the association of the
    // accumulating additions (the second pass adds the same rank-1 terms to a larger running sum).
    let mut once = Hocapture::new(&c, 2, 7).unwrap();
    once.accumulate(&w, &c, &tokens).unwrap();
    let mut twice = Hocapture::new(&c, 2, 7).unwrap();
    twice.accumulate(&w, &c, &tokens).unwrap();
    twice.accumulate(&w, &c, &tokens).unwrap();
    for l in 0..L {
        for kind in Kind::ALL {
            let a = once.stats()[l].kind(kind);
            let b = twice.stats()[l].kind(kind);
            // The accumulator is integral, so adding the same rounded terms twice is exactly `2×`;
            // `s4`/`q` are renormalised once per call and once per fold in the `Fx` carrier, so their
            // relation is `2×` to within those roundings.
            let doubled: Vec<i64> = a.ho.iter().map(|v| 2 * v).collect();
            assert_eq!(doubled, b.ho, "{kind:?} ho is not exactly doubled");
            assert!(
                fx_scaled(&a.s4, &b.s4, 2.0, 2.0),
                "{kind:?} s4 {} is not twice {}",
                from_fx(&b.s4),
                from_fx(&a.s4)
            );
            for i in 0..a.q.len() {
                assert!(
                    fx_scaled(&a.q[i], &b.q[i], 2.0, 2.0),
                    "{kind:?} q[{i}] {} is not twice {}",
                    from_fx(&b.q[i]),
                    from_fx(&a.q[i])
                );
            }
            assert_eq!(b.samples, 2 * a.samples);
        }
    }
}

/// Scaling the input by `s` scales `HO` by `s²` and `s4`/`Q` by `s⁴`, exactly for `s = 2`.
#[test]
fn input_scaling_scales_ho_by_the_squared_norm_ratio() {
    let mut s = 99u64;
    let t = 3usize;
    let w = 4usize;
    let o = 3usize;
    let x = buf(t * w, &mut s);
    let g = buf(t * o, &mut s);
    let scaled: Vec<i64> = x.iter().map(|v| 2 * v).collect();
    assert!(x.iter().any(|&v| v != 0));
    let mut base = KindStats::zeros(o).unwrap();
    accumulate_kind(&mut base, &x, &g, t, w, o).unwrap();
    let mut dou = KindStats::zeros(o).unwrap();
    accumulate_kind(&mut dou, &scaled, &g, t, w, o).unwrap();
    assert_eq!(base.samples, u64::try_from(t).unwrap());
    for (a, b) in base.ho.iter().zip(&dou.ho) {
        assert_eq!(*a * 4, *b, "HO must scale by 2²");
    }
    // `s4`/`Q` scale by `2⁴`; the `Fx` carrier renormalises once per call, so the law holds to that
    // one rounding.
    assert!(
        fx_scaled(&base.s4, &dou.s4, 16.0, 1.0),
        "s4 {} must scale by 2⁴ from {}",
        from_fx(&dou.s4),
        from_fx(&base.s4)
    );
    for i in 0..base.q.len() {
        assert!(
            fx_scaled(&base.q[i], &dou.q[i], 16.0, 1.0),
            "Q[{i}] {} must scale by 2⁴ from {}",
            from_fx(&dou.q[i]),
            from_fx(&base.q[i])
        );
    }
    assert!(
        base.ho.iter().any(|&v| v != 0),
        "the fixture is not vacuous"
    );
}

/// The tapped backward returns exactly `grad::model::backward`'s weight gradients.
#[test]
fn taps_reproduce_the_backward_and_leave_the_tape_untouched() {
    let c = cfg();
    let w = model();
    let tokens = [2u32, 0, 6];
    let (logits, tape) = forward(&w, &c, &tokens).unwrap();
    let before = tape.clone();
    let targets = sample_targets(&logits, tokens.len(), c.vocab, 3, 11).unwrap();
    let mut dlogits = vec![0i64; tokens.len() * c.vocab];
    logits_grad_ce(&logits, &targets[..tokens.len()], c.vocab, &mut dlogits).unwrap();
    let (grads, taps) = backward_with_taps(&w, &c, &tape, &dlogits).unwrap();
    assert_eq!(grads, backward(&w, &c, &tape, &dlogits).unwrap());
    assert_eq!(tape, before);
    assert_eq!(taps.layers.len(), L);
    for l in 0..L {
        assert_eq!(taps.get(l, Kind::Q).unwrap().len(), tokens.len() * QW);
        assert_eq!(taps.get(l, Kind::K).unwrap().len(), tokens.len() * KVW);
        assert_eq!(taps.get(l, Kind::V).unwrap().len(), tokens.len() * KVW);
        assert_eq!(taps.get(l, Kind::O).unwrap().len(), tokens.len() * QW);
        assert_eq!(taps.get(l, Kind::Gate).unwrap().len(), tokens.len() * FF);
        assert_eq!(taps.get(l, Kind::Up).unwrap().len(), tokens.len() * FF);
        assert_eq!(taps.get(l, Kind::Down).unwrap().len(), tokens.len() * E);
    }
    assert!(taps.get(L, Kind::Down).is_none());
}

/// `softmax_row` is `head`'s softmax: `kl_row` against a zero teacher row returns `p_s − p_t`, and
/// `p_t` is the uniform `round(ONE/v)`.
#[test]
fn softmax_matches_head() {
    let v = V;
    // `softmax_row(zeros)`: `e_i = ONE`, `S = v·ONE`, so `p_t` is `round_div(ONE·ONE, v·ONE)`.
    // `v = 8` divides `ONE` exactly, so that is `ONE / v`.
    let p_uniform = ONE / i64::try_from(v).unwrap();
    let rows: [Vec<i64>; 3] = [
        vec![0, ONE / 2, -ONE, 2 * ONE, 0, 0, 0, 0],
        vec![-ONE, -ONE / 3, 0, ONE, 3 * ONE, 0, ONE, -ONE],
        vec![ONE, ONE, ONE, ONE, ONE, ONE, ONE, ONE],
    ];
    for z in rows {
        let p = softmax_row(&z).unwrap();
        let zeros = vec![0i64; v];
        let (_loss, dz) = kl_row(&z, &zeros).unwrap();
        for i in 0..v {
            let head_p = i128::from(dz[i]) + i128::from(p_uniform);
            assert_eq!(i128::from(p[i]), head_p, "softmax row {i}");
        }
        let sum: i128 = p.iter().map(|&x| i128::from(x)).sum();
        assert!((sum - i128::from(ONE)).abs() <= i128::try_from(v).unwrap());
    }
    assert!(matches!(softmax_row(&[]), Err(HocaptureError::Shape(_))));
}

/// The sampler is deterministic in `(logits, K, seed)`, stays inside the support, and follows a
/// peaked row.
#[test]
fn sampler_is_deterministic_and_in_support() {
    let c = cfg();
    let w = model();
    let tokens = [1u32, 5];
    let (logits, _tape) = forward(&w, &c, &tokens).unwrap();
    let a = sample_targets(&logits, 2, c.vocab, 4, 12345).unwrap();
    let b = sample_targets(&logits, 2, c.vocab, 4, 12345).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 8);
    let d = sample_targets(&logits, 2, c.vocab, 4, 54321).unwrap();
    assert_ne!(a, d, "two seeds drew the same eight targets");
    for r in 0..2 {
        let p = softmax_row(&logits[r * c.vocab..(r + 1) * c.vocab]).unwrap();
        for j in 0..4 {
            let t = a[j * 2 + r] as usize;
            assert!(t < c.vocab);
            assert!(
                p[t] > 0,
                "draw {j} at token {r} landed on a zero-probability entry"
            );
        }
    }
    // A peaked row: the inverse CDF can only land on the peak.
    let mut peak = vec![0i64; c.vocab];
    peak[c.vocab - 1] = 10 * ONE;
    let picked = sample_targets(&peak, 1, c.vocab, 8, 777).unwrap();
    assert!(picked
        .iter()
        .all(|&t| t == u32::try_from(c.vocab - 1).unwrap()));
    assert_eq!(sample_targets(&peak, 1, c.vocab, 8, 777).unwrap(), picked);
}

/// Refusals: a zero draw count, a geometry mismatch, a bad target list, an empty accumulator's
/// finish, and the shape refusals of the pieces.
#[test]
fn refusals_are_reached() {
    let c = cfg();
    let w = model();
    let tokens = [1u32, 2];
    let (logits, tape) = forward(&w, &c, &tokens).unwrap();

    assert!(matches!(
        Hocapture::new(&c, 0, 0),
        Err(HocaptureError::Shape(_))
    ));
    // A zero model dimension and a `q_out` that leaves `usize` are the model's own refusals, wrapped.
    let mut zero = cfg();
    zero.n_embd = 0;
    assert!(matches!(
        Hocapture::new(&zero, 1, 0),
        Err(HocaptureError::Model(_))
    ));
    let mut big = cfg();
    big.n_head = usize::MAX;
    assert!(matches!(
        Hocapture::new(&big, 1, 0),
        Err(HocaptureError::Model(_))
    ));
    assert!(matches!(
        sample_targets(&logits, 2, c.vocab, 0, 1),
        Err(HocaptureError::Shape(_))
    ));
    assert!(matches!(
        sample_targets(&logits, 2, 0, 1, 1),
        Err(HocaptureError::Shape(_))
    ));
    assert!(matches!(
        sample_targets(&logits, 3, c.vocab, 1, 1),
        Err(HocaptureError::Shape(_))
    ));

    let mut cap = Hocapture::new(&c, 1, 0).unwrap();
    // The mean over no samples is undefined.
    assert!(matches!(
        cap.finish(DEFAULT_EPS),
        Err(HocaptureError::Shape(_))
    ));
    // A target list that is not `k·t`.
    assert!(matches!(
        cap.add_taps(&w, &c, &logits, &tape, &[0u32; 3]),
        Err(HocaptureError::Shape(_))
    ));
    // A target past the vocabulary, and a logits buffer that is not `t·vocab`.
    assert!(matches!(
        cap.add_taps(&w, &c, &logits, &tape, &[u32::try_from(V).unwrap(), 0]),
        Err(HocaptureError::Shape(_))
    ));
    assert!(matches!(
        cap.add_taps(&w, &c, &logits[1..], &tape, &[0u32, 0]),
        Err(HocaptureError::Shape(_))
    ));
    // A config with different geometry than the accumulator was built for.
    let mut other = cfg();
    other.n_ff = FF + 1;
    let mut bad_w = w.clone();
    bad_w.layers[0].w_gate = buf((FF + 1) * E, &mut 1u64);
    assert!(matches!(
        cap.add_taps(&bad_w, &other, &logits, &tape, &[0u32, 0]),
        Err(HocaptureError::Shape(_))
    ));
    // A tape with the wrong layer count, through the tapped backward.
    let mut short = tape.clone();
    short.layers.pop();
    let mut dlogits = vec![0i64; 2 * c.vocab];
    logits_grad_ce(&logits, &[0u32, 1], c.vocab, &mut dlogits).unwrap();
    assert!(matches!(
        backward_with_taps(&w, &c, &short, &dlogits),
        Err(HocaptureError::Shape(_))
    ));
    assert!(matches!(
        backward_with_taps(&w, &c, &tape, &dlogits[1..]),
        Err(HocaptureError::Shape(_))
    ));
    // A logits gradient for an empty target list, and an accumulator of the wrong width.
    assert!(matches!(
        logits_grad_ce(&logits, &[], c.vocab, &mut []),
        Err(HocaptureError::Shape(_))
    ));
    let mut ks = KindStats::zeros(FF).unwrap();
    assert!(matches!(
        accumulate_kind(&mut ks, &[0i64], &[0i64], 1, 1, E),
        Err(HocaptureError::Shape(_))
    ));
    // A well-shaped gradient whose `o` disagrees with the accumulator's own width.
    assert!(matches!(
        accumulate_kind(&mut ks, &[0i64; E], &[0i64; E], 1, E, E),
        Err(HocaptureError::Shape(_))
    ));
    // A zero output width.
    let mut empty = KindStats::zeros(0).unwrap();
    assert!(matches!(
        accumulate_kind(&mut empty, &[0i64], &[0i64], 1, 1, 0),
        Err(HocaptureError::Shape(_))
    ));
    assert_eq!(Kind::ALL.len(), N_KINDS);
}
