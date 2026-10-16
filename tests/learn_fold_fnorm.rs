//! `tests/learn_fold_fnorm.rs` — the damped Newton-Fisher RMSNorm-gain fit
//! ([`fnorm_fit`]): a fixed point when teacher and student hidden are identical,
//! planted-diagonal recovery with a monotonically decreasing readout KL (computed here with an
//! independent `f64` softmax-KL, not the body under test), the coh-prior path scoring positions
//! `1..T`, and the named refusals. Grade `tested`.
//!
//! Mutation control (see the lane return): `(ps[k] - pt[k])` in [`fnorm_fit`]'s gradient changed
//! to `(pt[k] - ps[k])` turns `planted_diagonal_moves_g_and_cuts_kl` red.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::unreadable_literal,
    reason = "the tests compare `f64` carriers and recompute the softmax-KL independently, index \
              the row-major `[n, t, hid]`/`[vocab, hid]` buffers by `s/p/j/k`, and name the carriers \
              `ut/us`, `pt/ps`, `mu`, `sec` as the reference does"
)]

use apeiron::learn::fold::fit::fnorm::{fnorm_fit, FnormError};
use apeiron::learn::fold::fit::quant::f16r;

/// Deterministic 64-bit LCG (Knuth's MMIX constants); the crate's test drift.
struct Lcg(u64);

impl Lcg {
    /// Seed the generator.
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next word.
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// A value uniform on `[-1, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 / 9_007_199_254_740_992.0) * 2.0 - 1.0
    }
}

/// `len` values uniform on `[-1, 1)` from a fixed seed.
fn lcg_vec(len: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| rng.unit()).collect()
}

/// Independent RMS normalization `u_j = h_j · rsqrt(mean(h²) + eps)`.
fn rms(h: &[f64], eps: f64) -> Vec<f64> {
    let msq = h.iter().map(|x| x * x).sum::<f64>() / (h.len() as f64);
    let inv = 1.0 / (msq + eps).sqrt();
    h.iter().map(|x| x * inv).collect()
}

/// Independent max-shifted softmax.
fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

/// Independent logits `z_k = Σ_j u_j · g_j · e_kj`.
fn logits(u: &[f64], g: &[f64], e: &[f64], hid: usize, vocab: usize) -> Vec<f64> {
    (0..vocab)
        .map(|k| (0..hid).map(|j| u[j] * g[j] * e[k * hid + j]).sum())
        .collect()
}

/// Mean readout `KL(pt ‖ ps)` over the scored tokens, with an independent softmax.
#[allow(
    clippy::too_many_arguments,
    reason = "the fixture and the channel-scaled gain; the dims are explicit as the port's are"
)]
fn readout_kl(
    ht: &[f64],
    hs: &[f64],
    g: &[f64],
    g0: &[f64],
    e: &[f64],
    n: usize,
    t: usize,
    hid: usize,
    vocab: usize,
    eps: f64,
    coh: bool,
) -> f64 {
    let start = usize::from(coh);
    let mut tot = 0.0_f64;
    let mut cnt = 0usize;
    for s in 0..n {
        for p in start..t {
            let base = (s * t + p) * hid;
            let ut = rms(&ht[base..base + hid], eps);
            let us = rms(&hs[base..base + hid], eps);
            let pt = softmax(&logits(&ut, g0, e, hid, vocab));
            let ps = softmax(&logits(&us, g, e, hid, vocab));
            let mut kl = 0.0_f64;
            for k in 0..vocab {
                kl += pt[k] * (pt[k].ln() - ps[k].ln());
            }
            tot += kl;
            cnt += 1;
        }
    }
    tot / (cnt as f64)
}

const N: usize = 3;
const T: usize = 4;
const HID: usize = 8;
const VOCAB: usize = 16;
const EPS: f64 = 1e-5;

/// The fixture's teacher/student hidden, gain and tied embedding.
fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ht = lcg_vec(N * T * HID, 0x1234_5678);
    let hs = lcg_vec(N * T * HID, 0x9abc_def0);
    let g0 = vec![1.0, 0.5, 1.5, 0.75, 2.0, 0.25, 1.0, 0.5];
    let e = lcg_vec(VOCAB * HID, 0x0f0f_1234);
    (ht, hs, g0, e)
}

#[test]
fn identical_hidden_at_the_start_keeps_g_fixed() {
    let (ht, _hs, g0, e) = fixture();
    let hs = ht.clone();
    // The start is f16-exact, so the per-step `f16r` round is the identity on it.
    for &x in &g0 {
        assert_eq!(f16r(x), x, "start channel {x} is not f16-exact");
    }
    let out = fnorm_fit(
        &ht, &hs, N, T, HID, &g0, &e, VOCAB, EPS, 8, 0.05, 1.0, false,
    )
    .unwrap();
    assert_eq!(out.len(), HID);
    assert_eq!(out, g0, "zero gradient must not move the gain");
}

/// The planted diagonal: hidden = equal rows `v`; student channels scaled by `s != 1`.
fn planted() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let v = [1.0_f64; HID];
    let s = [1.5_f64, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5];
    let mut ht = Vec::with_capacity(N * T * HID);
    let mut hs = Vec::with_capacity(N * T * HID);
    for _ in 0..N * T {
        for j in 0..HID {
            ht.push(v[j]);
            hs.push(v[j] * s[j]);
        }
    }
    let g0 = vec![1.0_f64; HID];
    let e = lcg_vec(VOCAB * HID, 0x7e57_1234);
    (ht, hs, g0, e)
}

#[test]
fn planted_diagonal_moves_g_and_cuts_kl() {
    let (ht, hs, g0, e) = planted();
    let steps = 200usize;
    let out = fnorm_fit(
        &ht, &hs, N, T, HID, &g0, &e, VOCAB, EPS, steps, 0.05, 0.3, false,
    )
    .expect("fit");

    // The student's normalized hidden is `diag(c·s)` times the teacher's, for the common scalar
    // `c = rsqrt(mean(hs²)+eps) / rsqrt(mean(ht²)+eps)`, so matching readout logits requires
    // `g = g0 ./ (c·s)` — the planted target `g0/s` up to `c`, computed here independently as
    // `g0·ut/us`. The ridge pull keeps the fitted gain between that point and `g0`, so every
    // moved channel keeps the planted direction.
    let ut0 = rms(&ht[0..HID], EPS);
    let us0 = rms(&hs[0..HID], EPS);
    for j in 0..HID {
        let gstar = g0[j] * ut0[j] / us0[j];
        assert!(
            (out[j] - g0[j]) * (gstar - g0[j]) > 0.0,
            "channel {j}: g {} vs g0 {} (planted target {gstar}) moved the wrong way",
            out[j],
            g0[j]
        );
    }

    // The readout KL falls monotonically; each prefix call re-runs the fit from the same start.
    let mut prev = readout_kl(&ht, &hs, &g0, &g0, &e, N, T, HID, VOCAB, EPS, false);
    let start = prev;
    for k in 1..=12usize {
        let gk = fnorm_fit(
            &ht, &hs, N, T, HID, &g0, &e, VOCAB, EPS, k, 0.05, 0.3, false,
        )
        .unwrap();
        let kl = readout_kl(&ht, &hs, &gk, &g0, &e, N, T, HID, VOCAB, EPS, false);
        assert!(kl < prev, "step {k}: KL did not fall {prev} -> {kl}");
        prev = kl;
    }
    assert!(prev < start, "KL did not fall: {start} -> {prev}");
}

#[test]
fn coh_prior_scores_positions_one_onward() {
    let (ht, hs, g0, e) = fixture();
    let a = fnorm_fit(&ht, &hs, N, T, HID, &g0, &e, VOCAB, EPS, 6, 0.05, 1.0, true).unwrap();
    assert_eq!(a.len(), HID);
    assert!(
        a.iter().all(|x| x.is_finite()),
        "coh-prior gain not finite: {a:?}"
    );

    // Perturb position 0 of every sequence on both hidden states: the coh-prior path drops it.
    let mut ht2 = ht.clone();
    let mut hs2 = hs.clone();
    for s in 0..N {
        let base = s * T * HID;
        for j in 0..HID {
            ht2[base + j] += 0.7;
            hs2[base + j] -= 1.3;
        }
    }
    let b = fnorm_fit(
        &ht2, &hs2, N, T, HID, &g0, &e, VOCAB, EPS, 6, 0.05, 1.0, true,
    )
    .unwrap();
    assert_eq!(a, b, "position 0 must not enter the coh-prior fit");
}

#[test]
fn refusals_are_reached() {
    let (ht, hs, g0, e) = fixture();
    let call = |ht: &[f64], hs: &[f64], g0: &[f64], e: &[f64], eps: f64, coh: bool| {
        fnorm_fit(ht, hs, N, T, HID, g0, e, VOCAB, eps, 4, 0.05, 1.0, coh)
    };

    // Shape mismatches: a short hidden, embedding or gain, a zero width.
    assert_eq!(
        call(&ht[1..], &hs, &g0, &e, EPS, false).unwrap_err(),
        FnormError::ShapeMismatch
    );
    assert_eq!(
        call(&ht, &hs, &g0[1..], &e, EPS, false).unwrap_err(),
        FnormError::ShapeMismatch
    );
    assert_eq!(
        call(&ht, &hs, &g0, &e[1..], EPS, false).unwrap_err(),
        FnormError::ShapeMismatch
    );
    assert_eq!(
        fnorm_fit(&[], &[], N, T, 0, &[], &[], VOCAB, EPS, 4, 0.05, 1.0, false).unwrap_err(),
        FnormError::ShapeMismatch
    );
    // coh_prior with a single position has no scored token.
    let flat_t = lcg_vec(N * HID, 0x5eed);
    assert_eq!(
        fnorm_fit(&flat_t, &flat_t, N, 1, HID, &g0, &e, VOCAB, EPS, 4, 0.05, 1.0, true)
            .unwrap_err(),
        FnormError::ShapeMismatch
    );

    // Non-finite inputs.
    let mut bad = ht.clone();
    bad[0] = f64::NAN;
    assert_eq!(
        call(&bad, &hs, &g0, &e, EPS, false).unwrap_err(),
        FnormError::NonFinite
    );
    let mut bad_e = e.clone();
    bad_e[0] = f64::INFINITY;
    assert_eq!(
        call(&ht, &hs, &g0, &bad_e, EPS, false).unwrap_err(),
        FnormError::NonFinite
    );

    // The RMSNorm epsilon must be finite and positive.
    assert_eq!(
        call(&ht, &hs, &g0, &e, 0.0, false).unwrap_err(),
        FnormError::BadEps
    );
    assert_eq!(
        call(&ht, &hs, &g0, &e, -1.0, false).unwrap_err(),
        FnormError::BadEps
    );
}
