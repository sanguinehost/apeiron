//! `tests/learn_fold_heldout.rs` — the held-out quality evaluator: identical models give zero KL and
//! TV and unit top-1 agreement, a perturbed `lm_head` row matches an independent `f64` softmax-KL
//! recomputation to `1e-6` relative, the mean over several sequences is the token-weighted mean of
//! the per-sequence values, and the named refusals hold. Grade `tested`.
//!
//! The fixture is a tiny one-layer `qwen2` [`Config`]/[`Weights`] built directly from the public
//! structs (the minimum of the `src/learn/grad/model/tests.rs` fixture pattern, which otherwise
//! serializes a GGUF and loads it through [`apeiron::engine::Engine`]).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    reason = "the test's fixed-seed LCG maps an integer word into a Q32.32 weight, and the f64 \
              softmax-KL reference is the independent recomputation the port is checked against; the \
              exact float comparisons are the point of the identical-model test (equal logits give 0 \
              and 1 bit-for-bit), which `assert_eq!` states directly; the pedantic lints stay on for \
              the crate"
)]

use apeiron::engine::ONE;
use apeiron::learn::fold::heldout::{held_out, held_out_seq};
use apeiron::learn::grad::model::{forward, Config, LayerWeights, Weights};

/// Embedding width, query heads, KV head width, feed-forward width, vocabulary, layers.
const E: usize = 4;
const NH: usize = 1;
const NKV: usize = 1;
const HD: usize = 2;
const QW: usize = NH * HD; // 2
const KVW: usize = NKV * HD; // 2
const FF: usize = 6;
const V: usize = 8;
const L: usize = 1;

/// A fixed-seed 64-bit LCG (Knuth's MMIX constants), the crate's test drift.
struct Lcg(u64);

impl Lcg {
    /// The next word.
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// A Q32.32 weight uniform on `[−0.5, 0.5)`, i.e. `[−ONE/2, ONE/2)`.
    fn weight(&mut self) -> i64 {
        let r = ((self.next() >> 33) & 0xffff) as i64; // 0..=65535
        (r - 32768) * (ONE >> 16)
    }

    /// `n` weights.
    fn vec(&mut self, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.weight()).collect()
    }
}

/// The tiny model geometry; `eps` is `round(2^-12 · ONE)`.
fn config() -> Config {
    Config {
        n_layer: L,
        n_head: NH,
        n_head_kv: NKV,
        n_embd: E,
        n_ff: FF,
        vocab: V,
        head_dim: HD,
        eps: ONE >> 12,
        rope_theta: 10_000.0,
    }
}

/// A one-layer Q32.32 model with `seed`'s weights and unit norm gains.
fn weights(seed: u64) -> Weights {
    let mut g = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
    let one = vec![ONE; E];
    let layers = (0..L)
        .map(|_| LayerWeights {
            attn_norm: one.clone(),
            wq: g.vec(QW * E),
            bq: Some(g.vec(QW)),
            wk: g.vec(KVW * E),
            bk: Some(g.vec(KVW)),
            wv: g.vec(KVW * E),
            bv: Some(g.vec(KVW)),
            wo: g.vec(E * QW),
            ffn_norm: one.clone(),
            w_gate: g.vec(FF * E),
            w_up: g.vec(FF * E),
            w_down: g.vec(E * FF),
        })
        .collect();
    Weights {
        layers,
        output_norm: one,
        token_embd: g.vec(V * E),
        lm_head: g.vec(V * E),
    }
}

/// The `f64` softmax of one row of real logits.
fn softmax_f64(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}

/// The independent reference: the mean forward `KL(teacher ‖ student)` in nats from the Q32.32
/// logits, computed in `f64`, one row at a time (`Σ_x p_t(x)·(ln p_t(x) − ln p_s(x)) / T`).
fn f64_kl_mean(teacher: &Weights, student: &Weights, cfg: &Config, toks: &[u32]) -> f64 {
    let (lt, _) = forward(teacher, cfg, toks).expect("teacher forward");
    let (ls, _) = forward(student, cfg, toks).expect("student forward");
    let t = toks.len();
    let v = cfg.vocab;
    let one = ONE as f64;
    let mut total = 0.0;
    for r in 0..t {
        let zt: Vec<f64> = lt[r * v..(r + 1) * v]
            .iter()
            .map(|&z| z as f64 / one)
            .collect();
        let zs: Vec<f64> = ls[r * v..(r + 1) * v]
            .iter()
            .map(|&z| z as f64 / one)
            .collect();
        let pt = softmax_f64(&zt);
        let ps = softmax_f64(&zs);
        for x in 0..v {
            total += pt[x] * (pt[x].ln() - ps[x].ln());
        }
    }
    total / t as f64
}

/// A white-box check that the fixture's logits are non-degenerate: the two models differ, so the
/// KL tests below are not vacuous.
#[test]
fn fixture_is_non_degenerate() {
    let cfg = config();
    let teacher = weights(1);
    let student = weights(2);
    let toks: [u32; 4] = [1, 2, 3, 4];
    let h = held_out_seq(&teacher, &student, &cfg, &toks).expect("eval");
    assert!(h.kl > 0.0, "distinct models must give positive KL");
    assert!((0.0..=1.0).contains(&h.tv));
    assert!((0.0..=1.0).contains(&h.top1));
    assert!(h.nll_teacher > 0.0 && h.nll_student > 0.0);
}

/// Teacher equal to student: zero KL, zero TV, unit top-1, equal NLLs — exactly, not approximately.
#[test]
fn identical_teacher_student_is_zero() {
    let cfg = config();
    let w = weights(1);
    let s1: [u32; 3] = [1, 2, 3];
    let s2: [u32; 4] = [4, 5, 6, 7];
    let seqs: [&[u32]; 2] = [&s1, &s2];
    let h = held_out(&w, &w, &cfg, &seqs).expect("eval");
    assert_eq!(h.tokens, 7);
    assert_eq!(h.kl, 0.0, "equal logits give 0 exactly");
    assert_eq!(h.tv, 0.0);
    assert_eq!(h.top1, 1.0);
    assert_eq!(h.nll_student, h.nll_teacher);
}

/// A perturbed `lm_head` row gives a positive KL that equals the independent `f64` softmax-KL
/// recomputation to `1e-6` relative. The mutation control swaps `kl_row`'s arguments, which breaks
/// this equality: `KL(p_t ‖ p_s) ≠ KL(p_s ‖ p_t)`.
#[test]
fn perturbed_lm_head_matches_f64_softmax_kl() {
    let cfg = config();
    let teacher = weights(1);
    let mut student = teacher.clone();
    for k in 0..E {
        student.lm_head[k] += 3 * (ONE >> 2);
    }
    let toks: [u32; 5] = [1, 2, 3, 4, 5];
    let h = held_out_seq(&teacher, &student, &cfg, &toks).expect("eval");
    assert!(h.kl > 0.0, "the perturbed student must give positive KL");
    let want = f64_kl_mean(&teacher, &student, &cfg, &toks);
    let rel = (h.kl - want).abs() / want.abs().max(1e-12);
    assert!(
        rel <= 1e-6,
        "KL {:.12} vs f64 reference {:.12} (relative {rel:e})",
        h.kl,
        want
    );
}

/// The pooled mean over several sequences is the token-weighted mean of the per-sequence values:
/// KL/TV/top-1 weighted by `T_i`, the NLL by the `T_i − 1` next-token positions.
#[test]
fn multi_sequence_mean_is_token_weighted() {
    let cfg = config();
    let teacher = weights(1);
    let student = weights(2);
    let s1: [u32; 3] = [1, 2, 3];
    let s2: [u32; 5] = [7, 6, 5, 4, 3];
    let seqs: [&[u32]; 2] = [&s1, &s2];
    let h = held_out(&teacher, &student, &cfg, &seqs).expect("pooled");
    let a = held_out_seq(&teacher, &student, &cfg, &s1).expect("s1");
    let b = held_out_seq(&teacher, &student, &cfg, &s2).expect("s2");
    assert_eq!(h.tokens, a.tokens + b.tokens);

    let n = (a.tokens + b.tokens) as f64;
    let wkl = (a.kl * a.tokens as f64 + b.kl * b.tokens as f64) / n;
    let wtv = (a.tv * a.tokens as f64 + b.tv * b.tokens as f64) / n;
    let wt1 = (a.top1 * a.tokens as f64 + b.top1 * b.tokens as f64) / n;
    let m = ((a.tokens - 1) + (b.tokens - 1)) as f64;
    let wnll_s =
        (a.nll_student * (a.tokens - 1) as f64 + b.nll_student * (b.tokens - 1) as f64) / m;
    let wnll_t =
        (a.nll_teacher * (a.tokens - 1) as f64 + b.nll_teacher * (b.tokens - 1) as f64) / m;
    assert!((h.kl - wkl).abs() <= 1e-12 * wkl.abs().max(1.0), "kl {h:?}");
    assert!((h.tv - wtv).abs() <= 1e-12 * wtv.abs().max(1.0), "tv {h:?}");
    assert!((h.top1 - wt1).abs() <= 1e-12, "top1 {h:?}");
    assert!(
        (h.nll_student - wnll_s).abs() <= 1e-12 * wnll_s.abs().max(1.0),
        "nll_student {h:?}"
    );
    assert!(
        (h.nll_teacher - wnll_t).abs() <= 1e-12 * wnll_t.abs().max(1.0),
        "nll_teacher {h:?}"
    );
}

/// Refusals: an empty sequence, a one-token sequence (no next-token position), a token id at or past
/// the vocabulary, and an empty sequence list.
#[test]
fn refusals() {
    let cfg = config();
    let w = weights(1);
    assert!(held_out_seq(&w, &w, &cfg, &[]).is_err(), "empty sequence");
    assert!(
        held_out_seq(&w, &w, &cfg, &[1]).is_err(),
        "one-token sequence"
    );
    assert!(
        held_out_seq(&w, &w, &cfg, &[1, V as u32]).is_err(),
        "token id == vocab"
    );
    assert!(
        held_out_seq(&w, &w, &cfg, &[1, V as u32 + 3]).is_err(),
        "token id > vocab"
    );
    let empty: [&[u32]; 0] = [];
    assert!(held_out(&w, &w, &cfg, &empty).is_err(), "no sequences");
}
