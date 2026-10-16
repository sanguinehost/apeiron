//! Gate for `apeiron::learn::fold::run::convert`: the `Q32.32 ↔ f64` round trip, the forward
//! agreement, and the typed refusals.
//!
//! A tiny `Qwen2` is built by a fixed LCG (`hidden` 128, `n_head` 2, `head_dim` 64, `n_kv_head` 1,
//! `inter` 128, `vocab` 32, `n_layer` 2) and run through both directions of the conversion. Every
//! value is `tested` here; nothing is `proved`.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "the inline LCG yields unsigned words and casts them into a small integer/float range; \
              the wrap and the precision loss are the generator's, not a measured quantity"
)]

use apeiron::learn::fold::run::convert::{
    f64_to_q32, q32_to_f64, student_to_grad, teacher_from_grad, ConvertError, Teacher,
};
use apeiron::learn::grad::model::{self, Config, LayerWeights as GradLayer, Weights};

const HID: usize = 128;
const NH: usize = 2;
const HD: usize = 64;
const NKV: usize = 1;
const INTER: usize = 128;
const VOCAB: usize = 32;
const NLAYER: usize = 2;
const Q_OUT: usize = NH * HD; // 128
const KV_OUT: usize = NKV * HD; // 64

/// A 64-bit LCG (Numerical Recipes constants); deterministic, no external crate.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// A small `Q32.32` word (magnitude `<= 1000`, so `|v| < 2^21` and the round trip is exact).
    fn word(&mut self) -> i64 {
        (self.next_u64() >> 33) as i64 % 2001 - 1000
    }

    fn fill(&mut self, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.word()).collect()
    }
}

fn cfg() -> Config {
    Config {
        n_layer: NLAYER,
        n_head: NH,
        n_head_kv: NKV,
        n_embd: HID,
        n_ff: INTER,
        vocab: VOCAB,
        head_dim: HD,
        // `round(1e-6 · 2^32)`, inside the `fixed_rsqrt` window `[2^12, 2^52]`.
        eps: 4_295,
        rope_theta: 10_000.0,
    }
}

fn grad_layer(rng: &mut Lcg) -> GradLayer {
    GradLayer {
        attn_norm: rng.fill(HID),
        wq: rng.fill(Q_OUT * HID),
        bq: Some(rng.fill(Q_OUT)),
        wk: rng.fill(KV_OUT * HID),
        bk: Some(rng.fill(KV_OUT)),
        wv: rng.fill(KV_OUT * HID),
        bv: Some(rng.fill(KV_OUT)),
        wo: rng.fill(HID * Q_OUT),
        ffn_norm: rng.fill(HID),
        w_gate: rng.fill(INTER * HID),
        w_up: rng.fill(INTER * HID),
        w_down: rng.fill(HID * INTER),
    }
}

fn build_weights(seed: u64) -> Weights {
    let mut rng = Lcg::new(seed);
    let layers = (0..NLAYER).map(|_| grad_layer(&mut rng)).collect();
    Weights {
        layers,
        output_norm: rng.fill(HID),
        token_embd: rng.fill(VOCAB * HID),
        lm_head: rng.fill(VOCAB * HID),
    }
}

/// (a) Every `Q32.32` word survives `f64 → Q32.32` exactly, so the round trip reproduces `w`.
#[test]
fn round_trip_is_exact_on_the_grid() {
    let w = build_weights(0xA11CE);
    let c = cfg();
    let teacher: Teacher = teacher_from_grad(&w, &c).expect("convert to teacher");
    assert_eq!(teacher.layers.len(), NLAYER);
    assert_eq!(teacher.n_layer, NLAYER);
    assert_eq!(teacher.fnorm.len(), HID);
    assert_eq!(teacher.embed.len(), VOCAB * HID);
    let back = student_to_grad(&w, &teacher.layers, &teacher.fnorm).expect("convert to grad");
    assert_eq!(back, w, "the Q32.32 round trip must be exact");
}

/// (b) An `f64` off the grid rounds to the nearest `Q32.32` word, within `2^-33`.
#[test]
fn off_grid_rounds_to_nearest() {
    let eps = 2f64.powi(-33);
    let mut rng = Lcg::new(0xBEEF);
    for _ in 0..2000 {
        // A value with more than 32 fractional bits, in range.
        let x = (rng.next_u64() >> 40) as f64 * 1e-6 - 1.0;
        let q = f64_to_q32(x).expect("in range");
        let back = q32_to_f64(q);
        assert!(
            (back - x).abs() <= eps,
            "x={x} back={back} err={}",
            (back - x).abs()
        );
    }
}

/// (b, through the layer path) A layer built from off-grid `f64` lowers within `2^-33` per element.
#[test]
fn off_grid_layer_rounds_within_half_ulp() {
    let w = build_weights(7);
    let c = cfg();
    let teacher = teacher_from_grad(&w, &c).expect("teacher");
    let mut layers = teacher.layers.clone();
    let mut rng = Lcg::new(0xF00D);
    for j in 0..HID {
        layers[0].ln1[j] = (rng.next_u64() >> 40) as f64 * 1e-6 - 1.0;
    }
    let back = student_to_grad(&w, &layers, &teacher.fnorm).expect("grad");
    let eps = 2f64.powi(-33);
    for j in 0..HID {
        let got = q32_to_f64(back.layers[0].attn_norm[j]);
        assert!((got - layers[0].ln1[j]).abs() <= eps);
    }
}

/// (c) The engine forward on the round-tripped weights equals the forward on the original, exactly.
#[test]
fn forward_agrees_after_round_trip() {
    let w = build_weights(0x1234_5678);
    let c = cfg();
    let teacher = teacher_from_grad(&w, &c).expect("teacher");
    let back = student_to_grad(&w, &teacher.layers, &teacher.fnorm).expect("grad");
    let tokens = [1u32, 2, 3, 7];
    let (logits_a, tape_a) = model::forward(&w, &c, &tokens).expect("forward original");
    let (logits_b, tape_b) = model::forward(&back, &c, &tokens).expect("forward round-tripped");
    assert_eq!(logits_a, logits_b, "logits must be bit-identical");
    assert_eq!(
        tape_a.final_hidden, tape_b.final_hidden,
        "the hidden states must be bit-identical"
    );
}

/// (d) A model with an absent attention bias is refused by name.
#[test]
fn missing_bias_is_refused() {
    let mut w = build_weights(3);
    w.layers[0].bk = None;
    match teacher_from_grad(&w, &cfg()) {
        Err(ConvertError::MissingBias(m)) => assert!(m.contains("attn_k.bias"), "{m}"),
        other => panic!("expected MissingBias, got {other:?}"),
    }
}

/// (d) A `f64` layer whose tensor length disagrees with the teacher is refused.
#[test]
fn wrong_length_is_refused() {
    let w = build_weights(4);
    let c = cfg();
    let teacher = teacher_from_grad(&w, &c).expect("teacher");
    let mut layers = teacher.layers.clone();
    layers[0].q.pop();
    match student_to_grad(&w, &layers, &teacher.fnorm) {
        Err(ConvertError::Shape(m)) => assert!(m.contains("attn_q.weight"), "{m}"),
        other => panic!("expected Shape, got {other:?}"),
    }
}

/// (d) A non-finite value is refused with the value in the diagnostic.
#[test]
fn non_finite_is_refused() {
    assert!(matches!(
        f64_to_q32(f64::NAN),
        Err(ConvertError::NonFinite(_))
    ));
    assert!(matches!(
        f64_to_q32(f64::INFINITY),
        Err(ConvertError::NonFinite(_))
    ));
    let w = build_weights(5);
    let c = cfg();
    let teacher = teacher_from_grad(&w, &c).expect("teacher");
    let mut layers = teacher.layers.clone();
    layers[0].ln1[0] = f64::NAN;
    assert!(matches!(
        student_to_grad(&w, &layers, &teacher.fnorm),
        Err(ConvertError::NonFinite(_))
    ));
}

/// (d) A finite value past `2^31` is out of the `Q32.32` range and refused.
#[test]
fn out_of_range_is_refused() {
    assert!(matches!(f64_to_q32(3.0e9), Err(ConvertError::Range(_))));
    assert!(matches!(f64_to_q32(-3.0e9), Err(ConvertError::Range(_))));
    let w = build_weights(6);
    let c = cfg();
    let teacher = teacher_from_grad(&w, &c).expect("teacher");
    let mut layers = teacher.layers.clone();
    layers[1].down[0] = 5.0e9;
    assert!(matches!(
        student_to_grad(&w, &layers, &teacher.fnorm),
        Err(ConvertError::Range(_))
    ));
}

/// (d) A student layer count that disagrees with the teacher is refused.
#[test]
fn layer_count_mismatch_is_refused() {
    let w = build_weights(8);
    let c = cfg();
    let teacher = teacher_from_grad(&w, &c).expect("teacher");
    match student_to_grad(&w, &teacher.layers[..1], &teacher.fnorm) {
        Err(ConvertError::Shape(m)) => assert!(m.contains("layer count"), "{m}"),
        other => panic!("expected Shape, got {other:?}"),
    }
}

/// The real `Qwen2 0.5B` f16 model loads, converts, and reports its geometry and wall time.
#[test]
#[ignore = "loads ~1 GB from disk; run explicitly with --ignored"]
fn real_qwen05b_loads() {
    let path = "/home/socol/models/gguf/qwen05b-f16.gguf";
    let t0 = std::time::Instant::now();
    let engine = apeiron::engine::Engine::open(path).expect("open gguf");
    let (w, c) = Weights::from_engine(&engine).expect("load weights");
    let teacher = teacher_from_grad(&w, &c).expect("convert to teacher");
    let elapsed = t0.elapsed();
    assert_eq!(teacher.layer_cfg.n_head, 14);
    assert_eq!(teacher.layer_cfg.n_kv_head, 2);
    assert_eq!(teacher.layer_cfg.head_dim, 64);
    assert_eq!(teacher.layer_cfg.hidden, 896);
    assert_eq!(teacher.layer_cfg.inter, 4864);
    assert_eq!(teacher.n_layer, 24);
    assert_eq!(teacher.embed.len(), teacher.vocab * 896);
    println!(
        "qwen05b-f16 load+convert wall time: {:.3} s (vocab {}, n_layer {})",
        elapsed.as_secs_f64(),
        teacher.vocab,
        teacher.n_layer
    );
}
