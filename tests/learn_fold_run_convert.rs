//! Gate for `apeiron::learn::fold::run::convert`: the `Q32.32 ↔ f64` round trip, the forward
//! agreement, and the typed refusals.
//!
//! A tiny `Qwen2` is built by a fixed LCG (`hidden` 128, `n_head` 2, `head_dim` 64, `n_kv_head` 1,
//! `inter` 128, `vocab` 32, `n_layer` 2) and run through both directions of the conversion. Every
//! value is `tested` here; nothing is `proved`.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "the inline LCG yields unsigned words and casts them into a small integer/float range; \
              the wrap and the precision loss are the generator's, not a measured quantity; the \
              float comparisons test exact dyadic ties (`x·2^32` is exact, so its fraction is \
              exactly `0.5` or not), not an approximate equality"
)]

use apeiron::learn::fold::run::convert::{
    encode_f16, encode_f32, f64_to_q32, q32_to_f64, student_to_grad, teacher_from_grad,
    ConvertError, Teacher,
};
use apeiron::learn::fold::run::transfer::{h0_from_chunks, TransferError};
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
    assert!(matches!(
        f64_to_q32(2_147_483_648.0),
        Err(ConvertError::Range(_))
    ));
    assert!(matches!(
        f64_to_q32(-2_147_483_648.0),
        Err(ConvertError::Range(_))
    ));
    // The largest representable magnitude below `2^31` in `f64`: `2^31 − 2^-21` (a 53-bit
    // significand) maps to `i64::MIN + 2048`, the last grid word.
    assert_eq!(
        f64_to_q32(-(2_147_483_648.0 - 2f64.powi(-21))).expect("in range"),
        i64::MIN + 2048
    );
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

/// The pre-migration `f64_to_q32` body, copied: `round(x·2^32)` in `f64` (ties away from zero),
/// `None` outside the admitted domain.
fn ref_f64_to_q32(x: f64) -> Option<i64> {
    if !x.is_finite() || x.abs() >= 2_147_483_648.0 {
        return None;
    }
    let q = (x * 4_294_967_296.0).round();
    if !q.is_finite() || q.abs() >= 9_223_372_036_854_775_808.0 {
        return None;
    }
    Some(q as i64)
}

/// A random admissible value: an exact 53-bit significand `m/2^53 ∈ [0, 1)` times `2^e` with
/// `e ∈ [-31, 31]`, so `|x| < 2^31` and both `x·2^32` and the sign change are exact.
fn lcg_value(rng: &mut Lcg) -> f64 {
    let m = (rng.next_u64() >> 11) as f64 * 2f64.powi(-53);
    let e = (rng.next_u64() % 63) as i32 - 31;
    let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
    sign * m * 2f64.powi(e)
}

/// (a) The integer decode agrees with the copied `f64` oracle.
///
/// Bound: `x·2^32` is an exact power-of-two rescale (`|x| < 2^31`, so no significand bit moves), so
/// the oracle is nearest with ties away from zero, while the integer body is nearest with ties to
/// even. They agree on every non-tie; at an exact tie (`x·2^32` a half-integer) they differ by one
/// quantum, `2^-32`, and each is exactly `2^-33` from `x`, so the value-level agreement bound is
/// `|q32_to_f64(got) − x| <= 2^-33`.
#[test]
fn integer_decode_agrees_with_the_f64_oracle() {
    let mut rng = Lcg::new(0x0DEC_0DE5);
    let mut max_err = 0.0_f64;
    let mut ties = 0usize;
    for _ in 0..20_000 {
        let x = lcg_value(&mut rng);
        let got = f64_to_q32(x).expect("in range");
        let want = ref_f64_to_q32(x).expect("oracle in range");
        if (x * 4_294_967_296.0).fract().abs() == 0.5 {
            // An exact tie between grid words `lo` and `lo + 1`: the oracle rounds away from zero,
            // the integer body to the even neighbour; both are `2^-33` from `x`.
            let lo = (x * 4_294_967_296.0).floor() as i64;
            let even = if lo % 2 == 0 { lo } else { lo + 1 };
            assert_eq!(got, even, "tie x={x}");
            assert_eq!(want, if x > 0.0 { lo + 1 } else { lo }, "tie x={x}");
            ties += 1;
        } else {
            assert_eq!(got, want, "x={x}");
        }
        max_err = max_err.max((q32_to_f64(got) - x).abs());
        assert!(max_err <= 2f64.powi(-33), "x={x} got={got}");
    }
    println!(
        "20000 values: max |q32_to_f64(f64_to_q32(x)) − x| = {max_err:.3e}, {ties} exact ties"
    );
}

/// (a) An exact tie at the grid quantum rounds to the even word.
#[test]
fn ties_round_to_even_at_the_grid_quantum() {
    for h in [-9i64, -7, -5, -3, -1, 1, 3, 5, 7, 9] {
        // `x·2^32 = h/2` with `h` odd: the two neighbours are `(h−1)/2` and `(h+1)/2`.
        let x = h as f64 * 2f64.powi(-33);
        let lo = (h - 1) / 2;
        let even = if lo % 2 == 0 { lo } else { lo + 1 };
        assert_eq!(f64_to_q32(x).expect("in range"), even, "h={h}");
        assert_eq!(even % 2, 0, "h={h}");
        let away = ref_f64_to_q32(x).expect("oracle in range");
        assert_eq!(away, if h > 0 { lo + 1 } else { lo }, "h={h}");
    }
}

/// (a) The gather copies the embedding rows in chunk order; a token past the vocabulary is refused.
#[test]
fn h0_gathers_the_embedding_rows() {
    let w = build_weights(0xC4);
    let teacher = teacher_from_grad(&w, &cfg()).expect("teacher");
    let chunks = vec![vec![3u32, 0, 31], Vec::new(), vec![7u32]];
    let got = h0_from_chunks(&teacher, &chunks).expect("h0");
    let mut want = Vec::new();
    for c in &chunks {
        for &id in c {
            let at = usize::try_from(id).expect("token id") * HID;
            want.extend_from_slice(&teacher.embed[at..at + HID]);
        }
    }
    assert_eq!(got, want);
    match h0_from_chunks(&teacher, &[vec![u32::try_from(VOCAB).expect("fits")]]) {
        Err(TransferError::Shape(m)) => assert!(m.contains("vocab"), "{m}"),
        other => panic!("expected Shape, got {other:?}"),
    }
}

/// (a) The F32 payload is the copied `f64` map's bytes: `(w/2^32)` as binary32.
#[test]
fn f32_payload_equals_the_f64_oracle() {
    let mut words = vec![
        0i64,
        1,
        -1,
        1_i64 << 32,
        -(1_i64 << 32),
        i64::MAX,
        i64::MIN,
        123_456_789_012,
        -987_654_321,
    ];
    let mut rng = Lcg::new(0x0F32);
    for _ in 0..3000 {
        words.push(rng.word() * 1_000_003);
    }
    let got = encode_f32(&words);
    assert_eq!(got.len(), words.len() * 4);
    for (i, &w) in words.iter().enumerate() {
        let want = (w as f64 / 4_294_967_296.0) as f32;
        assert_eq!(&got[i * 4..i * 4 + 4], want.to_le_bytes(), "word {w}");
    }
}

/// Decode an IEEE-754 binary16 bit pattern to `f64` (the F16 encoder's oracle).
fn half_to_f64(b: u16) -> f64 {
    let sign = if b & 0x8000 == 0 { 1.0 } else { -1.0 };
    let e = i32::from((b >> 10) & 0x1f);
    let f = f64::from(b & 0x03ff);
    if e == 0 {
        sign * f * 2f64.powi(-24)
    } else {
        sign * (1.0 + f / 1024.0) * 2f64.powi(e - 15)
    }
}

/// (a) The F16 payload is the nearest binary16 of each word: fixed values exactly, and on a random
/// set within half an ulp of the value's own binade. A value reaching `2^16` is refused.
#[test]
fn f16_payload_is_the_nearest_binary16() {
    for (word, bits) in [
        (0i64, 0x0000u16),
        (1_i64 << 32, 0x3c00),
        (-(1_i64 << 32), 0xbc00),
        (1_i64 << 31, 0x3800),
        (1_i64 << 33, 0x4000),
        (-(1_i64 << 33), 0xc000),
        (384, 0x0002),
        // `127.984375`: the rounded significand carries to the next binade, `128` exactly.
        (549_688_705_024, 0x5800),
    ] {
        assert_eq!(
            encode_f16(&[word]).expect("f16"),
            bits.to_le_bytes(),
            "word {word}"
        );
    }
    assert!(matches!(
        encode_f16(&[1_i64 << 48]),
        Err(ConvertError::Range(_))
    ));
    let mut rng = Lcg::new(0x0F16);
    let mut words = Vec::new();
    for _ in 0..2000 {
        // A real value `±mant · 2^e`, `mant ∈ [1, 2)` a 24-bit dyadic and `e ∈ [-13, 14]`, so
        // `|value| ∈ [2^-13, 2^15)` is a normal binary16 and `value · 2^32` is an exact integer.
        let mant = (rng.next_u64() >> 40) as f64 * 2f64.powi(-24) + 1.0;
        let e = (rng.next_u64() % 28) as i32 - 13;
        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        words.push((sign * mant * 2f64.powi(e) * 4_294_967_296.0) as i64);
    }
    let bytes = encode_f16(&words).expect("f16");
    assert_eq!(bytes.len(), words.len() * 2);
    let mut worst = 0.0_f64;
    for (i, &word) in words.iter().enumerate() {
        let got = half_to_f64(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
        let real = word as f64 / 4_294_967_296.0;
        let ulp = 2f64.powi(real.abs().log2().floor() as i32 - 10);
        worst = worst.max((got - real).abs() / (ulp / 2.0));
        assert!(
            (got - real).abs() <= ulp / 2.0,
            "word {word} real {real} got {got}"
        );
    }
    println!("2000 binary16 roundings: worst error = {worst:.3} half-ulps");
}
