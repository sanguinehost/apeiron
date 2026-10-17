//! Gate for `apeiron::learn::fold::run::convert`: the binary16/binary32 payload edges, the forward
//! round trip, the recorded word digests, and the typed refusals.
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
    decode_bf16, decode_f16, decode_f32, encode_f16, encode_f32, student_to_grad,
    teacher_from_grad, ConvertError, Teacher,
};
use apeiron::learn::fold::run::transfer::{h0_from_chunks, TransferError};
use apeiron::learn::grad::model::{self, Config, LayerWeights as GradLayer, Weights};
use apeiron::learn::solve::q32::{to_grid, GRID_BOUND};

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
    // The one floating-point field this boundary still fills: `eps` is the `Q32.32` config word
    // divided by `2^32`, exactly the pre-migration `cfg.eps as f64 / 4_294_967_296.0`.
    assert_eq!(
        teacher.layer_cfg.eps.to_bits(),
        (4_295.0_f64 / 4_294_967_296.0).to_bits()
    );
    let back = student_to_grad(&w, &teacher.layers, &teacher.fnorm).expect("convert to grad");
    assert_eq!(back, w, "the Q32.32 round trip must be exact");
}

/// (b, through the loader) An off-grid binary32 decodes to the nearest `Q32.32` word: exactly
/// `to_grid` of the decoded value except at a tie, where the word is the even neighbour. The value
/// is a 24-bit significand, so a tie is `e = 117` (`value·2^32` a half-integer) and the tie rule is
/// asserted separately.
#[test]
fn decode_words_equal_to_grid_of_the_decoded_value() {
    let grid = |x: f64| to_grid(x, GRID_BOUND).expect("on grid");
    for &bits in &[0x3c00u16, 0xbe00, 0x3400, 0x0001, 0x5800, 0x7bff, 0x8001] {
        let got = decode_f16(&bits.to_le_bytes()).expect("f16");
        assert_eq!(got, vec![grid(half_to_f64(bits))], "f16 {bits:#06x}");
    }
    for &bits in &[0x3f80u16, 0xbfc0, 0x3e80, 0x3dcc, 0x8001, 0x0000] {
        let wide = f64::from(f32::from_bits(u32::from(bits) << 16));
        let got = decode_bf16(&bits.to_le_bytes()).expect("bf16");
        assert_eq!(got, vec![grid(wide)], "bf16 {bits:#06x}");
    }
    let mut rng = Lcg::new(0x00DE_C0DE);
    for _ in 0..20_000 {
        let x = (rng.next_u64() >> 40) as f64 * 1e-6 - 1.0;
        let x32 = x as f32;
        let got = decode_f32(&x32.to_le_bytes()).expect("f32");
        let q = f64::from(x32) * 4_294_967_296.0;
        if q.fract().abs() == 0.5 {
            // The tie rule is `ties_round_to_even_at_the_grid_quantum`'s.
            continue;
        }
        assert_eq!(got, vec![grid(f64::from(x32))], "f32 {x32}");
    }
    // Non-finite and out-of-range elements are refused by name, at the same edge.
    assert!(matches!(
        decode_bf16(&0x7f80u16.to_le_bytes()),
        Err(ConvertError::NonFinite(_))
    ));
    assert!(matches!(
        decode_f16(&0x7c00u16.to_le_bytes()),
        Err(ConvertError::NonFinite(_))
    ));
    assert!(matches!(
        decode_f32(&0x7fc0_0000u32.to_le_bytes()),
        Err(ConvertError::NonFinite(_))
    ));
    assert!(matches!(decode_f16(&[0u8; 3]), Err(ConvertError::Shape(_))));
    assert!(matches!(
        decode_f32(&3.0e9f32.to_bits().to_le_bytes()),
        Err(ConvertError::Range(_))
    ));
    assert!(matches!(
        decode_bf16(&(3.0e9f32.to_bits() >> 16).to_le_bytes()),
        Err(ConvertError::Range(_))
    ));
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
    layers[0].wq.pop();
    match student_to_grad(&w, &layers, &teacher.fnorm) {
        Err(ConvertError::Shape(m)) => assert!(m.contains("attn_q.weight"), "{m}"),
        other => panic!("expected Shape, got {other:?}"),
    }
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
    let path = "~/models/gguf/qwen05b-f16.gguf";
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

/// (a) At a tie at the `2^-32` quantum the word is the even neighbour; the grid's top word is the
/// largest finite binary32 below `2^31`, and `2^31` itself refuses.
#[test]
fn ties_round_to_even_at_the_grid_quantum() {
    for mant in [1u32, 3, 5, 0x7f_ffff] {
        // `e = 117`: `value·2^32 = a/2` with `a = 2^23 + mant` odd, so the two neighbours are
        // `a/2` and `a/2 + 1` and the even one wins.
        let a = (1u32 << 23) | mant;
        let lo = a / 2;
        let even = if lo.is_multiple_of(2) { lo } else { lo + 1 };
        let bits = (117u32 << 23) | mant;
        assert_eq!(
            decode_f32(&bits.to_le_bytes()).expect("f32"),
            vec![i64::from(even)],
            "mant {mant:#x}"
        );
        assert_eq!(
            decode_f32(&(bits | 0x8000_0000).to_le_bytes()).expect("f32"),
            vec![-i64::from(even)],
            "mant {mant:#x} negative"
        );
        assert_eq!(even % 2, 0, "mant {mant:#x}");
    }
    // `(2^24 − 1)·2^7 = 2^31 − 2^7`: the largest finite binary32 below the bound.
    let last = (157u32 << 23) | 0x7f_ffff;
    assert_eq!(
        decode_f32(&last.to_le_bytes()).expect("f32"),
        vec![i64::MAX - ((1_i64 << 39) - 1)]
    );
    assert!(matches!(
        decode_f32(&(158u32 << 23).to_le_bytes()),
        Err(ConvertError::Range(_))
    ));
}

/// FNV-1a 64 fold of `bytes` into `h`.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a 64 over the little-endian bytes of the words.
fn fnv_words(v: &[i64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &w in v {
        h = fnv1a(h, &w.to_le_bytes());
    }
    h
}

/// The convert-side pre-migration digest: 65 536 binary16 patterns, 20 000 random binary32 and
/// binary16 patterns, and 20 000 encoded words, folded in the recorded order.
fn witness_convert(seed: u64) -> u64 {
    let mut rng = Lcg::new(seed);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let fold = |h: u64, v: &[i64]| fnv1a(h, &fnv_words(v).to_le_bytes());
    let mut f16_words = Vec::new();
    for b in 0..=0xffffu32 {
        let b = b as u16;
        if let Ok(v) = decode_f16(&b.to_le_bytes()) {
            f16_words.extend(v);
        }
    }
    h = fnv1a(h, b"f16");
    h = fnv1a(h, &(f16_words.len() as i64).to_le_bytes());
    h = fold(h, &f16_words);
    let mut f32_words = Vec::new();
    let mut f32_refused = 0i64;
    for _ in 0..20_000 {
        let bits = (rng.next_u64() >> 32) as u32;
        match decode_f32(&bits.to_le_bytes()) {
            Ok(v) => f32_words.extend(v),
            Err(_) => f32_refused += 1,
        }
    }
    h = fnv1a(h, b"f32");
    h = fnv1a(h, &(f32_words.len() as i64).to_le_bytes());
    h = fold(h, &f32_words);
    h = fnv1a(h, &f32_refused.to_le_bytes());
    let mut bf_words = Vec::new();
    for _ in 0..20_000 {
        let bits = (rng.next_u64() >> 48) as u16;
        if let Ok(v) = decode_bf16(&bits.to_le_bytes()) {
            bf_words.extend(v);
        }
    }
    h = fnv1a(h, b"bf16");
    h = fnv1a(h, &(bf_words.len() as i64).to_le_bytes());
    h = fold(h, &bf_words);
    let words: Vec<i64> = (0..20_000).map(|_| rng.word() * 1_000_003).collect();
    let enc32 = encode_f32(&words);
    h = fnv1a(h, b"enc32");
    h = fnv1a(
        h,
        &fnv_words(&[i64::try_from(enc32.len()).expect("len")]).to_le_bytes(),
    );
    h = fnv1a(h, &enc32);
    h = fnv1a(h, b"enc16");
    match encode_f16(&words) {
        Ok(b) => {
            h = fnv1a(h, &(b.len() as i64).to_le_bytes());
            h = fnv1a(h, &b);
        }
        Err(_) => h = fnv1a(h, &(-1i64).to_le_bytes()),
    }
    h
}

/// The recorded pre-migration digests of the binary16/binary32 edges, from the source at base
/// `57d21789` (whose binary32 decode widened to a 64-bit exponent-and-significand field and called
/// the scalar converter this lane deleted). The integer decode reproduces them exactly on both
/// seeds; a change to the rounding, the range bound or the payload order reddens this test.
#[test]
fn recorded_decode_and_encode_digests_hold() {
    for (seed, want) in [
        (0xC0FF_EE01u64, 0x5e29_aec3_81a8_ff10u64),
        (0xC0FF_EE02, 0x3d3a_faa0_91fa_09cf),
    ] {
        assert_eq!(witness_convert(seed), want, "seed {seed:016x}");
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
