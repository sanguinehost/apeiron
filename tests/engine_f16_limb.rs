//! The `F16` multi-column projection on the limb kernel, against the scalar oracle.
//! Board: APEIRON-QUALITY-118-W1-T01.
//!
//! The fixture is one dense `qwen2` layer whose matrices are `F16`. The multi-column
//! [`Engine::prefill_tap_cols`] pass takes the limb route (`src/engine/step/cols.rs` ->
//! `src/engine/kern/limb.rs`); the sequential public single-token path keeps the scalar
//! `matvec_q8act` oracle (K4). The test compares the head of every column against the sequential
//! run, asserts the deviation stays inside a stated word bound, and asserts it is nonzero — the
//! scalar path is bit-identical per column, so a zero deviation would mean the new route never ran.
//!
//! The residue-byte mutation control for the packed tile is the limb unit test
//! `f16_tile_is_within_the_limb_bound_and_a_flipped_digit_breaks_it` in
//! `src/engine/kern/limb.rs`: one flipped limb byte leaves the bound.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "test-only fixture arithmetic on small, bounded values"
)]

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::quant;

const F_E: usize = 64;
const F_FF: usize = 128;
const F_Q: usize = 64;
const F_KO: usize = 32;

/// The stated gross-error bound on the head's Q32.32 words. The head is the scalar `lm_head_fixed`
/// path; its input is the multi-column state, so this number folds the four limb projections and
/// the fixture's own wide weight range. The per-projection bound `G(n, 3, max|w|, max|x|)` and its
/// tight assertion are the limb unit test; this constant is four times the observed deviation on
/// this fixture and far below a route mix-up, which is whole units of `max|logit| ≈ 2^36`.
const HEAD_BOUND: u64 = 1 << 31;

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, u64::try_from(s.len()).expect("small"));
    b.extend_from_slice(s.as_bytes());
}

fn kv_u32(b: &mut Vec<u8>, k: &str, v: u32) {
    put_str(b, k);
    put_u32(b, 4);
    put_u32(b, v);
}

fn kv_str(b: &mut Vec<u8>, k: &str, v: &str) {
    put_str(b, k);
    put_u32(b, 8);
    put_str(b, v);
}

fn u(x: usize) -> u64 {
    u64::try_from(x).expect("fixture dims fit u64")
}

/// `rows · cols` finite half words in two exponent bands eleven powers apart, so the 21-bit block
/// expansion truncates the low band: a zero deviation cannot come from an exact fixture.
fn f16_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    let mut st = 0x9e37_79b9_7f4a_7c15u64
        ^ (u64::try_from(salt).expect("small") * 0x0100_0000_01b3)
        ^ u64::try_from(rows * cols).expect("small");
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    let mut i = 0usize;
    while i < rows * cols {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let e: u16 = if st.is_multiple_of(5) { 3 } else { 14 };
        let h = ((st >> 32) as u16 & 0x83ff) | (e << 10);
        b.extend_from_slice(&h.to_le_bytes());
        i += 1;
    }
    b
}

fn f32_gain(n: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    let mut i = 0usize;
    while i < n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        i += 1;
    }
    b
}

/// One dense `qwen2` layer plus the tied head, every matrix `F16`, every norm gain `1.0`.
#[allow(clippy::too_many_lines, reason = "one tabular fixture")]
fn build_layer_gguf() -> Vec<u8> {
    let ts: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        (
            "blk.0.attn_norm.weight",
            vec![u(F_E)],
            quant::GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "blk.0.attn_q.weight",
            vec![u(F_E), u(F_Q)],
            quant::GGML_TYPE_F16,
            f16_weight(F_Q, F_E, 1),
        ),
        (
            "blk.0.attn_k.weight",
            vec![u(F_E), u(F_KO)],
            quant::GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 2),
        ),
        (
            "blk.0.attn_v.weight",
            vec![u(F_E), u(F_KO)],
            quant::GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 3),
        ),
        (
            "blk.0.attn_output.weight",
            vec![u(F_Q), u(F_E)],
            quant::GGML_TYPE_F16,
            f16_weight(F_E, F_Q, 4),
        ),
        (
            "blk.0.ffn_norm.weight",
            vec![u(F_E)],
            quant::GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "blk.0.ffn_gate.weight",
            vec![u(F_E), u(F_FF)],
            quant::GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 5),
        ),
        (
            "blk.0.ffn_up.weight",
            vec![u(F_E), u(F_FF)],
            quant::GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 6),
        ),
        (
            "blk.0.ffn_down.weight",
            vec![u(F_FF), u(F_E)],
            quant::GGML_TYPE_F16,
            f16_weight(F_E, F_FF, 7),
        ),
        (
            "output_norm.weight",
            vec![u(F_E)],
            quant::GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "token_embd.weight",
            vec![u(F_E), u(F_FF)],
            quant::GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 9),
        ),
    ];
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, u64::try_from(ts.len()).expect("small"));
    put_u64(&mut b, 6);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(&mut b, "qwen2.block_count", 1);
    kv_u32(&mut b, "qwen2.attention.head_count", 2);
    kv_u32(&mut b, "qwen2.attention.head_count_kv", 1);
    kv_u32(
        &mut b,
        "qwen2.embedding_length",
        u32::try_from(F_E).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.feed_forward_length",
        u32::try_from(F_FF).expect("small"),
    );
    let mut off = 0u64;
    for (name, dims, ty, payload) in &ts {
        put_str(&mut b, name);
        put_u32(&mut b, u32::try_from(dims.len()).expect("rank 2"));
        for &d in dims {
            put_u64(&mut b, d);
        }
        put_u32(&mut b, *ty);
        put_u64(&mut b, off);
        off += u64::try_from(payload.len()).expect("small");
    }
    let pad = (32 - b.len() % 32) % 32;
    b.resize(b.len() + pad, 0);
    for (_, _, _, payload) in &ts {
        b.extend_from_slice(payload);
    }
    b
}

fn open(tag: &str) -> (Engine, String) {
    let path = std::env::temp_dir().join(format!("apeiron_f16_limb_{}_{tag}", std::process::id()));
    std::fs::write(&path, build_layer_gguf()).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

/// The multi-column `F16` projection is the sequential single-token oracle within [`HEAD_BOUND`]
/// words, and it is not bit-identical to it: the limb route ran. A second multi-column pass over
/// the cached tile reproduces the first bit for bit.
#[test]
fn f16_prefill_columns_are_within_the_limb_bound_of_the_scalar_oracle() {
    let (engine, path) = open("bound");
    let tokens = [1u32, 2, 3, 5, 8];
    let mut st = engine.state(64).expect("state");
    let (cols, _) = engine
        .prefill_tap_cols(&mut st, &tokens, KernelMode::Q8Act, false)
        .expect("limb prefill");
    assert_eq!(cols.len(), tokens.len());

    let mut st_seq = engine.state(64).expect("state");
    let mut max_dev: u64 = 0;
    let mut max_abs: u64 = 0;
    for (c, &t) in tokens.iter().enumerate() {
        let oracle = engine
            .step_logits_state(&mut st_seq, t, KernelMode::Q8Act)
            .expect("oracle step");
        let got = engine.lm_head_fixed(&cols[c]).expect("head");
        assert_eq!(got.len(), oracle.len());
        let mut i = 0usize;
        while i < got.len() {
            max_dev = max_dev.max(got[i].abs_diff(oracle[i]));
            max_abs = max_abs.max(oracle[i].unsigned_abs());
            i += 1;
        }
    }
    println!("oracle max |logit| {max_abs} words, deviation {max_dev}");
    assert!(
        max_dev > 0,
        "the multi-column F16 path was bit-identical to the scalar oracle; the limb route did not run"
    );
    assert!(
        max_dev <= HEAD_BOUND,
        "max |C_limb - C_scalar| {max_dev} words exceeds the stated {HEAD_BOUND}"
    );
    println!("f16 engine: max head deviation {max_dev} words of at most {HEAD_BOUND}");

    // Cache determinism: a fresh state's multi-column pass over the same tensor is bit-identical.
    let mut st2 = engine.state(64).expect("state");
    let (cols2, _) = engine
        .prefill_tap_cols(&mut st2, &tokens, KernelMode::Q8Act, false)
        .expect("limb prefill 2");
    assert_eq!(cols, cols2, "the cached tile is not deterministic");
    let _ = std::fs::remove_file(&path);
}
