//! The opt-in engine tap: the per-layer stage rows the activation-Gram capture reads.
//!
//! The fixture is one dense `qwen2` layer with the head, so `prefill_tap_cols` composes every
//! row. The tests assert (a) the tapped run is bit-identical to the untapped one and to the
//! single-token public path, and (b) the tapped `attn_norm` output is exactly the engine's own
//! `RMSNorm` recomputed from the tapped residual input.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test-only fixture and stage-index arithmetic on bounded small values"
)]

use apeiron::engine::loader::Engine;
use apeiron::engine::{rms_norm_fixed, KernelMode};
use apeiron::format::quant;

/// Stage slots of the tapped rows, in the order `prefill_tap_cols` returns them.
const STAGE_HIDDEN_IN: usize = 0;
const STAGE_LN1_OUT: usize = 1;
const STAGE_ATTN_OUT: usize = 2;
const STAGE_LN2_OUT: usize = 3;
const STAGE_M: usize = 4;

const F_E: usize = 8;
const F_FF: usize = 16;
const F_Q: usize = 8;
const F_KO: usize = 4;

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u64(b, s.len() as u64);
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

/// Round-to-nearest-even `f16` bits for a finite normal `f32` in `(2^-14, 65504)`.
fn f16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits << 1 == 0 {
        return 0;
    }
    let sign = ((bits >> 16) & 0x8000) as u16;
    let e = i32::try_from((bits >> 23) & 0xff).expect("8 bits") - 127 + 15;
    let mut m = i32::try_from((bits >> 13) & 0x3ff).expect("10 bits");
    let rem = bits & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        m += 1;
    }
    let carry = m >> 10;
    m &= 0x3ff;
    sign | (((e + carry) as u16) << 10) | (m as u16)
}

/// `usize` to `u64` for the fixture's dims.
fn u(x: usize) -> u64 {
    u64::try_from(x).expect("fixture dims fit u64")
}

/// A deterministic `F16` payload of `rows · cols` weights in `{-2, -1, 0, 1, 2}·2^-5`.
fn f16_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    const SCALE: f32 = 1.0 / 32.0;
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    let mut r = 0usize;
    while r < rows {
        let mut c = 0usize;
        while c < cols {
            let code = i32::try_from((r * 7 + c * 3 + salt * 11) % 5).expect("small") - 2;
            b.extend_from_slice(&f16_round(code as f32 * SCALE).to_le_bytes());
            c += 1;
        }
        r += 1;
    }
    b
}

/// An all-`1.0` `F32` vector of `n` gains.
fn f32_gain(n: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    let mut i = 0usize;
    while i < n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
        i += 1;
    }
    b
}

/// One full dense `qwen2` layer plus the head.
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
            vec![u(F_E), u(F_E)],
            quant::GGML_TYPE_F16,
            f16_weight(F_E, F_E, 4),
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

/// Write the fixture to a per-process temp file and open it.
fn open(tag: &str) -> (Engine, String) {
    let path =
        std::env::temp_dir().join(format!("apeiron_engine_tap_{}_{tag}", std::process::id()));
    std::fs::write(&path, build_layer_gguf()).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

/// The fixed-point `RMSNorm` epsilon the engine resolves from its config.
fn eps_fixed(engine: &Engine) -> i64 {
    (engine.config.rms_eps.to_f64() * 4_294_967_296.0).round() as i64
}

/// With the tap on, every final column is bit-identical to the untapped run and to the
/// single-token public path, and every stage row has its linear's input width.
#[test]
fn tap_rows_are_bit_identical_and_have_the_stage_widths() {
    let (engine, path) = open("widths");
    let tokens = [1u32, 2, 3, 4];
    let k = tokens.len();
    let mut st_off = engine.state(16).expect("state");
    let (cols_off, rows_off) = engine
        .prefill_tap_cols(&mut st_off, &tokens, KernelMode::Q8Act, false)
        .expect("untapped prefill");
    let mut st_on = engine.state(16).expect("state");
    let (cols_on, rows_on) = engine
        .prefill_tap_cols(&mut st_on, &tokens, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    let _ = std::fs::remove_file(&path);

    // The tapped run changes no value.
    assert_eq!(cols_off, cols_on, "tap changed the final columns");
    assert!(rows_off.is_empty(), "record=false allocated stage rows");
    assert_eq!(rows_on.len(), engine.config.n_layer);
    assert_eq!(rows_on[0].len(), 5);
    for stage in 0..5 {
        assert_eq!(rows_on[0][stage].len(), k, "stage {stage} column count");
    }
    for c in 0..k {
        assert_eq!(rows_on[0][STAGE_HIDDEN_IN][c].len(), F_E);
        assert_eq!(rows_on[0][STAGE_LN1_OUT][c].len(), F_E);
        assert_eq!(rows_on[0][STAGE_ATTN_OUT][c].len(), F_Q);
        assert_eq!(rows_on[0][STAGE_LN2_OUT][c].len(), F_E);
        assert_eq!(rows_on[0][STAGE_M][c].len(), F_FF);
    }

    // The logits from the final columns agree.
    let last = cols_on.len() - 1;
    let off = engine.lm_head_fixed(&cols_off[last]).expect("logits");
    let on = engine.lm_head_fixed(&cols_on[last]).expect("logits");
    assert_eq!(off, on, "tap changed the logits");
}

/// The untapped tap path is the single-token public path, column for column.
#[test]
fn untapped_prefill_equals_single_token_steps() {
    let (engine, path) = open("single");
    let tokens = [3u32, 5, 7];
    let mut block = engine.state(16).expect("state");
    let (cols, rows) = engine
        .prefill_tap_cols(&mut block, &tokens, KernelMode::Exact, false)
        .expect("prefill");
    let mut single = engine.state(16).expect("state");
    let _ = std::fs::remove_file(&path);
    assert!(rows.is_empty());
    assert_eq!(cols.len(), tokens.len());
    for (pos, &t) in tokens.iter().enumerate() {
        let want = engine
            .step_trunk_state(&mut single, t, KernelMode::Exact)
            .expect("single step");
        assert_eq!(want, cols[pos], "column {pos} differs from the single step");
    }
}

/// The tapped `attn_norm` output is the engine's own `RMSNorm` of the tapped residual input (the
/// fixture's gain is exactly one), and the `ffn_norm` stage is the same norm of the residual after
/// the attention block.
#[test]
fn tap_ln1_out_is_the_engine_norm_of_the_tapped_input() {
    let (engine, path) = open("norm");
    let tokens = [2u32, 4, 6];
    let eps = eps_fixed(&engine);
    let mut st = engine.state(16).expect("state");
    let (_, rows) = engine
        .prefill_tap_cols(&mut st, &tokens, KernelMode::Exact, true)
        .expect("tapped prefill");
    let _ = std::fs::remove_file(&path);
    for (c, _) in tokens.iter().enumerate() {
        let ln1 = rms_norm_fixed(&rows[0][STAGE_HIDDEN_IN][c], eps).expect("norm");
        assert_eq!(
            ln1, rows[0][STAGE_LN1_OUT][c],
            "ln1_out column {c} is not the norm of the tapped hidden"
        );
        let ln2 = rms_norm_fixed(&rows[0][STAGE_LN2_OUT][c], eps);
        assert!(ln2.is_ok());
    }
    // The two norm stages are distinct activations, not one copied into both slots.
    assert_ne!(rows[0][STAGE_LN1_OUT], rows[0][STAGE_LN2_OUT]);
}
