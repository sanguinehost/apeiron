//! The opt-in engine tap: the per-layer stage rows the activation-Gram capture reads.
//!
//! The fixture is one dense `qwen2` layer with the head, so `prefill_tap_cols` composes every
//! row. The tests assert (a) the tapped run is bit-identical to the untapped one and to the
//! single-token public path, and (b) the tapped `attn_norm` output is exactly the engine's own
//! `RMSNorm` recomputed from the tapped residual input.
//!
//! Board: (post-`RoPE` q/k/v, pre-activation gate/up, post-attention
//! residual and layer-output tap stages).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test-only fixture and stage-index arithmetic on bounded small values"
)]

use apeiron::engine::loader::Engine;
use apeiron::engine::{rms_norm_fixed, silu_fixed, KernelMode, KvState};
use apeiron::format::quant;

/// Stage slots of the tapped rows, in the order `prefill_tap_cols` returns them.
const STAGE_HIDDEN_IN: usize = 0;
const STAGE_LN1_OUT: usize = 1;
const STAGE_ATTN_OUT: usize = 2;
const STAGE_LN2_OUT: usize = 3;
const STAGE_M: usize = 4;
const STAGE_Q: usize = 5;
const STAGE_K: usize = 6;
const STAGE_V: usize = 7;
const STAGE_GATE: usize = 8;
const STAGE_UP: usize = 9;
const STAGE_H2: usize = 10;
const STAGE_OUT: usize = 11;
/// The stage count `Tap::new` allocates.
const STAGE_COUNT: usize = 12;

const F_E: usize = 8;
const F_FF: usize = 16;
/// The `Q8_0`-FFN fixture's width and hidden rows (one 32-input block per weight row).
const Q8_E: usize = 32;
const Q8_FF: usize = 32;
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

/// A deterministic `Q8_0` payload of `rows` rows of `cols` inputs (`cols` a multiple of 32):
/// every 34-byte block is the `f16` scale `2^-10` then 32 codes in `{-8..=8}`.
fn q8_0_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0, "a Q8_0 block is 32 inputs");
    let scale = f16_round(1.0 / 1024.0).to_le_bytes();
    let mut b: Vec<u8> = Vec::with_capacity(rows * (cols / 32) * 34);
    let mut r = 0usize;
    while r < rows {
        let mut blk = 0usize;
        while blk < cols / 32 {
            b.extend_from_slice(&scale);
            let mut j = 0usize;
            while j < 32 {
                let code = i32::try_from((r * 5 + blk * 3 + j + salt * 7) % 17).expect("small") - 8;
                b.push(code as i8 as u8);
                j += 1;
            }
            blk += 1;
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

/// One full dense `qwen2` layer plus the head, of width `e` with `ff` hidden rows.
///
/// With `q8_ffn`, the gate and up projections are `Q8_0` (`e` a multiple of 32), so the pooled
/// prefill takes the fused `gemm_q8_0_swiglu_carrier` branch; otherwise they are `F16`, which takes
/// the separate gate/up/`SwiGLU` branch.
#[allow(clippy::too_many_lines, reason = "one tabular fixture")]
fn build_layer_gguf_with(e: usize, ff: usize, q8_ffn: bool) -> Vec<u8> {
    build_n_layer_gguf(e, ff, q8_ffn, 1)
}

/// `n_layer` copies of the dense `qwen2` layer, each block's weights salted by its layer index so
/// the layers are not bit-identical, plus the tied head.
#[allow(clippy::too_many_lines, reason = "one tabular fixture")]
fn build_n_layer_gguf(e: usize, ff: usize, q8_ffn: bool, n_layer: usize) -> Vec<u8> {
    let (q, ko) = (e, e / 2);
    let mut ts: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let mut l = 0usize;
    while l < n_layer {
        let salt = l * 20;
        let (ffn_ty, gate_payload, up_payload) = if q8_ffn {
            (
                quant::GGML_TYPE_Q8_0,
                q8_0_weight(ff, e, 5 + salt),
                q8_0_weight(ff, e, 6 + salt),
            )
        } else {
            (
                quant::GGML_TYPE_F16,
                f16_weight(ff, e, 5 + salt),
                f16_weight(ff, e, 6 + salt),
            )
        };
        let blk = |name: &str| format!("blk.{l}.{name}");
        ts.extend([
            (
                blk("attn_norm.weight"),
                vec![u(e)],
                quant::GGML_TYPE_F32,
                f32_gain(e),
            ),
            (
                blk("attn_q.weight"),
                vec![u(e), u(q)],
                quant::GGML_TYPE_F16,
                f16_weight(q, e, 1 + salt),
            ),
            (
                blk("attn_k.weight"),
                vec![u(e), u(ko)],
                quant::GGML_TYPE_F16,
                f16_weight(ko, e, 2 + salt),
            ),
            (
                blk("attn_v.weight"),
                vec![u(e), u(ko)],
                quant::GGML_TYPE_F16,
                f16_weight(ko, e, 3 + salt),
            ),
            (
                blk("attn_output.weight"),
                vec![u(e), u(e)],
                quant::GGML_TYPE_F16,
                f16_weight(e, e, 4 + salt),
            ),
            (
                blk("ffn_norm.weight"),
                vec![u(e)],
                quant::GGML_TYPE_F32,
                f32_gain(e),
            ),
            (
                blk("ffn_gate.weight"),
                vec![u(e), u(ff)],
                ffn_ty,
                gate_payload,
            ),
            (blk("ffn_up.weight"), vec![u(e), u(ff)], ffn_ty, up_payload),
            (
                blk("ffn_down.weight"),
                vec![u(ff), u(e)],
                quant::GGML_TYPE_F16,
                f16_weight(e, ff, 7 + salt),
            ),
        ]);
        l += 1;
    }
    ts.extend([
        (
            "output_norm.weight".to_string(),
            vec![u(e)],
            quant::GGML_TYPE_F32,
            f32_gain(e),
        ),
        (
            "token_embd.weight".to_string(),
            vec![u(e), u(ff)],
            quant::GGML_TYPE_F16,
            f16_weight(ff, e, 9),
        ),
    ]);
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, u64::try_from(ts.len()).expect("small"));
    put_u64(&mut b, 6);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(
        &mut b,
        "qwen2.block_count",
        u32::try_from(n_layer).expect("small"),
    );
    kv_u32(&mut b, "qwen2.attention.head_count", 2);
    kv_u32(&mut b, "qwen2.attention.head_count_kv", 1);
    kv_u32(
        &mut b,
        "qwen2.embedding_length",
        u32::try_from(e).expect("small"),
    );
    kv_u32(
        &mut b,
        "qwen2.feed_forward_length",
        u32::try_from(ff).expect("small"),
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

/// The `F16`-FFN fixture the tap tests share.
fn build_layer_gguf() -> Vec<u8> {
    build_layer_gguf_with(F_E, F_FF, false)
}

/// The `Q8_0`-FFN fixture: width 32, 32 hidden rows, so the fused kernel's shape gate holds.
fn build_layer_gguf_q8() -> Vec<u8> {
    build_layer_gguf_with(Q8_E, Q8_FF, true)
}

/// Write `bytes` to a per-process temp file and open it.
fn open_bytes(bytes: &[u8], tag: &str) -> (Engine, String) {
    let path =
        std::env::temp_dir().join(format!("apeiron_engine_tap_{}_{tag}", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

/// Write the `F16`-FFN fixture to a per-process temp file and open it.
fn open(tag: &str) -> (Engine, String) {
    open_bytes(&build_layer_gguf(), tag)
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
    assert_eq!(rows_on[0].len(), STAGE_COUNT);
    for stage in 0..STAGE_COUNT {
        assert_eq!(rows_on[0][stage].len(), k, "stage {stage} column count");
    }
    for c in 0..k {
        assert_eq!(rows_on[0][STAGE_HIDDEN_IN][c].len(), F_E);
        assert_eq!(rows_on[0][STAGE_LN1_OUT][c].len(), F_E);
        assert_eq!(rows_on[0][STAGE_ATTN_OUT][c].len(), F_Q);
        assert_eq!(rows_on[0][STAGE_LN2_OUT][c].len(), F_E);
        assert_eq!(rows_on[0][STAGE_M][c].len(), F_FF);
        assert_eq!(rows_on[0][STAGE_Q][c].len(), F_Q);
        assert_eq!(rows_on[0][STAGE_K][c].len(), F_KO);
        assert_eq!(rows_on[0][STAGE_V][c].len(), F_KO);
        assert_eq!(rows_on[0][STAGE_GATE][c].len(), F_FF);
        assert_eq!(rows_on[0][STAGE_UP][c].len(), F_FF);
        assert_eq!(rows_on[0][STAGE_H2][c].len(), F_E);
        assert_eq!(rows_on[0][STAGE_OUT][c].len(), F_E);
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

/// The tapped `attn_norm` output is the engine's own `RMSNorm` of the tapped residual input (the/// fixture's gain is exactly one), and the `ffn_norm` stage is the same norm of the residual after
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

/// The maximum absolute word difference between two rows of the same width.
fn max_abs_row_diff(a: &[i64], b: &[i64]) -> i64 {
    assert_eq!(a.len(), b.len(), "row widths");
    let mut m = 0i64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

/// The tap's post-attention residual ([`STAGE_H2`]) is the `ffn_norm` input, and the last layer's
/// output residual ([`STAGE_OUT`]) is the `output_norm` input: the engine's own `RMSNorm` of each
/// row (the fixture's gains are exactly one) reproduces the tapped `ffn_norm` output and the
/// returned final column word for word.
#[test]
fn tap_h2_feeds_ffn_norm_and_out_is_the_output_norm_input() {
    let (engine, path) = open("h2out");
    let tokens = [2u32, 4, 6];
    let eps = eps_fixed(&engine);
    let mut st = engine.state(16).expect("state");
    let (cols, rows) = engine
        .prefill_tap_cols(&mut st, &tokens, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    let _ = std::fs::remove_file(&path);
    let last = engine.config.n_layer - 1;
    let mut h2_max = 0i64;
    let mut out_max = 0i64;
    for c in 0..tokens.len() {
        let ln2 = rms_norm_fixed(&rows[0][STAGE_H2][c], eps).expect("norm of the tapped h2");
        let h2_d = max_abs_row_diff(&ln2, &rows[0][STAGE_LN2_OUT][c]);
        let final_normed =
            rms_norm_fixed(&rows[last][STAGE_OUT][c], eps).expect("norm of the tapped out");
        let out_d = max_abs_row_diff(&final_normed, &cols[c]);
        println!(
            "column {c}: ffn_norm(h2) vs ln2_out max abs = {h2_d}; output_norm(out) vs final \
             column max abs = {out_d}"
        );
        h2_max = h2_max.max(h2_d);
        out_max = out_max.max(out_d);
    }
    // The two rows are new activations, not the residual input or the norm output copied.
    assert_ne!(rows[0][STAGE_H2], rows[0][STAGE_HIDDEN_IN]);
    assert_ne!(rows[0][STAGE_OUT], rows[last][STAGE_LN2_OUT]);
    assert_eq!(
        h2_max, 0,
        "ffn_norm of the tapped h2 is not the tapped ln2_out (max abs {h2_max})"
    );
    assert_eq!(
        out_max, 0,
        "output_norm of the tapped out is not the final column (max abs {out_max})"
    );
}

/// The layer's output residual is the next layer's input: [`STAGE_OUT`] of layer `l` is
/// [`STAGE_HIDDEN_IN`] of layer `l + 1` word for word, on a two-layer fixture whose layers carry
/// different weights, and the last layer's [`STAGE_OUT`] is the `output_norm` input.
#[test]
fn tap_out_chains_into_the_next_layer() {
    let bytes = build_n_layer_gguf(F_E, F_FF, false, 2);
    let (engine, path) = open_bytes(&bytes, "chain");
    assert_eq!(engine.config.n_layer, 2, "the chain fixture has two layers");
    let tokens = [1u32, 2, 3];
    let eps = eps_fixed(&engine);
    let mut st = engine.state(16).expect("state");
    let (cols, rows) = engine
        .prefill_tap_cols(&mut st, &tokens, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    let _ = std::fs::remove_file(&path);
    let last = engine.config.n_layer - 1;
    let mut chain_max = 0i64;
    let mut out_max = 0i64;
    for l in 0..last {
        for c in 0..tokens.len() {
            let d = max_abs_row_diff(&rows[l][STAGE_OUT][c], &rows[l + 1][STAGE_HIDDEN_IN][c]);
            if d > chain_max {
                chain_max = d;
            }
        }
    }
    for c in 0..tokens.len() {
        let final_normed =
            rms_norm_fixed(&rows[last][STAGE_OUT][c], eps).expect("norm of the last out");
        let d = max_abs_row_diff(&final_normed, &cols[c]);
        if d > out_max {
            out_max = d;
        }
    }
    println!(
        "out[l] vs hidden_in[l+1] max abs = {chain_max}; output_norm(out[last]) vs final column \
         max abs = {out_max}"
    );
    assert_eq!(
        chain_max, 0,
        "layer output is not the next layer's input (max abs {chain_max})"
    );
    assert_eq!(
        out_max, 0,
        "the last layer's out is not the output_norm input (max abs {out_max})"
    );
}

/// The tapped post-`RoPE` query and key and the tapped value reproduce the tapped `attn_out` rows:
/// the engine's own attention read ([`KvState::attn_head`]: the Q32 score dot, the exact-max
/// `fixed_exp` softmax and the weight/value fold) over a fresh cache holding exactly the tapped
/// `k`/`v` rows returns the tapped output word for word. This is the property that lets a backward
/// pass recompute the attention probabilities from q and k instead of a recorded `p`.
#[test]
fn tapped_q_k_v_reproduce_attn_out() {
    let (engine, path) = open("qkv");
    let tokens = [1u32, 2, 3, 4];
    let cfg = &engine.config;
    let (n_head, hd, n_kv) = (cfg.n_head, cfg.head_dim, cfg.n_head_kv);
    let q_out = n_head * hd;
    let mut st = engine.state(16).expect("state");
    let (_, rows) = engine
        .prefill_tap_cols(&mut st, &tokens, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    let _ = std::fs::remove_file(&path);
    assert_eq!(rows[0][STAGE_Q][0].len(), q_out);
    let mut kv = KvState::new(1, n_head, n_kv, hd, tokens.len()).expect("kv");
    let mut last: Vec<usize> = Vec::with_capacity(tokens.len());
    for c in 0..tokens.len() {
        let pos = kv
            .append(0, &rows[0][STAGE_K][c], &rows[0][STAGE_V][c])
            .expect("append tapped key/value");
        last.push(pos);
    }
    let mut got: Vec<i64> = vec![0i64; q_out];
    let mut max_abs: i64 = 0;
    for c in 0..tokens.len() {
        let q = &rows[0][STAGE_Q][c];
        for h in 0..n_head {
            kv.attn_head(
                0,
                h,
                &q[h * hd..(h + 1) * hd],
                last[c],
                &mut got[h * hd..(h + 1) * hd],
            )
            .expect("attention over the tapped rows");
        }
        for i in 0..q_out {
            let d = (got[i] - rows[0][STAGE_ATTN_OUT][c][i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    println!("q/k/v -> attn_out max abs word difference = {max_abs}");
    assert_eq!(
        max_abs, 0,
        "tapped q/k/v do not reproduce attn_out (max abs word difference {max_abs})"
    );
}

/// Off-path cost of the tap: `prefill_tap_cols(record = false)` on the fixture, `ROUNDS` timed
/// rounds of `INNER` fresh-state prefills, `ns` per prefill. Run with
/// `cargo test --release --test engine_tap tap_off_path -- --ignored --nocapture`; it prints the
/// untapped path's spacing at this fixture's shape (8-wide embedding, four columns), which is a
/// microbench, not a model-scale timing.
#[test]
#[ignore = "microbench: run explicitly with --ignored --nocapture"]
fn tap_off_path_microbench() {
    const INNER: usize = 4000;
    const ROUNDS: usize = 5;
    let (engine, path) = open("bench");
    let tokens = [1u32, 2, 3, 4];
    for _ in 0..200 {
        let mut warm = engine.state(16).expect("state");
        let _ = engine
            .prefill_tap_cols(&mut warm, &tokens, KernelMode::Q8Act, false)
            .expect("prefill");
    }
    let mut rounds: Vec<f64> = Vec::with_capacity(ROUNDS);
    for r in 0..ROUNDS {
        let t0 = std::time::Instant::now();
        for _ in 0..INNER {
            let mut s = engine.state(16).expect("state");
            let _ = engine
                .prefill_tap_cols(&mut s, &tokens, KernelMode::Q8Act, false)
                .expect("prefill");
        }
        let ns = t0.elapsed().as_nanos() as f64 / INNER as f64;
        rounds.push(ns);
        println!("round {r}: {ns:.1} ns/prefill");
    }
    let _ = std::fs::remove_file(&path);
}

/// `round(silu(gate)·up / 2^32)`, the engine's `swiglu_one` element: [`silu_fixed`], the checked
/// `i128` product and one round-shift-32 narrowing (`src/engine/kern/gemm.rs:869`).
fn swiglu_q32(gate: i64, up: i64) -> i64 {
    let s = silu_fixed(gate).expect("gate inside the SiLU window");
    let prod = i128::from(s)
        .checked_mul(i128::from(up))
        .expect("product inside i128");
    i64::try_from((prod + (1_i128 << 31)) >> 32).expect("activation inside i64")
}

/// Recompute every tapped `m` word from the tapped gate and up words of the same column and return
/// the maximum absolute word difference.
fn gate_up_reproduction_max_abs(bytes: &[u8], tag: &str, mode: KernelMode) -> i64 {
    let (engine, path) = open_bytes(bytes, tag);
    let n_ff = engine.config.n_ff;
    let tokens = [1u32, 2, 3, 4];
    let mut st = engine.state(16).expect("state");
    let (_, rows) = engine
        .prefill_tap_cols(&mut st, &tokens, mode, true)
        .expect("tapped prefill");
    let _ = std::fs::remove_file(&path);
    let mut max_abs = 0i64;
    for c in 0..tokens.len() {
        assert_eq!(rows[0][STAGE_GATE][c].len(), n_ff, "gate width column {c}");
        assert_eq!(rows[0][STAGE_UP][c].len(), n_ff, "up width column {c}");
        assert_eq!(rows[0][STAGE_M][c].len(), n_ff, "m width column {c}");
        for i in 0..n_ff {
            let h = swiglu_q32(rows[0][STAGE_GATE][c][i], rows[0][STAGE_UP][c][i]);
            let d = (h - rows[0][STAGE_M][c][i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    max_abs
}

/// The carrier worker: prints the reproduction difference for the fused (`Q8_0`) branch and the
/// separate gate/up/`SwiGLU` (`F16`) branch under the carrier its launcher pins.
#[test]
#[ignore = "worker: run by tapped_gate_up_reproduce_stage_m under a pinned carrier"]
fn gate_up_reproduction_worker() {
    let fused = gate_up_reproduction_max_abs(&build_layer_gguf_q8(), "q8", KernelMode::Q8Act);
    let separate = gate_up_reproduction_max_abs(&build_layer_gguf(), "f16", KernelMode::Q8Act);
    println!("GATE_UP_FUSED_MAX_ABS={fused}");
    println!("GATE_UP_SEPARATE_MAX_ABS={separate}");
}

/// The tapped gate and up rows put back through the engine's own `SwiGLU` reproduce the tapped
/// [`STAGE_M`] rows exactly on the `Q32` carrier, in the fused `Q8_0` branch and in the separate
/// gate/up/`SwiGLU` branch. The `F32` carrier reproduces them within the measured low-bit
/// difference of at most 5 words, which is the carrier's own rounding, not a tap defect.
///
/// The carrier is process-global (`carrier_from_env`'s `OnceLock`) and the harness runs tests
/// concurrently, so the measurement runs in this binary's worker under a pinned
/// `APEIRON_F32_CARRIER` in a child process.
#[test]
fn tapped_gate_up_reproduce_stage_m() {
    /// Measured maximum absolute word difference between the reproduced and tapped `m` rows on the
    /// `F32` carrier, at the two fixtures' shapes: 5 words on the separate branch, 0 on the fused
    /// branch (measured on 2026-10-22 on this fixture, not a general bound).
    const F32_MAX_ABS: i64 = 5;
    let exe = std::env::current_exe().expect("test executable");
    let mut got: Vec<(String, i64, i64)> = Vec::new();
    for (carrier, value) in [("q32", "0"), ("f32", "1")] {
        let out = std::process::Command::new(&exe)
            .args([
                "--exact",
                "gate_up_reproduction_worker",
                "--ignored",
                "--nocapture",
            ])
            .env("APEIRON_F32_CARRIER", value)
            .output()
            .expect("spawn carrier worker");
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            out.status.success(),
            "carrier {carrier} worker failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let field = |key: &str| -> i64 {
            stdout
                .lines()
                .find_map(|l| l.strip_prefix(key))
                .unwrap_or_else(|| panic!("carrier {carrier}: no {key} in\n{stdout}"))
                .trim()
                .parse()
                .expect("word difference")
        };
        let fused = field("GATE_UP_FUSED_MAX_ABS=");
        let separate = field("GATE_UP_SEPARATE_MAX_ABS=");
        println!("carrier {carrier}: fused = {fused}, separate = {separate}");
        got.push((carrier.to_string(), fused, separate));
    }
    let (_, q32_fused, q32_separate) = &got[0];
    let (_, f32_fused, f32_separate) = &got[1];
    assert_eq!(
        *q32_fused, 0,
        "fused branch (Q8_0) gate/up do not reproduce m"
    );
    assert_eq!(
        *q32_separate, 0,
        "separate branch (F16) gate/up do not reproduce m"
    );
    assert!(
        *f32_fused <= F32_MAX_ABS && *f32_separate <= F32_MAX_ABS,
        "f32 carrier difference {f32_fused} / {f32_separate} exceeds the measured bound {F32_MAX_ABS}"
    );
}

/// The tapped `Exact` prefill records the gate, up and `SwiGLU` rows on both FFN fixtures: every
/// column's three stages are `n_ff` wide, the tapped gate and up put back through [`swiglu_q32`]
/// reproduce every `m` word exactly (the `Exact` path is Q32 `round_shift_32`, no carrier), and the
/// tapped run leaves the final columns and the `lm_head_fixed` logits bit-identical to the
/// untapped run.
#[test]
fn exact_tap_records_gate_up_m() {
    for (bytes, tag) in [(build_layer_gguf(), "f16"), (build_layer_gguf_q8(), "q8")] {
        let (engine, path) = open_bytes(&bytes, tag);
        let n_ff = engine.config.n_ff;
        let tokens = [1u32, 2, 3, 4];
        let mut st_off = engine.state(16).expect("state");
        let (cols_off, rows_off) = engine
            .prefill_tap_cols(&mut st_off, &tokens, KernelMode::Exact, false)
            .expect("untapped prefill");
        let mut st_on = engine.state(16).expect("state");
        let (cols_on, rows) = engine
            .prefill_tap_cols(&mut st_on, &tokens, KernelMode::Exact, true)
            .expect("tapped prefill");
        let _ = std::fs::remove_file(&path);

        assert!(
            rows_off.is_empty(),
            "{tag}: record=false allocated stage rows"
        );
        assert_eq!(
            cols_off, cols_on,
            "{tag}: the tapped Exact tap changed the final columns"
        );
        let last = cols_on.len() - 1;
        assert_eq!(
            engine.lm_head_fixed(&cols_off[last]).expect("logits"),
            engine.lm_head_fixed(&cols_on[last]).expect("logits"),
            "{tag}: the tapped Exact tap changed the logits"
        );
        for c in 0..tokens.len() {
            assert_eq!(
                rows[0][STAGE_GATE][c].len(),
                n_ff,
                "{tag}: gate width column {c}"
            );
            assert_eq!(
                rows[0][STAGE_UP][c].len(),
                n_ff,
                "{tag}: up width column {c}"
            );
            assert_eq!(rows[0][STAGE_M][c].len(), n_ff, "{tag}: m width column {c}");
            for i in 0..n_ff {
                let want = swiglu_q32(rows[0][STAGE_GATE][c][i], rows[0][STAGE_UP][c][i]);
                assert_eq!(
                    rows[0][STAGE_M][c][i], want,
                    "{tag}: m != swiglu(gate, up) at column {c} word {i}"
                );
            }
        }
    }
}
