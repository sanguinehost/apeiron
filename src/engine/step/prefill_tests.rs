#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test-only fixture generation"
)]

use super::*;
use crate::format::quant::{GGML_TYPE_F16, GGML_TYPE_F32};

const F_E: usize = 8;
const F_FF: usize = 128;
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

fn f16_weight(rows: usize, cols: usize, salt: usize) -> Vec<u8> {
    const SCALE: f32 = 1.0 / 32.0;
    let mut b: Vec<u8> = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        for c in 0..cols {
            let code = i32::try_from((r * 7 + c * 3 + salt * 11) % 5).expect("small") - 2;
            b.extend_from_slice(&f16_round(code as f32 * SCALE).to_le_bytes());
        }
    }
    b
}

fn f32_gain(n: usize) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(n * 4);
    for _ in 0..n {
        b.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
    }
    b
}

fn build_layer_gguf() -> Vec<u8> {
    let ts: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        (
            "blk.0.attn_norm.weight",
            vec![F_E as u64],
            GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "blk.0.attn_q.weight",
            vec![F_E as u64, F_Q as u64],
            GGML_TYPE_F16,
            f16_weight(F_Q, F_E, 1),
        ),
        (
            "blk.0.attn_k.weight",
            vec![F_E as u64, F_KO as u64],
            GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 2),
        ),
        (
            "blk.0.attn_v.weight",
            vec![F_E as u64, F_KO as u64],
            GGML_TYPE_F16,
            f16_weight(F_KO, F_E, 3),
        ),
        (
            "blk.0.attn_output.weight",
            vec![F_E as u64, F_E as u64],
            GGML_TYPE_F16,
            f16_weight(F_E, F_E, 4),
        ),
        (
            "blk.0.ffn_norm.weight",
            vec![F_E as u64],
            GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "blk.0.ffn_gate.weight",
            vec![F_E as u64, F_FF as u64],
            GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 5),
        ),
        (
            "blk.0.ffn_up.weight",
            vec![F_E as u64, F_FF as u64],
            GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 6),
        ),
        (
            "blk.0.ffn_down.weight",
            vec![F_FF as u64, F_E as u64],
            GGML_TYPE_F16,
            f16_weight(F_E, F_FF, 7),
        ),
        (
            "output_norm.weight",
            vec![F_E as u64],
            GGML_TYPE_F32,
            f32_gain(F_E),
        ),
        (
            "token_embd.weight",
            vec![F_E as u64, F_FF as u64],
            GGML_TYPE_F16,
            f16_weight(F_FF, F_E, 9),
        ),
    ];
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(b"GGUF");
    put_u32(&mut b, 3);
    put_u64(&mut b, ts.len() as u64);
    put_u64(&mut b, 6);
    kv_str(&mut b, "general.architecture", "qwen2");
    kv_u32(&mut b, "qwen2.block_count", 1);
    kv_u32(&mut b, "qwen2.attention.head_count", 2);
    kv_u32(&mut b, "qwen2.attention.head_count_kv", 1);
    kv_u32(&mut b, "qwen2.embedding_length", F_E as u32);
    kv_u32(&mut b, "qwen2.feed_forward_length", F_FF as u32);
    let mut off = 0u64;
    for (name, dims, ty, payload) in &ts {
        put_str(&mut b, name);
        put_u32(&mut b, dims.len() as u32);
        for &d in dims {
            put_u64(&mut b, d);
        }
        put_u32(&mut b, *ty);
        put_u64(&mut b, off);
        off += payload.len() as u64;
    }
    let pad = (32 - b.len() % 32) % 32;
    b.resize(b.len() + pad, 0);
    for (_, _, _, payload) in &ts {
        b.extend_from_slice(payload);
    }
    b
}

fn open_bytes(tag: &str, bytes: &[u8]) -> (Engine, String) {
    let path = std::env::temp_dir().join(format!("apeiron_prefill_{}_{tag}", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    let path = path.to_str().expect("utf8 temp path").to_string();
    (Engine::open(&path).expect("open fixture"), path)
}

#[test]
fn prefill_state_matches_two_phase_computation() {
    let (engine, path) = open_bytes("prefill_test", &build_layer_gguf());
    let prompt_fixture = [
        0u32, 1, 2, 5, 9, 3, 7, 4, 8, 6, 0, 2, 4, 1, 3, 5, 7, 9, 8, 6, 1, 0, 5, 2, 8, 4, 9, 7, 3,
        6, 2, 1, 4, 0, 7, 5, 6, 8, 9, 3, 4, 7, 1, 8, 0, 9, 2, 6, 5, 3, 8, 1, 6, 0, 4, 9, 7, 2, 5,
        3, 9, 4, 2, 7, 1, 8, 5, 0, 6, 3, 5, 8, 0, 4, 7, 1, 9, 3, 6, 2, 3, 6, 9, 1, 4, 7, 0, 8, 2,
        5, 7, 0, 3, 6, 9, 2, 5, 8, 1, 4, 2, 5, 8, 1, 4, 7, 0, 3, 6, 9, 6, 9, 2, 5, 8, 1, 4, 7, 0,
        3, 0, 3, 6, 9, 2, 5, 8, 1, 4, 7, 1, 4, 7, 0, 3, 6, 9, 2, 5, 8,
    ];

    for &w in &[1usize, 3, 64] {
        let lens = [1usize, 2, w.saturating_sub(1).max(1), w, w + 1, 2 * w + 3];
        for &len in &lens {
            let prompt = &prompt_fixture[..len];

            // 1. Two-phase computation (reference):
            // prefill_cols over head + next_token_state over last
            let mut st_ref = engine.state(len + 2).expect("state ref");
            let (last, head) = prompt.split_last().expect("non-empty");
            // Run blocks of width w
            for chunk in head.chunks(w) {
                engine
                    .compose_cols(
                        &mut st_ref.kv,
                        st_ref.affine.as_mut(),
                        chunk,
                        Sched::Resident,
                        KernelMode::Exact,
                        false,
                        None,
                    )
                    .expect("compose_cols");
            }
            let tok_ref = engine
                .next_token_state(&mut st_ref, *last, KernelMode::Exact)
                .expect("next_token_state");
            let dec_ref = engine
                .step_state(&mut st_ref, tok_ref, KernelMode::Exact)
                .expect("step_state ref")
                .token;

            // 2. prefill_state (new single-pass block computation):
            // Override prefill block width to w
            std::env::set_var("APEIRON_PREFILL_COLS", w.to_string());
            let mut st_act = engine.state(len + 2).expect("state act");
            let tok_act = engine
                .prefill_state_with_width(&mut st_act, prompt, KernelMode::Exact, w)
                .expect("prefill_state");
            let dec_act = engine
                .step_state(&mut st_act, tok_act, KernelMode::Exact)
                .expect("step_state act")
                .token;

            assert_eq!(
                tok_act, tok_ref,
                "token mismatch at w={w}, len={len}: got {tok_act}, expected {tok_ref}"
            );
            assert_eq!(
                dec_act, dec_ref,
                "decode token mismatch at w={w}, len={len}: got {dec_act}, expected {dec_ref}"
            );
        }
    }
    std::env::remove_var("APEIRON_PREFILL_COLS");
    let _ = std::fs::remove_file(&path);
}
