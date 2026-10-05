//! Whole-model GPU greedy decode (`GpuDecode`, Qwen2.5-0.5B-Q4_K_M) against apeiron's CPU engine.
//!
//! Five fixed prompts, each fed token by token on the GPU and then greedy-decoded for 32 tokens.
//! Asserts (a) the GPU's first generated token equals the CPU engine's greedy first token for every
//! prompt, and (b) the GPU text for "The capital of France is" contains "Paris". Reports (does not
//! assert) the per-prompt greedy agreement length with the CPU engine over 32 tokens, the GPU texts
//! and the decode speed (tokens/s over 128 greedy steps after warm-up, 3 runs).
//!
//! The CPU engine is the slow side (about 1.5 s per token in a release build, far more in the debug
//! profile `cargo test` uses), so the CPU reference decodes `APEIRON_GPU_CPU_TOKENS` tokens per
//! prompt (default 1: the first token, enough for assertion (a)). The agreement report covers that
//! many tokens; run with `APEIRON_GPU_CPU_TOKENS=32 cargo test --release ...` for the full 32.
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's modelfold::gpu Vulkan runtime; lint profile is fold's, not apeiron's"
)]

use apeiron::engine::{Engine, KernelMode};
use apeiron::format::bpe::Bpe;
use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::GpuDecode;
use apeiron::gpu::vulkan::Vk;
use apeiron::simt::Carrier;
use std::path::Path;
use std::time::Instant;

const MODEL: &str = "~/models/gguf/qwen05b-Q4_K_M.gguf";
const TOKENIZER: &str = "~/models/qwen2.5-0.5b/tokenizer.json";
const PROMPTS: [&str; 5] = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water boils at a temperature of",
    "The author of Hamlet is",
    "Two plus two equals",
];
const N_GEN: usize = 32;
const TMAX: u32 = 256;
const BENCH_STEPS: usize = 128;
const BENCH_RUNS: usize = 3;

/// Feed `ids` at positions `0..`, then greedy-decode `n` tokens. Returns the `n` generated ids.
fn gpu_generate(d: &mut GpuDecode, ids: &[u32], n: usize) -> Vec<u32> {
    for (pos, &t) in ids.iter().enumerate() {
        d.step(pos, t as usize);
    }
    let mut out = Vec::with_capacity(n);
    let mut next = d.last_argmax() as u32;
    let mut pos = ids.len();
    for i in 0..n {
        out.push(next);
        if i + 1 == n {
            break;
        }
        d.step(pos, next as usize);
        pos += 1;
        next = d.last_argmax() as u32;
    }
    out
}

#[test]
#[ignore = "whole-model decode with a CPU reference: run in release, `cargo test --release --features gpu --test gpu_decode_step -- --ignored`"]
fn test_gpu_decode_step() {
    if unsafe { Vk::load() }.is_err() {
        println!("SKIP: No Vulkan device present");
        return;
    }
    if !Path::new(MODEL).exists() || !Path::new(TOKENIZER).exists() {
        println!("SKIP: model or tokenizer not found");
        return;
    }
    let gguf = Gguf::open(MODEL).expect("gguf open");
    let engine = Engine::open(MODEL).expect("engine open");
    let bpe = Bpe::from_tokenizer_json(TOKENIZER).expect("tokenizer");

    let n_cpu: usize = std::env::var("APEIRON_GPU_CPU_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
        .clamp(1, N_GEN);
    let t0 = Instant::now();
    let mut gpu = GpuDecode::new(&gguf, &engine.config, TMAX, Carrier::Q8).expect("GpuDecode::new");
    println!(
        "GpuDecode::new {:.2} s, {} dispatches per token",
        t0.elapsed().as_secs_f64(),
        gpu.n_disp
    );

    let mut first_match = Vec::new();
    for prompt in PROMPTS {
        let ids = bpe.encode(prompt).expect("encode");
        let g = gpu_generate(&mut gpu, &ids, N_GEN);

        let mode = KernelMode::Exact;
        let mut st = engine.state(ids.len() + n_cpu).expect("cpu state");
        let mut next = engine
            .prefill_state(&mut st, &ids, mode)
            .expect("cpu prefill");
        let mut c = Vec::with_capacity(N_GEN);
        for i in 0..n_cpu {
            c.push(next);
            if i + 1 == n_cpu {
                break;
            }
            next = engine
                .next_token_state(&mut st, next, mode)
                .expect("cpu step");
        }
        let agree = g.iter().zip(&c).take_while(|(a, b)| a == b).count();
        let text = bpe
            .decode(&g)
            .unwrap_or_else(|e| format!("<decode refused: {e}>"));
        println!(
            "prompt {prompt:?} ids {ids:?}\n  first token gpu={} cpu={} match={}\n  agreement {agree}/{n_cpu} (cpu tokens decoded; gpu decoded {N_GEN})\n  gpu text {text:?}\n  gpu ids {g:?}\n  cpu ids {c:?}",
            g[0],
            c[0],
            g[0] == c[0]
        );
        first_match.push((prompt, g[0], c[0]));
        if prompt == PROMPTS[0] {
            assert!(text.contains("Paris"), "GPU text lacks Paris: {text:?}");
        }
    }
    for (prompt, g, c) in &first_match {
        assert_eq!(g, c, "first token mismatch for {prompt:?}: gpu {g} cpu {c}");
    }

    // decode speed: BENCH_STEPS greedy steps after the prompt, warm-up discarded
    let ids = bpe.encode(PROMPTS[0]).expect("encode");
    let mut tps = Vec::new();
    for run in 0..=BENCH_RUNS {
        for (pos, &t) in ids.iter().enumerate() {
            gpu.step(pos, t as usize);
        }
        let mut next = gpu.last_argmax();
        let mut pos = ids.len();
        let steps = if run == 0 { 16 } else { BENCH_STEPS };
        let t = Instant::now();
        for _ in 0..steps {
            gpu.step(pos, next);
            pos += 1;
            next = gpu.last_argmax();
        }
        let r = steps as f64 / t.elapsed().as_secs_f64();
        if run > 0 {
            tps.push(r);
        }
    }
    let mean = tps.iter().sum::<f64>() / tps.len() as f64;
    let (lo, hi) = tps
        .iter()
        .fold((f64::MAX, 0f64), |(l, h), &v| (l.min(v), h.max(v)));
    println!("decode tok/s over {BENCH_STEPS} steps, {BENCH_RUNS} runs: mean {mean:.1} min {lo:.1} max {hi:.1} (shared box)");
}
