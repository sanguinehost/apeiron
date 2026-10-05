//! Whole-prompt GPU prefill (`GpuDecode::prefill`) against the token-by-token GPU path and the CPU
//! engine, Qwen2.5-0.5B-Q4_K_M.
//!
//! For each of the five prompts of `gpu_decode_step`: prefill the whole prompt in one call, take the
//! next-token argmax, and assert it equals (a) the token-by-token `GpuDecode::step` argmax and
//! (b) the CPU engine's greedy first token. Reports (does not assert) the max relative error of the
//! final-normed hidden at the last prompt position, GPU prefill vs GPU token-by-token, and how many
//! of 8 greedy continuation tokens (decoding on from the prefilled KV cache) agree with the
//! token-by-token path.
//!
//!   cargo test --release --locked --features gpu --test gpu_prefill_match -- --ignored --nocapture
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_decode_step test"
)]

use apeiron::engine::{Engine, KernelMode};
use apeiron::format::bpe::Bpe;
use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::GpuDecode;
use apeiron::gpu::vulkan::Vk;
use apeiron::simt::Carrier;
use std::path::Path;

const MODEL: &str = "~/models/gguf/qwen05b-Q4_K_M.gguf";
const TOKENIZER: &str = "~/models/qwen2.5-0.5b/tokenizer.json";
const PROMPTS: [&str; 5] = [
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water boils at a temperature of",
    "The author of Hamlet is",
    "Two plus two equals",
];
const TMAX: u32 = 256;
const MAXB: usize = 64;
const N_CONT: usize = 8;

fn continue_greedy(d: &mut GpuDecode, first: usize, mut pos: usize) -> Vec<usize> {
    let mut out = vec![first];
    let mut next = first;
    for _ in 1..N_CONT {
        d.step(pos, next);
        pos += 1;
        next = d.last_argmax();
        out.push(next);
    }
    out
}

#[test]
#[ignore = "whole-model GPU run with a CPU reference: release, `--ignored`"]
fn test_gpu_prefill_match_q8() {
    run(Carrier::Q8);
}

#[test]
#[ignore = "whole-model GPU run with a CPU reference: release, `--ignored`"]
fn test_gpu_prefill_match_q4() {
    run(Carrier::Q4);
}

fn run(carrier: Carrier) {
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
    let mut gpu = GpuDecode::new_with_prefill(&gguf, &engine.config, TMAX, carrier, MAXB)
        .expect("GpuDecode::new_with_prefill");

    let mut worst_rel = 0f64;
    let mut results = Vec::new();
    for prompt in PROMPTS {
        let ids = bpe.encode(prompt).expect("encode");
        let toks: Vec<usize> = ids.iter().map(|&t| t as usize).collect();
        let n = toks.len();

        // token-by-token GPU path
        for (pos, &t) in toks.iter().enumerate() {
            gpu.step(pos, t);
        }
        let tbt_first = gpu.last_argmax();
        let tbt_logits: Vec<f32> = gpu.step_logits().to_vec();
        let tbt_hidden: Vec<f64> = gpu.last_hidden().to_vec();
        let tbt_cont = continue_greedy(&mut gpu, tbt_first, n);

        // whole-prompt GPU prefill, then decode on from its KV cache
        let logits_head: Vec<f32> = gpu.prefill(&toks).expect("prefill").to_vec();
        let pf_first = gpu.last_argmax();
        let pf_hidden: Vec<f64> = gpu.last_hidden().to_vec();
        let lg_ref = tbt_logits
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();
        let lg_d = tbt_logits
            .iter()
            .zip(&logits_head)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        println!(
            "  logits rel L2 err (last pos, prefill vs tbt) {:.3e}",
            lg_d / lg_ref
        );
        let host_arg = logits_head
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 {
                    (i, v)
                } else {
                    b
                }
            })
            .0;
        let pf_cont = continue_greedy(&mut gpu, pf_first, n);

        // causality / stale-state control: a longer unrelated prompt leaves its K/V at positions
        // `n..` of the prefill cache; re-running the same prompt must reproduce the first run
        // bit for bit (a mask that lets position i read key i+1 reads that stale data instead)
        let poison: Vec<usize> = (0..48).map(|i| 1000 + 17 * i).collect();
        gpu.prefill(&poison).expect("poison prefill");
        let logits_again: Vec<f32> = gpu.prefill(&toks).expect("prefill again").to_vec();
        let hidden_again: Vec<f64> = gpu.last_hidden().to_vec();
        let repeat_exact = logits_again == logits_head && hidden_again == pf_hidden;

        let mut st = engine.state(n + 1).expect("cpu state");
        // CPU logits at the last prompt position (token-serial exact engine) as the accuracy reference
        let mut st2 = engine.state(n + 1).expect("cpu state");
        let mut cl = Vec::new();
        for &t in &ids {
            cl = engine
                .step_logits_state(&mut st2, t, KernelMode::Exact)
                .expect("cpu step");
        }
        let cpu_logits: Vec<f64> = cl.iter().map(|&v| v as f64 / 4294967296.0).collect();
        let l2f = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
                / b.iter().map(|y| y * y).sum::<f64>().sqrt()
        };
        let pf_l: Vec<f64> = logits_head.iter().map(|&v| v as f64).collect();
        let tb_l: Vec<f64> = tbt_logits.iter().map(|&v| v as f64).collect();
        println!(
            "  logits rel L2 vs CPU exact: prefill {:.3e}, token-by-token {:.3e}",
            l2f(&pf_l, &cpu_logits),
            l2f(&tb_l, &cpu_logits)
        );
        let cpu_first = engine
            .prefill_state(&mut st, &ids, KernelMode::Exact)
            .expect("cpu prefill") as usize;

        let refmax = tbt_hidden.iter().fold(0f64, |m, v| m.max(v.abs()));
        let dmax = tbt_hidden
            .iter()
            .zip(&pf_hidden)
            .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
        let rel = dmax / refmax;
        let l2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let diff: Vec<f64> = tbt_hidden
            .iter()
            .zip(&pf_hidden)
            .map(|(a, b)| a - b)
            .collect();
        let rel_l2 = l2(&diff) / l2(&tbt_hidden);

        // single-token control: attention over one position is trivial, so this isolates the
        // projection / norm / FFN stack from the attention and KV re-layout
        gpu.step(0, toks[0]);
        let h1: Vec<f64> = gpu.last_hidden().to_vec();
        gpu.prefill(&toks[..1]).expect("prefill n=1");
        let d1: Vec<f64> = h1
            .iter()
            .zip(gpu.last_hidden())
            .map(|(a, b)| a - b)
            .collect();
        let rel_l2_n1 = l2(&d1) / l2(&h1);
        worst_rel = worst_rel.max(rel);
        let cont_agree = tbt_cont
            .iter()
            .zip(&pf_cont)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "prompt {prompt:?} n={n}\n  next token: prefill={pf_first} token-by-token={tbt_first} cpu={cpu_first} host-scan={host_arg}\n  hidden(last pos) max|d|={dmax:.3e} max|ref|={refmax:.3e} max rel err={rel:.3e}, rel L2 err={rel_l2:.3e}; n=1 rel L2 err={rel_l2_n1:.3e}\n  continuation agreement {cont_agree}/{N_CONT}: prefill {pf_cont:?} vs tbt {tbt_cont:?}"
        );
        println!("  repeat after a longer unrelated prefill bit-identical: {repeat_exact}");
        results.push((
            prompt,
            pf_first,
            tbt_first,
            cpu_first,
            host_arg,
            repeat_exact,
        ));
    }
    println!("carrier {carrier:?}: worst hidden max rel err {worst_rel:.3e}");
    for (prompt, pf, tbt, cpu, host, repeat_exact) in results {
        assert!(
            repeat_exact,
            "prefill depends on stale cache state for {prompt:?} (causal mask)"
        );
        assert_eq!(
            pf, host,
            "device argmax != host scan of the prefill logits for {prompt:?}"
        );
        assert_eq!(
            pf, tbt,
            "prefill vs token-by-token first token for {prompt:?}"
        );
        assert_eq!(pf, cpu, "prefill vs CPU first token for {prompt:?}");
    }
}
