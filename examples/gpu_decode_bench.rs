// Row: `inf-decode-measure-E3CPdJamkoVAQSrj96zC6o` (INF-DECODE-MEASURE): the tg comparator against llama-bench.
//! Whole-model GPU greedy decode throughput (`GpuDecode`), the apeiron side of the llama-bench `tg`
//! comparison. Ported from fold `bedrock/examples/gpu_decode_bench.rs` (d497161f).
//!
//! The model is constructed once, warmed for 16 steps, then each of `--reps` repetitions replays
//! `--gen` greedy steps from the same prefix (the device argmax feeds the next token, as `llama-bench
//! -n` feeds a fixed token and runs no sampler). After the timed repetitions an untimed pass over the
//! same steps compares the device argmax with a host scan of the full logit row (fold's cross-check).
//!
//!   cargo run --release --locked --features gpu --example gpu_decode_bench -- <gguf> --gen 128 --reps 5 --carrier q4
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "measurement harness: casts over counters and timings are the point"
)]

use apeiron::engine::Engine;
use apeiron::format::bpe::Bpe;
use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::GpuDecode;
use apeiron::simt::Carrier;
use std::process::exit;
use std::time::Instant;

const PROMPT: &str = "The capital of France is";
const WARM: usize = 16;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: gpu_decode_bench <gguf> [--gen N] [--reps R] [--carrier q4|q8] [--tmax T] [--tokenizer PATH]");
        exit(2);
    };
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let gen_n: usize = flag("--gen").and_then(|v| v.parse().ok()).unwrap_or(128);
    let reps: usize = flag("--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let tmax: u32 = flag("--tmax").and_then(|v| v.parse().ok()).unwrap_or(512);
    let carrier_s = flag("--carrier").unwrap_or_else(|| "q4".into());
    let carrier = match carrier_s.as_str() {
        "q4" => Carrier::Q4,
        "q8" => Carrier::Q8,
        other => {
            eprintln!("--carrier {other}: expected q4 or q8");
            exit(2);
        }
    };
    let tok_path = flag("--tokenizer")
        .unwrap_or_else(|| "/home/socol/models/qwen2.5-0.5b/tokenizer.json".into());

    let gguf = Gguf::open(path).expect("gguf open");
    let engine = Engine::open(path).expect("engine open");
    let bpe = Bpe::from_tokenizer_json(&tok_path).expect("tokenizer");
    let ids = bpe.encode(PROMPT).expect("encode");
    assert!(ids.len() + WARM + gen_n < tmax as usize, "tmax too small");

    let t0 = Instant::now();
    let mut gpu = GpuDecode::new(&gguf, &engine.config, tmax, carrier).unwrap_or_else(|e| {
        eprintln!("GpuDecode::new: {e}");
        exit(2);
    });
    println!(
        "[GPU-DECODE] {path} | carrier {carrier_s} | construct {:.2} s | {} dispatches/token | tmax {tmax}",
        t0.elapsed().as_secs_f64(),
        gpu.n_disp
    );

    // prefix: the prompt minus its last token; the last token is the first timed input
    let mut pos = 0usize;
    for &t in &ids[..ids.len() - 1] {
        gpu.step(pos, t as usize);
        pos += 1;
    }
    let (prefix_pos, prefix_tok) = (pos, ids[ids.len() - 1] as usize);

    let (mut p, mut c) = (prefix_pos, prefix_tok);
    for _ in 0..WARM {
        gpu.step(p, c);
        c = gpu.last_argmax();
        p += 1;
    }

    let mut rates = Vec::with_capacity(reps);
    let mut seq: Vec<usize> = Vec::new();
    for rep in 0..reps {
        let (mut p, mut c) = (prefix_pos, prefix_tok);
        let t = Instant::now();
        for _ in 0..gen_n {
            gpu.step(p, c);
            c = gpu.last_argmax();
            if rep == 0 {
                seq.push(c);
            }
            p += 1;
        }
        rates.push(gen_n as f64 / t.elapsed().as_secs_f64());
    }

    // untimed cross-check: device argmax vs host scan over the same steps
    let host_argmax = |v: &[f32]| -> usize {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        v.iter().position(|&x| x == m).unwrap_or(0)
    };
    let (mut p, mut c) = (prefix_pos, prefix_tok);
    let mut mismatches = 0usize;
    let mut check_seq = Vec::with_capacity(gen_n);
    for _ in 0..gen_n {
        let h = host_argmax(gpu.step(p, c));
        let d = gpu.last_argmax();
        mismatches += usize::from(h != d);
        check_seq.push(d);
        c = d;
        p += 1;
    }
    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in &seq {
        for b in (t as u64).to_le_bytes() {
            fnv ^= u64::from(b);
            fnv = fnv.wrapping_mul(0x0100_0000_01b3);
        }
    }
    let text = bpe
        .decode(&seq.iter().take(24).map(|&t| t as u32).collect::<Vec<_>>())
        .unwrap_or_default();
    println!("  prompt {PROMPT:?} -> {text:?}");
    println!(
        "  SEQHASH {fnv:016x} over {} tokens; rep0 == check pass: {}",
        seq.len(),
        seq == check_seq
    );
    println!(
        "  device argmax vs host scan over {gen_n} timed steps: {mismatches} mismatches{}",
        if mismatches == 0 {
            ""
        } else {
            "  ** GATE FAILS **"
        }
    );
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let lo = rates.iter().copied().fold(f64::MAX, f64::min);
    let hi = rates.iter().copied().fold(0f64, f64::max);
    let rs: Vec<String> = rates.iter().map(|r| format!("{r:.1}")).collect();
    println!("  reps in temporal order: [{}]", rs.join(", "));
    println!(
        "  BENCH carrier={carrier_s} tg{gen_n}: mean {mean:.1} tok/s min {lo:.1} max {hi:.1} over {reps} reps"
    );
    // The Vulkan loader's atexit teardown intermittently segfaults (exit 139, after every line above
    // is printed) because `GpuDecode` is never destroyed; `_exit` skips it so `$?` carries the gate.
    use std::io::Write;
    let _ = std::io::stdout().flush();
    extern "C" {
        fn _exit(code: i32) -> !;
    }
    // SAFETY: `_exit` terminates the process immediately; stdout is flushed above.
    unsafe { _exit(i32::from(mismatches != 0)) }
}
