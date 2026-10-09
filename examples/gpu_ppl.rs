//! Perplexity comparison harness: apeiron GPU decode vs llama.cpp on Qwen2.5-0.5B.
//!
//! Row: `inf-decode-measure-E3CPdJamkoVAQSrj96zC6o` (INF-DECODE-MEASURE).
//!
//! Evaluates PPL over 512-token chunks, scoring the second half of each chunk
//! (positions 256..=510 predicting targets at 257..=511), matching llama.cpp's
//! `tools/perplexity/perplexity.cpp`.
//!
//! Computes PPL from `GpuDecode` logits two ways:
//!   (a) token-by-token: `gpu.step(pos, tok)` for all positions 0..=510
//!   (b) prefill: `gpu.prefill(&chunk[0..256])` then `step(pos, tok)` for 256..=510
//! for both Carrier::Q4 and Carrier::Q8.
//!
//! Also computes CPU engine reference PPL (KernelMode::Exact) over the first 2 chunks.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "measurement harness: numeric casts and timings are required"
)]

use apeiron::engine::{Engine, KernelMode};
use apeiron::format::bpe::Bpe;
use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::GpuDecode;
use apeiron::simt::Carrier;
use std::fs;
use std::path::Path;
use std::process::{exit, Command};
use std::time::Instant;

const DEFAULT_MODEL: &str = concat!(env!("HOME"), "/models/gguf/qwen05b-Q4_K_M.gguf");
const DEFAULT_CORPUS: &str = concat!(env!("HOME"), "/scratch/gpu-ppl-corpus.txt");
const DEFAULT_TOKENIZER: &str = concat!(env!("HOME"), "/models/qwen2.5-0.5b/tokenizer.json");
const CTX: usize = 512;
const FIRST: usize = CTX / 2; // 256

#[derive(Default, Clone, Copy, Debug)]
pub struct PplAccum {
    pub count: usize,
    pub sum_nll: f64,
    pub sum_nll2: f64,
}

impl PplAccum {
    pub fn add(&mut self, nll: f64) {
        self.count += 1;
        self.sum_nll += nll;
        self.sum_nll2 += nll * nll;
    }

    pub fn result(&self) -> (f64, f64) {
        if self.count == 0 {
            return (0.0, 0.0);
        }
        let n = self.count as f64;
        let mean_nll = self.sum_nll / n;
        let ppl = mean_nll.exp();
        let var = (self.sum_nll2 / n) - (mean_nll * mean_nll);
        let stderr = if var > 0.0 && self.count > 1 {
            ppl * (var / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        (ppl, stderr)
    }
}

pub fn log_loss_f32(logits: &[f32], target: usize) -> f64 {
    let mut max_l = f32::NEG_INFINITY;
    for &x in logits {
        if x > max_l {
            max_l = x;
        }
    }
    let mut sum_exp = 0.0f64;
    for &x in logits {
        sum_exp += ((x - max_l) as f64).exp();
    }
    let log_sum_exp = sum_exp.ln();
    let target_logit = logits[target] as f64;
    (max_l as f64) + log_sum_exp - target_logit
}

pub fn log_loss_i64(logits: &[i64], target: usize) -> f64 {
    let mut max_l = i64::MIN;
    for &x in logits {
        if x > max_l {
            max_l = x;
        }
    }
    let mut sum_exp = 0.0f64;
    for &x in logits {
        let diff = (x - max_l) as f64 / 4_294_967_296.0;
        sum_exp += diff.exp();
    }
    let log_sum_exp = sum_exp.ln();
    let target_diff = (logits[target] - max_l) as f64 / 4_294_967_296.0;
    log_sum_exp - target_diff
}

fn check_tokenization(apeiron_tokens: &[u32], corpus_path: &str, model_path: &str) -> bool {
    let candidates = [
        concat!(
            env!("HOME"),
            "/Workspace/llama.cpp/build-cuda/bin/llama-tokenize"
        ),
        concat!(
            env!("HOME"),
            "/Workspace/llama.cpp/build-vulkan/bin/llama-tokenize"
        ),
        concat!(
            env!("HOME"),
            "/Workspace/llama.cpp/build/bin/llama-tokenize"
        ),
    ];
    let Some(bin) = candidates.iter().find(|p| Path::new(p).exists()) else {
        println!("[TOKENIZE-CHECK] llama-tokenize binary not found; skipped external check");
        return true;
    };

    println!("[TOKENIZE-CHECK] running {bin} on {corpus_path} (--no-escape)...");
    let output = match Command::new(bin)
        .arg("-m")
        .arg(model_path)
        .arg("-f")
        .arg(corpus_path)
        .arg("--no-escape")
        .arg("--ids")
        .output()
    {
        Ok(out) => out,
        Err(e) => {
            println!("[TOKENIZE-CHECK] failed to execute {bin}: {e}");
            return false;
        }
    };

    if !output.status.success() {
        println!(
            "[TOKENIZE-CHECK] {bin} failed with status {}",
            output.status
        );
        return false;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        println!("[TOKENIZE-CHECK] unexpected output format from llama-tokenize");
        return false;
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    let llama_tokens: Vec<u32> = inner
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();

    let min_len = apeiron_tokens.len().min(llama_tokens.len());
    let mut mismatches = 0;
    for (i, (&a, &b)) in apeiron_tokens[..min_len]
        .iter()
        .zip(&llama_tokens[..min_len])
        .enumerate()
    {
        if a != b {
            if mismatches < 5 {
                println!("[TOKENIZE-CHECK] mismatch at index {i}: apeiron={a} llama={b}");
            }
            mismatches += 1;
        }
    }

    if apeiron_tokens.len() != llama_tokens.len() {
        println!(
            "[TOKENIZE-CHECK] LENGTH MISMATCH: apeiron {} tokens, llama.cpp {} tokens (first diff at index {})",
            apeiron_tokens.len(),
            llama_tokens.len(),
            apeiron_tokens
                .iter()
                .zip(&llama_tokens)
                .position(|(a, b)| a != b)
                .unwrap_or(min_len)
        );
        return false;
    }

    if mismatches == 0 {
        println!(
            "[TOKENIZE-CHECK] EXACT MATCH: all {} tokens match llama.cpp bit-for-bit",
            apeiron_tokens.len()
        );
        true
    } else {
        println!(
            "[TOKENIZE-CHECK] MISMATCH: {mismatches} / {} tokens differ",
            apeiron_tokens.len()
        );
        false
    }
}

fn sha256_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone)]
struct RunRecord {
    arm: String,
    ppl_2chunk: (f64, f64),
    ppl_final: (f64, f64),
    chunks: usize,
    tokens: usize,
    elapsed_sec: f64,
    tok_per_sec: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let model_path = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .or_else(|| flag("--model"))
        .unwrap_or_else(|| DEFAULT_MODEL.into());

    let corpus_path = flag("--file")
        .or_else(|| flag("-f"))
        .or_else(|| flag("--corpus"))
        .unwrap_or_else(|| DEFAULT_CORPUS.into());

    let tok_path = flag("--tokenizer").unwrap_or_else(|| DEFAULT_TOKENIZER.into());
    let max_chunks: Option<usize> = flag("--chunks").and_then(|v| v.parse().ok());
    let cpu_chunks: usize = flag("--cpu-chunks")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let start: usize = flag("--start").and_then(|v| v.parse().ok()).unwrap_or(0);
    let fresh = args.iter().any(|a| a == "--fresh");
    let carrier_opt = flag("--carrier").unwrap_or_else(|| "all".into());

    println!("=== Apeiron GPU / CPU Perplexity Comparator ===");
    println!("Model:      {model_path}");
    println!("Corpus:     {corpus_path}");
    println!("Tokenizer:  {tok_path}");

    let corpus_bytes = fs::read(&corpus_path).expect("read corpus file");
    let corpus_sha = sha256_digest(&corpus_bytes);
    println!(
        "Corpus:     {} bytes, sha256 {}",
        corpus_bytes.len(),
        corpus_sha
    );

    let text = String::from_utf8(corpus_bytes).expect("valid utf-8 corpus text");

    let t0 = Instant::now();
    let bpe = Bpe::from_tokenizer_json(&tok_path).expect("load tokenizer");
    let tokens = bpe.encode(&text).expect("encode text");
    println!(
        "Tokenized in {:.2} ms: {} tokens",
        t0.elapsed().as_secs_f64() * 1000.0,
        tokens.len()
    );

    let tok_ok = check_tokenization(&tokens, &corpus_path, &model_path);
    if !tok_ok {
        eprintln!("WARNING: Token ids differ from llama.cpp. PPL comparison may be invalid.");
    }

    let n_chunk_avail = tokens.len() / CTX;
    let n_chunk = max_chunks.unwrap_or(n_chunk_avail).min(n_chunk_avail);
    println!(
        "Evaluating over {n_chunk} chunks (n_ctx={CTX}, scored per chunk={})\n",
        CTX - 1 - FIRST
    );

    let gguf = Gguf::open(&model_path).expect("gguf open");
    let engine = Engine::open(&model_path).expect("engine open");

    let carriers = match carrier_opt.as_str() {
        "q4" => vec![(Carrier::Q4, "q4")],
        "q8" => vec![(Carrier::Q8, "q8")],
        _ => vec![(Carrier::Q4, "q4"), (Carrier::Q8, "q8")],
    };

    let mut records: Vec<RunRecord> = Vec::new();

    // 1. GPU token-by-token
    for (carrier, name) in carriers.iter().filter(|_| std::env::var("NOTBT").is_err()) {
        let t_start = Instant::now();
        let mut gpu =
            GpuDecode::new(&gguf, &engine.config, CTX as u32, *carrier).expect("GpuDecode::new");
        let mut accum = PplAccum::default();
        let mut ppl_2chunk = (0.0, 0.0);

        print!("[GPU-TBT]  carrier={name:<2} | ");
        for c in start..n_chunk {
            let chunk = &tokens[c * CTX..(c + 1) * CTX];
            if fresh {
                gpu = GpuDecode::new(&gguf, &engine.config, CTX as u32, *carrier).expect("new");
            }
            let mut ca = PplAccum::default();
            for pos in 0..CTX - 1 {
                let logits = gpu.step(pos, chunk[pos] as usize);
                if pos >= FIRST {
                    let target = chunk[pos + 1] as usize;
                    let nll = log_loss_f32(logits, target);
                    accum.add(nll);
                    ca.add(nll);
                }
            }
            println!("PERCHUNK tbt {name} {c} {:.4}", ca.result().0);
            if c + 1 == 2 {
                ppl_2chunk = accum.result();
            }
            if c < 4 || c + 1 == n_chunk {
                print!("[{}]{:.4},", c + 1, accum.result().0);
            } else if c == 4 {
                print!("...,");
            }
        }
        println!();
        let elapsed = t_start.elapsed().as_secs_f64();
        let (ppl, stderr) = accum.result();
        if n_chunk < 2 {
            ppl_2chunk = (ppl, stderr);
        }
        let total_tokens_run = n_chunk * (CTX - 1);
        let tok_rate = total_tokens_run as f64 / elapsed;
        println!(
            "  Final: {n_chunk} chunks ({} scored) | PPL = {:.4} +/- {:.5} | 2-chunk = {:.4} | elapsed {:.2}s ({:.1} tok/s)",
            accum.count, ppl, stderr, ppl_2chunk.0, elapsed, tok_rate
        );
        records.push(RunRecord {
            arm: format!("apeiron GPU TBT {name}"),
            ppl_2chunk,
            ppl_final: (ppl, stderr),
            chunks: n_chunk,
            tokens: accum.count,
            elapsed_sec: elapsed,
            tok_per_sec: tok_rate,
        });
    }

    // 2. GPU prefill of first half (0..256) + step for scored half (256..=510)
    for (carrier, name) in &carriers {
        let t_start = Instant::now();
        let mut gpu =
            GpuDecode::new_with_prefill(&gguf, &engine.config, CTX as u32, *carrier, FIRST)
                .expect("GpuDecode::new_with_prefill");
        let mut accum = PplAccum::default();
        let mut ppl_2chunk = (0.0, 0.0);

        print!("[GPU-PF]   carrier={name:<2} | ");
        for c in start..n_chunk {
            let chunk = &tokens[c * CTX..(c + 1) * CTX];
            if fresh {
                gpu =
                    GpuDecode::new_with_prefill(&gguf, &engine.config, CTX as u32, *carrier, FIRST)
                        .expect("new");
            }
            let mut ca = PplAccum::default();
            let prefill_toks: Vec<usize> = chunk[..FIRST].iter().map(|&t| t as usize).collect();
            gpu.prefill(&prefill_toks).expect("gpu.prefill");

            for pos in FIRST..CTX - 1 {
                let logits = gpu.step(pos, chunk[pos] as usize);
                let target = chunk[pos + 1] as usize;
                let nll = log_loss_f32(logits, target);
                accum.add(nll);
                ca.add(nll);
            }
            println!("PERCHUNK pf {name} {c} {:.4}", ca.result().0);
            if c + 1 == 2 {
                ppl_2chunk = accum.result();
            }
            if c < 4 || c + 1 == n_chunk {
                print!("[{}]{:.4},", c + 1, accum.result().0);
            } else if c == 4 {
                print!("...,");
            }
        }
        println!();
        let elapsed = t_start.elapsed().as_secs_f64();
        let (ppl, stderr) = accum.result();
        if n_chunk < 2 {
            ppl_2chunk = (ppl, stderr);
        }
        let total_tokens_run = n_chunk * (CTX - 1 - FIRST);
        let tok_rate = total_tokens_run as f64 / elapsed;
        println!(
            "  Final: {n_chunk} chunks ({} scored) | PPL = {:.4} +/- {:.5} | 2-chunk = {:.4} | elapsed {:.2}s ({:.1} tok/s)",
            accum.count, ppl, stderr, ppl_2chunk.0, elapsed, tok_rate
        );
        records.push(RunRecord {
            arm: format!("apeiron GPU PF {name}"),
            ppl_2chunk,
            ppl_final: (ppl, stderr),
            chunks: n_chunk,
            tokens: accum.count,
            elapsed_sec: elapsed,
            tok_per_sec: tok_rate,
        });
    }

    // 3. CPU engine reference on first cpu_chunks (KernelMode::Exact)
    if cpu_chunks > 0 && n_chunk > 0 {
        let chunks_to_run = cpu_chunks.min(n_chunk);
        let t_start = Instant::now();
        let mut accum = PplAccum::default();
        let mut ppl_2chunk = (0.0, 0.0);

        print!("[CPU-EXACT]            | ");
        for c in 0..chunks_to_run {
            let chunk = &tokens[c * CTX..(c + 1) * CTX];
            let mut st = engine.state(CTX).expect("cpu decode state");

            for pos in 0..CTX - 1 {
                let logits = engine
                    .step_logits_state(&mut st, chunk[pos], KernelMode::Exact)
                    .expect("cpu step_logits_state");
                if pos >= FIRST {
                    let target = chunk[pos + 1] as usize;
                    let nll = log_loss_i64(&logits, target);
                    accum.add(nll);
                }
            }
            if c + 1 == 2 {
                ppl_2chunk = accum.result();
            }
            print!("[{}]{:.4},", c + 1, accum.result().0);
        }
        println!();
        let elapsed = t_start.elapsed().as_secs_f64();
        let (ppl, stderr) = accum.result();
        if chunks_to_run < 2 {
            ppl_2chunk = (ppl, stderr);
        }
        let total_tokens_run = chunks_to_run * (CTX - 1);
        let tok_rate = total_tokens_run as f64 / elapsed;
        println!(
            "  Final: {chunks_to_run} chunks ({} scored) | PPL = {:.4} +/- {:.5} | elapsed {:.2}s ({:.2} tok/s)",
            accum.count, ppl, stderr, elapsed, tok_rate
        );
        records.push(RunRecord {
            arm: "apeiron CPU Exact".into(),
            ppl_2chunk,
            ppl_final: (ppl, stderr),
            chunks: chunks_to_run,
            tokens: accum.count,
            elapsed_sec: elapsed,
            tok_per_sec: tok_rate,
        });
    }

    println!("\n=== Summary Table ===");
    println!(
        "| {:<24} | {:<6} | {:<11} | {:<12} | {:<18} | {:<8} | {:<9} |",
        "Arm", "Chunks", "Scored Toks", "2-Chunk PPL", "Final PPL +/- SE", "tok/s", "Wall Time"
    );
    println!(
        "|{:-<26}|{:-<8}|{:-<13}|{:-<14}|{:-<20}|{:-<10}|{:-<11}|",
        "", "", "", "", "", "", ""
    );
    for r in &records {
        let final_s = format!("{:.4} +/- {:.5}", r.ppl_final.0, r.ppl_final.1);
        let c2_s = format!("{:.4}", r.ppl_2chunk.0);
        let time_s = format!("{:.1} s", r.elapsed_sec);
        let rate_s = format!("{:.1}", r.tok_per_sec);
        println!(
            "| {:<24} | {:<6} | {:<11} | {:<12} | {:<18} | {:<8} | {:<9} |",
            r.arm, r.chunks, r.tokens, c2_s, final_s, rate_s, time_s
        );
    }

    println!("\nMeasurement completed successfully.");
    exit(0);
}
