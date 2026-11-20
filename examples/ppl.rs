//! Perplexity of a GGUF (optionally with a `NanoQuant` pack) on a raw text file.
//!
//! Run:
//! `cargo run --release --locked --example ppl -- --model <gguf> --tokenizer <tokenizer.json>
//!  --data <wiki.test.raw> [--pack <model.nqp>] [--tokens 8192] [--skip 0] [--ctx 512]
//!  [--head-batch 16] [--kernel q8act|ternary|exact] [--trace]`
//!
//! The text is tokenized once, `--skip` tokens are dropped, the rest is truncated to `--tokens`,
//! and scored in consecutive non-overlapping windows of at most `--ctx` tokens: each window is one
//! prefill, every position but the last predicts the next token, and the cross-entropy is the
//! softmax of [`Engine::lm_head_cols`]. The number is `exp(Σ −ln p(target))` over the scored
//! positions.
//!
//! `--pack` installs the sign-rank factors over the GGUF skeleton ([`Engine::open_with_nq`]). The
//! kernel mode then selects the activation reader over those factors: `q8act` quantizes each stage
//! activation to one int8 code per element (`NqLinear::project_q8`), `ternary` quantizes it to one
//! trit per term with one scale per 64 (`NqLinear::project_ternary`), `exact` reads
//! the Q32.32 words (`NqLinear::project`), and the same mode selects the head and any non-fitted
//! tensor. Without `--pack` the file's own weights run the dense path. The head is projected in
//! `--head-batch` columns at a time, which bounds the `vocab × batch` logit buffer.
//!
//! Domain: the file must tokenize, every id must be below the model vocabulary, and at least one
//! window must hold two tokens; the text is truncated, never padded. Cost is one prefill pass over
//! `--tokens` tokens plus one head projection per scored position; space is `O(ctx·n_embd)` states
//! and `O(head-batch·vocab)` logits.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the perplexity is an f64 report over counts far below 2^53; the token count feeds a usize window"
)]

use std::error::Error;
use std::process::ExitCode;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::bpe::Bpe;
use apeiron::format::nq;

struct Args {
    model: String,
    pack: Option<String>,
    tokenizer: String,
    data: String,
    tokens: usize,
    skip: usize,
    ctx: usize,
    head_batch: usize,
    mode: KernelMode,
    trace: bool,
    dump: bool,
    second_half: bool,
    ids_out: Option<String>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        model: String::new(),
        pack: None,
        tokenizer: String::new(),
        data: String::new(),
        tokens: 8192,
        skip: 0,
        ctx: 512,
        head_batch: 16,
        mode: KernelMode::Q8Act,
        trace: false,
        dump: false,
        second_half: false,
        ids_out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--pack" => a.pack = Some(val()?),
            "--tokenizer" => a.tokenizer = val()?,
            "--data" => a.data = val()?,
            "--tokens" => a.tokens = val()?.parse()?,
            "--skip" => a.skip = val()?.parse()?,
            "--ctx" => a.ctx = val()?.parse()?,
            "--head-batch" => a.head_batch = val()?.parse()?,
            "--trace" => a.trace = true,
            "--dump" => a.dump = true,
            "--second-half" => a.second_half = true,
            "--ids-out" => a.ids_out = Some(val()?),
            "--kernel" => {
                a.mode = match val()?.as_str() {
                    "exact" => KernelMode::Exact,
                    "q8act" => KernelMode::Q8Act,
                    "ternary" => KernelMode::Ternary,
                    other => {
                        return Err(format!("--kernel is q8act|ternary|exact, got {other:?}").into())
                    }
                }
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if a.model.is_empty() || a.tokenizer.is_empty() || a.data.is_empty() {
        return Err("--model, --tokenizer and --data are required".into());
    }
    if a.ctx < 2 {
        return Err("--ctx must be at least 2".into());
    }
    if a.head_batch == 0 {
        return Err("--head-batch must be positive".into());
    }
    Ok(a)
}

/// The log-sum-exp of one logit vector in nats, and the logit at `target`.
fn log_softmax(logits: &[i64], target: usize) -> Result<f64, Box<dyn Error>> {
    let scale = 1.0 / 4_294_967_296.0; // Q32.32
    let mut max = f64::NEG_INFINITY;
    for &v in logits {
        let f = v as f64 * scale;
        if f > max {
            max = f;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += (v as f64 * scale - max).exp();
    }
    let target = *logits
        .get(target)
        .ok_or_else(|| format!("target {target} outside the vocabulary"))?;
    Ok(target as f64 * scale - max - sum.ln())
}

/// The process's peak resident set in KiB (`VmHWM`), or `None` off Linux.
fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

fn run(a: &Args) -> Result<(), Box<dyn Error>> {
    let text = std::fs::read_to_string(&a.data)?;
    let bpe = Bpe::from_tokenizer_json(&a.tokenizer)?;
    let ids = bpe.encode(&text)?;
    if a.skip >= ids.len().saturating_sub(1) {
        return Err(format!("--skip {} leaves fewer than two tokens", a.skip).into());
    }
    let ids = &ids[a.skip..];
    let ids = &ids[..ids.len().min(a.tokens)];
    if let Some(path) = &a.ids_out {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        std::fs::write(path, bytes)?;
    }

    let engine = match &a.pack {
        Some(path) => {
            let pack = nq::open(path).map_err(|e| e.to_string())?;
            println!(
                "pack: {} linears, {:.6} b/w, {} factor bytes",
                pack.linears.len(),
                pack.bits_per_weight().unwrap_or(0.0),
                pack.stored_bits().div_ceil(8)
            );
            Engine::open_with_nq(&a.model, pack)?
        }
        None => Engine::open(&a.model)?,
    };
    if let Some(&bad) = ids
        .iter()
        .find(|&&id| id as usize >= engine.config.vocab_size)
    {
        return Err(format!("id {bad} >= vocabulary {}", engine.config.vocab_size).into());
    }
    let (stored, weights) = engine.nq_storage();
    if stored > 0 {
        println!(
            "pack factors: {stored} stored bytes for {weights} weights, {:.6} b/w",
            stored as f64 * 8.0 / weights as f64
        );
    }

    let t0 = Instant::now();
    let (mut nll, mut count, mut windows) = (0.0f64, 0usize, 0usize);
    let mut start = 0usize;
    while start + 1 < ids.len() {
        let end = (start + a.ctx).min(ids.len());
        let window = &ids[start..end];
        let mut st = engine.state(window.len())?;
        let states = engine.prefill_states(&mut st, window, a.mode)?;
        let predict = states.len().saturating_sub(1);
        // llama.cpp's perplexity scores positions ctx/2..ctx-2 of each chunk, using the first
        // half only as context; `--second-half` matches that span for a cross-engine bar.
        let mut at = if a.second_half {
            (window.len() / 2).min(predict)
        } else {
            0
        };
        while at < predict {
            let take = (predict - at).min(a.head_batch);
            let cols = &states[at..at + take];
            let logits = engine
                .lm_head_cols(cols, a.mode)
                .map_err(|e| format!("lm_head: {e:?}"))?;
            for (c, row) in logits.iter().enumerate() {
                let target = window[at + c + 1] as usize;
                let lp = log_softmax(row, target)?;
                nll -= lp;
                if a.trace && -lp > 20.0 {
                    println!("  position {}: target {target} nll {:.3}", at + c, -lp);
                }
                if a.dump {
                    println!("nll {} {:.8}", start + at + c, -lp);
                }
            }
            count += take;
            at += take;
        }
        windows += 1;
        start = end;
    }
    if count == 0 {
        return Err("no position had a next token to score".into());
    }
    let secs = t0.elapsed().as_secs_f64();
    let mean = nll / count as f64;
    println!(
        "ppl {:.6} on {count} tokens ({windows} windows of <= {}, kernel {:?}): mean nll {mean:.6}, \
         {:.1} tok/s in {secs:.1} s",
        mean.exp(),
        a.ctx,
        a.mode,
        count as f64 / secs,
    );
    if let Some(kib) = peak_rss_kib() {
        println!("peak rss {:.1} MiB", kib as f64 / 1024.0);
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args().and_then(|a| run(&a)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ppl: {e}");
            ExitCode::FAILURE
        }
    }
}
