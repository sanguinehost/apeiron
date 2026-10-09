//! Greedy-decode a fixed raw-completion prompt through the real GGUF engine.
//!
//! Run: `cargo run --release --locked --example engine_decode -- [gguf] [tokenizer.json] [n]`
//!
//! The example loads `qwen05b-Q4_K_M.gguf`, tokenizes the prompt with the ported byte-level
//! [`Bpe`], prefills the prompt through [`Engine::step_logits_state`] over an [`Engine::state`]
//! capacity-sized KV cache, then greedy-decodes `n` tokens (`argmax_fixed` over Q32.32 logits). It
//! prints the exact prompt, the prompt ids, the generated ids, the detokenized completion and the
//! wall-clock tokens/second. This is the end-to-end executing face: rows 0–8 composed on the real
//! tensor table, no float on the token path.

use std::error::Error;
use std::process::ExitCode;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::bpe::Bpe;

/// The model named by the task.
const DEFAULT_MODEL: &str = concat!(env!("HOME"), "/models/gguf/qwen05b-Q4_K_M.gguf");
/// Qwen2.5-0.5B's `tokenizer.json`, the reference the `Bpe` port is matched against.
const DEFAULT_TOKENIZER: &str = concat!(env!("HOME"), "/models/qwen2.5-0.5b/tokenizer.json");
/// The fixed raw-completion prompt.
const PROMPT: &str = "The capital of France is";
/// Greedy tokens to generate.
const N_GEN: usize = 16;

fn run(model: &str, tokenizer: &str, n: usize) -> Result<(), Box<dyn Error>> {
    // `APEIRON_KERNEL=q8act` selects the int8-activation projections; anything else is `Exact`.
    let mode = match std::env::var("APEIRON_KERNEL").as_deref() {
        Ok("q8act") => KernelMode::Q8Act,
        _ => KernelMode::Exact,
    };
    let bpe = Bpe::from_tokenizer_json(tokenizer)?;
    let engine = Engine::open(model)?;

    // `APEIRON_PROMPT_IDS=<file>`: whitespace-separated prompt ids instead of the built-in prompt.
    let prompt_ids: Vec<u32> = match std::env::var("APEIRON_PROMPT_IDS") {
        Ok(path) => std::fs::read_to_string(path)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?,
        Err(_) => bpe.encode(PROMPT)?,
    };
    let vocab = engine.config.vocab_size;
    for &id in &prompt_ids {
        if id as usize >= vocab {
            return Err(format!("prompt id {id} >= model vocabulary {vocab}").into());
        }
    }

    println!("model         {model}");
    println!("kernel        {mode:?}");
    println!("prompt        {PROMPT:?}");
    println!("prompt_ids    {prompt_ids:?}");

    // `APEIRON_SPEC=<draft rows>`: self-speculative decoding with the model's MTP block (exactly
    // the greedy ids); timed end to end, prefill included, like `APEIRON_E2E=1` for the baseline.
    if let Ok(rows) = std::env::var("APEIRON_SPEC") {
        let rows: usize = rows
            .parse()
            .map_err(|_| "APEIRON_SPEC must be a row count")?;
        let start = Instant::now();
        let (generated, stats) = engine.generate_speculative(&prompt_ids, n, rows)?;
        let secs = start.elapsed().as_secs_f64();
        println!("generated_ids {generated:?}");
        println!("spec          {stats:?}, draft rows {rows}");
        #[allow(clippy::cast_precision_loss, reason = "a display ratio")]
        let tps = n as f64 / secs;
        println!("e2e tok/s     {tps:.4}");
        return Ok(());
    }
    let e2e_start = Instant::now();

    let cap = prompt_ids.len() + n;
    let mut st = engine.state(cap)?;

    // Prefill: the whole prompt through the composed step; the logits at the last prompt position
    // choose the first generated token.
    let prefill_start = Instant::now();
    let mut last = None;
    if !prompt_ids.is_empty() {
        last = Some(engine.prefill_state(&mut st, &prompt_ids, mode)?);
    }
    let prefill = prefill_start.elapsed();
    let mut next = last.ok_or("the prompt is empty")?;

    // Greedy decode: append the sampled token, feed it back at the next position.
    let gen_start = Instant::now();
    let mut generated: Vec<u32> = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        generated.push(next);
        next = engine.next_token_state(&mut st, next, mode)?;
        i += 1;
    }
    let gen = gen_start.elapsed();
    let secs = gen.as_secs_f64();

    println!("generated_ids {generated:?}");
    match bpe.decode(&generated) {
        Ok(text) => println!("generated     {text:?}"),
        Err(e) => println!("generated     <decode refused: {e}>"),
    }
    println!(
        "prefill       {:.3} s for {} tokens",
        prefill.as_secs_f64(),
        prompt_ids.len()
    );
    println!("decode        {secs:.3} s for {n} tokens");
    #[allow(
        clippy::cast_precision_loss,
        reason = "n is 16; the ratio is a display measurement, not a token count"
    )]
    let tps = n as f64 / secs;
    println!("tok/s         {tps:.4}");
    #[allow(clippy::cast_precision_loss, reason = "a display ratio")]
    let e2e = n as f64 / e2e_start.elapsed().as_secs_f64();
    println!("e2e tok/s     {e2e:.4}");
    Ok(())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let tokenizer = args.next().unwrap_or_else(|| DEFAULT_TOKENIZER.to_string());
    let n = match args.next() {
        Some(s) => match s.parse::<usize>() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("engine_decode: bad token count {s:?}: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => N_GEN,
    };
    match run(&model, &tokenizer, n) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("engine_decode: {e}");
            ExitCode::FAILURE
        }
    }
}
