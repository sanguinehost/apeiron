//! `lm-eval` bridge: score continuation tokens through the real GGUF engine on stdin/stdout.
//!
//! Run: `cargo run --release --locked --example lm_eval_bridge [gguf]`
//!
//! One request per stdin line, `{"context_ids":[...],"continuation_ids":[...]}`. The bridge loads
//! the `Q4_K_M` GGUF once and, for each request, rebuilds a fresh [`KvState`], prefills `context_ids`,
//! then scores every `continuation_ids` entry by the `f64` log-softmax of its Q32.32 logits
//! (`logits / 2^32`). It writes one `{"logprob": <f64>, "is_greedy": <bool>}` line per request: the
//! summed continuation log-probability and whether every scored token equals the step's argmax. The
//! KV state is rebuilt per request, so no context crosses requests (correctness over speed).
//!
//! The tokenizer lives in the caller; ids cross this boundary already encoded.

use std::error::Error;
use std::io::{BufRead, Write};
use std::process::ExitCode;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;

/// The model named by the task.
const DEFAULT_MODEL: &str = concat!(env!("HOME"), "/models/gguf/qwen05b-Q4_K_M.gguf");
/// The Q32.32 fractional base `2^-32`, the scale between a stored logit and its real value.
const Q32: f64 = 4_294_967_296.0;

/// The first `[ ... ]` integer array following `"key"` in `line`.
///
/// The protocol is self-produced (`json.dumps`), so this reads the two known keys and their
/// bracketed `u32` lists; anything else is a hard protocol error.
fn extract_ids(line: &str, key: &str) -> Result<Vec<u32>, String> {
    let needle = format!("\"{key}\"");
    let kpos = line
        .find(&needle)
        .ok_or_else(|| format!("missing key {key}"))?;
    let rest = &line[kpos + needle.len()..];
    let lb = rest
        .find('[')
        .ok_or_else(|| format!("missing '[' after {key}"))?;
    let rb = rest[lb..]
        .find(']')
        .ok_or_else(|| format!("missing ']' after {key}"))?;
    let mut out: Vec<u32> = Vec::new();
    for tok in rest[lb + 1..lb + rb].split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<u32>().map_err(|e| format!("bad id {t:?}: {e}"))?);
    }
    Ok(out)
}

/// Log-softmax of one Q32.32 step for a single target token: `(logprob, target is the argmax)`.
///
/// The shifted maximum keeps `exp` in `f64` range; the argmax mirrors [`apeiron::engine::argmax_fixed`]
/// (strict `>`: ties keep the lowest index). The Q32.32 logit is divided by `2^32` only here, off the
/// engine's integer path.
#[allow(
    clippy::cast_precision_loss,
    reason = "the Q32.32 logit is narrowed to f64 only for this display-grade log-softmax; the \
              engine's executing path stays integer"
)]
fn step_logprob(logits: &[i64], target: u32) -> Result<(f64, bool), String> {
    if logits.is_empty() {
        return Err("empty logits".to_string());
    }
    let tvi = usize::try_from(target).map_err(|_| format!("target {target} out of range"))?;
    if tvi >= logits.len() {
        return Err(format!("target {target} >= vocabulary {}", logits.len()));
    }
    let mut max = logits[0] as f64 / Q32;
    let mut i = 1usize;
    while i < logits.len() {
        let v = logits[i] as f64 / Q32;
        if v > max {
            max = v;
        }
        i += 1;
    }
    let mut sum = 0.0f64;
    i = 0;
    while i < logits.len() {
        sum += ((logits[i] as f64 / Q32) - max).exp();
        i += 1;
    }
    let lse = max + sum.ln();
    let tv = logits[tvi] as f64 / Q32;

    let mut am = 0usize;
    let mut best = logits[0];
    i = 1;
    while i < logits.len() {
        if logits[i] > best {
            best = logits[i];
            am = i;
        }
        i += 1;
    }
    Ok((tv - lse, am == tvi))
}

/// Prefill `context` into a fresh [`KvState`] and sum the continuation log-probabilities.
///
/// Returns `(sum of per-token log-probs, every continuation token is greedy)`. Each request gets its
/// own cache, so no position or attention window crosses requests.
fn score(
    engine: &Engine,
    mode: KernelMode,
    context: &[u32],
    cont: &[u32],
) -> Result<(f64, bool), Box<dyn Error>> {
    let cap = context.len() + cont.len();
    let mut kv = engine.decode_state(cap)?;
    let mut cur: Option<Vec<i64>> = None;
    for &id in context {
        cur = Some(engine.step_logits_with_mode(&mut kv, id, mode)?);
    }
    let mut logits = cur.ok_or("empty context has no scoring position")?;
    let mut total = 0.0f64;
    let mut greedy = true;
    let mut i = 0usize;
    while i < cont.len() {
        let (lp, g) = step_logprob(&logits, cont[i])?;
        total += lp;
        greedy &= g;
        if i + 1 < cont.len() {
            logits = engine.step_logits_with_mode(&mut kv, cont[i], mode)?;
        }
        i += 1;
    }
    Ok((total, greedy))
}

fn run(model: &str) -> Result<(), Box<dyn Error>> {
    // `APEIRON_KERNEL=q8act` selects the int8-activation projections; anything else is `Exact`.
    let mode = match std::env::var("APEIRON_KERNEL").as_deref() {
        Ok("q8act") => KernelMode::Q8Act,
        _ => KernelMode::Exact,
    };
    let engine = Engine::open(model)?;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let context = extract_ids(&line, "context_ids")?;
        let cont = extract_ids(&line, "continuation_ids")?;
        let (logprob, is_greedy) = score(&engine, mode, &context, &cont)?;
        writeln!(
            out,
            "{{\"logprob\": {logprob}, \"is_greedy\": {is_greedy}}}"
        )?;
        out.flush()?;
    }
    Ok(())
}

fn main() -> ExitCode {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    match run(&model) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lm_eval_bridge: {e}");
            ExitCode::FAILURE
        }
    }
}
