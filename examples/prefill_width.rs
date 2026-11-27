//! Prefill width scaling: the trunk pass wall per token at `APEIRON_PREFILL_COLS = w`.
//!
//! `prefill_width --model <gguf> [--pack <nqp>] [--ids <file>] [--tokens 64] [--reps 3]`
//!
//! The engine splits a prefill of `n` tokens into `⌈n / w⌉` column blocks of `w` (or `w + 1` for
//! the first `n mod ⌈n/w⌉` blocks). Running the same `n` under several `w` values measures the
//! pooled small-`k` path directly: `w = 1` is one column per pass, `w = 2` verifies two, and the
//! wall per token is the batch lever's own ratio. The env read is a `OnceLock`, so one process
//! measures one width; the wrapper runs one process per width.
//!
//! The head is not in this measurement: [`Engine::prefill_states`] returns every position's
//! output-normed state and does not project the vocabulary.
use std::process::ExitCode;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::nq;

struct Args {
    model: String,
    pack: Option<String>,
    ids: Option<String>,
    tokens: usize,
    reps: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        pack: None,
        ids: None,
        tokens: 64,
        reps: 3,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--pack" => a.pack = Some(val()?),
            "--ids" => a.ids = Some(val()?),
            "--tokens" => a.tokens = val()?.parse().map_err(|e| format!("--tokens: {e}"))?,
            "--reps" => a.reps = val()?.parse().map_err(|e| format!("--reps: {e}"))?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.model.is_empty() {
        return Err("--model is required".into());
    }
    if a.tokens == 0 || a.reps == 0 {
        return Err("--tokens and --reps must be positive".into());
    }
    Ok(a)
}

fn open(a: &Args) -> Result<Engine, String> {
    match &a.pack {
        Some(p) => {
            let pack = nq::open(p).map_err(|e| format!("{p}: {e:?}"))?;
            Engine::open_with_nq(&a.model, pack).map_err(|e| format!("{}: {e:?}", a.model))
        }
        None => Engine::open(&a.model).map_err(|e| format!("{}: {e:?}", a.model)),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("prefill_width: {e}");
            ExitCode::FAILURE
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "display only: a measured nanosecond count and token count to a printed rate"
)]
fn run() -> Result<(), String> {
    let a = parse_args()?;
    let e = open(&a)?;
    let vocab = e.config.vocab_size;
    let ids: Vec<u32> = match &a.ids {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| format!("{path}: {e}"))?
            .split_whitespace()
            .map(|t| t.parse::<u32>().map_err(|e| format!("{path}: {e}")))
            .collect::<Result<_, _>>()?,
        None => (0..a.tokens)
            .map(|i| u32::try_from((i * 37 + 100) % vocab).unwrap_or(0))
            .collect(),
    };
    let tokens = &ids[..a.tokens.min(ids.len())];
    let width: usize = std::env::var("APEIRON_PREFILL_COLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut warm = e.state(8).map_err(|e| format!("{e:?}"))?;
    let _ = e
        .prefill_states(&mut warm, &tokens[..tokens.len().min(8)], KernelMode::Q8Act)
        .map_err(|e| format!("{e:?}"))?;
    let mut best = f64::INFINITY;
    let mut walls = Vec::with_capacity(a.reps);
    for _ in 0..a.reps {
        let mut st = e.state(tokens.len() + 8).map_err(|e| format!("{e:?}"))?;
        let t = Instant::now();
        let states = e
            .prefill_states(&mut st, tokens, KernelMode::Q8Act)
            .map_err(|e| format!("{e:?}"))?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(states.len());
        walls.push(ms);
        best = best.min(ms);
    }
    let mean = walls.iter().sum::<f64>() / walls.len() as f64;
    println!(
        "width {width:4} tokens {:4} mean {mean:9.3} ms  best {best:9.3} ms  \
         {:.3} ms/token  {:.1} tok/s",
        tokens.len(),
        mean / tokens.len() as f64,
        tokens.len() as f64 / (mean / 1e3),
    );
    Ok(())
}
