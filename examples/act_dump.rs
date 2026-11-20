//! Per-stage activation capture for the ternary-activation error screen.
//!
//! `act_dump --model <gguf> [--pack <nqp>] --tokenizer <tokenizer.json> --data <raw text>`
//! ` [--tokens 512] [--chunk 32] [--out /tmp/opencode/act-dump]`
//!
//! Tokenizes `--data`, keeps its first `--tokens` tokens, and runs
//! [`Engine::prefill_tap_cols`] with the tap on and [`KernelMode::Q8Act`] over `--chunk`-token
//! blocks of one continuing state. For layers 0, 13 and 27 it writes the `attn_norm` stage output
//! (the Q/K/V projection input, stage 1) and the `ffn_norm` stage output (the gate/up projection
//! input, stage 3) as little-endian `i64`, row-major `[column][width]`, after one ASCII header line
//! naming the stage, layer, width and column count:
//!
//! ```text
//! apeiron-act-dump v1 stage=attn_norm layer=0 width=1024 columns=512 dtype=i64le\n
//! <512·1024 little-endian i64, column 0's 1024 words first>
//! ```
//!
//! The chunk is a memory bound, not a different pass: `prefill_tap_cols` records every stage of
//! every layer for the block it carries, 28 layers · 19,456 words · 8 B per column for Qwen3-0.6B,
//! so one 512-token call would hold ~2.2 GB of rows. Consecutive chunked calls over one
//! `DecodeState` see the same causal prefix as one call, and the engine's own prefill block is
//! 256 tokens anyway (`apeiron::engine::step::prefill_block`, crate-internal).
//!
//! The rows are the stage inputs `x`; a fitted `NanoQuant` linear reads the pre-scaled
//! `t = round(s_pre·x / 2^32)` instead. Without the pack's per-linear `s_pre` vector this capture
//! is `x`, which is the caveat the note carries.
//!
//! Cost: one trunk pass over `--tokens` tokens plus `O(tokens·width)` words written per stage;
//! space `O(chunk·Σ stage widths)` tap rows besides the model, KV cache and state.

#![allow(
    clippy::cast_precision_loss,
    reason = "elapsed seconds and peak RSS are a report over counters far below 2^53"
)]

use std::error::Error;
use std::process::ExitCode;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::bpe::Bpe;
use apeiron::format::nq;

/// The layers the screen reads: first, middle, last of the 28.
const LAYERS: [usize; 3] = [0, 13, 27];
/// `(stage slot, file-name tag)`: the attention-side and FFN-side norm outputs.
const STAGES: [(usize, &str); 2] = [(1, "attn_norm"), (3, "ffn_norm")];

struct Args {
    model: String,
    pack: Option<String>,
    tokenizer: String,
    data: String,
    tokens: usize,
    chunk: usize,
    out: String,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        pack: None,
        tokenizer: String::new(),
        data: String::new(),
        tokens: 512,
        chunk: 32,
        out: String::from("/tmp/opencode/act-dump"),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--pack" => a.pack = Some(val()?),
            "--tokenizer" => a.tokenizer = val()?,
            "--data" => a.data = val()?,
            "--tokens" => a.tokens = val()?.parse().map_err(|e| format!("--tokens: {e}"))?,
            "--chunk" => a.chunk = val()?.parse().map_err(|e| format!("--chunk: {e}"))?,
            "--out" => a.out = val()?,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if a.model.is_empty() || a.tokenizer.is_empty() || a.data.is_empty() {
        return Err("--model, --tokenizer and --data are required".into());
    }
    if a.tokens < 2 {
        return Err("--tokens must be at least 2".into());
    }
    if a.chunk == 0 {
        return Err("--chunk must be positive".into());
    }
    Ok(a)
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

/// Write one stage's rows: the ASCII header line, then every row's words in column order.
///
/// Cost `O(columns·width)` bytes written; space the same bytes once.
fn write_stage(
    path: &str,
    tag: &str,
    layer: usize,
    width: usize,
    cols: &[Vec<i64>],
) -> Result<(), Box<dyn Error>> {
    let header = format!(
        "apeiron-act-dump v1 stage={tag} layer={layer} width={width} columns={} dtype=i64le\n",
        cols.len()
    );
    let mut bytes: Vec<u8> = Vec::with_capacity(header.len() + cols.len() * width * 8);
    bytes.extend_from_slice(header.as_bytes());
    for row in cols {
        for &word in row {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn run(a: &Args) -> Result<(), Box<dyn Error>> {
    let text = std::fs::read_to_string(&a.data)?;
    let bpe = Bpe::from_tokenizer_json(&a.tokenizer)?;
    let ids = bpe.encode(&text)?;
    if ids.len() < a.tokens {
        return Err(format!(
            "--data tokenizes to {} tokens, --tokens {}",
            ids.len(),
            a.tokens
        )
        .into());
    }
    let ids = &ids[..a.tokens];

    let engine = match &a.pack {
        Some(path) => {
            let pack = nq::open(path).map_err(|e| e.to_string())?;
            println!(
                "pack: {} linears, {:.6} b/w",
                pack.linears.len(),
                pack.bits_per_weight().unwrap_or(0.0)
            );
            Engine::open_with_nq(&a.model, pack)?
        }
        None => Engine::open(&a.model)?,
    };
    let (n_layer, width) = (engine.config.n_layer, engine.config.n_embd);
    for &layer in &LAYERS {
        if layer >= n_layer {
            return Err(format!("layer {layer} past the model's {n_layer}").into());
        }
    }
    println!(
        "model: {n_layer} layers, width {width}, n_ff {}, chunk {}, tokens {}",
        engine.config.n_ff,
        a.chunk,
        ids.len()
    );
    std::fs::create_dir_all(&a.out)?;

    // `[layer][stage][column]`, appended a chunk at a time; the tap itself is dropped per chunk.
    let mut acc: Vec<Vec<Vec<Vec<i64>>>> = vec![vec![Vec::new(); STAGES.len()]; LAYERS.len()];
    for cols in &mut acc {
        for stage in cols {
            stage.reserve(ids.len());
        }
    }
    let mut st = engine.state(ids.len())?;
    let t0 = Instant::now();
    let mut done = 0usize;
    for (bi, block) in ids.chunks(a.chunk).enumerate() {
        let (_, rows) = engine.prefill_tap_cols(&mut st, block, KernelMode::Q8Act, true)?;
        for (li, &layer) in LAYERS.iter().enumerate() {
            for (si, &(stage, tag)) in STAGES.iter().enumerate() {
                let got = &rows[layer][stage];
                if got.len() != block.len() {
                    return Err(format!(
                        "block {bi} layer {layer} {tag}: {} rows for {} columns",
                        got.len(),
                        block.len()
                    )
                    .into());
                }
                for row in got {
                    if row.len() != width {
                        return Err(format!(
                            "block {bi} layer {layer} {tag}: row width {}",
                            row.len()
                        )
                        .into());
                    }
                    acc[li][si].push(row.clone());
                }
            }
        }
        done += block.len();
        println!("block {bi}: {done}/{} columns", ids.len());
    }
    let secs = t0.elapsed().as_secs_f64();

    for (li, &layer) in LAYERS.iter().enumerate() {
        for (si, &(_, tag)) in STAGES.iter().enumerate() {
            let path = format!("{}/layer{layer}-{tag}.i64", a.out);
            write_stage(&path, tag, layer, width, &acc[li][si])?;
            println!("wrote {path}: {} columns of {width} i64", acc[li][si].len());
        }
    }
    println!(
        "capture: {:.1} s, {:.2} tok/s",
        secs,
        ids.len() as f64 / secs
    );
    if let Some(kib) = peak_rss_kib() {
        println!("peak rss {:.1} MiB", kib as f64 / 1024.0);
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(args) => match run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("act_dump: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("act_dump: {e}");
            ExitCode::FAILURE
        }
    }
}
