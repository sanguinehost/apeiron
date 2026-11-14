//! `tests/learn_calib.rs` — the calibration Grams `NanoQuant` fits against (`tests/learn_nanoquant.rs`
//! reads them as `{cache}/g{l}.{tag}.{gate_up,down,o}.i64`).
//!
//! For each layer the file writes three activation Grams `Σ_t x_t x_tᵀ / T` in Q32.32 words with an
//! `(o, n)` header: `gate_up` is the `ffn_gate`/`ffn_up` input (tap stage 3), `down` the `ffn_down`
//! input (stage 4, the engine's own `SwiGLU` product) and `o` the attention-output input. The tokens
//! are the chunk block of a `_logits_` base file (`learn_nanoquant::nq_calib_base` writes one from
//! text); the Grams come from the engine's own tapped `Q8Act` prefill, `Grams::accumulate_tap`, then
//! `Normalisation::Mean`.
//!
//! Chunks: `0..APEIRON_CALIB_CHUNKS` (default 8) plus `APEIRON_CALIB_EXTRA` (default 0) chunks from
//! `CALIB + 3 = 11` on, so chunks 8, 9 and 10 stay held out. The tag is `c{calib}x{extra}`;
//! `NanoQuant`'s default `APEIRON_NQ_GRAM_TAG` is `c6x29` (6 + 29 chunks, a 40-chunk base file).
//!
//! Layout control: per layer, the diagonal of `gate_up` and `down` against `Σ_t x_t[k]² / T` summed
//! straight from the tap columns, relative error below `1e-4`.
//!
//! Grade: `measured` for the written Grams (the print of the run); the layout control is `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the Q32.32 ↔ f64 boundary is range-checked by the grid it came from; token counts \
              fit f64 exactly"
)]

use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::learn::fold::stats::capture::{Grams, LayerGrams, Normalisation};
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::model::Config;
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// Calibration chunks `0..CALIB` by default: 8 chunks of 512 tokens give 4096 rows, above the
/// `n_ff` width of the down Gram, so the Gram is not rank-limited by the row count.
const CALIB: usize = 8;

/// Tap slot of the FFN-side `RMSNorm` output: the gate/up input (`STAGE_LN2_OUT`).
const TAP_LN2: usize = 3;
/// Tap slot of the `SwiGLU` product, the `ffn_down` input (`STAGE_M`).
const TAP_M: usize = 4;

/// `~/` expanded against `HOME`; any other path is returned as given.
fn expand(p: &str) -> String {
    match (p.strip_prefix("~/"), std::env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => p.to_string(),
    }
}

/// The environment value of `key`, or `default`, with `~/` expanded.
fn env_path(key: &str, default: &str) -> String {
    expand(&std::env::var(key).unwrap_or_else(|_| default.to_string()))
}

/// An environment knob with a default.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// `words` as reals, one conversion at the boundary.
fn unq(words: &[i64]) -> Vec<f64> {
    words.iter().map(|&w| from_grid(w)).collect()
}

/// `x` as Q32.32 words, the one rounding into the grid.
fn qv(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("grid"))
        .collect()
}

/// Write `words` with an `(o, n)` header through a temporary file and a rename, so a killed run
/// leaves no short payload. `O(words)`.
fn write_words(path: &str, o: usize, n: usize, words: &[i64]) -> std::io::Result<()> {
    let tmp = format!("{path}.tmp");
    let mut b = [(o as u64).to_le_bytes(), (n as u64).to_le_bytes()].concat();
    b.extend(words.iter().flat_map(|w| w.to_le_bytes()));
    std::fs::write(&tmp, b)?;
    std::fs::rename(&tmp, path)
}

/// The chunk token block of a base file, `n_chunk` rows of `n_ctx` tokens.
fn base_tokens(path: &str) -> (Vec<Vec<u32>>, usize) {
    let mut base = KldBase::open(Path::new(path)).expect("open base file");
    let n = base.n_chunk();
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut n_ctx = 0usize;
    for _ in 0..n {
        let chunk = base.next_chunk().expect("chunk");
        n_ctx = chunk.tokens.len();
        out.push(chunk.tokens);
    }
    (out, n_ctx)
}

/// One tapped `Q8Act` prefill of `tokens`, returning the `[layer][stage][column][width]` rows.
fn tapped(engine: &Engine, tokens: &[u32]) -> Vec<Vec<Vec<Vec<i64>>>> {
    let mut st = engine.state(tokens.len()).expect("decode state");
    let (_out, rows) = engine
        .prefill_tap_cols(&mut st, tokens, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    rows
}

/// Calibration chunks `0..calib` plus `extra` chunks from `CALIB + 3` on. `O(chunks)` copies.
fn calib_set(chunks: &[Vec<u32>], calib: usize, extra: usize) -> Vec<Vec<u32>> {
    let from = CALIB + 3;
    assert!(
        extra == 0 || calib <= CALIB,
        "extra chunks need calib <= {CALIB}"
    );
    assert!(
        calib <= chunks.len(),
        "calib {calib}: {} chunks",
        chunks.len()
    );
    assert!(
        extra == 0 || from + extra <= chunks.len(),
        "extra {extra}: {} chunks",
        chunks.len()
    );
    let tail = if extra == 0 {
        &chunks[0..0]
    } else {
        &chunks[from..from + extra]
    };
    chunks[..calib].iter().chain(tail).cloned().collect()
}

/// The calibration Grams over `chunks`, with the diagonal layout control for every layer in
/// `layers`. Cost: `O(chunks·n_layer·t·(n_embd² + q_out² + n_ff²)/2)` multiply-adds in the
/// accumulator plus `O(chunks·|layers|·t·(n_embd + n_ff))` for the control; space the
/// accumulator's `O(n_layer·(n_embd² + q_out² + n_ff²))` words.
fn calib_grams(
    engine: &Engine,
    cfg: &Config,
    chunks: &[Vec<u32>],
    layers: &[usize],
) -> Vec<LayerGrams> {
    let (n, f) = (cfg.n_embd, cfg.n_ff);
    let mut grams = Grams::new(cfg).expect("grams");
    let mut diag: Vec<(Vec<f64>, Vec<f64>)> = vec![(vec![0.0; n], vec![0.0; f]); cfg.n_layer];
    let mut seen = 0usize;
    let t0 = Instant::now();
    for (c, tokens) in chunks.iter().enumerate() {
        let rows = tapped(engine, tokens);
        grams.accumulate_tap(&rows).expect("accumulate");
        for &l in layers {
            for col in &rows[l][TAP_LN2] {
                for (k, v) in unq(col).iter().enumerate() {
                    diag[l].0[k] += v * v;
                }
            }
            for col in &rows[l][TAP_M] {
                for (k, v) in unq(col).iter().enumerate() {
                    diag[l].1[k] += v * v;
                }
            }
        }
        seen += tokens.len();
        println!("CALIB chunk {c} at {:.1} s", t0.elapsed().as_secs_f64());
    }
    let done = grams.finish(Normalisation::Mean).expect("finish");
    for &l in layers {
        for (name, g, d, w) in [
            ("gate_up", &done[l].gate_up, &diag[l].0, n),
            ("down", &done[l].down, &diag[l].1, f),
        ] {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for k in 0..w {
                let e = d[k] / seen as f64;
                num += (g[k * w + k] - e).powi(2);
                den += e * e;
            }
            let rel = (num / den).sqrt();
            println!("CALIB layer {l} {name} diag rel {rel:.3e}");
            assert!(
                rel < 1e-4,
                "layer {l} {name}: Gram diagonal off the tap columns by {rel}"
            );
        }
    }
    done
}

/// Write `{cache}/g{l}.c{calib}x{extra}.{o,gate_up,down}.i64` for every layer of the teacher (or
/// `APEIRON_CALIB_LAYERS`, a comma list). An existing complete set is left as it is.
///
/// `APEIRON_CALIB_MODEL` (default `~/nq-ref/q3-0p6b-f16.gguf`), `APEIRON_CALIB_BASE` (default
/// `~/kld08/calib.c40.kld`), `APEIRON_CALIB_CACHE` (default `~/kld08/gatecode-tok`, `NanoQuant`'s
/// `APEIRON_NQ_CACHE` default), `APEIRON_CALIB_CHUNKS`, `APEIRON_CALIB_EXTRA`.
#[test]
#[ignore = "opens the teacher and prefills every calibration chunk; run explicitly"]
fn calib_grams_write() {
    let model = env_path("APEIRON_CALIB_MODEL", "~/nq-ref/q3-0p6b-f16.gguf");
    let base = env_path("APEIRON_CALIB_BASE", "~/kld08/calib.c40.kld");
    let cache = env_path("APEIRON_CALIB_CACHE", "~/kld08/gatecode-tok");
    let calib = env_usize("APEIRON_CALIB_CHUNKS", CALIB);
    let extra = env_usize("APEIRON_CALIB_EXTRA", 0);
    let tag = format!("c{calib}x{extra}");
    let engine = Engine::open(&model).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let layers: Vec<usize> = match std::env::var("APEIRON_CALIB_LAYERS") {
        Ok(s) if !s.is_empty() => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        _ => (0..cfg.n_layer).collect(),
    };
    std::fs::create_dir_all(&cache).expect("cache dir");
    let path = |l: usize, k: &str| format!("{cache}/g{l}.{tag}.{k}.i64");
    let kinds = ["o", "gate_up", "down"];
    if layers
        .iter()
        .all(|&l| kinds.iter().all(|k| Path::new(&path(l, k)).exists()))
    {
        println!("CALIB {tag}: every Gram exists in {cache}, nothing written");
        return;
    }
    let (chunks, n_ctx) = base_tokens(&base);
    let set = calib_set(&chunks, calib, extra);
    let grams = calib_grams(&engine, &cfg, &set, &layers);
    for &l in &layers {
        for (k, g) in kinds
            .iter()
            .zip([&grams[l].o, &grams[l].gate_up, &grams[l].down])
        {
            let w = g.len().isqrt();
            write_words(&path(l, k), w, w, &qv(g)).expect("write Gram");
        }
    }
    println!(
        "CALIB {tag}: {} layers, {} tokens, written to {cache}",
        layers.len(),
        set.len() * n_ctx
    );
}

#[test]
fn calib_set_keeps_chunks_8_to_10_out() {
    let chunks: Vec<Vec<u32>> = (0..40u32).map(|c| vec![c]).collect();
    let set = calib_set(&chunks, 6, 29);
    let ids: Vec<u32> = set.iter().map(|c| c[0]).collect();
    assert_eq!(ids.len(), 35);
    assert!(!ids.iter().any(|&c| (8..=10).contains(&c)), "{ids:?}");
    assert_eq!(&ids[..6], &[0, 1, 2, 3, 4, 5]);
    assert_eq!(ids[6], 11);
    assert_eq!(calib_set(&chunks, 8, 0).len(), 8);
}

#[test]
fn expand_reads_home() {
    std::env::set_var("HOME", "/h");
    assert_eq!(expand("~/a/b"), "/h/a/b");
    assert_eq!(expand("/abs"), "/abs");
}
