//! `tests/learn_calib.rs` — the calibration Grams `NanoQuant` fits against (`tests/learn_nanoquant.rs`
//! reads them as `{cache}/g{l}.{tag}.{gate_up,down,o}.i64`).
//!
//! For each layer the file writes three activation Grams `Σ_t x_t x_tᵀ / T` in Q32.32 words with an
//! `(o, n)` header: `gate_up` is the `ffn_gate`/`ffn_up` input (tap stage 3), `down` the `ffn_down`
//! input (stage 4, the engine's own `SwiGLU` product) and `o` the attention-output input. The tokens
//! come from a `_logits_` base file (`learn_nanoquant::nq_calib_base` writes one from raw text); the
//! Grams come from the engine's own tapped `Q8Act` prefill, `Grams::accumulate_tap`, then
//! `Normalisation::Mean`.
//!
//! Two calibration products share the base file:
//!
//! - `calib_grams_write` writes the full Grams for the 8B/full-Gram path (`APEIRON_CALIB_CHUNKS`).
//! - `calib_diag_write` and `calib_onorm_write` write the reference campaign's diagonal statistics
//!   over `APEIRON_NQ_SAMPLES` samples of [`SEQ`] = 2048 tokens (default 128), tag `s{n}`, with the
//!   reference's per-sample `99.9`th-percentile clip (`APEIRON_CALIB_STRATEGY`, default `online`).
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

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::gguf::Gguf;
use apeiron::learn::fold::stats::capture::{Grams, LayerGrams, Normalisation};
use apeiron::learn::fold::stats::hocapture::{
    backward_with_taps, engine_tape, logits_grad_ce, Kind,
};
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::model::{Config, Weights};
use apeiron::learn::nanoquant::rank_for_bits;
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// Chunks of the full-Gram cache (`calib_grams_write`), `0..CALIB` by default: 8 chunks of 512
/// tokens give 4096 rows, above the `n_ff` width of the down Gram, so the Gram is not rank-limited
/// by the row count. This is the legacy/8B full-Gram path, not the reference campaign's
/// diagonal statistic.
const CALIB: usize = 8;

/// The reference's calibration geometry (`NanoQuantConfig`, `modules/quant_config.py:15-16`):
/// `num_calib_samples = 128` samples of `seqlen = 2048` tokens, drawn once from wikitext2 train
/// and reused for the statistics, the block loop and the model KD (`hub.py:336-353`).
const CALIB_SAMPLES: usize = 128;
/// Tokens per calibration sample (`quant_config.py:16`).
const SEQ: usize = 2048;

/// The four diagonal kinds in [`calib_diag_write`]'s and [`calib_onorm_write`]'s cache order.
const DIAG_KINDS: [&str; 4] = ["o", "gate_up", "down", "qkv"];
/// The tap slot each diagonal kind accumulates.
const DIAG_TAPS: [usize; 4] = [TAP_ATTN, TAP_LN2, TAP_M, TAP_LN1];

/// The reference's `calib_strategy` (`quant_config.py:14`, default `online`): the threshold is the
/// `99.9`th percentile of the current sample's row norms, the running maximum only grows, and the
/// accumulated statistic is rescaled by `(tau/gmax)²` when it grows (`importance.py:104-140`).
/// `two_phase` profiles every sample, keeps the maximum, then accumulates once against it
/// (`importance.py:47-63,181-223`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Strategy {
    /// One interleaved profile/accumulate pass per sample with the running threshold.
    Online,
    /// A profiling pass over every sample, then one fixed-threshold accumulation pass.
    TwoPhase,
}

/// [`Strategy`] from `APEIRON_CALIB_STRATEGY` (`online` unless set).
fn strategy() -> Strategy {
    match std::env::var("APEIRON_CALIB_STRATEGY").as_deref() {
        Ok("two_phase") => Strategy::TwoPhase,
        Ok("online") | Err(std::env::VarError::NotPresent) => Strategy::Online,
        Ok(s) => panic!("APEIRON_CALIB_STRATEGY={s}: expected online or two_phase"),
        Err(e) => panic!("APEIRON_CALIB_STRATEGY: {e}"),
    }
}

/// Tap slot of the attention-side `RMSNorm` output, the q/k/v input (`STAGE_LN1_OUT`).
const TAP_LN1: usize = 1;
/// Tap slot of the attention output, the `o` projection input (`STAGE_ATTN_OUT`).
const TAP_ATTN: usize = 2;
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

/// The chunk token block of a base file, `n_chunk` rows of `n_ctx` tokens, read from the token
/// block alone ([`KldBase::tokens`]): the row block is teacher logits the calibration path does not
/// use, and for the reference's 128×2048 tape it is 40 GB.
fn base_tokens(path: &str) -> (Vec<Vec<u32>>, usize) {
    let base = KldBase::open(Path::new(path)).expect("open base file");
    let n = base.n_chunk();
    let all = base.tokens();
    let n_ctx = all.len() / n;
    let chunks = all.chunks_exact(n_ctx).map(<[u32]>::to_vec).collect();
    (chunks, n_ctx)
}

/// The reference's calibration tensor: the base's first `n` samples of [`SEQ`] tokens, in base
/// order. A sample crosses chunk boundaries, so a base chunked at 512 tokens serves the same
/// stream as one chunked at 2048. `O(n·SEQ)` tokens.
fn calib_samples(base: &str, n: usize) -> Vec<Vec<u32>> {
    let (chunks, n_ctx) = base_tokens(base);
    let mut flat = Vec::with_capacity(chunks.len() * n_ctx);
    for c in &chunks {
        flat.extend_from_slice(c);
    }
    assert!(
        flat.len() >= n * SEQ,
        "{} tokens for {n} samples of {SEQ}: base {base} is too short",
        flat.len()
    );
    flat[..n * SEQ].chunks(SEQ).map(<[u32]>::to_vec).collect()
}

/// One tapped `Q8Act` prefill of `tokens`, returning the `[layer][stage][column][width]` rows.
fn tapped(engine: &Engine, tokens: &[u32]) -> Vec<Vec<Vec<Vec<i64>>>> {
    let mut st = engine.state(tokens.len()).expect("decode state");
    let (_out, rows) = engine
        .prefill_tap_cols(&mut st, tokens, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    rows
}

/// Columns per tapped prefill in [`calib_diag_write`]. A recorded pass holds
/// `n_layer · STAGE_COUNT · t · n_embd` Q32.32 words: 1.41 GB at 28 layers, 512 columns and
/// 1024 wide, against this laptop's guard cap. The KV state carries the earlier columns, so the
/// split changes no column's attention context; the diagonal is a sum over columns and does not
/// depend on the split either.
const TAP_BLOCK: usize = 64;

/// Call `f` on the tap rows of each `≤ TAP_BLOCK`-column block of `tokens`, on one KV state, so
/// the recorded pass is never wider than `TAP_BLOCK` columns.
fn tap_blocks(engine: &Engine, tokens: &[u32], mut f: impl FnMut(&[Vec<Vec<Vec<i64>>>])) {
    let mut st = engine.state(tokens.len()).expect("decode state");
    let mut at = 0usize;
    while at < tokens.len() {
        let end = (at + TAP_BLOCK).min(tokens.len());
        let (_out, rows) = engine
            .prefill_tap_cols(&mut st, &tokens[at..end], KernelMode::Q8Act, true)
            .expect("tapped prefill");
        f(&rows);
        at = end;
    }
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

/// Add `Σ_t x_t²` for each column of `cols` (one column per token) into `d`, the mean-diagonal
/// accumulator. `O(cols·width)`; [`calib_grams`] checks this sum against the full Gram's diagonal
/// to `1e-4` relative.
fn add_diag(d: &mut [f64], cols: &[Vec<i64>]) {
    for col in cols {
        assert_eq!(col.len(), d.len(), "diag column width");
        for (k, &w) in col.iter().enumerate() {
            let v = from_grid(w);
            d[k] += v * v;
        }
    }
}

/// The reference's robust clip (`importance.py:25-34,47-63,104-140`): a row-norm threshold `tau`
/// at the `99.9`th percentile, then each row scaled by `min(1, tau/‖x‖)` before its outer product
/// enters the statistic.
///
/// The threshold is taken per **sample** of [`SEQ`] rows, not per tap block. `k =
/// max(1, ⌊0.001·rows⌋)` over 2048 rows is 2, so the single largest row is clipped; that matches
/// the reference's `int(n·(1−0.999))`. Its `online` mode grows a running maximum and rescales the
/// accumulator (`importance.py:120-133`); `two_phase` profiles every sample and fixes the maximum
/// (`:181-223`). Both are [`Strategy`] here.
#[derive(Clone, Copy, Default)]
struct Clip {
    /// The row-norm threshold, `0` before [`Clip::tau`] sets it.
    tau: f64,
}

impl Clip {
    /// The row L2 norms of `cols`, appended to `out` (one per column).
    fn norms(cols: &[Vec<i64>], out: &mut Vec<f64>) {
        for col in cols {
            out.push(
                col.iter()
                    .map(|&w| from_grid(w).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            );
        }
    }

    /// The `99.9`th percentile of `norms`: the `max(1, ⌊0.001·n⌋)`-th largest. `O(n log n)`.
    fn tau(norms: &mut [f64]) -> f64 {
        if norms.is_empty() {
            return 0.0;
        }
        // `max(1, floor(n·0.001))` without a float→integer cast.
        let k = (norms.len() / 1000).max(1);
        norms.sort_by(|a, b| b.partial_cmp(a).expect("finite norms"));
        norms[k - 1]
    }

    /// Add the clipped `x²` of one row into `d`. `O(width)`.
    fn add_row(self, d: &mut [f64], row: &[i64]) {
        assert_eq!(row.len(), d.len(), "diag row width");
        let norm = row
            .iter()
            .map(|&w| from_grid(w).powi(2))
            .sum::<f64>()
            .sqrt();
        let c = if norm > self.tau {
            (self.tau / norm).powi(2)
        } else {
            1.0
        };
        for (k, &w) in row.iter().enumerate() {
            let v = from_grid(w);
            d[k] += c * v * v;
        }
    }

    /// Add the clipped `Σ_t x_t²` of `cols` into `d`. `O(cols·width)`.
    fn add(self, d: &mut [f64], cols: &[Vec<i64>]) {
        for col in cols {
            self.add_row(d, col);
        }
    }
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
            add_diag(&mut diag[l].0, &rows[l][TAP_LN2]);
            add_diag(&mut diag[l].1, &rows[l][TAP_M]);
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
    let kinds = ["o", "gate_up", "down", "qkv"];
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
        for (k, g) in kinds.iter().zip([
            &grams[l].o,
            &grams[l].gate_up,
            &grams[l].down,
            &grams[l].qkv,
        ]) {
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

/// One layer's four mean-diagonal accumulators, in [`DIAG_KINDS`] order.
struct Diag {
    acc: [Vec<f64>; 4],
}

impl Diag {
    /// Four zeroed accumulators of the layer's widths. `O(q + n + f)` words.
    fn zeros(widths: [usize; 4]) -> Self {
        Diag {
            acc: widths.map(|w| vec![0.0; w]),
        }
    }
}

/// Write only the mean diagonal of each calibration Gram, `{cache}/g{l}.{tag}.{kind}.diag.i64`
/// (header `(1, w)`, Q32.32), instead of the full `w × w` Gram.
///
/// `nq_fit` uses only `diag(H)` as its preconditioner; the full `n_ff × n_ff` down Gram of a 7B
/// model is 2.87 GB per layer, 80 GB over 28 layers, and its `finish` copy doubles that. The
/// diagonal fits in megabytes and the same one-pass prefill accumulates it.
///
/// The statistic is the reference's: `n` samples of [`SEQ`] tokens (`APEIRON_NQ_SAMPLES`,
/// default [`CALIB_SAMPLES`]), each column clipped at the sample's `99.9`th-percentile row norm
/// under `APEIRON_CALIB_STRATEGY` (`online` by default, `two_phase` otherwise), then the mean over
/// all tokens and the `0.4` covariance shrinkage. The tag is `s{n}`.
///
/// `APEIRON_CALIB_CHECK_LAYER` also accumulates that layer's full Gram in the same pass and asserts
/// its diagonal against this accumulator to `1e-4` relative, the full-Gram control's bound.
///
/// Env: `APEIRON_CALIB_MODEL`, `APEIRON_CALIB_BASE`, `APEIRON_CALIB_CACHE`, `APEIRON_NQ_SAMPLES`,
/// `APEIRON_CALIB_STRATEGY`, `APEIRON_CALIB_LAYERS`, `APEIRON_CALIB_CHECK_LAYER`.
#[test]
#[ignore = "opens the teacher and prefills every calibration sample; run explicitly"]
#[allow(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    reason = "one calibration pass: the single-letter names are the config's n_embd, q_out and \
              n_ff, and the stages share the open engine and cache"
)]
fn calib_diag_write() {
    let model = env_path("APEIRON_CALIB_MODEL", "~/nq-ref/q3-0p6b-f16.gguf");
    let base = env_path("APEIRON_CALIB_BASE", "~/kld08/calib.s128.kld");
    let cache = env_path("APEIRON_CALIB_CACHE", "~/kld08/gatecode-tok");
    let samples = env_usize("APEIRON_NQ_SAMPLES", CALIB_SAMPLES);
    let shrink_to: f64 = std::env::var("APEIRON_CALIB_SHRINK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.4);
    let strat = strategy();
    let tag = format!("s{samples}");
    let engine = Engine::open(&model).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let layers: Vec<usize> = match std::env::var("APEIRON_CALIB_LAYERS") {
        Ok(s) if !s.is_empty() => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        _ => (0..cfg.n_layer).collect(),
    };
    std::fs::create_dir_all(&cache).expect("cache dir");
    let path = |l: usize, k: &str| format!("{cache}/g{l}.{tag}.{k}.diag.i64");
    if layers
        .iter()
        .all(|&l| DIAG_KINDS.iter().all(|k| Path::new(&path(l, k)).exists()))
    {
        println!("DIAG {tag}: every diagonal exists in {cache}, nothing written");
        return;
    }
    let set = calib_samples(&base, samples);
    let (n, q, f) = (cfg.n_embd, cfg.q_out().expect("q_out"), cfg.n_ff);
    let widths = [q, n, f, n];
    let mut diag: Vec<Diag> = (0..cfg.n_layer).map(|_| Diag::zeros(widths)).collect();
    let mut gmax: Vec<[f64; 4]> = vec![[0.0; 4]; cfg.n_layer];
    let mut tau_max: Vec<[f64; 4]> = vec![[0.0; 4]; cfg.n_layer];
    let mut check = std::env::var("APEIRON_CALIB_CHECK_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|l| {
            assert!(l < cfg.n_layer, "check layer {l}");
            let cfg1 = Config {
                n_layer: 1,
                ..cfg.clone()
            };
            (l, Grams::new(&cfg1).expect("check grams"))
        });
    // The full-Gram control is unclipped, so it compares against a raw copy of the check layer's
    // diagonal; the written files carry the clipped statistic.
    let mut raw: Option<Vec<Diag>> = check
        .as_ref()
        .map(|_| (0..cfg.n_layer).map(|_| Diag::zeros(widths)).collect());
    let t0 = Instant::now();
    // `two_phase` (`importance.py:181-223`): profile every sample, keep the maximum threshold.
    if strat == Strategy::TwoPhase {
        for tokens in &set {
            let mut norms: Vec<[Vec<f64>; 4]> = (0..cfg.n_layer)
                .map(|_| std::array::from_fn(|_| Vec::new()))
                .collect();
            tap_blocks(&engine, tokens, |rows| {
                for l in 0..cfg.n_layer {
                    for (k, tap) in DIAG_TAPS.iter().enumerate() {
                        Clip::norms(&rows[l][*tap], &mut norms[l][k]);
                    }
                }
            });
            for l in 0..cfg.n_layer {
                for k in 0..4 {
                    tau_max[l][k] = tau_max[l][k].max(Clip::tau(&mut norms[l][k]));
                }
            }
        }
        println!("DIAG thresholds at {:.1} s", t0.elapsed().as_secs_f64());
    }
    let mut seen_tokens = 0usize;
    for (c, tokens) in set.iter().enumerate() {
        // `online` (`importance.py:104-140`): this sample's thresholds grow the running maximum
        // and rescale what earlier samples accumulated, before this sample is clipped at it.
        if strat == Strategy::Online {
            let mut norms: Vec<[Vec<f64>; 4]> = (0..cfg.n_layer)
                .map(|_| std::array::from_fn(|_| Vec::new()))
                .collect();
            tap_blocks(&engine, tokens, |rows| {
                for l in 0..cfg.n_layer {
                    for (k, tap) in DIAG_TAPS.iter().enumerate() {
                        Clip::norms(&rows[l][*tap], &mut norms[l][k]);
                    }
                }
            });
            for l in 0..cfg.n_layer {
                for k in 0..4 {
                    let t = Clip::tau(&mut norms[l][k]);
                    let g = gmax[l][k];
                    if t > g {
                        let r = if g > 0.0 { (t / g).powi(2) } else { 1.0 };
                        for x in &mut diag[l].acc[k] {
                            *x *= r;
                        }
                        gmax[l][k] = t;
                    }
                }
            }
        }
        let thresh = if strat == Strategy::Online {
            &gmax
        } else {
            &tau_max
        };
        tap_blocks(&engine, tokens, |rows| {
            for l in 0..cfg.n_layer {
                for (k, tap) in DIAG_TAPS.iter().enumerate() {
                    Clip { tau: thresh[l][k] }.add(&mut diag[l].acc[k], &rows[l][*tap]);
                }
            }
            if let (Some((l, g)), Some(raw)) = (check.as_mut(), raw.as_mut()) {
                g.accumulate_tap(&rows[*l..=*l]).expect("check accumulate");
                for (k, tap) in DIAG_TAPS.iter().enumerate() {
                    add_diag(&mut raw[*l].acc[k], &rows[*l][*tap]);
                }
            }
        });
        seen_tokens += tokens.len();
        println!("DIAG sample {c} at {:.1} s", t0.elapsed().as_secs_f64());
    }
    if let Some((l, g)) = check.take() {
        let full = g.finish(Normalisation::Mean).expect("check finish");
        let lg = &full[0];
        let d = &raw.as_ref().expect("raw accumulator")[l];
        for (k, name) in DIAG_KINDS.iter().enumerate() {
            let (mat, acc, w) = match *name {
                "qkv" => (&lg.qkv, &d.acc[k], n),
                "o" => (&lg.o, &d.acc[k], q),
                "gate_up" => (&lg.gate_up, &d.acc[k], n),
                _ => (&lg.down, &d.acc[k], f),
            };
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for j in 0..w {
                let e = acc[j] / seen_tokens as f64;
                num += (mat[j * w + j] - e).powi(2);
                den += e * e;
            }
            let rel = (num / den).sqrt();
            println!("DIAG layer {l} {name} full-Gram rel {rel:.3e}");
            assert!(
                rel < 1e-4,
                "layer {l} {name}: diagonal off the full Gram by {rel}"
            );
        }
    }
    for &l in &layers {
        for (k, name) in DIAG_KINDS.iter().enumerate() {
            let mut v: Vec<f64> = diag[l].acc[k]
                .iter()
                .map(|&x| x / seen_tokens as f64)
                .collect();
            shrink(&mut v, shrink_to);
            let words: Vec<i64> = v
                .iter()
                .map(|&x| to_grid(x, GRID_BOUND).expect("diag grid"))
                .collect();
            write_words(&path(l, name), 1, v.len(), &words).expect("write diagonal");
        }
    }
    println!(
        "DIAG {tag}: {} layers, {} samples x {SEQ} tokens, shrink {shrink_to}, written to {cache}",
        layers.len(),
        seen_tokens / SEQ
    );
}

/// The seven fitted linears' `o_norm` file names, in [`Kind::ALL`] order.
const ONORM_NAME: [&str; 7] = ["q", "k", "v", "o", "gate", "up", "down"];

/// `v ← (1 − s)·v + s·mean(v)`, the reference's `get_shrunk_stats` (`importance.py:181-223`).
/// `O(n)`; `s = 0` is a copy and `s = 1` a constant.
fn shrink(v: &mut [f64], s: f64) {
    if v.is_empty() {
        return;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in v.iter_mut() {
        *x = (1.0 - s) * *x + s * mean;
    }
}

/// One engine-taped forward, cross-entropy gradient and whole-model backward at `tokens`, calling
/// `f(layer, kind, ∂L/∂(kind output))`.
///
/// The reference's loss is `linear_cross_entropy(..., shift=1)` with its default mean reduction
/// (`importance.py:167-178`): position `i` predicts `tokens[i + 1]`, the last position carries no
/// target, and the backward is divided by the `tokens.len() − 1` valid positions. The last row's
/// logit gradient is set to zero, as in the reference, where the last position's output does not
/// enter the loss.
///
/// The forward is the engine's `Q8Act` prefill ([`engine_tape`]); the backward is
/// [`backward_with_taps`], the same chain as [`apeiron::learn::grad::model::backward`] plus the
/// seven output adjoints the reference hooks on its `nn.Linear`s (`importance.py:71-96`).
fn onorm_pass(
    engine: &Engine,
    cfg: &Config,
    w: &Weights,
    tokens: &[u32],
    mut f: impl FnMut(usize, Kind, &[i64]),
) {
    let (logits, tape) = engine_tape(engine, cfg, tokens).expect("engine tape");
    assert!(
        tape.t >= 2,
        "onorm_pass: one token has no next-token target"
    );
    let mut targets: Vec<u32> = tokens[1..].to_vec();
    targets.push(0);
    let mut d = vec![0i64; tape.t * cfg.vocab];
    logits_grad_ce(&logits, &targets, cfg.vocab, &mut d).expect("ce grad");
    // The last row has no target: zero it and divide the sum gradient by the mean's valid count.
    let valid = i64::try_from(tape.t - 1).expect("valid positions fit i64");
    for q in &mut d[(tape.t - 1) * cfg.vocab..] {
        *q = 0;
    }
    for q in &mut d[..(tape.t - 1) * cfg.vocab] {
        *q /= valid;
    }
    let (_grad, adj) = backward_with_taps(w, cfg, &tape, &d).expect("backward");
    for l in 0..cfg.n_layer {
        for k in Kind::ALL {
            f(l, k, adj.get(l, k).expect("tap"));
        }
    }
}

/// Write the reference's output statistic `o_norm` per linear, `{cache}/g{l}.{tag}.{kind}.o_norm.i64`
/// (header `(1, width)`, Q32.32).
///
/// The value is the reference's (`importance.py:65-140`): the mean squared output gradient over the
/// calibration samples, multiplied by `GRAD_SCALE_FACTOR = 1e6`, robust-clipped per sample under
/// [`Strategy`] and shrunk by `0.4`. The mean is over each sample's [`SEQ`] rows, as the hook's
/// `.mean(dim=0)` over `T` rows, and the last row's gradient is zero; the final division is
/// `samples × SEQ`. The `1e6` is the reference's saved unit: its hooks scale the
/// gradient by `1e6` before the norm and divide the squared update by `1e6`, so the stored statistic
/// is `1e6 ×` the mean square (`importance.py:16,86-99`). The clip is homogeneous, so computing it
/// on the unscaled gradient and multiplying the mean by `1e6` is the same statistic.
///
/// Env: `APEIRON_CALIB_MODEL`, `APEIRON_CALIB_BASE`, `APEIRON_CALIB_CACHE`, `APEIRON_NQ_SAMPLES`,
/// `APEIRON_CALIB_STRATEGY`, `APEIRON_CALIB_SHRINK` (default `0.4`).
#[test]
#[ignore = "two whole-model backwards over every calibration sample; run explicitly"]
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "one calibration pass: the single-letter names are the config's n_embd and the \
              kind's width, and the two strategies share the accumulator"
)]
fn calib_onorm_write() {
    let model = env_path("APEIRON_CALIB_MODEL", "~/nq-ref/q3-0p6b-f16.gguf");
    let base = env_path("APEIRON_CALIB_BASE", "~/kld08/calib.s128.kld");
    let cache = env_path("APEIRON_CALIB_CACHE", "~/kld08/gatecode-tok");
    let samples = env_usize("APEIRON_NQ_SAMPLES", CALIB_SAMPLES);
    let shrink_to: f64 = std::env::var("APEIRON_CALIB_SHRINK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.4);
    let strat = strategy();
    let tag = format!("s{samples}");
    let engine = Engine::open(&model).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (w, cfg_w) = Weights::from_engine(&engine).expect("weights");
    assert_eq!(cfg, cfg_w, "engine and weight configs");
    let set = calib_samples(&base, samples);
    std::fs::create_dir_all(&cache).expect("cache dir");
    let zero_layer = || -> [Vec<f64>; 7] {
        std::array::from_fn(|k| vec![0.0f64; Kind::ALL[k].width(&cfg).expect("width")])
    };
    let mut sums: Vec<[Vec<f64>; 7]> = (0..cfg.n_layer).map(|_| zero_layer()).collect();
    let mut gmax: Vec<[f64; 7]> = vec![[0.0; 7]; cfg.n_layer];
    let mut tau_max: Vec<[f64; 7]> = vec![[0.0; 7]; cfg.n_layer];
    let t0 = Instant::now();
    // `two_phase` (`importance.py:181-223`): profile every sample, keep the maximum threshold.
    if strat == Strategy::TwoPhase {
        for (c, tokens) in set.iter().enumerate() {
            onorm_pass(&engine, &cfg, &w, tokens, |l, k, g| {
                let wd = k.width(&cfg).expect("width");
                assert_eq!(g.len() % wd, 0, "l{l} {k:?}: gradient width");
                let mut norms: Vec<f64> = g
                    .chunks_exact(wd)
                    .map(|row| {
                        row.iter()
                            .map(|&v| from_grid(v).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect();
                let t = Clip::tau(&mut norms);
                tau_max[l][k.index()] = tau_max[l][k.index()].max(t);
            });
            println!(
                "ONORM profiling sample {c} at {:.1} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let mut seen_rows = 0usize;
    for (c, tokens) in set.iter().enumerate() {
        onorm_pass(&engine, &cfg, &w, tokens, |l, k, g| {
            let i = k.index();
            let d = &mut sums[l][i];
            let wd = d.len();
            assert_eq!(g.len() % wd, 0, "l{l} {k:?}: gradient width");
            // `online` (`importance.py:104-140`): this sample's threshold first grows the running
            // maximum and rescales what earlier samples accumulated; `two_phase` uses the fixed
            // maximum profiled above.
            let tau = if strat == Strategy::Online {
                let mut norms: Vec<f64> = g
                    .chunks_exact(wd)
                    .map(|row| {
                        row.iter()
                            .map(|&v| from_grid(v).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect();
                let t = Clip::tau(&mut norms);
                let gm = &mut gmax[l][i];
                if t > *gm {
                    let r = if *gm > 0.0 { (t / *gm).powi(2) } else { 1.0 };
                    for x in d.iter_mut() {
                        *x *= r;
                    }
                    *gm = t;
                }
                *gm
            } else {
                tau_max[l][i]
            };
            let clip = Clip { tau };
            for row in g.chunks_exact(wd) {
                clip.add_row(d, row);
            }
        });
        seen_rows += tokens.len();
        println!("ONORM sample {c} at {:.1} s", t0.elapsed().as_secs_f64());
    }
    for (l, layer) in sums.iter_mut().enumerate() {
        for k in Kind::ALL {
            let v = &mut layer[k.index()];
            for x in v.iter_mut() {
                // Mean over the sample's rows in the reference's `1e6` unit: the hook's
                // `.mean(dim=0)` divides by all `SEQ` rows, and the last row's gradient is zero.
                *x = *x / seen_rows as f64 * 1e6;
            }
            shrink(v, shrink_to);
            let words: Vec<i64> = v
                .iter()
                .map(|&x| to_grid(x, GRID_BOUND).expect("o_norm grid"))
                .collect();
            write_words(
                &format!("{cache}/g{l}.{tag}.{}.o_norm.i64", ONORM_NAME[k.index()]),
                1,
                v.len(),
                &words,
            )
            .expect("write o_norm");
        }
    }
    println!(
        "ONORM {tag}: {} layers, {} samples x {SEQ} tokens, shrink {shrink_to}, written to {cache}",
        cfg.n_layer,
        seen_rows / SEQ
    );
}

/// The reference dump's per-linear names, in the pack's kind order.
const DUMP_KINDS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];
/// The calibration cache's short kind names, the same order.
const DUMP_SHORT: [&str; 7] = ["q", "k", "v", "o", "gate", "up", "down"];
/// The `GGUF` tensor suffix of each kind.
const DUMP_TENSORS: [&str; 7] = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];
/// The input Gram group of each kind in [`calib_diag_write`]'s cache.
const DUMP_GRAMS: [&str; 7] = ["qkv", "qkv", "qkv", "o", "gate_up", "gate_up", "down"];

/// Read a `(1, n)` Q32.32 statistic written by [`write_words`]. `O(n)`.
fn read_stat_words(path: &str, want: usize) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(b.len() >= 16, "{path}: short header");
    let word = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().expect("eight bytes"));
    let (o, n) = (word(0), word(8));
    assert_eq!((o, n), (1, want as u64), "{path}: header");
    assert_eq!(b.len(), 16 + 8 * want, "{path}: payload");
    let mut out = Vec::with_capacity(want);
    for c in b[16..].chunks(8) {
        out.push(from_grid(i64::from_le_bytes(
            c.try_into().expect("eight bytes"),
        )));
    }
    out
}

/// Write `v` as little-endian f32. `O(v.len())`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the dump is the reference's f32 carrier; the statistic is already below 2^127"
)]
fn write_f32(path: &str, v: &[f64]) {
    let b: Vec<u8> = v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect();
    std::fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}

/// One shuffle of `0..n` for draw `seed`: xorshift64\* and Fisher–Yates, `O(n)`.
fn perm_draw(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut s = if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    };
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// The eight epoch draws of one `tune_nonfact` / `tune_fact` call. The reference calls
/// `set_seed(0)` at the start of every call (`compress_block.py:53,161`) and draws one
/// `randperm(num_calib_samples)` per epoch, so every slot and both phases replay the same eight
/// permutations. The draw is our xorshift Fisher–Yates, not torch's CPU MT19937; the re-seeding,
/// which is what fixes the order, is reproduced. `O(epochs·n)`.
fn epoch_perms(n: usize) -> Vec<Vec<usize>> {
    (0..8u64)
        .map(|e| perm_draw(n, 0x9e37_79b9_7f4a_7c15 ^ e))
        .collect()
}

/// Convert the calibration cache into the layout `NanoQuant`'s device drivers read:
/// `calib_ids.i32`, `ranks.json`, `perm.json` and `stats/model.layers.{l}.{kind}.{i,o}_norm.f32`.
///
/// The ids are the base file's first `APEIRON_NQ_SAMPLES` samples of [`SEQ`] tokens (default
/// [`CALIB_SAMPLES`]); the ranks are the same `rank_for_bits` dial `nq_fit` uses at
/// `APEIRON_NQ_PPM` (default 1,174,000); the perms are the reference's re-seeded eight epoch draws
/// (`epoch_perms`), replayed by every slot and both phases. This is the campaign's input step, not
/// a fit: it reads the teacher's directory and the statistics the two calibration tests wrote.
///
/// Env: `APEIRON_CALIB_MODEL`, `APEIRON_CALIB_BASE`, `APEIRON_CALIB_CACHE`, `APEIRON_NQ_DUMP`,
/// `APEIRON_NQ_SAMPLES`, `APEIRON_NQ_PPM`.
#[test]
#[ignore = "converts the calibration cache; run once before the device campaign"]
#[allow(
    clippy::too_many_lines,
    reason = "one dump: the ids, the ranks from the tensor directory, the two statistics per \
              linear and the 112 epoch draws share one open header and one base file"
)]
fn nq_dump_write() {
    let model = env_path("APEIRON_CALIB_MODEL", "~/nq-ref/q3-0p6b-f16.gguf");
    let base = env_path("APEIRON_CALIB_BASE", "~/kld08/calib.s128.kld");
    let cache = env_path("APEIRON_CALIB_CACHE", "~/kld08/gatecode-tok");
    let dump = env_path("APEIRON_NQ_DUMP", "~/nq-ref/dump");
    let samples = env_usize("APEIRON_NQ_SAMPLES", CALIB_SAMPLES);
    let ppm: u64 = std::env::var("APEIRON_NQ_PPM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_174_000);
    let tag = format!("s{samples}");
    std::fs::create_dir_all(format!("{dump}/stats")).expect("dump stats dir");
    let set = calib_samples(&base, samples);
    let mut ids: Vec<u8> = Vec::with_capacity(4 * samples * SEQ);
    for c in &set {
        assert_eq!(c.len(), SEQ, "sample width");
        for &t in c {
            ids.extend_from_slice(&t.cast_signed().to_le_bytes());
        }
    }
    std::fs::write(format!("{dump}/calib_ids.i32"), &ids).expect("calib ids");
    let g = Gguf::open(&model).expect("gguf header");
    let mut n_layer = 0usize;
    while g
        .header
        .tensor(&format!("blk.{n_layer}.attn_q.weight"))
        .is_some()
    {
        n_layer += 1;
    }
    assert!(n_layer > 0, "no blk.0.attn_q.weight in {model}");
    let width = |l: usize, k: usize| -> (usize, usize) {
        let name = format!("blk.{l}.{}.weight", DUMP_TENSORS[k]);
        let t = g.header.tensor(&name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(t.dims.len(), 2, "{name}: rank");
        let inn = usize::try_from(t.dims[0]).expect("inn");
        let out = usize::try_from(t.dims[1]).expect("out");
        (out, inn)
    };
    let mut ranks = String::from("{");
    for l in 0..n_layer {
        for (k, name) in DUMP_KINDS.iter().enumerate() {
            let (out, inn) = width(l, k);
            let r = rank_for_bits(out, inn, ppm, 32);
            if l + k > 0 {
                ranks.push(',');
            }
            write!(ranks, "\"{l}.{name}\": {r}").expect("ranks");
        }
    }
    ranks.push('}');
    std::fs::write(format!("{dump}/ranks.json"), ranks).expect("ranks");
    for l in 0..n_layer {
        for (k, name) in DUMP_KINDS.iter().enumerate() {
            let (out, inn) = width(l, k);
            let diag = read_stat_words(
                &format!("{cache}/g{l}.{tag}.{}.diag.i64", DUMP_GRAMS[k]),
                inn,
            );
            write_f32(
                &format!("{dump}/stats/model.layers.{l}.{name}.i_norm.f32"),
                &diag,
            );
            let onorm = read_stat_words(
                &format!("{cache}/g{l}.{tag}.{}.o_norm.i64", DUMP_SHORT[k]),
                out,
            );
            write_f32(
                &format!("{dump}/stats/model.layers.{l}.{name}.o_norm.f32"),
                &onorm,
            );
        }
    }
    let mut perms = String::from("[");
    let mut first = true;
    let draws = epoch_perms(samples);
    for _si in 0..7usize {
        for phase in 0..2u64 {
            for idx in &draws {
                if !first {
                    perms.push(',');
                }
                first = false;
                let caller = if phase == 0 {
                    "tune_nonfact"
                } else {
                    "tune_fact"
                };
                let index: Vec<String> = idx.iter().map(usize::to_string).collect();
                write!(
                    perms,
                    "{{\"caller\": \"{caller}\", \"index\": [{}]}}",
                    index.join(", ")
                )
                .expect("perms");
            }
        }
    }
    perms.push(']');
    std::fs::write(format!("{dump}/perm.json"), perms).expect("perms");
    println!(
        "DUMP {dump}: {n_layer} layers, {samples} samples x {SEQ} tokens, ppm {ppm}, tag {tag}"
    );
}

#[test]
fn epoch_perms_repeat_for_every_slot_and_phase() {
    let draws = epoch_perms(128);
    assert_eq!(draws.len(), 8);
    for (e, d) in draws.iter().enumerate() {
        assert_eq!(d.len(), 128, "epoch {e}");
        let mut s = d.clone();
        s.sort_unstable();
        assert_eq!(s, (0..128).collect::<Vec<_>>(), "epoch {e}");
    }
    // The writer emits 14 groups (7 slots × 2 phases); every one replays these eight draws, because
    // the reference calls `set_seed(0)` at the start of every `tune_nonfact` / `tune_fact`.
    let mut all = Vec::new();
    for _ in 0..14 {
        all.extend(draws.iter().cloned());
    }
    for g in all.chunks(8) {
        assert_eq!(g, &draws[..]);
    }
    assert_ne!(draws[0], draws[1], "the eight epochs are distinct draws");
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

/// [`Clip`] pins the two rules the reference's `two_phase` mode states: the threshold is the
/// `99.9`th percentile of the row norms (`k = max(1, ⌊0.001·n⌋)`-th largest), and a row above it
/// contributes `(tau/‖x‖)²·x²`.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "the percentile of the integers 1..=2000 and the single-row norm are exact"
)]
fn clip_takes_the_percentile_and_scales_rows_above_it() {
    let mut norms: Vec<f64> = (1..=2000).map(f64::from).collect();
    assert_eq!(Clip::tau(&mut norms), 1999.0);
    let mut small = vec![7.0];
    assert_eq!(Clip::tau(&mut small), 7.0);
    // Rows of norm 1 and 4 under tau 2: the second is scaled to norm 2, so 1² + 2² = 5.
    let one = 1i64 << 32;
    let mut d = vec![0.0f64; 2];
    Clip { tau: 2.0 }.add(&mut d, &[vec![one, 0], vec![4 * one, 0]]);
    assert_eq!(d, vec![5.0, 0.0]);
}

/// [`shrink`] pins the reference's `0.4` covariance shrinkage: the endpoints are the identity and
/// the constant mean, and the half-way point is their midpoint.
#[test]
fn shrink_pulls_each_coordinate_to_the_mean() {
    let mut v = vec![1.0, 3.0];
    shrink(&mut v, 0.5);
    assert_eq!(v, vec![1.5, 2.5]);
    shrink(&mut v, 0.0);
    assert_eq!(v, vec![1.5, 2.5]);
    shrink(&mut v, 1.0);
    assert_eq!(v, vec![2.0, 2.0]);
}
