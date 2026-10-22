//! Board . Lane (gate, up) and the `ffn_down` arm.
//!
//! The FFN gate as a hyperplane code on `Qwen3.5-0.8B`: under the production `fold_fit` arm,
//! per layer, (1) the off/near/on split of `g·x`, (2) the sign-flip rate of `ĝ·x` against
//! `arccos(ρ_H)/π` (Sheppard 1899; Charikar 2002), (3) the gate/up split of `E‖δh‖²`, and (4) the
//! held-out KL with only `ffn_gate`, only `ffn_up` or only `ffn_down` replaced.
//!
//! Every statistic is a Q32.32 word computed from the engine's own tap stage 3 (`STAGE_LN2_OUT`,
//! the gate/up input) and the teacher's own `ffn_gate`/`ffn_up` rows, so the flip is the weight's
//! and not the activation quantiser's. The engine is read on its one inference path
//! (`Engine::prefill_tap_cols`, `KernelMode::Q8Act`).
//!
//! Stages, selected by `APEIRON_GATE_STAGE`:
//! - `probe`: one 512-token tapped prefill and one production `fold_fit`, so the cost equation is
//!   measured before a campaign is launched.
//! - `fit`: calibration `H` from chunks `0..CALIB`, then the production fit of
//!   `ffn_gate` and `ffn_up` for `APEIRON_GATE_LAYERS`, cached as Q32.32 words.
//! - `stats`: the per-layer table over the held-out chunks.
//! - `kl`: held-out KL of one arm against the llama.cpp `--kl-divergence-base` file.
//!
//! Grade `measured`: every number is the print of a command that ran.
//!
//! Token disjointness: `H` is accumulated over chunk indices `0..calib` of the base file's token
//! block; every statistic and every KL row is scored on chunk indices `calib..`, a distinct text
//! range of the same corpus, so no token is in both sets.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "the Q32.32 ↔ f64 boundary is range-checked by the grid it came from; the \
              single-letter names are the geometry's symbols (`g`, `u`, `h`, `x`, `t`, `o`, `n`) \
              and every loop is one linear composition of the stage it measures"
)]

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::gguf::Gguf;
use apeiron::format::quant;
use apeiron::learn::fold::fit::fold_fit::{fold_fit, FoldFitConfig};
use apeiron::learn::fold::run::convert::{encode_f16, encode_f32};
use apeiron::learn::fold::stats::capture::{Grams, LayerGrams, Normalisation};
use apeiron::learn::fold::stats::heldout::{held_out, held_out_kldbase_chunk};
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::load::layer_from_engine;
use apeiron::learn::grad::model::Config;
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// The teacher image: the `Qwen3.5-0.8B` artefact, binary16. The engine refuses an `F32`
/// embedding row (`src/engine/mod.rs:29`), so `qwen08b-f32.gguf` is transcoded to binary16 with
/// `llama-quantize`; every other tensor keeps the `f32` value within binary16's 11-bit mantissa.
const MODEL_DEFAULT: &str = "~/kld08/qwen08b-f16.gguf";

/// The teacher image path, overridable by `APEIRON_GATE_MODEL`.
fn model_path() -> String {
    std::env::var("APEIRON_GATE_MODEL").unwrap_or_else(|_| String::from(MODEL_DEFAULT))
}
/// llama.cpp's `--kl-divergence-base` file for the binary32 teacher, 40 chunks of `n_ctx = 512`.
const BASE: &str = "~/kld08/calib.c40.kld";

/// Calibration chunks `0..CALIB`: 8 chunks of 512 tokens give 4096 rows, above the `n_ff = 3584`
/// width of the down Gram; 2 chunks bound its rank by 1024 and `fold_fit` refuses it as not
/// positive-definite. Every held-out read starts at chunk `CALIB` or later.
/// Fixed with the production Grams under board .
const CALIB: usize = 8;

/// The base file path, overridable by `APEIRON_GATE_BASE` (the binary16 teacher's own base).
fn base_path() -> String {
    std::env::var("APEIRON_GATE_BASE").unwrap_or_else(|_| String::from(BASE))
}
/// Where the fitted rows and the calibration Gram are cached as Q32.32 words.
const CACHE: &str = "~/kld08/gatecode-tok";

/// Tap slot of the FFN-side `RMSNorm` output: the gate/up input (`STAGE_LN2_OUT`).
const TAP_LN2: usize = 3;
/// Tap slot of the gate pre-activation (`STAGE_GATE`).
const TAP_GATE: usize = 8;
/// Tap slot of the `SwiGLU` product `m = silu(g·x)⊙(u·x)`, the `ffn_down` input (`STAGE_M`).
const TAP_M: usize = 4;

/// The production call-site configuration (`fold_transfer.py:1733-1736`).
fn production_cfg() -> FoldFitConfig {
    FoldFitConfig {
        iters: 1,
        ridge: 4_294_967,
        damp: 42_949_673,
        inner_iters: 3,
        e8_beam: 4,
        icm_ridge: 858_993_459,
        sweeps: 2,
    }
}

/// An environment knob with a default.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// The layer list from `APEIRON_GATE_LAYERS` (`0,3,7`), or every layer.
fn env_layers(n_layer: usize) -> Vec<usize> {
    match std::env::var("APEIRON_GATE_LAYERS") {
        Ok(s) if !s.is_empty() => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        _ => (0..n_layer).collect(),
    }
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

/// Write `words` as a little-endian `i64` payload with an `(o, n)` header.
fn write_words(path: &str, o: usize, n: usize, words: &[i64]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(o as u64).to_le_bytes())?;
    f.write_all(&(n as u64).to_le_bytes())?;
    for w in words {
        f.write_all(&w.to_le_bytes())?;
    }
    Ok(())
}

/// Read a [`write_words`] payload back as `(o, n, words)`.
fn read_words(path: &str) -> (usize, usize, Vec<i64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let o = u64::from_le_bytes(b[0..8].try_into().expect("8")) as usize;
    let n = u64::from_le_bytes(b[8..16].try_into().expect("8")) as usize;
    let words = b[16..]
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8")))
        .collect();
    (o, n, words)
}

/// `Y = W·X` in `f64`: `w` is `o × n` row-major, `x` is `t × n` token-major, the result `o × t`
/// unit-major. `O(o·n·t)` multiply-adds; the inner loop runs over the contiguous token axis.
fn project(w: &[f64], x: &[f64], o: usize, n: usize, t: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; o * t];
    for u in 0..o {
        let row = &w[u * n..(u + 1) * n];
        let dst = &mut out[u * t..(u + 1) * t];
        for (k, &wk) in row.iter().enumerate() {
            let xk = &x[k * t..(k + 1) * t];
            for (d, &x) in dst.iter_mut().zip(xk) {
                *d += wk * x;
            }
        }
    }
    out
}

/// The logistic function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `silu(x) = x·σ(x)`.
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// `d/dx silu(x) = σ(x)·(1 + x·σ(−x))`.
fn dsilu(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
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

/// `vᵀ·H·v` for the symmetric `h`, `n × n`.
fn quad(h: &[f64], v: &[f64], n: usize) -> f64 {
    let mut hv = vec![0.0f64; n];
    for i in 0..n {
        let row = &h[i * n..(i + 1) * n];
        let mut s = 0.0f64;
        for (a, b) in row.iter().zip(v) {
            s += a * b;
        }
        hv[i] = s;
    }
    let mut out = 0.0f64;
    for (a, b) in hv.iter().zip(v) {
        out += a * b;
    }
    out
}

/// The gate and up row of one layer as the teacher's Q32.32 words, with the layer's mixer name.
fn gate_up(engine: &Engine, cfg: &Config, l: usize) -> (Vec<i64>, Vec<i64>, String) {
    let ll = layer_from_engine(engine, cfg, l).expect("layer linears");
    let take = |name: &str| {
        ll.linears
            .iter()
            .find(|x| x.name == name)
            .unwrap_or_else(|| panic!("blk.{l}.{name}"))
            .clone()
    };
    let g = take("ffn_gate");
    let u = take("ffn_up");
    assert_eq!(g.rows, cfg.n_ff, "layer {l}: n_ff");
    assert_eq!(g.cols, cfg.n_embd, "layer {l}: n_embd");
    (g.w, u.w, format!("{:?}", ll.mixer))
}

/// The teacher's `ffn_down` words (`n_embd × n_ff`) of one layer, with the layer's mixer name.
fn ffn_down(engine: &Engine, cfg: &Config, l: usize) -> (Vec<i64>, String) {
    let ll = layer_from_engine(engine, cfg, l).expect("layer linears");
    let d = ll
        .linears
        .iter()
        .find(|x| x.name == "ffn_down")
        .unwrap_or_else(|| panic!("blk.{l}.ffn_down"))
        .clone();
    assert_eq!(d.rows, cfg.n_embd, "layer {l}: down rows");
    assert_eq!(d.cols, cfg.n_ff, "layer {l}: down cols");
    (d.w, format!("{:?}", ll.mixer))
}

/// The per-layer geometric piece of the metric.
#[derive(Debug, Clone, Copy)]
struct LayerFit {
    /// `arccos(ρ_H)/π`.
    angle: f64,
    /// `‖g − ĝ‖_H / ‖g‖_H`.
    rel_g: f64,
    /// `‖u − û‖_H / ‖u‖_H`.
    rel_u: f64,
}

/// `ρ_H(g, ĝ)` and the two relative `H`-errors for one layer.
fn layer_fit(h: &[f64], g: &[f64], gh: &[f64], u: &[f64], uh: &[f64], n: usize) -> LayerFit {
    let (ghg, ghgh) = (quad(h, g, n), quad(h, gh, n));
    let (uhu, _uhuh) = (quad(h, u, n), quad(h, uh, n));
    let mut dot = 0.0f64;
    for i in 0..n {
        let row = &h[i * n..(i + 1) * n];
        let mut s = 0.0f64;
        for (a, b) in row.iter().zip(gh) {
            s += a * b;
        }
        dot += s * g[i];
    }
    let rho = dot / (ghg.max(0.0).sqrt() * ghgh.max(0.0).sqrt());
    let dg: Vec<f64> = g.iter().zip(gh).map(|(a, b)| a - b).collect();
    let du: Vec<f64> = u.iter().zip(uh).map(|(a, b)| a - b).collect();
    LayerFit {
        angle: rho.clamp(-1.0, 1.0).acos() / std::f64::consts::PI,
        rel_g: (quad(h, &dg, n) / ghg).max(0.0).sqrt(),
        rel_u: (quad(h, &du, n) / uhu).max(0.0).sqrt(),
    }
}

/// The production calibration Grams over chunks `0..calib`: `Grams::accumulate_tap` on each tapped
/// chunk, then `Normalisation::Mean`. `gate_up` is the `ffn_gate`/`ffn_up` input Gram (stage 3) and
/// `down` the `ffn_down` input Gram (stage 4, the engine's own `SwiGLU` product), both `Σ_t x_t x_tᵀ / T`
/// over tokens, so this file restates no layout.
///
/// Layout control, per layer in `layers`: the diagonal of each Gram against `Σ_t x_t[k]² / T`
/// summed straight from the tap columns, relative error below `1e-4`. A feature-major buffer read as
/// token rows (the defect this replaces) misses it by `O(1)`.
///
/// `O(calib·n_layer·t·(n_embd² + q_out² + n_ff²)/2)` multiply-adds in the accumulator plus
/// `O(calib·|layers|·t·(n_embd + n_ff))` for the control; space the accumulator's
/// `O(n_layer·(n_embd² + q_out² + n_ff²))` words and one `f64` copy.
fn calib_grams(
    engine: &Engine,
    cfg: &Config,
    chunks: &[Vec<u32>],
    calib: usize,
    layers: &[usize],
) -> Vec<LayerGrams> {
    let (n, f) = (cfg.n_embd, cfg.n_ff);
    let mut grams = Grams::new(cfg).expect("grams");
    let mut diag: Vec<(Vec<f64>, Vec<f64>)> = vec![(vec![0.0; n], vec![0.0; f]); cfg.n_layer];
    let mut seen = 0usize;
    let t0 = Instant::now();
    for (c, tokens) in chunks.iter().enumerate().take(calib) {
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
        println!(
            "GRAMS calib chunk {c} at {:.1} s",
            t0.elapsed().as_secs_f64()
        );
    }
    let done = grams.finish(Normalisation::Mean).expect("finish");
    for &l in layers {
        let checks = [
            ("gate_up", &done[l].gate_up, &diag[l].0, n),
            ("down", &done[l].down, &diag[l].1, f),
        ];
        for (name, g, d, w) in checks {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for k in 0..w {
                let e = d[k] / seen as f64;
                num += (g[k * w + k] - e).powi(2);
                den += e * e;
            }
            let rel = (num / den).sqrt();
            println!("GRAMS layer {l} {name} diag rel {rel:.3e}");
            assert!(
                rel < 1e-4,
                "layer {l} {name}: Gram diagonal off the tap columns by {rel}"
            );
        }
    }
    done
}

/// The cost equation and the two fixtures: one 512-token tapped prefill, one production `fold_fit`.
#[test]
#[ignore = "opens the 3.1 GB F32 teacher; run explicitly with --ignored"]
fn gate_code_probe() {
    let t = Instant::now();
    let engine = Engine::open(&model_path()).expect("open teacher");
    println!("PROBE load {:.1} s", t.elapsed().as_secs_f64());
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    println!(
        "PROBE geometry n_embd {} n_ff {} n_layer {} chunks {} n_ctx {}",
        cfg.n_embd,
        cfg.n_ff,
        cfg.n_layer,
        chunks.len(),
        n_ctx
    );
    let t = Instant::now();
    let rows = tapped(&engine, &chunks[0]);
    let prefill = t.elapsed().as_secs_f64();
    for l in 0..cfg.n_layer {
        assert_eq!(
            rows[l][TAP_LN2].len(),
            n_ctx,
            "layer {l}: stage 3 records every column"
        );
        assert_eq!(
            rows[l][TAP_LN2][0].len(),
            cfg.n_embd,
            "layer {l}: stage 3 width"
        );
    }
    println!("PROBE tap {prefill:.2} s per {n_ctx}-token chunk");
    let n = cfg.n_embd;
    let t = Instant::now();
    let grams = calib_grams(&engine, &cfg, &chunks, 1, &[0]);
    let gram = t.elapsed().as_secs_f64();
    println!("PROBE grams {gram:.2} s per chunk, all layers, tap included");
    let (g, u, mixer) = gate_up(&engine, &cfg, 0);
    let hw = qv(&grams[0].gate_up);
    let t = Instant::now();
    let fit = fold_fit(&g, &hw, cfg.n_ff, n, None, &production_cfg()).expect("fit");
    let fw = t.elapsed().as_secs_f64();
    println!(
        "PROBE fit layer 0 gate {mixer} {fw:.2} s, {} words, bits r/c/d {}/{}/{}",
        fit.wp.len(),
        fit.bits_r,
        fit.bits_c,
        fit.bits_d
    );
    let _ = u;
    let layers2 = 2.0 * f64::from(u32::try_from(cfg.n_layer).expect("24"));
    println!(
        "PROBE equation: fit {:.0} s for {layers2} tensors; H {gram:.0} s per calib chunk; \
         stats {prefill:.0} s per layer-chunk",
        fw * layers2
    );
}

/// Accumulate the calibration `H` over chunks `0..CALIB` and cache it with the fits.
#[test]
#[ignore = "opens the 3.1 GB F32 teacher and runs production fold_fit per tensor; run explicitly"]
fn gate_code_fit() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = env_layers(cfg.n_layer);
    let calib = env_usize("APEIRON_GATE_CALIB", CALIB);
    let n = cfg.n_embd;
    std::fs::create_dir_all(CACHE).expect("cache dir");
    let t0 = Instant::now();
    let grams = calib_grams(&engine, &cfg, &chunks, calib, &layers);
    let tokens_seen = calib * n_ctx;
    for &l in &layers {
        let hw = qv(&grams[l].gate_up);
        write_words(&format!("{CACHE}/h{l}.i64"), n, n, &hw).expect("cache H");
        let (g, u, mixer) = gate_up(&engine, &cfg, l);
        for (name, w) in [("gate", &g[..]), ("up", &u[..])] {
            let t = Instant::now();
            let fit = fold_fit(w, &hw, cfg.n_ff, n, None, &production_cfg()).expect("fit");
            let wall = t.elapsed().as_secs_f64();
            write_words(&format!("{CACHE}/l{l}.{name}.i64"), cfg.n_ff, n, &fit.wp)
                .expect("cache fit");
            println!(
                "FIT layer {l} {mixer} {name} {wall:.2} s bits r/c/d {}/{}/{} tag {}",
                fit.bits_r,
                fit.bits_c,
                fit.bits_d,
                fit.tag.as_str()
            );
        }
    }
    println!(
        "FIT done {} layers, {tokens_seen} calibration tokens, {:.1} s",
        layers.len(),
        t0.elapsed().as_secs_f64()
    );
}

/// The down calibration `H` (the production `Grams::down`, stage 4, `n_ff × n_ff`) over chunks `0..CALIB`, then
/// the production fit of `ffn_down` for `APEIRON_GATE_LAYERS`, cached beside the gate/up fits.
///
/// `H` costs what [`calib_grams`] states; then one `fold_fit` per layer.
#[test]
#[ignore = "opens the teacher and runs production fold_fit on ffn_down; run explicitly"]
fn gate_code_fit_down() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = env_layers(cfg.n_layer);
    let calib = env_usize("APEIRON_GATE_CALIB", CALIB);
    let (n, f) = (cfg.n_embd, cfg.n_ff);
    std::fs::create_dir_all(CACHE).expect("cache dir");
    let t0 = Instant::now();
    let grams = calib_grams(&engine, &cfg, &chunks, calib, &layers);
    let tokens_seen = calib * n_ctx;
    for &l in &layers {
        let hw = qv(&grams[l].down);
        let (d, mixer) = ffn_down(&engine, &cfg, l);
        let t = Instant::now();
        let fit = fold_fit(&d, &hw, n, f, None, &production_cfg()).expect("fit");
        let wall = t.elapsed().as_secs_f64();
        write_words(&format!("{CACHE}/l{l}.down.i64"), n, f, &fit.wp).expect("cache fit");
        println!(
            "FITD layer {l} {mixer} down {wall:.2} s bits r/c/d {}/{}/{} tag {}",
            fit.bits_r,
            fit.bits_c,
            fit.bits_d,
            fit.tag.as_str()
        );
    }
    println!(
        "FITD done {} layers, {tokens_seen} calibration tokens, {:.1} s",
        layers.len(),
        t0.elapsed().as_secs_f64()
    );
}

/// `tr(E H Eᵀ)` for the `o × n` word matrix `e` under the symmetric `n × n` Gram `h`: one [`quad`]
/// per row, `Σ_u e_uᵀ H e_u`.
///
/// `O(o·n²)` multiply-adds for the `o` quadratic forms, the dominant per-layer term beside
/// `fold_fit`; space `O(n)` for the converted row.
fn energy(h: &[f64], e: &[i64], o: usize, n: usize) -> f64 {
    let mut acc = 0.0f64;
    for u in 0..o {
        acc += quad(h, &unq(&e[u * n..(u + 1) * n]), n);
    }
    acc
}

/// The input saliency of each of the `n` input columns of the `o × n` teacher tensor `w`
/// (LLM.int8, Dettmers et al. 2022; `SpQR`, Dettmers et al. 2023, arXiv 2306.03078):
/// `s_i = H_ii · Σ_u W_ui²`.
///
/// `O(o·n)` multiply-adds, one pass over `w`; space `O(n)`.
fn salience(h: &[f64], w: &[i64], o: usize, n: usize) -> Vec<f64> {
    let mut sq = vec![0.0f64; n];
    for u in 0..o {
        for (i, &word) in w[u * n..(u + 1) * n].iter().enumerate() {
            let v = from_grid(word);
            sq[i] += v * v;
        }
    }
    (0..n).map(|i| h[i * n + i] * sq[i]).collect()
}

/// The per-input score `c_i = Σ_j (EᵀE)_ij H_ji` of the `o × n` word error `e`: the `energy`
/// selector, the closed form of the energy a column's removal takes (`down_keep_fit`'s `c`).
///
/// `O(o·n²)` multiply-adds for the symmetric `EᵀE` (each of the `n²/2` upper entries one `o`-term
/// dot), then `O(n²)` for the contraction; space `O(n²)` for the Gram.
fn energy_cols(e: &[i64], h: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; n * n];
    for u in 0..o {
        let row = unq(&e[u * n..(u + 1) * n]);
        for i in 0..n {
            let vi = row[i];
            let dst = &mut g[i * n + i..(i + 1) * n];
            for (a, &vj) in dst.iter_mut().zip(&row[i..]) {
                *a += vi * vj;
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            g[i * n + j] = g[j * n + i];
        }
    }
    (0..n)
        .map(|i| (0..n).map(|j| g[i * n + j] * h[j * n + i]).sum::<f64>())
        .collect()
}

/// The `k` largest entries of `score`, ties broken by the lower index: a descending sort on the
/// score with ascending index as its second key, truncated.
///
/// `O(n log n)`; space `O(n)` for the index vector.
fn top_k(score: &[f64], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| {
        score[b]
            .partial_cmp(&score[a])
            .expect("finite score")
            .then(a.cmp(&b))
    });
    idx.truncate(k.min(idx.len()));
    idx
}

/// **Measurement 6** (board ). The `SpQR` masked refit of `ffn_up` or
/// `ffn_gate`, ported from `down_spqr_fit` (`tests/learn_down_code.rs`): `K` input columns `S` are
/// restored from the teacher after a fit of the other columns that already knows they will be, so
/// the fit minimises its error on the masked metric `Hm` — `H` with `S`'s rows and columns zeroed
/// and `ε` on `S`'s diagonal — against the target `W' = zeroCols(S, W)`.
///
/// The identity (`QuantVerdicts.ColumnRestore.masked_energy`) is
/// `tr(E Hm Eᵀ) = J(zeroCols_S E) + ε Σ_S ‖e_i‖²`, so the masked fit minimises the restored energy
/// plus an `ε` penalty on `S` alone; `Hm`'s off-`S` blocks are `H`'s, so the fit spends no error on
/// columns that will be overwritten. `ε = 1e-2 · mean(diag H)` keeps `Hm`'s `S`-diagonal above the
/// rounding grid.
///
/// `APEIRON_GATE_KIND` (`up` | `gate`), `APEIRON_GATE_LAYERS` (default: every layer), `K` =
/// `APEIRON_GATE_KEEP` (default 10, the top 0.98% of `n_embd`), `APEIRON_GATE_SELECT`
/// (`salience` | `energy`). `H` is the cached `h{l}.i64`, the calibration `gate_up` Gram of chunks
/// `0..CALIB`; no held-out chunk is read. Writes `{CACHE}/l{l}.{kind}-spqr{K}-{select}.i64`, the fit
/// with `S`'s columns taken from the teacher, for [`gate_code_kl`] (`APEIRON_GATE_ARM`).
///
/// Cost: `fold_fit` dominates at ~50 s per layer measured on this box (the count, not the wall, is
/// the claim: `O(o·n² + gptq + inner_iters·(o·n² + n³ + e8_icm))`); three [`energy`] calls at
/// `O(n_ff·n_embd²)` each — the `before` one, the masked `obj` one and the restored `after` one —
/// with the `energy` selector's `EᵀE` another `O(n_ff·n_embd²)`, `O(n²)` to mask `Hm`, and `O(o·n)`
/// to zero `W'` and restore `S`. Space `O(n² + o·n)` per layer.
#[test]
#[ignore = "reads the cached gate_up Gram and runs production fold_fit on the masked metric"]
fn ffn_spqr_fit() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let layers = env_layers(cfg.n_layer);
    let kind = std::env::var("APEIRON_GATE_KIND").unwrap_or_else(|_| String::from("up"));
    assert!(
        matches!(kind.as_str(), "gate" | "up"),
        "APEIRON_GATE_KIND {kind}"
    );
    let k = env_usize("APEIRON_GATE_KEEP", 10);
    let select = std::env::var("APEIRON_GATE_SELECT").unwrap_or_else(|_| String::from("salience"));
    assert!(
        matches!(select.as_str(), "salience" | "energy"),
        "APEIRON_GATE_SELECT {select}"
    );
    // The tensor is `n_ff × n_embd`: `o` rows, `n` input columns, `H` the `gate_up` Gram.
    let (o, n) = (cfg.n_ff, cfg.n_embd);
    let bits = k * o * 16 + k * 12;
    println!(
        "SPQR layers {layers:?} kind {kind} K {k} select {select} n_ff {o} n_embd {n} \
         (H = calibration chunks 0..{CALIB} gate_up Gram)"
    );
    let t0 = Instant::now();
    let (mut sum_before, mut sum_after) = (0.0f64, 0.0f64);
    for &l in &layers {
        let (ho, hn, hw) = read_words(&format!("{CACHE}/h{l}.i64"));
        assert_eq!((ho, hn, hw.len()), (n, n, n * n), "layer {l}: cached H");
        let h = unq(&hw);
        let (g, up, _mixer) = gate_up(&engine, &cfg, l);
        let w = if kind == "gate" { g } else { up };
        let (wo, wn, cached) = read_words(&format!("{CACHE}/l{l}.{kind}.i64"));
        assert_eq!((wo, wn), (o, n), "layer {l}: cached {kind} fit shape");
        let e: Vec<i64> = cached.iter().zip(&w).map(|(a, b)| a - b).collect();
        let before = energy(&h, &e, o, n);
        let score = if select == "salience" {
            salience(&h, &w, o, n)
        } else {
            energy_cols(&e, &h, o, n)
        };
        let keep = top_k(&score, k);
        let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
        let eps = 1e-2 * mean_diag;
        let mut mask = vec![false; n];
        for &i in &keep {
            mask[i] = true;
        }
        let mut hm = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                hm[i * n + j] = if mask[i] || mask[j] {
                    if i == j {
                        eps
                    } else {
                        0.0
                    }
                } else {
                    h[i * n + j]
                };
            }
        }
        let mut dp = w.clone();
        for &i in &keep {
            for row in 0..o {
                dp[row * n + i] = 0;
            }
        }
        let t = Instant::now();
        let fit = match fold_fit(&dp, &qv(&hm), o, n, None, &production_cfg()) {
            Ok(fit) => fit,
            Err(err) => {
                println!("SPQR layer {l:02} kind {kind} select {select} K {k} fit error {err:?}");
                continue;
            }
        };
        let wall = t.elapsed().as_secs_f64();
        let mut out = fit.wp.clone();
        for &i in &keep {
            for row in 0..o {
                out[row * n + i] = w[row * n + i];
            }
        }
        let e2: Vec<i64> = out.iter().zip(&w).map(|(a, b)| a - b).collect();
        let after = energy(&h, &e2, o, n);
        let ego: Vec<i64> = fit.wp.iter().zip(&dp).map(|(a, b)| a - b).collect();
        let obj = energy(&hm, &ego, o, n);
        let mut pen = 0.0f64;
        for &i in &keep {
            for row in 0..o {
                let v = from_grid(ego[row * n + i]);
                pen += v * v;
            }
        }
        let rhs = after + eps * pen;
        let rel = (obj - rhs).abs() / rhs.abs().max(f64::MIN_POSITIVE);
        let ratio = if before > 0.0 { after / before } else { 0.0 };
        println!(
            "SPQR layer {l:02} kind {kind} select {select} K {k} before {before:.6e} after \
             {after:.6e} ratio {ratio:.4} ident rel {rel:.2e} fit {wall:.1} s bits +{bits}"
        );
        let path = format!("{CACHE}/l{l}.{kind}-spqr{k}-{select}.i64");
        write_words(&path, o, n, &out).expect("write spqr fit");
        println!("SPQR wrote {path}");
        sum_before += before;
        sum_after += after;
    }
    println!(
        "SPQR sum layers {} kind {kind} select {select} K {k} before {sum_before:.6e} after \
         {sum_after:.6e} ratio {:.4} bits +{bits} (+{:.4} b/w) wall {:.1} s",
        layers.len(),
        sum_after / sum_before.max(f64::MIN_POSITIVE),
        bits as f64 / (o * n) as f64,
        t0.elapsed().as_secs_f64()
    );
}

/// **Measurement 8** (board ). A second production `fold_fit` on the
/// residual of `ffn_up` or `ffn_gate`: `R = W − Ŵ`, `out = Ŵ + R̂`, so the added bits are one more
/// fit of the same tensor: its sign plane `o·n` plus `bits_r + bits_c + bits_d`; the metric is
/// unchanged.
///
/// `APEIRON_GATE_KIND` (`up` | `gate`, default `up`), `APEIRON_GATE_LAYERS` (default: every layer),
/// `APEIRON_GATE_PASSES` (`P`, default 2: the cached fit plus `P − 1` residual passes). `H` is the
/// cached `h{l}.i64`, the calibration `gate_up` Gram of chunks `0..CALIB`; no held-out chunk is read.
/// Writes `{CACHE}/l{l}.{kind}-res{P}.i64`, the summed words, for [`gate_code_kl`]
/// (`APEIRON_GATE_ARM=up-res2`); `P = 1` writes the cached fit back unchanged.
///
/// Cost: `(P−1)` `fold_fit` per layer (~33 s each, measured) plus `2(P−1)` [`energy`] at `O(o·n²)`;
/// space `O(n² + o·n)` per layer for the Gram, the words and the residual.
#[test]
#[ignore = "opens the teacher and runs production fold_fit on the residual; run explicitly"]
fn ffn_residual_fit() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let layers = env_layers(cfg.n_layer);
    let kind = std::env::var("APEIRON_GATE_KIND").unwrap_or_else(|_| String::from("up"));
    assert!(
        matches!(kind.as_str(), "gate" | "up"),
        "APEIRON_GATE_KIND {kind}"
    );
    let passes = env_usize("APEIRON_GATE_PASSES", 2).max(1);
    let (o, n) = (cfg.n_ff, cfg.n_embd);
    println!("RES layers {layers:?} kind {kind} passes {passes} n_ff {o} n_embd {n}");
    let t0 = Instant::now();
    for &l in &layers {
        let (ho, hn, hw) = read_words(&format!("{CACHE}/h{l}.i64"));
        assert_eq!((ho, hn, hw.len()), (n, n, n * n), "layer {l}: cached H");
        let h = unq(&hw);
        let (g, up, _mixer) = gate_up(&engine, &cfg, l);
        let w = if kind == "gate" { g } else { up };
        let (wo, wn, cached) = read_words(&format!("{CACHE}/l{l}.{kind}.i64"));
        assert_eq!(
            (wo, wn, cached.len()),
            (o, n, o * n),
            "layer {l}: cached {kind}"
        );
        let mut what = cached;
        let mut failed = false;
        for p in 1..passes {
            let e: Vec<i64> = what.iter().zip(&w).map(|(a, b)| a - b).collect();
            let before = energy(&h, &e, o, n);
            let r: Vec<i64> = w.iter().zip(&what).map(|(a, b)| a - b).collect();
            let t = Instant::now();
            let fit = match fold_fit(&r, &qv(&h), o, n, None, &production_cfg()) {
                Ok(fit) => fit,
                Err(err) => {
                    println!("RES layer {l:02} kind {kind} pass {p} fit error {err:?}");
                    failed = true;
                    break;
                }
            };
            let wall = t.elapsed().as_secs_f64();
            for (dst, add) in what.iter_mut().zip(&fit.wp) {
                *dst += add;
            }
            let e2: Vec<i64> = what.iter().zip(&w).map(|(a, b)| a - b).collect();
            let after = energy(&h, &e2, o, n);
            let ratio = if before > 0.0 { after / before } else { 0.0 };
            let bits = (o * n) as u64 + fit.bits_r + fit.bits_c + fit.bits_d;
            println!(
                "RES layer {l:02} kind {kind} pass {p} energy before {before:.6e} after {after:.6e} \
                 ratio {ratio:.4} fit {wall:.1} s bits sign/r/c/d {}/{}/{}/{} (+{:.4} b/w)",
                o * n,
                fit.bits_r,
                fit.bits_c,
                fit.bits_d,
                bits as f64 / (o * n) as f64
            );
        }
        if failed {
            continue;
        }
        let path = format!("{CACHE}/l{l}.{kind}-res{passes}.i64");
        write_words(&path, o, n, &what).expect("write residual fit");
        println!("RES wrote {path}");
    }
    println!(
        "RES done layers {} kind {kind} passes {passes} wall {:.1} s",
        layers.len(),
        t0.elapsed().as_secs_f64()
    );
}

/// The per-layer table: off/near/on, flip against `arccos(ρ_H)/π`, the `H` errors and the split.
#[test]
#[ignore = "opens the 3.1 GB F32 teacher and reads the cached fits; run explicitly with --ignored"]
fn gate_code_stats() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = env_layers(cfg.n_layer);
    let calib = env_usize("APEIRON_GATE_CALIB", CALIB);
    let held = env_usize("APEIRON_GATE_HELD", 1);
    let (n, o) = (cfg.n_embd, cfg.n_ff);
    let t = held * n_ctx;
    let mut xs: Vec<Vec<i64>> = vec![vec![0i64; n * t]; cfg.n_layer];
    let mut g8: Vec<Vec<i64>> = vec![vec![0i64; o * t]; cfg.n_layer];
    let t0 = Instant::now();
    for (c, tokens) in chunks.iter().enumerate().skip(calib).take(held) {
        let rows = tapped(&engine, tokens);
        let at = (c - calib) * n_ctx;
        for l in 0..cfg.n_layer {
            for (ti, col) in rows[l][TAP_LN2].iter().enumerate() {
                for (k, &w) in col.iter().enumerate() {
                    xs[l][k * t + at + ti] = w;
                }
            }
            for (ti, col) in rows[l][TAP_GATE].iter().enumerate() {
                g8[l][(at + ti) * o..(at + ti + 1) * o].copy_from_slice(col);
            }
        }
        println!(
            "STATS held chunk {c} at {:.1} s",
            t0.elapsed().as_secs_f64()
        );
    }
    let mut tot = [0.0f64; 6];
    let mut cnt = 0.0f64;
    for &l in &layers {
        let (gw, uw, mixer) = gate_up(&engine, &cfg, l);
        let (_, _, ghw) = read_words(&format!("{CACHE}/l{l}.gate.i64"));
        let (_, _, uhw) = read_words(&format!("{CACHE}/l{l}.up.i64"));
        let (_, _, hw) = read_words(&format!("{CACHE}/h{l}.i64"));
        let (gf, uf, ghat, uhat, hf) = (unq(&gw), unq(&uw), unq(&ghw), unq(&uhw), unq(&hw));
        let ti = Instant::now();
        let xf = unq(&xs[l]);
        let gx = project(&gf, &xf, o, n, t);
        let ghx = project(&ghat, &xf, o, n, t);
        let ux = project(&uf, &xf, o, n, t);
        let uhx = project(&uhat, &xf, o, n, t);
        let fit = layer_fit(&hf, &gf, &ghat, &uf, &uhat, n);
        let (mut off, mut near, mut on, mut flip, mut nz) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut tg2, mut tu2, mut dh2, mut tg2f) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for u in 0..o {
            for s in 0..t {
                let a = gx[u * t + s];
                let b = ghx[u * t + s];
                let c = ux[u * t + s];
                let d = uhx[u * t + s];
                if a < -2.0 {
                    off += 1.0;
                } else if a > 2.0 {
                    on += 1.0;
                } else {
                    near += 1.0;
                }
                let mut flipped = false;
                if a != 0.0 {
                    nz += 1.0;
                    if (a > 0.0) != (b > 0.0) {
                        flip += 1.0;
                        flipped = true;
                    }
                }
                let dh = silu(b) * d - silu(a) * c;
                let tg = dsilu(a) * (b - a) * c;
                let tu = silu(a) * (d - c);
                dh2 += dh * dh;
                tg2 += tg * tg;
                tu2 += tu * tu;
                if flipped {
                    tg2f += tg * tg;
                }
            }
        }
        // The engine's own Q8Act gate against `W_gate·x` from the same stage-3 row.
        let (mut en, mut ed) = (0.0f64, 0.0f64);
        for s in 0..t {
            for u in 0..o {
                let e = from_grid(g8[l][s * o + u]) - gx[u * t + s];
                en += e * e;
                ed += gx[u * t + s] * gx[u * t + s];
            }
        }
        let cells = (o * t) as f64;
        println!(
            "LAYER {l:02} {mixer} off {:.4} near {:.4} on {:.4} flip {:.6} acos {:.6} relg {:.3e} relu {:.3e} share_g {:.4} share_u {:.4} cross {:+.4} flipshare {:.4} q8 {:.2e} proj {:.1} s",
            off / cells,
            near / cells,
            on / cells,
            flip / nz,
            fit.angle,
            fit.rel_g,
            fit.rel_u,
            tg2 / dh2,
            tu2 / dh2,
            1.0 - tg2 / dh2 - tu2 / dh2,
            tg2f / tg2,
            (en / ed).sqrt(),
            ti.elapsed().as_secs_f64()
        );
        for (a, b) in tot.iter_mut().zip([
            off / cells,
            near / cells,
            on / cells,
            flip / nz,
            tg2 / dh2,
            tu2 / dh2,
        ]) {
            *a += b;
        }
        cnt += 1.0;
        if l == 0 {
            // Positive control: `ĝ = g` on the same words and tokens.
            let mut p = 0.0f64;
            let mut pnz = 0.0f64;
            for u in 0..o {
                for s in 0..t {
                    let a = gx[u * t + s];
                    if a != 0.0 {
                        pnz += 1.0;
                        if (a > 0.0) != (a > 0.0) {
                            p += 1.0;
                        }
                    }
                }
            }
            println!(
                "CONTROL positive flip {:.9} over {:.0} nonzero",
                p / pnz,
                pnz
            );
            // Negative control: with `ĝ = g`, negating one row must flip every token
            // (`sign(g·x) != sign(-g·x)` on every column whose `g·x` is nonzero).
            let mut flips = 0.0f64;
            let mut nzt = 0.0f64;
            for s in 0..t {
                let mut acc = 0.0f64;
                for k in 0..n {
                    acc -= gf[k] * xf[k * t + s];
                }
                if gx[s] != 0.0 {
                    nzt += 1.0;
                    if (gx[s] > 0.0) != (acc > 0.0) {
                        flips += 1.0;
                    }
                }
            }
            println!(
                "CONTROL negative unit-0 flip {:.6} over {:.0} tokens",
                flips / nzt,
                nzt
            );
            assert_eq!(p, 0.0, "the positive control must be exactly zero");
        }
    }
    println!(
        "MEAN over {:.0} layers: off {:.4} near {:.4} on {:.4} flip {:.6} share_g {:.4} share_u {:.4}",
        cnt,
        tot[0] / cnt,
        tot[1] / cnt,
        tot[2] / cnt,
        tot[3] / cnt,
        tot[4] / cnt,
        tot[5] / cnt
    );
}

/// The measured layers: every layer with a cached `key` fit.
fn fitted_layers(key: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for l in 0..1024 {
        if Path::new(&format!("{CACHE}/l{l}.{key}.i64")).exists() {
            out.push(l);
        }
    }
    out
}

/// The teacher image with one FFN tensor replaced by an arm's cached words in every listed layer,
/// every other byte the teacher's. The tensor kind is the arm's prefix before the first `-`
/// (`up-spqr10-salience` → `up`, `gate` → `gate`); the fit is read from `{CACHE}/l{l}.{arm}.i64`.
///
/// `O(file bytes)` to read plus `O(layers·n_ff·n_embd)` written; the whole image is held once.
fn student_bytes(arm: &str, layers: &[usize]) -> Vec<u8> {
    let kind = arm.split('-').next().unwrap_or(arm);
    assert!(
        matches!(kind, "gate" | "up" | "down"),
        "APEIRON_GATE_ARM {arm}: tensor kind {kind} is not gate | up | down"
    );
    let mut data = std::fs::read(model_path()).expect("read teacher");
    let gguf = Gguf::open(&model_path()).expect("teacher directory");
    for &l in layers {
        let name = format!("blk.{l}.ffn_{kind}.weight");
        let (_, _, words) = read_words(&format!("{CACHE}/l{l}.{arm}.i64"));
        let t = gguf
            .header
            .tensors
            .iter()
            .find(|t| String::from_utf8_lossy(&t.name) == name)
            .unwrap_or_else(|| panic!("{name}"));
        let off = usize::try_from(gguf.header.data0 + t.offset).expect("offset");
        let bytes = match t.ggml_type {
            quant::GGML_TYPE_F16 => encode_f16(&words).expect("binary16 encode"),
            quant::GGML_TYPE_F32 => encode_f32(&words),
            other => panic!("{name}: ggml type {other}"),
        };
        data[off..off + bytes.len()].copy_from_slice(&bytes);
    }
    data
}

/// Held-out KL of one arm against llama.cpp's own base distribution.
#[test]
#[ignore = "reads the ~5 GB base file and a 3.1 GB student image; run explicitly with --ignored"]
fn gate_code_kl() {
    let arm = std::env::var("APEIRON_GATE_ARM").unwrap_or_else(|_| String::from("teacher"));
    let skip = env_usize("APEIRON_GATE_SKIP", CALIB);
    let calib = env_usize("APEIRON_GATE_CALIB", CALIB);
    assert!(
        skip >= calib,
        "held-out chunk {skip} lies inside calibration 0..{calib}"
    );
    let count = env_usize("APEIRON_GATE_CHUNKS", 10);
    let layers = fitted_layers(if arm == "teacher" { "gate" } else { &arm });
    let t0 = Instant::now();
    let data = if arm == "teacher" {
        std::fs::read(model_path()).expect("read teacher")
    } else {
        student_bytes(&arm, &layers)
    };
    println!(
        "KL arm {arm} layers {} image {:.1} GB in {:.1} s",
        layers.len(),
        data.len() as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let engine = Engine::from_bytes(&format!("student-{arm}"), data).expect("student engine");
    let mut base = KldBase::open(Path::new(&base_path())).expect("base file");
    for _ in 0..skip.min(base.n_chunk()) {
        let _ = base.next_chunk().expect("skip");
    }
    let n = base.n_chunk().saturating_sub(skip).min(count);
    let (mut kl, mut top1, mut nll, mut tokens) = (0.0f64, 0.0f64, 0.0f64, 0u64);
    let mut per = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = base.next_chunk().expect("chunk");
        let t = Instant::now();
        let s = held_out_kldbase_chunk(&engine, &chunk).expect("score chunk");
        let (k, p, m) = (from_grid(s.kl), from_grid(s.top1), from_grid(s.nll_student));
        per.push(k);
        kl += k;
        top1 += p;
        nll += m;
        tokens += s.tokens;
        println!(
            "KL {arm} chunk {} kl {k:.6} top1 {p:.6} nll {m:.4} wall {:.1} s",
            skip + i,
            t.elapsed().as_secs_f64()
        );
    }
    let nf = n as f64;
    let mean = kl / nf;
    let sd = (per.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / (nf - 1.0).max(1.0)).sqrt();
    println!(
        "KL {arm} MEAN chunks {skip}..{} KL {mean:.6} +- {sd:.6} (sd) top1 {:.6} nll {:.4} tokens {tokens} wall {:.1} s",
        skip + n,
        top1 / nf,
        nll / nf,
        t0.elapsed().as_secs_f64()
    );
}

/// The positive control: the teacher's own bytes as the student must give KL exactly zero.
#[test]
#[ignore = "opens two 3.1 GB images; run explicitly with --ignored"]
fn gate_code_control_self() {
    let teacher = Engine::open(&model_path()).expect("teacher");
    let data = std::fs::read(model_path()).expect("read teacher");
    let student = Engine::from_bytes("roundtrip", data).expect("student engine");
    let (chunks, _n_ctx) = base_tokens(&base_path());
    let at = env_usize("APEIRON_GATE_CALIB", CALIB);
    let len = env_usize("APEIRON_GATE_T", 128);
    let seq: &[u32] = chunks[at].get(..len).expect("column prefix");
    let h = held_out(&teacher, &student, &[seq]).expect("held out");
    println!(
        "CONTROL self KL {} top1 {} tokens {}",
        from_grid(h.kl),
        from_grid(h.top1),
        h.tokens
    );
    assert_eq!(h.kl, 0, "identical images must score KL 0");
    assert_eq!(
        h.top1,
        apeiron::engine::ONE,
        "identical images agree on top-1"
    );
}
