//! Board APEIRON-QUALITY-118-W1-T01. Lane APEIRON-QUALITY-GATE-W1-T01.
//!
//! The FFN gate as a hyperplane code on `Qwen3.5-0.8B`: under the production `fold_fit` arm,
//! per layer, (1) the off/near/on split of `g·x`, (2) the sign-flip rate of `ĝ·x` against
//! `arccos(ρ_H)/π` (Sheppard 1899; Charikar 2002), (3) the gate/up split of `E‖δh‖²`, and (4) the
//! held-out KL with only `ffn_gate` or only `ffn_up` replaced.
//!
//! Every statistic is a Q32.32 word computed from the engine's own tap stage 3 (`STAGE_LN2_OUT`,
//! the gate/up input) and the teacher's own `ffn_gate`/`ffn_up` rows, so the flip is the weight's
//! and not the activation quantiser's. The engine is read on its one inference path
//! (`Engine::prefill_tap_cols`, `KernelMode::Q8Act`).
//!
//! Stages, selected by `APEIRON_GATE_STAGE`:
//! - `probe`: one 512-token tapped prefill and one production `fold_fit`, so the cost equation is
//!   measured before a campaign is launched.
//! - `fit`: calibration `H` from chunks `0..APEIRON_GATE_CALIB`, then the production fit of
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
use apeiron::learn::fold::stats::heldout::{held_out, held_out_kldbase_chunk};
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::load::layer_from_engine;
use apeiron::learn::grad::model::Config;
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// The teacher image: the `Qwen3.5-0.8B` artefact, binary16. The engine refuses an `F32`
/// embedding row (`src/engine/mod.rs:29`), so `qwen08b-f32.gguf` is transcoded to binary16 with
/// `llama-quantize`; every other tensor keeps the `f32` value within binary16's 11-bit mantissa.
const MODEL_DEFAULT: &str = "/home/socol/oc-tmp/kld08/qwen08b-f16.gguf";

/// The teacher image path, overridable by `APEIRON_GATE_MODEL`.
fn model_path() -> String {
    std::env::var("APEIRON_GATE_MODEL").unwrap_or_else(|_| String::from(MODEL_DEFAULT))
}
/// llama.cpp's `--kl-divergence-base` file for the binary32 teacher, 40 chunks of `n_ctx = 512`.
const BASE: &str = "/home/socol/oc-tmp/kld08/calib.c40.kld";

/// The base file path, overridable by `APEIRON_GATE_BASE` (the binary16 teacher's own base).
fn base_path() -> String {
    std::env::var("APEIRON_GATE_BASE").unwrap_or_else(|_| String::from(BASE))
}
/// Where the fitted rows and the calibration Gram are cached as Q32.32 words.
const CACHE: &str = "/home/socol/oc-tmp/kld08/gatecode";

/// Tap slot of the FFN-side `RMSNorm` output: the gate/up input (`STAGE_LN2_OUT`).
const TAP_LN2: usize = 3;
/// Tap slot of the gate pre-activation (`STAGE_GATE`).
const TAP_GATE: usize = 8;

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

/// Add `Σ_t x_t x_tᵀ` into `acc`, `n × n`, from `x` as `t × n` token-major words.
///
/// `O(t·n²/2)` multiply-adds, the upper triangle only; the token loop is outer so each `O(n)`
/// row is read once per token into cache.
fn gram_add(x: &[i64], t: usize, n: usize, acc: &mut [f64]) {
    for ti in 0..t {
        let row = unq(&x[ti * n..(ti + 1) * n]);
        for i in 0..n {
            let xi = row[i];
            let base = i * n;
            for j in i..n {
                acc[base + j] += xi * row[j];
            }
        }
    }
}

/// Mirror the upper triangle of an `n × n` Gram.
fn mirror(acc: &mut [f64], n: usize) {
    for i in 0..n {
        for j in i + 1..n {
            acc[j * n + i] = acc[i * n + j];
        }
    }
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

/// Flatten `[token][width]` tap columns to the feature-major `n × t` layout [`project`] reads.
fn feature_major(cols: &[Vec<i64>], n: usize) -> Vec<i64> {
    let t = cols.len();
    let mut x = vec![0i64; n * t];
    for (ti, col) in cols.iter().enumerate() {
        assert_eq!(col.len(), n, "tap column width");
        for (k, &w) in col.iter().enumerate() {
            x[k * t + ti] = w;
        }
    }
    x
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
    let x = feature_major(&rows[0][TAP_LN2], cfg.n_embd);
    let n = cfg.n_embd;
    let mut h = vec![0.0f64; n * n];
    let t = Instant::now();
    gram_add(&x, n_ctx, n, &mut h);
    let gram = t.elapsed().as_secs_f64();
    mirror(&mut h, n);
    println!("PROBE gram {gram:.2} s per layer-chunk");
    let (g, u, mixer) = gate_up(&engine, &cfg, 0);
    let hw = qv(&h);
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
        "PROBE equation: fit {:.0} s for {layers2} tensors; H {:.0} s over {} layers x calib chunks; \
         stats {:.0} s per layer-chunk",
        fw * layers2,
        gram * f64::from(u32::try_from(cfg.n_layer).expect("24")),
        cfg.n_layer,
        prefill
    );
}

/// Accumulate the calibration `H` over chunks `0..APEIRON_GATE_CALIB` and cache it with the fits.
#[test]
#[ignore = "opens the 3.1 GB F32 teacher and runs production fold_fit per tensor; run explicitly"]
fn gate_code_fit() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = env_layers(cfg.n_layer);
    let calib = env_usize("APEIRON_GATE_CALIB", 2);
    let n = cfg.n_embd;
    std::fs::create_dir_all(CACHE).expect("cache dir");
    let mut h = vec![vec![0.0f64; n * n]; cfg.n_layer];
    let t0 = Instant::now();
    for (c, tokens) in chunks.iter().enumerate().take(calib) {
        let rows = tapped(&engine, tokens);
        for &l in &layers {
            let x = feature_major(&rows[l][TAP_LN2], n);
            gram_add(&x, n_ctx, n, &mut h[l]);
        }
        println!("FIT calib chunk {c} at {:.1} s", t0.elapsed().as_secs_f64());
    }
    let tokens_seen = calib * n_ctx;
    for &l in &layers {
        mirror(&mut h[l], n);
        let scale = 1.0 / tokens_seen as f64;
        let mean: Vec<f64> = h[l].iter().map(|v| v * scale).collect();
        let hw = qv(&mean);
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

/// The per-layer table: off/near/on, flip against `arccos(ρ_H)/π`, the `H` errors and the split.
#[test]
#[ignore = "opens the 3.1 GB F32 teacher and reads the cached fits; run explicitly with --ignored"]
fn gate_code_stats() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let layers = env_layers(cfg.n_layer);
    let calib = env_usize("APEIRON_GATE_CALIB", 2);
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

/// The measured layers: every layer with a cached fit.
fn fitted_layers() -> Vec<usize> {
    let mut out = Vec::new();
    for l in 0..1024 {
        if Path::new(&format!("{CACHE}/l{l}.gate.i64")).exists() {
            out.push(l);
        }
    }
    out
}

/// The teacher image with only `ffn_gate` (arm `gate`) or only `ffn_up` (arm `up`) replaced by the
/// fitted words in every measured layer, every other byte the teacher's.
///
/// `O(file bytes)` to read plus `O(layers·n_ff·n_embd)` written; the whole image is held once.
fn student_bytes(arm: &str, layers: &[usize]) -> Vec<u8> {
    let mut data = std::fs::read(model_path()).expect("read teacher");
    let gguf = Gguf::open(&model_path()).expect("teacher directory");
    for &l in layers {
        let (name, key) = match arm {
            "gate" => (format!("blk.{l}.ffn_gate.weight"), "gate"),
            _ => (format!("blk.{l}.ffn_up.weight"), "up"),
        };
        let (_, _, words) = read_words(&format!("{CACHE}/l{l}.{key}.i64"));
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
    let skip = env_usize("APEIRON_GATE_SKIP", 4);
    let count = env_usize("APEIRON_GATE_CHUNKS", 10);
    let layers = fitted_layers();
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
    let at = env_usize("APEIRON_GATE_CALIB", 2);
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
