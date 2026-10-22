//! Board . Lane : where `ffn_down`'s
//! held-out KL lives on `Qwen3.5-0.8B`.
//!
//! Two measurements, both on the engine's one inference path (`KernelMode::Q8Act`):
//!
//! - `down_one_layer_kl`: the teacher image with `blk.<l>.ffn_down.weight` replaced by the cached
//!   production fit of ONE layer, held-out KL against llama.cpp's own base distribution on chunk 2.
//!   One image per listed layer, scored in turn, so the per-layer KL costs are additive by
//!   construction (`APEIRON_DOWN_LAYERS`, `APEIRON_DOWN_SKIP`, `APEIRON_DOWN_CHUNKS`).
//! - `down_unit_energy`: the exact decomposition of the hidden-layer error energy
//!   `tr(E H Eᵀ) = Σ_i c_i` with `E = D̂ − D` (`n_embd × n_ff`) and `H` the held-out down Gram
//!   `E[h hᵀ]` (`n_ff × n_ff`, `h = silu(g·x)⊙(u·x)` from tap stages 8 and 9).
//!
//! Token disjointness: `H` for the energy is the mean Gram over chunk 2 of the base file's token
//! block — the same held-out chunk the KL is scored on, and no calibration chunk. The cached fits
//! were fitted on chunks `0..2` (`gate_code_fit_down`, `tests/learn_gate_code.rs`).
//!
//! Grade `measured`: every number is the print of a command that ran.
//!
//! Controls: an image with no tensor replaced (KL 0 and top-1 1 exactly); `E = 0` (zero energy and
//! every `c_i = 0`); `E` one column of `D` zeroed (`c_i = 0` off that unit and
//! `c_k = H_kk ‖D[:,k]‖²`).

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
              single-letter names are the geometry's symbols (`D`, `E`, `H`, `g`, `u`, `h`, `c`) \
              and every loop is one linear composition of the stage it measures"
)]

use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::gguf::Gguf;
use apeiron::format::quant;
use apeiron::learn::fold::run::convert::{encode_f16, encode_f32};
use apeiron::learn::fold::stats::heldout::{held_out, held_out_kldbase_chunk};
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::load::layer_from_engine;
use apeiron::learn::grad::model::Config;
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

/// The teacher image: the `Qwen3.5-0.8B` artefact, binary16 (`llama-quantize f32 f16`), the image
/// the cached fits were scored against.
const MODEL_DEFAULT: &str = "~/kld08/qwen08b-f16.gguf";

/// The teacher image path, overridable by `APEIRON_DOWN_MODEL`.
fn model_path() -> String {
    std::env::var("APEIRON_DOWN_MODEL").unwrap_or_else(|_| String::from(MODEL_DEFAULT))
}

/// llama.cpp's `--kl-divergence-base` file for the binary16 teacher, 40 chunks of `n_ctx = 512`.
const BASE: &str = "~/kld08/calib.c40.kld";

/// The base file path, overridable by `APEIRON_DOWN_BASE`.
fn base_path() -> String {
    std::env::var("APEIRON_DOWN_BASE").unwrap_or_else(|_| String::from(BASE))
}

/// Where the fitted rows and the calibration Grams are cached as Q32.32 words.
const CACHE: &str = "~/kld08/gatecode";

/// Tap slot of the gate pre-activation (`STAGE_GATE`), `n_ff` wide.
const TAP_GATE: usize = 8;
/// Tap slot of the up pre-activation (`STAGE_UP`), `n_ff` wide.
const TAP_UP: usize = 9;

/// An environment knob with a default.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// The layer list from `APEIRON_DOWN_LAYERS`, or every layer with a cached `ffn_down` fit.
fn down_layers() -> Vec<usize> {
    match std::env::var("APEIRON_DOWN_LAYERS") {
        Ok(s) if !s.is_empty() => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        _ => fitted_layers("down"),
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

/// Read the `(o, n)`-headed little-endian `i64` payload back as `(o, n, words)`.
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

/// The logistic function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `silu(x) = x·σ(x)`.
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// The down input `h = silu(g·x)⊙(u·x)` per column, from the tapped gate and up pre-activations,
/// as Q32.32 words in the feature-major `n_ff × t` layout. `O(t·n_ff)`.
fn down_input(gate: &[Vec<i64>], up: &[Vec<i64>], f: usize) -> Vec<i64> {
    let t = gate.len();
    let mut out = vec![0i64; f * t];
    for (c, (gc, uc)) in gate.iter().zip(up).enumerate() {
        assert_eq!(gc.len(), f, "stage 8 width");
        assert_eq!(uc.len(), f, "stage 9 width");
        let g = unq(gc);
        let u = unq(uc);
        let h: Vec<f64> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        for (j, w) in qv(&h).into_iter().enumerate() {
            out[j * t + c] = w;
        }
    }
    out
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

/// Add `Σ_r x_r x_rᵀ` into `acc`, `n × n`, from `x` as `rows × n` row-major words.
///
/// `O(rows·n²/2)` multiply-adds, the upper triangle only; the row loop is outer so each `O(n)`
/// row is read once per row index.
fn gram_add(x: &[i64], rows: usize, n: usize, acc: &mut [f64]) {
    for r in 0..rows {
        let row = unq(&x[r * n..(r + 1) * n]);
        for i in 0..n {
            let xi = row[i];
            let base = i * n;
            for j in i..n {
                acc[base + j] += xi * row[j];
            }
        }
    }
}

/// Mirror the upper triangle of an `n × n` Gram. `O(n²)` copies.
fn mirror(acc: &mut [f64], n: usize) {
    for i in 0..n {
        for j in i + 1..n {
            acc[j * n + i] = acc[i * n + j];
        }
    }
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

/// The teacher image with `blk.<l>.ffn_down.weight` replaced by the cached fitted words for the
/// listed layers, every other byte the teacher's.
///
/// `O(file bytes)` to read plus `O(layers·n_ff·n_embd)` written; the whole image is held once.
fn student_bytes(layers: &[usize]) -> Vec<u8> {
    let mut data = std::fs::read(model_path()).expect("read teacher");
    let gguf = Gguf::open(&model_path()).expect("teacher directory");
    for &l in layers {
        let name = format!("blk.{l}.ffn_down.weight");
        let (_, _, words) = read_words(&format!("{CACHE}/l{l}.down.i64"));
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

/// Score the held-out chunks `skip..skip+count` of the base file on `engine`, `O(cost(prefill) +
/// rows·cost(head))` each; returns `(kl, top1)` means and the per-chunk rows.
fn held_out_kl(
    engine: &Engine,
    path: &str,
    skip: usize,
    count: usize,
) -> (f64, f64, Vec<(usize, f64, f64)>) {
    let mut base = KldBase::open(Path::new(path)).expect("base file");
    for _ in 0..skip.min(base.n_chunk()) {
        let _ = base.next_chunk().expect("skip");
    }
    let n = base.n_chunk().saturating_sub(skip).min(count).max(1);
    let (mut kl, mut top1) = (0.0f64, 0.0f64);
    let mut per = Vec::with_capacity(n);
    for i in 0..n {
        let chunk = base.next_chunk().expect("chunk");
        let t = Instant::now();
        let s = held_out_kldbase_chunk(engine, &chunk).expect("score chunk");
        let (k, p) = (from_grid(s.kl), from_grid(s.top1));
        println!(
            "  chunk {} kl {k:.6} top1 {p:.6} nll {:.4} tokens {} wall {:.1} s",
            skip + i,
            from_grid(s.nll_student),
            s.tokens,
            t.elapsed().as_secs_f64()
        );
        kl += k;
        top1 += p;
        per.push((skip + i, k, p));
    }
    (kl / n as f64, top1 / n as f64, per)
}

/// **Measurement 1.** Held-out KL of the teacher with only `ffn_down` replaced in one layer at a
/// time, chunk 2 by default.
///
/// `APEIRON_DOWN_LAYERS` names the layers (default: every cached fit); one image is built and one
/// engine opened per layer, so the wall is additive and each invocation stays inside its budget.
#[test]
#[ignore = "reads the 1.6 GB teacher image per layer and scores ~174 s of held-out KL each"]
fn down_one_layer_kl() {
    let layers = down_layers();
    let skip = env_usize("APEIRON_DOWN_SKIP", 2);
    let count = env_usize("APEIRON_DOWN_CHUNKS", 1);
    println!(
        "DOWNKL layers {layers:?} chunks {skip}..{} (all-seven reference 0.5493 at 37c7e59c)",
        skip + count
    );
    let mut per = Vec::new();
    let mut sum = 0.0f64;
    for &l in &layers {
        let t0 = Instant::now();
        let data = student_bytes(&[l]);
        let engine = Engine::from_bytes(&format!("down-l{l}"), data).expect("student engine");
        println!(
            "DOWNKL layer {l} image ready at {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        let (kl, top1, _rows) = held_out_kl(&engine, &base_path(), skip, count);
        println!(
            "DOWNKL layer {l:02} kl {kl:.6} top1 {top1:.6} wall {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        per.push((l, kl, top1));
        sum += kl;
    }
    for (l, kl, top1) in &per {
        println!(
            "DOWNKL share layer {l:02} kl {kl:.6} top1 {top1:.6} share {:.4}",
            kl / sum
        );
    }
    println!(
        "DOWNKL sum over {} layers {sum:.6} mean {:.6}",
        per.len(),
        sum / per.len() as f64
    );
}

/// **Control, positive.** The teacher's own bytes as the student, scored against the teacher's own
/// engine: held-out KL exactly 0 and top-1 exactly 1. The image is unmodified, so the only
/// difference the scorer can see is none.
///
/// The engine's floor against llama.cpp's own base file is a separate, larger number — measured
/// `0.000735` against `calib.c40.f16.kld` and `0.000907` against `calib.c40.kld` on chunk 2 — and
/// is the numerical distance between the Q32.32 path and llama.cpp's, not between two of our
/// images. It is why the base-file KL of an unmodified image is not zero.
#[test]
#[ignore = "opens two 1.6 GB teacher images; run explicitly with --ignored"]
fn down_control_self() {
    let teacher = Engine::open(&model_path()).expect("teacher");
    let data = std::fs::read(model_path()).expect("read teacher");
    let student = Engine::from_bytes("down-roundtrip", data).expect("student engine");
    let (chunks, _n_ctx) = base_tokens(&base_path());
    let at = env_usize("APEIRON_DOWN_SKIP", 2);
    let len = env_usize("APEIRON_DOWN_T", 128);
    let seq: &[u32] = chunks[at].get(..len).expect("column prefix");
    let t = Instant::now();
    let h = held_out(&teacher, &student, &[seq]).expect("held out");
    println!(
        "CONTROL positive kl {} top1 {} tokens {} wall {:.1} s",
        from_grid(h.kl),
        from_grid(h.top1),
        h.tokens,
        t.elapsed().as_secs_f64()
    );
    assert_eq!(h.kl, 0, "identical images must score KL 0");
    assert_eq!(
        h.top1,
        apeiron::engine::ONE,
        "identical images agree on top-1"
    );
}

/// The exact energy of `E` (`o × n`, Q32.32 words) under the `n × n` Gram `h`:
/// `(tr(E H Eᵀ), Σ_i H_ii‖E[:,i]‖², c)` with `c_i = Σ_j (EᵀE)_ij H_ji`.
///
/// `O(o·n²/2 + n²)` multiply-adds: one `o × n × n` symmetric Gram `EᵀE`, then the `n²` contraction
/// with `h`. Space `O(n²)` for the Gram.
fn energy_of(e: &[i64], h: &[f64], o: usize, n: usize) -> (f64, f64, Vec<f64>) {
    let mut g = vec![0.0f64; n * n];
    gram_add(e, o, n, &mut g);
    mirror(&mut g, n);
    let mut c = vec![0.0f64; n];
    for i in 0..n {
        let gi = &g[i * n..(i + 1) * n];
        let mut s = 0.0f64;
        for j in 0..n {
            s += gi[j] * h[j * n + i];
        }
        c[i] = s;
    }
    let diag: f64 = (0..n).map(|i| h[i * n + i] * g[i * n + i]).sum();
    (c.iter().sum(), diag, c)
}

/// The share of `v`'s sum carried by its `frac` largest entries (at least one), descending sort so
/// `O(n log n)`; `0` for a zero or non-finite sum.
fn top_share(v: &[f64], frac: f64) -> f64 {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
    let k = (((v.len() as f64) * frac).ceil() as usize).clamp(1, v.len());
    let total: f64 = s.iter().sum();
    if total == 0.0 || !total.is_finite() {
        return 0.0;
    }
    s[..k].iter().sum::<f64>() / total
}

/// The median of `v`, `O(n log n)`; the mean of the two middle entries for even `n`.
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// **Measurement 2.** The per-unit decomposition of `tr(E H Eᵀ)` for each layer's `ffn_down` fit,
/// with `H` the held-out down Gram of chunk `APEIRON_DOWN_HELD` (default 2) — the held-out chunk,
/// not the calibration chunks the fit used.
#[test]
#[ignore = "opens the 1.6 GB teacher and rebuilds the n_ff x n_ff down Gram per layer"]
fn down_unit_energy() {
    let engine = Engine::open(&model_path()).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (chunks, n_ctx) = base_tokens(&base_path());
    let held = env_usize("APEIRON_DOWN_HELD", 2);
    let layers = down_layers();
    let (o, n) = (cfg.n_embd, cfg.n_ff);
    println!(
        "DOWNEN layers {layers:?} held chunk {held} tokens {n_ctx} n_embd {o} n_ff {n} \
         (H = held-out chunk {held} mean Gram, no calibration chunk)"
    );
    let t0 = Instant::now();
    let rows = tapped(&engine, &chunks[held]);
    println!("DOWNEN tap {:.1} s", t0.elapsed().as_secs_f64());
    let mut control_done = false;
    let mut best_c = (0usize, 0.0f64);
    for &l in &layers {
        let t = Instant::now();
        let x = down_input(&rows[l][TAP_GATE], &rows[l][TAP_UP], n);
        let mut h = vec![0.0f64; n * n];
        gram_add(&x, n_ctx, n, &mut h);
        mirror(&mut h, n);
        let scale = 1.0 / n_ctx as f64;
        for v in &mut h {
            *v *= scale;
        }
        let (d, mixer) = ffn_down(&engine, &cfg, l);
        let (ro, co, dhw) = read_words(&format!("{CACHE}/l{l}.down.i64"));
        assert_eq!((ro, co), (o, n), "layer {l}: cached down shape");
        let e: Vec<i64> = dhw.iter().zip(&d).map(|(a, b)| a - b).collect();
        let (total, diag, c) = energy_of(&e, &h, o, n);
        let mut gd = vec![0.0f64; n * n];
        gram_add(&d, o, n, &mut gd);
        mirror(&mut gd, n);
        let mut base = 0.0f64;
        for i in 0..n {
            let gi = &gd[i * n..(i + 1) * n];
            for j in 0..n {
                base += gi[j] * h[j * n + i];
            }
        }
        let hi: Vec<f64> = (0..n).map(|i| h[i * n + i]).collect();
        let sum_h: f64 = hi.iter().sum();
        let sum_h2: f64 = hi.iter().map(|v| v * v).sum();
        let pr = sum_h * sum_h / ((n as f64) * sum_h2);
        let absfrac: f64 = c.iter().map(|v| v.abs()).sum::<f64>() / total;
        let hmin = hi.iter().copied().fold(f64::INFINITY, f64::min);
        let top_c = (0..n)
            .max_by(|&a, &b| c[a].partial_cmp(&c[b]).expect("finite"))
            .expect("n>0");
        if c[top_c] / total > best_c.1 {
            best_c = (l, c[top_c] / total);
        }
        println!(
            "LAYER {l:02} {mixer} total {total:.6e} rel {:.6e} diag {diag:.6e} diagshare {:.4} \
             c_top1 {:.4} c_top10 {:.4} h_top1 {:.4} h_top10 {:.4} pr {:.4} hmax {:.4e} hmed {:.4e} \
             hmin {:.4e} absfrac {:.4} topc_unit {top_c} topc_share {:.4} wall {:.1} s",
            total / base,
            diag / total,
            top_share(&c, 0.01),
            top_share(&c, 0.10),
            top_share(&hi, 0.01),
            top_share(&hi, 0.10),
            pr,
            hi.iter().copied().fold(0.0f64, f64::max),
            median(&hi),
            hmin,
            absfrac,
            c[top_c] / total,
            t.elapsed().as_secs_f64()
        );
        if !control_done {
            control_done = true;
            // Negative: `D̂ = D`, so `E = 0`: zero energy and every `c_i = 0`.
            let (zt, zd, zc) = energy_of(&vec![0i64; o * n], &h, o, n);
            println!(
                "CONTROL negative total {zt:.1e} diag {zd:.1e} max|c| {:.1e}",
                zc.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
            );
            assert_eq!(zt, 0.0, "E = 0 must give zero energy");
            assert_eq!(zd, 0.0, "E = 0 must give zero diagonal energy");
            assert!(
                zc.iter().all(|v| *v == 0.0),
                "E = 0 must give every c_i = 0"
            );
            // Attribution: `D̂ = D` with unit `k`'s column zeroed, so `E` is `-D[:,k]` in column `k`.
            let k = n / 2;
            let mut ek = vec![0i64; o * n];
            for u in 0..o {
                ek[u * n + k] = -d[u * n + k];
            }
            let (kt, _kd, kc) = energy_of(&ek, &h, o, n);
            let want = h[k * n + k] * gd[k * n + k];
            let off: f64 = kc
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != k)
                .map(|(_, v)| v.abs())
                .fold(0.0f64, f64::max);
            let rel = (kc[k] - want).abs() / want.abs().max(f64::MIN_POSITIVE);
            println!(
                "CONTROL attribution unit {k} c_k {:.6e} H_kk|D_k|^2 {want:.6e} rel {rel:.2e} \
                 total {kt:.6e} max|c_i| i!=k {off:.1e}",
                kc[k]
            );
            assert_eq!(kt, kc[k], "one column carries the whole energy");
            assert!(off <= 1e-9 * want.abs(), "c_i = 0 off unit k");
            assert!(rel <= 1e-9, "c_k = H_kk ‖D[:,k]‖^2");
        }
    }
    println!(
        "DOWNEN done {} layers, largest single-unit share {:.4} at layer {:02}, {:.1} s",
        layers.len(),
        best_c.1,
        best_c.0,
        t0.elapsed().as_secs_f64()
    );
}
