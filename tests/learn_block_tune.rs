//! The `NanoQuant` block fit over the whole trunk on the device (`gpu::nn::train::block`), on the
//! student stream: layer `l`'s binary factors for `o`, `gate`, `up` and `down` are trained by Adam
//! through the straight-through sign so that the block output on the student's own input matches
//! the teacher's block output, then the student stream advances through the fitted block
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `tune_fact`). Board `APEIRON-QUALITY-118-W1-T01`.
//!
//! Two phases, each one invocation under the 580 s round bound, resumable through files under
//! `{CACHE}/bt/`:
//! - `APEIRON_BT_PHASE=teacher`: one tapped prefill per calibration chunk (`c6x29`: chunks `0..6`
//!   and `11..40`); writes every layer's teacher block output `t-l{l}.f32` and the trunk input
//!   `x-{tag}-l0.f32`.
//! - `APEIRON_BT_PHASE=tune`: for each layer in `APEIRON_BT_LAYERS` (`a..b`), the o-projection input
//!   on the student stream by `Engine::layer_tap_cols`, the device fit from the `nq{from}` factors,
//!   the export to `l{l}.{kind}-{tag}.{i64,nqf}`, and the next layer's student input
//!   `x-{tag}-l{l+1}.f32`.
//!
//! The judge is `learn_gate_code gate_code_kl` with `APEIRON_GATE_ARM=gate-{tag}+up-{tag}+...`.
#![cfg(feature = "gpu")]

use std::io::{Read as _, Write as _};
use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::gpu::decode::GpuCtx;
use apeiron::gpu::nn::train::block::{BlockShape, BlockTune, HostFactor};
use apeiron::gpu::vulkan::Vk;
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::load::layer_from_engine;
use apeiron::learn::grad::model::Config;
use apeiron::learn::nanoquant::{stored_bits, NqFactor};

const MODEL: &str = "/home/socol/oc-tmp/kld08/qwen08b-f16.gguf";
const BASE: &str = "/home/socol/oc-tmp/kld08/calib.c40.kld";
const CACHE: &str = "/home/socol/oc-tmp/kld08/gatecode-tok";
/// Tap slots (`engine::step::tap`): residual in, o-projection input, FFN norm input and output,
/// layer output.
const HIDDEN_IN: usize = 0;
const ATTN_OUT: usize = 2;
const LN2_OUT: usize = 3;
const H2: usize = 10;
const OUT: usize = 11;
/// Chunks per device step: `T = 4·512` columns.
const PER_STEP: usize = 4;
const KINDS: [&str; 4] = ["o", "gate", "up", "down"];

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn bt_dir() -> String {
    let d = format!("{CACHE}/bt");
    std::fs::create_dir_all(&d).expect("bt dir");
    d
}

/// The `c6x29` calibration chunks: `0..6` and `11..40`.
fn calib_chunks() -> Vec<Vec<u32>> {
    let mut base = KldBase::open(Path::new(BASE)).expect("open base file");
    let all: Vec<Vec<u32>> = (0..base.n_chunk())
        .map(|_| base.next_chunk().expect("chunk").tokens)
        .collect();
    all[..6].iter().chain(&all[11..40]).cloned().collect()
}

fn q32_f32(v: i64) -> f32 {
    (v as f64 / 4_294_967_296.0) as f32
}

fn f32_q32(v: f32) -> i64 {
    (f64::from(v) * 4_294_967_296.0).round() as i64
}

fn write_f32(path: &str, v: &[f32]) {
    let tmp = format!("{path}.tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp).expect("create"));
    for x in v {
        f.write_all(&x.to_le_bytes()).expect("write");
    }
    f.flush().expect("flush");
    drop(f);
    std::fs::rename(&tmp, path).expect("rename");
}

fn read_f32(path: &str) -> Vec<f32> {
    let mut b = Vec::new();
    std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .read_to_end(&mut b)
        .expect("read");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4")))
        .collect()
}

fn write_words(path: &str, o: usize, n: usize, w: &[i64]) {
    let mut b = Vec::with_capacity(16 + 8 * w.len());
    b.extend_from_slice(&(o as u64).to_le_bytes());
    b.extend_from_slice(&(n as u64).to_le_bytes());
    for x in w {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}

/// The `.nqf` layout of `tests/learn_nanoquant.rs`.
fn save_factor(path: &str, f: &NqFactor) {
    let mut b = Vec::new();
    for x in [f.out, f.inn, f.rank] {
        b.extend_from_slice(&(x as u64).to_le_bytes());
    }
    b.extend_from_slice(&f.admm_loss.to_le_bytes());
    for v in [&f.s_post, &f.s_pre, &f.u_latent, &f.v_latent] {
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    b.extend(f.u.iter().chain(&f.v).map(|&s| s.cast_unsigned()));
    std::fs::write(path, b).unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn load_factor(path: &str) -> NqFactor {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let word = |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().expect("word"));
    let (o, n, r) = (word(0) as usize, word(1) as usize, word(2) as usize);
    let mut at = 4;
    let mut take = |len: usize| -> Vec<f64> {
        let v = (at..at + len).map(|i| f64::from_bits(word(i))).collect();
        at += len;
        v
    };
    let (s_post, s_pre, u_latent, v_latent) = (take(o), take(n), take(o * r), take(r * n));
    let signs = &b[8 * (4 + o + n + o * r + r * n)..];
    assert_eq!(signs.len(), o * r + r * n, "{path}: sign bytes");
    NqFactor {
        out: o,
        inn: n,
        rank: r,
        u: signs[..o * r].iter().map(|&s| s.cast_signed()).collect(),
        v: signs[o * r..].iter().map(|&s| s.cast_signed()).collect(),
        s_pre,
        s_post,
        admm_loss: f64::from_bits(word(3)),
        u_latent,
        v_latent,
    }
}

/// The device image of `f`: latents carry the stored sign with the latent's magnitude (floored
/// at `1e-6` so a zero latent still reads its stored sign), `V` transposed to `inn × r`.
fn to_host(f: &NqFactor) -> (HostFactor, usize) {
    let (o, n, r) = (f.out, f.inn, f.rank);
    let mut flips = 0;
    let mut lat = |s: i8, l: f64| {
        if (l < 0.0) != (s < 0) {
            flips += 1;
        }
        (f64::from(s) * l.abs().max(1e-6)) as f32
    };
    let u = (0..o * r).map(|i| lat(f.u[i], f.u_latent[i])).collect();
    let mut vt = vec![0f32; n * r];
    for k in 0..r {
        for c in 0..n {
            vt[c * r + k] = lat(f.v[k * n + c], f.v_latent[k * n + c]);
        }
    }
    let h = HostFactor {
        out: o,
        inn: n,
        rank: r,
        u,
        vt,
        s_post: f.s_post.iter().map(|&v| v as f32).collect(),
        s_pre: f.s_pre.iter().map(|&v| v as f32).collect(),
    };
    (h, flips)
}

/// The fitted factor as an [`NqFactor`]: signs of the latents (0 reads `+1`, as on the device).
fn from_host(h: &HostFactor, admm_loss: f64) -> NqFactor {
    let (o, n, r) = (h.out, h.inn, h.rank);
    let sg = |v: f32| if v < 0.0 { -1i8 } else { 1 };
    let mut v = vec![0i8; r * n];
    let mut v_latent = vec![0f64; r * n];
    for k in 0..r {
        for c in 0..n {
            v[k * n + c] = sg(h.vt[c * r + k]);
            v_latent[k * n + c] = f64::from(h.vt[c * r + k]);
        }
    }
    NqFactor {
        out: o,
        inn: n,
        rank: r,
        u: h.u.iter().map(|&x| sg(x)).collect(),
        v,
        s_pre: h.s_pre.iter().map(|&x| f64::from(x)).collect(),
        s_post: h.s_post.iter().map(|&x| f64::from(x)).collect(),
        admm_loss,
        u_latent: h.u.iter().map(|&x| f64::from(x)).collect(),
        v_latent,
    }
}

/// Phase `teacher`: one tapped prefill per calibration chunk; every layer's block output and the
/// trunk input as f32, chunk-major. Also the FFN norm convention: `LN2_OUT` against
/// `rmsnorm(H2)·g2` on the device's formula, relative error printed per layer.
fn teacher(engine: &Engine, cfg: &Config, tag: &str) {
    let dir = bt_dir();
    let chunks = calib_chunks();
    let (d, nl) = (cfg.n_embd, cfg.n_layer);
    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); nl];
    let mut x0 = Vec::new();
    let mut norm_err = vec![0f64; nl];
    let t0 = Instant::now();
    for (ci, toks) in chunks.iter().enumerate() {
        let mut st = engine.state(toks.len()).expect("state");
        let (_o, rows) = engine
            .prefill_tap_cols(&mut st, toks, KernelMode::Q8Act, true)
            .expect("tapped prefill");
        for (l, lr) in rows.iter().enumerate() {
            for col in &lr[OUT] {
                assert_eq!(col.len(), d);
                outs[l].extend(col.iter().map(|&v| q32_f32(v)));
            }
            if ci == 0 {
                let g2 = layer_from_engine(engine, cfg, l).expect("layer").ffn_norm;
                let mut num = 0f64;
                let mut den = 0f64;
                for (h, n) in lr[H2].iter().zip(&lr[LN2_OUT]).take(64) {
                    let hf: Vec<f64> = h.iter().map(|&v| v as f64 / 4_294_967_296.0).collect();
                    let ms = hf.iter().map(|v| v * v).sum::<f64>() / d as f64;
                    let inv = 1.0 / (ms + 1e-6).sqrt();
                    for c in 0..d {
                        let want = n[c] as f64 / 4_294_967_296.0;
                        let got = hf[c] * inv * (g2[c] as f64 / 4_294_967_296.0);
                        num += (got - want).powi(2);
                        den += want * want;
                    }
                }
                norm_err[l] = (num / den).sqrt();
            }
        }
        for col in &rows[0][HIDDEN_IN] {
            x0.extend(col.iter().map(|&v| q32_f32(v)));
        }
        println!("BT teacher chunk {ci} {:.1} s", t0.elapsed().as_secs_f64());
    }
    println!(
        "BT ffn-norm relative error per layer {:?}",
        norm_err
            .iter()
            .map(|e| format!("{e:.2e}"))
            .collect::<Vec<_>>()
    );
    for (l, o) in outs.iter().enumerate() {
        write_f32(&format!("{dir}/t-l{l}.f32"), o);
    }
    write_f32(&format!("{dir}/x-{tag}-l0.f32"), &x0);
    println!(
        "BT teacher done {} chunks {:.1} s",
        chunks.len(),
        t0.elapsed().as_secs_f64()
    );
}

/// Gather the columns of chunks `ids` (each `n_ctx` columns of width `w`) into one step buffer.
fn gather(src: &[f32], ids: &[usize], n_ctx: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ids.len() * n_ctx * w);
    for &c in ids {
        out.extend_from_slice(&src[c * n_ctx * w..(c + 1) * n_ctx * w]);
    }
    out
}

/// A SplitMix64 permutation of `0..n` for epoch `e`.
fn perm(n: usize, e: u64) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    let mut s = 0x5EED_0000 + e;
    for i in (1..n).rev() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        p.swap(i, (z % (i as u64 + 1)) as usize);
    }
    p
}

/// The step batches over `n` chunks: `PER_STEP` chunks each, the last padded by wrapping.
fn batches(order: &[usize]) -> Vec<Vec<usize>> {
    let n = order.len();
    (0..n.div_ceil(PER_STEP))
        .map(|b| {
            (0..PER_STEP)
                .map(|k| order[(b * PER_STEP + k) % n])
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_lines, reason = "one driver")]
fn tune(engine: &Engine, cfg: &Config, tag: &str) {
    let dir = bt_dir();
    let from = env("APEIRON_BT_FROM", "w24");
    let epochs: u64 = env("APEIRON_BT_EPOCHS", "8").parse().expect("epochs");
    let lr: f64 = env("APEIRON_BT_LR", "1e-5").parse().expect("lr");
    let lr_s: f64 = env("APEIRON_BT_LR_SCALE", "1e-5")
        .parse()
        .expect("lr scale");
    let range = env("APEIRON_BT_LAYERS", &format!("0..{}", cfg.n_layer));
    let (a, b) = range.split_once("..").expect("a..b");
    let (la, lb): (usize, usize) = (a.parse().expect("a"), b.parse().expect("b"));
    let (d, f) = (cfg.n_embd, cfg.n_ff);
    let n_chunks = calib_chunks().len();
    let n_ctx = 512usize;
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
    let mut bt: Option<(BlockShape, BlockTune)> = None;
    for l in la..lb {
        let t_layer = Instant::now();
        let x = read_f32(&format!("{dir}/x-{tag}-l{l}.f32"));
        let yt = read_f32(&format!("{dir}/t-l{l}.f32"));
        assert_eq!(x.len(), n_chunks * n_ctx * d, "x l{l}");
        // The o-projection input on the student stream.
        let mut at = Vec::new();
        let mut aw = 0;
        for c in 0..n_chunks {
            let cols: Vec<Vec<i64>> = x[c * n_ctx * d..(c + 1) * n_ctx * d]
                .chunks_exact(d)
                .map(|r| r.iter().map(|&v| f32_q32(v)).collect())
                .collect();
            let mut st = engine.state(n_ctx).expect("state");
            let rows = engine.layer_tap_cols(&mut st, l, &cols).expect("layer tap");
            for col in &rows[ATTN_OUT] {
                aw = col.len();
                at.extend(col.iter().map(|&v| q32_f32(v)));
            }
        }
        let t_tap = t_layer.elapsed().as_secs_f64();
        let src: Vec<NqFactor> = KINDS
            .iter()
            .map(|k| load_factor(&format!("{CACHE}/l{l}.{k}-nq{from}.nqf")))
            .collect();
        let mut flips = 0;
        let host: Vec<HostFactor> = src
            .iter()
            .map(|nf| {
                let (h, fl) = to_host(nf);
                flips += fl;
                h
            })
            .collect();
        let shape = BlockShape {
            t: PER_STEP * n_ctx,
            d,
            a: aw,
            f,
            ranks: [src[0].rank, src[1].rank, src[2].rank, src[3].rank],
        };
        if bt.as_ref().is_none_or(|(s, _)| *s != shape) {
            bt = Some((shape, unsafe { BlockTune::new(&g, shape, lr, lr_s) }));
        }
        let (_, tuner) = bt.as_mut().expect("tuner");
        let g2: Vec<f32> = layer_from_engine(engine, cfg, l)
            .expect("layer")
            .ffn_norm
            .iter()
            .map(|&v| q32_f32(v))
            .collect();
        let natural: Vec<usize> = (0..n_chunks).collect();
        let eval = |tuner: &mut BlockTune| -> f64 {
            let mut s = 0.0;
            for ids in batches(&natural) {
                unsafe {
                    tuner.batch(
                        &g,
                        &gather(&x, &ids, n_ctx, d),
                        &gather(&at, &ids, n_ctx, aw),
                        &gather(&yt, &ids, n_ctx, d),
                    );
                    s += tuner.run(&g, false);
                }
            }
            s
        };
        unsafe { tuner.load(&g, &host, &g2) };
        if std::env::var("APEIRON_BT_CHECK").is_ok() {
            let ids = [0, 1, 2, 3];
            let (xs, as_, ys) = (
                gather(&x, &ids, n_ctx, d),
                gather(&at, &ids, n_ctx, aw),
                gather(&yt, &ids, n_ctx, d),
            );
            let dev = unsafe {
                tuner.batch(&g, &xs, &as_, &ys);
                tuner.run(&g, false);
                tuner.output(&g)
            };
            let cols = 16;
            let lw = layer_from_engine(engine, cfg, l).expect("layer");
            let by = |names: &[&str]| -> &[i64] {
                &lw.linears
                    .iter()
                    .find(|x| names.contains(&x.name))
                    .expect("linear")
                    .w
            };
            let dense = host_block(
                &xs,
                &as_,
                [
                    by(&["attn_output", "ssm_out"]),
                    by(&["ffn_gate"]),
                    by(&["ffn_up"]),
                    by(&["ffn_down"]),
                ],
                &g2,
                d,
                aw,
                f,
                cols,
            );
            let ws: Vec<Vec<i64>> = src.iter().map(|s| s.weight().expect("weight")).collect();
            let nq = host_block(
                &xs,
                &as_,
                [&ws[0], &ws[1], &ws[2], &ws[3]],
                &g2,
                d,
                aw,
                f,
                cols,
            );
            let devf: Vec<f64> = dev[..cols * d].iter().map(|&v| f64::from(v)).collect();
            let nqf: Vec<f32> = nq.iter().map(|&v| v as f32).collect();
            println!(
                "BT check l{l}: host-dense vs teacher {:.3e}; host-nq vs teacher {:.5}; device vs host-nq {:.3e}; device vs teacher {:.5}",
                rel(&dense, &ys[..cols * d]),
                rel(&nq, &ys[..cols * d]),
                rel(&devf, &nqf),
                rel(&devf, &ys[..cols * d])
            );
            continue;
        }
        let loss0 = eval(tuner);
        let t_fit = Instant::now();
        let mut ep_loss = Vec::new();
        for e in 0..epochs {
            let mut s = 0.0;
            for ids in batches(&perm(n_chunks, e)) {
                unsafe {
                    tuner.batch(
                        &g,
                        &gather(&x, &ids, n_ctx, d),
                        &gather(&at, &ids, n_ctx, aw),
                        &gather(&yt, &ids, n_ctx, d),
                    );
                    s += tuner.run(&g, true);
                }
            }
            ep_loss.push(s);
        }
        let fit_s = t_fit.elapsed().as_secs_f64();
        let loss1 = eval(tuner);
        // Next layer's student input: the fitted block's output, chunk by chunk.
        let mut next = vec![0f32; x.len()];
        for ids in batches(&natural) {
            unsafe {
                tuner.batch(
                    &g,
                    &gather(&x, &ids, n_ctx, d),
                    &gather(&at, &ids, n_ctx, aw),
                    &gather(&yt, &ids, n_ctx, d),
                );
                tuner.run(&g, false);
                let y = tuner.output(&g);
                for (k, &c) in ids.iter().enumerate() {
                    next[c * n_ctx * d..(c + 1) * n_ctx * d]
                        .copy_from_slice(&y[k * n_ctx * d..(k + 1) * n_ctx * d]);
                }
            }
        }
        write_f32(&format!("{dir}/x-{tag}-l{}.f32", l + 1), &next);
        let fitted = unsafe { tuner.factors(&g) };
        let mut bits = 0u64;
        let mut wts = 0usize;
        for ((kind, h), s) in KINDS.iter().zip(&fitted).zip(&src) {
            let nf = from_host(h, s.admm_loss);
            let w = nf.weight().expect("weight");
            write_words(
                &format!("{CACHE}/l{l}.{kind}-{tag}.i64"),
                nf.out,
                nf.inn,
                &w,
            );
            save_factor(&format!("{CACHE}/l{l}.{kind}-{tag}.nqf"), &nf);
            bits += stored_bits(nf.out, nf.inn, nf.rank);
            wts += nf.out * nf.inn;
        }
        let den: f64 = yt.iter().map(|&v| f64::from(v).powi(2)).sum();
        let eps: Vec<String> = ep_loss.iter().map(|v| format!("{:.4}", v / den)).collect();
        println!(
            "BT layer {l:02} b/w {:.4} flips {flips} rel0 {:.5} rel1 {:.5} x{:.2} epochs [{}] tap {t_tap:.1} s fit {fit_s:.1} s total {:.1} s",
            bits as f64 / wts as f64,
            loss0 / den,
            loss1 / den,
            loss0 / loss1,
            eps.join(" "),
            t_layer.elapsed().as_secs_f64()
        );
    }
}

/// The phase named by `APEIRON_BT_PHASE` (`teacher` or `tune`), tag `APEIRON_BT_TAG` (`bt24`).
#[test]
#[ignore = "measurement: the whole-trunk device block fit; reads the teacher model"]
fn block_tune() {
    let tag = env("APEIRON_BT_TAG", "bt24");
    let engine = Engine::open(MODEL).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    match env("APEIRON_BT_PHASE", "tune").as_str() {
        "teacher" => teacher(&engine, &cfg, &tag),
        "tune" => tune(&engine, &cfg, &tag),
        p => panic!("APEIRON_BT_PHASE {p}: teacher | tune"),
    }
}

/// `Engine::layer_tap_cols` fed the teacher's recorded layer input reproduces the tapped prefill's
/// rows for that layer bit for bit, on a DeltaNet layer and a softmax layer.
#[test]
#[ignore = "reads the teacher model"]
fn layer_tap_matches_prefill() {
    let engine = Engine::open(MODEL).expect("open teacher");
    let toks: Vec<u32> = calib_chunks()[0][..64].to_vec();
    let mut st = engine.state(toks.len()).expect("state");
    let (_o, rows) = engine
        .prefill_tap_cols(&mut st, &toks, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    for l in [0, 3, 5] {
        let mut st = engine.state(toks.len()).expect("state");
        let got = engine
            .layer_tap_cols(&mut st, l, &rows[l][HIDDEN_IN])
            .expect("layer tap");
        for s in [ATTN_OUT, H2, LN2_OUT, OUT] {
            assert_eq!(got[s], rows[l][s], "layer {l} stage {s}");
        }
    }
}

/// The block forward in f64 on dense Q32 row-major `out × in` weights, for `cols` columns:
/// `h = x + a·Woᵀ`, `n = rmsnorm(h)·g2`, `y = h + (silu(n·Wgᵀ) ⊙ n·Wuᵀ)·Wdᵀ`.
#[allow(clippy::too_many_arguments, reason = "the block's operands")]
fn host_block(
    x: &[f32],
    a: &[f32],
    w: [&[i64]; 4],
    g2: &[f32],
    d: usize,
    aw: usize,
    f: usize,
    cols: usize,
) -> Vec<f64> {
    let q = |v: i64| v as f64 / 4_294_967_296.0;
    let mv = |m: &[i64], rows: usize, inn: usize, v: &[f64]| -> Vec<f64> {
        (0..rows)
            .map(|r| {
                m[r * inn..(r + 1) * inn]
                    .iter()
                    .zip(v)
                    .map(|(&w, &x)| q(w) * x)
                    .sum()
            })
            .collect()
    };
    let mut y = Vec::with_capacity(cols * d);
    for c in 0..cols {
        let xa: Vec<f64> = a[c * aw..(c + 1) * aw]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let o = mv(w[0], d, aw, &xa);
        let h: Vec<f64> = (0..d).map(|i| f64::from(x[c * d + i]) + o[i]).collect();
        let inv = 1.0 / (h.iter().map(|v| v * v).sum::<f64>() / d as f64 + 1e-6).sqrt();
        let n: Vec<f64> = (0..d).map(|i| h[i] * inv * f64::from(g2[i])).collect();
        let gt = mv(w[1], f, d, &n);
        let up = mv(w[2], f, d, &n);
        let s: Vec<f64> = gt
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let dn = mv(w[3], d, f, &s);
        y.extend((0..d).map(|i| h[i] + dn[i]));
    }
    y
}

fn rel(got: &[f64], want: &[f32]) -> f64 {
    let (mut n, mut dd) = (0.0, 0.0);
    for (g, &w) in got.iter().zip(want) {
        n += (g - f64::from(w)).powi(2);
        dd += f64::from(w).powi(2);
    }
    n / dd
}
