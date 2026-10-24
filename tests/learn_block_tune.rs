//! The `NanoQuant` block fit over the whole trunk on the device (`gpu::nn::train::block`), on the
//! student stream: layer `l`'s binary factors for `o`, `gate`, `up` and `down` are trained by Adam
//! through the straight-through sign so that the block output on the student's own input matches
//! the teacher's block output, then the student stream advances through the fitted block
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `tune_fact`). Board `APEIRON-QUALITY-118-W1-T01`.
//!
//! Phase `stream` (`block_tune`) runs both streams on the device (`gpu::nn::train::mixer`): the
//! teacher stream through the dense layer, the student stream through the dense mixer and the
//! fitted block. Layers `APEIRON_BT_LAYERS` (`a..b`) per invocation under the 580 s round bound;
//! the two streams at layer `b` are written to `{CACHE}/bt/s-{tag}-l{b}.{xt,xs}.f32` and the next
//! round resumes from them. Factors go to `l{l}.{kind}-{tag}.{i64,nqf}`.
//!
//! The judge is `learn_gate_code gate_code_kl` with `APEIRON_GATE_ARM=gate-{tag}+up-{tag}+...`;
//! it scores the base file's chunks `8..10`, inside the held-out chunks `6..11` here.
#![cfg(feature = "gpu")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "test geometry: single letters are the tape's symbols (`x`, `h`, `n`, `y`, `a`, `g`); \
              f64 statistics narrow to the device's f32 and token counts widen to f64 for means"
)]

use std::io::{Read as _, Write as _};
use std::path::Path;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::bpe::Bpe;
use apeiron::gpu::decode::GpuCtx;
use apeiron::gpu::nn::train::block::{
    chunk_from, chunk_into, stream_download, stream_upload, BlockShape, BlockTune, HostFactor,
    Stream,
};
use apeiron::gpu::nn::train::mixer::{LayerFwd, LayerHost, Mixer};
use apeiron::gpu::vulkan::Vk;
use apeiron::learn::fold::stats::kldbase::KldBase;
use apeiron::learn::grad::load::load_rows;
use apeiron::learn::grad::model::Config;
use apeiron::learn::nanoquant::{factorize, stored_bits, NqConfig, NqFactor};
use apeiron::learn::solve::q32::from_grid;

const MODEL: &str = "/home/socol/oc-tmp/kld08/qwen08b-f16.gguf";
const BASE: &str = "/home/socol/oc-tmp/kld08/calib.c40.kld";
const CACHE: &str = "/home/socol/oc-tmp/kld08/gatecode-tok";
const TOKENIZER: &str = "/home/socol/oc-tmp/kld08/hf/tokenizer.json";
/// The text behind `BASE` (C4 train, llama.cpp `perplexity`, 40 chunks).
const CALIB_TXT: &str = "/home/socol/oc-tmp/kld/calib.txt";
/// C4 validation shard 0 (`allenai/c4` `en/c4-validation.00000-of-00008`), disjoint from `BASE`.
const C4: &str = "/home/socol/oc-tmp/corpus/c4val/val0.txt";
const N_CTX: usize = 512;
/// Tap slots (`engine::step::tap`): residual in, o-projection input, FFN norm output and input,
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

/// The base file's 40 token chunks.
fn base_chunks() -> Vec<Vec<u32>> {
    base_chunks_at(BASE)
}

/// The `40` token chunks of the base file at `path`.
fn base_chunks_at(path: &str) -> Vec<Vec<u32>> {
    let mut base = KldBase::open(Path::new(path)).expect("open base file");
    (0..base.n_chunk())
        .map(|_| base.next_chunk().expect("chunk").tokens)
        .collect()
}

/// Training and held-out chunks. Held out: the base file's chunks `6..11`. Training
/// (`APEIRON_BT_DATA`): `calib`, the base file's chunks `0..6` and `11..40`; or `c4`, the first
/// `APEIRON_BT_N` chunks of [`C4`].
fn stream_chunks() -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let base = base_chunks();
    let held = base[6..11].to_vec();
    let train = match env("APEIRON_BT_DATA", "c4").as_str() {
        "calib" => base[..6].iter().chain(&base[11..]).cloned().collect(),
        "c4" => {
            let n: usize = env("APEIRON_BT_N", "256").parse().expect("n");
            let c = text_chunks(C4, ((n + 1) * N_CTX * 6).max(8_000_000));
            assert!(c.len() >= n, "{} C4 chunks < {n}", c.len());
            c[..n].to_vec()
        }
        s => panic!("APEIRON_BT_DATA {s}: calib | c4"),
    };
    (train, held)
}

/// The `N_CTX`-token chunks of the first `max_bytes` of `path`, cut at a line end; the token
/// stream is cached as little-endian `u32` under `bt/`.
fn text_chunks(path: &str, max_bytes: usize) -> Vec<Vec<u32>> {
    let stem = Path::new(path).file_stem().expect("stem").to_string_lossy();
    let cache = format!("{}/tok-{stem}-{max_bytes}.u32", bt_dir());
    let toks: Vec<u32> = if Path::new(&cache).exists() {
        std::fs::read(&cache)
            .expect("token cache")
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    } else {
        let raw = std::fs::read(path).expect("text");
        let cut = &raw[..max_bytes.min(raw.len())];
        let end = cut
            .iter()
            .rposition(|&b| b == b'\n')
            .map_or(cut.len(), |i| i + 1);
        let text = std::str::from_utf8(&cut[..end]).expect("utf-8 text");
        let t0 = Instant::now();
        let t = Bpe::from_tokenizer_json(TOKENIZER)
            .expect("tokenizer")
            .encode(text)
            .expect("encode");
        println!(
            "BT tokenized {end} bytes -> {} tokens {:.1} s",
            t.len(),
            t0.elapsed().as_secs_f64()
        );
        let bytes: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&cache, bytes).expect("write token cache");
        t
    };
    toks.chunks_exact(N_CTX).map(<[u32]>::to_vec).collect()
}

/// The trunk input of `chunks`: their `token_embd` rows, chunk-major `T × d`.
fn embed(engine: &Engine, cfg: &Config, chunks: &[Vec<u32>]) -> Vec<f32> {
    let mut x = Vec::with_capacity(chunks.len() * N_CTX * cfg.n_embd);
    for c in chunks {
        let ids: Vec<usize> = c.iter().map(|&t| t as usize).collect();
        let rows = load_rows(engine, "token_embd.weight", cfg.vocab, cfg.n_embd, &ids)
            .expect("embedding rows");
        x.extend(rows.into_iter().map(q32_f32));
    }
    x
}

fn q32_f32(v: i64) -> f32 {
    (v as f64 / 4_294_967_296.0) as f32
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

/// The `out`, `inn` and words of an `.i64` buffer written by [`write_words`].
fn read_words(path: &str) -> (usize, usize, Vec<i64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let o = u64::from_le_bytes(b[0..8].try_into().expect("o")) as usize;
    let n = u64::from_le_bytes(b[8..16].try_into().expect("n")) as usize;
    let w = b[16..]
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("word")))
        .collect();
    (o, n, w)
}

/// The diagonal of the calibration Gram `{CACHE}/g{l}.c6x29.{gram}.i64`: the reference's `i_norm`
/// (`tests/learn_nanoquant.rs:455-458`). Slot `o` reads `o`, `gate`/`up` `gate_up`, `down`
/// `down`.
fn gram_i_norm(l: usize, s: BlockShape, k: usize) -> Vec<f64> {
    let gram = match k {
        0 => "o",
        1 | 2 => "gate_up",
        _ => "down",
    };
    let (o, n, w) = read_words(&format!("{CACHE}/g{l}.c6x29.{gram}.i64"));
    assert_eq!(o, n, "layer {l} slot {k}: Gram shape");
    assert_eq!(n, s.dims(k).1, "layer {l} slot {k}: Gram width");
    (0..n).map(|i| from_grid(w[i * n + i])).collect()
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

/// Gather the columns of chunks `ids` (each `n_ctx` columns of width `w`) into one step buffer.
#[allow(dead_code, reason = "the resident stream copies on the device")]
fn gather(src: &[f32], ids: &[usize], n_ctx: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ids.len() * n_ctx * w);
    for &c in ids {
        out.extend_from_slice(&src[c * n_ctx * w..(c + 1) * n_ctx * w]);
    }
    out
}

/// A `SplitMix64` permutation of `0..n` for epoch `e`.
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

/// One fit step on the resident streams: the step's columns copied device-to-device into the
/// block's `x`, `a` and `yt`, the tape run, and `y` copied into `next`'s chunk regions when
/// `next` is set. Returns the step's summed loss and its `Σ yt²`.
#[allow(
    clippy::too_many_arguments,
    reason = "one step's three streams, the optional next stream, its ids and its train flag"
)]
fn step(
    g: &GpuCtx,
    tuner: &mut BlockTune,
    xs: &Stream,
    at: &Stream,
    yt: &Stream,
    next: Option<&Stream>,
    ids: &[usize],
    train: bool,
) -> (f64, f64) {
    let (bx, ba, byt) = tuner.cols();
    let pre = vec![
        chunk_into(xs, ids, bx),
        chunk_into(at, ids, ba),
        chunk_into(yt, ids, byt),
    ];
    let post = next.map_or_else(Vec::new, |n| vec![chunk_from(n, ids, tuner.out())]);
    unsafe {
        tuner.set_cols(g, pre, post, train);
        let s = tuner.run(g, train);
        (s, tuner.den())
    }
}

/// Write the `n_ctx`-column chunks of `src` (width `w`) back to chunks `ids` of `dst`.
#[allow(dead_code, reason = "the resident stream copies on the device")]
fn scatter(dst: &mut [f32], src: &[f32], ids: &[usize], n_ctx: usize, w: usize) {
    for (k, &c) in ids.iter().enumerate() {
        dst[c * n_ctx * w..(c + 1) * n_ctx * w]
            .copy_from_slice(&src[k * n_ctx * w..(k + 1) * n_ctx * w]);
    }
}

/// Phase `stream`: layers `a..b` of both streams on the device, the block fit per layer.
///
/// Per layer, over `n` chunks: two layer forwards per chunk (`O(n·T·d·(C + a + 3f))` flops),
/// `epochs·n` fit steps, one output pass; host memory `n·T·(3d + a)` words.
#[allow(clippy::too_many_lines, reason = "one driver")]
fn stream(engine: &Engine, cfg: &Config, tag: &str) {
    let dir = bt_dir();
    let from = env("APEIRON_BT_FROM", "w24");
    let epochs: u64 = env("APEIRON_BT_EPOCHS", "8").parse().expect("epochs");
    let lr: f64 = env("APEIRON_BT_LR", "3e-4").parse().expect("lr");
    let lr_s: f64 = env("APEIRON_BT_LR_SCALE", "1e-3")
        .parse()
        .expect("lr scale");
    let lr_d: f64 = env("APEIRON_BT_LR_DENSE", "1e-4")
        .parse()
        .expect("lr dense");
    let range = env("APEIRON_BT_LAYERS", &format!("0..{}", cfg.n_layer));
    let (a, b) = range.split_once("..").expect("a..b");
    let (la, lb): (usize, usize) = (a.parse().expect("a"), b.parse().expect("b"));
    let (d, f) = (cfg.n_embd, cfg.n_ff);
    let t0 = Instant::now();
    let (train_toks, held_toks) = stream_chunks();
    let n_train = train_toks.len();
    let n_all = n_train + held_toks.len();
    let held: Vec<usize> = (n_train..n_all).collect();
    let natural: Vec<usize> = (0..n_all).collect();
    let state = |l: usize, s: &str| format!("{dir}/s-{tag}-l{l}.{s}.f32");
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
    // The streams are device-resident for the whole call: the layer input of the teacher and of
    // the student, the teacher's block output, the student's o-projection input and the student's
    // next layer input. One upload at the start, one download of the two state files at the end,
    // device-to-device chunk copies per step.
    let mut xt = unsafe { Stream::new(&g, N_CTX, d, n_all) };
    let mut xs = unsafe { Stream::new(&g, N_CTX, d, n_all) };
    let mut yt = unsafe { Stream::new(&g, N_CTX, d, n_all) };
    let mut next = unsafe { Stream::new(&g, N_CTX, d, n_all) };
    unsafe {
        if la == 0 {
            let all: Vec<Vec<u32>> = train_toks.into_iter().chain(held_toks).collect();
            let x0 = embed(engine, cfg, &all);
            stream_upload(&g, xt.buf, &x0);
            stream_upload(&g, xs.buf, &x0);
        } else {
            stream_upload(&g, xt.buf, &read_f32(&state(la, "xt")));
            stream_upload(&g, xs.buf, &read_f32(&state(la, "xs")));
        }
    }
    println!(
        "BT stream {n_train} train + {} held chunks, input {:.1} s",
        n_all - n_train,
        t0.elapsed().as_secs_f64()
    );
    let mut fwds: [Option<LayerFwd>; 2] = [None, None];
    let mut bt: Option<(BlockShape, BlockTune)> = None;
    let mut at: Option<Stream> = None;
    for l in la..lb {
        let t_layer = Instant::now();
        let host = LayerHost::from_engine(engine, cfg, l).expect("layer host");
        let kind = usize::from(matches!(host.mixer, Mixer::Softmax { .. }));
        let fwd =
            fwds[kind].get_or_insert_with(|| unsafe { LayerFwd::new(&g, &host, N_CTX, PER_STEP) });
        unsafe { fwd.load(&g, &host) };
        let aw = host.a;
        // The teacher's layer output and the student's mixer output, both through the dense layer:
        // the step's chunk regions copied device-to-device, the tape, the outputs copied back.
        if at.as_ref().is_none_or(|s| s.w != aw) {
            at = Some(unsafe { Stream::new(&g, N_CTX, aw, n_all) });
        }
        let ats = at.as_ref().expect("at stream");
        let (fx, fao, fy) = fwd.handles();
        for ids in batches(&natural) {
            unsafe {
                fwd.run_dev(
                    &g,
                    &[chunk_into(&xt, &ids, fx)],
                    &[chunk_from(&yt, &ids, fy)],
                );
                fwd.run_dev(
                    &g,
                    &[chunk_into(&xs, &ids, fx)],
                    &[chunk_from(ats, &ids, fao)],
                );
            }
        }
        let t_fwd = t_layer.elapsed().as_secs_f64();
        let src: Vec<NqFactor> = KINDS
            .iter()
            .map(|k| load_factor(&format!("{CACHE}/l{l}.{k}-nq{from}.nqf")))
            .collect();
        let mut flips = 0;
        let factors: Vec<HostFactor> = src
            .iter()
            .map(|nf| {
                let (h, fl) = to_host(nf);
                flips += fl;
                h
            })
            .collect();
        let shape = BlockShape {
            t: PER_STEP * N_CTX,
            d,
            a: aw,
            f,
            ranks: [src[0].rank, src[1].rank, src[2].rank, src[3].rank],
        };
        if bt.as_ref().is_none_or(|(s, _)| *s != shape) {
            bt = Some((shape, unsafe { BlockTune::new(&g, shape, lr, lr_s, lr_d) }));
        }
        let (_, tuner) = bt.as_mut().expect("tuner");
        let eval = |tuner: &mut BlockTune, set: &[usize]| -> f64 {
            let (mut s, mut den) = (0.0, 0.0);
            for ids in batches(set) {
                let (a, b) = step(&g, tuner, &xs, ats, &yt, None, &ids, false);
                s += a;
                den += b;
            }
            s / den
        };
        unsafe { tuner.load(&g, &factors, &host.ffn_norm) };
        // The per-linear schedule (`nanoquant/core/compress_model.py:83-101`): every slot starts
        // dense with the teacher's weight, and each linear's factorisation is preceded by a dense
        // tune over the slots still dense (`tune_nonfact`) and followed by a factor tune over the
        // slots factorised so far (`tune_fact`).
        let em = env("APEIRON_BT_SCHED", "joint") == "em";
        if em {
            unsafe {
                tuner.load_dense(&g, 0, &host.wo);
                tuner.load_dense(&g, 1, &host.ffn[0]);
                tuner.load_dense(&g, 2, &host.ffn[1]);
                tuner.load_dense(&g, 3, &host.ffn[2]);
                tuner.set_stage(&g, [false; 4], true);
            }
        }
        let held0 = eval(tuner, &held);
        let t_fit = Instant::now();
        let mut ep_loss = Vec::new();
        // One epoch over `order`: the step's summed squared error over the teacher target's norm.
        let tune = |tuner: &mut BlockTune, order: &[usize]| -> f64 {
            let (mut s, mut den) = (0.0, 0.0);
            for ids in batches(order) {
                let (a, b) = step(&g, tuner, &xs, ats, &yt, None, &ids, true);
                s += a;
                den += b;
            }
            s / den
        };
        if em {
            let dense_ep: u64 = env("APEIRON_BT_EM_DENSE", "1").parse().expect("em dense");
            let fact_ep: u64 = env("APEIRON_BT_EM_FACT", "1").parse().expect("em fact");
            let i_norm: Vec<Vec<f64>> = (0..4).map(|k| gram_i_norm(l, shape, k)).collect();
            let mut done = [false; 4];
            for k in 0..4 {
                unsafe { tuner.set_stage(&g, done, true) };
                let t_stage = Instant::now();
                let mut dense_loss = 0.0;
                for e in 0..dense_ep {
                    dense_loss = tune(tuner, &perm(n_train, e));
                }
                let dense_s = t_stage.elapsed().as_secs_f64();
                let dense_held = eval(tuner, &held);
                ep_loss.push(dense_loss);
                println!(
                    "BT em l{l:02} {} dense held {dense_held:.5} train {dense_loss:.4} admm 0.0 admm 0.0 s stage {dense_s:.1} s",
                    KINDS[k]
                );
                let (o, inn) = shape.dims(k);
                let w: Vec<i64> = unsafe { tuner.weight(&g, k) }
                    .iter()
                    .map(|&v| (f64::from(v) * 4_294_967_296.0).round() as i64)
                    .collect();
                let t_admm = Instant::now();
                let nf = factorize(
                    &w,
                    o,
                    inn,
                    &i_norm[k],
                    &vec![1.0; o],
                    &NqConfig::reference(src[k].rank),
                )
                .expect("factorize");
                let admm_s = t_admm.elapsed().as_secs_f64();
                let (hf, _) = to_host(&nf);
                unsafe { tuner.load_factor(&g, k, &hf) };
                done[k] = true;
                unsafe { tuner.set_stage(&g, done, false) };
                let t_stage = Instant::now();
                let mut fact_loss = 0.0;
                for e in 0..fact_ep {
                    fact_loss = tune(tuner, &perm(n_train, e));
                }
                let fact_s = t_stage.elapsed().as_secs_f64();
                let fact_held = eval(tuner, &held);
                ep_loss.push(fact_loss);
                println!(
                    "BT em l{l:02} {} factor held {fact_held:.5} train {fact_loss:.4} admm {:.4e} admm {admm_s:.1} s stage {fact_s:.1} s",
                    KINDS[k], nf.admm_loss
                );
            }
        } else {
            for e in 0..epochs {
                ep_loss.push(tune(tuner, &perm(n_train, e)));
            }
        }
        let fit_s = t_fit.elapsed().as_secs_f64();
        let held1 = eval(tuner, &held);
        // The student stream advances through the fitted block, `y` copied device-to-device into
        // `next`'s chunk regions.
        for ids in batches(&natural) {
            step(&g, tuner, &xs, ats, &yt, Some(&next), &ids, false);
        }
        let fitted = unsafe { tuner.factors(&g) };
        let (mut bits, mut wts) = (0u64, 0usize);
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
        std::mem::swap(&mut xt, &mut yt);
        std::mem::swap(&mut xs, &mut next);
        let eps: Vec<String> = ep_loss.iter().map(|v| format!("{v:.4}")).collect();
        println!(
            "BT layer {l:02} b/w {:.4} flips {flips} held0 {held0:.5} held1 {held1:.5} x{:.2} epochs [{}] fwd {t_fwd:.1} s fit {fit_s:.1} s total {:.1} s",
            bits as f64 / wts as f64,
            held0 / held1,
            eps.join(" "),
            t_layer.elapsed().as_secs_f64()
        );
    }
    unsafe {
        write_f32(
            &state(lb, "xt"),
            &stream_download(&g, xt.buf, n_all * N_CTX * d),
        );
        write_f32(
            &state(lb, "xs"),
            &stream_download(&g, xs.buf, n_all * N_CTX * d),
        );
    }
    println!("BT round l{la}..l{lb} {:.1} s", t0.elapsed().as_secs_f64());
}

/// The device stream fit (`stream`), tag `APEIRON_BT_TAG` (`bt24`).
#[test]
#[ignore = "measurement: the whole-trunk device block fit; reads the teacher model"]
fn block_tune() {
    let tag = env("APEIRON_BT_TAG", "bt24");
    let engine = Engine::open(MODEL).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    stream(&engine, &cfg, &tag);
}

/// The crate's BPE on the base file's text reproduces the base file's token chunks (llama.cpp's
/// tokenizer), so C4 chunks tokenized here are the judge's token distribution.
#[test]
#[ignore = "reads the tokenizer and the base file"]
fn bpe_matches_base() {
    let text = std::fs::read_to_string(CALIB_TXT).expect("calib text");
    let toks = Bpe::from_tokenizer_json(TOKENIZER)
        .expect("tokenizer")
        .encode(&text)
        .expect("encode");
    let base = base_chunks();
    let (mut same, mut first_diff) = (0usize, None);
    for (i, &t) in base.iter().flatten().enumerate() {
        if toks.get(i) == Some(&t) {
            same += 1;
        } else if first_diff.is_none() {
            first_diff = Some(i);
        }
    }
    let n = base.len() * N_CTX;
    println!("BPE {same}/{n} tokens equal; first difference at {first_diff:?}");
    assert_eq!(same, n, "first difference at {first_diff:?}");
}

/// `Engine::layer_tap_cols` fed the teacher's recorded layer input reproduces the tapped prefill's
/// rows for that layer bit for bit, on a `DeltaNet` layer and a softmax layer.
#[test]
#[ignore = "reads the teacher model"]
fn layer_tap_matches_prefill() {
    let engine = Engine::open(MODEL).expect("open teacher");
    let toks: Vec<u32> = base_chunks()[0][..64].to_vec();
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

/// The device layer forward (`gpu::nn::train::mixer`) reproduces the CPU tap's mixer output and
/// layer output on a `DeltaNet` layer and a softmax layer, to the `Q8Act` schedule's relative
/// error; prints both walls.
///
/// The model and base file are `APEIRON_BT_MODEL` / `APEIRON_BT_BASE` (default [`MODEL`] /
/// [`BASE`]), the layers `APEIRON_BT_LF_LAYERS` (default `0,3`), so the same assertion covers the
/// ungated softmax face (`APEIRON_BT_MODEL=<Qwen3>`, layers `0,3`).
#[test]
#[ignore = "reads the teacher model; needs a Vulkan device"]
fn layer_fwd_matches_cpu() {
    let model = env("APEIRON_BT_MODEL", MODEL);
    let base = env("APEIRON_BT_BASE", BASE);
    let layers: Vec<usize> = env("APEIRON_BT_LF_LAYERS", "0,3")
        .split(',')
        .map(|s| s.trim().parse().expect("LF layer"))
        .collect();
    let engine = Engine::open(&model).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let toks: Vec<u32> = base_chunks_at(&base)[0].clone();
    let n_ctx = toks.len();
    let mut st = engine.state(n_ctx).expect("state");
    let (_o, rows) = engine
        .prefill_tap_cols(&mut st, &toks, KernelMode::Q8Act, true)
        .expect("tapped prefill");
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
    let flat = |r: &[Vec<i64>]| -> Vec<f32> { r.iter().flatten().map(|&v| q32_f32(v)).collect() };
    let rel32 = |got: &[f32], want: &[f32]| -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&a, &b) in got.iter().zip(want) {
            num += f64::from(a - b).powi(2);
            den += f64::from(b).powi(2);
        }
        (num / den).sqrt()
    };
    for l in layers {
        let host = LayerHost::from_engine(&engine, &cfg, l).expect("layer host");
        let fwd = unsafe { LayerFwd::new(&g, &host, n_ctx, 1) };
        let x = flat(&rows[l][HIDDEN_IN]);
        let t0 = Instant::now();
        let (ao, y) = unsafe { fwd.run(&g, &x) };
        let first = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..5 {
            let _ = unsafe { fwd.run(&g, &x) };
        }
        let gpu = t1.elapsed().as_secs_f64() / 5.0;
        let mut st = engine.state(n_ctx).expect("state");
        let t2 = Instant::now();
        let _ = engine
            .layer_tap_cols(&mut st, l, &rows[l][HIDDEN_IN])
            .expect("layer tap");
        let cpu = t2.elapsed().as_secs_f64();
        let (ea, ey) = (
            rel32(&ao, &flat(&rows[l][ATTN_OUT])),
            rel32(&y, &flat(&rows[l][OUT])),
        );
        println!(
            "LF l{l} rel attn_out {ea:.3e} out {ey:.3e}; gpu {gpu:.4} s (first {first:.4}); cpu {cpu:.3} s; {n_ctx} cols"
        );
        assert!(ea < 3e-2 && ey < 3e-2, "layer {l}: {ea:.3e} {ey:.3e}");
    }
}
