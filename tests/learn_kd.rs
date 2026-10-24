//! The KD step of `NanoQuant`'s global scale fit on the device, against the reference's own step 0
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `src/nanoquant/core/compress_model.py:125-297`; the trace
//! is `~/nq-ref/kd/trace/`).
//!
//! Three tests, all `#[ignore]` because each opens the device, the teacher and the trace:
//!
//! - `kd_memory`: the device bytes one layer's tape holds at `T = 2048` beside the free device
//!   memory, and the resident/swapped choice those numbers make.
//! - `kd_step0_golden`: the reference's pre-KD student at step 0 on sample `perm8[0][0] = 48`, its
//!   loss, `h_l`/`dh_l` and every scale gradient against the trace, and `p1` against `p0` plus our
//!   update.
//! - `kd_wall`: steps 0..5 on our student, the median wall per step split into forward, head,
//!   backward and optimiser.
#![cfg(feature = "gpu")]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "test geometry: single letters are the tape's symbols; ids, widths and lengths index \
              u32 dispatches and f64 statistics narrow to the device's f32"
)]

use apeiron::engine::loader::Engine;
use apeiron::gpu::decode::GpuCtx;
use apeiron::gpu::nn::train::block::stream_buffer;
use apeiron::gpu::nn::train::block::stream_upload;
use apeiron::gpu::nn::train::chain::ChainLayer;
use apeiron::gpu::nn::train::kd::{
    cosine, free_mib, read_f32, rel_l2, shapes_of, KdShapes, KdTrain, LayerPars, ScaleAdam,
    BLOCK_KEYS, FRONT_KEYS, KEYS,
};
use apeiron::gpu::nn::train::mixer::{LayerHost, Mixer};
use apeiron::gpu::nn::{cosine_lr, AdamHyper};
use apeiron::gpu::vulkan::Vk;
use apeiron::learn::grad::load::load_rows;
use apeiron::learn::grad::model::Config;

/// The teacher, `Qwen3-0.6B-Base` in f16.
const MODEL: &str = "~/nq-ref/q3-0p6b-f16.gguf";
/// The reference's dumps: `ranks.json`, `calib_ids.i32`, `perm.json`.
const DUMP: &str = "~/nq-ref/dump";
/// The block loop's input streams: `s-0.xs.f32` (the embedding) and `s-28.xt.f32` (the teacher's
/// final-norm input), 128 samples of 2048 tokens each.
const LOOP: &str = "~/nq-ref/loop/loop-nctx2048";
/// The reference's KD state: `ref-blocks/l{0..27}` (its pre-KD student) and `trace/` (step 0).
const KD: &str = "~/nq-ref/kd";

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn envu(key: &str, default: usize) -> usize {
    env(key, &default.to_string()).parse().unwrap_or(default)
}

/// The step-0 golden directory: the bf16 `trace/` by default, or `KD_TRACE` (e.g. the f32 rerun
/// `kd/llvm/rt/out-f32`, the precision floor).
/// The golden trace: the reference driver run in f32 (`kd/llvm/rt/trace_f32.py`). Its bf16 run
/// (`kd/trace`, `kd/llvm/rt/out-trace`) is the measured floor below, not the want side.
fn trace() -> String {
    env("KD_TRACE", &format!("{KD}/llvm/rt/out-f32"))
}

/// The reference's own distances, in this test's metrics (`rel_l2`, `cosine`), from
/// `~/nq-ref/kd/receipts/` (`MANIFEST.txt` sha256; `floor_l2.py`,
/// `floor_sign.py`): its bf16 run against its f32 run (`floor-l2.txt`, `floor-sign.txt` second
/// block) and its bf16 run against itself (first block). The bf16 is the reference's SASS:
/// `HMMA.16816.F32.BF16` GEMMs, the fmha backward's `HMMA.1688.F32.BF16`, elementwise f32 with a
/// bf16 RNE store (`~/nq-ref/kd/llvm/MANIFEST.txt`). The port meets the f32 trace
/// when it sits no farther from it than the reference's bf16 run does, widened by the run-to-run
/// spread: relative errors add (triangle inequality), cosines add as angles.
struct Floor {
    /// `(bf16 vs f32, run to run)`.
    prec: f64,
    spread: f64,
}

impl Floor {
    const LOSS_REL: Floor = Floor {
        prec: 2.2653e-3,
        spread: 0.0,
    };
    /// `h.l28`, the worst of the checked layers.
    const H_REL: Floor = Floor {
        prec: 1.0800e-2,
        spread: 0.0,
    };
    /// `dh.l1`, the worst checked cosine; run to run `dh.l0`.
    const DH_COS: Floor = Floor {
        prec: 0.995_203,
        spread: 0.999_855,
    };
    /// `g.1.self_attn.v_proj.scale_pre`; run to run `g.1.mlp.up_proj.scale_pre`.
    const G_COS: Floor = Floor {
        prec: 0.985_566,
        spread: 0.999_529,
    };
    /// `sign(g)` disagreement where `|g| > 0.1 max|g|`: 2 of 33624, run to run 0 of 33411.
    const SIGN_BIG: Floor = Floor {
        prec: 5.948e-5,
        spread: 0.0,
    };
    /// Below that: 12162 of 597160, run to run 1400 of 597373.
    const SIGN_SMALL: Floor = Floor {
        prec: 2.037e-2,
        spread: 2.344e-3,
    };

    fn rel(&self) -> f64 {
        self.prec + self.spread
    }

    fn cos(&self) -> f64 {
        (self.prec.acos() + self.spread.acos()).cos()
    }
}

/// `ranks.json`'s `{"<layer>.<key>": rank}` as `[layer][slot]`, slots in [`KEYS`] order
/// (`tests/learn_block_tune.rs:159-190`).
fn ranks(dump: &str) -> Vec<[usize; 7]> {
    let s = std::fs::read_to_string(format!("{dump}/ranks.json")).expect("ranks.json");
    let mut out = vec![[0usize; 7]; 28];
    for kv in s
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
    {
        let (k, v) = kv.split_once(':').expect("key: value");
        let key = k.trim().trim_matches('"');
        let (l, name) = key.split_once('.').expect("layer.key");
        let l: usize = l.parse().expect("layer");
        let slot = KEYS
            .iter()
            .position(|x| *x == name)
            .unwrap_or_else(|| panic!("{key}"));
        out[l][slot] = v.trim().parse().expect("rank");
    }
    out
}

fn read_i32(path: &str) -> Vec<i32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().expect("i32")))
        .collect()
}

/// A `Q32.32` word as the real it denotes.
fn q32_f32(v: i64) -> f32 {
    (v as f64 / 4_294_967_296.0) as f32
}

/// An f32 rounded to bf16 and back: `torch.Tensor.bfloat16()`'s round-to-nearest-even
/// (`tests/learn_block_tune.rs:148-157`).
#[allow(
    clippy::verbose_bit_mask,
    reason = "the mask is the bf16 round's zero test, not a bit scan"
)]
fn round_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    if b & 0x7fff_ffff == 0 {
        return x;
    }
    let round = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    f32::from_bits(round & 0xffff_0000)
}

/// End the process on the success path instead of racing the driver's exit: this file allocates the
/// layer tape plus the KD head (several hundred pipelines), and the driver's compiler threads fault
/// in `libnvidia-gpucomp` during process teardown after the assertions have run
/// (`tests/learn_block_tune.rs:864-874`).
fn finish() -> ! {
    std::io::Write::flush(&mut std::io::stdout()).expect("flush");
    unsafe { libc::_exit(0) }
}

/// Sample `s`'s slice of a `[128, 2048, 1024]` dump file.
fn sample_of(path: &str, s: usize, n: usize) -> Vec<f32> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    f.seek(SeekFrom::Start((s * n * 4) as u64)).expect("seek");
    let mut b = vec![0u8; n * 4];
    f.read_exact(&mut b).expect("read");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("f32")))
        .collect()
}

/// Everything a step needs on the host: the 28 layer images, the embedding table, the final-norm
/// gain and the student's `h0` for all 128 samples.
struct Prep {
    pars: Vec<LayerPars>,
    h0: Vec<f32>,
    w_emb: Vec<f32>,
    g_final: Vec<f32>,
}

fn prep(engine: &Engine, cfg: &Config, student: &str) -> Prep {
    let d = cfg.n_embd;
    let rk = ranks(DUMP);
    let ids = read_i32(&format!("{DUMP}/calib_ids.i32"));
    let ns = 128;
    let t = ids.len() / ns;
    let mut hosts = Vec::with_capacity(28);
    for l in 0..28 {
        hosts.push(LayerHost::from_engine(engine, cfg, l).expect("layer host"));
    }
    let all: Vec<usize> = (0..cfg.vocab).collect();
    let we = load_rows(engine, "token_embd.weight", cfg.vocab, d, &all).expect("embedding");
    println!(
        "KD prep {} layers, {ns} samples x {t} tokens, d {d}, vocab {}, ranks {:?}",
        hosts.len(),
        cfg.vocab,
        rk[0]
    );
    let w_emb: Vec<f32> = we.iter().map(|&x| q32_f32(x)).collect();
    drop(we);
    let g_final: Vec<f32> = engine
        .decode_vectors()
        .output_norm
        .clone()
        .expect("output_norm")
        .iter()
        .map(|&x| q32_f32(x))
        .collect();
    let mut h0 = Vec::with_capacity(ns * t * d);
    for s in 0..ns {
        for &tok in &ids[s * t..(s + 1) * t] {
            let row = &w_emb[tok as usize * d..(tok as usize + 1) * d];
            h0.extend(row.iter().map(|&x| round_bf16(x)));
        }
    }
    let pars = (0..28)
        .map(|l| {
            let gains = match &hosts[l].mixer {
                Mixer::Softmax { gains, .. } => gains.clone(),
                Mixer::Affine { .. } => panic!("layer {l}: not a softmax layer"),
            };
            LayerPars::load(
                &format!("{student}/l{l}"),
                &hosts[l].attn_norm,
                &gains,
                &hosts[l].ffn_norm,
            )
            .unwrap_or_else(|e| panic!("layer {l}: {e}"))
        })
        .collect();
    Prep {
        pars,
        h0,
        w_emb,
        g_final,
    }
}

/// The reference's seven keys in [`LayerPars::scales`]' order, for the trace's `g.*`/`p0.*`/`p1.*`
/// file names.
fn scale_keys() -> Vec<(&'static str, bool)> {
    let mut v = Vec::with_capacity(14);
    for k in 0..3 {
        v.push((FRONT_KEYS[k], true));
        v.push((FRONT_KEYS[k], false));
    }
    for k in 0..4 {
        v.push((BLOCK_KEYS[k], true));
        v.push((BLOCK_KEYS[k], false));
    }
    v
}

#[test]
#[ignore = "measurement: opens the device"]
fn kd_memory() {
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _mem) = GpuCtx::new(&vk).expect("gpu ctx");
    let engine = Engine::open(MODEL).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let rk = ranks(DUMP);
    let host = LayerHost::from_engine(&engine, &cfg, 0).expect("layer host");
    let (nh, nkv, hd, rot, theta) = match &host.mixer {
        Mixer::Softmax {
            nh,
            nkv,
            hd,
            rot,
            theta,
            ..
        } => (*nh, *nkv, *hd, *rot, *theta),
        Mixer::Affine { .. } => panic!("not a softmax layer"),
    };
    let (t, d) = (2048, cfg.n_embd);
    let (fs, bs) = shapes_of(
        t, d, nh, nkv, hd, rot, theta, host.eps, host.a, cfg.n_ff, rk[0],
    );
    let free0 = free_mib();
    let a0 = g.alloc.get();
    let one = unsafe { ChainLayer::new(&g, fs, bs, 0.0, 0.0, 0.0) };
    let per = g.alloc.get() - a0;
    println!(
        "KD MEM free {free0} MiB, front qw {} kw {}, block a {} f {}, 2048 columns",
        fs.qw(),
        fs.kw(),
        bs.a,
        bs.f
    );
    println!("{}", KdTrain::memory_line(per, free_mib(), 1, d, cfg.vocab));
    println!(
        "KD MEM one trainer {per} B, 28 would be {} MiB, free after one {} MiB -> {}",
        (28 * per) / (1 << 20),
        free_mib(),
        if 28 * per < free_mib() * (1 << 20) {
            "resident"
        } else {
            "swapped"
        }
    );
    drop(one);
    finish();
}

/// The reference's step 0, whole: sample 48 through the reference's pre-KD student, the head, the
/// chain and one Adam step, against `~/nq-ref/kd/trace`.
#[test]
#[ignore = "measurement: opens the device, the teacher and the trace"]
fn kd_step0_golden() {
    let engine = Engine::open(MODEL).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let t = 2048;
    let d = cfg.n_embd;
    let rk = ranks(DUMP);
    let perm = read_i32(&format!("{KD}/perm8.i32"));
    let sample = perm[0] as usize;
    println!("KD GOLDEN sample {sample} (perm8[0][0]), t {t} d {d}");

    // ── the memory line first: one trainer, 28 layers' parameters, the free device memory ──
    let h = &LayerHost::from_engine(&engine, &cfg, 0).expect("layer host");
    let (nh, nkv, hd, rot, theta) = match &h.mixer {
        Mixer::Softmax {
            nh,
            nkv,
            hd,
            rot,
            theta,
            ..
        } => (*nh, *nkv, *hd, *rot, *theta),
        Mixer::Affine { .. } => panic!("not a softmax layer"),
    };
    let (fs, bs) = shapes_of(t, d, nh, nkv, hd, rot, theta, h.eps, h.a, cfg.n_ff, rk[0]);

    let p = prep(&engine, &cfg, &format!("{KD}/ref-blocks"));
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _mem) = GpuCtx::new(&vk).expect("gpu ctx");
    let free0 = free_mib();
    let h0 = unsafe { stream_buffer(&g, 128 * t * d) };
    unsafe { stream_upload(&g, h0, &p.h0) };
    let n = t * d;
    let we = unsafe { g.up_f32(&p.w_emb) };
    let gf = unsafe { g.up_f32(&p.g_final) };
    let a0 = g.alloc.get();
    let shapes = KdShapes {
        layers: 28,
        t,
        d,
        v: cfg.vocab,
    };
    let mut tr = unsafe { KdTrain::new(&g, shapes, fs, bs, p.pars, we, gf, h0, 256, 1) };
    tr.capture_dh(true);
    println!(
        "KD GOLDEN memory: free before {free0} MiB, step resident {} MiB (one trainer, 28 layers \
         swapped through it), h0 pool {} MiB, head tiles {} MiB",
        (g.alloc.get() - a0) >> 20,
        (128 * n * 4) >> 20,
        (2 * 256 * cfg.vocab * 4) >> 20
    );

    // ── the step ──
    let ht = sample_of(&format!("{LOOP}/s-28.xt.f32"), sample, n);
    let t0 = std::time::Instant::now();
    unsafe {
        tr.begin(&g, sample, &ht);
    }
    let t_beg = t0.elapsed().as_secs_f64();
    let t_f = std::time::Instant::now();
    unsafe { tr.forward(&g) };
    let t_fwd = t_f.elapsed().as_secs_f64();
    let t_h = std::time::Instant::now();
    let loss = unsafe { tr.head(&g) };
    let t_head = t_h.elapsed().as_secs_f64();
    let t_b = std::time::Instant::now();
    let bwd = unsafe { tr.backward(&g) };
    let t_bwd = t_b.elapsed().as_secs_f64();
    println!(
        "KD GOLDEN wall: begin {t_beg:.3} s, forward {t_fwd:.3} s, head {t_head:.3} s, backward \
         {t_bwd:.3} s, total {:.3} s",
        t0.elapsed().as_secs_f64()
    );

    // ── 1. the loss against the trace's 2.96875 (the reference computes logits in bf16, whose
    // resolution near 3 is 0.0156) ──
    let want = read_f32(&format!("{}/loss.f32", trace())).expect("loss");
    let want = want[0];
    let rel = (loss - f64::from(want)).abs() / f64::from(want);
    println!("KD GOLDEN loss: ours {loss:.6} trace {want:.6} rel {rel:.2e}");
    assert!(
        rel <= Floor::LOSS_REL.rel(),
        "kd golden: loss {loss:.6} is {rel:.2e} from the trace's {want:.6}, over the reference's \
         own {:.3e}",
        Floor::LOSS_REL.rel()
    );

    // ── 2. h_l and dh_l ──
    let mut worst_h = (0.0f64, 0usize);
    let mut worst_dh = (0.0f64, 0usize);
    let mut checks: Vec<(usize, f64, f64)> = Vec::new();
    let mut dh_cos: Vec<(usize, f64)> = Vec::new();
    let mut our_dh = bwd.dh.clone();
    our_dh.push(unsafe { tr.head_dh(&g) });
    for &l in &[0usize, 1, 14, 24, 25, 26, 27, 28] {
        let hw = read_f32(&format!("{}/h.l{l}.f32", trace())).expect("h");
        let dw = read_f32(&format!("{}/dh.l{l}.f32", trace())).expect("dh");
        let hg = unsafe { tr.residual(&g, l) };
        let rh = rel_l2(&hg, &hw);
        let rd = rel_l2(&our_dh[l], &dw);
        println!("KD GOLDEN h.l{l}: rel {rh:.3e}   dh.l{l}: rel {rd:.3e}");
        // Where the trace's tensor sits, if not at the index the name suggests: the cosine of our
        // `dh[l]` with every other layer's, printed for the misalignment the name would hide.
        for j in [l.saturating_sub(1), l, (l + 1).min(28)] {
            let other = read_f32(&format!("{}/dh.l{j}.f32", trace())).expect("dh");
            println!(
                "KD GOLDEN   cos(our dh[{l}], trace dh.l{j}) {:.6}",
                cosine(&our_dh[l], &other)
            );
        }
        let n_our = our_dh[l]
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let n_want = dw
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        println!("KD GOLDEN   |dh[{l}]| {n_our:.4e}, |trace| {n_want:.4e}");
        if rh > worst_h.0 {
            worst_h = (rh, l);
        }
        if rd > worst_dh.0 {
            worst_dh = (rd, l);
        }
        if l == 28 || l == 0 {
            checks.push((l, rh, rd));
            dh_cos.push((l, cosine(&our_dh[l], &dw)));
        }
    }
    let hd = unsafe { tr.head_dh(&g) };
    let hw28 = read_f32(&format!("{}/dh.l28.f32", trace())).expect("dh28");
    println!(
        "KD GOLDEN head dh: cos with trace dh.l28 {:.6}, rel {:.3e}",
        cosine(&hd, &hw28),
        rel_l2(&hd, &hw28)
    );
    println!(
        "KD GOLDEN worst h.l{} {:.3e}, dh.l{} {:.3e}",
        worst_h.1, worst_h.0, worst_dh.1, worst_dh.0
    );
    // Isolate the head's own fidelity from the forward's: the same tape over the trace's h28.
    let floor_rel = {
        let hw = read_f32(&format!("{}/h.l28.f32", trace())).expect("h28");
        unsafe { tr.set_residual(&g, 28, &hw) };
        unsafe { tr.head(&g) };
        let hd = unsafe { tr.head_dh(&g) };
        let rel = rel_l2(&hd, &hw28);
        println!(
            "KD GOLDEN head over the TRACE's h28: cos {:.6}, rel {rel:.3e} — the floor any chain \
             from this teacher can reach",
            cosine(&hd, &hw28)
        );
        rel
    };
    for (l, rh, _) in &checks {
        let b = Floor::H_REL.rel();
        assert!(
            *rh <= b,
            "kd golden: h.l{l} rel {rh:.3e} over the reference's own {b:.3e}"
        );
    }
    // The assertion is on the direction; the relative error is printed against the head's floor
    // over the trace's own `h28` (the teacher logits are the HF model's, the cached `h28` our
    // engine's stream).
    for (l, c) in &dh_cos {
        println!("KD GOLDEN dh.l{l} cosine {c:.6} (the head's own relative floor {floor_rel:.3e})");
        let b = Floor::DH_COS.cos();
        assert!(
            *c >= b,
            "kd golden: dh.l{l} cosine {c:.6} under the reference's own {b:.6}"
        );
    }

    // ── 3. every scale gradient's cosine with the trace's `g.*` ──
    let keys = scale_keys();
    let mut cos: Vec<f64> = Vec::with_capacity(392);
    let mut worst = (2.0f64, 0usize, 0usize);
    for l in 0..28 {
        for (j, (key, pre)) in keys.iter().enumerate() {
            let which = if *pre { "scale_pre" } else { "scale_post" };
            let gw = read_f32(&format!("{}/g.{l}.{key}.{which}.f32", trace()))
                .unwrap_or_else(|e| panic!("g.{l}.{key}.{which}: {e}"));
            // The tape's grads hold `[s_post, s_pre]`; `scales()` holds `pre` then `post`.
            let given = &bwd.scales[l][j >> 1][usize::from(*pre)];
            assert_eq!(given.len(), gw.len(), "g.{l}.{key}.{which}: width");
            let c = cosine(given, &gw);
            if c < worst.0 {
                worst = (c, l, j);
            }
            cos.push(c);
        }
    }
    cos.sort_by(|a, b| a.partial_cmp(b).expect("cos"));
    let (min, med) = (cos[0], cos[cos.len() / 2]);
    let g_floor = Floor::G_COS.cos();
    let under = cos.iter().filter(|c| **c < g_floor).count();
    println!(
        "KD GOLDEN scale-gradient cosine over {}: min {min:.6} ({}.{}), p05 {:.6}, median \
         {med:.6}, {} under the reference's own {g_floor:.6}",
        cos.len(),
        worst.1,
        keys[worst.2].0,
        cos[cos.len() / 20],
        under
    );
    for l in 0..28 {
        for (j, (key, pre)) in keys.iter().enumerate() {
            let which = if *pre { "scale_pre" } else { "scale_post" };
            let gw = read_f32(&format!("{}/g.{l}.{key}.{which}.f32", trace())).expect("g");
            let given = &bwd.scales[l][j >> 1][usize::from(*pre)];
            let c = cosine(given, &gw);
            if c < g_floor {
                let ng = given
                    .iter()
                    .map(|x| f64::from(*x).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let nw = gw.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                println!(
                    "KD GOLDEN   low cos {l}.{key}.{which} {c:.6}: |g| {ng:.4e} |trace| \
                     {nw:.4e} width {}",
                    given.len()
                );
            }
        }
    }
    assert!(
        min >= g_floor,
        "kd golden: minimum gradient cosine {min:.6} under the reference's own {g_floor:.6}"
    );

    // ── 4. one Adam step, and `p1` against `p0` + our update ──
    let lr = cosine_lr(1e-5, 0.0, 1024, 1024);
    let p0 = tr.pars()[0].scales();
    let before: Vec<Vec<Vec<f32>>> = tr.pars().iter().map(LayerPars::scales).collect();
    let mut adam = ScaleAdam::zero(28, &p0);
    let t_a = std::time::Instant::now();
    tr.adam(&mut adam, &bwd.scales, 1, lr[0]);
    let t_adam = t_a.elapsed().as_secs_f64();
    println!("KD GOLDEN optimiser {t_adam:.3} s at lr {:.3e}", lr[0]);
    let after: Vec<Vec<Vec<f32>>> = tr.pars().iter().map(LayerPars::scales).collect();
    // `AdamHyper::OPTIMI` is the same three constants the reference's optimi.AdamW defaults to.
    let (b1t, rbc2) = AdamHyper::OPTIMI.bias_corrections(1);
    println!("KD GOLDEN adam bias correction step 1: beta1^t {b1t}, sqrt(1-beta2^t) {rbc2}");
    // After step 1 Adam's update is `−lr·r(g)` with `r(g) = g/(|g| + ε)` (m̂ = g, v̂ = g²), plus
    // optimi's decoupled decay `−lr·wd·p0` (`optimi/adam.py:372-375`, `wd = 1e-2`), which the f32
    // trace keeps and this driver drops. The trace's `p0` is bf16-exact and its `p1` is f32 (read
    // off the words). So, given each side's own `g`, `|ours − p1| ≤ lr·|r(g) − r(g_ref)| +
    // lr·wd·|p0| + 2 f32 steps at `p1` (one rounding each side): a derived bound that checks the
    // optimiser, while the sign counts below check the gradient against the reference's own bf16
    // floor.
    let eps_adam = 1e-6_f64;
    let wd = 1e-2_f64;
    let r = |g: f64| g / (g.abs() + eps_adam);
    let (mut dp0, mut dp1, mut dp1_over) = (0f32, 0f32, 0f64);
    let (mut bad, mut n_big, mut bad_small, mut n_small) = (0usize, 0usize, 0usize, 0usize);
    for l in 0..28 {
        for (j, (key, pre)) in keys.iter().enumerate() {
            let which = if *pre { "scale_pre" } else { "scale_post" };
            let p0w = read_f32(&format!("{}/p0.{l}.{key}.{which}.f32", trace()))
                .unwrap_or_else(|e| panic!("p0.{l}.{key}.{which}: {e}"));
            let p1w = read_f32(&format!("{}/p1.{l}.{key}.{which}.f32", trace()))
                .unwrap_or_else(|e| panic!("p1.{l}.{key}.{which}: {e}"));
            let g = &bwd.scales[l][j >> 1][usize::from(*pre)];
            let gw = read_f32(&format!("{}/g.{l}.{key}.{which}.f32", trace())).expect("g");
            let gmax = g.iter().fold(0f32, |m, x| m.max(x.abs()));
            for i in 0..p1w.len() {
                let scale = p1w[i].abs().max(1e-3);
                dp0 = dp0.max((before[l][j][i] - p0w[i]).abs() / scale);
                dp1 = dp1.max((after[l][j][i] - p1w[i]).abs() / scale);
                let f32_step = f64::from(f32::EPSILON) * f64::from(p1w[i].abs());
                let bound = lr[0] * (r(f64::from(g[i])) - r(f64::from(gw[i]))).abs()
                    + lr[0] * wd * f64::from(p0w[i].abs())
                    + 2.0 * f32_step;
                let d = f64::from((after[l][j][i] - p1w[i]).abs());
                dp1_over = dp1_over.max(d / bound);
                let want_d = p1w[i] - p0w[i];
                let our_d = after[l][j][i] - before[l][j][i];
                if want_d.abs() > 1e-9 {
                    if g[i].abs() > 0.1 * gmax {
                        n_big += 1;
                        if (want_d > 0.0) != (our_d > 0.0) {
                            bad += 1;
                        }
                    } else {
                        n_small += 1;
                        if (want_d > 0.0) != (our_d > 0.0) {
                            bad_small += 1;
                        }
                    }
                }
            }
        }
    }
    println!(
        "KD GOLDEN p0: our export vs trace max rel {dp0:.3e}; p1: ours vs trace max rel \
         {dp1:.3e}, max over the derived update bound {dp1_over:.3e}"
    );
    println!(
        "KD GOLDEN update sign disagrees {bad} of {n_big} where |g| > 0.1 max|g|, {bad_small} of \
         {n_small} below that"
    );
    // Both sides start from the same bf16 words.
    assert!(
        dp0 == 0.0,
        "kd golden: p0 export {dp0:.3e} from the trace's pre-step scales"
    );
    assert!(
        dp1_over <= 1.0,
        "kd golden: p1 sits {dp1_over:.3e}× the derived update bound from the trace's"
    );
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "counts under 2^24; the bound is a floor of a non-negative product"
    )]
    let (big_max, small_max) = (
        (Floor::SIGN_BIG.rel() * n_big as f64).floor() as usize,
        (Floor::SIGN_SMALL.rel() * n_small as f64).floor() as usize,
    );
    assert!(
        n_big > 0 && bad <= big_max,
        "kd golden: {bad} of {n_big} update signs disagree with p1 − p0 where |g| > 0.1 max|g|, \
         over the reference's own {big_max}"
    );
    assert!(
        bad_small <= small_max,
        "kd golden: {bad_small} of {n_small} update signs disagree below that, over the \
         reference's own {small_max}"
    );
    finish();
}

/// One layer's trainer, its three phases timed alone: the factor load `KdTrain::use_layer` makes,
/// the forward and the backward, each a synchronous submission, eight reps after one warm rep.
#[test]
#[ignore = "measurement: opens the device and the teacher"]
fn kd_layer_wall() {
    let engine = Engine::open(MODEL).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let (t, d) = (2048, cfg.n_embd);
    let rk = ranks(DUMP);
    let h = &LayerHost::from_engine(&engine, &cfg, 0).expect("layer host");
    let (nh, nkv, hd, rot, theta) = match &h.mixer {
        Mixer::Softmax {
            nh,
            nkv,
            hd,
            rot,
            theta,
            ..
        } => (*nh, *nkv, *hd, *rot, *theta),
        Mixer::Affine { .. } => panic!("not a softmax layer"),
    };
    let (fs, bs) = shapes_of(t, d, nh, nkv, hd, rot, theta, h.eps, h.a, cfg.n_ff, rk[0]);
    let p = prep(&engine, &cfg, &env("APEIRON_KD_STUDENT", LOOP));
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _mem) = GpuCtx::new(&vk).expect("gpu ctx");
    let mut ch = unsafe { ChainLayer::new(&g, fs, bs, 0.0, 0.0, 0.0) };
    let n = t * d;
    unsafe { stream_upload(&g, ch.x(), &p.h0[..n]) };
    unsafe { stream_upload(&g, ch.dy(), &vec![1e-4f32; n]) };
    let reps = 8;
    let (mut tl, mut tf, mut tb) = (0f64, 0f64, 0f64);
    for r in 0..=reps {
        let lp = &p.pars[r % 2];
        let t0 = std::time::Instant::now();
        unsafe { ch.load(&g, &lp.ff, &lp.g1, &lp.gn, &lp.bf, &lp.g2) };
        let t1 = std::time::Instant::now();
        unsafe { ch.forward(&g) };
        let t2 = std::time::Instant::now();
        unsafe { ch.backward(&g) };
        let t3 = std::time::Instant::now();
        if r > 0 {
            tl += (t1 - t0).as_secs_f64();
            tf += (t2 - t1).as_secs_f64();
            tb += (t3 - t2).as_secs_f64();
        }
    }
    let ms = 1e3 / reps as f64;
    println!(
        "KD LAYER mean of {reps}: load {:.2} fwd {:.2} bwd {:.2} ms; per step as KdTrain runs it \
         (56 loads, 56 fwd, 28 bwd) {:.3} s",
        tl * ms,
        tf * ms,
        tb * ms,
        (56.0 * tl + 56.0 * tf + 28.0 * tb) / reps as f64
    );
    finish();
}

/// Steps 0..5 on our student, the median wall per step split into its four parts.
#[test]
#[ignore = "measurement: opens the device and the teacher"]
fn kd_wall() {
    let engine = Engine::open(MODEL).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let t = 2048;
    let d = cfg.n_embd;
    let rk = ranks(DUMP);
    let perm = read_i32(&format!("{KD}/perm8.i32"));
    let student = env("APEIRON_KD_STUDENT", LOOP);
    let h = &LayerHost::from_engine(&engine, &cfg, 0).expect("layer host");
    let (nh, nkv, hd, rot, theta) = match &h.mixer {
        Mixer::Softmax {
            nh,
            nkv,
            hd,
            rot,
            theta,
            ..
        } => (*nh, *nkv, *hd, *rot, *theta),
        Mixer::Affine { .. } => panic!("not a softmax layer"),
    };
    let (fs, bs) = shapes_of(t, d, nh, nkv, hd, rot, theta, h.eps, h.a, cfg.n_ff, rk[0]);
    let p = prep(&engine, &cfg, &student);
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _mem) = GpuCtx::new(&vk).expect("gpu ctx");
    let n = t * d;
    let h0 = unsafe { stream_buffer(&g, 128 * n) };
    unsafe { stream_upload(&g, h0, &p.h0) };
    let we = unsafe { g.up_f32(&p.w_emb) };
    let gf = unsafe { g.up_f32(&p.g_final) };
    let shapes = KdShapes {
        layers: 28,
        t,
        d,
        v: cfg.vocab,
    };
    let mut tr = unsafe { KdTrain::new(&g, shapes, fs, bs, p.pars, we, gf, h0, 256, 1) };
    let mut adam = ScaleAdam::zero(28, &tr.pars()[0].scales());
    let lr = cosine_lr(1e-5, 0.0, 1024, 1024);
    let steps = envu("APEIRON_KD_WALL_STEPS", 6);
    let (mut tb, mut tf, mut th, mut td, mut to) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for step in 0..steps {
        let s = perm[step] as usize;
        let ht = sample_of(&format!("{LOOP}/s-28.xt.f32"), s, n);
        let t0 = std::time::Instant::now();
        unsafe { tr.begin(&g, s, &ht) };
        tb += t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        unsafe { tr.forward(&g) };
        tf += t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let loss = unsafe { tr.head(&g) };
        th += t2.elapsed().as_secs_f64();
        let t3 = std::time::Instant::now();
        let bwd = unsafe { tr.backward(&g) };
        td += t3.elapsed().as_secs_f64();
        let t4 = std::time::Instant::now();
        tr.adam(&mut adam, &bwd.scales, step + 1, lr[step]);
        to += t4.elapsed().as_secs_f64();
        println!(
            "KD WALL step {step} sample {s} loss {loss:.5} begin/upload {:.3} fwd {:.3} head \
             {:.3} bwd {:.3} optim {:.3} total {:.3} s",
            tb,
            tf,
            th,
            td,
            to,
            tb + tf + th + td + to
        );
    }
    let per = (tb + tf + th + td + to) / steps as f64;
    println!(
        "KD WALL per step (mean of {steps}): begin {:.3} fwd {:.3} head {:.3} bwd {:.3} optim \
         {:.3} total {per:.3} s; 1024 steps {:.0} s",
        tb / steps as f64,
        tf / steps as f64,
        th / steps as f64,
        td / steps as f64,
        to / steps as f64,
        per * 1024.0
    );
    finish();
}

/// One layer's use, per dispatch: the layer's forward, one head run, the backward's checkpointing
/// re-run of that forward, and the layer's backward, as GPU microseconds per kernel
/// ([`KdTrain::profile_layer`]).
///
/// The control is the benchmark: the per-dispatch sum of the forward band sits within 25% of the
/// per-layer forward wall `kd_wall` measures (0.455 s / 28 = 16.2 ms), the difference being the
/// barrier and submission time a dispatch's own span excludes.
#[test]
#[ignore = "measurement: opens the device and the teacher"]
fn kd_kernel_profile() {
    let engine = Engine::open(MODEL).expect("open teacher");
    let cfg = Config::from_engine(&engine).expect("config");
    let t = 2048;
    let d = cfg.n_embd;
    let rk = ranks(DUMP);
    let perm = read_i32(&format!("{KD}/perm8.i32"));
    let student = env("APEIRON_KD_STUDENT", LOOP);
    let h = &LayerHost::from_engine(&engine, &cfg, 0).expect("layer host");
    let (nh, nkv, hd, rot, theta) = match &h.mixer {
        Mixer::Softmax {
            nh,
            nkv,
            hd,
            rot,
            theta,
            ..
        } => (*nh, *nkv, *hd, *rot, *theta),
        Mixer::Affine { .. } => panic!("not a softmax layer"),
    };
    let (fs, bs) = shapes_of(t, d, nh, nkv, hd, rot, theta, h.eps, h.a, cfg.n_ff, rk[0]);
    let p = prep(&engine, &cfg, &student);
    let s = perm[0] as usize;
    let vk = unsafe { Vk::load() }.expect("vulkan");
    let (g, _mem) = GpuCtx::new(&vk).expect("gpu ctx");
    let n = t * d;
    // One sample's `h0`, not the pool of 128: the profile runs one layer, and `begin(.., 0, ..)`
    // then reads sample `s` out of a `T × d` buffer instead of uploading 1 GiB.
    let h0 = unsafe { stream_buffer(&g, n) };
    unsafe { stream_upload(&g, h0, &p.h0[s * n..(s + 1) * n]) };
    let we = unsafe { g.up_f32(&p.w_emb) };
    let gf = unsafe { g.up_f32(&p.g_final) };
    let shapes = KdShapes {
        layers: 28,
        t,
        d,
        v: cfg.vocab,
    };
    let t0 = std::time::Instant::now();
    let mut tr = unsafe { KdTrain::new(&g, shapes, fs, bs, p.pars, we, gf, h0, 256, 1) };
    println!(
        "KD PROF setup {:.1} s: free {} MiB, trainers resident 1, head rc 256",
        t0.elapsed().as_secs_f64(),
        free_mib()
    );
    let ht = sample_of(&format!("{LOOP}/s-28.xt.f32"), s, n);
    unsafe { tr.begin(&g, 0, &ht) };
    let layer = envu("APEIRON_KD_PROF_LAYER", 13);
    println!(
        "KD PROF layer {layer} sample {s} t {t} d {d} v {}",
        cfg.vocab
    );
    let t1 = std::time::Instant::now();
    let prof = unsafe { tr.profile_layer(&g, layer) };
    println!(
        "KD PROF wall {:.3} s, {} dispatches stamped",
        t1.elapsed().as_secs_f64(),
        prof.len()
    );
    // The control's other half on this run and this source: one 28-layer forward, per layer,
    // including each layer's `use_layer` image copy — the part a dispatch span cannot see. The
    // first pass also pays the one-time host→store upload of each layer's 70.9 MiB image; the
    // second is the steady-state per-layer wall a step pays.
    let t2 = std::time::Instant::now();
    unsafe { tr.forward(&g) };
    let cold = t2.elapsed().as_secs_f64() * 1e3 / 28.0;
    let t3 = std::time::Instant::now();
    unsafe { tr.forward(&g) };
    let per_layer = t3.elapsed().as_secs_f64() * 1e3 / 28.0;
    println!(
        "KD PROF fwd wall per layer: first pass (28 image uploads) {cold:.3} ms, second pass \
         (steady) {per_layer:.3} ms"
    );
    for (band, name, us) in &prof {
        println!("PROF {band} {name} {us:.1}");
    }
    let mut agg: Vec<(&'static str, usize, f64)> = Vec::new();
    for (_, name, us) in &prof {
        match agg.iter_mut().find(|a| a.0 == *name) {
            Some(a) => {
                a.1 += 1;
                a.2 += us;
            }
            None => agg.push((name, 1, *us)),
        }
    }
    agg.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (name, calls, us) in &agg {
        println!("PROFSUM {name} calls {calls} us {us:.1}");
    }
    for band in ["fwd", "fwd2", "head", "bwd"] {
        let mut calls = 0usize;
        let mut us = 0f64;
        for (b, _, u) in &prof {
            if *b == band {
                calls += 1;
                us += u;
            }
        }
        println!(
            "PROFSUM TAPE {band} calls {calls} us {us:.1} ms {:.3}",
            us / 1000.0
        );
    }
    let fwd_us = prof
        .iter()
        .filter(|(b, _, _)| *b == "fwd")
        .map(|(_, _, u)| *u)
        .sum::<f64>();
    println!(
        "KD PROF CONTROL fwd dispatches {:.3} ms vs wall {per_layer:.3} ms: ratio {:.3} \
         (band 0.75..1.25 of the wall)",
        fwd_us / 1000.0,
        fwd_us / 1000.0 / per_layer
    );
    finish();
}
