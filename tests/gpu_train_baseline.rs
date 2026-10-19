//! GPU-TRAIN baseline: the measured state of the existing GPU forward, its parity with the CPU
//! engine on a small shape, and the statement of what a training step must match.
//!
//! (a) **Small-shape parity.** `tests/gpu_layer_match.rs`'s exact shape and contract: one decoder
//! layer of Qwen2.5-0.5B-Q4_K_M (layer 0 plus the output `RMSNorm`) at positions 0 and 1, the GPU
//! dispatch (`embed_step` -> `record_layer` -> `final_norm_step`, `Carrier::Q8`) against the
//! engine's own truncated layer loop (`Engine::step_trunk_state`, `KernelMode::Exact`, with
//! `config.n_layer = 1`). Tolerance: that test's pinned per-position bounds
//! `max |gpu - cpu| / max |cpu| <= BOUNDS = [2.7e-2, 9.0e-2]`, measured there at 1.36e-2 and
//! 4.39e-2 on the same token pair.
//!
//! (b) **Production prefill microbench.** Whole-prompt prefill of 512 tokens (pp512) on the RTX
//! 5080: the GPU arm is `GpuDecode::prefill` on `Carrier::Q4` (the production prefill carrier),
//! the CPU arm is the engine's production prefill path
//! `Engine::prefill_state(..., KernelMode::Q8Act)` (the pooled int8-activation kernel; the
//! `KernelMode::Exact` arm runs one 512-token prompt in ~2.5e2 s and exceeded 560 s on a loaded
//! host, so it is a single unpaired sample behind `GPU_TRAIN_EXACT=1`, never part of this gate).
//! Five interleaved paired runs, order alternated per pair, one warm-up pass per arm outside the
//! pairs; each pair, the mean ratio, and the observed min/max are reported. This is a microbench,
//! not a G1/llama.cpp claim: the host is shared with other build lanes and the load average is
//! printed with the run.
//!
//! (c) **The training-step reference.** The reference is
//! `apeiron::learn::grad::model::{forward, backward}`: `forward(&Weights, &Config, &tokens)`
//! returns the `t x vocab` Q32.32 logits and the `Tape`; `backward(&Weights, &Config, &Tape,
//! &dlogits)` returns `Grads` (one `LayerGrads` per layer plus `output_norm`, `token_embd`,
//! `lm_head`), every buffer in Q32.32, no float and no `unsafe` after the load boundary. The
//! reference itself is pinned by two tests in `src/learn/grad/model/tests.rs`: gate A, its logits
//! are bit-identical to `Engine::step_logits` at every position under the Q32 carrier
//! (`APEIRON_F32_CARRIER=0`); gate B, its analytic gradient matches a central difference of the
//! summed KL loss at sampled coordinates within `FD_TOL = 2^14` Q32.32 units at `H = 2^20`
//! Q32.32 units (`backward_matches_finite_difference`).
//!
//! A GPU training step must match that reference as follows. **Forward:** the GPU logits equal
//! the CPU `forward` logits at every position under the tolerance the matching GPU test states
//! (bit-identical when the GPU path keeps the Q32.32 carrier; the int8-chart path's bound is the
//! one this file asserts on the small shape). **Backward:** the parity assertion the backward
//! kernels will satisfy is elementwise closeness of every `Grads` buffer to the CPU `backward`
//! within `FD_TOL` Q32.32 units for a fixed `dlogits` — the same bound the CPU reference is
//! itself held to against the finite difference; an exact-integer GPU kernel satisfies it with 0.
//! **Step:** one optimizer update applied to both arms must leave the weight buffers' Q32.32 words
//! equal, so the next forward is bit-identical. Until those kernels exist there is no GPU
//! training parity to measure; this file records the forward baseline they start from.
//!
//! Board row: `apeiron-gpu-train-w1-t01-EQpaf3sVmt04RegjErRx1k`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_train_baseline -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_layer_match/gpu_prefill_match tests"
)]

use apeiron::engine::testdata::model_path;
use apeiron::engine::{Engine, KernelMode};
use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::layer::{embed_step, final_norm_step, rec_nodes, record_layer};
use apeiron::gpu::decode::pack::pack_proj_f32;
use apeiron::gpu::decode::state::{dequant_rows, LayerCfg, LayerW, Scratch, Stream};
use apeiron::gpu::decode::{rec_step, GpuCtx, GpuDecode};
use apeiron::gpu::simt::Carrier;
use apeiron::gpu::vulkan::*;
use std::time::Instant;

const MODEL: &str = "models/gguf/qwen05b-Q4_K_M.gguf";
/// Token ids and per-position bounds, identical to `tests/gpu_layer_match.rs`.
const TOKENS: [usize; 2] = [9707, 11];
const TMAX: u32 = 128;
const BOUNDS: [f64; 2] = [2.7e-2, 9.0e-2];
/// Production prefill shape (pp512) and the paired-run protocol.
const PP: usize = 512;
const PAIRS: usize = 5;

#[test]
fn gpu_train_baseline() {
    println!("device: {}", gpu_identity());
    println!(
        "load: {} (host shared with other build lanes; microbench, not a G1 claim)",
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    );
    let Some(model) = model_path(MODEL) else {
        panic!("model fixture {MODEL} not found (set APEIRON_MODELS)");
    };

    small_shape(&model);
    production_pp512(&model);
}

/// (a) `tests/gpu_layer_match.rs`'s shape and bounds: layer 0 + final norm, positions 0 and 1.
fn small_shape(model: &str) {
    let mut engine = Engine::open(model).expect("engine open");
    engine.config.n_layer = 1;
    let ec = engine.config.clone();
    let h = ec.n_embd;
    let mut st = engine.state(TOKENS.len()).expect("cpu state");
    let cpu: Vec<Vec<f64>> = TOKENS
        .iter()
        .map(|&t| {
            engine
                .step_trunk_state(&mut st, t as u32, KernelMode::Exact)
                .expect("cpu trunk step")
                .iter()
                .map(|&v| v as f64 / 4294967296.0)
                .collect()
        })
        .collect();

    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => panic!("No Vulkan device available: {e}"),
    };
    let gguf = Gguf::open(model).expect("gguf open");
    let cfg = LayerCfg {
        hidden: ec.n_embd as u32,
        dff: ec.n_ff as u32,
        nh: ec.n_head as u32,
        nkv: ec.n_head_kv as u32,
        hd: ec.head_dim as u32,
        tmax: TMAX,
        eps: ec.rms_eps.to_f64() as f32,
        theta: ec.rope_theta.to_f64() as f32,
        rope_dims: ec.rot_dims as u32,
        carrier: Carrier::Q8,
    };
    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let sc = unsafe { Scratch::new(&g, &cfg) }.expect("scratch");
    let lw = unsafe { LayerW::load(&g, &gguf, 0, &cfg) }.expect("layer 0 weights");
    let rows: Vec<Vec<f32>> = TOKENS
        .iter()
        .map(|&t| dequant_rows(&gguf, "token_embd.weight", t, 1).expect("embd row"))
        .collect();
    let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let emb = pack_proj_f32(cfg.carrier, &row_refs, h, false);
    let emb_w = unsafe { g.up_u32(&emb.words) };
    let emb_dm = unsafe { g.up_u32(&emb.dm) };
    let out_gain = unsafe {
        g.up_f32(&dequant_rows(&gguf, "output_norm.weight", 0, 1).expect("output_norm.weight"))
    };
    let (hid_buf, _) = unsafe {
        vk.buffer(
            h as u64 * 4,
            BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC,
            MEMORY_DEVICE_LOCAL,
        )
    }
    .expect("hid buf");

    let mut stream = Stream::new(&sc, &cfg);
    let emb_node = unsafe { embed_step(&g, &cfg, TOKENS.len() as u32, emb_w, emb_dm, &sc) };
    let nodes = unsafe { record_layer(&g, &cfg, &lw, &sc, 0, &mut stream) };
    let fin = unsafe { final_norm_step(&g, &cfg, &stream, &sc, out_gain, hid_buf) };
    let live = stream.cur;

    let host = |n: usize| unsafe {
        let (b, m) = vk
            .buffer(
                n as u64 * 4,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .expect("readback buffer");
        (b, vk.map(m).cast::<f32>())
    };
    let (rb_norm, rb_norm_p) = host(h);
    let (rb_hid, rb_hid_p) = host(h);
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_step(&vk, cb, &emb_node);
        let _ = rec_nodes(&vk, cb, &nodes);
        rec_step(&vk, cb, &fin);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_TRANSFER,
            ACCESS_SHADER_WRITE,
            ACCESS_TRANSFER_READ,
        );
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: h as u64 * 4,
        };
        (vk.copy_buffer)(cb, hid_buf, rb_norm, 1, &region);
        (vk.copy_buffer)(cb, live, rb_hid, 1, &region);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_HOST,
            ACCESS_TRANSFER_WRITE,
            ACCESS_HOST_READ,
        );
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
    }

    for (pos, c) in cpu.iter().enumerate() {
        unsafe {
            *sc.params_map = pos as u32;
            *sc.params_map.add(1) = pos as u32;
            vk.submit_wait(cb);
        }
        let gh: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_hid_p, h) }.to_vec();
        let gn: Vec<f64> = unsafe { std::slice::from_raw_parts(rb_norm_p, h) }
            .iter()
            .map(|&v| v as f64)
            .collect();
        let scale = c.iter().fold(0f64, |m, &v| m.max(v.abs())).max(1e-30);
        let max_abs = gn
            .iter()
            .zip(c)
            .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
        let rel = max_abs / scale;
        println!(
            "small shape pos {pos}: max|cpu|={scale:.6} max|d|={max_abs:.6e} max rel err={rel:.6e} bound={:.1e}",
            BOUNDS[pos]
        );
        assert!(gh.iter().all(|v| v.is_finite()), "non-finite GPU hidden");
        assert!(
            rel <= BOUNDS[pos],
            "pos {pos}: max relative error {rel:e} exceeds {:e}",
            BOUNDS[pos]
        );
    }
    drop(gguf);
    drop(engine);
}

/// (b) pp512: five interleaved GPU/CPU pairs on the production prefill shape.
fn production_pp512(model: &str) {
    let gguf = Gguf::open(model).expect("gguf open");
    let engine = Engine::open(model).expect("engine open");
    let ec = engine.config.clone();
    let vocab = ec.vocab_size as usize;
    let ids: Vec<usize> = (0..PP).map(|i| (7 + 131 * i) % vocab).collect();
    let ids32: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
    let mut gpu = GpuDecode::new_with_prefill(&gguf, &ec, PP as u32, Carrier::Q4, PP)
        .expect("GpuDecode::new_with_prefill pp512");

    // Warm-up, outside the pairs.
    assert_eq!(gpu.prefill(&ids).expect("gpu warm-up prefill").len(), vocab);
    let mut st = engine.state(PP).expect("cpu state");
    let cpu_first = engine
        .prefill_state(&mut st, &ids32, KernelMode::Q8Act)
        .expect("cpu warm-up prefill");
    println!(
        "production pp512: gpu argmax {} cpu (q8act) argmax {} agreement {}",
        gpu.last_argmax(),
        cpu_first,
        gpu.last_argmax() == cpu_first as usize
    );

    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(PAIRS);
    for i in 0..PAIRS {
        let gpu_run = |gpu: &mut GpuDecode, ids: &[usize]| {
            time_ms(|| {
                let l = gpu.prefill(ids).expect("gpu prefill");
                std::hint::black_box(l.len());
            })
        };
        // Alternate the order so a drifting clock cannot favour one arm.
        let (gpu_ms, cpu_ms) = if i % 2 == 0 {
            let g = gpu_run(&mut gpu, &ids);
            let mut st = engine.state(PP).expect("cpu state");
            let c = time_ms(|| {
                std::hint::black_box(
                    engine
                        .prefill_state(&mut st, &ids32, KernelMode::Q8Act)
                        .expect("cpu prefill"),
                );
            });
            (g, c)
        } else {
            let mut st = engine.state(PP).expect("cpu state");
            let c = time_ms(|| {
                std::hint::black_box(
                    engine
                        .prefill_state(&mut st, &ids32, KernelMode::Q8Act)
                        .expect("cpu prefill"),
                );
            });
            let g = gpu_run(&mut gpu, &ids);
            (g, c)
        };
        assert!(
            gpu_ms.is_finite() && gpu_ms > 0.0 && cpu_ms.is_finite() && cpu_ms > 0.0,
            "pair {}: non-finite or zero timing",
            i + 1
        );
        println!(
            "pair {}: gpu {gpu_ms:.3} ms ({:.0} tok/s) cpu {cpu_ms:.3} ms ({:.0} tok/s) cpu/gpu {:.2}x",
            i + 1,
            PP as f64 * 1e3 / gpu_ms,
            PP as f64 * 1e3 / cpu_ms,
            cpu_ms / gpu_ms
        );
        pairs.push((gpu_ms, cpu_ms));
    }
    let ratios: Vec<f64> = pairs.iter().map(|&(g, c)| c / g).collect();
    let mean = |f: &dyn Fn(&(f64, f64)) -> f64| pairs.iter().map(f).sum::<f64>() / PAIRS as f64;
    let rmean = ratios.iter().sum::<f64>() / PAIRS as f64;
    let rmin = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let rmax = ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "pp512 mean gpu {:.3} ms [{:.3}, {:.3}], mean cpu {:.3} ms [{:.3}, {:.3}], mean cpu/gpu {rmean:.2}x [{rmin:.2}, {rmax:.2}]",
        mean(&|p| p.0),
        pairs.iter().map(|p| p.0).fold(f64::INFINITY, f64::min),
        pairs.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max),
        mean(&|p| p.1),
        pairs.iter().map(|p| p.1).fold(f64::INFINITY, f64::min),
        pairs.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max),
    );
    // The GPU arm must beat the CPU arm on every pair: this is the data path the training loop
    // exists to move onto the device.
    assert!(
        ratios.iter().all(|&r| r > 1.0),
        "GPU prefill not faster than the CPU arm on every pair: {ratios:?}"
    );
    if std::env::var("GPU_TRAIN_EXACT").as_deref() == Ok("1") {
        let mut st = engine.state(PP).expect("cpu state");
        let ms = time_ms(|| {
            std::hint::black_box(
                engine
                    .prefill_state(&mut st, &ids32, KernelMode::Exact)
                    .expect("cpu exact prefill"),
            );
        });
        println!(
            "pp512 CPU Exact one shot (single sample, not averaged): {ms:.3} ms ({:.3} tok/s)",
            PP as f64 * 1e3 / ms
        );
    }
}

fn time_ms(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e3
}

fn gpu_identity() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "device name/driver unavailable (nvidia-smi absent)".to_string(),
    }
}
