//! One decoder layer's GPU dispatch (layer 0 of Qwen2.5-0.5B-Q4_K_M, positions 0 and 1) against
//! apeiron's CPU engine: the same trunk executed by `Engine::step_trunk_state` with
//! `config.n_layer = 1`, Q32.32 converted to f64.
//!
//! The CPU side is the engine's own layer loop (`compose_cols`); nothing is reimplemented. Both
//! sides end in the output `RMSNorm`, so the compared vectors are the output-normed layer-0 states.
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "faithful port of fold's modelfold::gpu Vulkan runtime; lint profile is fold's, not apeiron's"
)]

use apeiron::engine::testdata::model_path;
use apeiron::engine::{Engine, KernelMode};
use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::layer::{embed_step, final_norm_step, rec_nodes, record_layer};
use apeiron::gpu::decode::pack::pack_proj_f32;
use apeiron::gpu::decode::state::{dequant_rows, LayerCfg, LayerW, Scratch, Stream};
use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::vulkan::*;
use apeiron::simt::Carrier;

const MODEL: &str = "models/gguf/qwen05b-Q4_K_M.gguf";
/// Token ids fed at positions 0 and 1.
const TOKENS: [u32; 2] = [9707, 11];
const TMAX: u32 = 128;
/// Per-position bounds on the max relative error (max |gpu - cpu| / max |cpu|), each about twice the
/// clean value measured on this machine: position 0 measures 1.36e-2 (bound 2.7e-2), position 1
/// measures 4.39e-2 (bound 9.0e-2). Position 1 is limited by the device KV format, an int8 K cache
/// with one scale per (kv-head, position) against this model's K outliers (|k| up to 130), not by
/// the layer arithmetic: an f64 reference that quantizes K the same way tracks the device to 2-6%
/// where the engine's exact-KV state differs by up to 100% on other token sequences.
const BOUNDS: [f64; 2] = [2.7e-2, 9.0e-2];

#[test]
fn test_gpu_layer_match() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    let Some(model) = model_path(MODEL) else {
        println!("SKIP: Model file {MODEL} not found");
        return;
    };
    let gguf = Gguf::open(&model).expect("gguf open");
    let mut engine = Engine::open(&model).expect("engine open");
    let ec = engine.config.clone();
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
    let h = cfg.hidden as usize;
    println!("layer cfg: {cfg:?}");

    // ---- CPU: the engine's own layer loop, truncated to layer 0 ----
    engine.config.n_layer = 1;
    let mut st = engine.state(TOKENS.len()).expect("decode state");
    let cpu: Vec<Vec<f64>> = TOKENS
        .iter()
        .map(|&t| {
            engine
                .step_trunk_state(&mut st, t, KernelMode::Exact)
                .expect("cpu trunk step")
                .iter()
                .map(|&v| v as f64 / 4294967296.0)
                .collect()
        })
        .collect();

    // ---- GPU: embed -> layer 0 -> final norm ----
    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let sc = unsafe { Scratch::new(&g, &cfg) }.expect("scratch");
    let lw = unsafe { LayerW::load(&g, &gguf, 0, &cfg) }.expect("layer 0 weights");
    // embedding chart over the rows of the fed tokens (vocab = TOKENS.len()); params[1] selects the row
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for &t in &TOKENS {
        rows.push(dequant_rows(&gguf, "token_embd.weight", t as usize, 1).expect("embd row"));
    }
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
        let n = rec_nodes(&vk, cb, &nodes);
        rec_step(&vk, cb, &fin);
        println!("recorded {} dispatches (layer) + embed + final norm", n);
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

    let argmax = |v: &[f64]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };
    let mut worst = 0f64;
    for (pos, c) in cpu.iter().enumerate() {
        unsafe {
            *sc.params_map = pos as u32;
            *sc.params_map.add(1) = pos as u32; // row of the embedding chart
            vk.submit_wait(cb);
        }
        let gn: Vec<f64> = unsafe { std::slice::from_raw_parts(rb_norm_p, h) }
            .iter()
            .map(|&v| v as f64)
            .collect();
        let gh: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_hid_p, h) }.to_vec();
        let scale = c.iter().fold(0f64, |m, &v| m.max(v.abs())).max(1e-30);
        let max_abs = gn
            .iter()
            .zip(c)
            .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
        let rel = max_abs / scale;
        let (ga, ca) = (argmax(&gn), argmax(c));
        let hid_scale = gh.iter().fold(0f32, |m, &v| m.max(v.abs()));
        println!(
            "pos {pos}: max|cpu|={scale:.6} max abs diff={max_abs:.6e} max rel err={rel:.6e} \
             argmax gpu={ga} cpu={ca} (gpu hidden max|x|={hid_scale:.4})"
        );
        assert!(gh.iter().all(|v| v.is_finite()), "non-finite GPU hidden");
        assert!(
            rel <= BOUNDS[pos],
            "pos {pos}: max relative error {rel:e} exceeds {:e}",
            BOUNDS[pos]
        );
        assert_eq!(ga, ca, "pos {pos}: argmax mismatch gpu {ga} vs cpu {ca}");
        worst = worst.max(rel);
    }
    println!("Measured max relative error (worst over positions): {worst:.6e}");
}
