//! Prefill projection GEMM microbench: the DP4A batch GEMM the prefill path runs
//! (`gemv_dp4a_batch_rt_bytes` at `prefill_tile`'s tile) against the cooperative-matrix chart GEMM
//! (`simt_nn::coopmat_gemm`), on the Qwen2.5-0.5B layer-0 projection shapes (qkv, o, gate/up, down)
//! at 512 tokens, both chart carriers (Q8, Q4), real layer-0 weights packed with `pack_proj_f32`.
//!
//! Timing: GPU timestamps around `INNER` back-to-back dispatches (compute barrier between), per
//! dispatch = span / `INNER`; `--reps` rounds, the two arms and all shapes interleaved inside each
//! round (arm order alternates per round). Reports mean and min–max µs, GMAC/s and int8 TOPS
//! (2 ops per MAC), and the pp512 implied by scaling the GEMM node times of the receipt
//! the gpu-prefill-vs-llamacpp note by the measured ratios.
//!
//! `cargo run --release --locked --features gpu --example gemm_bench -- [--reps N] [--model PATH]
//! [--shape NAME] [--stats] [--ir DIR] [--variants DIR]` — `--shape` keeps one shape (the implied
//! pp512 then needs all four and is skipped), `--stats` prints the driver's pipeline statistics
//! (registers, spills), `--blob DIR` writes each kernel's one-pipeline cache blob (the SM120 SASS:
//! `fold/scripts/nvuc_sass.py`),
//! `--variants DIR` adds one arm per `NAME.bm<BM>.bn<BN>.spv` compile of the GLSL in DIR.
#![allow(
    clippy::pedantic,
    missing_docs,
    reason = "GPU microbench; the lint profile of the other gpu examples"
)]

use apeiron::format::gguf::Gguf;
use apeiron::gpu::decode::pack::pack_proj_f32;
use apeiron::gpu::decode::prefill::prefill_tile;
use apeiron::gpu::decode::state::dequant_rows;
use apeiron::gpu::decode::{local_wg, make_step_named, step_barrier, GpuCtx, Step};
use apeiron::gpu::vulkan::*;
use apeiron::simt::{gemv_dp4a_batch_rt_bytes, Carrier};
use apeiron::simt_nn::{coopmat_gemm_module_from, COOPMAT_GEMM_SPV, TILE_BM, TILE_BN};

const TOKENS: u32 = 512;
const INNER: u32 = 10;

/// The receipt's pp512 profile (the gpu-prefill-vs-llamacpp note, `PREFILL_PROFILE=1`,
/// q4): GPU-active µs per call summed over 24 layers, and the whole call.
const RECEIPT_TOTAL_US: f64 = 22156.0;
const RECEIPT_TOK_S: f64 = 26587.8;
/// (shape, carrier, µs): qkv, o, gate/up at Q8; down 12 layers at Q4 and 12 at Q8.
const RECEIPT_NODES: [(&str, Carrier, f64); 5] = [
    ("qkv", Carrier::Q8, 1646.0),
    ("o", Carrier::Q8, 1551.0),
    ("gate_up", Carrier::Q8, 6486.0),
    ("down", Carrier::Q4, 4017.0),
    ("down", Carrier::Q8, 4007.0),
];
/// llama.cpp Vulkan pp512 in the same receipt.
const LLAMA_VULKAN_TOK_S: f64 = 50682.8;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

struct Case {
    name: &'static str,
    carrier: Carrier,
    m: u32,
    k: u32,
    /// arm 0 is dp4a, arm 1 the committed coopmat kernel, then the `--variants`
    arms: Vec<(String, Step)>,
    us: Vec<Vec<f64>>,
}

/// Mean, min, max.
fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

/// GPU-active µs per dispatch of `s`, over `INNER` back-to-back dispatches.
unsafe fn time(vk: &Vk, s: &Step) -> f64 {
    let pool = vk.timestamp_pool(2);
    let cb = vk.command_buffer();
    vk.begin(cb);
    vk.cmd_reset_timestamps(cb, &pool);
    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, 0);
    for _ in 0..INNER {
        (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, s.pipe);
        (vk.bind_sets)(
            cb,
            PIPELINE_BIND_POINT_COMPUTE,
            s.layout,
            0,
            1,
            &s.dset,
            0,
            std::ptr::null(),
        );
        (vk.dispatch)(cb, s.groups.0, s.groups.1, 1);
        step_barrier(vk, cb);
    }
    vk.cmd_write_timestamp(cb, &pool, STAGE_ALL_COMMANDS, 1);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    let ts = vk.read_timestamps(&pool);
    ts[1].wrapping_sub(ts[0]) as f64 * f64::from(vk.timestamp_period) / 1000.0 / f64::from(INNER)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut reps = 5usize;
    let mut model = "~/models/gguf/qwen05b-Q4_K_M.gguf".to_string();
    let mut only: Option<String> = None;
    let mut stats_on = false;
    let mut blob_dir: Option<String> = None;
    let mut variants_dir: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reps" => reps = args.next().and_then(|v| v.parse().ok()).expect("--reps N"),
            "--model" => model = args.next().expect("--model PATH"),
            "--shape" => only = Some(args.next().expect("--shape NAME")),
            "--stats" => stats_on = true,
            "--blob" => blob_dir = Some(args.next().expect("--blob DIR")),
            "--variants" => variants_dir = Some(args.next().expect("--variants DIR")),
            _ => panic!("unknown argument {a}"),
        }
    }
    let vk = unsafe { Vk::load() }.expect("Vulkan device");
    let gguf = Gguf::open(&model).expect("gguf open");
    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let lw = local_wg();
    let n = TOKENS as usize;
    let mut rng = Rng(0x6e66_b3a7_c0de_0005);
    println!(
        "device {} | model {model} | tokens {TOKENS} | reps {reps} x {INNER} dispatches | dp4a local {lw}",
        vk.device_name
    );

    let l0 = |s: &str| format!("blk.0.{s}.weight");
    let groups: [(&'static str, Vec<String>); 4] = [
        ("qkv", vec![l0("attn_q"), l0("attn_k"), l0("attn_v")]),
        ("o", vec![l0("attn_output")]),
        ("gate_up", vec![l0("ffn_gate"), l0("ffn_up")]),
        ("down", vec![l0("ffn_down")]),
    ];
    // (label, spv, bm, bn): the committed kernel, then each variant
    let mut kernels: Vec<(String, Vec<u8>, u32, u32)> = vec![(
        "coopmat".into(),
        COOPMAT_GEMM_SPV.to_vec(),
        TILE_BM,
        TILE_BN,
    )];
    if let Some(dir) = &variants_dir {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .expect("variants dir")
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "spv"))
            .collect();
        files.sort();
        for f in files {
            let stem = f.file_stem().unwrap().to_string_lossy().to_string();
            let tile = |key: &str| -> u32 {
                stem.split('.')
                    .find_map(|p| p.strip_prefix(key))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| panic!("{stem}: no .{key}<n> in the name"))
            };
            let (bm, bn) = (tile("bm"), tile("bn"));
            kernels.push((
                stem.clone(),
                std::fs::read(&f).expect("variant spv"),
                bm,
                bn,
            ));
        }
    }
    let mut cases: Vec<Case> = Vec::new();
    for (name, names) in &groups {
        if only.as_deref().is_some_and(|o| o != *name) {
            continue;
        }
        let rows: Vec<Vec<f32>> = names
            .iter()
            .map(|nm| {
                let t = gguf.header.tensor(nm).expect("tensor");
                dequant_rows(&gguf, nm, 0, t.dims[1] as usize).expect("dequant")
            })
            .collect();
        let k = gguf.header.tensor(&names[0]).expect("tensor").dims[0] as usize;
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let (ku, blocks) = (k, k / 32);
        let xq: Vec<u32> = (0..n * ku / 4).map(|_| rng.next()).collect();
        let sx: Vec<f32> = (0..n * blocks)
            .map(|_| 1e-3 + (rng.next() % 4096) as f32 * 4e-6)
            .collect();
        let xs: Vec<f32> = (0..n * blocks)
            .map(|_| (rng.next() % 2001) as f32 * 1e-3 - 1.0)
            .collect();
        let (bxq, bsx, bxs) = unsafe { (g.up_u32(&xq), g.up_f32(&sx), g.up_f32(&xs)) };
        for carrier in [Carrier::Q8, Carrier::Q4] {
            let p = pack_proj_f32(carrier, &refs, k, false);
            let (m, k) = (p.m as u32, p.k as u32);
            let (wb, dmb) = unsafe { (g.up_u32(&p.words), g.up_u32(&p.dm)) };
            let out = unsafe { g.dev(n * p.m * 4) };
            let bufs = [wb, bxq, out, dmb, bxs, bsx];
            let (bt, rt) = prefill_tile(m, k, carrier, lw);
            let dp = unsafe {
                make_step_named(
                    &g,
                    &gemv_dp4a_batch_rt_bytes(m, k, lw, carrier, bt, rt),
                    &bufs,
                    m.div_ceil(lw * rt),
                    TOKENS / bt,
                    "dp4a",
                )
            };
            let mut arms = vec![("dp4a".to_string(), dp)];
            for (label, spv, bm, bn) in &kernels {
                assert!(
                    TOKENS.is_multiple_of(*bn),
                    "{label}: tokens must fill the token tile"
                );
                let code: Vec<u8> = coopmat_gemm_module_from(spv, m, k, carrier)
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                let (gx, gy) = (m.div_ceil(*bm), TOKENS / bn);
                let st = unsafe { make_step_named(&g, &code, &bufs, gx, gy, "coopmat") };
                arms.push((label.clone(), st));
            }
            println!("shape {name:8} {carrier:?} m={m} k={k}: dp4a bt={bt} rt={rt}");
            for (arm, s) in &arms {
                if stats_on {
                    let st = unsafe { vk.stats(s.pipe) };
                    let line: Vec<String> = st
                        .iter()
                        .map(|x| format!("{}={}", x.name, x.value))
                        .collect();
                    println!("  stats {arm}: {}", line.join(", "));
                }
                if let Some(dir) = &blob_dir {
                    let code: Vec<u8> = if arm == "dp4a" {
                        gemv_dp4a_batch_rt_bytes(m, k, lw, carrier, bt, rt)
                    } else {
                        let spv = &kernels.iter().find(|x| &x.0 == arm).expect("kernel").1;
                        coopmat_gemm_module_from(spv, m, k, carrier)
                            .iter()
                            .flat_map(|w| w.to_le_bytes())
                            .collect()
                    };
                    let (layout, _) = unsafe { vk.ssbo_set(&bufs) };
                    let (_, blob) = unsafe { vk.isolated_cache_blob(&code, layout) };
                    let f = format!("{dir}/{name}_{carrier:?}_{arm}.pcache");
                    std::fs::write(&f, &blob).expect("write blob");
                }
            }
            let narms = arms.len();
            cases.push(Case {
                name,
                carrier,
                m,
                k,
                arms,
                us: vec![Vec::new(); narms],
            });
        }
    }
    // warm-up, then interleaved rounds
    for c in &cases {
        for (_, s) in &c.arms {
            unsafe { time(&vk, s) };
        }
    }
    for rep in 0..reps {
        for c in cases.iter_mut() {
            let n = c.arms.len();
            for i in 0..n {
                // rotate the arm order every round
                let a = (i + rep) % n;
                let t = unsafe { time(&vk, &c.arms[a].1) };
                c.us[a].push(t);
            }
        }
    }

    println!();
    println!(
        "| shape | carrier | m | k | arm | µs mean (min-max) | GMAC/s | TOPS | dp4a/arm mean (min-max) |"
    );
    println!("|---|---|---:|---:|---|---|---:|---:|---:|");
    let mut ratio_of = Vec::new();
    for c in &cases {
        let macs = f64::from(c.m) * f64::from(c.k) * f64::from(TOKENS);
        for (a, (label, _)) in c.arms.iter().enumerate() {
            let (mean, lo, hi) = stats(&c.us[a]);
            let gmac = macs / mean / 1e3;
            let rs: Vec<f64> = c.us[0].iter().zip(&c.us[a]).map(|(d, x)| d / x).collect();
            let (rm, rlo, rhi) = stats(&rs);
            println!(
                "| {} | {:?} | {} | {} | {label} | {mean:.1} ({lo:.1}-{hi:.1}) | {gmac:.0} | {:.1} | {rm:.2} ({rlo:.2}-{rhi:.2}) |",
                c.name,
                c.carrier,
                c.m,
                c.k,
                2.0 * gmac / 1e3,
            );
            if a == 1 {
                ratio_of.push((c.name, c.carrier, rm));
            }
        }
    }
    if only.is_some() {
        return;
    }
    // implied pp512: replace each receipt GEMM node by node · (coopmat/dp4a) of its shape
    let mut saved = 0.0;
    for (name, carrier, us) in RECEIPT_NODES {
        let r = ratio_of
            .iter()
            .find(|(n, c, _)| *n == name && *c == carrier)
            .expect("shape measured")
            .2;
        saved += us - us / r;
    }
    let total = RECEIPT_TOTAL_US - saved;
    let pp = RECEIPT_TOK_S * RECEIPT_TOTAL_US / total;
    println!();
    println!(
        "implied pp512 (derived, not measured end to end): GPU-active {RECEIPT_TOTAL_US:.0} µs -> {total:.0} µs; \
{RECEIPT_TOK_S:.0} tok/s -> {pp:.0} tok/s = {:.3}x llama.cpp Vulkan ({LLAMA_VULKAN_TOK_S:.0})",
        pp / LLAMA_VULKAN_TOK_S
    );
}
