// Row: `inf-resident-prefill-GEB633UfORCK7XSmzJifHd` (INF-RESIDENT-PREFILL, wave 2): the cooperative-matrix GEMM equals the DP4A batch GEMM exactly.
//! The cooperative-matrix chart GEMM (`simt_nn::coopmat_gemm`) against the DP4A batch GEMM the
//! prefill path runs (`simt::gemv_dp4a_batch_rt_bytes` at `prefill_tile`'s tile), on the real
//! Qwen2.5-0.5B layer-0 projections (qkv, o, gate/up, down, plus the first Q4_K `ffn_down`) at the
//! Q8, Q4 and Q2 charts, over a 512-token random int8 activation block.
//!
//! Two checks per projection, every output cell:
//! 1. **integer**: unit scales (`d = 1, lo = 0, sx = 1, xs = 0`) make each output `Σ_b q_b`, an
//!    integer the f32 sum holds exactly while every partial is below 2^24 (asserted). Both kernels
//!    must equal the CPU integer dot of the chart codes on a sampled row set, and each other on all.
//! 2. **f32**: the real `(d, lo)` chart and random `sx`/`xs`: the two kernels' outputs are equal bit
//!    for bit.
#![cfg(feature = "gpu")]
#![allow(
    clippy::pedantic,
    missing_docs,
    reason = "GPU integration test; the lint profile of the other gpu_* tests"
)]

use apeiron::format::gguf::Gguf;
use apeiron::format::quant::GGML_TYPE_Q4_K;
use apeiron::gpu::decode::pack::{pack_proj_f32, pack_q4_k_raw, Proj};
use apeiron::gpu::decode::prefill::prefill_tile;
use apeiron::gpu::decode::state::dequant_rows;
use apeiron::gpu::decode::{local_wg, make_step_named, rec_step, GpuCtx, Step};
use apeiron::gpu::vulkan::*;
use apeiron::simt::{gemv_dp4a_batch_rt_bytes, Carrier};
use apeiron::simt_nn::{coopmat_gemm_bytes, coopmat_gemm_groups, GLSLC_ARGS, TILE_BN};
use std::path::Path;

const MODEL: &str = "~/models/gguf/qwen05b-Q4_K_M.gguf";
const TOKENS: u32 = 512;
const ROW_SAMPLES: usize = 48;

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

/// Signed activation code of token `t`, chart position `p` (k index in activation byte order).
fn xcode(xq: &[u32], k: usize, t: usize, p: usize) -> i64 {
    let word = xq[t * k / 4 + p / 4];
    i64::from((word >> (8 * (p % 4))) as u8 as i8)
}

/// Unsigned weight code at chart position `p` of row `r`: the code `OpSUDot` pairs with activation
/// byte `p` (word `c = (p/4)/dpw`, phase `l = (p/4)%dpw`, byte `p%4`).
fn wcode(words: &[u32], m: usize, carrier: Carrier, r: usize, p: usize) -> i64 {
    let pack = carrier.pack() as usize;
    let (dpw, cpb, wbits) = (pack / 4, 32 / pack, 32 / pack);
    let (b, q) = (p / 32, p % 32);
    let (c, l, i) = ((q / 4) / dpw, (q / 4) % dpw, q % 4);
    let v = words[(b * cpb + c) * m + r];
    i64::from((v >> (8 * i + wbits * l)) & carrier.qmask())
}

unsafe fn readback(vk: &Vk, src: Handle, n: usize) -> Vec<f32> {
    let (rb, mem) = vk
        .buffer(
            n as u64 * 4,
            BUFFER_USAGE_TRANSFER_DST,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("readback buffer");
    let cb = vk.command_buffer();
    vk.begin(cb);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: n as u64 * 4,
    };
    (vk.copy_buffer)(cb, src, rb, 1, &region);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    std::slice::from_raw_parts(vk.map(mem).cast::<f32>(), n).to_vec()
}

unsafe fn run(vk: &Vk, s: &Step) {
    let cb = vk.command_buffer();
    vk.begin(cb);
    rec_step(vk, cb, s);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
}

/// The chart at `carrier`, host side so the codes stay readable. Q4 with every tensor Q4_K packs
/// the raw bytes losslessly (`LayerW::load`'s rule); otherwise the f32 dequantization is packed at
/// `carrier` — Q4 included, unlike `LayerW::load` (which falls back to Q8), so the Q4 unpack runs on
/// every shape.
fn pack(gguf: &Gguf, names: &[String], carrier: Carrier) -> (Proj, Carrier) {
    let ts: Vec<_> = names
        .iter()
        .map(|nm| gguf.header.tensor(nm).expect("tensor"))
        .collect();
    let k = ts[0].dims[0] as usize;
    let m: usize = ts.iter().map(|t| t.dims[1] as usize).sum();
    if carrier == Carrier::Q4
        && ts
            .iter()
            .all(|t| t.ggml_type == GGML_TYPE_Q4_K && k.is_multiple_of(256))
    {
        let mut raw = Vec::new();
        for t in &ts {
            raw.extend_from_slice(&gguf.tensor_bytes(t).expect("tensor bytes"));
        }
        return (
            pack_q4_k_raw(&raw, m, k, false).expect("q4_k pack"),
            Carrier::Q4,
        );
    }
    let rows: Vec<Vec<f32>> = ts
        .iter()
        .zip(names)
        .map(|(t, nm)| dequant_rows(gguf, nm, 0, t.dims[1] as usize).expect("dequant"))
        .collect();
    let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    (pack_proj_f32(carrier, &refs, k, false), carrier)
}

unsafe fn out_buf(vk: &Vk, bytes: usize) -> Handle {
    vk.buffer(
        bytes as u64,
        BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST,
        MEMORY_DEVICE_LOCAL,
    )
    .expect("output buffer")
    .0
}

/// The committed SPIR-V is what `glslc` makes of the committed GLSL.
#[test]
fn coopmat_spirv_is_the_glsl() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/spirv");
    let out = std::env::temp_dir().join(format!("coopmat_gemm_{}.spv", std::process::id()));
    let st = std::process::Command::new("glslc")
        .args(GLSLC_ARGS)
        .arg(dir.join("coopmat_gemm.comp"))
        .arg("-o")
        .arg(&out)
        .status()
        .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
    assert!(st.success(), "glslc failed: {st}");
    let fresh = std::fs::read(&out).expect("glslc output");
    let _ = std::fs::remove_file(&out);
    let committed = std::fs::read(dir.join("coopmat_gemm.spv")).expect("committed spv");
    assert!(
        fresh == committed,
        "src/spirv/coopmat_gemm.spv is stale: recompile with glslc {GLSLC_ARGS:?}"
    );
}

#[test]
fn coopmat_gemm_matches_dp4a_on_qwen_layer0() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    if !Path::new(MODEL).exists() {
        println!("SKIP: Model file {MODEL} not found");
        return;
    }
    assert_eq!(TOKENS % TILE_BN, 0, "the block is padded to the token tile");
    let gguf = Gguf::open(MODEL).expect("gguf open");
    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let lw = local_wg();
    let n = TOKENS as usize;
    let mut rng = Rng(0x00c0_0b4a_7e11_0001);
    let mut cells = 0usize;
    let l0 = |s: &str| format!("blk.0.{s}.weight");
    // the first ffn_down stored as Q4_K: the lossless Q4 chart prefill serves in `--carrier q4`
    let q4k_down = (0..64)
        .map(|l| format!("blk.{l}.ffn_down.weight"))
        .find(|nm| {
            gguf.header
                .tensor(nm)
                .is_some_and(|t| t.ggml_type == GGML_TYPE_Q4_K)
        })
        .expect("a Q4_K ffn_down");
    let groups: [(&str, Vec<String>); 5] = [
        ("qkv", vec![l0("attn_q"), l0("attn_k"), l0("attn_v")]),
        ("o", vec![l0("attn_output")]),
        ("gate_up", vec![l0("ffn_gate"), l0("ffn_up")]),
        ("down", vec![l0("ffn_down")]),
        ("down_q4k", vec![q4k_down]),
    ];
    for mode in [Carrier::Q8, Carrier::Q4, Carrier::Q2] {
        for (name, names) in &groups {
            let (proj, pc) = pack(&gguf, names, mode);
            let (m, k) = (proj.m as u32, proj.k as u32);
            let (mu, ku, blocks) = (proj.m, proj.k, proj.k / 32);
            let words = &proj.words;
            let (wb, dmb) = unsafe { (g.up_u32(&proj.words), g.up_u32(&proj.dm)) };
            let xq: Vec<u32> = (0..n * ku / 4)
                .map(|_| {
                    (0..4).fold(0u32, |a, i| {
                        let c = (rng.next() % 255) as i32 - 127;
                        a | (u32::from(c as i8 as u8) << (8 * i))
                    })
                })
                .collect();
            let sx: Vec<f32> = (0..n * blocks)
                .map(|_| 1e-3 + (rng.next() % 4096) as f32 * 4e-6)
                .collect();
            let xs: Vec<f32> = (0..n * blocks)
                .map(|i| {
                    let (t, b) = (i / blocks, i % blocks);
                    let s: i64 = (0..32).map(|q| xcode(&xq, ku, t, b * 32 + q)).sum();
                    s as f32 * sx[i]
                })
                .collect();
            let unit_dm = vec![0x3c00u32; mu * blocks]; // (d, lo) = (1.0, 0.0) in f16
            let ones = vec![1f32; n * blocks];
            let zeros = vec![0f32; n * blocks];
            let (bxq, bsx, bxs) = unsafe { (g.up_u32(&xq), g.up_f32(&sx), g.up_f32(&xs)) };
            let (bdm1, bone, bzero) =
                unsafe { (g.up_u32(&unit_dm), g.up_f32(&ones), g.up_f32(&zeros)) };
            let (o_dp, o_cm) = unsafe { (out_buf(&vk, n * mu * 4), out_buf(&vk, n * mu * 4)) };
            let (bt, rt) = prefill_tile(m, k, pc, lw);
            let dp_code = gemv_dp4a_batch_rt_bytes(m, k, lw, pc, bt, rt);
            let cm_code = coopmat_gemm_bytes(m, k, pc);
            let (cgx, cgy) = coopmat_gemm_groups(m, TOKENS);
            let mk = |code: &[u8], dm, xs_b, sx_b, out, gx, gy| unsafe {
                make_step_named(&g, code, &[wb, bxq, out, dm, xs_b, sx_b], gx, gy, "gemm")
            };
            let dp_gx = m.div_ceil(lw * rt);
            let dp_gy = TOKENS / bt;
            for (face, dm, xs_b, sx_b) in [("int", bdm1, bzero, bone), ("f32", dmb, bxs, bsx)] {
                unsafe {
                    run(&vk, &mk(&dp_code, dm, xs_b, sx_b, o_dp, dp_gx, dp_gy));
                    run(&vk, &mk(&cm_code, dm, xs_b, sx_b, o_cm, cgx, cgy));
                }
                let a = unsafe { readback(&vk, o_dp, n * mu) };
                let b = unsafe { readback(&vk, o_cm, n * mu) };
                if face == "int" {
                    // CPU integer reference on a sampled row set, all tokens
                    for s in 0..ROW_SAMPLES {
                        let r = if s == 0 {
                            mu - 1
                        } else {
                            rng.next() as usize % mu
                        };
                        let wr: Vec<i64> = (0..ku).map(|p| wcode(words, mu, pc, r, p)).collect();
                        for t in 0..n {
                            let mut tot = 0i64;
                            let mut peak = 0i64;
                            for bk in 0..blocks {
                                let q: i64 = (0..32)
                                    .map(|j| wr[bk * 32 + j] * xcode(&xq, ku, t, bk * 32 + j))
                                    .sum();
                                tot += q;
                                peak = peak.max(tot.abs());
                            }
                            assert!(peak < 1 << 24, "{name}: partial {peak} not f32-exact");
                            let (gd, gc) = (a[t * mu + r], b[t * mu + r]);
                            assert_eq!(gd as i64, tot, "{name} {pc:?}: dp4a r{r} t{t}");
                            assert_eq!(gc as i64, tot, "{name} {pc:?}: coopmat r{r} t{t}");
                        }
                    }
                }
                let mut diff = 0usize;
                let mut first = None;
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    if x.to_bits() != y.to_bits() {
                        diff += 1;
                        first.get_or_insert((i, *x, *y));
                    }
                }
                println!(
                    "{name:8} {mode:?}/{pc:?} m={m} k={k} {face}: {} cells, {diff} differ{}",
                    a.len(),
                    first.map_or(String::new(), |(i, x, y)| format!(
                        " (first at token {} row {}: dp4a {x:e} coopmat {y:e})",
                        i / mu,
                        i % mu
                    ))
                );
                assert_eq!(diff, 0, "{name} {pc:?} {face}: coopmat != dp4a");
                cells += a.len();
            }
        }
    }
    println!("all {cells} cells equal bit for bit");
}
