// Split-K down GEMM + reduction against the unsplit coopmat GEMM.
//! The split-K coopmat down GEMM (`coopmat_gemm_split_bytes`, grid z = splits) followed by
//! `splitk_reduce_bytes`, on the real Qwen2.5-0.5B layer-0 `ffn_down` (Q8) and the first `Q4_K` `ffn_down`
//! (Q4), `m = 896`, `k = 4864`, over a 512-token random int8 activation block.
//!
//! 1. **integer** (unit scales: each output is `Σ_b q_b`, an integer the f32 sum holds exactly while
//!    every partial is below 2^24): the split sums equal the unsplit kernel and the CPU integer dot of the
//!    chart codes on a sampled row set, for S = 2, 3, 5 (every K range counted exactly once).
//! 2. **f32** (the real `(d, lo)` chart, random `sx`/`xs`): S = 3 reassociates the per-32-block serial
//!    sum into three chains added `(p0 + p1) + p2`; the largest |difference| from the unsplit output is
//!    printed and bounded.
#![cfg(feature = "gpu")]
#![allow(
    clippy::pedantic,
    missing_docs,
    reason = "GPU integration test; the lint profile of the other gpu_* tests"
)]

use apeiron::engine::testdata::model_path;
use apeiron::format::gguf::Gguf;
use apeiron::format::quant::GGML_TYPE_Q4_K;
use apeiron::gpu::decode::pack::{pack_proj_f32, pack_q4_k_raw, Proj};
use apeiron::gpu::decode::state::dequant_rows;
use apeiron::gpu::decode::{local_wg, make_step_named, rec_step, GpuCtx, Step};
use apeiron::gpu::nn::{
    coopmat_gemm_bytes, coopmat_gemm_groups, coopmat_gemm_split_bytes, splitk_reduce_bytes,
    REDUCE_GX,
};
use apeiron::gpu::simt::Carrier;
use apeiron::gpu::vulkan::*;
use std::sync::Mutex;

static VK_LOAD_LOCK: Mutex<()> = Mutex::new(());

const MODEL: &str = "models/gguf/qwen05b-Q4_K_M.gguf";
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

/// Partials then reduction: the split GEMM at grid z = `ks`, the reduce over its `[ks][rows][m]` tile.
unsafe fn run_split(vk: &Vk, gemm: &Step, red: &Step, ks: u32) {
    let cb = vk.command_buffer();
    vk.begin(cb);
    (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, gemm.pipe);
    (vk.bind_sets)(
        cb,
        PIPELINE_BIND_POINT_COMPUTE,
        gemm.layout,
        0,
        1,
        &gemm.dset,
        0,
        std::ptr::null(),
    );
    (vk.dispatch)(cb, gemm.groups.0, gemm.groups.1, ks);
    vk.barrier(
        cb,
        STAGE_COMPUTE,
        STAGE_COMPUTE,
        ACCESS_SHADER_WRITE,
        ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
    );
    rec_step(vk, cb, red);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
}

#[test]
fn splitk_down_gemm_sums_every_k_range_once() {
    let vk = {
        let _guard = VK_LOAD_LOCK.lock().unwrap();
        match unsafe { Vk::load() } {
            Ok(vk) => vk,
            Err(e) => {
                println!("SKIP: No Vulkan device present: {e}");
                return;
            }
        }
    };
    if !vk.has_coopmat {
        println!("SKIP: no cooperative matrix");
        return;
    }
    let Some(model) = model_path(MODEL) else {
        println!("SKIP: Model file {MODEL} not found");
        return;
    };
    let gguf = Gguf::open(&model).expect("gguf open");
    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let _ = local_wg();
    let n = TOKENS as usize;
    let mut rng = Rng(0x5b17_0c0f_fee1_0002);
    let q4k_down = (0..64)
        .map(|l| format!("blk.{l}.ffn_down.weight"))
        .find(|nm| {
            gguf.header
                .tensor(nm)
                .is_some_and(|t| t.ggml_type == GGML_TYPE_Q4_K)
        })
        .expect("a Q4_K ffn_down");
    let groups: [(Carrier, Vec<String>); 2] = [
        (Carrier::Q8, vec!["blk.0.ffn_down.weight".to_string()]),
        (Carrier::Q4, vec![q4k_down]),
    ];
    for (mode, names) in &groups {
        let (proj, pc) = pack(&gguf, names, *mode);
        let (m, k) = (proj.m as u32, proj.k as u32);
        let (mu, ku, blocks) = (proj.m, proj.k, proj.k / 32);
        assert_eq!((m, k), (896, 4864), "the pp512 down shape");
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
        let unit_dm = vec![0x3c00u32; mu * blocks];
        let ones = vec![1f32; n * blocks];
        let zeros = vec![0f32; n * blocks];
        let (bxq, bsx, bxs) = unsafe { (g.up_u32(&xq), g.up_f32(&sx), g.up_f32(&xs)) };
        let (bdm1, bone, bzero) =
            unsafe { (g.up_u32(&unit_dm), g.up_f32(&ones), g.up_f32(&zeros)) };
        let o_un = unsafe { out_buf(&vk, n * mu * 4) };
        let (cgx, cgy) = coopmat_gemm_groups(m, TOKENS);
        let un_code = coopmat_gemm_bytes(m, k, pc);
        for (face, dm, xs_b, sx_b) in [("int", bdm1, bzero, bone), ("f32", dmb, bxs, bsx)] {
            unsafe {
                run(
                    &vk,
                    &make_step_named(
                        &g,
                        &un_code,
                        &[wb, bxq, o_un, dm, xs_b, sx_b],
                        cgx,
                        cgy,
                        "un",
                    ),
                );
            }
            let a = unsafe { readback(&vk, o_un, n * mu) };
            let splits: &[u32] = if face == "int" { &[2, 3, 5] } else { &[3] };
            for &ks in splits {
                let o_part = unsafe { out_buf(&vk, ks as usize * n * mu * 4) };
                let o_sp = unsafe { out_buf(&vk, n * mu * 4) };
                let sp_code = coopmat_gemm_split_bytes(m, k, pc, 0, 0, ks);
                let rd_code = splitk_reduce_bytes(m, ks);
                let gemm = unsafe {
                    make_step_named(
                        &g,
                        &sp_code,
                        &[wb, bxq, o_part, dm, xs_b, sx_b],
                        cgx,
                        cgy,
                        "split",
                    )
                };
                let red = unsafe {
                    make_step_named(&g, &rd_code, &[o_part, o_sp], REDUCE_GX, cgy, "reduce")
                };
                unsafe { run_split(&vk, &gemm, &red, ks) };
                let b = unsafe { readback(&vk, o_sp, n * mu) };
                if face == "int" {
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
                            assert!(peak < 1 << 24, "partial {peak} not f32-exact");
                            assert_eq!(b[t * mu + r] as i64, tot, "{pc:?} S={ks}: r{r} t{t}");
                        }
                    }
                    let diff = a
                        .iter()
                        .zip(&b)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    println!(
                        "{pc:?} int S={ks}: {} cells, {diff} differ from unsplit",
                        a.len()
                    );
                    assert_eq!(
                        diff, 0,
                        "{pc:?} S={ks}: split != unsplit on the exact integer face"
                    );
                } else {
                    let peak = a.iter().fold(0f32, |m, x| m.max(x.abs()));
                    let (mut dmax, mut ndiff, mut sum_d, mut sum_r) = (0f32, 0usize, 0f64, 0f64);
                    for (x, y) in a.iter().zip(&b) {
                        let d = (x - y).abs();
                        if x.to_bits() != y.to_bits() {
                            ndiff += 1;
                        }
                        dmax = dmax.max(d);
                        sum_d += d as f64;
                        sum_r += (x.abs() as f64).max(1e-30);
                    }
                    println!(
                        "{pc:?} f32 S={ks}: {} cells, {ndiff} differ; max|unsplit| {peak:e}, max abs diff {dmax:e} \
                         (rel to max {:e}), mean abs diff {:e}, sum|d|/sum|x| {:e}",
                        a.len(),
                        dmax / peak,
                        sum_d / a.len() as f64,
                        sum_d / sum_r,
                    );
                    assert!(
                        dmax <= 1e-5 * peak,
                        "{pc:?} S={ks}: split differs by more than f32 reassociation"
                    );
                }
            }
        }
    }
}
