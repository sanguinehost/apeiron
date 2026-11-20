//! Synthetic benchmark of the served GPU decode GEMV and prefill batch GEMV.
//!
//! One projection of the Qwen2.5-7B `ffn_down` shape (`m = 3584`, `k = 18944`), packed to the
//! `Carrier::Q4` chart by the real `pack_proj_f32`. The decode path is the integer-dot GEMV the
//! decoder serves (`gemv_dp4a_dyn_bytes`, f16-packed `(d, lo)` and a per-block activation scale),
//! with the production split-K policy `segs(m, k/32)` swept around its value. The prefill path is
//! `gemv_dp4a_batch_bytes`, which reuses one weight read across `bt` tokens.
//!
//! Every configuration is checked before it is timed: an integer face (unit `(d, lo)`, codes in
//! `[-7, 7]`) must equal the exact integer dot, and the real chart face is compared with an f64
//! evaluation of `d·q + lo` over the same packed words.
//!
//! Run: `cargo run --release --locked --features gpu --example gpu_gemv_bench`
//! Flags: `--quick` (one segment and one batch tile), `--no-prefill`, `--no-float`.

#![cfg(feature = "gpu")]
#![allow(
    clippy::pedantic,
    missing_docs,
    reason = "GPU benchmark harness: the measured result is the artifact, and the arithmetic casts \
              are the chart's own"
)]

use apeiron::format::quant::f16_to_f32;
use apeiron::gpu::decode::pack::pack_proj_f32;
use apeiron::gpu::simt::{gemv_bytes, gemv_dp4a_batch_bytes, gemv_dp4a_dyn_bytes, Carrier};
use apeiron::gpu::vulkan::{
    check, BufferCopy, Handle, ACCESS_HOST_READ, ACCESS_TRANSFER_WRITE, BUFFER_USAGE_STORAGE,
    BUFFER_USAGE_TRANSFER_DST, BUFFER_USAGE_TRANSFER_SRC, MEMORY_DEVICE_LOCAL, STAGE_HOST,
    STAGE_TRANSFER,
};
use apeiron::gpu::{Buffer, Dispatch, Gpu};
use std::time::Instant;

const M: usize = 3584; // Qwen2.5-7B ffn_down rows
const K: usize = 18944; // Qwen2.5-7B ffn_down columns
const LOCAL: u32 = 256;
const BATCH_TOKENS: usize = 512;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn f32_sym(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
    fn i8_code(&mut self) -> i8 {
        ((self.next() % 255) as i32 - 127) as i8
    }
}

/// DP4A phase order, not consecutive order.
///
/// A Q4 weight word holds eight codes in nibbles; phase `l` masks nibble lane `l` of each byte, so
/// the four byte lanes carry logical codes `8c + 2b + l` (`b = 0..3`, `dpw = 2`). The activation
/// word `2c + l` must pack exactly those four codes. A consecutive `codes[4w..4w+4]` pack pairs
/// phase 0 with codes 0..3 and every DP4A against the wrong activation lane.
fn xq_q4_of(codes: &[i8]) -> Vec<u32> {
    codes
        .as_chunks::<8>()
        .0
        .iter()
        .flat_map(|g| {
            (0..2).map(move |l| {
                u32::from(g[l] as u8)
                    | (u32::from(g[2 + l] as u8) << 8)
                    | (u32::from(g[4 + l] as u8) << 16)
                    | (u32::from(g[6 + l] as u8) << 24)
            })
        })
        .collect()
}

/// Transpose the DP4A chart words into the layout the scalar `gemv_bytes` chart decoder reads.
///
/// DP4A stores `words[(kk/8)·M + r]` with nibble `kk%8`; `gemv_module` reads
/// `words[kk·(M/8) + r/8]` with nibble `r%8`. The two hold the same code set in different axes:
/// the float path is element-major and row-grouped, the DP4A path is word-major and row-flat.
///
/// Time O(k·M), space O(k·M/8) words.
fn float_words_of(words: &[u32], m: usize, k: usize) -> Vec<u32> {
    let groups = m / 8;
    let mut out = vec![0u32; k * groups];
    for kk in 0..k {
        let shift = (kk % 8) as u32 * 4;
        let src = (kk / 8) * m;
        for r0 in 0..groups {
            let mut word = 0u32;
            for l in 0..8 {
                let q = (words[src + r0 * 8 + l] >> shift) & 0xF;
                word |= q << (l as u32 * 4);
            }
            out[kk * groups + r0] = word;
        }
    }
    out
}

fn xs_of(codes: &[i8]) -> Vec<f32> {
    codes
        .as_chunks::<32>()
        .0
        .iter()
        .map(|b| b.iter().map(|&c| c as f32).sum())
        .collect()
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn min_max(v: &[f64]) -> (f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (s[0], s[s.len() - 1])
}

fn output_buffer(g: &Gpu, floats: usize) -> Buffer {
    let (buffer, memory) = unsafe {
        g.vk.buffer(
            (floats * 4) as u64,
            BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST,
            MEMORY_DEVICE_LOCAL,
        )
        .expect("output buffer")
    };
    Buffer { buffer, memory }
}

unsafe fn readback(g: &Gpu, src: Handle, n: usize) -> Vec<f32> {
    let (rb, ptr) = g.host_read_buffer(n * 4);
    let cb = g.vk.command_buffer();
    g.vk.begin(cb);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: (n * 4) as u64,
    };
    (g.vk.copy_buffer)(cb, src, rb.buffer, 1, &region);
    g.vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((g.vk.end_cb)(cb), "vkEndCommandBuffer");
    g.vk.submit_wait(cb);
    std::slice::from_raw_parts(ptr.cast::<f32>(), n).to_vec()
}

/// The chart's own f64 evaluation: `Σ_blocks (d·q + lo)·x`, over the packed words.
fn cpu_chart(words: &[u32], dm: &[u32], codes: &[i8], m: usize, k: usize, r: usize) -> f64 {
    let mut acc = 0f64;
    for b in 0..k / 32 {
        let dmw = dm[b * m + r];
        let d = f64::from(f16_to_f32((dmw & 0xffff) as u16));
        let lo = f64::from(f16_to_f32((dmw >> 16) as u16));
        for j in 0..32 {
            let kk = b * 32 + j;
            let q = ((words[(kk / 8) * m + r] >> ((kk % 8) * 4)) & 0xF) as f64;
            acc += (d * q + lo) * f64::from(codes[kk]);
        }
    }
    acc
}

/// Exact integer dot of the chart codes, for the unit-scale face.
fn cpu_int(words: &[u32], codes: &[i8], m: usize, k: usize, r: usize) -> i64 {
    let mut acc = 0i64;
    for kk in 0..k {
        let q = i64::from((words[(kk / 8) * m + r] >> ((kk % 8) * 4)) & 0xF);
        acc += q * i64::from(codes[kk]);
    }
    acc
}

fn report(label: &str, us: &[f64], ms: &[f64], weight_bytes: u64, all_bytes: u64, tokens: f64) {
    let have_ts = us.iter().all(|v| v.is_finite() && *v > 0.0);
    let effective: Vec<f64> = if have_ts {
        us.to_vec()
    } else {
        ms.iter().map(|v| v * 1000.0).collect()
    };
    let (ts_med, ts_lo, ts_hi) = (
        median(&effective),
        min_max(&effective).0,
        min_max(&effective).1,
    );
    let (ms_med, ms_lo, ms_hi) = (median(ms), min_max(ms).0, min_max(ms).1);
    let secs = ts_med * 1e-6;
    println!(
        "{label} gpu_us={ts_med:.3} [{ts_lo:.3}..{ts_hi:.3}] wall_ms={ms_med:.3} \
[{ms_lo:.3}..{ms_hi:.3}] weight_GBps={:.2} all_GBps={:.2} tok_s={:.1}",
        weight_bytes as f64 / secs / 1e9,
        all_bytes as f64 / secs / 1e9,
        tokens / secs,
    );
    println!(
        "{label} samples_us={:?} samples_ms={:?} timestamps={have_ts}",
        us.iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>(),
        ms.iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>(),
    );
}

unsafe fn run_timed(
    g: &Gpu,
    disps: &[Dispatch],
    warmup: usize,
    reps: usize,
) -> (Vec<f64>, Vec<f64>) {
    for _ in 0..warmup {
        g.run(disps);
    }
    let mut us = Vec::with_capacity(reps);
    let mut ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let u = g.time(disps);
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        us.push(u);
    }
    (us, ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let no_prefill = args.iter().any(|a| a == "--no-prefill");
    let no_float = args.iter().any(|a| a == "--no-float");

    let g = Gpu::new().expect("Vulkan device");
    println!("device: {}", g.device_name());
    println!(
        "shape: m={M} k={K} carrier=Q4 local={LOCAL} batch_tokens={BATCH_TOKENS} quick={quick}"
    );

    let mut rng = Rng(0x00c0_0b4a_7e11_0b01);
    let mut weights = vec![0f32; M * K];
    for w in &mut weights {
        *w = rng.f32_sym();
    }
    let rows: Vec<&[f32]> = (0..M).map(|r| &weights[r * K..(r + 1) * K]).collect();
    let proj = pack_proj_f32(Carrier::Q4, &rows, K, false);
    let words = proj.words.clone();
    let dm = proj.dm.clone();
    let blocks = K / 32;
    let words_bytes = (words.len() * 4) as u64;
    let dm_bytes = (dm.len() * 4) as u64;
    let weight_bytes = words_bytes + dm_bytes;
    println!(
        "pack: weights={} words_bytes={words_bytes} dm_bytes={dm_bytes} total={weight_bytes} \
bits_per_weight={:.4}",
        M * K,
        weight_bytes as f64 * 8.0 / (M * K) as f64
    );
    assert_eq!(words.len(), M * K / 8);
    assert_eq!(dm.len(), M * blocks);

    // The real activation face: one token's int8 codes, unit scale.
    let codes: Vec<i8> = (0..K).map(|_| rng.i8_code()).collect();
    let xq = xq_q4_of(&codes);
    let xs = xs_of(&codes);
    let sx = vec![1f32; blocks];

    // The integer face: codes in [-7, 7] keep every f32 partial below 2^24.
    let int_codes: Vec<i8> = (0..K).map(|_| (rng.next() % 15) as i8 - 7).collect();
    let int_xq = xq_q4_of(&int_codes);
    let int_xs = vec![0f32; blocks];
    let unit_dm: Vec<u32> = vec![0x3c00u32; M * blocks]; // (d, lo) = (1.0, 0.0)

    let wb = g.upload(&words);
    let dmb = g.upload(&dm);
    let unit_dmb = g.upload(&unit_dm);
    let xqb = g.upload(&xq);
    let xsb = g.upload(&xs);
    let sxb = g.upload(&sx);
    let int_xqb = g.upload(&int_xq);
    let int_xsb = g.upload(&int_xs);

    // ── decode: the production dyn-sx DP4A gemv + split-K stage 2 ──────────────────────────
    let seg_candidates: Vec<u32> = if quick { vec![8] } else { vec![1, 2, 4, 8, 16] };
    for &segs in &seg_candidates {
        let out = output_buffer(&g, M);
        let partial = output_buffer(&g, segs as usize * M);
        let first = if segs > 1 { &partial } else { &out };
        let pipe1 = g.pipeline(
            &gemv_dp4a_dyn_bytes(M as u32, K as u32, LOCAL, segs, Carrier::Q4),
            6,
        );
        let ds1 = g.bind(&pipe1, &[&wb, &xqb, first, &dmb, &xsb, &sxb]);
        let d1 = Dispatch {
            pipe: pipe1,
            dset: ds1,
            groups: ((M as u32).div_ceil(LOCAL), segs),
        };
        let mut disps = vec![d1];
        if segs > 1 {
            let ones = g.upload(&vec![1f32; segs as usize]);
            let pipe2 = g.pipeline(&gemv_bytes(M as u32, segs, LOCAL, 1, Carrier::F32), 3);
            let ds2 = g.bind(&pipe2, &[&partial, &ones, &out]);
            disps.push(Dispatch {
                pipe: pipe2,
                dset: ds2,
                groups: ((M as u32).div_ceil(LOCAL), 1),
            });
        }
        // f32 face check — submit first: `run_timed` below is the only other caller of `g.run`,
        // and it runs after this check, so without this line the readback sees an unwritten buffer.
        g.run(&disps);
        let got = unsafe { readback(&g, out.buffer, M) };
        let mut worst = 0f64;
        let mut scale = 0f64;
        for &r in &[0usize, M / 2, M - 1] {
            let want = cpu_chart(&words, &dm, &codes, M, K, r);
            let d = (f64::from(got[r]) - want).abs();
            worst = worst.max(d);
            scale = scale.max(want.abs());
            println!(
                "check decode segs={segs} row={r} got={:e} want={want:e} diff={d:e}",
                got[r]
            );
        }
        let rel = worst / scale.max(1e-30);
        assert!(rel <= 1e-4, "decode segs={segs}: rel {rel:e} > 1e-4");
        println!("check decode segs={segs} f32 max_rel={rel:e}");

        // integer face
        let out_i = output_buffer(&g, M);
        let partial_i = output_buffer(&g, segs as usize * M);
        let first_i = if segs > 1 { &partial_i } else { &out_i };
        let unit_pipe = g.pipeline(
            &gemv_dp4a_dyn_bytes(M as u32, K as u32, LOCAL, segs, Carrier::Q4),
            6,
        );
        let unit_ds = g.bind(
            &unit_pipe,
            &[&wb, &int_xqb, first_i, &unit_dmb, &int_xsb, &sxb],
        );
        let mut unit = vec![Dispatch {
            pipe: unit_pipe,
            dset: unit_ds,
            groups: ((M as u32).div_ceil(LOCAL), segs),
        }];
        if segs > 1 {
            let ones = g.upload(&vec![1f32; segs as usize]);
            let pipe2 = g.pipeline(&gemv_bytes(M as u32, segs, LOCAL, 1, Carrier::F32), 3);
            let ds2 = g.bind(&pipe2, &[&partial_i, &ones, &out_i]);
            unit.push(Dispatch {
                pipe: pipe2,
                dset: ds2,
                groups: ((M as u32).div_ceil(LOCAL), 1),
            });
        }
        g.run(&unit);
        let got_i = unsafe { readback(&g, out_i.buffer, M) };
        for &r in &[0usize, M / 2, M - 1] {
            let want = cpu_int(&words, &int_codes, M, K, r);
            let gotv = got_i[r] as i64;
            assert_eq!(gotv, want, "decode segs={segs} int row={r}");
        }
        println!("check decode segs={segs} int exact rows=3");

        let act_bytes = (K + blocks * 4 * 2) as u64;
        let all_bytes = weight_bytes
            + act_bytes
            + (M * 4) as u64
            + if segs > 1 {
                (2 * segs as usize * M * 4) as u64
            } else {
                0
            };
        let (us, ms) = unsafe { run_timed(&g, &disps, 5, 5) };
        report(
            &format!("decode segs={segs}"),
            &us,
            &ms,
            weight_bytes,
            all_bytes,
            1.0,
        );
    }

    // ── prefill: one weight read feeds bt tokens ─────────────────────────────────────────────
    if !no_prefill {
        let b = BATCH_TOKENS;
        let mut bxq = Vec::with_capacity(b * K / 4);
        let mut bxs = Vec::with_capacity(b * blocks);
        let mut bsx = Vec::with_capacity(b * blocks);
        let mut bcodes = vec![0i8; b * K];
        for t in 0..b {
            for kk in 0..K {
                bcodes[t * K + kk] = rng.i8_code();
            }
            let row = &bcodes[t * K..(t + 1) * K];
            bxq.extend_from_slice(&xq_q4_of(row));
            bxs.extend_from_slice(&xs_of(row));
            bsx.extend_from_slice(&vec![1f32; blocks]);
        }
        let bxqb = g.upload(&bxq);
        let bxsb = g.upload(&bxs);
        let bsxb = g.upload(&bsx);

        let tiles: Vec<u32> = if quick { vec![32] } else { vec![8, 16, 32, 64] };
        for &bt in &tiles {
            assert_eq!(b as u32 % bt, 0, "batch tokens must divide the tile");
            let out = output_buffer(&g, b * M);
            let pipe = g.pipeline(
                &gemv_dp4a_batch_bytes(M as u32, K as u32, LOCAL, Carrier::Q4, bt),
                6,
            );
            let ds = g.bind(&pipe, &[&wb, &bxqb, &out, &dmb, &bxsb, &bsxb]);
            let disps = [Dispatch {
                pipe,
                dset: ds,
                groups: ((M as u32).div_ceil(LOCAL), b as u32 / bt),
            }];
            // check six cells: tokens 0, 1, b-1 at rows 0 and M-1
            g.run(&disps);
            let got = unsafe { readback(&g, out.buffer, b * M) };
            let mut worst = 0f64;
            let mut scale = 0f64;
            for &t in &[0usize, 1, b - 1] {
                for &r in &[0usize, M - 1] {
                    let want = cpu_chart(&words, &dm, &bcodes[t * K..(t + 1) * K], M, K, r);
                    let d = (f64::from(got[t * M + r]) - want).abs();
                    worst = worst.max(d);
                    scale = scale.max(want.abs());
                }
            }
            let rel = worst / scale.max(1e-30);
            assert!(rel <= 1e-4, "prefill bt={bt}: rel {rel:e} > 1e-4");
            println!("check prefill bt={bt} f32 max_rel={rel:e} cells=6");
            let weight_read = weight_bytes * (b as u64 / bt as u64);
            let act_read = (M as u64 / LOCAL as u64) * b as u64 * K as u64;
            let all_read = weight_read + act_read + (b * M * 4) as u64;
            let (us, ms) = unsafe { run_timed(&g, &disps, 5, 5) };
            report(
                &format!("prefill bt={bt} tokens={b}"),
                &us,
                &ms,
                weight_read,
                all_read,
                b as f64,
            );
        }
    }

    // ── the float-carrier control: the same chart through gemv_bytes ─────────────────────────
    if !no_float {
        let segs = 8u32;
        // `gemv_module`'s chart decode is element-major; the DP4A words are word-major. Repack
        // once — the bytes on the wire are the same, and `float_weight_bytes` counts the same words.
        let fwords = float_words_of(&words, M, K);
        let fwb = g.upload(&fwords);
        let dmf = g.upload(&proj.dmf);
        let xf = g.upload(&codes.iter().map(|&c| c as f32).collect::<Vec<_>>());
        let out = output_buffer(&g, M);
        let partial = output_buffer(&g, segs as usize * M);
        let pipe1 = g.pipeline(&gemv_bytes(M as u32, K as u32, LOCAL, segs, Carrier::Q4), 5);
        let ds1 = g.bind(&pipe1, &[&fwb, &xf, &partial, &dmf, &xsb]);
        let ones = g.upload(&vec![1f32; segs as usize]);
        let pipe2 = g.pipeline(&gemv_bytes(M as u32, segs, LOCAL, 1, Carrier::F32), 3);
        let ds2 = g.bind(&pipe2, &[&partial, &ones, &out]);
        let disps = [
            Dispatch {
                pipe: pipe1,
                dset: ds1,
                groups: ((M as u32).div_ceil(LOCAL), segs),
            },
            Dispatch {
                pipe: pipe2,
                dset: ds2,
                groups: ((M as u32).div_ceil(LOCAL), 1),
            },
        ];
        g.run(&disps);
        let got = unsafe { readback(&g, out.buffer, M) };
        let want = cpu_chart(&words, &dm, &codes, M, K, 0);
        let rel = (f64::from(got[0]) - want).abs() / want.abs().max(1e-30);
        assert!(rel <= 1e-4, "float gemv: rel {rel:e} > 1e-4");
        println!("check float segs={segs} f32 rel={rel:e}");
        let float_weight_bytes = words_bytes + (2 * M * blocks * 4) as u64;
        let all_bytes = float_weight_bytes + (K * 4 + blocks * 4 * 2) as u64 + (M * 4) as u64;
        let (us, ms) = unsafe { run_timed(&g, &disps, 5, 5) };
        report(
            &format!("float segs={segs}"),
            &us,
            &ms,
            float_weight_bytes,
            all_bytes,
            1.0,
        );
    }
}
