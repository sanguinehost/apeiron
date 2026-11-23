//! Localise the zero-output failure of the served GPU decode GEMV.
//!
//! Packs a small `m × k` chart with known codes, uploads it in the bench's six-buffer order, and
//! dispatches `gemv_dp4a_dyn_bytes(m, k, 256, segs, Carrier::Q4)` at `segs = 1` and `segs = 8`
//! (the production split-K shape). It prints the module's entry point, local size and descriptor
//! bindings, then the first eight outputs beside the CPU reference. A direct `Carrier::F32`
//! `gemv_bytes` dispatch is the control: three buffers, no chart, no DP4A.
//!
//! Run: `cargo run --release --locked --features gpu --example gpu_gemv_debug [--segs N]`

#![cfg(feature = "gpu")]
#![allow(
    clippy::pedantic,
    missing_docs,
    reason = "GPU debug harness: the printed values are the artifact"
)]

use apeiron::format::quant::f16_to_f32;
use apeiron::gpu::decode::pack::f32_to_f16;
use apeiron::gpu::simt::{gemv_bytes, gemv_dp4a_dyn_bytes, Carrier};
use apeiron::gpu::vulkan::{
    check, BufferCopy, Handle, ACCESS_HOST_READ, ACCESS_TRANSFER_WRITE, BUFFER_USAGE_STORAGE,
    BUFFER_USAGE_TRANSFER_DST, BUFFER_USAGE_TRANSFER_SRC, MEMORY_DEVICE_LOCAL, STAGE_HOST,
    STAGE_TRANSFER,
};
use apeiron::gpu::{Buffer, Dispatch, Gpu};

const LOCAL: u32 = 256;
const SPV_ENTRY_POINT: u32 = 15;
const SPV_EXECUTION_MODE: u32 = 16;
const SPV_DECORATE: u32 = 71;
const SPV_VARIABLE: u32 = 59;
const DEC_BINDING: u32 = 33;
const MODE_LOCAL_SIZE: u32 = 17;

/// The bits of a module a runtime can get wrong: entry name, local size, binding numbers.
fn describe(spv: &[u8], label: &str) {
    let words: Vec<u32> = spv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let mut entry = String::from("<none>");
    let mut local = String::from("<none>");
    let mut bindings: Vec<(u32, u32)> = Vec::new();
    let mut variables: Vec<u32> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        if wc == 0 || i + wc > words.len() {
            println!("{label}: malformed at word {i}");
            return;
        }
        let op = words[i] & 0xffff;
        match op {
            SPV_ENTRY_POINT => {
                let name_words = &words[i + 3..i + wc];
                let bytes: Vec<u8> = name_words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .take_while(|&b| b != 0)
                    .collect();
                entry = String::from_utf8_lossy(&bytes).into_owned();
            }
            SPV_EXECUTION_MODE if wc >= 6 && words[i + 2] == MODE_LOCAL_SIZE => {
                local = format!("{}x{}x{}", words[i + 3], words[i + 4], words[i + 5]);
            }
            SPV_DECORATE if wc == 4 && words[i + 2] == DEC_BINDING => {
                bindings.push((words[i + 1], words[i + 3]));
            }
            SPV_VARIABLE => variables.push(words[i + 2]),
            _ => {}
        }
        i += wc;
    }
    bindings.sort_by_key(|&(_, b)| b);
    println!(
        "{label}: words={} entry={entry:?} local={local} bindings={bindings:?} variables={variables:?}",
        words.len()
    );
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

/// DP4A phase order, not consecutive order.
///
/// One Q4 weight word holds eight codes in nibbles; phase `l` masks nibble lane `l` of each byte,
/// so the four byte lanes carry logical codes `8c + 2b + l` (`b = 0..3`, `dpw = 2`). The matching
/// activation word `2c + l` must therefore hold exactly those four codes. A consecutive
/// `codes[4w..4w+4]` pack pairs phase 0 with codes 0..3 and every dot is against the wrong lane.
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

/// `words[(kk/8)·m + r]` nibble plane and `dm[b·m + r] = f16(d) | f16(lo)<<16`, the served layout.
struct Chart {
    words: Vec<u32>,
    dm: Vec<u32>,
    w: Vec<f32>,
}

fn chart_of(m: usize, k: usize, unit: bool) -> Chart {
    let blocks = k / 32;
    let mut words = vec![0u32; (k / 8) * m];
    let mut dm = vec![0u32; blocks * m];
    let mut w = vec![0f32; m * k];
    for r in 0..m {
        for b in 0..blocks {
            let (d, lo) = if unit {
                (1.0f32, 0.0f32)
            } else {
                (
                    f16_to_f32(f32_to_f16(0.25 * (1 + (b * 3 + r) % 7) as f32)),
                    f16_to_f32(f32_to_f16(-0.5 + 0.125 * ((b + r) % 9) as f32)),
                )
            };
            dm[b * m + r] = (f32_to_f16(d) as u32) | ((f32_to_f16(lo) as u32) << 16);
            for j in 0..32 {
                let q = ((r * 7 + (b * 32 + j) * 3) % 16) as u32;
                let kk = b * 32 + j;
                words[(kk / 8) * m + r] |= q << ((kk % 8) as u32 * 4);
                w[r * k + kk] = d * q as f32 + lo;
            }
        }
    }
    Chart { words, dm, w }
}

fn chart_ref(chart: &Chart, codes: &[i8], m: usize, k: usize, r: usize) -> f64 {
    let mut acc = 0f64;
    for b in 0..k / 32 {
        let word = chart.dm[b * m + r];
        let d = f64::from(f16_to_f32((word & 0xffff) as u16));
        let lo = f64::from(f16_to_f32((word >> 16) as u16));
        for j in 0..32 {
            let kk = b * 32 + j;
            let q = ((chart.words[(kk / 8) * m + r] >> ((kk % 8) * 4)) & 0xF) as f64;
            acc += (d * q + lo) * f64::from(codes[kk]);
        }
    }
    acc
}

fn print_check(label: &str, got: &[f32], want: &[f64], m: usize) -> f64 {
    let mut worst = 0f64;
    let mut scale = 0f64;
    for r in 0..m {
        worst = worst.max((f64::from(got[r]) - want[r]).abs());
        scale = scale.max(want[r].abs());
    }
    for r in 0..8 {
        println!(
            "{label} row={r} got={:e} want={:e} diff={:e}",
            got[r],
            want[r],
            (f64::from(got[r]) - want[r]).abs()
        );
    }
    let rel = worst / scale.max(1e-30);
    println!("{label} max_rel={rel:e} over m={m}");
    rel
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let segs: u32 = args
        .iter()
        .position(|a| a == "--segs")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let (m, k) = (256usize, 1024usize);
    let blocks = k / 32;
    let g = Gpu::new().expect("Vulkan device");
    println!("device: {}", g.device_name());
    println!("shape: m={m} k={k} local={LOCAL} segs={segs}");

    let chart = chart_of(m, k, false);
    let unit = chart_of(m, k, true);
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (rng >> 33) as u32
    };
    let codes: Vec<i8> = (0..k).map(|_| (next() % 15) as i8 - 7).collect();
    let xq = xq_q4_of(&codes);
    let xs: Vec<f32> = codes
        .as_chunks::<32>()
        .0
        .iter()
        .map(|c| c.iter().map(|&v| v as f32).sum())
        .collect();
    let sx = vec![1f32; blocks];
    let xf: Vec<f32> = codes.iter().map(|&c| c as f32).collect();

    let wb = g.upload(&chart.words);
    let dmb = g.upload(&chart.dm);
    let xqb = g.upload(&xq);
    let xsb = g.upload(&xs);
    let sxb = g.upload(&sx);
    let unit_dmb = g.upload(&unit.dm);

    let want_f32: Vec<f64> = (0..m).map(|r| chart_ref(&chart, &codes, m, k, r)).collect();
    let want_int: Vec<f64> = (0..m).map(|r| chart_ref(&unit, &codes, m, k, r)).collect();

    // ── Q4 dyn, split-K as the bench builds it ──
    let spv = gemv_dp4a_dyn_bytes(m as u32, k as u32, LOCAL, segs, Carrier::Q4);
    describe(&spv, "dp4a_dyn Q4");
    let out = output_buffer(&g, m);
    let partial = output_buffer(&g, segs as usize * m);
    let first = if segs > 1 { &partial } else { &out };
    let pipe = g.pipeline(&spv, 6);
    let ds = g.bind(&pipe, &[&wb, &xqb, first, &dmb, &xsb, &sxb]);
    let mut disps = vec![Dispatch {
        pipe,
        dset: ds,
        groups: ((m as u32).div_ceil(LOCAL), segs),
    }];
    if segs > 1 {
        let ones = g.upload(&vec![1f32; segs as usize]);
        let spv2 = gemv_bytes(m as u32, segs, LOCAL, 1, Carrier::F32);
        describe(&spv2, "reduce F32");
        let pipe2 = g.pipeline(&spv2, 3);
        let ds2 = g.bind(&pipe2, &[&partial, &ones, &out]);
        disps.push(Dispatch {
            pipe: pipe2,
            dset: ds2,
            groups: ((m as u32).div_ceil(LOCAL), 1),
        });
    }
    g.run(&disps);
    let got = unsafe { readback(&g, out.buffer, m) };
    let rel = print_check("q4_chart", &got, &want_f32, m);

    // ── the integer face: unit (d, lo), exact Σ q·x ──
    let out_i = output_buffer(&g, m);
    let partial_i = output_buffer(&g, segs as usize * m);
    let first_i = if segs > 1 { &partial_i } else { &out_i };
    let unit_pipe = g.pipeline(&spv, 6);
    let unit_ds = g.bind(&unit_pipe, &[&wb, &xqb, first_i, &unit_dmb, &xsb, &sxb]);
    let mut unit_disps = vec![Dispatch {
        pipe: unit_pipe,
        dset: unit_ds,
        groups: ((m as u32).div_ceil(LOCAL), segs),
    }];
    if segs > 1 {
        let ones = g.upload(&vec![1f32; segs as usize]);
        let pipe2 = g.pipeline(&gemv_bytes(m as u32, segs, LOCAL, 1, Carrier::F32), 3);
        let ds2 = g.bind(&pipe2, &[&partial_i, &ones, &out_i]);
        unit_disps.push(Dispatch {
            pipe: pipe2,
            dset: ds2,
            groups: ((m as u32).div_ceil(LOCAL), 1),
        });
    }
    g.run(&unit_disps);
    let got_i = unsafe { readback(&g, out_i.buffer, m) };
    let int_rel = print_check("q4_unit", &got_i, &want_int, m);

    // ── the direct f32 control: 3 bindings, no chart, no DP4A ──
    // `gemv_module` reads `W[k·M + r]` (column-major), not the row-major `chart.w`.
    let mut wf_host = vec![0f32; m * k];
    for r in 0..m {
        for kk in 0..k {
            wf_host[kk * m + r] = chart.w[r * k + kk];
        }
    }
    let wf = g.upload(&wf_host);
    let xfb = g.upload(&xf);
    let out_f = output_buffer(&g, m);
    let spv_f = gemv_bytes(m as u32, k as u32, LOCAL, 1, Carrier::F32);
    describe(&spv_f, "gemv F32");
    let pipe_f = g.pipeline(&spv_f, 3);
    let ds_f = g.bind(&pipe_f, &[&wf, &xfb, &out_f]);
    g.run(&[Dispatch {
        pipe: pipe_f,
        dset: ds_f,
        groups: ((m as u32).div_ceil(LOCAL), 1),
    }]);
    let got_f = unsafe { readback(&g, out_f.buffer, m) };
    let mut want_f = vec![0f64; m];
    for (r, want) in want_f.iter_mut().enumerate() {
        *want = (0..k)
            .map(|kk| f64::from(chart.w[r * k + kk]) * f64::from(xf[kk]))
            .sum();
    }
    let f_rel = print_check("f32_direct", &got_f, &want_f, m);

    let ok = rel <= 1e-4 && int_rel <= 1e-4 && f_rel <= 1e-4;
    println!("RESULT {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
