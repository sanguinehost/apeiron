//! Ported-lineage GEMV check + measurement: the carrier-dial fold (`gemv_bytes`) and the
//! integer-dot chart fold (`gemv_dp4a_bytes`), run on the metal and graded against a CPU reference,
//! then timed with device timestamps.
//!
//! `--check` compares device output to the CPU fold (F32: `simt::gemv_ref`, exact same accumulation
//! order; Q8: the (code, scale) chart fold `Σ_b d·(Σ q·xq) + m·Σx`). `--bench` reports GFLOP/s and
//! GB/s (F32) and GMAC/s (Q8) over best-of-`REPS` device-timestamp intervals.
//!
//! Usage: `cargo run --release --features gpu --example simt_gemv -- --check --bench`
//! Env: `SIMT_M`, `SIMT_K`, `SIMT_LOCAL`, `SIMT_REPS`.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "measurement/parity harness: numeric casts over raw device buffer images are the point"
)]

use apeiron::gpu::*;
use apeiron::simt::{gemv_bytes, gemv_dp4a_bytes, gemv_ref, Carrier};
use std::process::exit;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic splitmix64 → uniform bits, so a rerun reproduces the exact buffer images.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// F32 GEMV: W column-major `[k, m]`, `y[r] = Σ_k W[k·m + r]·x[k]`. Exact CPU parity via `gemv_ref`.
fn check_f32(gpu: &Gpu, m: u32, k: u32, local: u32) -> Vec<f32> {
    let (mu, ku) = (m as usize, k as usize);
    let w: Vec<f32> = (0..mu * ku)
        .map(|i| (mix(i as u64) >> 40) as f32 / 8_388_608.0 - 1.0)
        .collect();
    let x: Vec<f32> = (0..ku)
        .map(|i| (mix(0x1234_5678 + i as u64) >> 40) as f32 / 8_388_608.0 - 1.0)
        .collect();
    let want = gemv_ref(&w, &x, mu, ku);

    let wb = gpu.upload(&w);
    let xb = gpu.upload(&x);
    let (yb, yptr) = gpu.host_read_buffer(mu * 4);
    let pipe = gpu.pipeline(&gemv_bytes(m, k, local, 1, Carrier::F32), 3);
    let ds = gpu.bind(&pipe, &[&wb, &xb, &yb]);
    gpu.run(&[Dispatch {
        pipe,
        dset: ds,
        groups: (m.div_ceil(local), 1),
    }]);

    let mut max_diff = 0f32;
    for (r, &wv) in want.iter().enumerate() {
        let got = unsafe { yptr.cast::<f32>().add(r).read_volatile() };
        let d = (got - wv).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("CHECK f32 gemv ({m}x{k}): max_abs_diff={max_diff:e}");
    if max_diff > 1e-3 {
        eprintln!("FAIL f32 gemv: parity exceeds 1e-3");
        exit(1);
    }
    want
}

/// Q8 chart GEMV: codes `W[c·m + r]` (4 uint8 per word, k = 4c..4c+3), int8 activations `XQ[c]`,
/// interleaved `DM[2·(b·m + r)] = d`, `DM[.. + 1] = m0`, `XS[b] = Σx`. Device output graded against
/// the same integer-fold + affine epilogue on the CPU.
fn check_q8(gpu: &Gpu, m: u32, k: u32, local: u32) {
    let (mu, ku) = (m as usize, k as usize);
    let blocks = ku / 32;
    let cpb = 8; // Q8: 32 codes / 4 per word
                 // packed weight words: W[(b·8 + c)·m + r]
    let w: Vec<u32> = (0..blocks * cpb * mu)
        .map(|i| {
            let h = mix(0xABCD_0000 + i as u64);
            ((h & 0xff) as u32)
                | (((h >> 8) & 0xff) as u32) << 8
                | (((h >> 16) & 0xff) as u32) << 16
                | (((h >> 24) & 0xff) as u32) << 24
        })
        .collect();
    // int8 activation words: XQ[b·8 + c]
    let xq: Vec<u32> = (0..blocks * cpb)
        .map(|i| {
            let h = mix(0x77AA_0000 + i as u64);
            let b0 = (h & 0xff) as u8 as i8 as i32 as u32 & 0xff;
            let b1 = ((h >> 8) & 0xff) as u8 as i8 as i32 as u32 & 0xff;
            let b2 = ((h >> 16) & 0xff) as u8 as i8 as i32 as u32 & 0xff;
            let b3 = ((h >> 24) & 0xff) as u8 as i8 as i32 as u32 & 0xff;
            b0 | b1 << 8 | b2 << 16 | b3 << 24
        })
        .collect();
    // DM interleaved (d, m0) per (block, row); XS[b] = Σ x over the block (arbitrary, m0 exercises it)
    let mut dm = vec![0f32; blocks * mu * 2];
    for b in 0..blocks {
        for r in 0..mu {
            dm[2 * (b * mu + r)] = 0.25 + (b as f32) * 1e-3;
            dm[2 * (b * mu + r) + 1] = -0.05 + (r as f32) * 1e-5;
        }
    }
    let xs: Vec<f32> = (0..blocks).map(|b| (b as f32) * 0.5 - 1.0).collect();

    // CPU reference: b·(Σ q·xq) + m0·XS, folded in the kernel's order (acc += d*dot + m0*xs).
    let mut want = vec![0f32; mu];
    for r in 0..mu {
        let mut acc = 0f32;
        for b in 0..blocks {
            let mut dot = 0i32;
            for c in 0..cpb {
                let word = w[(b * cpb + c) * mu + r];
                let act = xq[b * cpb + c];
                for l in 0..4 {
                    let code = ((word >> (8 * l)) & 0xff) as i32;
                    let av = ((act >> (8 * l)) & 0xff) as u8 as i8 as i32;
                    dot += code * av;
                }
            }
            let d = dm[2 * (b * mu + r)];
            let m0 = dm[2 * (b * mu + r) + 1];
            acc += d * dot as f32 + m0 * xs[b];
        }
        want[r] = acc;
    }

    let wb = gpu.upload(&w);
    let xqb = gpu.upload(&xq);
    let (yb, yptr) = gpu.host_read_buffer(mu * 4);
    let dmb = gpu.upload(&dm);
    let xsb = gpu.upload(&xs);
    let pipe = gpu.pipeline(&gemv_dp4a_bytes(m, k, local, 1, Carrier::Q8), 5);
    let ds = gpu.bind(&pipe, &[&wb, &xqb, &yb, &dmb, &xsb]);
    gpu.run(&[Dispatch {
        pipe,
        dset: ds,
        groups: (m.div_ceil(local), 1),
    }]);

    let mut max_diff = 0f32;
    for (r, &wv) in want.iter().enumerate() {
        let got = unsafe { yptr.cast::<f32>().add(r).read_volatile() };
        let denom = wv.abs().max(1.0);
        let d = (got - wv).abs() / denom;
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("CHECK q8 dp4a gemv ({m}x{k}): max_rel_diff={max_diff:e}");
    if max_diff > 1e-3 {
        eprintln!("FAIL q8 dp4a gemv: parity exceeds 1e-3 relative");
        exit(1);
    }
}

/// Best-of-`reps` device-timestamp interval for one recorded dispatch, in microseconds.
fn time_dispatch(gpu: &Gpu, d: Dispatch, reps: u32) -> f64 {
    // warmup (clock boost) then best-of
    gpu.run(&[d]);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = gpu.time(&[d]);
        if t.is_finite() && t < best {
            best = t;
        }
    }
    best
}

fn bench_f32(gpu: &Gpu, m: u32, k: u32, local: u32, reps: u32) {
    let (mu, ku) = (m as usize, k as usize);
    let w: Vec<f32> = (0..mu * ku)
        .map(|i| (mix(i as u64) >> 40) as f32 * 1e-9)
        .collect();
    let x: Vec<f32> = (0..ku)
        .map(|i| (mix(i as u64) >> 40) as f32 * 1e-9)
        .collect();
    let wb = gpu.upload(&w);
    let xb = gpu.upload(&x);
    let (yb, _) = gpu.host_read_buffer(mu * 4);
    let pipe = gpu.pipeline(&gemv_bytes(m, k, local, 1, Carrier::F32), 3);
    let ds = gpu.bind(&pipe, &[&wb, &xb, &yb]);
    let d = Dispatch {
        pipe,
        dset: ds,
        groups: (m.div_ceil(local), 1),
    };
    let us = time_dispatch(gpu, d, reps);
    if !us.is_finite() {
        println!("BENCH f32 gemv: SKIP (no timestamp support)");
        return;
    }
    let secs = us * 1e-6;
    let bytes = (mu * ku * 4 + ku * 4 + mu * 4) as f64;
    let flops = 2.0 * (mu * ku) as f64;
    println!(
        "BENCH f32 gemv ({m}x{k}, local={local}): {us:.2} us  {:.1} GFLOP/s  {:.1} GB/s",
        flops / secs / 1e9,
        bytes / secs / 1e9
    );
}

fn bench_q8(gpu: &Gpu, m: u32, k: u32, local: u32, reps: u32) {
    let (mu, ku) = (m as usize, k as usize);
    let blocks = ku / 32;
    let w: Vec<u32> = (0..blocks * 8 * mu).map(|i| mix(i as u64) as u32).collect();
    let xq: Vec<u32> = (0..blocks * 8).map(|i| mix(i as u64) as u32).collect();
    let dm: Vec<f32> = (0..blocks * mu * 2)
        .map(|i| mix(i as u64) as f32 * 1e-9)
        .collect();
    let xs: Vec<f32> = (0..blocks).map(|i| mix(i as u64) as f32 * 1e-9).collect();
    let wb = gpu.upload(&w);
    let xqb = gpu.upload(&xq);
    let (yb, _) = gpu.host_read_buffer(mu * 4);
    let dmb = gpu.upload(&dm);
    let xsb = gpu.upload(&xs);
    let pipe = gpu.pipeline(&gemv_dp4a_bytes(m, k, local, 1, Carrier::Q8), 5);
    let ds = gpu.bind(&pipe, &[&wb, &xqb, &yb, &dmb, &xsb]);
    let d = Dispatch {
        pipe,
        dset: ds,
        groups: (m.div_ceil(local), 1),
    };
    let us = time_dispatch(gpu, d, reps);
    if !us.is_finite() {
        println!("BENCH q8 dp4a gemv: SKIP (no timestamp support)");
        return;
    }
    let secs = us * 1e-6;
    let bytes = (mu * ku + ku + mu * 4 + blocks * mu * 8 + blocks * 4) as f64;
    let macs = (mu * ku) as f64;
    println!(
        "BENCH q8 dp4a gemv ({m}x{k}, local={local}): {us:.2} us  {:.1} GMAC/s  {:.1} GB/s",
        macs / secs / 1e9,
        bytes / secs / 1e9
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let want_check = args.iter().any(|a| a == "--check") || !args.iter().any(|a| a == "--bench");
    let want_bench = args.iter().any(|a| a == "--bench") || !args.iter().any(|a| a == "--check");

    let m = env_u32("SIMT_M", 4096);
    let k = env_u32("SIMT_K", 4096);
    let local = env_u32("SIMT_LOCAL", 256);
    let reps = env_u32("SIMT_REPS", 5);

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(msg) => {
            println!("SKIP: {msg}");
            return;
        }
    };
    println!("device: {}", gpu.device_name());

    if want_check {
        check_f32(&gpu, m, k, local);
        check_q8(&gpu, m, k, local);
    }
    if want_bench {
        bench_f32(&gpu, m, k, local, reps);
        bench_q8(&gpu, m, k, local, reps);
    }
}
