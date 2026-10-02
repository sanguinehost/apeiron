//! Matched-shape measurement of the ported batch-tiled integer-dot GEMM
//! ([`apeiron::simt::gemv_dp4a_batch_bytes`], `gemv_dp4a_batch_module`) at the exact shapes and
//! token counts `tools/gpu/cuda_q8_gate.py (unpublished)` measures llama.cpp's CUDA MMQ Q8 prefill at.
//!
//! Why: the port's batch-1 decode GEMV number (729.2 GMAC/s) was compared against CUDA's
//! batched-prefill MMQ number (6325.87–51982.08 GMAC/s) and the regimes differ. This harness runs
//! the ported BATCH kernel at the same (M, K, tokens) so the ratio is per-shape and like-for-like.
//!
//! Layouts (from `gemv_dp4a_batch_module`'s doc): 0=W (u32 k-packed `W[c·M+r]`), 1=XQ (u32 int8
//! `XQ[i·k/4+c]`), 2=OUT (f32 `i·M+r`), 3=DM (u32 f16-packed `(d,lo)` at `b·M+r`), 4=XS (f32
//! `Σ_block x` at `i·blocks+b`), 5=SX (f32 per-block scale at `i·blocks+b`). Dispatch
//! `(ceil(M/local), tokens/bt)`; `bt` must divide `tokens` exactly so no padding is needed
//! (`bt` ∈ {1,2,4,8} divides both 200 and 512; default 8).
//!
//! `--check` grades a deterministic sample of output elements against a CPU reference that folds in
//! the kernel's exact order; `--bench`/`--json` times every canonical shape with device timestamps.
//!
//! Usage: `cargo run --release --features gpu --example simt_batch -- --check --bench --json`
//! Env: `SIMT_LOCAL`, `SIMT_BT`, `SIMT_REPS`, `SIMT_CHECK_TOKENS`.
#![allow(
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "measurement/parity harness: numeric casts over raw device buffer images are the point"
)]

use apeiron::gpu::*;
use apeiron::simt::{gemv_dp4a_batch_bytes, gemv_dp4a_batch_rt_bytes, Carrier};
use std::process::exit;

/// The canonical shape table, identical to `CANONICAL_SHAPES` in `tools/gpu/cuda_q8_bench.cpp (unpublished)` and
/// `APEIRON_BASELINES`/`SHAPES` in `tools/gpu/cuda_q8_gate.py (unpublished)`.
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("attn_q", 4096, 1024, 200),
    ("attn_q", 4096, 1024, 512),
    ("attn_k_v", 512, 1024, 200),
    ("attn_k_v", 512, 1024, 512),
    ("attn_output", 1024, 2048, 200),
    ("attn_output", 1024, 2048, 512),
    ("ffn_gate_up", 3584, 1024, 200),
    ("ffn_gate_up", 3584, 1024, 512),
    ("ffn_down", 1024, 3584, 200),
    ("ffn_down", 1024, 3584, 512),
];

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

/// Round-to-nearest-even `f32` → IEEE binary16 bits (the DM pack the kernel unpacks). Hand-rolled
/// because `core::f16` is unstable on this toolchain and the crate pins zero dependencies.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        // inf / nan
        let m = if mant != 0 { 0x0200 } else { 0 };
        return sign | 0x7c00 | m;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow → zero
        }
        let m = (mant | 0x0080_0000) >> (1 - e);
        let mut h = (m >> 13) as u16;
        if (m & 0x1fff) > 0x1000 || ((m & 0x1fff) == 0x1000 && (h & 1) == 1) {
            h = h.wrapping_add(1);
        }
        return sign | h;
    }
    let mut h = (mant >> 13) as u16;
    if (mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (h & 1) == 1) {
        h = h.wrapping_add(1);
        if h == 0x0400 {
            return sign | (((e + 1) as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | h
}

/// IEEE binary16 bits → `f32` (the reference's decode of what the kernel reads).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 127 - 15 + 1;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// The synthesized operands for one shape, in the batch kernel's own layouts.
struct Operands {
    w: Vec<u32>,
    xq: Vec<u32>,
    dm: Vec<u32>,
    xs: Vec<f32>,
    sx: Vec<f32>,
}

fn synthesize(m: u32, k: u32, tokens: u32) -> Operands {
    let (mu, ku, b) = (m as usize, k as usize, tokens as usize);
    let blocks = ku / 32;
    let cpb = 8usize;
    let xstride = ku / 4;
    let seed = 0x00A9_0000_u64 + (m as u64) * 131 + (k as u64) * 17 + tokens as u64;

    // unsigned weight codes, k-packed W[c·M + r]
    let w: Vec<u32> = (0..blocks * cpb * mu)
        .map(|i| mix(seed ^ (i as u64).wrapping_mul(0x9E37)) as u32)
        .collect();
    // signed activation words, XQ[i·xstride + c]
    let xq: Vec<u32> = (0..b * xstride)
        .map(|i| mix(seed ^ 0x1234_5678 ^ (i as u64).wrapping_mul(0x2545)) as u32)
        .collect();
    // f16-packed (d, lo) per (block, row), shared across columns
    let dm: Vec<u32> = (0..blocks * mu)
        .map(|idx| {
            let blk = idx / mu;
            let r = idx % mu;
            let d = 0.25 + (r % 7) as f32 * 0.0625;
            let lo = -0.125 + (blk % 5) as f32 * 0.03125;
            (f32_to_f16_bits(d) as u32) | ((f32_to_f16_bits(lo) as u32) << 16)
        })
        .collect();
    // XS = Σ_block x (arbitrary fp32) and SX = per-block activation scale, both token-major
    let xs: Vec<f32> = (0..b * blocks)
        .map(|i| ((i % 23) as f32) * 0.125 - 1.0)
        .collect();
    let sx: Vec<f32> = (0..b * blocks)
        .map(|i| 0.5 + ((i % 11) as f32) * 0.0625)
        .collect();
    Operands { w, xq, dm, xs, sx }
}

/// CPU reference, folding in the kernel's order: per (token i, row r),
///   Σ_b [ d(b,r)·(Σ_c Σ_l code·act)·sx(i,b) + lo(b,r)·XS(i,b) ].
fn reference(o: &Operands, m: u32, k: u32, check_tokens: usize) -> Vec<f32> {
    let (mu, ku) = (m as usize, k as usize);
    let blocks = ku / 32;
    let cpb = 8usize;
    let xstride = ku / 4;
    let mut out = vec![0f32; check_tokens * mu];
    for i in 0..check_tokens {
        for r in 0..mu {
            let mut acc = 0f32;
            for blk in 0..blocks {
                let mut dot = 0i32;
                for ch in 0..cpb {
                    let c = blk * cpb + ch;
                    let word = o.w[c * mu + r];
                    let act = o.xq[i * xstride + c];
                    for l in 0..4 {
                        let code = ((word >> (8 * l)) & 0xff) as i32;
                        let av = ((act >> (8 * l)) & 0xff) as u8 as i8 as i32;
                        dot = dot.wrapping_add(code.wrapping_mul(av));
                    }
                }
                let dmw = o.dm[blk * mu + r];
                let d = f16_bits_to_f32((dmw & 0xffff) as u16);
                let lo = f16_bits_to_f32((dmw >> 16) as u16);
                let idx = i * blocks + blk;
                acc += d * (dot as f32) * o.sx[idx] + lo * o.xs[idx];
            }
            out[i * mu + r] = acc;
        }
    }
    out
}

/// Per-run knobs, so `run_case` does not take ten positional arguments.
struct Cfg {
    local: u32,
    bt: u32,
    rt: u32,
    reps: u32,
    check: bool,
    check_tokens: usize,
}

fn run_case(gpu: &Gpu, m: u32, k: u32, tokens: u32, cfg: &Cfg) -> (f64, f64, f32) {
    let (local, bt, rt, reps) = (cfg.local, cfg.bt, cfg.rt, cfg.reps);
    assert!(
        tokens.is_multiple_of(bt),
        "bt={bt} must divide tokens={tokens} (no padding in this harness)"
    );
    let o = synthesize(m, k, tokens);
    let macs = (m as f64) * (tokens as f64) * (k as f64);

    let wb = gpu.upload(&o.w);
    let xqb = gpu.upload(&o.xq);
    let (yb, yptr) = gpu.host_read_buffer((tokens as usize) * (m as usize) * 4);
    let dmb = gpu.upload(&o.dm);
    let xsb = gpu.upload(&o.xs);
    let sxb = gpu.upload(&o.sx);
    // rt=1 is `gemv_dp4a_batch_module` (byte-identical); rt>1 adds the row tile
    // (`gemv_dp4a_batch_rt_module`), reusing one activation load across `rt` output rows.
    let spv = if rt > 1 {
        gemv_dp4a_batch_rt_bytes(m, k, local, Carrier::Q8, bt, rt)
    } else {
        gemv_dp4a_batch_bytes(m, k, local, Carrier::Q8, bt)
    };
    let pipe = gpu.pipeline(&spv, 6);
    let ds = gpu.bind(&pipe, &[&wb, &xqb, &yb, &dmb, &xsb, &sxb]);
    let d = Dispatch {
        pipe,
        dset: ds,
        groups: (m.div_ceil(local * rt), tokens / bt),
    };

    // warmup then best-of-reps device-timestamp interval
    gpu.run(&[d]);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let us = gpu.time(&[d]);
        if us.is_finite() && us < best {
            best = us;
        }
    }
    let dev_ns = best * 1000.0; // time() is µs

    // correctness: sample the first `check_tokens` columns against the CPU fold
    let mut max_rel = 0f32;
    if cfg.check {
        let vt = cfg.check_tokens.min(tokens as usize);
        let want = reference(&o, m, k, vt);
        for i in 0..vt {
            for r in 0..m as usize {
                let got = unsafe { yptr.cast::<f32>().add(i * m as usize + r).read_volatile() };
                let wv = want[i * m as usize + r];
                let denom = wv.abs().max(1.0);
                let rel = (got - wv).abs() / denom;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
    }
    (dev_ns, macs, max_rel)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let want_json = args.iter().any(|a| a == "--json");
    let want_check = args.iter().any(|a| a == "--check") || !args.iter().any(|a| a == "--bench");
    let want_bench = args.iter().any(|a| a == "--bench") || !args.iter().any(|a| a == "--check");

    // Swept on the RTX 5080 over local ∈ {8,16,32,64,128,256} × bt ∈ {1,2,4,8}: bt=4/local=32 is
    // the throughput optimum (bt=8 halves occupancy for reuse the weight path does not need; bt≤2
    // leaves reuse on the table). bt=4 divides both 200 (50) and 512 (128), so no padding.
    let local = env_u32("SIMT_LOCAL", 32);
    let bt = env_u32("SIMT_BT", 4);
    let rt = env_u32("SIMT_RT", 1);
    let reps = env_u32("SIMT_REPS", 5);
    let check_tokens = env_u32("SIMT_CHECK_TOKENS", 4) as usize;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(msg) => {
            println!("SKIP: {msg}");
            return;
        }
    };
    println!("device: {}", gpu.device_name());

    let mut failures: Vec<String> = Vec::new();
    let mut rows: Vec<(String, u32, u32, u32, f64, f64, f32)> = Vec::new();
    let cfg = Cfg {
        local,
        bt,
        rt,
        reps,
        check: want_check,
        check_tokens,
    };
    for &(name, m, k, tokens) in SHAPES {
        let (dev_ns, macs, max_rel) = run_case(&gpu, m, k, tokens, &cfg);
        let gmac_s = macs / dev_ns;
        if want_check {
            println!(
                "CHECK {name} tokens={tokens}: max_rel_diff={max_rel:e} (bt={bt} local={local} rt={rt})"
            );
            if max_rel > 1e-2 {
                failures.push(format!("check {name} tokens={tokens}"));
            }
        }
        if want_bench {
            println!(
                "RESULT shape={name} m={m} k={k} tokens={tokens} local={local} bt={bt} rt={rt} \
dev_ns={dev_ns:.1} macs={macs:.0} gmac_s={gmac_s:.2}"
            );
        }
        rows.push((name.to_string(), m, k, tokens, dev_ns, gmac_s, max_rel));
    }

    if want_json {
        let mut s = String::from("{\"apeiron\":[");
        for (i, (name, m, k, tokens, dev_ns, gmac_s, max_rel)) in rows.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"shape\":\"{name}\",\"m\":{m},\"k\":{k},\"tokens\":{tokens},\
\"dev_ns\":{dev_ns:.1},\"gmac_s\":{gmac_s:.2},\"max_rel_diff\":{max_rel:e}}}"
            ));
        }
        s.push_str("]}");
        println!("{s}");
    }

    if !failures.is_empty() {
        eprintln!("FAIL: {failures:?}");
        exit(1);
    }
}
