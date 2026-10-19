//! Adam elementwise update on the device (`gpu::nn::bwd::opt::adam_update_dispatch`) against the
//! CPU oracle `learn::fold::finish::train::Adam::apply`, one dispatch per step.
//!
//! Oracle, read from `src/learn/fold/finish/train.rs`: `p`, `g` and the first moment `m` are
//! Q32.32 `i64` (`ONE = 1 << 32`); the second moment `v` is Q64.64 `i128`. `mul(a, b)` is
//! `round(a·b / 2^32)` (ties to even), so the second-moment line
//! `t1 = rshift_rne(β₂·v, 32)` with the Q32.32 `β₂` word is `round(β₂·v / 2^32)` in the same
//! Q64.64 word as the exact square `g·g` (`g` Q32.32 squared is Q64.64). `√v` is
//! `narrow(isqrt(v))`: `√(v_real·2^64) = √v_real · 2^32`, Q32.32. Per element:
//!
//! ```text
//! m ← β₁·m + (1−β₁)·g
//! v ← β₂·v + (1−β₂)·g²
//! p ← p − lr·(m/(1−β₁ᵗ)) / (√v/√(1−β₂ᵗ) + ε)
//! ```
//!
//! The kernel carries real units (`p/2^32`, `g/2^32`, `m/2^32`, `v/2^64`), performs the same four
//! steps in f32, and writes real units back; the comparison converts each device output to the
//! oracle's word (`p`, `m` → Q32.32; `v` → Q64.64) and measures `max |gpu − cpu|` in that word.
//!
//! `Adam`'s moment fields are private and expose no accessor, so the oracle's `m` and `v` are read
//! from its derived `Debug` rendering (exact decimal `i64`/`i128`), not re-derived here.
//!
//! Fixtures `n = 1, 7, 896`; `p`, `g` from a fixed-seed LCG in `[-0.5, 0.5)`; `lr = 1e-3` as the
//! Q32.32 word `4294967`; steps `1, 2, 5` compared after each step.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_adam -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_linear_bwd/gpu_norm_backward tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::adam_update_dispatch;
use apeiron::gpu::vulkan::*;
use apeiron::learn::fold::finish::train::Adam;

/// `round(1e-3·2^32)`, the oracle's `lr` argument.
const LR_Q32: i64 = 4_294_967;

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

fn real_to_q32(v: f64) -> i64 {
    (v * 4294967296.0).round() as i64
}

fn to_f32(v: i64) -> f32 {
    (v as f64 / 4294967296.0) as f32
}

/// f32 real units → Q32.32, round-to-nearest and saturating.
fn to_q32(v: f32) -> i64 {
    let scaled = f64::from(v) * 4294967296.0;
    if scaled >= i64::MAX as f64 {
        i64::MAX
    } else if scaled <= i64::MIN as f64 {
        i64::MIN
    } else {
        scaled.round() as i64
    }
}

/// f32 real units → Q64.64, round-to-nearest (the second moment's word).
fn to_q64(v: f32) -> i128 {
    (f64::from(v) * 18_446_744_073_709_551_616.0).round() as i128
}

/// The oracle's `(m, v)` words, read from the derived `Debug` rendering (no public accessor).
fn moments(a: &Adam) -> (Vec<i64>, Vec<i128>) {
    let s = format!("{a:?}");
    let rest = s
        .split_once("m: [")
        .unwrap_or_else(|| panic!("Adam debug has no `m:` field: {s}"))
        .1;
    let (mtxt, rest) = ms_and(rest);
    let (vtxt, _) = ms_and(
        rest.split_once("v: [")
            .unwrap_or_else(|| panic!("Adam debug has no `v:` field: {rest}"))
            .1,
    );
    let parse_i64 = |t: &str| -> Vec<i64> { t.split(", ").map(|x| x.parse().unwrap()).collect() };
    let parse_i128 = |t: &str| -> Vec<i128> { t.split(", ").map(|x| x.parse().unwrap()).collect() };
    (parse_i64(&mtxt), parse_i128(&vtxt))
}

/// The list text before the closing `]`, and the remainder after it.
fn ms_and(s: &str) -> (&str, &str) {
    let end = s.find(']').expect("list closes");
    (&s[..end], &s[end + 1..])
}

/// One shape and its pinned per-entry bounds, in each tensor's own word (`p`, `m` Q32.32;
/// `v` Q64.64). Pinned at about twice the observed maximum.
struct Fixture {
    label: &'static str,
    n: usize,
    bound_p: f64,
    bound_m: f64,
    bound_v: f64,
}

#[test]
fn adam_update_matches_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {}", gpu_identity());
    let fixtures = [
        Fixture {
            label: "n=1",
            n: 1,
            bound_p: 5_000.0,
            bound_m: 300.0,
            bound_v: 6_000_000_000.0,
        },
        Fixture {
            label: "n=7",
            n: 7,
            bound_p: 5_000.0,
            bound_m: 300.0,
            bound_v: 6_000_000_000.0,
        },
        Fixture {
            label: "n=896",
            n: 896,
            bound_p: 5_000.0,
            bound_m: 300.0,
            bound_v: 6_000_000_000.0,
        },
    ];
    for f in &fixtures {
        run_fixture(&vk, f);
    }
}

fn run_fixture(vk: &Vk, f: &Fixture) {
    let n = f.n;
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let p0: Vec<i64> = (0..n).map(|_| real_to_q32(0.5 * rng.unit())).collect();
    let grad: Vec<i64> = (0..n).map(|_| real_to_q32(0.5 * rng.unit())).collect();

    // CPU oracle: one `apply` per step on the same words.
    let mut adam = Adam::zeros(n);
    let mut cpu_p = p0.clone();

    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let pf: Vec<f32> = p0.iter().map(|&v| to_f32(v)).collect();
    let gf: Vec<f32> = grad.iter().map(|&v| to_f32(v)).collect();
    let zeros = vec![0.0f32; n];
    let p_buf = unsafe { g.up_f32(&pf) };
    let g_buf = unsafe { g.up_f32(&gf) };
    let m_buf = unsafe { g.up_f32(&zeros) };
    let v_buf = unsafe { g.up_f32(&zeros) };

    let host = |count: usize| unsafe {
        let (b, m) = vk
            .buffer(
                count as u64 * 4,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .expect("readback buffer");
        (b, vk.map(m).cast::<f32>())
    };
    let (rb_p, rb_p_p) = host(n);
    let (rb_m, rb_m_p) = host(n);
    let (rb_v, rb_v_p) = host(n);

    for step in 1u32..=5 {
        adam.apply(LR_Q32, &mut cpu_p, &grad).expect("cpu apply");
        let dstep =
            unsafe { adam_update_dispatch(&g, p_buf, g_buf, m_buf, v_buf, n as u32, LR_Q32, step) };
        let (cpu_m, cpu_v) = moments(&adam);
        let record = step == 1 || step == 2 || step == 5;
        let cb = unsafe { vk.command_buffer() };
        unsafe {
            vk.begin(cb);
            rec_step(vk, cb, &dstep);
            if record {
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
                    size: (n * 4) as u64,
                };
                (vk.copy_buffer)(cb, p_buf, rb_p, 1, &region);
                (vk.copy_buffer)(cb, m_buf, rb_m, 1, &region);
                (vk.copy_buffer)(cb, v_buf, rb_v, 1, &region);
                vk.barrier(
                    cb,
                    STAGE_TRANSFER,
                    STAGE_HOST,
                    ACCESS_TRANSFER_WRITE,
                    ACCESS_HOST_READ,
                );
            }
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            vk.submit_wait(cb);
        }
        if !record {
            continue;
        }
        let gpu_p: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_p_p, n).to_vec() };
        let gpu_m: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_m_p, n).to_vec() };
        let gpu_v: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_v_p, n).to_vec() };
        assert!(gpu_p.iter().all(|v| v.is_finite()), "non-finite p");
        assert!(gpu_m.iter().all(|v| v.is_finite()), "non-finite m");
        assert!(gpu_v.iter().all(|v| v.is_finite()), "non-finite v");

        let (dp, at_p) = max_err_q32(&gpu_p, &cpu_p);
        let (dm, at_m) = max_err_q32(&gpu_m, &cpu_m);
        let (dv, at_v) = max_err_q64(&gpu_v, &cpu_v);
        println!(
            "{} step {step}: p max |gpu-cpu| = {dp} units at {at_p}; m max = {dm} units at {at_m}; v max = {dv} units at {at_v}; pinned p {} / m {} / v {}",
            f.label, f.bound_p, f.bound_m, f.bound_v
        );
        assert!(
            dp as f64 <= f.bound_p,
            "{} step {step}: p error {dp} units at {at_p} exceeds the pinned bound {}",
            f.label,
            f.bound_p
        );
        assert!(
            dm as f64 <= f.bound_m,
            "{} step {step}: m error {dm} units at {at_m} exceeds the pinned bound {}",
            f.label,
            f.bound_m
        );
        assert!(
            dv as f64 <= f.bound_v,
            "{} step {step}: v error {dv} units at {at_v} exceeds the pinned bound {}",
            f.label,
            f.bound_v
        );
    }
}

fn max_err_q32(gpu: &[f32], cpu: &[i64]) -> (i128, usize) {
    let mut max = 0i128;
    let mut at = 0usize;
    for (idx, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (i128::from(to_q32(g)) - i128::from(c)).abs();
        if d > max {
            max = d;
            at = idx;
        }
    }
    (max, at)
}

fn max_err_q64(gpu: &[f32], cpu: &[i128]) -> (i128, usize) {
    let mut max = 0i128;
    let mut at = 0usize;
    for (idx, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (to_q64(g) - c).abs();
        if d > max {
            max = d;
            at = idx;
        }
    }
    (max, at)
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
