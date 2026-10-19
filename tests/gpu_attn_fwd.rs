//! GPU causal GQA attention forward (`gpu::nn::attn::fwd`) against the Q32.32 CPU reference
//! `learn::grad::attn::attn_fwd_kern`.
//!
//! The device writes the probabilities `P` and the output `out`; the reference returns the same two
//! tensors from `attn_row_fwd`'s max-subtracted softmax. The device accumulates in f32 with fused
//! multiply-add and a workgroup tree reduction; the reference accumulates Q32.32 products in `i128`
//! and rounds once per score, once per exponential sum and once per output entry. Each device output
//! is converted to Q32.32 by round-to-nearest (saturating) and the per-entry difference is bounded
//! in Q32.32 units, measured per shape and product.
//!
//! Shapes `(t, nh, nkv, hd)`: `(1,2,1,8)`, `(7,2,1,16)`, `(16,4,2,16)`, `(64,4,2,32)`. Inputs are a
//! fixed-seed LCG in `[-0.5, 0.5)` real.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_attn_fwd -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_attn_bwd/gpu_norm_backward tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::attn_fwd_dispatch;
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::attn::attn_fwd_kern;

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-0.5, 0.5)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u - 0.5
    }
}

fn real_to_q32(v: f64) -> i64 {
    (v * 4294967296.0).round() as i64
}

fn to_f32(v: i64) -> f32 {
    (v as f64 / 4294967296.0) as f32
}

/// f32 → Q32.32, round-to-nearest and saturating.
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

/// One shape and its pinned per-entry bounds in Q32.32 units (measured on this device, about
/// twice the observed maximum, stated per product).
struct Fixture {
    label: &'static str,
    t: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    bound_p: f64,
    bound_out: f64,
}

#[test]
fn attn_forward_matches_cpu() {
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
            label: "t=1 nh=2 nkv=1 hd=8",
            t: 1,
            nh: 2,
            nkv: 1,
            hd: 8,
            bound_p: 2.0,
            bound_out: 1800.0,
        },
        Fixture {
            label: "t=7 nh=2 nkv=1 hd=16",
            t: 7,
            nh: 2,
            nkv: 1,
            hd: 16,
            bound_p: 400.0,
            bound_out: 14600.0,
        },
        Fixture {
            label: "t=16 nh=4 nkv=2 hd=16",
            t: 16,
            nh: 4,
            nkv: 2,
            hd: 16,
            bound_p: 900.0,
            bound_out: 28000.0,
        },
        Fixture {
            label: "t=64 nh=4 nkv=2 hd=32",
            t: 64,
            nh: 4,
            nkv: 2,
            hd: 32,
            bound_p: 800.0,
            bound_out: 62400.0,
        },
    ];
    for f in &fixtures {
        run_fixture(&vk, f);
    }
}

#[allow(clippy::too_many_lines)]
fn run_fixture(vk: &Vk, f: &Fixture) {
    let (t, nh, nkv, hd) = (f.t, f.nh, f.nkv, f.hd);
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let q: Vec<i64> = (0..t * nh * hd).map(|_| real_to_q32(rng.unit())).collect();
    let k: Vec<i64> = (0..t * nkv * hd).map(|_| real_to_q32(rng.unit())).collect();
    let v: Vec<i64> = (0..t * nkv * hd).map(|_| real_to_q32(rng.unit())).collect();

    let (cpu_out, cpu_p) = attn_fwd_kern(&q, &k, &v, nh, nkv, hd).expect("cpu attn_fwd_kern");

    let qf: Vec<f32> = q.iter().map(|&x| to_f32(x)).collect();
    let kf: Vec<f32> = k.iter().map(|&x| to_f32(x)).collect();
    let vf: Vec<f32> = v.iter().map(|&x| to_f32(x)).collect();

    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let q_buf = unsafe { g.up_f32(&qf) };
    let k_buf = unsafe { g.up_f32(&kf) };
    let v_buf = unsafe { g.up_f32(&vf) };
    let p_buf = unsafe { g.dev(nh * t * t * 4) };
    let out_buf = unsafe { g.dev(t * nh * hd * 4) };

    let (tu, nhu, nkvu, hdu) = (t as u32, nh as u32, nkv as u32, hd as u32);
    let step =
        unsafe { attn_fwd_dispatch(&g, q_buf, k_buf, v_buf, p_buf, out_buf, tu, nhu, nkvu, hdu) };

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
    let (rb_p, rb_p_p) = host(nh * t * t);
    let (rb_out, rb_out_p) = host(t * nh * hd);
    let cb = unsafe { vk.command_buffer() };
    let region = |bytes: usize| BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: bytes as u64,
    };
    unsafe {
        vk.begin(cb);
        rec_step(vk, cb, &step);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_TRANSFER,
            ACCESS_SHADER_WRITE,
            ACCESS_TRANSFER_READ,
        );
        (vk.copy_buffer)(cb, p_buf, rb_p, 1, &region(nh * t * t * 4));
        (vk.copy_buffer)(cb, out_buf, rb_out, 1, &region(t * nh * hd * 4));
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_HOST,
            ACCESS_TRANSFER_WRITE,
            ACCESS_HOST_READ,
        );
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
    }
    let gpu_p: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_p_p, nh * t * t).to_vec() };
    let gpu_out: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_out_p, t * nh * hd).to_vec() };
    assert!(gpu_p.iter().all(|v| v.is_finite()), "non-finite p");
    assert!(gpu_out.iter().all(|v| v.is_finite()), "non-finite out");
    // The causal upper triangle is an exact zero on both sides.
    for h in 0..nh {
        for i in 0..t {
            for j in i + 1..t {
                assert_eq!(gpu_p[(h * t + i) * t + j], 0.0, "p above the diagonal");
            }
        }
    }

    let (max_p, at_p) = max_err(&gpu_p, &cpu_p, t);
    let (max_out, at_out) = max_err(&gpu_out, &cpu_out, nh * hd);
    println!(
        "{}: p max |gpu-cpu| = {max_p} units at {at_p:?}; out max = {max_out} at {at_out:?}; \
         pinned bounds p {} / out {}",
        f.label, f.bound_p, f.bound_out
    );
    assert!(
        max_p as f64 <= f.bound_p,
        "{}: p error {max_p} units at {at_p:?} exceeds the pinned bound {}",
        f.label,
        f.bound_p
    );
    assert!(
        max_out as f64 <= f.bound_out,
        "{}: out error {max_out} units at {at_out:?} exceeds the pinned bound {}",
        f.label,
        f.bound_out
    );
}

fn max_err(gpu: &[f32], cpu: &[i64], cols: usize) -> (i128, (usize, usize)) {
    let mut max = 0i128;
    let mut at = (0usize, 0usize);
    for (idx, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (i128::from(to_q32(g)) - i128::from(c)).abs();
        if d > max {
            max = d;
            at = (idx / cols, idx % cols);
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
