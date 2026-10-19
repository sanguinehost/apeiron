//! GPU causal GQA attention backward (`gpu::nn::attn::bwd`) against the Q32.32 CPU reference
//! `learn::grad::attn::attn_bwd_kern`.
//!
//! The oracle forward `attn_fwd_kern` produces the probabilities `p`; the same `p` is passed to the
//! CPU backward and uploaded to the device, so the four device passes and the reference share one
//! softmax row exactly. The device accumulates in f32 with fused multiply-add; the reference
//! accumulates Q32.32 products in `i128` and rounds once. Each device output is converted to
//! Q32.32 by round-to-nearest (saturating) and the per-entry difference is bounded in Q32.32 units,
//! measured per shape and product.
//!
//! Shapes `(t, nh, nkv, hd)`: `(1,2,1,8)`, `(7,2,1,16)`, `(16,4,2,16)`, `(64,4,2,32)`. Inputs are a
//! fixed-seed LCG in `[-0.5, 0.5)` real. The four dispatches are recorded in one command buffer in
//! the order rowsum, ds, dq, dk/dv; `rec_step` emits the compute→compute barrier
//! (`vk.barrier(cb, STAGE_COMPUTE, STAGE_COMPUTE, ACCESS_SHADER_WRITE, ACCESS_SHADER_READ |
//! ACCESS_SHADER_WRITE)`) after each, so every dependent pass reads the previous results.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_attn_bwd -- --nocapture
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
use apeiron::gpu::nn::{
    attn_bwd_dkdv_dispatch, attn_bwd_dq_dispatch, attn_bwd_ds_dispatch, attn_bwd_rowsum_dispatch,
};
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::attn::{attn_bwd_kern, attn_fwd_kern};

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
    bound_dq: f64,
    bound_dk: f64,
    bound_dv: f64,
}

#[test]
fn attn_backward_matches_cpu() {
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
            bound_dq: 70.0,
            bound_dk: 50.0,
            bound_dv: 2400.0,
        },
        Fixture {
            label: "t=7 nh=2 nkv=1 hd=16",
            t: 7,
            nh: 2,
            nkv: 1,
            hd: 16,
            bound_dq: 2800.0,
            bound_dk: 1700.0,
            bound_dv: 18100.0,
        },
        Fixture {
            label: "t=16 nh=4 nkv=2 hd=16",
            t: 16,
            nh: 4,
            nkv: 2,
            hd: 16,
            bound_dq: 3600.0,
            bound_dk: 3600.0,
            bound_dv: 40400.0,
        },
        Fixture {
            label: "t=64 nh=4 nkv=2 hd=32",
            t: 64,
            nh: 4,
            nkv: 2,
            hd: 32,
            bound_dq: 4650.0,
            bound_dk: 3250.0,
            bound_dv: 64900.0,
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
    let dout: Vec<i64> = (0..t * nh * hd).map(|_| real_to_q32(rng.unit())).collect();

    let (_out, p) = attn_fwd_kern(&q, &k, &v, nh, nkv, hd).expect("cpu attn_fwd_kern");
    let (cpu_dq, cpu_dk, cpu_dv) =
        attn_bwd_kern(&q, &k, &v, &p, &dout, nh, nkv, hd).expect("cpu attn_bwd_kern");

    let qf: Vec<f32> = q.iter().map(|&x| to_f32(x)).collect();
    let kf: Vec<f32> = k.iter().map(|&x| to_f32(x)).collect();
    let vf: Vec<f32> = v.iter().map(|&x| to_f32(x)).collect();
    let pf: Vec<f32> = p.iter().map(|&x| to_f32(x)).collect();
    let dof: Vec<f32> = dout.iter().map(|&x| to_f32(x)).collect();

    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let q_buf = unsafe { g.up_f32(&qf) };
    let k_buf = unsafe { g.up_f32(&kf) };
    let v_buf = unsafe { g.up_f32(&vf) };
    let p_buf = unsafe { g.up_f32(&pf) };
    let do_buf = unsafe { g.up_f32(&dof) };
    let rs_buf = unsafe { g.dev(t * nh * 4) };
    let ds_buf = unsafe { g.dev(nh * t * t * 4) };
    let dq_buf = unsafe { g.dev(t * nh * hd * 4) };
    let dk_buf = unsafe { g.dev(t * nkv * hd * 4) };
    let dv_buf = unsafe { g.dev(t * nkv * hd * 4) };

    let (tu, nhu, nkvu, hdu) = (t as u32, nh as u32, nkv as u32, hd as u32);
    let step_rs =
        unsafe { attn_bwd_rowsum_dispatch(&g, v_buf, p_buf, do_buf, rs_buf, tu, nhu, nkvu, hdu) };
    let step_ds = unsafe {
        attn_bwd_ds_dispatch(&g, v_buf, p_buf, do_buf, rs_buf, ds_buf, tu, nhu, nkvu, hdu)
    };
    let step_dq = unsafe { attn_bwd_dq_dispatch(&g, k_buf, ds_buf, dq_buf, tu, nhu, nkvu, hdu) };
    let step_dkdv = unsafe {
        attn_bwd_dkdv_dispatch(
            &g, q_buf, p_buf, ds_buf, do_buf, dk_buf, dv_buf, tu, nhu, nkvu, hdu,
        )
    };

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
    let (rb_dq, rb_dq_p) = host(t * nh * hd);
    let (rb_dk, rb_dk_p) = host(t * nkv * hd);
    let (rb_dv, rb_dv_p) = host(t * nkv * hd);
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_step(vk, cb, &step_rs);
        rec_step(vk, cb, &step_ds);
        rec_step(vk, cb, &step_dq);
        rec_step(vk, cb, &step_dkdv);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_TRANSFER,
            ACCESS_SHADER_WRITE,
            ACCESS_TRANSFER_READ,
        );
        let region = |bytes: usize| BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: bytes as u64,
        };
        (vk.copy_buffer)(cb, dq_buf, rb_dq, 1, &region(t * nh * hd * 4));
        (vk.copy_buffer)(cb, dk_buf, rb_dk, 1, &region(t * nkv * hd * 4));
        (vk.copy_buffer)(cb, dv_buf, rb_dv, 1, &region(t * nkv * hd * 4));
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
    let gpu_dq: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dq_p, t * nh * hd).to_vec() };
    let gpu_dk: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dk_p, t * nkv * hd).to_vec() };
    let gpu_dv: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dv_p, t * nkv * hd).to_vec() };
    assert!(gpu_dq.iter().all(|v| v.is_finite()), "non-finite dq");
    assert!(gpu_dk.iter().all(|v| v.is_finite()), "non-finite dk");
    assert!(gpu_dv.iter().all(|v| v.is_finite()), "non-finite dv");

    let (max_dq, at_dq) = max_err(&gpu_dq, &cpu_dq, nh * hd);
    let (max_dk, at_dk) = max_err(&gpu_dk, &cpu_dk, nkv * hd);
    let (max_dv, at_dv) = max_err(&gpu_dv, &cpu_dv, nkv * hd);
    println!(
        "{}: dq max |gpu-cpu| = {max_dq} units at {at_dq:?}; dk max = {max_dk} at {at_dk:?}; \
         dv max = {max_dv} at {at_dv:?}; pinned bounds dq {} / dk {} / dv {}",
        f.label, f.bound_dq, f.bound_dk, f.bound_dv
    );
    assert!(
        max_dq as f64 <= f.bound_dq,
        "{}: dq error {max_dq} units at {at_dq:?} exceeds the pinned bound {}",
        f.label,
        f.bound_dq
    );
    assert!(
        max_dk as f64 <= f.bound_dk,
        "{}: dk error {max_dk} units at {at_dk:?} exceeds the pinned bound {}",
        f.label,
        f.bound_dk
    );
    assert!(
        max_dv as f64 <= f.bound_dv,
        "{}: dv error {max_dv} units at {at_dv:?} exceeds the pinned bound {}",
        f.label,
        f.bound_dv
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
