//! GPU SwiGLU forward and backward (`gpu::nn::act::swiglu_*_module`) against the Q32.32 CPU
//! reference `learn::grad::dense::{swiglu_fwd, swiglu_bwd}`.
//!
//! The kernels are one thread per element: `out = silu(gate)·up` and
//! `dgate = (dy·up)·silu'(gate)`, `dup = dy·silu(gate)`. The CPU reference rounds every product to
//! Q32.32; the device keeps f32. Each GPU output is converted to Q32.32 by round-to-nearest
//! (saturating) and the per-entry difference is bounded in Q32.32 units.
//!
//! Shapes `n`: `1`, `7`, `896` (Qwen2.5-0.5B FFN width). Inputs are a fixed-seed LCG in
//! `[-2, 2)` real.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_swiglu -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_linear_fwd/gpu_norm_backward tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::act::{swiglu_bwd_dispatch, swiglu_fwd_dispatch};
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::dense::{swiglu_bwd, swiglu_fwd};

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-2, 2)`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        4.0 * u - 2.0
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

/// One shape and its pinned per-entry bounds in Q32.32 units (measured on this device, about twice
/// the observed maximum).
struct Fixture {
    label: &'static str,
    n: usize,
    bound_out: f64,
    bound_dgate: f64,
    bound_dup: f64,
}

#[test]
fn swiglu_matches_cpu() {
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
            bound_out: 20.0,
            bound_dgate: 30.0,
            bound_dup: 64.0,
        },
        Fixture {
            label: "n=7",
            n: 7,
            bound_out: 500.0,
            bound_dgate: 2400.0,
            bound_dup: 500.0,
        },
        Fixture {
            label: "n=896",
            n: 896,
            bound_out: 3100.0,
            bound_dgate: 3100.0,
            bound_dup: 2700.0,
        },
    ];
    for f in &fixtures {
        run_fixture(&vk, f);
    }
}

fn run_fixture(vk: &Vk, f: &Fixture) {
    let n = f.n;
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let gate: Vec<i64> = (0..n).map(|_| real_to_q32(rng.unit())).collect();
    let up: Vec<i64> = (0..n).map(|_| real_to_q32(rng.unit())).collect();
    let dy: Vec<i64> = (0..n).map(|_| real_to_q32(rng.unit())).collect();

    let cpu_out = swiglu_fwd(&gate, &up).expect("cpu swiglu_fwd");
    let (cpu_dgate, cpu_dup) = swiglu_bwd(&gate, &up, &dy).expect("cpu swiglu_bwd");

    let gf: Vec<f32> = gate.iter().map(|&v| to_f32(v)).collect();
    let uf: Vec<f32> = up.iter().map(|&v| to_f32(v)).collect();
    let yf: Vec<f32> = dy.iter().map(|&v| to_f32(v)).collect();
    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let gate_buf = unsafe { g.up_f32(&gf) };
    let up_buf = unsafe { g.up_f32(&uf) };
    let dy_buf = unsafe { g.up_f32(&yf) };
    let out_buf = unsafe { g.dev(n * 4) };
    let dgate_buf = unsafe { g.dev(n * 4) };
    let dup_buf = unsafe { g.dev(n * 4) };

    let fstep = unsafe { swiglu_fwd_dispatch(&g, gate_buf, up_buf, out_buf, n as u32) };
    let bstep =
        unsafe { swiglu_bwd_dispatch(&g, gate_buf, up_buf, dy_buf, dgate_buf, dup_buf, n as u32) };

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
    let (rb_out, rb_out_p) = host(n);
    let (rb_dgate, rb_dgate_p) = host(n);
    let (rb_dup, rb_dup_p) = host(n);
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_step(vk, cb, &fstep);
        rec_step(vk, cb, &bstep);
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
        (vk.copy_buffer)(cb, out_buf, rb_out, 1, &region);
        (vk.copy_buffer)(cb, dgate_buf, rb_dgate, 1, &region);
        (vk.copy_buffer)(cb, dup_buf, rb_dup, 1, &region);
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
    let gpu_out: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_out_p, n).to_vec() };
    let gpu_dgate: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dgate_p, n).to_vec() };
    let gpu_dup: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dup_p, n).to_vec() };
    assert!(gpu_out.iter().all(|v| v.is_finite()), "non-finite out");
    assert!(gpu_dgate.iter().all(|v| v.is_finite()), "non-finite dgate");
    assert!(gpu_dup.iter().all(|v| v.is_finite()), "non-finite dup");

    let (max_out, at_out) = max_err(&gpu_out, &cpu_out);
    let (max_dgate, at_dgate) = max_err(&gpu_dgate, &cpu_dgate);
    let (max_dup, at_dup) = max_err(&gpu_dup, &cpu_dup);
    println!(
        "{}: out max |gpu-cpu| = {max_out} units at {at_out}; dgate max = {max_dgate} at {at_dgate}; dup max = {max_dup} at {at_dup}; pinned out {} / dgate {} / dup {}",
        f.label, f.bound_out, f.bound_dgate, f.bound_dup
    );
    assert!(
        max_out as f64 <= f.bound_out,
        "{}: out error {max_out} units at {at_out} exceeds the pinned bound {}",
        f.label,
        f.bound_out
    );
    assert!(
        max_dgate as f64 <= f.bound_dgate,
        "{}: dgate error {max_dgate} units at {at_dgate} exceeds the pinned bound {}",
        f.label,
        f.bound_dgate
    );
    assert!(
        max_dup as f64 <= f.bound_dup,
        "{}: dup error {max_dup} units at {at_dup} exceeds the pinned bound {}",
        f.label,
        f.bound_dup
    );
}

fn max_err(gpu: &[f32], cpu: &[i64]) -> (i128, usize) {
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

fn gpu_identity() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "device name/driver unavailable (nvidia-smi absent)".to_string(),
    }
}
