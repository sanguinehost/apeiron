//! GPU dense-layer backward GEMMs (`gpu::nn::linear_bwd_input_module`,
//! `gpu::nn::linear_bwd_weight_module`) against the Q32.32 CPU reference
//! `learn::grad::dense::{linear_bwd_input,linear_bwd_weight}`.
//!
//! The kernel is naive: one thread per output, the reduction length baked per shape, f32
//! accumulation with fused multiply-add, and no reuse (each output reads its whole `dy` row/column
//! and its whole `w`/`x` column/row). The CPU reference accumulates the Q32.32 products in `i128`
//! and rounds once; the f32 kernel rounds every product and accumulate. The comparison therefore
//! converts each GPU output to Q32.32 by round-to-nearest (saturating) and bounds the per-entry
//! difference in Q32.32 units, measured on the pinned shapes and stated per shape and product.
//!
//! Shapes `(n, i, o)`: `(1, 32, 64)`, `(7, 64, 128)`, `(512, 256, 256)`. Inputs are a fixed-seed
//! LCG in `[-1, 1)` real (Q32.32 units ±2³²). `i`/`o` are less than 256 so the third fixture's
//! reduction lengths (`o = 256` for dX, `n = 512` for dW) are the long-k cases.
//!
//! Board row: `apeiron-gpu-train-w1-t01-EQpaf3sVmt04RegjErRx1k`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_linear_bwd -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_norm_backward/gpu_layer_match tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::{linear_bwd_input_dispatch, linear_bwd_weight_dispatch};
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::dense::{linear_bwd_input, linear_bwd_weight};

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
    n: usize,
    i: usize,
    o: usize,
    bound_dx: f64,
    bound_dw: f64,
}

#[test]
fn linear_backward_matches_cpu() {
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
            label: "n=1 i=32 o=64",
            n: 1,
            i: 32,
            o: 64,
            bound_dx: 9000.0,
            bound_dw: 600.0,
        },
        Fixture {
            label: "n=7 i=64 o=128",
            n: 7,
            i: 64,
            o: 128,
            bound_dx: 40000.0,
            bound_dw: 3500.0,
        },
        Fixture {
            label: "n=512 i=256 o=256",
            n: 512,
            i: 256,
            o: 256,
            bound_dx: 120000.0,
            bound_dw: 220000.0,
        },
    ];
    for f in &fixtures {
        run_fixture(&vk, f);
    }
}

fn run_fixture(vk: &Vk, f: &Fixture) {
    let (n, i, o) = (f.n, f.i, f.o);
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let x: Vec<i64> = (0..n * i).map(|_| real_to_q32(rng.unit())).collect();
    let dy: Vec<i64> = (0..n * o).map(|_| real_to_q32(rng.unit())).collect();
    let w: Vec<i64> = (0..o * i).map(|_| real_to_q32(rng.unit())).collect();

    let cpu_dx = linear_bwd_input(&dy, &w, n, i, o).expect("cpu linear_bwd_input");
    let cpu_dw = linear_bwd_weight(&dy, &x, n, i, o).expect("cpu linear_bwd_weight");

    let yf: Vec<f32> = dy.iter().map(|&v| to_f32(v)).collect();
    let wf: Vec<f32> = w.iter().map(|&v| to_f32(v)).collect();
    let xf: Vec<f32> = x.iter().map(|&v| to_f32(v)).collect();
    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let y_buf = unsafe { g.up_f32(&yf) };
    let w_buf = unsafe { g.up_f32(&wf) };
    let x_buf = unsafe { g.up_f32(&xf) };
    let dx_buf = unsafe { g.dev(n * i * 4) };
    let dw_buf = unsafe { g.dev(o * i * 4) };

    let step_dx = unsafe {
        linear_bwd_input_dispatch(&g, y_buf, w_buf, dx_buf, n as u32, i as u32, o as u32)
    };
    let step_dw = unsafe {
        linear_bwd_weight_dispatch(&g, y_buf, x_buf, dw_buf, n as u32, i as u32, o as u32)
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
    let (rb_dx, rb_dx_p) = host(n * i);
    let (rb_dw, rb_dw_p) = host(o * i);
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_step(vk, cb, &step_dx);
        rec_step(vk, cb, &step_dw);
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_TRANSFER,
            ACCESS_SHADER_WRITE,
            ACCESS_TRANSFER_READ,
        );
        let dx_region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (n * i * 4) as u64,
        };
        let dw_region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (o * i * 4) as u64,
        };
        (vk.copy_buffer)(cb, dx_buf, rb_dx, 1, &dx_region);
        (vk.copy_buffer)(cb, dw_buf, rb_dw, 1, &dw_region);
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
    let gpu_dx: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dx_p, n * i).to_vec() };
    let gpu_dw: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dw_p, o * i).to_vec() };
    assert!(gpu_dx.iter().all(|v| v.is_finite()), "non-finite dx");
    assert!(gpu_dw.iter().all(|v| v.is_finite()), "non-finite dw");

    let (max_dx, at_dx) = max_err(&gpu_dx, &cpu_dx, i);
    let (max_dw, at_dw) = max_err(&gpu_dw, &cpu_dw, i);
    println!(
        "{}: dx max |gpu-cpu| = {max_dx} units at {at_dx:?}; dw max = {max_dw} units at {at_dw:?}; pinned bounds dx {} / dw {}",
        f.label, f.bound_dx, f.bound_dw
    );
    assert!(
        max_dx as f64 <= f.bound_dx,
        "{}: dx error {max_dx} units at {at_dx:?} exceeds the pinned bound {}",
        f.label,
        f.bound_dx
    );
    assert!(
        max_dw as f64 <= f.bound_dw,
        "{}: dw error {max_dw} units at {at_dw:?} exceeds the pinned bound {}",
        f.label,
        f.bound_dw
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
