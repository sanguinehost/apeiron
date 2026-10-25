//! GPU dense-layer forward GEMM (`gpu::nn::linear_fwd_module`) against the Q32.32 CPU reference
//! `learn::grad::dense::linear_fwd`.
//!
//! The kernel owns one 16×16 output tile per workgroup: 256 threads stage a 16×16 A tile (x rows)
//! and a 16×16 B tile (w transposed) in two shared arrays per k block and accumulate the same
//! ascending `Fma(a, b, acc)` as `bwd::lin`. The CPU reference accumulates the Q32.32 products in
//! `i128` and rounds once; the f32 kernel rounds every product and accumulate. The comparison
//! therefore converts each GPU output to Q32.32 by round-to-nearest (saturating) and bounds the
//! per-entry difference in Q32.32 units, measured on the pinned shapes and stated per shape.
//!
//! Shapes `(n, i, o)`: `(1, 32, 64)`, `(7, 64, 128)`, `(512, 256, 256)`. Inputs are a fixed-seed
//! LCG in `[-0.5, 0.5)` real (Q32.32 units ±2³¹).
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_linear_fwd -- --nocapture
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
use apeiron::gpu::nn::fwd::linear_fwd_dispatch;
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::dense::linear_fwd;

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

/// One shape. The per-entry bound is not a constant: [`derived_bound`] computes it from the
/// fixture's own operands.
struct Fixture {
    label: &'static str,
    n: usize,
    i: usize,
    o: usize,
}

#[test]
fn linear_forward_matches_cpu() {
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
        },
        Fixture {
            label: "n=7 i=64 o=128",
            n: 7,
            i: 64,
            o: 128,
        },
        Fixture {
            label: "n=512 i=256 o=256",
            n: 512,
            i: 256,
            o: 256,
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
    let w: Vec<i64> = (0..o * i).map(|_| real_to_q32(rng.unit())).collect();

    let cpu_y = linear_fwd(&x, &w, n, i, o).expect("cpu linear_fwd");

    let xf: Vec<f32> = x.iter().map(|&v| to_f32(v)).collect();
    let wf: Vec<f32> = w.iter().map(|&v| to_f32(v)).collect();
    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let x_buf = unsafe { g.up_f32(&xf) };
    let w_buf = unsafe { g.up_f32(&wf) };
    let y_buf = unsafe { g.dev(n * o * 4) };

    let step =
        unsafe { linear_fwd_dispatch(&g, x_buf, w_buf, y_buf, n as u32, i as u32, o as u32) };

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
    let (rb_y, rb_y_p) = host(n * o);
    let cb = unsafe { vk.command_buffer() };
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
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (n * o * 4) as u64,
        };
        (vk.copy_buffer)(cb, y_buf, rb_y, 1, &region);
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
    let gpu_y: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_y_p, n * o).to_vec() };
    assert!(gpu_y.iter().all(|v| v.is_finite()), "non-finite y");

    let (max_y, at_y) = max_err(&gpu_y, &cpu_y, o);
    // The bound of the kernel's own precision chain, from this fixture's operands:
    // |C − C64| <= (2u + k·2^-24)·Σ_k|a||b| with u = 2^-8, u 32-bit fractional unit = 2^-32.
    let bound = derived_bound(&xf, &wf, n, i, o);
    println!(
        "{}: y max |gpu-cpu| = {max_y} units at {at_y:?}; derived bound {bound:.0} units, \
         ratio {:.3}",
        f.label,
        max_y as f64 / bound
    );
    assert!(
        max_y as f64 <= bound,
        "{}: y error {max_y} units at {at_y:?} exceeds the derived bound {bound:.0}",
        f.label
    );
}

fn bf16_dot_bound(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "bf16_dot_bound: lengths");
    let mag: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x).abs() * f64::from(y).abs())
        .sum();
    (2.0 * (1.0 / 256.0) + (a.len() as f64) * 5.960_464_477_539_063e-8) * mag
}

/// `(2u + k·2^-24)·Σ_k|a||b|` in Q32.32 units, over the worst output row of `x·wᵀ`: the bound the
/// bf16 face's precision chain implies (operands rounded once to bf16, u = 2^-8; f32 accumulate,
/// `eps = 2^-24`). One 32-bit fractional unit is `2^-32` of the real value.
fn derived_bound(x: &[f32], w: &[f32], n: usize, i: usize, o: usize) -> f64 {
    let mut worst = 0.0f64;
    for r in 0..n {
        for c in 0..o {
            worst = worst.max(bf16_dot_bound(
                &x[r * i..(r + 1) * i],
                &w[c * i..(c + 1) * i],
            ));
        }
    }
    worst * 4294967296.0
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
