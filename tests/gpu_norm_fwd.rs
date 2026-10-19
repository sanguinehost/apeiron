//! GPU row-batched RMSNorm forward (`gpu::nn::norm_fwd::rmsnorm_fwd_module`) against the CPU
//! Q32.32 reference `learn::grad::dense::rmsnorm_fwd`, one row at a time.
//!
//! The kernel computes `r = (mean(x²) + eps)^(-1/2)` with the same shared 256-lane tree reduce and
//! the same Q32.32 mean rounding `rmsnorm_bwd_module` uses, so the layer step's backward pass can
//! reuse this normalisation on the same rows. Every entry of `out = x·r·γ` is converted to
//! Q32.32 and compared per entry against the CPU forward at the pinned bound (about twice the
//! measured maximum per shape).
//!
//! Shapes: `(t,w) ∈ {(1,8), (7,64), (16,896)}`, `w = 896` the Qwen2.5-0.5B hidden width;
//! `eps = 1e-6` (the sibling backward test's `1e-6f32` kernel argument and `EPS_Q32 = 4295`);
//! inputs and `γ` from a fixed LCG in `[-0.5, 0.5)`.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_norm_fwd -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_layer_match/gpu_train_baseline tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::fwd::rmsnorm_fwd_dispatch;
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::dense::rmsnorm_fwd;

/// `eps = 1e-6` in Q32.32 units (the same value the f32 kernel is given as `1e-6f32`).
const EPS_Q32: i64 = 4295;

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

/// One shape: `t` rows of width `w`.
struct Fixture {
    label: &'static str,
    t: usize,
    w: usize,
    /// Pinned per-entry bound on `out`, in Q32.32 units: about twice the measured value.
    bound: f64,
}

/// Deterministic inputs: `x` is `t × w` and `γ` is `w`, both uniform in `[-0.5, 0.5)` Q32.32.
fn inputs(f: &Fixture) -> (Vec<i64>, Vec<i64>) {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let x: Vec<i64> = (0..f.t * f.w)
        .map(|_| real_to_q32(0.5 * rng.unit()))
        .collect();
    let gamma: Vec<i64> = (0..f.w).map(|_| real_to_q32(0.5 * rng.unit())).collect();
    (x, gamma)
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

fn to_f32(v: i64) -> f32 {
    (v as f64 / 4294967296.0) as f32
}

#[test]
fn rmsnorm_forward_matches_cpu() {
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
            label: "t=1,w=8",
            t: 1,
            w: 8,
            bound: 210.0,
        },
        Fixture {
            label: "t=7,w=64",
            t: 7,
            w: 64,
            bound: 700.0,
        },
        Fixture {
            label: "t=16,w=896",
            t: 16,
            w: 896,
            bound: 1140.0,
        },
    ];
    for f in &fixtures {
        run_fixture(&vk, f);
    }
}

fn run_fixture(vk: &Vk, f: &Fixture) {
    let (t, w) = (f.t, f.w);
    let (x, gamma) = inputs(f);

    // CPU oracle: per-row `dense::rmsnorm_fwd`.
    let mut cpu_out = vec![0i64; t * w];
    for r in 0..t {
        let row = rmsnorm_fwd(&x[r * w..(r + 1) * w], &gamma, EPS_Q32).expect("cpu rmsnorm_fwd");
        cpu_out[r * w..(r + 1) * w].copy_from_slice(&row);
    }

    // Upload the same Q32.32 words as f32.
    let xf: Vec<f32> = x.iter().map(|&v| to_f32(v)).collect();
    let gf: Vec<f32> = gamma.iter().map(|&v| to_f32(v)).collect();
    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let x_buf = unsafe { g.up_f32(&xf) };
    let g_buf = unsafe { g.up_f32(&gf) };
    let out_buf = unsafe { g.dev(t * w * 4) };

    let step = unsafe { rmsnorm_fwd_dispatch(&g, x_buf, g_buf, out_buf, t as u32, w as u32) };

    let (rb, rb_p) = unsafe {
        let (b, m) = vk
            .buffer(
                t as u64 * w as u64 * 4,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .expect("readback buffer");
        (b, vk.map(m).cast::<f32>())
    };
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
            size: (t * w * 4) as u64,
        };
        (vk.copy_buffer)(cb, out_buf, rb, 1, &region);
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
    let gpu_out: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_p, t * w).to_vec() };
    assert!(gpu_out.iter().all(|v| v.is_finite()), "non-finite out");

    // Per-entry comparison in Q32.32 units.
    let mut max = 0i128;
    let mut at = (0usize, 0usize);
    for r in 0..t {
        let mut row_max = 0i128;
        for j in 0..w {
            let d = (i128::from(to_q32(gpu_out[r * w + j])) - i128::from(cpu_out[r * w + j])).abs();
            if d > row_max {
                row_max = d;
            }
            if d > max {
                max = d;
                at = (r, j);
            }
        }
        if t <= 7 {
            println!("  {} row {r}: gpu max |out-cpu| {row_max} units", f.label);
        }
    }
    println!(
        "{}: out max |gpu-cpu| = {max} units at (row {}, col {}); pinned bound {}",
        f.label, at.0, at.1, f.bound
    );
    assert!(
        max as f64 <= f.bound,
        "{}: out error {max} units at (row {}, col {}) exceeds the pinned bound {}",
        f.label,
        at.0,
        at.1,
        f.bound
    );
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
