//! GPU KL loss head (`gpu::nn::kl_row_module`) against the CPU Q32.32 reference
//! `learn::grad::head::kl_row`.
//!
//! The reference computes `loss = Σ p_t·(log p_t − log p_s)` in exact `i128`, rounded once, and
//! `dz = p_s − p_t`, both in Q32.32. The device kernel is f32 with three strided passes (maxima,
//! partition functions, loss/dz) and workgroup tree reductions, so every output is converted to
//! Q32.32 by round-to-nearest and compared: the per-row loss and `dz` elementwise.
//!
//! Shapes `(rows, v) = (1, 8), (7, 64), (3, 1000)`; logits from a fixed LCG in `[-4, 4)`.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_head -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_norm_backward/gpu_attn_fwd tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::fwd::kl_row_dispatch;
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::head::kl_row;

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

fn gpu_identity() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "device name/driver unavailable (nvidia-smi absent)".to_string(),
    }
}

/// One shape and its pinned bounds in Q32.32 units (about twice the measured maximum).
struct Fixture {
    label: &'static str,
    rows: usize,
    v: usize,
    pin_loss: i128,
    pin_dz: i128,
}

#[test]
fn kl_head_matches_cpu() {
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
            label: "rows=1 v=8",
            rows: 1,
            v: 8,
            pin_loss: 400,
            pin_dz: 300,
        },
        Fixture {
            label: "rows=7 v=64",
            rows: 7,
            v: 64,
            pin_loss: 5000,
            pin_dz: 200,
        },
        Fixture {
            label: "rows=3 v=1000",
            rows: 3,
            v: 1000,
            pin_loss: 4000,
            pin_dz: 20,
        },
    ];
    for f in &fixtures {
        run_fixture(&vk, f);
    }
}

fn run_fixture(vk: &Vk, f: &Fixture) {
    let (rows, v) = (f.rows, f.v);
    // Fixed-seed logits in [-4, 4), Q32.32.
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let zs_q: Vec<i64> = (0..rows * v)
        .map(|_| real_to_q32(4.0 * rng.unit()))
        .collect();
    let zt_q: Vec<i64> = (0..rows * v)
        .map(|_| real_to_q32(4.0 * rng.unit()))
        .collect();

    // CPU reference, per row.
    let mut ref_loss = vec![0i64; rows];
    let mut ref_dz = vec![0i64; rows * v];
    for r in 0..rows {
        let (loss, dz) =
            kl_row(&zs_q[r * v..(r + 1) * v], &zt_q[r * v..(r + 1) * v]).expect("cpu kl_row");
        ref_loss[r] = loss;
        ref_dz[r * v..(r + 1) * v].copy_from_slice(&dz);
    }

    let zs_f: Vec<f32> = zs_q.iter().map(|&x| to_f32(x)).collect();
    let zt_f: Vec<f32> = zt_q.iter().map(|&x| to_f32(x)).collect();
    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let zs_buf = unsafe { g.up_f32(&zs_f) };
    let zt_buf = unsafe { g.up_f32(&zt_f) };
    let loss_buf = unsafe { g.dev(rows * 4) };
    let dz_buf = unsafe { g.dev(rows * v * 4) };

    let step =
        unsafe { kl_row_dispatch(&g, zs_buf, zt_buf, loss_buf, dz_buf, rows as u32, v as u32) };

    let host = |n: usize| unsafe {
        let (b, m) = vk
            .buffer(
                n as u64 * 4,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .expect("readback buffer");
        (b, vk.map(m).cast::<f32>())
    };
    let (rb_loss, rb_loss_p) = host(rows);
    let (rb_dz, rb_dz_p) = host(rows * v);
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
        let loss_region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (rows * 4) as u64,
        };
        let dz_region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (rows * v * 4) as u64,
        };
        (vk.copy_buffer)(cb, loss_buf, rb_loss, 1, &loss_region);
        (vk.copy_buffer)(cb, dz_buf, rb_dz, 1, &dz_region);
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
    let gpu_loss: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_loss_p, rows).to_vec() };
    let gpu_dz: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dz_p, rows * v).to_vec() };
    assert!(gpu_loss.iter().all(|x| x.is_finite()), "non-finite loss");
    assert!(gpu_dz.iter().all(|x| x.is_finite()), "non-finite dz");

    let mut max_loss = 0i128;
    let mut at_loss = 0usize;
    for r in 0..rows {
        let d = (i128::from(to_q32(gpu_loss[r])) - i128::from(ref_loss[r])).abs();
        if d > max_loss {
            max_loss = d;
            at_loss = r;
        }
    }
    let mut max_dz = 0i128;
    let mut at_dz = 0usize;
    for i in 0..rows * v {
        let d = (i128::from(to_q32(gpu_dz[i])) - i128::from(ref_dz[i])).abs();
        if d > max_dz {
            max_dz = d;
            at_dz = i;
        }
    }
    println!(
        "{:<16} loss max {max_loss} units (row {at_loss}, pin {}), dz max {max_dz} units \
         (index {at_dz}, pin {})",
        f.label, f.pin_loss, f.pin_dz
    );
    assert!(
        max_loss <= f.pin_loss,
        "{}: loss error {max_loss} exceeds pin {}",
        f.label,
        f.pin_loss
    );
    assert!(
        max_dz <= f.pin_dz,
        "{}: dz error {max_dz} exceeds pin {}",
        f.label,
        f.pin_dz
    );
}
