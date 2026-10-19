//! GPU RMSNorm backward (`gpu::nn::rmsnorm_bwd_module`) against the CPU Q32.32 reference
//! `learn::grad::dense::rmsnorm_bwd` at the training contract's bound.
//!
//! Contract (`tests/gpu_train_baseline.rs`): every gradient entry is within `FD_TOL = 2^14`
//! Q32.32 units of the CPU backward for a fixed upstream gradient. The GPU kernel is f32, so each
//! output is converted to Q32.32 by round-to-nearest with saturation and compared per entry: `dx`
//! on every row, and the `dγ` column sum produced on the device by the second dispatch
//! (`rmsnorm_dgamma_dispatch`, one thread per column, ascending row order, no atomics).
//!
//! Shapes: `t ∈ {1, 7, 512}`, `w = 896` (Qwen2.5-0.5B hidden), `eps = 1e-6`; inputs from a fixed
//! LCG. The fourth fixture is the big-row case `dense::rmsnorm_bwd`'s rounding comment names:
//! `inv_rms ≈ 2^-10`, where rounding `r³` to Q32.32 would cost 2^-5 relative — the f32 kernel
//! keeps `r`, `r²`, `r³` in registers with no intermediate rounding.
//!
//! Board row: `apeiron-gpu-train-w1-t01-EQpaf3sVmt04RegjErRx1k`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_norm_backward -- --nocapture
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
use apeiron::gpu::nn::{rmsnorm_bwd_dispatch, rmsnorm_dgamma_dispatch};
use apeiron::gpu::vulkan::*;
use apeiron::learn::grad::dense::rmsnorm_bwd;

/// Qwen2.5-0.5B hidden width.
const W: usize = 896;
/// `FD_TOL = 2^14` Q32.32 units — the training contract's per-entry bound.
const FD_TOL: f64 = 16384.0;
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

/// One shape: `t` rows at `w = 896`, with the input amplitudes in Q32.32 units.
struct Fixture {
    label: &'static str,
    t: usize,
    /// Peak `|x|` in Q32.32 units.
    x_units: f64,
    /// Peak `|dy|` in Q32.32 units.
    dy_units: f64,
    /// Pinned per-entry bound on `dx`, in Q32.32 units: about twice the measured value and never
    /// above the contract's `FD_TOL`.
    bound_dx: f64,
    /// Pinned per-entry bound on the `dγ` row sum, in Q32.32 units.
    bound_dg: f64,
}

/// Deterministic inputs for one fixture: `x` and `dy` are `t × w`, `γ` is `w` in `[0.5, 1.5)`.
fn inputs(f: &Fixture) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let xa = f.x_units / 4294967296.0;
    let ya = f.dy_units / 4294967296.0;
    let x: Vec<i64> = (0..f.t * W).map(|_| real_to_q32(xa * rng.unit())).collect();
    let dy: Vec<i64> = (0..f.t * W).map(|_| real_to_q32(ya * rng.unit())).collect();
    let gamma: Vec<i64> = (0..W)
        .map(|_| real_to_q32(0.75 + 0.25 * rng.unit()))
        .collect();
    (x, dy, gamma)
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
fn rmsnorm_backward_matches_cpu() {
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
            label: "t=1",
            t: 1,
            x_units: (1u64 << 20) as f64,
            dy_units: (1u64 << 20) as f64,
            bound_dx: 10_000.0,
            bound_dg: 64.0,
        },
        Fixture {
            label: "t=7",
            t: 7,
            x_units: (1u64 << 20) as f64,
            dy_units: (1u64 << 20) as f64,
            bound_dx: 10_000.0,
            bound_dg: 64.0,
        },
        Fixture {
            label: "t=512",
            t: 512,
            x_units: (1u64 << 20) as f64,
            dy_units: (1u64 << 20) as f64,
            bound_dx: 10_000.0,
            bound_dg: 4096.0,
        },
        // mean(x²) ≈ (0.92·√3·2^10)²/3 ≈ 8.87e5 < 2^20 with sample-variance headroom, so r ≈ 2^-9.75 ≈ 2^-10: the big-row
        // fixture `dense::rmsnorm_bwd` names for the r³ rounding argument.
        Fixture {
            label: "bigrow t=7 inv_rms~2^-10",
            t: 7,
            x_units: 0.92 * 3.0f64.sqrt() * 1024.0 * 4294967296.0,
            dy_units: (1u64 << 20) as f64,
            bound_dx: 64.0,
            bound_dg: 64.0,
        },
    ];
    for f in &fixtures {
        run_fixture(&vk, f);
    }
}

fn run_fixture(vk: &Vk, f: &Fixture) {
    let (t, w) = (f.t, W);
    let (x, dy, gamma) = inputs(f);

    // CPU reference: per-row `dense::rmsnorm_bwd`, dγ summed in i64 exactly.
    let mut cpu_dx = vec![0i64; t * w];
    let mut cpu_dg = vec![0i64; w];
    for r in 0..t {
        let (dxr, dgr) = rmsnorm_bwd(
            &x[r * w..(r + 1) * w],
            &gamma,
            EPS_Q32,
            &dy[r * w..(r + 1) * w],
        )
        .expect("cpu rmsnorm_bwd");
        cpu_dx[r * w..(r + 1) * w].copy_from_slice(&dxr);
        for j in 0..w {
            cpu_dg[j] = cpu_dg[j]
                .checked_add(dgr[j])
                .expect("cpu dgamma sum overflow");
        }
    }

    // Upload the same Q32.32 words as f32.
    let xf: Vec<f32> = x.iter().map(|&v| to_f32(v)).collect();
    let yf: Vec<f32> = dy.iter().map(|&v| to_f32(v)).collect();
    let gf: Vec<f32> = gamma.iter().map(|&v| to_f32(v)).collect();
    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let x_buf = unsafe { g.up_f32(&xf) };
    let g_buf = unsafe { g.up_f32(&gf) };
    let y_buf = unsafe { g.up_f32(&yf) };
    let dx_buf = unsafe { g.dev(t * w * 4) };
    let dgr_buf = unsafe { g.dev(t * w * 4) };
    let dg_buf = unsafe { g.dev(w * 4) };

    let step = unsafe {
        rmsnorm_bwd_dispatch(
            &g, x_buf, g_buf, y_buf, dx_buf, dgr_buf, t as u32, w as u32, 1e-6f32,
        )
    };
    let dstep = unsafe { rmsnorm_dgamma_dispatch(&g, dgr_buf, dg_buf, t as u32, w as u32) };

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
    let (rb_dx, rb_dx_p) = host(t * w);
    let (rb_dg, rb_dg_p) = host(w);
    let cb = unsafe { vk.command_buffer() };
    unsafe {
        vk.begin(cb);
        rec_step(vk, cb, &step);
        rec_step(vk, cb, &dstep);
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
            size: (t * w * 4) as u64,
        };
        let dg_region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (w * 4) as u64,
        };
        (vk.copy_buffer)(cb, dx_buf, rb_dx, 1, &dx_region);
        (vk.copy_buffer)(cb, dg_buf, rb_dg, 1, &dg_region);
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
    let gpu_dx: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dx_p, t * w).to_vec() };
    let gpu_dg: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_dg_p, w).to_vec() };
    assert!(gpu_dx.iter().all(|v| v.is_finite()), "non-finite dx");
    assert!(gpu_dg.iter().all(|v| v.is_finite()), "non-finite dgamma");

    // Per-entry comparison in Q32.32 units.
    let mut max_dx = 0i128;
    let mut at_dx = (0usize, 0usize);
    for r in 0..t {
        let mut row_max = 0i128;
        for j in 0..w {
            let d = (i128::from(to_q32(gpu_dx[r * w + j])) - i128::from(cpu_dx[r * w + j])).abs();
            if d > row_max {
                row_max = d;
            }
            if d > max_dx {
                max_dx = d;
                at_dx = (r, j);
            }
        }
        if t <= 7 {
            println!("  {} row {r}: gpu max |dx-cpu| {row_max} units", f.label);
        }
    }
    let mut max_dg = 0i128;
    let mut at_dg = 0usize;
    for j in 0..w {
        let d = (i128::from(to_q32(gpu_dg[j])) - i128::from(cpu_dg[j])).abs();
        if d > max_dg {
            max_dg = d;
            at_dg = j;
        }
    }
    println!(
        "{}: dx max |gpu-cpu| = {max_dx} units at (row {}, col {}); dgamma max = {max_dg} units at col {at_dg}; pinned bounds dx {} / dgamma {}; contract tolerance = {FD_TOL} units",
        f.label, at_dx.0, at_dx.1, f.bound_dx, f.bound_dg
    );
    assert!(
        f.bound_dx <= FD_TOL && f.bound_dg <= FD_TOL,
        "{}: pinned bound exceeds the FD_TOL contract",
        f.label
    );
    assert!(
        max_dx as f64 <= f.bound_dx,
        "{}: dx error {max_dx} units at (row {}, col {}) exceeds the pinned bound {} (FD_TOL {FD_TOL})",
        f.label,
        at_dx.0,
        at_dx.1,
        f.bound_dx
    );
    assert!(
        max_dg as f64 <= f.bound_dg,
        "{}: dgamma error {max_dg} units at col {at_dg} exceeds the pinned bound {} (FD_TOL {FD_TOL})",
        f.label, f.bound_dg
    );
    assert!(
        max_dx as f64 <= FD_TOL && max_dg as f64 <= FD_TOL,
        "{}: gradient exceeds the FD_TOL contract",
        f.label
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
