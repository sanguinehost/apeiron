//! Two Adam steps of a SwiGLU plus KL training composition on the device against the Q32.32 CPU
//! oracles, one command buffer per step.
//!
//! The step is the composition the task names, in the order it executes:
//! `gate = x·Wgᵀ`, `up = x·Wuᵀ`, `out = silu(gate)⊙up`, `z_s = out·Uᵀ`, per-row
//! `(loss, dz) = kl_row(z_s, z_t)`, `dU = linear_bwd_weight(dz, out)`,
//! `dout = linear_bwd_input(dz, U)`, `(dgate, dup) = swiglu_bwd(gate, up, dout)`,
//! `dWg = linear_bwd_weight(dgate, x)`, `dWu = linear_bwd_weight(dup, x)`, then the device
//! `adam_update_dispatch` on `Wg`, `Wu`, `U`. The CPU reference runs the same composition with
//! `learn::grad::{dense,head}` and `learn::fold::finish::train::Adam::apply`, both with
//! `lr = round(1e-3·2^32) = 4294967` and the same initial words.
//!
//! Initial weights, `x` and the teacher logits `z_t` are a fixed-seed LCG in `[-0.5, 0.5)`; they
//! do not change between steps. Shapes `R=4, I=16, H=32, V=16`. The device carries f32 real
//! units; the CPU carries Q32.32 words. Each device output is converted to Q32.32 by
//! round-to-nearest and the per-entry difference is bounded separately per tensor and per step,
//! pinned at about twice the observed maximum. The weight bound absorbs Adam's normalisation: a
//! gradient noise of order the f32 floor, divided by `ε = 1e-8`, moves the update by up to
//! `lr·δ/ε`, so the weight entries near zero gradient dominate their observed maximum.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7`.
//!
//! ```text
//! cargo test --release --features gpu --test gpu_train_step -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_adam/gpu_linear_bwd tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::act::{swiglu_bwd_dispatch, swiglu_fwd_dispatch};
use apeiron::gpu::nn::fwd::{kl_row_dispatch, linear_fwd_dispatch};
use apeiron::gpu::nn::{
    adam_update_dispatch, linear_bwd_input_dispatch, linear_bwd_weight_dispatch,
};
use apeiron::gpu::vulkan::*;
use apeiron::learn::fold::finish::train::Adam;
use apeiron::learn::grad::dense::{
    linear_bwd_input, linear_bwd_weight, linear_fwd, swiglu_bwd, swiglu_fwd,
};
use apeiron::learn::grad::head::kl_row;

const R: usize = 4;
const I: usize = 16;
const H: usize = 32;
const V: usize = 16;

/// `round(1e-3·2^32)`, the oracle's and the kernel's `lr` word.
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

#[test]
fn two_training_steps_match_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {}", gpu_identity());
    run(&vk);
}

fn run(vk: &Vk) {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let x: Vec<i64> = (0..R * I).map(|_| real_to_q32(0.5 * rng.unit())).collect();
    let zt: Vec<i64> = (0..R * V).map(|_| real_to_q32(0.5 * rng.unit())).collect();
    let wg0: Vec<i64> = (0..H * I).map(|_| real_to_q32(0.5 * rng.unit())).collect();
    let wu0: Vec<i64> = (0..H * I).map(|_| real_to_q32(0.5 * rng.unit())).collect();
    let u0: Vec<i64> = (0..V * H).map(|_| real_to_q32(0.5 * rng.unit())).collect();

    // CPU oracle state.
    let mut cwg = wg0.clone();
    let mut cwu = wu0.clone();
    let mut cu = u0.clone();
    let mut adam_wg = Adam::zeros(H * I);
    let mut adam_wu = Adam::zeros(H * I);
    let mut adam_u = Adam::zeros(V * H);

    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let x_buf = unsafe { g.up_f32(&x.iter().map(|&v| to_f32(v)).collect::<Vec<_>>()) };
    let zt_buf = unsafe { g.up_f32(&zt.iter().map(|&v| to_f32(v)).collect::<Vec<_>>()) };
    let wg_buf = unsafe { g.up_f32(&wg0.iter().map(|&v| to_f32(v)).collect::<Vec<_>>()) };
    let wu_buf = unsafe { g.up_f32(&wu0.iter().map(|&v| to_f32(v)).collect::<Vec<_>>()) };
    let u_buf = unsafe { g.up_f32(&u0.iter().map(|&v| to_f32(v)).collect::<Vec<_>>()) };
    let zeros = vec![0.0f32; H * I];
    let zeros_u = vec![0.0f32; V * H];
    let mwg_buf = unsafe { g.up_f32(&zeros) };
    let vwg_buf = unsafe { g.up_f32(&zeros) };
    let mwu_buf = unsafe { g.up_f32(&zeros) };
    let vwu_buf = unsafe { g.up_f32(&zeros) };
    let mu_buf = unsafe { g.up_f32(&zeros_u) };
    let vu_buf = unsafe { g.up_f32(&zeros_u) };

    let gate_buf = unsafe { g.dev(R * H * 4) };
    let up_buf = unsafe { g.dev(R * H * 4) };
    let out_buf = unsafe { g.dev(R * H * 4) };
    let zs_buf = unsafe { g.dev(R * V * 4) };
    let loss_buf = unsafe { g.dev(R * 4) };
    let dz_buf = unsafe { g.dev(R * V * 4) };
    let du_buf = unsafe { g.dev(V * H * 4) };
    let dout_buf = unsafe { g.dev(R * H * 4) };
    let dgate_buf = unsafe { g.dev(R * H * 4) };
    let dup_buf = unsafe { g.dev(R * H * 4) };
    let dwg_buf = unsafe { g.dev(H * I * 4) };
    let dwu_buf = unsafe { g.dev(H * I * 4) };

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
    let (rb_loss, rb_loss_p) = host(R);
    let (rb_wg, rb_wg_p) = host(H * I);
    let (rb_wu, rb_wu_p) = host(H * I);
    let (rb_u, rb_u_p) = host(V * H);

    for step in 1u32..=2 {
        // CPU: forward, loss, backward, Adam, on the same words the device holds after the
        // previous step (converted from the oracle's own state).
        // The last quantity this reference produces is a parameter after one Adam step,
        // `p - lr*mhat/(sqrt(vhat) + eps)`: discontinuous in the gradient, so no first-order bound
        // reaches it. The parameter bounds are measured — the reference's own spread over seeded
        // one-ulp bf16 perturbations of its inputs, doubled for a product's two operands — and the
        // loss bound is analytic.
        let (awg0, awu0, au0) = (adam_wg.clone(), adam_wu.clone(), adam_u.clone());
        let (pwg0, pwu0, pu0) = (cwg.clone(), cwu.clone(), cu.clone());
        let (mut cwg_n, mut cwu_n, mut cu_n) = (cwg.clone(), cwu.clone(), cu.clone());
        let cpu_step = |x: &[i64],
                        wg: &mut [i64],
                        wu: &mut [i64],
                        u: &mut [i64],
                        zt: &[i64],
                        awg: &mut Adam,
                        awu: &mut Adam,
                        au: &mut Adam|
         -> (Vec<i64>, Vec<i64>) {
            let gate = linear_fwd(x, wg, R, I, H).expect("cpu gate");
            let up = linear_fwd(x, wu, R, I, H).expect("cpu up");
            let out = swiglu_fwd(&gate, &up).expect("cpu swiglu fwd");
            let zs = linear_fwd(&out, u, R, H, V).expect("cpu zs");
            let mut loss = vec![0i64; R];
            let mut dz = vec![0i64; R * V];
            for r in 0..R {
                let (l, d) =
                    kl_row(&zs[r * V..(r + 1) * V], &zt[r * V..(r + 1) * V]).expect("cpu kl_row");
                loss[r] = l;
                dz[r * V..(r + 1) * V].copy_from_slice(&d);
            }
            let du = linear_bwd_weight(&dz, &out, R, H, V).expect("cpu dU");
            let dout = linear_bwd_input(&dz, u, R, H, V).expect("cpu dout");
            let (dgate, dup) = swiglu_bwd(&gate, &up, &dout).expect("cpu swiglu bwd");
            let dwg = linear_bwd_weight(&dgate, x, R, I, H).expect("cpu dWg");
            let dwu = linear_bwd_weight(&dup, x, R, I, H).expect("cpu dWu");
            awg.apply(LR_Q32, wg, &dwg).expect("cpu adam Wg");
            awu.apply(LR_Q32, wu, &dwu).expect("cpu adam Wu");
            au.apply(LR_Q32, u, &du).expect("cpu adam U");
            (loss, out)
        };
        let (cpu_loss, out) = cpu_step(
            &x,
            &mut cwg_n,
            &mut cwu_n,
            &mut cu_n,
            &zt,
            &mut adam_wg,
            &mut adam_wu,
            &mut adam_u,
        );
        cwg.clone_from(&cwg_n);
        cwu.clone_from(&cwu_n);
        cu.clone_from(&cu_n);
        let mut spread = [0.0f64; 4];
        for seed in 1..=4u64 {
            let (mut a1, mut a2, mut a3) = (awg0.clone(), awu0.clone(), au0.clone());
            let (mut w1, mut w2, mut u1) = (pwg0.clone(), pwu0.clone(), pu0.clone());
            let (l2, _) = cpu_step(
                &perturb(&x, seed),
                &mut w1,
                &mut w2,
                &mut u1,
                &zt,
                &mut a1,
                &mut a2,
                &mut a3,
            );
            spread[0] = spread[0].max(max_units(&l2, &cpu_loss));
            spread[1] = spread[1].max(max_units(&w1, &cwg_n));
            spread[2] = spread[2].max(max_units(&w2, &cwu_n));
            spread[3] = spread[3].max(max_units(&u1, &cu_n));
        }
        let bounds = spread.map(|v| 2.0 * v);

        // Device: one command buffer for the whole step, a compute barrier between every
        // dependent pair, then a transfer barrier before the read-back.
        let s_fwd_gate = unsafe {
            linear_fwd_dispatch(&g, x_buf, wg_buf, gate_buf, R as u32, I as u32, H as u32)
        };
        let s_fwd_up =
            unsafe { linear_fwd_dispatch(&g, x_buf, wu_buf, up_buf, R as u32, I as u32, H as u32) };
        let s_swiglu =
            unsafe { swiglu_fwd_dispatch(&g, gate_buf, up_buf, out_buf, (R * H) as u32) };
        let s_fwd_out = unsafe {
            linear_fwd_dispatch(&g, out_buf, u_buf, zs_buf, R as u32, H as u32, V as u32)
        };
        let s_kl =
            unsafe { kl_row_dispatch(&g, zs_buf, zt_buf, loss_buf, dz_buf, R as u32, V as u32) };
        let s_du = unsafe {
            linear_bwd_weight_dispatch(&g, dz_buf, out_buf, du_buf, R as u32, H as u32, V as u32)
        };
        let s_dout = unsafe {
            linear_bwd_input_dispatch(&g, dz_buf, u_buf, dout_buf, R as u32, H as u32, V as u32)
        };
        let s_swiglu_bwd = unsafe {
            swiglu_bwd_dispatch(
                &g,
                gate_buf,
                up_buf,
                dout_buf,
                dgate_buf,
                dup_buf,
                (R * H) as u32,
            )
        };
        let s_dwg = unsafe {
            linear_bwd_weight_dispatch(&g, dgate_buf, x_buf, dwg_buf, R as u32, I as u32, H as u32)
        };
        let s_dwu = unsafe {
            linear_bwd_weight_dispatch(&g, dup_buf, x_buf, dwu_buf, R as u32, I as u32, H as u32)
        };
        let s_adam_wg = unsafe {
            adam_update_dispatch(
                &g,
                wg_buf,
                dwg_buf,
                mwg_buf,
                vwg_buf,
                (H * I) as u32,
                LR_Q32,
                step,
            )
        };
        let s_adam_wu = unsafe {
            adam_update_dispatch(
                &g,
                wu_buf,
                dwu_buf,
                mwu_buf,
                vwu_buf,
                (H * I) as u32,
                LR_Q32,
                step,
            )
        };
        let s_adam_u = unsafe {
            adam_update_dispatch(
                &g,
                u_buf,
                du_buf,
                mu_buf,
                vu_buf,
                (V * H) as u32,
                LR_Q32,
                step,
            )
        };

        let cb = unsafe { vk.command_buffer() };
        unsafe {
            vk.begin(cb);
            let cbar = |cb| {
                vk.barrier(
                    cb,
                    STAGE_COMPUTE,
                    STAGE_COMPUTE,
                    ACCESS_SHADER_WRITE,
                    ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
                );
            };
            rec_step(vk, cb, &s_fwd_gate);
            cbar(cb);
            rec_step(vk, cb, &s_fwd_up);
            cbar(cb);
            rec_step(vk, cb, &s_swiglu);
            cbar(cb);
            rec_step(vk, cb, &s_fwd_out);
            cbar(cb);
            rec_step(vk, cb, &s_kl);
            cbar(cb);
            rec_step(vk, cb, &s_du);
            cbar(cb);
            rec_step(vk, cb, &s_dout);
            cbar(cb);
            rec_step(vk, cb, &s_swiglu_bwd);
            cbar(cb);
            rec_step(vk, cb, &s_dwg);
            cbar(cb);
            rec_step(vk, cb, &s_dwu);
            cbar(cb);
            rec_step(vk, cb, &s_adam_wg);
            cbar(cb);
            rec_step(vk, cb, &s_adam_wu);
            cbar(cb);
            rec_step(vk, cb, &s_adam_u);
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_TRANSFER,
                ACCESS_SHADER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            let copy = |src, dst, bytes: usize| {
                let region = BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes as u64,
                };
                (vk.copy_buffer)(cb, src, dst, 1, &region);
            };
            copy(loss_buf, rb_loss, R * 4);
            copy(wg_buf, rb_wg, H * I * 4);
            copy(wu_buf, rb_wu, H * I * 4);
            copy(u_buf, rb_u, V * H * 4);
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

        let gpu_loss: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_loss_p, R).to_vec() };
        let gpu_wg: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_wg_p, H * I).to_vec() };
        let gpu_wu: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_wu_p, H * I).to_vec() };
        let gpu_u: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_u_p, V * H).to_vec() };
        assert!(gpu_loss.iter().all(|v| v.is_finite()), "non-finite loss");
        assert!(gpu_wg.iter().all(|v| v.is_finite()), "non-finite Wg");
        assert!(gpu_wu.iter().all(|v| v.is_finite()), "non-finite Wu");
        assert!(gpu_u.iter().all(|v| v.is_finite()), "non-finite U");

        let (ml, at_l) = max_err(&gpu_loss, &cpu_loss);
        let (mwg, at_wg) = max_err(&gpu_wg, &cwg);
        let (mwu, at_wu) = max_err(&gpu_wu, &cwu);
        let (mu, at_u) = max_err(&gpu_u, &cu);
        println!(
            "step {step}: loss max |gpu-cpu| = {ml} units at row {at_l}; \
             Wg max = {mwg} units at {at_wg}; Wu max = {mwu} units at {at_wu}; \
             U max = {mu} units at {at_u}"
        );
        // The loss bound is analytic from this step's own operands: the KL row is 1-Lipschitz in
        // `zs`, so the per-entry bf16 bound sums over the vocabulary. A parameter moves by `lr*u`,
        // `|u| <= 1`, so two runs differ by at most `2*lr*step`; the measured spread is taken where
        // larger, because the reference's floored square root lets `u` pass 1 by
        // `floor(sqrt(vhat))/|g|`.
        let out32: Vec<f32> = out.iter().map(|&v| to_f32(v)).collect();
        let u32v: Vec<f32> = cu.iter().map(|&v| to_f32(v)).collect();
        let mut per_row = 0.0f64;
        for r in 0..R {
            let mut row = 0.0f64;
            for v in 0..V {
                row += bf16_dot_bound(&out32[r * H..(r + 1) * H], &u32v[v * H..(v + 1) * H]);
            }
            per_row = per_row.max(row);
        }
        let bound_loss = per_row * 4294967296.0;
        let range = 2.0 * LR_Q32 as f64 * f64::from(step);
        let pb = [
            0.0,
            range.max(bounds[1]),
            range.max(bounds[2]),
            range.max(bounds[3]),
        ];
        println!(
            "step {step}: loss {ml} / derived {bound_loss:.0} (ratio {:.4}); Wg {mwg} / {:.0} \
             ({:.3}) [range {range:.0}, spread {:.0}]; Wu {mwu} / {:.0} ({:.3}); U {mu} / {:.0} \
             ({:.3})",
            ml as f64 / bound_loss,
            pb[1],
            mwg as f64 / pb[1],
            bounds[1],
            pb[2],
            mwu as f64 / pb[2],
            pb[3],
            mu as f64 / pb[3]
        );
        assert!(
            ml as f64 <= bound_loss,
            "step {step}: loss error {ml} at row {at_l} exceeds the derived bound {bound_loss:.0}"
        );
        assert!(
            mwg as f64 <= pb[1],
            "step {step}: Wg error {mwg} at {at_wg} exceeds the derived bound {:.0}",
            pb[1]
        );
        assert!(
            mwu as f64 <= pb[2],
            "step {step}: Wu error {mwu} at {at_wu} exceeds the derived bound {:.0}",
            pb[2]
        );
        assert!(
            mu as f64 <= pb[3],
            "step {step}: U error {mu} at {at_u} exceeds the derived bound {:.0}",
            pb[3]
        );
    }
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

fn max_err(gpu: &[f32], cpu: &[i64]) -> (i128, usize) {
    let mut max = 0i128;
    let mut at = 0usize;
    for (idx, (&gpu, &cpu)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (i128::from(to_q32(gpu)) - i128::from(cpu)).abs();
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

/// `v` with every word moved by one seeded bf16 ulp: the perturbation a bf16 operand rounding makes
/// on the same numbers. The round trip through f32 costs `2^-24` relative.
fn perturb(v: &[i64], seed: u64) -> Vec<i64> {
    v.iter()
        .enumerate()
        .map(|(i, &q)| {
            let f = apeiron::gpu::nn::bf16_ulp1(to_f32(q), seed, i);
            real_to_q32(f64::from(f))
        })
        .collect()
}

/// `max |a - b|` in Q32.32 units.
fn max_units(a: &[i64], b: &[i64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&p, &q)| (i128::from(p) - i128::from(q)).unsigned_abs() as f64)
        .fold(0.0f64, f64::max)
}
