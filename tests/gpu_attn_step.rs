//! Two Adam training steps through one causal GQA attention block on the device against the
//! Q32.32 CPU oracles, one command buffer per step, weights and moments device-resident.
//!
//! The step is the composition the task names, in the order it executes:
//! `q = x·Wqᵀ`, `k = x·Wkᵀ`, `v = x·Wvᵀ` (`linear_fwd_dispatch` ×3), causal GQA attention
//! `(p, out) = attn(q,k,v)` (`attn_fwd_dispatch`), `z_s = out·Uᵀ` (`linear_fwd_dispatch`),
//! per-row `(loss, dz) = kl_row(z_s, z_t)` (`kl_row_dispatch`), `dU = linear_bwd_weight(dz, out)`,
//! `dout = linear_bwd_input(dz, U)` (`linear_bwd_weight_dispatch`, `linear_bwd_input_dispatch`),
//! `(dq,dk,dv) = attn_bwd(q,k,v,p,dout)` (`attn_bwd_rowsum`, `_ds`, `_dq`, `_dkdv` dispatches),
//! `dWq = linear_bwd_weight(dq, x)`, `dWk = linear_bwd_weight(dk, x)`,
//! `dWv = linear_bwd_weight(dv, x)`, then the device `adam_update_dispatch` on `Wq`, `Wk`, `Wv`,
//! `U`. The CPU reference runs the same composition with `learn::grad::{dense, attn, head}` and
//! `learn::fold::finish::train::Adam::apply`, both with `lr = round(1e-3·2^32) = 4294967` and the
//! same initial words.
//!
//! Shapes `t=8, i=8, nh=2, nkv=1, hd=8, v=8`; weights are a fixed-seed LCG in `[-0.5, 0.5)`, `x`
//! and the teacher logits `z_t` likewise, and none of them changes between steps. The device
//! carries f32 real units; the CPU carries Q32.32 words. Each device output is converted to
//! Q32.32 by round-to-nearest and the per-entry difference is bounded separately per tensor and
//! per step, pinned at about twice the observed maximum.
//!
//! **The bounds here are looser than the toy training-step bounds, by construction.** The
//! attention backward's `rowsum` reduces its key row over a fixed 256-lane shared-memory tree and
//! `dk`/`dv` sum `h` then `i` in f32, while `attn_bwd_kern` sums the same terms in `i128` and
//! rounds once; the forward softmax is likewise evaluated in f32 per lane. The oracle and the
//! device therefore disagree on association, not only on precision, and the difference reaches the
//! weight gradients through four more products before Adam normalises it. The weight bound also
//! absorbs Adam's `ε = 1e-8` denominator: a gradient difference of the f32 floor moves a
//! near-zero-gradient weight entry by up to `lr·δ/ε`.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7` (lane APEIRON-GPU-TRAIN-W1-T18).
//!
//! ```text
//! cargo test --release --features gpu --test gpu_attn_step -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_attn_bwd/gpu_train_step tests"
)]

use apeiron::gpu::decode::{rec_step, GpuCtx};
use apeiron::gpu::nn::fwd::{kl_row_dispatch, linear_fwd_dispatch};
use apeiron::gpu::nn::{
    adam_update_dispatch, attn_bwd_dkdv_dispatch, attn_bwd_dq_dispatch, attn_bwd_ds_dispatch,
    attn_bwd_rowsum_dispatch, attn_fwd_dispatch, linear_bwd_input_dispatch,
    linear_bwd_weight_dispatch,
};
use apeiron::gpu::vulkan::*;
use apeiron::learn::fold::finish::train::Adam;
use apeiron::learn::grad::attn::{attn_bwd_kern, attn_fwd_kern};
use apeiron::learn::grad::dense::{linear_bwd_input, linear_bwd_weight, linear_fwd};
use apeiron::learn::grad::head::kl_row;

const T: usize = 8;
const I: usize = 8;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 8;
const V: usize = 8;
/// Query width `nh·hd` (`q`, `dout`, `out`).
const QW: usize = NH * HD;
/// Key/value width `nkv·hd` (`k`, `v`, `dk`, `dv`).
const KW: usize = NKV * HD;

/// `round(1e-3·2^32)`, the oracle's and the kernel's `lr` word.
const LR_Q32: i64 = 4_294_967;

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

/// Pinned per-entry bounds in Q32.32 units (about twice the observed maximum), per step.
struct Pins {
    loss: f64,
    wq: f64,
    wk: f64,
    wv: f64,
    u: f64,
}

#[test]
fn attention_block_training_steps_match_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {}", gpu_identity());
    // Pinned at about twice the measured maximum: step 1 / step 2.
    // Observed |gpu-cpu| in Q32.32 units (RTX 5080, driver 615.78.08), bit-identical over three
    // runs: step 1 loss 1752, Wq 167077, Wk 1325918, Wv 168, U 295; step 2 loss 687, Wq 194833,
    // Wk 1355977, Wv 254, U 252.
    let pins = [
        Pins {
            loss: 3_500.0,
            wq: 340_000.0,
            wk: 2_700_000.0,
            wv: 600.0,
            u: 600.0,
        },
        Pins {
            loss: 1_400.0,
            wq: 390_000.0,
            wk: 2_800_000.0,
            wv: 600.0,
            u: 600.0,
        },
    ];
    run(&vk, &pins);
}

#[allow(clippy::too_many_lines)]
fn run(vk: &Vk, pins: &[Pins; 2]) {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let x: Vec<i64> = (0..T * I).map(|_| real_to_q32(rng.unit())).collect();
    let zt: Vec<i64> = (0..T * V).map(|_| real_to_q32(rng.unit())).collect();
    let wq0: Vec<i64> = (0..QW * I).map(|_| real_to_q32(rng.unit())).collect();
    let wk0: Vec<i64> = (0..KW * I).map(|_| real_to_q32(rng.unit())).collect();
    let wv0: Vec<i64> = (0..KW * I).map(|_| real_to_q32(rng.unit())).collect();
    let u0: Vec<i64> = (0..V * QW).map(|_| real_to_q32(rng.unit())).collect();

    // CPU oracle state.
    let mut cwq = wq0.clone();
    let mut cwk = wk0.clone();
    let mut cwv = wv0.clone();
    let mut cu = u0.clone();
    let mut adam_wq = Adam::zeros(QW * I);
    let mut adam_wk = Adam::zeros(KW * I);
    let mut adam_wv = Adam::zeros(KW * I);
    let mut adam_u = Adam::zeros(V * QW);

    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let up = |words: &[i64]| -> u64 {
        let real: Vec<f32> = words.iter().map(|&v| to_f32(v)).collect();
        unsafe { g.up_f32(&real) }
    };
    let x_buf = up(&x);
    let zt_buf = up(&zt);
    let wq_buf = up(&wq0);
    let wk_buf = up(&wk0);
    let wv_buf = up(&wv0);
    let u_buf = up(&u0);
    let zeros_qw_i = vec![0.0f32; QW * I];
    let zeros_kw_i = vec![0.0f32; KW * I];
    let zeros_v_qw = vec![0.0f32; V * QW];
    let mwq_buf = unsafe { g.up_f32(&zeros_qw_i) };
    let vwq_buf = unsafe { g.up_f32(&zeros_qw_i) };
    let mwk_buf = unsafe { g.up_f32(&zeros_kw_i) };
    let vwk_buf = unsafe { g.up_f32(&zeros_kw_i) };
    let mwv_buf = unsafe { g.up_f32(&zeros_kw_i) };
    let vwv_buf = unsafe { g.up_f32(&zeros_kw_i) };
    let mu_buf = unsafe { g.up_f32(&zeros_v_qw) };
    let vu_buf = unsafe { g.up_f32(&zeros_v_qw) };

    let q_buf = unsafe { g.dev(T * QW * 4) };
    let k_buf = unsafe { g.dev(T * KW * 4) };
    let val_buf = unsafe { g.dev(T * KW * 4) };
    let p_buf = unsafe { g.dev(NH * T * T * 4) };
    let out_buf = unsafe { g.dev(T * QW * 4) };
    let zs_buf = unsafe { g.dev(T * V * 4) };
    let loss_buf = unsafe { g.dev(T * 4) };
    let dz_buf = unsafe { g.dev(T * V * 4) };
    let du_buf = unsafe { g.dev(V * QW * 4) };
    let dout_buf = unsafe { g.dev(T * QW * 4) };
    let rs_buf = unsafe { g.dev(T * NH * 4) };
    let ds_buf = unsafe { g.dev(NH * T * T * 4) };
    let dq_buf = unsafe { g.dev(T * QW * 4) };
    let dk_buf = unsafe { g.dev(T * KW * 4) };
    let dv_buf = unsafe { g.dev(T * KW * 4) };
    let dwq_buf = unsafe { g.dev(QW * I * 4) };
    let dwk_buf = unsafe { g.dev(KW * I * 4) };
    let dwv_buf = unsafe { g.dev(KW * I * 4) };

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
    let (rb_loss, rb_loss_p) = host(T);
    let (rb_wq, rb_wq_p) = host(QW * I);
    let (rb_wk, rb_wk_p) = host(KW * I);
    let (rb_wv, rb_wv_p) = host(KW * I);
    let (rb_u, rb_u_p) = host(V * QW);

    let (tn, ti, tw, tq, tv) = (T as u32, I as u32, KW as u32, QW as u32, V as u32);
    let (nhu, nkvu, hdu) = (NH as u32, NKV as u32, HD as u32);

    for (step, pin) in (1u32..=2).zip(pins) {
        // CPU: forward, loss, attention backward, weight gradients, Adam, on the same words the
        // device holds after the previous step.
        let q = linear_fwd(&x, &cwq, T, I, QW).expect("cpu q");
        let k = linear_fwd(&x, &cwk, T, I, KW).expect("cpu k");
        let val = linear_fwd(&x, &cwv, T, I, KW).expect("cpu v");
        let (out, p) = attn_fwd_kern(&q, &k, &val, NH, NKV, HD).expect("cpu attn_fwd_kern");
        let zs = linear_fwd(&out, &cu, T, QW, V).expect("cpu zs");
        let mut cpu_loss = vec![0i64; T];
        let mut dz = vec![0i64; T * V];
        for r in 0..T {
            let (l, d) =
                kl_row(&zs[r * V..(r + 1) * V], &zt[r * V..(r + 1) * V]).expect("cpu kl_row");
            cpu_loss[r] = l;
            dz[r * V..(r + 1) * V].copy_from_slice(&d);
        }
        let du = linear_bwd_weight(&dz, &out, T, QW, V).expect("cpu dU");
        let dout = linear_bwd_input(&dz, &cu, T, QW, V).expect("cpu dout");
        let (cdq, cdk, cdv) =
            attn_bwd_kern(&q, &k, &val, &p, &dout, NH, NKV, HD).expect("cpu attn_bwd_kern");
        let dwq = linear_bwd_weight(&cdq, &x, T, I, QW).expect("cpu dWq");
        let dwk = linear_bwd_weight(&cdk, &x, T, I, KW).expect("cpu dWk");
        let dwv = linear_bwd_weight(&cdv, &x, T, I, KW).expect("cpu dWv");

        adam_wq.apply(LR_Q32, &mut cwq, &dwq).expect("cpu adam Wq");
        adam_wk.apply(LR_Q32, &mut cwk, &dwk).expect("cpu adam Wk");
        adam_wv.apply(LR_Q32, &mut cwv, &dwv).expect("cpu adam Wv");
        adam_u.apply(LR_Q32, &mut cu, &du).expect("cpu adam U");

        // Device: one command buffer for the whole step, a compute barrier between every
        // dependent pair, then a transfer barrier before the read-back.
        let s_q = unsafe { linear_fwd_dispatch(&g, x_buf, wq_buf, q_buf, tn, ti, tq) };
        let s_k = unsafe { linear_fwd_dispatch(&g, x_buf, wk_buf, k_buf, tn, ti, tw) };
        let s_v = unsafe { linear_fwd_dispatch(&g, x_buf, wv_buf, val_buf, tn, ti, tw) };
        let s_attn = unsafe {
            attn_fwd_dispatch(
                &g, q_buf, k_buf, val_buf, p_buf, out_buf, tn, nhu, nkvu, hdu,
            )
        };
        let s_zs = unsafe { linear_fwd_dispatch(&g, out_buf, u_buf, zs_buf, tn, tq, tv) };
        let s_kl = unsafe { kl_row_dispatch(&g, zs_buf, zt_buf, loss_buf, dz_buf, tn, tv) };
        let s_du = unsafe { linear_bwd_weight_dispatch(&g, dz_buf, out_buf, du_buf, tn, tq, tv) };
        let s_dout = unsafe { linear_bwd_input_dispatch(&g, dz_buf, u_buf, dout_buf, tn, tq, tv) };
        let s_rs = unsafe {
            attn_bwd_rowsum_dispatch(&g, val_buf, p_buf, dout_buf, rs_buf, tn, nhu, nkvu, hdu)
        };
        let s_ds = unsafe {
            attn_bwd_ds_dispatch(
                &g, val_buf, p_buf, dout_buf, rs_buf, ds_buf, tn, nhu, nkvu, hdu,
            )
        };
        let s_dq = unsafe { attn_bwd_dq_dispatch(&g, k_buf, ds_buf, dq_buf, tn, nhu, nkvu, hdu) };
        let s_dkdv = unsafe {
            attn_bwd_dkdv_dispatch(
                &g, q_buf, p_buf, ds_buf, dout_buf, dk_buf, dv_buf, tn, nhu, nkvu, hdu,
            )
        };
        let s_dwq = unsafe { linear_bwd_weight_dispatch(&g, dq_buf, x_buf, dwq_buf, tn, ti, tq) };
        let s_dwk = unsafe { linear_bwd_weight_dispatch(&g, dk_buf, x_buf, dwk_buf, tn, ti, tw) };
        let s_dwv = unsafe { linear_bwd_weight_dispatch(&g, dv_buf, x_buf, dwv_buf, tn, ti, tw) };
        let s_adam_wq = unsafe {
            adam_update_dispatch(&g, wq_buf, dwq_buf, mwq_buf, vwq_buf, tq * ti, LR_Q32, step)
        };
        let s_adam_wk = unsafe {
            adam_update_dispatch(&g, wk_buf, dwk_buf, mwk_buf, vwk_buf, tw * ti, LR_Q32, step)
        };
        let s_adam_wv = unsafe {
            adam_update_dispatch(&g, wv_buf, dwv_buf, mwv_buf, vwv_buf, tw * ti, LR_Q32, step)
        };
        let s_adam_u = unsafe {
            adam_update_dispatch(&g, u_buf, du_buf, mu_buf, vu_buf, tv * tq, LR_Q32, step)
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
            rec_step(vk, cb, &s_q);
            rec_step(vk, cb, &s_k);
            rec_step(vk, cb, &s_v);
            cbar(cb);
            rec_step(vk, cb, &s_attn);
            cbar(cb);
            rec_step(vk, cb, &s_zs);
            cbar(cb);
            rec_step(vk, cb, &s_kl);
            cbar(cb);
            rec_step(vk, cb, &s_du);
            rec_step(vk, cb, &s_dout);
            cbar(cb);
            rec_step(vk, cb, &s_rs);
            cbar(cb);
            rec_step(vk, cb, &s_ds);
            cbar(cb);
            rec_step(vk, cb, &s_dq);
            rec_step(vk, cb, &s_dkdv);
            cbar(cb);
            rec_step(vk, cb, &s_dwq);
            rec_step(vk, cb, &s_dwk);
            rec_step(vk, cb, &s_dwv);
            cbar(cb);
            rec_step(vk, cb, &s_adam_wq);
            cbar(cb);
            rec_step(vk, cb, &s_adam_wk);
            cbar(cb);
            rec_step(vk, cb, &s_adam_wv);
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
            copy(loss_buf, rb_loss, T * 4);
            copy(wq_buf, rb_wq, QW * I * 4);
            copy(wk_buf, rb_wk, KW * I * 4);
            copy(wv_buf, rb_wv, KW * I * 4);
            copy(u_buf, rb_u, V * QW * 4);
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

        let gpu_loss: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_loss_p, T).to_vec() };
        let gpu_wq: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_wq_p, QW * I).to_vec() };
        let gpu_wk: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_wk_p, KW * I).to_vec() };
        let gpu_wv: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_wv_p, KW * I).to_vec() };
        let gpu_u: Vec<f32> = unsafe { std::slice::from_raw_parts(rb_u_p, V * QW).to_vec() };
        assert!(gpu_loss.iter().all(|v| v.is_finite()), "non-finite loss");
        assert!(gpu_wq.iter().all(|v| v.is_finite()), "non-finite Wq");
        assert!(gpu_wk.iter().all(|v| v.is_finite()), "non-finite Wk");
        assert!(gpu_wv.iter().all(|v| v.is_finite()), "non-finite Wv");
        assert!(gpu_u.iter().all(|v| v.is_finite()), "non-finite U");

        let (ml, at_l) = max_err(&gpu_loss, &cpu_loss, 1);
        let (mwq, at_wq) = max_err(&gpu_wq, &cwq, I);
        let (mwk, at_wk) = max_err(&gpu_wk, &cwk, I);
        let (mwv, at_wv) = max_err(&gpu_wv, &cwv, I);
        let (mu, at_u) = max_err(&gpu_u, &cu, QW);
        println!(
            "step {step}: loss max |gpu-cpu| = {ml} units at row {at_l:?}; \
             Wq max = {mwq} units at {at_wq:?}; Wk max = {mwk} units at {at_wk:?}; \
             Wv max = {mwv} units at {at_wv:?}; U max = {mu} units at {at_u:?}"
        );
        println!(
            "step {step}: pins loss {} / Wq {} / Wk {} / Wv {} / U {}",
            pin.loss, pin.wq, pin.wk, pin.wv, pin.u
        );
        assert!(
            ml as f64 <= pin.loss,
            "step {step}: loss error {ml} at {at_l:?} exceeds the pin {}",
            pin.loss
        );
        assert!(
            mwq as f64 <= pin.wq,
            "step {step}: Wq error {mwq} at {at_wq:?} exceeds the pin {}",
            pin.wq
        );
        assert!(
            mwk as f64 <= pin.wk,
            "step {step}: Wk error {mwk} at {at_wk:?} exceeds the pin {}",
            pin.wk
        );
        assert!(
            mwv as f64 <= pin.wv,
            "step {step}: Wv error {mwv} at {at_wv:?} exceeds the pin {}",
            pin.wv
        );
        assert!(
            mu as f64 <= pin.u,
            "step {step}: U error {mu} at {at_u:?} exceeds the pin {}",
            pin.u
        );
    }
}

fn max_err(gpu: &[f32], cpu: &[i64], cols: usize) -> (i128, (usize, usize)) {
    let mut max = 0i128;
    let mut at = (0usize, 0usize);
    for (idx, (&gpu, &cpu)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (i128::from(to_q32(gpu)) - i128::from(cpu)).abs();
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
