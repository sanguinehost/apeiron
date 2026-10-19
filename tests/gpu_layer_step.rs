//! Two Adam training steps through one complete pre-norm transformer layer on the device, against
//! the Q32.32 CPU oracles, one command buffer per step, weights and moments device-resident.
//!
//! The step is the composition the task names, in the order it executes:
//! `xn = rmsnorm(x, g1)`; `q = xn·Wqᵀ`, `k = xn·Wkᵀ`, `v = xn·Wvᵀ`; `q = rope(q)`, `k = rope(k)`;
//! `(p, oa) = attn(q,k,v)`; `o = oa·Woᵀ`; `x1 = x + o`; `xn2 = rmsnorm(x1, g2)`;
//! `g = xn2·Wgᵀ`, `u = xn2·Wuᵀ`; `ha = silu(g)⊙u`; `f = ha·Wdᵀ`; `x2 = x1 + f`;
//! `(loss, dz) = kl_row(x2, target)`; `dWd = dzᵀ·ha`, `dha = dz·Wd`, `(dg,du) = swiglu_bwd`,
//! `dxn2 = dg·Wgᵀ + du·Wuᵀ`, `dWg = dgᵀ·xn2`, `dWu = duᵀ·xn2`,
//! `dx1 = dz + rmsnorm_bwd_dx(x1,g2,dxn2)`, `dg2 = dgamma_reduce`, `dWo = dx1ᵀ·oa`,
//! `do = dx1·Wo`, the four attention-backward passes on `(do,q,k,v,p)`, `dq = rope_bwd(dq)`,
//! `dk = rope_bwd(dk)`, `dWq = dqᵀ·xn`, `dWk = dkᵀ·xn`, `dWv = dvᵀ·xn`,
//! `dxn = dq·Wqᵀ + dk·Wkᵀ + dv·Wvᵀ`, `dx = rmsnorm_bwd_dx(x,g1,dxn)`, `dg1 = dgamma_reduce`, then
//! `adam_update_dispatch` on the nine tensors `Wq, Wk, Wv, Wo, Wg, Wu, Wd, g1, g2` with
//! `lr = round(1e-3·2^32) = 4294967` and the dense step test's betas.
//!
//! Shapes `t=4, i=8, h=8, v=8, nh=2, nkv=1, hd=8`, positions = row index; `q`/`oa` are `t×16`,
//! `k`/`v` are `t×8`, and the hidden width equals the vocabulary width, so `kl_row` reads `x2`
//! directly as the row logits. The residual identity means the KL `dz` reaches the direct `dx1`
//! **and** the FFN branch through `dha`; it is added once at each place the composition names and
//! nowhere else.
//!
//! **The bounds are the evidence, and they are loose by construction.** The device carries f32 real
//! units and the CPU Q32.32 words; an f32-versus-Q32.32 difference is created at every one of the
//! forward's seventeen dispatches and is carried back through nineteen more before it reaches `dx`
//! — so it accumulates across far more than twelve chained ops — summed by four
//! attention-backward passes, and finally divided by Adam's `ε = 1e-8` normaliser: a gradient
//! difference at the f32 floor moves a weight entry by up to `lr·δ/ε` after the update. Each row is
//! pinned at about twice the observed maximum and every observed maximum is printed with its entry
//! index. No pin here is loose: each lies within 2.2× of its measurement, and [`check_row`] prints
//! `loose pin` on any row whose pin exceeds that, so a widened bound cannot pass silently. The
//! `Wg` row carries the largest absolute pin (`4.7e6` / `1.11e7` units) for the Adam-`ε` reason
//! above, not because the gate and up projections disagree more than the others.
//!
//! Step 1 compares the input gradient `dx`, both reduced `dgamma`s, and all nine updated tensors;
//! step 2 re-runs the forward on the step-1 weights and compares `loss`, `dx` and all nine again.
//! The device buffers are the same allocation across both steps; only the host read-backs are new.
//!
//! Board row: `apeiron-gpu-train-w1-t05-GVrduzS5mnRVc4oeQ1xRD7` (lane ).
//!
//! ```text
//! cargo test --release --features gpu --test gpu_layer_step -- --nocapture
//! ```
#![cfg(feature = "gpu")]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "measurement test; lint profile follows the gpu_attn_step/gpu_train_step tests"
)]

use apeiron::engine::RopeTable;
use apeiron::gpu::decode::{make_step_named, rec_step, GpuCtx};
use apeiron::gpu::nn::act::{swiglu_bwd_dispatch, swiglu_fwd_dispatch};
use apeiron::gpu::nn::fwd::{
    kl_row_dispatch, linear_fwd_dispatch, rmsnorm_fwd_dispatch, rope_bwd_dispatch,
    rope_fwd_dispatch,
};
use apeiron::gpu::nn::{
    adam_update_dispatch, add_inplace_bytes, attn_bwd_dkdv_dispatch, attn_bwd_dq_dispatch,
    attn_bwd_ds_dispatch, attn_bwd_rowsum_dispatch, attn_fwd_dispatch, linear_bwd_input_dispatch,
    linear_bwd_weight_dispatch, rmsnorm_bwd_dispatch, rmsnorm_dgamma_dispatch,
};
use apeiron::gpu::vulkan::*;
use apeiron::learn::fold::finish::train::Adam;
use apeiron::learn::grad::attn::{attn_bwd_kern, attn_fwd_kern, rope_bwd, rope_fwd};
use apeiron::learn::grad::dense::{
    linear_bwd_input, linear_bwd_weight, linear_fwd, rmsnorm_bwd, rmsnorm_fwd, swiglu_bwd,
    swiglu_fwd,
};
use apeiron::learn::grad::head::kl_row;
use apeiron::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians};

const T: usize = 4;
/// Hidden width; also the vocabulary width the KL head reads, so `dz` is `t × i`.
const I: usize = 8;
/// FFN intermediate width.
const H: usize = 8;
const V: usize = 8;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 8;
/// Query width `nh·hd`.
const QW: usize = NH * HD;
/// Key/value width `nkv·hd`.
const KW: usize = NKV * HD;

/// `round(1e-3·2^32)`, the oracle's and the kernel's `lr` word.
const LR_Q32: i64 = 4_294_967;
/// `eps = 1e-6` in Q32.32 words (the f32 kernels are given `1e-6f32`).
const EPS_Q32: i64 = 4295;
/// Local size of the residual-add kernel and the grid divisor for its dispatch.
const ADD_LOCAL: u32 = 256;

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

/// f32 real units → Q32.32, round-to-nearest and saturating (copied from `gpu_attn_step`).
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

/// `RopeTable::new(F64::from_f64(10000.0), hd)` by its documented turn formula, since `F64` is
/// crate-private.
fn rope_table(hd: usize) -> RopeTable {
    let theta = 10_000.0_f64;
    let unit = 18_446_744_073_709_551_616.0_f64; // 2^64 phases per full turn
    let two_pi = 2.0 * std::f64::consts::PI;
    let turns: Vec<u64> = (0..hd / 2)
        .map(|j| {
            let q = (theta.powf(-2.0 * (j as f64) / (hd as f64)) * unit / two_pi).round();
            assert!(
                q.is_finite() && (0.0..unit).contains(&q),
                "turn out of range"
            );
            q as u64
        })
        .collect();
    RopeTable::from_turns(hd, turns).expect("turn table")
}

/// `cos`/`sin` f32 tables per `(pos, j)`, from the oracle's own integer phase and trig.
fn trig_tables(table: &RopeTable, t: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let d2 = hd / 2;
    let mut cos = vec![0f32; t * d2];
    let mut sin = vec![0f32; t * d2];
    for pos in 0..t {
        for j in 0..d2 {
            let phase = ((u128::from(table.turns()[j]) * u128::from(pos as u32))
                & u128::from(u64::MAX)) as u64;
            let sc = fixed_sin_cos(turn_to_radians(phase)).expect("fixed_sin_cos");
            let off = pos * d2 + j;
            cos[off] = (sc.cos as f64 / 4294967296.0) as f32;
            sin[off] = (sc.sin as f64 / 4294967296.0) as f32;
        }
    }
    (cos, sin)
}

/// Exact Q32.32 add, in place; `a += b`.
fn add(a: &mut [i64], b: &[i64]) {
    assert_eq!(a.len(), b.len(), "add width");
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

/// Per-row `dense::rmsnorm_fwd`, `t × w`.
fn norm_rows(x: &[i64], g: &[i64], t: usize, w: usize) -> Vec<i64> {
    let mut out = vec![0i64; t * w];
    for r in 0..t {
        out[r * w..(r + 1) * w]
            .copy_from_slice(&rmsnorm_fwd(&x[r * w..(r + 1) * w], g, EPS_Q32).expect("cpu norm"));
    }
    out
}

/// Per-row `dense::rmsnorm_bwd` and the exact `dγ` column sum, `t × w`.
fn norm_rows_bwd(x: &[i64], g: &[i64], dy: &[i64], t: usize, w: usize) -> (Vec<i64>, Vec<i64>) {
    let mut dx = vec![0i64; t * w];
    let mut dg = vec![0i64; w];
    for r in 0..t {
        let (dxr, dgr) = rmsnorm_bwd(&x[r * w..(r + 1) * w], g, EPS_Q32, &dy[r * w..(r + 1) * w])
            .expect("cpu norm bwd");
        dx[r * w..(r + 1) * w].copy_from_slice(&dxr);
        for j in 0..w {
            dg[j] = dg[j].checked_add(dgr[j]).expect("cpu dgamma sum");
        }
    }
    (dx, dg)
}

/// Per-row `attn::rope_fwd`, position = row index.
fn rope_rows(x: &[i64], t: usize, row: usize, table: &RopeTable) -> Vec<i64> {
    let mut out = Vec::with_capacity(t * row);
    for r in 0..t {
        out.extend(rope_fwd(&x[r * row..(r + 1) * row], r as u32, table).expect("cpu rope fwd"));
    }
    out
}

/// Per-row `attn::rope_bwd`, position = row index.
fn rope_rows_bwd(dy: &[i64], t: usize, row: usize, table: &RopeTable) -> Vec<i64> {
    let mut out = Vec::with_capacity(t * row);
    for r in 0..t {
        out.extend(rope_bwd(&dy[r * row..(r + 1) * row], r as u32, table).expect("cpu rope bwd"));
    }
    out
}

/// The CPU oracle's weights, gains and Adam state, in the layer's own order.
struct Oracle {
    wq: Vec<i64>,
    wk: Vec<i64>,
    wv: Vec<i64>,
    wo: Vec<i64>,
    wg: Vec<i64>,
    wu: Vec<i64>,
    wd: Vec<i64>,
    g1: Vec<i64>,
    g2: Vec<i64>,
    a_wq: Adam,
    a_wk: Adam,
    a_wv: Adam,
    a_wo: Adam,
    a_wg: Adam,
    a_wu: Adam,
    a_wd: Adam,
    a_g1: Adam,
    a_g2: Adam,
}

/// What the oracle returns per step: the row losses and the input gradient plus both `dγ`s.
struct OracleStep {
    loss: Vec<i64>,
    dx: Vec<i64>,
    dg1: Vec<i64>,
    dg2: Vec<i64>,
}

impl Oracle {
    /// One layer forward, its exact backward against `zt`, then Adam in place on the nine tensors.
    /// `O(t·i·(i + qw + 2·kw + 2·h) + nh·t²·hd)` per step, as the composition's products dictate.
    fn step(&mut self, x: &[i64], zt: &[i64], rope: &RopeTable) -> OracleStep {
        let (t, i, h) = (T, I, H);
        let xn = norm_rows(x, &self.g1, t, i);
        let q0 = linear_fwd(&xn, &self.wq, t, i, QW).expect("cpu q");
        let k0 = linear_fwd(&xn, &self.wk, t, i, KW).expect("cpu k");
        let val = linear_fwd(&xn, &self.wv, t, i, KW).expect("cpu v");
        let q = rope_rows(&q0, t, QW, rope);
        let k = rope_rows(&k0, t, KW, rope);
        let (oa, p) = attn_fwd_kern(&q, &k, &val, NH, NKV, HD).expect("cpu attn_fwd");
        let o = linear_fwd(&oa, &self.wo, t, QW, i).expect("cpu o");
        let mut x1 = x.to_vec();
        add(&mut x1, &o);
        let xn2 = norm_rows(&x1, &self.g2, t, i);
        let gate = linear_fwd(&xn2, &self.wg, t, i, h).expect("cpu gate");
        let up = linear_fwd(&xn2, &self.wu, t, i, h).expect("cpu up");
        let ha = swiglu_fwd(&gate, &up).expect("cpu swiglu_fwd");
        let f = linear_fwd(&ha, &self.wd, t, h, i).expect("cpu f");
        let mut x2 = x1.clone();
        add(&mut x2, &f);
        let mut loss = vec![0i64; t];
        let mut dz = vec![0i64; t * i];
        for r in 0..t {
            let (l, d) =
                kl_row(&x2[r * i..(r + 1) * i], &zt[r * i..(r + 1) * i]).expect("cpu kl_row");
            loss[r] = l;
            dz[r * i..(r + 1) * i].copy_from_slice(&d);
        }

        let dwd = linear_bwd_weight(&dz, &ha, t, h, i).expect("cpu dWd");
        let dha = linear_bwd_input(&dz, &self.wd, t, h, i).expect("cpu dha");
        let (dg, du) = swiglu_bwd(&gate, &up, &dha).expect("cpu swiglu_bwd");
        let mut dxn2 = linear_bwd_input(&dg, &self.wg, t, i, h).expect("cpu dxn2 gate");
        add(
            &mut dxn2,
            &linear_bwd_input(&du, &self.wu, t, i, h).expect("cpu dxn2 up"),
        );
        let dwg = linear_bwd_weight(&dg, &xn2, t, i, h).expect("cpu dWg");
        let dwu = linear_bwd_weight(&du, &xn2, t, i, h).expect("cpu dWu");
        let (mut dx1, dg2) = norm_rows_bwd(&x1, &self.g2, &dxn2, t, i);
        // The residual identity: the KL `dz` is the direct `dx1` term and, through `dha`, the FFN
        // branch's; the two meet here and nowhere else.
        add(&mut dx1, &dz);
        let dwo = linear_bwd_weight(&dx1, &oa, t, QW, i).expect("cpu dWo");
        let d_out = linear_bwd_input(&dx1, &self.wo, t, QW, i).expect("cpu do");
        let (dqr, dkr, dv) =
            attn_bwd_kern(&q, &k, &val, &p, &d_out, NH, NKV, HD).expect("cpu attn_bwd");
        let dq = rope_rows_bwd(&dqr, t, QW, rope);
        let dk = rope_rows_bwd(&dkr, t, KW, rope);
        let dwq = linear_bwd_weight(&dq, &xn, t, i, QW).expect("cpu dWq");
        let dwk = linear_bwd_weight(&dk, &xn, t, i, KW).expect("cpu dWk");
        let dwv = linear_bwd_weight(&dv, &xn, t, i, KW).expect("cpu dWv");
        let mut dxn = linear_bwd_input(&dq, &self.wq, t, i, QW).expect("cpu dxn q");
        add(
            &mut dxn,
            &linear_bwd_input(&dk, &self.wk, t, i, KW).expect("cpu dxn k"),
        );
        add(
            &mut dxn,
            &linear_bwd_input(&dv, &self.wv, t, i, KW).expect("cpu dxn v"),
        );
        let (dx, dg1) = norm_rows_bwd(x, &self.g1, &dxn, t, i);

        self.a_wq
            .apply(LR_Q32, &mut self.wq, &dwq)
            .expect("adam Wq");
        self.a_wk
            .apply(LR_Q32, &mut self.wk, &dwk)
            .expect("adam Wk");
        self.a_wv
            .apply(LR_Q32, &mut self.wv, &dwv)
            .expect("adam Wv");
        self.a_wo
            .apply(LR_Q32, &mut self.wo, &dwo)
            .expect("adam Wo");
        self.a_wg
            .apply(LR_Q32, &mut self.wg, &dwg)
            .expect("adam Wg");
        self.a_wu
            .apply(LR_Q32, &mut self.wu, &dwu)
            .expect("adam Wu");
        self.a_wd
            .apply(LR_Q32, &mut self.wd, &dwd)
            .expect("adam Wd");
        self.a_g1
            .apply(LR_Q32, &mut self.g1, &dg1)
            .expect("adam g1");
        self.a_g2
            .apply(LR_Q32, &mut self.g2, &dg2)
            .expect("adam g2");

        OracleStep { loss, dx, dg1, dg2 }
    }
}

/// Pinned per-entry bounds in Q32.32 units (about twice the observed maximum), per step.
struct Pins {
    loss: f64,
    dx: f64,
    dg1: f64,
    dg2: f64,
    wq: f64,
    wk: f64,
    wv: f64,
    wo: f64,
    wg: f64,
    wu: f64,
    wd: f64,
}

#[test]
fn layer_training_steps_match_cpu() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("SKIP: No Vulkan device present: {e}");
            return;
        }
    };
    println!("device: {}", gpu_identity());
    // Pinned at about twice the measured maximum (RTX 5080, driver 615.78.08, bit-identical over
    // two runs). Observed |gpu-cpu| in Q32.32 units, step 1 / step 2: loss 2457 / 1365, dx 4590 /
    // 3841, dg1 3604 / 2480, dg2 832 / 895, Wq 144321 / 162230, Wk 45458 / 60034, Wv 24359 /
    // 23826, Wo 283 / 481, Wg 2337300 / 5536181, Wu 50569 / 130258, Wd 6581 / 14247.
    let pins = [
        Pins {
            loss: 5_000.0,
            dx: 9_200.0,
            dg1: 7_300.0,
            dg2: 1_700.0,
            wq: 290_000.0,
            wk: 92_000.0,
            wv: 49_000.0,
            wo: 600.0,
            wg: 4_700_000.0,
            wu: 102_000.0,
            wd: 13_200.0,
        },
        Pins {
            loss: 2_800.0,
            dx: 7_700.0,
            dg1: 5_000.0,
            dg2: 1_800.0,
            wq: 325_000.0,
            wk: 121_000.0,
            wv: 48_000.0,
            wo: 1_000.0,
            wg: 11_100_000.0,
            wu: 261_000.0,
            wd: 28_500.0,
        },
    ];
    run(&vk, &pins);
}

fn run(vk: &Vk, pins: &[Pins; 2]) {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let x: Vec<i64> = (0..T * I).map(|_| real_to_q32(rng.unit())).collect();
    let zt: Vec<i64> = (0..T * V).map(|_| real_to_q32(rng.unit())).collect();
    let wq0: Vec<i64> = (0..QW * I).map(|_| real_to_q32(rng.unit())).collect();
    let wk0: Vec<i64> = (0..KW * I).map(|_| real_to_q32(rng.unit())).collect();
    let wv0: Vec<i64> = (0..KW * I).map(|_| real_to_q32(rng.unit())).collect();
    let wo0: Vec<i64> = (0..I * QW).map(|_| real_to_q32(rng.unit())).collect();
    let wg0: Vec<i64> = (0..H * I).map(|_| real_to_q32(rng.unit())).collect();
    let wu0: Vec<i64> = (0..H * I).map(|_| real_to_q32(rng.unit())).collect();
    let wd0: Vec<i64> = (0..I * H).map(|_| real_to_q32(rng.unit())).collect();
    let g1_0: Vec<i64> = (0..I).map(|_| real_to_q32(rng.unit())).collect();
    let g2_0: Vec<i64> = (0..I).map(|_| real_to_q32(rng.unit())).collect();

    let table = rope_table(HD);
    let (cosv, sinv) = trig_tables(&table, T, HD);

    let mut oracle = Oracle {
        wq: wq0.clone(),
        wk: wk0.clone(),
        wv: wv0.clone(),
        wo: wo0.clone(),
        wg: wg0.clone(),
        wu: wu0.clone(),
        wd: wd0.clone(),
        g1: g1_0.clone(),
        g2: g2_0.clone(),
        a_wq: Adam::zeros(QW * I),
        a_wk: Adam::zeros(KW * I),
        a_wv: Adam::zeros(KW * I),
        a_wo: Adam::zeros(I * QW),
        a_wg: Adam::zeros(H * I),
        a_wu: Adam::zeros(H * I),
        a_wd: Adam::zeros(I * H),
        a_g1: Adam::zeros(I),
        a_g2: Adam::zeros(I),
    };

    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let up = |words: &[i64]| -> u64 {
        let real: Vec<f32> = words.iter().map(|&v| to_f32(v)).collect();
        unsafe { g.up_f32(&real) }
    };
    let x_buf = up(&x);
    let zt_buf = up(&zt);
    let cos_buf = unsafe { g.up_f32(&cosv) };
    let sin_buf = unsafe { g.up_f32(&sinv) };
    let wq_buf = up(&wq0);
    let wk_buf = up(&wk0);
    let wv_buf = up(&wv0);
    let wo_buf = up(&wo0);
    let wg_buf = up(&wg0);
    let wu_buf = up(&wu0);
    let wd_buf = up(&wd0);
    let g1_buf = up(&g1_0);
    let g2_buf = up(&g2_0);
    // Adam moments: (m, v) per tensor, device-resident and carried across both steps.
    let mut moments: Vec<(u64, u64)> = Vec::with_capacity(9);
    for n in [QW * I, KW * I, KW * I, I * QW, H * I, H * I, I * H, I, I] {
        let zeros = vec![0.0f32; n];
        let m = unsafe { g.up_f32(&zeros) };
        let v = unsafe { g.up_f32(&zeros) };
        moments.push((m, v));
    }

    // Activations and gradients, `n × 4` bytes each.
    let xn_buf = unsafe { g.dev(T * I * 4) };
    let q_buf = unsafe { g.dev(T * QW * 4) };
    let k_buf = unsafe { g.dev(T * KW * 4) };
    let val_buf = unsafe { g.dev(T * KW * 4) };
    let q0_buf = unsafe { g.dev(T * QW * 4) };
    let k0_buf = unsafe { g.dev(T * KW * 4) };
    let p_buf = unsafe { g.dev(NH * T * T * 4) };
    let oa_buf = unsafe { g.dev(T * QW * 4) };
    let o_buf = unsafe { g.dev(T * I * 4) };
    let x1_buf = unsafe { g.dev(T * I * 4) };
    let x2_buf = unsafe { g.dev(T * I * 4) };
    let xn2_buf = unsafe { g.dev(T * I * 4) };
    let gate_buf = unsafe { g.dev(T * H * 4) };
    let up_buf = unsafe { g.dev(T * H * 4) };
    let ha_buf = unsafe { g.dev(T * H * 4) };
    let f_buf = unsafe { g.dev(T * I * 4) };
    let loss_buf = unsafe { g.dev(T * 4) };
    let dz_buf = unsafe { g.dev(T * I * 4) };
    let dwd_buf = unsafe { g.dev(I * H * 4) };
    let dha_buf = unsafe { g.dev(T * H * 4) };
    let dgate_buf = unsafe { g.dev(T * H * 4) };
    let dup_buf = unsafe { g.dev(T * H * 4) };
    let dxn2_buf = unsafe { g.dev(T * I * 4) };
    let dwg_buf = unsafe { g.dev(H * I * 4) };
    let dwu_buf = unsafe { g.dev(H * I * 4) };
    let dx1_buf = unsafe { g.dev(T * I * 4) };
    let dg2_rows_buf = unsafe { g.dev(T * I * 4) };
    let dg2_buf = unsafe { g.dev(I * 4) };
    let dwo_buf = unsafe { g.dev(I * QW * 4) };
    let d_out_buf = unsafe { g.dev(T * QW * 4) };
    let rs_buf = unsafe { g.dev(NH * T * 4) };
    let ds_buf = unsafe { g.dev(NH * T * T * 4) };
    let dq_rope_buf = unsafe { g.dev(T * QW * 4) };
    let dk_rope_buf = unsafe { g.dev(T * KW * 4) };
    let dq_buf = unsafe { g.dev(T * QW * 4) };
    let dk_buf = unsafe { g.dev(T * KW * 4) };
    let dv_buf = unsafe { g.dev(T * KW * 4) };
    let dwq_buf = unsafe { g.dev(QW * I * 4) };
    let dwk_buf = unsafe { g.dev(KW * I * 4) };
    let dwv_buf = unsafe { g.dev(KW * I * 4) };
    let dxn_buf = unsafe { g.dev(T * I * 4) };
    let dx_buf = unsafe { g.dev(T * I * 4) };
    let dg1_rows_buf = unsafe { g.dev(T * I * 4) };
    let dg1_buf = unsafe { g.dev(I * 4) };
    let tmp_a_buf = unsafe { g.dev(T * I * 4) };
    let tmp_b_buf = unsafe { g.dev(T * I * 4) };
    let tmp_c_buf = unsafe { g.dev(T * I * 4) };

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
    let (rb_dx, rb_dx_p) = host(T * I);
    let (rb_dg1, rb_dg1_p) = host(I);
    let (rb_dg2, rb_dg2_p) = host(I);
    let (rb_wq, rb_wq_p) = host(QW * I);
    let (rb_wk, rb_wk_p) = host(KW * I);
    let (rb_wv, rb_wv_p) = host(KW * I);
    let (rb_wo, rb_wo_p) = host(I * QW);
    let (rb_wg, rb_wg_p) = host(H * I);
    let (rb_wu, rb_wu_p) = host(H * I);
    let (rb_wd, rb_wd_p) = host(I * H);

    let tu = T as u32;
    let (iu, hu, vu) = (I as u32, H as u32, V as u32);
    let (qw, kw, nhu, nkvu, hdu) = (QW as u32, KW as u32, NH as u32, NKV as u32, HD as u32);

    for (step, pin) in (1u32..=2).zip(pins) {
        let cpu = oracle.step(&x, &zt, &table);

        let add_step = |dst: u64, src: u64, n: u32| unsafe {
            make_step_named(
                &g,
                &add_inplace_bytes(n, ADD_LOCAL),
                &[dst, src],
                n.div_ceil(ADD_LOCAL),
                1,
                "add_inplace",
            )
        };

        let s_xn = unsafe { rmsnorm_fwd_dispatch(&g, x_buf, g1_buf, xn_buf, tu, iu) };
        let s_q = unsafe { linear_fwd_dispatch(&g, xn_buf, wq_buf, q0_buf, tu, iu, qw) };
        let s_k = unsafe { linear_fwd_dispatch(&g, xn_buf, wk_buf, k0_buf, tu, iu, kw) };
        let s_v = unsafe { linear_fwd_dispatch(&g, xn_buf, wv_buf, val_buf, tu, iu, kw) };
        let s_rq = unsafe { rope_fwd_dispatch(&g, q0_buf, cos_buf, sin_buf, q_buf, tu, nhu, hdu) };
        let s_rk = unsafe { rope_fwd_dispatch(&g, k0_buf, cos_buf, sin_buf, k_buf, tu, nkvu, hdu) };
        let s_attn = unsafe {
            attn_fwd_dispatch(&g, q_buf, k_buf, val_buf, p_buf, oa_buf, tu, nhu, nkvu, hdu)
        };
        let s_o = unsafe { linear_fwd_dispatch(&g, oa_buf, wo_buf, o_buf, tu, qw, iu) };
        let s_x1 = add_step(x1_buf, o_buf, (T * I) as u32);
        let s_x2a = add_step(x2_buf, o_buf, (T * I) as u32);
        let s_xn2 = unsafe { rmsnorm_fwd_dispatch(&g, x1_buf, g2_buf, xn2_buf, tu, iu) };
        let s_gate = unsafe { linear_fwd_dispatch(&g, xn2_buf, wg_buf, gate_buf, tu, iu, hu) };
        let s_up = unsafe { linear_fwd_dispatch(&g, xn2_buf, wu_buf, up_buf, tu, iu, hu) };
        let s_sw = unsafe { swiglu_fwd_dispatch(&g, gate_buf, up_buf, ha_buf, (T * H) as u32) };
        let s_f = unsafe { linear_fwd_dispatch(&g, ha_buf, wd_buf, f_buf, tu, hu, iu) };
        let s_x2b = add_step(x2_buf, f_buf, (T * I) as u32);
        let s_kl = unsafe { kl_row_dispatch(&g, x2_buf, zt_buf, loss_buf, dz_buf, tu, vu) };

        let s_dwd = unsafe { linear_bwd_weight_dispatch(&g, dz_buf, ha_buf, dwd_buf, tu, hu, iu) };
        let s_dha = unsafe { linear_bwd_input_dispatch(&g, dz_buf, wd_buf, dha_buf, tu, hu, iu) };
        let s_swb = unsafe {
            swiglu_bwd_dispatch(
                &g,
                gate_buf,
                up_buf,
                dha_buf,
                dgate_buf,
                dup_buf,
                (T * H) as u32,
            )
        };
        let s_dxn2a =
            unsafe { linear_bwd_input_dispatch(&g, dgate_buf, wg_buf, dxn2_buf, tu, iu, hu) };
        let s_dxn2b =
            unsafe { linear_bwd_input_dispatch(&g, dup_buf, wu_buf, tmp_a_buf, tu, iu, hu) };
        let s_dxn2 = add_step(dxn2_buf, tmp_a_buf, (T * I) as u32);
        let s_dwg =
            unsafe { linear_bwd_weight_dispatch(&g, dgate_buf, xn2_buf, dwg_buf, tu, iu, hu) };
        let s_dwu =
            unsafe { linear_bwd_weight_dispatch(&g, dup_buf, xn2_buf, dwu_buf, tu, iu, hu) };
        let s_nb2 = unsafe {
            rmsnorm_bwd_dispatch(
                &g,
                x1_buf,
                g2_buf,
                dxn2_buf,
                dx1_buf,
                dg2_rows_buf,
                tu,
                iu,
                1e-6f32,
            )
        };
        let s_dg2 = unsafe { rmsnorm_dgamma_dispatch(&g, dg2_rows_buf, dg2_buf, tu, iu) };
        let s_dx1 = add_step(dx1_buf, dz_buf, (T * I) as u32);
        let s_dwo = unsafe { linear_bwd_weight_dispatch(&g, dx1_buf, oa_buf, dwo_buf, tu, qw, iu) };
        let s_do = unsafe { linear_bwd_input_dispatch(&g, dx1_buf, wo_buf, d_out_buf, tu, qw, iu) };
        let s_rs = unsafe {
            attn_bwd_rowsum_dispatch(&g, val_buf, p_buf, d_out_buf, rs_buf, tu, nhu, nkvu, hdu)
        };
        let s_ds = unsafe {
            attn_bwd_ds_dispatch(
                &g, val_buf, p_buf, d_out_buf, rs_buf, ds_buf, tu, nhu, nkvu, hdu,
            )
        };
        let s_dq =
            unsafe { attn_bwd_dq_dispatch(&g, k_buf, ds_buf, dq_rope_buf, tu, nhu, nkvu, hdu) };
        let s_dkdv = unsafe {
            attn_bwd_dkdv_dispatch(
                &g,
                q_buf,
                p_buf,
                ds_buf,
                d_out_buf,
                dk_rope_buf,
                dv_buf,
                tu,
                nhu,
                nkvu,
                hdu,
            )
        };
        let s_rbq =
            unsafe { rope_bwd_dispatch(&g, dq_rope_buf, cos_buf, sin_buf, dq_buf, tu, nhu, hdu) };
        let s_rbk =
            unsafe { rope_bwd_dispatch(&g, dk_rope_buf, cos_buf, sin_buf, dk_buf, tu, nkvu, hdu) };
        let s_dwq = unsafe { linear_bwd_weight_dispatch(&g, dq_buf, xn_buf, dwq_buf, tu, iu, qw) };
        let s_dwk = unsafe { linear_bwd_weight_dispatch(&g, dk_buf, xn_buf, dwk_buf, tu, iu, kw) };
        let s_dwv = unsafe { linear_bwd_weight_dispatch(&g, dv_buf, xn_buf, dwv_buf, tu, iu, kw) };
        let s_dxnq = unsafe { linear_bwd_input_dispatch(&g, dq_buf, wq_buf, dxn_buf, tu, iu, qw) };
        let s_dxnk =
            unsafe { linear_bwd_input_dispatch(&g, dk_buf, wk_buf, tmp_b_buf, tu, iu, kw) };
        let s_dxnv =
            unsafe { linear_bwd_input_dispatch(&g, dv_buf, wv_buf, tmp_c_buf, tu, iu, kw) };
        let s_dxn1 = add_step(dxn_buf, tmp_b_buf, (T * I) as u32);
        let s_dxn2b2 = add_step(dxn_buf, tmp_c_buf, (T * I) as u32);
        let s_nb1 = unsafe {
            rmsnorm_bwd_dispatch(
                &g,
                x_buf,
                g1_buf,
                dxn_buf,
                dx_buf,
                dg1_rows_buf,
                tu,
                iu,
                1e-6f32,
            )
        };
        let s_dg1 = unsafe { rmsnorm_dgamma_dispatch(&g, dg1_rows_buf, dg1_buf, tu, iu) };

        let weights = [
            (wq_buf, QW * I),
            (wk_buf, KW * I),
            (wv_buf, KW * I),
            (wo_buf, I * QW),
            (wg_buf, H * I),
            (wu_buf, H * I),
            (wd_buf, I * H),
            (g1_buf, I),
            (g2_buf, I),
        ];
        let grads = [
            dwq_buf, dwk_buf, dwv_buf, dwo_buf, dwg_buf, dwu_buf, dwd_buf, dg1_buf, dg2_buf,
        ];
        let adam: Vec<_> = weights
            .iter()
            .zip(grads)
            .zip(moments.iter())
            .map(|((&(w, n), d), &(m, v))| unsafe {
                adam_update_dispatch(&g, w, d, m, v, n as u32, LR_Q32, step)
            })
            .collect();

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
            // Seed both residual buffers with `x` (the tape's `x1 = x + o`, `x2 = x + o + f`; the
            // second `o` add rounds identically, so `x2 = x1 + f` entry-wise).
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
            copy(x_buf, x1_buf, T * I * 4);
            copy(x_buf, x2_buf, T * I * 4);
            vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_COMPUTE,
                ACCESS_TRANSFER_WRITE,
                ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
            );
            rec_step(vk, cb, &s_xn);
            cbar(cb);
            rec_step(vk, cb, &s_q);
            cbar(cb);
            rec_step(vk, cb, &s_k);
            cbar(cb);
            rec_step(vk, cb, &s_v);
            cbar(cb);
            rec_step(vk, cb, &s_rq);
            cbar(cb);
            rec_step(vk, cb, &s_rk);
            cbar(cb);
            rec_step(vk, cb, &s_attn);
            cbar(cb);
            rec_step(vk, cb, &s_o);
            cbar(cb);
            rec_step(vk, cb, &s_x1);
            cbar(cb);
            rec_step(vk, cb, &s_x2a);
            cbar(cb);
            rec_step(vk, cb, &s_xn2);
            cbar(cb);
            rec_step(vk, cb, &s_gate);
            cbar(cb);
            rec_step(vk, cb, &s_up);
            cbar(cb);
            rec_step(vk, cb, &s_sw);
            cbar(cb);
            rec_step(vk, cb, &s_f);
            cbar(cb);
            rec_step(vk, cb, &s_x2b);
            cbar(cb);
            rec_step(vk, cb, &s_kl);
            cbar(cb);
            rec_step(vk, cb, &s_dwd);
            cbar(cb);
            rec_step(vk, cb, &s_dha);
            cbar(cb);
            rec_step(vk, cb, &s_swb);
            cbar(cb);
            rec_step(vk, cb, &s_dxn2a);
            cbar(cb);
            rec_step(vk, cb, &s_dxn2b);
            cbar(cb);
            rec_step(vk, cb, &s_dxn2);
            cbar(cb);
            rec_step(vk, cb, &s_dwg);
            cbar(cb);
            rec_step(vk, cb, &s_dwu);
            cbar(cb);
            rec_step(vk, cb, &s_nb2);
            cbar(cb);
            rec_step(vk, cb, &s_dg2);
            cbar(cb);
            rec_step(vk, cb, &s_dx1);
            cbar(cb);
            rec_step(vk, cb, &s_dwo);
            cbar(cb);
            rec_step(vk, cb, &s_do);
            cbar(cb);
            rec_step(vk, cb, &s_rs);
            cbar(cb);
            rec_step(vk, cb, &s_ds);
            cbar(cb);
            rec_step(vk, cb, &s_dq);
            cbar(cb);
            rec_step(vk, cb, &s_dkdv);
            cbar(cb);
            rec_step(vk, cb, &s_rbq);
            cbar(cb);
            rec_step(vk, cb, &s_rbk);
            cbar(cb);
            rec_step(vk, cb, &s_dwq);
            cbar(cb);
            rec_step(vk, cb, &s_dwk);
            cbar(cb);
            rec_step(vk, cb, &s_dwv);
            cbar(cb);
            rec_step(vk, cb, &s_dxnq);
            cbar(cb);
            rec_step(vk, cb, &s_dxnk);
            cbar(cb);
            rec_step(vk, cb, &s_dxnv);
            cbar(cb);
            rec_step(vk, cb, &s_dxn1);
            cbar(cb);
            rec_step(vk, cb, &s_dxn2b2);
            cbar(cb);
            rec_step(vk, cb, &s_nb1);
            cbar(cb);
            rec_step(vk, cb, &s_dg1);
            for (i, s) in adam.iter().enumerate() {
                if i > 0 {
                    cbar(cb);
                }
                rec_step(vk, cb, s);
            }
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_TRANSFER,
                ACCESS_SHADER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            copy(loss_buf, rb_loss, T * 4);
            copy(dx_buf, rb_dx, T * I * 4);
            copy(dg1_buf, rb_dg1, I * 4);
            copy(dg2_buf, rb_dg2, I * 4);
            copy(wq_buf, rb_wq, QW * I * 4);
            copy(wk_buf, rb_wk, KW * I * 4);
            copy(wv_buf, rb_wv, KW * I * 4);
            copy(wo_buf, rb_wo, I * QW * 4);
            copy(wg_buf, rb_wg, H * I * 4);
            copy(wu_buf, rb_wu, H * I * 4);
            copy(wd_buf, rb_wd, I * H * 4);
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

        let rd = |ptr: *mut f32, n: usize| -> Vec<f32> {
            unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
        };
        let gpu_loss = rd(rb_loss_p, T);
        let gpu_dx = rd(rb_dx_p, T * I);
        let gpu_dg1 = rd(rb_dg1_p, I);
        let gpu_dg2 = rd(rb_dg2_p, I);
        let gpu_wq = rd(rb_wq_p, QW * I);
        let gpu_wk = rd(rb_wk_p, KW * I);
        let gpu_wv = rd(rb_wv_p, KW * I);
        let gpu_wo = rd(rb_wo_p, I * QW);
        let gpu_wg = rd(rb_wg_p, H * I);
        let gpu_wu = rd(rb_wu_p, H * I);
        let gpu_wd = rd(rb_wd_p, I * H);
        for (name, v) in [
            ("loss", &gpu_loss),
            ("dx", &gpu_dx),
            ("dg1", &gpu_dg1),
            ("dg2", &gpu_dg2),
            ("Wq", &gpu_wq),
            ("Wk", &gpu_wk),
            ("Wv", &gpu_wv),
            ("Wo", &gpu_wo),
            ("Wg", &gpu_wg),
            ("Wu", &gpu_wu),
            ("Wd", &gpu_wd),
        ] {
            assert!(v.iter().all(|x| x.is_finite()), "non-finite {name}");
        }

        check_row(step, "loss", &gpu_loss, &cpu.loss, 1, pin.loss);
        check_row(step, "dx", &gpu_dx, &cpu.dx, I, pin.dx);
        check_row(step, "dg1", &gpu_dg1, &cpu.dg1, I, pin.dg1);
        check_row(step, "dg2", &gpu_dg2, &cpu.dg2, I, pin.dg2);
        check_row(step, "Wq", &gpu_wq, &oracle.wq, I, pin.wq);
        check_row(step, "Wk", &gpu_wk, &oracle.wk, I, pin.wk);
        check_row(step, "Wv", &gpu_wv, &oracle.wv, I, pin.wv);
        check_row(step, "Wo", &gpu_wo, &oracle.wo, QW, pin.wo);
        check_row(step, "Wg", &gpu_wg, &oracle.wg, I, pin.wg);
        check_row(step, "Wu", &gpu_wu, &oracle.wu, I, pin.wu);
        check_row(step, "Wd", &gpu_wd, &oracle.wd, H, pin.wd);
    }
}

/// One row of the table: the per-entry maximum in Q32.32 units, its entry index, the pin, and a
/// pass/fail note on the pin.
fn check_row(step: u32, label: &str, gpu: &[f32], cpu: &[i64], cols: usize, pin: f64) {
    let (m, at) = max_err(gpu, cpu, cols);
    let note = if m as f64 > pin {
        "EXCEEDS PIN"
    } else if (m as f64) * 2.0 < pin * 0.9 {
        "loose pin (measured maximum is well below half the pin)"
    } else {
        "ok"
    };
    println!("step {step}: {label} max |gpu-cpu| = {m} units at {at:?}; pin {pin}; {note}");
    assert!(
        m as f64 <= pin,
        "step {step}: {label} error {m} units at {at:?} exceeds the pin {pin}"
    );
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
