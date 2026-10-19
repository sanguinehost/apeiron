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
use apeiron::gpu::decode::GpuCtx;
use apeiron::gpu::nn::train::{rope_tables, HostWeights, LayerCfg, Trainer};
use apeiron::gpu::vulkan::Vk;
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
    let seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rng = Lcg(seed);
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

    // The production trainer owns the buffer plan and the step construction; this test keeps the
    // seed, the CPU chain and the pins. The weight tensors and RoPE tables are asserted equal to
    // the oracle's words, so the moved composition is the same computation.
    let cfg = LayerCfg::new(
        T as u32, I as u32, H as u32, V as u32, NH as u32, NKV as u32, HD as u32,
    )
    .expect("layer cfg");
    let hw = HostWeights::lcg(&cfg, seed);
    let f32s = |v: &[i64]| -> Vec<f32> { v.iter().map(|&w| to_f32(w)).collect() };
    assert_eq!(hw.wq, f32s(&wq0), "weight stream Wq");
    assert_eq!(hw.wk, f32s(&wk0), "weight stream Wk");
    assert_eq!(hw.wv, f32s(&wv0), "weight stream Wv");
    assert_eq!(hw.wo, f32s(&wo0), "weight stream Wo");
    assert_eq!(hw.wg, f32s(&wg0), "weight stream Wg");
    assert_eq!(hw.wu, f32s(&wu0), "weight stream Wu");
    assert_eq!(hw.wd, f32s(&wd0), "weight stream Wd");
    assert_eq!(hw.g1, f32s(&g1_0), "weight stream g1");
    assert_eq!(hw.g2, f32s(&g2_0), "weight stream g2");
    let (cos_t, sin_t) = rope_tables(&cfg, 10_000.0);
    assert_eq!(cos_t, cosv, "RoPE cos table");
    assert_eq!(sin_t, sinv, "RoPE sin table");

    let (g, _stage) = GpuCtx::new(vk).expect("GpuCtx::new");
    let mut trainer = Trainer::new(&g, cfg, &hw);
    let xf: Vec<f32> = x.iter().map(|&w| to_f32(w)).collect();
    let ztf: Vec<f32> = zt.iter().map(|&w| to_f32(w)).collect();

    for (step, pin) in (1u32..=2).zip(pins) {
        let cpu = oracle.step(&x, &zt, &table);

        trainer.set_batch(&xf, &ztf);
        trainer.step(&g);
        trainer.sync_weights(&g);

        let gpu_loss = trainer.losses();
        let gpu_dx = trainer.dx();
        let gpu_dg1 = trainer.dgamma1();
        let gpu_dg2 = trainer.dgamma2();
        let updated = trainer.weights();
        let gpu_wq = updated.wq;
        let gpu_wk = updated.wk;
        let gpu_wv = updated.wv;
        let gpu_wo = updated.wo;
        let gpu_wg = updated.wg;
        let gpu_wu = updated.wu;
        let gpu_wd = updated.wd;
        assert_eq!(
            trainer.loss().to_bits(),
            gpu_loss[0].to_bits(),
            "loss() is row 0"
        );
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
