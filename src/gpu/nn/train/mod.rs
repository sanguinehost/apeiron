//! One pre-norm transformer layer's training step as host composition of the SPIR-V dispatches.
//!
//! This is the production home of the composition the layer-step test used to build inline. The
//! order is the layer's: `xn = rmsnorm(x, g1)`; `q = xn·Wqᵀ`, `k = xn·Wkᵀ`, `v = xn·Wvᵀ`;
//! `q = rope(q)`, `k = rope(k)`; `(p, oa) = attn(q,k,v)`; `o = oa·Woᵀ`; `x1 = x + o`;
//! `xn2 = rmsnorm(x1, g2)`; `g = xn2·Wgᵀ`, `u = xn2·Wuᵀ`; `ha = silu(g)⊙u`; `f = ha·Wdᵀ`;
//! `x2 = x1 + f`; `(loss, dz) = kl_row(x2, target)`; `dWd = dzᵀ·ha`, `dha = dz·Wd`,
//! `(dg,du) = swiglu_bwd`, `dxn2 = dg·Wgᵀ + du·Wuᵀ`, `dWg = dgᵀ·xn2`, `dWu = duᵀ·xn2`,
//! `dx1 = dz + rmsnorm_bwd_dx(x1,g2,dxn2)`, `dg2 = dgamma_reduce`, `dWo = dx1ᵀ·oa`,
//! `do = dx1·Wo`, the five attention-backward passes on `(do,q,k,v,p)`, `dq = rope_bwd(dq)`,
//! `dk = rope_bwd(dk)`, `dWq = dqᵀ·xn`, `dWk = dkᵀ·xn`, `dWv = dvᵀ·xn`,
//! `dxn = dq·Wqᵀ + dk·Wkᵀ + dv·Wvᵀ`, `dx = rmsnorm_bwd_dx(x,g1,dxn)`, `dg1 = dgamma_reduce`, then
//! `adam_update_dispatch` on the nine tensors `Wq, Wk, Wv, Wo, Wg, Wu, Wd, g1, g2`.
//!
//! Shapes are the caller's [`LayerCfg`]; hidden width equals vocabulary width, so `kl_row` reads
//! `x2` directly as the row logits. Positions are the row index. The residual identity means the
//! KL `dz` reaches the direct `dx1` **and** the FFN branch through `dha`; it is added once at each
//! place the composition names and nowhere else.
//!
//! `Trainer::step` records ONE command buffer per step (staging copies, forward, KL, backward,
//! Adam, the gradient readbacks), so a step is one submission; `Trainer::sync_weights` records the
//! nine weight readbacks on demand. Weights, moments and every activation are device-resident and
//! carried across steps.
//!
//! This module is host composition of SPIR-V dispatches, outside the extraction domain: the
//! builders and dispatches it calls are the waived `gpu`/`ffi` surface (`Gpu.Handoff`).

use crate::engine::num::fp::F64;
use crate::engine::RopeTable;
use crate::gpu::decode::{make_step_named, step_barrier, GpuCtx, Step};
use crate::gpu::nn::act::{swiglu_bwd_dispatch, swiglu_fwd_dispatch};
use crate::gpu::nn::fwd::{
    kl_row_dispatch, linear_fwd_dispatch, rmsnorm_fwd_dispatch, rope_bwd_dispatch,
    rope_fwd_dispatch,
};
use crate::gpu::nn::{
    adam_bias_corrections, adam_update_dispatch_reused, add_inplace_bytes, attn_bwd_dkdv_dispatch,
    attn_bwd_dq_dispatch, attn_bwd_ds_dispatch, attn_bwd_rowsum_dispatch,
    attn_bwd_rowsum_reduce_dispatch, attn_fwd_dispatch, linear_bwd_input_dispatch,
    linear_bwd_weight_dispatch, rmsnorm_bwd_dispatch, rmsnorm_dgamma_dispatch,
};
use crate::gpu::vulkan::*;
use crate::numeric::fixed_trig::{fixed_sin_cos, turn_to_radians};

pub mod attn_bwd;
pub mod block;
#[cfg(test)]
mod block_mask;
pub mod front;
pub mod mixer;
pub mod slot;

/// A layer shape. Hidden width `i`, FFN width `h`, vocabulary width `v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerCfg {
    pub t: u32,
    pub i: u32,
    pub h: u32,
    pub v: u32,
    pub nh: u32,
    pub nkv: u32,
    pub hd: u32,
}

impl LayerCfg {
    /// Validate and build a shape: every dimension non-zero, `nh % nkv == 0`, `hd` even and ≥ 2,
    /// so the RoPE pair `(j, j + hd/2)` and the GQA head partition are defined.
    ///
    /// # Errors
    ///
    /// Returns which constraint failed.
    pub fn new(t: u32, i: u32, h: u32, v: u32, nh: u32, nkv: u32, hd: u32) -> Result<Self, String> {
        if t == 0 || i == 0 || h == 0 || v == 0 || nh == 0 || nkv == 0 || hd == 0 {
            return Err("layer shape: every dimension must be non-zero".to_string());
        }
        if nh % nkv != 0 {
            return Err(format!(
                "layer shape: nh {nh} is not a multiple of nkv {nkv}"
            ));
        }
        if hd < 2 || hd % 2 != 0 {
            return Err(format!(
                "layer shape: head_dim {hd} must be even and at least 2"
            ));
        }
        Ok(Self {
            t,
            i,
            h,
            v,
            nh,
            nkv,
            hd,
        })
    }

    /// Query width `nh·hd`.
    fn qw(&self) -> u32 {
        self.nh * self.hd
    }

    /// Key/value width `nkv·hd`.
    fn kw(&self) -> u32 {
        self.nkv * self.hd
    }

    /// Rotary half-width `hd/2`.
    fn d2(&self) -> u32 {
        self.hd / 2
    }
}

/// Fixed-seed LCG (Knuth's MMIX constants), uniform in `[-0.5, 0.5)`, in Q32.32 words.
///
/// Complexity: O(1) time and space per draw.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u - 0.5
    }
}

/// Real `v` to Q32.32, round to nearest.
fn real_to_q32(v: f64) -> i64 {
    (v * 4_294_967_296.0).round() as i64
}

/// Q32.32 word to f32 real units (the device's upload image).
fn to_f32(v: i64) -> f32 {
    (v as f64 / 4_294_967_296.0) as f32
}

/// The layer's host weights, in the composition's order, as the device's f32 real units.
#[derive(Clone, Debug, PartialEq)]
pub struct HostWeights {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub wg: Vec<f32>,
    pub wu: Vec<f32>,
    pub wd: Vec<f32>,
    pub g1: Vec<f32>,
    pub g2: Vec<f32>,
}

impl HostWeights {
    /// The fixed-seed generator and the layout the layer test pins: `wq (qw·i)`, `wk`/`wv`
    /// (`kw·i`), `wo` (`i·qw`), `wg`/`wu` (`h·i`), `wd` (`i·h`), `g1`/`g2` (`i`), each entry
    /// `f32(round(2³²·u))` for the LCG's uniform `u`.
    ///
    /// The stream position is the layer test's: `t·i` batch inputs and `t·v` targets are drawn
    /// first (by the caller, from the same `seed`), then the nine tensors in the order above. The
    /// skip keeps the uploaded weights bit-identical to the published layer-step test.
    ///
    /// Complexity: O(t·i + t·v + entries) time, O(entries) space.
    pub fn lcg(cfg: &LayerCfg, seed: u64) -> Self {
        let (i, h) = (cfg.i as usize, cfg.h as usize);
        let (qw, kw) = (cfg.qw() as usize, cfg.kw() as usize);
        let mut rng = Lcg(seed);
        let skip = cfg.t as usize * i + cfg.t as usize * cfg.v as usize;
        for _ in 0..skip {
            let _ = rng.unit();
        }
        let mut draw =
            |n: usize| -> Vec<f32> { (0..n).map(|_| to_f32(real_to_q32(rng.unit()))).collect() };
        Self {
            wq: draw(qw * i),
            wk: draw(kw * i),
            wv: draw(kw * i),
            wo: draw(i * qw),
            wg: draw(h * i),
            wu: draw(h * i),
            wd: draw(i * h),
            g1: draw(i),
            g2: draw(i),
        }
    }
}

/// The `(t·hd/2)` `cos`/`sin` f32 tables the RoPE kernels take as host inputs, laid out row-major
/// over `(pos, j)`.
///
/// The phase is the engine's exact integer round `(turns[j]·pos) mod 2^64` mapped through
/// `turn_to_radians` and `fixed_sin_cos`, then converted to f32 once. Device f32 trigonometry
/// cannot reproduce that round across positions, so the table is built here.
///
/// # Panics
///
/// Panics when `theta` is not finite and positive, or a turn reaches `2^64`.
///
/// Complexity: O(t·hd) time and space in the table entries.
pub fn rope_tables(cfg: &LayerCfg, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let table = RopeTable::new(F64::from_f64(theta), cfg.hd as usize).expect("rope turn table");
    let (t, d2) = (cfg.t as usize, cfg.d2() as usize);
    let mut cos = vec![0f32; t * d2];
    let mut sin = vec![0f32; t * d2];
    for pos in 0..t {
        for j in 0..d2 {
            let phase = ((u128::from(table.turns()[j]) * u128::from(pos as u32))
                & u128::from(u64::MAX)) as u64;
            let sc = fixed_sin_cos(turn_to_radians(phase)).expect("fixed_sin_cos");
            let off = pos * d2 + j;
            cos[off] = (sc.cos as f64 / 4_294_967_296.0) as f32;
            sin[off] = (sc.sin as f64 / 4_294_967_296.0) as f32;
        }
    }
    (cos, sin)
}

/// A host-visible, host-coherent f32 buffer and its mapped pointer.
struct HostBuffer {
    buf: Handle,
    ptr: *mut f32,
}

/// Create and map a host-visible f32 buffer of `n` words with `usage`.
///
/// # Safety
///
/// Vulkan handle creation; the buffer is owned by the returned value.
unsafe fn host_buffer(g: &GpuCtx, n: usize, usage: u32) -> HostBuffer {
    let (buf, mem) =
        g.vk.buffer(
            n.max(1) as u64 * 4,
            usage,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )
        .expect("host-visible buffer");
    HostBuffer {
        buf,
        ptr: g.vk.map(mem).cast::<f32>(),
    }
}

/// Local size of the residual-add kernel and the grid divisor for its dispatch.
const ADD_LOCAL: u32 = 256;
/// `round(1e-3·2^32)`: the Adam learning-rate word the dense step test uses.
const LR_Q32: i64 = 4_294_967;

/// Dispatches one step records: 47 layer dispatches + 9 Adam (`Perf.Train.layerStepDispatches_eq`
/// counts the tape; the barriers and copies issue none).
pub const STEP_DISPATCHES: usize = 56;

/// One traced step: its device timeline and its host split.
pub struct StepProfile {
    /// Host microseconds of the last step, `[bias write, pool, record, submit+idle]`.
    pub host_us: [f64; 4],
    /// Device microseconds from the global start query to the global end query.
    pub total_us: f64,
    /// Copies and seed barriers before the first dispatch.
    pub prologue_us: f64,
    /// One device-microsecond span per dispatch, in tape order.
    pub dispatch_us: Vec<f64>,
    /// One microsecond gap per inter-dispatch barrier, `dispatch_us.len() - 1` of them.
    pub gap_us: Vec<f64>,
    /// The readback copies and the HOST barrier after the last dispatch.
    pub tail_us: f64,
    /// Dispatch names, in tape order.
    pub labels: Vec<&'static str>,
}

impl StepProfile {
    /// Timestamp queries one traced step writes: a global start, two per dispatch, a global end.
    pub const QUERIES: u32 = 2 + 2 * STEP_DISPATCHES as u32;
}

/// The timestamp recorder of one traced step.
struct Trace<'a> {
    pool: &'a QueryPool,
    next: u32,
    labels: Vec<&'static str>,
}

impl Trace<'_> {
    /// Write the next timestamp query in tape order.
    fn stamp(&mut self, vk: &Vk, cb: CommandBuffer) {
        unsafe { vk.cmd_write_timestamp(cb, self.pool, STAGE_ALL_COMMANDS, self.next) };
        self.next += 1;
    }
}

/// One layer's device-resident training state and its dispatches, built once and replayed per
/// step.
pub struct Trainer {
    cfg: LayerCfg,
    x_stage: HostBuffer,
    zt_stage: HostBuffer,
    rb_loss: HostBuffer,
    rb_dx: HostBuffer,
    rb_dg1: HostBuffer,
    rb_dg2: HostBuffer,
    rb_w: [HostBuffer; 9],
    w: [Handle; 9],
    /// Adam dispatches, built once per weight tensor at construction. `step` writes that step's
    /// two bias-correction words into `bias_stage` and replays these at binding 4.
    adam: [Step; 9],
    /// The current step's `[β₁ᵗ, √(1−β₂ᵗ)]`, host-visible; copied to `bias_buf` by `step`.
    bias_stage: HostBuffer,
    bias_buf: Handle,
    /// Host microseconds of the last `step`, `[bias write, pool, record, submit+idle]`.
    host_us: [f64; 4],
    step_no: u32,
    x: Handle,
    zt: Handle,
    /// Residual accumulators, written by the step's two seeding copies.
    x1: Handle,
    x2: Handle,
    loss: Handle,
    dg2: Handle,
    dx: Handle,
    dg1: Handle,
    s_xn: Step,
    s_q: Step,
    s_k: Step,
    s_v: Step,
    s_rq: Step,
    s_rk: Step,
    s_attn: Step,
    s_o: Step,
    s_x1: Step,
    s_x2a: Step,
    s_xn2: Step,
    s_gate: Step,
    s_up: Step,
    s_sw: Step,
    s_f: Step,
    s_x2b: Step,
    s_kl: Step,
    s_dwd: Step,
    s_dha: Step,
    s_swb: Step,
    s_dxn2a: Step,
    s_dxn2b: Step,
    s_dxn2: Step,
    s_dwg: Step,
    s_dwu: Step,
    s_nb2: Step,
    s_dg2: Step,
    s_dx1: Step,
    s_dwo: Step,
    s_do: Step,
    s_rs: Step,
    s_rsr: Step,
    s_ds: Step,
    s_dq: Step,
    s_dkdv: Step,
    s_rbq: Step,
    s_rbk: Step,
    s_dwq: Step,
    s_dwk: Step,
    s_dwv: Step,
    s_dxnq: Step,
    s_dxnk: Step,
    s_dxnv: Step,
    s_dxn1: Step,
    s_dxn2b2: Step,
    s_nb1: Step,
    s_dg1: Step,
}

impl Trainer {
    /// Upload the weights and zeroed moments, build every activation buffer and every dispatch
    /// once, and upload the RoPE tables (`theta = 10000`).
    ///
    /// # Panics
    ///
    /// Panics when a `HostWeights` vector length does not match `cfg`.
    ///
    /// Complexity: O(weights + activations) time and space, 9 Adam pipelines + 56 dispatch objects.
    pub fn new(g: &GpuCtx, cfg: LayerCfg, weights: &HostWeights) -> Self {
        let (t, i, h, v) = (
            cfg.t as usize,
            cfg.i as usize,
            cfg.h as usize,
            cfg.v as usize,
        );
        let (qw, kw) = (cfg.qw() as usize, cfg.kw() as usize);
        assert_eq!(weights.wq.len(), qw * i, "Wq width");
        assert_eq!(weights.wk.len(), kw * i, "Wk width");
        assert_eq!(weights.wv.len(), kw * i, "Wv width");
        assert_eq!(weights.wo.len(), i * qw, "Wo width");
        assert_eq!(weights.wg.len(), h * i, "Wg width");
        assert_eq!(weights.wu.len(), h * i, "Wu width");
        assert_eq!(weights.wd.len(), i * h, "Wd width");
        assert_eq!(weights.g1.len(), i, "g1 width");
        assert_eq!(weights.g2.len(), i, "g2 width");

        let (cosv, sinv) = rope_tables(&cfg, 10_000.0);
        let w = [
            unsafe { g.up_f32(&weights.wq) },
            unsafe { g.up_f32(&weights.wk) },
            unsafe { g.up_f32(&weights.wv) },
            unsafe { g.up_f32(&weights.wo) },
            unsafe { g.up_f32(&weights.wg) },
            unsafe { g.up_f32(&weights.wu) },
            unsafe { g.up_f32(&weights.wd) },
            unsafe { g.up_f32(&weights.g1) },
            unsafe { g.up_f32(&weights.g2) },
        ];
        let mut moments: [(Handle, Handle); 9] = [(0, 0); 9];
        let widths = [qw * i, kw * i, kw * i, i * qw, h * i, h * i, i * h, i, i];
        for (n, slot) in widths.into_iter().zip(moments.iter_mut()) {
            let zeros = vec![0f32; n];
            *slot = unsafe { (g.up_f32(&zeros), g.up_f32(&zeros)) };
        }
        let cos = unsafe { g.up_f32(&cosv) };
        let sin = unsafe { g.up_f32(&sinv) };

        // Activations and gradients, `n × 4` bytes each.
        let dev = |bytes: usize| unsafe { g.dev(bytes.max(4)) };
        let x = dev(t * i * 4);
        let zt = dev(t * v * 4);
        let xn = dev(t * i * 4);
        let q = dev(t * qw * 4);
        let k = dev(t * kw * 4);
        let val = dev(t * kw * 4);
        let q0 = dev(t * qw * 4);
        let k0 = dev(t * kw * 4);
        let p = dev(cfg.nh as usize * t * t * 4);
        let oa = dev(t * qw * 4);
        let o = dev(t * i * 4);
        let x1 = dev(t * i * 4);
        let x2 = dev(t * i * 4);
        let xn2 = dev(t * i * 4);
        let gate = dev(t * h * 4);
        let up_act = dev(t * h * 4);
        let ha = dev(t * h * 4);
        let f = dev(t * i * 4);
        let loss = dev(t * 4);
        let dz = dev(t * i * 4);
        let dwd = dev(i * h * 4);
        let dha = dev(t * h * 4);
        let dgate = dev(t * h * 4);
        let dup = dev(t * h * 4);
        let dxn2 = dev(t * i * 4);
        let dwg = dev(h * i * 4);
        let dwu = dev(h * i * 4);
        let dx1 = dev(t * i * 4);
        let dg2_rows = dev(t * i * 4);
        let dg2 = dev(i * 4);
        let dwo = dev(i * qw * 4);
        let d_out = dev(t * qw * 4);
        let rs = dev(cfg.nh as usize * t * 4);
        let rs_parts = dev(cfg.nh as usize * t * t.div_ceil(16) * 4);
        let ds = dev(cfg.nh as usize * t * t * 4);
        let dq_rope = dev(t * qw * 4);
        let dk_rope = dev(t * kw * 4);
        let dq = dev(t * qw * 4);
        let dk = dev(t * kw * 4);
        let dv = dev(t * kw * 4);
        let dwq = dev(qw * i * 4);
        let dwk = dev(kw * i * 4);
        let dwv = dev(kw * i * 4);
        let dxn = dev(t * i * 4);
        let dx = dev(t * i * 4);
        let dg1_rows = dev(t * i * 4);
        let dg1 = dev(i * 4);
        let tmp_a = dev(t * i * 4);
        let tmp_b = dev(t * i * 4);
        let tmp_c = dev(t * i * 4);

        let x_stage = unsafe { host_buffer(g, t * i, BUFFER_USAGE_TRANSFER_SRC) };
        let zt_stage = unsafe { host_buffer(g, t * v, BUFFER_USAGE_TRANSFER_SRC) };
        let rb = |n: usize| unsafe { host_buffer(g, n, BUFFER_USAGE_TRANSFER_DST) };
        let rb_loss = rb(t);
        let rb_dx = rb(t * i);
        let rb_dg1 = rb(i);
        let rb_dg2 = rb(i);
        let rb_w = [
            rb(qw * i),
            rb(kw * i),
            rb(kw * i),
            rb(i * qw),
            rb(h * i),
            rb(h * i),
            rb(i * h),
            rb(i),
            rb(i),
        ];

        let (tu, iu, hu, vu) = (cfg.t, cfg.i, cfg.h, cfg.v);
        let (qwu, kwu, nhu, nkvu, hdu) = (cfg.qw(), cfg.kw(), cfg.nh, cfg.nkv, cfg.hd);
        let add = |dst: Handle, src: Handle, n: u32| unsafe {
            make_step_named(
                g,
                &add_inplace_bytes(n, ADD_LOCAL),
                &[dst, src],
                n.div_ceil(ADD_LOCAL),
                1,
                "add_inplace",
            )
        };

        let s_xn = unsafe { rmsnorm_fwd_dispatch(g, x, w[7], xn, tu, iu) };
        let s_q = unsafe { linear_fwd_dispatch(g, xn, w[0], q0, tu, iu, qwu) };
        let s_k = unsafe { linear_fwd_dispatch(g, xn, w[1], k0, tu, iu, kwu) };
        let s_v = unsafe { linear_fwd_dispatch(g, xn, w[2], val, tu, iu, kwu) };
        let s_rq = unsafe { rope_fwd_dispatch(g, q0, cos, sin, q, tu, nhu, hdu) };
        let s_rk = unsafe { rope_fwd_dispatch(g, k0, cos, sin, k, tu, nkvu, hdu) };
        let s_attn = unsafe { attn_fwd_dispatch(g, q, k, val, p, oa, tu, nhu, nkvu, hdu) };
        let s_o = unsafe { linear_fwd_dispatch(g, oa, w[3], o, tu, qwu, iu) };
        let s_x1 = add(x1, o, (t * i) as u32);
        let s_x2a = add(x2, o, (t * i) as u32);
        let s_xn2 = unsafe { rmsnorm_fwd_dispatch(g, x1, w[8], xn2, tu, iu) };
        let s_gate = unsafe { linear_fwd_dispatch(g, xn2, w[4], gate, tu, iu, hu) };
        let s_up = unsafe { linear_fwd_dispatch(g, xn2, w[5], up_act, tu, iu, hu) };
        let s_sw = unsafe { swiglu_fwd_dispatch(g, gate, up_act, ha, (t * h) as u32) };
        let s_f = unsafe { linear_fwd_dispatch(g, ha, w[6], f, tu, hu, iu) };
        let s_x2b = add(x2, f, (t * i) as u32);
        let s_kl = unsafe { kl_row_dispatch(g, x2, zt, loss, dz, tu, vu) };

        let s_dwd = unsafe { linear_bwd_weight_dispatch(g, dz, ha, dwd, tu, hu, iu) };
        let s_dha = unsafe { linear_bwd_input_dispatch(g, dz, w[6], dha, tu, hu, iu) };
        let s_swb =
            unsafe { swiglu_bwd_dispatch(g, gate, up_act, dha, dgate, dup, (t * h) as u32) };
        let s_dxn2a = unsafe { linear_bwd_input_dispatch(g, dgate, w[4], dxn2, tu, iu, hu) };
        let s_dxn2b = unsafe { linear_bwd_input_dispatch(g, dup, w[5], tmp_a, tu, iu, hu) };
        let s_dxn2 = add(dxn2, tmp_a, (t * i) as u32);
        let s_dwg = unsafe { linear_bwd_weight_dispatch(g, dgate, xn2, dwg, tu, iu, hu) };
        let s_dwu = unsafe { linear_bwd_weight_dispatch(g, dup, xn2, dwu, tu, iu, hu) };
        let s_nb2 =
            unsafe { rmsnorm_bwd_dispatch(g, x1, w[8], dxn2, dx1, dg2_rows, tu, iu, 1e-6f32) };
        let s_dg2 = unsafe { rmsnorm_dgamma_dispatch(g, dg2_rows, dg2, tu, iu) };
        let s_dx1 = add(dx1, dz, (t * i) as u32);
        let s_dwo = unsafe { linear_bwd_weight_dispatch(g, dx1, oa, dwo, tu, qwu, iu) };
        let s_do = unsafe { linear_bwd_input_dispatch(g, dx1, w[3], d_out, tu, qwu, iu) };
        let s_rs =
            unsafe { attn_bwd_rowsum_dispatch(g, val, p, d_out, rs_parts, tu, nhu, nkvu, hdu) };
        let s_rsr = unsafe { attn_bwd_rowsum_reduce_dispatch(g, rs_parts, rs, tu, nhu) };
        let s_ds = unsafe { attn_bwd_ds_dispatch(g, val, p, d_out, rs, ds, tu, nhu, nkvu, hdu) };
        let s_dq = unsafe { attn_bwd_dq_dispatch(g, k, ds, dq_rope, tu, nhu, nkvu, hdu) };
        let s_dkdv =
            unsafe { attn_bwd_dkdv_dispatch(g, q, p, ds, d_out, dk_rope, dv, tu, nhu, nkvu, hdu) };
        let s_rbq = unsafe { rope_bwd_dispatch(g, dq_rope, cos, sin, dq, tu, nhu, hdu) };
        let s_rbk = unsafe { rope_bwd_dispatch(g, dk_rope, cos, sin, dk, tu, nkvu, hdu) };
        let s_dwq = unsafe { linear_bwd_weight_dispatch(g, dq, xn, dwq, tu, iu, qwu) };
        let s_dwk = unsafe { linear_bwd_weight_dispatch(g, dk, xn, dwk, tu, iu, kwu) };
        let s_dwv = unsafe { linear_bwd_weight_dispatch(g, dv, xn, dwv, tu, iu, kwu) };
        let s_dxnq = unsafe { linear_bwd_input_dispatch(g, dq, w[0], dxn, tu, iu, qwu) };
        let s_dxnk = unsafe { linear_bwd_input_dispatch(g, dk, w[1], tmp_b, tu, iu, kwu) };
        let s_dxnv = unsafe { linear_bwd_input_dispatch(g, dv, w[2], tmp_c, tu, iu, kwu) };
        let s_dxn1 = add(dxn, tmp_b, (t * i) as u32);
        let s_dxn2b2 = add(dxn, tmp_c, (t * i) as u32);
        let s_nb1 = unsafe { rmsnorm_bwd_dispatch(g, x, w[7], dxn, dx, dg1_rows, tu, iu, 1e-6f32) };
        let s_dg1 = unsafe { rmsnorm_dgamma_dispatch(g, dg1_rows, dg1, tu, iu) };

        // Adam: one pipeline per weight tensor, reused at every step; the step's bias corrections
        // reach the module through `bias_buf` (binding 4), written by `step`.
        let bias_stage = unsafe { host_buffer(g, 2, BUFFER_USAGE_TRANSFER_SRC) };
        let bias_buf = dev(8);
        let grads = [dwq, dwk, dwv, dwo, dwg, dwu, dwd, dg1, dg2];
        let adam: [Step; 9] = std::array::from_fn(|n| {
            let (m, vv) = moments[n];
            unsafe {
                adam_update_dispatch_reused(
                    g,
                    w[n],
                    grads[n],
                    m,
                    vv,
                    bias_buf,
                    widths[n] as u32,
                    LR_Q32,
                )
            }
        });

        Self {
            cfg,
            x_stage,
            zt_stage,
            rb_loss,
            rb_dx,
            rb_dg1,
            rb_dg2,
            rb_w,
            w,
            adam,
            bias_stage,
            bias_buf,
            host_us: [0.0; 4],
            step_no: 0,
            x,
            zt,
            x1,
            x2,
            loss,
            dg2,
            dx,
            dg1,
            s_xn,
            s_q,
            s_k,
            s_v,
            s_rq,
            s_rk,
            s_attn,
            s_o,
            s_x1,
            s_x2a,
            s_xn2,
            s_gate,
            s_up,
            s_sw,
            s_f,
            s_x2b,
            s_kl,
            s_dwd,
            s_dha,
            s_swb,
            s_dxn2a,
            s_dxn2b,
            s_dxn2,
            s_dwg,
            s_dwu,
            s_nb2,
            s_dg2,
            s_dx1,
            s_dwo,
            s_do,
            s_rs,
            s_rsr,
            s_ds,
            s_dq,
            s_dkdv,
            s_rbq,
            s_rbk,
            s_dwq,
            s_dwk,
            s_dwv,
            s_dxnq,
            s_dxnk,
            s_dxnv,
            s_dxn1,
            s_dxn2b2,
            s_nb1,
            s_dg1,
        }
    }

    /// Write the batch input and its target into host-visible staging; `step` copies them to the
    /// device. `x` is `t·i` f32 and `target` is `t·v` f32.
    ///
    /// # Panics
    ///
    /// Panics when either slice does not match the configured shape.
    ///
    /// Complexity: O(t·(i + v)) time and space copied.
    pub fn set_batch(&mut self, x: &[f32], target: &[f32]) {
        let (ni, nv) = (
            self.cfg.t as usize * self.cfg.i as usize,
            self.cfg.t as usize * self.cfg.v as usize,
        );
        assert_eq!(x.len(), ni, "batch input width");
        assert_eq!(target.len(), nv, "batch target width");
        // SAFETY: `ptr` is the mapping of a buffer of exactly this word count, host-coherent.
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr(), self.x_stage.ptr, ni);
            std::ptr::copy_nonoverlapping(target.as_ptr(), self.zt_stage.ptr, nv);
        }
    }

    /// Record one command buffer: staging→device copies, the forward, the KL head, the backward,
    /// Adam, and the gradient readbacks; submit and wait.
    ///
    /// Dispatch count: 56 (47 layer dispatches + 9 Adam), excluding the compute barriers and the
    /// copies. No pipeline is built here: the nine Adam dispatches are built in
    /// [`Trainer::new`], and this call writes the step's two bias-correction words.
    ///
    /// Complexity: O(1) host records, 56 device dispatches.
    pub fn step(&mut self, g: &GpuCtx) {
        self.step_inner(g, None);
    }

    /// [`Trainer::step`] with a device timestamp written before and after every dispatch and the
    /// host phases timed. `pool` must hold [`StepProfile::QUERIES`] queries.
    ///
    /// Complexity: as [`Trainer::step`], plus 2·56 timestamp commands.
    pub fn step_traced(&mut self, g: &GpuCtx, pool: &QueryPool) -> StepProfile {
        assert_eq!(pool.count, StepProfile::QUERIES, "timestamp pool size");
        self.step_inner(g, Some(pool))
    }

    /// Host microseconds of the last step, `[bias write, pool, record, submit+idle]`.
    #[must_use]
    pub fn last_host_us(&self) -> [f64; 4] {
        self.host_us
    }

    /// The step and its optional trace. The command-buffer tape is identical either way; a trace
    /// adds one timestamp query before each dispatch and one after it, before the barriers.
    fn step_inner(&mut self, g: &GpuCtx, pool: Option<&QueryPool>) -> StepProfile {
        let vk = g.vk;
        let t0 = std::time::Instant::now();
        self.step_no += 1;
        let (b1t, rbc2) = adam_bias_corrections(self.step_no);
        // SAFETY: the mapping of a two-word host-coherent buffer.
        unsafe {
            *self.bias_stage.ptr = b1t;
            *self.bias_stage.ptr.add(1) = rbc2;
        }
        let mut host = [0f64; 4];
        host[0] = t0.elapsed().as_secs_f64() * 1e6;

        let t1 = std::time::Instant::now();
        let cb = unsafe { vk.command_buffer() };
        host[1] = t1.elapsed().as_secs_f64() * 1e6;

        let mut trace = pool.map(|pool| Trace {
            pool,
            next: 0,
            labels: Vec::with_capacity(STEP_DISPATCHES),
        });
        let t2 = std::time::Instant::now();
        unsafe {
            vk.begin(cb);
            if let Some(tr) = trace.as_mut() {
                vk.cmd_reset_timestamps(cb, tr.pool);
                tr.stamp(vk, cb);
            }
            self.record(vk, cb, trace.as_mut());
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
        }
        host[2] = t2.elapsed().as_secs_f64() * 1e6;

        let t3 = std::time::Instant::now();
        unsafe { vk.submit_wait(cb) };
        host[3] = t3.elapsed().as_secs_f64() * 1e6;
        self.host_us = host;

        let mut profile = StepProfile {
            host_us: host,
            total_us: 0.0,
            prologue_us: 0.0,
            dispatch_us: Vec::new(),
            gap_us: Vec::new(),
            tail_us: 0.0,
            labels: Vec::new(),
        };
        if let Some(tr) = trace.as_ref() {
            let ts = unsafe { vk.read_timestamps(tr.pool) };
            let ticks = f64::from(vk.timestamp_period) / 1000.0;
            let at = |q: u32| ts[q as usize] as f64 * ticks;
            let n = STEP_DISPATCHES as u32;
            profile.total_us = at(1 + 2 * n) - at(0);
            profile.prologue_us = at(1) - at(0);
            for k in 0..n {
                let pre = at(1 + 2 * k);
                let post = at(2 + 2 * k);
                profile.dispatch_us.push(post - pre);
                if k + 1 < n {
                    profile.gap_us.push(at(1 + 2 * (k + 1)) - post);
                }
            }
            profile.tail_us = at(1 + 2 * n) - at(2 * n);
            profile.labels = tr.labels.clone();
        }
        profile
    }

    /// Record the tape. With `trace`, a timestamp is written before each dispatch and one after it
    /// (before [`step_barrier`] and the caller's barrier), so a dispatch span excludes the barrier
    /// gap that follows it.
    fn record(&self, vk: &Vk, cb: CommandBuffer, mut trace: Option<&mut Trace<'_>>) {
        let (t, i) = (self.cfg.t as usize, self.cfg.i as usize);
        unsafe {
            let cbar = |cb| {
                vk.barrier(
                    cb,
                    STAGE_COMPUTE,
                    STAGE_COMPUTE,
                    ACCESS_SHADER_WRITE,
                    ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
                );
            };
            let copy = |src, dst, bytes: usize| {
                let region = BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes as u64,
                };
                (vk.copy_buffer)(cb, src, dst, 1, &region);
            };
            let d = |trace: &mut Option<&mut Trace<'_>>, s: &Step| {
                if let Some(tr) = trace.as_deref_mut() {
                    tr.stamp(vk, cb);
                    tr.labels.push(s.name);
                }
                (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, s.pipe);
                (vk.bind_sets)(
                    cb,
                    PIPELINE_BIND_POINT_COMPUTE,
                    s.layout,
                    0,
                    1,
                    &s.dset,
                    0,
                    std::ptr::null(),
                );
                (vk.dispatch)(cb, s.groups.0, s.groups.1, 1);
                if let Some(tr) = trace.as_deref_mut() {
                    tr.stamp(vk, cb);
                }
                step_barrier(vk, cb);
            };
            // Staging → device input, and the step's two bias-correction words.
            copy(self.x_stage.buf, self.x, t * i * 4);
            copy(self.zt_stage.buf, self.zt, t * self.cfg.v as usize * 4);
            copy(self.bias_stage.buf, self.bias_buf, 8);
            // Seed both residual buffers with `x` (the tape's `x1 = x + o`, `x2 = x + o + f`; the
            // second `o` add rounds identically, so `x2 = x1 + f` entry-wise).
            vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_TRANSFER,
                ACCESS_TRANSFER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_TRANSFER,
                ACCESS_SHADER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            copy(self.x, self.x1, t * i * 4);
            copy(self.x, self.x2, t * i * 4);
            vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_COMPUTE,
                ACCESS_TRANSFER_WRITE,
                ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
            );
            let layer: [&Step; 47] = [
                &self.s_xn,
                &self.s_q,
                &self.s_k,
                &self.s_v,
                &self.s_rq,
                &self.s_rk,
                &self.s_attn,
                &self.s_o,
                &self.s_x1,
                &self.s_x2a,
                &self.s_xn2,
                &self.s_gate,
                &self.s_up,
                &self.s_sw,
                &self.s_f,
                &self.s_x2b,
                &self.s_kl,
                &self.s_dwd,
                &self.s_dha,
                &self.s_swb,
                &self.s_dxn2a,
                &self.s_dxn2b,
                &self.s_dxn2,
                &self.s_dwg,
                &self.s_dwu,
                &self.s_nb2,
                &self.s_dg2,
                &self.s_dx1,
                &self.s_dwo,
                &self.s_do,
                &self.s_rs,
                &self.s_rsr,
                &self.s_ds,
                &self.s_dq,
                &self.s_dkdv,
                &self.s_rbq,
                &self.s_rbk,
                &self.s_dwq,
                &self.s_dwk,
                &self.s_dwv,
                &self.s_dxnq,
                &self.s_dxnk,
                &self.s_dxnv,
                &self.s_dxn1,
                &self.s_dxn2b2,
                &self.s_nb1,
                &self.s_dg1,
            ];
            for s in layer {
                d(&mut trace, s);
                cbar(cb);
            }
            for (n, s) in self.adam.iter().enumerate() {
                if n > 0 {
                    cbar(cb);
                }
                d(&mut trace, s);
            }
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_TRANSFER,
                ACCESS_SHADER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            copy(self.loss, self.rb_loss.buf, t * 4);
            copy(self.dx, self.rb_dx.buf, t * i * 4);
            copy(self.dg1, self.rb_dg1.buf, self.cfg.i as usize * 4);
            copy(self.dg2, self.rb_dg2.buf, self.cfg.i as usize * 4);
            vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_HOST,
                ACCESS_TRANSFER_WRITE,
                ACCESS_HOST_READ,
            );
            if let Some(tr) = trace.as_deref_mut() {
                tr.stamp(vk, cb);
            }
        }
    }

    /// Record the nine weight readbacks on demand; call after `step` before `weights()`.
    ///
    /// Complexity: O(1) host records, 9 device→host copies.
    pub fn sync_weights(&mut self, g: &GpuCtx) {
        let vk = g.vk;
        let cb = unsafe { vk.command_buffer() };
        unsafe {
            vk.begin(cb);
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_TRANSFER,
                ACCESS_SHADER_WRITE,
                ACCESS_TRANSFER_READ,
            );
            for n in 0..9 {
                let bytes = self.cfg_bytes(n);
                let region = BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes as u64,
                };
                (vk.copy_buffer)(cb, self.w[n], self.rb_w[n].buf, 1, &region);
            }
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
    }

    /// Bytes of weight tensor `n` in the composition's order.
    fn cfg_bytes(&self, n: usize) -> usize {
        let (i, h) = (self.cfg.i as usize, self.cfg.h as usize);
        let (qw, kw) = (self.cfg.qw() as usize, self.cfg.kw() as usize);
        match n {
            0 => qw * i * 4,
            1 | 2 => kw * i * 4,
            3 => i * qw * 4,
            4 | 5 => h * i * 4,
            6 => i * h * 4,
            _ => i * 4,
        }
    }

    /// Row 0's KL loss from the last `step`. `losses()` carries every row.
    #[must_use]
    pub fn loss(&self) -> f32 {
        // SAFETY: the mapped readback buffer holds `t` words after `step`'s submit-wait.
        unsafe { *self.rb_loss.ptr }
    }

    /// The per-row KL losses from the last `step`, `t` of them.
    #[must_use]
    pub fn losses(&self) -> Vec<f32> {
        let n = self.cfg.t as usize;
        // SAFETY: as `loss`; the buffer holds `t` words.
        unsafe { std::slice::from_raw_parts(self.rb_loss.ptr, n).to_vec() }
    }

    /// The input gradient `dx` (`t × i`) from the last `step`.
    #[must_use]
    pub fn dx(&self) -> Vec<f32> {
        let n = self.cfg.t as usize * self.cfg.i as usize;
        // SAFETY: as `loss`; the buffer holds `t·i` words.
        unsafe { std::slice::from_raw_parts(self.rb_dx.ptr, n).to_vec() }
    }

    /// The reduced first-norm gain gradient `dγ1` (`i`) from the last `step`.
    #[must_use]
    pub fn dgamma1(&self) -> Vec<f32> {
        let n = self.cfg.i as usize;
        // SAFETY: as `loss`; the buffer holds `i` words.
        unsafe { std::slice::from_raw_parts(self.rb_dg1.ptr, n).to_vec() }
    }

    /// The reduced second-norm gain gradient `dγ2` (`i`) from the last `step`.
    #[must_use]
    pub fn dgamma2(&self) -> Vec<f32> {
        let n = self.cfg.i as usize;
        // SAFETY: as `loss`; the buffer holds `i` words.
        unsafe { std::slice::from_raw_parts(self.rb_dg2.ptr, n).to_vec() }
    }

    /// The nine updated tensors as host f32. **Only valid after [`Trainer::sync_weights`]**; before
    /// that the readback buffers hold the previous step's (or undefined) contents.
    #[must_use]
    pub fn weights(&self) -> HostWeights {
        let n = |k: usize| self.cfg_bytes(k) / 4;
        let get = |k: usize| {
            // SAFETY: `sync_weights` copied `cfg_bytes(k)` bytes into the mapping.
            unsafe { std::slice::from_raw_parts(self.rb_w[k].ptr, n(k)).to_vec() }
        };
        HostWeights {
            wq: get(0),
            wk: get(1),
            wv: get(2),
            wo: get(3),
            wg: get(4),
            wu: get(5),
            wd: get(6),
            g1: get(7),
            g2: get(8),
        }
    }
}
