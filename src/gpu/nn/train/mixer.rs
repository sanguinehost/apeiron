//! One Qwen3.5 trunk layer forward on the device in f32, over whole chunks of `n_ctx` columns that
//! each start at position 0: the twin of the CPU [`Engine::layer_tap_cols`] for the two stage rows a
//! block fit reads, the mixer output (`TAP_ATTN_OUT`, the o-projection input) and the layer output.
//!
//! `n1 = rmsnorm(x, g1)`; the mixer's input projections of `n1`; the mixer
//! ([`DELTA_SCAN_SPV`] for a gated `DeltaNet` layer, [`ATTN_MAP_SPV`] for a softmax layer) to `ao`;
//! `h = x + ao·Woᵀ`; `n2 = rmsnorm(h, g2)`; `y = h + (silu(n2·Wgᵀ) ⊙ n2·Wuᵀ)·Wdᵀ`.
//!
//! The softmax mixer has two faces, chosen by the engine's `attn_gate`. Gated (Qwen3.5): `attn_q` is
//! `[q | gate]` per head and the attention output is scaled by `sigmoid(gate)`. Ungated (Qwen3):
//! `attn_q` is `q` alone and there is no output gate, which is bit 2 of the shader's `mode` word
//! (`src/gpu/spirv/train/attn_map.comp:41`) — Qwen3 attention as
//! `transformers/models/qwen3/modeling_qwen3.py:200-201` computes it (q and k RMS-normalised over the
//! head dimension, then RoPE, no gate).
//!
//! # Evidence grade
//!
//! `tested` against the CPU tap by `tests/learn_block_tune.rs` `layer_fwd_matches_cpu`. The CPU
//! tap runs the `Q8Act` integer schedule; this path is f32 throughout, so the two agree to a
//! relative error, not bit for bit.
//!
//! # Trust boundary
//!
//! Not covered: the GPU driver, the SPIR-V compiler and the f32 transcendental precision of the
//! device.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "widths, rows and workgroup counts index u32 invocations; Q32.32 words map to f32"
)]

use super::block::{download, host_buffer, record_copies, upload, Copies, HostBuffer};
use crate::engine::{Engine, MixerKind};
use crate::gpu::decode::{make_step_named, rec_step, GpuCtx, Step};
use crate::gpu::nn::act::swiglu_fwd_dispatch;
use crate::gpu::nn::add_inplace_bytes;
use crate::gpu::nn::fwd::{linear_fwd_dispatch, rmsnorm_fwd_dispatch};
use crate::gpu::vulkan::*;
use crate::learn::grad::load::{layer_from_engine, load_matrix};
use crate::learn::grad::model::{Config, ModelError};

/// `delta_scan.comp` compiled by `glslc` with [`crate::gpu::nn::GLSLC_ARGS`].
pub static DELTA_SCAN_SPV: &[u8] = include_bytes!("../../spirv/train/delta_scan.spv");
/// `attn_map.comp` compiled by `glslc` with [`crate::gpu::nn::GLSLC_ARGS`].
pub static ATTN_MAP_SPV: &[u8] = include_bytes!("../../spirv/train/attn_map.spv");
/// `delta_bwd.comp` compiled by `glslc` with [`crate::gpu::nn::GLSLC_ARGS`].
pub static DELTA_BWD_SPV: &[u8] = include_bytes!("../../spirv/train/delta_bwd.spv");

const MAX_GROUPS_X: usize = 65_535;

/// The sequence mixer of one layer and its parameters, f32.
#[derive(Debug, Clone)]
pub enum Mixer {
    /// Gated `DeltaNet`: `nk` key heads, `nv` value heads, 128 wide; `conv` channel-major `[C][4]`;
    /// `par` = `[ssm_a (nv) | dt (nv) | ssm_norm (128)]`.
    Affine {
        nk: usize,
        nv: usize,
        conv: Vec<f32>,
        par: Vec<f32>,
    },
    /// Softmax attention: `nh` query heads, `nkv` key heads, `hd` wide, `rot` rotated dims at base
    /// `theta`; `gains` = `[q gains (hd) | k gains (hd)]`. `gated`: `attn_q` carries `[q | gate]`
    /// per head and the output is scaled by `sigmoid(gate)`; when false `attn_q` is `q` alone and
    /// there is no gate (Qwen3, bit 2 of the `mode` word, `src/gpu/spirv/train/attn_map.comp:41`).
    Softmax {
        nh: usize,
        nkv: usize,
        hd: usize,
        rot: usize,
        theta: f32,
        gains: Vec<f32>,
        gated: bool,
    },
}

/// One layer's weights in f32: row-major `rows × cols` matrices.
#[derive(Debug, Clone)]
pub struct LayerHost {
    /// Residual width.
    pub d: usize,
    /// FFN width.
    pub f: usize,
    /// Mixer output width, the o-projection input.
    pub a: usize,
    /// RMS epsilon.
    pub eps: f32,
    pub mixer: Mixer,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    /// The mixer's input projections of `n1`, `(rows, rows × d)`: affine `[qkv, z|b|a]`, softmax
    /// `[q|gate, k, v]`.
    pub proj: Vec<(usize, Vec<f32>)>,
    /// `d × a`.
    pub wo: Vec<f32>,
    /// `[gate (f × d), up (f × d), down (d × f)]`.
    pub ffn: [Vec<f32>; 3],
}

fn f32s(v: &[i64]) -> Vec<f32> {
    v.iter()
        .map(|&w| (w as f64 / 4_294_967_296.0) as f32)
        .collect()
}

impl LayerHost {
    /// Layer `l` of `engine`, its Q32.32 weights mapped to f32.
    ///
    /// # Errors
    ///
    /// As [`layer_from_engine`] and [`load_matrix`]; [`ModelError::Missing`] for an absent affine
    /// vector or QK gain; [`ModelError::Shape`] for a bias, which this forward does not carry.
    ///
    /// # Complexity
    ///
    /// `O(Σ rows·cols)` dequantized weights, each mapped once.
    pub fn from_engine(engine: &Engine, cfg: &Config, l: usize) -> Result<Self, ModelError> {
        let lin = layer_from_engine(engine, cfg, l)?;
        if lin.linears.iter().any(|x| x.bias.is_some()) {
            return Err(ModelError::Shape(format!("blk.{l}: bias")));
        }
        let ec = &engine.config;
        let (d, f) = (cfg.n_embd, cfg.n_ff);
        let w = |k: usize| f32s(&lin.linears[k].w);
        let rows = |k: usize| lin.linears[k].rows;
        let n = lin.linears.len();
        let ffn = [w(n - 3), w(n - 2), w(n - 1)];
        let vectors = engine.decode_vectors();
        let (mixer, proj, wo, a) = match lin.mixer {
            MixerKind::Affine => {
                let geo = ec
                    .affine
                    .ok_or_else(|| ModelError::Missing(format!("blk.{l} affine geometry")))?;
                let av = vectors
                    .affine
                    .get(l)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| ModelError::Missing(format!("blk.{l} affine vectors")))?;
                if geo.head_k != 128 || geo.head_v != 128 || geo.conv_kernel != 4 {
                    return Err(ModelError::Shape(format!(
                        "blk.{l}: affine geometry {geo:?}"
                    )));
                }
                let nv = geo.n_v_heads;
                let mut zba = w(1);
                zba.extend(f32s(&load_matrix(
                    engine,
                    &format!("blk.{l}.ssm_beta.weight"),
                    nv,
                    d,
                )?));
                zba.extend(f32s(&load_matrix(
                    engine,
                    &format!("blk.{l}.ssm_alpha.weight"),
                    nv,
                    d,
                )?));
                let mut par = f32s(&av.a);
                par.extend(f32s(&av.dt));
                par.extend(f32s(&av.norm));
                let v = rows(1);
                (
                    Mixer::Affine {
                        nk: geo.n_k_heads,
                        nv,
                        conv: f32s(&av.conv),
                        par,
                    },
                    vec![(rows(0), w(0)), (v + 2 * nv, zba)],
                    w(2),
                    v,
                )
            }
            MixerKind::Softmax => {
                let qk = vectors
                    .qk_norms_of(l)
                    .ok_or_else(|| ModelError::Missing(format!("blk.{l} qk norms")))?;
                let mut gains = Vec::new();
                for gv in qk {
                    gains
                        .extend(f32s(gv.as_ref().ok_or_else(|| {
                            ModelError::Missing(format!("blk.{l} qk norm"))
                        })?));
                }
                let (nh, hd) = (cfg.n_head, cfg.head_dim);
                (
                    Mixer::Softmax {
                        nh,
                        nkv: cfg.n_head_kv,
                        hd,
                        rot: ec.rot_dims,
                        theta: cfg.rope_theta as f32,
                        gains,
                        gated: ec.attn_gate,
                    },
                    vec![(rows(0), w(0)), (rows(1), w(1)), (rows(2), w(2))],
                    w(3),
                    nh * hd,
                )
            }
        };
        Ok(Self {
            d,
            f,
            a,
            eps: ec.rms_eps.to_f64() as f32,
            mixer,
            attn_norm: f32s(&lin.attn_norm),
            ffn_norm: f32s(&lin.ffn_norm),
            proj,
            wo,
            ffn,
        })
    }

    /// Every weight in upload order: the two norms, the projections, `wo`, the three FFN
    /// matrices, then the mixer's own parameters.
    fn tensors(&self) -> Vec<&[f32]> {
        let mut v: Vec<&[f32]> = vec![&self.attn_norm, &self.ffn_norm];
        v.extend(self.proj.iter().map(|p| p.1.as_slice()));
        v.push(&self.wo);
        v.extend(self.ffn.iter().map(Vec::as_slice));
        match &self.mixer {
            Mixer::Affine { conv, par, .. } => v.extend([conv.as_slice(), par.as_slice()]),
            Mixer::Softmax { gains, .. } => v.push(gains),
        }
        v
    }
}

/// A grid of `wg` workgroups folded into `(x ≤ 65535, y)`.
fn fold(wg: usize) -> (u32, u32) {
    let gx = wg.clamp(1, MAX_GROUPS_X);
    (gx as u32, wg.div_ceil(gx).max(1) as u32)
}

/// One step of `spv` with `meta` at binding 0 and `bufs` after it.
///
/// # Safety
///
/// The buffers hold what the shader reads and writes for `meta`.
unsafe fn shader(
    g: &GpuCtx,
    spv: &[u8],
    meta: &[u32],
    bufs: &[Handle],
    groups: (u32, u32),
    name: &'static str,
) -> Step {
    let mut all = vec![g.up_u32(meta)];
    all.extend_from_slice(bufs);
    let (layout, dset) = g.vk.ssbo_set(&all);
    Step {
        pipe: g.vk.pipeline(spv, layout),
        layout,
        dset,
        groups,
        name,
        bytes: 0,
    }
}

/// The device layer forward: weights resident, one recorded tape over `chunks · n_ctx` columns.
pub struct LayerFwd {
    t: usize,
    d: usize,
    a: usize,
    x: Handle,
    ao: Handle,
    y: Handle,
    /// The weight buffers in [`LayerHost::tensors`] order, and their lengths.
    weights: Vec<Handle>,
    lens: Vec<usize>,
    /// Weight uploads are staged through `stage`; `x` goes in and `ao`, `y` come out through
    /// `stage` and `rb`.
    stage: HostBuffer,
    rb: HostBuffer,
    cb: CommandBuffer,
    /// The same tape, re-recordable for a resident stream's column copies ([`LayerFwd::run_dev`]).
    cb_dev: CommandBuffer,
    /// The recorded tape, kept so [`LayerFwd::run_dev`] can re-record it beside the step's copies.
    steps: Vec<Step>,
    xfer: CommandBuffer,
}

impl LayerFwd {
    /// Allocate, upload `host` and record the tape for `chunks` chunks of `n_ctx` columns.
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value.
    ///
    /// # Complexity
    ///
    /// One upload of every weight, `O(Σ rows·cols)` words; `O(T·(d + f + C))` device words.
    #[allow(clippy::too_many_lines, reason = "one tape")]
    pub unsafe fn new(g: &GpuCtx, host: &LayerHost, n_ctx: usize, chunks: usize) -> Self {
        assert!(n_ctx <= 1024, "attn_map holds 1024 scores");
        let LayerHost { d, f, a, eps, .. } = *host;
        let t = n_ctx * chunks;
        let dev = |n: usize| g.dev(n * 4);
        let biggest = host
            .proj
            .iter()
            .map(|p| p.1.len())
            .chain([host.wo.len(), f * d, t * d, t * a])
            .max()
            .unwrap_or(1);
        let stage = host_buffer(g, biggest, BUFFER_USAGE_TRANSFER_SRC);
        let rb = host_buffer(g, t * d.max(a), BUFFER_USAGE_TRANSFER_DST);
        let xfer = g.vk.reusable_command_buffer();
        let put = |data: &[f32]| {
            let h = dev(data.len());
            upload(g, xfer, &stage, h, data);
            h
        };
        let tensors = host.tensors();
        let lens: Vec<usize> = tensors.iter().map(|w| w.len()).collect();
        let weights: Vec<Handle> = tensors.iter().map(|w| put(w)).collect();
        let np = host.proj.len();
        let (g1, g2) = (weights[0], weights[1]);
        let wp: Vec<(usize, Handle)> = host
            .proj
            .iter()
            .zip(&weights[2..])
            .map(|(p, &h)| (p.0, h))
            .collect();
        let wo = weights[2 + np];
        let wf = [weights[3 + np], weights[4 + np], weights[5 + np]];
        let extra = &weights[6 + np..];
        let (x, n1, ao, h, n2, y) = (
            dev(t * d),
            dev(t * d),
            dev(t * a),
            dev(t * d),
            dev(t * d),
            dev(t * d),
        );
        let (gt, up, mm) = (dev(t * f), dev(t * f), dev(t * f));
        let pb: Vec<Handle> = wp.iter().map(|&(r, _)| dev(t * r)).collect();
        let (tu, du, fu) = (t as u32, d as u32, f as u32);
        let add = |dst: Handle, src: Handle, n: usize| {
            make_step_named(
                g,
                &add_inplace_bytes(n as u32, 256),
                &[dst, src],
                (n as u32).div_ceil(256),
                1,
                "add_inplace",
            )
        };
        let mut steps = vec![rmsnorm_fwd_dispatch(g, x, g1, n1, tu, du)];
        for (&(r, w), &p) in wp.iter().zip(&pb) {
            steps.push(linear_fwd_dispatch(g, n1, w, p, tu, du, r as u32));
        }
        match &host.mixer {
            Mixer::Affine { nk, nv, .. } => {
                let c = wp[0].0;
                let kd = nk * 128;
                assert_eq!(c, 2 * kd + a, "qkv rows");
                let (cw, pr) = (extra[0], extra[1]);
                let meta = [
                    n_ctx as u32,
                    c as u32,
                    kd as u32,
                    a as u32,
                    *nv as u32,
                    eps.to_bits(),
                    *nk as u32,
                ];
                steps.push(shader(
                    g,
                    DELTA_SCAN_SPV,
                    &meta,
                    &[pb[0], pb[1], cw, pr, ao],
                    ((chunks * nv) as u32, 1),
                    "delta_scan",
                ));
            }
            Mixer::Softmax {
                nh,
                nkv,
                hd,
                rot,
                theta,
                gated,
                ..
            } => {
                assert!(*hd <= 256 && nh % nkv == 0, "attn_map geometry");
                let qrows = nh * hd * if *gated { 2 } else { 1 };
                assert_eq!(wp[0].0, qrows, "attn_q rows");
                let gn = extra[0];
                // The gate is read and written only when gated; ungated, one word keeps binding 5
                // valid without a `t·nh·hd` buffer the shader never touches.
                let gate = if *gated { dev(t * nh * hd) } else { dev(1) };
                let (q, k) = (dev(t * nh * hd), dev(t * nkv * hd));
                let bufs = [pb[0], pb[1], gn, q, gate, k, pb[2], ao];
                let flag = if *gated { 0u32 } else { 4 };
                for (mode, wg, name) in [
                    (flag, t * (nh + nkv), "attn_prep"),
                    (1 | flag, chunks * nh * n_ctx, "attn_attend"),
                ] {
                    let meta = [
                        mode,
                        n_ctx as u32,
                        *nh as u32,
                        *nkv as u32,
                        *hd as u32,
                        *rot as u32,
                        theta.to_bits(),
                        eps.to_bits(),
                        wg as u32,
                    ];
                    steps.push(shader(g, ATTN_MAP_SPV, &meta, &bufs, fold(wg), name));
                }
            }
        }
        steps.extend([
            linear_fwd_dispatch(g, ao, wo, h, tu, a as u32, du),
            add(h, x, t * d),
            rmsnorm_fwd_dispatch(g, h, g2, n2, tu, du),
            linear_fwd_dispatch(g, n2, wf[0], gt, tu, du, fu),
            linear_fwd_dispatch(g, n2, wf[1], up, tu, du, fu),
            swiglu_fwd_dispatch(g, gt, up, mm, (t * f) as u32),
            linear_fwd_dispatch(g, mm, wf[2], y, tu, fu, du),
            add(y, h, t * d),
        ]);
        let vk = g.vk;
        let cb = vk.command_buffer();
        vk.begin(cb);
        for s in &steps {
            rec_step(vk, cb, s);
        }
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        Self {
            t,
            d,
            a,
            x,
            ao,
            y,
            weights,
            lens,
            stage,
            rb,
            cb,
            cb_dev: g.vk.reusable_command_buffer(),
            steps,
            xfer,
        }
    }

    /// Re-upload the weights of `host`, a layer of the same geometry, into the recorded tape:
    /// the device has no free, so one tape per mixer kind serves every layer.
    ///
    /// # Safety
    ///
    /// No submission of this value's command buffers is pending.
    ///
    /// # Complexity
    ///
    /// One upload of every weight, `O(Σ rows·cols)` words.
    pub unsafe fn load(&self, g: &GpuCtx, host: &LayerHost) {
        let tensors = host.tensors();
        assert_eq!(tensors.len(), self.lens.len(), "layer geometry");
        for ((w, &h), &n) in tensors.iter().zip(&self.weights).zip(&self.lens) {
            assert_eq!(w.len(), n, "layer geometry");
            upload(g, self.xfer, &self.stage, h, w);
        }
    }

    /// The layer over the `T × d` input `x`: the mixer output `T × a` and the layer output `T × d`.
    ///
    /// # Safety
    ///
    /// No other submission of this value's command buffers is pending.
    ///
    /// # Complexity
    ///
    /// One tape submission: the linears' `2·T·d·(C + a + 2·f) + 2·T·a·d + 2·T·f·d` flops, the
    /// mixer's `O(T·NV·128²)` scan or `O(T·n_ctx·nh·hd)` attention; `T·(d + a + d)` words moved.
    pub unsafe fn run(&self, g: &GpuCtx, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), self.t * self.d, "layer input");
        upload(g, self.xfer, &self.stage, self.x, x);
        g.vk.submit_wait(self.cb);
        let ao = download(g, self.xfer, &self.rb, self.ao, self.t * self.a);
        let y = download(g, self.xfer, &self.rb, self.y, self.t * self.d);
        (ao, y)
    }

    /// The layer on a resident stream: `pre` copies the step's columns into `x`, the tape runs,
    /// and `post` copies `ao` and `y` back out, all in one command buffer and one submission.
    ///
    /// # Safety
    ///
    /// Every group's buffers hold its regions; no submission of this value is pending.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work and the tape's `O(1)` dispatches re-recorded; the copies move
    /// `O(chunks·n_ctx·(d + a + d))` bytes on the device.
    pub unsafe fn run_dev(&self, g: &GpuCtx, pre: &[Copies], post: &[Copies]) {
        let vk = g.vk;
        let cb = self.cb_dev;
        vk.reset_cb(cb);
        vk.begin(cb);
        record_copies(vk, cb, pre);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_COMPUTE,
            ACCESS_TRANSFER_WRITE,
            ACCESS_SHADER_READ,
        );
        for s in &self.steps {
            rec_step(vk, cb, s);
        }
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_TRANSFER,
            ACCESS_SHADER_WRITE,
            ACCESS_TRANSFER_READ,
        );
        record_copies(vk, cb, post);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
    }

    /// The tape's device buffers: the input `x`, the mixer output `ao` (`t × a`) and the layer
    /// output `y` (`t × d`).
    #[must_use]
    pub fn handles(&self) -> (Handle, Handle, Handle) {
        (self.x, self.ao, self.y)
    }
}

/// The geometry of one affine mixer's backward: `kd = nk·128`, `v = nv·128`, `c = 2·kd + v`, and
/// `chunks` sequences of `n_ctx` tokens that each start from a zero state, as [`LayerFwd`] reads
/// them.
#[derive(Debug, Clone, Copy)]
pub struct DeltaBwdShape {
    /// Tokens per sequence.
    pub n_ctx: usize,
    /// Sequences.
    pub chunks: usize,
    /// `qkv` width, `2·kd + v`.
    pub c: usize,
    /// Query/key width.
    pub kd: usize,
    /// Value width.
    pub v: usize,
    /// Value heads.
    pub nv: usize,
    /// Key heads; must equal `nv`.
    pub nk: usize,
    /// RMS epsilon.
    pub eps: f32,
}

/// The gradients [`DeltaBwd`] returns, host f32.
#[derive(Debug, Clone)]
pub struct DeltaBwdGrads {
    /// `[T][C]`: the pre-conv input projection.
    pub dqkv: Vec<f32>,
    /// `[T][V + 2·NV]`: the `z`, `b` and `a` projections.
    pub dzba: Vec<f32>,
    /// `[C][4]`: the conv weights, summed over tokens and sequences.
    pub dconv: Vec<f32>,
    /// `[NV]`: the `ssm_a` weights, summed over tokens and sequences.
    pub dssm_a: Vec<f32>,
    /// `[NV]`: the `dt` weights, summed over tokens and sequences.
    pub ddt: Vec<f32>,
    /// `[128]`: the output norm gains, summed over value heads and tokens.
    pub dgain: Vec<f32>,
}

/// The device backward of one affine mixer: [`DELTA_BWD_SPV`] recorded as one recompute, one
/// reverse and one reduce dispatch, over a state store the reverse pass reads once per token.
///
/// The forward recompute writes `M_{t−1}` for every token to `mst` (`T·NV·128²` words) and
/// `q̂, k̂, v, β, α, qs, ks` to `vec`; the reverse pass holds `G = ∂L/∂M_t` and the token's
/// `M_{t−1}` in registers and never crosses a chunk boundary, so the chunks stay independent, as in
/// the forward.
///
/// # Evidence grade
///
/// `tested`: `delta_bwd_device_matches_ref` against the f64 reference in this module to `1e-3`
/// relative per tensor.
///
/// # Trust boundary
///
/// Not covered: the driver, the SPIR-V compiler, and the f32 precision of the device.
#[allow(
    dead_code,
    reason = "the state store, the vector store and the three partial buffers are read and written 
              by the recorded tape only; the handles are kept so they name the tape's allocation"
)]
pub struct DeltaBwd {
    s: DeltaBwdShape,
    t: usize,
    qkv: Handle,
    zba: Handle,
    conv: Handle,
    par: Handle,
    dy: Handle,
    mst: Handle,
    vsc: Handle,
    dqkv: Handle,
    dzba: Handle,
    p1: Handle,
    p2: Handle,
    p3: Handle,
    dpar: Handle,
    dconv: Handle,
    stage: HostBuffer,
    rb: HostBuffer,
    cb: CommandBuffer,
    xfer: CommandBuffer,
}

impl DeltaBwd {
    /// Allocate the buffers and record the tape for `s`.
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value.
    ///
    /// # Panics
    ///
    /// `nk != nv`: the reverse pass writes each key head's channel gradient from the one value head
    /// that shares it, so `nv > nk` would need the head sum before the conv inverse, which this
    /// shader does not implement.
    ///
    /// # Complexity
    ///
    /// `O(T·NV·128²)` device words for the state store and `O(T·(C + V + 2·NV))` for the inputs and
    /// outputs; the tape is three dispatches, each `O(T·NV·128²)` multiply-adds.
    pub unsafe fn new(g: &GpuCtx, s: DeltaBwdShape) -> Self {
        assert_eq!(
            s.nv, s.nk,
            "delta_bwd: nv == nk, one value head per key head"
        );
        assert_eq!(s.kd, s.nk * 128, "delta_bwd: kd = nk·128");
        assert_eq!(s.v, s.nv * 128, "delta_bwd: v = nv·128");
        assert_eq!(s.c, 2 * s.kd + s.v, "delta_bwd: c = 2·kd + v");
        let DeltaBwdShape {
            n_ctx,
            chunks,
            c,
            kd,
            v,
            nv,
            nk,
            eps,
        } = s;
        let t = n_ctx * chunks;
        let zw = v + 2 * nv;
        let nwg = chunks * nv;
        let dev = |n: usize| g.dev(n * 4);
        let qkv = dev(t * c);
        let zba = dev(t * zw);
        let conv = dev(c * 4);
        let par = dev(2 * nv + 128);
        let dy = dev(t * v);
        let mst = dev(t * nv * 16384);
        let vsc = dev(t * nv * 388);
        let dqkv = dev(t * c);
        let dzba = dev(t * zw);
        let p1 = dev(nwg * 2);
        let p2 = dev(nwg * c * 4);
        let p3 = dev(nwg * 128);
        let dpar = dev(2 * nv + 128);
        let dconv = dev(c * 4);
        let biggest = (t * c).max(t * zw).max(c * 4).max(2 * nv + 128);
        let stage = host_buffer(g, biggest, BUFFER_USAGE_TRANSFER_SRC);
        let rb = host_buffer(g, biggest, BUFFER_USAGE_TRANSFER_DST);
        let xfer = g.vk.reusable_command_buffer();
        let bufs = [
            qkv, zba, conv, par, dy, mst, vsc, dqkv, dzba, p1, p2, p3, dpar, dconv,
        ];
        let meta = |mode: u32| {
            [
                mode,
                n_ctx as u32,
                c as u32,
                kd as u32,
                v as u32,
                nv as u32,
                eps.to_bits(),
                nk as u32,
                nwg as u32,
            ]
        };
        let mut steps = Vec::new();
        for (mode, name) in [
            (0u32, "delta_bwd_recompute"),
            (1, "delta_bwd_reverse"),
            (2, "delta_bwd_reduce"),
        ] {
            let groups = if mode == 2 { 1 } else { nwg };
            steps.push(shader(
                g,
                DELTA_BWD_SPV,
                &meta(mode),
                &bufs,
                (groups as u32, 1),
                name,
            ));
        }
        let vk = g.vk;
        let cb = vk.command_buffer();
        vk.begin(cb);
        // The reverse pass accumulates the head's channel gradient into `dqkv` from up to four taps.
        (vk.fill_buffer)(cb, dqkv, 0, (t * c * 4) as u64, 0);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_COMPUTE,
            ACCESS_TRANSFER_WRITE,
            ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
        );
        for st in &steps {
            rec_step(vk, cb, st);
        }
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        Self {
            s,
            t,
            qkv,
            zba,
            conv,
            par,
            dy,
            mst,
            vsc,
            dqkv,
            dzba,
            p1,
            p2,
            p3,
            dpar,
            dconv,
            stage,
            rb,
            cb,
            xfer,
        }
    }

    /// The output gradient `dy` alone does not enter; the tape submission, transfers excluded.
    ///
    /// # Safety
    ///
    /// No other submission of this value's command buffers is pending.
    pub unsafe fn submit(&self, g: &GpuCtx) {
        g.vk.submit_wait(self.cb);
    }

    /// The backward at `(qkv, zba, conv, par)` for the output gradient `dy`: upload, one tape
    /// submission, download.
    ///
    /// # Safety
    ///
    /// No other submission of this value's command buffers is pending.
    ///
    /// # Complexity
    ///
    /// One tape submission, `O(T·NV·128²)` multiply-adds and `O(T·(C + V))` words moved each way.
    pub unsafe fn run(
        &self,
        g: &GpuCtx,
        qkv: &[f32],
        zba: &[f32],
        conv: &[f32],
        par: &[f32],
        dy: &[f32],
    ) -> DeltaBwdGrads {
        let DeltaBwdShape { c, v, nv, .. } = self.s;
        assert_eq!(qkv.len(), self.t * c, "qkv");
        assert_eq!(zba.len(), self.t * (v + 2 * nv), "zba");
        assert_eq!(conv.len(), c * 4, "conv");
        assert_eq!(par.len(), 2 * nv + 128, "par");
        assert_eq!(dy.len(), self.t * v, "dy");
        for (h, d) in [
            (self.qkv, qkv),
            (self.zba, zba),
            (self.conv, conv),
            (self.par, par),
            (self.dy, dy),
        ] {
            upload(g, self.xfer, &self.stage, h, d);
        }
        self.submit(g);
        let dqkv = download(g, self.xfer, &self.rb, self.dqkv, self.t * c);
        let dzba = download(g, self.xfer, &self.rb, self.dzba, self.t * (v + 2 * nv));
        let dconv = download(g, self.xfer, &self.rb, self.dconv, c * 4);
        let dpar = download(g, self.xfer, &self.rb, self.dpar, 2 * nv + 128);
        DeltaBwdGrads {
            dqkv,
            dzba,
            dconv,
            dssm_a: dpar[..nv].to_vec(),
            ddt: dpar[nv..2 * nv].to_vec(),
            dgain: dpar[2 * nv..].to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::nn::GLSLC_ARGS;
    use std::path::Path;
    use std::time::Instant;

    fn fresh(name: &str) -> Vec<u8> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv/train");
        let out = std::env::temp_dir().join(format!("{name}_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(dir.join(format!("{name}.comp")))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let b = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        b
    }

    #[test]
    fn mixer_spirv_is_the_glsl() {
        assert!(
            fresh("delta_scan") == DELTA_SCAN_SPV,
            "delta_scan.spv is stale"
        );
        assert!(fresh("attn_map") == ATTN_MAP_SPV, "attn_map.spv is stale");
        assert!(
            fresh("delta_bwd") == DELTA_BWD_SPV,
            "delta_bwd.spv is stale"
        );
    }

    // ---- the f64 reference: the forward of `delta_scan.comp` and its adjoint -------------------
    //
    // The oracle for both gates: central finite differences on the forward (`delta_bwd_ref_matches_fd`)
    // and the device's answer (`delta_bwd_device_matches_ref`). f64 throughout, the gather/scatter
    // layout of the shader, the same chunk-local zero state.

    const HD: usize = 128;
    const FD_H: f64 = 1e-6;

    /// One affine mixer's inputs in f64.
    #[derive(Clone)]
    struct RefIn {
        n_ctx: usize,
        chunks: usize,
        c: usize,
        kd: usize,
        v: usize,
        nv: usize,
        nk: usize,
        eps: f64,
        qkv: Vec<f64>,
        zba: Vec<f64>,
        conv: Vec<f64>,
        par: Vec<f64>,
        dy: Vec<f64>,
    }

    /// The forward's per-token state the reverse pass reads.
    struct HeadCache {
        /// `M_{t−1}` per token, `n_ctx·128²`.
        mprev: Vec<f64>,
        /// `q̂`, `k̂`, `v`, `u = o/r`, `silu(z)` per token, `n_ctx·128` each.
        qh: Vec<f64>,
        kh: Vec<f64>,
        vv: Vec<f64>,
        u: Vec<f64>,
        sz: Vec<f64>,
        /// `β`, `α`, `qs`, `ks` per token.
        scal: Vec<[f64; 4]>,
        /// `r` per token.
        r: Vec<f64>,
    }

    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    fn sigm(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn splus(x: f64) -> f64 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// The forward of one chunk sequence and value head `h`; `keep` also stores the reverse pass's
    /// cache. Returns the chunk-local `[n_ctx·128]` mixer output.
    fn head_forward(x: &RefIn, seq: usize, h: usize, keep: bool) -> (Vec<f64>, Option<HeadCache>) {
        let (n_ctx, c, kdw, v, nv, nk) = (x.n_ctx, x.c, x.kd, x.v, x.nv, x.nk);
        let zw = v + 2 * nv;
        let g = h % nk;
        let (ssm_a, dt) = (x.par[h], x.par[nv + h]);
        let mut m = vec![0.0f64; HD * HD];
        let mut y = vec![0.0f64; n_ctx * HD];
        let mut cache = HeadCache {
            mprev: vec![0.0; if keep { n_ctx * HD * HD } else { 0 }],
            qh: vec![0.0; n_ctx * HD],
            kh: vec![0.0; n_ctx * HD],
            vv: vec![0.0; n_ctx * HD],
            u: vec![0.0; n_ctx * HD],
            sz: vec![0.0; n_ctx * HD],
            scal: vec![[0.0; 4]; n_ctx],
            r: vec![0.0; n_ctx],
        };
        let (mut o, mut kdd) = ([0.0f64; HD], [0.0f64; HD]);
        for t in 0..n_ctx {
            let row = seq * n_ctx + t;
            // conv: silu(Σ_k w[k]·x[t−3+k][ch]), zero before the chunk start.
            for i in 0..HD {
                for (ci, ch) in [g * HD + i, kdw + g * HD + i, 2 * kdw + h * HD + i]
                    .into_iter()
                    .enumerate()
                {
                    let mut s = 0.0;
                    for tap in 0..4 {
                        let ti = t as isize - 3 + tap as isize;
                        if ti >= 0 {
                            s += x.conv[ch * 4 + tap] * x.qkv[(seq * n_ctx + ti as usize) * c + ch];
                        }
                    }
                    match ci {
                        0 => cache.qh[t * HD + i] = silu(s),
                        1 => cache.kh[t * HD + i] = silu(s),
                        _ => cache.vv[t * HD + i] = silu(s),
                    }
                }
            }
            let nq = dot(
                &cache.qh[t * HD..(t + 1) * HD],
                &cache.qh[t * HD..(t + 1) * HD],
            )
            .sqrt()
            .max(x.eps);
            let nk = dot(
                &cache.kh[t * HD..(t + 1) * HD],
                &cache.kh[t * HD..(t + 1) * HD],
            )
            .sqrt()
            .max(x.eps);
            let qs = 1.0 / (HD as f64).sqrt() / nq;
            let ks = 1.0 / nk;
            for i in 0..HD {
                cache.qh[t * HD + i] *= qs;
                cache.kh[t * HD + i] *= ks;
            }
            let beta = sigm(x.zba[row * zw + v + h]);
            let ar = x.zba[row * zw + v + nv + h] + dt;
            let alpha = (ssm_a * splus(ar)).exp();
            if keep {
                cache.mprev[t * HD * HD..(t + 1) * HD * HD].copy_from_slice(&m);
            }
            let qhat = cache.qh[t * HD..(t + 1) * HD].to_vec();
            let khat = cache.kh[t * HD..(t + 1) * HD].to_vec();
            for p in m.iter_mut() {
                *p *= alpha;
            }
            for jj in 0..HD {
                kdd[jj] = dot(&m[jj * HD..jj * HD + HD], &khat);
            }
            for jj in 0..HD {
                let d = (cache.vv[t * HD + jj] - kdd[jj]) * beta;
                for i in 0..HD {
                    m[jj * HD + i] += d * khat[i];
                }
            }
            for jj in 0..HD {
                o[jj] = dot(&m[jj * HD..jj * HD + HD], &qhat);
            }
            let msq: f64 = o.iter().map(|p| p * p).sum();
            let r = (msq / HD as f64 + x.eps).sqrt();
            for jj in 0..HD {
                let zn = x.zba[row * zw + h * HD + jj];
                let sz = silu(zn);
                let uu = o[jj] / r;
                cache.u[t * HD + jj] = uu;
                cache.sz[t * HD + jj] = sz;
                y[t * HD + jj] = uu * x.par[2 * nv + jj] * sz;
            }
            cache.r[t] = r;
            cache.scal[t] = [beta, alpha, qs, ks];
        }
        (y, keep.then_some(cache))
    }

    /// The forward over every (sequence, head) in the shader's `[T][V]` layout.
    fn ref_forward(x: &RefIn) -> Vec<f64> {
        let mut y = vec![0.0f64; x.chunks * x.n_ctx * x.v];
        for seq in 0..x.chunks {
            for h in 0..x.nv {
                let (yl, _) = head_forward(x, seq, h, false);
                for t in 0..x.n_ctx {
                    let row = seq * x.n_ctx + t;
                    y[row * x.v + h * HD..row * x.v + (h + 1) * HD]
                        .copy_from_slice(&yl[t * HD..(t + 1) * HD]);
                }
            }
        }
        y
    }

    /// The gradients of every input and parameter.
    struct RefOut {
        dqkv: Vec<f64>,
        dzba: Vec<f64>,
        dconv: Vec<f64>,
        dssm_a: Vec<f64>,
        ddt: Vec<f64>,
        dgain: Vec<f64>,
    }

    fn ref_grads(x: &RefIn) -> RefOut {
        let (n_ctx, chunks, c, kdw, v, nv, nk) = (x.n_ctx, x.chunks, x.c, x.kd, x.v, x.nv, x.nk);
        let zw = v + 2 * nv;
        let mut dqkv = vec![0.0f64; chunks * n_ctx * c];
        let mut dzba = vec![0.0f64; chunks * n_ctx * zw];
        let mut dconv = vec![0.0f64; c * 4];
        let mut dssm_a = vec![0.0f64; nv];
        let mut ddt = vec![0.0f64; nv];
        let mut dgain = vec![0.0f64; HD];
        for seq in 0..chunks {
            for h in 0..nv {
                let (_, cache) = head_forward(x, seq, h, true);
                let cache = cache.expect("cache");
                let g = h % nk;
                let (ssm_a, dt) = (x.par[h], x.par[nv + h]);
                let mut gg = vec![0.0f64; HD * HD];
                for ti in (0..n_ctx).rev() {
                    let row = seq * n_ctx + ti;
                    let [beta, alpha, qs, ks] = cache.scal[ti];
                    let r = cache.r[ti];
                    let mprev = &cache.mprev[ti * HD * HD..(ti + 1) * HD * HD];
                    let qhat = &cache.qh[ti * HD..(ti + 1) * HD];
                    let khat = &cache.kh[ti * HD..(ti + 1) * HD];
                    let vrow = &cache.vv[ti * HD..(ti + 1) * HD];
                    // The output path: do_n, do, dz, dgain.
                    let (mut don, mut uu, mut dof) = ([0.0f64; HD], [0.0f64; HD], [0.0f64; HD]);
                    let mut m1 = 0.0;
                    for jj in 0..HD {
                        let dyj = x.dy[row * v + h * HD + jj];
                        let sz = cache.sz[ti * HD + jj];
                        let gn = x.par[2 * nv + jj];
                        don[jj] = dyj * gn * sz;
                        uu[jj] = cache.u[ti * HD + jj];
                        m1 += uu[jj] * don[jj];
                        dgain[jj] += dyj * uu[jj] * sz;
                        let zn = x.zba[row * zw + h * HD + jj];
                        let sg = sigm(zn);
                        dzba[row * zw + h * HD + jj] =
                            dyj * uu[jj] * gn * sg * (1.0 + zn * (1.0 - sg));
                    }
                    m1 /= HD as f64;
                    for jj in 0..HD {
                        dof[jj] = (don[jj] - uu[jj] * m1) / r;
                    }
                    for jj in 0..HD {
                        for i in 0..HD {
                            gg[jj * HD + i] += dof[jj] * qhat[i];
                        }
                    }
                    // M_til = α·M_{t−1}: the scan's row dots and the delta.
                    let (mut kdd, mut del) = ([0.0f64; HD], [0.0f64; HD]);
                    for jj in 0..HD {
                        let mut s = 0.0;
                        for i in 0..HD {
                            s += alpha * mprev[jj * HD + i] * khat[i];
                        }
                        kdd[jj] = s;
                        del[jj] = (vrow[jj] - s) * beta;
                    }
                    // dd, dv, dβ.
                    let (mut dd, mut dcv, mut dbeta) = ([0.0f64; HD], [0.0f64; HD], 0.0);
                    for jj in 0..HD {
                        let mut s = 0.0;
                        for i in 0..HD {
                            s += gg[jj * HD + i] * khat[i];
                        }
                        dd[jj] = s;
                        dbeta += s * (vrow[jj] - kdd[jj]);
                        dcv[jj] = beta * s;
                    }
                    // dq̂ = M_tᵀ·do, dk̂ = Gᵀ·d − β·M_tilᵀ·dd.
                    let (mut dqh, mut dkh) = ([0.0f64; HD], [0.0f64; HD]);
                    for i in 0..HD {
                        let (mut a1, mut a2, mut a3) = (0.0, 0.0, 0.0);
                        for jj in 0..HD {
                            let mt = alpha * mprev[jj * HD + i] + del[jj] * khat[i];
                            a1 += gg[jj * HD + i] * del[jj];
                            a2 += mprev[jj * HD + i] * dd[jj];
                            a3 += mt * dof[jj];
                        }
                        dkh[i] = a1 - beta * alpha * a2;
                        dqh[i] = a3;
                    }
                    // dα, then G ← α·dM_til.
                    let mut dalpha = 0.0;
                    for jj in 0..HD {
                        for i in 0..HD {
                            let dmt = gg[jj * HD + i] - beta * dd[jj] * khat[i];
                            dalpha += dmt * mprev[jj * HD + i];
                            gg[jj * HD + i] = alpha * dmt;
                        }
                    }
                    // The two norm inverses.
                    let sqd = dot(qhat, &dqh);
                    let skd = dot(khat, &dkh);
                    let (mut dq, mut dk) = ([0.0f64; HD], [0.0f64; HD]);
                    for i in 0..HD {
                        dq[i] = qs * (dqh[i] - HD as f64 * qhat[i] * sqd);
                        dk[i] = ks * (dkh[i] - khat[i] * skd);
                    }
                    // The gate chain.
                    dzba[row * zw + v + h] = dbeta * beta * (1.0 - beta);
                    let ar = x.zba[row * zw + v + nv + h] + dt;
                    let dsp = dalpha * alpha * ssm_a;
                    dssm_a[h] += dalpha * alpha * splus(ar);
                    let dsum = dsp * if ar > 20.0 { 1.0 } else { sigm(ar) };
                    ddt[h] += dsum;
                    dzba[row * zw + v + nv + h] = dsum;
                    // The conv inverse: ds = dc·silu'(s), dw[k] += ds·x[t−3+k], dx[t−3+k] += ds·w[k].
                    for i in 0..HD {
                        let dcs = [dq[i], dk[i], dcv[i]];
                        for (ci, ch) in [g * HD + i, kdw + g * HD + i, 2 * kdw + h * HD + i]
                            .into_iter()
                            .enumerate()
                        {
                            let mut s = 0.0;
                            for tap in 0..4 {
                                let tix = ti as isize - 3 + tap as isize;
                                if tix >= 0 {
                                    s += x.conv[ch * 4 + tap]
                                        * x.qkv[(seq * n_ctx + tix as usize) * c + ch];
                                }
                            }
                            let sg = sigm(s);
                            let ds = dcs[ci] * sg * (1.0 + s * (1.0 - sg));
                            for tap in 0..4 {
                                let tix = ti as isize - 3 + tap as isize;
                                if tix >= 0 {
                                    let rr = seq * n_ctx + tix as usize;
                                    dconv[ch * 4 + tap] += ds * x.qkv[rr * c + ch];
                                    dqkv[rr * c + ch] += ds * x.conv[ch * 4 + tap];
                                }
                            }
                        }
                    }
                }
            }
        }
        RefOut {
            dqkv,
            dzba,
            dconv,
            dssm_a,
            ddt,
            dgain,
        }
    }

    /// A deterministic uniform generator: no dependency, bounded values, smooth at every knee.
    struct Rng(u64);

    impl Rng {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
        }

        /// A uniform value in `[lo, hi)`.
        fn r(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn f32s(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    /// The model's scaling: hidden states `±0.5`, gates `b`, `a` `±0.5`, `ssm_a` in `(−1.2, −0.2)`,
    /// `dt` `±0.5`, gains in `(0.5, 1.5)`, conv weights `±0.5`, output gradient `±1`.
    fn make_case(n_ctx: usize, chunks: usize, nv: usize, nk: usize, seed: u64) -> RefIn {
        let (c, v) = (2 * nk * HD + nv * HD, nv * HD);
        let zw = v + 2 * nv;
        let t = n_ctx * chunks;
        let mut r = Rng(seed);
        let mut qkv = vec![0.0f64; t * c];
        for p in &mut qkv {
            *p = r.r(-0.5, 0.5) as f32 as f64;
        }
        let mut zba = vec![0.0f64; t * zw];
        for p in &mut zba {
            *p = r.r(-0.5, 0.5) as f32 as f64;
        }
        let mut conv = vec![0.0f64; c * 4];
        for p in &mut conv {
            *p = r.r(-0.5, 0.5) as f32 as f64;
        }
        let mut par = vec![0.0f64; 2 * nv + HD];
        for p in par[..nv].iter_mut() {
            *p = r.r(-1.2, -0.2) as f32 as f64;
        }
        for p in par[nv..2 * nv].iter_mut() {
            *p = r.r(-0.5, 0.5) as f32 as f64;
        }
        for p in par[2 * nv..].iter_mut() {
            *p = r.r(0.5, 1.5) as f32 as f64;
        }
        let mut dy = vec![0.0f64; t * v];
        for p in &mut dy {
            *p = r.r(-1.0, 1.0) as f32 as f64;
        }
        RefIn {
            n_ctx,
            chunks,
            c,
            kd: nk * HD,
            v,
            nv,
            nk,
            eps: 1e-6,
            qkv,
            zba,
            conv,
            par,
            dy,
        }
    }

    fn rel32(got: &[f32], want: &[f64]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 1e-30f64;
        for (&p, &q) in got.iter().zip(want) {
            num = num.max((f64::from(p) - q).abs());
            den = den.max(q.abs());
        }
        num / den
    }

    /// Central finite differences on the f64 forward against the f64 backward, every input sampled.
    #[test]
    fn delta_bwd_ref_matches_fd() {
        // The brief's geometry: NV = 2, NK = 1 (two value heads share a key head), 64 tokens.
        let x = make_case(64, 1, 2, 1, 0x5eed_1234);
        let gr = ref_grads(&x);
        let loss =
            |y: &RefIn| -> f64 { ref_forward(y).iter().zip(&y.dy).map(|(p, q)| p * q).sum() };
        let fd =
            |name: &str, analytic: &[f64], n: usize, pert: &dyn Fn(&RefIn, usize, f64) -> RefIn| {
                let (mut num, mut den, mut k) = (0.0f64, 1e-30f64, 0usize);
                let step = (n / 12).max(1);
                let mut i = 0;
                while i < n {
                    let d = (loss(&pert(&x, i, FD_H)) - loss(&pert(&x, i, -FD_H))) / (2.0 * FD_H);
                    num = num.max((d - analytic[i]).abs());
                    den = den.max(d.abs());
                    k += 1;
                    i += step;
                }
                let rel = num / den;
                println!("FD {name}: rel {rel:.3e} over {k} of {n} coordinates");
                assert!(rel <= 1e-6, "FD {name}: rel {rel:.3e}");
            };
        fd("qkv", &gr.dqkv, x.qkv.len(), &|y, i, d| {
            let mut z = y.clone();
            z.qkv[i] += d;
            z
        });
        fd("zba", &gr.dzba, x.zba.len(), &|y, i, d| {
            let mut z = y.clone();
            z.zba[i] += d;
            z
        });
        fd("conv", &gr.dconv, x.conv.len(), &|y, i, d| {
            let mut z = y.clone();
            z.conv[i] += d;
            z
        });
        fd("ssm_a", &gr.dssm_a, x.nv, &|y, i, d| {
            let mut z = y.clone();
            z.par[i] += d;
            z
        });
        fd("dt", &gr.ddt, x.nv, &|y, i, d| {
            let mut z = y.clone();
            z.par[x.nv + i] += d;
            z
        });
        fd("gain", &gr.dgain, HD, &|y, i, d| {
            let mut z = y.clone();
            z.par[2 * y.nv + i] += d;
            z
        });
    }

    /// The device against the f64 reference: a small geometry first, then the model's `T = 2048`.
    #[test]
    fn delta_bwd_device_matches_ref() {
        let vk = unsafe { Vk::load() }.expect("vulkan");
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        for (n_ctx, chunks, nv, tol) in [(64usize, 2usize, 2usize, 1e-4f64), (512, 4, 16, 1e-3)] {
            let x = make_case(n_ctx, chunks, nv, nv, 0x1234_5678 + nv as u64);
            let (qkv, zba, conv, par, dy) = (
                f32s(&x.qkv),
                f32s(&x.zba),
                f32s(&x.conv),
                f32s(&x.par),
                f32s(&x.dy),
            );
            let shape = DeltaBwdShape {
                n_ctx,
                chunks,
                c: x.c,
                kd: x.kd,
                v: x.v,
                nv,
                nk: nv,
                eps: x.eps as f32,
            };
            let bwd = unsafe { DeltaBwd::new(&g, shape) };
            let t0 = Instant::now();
            let got = unsafe { bwd.run(&g, &qkv, &zba, &conv, &par, &dy) };
            let dev_s = t0.elapsed().as_secs_f64();
            let want = ref_grads(&x);
            let rows = [
                ("dqkv", rel32(&got.dqkv, &want.dqkv)),
                ("dzba", rel32(&got.dzba, &want.dzba)),
                ("dconv", rel32(&got.dconv, &want.dconv)),
                ("dssm_a", rel32(&got.dssm_a, &want.dssm_a)),
                ("ddt", rel32(&got.ddt, &want.ddt)),
                ("dgain", rel32(&got.dgain, &want.dgain)),
            ];
            let line: Vec<String> = rows.iter().map(|(n, r)| format!("{n} {r:.3e}")).collect();
            println!(
                "DB T={} NV={} dev {dev_s:.4} s: {}",
                n_ctx * chunks,
                nv,
                line.join(" ")
            );
            for (n, r) in rows {
                assert!(
                    r <= tol && r.is_finite(),
                    "{n}: rel {r:.3e} > {tol:.0e} at T={} NV={nv}",
                    n_ctx * chunks
                );
            }
        }
    }

    /// The device backward wall against the forward `delta_scan` wall at the model's `T = 2048`,
    /// paired: both timings are tape submissions on the same inputs, transfers excluded.
    #[test]
    #[ignore = "needs a Vulkan device; reports the paired walls"]
    fn delta_bwd_timing() {
        let vk = unsafe { Vk::load() }.expect("vulkan");
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let (n_ctx, chunks, nv) = (512usize, 4usize, 16usize);
        let x = make_case(n_ctx, chunks, nv, nv, 0xabcd_0f0f);
        let (qkv, zba, conv, par, dy) = (
            f32s(&x.qkv),
            f32s(&x.zba),
            f32s(&x.conv),
            f32s(&x.par),
            f32s(&x.dy),
        );
        let shape = DeltaBwdShape {
            n_ctx,
            chunks,
            c: x.c,
            kd: x.kd,
            v: x.v,
            nv,
            nk: nv,
            eps: x.eps as f32,
        };
        let bwd = unsafe { DeltaBwd::new(&g, shape) };
        // The forward arm: the same input buffers, `delta_scan` over the same geometry.
        let out = unsafe { g.dev(x.chunks * x.n_ctx * x.v * 4) };
        let meta = [
            n_ctx as u32,
            x.c as u32,
            x.kd as u32,
            x.v as u32,
            nv as u32,
            (x.eps as f32).to_bits(),
            nv as u32,
        ];
        let fstep = unsafe {
            shader(
                &g,
                DELTA_SCAN_SPV,
                &meta,
                &[bwd.qkv, bwd.zba, bwd.conv, bwd.par, out],
                ((chunks * nv) as u32, 1),
                "delta_scan",
            )
        };
        let vkk = g.vk;
        let fcb = unsafe { vkk.command_buffer() };
        unsafe {
            vkk.begin(fcb);
            rec_step(vkk, fcb, &fstep);
            check((vkk.end_cb)(fcb), "vkEndCommandBuffer");
        }
        // Warm both, then pair: no transfer is timed on either arm.
        let _ = unsafe { bwd.run(&g, &qkv, &zba, &conv, &par, &dy) };
        unsafe { g.vk.submit_wait(fcb) };
        let (mut fw, mut bw) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            let t0 = Instant::now();
            unsafe { g.vk.submit_wait(fcb) };
            fw.push(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            unsafe { bwd.submit(&g) };
            bw.push(t1.elapsed().as_secs_f64());
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let spread = |v: &[f64]| {
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            for &p in v {
                lo = lo.min(p);
                hi = hi.max(p);
            }
            (lo, hi)
        };
        let (fl, fh) = spread(&fw);
        let (bl, bh) = spread(&bw);
        println!(
            "TB T={} fwd {:.4} ms [{fl:.4}, {fh:.4}] bwd {:.4} ms [{bl:.4}, {bh:.4}] x{:.2} (n=5)",
            n_ctx * chunks,
            mean(&fw) * 1e3,
            mean(&bw) * 1e3,
            mean(&bw) / mean(&fw)
        );
        assert!(
            bw.iter().all(|&p| p.is_finite()) && mean(&bw) > 0.0,
            "bwd wall"
        );
    }
}
