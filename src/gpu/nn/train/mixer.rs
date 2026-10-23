//! One Qwen3.5 trunk layer forward on the device in f32, over whole chunks of `n_ctx` columns that
//! each start at position 0: the twin of the CPU [`Engine::layer_tap_cols`] for the two stage rows a
//! block fit reads, the mixer output (`TAP_ATTN_OUT`, the o-projection input) and the layer output.
//!
//! `n1 = rmsnorm(x, g1)`; the mixer's input projections of `n1`; the mixer
//! ([`DELTA_SCAN_SPV`] for a Gated `DeltaNet` layer, [`ATTN_MAP_SPV`] for a gated softmax layer) to
//! `ao`; `h = x + ao·Woᵀ`; `n2 = rmsnorm(h, g2)`; `y = h + (silu(n2·Wgᵀ) ⊙ n2·Wuᵀ)·Wdᵀ`.
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

use super::block::{download, host_buffer, upload, HostBuffer};
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
    /// Gated softmax attention: `nh` query heads, `nkv` key heads, `hd` wide, `rot` rotated dims at
    /// base `theta`; `gains` = `[q gains (hd) | k gains (hd)]`.
    Softmax {
        nh: usize,
        nkv: usize,
        hd: usize,
        rot: usize,
        theta: f32,
        gains: Vec<f32>,
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
                if !ec.attn_gate {
                    return Err(ModelError::Shape(format!("blk.{l}: ungated attention")));
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
                ..
            } => {
                assert!(*hd <= 256 && nh % nkv == 0, "attn_map geometry");
                let gn = extra[0];
                let (q, gate, k) = (dev(t * nh * hd), dev(t * nh * hd), dev(t * nkv * hd));
                let bufs = [pb[0], pb[1], gn, q, gate, k, pb[2], ao];
                for (mode, wg, name) in [
                    (0u32, t * (nh + nkv), "attn_prep"),
                    (1u32, chunks * nh * n_ctx, "attn_attend"),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::nn::GLSLC_ARGS;
    use std::path::Path;

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
    }
}
