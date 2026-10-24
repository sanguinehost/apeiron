//! The attention front of one softmax layer of the block fit on the device: `q`, `k` and `v` held
//! as binary factors `W = diag(s_post)·sign(U)·sign(V)ᵀ·diag(s_pre)` and trained by Adam through
//! the straight-through sign against the o-projection's input adjoint.
//!
//! This is the part of a block's training graph that [`BlockTune`](super::block::BlockTune) — the
//! o-projection and the `SwiGLU` FFN — does not carry. The reference tunes every linear of a block
//! in one sequential loop, `tune_nonfact` then `tune_fact` (`SamsungLabs/NanoQuant` @ `a9e0a43`,
//! `src/nanoquant/core/compress_block.py:52-98` and `:160-201`), over the slot order
//! `q, v, o, k, gate, up, down` (`src/nanoquant/utils/utils.py:77-88`). This tape is that order's
//! `q`, `k`, `v` slots; the factor form is `src/nanoquant/modules/linear.py:248-257`
//! (`_compute_forward`), the same one `slot::Factor` materialises.
//!
//! The tape over `T = chunks·n_ctx` columns:
//! `n1 = rmsnorm(x, g1)`; `pq = n1·Wqᵀ`, `pk = n1·Wkᵀ`, `pv = n1·Wvᵀ`; then `attn_map`'s modes 4
//! and 5 (ungated, bit 2 of the `mode` word, `src/gpu/spirv/train/attn_map.comp:41`) over
//! `[pq, pk, gn, q, gate, k, pv, ao]` — the per-head `q`/`k` RMS norm with gains, the RoPE and the
//! causal `GQA` softmax, the same two dispatches a `Mixer::Softmax { gated: false }` layer records
//! (`src/gpu/nn/train/mixer.rs:381-417`) — giving `ao` `[T][nh·hd]`.
//!
//! `L = Σ ao ⊙ R` for the caller's fixed `R`, so the output cotangent is `dao = R` and
//! [`FrontTune::backward`] takes it as its input: the three-dispatch ungated backward
//! (`attn_bwd::tape`) gives `dqf`, `dkf`, `dv`; then `dWq = dqfᵀ·n1`, `dWk = dkfᵀ·n1`,
//! `dWv = dvᵀ·n1` and each slot's factor adjoint and Adam step (`slot::factor_bwd`,
//! `slot::adam_factor`). The input adjoint is `dn1 = dqf·Wq + dkf·Wk + dv·Wv`, then
//! `dx = rmsnorm_bwd(x, g1, dn1) + dres`, where `dres` is the caller's residual adjoint — the
//! block's `dh` — held in [`FrontTune::dres`] for a device copy. [`FrontTune::dx`] exposes the
//! result, so two layers chain on the device without a host round trip
//! ([`chain`](super::chain)).
//!
//! # Evidence grade
//!
//! `tested` by `front_tune_matches_host` in this file: the f64 host reference against central
//! differences, the device factor gradients against that reference, and a short Adam run; and by
//! `chain_dx_matches_fd` in `chain.rs`: the chained input adjoint of two layers against central
//! differences of the chained loss.
//!
//! # Trust boundary
//!
//! Not covered: the GPU driver, the SPIR-V compiler, the device's f32 transcendental precision,
//! and the correctness of the caller's `dres` (the block's `dh`), which the caller supplies.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "widths, rows and workgroup counts index u32 invocations; Q32.32 words map to f32"
)]

use crate::gpu::decode::{rec_step, GpuCtx, Step};
use crate::gpu::nn::fwd::{linear_fwd_dispatch, rmsnorm_fwd_dispatch};
use crate::gpu::nn::{
    adam_scheduled_bias, add_inplace_bytes, linear_bwd_input_dispatch, linear_bwd_weight_dispatch,
    rmsnorm_bwd_dispatch, AdamHyper,
};
use crate::gpu::vulkan::*;

use super::attn_bwd::{tape, AttnBwdShape};
use super::block::{download, host_buffer, stream_buffer, upload, HostBuffer, HostFactor};
use super::mixer::ATTN_MAP_SPV;
use super::slot::{q32, record_factor_swap, upload_dense, upload_factor, Factor};

const MAX_GROUPS_X: usize = 65_535;

/// The attention front's geometry: `chunks` independent sequences of `n_ctx` columns of residual
/// width `d`, `nh` query heads and `nkv` key/value heads `hd` wide, `rot` rotated dims at base
/// `theta`, RMS epsilon `eps`, and the ranks of the `q`, `k`, `v` factors in that order (the
/// reference's `calculate_ranks` gives `q` a wider factor than `k`/`v`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontShape {
    pub chunks: usize,
    pub n_ctx: usize,
    pub d: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub rot: usize,
    pub theta: f32,
    pub eps: f32,
    pub ranks: [usize; 3],
}

impl FrontShape {
    /// The tape's column count.
    #[must_use]
    pub fn t(&self) -> usize {
        self.chunks * self.n_ctx
    }

    /// The query width `nh·hd`.
    #[must_use]
    pub fn qw(&self) -> usize {
        self.nh * self.hd
    }

    /// The key/value width `nkv·hd`.
    #[must_use]
    pub fn kw(&self) -> usize {
        self.nkv * self.hd
    }

    /// The backward's geometry.
    #[must_use]
    pub fn attn(&self) -> AttnBwdShape {
        AttnBwdShape {
            n_ctx: self.n_ctx,
            chunks: self.chunks,
            nh: self.nh,
            nkv: self.nkv,
            hd: self.hd,
            rot: self.rot,
            theta: self.theta,
            eps: self.eps,
        }
    }

    /// `(out, inn)` of slot `k` in the order `[q, k, v]`: every projection reads `n1` (`inn = d`)
    /// and writes its own rows.
    #[must_use]
    pub fn dims(&self, k: usize) -> (usize, usize) {
        if k == 0 {
            (self.qw(), self.d)
        } else {
            (self.kw(), self.d)
        }
    }
}

/// A map of [`ATTN_MAP_SPV`] with the 9-word meta `mixer.rs` records, over `bufs`.
///
/// # Safety
///
/// The buffers hold what the mode reads and writes for `s` and `mode`.
///
/// # Complexity
///
/// `O(1)` host work; the dispatch covers `wg` invocations.
unsafe fn attn_step(
    g: &GpuCtx,
    s: &FrontShape,
    mode: u32,
    wg: usize,
    bufs: [Handle; 8],
    name: &'static str,
) -> Step {
    let meta = [
        mode,
        s.n_ctx as u32,
        s.nh as u32,
        s.nkv as u32,
        s.hd as u32,
        s.rot as u32,
        s.theta.to_bits(),
        s.eps.to_bits(),
        wg as u32,
    ];
    let mut all = vec![g.up_u32(&meta)];
    all.extend_from_slice(&bufs);
    let gx = wg.clamp(1, MAX_GROUPS_X);
    let gy = wg.div_ceil(gx).max(1);
    crate::gpu::decode::make_step_named(g, ATTN_MAP_SPV, &all, gx as u32, gy as u32, name)
}

/// The three recorded tapes: the evaluation pass, the same pass with the backward, and one training
/// step (backward plus Adam).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tape {
    Fwd,
    Bwd,
    Train,
}

/// The device attention front: `q`, `k`, `v` as factors and the tapes that train them.
#[allow(
    dead_code,
    reason = "the handles keep the tape's buffers alive; the tape is the only reader"
)]
pub struct FrontTune {
    shape: FrontShape,
    attn: AttnBwdShape,
    /// The three slots in the order `[q, k, v]`.
    factors: [Factor; 3],
    /// Each slot's Adam moments, `[param][m, v]` in the order `[u, vt, s_post, s_pre]`.
    moments: [[[Handle; 2]; 4]; 3],
    /// The block input `T × d` and the two norm gains: `g1` (`d`, the front's norm) and `gn`
    /// (`2·hd`, the per-head q then k gains).
    x: Handle,
    g1: Handle,
    gn: Handle,
    /// `n1 = rmsnorm(x, g1)`, the projections' input.
    n1: Handle,
    /// The projections' outputs, the `attn_map` prep's `qf`, `kf` and `v`.
    pq: Handle,
    pk: Handle,
    pv: Handle,
    /// The prep's outputs: the rotated queries, the gate word (ungated: never read or written) and
    /// the rotated keys.
    q: Handle,
    gate: Handle,
    k: Handle,
    /// The output `[T][nh·hd]` and the caller's output cotangent of the same extent.
    ao: Handle,
    dao: Handle,
    /// The backward's outputs: `dqf` (`[T][nh·hd]` ungated), `dkf` and `dv` (`[T][nkv·hd]`).
    dqf: Handle,
    dkf: Handle,
    dv: Handle,
    /// The backward's row statistics and its recompute scratch.
    lse: Handle,
    dsum: Handle,
    da: Handle,
    dq_rot: Handle,
    dk_rot: Handle,
    amax: Handle,
    /// The input adjoint's buffers: `dn1`'s three `[T][d]` accumulators (`q`, `k`, `v`), the
    /// result `dx`, the caller's residual adjoint `dres` (zeroed at construction) and the norm's
    /// thread-local part of `dγ` (dropped: `g1` is frozen).
    dn1: [Handle; 3],
    dx: Handle,
    dres: Handle,
    dgr: Handle,
    /// The materialise tape of each slot: the two signs, `M`, then the scales. Recorded only for a
    /// slot whose `W` is held as a factor.
    materialise: [Vec<Step>; 3],
    forward: Vec<Step>,
    /// The three backward dispatches, then the three dense weight adjoints into each slot's `dw`,
    /// then the input adjoint into `dx`.
    backward: Vec<Step>,
    factor_bwd: [Vec<Step>; 3],
    adam_fact: [Vec<Step>; 3],
    /// Each slot's dense moment pair over `out × inn`, and the Adam step over its dense `W`.
    dense_moments: [[Handle; 2]; 3],
    adam_dense: [Step; 3],
    /// The stage: `factor[k]` holds slot `k`'s `W` in its latents (`true`) or in its `w` buffer
    /// (`false`); `train[k]` puts slot `k` in the trained set.
    factor: [bool; 3],
    train: [bool; 3],
    /// The two base learning rates `[latent, scale]` in real units, the optimiser constants their
    /// steps bake, and the per-step schedule multiplier `c = lr_t/lr_base` (empty: constant rate).
    lr: [f64; 2],
    hyper: AdamHyper,
    sched: Vec<f64>,
    /// Host → device staging, as large as the largest buffer; host ← device readback.
    stage: HostBuffer,
    rb: HostBuffer,
    bias_stage: HostBuffer,
    bias_buf: Handle,
    step_no: u32,
    cb_fwd: CommandBuffer,
    cb_bwd: CommandBuffer,
    cb_train: CommandBuffer,
    xfer: CommandBuffer,
}

impl FrontTune {
    /// Allocate every buffer and build every pipeline for `shape`, with Adam learning rate `lr`
    /// for the two latents and `lr_scale` for the two scale vectors (real units).
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value.
    ///
    /// # Complexity
    ///
    /// `O(1)` pipelines and `O(T·(d + nh·hd + nkv·hd) + Σ out·inn)` device words.
    #[allow(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "one tape; pq/pk/pv and dqf/dkf/dv are the reference's own slot names"
    )]
    #[must_use]
    pub unsafe fn new(g: &GpuCtx, shape: FrontShape, lr: f64, lr_scale: f64) -> Self {
        assert!(
            shape.hd <= 256
                && shape.nh % shape.nkv == 0
                && shape.n_ctx <= crate::gpu::nn::train::mixer::MAX_N_CTX,
            "attn_map geometry"
        );
        assert!(
            shape.rot == 0 || shape.rot % 2 == 0,
            "rope pairs must be even"
        );
        let attn = shape.attn();
        let (t, d, hd) = (shape.t(), shape.d, shape.hd);
        let (qw, kw) = (shape.qw(), shape.kw());
        let dev = |n: usize| g.dev(n * 4);
        // The buffers a chained layer copies into or out of (`chain.rs`) carry both transfer bits.
        let stream = |n: usize| stream_buffer(g, n);
        let (fq, mq) = Factor::new(g, qw, d, shape.ranks[0]);
        let (fk, mk) = Factor::new(g, kw, d, shape.ranks[1]);
        let (fv, mv) = Factor::new(g, kw, d, shape.ranks[2]);
        let factors = [fq, fk, fv];
        let moments = [mq, mk, mv];
        let (x, g1, gn, n1) = (stream(t * d), dev(d), dev(2 * hd), dev(t * d));
        let (pq, pk, pv) = (dev(t * qw), dev(t * kw), dev(t * kw));
        // The gate is read and written only when gated; ungated, one word keeps binding 5 valid
        // without a `T·nh·hd` buffer the shaders never touch.
        let (q, gate, k) = (dev(t * qw), dev(1), dev(t * kw));
        let (ao, dao) = (stream(t * qw), stream(t * qw));
        let (dqf, dkf, dv) = (dev(t * qw), dev(t * kw), dev(t * kw));
        let (lse, dsum) = (dev(t * shape.nh), dev(t * shape.nh));
        let (da, dq_rot, dk_rot) = (dev(t * qw), dev(t * qw), dev(t * kw));
        let amax = dev(super::attn_bwd::amax_len(&attn));
        // Three buffers, not one: `[dev(n); 3]` evaluates the allocation once and aliases the
        // three accumulators.
        let dn1 = [dev(t * d), dev(t * d), dev(t * d)];
        let (dx, dres, dgr) = (stream(t * d), g.dev_zeroed(t * d * 4), dev(t * d));
        let (tu, du) = (t as u32, d as u32);
        let (qu, ku) = (qw as u32, kw as u32);
        let forward = vec![
            rmsnorm_fwd_dispatch(g, x, g1, n1, tu, du),
            linear_fwd_dispatch(g, n1, factors[0].w, pq, tu, du, qu),
            linear_fwd_dispatch(g, n1, factors[1].w, pk, tu, du, ku),
            linear_fwd_dispatch(g, n1, factors[2].w, pv, tu, du, ku),
            attn_step(
                g,
                &shape,
                4,
                t * (shape.nh + shape.nkv),
                [pq, pk, gn, q, gate, k, pv, ao],
                "attn_prep",
            ),
            attn_step(
                g,
                &shape,
                5,
                shape.chunks * shape.nh * shape.n_ctx,
                [pq, pk, gn, q, gate, k, pv, ao],
                "attn_attend",
            ),
        ];
        let mut backward = tape(
            g,
            attn,
            false,
            [
                pq, pk, gn, q, gate, k, pv, dao, lse, dsum, da, dq_rot, dk_rot, dqf, dkf, dv, amax,
            ],
        );
        backward.push(linear_bwd_weight_dispatch(
            g,
            dqf,
            n1,
            factors[0].dw,
            tu,
            du,
            qu,
        ));
        backward.push(linear_bwd_weight_dispatch(
            g,
            dkf,
            n1,
            factors[1].dw,
            tu,
            du,
            ku,
        ));
        backward.push(linear_bwd_weight_dispatch(
            g,
            dv,
            n1,
            factors[2].dw,
            tu,
            du,
            ku,
        ));
        let add = |dst: Handle, src: Handle| {
            crate::gpu::decode::make_step_named(
                g,
                &add_inplace_bytes((t * d) as u32, 256),
                &[dst, src],
                ((t * d) as u32).div_ceil(256),
                1,
                "add_inplace",
            )
        };
        // The input adjoint: `dn1 = dqf·Wq + dkf·Wk + dv·Wv` (the block's `da = dh·Wo` shape per
        // projection), then `dx = rmsnorm_bwd(x, g1, dn1)`, then `dx += dres`.
        backward.push(linear_bwd_input_dispatch(
            g,
            dqf,
            factors[0].w,
            dn1[0],
            tu,
            du,
            qu,
        ));
        backward.push(linear_bwd_input_dispatch(
            g,
            dkf,
            factors[1].w,
            dn1[1],
            tu,
            du,
            ku,
        ));
        backward.push(linear_bwd_input_dispatch(
            g,
            dv,
            factors[2].w,
            dn1[2],
            tu,
            du,
            ku,
        ));
        backward.push(add(dn1[0], dn1[1]));
        backward.push(add(dn1[0], dn1[2]));
        backward.push(rmsnorm_bwd_dispatch(
            g, x, g1, dn1[0], dx, dgr, tu, du, 1e-6,
        ));
        backward.push(add(dx, dres));
        let bias_stage = host_buffer(g, 2, BUFFER_USAGE_TRANSFER_SRC);
        let bias_buf = dev(2);
        // One tape per slot, built by name so the field names below shadow nothing.
        let [f0, f1, f2] = &factors;
        let [m0, m1, m2] = &moments;
        let materialise = [
            super::slot::materialise(g, f0),
            super::slot::materialise(g, f1),
            super::slot::materialise(g, f2),
        ];
        let factor_bwd = [
            super::slot::factor_bwd(g, f0),
            super::slot::factor_bwd(g, f1),
            super::slot::factor_bwd(g, f2),
        ];
        let adam_fact = [
            super::slot::adam_factor(
                g,
                f0,
                m0,
                bias_buf,
                q32(lr),
                q32(lr_scale),
                AdamHyper::ORACLE,
            ),
            super::slot::adam_factor(
                g,
                f1,
                m1,
                bias_buf,
                q32(lr),
                q32(lr_scale),
                AdamHyper::ORACLE,
            ),
            super::slot::adam_factor(
                g,
                f2,
                m2,
                bias_buf,
                q32(lr),
                q32(lr_scale),
                AdamHyper::ORACLE,
            ),
        ];
        let dense_moments = [
            super::slot::dense_moments(g, f0),
            super::slot::dense_moments(g, f1),
            super::slot::dense_moments(g, f2),
        ];
        let adam_dense = std::array::from_fn(|k| {
            super::slot::adam_dense(
                g,
                &factors[k],
                &dense_moments[k],
                bias_buf,
                q32(lr),
                AdamHyper::ORACLE,
            )
        });
        let big = [t * d, t * qw, t * kw, d, 2 * hd]
            .into_iter()
            .chain(factors.iter().map(|fa| {
                (fa.out * fa.rank)
                    .max(fa.inn * fa.rank)
                    .max(fa.out * fa.inn)
            }))
            .max()
            .unwrap_or(1);
        let s = Self {
            shape,
            attn,
            factors,
            moments,
            x,
            g1,
            gn,
            n1,
            pq,
            pk,
            pv,
            q,
            gate,
            k,
            ao,
            dao,
            dqf,
            dkf,
            dv,
            lse,
            dsum,
            da,
            dq_rot,
            dk_rot,
            amax,
            dn1,
            dx,
            dres,
            dgr,
            materialise,
            forward,
            backward,
            factor_bwd,
            adam_fact,
            dense_moments,
            adam_dense,
            factor: [true; 3],
            train: [true; 3],
            lr: [lr, lr_scale],
            hyper: AdamHyper::ORACLE,
            sched: Vec::new(),
            stage: host_buffer(g, big, BUFFER_USAGE_TRANSFER_SRC),
            rb: host_buffer(g, big, BUFFER_USAGE_TRANSFER_DST),
            bias_stage,
            bias_buf,
            step_no: 0,
            cb_fwd: g.vk.reusable_command_buffer(),
            cb_bwd: g.vk.reusable_command_buffer(),
            cb_train: g.vk.reusable_command_buffer(),
            xfer: g.vk.reusable_command_buffer(),
        };
        for t in [Tape::Fwd, Tape::Bwd, Tape::Train] {
            s.record(g, t);
        }
        s
    }

    /// Record one of the three tapes; [`FrontTune::run`] resubmits it. The buffer is reset first.
    ///
    /// # Safety
    ///
    /// Called from [`FrontTune::new`]; no tape of this value is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; the recorded tape carries `O(1)` dispatches.
    unsafe fn record(&self, g: &GpuCtx, tape: Tape) {
        let vk = g.vk;
        let cb = match tape {
            Tape::Fwd => self.cb_fwd,
            Tape::Bwd => self.cb_bwd,
            Tape::Train => self.cb_train,
        };
        vk.reset_cb(cb);
        vk.begin(cb);
        if tape == Tape::Train {
            let region = BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: 8,
            };
            (vk.copy_buffer)(cb, self.bias_stage.buf, self.bias_buf, 1, &region);
            vk.barrier(
                cb,
                STAGE_TRANSFER,
                STAGE_COMPUTE,
                ACCESS_TRANSFER_WRITE,
                ACCESS_SHADER_READ,
            );
        }
        for (k, steps) in self.materialise.iter().enumerate() {
            if !self.factor[k] {
                continue;
            }
            for s in steps {
                rec_step(vk, cb, s);
            }
        }
        for s in &self.forward {
            rec_step(vk, cb, s);
        }
        if tape != Tape::Fwd {
            for s in &self.backward {
                rec_step(vk, cb, s);
            }
            for k in 0..3 {
                if self.factor[k] {
                    for s in &self.factor_bwd[k] {
                        rec_step(vk, cb, s);
                    }
                }
                if tape == Tape::Train && self.train[k] {
                    if self.factor[k] {
                        for s in &self.adam_fact[k] {
                            rec_step(vk, cb, s);
                        }
                    } else {
                        rec_step(vk, cb, &self.adam_dense[k]);
                    }
                }
            }
        }
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
    }

    /// Copy `data` into the device buffer `dst` through the staging buffer.
    ///
    /// # Safety
    ///
    /// `dst` holds at least `data.len()` f32.
    unsafe fn put(&self, g: &GpuCtx, dst: Handle, data: &[f32]) {
        upload(g, self.xfer, &self.stage, dst, data);
    }

    /// Read `n` f32 of the device buffer `src`.
    ///
    /// # Safety
    ///
    /// `src` holds at least `n` f32.
    unsafe fn get(&self, g: &GpuCtx, src: Handle, n: usize) -> Vec<f32> {
        download(g, self.xfer, &self.rb, src, n)
    }

    /// Load the three factors (`[q, k, v]`) and the two norm gains, zero every moment and restart
    /// the step count.
    ///
    /// # Safety
    ///
    /// The factors match the shape's dimensions and rank (asserted); no tape is pending.
    ///
    /// # Complexity
    ///
    /// `O(Σ (out + inn)·rank + d)` words copied.
    pub unsafe fn load(&mut self, g: &GpuCtx, factors: &[HostFactor], g1: &[f32], gn: &[f32]) {
        assert_eq!(factors.len(), 3, "front load: three factors");
        assert_eq!(
            (g1.len(), gn.len()),
            (self.shape.d, 2 * self.shape.hd),
            "front load: gains"
        );
        for (k, hf) in factors.iter().enumerate() {
            self.load_factor(g, k, hf);
        }
        self.put(g, self.g1, g1);
        self.put(g, self.gn, gn);
        self.step_no = 0;
    }

    /// Record into `cb` the device form of [`FrontTune::load`]: the three factors and the gains
    /// `g1`, `gn` copied from `store` at word `at` onward, every moment zeroed, the step count
    /// restarted. Returns the word after the front's image.
    ///
    /// # Safety
    ///
    /// `store` holds the image from `at`; `cb` is recording and no tape is pending.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; `O(Σ (out + inn)·(rank + 1) + d)` words moved on the device.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "word offsets index a u64 byte offset"
    )]
    pub(super) unsafe fn record_swap(
        &mut self,
        vk: &Vk,
        cb: CommandBuffer,
        store: Handle,
        at: usize,
    ) -> usize {
        let mut at = at;
        for k in 0..3 {
            at = record_factor_swap(vk, cb, store, at, &self.factors[k], &self.moments[k]);
        }
        for (dst, n) in [(self.g1, self.shape.d), (self.gn, 2 * self.shape.hd)] {
            let region = BufferCopy {
                src_offset: (at * 4) as u64,
                dst_offset: 0,
                size: (n * 4) as u64,
            };
            (vk.copy_buffer)(cb, store, dst, 1, &region);
            at += n;
        }
        self.step_no = 0;
        at
    }

    /// Put slot `k`'s factor and zero its four moment pairs.
    ///
    /// # Safety
    ///
    /// `hf` matches slot `k`'s dimensions and rank (asserted); no tape is pending.
    ///
    /// # Complexity
    ///
    /// `O((out + inn)·rank + out + inn)` words copied.
    pub unsafe fn load_factor(&mut self, g: &GpuCtx, k: usize, hf: &HostFactor) {
        let fa = &self.factors[k];
        assert_eq!(
            (fa.out, fa.inn, fa.rank),
            (hf.out, hf.inn, hf.rank),
            "front load_factor {k}: shape"
        );
        upload_factor(g, self.xfer, &self.stage, fa, hf, &self.moments[k]);
    }

    /// Put slot `k`'s dense `W` (`out × inn`, row-major) into its `w` buffer and zero that slot's
    /// dense moments. Leaves the stage as it is.
    ///
    /// The reference's still-dense `nn.Linear`s are trained in place (`tune_nonfact`,
    /// `compress_block.py:58-62`), before their own factorisation replaces them.
    ///
    /// # Safety
    ///
    /// `w` has slot `k`'s `out·inn` words (asserted); no tape is pending.
    ///
    /// # Complexity
    ///
    /// `O(out·inn)` words copied, three times.
    pub unsafe fn load_dense(&mut self, g: &GpuCtx, k: usize, w: &[f32]) {
        let fa = &self.factors[k];
        assert_eq!(w.len(), fa.out * fa.inn, "front load_dense {k}: shape");
        upload_dense(g, self.xfer, &self.stage, fa, w, &self.dense_moments[k]);
    }

    /// Set the stage: `factor[k]` places slot `k`'s `W` in its latents (`true`) or in its `w`
    /// buffer (`false`), and `train[k]` puts slot `k` in the trained set — it updates its factor's
    /// latents and scales when `factor[k]`, or its dense `W` when not. A slot outside the set keeps
    /// its parameters and moments bit-identical. Re-records all three tapes.
    ///
    /// # Safety
    ///
    /// No tape is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; three tapes re-recorded.
    pub unsafe fn set_stage(&mut self, g: &GpuCtx, factor: [bool; 3], train: [bool; 3]) {
        self.factor = factor;
        self.train = train;
        for t in [Tape::Fwd, Tape::Bwd, Tape::Train] {
            self.record(g, t);
        }
    }

    /// Re-record the evaluation tape with the current stage flags — the profiler's hook
    /// ([`crate::gpu::decode::prof_arm`]). Armed, `rec_step` writes a timestamp pair around each
    /// dispatch of this tape; every other tape keeps its unstamped recording.
    ///
    /// # Safety
    ///
    /// No tape of this value is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; the evaluation tape carries `O(1)` dispatches.
    pub unsafe fn rerecord_fwd(&self, g: &GpuCtx) {
        self.record(g, Tape::Fwd);
    }

    /// As [`FrontTune::rerecord_fwd`], for the backward tape.
    ///
    /// # Safety
    ///
    /// No tape of this value is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; the backward tape carries `O(1)` dispatches.
    pub unsafe fn rerecord_bwd(&self, g: &GpuCtx) {
        self.record(g, Tape::Bwd);
    }

    /// Set the optimiser constants the Adam steps bake; the moments are kept and the three tapes
    /// are re-recorded. [`AdamHyper::OPTIMI`] is the NanoQuant reference's set.
    ///
    /// # Safety
    ///
    /// No tape is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; `O(1)` pipelines rebuilt.
    pub unsafe fn set_hyper(&mut self, g: &GpuCtx, h: AdamHyper) {
        self.hyper = h;
        let (lr, ls) = (q32(self.lr[0]), q32(self.lr[1]));
        let [f0, f1, f2] = &self.factors;
        self.adam_fact = [
            super::slot::adam_factor(g, f0, &self.moments[0], self.bias_buf, lr, ls, h),
            super::slot::adam_factor(g, f1, &self.moments[1], self.bias_buf, lr, ls, h),
            super::slot::adam_factor(g, f2, &self.moments[2], self.bias_buf, lr, ls, h),
        ];
        let mut dense = std::array::from_fn(|k| {
            super::slot::adam_dense(
                g,
                &self.factors[k],
                &self.dense_moments[k],
                self.bias_buf,
                lr,
                h,
            )
        });
        std::mem::swap(&mut self.adam_dense, &mut dense);
        for t in [Tape::Fwd, Tape::Bwd, Tape::Train] {
            self.record(g, t);
        }
    }

    /// Set the per-step schedule multiplier `c_t = lr_t/lr_base` every Adam step of a run shares:
    /// entry `t−1` is step `t`'s. An empty vector (the default) keeps each group's base rate.
    ///
    /// # Complexity
    ///
    /// `O(steps)` words held; `O(1)` host work.
    pub fn set_schedule(&mut self, c: Vec<f64>) {
        self.sched = c;
    }

    /// Set the step's columns: the front's input `x` (`T × d`) and the output cotangent `dao`
    /// (`T × nh·hd`), both row-major.
    ///
    /// # Safety
    ///
    /// The slices have the shape's lengths (asserted); no tape is pending.
    ///
    /// # Complexity
    ///
    /// `O(T·(d + nh·hd))` words copied, each through one submission.
    pub unsafe fn batch(&mut self, g: &GpuCtx, x: &[f32], dao: &[f32]) {
        let (t, d, qw) = (self.shape.t(), self.shape.d, self.shape.qw());
        assert_eq!((x.len(), dao.len()), (t * d, t * qw), "front batch");
        self.put(g, self.x, x);
        self.put(g, self.dao, dao);
    }

    /// Resubmit `tape`; the training tape first advances the Adam bias corrections.
    ///
    /// # Safety
    ///
    /// [`FrontTune::load`] and [`FrontTune::batch`] have run; no tape is pending.
    ///
    /// # Complexity
    ///
    /// One submission of the tape's `O(T·n_ctx·nh·hd + T·d·(nh·hd + nkv·hd) + Σ out·inn·rank)`
    /// multiply-adds.
    unsafe fn run(&mut self, g: &GpuCtx, tape: Tape) {
        if tape == Tape::Train {
            self.step_no += 1;
            let t = self.step_no;
            let (b1t, rbc2) = match self.sched.get(t as usize - 1) {
                Some(&c) => adam_scheduled_bias(self.hyper, t, c),
                None => self.hyper.bias_corrections(t),
            };
            *self.bias_stage.ptr = b1t;
            *self.bias_stage.ptr.add(1) = rbc2;
        }
        let cb = match tape {
            Tape::Fwd => self.cb_fwd,
            Tape::Bwd => self.cb_bwd,
            Tape::Train => self.cb_train,
        };
        g.vk.submit_wait(cb);
    }

    /// The evaluation pass: materialise the factors, then the forward. Leaves `ao` holding the
    /// attention output.
    ///
    /// # Safety
    ///
    /// [`FrontTune::load`] and [`FrontTune::batch`] have run; no tape is pending.
    pub unsafe fn forward(&mut self, g: &GpuCtx) {
        self.run(g, Tape::Fwd);
    }

    /// The backward from the output cotangent `dao`: materialise, forward, the three backward
    /// dispatches and the three dense weight adjoints, leaving every slot's factor gradients in
    /// its `grad` buffers and touching no parameter.
    ///
    /// # Safety
    ///
    /// `dao` is `T·nh·hd` words (asserted); no tape is pending.
    pub unsafe fn backward(&mut self, g: &GpuCtx, dao: &[f32]) {
        assert_eq!(
            dao.len(),
            self.shape.t() * self.shape.qw(),
            "front backward"
        );
        self.put(g, self.dao, dao);
        self.run(g, Tape::Bwd);
    }

    /// One training step from the `dao` [`FrontTune::batch`] uploaded: the backward, then Adam over
    /// the three slots' latents and scales.
    ///
    /// # Safety
    ///
    /// [`FrontTune::load`] and [`FrontTune::batch`] have run; no tape is pending.
    pub unsafe fn train(&mut self, g: &GpuCtx) {
        self.run(g, Tape::Train);
    }

    /// The attention output `ao` (`T × nh·hd`) buffer, without a download.
    #[must_use]
    pub fn ao(&self) -> Handle {
        self.ao
    }

    /// The front's input `x` (`T × d`) and the output cotangent `dao` (`T × nh·hd`) the tapes read,
    /// for a caller that fills them by a device copy rather than through [`FrontTune::batch`].
    #[must_use]
    pub fn cols(&self) -> (Handle, Handle) {
        (self.x, self.dao)
    }

    /// Slot `k`'s materialised weight, `out × inn`: the dense `W` while `factor[k]` is false and the
    /// materialised factor after a pass, the same buffer [`BlockTune::weight`] reads.
    ///
    /// # Safety
    ///
    /// A pass has completed.
    ///
    /// # Complexity
    ///
    /// `O(out·inn)` words downloaded.
    #[must_use]
    pub unsafe fn weight(&self, g: &GpuCtx, k: usize) -> Vec<f32> {
        let fa = &self.factors[k];
        self.get(g, fa.w, fa.out * fa.inn)
    }

    /// The output cotangent `dao` (`T × nh·hd`) buffer the backward reads.
    #[must_use]
    pub fn dao(&self) -> Handle {
        self.dao
    }

    /// The front's input adjoint `dx` (`T × d`), valid after a backward: `rmsnorm_bwd(x, g1, dn1)`
    /// plus [`FrontTune::dres`]. It is the layer input's cotangent, so it is the previous layer's
    /// `dy` in a chain ([`chain`](super::chain)).
    #[must_use]
    pub fn dx(&self) -> Handle {
        self.dx
    }

    /// The residual adjoint `dres` (`T × d`) the backward adds to `dx`: the block's `dh`, which the
    /// caller copies in by a device copy ([`chain`](super::chain)). Zeroed at construction.
    #[must_use]
    pub fn dres(&self) -> Handle {
        self.dres
    }

    /// Run the backward tape alone, over the `dao` and `dres` buffers a caller filled by device
    /// copies, leaving `dx` holding the input adjoint.
    ///
    /// # Safety
    ///
    /// [`FrontTune::load`] has run and no tape is pending.
    ///
    /// # Complexity
    ///
    /// One submission of the backward's `O(T·n_ctx·nh·hd + T·d·(nh·hd + nkv·hd) + Σ out·inn·rank)`
    /// multiply-adds.
    pub unsafe fn backward_cols(&mut self, g: &GpuCtx) {
        self.run(g, Tape::Bwd);
    }

    /// The attention output `ao` of the last pass, `T × nh·hd`.
    ///
    /// # Safety
    ///
    /// A pass has completed.
    #[must_use]
    pub unsafe fn output(&self, g: &GpuCtx) -> Vec<f32> {
        self.get(g, self.ao, self.shape.t() * self.shape.qw())
    }

    /// The three factors' current latents and scales.
    ///
    /// # Safety
    ///
    /// The device is idle.
    #[must_use]
    pub unsafe fn factors(&self, g: &GpuCtx) -> Vec<HostFactor> {
        self.factors
            .iter()
            .map(|fa| HostFactor {
                out: fa.out,
                inn: fa.inn,
                rank: fa.rank,
                u: self.get(g, fa.p[0], fa.out * fa.rank),
                vt: self.get(g, fa.p[1], fa.inn * fa.rank),
                s_post: self.get(g, fa.p[2], fa.out),
                s_pre: self.get(g, fa.p[3], fa.inn),
            })
            .collect()
    }

    /// The two scale gradients of the last backward, `[slot][s_post, s_pre]`, `O(out + inn)` words
    /// each — the KD driver's read, which does not need the latent gradients next to them.
    ///
    /// # Safety
    ///
    /// [`FrontTune::backward_cols`] has completed and no submission is pending.
    ///
    /// # Complexity
    ///
    /// `O(Σ (out + inn))` words downloaded: 22 528 at Qwen3-0.6B's geometry, 90 KiB.
    pub(crate) unsafe fn scale_grads(&self, g: &GpuCtx) -> Vec<[Vec<f32>; 2]> {
        self.factors
            .iter()
            .map(|fa| {
                [
                    self.get(g, fa.grad[2], fa.out),
                    self.get(g, fa.grad[3], fa.inn),
                ]
            })
            .collect()
    }
}

#[cfg(test)]
impl FrontTune {
    /// The factor gradients of the last backward, `[slot][u, vt, s_post, s_pre]`.
    ///
    /// # Safety
    ///
    /// A backward has completed.
    pub(crate) unsafe fn grads(&self, g: &GpuCtx) -> Vec<[Vec<f32>; 4]> {
        self.factors
            .iter()
            .map(|fa| {
                let n = [fa.out * fa.rank, fa.inn * fa.rank, fa.out, fa.inn];
                [0, 1, 2, 3].map(|j| self.get(g, fa.grad[j], n[j]))
            })
            .collect()
    }

    /// The dense `W` of slot `k` (`out × inn`).
    ///
    /// # Safety
    ///
    /// A pass has completed.
    unsafe fn dense_w(&self, g: &GpuCtx, k: usize) -> Vec<f32> {
        let fa = &self.factors[k];
        self.get(g, fa.w, fa.out * fa.inn)
    }

    /// The dense `W` gradient of slot `k` from the last backward, `out × inn`.
    ///
    /// # Safety
    ///
    /// A backward has completed.
    pub(crate) unsafe fn dense_grad(&self, g: &GpuCtx, k: usize) -> Vec<f32> {
        let fa = &self.factors[k];
        self.get(g, fa.dw, fa.out * fa.inn)
    }

    /// Every parameter and moment word of slot `k` in its current form — the four factor buffers
    /// with their moment pairs, or the dense `W` with its pair — as bits.
    ///
    /// # Safety
    ///
    /// The device is idle.
    pub(super) unsafe fn slot_words(&self, g: &GpuCtx, k: usize) -> Vec<u32> {
        let fa = &self.factors[k];
        let n = [fa.out * fa.rank, fa.inn * fa.rank, fa.out, fa.inn];
        let mut src: Vec<(Handle, usize)> = Vec::new();
        if self.factor[k] {
            for j in 0..4 {
                src.push((fa.p[j], n[j]));
                src.push((self.moments[k][j][0], n[j]));
                src.push((self.moments[k][j][1], n[j]));
            }
        } else {
            src.push((fa.w, fa.out * fa.inn));
            src.push((self.dense_moments[k][0], fa.out * fa.inn));
            src.push((self.dense_moments[k][1], fa.out * fa.inn));
        }
        let mut words = Vec::new();
        for (h, len) in src {
            words.extend(self.get(g, h, len).iter().map(|v| v.to_bits()));
        }
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::nn::train::attn_bwd::tests::{ref_bwd_ungated, ref_fwd_ungated};

    /// A fixed-seed LCG value in `[-1, 1)` (Knuth's MMIX constants).
    fn unit(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 33) as f64 / f64::from(1u32 << 30) - 1.0) as f32
            })
            .collect()
    }

    fn to64(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x)).collect()
    }

    fn factor(out: usize, inn: usize, rank: usize, seed: u64) -> HostFactor {
        let pos = |v: Vec<f32>| v.into_iter().map(|x| 0.05 + 0.1 * x.abs()).collect();
        HostFactor {
            out,
            inn,
            rank,
            u: unit(out * rank, seed),
            vt: unit(inn * rank, seed + 1),
            s_post: pos(unit(out, seed + 2)),
            s_pre: pos(unit(inn, seed + 3)),
        }
    }

    fn sg(v: f64) -> f64 {
        if v < 0.0 {
            -1.0
        } else {
            1.0
        }
    }

    /// `M = sign(U)·sign(V)ᵀ` in f64, `out × inn`.
    fn signs(f: &HostFactor) -> Vec<f64> {
        let (o, i, r) = (f.out, f.inn, f.rank);
        let mut m = vec![0f64; o * i];
        for row in 0..o {
            for c in 0..i {
                m[row * i + c] = (0..r)
                    .map(|j| sg(f64::from(f.u[row * r + j])) * sg(f64::from(f.vt[c * r + j])))
                    .sum();
            }
        }
        m
    }

    /// `W = s_post ⊙ M ⊙ s_pre` in f64, `out × inn`.
    fn dense(f: &HostFactor) -> Vec<f64> {
        let m = signs(f);
        (0..f.out * f.inn)
            .map(|x| f64::from(f.s_post[x / f.inn]) * m[x] * f64::from(f.s_pre[x % f.inn]))
            .collect()
    }

    /// `y = x·wᵀ`: `x` is `n × i`, `w` is `o × i`.
    fn lin(x: &[f64], w: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
        let mut y = vec![0f64; n * o];
        for row in 0..n {
            for c in 0..o {
                y[row * o + c] = (0..i).map(|k| x[row * i + k] * w[c * i + k]).sum();
            }
        }
        y
    }

    /// `dw = dyᵀ·x`: `dy` is `n × o`, `x` is `n × i`, the result `o × i`.
    fn dwt(dy: &[f64], x: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
        let mut dw = vec![0f64; o * i];
        for r in 0..o {
            for c in 0..i {
                dw[r * i + c] = (0..n).map(|k| dy[k * o + r] * x[k * i + c]).sum();
            }
        }
        dw
    }

    /// `n1 = rmsnorm(x, g)` over `d` at the device forward's epsilon.
    fn rmsnorm(x: &[f64], g: &[f64], t: usize, d: usize) -> Vec<f64> {
        let mut n = vec![0f64; t * d];
        for row in 0..t {
            let ms: f64 = x[row * d..(row + 1) * d].iter().map(|v| v * v).sum::<f64>() / d as f64;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for c in 0..d {
                n[row * d + c] = x[row * d + c] * inv * g[c];
            }
        }
        n
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// The f64 host case of one front: the shape, the per-head norm gains, the output cotangent
    /// `R` and `n1` (which no factor moves).
    struct Host {
        sh: FrontShape,
        gn: Vec<f64>,
        r: Vec<f64>,
        n1: Vec<f64>,
    }

    impl Host {
        /// The projections' outputs `(pq, pk, pv)` from the factors' dense `W`.
        fn proj(&self, fs: &[HostFactor]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let (t, d) = (self.sh.t(), self.sh.d);
            let (qw, kw) = (self.sh.qw(), self.sh.kw());
            (
                lin(&self.n1, &dense(&fs[0]), t, d, qw),
                lin(&self.n1, &dense(&fs[1]), t, d, kw),
                lin(&self.n1, &dense(&fs[2]), t, d, kw),
            )
        }

        /// The forward alone, `ao`.
        fn ao(&self, fs: &[HostFactor]) -> Vec<f64> {
            let (pq, pk, pv) = self.proj(fs);
            ref_fwd_ungated(self.sh.attn(), &pq, &pk, &self.gn, &pv)
        }

        /// `L = Σ ao ⊙ R`.
        fn loss(&self, fs: &[HostFactor]) -> f64 {
            dot(&self.ao(fs), &self.r)
        }

        /// The three slots' dense `dW = dyᵀ·n1` in `[q, k, v]` order.
        fn dw(&self, fs: &[HostFactor]) -> [Vec<f64>; 3] {
            let (t, d) = (self.sh.t(), self.sh.d);
            let (qw, kw) = (self.sh.qw(), self.sh.kw());
            let (pq, pk, pv) = self.proj(fs);
            let (_, dqf, dkf, dv) =
                ref_bwd_ungated(self.sh.attn(), &pq, &pk, &self.gn, &pv, &self.r);
            [
                dwt(&dqf, &self.n1, t, d, qw),
                dwt(&dkf, &self.n1, t, d, kw),
                dwt(&dv, &self.n1, t, d, kw),
            ]
        }
    }

    /// The factor adjoint of `f` given its dense `dW`: `[u, vt, s_post, s_pre]`, the twin of
    /// `slot::factor_bwd`'s five dispatches (`src/gpu/spirv/train/block_map.comp` modes 1, 2, 3).
    fn factor_grads(f: &HostFactor, dw: &[f64]) -> [Vec<f64>; 4] {
        let (o, i, r) = (f.out, f.inn, f.rank);
        let m = signs(f);
        let mut dm = vec![0f64; o * i];
        let mut ds_post = vec![0f64; o];
        let mut ds_pre = vec![0f64; i];
        for row in 0..o {
            for c in 0..i {
                let x = row * i + c;
                let sp = f64::from(f.s_post[row]);
                let sq = f64::from(f.s_pre[c]);
                dm[x] = dw[x] * sp * sq;
                ds_post[row] += dw[x] * m[x] * sq;
                ds_pre[c] += dw[x] * m[x] * sp;
            }
        }
        let mut du = vec![0f64; o * r];
        let mut dvt = vec![0f64; i * r];
        for row in 0..o {
            for c in 0..i {
                for j in 0..r {
                    du[row * r + j] += dm[row * i + c] * sg(f64::from(f.vt[c * r + j]));
                    dvt[c * r + j] += dm[row * i + c] * sg(f64::from(f.u[row * r + j]));
                }
            }
        }
        [du, dvt, ds_post, ds_pre]
    }

    /// `max|got − want| / max|want|`.
    fn rel32(got: &[f32], want: &[f64]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 1e-30f64;
        for (&p, &q) in got.iter().zip(want) {
            num = num.max((f64::from(p) - q).abs());
            den = den.max(q.abs());
        }
        num / den
    }

    /// The attention front against the f64 host reference at a small shape: (a) the reference's
    /// scale gradients against central differences of its own loss, (b) the device factor
    /// gradients of `q`, `k`, `v` against the reference, (c) a short Adam run lowers
    /// `L = Σ ao ⊙ R`.
    #[test]
    #[allow(clippy::too_many_lines, reason = "one tape comparison")]
    fn front_tune_matches_host() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let sh = FrontShape {
            chunks: 2,
            n_ctx: 8,
            d: 32,
            nh: 4,
            nkv: 2,
            hd: 16,
            rot: 16,
            theta: 1e4,
            eps: 1e-6,
            ranks: [4; 3],
        };
        let fs: Vec<HostFactor> = (0..3)
            .map(|k| {
                let (o, i) = sh.dims(k);
                factor(o, i, sh.ranks[k], 10 * k as u64 + 1)
            })
            .collect();
        let (t, d) = (sh.t(), sh.d);
        let x = unit(t * d, 91);
        let g1: Vec<f32> = unit(d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
        let gn: Vec<f32> = unit(2 * sh.hd, 92).iter().map(|v| 1.0 + 0.2 * v).collect();
        let r: Vec<f32> = unit(t * sh.qw(), 93).iter().map(|v| 0.5 * v).collect();
        let host = Host {
            sh,
            gn: to64(&gn),
            r: to64(&r),
            n1: rmsnorm(&to64(&x), &to64(&g1), t, d),
        };

        // (a) the reference's scale gradients against central differences of its own loss.
        let analytic: Vec<[Vec<f64>; 4]> = {
            let dw = host.dw(&fs);
            (0..3).map(|k| factor_grads(&fs[k], &dw[k])).collect()
        };
        let mut worst = 0f64;
        for k in 0..3 {
            for j in [2usize, 3] {
                let len = if j == 2 { fs[k].out } else { fs[k].inn };
                for idx in [0, len / 2, len - 1] {
                    let eps = 1e-4f32;
                    let (mut p, mut q) = (fs.clone(), fs.clone());
                    if j == 2 {
                        p[k].s_post[idx] += eps;
                        q[k].s_post[idx] -= eps;
                    } else {
                        p[k].s_pre[idx] += eps;
                        q[k].s_pre[idx] -= eps;
                    }
                    // The perturbed entries are f32, so the exact step is their difference, not
                    // `2·eps`: rounding `s + eps` to f32 is a 6e-5 relative error at these scales
                    // and it would dominate the difference.
                    let (hi, lo) = if j == 2 {
                        (p[k].s_post[idx], q[k].s_post[idx])
                    } else {
                        (p[k].s_pre[idx], q[k].s_pre[idx])
                    };
                    let fd = (host.loss(&p) - host.loss(&q)) / (f64::from(hi) - f64::from(lo));
                    let got = analytic[k][j][idx];
                    let err = (got - fd).abs() / fd.abs().max(1e-6);
                    worst = worst.max(err);
                }
            }
        }
        println!("fd worst relative error {worst:.2e}");
        assert!(worst < 1e-6, "fd worst relative error {worst:.2e}");

        unsafe {
            let mut ft = FrontTune::new(&g, sh, 1e-3, 2e-3);
            ft.load(&g, &fs, &g1, &gn);
            ft.batch(&g, &x, &r);
            ft.forward(&g);
            println!("ao vs host {:.3e}", rel32(&ft.output(&g), &host.ao(&fs)));

            // (b) the device factor gradients of the three slots against the reference.
            ft.backward(&g, &r);
            let gr = ft.grads(&g);
            for k in 0..3 {
                let mut w = 0f64;
                for j in 0..4 {
                    w = w.max(rel32(&gr[k][j], &analytic[k][j]));
                }
                println!(
                    "qkv grad worst relative error {w:.2e} ({})",
                    ["q", "k", "v"][k]
                );
                assert!(w < 1e-3, "slot {k}: worst relative error {w:.2e}");
            }

            // (c) a short Adam run on the same loss.
            ft.forward(&g);
            let l0 = dot(&to64(&ft.output(&g)), &host.r);
            for _ in 0..20 {
                ft.train(&g);
            }
            ft.forward(&g);
            let l1 = dot(&to64(&ft.output(&g)), &host.r);
            println!("adam: loss {l0:.6} -> {l1:.6}");
            assert!(
                l1 < l0,
                "20 Adam steps did not lower the loss: {l0} -> {l1}"
            );
        }
    }

    /// Dense `q`, `k`, `v`: the device `ao` and the three `dW` equal the f64 host reference of the
    /// same dense weights (the twin of `front_tune_matches_host`), and a train mask over one slot
    /// moves that slot only.
    #[test]
    fn front_dense_matches_host() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let sh = FrontShape {
            chunks: 2,
            n_ctx: 8,
            d: 32,
            nh: 4,
            nkv: 2,
            hd: 16,
            rot: 16,
            theta: 1e4,
            eps: 1e-6,
            ranks: [4; 3],
        };
        let (t, d) = (sh.t(), sh.d);
        let (qw, kw) = (sh.qw(), sh.kw());
        let x = unit(t * d, 91);
        let g1: Vec<f32> = unit(d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
        let gn: Vec<f32> = unit(2 * sh.hd, 92).iter().map(|v| 1.0 + 0.2 * v).collect();
        let r: Vec<f32> = unit(t * qw, 93).iter().map(|v| 0.5 * v).collect();
        let ws: Vec<Vec<f32>> = [(qw, 4u64), (kw, 5), (kw, 6)]
            .iter()
            .map(|&(o, sd)| unit(o * d, sd).iter().map(|&v| 0.3 * v).collect())
            .collect();
        let host = Host {
            sh,
            gn: to64(&gn),
            r: to64(&r),
            n1: rmsnorm(&to64(&x), &to64(&g1), t, d),
        };
        let w64: Vec<Vec<f64>> = ws.iter().map(|w| to64(w)).collect();
        let pq = lin(&host.n1, &w64[0], t, d, qw);
        let pk = lin(&host.n1, &w64[1], t, d, kw);
        let pv = lin(&host.n1, &w64[2], t, d, kw);
        let ao = ref_fwd_ungated(sh.attn(), &pq, &pk, &host.gn, &pv);
        let (_, dqf, dkf, dv) = ref_bwd_ungated(sh.attn(), &pq, &pk, &host.gn, &pv, &host.r);
        let want = [
            dwt(&dqf, &host.n1, t, d, qw),
            dwt(&dkf, &host.n1, t, d, kw),
            dwt(&dv, &host.n1, t, d, kw),
        ];
        unsafe {
            let mut ft = FrontTune::new(&g, sh, 1e-3, 1e-3);
            // The norm gains come from `load`, whose factors a dense stage never reads.
            let fs: Vec<HostFactor> = (0..3)
                .map(|k| {
                    let (o, i) = sh.dims(k);
                    factor(o, i, sh.ranks[k], 10 * k as u64 + 1)
                })
                .collect();
            ft.load(&g, &fs, &g1, &gn);
            for k in 0..3 {
                ft.load_dense(&g, k, &ws[k]);
            }
            ft.set_stage(&g, [false; 3], [true; 3]);
            for k in 0..3 {
                let bits = |v: &[f32]| v.iter().map(|&x| x.to_bits()).collect::<Vec<u32>>();
                assert_eq!(bits(&ft.dense_w(&g, k)), bits(&ws[k]), "dense W {k} upload");
            }
            ft.batch(&g, &x, &r);
            ft.forward(&g);
            println!("dense ao vs host {:.3e}", rel32(&ft.output(&g), &ao));
            ft.backward(&g, &r);
            let mut worst = 0f64;
            for k in 0..3 {
                let w = rel32(&ft.dense_grad(&g, k), &want[k]);
                println!(
                    "dense dW worst relative error {w:.2e} ({})",
                    ["q", "k", "v"][k]
                );
                worst = worst.max(w);
            }
            assert!(worst < 1e-3, "dense dW worst relative error {worst:.2e}");
            // The mask: training slot 1 leaves slots 0 and 2 bit-identical.
            ft.set_stage(&g, [false; 3], [false, true, false]);
            let before: Vec<Vec<u32>> = (0..3).map(|k| ft.slot_words(&g, k)).collect();
            for _ in 0..3 {
                ft.train(&g);
            }
            let after: Vec<Vec<u32>> = (0..3).map(|k| ft.slot_words(&g, k)).collect();
            assert!(before[1] != after[1], "slot 1 did not move");
            assert_eq!(before[0], after[0], "slot 0 moved");
            assert_eq!(before[2], after[2], "slot 2 moved");
            println!("front mask: slot 1 moved, slots 0 and 2 hold");
            // The optimiser stage: the reference's constants and its schedule through the tape.
            ft.set_hyper(&g, AdamHyper::OPTIMI);
            ft.set_schedule(crate::gpu::nn::cosine_lr(1.0, 1e-4, 256, 2));
            ft.set_stage(&g, [true; 3], [true; 3]);
            let before = ft.slot_words(&g, 0);
            ft.train(&g);
            assert!(
                before != ft.slot_words(&g, 0),
                "the scheduled step moved nothing"
            );
            println!("front optimi + cosine schedule: one step moved slot 0");
        }
    }
}
