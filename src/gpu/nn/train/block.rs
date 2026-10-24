//! The `NanoQuant` block fit on the device: one residual block's output projection and `SwiGLU` FFN
//! with every weight a binary factor `W = diag(s_post)·sign(U)·sign(V)ᵀ·diag(s_pre)`
//! (`SamsungLabs/NanoQuant` @ `a9e0a43`, `tune_fact`), trained by Adam through the
//! straight-through sign against the teacher's block output.
//!
//! The tape over `T` columns, `x` the block input and `a` the o-projection input of the student
//! stream, `y_t` the teacher's block output:
//! `h = x + a·Woᵀ`; `n = rmsnorm(h, g2)`; `y = h + (silu(n·Wgᵀ) ⊙ n·Wuᵀ)·Wdᵀ`;
//! `L = Σ (y − y_t)²`, `dy = (y − y_t)·scale`; then the adjoints of the dense weights
//! (`dWd = dyᵀ·m`, `dWg`, `dWu` through `swiglu_bwd`, `dh = dy + rmsnorm_bwd(dn)`, `dWo = dhᵀ·a`)
//! and of each factor: `dM = dW ⊙ s_post·s_preᵀ`, `ds_post = rowsum(dW ⊙ M ⊙ s_preᵀ)`,
//! `ds_pre = colsum(dW ⊙ M ⊙ s_post)`, `dU = dM·sign(V)`, `dV = dMᵀ·sign(U)` (the sign's
//! straight-through derivative is the identity). Consumer: `tests/learn_block_tune.rs`, and
//! `chain.rs`, which stacks a front and a block into a chained layer.
//!
//! Under [`BlockTune::set_external_cotangent`] the tape records no `mse_grad`: the caller supplies
//! `dy`, the adjoint of the block output, in the device buffer [`BlockTune::dy_col`] names, and
//! [`BlockTune::dh`] exposes the residual adjoint at `h = x + a·Woᵀ` — the layer input's cotangent
//! down the residual path, which an attention front adds to its own input adjoint. That is the
//! `dL/dx` a quantisation-aware distillation needs from every block, not only from the last.

use crate::gpu::decode::{rec_step, GpuCtx, Step};
use crate::gpu::nn::act::{swiglu_bwd_dispatch, swiglu_fwd_dispatch};
use crate::gpu::nn::fwd::{linear_fwd_dispatch, rmsnorm_fwd_dispatch};
use crate::gpu::nn::{
    adam_scheduled_bias, add_inplace_bytes, linear_bwd_input_dispatch, linear_bwd_weight_dispatch,
    rmsnorm_bwd_dispatch, AdamHyper,
};
use crate::gpu::vulkan::*;

use super::slot::{map, q32, upload_dense, upload_factor, Factor};

/// The block maps (sign, scale, row and column reductions, MSE gradient, loss rows).
pub static BLOCK_MAP_SPV: &[u8] = include_bytes!("../../spirv/train/block_map.spv");

/// One factor's host image: `u` is `out × rank`, `vt` is `inn × rank` (the reference's `V`
/// transposed), both latent reals; `s_post` has `out` entries, `s_pre` `inn`.
#[derive(Clone, Debug, PartialEq)]
pub struct HostFactor {
    pub out: usize,
    pub inn: usize,
    pub rank: usize,
    pub u: Vec<f32>,
    pub vt: Vec<f32>,
    pub s_post: Vec<f32>,
    pub s_pre: Vec<f32>,
}

/// The block geometry: `t` columns per step, model width `d`, o-projection input `a`, FFN width
/// `f`, and the ranks of the factors in the order `[o, gate, up, down]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockShape {
    pub t: usize,
    pub d: usize,
    pub a: usize,
    pub f: usize,
    pub ranks: [usize; 4],
}

impl BlockShape {
    /// `(out, inn)` of factor `k` in `[o, gate, up, down]`.
    #[must_use]
    pub fn dims(&self, k: usize) -> (usize, usize) {
        match k {
            0 => (self.d, self.a),
            1 | 2 => (self.f, self.d),
            _ => (self.d, self.f),
        }
    }
}

/// A host-visible, host-coherent f32 buffer and its mapped pointer.
pub(super) struct HostBuffer {
    pub(super) buf: Handle,
    pub(super) ptr: *mut f32,
    pub(super) n: usize,
}

/// Copy `data` into the device buffer `dst` through `stage`, on the reusable command buffer `cb`.
///
/// # Safety
///
/// `dst` holds at least `data.len()` f32; `cb` is not pending.
pub(super) unsafe fn upload(
    g: &GpuCtx,
    cb: CommandBuffer,
    stage: &HostBuffer,
    dst: Handle,
    data: &[f32],
) {
    upload_at(g, cb, stage, dst, 0, data);
}

/// [`upload`] into `dst` at word offset `at`.
///
/// # Safety
///
/// `dst` holds at least `at + data.len()` f32; `cb` is not pending.
#[allow(
    clippy::cast_possible_truncation,
    reason = "word offsets index a u64 byte offset"
)]
pub(super) unsafe fn upload_at(
    g: &GpuCtx,
    cb: CommandBuffer,
    stage: &HostBuffer,
    dst: Handle,
    at: usize,
    data: &[f32],
) {
    assert!(data.len() <= stage.n, "upload: {} words", data.len());
    std::ptr::copy_nonoverlapping(data.as_ptr(), stage.ptr, data.len());
    let vk = g.vk;
    vk.reset_cb(cb);
    vk.begin(cb);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: (at * 4) as u64,
        size: (data.len() * 4) as u64,
    };
    (vk.copy_buffer)(cb, stage.buf, dst, 1, &region);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
}

/// A device buffer of `words` f32 a stream copy reads from and writes to.
///
/// # Safety
///
/// Vulkan handle creation; the buffer is owned by the returned handle for the process.
pub unsafe fn stream_buffer(g: &GpuCtx, words: usize) -> Handle {
    g.vk.buffer(
        words.max(1) as u64 * 4,
        BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST,
        MEMORY_DEVICE_LOCAL,
    )
    .expect("stream buffer")
    .0
}

/// A device-resident token stream: `n_all` chunks of `n_ctx` columns of width `w`, chunk-major.
pub struct Stream {
    pub buf: Handle,
    pub n_ctx: usize,
    pub w: usize,
    pub n_all: usize,
}

impl Stream {
    /// Allocate `n_all` chunks of `n_ctx × w` on the device.
    ///
    /// # Safety
    ///
    /// `g` outlives the value.
    pub unsafe fn new(g: &GpuCtx, n_ctx: usize, w: usize, n_all: usize) -> Self {
        Self {
            buf: stream_buffer(g, n_ctx * w * n_all),
            n_ctx,
            w,
            n_all,
        }
    }

    /// f32 per chunk.
    #[must_use]
    pub fn chunk_words(&self) -> usize {
        self.n_ctx * self.w
    }

    /// The words of the whole stream.
    #[must_use]
    pub fn words(&self) -> usize {
        self.chunk_words() * self.n_all
    }
}

/// The step's copies from `s` into words `[k·chunk, (k+1)·chunk)` of `buf`, in `ids` order: chunk
/// `ids[k]` of `s` into slot `k`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "word offsets index a u64 byte offset"
)]
pub fn chunk_into(s: &Stream, ids: &[usize], buf: Handle) -> Copies {
    let c = s.chunk_words();
    Copies {
        src: s.buf,
        dst: buf,
        regions: ids
            .iter()
            .enumerate()
            .map(|(k, &id)| BufferCopy {
                src_offset: (id * c * 4) as u64,
                dst_offset: (k * c * 4) as u64,
                size: (c * 4) as u64,
            })
            .collect(),
    }
}

/// The mirror of [`chunk_into`]: slot `k` of `buf` into chunk `ids[k]` of `s`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "word offsets index a u64 byte offset"
)]
pub fn chunk_from(s: &Stream, ids: &[usize], buf: Handle) -> Copies {
    let c = s.chunk_words();
    Copies {
        src: buf,
        dst: s.buf,
        regions: ids
            .iter()
            .enumerate()
            .map(|(k, &id)| BufferCopy {
                src_offset: (k * c * 4) as u64,
                dst_offset: (id * c * 4) as u64,
                size: (c * 4) as u64,
            })
            .collect(),
    }
}

/// One device-to-device transfer group, recorded beside the compute: `regions` from `src` into
/// `dst`.
pub struct Copies {
    pub src: Handle,
    pub dst: Handle,
    pub regions: Vec<BufferCopy>,
}

/// Record `groups` on `cb`.
///
/// # Safety
///
/// Every group's buffers hold its regions; `cb` is recording.
pub(super) unsafe fn record_copies(vk: &Vk, cb: CommandBuffer, groups: &[Copies]) {
    for c in groups {
        if !c.regions.is_empty() {
            let n = c.regions.len() as u32;
            (vk.copy_buffer)(cb, c.src, c.dst, n, c.regions.as_ptr());
        }
    }
}

/// Upload `data` into the whole of `dst`, one submission per 2 M-word slice.
///
/// # Safety
///
/// `dst` holds `data.len()` f32.
pub unsafe fn stream_upload(g: &GpuCtx, dst: Handle, data: &[f32]) {
    let n = (1 << 21).min(data.len().max(1));
    let stage = host_buffer(g, n, BUFFER_USAGE_TRANSFER_SRC);
    let cb = g.vk.reusable_command_buffer();
    for (i, part) in data.chunks(n).enumerate() {
        upload_at(g, cb, &stage, dst, i * n, part);
    }
}

/// Download the whole of `src`'s `n` f32 into a host vector, one submission per 2 M-word slice.
///
/// # Safety
///
/// `src` holds `n` f32.
pub unsafe fn stream_download(g: &GpuCtx, src: Handle, n: usize) -> Vec<f32> {
    let m = (1 << 21).min(n.max(1));
    let rb = host_buffer(g, m, BUFFER_USAGE_TRANSFER_DST);
    let cb = g.vk.reusable_command_buffer();
    let mut out = Vec::with_capacity(n);
    let mut at = 0;
    while at < n {
        let take = m.min(n - at);
        out.extend_from_slice(&download_at(g, cb, &rb, src, at, take));
        at += take;
    }
    out
}

/// Read `n` f32 of the device buffer `src` through `rb`, on the reusable command buffer `cb`.
///
/// # Safety
///
/// `src` holds at least `n` f32; `cb` is not pending.
pub(super) unsafe fn download(
    g: &GpuCtx,
    cb: CommandBuffer,
    rb: &HostBuffer,
    src: Handle,
    n: usize,
) -> Vec<f32> {
    download_at(g, cb, rb, src, 0, n)
}

/// [`download`] of `n` f32 of `src` from word offset `at`.
///
/// # Safety
///
/// `src` holds at least `at + n` f32; `cb` is not pending.
#[allow(
    clippy::cast_possible_truncation,
    reason = "word offsets index a u64 byte offset"
)]
pub(super) unsafe fn download_at(
    g: &GpuCtx,
    cb: CommandBuffer,
    rb: &HostBuffer,
    src: Handle,
    at: usize,
    n: usize,
) -> Vec<f32> {
    assert!(n <= rb.n, "download: {n} words");
    let vk = g.vk;
    vk.reset_cb(cb);
    vk.begin(cb);
    vk.barrier(
        cb,
        STAGE_COMPUTE,
        STAGE_TRANSFER,
        ACCESS_SHADER_WRITE,
        ACCESS_TRANSFER_READ,
    );
    let region = BufferCopy {
        src_offset: (at * 4) as u64,
        dst_offset: 0,
        size: (n * 4) as u64,
    };
    (vk.copy_buffer)(cb, src, rb.buf, 1, &region);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
    std::slice::from_raw_parts(rb.ptr, n).to_vec()
}

/// # Safety
///
/// Vulkan handle creation; the buffer is owned by the returned value.
pub(super) unsafe fn host_buffer(g: &GpuCtx, n: usize, usage: u32) -> HostBuffer {
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
        n,
    }
}

/// The device block fit: the buffers and the recorded tape of one step.
pub struct BlockTune {
    shape: BlockShape,
    factors: Vec<Factor>,
    x: Handle,
    a: Handle,
    /// The o-projection's input adjoint `[t][a]`, written by the backward.
    da: Handle,
    /// The residual adjoint `dh` at `h = x + a·Woᵀ`, `[t][d]`, written by the backward: the input
    /// `x`'s cotangent down the residual path.
    dh: Handle,
    /// The weighted MSE gradient `dy[r,c] = scale·imp[c]·(y−y_t)[r,c]`, `[t][d]` — or, under
    /// [`BlockTune::set_external_cotangent`], the caller's adjoint of `y`.
    dy: Handle,
    yt: Handle,
    y: Handle,
    g2: Handle,
    loss: Handle,
    /// A zeroed `t × d` buffer and the per-column sums of `yt²` (`norm`), beside `loss`'s rows.
    loss2: Handle,
    norm: Step,
    /// Device → host: the loss rows and the `yt²` column sums.
    rb_loss: HostBuffer,
    rb: HostBuffer,
    /// Host → device staging, as large as the largest buffer.
    stage: HostBuffer,
    bias_stage: HostBuffer,
    bias_buf: Handle,
    /// Materialise `W` from the latents, per factor: the two signs, `M`, then the scales.
    materialise: Vec<Vec<Step>>,
    forward: Vec<Step>,
    /// The dense adjoints: every slot's `dW` and the residual adjoints down to `a`.
    backward: Vec<Step>,
    /// The factor adjoint of slot `k`: `dM`, the two scale gradients, `dU`, `dV`.
    factor_bwd: Vec<Vec<Step>>,
    /// Adam over a factor's latents and scales, per factor.
    adam_fact: Vec<Vec<Step>>,
    /// One Adam step per slot over its dense `W` and `dW`.
    adam_dense: Vec<Step>,
    /// Adam moments, zeroed by [`BlockTune::load`]; `[factor][param][m, v]`.
    moments: Vec<[[Handle; 2]; 4]>,
    /// Each slot's dense moments `[m, v]` over `out × inn`, zeroed by [`BlockTune::load_dense`].
    dense_moments: Vec<[Handle; 2]>,
    /// The block's state per slot `[o, gate, up, down]`: `true` where `W` is held as a factor,
    /// `false` where it is a dense `W` in its `w` buffer.
    factor: [bool; 4],
    /// The stage's trained set per slot: `true` where slot `k`'s parameters are updated, whichever
    /// of the two forms `factor[k]` selects; outside it, parameters and moments stay bit-identical.
    train: [bool; 4],
    /// The loss's per-output importance `D`, `d` words; all ones until [`BlockTune::load_importance`].
    imp: Handle,
    /// `true` under [`BlockTune::set_external_cotangent`]: the tapes record no `mse_grad` step and
    /// the caller fills `dy`.
    cot: bool,
    /// The base rates `[latent, scale, dense]` in real units and the constants their steps bake.
    lr: [f64; 3],
    hyper: AdamHyper,
    /// The steps' schedule multiplier `c = lr_t/lr_base`; empty means a constant rate.
    sched: Vec<f64>,
    step_no: u32,
    /// The recorded tapes: a training step (bias copy, materialise, forward, backward, Adam,
    /// loss readback) and an evaluation pass (materialise, forward, loss readback).
    cb_train: CommandBuffer,
    cb_eval: CommandBuffer,
    /// The step's column copies: `pre` at the head of the tape (stream → `x`/`a`/`yt`), `post`
    /// at its tail (`y` → stream). Empty on the host-fed [`BlockTune::batch`] path.
    pre: Vec<Copies>,
    post: Vec<Copies>,
    /// Re-recorded for every host transfer.
    xfer: CommandBuffer,
}

impl BlockTune {
    /// Allocate every buffer and build every pipeline for `shape`, with Adam learning rates `lr`
    /// for the latents, `lr_scale` for the two scale vectors and `lr_dense` for a dense slot's
    /// `W` (real units).
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value.
    ///
    /// # Complexity
    ///
    /// `O(1)` pipelines (about 70) and `O(t·(d + a + f) + Σ out·inn)` device words.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::similar_names,
        reason = "one tape; the dimensions index u32 dispatch shapes"
    )]
    #[must_use]
    pub unsafe fn new(
        g: &GpuCtx,
        shape: BlockShape,
        lr: f64,
        lr_scale: f64,
        lr_dense: f64,
    ) -> Self {
        let BlockShape { t, d, a, f, ranks } = shape;
        let dev = |n: usize| g.dev(n * 4);
        // A buffer a chained layer reads by `vkCmdCopyBuffer` carries `TRANSFER_SRC` as well; `dev`
        // allocates `STORAGE | TRANSFER_DST` only.
        let stream = |n: usize| stream_buffer(g, n);
        let mut factors = Vec::with_capacity(4);
        let mut moments = Vec::with_capacity(4);
        for (k, &rank) in ranks.iter().enumerate() {
            let (out, inn) = shape.dims(k);
            let (fa, mo) = Factor::new(g, out, inn, rank);
            factors.push(fa);
            moments.push(mo);
        }
        // The five buffers a chained layer copies into or out of (`chain.rs`) carry both transfer
        // bits: the input `x`, the o-projection input `a`, the output `y` and the two adjoints.
        let (x, at, y) = (
            stream_buffer(g, t * d),
            stream_buffer(g, t * a),
            stream(t * d),
        );
        let (da, dy, dh) = (stream_buffer(g, t * a), stream(t * d), stream(t * d));
        let (yt, h, n2) = (dev(t * d), dev(t * d), dev(t * d));
        let (gt, up, mm) = (dev(t * f), dev(t * f), dev(t * f));
        let (dmm, dgt, dup) = (dev(t * f), dev(t * f), dev(t * f));
        let (dn2, tmp, dgr) = (dev(t * d), dev(t * d), dev(t * d));
        let (g2, loss) = (dev(d), dev(t));
        let (zero, loss2) = (g.dev_zeroed(t * d * 4), dev(t));
        // The per-output importance of the loss; all ones until `load_importance` writes it.
        let imp = dev(d);
        let norm = map(
            g,
            5,
            [t, d, t * d],
            0.0,
            [yt, zero, loss2, imp],
            "norm_rows",
        );
        let (to, tg, tu, td) = (t as u32, d as u32, a as u32, f as u32);
        let wt = |k: usize| factors[k].w;
        let dwt = |k: usize| factors[k].dw;

        let mut materialise = Vec::with_capacity(4);
        for fa in &factors {
            materialise.push(super::slot::materialise(g, fa));
        }

        let add = |dst: Handle, src: Handle, n: usize| {
            crate::gpu::decode::make_step_named(
                g,
                &add_inplace_bytes(n as u32, 256),
                &[dst, src],
                (n as u32).div_ceil(256),
                1,
                "add_inplace",
            )
        };
        // The weighted MSE gradient's scale `s` in `dy[r,c] = s·imp[c]·(y−yt)[r,c]`: the mean over
        // the entries, `2/(t·d)`. The reference's divisor is the batch size (`compress_block.py:88`),
        // which the caller sets through `set_loss_scale`.
        let mse = 2.0 / (t * d) as f32;
        let forward = vec![
            linear_fwd_dispatch(g, at, wt(0), h, to, tu, tg),
            add(h, x, t * d),
            rmsnorm_fwd_dispatch(g, h, g2, n2, to, tg),
            linear_fwd_dispatch(g, n2, wt(1), gt, to, tg, td),
            linear_fwd_dispatch(g, n2, wt(2), up, to, tg, td),
            swiglu_fwd_dispatch(g, gt, up, mm, (t * f) as u32),
            linear_fwd_dispatch(g, mm, wt(3), y, to, td, tg),
            add(y, h, t * d),
            map(g, 5, [t, d, t * d], 0.0, [y, yt, loss, imp], "loss_rows"),
            map(g, 4, [t, d, t * d], mse, [y, yt, dy, imp], "mse_grad"),
        ];

        let backward = vec![
            linear_bwd_weight_dispatch(g, dy, mm, dwt(3), to, td, tg),
            linear_bwd_input_dispatch(g, dy, wt(3), dmm, to, td, tg),
            swiglu_bwd_dispatch(g, gt, up, dmm, dgt, dup, (t * f) as u32),
            linear_bwd_weight_dispatch(g, dgt, n2, dwt(1), to, tg, td),
            linear_bwd_weight_dispatch(g, dup, n2, dwt(2), to, tg, td),
            linear_bwd_input_dispatch(g, dgt, wt(1), dn2, to, tg, td),
            linear_bwd_input_dispatch(g, dup, wt(2), tmp, to, tg, td),
            add(dn2, tmp, t * d),
            rmsnorm_bwd_dispatch(g, h, g2, dn2, dh, dgr, to, tg, 1e-6),
            add(dh, dy, t * d),
            linear_bwd_weight_dispatch(g, dh, at, dwt(0), to, tu, tg),
            // The o-projection's input adjoint `da = dh·Wo`, `[t][a]`: the forward's dims
            // verbatim, `n = t`, `i = a`, `o = d`.
            linear_bwd_input_dispatch(g, dh, wt(0), da, to, tu, tg),
        ];
        let mut factor_bwd = Vec::with_capacity(4);
        for fa in &factors {
            factor_bwd.push(super::slot::factor_bwd(g, fa));
        }

        let bias_stage = host_buffer(g, 2, BUFFER_USAGE_TRANSFER_SRC);
        let bias_buf = dev(2);
        let dense_moments: Vec<[Handle; 2]> = factors
            .iter()
            .map(|fa| super::slot::dense_moments(g, fa))
            .collect();
        let big = factors
            .iter()
            .map(|fa| fa.out * fa.inn)
            .chain([t * a, t * d])
            .max()
            .unwrap_or(1);
        let mut s = Self {
            shape,
            factors,
            x,
            a: at,
            da,
            dh,
            dy,
            yt,
            y,
            g2,
            loss,
            loss2,
            norm,
            rb_loss: host_buffer(g, 2 * t, BUFFER_USAGE_TRANSFER_DST),
            rb: host_buffer(g, big, BUFFER_USAGE_TRANSFER_DST),
            stage: host_buffer(g, big, BUFFER_USAGE_TRANSFER_SRC),
            bias_stage,
            bias_buf,
            materialise,
            forward,
            backward,
            factor_bwd,
            adam_fact: Vec::new(),
            adam_dense: Vec::new(),
            moments,
            dense_moments,
            factor: [true; 4],
            train: [true; 4],
            imp,
            cot: false,
            lr: [lr, lr_scale, lr_dense],
            hyper: AdamHyper::ORACLE,
            sched: Vec::new(),
            step_no: 0,
            pre: Vec::new(),
            post: Vec::new(),
            cb_train: g.vk.reusable_command_buffer(),
            cb_eval: g.vk.reusable_command_buffer(),
            xfer: g.vk.reusable_command_buffer(),
        };
        s.build_adam(g);
        let ones = vec![1.0f32; d];
        upload(g, s.xfer, &s.stage, s.imp, &ones);
        s.record(g, true);
        s.record(g, false);
        s
    }

    /// Record the training or the evaluation tape for the current stage; [`BlockTune::run`]
    /// resubmits it. Both buffers are reset before the re-record.
    ///
    /// # Safety
    ///
    /// Called from [`BlockTune::new`], [`BlockTune::load`] and [`BlockTune::set_stage`]; no tape is
    /// pending on the device.
    unsafe fn record(&self, g: &GpuCtx, train: bool) {
        let vk = g.vk;
        let cb = if train { self.cb_train } else { self.cb_eval };
        vk.reset_cb(cb);
        vk.begin(cb);
        record_copies(vk, cb, &self.pre);
        if train {
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
            if self.factor[k] {
                for s in steps {
                    rec_step(vk, cb, s);
                }
            }
        }
        // Under an external cotangent the last forward step (`mse_grad`) is not recorded: `dy` is
        // the caller's.
        let nfwd = if self.cot {
            self.forward.len() - 1
        } else {
            self.forward.len()
        };
        for s in &self.forward[..nfwd] {
            rec_step(vk, cb, s);
        }
        rec_step(vk, cb, &self.norm);
        if train {
            for s in &self.backward {
                rec_step(vk, cb, s);
            }
            // The trained set: a slot outside it records no adjoint and no Adam step, so its
            // parameters and its moments are untouched. A factor slot trains the adjoint of its
            // latents and scales, a dense slot the Adam step over its `W`.
            for k in 0..4 {
                if !self.train[k] {
                    continue;
                }
                if self.factor[k] {
                    for s in &self.factor_bwd[k] {
                        rec_step(vk, cb, s);
                    }
                    for s in &self.adam_fact[k] {
                        rec_step(vk, cb, s);
                    }
                } else {
                    rec_step(vk, cb, &self.adam_dense[k]);
                }
            }
        }
        vk.barrier(
            cb,
            STAGE_COMPUTE,
            STAGE_TRANSFER,
            ACCESS_SHADER_WRITE,
            ACCESS_TRANSFER_READ,
        );
        let region = BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (self.shape.t * 4) as u64,
        };
        let region2 = BufferCopy {
            src_offset: 0,
            dst_offset: (self.shape.t * 4) as u64,
            size: (self.shape.t * 4) as u64,
        };
        (vk.copy_buffer)(cb, self.loss, self.rb_loss.buf, 1, &region);
        (vk.copy_buffer)(cb, self.loss2, self.rb_loss.buf, 1, &region2);
        record_copies(vk, cb, &self.post);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_HOST,
            ACCESS_TRANSFER_WRITE,
            ACCESS_HOST_READ,
        );
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

    /// Load the four factors (`[o, gate, up, down]`) and the FFN norm gain, zero every moment,
    /// restart the step count and set the stage to all four factors.
    ///
    /// # Safety
    ///
    /// The factors match the shape's dimensions and ranks (asserted); no tape is pending.
    ///
    /// # Complexity
    ///
    /// `O(Σ (out + inn)·rank)` words copied.
    pub unsafe fn load(&mut self, g: &GpuCtx, factors: &[HostFactor], g2: &[f32]) {
        assert_eq!(factors.len(), 4, "block load: four factors");
        for (k, hf) in factors.iter().enumerate() {
            self.load_factor(g, k, hf);
        }
        self.put(g, self.g2, g2);
        self.step_no = 0;
        self.set_stage(g, [true; 4], [true; 4]);
    }

    /// Put slot `k`'s factor and zero its four moment pairs. Leaves the stage as it is.
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
            "block load_factor {k}: shape"
        );
        upload_factor(g, self.xfer, &self.stage, fa, hf, &self.moments[k]);
    }

    /// Put slot `k`'s dense `W` (`out × inn`, row-major) into its `w` buffer and zero that slot's
    /// dense moments. Leaves the stage as it is.
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
        assert_eq!(w.len(), fa.out * fa.inn, "block load_dense {k}: shape");
        upload_dense(g, self.xfer, &self.stage, fa, w, &self.dense_moments[k]);
    }

    /// Set the stage: `factor[k]` places slot `k`'s `W` in its latents (`true`) or in its `w`
    /// buffer (`false`), and `train[k]` puts slot `k` in the trained set — it updates its factor's
    /// latents and scales when `factor[k]`, or its dense `W` when not; a slot outside the set keeps
    /// its parameters and moments bit-identical. The reference's two scopes: `tune_nonfact` trains
    /// the still-dense linears (`compress_block.py:58-62`), `tune_fact` one linear's factor
    /// parameters (`:22-48`), one slot at a time (`compress_model.py:83-101`).
    ///
    /// # Safety
    ///
    /// No tape is pending on the device (the last [`BlockTune::run`] returned).
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; the re-recorded tapes carry `O(1)` dispatches.
    pub unsafe fn set_stage(&mut self, g: &GpuCtx, factor: [bool; 4], train: [bool; 4]) {
        self.factor = factor;
        self.train = train;
        self.record(g, true);
        self.record(g, false);
    }

    /// Set the per-output importance `D` of the loss, `d` words: the loss is
    /// `Σ_r Σ_c D[c]·(y − y_t)[r,c]²` and its gradient `dy[r,c] = scale·D[c]·(y − y_t)[r,c]`.
    /// All ones is the default, the unweighted sum. The reference weights the same way
    /// (`compress_block.py:17-19`), `D` its `[hidden]` vector `sublayers['mlp.down_proj'].o_norm`
    /// (`compress_model.py:69-78`).
    ///
    /// # Safety
    ///
    /// `imp` has `d` words (asserted); no tape is pending.
    ///
    /// # Complexity
    ///
    /// `O(d)` words copied.
    pub unsafe fn load_importance(&mut self, g: &GpuCtx, imp: &[f32]) {
        assert_eq!(imp.len(), self.shape.d, "block load_importance: d words");
        upload(g, self.xfer, &self.stage, self.imp, imp);
    }

    /// Set the loss scale `s` in `dy[r,c] = s·D[c]·(y − y_t)[r,c]` and re-record both tapes. The
    /// reference's step is `(loss/batch_size).backward()` (`compress_block.py:88`), so a caller
    /// reproducing it sets `s = 2/bs`; the default is the mean over the entries, `2/(t·d)`.
    ///
    /// # Safety
    ///
    /// No tape is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; two tapes re-recorded.
    pub unsafe fn set_loss_scale(&mut self, g: &GpuCtx, scale: f32) {
        let (t, d) = (self.shape.t, self.shape.d);
        let end = self.forward.len() - 1;
        self.forward[end] = map(
            g,
            4,
            [t, d, t * d],
            scale,
            [self.y, self.yt, self.dy, self.imp],
            "mse_grad",
        );
        self.record(g, true);
        self.record(g, false);
    }

    /// Set the optimiser constants the Adam steps bake; the moments are kept and both tapes are
    /// re-recorded. [`AdamHyper::OPTIMI`] is the NanoQuant reference's set.
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
        self.build_adam(g);
        self.record(g, true);
        self.record(g, false);
    }

    /// Set the per-step schedule multiplier `c_t = lr_t/lr_base` every Adam step of the run shares:
    /// entry `t−1` is step `t`'s; empty (the default) keeps each group's base rate. The reference
    /// steps a `CosineAnnealingLR` once per optimiser step (`compress_block.py:96-98`), so `c` is
    /// `cosine_lr(1.0, 1e-4, t_max, steps)`; one scalar serves every group because `eta_min` is
    /// `1e-4` times each group's base rate (`:74`, `:177`), which makes the recursion linear in it.
    ///
    /// # Complexity
    ///
    /// `O(steps)` words held; `O(1)` host work.
    pub fn set_schedule(&mut self, c: Vec<f64>) {
        self.sched = c;
    }

    /// Build the Adam steps of every slot under `self.hyper`, leaving the moment buffers in place:
    /// `O(1)` pipelines, `O(Σ out·inn·rank + out·inn)` words per step.
    ///
    /// # Safety
    ///
    /// `g` is a live device context; no tape is pending when the caller re-records.
    unsafe fn build_adam(&mut self, g: &GpuCtx) {
        let (lr, ls, ld) = (q32(self.lr[0]), q32(self.lr[1]), q32(self.lr[2]));
        let mut adam_fact = Vec::with_capacity(4);
        let mut adam_dense = Vec::with_capacity(4);
        for (k, fa) in self.factors.iter().enumerate() {
            adam_fact.push(super::slot::adam_factor(
                g,
                fa,
                &self.moments[k],
                self.bias_buf,
                lr,
                ls,
                self.hyper,
            ));
            adam_dense.push(super::slot::adam_dense(
                g,
                fa,
                &self.dense_moments[k],
                self.bias_buf,
                ld,
                self.hyper,
            ));
        }
        self.adam_fact = adam_fact;
        self.adam_dense = adam_dense;
    }

    /// Set the step's columns: `x` and `yt` are `t × d`, `a` is `t × a`, row-major.
    ///
    /// # Safety
    ///
    /// The slices have the shape's lengths (asserted).
    pub unsafe fn batch(&mut self, g: &GpuCtx, x: &[f32], a: &[f32], yt: &[f32]) {
        let BlockShape { t, d, a: aw, .. } = self.shape;
        assert_eq!(
            (x.len(), a.len(), yt.len()),
            (t * d, t * aw, t * d),
            "block batch"
        );
        if !self.pre.is_empty() || !self.post.is_empty() {
            self.pre.clear();
            self.post.clear();
            self.record(g, true);
            self.record(g, false);
        }
        self.put(g, self.x, x);
        self.put(g, self.a, a);
        self.put(g, self.yt, yt);
    }

    /// Set the step's columns from device streams: `pre` copies the step's chunk regions into
    /// `x`, `a` and `yt`, `post` copies `y` back out, and both are recorded in the step's own
    /// command buffer, so one submission runs the copies and the tape.
    ///
    /// # Safety
    ///
    /// Every group's buffers hold its regions; no submission of this value is pending.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; the selected tape's `O(1)` dispatches re-recorded.
    pub unsafe fn set_cols(
        &mut self,
        g: &GpuCtx,
        pre: Vec<Copies>,
        post: Vec<Copies>,
        train: bool,
    ) {
        self.pre = pre;
        self.post = post;
        self.record(g, train);
    }

    /// The step's `Σ yt²`, from the per-column sums the tape recorded beside the loss rows.
    ///
    /// # Safety
    ///
    /// A run has completed.
    #[must_use]
    pub unsafe fn den(&self) -> f64 {
        std::slice::from_raw_parts(self.rb_loss.ptr.add(self.shape.t), self.shape.t)
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    }

    /// Record and run materialise + forward (+ backward and Adam when `train`); returns the step's
    /// weighted summed squared error `Σ D[c]·(y − y_t)[·,c]²` before the update.
    ///
    /// # Safety
    ///
    /// [`BlockTune::load`] and [`BlockTune::batch`] have run.
    ///
    /// # Complexity
    ///
    /// About `2·t·(a·d + 3·d·f)` multiply-adds forward, twice that backward, and
    /// `Σ out·inn·rank` per materialisation and per factor adjoint.
    pub unsafe fn run(&mut self, g: &GpuCtx, train: bool) -> f64 {
        if train {
            self.step_no += 1;
            let t = self.step_no;
            let (b1t, rbc2) = match self.sched.get(t as usize - 1) {
                Some(&c) => adam_scheduled_bias(self.hyper, t, c),
                None => self.hyper.bias_corrections(t),
            };
            *self.bias_stage.ptr = b1t;
            *self.bias_stage.ptr.add(1) = rbc2;
        }
        g.vk.submit_wait(if train { self.cb_train } else { self.cb_eval });
        std::slice::from_raw_parts(self.rb_loss.ptr, self.shape.t)
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    }

    /// The step's buffers: the block input `x` (`t × d`), the o-projection input `a` (`t × a`) and
    /// the teacher target `yt` (`t × d`).
    #[must_use]
    pub fn cols(&self) -> (Handle, Handle, Handle) {
        (self.x, self.a, self.yt)
    }

    /// The block output `y` (`t × d`) buffer, without a download.
    #[must_use]
    pub fn out(&self) -> Handle {
        self.y
    }

    /// The adjoint of the o-projection input, `[t][a]`, valid after a backward.
    #[must_use]
    pub fn da(&self) -> Handle {
        self.da
    }

    /// The residual adjoint at `h = x + a·Woᵀ`, `[t][d]`, valid after a backward: the cotangent of
    /// the block input `x` down the residual path, which the attention front adds to its own input
    /// adjoint ([`chain`](super::chain)).
    #[must_use]
    pub fn dh(&self) -> Handle {
        self.dh
    }

    /// The `dy` buffer (`t × d`) the backward reads: the MSE gradient the tape computes, or, under
    /// [`BlockTune::set_external_cotangent`], the caller's device buffer for the adjoint of `y`.
    #[must_use]
    pub fn dy_col(&self) -> Handle {
        self.dy
    }

    /// Set the external-cotangent mode: when `on`, neither tape records the `mse_grad` step and the
    /// backward reads `dy` from [`BlockTune::dy_col`], which the caller fills by a device copy.
    /// `yt` and the loss rows then carry no meaning.
    ///
    /// # Safety
    ///
    /// No tape is pending on the device.
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; both tapes re-recorded.
    pub unsafe fn set_external_cotangent(&mut self, g: &GpuCtx, on: bool) {
        self.cot = on;
        self.record(g, true);
        self.record(g, false);
    }

    /// The block output `y` of the last [`BlockTune::run`], `t × d`.
    ///
    /// # Safety
    ///
    /// A run has completed.
    #[must_use]
    pub unsafe fn output(&self, g: &GpuCtx) -> Vec<f32> {
        self.get(g, self.y, self.shape.t * self.shape.d)
    }

    /// The four factors' current latents and scales.
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

    /// The materialised dense weight of factor `k` after the last run, `out × inn`.
    ///
    /// # Safety
    ///
    /// A run has completed.
    #[must_use]
    pub unsafe fn weight(&self, g: &GpuCtx, k: usize) -> Vec<f32> {
        let fa = &self.factors[k];
        self.get(g, fa.w, fa.out * fa.inn)
    }
}

#[cfg(test)]
impl BlockTune {
    /// The gradients of the last training step, `[factor][u, vt, s_post, s_pre]`.
    pub(crate) unsafe fn grads(&self, g: &GpuCtx) -> Vec<[Vec<f32>; 4]> {
        self.factors
            .iter()
            .map(|fa| {
                let n = [fa.out * fa.rank, fa.inn * fa.rank, fa.out, fa.inn];
                [0, 1, 2, 3].map(|j| self.get(g, fa.grad[j], n[j]))
            })
            .collect()
    }

    /// The dense `W` gradient of slot `k` from the last training step, `out × inn`.
    unsafe fn dense_grad(&self, g: &GpuCtx, k: usize) -> Vec<f32> {
        let fa = &self.factors[k];
        self.get(g, fa.dw, fa.out * fa.inn)
    }

    /// The weighted gradient `dy` of the last step, `[t][d]`.
    pub(crate) unsafe fn dy(&self, g: &GpuCtx) -> Vec<f32> {
        self.get(g, self.dy, self.shape.t * self.shape.d)
    }

    /// Every parameter and moment word of slot `k` in its current form — the four factor buffers
    /// with their moment pairs, or the dense `W` with its pair — as bits.
    pub(crate) unsafe fn slot_words(&self, g: &GpuCtx, k: usize) -> Vec<u32> {
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
pub(crate) mod tests;
