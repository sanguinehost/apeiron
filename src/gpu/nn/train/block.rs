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
//! straight-through derivative is the identity). Consumer: `tests/learn_block_tune.rs`.

use crate::gpu::decode::{rec_step, GpuCtx, Step};
use crate::gpu::nn::act::{swiglu_bwd_dispatch, swiglu_fwd_dispatch};
use crate::gpu::nn::fwd::{linear_fwd_dispatch, rmsnorm_fwd_dispatch};
use crate::gpu::nn::{
    adam_bias_corrections, adam_update_dispatch_reused, add_inplace_bytes,
    linear_bwd_input_dispatch, linear_bwd_weight_dispatch, rmsnorm_bwd_dispatch,
};
use crate::gpu::vulkan::*;

/// The block maps (sign, scale, row and column reductions, MSE gradient, loss rows).
pub static BLOCK_MAP_SPV: &[u8] = include_bytes!("../../spirv/train/block_map.spv");

const LOCAL: usize = 256;
const MAX_GROUPS_X: usize = 65_535;

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
    assert!(data.len() <= stage.n, "upload: {} words", data.len());
    std::ptr::copy_nonoverlapping(data.as_ptr(), stage.ptr, data.len());
    let vk = g.vk;
    vk.reset_cb(cb);
    vk.begin(cb);
    let region = BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: (data.len() * 4) as u64,
    };
    (vk.copy_buffer)(cb, stage.buf, dst, 1, &region);
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    vk.submit_wait(cb);
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
        src_offset: 0,
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

/// One factor's device buffers.
struct Factor {
    out: usize,
    inn: usize,
    rank: usize,
    /// Latents and scales, in the Adam order `[u, vt, s_post, s_pre]`, with their gradients and
    /// moments.
    p: [Handle; 4],
    grad: [Handle; 4],
    us: Handle,
    vts: Handle,
    m: Handle,
    w: Handle,
    dw: Handle,
    dm: Handle,
}

/// Map `mode` of [`BLOCK_MAP_SPV`] over `[rows, cols, total]` with bindings `bufs` (1..=4).
///
/// # Safety
///
/// The buffers hold what the mode reads and writes.
#[allow(
    clippy::cast_possible_truncation,
    reason = "rows, cols and totals index u32 invocations"
)]
unsafe fn map(
    g: &GpuCtx,
    mode: u32,
    [rows, cols, total]: [usize; 3],
    scale: f32,
    bufs: [Handle; 4],
    name: &'static str,
) -> Step {
    let wg = match mode {
        2 | 5 => rows,
        _ => total.div_ceil(LOCAL),
    };
    let gx = wg.clamp(1, MAX_GROUPS_X);
    let meta = [
        mode,
        rows as u32,
        cols as u32,
        total as u32,
        gx as u32,
        scale.to_bits(),
    ];
    let mb = g.up_u32(&meta);
    let (layout, dset) = g.vk.ssbo_set(&[mb, bufs[0], bufs[1], bufs[2], bufs[3]]);
    Step {
        pipe: g.vk.pipeline(BLOCK_MAP_SPV, layout),
        layout,
        dset,
        groups: (gx as u32, wg.div_ceil(gx).max(1) as u32),
        name,
        bytes: 0,
    }
}

/// The device block fit: the buffers and the recorded tape of one step.
pub struct BlockTune {
    shape: BlockShape,
    factors: Vec<Factor>,
    x: Handle,
    a: Handle,
    yt: Handle,
    y: Handle,
    g2: Handle,
    loss: Handle,
    /// Device → host: the loss rows and `y`.
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
    /// The stage's trained set: `true` trains the dense slots, `false` the factor slots.
    train_dense: bool,
    step_no: u32,
    /// The recorded tapes: a training step (bias copy, materialise, forward, backward, Adam,
    /// loss readback) and an evaluation pass (materialise, forward, loss readback).
    cb_train: CommandBuffer,
    cb_eval: CommandBuffer,
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
        let mut factors = Vec::with_capacity(4);
        let mut moments = Vec::with_capacity(4);
        for (k, &rank) in ranks.iter().enumerate() {
            let (out, inn) = shape.dims(k);
            let sizes = [out * rank, inn * rank, out, inn];
            factors.push(Factor {
                out,
                inn,
                rank,
                p: sizes.map(dev),
                grad: sizes.map(dev),
                us: dev(out * rank),
                vts: dev(inn * rank),
                m: dev(out * inn),
                w: dev(out * inn),
                dw: dev(out * inn),
                dm: dev(out * inn),
            });
            moments.push(sizes.map(|n| [dev(n), dev(n)]));
        }
        let (x, at, yt, h, n2) = (dev(t * d), dev(t * a), dev(t * d), dev(t * d), dev(t * d));
        let (gt, up, mm, y) = (dev(t * f), dev(t * f), dev(t * f), dev(t * d));
        let (dy, dmm, dgt, dup) = (dev(t * d), dev(t * f), dev(t * f), dev(t * f));
        let (dn2, tmp, dh, dgr) = (dev(t * d), dev(t * d), dev(t * d), dev(t * d));
        let (g2, loss) = (dev(d), dev(t));
        let (to, tg, tu, td) = (t as u32, d as u32, a as u32, f as u32);
        let wt = |k: usize| factors[k].w;
        let dwt = |k: usize| factors[k].dw;

        let mut materialise = Vec::with_capacity(4);
        for fa in &factors {
            let (out, inn, r) = (fa.out, fa.inn, fa.rank);
            materialise.push(vec![
                map(
                    g,
                    0,
                    [0, 0, out * r],
                    0.0,
                    [fa.p[0], fa.us, fa.us, fa.us],
                    "sign_u",
                ),
                map(
                    g,
                    0,
                    [0, 0, inn * r],
                    0.0,
                    [fa.p[1], fa.vts, fa.vts, fa.vts],
                    "sign_v",
                ),
                linear_fwd_dispatch(g, fa.us, fa.vts, fa.m, out as u32, r as u32, inn as u32),
                map(
                    g,
                    1,
                    [out, inn, out * inn],
                    0.0,
                    [fa.m, fa.w, fa.p[2], fa.p[3]],
                    "scale_w",
                ),
            ]);
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
        // The MSE gradient's scale: `2 / (t·d)`, the mean over the step's entries.
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
            map(g, 5, [t, d, t * d], 0.0, [y, yt, loss, loss], "loss_rows"),
            map(g, 4, [0, 0, t * d], mse, [y, yt, dy, dy], "mse_grad"),
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
        ];
        let mut factor_bwd = Vec::with_capacity(4);
        for fa in &factors {
            let (out, inn, r) = (fa.out, fa.inn, fa.rank);
            factor_bwd.push(vec![
                map(
                    g,
                    1,
                    [out, inn, out * inn],
                    0.0,
                    [fa.dw, fa.dm, fa.p[2], fa.p[3]],
                    "scale_dm",
                ),
                map(
                    g,
                    2,
                    [out, inn, out * inn],
                    0.0,
                    [fa.dw, fa.m, fa.grad[2], fa.p[3]],
                    "ds_post",
                ),
                map(
                    g,
                    3,
                    [out, inn, inn],
                    0.0,
                    [fa.dw, fa.m, fa.p[2], fa.grad[3]],
                    "ds_pre",
                ),
                linear_bwd_input_dispatch(
                    g, fa.dm, fa.vts, fa.grad[0], out as u32, r as u32, inn as u32,
                ),
                linear_bwd_weight_dispatch(
                    g, fa.dm, fa.us, fa.grad[1], out as u32, r as u32, inn as u32,
                ),
            ]);
        }

        let bias_stage = host_buffer(g, 2, BUFFER_USAGE_TRANSFER_SRC);
        let bias_buf = dev(2);
        let q32 = |v: f64| (v * 4_294_967_296.0).round() as i64;
        let mut adam_fact = Vec::with_capacity(4);
        let mut adam_dense = Vec::with_capacity(4);
        let mut dense_moments = Vec::with_capacity(4);
        for (fa, mo) in factors.iter().zip(&moments) {
            let mut steps = Vec::with_capacity(4);
            for j in 0..4 {
                let n = [fa.out * fa.rank, fa.inn * fa.rank, fa.out, fa.inn][j];
                let rate = if j < 2 { lr } else { lr_scale };
                steps.push(adam_update_dispatch_reused(
                    g,
                    fa.p[j],
                    fa.grad[j],
                    mo[j][0],
                    mo[j][1],
                    bias_buf,
                    n as u32,
                    q32(rate),
                ));
            }
            adam_fact.push(steps);
            let n = fa.out * fa.inn;
            let dmo = [dev(n), dev(n)];
            adam_dense.push(adam_update_dispatch_reused(
                g,
                fa.w,
                fa.dw,
                dmo[0],
                dmo[1],
                bias_buf,
                n as u32,
                q32(lr_dense),
            ));
            dense_moments.push(dmo);
        }
        let big = factors
            .iter()
            .map(|fa| fa.out * fa.inn)
            .chain([t * a, t * d])
            .max()
            .unwrap_or(1);
        let s = Self {
            shape,
            factors,
            x,
            a: at,
            yt,
            y,
            g2,
            loss,
            rb_loss: host_buffer(g, t, BUFFER_USAGE_TRANSFER_DST),
            rb: host_buffer(g, big, BUFFER_USAGE_TRANSFER_DST),
            stage: host_buffer(g, big, BUFFER_USAGE_TRANSFER_SRC),
            bias_stage,
            bias_buf,
            materialise,
            forward,
            backward,
            factor_bwd,
            adam_fact,
            adam_dense,
            moments,
            dense_moments,
            factor: [true; 4],
            train_dense: false,
            step_no: 0,
            cb_train: g.vk.reusable_command_buffer(),
            cb_eval: g.vk.reusable_command_buffer(),
            xfer: g.vk.reusable_command_buffer(),
        };
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
        for s in &self.forward {
            rec_step(vk, cb, s);
        }
        if train {
            for s in &self.backward {
                rec_step(vk, cb, s);
            }
            if self.train_dense {
                for (k, s) in self.adam_dense.iter().enumerate() {
                    if !self.factor[k] {
                        rec_step(vk, cb, s);
                    }
                }
            } else {
                for (k, steps) in self.factor_bwd.iter().enumerate() {
                    if !self.factor[k] {
                        continue;
                    }
                    for s in steps {
                        rec_step(vk, cb, s);
                    }
                    for s in &self.adam_fact[k] {
                        rec_step(vk, cb, s);
                    }
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
        (vk.copy_buffer)(cb, self.loss, self.rb_loss.buf, 1, &region);
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
        self.set_stage(g, [true; 4], false);
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
        self.put(g, fa.p[0], &hf.u);
        self.put(g, fa.p[1], &hf.vt);
        self.put(g, fa.p[2], &hf.s_post);
        self.put(g, fa.p[3], &hf.s_pre);
        for (j, pair) in self.moments[k].iter().enumerate() {
            let n = [fa.out * fa.rank, fa.inn * fa.rank, fa.out, fa.inn][j];
            let zero = vec![0f32; n];
            self.put(g, pair[0], &zero);
            self.put(g, pair[1], &zero);
        }
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
        self.put(g, fa.w, w);
        let zero = vec![0f32; fa.out * fa.inn];
        self.put(g, self.dense_moments[k][0], &zero);
        self.put(g, self.dense_moments[k][1], &zero);
    }

    /// Set the stage: `factor[k]` places slot `k`'s `W` in its latents (`true`) or in its `w`
    /// buffer (`false`), and `train_dense` selects the trained set — the dense slots' `W` or the
    /// factor slots' latents and scales. Re-records both tapes for that selection and touches no
    /// parameter.
    ///
    /// # Safety
    ///
    /// No tape is pending on the device (the last [`BlockTune::run`] returned).
    ///
    /// # Complexity
    ///
    /// `O(1)` host work; the re-recorded tapes carry `O(1)` dispatches.
    pub unsafe fn set_stage(&mut self, g: &GpuCtx, factor: [bool; 4], train_dense: bool) {
        self.factor = factor;
        self.train_dense = train_dense;
        self.record(g, true);
        self.record(g, false);
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
        self.put(g, self.x, x);
        self.put(g, self.a, a);
        self.put(g, self.yt, yt);
    }

    /// Record and run materialise + forward (+ backward and Adam when `train`); returns the step's
    /// summed squared error `Σ (y − y_t)²` before the update.
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
            let (b1t, rbc2) = adam_bias_corrections(self.step_no);
            *self.bias_stage.ptr = b1t;
            *self.bias_stage.ptr.add(1) = rbc2;
        }
        g.vk.submit_wait(if train { self.cb_train } else { self.cb_eval });
        std::slice::from_raw_parts(self.rb_loss.ptr, self.shape.t)
            .iter()
            .map(|&v| f64::from(v))
            .sum()
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
    unsafe fn grads(&self, g: &GpuCtx) -> Vec<[Vec<f32>; 4]> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::nn::GLSLC_ARGS;
    use std::path::Path;

    #[test]
    fn block_map_spirv_is_the_glsl() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv/train");
        let out = std::env::temp_dir().join(format!("block_map_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(dir.join("block_map.comp"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let fresh = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        assert!(
            fresh == BLOCK_MAP_SPV,
            "block_map.spv is stale: re-run glslc"
        );
    }

    /// A fixed-seed LCG value in `[-1, 1)`.
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

    /// `W = s_post ⊙ sign(U)·sign(V)ᵀ ⊙ s_pre` in f64.
    fn dense(f: &HostFactor) -> Vec<f64> {
        let sg = |v: f32| if v < 0.0 { -1.0 } else { 1.0 };
        let mut w = vec![0f64; f.out * f.inn];
        for r in 0..f.out {
            for c in 0..f.inn {
                let m: f64 = (0..f.rank)
                    .map(|k| sg(f.u[r * f.rank + k]) * sg(f.vt[c * f.rank + k]))
                    .sum();
                w[r * f.inn + c] = f64::from(f.s_post[r]) * m * f64::from(f.s_pre[c]);
            }
        }
        w
    }

    /// `y = x·wᵀ`, `x` is `n × i`, `w` is `o × i`.
    fn lin(x: &[f64], w: &[f64], n: usize, i: usize, o: usize) -> Vec<f64> {
        let mut y = vec![0f64; n * o];
        for r in 0..n {
            for c in 0..o {
                y[r * o + c] = (0..i).map(|k| x[r * i + k] * w[c * i + k]).sum();
            }
        }
        y
    }

    /// The host block loss `Σ (y − y_t)²` in f64 with every slot's dense `W` given explicitly.
    fn host_loss_w(
        s: BlockShape,
        w: &[Vec<f64>],
        g2: &[f32],
        x: &[f32],
        a: &[f32],
        yt: &[f32],
    ) -> f64 {
        let BlockShape { t, d, a: aw, f, .. } = s;
        let up = |v: &[f32]| v.iter().map(|&q| f64::from(q)).collect::<Vec<f64>>();
        let (x, a) = (up(x), up(a));
        let mut h = lin(&a, &w[0], t, aw, d);
        for (hv, xv) in h.iter_mut().zip(&x) {
            *hv += xv;
        }
        let mut n = vec![0f64; t * d];
        for r in 0..t {
            let ms: f64 = h[r * d..(r + 1) * d].iter().map(|v| v * v).sum::<f64>() / d as f64;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for c in 0..d {
                n[r * d + c] = h[r * d + c] * inv * f64::from(g2[c]);
            }
        }
        let gt = lin(&n, &w[1], t, d, f);
        let uu = lin(&n, &w[2], t, d, f);
        let m: Vec<f64> = gt
            .iter()
            .zip(&uu)
            .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
            .collect();
        let y = lin(&m, &w[3], t, f, d);
        y.iter()
            .zip(&h)
            .zip(yt)
            .map(|((yv, hv), tv)| {
                let e = yv + hv - f64::from(*tv);
                e * e
            })
            .sum()
    }

    /// The host block loss `Σ (y − y_t)²` in f64, the factors materialised.
    fn host_loss(
        s: BlockShape,
        fs: &[HostFactor],
        g2: &[f32],
        x: &[f32],
        a: &[f32],
        yt: &[f32],
    ) -> f64 {
        let w: Vec<Vec<f64>> = fs.iter().map(dense).collect();
        host_loss_w(s, &w, g2, x, a, yt)
    }

    /// The device loss equals the f64 host loss; the scale gradients equal the central
    /// differences of the host loss (the scales enter without the straight-through sign); and a
    /// short Adam run lowers the loss.
    #[test]
    fn block_tune_matches_host() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let s = BlockShape {
            t: 24,
            d: 32,
            a: 48,
            f: 64,
            ranks: [8, 8, 8, 8],
        };
        let fs: Vec<HostFactor> = (0..4)
            .map(|k| {
                let (o, i) = s.dims(k);
                factor(o, i, s.ranks[k], 10 * k as u64 + 1)
            })
            .collect();
        let g2: Vec<f32> = unit(s.d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
        let x = unit(s.t * s.d, 91);
        let a = unit(s.t * s.a, 92);
        let yt: Vec<f32> = unit(s.t * s.d, 93).iter().map(|v| 0.5 * v).collect();
        unsafe {
            let mut bt = BlockTune::new(&g, s, 1e-3, 1e-3, 1e-3);
            bt.load(&g, &fs, &g2);
            bt.batch(&g, &x, &a, &yt);
            let dev = bt.run(&g, false);
            let host = host_loss(s, &fs, &g2, &x, &a, &yt);
            println!("loss device {dev:.6} host {host:.6}");
            assert!(
                (dev - host).abs() <= 1e-4 * host.abs().max(1.0),
                "loss {dev} vs {host}"
            );

            let l0 = bt.run(&g, true);
            let gr = bt.grads(&g);
            let td = (s.t * s.d) as f64;
            let mut worst = 0f64;
            for k in 0..4 {
                for (j, len) in [(2usize, fs[k].out), (3, fs[k].inn)] {
                    for idx in [0, len / 2, len - 1] {
                        let eps = 1e-3f32;
                        let mut p = fs.clone();
                        let mut q = fs.clone();
                        if j == 2 {
                            p[k].s_post[idx] += eps;
                            q[k].s_post[idx] -= eps;
                        } else {
                            p[k].s_pre[idx] += eps;
                            q[k].s_pre[idx] -= eps;
                        }
                        let fd = (host_loss(s, &p, &g2, &x, &a, &yt)
                            - host_loss(s, &q, &g2, &x, &a, &yt))
                            / (2.0 * f64::from(eps))
                            / td;
                        let got = f64::from(gr[k][j][idx]);
                        let err = (got - fd).abs() / fd.abs().max(1e-4);
                        worst = worst.max(err);
                        assert!(
                            err < 0.02,
                            "factor {k} param {j} [{idx}]: device {got} fd {fd}"
                        );
                    }
                }
            }
            println!("scale gradient worst relative error {worst:.2e}");
            let mut last = l0;
            for _ in 0..60 {
                last = bt.run(&g, true);
            }
            println!("adam: loss {l0:.4} -> {last:.4}");
            assert!(
                last < 0.9 * l0,
                "60 Adam steps did not lower the loss: {l0} -> {last}"
            );
        }
    }

    /// No update without loss: with the target set to the block's own output, the training step
    /// computes loss exactly `0.0`, every latent and scale gradient exactly `0.0`, and Adam leaves
    /// every parameter bit-identical. The backward is the adjoint of the forward seeded by the
    /// computed residual `2/(t·d)·(y − yt)`; a loss term whose forward value is absorbed by
    /// rounding while its gradient is nonzero (the `(C + u) − C` construction of
    /// elonlit.com/scrivings/how-to-run-turing-complete-malware-in-pytorchs-backward-pass)
    /// would show here as a nonzero gradient at zero loss.
    #[test]
    fn zero_residual_seeds_zero_gradient() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let s = BlockShape {
            t: 24,
            d: 32,
            a: 48,
            f: 64,
            ranks: [8, 8, 8, 8],
        };
        let fs: Vec<HostFactor> = (0..4)
            .map(|k| {
                let (o, i) = s.dims(k);
                factor(o, i, s.ranks[k], 10 * k as u64 + 1)
            })
            .collect();
        let g2: Vec<f32> = unit(s.d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
        let x = unit(s.t * s.d, 91);
        let a = unit(s.t * s.a, 92);
        unsafe {
            let mut bt = BlockTune::new(&g, s, 1e-3, 1e-3, 1e-3);
            bt.load(&g, &fs, &g2);
            bt.batch(&g, &x, &a, &vec![0.0; s.t * s.d]);
            assert!(bt.run(&g, false) > 0.0);
            let y = bt.output(&g);
            bt.batch(&g, &x, &a, &y);
            let before = bt.factors(&g);
            let loss = bt.run(&g, true);
            assert_eq!(loss.to_bits(), 0f64.to_bits(), "loss {loss} at yt = y");
            for (k, gk) in bt.grads(&g).iter().enumerate() {
                for (j, gj) in gk.iter().enumerate() {
                    let nz = gj.iter().filter(|v| v.to_bits() != 0).count();
                    assert_eq!(nz, 0, "factor {k} param {j}: {nz} nonzero gradient words");
                }
            }
            let bits = |f: &[HostFactor]| -> Vec<u32> {
                f.iter()
                    .flat_map(|h| [&h.u, &h.vt, &h.s_post, &h.s_pre])
                    .flat_map(|v| v.iter().map(|w| w.to_bits()))
                    .collect()
            };
            assert_eq!(
                bits(&before),
                bits(&bt.factors(&g)),
                "a zero-loss step moved a parameter"
            );
        }
    }

    /// A dense slot trains through its own `dW` and moment pair: in the mixed state (slot 0 a
    /// factor, `gate`/`up`/`down` dense) one dense-stage step's `dW` of every dense slot equals the
    /// f64 host central difference of the block loss, every dense `W` moves and the factor slot's
    /// latents and scales stay bit-identical; then the factor stage moves the factor slot and
    /// leaves every dense `W` bit-identical.
    #[test]
    #[allow(clippy::too_many_lines, reason = "one tape comparison")]
    fn block_dense_matches_host() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        let s = BlockShape {
            t: 24,
            d: 32,
            a: 48,
            f: 64,
            ranks: [8, 8, 8, 8],
        };
        let fs: Vec<HostFactor> = (0..4)
            .map(|k| {
                let (o, i) = s.dims(k);
                factor(o, i, s.ranks[k], 10 * k as u64 + 1)
            })
            .collect();
        let g2: Vec<f32> = unit(s.d, 90).iter().map(|v| 1.0 + 0.2 * v).collect();
        let x = unit(s.t * s.d, 91);
        let a = unit(s.t * s.a, 92);
        let yt: Vec<f32> = unit(s.t * s.d, 93).iter().map(|v| 0.5 * v).collect();
        let dense_w: Vec<Vec<f32>> = (0..4)
            .map(|k| {
                let (o, i) = s.dims(k);
                unit(o * i, 10 * k as u64 + 5)
                    .iter()
                    .map(|&v| 0.3 * v)
                    .collect()
            })
            .collect();
        let words = |v: &[f32]| -> Vec<u32> { v.iter().map(|&q| q.to_bits()).collect() };
        unsafe {
            let mut bt = BlockTune::new(&g, s, 1e-3, 1e-3, 1e-3);
            bt.load(&g, &fs, &g2);
            for k in [1, 2, 3] {
                bt.load_dense(&g, k, &dense_w[k]);
            }
            bt.set_stage(&g, [true, false, false, false], true);
            bt.batch(&g, &x, &a, &yt);
            let _ = bt.run(&g, false);
            let before: Vec<Vec<f32>> = (0..4).map(|k| bt.weight(&g, k)).collect();
            let f_before = bt.factors(&g);
            bt.batch(&g, &x, &a, &yt);
            let _ = bt.run(&g, true);
            let after: Vec<Vec<f32>> = (0..4).map(|k| bt.weight(&g, k)).collect();
            for k in [1, 2, 3] {
                assert!(
                    words(&before[k]) != words(&after[k]),
                    "dense slot {k}: the dense stage left W unchanged"
                );
            }
            let f_bits = |f: &[HostFactor]| -> Vec<u32> {
                f.iter()
                    .flat_map(|h| [&h.u, &h.vt, &h.s_post, &h.s_pre])
                    .flat_map(|v| v.iter().map(|w| w.to_bits()))
                    .collect()
            };
            assert_eq!(
                f_bits(&f_before),
                f_bits(&bt.factors(&g)),
                "the dense stage moved the factor slot"
            );
            let w64: Vec<Vec<f64>> = before
                .iter()
                .map(|w| w.iter().map(|&v| f64::from(v)).collect())
                .collect();
            let td = (s.t * s.d) as f64;
            let mut worst = 0f64;
            for k in [1, 2, 3] {
                let (o, i) = s.dims(k);
                let dw = bt.dense_grad(&g, k);
                for r in [0, o / 2, o - 1] {
                    for c in [0, i / 2, i - 1] {
                        let eps = 1e-3;
                        let mut p = w64.clone();
                        p[k][r * i + c] += eps;
                        let mut q = w64.clone();
                        q[k][r * i + c] -= eps;
                        let fd = (host_loss_w(s, &p, &g2, &x, &a, &yt)
                            - host_loss_w(s, &q, &g2, &x, &a, &yt))
                            / (2.0 * eps)
                            / td;
                        let got = f64::from(dw[r * i + c]);
                        let err = (got - fd).abs() / fd.abs().max(1e-4);
                        worst = worst.max(err);
                        assert!(err < 0.02, "dense slot {k} [{r},{c}]: device {got} fd {fd}");
                    }
                }
            }
            println!("dense dW worst relative error {worst:.2e}");
            let frozen: Vec<Vec<f32>> = (0..4).map(|k| bt.weight(&g, k)).collect();
            let f_mid = bt.factors(&g);
            bt.set_stage(&g, [true, false, false, false], false);
            let _ = bt.run(&g, true);
            let kept: Vec<Vec<f32>> = (0..4).map(|k| bt.weight(&g, k)).collect();
            for k in [1, 2, 3] {
                assert_eq!(
                    words(&frozen[k]),
                    words(&kept[k]),
                    "dense slot {k}: the factor stage moved W"
                );
            }
            assert!(
                f_bits(&f_mid) != f_bits(&bt.factors(&g)),
                "the factor stage did not move the factor slot"
            );
        }
    }
}
