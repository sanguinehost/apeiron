//! The per-slot factor machinery of the block fit: one weight held as
//! `W = diag(s_post)·sign(U)·sign(V)ᵀ·diag(s_pre)`, with its materialise tape, its adjoint tape,
//! Adam over its four parameters and its dense-`W` alternative.
//!
//! The factor form is the reference's forward,
//! `y = ((x ⊙ scale_pre)·sign(V)ᵀ)·sign(U)ᵀ ⊙ scale_post` (`SamsungLabs/NanoQuant` @ `a9e0a43`,
//! `src/nanoquant/modules/linear.py:248-257`, `_compute_forward`: `F.linear(x * scale_pre, Vq)`,
//! then `F.linear(y, Uq)`, then `y * scale_post`).
//!
//! Every builder is a function of one [`Factor`] and its Adam moments, so a slot the block tape does
//! not carry yet — the attention front's `q`, `k`, `v` — reuses it unchanged. Consumer:
//! [`BlockTune`](super::block::BlockTune).

use crate::gpu::decode::{GpuCtx, Step};
use crate::gpu::nn::fwd::linear_fwd_dispatch;
use crate::gpu::nn::{
    adam_update_dispatch_reused, linear_bwd_input_dispatch, linear_bwd_weight_dispatch,
};
use crate::gpu::vulkan::*;

use super::block::{upload, HostBuffer, HostFactor, BLOCK_MAP_SPV};

const LOCAL: usize = 256;
const MAX_GROUPS_X: usize = 65_535;

/// One factor's device buffers: `u` the `out × rank` latent, `vt` the `inn × rank` latent (the
/// reference's `V` transposed), `s_post` and `s_pre` the two scale vectors.
pub(super) struct Factor {
    pub(super) out: usize,
    pub(super) inn: usize,
    pub(super) rank: usize,
    /// Latents and scales, in the Adam order `[u, vt, s_post, s_pre]`, with their gradients and
    /// moments.
    pub(super) p: [Handle; 4],
    pub(super) grad: [Handle; 4],
    pub(super) us: Handle,
    pub(super) vts: Handle,
    pub(super) m: Handle,
    pub(super) w: Handle,
    pub(super) dw: Handle,
    pub(super) dm: Handle,
}

impl Factor {
    /// Allocate one slot's eleven work buffers and its four `[m, v]` moment pairs.
    ///
    /// # Safety
    ///
    /// `g` is a live device context that outlives the value.
    ///
    /// # Complexity
    ///
    /// `O(1)` device allocations; `O((out + inn)·rank + out·inn)` device words held.
    pub(super) unsafe fn new(
        g: &GpuCtx,
        out: usize,
        inn: usize,
        rank: usize,
    ) -> (Self, [[Handle; 2]; 4]) {
        let dev = |n: usize| g.dev(n * 4);
        let sizes = [out * rank, inn * rank, out, inn];
        (
            Self {
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
            },
            sizes.map(|n| [dev(n), dev(n)]),
        )
    }
}

/// Map `mode` of [`BLOCK_MAP_SPV`] over `[rows, cols, total]` with bindings `bufs` (1..=4).
///
/// # Safety
///
/// The buffers hold what the mode reads and writes.
///
/// # Complexity
///
/// `O(1)` host work; the dispatch covers `total` (modes 0, 1, 3, 4) or `rows` (modes 2, 5)
/// invocations.
#[allow(
    clippy::cast_possible_truncation,
    reason = "rows, cols and totals index u32 invocations"
)]
pub(super) unsafe fn map(
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

/// The materialise tape of `fa`: `sign_u`, `sign_v`, `M = sign(U)·sign(V)ᵀ`, then the two scales
/// into `W`.
///
/// # Safety
///
/// `fa` is live and its `p` buffers hold its latents and scales.
///
/// # Complexity
///
/// `O(out·rank + inn·rank + out·inn·rank)` device multiply-adds per step.
pub(super) unsafe fn materialise(g: &GpuCtx, fa: &Factor) -> Vec<Step> {
    let (out, inn, r) = (fa.out, fa.inn, fa.rank);
    vec![
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
    ]
}

/// The adjoint tape of `fa` over its materialised `W`: `dM = dW ⊙ s_post·s_preᵀ`, the two scale
/// gradients, then `dU = dM·sign(V)` and `dV = dMᵀ·sign(U)` (the sign's straight-through derivative
/// is the identity).
///
/// # Safety
///
/// `fa` is live and its `dw` buffer holds the slot's dense gradient.
///
/// # Complexity
///
/// `O(out·inn·rank)` device multiply-adds; the row and column reductions are `O(out·inn)`.
pub(super) unsafe fn factor_bwd(g: &GpuCtx, fa: &Factor) -> Vec<Step> {
    let (out, inn, r) = (fa.out, fa.inn, fa.rank);
    vec![
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
    ]
}

/// Adam over `fa`'s four parameters against the moment pairs `mo`, at `lr` for the two latents and
/// `lr_scale` for the two scale vectors (both Q32.32).
///
/// # Safety
///
/// `mo` holds `fa`'s lengths and `bias_buf` the step's two bias-correction words.
///
/// # Complexity
///
/// `O(Σ out·inn·rank + out + inn)` device words per step.
pub(super) unsafe fn adam_factor(
    g: &GpuCtx,
    fa: &Factor,
    mo: &[[Handle; 2]; 4],
    bias_buf: Handle,
    lr: i64,
    lr_scale: i64,
) -> Vec<Step> {
    (0..4)
        .map(|j| {
            let n = [fa.out * fa.rank, fa.inn * fa.rank, fa.out, fa.inn][j];
            let rate = if j < 2 { lr } else { lr_scale };
            adam_update_dispatch_reused(
                g, fa.p[j], fa.grad[j], mo[j][0], mo[j][1], bias_buf, n as u32, rate,
            )
        })
        .collect()
}

/// Allocate slot `fa`'s dense moment pair over `out × inn` and build the Adam step over its dense
/// `W` and `dW`, at `lr_dense` (Q32.32).
///
/// # Safety
///
/// `g` is a live device context; `bias_buf` holds the step's two bias-correction words.
///
/// # Complexity
///
/// `O(out·inn)` device allocations and words updated per step.
pub(super) unsafe fn adam_dense(
    g: &GpuCtx,
    fa: &Factor,
    bias_buf: Handle,
    lr_dense: i64,
) -> (Step, [Handle; 2]) {
    let n = fa.out * fa.inn;
    let dmo = [g.dev(n * 4), g.dev(n * 4)];
    let step =
        adam_update_dispatch_reused(g, fa.w, fa.dw, dmo[0], dmo[1], bias_buf, n as u32, lr_dense);
    (step, dmo)
}

/// A real learning rate as Q32.32, the integer form [`adam_update_dispatch_reused`] takes.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the Adam rate is Q32.32 and the product fits i64 for every rate used"
)]
pub(super) fn q32(v: f64) -> i64 {
    (v * 4_294_967_296.0).round() as i64
}

/// Put `hf`'s latents and scales into `fa` and zero its four moment pairs in `mo`, one copy per
/// parameter in the Adam order.
///
/// # Safety
///
/// `hf` matches `fa`'s dimensions and rank; `stage` holds the largest slice; `cb` is not pending.
///
/// # Complexity
///
/// `O((out + inn)·rank + out + inn)` words copied, each through one submission.
pub(super) unsafe fn upload_factor(
    g: &GpuCtx,
    cb: CommandBuffer,
    stage: &HostBuffer,
    fa: &Factor,
    hf: &HostFactor,
    mo: &[[Handle; 2]; 4],
) {
    upload(g, cb, stage, fa.p[0], &hf.u);
    upload(g, cb, stage, fa.p[1], &hf.vt);
    upload(g, cb, stage, fa.p[2], &hf.s_post);
    upload(g, cb, stage, fa.p[3], &hf.s_pre);
    for (j, pair) in mo.iter().enumerate() {
        let n = [fa.out * fa.rank, fa.inn * fa.rank, fa.out, fa.inn][j];
        let zero = vec![0f32; n];
        upload(g, cb, stage, pair[0], &zero);
        upload(g, cb, stage, pair[1], &zero);
    }
}

/// Put the dense `W` `w` into `fa`'s `w` buffer and zero the dense moment pair `dmo`.
///
/// # Safety
///
/// `w` has `fa.out·inn` words; `stage` holds the largest slice; `cb` is not pending.
///
/// # Complexity
///
/// `O(out·inn)` words copied, three times.
pub(super) unsafe fn upload_dense(
    g: &GpuCtx,
    cb: CommandBuffer,
    stage: &HostBuffer,
    fa: &Factor,
    w: &[f32],
    dmo: &[Handle; 2],
) {
    upload(g, cb, stage, fa.w, w);
    let zero = vec![0f32; fa.out * fa.inn];
    upload(g, cb, stage, dmo[0], &zero);
    upload(g, cb, stage, dmo[1], &zero);
}
