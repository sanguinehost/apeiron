use crate::gpu::vulkan::*;

/// The compiled finish.
pub(super) static LIMB_FINISH_SPV: &[u8] = include_bytes!("../spirv/limb_finish.spv");

/// Invocations per workgroup of `limb_finish.comp`.
const LOCAL: usize = 256;
/// Workgroups along x; the rest go along y.
const GX: usize = 1 << 15;

/// [`super::FinishTask`] on the device over fixed buffers: `o` holds the lo/hi planes of `G`
/// pair groups as `limb_gemm.comp` leaves them, `meta` the [`super::meta_head`] words, `place`
/// `[range, fallback, Tmin_g per group]` as `limb_place.comp` leaves it; `x` receives the
/// `mp × pp` product, row stride `pp`, and `place`'s `range` is or-ed with 1 when a group word
/// or a sum leaves `i64`. With `range` and `fallback` clear, `x` restricted to `m × p` equals
/// [`crate::engine::kern::limb::gemm_q32`].
pub(crate) struct Finish {
    pipe: Handle,
    layout: Handle,
    dset: Handle,
    groups: (u32, u32),
}

impl Finish {
    /// The pipeline and descriptor set over the given buffers, for a `plane = mp·pp` product.
    ///
    /// # Safety
    ///
    /// `vk` is live; `o` holds `2·G·plane` `i32`, `meta` the head of `G` pair groups, `place`
    /// `2 + G` `u32`, `x` `plane` `i64`; `plane` is a multiple of 256.
    ///
    /// # Complexity
    ///
    /// One pipeline build.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "plane ≤ 2^32 words by the buffer sizes, so both workgroup counts fit u32"
    )]
    pub(crate) unsafe fn new(vk: &Vk, [o, meta, place, x]: [Handle; 4], plane: usize) -> Self {
        let (layout, dset) = vk.ssbo_set(&[o, meta, place, x]);
        let total = plane.div_ceil(LOCAL);
        let gx = total.min(GX);
        Self {
            pipe: vk.pipeline(LIMB_FINISH_SPV, layout),
            layout,
            dset,
            groups: (gx as u32, total.div_ceil(gx) as u32),
        }
    }

    /// Records the finish into `cb`. The caller orders the writes of `o` and `place` before it
    /// and its reads of `x` and `place` after it.
    ///
    /// # Safety
    ///
    /// `cb` is recording on the device `self` was built on.
    ///
    /// # Complexity
    ///
    /// One dispatch of `plane` invocations, `O(G)` 64-bit operations each.
    pub(crate) unsafe fn record(&self, vk: &Vk, cb: CommandBuffer) {
        (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, self.pipe);
        (vk.bind_sets)(
            cb,
            PIPELINE_BIND_POINT_COMPUTE,
            self.layout,
            0,
            1,
            &self.dset,
            0,
            std::ptr::null(),
        );
        (vk.dispatch)(cb, self.groups.0, self.groups.1, 1);
    }
}
