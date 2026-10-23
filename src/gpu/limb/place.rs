use crate::gpu::vulkan::*;

/// The compiled placement.
pub(super) static LIMB_PLACE_SPV: &[u8] = include_bytes!("../spirv/limb_place.spv");

/// [`super::host_meta`] on the device over fixed buffers: `ea`, `eb` hold the two planes'
/// `⌈n/32⌉` block exponents (as `limb_plane.comp` leaves them), `meta` the
/// [`super::meta_head`] words followed by room for the shift table, which the kernel writes;
/// `out` receives `[range, fallback, Tmin_g per group]`. `range` nonzero is
/// `Layout::placement`'s `Range`, `fallback` nonzero is `host_meta`'s `None`; with both clear,
/// the shift table and `Tmin` equal `host_meta`'s.
pub(crate) struct Place {
    pipe: Handle,
    layout: Handle,
    dset: Handle,
    out: Handle,
    groups: u32,
}

impl Place {
    /// The pipeline and descriptor set over the given buffers.
    ///
    /// # Safety
    ///
    /// `vk` is live; `ea`, `eb` hold `⌈n/32⌉` `u32`, `meta` the head of `groups` pair groups
    /// and its shift table, `out` `2 + groups` `u32` and admits transfer writes.
    ///
    /// # Complexity
    ///
    /// One pipeline build.
    pub(crate) unsafe fn new(vk: &Vk, [ea, eb, meta, out]: [Handle; 4], groups: u32) -> Self {
        let (layout, dset) = vk.ssbo_set(&[ea, eb, meta, out]);
        Self {
            pipe: vk.pipeline(LIMB_PLACE_SPV, layout),
            layout,
            dset,
            out,
            groups,
        }
    }

    /// Records the placement into `cb`: both flags ← 0, then one workgroup per pair group. The
    /// caller orders the writes of `ea` and `eb` before it and its reads of `meta` and `out`
    /// after it.
    ///
    /// # Safety
    ///
    /// `cb` is recording on the device `self` was built on.
    ///
    /// # Complexity
    ///
    /// One dispatch of `64·G` invocations; `O(segments·⌈n/32⌉ / 64)` entries each, twice.
    pub(crate) unsafe fn record(&self, vk: &Vk, cb: CommandBuffer) {
        (vk.fill_buffer)(cb, self.out, 0, 8, 0);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_COMPUTE,
            ACCESS_TRANSFER_WRITE,
            ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
        );
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
        (vk.dispatch)(cb, self.groups, 1, 1);
    }
}
