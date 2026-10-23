use crate::gpu::vulkan::*;

/// The compiled plane build.
pub(super) static LIMB_PLANE_SPV: &[u8] = include_bytes!("../spirv/limb_plane.spv");

/// Workgroup width of `limb_plane.comp`.
const PLANE_WG: usize = 256;
/// The largest workgroup count of one dispatch dimension.
const MAX_GROUPS_X: usize = 65_535;

/// The plane build of [`crate::engine::kern::limb`] on the device over fixed buffers: `x` holds the
/// `rows × n` words, `e` receives the `⌈n/32⌉` block exponents, `dig` the
/// `[row][block][limb][32]` digits and `flag` a nonzero word when a block exponent passes 62 (the
/// caller then builds on the CPU). Bit-identical to `Plane::build` when the flag is clear.
pub(crate) struct PlaneBuild {
    pipe: [Handle; 2],
    layout: [Handle; 2],
    dset: [Handle; 2],
    e: Handle,
    flag: Handle,
    blocks: usize,
    groups: (u32, u32),
}

impl PlaneBuild {
    /// The two passes' pipelines and descriptor sets over the given buffers.
    ///
    /// # Safety
    ///
    /// `vk` is live; `x` holds `rows·n` `i64`, `e` `⌈n/32⌉` `u32`, `dig` `rows·⌈n/32⌉·limbs·32`
    /// bytes, `flag` one `u32`; `e` and `flag` admit transfer writes.
    ///
    /// # Complexity
    ///
    /// Two pipeline builds and two 20-byte meta buffers.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "rows, n, blocks and limbs are below 2^32: n ≤ MAX_INNER and the caller's rows \
                  index a u32 plane"
    )]
    pub(crate) unsafe fn new(
        vk: &Vk,
        [x, e, dig, flag]: [Handle; 4],
        rows: usize,
        n: usize,
        limbs: u32,
    ) -> Self {
        let blocks = n.div_ceil(32);
        let host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
        let mut pipe = [0 as Handle; 2];
        let mut layout = [0 as Handle; 2];
        let mut dset = [0 as Handle; 2];
        for pass in 0..2u32 {
            let (meta, mem) = vk
                .buffer(20, BUFFER_USAGE_STORAGE, host)
                .expect("plane meta buffer");
            let words = [rows as u32, n as u32, blocks as u32, limbs, pass];
            std::ptr::copy_nonoverlapping(words.as_ptr(), vk.map(mem).cast(), 5);
            let (l, d) = vk.ssbo_set(&[x, e, dig, flag, meta]);
            pipe[pass as usize] = vk.pipeline(LIMB_PLANE_SPV, l);
            layout[pass as usize] = l;
            dset[pass as usize] = d;
        }
        let wg = (rows * blocks * 32).div_ceil(PLANE_WG);
        let gx = wg.clamp(1, MAX_GROUPS_X);
        Self {
            pipe,
            layout,
            dset,
            e,
            flag,
            blocks,
            groups: (gx as u32, wg.div_ceil(gx) as u32),
        }
    }

    /// Records the build into `cb`: `e ← 1`, `flag ← 0`, the exponent pass, then the digit pass.
    /// The caller orders its own reads of `e`, `dig` and `flag` after it.
    ///
    /// # Safety
    ///
    /// `cb` is recording on the device `self` was built on.
    ///
    /// # Complexity
    ///
    /// Two dispatches of `rows·⌈n/32⌉·32` invocations; `O(limbs)` digit steps each.
    pub(crate) unsafe fn record(&self, vk: &Vk, cb: CommandBuffer) {
        (vk.fill_buffer)(cb, self.e, 0, 4 * self.blocks as u64, 1);
        (vk.fill_buffer)(cb, self.flag, 0, 4, 0);
        vk.barrier(
            cb,
            STAGE_TRANSFER,
            STAGE_COMPUTE,
            ACCESS_TRANSFER_WRITE,
            ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
        );
        for pass in 0..2 {
            if pass == 1 {
                vk.barrier(
                    cb,
                    STAGE_COMPUTE,
                    STAGE_COMPUTE,
                    ACCESS_SHADER_WRITE,
                    ACCESS_SHADER_READ,
                );
            }
            (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, self.pipe[pass]);
            (vk.bind_sets)(
                cb,
                PIPELINE_BIND_POINT_COMPUTE,
                self.layout[pass],
                0,
                1,
                &self.dset[pass],
                0,
                std::ptr::null(),
            );
            (vk.dispatch)(cb, self.groups.0, self.groups.1, 1);
        }
    }
}
