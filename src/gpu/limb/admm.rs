//! `learn::nanoquant::x_update` resident on the device, bit for bit.
//!
//! **Contract.** [`Side::x_update`] returns `Some(Ok(Fᵀ))` equal to `x_update` on the same operands
//! in every bit, `Some(Err(e))` when the host inverse refuses `hw` (the CPU builds the same `hw`
//! and refuses it the same way), and `None` when a device stage flags a refusal or a fallback: a
//! plane exponent past 62, a placement `Range` or accumulator bound, a finish word outside `i64`,
//! or a map value off the grid. On `None` the caller runs the CPU `x_update`, which computes the
//! value or reproduces the refusal.
//!
//! **Route.** One [`Frame`] per `(k, n, r, limbs)` holds the target's limb plane, the iterate
//! buffers, the three limb products (`Z_xᵀZ_x`, `W·Z_x`, `Rᵀ·H⁻¹`, each placement → `limb_gemm` →
//! finish) and the maps of `src/gpu/spirv/admm_map.comp` between them, recorded once into two
//! command buffers around the host inverse: `cb1` uploads `Z_x`, `Z`, `U` and runs every stage up to
//! `hw` and `R`'s plane; the host inverts `hw`; `cb2` uploads `H⁻¹` and runs `Rᵀ·H⁻¹` and the read
//! map. The device mutex is held for each submit only, so the inverse's own products reach it.
//!
//! **Buffers.** No device buffer is ever freed, so frames are kept in [`FRAMES`] and checked out
//! by a [`Side`] for one factorisation; two concurrent factorisations of one shape hold two frames.

use std::sync::{Mutex, PoisonError};

use super::finish::Finish;
use super::place::Place;
use super::plane::PlaneBuild;
use super::{device, meta_head, segments, LIMB_GEMM_SPV, TILE};
use crate::engine::kern::limb::{device_pairs, MAX_INNER};
use crate::gpu::vulkan::*;

/// The compiled maps.
pub(super) static ADMM_MAP_SPV: &[u8] = include_bytes!("../spirv/admm_map.spv");

/// Invocations per workgroup of `admm_map.comp`.
const LOCAL: usize = 256;
/// The largest workgroup count of one dispatch dimension.
const MAX_GROUPS_X: usize = 65_535;
/// Usage of every device buffer: storage, filled and copied both ways.
const DEV_USAGE: u32 = BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST;

/// Readback flag words (`u32` index): the four plane flags, the `[range, fallback]` of the three
/// products, and the map flag after each command buffer.
const W_PLANE: usize = 0;
const ZT_PLANE: usize = 1;
const RT_PLANE: usize = 2;
const HI_PLANE: usize = 3;
const GRAM: usize = 4;
const WZ: usize = 6;
const F: usize = 8;
const MAP1: usize = 10;
const MAP2: usize = 11;
/// Bytes of the flag block at the head of the readback.
const FLAG_BYTES: usize = 64;

/// A device-local buffer of at least `bytes`.
///
/// # Complexity
///
/// One allocation.
pub(super) unsafe fn dev_buf(vk: &Vk, bytes: usize) -> Option<Handle> {
    vk.buffer(bytes.max(16) as u64, DEV_USAGE, MEMORY_DEVICE_LOCAL)
        .map(|b| b.0)
}

/// Every compute and transfer write before every later compute and transfer access.
pub(super) unsafe fn full(vk: &Vk, cb: CommandBuffer) {
    vk.barrier(
        cb,
        STAGE_COMPUTE | STAGE_TRANSFER,
        STAGE_COMPUTE | STAGE_TRANSFER,
        ACCESS_SHADER_WRITE | ACCESS_TRANSFER_WRITE,
        ACCESS_SHADER_READ | ACCESS_SHADER_WRITE | ACCESS_TRANSFER_READ | ACCESS_TRANSFER_WRITE,
    );
}

/// Every transfer write before the host reads.
pub(super) unsafe fn to_host(vk: &Vk, cb: CommandBuffer) {
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
}

/// Records a copy of `size` bytes from `src` at `so` to `dst` at `dO`.
pub(super) unsafe fn copy(
    vk: &Vk,
    cb: CommandBuffer,
    src: Handle,
    so: usize,
    dst: Handle,
    d_o: usize,
    size: usize,
) {
    let region = BufferCopy {
        src_offset: so as u64,
        dst_offset: d_o as u64,
        size: size as u64,
    };
    (vk.copy_buffer)(cb, src, dst, 1, &region);
}

/// Binds `pipe` with `dset` and dispatches `groups`.
unsafe fn run(
    vk: &Vk,
    cb: CommandBuffer,
    pipe: Handle,
    layout: Handle,
    dset: Handle,
    groups: [u32; 3],
) {
    (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, pipe);
    (vk.bind_sets)(
        cb,
        PIPELINE_BIND_POINT_COMPUTE,
        layout,
        0,
        1,
        &dset,
        0,
        std::ptr::null(),
    );
    (vk.dispatch)(cb, groups[0], groups[1], groups[2]);
}

/// The bytes of `v`.
pub(super) fn bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_ne_bytes()).collect()
}

/// One limb product `a·bᵀ` over fixed planes: the placement into `out = [range, fallback, Tmin_g…]`,
/// `limb_gemm.comp` into the shared `o`, and the finish into `x` (row stride `pp`).
pub(super) struct Product {
    place: Place,
    pipe: Handle,
    layout: Handle,
    dset: Handle,
    groups: [u32; 3],
    finish: Finish,
    pub(super) out: Handle,
}

impl Product {
    /// Bytes of `o` an `m × p` product over inner length `inner` writes: two `i32` planes per pair
    /// group.
    ///
    /// # Complexity
    ///
    /// `O(limbs²)`.
    pub(super) fn o_bytes(m: usize, p: usize, inner: usize, limbs: u32) -> usize {
        let (pairs, group) = device_pairs(inner, limbs);
        let ngroups = pairs.len().div_ceil(group);
        2 * ngroups * m.div_ceil(TILE) * TILE * p.div_ceil(TILE) * TILE * 4
    }

    /// The product of the planes `a = [dig, e]` (`m` rows) and `b` (`p` rows) over `inner`; the
    /// meta head is queued on `ups` for upload.
    ///
    /// # Safety
    ///
    /// `vk` is live; the planes are `limb_plane.comp`'s layout with rows padded to 64 and zero past
    /// the row count; `o` holds [`Product::o_bytes`], `x` `mp·pp` `i64`.
    ///
    /// # Complexity
    ///
    /// Three pipeline builds and `O(P·⌈inner/32⌉)` meta words.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::too_many_arguments,
        reason = "group and tile counts are far below 2^32; the operands are distinct buffers"
    )]
    pub(super) unsafe fn new(
        vk: &Vk,
        a: [Handle; 2],
        b: [Handle; 2],
        (m, p, inner, limbs): (usize, usize, usize, u32),
        o: Handle,
        x: Handle,
        ups: &mut Vec<(Handle, Vec<u8>)>,
    ) -> Option<Self> {
        let (pairs, group) = device_pairs(inner, limbs);
        let blocks = inner.div_ceil(32);
        let head = meta_head(&pairs, group, blocks, limbs, m, p);
        let (groups, segs) = segments(&pairs, group);
        let meta = dev_buf(vk, 4 * (head.len() + segs.len() * blocks))?;
        let out = dev_buf(vk, 4 * (2 + groups.len()))?;
        ups.push((meta, bytes(&head)));
        let place = Place::new(vk, [a[1], b[1], meta, out], groups.len() as u32);
        let (layout, dset) = vk.ssbo_set(&[a[0], b[0], o, meta]);
        let pipe = vk.pipeline(LIMB_GEMM_SPV, layout);
        let (mp, pp) = (m.div_ceil(TILE) * TILE, p.div_ceil(TILE) * TILE);
        Some(Self {
            place,
            pipe,
            layout,
            dset,
            groups: [(mp / TILE) as u32, (pp / TILE) as u32, groups.len() as u32],
            finish: Finish::new(vk, [o, meta, out, x], mp * pp),
            out,
        })
    }

    /// Records placement, product and finish, each ordered after the one before.
    ///
    /// # Complexity
    ///
    /// Three dispatches; device work as [`super::LimbGemm::run`].
    pub(super) unsafe fn record(&self, vk: &Vk, cb: CommandBuffer) {
        self.place.record(vk, cb);
        full(vk, cb);
        run(vk, cb, self.pipe, self.layout, self.dset, self.groups);
        full(vk, cb);
        self.finish.record(vk, cb);
    }
}

/// One mode of a map shader (`admm_map.comp`, `rank1_map.comp`: meta plus eight bindings) over its
/// bound buffers.
pub(super) struct Map {
    pipe: Handle,
    layout: Handle,
    dset: Handle,
    groups: [u32; 3],
}

impl Map {
    /// The map `mode` of the shader `spv` over `total` invocations; `bufs` are its bindings 1–8, the
    /// meta words are queued on `ups`.
    ///
    /// # Safety
    ///
    /// `vk` is live and the buffers hold what `spv` reads and writes in `mode`.
    ///
    /// # Complexity
    ///
    /// One pipeline build.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "r, cols, rp and total index u32 invocations"
    )]
    pub(super) unsafe fn new(
        vk: &Vk,
        spv: &[u8],
        [mode, r, cols, rp, total]: [usize; 5],
        bufs: [Handle; 8],
        ups: &mut Vec<(Handle, Vec<u8>)>,
    ) -> Option<Self> {
        let meta = dev_buf(vk, 20)?;
        ups.push((meta, bytes(&[mode, r, cols, rp, total].map(|w| w as u32))));
        let mut set = vec![meta];
        set.extend_from_slice(&bufs);
        let (layout, dset) = vk.ssbo_set(&set);
        let wg = total.div_ceil(LOCAL);
        let gx = wg.clamp(1, MAX_GROUPS_X);
        Some(Self {
            pipe: vk.pipeline(spv, layout),
            layout,
            dset,
            groups: [gx as u32, wg.div_ceil(gx).max(1) as u32, 1],
        })
    }

    pub(super) unsafe fn record(&self, vk: &Vk, cb: CommandBuffer) {
        run(vk, cb, self.pipe, self.layout, self.dset, self.groups);
    }
}

/// The device frame of one `(k, n, r, limbs)`: recorded command buffers over buffers that live as
/// long as the device. `stage` holds `[rho, reg]`, `Z_x`, `Z`, `U`, `H⁻¹` at [`Frame::off`] (and the
/// target's words at 0 for `cbw`); `back` the flag block, `hw` and `Fᵀ`.
pub(crate) struct Frame {
    pub(super) k: usize,
    n: usize,
    pub(super) r: usize,
    /// The resident `rank1` over `k × r`, built on its first call.
    pub(super) r1: Option<Box<super::rank1::Rank1>>,
    stage: *mut u8,
    back: *const u8,
    /// Upload of the target's words and its plane build.
    cbw: CommandBuffer,
    cb1: CommandBuffer,
    cb2: CommandBuffer,
}

// SAFETY: a frame is reached by one `Side` at a time (checked out of `FRAMES` under its mutex), and
// every submit of its command buffers holds the device mutex, which is the external synchronisation
// Vulkan requires; the mapped pointers are written and read only by the frame's holder.
unsafe impl Send for Frame {}

impl Frame {
    /// Staging byte offsets of `Z_x`, `Z`, `U`, `H⁻¹` (`[rho, reg]` at 0), and the staging size.
    fn off(&self) -> [usize; 5] {
        let (k, n, r) = (self.k, self.n, self.r);
        let zx = 16;
        let z = zx + 8 * n * r;
        let du = z + 8 * k * r;
        let hi = du + 8 * k * r;
        [zx, z, du, hi, hi + 8 * r * r]
    }

    /// The flag words of the last submits.
    unsafe fn flags(&self) -> &[u32] {
        std::slice::from_raw_parts(self.back.cast::<u32>(), FLAG_BYTES / 4)
    }

    /// Builds every buffer, pipeline and command buffer of the frame and runs its initialisation
    /// (zero digit planes, meta uploads); `None` when an allocation fails.
    ///
    /// # Safety
    ///
    /// `vk` is live with cooperative matrices and binary64, and its mutex is held.
    ///
    /// # Complexity
    ///
    /// `O(1)` allocations and pipeline builds; device memory `O(k·n + (k + n + r)·r)` words.
    #[allow(
        clippy::too_many_lines,
        reason = "one frame's buffer graph, recorded in order"
    )]
    unsafe fn build(vk: &Vk, k: usize, n: usize, r: usize, limbs: u32) -> Option<Self> {
        let (kp, rp) = (k.div_ceil(TILE) * TILE, r.div_ceil(TILE) * TILE);
        let (bn, br) = (n.div_ceil(32), r.div_ceil(32));
        let lb = limbs as usize * 32;
        let plane = |rows: usize, padded: usize, blocks: usize| -> Option<[Handle; 4]> {
            Some([
                dev_buf(vk, 8 * rows * blocks * 32)?,
                dev_buf(vk, 4 * blocks)?,
                dev_buf(vk, padded * blocks * lb)?,
                dev_buf(vk, 4)?,
            ])
        };
        // [words, e, dig, flag]; the words buffer holds `rows × inner` (blocks·32 ≥ inner).
        let w = plane(k, kp, bn)?;
        let zt = plane(r, rp, bn)?;
        let rt = plane(k, kp, br)?;
        let hi = plane(r, rp, br)?;
        let (zx, z, du, prm) = (
            dev_buf(vk, 8 * n * r)?,
            dev_buf(vk, 8 * k * r)?,
            dev_buf(vk, 8 * k * r)?,
            dev_buf(vk, 16)?,
        );
        let (hw, d, out, mflag) = (
            dev_buf(vk, 8 * r * r)?,
            dev_buf(vk, 8 * (r + 1))?,
            dev_buf(vk, 8 * k * r)?,
            dev_buf(vk, 4)?,
        );
        let (gram_x, kx) = (dev_buf(vk, 8 * rp * rp)?, dev_buf(vk, 8 * kp * rp)?);
        let o = dev_buf(
            vk,
            Product::o_bytes(r, r, n, limbs)
                .max(Product::o_bytes(k, r, n, limbs))
                .max(Product::o_bytes(k, r, r, limbs)),
        )?;
        let mut ups: Vec<(Handle, Vec<u8>)> = Vec::new();
        let gram = Product::new(
            vk,
            [zt[2], zt[1]],
            [zt[2], zt[1]],
            (r, r, n, limbs),
            o,
            gram_x,
            &mut ups,
        )?;
        let wz = Product::new(
            vk,
            [w[2], w[1]],
            [zt[2], zt[1]],
            (k, r, n, limbs),
            o,
            kx,
            &mut ups,
        )?;
        let f = Product::new(
            vk,
            [rt[2], rt[1]],
            [hi[2], hi[1]],
            (k, r, r, limbs),
            o,
            kx,
            &mut ups,
        )?;
        // Bindings 1–8: prm, fin, du, prod, d, wout, fout, flag.
        let map = |mode: usize,
                   cols: usize,
                   total: usize,
                   fin: Handle,
                   prod: Handle,
                   wout: Handle,
                   ups: &mut Vec<_>| {
            Map::new(
                vk,
                ADMM_MAP_SPV,
                [mode, r, cols, rp, total],
                [prm, fin, du, prod, d, wout, out, mflag],
                ups,
            )
        };
        let maps = [
            map(0, n, r * n, zx, gram_x, zt[0], &mut ups)?,
            map(1, 1, r, z, gram_x, hw, &mut ups)?,
            map(2, 1, 1, z, gram_x, hw, &mut ups)?,
            map(3, r, r * r, z, gram_x, hw, &mut ups)?,
            map(4, r, k * r, z, kx, rt[0], &mut ups)?,
            map(5, r, k * r, z, kx, rt[0], &mut ups)?,
        ];
        let [w_plane, zt_plane, rt_plane, hi_plane] =
            [(w, k, n), (zt, r, n), (rt, k, r), (hi, r, r)]
                .map(|(b, rows, inner)| PlaneBuild::new(vk, b, rows, inner, limbs));
        let up_bytes: usize = ups.iter().map(|u| u.1.len()).sum();
        let mut me = Self {
            k,
            n,
            r,
            r1: None,
            stage: std::ptr::null_mut(),
            back: std::ptr::null(),
            cbw: std::ptr::null_mut(),
            cb1: std::ptr::null_mut(),
            cb2: std::ptr::null_mut(),
        };
        let off = me.off();
        let stage_bytes = off[4].max(8 * k * n).max(up_bytes);
        let (stage, stage_mem) = vk.buffer(
            stage_bytes as u64,
            BUFFER_USAGE_TRANSFER_SRC,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )?;
        let back_bytes = FLAG_BYTES + 8 * r * r + 8 * k * r;
        let (back, back_mem) = vk
            .buffer(
                back_bytes as u64,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT | MEMORY_HOST_CACHED,
            )
            .or_else(|| {
                vk.buffer(
                    back_bytes as u64,
                    BUFFER_USAGE_TRANSFER_DST,
                    MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
                )
            })?;
        me.stage = vk.map(stage_mem).cast();
        me.back = vk.map(back_mem).cast_const().cast();
        let (o_hw, o_out) = (FLAG_BYTES, FLAG_BYTES + 8 * r * r);
        let flag = |cb, src: Handle, slot: usize, words: usize| {
            copy(vk, cb, src, 0, back, 4 * slot, 4 * words)
        };

        // Initialisation: zero digit planes (rows past the count stay zero) and the meta words.
        let cb = vk.command_buffer();
        vk.begin(cb);
        for p in [w, zt, rt, hi] {
            (vk.fill_buffer)(cb, p[2], 0, WHOLE_SIZE, 0);
        }
        let mut at = 0usize;
        for (dst, b) in &ups {
            std::ptr::copy_nonoverlapping(b.as_ptr(), me.stage.add(at), b.len());
            copy(vk, cb, stage, at, *dst, 0, b.len());
            at += b.len();
        }
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);

        // cbw: the target's words and plane.
        me.cbw = vk.command_buffer();
        let cb = me.cbw;
        vk.begin(cb);
        copy(vk, cb, stage, 0, w[0], 0, 8 * k * n);
        full(vk, cb);
        w_plane.record(vk, cb);
        full(vk, cb);
        flag(cb, w[3], W_PLANE, 1);
        to_host(vk, cb);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");

        // cb1: inputs, Z_xᵀ and its plane, the Gram, d, stab, hw, W·Z_x, R and its plane.
        me.cb1 = vk.command_buffer();
        let cb = me.cb1;
        vk.begin(cb);
        copy(vk, cb, stage, 0, prm, 0, 16);
        copy(vk, cb, stage, off[0], zx, 0, 8 * n * r);
        copy(vk, cb, stage, off[1], z, 0, 8 * k * r);
        copy(vk, cb, stage, off[2], du, 0, 8 * k * r);
        (vk.fill_buffer)(cb, mflag, 0, 4, 0);
        full(vk, cb);
        maps[0].record(vk, cb);
        full(vk, cb);
        zt_plane.record(vk, cb);
        full(vk, cb);
        gram.record(vk, cb);
        for m in &maps[1..4] {
            full(vk, cb);
            m.record(vk, cb);
        }
        full(vk, cb);
        wz.record(vk, cb);
        full(vk, cb);
        maps[4].record(vk, cb);
        full(vk, cb);
        rt_plane.record(vk, cb);
        full(vk, cb);
        copy(vk, cb, hw, 0, back, o_hw, 8 * r * r);
        flag(cb, zt[3], ZT_PLANE, 1);
        flag(cb, rt[3], RT_PLANE, 1);
        flag(cb, gram.out, GRAM, 2);
        flag(cb, wz.out, WZ, 2);
        flag(cb, mflag, MAP1, 1);
        to_host(vk, cb);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");

        // cb2: H⁻¹ and its plane, Rᵀ·H⁻¹, the read map.
        me.cb2 = vk.command_buffer();
        let cb = me.cb2;
        vk.begin(cb);
        copy(vk, cb, stage, off[3], hi[0], 0, 8 * r * r);
        (vk.fill_buffer)(cb, mflag, 0, 4, 0);
        full(vk, cb);
        hi_plane.record(vk, cb);
        full(vk, cb);
        f.record(vk, cb);
        full(vk, cb);
        maps[5].record(vk, cb);
        full(vk, cb);
        copy(vk, cb, out, 0, back, o_out, 8 * k * r);
        flag(cb, hi[3], HI_PLANE, 1);
        flag(cb, f.out, F, 2);
        flag(cb, mflag, MAP2, 1);
        to_host(vk, cb);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        Some(me)
    }
}

/// Frames not checked out, by `[k, n, r, limbs]`.
static FRAMES: Mutex<Vec<([usize; 4], Box<Frame>)>> = Mutex::new(Vec::new());

/// One side of a factorisation (`A` over `W`, or `B` over `Wᵀ`) holding its frame, with the
/// target's plane built; the frame returns to [`FRAMES`] on drop.
pub(crate) struct Side {
    pub(super) key: [usize; 4],
    pub(super) frame: Option<Box<Frame>>,
}

impl Side {
    /// The frame of the `k × n` target words `w` at rank `r`, with `w`'s plane built on the device;
    /// `None` when no device with binary64 loads, `APEIRON_ADMM_DEVICE=0`, a dimension is zero or
    /// past `MAX_INNER`, `w` is not `k·n` words, or the plane build flags (the CPU refuses or runs
    /// the step).
    ///
    /// # Complexity
    ///
    /// `O(k·n)` bytes staged and one submit; a frame build on the first call per shape.
    pub(crate) fn new(w: &[i64], k: usize, n: usize, r: usize, limbs: u32) -> Option<Self> {
        if std::env::var("APEIRON_ADMM_DEVICE").is_ok_and(|v| v.trim() == "0") {
            return None;
        }
        if k == 0 || n == 0 || r == 0 || n > MAX_INNER || r > MAX_INNER || w.len() != k * n {
            return None;
        }
        let key = [k, n, r, limbs as usize];
        let dev = device()?;
        let cached = {
            let mut fr = FRAMES.lock().unwrap_or_else(PoisonError::into_inner);
            fr.iter()
                .position(|e| e.0 == key)
                .map(|i| fr.swap_remove(i).1)
        };
        let guard = dev.lock().unwrap_or_else(PoisonError::into_inner);
        let vk = &guard.vk;
        if !vk.has_f64 {
            return None;
        }
        // SAFETY: the device mutex is held; the frame was built on this device and is ours alone.
        let frame = match cached {
            Some(f) => f,
            None => Box::new(unsafe { Frame::build(vk, k, n, r, limbs) }?),
        };
        unsafe {
            std::ptr::copy_nonoverlapping(w.as_ptr().cast::<u8>(), frame.stage, 8 * w.len());
            vk.submit_wait(frame.cbw);
        }
        drop(guard);
        let side = Self {
            key,
            frame: Some(frame),
        };
        let clear = side
            .frame
            .as_deref()
            .is_some_and(|f| unsafe { f.flags()[W_PLANE] } == 0);
        clear.then_some(side)
    }

    /// `x_update(zx, n, r, w, k, z, du, rho, reg)` on the device, `inv` the host inverse of `hw`
    /// (`spd_inverse`). See the module contract for the three outcomes.
    ///
    /// # Complexity
    ///
    /// Host: `O((n + 2k)·r + r²)` words staged and `O(k·r + r²)` read, plus `inv`. Device: the
    /// three limb products, `r²·n + k·r·n + k·r²` multiply-adds per kept pair, and `O(r·n + k·r +
    /// r²)` map invocations; two submits.
    #[allow(
        clippy::too_many_arguments,
        reason = "the ADMM step's operands are distinct matrices"
    )]
    pub(crate) fn x_update<E>(
        &mut self,
        zx: &[f64],
        z: &[f64],
        du: &[f64],
        rho: f64,
        reg: f64,
        inv: impl FnOnce(&[i64], usize) -> Result<Vec<i64>, E>,
    ) -> Option<Result<Vec<f64>, E>> {
        let f = self.frame.as_deref()?;
        let (k, n, r) = (f.k, f.n, f.r);
        if zx.len() != n * r || z.len() != k * r || du.len() != k * r {
            return None;
        }
        let dev = device()?;
        let off = f.off();
        // SAFETY: the frame is ours alone; the staging and readback are mapped at the offsets the
        // command buffers were recorded with, and each submit holds the device mutex.
        unsafe {
            let st = f.stage;
            std::ptr::copy_nonoverlapping([rho, reg].as_ptr().cast::<u8>(), st, 16);
            for (v, at) in [(zx, off[0]), (z, off[1]), (du, off[2])] {
                std::ptr::copy_nonoverlapping(v.as_ptr().cast::<u8>(), st.add(at), 8 * v.len());
            }
            dev.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .vk
                .submit_wait(f.cb1);
            let fl = f.flags();
            if [ZT_PLANE, RT_PLANE, GRAM, GRAM + 1, WZ, WZ + 1, MAP1]
                .iter()
                .any(|&s| fl[s] != 0)
            {
                return None;
            }
            let hw = std::slice::from_raw_parts(f.back.add(FLAG_BYTES).cast::<i64>(), r * r);
            let hi = match inv(hw, r) {
                Ok(h) => h,
                Err(e) => return Some(Err(e)),
            };
            if hi.len() != r * r {
                return None;
            }
            std::ptr::copy_nonoverlapping(hi.as_ptr().cast::<u8>(), st.add(off[3]), 8 * hi.len());
            dev.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .vk
                .submit_wait(f.cb2);
            let fl = f.flags();
            if [HI_PLANE, F, F + 1, MAP2].iter().any(|&s| fl[s] != 0) {
                return None;
            }
            let out =
                std::slice::from_raw_parts(f.back.add(FLAG_BYTES + 8 * r * r).cast::<f64>(), k * r);
            Some(Ok(out.to_vec()))
        }
    }
}

impl Drop for Side {
    fn drop(&mut self) {
        if let Some(f) = self.frame.take() {
            FRAMES
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((self.key, f));
        }
    }
}
