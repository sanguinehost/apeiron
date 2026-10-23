//! The exact limb product [`crate::engine::kern::limb::gemm_q32`] on the GPU, bit for bit.
//!
//! **Contract.** [`gemm_q32`]'s value per pair group is `round_shift_32(Σ_{(pair, block) live}
//! D·2^T)`, `D` the 32-lane signed digit dot and `T` the placement exponent, which depends on the
//! block and the level `s + t` only ([`DeviceOperands`]). The kernel
//! `src/gpu/spirv/limb_gemm.comp` forms every `D` on signed-int8 M16N16K32 cooperative matrices and
//! accumulates `v_g = Σ D·2^{T − Tmin_g}` exactly in 64 bits; the host rounds once by `2^{Tmin_g}`
//! ([`device_group_word`]) and sums the groups in `i128`, which is the CPU's arithmetic. The
//! accumulator is exact only when `Σ |D|·2^{T − Tmin_g} < 2^63`; the host bounds that sum from
//! `|D| ≤ pairs·2^17` before dispatch and returns `Ok(None)` when the bound fails, and the caller
//! runs [`gemm_q32`].
//!
//! **Route.** `engine::kern::limb` is the one adapter: its full products (`gemm_q32`,
//! `Prepared::gemm`, `Prepared::syrk`) call [`product`] when [`takes`] holds and run the CPU kernel
//! otherwise. `APEIRON_LIMB_DEVICE=0` keeps every product on the CPU.
//!
//! **Source.** GLSL compiled once by `glslc` ([`crate::gpu::nn::GLSLC_ARGS`]) into the committed
//! `src/gpu/spirv/limb_gemm.spv`; the test recompiles it and requires byte identity.
//!
//! **Buffers.** [`LimbGemm`] keeps its device buffers, staging, readback and one recorded command
//! buffer per shape across calls, so a fit that runs the same products every iteration allocates
//! once.
//!
//! [`gemm_q32`]: crate::engine::kern::limb::gemm_q32

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::engine::chart::pool::{chunks_mut_global, width_global};
use crate::engine::chart::task::ChunkTask;
use crate::engine::kern::limb::{device_group_word, DeviceOperands, LimbError};
use crate::gpu::vulkan::*;

/// The compiled kernel.
pub static LIMB_GEMM_SPV: &[u8] = include_bytes!("spirv/limb_gemm.spv");

/// Rows and columns per workgroup (2×2 subgroups, each 2×2 fragments of 16).
const TILE: usize = 64;
/// `|Σ_32 d_s·d_t| ≤ 32·64·64 = 2^17` for balanced 7-bit digits.
const DOT_BITS: u32 = 17;
/// The shift-table entry of a dead block pair.
const DEAD: u32 = u32::MAX;
/// Header words of the meta buffer.
const HEADER: usize = 8;

/// The buffers one capacity owns, the descriptor set binding them and the pipeline over it.
struct Bound {
    a: Handle,
    b: Handle,
    o: Handle,
    meta: Handle,
    stage: Handle,
    stage_ptr: *mut u8,
    back: Handle,
    back_ptr: *const u8,
    /// Byte capacities of `a`, `b`, `meta`, `o`.
    cap: [usize; 4],
    layout: Handle,
    dset: Handle,
    pipe: Handle,
    /// One recorded command buffer per `(bytes a, b, meta, o, groups x, y, z)`.
    cbs: HashMap<[usize; 7], CommandBuffer>,
}

/// Output entries below which the finish runs on the calling thread.
const POOL_MIN: usize = 1 << 15;

/// The host finish of [`LimbGemm::run`] over whole output rows: entry `(i, j)` is
/// `Σ_g device_group_word(v_g(i, j), Tmin_g)` in `i128`, `v_g` the lo/hi pair of group `g`'s planes.
struct FinishTask<'a> {
    back: &'a [i32],
    tmin: &'a [u32],
    p: usize,
    pp: usize,
    plane: usize,
    /// Set when a group word or the sum leaves `i64`.
    range: AtomicBool,
}

impl ChunkTask<i64> for FinishTask<'_> {
    /// # Complexity
    ///
    /// `O(len·groups)` `i128` additions and two `i32` reads per group and entry.
    fn run(&self, _c: usize, start: usize, piece: &mut [i64]) {
        for (k, slot) in piece.iter_mut().enumerate() {
            let (i, j) = ((start + k) / self.p, (start + k) % self.p);
            let at = i * self.pp + j;
            let mut acc: i128 = 0;
            for (g, &t) in self.tmin.iter().enumerate() {
                let lo = self.back[2 * g * self.plane + at];
                let hi = self.back[(2 * g + 1) * self.plane + at];
                let v = (i64::from(hi) << 32) | i64::from(lo as u32);
                match device_group_word(v, t) {
                    Some(w) => acc += i128::from(w),
                    None => self.range.store(true, Ordering::Relaxed),
                }
            }
            match i64::try_from(acc) {
                Ok(x) => *slot = x,
                Err(_) => self.range.store(true, Ordering::Relaxed),
            }
        }
    }
}

/// The least `m·p·n` the device takes. Derived from two measured points on the RTX 5080
/// (`limb_gemm_matches_cpu`): `768·768·1024 ≈ 2^29.2` runs 1.84× faster than the CPU and
/// `100·37·70 ≈ 2^18` 0.6×; `2^26` sits between them and is not itself measured.
const MIN_WORK: usize = 1 << 26;

/// The least inner length `n` the device takes. At `m·p = 1024·768` (`limb_gemm_matches_cpu`, pooled
/// finish) the routed wall is `0.89 ms + 2.1·10⁻⁹ ms` per multiply-add and the CPU's `1.0·10⁻⁸ ms`,
/// so they meet at `n ≈ 0.89 / (786432 · 7.9·10⁻⁹) ≈ 143`; `n = 128` measured a tie (1.10 against
/// 1.07 ms). `256` is the next power of two above the meeting point.
const MIN_INNER: usize = 256;

/// The process's device and its limb product, loaded on first use; `None` when no Vulkan device
/// with cooperative matrices loads or `APEIRON_LIMB_DEVICE=0`.
struct Device {
    vk: Vk,
    gemm: LimbGemm,
}

// SAFETY: `Vk` and `Bound` hold raw Vulkan handles and mapped pointers. The one `Device` is reached
// only through `DEVICE`'s mutex, so no two threads use its queue, command buffers or mappings at once,
// which is the external synchronisation Vulkan requires.
unsafe impl Send for Device {}

static DEVICE: OnceLock<Option<Mutex<Device>>> = OnceLock::new();

/// The device, loading it on the first call.
///
/// # Complexity
///
/// `O(1)` after the first call; the first loads the Vulkan instance and device.
fn device() -> Option<&'static Mutex<Device>> {
    DEVICE
        .get_or_init(|| {
            if std::env::var("APEIRON_LIMB_DEVICE").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            // SAFETY: loading creates a fresh instance and device owned by `Device`.
            let vk = unsafe { Vk::load() }.ok()?;
            vk.has_coopmat.then(|| {
                Mutex::new(Device {
                    vk,
                    gemm: LimbGemm::default(),
                })
            })
        })
        .as_ref()
}

/// Whether [`product`] is tried for an `m × n` by `p × n` product: the work reaches [`MIN_WORK`],
/// the inner length reaches [`MIN_INNER`] and a device loads.
///
/// # Complexity
///
/// `O(1)` after the first call.
pub(crate) fn takes(m: usize, p: usize, n: usize) -> bool {
    n >= MIN_INNER && m.saturating_mul(p).saturating_mul(n) >= MIN_WORK && device().is_some()
}

/// The device product of `ops`, as [`LimbGemm::run`] on the process's device; `Ok(None)` when no
/// device loads or the accumulator bound fails.
///
/// # Errors
///
/// As [`crate::engine::kern::limb::gemm_q32`].
///
/// # Complexity
///
/// As [`LimbGemm::run`], serialised by the device mutex.
pub(crate) fn product(ops: &DeviceOperands<'_>) -> Result<Option<Vec<i64>>, LimbError> {
    let Some(dev) = device() else {
        return Ok(None);
    };
    let mut dev = dev
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Device { vk, gemm } = &mut *dev;
    // SAFETY: `vk` loaded with cooperative matrices, and `gemm` is only ever used with it.
    unsafe { gemm.run(vk, ops) }
}

/// The device limb product with buffers reused across calls.
#[derive(Default)]
pub(crate) struct LimbGemm {
    bound: Option<Bound>,
}

/// Bytes of `v` as a byte slice.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is a plain integer type here (`i8`, `u32`); every bit pattern is a byte and the
    // length is the slice's own size in bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

impl LimbGemm {
    /// The product of `ops` on the device: `Ok(Some(C))` bit-identical to
    /// [`crate::engine::kern::limb::gemm_q32`], `Ok(None)` when the 64-bit accumulator bound fails
    /// (run the CPU product), or the CPU's refusal.
    ///
    /// # Errors
    ///
    /// As [`crate::engine::kern::limb::gemm_q32`].
    ///
    /// # Safety
    ///
    /// `vk` is a live device with cooperative matrices; every call on one `LimbGemm` uses the same
    /// `vk`.
    ///
    /// # Complexity
    ///
    /// Host: `O(P·⌈n/32⌉)` meta words
    /// and `O(G·m·p)` finishing words for `G` pair groups. Device: `P·M·P'·n` signed byte
    /// multiply-adds, `M`, `P'` the row counts padded to 64, plus `O(G·L·⌈n/32⌉·M·P')` 64-bit
    /// accumulates (one per level and block). Transfers: `(M + P')·n·limbs` bytes up,
    /// `8·G·M·P'` bytes down.
    pub(crate) unsafe fn run(
        &mut self,
        vk: &Vk,
        ops: &DeviceOperands<'_>,
    ) -> Result<Option<Vec<i64>>, LimbError> {
        let (m, p) = (ops.m, ops.p);
        let blocks = ops.blocks;
        let Some((meta, tmin)) = host_meta(ops) else {
            return Ok(None);
        };
        let ngroups = tmin.len();
        let (mp, pp) = (m.div_ceil(TILE) * TILE, p.div_ceil(TILE) * TILE);
        let row_bytes = blocks * ops.limbs as usize * 32;
        let a_bytes = mp * row_bytes;
        let b_bytes = pp * row_bytes;
        let meta_bytes = meta.len() * 4;
        let plane = mp * pp;
        let o_bytes = ngroups * 2 * plane * 4;
        let bound = self.bind(vk, [a_bytes, b_bytes, meta_bytes, o_bytes]);
        // Staging: a, b (zero rows past m and p), meta, at the capacity offsets.
        let st = bound.stage_ptr;
        std::ptr::copy_nonoverlapping(ops.a_dig.as_ptr().cast::<u8>(), st, ops.a_dig.len());
        std::ptr::write_bytes(st.add(ops.a_dig.len()), 0, a_bytes - ops.a_dig.len());
        let sb = st.add(bound.cap[0]);
        std::ptr::copy_nonoverlapping(ops.b_dig.as_ptr().cast::<u8>(), sb, ops.b_dig.len());
        std::ptr::write_bytes(sb.add(ops.b_dig.len()), 0, b_bytes - ops.b_dig.len());
        let sm = sb.add(bound.cap[1]);
        let mb = as_bytes(&meta);
        std::ptr::copy_nonoverlapping(mb.as_ptr(), sm, mb.len());
        let key = [
            a_bytes,
            b_bytes,
            meta_bytes,
            o_bytes,
            mp / TILE,
            pp / TILE,
            ngroups,
        ];
        let cb = match bound.cbs.get(&key) {
            Some(&cb) => cb,
            None => {
                let cb = record(vk, bound, key);
                bound.cbs.insert(key, cb);
                cb
            }
        };
        vk.submit_wait(cb);
        // Finish: one rounding per group, the groups summed in i128, pooled over output rows.
        let back = std::slice::from_raw_parts(bound.back_ptr.cast::<i32>(), 2 * ngroups * plane);
        let finish = FinishTask {
            back,
            tmin: &tmin,
            p,
            pp,
            plane,
            range: AtomicBool::new(false),
        };
        let mut out = vec![0i64; m * p];
        if out.len() >= POOL_MIN {
            let rows = m.div_ceil(width_global().max(1) * 2).max(1);
            chunks_mut_global(&mut out, rows * p, &finish);
        } else {
            finish.run(0, 0, &mut out);
        }
        if finish.range.load(Ordering::Relaxed) {
            return Err(LimbError::Range);
        }
        Ok(Some(out))
    }

    /// The bound buffers at no less than `need` bytes each, reallocated (all four, with the
    /// staging, readback, descriptor set and pipeline) when any capacity is short.
    ///
    /// # Complexity
    ///
    /// `O(1)` calls when the capacity holds; otherwise six allocations and one pipeline build.
    unsafe fn bind(&mut self, vk: &Vk, need: [usize; 4]) -> &mut Bound {
        let fits = self
            .bound
            .as_ref()
            .is_some_and(|b| (0..4).all(|i| b.cap[i] >= need[i]));
        if !fits {
            let cap = need.map(|x| x.max(16));
            let dev = |bytes: usize| {
                vk.buffer(
                    bytes as u64,
                    BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST,
                    MEMORY_DEVICE_LOCAL,
                )
                .expect("limb gemm device buffer")
                .0
            };
            let (a, b, meta, o) = (dev(cap[0]), dev(cap[1]), dev(cap[2]), dev(cap[3]));
            let (stage, stage_mem) = vk
                .buffer(
                    (cap[0] + cap[1] + cap[2]) as u64,
                    BUFFER_USAGE_TRANSFER_SRC,
                    MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
                )
                .expect("limb gemm staging buffer");
            let (back, back_mem) = vk
                .buffer(
                    cap[3] as u64,
                    BUFFER_USAGE_TRANSFER_DST,
                    MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT | MEMORY_HOST_CACHED,
                )
                .or_else(|| {
                    vk.buffer(
                        cap[3] as u64,
                        BUFFER_USAGE_TRANSFER_DST,
                        MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
                    )
                })
                .expect("limb gemm readback buffer");
            let (layout, dset) = vk.ssbo_set(&[a, b, o, meta]);
            let pipe = vk.pipeline(LIMB_GEMM_SPV, layout);
            self.bound = Some(Bound {
                a,
                b,
                o,
                meta,
                stage,
                stage_ptr: vk.map(stage_mem).cast(),
                back,
                back_ptr: vk.map(back_mem).cast_const().cast(),
                cap,
                layout,
                dset,
                pipe,
                cbs: HashMap::new(),
            });
        }
        self.bound.as_mut().expect("bound above")
    }
}

/// The pair groups and their segments: `ops.pairs` in consecutive runs of `group` pairs, each run
/// split into its segments, the pairs of one level `s + t`. `groups[g]` is `g`'s segment range,
/// `segs[k]` segment `k`'s first pair and pair count.
///
/// # Complexity
///
/// `O(P)` for `P` kept pairs.
fn segments(pairs: &[(u32, u32)], group: usize) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let np = pairs.len();
    let mut groups = Vec::with_capacity(np.div_ceil(group));
    let mut segs = Vec::new();
    let mut g0 = 0usize;
    while g0 < np {
        let g1 = (g0 + group).min(np);
        let first = segs.len();
        let mut q = g0;
        while q < g1 {
            let level = pairs[q].0 + pairs[q].1;
            let mut e = q;
            while e < g1 && pairs[e].0 + pairs[e].1 == level {
                e += 1;
            }
            segs.push((q, e - q));
            q = e;
        }
        groups.push((first, segs.len()));
        g0 = g1;
    }
    (groups, segs)
}

/// The meta buffer `limb_gemm.comp` and `limb_place.comp` read, without its shift table:
/// `[blocks, limbs, mp, pp, off_groups, off_segs, off_pairs, off_shift]`, the groups, the segments
/// and the pairs. Every word depends on the shape and the pairs only, never on the operands.
///
/// # Complexity
///
/// `O(P)` words.
#[allow(
    clippy::cast_possible_truncation,
    reason = "meta words are block, pair and row counts below 2^32"
)]
fn meta_head(
    pairs: &[(u32, u32)],
    group: usize,
    blocks: usize,
    limbs: u32,
    m: usize,
    p: usize,
) -> Vec<u32> {
    let (groups, segs) = segments(pairs, group);
    let off_groups = HEADER;
    let off_segs = off_groups + 2 * groups.len();
    let off_pairs = off_segs + 2 * segs.len();
    let off_shift = off_pairs + 2 * pairs.len();
    let mut meta: Vec<u32> = vec![
        blocks as u32,
        limbs,
        (m.div_ceil(TILE) * TILE) as u32,
        (p.div_ceil(TILE) * TILE) as u32,
        off_groups as u32,
        off_segs as u32,
        off_pairs as u32,
        off_shift as u32,
    ];
    meta.extend(groups.iter().flat_map(|&(x, y)| [x as u32, y as u32]));
    meta.extend(segs.iter().flat_map(|&(x, y)| [x as u32, y as u32]));
    meta.extend(pairs.iter().flat_map(|&(s, t)| [s, t]));
    meta
}

/// The full meta buffer of `ops` (the head and the `[segment][block]` shift table, `T − Tmin_g` or
/// [`DEAD`]) and `Tmin_g` per group (0 for a group with no live block pair); `None` when a group's
/// 64-bit accumulator bound `Σ pc·2^{17 + sh} < 2^63` fails.
///
/// # Complexity
///
/// `O(P·⌈n/32⌉)`: two passes over the live table per group.
#[allow(
    clippy::cast_sign_loss,
    reason = "a shift T − Tmin_g of a live T is non-negative, and Tmin_g ≥ MIN_PAIR_EXP"
)]
fn host_meta(ops: &DeviceOperands<'_>) -> Option<(Vec<u32>, Vec<u32>)> {
    let blocks = ops.blocks;
    let (groups, segs) = segments(&ops.pairs, ops.group);
    let mut tmin = vec![0u32; groups.len()];
    let mut shift = vec![DEAD; segs.len() * blocks];
    for (g, &(s0, s1)) in groups.iter().enumerate() {
        let mut lo: Option<i32> = None;
        for &(p0, _) in &segs[s0..s1] {
            for blk in 0..blocks {
                if let Some(t) = ops.exp[p0 * blocks + blk] {
                    lo = Some(lo.map_or(t, |x| x.min(t)));
                }
            }
        }
        let Some(lo) = lo else { continue };
        let mut bound: u128 = 0;
        for (k, &(p0, pc)) in segs.iter().enumerate().take(s1).skip(s0) {
            for blk in 0..blocks {
                if let Some(t) = ops.exp[p0 * blocks + blk] {
                    let sh = (t - lo) as u32;
                    if sh > 62 {
                        return None;
                    }
                    bound += (pc as u128) << (DOT_BITS + sh);
                    shift[k * blocks + blk] = sh;
                }
            }
        }
        if bound >= 1u128 << 63 {
            return None;
        }
        tmin[g] = lo as u32;
    }
    let mut meta = meta_head(&ops.pairs, ops.group, blocks, ops.limbs, ops.m, ops.p);
    meta.extend_from_slice(&shift);
    Some((meta, tmin))
}

/// The command buffer of one shape: staging to device, the dispatch, device to readback.
///
/// # Complexity
///
/// `O(1)` recorded commands.
unsafe fn record(vk: &Vk, s: &Bound, key: [usize; 7]) -> CommandBuffer {
    let [a_bytes, b_bytes, meta_bytes, o_bytes, gx, gy, gz] = key;
    let cb = vk.command_buffer();
    vk.begin(cb);
    let copy = |src: Handle, src_offset: usize, dst: Handle, size: usize| {
        let region = BufferCopy {
            src_offset: src_offset as u64,
            dst_offset: 0,
            size: size as u64,
        };
        (vk.copy_buffer)(cb, src, dst, 1, &region);
    };
    copy(s.stage, 0, s.a, a_bytes);
    copy(s.stage, s.cap[0], s.b, b_bytes);
    copy(s.stage, s.cap[0] + s.cap[1], s.meta, meta_bytes);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_COMPUTE,
        ACCESS_TRANSFER_WRITE,
        ACCESS_SHADER_READ,
    );
    (vk.bind_pipe)(cb, PIPELINE_BIND_POINT_COMPUTE, s.pipe);
    (vk.bind_sets)(
        cb,
        PIPELINE_BIND_POINT_COMPUTE,
        s.layout,
        0,
        1,
        &s.dset,
        0,
        std::ptr::null(),
    );
    #[allow(
        clippy::cast_possible_truncation,
        reason = "group counts are rows/64 and pair groups, far below 2^32"
    )]
    (vk.dispatch)(cb, gx as u32, gy as u32, gz as u32);
    vk.barrier(
        cb,
        STAGE_COMPUTE,
        STAGE_TRANSFER,
        ACCESS_SHADER_WRITE,
        ACCESS_TRANSFER_READ,
    );
    copy(s.o, 0, s.back, o_bytes);
    vk.barrier(
        cb,
        STAGE_TRANSFER,
        STAGE_HOST,
        ACCESS_TRANSFER_WRITE,
        ACCESS_HOST_READ,
    );
    check((vk.end_cb)(cb), "vkEndCommandBuffer");
    cb
}

/// The device plane build, the first stage of the device-resident ADMM iteration
/// (`docs/epics/nanoquant-subsume.md`, W6).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed by the resident x_update, W6 T4 in docs/epics/nanoquant-subsume.md"
    )
)]
mod plane {
    use crate::gpu::vulkan::*;

    /// The compiled plane build.
    pub(super) static LIMB_PLANE_SPV: &[u8] = include_bytes!("spirv/limb_plane.spv");

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
}

/// The device placement, the second stage of the device-resident ADMM iteration
/// (`docs/epics/nanoquant-subsume.md`, W6).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed by the resident x_update, W6 T4 in docs/epics/nanoquant-subsume.md"
    )
)]
mod place {
    use crate::gpu::vulkan::*;

    /// The compiled placement.
    pub(super) static LIMB_PLACE_SPV: &[u8] = include_bytes!("spirv/limb_place.spv");

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::kern::limb::{gemm_q32, with_device_operands};
    use crate::gpu::nn::GLSLC_ARGS;
    use std::path::Path;
    use std::time::Instant;

    /// A fixed-seed LCG word in `[-2^bits, 2^bits)`.
    fn words(len: usize, seed: u64, bits: u32) -> Vec<i64> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 11) as i64 >> (53 - bits - 1)) - (1i64 << bits)
            })
            .collect()
    }

    /// The committed SPIR-V is what `glslc` makes of the committed GLSL.
    #[test]
    fn limb_spirv_is_the_glsl() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
        let out = std::env::temp_dir().join(format!("limb_gemm_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(dir.join("limb_gemm.comp"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let fresh = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        assert!(
            fresh == LIMB_GEMM_SPV,
            "limb_gemm.spv is stale: re-run glslc"
        );
    }

    /// The committed plane SPIR-V is what `glslc` makes of the committed GLSL.
    #[test]
    fn plane_spirv_is_the_glsl() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
        let out = std::env::temp_dir().join(format!("limb_plane_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(dir.join("limb_plane.comp"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let fresh = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        assert!(
            fresh == plane::LIMB_PLANE_SPV,
            "limb_plane.spv is stale: re-run glslc"
        );
    }

    /// The device plane build equals `Plane::build` (exponents and every digit byte) at the fit's
    /// operand shapes and an odd one, and flags a block exponent past 62; with the CPU wall.
    #[test]
    fn plane_build_matches_cpu() {
        let vk = match unsafe { Vk::load() } {
            Ok(vk) => vk,
            Err(e) => {
                println!("SKIP: No Vulkan device present: {e}");
                return;
            }
        };
        // (rows, n, bits, flag expected): W of blk.23 ffn_down (o 3584, n 1024), Z_xᵀ (r 768 rows
        // over n 1024) and its transpose over o, a tail block, and words past 2^61.
        for &(rows, n, bits, flagged) in &[
            (3584usize, 1024usize, 34u32, false),
            (768, 1024, 33, false),
            (1024, 3584, 33, false),
            (100, 70, 40, false),
            (37, 33, 5, false),
            (64, 96, 61, false),
            (64, 96, 62, true),
        ] {
            let limbs = 3u32;
            // `words` draws at most 52 bits; wider words are those shifted up.
            let x: Vec<i64> = words(
                rows * n,
                0x51ed + rows as u64 + u64::from(bits),
                bits.min(50),
            )
            .into_iter()
            .map(|w| w << bits.saturating_sub(50))
            .collect();
            let t0 = Instant::now();
            let (e_cpu, dig_cpu) =
                crate::engine::kern::limb::plane_parts(&x, rows, n, limbs).expect("cpu plane");
            let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
            let blocks = n.div_ceil(32);
            let dig_bytes = rows * blocks * limbs as usize * 32;
            let (e_dev, dig_dev, flag, dev_ms) = unsafe {
                let host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
                let usage = BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST;
                let (bx, mx) = vk.buffer(8 * x.len() as u64, usage, host).expect("x");
                let (be, me) = vk.buffer(4 * blocks as u64, usage, host).expect("e");
                let (bd, md) = vk.buffer(dig_bytes as u64, usage, host).expect("dig");
                let (bf, mf) = vk.buffer(4, usage, host).expect("flag");
                std::ptr::copy_nonoverlapping(x.as_ptr(), vk.map(mx).cast(), x.len());
                let pb = plane::PlaneBuild::new(&vk, [bx, be, bd, bf], rows, n, limbs);
                let cb = vk.command_buffer();
                vk.begin(cb);
                pb.record(&vk, cb);
                vk.barrier(
                    cb,
                    STAGE_COMPUTE,
                    STAGE_HOST,
                    ACCESS_SHADER_WRITE,
                    ACCESS_HOST_READ,
                );
                check((vk.end_cb)(cb), "vkEndCommandBuffer");
                let t1 = Instant::now();
                vk.submit_wait(cb);
                let dev_ms = t1.elapsed().as_secs_f64() * 1e3;
                (
                    std::slice::from_raw_parts(vk.map(me).cast::<u32>(), blocks).to_vec(),
                    std::slice::from_raw_parts(vk.map(md).cast::<i8>(), dig_bytes).to_vec(),
                    *vk.map(mf).cast::<u32>(),
                    dev_ms,
                )
            };
            assert_eq!(flag != 0, flagged, "range flag at {rows}x{n} bits {bits}");
            if flagged {
                continue;
            }
            assert_eq!(e_dev, e_cpu, "block exponents differ at {rows}x{n}");
            let diff = dig_dev.iter().zip(&dig_cpu).filter(|(a, b)| a != b).count();
            println!(
                "limb_plane {rows}x{n} bits {bits}: cpu {cpu_ms:.2} ms, device {dev_ms:.2} ms \
                 (submit incl.), differing digits {diff} of {dig_bytes}"
            );
            assert_eq!(dig_cpu.len(), dig_bytes);
            assert_eq!(diff, 0, "digits differ at {rows}x{n}");
        }
    }

    /// The committed placement SPIR-V is what `glslc` makes of the committed GLSL.
    #[test]
    fn place_spirv_is_the_glsl() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/spirv");
        let out = std::env::temp_dir().join(format!("limb_place_{}.spv", std::process::id()));
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(dir.join("limb_place.comp"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("glslc must be on PATH: the committed SPIR-V is checked against it");
        assert!(st.success(), "glslc failed: {st}");
        let fresh = std::fs::read(&out).expect("glslc output");
        let _ = std::fs::remove_file(&out);
        assert!(
            fresh == place::LIMB_PLACE_SPV,
            "limb_place.spv is stale: re-run glslc"
        );
    }

    /// `rows × n` words whose block `b` draws `bits(b)` bits: a fixed-seed LCG as [`words`].
    fn block_words(rows: usize, n: usize, seed: u64, bits: impl Fn(usize) -> u32) -> Vec<i64> {
        let mut s = seed;
        (0..rows * n)
            .map(|i| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let b = bits((i % n) / 32);
                let w = ((s >> 11) as i64 >> (53 - b.min(50) - 1)) - (1i64 << b.min(50));
                w << b.saturating_sub(50)
            })
            .collect()
    }

    /// The device placement equals `host_meta` (every meta word and `Tmin_g`), its fallback flag
    /// equals `host_meta`'s `None` (both of its causes: a shift past 62, and the bound at shifts in
    /// 46..=62) and its range flag `Layout::placement`'s `Range`, at one and two pair groups, with
    /// dead blocks, an all-dead operand pair and live entries at `T = 8`.
    #[test]
    fn place_matches_host_meta() {
        let vk = match unsafe { Vk::load() } {
            Ok(vk) => vk,
            Err(e) => {
                println!("SKIP: No Vulkan device present: {e}");
                return;
            }
        };
        type Bits = fn(usize) -> u32;
        // (name, n, bits of a's blocks, bits of b's blocks, expected outcome).
        let cases: [(&str, usize, Bits, Bits, &str); 8] = [
            ("fit 1024", 1024, |_| 34, |_| 33, "meta"),
            ("fit 3584, two groups", 3584, |_| 33, |_| 33, "meta"),
            (
                "dead blocks",
                1024,
                |b| if b % 3 == 0 { 2 } else { 20 + b as u32 % 11 },
                |b| if b % 5 == 0 { 3 } else { 21 + b as u32 % 9 },
                "meta",
            ),
            (
                "T = 8 edge",
                3584,
                |b| 15 + b as u32 % 5,
                |b| 15 + b as u32 % 4,
                "meta",
            ),
            ("all dead", 1024, |_| 3, |_| 3, "meta"),
            (
                "shift past 62",
                1024,
                |b| if b % 2 == 0 { 12 } else { 45 },
                |b| if b % 2 == 0 { 12 } else { 45 },
                "fallback",
            ),
            (
                "bound at shift 46..=62",
                1024,
                |b| if b % 2 == 0 { 20 } else { 45 },
                |b| if b % 2 == 0 { 20 } else { 45 },
                "fallback",
            ),
            ("range", 1024, |_| 56, |_| 56, "range"),
        ];
        let (m, p, limbs) = (40usize, 24usize, 3u32);
        let mut edge = 0usize;
        for (name, n, fa, fb, want) in cases {
            let a = block_words(m, n, 0xa11 + n as u64, fa);
            let b = block_words(p, n, 0xb22 + n as u64, fb);
            let blocks = n.div_ceil(32);
            let (ea, eb, pairs, group, exp) =
                crate::engine::kern::limb::placement_parts(&a, m, &b, p, n, limbs).expect("planes");
            // `host_meta` reads the shape, the pairs and the exponents only, never the digits.
            let host = exp.map(|exp| {
                edge += exp.iter().filter(|&&t| t == Some(8)).count();
                host_meta(&DeviceOperands {
                    a_dig: &[],
                    b_dig: &[],
                    m,
                    p,
                    blocks,
                    limbs,
                    pairs: pairs.clone(),
                    group,
                    exp,
                })
            });
            let head = meta_head(&pairs, group, blocks, limbs, m, p);
            let (groups, segs) = segments(&pairs, group);
            let (ngroups, nsegs) = (groups.len(), segs.len());
            let meta_words = head.len() + nsegs * blocks;
            let (meta_dev, out_dev, dev_ms) = unsafe {
                let host_mem = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
                let usage = BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_DST;
                let (bea, mea) = vk.buffer(4 * blocks as u64, usage, host_mem).expect("ea");
                let (beb, meb) = vk.buffer(4 * blocks as u64, usage, host_mem).expect("eb");
                let (bm, mm) = vk
                    .buffer(4 * meta_words as u64, usage, host_mem)
                    .expect("meta");
                let (bo, mo) = vk
                    .buffer(4 * (2 + ngroups) as u64, usage, host_mem)
                    .expect("out");
                std::ptr::copy_nonoverlapping(ea.as_ptr(), vk.map(mea).cast(), blocks);
                std::ptr::copy_nonoverlapping(eb.as_ptr(), vk.map(meb).cast(), blocks);
                // The shift table and the outputs start as a pattern the kernel must overwrite.
                let pm = vk.map(mm).cast::<u32>();
                std::ptr::write_bytes(pm, 0xA5, meta_words);
                std::ptr::copy_nonoverlapping(head.as_ptr(), pm, head.len());
                std::ptr::write_bytes(vk.map(mo).cast::<u32>(), 0xA5, 2 + ngroups);
                let pl = place::Place::new(&vk, [bea, beb, bm, bo], ngroups as u32);
                let cb = vk.command_buffer();
                vk.begin(cb);
                pl.record(&vk, cb);
                vk.barrier(
                    cb,
                    STAGE_COMPUTE,
                    STAGE_HOST,
                    ACCESS_SHADER_WRITE,
                    ACCESS_HOST_READ,
                );
                check((vk.end_cb)(cb), "vkEndCommandBuffer");
                let t1 = Instant::now();
                vk.submit_wait(cb);
                let dev_ms = t1.elapsed().as_secs_f64() * 1e3;
                (
                    std::slice::from_raw_parts(pm, meta_words).to_vec(),
                    std::slice::from_raw_parts(vk.map(mo).cast::<u32>(), 2 + ngroups).to_vec(),
                    dev_ms,
                )
            };
            let (range, fallback) = (out_dev[0] != 0, out_dev[1] != 0);
            let got = match &host {
                Err(_) => "range",
                Ok(None) => "fallback",
                Ok(Some(_)) => "meta",
            };
            assert_eq!(got, want, "{name}: the host outcome is not the case's");
            assert_eq!(range, got == "range", "{name}: range flag");
            if got != "range" {
                assert_eq!(fallback, got == "fallback", "{name}: fallback flag");
            }
            let mut diff = 0usize;
            if let Ok(Some((meta, tmin))) = &host {
                diff = meta.iter().zip(&meta_dev).filter(|(x, y)| x != y).count()
                    + tmin
                        .iter()
                        .zip(&out_dev[2..])
                        .filter(|(x, y)| x != y)
                        .count();
                assert_eq!(meta.len(), meta_dev.len());
            }
            println!(
                "limb_place {name}: n {n}, groups {ngroups}, segments {nsegs}, outcome {got}, \
                 differing words {diff}, device {dev_ms:.3} ms (submit incl.)"
            );
            assert_eq!(diff, 0, "{name}: meta or Tmin differ");
        }
        println!("limb_place: {edge} live block pairs at T = 8");
        assert!(
            edge > 0,
            "no live block pair at T = 8: the MIN_PAIR_EXP edge is untested"
        );
    }

    /// Bit identity with the CPU product at the fit's shapes and an odd one, directly and through
    /// the routed `gemm_q32`, plus the wall of the CPU kernel and of the routed call.
    #[test]
    fn limb_gemm_matches_cpu() {
        let vk = match unsafe { Vk::load() } {
            Ok(vk) => vk,
            Err(e) => {
                println!("SKIP: No Vulkan device present: {e}");
                return;
            }
        };
        if !vk.has_coopmat {
            println!("SKIP: no cooperative matrices");
            return;
        }
        let mut dev = LimbGemm::default();
        // (m, p, n, bits of a, bits of b): the W·Z product of blk.23 ffn_down at r 768, its Gram
        // ZᵀZ, the sketched tune's V·Qᵀ (s 128) and U·Pᵀ (inner s), the ridge solve's Rᵀ·H⁻¹, the
        // Gram-like 768² over 1024, a mid shape and an odd shape below one tile.
        for &(m, p, n, ba, bb) in &[
            (1024usize, 768usize, 3584usize, 34u32, 33u32),
            (768, 768, 3584, 33, 33),
            (768, 128, 3584, 33, 33),
            (1024, 768, 128, 33, 33),
            (1024, 768, 768, 33, 33),
            (768, 768, 1024, 33, 33),
            (256, 256, 1024, 33, 33),
            (100, 37, 70, 40, 20),
        ] {
            let a = words(m * n, 0x9e37 + m as u64, ba);
            let b = words(p * n, 0x7f4a + p as u64, bb);
            let (cpu, direct) = with_device_operands(&a, m, &b, p, n, 3, |ops| {
                let first = unsafe { dev.run(&vk, ops) }
                    .expect("device product")
                    .expect("accumulator bound holds");
                let again = unsafe { dev.run(&vk, ops) }
                    .expect("device product")
                    .expect("accumulator bound holds");
                assert!(again == first, "device product is not deterministic");
                first
            })
            .expect("planes");
            let diff = cpu.iter().zip(&direct).filter(|(x, y)| x != y).count();
            assert_eq!(diff, 0, "device differs from gemm_q32 at {m}x{p}x{n}");
            let mut cpu_ms = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                let _ = with_device_operands(&a, m, &b, p, n, 3, |_| ()).expect("cpu product");
                cpu_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let routed = gemm_q32(&a, m, &b, p, n, 3).expect("routed product");
            assert!(routed == cpu, "routed gemm_q32 differs at {m}x{p}x{n}");
            let mut routed_ms = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                let _ = gemm_q32(&a, m, &b, p, n, 3).expect("routed product");
                routed_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            println!(
                "limb_gemm {m}x{p}x{n}: cpu {:.2}/{:.2}/{:.2} ms, routed {:.2}/{:.2}/{:.2} ms \
                 (device taken: {}), differing entries {diff} of {}",
                cpu_ms[0],
                cpu_ms[1],
                cpu_ms[2],
                routed_ms[0],
                routed_ms[1],
                routed_ms[2],
                takes(m, p, n),
                cpu.len()
            );
        }
    }

    /// Binary64 `/`, `sqrt`, unfused `a·b + c`, round-half-away of
    /// `a·2³²` and `i64 → f64` on the device against the CPU, bit for bit.
    #[test]
    fn f64_maps_match_cpu() {
        let vk = match unsafe { Vk::load() } {
            Ok(vk) => vk,
            Err(e) => {
                println!("SKIP: No Vulkan device present: {e}");
                return;
            }
        };
        if !vk.has_f64 {
            println!("SKIP: no shaderFloat64");
            return;
        }
        let src = r#"#version 450
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer I { double x[]; };
layout(std430, binding = 1) writeonly buffer O { uint64_t y[]; };
layout(std430, binding = 2) readonly buffer M { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    double a = x[3 * i], b = x[3 * i + 1], c = x[3 * i + 2];
    precise double q = a / b;
    precise double s = sqrt(abs(a));
    precise double p = a * b;
    precise double m = p + c;
    precise double sc = a * 4294967296.0;
    double t = trunc(sc);
    precise double fr = abs(sc - t);
    double r = fr >= 0.5 ? t + sign(sc) : t;
    int64_t ri = int64_t(r);
    double back = double(int64_t(doubleBitsToUint64(c) >> 1) - int64_t(0x2000000000000000ul));
    y[6 * i] = doubleBitsToUint64(q);
    y[6 * i + 1] = doubleBitsToUint64(s);
    y[6 * i + 2] = doubleBitsToUint64(p);
    y[6 * i + 3] = doubleBitsToUint64(m);
    y[6 * i + 4] = uint64_t(ri);
    y[6 * i + 5] = doubleBitsToUint64(back);
}
"#;
        let dir = std::env::temp_dir();
        let comp = dir.join(format!("f64probe_{}.comp", std::process::id()));
        let spv = dir.join(format!("f64probe_{}.spv", std::process::id()));
        std::fs::write(&comp, src).expect("write probe");
        let st = std::process::Command::new("glslc")
            .args(GLSLC_ARGS)
            .arg(&comp)
            .arg("-o")
            .arg(&spv)
            .status()
            .expect("glslc");
        assert!(st.success(), "glslc failed");
        let code = std::fs::read(&spv).expect("spv");
        let n = 1usize << 20;
        let mut st = 0x2545f4914f6cdd1du64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let mut val = |k: u64| -> f64 {
            let r = next();
            let mag = ((r >> 11) as f64 / (1u64 << 53) as f64) * 2f64.powi(((r & 63) as i32) - 32);
            let v = if k % 7 == 0 {
                // a·2³² on a half: k + 0.5 over 2³².
                ((r >> 34) as f64 + 0.5) / 4294967296.0
            } else {
                mag
            };
            if r & 64 != 0 {
                -v
            } else {
                v
            }
        };
        let mut x = vec![0f64; 3 * n];
        for (k, e) in x.iter_mut().enumerate() {
            *e = val(k as u64);
            if k % 3 == 1 && *e == 0.0 {
                *e = 1.0;
            }
        }
        let want: Vec<u64> = (0..n)
            .flat_map(|i| {
                let (a, b, c) = (x[3 * i], x[3 * i + 1], x[3 * i + 2]);
                let sc = a * 4294967296.0;
                let back = ((c.to_bits() >> 1) as i64 - 0x2000000000000000i64) as f64;
                [
                    (a / b).to_bits(),
                    a.abs().sqrt().to_bits(),
                    (a * b).to_bits(),
                    (a * b + c).to_bits(),
                    sc.round() as i64 as u64,
                    back.to_bits(),
                ]
            })
            .collect();
        let got = unsafe {
            let host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
            let (bi, mi) = vk
                .buffer((24 * n) as u64, BUFFER_USAGE_STORAGE, host)
                .expect("in");
            let (bo, mo) = vk
                .buffer((48 * n) as u64, BUFFER_USAGE_STORAGE, host)
                .expect("out");
            let (bm, mm) = vk.buffer(16, BUFFER_USAGE_STORAGE, host).expect("meta");
            std::ptr::copy_nonoverlapping(x.as_ptr().cast::<u8>(), vk.map(mi).cast(), 24 * n);
            *vk.map(mm).cast::<u32>() = n as u32;
            let (layout, dset) = vk.ssbo_set(&[bi, bo, bm]);
            let pipe = vk.pipeline(&code, layout);
            let cb = vk.command_buffer();
            vk.begin(cb);
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
            (vk.dispatch)(cb, (n / 64) as u32, 1, 1);
            vk.barrier(
                cb,
                STAGE_COMPUTE,
                STAGE_HOST,
                ACCESS_SHADER_WRITE,
                ACCESS_HOST_READ,
            );
            check((vk.end_cb)(cb), "vkEndCommandBuffer");
            vk.submit_wait(cb);
            std::slice::from_raw_parts(vk.map(mo).cast::<u64>(), 6 * n).to_vec()
        };
        let _ = std::fs::remove_file(&comp);
        let _ = std::fs::remove_file(&spv);
        for (op, name) in ["div", "sqrt", "mul", "mul+add", "round", "i64->f64"]
            .iter()
            .enumerate()
        {
            let bad: Vec<usize> = (0..n)
                .filter(|&i| got[6 * i + op] != want[6 * i + op])
                .collect();
            println!("f64 {name}: {} of {n} differ", bad.len());
            for &i in bad.iter().take(3) {
                println!(
                    "  a {:e} b {:e} c {:e}: device {:#x} cpu {:#x}",
                    x[3 * i],
                    x[3 * i + 1],
                    x[3 * i + 2],
                    got[6 * i + op],
                    want[6 * i + op]
                );
            }
            assert!(
                bad.is_empty(),
                "device binary64 {name} differs from the CPU"
            );
        }
    }
}
