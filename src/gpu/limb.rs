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
mod plane;

/// The device placement, the second stage of the device-resident ADMM iteration.
mod place;

/// The device finish, the third stage of the device-resident ADMM iteration.
mod finish;

/// The resident `x_update`: the three stages and the binary64 maps between them.
mod admm;

/// The resident `rank1` on an `admm` frame: power steps, norms and the sign product.
mod rank1;

/// The ADMM iterate resident between outer iterations.
mod resident;

/// `spd_inverse` on the device above a cutoff order, host leaves below it.
mod inverse;

pub(crate) use admm::Side;
pub(crate) use resident::commit;

#[cfg(test)]
mod tests;
