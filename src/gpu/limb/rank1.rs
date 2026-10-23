//! `learn::nanoquant::rank1` resident on the device, bit for bit.
//!
//! **Contract.** [`Side::rank1`] returns `Some(out)` equal to the CPU `rank1` of the same `m` and
//! start vector in every bit, and `None` when a shape is outside the frame, `iters` passes
//! [`MAX_ITERS`], or a device stage flags (a plane exponent past 62, a placement bound, a finish word
//! outside `i64`, a map value off the grid). On `None` the caller runs the CPU `rank1`, which
//! computes the value or reproduces the refusal.
//!
//! **Route.** One [`Rank1`] per `rows × cols` frame holds the planes of `|M|` and `|M|ᵀ`, the two
//! vector planes, the four limb products (`|M|·v`, `|M|ᵀ·u`, `u·uᵀ`, `v·vᵀ`) and the maps of
//! `src/gpu/spirv/rank1_map.comp`. One command buffer per `(iters, start_rows)` runs the whole power
//! iteration and the sign product in one submit; every plane and product flag is copied to its own
//! readback word after its stage, because each record resets it.

use std::sync::PoisonError;

use super::admm::{copy, dev_buf, full, to_host, Map, Product, Side};
use super::plane::PlaneBuild;
use super::{device, TILE};
use crate::engine::kern::limb::MAX_INNER;
use crate::gpu::vulkan::*;

/// The compiled maps.
pub(super) static RANK1_MAP_SPV: &[u8] = include_bytes!("../spirv/rank1_map.spv");

/// Bytes of the flag block at the head of the readback.
const FLAG_BYTES: usize = 1024;
/// The largest `iters` whose flag words fit the block: `9 + 12·iters ≤ FLAG_BYTES / 4`.
pub(crate) const MAX_ITERS: usize = (FLAG_BYTES / 4 - 9) / 12;

/// The vector side: `U` on the rows (length `rows`), `V` on the columns (length `cols`).
const U: usize = 0;
const V: usize = 1;

/// One vector side's buffers and stages.
struct Vector {
    /// `[words, e, dig, flag]` of the one-row plane.
    plane: [Handle; 4],
    build: PlaneBuild,
    /// `x·xᵀ`, the `fro2` word.
    norm2: Product,
    raw: Handle,
    /// `raw → words` (mode 1), the norm (mode 2), `raw / norm → unit, words` (mode 3), and the
    /// matvec read `prod → raw, words` (mode 4).
    grid: Map,
    norm: Map,
    div: Map,
    read: Map,
}

/// The device frame of `rank1` on a `rows × cols` matrix.
pub(crate) struct Rank1 {
    rows: usize,
    cols: usize,
    stage: *mut u8,
    back: *const u8,
    stage_h: Handle,
    back_h: Handle,
    m: Handle,
    out: Handle,
    mflag: Handle,
    /// `[words, e, dig, flag]` of `|M|` and `|M|ᵀ`.
    pa: [Handle; 4],
    pt: [Handle; 4],
    builds: [PlaneBuild; 2],
    /// `|M|·v` (into `U`) and `|M|ᵀ·u` (into `V`), by destination side.
    mul: [Product; 2],
    vec: [Vector; 2],
    abs: Map,
    /// The sign product of `(raw_u, unit_v)` (`start_rows = false`) and `(unit_u, raw_v)`.
    sign: [Map; 2],
    /// Recorded command buffers by `(iters, start_rows)`, with their flag word count.
    cbs: Vec<(usize, bool, CommandBuffer, usize)>,
}

/// A recording command buffer and the next free readback flag word.
struct Rec<'a> {
    vk: &'a Vk,
    cb: CommandBuffer,
    back: Handle,
    slot: usize,
}

impl Rec<'_> {
    /// Copies `words` flag words of `src` to the next readback slots.
    unsafe fn flag(&mut self, src: Handle, words: usize) {
        copy(
            self.vk,
            self.cb,
            src,
            0,
            self.back,
            4 * self.slot,
            4 * words,
        );
        self.slot += words;
    }

    /// A barrier, `stage`, a barrier.
    unsafe fn step(&self, stage: impl FnOnce(&Vk, CommandBuffer)) {
        stage(self.vk, self.cb);
        full(self.vk, self.cb);
    }
}

impl Rank1 {
    /// Builds every buffer and pipeline of the frame and runs its initialisation (zero digit
    /// planes, meta uploads); `None` when an allocation fails.
    ///
    /// # Safety
    ///
    /// `vk` is live with cooperative matrices and binary64, and its mutex is held.
    ///
    /// # Complexity
    ///
    /// `O(1)` allocations and pipeline builds; device memory `O(rows·cols)` words.
    #[allow(
        clippy::too_many_lines,
        reason = "one frame's buffer graph, built in order"
    )]
    unsafe fn build(vk: &Vk, rows: usize, cols: usize, limbs: u32) -> Option<Self> {
        let (rp, cp) = (rows.div_ceil(TILE) * TILE, cols.div_ceil(TILE) * TILE);
        let (br, bc) = (rows.div_ceil(32), cols.div_ceil(32));
        let lb = limbs as usize * 32;
        let plane = |rows: usize, padded: usize, blocks: usize| -> Option<[Handle; 4]> {
            Some([
                dev_buf(vk, 8 * rows * blocks * 32)?,
                dev_buf(vk, 4 * blocks)?,
                dev_buf(vk, padded * blocks * lb)?,
                dev_buf(vk, 4)?,
            ])
        };
        let pa = plane(rows, rp, bc)?;
        let pt = plane(cols, cp, br)?;
        let pv = [plane(1, TILE, br)?, plane(1, TILE, bc)?];
        let len = [rows, cols];
        let (m, out, mflag, nil) = (
            dev_buf(vk, 8 * rows * cols)?,
            dev_buf(vk, 8 * rows * cols)?,
            dev_buf(vk, 4)?,
            dev_buf(vk, 16)?,
        );
        let raw = [dev_buf(vk, 8 * rows)?, dev_buf(vk, 8 * cols)?];
        let unit = [dev_buf(vk, 8 * rows)?, dev_buf(vk, 8 * cols)?];
        let nrm = [dev_buf(vk, 8)?, dev_buf(vk, 8)?];
        // The matvec results (row stride TILE) and the one shared norm word block.
        let xm = [dev_buf(vk, 8 * rp * TILE)?, dev_buf(vk, 8 * cp * TILE)?];
        let xn = dev_buf(vk, 8 * TILE * TILE)?;
        let o = dev_buf(
            vk,
            Product::o_bytes(rows, 1, cols, limbs)
                .max(Product::o_bytes(cols, 1, rows, limbs))
                .max(Product::o_bytes(1, 1, rows.max(cols), limbs)),
        )?;
        let mut ups: Vec<(Handle, Vec<u8>)> = Vec::new();
        let mul = [
            Product::new(
                vk,
                [pa[2], pa[1]],
                [pv[V][2], pv[V][1]],
                (rows, 1, cols, limbs),
                o,
                xm[U],
                &mut ups,
            )?,
            Product::new(
                vk,
                [pt[2], pt[1]],
                [pv[U][2], pv[U][1]],
                (cols, 1, rows, limbs),
                o,
                xm[V],
                &mut ups,
            )?,
        ];
        // Bindings 1–8: fin, fa, fb, prod, fo, wa, wb, flag.
        let map = |mode: usize,
                   total: usize,
                   [fin, fa, fb, prod, fo, wa, wb]: [Handle; 7],
                   ups: &mut Vec<_>| {
            Map::new(
                vk,
                RANK1_MAP_SPV,
                [mode, rows, cols, TILE, total],
                [fin, fa, fb, prod, fo, wa, wb, mflag],
                ups,
            )
        };
        let mut vecs = Vec::with_capacity(2);
        for s in [U, V] {
            let p = pv[s];
            vecs.push(Vector {
                plane: p,
                build: PlaneBuild::new(vk, p, 1, len[s], limbs),
                norm2: Product::new(
                    vk,
                    [p[2], p[1]],
                    [p[2], p[1]],
                    (1, 1, len[s], limbs),
                    o,
                    xn,
                    &mut ups,
                )?,
                raw: raw[s],
                grid: map(1, len[s], [raw[s], nil, nil, nil, nil, p[0], nil], &mut ups)?,
                norm: map(2, 1, [nil, nil, nil, xn, nrm[s], nil, nil], &mut ups)?,
                div: map(
                    3,
                    len[s],
                    [raw[s], nrm[s], nil, nil, unit[s], p[0], nil],
                    &mut ups,
                )?,
                read: map(
                    4,
                    len[s],
                    [nil, nil, nil, xm[s], raw[s], p[0], nil],
                    &mut ups,
                )?,
            });
        }
        let vec: [Vector; 2] = vecs.try_into().ok()?;
        let abs = map(
            0,
            rows * cols,
            [m, nil, nil, nil, nil, pa[0], pt[0]],
            &mut ups,
        )?;
        let sign = [
            map(
                5,
                rows * cols,
                [m, raw[U], unit[V], nil, out, nil, nil],
                &mut ups,
            )?,
            map(
                5,
                rows * cols,
                [m, unit[U], raw[V], nil, out, nil, nil],
                &mut ups,
            )?,
        ];
        let builds = [
            PlaneBuild::new(vk, pa, rows, cols, limbs),
            PlaneBuild::new(vk, pt, cols, rows, limbs),
        ];
        let up_bytes: usize = ups.iter().map(|u| u.1.len()).sum();
        let stage_bytes = (8 * rows * cols + 8 * rows.max(cols)).max(up_bytes);
        let (stage_h, stage_mem) = vk.buffer(
            stage_bytes as u64,
            BUFFER_USAGE_TRANSFER_SRC,
            MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
        )?;
        let back_bytes = (FLAG_BYTES + 8 * rows * cols) as u64;
        let (back_h, back_mem) = vk
            .buffer(
                back_bytes,
                BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT | MEMORY_HOST_CACHED,
            )
            .or_else(|| {
                vk.buffer(
                    back_bytes,
                    BUFFER_USAGE_TRANSFER_DST,
                    MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
                )
            })?;
        let stage: *mut u8 = vk.map(stage_mem).cast();
        let back: *const u8 = vk.map(back_mem).cast_const().cast();

        // Initialisation: zero digit planes (rows past the count stay zero) and the meta words.
        let cb = vk.command_buffer();
        vk.begin(cb);
        for p in [pa, pt, pv[U], pv[V]] {
            (vk.fill_buffer)(cb, p[2], 0, WHOLE_SIZE, 0);
        }
        let mut at = 0usize;
        for (dst, b) in &ups {
            std::ptr::copy_nonoverlapping(b.as_ptr(), stage.add(at), b.len());
            copy(vk, cb, stage_h, at, *dst, 0, b.len());
            at += b.len();
        }
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        Some(Self {
            rows,
            cols,
            stage,
            back,
            stage_h,
            back_h,
            m,
            out,
            mflag,
            pa,
            pt,
            builds,
            mul,
            vec,
            abs,
            sign,
            cbs: Vec::new(),
        })
    }

    /// `unit` of side `s` from its grid words: plane, `fro2`, norm, divide, plane of the unit.
    unsafe fn unit(&self, r: &mut Rec<'_>, s: usize) {
        let x = &self.vec[s];
        r.step(|vk, cb| x.build.record(vk, cb));
        r.flag(x.plane[3], 1);
        r.step(|vk, cb| x.norm2.record(vk, cb));
        r.flag(x.norm2.out, 2);
        r.step(|vk, cb| x.norm.record(vk, cb));
        r.step(|vk, cb| x.div.record(vk, cb));
        r.step(|vk, cb| x.build.record(vk, cb));
        r.flag(x.plane[3], 1);
    }

    /// The product into side `t` (`|M|·v` for `U`, `|M|ᵀ·u` for `V`) and its read into `raw_t`.
    unsafe fn mul(&self, r: &mut Rec<'_>, t: usize) {
        r.step(|vk, cb| self.mul[t].record(vk, cb));
        r.flag(self.mul[t].out, 2);
        r.step(|vk, cb| self.vec[t].read.record(vk, cb));
    }

    /// Records the whole `rank1` for `(iters, start_rows)`; returns the buffer and its flag words.
    ///
    /// # Safety
    ///
    /// The device mutex is held; `9 + 12·iters ≤ FLAG_BYTES / 4`.
    ///
    /// # Complexity
    ///
    /// `O(iters)` recorded dispatches.
    unsafe fn record(&self, vk: &Vk, iters: usize, start_rows: bool) -> (CommandBuffer, usize) {
        let (rows, cols) = (self.rows, self.cols);
        let (s, t) = if start_rows { (U, V) } else { (V, U) };
        let cb = vk.command_buffer();
        vk.begin(cb);
        let mut r = Rec {
            vk,
            cb,
            back: self.back_h,
            slot: 0,
        };
        copy(vk, cb, self.stage_h, 0, self.m, 0, 8 * rows * cols);
        let off = 8 * rows * cols;
        let ls = [rows, cols][s];
        copy(vk, cb, self.stage_h, off, self.vec[s].raw, 0, 8 * ls);
        (vk.fill_buffer)(cb, self.mflag, 0, 4, 0);
        full(vk, cb);
        r.step(|vk, cb| self.abs.record(vk, cb));
        r.step(|vk, cb| {
            self.builds[0].record(vk, cb);
            self.builds[1].record(vk, cb);
        });
        r.flag(self.pa[3], 1);
        r.flag(self.pt[3], 1);
        r.step(|vk, cb| self.vec[s].grid.record(vk, cb));
        self.unit(&mut r, s);
        for _ in 0..iters {
            self.mul(&mut r, t);
            self.unit(&mut r, t);
            self.mul(&mut r, s);
            self.unit(&mut r, s);
        }
        self.mul(&mut r, t);
        r.step(|vk, cb| self.sign[usize::from(start_rows)].record(vk, cb));
        copy(
            vk,
            cb,
            self.out,
            0,
            self.back_h,
            FLAG_BYTES,
            8 * rows * cols,
        );
        r.flag(self.mflag, 1);
        to_host(vk, cb);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        (cb, r.slot)
    }
}

impl Side {
    /// `rank1(m, k, r, iters, start_rows, start)` on the device over this side's `k × r` frame
    /// (`start` the Gaussian start vector, `k` long when `start_rows`, else `r`). See the module
    /// contract for the two outcomes.
    ///
    /// # Complexity
    ///
    /// Host: `O(k·r)` words staged and read. Device: `2·iters + 1` matvecs and `2·iters + 1` norms,
    /// `O((iters + 1)·k·r)` multiply-adds per kept pair, and `O(k·r)` map invocations; one submit.
    pub(crate) fn rank1(
        &mut self,
        m: &[f64],
        iters: usize,
        start_rows: bool,
        start: &[f64],
    ) -> Option<Vec<f64>> {
        let limbs = u32::try_from(self.key[3]).ok()?;
        let f = self.frame.as_deref_mut()?;
        let (rows, cols) = (f.k, f.r);
        let ls = if start_rows { rows } else { cols };
        if m.len() != rows * cols || start.len() != ls || rows > MAX_INNER || iters > MAX_ITERS {
            return None;
        }
        let dev = device()?;
        let guard = dev.lock().unwrap_or_else(PoisonError::into_inner);
        let vk = &guard.vk;
        // SAFETY: the device mutex is held for the build, the record and the submit; the frame is
        // ours alone; the staging and readback are mapped at the offsets the command buffer reads.
        unsafe {
            if f.r1.is_none() {
                f.r1 = Some(Box::new(Rank1::build(vk, rows, cols, limbs)?));
            }
            let r1 = f.r1.as_deref_mut()?;
            let pos = r1
                .cbs
                .iter()
                .position(|c| c.0 == iters && c.1 == start_rows);
            let (cb, words) = match pos {
                Some(i) => (r1.cbs[i].2, r1.cbs[i].3),
                None => {
                    let (cb, words) = r1.record(vk, iters, start_rows);
                    r1.cbs.push((iters, start_rows, cb, words));
                    (cb, words)
                }
            };
            let st = r1.stage;
            std::ptr::copy_nonoverlapping(m.as_ptr().cast::<u8>(), st, 8 * m.len());
            std::ptr::copy_nonoverlapping(start.as_ptr().cast::<u8>(), st.add(8 * m.len()), 8 * ls);
            vk.submit_wait(cb);
            drop(guard);
            let fl = std::slice::from_raw_parts(r1.back.cast::<u32>(), words);
            if fl.iter().any(|&w| w != 0) {
                return None;
            }
            let out =
                std::slice::from_raw_parts(r1.back.add(FLAG_BYTES).cast::<f64>(), rows * cols);
            Some(out.to_vec())
        }
    }
}
