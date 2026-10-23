//! `learn::nanoquant::spd_inverse` on the device above a cutoff order, bit for bit.
//!
//! **Contract.** [`Inverse::run`] writes `spd_inverse(hw, r)` into the frame's `H⁻¹` words in every
//! bit and returns `Some(())`, or returns `None` when a device stage flags (a plane exponent past 62,
//! a product's range or fallback word, a checked word sum outside `i64`) or the host inverse of a
//! leaf refuses. On `None` the caller runs the host `spd_inverse` on `hw`, which computes the value
//! or reproduces the refusal.
//!
//! **Route.** Every recursion node of `spd_inverse` of order `n > cut` runs on the device: the block
//! reads, `S = C − BᵀX` and the assembly of `src/gpu/spirv/inv_map.comp`, and the four limb products
//! `X = A⁻¹·B`, `Bᵀ·X`, `Y = X·S⁻¹`, `Y·Xᵀ` (placement → `limb_gemm` → finish over planes built
//! on the device), each the `gemm_q32` of the same operands. A node of order `≤ cut` is a leaf: its
//! input is read back, the host `inv` (`spd_inverse` at that order, the CPU recursion's own call)
//! inverts it and the result is staged into the parent's words. The command stream is cut at each
//! leaf, so a tree with `L` leaves is `L + 1` submits; `cut` is at least the host recursion base, so
//! every device node is a node the CPU recursion splits.

use std::sync::PoisonError;

use super::admm::{copy, dev_buf, full, to_host, Map, Product};
use super::plane::PlaneBuild;
use super::{device, TILE};
use crate::gpu::vulkan::*;

/// The compiled word maps.
pub(super) static INV_MAP_SPV: &[u8] = include_bytes!("../spirv/inv_map.spv");

/// Flag words per device node: six plane flags and the `[range, fallback]` of four products.
const NODE_FLAGS: usize = 14;

/// One recorded submit and the order of the leaf whose input it reads back last.
struct Segment {
    cb: CommandBuffer,
    leaf: Option<usize>,
}

/// The device inverse of one frame's `hw` (order `r`) into its `H⁻¹` words.
pub(crate) struct Inverse {
    segs: Vec<Segment>,
    stage: *mut u8,
    back: *mut u8,
    /// Flag words at the head of the readback: the map flag, then each node's.
    flags: usize,
    /// Byte offset of the leaf input in the readback.
    leaf_at: usize,
}

/// Device nodes and the largest leaf order of the recursion below order `n` at cutoff `cut`.
///
/// # Complexity
///
/// `O(n / cut)` calls: one per node and leaf.
fn shape(n: usize, cut: usize) -> (usize, usize) {
    if n <= cut {
        return (0, n);
    }
    let (a, la) = shape(n / 2, cut);
    let (b, lb) = shape(n - n / 2, cut);
    (1 + a + b, la.max(lb))
}

/// Bytes of the shared `o` the products below order `n` write.
///
/// # Complexity
///
/// `O(n / cut)` calls.
fn o_need(n: usize, cut: usize, limbs: u32) -> usize {
    if n <= cut {
        return 0;
    }
    let (p, q) = (n / 2, n - n / 2);
    [(p, q, p), (q, q, p), (p, q, q), (p, p, q)]
        .iter()
        .map(|&(m, c, i)| Product::o_bytes(m, c, i, limbs))
        .chain([o_need(p, cut, limbs), o_need(q, cut, limbs)])
        .max()
        .unwrap_or(0)
}

/// The recording state: the open command buffer, the finished segments and the flag sources
/// recorded since the last cut.
struct Rec<'a> {
    vk: &'a Vk,
    limbs: u32,
    cut: usize,
    o: Handle,
    mflag: Handle,
    stage: Handle,
    back: Handle,
    leaf_at: usize,
    cb: CommandBuffer,
    segs: Vec<Segment>,
    pending: Vec<(Handle, usize)>,
    slot: usize,
    digs: Vec<Handle>,
    ups: Vec<(Handle, Vec<u8>)>,
}

impl Rec<'_> {
    /// A plane `[words, e, dig, flag]` of `rows × inner` words and its build.
    unsafe fn plane(&mut self, rows: usize, inner: usize) -> Option<([Handle; 4], PlaneBuild)> {
        let vk = self.vk;
        let blocks = inner.div_ceil(32);
        let padded = rows.div_ceil(TILE) * TILE;
        let b = [
            dev_buf(vk, 8 * rows * blocks * 32)?,
            dev_buf(vk, 4 * blocks)?,
            dev_buf(vk, padded * blocks * self.limbs as usize * 32)?,
            dev_buf(vk, 4)?,
        ];
        self.digs.push(b[2]);
        Some((b, PlaneBuild::new(vk, b, rows, inner, self.limbs)))
    }

    /// The product `a·bᵀ` (`m × p` over `inner`) and its output (row stride `p` padded to 64).
    unsafe fn product(
        &mut self,
        a: &[Handle; 4],
        b: &[Handle; 4],
        (m, p, inner): (usize, usize, usize),
    ) -> Option<(Product, Handle)> {
        let vk = self.vk;
        let x = dev_buf(vk, 8 * m.div_ceil(TILE) * TILE * p.div_ceil(TILE) * TILE)?;
        let dims = (m, p, inner, self.limbs);
        let pr = Product::new(
            vk,
            [a[2], a[1]],
            [b[2], b[1]],
            dims,
            self.o,
            x,
            &mut self.ups,
        )?;
        Some((pr, x))
    }

    /// The `inv_map.comp` map of the meta words `[mode, total, cols, a, b, c, d]` over
    /// `[s0, s1, s2, s3, dst]`.
    unsafe fn map(&mut self, meta: [usize; 7], bufs: [Handle; 5]) -> Option<Map> {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "orders and their products index u32 invocations"
        )]
        let words = meta.map(|w| w as u32);
        let [s0, s1, s2, s3, dst] = bufs;
        let set = [s0, s1, s2, s3, dst, self.mflag];
        Map::with(self.vk, INV_MAP_SPV, &words, meta[1], &set, &mut self.ups)
    }

    /// Ends the open segment: the leaf input `(h, order)` and every pending flag to the readback.
    unsafe fn cut_seg(&mut self, leaf: Option<(Handle, usize)>) {
        let (vk, cb) = (self.vk, self.cb);
        full(vk, cb);
        if let Some((h, n)) = leaf {
            copy(vk, cb, h, 0, self.back, self.leaf_at, 8 * n * n);
        }
        copy(vk, cb, self.mflag, 0, self.back, 0, 4);
        for (h, w) in std::mem::take(&mut self.pending) {
            copy(vk, cb, h, 0, self.back, 4 * self.slot, 4 * w);
            self.slot += w;
        }
        to_host(vk, cb);
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        self.segs.push(Segment {
            cb,
            leaf: leaf.map(|l| l.1),
        });
    }

    /// Opens a segment that first copies the staged leaf inverse (`order²` words) to `dst`.
    unsafe fn open(&mut self, dst: Option<(Handle, usize)>) {
        let vk = self.vk;
        self.cb = vk.command_buffer();
        vk.begin(self.cb);
        if let Some((d, n)) = dst {
            copy(vk, self.cb, self.stage, 0, d, 0, 8 * n * n);
            full(vk, self.cb);
        }
    }

    /// The inverse of the order-`n` words `h` into `dest`: a device node above the cutoff, a host
    /// leaf at or below it.
    unsafe fn child(&mut self, n: usize, h: Handle, dest: Handle) -> Option<()> {
        if n > self.cut {
            return self.node(n, h, dest);
        }
        self.cut_seg(Some((h, n)));
        self.open(Some((dest, n)));
        Some(())
    }

    /// One recursion node of order `n` over the words `h` (`n × n`) into `dest`, as
    /// `spd_inverse_at`: `A⁻¹` by recursion, `X = A⁻¹B`, `S = C − mean(BᵀX)`, `S⁻¹` by recursion,
    /// `Y = X·S⁻¹`, `H⁻¹ = [A⁻¹ + mean(Y·Xᵀ), −Y; −Yᵀ, S⁻¹]`.
    ///
    /// # Complexity
    ///
    /// Four products of `n³/8` multiply-adds per kept pair at `p = q = n/2`, six plane builds and
    /// seven maps of `O(n²)` invocations; `O(n²)` device words.
    unsafe fn node(&mut self, n: usize, h: Handle, dest: Handle) -> Option<()> {
        let vk = self.vk;
        let (p, q) = (n / 2, n - n / 2);
        let (pp, qp) = (p.div_ceil(TILE) * TILE, q.div_ceil(TILE) * TILE);
        let (ha, hs) = (dev_buf(vk, 8 * p * p)?, dev_buf(vk, 8 * q * q)?);
        let (ai, ai_b) = self.plane(p, p)?;
        let (bt, bt_b) = self.plane(q, p)?;
        let (xt, xt_b) = self.plane(q, p)?;
        let (x, x_b) = self.plane(p, q)?;
        let (si, si_b) = self.plane(q, q)?;
        let (y, y_b) = self.plane(p, q)?;
        let (px, xo) = self.product(&ai, &bt, (p, q, p))?;
        let (pb, bo) = self.product(&bt, &xt, (q, q, p))?;
        let (py, yo) = self.product(&x, &si, (p, q, q))?;
        let (pyx, yxo) = self.product(&y, &x, (p, p, q))?;
        let ex_a = self.map([0, p * p, p, n, 0, 0, 0], [h, h, h, h, ha])?;
        let ex_bt = self.map([0, q * p, p, n, p, 1, 0], [h, h, h, h, bt[0]])?;
        let ex_x = self.map([0, p * q, q, qp, 0, 0, 0], [xo, xo, xo, xo, x[0]])?;
        let ex_xt = self.map([0, q * p, p, qp, 0, 1, 0], [xo, xo, xo, xo, xt[0]])?;
        let s_map = self.map([1, q * q, q, n, p, qp, 0], [h, bo, bo, bo, hs])?;
        let ex_y = self.map([0, p * q, q, qp, 0, 0, 0], [yo, yo, yo, yo, y[0]])?;
        let asm = self.map([2, n * n, n, p, q, pp, qp], [ai[0], yxo, yo, si[0], dest])?;

        ex_a.record(vk, self.cb);
        full(vk, self.cb);
        self.child(p, ha, ai[0])?;
        let cb = self.cb;
        full(vk, cb);
        ai_b.record(vk, cb);
        ex_bt.record(vk, cb);
        full(vk, cb);
        bt_b.record(vk, cb);
        full(vk, cb);
        px.record(vk, cb);
        full(vk, cb);
        ex_x.record(vk, cb);
        ex_xt.record(vk, cb);
        full(vk, cb);
        x_b.record(vk, cb);
        xt_b.record(vk, cb);
        full(vk, cb);
        pb.record(vk, cb);
        full(vk, cb);
        s_map.record(vk, cb);
        full(vk, cb);
        self.pending.extend([
            (ai[3], 1),
            (bt[3], 1),
            (x[3], 1),
            (xt[3], 1),
            (px.out, 2),
            (pb.out, 2),
        ]);
        self.child(q, hs, si[0])?;
        let cb = self.cb;
        full(vk, cb);
        si_b.record(vk, cb);
        full(vk, cb);
        py.record(vk, cb);
        full(vk, cb);
        ex_y.record(vk, cb);
        full(vk, cb);
        y_b.record(vk, cb);
        full(vk, cb);
        pyx.record(vk, cb);
        full(vk, cb);
        asm.record(vk, cb);
        self.pending
            .extend([(si[3], 1), (y[3], 1), (py.out, 2), (pyx.out, 2)]);
        Some(())
    }
}

impl Inverse {
    /// The device inverse of the order-`r` words `hw` into `hi` at cutoff `cut`, recorded and
    /// initialised; `None` when `r ≤ cut` (no device node) or an allocation fails.
    ///
    /// # Safety
    ///
    /// `vk` is live with cooperative matrices, its mutex is held, and `hw` and `hi` hold `r²` words.
    ///
    /// # Complexity
    ///
    /// `O(r / cut)` nodes, each six planes, four products and seven maps of pipeline builds;
    /// device memory `O(r²)` words per tree level; one initialisation submit.
    pub(super) unsafe fn build(
        vk: &Vk,
        r: usize,
        cut: usize,
        limbs: u32,
        hw: Handle,
        hi: Handle,
    ) -> Option<Self> {
        let (nodes, leaf) = shape(r, cut);
        if nodes == 0 {
            return None;
        }
        let flags = 1 + NODE_FLAGS * nodes;
        let leaf_at = (4 * flags).div_ceil(8) * 8;
        let host = MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT;
        let (stage, stage_mem) =
            vk.buffer((8 * leaf * leaf) as u64, BUFFER_USAGE_TRANSFER_SRC, host)?;
        let back_bytes = (leaf_at + 8 * leaf * leaf) as u64;
        let (back, back_mem) = vk
            .buffer(
                back_bytes,
                BUFFER_USAGE_TRANSFER_DST,
                host | MEMORY_HOST_CACHED,
            )
            .or_else(|| vk.buffer(back_bytes, BUFFER_USAGE_TRANSFER_DST, host))?;
        let mut rec = Rec {
            vk,
            limbs,
            cut,
            o: dev_buf(vk, o_need(r, cut, limbs))?,
            mflag: dev_buf(vk, 4)?,
            stage,
            back,
            leaf_at,
            cb: std::ptr::null_mut(),
            segs: Vec::new(),
            pending: Vec::new(),
            slot: 1,
            digs: Vec::new(),
            ups: Vec::new(),
        };
        rec.open(None);
        (vk.fill_buffer)(rec.cb, rec.mflag, 0, 4, 0);
        full(vk, rec.cb);
        rec.node(r, hw, hi)?;
        rec.cut_seg(None);

        // Initialisation: zero digit planes (rows past the count stay zero) and the meta words.
        let up_bytes: usize = rec.ups.iter().map(|u| u.1.len()).sum();
        let (ups, ups_mem) = vk.buffer(up_bytes.max(16) as u64, BUFFER_USAGE_TRANSFER_SRC, host)?;
        let at0: *mut u8 = vk.map(ups_mem).cast();
        let cb = vk.command_buffer();
        vk.begin(cb);
        for d in &rec.digs {
            (vk.fill_buffer)(cb, *d, 0, WHOLE_SIZE, 0);
        }
        let mut at = 0usize;
        for (dst, b) in &rec.ups {
            std::ptr::copy_nonoverlapping(b.as_ptr(), at0.add(at), b.len());
            copy(vk, cb, ups, at, *dst, 0, b.len());
            at += b.len();
        }
        check((vk.end_cb)(cb), "vkEndCommandBuffer");
        vk.submit_wait(cb);
        Some(Self {
            segs: rec.segs,
            stage: vk.map(stage_mem).cast(),
            back: vk.map(back_mem).cast(),
            flags,
            leaf_at,
        })
    }

    /// Runs the segments, inverting each leaf on the host with `inv`; see the module contract.
    ///
    /// # Complexity
    ///
    /// `L + 1` submits for `L` leaves; host `O(leaf²)` words read and staged per leaf plus `inv`.
    pub(super) fn run<E>(&self, inv: &impl Fn(&[i64], usize) -> Result<Vec<i64>, E>) -> Option<()> {
        let dev = device()?;
        // SAFETY: the inverse is its frame's alone; the readback and staging are mapped at the
        // offsets the segments were recorded with, and each submit holds the device mutex.
        unsafe {
            std::ptr::write_bytes(self.back, 0, 4 * self.flags);
            for s in &self.segs {
                dev.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .vk
                    .submit_wait(s.cb);
                let fl = std::slice::from_raw_parts(self.back.cast::<u32>(), self.flags);
                if fl.iter().any(|&w| w != 0) {
                    return None;
                }
                if let Some(n) = s.leaf {
                    let h = std::slice::from_raw_parts(
                        self.back.add(self.leaf_at).cast::<i64>(),
                        n * n,
                    );
                    let out = {
                        let _inline = crate::engine::chart::pool::inline();
                        inv(h, n).ok()?
                    };
                    if out.len() != n * n {
                        return None;
                    }
                    std::ptr::copy_nonoverlapping(out.as_ptr().cast::<u8>(), self.stage, 8 * n * n);
                }
            }
        }
        Some(())
    }
}
