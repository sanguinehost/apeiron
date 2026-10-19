//! simt_nn::coop — **the cooperative-row GEMV: lanes share a row, no split-K partials.**
//!
//! The shipped `simt::gemv_dp4a_dyn` maps **one thread per output row** (`R =
//! GlobalInvocationId.x`, `dp4a.rs:322`): thread `r` walks all of K by itself. At m=896 with
//! 128-lane workgroups that is 7 workgroups on an 84-SM card, so occupancy has to be bought
//! elsewhere — which is what split-K does, multiplying the grid by 28-32 K-segments whose
//! partials a second pass (or the consumer's fold) then sums.
//!
//! `results/decode-suite.md` measured the consequences. The dial sweep says that arrangement is
//! already at its optimum (20/20 cells; the shipped 128/16384 wins, and `DEC_SPLITK=1` costs
//! 2.1×), and the roofline says the small shapes still only reach **79 GB/s** where `lm_head`
//! reaches 874. So the remaining lever is not a dial, it is the mapping.
//!
//! llama.cpp's Vulkan `mul_mat_vec` (`ggml-vulkan.cpp:4950`, `mul_mat_vec.comp`) inverts it: a
//! workgroup owns a few ROWS, its lanes cooperate along K, and the per-row reduction happens
//! inside the workgroup. Occupancy comes from having many small workgroups rather than from
//! splitting K, so there are no partials to write, re-read and reduce.
//!
//! This is that mapping on our chart. `rows_per_wg` rows per workgroup, `local / rows_per_wg`
//! lanes cooperating on each, each lane folding a contiguous block range with the SAME
//! [`Kb::dp4a_chart_fold`] the fused-norm gemv uses — so the arithmetic per block is
//! bit-identical to the shipped kernel and only the DECOMPOSITION differs. The per-row reduce is
//! the shared-memory tree, deliberately: the corpus already measured subgroup reduction as no win
//! on this decode path (`Kb::tree_reduce`'s note, 2026-07-08, 944.6 vs 938.5 µs/token), so
//! borrowing it here would import a refuted change into a test of a different one.
//!
//! Bindings are the shipped kernel's, unchanged, so this is a drop-in for `make_gemv`'s chart
//! branch: 0 = W (k-packed col-major codes), 1 = XQ, 2 = y, 3 = DM (f16 `(d, lo)` per block·row),
//! 4 = XS (`Σ_block x`), 5 = SX (per-block activation scale).

use super::*;

impl Kb {
    /// The `WorkgroupId.x` builtin, lazily made. The row a workgroup owns is a function of WHICH
    /// workgroup it is, which `GlobalInvocationId` can only recover by dividing out the local
    /// size.
    fn wg_index(&mut self) -> u32 {
        if self.wgid == 0 {
            let ptr_in_v3 = self.id();
            let uv3 = self.uvec3;
            self.ty(OP_TYPE_POINTER, &[ptr_in_v3, SC_INPUT, uv3]);
            self.wgid = self.id();
            let v = self.wgid;
            self.ty(OP_VARIABLE, &[ptr_in_v3, v, SC_INPUT]);
            self.deco(&[OP_DECORATE, v, DEC_BUILT_IN, BUILTIN_WORKGROUP_ID]);
        }
        let (uv3, u, v) = (self.uvec3, self.uint, self.wgid);
        let loaded = self.e(OP_LOAD, uv3, &[v]);
        self.e(OP_COMPOSITE_EXTRACT, u, &[loaded, 0])
    }
}

const BUILTIN_WORKGROUP_ID: u32 = 26;
const OP_GROUP_NON_UNIFORM_FADD: u32 = 350; // verified against SPIRV-Headers spirv.h:2062 (349 is IAdd)
const SCOPE_SUBGROUP: u32 = 3;
const GROUP_OP_REDUCE: u32 = 0;

impl Kb {
    /// `subgroupAdd(v)` — the whole subgroup's sum, in every lane, with NO shared memory and NO
    /// barrier. This is what llama.cpp's CUDA `mmvq` gets for free by giving each WARP a whole row
    /// (`mmvq.cu:462`, `rows_per_cuda_block = nwarps` under its `small_k` heuristic): the reduce
    /// is a warp shuffle, not a tree.
    fn sg_fadd(&mut self, v: u32) -> u32 {
        self.sg_arith = true;
        let f = self.float;
        let (sc, op) = (self.cu(SCOPE_SUBGROUP), GROUP_OP_REDUCE);
        self.e(OP_GROUP_NON_UNIFORM_FADD, f, &[sc, op, v])
    }
}

/// **Cooperative-row DP4A GEMV.** Grid is `(m.div_ceil(rows_per_wg), 1)`; `local` lanes per
/// workgroup, `local / rows_per_wg` of them per row.
///
/// `rows_per_wg` must divide `local` and both must be powers of two — the lane split is a shift
/// and a mask, because this codebase does not emit `UDiv`/`UMod` in a hot loop.
pub fn gemv_coop_module(
    m: u32,
    k: u32,
    local: u32,
    rows_per_wg: u32,
    carrier: crate::gpu::simt::Carrier,
) -> Vec<u32> {
    gemv_coop_module_sg(m, k, local, rows_per_wg, carrier, 0)
}

/// [`gemv_coop_module`] with the subgroup width declared. Pass `subgroup = 32` on NVIDIA to take
/// the shuffle path when a row lands on exactly one subgroup; `0` forces the shared-memory tree.
pub fn gemv_coop_module_sg(
    m: u32,
    k: u32,
    local: u32,
    rows_per_wg: u32,
    carrier: crate::gpu::simt::Carrier,
    subgroup: u32,
) -> Vec<u32> {
    gemv_coop_module_sc(m, k, local, rows_per_wg, carrier, subgroup, false)
}

/// [`gemv_coop_module_sg`] with the **scale face** chosen. Bindings are UNCHANGED at six — `sup`
/// shares `dm`'s buffer at a compile-time offset. See [`crate::gpu::nn::ChartScale`].
#[allow(clippy::too_many_arguments)]
pub fn gemv_coop_module_sc(
    m: u32,
    k: u32,
    local: u32,
    rows_per_wg: u32,
    carrier: crate::gpu::simt::Carrier,
    subgroup: u32,
    two_level: bool,
) -> Vec<u32> {
    assert!(local % rows_per_wg == 0, "rows_per_wg must divide local");
    assert!(local.is_power_of_two() && rows_per_wg.is_power_of_two());
    assert!(k % 32 == 0, "the chart is per-32-block");
    let lanes_per_row = local / rows_per_wg;
    let blocks = k / 32;
    // each lane folds a contiguous run of blocks; the last lanes may get nothing, and
    // `dp4a_chart_fold`'s loop simply does not execute when start >= bound
    let chunk = blocks.div_ceil(lanes_per_row);
    let lane_shift = lanes_per_row.trailing_zeros();
    let lane_mask = lanes_per_row - 1;

    let mut k_ = Kb::new();
    let w = k_.ssbo_u(0);
    let xq = k_.ssbo_u(1);
    let y = k_.ssbo_f(2);
    let dm = k_.ssbo_u(3);
    let xs = k_.ssbo_f(4);
    let sx = k_.ssbo_f(5);
    // NOTE: the two-level face adds NO binding — its per-row super-pairs live at a compile-time
    // offset inside `dm`'s own buffer. See `ChartScale`.

    let tid = k_.local_index();
    let wg = k_.wg_index();

    // row_in_wg = tid >> log2(lanes_per_row) ; lane = tid & (lanes_per_row - 1)
    let cshift = k_.cu(lane_shift);
    let u = k_.uint;
    let row_in_wg = k_.e(OP_SHIFT_RIGHT_LOGICAL2, u, &[tid, cshift]);
    let cmask = k_.cu(lane_mask);
    let lane = k_.e(OP_BITWISE_AND, u, &[tid, cmask]);
    let crpw = k_.cu(rows_per_wg);
    let rbase = k_.imul(wg, crpw);
    let r = k_.iadd(rbase, row_in_wg);

    // the per-lane partial lands in shared memory; zero first so an out-of-range row still
    // contributes a defined value to its row's reduction
    // When a row is owned by EXACTLY one subgroup, the reduce is a shuffle and the shared array
    // and its log2(lanes) barriers disappear entirely. That is the llama.cpp CUDA shape; the
    // shared-memory tree below is the fallback for every other lane split.
    let sg_path = lanes_per_row == subgroup;
    let (red, redp) = k_.shared_f32(local);
    let zf = k_.cf(0.0);
    if !sg_path {
        k_.shared_store(red, redp, tid, zf);
    }

    let cm = k_.cu(m);
    let in_range = k_.ult(r, cm);
    k_.iff(in_range, |mut kb| {
        let cchunk = kb.cu(chunk);
        let start = kb.imul(lane, cchunk);
        let raw_end = kb.iadd(start, cchunk);
        let cblocks = kb.cu(blocks);
        let end = kb.umin(raw_end, cblocks);
        // `cm` is the ID of the constant m (the col-major row stride), not the literal —
        // dp4a_chart_fold multiplies by it, so a raw value emits a reference to an undefined id.
        let sc = if two_level {
            crate::gpu::nn::ChartScale::two_level(dm, m, blocks)
        } else {
            crate::gpu::nn::ChartScale::dm(dm)
        };
        let part = kb.dp4a_chart_fold(
            w,
            sc,
            r,
            cm,
            start,
            end,
            xq,
            0,
            sx,
            0,
            xs,
            0,
            carrier,
            false,
            None,
            carrier.plane_b_off(m, k),
        );
        if sg_path {
            // every lane of this subgroup holds the row's total after the shuffle
            let tot = kb.sg_fadd(part);
            kb.shared_store(red, redp, tid, tot);
        } else {
            kb.shared_store(red, redp, tid, part);
        }
        kb
    });
    if !sg_path {
        k_.wg_barrier();
    }

    // segmented tree reduce: each row's lanes are CONTIGUOUS in tid, so the usual halving works
    // per row with no cross-row leakage as long as the stride stays below lanes_per_row.
    let mut stride = if sg_path { 0 } else { lanes_per_row / 2 };
    while stride >= 1 {
        let cs = k_.cu(stride);
        let lt = k_.ult(lane, cs);
        k_.iff(lt, |mut kb| {
            let a = kb.shared_load(red, redp, tid);
            let ti = kb.iadd(tid, cs);
            let b = kb.shared_load(red, redp, ti);
            let f = kb.float;
            let s = kb.e(OP_FADD, f, &[a, b]);
            kb.shared_store(red, redp, tid, s);
            kb
        });
        // OUTSIDE the conditional: a barrier must be reached by every lane of the workgroup
        k_.wg_barrier();
        stride /= 2;
    }

    let zero = k_.cu(0);
    let is_lead = k_.e(OP_IEQ, k_.bool_t, &[lane, zero]);
    k_.iff(is_lead, |mut kb| {
        let ok = kb.ult(r, cm);
        kb.iff(ok, |mut kb| {
            let v = kb.shared_load(red, redp, tid);
            kb.store_f(y, r, v);
            kb
        });
        kb
    });
    k_.finish(local)
}

pub fn gemv_coop_bytes(
    m: u32,
    k: u32,
    local: u32,
    rows_per_wg: u32,
    carrier: crate::gpu::simt::Carrier,
) -> Vec<u8> {
    bytes(gemv_coop_module(m, k, local, rows_per_wg, carrier))
}

pub fn gemv_coop_sg_bytes(
    m: u32,
    k: u32,
    local: u32,
    rows_per_wg: u32,
    carrier: crate::gpu::simt::Carrier,
    subgroup: u32,
) -> Vec<u8> {
    bytes(gemv_coop_module_sg(
        m,
        k,
        local,
        rows_per_wg,
        carrier,
        subgroup,
    ))
}

/// [`gemv_coop_sg_bytes`] with the scale face chosen — see [`gemv_coop_module_sc`]. The pipeline
/// still takes **6** bindings either way; only binding 3's layout differs.
#[allow(clippy::too_many_arguments)]
pub fn gemv_coop_bytes_sc(
    m: u32,
    k: u32,
    local: u32,
    rows_per_wg: u32,
    carrier: crate::gpu::simt::Carrier,
    subgroup: u32,
    two_level: bool,
) -> Vec<u8> {
    bytes(gemv_coop_module_sc(
        m,
        k,
        local,
        rows_per_wg,
        carrier,
        subgroup,
        two_level,
    ))
}
