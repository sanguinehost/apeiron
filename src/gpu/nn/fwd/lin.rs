//! simt_nn::fwd::lin — the dense-layer forward product as f32 SPIR-V, the twin of
//! `learn::grad::dense::linear_fwd`.
//!
//! **One register-tiled body for every linear GEMM.** A workgroup owns a `BM × BN` output tile and
//! steps the reduction in `BK = 16` slices. Each slice stages the `BM × 16` A tile and the `BN × 16`
//! B tile into two shared arrays (`As[kk·(BM+1) + m]`, `Bs[kk·(BN+1) + n]`; the `+1` row stride is
//! the bank-conflict pad) and every lane then holds `TM = BM/16` staged A words and `TN = BN/16`
//! staged B words and issues their `TM·TN` products as register `Fma`s. The tile is read from
//! shared once per lane per k, not once per output: the one-output-per-lane body it replaces read
//! two staged words per output per k, this body reads `TM + TN`, so the shared traffic per product
//! falls by `2BM·BN/(16(BM+BN))` — 8× at `BM = BN = 128` — and the global traffic falls by
//! `BM/16 = 8×` in the row direction and `BN/16` in the column direction.
//!
//! **The f32 association is the naive kernel's, word for word.** Every output is one accumulator
//! seeded `0.0` and updated by `Fma(a, b, acc)` for `k` strictly ascending: the block loop walks
//! `k0 = 0, 16, 32, …` and the inner loop walks `kk = 0 … 15` inside each block, so the accumulator
//! sees the same `k` order the naive loop does. A partial last block stages zeros, and
//! `fma(0, 0, acc) = acc` leaves the accumulator exactly where it was. No split-k, no reduction
//! across k, no cooperative-matrix or f16 path: those change the words and are not this kernel.
//!
//! The backward adjoints that invert this map live in the sibling `bwd` module and call
//! [`tiled_gemm`] with their own operand description; there is one emitted body, three operand
//! descriptions.

use super::super::*;

/// Workgroup width in lanes.
pub(crate) const WG: u32 = 256;
/// Lanes along the output-column direction; the tile-column stride is [`TX`].
pub(crate) const TX: u32 = 16;
/// Lanes along the output-row direction; the tile-row stride is [`TY`].
pub(crate) const TY: u32 = 16;
/// Reduction width staged per block.
pub(crate) const BK: u32 = 16;
/// Output-tile rows per workgroup (8 lanes' worth of `TM = 8` accumulators each).
pub(crate) const BM: u32 = 128;
/// Output-tile columns per workgroup.
pub(crate) const BN: u32 = 128;

/// One staged operand: which buffer a `BM × BK` (A) or `BN × BK` (B) tile is read from, and where.
pub(crate) struct Weld {
    /// the f32 SSBO binding holding the operand
    pub buf: u32,
    /// the operand's inner extent in f32 words (the stride between its rows), as a `u32` constant id
    pub rs: u32,
    /// extent of the operand's tile dimension (`m` for A, `n` for B), for the zero guard
    pub tile_extent: u32,
    /// extent of the reduction dimension, for the zero guard
    pub k_extent: u32,
    /// `true` when the reduction index is the inner one (`buf[tile·rs + k]`); `false` when the tile
    /// index is (`buf[k·rs + tile]`). It also fixes which half of the lane id walks the tile: the
    /// inner dimension is the one consecutive lanes must cover for a coalesced global read.
    pub k_inner: bool,
}

/// `dst = A·B` (or `A·Bᵀ`, per the operands' [`Weld`]) as the tiled body: `rows × cols` output,
/// reduction length `kdim`, `BM × BN` tile per workgroup, `BK`-wide slice, `TM × TN` accumulators
/// per lane. Returns the workgroup width for `Kb::finish`.
///
/// Bindings are the caller's: `a.buf`, `b.buf`, `out`, in that descriptor order.
///
/// Complexity: `⌈rows/BM⌉·⌈cols/BN⌉` workgroups; per lane per reduction step `TM + TN` shared
/// loads and `TM·TN` FMAs, so `O(kdim·(TM+TN))` shared loads and `O(kdim·TM·TN)` FMAs per lane;
/// global reads are `rows·kdim·⌈cols/BN⌉ + cols·kdim·⌈rows/BM⌉` words and writes `rows·cols`,
/// against the untiled `2·rows·cols·kdim`. O(1) shared words per lane beyond the staged tiles:
/// `BK·(BM+BN+2)`.
pub(crate) fn tiled_gemm(
    k: &mut Kb,
    a: &Weld,
    b: &Weld,
    out: u32,
    rows: u32,
    cols: u32,
    kdim: u32,
) -> u32 {
    assert!(
        rows > 0 && cols > 0 && kdim > 0,
        "tiled_gemm: shape must be non-zero"
    );
    assert!(
        BM % TY == 0 && BN % TX == 0,
        "tiled_gemm: tile must divide by the lane grid"
    );
    let tm = BM / TY;
    let tn = BN / TX;
    let sa_stride = BM + 1;
    let sb_stride = BN + 1;
    let (sa, sap) = k.shared_f32(BK * sa_stride);
    let (sb, sbp) = k.shared_f32(BK * sb_stride);
    let gid = k.gid_comp(0);
    let tid = k.local_index();
    let cwg = k.cu(WG);
    let ctx = k.cu(TX);
    let cbk = k.cu(BK);
    let c0 = k.cu(0);
    let ctile_r = k.cu(BM);
    let ctile_c = k.cu(BN);
    let csa_stride = k.cu(sa_stride);
    let csb_stride = k.cu(sb_stride);
    let zero = k.cf_bits(0x0000_0000);
    // Lane `tid` belongs to workgroup `gid / WG`; that workgroup is one output tile, laid out
    // `tile_r·⌈cols/BN⌉ + tile_c` in the grid.
    let tile = k.udiv(gid, cwg);
    let tiles_c = k.cu(cols.div_ceil(BN));
    let tile_r = k.udiv(tile, tiles_c);
    let tile_c = k.umod(tile, tiles_c);
    let tx = k.umod(tid, ctx);
    let ty = k.udiv(tid, ctx);
    let row_base = k.imul(tile_r, ctile_r);
    let col_base = k.imul(tile_c, ctile_c);
    let ctb = k.cu(tm);
    let ctn = k.cu(tn);
    let a_off = k.imul(ty, ctb);
    let b_off = k.imul(tx, ctn);
    // `TM·TN` accumulators, one OpVariable each so the block loop carries them across iterations.
    let mut accs: Vec<u32> = Vec::new();
    let mut n = 0;
    while n < tm * tn {
        accs.push(k.fvar_f());
        n += 1;
    }
    for v in &accs {
        k.store_var(*v, zero);
    }
    let ck = k.cu(kdim.div_ceil(BK));
    k.loop_u(c0, ck, |mut k, kt| {
        let k0 = k.imul(kt, cbk);
        // A: lane walks the tile dimension, the reduction index is `tx`.
        let (a_free, a_kk) = if a.k_inner { (ty, tx) } else { (tx, ty) };
        stage(
            &mut k, a, sa, sap, csa_stride, a_free, a_kk, row_base, k0, tm, zero,
        );
        let (b_free, b_kk) = if b.k_inner { (ty, tx) } else { (tx, ty) };
        stage(
            &mut k, b, sb, sbp, csb_stride, b_free, b_kk, col_base, k0, tn, zero,
        );
        k.wg_barrier();
        let mut acc: Vec<u32> = Vec::new();
        for v in &accs {
            acc.push(k.load_var_f(*v));
        }
        let mut kk = 0;
        while kk < BK {
            let ckk = k.cu(kk);
            let kka = k.imul(ckk, csa_stride);
            let abase = k.iadd(kka, a_off);
            let mut av: Vec<u32> = Vec::new();
            let mut i = 0;
            while i < tm {
                let ci = k.cu(i);
                let idx = k.iadd(abase, ci);
                av.push(k.shared_load(sa, sap, idx));
                i += 1;
            }
            let kkb = k.imul(ckk, csb_stride);
            let bbase = k.iadd(kkb, b_off);
            let mut bv: Vec<u32> = Vec::new();
            let mut j = 0;
            while j < tn {
                let cj = k.cu(j);
                let idx = k.iadd(bbase, cj);
                bv.push(k.shared_load(sb, sbp, idx));
                j += 1;
            }
            let mut i = 0;
            while i < tm {
                let mut j = 0;
                while j < tn {
                    let at = (i * tn + j) as usize;
                    acc[at] = k.fma(av[i as usize], bv[j as usize], acc[at]);
                    j += 1;
                }
                i += 1;
            }
            kk += 1;
        }
        let mut n = 0;
        while n < accs.len() {
            k.store_var(accs[n], acc[n]);
            n += 1;
        }
        k.wg_barrier();
        k
    });
    // Writeback: lane `(ty, tx)` owns `(row_base + ty·TM + i, col_base + tx·TN + j)`.
    let crows = k.cu(rows);
    let ccols = k.cu(cols);
    let mut i = 0;
    while i < tm {
        let tyi = k.imul(ty, ctb);
        let ci = k.cu(i);
        let roff = k.iadd(tyi, ci);
        let row = k.iadd(row_base, roff);
        let r_ok = k.ult(row, crows);
        let mut j = 0;
        while j < tn {
            let txj = k.imul(tx, ctn);
            let cj = k.cu(j);
            let coff = k.iadd(txj, cj);
            let col = k.iadd(col_base, coff);
            let c_ok = k.ult(col, ccols);
            let bt = k.bool_t;
            let ok = k.e(OP_LOGICAL_AND, bt, &[r_ok, c_ok]);
            let s = k.load_var_f(accs[(i * tn + j) as usize]);
            k.iff(ok, |mut k| {
                let base = k.imul(row, ccols);
                let ix = k.iadd(base, col);
                k.store_f(out, ix, s);
                k
            });
            j += 1;
        }
        i += 1;
    }
    WG
}

/// Stage one `BK`-wide operand slice: `blocks` elements per lane, `As[kk·stride + el]` from
/// `w.buf`. A lane covers the tile dimension (`t_free + t·16`) and holds one reduction index
/// (`t_kk`); out-of-range elements are staged as `+0.0`, which `Fma` leaves the accumulator on.
///
/// Complexity: `O(blocks)` global reads and shared stores per lane per block, `O(1)` space.
fn stage(
    k: &mut Kb,
    w: &Weld,
    sv: u32,
    sp: u32,
    stride: u32,
    t_free: u32,
    t_kk: u32,
    tile_base: u32,
    k0: u32,
    blocks: u32,
    zero: u32,
) {
    let bt = k.bool_t;
    let ck_extent = k.cu(w.k_extent);
    let ct_extent = k.cu(w.tile_extent);
    let mut t = 0;
    while t < blocks {
        let toff = k.cu(t * TY);
        let el = k.iadd(t_free, toff);
        let g_el = k.iadd(tile_base, el);
        let g_k = k.iadd(k0, t_kk);
        let kt = k.imul(t_kk, stride);
        let sidx = k.iadd(kt, el);
        let (gidx, el_ok, k_ok) = if w.k_inner {
            let t1 = k.imul(g_el, w.rs);
            let ix = k.iadd(t1, g_k);
            (ix, k.ult(g_el, ct_extent), k.ult(g_k, ck_extent))
        } else {
            let t1 = k.imul(g_k, w.rs);
            let ix = k.iadd(t1, g_el);
            (ix, k.ult(g_el, ct_extent), k.ult(g_k, ck_extent))
        };
        let ok = k.e(OP_LOGICAL_AND, bt, &[el_ok, k_ok]);
        k.shared_store(sv, sp, sidx, zero);
        k.iff(ok, |mut k| {
            let v = k.load_f(w.buf, gidx);
            k.shared_store(sv, sp, sidx, v);
            k
        });
        t += 1;
    }
}

/// **Linear forward (f32)** — the device twin of `learn::grad::dense::linear_fwd`: `x` is `n × i`,
/// `w` is `o × i`, the output `y` is `n × o` and `y[r,c] = Σ_k x[r·i + k] · w[c·i + k]` in f32 with
/// fused multiply-add accumulation. Workgroup `q = ⌈o/BN⌉·r_tile + c_tile` owns the `BM × BN`
/// output tile at `(BM·r_tile, BN·c_tile)`; lane `tid` holds `(TM × TN)` accumulators stepping `k`
/// in ascending 16-wide slices ([`tiled_gemm`]). Grid `⌈n/BM⌉·⌈o/BN⌉`, LocalSize 256.
///
/// Bindings: 0 = x (f32, n×i), 1 = w (f32, o×i), 2 = y (f32, n×o).
///
/// Complexity: see [`tiled_gemm`] with `rows = n`, `cols = o`, `kdim = i`.
pub fn linear_fwd_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    assert!(
        n > 0 && i > 0 && o > 0,
        "linear_fwd: shape must be non-zero"
    );
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let w = k.ssbo_f(1);
    let y = k.ssbo_f(2);
    let a = Weld {
        buf: x,
        rs: k.cu(i),
        tile_extent: n,
        k_extent: i,
        k_inner: true,
    };
    let b = Weld {
        buf: w,
        rs: k.cu(i),
        tile_extent: o,
        k_extent: i,
        k_inner: true,
    };
    let local = tiled_gemm(&mut k, &a, &b, y, n, o, i);
    k.finish(local)
}

/// [`linear_fwd_module`] as bytes.
pub fn linear_fwd_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    bytes(linear_fwd_module(n, i, o))
}

/// Host wrapper: build and record the linear forward product. `x` is the device buffer of `n·i`
/// f32, `w` of `o·i`, `y` of `n·o`. Grid `(⌈n/BM⌉·⌈o/BN⌉, 1, 1)`, one `BM × BN` output tile per
/// workgroup, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_fwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    x: crate::gpu::vulkan::Handle,
    w: crate::gpu::vulkan::Handle,
    y: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &linear_fwd_bytes(n, i, o),
        &[x, w, y],
        n.div_ceil(BM) * o.div_ceil(BN),
        1,
        "linear_fwd",
    )
}
