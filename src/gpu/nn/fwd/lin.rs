//! simt_nn::fwd::lin — the dense-layer forward product as f32 SPIR-V, the twin of
//! `learn::grad::dense::linear_fwd`.
//!
//! The kernel owns one 16×16 output tile per workgroup: 256 threads, thread `(ty, tx)` writes
//! output `(row_base + ty, col_base + tx)`, and the reduction length `K` is stepped in 16-wide
//! blocks. Each block stages a 16×16 A tile (x rows) and a 16×16 B tile (w transposed) into two
//! 256-word shared arrays (`As[ty·16+kk]`, `Bs[kk·16+tx]`), so one block of A and one of B serves
//! all 256 outputs of the tile instead of every output re-reading its own x row and w row. The k
//! accumulation stays ascending `0..K` in 16-wide steps with the same per-k `Fma(a, b, acc)` as
//! `bwd::lin`, so a partial last tile stages zeros and `fma(0, 0, acc) = acc` leaves the
//! accumulator exactly where it was.

use super::super::*;

/// **Linear forward (f32)** — the device twin of `learn::grad::dense::linear_fwd`: `x` is `n × i`,
/// `w` is `o × i`, the output `y` is `n × o` and `y[r,c] = Σ_k x[r·i + k] · w[c·i + k]` in f32 with
/// fused multiply-add accumulation. Workgroup `q = ⌈o/16⌉·r_tile + c_tile` owns the 16×16 output
/// tile at `(16·r_tile, 16·c_tile)`; lane `tid = 16·ty + tx` writes
/// `(16·r_tile + ty, 16·c_tile + tx)` and walks `k` in ascending 16-wide blocks. Grid
/// `⌈n/16⌉·⌈o/16⌉`, LocalSize 256.
///
/// Bindings: 0 = x (f32, n×i), 1 = w (f32, o×i), 2 = y (f32, n×o).
///
/// Complexity: per dispatch `⌈n/16⌉⌈o/16⌉` workgroups each read one 16-entry row of `x` and one
/// 16-entry row of `w` per k block (`i/16` blocks) and write 256 words: `n·i·⌈o/16⌉` words of `x`,
/// `o·i·⌈n/16⌉` of `w` and `n·o` of `y` — at square multiples of 16 that is
/// `2·n·i·o/16 + n·o`, 16× below the naive `2·n·i·o + n·o` (P92); each thread issues `2·i` shared
/// loads and `i` FMAs, `O(i)` per thread, `O(1)` lanes.
pub fn linear_fwd_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    const WG: u32 = 256;
    const TILE: u32 = 16;
    assert!(
        n > 0 && i > 0 && o > 0,
        "linear_fwd: shape must be non-zero"
    );
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let w = k.ssbo_f(1);
    let y = k.ssbo_f(2);
    let (sa, sap) = k.shared_f32(WG);
    let (sb, sbp) = k.shared_f32(WG);
    let gid = k.gid_comp(0);
    let tid = k.local_index();
    let cwg = k.cu(WG);
    let c16 = k.cu(TILE);
    let c0 = k.cu(0);
    let ci = k.cu(i);
    let co = k.cu(o);
    let cn = k.cu(n);
    // Lane `tid` belongs to the workgroup `gid / WG`; that workgroup is one output tile.
    let tile = k.udiv(gid, cwg);
    let tiles_c = k.cu(o.div_ceil(TILE));
    let tile_r = k.udiv(tile, tiles_c);
    let tile_c = k.umod(tile, tiles_c);
    let tx = k.umod(tid, c16);
    let ty = k.udiv(tid, c16);
    let row_base = k.imul(tile_r, c16);
    let col_base = k.imul(tile_c, c16);
    let row = k.iadd(row_base, ty);
    let col = k.iadd(col_base, tx);
    let acc = k.fvar_f();
    let zero = k.cf_bits(0x0000_0000);
    k.store_var(acc, zero);
    let bt = k.bool_t;
    let ck = k.cu(i.div_ceil(TILE));
    k.loop_u(c0, ck, |mut k, kt| {
        let k0 = k.imul(kt, c16);
        // A = x: thread (ty, tx) stages x[row_base+ty][k0+tx] as As[ty·16+tx].
        let ar = k.iadd(row_base, ty);
        let ak = k.iadd(k0, tx);
        let a_r = k.ult(ar, cn);
        let a_k = k.ult(ak, ci);
        let a_ok = k.e(OP_LOGICAL_AND, bt, &[a_r, a_k]);
        k.shared_store(sa, sap, tid, zero);
        k.iff(a_ok, |mut k| {
            let arow = k.imul(ar, ci);
            let ia = k.iadd(arow, ak);
            let v = k.load_f(x, ia);
            k.shared_store(sa, sap, tid, v);
            k
        });
        // B = w, transposed: thread (ty, tx) stages w[col_base+tx][k0+ty] as Bs[ty·16+tx], so
        // compute reads Bs[kk·16+tx] = w[col_base+tx][k0+kk].
        let bk = k.iadd(k0, ty);
        let b_k = k.ult(bk, ci);
        let b_c = k.ult(col, co);
        let b_ok = k.e(OP_LOGICAL_AND, bt, &[b_k, b_c]);
        k.shared_store(sb, sbp, tid, zero);
        k.iff(b_ok, |mut k| {
            let brow = k.imul(col, ci);
            let ib = k.iadd(brow, bk);
            let v = k.load_f(w, ib);
            k.shared_store(sb, sbp, tid, v);
            k
        });
        k.wg_barrier();
        k.loop_u(c0, c16, |mut k, kk| {
            let arow = k.imul(ty, c16);
            let ai = k.iadd(arow, kk);
            let a = k.shared_load(sa, sap, ai);
            let brow = k.imul(kk, c16);
            let bi = k.iadd(brow, tx);
            let b = k.shared_load(sb, sbp, bi);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(a, b, a0);
            k.store_var(acc, a1);
            k
        });
        k.wg_barrier();
        k
    });
    let o_r = k.ult(row, cn);
    let o_c = k.ult(col, co);
    let o_ok = k.e(OP_LOGICAL_AND, bt, &[o_r, o_c]);
    let s = k.load_var_f(acc);
    k.iff(o_ok, |mut k| {
        let row_off = k.imul(row, co);
        let ix = k.iadd(row_off, col);
        k.store_f(y, ix, s);
        k
    });
    k.finish(WG)
}

/// [`linear_fwd_module`] as bytes.
pub fn linear_fwd_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    bytes(linear_fwd_module(n, i, o))
}

/// Host wrapper: build and record the linear forward product. `x` is the device buffer of `n·i`
/// f32, `w` of `o·i`, `y` of `n·o`. Grid `(⌈n/16⌉·⌈o/16⌉, 1, 1)`, one 16×16 output tile per
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
        n.div_ceil(16) * o.div_ceil(16),
        1,
        "linear_fwd",
    )
}
