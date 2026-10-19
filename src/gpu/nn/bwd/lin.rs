//! simt_nn::bwd::lin — the two dense-layer backward products as f32 SPIR-V, the twins of
//! `learn::grad::dense::{linear_bwd_input,linear_bwd_weight}`.
//!
//! Both kernels own one 16×16 output tile per workgroup: 256 threads, thread `(ty, tx)` writes
//! output `(row_base + ty, col_base + tx)`, and the reduction length `K` is stepped in 16-wide
//! blocks. Each block stages a 16×16 A tile and a 16×16 B tile into two 256-word shared arrays
//! (`As[ty·16+kk]`, `Bs[kk·16+tx]`), so one block of A and one of B serves all 256 outputs of the
//! tile instead of every output re-reading its own row and column. The k accumulation stays
//! ascending `0..K` in 16-wide steps with the same per-k `Fma(a, b, acc)`, so the f32 association
//! is the naive kernel's and the outputs are bit-identical; a partial last tile stages zeros, and
//! `fma(0, 0, acc) = acc` leaves the accumulator exactly where it was.

use super::super::*;

/// **Linear input adjoint (f32)** — the device twin of `learn::grad::dense::linear_bwd_input`:
/// `dy` is `n × o`, `w` is `o × i`, the output `dx` is `n × i` and
/// `dx[r,c] = Σ_k dy[r·o + k] · w[k·i + c]` in f32 with fused multiply-add accumulation.
/// Workgroup `q = ⌈i/16⌉·r_tile + c_tile` owns the 16×16 output tile at `(16·r_tile, 16·c_tile)`;
/// lane `tid = 16·ty + tx` writes `(16·r_tile + ty, 16·c_tile + tx)` and walks `k` in ascending
/// 16-wide blocks. Grid `⌈n/16⌉·⌈i/16⌉`, LocalSize 256.
///
/// Bindings: 0 = dy (f32, n×o), 1 = w (f32, o×i), 2 = dx (f32, n×i).
///
/// Complexity: per dispatch `⌈n/16⌉⌈i/16⌉` workgroups each read one 16-entry row of `dy` and one
/// 16-entry row of `w` per k block (`o/16` blocks) and write 256 words: `n·o·⌈i/16⌉` words of
/// `dy`, `o·i·⌈n/16⌉` of `w` and `n·i` of `dx` — at square multiples of 16 that is
/// `2·n·i·o/16 + n·i`, 8× below the naive `2·n·i·o + n·i` (P89); each thread issues `2·o`
/// shared loads and `o` FMAs, `O(o)` per thread, `O(1)` lanes (P86).
pub fn linear_bwd_input_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    const WG: u32 = 256;
    const TILE: u32 = 16;
    assert!(
        n > 0 && i > 0 && o > 0,
        "linear_bwd_input: shape must be non-zero"
    );
    let mut k = Kb::new();
    let dy = k.ssbo_f(0);
    let w = k.ssbo_f(1);
    let dx = k.ssbo_f(2);
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
    let tiles_c = k.cu(i.div_ceil(TILE));
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
    let ck = k.cu(o.div_ceil(TILE));
    k.loop_u(c0, ck, |mut k, kt| {
        let k0 = k.imul(kt, c16);
        // A = dy: thread (ty, tx) stages dy[row_base+ty][k0+tx] as As[ty·16+tx].
        let ar = k.iadd(row_base, ty);
        let ak = k.iadd(k0, tx);
        let a_r = k.ult(ar, cn);
        let a_k = k.ult(ak, co);
        let a_ok = k.e(OP_LOGICAL_AND, bt, &[a_r, a_k]);
        k.shared_store(sa, sap, tid, zero);
        k.iff(a_ok, |mut k| {
            let arow = k.imul(ar, co);
            let ia = k.iadd(arow, ak);
            let v = k.load_f(dy, ia);
            k.shared_store(sa, sap, tid, v);
            k
        });
        // B = w: thread (ty, tx) stages w[k0+ty][col_base+tx] as Bs[ty·16+tx].
        let bk = k.iadd(k0, ty);
        let b_k = k.ult(bk, co);
        let b_c = k.ult(col, ci);
        let b_ok = k.e(OP_LOGICAL_AND, bt, &[b_k, b_c]);
        k.shared_store(sb, sbp, tid, zero);
        k.iff(b_ok, |mut k| {
            let brow = k.imul(bk, ci);
            let ib = k.iadd(brow, col);
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
    let o_c = k.ult(col, ci);
    let o_ok = k.e(OP_LOGICAL_AND, bt, &[o_r, o_c]);
    let s = k.load_var_f(acc);
    k.iff(o_ok, |mut k| {
        let row_off = k.imul(row, ci);
        let ix = k.iadd(row_off, col);
        k.store_f(dx, ix, s);
        k
    });
    k.finish(WG)
}

/// [`linear_bwd_input_module`] as bytes.
pub fn linear_bwd_input_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    bytes(linear_bwd_input_module(n, i, o))
}

/// **Linear weight adjoint (f32)** — the device twin of
/// `learn::grad::dense::linear_bwd_weight`: `dy` is `n × o`, `x` is `n × i`, the output `dw` is
/// `o × i` and `dw[r,c] = Σ_k dy[k·o + r] · x[k·i + c]` in f32 with fused multiply-add
/// accumulation. Workgroup `q = ⌈i/16⌉·r_tile + c_tile` owns the 16×16 output tile at
/// `(16·r_tile, 16·c_tile)`; lane `tid = 16·ty + tx` writes `(16·r_tile + ty, 16·c_tile + tx)`
/// and walks `k` in ascending 16-wide blocks. The A tile is staged transposed
/// (`As[tx·16+ty] = dy[k0+ty][row_base+tx]`) so the global `dy` reads stay row-contiguous while
/// the compute step reads `As[ty·16+kk]` like the input adjoint. Grid
/// `⌈o/16⌉·⌈i/16⌉`, LocalSize 256.
///
/// Bindings: 0 = dy (f32, n×o), 1 = x (f32, n×i), 2 = dw (f32, o×i).
///
/// Complexity: per dispatch `⌈o/16⌉⌈i/16⌉` workgroups each read one 16-entry k row of `dy` and
/// one of `x` per k block (`n/16` blocks) and write 256 words: `n·o·⌈i/16⌉` words of `dy`,
/// `n·i·⌈o/16⌉` of `x` and `o·i` of `dw` — at square multiples of 16 that is
/// `2·n·i·o/16 + o·i`, 8× below the naive `2·n·i·o + o·i` (P89); each thread issues `2·n`
/// shared loads and `n` FMAs, `O(n)` per thread, `O(1)` lanes (P87).
pub fn linear_bwd_weight_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    const WG: u32 = 256;
    const TILE: u32 = 16;
    assert!(
        n > 0 && i > 0 && o > 0,
        "linear_bwd_weight: shape must be non-zero"
    );
    let mut k = Kb::new();
    let dy = k.ssbo_f(0);
    let x = k.ssbo_f(1);
    let dw = k.ssbo_f(2);
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
    let tiles_c = k.cu(i.div_ceil(TILE));
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
    let ck = k.cu(n.div_ceil(TILE));
    k.loop_u(c0, ck, |mut k, kt| {
        let k0 = k.imul(kt, c16);
        // A = dy, transposed: thread (ty, tx) reads dy[k0+ty][row_base+tx] and stages it at
        // As[tx·16+ty], so compute reads As[ty·16+kk] = dy[k0+kk][row_base+ty].
        let ar = k.iadd(row_base, tx);
        let ak = k.iadd(k0, ty);
        let a_r = k.ult(ar, co);
        let a_k = k.ult(ak, cn);
        let a_ok = k.e(OP_LOGICAL_AND, bt, &[a_r, a_k]);
        let st = k.imul(tx, c16);
        let ai = k.iadd(st, ty);
        k.shared_store(sa, sap, ai, zero);
        k.iff(a_ok, |mut k| {
            let arow = k.imul(ak, co);
            let ia = k.iadd(arow, ar);
            let v = k.load_f(dy, ia);
            k.shared_store(sa, sap, ai, v);
            k
        });
        // B = x: thread (ty, tx) stages x[k0+ty][col_base+tx] as Bs[ty·16+tx].
        let bk = k.iadd(k0, ty);
        let b_k = k.ult(bk, cn);
        let b_c = k.ult(col, ci);
        let b_ok = k.e(OP_LOGICAL_AND, bt, &[b_k, b_c]);
        k.shared_store(sb, sbp, tid, zero);
        k.iff(b_ok, |mut k| {
            let brow = k.imul(bk, ci);
            let ib = k.iadd(brow, col);
            let v = k.load_f(x, ib);
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
    let o_r = k.ult(row, co);
    let o_c = k.ult(col, ci);
    let o_ok = k.e(OP_LOGICAL_AND, bt, &[o_r, o_c]);
    let s = k.load_var_f(acc);
    k.iff(o_ok, |mut k| {
        let row_off = k.imul(row, ci);
        let ix = k.iadd(row_off, col);
        k.store_f(dw, ix, s);
        k
    });
    k.finish(WG)
}

/// [`linear_bwd_weight_module`] as bytes.
pub fn linear_bwd_weight_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    bytes(linear_bwd_weight_module(n, i, o))
}

/// Host wrapper: build and record the linear input adjoint. `dy` is the device buffer of `n·o`
/// f32, `w` of `o·i`, `dx` of `n·i`. Grid `(⌈n/16⌉·⌈i/16⌉, 1, 1)`, one 16×16 output tile per
/// workgroup, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_bwd_input_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    dy: crate::gpu::vulkan::Handle,
    w: crate::gpu::vulkan::Handle,
    dx: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &linear_bwd_input_bytes(n, i, o),
        &[dy, w, dx],
        n.div_ceil(16) * i.div_ceil(16),
        1,
        "linear_bwd_input",
    )
}

/// Host wrapper: build and record the linear weight adjoint. `dy` is the device buffer of `n·o`
/// f32, `x` of `n·i`, `dw` of `o·i`. Grid `(⌈o/16⌉·⌈i/16⌉, 1, 1)`, one 16×16 output tile per
/// workgroup, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_bwd_weight_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    dy: crate::gpu::vulkan::Handle,
    x: crate::gpu::vulkan::Handle,
    dw: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &linear_bwd_weight_bytes(n, i, o),
        &[dy, x, dw],
        o.div_ceil(16) * i.div_ceil(16),
        1,
        "linear_bwd_weight",
    )
}
