//! simt_nn::attn::bwd — the causal GQA attention backward as f32 SPIR-V, the twin of
//! `learn::grad::attn::attn_bwd_kern`.
//!
//! Four deterministic dispatches, no float atomics, every output sums in a fixed association:
//! `ds`, `dq` and `dk`/`dv` sum per-thread in ascending order; `rowsum` reduces its keys over a
//! fixed 256-lane tree.
//!
//! 1. `rowsum`: `r[h,i] = Σ_{j≤i} P[h,i,j]·Σ_e dO[i,h,e]·V[j,h/rep,e]`, one workgroup of 256
//!    threads per `(h,i)`; lane `l` owns `j = l, l+256, …`, and the per-key terms are summed by a
//!    shared-memory tree with a barrier per level.
//! 2. `ds`: `dS[h,i,j] = P[h,i,j]·(dP_ij − r[h,i])` with `dP_ij = Σ_e dO[i,h,e]·V[j,h/rep,e]`,
//!    one thread per `(h,i,j)`; `j > i` writes 0.
//! 3. `dq`: `dq[i,h,d] = scale·Σ_{j≤i} dS[h,i,j]·K[j,h/rep,d]`, one thread per `(i,h,d)`.
//! 4. `dk`/`dv`: `dk[j,g,d] = scale·Σ_{h∈g} Σ_{i≥j} dS[h,i,j]·Q[i,h,d]`,
//!    `dv[j,g,d] = Σ_{h∈g} Σ_{i≥j} P[h,i,j]·dO[i,h,d]`, one thread per `(j,g,d)`.
//!
//! Dispatch 2 materialises the score gradient once; before it, `dq` and `dk`/`dv` each recomputed
//! `dP_ij` (a dot product of `dO_i` and `V_j` over the head width) for every `(h,i,j)` they visited.
//! The materialised form pays `2·hd + 2` words in dispatch 2 and one `ds` word per visit in flight
//! 3 and 4, replacing `2·hd` words per visit in each of them. The arithmetic order is unchanged:
//! `dP_ij` sums `e` ascending with fused multiply-add, `dS` rounds once, `dq` sums `j` ascending and
//! `dk`/`dv` sum `h` then `i` ascending, so the f32 values are bit-identical to the three-pass form.
//!
//! The shapes `t`, `nh`, `nkv`, `hd` are module constants, the reduction bounds are compile-time
//! integers, and the `rep = nh/nkv` query heads of a kv head are folded into one thread's
//! accumulator in ascending head order. Tiling and reuse are open work.
//!
//! Bindings follow the order of the twin's tensors; the layouts are `q`/`dout` token-major
//! `[(i·nh+h)·hd+d]`, `k`/`v` `[(j·nkv+g)·hd+d]`, `P`/`dS` `[(h·t+i)·t+j]` and `rowsum[h·t+i]`. The
//! four dispatches run in one command buffer with a compute→compute barrier after each, so
//! `rowsum` is visible to `ds` and `dS` is visible to `dq` and `dk`/`dv`.

use super::*;

/// **Attention-backward row sum (f32)** — dispatch 1 of the four, the twin of `attn_bwd_kern`'s
/// `rowsum(dP ⊙ P)`. One workgroup per `(h,i)`: workgroup `h·t + i` of `t·nh` splits its key row
/// over 256 lanes, lane `l` taking `j = l, l+256, …` up to `i` (one key per lane at `t ≤ 256`).
/// Each lane accumulates `P[h,i,j]·Σ_e dO[i,h,e]·V[j,h/rep,e]` over `e` ascending with fused
/// multiply-add, exactly the per-key order the single-thread body used; the lane terms go to a
/// 256-word shared array (`0` for a lane past `i`) and are summed by [`Kb::tree_reduce`] — a fixed
/// association, no float atomics — whose lane 0 writes `rowsum[h·t+i]`. The reduction over `j` is
/// therefore a tree, not the former ascending walk; the `e` sum is unchanged. Grid `t·nh`,
/// LocalSize 256.
///
/// Bindings: 0 = v (f32, `t·nkv·hd`), 1 = p (f32, `nh·t·t`), 2 = dout (f32, `t·nh·hd`),
/// 3 = rowsum (f32, `t·nh`).
///
/// Complexity: workgroup `(h,i)` reads `(i+1)·(2·hd+1)` words of `dout`/`v`/`p` once — `O(t·hd)`
/// words per workgroup, `O(nh·t²·hd)` for the dispatch, unchanged from the single-thread form and
/// now spread across `nh·t` workgroups instead of one; the tree adds `log₂ 256 = 8` shared
/// add/barrier levels per workgroup.
pub fn attn_bwd_rowsum_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(t > 0 && hd > 0, "attn_bwd_rowsum: shape must be non-zero");
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_bwd_rowsum: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let v = k.ssbo_f(0);
    let p = k.ssbo_f(1);
    let dout = k.ssbo_f(2);
    let rowsum = k.ssbo_f(3);
    let (red, redp) = k.shared_f32(WG);
    let gid = k.gid_comp(0);
    let tid = k.local_index();
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let cwg = k.cu(WG);
    let total = k.cu(t * nh);
    let zero = k.cf_bits(0x0000_0000);
    // One workgroup per (h,i): the row is the workgroup id, the lane is the local index.
    let wg = k.udiv(gid, cwg);
    // Every lane owns one shared word, so a lane past the row's key count contributes 0 to the tree.
    k.shared_store(red, redp, tid, zero);
    let inb = k.ult(wg, total);
    k.iff(inb, |mut k| {
        let h = k.udiv(wg, ct);
        let i = k.umod(wg, ct);
        let kvh = k.udiv(h, crep);
        let acc = k.fvar_f();
        k.store_var(acc, zero);
        let nrow = k.iadd(i, c1);
        k.loop_step(tid, nrow, cwg, |k, j| {
            let dp = k.fvar_f();
            k.store_var(dp, zero);
            // dout row of query i, head h: (i·nh+h)·hd.
            let ih = k.imul(i, cnh);
            let rb = k.iadd(ih, h);
            let db = k.imul(rb, chd);
            // v row of key j, kv head kvh: (j·nkv+kvh)·hd.
            let jn = k.imul(j, ckv);
            let jk = k.iadd(jn, kvh);
            let vb = k.imul(jk, chd);
            k.loop_u(c0, chd, |mut k, e| {
                let da = k.iadd(db, e);
                let dov = k.load_f(dout, da);
                let va = k.iadd(vb, e);
                let vv = k.load_f(v, va);
                let a0 = k.load_var_f(dp);
                let a1 = k.fma(dov, vv, a0);
                k.store_var(dp, a1);
                k
            });
            // P[h,i,j] at (h·t+i)·t+j.
            let ht = k.imul(h, ct);
            let hi = k.iadd(ht, i);
            let pb = k.imul(hi, ct);
            let pidx = k.iadd(pb, j);
            let pij = k.load_f(p, pidx);
            let dpv = k.load_var_f(dp);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(pij, dpv, a0);
            k.store_var(acc, a1);
        });
        let s = k.load_var_f(acc);
        k.shared_store(red, redp, tid, s);
        k
    });
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, false);
    let inw = k.ult(wg, total);
    k.iff(inw, |mut k| {
        let lane0 = k.ult(tid, c1);
        k.iff(lane0, |mut k| {
            let s = k.shared_load(red, redp, c0);
            k.store_f(rowsum, wg, s);
            k
        });
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_rowsum_module`] as bytes.
pub fn attn_bwd_rowsum_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_rowsum_module(t, nh, nkv, hd))
}

/// **Attention-backward score gradient (f32)** — dispatch 2, the materialisation of
/// `dS[h,i,j] = P[h,i,j]·(dP_ij − rowsum[h·t+i])`. Thread `(h·t+i)·t+j` (of `nh·t·t`) computes
/// `dP_ij = Σ_e dO[i,h,e]·V[j,h/rep,e]` with `e` ascending and fused multiply-add, rounds
/// `dS` once and writes it; a thread with `j > i` writes `0` and reads nothing. It replaces the
/// two in-thread recomputations in the `dq` and `dk`/`dv` passes. Grid `⌈nh·t·t/256⌉`,
/// LocalSize 256.
///
/// Bindings: 0 = v (f32, `t·nkv·hd`), 1 = p (f32, `nh·t·t`), 2 = dout (f32, `t·nh·hd`),
/// 3 = rowsum (f32, `t·nh`), 4 = ds (f32, `nh·t·t`).
///
/// Complexity: thread `(h,i,j)` with `j ≤ i` reads `2·hd` words of `dout`/`v`, one `p` word and
/// one `rowsum` word and writes one `ds` word; `j > i` writes one word. The dispatch therefore
/// moves `nh·t·t·(2·hd + 2) + nh·t·t` words — `O(hd)` per thread, `O(nh·t²·hd)` total — against
/// the reused-read floor `t·nh·hd + t·nkv·hd + nh·t² + t·nh + nh·t²`.
pub fn attn_bwd_ds_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(t > 0 && hd > 0, "attn_bwd_ds: shape must be non-zero");
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_bwd_ds: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let v = k.ssbo_f(0);
    let p = k.ssbo_f(1);
    let dout = k.ssbo_f(2);
    let rowsum = k.ssbo_f(3);
    let ds_buf = k.ssbo_f(4);
    let gid = k.gid_comp(0);
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let ctt = k.cu(t * t);
    let total = k.cu(nh * t * t);
    let zero = k.cf_bits(0x0000_0000);
    let inb = k.ult(gid, total);
    k.iff(inb, |mut k| {
        // (h, i, j) from the flat id; every thread writes exactly one ds word.
        let h = k.udiv(gid, ctt);
        let rem = k.umod(gid, ctt);
        let i = k.udiv(rem, ct);
        let j = k.umod(rem, ct);
        let kvh = k.udiv(h, crep);
        let ht = k.imul(h, ct);
        let hi = k.iadd(ht, i);
        let pb = k.imul(hi, ct);
        let didx = k.iadd(pb, j);
        let out = k.fvar_f();
        k.store_var(out, zero);
        // dP ascending e over the head width; unchanged from the dq pass's inner loop.
        let cj = k.iadd(i, c1);
        let causal = k.ult(j, cj);
        k.iff(causal, |mut k| {
            let dp = k.fvar_f();
            k.store_var(dp, zero);
            // dout row of query i, head h: (i·nh+h)·hd.
            let ih = k.imul(i, cnh);
            let rb = k.iadd(ih, h);
            let db = k.imul(rb, chd);
            // v row of key j, kv head kvh: (j·nkv+kvh)·hd.
            let jn = k.imul(j, ckv);
            let jk = k.iadd(jn, kvh);
            let vb = k.imul(jk, chd);
            k.loop_u(c0, chd, |mut k, e| {
                let da = k.iadd(db, e);
                let dov = k.load_f(dout, da);
                let va = k.iadd(vb, e);
                let vv = k.load_f(v, va);
                let a0 = k.load_var_f(dp);
                let a1 = k.fma(dov, vv, a0);
                k.store_var(dp, a1);
                k
            });
            let pij = k.load_f(p, didx);
            let rs = k.load_f(rowsum, hi);
            let dpv = k.load_var_f(dp);
            // One rounding for dS, as in the two in-thread passes it replaces.
            let diff = k.fsub(dpv, rs);
            let s = k.fmul(pij, diff);
            k.store_var(out, s);
            k
        });
        let s = k.load_var_f(out);
        k.store_f(ds_buf, didx, s);
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_ds_module`] as bytes.
pub fn attn_bwd_ds_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_ds_module(t, nh, nkv, hd))
}

/// **Attention-backward dQ (f32)** — dispatch 3, the twin of `attn_bwd_kern`'s `dQ = scale·dS·K`.
/// Thread `(i·nh+h)·hd+d` (of `t·nh·hd`) walks `j` in `0..=i` ascending and accumulates
/// `ds[h,i,j]·K[j,h/rep,d]` with fused multiply-add; it writes `scale·acc` with
/// `scale = 1/sqrt(hd)` held as an f32 module constant. The `dP_ij` recomputation is gone: the
/// value is read from dispatch 2, with the same one-rounding `dS`. Grid `⌈t·nh·hd/256⌉`,
/// LocalSize 256.
///
/// Bindings: 0 = k (f32, `t·nkv·hd`), 1 = ds (f32, `nh·t·t`), 2 = dq (f32, `t·nh·hd`).
///
/// Complexity: one thread reads `i+1` `ds` words and `i+1` `k` words and writes one `dq` word,
/// so the dispatch moves `t·nh·hd·(t + 2)` words — `O(t)` per thread — against the pre-change
/// `2·t²·nh·hd² + t·nh·hd` words of the in-thread `dP` form and the reused-read floor
/// `t·nkv·hd + nh·t² + t·nh·hd`.
pub fn attn_bwd_dq_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(t > 0 && hd > 0, "attn_bwd_dq: shape must be non-zero");
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_bwd_dq: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let kb = k.ssbo_f(0);
    let ds_buf = k.ssbo_f(1);
    let dq = k.ssbo_f(2);
    let gid = k.gid_comp(0);
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let total = k.cu(t * nh * hd);
    let zero = k.cf_bits(0x0000_0000);
    let scale = k.cf(1.0 / (hd as f32).sqrt());
    let inb = k.ult(gid, total);
    k.iff(inb, |mut k| {
        let d = k.umod(gid, chd);
        let r = k.udiv(gid, chd);
        let h = k.umod(r, cnh);
        let i = k.udiv(r, cnh);
        let kvh = k.udiv(h, crep);
        let ht = k.imul(h, ct);
        let rsi = k.iadd(ht, i);
        let pb = k.imul(rsi, ct);
        let acc = k.fvar_f();
        k.store_var(acc, zero);
        let nrow = k.iadd(i, c1);
        k.loop_u(c0, nrow, |mut k, j| {
            let didx = k.iadd(pb, j);
            let dsij = k.load_f(ds_buf, didx);
            let jn = k.imul(j, ckv);
            let jk = k.iadd(jn, kvh);
            let kbse = k.imul(jk, chd);
            let ka = k.iadd(kbse, d);
            let kk = k.load_f(kb, ka);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(dsij, kk, a0);
            k.store_var(acc, a1);
            k
        });
        let accv = k.load_var_f(acc);
        let s = k.fmul(scale, accv);
        k.store_f(dq, gid, s);
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_dq_module`] as bytes.
pub fn attn_bwd_dq_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_dq_module(t, nh, nkv, hd))
}

/// **Attention-backward dK/dV (f32)** — dispatch 4, the twin of `attn_bwd_kern`'s `dK = scale·dSᵀ·Q`
/// and `dV = Pᵀ·dO`. Thread `(j·nkv+g)·hd+d` (of `t·nkv·hd`) walks the `rep` query heads
/// `h ∈ [g·rep, (g+1)·rep)` ascending, and inside each head the rows `i ∈ [j, t)` ascending: it
/// reads `ds[h,i,j]`, accumulates `ds·Q[i,h,d]` into `dk` and `P[h,i,j]·dO[i,h,d]` into `dv`; it
/// writes `scale·dk_acc` (the sum over the `rep` heads, one rounding at the end) and `dv_acc`. The
/// `dP_ij` recomputation is gone. Grid `⌈t·nkv·hd/256⌉`, LocalSize 256.
///
/// Bindings: 0 = q (f32, `t·nh·hd`), 1 = p (f32, `nh·t·t`), 2 = ds (f32, `nh·t·t`),
/// 3 = dout (f32, `t·nh·hd`), 4 = dk (f32, `t·nkv·hd`), 5 = dv (f32, `t·nkv·hd`).
///
/// Complexity: one thread walks `rep·(t−j)` `(h,i)` pairs, reading one `ds`, one `q`, one `dout`
/// and one `p` word per pair and writing two words, so the dispatch moves
/// `4·t²·nh·hd + 2·t·nkv·hd` words — `O(rep·t)` per thread — against the pre-change
/// `≈ t²·nh·hd·(2·hd + 4)` words of the in-thread `dP` form and the reused-read floor
/// `t·nh·hd + t·nkv·hd + nh·t² + nh·t² + 2·t·nkv·hd`.
pub fn attn_bwd_dkdv_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(t > 0 && hd > 0, "attn_bwd_dkdv: shape must be non-zero");
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_bwd_dkdv: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let q = k.ssbo_f(0);
    let p = k.ssbo_f(1);
    let ds_buf = k.ssbo_f(2);
    let dout = k.ssbo_f(3);
    let dk = k.ssbo_f(4);
    let dv = k.ssbo_f(5);
    let gid = k.gid_comp(0);
    let c0 = k.cu(0);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let total = k.cu(t * nkv * hd);
    let zero = k.cf_bits(0x0000_0000);
    let scale = k.cf(1.0 / (hd as f32).sqrt());
    let inb = k.ult(gid, total);
    k.iff(inb, |mut k| {
        let d = k.umod(gid, chd);
        let r = k.udiv(gid, chd);
        let g = k.umod(r, ckv);
        let j = k.udiv(r, ckv);
        let dk_acc = k.fvar_f();
        let dv_acc = k.fvar_f();
        k.store_var(dk_acc, zero);
        k.store_var(dv_acc, zero);
        let h0 = k.imul(g, crep);
        let h1 = k.iadd(h0, crep);
        k.loop_u(h0, h1, |mut k, h| {
            let ht = k.imul(h, ct);
            k.loop_u(j, ct, |mut k, i| {
                // q/dout row of query i, head h: (i·nh+h)·hd; ds/P at (h·t+i)·t+j.
                let ih = k.imul(i, cnh);
                let rib = k.iadd(ih, h);
                let qb = k.imul(rib, chd);
                let qa = k.iadd(qb, d);
                let hi = k.iadd(ht, i);
                let pbase = k.imul(hi, ct);
                let pj = k.iadd(pbase, j);
                let dsij = k.load_f(ds_buf, pj);
                let qd = k.load_f(q, qa);
                let a0 = k.load_var_f(dk_acc);
                let a1 = k.fma(dsij, qd, a0);
                k.store_var(dk_acc, a1);
                let pij = k.load_f(p, pj);
                let dod = k.load_f(dout, qa);
                let b0 = k.load_var_f(dv_acc);
                let b1 = k.fma(pij, dod, b0);
                k.store_var(dv_acc, b1);
                k
            });
            k
        });
        let dkv = k.load_var_f(dk_acc);
        let s = k.fmul(scale, dkv);
        k.store_f(dk, gid, s);
        let dvv = k.load_var_f(dv_acc);
        k.store_f(dv, gid, dvv);
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_dkdv_module`] as bytes.
pub fn attn_bwd_dkdv_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_dkdv_module(t, nh, nkv, hd))
}

/// Host wrapper: build and record the backward row sum. `v` is `t·nkv·hd` f32, `p` `nh·t·t`,
/// `dout` `t·nh·hd`, `rowsum` `t·nh`. Grid `(t·nh, 1, 1)` — one workgroup per `(h,i)`, each
/// splitting its key row over 256 lanes and tree-reducing them; LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn attn_bwd_rowsum_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    v: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    dout: crate::gpu::vulkan::Handle,
    rowsum: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_rowsum_bytes(t, nh, nkv, hd),
        &[v, p, dout, rowsum],
        t * nh,
        1,
        "attn_bwd_rowsum",
    )
}

/// Host wrapper: build and record the backward score-gradient pass. `ds` holds `nh·t·t` f32.
/// Grid `(⌈nh·t·t/256⌉, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_bwd_ds_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    v: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    dout: crate::gpu::vulkan::Handle,
    rowsum: crate::gpu::vulkan::Handle,
    ds: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_ds_bytes(t, nh, nkv, hd),
        &[v, p, dout, rowsum, ds],
        (nh * t * t).div_ceil(256),
        1,
        "attn_bwd_ds",
    )
}

/// Host wrapper: build and record the backward dQ pass. `ds` holds `nh·t·t` f32.
/// Grid `(⌈t·nh·hd/256⌉, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn attn_bwd_dq_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    k: crate::gpu::vulkan::Handle,
    ds: crate::gpu::vulkan::Handle,
    dq: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_dq_bytes(t, nh, nkv, hd),
        &[k, ds, dq],
        (t * nh * hd).div_ceil(256),
        1,
        "attn_bwd_dq",
    )
}

/// Host wrapper: build and record the backward dK/dV pass. `ds` holds `nh·t·t` f32.
/// Grid `(⌈t·nkv·hd/256⌉, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_bwd_dkdv_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    q: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    ds: crate::gpu::vulkan::Handle,
    dout: crate::gpu::vulkan::Handle,
    dk: crate::gpu::vulkan::Handle,
    dv: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_dkdv_bytes(t, nh, nkv, hd),
        &[q, p, ds, dout, dk, dv],
        (t * nkv * hd).div_ceil(256),
        1,
        "attn_bwd_dkdv",
    )
}
