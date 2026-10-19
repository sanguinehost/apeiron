//! simt_nn::attn::bwd — the causal GQA attention backward as f32 SPIR-V, the twin of
//! `learn::grad::attn::attn_bwd_kern`.
//!
//! Three deterministic dispatches, no float atomics, every output thread sums in a fixed ascending
//! order:
//!
//! 1. `rowsum`: `r[h,i] = Σ_{j≤i} P[h,i,j]·Σ_e dO[i,h,e]·V[j,h/rep,e]`, one thread per `(h,i)`.
//! 2. `dq`: `dq[i,h,d] = scale·Σ_{j≤i} P[h,i,j]·(dP_ij − r[h,i])·K[j,h/rep,d]`, one thread per
//!    `(i,h,d)`.
//! 3. `dk`/`dv`: `dk[j,g,d] = scale·Σ_{h∈g} Σ_{i≥j} dS_ij·Q[i,h,d]`,
//!    `dv[j,g,d] = Σ_{h∈g} Σ_{i≥j} P[h,i,j]·dO[i,h,d]`, one thread per `(j,g,d)`.
//!
//! The shapes `t`, `nh`, `nkv`, `hd` are module constants, the reduction bounds are compile-time
//! integers, and the `rep = nh/nkv` query heads of a kv head are folded into one thread's
//! accumulator in ascending head order. Tiling and reuse are open work.
//!
//! Bindings follow the order of the twin's tensors; the layouts are `q`/`dout` token-major
//! `[(i·nh+h)·hd+d]`, `k`/`v` `[(j·nkv+g)·hd+d]` and `P[(h·t+i)·t+j]`. The three dispatches run in
//! one command buffer with a compute→compute barrier after each, so `rowsum` is visible to both
//! dependent passes.

use super::*;

/// **Attention-backward row sum (f32)** — dispatch 1 of the three, the twin of `attn_bwd_kern`'s
/// `rowsum(dP ⊙ P)`. Thread `h·t + i` (of `t·nh`) walks `j` in `0..=i` ascending and accumulates
/// `P[h,i,j] · Σ_e dO[i,h,e]·V[j,h/rep,e]` with fused multiply-add; it writes `rowsum[h·t+i]`.
/// Grid `⌈t·nh/256⌉`, LocalSize 256.
///
/// Bindings: 0 = v (f32, `t·nkv·hd`), 1 = p (f32, `nh·t·t`), 2 = dout (f32, `t·nh·hd`),
/// 3 = rowsum (f32, `t·nh`).
///
/// Complexity: thread `(h,i)` reads `2·(i+1)·hd` words of `dout`/`v` and `i+1` of `p`, so the
/// dispatch reads `Σ_i (2·(i+1)·hd + i+1)·nh = nh·(t·(t+1)·hd + t·(t+1)/2)` words — `O(t²·hd)`
/// per thread, `O(nh·t²·hd)` total — against the reused-read floor `t·nh·hd + nh·t² + t·nh`.
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
    let gid = k.gid_comp(0);
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let total = k.cu(t * nh);
    let zero = k.cf_bits(0x0000_0000);
    let inb = k.ult(gid, total);
    k.iff(inb, |mut k| {
        let h = k.udiv(gid, ct);
        let i = k.umod(gid, ct);
        let kvh = k.udiv(h, crep);
        let acc = k.fvar_f();
        k.store_var(acc, zero);
        let nrow = k.iadd(i, c1);
        k.loop_u(c0, nrow, |mut k, j| {
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
            k
        });
        let s = k.load_var_f(acc);
        k.store_f(rowsum, gid, s);
        k
    });
    k.finish(WG)
}

/// [`attn_bwd_rowsum_module`] as bytes.
pub fn attn_bwd_rowsum_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_bwd_rowsum_module(t, nh, nkv, hd))
}

/// **Attention-backward dQ (f32)** — dispatch 2, the twin of `attn_bwd_kern`'s `dQ = scale·dS·K`.
/// Thread `(i·nh+h)·hd+d` (of `t·nh·hd`) walks `j` in `0..=i` ascending, recomputes
/// `dP_ij = Σ_e dO[i,h,e]·V[j,h/rep,e]`, forms `dS_ij = P[h,i,j]·(dP_ij − rowsum[h·t+i])` and
/// accumulates `dS_ij·K[j,h/rep,d]`; it writes `scale·acc` with `scale = 1/sqrt(hd)` held as an
/// f32 module constant. Grid `⌈t·nh·hd/256⌉`, LocalSize 256.
///
/// Bindings: 0 = k (f32, `t·nkv·hd`), 1 = v (f32, `t·nkv·hd`), 2 = p (f32, `nh·t·t`),
/// 3 = dout (f32, `t·nh·hd`), 4 = rowsum (f32, `t·nh`), 5 = dq (f32, `t·nh·hd`).
///
/// Complexity: one thread reads `(2·hd + 2)·(i+1)` words, so the dispatch reads
/// `nh·hd·Σ_i (2·hd+2)·(i+1) = O(nh·t²·hd²)` words — `O(t·hd)` per thread — against the
/// reused-read floor `2·t·nkv·hd + 2·t·nh·hd + nh·t² + t·nh`.
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
    let v = k.ssbo_f(1);
    let p = k.ssbo_f(2);
    let dout = k.ssbo_f(3);
    let rowsum = k.ssbo_f(4);
    let dq = k.ssbo_f(5);
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
        let rs = k.load_f(rowsum, rsi);
        let acc = k.fvar_f();
        k.store_var(acc, zero);
        let nrow = k.iadd(i, c1);
        k.loop_u(c0, nrow, |mut k, j| {
            let dp = k.fvar_f();
            k.store_var(dp, zero);
            let ih = k.imul(i, cnh);
            let rb = k.iadd(ih, h);
            let db = k.imul(rb, chd);
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
            let pb = k.imul(rsi, ct);
            let pidx = k.iadd(pb, j);
            let pij = k.load_f(p, pidx);
            let dpv = k.load_var_f(dp);
            let diff = k.fsub(dpv, rs);
            let ds = k.fmul(pij, diff);
            let ka = k.iadd(vb, d);
            let kk = k.load_f(kb, ka);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(ds, kk, a0);
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

/// **Attention-backward dK/dV (f32)** — dispatch 3, the twin of `attn_bwd_kern`'s `dK = scale·dSᵀ·Q`
/// and `dV = Pᵀ·dO`. Thread `(j·nkv+g)·hd+d` (of `t·nkv·hd`) walks the `rep` query heads
/// `h ∈ [g·rep, (g+1)·rep)` ascending, and inside each head the rows `i ∈ [j, t)` ascending:
/// it recomputes `dP_ij`, forms `dS_ij = P[h,i,j]·(dP_ij − rowsum[h·t+i])`, accumulates
/// `dS_ij·Q[i,h,d]` into `dk` and `P[h,i,j]·dO[i,h,d]` into `dv`; it writes `scale·dk_acc` (the
/// sum over the `rep` heads, one rounding at the end) and `dv_acc`. Grid `⌈t·nkv·hd/256⌉`,
/// LocalSize 256.
///
/// Bindings: 0 = q (f32, `t·nh·hd`), 1 = v (f32, `t·nkv·hd`), 2 = p (f32, `nh·t·t`),
/// 3 = dout (f32, `t·nh·hd`), 4 = rowsum (f32, `t·nh`), 5 = dk (f32, `t·nkv·hd`),
/// 6 = dv (f32, `t·nkv·hd`).
///
/// Complexity: one thread reads `rep·(t−j)·(2·hd + 2)` words plus `2·rep·(t−j)` for `q`/`dout`,
/// so the dispatch reads `O(nkv·t²·hd²)` words — `O(rep·t·hd)` per thread — against the reused-read
/// floor `t·nh·hd + t·nkv·hd + nh·t² + t·nh·hd + t·nh + 2·t·nkv·hd`.
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
    let v = k.ssbo_f(1);
    let p = k.ssbo_f(2);
    let dout = k.ssbo_f(3);
    let rowsum = k.ssbo_f(4);
    let dk = k.ssbo_f(5);
    let dv = k.ssbo_f(6);
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
        // v row of key j, kv head g: (j·nkv+g)·hd.
        let jn = k.imul(j, ckv);
        let jg = k.iadd(jn, g);
        let db = k.imul(jg, chd);
        let h0 = k.imul(g, crep);
        let h1 = k.iadd(h0, crep);
        k.loop_u(h0, h1, |mut k, h| {
            let ht = k.imul(h, ct);
            k.loop_u(j, ct, |mut k, i| {
                // q/dout row of query i, head h: (i·nh+h)·hd.
                let ih = k.imul(i, cnh);
                let rib = k.iadd(ih, h);
                let qb = k.imul(rib, chd);
                let dp = k.fvar_f();
                k.store_var(dp, zero);
                k.loop_u(c0, chd, |mut k, e| {
                    let qa = k.iadd(qb, e);
                    let dov = k.load_f(dout, qa);
                    let va = k.iadd(db, e);
                    let vv = k.load_f(v, va);
                    let a0 = k.load_var_f(dp);
                    let a1 = k.fma(dov, vv, a0);
                    k.store_var(dp, a1);
                    k
                });
                let hi = k.iadd(ht, i);
                let pbase = k.imul(hi, ct);
                let pj = k.iadd(pbase, j);
                let pij = k.load_f(p, pj);
                let rs = k.load_f(rowsum, hi);
                let dpv = k.load_var_f(dp);
                let diff = k.fsub(dpv, rs);
                let ds = k.fmul(pij, diff);
                let dka = k.iadd(qb, d);
                let qd = k.load_f(q, dka);
                let a0 = k.load_var_f(dk_acc);
                let a1 = k.fma(ds, qd, a0);
                k.store_var(dk_acc, a1);
                let dod = k.load_f(dout, dka);
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
/// `dout` `t·nh·hd`, `rowsum` `t·nh`. Grid `(⌈t·nh/256⌉, 1, 1)`, LocalSize 256.
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
        (t * nh).div_ceil(256),
        1,
        "attn_bwd_rowsum",
    )
}

/// Host wrapper: build and record the backward dQ pass. Grid `(⌈t·nh·hd/256⌉, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_bwd_dq_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    k: crate::gpu::vulkan::Handle,
    v: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    dout: crate::gpu::vulkan::Handle,
    rowsum: crate::gpu::vulkan::Handle,
    dq: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_bwd_dq_bytes(t, nh, nkv, hd),
        &[k, v, p, dout, rowsum, dq],
        (t * nh * hd).div_ceil(256),
        1,
        "attn_bwd_dq",
    )
}

/// Host wrapper: build and record the backward dK/dV pass. Grid `(⌈t·nkv·hd/256⌉, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_bwd_dkdv_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    q: crate::gpu::vulkan::Handle,
    v: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    dout: crate::gpu::vulkan::Handle,
    rowsum: crate::gpu::vulkan::Handle,
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
        &[q, v, p, dout, rowsum, dk, dv],
        (t * nkv * hd).div_ceil(256),
        1,
        "attn_bwd_dkdv",
    )
}
