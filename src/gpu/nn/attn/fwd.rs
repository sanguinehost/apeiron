//! simt_nn::attn::fwd — the causal GQA attention forward as one f32 SPIR-V dispatch, the twin of
//! `learn::grad::attn::attn_fwd_kern` (whose row body is `attn_row_fwd`).
//!
//! One workgroup owns one query row `(i, h)`, grid `nh·t`, LocalSize 256. Three phases, all inside
//! that workgroup, so the `t` scores of the row never reach global memory:
//!
//! 1. **scores.** Lane `p` computes `s_j = scale·Σ_e q[i,h,e]·k[j,gh,e]` for the keys `j` it owns
//!    (`j = p, p+256, … < i+1`), summing `e` ascending with fused multiply-add, and stores `s_j` in
//!    the shared `t`-word score array. `scale = 1/√hd` is a baked f32 module constant.
//! 2. **softmax.** A whole-workgroup max over the causal row (`Kb::tree_reduce`, `is_max = true`),
//!    then `exp(s_j − max)` per owned key, written back to the shared array and summed by the same
//!    tree reduce. The max-subtracted exponential and the `1/sum` normalisation are the oracle's
//!    (`attn_row_fwd`: `exp_gap` then `prob`).
//! 3. **output and probabilities.** Lane `d < hd` writes
//!    `out[i,h,d] = (1/sum)·Σ_j exp(s_j − max)·v[j,gh,d]` summing `j` ascending; lane `j < t` writes
//!    `P[h,i,j] = exp(s_j − max)/sum` for `j ≤ i` and `0` for `j > i`.
//!
//! Bindings: 0 = q (f32, `t·nh·hd`), 1 = k (f32, `t·nkv·hd`), 2 = v (f32, `t·nkv·hd`),
//! 3 = p (f32, `nh·t·t`), 4 = out (f32, `t·nh·hd`). Layouts are token-major as in the backward:
//! `q`/`out` at `[(i·nh+h)·hd+d]`, `k`/`v` at `[(j·nkv+g)·hd+d]`, `P` at `[(h·t+i)·t+j]`.
//!
//! The shared footprint is `t + 2·256` f32 (the scores plus the two reduction scratch arrays), so
//! the whole row fits one workgroup for `t ≤ 64`; the builder asserts that bound and `hd ≤ 128`.
//! One workgroup per row bounds the grid to `nh·t` workgroups and keeps the row's `t` scores and
//! `t·hd` key/value reads off global memory, against the materialised three-pass alternative that
//! writes `nh·t·t` scores, re-reads them for the max, for the partition function and for every
//! `(i,h,d)` output row.
//!
//! The f32 association is fixed: scores sum `e` ascending with `Fma`, the exponentials sum through
//! the workgroup tree, and the value mix sums `j` ascending with `Fma`. The oracle accumulates
//! Q32.32 products in `i128` and rounds once per score and once per output, so the device result is
//! bound by the pinned parity tolerance (`tests/gpu_attn_fwd.rs`), not bit-identical.

use super::*;

/// **Causal GQA attention forward (f32)** — one workgroup per `(i, h)` row, 256 lanes, the `t`
/// scores of the row in shared memory. See the module docs for the three phases and the layouts.
///
/// Bindings: 0 = q, 1 = k, 2 = v, 3 = p (`nh·t·t`), 4 = out. Grid `(nh·t, 1, 1)`.
///
/// # Panics
/// On a zero shape, `nkv ∤ nh`, `t > 64` or `hd > 128`.
///
/// Complexity: the workgroup reads `hd` words of `q`, `i·hd` of `k` and `i·hd` of `v` for the row,
/// writes `hd` of `out` and `t` of `p`; when it computes, the dispatch moves
/// `nh·t²·hd/2 + 2·t·nh·hd + nh·t²/2 + t·nh·hd` words — `O(t·hd)` per lane in the score pass and
/// `O(t)` per lane in the output pass — against the materialised alternative's three score passes.
pub fn attn_fwd_module(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(t > 0 && hd > 0, "attn_fwd: shape must be non-zero");
    assert!(t <= 64, "attn_fwd: t above 64 does not fit the shared row");
    assert!(
        hd <= 128,
        "attn_fwd: hd above 128 is outside the row contract"
    );
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "attn_fwd: nh must be a positive multiple of nkv"
    );
    let rep = nh / nkv;
    let mut k = Kb::new();
    let q = k.ssbo_f(0);
    let kb = k.ssbo_f(1);
    let vb = k.ssbo_f(2);
    let p = k.ssbo_f(3);
    let out = k.ssbo_f(4);
    let (scores, scorep) = k.shared_f32(t);
    let (red, redp) = k.shared_f32(WG);
    let (red2, red2p) = k.shared_f32(WG);
    let gid = k.gid_comp(0);
    let tid = k.local_index();
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let cwg = k.cu(WG);
    let ct = k.cu(t);
    let cnh = k.cu(nh);
    let ckv = k.cu(nkv);
    let chd = k.cu(hd);
    let crep = k.cu(rep);
    let zero = k.cf_bits(0x0000_0000);
    let neg_inf = k.cf_bits(0xff80_0000);
    let scale = k.cf(1.0 / (hd as f32).sqrt());
    // Workgroup → row: the grid is a linear walk over (h, i).
    let rid = k.udiv(gid, cwg);
    let h = k.udiv(rid, ct);
    let i = k.umod(rid, ct);
    let n = k.iadd(i, c1);
    let kvh = k.udiv(h, crep);
    // The query/output row of (i, h): (i·nh+h)·hd.
    let ih = k.imul(i, cnh);
    let qr = k.iadd(ih, h);
    let qbase = k.imul(qr, chd);
    // ── phase 1: s_j = scale·Σ_e q_e·k_e, e ascending, for the keys this lane owns ──
    k.loop_step(tid, n, cwg, |k, j| {
        let jn = k.imul(j, ckv);
        let jk = k.iadd(jn, kvh);
        let kbase = k.imul(jk, chd);
        let acc = k.fvar_f();
        k.store_var(acc, zero);
        k.loop_u(c0, chd, |mut k, e| {
            let qa = k.iadd(qbase, e);
            let qv = k.load_f(q, qa);
            let ka = k.iadd(kbase, e);
            let kv = k.load_f(kb, ka);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(qv, kv, a0);
            k.store_var(acc, a1);
            k
        });
        let d = k.load_var_f(acc);
        let s = k.fmul(scale, d);
        k.shared_store(scores, scorep, j, s);
    });
    k.wg_barrier();
    // ── phase 2a: whole-workgroup max over the causal row ──
    let mx = k.fvar_f();
    k.store_var(mx, neg_inf);
    let own = k.ult(tid, n);
    k.iff(own, |mut k| {
        let s = k.shared_load(scores, scorep, tid);
        k.store_var(mx, s);
        k
    });
    let mv = k.load_var_f(mx);
    k.shared_store(red, redp, tid, mv);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, true);
    let m = k.shared_load(red, redp, c0);
    // ── phase 2b: exp(s_j − m) in place, summed by the same tree ──
    let acc = k.fvar_f();
    k.store_var(acc, zero);
    k.loop_step(tid, n, cwg, |k, j| {
        let s = k.shared_load(scores, scorep, j);
        let gap = k.fsub(s, m);
        let e = k.ext1(GLSL_EXP, gap);
        k.shared_store(scores, scorep, j, e);
        let a0 = k.load_var_f(acc);
        let a1 = k.fadd(a0, e);
        k.store_var(acc, a1);
    });
    let av = k.load_var_f(acc);
    k.shared_store(red2, red2p, tid, av);
    k.wg_barrier();
    k.tree_reduce(red2, red2p, WG, tid, false);
    let z = k.shared_load(red2, red2p, c0);
    let one = k.cf_bits(0x3f80_0000);
    let inv = k.fdiv(one, z);
    // ── phase 3: out = (1/z)·Σ_j exp_j·v_j (j ascending), lane per head dimension ──
    let lane = k.ult(tid, chd);
    k.iff(lane, |mut k| {
        let acc2 = k.fvar_f();
        k.store_var(acc2, zero);
        k.loop_u(c0, n, |mut k, j| {
            let e = k.shared_load(scores, scorep, j);
            let jn = k.imul(j, ckv);
            let jk = k.iadd(jn, kvh);
            let kbase = k.imul(jk, chd);
            let va = k.iadd(kbase, tid);
            let vv = k.load_f(vb, va);
            let b0 = k.load_var_f(acc2);
            let b1 = k.fma(e, vv, b0);
            k.store_var(acc2, b1);
            k
        });
        let dot = k.load_var_f(acc2);
        let o = k.fmul(inv, dot);
        let oa = k.iadd(qbase, tid);
        k.store_f(out, oa, o);
        k
    });
    // ── P row: exp_j/z for j ≤ i, 0 above the diagonal ──
    let col = k.ult(tid, ct);
    k.iff(col, |mut k| {
        let e = k.shared_load(scores, scorep, tid);
        let ratio = k.fmul(e, inv);
        let f = k.float;
        let causal = k.ult(tid, n);
        let val = k.e(OP_SELECT, f, &[causal, ratio, zero]);
        let ht = k.imul(h, ct);
        let hi = k.iadd(ht, i);
        let pb = k.imul(hi, ct);
        let pa = k.iadd(pb, tid);
        k.store_f(p, pa, val);
        k
    });
    k.finish(WG)
}

/// [`attn_fwd_module`] as bytes.
pub fn attn_fwd_bytes(t: u32, nh: u32, nkv: u32, hd: u32) -> Vec<u8> {
    bytes(attn_fwd_module(t, nh, nkv, hd))
}

/// Host wrapper: build and record the causal attention forward. `q` is `t·nh·hd` f32, `k` and `v`
/// `t·nkv·hd`, `p` `nh·t·t`, `out` `t·nh·hd`. Grid `(nh·t, 1, 1)`, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_fwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    q: crate::gpu::vulkan::Handle,
    k: crate::gpu::vulkan::Handle,
    v: crate::gpu::vulkan::Handle,
    p: crate::gpu::vulkan::Handle,
    out: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &attn_fwd_bytes(t, nh, nkv, hd),
        &[q, k, v, p, out],
        nh * t,
        1,
        "attn_fwd",
    )
}
