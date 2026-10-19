//! simt_nn::fwd::head — the KL loss head as one f32 SPIR-V dispatch, the twin of
//! `learn::grad::head::kl_row`.
//!
//! One workgroup owns one row, grid `rows`, LocalSize 256. Lanes stride the vocabulary three times:
//!
//! 1. **maxima.** Lane `p` folds `max` over `z_s[p], z_s[p+256], …` and the same over `z_t`, and
//!    the two per-lane maxima go through `Kb::tree_reduce` (`is_max = true`), giving `max_s` and
//!    `max_t` (read back from `red[0]` after the final barrier).
//! 2. **partition functions.** Lane `p` sums `exp(z_s[i] − max_s)` over its indices, then the same
//!    for `z_t`, each through the workgroup tree; `ln(sum)` is `GLSL.std.450 Log`.
//! 3. **loss and gradient.** With `lp_s = (z_s − max_s) − ln Σ_s`, `lp_t = (z_t − max_t) − ln Σ_t`,
//!    `p_s = exp(z_s − max_s)/Σ_s` and `p_t = exp(z_t − max_t)/Σ_t`: lane `p` accumulates
//!    `p_t·(lp_t − lp_s)` over its indices and writes `dz[i] = p_s − p_t`; the loss sum goes through
//!    the workgroup tree and lane 0 writes `loss[row]`.
//!
//! Bindings: 0 = z_s (f32, `rows·v`), 1 = z_t (f32, `rows·v`), 2 = loss (f32, `rows`), 3 = dz
//! (f32, `rows·v`). The reference rounds the loss once from the exact `i128` sum; the device sums
//! in f32 through the workgroup tree, so the parity test bounds the difference, it is not
//! bit-identical (`tests/gpu_head.rs`).

use super::super::*;

/// **KL loss head (f32)** — one workgroup per row, 256 lanes striding the vocabulary. See the
/// module docs for the three phases and the bindings.
///
/// Bindings: 0 = z_s, 1 = z_t, 2 = loss (`rows`), 3 = dz (`rows·v`). Grid `(rows, 1, 1)`.
///
/// Complexity (device): per row three strided passes over `v` words of each input — the maxima
/// pass (`2·v` reads), the partition pass (`2·v` reads) and the loss/dz pass (`2·v` reads, `v`
/// writes) — plus four `O(log₂ 256)` shared tree reductions and one `v`-word write of `loss`:
/// `6·v + 1` words moved per row, `O(v)` per lane `⌈v/256⌉` loads of each input and `O(1)` lanes.
/// The builder emits `O(1)` instructions per module regardless of `v` (all three walks are loops).
pub fn kl_row_module(rows: u32, v: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(rows > 0, "kl_row: rows must be non-zero");
    assert!(v > 0 && v <= 4096, "kl_row: vocabulary must be in 1..=4096");
    let mut k = Kb::new();
    let zs = k.ssbo_f(0);
    let zt = k.ssbo_f(1);
    let loss = k.ssbo_f(2);
    let dz = k.ssbo_f(3);
    let (red, redp) = k.shared_f32(WG);
    let gid = k.gid_comp(0);
    let tid = k.local_index();
    let cwg = k.cu(WG);
    let cv = k.cu(v);
    let c0 = k.cu(0);
    let c1 = k.cu(1);
    let zero = k.cf_bits(0x0000_0000);
    let ninf = k.cf_bits(0xff80_0000);
    let row = k.udiv(gid, cwg);
    let base = k.imul(row, cv);
    // ── phase 1: per-lane maxima of both rows, then two whole-workgroup maxima ──
    let ms = k.fvar_f();
    let mt = k.fvar_f();
    k.store_var(ms, ninf);
    k.store_var(mt, ninf);
    k.loop_step(tid, cv, cwg, |k, i| {
        let idx = k.iadd(base, i);
        let a = k.load_f(zs, idx);
        let b = k.load_f(zt, idx);
        let s0 = k.load_var_f(ms);
        let s1 = k.ext2(GLSL_FMAX, s0, a);
        k.store_var(ms, s1);
        let t0 = k.load_var_f(mt);
        let t1 = k.ext2(GLSL_FMAX, t0, b);
        k.store_var(mt, t1);
    });
    let sm = k.load_var_f(ms);
    k.shared_store(red, redp, tid, sm);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, true);
    let max_s = k.shared_load(red, redp, c0);
    k.wg_barrier();
    let tm = k.load_var_f(mt);
    k.shared_store(red, redp, tid, tm);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, true);
    let max_t = k.shared_load(red, redp, c0);
    k.wg_barrier();
    // ── phase 2: Σ exp(z − max) for each row, two workgroup sums ──
    let ss = k.fvar_f();
    let st = k.fvar_f();
    k.store_var(ss, zero);
    k.store_var(st, zero);
    k.loop_step(tid, cv, cwg, |k, i| {
        let idx = k.iadd(base, i);
        let a = k.load_f(zs, idx);
        let gap = k.fsub(a, max_s);
        let e = k.ext1(GLSL_EXP, gap);
        let a0 = k.load_var_f(ss);
        let a1 = k.fadd(a0, e);
        k.store_var(ss, a1);
        let b = k.load_f(zt, idx);
        let gapt = k.fsub(b, max_t);
        let et = k.ext1(GLSL_EXP, gapt);
        let b0 = k.load_var_f(st);
        let b1 = k.fadd(b0, et);
        k.store_var(st, b1);
    });
    let sv = k.load_var_f(ss);
    k.shared_store(red, redp, tid, sv);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, false);
    let sum_s = k.shared_load(red, redp, c0);
    k.wg_barrier();
    let tv = k.load_var_f(st);
    k.shared_store(red, redp, tid, tv);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, false);
    let sum_t = k.shared_load(red, redp, c0);
    k.wg_barrier();
    let ln_s = k.ext1(GLSL_LOG, sum_s);
    let ln_t = k.ext1(GLSL_LOG, sum_t);
    // ── phase 3: loss = Σ p_t·(lp_t − lp_s) and dz = p_s − p_t ──
    let acc = k.fvar_f();
    k.store_var(acc, zero);
    k.loop_step(tid, cv, cwg, |k, i| {
        let idx = k.iadd(base, i);
        let a = k.load_f(zs, idx);
        let b = k.load_f(zt, idx);
        let gap_s = k.fsub(a, max_s);
        let gap_t = k.fsub(b, max_t);
        let e_s = k.ext1(GLSL_EXP, gap_s);
        let e_t = k.ext1(GLSL_EXP, gap_t);
        let p_s = k.fdiv(e_s, sum_s);
        let p_t = k.fdiv(e_t, sum_t);
        let lp_s = k.fsub(gap_s, ln_s);
        let lp_t = k.fsub(gap_t, ln_t);
        let d = k.fsub(lp_t, lp_s);
        let term = k.fmul(p_t, d);
        let a0 = k.load_var_f(acc);
        let a1 = k.fadd(a0, term);
        k.store_var(acc, a1);
        let g = k.fsub(p_s, p_t);
        k.store_f(dz, idx, g);
    });
    let av = k.load_var_f(acc);
    k.shared_store(red, redp, tid, av);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, false);
    let lt = k.ult(tid, c1);
    k.iff(lt, |mut k| {
        let s = k.shared_load(red, redp, c0);
        k.store_f(loss, row, s);
        k
    });
    k.finish(WG)
}

/// [`kl_row_module`] as bytes.
pub fn kl_row_bytes(rows: u32, v: u32) -> Vec<u8> {
    bytes(kl_row_module(rows, v))
}

/// Host wrapper: build and record the KL loss head. `zs` and `zt` are the device buffers of
/// `rows·v` f32 logits, `loss` of `rows` f32 and `dz` of `rows·v` f32. Grid `(rows, 1, 1)`, one
/// workgroup per row, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn kl_row_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    zs: crate::gpu::vulkan::Handle,
    zt: crate::gpu::vulkan::Handle,
    loss: crate::gpu::vulkan::Handle,
    dz: crate::gpu::vulkan::Handle,
    rows: u32,
    v: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &kl_row_bytes(rows, v),
        &[zs, zt, loss, dz],
        rows,
        1,
        "kl_row",
    )
}
