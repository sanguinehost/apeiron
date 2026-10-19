//! simt_nn::fwd::norm — the row-batched RMSNorm forward kernel and its host dispatch: the f32
//! SPIR-V twin of `learn::grad::dense::rmsnorm_fwd`, one workgroup per row with a row base, so a
//! `t`-row layer runs one dispatch over one `t·w` buffer instead of `t` buffers. Its inverse RMS
//! is the same expression `norm_bwd.rs` computes, so the layer step's backward pass reuses this
//! normalisation.

use super::super::*;

/// **RMSNorm forward (f32, row-batched)** — the GPU twin of `learn::grad::dense::rmsnorm_fwd`,
/// `out[r,j] = x[r,j]·r_r·γ_j` with `r_r = (mean(x_r²) + eps)^(-1/2)`. One workgroup per row,
/// LocalSize 256; workgroup `g` owns row `g` through the row base `g·w`, so the grid is `(t,1,1)`
/// over one `t×w` x and one `t×w` out.
///
/// The inverse RMS is bit-identical to [`crate::gpu::nn::rmsnorm_bwd_module`]'s: the same lane-strided
/// `Fma(x,x,acc)` sum of squares in ascending strided `i`, the same shared 256-lane tree reduce,
/// the same round-to-whole-Q32.32-unit `mean`, and the same `InverseSqrt(mean + eps)` expression
/// (the reference's `inv_rms` rounds the mean before adding `eps`). The backward dispatch reuses
/// this `r` unchanged.
///
/// Domain: `w ≥ 1` (asserted), `t ≥ 1` (the grid is `t`). Bindings: 0 = x (f32, `t·w`),
/// 1 = γ (f32, `w`), 2 = out (f32, `t·w`).
///
/// Complexity: each lane caches its `⌈w/256⌉` x words in registers, then issues one `Fma` per
/// cached word on the sumsq pass and one `Fmul` pair plus one store per word on the scale pass —
/// `O(⌈w/256⌉)` global loads, arithmetic ops and stores per lane, counted in events. The tree
/// reduce adds `log₂ 256 = 8` shared levels with one barrier each.
///
/// Traffic: `t·w` (x) + `w` (γ) words read, `t·w` (out) written — `2·t·w + w` words, since the x
/// load for the sum of squares is the one kept for the scale pass.
pub fn rmsnorm_fwd_module(w: u32, eps: f32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(w > 0, "rmsnorm_fwd: width must be non-zero");
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let gain = k.ssbo_f(1);
    let out = k.ssbo_f(2);
    let (red, redp) = k.shared_f32(WG);
    let tid = k.local_index();
    let gid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let cw = k.cu(w);
    let wgi = k.udiv(gid, cwg);
    let base = k.imul(wgi, cw);
    // ── this lane's x words, cached for the scale pass (one global read per element) ──
    let u = w.div_ceil(WG);
    let mut xv: Vec<u32> = Vec::new();
    for t in 0..u {
        xv.push(k.fvar_f());
        let ti = k.strided_idx(tid, t, WG);
        let inb = k.ult(ti, cw);
        let var = xv[t as usize];
        k.iff(inb, |mut k| {
            let idx = k.iadd(base, ti);
            let v = k.load_f(x, idx);
            k.store_var(var, v);
            k
        });
    }
    // ── Σ x² in the backward kernel's order: ascending strided `i`, explicit `Fma` ──
    let acc = k.fvar_f();
    let zero = k.cf_bits(0x0000_0000);
    k.store_var(acc, zero);
    for t in 0..u {
        let ti = k.strided_idx(tid, t, WG);
        let inb = k.ult(ti, cw);
        let var = xv[t as usize];
        k.iff(inb, |mut k| {
            let v = k.load_var_f(var);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(v, v, a0);
            k.store_var(acc, a1);
            k
        });
    }
    let af = k.load_var_f(acc);
    k.shared_store(red, redp, tid, af);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, false);
    let c0 = k.cu(0);
    let tot = k.shared_load(red, redp, c0);
    let invn = k.cf(1.0 / w as f32);
    let mean = k.fmul(tot, invn);
    // The reference rounds `mean` to a whole Q32.32 unit before adding eps (`inv_rms`:
    // `round(Σx² / (n·2³²))`); `norm_bwd.rs` states why the f32 mean's one-unit drift is visible at
    // `eps = 1e-6`. The same rounding here keeps `r` bit-identical to the backward kernel's.
    let c2p32 = k.cf(4294967296.0);
    let units = k.fmul(mean, c2p32);
    let rounded = k.ext1(GLSL_ROUND, units);
    let c2m32 = k.cf(1.0 / 4294967296.0);
    let mean_q = k.fmul(rounded, c2m32);
    let epsv = k.cf(eps);
    let me = k.fadd(mean_q, epsv);
    let r = k.ext1(GLSL_INVERSE_SQRT, me);
    // ── out_i = x_i·r·γ_i for each cached strided i ──
    for t in 0..u {
        let ti = k.strided_idx(tid, t, WG);
        let inb = k.ult(ti, cw);
        let var = xv[t as usize];
        k.iff(inb, |mut k| {
            let xj = k.load_var_f(var);
            let gj = k.load_f(gain, ti);
            let t0 = k.fmul(xj, r);
            let ov = k.fmul(t0, gj);
            let idx = k.iadd(base, ti);
            k.store_f(out, idx, ov);
            k
        });
    }
    k.finish(WG)
}

/// [`rmsnorm_fwd_module`] as bytes.
pub fn rmsnorm_fwd_bytes(w: u32, eps: f32) -> Vec<u8> {
    bytes(rmsnorm_fwd_module(w, eps))
}

/// Host wrapper: build and record the row-batched RMSNorm-forward dispatch. `x` and `out` are
/// device buffers of `t·w` f32 and `gain` of `w` f32. Grid `(t, 1, 1)`, one workgroup per row;
/// `eps` is the training contract's `1e-6`. [`super::rmsnorm_bwd_dispatch`] then reuses the same
/// inverse RMS on the same `x`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; `x` and `out` must hold `t·w`
/// f32 and `gain` `w` f32.
#[cfg(feature = "gpu")]
pub unsafe fn rmsnorm_fwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    x: crate::gpu::vulkan::Handle,
    gain: crate::gpu::vulkan::Handle,
    out: crate::gpu::vulkan::Handle,
    t: u32,
    w: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &rmsnorm_fwd_bytes(w, 1e-6f32),
        &[x, gain, out],
        t,
        1,
        "rmsnorm_fwd",
    )
}
