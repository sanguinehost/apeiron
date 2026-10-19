//! simt_nn::norm_bwd — the RMSNorm backward kernels and their host dispatches: the f32 SPIR-V
//! twin of `learn::grad::dense::rmsnorm_bwd`, the per-row dγ output and its device-side column
//! reduction. Split from `norm.rs` to keep both files under the 1500-line tree cap.

use super::*;

/// **RMSNorm backward (f32)** — the GPU twin of `learn::grad::dense::rmsnorm_bwd`, one workgroup
/// per row. For `r = (mean(x²) + eps)^(-1/2)`, `a_j = γ_j·dy_j`, `S = Σ_j a_j·x_j`:
/// `dx_i = r·a_i − x_i·r³·S/w` and `dγ_i = dy_i·x_i·r` per row. Two workgroup tree reductions
/// (Σx², then S) at LocalSize 256; the element pass is one strided loop.
///
/// Bindings: 0 = x (f32, t×w), 1 = γ (f32, w), 2 = dy (f32, t×w), 3 = dx (f32, t×w),
/// 4 = dγ_rows (f32, t×w — this workgroup's `dγ` row, not a cross-row sum). Dispatch `(t, 1, 1)`:
/// workgroup `g` owns row `g`. The cross-row `dγ` accumulation is the second dispatch
/// [`rmsnorm_dgamma_dispatch`], recorded after this one; it sums the rows per column with no
/// float atomics, so the result is deterministic.
/// GLSL.std.450 `Floor`.
const GLSL_FLOOR: u32 = 8;

/// `floor(v·2³² + 0.5)·2⁻³²` — the reference's Q32.32 rounding (`mul_round`'s `(p + 2³¹) >> 32`
/// for a signed product, which is `floor(x + 0.5)`; an exact ±0.5 tie goes to +∞ on both sides).
/// The scale factors are powers of two, so both multiplies are exact. Above 2²⁴ units f32 has no
/// unit grid left and this is the identity, matching a rounding the reference cannot observe
/// either.
fn round_q32(k: &mut Kb, v: u32) -> u32 {
    let up = k.cf(4294967296.0);
    let half = k.cf(0.5);
    let down = k.cf(1.0 / 4294967296.0);
    let units = k.fmul(v, up);
    let biased = k.fadd(units, half);
    let floored = k.ext1(GLSL_FLOOR, biased);
    k.fmul(floored, down)
}

pub fn rmsnorm_bwd_module(w: u32, eps: f32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(w > 0, "rmsnorm_bwd: width must be non-zero");
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let gain = k.ssbo_f(1);
    let dy = k.ssbo_f(2);
    let dx = k.ssbo_f(3);
    let dgr = k.ssbo_f(4);
    let (red, redp) = k.shared_f32(WG);
    let (red2, red2p) = k.shared_f32(WG);
    let tid = k.local_index();
    let gid = k.gid_comp(0);
    let cwg = k.cu(WG);
    let cw = k.cu(w);
    let wgi = k.udiv(gid, cwg);
    let base = k.imul(wgi, cw);
    // ── Σ x² per lane → shared tree reduce → r = rsqrt(mean + eps) ──
    let acc = k.fvar_f();
    let zero = k.cf_bits(0x0000_0000);
    k.store_var(acc, zero);
    k.loop_step(tid, cw, cwg, |k, i| {
        let idx = k.iadd(base, i);
        let v = k.load_f(x, idx);
        let a0 = k.load_var_f(acc);
        let a1 = k.fma(v, v, a0);
        k.store_var(acc, a1);
    });
    let af = k.load_var_f(acc);
    k.shared_store(red, redp, tid, af);
    k.wg_barrier();
    k.tree_reduce(red, redp, WG, tid, false);
    let c0 = k.cu(0);
    let tot = k.shared_load(red, redp, c0);
    let invn = k.cf(1.0 / w as f32);
    let mean = k.fmul(tot, invn);
    // The reference rounds the mean to a whole Q32.32 unit before adding eps
    // (`learn::grad::dense::inv_rms`: `round(Σx² / (n·2³²))`). At `eps = 1e-6` and `mean ≪ eps`
    // that unit is 2^-32 real — a 2.3e-4 relative step on `mean + eps` when the mean is a few
    // units, and therefore on `r³` and on `dx`'s `r³·S` term. Without the same rounding the f32
    // mean is up to one unit away from the reference's, which is 5.9e4 Q32.32 units of `dx` on
    // the t=7 fixture. `Round` is ties-away-from-zero = the reference's `(2s + d) / (2d)` for
    // positive `s`; the scale factors are exact powers of two.
    let c2p32 = k.cf(4294967296.0);
    let units = k.fmul(mean, c2p32);
    let rounded = k.ext1(GLSL_ROUND, units);
    let c2m32 = k.cf(1.0 / 4294967296.0);
    let mean_q = k.fmul(rounded, c2m32);
    let epsv = k.cf(eps);
    let me = k.fadd(mean_q, epsv);
    let r = k.ext1(GLSL_INVERSE_SQRT, me);
    // ── S = Σ γ·dy·x → shared tree reduce ──
    let acc2 = k.fvar_f();
    k.store_var(acc2, zero);
    k.loop_step(tid, cw, cwg, |k, i| {
        let idx = k.iadd(base, i);
        let xv = k.load_f(x, idx);
        let dv = k.load_f(dy, idx);
        let gv = k.load_f(gain, i);
        let a_raw = k.fmul(gv, dv);
        let a = round_q32(k, a_raw);
        let t = k.fmul(a, xv);
        let s0 = k.load_var_f(acc2);
        let s1 = k.fadd(s0, t);
        k.store_var(acc2, s1);
    });
    let sf = k.load_var_f(acc2);
    k.shared_store(red2, red2p, tid, sf);
    k.wg_barrier();
    k.tree_reduce(red2, red2p, WG, tid, false);
    let s = k.shared_load(red2, red2p, c0);
    let r2 = k.fmul(r, r);
    let r3 = k.fmul(r2, r);
    let cwf = k.cf(w as f32);
    // ── dx_i = r·a_i − x_i·r³·S/w ; dγ_i = dy_i·x_i·r ──
    k.loop_step(tid, cw, cwg, |k, i| {
        let idx = k.iadd(base, i);
        let xv = k.load_f(x, idx);
        let dv = k.load_f(dy, idx);
        let gv = k.load_f(gain, i);
        let a_raw = k.fmul(gv, dv);
        let a = round_q32(k, a_raw);
        let p = k.fmul(r, a);
        let xr3 = k.fmul(xv, r3);
        let qn = k.fmul(xr3, s);
        let q = k.fdiv(qn, cwf);
        let d = k.fsub(p, q);
        k.store_f(dx, idx, d);
        let dg0 = k.fmul(dv, xv);
        let b = round_q32(k, dg0);
        let dg1 = k.fmul(b, r);
        k.store_f(dgr, idx, dg1);
    });
    k.finish(WG)
}

/// [`rmsnorm_bwd_module`] as bytes.
pub fn rmsnorm_bwd_bytes(w: u32, eps: f32) -> Vec<u8> {
    bytes(rmsnorm_bwd_module(w, eps))
}

/// **RMSNorm backward dγ column reduction (f32)** — the device-side replacement for the host
/// row fold. Binding 0 = dγ_rows (f32, t×w), binding 1 = dγ (f32, w). Thread `j` sums
/// `dγ_rows[i·w + j]` over `i` in `0..t` in f32 ascending and writes `dγ[j]`; no atomics, so
/// every output is a fixed serial association. Grid `⌈w/256⌉`, LocalSize 256; lanes past `w`
/// write nothing.
///
/// Complexity: one lane reads `t` elements and performs `t` adds — `O(t)` global loads and
/// adds per lane, counted in elements, with `w/t` of the `t·w` rows read per column.
pub fn rmsnorm_dgamma_module(t: u32, w: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(t > 0 && w > 0, "rmsnorm_dgamma: shape must be non-zero");
    let mut k = Kb::new();
    let rows = k.ssbo_f(0);
    let dg = k.ssbo_f(1);
    let gid = k.gid_comp(0);
    let cw = k.cu(w);
    let incol = k.ult(gid, cw);
    k.iff(incol, |mut k| {
        let acc = k.fvar_f();
        let zero = k.cf_bits(0x0000_0000);
        k.store_var(acc, zero);
        let c0 = k.cu(0);
        let ct = k.cu(t);
        k.loop_u(c0, ct, |mut k, i| {
            let base = k.imul(i, cw);
            let idx = k.iadd(base, gid);
            let v = k.load_f(rows, idx);
            let a0 = k.load_var_f(acc);
            let a1 = k.fadd(a0, v);
            k.store_var(acc, a1);
            k
        });
        let s = k.load_var_f(acc);
        k.store_f(dg, gid, s);
        k
    });
    k.finish(WG)
}

/// [`rmsnorm_dgamma_module`] as bytes.
pub fn rmsnorm_dgamma_bytes(t: u32, w: u32) -> Vec<u8> {
    bytes(rmsnorm_dgamma_module(t, w))
}

/// Host wrapper: build and record the RMSNorm-backward dγ reduction. `dgamma_rows` is the device
/// buffer of `t·w` f32 the main dispatch wrote; `dgamma` is `w` f32. Record after
/// [`rmsnorm_bwd_dispatch`] in the same command buffer (`rec_step` emits the compute→compute
/// barrier). Grid `(⌈w/256⌉, 1, 1)`.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; `dgamma_rows` must hold `t·w`
/// f32 and `dgamma` `w` f32.
#[cfg(feature = "gpu")]
pub unsafe fn rmsnorm_dgamma_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    dgamma_rows: crate::gpu::vulkan::Handle,
    dgamma: crate::gpu::vulkan::Handle,
    t: u32,
    w: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &rmsnorm_dgamma_bytes(t, w),
        &[dgamma_rows, dgamma],
        w.div_ceil(256),
        1,
        "rmsnorm_dgamma",
    )
}

/// Host wrapper: build and record the RMSNorm-backward dispatch. `dx` and `dgamma_rows` are
/// device buffers of `t·w` f32; `dgamma_rows` receives the per-row `dγ` and is reduced by
/// [`rmsnorm_dgamma_dispatch`]. Grid `(t, 1, 1)`, one workgroup per row.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// `t·w` (`x`, `dy`, `dx`, `dgamma_rows`) or `w` (`gain`) f32.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rmsnorm_bwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    x: crate::gpu::vulkan::Handle,
    gain: crate::gpu::vulkan::Handle,
    dy: crate::gpu::vulkan::Handle,
    dx: crate::gpu::vulkan::Handle,
    dgamma_rows: crate::gpu::vulkan::Handle,
    t: u32,
    w: u32,
    eps: f32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &rmsnorm_bwd_bytes(w, eps),
        &[x, gain, dy, dx, dgamma_rows],
        t,
        1,
        "rmsnorm_bwd",
    )
}
