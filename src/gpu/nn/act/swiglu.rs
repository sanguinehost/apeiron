//! simt_nn::act::swiglu — the SwiGLU forward and backward maps as f32 SPIR-V, the twins of
//! `learn::grad::dense::swiglu_fwd` and `learn::grad::dense::swiglu_bwd`.
//!
//! Both kernels are one thread per element. The reference evaluates in Q32.32 with one rounded
//! product per multiply; the device keeps every value in f32 and issues the same product sequence,
//! so the parity test compares the two in Q32.32 units with a pinned per-entry bound.
//!
//! * forward: `s = 1/(1+exp(−g))`, `out = (g/(1+exp(−g)))·u`.
//! * backward: `s = 1/(1+exp(−g))`, `dsilu = s·(1 + g·(1−s))`, `dgate = (dy·u)·dsilu`,
//!   `dup = dy·silu(g)`.

use super::super::*;

/// **SwiGLU forward (f32)** — the device twin of `learn::grad::dense::swiglu_fwd`:
/// `out[i] = silu(gate[i])·up[i]` with `silu(g) = g·σ(g)` and `σ(g) = 1/(1+exp(−g))`.
/// One thread per element; grid `⌈n/256⌉`, LocalSize 256.
///
/// Bindings: 0 = gate (f32, n), 1 = up (f32, n), 2 = out (f32, n).
///
/// Complexity: one thread issues two loads, one store, one `Exp` and one division — O(1) time and
/// O(1) space per thread; the dispatch reads `2·n` words and writes `n` (P96).
pub fn swiglu_fwd_module(n: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(n > 0, "swiglu_fwd: length must be non-zero");
    let mut k = Kb::new();
    let gate = k.ssbo_f(0);
    let up = k.ssbo_f(1);
    let out = k.ssbo_f(2);
    let i = k.gid_comp(0);
    let cn = k.cu(n);
    let guard = k.ult(i, cn);
    k.iff(guard, |mut k| {
        let gv = k.load_f(gate, i);
        let uv = k.load_f(up, i);
        let f = k.float;
        let ng = k.e(OP_FNEG, f, &[gv]);
        let eg = k.ext1(GLSL_EXP, ng);
        let one = k.cf_bits(0x3f80_0000);
        let den = k.fadd(one, eg);
        // `g/den` is the reference's `silu_one`: one rounding of `g·2³²/den`, not `g·round(σ)`.
        let sil = k.fdiv(gv, den);
        let o = k.fmul(sil, uv);
        k.store_f(out, i, o);
        k
    });
    k.finish(WG)
}

/// [`swiglu_fwd_module`] as bytes.
pub fn swiglu_fwd_bytes(n: u32) -> Vec<u8> {
    bytes(swiglu_fwd_module(n))
}

/// Host wrapper: build and record the SwiGLU forward. `gate`, `up` and `out` hold `n` f32 each;
/// grid `(⌈n/256⌉, 1, 1)`, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold `n` f32.
#[cfg(feature = "gpu")]
pub unsafe fn swiglu_fwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    gate: crate::gpu::vulkan::Handle,
    up: crate::gpu::vulkan::Handle,
    out: crate::gpu::vulkan::Handle,
    n: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &swiglu_fwd_bytes(n),
        &[gate, up, out],
        n.div_ceil(256),
        1,
        "swiglu_fwd",
    )
}

/// **SwiGLU backward (f32)** — the device twin of `learn::grad::dense::swiglu_bwd`.
/// With `s = σ(g)`, `dsilu = s·(1 + g·(1−s))`: `dgate[i] = (dy[i]·up[i])·dsilu` and
/// `dup[i] = dy[i]·silu(g)`, the same two rounded products in the same order as the reference.
/// One thread per element; grid `⌈n/256⌉`, LocalSize 256.
///
/// Bindings: 0 = gate (f32, n), 1 = up (f32, n), 2 = dy (f32, n), 3 = dgate (f32, n),
/// 4 = dup (f32, n).
///
/// Complexity: one thread issues three loads, two stores, one `Exp`, two divisions and four
/// multiplies — O(1) time and O(1) space per thread; the dispatch reads `3·n` words and writes
/// `2·n` (P96).
pub fn swiglu_bwd_module(n: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(n > 0, "swiglu_bwd: length must be non-zero");
    let mut k = Kb::new();
    let gate = k.ssbo_f(0);
    let up = k.ssbo_f(1);
    let dy = k.ssbo_f(2);
    let dgate = k.ssbo_f(3);
    let dup = k.ssbo_f(4);
    let i = k.gid_comp(0);
    let cn = k.cu(n);
    let guard = k.ult(i, cn);
    k.iff(guard, |mut k| {
        let gv = k.load_f(gate, i);
        let uv = k.load_f(up, i);
        let dv = k.load_f(dy, i);
        let f = k.float;
        let one = k.cf_bits(0x3f80_0000);
        let ng = k.e(OP_FNEG, f, &[gv]);
        let eg = k.ext1(GLSL_EXP, ng);
        let den = k.fadd(one, eg);
        let s = k.fdiv(one, den);
        // silu(g) = g·σ(g): one rounding of `g/den`, the reference's `silu_one`.
        let sil = k.fdiv(gv, den);
        // dsilu = s·(1 + g·(1 − s)): `(ONE − s)` then `g·(1−s)` then `1 + …` then `s·…`.
        let om = k.fsub(one, s);
        let gm = k.fmul(gv, om);
        let t = k.fadd(one, gm);
        let dsilu = k.fmul(s, t);
        // `dgu = round(dy·u)` then `dgate = round(dgu·dsilu)`; `dup = round(dy·silu)`.
        let dgu = k.fmul(dv, uv);
        let dg = k.fmul(dgu, dsilu);
        let du = k.fmul(dv, sil);
        k.store_f(dgate, i, dg);
        k.store_f(dup, i, du);
        k
    });
    k.finish(WG)
}

/// [`swiglu_bwd_module`] as bytes.
pub fn swiglu_bwd_bytes(n: u32) -> Vec<u8> {
    bytes(swiglu_bwd_module(n))
}

/// Host wrapper: build and record the SwiGLU backward. `gate`, `up`, `dy`, `dgate` and `dup`
/// hold `n` f32 each; grid `(⌈n/256⌉, 1, 1)`, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold `n` f32.
#[cfg(feature = "gpu")]
pub unsafe fn swiglu_bwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    gate: crate::gpu::vulkan::Handle,
    up: crate::gpu::vulkan::Handle,
    dy: crate::gpu::vulkan::Handle,
    dgate: crate::gpu::vulkan::Handle,
    dup: crate::gpu::vulkan::Handle,
    n: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &swiglu_bwd_bytes(n),
        &[gate, up, dy, dgate, dup],
        n.div_ceil(256),
        1,
        "swiglu_bwd",
    )
}
