//! simt_nn::bwd::lin — the two dense-layer backward products as f32 SPIR-V, the twins of
//! `learn::grad::dense::{linear_bwd_input,linear_bwd_weight}`.
//!
//! Naive one-thread-per-output kernels: the reduction length is a module constant (like `w` in
//! [`super::super::rmsnorm_bwd_module`]), so a shape gets its own module and the k-loop is a
//! compile-time bound with no dynamic state. Tiling and reuse are open work; these exist so the
//! parity gate has a device body to compare against the Q32.32 reference.

use super::super::*;

/// **Linear input adjoint (f32)** — the device twin of `learn::grad::dense::linear_bwd_input`:
/// `dy` is `n × o`, `w` is `o × i`, the output `dx` is `n × i` and
/// `dx[r,c] = Σ_k dy[r·o + k] · w[k·i + c]` in f32 with fused multiply-add accumulation.
/// Thread `g` owns output entry `g = r·i + c` (`r = g / i`, `c = g % i`) and walks `k` in `0..o`
/// ascending; grid `⌈n·i/256⌉`, LocalSize 256.
///
/// Bindings: 0 = dy (f32, n×o), 1 = w (f32, o×i), 2 = dx (f32, n×i).
///
/// Complexity: one thread reads `o` words of `dy` and `o` of `w` (2·o loads) and writes one word —
/// `O(o)` global loads per output, `O(1)` per thread; the dispatch reads `2·n·i·o` words against
/// the reused-read floor `n·o + o·i + n·i` (P86).
pub fn linear_bwd_input_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(
        n > 0 && i > 0 && o > 0,
        "linear_bwd_input: shape must be non-zero"
    );
    let mut k = Kb::new();
    let dy = k.ssbo_f(0);
    let w = k.ssbo_f(1);
    let dx = k.ssbo_f(2);
    let gid = k.gid_comp(0);
    let ci = k.cu(i);
    let co = k.cu(o);
    let total = k.cu(n * i);
    let inb = k.ult(gid, total);
    k.iff(inb, |mut k| {
        let r = k.udiv(gid, ci);
        let c = k.umod(gid, ci);
        let ro = k.imul(r, co);
        let acc = k.fvar_f();
        let zero = k.cf_bits(0x0000_0000);
        k.store_var(acc, zero);
        let c0 = k.cu(0);
        k.loop_u(c0, co, |mut k, kk| {
            let ia = k.iadd(ro, kk);
            let a = k.load_f(dy, ia);
            let ko = k.imul(kk, ci);
            let ib = k.iadd(ko, c);
            let b = k.load_f(w, ib);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(a, b, a0);
            k.store_var(acc, a1);
            k
        });
        let s = k.load_var_f(acc);
        k.store_f(dx, gid, s);
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
/// accumulation. Thread `g = r·i + c` walks `k` in `0..n` ascending; grid `⌈o·i/256⌉`, LocalSize
/// 256.
///
/// Bindings: 0 = dy (f32, n×o), 1 = x (f32, n×i), 2 = dw (f32, o×i).
///
/// Complexity: one thread reads `n` words of `dy` and `n` of `x` (2·n loads) and writes one word —
/// `O(n)` global loads per output, `O(1)` per thread; the dispatch reads `2·n·o·i` words against
/// the reused-read floor `n·o + n·i + o·i`.
pub fn linear_bwd_weight_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(
        n > 0 && i > 0 && o > 0,
        "linear_bwd_weight: shape must be non-zero"
    );
    let mut k = Kb::new();
    let dy = k.ssbo_f(0);
    let x = k.ssbo_f(1);
    let dw = k.ssbo_f(2);
    let gid = k.gid_comp(0);
    let ci = k.cu(i);
    let co = k.cu(o);
    let total = k.cu(o * i);
    let inb = k.ult(gid, total);
    k.iff(inb, |mut k| {
        let r = k.udiv(gid, ci);
        let c = k.umod(gid, ci);
        let acc = k.fvar_f();
        let zero = k.cf_bits(0x0000_0000);
        k.store_var(acc, zero);
        let c0 = k.cu(0);
        let cn = k.cu(n);
        k.loop_u(c0, cn, |mut k, kk| {
            let ko = k.imul(kk, co);
            let ia = k.iadd(ko, r);
            let a = k.load_f(dy, ia);
            let ki = k.imul(kk, ci);
            let ib = k.iadd(ki, c);
            let b = k.load_f(x, ib);
            let a0 = k.load_var_f(acc);
            let a1 = k.fma(a, b, a0);
            k.store_var(acc, a1);
            k
        });
        let s = k.load_var_f(acc);
        k.store_f(dw, gid, s);
        k
    });
    k.finish(WG)
}

/// [`linear_bwd_weight_module`] as bytes.
pub fn linear_bwd_weight_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    bytes(linear_bwd_weight_module(n, i, o))
}

/// Host wrapper: build and record the linear input adjoint. `dy` is the device buffer of `n·o`
/// f32, `w` of `o·i`, `dx` of `n·i`. Grid `(⌈n·i/256⌉, 1, 1)`, LocalSize 256.
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
        (n * i).div_ceil(256),
        1,
        "linear_bwd_input",
    )
}

/// Host wrapper: build and record the linear weight adjoint. `dy` is the device buffer of `n·o`
/// f32, `x` of `n·i`, `dw` of `o·i`. Grid `(⌈o·i/256⌉, 1, 1)`, LocalSize 256.
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
        (o * i).div_ceil(256),
        1,
        "linear_bwd_weight",
    )
}
