//! simt_nn::bwd::lin — the two linear adjoint GEMMs as f32 SPIR-V, the twins of
//! `learn::grad::dense::{linear_bwd_input,linear_bwd_weight}`.
//!
//! Both run the [`tiled_gemm`] body the forward product uses — `BM × BN` output tile per
//! workgroup, `BK`-wide reduction slice staged in the padded shared arrays, `TM × TN` register
//! accumulators per lane — and differ only in which operand is `A`, which is `B` and which
//! dimension the reduction runs over. One emitted body, three operand descriptions across the two
//! `lin` modules.
//!
//! The f32 association is the naive kernel's, word for word: one accumulator per output seeded
//! `0.0`, `Fma(a, b, acc)` for `k` strictly ascending over the whole reduction, zeros staged on a
//! partial last slice (`fma(0, 0, acc) = acc`). No split-k, no reduction across k, no
//! cooperative-matrix or f16 path: those change the words and are not these kernels.

use super::super::fwd::lin::{tiled_gemm, Weld, BM, BN};
use super::super::*;

/// **Linear input adjoint (f32)** — the device twin of `learn::grad::dense::linear_bwd_input`:
/// `dy` is `n × o`, `w` is `o × i`, the output `dx` is `n × i` and
/// `dx[r,c] = Σ_k dy[r·o + k] · w[k·i + c]` in f32 with fused multiply-add accumulation.
/// Workgroup `q = ⌈i/BN⌉·r_tile + c_tile` owns the `BM × BN` output tile at
/// `(BM·r_tile, BN·c_tile)`; the reduction runs over `o` in ascending 16-wide slices
/// ([`tiled_gemm`]). Grid `⌈n/BM⌉·⌈i/BN⌉`, LocalSize 256.
///
/// Bindings: 0 = dy (f32, n×o), 1 = w (f32, o×i), 2 = dx (f32, n×i).
///
/// Complexity: see [`tiled_gemm`] with `rows = n`, `cols = i`, `kdim = o`.
pub fn linear_bwd_input_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    assert!(
        n > 0 && i > 0 && o > 0,
        "linear_bwd_input: shape must be non-zero"
    );
    let mut k = Kb::new();
    let dy = k.ssbo_f(0);
    let w = k.ssbo_f(1);
    let dx = k.ssbo_f(2);
    let a = Weld {
        buf: dy,
        rs: k.cu(o),
        tile_extent: n,
        k_extent: o,
        k_inner: true,
    };
    let b = Weld {
        buf: w,
        rs: k.cu(i),
        tile_extent: i,
        k_extent: o,
        k_inner: false,
    };
    let local = tiled_gemm(&mut k, &a, &b, dx, n, i, o);
    k.finish(local)
}

/// [`linear_bwd_input_module`] as bytes.
pub fn linear_bwd_input_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    bytes(linear_bwd_input_module(n, i, o))
}

/// **Linear weight adjoint (f32)** — the device twin of
/// `learn::grad::dense::linear_bwd_weight`: `dy` is `n × o`, `x` is `n × i`, the output `dw` is
/// `o × i` and `dw[r,c] = Σ_k dy[k·o + r] · x[k·i + c]` in f32 with fused multiply-add
/// accumulation. Workgroup `q = ⌈i/BN⌉·r_tile + c_tile` owns the `BM × BN` output tile at
/// `(BM·r_tile, BN·c_tile)`; the reduction runs over `n` in ascending 16-wide slices, and both
/// operands are read k-major (`dy[k·o + m]`, `x[k·i + n]`) so consecutive lanes cover consecutive
/// output positions ([`tiled_gemm`]). Grid `⌈o/BM⌉·⌈i/BN⌉`, LocalSize 256.
///
/// Bindings: 0 = dy (f32, n×o), 1 = x (f32, n×i), 2 = dw (f32, o×i).
///
/// Complexity: see [`tiled_gemm`] with `rows = o`, `cols = i`, `kdim = n`.
pub fn linear_bwd_weight_module(n: u32, i: u32, o: u32) -> Vec<u32> {
    assert!(
        n > 0 && i > 0 && o > 0,
        "linear_bwd_weight: shape must be non-zero"
    );
    let mut k = Kb::new();
    let dy = k.ssbo_f(0);
    let x = k.ssbo_f(1);
    let dw = k.ssbo_f(2);
    let a = Weld {
        buf: dy,
        rs: k.cu(o),
        tile_extent: o,
        k_extent: n,
        k_inner: false,
    };
    let b = Weld {
        buf: x,
        rs: k.cu(i),
        tile_extent: i,
        k_extent: n,
        k_inner: false,
    };
    let local = tiled_gemm(&mut k, &a, &b, dw, o, i, n);
    k.finish(local)
}

/// [`linear_bwd_weight_module`] as bytes.
pub fn linear_bwd_weight_bytes(n: u32, i: u32, o: u32) -> Vec<u8> {
    bytes(linear_bwd_weight_module(n, i, o))
}

/// Host wrapper: build and record the linear input adjoint. `dy` is the device buffer of `n·o`
/// f32, `w` of `o·i`, `dx` of `n·i`. Grid `(⌈n/BM⌉·⌈i/BN⌉, 1, 1)`, one `BM × BN` output tile per
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
        n.div_ceil(BM) * i.div_ceil(BN),
        1,
        "linear_bwd_input",
    )
}

/// Host wrapper: build and record the linear weight adjoint. `dy` is the device buffer of `n·o`
/// f32, `x` of `n·i`, `dw` of `o·i`. Grid `(⌈o/BM⌉·⌈i/BN⌉, 1, 1)`, one `BM × BN` output tile per
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
        o.div_ceil(BM) * i.div_ceil(BN),
        1,
        "linear_bwd_weight",
    )
}
