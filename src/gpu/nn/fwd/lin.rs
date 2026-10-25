//! simt_nn::fwd::lin — the dense-layer forward product, on the engine's one GEMM.
//!
//! `linear_fwd_dispatch` is the `Y = X·Wᵀ` operand description of the bf16 face of
//! `src/gpu/spirv/coopmat_gemm.comp` (`-DOPF=1`, builder
//! [`crate::gpu::nn::coopmat_gemm_bf16_module`]). The chart GEMM that holds pp512 1.795× llama.cpp
//! CUDA is the same file; this face differs only in the operand format and its guards, so the
//! layer's three products and prefill's GEMM are one kernel with faces, never two.
//!
//! The f32 scalar `tiled_gemm` body this module carried until the KD port (`BM × BN` register
//! tiling, `BK = 16` shared slices, `Fma` per k) was the second implementation of the same role: it
//! is deleted with its `.spv`, reached by nothing. Its measured cost stands in
//! `tests/gpu_lin_bf16.rs`'s record: 401 ms of linear GEMM a KD step against the face's 131–146 ms
//! on the same harness.
//!
//! Precision: operands are rounded to bf16 once at the shared-tile store and accumulated in f32
//! (`bf16·bf16 → f32`, `M16N16K16` cooperative matrix at subgroup scope), the NanoQuant
//! reference's own chain (`HMMA.16816.F32.BF16`).

use super::super::*;

/// Host wrapper: build and record the linear forward product. `x` is the device buffer of `n·i`
/// f32, `w` of `o·i`, `y` of `n·o`. Grid `(⌈o/BM⌉, ⌈n/BN⌉)` at the tile [`tile_for`] measures for
/// `o` rows.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn linear_fwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    x: crate::gpu::vulkan::Handle,
    w: crate::gpu::vulkan::Handle,
    y: crate::gpu::vulkan::Handle,
    n: u32,
    i: u32,
    o: u32,
) -> crate::gpu::decode::Step {
    let d = LinBf16::fwd(n, i, o);
    lin_bf16_step(g, w, x, y, &d, tile_for(d.rows), "linear_fwd")
}
