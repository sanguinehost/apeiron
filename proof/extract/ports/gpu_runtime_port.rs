//! The `word_gpu_runtime_*` extraction roots for the pure handoff arithmetic of
//! `src/gpu/runtime.rs`.
//!
//! `lib.rs` declares the real `src/gpu/mod.rs` as the crate-root `gpu` module; its `runtime`
//! child is unconditional, and the Vulkan half of `runtime.rs` is behind `cfg(feature = "gpu")`,
//! which this unit does not enable. So only the pure functions below are in the unit. Every
//! wrapper calls the real body and restates no arithmetic. `bar_mode` is not rooted: it reads the
//! `PREFILL_BAR` process flag through `std::env` and `OnceLock`, which are host boundaries; its
//! pure consequence is `bar_emits`, rooted here and modelled in `Gpu.Handoff`.

/// Real `gpu::runtime::stage_align`.
pub fn word_gpu_runtime_stage_align(n: usize) -> usize {
    crate::gpu::runtime::stage_align(n)
}

/// Real `gpu::runtime::stage_fit`.
pub fn word_gpu_runtime_stage_fit(off: usize, len: usize, cap: usize) -> bool {
    crate::gpu::runtime::stage_fit(off, len, cap)
}

/// Real `gpu::runtime::stage_region`.
pub fn word_gpu_runtime_stage_region(off: usize, len: usize) -> (u64, u64, u64) {
    crate::gpu::runtime::stage_region(off, len)
}

/// Real `gpu::runtime::scatter_src`.
pub fn word_gpu_runtime_scatter_src(off: usize, s: u64) -> u64 {
    crate::gpu::runtime::scatter_src(off, s)
}

/// Real `gpu::runtime::binds_exactly`.
pub fn word_gpu_runtime_binds_exactly(n: usize, bindings: u32) -> bool {
    crate::gpu::runtime::binds_exactly(n, bindings)
}

/// Real `gpu::runtime::write_chunk`.
pub fn word_gpu_runtime_write_chunk(done: usize, bytes: usize, cap: usize) -> usize {
    crate::gpu::runtime::write_chunk(done, bytes, cap)
}

/// Real `gpu::runtime::bar_emits`.
pub fn word_gpu_runtime_bar_emits(mode: u32) -> bool {
    crate::gpu::runtime::bar_emits(mode)
}
