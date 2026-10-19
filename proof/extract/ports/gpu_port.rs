//! The `word_gpu_*` extraction roots for the real GPU module tree.
//!
//! `lib.rs` declares the real `src/gpu/mod.rs` as the crate-root `gpu` module, so this port file
//! only carries the monomorphic roots. Every wrapper calls the real builder body and restates no
//! SPIR-V word; the wrappers fix the concrete `u32` argument and `Vec<u32>` return types.

/// Real `gpu::nn::two_level_chart_buffer`.
pub fn word_gpu_two_level_chart_buffer(sub: &[u32], sup: &[u32]) -> Vec<u32> {
    crate::gpu::nn::two_level_chart_buffer(sub, sup)
}

/// Real `gpu::nn::embed_f16_module`.
pub fn word_gpu_embed_f16_module(hidden: u32, vocab: u32, local: u32) -> Vec<u32> {
    crate::gpu::nn::embed_f16_module(hidden, vocab, local)
}

/// Real `gpu::nn::silu_mul_module`.
pub fn word_gpu_silu_mul_module(n: u32, local: u32, u_off: u32) -> Vec<u32> {
    crate::gpu::nn::silu_mul_module(n, local, u_off)
}

/// Real `gpu::nn::add_inplace_module`.
pub fn word_gpu_add_inplace_module(n: u32, local: u32) -> Vec<u32> {
    crate::gpu::nn::add_inplace_module(n, local)
}

/// Real `gpu::nn::commit_advance_module`.
pub fn word_gpu_commit_advance_module(slot: u32) -> Vec<u32> {
    crate::gpu::nn::commit_advance_module(slot)
}

/// Real `gpu::nn::copy_vec_module`.
pub fn word_gpu_copy_vec_module(n: u32, src_off: u32) -> Vec<u32> {
    crate::gpu::nn::copy_vec_module(n, src_off)
}
