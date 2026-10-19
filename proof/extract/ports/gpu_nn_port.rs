//! The `word_gpu_nn_*` extraction roots for the `Kb` SPIR-V builder in `src/gpu/nn/mod.rs`.
//!
//! `lib.rs` declares the real `src/gpu/mod.rs` as the crate-root `gpu` module, so this port file
//! only carries the monomorphic roots for the builder methods the six module roots in
//! `ports/gpu_port.rs` do not reach. Every wrapper constructs the real [`Kb`] and calls the real
//! method; none restates a SPIR-V word, an opcode or a section layout.
//!
//! # Merge (`proof/extract/lib.rs`)
//!
//! The unit is declared at the crate root, beside `gpu_port`:
//!
//! ```text
//! #[path = "ports/gpu_nn_port.rs"]
//! pub mod gpu_nn_port;
//! ```
//!
//! and `proof/regen.sh` names each root under `--start-from crate::gpu_nn_port::…`.
//!
//! # Visibility
//!
//! `Kb` and the 25 builder methods below are `pub(crate)` in `src/gpu/nn/mod.rs`, so
//! this crate-root unit can name them; the change widens no public API.
//!
//! # Refusals (recorded on the `src/gpu/nn/mod.rs` debt row)
//!
//! - `Kb::tree_reduce`: `Unreachable`, `mod.rs` 912:24-916:17 (the `if is_max` inside the
//!   `iff` closure).
//! - `Kb::rmsnorm_wg`: `Improperly typed constant value`, `mod.rs` 993:27-993:41
//!   (`self.cf(1.0 / n as f32)`); `Kb::rmsnorm_quant_wg` inherits the refusal through it.
//! - `Kb::loop_step`: its `impl FnOnce(&mut Kb, u32)` body extracts as
//!   `PairMutKbU32Tuple.call_once : … → Result Kb` against an expected `Result Unit`, so the
//!   generated Lean does not typecheck (`ApeironWord.lean` `Kb.loop_step`). `Kb::loop_u`'s
//!   by-value `FnOnce(Kb, u32) -> Kb` is the shape that works.
//! - `Kb::emit_quantize_from_shared` (global chart twin): reaches `loop_step`, and
//!   `Kb::emit_quantize_to_shared` / `…_win` additionally reach `attn_dp4a::Kb::int8_code`,
//!   `Improperly typed constant value` at `mod.rs` 417:4-420:5 (`cf` of a literal `f32`; the
//!   `cf_bits` form is the extraction-safe twin).
//! - `Kb::cf`: its body's `f32::to_bits` is an unmodelled external. Rooting it adds
//!   `core.f32.F32.to_bits` to the generated axiom set, which `proof/boundary-axioms.txt`
//!   does not admit and this lane may not widen.
//! - `ChartScale::two_level`: `u32::div_ceil` is the same class (`core.num.U32.div_ceil`).

use crate::gpu::nn::{ChartScale, Kb};

// ── lazily declared types and SSBO bindings ───────────────────────────────────────────────────

/// Real `Kb::ensure_f16`.
pub fn word_gpu_kb_ensure_f16() {
    let mut k = Kb::new();
    k.ensure_f16();
}

/// Real `Kb::ssbo_f16`.
pub fn word_gpu_kb_ssbo_f16(binding: u32) -> u32 {
    let mut k = Kb::new();
    k.ssbo_f16(binding)
}

/// Real `Kb::load_f16`.
pub fn word_gpu_kb_load_f16(buf: u32, idx: u32) -> u32 {
    let mut k = Kb::new();
    k.load_f16(buf, idx)
}

/// Real `Kb::store_f16`.
pub fn word_gpu_kb_store_f16(buf: u32, idx: u32, val: u32) {
    let mut k = Kb::new();
    k.store_f16(buf, idx, val);
}

// ── constants and function-scope variables ────────────────────────────────────────────────────

/// Real `Kb::fvar`.
pub fn word_gpu_kb_fvar(ptr_ty: u32) -> u32 {
    let mut k = Kb::new();
    k.fvar(ptr_ty)
}

/// Real `Kb::fvar_f`.
pub fn word_gpu_kb_fvar_f() -> u32 {
    let mut k = Kb::new();
    k.fvar_f()
}

/// Real `Kb::fvar_u`.
pub fn word_gpu_kb_fvar_u() -> u32 {
    let mut k = Kb::new();
    k.fvar_u()
}

/// Real `Kb::load_var_f`.
pub fn word_gpu_kb_load_var_f(var: u32) -> u32 {
    let mut k = Kb::new();
    k.load_var_f(var)
}

/// Real `Kb::load_var_u`.
pub fn word_gpu_kb_load_var_u(var: u32) -> u32 {
    let mut k = Kb::new();
    k.load_var_u(var)
}

/// Real `Kb::store_var`.
pub fn word_gpu_kb_store_var(var: u32, val: u32) {
    let mut k = Kb::new();
    k.store_var(var, val);
}

/// Real `Kb::sint_ty`.
pub fn word_gpu_kb_sint_ty() -> u32 {
    let mut k = Kb::new();
    k.sint_ty()
}

// ── arithmetic sugar ──────────────────────────────────────────────────────────────────────────

/// Real `Kb::mandala`: `idx &&& mask` as one instruction.
pub fn word_gpu_kb_mandala(idx: u32, mask: u32) -> u32 {
    let mut k = Kb::new();
    k.mandala(idx, mask)
}

/// Real `Kb::fsub`.
pub fn word_gpu_kb_fsub(a: u32, b: u32) -> u32 {
    let mut k = Kb::new();
    k.fsub(a, b)
}

/// Real `Kb::fma`.
pub fn word_gpu_kb_fma(a: u32, b: u32, c: u32) -> u32 {
    let mut k = Kb::new();
    k.fma(a, b, c)
}

/// Real `Kb::utof`.
pub fn word_gpu_kb_utof(a: u32) -> u32 {
    let mut k = Kb::new();
    k.utof(a)
}

/// Real `Kb::ext2`, a two-operand `GLSL.std.450` call.
pub fn word_gpu_kb_ext2(inst: u32, x: u32, y: u32) -> u32 {
    let mut k = Kb::new();
    k.ext2(inst, x, y)
}

/// Real `Kb::umin`, the uint `min` of a partial last fold.
pub fn word_gpu_kb_umin(a: u32, b: u32) -> u32 {
    let mut k = Kb::new();
    k.umin(a, b)
}

// ── structured control flow ──────────────────────────────────────────────────────────────────

/// Real `Kb::loop_u`, with a concrete body closure.
pub fn word_gpu_kb_loop_u(start: u32, bound: u32) {
    let mut k = Kb::new();
    k.loop_u(start, bound, |k, _c| k);
}

/// Real `Kb::fold_bias_store`, with the optional bias and the split-K reduce.
pub fn word_gpu_kb_fold_bias_store(
    src: u32,
    bias: Option<u32>,
    dst: u32,
    i: u32,
    n: u32,
    segs: u32,
) {
    let mut k = Kb::new();
    k.fold_bias_store(src, bias, dst, i, n, segs);
}

// ── workgroup memory, barriers and reductions ────────────────────────────────────────────────

/// Real `Kb::shared_f32`.
pub fn word_gpu_kb_shared_f32(len: u32) -> (u32, u32) {
    let mut k = Kb::new();
    k.shared_f32(len)
}

/// Real `Kb::shared_load`.
pub fn word_gpu_kb_shared_load(var: u32, ptr_el: u32, idx: u32) -> u32 {
    let mut k = Kb::new();
    k.shared_load(var, ptr_el, idx)
}

/// Real `Kb::shared_store`.
pub fn word_gpu_kb_shared_store(var: u32, ptr_el: u32, idx: u32, val: u32) {
    let mut k = Kb::new();
    k.shared_store(var, ptr_el, idx, val);
}

/// Real `Kb::wg_barrier`.
pub fn word_gpu_kb_wg_barrier() {
    let mut k = Kb::new();
    k.wg_barrier();
}

/// Real `Kb::strided_idx`.
pub fn word_gpu_kb_strided_idx(tid: u32, t: u32, wg: u32) -> u32 {
    let mut k = Kb::new();
    k.strided_idx(tid, t, wg)
}

/// Real `Kb::local_index`, the lazily made `LocalInvocationIndex` builtin.
pub fn word_gpu_kb_local_index() -> u32 {
    let mut k = Kb::new();
    k.local_index()
}

// ── word-slice conversion and the chart scale face ──────────────────────────────────────────

/// Real `gpu::nn::bytes`: the little-endian byte image of the assembled words.
pub fn word_gpu_bytes(words: Vec<u32>) -> Vec<u8> {
    crate::gpu::nn::bytes(words)
}

/// Real `ChartScale::dm`, the single-level scale face.
pub fn word_gpu_chart_scale_dm(dm: u32) -> ChartScale {
    ChartScale::dm(dm)
}
