//! simt_nn::fwd::rope — the rotary position embedding as f32 SPIR-V, the device twin of
//! `learn::grad::attn::rope_fwd` and its adjoint `rope_bwd`.
//!
//! The engine table (`engine::ops::rope::RopeTable`) computes the phase of rotary dimension `j` at
//! position `pos` as the exact integer `(turns[j]·pos) mod 2^64` in `Z/2^64`, then maps it through
//! `turn_to_radians` into the `fixed_sin_cos` window. That integer round is not reproducible from
//! device f32 trigonometry across positions, so the `cos`/`sin` tables are **host inputs** to these
//! kernels (the table builder's home is a separate task): binding 1 is `cos`, binding 2 is `sin`,
//! each `t·hd/2` f32 laid out row-major over `(pos, j)`, already advanced by the one integer round.
//!
//! One thread owns one rotary pair. With `p = gid` over `t·nh·hd/2` pairs, `j = p mod (hd/2)`,
//! `vec = ⌊p / (hd/2)⌋` and `row = ⌊vec / nh⌋`; the pair is `(x[vec·hd + j], x[vec·hd + j + hd/2])`
//! and its phase index is `row·(hd/2) + j`.
//!
//! Domain: `hd ≥ 2` and even, `nh > 0`, `t > 0`. `pairs = t·nh·hd/2` is baked into the module
//! because the builder has no runtime parameter buffer, so the trailing lanes of the last
//! workgroup guard on `gid < pairs` and touch nothing.
//!
//! Complexity: O(1) time and O(1) space per thread (two loads of `x`, one of `cos`, one of `sin`,
//! two stores), no shared memory and no barrier. One dispatch moves `2·t·nh·hd + t·hd` words:
//! `x` read (`t·nh·hd`), the two half-tables read (`t·hd`), `out` written (`t·nh·hd`).

use super::super::*;

/// **RoPE forward (f32)** — `out0 = x0·c − x1·s`, `out1 = x1·c + x0·s` per rotary pair, in that
/// arithmetic order (two separately rounded products, then the add/sub).
///
/// Bindings: 0 = `x` (f32, `t·nh·hd`), 1 = `cos` (f32, `t·hd/2`), 2 = `sin` (f32, `t·hd/2`),
/// 3 = `out` (f32, `t·nh·hd`). Grid `(⌈t·nh·hd/2 / 256⌉, 1, 1)`, LocalSize 256.
pub fn rope_fwd_module(hd: u32, nh: u32, t: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(
        hd >= 2 && hd % 2 == 0,
        "rope_fwd: head_dim must be at least 2 and even"
    );
    assert!(nh > 0 && t > 0, "rope_fwd: shape must be non-zero");
    let pairs = u64::from(t) * u64::from(nh) * u64::from(hd) / 2;
    assert!(
        pairs <= u64::from(u32::MAX),
        "rope_fwd: pair count must fit u32"
    );
    let mut k = Kb::new();
    let x = k.ssbo_f(0);
    let cos = k.ssbo_f(1);
    let sin = k.ssbo_f(2);
    let out = k.ssbo_f(3);
    let gid = k.gid_comp(0);
    let cpairs = k.cu(pairs as u32);
    let d2 = k.cu(hd / 2);
    let cnh = k.cu(nh);
    let chd = k.cu(hd);
    let inb = k.ult(gid, cpairs);
    k.iff(inb, |mut k| {
        let j = k.umod(gid, d2);
        let vec = k.udiv(gid, d2);
        let row = k.udiv(vec, cnh);
        let base = k.imul(vec, chd);
        let i0 = k.iadd(base, j);
        let i1 = k.iadd(i0, d2);
        let rowbase = k.imul(row, d2);
        let tc = k.iadd(rowbase, j);
        let a = k.load_f(x, i0);
        let b = k.load_f(x, i1);
        let c = k.load_f(cos, tc);
        let s = k.load_f(sin, tc);
        let ac = k.fmul(a, c);
        let bs = k.fmul(b, s);
        let o0 = k.fsub(ac, bs);
        let bc = k.fmul(b, c);
        let ag = k.fmul(a, s);
        let o1 = k.fadd(bc, ag);
        k.store_f(out, i0, o0);
        k.store_f(out, i1, o1);
        k
    });
    k.finish(WG)
}

/// [`rope_fwd_module`] as bytes.
pub fn rope_fwd_bytes(hd: u32, nh: u32, t: u32) -> Vec<u8> {
    bytes(rope_fwd_module(hd, nh, t))
}

/// Host wrapper: build and record the RoPE forward. `x` is the device buffer of `t·nh·hd` f32,
/// `cos`/`sin` of `t·hd/2` each, `out` of `t·nh·hd`. Grid `(⌈t·nh·hd/2 / 256⌉, 1, 1)`,
/// LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn rope_fwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    x: crate::gpu::vulkan::Handle,
    cos: crate::gpu::vulkan::Handle,
    sin: crate::gpu::vulkan::Handle,
    out: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &rope_fwd_bytes(hd, nh, t),
        &[x, cos, sin, out],
        (t * nh * hd / 2).div_ceil(256),
        1,
        "rope_fwd",
    )
}

/// **RoPE adjoint (f32)** — the transpose of [`rope_fwd_module`]: `dx0 = dy0·c + dy1·s`,
/// `dx1 = dy1·c − dy0·s`, the same `c`/`s` of the forward phase.
///
/// Bindings: 0 = `dy` (f32, `t·nh·hd`), 1 = `cos` (f32, `t·hd/2`), 2 = `sin` (f32, `t·hd/2`),
/// 3 = `dx` (f32, `t·nh·hd`). Grid `(⌈t·nh·hd/2 / 256⌉, 1, 1)`, LocalSize 256.
pub fn rope_bwd_module(hd: u32, nh: u32, t: u32) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(
        hd >= 2 && hd % 2 == 0,
        "rope_bwd: head_dim must be at least 2 and even"
    );
    assert!(nh > 0 && t > 0, "rope_bwd: shape must be non-zero");
    let pairs = u64::from(t) * u64::from(nh) * u64::from(hd) / 2;
    assert!(
        pairs <= u64::from(u32::MAX),
        "rope_bwd: pair count must fit u32"
    );
    let mut k = Kb::new();
    let dy = k.ssbo_f(0);
    let cos = k.ssbo_f(1);
    let sin = k.ssbo_f(2);
    let dx = k.ssbo_f(3);
    let gid = k.gid_comp(0);
    let cpairs = k.cu(pairs as u32);
    let d2 = k.cu(hd / 2);
    let cnh = k.cu(nh);
    let chd = k.cu(hd);
    let inb = k.ult(gid, cpairs);
    k.iff(inb, |mut k| {
        let j = k.umod(gid, d2);
        let vec = k.udiv(gid, d2);
        let row = k.udiv(vec, cnh);
        let base = k.imul(vec, chd);
        let i0 = k.iadd(base, j);
        let i1 = k.iadd(i0, d2);
        let rowbase = k.imul(row, d2);
        let tc = k.iadd(rowbase, j);
        let a = k.load_f(dy, i0);
        let b = k.load_f(dy, i1);
        let c = k.load_f(cos, tc);
        let s = k.load_f(sin, tc);
        let ac = k.fmul(a, c);
        let bs = k.fmul(b, s);
        let o0 = k.fadd(ac, bs);
        let bc = k.fmul(b, c);
        let ag = k.fmul(a, s);
        let o1 = k.fsub(bc, ag);
        k.store_f(dx, i0, o0);
        k.store_f(dx, i1, o1);
        k
    });
    k.finish(WG)
}

/// [`rope_bwd_module`] as bytes.
pub fn rope_bwd_bytes(hd: u32, nh: u32, t: u32) -> Vec<u8> {
    bytes(rope_bwd_module(hd, nh, t))
}

/// Host wrapper: build and record the RoPE adjoint. `dy` is the device buffer of `t·nh·hd` f32,
/// `cos`/`sin` of `t·hd/2` each, `dx` of `t·nh·hd`. Grid `(⌈t·nh·hd/2 / 256⌉, 1, 1)`,
/// LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// its declared shape in f32.
#[cfg(feature = "gpu")]
pub unsafe fn rope_bwd_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    dy: crate::gpu::vulkan::Handle,
    cos: crate::gpu::vulkan::Handle,
    sin: crate::gpu::vulkan::Handle,
    dx: crate::gpu::vulkan::Handle,
    t: u32,
    nh: u32,
    hd: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &rope_bwd_bytes(hd, nh, t),
        &[dy, cos, sin, dx],
        (t * nh * hd / 2).div_ceil(256),
        1,
        "rope_bwd",
    )
}
