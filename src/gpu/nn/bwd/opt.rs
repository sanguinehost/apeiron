//! simt_nn::bwd::opt — the Adam update as one f32 SPIR-V elementwise pass, the device twin of
//! `learn::fold::finish::train::Adam::apply`.
//!
//! The oracle's words and scales (read from `src/learn/fold/finish/train.rs`): `p`, `g` and the
//! first moment `m` are Q32.32 `i64` (`ONE = 1 << 32`); the second moment `v` is Q64.64 `i128`,
//! because `t1 = rshift_rne(β₂·v, 32)` with `β₂` at Q32.32 is `round(β₂·v/2^32)` and the squared
//! Q32.32 gradient `g·g` is exactly Q64.64. `narrow(isqrt(v))` then returns Q32.32 because
//! `√(v_real·2^64) = √v_real·2^32`. Per element and step `t` the oracle computes
//!
//!     m ← β₁·m + (1−β₁)·g
//!     v ← β₂·v + (1−β₂)·g²
//!     p ← p − lr·(m/(1−β₁ᵗ)) / (√v/√(1−β₂ᵗ) + ε)
//!
//! with each `mul` a round-to-even Q32.32 product and `div` a round-to-even quotient. The kernel
//! carries the same four quantities in real units (host divides by `2^32`, `2^32`, `2^32`, `2^64`)
//! and performs the same operation order in f32: moment one, moment two, the bias-corrected
//! denominator `√v/√(1−β₂ᵗ) + ε`, then the parameter step. `lr`, `β₁`, `1−β₁`, `β₂`, `1−β₂`, `ε`,
//! `β₁ᵗ` and `√(1−β₂ᵗ)` are module constants computed here in f64 from the oracle's exact Q32.32
//! words; no thread evaluates a power or a square root of a scale.
//!
//! One thread per element, grid `⌈n/256⌉`, LocalSize 256.

use super::super::*;

/// Q32.32 unit: the word `1 << 32` denotes `1.0` (`train.rs` `ONE`). The host divides `p`, `g`,
/// `m` by it and `v` by its square `2^64`, the Q64.64 unit.
const ONE_Q32: f64 = 4_294_967_296.0;
/// `round(0.9·2^32)` (`train.rs` `BETA1_Q32`).
const BETA1_Q32: f64 = 3_865_470_566.0;
/// `round(0.1·2^32)` (`train.rs` `ONE_MINUS_BETA1_Q32`).
const ONE_MINUS_BETA1_Q32: f64 = 429_496_730.0;
/// `round(0.999·2^32)` (`train.rs` `BETA2_Q32`).
const BETA2_Q32: f64 = 4_290_672_329.0;
/// `round(0.001·2^32)` (`train.rs` `ONE_MINUS_BETA2_Q32`).
const ONE_MINUS_BETA2_Q32: f64 = 4_294_967.0;
/// `round(1e-8·2^32) = 43` (`train.rs` `ADAM_EPS_Q32`).
const ADAM_EPS_Q32: f64 = 43.0;

/// The two bias-correction factors the kernel bakes at step `t`: `(β₁ᵗ, √(1−β₂ᵗ))`, each
/// computed once in f64 from the oracle's exact Q32.32 words and rounded to f32 once.
///
/// Complexity: O(1) time and space (one `powi` and one `sqrt`).
pub fn adam_bias_corrections(t: u32) -> (f32, f32) {
    assert!(t > 0, "adam_update: step must be non-zero");
    let b1 = BETA1_Q32 / ONE_Q32;
    let b2 = BETA2_Q32 / ONE_Q32;
    let b1t = b1.powi(t as i32);
    let b2t = b2.powi(t as i32);
    (b1t as f32, (1.0 - b2t).sqrt() as f32)
}

/// Where the two bias-correction factors come from: baked at module-build time for step `t`, or
/// read from the f32 storage buffer at `binding`, so one pipeline replays at every step.
#[derive(Clone, Copy)]
enum BiasSource {
    Baked(f32, f32),
    Buffer(u32),
}

/// The elementwise body shared by [`adam_update_module`] and [`adam_update_module_reused`]; the
/// two paths differ only in the provenance of the two factors, and every operand and operation
/// after them is identical.
fn adam_module(n: u32, lr: i64, bias: BiasSource) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(n > 0, "adam_update: shape must be non-zero");
    let b1 = BETA1_Q32 / ONE_Q32;
    let om1 = ONE_MINUS_BETA1_Q32 / ONE_Q32;
    let b2 = BETA2_Q32 / ONE_Q32;
    let om2 = ONE_MINUS_BETA2_Q32 / ONE_Q32;
    let eps = ADAM_EPS_Q32 / ONE_Q32;
    let step_lr = lr as f64 / ONE_Q32;
    let mut k = Kb::new();
    let bias_buf = match bias {
        BiasSource::Baked(..) => 0,
        BiasSource::Buffer(binding) => k.ssbo_f(binding),
    };
    let p = k.ssbo_f(0);
    let g = k.ssbo_f(1);
    let m = k.ssbo_f(2);
    let v = k.ssbo_f(3);
    let gid = k.gid_comp(0);
    let cn = k.cu(n);
    let inb = k.ult(gid, cn);
    k.iff(inb, |mut k| {
        let (f_lr, f_b1, f_om1) = (k.cf(step_lr as f32), k.cf(b1 as f32), k.cf(om1 as f32));
        let (f_b2, f_om2) = (k.cf(b2 as f32), k.cf(om2 as f32));
        let f_eps = k.cf(eps as f32);
        let (f_b1t, f_rbc2) = match bias {
            BiasSource::Baked(b1t, rbc2) => (k.cf(b1t), k.cf(rbc2)),
            BiasSource::Buffer(..) => {
                let z = k.cu(0);
                let one = k.cu(1);
                (k.load_f(bias_buf, z), k.load_f(bias_buf, one))
            }
        };
        let f_one = k.cf(1.0);
        let p0 = k.load_f(p, gid);
        let g0 = k.load_f(g, gid);
        let m0 = k.load_f(m, gid);
        let v0 = k.load_f(v, gid);
        // m ← β₁·m + (1−β₁)·g, the oracle's two rounded products then the sum.
        let m1 = k.fmul(f_b1, m0);
        let m2 = k.fmul(f_om1, g0);
        let mn = k.fadd(m1, m2);
        // v ← β₂·v + (1−β₂)·g², same order.
        let g2 = k.fmul(g0, g0);
        let v1 = k.fmul(f_b2, v0);
        let v2 = k.fmul(f_om2, g2);
        let vn = k.fadd(v1, v2);
        // m̂ = m/(1−β₁ᵗ); √v̂ = √v/√(1−β₂ᵗ), the division after the root as in the oracle.
        let bc1 = k.fsub(f_one, f_b1t);
        let mhat = k.fdiv(mn, bc1);
        let rootv = k.ext1(GLSL_SQRT, vn);
        let root = k.fdiv(rootv, f_rbc2);
        let denom = k.fadd(root, f_eps);
        let num = k.fmul(f_lr, mhat);
        let step = k.fdiv(num, denom);
        let pn = k.fsub(p0, step);
        k.store_f(p, gid, pn);
        k.store_f(m, gid, mn);
        k.store_f(v, gid, vn);
        k
    });
    k.finish(WG)
}

/// **Adam elementwise update (f32)** — one thread per parameter entry. Bindings: 0 = `p`
/// (f32, `n`), 1 = `g` (f32, `n`), 2 = `m` (f32, `n`), 3 = `v` (f32, `n`); all four in real units.
/// Reads `p`, `g`, `m`, `v` and writes `p`, `m`, `v`: `7·n` words = `28·n` bytes, `O(1)` per
/// thread. Grid `⌈n/256⌉`, LocalSize 256.
///
/// [`adam_module`] with the step's bias corrections baked in: the module for one step ordinal
/// `t = 1, 2, …`. Use [`adam_update_module_reused`] to build the pipeline once for every step.
pub fn adam_update_module(n: u32, lr: i64, t: u32) -> Vec<u32> {
    assert!(t > 0, "adam_update: step must be non-zero");
    let (b1t, rbc2) = adam_bias_corrections(t);
    adam_module(n, lr, BiasSource::Baked(b1t, rbc2))
}

/// [`adam_update_module`] with the bias corrections read from the f32 storage buffer at binding 4
/// (`[β₁ᵗ, √(1−β₂ᵗ)]`). One pipeline replays at every step; the factors carry the bits
/// [`adam_bias_corrections`] produces, and the arithmetic, operand order and grid are unchanged.
pub fn adam_update_module_reused(n: u32, lr: i64) -> Vec<u32> {
    adam_module(n, lr, BiasSource::Buffer(4))
}

/// [`adam_update_module`] as bytes.
pub fn adam_update_bytes(n: u32, lr: i64, t: u32) -> Vec<u8> {
    bytes(adam_update_module(n, lr, t))
}

/// [`adam_update_module_reused`] as bytes.
pub fn adam_update_bytes_reused(n: u32, lr: i64) -> Vec<u8> {
    bytes(adam_update_module_reused(n, lr))
}

/// Host wrapper: build and record the Adam elementwise pass. `p`, `g`, `m`, `v` are device
/// buffers of `n` f32 in real units; the pass updates `p`, `m`, `v` in place. Grid
/// `(⌈n/256⌉, 1, 1)`, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// `n` f32.
#[cfg(feature = "gpu")]
pub unsafe fn adam_update_dispatch(
    g: &crate::gpu::decode::GpuCtx,
    p: crate::gpu::vulkan::Handle,
    grad: crate::gpu::vulkan::Handle,
    m: crate::gpu::vulkan::Handle,
    v: crate::gpu::vulkan::Handle,
    n: u32,
    lr: i64,
    t: u32,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &adam_update_bytes(n, lr, t),
        &[p, grad, m, v],
        n.div_ceil(256),
        1,
        "adam_update",
    )
}

/// Host wrapper for [`adam_update_module_reused`]: `bias` is a device f32 buffer holding
/// `[β₁ᵗ, √(1−β₂ᵗ)]` for the current step, and is binding 4 after `p`, `grad`, `m`, `v`. Built
/// once per tensor; replay it at every step after writing that step's two words.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// `n` f32, and `bias` at least two.
#[cfg(feature = "gpu")]
pub unsafe fn adam_update_dispatch_reused(
    g: &crate::gpu::decode::GpuCtx,
    p: crate::gpu::vulkan::Handle,
    grad: crate::gpu::vulkan::Handle,
    m: crate::gpu::vulkan::Handle,
    v: crate::gpu::vulkan::Handle,
    bias: crate::gpu::vulkan::Handle,
    n: u32,
    lr: i64,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &adam_update_bytes_reused(n, lr),
        &[p, grad, m, v, bias],
        n.div_ceil(256),
        1,
        "adam_update",
    )
}
