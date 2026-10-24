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

/// The three constants an Adam pipeline bakes: `β₁`, `β₂` and `ε` in real units, from which the
/// kernel's five module constants `β₁`, `1−β₁`, `β₂`, `1−β₂`, `ε` follow.
///
/// `ORACLE` is the fold finisher's set, the exact Q32.32 words over `2^32` (`train.rs`
/// `BETA1_Q32 = round(0.9·2^32)`, `BETA2_Q32 = round(0.999·2^32)`, `ADAM_EPS_Q32 = round(1e-8·2^32)`),
/// and reproduces every constant this module baked before the type existed bit for bit.
///
/// `OPTIMI` is the NanoQuant reference's optimiser, read from source: betas `(0.9, 0.99)`, eps
/// `1e-6`, weight decay `0` (`SamsungLabs/NanoQuant` @ `a9e0a43`, `src/nanoquant/optimi/adamw.py:53-56`,
/// `src/nanoquant/core/compress_block.py:71,175`).
///
/// # The debiased-β form is bias-corrected Adam
///
/// The reference never divides by `1−βᵗ`. It folds the correction into the decay factor
/// (`optimi/utils.py:30-35`, `debiased_beta`):
///
/// ```text
/// β̂ₜ = (βᵗ − β)/(βᵗ − 1) = β(1−βᵗ⁻¹)/(1−βᵗ)
/// mₜ ← mₜ₋₁ + (1−β̂ₜ)(gₜ − mₜ₋₁) = β̂ₜ·mₜ₋₁ + (1−β̂ₜ)·gₜ        (adam.py:377)
/// vₜ ← β̂₂ₜ·vₜ₋₁ + (1−β̂₂ₜ)·gₜ²                            (adam.py:378)
/// pₜ ← pₜ₋₁ − lr·mₜ/(√vₜ + ε)                             (adam.py:394)
/// ```
///
/// Substituting `m̂ₜ₋₁ = mₜ₋₁/(1−βᵗ⁻¹)` into the first line,
/// `β̂ₜ·m̂ₜ₋₁ = [β(1−βᵗ⁻¹)/(1−βᵗ)]·mₜ₋₁/(1−βᵗ⁻¹) = β·mₜ₋₁/(1−βᵗ)` and
/// `1−β̂ₜ = (1−βᵗ−β+βᵗ)/(1−βᵗ) = (1−β)/(1−βᵗ)`, so
/// `m̂ₜ = [β·mₜ₋₁ + (1−β)·gₜ]/(1−βᵗ) = mₜ/(1−βᵗ)`: the recursion *is* bias-corrected Adam with the
/// corrected moment, identically, and the same substitution gives `v̂ₜ = vₜ/(1−β₂ᵗ)`.
///
/// Where it does **not** reduce exactly: the reference's denominator is `√v̂ + ε`; this kernel's is
/// `√v/√(1−β₂ᵗ) + ε` (`adam_module`), i.e. `(√v + ε·√(1−β₂ᵗ))/√(1−β₂ᵗ)`, so its step is the
/// reference's with `ε` replaced by `ε·√(1−β₂ᵗ)`. The two agree to first order and differ by
/// `ε(1−√(1−β₂ᵗ))/√v̂`, which at `ε = 1e-6`, `√(1−β₂) = 0.1` (step 1) and `√v̂ ≳ 1e-3` is below
/// `1e-3` relative on the first step and shrinks as `β₂ᵗ → 0`; the whole difference vanishes with
/// `ε`. Kahan summation (`adam.py:384-390`) applies only to bf16 parameters and these are f32, so
/// it is not applicable here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamHyper {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl AdamHyper {
    /// The fold finisher's constants: `3_865_470_566`, `4_290_672_329` and `43` over `2^32`.
    pub const ORACLE: Self = Self {
        beta1: 3_865_470_566.0 / 4_294_967_296.0,
        beta2: 4_290_672_329.0 / 4_294_967_296.0,
        eps: 43.0 / 4_294_967_296.0,
    };
    /// The NanoQuant reference's optimiser (`optimi/adamw.py:53-56`: `betas=(0.9, 0.99)`, `eps=1e-6`).
    pub const OPTIMI: Self = Self {
        beta1: 0.9,
        beta2: 0.99,
        eps: 1e-6,
    };

    /// The two bias-correction factors the kernel bakes at step `t`: `(β₁ᵗ, √(1−β₂ᵗ))`, each
    /// computed once in f64 and rounded to f32 once.
    ///
    /// Complexity: O(1) time and space (two `powi` and one `sqrt`).
    #[must_use]
    pub fn bias_corrections(self, t: u32) -> (f32, f32) {
        assert!(t > 0, "adam_update: step must be non-zero");
        let b1t = self.beta1.powi(t as i32);
        let b2t = self.beta2.powi(t as i32);
        (b1t as f32, (1.0 - b2t).sqrt() as f32)
    }
}

/// The two bias-correction factors the kernel bakes at step `t`: `(β₁ᵗ, √(1−β₂ᵗ))` for the
/// oracle's constants.
///
/// Complexity: O(1) time and space (one `powi` and one `sqrt`).
pub fn adam_bias_corrections(t: u32) -> (f32, f32) {
    AdamHyper::ORACLE.bias_corrections(t)
}

/// The same two words with the learning-rate schedule folded in: `c = lr_t/lr_base` is the step's
/// multiplier, and the first word becomes `β₁ᵗ' = 1 − (1−β₁ᵗ)/c`, so the unchanged shader computes
/// `lr·m/(1−β₁ᵗ') = c·lr·m/(1−β₁ᵗ) = lr_t·m̂` and the buffer layout and every shader stay as they are.
///
/// The identity is exact in reals. In f32 the error is that of rounding `β₁ᵗ'`, whose deficit
/// `δ = 1−β₁ᵗ' = (1−β₁ᵗ)/c` carries absolute error `≤ 2⁻²⁵` (half an ulp of `1`) and therefore
/// relative error `≤ 2⁻²⁵/δ ≤ 2⁻²⁵/(1−β₁) = 3.0e-7` for `c ≤ 1`, since `1−β₁ᵗ ≥ 1−β₁ = 0.1` for every
/// `t`. The second word is [`AdamHyper::bias_corrections`]` .1` unchanged, and the kernel's `ε` is
/// not scaled by `c` (`adam_module` adds it after the division), which is the same
/// `ε·√(1−β₂ᵗ)` deviation documented on [`AdamHyper`].
///
/// Complexity: O(1) time and space (one `powi` and one `sqrt`).
#[must_use]
pub fn adam_scheduled_bias(h: AdamHyper, t: u32, c: f64) -> (f32, f32) {
    assert!(
        c > 0.0,
        "adam_update: the schedule multiplier must be positive"
    );
    let (b1t, rbc2) = h.bias_corrections(t);
    let deficit = (1.0 - f64::from(b1t)) / c;
    ((1.0 - deficit) as f32, rbc2)
}

/// The learning rates `torch.optim.lr_scheduler.CosineAnnealingLR` produces over `steps` optimiser
/// steps, the recursive `get_lr` written literally (torch 2.8.0, `torch/optim/lr_scheduler.py:1085-1111`),
/// not the closed form at `:1113-1120`.
///
/// The scheduler is stepped *after* each optimiser step (`compress_block.py:96-98`), so the rate at
/// optimiser step `t = 1..=steps` is the group's rate after `t−1` scheduler steps. Entry `t = 1` is
/// the constructor's initial step (`_initial_step`, `:143-146`, returns `group["lr"]` = `base`);
/// after that the recursion multiplies the decayed remainder `lr − eta_min` by
/// `(1+cos(π·l/T_max))/(1+cos(π·(l−1)/T_max))`, except at `l = T_max+1 (mod 2·T_max)`, where it
/// adds the first half-step back (`:1101-1108`). `l` is `last_epoch`, one per scheduler step.
///
/// Complexity: `O(steps)` time and space; one `cos` per term.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the step ordinals are the scheduler's own integer epoch, and a run has < 2^31 steps"
)]
pub fn cosine_lr(base: f64, eta_min: f64, t_max: usize, steps: usize) -> Vec<f64> {
    assert!(t_max > 0, "cosine_lr: T_max must be non-zero");
    let tw = t_max as i64;
    // The state the constructor leaves (`_initial_step`): `last_epoch = 0`, `_step_count = 1`.
    let (mut group, mut epoch, mut count) = (base, 0i64, 1u32);
    let mut out = Vec::with_capacity(steps);
    for step in 0..steps {
        out.push(group);
        if step + 1 == steps {
            break;
        }
        // One `step()`: `_step_count += 1`, `last_epoch += 1`, then `get_lr`.
        count = count.wrapping_add(1);
        epoch += 1;
        let tau = std::f64::consts::PI;
        group = if count == 1 && epoch > 0 {
            eta_min + (base - eta_min) * (1.0 + (epoch as f64 * tau / t_max as f64).cos()) / 2.0
        } else if (epoch - 1 - tw).rem_euclid(2 * tw) == 0 {
            group + (base - eta_min) * (1.0 - (tau / t_max as f64).cos()) / 2.0
        } else {
            let l = epoch as f64;
            let r = (1.0 + (tau * l / t_max as f64).cos())
                / (1.0 + (tau * (l - 1.0) / t_max as f64).cos());
            r * (group - eta_min) + eta_min
        };
    }
    out
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
fn adam_module(n: u32, lr: i64, h: AdamHyper, bias: BiasSource) -> Vec<u32> {
    const WG: u32 = 256;
    assert!(n > 0, "adam_update: shape must be non-zero");
    let (b1, om1) = (h.beta1, 1.0 - h.beta1);
    let (b2, om2) = (h.beta2, 1.0 - h.beta2);
    let eps = h.eps;
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
/// `t = 1, 2, …`, under `h`'s constants. Use [`adam_update_module_reused_h`] to build the pipeline
/// once for every step.
pub fn adam_update_module_h(n: u32, lr: i64, t: u32, h: AdamHyper) -> Vec<u32> {
    assert!(t > 0, "adam_update: step must be non-zero");
    let (b1t, rbc2) = h.bias_corrections(t);
    adam_module(n, lr, h, BiasSource::Baked(b1t, rbc2))
}

/// [`adam_update_module_h`] under the oracle's constants, the fold finisher's.
pub fn adam_update_module(n: u32, lr: i64, t: u32) -> Vec<u32> {
    adam_update_module_h(n, lr, t, AdamHyper::ORACLE)
}

/// [`adam_update_module`] with the bias corrections read from the f32 storage buffer at binding 4
/// (`[β₁ᵗ, √(1−β₂ᵗ)]`), under `h`'s constants. One pipeline replays at every step; the factors
/// carry the bits [`adam_scheduled_bias`] produces, and the arithmetic, operand order and grid are
/// unchanged.
pub fn adam_update_module_reused_h(n: u32, lr: i64, h: AdamHyper) -> Vec<u32> {
    adam_module(n, lr, h, BiasSource::Buffer(4))
}

/// [`adam_update_module_reused_h`] under the oracle's constants.
pub fn adam_update_module_reused(n: u32, lr: i64) -> Vec<u32> {
    adam_update_module_reused_h(n, lr, AdamHyper::ORACLE)
}

/// [`adam_update_module`] as bytes.
pub fn adam_update_bytes(n: u32, lr: i64, t: u32) -> Vec<u8> {
    bytes(adam_update_module(n, lr, t))
}

/// [`adam_update_module_h`] as bytes.
pub fn adam_update_bytes_h(n: u32, lr: i64, t: u32, h: AdamHyper) -> Vec<u8> {
    bytes(adam_update_module_h(n, lr, t, h))
}

/// [`adam_update_module_reused`] as bytes.
pub fn adam_update_bytes_reused(n: u32, lr: i64) -> Vec<u8> {
    bytes(adam_update_module_reused(n, lr))
}

/// [`adam_update_module_reused_h`] as bytes.
pub fn adam_update_bytes_reused_h(n: u32, lr: i64, h: AdamHyper) -> Vec<u8> {
    bytes(adam_update_module_reused_h(n, lr, h))
}

/// Host wrapper: build and record the Adam elementwise pass. `p`, `g`, `m`, `v` are device
/// buffers of `n` f32 in real units; the pass updates `p`, `m`, `v` in place. Grid
/// `(⌈n/256⌉, 1, 1)`, LocalSize 256.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// `n` f32.
#[cfg(feature = "gpu")]
pub unsafe fn adam_update_dispatch_h(
    g: &crate::gpu::decode::GpuCtx,
    p: crate::gpu::vulkan::Handle,
    grad: crate::gpu::vulkan::Handle,
    m: crate::gpu::vulkan::Handle,
    v: crate::gpu::vulkan::Handle,
    n: u32,
    lr: i64,
    t: u32,
    h: AdamHyper,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &adam_update_bytes_h(n, lr, t, h),
        &[p, grad, m, v],
        n.div_ceil(256),
        1,
        "adam_update",
    )
}

/// [`adam_update_dispatch`]: [`adam_update_dispatch_h`] under the oracle's constants.
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
    adam_update_dispatch_h(g, p, grad, m, v, n, lr, t, AdamHyper::ORACLE)
}

/// Host wrapper for [`adam_update_module_reused`]: `bias` is a device f32 buffer holding
/// `[β₁ᵗ, √(1−β₂ᵗ)]` for the current step, and is binding 4 after `p`, `grad`, `m`, `v`. Built
/// once per tensor; replay it at every step after writing that step's two words.
///
/// # Safety
/// Vulkan pipeline/descriptor creation over caller-owned handles; every buffer must hold at least
/// `n` f32, and `bias` at least two.
#[cfg(feature = "gpu")]
pub unsafe fn adam_update_dispatch_reused_h(
    g: &crate::gpu::decode::GpuCtx,
    p: crate::gpu::vulkan::Handle,
    grad: crate::gpu::vulkan::Handle,
    m: crate::gpu::vulkan::Handle,
    v: crate::gpu::vulkan::Handle,
    bias: crate::gpu::vulkan::Handle,
    n: u32,
    lr: i64,
    h: AdamHyper,
) -> crate::gpu::decode::Step {
    crate::gpu::decode::make_step_named(
        g,
        &adam_update_bytes_reused_h(n, lr, h),
        &[p, grad, m, v, bias],
        n.div_ceil(256),
        1,
        "adam_update",
    )
}

/// [`adam_update_dispatch_reused`]: [`adam_update_dispatch_reused_h`] under the oracle's
/// constants.
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
    adam_update_dispatch_reused_h(g, p, grad, m, v, bias, n, lr, AdamHyper::ORACLE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::decode::{rec_step, GpuCtx};
    use crate::gpu::vulkan::*;

    /// A fixed-seed LCG value in `[-1, 1)` (Knuth's MMIX constants).
    fn unit(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 1.0
            })
            .collect()
    }

    /// One host-visible, host-coherent f32 buffer and its mapped pointer.
    unsafe fn host_words(vk: &Vk, n: usize) -> (Handle, *mut f32) {
        let (b, m) = vk
            .buffer(
                n.max(1) as u64 * 4,
                BUFFER_USAGE_STORAGE | BUFFER_USAGE_TRANSFER_SRC | BUFFER_USAGE_TRANSFER_DST,
                MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT,
            )
            .expect("host-visible buffer");
        (b, vk.map(m).cast::<f32>())
    }

    /// The reference's optimiser step, written literally from `SamsungLabs/NanoQuant` @ `a9e0a43`,
    /// `src/nanoquant/optimi/adam.py:371-394`: the debiased betas `β̂ = (βᵗ−β)/(βᵗ−1)`
    /// (`optimi/utils.py:30-35`, `debiased_beta`), `m.lerp_(g, 1−β̂₁)`, `v = β̂₂·v + (1−β̂₂)·g²`,
    /// `p −= lr·m/(√v + ε)` (`adam.py:377-378,394`). Not bias-corrected Adam: the correction is in
    /// the decay factor, as the source has it.
    fn host_step(p: &mut [f64], m: &mut [f64], v: &mut [f64], g: &[f64], t: u32, lr: f64) {
        // The reference's own literals (`AdamW.__init__`, `optimi/adamw.py:53-56`), so a change to
        // `AdamHyper::OPTIMI` diverges from this host instead of moving both sides together.
        const B1: f64 = 0.9;
        const B2: f64 = 0.99;
        const EPS: f64 = 1e-6;
        let (b1t, b2t) = (B1.powi(t as i32), B2.powi(t as i32));
        let (b1, b2) = ((b1t - B1) / (b1t - 1.0), (b2t - B2) / (b2t - 1.0));
        for i in 0..p.len() {
            m[i] += (1.0 - b1) * (g[i] - m[i]);
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] -= lr * m[i] / (v[i].sqrt() + EPS);
        }
    }

    /// The scheduled bias words put the device `adam_update_dispatch_reused` on the reference's
    /// `CosineAnnealingLR` run: 256 steps at `cosine_lr(1e-4, 1e-8, 256, 256)` from seeded `p` and a
    /// seeded gradient sequence, against the f64 host replica of `optimi/adam.py:371-394`.
    #[test]
    fn adam_optimi_schedule_matches_host() {
        let Ok(vk) = (unsafe { Vk::load() }) else {
            println!("SKIP: no Vulkan device");
            return;
        };
        let (g, _) = GpuCtx::new(&vk).expect("gpu ctx");
        const N: usize = 64;
        const STEPS: usize = 256;
        const BASE: f64 = 1e-4;
        let p0: Vec<f64> = unit(N, 7).iter().map(|v| 0.5 * v).collect();
        let gs: Vec<Vec<f64>> = (0..STEPS)
            .map(|t| unit(N, 1000 + t as u64).iter().map(|v| 0.5 * v).collect())
            .collect();
        let lrs = cosine_lr(BASE, 1e-8, STEPS, STEPS);
        // The same three ordinals from torch 2.8.0 itself (`CosineAnnealingLR(o, T_max=256,
        // eta_min=1e-8)` on an `AdamW` group at `lr=1e-4`, stepped after each use,
        // `torch/optim/lr_scheduler.py:1085-1111`): 1.000000e-04, 5.000500e-05, 1.376453e-08.
        println!(
            "cosine lr: {:.6e} first, {:.6e} mid, {:.6e} last",
            lrs[0],
            lrs[STEPS / 2],
            lrs[STEPS - 1]
        );
        for (got, want) in [
            (lrs[0], 1.000_000e-4),
            (lrs[STEPS / 2], 5.000_500e-5),
            (lrs[STEPS - 1], 1.376_453e-8),
        ] {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs(),
                "cosine_lr {got:e} vs torch {want:e}"
            );
        }
        // The host run.
        let (mut hp, mut hm, mut hv) = (p0.clone(), vec![0f64; N], vec![0f64; N]);
        for (t, g_t) in gs.iter().enumerate() {
            host_step(&mut hp, &mut hm, &mut hv, g_t, t as u32 + 1, lrs[t]);
        }
        unsafe {
            let (pb, pp) = host_words(&vk, N);
            let (gb, gp) = host_words(&vk, N);
            let (mb, mp) = host_words(&vk, N);
            let (vb, vp) = host_words(&vk, N);
            let (bb, bp) = host_words(&vk, 2);
            for i in 0..N {
                *pp.add(i) = p0[i] as f32;
                *gp.add(i) = 0.0;
                *mp.add(i) = 0.0;
                *vp.add(i) = 0.0;
            }
            let step = adam_update_dispatch_reused_h(
                &g,
                pb,
                gb,
                mb,
                vb,
                bb,
                N as u32,
                q32_base(BASE),
                AdamHyper::OPTIMI,
            );
            let cb = vk.command_buffer();
            for (t, g_t) in gs.iter().enumerate() {
                for i in 0..N {
                    *gp.add(i) = g_t[i] as f32;
                }
                let c = lrs[t] / BASE;
                let (b1t, rbc2) = adam_scheduled_bias(AdamHyper::OPTIMI, t as u32 + 1, c);
                *bp = b1t;
                *bp.add(1) = rbc2;
                vk.begin(cb);
                rec_step(&vk, cb, &step);
                check((vk.end_cb)(cb), "vkEndCommandBuffer");
                vk.submit_wait(cb);
            }
            let dev = std::slice::from_raw_parts(pp, N);
            let (mut num, mut den) = (0f64, 1e-30f64);
            for i in 0..N {
                num = num.max((f64::from(dev[i]) - hp[i]).abs());
                den = den.max((hp[i] - p0[i]).abs());
            }
            println!(
                "adam schedule worst |p - p0| relative error {:.2e} over {N} entries, {STEPS} steps",
                num / den
            );
            assert!(num / den <= 1e-4, "relative error {:.2e}", num / den);
        }
    }

    /// A real learning rate as the Q32.32 word the kernel takes.
    fn q32_base(v: f64) -> i64 {
        (v * 4_294_967_296.0).round() as i64
    }
}
