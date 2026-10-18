//! IEEE binary32 (f32) carrier for Gated `DeltaNet` state recurrence and causal Conv+SiLU.
//!
//! # Contract
//! Evaluates the per-head affine recurrence and causal convolution in IEEE 754 binary32 arithmetic
//! on [`F32`] (a binary32 bit pattern, `crate::engine::num::fp`):
//! - State matrix $S_t \in \mathbb{R}^{dk \times w}$, initialized to $+0.0\text{f32}$.
//! - Recurrence: $S_t = \text{decay} \cdot S_{t-1} + k_t d_t^T$, where $d_t = \beta_t (v_t - S_{t-1}^T k_t)$.
//! - Readout: $o_t = S_t^T q_t = S_{t-1}^T q_t + d_t (k_t^T q_t)$, written directly into caller slice `o: &mut [F32]`.
//! - Causal convolution: $x_j = \sum_{t=0}^{\text{kernel}-1} w_t \cdot \text{hist}_t$ evaluated in binary32.
//! - Activation: $\text{SiLU}(x) = \frac{x}{1 + \exp(-x)}$ evaluated in binary32 using a vectorisable
//!   Cephes-style range reduction and minimax polynomial approximation with maximum relative error $< 2 \times 10^{-7}$.
//! - Rounding mode: IEEE 754 round-to-nearest-even (`roundTiesToEven`).
//! - Lane summation order: fixed column-major order with row index $i \in [0, dk)$ ascending
//!   for dot products and state reductions.
//! - Conversions:
//!   - Inputs $q̃, k̂, v, \beta, \alpha, w$ arrive as Q32.32 signed fixed point (`i64`) and convert to [`F32`]
//!     via $x / 2^{32}$ (`q32_to_f32`).
//!   - Readout $o$ converts to Q32.32 `i64` via round-half-away of $o \cdot 2^{32}$ (`f32_readout_to_q32`),
//!     failing with `StepError::Overflow` on non-finite values (NaN, $\pm\infty$) or when the rounded value
//!     exceeds the signed 64-bit integer range $[-2^{63}, 2^{63}-1]$. No silent clamping is performed.
//! - Allocation: Zero per-step heap allocation on the hot path. Scratch and output buffers are allocated
//!   once per task.
//!
//! # Proved Error Bound
//! This carrier relies on the formal bound proved in `proof/corpus/Nous/Seq/DeltaCarrier.lean`:
//! `delta_carrier_error_uniform` establishes that if the per-step column state perturbation is
//! bounded by $\varepsilon$, the accumulated state perturbation is bounded by $\frac{\varepsilon}{1 - \alpha}$,
//! and `readout_error_le` bounds the readout perturbation by $\|q\| \frac{\varepsilon}{1 - \alpha}$.
//!
//! # Trust Boundary
//! - Hardware IEEE 754-2008 conformance of CPU FMA (fused multiply-add) and AVX2 vector execution units.
//! - The scalar twins compute on [`F32`] and reach the hardware only through `fp::prim`. The AVX2
//!   `*_inner` kernels use native `__m256` intrinsics and are not extracted; their equality with the
//!   scalar twins is tested (`f32_avx2_equals_scalar`, `exp_and_silu_avx2_equals_scalar`).

#![allow(
    clippy::doc_markdown,
    reason = "mathematical formulas and LaTeX notation in doc comments"
)]

use crate::engine::num::fp::F32;
use crate::engine::ops::norm::NormError;
use crate::engine::step::StepError;

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports, reason = "standard SIMD intrinsic imports")]
use std::arch::x86_64::*;

/// The native `f32` with the bits of `x`, for the SIMD intrinsics (which take `f32`).
#[cfg(target_arch = "x86_64")]
#[inline]
fn nat(x: F32) -> f32 {
    f32::from_bits(x.to_bits())
}

/// Scalar reference implementation of the Gated `DeltaNet` f32 step.
///
/// State `s` is `dk × w` row-major (`row i = key index`, `col j = value index`).
/// Summation order is fixed to `i` ascending per column.
/// Output readout is written into caller-provided slice `o`.
///
/// # Complexity
/// - Time: $2 \cdot dk \cdot w + O(dk + w)$ scalar floating-point operations.
/// - Space: $O(1)$ stack allocations (bounded `w <= 128`). State `s` and output `o` updated in place.
#[allow(
    clippy::many_single_char_names,
    reason = "s, q, k, v, d, o are the recurrence symbols"
)]
pub(crate) fn delta_step_f32_scalar(
    s: &mut [F32],
    o: &mut [F32],
    q: &[F32],
    k: &[F32],
    v: &[F32],
    decay: F32,
    beta: F32,
) {
    let (dk, w) = (k.len(), v.len());
    assert_eq!(q.len(), dk, "q length must match dk");
    assert_eq!(s.len(), dk * w, "state length must equal dk * w");
    assert_eq!(o.len(), w, "o length must match w");
    assert!(w <= 128, "w must be <= 128");

    // 1. Scalar k^T q dot product
    let mut k_dot_q = F32::ZERO;
    let mut i = 0;
    while i < dk {
        k_dot_q = k[i].fma(q[i], k_dot_q);
        i += 1;
    }

    // 2. Fused read pass: scale state, accumulate m = S^T k and S^T q
    let mut acc_m = [F32::ZERO; 128];
    let mut acc_o = [F32::ZERO; 128];

    let mut i = 0;
    while i < dk {
        let ki = k[i];
        let qi = q[i];
        let row_offset = i * w;
        let mut j = 0;
        while j < w {
            let idx = row_offset + j;
            let scaled_s = decay * s[idx];
            s[idx] = scaled_s;
            acc_m[j] = scaled_s.fma(ki, acc_m[j]);
            acc_o[j] = scaled_s.fma(qi, acc_o[j]);
            j += 1;
        }
        i += 1;
    }

    // 3. Delta and rank-1 readout: d = beta * (v - m), o = S^T q + d * (k^T q)
    let mut d = [F32::ZERO; 128];
    let mut j = 0;
    while j < w {
        let dj = beta * (v[j] - acc_m[j]);
        d[j] = dj;
        o[j] = dj.fma(k_dot_q, acc_o[j]);
        j += 1;
    }

    // 4. Write pass: S = S + k d^T
    let mut i = 0;
    while i < dk {
        let ki = k[i];
        let row_offset = i * w;
        let mut j = 0;
        while j < w {
            let idx = row_offset + j;
            s[idx] = ki.fma(d[j], s[idx]);
            j += 1;
        }
        i += 1;
    }
}

/// AVX2+FMA SIMD implementation of the Gated `DeltaNet` f32 step.
///
/// Processes 8 value columns per 256-bit vector.
/// Uses fused read pass (decay folded into read) and single write pass.
/// Writes readout directly into caller slice `o`.
/// Matches `delta_step_f32_scalar` bit for bit.
///
/// # Complexity
/// - Time: $2 \cdot dk \cdot (w / 8)$ vector FMAs + $O(dk + w)$ vector/scalar operations.
/// - Space: $O(1)$ stack/register allocations; zero heap allocation.
///
/// # Safety
/// Caller must ensure AVX2 and FMA target features are available on the host CPU,
/// and that `s`, `o`, `q`, `k`, `v` meet length constraints (`w % 8 == 0`, `w <= 128`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::many_single_char_names,
    clippy::cast_ptr_alignment,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "recurrence symbols; unaligned SIMD loads/stores; SIMD vector names"
)]
pub(crate) unsafe fn delta_step_f32_avx2_inner(
    s: &mut [F32],
    o: &mut [F32],
    q: &[F32],
    k: &[F32],
    v: &[F32],
    decay: F32,
    beta: F32,
) {
    let (dk, w) = (k.len(), v.len());
    let num_chunks = w / 8;

    // 1. k^T q dot product
    let mut k_dot_q = F32::ZERO;
    for i in 0..dk {
        k_dot_q = k[i].fma(q[i], k_dot_q);
    }
    let v_kq = _mm256_set1_ps(nat(k_dot_q));

    // 2. Fused read pass: S <- decay * S, accumulate acc_m and acc_o
    let mut acc_m = [_mm256_setzero_ps(); 16];
    let mut acc_o = [_mm256_setzero_ps(); 16];
    let v_decay = _mm256_set1_ps(nat(decay));

    for i in 0..dk {
        let ki = k[i];
        let qi = q[i];
        let v_ki = _mm256_set1_ps(nat(ki));
        let v_qi = _mm256_set1_ps(nat(qi));
        let row_ptr = s.as_mut_ptr().cast::<f32>().add(i * w);

        for c in 0..num_chunks {
            let ptr = row_ptr.add(c * 8);
            let s_val = _mm256_loadu_ps(ptr);
            let scaled_s = _mm256_mul_ps(v_decay, s_val);
            _mm256_storeu_ps(ptr, scaled_s);
            acc_m[c] = _mm256_fmadd_ps(scaled_s, v_ki, acc_m[c]);
            acc_o[c] = _mm256_fmadd_ps(scaled_s, v_qi, acc_o[c]);
        }
    }

    // 3. Compute delta d and readout o
    let v_beta = _mm256_set1_ps(nat(beta));
    let mut d_vec = [_mm256_setzero_ps(); 16];

    for c in 0..num_chunks {
        let v_val = _mm256_loadu_ps(v.as_ptr().cast::<f32>().add(c * 8));
        let diff = _mm256_sub_ps(v_val, acc_m[c]);
        let dj = _mm256_mul_ps(v_beta, diff);
        d_vec[c] = dj;
        let oj = _mm256_fmadd_ps(dj, v_kq, acc_o[c]);
        _mm256_storeu_ps(o.as_mut_ptr().cast::<f32>().add(c * 8), oj);
    }

    // 4. Write pass: S <- S + k d^T
    for i in 0..dk {
        let ki = k[i];
        let v_ki = _mm256_set1_ps(nat(ki));
        let row_ptr = s.as_mut_ptr().cast::<f32>().add(i * w);

        for c in 0..num_chunks {
            let ptr = row_ptr.add(c * 8);
            let s_val = _mm256_loadu_ps(ptr);
            let s_new = _mm256_fmadd_ps(v_ki, d_vec[c], s_val);
            _mm256_storeu_ps(ptr, s_new);
        }
    }
}

/// Dispatches `delta_step_f32` to AVX2+FMA when supported and `w % 8 == 0`,
/// falling back to the scalar implementation.
///
/// # Complexity
/// - Time: $O(dk \cdot w)$ floating-point operations.
/// - Space: zero heap allocations; writes directly into caller slice `o`.
#[allow(
    clippy::many_single_char_names,
    reason = "recurrence variables: s (state), o (output), q (query), k (key), v (value)"
)]
pub(crate) fn delta_step_f32_avx2(
    s: &mut [F32],
    o: &mut [F32],
    q: &[F32],
    k: &[F32],
    v: &[F32],
    decay: F32,
    beta: F32,
) {
    #[cfg(target_arch = "x86_64")]
    if v.len().is_multiple_of(8) && v.len() <= 128 && crate::engine::num::fp::prim::cpu_avx2_fma() {
        return unsafe { delta_step_f32_avx2_inner(s, o, q, k, v, decay, beta) };
    }
    delta_step_f32_scalar(s, o, q, k, v, decay, beta);
}

// ---------------------------------------------------------------------------
// Vectorisable exp and SiLU
// ---------------------------------------------------------------------------

/// Bits of the literals named in `bits_equal_original_literals`; the extraction cannot translate a
/// float literal, so each constant is `F32::from_bits` of the literal's pattern.
const LOG2E: F32 = F32::from_bits(0x3fb8_aa3b);
/// `355 / 512`, exactly representable in binary32.
const LN2_C1: F32 = F32::from_bits(0x3f31_8000);
const LN2_C2: F32 = F32::from_bits(0xb95e_8083);

const EXP_P0: F32 = F32::from_bits(0x3950_6967);
const EXP_P1: F32 = F32::from_bits(0x3ab7_43ce);
const EXP_P2: F32 = F32::from_bits(0x3c08_8908);
const EXP_P3: F32 = F32::from_bits(0x3d2a_a9c1);
const EXP_P4: F32 = F32::from_bits(0x3e2a_aaaa);
const EXP_P5: F32 = F32::from_bits(0x3f00_0000);

/// `exp(x) = 0` strictly below this.
const EXP_UNDERFLOW: F32 = F32::from_bits(0xc2b0_0000);
/// `exp(x) = inf` strictly above this.
const EXP_OVERFLOW: F32 = F32::from_bits(0x42b1_7218);

/// Fast vectorisable approximation of `exp(x)` in IEEE binary32.
///
/// Uses Cephes-style range reduction:
/// $x = n \ln(2) + r$, where $n = \text{round}(x \log_2(e))$ and $r \in [-\frac{\ln 2}{2}, \frac{\ln 2}{2}]$.
/// Polynomial approximation of degree 5 on $r$:
/// $P(r) = 1 + r + r^2 (p_5 + p_4 r + p_3 r^2 + p_2 r^3 + p_1 r^4 + p_0 r^5)$.
/// Max relative error over $[-88.0, 88.0]$ is $< 2 \times 10^{-7}$ (< 2 ULP).
#[allow(
    clippy::many_single_char_names,
    reason = "mathematical symbols and float exponent reconstruction"
)]
#[inline]
pub(crate) fn exp_f32_scalar(x: F32) -> F32 {
    if x.is_nan() {
        return F32::NAN;
    }
    if x.lt(EXP_UNDERFLOW) {
        return F32::ZERO;
    }
    if x.gt(EXP_OVERFLOW) {
        return F32::INFINITY;
    }

    let z = x * LOG2E;
    let n_flt = z.round_ties_even();
    let n_int = n_flt.to_i32_sat();

    let r = (x - n_flt * LN2_C1) - n_flt * LN2_C2;
    let r2 = r * r;

    let mut p = EXP_P0;
    p = p * r + EXP_P1;
    p = p * r + EXP_P2;
    p = p * r + EXP_P3;
    p = p * r + EXP_P4;
    p = p * r + EXP_P5;

    let y = (F32::ONE + r) + (r2 * p);

    let n1 = n_int >> 1;
    let n2 = n_int - n1;
    let s1 = F32::from_bits((n1 + 127).cast_unsigned() << 23);
    let s2 = F32::from_bits((n2 + 127).cast_unsigned() << 23);

    (y * s1) * s2
}

/// AVX2 implementation of `exp_f32`. Matches `exp_f32_scalar` bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    dead_code,
    reason = "vectorised AVX2 exp kernel and parity test target"
)]
#[inline]
pub(crate) unsafe fn exp_f32_avx2(vx: __m256) -> __m256 {
    let vz = _mm256_mul_ps(vx, _mm256_set1_ps(nat(LOG2E)));
    let vn_flt = _mm256_round_ps(vz, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let vn_int = _mm256_cvtps_epi32(vn_flt);

    let n_c1 = _mm256_mul_ps(vn_flt, _mm256_set1_ps(nat(LN2_C1)));
    let n_c2 = _mm256_mul_ps(vn_flt, _mm256_set1_ps(nat(LN2_C2)));
    let r = _mm256_sub_ps(_mm256_sub_ps(vx, n_c1), n_c2);
    let r2 = _mm256_mul_ps(r, r);

    let mut p = _mm256_set1_ps(nat(EXP_P0));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(nat(EXP_P1)));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(nat(EXP_P2)));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(nat(EXP_P3)));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(nat(EXP_P4)));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(nat(EXP_P5)));

    let y = _mm256_add_ps(_mm256_add_ps(_mm256_set1_ps(1.0), r), _mm256_mul_ps(r2, p));

    let vn1 = _mm256_srai_epi32(vn_int, 1);
    let vn2 = _mm256_sub_epi32(vn_int, vn1);
    let bias = _mm256_set1_epi32(127);
    let vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(vn1, bias), 23));
    let vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(vn2, bias), 23));

    let res = _mm256_mul_ps(_mm256_mul_ps(y, vs1), vs2);

    let zero = _mm256_setzero_ps();
    let inf = _mm256_set1_ps(f32::INFINITY);
    let mask_underflow = _mm256_cmp_ps(vx, _mm256_set1_ps(nat(EXP_UNDERFLOW)), _CMP_LT_OQ);
    let mask_overflow = _mm256_cmp_ps(vx, _mm256_set1_ps(nat(EXP_OVERFLOW)), _CMP_GT_OQ);
    let mut out = _mm256_blendv_ps(res, zero, mask_underflow);
    out = _mm256_blendv_ps(out, inf, mask_overflow);
    out
}

/// Scalar $\text{SiLU}(x) = \frac{x}{1 + \exp(-x)}$ in binary32.
#[inline]
pub(crate) fn silu_f32_scalar(x: F32) -> F32 {
    let exp_neg = exp_f32_scalar(-x);
    x / (F32::ONE + exp_neg)
}

/// AVX2 $\text{SiLU}(x) = \frac{x}{1 + \exp(-x)}$ in binary32.
/// Matches `silu_f32_scalar` bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    dead_code,
    reason = "vectorised AVX2 SiLU kernel and parity test target"
)]
#[inline]
pub(crate) unsafe fn silu_f32_avx2(vx: __m256) -> __m256 {
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), vx);
    let exp_neg = exp_f32_avx2(neg_x);
    let denom = _mm256_add_ps(_mm256_set1_ps(1.0), exp_neg);
    _mm256_div_ps(vx, denom)
}

/// Vectorisable `exp(x)` in IEEE binary32.
#[inline]
pub(crate) fn exp_f32(x: F32) -> F32 {
    exp_f32_scalar(x)
}

/// Scalar sigmoid $\sigma(x) = \frac{1}{1 + \exp(-x)}$ in IEEE binary32.
#[inline]
pub(crate) fn sigmoid_f32(x: F32) -> F32 {
    let exp_neg = exp_f32_scalar(-x);
    F32::ONE / (F32::ONE + exp_neg)
}

/// `softplus(x) = x` strictly above this.
const SOFTPLUS_HI: F32 = F32::from_bits(0x41a0_0000);
/// `softplus(x) = 0` strictly below this.
const SOFTPLUS_LO: F32 = F32::from_bits(0xc1f0_0000);

/// Scalar softplus $\text{softplus}(x) = \ln(1 + \exp(x))$ in IEEE binary32.
/// Evaluated with numerical stability across $(-\infty, +\infty)$.
#[inline]
pub(crate) fn softplus_f32(x: F32) -> F32 {
    if !x.is_finite() {
        return F32::NAN;
    }
    if x.gt(SOFTPLUS_HI) {
        x
    } else if x.lt(SOFTPLUS_LO) {
        F32::ZERO
    } else if x.ge(F32::ZERO) {
        x + ln_f32(F32::ONE + exp_f32_scalar(-x))
    } else {
        ln_f32(F32::ONE + exp_f32_scalar(x))
    }
}

/// $\sqrt{1/2}$, the Cephes `logf` reduction threshold.
const LN_SQRTHF: F32 = F32::from_bits(0x3f35_04f3);
/// $2^{23}$, lifts a subnormal argument into the normal range.
const LN_TWO23: F32 = F32::from_bits(0x4b00_0000);
/// Cephes `logf` polynomial coefficients $P_0 \dots P_8$ (Moshier, *Cephes Math Library*).
const LN_P0: F32 = F32::from_bits(0x3d90_21bb);
const LN_P1: F32 = F32::from_bits(0xbdeb_d1b8);
const LN_P2: F32 = F32::from_bits(0x3def_251a);
const LN_P3: F32 = F32::from_bits(0xbdfe_5d4f);
const LN_P4: F32 = F32::from_bits(0x3e11_e9bf);
const LN_P5: F32 = F32::from_bits(0xbe2a_ae50);
const LN_P6: F32 = F32::from_bits(0x3e4c_ceac);
const LN_P7: F32 = F32::from_bits(0xbe7f_fffc);
const LN_P8: F32 = F32::from_bits(0x3eaa_aaaa);

/// Natural logarithm in IEEE binary32: Cephes `logf` (Moshier), built from [`F32`] operations and
/// integer exponent/mantissa extraction, with no libm and no float cast.
///
/// $x = m \cdot 2^e$ with $m \in [\tfrac12, 1)$; if $m < \sqrt{1/2}$ then $e \mathrel{-}= 1$ and
/// $f = 2m - 1$, else $f = m - 1$, so $f \in [\sqrt{1/2} - 1, \sqrt{2} - 1)$ and
/// $\ln x = e \ln 2 + f - f^2/2 + f^3 P(f)$ with $P$ of degree 8. $\ln 2$ is split as
/// `LN2_C1 + LN2_C2` (the `exp` range-reduction constants) so $e \cdot$`LN2_C1` is exact.
/// Returns $-\infty$ at $\pm 0$, NaN below $0$, `x` at $+\infty$ and NaN.
///
/// Complexity: O(1) time and space (about 25 binary32 operations; no loop).
#[allow(
    clippy::many_single_char_names,
    reason = "x, m, e, f, z, y, r are the Cephes logf symbols"
)]
#[inline]
pub(crate) fn ln_f32(x: F32) -> F32 {
    if x.le(F32::ZERO) {
        return if x.eq_ieee(F32::ZERO) {
            F32::NEG_INFINITY
        } else {
            F32::NAN
        };
    }
    if !x.is_finite() {
        return x;
    }
    let mut bits = x.to_bits();
    let mut e: i32 = 0;
    if bits < 0x0080_0000 {
        // Subnormal: scale by 2^23 and take it back out of the exponent.
        bits = (x * LN_TWO23).to_bits();
        e = -23;
    }
    e += ((bits >> 23) & 0xff).cast_signed() - 126;
    let m = F32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let f = if m.lt(LN_SQRTHF) {
        e -= 1;
        (m + m) - F32::ONE
    } else {
        m - F32::ONE
    };
    let z = f * f;
    let mut y = LN_P0;
    y = y * f + LN_P1;
    y = y * f + LN_P2;
    y = y * f + LN_P3;
    y = y * f + LN_P4;
    y = y * f + LN_P5;
    y = y * f + LN_P6;
    y = y * f + LN_P7;
    y = y * f + LN_P8;
    y = (y * f) * z;
    // |e| <= 150, so the conversion is exact.
    let fe = F32::from_i64(i64::from(e));
    y += fe * LN2_C2;
    y -= F32::HALF * z;
    let r = f + y;
    r + fe * LN2_C1
}

// ---------------------------------------------------------------------------
// Causal Depthwise Conv + SiLU (Kernel = 4)
// ---------------------------------------------------------------------------

/// Scalar reference for 4-tap depthwise convolution and SiLU on a single channel.
#[allow(
    dead_code,
    reason = "scalar twin specification for AVX2 conv_silu_tap4_avx2"
)]
#[inline]
pub(crate) fn conv_silu_tap4_scalar(taps: [F32; 4], hist: [F32; 4]) -> F32 {
    let mut acc = taps[0] * hist[0];
    acc = taps[1].fma(hist[1], acc);
    acc = taps[2].fma(hist[2], acc);
    acc = taps[3].fma(hist[3], acc);
    silu_f32_scalar(acc)
}

/// AVX2 implementation of 4-tap depthwise convolution and SiLU over 8 channels simultaneously.
///
/// Unrolls all 4 taps into FMA vector operations.
/// Matches `conv_silu_tap4_scalar` bit for bit across all 8 vector lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::too_many_arguments,
    reason = "4 taps and 4 history registers are the unrolled conv state"
)]
#[inline]
pub(crate) unsafe fn conv_silu_tap4_avx2(
    tap0: __m256,
    tap1: __m256,
    tap2: __m256,
    tap3: __m256,
    hist0: __m256,
    hist1: __m256,
    hist2: __m256,
    hist3: __m256,
) -> __m256 {
    let mut acc = _mm256_mul_ps(tap0, hist0);
    acc = _mm256_fmadd_ps(tap1, hist1, acc);
    acc = _mm256_fmadd_ps(tap2, hist2, acc);
    acc = _mm256_fmadd_ps(tap3, hist3, acc);
    silu_f32_avx2(acc)
}

// ---------------------------------------------------------------------------
// Stage 0: L2 Normalization in f32
// ---------------------------------------------------------------------------

/// Sum of squares of `x` in IEEE binary32 using 8-way interleaved accumulation.
/// Scalar reference matching `l2_norm_sum_sq_avx2_inner` bit for bit.
#[inline]
pub(crate) fn l2_norm_sum_sq_scalar(x: &[F32]) -> F32 {
    let mut acc = [F32::ZERO; 8];
    let chunks = x.len() / 8;
    let mut c = 0;
    while c < chunks {
        let base = c * 8;
        let mut lane = 0;
        while lane < 8 {
            acc[lane] = x[base + lane].fma(x[base + lane], acc[lane]);
            lane += 1;
        }
        c += 1;
    }
    let mut rem_sum = F32::ZERO;
    let mut r = chunks * 8;
    while r < x.len() {
        rem_sum = x[r].fma(x[r], rem_sum);
        r += 1;
    }
    let sum8 = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
    sum8 + rem_sum
}

/// AVX2 implementation of sum of squares using 8-way FMA accumulation.
/// Matches `l2_norm_sum_sq_scalar` bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn l2_norm_sum_sq_avx2_inner(x: &[F32]) -> F32 {
    let mut v_acc = _mm256_setzero_ps();
    let chunks = x.len() / 8;
    let ptr = x.as_ptr().cast::<f32>();
    for c in 0..chunks {
        let vx = _mm256_loadu_ps(ptr.add(c * 8));
        v_acc = _mm256_fmadd_ps(vx, vx, v_acc);
    }
    let mut acc = [F32::ZERO; 8];
    _mm256_storeu_ps(acc.as_mut_ptr().cast::<f32>(), v_acc);
    let mut rem_sum = F32::ZERO;
    for &val in &x[chunks * 8..] {
        rem_sum = val.fma(val, rem_sum);
    }
    let sum8 = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
    sum8 + rem_sum
}

#[allow(dead_code, reason = "simd sum of squares dispatch helper")]
#[inline]
pub(crate) fn l2_norm_sum_sq_avx2(x: &[F32]) -> F32 {
    #[cfg(target_arch = "x86_64")]
    if crate::engine::num::fp::prim::cpu_avx2_fma() {
        return unsafe { l2_norm_sum_sq_avx2_inner(x) };
    }
    l2_norm_sum_sq_scalar(x)
}

/// In-place L2 normalization of `x` with `scale_multiplier` in IEEE binary32.
/// Evaluates `x[i] = x[i] * scale_multiplier / max(sqrt(sum(x^2)), eps)`.
pub(crate) fn l2_normalize_f32_in_place_scalar(
    x: &mut [F32],
    eps: F32,
    scale_multiplier: F32,
) -> Result<(), StepError> {
    let sum_sq = l2_norm_sum_sq_scalar(x);
    if !sum_sq.is_finite() {
        return Err(StepError::Overflow);
    }
    let norm = sum_sq.sqrt().max(eps);
    if !norm.is_finite() || norm.le(F32::ZERO) {
        return Err(StepError::BadConfig(String::from("l2 norm eps")));
    }
    let scale = (F32::ONE / norm) * scale_multiplier;
    if !scale.is_finite() {
        return Err(StepError::Overflow);
    }
    let mut i = 0;
    while i < x.len() {
        x[i] *= scale;
        i += 1;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn l2_normalize_f32_in_place_avx2_inner(
    x: &mut [F32],
    eps: F32,
    scale_multiplier: F32,
) -> Result<(), StepError> {
    let sum_sq = l2_norm_sum_sq_avx2_inner(x);
    if !sum_sq.is_finite() {
        return Err(StepError::Overflow);
    }
    let norm = sum_sq.sqrt().max(eps);
    if !norm.is_finite() || norm.le(F32::ZERO) {
        return Err(StepError::BadConfig(String::from("l2 norm eps")));
    }
    let scale = (F32::ONE / norm) * scale_multiplier;
    if !scale.is_finite() {
        return Err(StepError::Overflow);
    }
    let chunks = x.len() / 8;
    let v_scale = _mm256_set1_ps(nat(scale));
    let ptr = x.as_mut_ptr().cast::<f32>();
    for c in 0..chunks {
        let chunk_ptr = ptr.add(c * 8);
        let vx = _mm256_loadu_ps(chunk_ptr);
        _mm256_storeu_ps(chunk_ptr, _mm256_mul_ps(vx, v_scale));
    }
    for val in &mut x[chunks * 8..] {
        *val *= scale;
    }
    Ok(())
}

/// Dispatches in-place L2 normalization to AVX2+FMA when supported.
/// Matches scalar twin bit for bit.
pub(crate) fn l2_normalize_f32_in_place(
    x: &mut [F32],
    eps: F32,
    scale_multiplier: F32,
) -> Result<(), StepError> {
    #[cfg(target_arch = "x86_64")]
    if crate::engine::num::fp::prim::cpu_avx2_fma() {
        return unsafe { l2_normalize_f32_in_place_avx2_inner(x, eps, scale_multiplier) };
    }
    l2_normalize_f32_in_place_scalar(x, eps, scale_multiplier)
}

// ---------------------------------------------------------------------------
// Stage 2: Per-Head Gated RMS Norm in f32
// ---------------------------------------------------------------------------

/// Scalar reference for Stage 2 per-head RMS norm, ssm_norm scaling, and SiLU(z) gating.
/// Writes the final Q32.32 `i64` result into `dst`.
pub(crate) fn stage2_gated_norm_f32_scalar(
    o: &[F32],
    z: &[i64],
    ssm_norm: &[F32],
    eps: F32,
    dst: &mut [i64],
) -> Result<(), StepError> {
    let head_v = o.len();
    assert_eq!(z.len(), head_v, "z length mismatch");
    assert_eq!(ssm_norm.len(), head_v, "ssm_norm length mismatch");
    assert_eq!(dst.len(), head_v, "dst length mismatch");

    let sum_sq = l2_norm_sum_sq_scalar(o);
    if !sum_sq.is_finite() {
        return Err(StepError::Overflow);
    }
    // head_v is a head dimension (typically <= 256), far below i64::MAX.
    #[allow(clippy::cast_possible_wrap, reason = "head_v <= 256 in practice")]
    let mean = sum_sq / F32::from_i64(head_v as i64);
    let denom = mean + eps;
    if !denom.is_finite() || denom.le(F32::ZERO) {
        return Err(StepError::Norm(NormError::OutOfRange));
    }
    let scale = F32::ONE / denom.sqrt();
    if !scale.is_finite() {
        return Err(StepError::Overflow);
    }

    // One element at a time, ascending: the word written and the first refusal are those of the
    // 8-lane blocking this replaces, since every step is elementwise.
    let mut j = 0;
    while j < head_v {
        let gate = silu_f32_scalar(q32_to_f32(z[j]));
        let normed = o[j] * scale;
        let gained = normed * ssm_norm[j];
        let out_f = gained * gate;
        dst[j] = f32_readout_to_q32(out_f)?;
        j += 1;
    }
    Ok(())
}

/// AVX2 implementation of Stage 2 per-head RMS norm, ssm_norm scaling, and SiLU(z) gating.
/// Matches `stage2_gated_norm_f32_scalar` bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "mathematical symbols for gating and normalization"
)]
pub(crate) unsafe fn stage2_gated_norm_f32_avx2_inner(
    o: &[F32],
    z: &[i64],
    ssm_norm: &[F32],
    eps: F32,
    dst: &mut [i64],
) -> Result<(), StepError> {
    let head_v = o.len();
    assert_eq!(z.len(), head_v, "z length mismatch");
    assert_eq!(ssm_norm.len(), head_v, "ssm_norm length mismatch");
    assert_eq!(dst.len(), head_v, "dst length mismatch");

    let sum_sq = l2_norm_sum_sq_avx2_inner(o);
    if !sum_sq.is_finite() {
        return Err(StepError::Overflow);
    }
    #[allow(clippy::cast_possible_wrap, reason = "head_v <= 256 in practice")]
    let mean = sum_sq / F32::from_i64(head_v as i64);
    let denom = mean + eps;
    if !denom.is_finite() || denom.le(F32::ZERO) {
        return Err(StepError::Norm(NormError::OutOfRange));
    }
    let scale = F32::ONE / denom.sqrt();
    if !scale.is_finite() {
        return Err(StepError::Overflow);
    }

    let chunks = head_v / 8;
    let v_scale = _mm256_set1_ps(nat(scale));
    let o_ptr = o.as_ptr().cast::<f32>();
    let norm_ptr = ssm_norm.as_ptr().cast::<f32>();

    for c in 0..chunks {
        let base = c * 8;
        let mut z_buf = [F32::ZERO; 8];
        for lane in 0..8 {
            z_buf[lane] = q32_to_f32(z[base + lane]);
        }
        let vz = _mm256_loadu_ps(z_buf.as_ptr().cast::<f32>());
        let vgate = silu_f32_avx2(vz);
        let vo = _mm256_loadu_ps(o_ptr.add(base));
        let vnorm = _mm256_loadu_ps(norm_ptr.add(base));
        let vnormed = _mm256_mul_ps(vo, v_scale);
        let vgained = _mm256_mul_ps(vnormed, vnorm);
        let vout = _mm256_mul_ps(vgained, vgate);
        let mut out_buf = [F32::ZERO; 8];
        _mm256_storeu_ps(out_buf.as_mut_ptr().cast::<f32>(), vout);
        for lane in 0..8 {
            dst[base + lane] = f32_readout_to_q32(out_buf[lane])?;
        }
    }
    let rem_base = chunks * 8;
    for j in rem_base..head_v {
        let zj = q32_to_f32(z[j]);
        let gate = silu_f32_scalar(zj);
        let normed = o[j] * scale;
        let gained = normed * ssm_norm[j];
        let out_f = gained * gate;
        dst[j] = f32_readout_to_q32(out_f)?;
    }
    Ok(())
}

/// Dispatches Stage 2 per-head gated norm to AVX2+FMA when supported.
/// Matches scalar twin bit for bit.
pub(crate) fn stage2_gated_norm_f32(
    o: &[F32],
    z: &[i64],
    ssm_norm: &[F32],
    eps: F32,
    dst: &mut [i64],
) -> Result<(), StepError> {
    #[cfg(target_arch = "x86_64")]
    if crate::engine::num::fp::prim::cpu_avx2_fma() {
        return unsafe { stage2_gated_norm_f32_avx2_inner(o, z, ssm_norm, eps, dst) };
    }
    stage2_gated_norm_f32_scalar(o, z, ssm_norm, eps, dst)
}

// ---------------------------------------------------------------------------
// Conversion functions
// ---------------------------------------------------------------------------

/// Convert an f32 readout value `o` back to Q32.32 fixed point `i64`.
///
/// Computes `round_half_away(o · 2^32)` by integer decode (`F32::to_q32`).
/// Range check: non-finite (NaN, $\pm\infty$) or values exceeding the signed 64-bit integer
/// range $[-2^{63}, 2^{63}-1]$ return `Err(StepError::Overflow)`. No silent clamping is performed.
#[inline]
pub(crate) fn f32_readout_to_q32(o: F32) -> Result<i64, StepError> {
    if !o.is_finite() {
        return Err(StepError::Overflow);
    }
    match o.to_q32() {
        Some(w) => Ok(w),
        None => Err(StepError::Overflow),
    }
}

/// [`f32_readout_to_q32`] over 8 lanes, written to `dst[0..8]`.
///
/// Lanes with `|o · 2^32| < 2^51` take the vector path: `⌊|s| + 0.5⌋` (the same f64 sum
/// `f32_readout_to_q32` truncates, so the same word), moved to `i64` by the `1.5 · 2^52`
/// mantissa offset, sign restored. A block holding any other lane (large, infinite or NaN) is
/// read out lane by lane with the scalar function, so refusals are identical.
///
/// # Safety
///
/// AVX2 is available and `dst` is valid for 8 `i64` writes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "unaligned 256-bit stores (`_mm256_storeu_si256`) take any address"
)]
#[inline]
pub(crate) unsafe fn readout8_avx2(vy: __m256, dst: *mut i64) -> Result<(), StepError> {
    let scale = _mm256_set1_pd(4_294_967_296.0);
    let sign = _mm256_set1_pd(-0.0);
    let limit = _mm256_set1_pd(2_251_799_813_685_248.0);
    let half = _mm256_set1_pd(0.5);
    let magic = _mm256_set1_pd(6_755_399_441_055_744.0);
    let magic_bits = _mm256_castpd_si256(magic);
    let zero = _mm256_setzero_si256();
    let lo = _mm256_mul_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(vy)), scale);
    let hi = _mm256_mul_pd(_mm256_cvtps_pd(_mm256_extractf128_ps::<1>(vy)), scale);
    let abs_lo = _mm256_andnot_pd(sign, lo);
    let abs_hi = _mm256_andnot_pd(sign, hi);
    let ok = _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LT_OQ>(abs_lo, limit))
        & _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_LT_OQ>(abs_hi, limit));
    if ok != 0xF {
        let mut b = [F32::ZERO; 8];
        _mm256_storeu_ps(b.as_mut_ptr().cast::<f32>(), vy);
        let mut res = Ok(());
        let mut i = 0;
        while i < 8 && res.is_ok() {
            match f32_readout_to_q32(b[i]) {
                Ok(w) => *dst.add(i) = w,
                Err(e) => res = Err(e),
            }
            i += 1;
        }
        return res;
    }
    let word = |s: __m256d, a: __m256d| -> __m256i {
        let f = _mm256_round_pd::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(_mm256_add_pd(
            a, half,
        ));
        let m = _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(f, magic)), magic_bits);
        let neg = _mm256_cmpgt_epi64(zero, _mm256_castpd_si256(s));
        _mm256_sub_epi64(_mm256_xor_si256(m, neg), neg)
    };
    _mm256_storeu_si256(dst.cast::<__m256i>(), word(lo, abs_lo));
    _mm256_storeu_si256(dst.add(4).cast::<__m256i>(), word(hi, abs_hi));
    Ok(())
}

/// Convert a Q32.32 fixed point `i64` to [`F32`] ($x / 2^{32}$).
#[inline]
pub(crate) fn q32_to_f32(x: i64) -> F32 {
    F32::from_q32(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "f32_carrier_tests.rs"]
mod tests;
