//! IEEE binary32 (f32) carrier for Gated `DeltaNet` state recurrence and causal Conv+SiLU.
//!
//! # Contract
//! Evaluates the per-head affine recurrence and causal convolution in IEEE 754 binary32 arithmetic:
//! - State matrix $S_t \in \mathbb{R}^{dk \times w}$, initialized to $+0.0\text{f32}$.
//! - Recurrence: $S_t = \text{decay} \cdot S_{t-1} + k_t d_t^T$, where $d_t = \beta_t (v_t - S_{t-1}^T k_t)$.
//! - Readout: $o_t = S_t^T q_t = S_{t-1}^T q_t + d_t (k_t^T q_t)$, written directly into caller slice `o: &mut [f32]`.
//! - Causal convolution: $x_j = \sum_{t=0}^{\text{kernel}-1} w_t \cdot \text{hist}_t$ evaluated in binary32.
//! - Activation: $\text{SiLU}(x) = \frac{x}{1 + \exp(-x)}$ evaluated in binary32 using a vectorisable
//!   Cephes-style range reduction and minimax polynomial approximation with maximum relative error $< 2 \times 10^{-7}$.
//! - Rounding mode: IEEE 754 round-to-nearest-even (`roundTiesToEven`).
//! - Lane summation order: fixed column-major order with row index $i \in [0, dk)$ ascending
//!   for dot products and state reductions.
//! - Conversions:
//!   - Inputs $q̃, k̂, v, \beta, \alpha, w$ arrive as Q32.32 signed fixed point (`i64`) and convert to `f32`
//!     via $x / 2^{32}$ (`q32_to_f32`).
//!   - Readout $o$ converts to Q32.32 `i64` via round-to-nearest of $o \cdot 2^{32}$ (`f32_readout_to_q32`),
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
//! - Unlike the Q32.32 ring and `Cell<i64>` word kernel, floating-point vector routines are not extracted
//!   into Lean by Aeneas/Charon; correctness relies on hardware IEEE conformance and empirical equivalence
//!   tested between scalar and AVX2 implementations (`f32_avx2_equals_scalar`, `exp_and_silu_avx2_equals_scalar`).

#![allow(
    clippy::doc_markdown,
    reason = "mathematical formulas and LaTeX notation in doc comments"
)]

use crate::engine::norm::NormError;
use crate::engine::step::StepError;

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports, reason = "standard SIMD intrinsic imports")]
use std::arch::x86_64::*;

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
    clippy::needless_range_loop,
    reason = "s, q, k, v, d, o are the recurrence symbols"
)]
pub(crate) fn delta_step_f32_scalar(
    s: &mut [f32],
    o: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
) {
    let (dk, w) = (k.len(), v.len());
    assert_eq!(q.len(), dk, "q length must match dk");
    assert_eq!(s.len(), dk * w, "state length must equal dk * w");
    assert_eq!(o.len(), w, "o length must match w");
    assert!(w <= 128, "w must be <= 128");

    // 1. Scalar k^T q dot product
    let mut k_dot_q = 0.0f32;
    for i in 0..dk {
        k_dot_q = k[i].mul_add(q[i], k_dot_q);
    }

    // 2. Fused read pass: scale state, accumulate m = S^T k and S^T q
    let mut acc_m = [0.0f32; 128];
    let mut acc_o = [0.0f32; 128];

    for i in 0..dk {
        let ki = k[i];
        let qi = q[i];
        let row_offset = i * w;
        for j in 0..w {
            let idx = row_offset + j;
            let scaled_s = decay * s[idx];
            s[idx] = scaled_s;
            acc_m[j] = scaled_s.mul_add(ki, acc_m[j]);
            acc_o[j] = scaled_s.mul_add(qi, acc_o[j]);
        }
    }

    // 3. Delta and rank-1 readout: d = beta * (v - m), o = S^T q + d * (k^T q)
    let mut d = [0.0f32; 128];
    for j in 0..w {
        let dj = beta * (v[j] - acc_m[j]);
        d[j] = dj;
        o[j] = dj.mul_add(k_dot_q, acc_o[j]);
    }

    // 4. Write pass: S = S + k d^T
    for i in 0..dk {
        let ki = k[i];
        let row_offset = i * w;
        for j in 0..w {
            let idx = row_offset + j;
            s[idx] = ki.mul_add(d[j], s[idx]);
        }
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
    s: &mut [f32],
    o: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
) {
    let (dk, w) = (k.len(), v.len());
    let num_chunks = w / 8;

    // 1. k^T q dot product
    let mut k_dot_q = 0.0f32;
    for i in 0..dk {
        k_dot_q = k[i].mul_add(q[i], k_dot_q);
    }
    let v_kq = _mm256_set1_ps(k_dot_q);

    // 2. Fused read pass: S <- decay * S, accumulate acc_m and acc_o
    let mut acc_m = [_mm256_setzero_ps(); 16];
    let mut acc_o = [_mm256_setzero_ps(); 16];
    let v_decay = _mm256_set1_ps(decay);

    for i in 0..dk {
        let ki = k[i];
        let qi = q[i];
        let v_ki = _mm256_set1_ps(ki);
        let v_qi = _mm256_set1_ps(qi);
        let row_ptr = s.as_mut_ptr().add(i * w);

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
    let v_beta = _mm256_set1_ps(beta);
    let mut d_vec = [_mm256_setzero_ps(); 16];

    for c in 0..num_chunks {
        let v_val = _mm256_loadu_ps(v.as_ptr().add(c * 8));
        let diff = _mm256_sub_ps(v_val, acc_m[c]);
        let dj = _mm256_mul_ps(v_beta, diff);
        d_vec[c] = dj;
        let oj = _mm256_fmadd_ps(dj, v_kq, acc_o[c]);
        _mm256_storeu_ps(o.as_mut_ptr().add(c * 8), oj);
    }

    // 4. Write pass: S <- S + k d^T
    for i in 0..dk {
        let ki = k[i];
        let v_ki = _mm256_set1_ps(ki);
        let row_ptr = s.as_mut_ptr().add(i * w);

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
    s: &mut [f32],
    o: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if v.len().is_multiple_of(8)
        && v.len() <= 128
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        return unsafe { delta_step_f32_avx2_inner(s, o, q, k, v, decay, beta) };
    }
    delta_step_f32_scalar(s, o, q, k, v, decay, beta);
}

// ---------------------------------------------------------------------------
// Vectorisable exp and SiLU
// ---------------------------------------------------------------------------

#[allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    reason = "exact binary32 constants for Cephes range reduction"
)]
const LOG2E: f32 = 1.442_695_04;
#[allow(
    clippy::excessive_precision,
    reason = "355 / 512, exactly representable in binary32"
)]
const LN2_C1: f32 = 0.693_359_375;
const LN2_C2: f32 = -2.121_944_4e-4;

const EXP_P0: f32 = 1.987_569_1e-4;
const EXP_P1: f32 = 1.398_199_9e-3;
const EXP_P2: f32 = 8.333_452e-3;
const EXP_P3: f32 = 4.166_579_6e-2;
const EXP_P4: f32 = 1.666_666_6e-1;
const EXP_P5: f32 = 5.0e-1;

/// Fast vectorisable approximation of `exp(x)` in IEEE binary32.
///
/// Uses Cephes-style range reduction:
/// $x = n \ln(2) + r$, where $n = \text{round}(x \log_2(e))$ and $r \in [-\frac{\ln 2}{2}, \frac{\ln 2}{2}]$.
/// Polynomial approximation of degree 5 on $r$:
/// $P(r) = 1 + r + r^2 (p_5 + p_4 r + p_3 r^2 + p_2 r^3 + p_1 r^4 + p_0 r^5)$.
/// Max relative error over $[-88.0, 88.0]$ is $< 2 \times 10^{-7}$ (< 2 ULP).
#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "mathematical symbols and float exponent reconstruction"
)]
#[inline]
pub(crate) fn exp_f32_scalar(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x < -88.0 {
        return 0.0;
    }
    if x > 88.722_84 {
        return f32::INFINITY;
    }

    let z = x * LOG2E;
    let n_flt = z.round_ties_even();
    let n_int = n_flt as i32;

    let r = (x - n_flt * LN2_C1) - n_flt * LN2_C2;
    let r2 = r * r;

    let mut p = EXP_P0;
    p = p * r + EXP_P1;
    p = p * r + EXP_P2;
    p = p * r + EXP_P3;
    p = p * r + EXP_P4;
    p = p * r + EXP_P5;

    let y = (1.0 + r) + (r2 * p);

    let n1 = n_int >> 1;
    let n2 = n_int - n1;
    let s1 = f32::from_bits((n1 + 127).cast_unsigned() << 23);
    let s2 = f32::from_bits((n2 + 127).cast_unsigned() << 23);

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
    let vz = _mm256_mul_ps(vx, _mm256_set1_ps(LOG2E));
    let vn_flt = _mm256_round_ps(vz, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let vn_int = _mm256_cvtps_epi32(vn_flt);

    let n_c1 = _mm256_mul_ps(vn_flt, _mm256_set1_ps(LN2_C1));
    let n_c2 = _mm256_mul_ps(vn_flt, _mm256_set1_ps(LN2_C2));
    let r = _mm256_sub_ps(_mm256_sub_ps(vx, n_c1), n_c2);
    let r2 = _mm256_mul_ps(r, r);

    let mut p = _mm256_set1_ps(EXP_P0);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_P1));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_P2));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_P3));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_P4));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(EXP_P5));

    let y = _mm256_add_ps(_mm256_add_ps(_mm256_set1_ps(1.0), r), _mm256_mul_ps(r2, p));

    let vn1 = _mm256_srai_epi32(vn_int, 1);
    let vn2 = _mm256_sub_epi32(vn_int, vn1);
    let bias = _mm256_set1_epi32(127);
    let vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(vn1, bias), 23));
    let vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(vn2, bias), 23));

    let res = _mm256_mul_ps(_mm256_mul_ps(y, vs1), vs2);

    let zero = _mm256_setzero_ps();
    let inf = _mm256_set1_ps(f32::INFINITY);
    let mask_underflow = _mm256_cmp_ps(vx, _mm256_set1_ps(-88.0), _CMP_LT_OQ);
    let mask_overflow = _mm256_cmp_ps(vx, _mm256_set1_ps(88.722_84), _CMP_GT_OQ);
    let mut out = _mm256_blendv_ps(res, zero, mask_underflow);
    out = _mm256_blendv_ps(out, inf, mask_overflow);
    out
}

/// Scalar $\text{SiLU}(x) = \frac{x}{1 + \exp(-x)}$ in binary32.
#[inline]
pub(crate) fn silu_f32_scalar(x: f32) -> f32 {
    let exp_neg = exp_f32_scalar(-x);
    x / (1.0 + exp_neg)
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
pub(crate) fn exp_f32(x: f32) -> f32 {
    exp_f32_scalar(x)
}

/// Scalar sigmoid $\sigma(x) = \frac{1}{1 + \exp(-x)}$ in IEEE binary32.
#[inline]
pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    let exp_neg = exp_f32_scalar(-x);
    1.0f32 / (1.0f32 + exp_neg)
}

/// Scalar softplus $\text{softplus}(x) = \ln(1 + \exp(x))$ in IEEE binary32.
/// Evaluated with numerical stability across $(-\infty, +\infty)$.
#[inline]
pub(crate) fn softplus_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    if x > 20.0 {
        x
    } else if x < -30.0 {
        0.0
    } else if x >= 0.0 {
        x + (1.0f32 + exp_f32_scalar(-x)).ln()
    } else {
        (1.0f32 + exp_f32_scalar(x)).ln()
    }
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
pub(crate) fn conv_silu_tap4_scalar(taps: [f32; 4], hist: [f32; 4]) -> f32 {
    let mut acc = taps[0] * hist[0];
    acc = taps[1].mul_add(hist[1], acc);
    acc = taps[2].mul_add(hist[2], acc);
    acc = taps[3].mul_add(hist[3], acc);
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
pub(crate) fn l2_norm_sum_sq_scalar(x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = x.len() / 8;
    for c in 0..chunks {
        let base = c * 8;
        for lane in 0..8 {
            acc[lane] = x[base + lane].mul_add(x[base + lane], acc[lane]);
        }
    }
    let mut rem_sum = 0.0f32;
    for &val in &x[chunks * 8..] {
        rem_sum = val.mul_add(val, rem_sum);
    }
    let sum8 = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
    sum8 + rem_sum
}

/// AVX2 implementation of sum of squares using 8-way FMA accumulation.
/// Matches `l2_norm_sum_sq_scalar` bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn l2_norm_sum_sq_avx2_inner(x: &[f32]) -> f32 {
    let mut v_acc = _mm256_setzero_ps();
    let chunks = x.len() / 8;
    let ptr = x.as_ptr();
    for c in 0..chunks {
        let vx = _mm256_loadu_ps(ptr.add(c * 8));
        v_acc = _mm256_fmadd_ps(vx, vx, v_acc);
    }
    let mut acc = [0.0f32; 8];
    _mm256_storeu_ps(acc.as_mut_ptr(), v_acc);
    let mut rem_sum = 0.0f32;
    for &val in &x[chunks * 8..] {
        rem_sum = val.mul_add(val, rem_sum);
    }
    let sum8 = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
    sum8 + rem_sum
}

#[allow(dead_code, reason = "simd sum of squares dispatch helper")]
#[inline]
pub(crate) fn l2_norm_sum_sq_avx2(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { l2_norm_sum_sq_avx2_inner(x) };
    }
    l2_norm_sum_sq_scalar(x)
}

/// In-place L2 normalization of `x` with `scale_multiplier` in IEEE binary32.
/// Evaluates `x[i] = x[i] * scale_multiplier / max(sqrt(sum(x^2)), eps)`.
pub(crate) fn l2_normalize_f32_in_place_scalar(
    x: &mut [f32],
    eps: f32,
    scale_multiplier: f32,
) -> Result<(), StepError> {
    let sum_sq = l2_norm_sum_sq_scalar(x);
    if !sum_sq.is_finite() {
        return Err(StepError::Overflow);
    }
    let norm = sum_sq.sqrt().max(eps);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(StepError::BadConfig("l2 norm eps"));
    }
    let scale = (1.0f32 / norm) * scale_multiplier;
    if !scale.is_finite() {
        return Err(StepError::Overflow);
    }
    for val in x.iter_mut() {
        *val *= scale;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn l2_normalize_f32_in_place_avx2_inner(
    x: &mut [f32],
    eps: f32,
    scale_multiplier: f32,
) -> Result<(), StepError> {
    let sum_sq = l2_norm_sum_sq_avx2_inner(x);
    if !sum_sq.is_finite() {
        return Err(StepError::Overflow);
    }
    let norm = sum_sq.sqrt().max(eps);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(StepError::BadConfig("l2 norm eps"));
    }
    let scale = (1.0f32 / norm) * scale_multiplier;
    if !scale.is_finite() {
        return Err(StepError::Overflow);
    }
    let chunks = x.len() / 8;
    let v_scale = _mm256_set1_ps(scale);
    let ptr = x.as_mut_ptr();
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
    x: &mut [f32],
    eps: f32,
    scale_multiplier: f32,
) -> Result<(), StepError> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { l2_normalize_f32_in_place_avx2_inner(x, eps, scale_multiplier) };
    }
    l2_normalize_f32_in_place_scalar(x, eps, scale_multiplier)
}

// ---------------------------------------------------------------------------
// Stage 2: Per-Head Gated RMS Norm in f32
// ---------------------------------------------------------------------------

/// Scalar reference for Stage 2 per-head RMS norm, ssm_norm scaling, and SiLU(z) gating.
/// Writes the final Q32.32 `i64` result into `dst`.
#[allow(
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "mathematical symbols for gating and normalization"
)]
pub(crate) fn stage2_gated_norm_f32_scalar(
    o: &[f32],
    z: &[i64],
    ssm_norm: &[f32],
    eps: f32,
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
    #[allow(
        clippy::cast_precision_loss,
        reason = "head_v is small (typically <= 256)"
    )]
    let mean = sum_sq / (head_v as f32);
    let denom = mean + eps;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(StepError::Norm(NormError::OutOfRange));
    }
    let scale = 1.0f32 / denom.sqrt();
    if !scale.is_finite() {
        return Err(StepError::Overflow);
    }

    let chunks = head_v / 8;
    for c in 0..chunks {
        let base = c * 8;
        let mut z_buf = [0.0f32; 8];
        for lane in 0..8 {
            z_buf[lane] = q32_to_f32(z[base + lane]);
        }
        for lane in 0..8 {
            let idx = base + lane;
            let gate = silu_f32_scalar(z_buf[lane]);
            let normed = o[idx] * scale;
            let gained = normed * ssm_norm[idx];
            let out_f = gained * gate;
            dst[idx] = f32_readout_to_q32(out_f)?;
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
    o: &[f32],
    z: &[i64],
    ssm_norm: &[f32],
    eps: f32,
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
    #[allow(
        clippy::cast_precision_loss,
        reason = "head_v is small (typically <= 256)"
    )]
    let mean = sum_sq / (head_v as f32);
    let denom = mean + eps;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(StepError::Norm(NormError::OutOfRange));
    }
    let scale = 1.0f32 / denom.sqrt();
    if !scale.is_finite() {
        return Err(StepError::Overflow);
    }

    let chunks = head_v / 8;
    let v_scale = _mm256_set1_ps(scale);
    let o_ptr = o.as_ptr();
    let norm_ptr = ssm_norm.as_ptr();

    for c in 0..chunks {
        let base = c * 8;
        let mut z_buf = [0.0f32; 8];
        for lane in 0..8 {
            z_buf[lane] = q32_to_f32(z[base + lane]);
        }
        let vz = _mm256_loadu_ps(z_buf.as_ptr());
        let vgate = silu_f32_avx2(vz);
        let vo = _mm256_loadu_ps(o_ptr.add(base));
        let vnorm = _mm256_loadu_ps(norm_ptr.add(base));
        let vnormed = _mm256_mul_ps(vo, v_scale);
        let vgained = _mm256_mul_ps(vnormed, vnorm);
        let vout = _mm256_mul_ps(vgained, vgate);
        let mut out_buf = [0.0f32; 8];
        _mm256_storeu_ps(out_buf.as_mut_ptr(), vout);
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
    o: &[f32],
    z: &[i64],
    ssm_norm: &[f32],
    eps: f32,
    dst: &mut [i64],
) -> Result<(), StepError> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { stage2_gated_norm_f32_avx2_inner(o, z, ssm_norm, eps, dst) };
    }
    stage2_gated_norm_f32_scalar(o, z, ssm_norm, eps, dst)
}

// ---------------------------------------------------------------------------
// Conversion functions
// ---------------------------------------------------------------------------

/// Convert an f32 readout value `o` back to Q32.32 fixed point `i64`.
///
/// Computes `round(o · 2^32)` using round-to-nearest.
/// Range check: non-finite (NaN, $\pm\infty$) or values exceeding the signed 64-bit integer
/// range $[-2^{63}, 2^{63}-1]$ return `Err(StepError::Overflow)`. No silent clamping is performed.
#[inline]
pub(crate) fn f32_readout_to_q32(o: f32) -> Result<i64, StepError> {
    if !o.is_finite() {
        return Err(StepError::Overflow);
    }
    let scaled = f64::from(o) * 4_294_967_296.0;
    let rounded = scaled.round();
    if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&rounded) {
        return Err(StepError::Overflow);
    }
    #[allow(clippy::cast_possible_truncation, reason = "bounds checked above")]
    Ok(rounded as i64)
}

#[cfg(test)]
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    reason = "conversion to Q32.32 fixed point for test harnesses"
)]
fn f32_to_q32(x: f32) -> i64 {
    (f64::from(x) * ((1i64 << 32) as f64)).round() as i64
}

/// Convert a Q32.32 fixed point `i64` to `f32` ($x / 2^{32}$).
#[inline]
#[allow(
    clippy::cast_precision_loss,
    reason = "conversion from Q32.32 fixed point to IEEE binary32"
)]
pub(crate) fn q32_to_f32(x: i64) -> f32 {
    (x as f32) / 4_294_967_296.0_f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    reason = "test harnesses use standard mathematical names, timing loops and literals"
)]
mod tests {
    use super::*;
    use crate::engine::mixer::{delta_head_step_fast, delta_head_step_ref};

    struct SimpleRng(u64);
    impl SimpleRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }
        fn next_f32(&mut self) -> f32 {
            ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
        }
        fn next_f32_range(&mut self, low: f32, high: f32) -> f32 {
            low + self.next_f32() * (high - low)
        }
    }

    #[test]
    fn f32_avx2_equals_scalar() {
        let mut rng = SimpleRng::new(0x1234_5678_9ABC_DEF0);
        let dk = 128;
        let w = 32;

        let mut s_scalar = vec![0.0f32; dk * w];
        let mut s_avx2 = vec![0.0f32; dk * w];
        let mut o_scalar = vec![0.0f32; w];
        let mut o_avx2 = vec![0.0f32; w];

        for step in 0..1200 {
            let mut k = Vec::with_capacity(dk);
            let mut q = Vec::with_capacity(dk);
            let mut k_sq = 0.0f32;
            let mut q_sq = 0.0f32;
            for _ in 0..dk {
                let ki = rng.next_f32_range(-1.0, 1.0);
                let qi = rng.next_f32_range(-1.0, 1.0);
                k_sq += ki * ki;
                q_sq += qi * qi;
                k.push(ki);
                q.push(qi);
            }
            let k_norm = k_sq.sqrt();
            let q_norm = q_sq.sqrt();
            let q_scale = 1.0f32 / (128.0f32).sqrt();
            for ki in &mut k {
                *ki /= k_norm;
            }
            for qi in &mut q {
                *qi = (*qi / q_norm) * q_scale;
            }

            let mut v = Vec::with_capacity(w);
            for _ in 0..w {
                v.push(rng.next_f32_range(-4.0, 4.0));
            }

            let decay = rng.next_f32_range(0.5, 0.999);
            let beta = rng.next_f32_range(0.0, 1.0);

            delta_step_f32_scalar(&mut s_scalar, &mut o_scalar, &q, &k, &v, decay, beta);
            delta_step_f32_avx2(&mut s_avx2, &mut o_avx2, &q, &k, &v, decay, beta);

            for j in 0..w {
                assert_eq!(
                    o_scalar[j].to_bits(),
                    o_avx2[j].to_bits(),
                    "readout mismatch at step {step}, col {j}: scalar={:e} ({:08x}) vs avx2={:e} ({:08x})",
                    o_scalar[j],
                    o_scalar[j].to_bits(),
                    o_avx2[j],
                    o_avx2[j].to_bits(),
                );
            }

            for idx in 0..s_scalar.len() {
                assert_eq!(
                    s_scalar[idx].to_bits(),
                    s_avx2[idx].to_bits(),
                    "state mismatch at step {step}, idx {idx}: scalar={:e} vs avx2={:e}",
                    s_scalar[idx],
                    s_avx2[idx],
                );
            }
        }
    }

    #[test]
    fn exp_f32_accuracy_over_domain() {
        let mut max_rel_err = 0.0f64;
        let mut worst_x = 0.0f32;
        let mut x = -88.0f32;
        while x <= 88.0f32 {
            let actual = f64::from(exp_f32_scalar(x));
            let expected = (f64::from(x)).exp();
            let rel_err = (actual - expected).abs() / expected;
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
                worst_x = x;
            }
            x += 0.005;
        }
        println!(
            "exp_f32 max relative error over [-88, 88]: {:.6e} at x = {:.4}",
            max_rel_err, worst_x
        );
        assert!(
            max_rel_err < 2.0e-7,
            "max relative error {max_rel_err:e} exceeded 2e-7 bound"
        );
    }

    #[test]
    fn exp_and_silu_avx2_equals_scalar() {
        let mut x = -88.0f32;
        while x <= 88.0f32 {
            let xs = [
                x,
                x + 0.001,
                x + 0.002,
                x + 0.003,
                x + 0.004,
                x + 0.005,
                x + 0.006,
                x + 0.007,
            ];
            let mut exp_scalar = [0.0f32; 8];
            let mut silu_scalar = [0.0f32; 8];
            for i in 0..8 {
                exp_scalar[i] = exp_f32_scalar(xs[i]);
                silu_scalar[i] = silu_f32_scalar(xs[i]);
            }

            #[cfg(target_arch = "x86_64")]
            unsafe {
                let vx = _mm256_loadu_ps(xs.as_ptr());
                let vexp = exp_f32_avx2(vx);
                let vsilu = silu_f32_avx2(vx);
                let mut exp_avx2 = [0.0f32; 8];
                let mut silu_avx2 = [0.0f32; 8];
                _mm256_storeu_ps(exp_avx2.as_mut_ptr(), vexp);
                _mm256_storeu_ps(silu_avx2.as_mut_ptr(), vsilu);

                for i in 0..8 {
                    assert_eq!(
                        exp_scalar[i].to_bits(),
                        exp_avx2[i].to_bits(),
                        "exp mismatch at x={}: scalar={:e} vs avx2={:e}",
                        xs[i],
                        exp_scalar[i],
                        exp_avx2[i]
                    );
                    assert_eq!(
                        silu_scalar[i].to_bits(),
                        silu_avx2[i].to_bits(),
                        "silu mismatch at x={}: scalar={:e} vs avx2={:e}",
                        xs[i],
                        silu_scalar[i],
                        silu_avx2[i]
                    );
                }
            }
            x += 0.008;
        }

        // Test boundary values
        let special = [
            0.0f32,
            -0.0f32,
            -90.0f32,
            90.0f32,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0f32,
        ];
        for &s in &special {
            let e_s = exp_f32_scalar(s);
            let si_s = silu_f32_scalar(s);
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let vx = _mm256_set1_ps(s);
                let vexp = exp_f32_avx2(vx);
                let vsilu = silu_f32_avx2(vx);
                let mut exp_a = [0.0f32; 8];
                let mut silu_a = [0.0f32; 8];
                _mm256_storeu_ps(exp_a.as_mut_ptr(), vexp);
                _mm256_storeu_ps(silu_a.as_mut_ptr(), vsilu);
                if s.is_nan() {
                    assert!(exp_a[0].is_nan());
                    assert!(silu_a[0].is_nan());
                } else {
                    assert_eq!(e_s.to_bits(), exp_a[0].to_bits());
                    assert_eq!(si_s.to_bits(), silu_a[0].to_bits());
                }
            }
        }
    }

    #[test]
    fn conv_silu_tap4_avx2_equals_scalar() {
        let mut rng = SimpleRng::new(0xdead_beef_cafe_babe);
        for _ in 0..1000 {
            let mut taps_s = [[0.0f32; 4]; 8];
            let mut hist_s = [[0.0f32; 4]; 8];
            let mut tap0 = [0.0f32; 8];
            let mut tap1 = [0.0f32; 8];
            let mut tap2 = [0.0f32; 8];
            let mut tap3 = [0.0f32; 8];
            let mut hist0 = [0.0f32; 8];
            let mut hist1 = [0.0f32; 8];
            let mut hist2 = [0.0f32; 8];
            let mut hist3 = [0.0f32; 8];

            for lane in 0..8 {
                for t in 0..4 {
                    let tv = rng.next_f32_range(-2.0, 2.0);
                    let hv = rng.next_f32_range(-4.0, 4.0);
                    taps_s[lane][t] = tv;
                    hist_s[lane][t] = hv;
                }
                tap0[lane] = taps_s[lane][0];
                tap1[lane] = taps_s[lane][1];
                tap2[lane] = taps_s[lane][2];
                tap3[lane] = taps_s[lane][3];
                hist0[lane] = hist_s[lane][0];
                hist1[lane] = hist_s[lane][1];
                hist2[lane] = hist_s[lane][2];
                hist3[lane] = hist_s[lane][3];
            }

            let mut out_scalar = [0.0f32; 8];
            for lane in 0..8 {
                out_scalar[lane] = conv_silu_tap4_scalar(taps_s[lane], hist_s[lane]);
            }

            #[cfg(target_arch = "x86_64")]
            unsafe {
                let v_tap0 = _mm256_loadu_ps(tap0.as_ptr());
                let v_tap1 = _mm256_loadu_ps(tap1.as_ptr());
                let v_tap2 = _mm256_loadu_ps(tap2.as_ptr());
                let v_tap3 = _mm256_loadu_ps(tap3.as_ptr());
                let v_hist0 = _mm256_loadu_ps(hist0.as_ptr());
                let v_hist1 = _mm256_loadu_ps(hist1.as_ptr());
                let v_hist2 = _mm256_loadu_ps(hist2.as_ptr());
                let v_hist3 = _mm256_loadu_ps(hist3.as_ptr());

                let vy = conv_silu_tap4_avx2(
                    v_tap0, v_tap1, v_tap2, v_tap3, v_hist0, v_hist1, v_hist2, v_hist3,
                );
                let mut out_avx2 = [0.0f32; 8];
                _mm256_storeu_ps(out_avx2.as_mut_ptr(), vy);

                for lane in 0..8 {
                    assert_eq!(
                        out_scalar[lane].to_bits(),
                        out_avx2[lane].to_bits(),
                        "mismatch at lane {lane}: scalar={:e} vs avx2={:e}",
                        out_scalar[lane],
                        out_avx2[lane]
                    );
                }
            }
        }
    }

    #[test]
    fn f32_tracks_q32_oracle() {
        let decays = [0.5f32, 0.9f32, 0.99f32, 0.999f32];
        let dk = 128;
        let w = 32;
        let steps = 1024;
        let q_scale = 1.0f32 / (dk as f32).sqrt();
        let norm_q = f64::from(q_scale);

        println!(
            "\n{:<7} | {:<12} | {:<12} | {:<14} | {:<12} | {:<14} | {:<10}",
            "decay",
            "max |Δo|",
            "max |o|",
            "max |Δo|/|o|",
            "meas ε",
            "bound ‖q‖ε/(1-α)",
            "actual/bound"
        );
        println!("{:-<95}", "");

        for &decay in &decays {
            let mut rng = SimpleRng::new(0x4242_4242_1337_0001);
            let mut s_f32 = vec![0.0f32; dk * w];
            let mut s_q32 = vec![0i64; dk * w];
            let mut c_q32 = 1i64 << 32;
            let mut o_f32 = vec![0.0f32; w];

            let mut max_abs_diff_o = 0.0f32;
            let mut max_abs_o = 0.0f32;
            let mut max_eps = 0.0f64;

            let decay_q32 = f32_to_q32(decay);

            for _step in 0..steps {
                let mut k = Vec::with_capacity(dk);
                let mut k_sq = 0.0f32;
                for _ in 0..dk {
                    let ki = rng.next_f32_range(-1.0, 1.0);
                    k_sq += ki * ki;
                    k.push(ki);
                }
                let k_norm = k_sq.sqrt();
                for ki in &mut k {
                    *ki /= k_norm;
                }

                let mut q = Vec::with_capacity(dk);
                let mut q_sq = 0.0f32;
                for _ in 0..dk {
                    let qi = rng.next_f32_range(-1.0, 1.0);
                    q_sq += qi * qi;
                    q.push(qi);
                }
                let q_norm = q_sq.sqrt();
                for qi in &mut q {
                    *qi = (*qi / q_norm) * q_scale;
                }

                let beta = rng.next_f32_range(0.0, 1.0);
                let mut v = Vec::with_capacity(w);
                for _ in 0..w {
                    v.push(rng.next_f32_range(-4.0, 4.0));
                }

                let k_q32: Vec<i64> = k.iter().map(|&x| f32_to_q32(x)).collect();
                let q_q32: Vec<i64> = q.iter().map(|&x| f32_to_q32(x)).collect();
                let v_q32: Vec<i64> = v.iter().map(|&x| f32_to_q32(x)).collect();
                let beta_q32 = f32_to_q32(beta);

                let s_f32_before = s_f32.clone();

                delta_step_f32_avx2(&mut s_f32, &mut o_f32, &q, &k, &v, decay, beta);
                let o_q32 = delta_head_step_ref(
                    &mut s_q32, &mut c_q32, &q_q32, &k_q32, &v_q32, decay_q32, beta_q32,
                )
                .expect("oracle step failed");

                for j in 0..w {
                    let o_q_float = q32_to_f32(o_q32[j]);
                    let diff = (o_f32[j] - o_q_float).abs();
                    if diff > max_abs_diff_o {
                        max_abs_diff_o = diff;
                    }
                    let mag = o_q_float.abs().max(o_f32[j].abs());
                    if mag > max_abs_o {
                        max_abs_o = mag;
                    }
                }

                for j in 0..w {
                    let mut sk = 0.0f64;
                    for i in 0..dk {
                        sk += f64::from(s_f32_before[i * w + j]) * f64::from(k[i]);
                    }
                    let mut sum_diff_sq = 0.0f64;
                    for i in 0..dk {
                        let exact_col_i = f64::from(decay)
                            * (f64::from(s_f32_before[i * w + j])
                                - f64::from(beta) * sk * f64::from(k[i]))
                            + f64::from(beta) * f64::from(v[j]) * f64::from(k[i]);
                        let diff_i = f64::from(s_f32[i * w + j]) - exact_col_i;
                        sum_diff_sq += diff_i * diff_i;
                    }
                    let col_err = sum_diff_sq.sqrt();
                    if col_err > max_eps {
                        max_eps = col_err;
                    }
                }
            }

            let bound = norm_q * max_eps / (1.0 - f64::from(decay));
            let rel_err = f64::from(max_abs_diff_o) / f64::from(max_abs_o);
            let ratio = f64::from(max_abs_diff_o) / bound;

            println!(
                "{:<7.3} | {:<12.6e} | {:<12.6} | {:<14.6e} | {:<12.6e} | {:<14.6e} | {:<10.4}",
                decay, max_abs_diff_o, max_abs_o, rel_err, max_eps, bound, ratio
            );

            assert!(
                rel_err <= 1e-3,
                "decay {decay}: relative error {rel_err:e} exceeded 1e-3"
            );
        }
    }

    #[test]
    #[ignore = "timing benchmark: run with cargo test --release --lib f32_vs_q32_timing -- --ignored --nocapture"]
    fn f32_vs_q32_timing() {
        let dk = 128;
        let w = 32;
        let steps = 200;
        let reps = 200;
        let runs = 5;
        let total_steps = steps * reps;
        let total_elem_steps = (total_steps * dk * w) as f64;

        let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_BABE);
        let q_scale = 1.0f32 / (dk as f32).sqrt();

        let mut k_steps = Vec::with_capacity(steps);
        let mut q_steps = Vec::with_capacity(steps);
        let mut v_steps = Vec::with_capacity(steps);
        let mut k_q32_steps = Vec::with_capacity(steps);
        let mut q_q32_steps = Vec::with_capacity(steps);
        let mut v_q32_steps = Vec::with_capacity(steps);

        let decay = 0.99f32;
        let beta = 0.8f32;
        let decay_q32 = f32_to_q32(decay);
        let beta_q32 = f32_to_q32(beta);

        for _ in 0..steps {
            let mut k = Vec::with_capacity(dk);
            let mut q = Vec::with_capacity(dk);
            let mut k_sq = 0.0f32;
            let mut q_sq = 0.0f32;
            for _ in 0..dk {
                let ki = rng.next_f32_range(-1.0, 1.0);
                let qi = rng.next_f32_range(-1.0, 1.0);
                k_sq += ki * ki;
                q_sq += qi * qi;
                k.push(ki);
                q.push(qi);
            }
            let k_norm = k_sq.sqrt();
            let q_norm = q_sq.sqrt();
            for ki in &mut k {
                *ki /= k_norm;
            }
            for qi in &mut q {
                *qi = (*qi / q_norm) * q_scale;
            }

            let mut v = Vec::with_capacity(w);
            for _ in 0..w {
                v.push(rng.next_f32_range(-4.0, 4.0));
            }

            let k_q32: Vec<i64> = k.iter().map(|&x| f32_to_q32(x)).collect();
            let q_q32: Vec<i64> = q.iter().map(|&x| f32_to_q32(x)).collect();
            let v_q32: Vec<i64> = v.iter().map(|&x| f32_to_q32(x)).collect();

            k_steps.push(k);
            q_steps.push(q);
            v_steps.push(v);
            k_q32_steps.push(k_q32);
            q_q32_steps.push(q_q32);
            v_q32_steps.push(v_q32);
        }

        println!("\n=== f32 vs Q32.32 Timing (single-thread, best of 5, 200 steps x 200 reps = 40,000 steps) ===");

        let mut s_f32 = vec![0.0f32; dk * w];
        let mut s_q32 = vec![0i64; dk * w];
        let mut c_q32 = 1i64 << 32;
        let mut o_f32 = vec![0.0f32; w];
        for s in 0..steps {
            delta_step_f32_avx2(
                &mut s_f32,
                &mut o_f32,
                &q_steps[s],
                &k_steps[s],
                &v_steps[s],
                decay,
                beta,
            );
            let _ = delta_head_step_fast(
                &mut s_q32,
                &mut c_q32,
                &q_q32_steps[s],
                &k_q32_steps[s],
                &v_q32_steps[s],
                decay_q32,
                beta_q32,
            );
        }

        let mut q32_best_ns = f64::INFINITY;
        for run in 0..runs {
            let mut total_dur = std::time::Duration::ZERO;
            for _rep in 0..reps {
                s_q32.fill(0);
                c_q32 = 1i64 << 32;
                let t0 = std::time::Instant::now();
                for step in 0..steps {
                    let o = delta_head_step_fast(
                        &mut s_q32,
                        &mut c_q32,
                        &q_q32_steps[step],
                        &k_q32_steps[step],
                        &v_q32_steps[step],
                        decay_q32,
                        beta_q32,
                    )
                    .unwrap();
                    std::hint::black_box(o);
                }
                total_dur += t0.elapsed();
            }
            let ns_per_elem = (total_dur.as_nanos() as f64) / total_elem_steps;
            println!(
                "Q32.32 run {run}: {:.3} ms total, {:.4} ns / elem-step",
                total_dur.as_secs_f64() * 1000.0,
                ns_per_elem
            );
            if ns_per_elem < q32_best_ns {
                q32_best_ns = ns_per_elem;
            }
        }

        let mut f32_best_ns = f64::INFINITY;
        for run in 0..runs {
            let mut total_dur = std::time::Duration::ZERO;
            for _rep in 0..reps {
                s_f32.fill(0.0);
                let t0 = std::time::Instant::now();
                for step in 0..steps {
                    delta_step_f32_avx2(
                        &mut s_f32,
                        &mut o_f32,
                        &q_steps[step],
                        &k_steps[step],
                        &v_steps[step],
                        decay,
                        beta,
                    );
                    std::hint::black_box(&o_f32);
                }
                total_dur += t0.elapsed();
            }
            let ns_per_elem = (total_dur.as_nanos() as f64) / total_elem_steps;
            println!(
                "f32 AVX2 run {run}: {:.3} ms total, {:.4} ns / elem-step",
                total_dur.as_secs_f64() * 1000.0,
                ns_per_elem
            );
            if ns_per_elem < f32_best_ns {
                f32_best_ns = ns_per_elem;
            }
        }

        let speedup = q32_best_ns / f32_best_ns;
        println!("\nSummary (best of 5):");
        println!(
            "  Q32.32 (delta_head_step_fast): {:.4} ns / elem-step ({:.2} µs / 4096-elem step)",
            q32_best_ns,
            q32_best_ns * 4096.0 / 1000.0
        );
        println!(
            "  f32 AVX2+FMA (delta_step_f32_avx2): {:.4} ns / elem-step ({:.2} µs / 4096-elem step)",
            f32_best_ns,
            f32_best_ns * 4096.0 / 1000.0
        );
        println!("  Speedup: {speedup:.2}x");
    }

    #[test]
    fn f32_readout_to_q32_overflow_and_nan() {
        assert!(matches!(
            f32_readout_to_q32(f32::NAN),
            Err(StepError::Overflow)
        ));
        assert!(matches!(
            f32_readout_to_q32(f32::INFINITY),
            Err(StepError::Overflow)
        ));
        assert!(matches!(
            f32_readout_to_q32(f32::NEG_INFINITY),
            Err(StepError::Overflow)
        ));
        assert!(matches!(
            f32_readout_to_q32(1e20f32),
            Err(StepError::Overflow)
        ));
        assert!(matches!(
            f32_readout_to_q32(-1e20f32),
            Err(StepError::Overflow)
        ));

        assert_eq!(f32_readout_to_q32(0.0).unwrap(), 0);
        assert_eq!(f32_readout_to_q32(1.0).unwrap(), 1i64 << 32);
        assert_eq!(f32_readout_to_q32(-1.0).unwrap(), -(1i64 << 32));
        assert_eq!(f32_readout_to_q32(0.5).unwrap(), 1i64 << 31);
        assert_eq!(f32_readout_to_q32(-0.5).unwrap(), -(1i64 << 31));
    }

    #[test]
    fn affine_head_f32_state_carrier() {
        use crate::engine::mixer::{AffineHead, Carrier, HeadState};

        let mut head = AffineHead::with_carrier(128 * 32, Carrier::F32);
        let HeadState::F32 { ref mut s } = head.state else {
            panic!("expected F32 head state");
        };
        assert_eq!(s.len(), 128 * 32);

        for &x in s.iter() {
            assert_eq!(x.to_bits(), 0.0f32.to_bits());
        }

        s[42] = 2.5f32;

        let cloned = head.clone();
        let HeadState::F32 { s: ref s_cloned } = cloned.state else {
            panic!("expected F32 head state");
        };
        assert_eq!(s_cloned[42].to_bits(), 2.5f32.to_bits());

        let mut target = AffineHead::with_carrier(128 * 32, Carrier::F32);
        target.clone_from(&head);
        let HeadState::F32 { s: ref s_target } = target.state else {
            panic!("expected F32 head state");
        };
        assert_eq!(s_target[42].to_bits(), 2.5f32.to_bits());
    }

    #[test]
    fn l2_norm_avx2_equals_scalar() {
        let mut rng = SimpleRng::new(0xABCD_EF01_2345_6789);
        for len in [8, 16, 32, 64, 128, 256] {
            let mut x_scalar = vec![0.0f32; len];
            let mut x_avx2 = vec![0.0f32; len];
            for i in 0..len {
                let v = rng.next_f32_range(-10.0, 10.0);
                x_scalar[i] = v;
                x_avx2[i] = v;
            }
            let eps = 1e-6f32;
            let q_scale = 1.0f32 / (len as f32).sqrt();

            l2_normalize_f32_in_place_scalar(&mut x_scalar, eps, q_scale).unwrap();
            #[cfg(target_arch = "x86_64")]
            unsafe {
                l2_normalize_f32_in_place_avx2_inner(&mut x_avx2, eps, q_scale).unwrap();
                for i in 0..len {
                    assert_eq!(
                        x_scalar[i].to_bits(),
                        x_avx2[i].to_bits(),
                        "L2 norm mismatch at len {len}, idx {i}: scalar={:e} vs avx2={:e}",
                        x_scalar[i],
                        x_avx2[i]
                    );
                }
            }
        }
    }

    #[test]
    fn stage2_gated_norm_avx2_equals_scalar() {
        let mut rng = SimpleRng::new(0x5678_9ABC_DEF0_1234);
        let head_v = 128;
        let mut o = vec![0.0f32; head_v];
        let mut z = vec![0i64; head_v];
        let mut ssm_norm = vec![0.0f32; head_v];
        for i in 0..head_v {
            o[i] = rng.next_f32_range(-5.0, 5.0);
            z[i] = f32_to_q32(rng.next_f32_range(-5.0, 5.0));
            ssm_norm[i] = rng.next_f32_range(0.1, 2.0);
        }
        let eps = 1e-6f32;
        let mut dst_scalar = vec![0i64; head_v];
        let mut dst_avx2 = vec![0i64; head_v];

        stage2_gated_norm_f32_scalar(&o, &z, &ssm_norm, eps, &mut dst_scalar).unwrap();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            stage2_gated_norm_f32_avx2_inner(&o, &z, &ssm_norm, eps, &mut dst_avx2).unwrap();
            for i in 0..head_v {
                assert_eq!(
                    dst_scalar[i], dst_avx2[i],
                    "stage2 norm mismatch at idx {i}: scalar={} vs avx2={}",
                    dst_scalar[i], dst_avx2[i]
                );
            }
        }
    }
}
