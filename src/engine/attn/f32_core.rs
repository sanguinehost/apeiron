//! IEEE binary32 (f32) carrier for the softmax attention core (Milakov & Gimelshein 2018).
//!
//! # Contract
//!
//! Computes causal scaled dot-product attention for one query head over positions `0..=pos`:
//! 1. Scores `score_s = (q · k_s) · scale` for each causal key `s ∈ 0..=pos`.
//! 2. Online running-max softmax: maintains running maximum `m`, denominator `den = Σ exp(score_s - m)`,
//!    and numerator accumulator `acc = Σ exp(score_s - m) · v_s`.
//! 3. Normalized readout: `out = acc / den`.
//!
//! # Conversions and Refusals
//!
//! - Input conversion: Q32.32 fixed-point words `x` map to binary32 via `(x as f32) / 2^32`.
//! - Readout conversion: binary32 output values map to Q32.32 via round-to-nearest (`round() · 2^32`).
//! - Refusals:
//!   - Non-finite floating-point inputs (NaN, ±Inf) or outputs exceeding the Q32.32 range refuse with
//!     [`AttnError::Overflow`].
//!   - Non-positive or non-finite softmax denominator sums refuse with [`AttnError::ZeroSum`].
//!   - Geometry violations (empty head dimension, row length mismatches) refuse with [`AttnError::BadGeometry`].
//!
//! # Trust Boundary
//!
//! The binary32 kernels (scalar twin and AVX2+FMA) are not Aeneas-extracted. Correctness relies on
//! IEEE 754-2019 standard conformance, exact bit-for-bit identity between the AVX2 kernel and its
//! scalar twin specification, test-measured bounded relative error on the polynomial exponential,
//! and bounded deviation against the Q32.32 fixed-point oracle.

#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::wildcard_imports,
    reason = "x86_64 SIMD intrinsics imported en masse for AVX2+FMA"
)]
use std::arch::x86_64::*;

use crate::engine::attn::AttnError;
use crate::engine::mixer::f32_carrier::exp_f32_scalar;

/// Convert a binary32 float back to Q32.32 fixed-point `i64` using round-to-nearest.
///
/// Refuses non-finite floats (NaN, ±Inf) and values outside the representable `i64` Q32.32 range
/// `[-2^31, 2^31)` with [`AttnError::Overflow`].
///
/// # Complexity
///
/// O(1) temporal and spatial: 1 float test, 1 multiplication, 1 rounding, 1 range check.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the finite range check strictly inside [-2^63, 2^63) guarantees that the cast to i64 does not truncate"
)]
pub fn f32_to_q32_checked(v: f32) -> Result<i64, AttnError> {
    if !v.is_finite() {
        return Err(AttnError::Overflow);
    }
    let q = (f64::from(v) * 4_294_967_296.0).round();
    if !(q.is_finite() && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&q))
    {
        return Err(AttnError::Overflow);
    }
    Ok(q as i64)
}

/// Compute scaled dot product `(q · k) · scale` in scalar arithmetic with fixed 8-lane reduction.
#[inline]
#[allow(
    clippy::needless_range_loop,
    reason = "fixed lane loop order pins execution order"
)]
fn dot_f32_scalar(q: &[f32], k: &[f32], scale: f32) -> f32 {
    let hd = q.len();
    if hd.is_multiple_of(8) {
        let mut lanes = [0.0f32; 8];
        let chunks = hd / 8;
        for c in 0..chunks {
            for lane in 0..8 {
                lanes[lane] = q[c * 8 + lane].mul_add(k[c * 8 + lane], lanes[lane]);
            }
        }
        let raw =
            lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        raw * scale
    } else {
        let mut raw = 0.0f32;
        for i in 0..hd {
            raw = q[i].mul_add(k[i], raw);
        }
        raw * scale
    }
}

/// Compute scaled dot product `(q · k) · scale` using AVX2+FMA.
/// Matches [`dot_f32_scalar`] bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_f32_avx2(q: &[f32], k: &[f32], scale: f32) -> f32 {
    let hd = q.len();
    let chunks = hd / 8;
    let mut v_dot = _mm256_setzero_ps();
    for c in 0..chunks {
        let v_q = _mm256_loadu_ps(q.as_ptr().add(c * 8));
        let v_k = _mm256_loadu_ps(k.as_ptr().add(c * 8));
        v_dot = _mm256_fmadd_ps(v_q, v_k, v_dot);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), v_dot);
    let raw = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    raw * scale
}

/// Scalar twin implementation of the binary32 online-softmax attention core with strided K/V layout.
///
/// Evaluates causal attention over positions `0..=pos` using the Milakov & Gimelshein 2018
/// online normalizer recurrence. Serves as the specification for [`attn_core_f32_strided_avx2_inner`].
///
/// # Complexity
///
/// `O((pos + 1) · head_dim)` time in floating-point multiply-adds (FMAs) and `O(pos + 1)`
/// exponential evaluations. Zero heap allocation for `head_dim <= 256` (stack buffer).
#[allow(
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "mathematical recurrence symbols, fixed lane loop and explicit strided geometry"
)]
pub fn attn_core_f32_strided_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    if head_dim == 0 || out.len() != head_dim || q.len() != head_dim || stride < head_dim {
        return Err(AttnError::BadGeometry);
    }
    let count = pos.checked_add(1).ok_or(AttnError::Overflow)?;
    let required_len = pos
        .checked_mul(stride)
        .and_then(|p| p.checked_add(head_dim))
        .ok_or(AttnError::Overflow)?;
    if k.len() < required_len || v.len() < required_len {
        return Err(AttnError::BadGeometry);
    }

    let score0 = dot_f32_scalar(q, &k[..head_dim], scale);
    if !score0.is_finite() {
        return Err(AttnError::Overflow);
    }

    let mut m = score0;
    let mut den = 1.0f32;

    let mut acc_stack = [0.0f32; 256];
    let mut acc_heap;
    let acc: &mut [f32] = if head_dim <= 256 {
        &mut acc_stack[..head_dim]
    } else {
        acc_heap = vec![0.0f32; head_dim];
        &mut acc_heap[..]
    };

    for i in 0..head_dim {
        let val = v[i];
        if !val.is_finite() {
            return Err(AttnError::Overflow);
        }
        acc[i] = val;
    }

    for s in 1..count {
        let base = s * stride;
        let k_row = &k[base..base + head_dim];
        let v_row = &v[base..base + head_dim];

        let score_s = dot_f32_scalar(q, k_row, scale);
        if !score_s.is_finite() {
            return Err(AttnError::Overflow);
        }

        let m_prev = m;
        let m_next = if score_s > m_prev { score_s } else { m_prev };
        let alpha = exp_f32_scalar(m_prev - m_next);
        let w = exp_f32_scalar(score_s - m_next);

        den = den.mul_add(alpha, w);

        for i in 0..head_dim {
            let vi = v_row[i];
            if !vi.is_finite() {
                return Err(AttnError::Overflow);
            }
            let prod = w * vi;
            acc[i] = acc[i].mul_add(alpha, prod);
        }
        m = m_next;
    }

    if den <= 0.0 || !den.is_finite() {
        return Err(AttnError::ZeroSum);
    }

    let inv_den = 1.0f32 / den;
    for i in 0..head_dim {
        let res = acc[i] * inv_den;
        if !res.is_finite() {
            return Err(AttnError::Overflow);
        }
        out[i] = res;
    }

    Ok(())
}

/// AVX2+FMA SIMD implementation of the binary32 online-softmax attention core with strided K/V layout.
///
/// Processes 8 value components per vector. Matches [`attn_core_f32_strided_scalar`] bit for bit.
///
/// # Complexity
///
/// `O((pos + 1) · head_dim)` time in 8-lane AVX2+FMA vector instructions and `O(pos + 1)`
/// scalar exponential evaluations. Zero heap allocations for `head_dim <= 256` (register/stack accumulators).
///
/// # Safety
///
/// Requires the target processor to support AVX2 and FMA feature sets.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "mathematical recurrence symbols, vector variables and explicit strided geometry"
)]
unsafe fn attn_core_f32_strided_avx2_inner(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    let count = pos.checked_add(1).ok_or(AttnError::Overflow)?;
    let chunks = head_dim / 8;

    let score0 = dot_f32_avx2(q, &k[..head_dim], scale);
    if !score0.is_finite() {
        return Err(AttnError::Overflow);
    }

    let mut m = score0;
    let mut den = 1.0f32;

    let mut acc_stack = [_mm256_setzero_ps(); 32];
    let mut acc_heap;
    let acc_vec: &mut [__m256] = if chunks <= 32 {
        &mut acc_stack[..chunks]
    } else {
        acc_heap = vec![_mm256_setzero_ps(); chunks];
        &mut acc_heap[..]
    };

    for c in 0..chunks {
        let v_val = _mm256_loadu_ps(v.as_ptr().add(c * 8));
        acc_vec[c] = v_val;
    }

    for s in 1..count {
        let base = s * stride;
        let k_row = &k[base..base + head_dim];
        let v_ptr = v.as_ptr().add(base);

        let score_s = dot_f32_avx2(q, k_row, scale);
        if !score_s.is_finite() {
            return Err(AttnError::Overflow);
        }

        let m_prev = m;
        let m_next = if score_s > m_prev { score_s } else { m_prev };
        let alpha = exp_f32_scalar(m_prev - m_next);
        let w = exp_f32_scalar(score_s - m_next);

        den = den.mul_add(alpha, w);

        let v_alpha = _mm256_set1_ps(alpha);
        let v_w = _mm256_set1_ps(w);

        for c in 0..chunks {
            let v_val = _mm256_loadu_ps(v_ptr.add(c * 8));
            let prod = _mm256_mul_ps(v_w, v_val);
            acc_vec[c] = _mm256_fmadd_ps(acc_vec[c], v_alpha, prod);
        }
        m = m_next;
    }

    if den <= 0.0 || !den.is_finite() {
        return Err(AttnError::ZeroSum);
    }

    let inv_den = 1.0f32 / den;
    let v_inv_den = _mm256_set1_ps(inv_den);

    for c in 0..chunks {
        let res = _mm256_mul_ps(acc_vec[c], v_inv_den);
        _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), res);
    }

    for item in out.iter().take(head_dim) {
        if !item.is_finite() {
            return Err(AttnError::Overflow);
        }
    }

    Ok(())
}

/// Dispatches strided attention core execution to AVX2+FMA when supported and `head_dim % 8 == 0`,
/// falling back to the scalar implementation.
///
/// # Complexity
///
/// `O((pos + 1) · head_dim)` time and zero heap allocation for `head_dim <= 256`.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit strided geometry and buffers passed to core kernel"
)]
pub fn attn_core_f32_strided(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    if head_dim == 0 || out.len() != head_dim || q.len() != head_dim || stride < head_dim {
        return Err(AttnError::BadGeometry);
    }
    let required_len = pos
        .checked_mul(stride)
        .and_then(|p| p.checked_add(head_dim))
        .ok_or(AttnError::Overflow)?;
    if k.len() < required_len || v.len() < required_len {
        return Err(AttnError::BadGeometry);
    }

    #[cfg(target_arch = "x86_64")]
    if head_dim.is_multiple_of(8)
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        return unsafe {
            attn_core_f32_strided_avx2_inner(q, k, v, head_dim, stride, pos, scale, out)
        };
    }
    attn_core_f32_strided_scalar(q, k, v, head_dim, stride, pos, scale, out)
}

/// Dispatches contiguous attention core execution (`stride = head_dim`).
///
/// # Complexity
///
/// `O((pos + 1) · head_dim)` time.
#[allow(
    dead_code,
    reason = "public contiguous core specification and test target"
)]
pub fn attn_core_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    attn_core_f32_strided(q, k, v, head_dim, head_dim, pos, scale, out)
}

/// Scalar twin for contiguous attention core (`stride = head_dim`).
#[allow(
    dead_code,
    reason = "public scalar twin specification for contiguous layout"
)]
pub fn attn_core_f32_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    pos: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    attn_core_f32_strided_scalar(q, k, v, head_dim, head_dim, pos, scale, out)
}

#[cfg(test)]
#[allow(
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "test harnesses, random generators, timing loops and numerical verification"
)]
pub(crate) mod tests {
    use super::*;
    use crate::engine::attn::KvState;
    use crate::engine::mixer::Carrier;

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
    fn test_shared_exp_accuracy_in_attention() {
        let steps = 100_000;
        let mut max_rel_err = 0.0f64;
        let mut max_err_at = 0.0f32;

        for step in 0..=steps {
            let x = -80.0f32 * (step as f32 / steps as f32);
            let approx = f64::from(exp_f32_scalar(x));
            let exact = f64::from(x).exp();

            if exact > 1e-35 {
                let err = (approx - exact).abs() / exact;
                if err > max_rel_err {
                    max_rel_err = err;
                    max_err_at = x;
                }
            }
        }

        // Relative error of shared exp must be bounded by 3.0e-7 across [-80.0, 0.0]
        assert!(
            max_rel_err < 3.0e-7,
            "max rel err {max_rel_err:e} at {max_err_at} exceeds 3.0e-7 bound"
        );
    }

    #[test]
    fn test_attn_core_avx2_equals_scalar() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut rng = SimpleRng::new(0x1337_C0DE_5EED_0001);
            let test_configs = [
                (64, 64, 0),
                (64, 128, 3),
                (128, 128, 7),
                (128, 256, 31),
                (256, 512, 15),
            ];

            for &(hd, stride, pos) in &test_configs {
                let count = pos + 1;
                let mut q = vec![0.0f32; hd];
                for item in q.iter_mut().take(hd) {
                    *item = rng.next_f32_range(-1.0, 1.0);
                }
                let total_len = pos * stride + hd;
                let mut k = vec![0.0f32; total_len];
                let mut v = vec![0.0f32; total_len];
                for s in 0..count {
                    let base = s * stride;
                    for i in 0..hd {
                        k[base + i] = rng.next_f32_range(-1.0, 1.0);
                        v[base + i] = rng.next_f32_range(-4.0, 4.0);
                    }
                }
                let scale = 1.0f32 / (hd as f32).sqrt();

                let mut out_scalar = vec![0.0f32; hd];
                let mut out_avx2 = vec![0.0f32; hd];

                attn_core_f32_strided_scalar(&q, &k, &v, hd, stride, pos, scale, &mut out_scalar)
                    .expect("scalar");
                unsafe {
                    attn_core_f32_strided_avx2_inner(
                        &q,
                        &k,
                        &v,
                        hd,
                        stride,
                        pos,
                        scale,
                        &mut out_avx2,
                    )
                    .expect("avx2");
                }

                for i in 0..hd {
                    assert_eq!(
                        out_scalar[i].to_bits(),
                        out_avx2[i].to_bits(),
                        "hd {hd} stride {stride} pos {pos} idx {i}: scalar={:e} ({:08x}) vs avx2={:e} ({:08x})",
                        out_scalar[i],
                        out_scalar[i].to_bits(),
                        out_avx2[i],
                        out_avx2[i].to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn test_attn_core_refusal_on_non_finite() {
        let hd = 16;
        let pos = 2;
        let count = pos + 1;
        let scale = 1.0 / (hd as f32).sqrt();

        // 1. NaN in query
        let mut q = vec![0.5f32; hd];
        q[3] = f32::NAN;
        let k = vec![0.5f32; count * hd];
        let v = vec![0.5f32; count * hd];
        let mut out = vec![0.0f32; hd];
        assert_eq!(
            attn_core_f32(&q, &k, &v, hd, pos, scale, &mut out),
            Err(AttnError::Overflow)
        );

        // 2. Inf in key
        let q = vec![0.5f32; hd];
        let mut k = vec![0.5f32; count * hd];
        k[hd + 2] = f32::INFINITY;
        assert_eq!(
            attn_core_f32(&q, &k, &v, hd, pos, scale, &mut out),
            Err(AttnError::Overflow)
        );

        // 3. NaN in value
        let k = vec![0.5f32; count * hd];
        let mut v = vec![0.5f32; count * hd];
        v[2 * hd + 1] = f32::NAN;
        assert_eq!(
            attn_core_f32(&q, &k, &v, hd, pos, scale, &mut out),
            Err(AttnError::Overflow)
        );

        // 4. Bad geometry: query width mismatch
        let v = vec![0.5f32; count * hd];
        assert_eq!(
            attn_core_f32(&q[..hd - 1], &k, &v, hd, pos, scale, &mut out),
            Err(AttnError::BadGeometry)
        );

        // 5. Bad geometry: stride < head_dim
        assert_eq!(
            attn_core_f32_strided(&q, &k, &v, hd, hd - 1, pos, scale, &mut out),
            Err(AttnError::BadGeometry)
        );
    }

    #[test]
    fn test_attn_core_vs_q32_oracle_deviation() {
        let mut rng = SimpleRng::new(0xCAFE_BABE_9876_5432);
        let test_positions = [0, 1, 5, 20, 50];
        let hd = 128;
        let n_head = 4;
        let n_head_kv = 2;
        let max_cap = 64;

        let mut kv_q32 =
            KvState::with_carrier(1, n_head, n_head_kv, hd, max_cap, Carrier::Q32).expect("kv q32");
        let mut kv_f32 =
            KvState::with_carrier(1, n_head, n_head_kv, hd, max_cap, Carrier::F32).expect("kv f32");

        // Fill KV caches with realistic Q32.32 values
        let scale_q32 = 1i64 << 32;
        for _ in 0..max_cap {
            let mut k_rows = Vec::with_capacity(n_head_kv * hd);
            let mut v_rows = Vec::with_capacity(n_head_kv * hd);
            for _ in 0..n_head_kv * hd {
                let k_f = rng.next_f32_range(-1.0, 1.0);
                let v_f = rng.next_f32_range(-2.0, 2.0);
                k_rows.push((f64::from(k_f) * (scale_q32 as f64)).round() as i64);
                v_rows.push((f64::from(v_f) * (scale_q32 as f64)).round() as i64);
            }
            kv_q32.append(0, &k_rows, &v_rows).expect("append q32");
            kv_f32.append(0, &k_rows, &v_rows).expect("append f32");
        }

        let mut max_abs_diff = 0i64;
        let mut max_rel_err = 0.0f64;

        for &pos in &test_positions {
            for h in 0..n_head {
                let mut q = Vec::with_capacity(hd);
                for _ in 0..hd {
                    let q_f = rng.next_f32_range(-1.0, 1.0);
                    q.push((f64::from(q_f) * (scale_q32 as f64)).round() as i64);
                }

                let mut oracle_out = vec![0i64; hd];
                let mut f32_out = vec![0i64; hd];

                // Run Q32 oracle
                kv_q32
                    .attn_head(0, h, &q, pos, &mut oracle_out)
                    .expect("q32 oracle");

                // Run f32 core on native f32 KV cache
                kv_f32
                    .attn_head(0, h, &q, pos, &mut f32_out)
                    .expect("f32 core");

                for i in 0..hd {
                    let o_val = oracle_out[i];
                    let f_val = f32_out[i];
                    let abs_diff = (o_val - f_val).abs();
                    if abs_diff > max_abs_diff {
                        max_abs_diff = abs_diff;
                    }
                    if o_val != 0 {
                        let rel = (abs_diff as f64) / (o_val.abs() as f64);
                        if rel > max_rel_err {
                            max_rel_err = rel;
                        }
                    }
                }
            }
        }

        let max_abs_q32_float = (max_abs_diff as f64) / (scale_q32 as f64);
        eprintln!(
            "Observed max abs diff in Q32.32: {max_abs_diff} ({max_abs_q32_float:e}), max rel err: {max_rel_err:e}"
        );
        assert!(
            max_abs_q32_float < 1.0e-3,
            "max abs deviation {max_abs_q32_float:e} exceeds 1.0e-3"
        );
    }

    /// Benchmark comparing Q32 path vs native f32 KV cache attention at pos = 200, `head_dim` = 256.
    /// Run with:
    /// `taskset -c 2 cargo test --release --lib attn_f32_vs_q32_timing -- --ignored --nocapture`
    #[test]
    #[ignore = "timing benchmark: run with cargo test --release --lib attn_f32_vs_q32_timing -- --ignored --nocapture"]
    fn attn_f32_vs_q32_timing() {
        let mut rng = SimpleRng::new(0x4242_1337_BEEF_0001);
        let hd = 256;
        let pos = 200;
        let n_head = 4;
        let n_head_kv = 2;
        let cap = pos + 1;

        let mut kv_q32 =
            KvState::with_carrier(1, n_head, n_head_kv, hd, cap, Carrier::Q32).expect("kv q32");
        let mut kv_f32 =
            KvState::with_carrier(1, n_head, n_head_kv, hd, cap, Carrier::F32).expect("kv f32");
        let scale_q32 = 1i64 << 32;

        for _ in 0..cap {
            let mut k_rows = Vec::with_capacity(n_head_kv * hd);
            let mut v_rows = Vec::with_capacity(n_head_kv * hd);
            for _ in 0..n_head_kv * hd {
                let k_f = rng.next_f32_range(-1.0, 1.0);
                let v_f = rng.next_f32_range(-2.0, 2.0);
                k_rows.push((f64::from(k_f) * (scale_q32 as f64)).round() as i64);
                v_rows.push((f64::from(v_f) * (scale_q32 as f64)).round() as i64);
            }
            kv_q32.append(0, &k_rows, &v_rows).expect("append q32");
            kv_f32.append(0, &k_rows, &v_rows).expect("append f32");
        }

        let mut q = Vec::with_capacity(hd);
        for _ in 0..hd {
            let q_f = rng.next_f32_range(-1.0, 1.0);
            q.push((f64::from(q_f) * (scale_q32 as f64)).round() as i64);
        }

        let reps = 200;
        let mut out_q32 = vec![0i64; hd];
        let mut out_f32 = vec![0i64; hd];

        // Warm up:
        for _ in 0..20 {
            kv_q32
                .attn_head(0, 0, &q, pos, &mut out_q32)
                .expect("warm q32");
            kv_f32
                .attn_head(0, 0, &q, pos, &mut out_f32)
                .expect("warm f32");
        }

        let elem_steps = (pos * hd) as f64;
        let total_elem_steps = elem_steps * f64::from(reps);

        let mut q32_times = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                kv_q32.attn_head(0, 0, &q, pos, &mut out_q32).expect("q32");
            }
            let el = t0.elapsed();
            let ns_per_elem = (el.as_nanos() as f64) / total_elem_steps;
            q32_times.push(ns_per_elem);
        }

        let mut f32_times = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                kv_f32.attn_head(0, 0, &q, pos, &mut out_f32).expect("f32");
            }
            let el = t0.elapsed();
            let ns_per_elem = (el.as_nanos() as f64) / total_elem_steps;
            f32_times.push(ns_per_elem);
        }

        let best_q32 = q32_times.iter().copied().fold(f64::INFINITY, f64::min);
        let best_f32 = f32_times.iter().copied().fold(f64::INFINITY, f64::min);
        let speedup = best_q32 / best_f32;

        println!("--- TIMING RESULTS (best of 5, pos={pos}, hd={hd}) ---");
        println!("Q32.32 path:            {best_q32:.4} ns per (pos × hd)");
        println!(
            "f32 native KV cache:    {best_f32:.4} ns per (pos × hd) (speedup: {speedup:.2}x)"
        );
    }
}
