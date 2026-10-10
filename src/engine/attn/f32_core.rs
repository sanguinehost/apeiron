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
//! The binary32 kernels (scalar twin and AVX2+FMA) are not Aeneas-extracted. The prepass now
//! admits both bodies — the loop-shape refusals are cleared — but the pinned Aeneas then stops in
//! its module analysis with an internal assert before emitting Lean, so extraction is not
//! established. Correctness relies on
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
#[cfg(target_arch = "x86_64")]
use crate::engine::mixer::f32_carrier::exp_f32_avx2;
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

    // One `ok` flag carries every in-loop `Overflow` refusal out of the three loops below, and the
    // two `return Err` sites that consume it stand after the position loop and after the readout
    // loop. This is forced by the pinned Aeneas model, which refuses this body with `Returns inside
    // of nested loops are not supported yet`: each `return Err(AttnError::Overflow)` sat inside a
    // loop body, and an early exit from a loop is the shape its translator cannot lower. No `zip`
    // appears in this body, so that refusal does not apply here. The flag keeps `Overflow` ahead of
    // the `ZeroSum` test, because the position loop's `if !ok` return precedes that test.
    //
    // `while i < head_dim && ok` writes the same prefix and stops at the same index as the `return`
    // it replaces, so `acc` and `out` carry exactly what the early exits left behind: an element is
    // stored only when the old code would have stored it before returning. Arithmetic is unchanged:
    // the same `v[i]`, `v_row[i]`, `acc[i].mul_add(alpha, w * v_row[i])` and `acc[i] * inv_den` on
    // the same elements in the same ascending order, over `0..head_dim` and `1..count`, which are
    // exactly the old `for` ranges. `m = m_next` stays guarded by `ok` because the old code reached
    // it only after a whole finite row, and `den` keeps its single `mul_add` per position. The
    // residual cost is bounded: at most one extra `head_dim` scan per failing position, since `ok`
    // stops the inner loop at the same index the `return` did.
    let mut ok = true;

    let mut i = 0usize;
    while i < head_dim && ok {
        let val = v[i];
        if val.is_finite() {
            acc[i] = val;
        } else {
            ok = false;
        }
        i += 1;
    }

    let mut s = 1usize;
    while s < count && ok {
        let base = s * stride;
        let k_row = &k[base..base + head_dim];
        let v_row = &v[base..base + head_dim];

        let score_s = dot_f32_scalar(q, k_row, scale);
        if score_s.is_finite() {
            let m_prev = m;
            let m_next = if score_s > m_prev { score_s } else { m_prev };
            let alpha = exp_f32_scalar(m_prev - m_next);
            let w = exp_f32_scalar(score_s - m_next);

            den = den.mul_add(alpha, w);

            let mut i = 0usize;
            while i < head_dim && ok {
                let vi = v_row[i];
                if vi.is_finite() {
                    let prod = w * vi;
                    acc[i] = acc[i].mul_add(alpha, prod);
                } else {
                    ok = false;
                }
                i += 1;
            }
            if ok {
                m = m_next;
            }
        } else {
            ok = false;
        }
        s += 1;
    }
    if !ok {
        return Err(AttnError::Overflow);
    }

    if den <= 0.0 || !den.is_finite() {
        return Err(AttnError::ZeroSum);
    }

    let inv_den = 1.0f32 / den;
    let mut i = 0usize;
    while i < head_dim && ok {
        let res = acc[i] * inv_den;
        if res.is_finite() {
            out[i] = res;
        } else {
            ok = false;
        }
        i += 1;
    }
    if !ok {
        return Err(AttnError::Overflow);
    }

    Ok(())
}

/// Key positions scored together by [`attn_tile8_avx2`].
const TILE: usize = 8;

/// Score, softmax-update and accumulate the `TILE` key positions `s..s + TILE` (`s >= 1`).
///
/// Bit-identical to `TILE` consecutive iterations of the single-position loop in
/// [`attn_core_f32_strided_avx2_inner`]: each score is the same 8-lane FMA chain (the eight chains
/// run interleaved, so the latency of one is hidden by the others), the running maximum is the
/// same sequential recurrence, the two `exp` calls per position are the lane-wise
/// [`exp_f32_avx2`] (equal to [`exp_f32_scalar`] bit for bit), the denominator is the same
/// sequential `mul_add` chain, and each accumulator chunk takes the same `fma(acc, alpha_j, w_j *
/// v_j)` sequence in position order (chunk-outer, position-inner, so the chunk stays in a register
/// across the tile). A non-finite score refuses with [`AttnError::Overflow`] before any of the
/// tile's state is written.
///
/// # Complexity
///
/// `O(TILE · head_dim)` FMAs and `O(1)` vector `exp` calls per tile: per position `head_dim / 8`
/// FMAs for the score and `2 · head_dim / 8` vector ops for the accumulator. No allocation.
///
/// # Safety
///
/// Requires AVX2 and FMA; `k` and `v` hold rows `0..s + TILE` at `stride`, each `head_dim` wide.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    reason = "explicit strided geometry, lane loops pin the operation order"
)]
unsafe fn attn_tile8_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    stride: usize,
    s: usize,
    scale: f32,
    m: &mut f32,
    den: &mut f32,
    acc_vec: &mut [__m256],
) -> Result<(), AttnError> {
    let chunks = head_dim / 8;
    let kp = k.as_ptr().add(s * stride);
    let vp = v.as_ptr().add(s * stride);
    let qp = q.as_ptr();
    let mut dots = [_mm256_setzero_ps(); TILE];
    for c in 0..chunks {
        let vq = _mm256_loadu_ps(qp.add(c * 8));
        for j in 0..TILE {
            let vk = _mm256_loadu_ps(kp.add(j * stride + c * 8));
            dots[j] = _mm256_fmadd_ps(vq, vk, dots[j]);
        }
    }
    let mut score = [0.0f32; TILE];
    for j in 0..TILE {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), dots[j]);
        let raw =
            lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        score[j] = raw * scale;
        if !score[j].is_finite() {
            return Err(AttnError::Overflow);
        }
    }
    let mut prev = [0.0f32; TILE];
    let mut next = [0.0f32; TILE];
    let mut run = *m;
    for j in 0..TILE {
        prev[j] = run;
        if score[j] > run {
            run = score[j];
        }
        next[j] = run;
    }
    let vnext = _mm256_loadu_ps(next.as_ptr());
    let valpha = exp_f32_avx2(_mm256_sub_ps(_mm256_loadu_ps(prev.as_ptr()), vnext));
    let vw = exp_f32_avx2(_mm256_sub_ps(_mm256_loadu_ps(score.as_ptr()), vnext));
    let mut alpha = [0.0f32; TILE];
    let mut w = [0.0f32; TILE];
    _mm256_storeu_ps(alpha.as_mut_ptr(), valpha);
    _mm256_storeu_ps(w.as_mut_ptr(), vw);
    let mut d = *den;
    for j in 0..TILE {
        d = d.mul_add(alpha[j], w[j]);
    }
    for c in 0..chunks {
        let mut a = acc_vec[c];
        for j in 0..TILE {
            let vv = _mm256_loadu_ps(vp.add(j * stride + c * 8));
            let prod = _mm256_mul_ps(_mm256_set1_ps(w[j]), vv);
            a = _mm256_fmadd_ps(a, _mm256_set1_ps(alpha[j]), prod);
        }
        acc_vec[c] = a;
    }
    *m = run;
    *den = d;
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

    // The two in-loop early exits become one `ok` flag and one `err`, both consumed after the
    // position loops, for the reason asserted on the scalar twin: the pinned Aeneas model refuses
    // this body with `Returns inside of nested loops are not supported yet`, and the `?` on
    // [`attn_tile8_avx2`] and the `return Err(AttnError::Overflow)` on a non-finite tail score were
    // both early exits from a loop body. No `zip` appears in this body, so that refusal does not
    // apply here.
    //
    // Neither loop is reassociated and no operation is reordered: the tile call still runs at
    // positions `1, 1 + TILE, ...` under the same `s + TILE <= count` bound, the tail still runs at
    // `s` up to `count - 1`, and the per-position chain is the same `mul_add` on `den`, the same
    // `_mm256_set1_ps` broadcasts of `alpha` and `w`, and the same `_mm256_fmadd_ps` chain over
    // `0..chunks` in ascending order. The `err` slot preserves the tile call's error, which is
    // always [`AttnError::Overflow`] because [`attn_tile8_avx2`] writes no tile state on its refusal
    // path, and its `if let Some(e) = err` return precedes the `den` test, so the refusal order is
    // the old one. The two `while` guards keep each loop from re-entering after a refusal, which is
    // what the old `?` and `return` did.
    let mut ok = true;
    let mut err: Option<AttnError> = None;

    let mut s = 1usize;
    while s + TILE <= count && ok {
        match attn_tile8_avx2(
            q, k, v, head_dim, stride, s, scale, &mut m, &mut den, acc_vec,
        ) {
            Ok(()) => s += TILE,
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
    }
    while s < count && ok {
        let base = s * stride;
        let k_row = &k[base..base + head_dim];
        let v_ptr = v.as_ptr().add(base);

        let score_s = dot_f32_avx2(q, k_row, scale);
        if score_s.is_finite() {
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
        } else {
            ok = false;
        }
        s += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    if !ok {
        return Err(AttnError::Overflow);
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

    // The `Overflow` refusal that the old `for item in out.iter().take(head_dim)` raised from
    // inside its own loop body now leaves as a counter tested after it, for the same asserted
    // reason. The check reads `out` and writes nothing, so `out` carries the same `head_dim` values
    // on the error path as it did when the loop returned at the first non-finite element: the vector
    // store loop above has already written them all. The index range `0..head_dim` and the ascending
    // order are the `take(head_dim)` range and order.
    let mut bad = 0u32;
    let mut i = 0usize;
    while i < head_dim {
        bad += u32::from(!out[i].is_finite());
        i += 1;
    }
    if bad != 0 {
        return Err(AttnError::Overflow);
    }

    Ok(())
}

/// Dispatches strided attention core execution to AVX2+FMA when supported and `head_dim % 8 == 0`,
/// falling back to the scalar implementation.
///
/// # Errors
///
/// [`AttnError`] on a geometry mismatch or a non-finite output.
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
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::manual_is_multiple_of,
    clippy::semicolon_if_nothing_returned,
    clippy::items_after_statements,
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

    /// The tiled AVX2 core equals the scalar twin on every tile boundary, at score magnitudes that
    /// move the running maximum often and underflow `exp`, and refuses a non-finite key at any
    /// position with the scalar twin's error.
    #[test]
    fn test_attn_core_tiled_equals_scalar_on_tile_edges() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut rng = SimpleRng::new(0x7117_ED00_0000_0001);
            let cfgs = [
                (256usize, 512usize, 0usize),
                (256, 512, 1),
                (256, 512, 8),
                (256, 512, 9),
                (256, 512, 10),
                (256, 512, 16),
                (256, 512, 17),
                (256, 512, 100),
                (256, 512, 199),
                (128, 128, 40),
                (64, 192, 63),
                (8, 8, 33),
            ];
            for amp in [0.5f32, 2.0, 6.0, 20.0] {
                for &(hd, stride, pos) in &cfgs {
                    let total = pos * stride + hd;
                    let mut q = vec![0.0f32; hd];
                    let mut k = vec![0.0f32; total];
                    let mut v = vec![0.0f32; total];
                    for x in &mut q {
                        *x = rng.next_f32_range(-amp, amp);
                    }
                    for x in &mut k {
                        *x = rng.next_f32_range(-amp, amp);
                    }
                    for x in &mut v {
                        *x = rng.next_f32_range(-4.0, 4.0);
                    }
                    let scale = 1.0f32 / (hd as f32).sqrt();
                    let mut a = vec![0.0f32; hd];
                    let mut b = vec![0.0f32; hd];
                    let ra =
                        attn_core_f32_strided_scalar(&q, &k, &v, hd, stride, pos, scale, &mut a);
                    let rb = unsafe {
                        attn_core_f32_strided_avx2_inner(&q, &k, &v, hd, stride, pos, scale, &mut b)
                    };
                    assert_eq!(ra, rb, "hd {hd} pos {pos} amp {amp}");
                    if ra.is_ok() {
                        for i in 0..hd {
                            assert_eq!(
                                a[i].to_bits(),
                                b[i].to_bits(),
                                "hd {hd} pos {pos} amp {amp} i {i}"
                            );
                        }
                    }
                    if pos >= 2 {
                        for bad in [1usize, pos / 2, pos] {
                            let mut k2 = k.clone();
                            k2[bad * stride + 3] = f32::INFINITY;
                            let ra = attn_core_f32_strided_scalar(
                                &q, &k2, &v, hd, stride, pos, scale, &mut a,
                            );
                            let rb = unsafe {
                                attn_core_f32_strided_avx2_inner(
                                    &q, &k2, &v, hd, stride, pos, scale, &mut b,
                                )
                            };
                            assert_eq!(ra, Err(AttnError::Overflow));
                            assert_eq!(rb, Err(AttnError::Overflow), "bad {bad} pos {pos}");
                        }
                    }
                }
            }
        }
    }

    /// Segment attribution of one prefill attention layer call (`k = 200`, `pos0 = 0`, the
    /// Qwen3.5-0.8B geometry: 8 query heads, 2 KV heads, head dim 256, rotary width 64, gated
    /// query) over the pool, mirroring the three stages of `softmax_core_block`.
    /// `cargo test --release --lib --locked attn_layer_segments -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing harness: run under the exclusive host lock"]
    fn attn_layer_segments_k200() {
        use crate::engine::chart::ffn::attn_heads;
        use crate::engine::chart::pool::Pool;
        use crate::engine::loader::qk_norm_heads;
        use crate::engine::rope::RopeTable;
        use crate::engine::step::attn_block::{rotate_pair_prefixes, sigmoid_gate};
        use crate::engine::value::mul_q32;
        use crate::numeric::fixed_ln::fixed_sigmoid;
        use std::time::Instant;

        let (k, nh, nkv, hd, rot) = (200usize, 8usize, 2usize, 256usize, 64usize);
        let (q_out, kv_out) = (nh * hd, nkv * hd);
        let stride = 2 * q_out + 2 * kv_out;
        let one = 1i64 << 32;
        let mut rng = SimpleRng::new(0xA77E_0001);
        let mut word =
            |lo: f32, hi: f32| (f64::from(rng.next_f32_range(lo, hi)) * 4_294_967_296.0) as i64;
        // Per column `[q | gate]` interleaved per head as the gated `attn_q` output, then k, v.
        let qfull: Vec<Vec<i64>> = (0..k)
            .map(|_| (0..2 * q_out).map(|_| word(-2.0, 2.0)).collect())
            .collect();
        let kraw: Vec<Vec<i64>> = (0..k)
            .map(|_| (0..kv_out).map(|_| word(-2.0, 2.0)).collect())
            .collect();
        let vraw: Vec<Vec<i64>> = (0..k)
            .map(|_| (0..kv_out).map(|_| word(-2.0, 2.0)).collect())
            .collect();
        let gain: Vec<i64> = (0..hd).map(|_| one + word(-0.1, 0.1)).collect();
        let eps = one / 1_000_000;
        let rope = RopeTable::new(10_000_000.0, rot).expect("rope");
        let mut kv = KvState::with_carrier(1, nh, nkv, hd, 256, Carrier::F32).expect("kv");
        let pool = Pool::global();

        let mut prep = vec![0i64; k * stride];
        let mut att = vec![0i64; k * q_out];
        let mut last = vec![0usize; k];
        let fold = |n: usize| if n % 2 == 0 { n / 2 } else { k - 1 - n / 2 };

        let stage1 = |prep: &mut [i64]| {
            let f = |c: usize, dst: &mut [i64]| -> Result<(), ()> {
                let (mut query, mut key, value) = (Vec::new(), kraw[c].clone(), vraw[c].clone());
                let mut gate = Vec::new();
                for h in 0..nh {
                    query.extend_from_slice(&qfull[c][h * 2 * hd..h * 2 * hd + hd]);
                    gate.extend_from_slice(&qfull[c][h * 2 * hd + hd..(h + 1) * 2 * hd]);
                }
                qk_norm_heads(Some(&gain), &mut query, hd, eps).map_err(|_| ())?;
                qk_norm_heads(Some(&gain), &mut key, hd, eps).map_err(|_| ())?;
                rotate_pair_prefixes(&rope, &mut query, &mut key, hd, c as u32).map_err(|_| ())?;
                let (dq, rest) = dst.split_at_mut(q_out);
                let (dk, rest) = rest.split_at_mut(kv_out);
                let (dv, dg) = rest.split_at_mut(kv_out);
                dq.copy_from_slice(&query);
                dk.copy_from_slice(&key);
                dv.copy_from_slice(&value);
                dg.copy_from_slice(&gate);
                Ok(())
            };
            assert!(attn_heads(k, stride, prep, true, f).is_none());
        };
        // Split of stage 1: norm only and rope only (same pool dispatch, same clones).
        let stage1_part = |prep: &mut [i64], do_norm: bool, do_rope: u8| {
            let f = |c: usize, dst: &mut [i64]| -> Result<(), ()> {
                let (mut query, mut key) = (qfull[c][..q_out].to_vec(), kraw[c].clone());
                if do_norm {
                    qk_norm_heads(Some(&gain), &mut query, hd, eps).map_err(|_| ())?;
                    qk_norm_heads(Some(&gain), &mut key, hd, eps).map_err(|_| ())?;
                }
                if do_rope == 1 {
                    let pos = c as u32;
                    for base in (0..q_out).step_by(hd) {
                        rope.rotate_one(&mut query[base..base + rot], pos)
                            .map_err(|_| ())?;
                    }
                    for base in (0..kv_out).step_by(hd) {
                        rope.rotate_one(&mut key[base..base + rot], pos)
                            .map_err(|_| ())?;
                    }
                } else if do_rope == 2 {
                    rotate_pair_prefixes(&rope, &mut query, &mut key, hd, c as u32)
                        .map_err(|_| ())?;
                }
                dst[..q_out].copy_from_slice(&query);
                dst[q_out..q_out + kv_out].copy_from_slice(&key);
                Ok(())
            };
            assert!(attn_heads(k, stride, prep, true, f).is_none());
        };
        let append_all = |kv: &mut KvState, prep: &[i64], last: &mut [usize]| {
            kv.truncate(0).unwrap();
            for c in 0..k {
                let b = c * stride;
                last[c] = kv
                    .append(
                        0,
                        &prep[b + q_out..b + q_out + kv_out],
                        &prep[b + q_out + kv_out..b + q_out + 2 * kv_out],
                    )
                    .unwrap();
            }
        };
        let attn_all =
            |kv: &KvState, prep: &[i64], last: &[usize], att: &mut [i64], gated: bool| {
                pool.chunks_mut(att, hd, &|t, _s, dst| {
                    let (n, h) = (t / nh, t % nh);
                    let c = fold(n);
                    let b = c * stride;
                    kv.attn_head(0, h, &prep[b + h * hd..b + (h + 1) * hd], last[c], dst)
                        .unwrap();
                    if gated {
                        let g0 = b + 2 * kv_out + q_out + h * hd;
                        sigmoid_gate(dst, &prep[g0..g0 + hd]).unwrap();
                    }
                });
            };
        let gate_only = |prep: &[i64], att: &mut [i64], fast: bool| {
            pool.chunks_mut(att, hd, &|t, _s, dst| {
                let (n, h) = (t / nh, t % nh);
                let b = fold(n) * stride;
                let g0 = b + 2 * kv_out + q_out + h * hd;
                if fast {
                    sigmoid_gate(dst, &prep[g0..g0 + hd]).unwrap();
                } else {
                    for i in 0..hd {
                        let s = fixed_sigmoid(prep[g0 + i]).unwrap();
                        dst[i] = mul_q32(dst[i], s).unwrap();
                    }
                }
            });
        };
        // core only: q cvt + online softmax, result discarded into a per-task stack buffer.
        let core_only =
            |kv: &KvState, prep: &[i64], last: &[usize], att: &mut [i64], readout: bool| {
                let (kp, vp) = match &kv.storage {
                    crate::engine::attn::KvStorage::F32 { k, v } => (&k[0][..], &v[0][..]),
                    crate::engine::attn::KvStorage::Q32 { .. } => unreachable!(),
                };
                pool.chunks_mut(att, hd, &|t, _s, dst| {
                    let (n, h) = (t / nh, t % nh);
                    let c = fold(n);
                    let b = c * stride;
                    let mut qf = [0.0f32; 256];
                    for i in 0..hd {
                        qf[i] = crate::engine::mixer::f32_carrier::q32_to_f32(prep[b + h * hd + i]);
                    }
                    let first = (h / (nh / nkv)) * hd;
                    let mut of = [0.0f32; 256];
                    attn_core_f32_strided(
                        &qf[..hd],
                        &kp[first..],
                        &vp[first..],
                        hd,
                        nkv * hd,
                        last[c],
                        1.0 / 16.0,
                        &mut of[..hd],
                    )
                    .unwrap();
                    if readout {
                        for i in 0..hd {
                            dst[i] = f32_to_q32_checked(of[i]).unwrap();
                        }
                    } else {
                        dst[0] = of[0].to_bits() as i64;
                    }
                });
            };
        let empty_dispatch = |att: &mut [i64]| {
            pool.chunks_mut(att, hd, &|t, _s, dst| {
                dst[0] = t as i64;
            });
        };
        let tail = |att: &[i64]| -> Vec<Vec<i64>> {
            let mut atts: Vec<Vec<i64>> = (0..k).map(|_| Vec::new()).collect();
            for n in 0..k {
                atts[fold(n)] = att[n * q_out..(n + 1) * q_out].to_vec();
            }
            atts
        };

        // One full composed call, in the stage order of `softmax_core_block`.
        let full =
            |kv: &mut KvState, prep: &mut Vec<i64>, att: &mut Vec<i64>, last: &mut Vec<usize>| {
                let mut p = vec![0i64; k * stride];
                stage1(&mut p);
                append_all(kv, &p, last);
                let mut a = vec![0i64; k * q_out];
                attn_all(kv, &p, last, &mut a, true);
                let t = tail(&a);
                std::hint::black_box(&t);
                *prep = p;
                *att = a;
            };

        fn med(v: &mut [f64]) -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        }
        let reps = 25usize;
        let mut rows: Vec<(&str, f64)> = Vec::new();
        macro_rules! seg {
            ($name:expr, $body:block) => {{
                let mut ts = Vec::new();
                for r in 0..reps + 3 {
                    let t0 = Instant::now();
                    $body;
                    if r >= 3 {
                        ts.push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                rows.push(($name, med(&mut ts)));
            }};
        }
        stage1(&mut prep);
        append_all(&mut kv, &prep, &mut last);
        seg!("full composed call", {
            full(&mut kv, &mut prep, &mut att, &mut last)
        });
        stage1(&mut prep);
        append_all(&mut kv, &prep, &mut last);
        seg!("stage1 (clone+split+qk norm+rope), pooled 200 tasks", {
            stage1(&mut prep)
        });
        seg!("  stage1 norm only", { stage1_part(&mut prep, true, 0) });
        seg!("  stage1 rope only (per-head fixed_sin_cos, old)", {
            stage1_part(&mut prep, false, 1)
        });
        seg!("  stage1 rope only (shared sin/cos, new)", {
            stage1_part(&mut prep, false, 2)
        });
        seg!("  stage1 clones/copies only", {
            stage1_part(&mut prep, false, 0)
        });
        seg!("prep calloc 200x5120 i64", {
            std::hint::black_box(vec![0i64; k * stride]);
        });
        seg!("stage2 kv append x200 (serial)", {
            append_all(&mut kv, &prep, &mut last)
        });
        seg!("stage3 attn (q cvt+core+readout), no gate", {
            attn_all(&kv, &prep, &last, &mut att, false)
        });
        seg!("  core only (q cvt + softmax), no readout", {
            core_only(&kv, &prep, &last, &mut att, false)
        });
        seg!("  core + readout", {
            core_only(&kv, &prep, &last, &mut att, true)
        });
        seg!("stage3 output gate only, oracle loop (old)", {
            gate_only(&prep, &mut att, false)
        });
        seg!("stage3 output gate only, sigmoid_gate (new)", {
            gate_only(&prep, &mut att, true)
        });
        seg!("stage3 attn + gate", {
            attn_all(&kv, &prep, &last, &mut att, true)
        });
        seg!("empty 1600-task dispatch", { empty_dispatch(&mut att) });
        seg!("att calloc 200x2048 i64", {
            std::hint::black_box(vec![0i64; k * q_out]);
        });
        seg!("tail: 200 to_vec gather", {
            std::hint::black_box(tail(&att));
        });
        println!("--- attn layer segments, k={k} pos0=0 (median of {reps}, ms/call) ---");
        let total = rows[0].1;
        for (n, ms) in &rows {
            println!("{ms:8.3} ms  {:5.1}%  {n}", 100.0 * ms / total);
        }
    }
}
