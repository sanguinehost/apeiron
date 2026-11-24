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

#![allow(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "bit-cast views are one instruction; the scalar and strided cores take the kernel operands"
)]

#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::wildcard_imports,
    reason = "x86_64 SIMD intrinsics imported en masse for AVX2+FMA"
)]
use std::arch::x86_64::*;

use std::sync::OnceLock;

use crate::engine::attn::AttnError;
#[cfg(target_arch = "x86_64")]
use crate::engine::mixer::f32_carrier::exp_f32_avx2;
use crate::engine::mixer::f32_carrier::exp_f32_scalar;
use crate::engine::num::fp::F32;

/// Native binary32 view of an [`F32`] for the AVX2 inners, which keep `f32` intrinsic bodies.
#[inline(always)]
fn nat(x: F32) -> f32 {
    f32::from_bits(x.to_bits())
}

/// [`F32`] view of a native binary32 value computed inside an AVX2 inner.
#[inline(always)]
fn fp(x: f32) -> F32 {
    F32::from_bits(x.to_bits())
}

/// Reinterprets a `&[F32]` as `&[f32]`. `F32` is `repr(transparent)` over `u32` with the layout of
/// `f32`, so the pointer and length carry over unchanged; every bit pattern is a valid `f32`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn native(s: &[F32]) -> &[f32] {
    // SAFETY: `F32` is `repr(transparent)` over `u32`, which has the size and alignment of `f32`;
    // the slice length counts the same elements.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<f32>(), s.len()) }
}

/// Mutable form of [`native`].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn native_mut(s: &mut [F32]) -> &mut [f32] {
    // SAFETY: as for `native`; the exclusive borrow is passed through unchanged.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<f32>(), s.len()) }
}

/// Convert a binary32 float back to Q32.32 fixed-point `i64` using round-to-nearest.
///
/// Refuses non-finite floats (NaN, ±Inf) and values outside the representable `i64` Q32.32 range
/// `[-2^31, 2^31)` with [`AttnError::Overflow`].
///
/// # Complexity
///
/// O(1) temporal and spatial: 1 float test, 1 multiplication, 1 rounding, 1 range check.
#[inline]
pub fn f32_to_q32_checked(v: F32) -> Result<i64, AttnError> {
    // `None` covers non-finite input and a result outside `[-2^63, 2^63)`: both were `Overflow`.
    v.to_q32().ok_or(AttnError::Overflow)
}

/// Compute scaled dot product `(q · k) · scale` in scalar arithmetic with fixed 8-lane reduction.
#[inline]
#[allow(
    clippy::needless_range_loop,
    reason = "fixed lane loop order pins execution order"
)]
fn dot_f32_scalar(q: &[F32], k: &[F32], scale: F32) -> F32 {
    let hd = q.len();
    if hd.is_multiple_of(8) {
        let mut lanes = [F32::ZERO; 8];
        let chunks = hd / 8;
        let mut c = 0usize;
        while c < chunks {
            let mut lane = 0usize;
            while lane < 8 {
                lanes[lane] = q[c * 8 + lane].fma(k[c * 8 + lane], lanes[lane]);
                lane += 1;
            }
            c += 1;
        }
        let raw =
            lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        raw * scale
    } else {
        let mut raw = F32::ZERO;
        let mut i = 0usize;
        while i < hd {
            raw = q[i].fma(k[i], raw);
            i += 1;
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
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: F32,
    out: &mut [F32],
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

    // Each branch lends one buffer to the recurrence: a join of a stack and a heap `&mut` is refused
    // by the pinned Aeneas model, so the buffer is a parameter of `attn_core_f32_scalar_rows`.
    if head_dim <= 256 {
        let mut acc_stack = [F32::ZERO; 256];
        attn_core_f32_scalar_rows(
            q,
            k,
            v,
            head_dim,
            stride,
            count,
            score0,
            scale,
            &mut acc_stack[..head_dim],
            out,
        )
    } else {
        let mut acc_heap = vec![F32::ZERO; head_dim];
        attn_core_f32_scalar_rows(
            q,
            k,
            v,
            head_dim,
            stride,
            count,
            score0,
            scale,
            &mut acc_heap[..],
            out,
        )
    }
}

/// Recurrence of [`attn_core_f32_strided_scalar`] after the first score, over a caller-owned
/// accumulator `acc` of length `head_dim`. Same operations, order and early exits as the body it was
/// split from.
#[allow(
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    reason = "mathematical recurrence symbols, fixed lane loop and explicit strided geometry"
)]
fn attn_core_f32_scalar_rows(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    stride: usize,
    count: usize,
    score0: F32,
    scale: F32,
    acc: &mut [F32],
    out: &mut [F32],
) -> Result<(), AttnError> {
    let mut m = score0;
    let mut den = F32::ONE;

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
            let m_next = if score_s.gt(m_prev) { score_s } else { m_prev };
            let alpha = exp_f32_scalar(m_prev - m_next);
            let w = exp_f32_scalar(score_s - m_next);

            den = den.fma(alpha, w);

            let mut i = 0usize;
            while i < head_dim && ok {
                let vi = v_row[i];
                if vi.is_finite() {
                    let prod = w * vi;
                    acc[i] = acc[i].fma(alpha, prod);
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

    if den.le(F32::ZERO) || !den.is_finite() {
        return Err(AttnError::ZeroSum);
    }

    let inv_den = F32::ONE / den;
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
            let alpha = nat(exp_f32_scalar(fp(m_prev - m_next)));
            let w = nat(exp_f32_scalar(fp(score_s - m_next)));

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
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    stride: usize,
    pos: usize,
    scale: F32,
    out: &mut [F32],
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
    if head_dim.is_multiple_of(8) && crate::engine::num::fp::prim::cpu_avx2_fma() {
        // SAFETY: `cpu_avx2_fma` checked AVX2 and FMA support on this processor.
        return unsafe {
            attn_core_f32_strided_avx2_inner(
                native(q),
                native(k),
                native(v),
                head_dim,
                stride,
                pos,
                nat(scale),
                native_mut(out),
            )
        };
    }
    attn_core_f32_strided_scalar(q, k, v, head_dim, stride, pos, scale, out)
}

/// Key positions scored in one pass of [`attn_core_f32_tile`]: the score stage of a tile holds
/// `nq · query_heads · KEY_CHUNK` values, so memory stays bounded as the context grows.
const KEY_CHUNK: usize = 256;

/// The `APEIRON_ATTN_EXP` flag: one `exp` per score unless set to `0`. A same-binary A/B switch
/// parsed once per process, the shape of `APEIRON_ATTN_TILE` in `step/softmax.rs`.
fn attn_exp_skip_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("APEIRON_ATTN_EXP").as_deref() != Ok("0"))
}

/// The softmax pair `(alpha, w) = (exp(m_prev − m_next), exp(score − m_next))` with
/// `m_next = max(m_prev, score)`, evaluating the one argument that is not `+0`.
///
/// Exactly one of the two differences is `+0.0`: `m_next` is `score` when the maximum moved and
/// `m_prev` otherwise, and both operands are finite, so the equal pair subtracts to `+0.0`
/// (`x − x = +0` in round-to-nearest, including `±0` operands). `exp_f32_scalar(+0.0)` returns
/// `1.0` bit-exactly: `z = +0·log₂e = +0`, `n = round_ties_even(+0) = +0`, `n_int = 0`,
/// `r = (+0 − +0) − +0 = +0`, `r² = +0`, `y = (1 + +0) + (+0·p) = 1`, `n1 = n2 = 0`, so
/// `s1 = s2 = 2⁰ = 1` and `(y·s1)·s2 = 1`. Returning [`F32::ONE`] for that argument therefore
/// feeds the same bit pattern to every consumer as the call it replaces, and the consumers keep
/// their words: `fma(1, y, a) = fl(1·y + a) = fl(y + a)` because `1·y` is exact, and `1·v = v`
/// exactly for the finite `v` the value-row check admits (so `w·v` and `fma(acc, alpha, w·v)` are
/// unchanged too). Source of the branch, not of the arithmetic: llama.cpp `049326a`
/// `ggml/src/ggml-cpu/ops.cpp:8546-8564`.
#[inline]
fn softmax_terms(score: F32, m_prev: F32, m_next: F32, skip: bool) -> (F32, F32) {
    if skip {
        if score.gt(m_prev) {
            (exp_f32_scalar(m_prev - m_next), F32::ONE)
        } else {
            (F32::ONE, exp_f32_scalar(score - m_next))
        }
    } else {
        (
            exp_f32_scalar(m_prev - m_next),
            exp_f32_scalar(score - m_next),
        )
    }
}

/// Causal scaled dot-product attention for a tile of query positions over one K/V plane: each key
/// and value row is loaded once per pass and folded against every query head of every position in
/// the tile.
///
/// Query head `g` of position `j` occupies `q[(j·query_heads + g)·head_dim..][..head_dim]`, has
/// last position `pos[j]` (strictly ascending, as consecutive causal columns are) and writes the
/// same slot of `out`. Each `(j, g)` runs exactly [`attn_core_f32_strided`]'s recurrence, in the
/// same order — the same 8-lane score chain, the same running maximum and `exp` values, the same
/// `mul_add` on the denominator and the same per-lane FMA on the accumulator, over the same
/// positions ascending — so every output word equals the per-query call's. The one `exp` whose
/// argument is exactly `+0` is replaced by its result `1` ([`softmax_terms`]). The loop order is
/// inverted: a key row is dotted against every query in the tile, and the scores of one
/// [`KEY_CHUNK`]-position pass are staged key-position-major so the value rows are read once too
/// and eight rows' weights are one 8-lane load.
///
/// # Errors
///
/// [`AttnError::BadGeometry`] for a shape mismatch or positions that are not strictly ascending,
/// [`AttnError::Overflow`] for a non-finite score, value or output, and [`AttnError::ZeroSum`] for
/// a non-positive or non-finite denominator. A refusal on the tile aborts the whole tile. The value
/// pass tests every row up to the last position, so a non-finite value row at a position no query
/// in the tile reaches still refuses the tile where those queries' single-query calls would
/// succeed; where a query does read the row, the tile may report a different query's fault first,
/// because its score pass covers the whole tile before the per-query denominators are tested. The
/// step discards the block on any refusal.
///
/// # Complexity
///
/// `O((pos[nq-1] + 1) · nq · query_heads · head_dim)` floating-point operations for the tile,
/// against `Σ_j (pos[j] + 1) · query_heads · head_dim` for the per-query calls — the same
/// arithmetic, with each K and V row read once per tile (`O((pos[nq-1] + 1) · head_dim)` words)
/// instead of once per query. Scratch `O(nq · query_heads · (KEY_CHUNK + head_dim))` words.
#[allow(
    clippy::too_many_arguments,
    reason = "explicit strided geometry and the four tile operands"
)]
pub fn attn_core_f32_tile(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    query_heads: usize,
    stride: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    if head_dim == 0 || query_heads == 0 || stride < head_dim || pos.is_empty() {
        return Err(AttnError::BadGeometry);
    }
    #[cfg(target_arch = "x86_64")]
    if head_dim.is_multiple_of(8) && crate::engine::num::fp::prim::cpu_avx2_fma() {
        // SAFETY: `cpu_avx2_fma` checked AVX2 and FMA support on this processor; the geometry is
        // checked again by the body it calls.
        return unsafe {
            attn_core_f32_tile_avx2_inner(
                native(q),
                native(k),
                native(v),
                head_dim,
                query_heads,
                stride,
                pos,
                nat(scale),
                native_mut(out),
            )
        };
    }
    attn_core_f32_tile_scalar(q, k, v, head_dim, query_heads, stride, pos, scale, out)
}

/// Scalar twin and specification of [`attn_core_f32_tile`]. Performs the same geometry checks.
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "mathematical recurrence symbols, explicit strided geometry, and the pass structure"
)]
pub fn attn_core_f32_tile_scalar(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    query_heads: usize,
    stride: usize,
    pos: &[usize],
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    if head_dim == 0 || query_heads == 0 || stride < head_dim || pos.is_empty() {
        return Err(AttnError::BadGeometry);
    }
    let nq = pos.len();
    let rows = nq.checked_mul(query_heads).ok_or(AttnError::Overflow)?;
    let need = rows.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    if q.len() != need || out.len() != need {
        return Err(AttnError::BadGeometry);
    }
    let mut j = 1usize;
    while j < nq {
        if pos[j] <= pos[j - 1] {
            return Err(AttnError::BadGeometry);
        }
        j += 1;
    }
    let required = pos[nq - 1]
        .checked_mul(stride)
        .and_then(|b| b.checked_add(head_dim))
        .ok_or(AttnError::Overflow)?;
    if k.len() < required || v.len() < required {
        return Err(AttnError::BadGeometry);
    }

    let score_len = rows.checked_mul(KEY_CHUNK).ok_or(AttnError::Overflow)?;
    let mut m = vec![F32::ZERO; rows];
    let mut den = vec![F32::ONE; rows];
    let mut acc = vec![F32::ZERO; need];
    // Staged key-position-major, `(at − s) · rows + row`: one key's scores over consecutive rows
    // are one contiguous run, which is the 8-lane load the value pass takes. The transpose is
    // internal to this module; the outputs are unchanged.
    let mut scores = vec![F32::ZERO; score_len];
    // Position 0: the first score and the first value row of every query head.
    let mut r = 0usize;
    while r < rows {
        let score = dot_f32_scalar(&q[r * head_dim..(r + 1) * head_dim], &k[..head_dim], scale);
        if !score.is_finite() {
            return Err(AttnError::Overflow);
        }
        m[r] = score;
        let mut i = 0usize;
        while i < head_dim {
            let vi = v[i];
            if !vi.is_finite() {
                return Err(AttnError::Overflow);
            }
            acc[r * head_dim + i] = vi;
            i += 1;
        }
        r += 1;
    }

    let last = pos[nq - 1];
    let skip = attn_exp_skip_enabled();
    let mut s = 1usize;
    while s <= last {
        let end = s.saturating_add(KEY_CHUNK).min(last + 1);
        // Queries with `pos < s` are inactive; positions ascend, so the active set is the suffix
        // `lo..nq` and `lo` only moves forward as `s` grows.
        let mut lo = 0usize;
        while lo < nq && pos[lo] < s {
            lo += 1;
        }
        // Score pass: each key row is loaded once and dotted against every active query head.
        let mut at = s;
        while at < end {
            let krow = &k[at * stride..at * stride + head_dim];
            let mut j = lo;
            while j < nq && pos[j] < at {
                j += 1;
            }
            while j < nq {
                let qrow = &q[j * query_heads * head_dim..(j + 1) * query_heads * head_dim];
                let mut g = 0usize;
                while g < query_heads {
                    let row = j * query_heads + g;
                    let score =
                        dot_f32_scalar(&qrow[g * head_dim..(g + 1) * head_dim], krow, scale);
                    if !score.is_finite() {
                        return Err(AttnError::Overflow);
                    }
                    scores[(at - s) * rows + row] = score;
                    g += 1;
                }
                j += 1;
            }
            at += 1;
        }
        // Value pass: the staged scores drive the same recurrence, one value row read per position.
        let mut at = s;
        while at < end {
            let vrow = &v[at * stride..at * stride + head_dim];
            let mut i = 0usize;
            while i < head_dim {
                if !vrow[i].is_finite() {
                    return Err(AttnError::Overflow);
                }
                i += 1;
            }
            let mut j = lo;
            while j < nq && pos[j] < at {
                j += 1;
            }
            while j < nq {
                let mut g = 0usize;
                while g < query_heads {
                    let row = j * query_heads + g;
                    let score = scores[(at - s) * rows + row];
                    let m_prev = m[row];
                    let m_next = if score.gt(m_prev) { score } else { m_prev };
                    let (alpha, w) = softmax_terms(score, m_prev, m_next, skip);
                    den[row] = den[row].fma(alpha, w);
                    let mut i = 0usize;
                    while i < head_dim {
                        let a = &mut acc[row * head_dim + i];
                        *a = a.fma(alpha, w * vrow[i]);
                        i += 1;
                    }
                    m[row] = m_next;
                    g += 1;
                }
                j += 1;
            }
            at += 1;
        }
        s = end;
    }

    let mut row = 0usize;
    while row < rows {
        if den[row].le(F32::ZERO) || !den[row].is_finite() {
            return Err(AttnError::ZeroSum);
        }
        let inv_den = F32::ONE / den[row];
        let mut i = 0usize;
        while i < head_dim {
            let res = acc[row * head_dim + i] * inv_den;
            if !res.is_finite() {
                return Err(AttnError::Overflow);
            }
            out[row * head_dim + i] = res;
            i += 1;
        }
        row += 1;
    }
    Ok(())
}

/// AVX2+FMA body of [`attn_core_f32_tile`]; the caller has checked the geometry and the feature.
///
/// Each score is [`dot_f32_avx2`]'s chain, equal to [`dot_f32_scalar`] bit for bit, and each
/// accumulator lane is the scalar body's `fma(acc, alpha, w·v)` in `_mm256_fmadd_ps`, so the tile's
/// words equal the scalar twin's and the per-query core's. Eight consecutive rows' `w` arguments
/// are one contiguous staged run and one [`exp_f32_avx2`] call, which equals eight
/// [`exp_f32_scalar`] calls lane for lane.
///
/// # Safety
///
/// Requires AVX2 and FMA; `q`, `k` and `v` hold the geometry the body checks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "mathematical recurrence symbols, explicit strided geometry, and the pass structure"
)]
unsafe fn attn_core_f32_tile_avx2_inner(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    query_heads: usize,
    stride: usize,
    pos: &[usize],
    scale: f32,
    out: &mut [f32],
) -> Result<(), AttnError> {
    if head_dim == 0 || query_heads == 0 || stride < head_dim || pos.is_empty() {
        return Err(AttnError::BadGeometry);
    }
    let nq = pos.len();
    let rows = nq.checked_mul(query_heads).ok_or(AttnError::Overflow)?;
    let need = rows.checked_mul(head_dim).ok_or(AttnError::Overflow)?;
    if q.len() != need || out.len() != need {
        return Err(AttnError::BadGeometry);
    }
    let mut j = 1usize;
    while j < nq {
        if pos[j] <= pos[j - 1] {
            return Err(AttnError::BadGeometry);
        }
        j += 1;
    }
    let required = pos[nq - 1]
        .checked_mul(stride)
        .and_then(|b| b.checked_add(head_dim))
        .ok_or(AttnError::Overflow)?;
    if k.len() < required || v.len() < required {
        return Err(AttnError::BadGeometry);
    }

    let chunks = head_dim / 8;
    let score_len = rows.checked_mul(KEY_CHUNK).ok_or(AttnError::Overflow)?;
    let mut m = vec![0.0f32; rows];
    let mut den = vec![1.0f32; rows];
    let mut acc = vec![0.0f32; need];
    let mut scores = vec![0.0f32; score_len];
    let mut r = 0usize;
    while r < rows {
        let score = dot_f32_avx2(&q[r * head_dim..(r + 1) * head_dim], &k[..head_dim], scale);
        if !score.is_finite() {
            return Err(AttnError::Overflow);
        }
        m[r] = score;
        // The scalar twin tests every element of position 0; check the row before the vector copy so
        // the same non-finite value refuses on both bodies.
        let mut i = 0usize;
        while i < head_dim {
            if !v[i].is_finite() {
                return Err(AttnError::Overflow);
            }
            i += 1;
        }
        let mut c = 0usize;
        while c < chunks {
            let vv = _mm256_loadu_ps(v.as_ptr().add(c * 8));
            _mm256_storeu_ps(acc.as_mut_ptr().add(r * head_dim + c * 8), vv);
            c += 1;
        }
        let mut i = chunks * 8;
        while i < head_dim {
            acc[r * head_dim + i] = v[i];
            i += 1;
        }
        r += 1;
    }

    let last = pos[nq - 1];
    let skip = attn_exp_skip_enabled();
    let mut s = 1usize;
    while s <= last {
        let end = s.saturating_add(KEY_CHUNK).min(last + 1);
        let mut lo = 0usize;
        while lo < nq && pos[lo] < s {
            lo += 1;
        }
        let mut at = s;
        while at < end {
            let krow = &k[at * stride..at * stride + head_dim];
            let mut j = lo;
            while j < nq && pos[j] < at {
                j += 1;
            }
            while j < nq {
                let qrow = &q[j * query_heads * head_dim..(j + 1) * query_heads * head_dim];
                let mut g = 0usize;
                while g < query_heads {
                    let row = j * query_heads + g;
                    let score = dot_f32_avx2(&qrow[g * head_dim..(g + 1) * head_dim], krow, scale);
                    if !score.is_finite() {
                        return Err(AttnError::Overflow);
                    }
                    scores[(at - s) * rows + row] = score;
                    g += 1;
                }
                j += 1;
            }
            at += 1;
        }
        let mut at = s;
        while at < end {
            let vrow = &v[at * stride..at * stride + head_dim];
            let mut i = 0usize;
            while i < head_dim {
                if !vrow[i].is_finite() {
                    return Err(AttnError::Overflow);
                }
                i += 1;
            }
            let mut j = lo;
            while j < nq && pos[j] < at {
                j += 1;
            }
            let stage = (at - s) * rows;
            let mut row = j * query_heads;
            // With the flag on, eight rows at a time, one 8-lane `exp` for the eight `w`
            // arguments: the staged scores of one key over consecutive rows are contiguous.
            // `alpha` is `exp(m_prev − m_next)`, `1.0` on every lane the maximum did not move, so
            // its vector `exp` is skipped whole-group when no lane moved
            // (`_mm256_movemask_ps == 0`, the common case) and computed lane-wise otherwise, where
            // `exp(+0) = 1` gives the unmoved lanes their exact `1.0`. `maxps(a, b) = a > b ? a :
            // b` is the scalar `if score > m_prev { score } else { m_prev }` for the finite inputs
            // the score stage admitted, and the stored `m` keeps the same words. With the flag
            // off, the row loop below runs the old two-`exp_f32_scalar` recurrence for every row.
            let mut w_lane = [0.0f32; 8];
            let mut a_lane = [0.0f32; 8];
            if skip {
                while row + 8 <= rows {
                    let v_s = _mm256_loadu_ps(scores.as_ptr().add(stage + row));
                    let v_prev = _mm256_loadu_ps(m.as_ptr().add(row));
                    let v_next = _mm256_max_ps(v_s, v_prev);
                    let v_w = exp_f32_avx2(_mm256_sub_ps(v_s, v_next));
                    let v_alpha = if _mm256_movemask_ps(_mm256_cmp_ps(v_s, v_prev, _CMP_GT_OQ)) != 0
                    {
                        exp_f32_avx2(_mm256_sub_ps(v_prev, v_next))
                    } else {
                        _mm256_set1_ps(1.0)
                    };
                    _mm256_storeu_ps(w_lane.as_mut_ptr(), v_w);
                    _mm256_storeu_ps(a_lane.as_mut_ptr(), v_alpha);
                    _mm256_storeu_ps(m.as_mut_ptr().add(row), v_next);
                    let mut lane = 0usize;
                    while lane < 8 {
                        let r = row + lane;
                        let alpha = a_lane[lane];
                        let w = w_lane[lane];
                        den[r] = den[r].mul_add(alpha, w);
                        let v_alpha_s = _mm256_set1_ps(alpha);
                        let v_w_s = _mm256_set1_ps(w);
                        let mut c = 0usize;
                        while c < chunks {
                            let vv = _mm256_loadu_ps(vrow.as_ptr().add(c * 8));
                            let prod = _mm256_mul_ps(v_w_s, vv);
                            let a = _mm256_loadu_ps(acc.as_ptr().add(r * head_dim + c * 8));
                            let next = _mm256_fmadd_ps(a, v_alpha_s, prod);
                            _mm256_storeu_ps(acc.as_mut_ptr().add(r * head_dim + c * 8), next);
                            c += 1;
                        }
                        let mut i = chunks * 8;
                        while i < head_dim {
                            let a = acc[r * head_dim + i];
                            acc[r * head_dim + i] = a.mul_add(alpha, w * vrow[i]);
                            i += 1;
                        }
                        lane += 1;
                    }
                    row += 8;
                }
            }
            while row < rows {
                let score = scores[stage + row];
                let m_prev = m[row];
                let m_next = if score > m_prev { score } else { m_prev };
                let (alpha, w) = softmax_terms(fp(score), fp(m_prev), fp(m_next), skip);
                let (alpha, w) = (nat(alpha), nat(w));
                den[row] = den[row].mul_add(alpha, w);
                let v_alpha = _mm256_set1_ps(alpha);
                let v_w = _mm256_set1_ps(w);
                let mut c = 0usize;
                while c < chunks {
                    let vv = _mm256_loadu_ps(vrow.as_ptr().add(c * 8));
                    let prod = _mm256_mul_ps(v_w, vv);
                    let a = _mm256_loadu_ps(acc.as_ptr().add(row * head_dim + c * 8));
                    let next = _mm256_fmadd_ps(a, v_alpha, prod);
                    _mm256_storeu_ps(acc.as_mut_ptr().add(row * head_dim + c * 8), next);
                    c += 1;
                }
                let mut i = chunks * 8;
                while i < head_dim {
                    let a = acc[row * head_dim + i];
                    acc[row * head_dim + i] = a.mul_add(alpha, w * vrow[i]);
                    i += 1;
                }
                m[row] = m_next;
                row += 1;
            }
            at += 1;
        }
        s = end;
    }

    let mut row = 0usize;
    while row < rows {
        if den[row] <= 0.0 || !den[row].is_finite() {
            return Err(AttnError::ZeroSum);
        }
        let inv_den = 1.0f32 / den[row];
        let v_inv = _mm256_set1_ps(inv_den);
        let mut c = 0usize;
        while c < chunks {
            let a = _mm256_loadu_ps(acc.as_ptr().add(row * head_dim + c * 8));
            let res = _mm256_mul_ps(a, v_inv);
            _mm256_storeu_ps(out.as_mut_ptr().add(row * head_dim + c * 8), res);
            c += 1;
        }
        let mut i = chunks * 8;
        while i < head_dim {
            let res = acc[row * head_dim + i] * inv_den;
            if !res.is_finite() {
                return Err(AttnError::Overflow);
            }
            out[row * head_dim + i] = res;
            i += 1;
        }
        // Scan this row before the next: a row with a non-finite output returns `Overflow` here,
        // and a later row's `ZeroSum` must not pre-empt it. The scalar twin checks each row in
        // order, so this is the per-column call's first error, not the tile's own preference.
        let base = row * head_dim;
        let mut i = 0usize;
        while i < head_dim {
            if !out[base + i].is_finite() {
                return Err(AttnError::Overflow);
            }
            i += 1;
        }
        row += 1;
    }
    Ok(())
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
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    pos: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    attn_core_f32_strided(q, k, v, head_dim, head_dim, pos, scale, out)
}

/// Scalar twin for contiguous attention core (`stride = head_dim`).
#[allow(
    dead_code,
    reason = "public scalar twin specification for contiguous layout"
)]
pub fn attn_core_f32_scalar(
    q: &[F32],
    k: &[F32],
    v: &[F32],
    head_dim: usize,
    pos: usize,
    scale: F32,
    out: &mut [F32],
) -> Result<(), AttnError> {
    attn_core_f32_strided_scalar(q, k, v, head_dim, head_dim, pos, scale, out)
}

#[cfg(test)]
#[path = "f32_core/tests.rs"]
pub(crate) mod tests;
