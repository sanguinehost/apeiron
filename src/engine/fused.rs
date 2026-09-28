//! Fused block-wise projection: [`matvec_fused`], value-identical to [`matvec_fixed`].
//!
//! For a K-quant block the Q32.32 weight is `w_i = d·sc_g·q_i` (`Q6_K`, `Q3_K`) or
//! `d·sc_g·q_i − dmin·m_g` (`Q4_K`, `Q2_K`); for an 0-quant block `w_i = d·q_i` (`Q8_0`, `Q5_0`).
//! By distributivity in exact `i128` arithmetic the row dot `Σ_i w_i·x_i` equals
//! `Σ_g (d·sc_g)·(Σ_{i∈g} q_i·x_i) − Σ_g (dmin·m_g)·(Σ_{i∈g} x_i)`. The kernel reads the small
//! codes directly and multiplies once per group instead of materializing every Q32.32 weight and
//! multiplying per element. The reduction is the same exact integer sum, hence the same single
//! [`round_shift_32`] epilogue and the same `Ok`/`Err` value for every input.
//!
//! **Fallback.** The reference dequantizer computes each `w_i` in checked `i64` and accumulates in
//! `i128`. The fused kernel accumulates in `i128` too, but its group order differs. Whenever a
//! per-group bound cannot prove the reference's `i64` products and the reference's `i128` partial
//! sums stay in range — a non-finite scale, an unrepresentable weight magnitude, a bound that
//! leaves `i128`, or any checked overflow — the call falls back to [`matvec_fixed`] for the whole
//! payload and returns that result unchanged, so no refusal can differ.

use crate::format::quant::{
    self, Q2_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, Q5_0_BLOCK_BYTES,
    Q5_0_QK, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_0_QK, QK_K,
};

use super::dequant::f16_to_fixed;
use super::matvec::{matvec_fixed, MatvecError};
use super::value::round_shift_32;

/// The magnitude of an `i64` as a non-negative `i128` (`i128::MIN` cannot arise from `i64`).
pub(super) fn abs_i128(v: i64) -> i128 {
    let w = i128::from(v);
    if w < 0 {
        -w
    } else {
        w
    }
}

/// The little-endian `u16` at byte offset `i`, or `None` when fewer than two bytes remain.
fn read_u16_at(raw: &[u8], i: usize) -> Option<u16> {
    let bytes: [u8; 2] = raw
        .get(i..i + 2)
        .and_then(|s| <[u8; 2]>::try_from(s).ok())?;
    Some(u16::from_le_bytes(bytes))
}

/// `Σ_{l<take} codes[base+l]·x[xbase+l]` exactly, or `None` when the checked `i128` sum leaves
/// range. The per-element product cannot leave `i128` (`|code| ≤ 128`, `|x| ≤ 2^63`).
///
/// # Complexity
///
/// `O(take)` time, `O(1)` extra space.
fn group_dot(codes: &[i8], base: usize, x: &[i64], xbase: usize, take: usize) -> Option<i128> {
    let mut acc: i128 = 0;
    let mut l = 0usize;
    let mut ok = true;
    while l < take && ok {
        let p = i128::from(codes[base + l]) * i128::from(x[xbase + l]);
        match acc.checked_add(p) {
            Some(total) => acc = total,
            None => ok = false,
        }
        l += 1;
    }
    if ok {
        Some(acc)
    } else {
        None
    }
}

/// `Σ_{l<take} x[xbase+l]` exactly, or `None` when the checked `i128` sum leaves range.
///
/// # Complexity
///
/// `O(take)` time, `O(1)` extra space.
fn group_xsum(x: &[i64], xbase: usize, take: usize) -> Option<i128> {
    let mut acc: i128 = 0;
    let mut l = 0usize;
    let mut ok = true;
    while l < take && ok {
        match acc.checked_add(i128::from(x[xbase + l])) {
            Some(total) => acc = total,
            None => ok = false,
        }
        l += 1;
    }
    if ok {
        Some(acc)
    } else {
        None
    }
}

/// The largest `|code|` over `count` codes starting at `base`. `-128` maps to `128`, in `i64`.
///
/// # Complexity
///
/// `O(count)` time, `O(1)` extra space.
pub(super) fn max_abs_code(codes: &[i8], base: usize, count: usize) -> i64 {
    let mut m: i64 = 0;
    let mut l = 0usize;
    while l < count {
        let q = i64::from(codes[base + l]);
        let a = if q < 0 { -q } else { q };
        if a > m {
            m = a;
        }
        l += 1;
    }
    m
}

/// One pure group `w = slope·q`: its contribution `slope·Σ q·x` to the row sum and the weight
/// magnitude bound `|slope|·max|q|`, or `None` on a checked overflow.
///
/// # Complexity
///
/// `O(take + count)` time, `O(1)` extra space.
fn pure_group(
    slope: i64,
    codes: &[i8],
    base: usize,
    x: &[i64],
    xbase: usize,
    take: usize,
    count: usize,
) -> Option<(i128, i128)> {
    let dot = group_dot(codes, base, x, xbase, take)?;
    let factor = abs_i128(slope).checked_mul(i128::from(max_abs_code(codes, base, count)))?;
    let contribution = i128::from(slope).checked_mul(dot)?;
    Some((contribution, factor))
}

/// One affine group `w = slope·q − intercept`: its contribution `slope·Σ q·x − intercept·Σ x` to
/// the row sum and the weight magnitude bound `|slope|·max|q| + |intercept|`, or `None` on a
/// checked overflow.
///
/// # Complexity
///
/// `O(take + count)` time, `O(1)` extra space.
#[allow(
    clippy::too_many_arguments,
    reason = "the slope/intercept, the code window and the matching activation window are the \
              group's full input; bundling them would hide which window is read"
)]
fn affine_group(
    slope: i64,
    intercept: i64,
    codes: &[i8],
    base: usize,
    x: &[i64],
    xbase: usize,
    take: usize,
    count: usize,
) -> Option<(i128, i128)> {
    let dot = group_dot(codes, base, x, xbase, take)?;
    let xs = group_xsum(x, xbase, take)?;
    let factor = abs_i128(slope)
        .checked_mul(i128::from(max_abs_code(codes, base, count)))?
        .checked_add(abs_i128(intercept))?;
    let a = i128::from(slope).checked_mul(dot)?;
    let b = i128::from(intercept).checked_mul(xs)?;
    let contribution = a.checked_sub(b)?;
    Some((contribution, factor))
}

/// Fold one group's `(contribution, factor)` into the running `(total, wmax)`, clearing `ok` on a
/// checked overflow.
pub(super) fn fold_group(
    total: &mut i128,
    wmax: &mut i128,
    ok: &mut bool,
    group: Option<(i128, i128)>,
) {
    match group {
        Some((contribution, factor)) => {
            match total.checked_add(contribution) {
                Some(t) => *total = t,
                None => *ok = false,
            }
            if factor > *wmax {
                *wmax = factor;
            }
        }
        None => *ok = false,
    }
}

/// Close a row: `None` unless every group was safe and `wmax·Σ|x| ≤ i128::MAX`, which bounds the
/// reference's sequential partial sums (`|w_i| ≤ wmax`) and so its accumulator cannot overflow
/// where this kernel returns a value.
pub(super) fn finish_row(ok: bool, total: i128, wmax: i128, x_abs: i128) -> Option<i128> {
    if !ok {
        return None;
    }
    if wmax > i128::from(i64::MAX) {
        return None;
    }
    wmax.checked_mul(x_abs)?;
    Some(total)
}

/// The Q32.32 weight magnitudes of `x` summed: `Σ_i |x_i|`, or `None` on a checked overflow.
///
/// # Complexity
///
/// `O(x.len())` time, `O(1)` extra space.
pub(super) fn x_magnitude(x: &[i64]) -> Option<i128> {
    let mut acc: i128 = 0;
    let mut i = 0usize;
    let mut ok = true;
    while i < x.len() && ok {
        let a = i128::from(x[i]);
        let mag = if a < 0 { -a } else { a };
        match acc.checked_add(mag) {
            Some(total) => acc = total,
            None => ok = false,
        }
        i += 1;
    }
    if ok {
        Some(acc)
    } else {
        None
    }
}

/// `Q8_0` block fields `(d, codes)`, or `None` on a malformed or non-finite block.
pub(super) fn q8_0_fields(row: &[u8], block: usize) -> Option<(i64, [i8; Q8_0_QK])> {
    let scale = f16_to_fixed(read_u16_at(row, block * Q8_0_BLOCK_BYTES)?).ok()?;
    let codes = quant::q8_0_codes(row, block).ok()?;
    Some((scale, codes))
}

/// `Q5_0` block fields `(d, codes)`, or `None` on a malformed or non-finite block.
pub(super) fn q5_0_fields(row: &[u8], block: usize) -> Option<(i64, [i8; Q5_0_QK])> {
    let off = block * Q5_0_BLOCK_BYTES;
    let blk = row.get(off..off + Q5_0_BLOCK_BYTES)?;
    let scale = f16_to_fixed(quant::q5_0_d(blk).ok()?).ok()?;
    let codes = quant::q5_0_codes(blk).ok()?;
    Some((scale, codes))
}

/// `Q2_K` block fields `(d, dmin, scales, codes)`, or `None` on a malformed or non-finite block.
///
/// The two-bit codes are `0..=3`, widened to `i8` here so every group helper takes one code type.
#[allow(
    clippy::cast_possible_wrap,
    reason = "the two-bit codes are 0..=3; the wide-to-i8 cast cannot wrap"
)]
pub(super) fn q2_k_fields(row: &[u8], block: usize) -> Option<(i64, i64, [u8; 16], [i8; QK_K])> {
    let off = block * Q2_K_BLOCK_BYTES;
    let blk = row.get(off..off + Q2_K_BLOCK_BYTES)?;
    let d = f16_to_fixed(quant::q2_k_d(blk).ok()?).ok()?;
    let dmin = f16_to_fixed(quant::q2_k_dmin(blk).ok()?).ok()?;
    let scales = quant::q2_k_scales(blk).ok()?;
    let raw_codes = quant::q2_k_codes(blk).ok()?;
    let mut codes = [0i8; QK_K];
    let mut l = 0usize;
    while l < QK_K {
        codes[l] = raw_codes[l] as i8;
        l += 1;
    }
    Some((d, dmin, scales, codes))
}

/// `Q3_K` block fields `(d, scales, codes)`, or `None` on a malformed or non-finite block.
pub(super) fn q3_k_fields(row: &[u8], block: usize) -> Option<(i64, [i8; 16], [i8; QK_K])> {
    let off = block * Q3_K_BLOCK_BYTES;
    let blk = row.get(off..off + Q3_K_BLOCK_BYTES)?;
    let d = f16_to_fixed(quant::q3_k_d(blk).ok()?).ok()?;
    let scales = quant::q3_k_scales(blk).ok()?;
    let codes = quant::q3_k_codes(blk).ok()?;
    Some((d, scales, codes))
}

/// `Q4_K` block fields `(scale, min_scale, packed scales, codes)`, or `None` on a malformed or
/// non-finite block.
///
/// The 12 packed scale bytes are collected here rather than through
/// [`quant::q4_k_scale_mins`](crate::format::quant::q4_k_scale_mins): that accessor's
/// `[(u8, u8); 8]` array repeat lowers to `core::array::repeat`, which the pinned Charon fragment
/// leaves opaque and the pinned Aeneas `FunsAnalysis` pass rejects.
#[allow(
    clippy::cast_possible_wrap,
    reason = "the four-bit codes are 0..=15; the wide-to-i8 cast cannot wrap"
)]
pub(super) fn q4_k_fields(
    row: &[u8],
    block: usize,
) -> Option<(i64, i64, [u8; Q4_K_SCALE_SIZE], [i8; QK_K])> {
    let off = block * Q4_K_BLOCK_BYTES;
    let blk = row.get(off..off + Q4_K_BLOCK_BYTES)?;
    let scale = f16_to_fixed(read_u16_at(blk, 0)?).ok()?;
    let min_scale = f16_to_fixed(read_u16_at(blk, 2)?).ok()?;
    let mut sc = [0u8; Q4_K_SCALE_SIZE];
    let mut k = 0usize;
    while k < Q4_K_SCALE_SIZE {
        sc[k] = blk[4 + k];
        k += 1;
    }
    let raw_codes = quant::q4_k_codes(blk).ok()?;
    let mut codes = [0i8; QK_K];
    let mut l = 0usize;
    while l < QK_K {
        codes[l] = raw_codes[l] as i8;
        l += 1;
    }
    Some((scale, min_scale, sc, codes))
}

/// `Q6_K` block fields `(d, scales, codes)`, or `None` on a malformed or non-finite block.
pub(super) fn q6_k_fields(row: &[u8], block: usize) -> Option<(i64, [i8; 16], [i8; QK_K])> {
    let off = block * Q6_K_BLOCK_BYTES;
    let blk = row.get(off..off + Q6_K_BLOCK_BYTES)?;
    let d = f16_to_fixed(quant::q6_k_d(blk).ok()?).ok()?;
    let scales = quant::q6_k_scales(blk).ok()?;
    let codes = quant::q6_k_codes(blk).ok()?;
    Some((d, scales, codes))
}

/// Lane count still inside the row for a group starting at `gbase` in a block of `take` used lanes.
pub(super) fn group_take(take: usize, gbase: usize, span: usize) -> usize {
    if gbase >= take {
        0
    } else {
        (take - gbase).min(span)
    }
}

/// `Q8_0` fused row: `w = d·q`, one group of 32 codes per block.
fn fused_q8_0_row(row: &[u8], n_in: usize, x: &[i64], x_abs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(Q8_0_QK);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut block = 0usize;
    let mut ok = true;
    while block < blocks && ok {
        match q8_0_fields(row, block) {
            Some((scale, codes)) => {
                let xbase = block * Q8_0_QK;
                let take = Q8_0_QK.min(n_in - xbase);
                fold_group(
                    &mut total,
                    &mut wmax,
                    &mut ok,
                    pure_group(scale, &codes, 0, x, xbase, take, Q8_0_QK),
                );
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, x_abs)
}

/// `Q5_0` fused row: `w = d·q`, one group of 32 signed codes per block.
fn fused_q5_0_row(row: &[u8], n_in: usize, x: &[i64], x_abs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(Q5_0_QK);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut block = 0usize;
    let mut ok = true;
    while block < blocks && ok {
        match q5_0_fields(row, block) {
            Some((scale, codes)) => {
                let xbase = block * Q5_0_QK;
                let take = Q5_0_QK.min(n_in - xbase);
                fold_group(
                    &mut total,
                    &mut wmax,
                    &mut ok,
                    pure_group(scale, &codes, 0, x, xbase, take, Q5_0_QK),
                );
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, x_abs)
}

/// `Q2_K` fused row: `w = d·sc·q − dmin·m`, sixteen 16-weight affine groups per block.
fn fused_q2_k_row(row: &[u8], n_in: usize, x: &[i64], x_abs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(QK_K);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut block = 0usize;
    let mut ok = true;
    while block < blocks && ok {
        match q2_k_fields(row, block) {
            Some((d, dmin, scales, codes)) => {
                let xblock = block * QK_K;
                let take = QK_K.min(n_in - xblock);
                let mut group = 0usize;
                while group < 16 && ok {
                    let (scale_code, min_code) = quant::q2_k_scale_min(scales[group]);
                    let slope = if let Some(s) = d.checked_mul(i64::from(scale_code)) {
                        s
                    } else {
                        ok = false;
                        0
                    };
                    let intercept = if let Some(s) = dmin.checked_mul(i64::from(min_code)) {
                        s
                    } else {
                        ok = false;
                        0
                    };
                    if ok {
                        let gbase = 16 * group;
                        fold_group(
                            &mut total,
                            &mut wmax,
                            &mut ok,
                            affine_group(
                                slope,
                                intercept,
                                &codes,
                                gbase,
                                x,
                                xblock + gbase,
                                group_take(take, gbase, 16),
                                16,
                            ),
                        );
                    }
                    group += 1;
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, x_abs)
}

/// `Q3_K` fused row: `w = d·(sc − 32)·q`, sixteen 16-weight pure groups per block.
fn fused_q3_k_row(row: &[u8], n_in: usize, x: &[i64], x_abs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(QK_K);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut block = 0usize;
    let mut ok = true;
    while block < blocks && ok {
        match q3_k_fields(row, block) {
            Some((d, scales, codes)) => {
                let xblock = block * QK_K;
                let take = QK_K.min(n_in - xblock);
                let mut group = 0usize;
                while group < 16 && ok {
                    let slope = if let Some(s) = d.checked_mul(i64::from(scales[group]) - 32) {
                        s
                    } else {
                        ok = false;
                        0
                    };
                    if ok {
                        let gbase = 16 * group;
                        fold_group(
                            &mut total,
                            &mut wmax,
                            &mut ok,
                            pure_group(
                                slope,
                                &codes,
                                gbase,
                                x,
                                xblock + gbase,
                                group_take(take, gbase, 16),
                                16,
                            ),
                        );
                    }
                    group += 1;
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, x_abs)
}

/// `Q4_K` fused row: `w = d·sc·q − dmin·m`, four 64-weight pairs of two 32-weight affine groups.
fn fused_q4_k_row(row: &[u8], n_in: usize, x: &[i64], x_abs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(QK_K);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut block = 0usize;
    let mut ok = true;
    while block < blocks && ok {
        match q4_k_fields(row, block) {
            Some((scale, min_scale, sc, codes)) => {
                let xblock = block * QK_K;
                let take = QK_K.min(n_in - xblock);
                let mut pair = 0usize;
                while pair < 4 && ok {
                    let (scale_lo, min_lo) = quant::q4_k_scale_min(&sc, 2 * pair);
                    let (scale_hi, min_hi) = quant::q4_k_scale_min(&sc, 2 * pair + 1);
                    let slope_lo = if let Some(s) = scale.checked_mul(i64::from(scale_lo)) {
                        s
                    } else {
                        ok = false;
                        0
                    };
                    let intercept_lo = if let Some(s) = min_scale.checked_mul(i64::from(min_lo)) {
                        s
                    } else {
                        ok = false;
                        0
                    };
                    let slope_hi = if let Some(s) = scale.checked_mul(i64::from(scale_hi)) {
                        s
                    } else {
                        ok = false;
                        0
                    };
                    let intercept_hi = if let Some(s) = min_scale.checked_mul(i64::from(min_hi)) {
                        s
                    } else {
                        ok = false;
                        0
                    };
                    if ok {
                        let lo_base = 64 * pair;
                        fold_group(
                            &mut total,
                            &mut wmax,
                            &mut ok,
                            affine_group(
                                slope_lo,
                                intercept_lo,
                                &codes,
                                lo_base,
                                x,
                                xblock + lo_base,
                                group_take(take, lo_base, 32),
                                32,
                            ),
                        );
                        let hi_base = 64 * pair + 32;
                        fold_group(
                            &mut total,
                            &mut wmax,
                            &mut ok,
                            affine_group(
                                slope_hi,
                                intercept_hi,
                                &codes,
                                hi_base,
                                x,
                                xblock + hi_base,
                                group_take(take, hi_base, 32),
                                32,
                            ),
                        );
                    }
                    pair += 1;
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, x_abs)
}

/// `Q6_K` fused row: `w = d·sc·q`, sixteen 16-weight pure groups per block.
fn fused_q6_k_row(row: &[u8], n_in: usize, x: &[i64], x_abs: i128) -> Option<i128> {
    let blocks = n_in.div_ceil(QK_K);
    let mut total: i128 = 0;
    let mut wmax: i128 = 0;
    let mut block = 0usize;
    let mut ok = true;
    while block < blocks && ok {
        match q6_k_fields(row, block) {
            Some((d, scales, codes)) => {
                let xblock = block * QK_K;
                let take = QK_K.min(n_in - xblock);
                let mut group = 0usize;
                while group < 16 && ok {
                    let slope = if let Some(s) = d.checked_mul(i64::from(scales[group])) {
                        s
                    } else {
                        ok = false;
                        0
                    };
                    if ok {
                        let gbase = 16 * group;
                        fold_group(
                            &mut total,
                            &mut wmax,
                            &mut ok,
                            pure_group(
                                slope,
                                &codes,
                                gbase,
                                x,
                                xblock + gbase,
                                group_take(take, gbase, 16),
                                16,
                            ),
                        );
                    }
                    group += 1;
                }
            }
            None => ok = false,
        }
        block += 1;
    }
    finish_row(ok, total, wmax, x_abs)
}

/// Whether `ggml_type` has a fused reader. Every other type goes straight to [`matvec_fixed`].
pub(super) fn fused_type(ggml_type: u32) -> bool {
    ggml_type == quant::GGML_TYPE_Q8_0
        || ggml_type == quant::GGML_TYPE_Q5_0
        || ggml_type == quant::GGML_TYPE_Q2_K
        || ggml_type == quant::GGML_TYPE_Q3_K
        || ggml_type == quant::GGML_TYPE_Q4_K
        || ggml_type == quant::GGML_TYPE_Q6_K
}

/// The fused row for `ggml_type`, or `None` when the fast path cannot prove equality with
/// [`matvec_fixed`] and the caller must fall back.
fn fused_row(ggml_type: u32, row: &[u8], n_in: usize, x: &[i64], x_abs: i128) -> Option<i128> {
    match ggml_type {
        quant::GGML_TYPE_Q8_0 => fused_q8_0_row(row, n_in, x, x_abs),
        quant::GGML_TYPE_Q5_0 => fused_q5_0_row(row, n_in, x, x_abs),
        quant::GGML_TYPE_Q2_K => fused_q2_k_row(row, n_in, x, x_abs),
        quant::GGML_TYPE_Q3_K => fused_q3_k_row(row, n_in, x, x_abs),
        quant::GGML_TYPE_Q4_K => fused_q4_k_row(row, n_in, x, x_abs),
        quant::GGML_TYPE_Q6_K => fused_q6_k_row(row, n_in, x, x_abs),
        _ => None,
    }
}

/// The block-wise fused projection kernel: the same `y[o] = round(Σ_i W[o,i]·x_i / 2^32)` as
/// [`matvec_fixed`], computed group-wise.
///
/// The preconditions, the row stride, the output count and the single [`round_shift_32`] epilogue
/// are [`matvec_fixed`]'s. For `Q8_0`, `Q5_0`, `Q2_K`, `Q3_K`, `Q4_K` and `Q6_K` the row is
/// reduced per group as described in the module docs; when any group cannot prove the reference's
/// arithmetic stays in range, the whole call falls back to [`matvec_fixed`]. Any other type is
/// exactly [`matvec_fixed`].
///
/// # Errors
///
/// The same [`MatvecError`] values, in the same order, as [`matvec_fixed`].
///
/// # Complexity
///
/// `O(n_out·n_in)` time in checked integer multiply-adds — the same element visits as
/// [`matvec_fixed`] with one multiply per group instead of one per element — plus `O(n_in)` for the
/// activation magnitude sum, `O(n_out)` output and no per-row Q32.32 scratch.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the one `rounded as i64` narrowing is immediately preceded by an explicit \
              `i64::MIN`/`i64::MAX` range check; `TryFrom` emits an unmodeled external the \
              pinned fail-closed extraction audit rejects"
)]
pub(crate) fn matvec_fused(
    ggml_type: u32,
    raw: &[u8],
    n_out: usize,
    n_in: usize,
    x: &[i64],
) -> Result<Vec<i64>, MatvecError> {
    if !fused_type(ggml_type) {
        return matvec_fixed(ggml_type, raw, n_out, n_in, x);
    }
    let (qk, bb) =
        quant::block_geom(ggml_type).ok_or(MatvecError::UnsupportedQuant { ggml_type })?;
    if x.len() != n_in {
        return Err(MatvecError::ShapeMismatch { x: x.len(), n_in });
    }
    let row_bytes = n_in
        .div_ceil(qk)
        .checked_mul(bb)
        .ok_or(MatvecError::LengthOverflow)?;
    let need = n_out
        .checked_mul(row_bytes)
        .ok_or(MatvecError::LengthOverflow)?;
    if raw.len() < need {
        return Err(MatvecError::ShortPayload {
            need,
            have: raw.len(),
        });
    }
    let Some(x_abs) = x_magnitude(x) else {
        return matvec_fixed(ggml_type, raw, n_out, n_in, x);
    };
    let mut out: Vec<i64> = Vec::with_capacity(n_out);
    let mut o = 0usize;
    let mut ok = true;
    let mut fallback = false;
    while o < n_out && ok && !fallback {
        let start = o * row_bytes;
        let row = &raw[start..start + row_bytes];
        match fused_row(ggml_type, row, n_in, x, x_abs) {
            Some(acc) => {
                let rounded = round_shift_32(acc);
                if rounded > i128::from(i64::MAX) || rounded < i128::from(i64::MIN) {
                    ok = false;
                } else {
                    out.push(rounded as i64);
                }
            }
            None => fallback = true,
        }
        o += 1;
    }
    if fallback {
        return matvec_fixed(ggml_type, raw, n_out, n_in, x);
    }
    if !ok {
        return Err(MatvecError::OutputOverflow);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::cast_lossless,
        clippy::many_single_char_names,
        reason = "test-only fixture arithmetic on bounded small values; not extracted"
    )]

    use super::*;
    use crate::engine::EngineError;

    /// xorshift64: a deterministic code/activation stream for the equality tests.
    fn next_rand(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    /// A `rows × n_in` payload of random bytes with every block's `f16` scale(s) overwritten by
    /// `d` (and `dmin`), so each block is well formed while its codes and packed scales are random.
    fn random_payload(
        ggml_type: u32,
        rows: usize,
        n_in: usize,
        d: u16,
        dmin: u16,
        seed: u64,
    ) -> Vec<u8> {
        let (qk, bb) = quant::block_geom(ggml_type).expect("supported geometry");
        let blocks = n_in.div_ceil(qk);
        let row_bytes = blocks * bb;
        let mut raw = vec![0u8; rows * row_bytes];
        let mut state = seed;
        let mut i = 0usize;
        while i < raw.len() {
            raw[i] = (next_rand(&mut state) & 0xff) as u8;
            i += 1;
        }
        let mut block = 0usize;
        while block < blocks {
            let o = block * bb;
            match ggml_type {
                quant::GGML_TYPE_Q8_0 | quant::GGML_TYPE_Q5_0 => {
                    raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
                }
                // Q2_K: `d` at `QK_K/16 + QK_K/4`, `dmin` two bytes later.
                quant::GGML_TYPE_Q2_K => {
                    raw[o + 80..o + 82].copy_from_slice(&d.to_le_bytes());
                    raw[o + 82..o + 84].copy_from_slice(&dmin.to_le_bytes());
                }
                // Q3_K: `d` after `hmask[32]`, `qs[64]` and `scales[12]`.
                quant::GGML_TYPE_Q3_K => {
                    raw[o + 108..o + 110].copy_from_slice(&d.to_le_bytes());
                }
                quant::GGML_TYPE_Q4_K => {
                    raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
                    raw[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
                }
                // Q6_K: `d` after `ql[128]`, `qh[64]` and `scales[16]`.
                quant::GGML_TYPE_Q6_K => {
                    raw[o + 208..o + 210].copy_from_slice(&d.to_le_bytes());
                }
                _ => {}
            }
            block += 1;
        }
        raw
    }

    /// `n` small Q32.32 activations in `[-2048, 2047]`, from a deterministic stream.
    fn random_x(n: usize, seed: u64) -> Vec<i64> {
        let mut state = seed;
        let mut x: Vec<i64> = Vec::with_capacity(n);
        while x.len() < n {
            x.push((next_rand(&mut state) % 4096) as i64 - 2048);
        }
        x
    }

    /// Every supported type, several shapes (including partial last blocks, a zero width and a
    /// whole block), and random codes/packed scales agree exactly with [`matvec_fixed`].
    #[test]
    fn fused_matches_fixed_on_random_payloads_for_every_supported_type() {
        const TYPES: [u32; 6] = [
            quant::GGML_TYPE_Q8_0,
            quant::GGML_TYPE_Q5_0,
            quant::GGML_TYPE_Q2_K,
            quant::GGML_TYPE_Q3_K,
            quant::GGML_TYPE_Q4_K,
            quant::GGML_TYPE_Q6_K,
        ];
        const SHAPES: [(usize, usize); 8] = [
            (4, 32),
            (3, 48),
            (2, 33),
            (2, 300),
            (1, 64),
            (5, 0),
            (1, 256),
            (2, 512),
        ];
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut t = 0usize;
        while t < TYPES.len() {
            let ggml_type = TYPES[t];
            let mut s = 0usize;
            while s < SHAPES.len() {
                let (rows, n_in) = SHAPES[s];
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let raw = random_payload(ggml_type, rows, n_in, 0x3800, 0x3800, seed);
                let x = random_x(n_in, seed ^ 0xdead_beef);
                let fused = matvec_fused(ggml_type, &raw, rows, n_in, &x);
                let reference = matvec_fixed(ggml_type, &raw, rows, n_in, &x);
                assert_eq!(
                    fused, reference,
                    "type {ggml_type} shape ({rows},{n_in}) diverged"
                );
                s += 1;
            }
            t += 1;
        }
    }

    /// Every refusal class the supported types can produce agrees exactly: the fused fast path
    /// returns the error it computed, or falls back to the reference that produces it.
    #[test]
    fn fused_agrees_with_fixed_on_every_refusal_class() {
        let q8 = quant::GGML_TYPE_Q8_0;
        let raw = random_payload(q8, 2, 32, 0x3800, 0x3800, 1);
        let x = random_x(32, 7);

        // Shape mismatch.
        let short_x = vec![1i64; 31];
        assert_eq!(
            matvec_fused(q8, &raw, 2, 32, &short_x),
            matvec_fixed(q8, &raw, 2, 32, &short_x)
        );

        // Short payload.
        let short = &raw[..raw.len() - 1];
        assert_eq!(
            matvec_fused(q8, short, 2, 32, &x),
            matvec_fixed(q8, short, 2, 32, &x)
        );

        // Unsupported quant family.
        assert_eq!(
            matvec_fused(quant::GGML_TYPE_F32, &[], 1, 1, &[1]),
            matvec_fixed(quant::GGML_TYPE_F32, &[], 1, 1, &[1])
        );

        // Row-stride / payload length overflow.
        assert_eq!(
            matvec_fused(q8, &raw, usize::MAX, 32, &x),
            matvec_fixed(q8, &raw, usize::MAX, 32, &x)
        );

        // Non-finite scale: the fused reader cannot represent it and falls back to the reference's
        // `Dequant(NonFiniteScale)`.
        let mut inf = random_payload(q8, 2, 32, 0x3800, 0x3800, 2);
        inf[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
        let non_finite = matvec_fused(q8, &inf, 2, 32, &x);
        assert!(matches!(
            non_finite,
            Err(MatvecError::Dequant(EngineError::NonFiniteScale {
                bits: 0x7C00
            }))
        ));
        assert_eq!(non_finite, matvec_fixed(q8, &inf, 2, 32, &x));

        // Output overflow: weight 1.0 against `i64::MAX` rounds past `i64`.
        let mut one = random_payload(q8, 1, 32, 0x3C00, 0x3800, 3);
        let mut l = 0usize;
        while l < 32 {
            one[2 + l] = 1;
            l += 1;
        }
        let x_max = vec![i64::MAX; 32];
        let output = matvec_fused(q8, &one, 1, 32, &x_max);
        assert_eq!(output, Err(MatvecError::OutputOverflow));
        assert_eq!(output, matvec_fixed(q8, &one, 1, 32, &x_max));

        // Accumulator overflow: the largest finite `f16` scale with maximal all-positive codes and
        // `x = i64::MAX` leaves the fused bound inexpressible, so it falls back to the reference's
        // `AccumulatorOverflow`.
        let mut big = random_payload(q8, 1, 1024, 0x7BFF, 0x3800, 4);
        let mut block = 0usize;
        while block < 32 {
            let o = block * Q8_0_BLOCK_BYTES;
            let mut lane = 0usize;
            while lane < 32 {
                big[o + 2 + lane] = 127;
                lane += 1;
            }
            block += 1;
        }
        let x_big = vec![i64::MAX; 1024];
        let accumulator = matvec_fused(q8, &big, 1, 1024, &x_big);
        assert_eq!(accumulator, Err(MatvecError::AccumulatorOverflow));
        assert_eq!(accumulator, matvec_fixed(q8, &big, 1, 1024, &x_big));
    }

    /// A hand-computed affine `Q4_K` value: `d = dmin = 1.0`, pair 0 = `(sc 5, min 7)`, code 3 →
    /// `w0 = 5·3 − 7 = 8` in Q32.32 (`8 << 32`), so an activation that is `ONE` on weight 0 and
    /// zero elsewhere gives the Q32.32 output `8 << 32` (real `8`). The fused path must reproduce
    /// it.
    #[test]
    fn fused_affine_q4_k_matches_the_hand_value() {
        let mut raw = random_payload(quant::GGML_TYPE_Q4_K, 1, 256, 0x3C00, 0x3C00, 9);
        raw[4] = 5;
        raw[8] = 7;
        raw[16] = 0x03;
        let mut x = vec![0i64; 256];
        x[0] = 1 << 32;
        let fused = matvec_fused(quant::GGML_TYPE_Q4_K, &raw, 1, 256, &x);
        assert_eq!(fused, matvec_fixed(quant::GGML_TYPE_Q4_K, &raw, 1, 256, &x));
        assert_eq!(fused, Ok(vec![8 << 32]));
    }
}
