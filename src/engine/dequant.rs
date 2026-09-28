//! Whole-row dequantizers from the GGML block kinds to Q32.32 (port-map row 1).

use crate::format::quant::{
    self, QuantError, Q2_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE,
    Q5_0_BLOCK_BYTES, Q5_0_QK, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_0_QK, QK_K,
};

use super::EngineError;

// The six low-bit faces live in a sibling file. `#[path]` is written relative to this file, so it
// resolves to the same `src/engine/dequant_lowbit.rs` whether this module is `crate::engine::dequant`
// or the extraction unit's `#[path]`-declared root module.
#[path = "dequant_lowbit.rs"]
mod lowbit;

/// Decode an `f16` bit pattern into Q32.32 exactly.
///
/// Every finite `f16` is an integer multiple of `2^-24`, hence of the `2^-32` unit, so the result
/// is the exact real value times `2^32` with no rounding. `Inf` and `NaN` refuse.
///
/// # Errors
///
/// Returns [`EngineError::NonFiniteScale`] for an `Inf`/`NaN` exponent and
/// [`EngineError::FixedOverflow`] if the checked shift overflows (unreachable for a finite `f16`).
pub fn f16_to_fixed(h: u16) -> Result<i64, EngineError> {
    let sign = h >> 15;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let mag = if e == 0 {
        // Subnormal: value = m * 2^-24, so Q32.32 = m * 2^8.
        i64::from(m) << 8
    } else if e == 0x1f {
        return Err(EngineError::NonFiniteScale { bits: h });
    } else {
        // Normal: value = (1024 + m) * 2^(e-25), so Q32.32 = (1024 + m) * 2^(e+7).
        let sig = i64::from(m) + 1024;
        sig.checked_shl(u32::from(e) + 7)
            .ok_or(EngineError::FixedOverflow)?
    };
    Ok(if sign == 1 { -mag } else { mag })
}

/// Read a little-endian `u16` at `i`, refusing a short buffer.
fn read_u16(raw: &[u8], i: usize) -> Result<u16, EngineError> {
    let bytes: [u8; 2] = raw
        .get(i..i + 2)
        .and_then(|s| <[u8; 2]>::try_from(s).ok())
        .ok_or(EngineError::Quant(QuantError::Truncated {
            need: i + 2,
            have: raw.len(),
        }))?;
    Ok(u16::from_le_bytes(bytes))
}

/// Checked `a * b` for the fixed-point products.
fn mul(a: i64, b: i64) -> Result<i64, EngineError> {
    a.checked_mul(b).ok_or(EngineError::FixedOverflow)
}

/// Checked `a - b` for the affine fixed-point products.
fn sub(a: i64, b: i64) -> Result<i64, EngineError> {
    a.checked_sub(b).ok_or(EngineError::FixedOverflow)
}

/// Slice the whole `bb`-byte block at `block` out of `raw`, refusing a short payload.
fn block_slice(raw: &[u8], block: usize, bb: usize) -> Result<&[u8], EngineError> {
    let offset = block.checked_mul(bb).ok_or(EngineError::LengthOverflow)?;
    let end = offset.checked_add(bb).ok_or(EngineError::LengthOverflow)?;
    raw.get(offset..end)
        .ok_or(EngineError::Quant(QuantError::Truncated {
            need: end,
            have: raw.len(),
        }))
}

/// Append the first `take` weights of a full assembled super-block to `out`.
fn push_prefix(out: &mut Vec<i64>, vals: &[i64; QK_K], take: usize) {
    let mut idx = 0usize;
    while idx < take {
        out.push(vals[idx]);
        idx += 1;
    }
}

/// Dequantize `n` weights of ggml type `t` from `raw` into Q32.32, appending to `out`.
///
/// The `raw` payload must hold whole blocks for the declared geometry; the quant code and scale
/// accessors bounds-check every block, and a short payload refuses through [`EngineError::Quant`].
/// Every supported product is exact integer arithmetic: the `f16` scale decodes exactly to Q32.32
/// and the integer code multiplies it, with no intermediate shift and no rounding. Supported:
/// `F16`, `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`, `Q2_K`, `Q3_K`, `Q1_0`, `Q2_0`, `PQ2_0`, `TQ2_0`,
/// `TQ1_0`, `PTQ1_0`.
pub(super) fn dequant_row_fixed(
    ggml_type: u32,
    raw: &[u8],
    n_elems: usize,
    out: &mut Vec<i64>,
) -> Result<(), EngineError> {
    match ggml_type {
        quant::GGML_TYPE_F16 => dequant_f16_row(raw, n_elems, out),
        quant::GGML_TYPE_Q8_0 => dequant_q8_0_row(raw, n_elems, out),
        quant::GGML_TYPE_Q5_0 => dequant_q5_0_row(raw, n_elems, out),
        quant::GGML_TYPE_Q4_K => dequant_q4_k_row(raw, n_elems, out),
        quant::GGML_TYPE_Q6_K => dequant_q6_k_row(raw, n_elems, out),
        quant::GGML_TYPE_Q2_K => dequant_q2_k_row(raw, n_elems, out),
        quant::GGML_TYPE_Q3_K => dequant_q3_k_row(raw, n_elems, out),
        quant::GGML_TYPE_Q1_0 => lowbit::dequant_q1_0_row(raw, n_elems, out),
        quant::GGML_TYPE_Q2_0 => lowbit::dequant_q2_0_row(raw, n_elems, out),
        quant::GGML_TYPE_PQ2_0 => lowbit::dequant_pq2_0_row(raw, n_elems, out),
        quant::GGML_TYPE_TQ2_0 => lowbit::dequant_tq2_0_row(raw, n_elems, out),
        quant::GGML_TYPE_TQ1_0 => lowbit::dequant_tq1_0_row(raw, n_elems, out),
        quant::GGML_TYPE_PTQ1_0 => lowbit::dequant_ptq1_0_row(raw, n_elems, out),
        other => Err(EngineError::UnsupportedEmbeddingQuant { ggml_type: other }),
    }
}

/// Read an `f16` scale at byte offset `i` and decode it exactly to Q32.32.
fn read_scale(raw: &[u8], i: usize) -> Result<i64, EngineError> {
    f16_to_fixed(read_u16(raw, i)?)
}

/// `Q8_0` block `block`'s `f16` scale (Q32.32) and its 32 signed codes.
fn q8_0_block(raw: &[u8], block: usize) -> Result<(i64, [i8; Q8_0_QK]), EngineError> {
    let scale = read_scale(raw, block * Q8_0_BLOCK_BYTES)?;
    let codes = quant::q8_0_codes(raw, block)?;
    Ok((scale, codes))
}

/// `Q5_0` block `block`'s `f16` scale (Q32.32) and its 32 signed five-bit codes.
fn q5_0_block(raw: &[u8], block: usize) -> Result<(i64, [i8; Q5_0_QK]), EngineError> {
    let blk = block_slice(raw, block, Q5_0_BLOCK_BYTES)?;
    let scale = f16_to_fixed(quant::q5_0_d(blk)?)?;
    let codes = quant::q5_0_codes(blk)?;
    Ok((scale, codes))
}

/// `Q2_K` block `block`'s `(d, dmin, scales, codes)`.
fn q2_k_block(raw: &[u8], block: usize) -> Result<(i64, i64, [u8; 16], [u8; QK_K]), EngineError> {
    let blk = block_slice(raw, block, Q2_K_BLOCK_BYTES)?;
    let d = f16_to_fixed(quant::q2_k_d(blk)?)?;
    let dmin = f16_to_fixed(quant::q2_k_dmin(blk)?)?;
    let scales = quant::q2_k_scales(blk)?;
    let codes = quant::q2_k_codes(blk)?;
    Ok((d, dmin, scales, codes))
}

/// `Q3_K` block `block`'s `(d, scales, codes)`.
fn q3_k_block(raw: &[u8], block: usize) -> Result<(i64, [i8; 16], [i8; QK_K]), EngineError> {
    let blk = block_slice(raw, block, Q3_K_BLOCK_BYTES)?;
    let d = f16_to_fixed(quant::q3_k_d(blk)?)?;
    let scales = quant::q3_k_scales(blk)?;
    let codes = quant::q3_k_codes(blk)?;
    Ok((d, scales, codes))
}

/// `Q4_K` block `block`'s `(scale, min_scale, packed scales, codes)`.
///
/// The 12 packed scale bytes are collected here rather than through
/// [`quant::q4_k_scale_mins`](crate::format::quant::q4_k_scale_mins): that accessor's
/// `[(u8, u8); 8]` array repeat lowers to `core::array::repeat`, which the pinned Charon fragment
/// leaves opaque and the pinned Aeneas `FunsAnalysis` pass rejects before translation. The bytes
/// and the per-pair accessor [`quant::q4_k_scale_min`](crate::format::quant::q4_k_scale_min) are
/// the same; only the container is not.
fn q4_k_block(
    raw: &[u8],
    block: usize,
) -> Result<(i64, i64, [u8; Q4_K_SCALE_SIZE], [u8; QK_K]), EngineError> {
    let blk = block_slice(raw, block, Q4_K_BLOCK_BYTES)?;
    let scale = read_scale(blk, 0)?;
    let min_scale = read_scale(blk, 2)?;
    let mut sc = [0u8; Q4_K_SCALE_SIZE];
    let mut k = 0usize;
    while k < Q4_K_SCALE_SIZE {
        sc[k] = blk[4 + k];
        k += 1;
    }
    let codes = quant::q4_k_codes(blk)?;
    Ok((scale, min_scale, sc, codes))
}

/// `Q6_K` block `block`'s `(d, scales, codes)`.
fn q6_k_block(raw: &[u8], block: usize) -> Result<(i64, [i8; 16], [i8; QK_K]), EngineError> {
    let blk = block_slice(raw, block, Q6_K_BLOCK_BYTES)?;
    let d = f16_to_fixed(quant::q6_k_d(blk)?)?;
    let scales = quant::q6_k_scales(blk)?;
    let codes = quant::q6_k_codes(blk)?;
    Ok((d, scales, codes))
}

/// `F16` row: one exact Q32.32 value per `f16` element.
///
/// The loop carries a `bool` failure flag and stops on the first refusal, so no `?` and no early
/// `return` appears inside it; the refusals are the same operations in the same order.
fn dequant_f16_row(raw: &[u8], n_elems: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
    let mut idx = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while idx < n_elems && ok {
        match read_u16(raw, 2 * idx) {
            Ok(bits) => match f16_to_fixed(bits) {
                Ok(v) => out.push(v),
                Err(e) => {
                    err = Some(e);
                    ok = false;
                }
            },
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        idx += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `Q8_0` row: `w = d * q`, 32 signed codes per `f16` scale.
///
/// The block and lane loops carry a `bool` failure flag and stop on the first refusal, so no `?`
/// and no early `return` appears inside either loop.
fn dequant_q8_0_row(raw: &[u8], n_elems: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(Q8_0_QK);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match q8_0_block(raw, block) {
            Ok((scale, codes)) => {
                let take = Q8_0_QK.min(n_elems - block * Q8_0_QK);
                let mut lane = 0usize;
                while lane < take && ok {
                    match mul(scale, i64::from(codes[lane])) {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                        }
                    }
                    lane += 1;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `Q5_0` row: `w = d * q`, 32 signed five-bit codes (`q - 16`) per `f16` scale.
///
/// The block and lane loops carry a `bool` failure flag and stop on the first refusal, so no `?`
/// and no early `return` appears inside either loop.
fn dequant_q5_0_row(raw: &[u8], n_elems: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(Q5_0_QK);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match q5_0_block(raw, block) {
            Ok((scale, codes)) => {
                let take = Q5_0_QK.min(n_elems - block * Q5_0_QK);
                let mut lane = 0usize;
                while lane < take && ok {
                    match mul(scale, i64::from(codes[lane])) {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                        }
                    }
                    lane += 1;
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `Q2_K` row: `w = d * sc * q - dmin * m`, sixteen 16-weight groups.
///
/// The group and lane loops carry a `bool` failure flag and stop on the first refusal, so no `?`
/// and no early `return` appears inside either loop.
fn dequant_q2_k_row(raw: &[u8], n_elems: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(QK_K);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match q2_k_block(raw, block) {
            Ok((d, dmin, scales, codes)) => {
                let take = QK_K.min(n_elems - block * QK_K);
                let mut vals = [0i64; QK_K];
                let mut group = 0usize;
                while group < 16 && ok {
                    let (scale_code, min_code) = quant::q2_k_scale_min(scales[group]);
                    let slope = match mul(d, i64::from(scale_code)) {
                        Ok(slope) => slope,
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                            0
                        }
                    };
                    let intercept = match mul(dmin, i64::from(min_code)) {
                        Ok(intercept) => intercept,
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                            0
                        }
                    };
                    let mut lane = 0usize;
                    while lane < 16 && ok {
                        match mul(slope, i64::from(codes[16 * group + lane])) {
                            Ok(prod) => match sub(prod, intercept) {
                                Ok(v) => vals[16 * group + lane] = v,
                                Err(e) => {
                                    err = Some(e);
                                    ok = false;
                                }
                            },
                            Err(e) => {
                                err = Some(e);
                                ok = false;
                            }
                        }
                        lane += 1;
                    }
                    group += 1;
                }
                if ok {
                    push_prefix(out, &vals, take);
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `Q3_K` row: `w = d * (sc - 32) * q`, sixteen 16-weight groups with signed 6-bit scales.
///
/// The group and lane loops carry a `bool` failure flag and stop on the first refusal, so no `?`
/// and no early `return` appears inside either loop.
fn dequant_q3_k_row(raw: &[u8], n_elems: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(QK_K);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match q3_k_block(raw, block) {
            Ok((d, scales, codes)) => {
                let take = QK_K.min(n_elems - block * QK_K);
                let mut vals = [0i64; QK_K];
                let mut group = 0usize;
                while group < 16 && ok {
                    let slope = match mul(d, i64::from(scales[group]) - 32) {
                        Ok(slope) => slope,
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                            0
                        }
                    };
                    let mut lane = 0usize;
                    while lane < 16 && ok {
                        match mul(slope, i64::from(codes[16 * group + lane])) {
                            Ok(v) => vals[16 * group + lane] = v,
                            Err(e) => {
                                err = Some(e);
                                ok = false;
                            }
                        }
                        lane += 1;
                    }
                    group += 1;
                }
                if ok {
                    push_prefix(out, &vals, take);
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `Q4_K` row: `w = d * sc * q - dmin * m`, four 64-weight pairs of 32-weight groups.
///
/// The group and lane loops carry a `bool` failure flag and stop on the first refusal, so no `?`
/// and no early `return` appears inside either loop.
#[allow(
    clippy::too_many_lines,
    reason = "the two 32-weight sub-groups of each of the four 64-weight pairs are written out \
              separately so every fallible product is an explicit match arm; the refusal flag \
              cannot be hoisted into a helper without reintroducing an early return in a loop"
)]
fn dequant_q4_k_row(raw: &[u8], n_elems: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(QK_K);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match q4_k_block(raw, block) {
            Ok((scale, min_scale, sc, codes)) => {
                let take = QK_K.min(n_elems - block * QK_K);
                let mut vals = [0i64; QK_K];
                let mut group = 0usize;
                while group < 4 && ok {
                    let (scale_code_lo, min_code_lo) = quant::q4_k_scale_min(&sc, 2 * group);
                    let (scale_code_hi, min_code_hi) = quant::q4_k_scale_min(&sc, 2 * group + 1);
                    let slope_lo = match mul(scale, i64::from(scale_code_lo)) {
                        Ok(slope) => slope,
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                            0
                        }
                    };
                    let intercept_lo = match mul(min_scale, i64::from(min_code_lo)) {
                        Ok(intercept) => intercept,
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                            0
                        }
                    };
                    let slope_hi = match mul(scale, i64::from(scale_code_hi)) {
                        Ok(slope) => slope,
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                            0
                        }
                    };
                    let intercept_hi = match mul(min_scale, i64::from(min_code_hi)) {
                        Ok(intercept) => intercept,
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                            0
                        }
                    };
                    let mut lane = 0usize;
                    while lane < 32 && ok {
                        match mul(slope_lo, i64::from(codes[64 * group + lane])) {
                            Ok(prod) => match sub(prod, intercept_lo) {
                                Ok(v) => vals[64 * group + lane] = v,
                                Err(e) => {
                                    err = Some(e);
                                    ok = false;
                                }
                            },
                            Err(e) => {
                                err = Some(e);
                                ok = false;
                            }
                        }
                        match mul(slope_hi, i64::from(codes[64 * group + 32 + lane])) {
                            Ok(prod) => match sub(prod, intercept_hi) {
                                Ok(v) => vals[64 * group + 32 + lane] = v,
                                Err(e) => {
                                    err = Some(e);
                                    ok = false;
                                }
                            },
                            Err(e) => {
                                err = Some(e);
                                ok = false;
                            }
                        }
                        lane += 1;
                    }
                    group += 1;
                }
                if ok {
                    push_prefix(out, &vals, take);
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

/// `Q6_K` row: `w = d * sc * q`, sixteen 16-weight groups with signed scales and codes.
///
/// The group and lane loops carry a `bool` failure flag and stop on the first refusal, so no `?`
/// and no early `return` appears inside either loop.
fn dequant_q6_k_row(raw: &[u8], n_elems: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
    let blocks = n_elems.div_ceil(QK_K);
    let mut block = 0usize;
    let mut ok = true;
    let mut err: Option<EngineError> = None;
    while block < blocks && ok {
        match q6_k_block(raw, block) {
            Ok((d, scales, codes)) => {
                let take = QK_K.min(n_elems - block * QK_K);
                let mut vals = [0i64; QK_K];
                let mut group = 0usize;
                while group < 16 && ok {
                    let slope = match mul(d, i64::from(scales[group])) {
                        Ok(slope) => slope,
                        Err(e) => {
                            err = Some(e);
                            ok = false;
                            0
                        }
                    };
                    let mut lane = 0usize;
                    while lane < 16 && ok {
                        match mul(slope, i64::from(codes[16 * group + lane])) {
                            Ok(v) => vals[16 * group + lane] = v,
                            Err(e) => {
                                err = Some(e);
                                ok = false;
                            }
                        }
                        lane += 1;
                    }
                    group += 1;
                }
                if ok {
                    push_prefix(out, &vals, take);
                }
            }
            Err(e) => {
                err = Some(e);
                ok = false;
            }
        }
        block += 1;
    }
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}
