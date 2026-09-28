//! ggml quant block geometry and integer code unpacking for GGUF tensor payloads.
//!
//! Ported from the served inference path in the sibling `fold` tree
//! (`fold/bedrock/src/serve/gguf.rs`, which in turn transliterates
//! `llama.cpp/ggml/src/ggml-quants.c`). The arithmetic is the reference arithmetic; this module
//! keeps the reproducibility contract by factoring every field offset and integer code unpack out
//! of the float scaling.
//!
//! Refusal is explicit: every length is checked against the block geometry and every failure is a
//! [`QuantError`]. There is no indexing before a bounds check and no panic on the data path.
//!
//! Implements the `F32`, `F16`, `Q8_0`, `Q5_0`, `Q2_K`, `Q3_K`, `Q4_K` and `Q6_K` formats, and the
//! geometry and type ids of the six low-bit faces `Q1_0`, `Q2_0`, `PQ2_0`, `TQ2_0`, `TQ1_0` and
//! `PTQ1_0` (whose fixed-point row dequantizers live in `crate::engine::dequant`). The codebook
//! `IQ*` i-quants need their lookup grids and are not ported here.

/// Weights per k-quant super-block (ggml `QK_K`).
pub const QK_K: usize = 256;

/// ggml `F32`.
pub const GGML_TYPE_F32: u32 = 0;
/// ggml `F16`.
pub const GGML_TYPE_F16: u32 = 1;
/// ggml `Q8_0`.
pub const GGML_TYPE_Q8_0: u32 = 8;
/// ggml `Q5_0`.
pub const GGML_TYPE_Q5_0: u32 = 6;
/// ggml `Q2_K`.
pub const GGML_TYPE_Q2_K: u32 = 10;
/// ggml `Q3_K`.
pub const GGML_TYPE_Q3_K: u32 = 11;
/// ggml `Q4_K`.
pub const GGML_TYPE_Q4_K: u32 = 12;
/// ggml `Q6_K`.
pub const GGML_TYPE_Q6_K: u32 = 14;
/// ggml `TQ1_0` (ternary base-3 trits, 256 weights per block).
pub const GGML_TYPE_TQ1_0: u32 = 34;
/// ggml `TQ2_0` (four-level 2-bit ternary codes, 256 weights per block).
pub const GGML_TYPE_TQ2_0: u32 = 35;
/// ggml `Q1_0` (one signed bit, 128 weights per block).
pub const GGML_TYPE_Q1_0: u32 = 41;
/// ggml `Q2_0` (four-level 2-bit codes, 64 weights per block).
pub const GGML_TYPE_Q2_0: u32 = 42;
/// ggml `PQ2_0` (fork-private `Q2_0` at 128 weights per block).
pub const GGML_TYPE_PQ2_0: u32 = 142;
/// ggml `PTQ1_0` (fork-private ternary base-3 trits, 128 weights per block).
pub const GGML_TYPE_PTQ1_0: u32 = 143;

/// Weights per Q8_0 block: one `f16` scale and 32 `i8` codes.
pub const Q8_0_QK: usize = 32;
/// Bytes per Q8_0 block.
pub const Q8_0_BLOCK_BYTES: usize = 2 + 32;
/// Weights per Q5_0 block: one `f16` scale, a 4-byte high-bit plane and 16 bytes of nibbles.
pub const Q5_0_QK: usize = Q8_0_QK;
/// Bytes per Q5_0 block: `f16 d`, `qh[4]`, `qs[16]`.
pub const Q5_0_BLOCK_BYTES: usize = 2 + 4 + Q5_0_QK / 2;
/// Bytes per Q2_K super-block: `scales[16]`, `qs[64]`, `f16 d`, `f16 dmin`.
pub const Q2_K_BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 4;
/// Bytes per Q3_K super-block: `hmask[32]`, `qs[64]`, `scales[12]`, `f16 d`.
pub const Q3_K_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2;
/// Bytes per Q4_K super-block: `f16 d`, `f16 dmin`, `scales[12]`, `qs[128]`.
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2;
/// Bytes per Q6_K super-block: `ql[128]`, `qh[64]`, `scales[16]`, `f16 d`.
pub const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
/// Bytes of the packed 6-bit `scales`/`mins` field of a `Q4_K` block (ggml `K_SCALE_SIZE`).
pub const Q4_K_SCALE_SIZE: usize = 12;

/// Weights per Q1_0 block: one `f16` scale and 16 packed sign bits.
pub const Q1_0_QK: usize = 128;
/// Bytes per Q1_0 block: `f16 d`, `qs[16]`.
pub const Q1_0_BLOCK_BYTES: usize = 2 + Q1_0_QK / 8;
/// Weights per Q2_0 block: one `f16` scale and 16 packed two-bit codes.
pub const Q2_0_QK: usize = 64;
/// Bytes per Q2_0 block: `f16 d`, `qs[16]`.
pub const Q2_0_BLOCK_BYTES: usize = 2 + Q2_0_QK / 4;
/// Weights per PQ2_0 block: one `f16` scale and 32 packed two-bit codes.
pub const PQ2_0_QK: usize = 128;
/// Bytes per PQ2_0 block: `f16 d`, `qs[32]`.
pub const PQ2_0_BLOCK_BYTES: usize = 2 + PQ2_0_QK / 4;
/// Weights per TQ2_0 block: 64 packed two-bit bytes and one `f16` scale.
pub const TQ2_0_QK: usize = 256;
/// Bytes per TQ2_0 block: `qs[64]`, `f16 d`.
pub const TQ2_0_BLOCK_BYTES: usize = TQ2_0_QK / 4 + 2;
/// Weights per TQ1_0 block: 240 base-3 trits in `qs` and 16 in `qh`.
pub const TQ1_0_QK: usize = 256;
/// Bytes per TQ1_0 block: `qs[48]`, `qh[4]`, `f16 d`.
pub const TQ1_0_BLOCK_BYTES: usize = (TQ1_0_QK - 4 * TQ1_0_QK / 64) / 5 + TQ1_0_QK / 64 + 2;
/// Weights per PTQ1_0 block: 120 base-3 trits in `qs` and 8 in `qh`.
pub const PTQ1_0_QK: usize = 128;
/// Bytes per PTQ1_0 block: `qs[24]`, `qh[2]`, `f16 d`.
pub const PTQ1_0_BLOCK_BYTES: usize = (PTQ1_0_QK - 4 * PTQ1_0_QK / 64) / 5 + PTQ1_0_QK / 64 + 2;

/// A refused quant operation. Integer-only so callers can match without formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantError {
    /// A block or payload was shorter than the geometry requires.
    Truncated {
        /// Bytes the geometry requires.
        need: usize,
        /// Bytes actually available.
        have: usize,
    },
    /// The ggml type id has no geometry in this module.
    UnknownType(u32),
    /// A length or byte count did not fit the target integer width.
    LengthOverflow,
}

impl core::fmt::Display for QuantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuantError::Truncated { need, have } => {
                write!(f, "truncated: need {need} bytes, have {have}")
            }
            QuantError::UnknownType(t) => write!(f, "unknown ggml type {t}"),
            QuantError::LengthOverflow => write!(f, "length overflow"),
        }
    }
}

impl std::error::Error for QuantError {}

/// `(weights per block, bytes per block)` for every type this module implements, or `None`.
///
/// One table, so the directory's size arithmetic, the payload read and the dequantizer dispatch
/// cannot drift apart.
pub fn block_geom(t: u32) -> Option<(usize, usize)> {
    Some(match t {
        GGML_TYPE_F32 => (1, 4),
        GGML_TYPE_F16 => (1, 2),
        GGML_TYPE_Q5_0 => (Q5_0_QK, Q5_0_BLOCK_BYTES),
        GGML_TYPE_Q8_0 => (Q8_0_QK, Q8_0_BLOCK_BYTES),
        GGML_TYPE_Q2_K => (QK_K, Q2_K_BLOCK_BYTES),
        GGML_TYPE_Q3_K => (QK_K, Q3_K_BLOCK_BYTES),
        GGML_TYPE_Q4_K => (QK_K, Q4_K_BLOCK_BYTES),
        GGML_TYPE_Q6_K => (QK_K, Q6_K_BLOCK_BYTES),
        GGML_TYPE_Q1_0 => (Q1_0_QK, Q1_0_BLOCK_BYTES),
        GGML_TYPE_Q2_0 => (Q2_0_QK, Q2_0_BLOCK_BYTES),
        GGML_TYPE_PQ2_0 => (PQ2_0_QK, PQ2_0_BLOCK_BYTES),
        GGML_TYPE_TQ2_0 => (TQ2_0_QK, TQ2_0_BLOCK_BYTES),
        GGML_TYPE_TQ1_0 => (TQ1_0_QK, TQ1_0_BLOCK_BYTES),
        GGML_TYPE_PTQ1_0 => (PTQ1_0_QK, PTQ1_0_BLOCK_BYTES),
        _ => return None,
    })
}

/// The human-readable ggml type name, for diagnostics only.
pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_Q8_0 => "Q8_0",
        GGML_TYPE_Q5_0 => "Q5_0",
        GGML_TYPE_Q2_K => "Q2_K",
        GGML_TYPE_Q3_K => "Q3_K",
        GGML_TYPE_Q4_K => "Q4_K",
        GGML_TYPE_Q6_K => "Q6_K",
        GGML_TYPE_Q1_0 => "Q1_0",
        GGML_TYPE_Q2_0 => "Q2_0",
        GGML_TYPE_PQ2_0 => "PQ2_0",
        GGML_TYPE_TQ2_0 => "TQ2_0",
        GGML_TYPE_TQ1_0 => "TQ1_0",
        GGML_TYPE_PTQ1_0 => "PTQ1_0",
        _ => "UNKNOWN",
    }
}

/// Number of storage blocks for `n_elements` weights of type `t`, checked.
pub fn block_count(t: u32, n_elements: u64) -> Result<u64, QuantError> {
    let (qk, _) = block_geom(t).ok_or(QuantError::UnknownType(t))?;
    let qk = qk as u64;
    // `n.div_ceil(qk)` without an intermediate `n + qk - 1` that could overflow.
    let whole = n_elements / qk;
    Ok(whole + u64::from(n_elements % qk != 0))
}

/// Storage bytes for `n_elements` weights of type `t`, checked for overflow.
///
/// This is the directory-size computation: a corrupt element count that overflows `u64` refuses
/// rather than wrapping.
pub fn required_bytes(t: u32, n_elements: u64) -> Result<u64, QuantError> {
    let (_, bb) = block_geom(t).ok_or(QuantError::UnknownType(t))?;
    block_count(t, n_elements)?
        .checked_mul(bb as u64)
        .ok_or(QuantError::LengthOverflow)
}

/// Exact IEEE-754 half → single conversion (inverse of the storage narrowing). Every `u16` maps to
/// a defined `f32`; NaN payloads collapse to the canonical quiet NaN, matching the reference.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let e = (h >> 10) & 0x1F;
    let m = u32::from(h & 0x3FF);
    match (e, m) {
        (0, 0) => f32::from_bits(sign),
        (0, _) => {
            let v = m as f32 * 5.960_464_5e-8;
            if sign != 0 {
                -v
            } else {
                v
            }
        }
        (0x1F, 0) => f32::from_bits(sign | 0x7F80_0000),
        (0x1F, _) => f32::from_bits(sign | 0x7FC0_0000),
        _ => f32::from_bits(sign | ((u32::from(e) + 112) << 23) | (m << 13)),
    }
}

// ── checked little-endian readers ───────────────────────────────────────────────────────────────

fn u16le(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn f16le(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16le(b, i))
}

#[inline]
fn need(blk: &[u8], n: usize) -> Result<(), QuantError> {
    if blk.len() < n {
        Err(QuantError::Truncated {
            need: n,
            have: blk.len(),
        })
    } else {
        Ok(())
    }
}

// ── Q8_0 ────────────────────────────────────────────────────────────────────────────────────────

/// The 32 signed codes of Q8_0 block `b` in `raw`.
pub fn q8_0_codes(raw: &[u8], b: usize) -> Result<[i8; Q8_0_QK], QuantError> {
    let o = b
        .checked_mul(Q8_0_BLOCK_BYTES)
        .ok_or(QuantError::LengthOverflow)?;
    need(raw, o + Q8_0_BLOCK_BYTES)?;
    let mut out = [0i8; Q8_0_QK];
    let mut j = 0usize;
    while j < Q8_0_QK {
        out[j] = raw[o + 2 + j] as i8;
        j += 1;
    }
    Ok(out)
}

/// The `f16` scale of Q8_0 block `b`.
pub fn q8_0_scale(raw: &[u8], b: usize) -> Result<f32, QuantError> {
    let o = b
        .checked_mul(Q8_0_BLOCK_BYTES)
        .ok_or(QuantError::LengthOverflow)?;
    need(raw, o + Q8_0_BLOCK_BYTES)?;
    Ok(f16le(raw, o))
}

// ── Q5_0 ────────────────────────────────────────────────────────────────────────────────────────

// A Q5_0 block is `f16 d`, then `qh[4]` (the fifth, high bit of each code, one bit per code) and
// then `qs[16]` (two 4-bit low nibbles per byte). ggml reads `qh` as one little-endian `u32`; the
// low-plane code `j` takes bit `j` and the high-plane code `QK/2 + j` takes bit `16 + j`.

/// The raw little-endian `f16` bits of the Q5_0 scale `d` in block `blk`.
///
/// Integer accessor for a fixed-point reader: the caller converts the bits with
/// [`f16_to_f32`] or its own exact decoder.
pub fn q5_0_d(blk: &[u8]) -> Result<u16, QuantError> {
    need(blk, Q5_0_BLOCK_BYTES)?;
    Ok(u16le(blk, 0))
}

/// The 32 signed five-bit Q5_0 codes of `blk`, in output order (already offset by −16).
///
/// ggml order: `qs[j]` low nibble is output `j`, high nibble is output `QK/2 + j`; the fifth bit of
/// output `j` is bit `j` of `qh` and of output `QK/2 + j` is bit `16 + j`.
pub fn q5_0_codes(blk: &[u8]) -> Result<[i8; Q5_0_QK], QuantError> {
    need(blk, Q5_0_BLOCK_BYTES)?;
    let qh = &blk[2..6];
    let qs = &blk[6..6 + Q5_0_QK / 2];
    let mut out = [0i8; Q5_0_QK];
    let mut j = 0usize;
    while j < Q5_0_QK / 2 {
        let nib = qs[j];
        let lo_bit = (qh[j / 8] >> (j % 8)) & 1;
        let hi_bit = (qh[2 + j / 8] >> (j % 8)) & 1;
        let lo = (nib & 0xF) | (lo_bit << 4);
        let hi = (nib >> 4) | (hi_bit << 4);
        out[j] = (i32::from(lo) - 16) as i8;
        out[Q5_0_QK / 2 + j] = (i32::from(hi) - 16) as i8;
        j += 1;
    }
    Ok(out)
}

/// The `f16` scale of Q5_0 block `b`, matching [`q8_0_scale`].
pub fn q5_0_scale(raw: &[u8], b: usize) -> Result<f32, QuantError> {
    let o = b
        .checked_mul(Q5_0_BLOCK_BYTES)
        .ok_or(QuantError::LengthOverflow)?;
    need(raw, o + Q5_0_BLOCK_BYTES)?;
    Ok(f16le(raw, o))
}

// ── Q2_K ────────────────────────────────────────────────────────────────────────────────────────

/// The sixteen packed `(scale, min)` bytes of a Q2_K block, in output-group order.
pub fn q2_k_scales(blk: &[u8]) -> Result<[u8; 16], QuantError> {
    need(blk, QK_K / 16)?;
    let mut out = [0u8; 16];
    out.copy_from_slice(&blk[..16]);
    Ok(out)
}

/// Split one packed Q2_K `(scale, min)` byte: low nibble is the scale, high nibble the min.
pub fn q2_k_scale_min(sc: u8) -> (u8, u8) {
    (sc & 0xF, sc >> 4)
}

/// The raw little-endian `f16` bits of the Q2_K scale `d`.
pub fn q2_k_d(blk: &[u8]) -> Result<u16, QuantError> {
    need(blk, Q2_K_BLOCK_BYTES)?;
    Ok(u16le(blk, QK_K / 16 + QK_K / 4))
}

/// The raw little-endian `f16` bits of the Q2_K minimum scale `dmin`.
pub fn q2_k_dmin(blk: &[u8]) -> Result<u16, QuantError> {
    need(blk, Q2_K_BLOCK_BYTES)?;
    Ok(u16le(blk, QK_K / 16 + QK_K / 4 + 2))
}

/// The 256 two-bit `Q2_K` codes, in output order (sixteen groups of sixteen).
pub fn q2_k_codes(blk: &[u8]) -> Result<[u8; QK_K], QuantError> {
    need(blk, QK_K / 16 + QK_K / 4)?;
    let qs = &blk[QK_K / 16..QK_K / 16 + QK_K / 4];
    let mut out = [0u8; QK_K];
    let mut y = 0usize;
    let mut half = 0usize;
    while half < 2 {
        let base = half * 32;
        let mut shift = 0u32;
        let mut round = 0usize;
        while round < 4 {
            let mut plane = 0usize;
            while plane < 2 {
                let lo = plane * 16;
                let mut l = 0usize;
                while l < 16 {
                    out[y] = (qs[base + l + lo] >> shift) & 3;
                    y += 1;
                    l += 1;
                }
                plane += 1;
            }
            shift += 2;
            round += 1;
        }
        half += 1;
    }
    Ok(out)
}

/// Q2_K → `f32`: `w = d·(sc & 0xF)·q − dmin·(sc >> 4)`, sixteen 16-weight groups.
pub fn dequant_q2_k_block(blk: &[u8], out: &mut [f32; QK_K]) -> Result<(), QuantError> {
    need(blk, Q2_K_BLOCK_BYTES)?;
    let d = f16le(blk, QK_K / 16 + QK_K / 4);
    let dmin = f16le(blk, QK_K / 16 + QK_K / 4 + 2);
    let scales = q2_k_scales(blk)?;
    let codes = q2_k_codes(blk)?;
    let mut g = 0usize;
    while g < 16 {
        let sc = scales[g];
        let dl = d * f32::from(sc & 0xF);
        let ml = dmin * f32::from(sc >> 4);
        let mut l = 0usize;
        while l < 16 {
            out[g * 16 + l] = dl * f32::from(codes[g * 16 + l]) - ml;
            l += 1;
        }
        g += 1;
    }
    Ok(())
}

// ── Q3_K ────────────────────────────────────────────────────────────────────────────────────────

/// The sixteen signed 6-bit Q3_K group scales, decoded from the packed `scales[12]` shuffle.
pub fn q3_k_scales(blk: &[u8]) -> Result<[i8; 16], QuantError> {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    need(blk, Q3_K_BLOCK_BYTES)?;
    let sc_raw = &blk[QK_K / 8 + QK_K / 4..QK_K / 8 + QK_K / 4 + 12];
    let mut aux = [0u32; 4];
    let mut k = 0usize;
    while k < 3 {
        let o = 4 * k;
        aux[k] = u32::from_le_bytes([sc_raw[o], sc_raw[o + 1], sc_raw[o + 2], sc_raw[o + 3]]);
        k += 1;
    }
    let tmp = aux[2];
    let words = [
        (aux[0] & KMASK2) | ((tmp & KMASK1) << 4),
        (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut scales = [0i8; 16];
    let mut j = 0usize;
    while j < 16 {
        scales[j] = (words[j / 4] >> (8 * (j % 4))) as u8 as i8;
        j += 1;
    }
    Ok(scales)
}

/// The raw little-endian `f16` bits of the Q3_K scale `d`.
pub fn q3_k_d(blk: &[u8]) -> Result<u16, QuantError> {
    need(blk, Q3_K_BLOCK_BYTES)?;
    Ok(u16le(blk, QK_K / 8 + QK_K / 4 + 12))
}

/// The 256 signed `Q3_K` codes with the high bit folded in: `q − 4·[hmask bit clear]`.
pub fn q3_k_codes(blk: &[u8]) -> Result<[i8; QK_K], QuantError> {
    need(blk, Q3_K_BLOCK_BYTES)?;
    let hm = &blk[..QK_K / 8];
    let qs = &blk[QK_K / 8..QK_K / 8 + QK_K / 4];
    let mut out = [0i8; QK_K];
    let mut m = 1u32;
    let mut y = 0usize;
    let mut half = 0usize;
    while half < 2 {
        let base = half * 32;
        let mut shift = 0u32;
        let mut round = 0usize;
        while round < 4 {
            let mut plane = 0usize;
            while plane < 2 {
                let lo = plane * 16;
                let mut l = 0usize;
                while l < 16 {
                    let hi: i32 = if u32::from(hm[l + lo]) & m != 0 { 0 } else { 4 };
                    out[y] = (i32::from((qs[base + l + lo] >> shift) & 3) - hi) as i8;
                    y += 1;
                    l += 1;
                }
                plane += 1;
            }
            shift += 2;
            m <<= 1;
            round += 1;
        }
        half += 1;
    }
    Ok(out)
}

/// Q3_K → `f32`: `w = d·(sc − 32)·code`.
pub fn dequant_q3_k_block(blk: &[u8], out: &mut [f32; QK_K]) -> Result<(), QuantError> {
    need(blk, Q3_K_BLOCK_BYTES)?;
    let d_all = f16le(blk, QK_K / 8 + QK_K / 4 + 12);
    let scales = q3_k_scales(blk)?;
    let codes = q3_k_codes(blk)?;
    let mut g = 0usize;
    while g < 16 {
        let dl = d_all * (f32::from(scales[g]) - 32.0);
        let mut l = 0usize;
        while l < 16 {
            out[g * 16 + l] = dl * f32::from(codes[g * 16 + l]);
            l += 1;
        }
        g += 1;
    }
    Ok(())
}

// ── Q4_K ────────────────────────────────────────────────────────────────────────────────────────

/// ggml `get_scale_min_k4`: the `j`-th of the eight packed 6-bit `(scale, min)` pairs, `j < 8`.
pub fn q4_k_scale_min(scales: &[u8; Q4_K_SCALE_SIZE], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// The 256 four-bit `Q4_K` codes, in output order: four 64-weight groups of 32 low nibbles then
/// 32 high nibbles over a shared 32-byte `qs` window.
pub fn q4_k_codes(blk: &[u8]) -> Result<[u8; QK_K], QuantError> {
    need(blk, Q4_K_BLOCK_BYTES)?;
    let qs = &blk[4 + Q4_K_SCALE_SIZE..];
    let mut out = [0u8; QK_K];
    let mut b = 0usize;
    while b < 4 {
        let mut l = 0usize;
        while l < 32 {
            out[64 * b + l] = qs[32 * b + l] & 0xF;
            out[64 * b + 32 + l] = qs[32 * b + l] >> 4;
            l += 1;
        }
        b += 1;
    }
    Ok(out)
}

/// The eight `(scale, min)` pairs of a Q4_K block, in output-group order (two groups per pair).
pub fn q4_k_scale_mins(blk: &[u8]) -> Result<[(u8, u8); 8], QuantError> {
    need(blk, 4 + Q4_K_SCALE_SIZE)?;
    let mut sc = [0u8; Q4_K_SCALE_SIZE];
    sc.copy_from_slice(&blk[4..4 + Q4_K_SCALE_SIZE]);
    let mut out = [(0u8, 0u8); 8];
    let mut j = 0usize;
    while j < 8 {
        out[j] = q4_k_scale_min(&sc, j);
        j += 1;
    }
    Ok(out)
}

/// Q4_K → `f32`: `w = d·sc·code − dmin·min`. Each 32-byte `qs` window serves two 32-weight groups,
/// low nibbles with pair `2b` and high nibbles with pair `2b+1` (ggml `get_scale_min_k4` order).
pub fn dequant_q4_k_block(blk: &[u8], out: &mut [f32; QK_K]) -> Result<(), QuantError> {
    need(blk, Q4_K_BLOCK_BYTES)?;
    let d = f16le(blk, 0);
    let dmin = f16le(blk, 2);
    let pairs = q4_k_scale_mins(blk)?;
    let codes = q4_k_codes(blk)?;
    let mut b = 0usize;
    while b < 4 {
        let (sl, ml) = pairs[2 * b];
        let (sh, mh) = pairs[2 * b + 1];
        let d1 = d * f32::from(sl);
        let m1 = dmin * f32::from(ml);
        let d2 = d * f32::from(sh);
        let m2 = dmin * f32::from(mh);
        let mut l = 0usize;
        while l < 32 {
            out[64 * b + l] = d1 * f32::from(codes[64 * b + l]) - m1;
            out[64 * b + 32 + l] = d2 * f32::from(codes[64 * b + 32 + l]) - m2;
            l += 1;
        }
        b += 1;
    }
    Ok(())
}

// ── Q6_K ────────────────────────────────────────────────────────────────────────────────────────

/// The sixteen signed Q6_K group scales (one per 16 output weights).
pub fn q6_k_scales(blk: &[u8]) -> Result<[i8; 16], QuantError> {
    need(blk, Q6_K_BLOCK_BYTES)?;
    let mut out = [0i8; 16];
    let mut k = 0usize;
    while k < 16 {
        out[k] = blk[QK_K / 2 + QK_K / 4 + k] as i8;
        k += 1;
    }
    Ok(out)
}

/// The raw little-endian `f16` bits of the Q6_K scale `d`.
pub fn q6_k_d(blk: &[u8]) -> Result<u16, QuantError> {
    need(blk, Q6_K_BLOCK_BYTES)?;
    Ok(u16le(blk, QK_K / 2 + QK_K / 4 + QK_K / 16))
}

/// The 256 signed six-bit `Q6_K` codes, in output order (already offset by −32).
pub fn q6_k_codes(blk: &[u8]) -> Result<[i8; QK_K], QuantError> {
    need(blk, Q6_K_BLOCK_BYTES)?;
    let ql = &blk[..QK_K / 2];
    let qh = &blk[QK_K / 2..QK_K / 2 + QK_K / 4];
    let mut out = [0i8; QK_K];
    let mut n = 0usize;
    while n < 2 {
        let qb64 = 64 * n;
        let qb32 = 32 * n;
        let mut l = 0usize;
        while l < 32 {
            let hi = qh[qb32 + l];
            let q1 = ((ql[qb64 + l] & 0xF) | (((hi >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[qb64 + l + 32] & 0xF) | (((hi >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[qb64 + l] >> 4) | (((hi >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[qb64 + l + 32] >> 4) | (((hi >> 6) & 3) << 4)) as i32 - 32;
            out[128 * n + l] = q1 as i8;
            out[128 * n + l + 32] = q2 as i8;
            out[128 * n + l + 64] = q3 as i8;
            out[128 * n + l + 96] = q4 as i8;
            l += 1;
        }
        n += 1;
    }
    Ok(out)
}

/// Q6_K → `f32`: `w = d·sc·code`, sixteen 16-weight groups.
pub fn dequant_q6_k_block(blk: &[u8], out: &mut [f32; QK_K]) -> Result<(), QuantError> {
    need(blk, Q6_K_BLOCK_BYTES)?;
    let d = f16le(blk, QK_K / 2 + QK_K / 4 + QK_K / 16);
    let scales = q6_k_scales(blk)?;
    let codes = q6_k_codes(blk)?;
    let mut g = 0usize;
    while g < 16 {
        let dl = d * f32::from(scales[g]);
        let mut l = 0usize;
        while l < 16 {
            out[g * 16 + l] = dl * f32::from(codes[g * 16 + l]);
            l += 1;
        }
        g += 1;
    }
    Ok(())
}

/// Dequantize one 256-weight super-block of type `t` from `blk` into `out`.
pub fn dequant_block(t: u32, blk: &[u8], out: &mut [f32; QK_K]) -> Result<(), QuantError> {
    match t {
        GGML_TYPE_Q2_K => dequant_q2_k_block(blk, out),
        GGML_TYPE_Q3_K => dequant_q3_k_block(blk, out),
        GGML_TYPE_Q4_K => dequant_q4_k_block(blk, out),
        GGML_TYPE_Q6_K => dequant_q6_k_block(blk, out),
        other => Err(QuantError::UnknownType(other)),
    }
}

/// Dequantize `n` weights of super-block type `t` from `raw` into `out[..n]`.
///
/// The payload must hold `ceil(n / QK_K)` whole blocks; a short payload refuses, it is not
/// truncated silently. Returns the number of weights written.
pub fn dequant_super_block(
    t: u32,
    raw: &[u8],
    n: usize,
    out: &mut [f32],
) -> Result<usize, QuantError> {
    let (qk, bb) = block_geom(t).ok_or(QuantError::UnknownType(t))?;
    if qk != QK_K {
        return Err(QuantError::UnknownType(t));
    }
    let take = n.min(out.len());
    let nb = take.div_ceil(QK_K);
    let need_bytes = nb.checked_mul(bb).ok_or(QuantError::LengthOverflow)?;
    need(raw, need_bytes)?;
    let mut blk_out = [0f32; QK_K];
    let mut block_err: Option<QuantError> = None;
    let mut b = 0usize;
    while b < nb && block_err.is_none() {
        let o = b * bb;
        match dequant_block(t, &raw[o..o + bb], &mut blk_out) {
            Ok(()) => {
                let base = b * QK_K;
                let count = QK_K.min(take - base);
                out[base..base + count].copy_from_slice(&blk_out[..count]);
                b += 1;
            }
            Err(e) => block_err = Some(e),
        }
    }
    if let Some(e) = block_err {
        return Err(e);
    }
    Ok(take)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        reason = "the fixtures use exactly representable powers of two, so the dequant output is \
                  compared bit-for-bit against the reference value; a tolerance would hide a \
                  wrong scale"
    )]

    use super::*;

    fn h(v: f32) -> u16 {
        // Test helper: nearest even half of a value in the normal range (matches the reference
        // narrowing for the exact powers of two used here).
        let x = v.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xFF) as i32;
        let m = x & 0x007F_FFFF;
        let he = e - 112;
        assert!(he > 0 && he < 31);
        sign | ((he as u16) << 10) | ((m >> 13) as u16)
    }

    #[test]
    fn geometry_is_consistent() {
        assert_eq!(block_geom(GGML_TYPE_Q8_0), Some((32, 34)));
        assert_eq!(block_geom(GGML_TYPE_Q5_0), Some((32, 22)));
        assert_eq!(block_geom(GGML_TYPE_Q2_K), Some((256, 84)));
        assert_eq!(block_geom(GGML_TYPE_Q3_K), Some((256, 110)));
        assert_eq!(block_geom(GGML_TYPE_Q4_K), Some((256, 144)));
        assert_eq!(block_geom(GGML_TYPE_Q6_K), Some((256, 210)));
        assert_eq!(block_geom(3), None);
    }

    #[test]
    fn low_bit_geometry_and_names() {
        // QK and block bytes from `ggml-common.h:185-316`: 18, 18, 34, 66, 54, 28.
        assert_eq!(block_geom(GGML_TYPE_Q1_0), Some((128, 18)));
        assert_eq!(block_geom(GGML_TYPE_Q2_0), Some((64, 18)));
        assert_eq!(block_geom(GGML_TYPE_PQ2_0), Some((128, 34)));
        assert_eq!(block_geom(GGML_TYPE_TQ2_0), Some((256, 66)));
        assert_eq!(block_geom(GGML_TYPE_TQ1_0), Some((256, 54)));
        assert_eq!(block_geom(GGML_TYPE_PTQ1_0), Some((128, 28)));
        assert_eq!(ggml_type_name(GGML_TYPE_Q1_0), "Q1_0");
        assert_eq!(ggml_type_name(GGML_TYPE_Q2_0), "Q2_0");
        assert_eq!(ggml_type_name(GGML_TYPE_PQ2_0), "PQ2_0");
        assert_eq!(ggml_type_name(GGML_TYPE_TQ2_0), "TQ2_0");
        assert_eq!(ggml_type_name(GGML_TYPE_TQ1_0), "TQ1_0");
        assert_eq!(ggml_type_name(GGML_TYPE_PTQ1_0), "PTQ1_0");
        assert_eq!(required_bytes(GGML_TYPE_TQ1_0, 256).unwrap(), 54);
        assert_eq!(required_bytes(GGML_TYPE_PTQ1_0, 129).unwrap(), 56);
    }

    #[test]
    fn required_bytes_checks_overflow() {
        assert_eq!(required_bytes(GGML_TYPE_Q4_K, 256).unwrap(), 144);
        assert_eq!(required_bytes(GGML_TYPE_Q6_K, 1).unwrap(), 210);
        assert_eq!(required_bytes(GGML_TYPE_Q5_0, 32).unwrap(), 22);
        assert_eq!(required_bytes(GGML_TYPE_Q5_0, 33).unwrap(), 44);
        assert_eq!(
            required_bytes(GGML_TYPE_F32, u64::MAX),
            Err(QuantError::LengthOverflow)
        );
        assert_eq!(
            required_bytes(GGML_TYPE_Q8_0, u64::MAX),
            Err(QuantError::LengthOverflow)
        );
        // Q5_0 is the one type whose 22-byte block cannot overflow: the largest block count is
        // ceil((2^64-1)/32) = 2^59, and 2^59 * 22 still fits in u64. Pin the boundary value.
        assert_eq!(
            required_bytes(GGML_TYPE_Q5_0, u64::MAX).unwrap(),
            (u64::MAX / 32 + 1) * 22
        );
        assert_eq!(required_bytes(3, 8), Err(QuantError::UnknownType(3)));
    }

    #[test]
    fn short_payload_is_refused_not_truncated() {
        let mut out = [0f32; QK_K];
        let short = [0u8; 83];
        assert_eq!(
            dequant_q2_k_block(&short, &mut out),
            Err(QuantError::Truncated { need: 84, have: 83 })
        );
        let short4 = [0u8; 143];
        assert_eq!(
            dequant_q4_k_block(&short4, &mut out),
            Err(QuantError::Truncated {
                need: 144,
                have: 143
            })
        );
        let short6 = [0u8; 209];
        assert_eq!(
            dequant_q6_k_block(&short6, &mut out),
            Err(QuantError::Truncated {
                need: 210,
                have: 209
            })
        );
        let short5 = [0u8; 21];
        assert_eq!(
            q5_0_codes(&short5),
            Err(QuantError::Truncated { need: 22, have: 21 })
        );
        assert_eq!(
            q5_0_d(&short5),
            Err(QuantError::Truncated { need: 22, have: 21 })
        );
        let mut dst = [0f32; QK_K];
        assert!(dequant_super_block(GGML_TYPE_Q4_K, &short4, QK_K, &mut dst).is_err());
    }

    #[test]
    fn q8_0_dequant_matches_the_reference() {
        let mut blk = vec![0u8; Q8_0_BLOCK_BYTES];
        let d = h(0.5);
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2] = 2u8; // +2
        blk[3] = 0xFE; // -2
        let codes = q8_0_codes(&blk, 0).unwrap();
        assert_eq!(codes[0], 2);
        assert_eq!(codes[1], -2);
        assert_eq!(q8_0_scale(&blk, 0).unwrap(), 0.5);
    }

    #[test]
    fn q4_k_codes_and_scale_min_unpack() {
        let mut blk = vec![0u8; Q4_K_BLOCK_BYTES];
        // scales[12] at offset 4: first four bytes are scale 0..3, next four min 0..3.
        blk[4] = 5;
        blk[8] = 7;
        // qs[128] at offset 16: group 0 low nibble / high nibble.
        blk[16] = 0xA3;
        let codes = q4_k_codes(&blk).unwrap();
        assert_eq!(codes[0], 0x3);
        assert_eq!(codes[32], 0xA);
        let pairs = q4_k_scale_mins(&blk).unwrap();
        assert_eq!(pairs[0], (5, 7));
    }

    #[test]
    fn q2_k_codes_are_two_bit_planes() {
        let mut blk = vec![0u8; Q2_K_BLOCK_BYTES];
        blk[16] = 0b11_10_01_00;
        let codes = q2_k_codes(&blk).unwrap();
        // Byte 16 carries four 2-bit fields; consecutive planes land 32 outputs apart.
        assert_eq!(codes[0], 0b00);
        assert_eq!(codes[32], 0b01);
        assert_eq!(codes[64], 0b10);
        assert_eq!(codes[96], 0b11);
        assert_eq!(codes[16], 0);
    }

    #[test]
    fn q6_k_codes_fold_the_high_plane() {
        let mut blk = vec![0u8; Q6_K_BLOCK_BYTES];
        // ql[0] low nibble = 0x1, qh[0] bits 0..2 = 0b01 -> code = 0x11 - 32 = -15.
        blk[0] = 0x01;
        blk[QK_K / 2] = 0b0000_0001;
        let codes = q6_k_codes(&blk).unwrap();
        assert_eq!(codes[0], -15);
        assert_eq!(codes[32], -32);
    }

    #[test]
    fn q5_0_codes_fold_the_high_plane() {
        // Hand-checked fixture. qh[0] = 0b0000_0101 sets the fifth bit of low-plane outputs 0 and 2;
        // qh[2] = 0b0000_0001 sets it for high-plane output 16 (overall bit 16).
        let mut blk = vec![0u8; Q5_0_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&h(0.5).to_le_bytes());
        blk[2] = 0b0000_0101;
        blk[4] = 0b0000_0001;
        blk[6 + 0] = 0xFF; // low nibble 0xF | 0x10 = 31 -> +15; high nibble 0xF | 0x10 -> +15
        blk[6 + 1] = 0x00; // low nibble 0 -> 0 | 0 = 0 -> -16
        blk[6 + 2] = 0x10; // low nibble 0 -> 0 | 0x10 = 16 -> 0; high nibble 1 -> 1 | 0 = 1 -> -15
        let codes = q5_0_codes(&blk).unwrap();
        assert_eq!(codes[0], 15);
        assert_eq!(codes[1], -16);
        assert_eq!(codes[2], 0);
        assert_eq!(codes[16], 15);
        assert_eq!(codes[17], -16);
        assert_eq!(codes[18], -15);
        assert_eq!(codes[3], -16);
        // d = 0.5 and the raw bits decode to the same value.
        assert_eq!(q5_0_d(&blk).unwrap(), h(0.5));
        assert_eq!(q5_0_scale(&blk, 0).unwrap(), 0.5);
    }

    #[test]
    fn k_quant_scale_and_scale_bits_accessors_are_public_and_checked() {
        let mut blk = vec![0u8; Q2_K_BLOCK_BYTES];
        blk[0] = 0x51; // group 0: scale 1, min 5
        blk[QK_K / 16 + QK_K / 4 + 1] = 0x3C; // d bits of 1.0 (little-endian f16 0x3C00)
        blk[QK_K / 16 + QK_K / 4 + 3] = 0x38; // dmin bits of 0.5 (little-endian f16 0x3800)
        let scales = q2_k_scales(&blk).unwrap();
        assert_eq!(scales[0], 0x51);
        assert_eq!(q2_k_scale_min(scales[0]), (1, 5));
        assert_eq!(q2_k_d(&blk).unwrap(), 0x3C00);
        assert_eq!(q2_k_dmin(&blk).unwrap(), 0x3800);
        let mut b3 = vec![0u8; Q3_K_BLOCK_BYTES];
        b3[QK_K / 8 + QK_K / 4 + 13] = 0x3C;
        assert_eq!(q3_k_d(&b3).unwrap(), 0x3C00);
        let mut b6 = vec![0u8; Q6_K_BLOCK_BYTES];
        b6[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] = 0x38;
        assert_eq!(q6_k_d(&b6).unwrap(), 0x3800);
        let short = [0u8; 1];
        assert!(q2_k_d(&short).is_err());
        assert!(q3_k_d(&short).is_err());
        assert!(q6_k_d(&short).is_err());
    }

    #[test]
    fn f16_roundtrip_known_values() {
        assert_eq!(f16_to_f32(h(0.5)), 0.5);
        assert_eq!(f16_to_f32(h(1.0)), 1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn dequant_zero_block_is_constant_min() {
        let mut blk = vec![0u8; Q4_K_BLOCK_BYTES];
        // d = 1.0, dmin = 1.0, but sc = 0, m = 0 -> every weight 0.
        blk[0..2].copy_from_slice(&h(1.0).to_le_bytes());
        blk[2..4].copy_from_slice(&h(1.0).to_le_bytes());
        let mut out = [9.0f32; QK_K];
        dequant_q4_k_block(&blk, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
