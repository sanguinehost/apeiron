//! One parameterized packing grammar for quantized weight blocks.
//!
//! The GGUF readers in [`crate::format::quant`] are import formats: each has its own byte
//! geometry, code layout and scale packing. This module states the single decode grammar those
//! formats are converted into. A [`Packing`] fixes the code width, the block and group geometry,
//! the scale representation, the affine min term and the code bit/nibble order; one
//! [`Packing::decode_block`] and one [`Packing::row_decode_fixed`] turn raw bytes into **Q32.32**
//! `i64` weights.
//!
//! The grammar is the executing share of the format layer. [`Packing::from_gguf_kind`] names the
//! parameter set a ggml type converts to, and [`convert_row`] imports that type's bytes into the
//! canonical block layout before decoding. The values [`convert_row`] returns equal the integer
//! values the `quant` accessors carry; the tests below pin that equality.
//!
//! **Conversion, not a decoder copy.** The import side extracts the ggml scale, coefficient and
//! code fields with the `quant` integer accessors and lays them out canonically; the grammar then
//! re-derives the affine value. Where a ggml type's layout has no parameterization here,
//! [`Packing::from_gguf_kind`] returns `None` and the type stays a `quant`-only decoder.
//!
//! `F16` and `F32` are the terminal, per-element faces of the grammar: `block = 1`, `group = 1`,
//! an identity scale and no code plane. [`Packing::decode_block`] is element-wise for them — `F16`
//! decodes exactly per element and `F32` applies one round-to-nearest [`f32_bits_to_fixed`]
//! reduction — so the terminal faces are the "knob off" limit of the block grammar rather than an
//! import-only special case. The remaining inexpressible kinds are the codebook `IQ*` types, the
//! `Q4_1`/`Q5_1` block-wide min layout and Q5_0's native 5-bit nuance.

use super::quant;

/// The Q32.32 unit: `ONE` denotes `1.0`, so one unit is `2^-32`.
pub const ONE: i64 = 1 << 32;

/// The canonical code width: the number of code bits per weight, before any high plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeWidth {
    /// A `W`-bit code, `W` in `{2, 3, 4, 6, 8}`.
    Fixed(u8),
    /// A balanced-trit code over `L` three-valued planes: `L` 2-bit trit fields, the field value
    /// `f` denoting the trit `f - 1`, plane `l` weighted by `3^l`.
    Trit(u8),
    /// The IEEE-754 half face: one 16-bit element per block, decoded exactly to Q32.32.
    F16,
    /// The IEEE-754 single face: one 32-bit element per block, decoded with one round-to-nearest
    /// reduction to Q32.32.
    F32,
}

/// How a scale field is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleKind {
    /// The exact IEEE-754 half bits. Every finite `f16` is an integer multiple of `2^-24`, hence
    /// of the `2^-32` unit, so the decode is exact and `Inf`/`NaN` refuse.
    F16,
    /// One byte carrying a biased power-of-two exponent `e`: the scale is `2^(e - 127)`, truncated
    /// toward zero at the `2^-32` unit.
    Pow2Exp8,
    /// The scale is the Q32.32 unit `1.0`. Used by the terminal `F16`/`F32` faces, which carry no
    /// scale field and decode each element by its own bit pattern.
    Identity,
}

/// The order in which a code's bits are laid down inside the code area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneOrder {
    /// Code bit `0` (least significant) first.
    LsbFirst,
    /// Code bit `total - 1` (most significant) first.
    MsbFirst,
}

/// Which nibble of a code-area byte holds the earlier bit group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NibbleOrder {
    /// Low nibble first.
    LowHigh,
    /// High nibble first.
    HighLow,
}

/// A refused packing operation. Integer-only so callers can match without formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingError {
    /// The code width is outside the grammar's admitted set.
    BadWidth(u8),
    /// The block size is not one of `32`, `64`, `256`.
    BadBlock(u32),
    /// The group does not divide the block, or a group count was inconsistent.
    BadGeometry {
        /// Block size as stated.
        block: u32,
        /// Group size as stated.
        group: u32,
    },
    /// The payload was shorter than the geometry requires.
    Truncated {
        /// Bytes the geometry requires.
        need: usize,
        /// Bytes actually available.
        have: usize,
    },
    /// A canonical block field was outside the declared count.
    BadBlockField,
    /// An `f16` scale was `Inf` or `NaN`.
    NonFiniteScale(u16),
    /// An `f32` value was `Inf` or `NaN`.
    NonFiniteFloat(u32),
    /// A value did not fit the Q32.32 range: overflow above `i64` or underflow of a nonzero value
    /// to zero. No value is clamped or flushed to zero.
    OutOfRange,
    /// A fixed-point product did not fit `i64`.
    Overflow,
    /// A ggml type id has no packing in this grammar.
    UnsupportedKind(u32),
    /// The `quant` import layer refused.
    Quant(quant::QuantError),
}

impl From<quant::QuantError> for PackingError {
    fn from(e: quant::QuantError) -> Self {
        PackingError::Quant(e)
    }
}

impl core::fmt::Display for PackingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PackingError::BadWidth(w) => write!(f, "code width {w} not in {{2,3,4,6,8}}"),
            PackingError::BadBlock(b) => write!(f, "block {b} not in {{32,64,256}}"),
            PackingError::BadGeometry { block, group } => {
                write!(f, "group {group} does not divide block {block}")
            }
            PackingError::Truncated { need, have } => {
                write!(f, "truncated: need {need} bytes, have {have}")
            }
            PackingError::BadBlockField => write!(f, "canonical block field count mismatch"),
            PackingError::NonFiniteScale(bits) => write!(f, "non-finite f16 scale 0x{bits:04x}"),
            PackingError::NonFiniteFloat(bits) => write!(f, "non-finite f32 value 0x{bits:08x}"),
            PackingError::OutOfRange => write!(f, "value outside the Q32.32 range"),
            PackingError::Overflow => write!(f, "fixed-point overflow"),
            PackingError::UnsupportedKind(t) => write!(f, "ggml type {t} has no packing"),
            PackingError::Quant(e) => write!(f, "quant: {e}"),
        }
    }
}

impl std::error::Error for PackingError {}

/// Decode the exact `f16` bit pattern `bits` into Q32.32, refusing `Inf`/`NaN`.
///
/// This mirrors the executing path's `f16_to_fixed` contract: a finite `f16` is an integer
/// multiple of `2^-24`, hence of the `2^-32` unit, so the decode is exact with no rounding.
///
/// # Errors
///
/// [`PackingError::NonFiniteScale`] when the exponent is `0x1f`.
pub fn f16_bits_to_fixed(bits: u16) -> Result<i64, PackingError> {
    let sign = bits >> 15;
    let e = (bits >> 10) & 0x1f;
    let m = bits & 0x3ff;
    let mag = if e == 0 {
        // Subnormal: value = m * 2^-24, Q32.32 = m * 2^8.
        (m as i64) << 8
    } else if e == 0x1f {
        return Err(PackingError::NonFiniteScale(bits));
    } else {
        // Normal: value = (1024 + m) * 2^(e-25), Q32.32 = (1024 + m) * 2^(e+7).
        // `e <= 30` here, so the shift is at most 37 and cannot leave `i64`.
        let sig = (m as i64) + 1024;
        sig << (e as u32 + 7)
    };
    Ok(if sign == 1 { -mag } else { mag })
}

/// Decode a biased power-of-two exponent byte into Q32.32, truncating toward zero.
///
/// The byte `e` denotes `2^(e - 127)`. The exact value is representable when `e >= 95`
/// (`2^-32`); below that the field truncates to `0`.
///
/// # Errors
///
/// [`PackingError::Overflow`] when the shift would leave `i64`.
pub fn pow2_exp_to_fixed(e: u8) -> Result<i64, PackingError> {
    let exp = (e as i32) - 127;
    let shift = exp + 32;
    if shift >= 0 {
        if shift >= 64 {
            Err(PackingError::Overflow)
        } else {
            Ok(1i64 << (shift as u32))
        }
    } else if -shift >= 64 {
        Ok(0)
    } else {
        Ok(1i64 >> (-shift as u32))
    }
}

/// Round `sig / 2^rbits` (`sig > 0`, `rbits >= 1`) to the nearest integer with ties toward `+∞` on
/// the signed value: a positive tie rounds up, a negative tie rounds toward zero. The caller fixes
/// the sign by negating the returned magnitude.
///
/// `sig` is a 24-bit significand, so for `rbits >= 25` the magnitude is below `1/2` and the result
/// is `0`.
fn round_scaled(sig: i64, rbits: u32, positive: bool) -> i64 {
    if rbits >= 25 {
        return 0;
    }
    let q = sig >> rbits;
    let rem = sig & ((1i64 << rbits) - 1);
    let half = 1i64 << (rbits - 1);
    if positive {
        if rem >= half {
            q + 1
        } else {
            q
        }
    } else if rem > half {
        q + 1
    } else {
        q
    }
}

/// Decode the exact IEEE-754 single bit pattern `bits` into Q32.32.
///
/// `v` denotes `round(v · 2^32)`, the module's one round-to-nearest reduction, ties toward `+∞`
/// (the [`round_shift_32`] convention: `floor(v · 2^32 + 1/2)`). The result is the exact value
/// times `2^32` for every `f32` that is already an integer multiple of `2^-32`; otherwise the
/// single rounding below is applied. `NaN`, `Inf`, overflow above the `i64` range and underflow of
/// a nonzero value to zero refuse. No value is clamped or flushed to zero.
///
/// The conversion is integer-only, read from the `u32` bit pattern, so no float external appears on
/// the extraction path.
///
/// # Errors
///
/// [`PackingError::NonFiniteFloat`] on `NaN`/`Inf`; [`PackingError::OutOfRange`] on overflow or
/// underflow.
pub fn f32_bits_to_fixed(bits: u32) -> Result<i64, PackingError> {
    let sign = bits >> 31;
    let e = (bits >> 23) & 0xff;
    let m = bits & 0x7f_ffff;
    if e == 0xff {
        return Err(PackingError::NonFiniteFloat(bits));
    }
    if e == 0 && m == 0 {
        return Ok(0);
    }
    // Normal: value = (2^23 + m) · 2^(e - 150); subnormal: value = m · 2^-149.
    let (sig, shift) = if e == 0 {
        (m as i64, -117i32)
    } else {
        ((m as i64) + (1i64 << 23), e as i32 - 118)
    };
    // value · 2^32 = sig · 2^shift with sig > 0. A shift at or above the word width, or a
    // significand that would leave `i64`, is an overflow rather than a wrapped shift.
    let magnitude = if shift >= 0 {
        if shift >= 64 || sig > (i64::MAX >> (shift as u32)) {
            return Err(PackingError::OutOfRange);
        }
        sig << (shift as u32)
    } else {
        round_scaled(sig, (-shift) as u32, sign == 0)
    };
    if magnitude == 0 {
        // The value is nonzero and rounds to zero: underflow, refused rather than flushed.
        return Err(PackingError::OutOfRange);
    }
    Ok(if sign == 1 { -magnitude } else { magnitude })
}

/// One canonical block: the scale fields, the per-group affine coefficients and the code fields.
///
/// `codes` holds one interpreted code per weight, in output order. When the packing's
/// `signed_code` is set the code is two's complement over `total_code_bits`; otherwise it is an
/// unsigned field. The effective code the epilogue multiplies is `codes[i] + bias`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalBlock {
    /// The scale field: `f16` bits, or the exponent byte in the low 8 bits.
    pub scale: u16,
    /// The min scale field when the packing has one, else `0`.
    pub min: u16,
    /// One `(scale coefficient, min coefficient)` pair per group.
    pub coeffs: Vec<(i32, i32)>,
    /// One code field per weight.
    pub codes: Vec<i64>,
}

/// The parameterized decode grammar. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Packing {
    /// Code width before the high planes.
    pub width: CodeWidth,
    /// Extra one-bit planes above `width` (for example `1` for a 5-bit code, `2` for a 6-bit one).
    pub high_planes: u8,
    /// Weights per block: `32`, `64` or `256`.
    pub block: u32,
    /// Weights per affine group; must divide `block`.
    pub group: u32,
    /// How the scale and min fields are encoded.
    pub scale_kind: ScaleKind,
    /// Whether the grammar carries a min scale and a per-group min coefficient.
    pub has_min: bool,
    /// Whether the code field is two's complement rather than unsigned.
    pub signed_code: bool,
    /// Constant added to the code field before the affine multiply.
    pub bias: i32,
    /// Whether the per-group coefficients are signed bytes rather than unsigned.
    pub signed_coeff: bool,
    /// Code bit order.
    pub plane_order: PlaneOrder,
    /// Code nibble order.
    pub nibble_order: NibbleOrder,
}

/// The ggml types this grammar expresses as a packing.
pub const PACKED_GGUF_KINDS: [u32; 8] = [
    quant::GGML_TYPE_F32,
    quant::GGML_TYPE_F16,
    quant::GGML_TYPE_Q8_0,
    quant::GGML_TYPE_Q5_0,
    quant::GGML_TYPE_Q2_K,
    quant::GGML_TYPE_Q3_K,
    quant::GGML_TYPE_Q4_K,
    quant::GGML_TYPE_Q6_K,
];

impl Packing {
    /// The number of bits in one code field (`width + high_planes`, or `2 * L` for a trit face).
    ///
    /// # Errors
    ///
    /// [`PackingError::BadWidth`] when the width is outside the grammar.
    pub fn total_code_bits(&self) -> Result<usize, PackingError> {
        match self.width {
            CodeWidth::Fixed(w) => {
                if !matches!(w, 2 | 3 | 4 | 6 | 8) {
                    return Err(PackingError::BadWidth(w));
                }
                Ok(w as usize + self.high_planes as usize)
            }
            CodeWidth::Trit(l) => {
                if l == 0 {
                    return Err(PackingError::BadWidth(0));
                }
                Ok(2 * (l as usize))
            }
            CodeWidth::F16 => Ok(16),
            CodeWidth::F32 => Ok(32),
        }
    }

    /// Whether the width is a terminal float face (`F16`/`F32`): one element per block, decoded
    /// element-wise from its own bit pattern with no scale, min or coefficient field.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self.width, CodeWidth::F16 | CodeWidth::F32)
    }

    /// The number of weights in a block as a `usize`.
    ///
    /// # Errors
    ///
    /// [`PackingError::BadBlock`] when the block size is not `32`, `64` or `256`, or is not `1` for
    /// a terminal `F16`/`F32` face.
    pub fn block_size(&self) -> Result<usize, PackingError> {
        if self.is_terminal() {
            return if self.block == 1 {
                Ok(1)
            } else {
                Err(PackingError::BadBlock(self.block))
            };
        }
        match self.block {
            32 | 64 | 256 => Ok(self.block as usize),
            other => Err(PackingError::BadBlock(other)),
        }
    }

    /// The number of affine groups in a block.
    ///
    /// # Errors
    ///
    /// [`PackingError::BadGeometry`] when the group is zero or does not divide the block.
    pub fn groups(&self) -> Result<usize, PackingError> {
        let b = self.block_size()?;
        if self.group == 0 || b % self.group as usize != 0 {
            return Err(PackingError::BadGeometry {
                block: self.block,
                group: self.group,
            });
        }
        Ok(b / self.group as usize)
    }

    /// The bytes of one scale or min field.
    #[must_use]
    pub fn scale_field_bytes(&self) -> usize {
        match self.scale_kind {
            ScaleKind::F16 => 2,
            ScaleKind::Pow2Exp8 => 1,
            ScaleKind::Identity => 0,
        }
    }

    /// The bytes of one canonical block in this grammar.
    ///
    /// # Errors
    ///
    /// A geometry refusal or [`PackingError::Overflow`] on the byte arithmetic.
    pub fn block_bytes(&self) -> Result<usize, PackingError> {
        let groups = self.groups()?;
        let scale = self.scale_field_bytes();
        let min = if self.has_min { scale } else { 0 };
        // A terminal face carries no coefficient field: its payload is exactly the value bytes.
        let coeff = if self.is_terminal() {
            0
        } else {
            groups * if self.has_min { 2 } else { 1 }
        };
        let bits = self
            .block_size()?
            .checked_mul(self.total_code_bits()?)
            .ok_or(PackingError::Overflow)?;
        let code = bits / 8 + (bits % 8 != 0) as usize;
        let total = scale.checked_add(min).ok_or(PackingError::Overflow)?;
        let total = total.checked_add(coeff).ok_or(PackingError::Overflow)?;
        total.checked_add(code).ok_or(PackingError::Overflow)
    }

    /// Validate the whole parameter set.
    ///
    /// # Errors
    ///
    /// Any of the geometry, width or block refusals.
    pub fn validate(&self) -> Result<(), PackingError> {
        self.total_code_bits()?;
        self.block_size()?;
        self.groups()?;
        self.block_bytes()?;
        Ok(())
    }

    /// The terminal `F16`/`F32` parameter set: one element per block, identity scale, no code
    /// plane, no min and no coefficient field.
    fn terminal(width: CodeWidth) -> Packing {
        Packing {
            width,
            high_planes: 0,
            block: 1,
            group: 1,
            scale_kind: ScaleKind::Identity,
            has_min: false,
            signed_code: false,
            bias: 0,
            signed_coeff: false,
            plane_order: PlaneOrder::LsbFirst,
            nibble_order: NibbleOrder::LowHigh,
        }
    }

    /// The parameter set a ggml type converts to, or `None` when the grammar does not express it.
    ///
    /// `F32` and `F16` are the terminal, per-element faces: `block = 1`, `group = 1`, identity
    /// scale, no code plane and no min. `F16` decodes exactly per element; `F32` applies the one
    /// round-to-nearest [`f32_bits_to_fixed`] reduction. The codebook `IQ*` types need their lookup
    /// grids; `Q4_1`/`Q5_1` keep a per-block `m` in a layout this grammar's flat coefficient field
    /// does not carry; `Q5_0`'s native 5-bit code is expressed as a 4-bit code plus one high plane.
    /// Those stay `quant`-only.
    #[must_use]
    pub fn from_gguf_kind(kind: u32) -> Option<Packing> {
        let p = match kind {
            quant::GGML_TYPE_F32 => Self::terminal(CodeWidth::F32),
            quant::GGML_TYPE_F16 => Self::terminal(CodeWidth::F16),
            quant::GGML_TYPE_Q8_0 => Packing {
                width: CodeWidth::Fixed(8),
                high_planes: 0,
                block: 32,
                group: 32,
                scale_kind: ScaleKind::F16,
                has_min: false,
                signed_code: true,
                bias: 0,
                signed_coeff: false,
                plane_order: PlaneOrder::LsbFirst,
                nibble_order: NibbleOrder::LowHigh,
            },
            quant::GGML_TYPE_Q5_0 => Packing {
                width: CodeWidth::Fixed(4),
                high_planes: 1,
                block: 32,
                group: 32,
                scale_kind: ScaleKind::F16,
                has_min: false,
                signed_code: false,
                bias: -16,
                signed_coeff: false,
                plane_order: PlaneOrder::LsbFirst,
                nibble_order: NibbleOrder::LowHigh,
            },
            quant::GGML_TYPE_Q2_K => Packing {
                width: CodeWidth::Fixed(2),
                high_planes: 0,
                block: 256,
                group: 16,
                scale_kind: ScaleKind::F16,
                has_min: true,
                signed_code: false,
                bias: 0,
                signed_coeff: false,
                plane_order: PlaneOrder::LsbFirst,
                nibble_order: NibbleOrder::LowHigh,
            },
            quant::GGML_TYPE_Q3_K => Packing {
                width: CodeWidth::Fixed(3),
                high_planes: 0,
                block: 256,
                group: 16,
                scale_kind: ScaleKind::F16,
                has_min: false,
                signed_code: false,
                bias: -4,
                signed_coeff: true,
                plane_order: PlaneOrder::LsbFirst,
                nibble_order: NibbleOrder::LowHigh,
            },
            quant::GGML_TYPE_Q4_K => Packing {
                width: CodeWidth::Fixed(4),
                high_planes: 0,
                block: 256,
                group: 32,
                scale_kind: ScaleKind::F16,
                has_min: true,
                signed_code: false,
                bias: 0,
                signed_coeff: false,
                plane_order: PlaneOrder::LsbFirst,
                nibble_order: NibbleOrder::LowHigh,
            },
            quant::GGML_TYPE_Q6_K => Packing {
                width: CodeWidth::Fixed(4),
                high_planes: 2,
                block: 256,
                group: 16,
                scale_kind: ScaleKind::F16,
                has_min: false,
                signed_code: false,
                bias: -32,
                signed_coeff: true,
                plane_order: PlaneOrder::LsbFirst,
                nibble_order: NibbleOrder::LowHigh,
            },
            _ => return None,
        };
        Some(p)
    }

    /// Decode the scale field `raw` to Q32.32 under this packing's [`ScaleKind`].
    ///
    /// # Errors
    ///
    /// A non-finite `f16` or a fixed-point overflow.
    pub fn scale_to_fixed(&self, raw: u16) -> Result<i64, PackingError> {
        match self.scale_kind {
            ScaleKind::F16 => f16_bits_to_fixed(raw),
            ScaleKind::Pow2Exp8 => pow2_exp_to_fixed(raw as u8),
            ScaleKind::Identity => Ok(ONE),
        }
    }

    /// Read the code field of weight `i` back out of the canonical code area.
    fn unpack_code(&self, area: &[u8], i: usize, total: usize) -> i64 {
        let mut v: i64 = 0;
        let mut k = 0usize;
        while k < total {
            let shift = match self.plane_order {
                PlaneOrder::LsbFirst => k,
                PlaneOrder::MsbFirst => total - 1 - k,
            };
            let pos = i * total + k;
            let bit = ((area[pos / 8] >> (pos % 8)) & 1) as i64;
            v |= bit << shift;
            k += 1;
        }
        if self.signed_code && total < 64 {
            let sign_bit = 1i64 << (total - 1);
            if v & sign_bit != 0 {
                v -= 1i64 << total;
            }
        }
        v
    }

    /// Pack the code fields into the canonical code area, appending its bytes to `out`.
    fn pack_codes(&self, codes: &[i64], out: &mut Vec<u8>) -> Result<(), PackingError> {
        let total = self.total_code_bits()?;
        let block = self.block_size()?;
        if codes.len() != block {
            return Err(PackingError::BadBlockField);
        }
        let bits = block.checked_mul(total).ok_or(PackingError::Overflow)?;
        let nbytes = bits / 8 + (bits % 8 != 0) as usize;
        let start = out.len();
        let mut z = 0usize;
        while z < nbytes {
            out.push(0u8);
            z += 1;
        }
        let mut i = 0usize;
        while i < block {
            let v = codes[i];
            let mut k = 0usize;
            while k < total {
                let shift = match self.plane_order {
                    PlaneOrder::LsbFirst => k,
                    PlaneOrder::MsbFirst => total - 1 - k,
                };
                let bit = ((v >> shift) & 1) as u8;
                let pos = i * total + k;
                out[start + pos / 8] |= bit << (pos % 8);
                k += 1;
            }
            i += 1;
        }
        if self.nibble_order == NibbleOrder::HighLow {
            let mut j = 0usize;
            while j < nbytes {
                out[start + j] = out[start + j].rotate_left(4);
                j += 1;
            }
        }
        Ok(())
    }

    /// Decode one canonical block `raw` into the grammar's fields.
    ///
    /// # Errors
    ///
    /// [`PackingError::Truncated`] when `raw` is shorter than [`Packing::block_bytes`].
    #[allow(
        clippy::vec_init_then_push,
        reason = "explicit Vec::with_capacity plus push keeps the extracted body off the \
                  alloc::vec::from_elem external that `vec![x]` would introduce"
    )]
    pub fn decode_block_fields(&self, raw: &[u8]) -> Result<CanonicalBlock, PackingError> {
        let need = self.block_bytes()?;
        if raw.len() < need {
            return Err(PackingError::Truncated {
                need,
                have: raw.len(),
            });
        }
        let sf = self.scale_field_bytes();
        let mut at = 0usize;
        let mut scale = 0u16;
        let mut min = 0u16;
        let mut coeffs: Vec<(i32, i32)>;
        if self.is_terminal() {
            // A terminal face carries no scale, min or coefficient field; the payload is the value
            // bytes and the single affine coefficient is the identity.
            coeffs = Vec::with_capacity(1);
            coeffs.push((1, 0));
        } else {
            scale = read_field(raw, at, sf);
            at += sf;
            if self.has_min {
                min = read_field(raw, at, sf);
                at += sf;
            }
            let groups = self.groups()?;
            coeffs = Vec::with_capacity(groups);
            let mut g = 0usize;
            while g < groups {
                let a = if self.signed_coeff {
                    (raw[at] as i8) as i32
                } else {
                    raw[at] as i32
                };
                at += 1;
                let b = if self.has_min {
                    let bv = if self.signed_coeff {
                        (raw[at] as i8) as i32
                    } else {
                        raw[at] as i32
                    };
                    at += 1;
                    bv
                } else {
                    0
                };
                coeffs.push((a, b));
                g += 1;
            }
        }
        let total = self.total_code_bits()?;
        let block = self.block_size()?;
        let bits = block * total;
        let nbytes = bits / 8 + (bits % 8 != 0) as usize;
        let mut area: Vec<u8> = Vec::with_capacity(nbytes);
        let mut j = 0usize;
        while j < nbytes {
            area.push(raw[at + j]);
            j += 1;
        }
        if self.nibble_order == NibbleOrder::HighLow {
            let mut j = 0usize;
            while j < nbytes {
                area[j] = area[j].rotate_left(4);
                j += 1;
            }
        }
        let mut codes: Vec<i64> = Vec::with_capacity(block);
        let mut i = 0usize;
        while i < block {
            codes.push(self.unpack_code(&area, i, total));
            i += 1;
        }
        Ok(CanonicalBlock {
            scale,
            min,
            coeffs,
            codes,
        })
    }

    /// Decode one canonical block into Q32.32 weights: `w = scale*a*(q + bias) - min*b`.
    ///
    /// A terminal `F16`/`F32` face is element-wise instead: each code field is one IEEE-754 bit
    /// pattern, decoded exactly (`F16`) or with one round-to-nearest reduction (`F32`).
    ///
    /// All arithmetic is checked `i64`; a product outside the fixed-point range refuses.
    ///
    /// # Errors
    ///
    /// A truncation, geometry or overflow refusal.
    pub fn decode_block(&self, raw: &[u8]) -> Result<Vec<i64>, PackingError> {
        let b = self.decode_block_fields(raw)?;
        self.epilogue(&b)
    }

    /// Apply the affine epilogue to a decoded block.
    fn epilogue(&self, b: &CanonicalBlock) -> Result<Vec<i64>, PackingError> {
        let block = self.block_size()?;
        let group = self.group as usize;
        if b.codes.len() != block || b.coeffs.len() != self.groups()? {
            return Err(PackingError::BadBlockField);
        }
        if self.is_terminal() {
            // Element-wise: each code field is the raw IEEE-754 bit pattern of one weight.
            let mut out: Vec<i64> = Vec::with_capacity(block);
            let mut failed = false;
            let mut error = PackingError::Overflow;
            let mut i = 0usize;
            while i < block {
                if !failed {
                    match terminal_bits_to_fixed(self.width, b.codes[i]) {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            failed = true;
                            error = e;
                        }
                    }
                }
                i += 1;
            }
            return if failed { Err(error) } else { Ok(out) };
        }
        let scale = self.scale_to_fixed(b.scale)?;
        let min_scale = if self.has_min {
            self.scale_to_fixed(b.min)?
        } else {
            0
        };
        let mut out: Vec<i64> = Vec::with_capacity(block);
        let mut failed = false;
        let mut error = PackingError::Overflow;
        let mut i = 0usize;
        while i < block {
            if !failed {
                match self.epilogue_entry(b, i, scale, min_scale, group) {
                    Ok(v) => out.push(v),
                    Err(e) => {
                        failed = true;
                        error = e;
                    }
                }
            }
            i += 1;
        }
        if failed {
            Err(error)
        } else {
            Ok(out)
        }
    }

    /// The affine epilogue value of weight `i`: the per-element body of [`Packing::epilogue`],
    /// factored out so no fallible operation sits inside the output loop (Aeneas rejects early
    /// returns inside loops).
    fn epilogue_entry(
        &self,
        b: &CanonicalBlock,
        i: usize,
        scale: i64,
        min_scale: i64,
        group: usize,
    ) -> Result<i64, PackingError> {
        let (a, mb) = b.coeffs[i / group];
        let q = match self.width {
            CodeWidth::Trit(l) => balanced_trit(b.codes[i], l)?,
            // The terminal faces are handled before the affine epilogue.
            _ => b.codes[i]
                .checked_add(self.bias as i64)
                .ok_or(PackingError::Overflow)?,
        };
        let mut acc = mul(mul(scale, a as i64)?, q)?;
        if self.has_min {
            acc = sub(acc, mul(min_scale, mb as i64)?)?;
        }
        Ok(acc)
    }

    /// Decode `n` weights from `raw`, whole blocks in the canonical layout, truncated to `n`.
    ///
    /// # Errors
    ///
    /// [`PackingError::Truncated`] when `raw` carries fewer than `ceil(n / block)` blocks.
    pub fn row_decode_fixed(&self, raw: &[u8], n: usize) -> Result<Vec<i64>, PackingError> {
        let bb = self.block_bytes()?;
        let block = self.block_size()?;
        let blocks = n / block + (n % block != 0) as usize;
        let need = blocks.checked_mul(bb).ok_or(PackingError::Overflow)?;
        if raw.len() < need {
            return Err(PackingError::Truncated {
                need,
                have: raw.len(),
            });
        }
        let mut out: Vec<i64> = Vec::with_capacity(n);
        let mut failed: Option<PackingError> = None;
        let mut b = 0usize;
        // One carried error keeps `?` out of the loop; the first refusing block stops the walk
        // and its error is returned after it, in the original block order.
        while b < blocks && failed.is_none() {
            match self.decode_block(&raw[b * bb..b * bb + bb]) {
                Ok(vals) => {
                    let take = block.min(n - b * block);
                    let mut lane = 0usize;
                    while lane < take {
                        out.push(vals[lane]);
                        lane += 1;
                    }
                    b += 1;
                }
                Err(e) => failed = Some(e),
            }
        }
        if let Some(e) = failed {
            return Err(e);
        }
        Ok(out)
    }

    /// Encode one canonical block `b` into this grammar's byte layout.
    ///
    /// # Errors
    ///
    /// [`PackingError::BadBlockField`] when the field counts do not match the geometry, or an
    /// overflow in the byte arithmetic.
    pub fn encode_block(&self, b: &CanonicalBlock) -> Result<Vec<u8>, PackingError> {
        let capacity = self.block_bytes()?;
        let groups = self.groups()?;
        if b.codes.len() != self.block_size()? || b.coeffs.len() != groups {
            return Err(PackingError::BadBlockField);
        }
        let mut out: Vec<u8> = Vec::with_capacity(capacity);
        if !self.is_terminal() {
            let sf = self.scale_field_bytes();
            write_field(&mut out, b.scale, sf);
            if self.has_min {
                write_field(&mut out, b.min, sf);
            }
            let mut g = 0usize;
            while g < groups {
                let (a, mb) = b.coeffs[g];
                if self.signed_coeff {
                    out.push(a as i8 as u8);
                } else {
                    out.push((a as u8) & 0xff);
                }
                if self.has_min {
                    if self.signed_coeff {
                        out.push(mb as i8 as u8);
                    } else {
                        out.push((mb as u8) & 0xff);
                    }
                }
                g += 1;
            }
        }
        self.pack_codes(&b.codes, &mut out)?;
        Ok(out)
    }
}

/// Decode one terminal-face element from its raw bit pattern: `F16` exactly, `F32` with the single
/// round-to-nearest reduction. The caller guards [`Packing::is_terminal`].
fn terminal_bits_to_fixed(width: CodeWidth, bits: i64) -> Result<i64, PackingError> {
    match width {
        CodeWidth::F16 => f16_bits_to_fixed(bits as u16),
        CodeWidth::F32 => f32_bits_to_fixed(bits as u32),
        CodeWidth::Fixed(w) => Err(PackingError::BadWidth(w)),
        CodeWidth::Trit(l) => Err(PackingError::BadWidth(l)),
    }
}

/// The balanced value of a `2*L`-bit packed trit field vector.
fn balanced_trit(field: i64, l: u8) -> Result<i64, PackingError> {
    let mut acc: i64 = 0;
    let mut weight: i64 = 1;
    let mut k = 0usize;
    while k < l as usize {
        let shift = 2 * k;
        let trit = ((field >> shift) & 3) - 1;
        acc = acc
            .checked_add(trit.checked_mul(weight).ok_or(PackingError::Overflow)?)
            .ok_or(PackingError::Overflow)?;
        weight = weight.checked_mul(3).ok_or(PackingError::Overflow)?;
        k += 1;
    }
    Ok(acc)
}

/// Checked `a * b`.
fn mul(a: i64, b: i64) -> Result<i64, PackingError> {
    a.checked_mul(b).ok_or(PackingError::Overflow)
}

/// Checked `a - b`.
fn sub(a: i64, b: i64) -> Result<i64, PackingError> {
    a.checked_sub(b).ok_or(PackingError::Overflow)
}

/// Read a `n`-byte little-endian field; `n` is `1` or `2`.
fn read_field(raw: &[u8], at: usize, n: usize) -> u16 {
    if n == 1 {
        raw[at] as u16
    } else {
        u16::from_le_bytes([raw[at], raw[at + 1]])
    }
}

/// Append a `n`-byte little-endian field.
fn write_field(out: &mut Vec<u8>, v: u16, n: usize) {
    if n == 1 {
        out.push((v & 0xff) as u8);
    } else {
        out.push((v & 0xff) as u8);
        out.push((v >> 8) as u8);
    }
}

/// Import one ggml row into the canonical layout of `p`, appending `ceil(n / block)` blocks.
///
/// The ggml scale, coefficient and code fields are read with the `quant` integer accessors; the
/// affine algebra itself lives in the grammar, so this function only relocates fields.
///
/// # Errors
///
/// A truncation from the `quant` accessors, a geometry refusal, or an overflow.
pub fn import_row(kind: u32, p: &Packing, raw: &[u8], n: usize) -> Result<Vec<u8>, PackingError> {
    let block = p.block_size()?;
    let bb = p.block_bytes()?;
    let blocks = n / block + (n % block != 0) as usize;
    let mut out: Vec<u8> = Vec::with_capacity(blocks * bb);
    let mut failed: Option<PackingError> = None;
    let mut b = 0usize;
    // The fallible import and encode sit behind one carried error so no `?` runs inside the loop;
    // the first refusal stops the walk and is returned after it, in the original block order.
    while b < blocks && failed.is_none() {
        let fields = match kind {
            quant::GGML_TYPE_F32 => import_f32(raw, b),
            quant::GGML_TYPE_F16 => import_f16(raw, b),
            quant::GGML_TYPE_Q8_0 => import_q8_0(raw, b),
            quant::GGML_TYPE_Q5_0 => import_q5_0(raw, b),
            quant::GGML_TYPE_Q2_K => import_q2_k(raw, b),
            quant::GGML_TYPE_Q3_K => import_q3_k(raw, b),
            quant::GGML_TYPE_Q4_K => import_q4_k(raw, b),
            quant::GGML_TYPE_Q6_K => import_q6_k(raw, b),
            other => Err(PackingError::UnsupportedKind(other)),
        };
        match fields {
            Ok(f) => match p.encode_block(&f) {
                Ok(bytes) => {
                    let mut j = 0usize;
                    while j < bytes.len() {
                        out.push(bytes[j]);
                        j += 1;
                    }
                    b += 1;
                }
                Err(e) => failed = Some(e),
            },
            Err(e) => failed = Some(e),
        }
    }
    if let Some(e) = failed {
        return Err(e);
    }
    Ok(out)
}

/// Map each ggml kind's row into the canonical form and decode it to Q32.32.
///
/// The returned values equal the `quant` decoders' values at the same block geometry.
///
/// # Errors
///
/// [`PackingError::UnsupportedKind`] when the grammar does not express `kind`, or any import and
/// decode refusal.
pub fn convert_row(kind: u32, raw: &[u8], n: usize) -> Result<Vec<i64>, PackingError> {
    let p = Packing::from_gguf_kind(kind).ok_or(PackingError::UnsupportedKind(kind))?;
    let canonical = import_row(kind, &p, raw, n)?;
    p.row_decode_fixed(&canonical, n)
}

/// Slice the whole ggml block `b` out of `raw`.
fn block_slice(raw: &[u8], b: usize, bb: usize) -> Result<&[u8], PackingError> {
    let at = b.checked_mul(bb).ok_or(PackingError::Overflow)?;
    let end = at.checked_add(bb).ok_or(PackingError::Overflow)?;
    raw.get(at..end).ok_or(PackingError::Truncated {
        need: end,
        have: raw.len(),
    })
}

/// Import one terminal `F16` element as its raw bit pattern.
#[allow(
    clippy::vec_init_then_push,
    reason = "explicit Vec::with_capacity plus push keeps the extracted body off the \
              alloc::vec::from_elem external that `vec![x]` would introduce"
)]
fn import_f16(raw: &[u8], b: usize) -> Result<CanonicalBlock, PackingError> {
    let blk = block_slice(raw, b, 2)?;
    let mut codes: Vec<i64> = Vec::with_capacity(1);
    codes.push(u16::from_le_bytes([blk[0], blk[1]]) as i64);
    let mut coeffs: Vec<(i32, i32)> = Vec::with_capacity(1);
    coeffs.push((1, 0));
    Ok(CanonicalBlock {
        scale: 0,
        min: 0,
        coeffs,
        codes,
    })
}

/// Import one terminal `F32` element as its raw bit pattern.
#[allow(
    clippy::vec_init_then_push,
    reason = "explicit Vec::with_capacity plus push keeps the extracted body off the \
              alloc::vec::from_elem external that `vec![x]` would introduce"
)]
fn import_f32(raw: &[u8], b: usize) -> Result<CanonicalBlock, PackingError> {
    let blk = block_slice(raw, b, 4)?;
    let mut codes: Vec<i64> = Vec::with_capacity(1);
    codes.push(u32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]) as i64);
    let mut coeffs: Vec<(i32, i32)> = Vec::with_capacity(1);
    coeffs.push((1, 0));
    Ok(CanonicalBlock {
        scale: 0,
        min: 0,
        coeffs,
        codes,
    })
}

#[allow(
    clippy::vec_init_then_push,
    reason = "explicit Vec::with_capacity plus push keeps the extracted body off the \
              alloc::vec::from_elem external that `vec![x]` would introduce"
)]
fn import_q8_0(raw: &[u8], b: usize) -> Result<CanonicalBlock, PackingError> {
    let blk = block_slice(raw, b, quant::Q8_0_BLOCK_BYTES)?;
    let mut codes: Vec<i64> = Vec::with_capacity(quant::Q8_0_QK);
    let mut i = 0usize;
    while i < quant::Q8_0_QK {
        codes.push((blk[2 + i] as i8) as i64);
        i += 1;
    }
    let mut coeffs: Vec<(i32, i32)> = Vec::with_capacity(1);
    coeffs.push((1, 0));
    Ok(CanonicalBlock {
        scale: u16::from_le_bytes([blk[0], blk[1]]),
        min: 0,
        coeffs,
        codes,
    })
}

#[allow(
    clippy::vec_init_then_push,
    reason = "explicit Vec::with_capacity plus push keeps the extracted body off the \
              alloc::vec::from_elem external that `vec![x]` would introduce"
)]
fn import_q5_0(raw: &[u8], b: usize) -> Result<CanonicalBlock, PackingError> {
    let blk = block_slice(raw, b, quant::Q5_0_BLOCK_BYTES)?;
    let d = quant::q5_0_d(blk)?;
    let signed = quant::q5_0_codes(blk)?;
    let mut codes: Vec<i64> = Vec::with_capacity(quant::Q5_0_QK);
    let mut i = 0usize;
    while i < quant::Q5_0_QK {
        // The `quant` accessor already applies the -16 offset; the canonical field is unsigned.
        codes.push(signed[i] as i64 + 16);
        i += 1;
    }
    let mut coeffs: Vec<(i32, i32)> = Vec::with_capacity(1);
    coeffs.push((1, 0));
    Ok(CanonicalBlock {
        scale: d,
        min: 0,
        coeffs,
        codes,
    })
}

fn import_q2_k(raw: &[u8], b: usize) -> Result<CanonicalBlock, PackingError> {
    let blk = block_slice(raw, b, quant::Q2_K_BLOCK_BYTES)?;
    let scales = quant::q2_k_scales(blk)?;
    let packed = quant::q2_k_codes(blk)?;
    let mut codes: Vec<i64> = Vec::with_capacity(quant::QK_K);
    let mut i = 0usize;
    while i < quant::QK_K {
        codes.push(packed[i] as i64);
        i += 1;
    }
    let mut coeffs: Vec<(i32, i32)> = Vec::with_capacity(16);
    let mut g = 0usize;
    while g < 16 {
        let (a, mb) = quant::q2_k_scale_min(scales[g]);
        coeffs.push((a as i32, mb as i32));
        g += 1;
    }
    Ok(CanonicalBlock {
        scale: quant::q2_k_d(blk)?,
        min: quant::q2_k_dmin(blk)?,
        coeffs,
        codes,
    })
}

fn import_q3_k(raw: &[u8], b: usize) -> Result<CanonicalBlock, PackingError> {
    let blk = block_slice(raw, b, quant::Q3_K_BLOCK_BYTES)?;
    let scales = quant::q3_k_scales(blk)?;
    let signed = quant::q3_k_codes(blk)?;
    let mut codes: Vec<i64> = Vec::with_capacity(quant::QK_K);
    let mut i = 0usize;
    while i < quant::QK_K {
        codes.push(signed[i] as i64 + 4);
        i += 1;
    }
    let mut coeffs: Vec<(i32, i32)> = Vec::with_capacity(16);
    let mut g = 0usize;
    while g < 16 {
        coeffs.push((scales[g] as i32 - 32, 0));
        g += 1;
    }
    Ok(CanonicalBlock {
        scale: quant::q3_k_d(blk)?,
        min: 0,
        coeffs,
        codes,
    })
}

fn import_q4_k(raw: &[u8], b: usize) -> Result<CanonicalBlock, PackingError> {
    let blk = block_slice(raw, b, quant::Q4_K_BLOCK_BYTES)?;
    let pairs = quant::q4_k_scale_mins(blk)?;
    let packed = quant::q4_k_codes(blk)?;
    let mut codes: Vec<i64> = Vec::with_capacity(quant::QK_K);
    let mut i = 0usize;
    while i < quant::QK_K {
        codes.push(packed[i] as i64);
        i += 1;
    }
    let mut coeffs: Vec<(i32, i32)> = Vec::with_capacity(8);
    let mut g = 0usize;
    while g < 8 {
        let (a, mb) = pairs[g];
        coeffs.push((a as i32, mb as i32));
        g += 1;
    }
    Ok(CanonicalBlock {
        scale: u16::from_le_bytes([blk[0], blk[1]]),
        min: u16::from_le_bytes([blk[2], blk[3]]),
        coeffs,
        codes,
    })
}

fn import_q6_k(raw: &[u8], b: usize) -> Result<CanonicalBlock, PackingError> {
    let blk = block_slice(raw, b, quant::Q6_K_BLOCK_BYTES)?;
    let scales = quant::q6_k_scales(blk)?;
    let signed = quant::q6_k_codes(blk)?;
    let mut codes: Vec<i64> = Vec::with_capacity(quant::QK_K);
    let mut i = 0usize;
    while i < quant::QK_K {
        codes.push(signed[i] as i64 + 32);
        i += 1;
    }
    let mut coeffs: Vec<(i32, i32)> = Vec::with_capacity(16);
    let mut g = 0usize;
    while g < 16 {
        coeffs.push((scales[g] as i32, 0));
        g += 1;
    }
    Ok(CanonicalBlock {
        scale: quant::q6_k_d(blk)?,
        min: 0,
        coeffs,
        codes,
    })
}

/// The fold pack's file magic: `APFOLDP1` (ASCII, no terminator).
pub const FOLD_MAGIC: [u8; 8] = *b"APFOLDP1";

/// A refusal from the fold pack's byte layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldError {
    /// The file does not start with [`FOLD_MAGIC`].
    Magic,
    /// A field runs past the end of the buffer, or a length does not fit the file.
    Truncated,
    /// A count, width or index disagrees with its declared shape.
    Shape(&'static str),
    /// A code value is wider than its field, or a row grid has no dictionary index.
    Range(&'static str),
}

/// One stored fold plane, every field already in the form the code plane executes with: an
/// `E8 ∪ O8` codebook index per 8 weights, a per-row `(lo, sp)` grid, the 4-bit row-scale levels,
/// and the `(f16 header, N-bit codes)` column and Hadamard scales. All scale values are Q32.32
/// words; the codes are the quantizers' own indices.
///
/// The layout is one file: the magic, a plane count, then per plane the shape, any output-row
/// restriction, the code indices, the row-grid dictionary, the 4-bit row levels, and the two
/// 6- and 5-bit column-scale fields with their `f16` headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldPlane {
    /// Output rows of the receiving weight this plane writes.
    pub o: usize,
    /// Input columns; a positive multiple of the row-scale group width (128).
    pub n: usize,
    /// The global output rows written, when the plane is a row restriction.
    pub rows: Option<Vec<u32>>,
    /// Codebook row per 8-column block, row-major `[o, n/8]`.
    pub idx: Vec<u8>,
    /// Per-row lower endpoint of the `r` grid, length `o`.
    pub lo: Vec<i64>,
    /// Per-row width of the `r` grid, length `o`.
    pub sp: Vec<i64>,
    /// Per `(row, group)` 4-bit level in `0..=15`, row-major `[o, n/128]`.
    pub r_k: Vec<u8>,
    /// The `c` field's `(f16 min, f16 max)` bit patterns.
    pub c_head: [u16; 2],
    /// The `c` field's 6-bit codes, length `n`.
    pub c_q: Vec<u8>,
    /// The `d` field's `(f16 min, f16 max)` bit patterns.
    pub d_head: [u16; 2],
    /// The `d` field's 5-bit codes, length `n`.
    pub d_q: Vec<u8>,
}

/// A least-significant-bit-first bit writer over a byte buffer.
struct BitOut {
    buf: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitOut {
    /// An empty writer.
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            n: 0,
        }
    }

    /// Append the low `bits` bits of `v` (at most 8). `O(1)`.
    fn push(&mut self, v: u32, bits: u32) {
        let mut left = bits;
        let mut val = v;
        while left > 0 {
            let take = (8 - self.n).min(left);
            let mask = (1_u32 << take) - 1;
            self.acc |= (val & mask) << self.n;
            self.n += take;
            val >>= take;
            left -= take;
            if self.n == 8 {
                self.buf.push(self.acc as u8);
                self.acc = 0;
                self.n = 0;
            }
        }
    }

    /// The bytes written, with a final partial byte zero-padded. `O(1)`.
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.buf.push(self.acc as u8);
        }
        self.buf
    }
}

/// A least-significant-bit-first bit reader over a byte slice.
struct BitIn<'a> {
    b: &'a [u8],
    at: usize,
    n: u32,
    acc: u32,
}

impl<'a> BitIn<'a> {
    /// A reader at `at`. `O(1)`.
    fn new(b: &'a [u8], at: usize) -> Self {
        Self {
            b,
            at,
            n: 0,
            acc: 0,
        }
    }

    /// The next `bits` bits (at most 8) as an integer, or `None` past the end. `O(1)`.
    fn take(&mut self, bits: u32) -> Option<u32> {
        let mut left = bits;
        let mut out = 0_u32;
        let mut shift = 0;
        while left > 0 {
            if self.n == 0 {
                self.acc = u32::from(*self.b.get(self.at)?);
                self.at += 1;
                self.n = 8;
            }
            let get = self.n.min(left);
            let mask = (1_u32 << get) - 1;
            out |= (self.acc & mask) << shift;
            self.acc >>= get;
            self.n -= get;
            shift += get;
            left -= get;
        }
        Some(out)
    }
}

/// Append `v` as four little-endian bytes. `O(1)`.
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Append `v` as two little-endian bytes. `O(1)`.
fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Append `v` as eight little-endian bytes. `O(1)`.
fn put_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// A cursor over a fold-pack buffer.
struct FoldIn<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> FoldIn<'a> {
    /// The next `n` bytes, or [`FoldError::Truncated`]. `O(1)`.
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], FoldError> {
        let end = self.at.checked_add(n).ok_or(FoldError::Truncated)?;
        let s = self.b.get(self.at..end).ok_or(FoldError::Truncated)?;
        self.at = end;
        Ok(s)
    }

    /// The next little-endian `u32`. `O(1)`.
    fn u32(&mut self) -> Result<u32, FoldError> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// The next little-endian `u16`. `O(1)`.
    fn u16(&mut self) -> Result<u16, FoldError> {
        let b = self.bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    /// The next little-endian `i64`. `O(1)`.
    fn i64(&mut self) -> Result<i64, FoldError> {
        let b = self.bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// A checked `usize` from a `u32` count. `O(1)`.
    fn len(&mut self, what: &'static str) -> Result<usize, FoldError> {
        let v = self.u32()?;
        usize::try_from(v).map_err(|_| FoldError::Shape(what))
    }
}

/// The number of bytes a `width`-bit code field of `n` values occupies.
fn bits_bytes(n: usize, width: u64) -> Option<usize> {
    let bits = u64::try_from(n).ok()?.checked_mul(width)?;
    let bytes = bits.checked_add(7)?.checked_div(8)?;
    usize::try_from(bytes).ok()
}

/// `o * (n / 8)`, the code-index count, checked.
fn idx_len(o: usize, n: usize) -> Option<usize> {
    if n == 0 || !n.is_multiple_of(8) {
        return None;
    }
    o.checked_mul(n / 8)
}

/// The shape checks every [`FoldPlane`] must pass before it is written.
fn fold_shape(p: &FoldPlane) -> Result<(), FoldError> {
    if p.o == 0 || p.n == 0 || !p.n.is_multiple_of(128) {
        return Err(FoldError::Shape("plane o, n"));
    }
    let nidx = idx_len(p.o, p.n).ok_or(FoldError::Shape("code indices"))?;
    let g = p.n / 128;
    if p.idx.len() != nidx
        || p.lo.len() != p.o
        || p.sp.len() != p.o
        || p.r_k.len() != p.o * g
        || p.c_q.len() != p.n
        || p.d_q.len() != p.n
        || p.rows.as_ref().is_some_and(|r| r.len() != p.o)
    {
        return Err(FoldError::Shape("plane field lengths"));
    }
    if p.sp.iter().any(|&s| s <= 0) {
        return Err(FoldError::Range("row grid span"));
    }
    if p.r_k.iter().any(|&k| k > 15)
        || p.c_q.iter().any(|&q| q > 63)
        || p.d_q.iter().any(|&q| q > 31)
    {
        return Err(FoldError::Range("code width"));
    }
    Ok(())
}

/// Append one plane's record. `O(o·n/8 + o·n/128 + n)`.
fn fold_write_plane(out: &mut Vec<u8>, p: &FoldPlane) -> Result<(), FoldError> {
    fold_shape(p)?;
    put_u32(out, u32::try_from(p.o).map_err(|_| FoldError::Shape("o"))?);
    put_u32(out, u32::try_from(p.n).map_err(|_| FoldError::Shape("n"))?);
    out.push(u8::from(p.rows.is_some()));
    if let Some(rows) = &p.rows {
        put_u32(
            out,
            u32::try_from(rows.len()).map_err(|_| FoldError::Shape("rows"))?,
        );
        for &r in rows {
            put_u32(out, r);
        }
    }
    out.extend_from_slice(&p.idx);

    // The row-grid dictionary: the distinct `(lo, sp)` pairs in order of first use, then one
    // 1-byte index per row (2 bytes when a plane takes more than 255 distinct grids).
    let mut dict: Vec<(i64, i64)> = Vec::new();
    let mut at: Vec<u32> = Vec::with_capacity(p.o);
    for i in 0..p.o {
        let key = (p.lo[i], p.sp[i]);
        if let Some(j) = dict.iter().position(|&d| d == key) {
            at.push(u32::try_from(j).map_err(|_| FoldError::Range("dictionary"))?);
            continue;
        }
        if dict.len() >= 65_535 {
            return Err(FoldError::Range("row grid dictionary"));
        }
        dict.push(key);
        at.push(u32::try_from(dict.len() - 1).map_err(|_| FoldError::Range("dictionary"))?);
    }
    let width = if dict.len() <= 255 { 1_u8 } else { 2_u8 };
    out.push(width);
    put_u32(
        out,
        u32::try_from(dict.len()).map_err(|_| FoldError::Shape("dict"))?,
    );
    for &(lo, sp) in &dict {
        put_i64(out, lo);
        put_i64(out, sp);
    }
    if width == 1 {
        for &j in &at {
            out.push(j as u8);
        }
    } else {
        for &j in &at {
            put_u16(
                out,
                u16::try_from(j).map_err(|_| FoldError::Range("dictionary"))?,
            );
        }
    }

    let mut bits = BitOut::new();
    for &k in &p.r_k {
        bits.push(u32::from(k), 4);
    }
    out.extend_from_slice(&bits.finish());

    put_u16(out, p.c_head[0]);
    put_u16(out, p.c_head[1]);
    let mut bits = BitOut::new();
    for &q in &p.c_q {
        bits.push(u32::from(q), 6);
    }
    out.extend_from_slice(&bits.finish());

    put_u16(out, p.d_head[0]);
    put_u16(out, p.d_head[1]);
    let mut bits = BitOut::new();
    for &q in &p.d_q {
        bits.push(u32::from(q), 5);
    }
    out.extend_from_slice(&bits.finish());
    Ok(())
}

/// `n` codes of `width` bits, least-significant bit first, or [`FoldError::Truncated`] past the
/// buffer. `O(n)`.
fn fold_read_codes(
    r: &mut FoldIn<'_>,
    n: usize,
    width: u32,
    what: &'static str,
) -> Result<Vec<u8>, FoldError> {
    let bytes = bits_bytes(n, u64::from(width)).ok_or(FoldError::Shape(what))?;
    let raw = r.bytes(bytes)?;
    let mut bits = BitIn::new(raw, 0);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let v = bits.take(width).ok_or(FoldError::Truncated)?;
        out.push(u8::try_from(v).map_err(|_| FoldError::Truncated)?);
    }
    Ok(out)
}

/// Read one plane's record. `O(o·n/8 + o·n/128 + n)`.
fn fold_read_plane(r: &mut FoldIn<'_>) -> Result<FoldPlane, FoldError> {
    let o = r.len("o")?;
    let n = r.len("n")?;
    if o == 0 || n == 0 || !n.is_multiple_of(128) {
        return Err(FoldError::Shape("plane o, n"));
    }
    let flags = r.bytes(1)?[0];
    let rows = if flags & 1 != 0 {
        let len = r.len("rows")?;
        if len != o {
            return Err(FoldError::Shape("rows length"));
        }
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(r.u32()?);
        }
        Some(v)
    } else {
        None
    };
    let nidx = idx_len(o, n).ok_or(FoldError::Shape("code indices"))?;
    let idx = r.bytes(nidx)?.to_vec();

    let width = r.bytes(1)?[0];
    let dict_len = r.len("dict")?;
    let mut dict: Vec<(i64, i64)> = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        let lo = r.i64()?;
        let sp = r.i64()?;
        if sp <= 0 {
            return Err(FoldError::Range("row grid span"));
        }
        dict.push((lo, sp));
    }
    let mut lo = Vec::with_capacity(o);
    let mut sp = Vec::with_capacity(o);
    match width {
        1 => {
            let at = r.bytes(o)?;
            for &j in at {
                let (l, s) = *dict
                    .get(usize::from(j))
                    .ok_or(FoldError::Range("row grid index"))?;
                lo.push(l);
                sp.push(s);
            }
        }
        2 => {
            let at = r.bytes(o.checked_mul(2).ok_or(FoldError::Shape("grid index"))?)?;
            for i in 0..o {
                let j = usize::from(u16::from_le_bytes([at[2 * i], at[2 * i + 1]]));
                let (l, s) = *dict.get(j).ok_or(FoldError::Range("row grid index"))?;
                lo.push(l);
                sp.push(s);
            }
        }
        _ => return Err(FoldError::Shape("row grid width")),
    }

    let g = n / 128;
    let nr = o.checked_mul(g).ok_or(FoldError::Shape("row levels"))?;
    let r_k = fold_read_codes(r, nr, 4, "row levels")?;

    let c_head = [r.u16()?, r.u16()?];
    let c_q = fold_read_codes(r, n, 6, "c codes")?;

    let d_head = [r.u16()?, r.u16()?];
    let d_q = fold_read_codes(r, n, 5, "d codes")?;

    let p = FoldPlane {
        o,
        n,
        rows,
        idx,
        lo,
        sp,
        r_k,
        c_head,
        c_q,
        d_head,
        d_q,
    };
    fold_shape(&p)?;
    Ok(p)
}

/// The fold pack's bytes: the magic, the plane count and one record per plane.
///
/// # Errors
/// [`FoldError::Shape`]/[`FoldError::Range`] on a plane that does not fit the layout, or a
/// `u32`/dictionary bound that a plane exceeds.
///
/// # Complexity
/// Time and space `O(Σ (o·n/8 + o·n/128 + n))`, one pass per plane.
pub fn fold_planes_write(planes: &[FoldPlane]) -> Result<Vec<u8>, FoldError> {
    let mut out = Vec::new();
    out.extend_from_slice(&FOLD_MAGIC);
    put_u32(
        &mut out,
        u32::try_from(planes.len()).map_err(|_| FoldError::Shape("plane count"))?,
    );
    for p in planes {
        fold_write_plane(&mut out, p)?;
    }
    Ok(out)
}

/// The planes of a fold-pack file.
///
/// # Errors
/// [`FoldError::Magic`] on a foreign file; [`FoldError::Truncated`] on a short buffer;
/// [`FoldError::Shape`]/[`FoldError::Range`] on a record that disagrees with its own shape.
///
/// # Complexity
/// Time and space `O(Σ (o·n/8 + o·n/128 + n))`, one pass per plane.
pub fn fold_planes_read(bytes: &[u8]) -> Result<Vec<FoldPlane>, FoldError> {
    let mut r = FoldIn { b: bytes, at: 0 };
    let magic = r.bytes(8)?;
    if magic != FOLD_MAGIC {
        return Err(FoldError::Magic);
    }
    let count = r.len("plane count")?;
    let mut out = Vec::with_capacity(count.min(1 << 16));
    for _ in 0..count {
        out.push(fold_read_plane(&mut r)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact Q32.32 value of an `f32` the tests compare against the reference decoders.
    fn q32(v: f32) -> i64 {
        (f64::from(v) * 4_294_967_296.0).round() as i64
    }

    fn canonical(width: CodeWidth, high: u8, block: u32, group: u32) -> Packing {
        Packing {
            width,
            high_planes: high,
            block,
            group,
            scale_kind: ScaleKind::F16,
            has_min: false,
            signed_code: false,
            bias: 0,
            signed_coeff: false,
            plane_order: PlaneOrder::LsbFirst,
            nibble_order: NibbleOrder::LowHigh,
        }
    }

    #[test]
    fn grammar_refuses_bad_width_block_and_geometry() {
        assert_eq!(
            canonical(CodeWidth::Fixed(5), 0, 32, 32).total_code_bits(),
            Err(PackingError::BadWidth(5))
        );
        assert_eq!(
            canonical(CodeWidth::Fixed(7), 0, 32, 32).validate(),
            Err(PackingError::BadWidth(7))
        );
        assert_eq!(
            canonical(CodeWidth::Fixed(4), 0, 48, 16).validate(),
            Err(PackingError::BadBlock(48))
        );
        assert_eq!(
            canonical(CodeWidth::Fixed(4), 0, 32, 5).validate(),
            Err(PackingError::BadGeometry {
                block: 32,
                group: 5
            })
        );
        assert_eq!(
            canonical(CodeWidth::Trit(0), 0, 32, 16).total_code_bits(),
            Err(PackingError::BadWidth(0))
        );
    }

    #[test]
    fn f16_and_pow2_scale_decoding_is_exact() {
        assert_eq!(f16_bits_to_fixed(0x3C00).expect("1.0"), ONE);
        assert_eq!(f16_bits_to_fixed(0x3800).expect("0.5"), ONE >> 1);
        assert_eq!(f16_bits_to_fixed(0xC000).expect("-2.0"), -(ONE << 1));
        assert_eq!(
            f16_bits_to_fixed(0x7C00),
            Err(PackingError::NonFiniteScale(0x7C00))
        );
        assert_eq!(pow2_exp_to_fixed(127).expect("2^0"), ONE);
        assert_eq!(pow2_exp_to_fixed(128).expect("2^1"), 2 * ONE);
        assert_eq!(pow2_exp_to_fixed(126).expect("2^-1"), ONE >> 1);
        assert_eq!(pow2_exp_to_fixed(95).expect("2^-32"), 1);
        assert_eq!(pow2_exp_to_fixed(60).expect("below unit"), 0);
    }

    #[test]
    fn canonical_round_trip_per_width_and_order() {
        // W = 4, block 64, group 16: four groups, four-bit codes, no min.
        let mut p = canonical(CodeWidth::Fixed(4), 0, 64, 16);
        assert_eq!(p.groups().expect("groups"), 4);
        let mut codes: Vec<i64> = Vec::new();
        let mut i = 0i64;
        while i < 64 {
            codes.push(i % 16);
            i += 1;
        }
        let block = CanonicalBlock {
            scale: 0x3C00, // 1.0
            min: 0,
            coeffs: vec![(1, 0), (2, 0), (3, 0), (4, 0)],
            codes,
        };
        let raw = p.encode_block(&block).expect("encode");
        let back = p.decode_block_fields(&raw).expect("decode fields");
        assert_eq!(back, block);
        let vals = p.decode_block(&raw).expect("decode");
        assert_eq!(vals.len(), 64);
        assert_eq!(vals[0], 0); // 1 * 1 * 0
        assert_eq!(vals[1], ONE); // 1 * 1 * 1
        assert_eq!(vals[16], 0); // group 1: 1 * 2 * 0
        assert_eq!(vals[17], 2 * ONE); // group 1: 1 * 2 * 1
        assert_eq!(vals[63], 60 * ONE); // group 3: 1 * 4 * 15
                                        // Re-encode the decoded fields: the canonical bytes round-trip.
        assert_eq!(p.encode_block(&back).expect("re-encode"), raw);

        // Nibble and plane order variants produce a different byte layout, same values.
        p.plane_order = PlaneOrder::MsbFirst;
        p.nibble_order = NibbleOrder::HighLow;
        let raw2 = p.encode_block(&block).expect("encode msb/highlow");
        assert_ne!(raw2, raw);
        assert_eq!(p.decode_block(&raw2).expect("decode msb/highlow"), vals);

        // trit face: two planes, L = 2, block 32, group 32.
        let t = canonical(CodeWidth::Trit(2), 0, 32, 32);
        let mut tcodes: Vec<i64> = Vec::new();
        let mut j = 0i64;
        while j < 32 {
            tcodes.push(j % 9); // field vector: trits (f0 - 1) + 3*(f1 - 1)
            j += 1;
        }
        let tb = CanonicalBlock {
            scale: 0x3C00,
            min: 0,
            coeffs: vec![(1, 0)],
            codes: tcodes,
        };
        let traw = t.encode_block(&tb).expect("trit encode");
        assert_eq!(t.decode_block_fields(&traw).expect("trit fields"), tb);
        let tv = t.decode_block(&traw).expect("trit decode");
        // fields are 2-bit, trit = field - 1, value = trit0 + 3*trit1.
        assert_eq!(tv[0], -4 * ONE); // fields (0,0) -> trits (-1,-1) -> -1 - 3
        assert_eq!(tv[1], -3 * ONE); // fields (1,0) -> trits (0,-1) -> -3
        assert_eq!(tv[2], -2 * ONE); // fields (2,0) -> trits (1,-1) -> 1 - 3
    }

    #[test]
    fn short_canonical_payload_is_refused() {
        let p = Packing::from_gguf_kind(quant::GGML_TYPE_Q8_0).expect("q8_0 packing");
        let need = p.block_bytes().expect("bytes");
        let short = vec![0u8; need - 1];
        assert_eq!(
            p.decode_block(&short),
            Err(PackingError::Truncated {
                need,
                have: need - 1
            })
        );
        assert_eq!(
            p.row_decode_fixed(&short, quant::Q8_0_QK),
            Err(PackingError::Truncated {
                need,
                have: need - 1
            })
        );
    }

    #[test]
    fn from_gguf_kind_covers_the_eight_packed_kinds() {
        let mut k = 0usize;
        while k < PACKED_GGUF_KINDS.len() {
            let kind = PACKED_GGUF_KINDS[k];
            let p = Packing::from_gguf_kind(kind).expect("packed kind");
            p.validate().expect("packing validates");
            let zeros = vec![0u8; p.block_bytes().expect("bb") * 8];
            let row = convert_row(kind, &zeros, 8 * p.block as usize).expect("zero row imports");
            assert_eq!(row.len(), 8 * p.block as usize);
            k += 1;
        }
        assert!(Packing::from_gguf_kind(3).is_none());
        assert_eq!(
            convert_row(3, &[], 1),
            Err(PackingError::UnsupportedKind(3))
        );
    }

    /// Little-endian bytes of a sequence of `f16` bit patterns.
    fn f16_le(values: &[u16]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut i = 0usize;
        while i < values.len() {
            out.push(values[i] as u8);
            out.push((values[i] >> 8) as u8);
            i += 1;
        }
        out
    }

    #[test]
    fn f16_terminal_decode_is_exact() {
        let p = Packing::from_gguf_kind(quant::GGML_TYPE_F16).expect("f16 packing");
        assert!(p.is_terminal());
        assert_eq!(p.block_size().expect("block"), 1);
        assert_eq!(p.block_bytes().expect("bytes"), 2);
        assert_eq!(p.decode_block(&f16_le(&[0x3C00])).expect("1.0"), vec![ONE]);
        assert_eq!(
            p.decode_block(&f16_le(&[0xC000])).expect("-2.0"),
            vec![-(ONE << 1)]
        );
        assert_eq!(
            p.decode_block(&f16_le(&[0x3E00])).expect("1.5"),
            vec![ONE + (ONE >> 1)]
        );
        assert_eq!(
            p.decode_block(&f16_le(&[0x0001])).expect("min subnormal"),
            vec![1 << 8]
        );
        assert_eq!(
            p.decode_block(&f16_le(&[0x7C00])),
            Err(PackingError::NonFiniteScale(0x7C00))
        );
        assert_eq!(
            p.decode_block(&f16_le(&[0x7E00])),
            Err(PackingError::NonFiniteScale(0x7E00))
        );
    }

    #[test]
    fn f32_terminal_decode_rounds_toward_positive_infinity() {
        let p = Packing::from_gguf_kind(quant::GGML_TYPE_F32).expect("f32 packing");
        assert!(p.is_terminal());
        assert_eq!(p.block_size().expect("block"), 1);
        assert_eq!(p.block_bytes().expect("bytes"), 4);
        assert_eq!(
            p.decode_block(&0.0f32.to_le_bytes()).expect("zero"),
            vec![0]
        );
        assert_eq!(
            p.decode_block(&(-0.0f32).to_le_bytes()).expect("-zero"),
            vec![0]
        );
        assert_eq!(
            p.decode_block(&1.0f32.to_le_bytes()).expect("1.0"),
            vec![ONE]
        );
        assert_eq!(
            p.decode_block(&(-1.0f32).to_le_bytes()).expect("-1.0"),
            vec![-ONE]
        );
        assert_eq!(
            p.decode_block(&1.5f32.to_le_bytes()).expect("1.5"),
            vec![ONE + (ONE >> 1)]
        );
        // A value that lands strictly between two units and rounds to the nearer one.
        let r25 = f32::from_bits((95u32 << 23) | (1u32 << 21)); // 1.25 · 2^-32
        assert_eq!(
            p.decode_block(&r25.to_le_bytes()).expect("1.25 unit"),
            vec![1]
        );
        let r75 = f32::from_bits((95u32 << 23) | (3u32 << 21)); // 1.75 · 2^-32
        assert_eq!(
            p.decode_block(&r75.to_le_bytes()).expect("1.75 unit"),
            vec![2]
        );
        // Exact ties: +1.5·2^-32 → 2, −1.5·2^-32 → −1 (ties toward +∞).
        let tie = f32::from_bits((95u32 << 23) | (1u32 << 22)); // 1.5 · 2^-32
        assert_eq!(p.decode_block(&tie.to_le_bytes()).expect("+tie"), vec![2]);
        let neg_tie = f32::from_bits((95u32 << 23) | (1u32 << 22) | (1u32 << 31));
        assert_eq!(
            p.decode_block(&neg_tie.to_le_bytes()).expect("-tie"),
            vec![-1]
        );
        // Overflow above the Q32.32 range refuses rather than clamping.
        assert_eq!(
            p.decode_block(&f32::MAX.to_le_bytes()),
            Err(PackingError::OutOfRange)
        );
        // A nonzero denormal underflows to zero: refused, not flushed.
        assert_eq!(
            p.decode_block(&f32::from_bits(1).to_le_bytes()),
            Err(PackingError::OutOfRange)
        );
        assert_eq!(
            p.decode_block(&f32::NAN.to_le_bytes()),
            Err(PackingError::NonFiniteFloat(f32::NAN.to_bits()))
        );
        assert_eq!(
            p.decode_block(&f32::INFINITY.to_le_bytes()),
            Err(PackingError::NonFiniteFloat(f32::INFINITY.to_bits()))
        );
    }

    #[test]
    fn f16_and_f32_conversion_matches_the_integer_reference() {
        let raw16 = f16_le(&[0x3C00, 0xC000, 0x3E00, 0x0001]);
        assert_eq!(
            convert_row(quant::GGML_TYPE_F16, &raw16, 4).expect("convert f16"),
            vec![ONE, -(ONE << 1), ONE + (ONE >> 1), 1 << 8]
        );

        let vals = [0.0f32, 1.0, -2.0, 1.5, 0.1, 0.0001];
        let mut raw32: Vec<u8> = Vec::new();
        let mut i = 0usize;
        while i < vals.len() {
            raw32.extend_from_slice(&vals[i].to_le_bytes());
            i += 1;
        }
        let got = convert_row(quant::GGML_TYPE_F32, &raw32, vals.len()).expect("convert f32");
        let mut j = 0usize;
        while j < vals.len() {
            assert_eq!(got[j], q32(vals[j]), "weight {j}");
            j += 1;
        }
    }

    #[test]
    fn q8_0_conversion_matches_the_quant_decoder() {
        let mut blk = vec![0u8; quant::Q8_0_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        blk[2] = 2;
        blk[3] = 0xFE; // -2
        let vals = convert_row(quant::GGML_TYPE_Q8_0, &blk, 2).expect("convert q8_0");
        assert_eq!(vals, vec![q32(1.0), q32(-1.0)]);
    }

    #[test]
    fn q5_0_conversion_matches_the_quant_decoder() {
        let mut blk = vec![0u8; quant::Q5_0_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        blk[2] = 0b0000_0101;
        blk[4] = 0b0000_0001;
        blk[6] = 0xFF;
        blk[7] = 0x00;
        blk[8] = 0x10;
        let n = quant::Q5_0_QK;
        let got = convert_row(quant::GGML_TYPE_Q5_0, &blk, n).expect("convert q5_0");
        // Q5_0 has no super-block decoder; the reference is the scale and code accessors.
        let d = quant::q5_0_scale(&blk, 0).expect("d");
        let codes = quant::q5_0_codes(&blk).expect("codes");
        let mut i = 0usize;
        while i < n {
            assert_eq!(got[i], q32(d * f32::from(codes[i])), "weight {i}");
            i += 1;
        }
    }

    #[test]
    fn q2_k_conversion_matches_the_quant_decoder() {
        let mut blk = vec![0u8; quant::Q2_K_BLOCK_BYTES];
        blk[0] = 0x51;
        blk[1] = 0x21;
        blk[16] = 0x03;
        blk[32] = 0x03;
        blk[80..82].copy_from_slice(&0x3C00u16.to_le_bytes());
        blk[82..84].copy_from_slice(&0x3800u16.to_le_bytes());
        let n = quant::QK_K;
        let got = convert_row(quant::GGML_TYPE_Q2_K, &blk, n).expect("convert q2_k");
        let mut want = [0f32; quant::QK_K];
        quant::dequant_q2_k_block(&blk, &mut want).expect("reference");
        let mut i = 0usize;
        while i < n {
            assert_eq!(got[i], q32(want[i]), "weight {i}");
            i += 1;
        }
    }

    #[test]
    fn q3_k_conversion_matches_the_quant_decoder() {
        let mut blk = vec![0u8; quant::Q3_K_BLOCK_BYTES];
        blk[96] = 0x08;
        blk[104] = 0x02;
        blk[108..110].copy_from_slice(&0x3C00u16.to_le_bytes());
        let n = quant::QK_K;
        let got = convert_row(quant::GGML_TYPE_Q3_K, &blk, n).expect("convert q3_k");
        let mut want = [0f32; quant::QK_K];
        quant::dequant_q3_k_block(&blk, &mut want).expect("reference");
        let mut i = 0usize;
        while i < n {
            assert_eq!(got[i], q32(want[i]), "weight {i}");
            i += 1;
        }
    }

    #[test]
    fn q4_k_conversion_matches_the_quant_decoder() {
        let mut blk = vec![0u8; quant::Q4_K_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x3C00u16.to_le_bytes());
        blk[4] = 5;
        blk[8] = 7;
        blk[16] = 0x03;
        let n = quant::QK_K;
        let got = convert_row(quant::GGML_TYPE_Q4_K, &blk, n).expect("convert q4_k");
        let mut want = [0f32; quant::QK_K];
        quant::dequant_q4_k_block(&blk, &mut want).expect("reference");
        let mut i = 0usize;
        while i < n {
            assert_eq!(got[i], q32(want[i]), "weight {i}");
            i += 1;
        }
    }

    #[test]
    fn q6_k_conversion_matches_the_quant_decoder() {
        let mut blk = vec![0u8; quant::Q6_K_BLOCK_BYTES];
        blk[192] = 2;
        blk[193] = 3;
        blk[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        let n = quant::QK_K;
        let got = convert_row(quant::GGML_TYPE_Q6_K, &blk, n).expect("convert q6_k");
        let mut want = [0f32; quant::QK_K];
        quant::dequant_q6_k_block(&blk, &mut want).expect("reference");
        let mut i = 0usize;
        while i < n {
            assert_eq!(got[i], q32(want[i]), "weight {i}");
            i += 1;
        }
    }

    /// The exact integer reference for a row, mirroring the executing path's epilogue.
    fn reference_row(kind: u32, raw: &[u8], n: usize) -> Vec<i64> {
        let p = Packing::from_gguf_kind(kind).expect("packed kind");
        let block = p.block_size().expect("block");
        let blocks = n / block + (n % block != 0) as usize;
        let mut out: Vec<i64> = Vec::with_capacity(n);
        let mut b = 0usize;
        while b < blocks {
            let vals = reference_block(kind, raw, b).expect("reference block");
            let take = block.min(n - b * block);
            let mut i = 0usize;
            while i < take {
                out.push(vals[i]);
                i += 1;
            }
            b += 1;
        }
        out
    }

    #[allow(clippy::too_many_lines, reason = "one arm per imported ggml kind")]
    fn reference_block(kind: u32, raw: &[u8], b: usize) -> Result<Vec<i64>, PackingError> {
        let p = Packing::from_gguf_kind(kind).expect("packed kind");
        let block = p.block_size()?;
        let mut out: Vec<i64> = Vec::with_capacity(block);
        match kind {
            quant::GGML_TYPE_F32 => {
                let blk = block_slice(raw, b, 4)?;
                out.push(f32_bits_to_fixed(u32::from_le_bytes([
                    blk[0], blk[1], blk[2], blk[3],
                ]))?);
            }
            quant::GGML_TYPE_F16 => {
                let blk = block_slice(raw, b, 2)?;
                out.push(f16_bits_to_fixed(u16::from_le_bytes([blk[0], blk[1]]))?);
            }
            quant::GGML_TYPE_Q8_0 => {
                let blk = block_slice(raw, b, quant::Q8_0_BLOCK_BYTES)?;
                let d = f16_bits_to_fixed(u16::from_le_bytes([blk[0], blk[1]]))?;
                let codes = quant::q8_0_codes(raw, b)?;
                let mut i = 0usize;
                while i < quant::Q8_0_QK {
                    out.push(mul(d, i64::from(codes[i]))?);
                    i += 1;
                }
            }
            quant::GGML_TYPE_Q5_0 => {
                let blk = block_slice(raw, b, quant::Q5_0_BLOCK_BYTES)?;
                let d = f16_bits_to_fixed(quant::q5_0_d(blk)?)?;
                let codes = quant::q5_0_codes(blk)?;
                let mut i = 0usize;
                while i < quant::Q5_0_QK {
                    out.push(mul(d, i64::from(codes[i]))?);
                    i += 1;
                }
            }
            quant::GGML_TYPE_Q2_K => {
                let blk = block_slice(raw, b, quant::Q2_K_BLOCK_BYTES)?;
                let d = f16_bits_to_fixed(quant::q2_k_d(blk)?)?;
                let dmin = f16_bits_to_fixed(quant::q2_k_dmin(blk)?)?;
                let sc = quant::q2_k_scales(blk)?;
                let codes = quant::q2_k_codes(blk)?;
                let mut g = 0usize;
                while g < 16 {
                    let (a, mb) = quant::q2_k_scale_min(sc[g]);
                    let slope = mul(d, i64::from(a))?;
                    let intercept = mul(dmin, i64::from(mb))?;
                    let mut l = 0usize;
                    while l < 16 {
                        out.push(sub(mul(slope, i64::from(codes[16 * g + l]))?, intercept)?);
                        l += 1;
                    }
                    g += 1;
                }
            }
            quant::GGML_TYPE_Q3_K => {
                let blk = block_slice(raw, b, quant::Q3_K_BLOCK_BYTES)?;
                let d = f16_bits_to_fixed(quant::q3_k_d(blk)?)?;
                let sc = quant::q3_k_scales(blk)?;
                let codes = quant::q3_k_codes(blk)?;
                let mut g = 0usize;
                while g < 16 {
                    let slope = mul(d, i64::from(sc[g]) - 32)?;
                    let mut l = 0usize;
                    while l < 16 {
                        out.push(mul(slope, i64::from(codes[16 * g + l]))?);
                        l += 1;
                    }
                    g += 1;
                }
            }
            quant::GGML_TYPE_Q4_K => {
                let blk = block_slice(raw, b, quant::Q4_K_BLOCK_BYTES)?;
                let d = f16_bits_to_fixed(u16::from_le_bytes([blk[0], blk[1]]))?;
                let dmin = f16_bits_to_fixed(u16::from_le_bytes([blk[2], blk[3]]))?;
                let pairs = quant::q4_k_scale_mins(blk)?;
                let codes = quant::q4_k_codes(blk)?;
                let mut g = 0usize;
                while g < 4 {
                    let (al, ml) = pairs[2 * g];
                    let (ah, mh) = pairs[2 * g + 1];
                    let sl = mul(d, i64::from(al))?;
                    let il = mul(dmin, i64::from(ml))?;
                    let sh = mul(d, i64::from(ah))?;
                    let ih = mul(dmin, i64::from(mh))?;
                    let mut l = 0usize;
                    while l < 32 {
                        out.push(sub(mul(sl, i64::from(codes[64 * g + l]))?, il)?);
                        out.push(sub(mul(sh, i64::from(codes[64 * g + 32 + l]))?, ih)?);
                        l += 1;
                    }
                    g += 1;
                }
            }
            quant::GGML_TYPE_Q6_K => {
                let blk = block_slice(raw, b, quant::Q6_K_BLOCK_BYTES)?;
                let d = f16_bits_to_fixed(quant::q6_k_d(blk)?)?;
                let sc = quant::q6_k_scales(blk)?;
                let codes = quant::q6_k_codes(blk)?;
                let mut g = 0usize;
                while g < 16 {
                    let slope = mul(d, i64::from(sc[g]))?;
                    let mut l = 0usize;
                    while l < 16 {
                        out.push(mul(slope, i64::from(codes[16 * g + l]))?);
                        l += 1;
                    }
                    g += 1;
                }
            }
            other => return Err(PackingError::UnsupportedKind(other)),
        }
        Ok(out)
    }

    #[test]
    fn conversion_matches_the_exact_integer_reference_on_the_real_model() {
        let Some(path) = crate::engine::testdata::model_path("models/gguf/qwen05b-Q4_K_M.gguf")
        else {
            eprintln!("skip conversion_matches_the_exact_integer_reference_on_the_real_model: model absent");
            return;
        };
        let g = crate::format::gguf::Gguf::open(&path).expect("open model");
        let mut seen = 0usize;
        let mut t = 0usize;
        while t < g.header.tensors.len() && seen < 4 {
            let desc = &g.header.tensors[t];
            if Packing::from_gguf_kind(desc.ggml_type).is_some() {
                let p = Packing::from_gguf_kind(desc.ggml_type).expect("packing");
                let block = p.block_size().expect("block");
                let bb = p.block_bytes().expect("bytes");
                let raw = g
                    .tensor_bytes_prefix(desc, (bb as u64) * 2)
                    .expect("tensor prefix");
                let n = 2 * block;
                let got = convert_row(desc.ggml_type, &raw, n).expect("convert");
                let want = reference_row(desc.ggml_type, &raw, n);
                assert_eq!(got, want, "tensor {}", String::from_utf8_lossy(&desc.name));
                seen += 1;
            }
            t += 1;
        }
        assert!(seen > 0, "no packed tensor found in {path}");
    }

    /// A synthetic fold plane: two output rows over two 128-column groups, two dictionary grids,
    /// `n = 256` with a 4-bit level per group.
    fn fold_plane() -> FoldPlane {
        FoldPlane {
            o: 2,
            n: 256,
            rows: None,
            idx: (0..64).map(|i| (i * 5 % 256) as u8).collect(),
            lo: vec![ONE, ONE + 5],
            sp: vec![ONE / 3, ONE / 4],
            r_k: vec![0, 15, 7, 8],
            c_head: [0x3c00, 0x4000],
            c_q: (0..256).map(|i| (i % 64) as u8).collect(),
            d_head: [0x3c00, 0x3c00],
            d_q: (0..256).map(|i| (i % 32) as u8).collect(),
        }
    }

    /// The fold pack's byte layout round-trips every field, the bad magic and a truncated buffer
    /// are refused, and a code past its field width is refused on write.
    #[test]
    fn fold_pack_round_trips_and_refuses() {
        let p = fold_plane();
        let bytes = fold_planes_write(&[p.clone(), p.clone()]).expect("write");
        let back = fold_planes_read(&bytes).expect("read");
        assert_eq!(back, vec![p.clone(), p]);
        assert_eq!(&bytes[..8], b"APFOLDP1");

        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert_eq!(fold_planes_read(&bad).unwrap_err(), FoldError::Magic);
        assert_eq!(
            fold_planes_read(&bytes[..bytes.len() - 1]).unwrap_err(),
            FoldError::Truncated
        );

        let mut wide = fold_plane();
        wide.r_k[0] = 16;
        assert_eq!(
            fold_planes_write(&[wide]).unwrap_err(),
            FoldError::Range("code width")
        );
        let mut short = fold_plane();
        short.c_q.pop();
        assert_eq!(
            fold_planes_write(&[short]).unwrap_err(),
            FoldError::Shape("plane field lengths")
        );

        let mut rows = fold_plane();
        rows.rows = Some(vec![0, 5]);
        let bytes = fold_planes_write(&[rows.clone()]).expect("write rows");
        assert_eq!(fold_planes_read(&bytes).expect("read rows"), vec![rows]);
    }
}
