//! The one shared Q32.32 product primitive on the `Q8_0` kernel: [`gemm_q32`], [`syrk_q32`],
//! [`gemv_q32`] and [`rowdot_q32`].
//!
//! **`Q8_0` raw layout.** A block is 2 little-endian bytes of `f16` scale followed by 32 signed code
//! bytes ([`Q8_0_BLOCK_BYTES`] = 34); a row of `n` inputs is `⌈n/32⌉` blocks
//! ([`Q8_0_QK`] = 32). Code `q` of a block decodes to the Q32.32 word `d·q/2^32` with
//! `d = f16_to_fixed(scale) = m·2^k`, `|m| ≤ 2047`, `k ∈ [8, 37]` ([`super::pow2::f16_split`]).
//!
//! **Exactness on the kernel.** [`gemm_q8_0`] computes
//! `round_shift_32(Σ_b d_b·S_b·D_b)` with `D_b` the integer code dot of block `b` and `S_b` the
//! activation block scale. Two facts make every term a shift and the call exact: `d = m·2^k` and
//! `S = 2^e` a power of two ([`crate::engine::chart::quant_act`] only ever returns a power of two),
//! so a block contributes `(m·D_b)·2^{k+e}`. Here every weight scale is the exact power of two
//! `2^{dw}` (`m = 1024`, `k = dw − 10`) and every activation scale the exact power of two `2^σ`, so
//! the block term is `1024·D·2^{dw+σ−10}` with `|1024·D| < 2^30` for the digits below; nothing is
//! rounded until the call's single `round_shift_32`.
//!
//! **The limb expansion.** Each 32-element block of each operand row is written
//! `x = 2^e·Σ_{s<limbs} d_s·2^{−7(s+1)} + 2^e·ρ` with balanced signed-byte digits
//! `|d_s| ≤ 64` and `|ρ| ≤ 2^{−7·limbs}/2`, `e` the block exponent shared by the block's rows
//! (`2^{e−1} > max|x|`, raised per block only when the pair placement needs it). A pair of
//! operands then expands into the kept limb pairs `(s, t)` with `s + t ≤ limbs − 1`
//! (`limbs(limbs+1)/2` of them: 6 of 9 at `limbs = 3`), each pair contributing the block term
//! `α_s·β_t·2^{ea+eb−7(s+t+2)−32}` in Q32.32 words. A pair's two scales are placed as
//! `d = 2^{dw}`, `S = 2^σ` with `dw = clamp(T, 8, 47)` and `σ = T − dw`, `T = ea + eb − 7(s+t+2)`
//! the pair's Q32.32 exponent: `dw ∈ [8, 47]` is exactly the range of a finite `f16` power of two
//! (`2^{−24} … 2^{15}` scaled by `2^32`), and `σ ∈ [0, 44]` keeps `S·32·128·127 ≤ i64::MAX`, the
//! fast-row bound [`crate::engine::chart::q8act_fast_bound_ok`] tests. `T < 8` cannot be placed
//! (the activation scale would be fractional); such a block pair's whole contribution is
//! `≤ 2^{T−15} < 2^{−8}` words and is dropped, and the bound below counts it. `T > 91` exceeds
//! every placement and is refused as [`LimbError::Range`].
//!
//! **Refusals.** [`LimbError::Shape`]: a zero or mismatched dimension, a slice whose length is not
//! `rows·n`, `limbs == 0`, or `n > 16384` (the kernel's 512-block `MAX_BLOCKS`). [`LimbError::Range`]:
//! a block exponent pair whose product the scale placement cannot carry, or a group partial sum
//! that leaves `i64`. [`LimbError::NoKernel`] is never returned: a host without AVX-VNNI makes
//! [`gemm_q8_0`] return `None` and the same planes go to [`matmat_q8act`], which is the `Q8_0`
//! contract itself; only a `MatvecError` refusal survives, and that is `Range`.
//!
//! **Trust boundary.** The AVX2/AVX-VNNI cores of `gemm_q8_0` and `matmat_q8act` are not
//! extracted; their equality with the scalar contract is `tested` in `gemm.rs` and `matmat.rs`.
//! The error bound below is `tested` against an `i128` oracle, not yet `proved`: its Lean theorem
//! is open (`docs/plan.md`), so this file carries a `tested` grade only.
#![allow(
    clippy::many_single_char_names,
    reason = "`a`, `b`, `m`, `p`, `n`, `s`, `t` are the matrix-product and limb indices the \
              API, the kernel's own signatures and the derivation all use; renaming them would \
              make the bound's arithmetic unreadable"
)]
#![allow(
    dead_code,
    reason = "the four entry points land in this wave; the KERN port lanes wire their call sites \
              in the next wave, and the tests call every one of them under cfg(test)"
)]

use super::gemm::gemm_q8_0;
use super::matmat::matmat_q8act;
use crate::engine::chart::QuantAct;
use crate::format::quant::{GGML_TYPE_Q8_0, Q8_0_BLOCK_BYTES, Q8_0_QK};

/// Lanes of one limb block, the kernel's activation block width.
const ACT_QK: usize = Q8_0_QK;

/// The widest contraction `gemm_q8_0` takes: its `MAX_BLOCKS` (512) blocks of 32.
const MAX_INNER: usize = 512 * Q8_0_QK;

/// The most limbs one block may be split into; the digit shift `e + 7(limbs − 1)` must stay
/// inside the `i128` digit accumulator.
const MAX_LIMBS: u32 = 8;

/// `2^−7`: one limb step.
const LIMB_BITS: i32 = 7;

/// The smallest pair exponent the scale placement carries: `dw ≥ 8` (the smallest finite `f16`
/// power of two is `2^−24`, i.e. `2^8` in Q32.32) with `σ ≥ 0`.
const MIN_PAIR_EXP: i32 = 8;

/// The largest pair exponent: `dw ≤ 47` (the largest `f16` power of two, `2^15`, is `2^47` in
/// Q32.32) with `σ ≤ 44` (`2^44·32·128·127 ≤ i64::MAX`).
const MAX_PAIR_EXP: i32 = 91;

/// The largest Q32.32 exponent of a finite `f16` power of two: `2^15` is `2^47` in Q32.32.
const MAX_WEIGHT_EXP: i32 = 47;

/// Why a limb product refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LimbError {
    /// A dimension, a slice length, or a contraction width the kernel call cannot carry.
    Shape,
    /// A word whose block exponents the `f16`/`i64` scale placement cannot carry, or an output
    /// that leaves `i64`.
    Range,
    /// No kernel: never returned. A host without AVX-VNNI falls back to `matmat_q8act` on the same
    /// planes, and the variant exists only so a caller can name the condition it cannot see.
    NoKernel,
}

/// `C[i·p + j] = Σ_l a[i·n + l]·b[j·n + l]` in Q32.32, both operands row-major, the contraction
/// contiguous, each 32-element block split into `limbs` signed-byte limbs and every limb product
/// carried on [`gemm_q8_0`].
///
/// # Error bound
///
/// Write `Ea = ⌊log₂ max|a|⌋ + 2`, `Eb = ⌊log₂ max|b|⌋ + 2` (so every block exponent `ea ≤ Ea`,
/// `eb ≤ Eb`), `L = limbs`, `P = L(L+1)/2` kept pairs, `B = ⌈n/32⌉` blocks. Then every entry obeys
/// `|C_limb − C_exact| ≤ G(n, L, max|a|, max|b|)` words, with
///
/// ```text
/// G = ⌈P/2⌉ + 1 + B·(1 + ⌈C_L·2^{Ea+Eb−32}⌉),
/// C_L = (33 + 16(L−1))·2^{−7L},
/// ```
///
/// which is [`bound_ulps`] rounded out to the integers. The three terms are the call's own
/// rounding (`⌈P/2⌉`: at most one `round_shift_32` per kept pair, each `≤ 1/2` ulp), the per-block
/// residual and dropped-pair mass `C_L·2^{Ea+Eb−32}` (both operands truncated to `L` digits, and
/// the pairs with `s+t ≥ L` dropped), and the per-block `+1` covering the `T < 8` blocks whose
/// whole contribution is `≤ 2^{−8}` words. Truncation dominates:
/// `|ρ_a|·|b| + |ρ_b|·|a| + |ρ_aρ_b| ≤ 2^{−7L}(1 + 2^{−7L}/4)` per element, and
/// `Σ_{s+t≥L}|α_s||β_t|2^{−7(s+t+2)} ≤ 64²·D_L ≤ 64²(L−1)·2^{−7L−13}` per element; both times 32
/// elements and `2^{ea+eb−32}` words give the `C_L` coefficient. At `L = 3` that is
/// `C_3 = 65·2^{−21}`, i.e. a relative per-block truncation of about `2^{−15}` of `2^{Ea+Eb}`.
///
/// # Errors
///
/// [`LimbError::Shape`] on a zero or mismatched dimension, `limbs == 0` or `n > 16384`;
/// [`LimbError::Range`] on a block pair the scale placement cannot carry, a group partial sum
/// that leaves `i64`, or an entry that leaves `i64`. A group partial is refused as `Range` even
/// when the exact total would cancel back into range: the refusal is conservative, never a wrong
/// word.
///
/// # Complexity
///
/// Temporal `O(n·(m + p)·limbs)` to build the limb planes plus `O(P·m·p·n)` byte multiply-adds —
/// the kernel's `vpdpbusd` counts 32 per instruction, so the executed instruction count is
/// `P·m·p·n/32` — plus the same `P·m·p·n` activation-code byte reads; each kernel row
/// task reads the `P·n` weight bytes of its own rows once per call. Spatial
/// `O((m + p)·n·limbs)` limb bytes, `O(P·m·n)` weight bytes for one pair group and `O(m·p)` `i128`
/// accumulators.
pub(crate) fn gemm_q32(
    a: &[i64],
    m: usize,
    b: &[i64],
    p: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    let need_a = m.checked_mul(n).ok_or(LimbError::Shape)?;
    let need_b = p.checked_mul(n).ok_or(LimbError::Shape)?;
    if a.len() != need_a || b.len() != need_b {
        return Err(LimbError::Shape);
    }
    let pa = Plane::build(a, m, n, limbs)?;
    let pb = Plane::build(b, p, n, limbs)?;
    pa.product_rows(&pb, 0, m, 0, p, n)
}

/// The upper triangle of `A·Aᵀ`, mirrored: `C[i·m + j] = Σ_l a[i·n + l]·a[j·n + l]`, `C` is `m×m`.
///
/// Only the columns `j` and the rows `i ≤ j` the upper triangle needs are computed (a column chunk
/// `[j0, j1)` runs the kernel over rows `0..j1`; the entries `i > j` are discarded), and the lower
/// triangle is the transpose of the upper, so the kernel work is `≈ m²n/2` rather than `m²n`. Each
/// column is computed by the same planes and the same pair grouping as [`gemm_q32`], so the two
/// agree bit for bit and the mirror is exact.
///
/// # Error bound
///
/// [`gemm_q32`]'s `G(n, limbs, max|a|, max|a|)`.
///
/// # Errors
///
/// As [`gemm_q32`].
///
/// # Complexity
///
/// Temporal `O(m·n·limbs + m²·n·limbs/2)` byte multiply-adds — half of [`gemm_q32`]'s full
/// product — and spatial `O(m·n·limbs)` for the one limb plane plus `O(m·COLS)` accumulators.
pub(crate) fn syrk_q32(a: &[i64], m: usize, n: usize, limbs: u32) -> Result<Vec<i64>, LimbError> {
    if a.len() != m.checked_mul(n).ok_or(LimbError::Shape)? {
        return Err(LimbError::Shape);
    }
    let plane = Plane::build(a, m, n, limbs)?;
    let mut c = vec![0i64; m.checked_mul(m).ok_or(LimbError::Shape)?];
    let mut j0 = 0usize;
    while j0 < m {
        let j1 = (j0 + SYRK_COLS).min(m);
        let width = j1 - j0;
        let block = plane.product_rows(&plane, 0, j1, j0, width, n)?;
        let mut i = 0usize;
        while i < j1 {
            let mut col = 0usize;
            while col < width {
                let j = j0 + col;
                if i <= j {
                    c[i * m + j] = block[i * width + col];
                }
                col += 1;
            }
            i += 1;
        }
        j0 = j1;
    }
    // The lower triangle is the upper one's transpose.
    let mut i = 0usize;
    while i < m {
        let mut j = i + 1;
        while j < m {
            c[j * m + i] = c[i * m + j];
            j += 1;
        }
        i += 1;
    }
    Ok(c)
}

/// One row against many: `y[j] = Σ_l x[l]·b[j·n + l]`.
///
/// Identical to [`gemm_q32`] with `m = 1` on the same planes, so the words are bit-equal.
///
/// # Error bound
///
/// [`gemm_q32`]'s `G(n, limbs, max|x|, max|b|)`.
///
/// # Errors
///
/// As [`gemm_q32`].
///
/// # Complexity
///
/// Temporal `O(n·(1 + p)·limbs + P·p·n)` byte multiply-adds; spatial `O((p + 1)·n·limbs)` limb
/// bytes and `O(p)` accumulators.
pub(crate) fn gemv_q32(
    x: &[i64],
    b: &[i64],
    p: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    if x.len() != n || b.len() != p.checked_mul(n).ok_or(LimbError::Shape)? {
        return Err(LimbError::Shape);
    }
    let px = Plane::build(x, 1, n, limbs)?;
    let pb = Plane::build(b, p, n, limbs)?;
    px.product_rows(&pb, 0, 1, 0, p, n)
}

/// Batched row-pair dots: `y[i] = Σ_l a[i·n + l]·b[i·n + l]`.
///
/// A pooled per-row call of the same limb planes: row `i` runs one [`gemm_q8_0`] call with one
/// weight row and one activation column, so the diagonal is the same plane and grouping as
/// [`gemm_q32`]'s and the words are bit-equal.
///
/// # Error bound
///
/// [`gemm_q32`]'s `G(n, limbs, max|a|, max|b|)`, entrywise.
///
/// # Errors
///
/// As [`gemm_q32`], at the first refusing row.
///
/// # Complexity
///
/// Temporal `O((m + m)·n·limbs + m·P·n)` byte multiply-adds — `m` kernel calls of one row against
/// one column, each `O(P·n)` — plus `m` pool dispatches; spatial `O((m + m)·n·limbs)` limb bytes
/// and one `i128` accumulator.
pub(crate) fn rowdot_q32(
    a: &[i64],
    b: &[i64],
    m: usize,
    n: usize,
    limbs: u32,
) -> Result<Vec<i64>, LimbError> {
    let need = m.checked_mul(n).ok_or(LimbError::Shape)?;
    if a.len() != need || b.len() != need {
        return Err(LimbError::Shape);
    }
    let pa = Plane::build(a, m, n, limbs)?;
    let pb = Plane::build(b, m, n, limbs)?;
    let mut y = Vec::with_capacity(m);
    let mut i = 0usize;
    while i < m {
        let row = pa.product_rows(&pb, i, 1, i, 1, n)?;
        y.push(row[0]);
        i += 1;
    }
    Ok(y)
}

/// Columns per [`syrk_q32`] kernel call: the row band the upper triangle needs at that column,
/// small enough that the discarded lower-triangle rows stay a sixth of the work.
const SYRK_COLS: usize = 16;

/// The per-block limb expansion of one operand matrix.
struct Plane {
    /// Rows of the operand.
    rows: usize,
    /// Inputs per row (`n`), unpadded.
    n: usize,
    /// Blocks per row, `⌈n/32⌉`.
    blocks: usize,
    /// `blocks·32`: the padded row length the kernel reads.
    padded: usize,
    /// Limbs per block.
    limbs: u32,
    /// Per block, the exponent shared by every row: `max_i(bitlen(max|x|) + 1)`, `≥ 1`.
    e: Vec<u32>,
    /// `[row][block][limb][lane]`, the balanced digits of `2^{−e}·x`.
    dig: Vec<i8>,
}

/// One kept limb pair: weight digit `s` against activation digit `t`.
type Pair = (u32, u32);

/// The kept pairs `(s, t)`, `s + t ≤ limbs − 1`, ordered by `s + t` then `s`.
///
/// The order matters twice: every prefix of it is closed under `(s, t) ↦ (t, s)` — which is what
/// makes [`syrk_q32`]'s mirror equal to [`gemm_q32`]'s transpose bit for bit — and the pair groups
/// it induces are the ones the two share.
///
/// # Complexity
///
/// `O(limbs²)` time and space; the double loop runs `limbs(limbs+1)/2` accepting steps.
fn kept_pairs(limbs: u32) -> Vec<Pair> {
    let mut out = Vec::with_capacity((limbs * (limbs + 1) / 2) as usize);
    let mut sum = 0u32;
    while sum < limbs {
        let mut s = 0u32;
        while s <= sum {
            let t = sum - s;
            if t < limbs {
                out.push((s, t));
            }
            s += 1;
        }
        sum += 1;
    }
    out
}

/// `⌊log₂ x⌋ + 1`, `0` for `x = 0`.
#[must_use]
fn bitlen(x: u64) -> u32 {
    u64::BITS - x.leading_zeros()
}

/// The exact `f16` word of the Q32.32 power of two `2^{dw}`, `dw ∈ [8, 47]`.
///
/// `2^{dw}` is `2^{dw−32}` as a real: a normal `f16` `(1024 + 0)·2^{e−25}` with
/// `e = dw − 17` when `dw ≥ 18`, and the subnormal `f = 2^{dw−8}` for `dw ∈ [8, 17]`. Reversing
/// through [`super::pow2::f16_split`] gives `(1024, dw − 10)` and `(2^{dw−8}, 8)`, both exactly
/// `2^{dw}`.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "dw is clamped to [8, 47]; the normals land in [1, 30] and the subnormal mantissa in \
              [1, 9], so both shifts are exact and the u16 is a 5- or 10-bit field"
)]
fn f16_pow2(dw: i32) -> u16 {
    let p = dw - 32;
    if p >= -14 {
        ((p + 15) as u16) << 10
    } else {
        1u16 << (p + 24)
    }
}

/// The largest `|word|` of one 32-element block.
///
/// # Complexity
///
/// `O(32)`.
fn block_absmax(words: &[i64]) -> u64 {
    let mut m = 0u64;
    for &w in words {
        m = m.max(w.unsigned_abs());
    }
    m
}

/// The block exponent of an absolute maximum: `bitlen(m) + 1` (`1` for an all-zero block), so that
/// `|x| < 2^{e−1}` and the first digit `round(x/2^{e−7})` obeys `|d₀| ≤ 64`.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
fn block_exp(absmax: u64) -> u32 {
    if absmax == 0 {
        1
    } else {
        bitlen(absmax) + 1
    }
}

/// The `limbs` balanced digits of `x` at block exponent `e`, written to `out`.
///
/// `x = 2^e·Σ_s d_s·2^{−7(s+1)} + ρ` with `|d_s| ≤ 64`, `|ρ| ≤ 2^{e−7·limbs}/2`. The residual is
/// carried in units of `2^{e−7·limbs}` so every step is an integer shift: digit `s` is the nearest
/// integer to `rem/2^{e+7(limbs−1−s)}` with ties away from zero, the same tie rule `quant_act`
/// uses.
///
/// # Complexity
///
/// `O(limbs)` shifts on `i128`, no allocation.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "|d_s| ≤ 64 by the block-exponent choice, so the i128-to-i8 cast is exact; the `as \
              u32` shift amounts are bounded by e + 7(limbs−1) ≤ 120"
)]
fn split_digits(x: i64, e: u32, limbs: u32, out: &mut [i8]) -> i128 {
    let mut rem: i128 = i128::from(x) << (LIMB_BITS as u32 * limbs);
    let mut s = 0u32;
    while s < limbs {
        let sh = e + (LIMB_BITS as u32) * (limbs - 1 - s);
        let d = (rem + (1i128 << (sh - 1))) >> sh;
        out[s as usize] = d as i8;
        rem -= d << sh;
        s += 1;
    }
    rem
}

impl Plane {
    /// The limb expansion of `data` (`rows × n`, row-major), or the shape refusal.
    ///
    /// # Complexity
    ///
    /// Temporal `O(rows·n·limbs)` digit steps in two passes (block maxima, then digits);
    /// spatial `O(rows·⌈n/32⌉·limbs·32)` bytes for the digits plus `O(⌈n/32⌉)` exponents.
    fn build(data: &[i64], rows: usize, n: usize, limbs: u32) -> Result<Self, LimbError> {
        if rows == 0 || n == 0 || limbs == 0 || limbs > MAX_LIMBS || n > MAX_INNER {
            return Err(LimbError::Shape);
        }
        if data.len() != rows.checked_mul(n).ok_or(LimbError::Shape)? {
            return Err(LimbError::Shape);
        }
        let blocks = n.div_ceil(ACT_QK);
        let padded = blocks * ACT_QK;
        let mut e = vec![1u32; blocks];
        let mut r = 0usize;
        while r < rows {
            let base = r * n;
            let mut b = 0usize;
            while b < blocks {
                let lo = base + b * ACT_QK;
                let hi = (lo + ACT_QK).min(base + n);
                let cand = block_exp(block_absmax(&data[lo..hi]));
                if cand > e[b] {
                    e[b] = cand;
                }
                b += 1;
            }
            r += 1;
        }
        let lane_words = limbs as usize * ACT_QK;
        let mut dig = vec![0i8; rows * blocks * lane_words];
        let mut tmp = vec![0i8; limbs as usize];
        let mut r = 0usize;
        while r < rows {
            let base = r * n;
            let mut b = 0usize;
            while b < blocks {
                let lo = base + b * ACT_QK;
                let hi = (lo + ACT_QK).min(base + n);
                let at = (r * blocks + b) * lane_words;
                let mut lane = 0usize;
                while lane < ACT_QK {
                    let x = if lo + lane < hi { data[lo + lane] } else { 0 };
                    let _ = split_digits(x, e[b], limbs, &mut tmp);
                    let mut s = 0usize;
                    while s < limbs as usize {
                        dig[at + s * ACT_QK + lane] = tmp[s];
                        s += 1;
                    }
                    lane += 1;
                }
                b += 1;
            }
            r += 1;
        }
        Ok(Self {
            rows,
            n,
            blocks,
            padded,
            limbs,
            e,
            dig,
        })
    }

    /// The digit of `(row, block, limb, lane)`.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    #[must_use]
    fn digit(&self, row: usize, block: usize, limb: u32, lane: usize) -> i8 {
        let lane_words = self.limbs as usize * ACT_QK;
        self.dig[(row * self.blocks + block) * lane_words + limb as usize * ACT_QK + lane]
    }
}

/// Where one `(pair, block)` lands on the kernel: the weight scale `2^dw`, the activation scale
/// `2^sigma`, and whether the block pair is carried at all.
#[derive(Clone, Copy)]
struct Place {
    /// Q32.32 exponent of the `f16` weight scale, `[8, 47]`.
    dw: i32,
    /// Exponent of the activation block scale, `[0, 44]`.
    sigma: i32,
    /// `false` when `T < 8`: the block pair's contribution is dropped and the bound counts it.
    live: bool,
}

/// One pair group's kernel inputs: the placement table, the kept pairs and the two `Plane`s that
/// supply the digits.
struct Layout<'a> {
    /// The weight side (its rows are the output rows).
    a: &'a Plane,
    /// The activation side (its rows are the output columns).
    b: &'a Plane,
    /// The kept pairs, in [`kept_pairs`] order.
    pairs: Vec<Pair>,
    /// `[pair][block]` placement.
    place: Vec<Place>,
}

impl Layout<'_> {
    /// The placement of every `(pair, block)`, or the range refusal when a block pair's product
    /// exponent is past the widest `f16` weight scale times the widest `i64` activation scale.
    ///
    /// # Complexity
    ///
    /// `O(P·blocks)` time and space.
    fn placement(a: &Plane, b: &Plane, pairs: &[Pair]) -> Result<Vec<Place>, LimbError> {
        let blocks = a.blocks;
        let mut place = vec![
            Place {
                dw: MIN_PAIR_EXP,
                sigma: 0,
                live: false,
            };
            pairs.len() * blocks
        ];
        let mut gi = 0usize;
        while gi < pairs.len() {
            let (s, t) = pairs[gi];
            let weight = LIMB_BITS * (i32::try_from(s + t).unwrap_or(0) + 2);
            let mut blk = 0usize;
            while blk < blocks {
                let t_exp = i32::try_from(a.e[blk]).unwrap_or(0)
                    + i32::try_from(b.e[blk]).unwrap_or(0)
                    - weight;
                if t_exp > MAX_PAIR_EXP {
                    return Err(LimbError::Range);
                }
                let slot = gi * blocks + blk;
                if t_exp >= MIN_PAIR_EXP {
                    let dw = t_exp.clamp(MIN_PAIR_EXP, MAX_WEIGHT_EXP);
                    place[slot] = Place {
                        dw,
                        sigma: t_exp - dw,
                        live: true,
                    };
                }
                // Below `MIN_PAIR_EXP` the pair cannot be placed (the activation scale would be
                // fractional); the block pair's whole contribution is below 2^-8 words and the
                // bound counts it, so the slot keeps `live: false`.
                blk += 1;
            }
            gi += 1;
        }
        Ok(place)
    }

    /// The `Q8_0` payload of rows `a0..a0+ra`: `ng` pair blocks per 32 inputs, each a `f16` power
    /// of two followed by the weight-side digits.
    ///
    /// # Complexity
    ///
    /// `O(ra·ng·blocks·32)` bytes written; one allocation of `ra·ng·blocks·34`.
    fn weight_payload(&self, a0: usize, ra: usize, g0: usize, g1: usize) -> Vec<u8> {
        let blocks = self.a.blocks;
        let row_bytes = (g1 - g0) * blocks * Q8_0_BLOCK_BYTES;
        let mut raw = vec![0u8; ra * row_bytes];
        let mut r = 0usize;
        while r < ra {
            let row = a0 + r;
            let mut g = g0;
            while g < g1 {
                let (s, _) = self.pairs[g];
                let mut blk = 0usize;
                while blk < blocks {
                    let at = r * row_bytes + ((g - g0) * blocks + blk) * Q8_0_BLOCK_BYTES;
                    let p = self.place[g * blocks + blk];
                    raw[at..at + 2].copy_from_slice(&f16_pow2(p.dw).to_le_bytes());
                    let mut lane = 0usize;
                    while lane < ACT_QK {
                        raw[at + 2 + lane] = if p.live {
                            self.a.digit(row, blk, s, lane).cast_unsigned()
                        } else {
                            0
                        };
                        lane += 1;
                    }
                    blk += 1;
                }
                g += 1;
            }
            r += 1;
        }
        raw
    }

    /// The activation columns of rows `b0..b0+rb`: `ng` pair blocks per 32 inputs, each a power of
    /// two and the activation-side digits.
    ///
    /// # Complexity
    ///
    /// `O(rb·ng·blocks·32)` bytes written; `rb` allocations of `ng·32·blocks` codes.
    fn columns(&self, b0: usize, rb: usize, g0: usize, g1: usize) -> Vec<QuantAct> {
        let blocks = self.b.blocks;
        let padded = self.b.padded;
        let mut qs = Vec::with_capacity(rb);
        let mut c = 0usize;
        while c < rb {
            let row = b0 + c;
            let mut codes = vec![0i8; (g1 - g0) * padded];
            let mut scales = vec![0i64; (g1 - g0) * blocks];
            let mut g = g0;
            while g < g1 {
                let (_, t) = self.pairs[g];
                let mut blk = 0usize;
                while blk < blocks {
                    let p = self.place[g * blocks + blk];
                    scales[(g - g0) * blocks + blk] = 1i64 << p.sigma;
                    if p.live {
                        let at = (g - g0) * padded + blk * ACT_QK;
                        let mut lane = 0usize;
                        while lane < ACT_QK {
                            codes[at + lane] = self.b.digit(row, blk, t, lane);
                            lane += 1;
                        }
                    }
                    blk += 1;
                }
                g += 1;
            }
            qs.push(QuantAct::from_parts(scales, codes));
            c += 1;
        }
        qs
    }
}

impl Plane {
    /// The kernel product of rows `a0..a0+ra` against rows `b0..b0+rb` (both over the same `n`),
    /// row-major `ra × rb`.
    ///
    /// # Complexity
    ///
    /// Temporal `O(ra·n·limbs)` plane-to-payload packing plus `O(P·ra·rb·n)` byte multiply-adds
    /// over `⌈P/g⌉` kernel calls, `g = ⌊16384/padded⌋` pairs per call; spatial `O(ra·g·n)` payload
    /// bytes plus `O(ra·rb)` `i128` accumulators.
    fn product_rows(
        &self,
        b: &Plane,
        a0: usize,
        ra: usize,
        b0: usize,
        rb: usize,
        n: usize,
    ) -> Result<Vec<i64>, LimbError> {
        if n != self.n
            || n != b.n
            || ra == 0
            || rb == 0
            || a0 + ra > self.rows
            || b0 + rb > b.rows
            || self.limbs != b.limbs
        {
            return Err(LimbError::Shape);
        }
        let pairs = kept_pairs(self.limbs);
        let place = Layout::placement(self, b, &pairs)?;
        let layout = Layout {
            a: self,
            b,
            pairs,
            place,
        };
        let group = (MAX_INNER / self.padded).max(1).min(layout.pairs.len());
        let mut acc = vec![0i128; ra * rb];
        let mut g0 = 0usize;
        while g0 < layout.pairs.len() {
            let g1 = (g0 + group).min(layout.pairs.len());
            let n_in = (g1 - g0) * self.padded;
            let raw = layout.weight_payload(a0, ra, g0, g1);
            let qs = layout.columns(b0, rb, g0, g1);
            let cols = match gemm_q8_0(GGML_TYPE_Q8_0, &raw, ra, n_in, &qs) {
                Some(cols) => cols,
                // No AVX-VNNI, an unsupported width or a non-finite scale: the same planes go to
                // the `Q8_0` contract itself, never to a scalar Q32.32 loop.
                None => matmat_q8act(GGML_TYPE_Q8_0, &raw, ra, n_in, &qs)
                    .map_err(|_| LimbError::Range)?,
            };
            let mut j = 0usize;
            while j < rb {
                let col = &cols[j];
                let mut i = 0usize;
                while i < ra {
                    acc[i * rb + j] += i128::from(col[i]);
                    i += 1;
                }
                j += 1;
            }
            g0 = g1;
        }
        let mut out = Vec::with_capacity(ra * rb);
        let mut k = 0usize;
        while k < acc.len() {
            out.push(i64::try_from(acc[k]).map_err(|_| LimbError::Range)?);
            k += 1;
        }
        Ok(out)
    }
}

/// The per-entry word bound of [`gemm_q32`] for `n`, `limbs` and the two operands' largest
/// magnitudes: `⌈P/2⌉ + 1 + ⌈n/32⌉·(1 + ⌈C_L·2^{Ea+Eb−32}⌉)` with `C_L = (33 + 16(L−1))·2^{−7L}`.
///
/// Every term is rounded up and every vanished fraction is carried by the trailing `+1`, so the
/// value is a strict upper bound for every entry the construction can produce; it is the assertion
/// the tests run against.
///
/// # Complexity
///
/// `O(1)`.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the exponents are bounded by 64 + 64; the shifts stay inside i128's 127-bit range \
              and the `as u32` amounts are at most 66"
)]
pub(crate) fn bound_ulps(n: usize, limbs: u32, max_a: u64, max_b: u64) -> i128 {
    let l = i128::from(limbs);
    let pairs = l * (l + 1) / 2;
    let blocks = i128::from((n.div_ceil(ACT_QK)) as u64);
    let ea = i128::from(block_exp(max_a));
    let eb = i128::from(block_exp(max_b));
    let exp = ea + eb - 32 - i128::from(LIMB_BITS) * l;
    let c = 33 + 16 * (l - 1);
    let trunc: i128 = if exp >= 0 {
        let sh = exp as u32;
        if sh >= 120 {
            i128::MAX / 4
        } else {
            (c << sh) + 1
        }
    } else {
        let sh = (-exp) as u32;
        if sh >= 120 {
            1
        } else {
            let d = 1i128 << sh;
            (c + d - 1) / d + 1
        }
    };
    (pairs / 2 + 1) + blocks * (1 + trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::num::value::round_shift_32;

    /// xorshift64*: the seeded generator the tests use (no `rand` dependency).
    fn step(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// `rows × n` Q32.32 words with `|x| ∈ [2^{k+32}, 2^{k+33})`, every 37th element zero and a
    /// random sign on the rest; `k = 0` means a magnitude around `1.0`.
    fn operand(state: &mut u64, rows: usize, n: usize, k: i32) -> Vec<i64> {
        let mut out = Vec::with_capacity(rows * n);
        let base = (k + 32).cast_unsigned();
        let mask = (1i64 << base) - 1;
        let mut i = 0usize;
        while i < rows * n {
            let r = step(state);
            if i % 37 == 3 {
                out.push(0);
            } else {
                let mag = (1i64 << base) | (r.cast_signed() & mask);
                out.push(if r >> 63 == 0 { mag } else { -mag });
            }
            i += 1;
        }
        out
    }

    /// The exact Q32.32 product: `round_shift_32(Σ_l a·b)`, the contract the kernel reproduces.
    fn reference(a: &[i64], m: usize, b: &[i64], p: usize, n: usize) -> Vec<i64> {
        let mut c = vec![0i64; m * p];
        let mut i = 0usize;
        while i < m {
            let mut j = 0usize;
            while j < p {
                let mut acc: i128 = 0;
                let mut l = 0usize;
                while l < n {
                    acc += i128::from(a[i * n + l]) * i128::from(b[j * n + l]);
                    l += 1;
                }
                c[i * p + j] = i64::try_from(round_shift_32(acc)).expect("reference fits i64");
                j += 1;
            }
            i += 1;
        }
        c
    }

    fn max_abs(v: &[i64]) -> u64 {
        v.iter().map(|&x| x.unsigned_abs()).max().unwrap_or(0)
    }

    /// The largest entry error of `got` against `want`.
    fn max_err(got: &[i64], want: &[i64]) -> i128 {
        let mut e: i128 = 0;
        for (g, w) in got.iter().zip(want) {
            let d = (i128::from(*g) - i128::from(*w)).abs();
            if d > e {
                e = d;
            }
        }
        e
    }

    /// (a) Every entry is inside [`bound_ulps`] at 3 and 4 limbs, over the shape grid and a
    /// magnitude grid that keeps every product inside `i64`.
    #[test]
    fn entries_are_inside_the_bound() {
        let shapes = [
            (1usize, 1usize),
            (1, 7),
            (7, 1),
            (7, 7),
            (64, 7),
            (7, 64),
            (64, 64),
        ];
        let ns = [1usize, 31, 32, 33, 896];
        let mags = [(-20, 20), (0, 0), (10, 7), (-20, -20), (20, -20), (-7, 10)];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut worst = 0i128;
        let mut checked = 0usize;
        for limbs in [3u32, 4] {
            for (t, &(m, p)) in shapes.iter().enumerate() {
                for (u, &n) in ns.iter().enumerate() {
                    let (ka, kb) = mags[(t + u) % mags.len()];
                    let a = operand(&mut seed, m, n, ka);
                    let b = operand(&mut seed, p, n, kb);
                    let got = gemm_q32(&a, m, &b, p, n, limbs).expect("carried");
                    let want = reference(&a, m, &b, p, n);
                    let bound = bound_ulps(n, limbs, max_abs(&a), max_abs(&b));
                    let err = max_err(&got, &want);
                    assert!(
                        err <= bound,
                        "limbs {limbs} m {m} p {p} n {n} ka {ka} kb {kb}: err {err} > bound {bound}"
                    );
                    worst = worst.max(if bound == 0 { 0 } else { err * 1000 / bound });
                    checked += 1;
                }
            }
        }
        println!("(a) {checked} products, worst error = {worst}⁄1000 of the bound");
    }

    /// (b) `syrk_q32` is `gemm_q32(a, a)` bit for bit and symmetric.
    #[test]
    fn syrk_matches_gemm_bit_for_bit() {
        let mut seed = 0xdead_beef_1234_5678u64;
        for limbs in [3u32, 4] {
            for (m, n) in [(1usize, 1usize), (1, 33), (7, 31), (64, 32), (7, 896)] {
                let a = operand(&mut seed, m, n, 3);
                let syrk = syrk_q32(&a, m, n, limbs).expect("carried");
                let gemm = gemm_q32(&a, m, &a, m, n, limbs).expect("carried");
                assert_eq!(syrk, gemm, "limbs {limbs} m {m} n {n}");
                for i in 0..m {
                    for j in 0..m {
                        assert_eq!(syrk[i * m + j], syrk[j * m + i], "asymmetric ({i},{j})");
                    }
                }
            }
        }
    }

    /// (c) `gemv_q32` is the `m = 1` `gemm_q32` bit for bit and inside the bound; `rowdot_q32` is
    /// the `gemm_q32` diagonal bit for bit.
    #[test]
    fn gemv_and_rowdot_agree_with_gemm() {
        let mut seed = 0x0123_4567_89ab_cdefu64;
        for limbs in [3u32, 4] {
            for (p, n) in [(1usize, 1usize), (7, 31), (64, 32), (33, 33), (7, 896)] {
                let x = operand(&mut seed, 1, n, 4);
                let b = operand(&mut seed, p, n, -3);
                let y = gemv_q32(&x, &b, p, n, limbs).expect("carried");
                let g = gemm_q32(&x, 1, &b, p, n, limbs).expect("carried");
                assert_eq!(y, g, "gemv p {p} n {n}");
                let want = reference(&x, 1, &b, p, n);
                let bound = bound_ulps(n, limbs, max_abs(&x), max_abs(&b));
                assert!(max_err(&y, &want) <= bound, "gemv bound p {p} n {n}");
            }
            for (m, n) in [(1usize, 1usize), (7, 32), (7, 33), (64, 896)] {
                let a = operand(&mut seed, m, n, 3);
                let b = operand(&mut seed, m, n, -2);
                let dots = rowdot_q32(&a, &b, m, n, limbs).expect("carried");
                let full = gemm_q32(&a, m, &b, m, n, limbs).expect("carried");
                for i in 0..m {
                    assert_eq!(dots[i], full[i * m + i], "rowdot m {m} n {n} i {i}");
                }
            }
        }
    }

    /// (d) Shape refusal: a mismatched slice, a zero dimension, no limbs, an over-wide
    /// contraction.
    #[test]
    fn shape_refusals() {
        let a = vec![1i64 << 32; 4];
        let b = vec![1i64 << 32; 4];
        assert_eq!(gemm_q32(&a[..3], 2, &b, 2, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 3, &b, 2, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 2, &b[..3], 2, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 0, &b, 2, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 2, &b, 0, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemm_q32(&a, 2, &b, 2, 2, 0), Err(LimbError::Shape));
        let wide = vec![1i64; MAX_INNER + 32];
        assert_eq!(
            gemm_q32(&wide, 1, &wide, 1, MAX_INNER + 32, 3),
            Err(LimbError::Shape)
        );
        assert_eq!(syrk_q32(&a, 3, 2, 3), Err(LimbError::Shape));
        assert_eq!(gemv_q32(&a, &b, 2, 3, 3), Err(LimbError::Shape));
        assert_eq!(rowdot_q32(&a, &b, 3, 2, 3), Err(LimbError::Shape));
    }

    /// (e) Range refusal: words whose product the scale placement cannot carry.
    #[test]
    fn range_refusal_on_unplaceable_words() {
        let big = vec![i64::MAX; 32];
        assert_eq!(gemm_q32(&big, 1, &big, 1, 32, 3), Err(LimbError::Range));
        let mid = vec![1i64 << 62; 32];
        assert_eq!(gemm_q32(&mid, 1, &mid, 1, 32, 3), Err(LimbError::Range));
        assert_eq!(gemv_q32(&big, &big, 1, 32, 3), Err(LimbError::Range));
        assert_eq!(rowdot_q32(&big, &big, 1, 32, 3), Err(LimbError::Range));
        // A huge operand against a zero column is exactly zero: `Range` is about the product, not
        // the operand.
        let zero = vec![0i64; 32];
        assert_eq!(gemm_q32(&big, 1, &zero, 1, 32, 3), Ok(vec![0i64]));
    }

    /// (f) Integer words (multiples of `2^32`, `|v| ≤ 127`) are bit exact: every digit beyond the
    /// second is zero, so no kept pair is dropped and no residual remains.
    #[test]
    fn integer_operands_are_bit_exact() {
        let mut seed = 0xfeed_face_cafe_beefu64;
        for limbs in [3u32, 4] {
            for (m, p, n) in [
                (1usize, 1usize, 1usize),
                (7, 7, 32),
                (7, 7, 33),
                (64, 7, 896),
            ] {
                let mut a = Vec::with_capacity(m * n);
                let mut b = Vec::with_capacity(p * n);
                for _ in 0..m * n {
                    let v = (step(&mut seed) % 255).cast_signed() - 127;
                    a.push(v << 32);
                }
                for _ in 0..p * n {
                    let v = (step(&mut seed) % 255).cast_signed() - 127;
                    b.push(v << 32);
                }
                let got = gemm_q32(&a, m, &b, p, n, limbs).expect("carried");
                let want = reference(&a, m, &b, p, n);
                assert_eq!(got, want, "limbs {limbs} m {m} p {p} n {n}");
            }
        }
    }

    /// (R8) Paired microbench at the Qwen2.5-0.5B shapes: 5 alternating runs of the limb product
    /// and of the scalar `i128` loop, mean and spread in ns per MAC.
    #[test]
    #[ignore = "measurement: 5 alternating runs at the production shapes"]
    #[allow(
        clippy::cast_precision_loss,
        reason = "m·p·n ≤ 2^32, exactly representable in f64, and only used to print ns/MAC"
    )]
    fn limb_bench() {
        let shapes = [(512usize, 896usize, 896usize), (512, 4864, 896)];
        let mut seed = 0xabcd_ef01_2345_6789u64;
        for (m, p, n) in shapes {
            let a = operand(&mut seed, m, n, 3);
            let b = operand(&mut seed, p, n, 3);
            // Warm both paths and validate the result against the oracle and the bound.
            let got = gemm_q32(&a, m, &b, p, n, 3).expect("carried");
            let want = reference(&a, m, &b, p, n);
            let bound = bound_ulps(n, 3, max_abs(&a), max_abs(&b));
            let err = max_err(&got, &want);
            assert!(err <= bound, "err {err} > bound {bound}");
            let macs = (m * p * n) as f64;
            let mut kern = Vec::new();
            let mut scal = Vec::new();
            for run in 0..5 {
                let t0 = std::time::Instant::now();
                let k = gemm_q32(&a, m, &b, p, n, 3).expect("carried");
                let dk = t0.elapsed();
                let t1 = std::time::Instant::now();
                let s = reference(&a, m, &b, p, n);
                let ds = t1.elapsed();
                assert_eq!(k, got, "run {run} kernel not reproducible");
                assert_eq!(s, want, "run {run} reference not reproducible");
                kern.push(dk.as_secs_f64());
                scal.push(ds.as_secs_f64());
                println!(
                    "{run}: limb {:>8.3} ms  i128 {:>8.3} ms",
                    dk.as_secs_f64() * 1e3,
                    ds.as_secs_f64() * 1e3
                );
            }
            for (name, v) in [("limb", &kern), ("i128", &scal)] {
                let mean = v.iter().sum::<f64>() / 5.0;
                let lo = v.iter().copied().fold(f64::MAX, f64::min);
                let hi = v.iter().copied().fold(0.0, f64::max);
                println!(
                    "m {m} p {p} n {n} {name}: mean {:.3} ms  spread [{:.3}, {:.3}] ms  \
                     {:.3} ns/MAC  bound {bound} observed {err}",
                    mean * 1e3,
                    lo * 1e3,
                    hi * 1e3,
                    mean * 1e9 / macs
                );
            }
        }
    }
}
