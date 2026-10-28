//! The fold face of the engine's one integer product role (`gemm_q8_0`, [`super::gemm`]): a stored
//! [`FoldPlane`] operand projected against the `Q8Act` activation, without expanding the plane to a
//! dense weight matrix.
//!
//! A plane's stored form is `(idx, lo, sp, r_k, c_head, c_q, d_head, d_q)`; its weight words are
//! `CodedPlane::weight` (`learn/fold/finish/plane.rs:268`), which is
//! `compose_plane(S, r_q, c_q, d_q)` = `rows_pass(M(S ⊙ d_q), r_q, c_q)` with `M` the normalised
//! 128-point Walsh–Hadamard operator of `Op::Fwht` (`plane.rs:434`): the butterfly, then
//! `mul(·, INV_SQRT128_Q32)` per element. `rows_pass` is `W[i,j] = round(r[i,j/128]·c[j]·Y[i,j]/2^32)`
//! with `Y = M(S ⊙ d_q)`.
//!
//! # The factorisation
//!
//! `M` is symmetric, so for the `Q8Act` codes `xq` (scale `S_b`, block 32) the projection is
//!
//! ```text
//! out[i] = round_shift_32( Σ_g 2^{emin(g)} · round( r[i,g] · Σ_{j∈g} σ[i,j]·b[j] / 2^64 ) )
//! b       = d_q ⊙ M( c_q ⊙ xq' ) / 2^32          (one rounding per element, `mul` twice)
//! xq'[j]  = xq[j] · 2^{e(j) − emin(j/128)}        (e = log2 S_b, exact integer widening)
//! ```
//!
//! The outer `round_shift_32` is the role's own epilogue: `gemm_q8_0`'s `exact_dot`
//! (`src/engine/kern/gemm.rs:434`) accumulates `Σ_b d_b·S_b·dot_b` in `i128` and applies
//! `num::value::round_shift_32` **once**, after every term is summed (`src/engine/kern/gemm.rs:461`).
//! Without it the written word is `2^32` times the `Q32.32` output the residual stream holds, and
//! at the step's activation magnitude (`|x| < ONE`) the unscaled word passes `i64` on every plane,
//! so the face refuses where the dense route does not.
//!
//! with `σ[i,j]` the `Q32.32` codebook word the plane's `idx` names. `S_b` is a power of two
//! (`kern::pow2::act_scale`), so the block scale is carried as an exponent: the four blocks of a
//! 128-group are normalised to the group's minimum exponent and the residual is exact. The
//! activation-side transform runs **once per activation**, not once per output row: `b` is `n`
//! words.
//!
//! # The block table (lookup-table product)
//!
//! The codebook has 256 rows of 8 coordinates, so a block's contribution to a row is one of 256
//! values that do not depend on the row. Per activation, for every 8-column block `k` of `b`, build
//!
//! ```text
//! T_k[c] = Σ_{m<8} σ[c][m]·b[8k+m]        (c < 256)
//! ```
//!
//! then the block's contribution to output row `i` is the gather `T_k[idx[i,k]]`, and a
//! `(row, group)` partial sum is the sum of the group's 16 gathered entries. This is the lookup-table
//! structure codebook-quantisation CPU kernels use (AQLM, Egiazarian et al., arXiv:2401.06118, §3;
//! `QuIP#`, Tseng et al., arXiv:2402.04396; `T-MAC`, Wei et al., arXiv:2404.00453); it is applied
//! here, not ported, and the base it subsumes is this file's own per-`(row, column)` `i128` codebook dot,
//! which is kept as the test oracle.
//!
//! `T_k` is exact: `Σ_m σ[c][m]·b[8k+m]` is an integer sum, and so is `Σ_{k∈g} T_k[idx[i,k]]`, so
//! the gather sum equals the per-column sum word for word and the epilogue is unchanged. Every
//! `T_k[c]` fits `i128` (`|b| < 2^47`, `|σ| ≤ 2^34`, 8 terms).
//!
//! The build exploits the codebook's own layout (`learn/code/e8.rs` row forms), so it costs far less
//! than 256·8 multiply-adds per block:
//!
//! - rows `0..128` are the sign vectors `(±1)^8`: the block's 256 sign sums `Σ_m ±b[m]` are filled a
//!   level at a time by `f[i + 2^m] = f[i] − 2·b[m]`, 255 subtractions for all 256, and a row reads
//!   one of them;
//! - rows `128..240` are `±2e_a ±2e_b`: a sum or difference of two doubled block words, 2 adds a row;
//! - rows `240..256` are `±√8·e_i`: one `i128` product each.
//!
//! # Per-call cost
//!
//! Per activation: `n/8` blocks, each 255 subtractions for the sign sums, 256 stores, 112
//! add/subtract pairs and 8 `i128` products; then `o·(n/8)` `i128` loads and adds, `o·n/128` `i128`
//! divisions for the epilogue, and `O(n·log2 128)` for the activation transform. At `896 × 896` that
//! is 112 builds (≈ 28 600 subtractions and 28 672 stores) and 100 352 gathers, against the
//! per-column body's 802 816 `i128` multiply-adds. The coded fields add the per-call decode: 1 792
//! column divisions and 6 272 row-grid divisions at that shape (about 4% of the call, measured in
//! `kern::fold::timing`).
//!
//! Table memory: one block's table is `256·16 = 4096` bytes and one group's per-row partial sums are
//! `8·o` bytes (7168 at `o = 896`); the table is rebuilt per block and stays in L1, and the whole
//! group's 16 tables are never resident (they would be 64 KiB). This is per-call scratch, never
//! resident weight.
//!
//! # The resident form
//!
//! The face stores the pack record's own coded fields ([`FoldFace`]): `idx`, the per-row `(lo, sp)`
//! grid, the 4-bit row levels, and the 6- and 5-bit column codes with their `f16` endpoints. Nothing
//! decoded is resident. Each call decodes into scratch: the `n` `c` and `n` `d` column words once
//! (`O(n)` divisions), and one row-grid word per `(row, group)` inside the epilogue (`O(o·g)`
//! divisions). That is `2n + o·g` divisions against the gather body's `o·n/8` `i128` adds, and it is
//! the field accounting `learn::onebit_bits` prices: at `896 × 896` the face is `119 064 B` for
//! `802 816` weights, `1.186 b/w`, where the decoded form it replaces is `164 864 B`, `1.643 b/w`.
//!
//! The decode is word-for-word the one the previous face performed at load
//! (`fold_plane_coded(...).qvals()`): `r_q[i,g] = lo[i] + rne(r_k[i,g]·sp[i] / 15)`
//! (`learn::fold::fit::quant::r_level`), and `c_q`/`d_q` from the `f16` endpoints and the codes
//! (`learn::fold::run::pack`'s own `col_words`). `kern::fold::oracle::coded_face_bit_identical` holds
//! that equality against the decoded face, field by field and word by word, on three seeded
//! `896 × 896` planes and one row-restricted `896 × 4864` plane.
//!
//! # Tolerance
//!
//! Three roundings separate this product from `CodedPlane::weight()`'s exact words: `c_q ⊙ xq'`
//! (`mul`, ≤ 0.5 ulp), `d_q ⊙ ·` (`mul`, ≤ 0.5 ulp), and the per-`(row, group)` epilogue (≤ 0.5
//! ulp); a fourth is the single `round_shift_32` (≤ 0.5 output word, shared with every `Q8Act`
//! projection of the same activation). A `c_q` rounding of `η_j ≤ 0.5` moves `M`'s output by at
//! most `√128/2` per element (128 unit coefficients of modulus `2^-7/2`), which the `d_q` scale and
//! the codebook sum carry; `tests/engine_fold_pack.rs` derives the resulting bound from the
//! fixture's own maxima, in the `Q32.32` units the shift leaves.

#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the face indexes `o/n/i/j/g/b` planes the way the pack layout and the transform \
              write them, and the exponent arithmetic is bounded by the checked products"
)]

use crate::engine::chart::QuantAct;
use crate::format::packing::{f16_bits_to_fixed, FoldPlane};
use crate::learn::code::e8::SQRT8_Q32;
use crate::learn::fold::finish::plane::mul;
use crate::learn::fold::fit::icm::codebook;
use crate::learn::fold::fit::quant::r_level;
use crate::learn::fold::fit::steps::GROUP;
use crate::learn::solve::q32::div_rne_pos;

/// Codes per codebook block (`learn::fold::fit::icm::BLOCK`).
const BLOCK_USIZE: usize = 8;
/// Codebook rows ([`crate::learn::code::e8::CODEBOOK_LEN`]).
const CODE_LEN: usize = 256;
/// `ONE` as a magnitude, for the codebook's word comparison.
const ONE_U: u64 = crate::engine::ONE.unsigned_abs();
/// `2·ONE` as a magnitude, the D8 roots' coordinate.
const TWO_ONE_U: u64 = 2 * ONE_U;
/// `√8·2^32` as a magnitude ([`SQRT8_Q32`]), the O8 vertices' coordinate.
const SQRT8_U: u64 = SQRT8_Q32.unsigned_abs();
/// Activation block width of the `Q8Act` quantiser.
const ACT_BLOCK: usize = 32;

/// `round(2^32 / √128)`, the transform's scale: the same constant `plane.rs:64` composes every
/// plane's weight with, so the activation-side transform is that operator, not an approximation of
/// it.
const INV_SQRT128_Q32: i64 = 379_625_062;

/// A stored plane in the form the product executes from: the pack record's own fields, not the words
/// they decode to. The codebook index per 8-column block, the per-row `(lo, sp)` grid, the 4-bit row
/// levels packed two per byte, and the two column fields as `f16` endpoints plus packed 6- and 5-bit
/// codes. The dense weight is never built, and neither are the `r`/`c`/`d` words between calls: they
/// are decoded into per-call scratch.
///
/// At `896 × 896` the resident form is `119 064 B` for `802 816` weights, `1.186 b/w`; the decoded
/// form it replaces is `164 864 B`, `1.643 b/w`. `bytes()` counts the fields below only, never the
/// scratch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldFace {
    /// Output rows.
    o: usize,
    /// Input columns; a positive multiple of 128.
    n: usize,
    /// Codebook row per 8-column block, row-major `[o, n/8]`.
    idx: Vec<u8>,
    /// Per-row lower endpoint of the `r` grid, `Q32.32`, length `o`.
    lo: Vec<i64>,
    /// Per-row width of the `r` grid, `Q32.32`, length `o`.
    sp: Vec<i64>,
    /// The 4-bit row level per `(row, group)`, two codes per byte, low nibble first.
    r_k: Vec<u8>,
    /// The `c` field's `(f16 min, f16 max)` bit patterns.
    c_head: [u16; 2],
    /// The `c` field's 6-bit column codes, packed least-significant bit first.
    c_q: Vec<u8>,
    /// The `d` field's `(f16 min, f16 max)` bit patterns.
    d_head: [u16; 2],
    /// The `d` field's 5-bit column codes, packed least-significant bit first.
    d_q: Vec<u8>,
}

impl FoldFace {
    /// The face of a stored plane: its coded fields as the pack writes them, with no decoded word
    /// resident. A plane the pack layout refuses, a code wider than its field or a non-finite `f16`
    /// endpoint is refused here, so a face that exists decodes on every call.
    ///
    /// # Complexity
    ///
    /// `O(o·n/8 + n)` reads and one pack of each code field; no `o × n` buffer is retained. The
    /// per-call decode is [`matvec_fold_quant`]'s: `O(o·g + n)` word operations.
    ///
    /// # Errors
    ///
    /// `None` on a shape the plane layout refuses, a code outside its field's width, or a code field
    /// the `f16` endpoints cannot place in `Q32.32`.
    #[must_use]
    pub fn from_plane(f: &FoldPlane) -> Option<Self> {
        let (o, n) = (f.o, f.n);
        if o == 0 || n == 0 || !n.is_multiple_of(GROUP) {
            return None;
        }
        let g = n / GROUP;
        if f.idx.len() != o * (n / BLOCK_USIZE)
            || f.lo.len() != o
            || f.sp.len() != o
            || f.r_k.len() != o * g
            || f.c_q.len() != n
            || f.d_q.len() != n
        {
            return None;
        }
        // The checks the pack reader made, kept because the codes are decoded here rather than at
        // the read: a level, a code wider than its field, or an `f16` endpoint at infinity is not a
        // plane the product can execute.
        if f.r_k.iter().any(|&k| k > 15)
            || f.c_q.iter().any(|&c| u32::from(c) >= (1 << 6))
            || f.d_q.iter().any(|&c| u32::from(c) >= (1 << 5))
        {
            return None;
        }
        f16_bits_to_fixed(f.c_head[0]).ok()?;
        f16_bits_to_fixed(f.c_head[1]).ok()?;
        f16_bits_to_fixed(f.d_head[0]).ok()?;
        f16_bits_to_fixed(f.d_head[1]).ok()?;
        Some(Self {
            o,
            n,
            idx: f.idx.clone(),
            lo: f.lo.clone(),
            sp: f.sp.clone(),
            r_k: pack_bits(&f.r_k, 4),
            c_head: f.c_head,
            c_q: pack_bits(&f.c_q, 6),
            d_head: f.d_head,
            d_q: pack_bits(&f.d_q, 5),
        })
    }

    /// The output rows written. `O(1)`.
    #[must_use]
    pub fn o(&self) -> usize {
        self.o
    }

    /// The input columns consumed. `O(1)`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The stored operand's resident bytes: one codebook index per 8 weights, one `(lo, sp)` word
    /// pair per row, one packed 4-bit level per `(row, group)`, the two `f16` head pairs, and the
    /// packed column codes. `O(1)`, the eight lengths' sum.
    ///
    /// Per-call scratch is excluded and is [`matvec_fold_quant`]'s own: two `n`-word column vectors
    /// (`2·8·n` = 14 336 B at `896 × 896`), the `o`-word group sums and `i128` accumulator
    /// (`24·o` = 21 504 B), and one 8-column block table of `256·16 = 4 096 B`.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.idx.len()
            + (self.lo.len() + self.sp.len()) * std::mem::size_of::<i64>()
            + self.r_k.len()
            + 2 * std::mem::size_of::<[u16; 2]>()
            + self.c_q.len()
            + self.d_q.len()
    }
}

/// `codes`, `bits` wide, packed least-significant bit first into `ceil(len·bits/8)` bytes. `O(len)`
/// shifts and stores.
fn pack_bits(codes: &[u8], bits: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((codes.len() * bits as usize).div_ceil(8));
    let mut acc = 0_u32;
    let mut have = 0_u32;
    for &c in codes {
        acc |= u32::from(c) << have;
        have += bits;
        while have >= 8 {
            out.push(u8::try_from(acc & 0xff).unwrap_or(0));
            acc >>= 8;
            have -= 8;
        }
    }
    if have > 0 {
        out.push(u8::try_from(acc & 0xff).unwrap_or(0));
    }
    out
}

/// Code `at` of a [`pack_bits`] stream, `bits` wide. `O(1)`; the caller keeps `at` inside
/// `len·bits/8`.
fn code_at(buf: &[u8], bits: u32, at: usize) -> u8 {
    let bit = at * bits as usize;
    let byte = bit / 8;
    let shift = u32::try_from(bit % 8).unwrap_or(0);
    let mut acc = u32::from(buf[byte]);
    if shift + bits > 8 {
        acc |= u32::from(buf[byte + 1]) << 8;
    }
    u8::try_from((acc >> shift) & ((1 << bits) - 1)).unwrap_or(0)
}

/// `round(num / DEN)`, the nearest word, for odd `DEN`: `learn::fold::fit::quant::decode_word`'s
/// `div_rne_pos(·, 15)` on one `i64`. An odd denominator admits no exact tie, so nearest is one
/// half-step round and the constant divisor folds to a multiply. `O(1)`.
fn rne_nearest<const DEN: i64>(num: i64) -> Option<i64> {
    num.checked_add(DEN / 2)?.checked_div_euclid(DEN)
}

/// The grid word of stored level `k` on the row grid `(lo, sp)`: `learn::fold::fit::quant::r_level`'s
/// `lo + rne(k·sp / 15)`, on `i64` words with the `i128` body kept for a span that leaves `i64`.
/// `O(1)`.
fn grid_word(lo: i64, sp: i64, k: u8) -> Option<i64> {
    let Some(num) = i64::from(k).checked_mul(sp) else {
        return r_level(lo, sp, k);
    };
    lo.checked_add(rne_nearest::<15>(num)?)
}

/// The `n` column words of one stored field: `c_min + rne(code·span / levels)` with `c_min`, `c_max`
/// the `f16` endpoints, `span = max(c_max − c_min, 1)` and `levels = 2^bits − 1` — the decode
/// `learn::fold::run::pack` reads the field with. `O(n)` divisions.
fn column_words(head: [u16; 2], codes: &[u8], bits: u32, n: usize) -> Option<Vec<i64>> {
    let c_min = f16_bits_to_fixed(head[0]).ok()?;
    let c_max = f16_bits_to_fixed(head[1]).ok()?;
    let span = (i128::from(c_max) - i128::from(c_min)).max(1);
    if !(bits == 5 || bits == 6) {
        return None;
    }
    let levels = (1_i128 << bits) - 1;
    let mut out = Vec::with_capacity(n);
    for at in 0..n {
        let code = i128::from(code_at(codes, bits, at));
        let off = match i64::try_from(code * span) {
            // Both fields' level counts are odd, so the nearest grid point is one half-step round
            // on a constant divisor.
            Ok(num) if bits == 6 => i128::from(rne_nearest::<63>(num)?),
            Ok(num) => i128::from(rne_nearest::<31>(num)?),
            Err(_) => div_rne_pos(code * span, levels),
        };
        out.push(i64::try_from(i128::from(c_min).checked_add(off)?).ok()?);
    }
    Some(out)
}

impl FoldFace {
    /// The row grid word of `(row, group)`: the stored 4-bit level on the row's `(lo, sp)`. `O(1)`.
    fn r_word(&self, i: usize, gg: usize) -> Option<i64> {
        let k = code_at(&self.r_k, 4, i * (self.n / GROUP) + gg);
        grid_word(self.lo[i], self.sp[i], k)
    }

    /// The `c` and `d` column words, decoded once per call. `O(n)`, scratch [`FoldFace::bytes`]
    /// excludes.
    fn column_fields(&self) -> Option<(Vec<i64>, Vec<i64>)> {
        Some((
            column_words(self.c_head, &self.c_q, 6, self.n)?,
            column_words(self.d_head, &self.d_q, 5, self.n)?,
        ))
    }
}

/// A whole quotient `v / 2^s`, ties away from zero, or `None` when it leaves `i64`. `O(1)`.
fn round_pow2(v: i128, s: u32) -> Option<i64> {
    let half = 1_i128 << (s - 1);
    let q = if v >= 0 {
        (v + half) >> s
    } else {
        -((-v + half) >> s)
    };
    i64::try_from(q).ok()
}

/// The `Q8Act` quantisation of `x`: per-block scales (powers of two) and per-element codes. The same
/// `chart::ffn::quant_act_pooled` body the `Q8Act` projection uses, so the fold face and the dense
/// face share one activation quantiser.
///
/// # Errors
///
/// `None` where the quantiser refuses (`QuantAct::dequant` would leave `Q32.32`).
#[must_use]
pub fn quant_act_fold(x: &[i64]) -> Option<(Vec<i64>, Vec<i8>)> {
    let q = crate::engine::chart::ffn::quant_act_pooled(x)?;
    Some((q.scales().to_vec(), q.codes().to_vec()))
}

/// The fold face's projection of the activation words `x`: quantise, then [`matvec_fold_codes`].
/// `O(o·n)` multiply-adds; see the module's per-call cost.
///
/// # Errors
///
/// `None` on the quantiser's refusal or a product that leaves range.
#[must_use]
pub fn matvec_fold(face: &FoldFace, x: &[i64]) -> Option<Vec<i64>> {
    let q = crate::engine::chart::ffn::quant_act_pooled(x)?;
    matvec_fold_quant(face, &q)
}

/// The activation-side transform, once per activation: the block words `b = d_q ⊙ M(c_q ⊙ xq')` and
/// each group's minimum activation exponent. `O(n·log2 128)` adds and `2n` `mul`s.
///
/// # Errors
///
/// `None` on a shape disagreement, a non-power-of-two scale or an activation-exponent spread past
/// 20.
fn activation_words(face: &FoldFace, q: &QuantAct) -> Option<(Vec<i64>, Vec<u32>)> {
    let (c_q, d_q) = face.column_fields()?;
    transform_words(face.n, &c_q, &d_q, q)
}

/// [`activation_words`] on the two column fields as words, so a caller that already holds them (the
/// test oracle of the pre-resident face) pays no decode. `O(n·log2 128)` adds and `2n` `mul`s.
///
/// # Errors
///
/// `None` on a shape disagreement, a non-power-of-two scale or an activation-exponent spread past 20.
fn transform_words(
    n: usize,
    c_q: &[i64],
    d_q: &[i64],
    q: &QuantAct,
) -> Option<(Vec<i64>, Vec<u32>)> {
    if c_q.len() != n || d_q.len() != n {
        return None;
    }
    let g = n / GROUP;
    let codes = q.codes();
    let scales = q.scales();
    if codes.len() < n || scales.len() != n.div_ceil(ACT_BLOCK) {
        return None;
    }
    // The activation-side transform, once per activation: widen each block's scale into the group's
    // minimum exponent (exact, since every scale is a power of two), then `t = c_q ⊙ xq'`, then `M`.
    let mut emins = vec![0_u32; g];
    for (gg, slot) in emins.iter_mut().enumerate() {
        let mut emin = u32::MAX;
        for bl in 0..(GROUP / ACT_BLOCK) {
            let sc = scales[gg * (GROUP / ACT_BLOCK) + bl];
            if sc > 0 {
                emin = emin.min(sc.unsigned_abs().trailing_zeros());
            }
        }
        *slot = if emin == u32::MAX { 0 } else { emin };
    }
    let mut t = vec![0_i64; n];
    for gg in 0..g {
        let emin = emins[gg];
        for b in 0..(GROUP / ACT_BLOCK) {
            let s = scales[gg * (GROUP / ACT_BLOCK) + b];
            let shift = if s > 0 {
                s.unsigned_abs().trailing_zeros() - emin
            } else {
                0
            };
            if shift > 20 {
                return None;
            }
            for j in 0..ACT_BLOCK {
                let at = gg * GROUP + b * ACT_BLOCK + j;
                let code = i64::from(codes[at]) << shift;
                t[at] = c_q[at].checked_mul(code)?;
            }
        }
    }
    let mut b = vec![0_i64; n];
    for chunk in 0..g {
        let base = chunk * GROUP;
        let mut h = 1;
        while h < GROUP {
            let mut i = 0;
            while i < GROUP {
                for j in i..(i + h) {
                    let (x, y) = (t[base + j], t[base + j + h]);
                    t[base + j] = x.checked_add(y)?;
                    t[base + j + h] = x.checked_sub(y)?;
                }
                i += 2 * h;
            }
            h *= 2;
        }
        for j in 0..GROUP {
            let at = base + j;
            b[at] = mul(d_q[at], mul(t[at], INV_SQRT128_Q32)?)?;
        }
    }
    Some((b, emins))
}

/// The codebook's rows in the form the block table builds them from.
struct Lut {
    /// Rows that are sign vectors `(±1)^8`, as `(row, transform slot)`; the slot's bit `m` is the
    /// sign of coordinate `m`.
    spinor: Vec<(u8, u8)>,
    /// Rows that are `±2e_a ±2e_b`, as `(row, a, b, sa, sb)`.
    root: Vec<(u8, u8, u8, bool, bool)>,
    /// Rows that are `±√8·e_i`, as `(row, i, neg)`.
    o8: Vec<(u8, u8, bool)>,
    /// Rows matching none of the above, as `(row, words)`: the exact sum, row by row.
    dense: Vec<(u8, [i64; BLOCK_USIZE])>,
}

impl Lut {
    /// Classifies [`codebook`]'s 256 rows. `O(256·8)` once per process; a row matching no form keeps
    /// its own words, so a codebook whose layout changes stays exact, only slower.
    fn build() -> Self {
        let cb = codebook();
        let mut out = Self {
            spinor: Vec::new(),
            root: Vec::new(),
            o8: Vec::new(),
            dense: Vec::new(),
        };
        for (j, row) in cb.iter().enumerate() {
            let Ok(c) = u8::try_from(j) else { continue };
            let nz: Vec<usize> = (0..BLOCK_USIZE).filter(|&m| row[m] != 0).collect();
            if nz.len() == BLOCK_USIZE && row.iter().all(|w| w.unsigned_abs() == ONE_U) {
                let slot = (0..BLOCK_USIZE).fold(0_u8, |acc, m| acc | (u8::from(row[m] < 0) << m));
                out.spinor.push((c, slot));
            } else if nz.len() == 2
                && row
                    .iter()
                    .all(|w| w.unsigned_abs() == 0 || w.unsigned_abs() == TWO_ONE_U)
            {
                match (u8::try_from(nz[0]), u8::try_from(nz[1])) {
                    (Ok(a), Ok(d)) => out.root.push((c, a, d, row[nz[0]] > 0, row[nz[1]] > 0)),
                    _ => out.dense.push((c, *row)),
                }
            } else if nz.len() == 1 && row[nz[0]].unsigned_abs() == SQRT8_U {
                match u8::try_from(nz[0]) {
                    Ok(i) => out.o8.push((c, i, row[nz[0]] < 0)),
                    Err(_) => out.dense.push((c, *row)),
                }
            } else {
                out.dense.push((c, *row));
            }
        }
        out
    }
}

/// The process's one [`Lut`]: the codebook is fixed, so its classification is computed once.
fn lut() -> &'static Lut {
    static LUT: std::sync::OnceLock<Lut> = std::sync::OnceLock::new();
    LUT.get_or_init(Lut::build)
}

/// One 8-column block's table: `t[c] = Σ_{m<8} σ[c][m]·b[base + m]` for all 256 rows, exactly.
/// `O(256)` stores, one 8-point Walsh–Hadamard transform (24 adds), 112 add/subtracts and 8 `i128`
/// products.
fn build_tile(l: &Lut, b: &[i64], base: usize, t: &mut [i128; CODE_LEN]) {
    let bb = &b[base..base + BLOCK_USIZE];
    // All 256 sign sums `Σ_m ±b[m]`, indexed by the sign bitmask (`bit m` set = coordinate `m`
    // negative): `f[i + 2^m] = f[i] − 2·b[m]` fills each level from the one below, 255 subtractions.
    let mut f = [0_i64; 256];
    f[0] = bb.iter().sum();
    for m in 0..BLOCK_USIZE {
        let step = 1_usize << m;
        let two = bb[m] << 1;
        for i in 0..step {
            f[i + step] = f[i] - two;
        }
    }
    let mut dbl = [0_i64; BLOCK_USIZE];
    for m in 0..BLOCK_USIZE {
        dbl[m] = bb[m] << 1;
    }
    for &(c, slot) in &l.spinor {
        t[usize::from(c)] = i128::from(f[usize::from(slot)]) << 32;
    }
    for &(c, a, d, sa, sb) in &l.root {
        let (va, vb) = (dbl[usize::from(a)], dbl[usize::from(d)]);
        let u = if sa { va } else { -va };
        let v = if sb { vb } else { -vb };
        t[usize::from(c)] = i128::from(u + v) << 32;
    }
    for &(c, i, neg) in &l.o8 {
        let p = i128::from(SQRT8_Q32) * i128::from(bb[usize::from(i)]);
        t[usize::from(c)] = if neg { -p } else { p };
    }
    for &(c, w) in &l.dense {
        let mut p = 0_i128;
        for m in 0..BLOCK_USIZE {
            p += i128::from(w[m]) * i128::from(bb[m]);
        }
        t[usize::from(c)] = p;
    }
}

/// The code-level entry, so a caller that already holds the `Q8Act` activation (a prefill block, an
/// in-engine tap) pays the quantisation once. Per-call cost in the module header.
///
/// # Errors
///
/// `None` on a shape disagreement, a non-power-of-two scale, an activation-exponent spread past 20,
/// or a product that leaves `i64`.
#[must_use]
pub(crate) fn matvec_fold_quant(face: &FoldFace, q: &QuantAct) -> Option<Vec<i64>> {
    let (b, emins) = activation_words(face, q)?;
    if face.r_k.len() != (face.o * (face.n / GROUP)).div_ceil(2) {
        return None;
    }
    gather_product(face.o, face.n, &face.idx, &b, &emins, |i, gg| {
        face.r_word(i, gg)
    })
}

/// The block-table product on the activation words `b`: the group gathers, the per-`(row, group)`
/// epilogue and the role's own `round_shift_32`, with the row grid supplied per `(row, group)` so the
/// resident face decodes it and the test oracle reads its word. `O(o·n/8)` `i128` adds and `o·g`
/// divisions; see the module's per-call cost.
///
/// # Errors
///
/// `None` on a shape disagreement, a row level that leaves the grid's word range, or a sum that
/// leaves `i64`.
fn gather_product(
    o: usize,
    n: usize,
    idx: &[u8],
    b: &[i64],
    emins: &[u32],
    r_word: impl Fn(usize, usize) -> Option<i64>,
) -> Option<Vec<i64>> {
    let (nb, g) = (n / BLOCK_USIZE, n / GROUP);
    if b.len() != n || idx.len() != o * nb || emins.len() != g {
        return None;
    }
    let l = lut();
    let mut t = [0_i128; CODE_LEN];
    let mut p = vec![0_i128; o];
    let mut acc = vec![0_i128; o];
    let per_group = GROUP / BLOCK_USIZE;
    for gg in 0..g {
        p.fill(0);
        for kk in 0..per_group {
            let k = gg * per_group + kk;
            build_tile(l, b, k * BLOCK_USIZE, &mut t);
            for i in 0..o {
                p[i] += t[usize::from(idx[i * nb + k])];
            }
        }
        let scale = 1_i128 << emins[gg];
        for i in 0..o {
            let val = round_pow2(i128::from(r_word(i, gg)?) * p[i], 64)?;
            acc[i] = acc[i].checked_add(i128::from(val).checked_mul(scale)?)?;
        }
    }
    // The role's epilogue, `exact_dot`'s own body (`src/engine/kern/gemm.rs:461`): one
    // `round_shift_32` of the `i128` accumulator, after every group's exponent is restored and
    // before any `i64` narrowing, so the word written is the `Q32.32` output the residual stream
    // holds.
    let mut out = vec![0_i64; o];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = i64::try_from(crate::engine::num::value::round_shift_32(acc[i])).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod fixture {
    //! The seeded operand both the oracle test and the timing campaign read: a `FoldPlane` whose
    //! stored column fields decode to non-zero words, so `max|b| > 0` and the product is not a zero.

    use crate::engine::chart::QuantAct;
    use crate::engine::kern::fold::{matvec_fold_quant, FoldFace};
    use crate::format::packing::FoldPlane;
    use crate::learn::fold::fit::steps::GROUP;

    /// Codes per codebook block.
    pub(super) const BLOCK: usize = 8;

    /// A seeded `o × n` plane; `seed` shifts the codebook indices, `rows` restricts the written rows.
    /// Column fields `c ∈ [2^-10, 2^-8]`, `d ∈ [2^-8, 2^-7]` (`f16` endpoints, 6- and 5-bit codes),
    /// magnitudes at the production step, and `r` grid `[0.25, 0.75]` stepped through its 16 levels,
    /// so every stored word is non-zero and the product's words stay inside `i64`.
    pub(super) fn plane(o: usize, n: usize, seed: usize, rows: Option<Vec<u32>>) -> FoldPlane {
        let nb = n / BLOCK;
        let g = n / GROUP;
        FoldPlane {
            o,
            n,
            rows,
            idx: (0..o * nb)
                .map(|b| u8::try_from((b * 37 + seed) % 256).unwrap())
                .collect(),
            lo: vec![(1_i64 << 32) / 4; o],
            sp: vec![(1_i64 << 32) / 2; o],
            r_k: (0..o * g)
                .map(|k| u8::try_from((k * 3 + seed) % 16).unwrap())
                .collect(),
            c_head: [0x1400, 0x1C00],
            c_q: (0..n)
                .map(|j| u8::try_from((j * 7 + seed) % 64).unwrap())
                .collect(),
            d_head: [0x1C00, 0x2000],
            d_q: (0..n)
                .map(|j| u8::try_from((j * 5 + seed) % 32).unwrap())
                .collect(),
        }
    }

    /// The seeded plane as the product's face.
    pub(super) fn face(o: usize, n: usize, seed: usize, rows: Option<Vec<u32>>) -> FoldFace {
        FoldFace::from_plane(&plane(o, n, seed, rows)).expect("the seeded plane is a face")
    }

    /// A signed pseudo-random activation of `n` words, `|x| < 2^28`.
    pub(super) fn activation(n: usize, seed: usize) -> Vec<i64> {
        (0..n)
            .map(|j| {
                let v = (j.wrapping_mul(2_654_435_761).wrapping_add(seed * 7919)) % (1 << 29);
                i64::try_from(v).unwrap() - (1 << 28)
            })
            .collect()
    }

    /// The `Q8Act` quantisation of [`activation`].
    pub(super) fn quantised(n: usize, seed: usize) -> QuantAct {
        crate::engine::chart::ffn::quant_act_pooled(&activation(n, seed)).expect("quantise")
    }

    /// The product's output on the seeded operand, with the activation words' maximum magnitude.
    pub(super) fn product(face: &FoldFace, q: &QuantAct) -> (Vec<i64>, i64) {
        let out = matvec_fold_quant(face, q).expect("fold product");
        let words = super::activation_words(face, q).expect("words").0;
        (out, words.iter().map(|v| v.abs()).max().unwrap_or(0))
    }
}

#[cfg(test)]
mod oracle {
    //! The per-`(row, column)` codebook dot this file's product executed before the block table, kept
    //! as the oracle the table must match word for word, and the fixture it is read on.

    use super::fixture::{face, plane, product, quantised};
    use super::{
        activation_words, gather_product, lut, matvec_fold_quant, round_pow2, transform_words,
        BLOCK_USIZE, CODE_LEN, GROUP,
    };
    use crate::engine::chart::QuantAct;
    use crate::engine::kern::fold::FoldFace;
    use crate::format::packing::FoldPlane;
    use crate::learn::fold::fit::icm::codebook;
    use crate::learn::fold::run::transfer::fold_plane_coded;

    /// The previous scalar body: one `i128` codebook multiply-add per `(row, column)`, the same
    /// activation words, the same epilogue. The epilogue is shared with the block table by
    /// construction, so what this oracle pins is the table's gather sum against the per-column
    /// product; the epilogue itself is pinned against the `Q8Act` dense route and the `F16` image
    /// by `tests/engine_fold_pack.rs`.
    fn per_column(face: &FoldFace, q: &QuantAct) -> Option<Vec<i64>> {
        let (b, emins) = activation_words(face, q)?;
        let (nb, g, o) = (face.n / BLOCK_USIZE, face.n / GROUP, face.o);
        let cb = codebook();
        let mut out = vec![0_i64; o];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut acc = 0_i128;
            for gg in 0..g {
                let mut p: i128 = 0;
                for j in 0..GROUP {
                    let at = gg * GROUP + j;
                    let code =
                        cb[usize::from(face.idx[i * nb + at / BLOCK_USIZE])][at % BLOCK_USIZE];
                    p += i128::from(code) * i128::from(b[at]);
                }
                let val = round_pow2(i128::from(face.r_word(i, gg)?) * p, 64)?;
                acc = acc.checked_add(i128::from(val) << emins[gg])?;
            }
            *slot = i64::try_from(crate::engine::num::value::round_shift_32(acc)).ok()?;
        }
        Some(out)
    }

    /// The block table is populated for every codebook row by the three structural forms alone, so a
    /// row the classification missed would read a stale entry.
    #[test]
    fn every_codebook_row_is_a_structural_form() {
        let l = lut();
        assert_eq!(
            (l.spinor.len(), l.root.len(), l.o8.len(), l.dense.len()),
            (128, 112, 16, 0)
        );
    }

    /// The table entries equal the exact per-column products they replace, on a seeded block.
    #[test]
    fn table_entries_are_the_exact_block_products() {
        let f = face(8, 128, 3, None);
        let q = quantised(128, 5);
        let (b, _) = activation_words(&f, &q).expect("words");
        let cb = codebook();
        let l = lut();
        let mut t = [0_i128; CODE_LEN];
        for k in 0..(f.n / BLOCK_USIZE) {
            super::build_tile(l, &b, k * BLOCK_USIZE, &mut t);
            for (c, row) in cb.iter().enumerate() {
                let mut exact = 0_i128;
                for m in 0..BLOCK_USIZE {
                    exact += i128::from(row[m]) * i128::from(b[k * BLOCK_USIZE + m]);
                }
                assert_eq!(t[c], exact, "row {c} of block {k}");
            }
        }
    }

    /// The block-table face equals the previous scalar body word for word, on three seeded `896 × 896`
    /// planes and one row-restricted `896 × 4864` plane, and the operand is not a zero product.
    #[test]
    fn lut_face_bit_identical() {
        let shapes = [
            (896_usize, 896_usize, None),
            (896, 896, None),
            (
                896,
                896,
                Some((0..896).map(|i| u32::try_from(i).unwrap()).collect()),
            ),
            (
                896,
                4864,
                Some((0..896).map(|i| u32::try_from(i * 2).unwrap()).collect()),
            ),
        ];
        for (seed, (o, n, rows)) in shapes.into_iter().enumerate() {
            let f = face(o, n, seed * 11 + 3, rows);
            let q = quantised(n, seed + 1);
            let (out, bmax) = product(&f, &q);
            let old = per_column(&f, &q).expect("oracle");
            assert!(bmax > 0, "{o}x{n}: max|b| = {bmax}");
            let omax = out.iter().map(|v| v.abs()).max().unwrap_or(0);
            assert!(omax > 0, "{o}x{n}: max|out| = {omax}");
            assert_eq!(out.len(), o);
            assert_eq!(
                out, old,
                "{o}x{n} seed {seed}: block table differs from the oracle"
            );
        }
    }

    /// The face `FoldFace::from_plane` built before the coded fields became resident: `r_q`, `c_q`
    /// and `d_q` decoded once at load (`fold_plane_coded(...).qvals()`). The oracle the per-call
    /// decode must reproduce word for word.
    struct Decoded {
        /// Output rows.
        o: usize,
        /// Input columns.
        n: usize,
        /// Codebook row per 8-column block.
        idx: Vec<u8>,
        /// Snapped row grid, `o·n/128` words.
        r_q: Vec<i64>,
        /// Column field, `n` words.
        c_q: Vec<i64>,
        /// Hadamard column field, `n` words.
        d_q: Vec<i64>,
    }

    impl Decoded {
        /// The pre-resident face of a stored plane, on the same refusals.
        fn of(f: &FoldPlane) -> Option<Self> {
            let c = fold_plane_coded(f).ok()?;
            if c.s_fix.is_some() || c.o == 0 || !c.n.is_multiple_of(GROUP) {
                return None;
            }
            let (r_q, c_q, d_q) = c.qvals();
            if r_q.len() != c.o * (c.n / GROUP) || c_q.len() != c.n || d_q.len() != c.n {
                return None;
            }
            Some(Self {
                o: c.o,
                n: c.n,
                idx: c.idx,
                r_q,
                c_q,
                d_q,
            })
        }

        /// The block-table product over the decoded words: the gather body with the row grid read
        /// from the resident `r_q` instead of a level.
        fn product(&self, q: &QuantAct) -> Option<Vec<i64>> {
            let (b, emins) = transform_words(self.n, &self.c_q, &self.d_q, q)?;
            let g = self.n / GROUP;
            gather_product(self.o, self.n, &self.idx, &b, &emins, |i, gg| {
                Some(self.r_q[i * g + gg])
            })
        }
    }

    /// The coded face equals the decoded face it replaces: every `r` word, both column fields word by
    /// word, and the product on a seeded activation. Three seeded `896 × 896` planes and one
    /// row-restricted `896 × 4864` plane.
    #[test]
    fn coded_face_bit_identical() {
        let shapes = [
            (896_usize, 896_usize, None),
            (896, 896, None),
            (
                896,
                896,
                Some((0..896).map(|i| u32::try_from(i).unwrap()).collect()),
            ),
            (
                896,
                4864,
                Some((0..896).map(|i| u32::try_from(i * 2).unwrap()).collect()),
            ),
        ];
        for (seed, (o, n, rows)) in shapes.into_iter().enumerate() {
            let p = plane(o, n, seed * 13 + 3, rows);
            let coded = FoldFace::from_plane(&p).expect("coded face");
            let decoded = Decoded::of(&p).expect("decoded oracle");
            let g = n / GROUP;
            for i in 0..o {
                for gg in 0..g {
                    assert_eq!(
                        coded.r_word(i, gg),
                        Some(decoded.r_q[i * g + gg]),
                        "{o}x{n}: r word at ({i}, {gg})"
                    );
                }
            }
            let (c_w, d_w) = coded.column_fields().expect("column words");
            assert_eq!(c_w, decoded.c_q, "{o}x{n}: c field");
            assert_eq!(d_w, decoded.d_q, "{o}x{n}: d field");

            let q = quantised(n, seed + 4);
            let got = matvec_fold_quant(&coded, &q).expect("coded product");
            let want = decoded.product(&q).expect("oracle product");
            assert!(
                want.iter().any(|&v| v != 0),
                "{o}x{n} seed {seed}: the oracle product is not a zero"
            );
            assert_eq!(got, want, "{o}x{n} seed {seed}: coded face differs");
        }
    }

    /// Control: one flipped codebook index and one moved row level each change the product, so the
    /// equality above is not the equality of two products that ignore their fields.
    #[test]
    fn coded_field_flips_change_output() {
        let (o, n) = (896_usize, 896_usize);
        let p = plane(o, n, 11, None);
        let q = quantised(n, 4);
        let base =
            matvec_fold_quant(&FoldFace::from_plane(&p).expect("face"), &q).expect("product");

        let mut flipped_idx = p.clone();
        let at = 5 * (n / BLOCK_USIZE) + 9;
        flipped_idx.idx[at] ^= 0b1000_0000;
        let idx_out = matvec_fold_quant(&FoldFace::from_plane(&flipped_idx).expect("face"), &q)
            .expect("product");
        assert_ne!(
            base, idx_out,
            "a flipped codebook index must move the output"
        );

        let mut moved_level = p.clone();
        let level = 3 * (n / GROUP) + 2;
        moved_level.r_k[level] = (moved_level.r_k[level] + 1) % 16;
        let level_out = matvec_fold_quant(&FoldFace::from_plane(&moved_level).expect("face"), &q)
            .expect("product");
        assert_ne!(base, level_out, "a moved row level must move the output");
    }

    /// Control: flipping one codebook index bit changes the output, so the equality above is not the
    /// equality of two zero products.
    #[test]
    fn index_flip_changes_output() {
        let f = face(896, 896, 7, None);
        let q = quantised(896, 2);
        let out = matvec_fold_quant(&f, &q).expect("fold product");
        let mut p = plane(896, 896, 7, None);
        let at = 5 * (896 / BLOCK_USIZE) + 9;
        p.idx[at] ^= 0b1000_0000;
        let g = FoldFace::from_plane(&p).expect("face");
        let flipped = matvec_fold_quant(&g, &q).expect("fold product");
        assert_ne!(out[5], flipped[5]);
        assert!(out.iter().zip(&flipped).any(|(a, b)| a != b));
    }
}

#[cfg(test)]
mod timing {
    //! The fold face's per-call cost beside the role's `Q8_0` face at the production shapes, on the
    //! same operator and the same `Q8Act` activation. Run in release:
    //! `cargo test --release --lib kern::fold -- --ignored --nocapture`.
    //!
    //! `gemm_q8_0` is `pub(crate)`, so the comparison lives inside this module. The cold leg is the
    //! kernel's first dispatch after its pool has parked (`SPIN_BEFORE_PARK` 5 ms); the hot leg
    //! follows another dispatch. Five alternating rounds, mean and observed spread.

    use super::fixture::{activation, face, quantised};
    use super::matvec_fold;
    use crate::engine::kern::gemm::gemm_q8_0;
    use crate::format::quant::GGML_TYPE_Q8_0;

    /// A well-formed `Q8_0` image of the shape: one `f16(1)` scale and 32 codes of 8 per block.
    fn q8_image(o: usize, n: usize) -> Vec<u8> {
        let blocks = n / 32;
        let mut raw = Vec::with_capacity(o * blocks * 34);
        for _ in 0..o * blocks {
            raw.extend_from_slice(&0x3C00_u16.to_le_bytes());
            raw.extend(std::iter::repeat_n(8_u8, 32));
        }
        raw
    }

    /// Mean and observed `[min..max]` of a set of microsecond samples.
    #[allow(
        clippy::cast_precision_loss,
        reason = "`v.len() as f64` is exact: the sample set is a five-round campaign, far below 2^52"
    )]
    fn stats(v: &[f64]) -> (f64, f64, f64) {
        let mean = v.iter().sum::<f64>() / (v.len() as f64);
        (
            mean,
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }

    /// Five alternating rounds of the kernel's cold leg, its hot leg and the fold face at one shape.
    fn campaign(o: usize, n: usize) {
        let f = face(o, n, 1, None);
        let x = activation(n, 3);
        let q = quantised(n, 4);
        let q1 = std::slice::from_ref(&q);
        let raw = q8_image(o, n);
        let (mut cold, mut hot, mut fold) = (Vec::new(), Vec::new(), Vec::new());
        let mut omax = 0_i64;
        for round in 0..5 {
            // Park the pool: the cold leg is a first dispatch, not a follow-on one.
            std::thread::sleep(std::time::Duration::from_millis(20));
            let t0 = std::time::Instant::now();
            let a = gemm_q8_0(GGML_TYPE_Q8_0, &raw, o, n, q1).expect("q8_0 cold");
            let t1 = std::time::Instant::now();
            let b = gemm_q8_0(GGML_TYPE_Q8_0, &raw, o, n, q1).expect("q8_0 hot");
            let t2 = std::time::Instant::now();
            let out = matvec_fold(&f, &x).expect("fold product");
            let t3 = std::time::Instant::now();
            assert_eq!((a.len(), b.len(), a[0].len(), b[0].len()), (1, 1, o, o));
            assert_eq!(out.len(), o);
            omax = omax.max(out.iter().map(|v| v.abs()).max().unwrap_or(0));
            cold.push(t1.duration_since(t0).as_secs_f64() * 1e6);
            hot.push(t2.duration_since(t1).as_secs_f64() * 1e6);
            fold.push(t3.duration_since(t2).as_secs_f64() * 1e6);
            eprintln!(
                "round {round}: cold {:.1} hot {:.1} fold {:.1}",
                cold[round], hot[round], fold[round]
            );
        }
        let (cm, cmin, cmax) = stats(&cold);
        let (hm, hmin, hmax) = stats(&hot);
        let (fm, fmin, fmax) = stats(&fold);
        eprintln!(
            "{o}x{n} x1 column: fold face {fm:.1} us [{fmin:.1}..{fmax:.1}], \
             gemm_q8_0 cold {cm:.1} us [{cmin:.1}..{cmax:.1}] (ratio {:.2}x), \
             hot {hm:.1} us [{hmin:.1}..{hmax:.1}] (ratio {:.2}x), max|out| {omax}",
            fm / cm,
            fm / hm
        );
    }

    #[test]
    #[ignore = "a release timing campaign; --release --ignored --nocapture"]
    fn fold_face_against_q8_0_face() {
        campaign(896, 896);
        campaign(896, 4864);
    }
}
