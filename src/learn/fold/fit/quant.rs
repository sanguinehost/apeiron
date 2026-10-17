//! The `f16` round-trip and the two scale carriers of the one-bit fold transfer: the affine column
//! quantizer and the hierarchical row-scale group quantizer, ported from `fold_transfer.py` 511–558
//! onto the `Q32.32` carrier `i64` (a real `x` is the word `x·2³²`): products in `i128`, rounding
//! through [`crate::learn::solve::q32`], the `f64` boundary converted once at entry and once at exit.
//!
//! [`quant_c_u8`] stores two `f16` endpoints plus `bits`/column codes. [`quant_r_hier`] stores a
//! per-row 6-bit mean, a 2-bit spread index into four levels between the 5% deviation quantile and
//! the max deviation, and a 4-bit code per element, choosing the `du ∈ {−1, 0, 1}` mean shift
//! crossed with the four spreads by least summed squared error, ties to the earlier candidate.
//!
//! No pass is pooled: the widest unit is one row's twelve-candidate search at `12·g` word ops
//! (≤ ~10⁵ at a fit layer), under the ~10⁶ a dispatch must amortise, so each is a serial loop.
//!
//! The `1e-12`/`1e-4` f64 floors become [`SPAN_MIN`]/[`DEV_MIN`]; endpoints round once from the
//! exact grid value, ≤ one binary16 ulp from the Python's `f64 → f32 → f16` double rounding.
//! Grade `tested` (`tests/learn_fold_quant.rs`); nothing here is `proved`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::must_use_candidate,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::too_many_lines,
    reason = "the kernels index `[o, g]` by `i/j` and name the carriers `r_lo`, `r_sp`, `rq`, \
              `u6`, `sp_idx` as the reference does; the `i128 → i64`/`i128 → u8` narrowings follow \
              a clamp or a construction bound (f16 endpoint < 2⁴⁹, grid level < 2⁶⁴)"
)]

use std::cmp::Ordering;

use crate::format::quant::f16_to_f32;
use crate::learn::solve::q32::{from_grid, round_f16, rshift_rne, to_grid, Fx, GRID_BOUND};

/// The one-word grid span `2⁻³²`, the integer image of the f64 `1e-12` floor.
const SPAN_MIN: i64 = 1;

/// `round(1e-4·2³²)`, the integer image of the f64 `1e-4` deviation floor.
const DEV_MIN: i64 = 429_497;

/// `2³²`, the Q32.32 scale for the `1 ± sp` products.
const ONE: i128 = 1_i128 << 32;

/// A refusal from a quantization carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantError {
    /// An empty slice (`quant_c_u8`) or a zero row/column count (`quant_r_hier`).
    Empty,
    /// A slice length that is not the declared `o · g` shape.
    ShapeMismatch,
    /// A bit width outside `1..=32` (a zero width divides by `levels = 0`).
    BadBits,
    /// A bit total `bits · n + 32` overflowing `u64`, or a value off the Q32.32 carrier: non-finite,
    /// `|x| ≥ 2³¹`, an f16 endpoint at binary16 infinity, or a squared error leaving `i128`.
    LengthOverflow,
}

/// Round an `f64` to the nearest IEEE-754 binary16 and back: torch `x.half().to(x.dtype)`.
/// Rtne, subnormal underflow, overflow to `±∞`, `±0.0` and quiet NaN preserved; narrows
/// `f64 → f32` (the platform's rtne cast) → `f16`. Time `O(1)`; the quantizers round on the grid.
pub fn f16r(x: f64) -> f64 {
    f64::from(f16_to_f32(f16_bits(x as f32)))
}

/// `f32` → nearest-even binary16 bits; a NaN maps to the canonical quiet `0x7E00`. Time `O(1)`.
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32;
    let m = b & 0x007F_FFFF;
    if e == 0xFF {
        return sign | if m == 0 { 0x7C00 } else { 0x7E00 };
    }
    let he = e - 112;
    if he >= 31 {
        return sign | 0x7C00;
    }
    if he <= 0 {
        // Subnormal: scale by 2²⁴ so half a subnormal step is half a unit, round to nearest-even,
        // then reinterpret the count as the `f16` code.
        let r = (f32::from_bits(b & 0x7FFF_FFFF) * 16_777_216.0).round_ties_even() as u32;
        return sign | r as u16;
    }
    let half = ((he as u32) << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    sign | (half + u32::from(rem > 0x1000 || (rem == 0x1000 && half & 1 == 1))) as u16
}

/// `num / den` rounded to nearest, ties to even, for `den > 0`: the truncated quotient plus
/// `sign(r)` when `2|r| > den`, or when `2|r| = den` and the quotient is odd. Time `O(1)`.
fn div_rne(num: i128, den: i128) -> i128 {
    let (q, r) = (num / den, num % den);
    q + match (2 * r.abs()).cmp(&den) {
        Ordering::Greater => r.signum(),
        Ordering::Equal if q % 2 != 0 => r.signum(),
        _ => 0,
    }
}

/// The `i128` value as a Q32.32 word, `LengthOverflow` when it leaves `i64`. Time `O(1)`.
fn to_word(v: i128) -> Result<i64, QuantError> {
    i64::try_from(v).map_err(|_| QuantError::LengthOverflow)
}

/// The `f64` slice as Q32.32 words. Time `O(n)` word conversions, space `O(n)`; the boundary map.
fn to_words(values: &[f64]) -> Result<Vec<i64>, QuantError> {
    values
        .iter()
        .map(|&x| to_grid(x, GRID_BOUND).ok_or(QuantError::LengthOverflow))
        .collect()
}

/// The real value of an exact integer carrier `m·2^e` as a Q32.32 word. Time `O(1)`.
fn fx_word(x: &Fx) -> Result<i64, QuantError> {
    let shift = i64::from(x.e) + 32;
    let s = u32::try_from(shift.unsigned_abs()).map_err(|_| QuantError::LengthOverflow)?;
    let v = if shift >= 0 {
        i128::from(x.m) << s
    } else {
        rshift_rne(i128::from(x.m), s).ok_or(QuantError::LengthOverflow)?
    };
    to_word(v)
}

/// The binary16 round-to-nearest-even of the real `w/2³²` as a Q32.32 word; `LengthOverflow` at
/// the binary16 infinity `2¹⁶`, which the grid cannot carry. Time `O(1)`.
fn f16_word(w: i64) -> Result<i64, QuantError> {
    let r = round_f16(Fx { m: w, e: -32 }).map_err(|_| QuantError::LengthOverflow)?;
    fx_word(&r)
}

/// The minimum and maximum of `words` (non-empty). Time `O(n)` comparisons, space `O(1)`.
fn min_max(words: &[i64]) -> (i64, i64) {
    let mut lo = words[0];
    let mut hi = lo;
    for &w in &words[1..] {
        lo = lo.min(w);
        hi = hi.max(w);
    }
    (lo, hi)
}

/// The stored field as a Q32.32 word: this module writes exact grid words, a caller-set non-finite
/// or out-of-grid value saturates instead of panicking. Time `O(1)`.
fn field_word(x: f64) -> i64 {
    let sat = if x.is_nan() || x < 0.0 {
        i64::MIN
    } else {
        i64::MAX
    };
    to_grid(x, GRID_BOUND).unwrap_or(sat)
}

/// The grid value of code `k` on `(lo, sp)`: `lo + (k / 15)·sp`. Time `O(1)`.
fn decode_word(lo: i64, sp: i64, k: u8) -> i128 {
    i128::from(lo) + div_rne(i128::from(k) * i128::from(sp), 15)
}

/// [`decode_word`] on an `f64` row grid `(lo, sp)` that is already on the Q32.32 grid: the level that
/// [`quant_r_hier`] stores for code `k`; `None` off the grid bound. Time `O(1)`.
pub fn r_level(lo: f64, sp: f64, k: u8) -> Option<f64> {
    let w = decode_word(to_grid(lo, GRID_BOUND)?, to_grid(sp, GRID_BOUND)?, k);
    Some(from_grid(i64::try_from(w).ok()?))
}

/// The 4-bit code of `x` on `(lo, sp)`, clamped to `0..=15`. Time `O(1)`.
fn code(x: i128, lo: i64, sp: i64) -> i128 {
    div_rne((x - i128::from(lo)) * 15, i128::from(sp)).clamp(0, 15)
}

/// Affine column-scale quantization: two `f16` scalars (32 bits) plus `bits`/column.
///
/// `c_min = f16(min c)`, `c_max = f16(max c)`, `span = max(c_max − c_min, 2⁻³²)`; each element is
/// rounded to the nearest of `2^bits` grid points on `[c_min, c_max]`; the bit count is `bits·n + 32`.
/// # Errors
/// [`QuantError::Empty`] for an empty slice; [`QuantError::BadBits`] for `bits` outside `1..=32`;
/// [`QuantError::LengthOverflow`] for a bit total overflowing `u64`, or an input or f16 endpoint
/// off the grid.
/// # Complexity
/// Time `O(n)` word ops (one scan, two endpoint rounds, one grid round per column), space `O(n)`.
pub fn quant_c_u8(c: &[f64], bits: u32) -> Result<(Vec<f64>, u64), QuantError> {
    if c.is_empty() {
        return Err(QuantError::Empty);
    }
    if bits == 0 || bits > 32 {
        return Err(QuantError::BadBits);
    }
    let total = (c.len() as u64)
        .checked_mul(u64::from(bits))
        .and_then(|t| t.checked_add(32))
        .ok_or(QuantError::LengthOverflow)?;

    let words = to_words(c)?;
    let (lo, hi) = min_max(&words);
    let c_min = f16_word(lo)?;
    let c_max = f16_word(hi)?;
    let span = to_word(i128::from(c_max) - i128::from(c_min))?.max(SPAN_MIN);
    let levels = ((1_u64 << bits) - 1) as i64;

    let mut out = vec![0.0_f64; words.len()];
    for (slot, &w) in out.iter_mut().zip(&words) {
        let x = i128::from(w) - i128::from(c_min);
        let q = div_rne(x * i128::from(levels), i128::from(span)).clamp(0, i128::from(levels));
        let off = div_rne(q * i128::from(span), i128::from(levels));
        // Bounded by the f16 endpoint range (`|word| ≤ 65504·2³²`), so the narrowing is exact.
        *slot = from_grid((i128::from(c_min) + off) as i64);
    }
    Ok((out, total))
}

/// A hierarchical row-scale quantization result: the codes and the values they decode to,
/// `rq[i·g + j] = r_lo[i] + (k[i·g + j] / 15) · r_sp[i]`.
#[derive(Clone, Debug)]
pub struct RHier {
    /// Reconstructed values, row-major `[o, g]`.
    pub rq: Vec<f64>,
    /// The bit budget `4·o·g + 8·o + 64`.
    pub bits: u64,
    /// Per-row lower endpoint `mu·(1 − sp)`.
    pub r_lo: Vec<f64>,
    /// Per-row grid width `max(mu·(1 + sp) − r_lo, 2⁻³²)`.
    pub r_sp: Vec<f64>,
    /// Per-element 4-bit codes in `0..=15`, row-major `[o, g]`.
    pub k: Vec<u8>,
    /// Per-row 6-bit mean code in `0..=63`, the chosen `du` shift applied.
    pub u6: Vec<u8>,
    /// Per-row spread level index in `0..=3`.
    pub sp_idx: Vec<u8>,
}

impl RHier {
    /// Decode `rq` from the stored per-row grid and 4-bit codes.
    /// # Complexity
    /// Time `O(o·g)` word ops, space `O(o·g)`; one pass over the `k` codes (`g = k.len() / o`).
    pub fn reconstruct(&self) -> Vec<f64> {
        let o = self.u6.len();
        if o == 0 {
            return Vec::new();
        }
        let g = self.k.len() / o;
        let mut out = vec![0.0_f64; self.k.len()];
        if g == 0 {
            return out;
        }
        let lo: Vec<i64> = self.r_lo.iter().map(|&x| field_word(x)).collect();
        let sp: Vec<i64> = self.r_sp.iter().map(|&x| field_word(x)).collect();
        for (idx, slot) in out.iter_mut().enumerate() {
            let row = idx / g;
            let w = decode_word(lo[row], sp[row], self.k[idx]);
            *slot = from_grid(i64::try_from(w).unwrap_or(if w < 0 { i64::MIN } else { i64::MAX }));
        }
        out
    }
}

/// The per-row-mean grid and the four spread levels shared by every candidate of a row.
struct HierHeader {
    /// `f16` of the smallest row mean, a Q32.32 word.
    mu_min: i64,
    /// `max(f16(max mean) − f16(min mean), 2⁻³²)`, a Q32.32 word.
    mu_span: i64,
    u6_base: Vec<u8>,
    /// The four spread levels `linspace(0, 1, 4)` between the deviation quantile and its max.
    sp_levels: [i64; 4],
}

/// The linear-interpolation 5% quantile of `values` on the Q32.32 grid: `idx = (n−1)/20`, then the
/// blend of the two bracketing order statistics with weight `rem/20`. Time `O(n log n)` word
/// comparisons for the sort, space `O(n)`.
fn quantile_05(values: &[i64]) -> i64 {
    let n = values.len();
    if n < 2 {
        return values.first().copied().unwrap_or(0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    // `q = 0.05 = 1/20`, so `idx·20 = n−1` exactly; the integer split avoids an f64 quotient.
    let idx = n as i128 - 1;
    let Ok(lo_i) = usize::try_from(idx / 20) else {
        return sorted[n - 1];
    };
    if lo_i >= n - 1 {
        return sorted[n - 1];
    }
    let rem = idx % 20;
    let diff = i128::from(sorted[lo_i + 1]) - i128::from(sorted[lo_i]);
    // The blend stays between the two order statistics, so the narrowing is exact.
    (i128::from(sorted[lo_i]) + div_rne(diff * rem, 20)) as i64
}

/// Build the row-mean grid and spread levels for [`quant_r_hier`].
/// # Complexity
/// Time `O(o·g + o log o)` word ops (three passes plus the sort), space `O(o)`.
fn hier_header(r: &[i64], o: usize, g: usize) -> Result<HierHeader, QuantError> {
    let rm: Vec<i64> = r
        .chunks_exact(g)
        .map(|row| div_rne(row.iter().map(|&x| i128::from(x)).sum(), g as i128) as i64)
        .collect();

    let (mu_lo, mu_hi) = min_max(&rm);
    let mu_min = f16_word(mu_lo)?;
    let mu_max = f16_word(mu_hi)?;
    let mu_span = to_word(i128::from(mu_max) - i128::from(mu_min))?.max(SPAN_MIN);

    let mut u6_base = vec![0_u8; o];
    let mut dev = vec![0_i64; o];
    for (i, (u, slot)) in u6_base.iter_mut().zip(dev.iter_mut()).enumerate() {
        let mean = i128::from(rm[i]);
        *u = div_rne((mean - i128::from(mu_min)) * 63, i128::from(mu_span)).clamp(0, 63) as u8;
        let mut mx: i128 = 0;
        for j in 0..g {
            mx = mx.max((i128::from(r[i * g + j]) - mean).abs());
        }
        let den = if mean.abs() > 1 { mean.abs() } else { 1 };
        *slot = to_word(div_rne(mx << 32, den))?;
    }

    let sp_min = f16_word(quantile_05(&dev).max(DEV_MIN))?;
    let sp_max = f16_word(min_max(&dev).1.max(sp_min.saturating_add(DEV_MIN)))?;
    let span = i128::from(sp_max) - i128::from(sp_min);
    let mut sp_levels = [sp_min; 4];
    for (s, level) in sp_levels.iter_mut().enumerate().skip(1) {
        *level = to_word(i128::from(sp_min) + div_rne(span * s as i128, 3))?;
    }

    Ok(HierHeader {
        mu_min,
        mu_span,
        u6_base,
        sp_levels,
    })
}

/// The winning candidate of one row: its summed squared error, its grid and its codes.
#[derive(Clone, Copy)]
struct RowBest {
    err: i128,
    lo: i64,
    sp: i64,
    u6: u8,
    sp_idx: u8,
}

/// The `(r_lo, r_sp)` grid of mean word `mu` at spread word `sp`: `mu·(1 ∓ sp)`, span floored at
/// one quantum. Time `O(1)`.
fn grid(mu: i64, sp: i64) -> Result<(i64, i64), QuantError> {
    let (m, s) = (i128::from(mu), i128::from(sp));
    let lo = to_word(rshift_rne(m * (ONE - s), 32).ok_or(QuantError::LengthOverflow)?)?;
    let hi = to_word(rshift_rne(m * (ONE + s), 32).ok_or(QuantError::LengthOverflow)?)?;
    Ok((lo, to_word(i128::from(hi) - i128::from(lo))?.max(SPAN_MIN)))
}

/// The summed squared error of row `i` on the grid `(lo, sp)`, Q32.32 with an `i128` accumulator;
/// refused when a square or the sum leaves `i128`. Time `O(g)` word ops, space `O(1)`.
fn row_err(r: &[i64], g: usize, i: usize, lo: i64, sp: i64) -> Result<i128, QuantError> {
    let mut err: i128 = 0;
    for j in 0..g {
        let x = i128::from(r[i * g + j]);
        let d = x - (i128::from(lo) + div_rne(code(x, lo, sp) * i128::from(sp), 15));
        err = err
            .checked_add(d.checked_mul(d).ok_or(QuantError::LengthOverflow)?)
            .ok_or(QuantError::LengthOverflow)?;
    }
    Ok(err)
}

/// Score the twelve candidates of row `i` (`du ∈ {−1, 0, 1}` × four spreads) and keep the least
/// summed squared error, ties to the earlier candidate. Time `O(12·g)` word ops, space `O(1)`.
fn row_best(r: &[i64], g: usize, i: usize, h: &HierHeader) -> Result<RowBest, QuantError> {
    let mut best = RowBest {
        err: i128::MAX,
        lo: 0,
        sp: SPAN_MIN,
        u6: 0,
        sp_idx: 0,
    };
    for &du in &[-1_i32, 0, 1] {
        let u6 = (i32::from(h.u6_base[i]) + du).clamp(0, 63);
        let mu =
            to_word(i128::from(h.mu_min) + div_rne(i128::from(u6) * i128::from(h.mu_span), 63))?;
        for (sp_idx, &sp) in h.sp_levels.iter().enumerate() {
            let (lo, span) = grid(mu, sp)?;
            let err = row_err(r, g, i, lo, span)?;
            if err < best.err {
                best = RowBest {
                    err,
                    lo,
                    sp: span,
                    u6: u6 as u8,
                    sp_idx: sp_idx as u8,
                };
            }
        }
    }
    Ok(best)
}

/// Hierarchical group-scale quantization: a per-row 6-bit mean, a 2-bit spread and 4-bit codes.
///
/// Port of `quant_r_hier(r)`. The row mean `rm` sits on the 6-bit grid `[mu_min, mu_max]` (`f16` of
/// the mean extremes); `dev = max_j |r − rm| / max(|rm|, 2⁻³²)` sets the spread range from
/// `f16(quantile(dev, 0.05) ≥ 1e-4)` to `f16(max(dev.max(), sp_min + 1e-4))`, discretized to four
/// levels by torch `linspace(0, 1, 4)`. Each candidate quantizes the row to
/// `k = clamp(round((r − r_lo)/r_sp · 15), 0, 15)` on `[r_lo, r_hi] = mu·(1 ∓ sp)`; the least summed
/// squared error wins, ties to the earlier candidate. The bit count is `4·o·g + 8·o + 64`.
/// # Errors
/// [`QuantError::Empty`] for `o == 0` or `g == 0`; [`QuantError::ShapeMismatch`] for a length that
/// is not `o·g` (or can't be, on overflow); [`QuantError::LengthOverflow`] for an input or f16
/// endpoint off the grid, or a squared error leaving `i128`.
/// # Complexity
/// Time `O(o·g)` word ops (`12·g` per row for the candidates, then one `g` pass for the codes and
/// one for the decode) plus `O(o log o)` for the quantile; space `O(o·g)`.
pub fn quant_r_hier(r: &[f64], o: usize, g: usize) -> Result<RHier, QuantError> {
    if o == 0 || g == 0 {
        return Err(QuantError::Empty);
    }
    let n = o.checked_mul(g).ok_or(QuantError::ShapeMismatch)?;
    if r.len() != n {
        return Err(QuantError::ShapeMismatch);
    }

    let words = to_words(r)?;
    let header = hier_header(&words, o, g)?;
    let best = (0..o)
        .map(|i| row_best(&words, g, i, &header))
        .collect::<Result<Vec<_>, _>>()?;

    let mut best_k = vec![0_u8; n];
    for (idx, slot) in best_k.iter_mut().enumerate() {
        let b = &best[idx / g];
        *slot = code(i128::from(words[idx]), b.lo, b.sp) as u8;
    }

    let mut rq = vec![0.0_f64; n];
    for (idx, slot) in rq.iter_mut().enumerate() {
        let b = &best[idx / g];
        // Bounded by the f16 endpoint range, so the narrowing is exact.
        *slot = from_grid(decode_word(b.lo, b.sp, best_k[idx]) as i64);
    }

    Ok(RHier {
        rq,
        bits: 4 * n as u64 + 8 * o as u64 + 64,
        r_lo: best.iter().map(|b| from_grid(b.lo)).collect(),
        r_sp: best.iter().map(|b| from_grid(b.sp)).collect(),
        k: best_k,
        u6: best.iter().map(|b| b.u6).collect(),
        sp_idx: best.iter().map(|b| b.sp_idx).collect(),
    })
}
