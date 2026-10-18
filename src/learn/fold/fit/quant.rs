//! The `f16` round-trip and the two scale carriers of the one-bit fold transfer: the affine column
//! quantizer and the hierarchical row-scale group quantizer, ported from `fold_transfer.py` 511–558
//! onto the `Q32.32` carrier `i64` (a real `x` is the word `x·2³²`): products in `i128`, rounding
//! through [`crate::learn::solve::q32`], every input and output a word — no `f64` crosses the API.
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
//! exact grid value by [`round_f16`], the Python's `f64 → f32 → f16` double rounding replaced by
//! one rounding from the exact grid word.
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

use crate::learn::solve::q32::{div_rne_pos, round_f16, rshift_rne, Fx};

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
    /// A bit total `bits · n + 32` overflowing `u64`, an f16 endpoint at binary16 infinity, or a
    /// squared error leaving `i128`.
    LengthOverflow,
}

/// Round a Q32.32 word to the nearest IEEE-754 binary16 and back: torch `x.half().to(x.dtype)`.
/// Rtne, subnormal underflow and `±0`; `LengthOverflow` at the binary16 infinity `2¹⁶`, which the
/// grid cannot carry. Time `O(1)`; the quantizers round on the grid.
///
/// # Errors
/// [`QuantError::LengthOverflow`] when the round leaves the binary16 finite range.
pub fn f16r(w: i64) -> Result<i64, QuantError> {
    f16_word(w)
}

/// The `i128` value as a Q32.32 word, `LengthOverflow` when it leaves `i64`. Time `O(1)`.
fn to_word(v: i128) -> Result<i64, QuantError> {
    i64::try_from(v).map_err(|_| QuantError::LengthOverflow)
}

/// The real value of an exact integer carrier `m·2^e` as a Q32.32 word. Time `O(1)`.
fn fx_word(x: &Fx) -> Result<i64, QuantError> {
    let shift = i64::from(x.e) + 32;
    let mag = shift.unsigned_abs();
    if mag > 127 {
        return Err(QuantError::LengthOverflow);
    }
    let s = mag as u32;
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
    for i in 1..words.len() {
        let w = words[i];
        lo = lo.min(w);
        hi = hi.max(w);
    }
    (lo, hi)
}

/// The grid value of code `k` on `(lo, sp)`: `lo + (k / 15)·sp`. Time `O(1)`.
fn decode_word(lo: i64, sp: i64, k: u8) -> i128 {
    i128::from(lo) + div_rne_pos(i128::from(i64::from(k)) * i128::from(sp), 15)
}

/// [`decode_word`] on the stored row grid `(lo, sp)`: the level that [`quant_r_hier`] stores for
/// code `k`; `None` when the level leaves the word. Time `O(1)`.
pub fn r_level(lo: i64, sp: i64, k: u8) -> Option<i64> {
    i64::try_from(decode_word(lo, sp, k)).ok()
}

/// The 4-bit code of `x` on `(lo, sp)`, clamped to `0..=15`. Time `O(1)`.
fn code(x: i128, lo: i64, sp: i64) -> i128 {
    div_rne_pos((x - i128::from(lo)) * 15, i128::from(sp)).clamp(0, 15)
}

/// Affine column-scale quantization: two `f16` scalars (32 bits) plus `bits`/column.
///
/// `c_min = f16(min c)`, `c_max = f16(max c)`, `span = max(c_max − c_min, 2⁻³²)`; each element is
/// rounded to the nearest of `2^bits` grid points on `[c_min, c_max]`; the bit count is `bits·n + 32`.
/// `c` is Q32.32 words, one real per entry.
/// # Errors
/// [`QuantError::Empty`] for an empty slice; [`QuantError::BadBits`] for `bits` outside `1..=32`;
/// [`QuantError::LengthOverflow`] for a bit total overflowing `u64`, or an f16 endpoint at binary16
/// infinity.
/// # Complexity
/// Time `O(n)` word ops (one scan, two endpoint rounds, one grid round per column), space `O(n)`.
pub fn quant_c_u8(c: &[i64], bits: u32) -> Result<(Vec<i64>, u64), QuantError> {
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

    let (lo, hi) = min_max(c);
    let c_min = f16_word(lo)?;
    let c_max = f16_word(hi)?;
    let span = to_word(i128::from(c_max) - i128::from(c_min))?.max(SPAN_MIN);
    let levels = ((1_u64 << bits) - 1) as i64;

    let mut out = vec![0_i64; c.len()];
    let mut ok = true;
    for i in 0..c.len() {
        if ok {
            let w = c[i];
            let x = i128::from(w) - i128::from(c_min);
            let q =
                div_rne_pos(x * i128::from(levels), i128::from(span)).clamp(0, i128::from(levels));
            let off = div_rne_pos(q * i128::from(span), i128::from(levels));
            match to_word(i128::from(c_min) + off) {
                Ok(v) => out[i] = v,
                Err(_) => ok = false,
            }
        }
    }
    if ok {
        Ok((out, total))
    } else {
        Err(QuantError::LengthOverflow)
    }
}

/// A hierarchical row-scale quantization result: the codes and the values they decode to,
/// `rq[i·g + j] = r_lo[i] + (k[i·g + j] / 15) · r_sp[i]`.
#[derive(Clone, Debug)]
pub struct RHier {
    /// Reconstructed values, row-major `[o, g]`, Q32.32 words.
    pub rq: Vec<i64>,
    /// The bit budget `4·o·g + 8·o + 64`.
    pub bits: u64,
    /// Per-row lower endpoint `mu·(1 − sp)`.
    pub r_lo: Vec<i64>,
    /// Per-row grid width `max(mu·(1 + sp) − r_lo, 2⁻³²)`.
    pub r_sp: Vec<i64>,
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
    pub fn reconstruct(&self) -> Vec<i64> {
        let o = self.u6.len();
        if o == 0 {
            return Vec::new();
        }
        let g = self.k.len() / o;
        let mut out = vec![0_i64; self.k.len()];
        if g == 0 {
            return out;
        }
        for idx in 0..out.len() {
            let row = idx / g;
            let w = decode_word(self.r_lo[row], self.r_sp[row], self.k[idx]);
            out[idx] = match i64::try_from(w) {
                Ok(v) => v,
                Err(_) => {
                    if w < 0 {
                        i64::MIN
                    } else {
                        i64::MAX
                    }
                }
            };
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

/// Ascending sort of `values` by bottom-up merge: each pass reads `src` and writes `dst`, so no
/// pass reads the buffer it mutates. Time `O(n log n)` comparisons, space `O(n)`.
#[allow(
    clippy::manual_swap,
    reason = "a `mem::swap` call would put an unmodelled core function in the extracted body; the \
              three-line exchange of two whole buffers is the extracted form"
)]
fn sort_ascending(values: &[i64]) -> Vec<i64> {
    let n = values.len();
    let mut src = values.to_vec();
    let mut dst = vec![0_i64; n];
    let mut width = 1usize;
    while width < n {
        let mut lo = 0usize;
        while lo < n {
            let mid = if lo + width < n { lo + width } else { n };
            let hi = if lo + 2 * width < n {
                lo + 2 * width
            } else {
                n
            };
            let (mut i, mut j, mut k) = (lo, mid, lo);
            while k < hi {
                if j >= hi || (i < mid && src[i] <= src[j]) {
                    dst[k] = src[i];
                    i += 1;
                } else {
                    dst[k] = src[j];
                    j += 1;
                }
                k += 1;
            }
            lo += 2 * width;
        }
        let t = src;
        src = dst;
        dst = t;
        width *= 2;
    }
    src
}

/// The linear-interpolation 5% quantile of `values` on the Q32.32 grid: `idx = (n−1)/20`, then the
/// blend of the two bracketing order statistics with weight `rem/20`. Time `O(n log n)` word
/// comparisons for the sort, space `O(n)`.
fn quantile_05(values: &[i64]) -> i64 {
    let n = values.len();
    if n < 2 {
        if n == 0 {
            return 0;
        }
        return values[0];
    }
    let sorted = sort_ascending(values);
    // `q = 0.05 = 1/20`, so `idx·20 = n−1` exactly; the integer split avoids an f64 quotient.
    let idx = n - 1;
    let lo_i = idx / 20;
    if lo_i >= n - 1 {
        return sorted[n - 1];
    }
    let rem = idx % 20;
    let diff = i128::from(sorted[lo_i + 1]) - i128::from(sorted[lo_i]);
    // The blend stays between the two order statistics, so the narrowing is exact.
    (i128::from(sorted[lo_i]) + div_rne_pos(diff * rem as i128, 20)) as i64
}

/// Build the row-mean grid and spread levels for [`quant_r_hier`].
/// # Complexity
/// Time `O(o·g + o log o)` word ops (three passes plus the sort), space `O(o)`.
fn hier_header(r: &[i64], o: usize, g: usize) -> Result<HierHeader, QuantError> {
    let mut rm = vec![0_i64; o];
    for i in 0..o {
        let mut sum: i128 = 0;
        for j in 0..g {
            sum += i128::from(r[i * g + j]);
        }
        rm[i] = div_rne_pos(sum, g as i128) as i64;
    }

    let (mu_lo, mu_hi) = min_max(&rm);
    let mu_min = f16_word(mu_lo)?;
    let mu_max = f16_word(mu_hi)?;
    let mu_span = to_word(i128::from(mu_max) - i128::from(mu_min))?.max(SPAN_MIN);

    let mut u6_base = vec![0_u8; o];
    let mut dev = vec![0_i64; o];
    let mut ok = true;
    for i in 0..o {
        if ok {
            let mean = i128::from(rm[i]);
            u6_base[i] = div_rne_pos((mean - i128::from(mu_min)) * 63, i128::from(mu_span))
                .clamp(0, 63) as u8;
            let mut mx: i128 = 0;
            for j in 0..g {
                mx = mx.max((i128::from(r[i * g + j]) - mean).abs());
            }
            let den = if mean.abs() > 1 { mean.abs() } else { 1 };
            match to_word(div_rne_pos(mx << 32, den)) {
                Ok(v) => dev[i] = v,
                Err(_) => ok = false,
            }
        }
    }
    if !ok {
        return Err(QuantError::LengthOverflow);
    }

    let sp_min = f16_word(quantile_05(&dev).max(DEV_MIN))?;
    let sp_max = f16_word(min_max(&dev).1.max(sp_min.saturating_add(DEV_MIN)))?;
    let sp_min_w = i128::from(sp_min);
    let span = i128::from(sp_max) - sp_min_w;
    let mut sp_levels = [sp_min; 4];
    sp_levels[1] = to_word(sp_min_w + div_rne_pos(span, 3))?;
    sp_levels[2] = to_word(sp_min_w + div_rne_pos(span * 2, 3))?;
    sp_levels[3] = to_word(sp_min_w + div_rne_pos(span * 3, 3))?;

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
    let mut ok = true;
    for j in 0..g {
        if ok {
            let x = i128::from(r[i * g + j]);
            let d = x - (i128::from(lo) + div_rne_pos(code(x, lo, sp) * i128::from(sp), 15));
            match d.checked_mul(d) {
                Some(sq) => match err.checked_add(sq) {
                    Some(e) => err = e,
                    None => ok = false,
                },
                None => ok = false,
            }
        }
    }
    if ok {
        Ok(err)
    } else {
        Err(QuantError::LengthOverflow)
    }
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
    let mut ok = true;
    for t in 0..3 {
        for sp_idx in 0..4 {
            if ok {
                let du = t - 1;
                let u6 = (i32::from(h.u6_base[i]) + du).clamp(0, 63);
                let mu =
                    i128::from(h.mu_min) + div_rne_pos(i128::from(u6) * i128::from(h.mu_span), 63);
                let sp = h.sp_levels[sp_idx];
                match to_word(mu) {
                    Ok(m) => match grid(m, sp) {
                        Ok((lo, span)) => match row_err(r, g, i, lo, span) {
                            Ok(err) => {
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
                            Err(_) => ok = false,
                        },
                        Err(_) => ok = false,
                    },
                    Err(_) => ok = false,
                }
            }
        }
    }
    if ok {
        Ok(best)
    } else {
        Err(QuantError::LengthOverflow)
    }
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
/// is not `o·g` (or can't be, on overflow); [`QuantError::LengthOverflow`] for an f16 endpoint at
/// binary16 infinity, or a squared error leaving the accumulator.
/// # Complexity
/// Time `O(o·g)` word ops (`12·g` per row for the candidates, then one `g` pass for the codes and
/// one for the decode) plus `O(o log o)` for the quantile; space `O(o·g)`.
pub fn quant_r_hier(r: &[i64], o: usize, g: usize) -> Result<RHier, QuantError> {
    if o == 0 || g == 0 {
        return Err(QuantError::Empty);
    }
    let n = o.checked_mul(g).ok_or(QuantError::ShapeMismatch)?;
    if r.len() != n {
        return Err(QuantError::ShapeMismatch);
    }

    let header = hier_header(r, o, g)?;
    let mut best: Vec<RowBest> = Vec::with_capacity(o);
    let mut ok = true;
    for i in 0..o {
        if ok {
            match row_best(r, g, i, &header) {
                Ok(b) => best.push(b),
                Err(_) => ok = false,
            }
        }
    }
    if !ok {
        return Err(QuantError::LengthOverflow);
    }

    let mut best_k = vec![0_u8; n];
    for idx in 0..n {
        let b = &best[idx / g];
        best_k[idx] = code(i128::from(r[idx]), b.lo, b.sp) as u8;
    }

    let mut rq = vec![0_i64; n];
    let mut ok = true;
    for idx in 0..n {
        if ok {
            let b = &best[idx / g];
            match to_word(decode_word(b.lo, b.sp, best_k[idx])) {
                Ok(v) => rq[idx] = v,
                Err(_) => ok = false,
            }
        }
    }
    if !ok {
        return Err(QuantError::LengthOverflow);
    }

    let mut r_lo = vec![0_i64; o];
    let mut r_sp = vec![0_i64; o];
    let mut u6 = vec![0_u8; o];
    let mut sp_idx = vec![0_u8; o];
    for i in 0..o {
        r_lo[i] = best[i].lo;
        r_sp[i] = best[i].sp;
        u6[i] = best[i].u6;
        sp_idx[i] = best[i].sp_idx;
    }

    Ok(RHier {
        rq,
        bits: 4 * n as u64 + 8 * o as u64 + 64,
        r_lo,
        r_sp,
        k: best_k,
        u6,
        sp_idx,
    })
}
