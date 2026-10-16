//! The `f16` round-trip and the two scale carriers of the one-bit fold transfer: the affine column
//! quantizer and the hierarchical row-scale group quantizer.
//!
//! Port of `fold_transfer.py` lines 511–558 (`f16r`, `quant_c_u8`, `quant_r_hier`). The Python
//! defines *what* each computes; the bodies here reproduce its arithmetic in `f64` on the CPU.
//!
//! - [`f16r`] rounds an `f64` to the nearest IEEE-754 binary16 value and back, the carrier the two
//!   scalars `c_min`/`c_max` (32 bits) and each row's `mu`/`sp` header are stored in.
//! - [`quant_c_u8`] stores two `f16` endpoint scalars plus `bits`/column affine codes.
//! - [`quant_r_hier`] stores a per-row 6-bit mean on the `f16` `mu_min`/`mu_max` grid, a 2-bit
//!   spread index into four levels between the 5% deviation quantile and the max deviation, and a
//!   4-bit code per element on the row's `(r_lo, r_sp)` grid. The row picks the best of the
//!   `du ∈ {−1, 0, 1}` mean shifts crossed with the four spreads by summed squared error, with ties
//!   resolved to the earlier candidate (the Python's `err < best_err` mask).
//!
//! The 4-bit codes `k`, the 6-bit row mean `u6` and the 2-bit spread index `sp_idx` are the on-disk
//! format; [`RHier`] returns them alongside `rq` and the decoded per-row `(r_lo, r_sp)`.
//!
//! # Deviations from the Python
//!
//! - [`quant_c_u8`] and [`quant_r_hier`] return `Result` and refuse an empty slice, a ragged slice
//!   (`len != o·g`) or an out-of-range `bits`, where the Python raises inside torch (`c.min()` on an
//!   empty tensor) or silently broadcasts.
//! - [`f16r`] narrows `f64 → f32` (round-to-nearest-even) and then `f32 → f16` on the bit pattern,
//!   rather than a single `f64 → f16` cast. For every `f64` whose value survives the `f32` step
//!   exactly the two agree; the `f32` step is the storage width the Python tensor arithmetic uses.
//!
//! # Grade
//!
//! `tested` (`tests/learn_fold_quant.rs`). No extraction or refinement links these bodies to a
//! Lean theorem, so nothing here is `proved`.

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
    reason = "the row-major kernels index `[o, g]` buffers by `i/j` and name the carriers `r_lo`, \
              `r_hi`, `r_sp`, `rq`, `u6`, `sp_idx` as the reference does; the `as` casts move \
              between `f64` and integer codes after the value's range is fixed by construction, and \
              `f16` is float by nature"
)]

use crate::format::quant::f16_to_f32;

/// A refusal from a quantization carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantError {
    /// An empty slice (`quant_c_u8`) or a zero row/column count (`quant_r_hier`).
    Empty,
    /// A slice length that is not the declared `o · g` shape.
    ShapeMismatch,
    /// A bit width outside `1..=32` (a zero width divides by `levels = 0`).
    BadBits,
    /// A bit total `bits · n + 32` that overflows `u64`.
    LengthOverflow,
}

/// Round an `f64` to the nearest IEEE-754 binary16 value and back: torch `x.half().to(x.dtype)`.
///
/// Round-to-nearest-even, gradual underflow to subnormals, overflow to `±∞`; `±0.0` is preserved
/// and a NaN maps to the canonical quiet `f16`. Narrowing runs `f64 → f32` (the platform's
/// round-to-nearest-even cast) then `f32 → f16` on the bit pattern. Time `O(1)`.
pub fn f16r(x: f64) -> f64 {
    f64::from(f16_to_f32(f16_bits(x as f32)))
}

/// `f32` → nearest-even binary16 bits: round-to-nearest-even with subnormal underflow and overflow
/// to `±∞`; a NaN maps to the canonical quiet `0x7E00` pattern. Time `O(1)`.
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
        // Subnormal range: scale the magnitude by 2^24 so half a subnormal step is half a unit,
        // round to nearest-even, then reinterpret the count as the `f16` code (which reaches the
        // smallest normal `0x0400` when the value rounds up out of the subnormal range).
        let r = (f32::from_bits(b & 0x7FFF_FFFF) * 16_777_216.0).round_ties_even() as u32;
        return sign | r as u16;
    }
    let half = ((he as u32) << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    sign | (half + u32::from(rem > 0x1000 || (rem == 0x1000 && half & 1 == 1))) as u16
}

/// Torch `Tensor.quantile(q)` over a 1-D slice, default linear interpolation: sort ascending,
/// `idx = q·(n−1)`, then the linear blend of the two bracketing order statistics. Time `O(n log n)`,
/// space `O(n)`.
fn quantile_linear(values: &[f64], q: f64) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = q * (n as f64 - 1.0);
    let lo = idx.floor();
    let lo_i = if lo < 0.0 {
        0
    } else if lo >= (n - 1) as f64 {
        n - 1
    } else {
        lo as usize
    };
    if lo_i >= n - 1 {
        return sorted[n - 1];
    }
    sorted[lo_i] + (sorted[lo_i + 1] - sorted[lo_i]) * (idx - lo_i as f64)
}

/// Affine column-scale quantization: two `f16` scalars (32 bits) plus `bits`/column.
///
/// `c_min = f16r(min c)`, `c_max = f16r(max c)`, `span = max(c_max − c_min, 1e-12)`; each element is
/// rounded to the nearest of `levels + 1 = 2^bits` grid points on `[c_min, c_max]` and the grid value
/// returned. The bit count is `bits · n + 32`.
///
/// # Errors
///
/// [`QuantError::Empty`] for an empty slice; [`QuantError::BadBits`] for `bits` outside `1..=32`;
/// [`QuantError::LengthOverflow`] if `bits · n + 32` overflows `u64`.
///
/// # Complexity
///
/// Time `O(n)`, space `O(n)` for the output (the scan and the grid round are each one pass over the
/// `n` columns; no sort).
pub fn quant_c_u8(c: &[f64], bits: u32) -> Result<(Vec<f64>, u64), QuantError> {
    if c.is_empty() {
        return Err(QuantError::Empty);
    }
    if bits == 0 || bits > 32 {
        return Err(QuantError::BadBits);
    }
    let n = c.len() as u64;
    let total = n
        .checked_mul(u64::from(bits))
        .and_then(|t| t.checked_add(32))
        .ok_or(QuantError::LengthOverflow)?;

    let mut lo = c[0];
    let mut hi = c[0];
    for &x in c {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    let c_min = f16r(lo);
    let c_max = f16r(hi);
    let span = (c_max - c_min).max(1e-12);
    let levels = f64::from(((1_u64 << bits) - 1) as u32);

    let mut out = vec![0.0_f64; c.len()];
    for (i, &x) in c.iter().enumerate() {
        let q = (((x - c_min) / span) * levels)
            .round_ties_even()
            .clamp(0.0, levels);
        out[i] = c_min + (q / levels) * span;
    }
    Ok((out, total))
}

/// A hierarchical row-scale quantization result: the on-disk codes and the values they decode to.
///
/// `rq[i·g + j] = r_lo[i] + (k[i·g + j] / 15) · r_sp[i]`, so `r_lo`, `r_sp` and `k` are the
/// per-row decoded grid and the stored 4-bit code respectively.
#[derive(Clone, Debug)]
pub struct RHier {
    /// Reconstructed values, row-major `[o, g]`.
    pub rq: Vec<f64>,
    /// The bit budget `4·o·g + 8·o + 64`.
    pub bits: u64,
    /// Per-row lower grid endpoint `mu·(1 − sp)`, length `o`.
    pub r_lo: Vec<f64>,
    /// Per-row grid width `max(mu·(1 + sp) − r_lo, 1e-12)`, length `o`.
    pub r_sp: Vec<f64>,
    /// Per-element 4-bit codes in `0..=15`, row-major `[o, g]`.
    pub k: Vec<u8>,
    /// Per-row 6-bit mean code in `0..=63`, the chosen `du` shift applied, length `o`.
    pub u6: Vec<u8>,
    /// Per-row spread level index in `0..=3`, length `o`.
    pub sp_idx: Vec<u8>,
}

impl RHier {
    /// Decode `rq` from the stored per-row grid and 4-bit codes: `r_lo + (k / 15) · r_sp`.
    ///
    /// # Complexity
    ///
    /// Time `O(o·g)`, space `O(o·g)`; one pass over the `k` codes, which carry the row count
    /// (`o = u6.len()`) and the column count (`g = k.len() / o`).
    pub fn reconstruct(&self) -> Vec<f64> {
        let o = self.u6.len();
        if o == 0 {
            return Vec::new();
        }
        let g = self.k.len() / o;
        let mut out = vec![0.0_f64; self.k.len()];
        for i in 0..o {
            let lo = self.r_lo[i];
            let sp = self.r_sp[i];
            for j in 0..g {
                out[i * g + j] = lo + (f64::from(self.k[i * g + j]) / 15.0) * sp;
            }
        }
        out
    }
}

/// Hierarchical group-scale quantization: a per-row 6-bit mean, a 2-bit spread and 4-bit codes.
///
/// Port of `quant_r_hier(r)`. The row mean `rm` is placed on the 6-bit grid `[mu_min, mu_max]`
/// (`f16r` of the mean extremes); `dev = max_j |r − rm| / max(|rm|, 1e-12)` gives the spread range
/// `[f16r(quantile(dev, 0.05) ≥ 1e-4), f16r(max(dev.max(), sp_min + 1e-4))]`, discretized to four
/// levels by torch `linspace(0, 1, 4)`. Across the `du ∈ {−1, 0, 1}` mean shifts and the four
/// spreads, each candidate quantizes the row to `k = clamp(round((r − r_lo)/r_sp · 15), 0, 15)` on
/// `[r_lo, r_hi] = mu·(1 ∓ sp)`, and the candidate with the least summed squared error is kept, ties
/// to the earlier one. The bit count is `4·o·g + 8·o + 64`.
///
/// # Errors
///
/// [`QuantError::Empty`] for `o == 0` or `g == 0`; [`QuantError::ShapeMismatch`] if `r.len() != o·g`
/// (or can't be, on overflow).
///
/// # Complexity
///
/// Time `O(o·g)` with the constant 12 (3 mean shifts × 4 spreads) per row; space `O(o·g)` for the
/// output and codes plus `O(o)` for the per-row state and the quantile sort.
pub fn quant_r_hier(r: &[f64], o: usize, g: usize) -> Result<RHier, QuantError> {
    if o == 0 || g == 0 {
        return Err(QuantError::Empty);
    }
    let n = o.checked_mul(g).ok_or(QuantError::ShapeMismatch)?;
    if r.len() != n {
        return Err(QuantError::ShapeMismatch);
    }

    let HierHeader {
        mu_min,
        mu_span,
        u6_base,
        sp_levels,
    } = hier_header(r, o, g);

    let mut best_err = vec![f64::INFINITY; o];
    let mut best_lo = vec![0.0_f64; o];
    let mut best_sp = vec![1.0_f64; o];
    let mut best_u6 = vec![0_u8; o];
    let mut best_sp_idx = vec![0_u8; o];
    let mut best_k = vec![0_u8; n];
    let mut best_rq = r.to_vec();

    let mut k_row = vec![0.0_f64; g];
    let mut rq_row = vec![0.0_f64; g];
    for &du in &[-1.0_f64, 0.0, 1.0] {
        for (i, &base) in u6_base.iter().enumerate() {
            let u6 = (base + du).clamp(0.0, 63.0);
            let mu = mu_min + (u6 / 63.0) * mu_span;
            for (s_idx, &sp) in sp_levels.iter().enumerate() {
                let r_lo = mu * (1.0 - sp);
                let r_sp = ((mu * (1.0 + sp)) - r_lo).max(1e-12);
                let err = row_candidate(r, g, i, r_lo, r_sp, &mut k_row, &mut rq_row);
                if err < best_err[i] {
                    best_err[i] = err;
                    best_lo[i] = r_lo;
                    best_sp[i] = r_sp;
                    best_u6[i] = u6 as u8;
                    best_sp_idx[i] = s_idx as u8;
                    for j in 0..g {
                        best_k[i * g + j] = k_row[j] as u8;
                        best_rq[i * g + j] = rq_row[j];
                    }
                }
            }
        }
    }

    let bits = 4 * n as u64 + 8 * o as u64 + 64;
    Ok(RHier {
        rq: best_rq,
        bits,
        r_lo: best_lo,
        r_sp: best_sp,
        k: best_k,
        u6: best_u6,
        sp_idx: best_sp_idx,
    })
}

/// The per-row-mean grid and the four spread levels shared by every candidate of a row.
struct HierHeader {
    /// `f16r` of the smallest row mean.
    mu_min: f64,
    /// `max(f16r(max mean) − f16r(min mean), 1e-12)`.
    mu_span: f64,
    /// The 6-bit mean code of each row before the `du` shift, in `0..=63`.
    u6_base: Vec<f64>,
    /// The four spread levels `linspace(0, 1, 4)` between the deviation quantile and its max.
    sp_levels: [f64; 4],
}

/// Build the row-mean grid and spread levels for [`quant_r_hier`]. Time `O(o·g log o)` (the
/// `o`-element quantile sort dominates), space `O(o)`.
fn hier_header(r: &[f64], o: usize, g: usize) -> HierHeader {
    let mut rm = vec![0.0_f64; o];
    for i in 0..o {
        let mut s = 0.0_f64;
        for j in 0..g {
            s += r[i * g + j];
        }
        rm[i] = s / (g as f64);
    }
    let mut mu_lo = rm[0];
    let mut mu_hi = rm[0];
    for &m in &rm {
        if m < mu_lo {
            mu_lo = m;
        }
        if m > mu_hi {
            mu_hi = m;
        }
    }
    let mu_min = f16r(mu_lo);
    let mu_max = f16r(mu_hi);
    let mu_span = (mu_max - mu_min).max(1e-12);

    let mut u6_base = vec![0.0_f64; o];
    for (i, &m) in rm.iter().enumerate() {
        u6_base[i] = ((((m - mu_min) / mu_span) * 63.0).round_ties_even()).clamp(0.0, 63.0);
    }

    let mut dev = vec![0.0_f64; o];
    for i in 0..o {
        let mut mx = 0.0_f64;
        for j in 0..g {
            let d = (r[i * g + j] - rm[i]).abs();
            if d > mx {
                mx = d;
            }
        }
        dev[i] = mx / rm[i].abs().max(1e-12);
    }
    let sp_min = f16r(quantile_linear(&dev, 0.05).max(1e-4));
    let dev_max = dev.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sp_max = f16r(dev_max.max(sp_min + 1e-4));

    // torch `linspace(0, 1, 4)`: step `1/3`, level `s` = `step · s`.
    let step = 1.0_f64 / 3.0;
    let mut sp_levels = [0.0_f64; 4];
    for (s, level) in sp_levels.iter_mut().enumerate() {
        *level = sp_min + (step * s as f64) * (sp_max - sp_min);
    }
    HierHeader {
        mu_min,
        mu_span,
        u6_base,
        sp_levels,
    }
}

/// Quantize row `i` on the grid `(r_lo, r_sp)` to 4-bit codes, writing `k_row = k` and
/// `rq_row = r_lo + (k/15)·r_sp`, and return the row's summed squared error. Time `O(g)`,
/// space `O(1)` beyond the `g`-length scratch slices.
fn row_candidate(
    r: &[f64],
    g: usize,
    i: usize,
    r_lo: f64,
    r_sp: f64,
    k_row: &mut [f64],
    rq_row: &mut [f64],
) -> f64 {
    let mut err = 0.0_f64;
    for j in 0..g {
        let x = r[i * g + j];
        let kf = (((x - r_lo) / r_sp) * 15.0)
            .round_ties_even()
            .clamp(0.0, 15.0);
        let rq = r_lo + (kf / 15.0) * r_sp;
        k_row[j] = kf;
        rq_row[j] = rq;
        let d = x - rq;
        err += d * d;
    }
    err
}
