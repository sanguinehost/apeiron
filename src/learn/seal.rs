//! Activation-weighted `Q4_K` block quantization — the seal core.
//!
//! The forward error a quantized weight causes is the function distortion
//! `‖X(W−Wq)‖² = Σ_j τ_j r_j²`, where `τ_j = Σ_t x_{t,j}²` is input channel `j`'s energy read
//! from the calibration activations `X`. An affine fit that minimises the `τ`-weighted residual
//! minimises that error; a weight-only fit cannot read `τ`, because `W` carries no information
//! about `X`. [`channel_energy`] forms `τ`; [`quantize_row_q4_k`] and [`quantize_matrix_q4_k`] fit
//! it.
//!
//! Output is `ggml` `Q4_K` super-blocks: 256 weights, 144 bytes — `f16 d`, `f16 dmin`, twelve
//! packed 6-bit group scales/mins and 128 nibble bytes — so
//! [`crate::format::quant::dequant_q4_k_block`] reads the result unchanged. Each block is eight
//! 32-weight groups; the importance-weighted affine fit `w ≈ d·q + m` runs per group, then the
//! group scales and mins are themselves fit to six bits.
//!
//! **Subsumption.** `subsumed: llama.cpp ece963f41 quantize_row_q4_K_impl`
//! (`ggml/src/ggml-quants.c`, sha256 `07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d`,
//! line 1553): the importance-matrix Q4_K quantizer is the measured best for this role. The port
//! keeps its weighted affine search (`make_qkx3_quants`, line 993: `rmin=-0.9`, `rdelta=0.05`,
//! `nstep=36`, squared error) and its weighted 6-bit scale/min fit (`make_qp_quants`), with the
//! importance weighting `weight[j] = qw[j]·√(σ² + x_j²)`, `σ² = 2·Σ x²/256`. The `None` path is
//! the `qw ≡ 1` uniform specialization of that same weighting; llama.cpp's separate weight-only
//! path (`quantize_row_q4_K_ref`) uses `av_x + |x|`, which this module does not need.
//!
//! **Lineage.** The activation-aware scale and the importance-weighted block fit port from
//! `fold/modelfold/src/quant.rs` rev `d497161f` (`pub mod seal`: `aware_scale`, `fit_blocks`) and
//! `spinor-ttl/src/spinor_ttl/seal.py` rev `a05bb1b` (`aware_scale`, `function_distortion`,
//! `fit_blocks`). Those compute the chart indirectly (mean-abs activation, `α = ½` change of
//! chart); this module consumes `τ` directly as the per-column importance and emits the `ggml`
//! byte layout. The byte layout and the weighted search come from llama.cpp.
//!
//! **Evidence.** Round-trip, calibrated-beats-uncalibrated, refusals and the real-tensor check are
//! `tested` (`tests/learn_seal.rs`). The arithmetic is a direct port of the cited C body, not a
//! refined extraction, so no `proved` claim is made here.
//!
//! **Complexity** (AGENTS §8). [`channel_energy`] is `O(n_tokens·n_in)` time (one pass over the
//! calibration matrix) and `O(n_in)` space. [`quantize_row_q4_k`] is `O(n)` time and output space
//! over `n` weights: `n/256` blocks, each a fixed 8·(32·37 + 8·14 + 8) reduction.
//! [`quantize_matrix_q4_k`] is `O(n_out·n_in)`. [`make_qkx3_quants`] is `O(32·37)` time and `O(32)`
//! space; [`make_qp_quants`] is `O(8·(9 + 5·8))`; [`nearest_int`] and [`f16_bits`] are `O(1)`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::missing_errors_doc,
    clippy::manual_is_multiple_of,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the seal is a byte-level port of llama.cpp's `quantize_row_q4_K_impl`: the `as` casts \
              follow explicit range or bounds checks (a check plus cast is what extracts, while \
              `TryFrom` emits an unmodeled external), a half-to-float conversion is float by nature, \
              and the short names (`n`, `l`, `g`, `x`) are the reference arithmetic's vocabulary. \
              The pedantic lints stay on for the rest of the crate."
)]

use crate::format::quant::{f16_to_f32, q4_k_scale_min, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, QK_K};

/// Weights per 32-weight affine group.
const GROUP: usize = 32;
/// Groups per 256-weight super-block.
const N_GROUPS: usize = QK_K / GROUP;
/// Largest 4-bit code.
const NMAX: usize = 15;
/// Largest 6-bit group scale/min.
const NMAX_SCALE: i32 = 63;
/// ggml `GROUP_MAX_EPS`: below this a group's scales are treated as all-zero.
const GROUP_MAX_EPS: f32 = 1e-15;
/// Search-grid start of `make_qkx3_quants` (llama.cpp call site).
const FIT_RMIN: f32 = -0.9;
/// Search-grid step of `make_qkx3_quants` (llama.cpp call site).
const FIT_RDELTA: f32 = 0.05;
/// Number of affine-search grid steps of `make_qkx3_quants` (llama.cpp call site).
const FIT_NSTEP: i32 = 36;

/// A refused seal-quantization operation. Integer-only, so callers can match without formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealError {
    /// A length or shape disagrees with the contract.
    Shape {
        /// The offending field.
        field: &'static str,
        /// The length the contract requires.
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },
    /// A row length is not a whole number of 256-weight `Q4_K` super-blocks.
    NotMultiple {
        /// The offending length.
        len: usize,
    },
    /// A non-finite value (NaN or infinity) was found at `index`.
    NonFinite {
        /// Index of the offending element.
        index: usize,
    },
    /// A length or product did not fit the target integer width.
    Overflow,
}

impl core::fmt::Display for SealError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SealError::Shape {
                field,
                expected,
                got,
            } => write!(f, "{field} shape mismatch: expected {expected}, got {got}"),
            SealError::NotMultiple { len } => {
                write!(f, "length {len} is not a multiple of {QK_K}")
            }
            SealError::NonFinite { index } => write!(f, "non-finite value at index {index}"),
            SealError::Overflow => write!(f, "length overflow"),
        }
    }
}

impl std::error::Error for SealError {}

/// The index of the first non-finite element of `s`, or `None` when every element is finite.
fn first_nonfinite(s: &[f32]) -> Option<usize> {
    s.iter().position(|v| !v.is_finite())
}

/// `τ_j = Σ_t x_{t,j}²` — input channel `j`'s energy over the calibration activations.
///
/// `acts` is the row-major `n_tokens × n_in` calibration matrix; a length mismatch or a non-finite
/// activation refuses. The accumulation is in `f64` and narrowed once, so `τ` is stable across
/// token counts. Time `O(n_tokens·n_in)`, space `O(n_in)`.
pub fn channel_energy(acts: &[f32], n_tokens: usize, n_in: usize) -> Result<Vec<f32>, SealError> {
    let expected = n_tokens.checked_mul(n_in).ok_or(SealError::Overflow)?;
    if acts.len() != expected {
        return Err(SealError::Shape {
            field: "acts",
            expected,
            got: acts.len(),
        });
    }
    if let Some(index) = first_nonfinite(acts) {
        return Err(SealError::NonFinite { index });
    }
    let mut acc = vec![0f64; n_in];
    for t in 0..n_tokens {
        let base = t * n_in;
        for (j, &v) in acts[base..base + n_in].iter().enumerate() {
            let x = f64::from(v);
            acc[j] += x * x;
        }
    }
    Ok(acc.iter().map(|&a| a as f32).collect())
}

/// ggml's `nearest_int`: round-to-nearest-even through the exact float trick
/// (`fval + 1.5·2²³`, mask the mantissa, subtract `2²²`). Faithful to the reference so the emitted
/// codes match llama.cpp bit for bit; `fval` is assumed `< 2²²` as at every call site.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0f32;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// f32 → nearest-even IEEE-754 half (`f16`), the storage width of `d` and `dmin`.
#[inline]
fn f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xFF) as i32;
    let m = x & 0x007F_FFFF;
    if e == 255 {
        return sign | 0x7C00 | u16::from(m != 0);
    }
    let he = e - 112;
    if he >= 31 {
        return sign | 0x7C00;
    }
    if he <= 0 {
        let r = (f32::from_bits(x & 0x7FFF_FFFF) * 16_777_216.0).round_ties_even() as u32;
        return sign | r as u16;
    }
    let half = ((he as u32) << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    sign | (half + u32::from(rem > 0x1000 || (rem == 0x1000 && half & 1 == 1))) as u16
}

/// The importance-weighted affine fit of one 32-weight group to `w ≈ d·q + m`, codes in `0..=15`.
///
/// Direct port of llama.cpp `make_qkx3_quants(32, 15, x, weights, L, &min, Laux, -0.9, 0.05, 36,
/// false)`: a 37-point scale grid `(rmin + rdelta·is + nmax)/(max−min)`, weighted least-squares
/// `(scale, min)` per candidate, best by weighted squared error. `weights` is the per-element
/// importance; `l` receives the chosen codes; the returned scale and `*the_min` (always `≥ 0`) are
/// the group's `(scale, min)`. Time `O(32·37)`, space `O(32)`.
fn make_qkx3_quants(
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
) -> f32 {
    let n = x.len();
    let nmax = NMAX as f32;
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        for v in l.iter_mut() {
            *v = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0f32;
    for i in 0..n {
        let code = nearest_int(iscale * (x[i] - min)).clamp(0, NMAX as i32) as u8;
        l[i] = code;
        let diff = scale * f32::from(code) + min - x[i];
        best_mad += weights[i] * diff * diff;
    }
    if FIT_NSTEP < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=FIT_NSTEP {
        iscale = (FIT_RMIN + FIT_RDELTA * (is as f32) + nmax) / (max - min);
        let mut sum_l = 0f32;
        let mut sum_l2 = 0f32;
        let mut sum_xl = 0f32;
        for i in 0..n {
            let code = nearest_int(iscale * (x[i] - min)).clamp(0, NMAX as i32) as u8;
            laux[i] = code;
            let w = weights[i];
            let c = f32::from(code);
            sum_l += w * c;
            sum_l2 += w * c * c;
            sum_xl += w * c * x[i];
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0f32;
            for i in 0..n {
                let diff = this_scale * f32::from(laux[i]) + this_min - x[i];
                mad += weights[i] * diff * diff;
            }
            if mad < best_mad {
                l.copy_from_slice(laux);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// The weighted 6-bit fit of the eight group scales (or mins) to `0..=63`.
///
/// Direct port of llama.cpp `make_qp_quants(8, 63, x, L, quant_weights)`: an initial `63/max`
/// round, a nine-point scale probe `(0.1·is + 63)/max`, then up to five coordinate-descent sweeps
/// maximising `sumlx²/suml2` under the importance `qw`. `x` is non-negative (scales/mins were
/// returned `≥ 0`); `l` receives the 6-bit codes. Time `O(8·(9 + 5·8))`, space `O(8)`.
fn make_qp_quants(x: &[f32], qw: &[f32], l: &mut [u8]) -> f32 {
    let n = x.len();
    let nmax = NMAX_SCALE as f32;
    let mut max = 0f32;
    for &v in x {
        if v > max {
            max = v;
        }
    }
    if max < GROUP_MAX_EPS {
        for v in l.iter_mut() {
            *v = 0;
        }
        return 0.0;
    }
    let mut iscale = nmax / max;
    for i in 0..n {
        l[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0f32;
    for i in 0..n {
        let diff = x[i] - scale * f32::from(l[i]);
        best_mse += qw[i] * diff * diff;
    }
    for is in -4..=4i32 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * (is as f32) + nmax) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0f32;
        for i in 0..n {
            let code = nearest_int(iscale_is * x[i]).clamp(0, NMAX_SCALE);
            let diff = x[i] - scale_is * (code as f32);
            mse += qw[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let code = nearest_int(iscale * x[i]).clamp(0, NMAX_SCALE);
        l[i] = code as u8;
        let c = code as f32;
        sumlx += qw[i] * x[i] * c;
        suml2 += qw[i] * c * c;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = qw[i];
            let mut slx = sumlx - w * x[i] * f32::from(l[i]);
            let mut sl2 = suml2 - w * f32::from(l[i]) * f32::from(l[i]);
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).clamp(0, NMAX_SCALE);
                if new_l != i32::from(l[i]) {
                    slx += w * x[i] * (new_l as f32);
                    sl2 += w * (new_l as f32) * (new_l as f32);
                    if slx * slx * suml2 > sumlx * sumlx * sl2 {
                        l[i] = new_l as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// Quantize one 256-weight `Q4_K` super-block. `x` is exactly 256 weights; `qw` is the per-element
/// importance (`None` = uniform). Writes all 144 bytes of `out`; the codes are re-derived from the
/// stored `f16 d`/`dmin` exactly as llama.cpp does, so the block matches its own dequantization.
fn quantize_block(x: &[f32], qw: Option<&[f32]>, out: &mut [u8; Q4_K_BLOCK_BYTES]) {
    let mut sum_x2 = 0f32;
    for &v in x {
        sum_x2 += v * v;
    }
    let sigma2 = 2.0f32 * sum_x2 / QK_K as f32;

    let mut codes = [0u8; QK_K];
    let mut laux = [0u8; GROUP];
    let mut lscales = [0u8; N_GROUPS];
    let mut lmins = [0u8; N_GROUPS];
    let mut scales = [0f32; N_GROUPS];
    let mut mins = [0f32; N_GROUPS];
    let mut sw = [0f32; N_GROUPS];

    for g in 0..N_GROUPS {
        let xg = &x[g * GROUP..g * GROUP + GROUP];
        let mut wgt = [0f32; GROUP];
        for (l, &xv) in xg.iter().enumerate() {
            let base = (sigma2 + xv * xv).sqrt();
            wgt[l] = match qw {
                Some(w) => w[g * GROUP + l] * base,
                None => base,
            };
        }
        sw[g] = wgt.iter().sum();
        let mut mn = 0f32;
        let start = g * GROUP;
        scales[g] = make_qkx3_quants(
            xg,
            &wgt,
            &mut codes[start..start + GROUP],
            &mut mn,
            &mut laux,
        );
        mins[g] = mn;
    }

    let d_block = make_qp_quants(&scales, &sw, &mut lscales);
    let m_block = make_qp_quants(&mins, &sw, &mut lmins);

    // Pack the eight 6-bit (scale, min) pairs into twelve bytes (ggml `quantize_row_q4_K_impl`).
    let mut sc12 = [0u8; Q4_K_SCALE_SIZE];
    for j in 0..N_GROUPS {
        let ls = lscales[j];
        let lm = lmins[j];
        if j < 4 {
            sc12[j] = ls;
            sc12[j + 4] = lm;
        } else {
            sc12[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            sc12[j - 4] |= (ls >> 4) << 6;
            sc12[j] |= (lm >> 4) << 6;
        }
    }

    let d_bits = f16_bits(d_block);
    let m_bits = f16_bits(m_block);
    out[0..2].copy_from_slice(&d_bits.to_le_bytes());
    out[2..4].copy_from_slice(&m_bits.to_le_bytes());
    out[4..4 + Q4_K_SCALE_SIZE].copy_from_slice(&sc12);

    // Re-derive the codes from the stored f16 `d`/`dmin`, as llama.cpp does, so the block's codes
    // are consistent with what a reader computes (a `d` of zero keeps the fit's codes).
    let d = f16_to_f32(d_bits);
    let dmin = f16_to_f32(m_bits);
    for j in 0..N_GROUPS {
        let (sc, m) = q4_k_scale_min(&sc12, j);
        let dd = d * f32::from(sc);
        if dd == 0.0 {
            continue;
        }
        let dm = dmin * f32::from(m);
        for ii in 0..GROUP {
            let code = nearest_int((x[j * GROUP + ii] + dm) / dd).clamp(0, NMAX as i32);
            codes[j * GROUP + ii] = code as u8;
        }
    }

    // Pack the 256 nibbles: each 32-byte window holds group `2b` low and `2b+1` high.
    let qs = &mut out[4 + Q4_K_SCALE_SIZE..];
    for b in 0..4 {
        for l in 0..GROUP {
            qs[b * GROUP + l] = codes[64 * b + l] | (codes[64 * b + 32 + l] << 4);
        }
    }
}

/// Quantize one row of weights to `ggml` `Q4_K`.
///
/// `row.len()` must be a multiple of 256. `weights` is the per-element importance (the column
/// energy `τ_j`), the same length as `row`; `None` uses uniform importance (`qw ≡ 1`). A row that
/// is not a whole number of blocks, a length mismatch, or a NaN/infinity refuses. Time and output
/// space `O(row.len())`.
pub fn quantize_row_q4_k(row: &[f32], weights: Option<&[f32]>) -> Result<Vec<u8>, SealError> {
    if row.len() % QK_K != 0 {
        return Err(SealError::NotMultiple { len: row.len() });
    }
    if let Some(w) = weights {
        if w.len() != row.len() {
            return Err(SealError::Shape {
                field: "weights",
                expected: row.len(),
                got: w.len(),
            });
        }
        if let Some(index) = first_nonfinite(w) {
            return Err(SealError::NonFinite { index });
        }
    }
    if let Some(index) = first_nonfinite(row) {
        return Err(SealError::NonFinite { index });
    }
    let mut out = Vec::with_capacity(row.len() / QK_K * Q4_K_BLOCK_BYTES);
    let mut block = [0u8; Q4_K_BLOCK_BYTES];
    for (b, chunk) in row.chunks_exact(QK_K).enumerate() {
        let qw = weights.map(|w| &w[b * QK_K..(b + 1) * QK_K]);
        quantize_block(chunk, qw, &mut block);
        out.extend_from_slice(&block);
    }
    Ok(out)
}

/// Quantize a row-major `n_out × n_in` weight matrix to `ggml` `Q4_K`.
///
/// Every row is quantized by [`quantize_row_q4_k`]; `tau` is the per-column energy (`τ`, length
/// `n_in`) shared by every row, or `None` for uniform importance. A shape mismatch, `n_in` not a
/// multiple of 256, or a NaN/infinity refuses. Time and output space `O(n_out·n_in)`.
pub fn quantize_matrix_q4_k(
    w: &[f32],
    n_out: usize,
    n_in: usize,
    tau: Option<&[f32]>,
) -> Result<Vec<u8>, SealError> {
    let expected = n_out.checked_mul(n_in).ok_or(SealError::Overflow)?;
    if w.len() != expected {
        return Err(SealError::Shape {
            field: "w",
            expected,
            got: w.len(),
        });
    }
    if let Some(t) = tau {
        if t.len() != n_in {
            return Err(SealError::Shape {
                field: "tau",
                expected: n_in,
                got: t.len(),
            });
        }
        if let Some(index) = first_nonfinite(t) {
            return Err(SealError::NonFinite { index });
        }
    }
    if let Some(index) = first_nonfinite(w) {
        return Err(SealError::NonFinite { index });
    }
    let bytes_per_row = (n_in / QK_K) * Q4_K_BLOCK_BYTES;
    let capacity = n_out
        .checked_mul(bytes_per_row)
        .ok_or(SealError::Overflow)?;
    let mut out = Vec::with_capacity(capacity);
    for r in 0..n_out {
        let row = &w[r * n_in..(r + 1) * n_in];
        out.extend_from_slice(&quantize_row_q4_k(row, tau)?);
    }
    Ok(out)
}
