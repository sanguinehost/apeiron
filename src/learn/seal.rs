//! Activation-weighted `Q4_K`, `Q6_K` and `Q5_0` block quantization — the seal core.
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
//! The `Q6_K` and `Q5_0` paths extend the same weighting to the other two weighted types the
//! `Q4_K_M` type map uses (`Q8_0` and `F32` are copied verbatim, as `llama-quantize` does).
//! `quantize_row_q6_k` ports `quantize_row_q6_K_impl`/`quantize_row_q6_K_ref` and `quantize_row_q5_0`
//! ports `quantize_row_q5_0_impl`/`quantize_row_q5_0_ref`, `ggml/src/ggml-quants.c` lines 1970, 1869,
//! 2188 and 187 respectively, sharing `make_qx_quants` (line 628, `rmse_type == 1`).
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
//! [`make_qx_quants`] is `O(n·19)` time and `O(1)` extra space over its `l` output.
//! [`quantize_row_q6_k`] is `O(n)` time and output space (`n/256` blocks, each 16·`make_qx_quants`
//! at width 16 plus a fixed 256-element repack); [`quantize_matrix_q6_k`] is `O(n_out·n_in)`.
//! [`quantize_row_q5_0`] is `O(n)` (a row pass for `σ²` then `n/32` blocks), and
//! [`quantize_matrix_q5_0`] is `O(n_out·n_in)`.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::missing_errors_doc,
    clippy::manual_is_multiple_of,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::if_not_else,
    reason = "the seal is a byte-level port of llama.cpp's `quantize_row_q4_K_impl`: the `as` casts \
              follow explicit range or bounds checks (a check plus cast is what extracts, while \
              `TryFrom` emits an unmodeled external), a half-to-float conversion is float by nature, \
              the short names (`n`, `l`, `g`, `x`) are the reference arithmetic's vocabulary, the \
              explicit `0..n` index loops mirror the C bodies' own loop bounds, and an `if !c { a } \
              else { b }` keeps the reference's branch order. The pedantic lints stay on for the \
              rest of the crate."
)]

use crate::format::quant::{
    f16_to_f32, q4_k_scale_min, Q4_K_BLOCK_BYTES, Q4_K_SCALE_SIZE, Q5_0_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES, QK_K,
};

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

/// Weights per `Q5_0` block (`ggml` `QK5_0`).
const Q5_0_NK: usize = 32;
/// `make_qx_quants` `nmax` for `Q5_0`: codes `l ∈ [−16, 15]`, stored `l + 16`.
const Q5_0_NMAX: i32 = 16;
/// Weights per `Q6_K` per-group affine fit.
const Q6_K_GROUP: usize = 16;
/// Number of `Q6_K` groups per 256-weight super-block.
const Q6_K_NGROUPS: usize = QK_K / Q6_K_GROUP;
/// `make_qx_quants` `nmax` for `Q6_K`: codes `l ∈ [−32, 31]`, stored `l + 32`.
const Q6_K_NMAX: i32 = 32;
/// `Q6_K` group-scale clamp of `quantize_row_q6_K_ref`/`_impl`.
const Q6_K_SCALE_MAX: i32 = 127;

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
                write!(f, "length {len} is not a whole number of blocks")
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

/// The per-element weight of `make_qx_quants` with `rmse_type == 1`: `qwᵢ` when importance is
/// supplied, else the weight-only `xᵢ²` fallback llama.cpp's `NULL` branch uses.
#[inline]
fn qx_weight(weights: Option<&[f32]>, x: &[f32], i: usize) -> f32 {
    match weights {
        Some(w) => w[i],
        None => x[i] * x[i],
    }
}

/// The importance-weighted symmetric affine fit of one group to `x ≈ scale·l`, codes `l + nmax`.
///
/// Direct port of llama.cpp `make_qx_quants(n, nmax, x, L, rmse_type=1, qw)`
/// (`ggml/src/ggml-quants.c:628`): an initial `−nmax/max` scale, per-code least-squares
/// `(scale, code)`, then a nineteen-point scale probe `−(nmax + 0.1·is)/max` (`is ∈ [−9, 9]`,
/// `is ≠ 0`) keeping the best `sumlx²/suml2`. `weights` is the per-element importance; `None` is
/// the weight-only `w = xᵢ²` fallback. `l` receives the codes `l + nmax ∈ [0, 2·nmax)`. Time
/// `O(n·19)`, space `O(1)` beyond `l`.
fn make_qx_quants(n: usize, nmax: i32, x: &[f32], l: &mut [u8], weights: Option<&[f32]>) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax {
            amax = ax;
            max = x[i];
        }
    }
    if amax < GROUP_MAX_EPS {
        for v in l[..n].iter_mut() {
            *v = 0;
        }
        return 0.0;
    }
    let nm = nmax as f32;
    let mut iscale = -nm / max;
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let code = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        let w = qx_weight(weights, x, i);
        let c = code as f32;
        l[i] = (code + nmax) as u8;
        sumlx += w * x[i] * c;
        suml2 += w * c * c;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9i32 {
        if is == 0 {
            continue;
        }
        iscale = -(nm + 0.1 * (is as f32)) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let code = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            let w = qx_weight(weights, x, i);
            let c = code as f32;
            sumlx += w * x[i] * c;
            suml2 += w * c * c;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let code = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
                l[i] = (code + nmax) as u8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Quantize one 256-weight `Q6_K` super-block. `x` is exactly 256 weights; `qw` is the per-element
/// importance for this block (`None` = uniform). Writes all 210 bytes of `out`: `ql[128]`, `qh[64]`,
/// `scales[16]` (signed), `f16 d`. The codes are re-derived from the stored `f16 d` and scales, as
/// llama.cpp does, so the block matches its own dequantization.
///
/// Time `O(QK_K·19)` (sixteen 16-wide `make_qx_quants` fits plus a constant repack); space `O(QK_K)`.
fn quantize_q6_k_block(x: &[f32], qw: Option<&[f32]>, out: &mut [u8; Q6_K_BLOCK_BYTES]) {
    let mut l = [0u8; QK_K];
    let mut scales = [0f32; Q6_K_NGROUPS];
    let mut max_scale = 0f32;
    let mut max_abs_scale = 0f32;
    for ib in 0..Q6_K_NGROUPS {
        let start = ib * Q6_K_GROUP;
        let wg = qw.map(|w| &w[start..start + Q6_K_GROUP]);
        let s = make_qx_quants(
            Q6_K_GROUP,
            Q6_K_NMAX,
            &x[start..start + Q6_K_GROUP],
            &mut l[start..start + Q6_K_GROUP],
            wg,
        );
        scales[ib] = s;
        let abs_scale = s.abs();
        if abs_scale > max_abs_scale {
            max_abs_scale = abs_scale;
            max_scale = s;
        }
    }

    // `d` lives at the tail (`QK_K/2 + QK_K/4 + QK_K/16`), scales just before it.
    let scales_off = QK_K / 2 + QK_K / 4;
    let d_off = scales_off + QK_K / 16;
    if max_abs_scale < GROUP_MAX_EPS {
        out.fill(0);
        return;
    }

    let iscale = -128.0f32 / max_scale;
    let d_bits = f16_bits(1.0f32 / iscale);
    out[d_off..d_off + 2].copy_from_slice(&d_bits.to_le_bytes());
    let mut lscales = [0i8; Q6_K_NGROUPS];
    for ib in 0..Q6_K_NGROUPS {
        lscales[ib] = nearest_int(iscale * scales[ib]).min(Q6_K_SCALE_MAX) as i8;
        out[scales_off + ib] = lscales[ib] as u8;
    }

    // Re-derive the codes from the stored f16 `d` and the 8-bit scales (llama.cpp `_ref`/`_impl`).
    let d_back = f16_to_f32(d_bits);
    for j in 0..Q6_K_NGROUPS {
        let dj = d_back * f32::from(lscales[j]);
        if dj == 0.0 {
            continue;
        }
        for ii in 0..Q6_K_GROUP {
            let code = nearest_int(x[j * Q6_K_GROUP + ii] / dj).clamp(-32, 31);
            l[j * Q6_K_GROUP + ii] = (code + 32) as u8;
        }
    }

    // Pack the 6-bit codes: `ql[2·64·n …]` low nibbles, `qh[32·n …]` the upper two bits.
    for n in 0..2 {
        let base = 128 * n;
        let qlo = 64 * n;
        let qho = 32 * n;
        for ll in 0..32 {
            let q1 = l[base + ll] & 0xF;
            let q2 = l[base + ll + 32] & 0xF;
            let q3 = l[base + ll + 64] & 0xF;
            let q4 = l[base + ll + 96] & 0xF;
            out[qlo + ll] = q1 | (q3 << 4);
            out[qlo + ll + 32] = q2 | (q4 << 4);
            out[QK_K / 2 + qho + ll] = (l[base + ll] >> 4)
                | ((l[base + ll + 32] >> 4) << 2)
                | ((l[base + ll + 64] >> 4) << 4)
                | ((l[base + ll + 96] >> 4) << 6);
        }
    }
}

/// Quantize one row of weights to `ggml` `Q6_K`.
///
/// `row.len()` must be a multiple of 256. `weights` is the per-element importance, the same length
/// as `row`; `None` uses the weight-only `x²` fit. A row that is not a whole number of blocks, a
/// length mismatch, or a NaN/infinity refuses. Time and output space `O(row.len())`.
pub fn quantize_row_q6_k(row: &[f32], weights: Option<&[f32]>) -> Result<Vec<u8>, SealError> {
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
    let mut out = Vec::with_capacity(row.len() / QK_K * Q6_K_BLOCK_BYTES);
    let mut block = [0u8; Q6_K_BLOCK_BYTES];
    for (b, chunk) in row.chunks_exact(QK_K).enumerate() {
        let qw = weights.map(|w| &w[b * QK_K..(b + 1) * QK_K]);
        quantize_q6_k_block(chunk, qw, &mut block);
        out.extend_from_slice(&block);
    }
    Ok(out)
}

/// Quantize a row-major `n_out × n_in` weight matrix to `ggml` `Q6_K`.
///
/// Every row is quantized by [`quantize_row_q6_k`]; `tau` is the per-column energy (`τ`, length
/// `n_in`) shared by every row, or `None` for the weight-only fit. A shape mismatch, `n_in` not a
/// multiple of 256, or a NaN/infinity refuses. Time and output space `O(n_out·n_in)`.
pub fn quantize_matrix_q6_k(
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
    let bytes_per_row = (n_in / QK_K) * Q6_K_BLOCK_BYTES;
    let capacity = n_out
        .checked_mul(bytes_per_row)
        .ok_or(SealError::Overflow)?;
    let mut out = Vec::with_capacity(capacity);
    for r in 0..n_out {
        let row = &w[r * n_in..(r + 1) * n_in];
        out.extend_from_slice(&quantize_row_q6_k(row, tau)?);
    }
    Ok(out)
}

/// Quantize one 32-weight `Q5_0` block, weight-only. `x` is exactly 32 weights; the scale is
/// `max/−16`, codes `round(x/d + 16.5)` clamped to `0..=31`, the fifth bit in the `qh` plane.
///
/// Direct port of llama.cpp `quantize_row_q5_0_ref` (`ggml/src/ggml-quants.c:187`). Writes all 22
/// bytes: `f16 d`, `qh[4]`, `qs[16]`. Time and space `O(32)`.
fn quantize_q5_0_block_ref(x: &[f32], out: &mut [u8; Q5_0_BLOCK_BYTES]) {
    let mut amax = 0f32;
    let mut max = 0f32;
    for &v in x {
        let a = v.abs();
        if amax < a {
            amax = a;
            max = v;
        }
    }
    let d = max / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    out[0..2].copy_from_slice(&f16_bits(d).to_le_bytes());
    let mut qh = 0u32;
    for j in 0..16 {
        let x0 = x[j] * id;
        let x1 = x[16 + j] * id;
        let xi0 = ((x0 + 16.5) as i8).min(31) as u8;
        let xi1 = ((x1 + 16.5) as i8).min(31) as u8;
        out[6 + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= u32::from((xi0 & 0x10) >> 4) << j;
        qh |= u32::from((xi1 & 0x10) >> 4) << (j + 16);
    }
    out[2..6].copy_from_slice(&qh.to_le_bytes());
}

/// Quantize one 32-weight `Q5_0` block with imatrix importance. `weight[j] = qw[j]·√(σ² + x_j²)`
/// (`σ²` is the whole-row mean square, passed in); the codes come from [`make_qx_quants`] at
/// `nmax = 16`.
///
/// Direct port of llama.cpp `quantize_row_q5_0_impl` (`ggml/src/ggml-quants.c:2188`). Writes all 22
/// bytes. Time and space `O(32)`.
fn quantize_q5_0_block(x: &[f32], qw: &[f32], sigma2: f32, out: &mut [u8; Q5_0_BLOCK_BYTES]) {
    let mut weight = [0f32; Q5_0_NK];
    for j in 0..Q5_0_NK {
        weight[j] = qw[j] * (sigma2 + x[j] * x[j]).sqrt();
    }
    let mut l = [0u8; Q5_0_NK];
    let d = make_qx_quants(Q5_0_NK, Q5_0_NMAX, x, &mut l, Some(&weight));
    out[0..2].copy_from_slice(&f16_bits(d).to_le_bytes());
    let mut qh = 0u32;
    for j in 0..16 {
        let xi0 = l[j];
        let xi1 = l[j + 16];
        out[6 + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= u32::from((xi0 & 0x10) >> 4) << j;
        qh |= u32::from((xi1 & 0x10) >> 4) << (j + 16);
    }
    out[2..6].copy_from_slice(&qh.to_le_bytes());
}

/// Quantize one row of weights to `ggml` `Q5_0`.
///
/// `row.len()` must be a multiple of 32. `weights` is the per-element importance, the same length as
/// `row`; `None` falls back to the `max/−16` reference fit. A row that is not a whole number of
/// blocks, a length mismatch, or a NaN/infinity refuses. Time and output space `O(row.len())`.
pub fn quantize_row_q5_0(row: &[f32], weights: Option<&[f32]>) -> Result<Vec<u8>, SealError> {
    if row.len() % Q5_0_NK != 0 {
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
    let mut out = Vec::with_capacity(row.len() / Q5_0_NK * Q5_0_BLOCK_BYTES);
    let mut block = [0u8; Q5_0_BLOCK_BYTES];
    match weights {
        None => {
            for chunk in row.chunks_exact(Q5_0_NK) {
                quantize_q5_0_block_ref(chunk, &mut block);
                out.extend_from_slice(&block);
            }
        }
        Some(w) => {
            let sum_x2: f32 = row.iter().map(|&v| v * v).sum();
            let sigma2 = sum_x2 / row.len() as f32;
            for (b, chunk) in row.chunks_exact(Q5_0_NK).enumerate() {
                quantize_q5_0_block(
                    chunk,
                    &w[b * Q5_0_NK..(b + 1) * Q5_0_NK],
                    sigma2,
                    &mut block,
                );
                out.extend_from_slice(&block);
            }
        }
    }
    Ok(out)
}

/// Quantize a row-major `n_out × n_in` weight matrix to `ggml` `Q5_0`.
///
/// Every row is quantized by [`quantize_row_q5_0`]; `tau` is the per-column energy (`τ`, length
/// `n_in`) shared by every row, or `None` for the reference fit. A shape mismatch, `n_in` not a
/// multiple of 32, or a NaN/infinity refuses. Time and output space `O(n_out·n_in)`.
pub fn quantize_matrix_q5_0(
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
    let bytes_per_row = (n_in / Q5_0_NK) * Q5_0_BLOCK_BYTES;
    let capacity = n_out
        .checked_mul(bytes_per_row)
        .ok_or(SealError::Overflow)?;
    let mut out = Vec::with_capacity(capacity);
    for r in 0..n_out {
        let row = &w[r * n_in..(r + 1) * n_in];
        out.extend_from_slice(&quantize_row_q5_0(row, tau)?);
    }
    Ok(out)
}
