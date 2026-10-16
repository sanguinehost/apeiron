//! `tests/learn_fold_quant.rs` — the one-bit fold-transfer carriers: the `f16` round-trip
//! ([`f16r`]) on known values, ties-to-even, idempotence and monotonicity; the affine column
//! quantizer ([`quant_c_u8`]) error bound, self-requantization and refusals; the hierarchical
//! row-scale quantizer ([`quant_r_hier`]) bitwise reconstruction from the stored codes, the
//! RTN-on-row-grid error bound, the bit-count formula and refusals. Grade `tested`.
//!
//! Mutation control (see the lane return): the `63.0` grid level in [`quant_r_hier`] changed to
//! `64.0` turns `quant_r_hier_error_within_rtn_bound` red.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::unreadable_literal,
    reason = "the tests compare `f64` carriers bit-for-bit (the whole point of the port), index the \
              row-major `[o, g]` buffers by `i/j/k`, and name the carriers `r_lo`/`r_sp`/`rq`/`u6` \
              as the reference does"
)]

use apeiron::learn::fold::fit::quant::{f16r, quant_c_u8, quant_r_hier, QuantError};

/// Deterministic xorshift64 PRNG, uniform in `[-1, 1)`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64 / 9_007_199_254_740_992.0) * 2.0 - 1.0
    }
}

/// Row-major `[o, g]` test matrix: varied row means and deviations, including a constant row.
fn make_rows(o: usize, g: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let means = [1.0_f64, -2.0, 0.5, 1.5, -0.75, 2.0, 0.25, -1.25];
    let mut r = vec![0.0_f64; o * g];
    for i in 0..o {
        let mean = means[i % means.len()];
        let scale = if i == 0 { 0.0 } else { 0.3 + 0.1 * (i as f64) };
        for j in 0..g {
            r[i * g + j] = mean + scale * rng.next_f64();
        }
    }
    r
}

/// Torch `quantile(q)` with linear interpolation, for the test's independent reference.
fn quantile(values: &[f64], q: f64) -> f64 {
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = q * (n as f64 - 1.0);
    let lo = idx.floor();
    let lo_i = lo as usize;
    sorted[lo_i] + (sorted[lo_i + 1] - sorted[lo_i]) * (idx - lo_i as f64)
}

/// Independent reference: squared error of row `i` on the `(du = 0, sp = sp_max)` RTN grid.
fn ref_row_err(r: &[f64], o: usize, g: usize, i: usize) -> f64 {
    let rm: Vec<f64> = (0..o)
        .map(|k| (0..g).map(|j| r[k * g + j]).sum::<f64>() / (g as f64))
        .collect();
    let mu_min = f16r(rm.iter().copied().fold(f64::INFINITY, f64::min));
    let mu_max = f16r(rm.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let mu_span = (mu_max - mu_min).max(1e-12);
    let u6_base = (((rm[i] - mu_min) / mu_span) * 63.0)
        .round_ties_even()
        .clamp(0.0, 63.0);
    let dev: Vec<f64> = (0..o)
        .map(|k| {
            let mx = (0..g)
                .map(|j| (r[k * g + j] - rm[k]).abs())
                .fold(0.0, f64::max);
            mx / rm[k].abs().max(1e-12)
        })
        .collect();
    let sp_min = f16r(quantile(&dev, 0.05).max(1e-4));
    let sp_max = f16r(
        dev.iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(sp_min + 1e-4),
    );
    let sp = sp_min + ((1.0 / 3.0) * 3.0) * (sp_max - sp_min);
    let mu = mu_min + (u6_base / 63.0) * mu_span;
    let r_lo = mu * (1.0 - sp);
    let r_hi = mu * (1.0 + sp);
    let r_sp = (r_hi - r_lo).max(1e-12);
    let mut err = 0.0_f64;
    for j in 0..g {
        let x = r[i * g + j];
        let k = (((x - r_lo) / r_sp) * 15.0)
            .round_ties_even()
            .clamp(0.0, 15.0);
        let rq = r_lo + (k / 15.0) * r_sp;
        let d = x - rq;
        err += d * d;
    }
    err
}

/// `f16r` on the named binary16 landmarks: exact normals, the largest finite value, overflow to
/// `±∞`, the smallest normal and subnormal, underflow to `0`, and the two ties-to-even cases.
#[test]
fn f16r_known_values() {
    assert_eq!(f16r(1.0), 1.0);
    assert_eq!(f16r(65504.0), 65504.0);
    assert!(f16r(65520.0).is_infinite() && f16r(65520.0) > 0.0);
    assert!(f16r(-65520.0).is_infinite() && f16r(-65520.0) < 0.0);
    assert_eq!(f16r(2f64.powi(-14)), 2f64.powi(-14));
    assert_eq!(f16r(6.1035156e-5), 2f64.powi(-14));
    assert_eq!(f16r(2f64.powi(-24)), 2f64.powi(-24));
    assert_eq!(f16r(5.96e-8), 2f64.powi(-24));
    assert_eq!(f16r(1e-8), 0.0);
    assert_eq!(f16r(1.0 + 2f64.powi(-11)), 1.0);
    assert_eq!(f16r(1.0 + 3.0 * 2f64.powi(-11)), 1.0 + 2f64.powi(-9));
    assert_eq!(f16r(-0.0_f64).to_bits(), (-0.0_f64).to_bits());
}

/// `f16r` is idempotent on its own image and nondecreasing over a sorted sweep.
#[test]
fn f16r_idempotent_and_monotone() {
    let mut rng = Rng::new(11);
    for _ in 0..500 {
        let v = rng.next_f64() * 1.0e3;
        let h = f16r(v);
        assert_eq!(f16r(h).to_bits(), h.to_bits(), "not idempotent at {v}");
    }
    let mut sorted: Vec<f64> = (0..400).map(|i| (i as f64 - 200.0) * 0.5).collect();
    sorted.sort_by(f64::total_cmp);
    let mut prev = f64::NEG_INFINITY;
    for &v in &sorted {
        let h = f16r(v);
        assert!(h >= prev, "f16r not monotone at {v}");
        prev = h;
    }
}

/// `quant_c_u8`: per-column error within half a grid step plus the `f16` endpoint slack, and the
/// exact `bits · n + 32` bit count.
#[test]
fn quant_c_u8_error_and_bits() {
    let mut rng = Rng::new(7);
    let c: Vec<f64> = (0..200).map(|_| rng.next_f64() * 3.0 - 1.0).collect();
    let bits = 8_u32;
    let (out, total) = quant_c_u8(&c, bits).expect("non-empty input");
    let true_min = c.iter().copied().fold(f64::INFINITY, f64::min);
    let true_max = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let c_min = f16r(true_min);
    let c_max = f16r(true_max);
    let span = (c_max - c_min).max(1e-12);
    let levels = f64::from((1_u32 << bits) - 1);
    let slack = (c_min - true_min).max(0.0) + (true_max - c_max).max(0.0);
    assert_eq!(total, u64::from(bits) * c.len() as u64 + 32);
    for i in 0..c.len() {
        let e = (out[i] - c[i]).abs();
        assert!(e <= span / (2.0 * levels) + slack + 1e-12, "col {i}: {e}");
    }
}

/// `quant_c_u8` reproduces its own output bit-for-bit under re-quantization.
#[test]
fn quant_c_u8_requantises_to_itself() {
    let mut rng = Rng::new(23);
    let c: Vec<f64> = (0..150).map(|_| rng.next_f64() * 2.0 + 0.5).collect();
    let (out, _) = quant_c_u8(&c, 8).expect("non-empty input");
    let (out2, _) = quant_c_u8(&out, 8).expect("non-empty input");
    for i in 0..out.len() {
        assert_eq!(out[i].to_bits(), out2[i].to_bits(), "column {i}");
    }
}

/// `quant_c_u8` refusals: empty input and an out-of-range bit width.
#[test]
fn quant_c_u8_refusals() {
    assert!(matches!(quant_c_u8(&[], 8), Err(QuantError::Empty)));
    assert!(matches!(quant_c_u8(&[1.0], 0), Err(QuantError::BadBits)));
    assert!(matches!(quant_c_u8(&[1.0], 33), Err(QuantError::BadBits)));
}

/// `quant_r_hier` reconstructs `rq` bit-for-bit from the stored per-row grid and 4-bit codes.
#[test]
fn quant_r_hier_reconstructs_bitwise() {
    let (o, g) = (6, 32);
    let r = make_rows(o, g, 3);
    let h = quant_r_hier(&r, o, g).expect("well-shaped input");
    let rec = h.reconstruct();
    assert_eq!(rec.len(), h.rq.len());
    for i in 0..rec.len() {
        assert_eq!(rec[i].to_bits(), h.rq[i].to_bits(), "element {i}");
        // The decode is the stored formula `r_lo + (k / 15) * r_sp`.
        let row = i / g;
        let k = f64::from(h.k[i]);
        let direct = h.r_lo[row] + (k / 15.0) * h.r_sp[row];
        assert_eq!(rec[i].to_bits(), direct.to_bits());
    }
}

/// The chosen candidate's per-row squared error is no worse than the `(du = 0, sp_max)` RTN grid.
#[test]
fn quant_r_hier_error_within_rtn_bound() {
    let (o, g) = (6, 32);
    let r = make_rows(o, g, 29);
    let h = quant_r_hier(&r, o, g).expect("well-shaped input");
    for i in 0..o {
        let mut err = 0.0_f64;
        for j in 0..g {
            let d = r[i * g + j] - h.rq[i * g + j];
            err += d * d;
        }
        let ref_err = ref_row_err(&r, o, g, i);
        assert!(
            err <= ref_err + 1e-9 * (1.0 + ref_err),
            "row {i}: {err} > {ref_err}"
        );
    }
}

/// `quant_r_hier`: the bit-count formula and the storage-code ranges.
#[test]
fn quant_r_hier_bit_count_and_code_ranges() {
    let (o, g) = (4, 16);
    let r = make_rows(o, g, 101);
    let h = quant_r_hier(&r, o, g).expect("well-shaped input");
    assert_eq!(h.bits, (4 * o * g + 8 * o + 64) as u64);
    assert_eq!(h.u6.len(), o);
    assert_eq!(h.sp_idx.len(), o);
    assert_eq!(h.k.len(), o * g);
    assert!(h.u6.iter().all(|&v| v <= 63), "u6 out of range");
    assert!(h.sp_idx.iter().all(|&v| v <= 3), "sp_idx out of range");
    assert!(h.k.iter().all(|&v| v <= 15), "k out of range");
}

/// `quant_r_hier` refusals: empty and ragged shapes.
#[test]
fn quant_r_hier_refusals() {
    assert!(matches!(quant_r_hier(&[], 0, 4), Err(QuantError::Empty)));
    assert!(matches!(quant_r_hier(&[], 0, 0), Err(QuantError::Empty)));
    assert!(matches!(
        quant_r_hier(&[1.0, 2.0], 1, 3),
        Err(QuantError::ShapeMismatch)
    ));
    assert!(matches!(
        quant_r_hier(&[1.0, 2.0, 3.0, 4.0], 2, 3),
        Err(QuantError::ShapeMismatch)
    ));
}
