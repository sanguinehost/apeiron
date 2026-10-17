//! `tests/learn_fold_quant.rs` — the one-bit fold-transfer carriers: the `f16` round-trip
//! ([`f16r`]) on known values, ties-to-even, idempotence and monotonicity; the affine column
//! quantizer ([`quant_c_u8`]) error bound, self-requantization and refusals; the hierarchical
//! row-scale quantizer ([`quant_r_hier`]) reconstruction from the stored codes, the RTN-on-row-grid
//! error bound, the bit-count formula and refusals; and the Q32.32 bodies against the pre-migration
//! `f64` bodies copied below as `ref_*` oracles with derived bounds. Grade `tested`.
//!
//! Mutation control (see the lane return): the `63` divisor of the 6-bit mean grid in `hier_header`
//! changed to `64` turns `quant_r_hier_agrees_with_the_f64_oracle` red (seed 29, "6-bit mean
//! codes"); restoring it is green.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::unreadable_literal,
    reason = "the tests compare the Q32.32 carriers against f64 oracles (the whole point of the \
              port), index the row-major `[o, g]` buffers by `i/j/k`, and name the carriers \
              `r_lo`/`r_sp`/`rq`/`u6` as the reference does"
)]

use apeiron::format::quant::f16_to_f32;
use apeiron::learn::fold::fit::quant::{f16r, quant_c_u8, quant_r_hier, QuantError};
use apeiron::learn::solve::q32::{from_grid, to_grid, GRID_BOUND};

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

/// The f64 fixture as Q32.32 words: the test's own data crosses to the carrier once, exactly as
/// the `src` callers do.
fn words(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| to_grid(v, GRID_BOUND).expect("fixture on the grid"))
        .collect()
}

/// The word grid as reals, for the oracle-bound comparisons.
fn reals(x: &[i64]) -> Vec<f64> {
    x.iter().map(|&w| from_grid(w)).collect()
}

/// The migrated [`f16r`] at an `f64` value of the test's own fixture: grid it, round, read back.
fn h16(x: f64) -> f64 {
    from_grid(f16r(to_grid(x, GRID_BOUND).expect("grid")).expect("in binary16 range"))
}

/// The largest binary16 spacing over `values`: one representable step of the same magnitude is at
/// most `2⁻¹⁰·max|v|`. This is the endpoint disagreement between a single rounding from the exact
/// grid word and the `f64 → f32 → f16` double rounding of the `f16r` oracle.
fn f16_ulp(values: &[f64]) -> f64 {
    let max = values.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    2f64.powi(-10) * max
}

// ---------------------------------------------------------------------------------------------
// Pre-migration `f64` bodies, copied verbatim from the retired `quant.rs` as the agreement oracle.
// ---------------------------------------------------------------------------------------------

/// `f32` → nearest-even binary16 bits (the retired `quant::f16_bits`).
fn ref_f16_bits(x: f32) -> u16 {
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
        let r = (f32::from_bits(b & 0x7FFF_FFFF) * 16_777_216.0).round_ties_even() as u32;
        return sign | r as u16;
    }
    let half = ((he as u32) << 10) | (m >> 13);
    let rem = m & 0x1FFF;
    sign | (half + u32::from(rem > 0x1000 || (rem == 0x1000 && half & 1 == 1))) as u16
}

/// The retired `f16r`.
fn ref_f16r(x: f64) -> f64 {
    f64::from(f16_to_f32(ref_f16_bits(x as f32)))
}

/// The retired `quantile_linear`.
fn ref_quantile_linear(values: &[f64], q: f64) -> f64 {
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

/// The retired `quant_c_u8`.
fn ref_quant_c_u8(c: &[f64], bits: u32) -> (Vec<f64>, u64) {
    let n = c.len() as u64;
    let total = n * u64::from(bits) + 32;
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
    let c_min = ref_f16r(lo);
    let c_max = ref_f16r(hi);
    let span = (c_max - c_min).max(1e-12);
    let levels = f64::from(((1_u64 << bits) - 1) as u32);
    let mut out = vec![0.0_f64; c.len()];
    for (i, &x) in c.iter().enumerate() {
        let q = (((x - c_min) / span) * levels)
            .round_ties_even()
            .clamp(0.0, levels);
        out[i] = c_min + (q / levels) * span;
    }
    (out, total)
}

/// The retired `hier_header` + candidate search + decode, one `f64` body returning the same fields.
///
/// The pre-migration `RHier` on reals; the shipped `RHier` is word-carried, so the oracle keeps its
/// own `f64` record.
struct RefHier {
    rq: Vec<f64>,
    bits: u64,
    k: Vec<u8>,
    u6: Vec<u8>,
    sp_idx: Vec<u8>,
}

fn ref_quant_r_hier(r: &[f64], o: usize, g: usize) -> RefHier {
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
    let mu_min = ref_f16r(mu_lo);
    let mu_max = ref_f16r(mu_hi);
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
    let sp_min = ref_f16r(ref_quantile_linear(&dev, 0.05).max(1e-4));
    let dev_max = dev.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sp_max = ref_f16r(dev_max.max(sp_min + 1e-4));
    let step = 1.0_f64 / 3.0;
    let mut sp_levels = [0.0_f64; 4];
    for (s, level) in sp_levels.iter_mut().enumerate() {
        *level = sp_min + (step * s as f64) * (sp_max - sp_min);
    }

    let n = o * g;
    let mut best_err = vec![f64::INFINITY; o];
    let mut best_u6 = vec![0_u8; o];
    let mut best_sp_idx = vec![0_u8; o];
    let mut best_k = vec![0_u8; n];
    let mut best_rq = r.to_vec();
    for &du in &[-1.0_f64, 0.0, 1.0] {
        for (i, &base) in u6_base.iter().enumerate() {
            let u6 = (base + du).clamp(0.0, 63.0);
            let mu = mu_min + (u6 / 63.0) * mu_span;
            for (s_idx, &sp) in sp_levels.iter().enumerate() {
                let r_lo = mu * (1.0 - sp);
                let r_sp = ((mu * (1.0 + sp)) - r_lo).max(1e-12);
                let mut err = 0.0_f64;
                let mut k_row = vec![0_u8; g];
                let mut rq_row = vec![0.0_f64; g];
                for j in 0..g {
                    let x = r[i * g + j];
                    let kf = (((x - r_lo) / r_sp) * 15.0)
                        .round_ties_even()
                        .clamp(0.0, 15.0);
                    let rq = r_lo + (kf / 15.0) * r_sp;
                    k_row[j] = kf as u8;
                    rq_row[j] = rq;
                    let d = x - rq;
                    err += d * d;
                }
                if err < best_err[i] {
                    best_err[i] = err;
                    best_u6[i] = u6 as u8;
                    best_sp_idx[i] = s_idx as u8;
                    for j in 0..g {
                        best_k[i * g + j] = k_row[j];
                        best_rq[i * g + j] = rq_row[j];
                    }
                }
            }
        }
    }
    RefHier {
        rq: best_rq,
        bits: 4 * n as u64 + 8 * o as u64 + 64,
        k: best_k,
        u6: best_u6,
        sp_idx: best_sp_idx,
    }
}

/// Independent reference: squared error of row `i` on the `(du = 0, sp = sp_max)` RTN grid.
fn ref_row_err(r: &[f64], o: usize, g: usize, i: usize) -> f64 {
    let rm: Vec<f64> = (0..o)
        .map(|k| (0..g).map(|j| r[k * g + j]).sum::<f64>() / (g as f64))
        .collect();
    let mu_min = ref_f16r(rm.iter().copied().fold(f64::INFINITY, f64::min));
    let mu_max = ref_f16r(rm.iter().copied().fold(f64::NEG_INFINITY, f64::max));
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
    let sp_min = ref_f16r(ref_quantile_linear(&dev, 0.05).max(1e-4));
    let sp_max = ref_f16r(
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

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

/// `f16r` on the named binary16 landmarks: exact normals, the largest finite value, the `2¹⁶`
/// overflow the grid cannot carry, the smallest normal and subnormal, underflow to `0`, and the two
/// ties-to-even cases. The carrier has no signed zero: `−0.0` is the word `0`.
#[test]
fn f16r_known_values() {
    let w = |x: f64| to_grid(x, GRID_BOUND).expect("grid");
    assert_eq!(f16r(w(1.0)), Ok(w(1.0)));
    assert_eq!(f16r(w(65504.0)), Ok(w(65504.0)));
    assert_eq!(f16r(w(65520.0)), Err(QuantError::LengthOverflow));
    assert_eq!(f16r(w(-65520.0)), Err(QuantError::LengthOverflow));
    assert_eq!(f16r(w(2f64.powi(-14))), Ok(w(2f64.powi(-14))));
    assert_eq!(f16r(w(6.1035156e-5)), Ok(w(2f64.powi(-14))));
    assert_eq!(f16r(w(2f64.powi(-24))), Ok(w(2f64.powi(-24))));
    assert_eq!(f16r(w(5.96e-8)), Ok(w(2f64.powi(-24))));
    assert_eq!(f16r(w(1e-8)), Ok(0));
    assert_eq!(f16r(w(1.0 + 2f64.powi(-11))), Ok(w(1.0)));
    assert_eq!(
        f16r(w(1.0 + 3.0 * 2f64.powi(-11))),
        Ok(w(1.0 + 2f64.powi(-9)))
    );
    assert_eq!(f16r(0), Ok(0));
}

/// `f16r` is idempotent on its own image and nondecreasing over a sorted sweep.
#[test]
fn f16r_idempotent_and_monotone() {
    let mut rng = Rng::new(11);
    for _ in 0..500 {
        let v = rng.next_f64() * 1.0e3;
        let w = to_grid(v, GRID_BOUND).expect("grid");
        let h = f16r(w).expect("in binary16 range");
        assert_eq!(f16r(h), Ok(h), "not idempotent at {v}");
    }
    let mut sorted: Vec<f64> = (0..400).map(|i| (i as f64 - 200.0) * 0.5).collect();
    sorted.sort_by(f64::total_cmp);
    let mut prev = i64::MIN;
    for &v in &sorted {
        let h = f16r(to_grid(v, GRID_BOUND).expect("grid")).expect("in range");
        assert!(h >= prev, "f16r not monotone at {v}");
        prev = h;
    }
}

/// `quant_c_u8`: per-column error within half a grid step plus the endpoint slack and one binary16
/// ulp of the Q32.32-vs-f64 endpoint rounding; the exact `bits · n + 32` bit count.
#[test]
fn quant_c_u8_error_and_bits() {
    let mut rng = Rng::new(7);
    let c: Vec<f64> = (0..200).map(|_| rng.next_f64() * 3.0 - 1.0).collect();
    let bits = 8_u32;
    let (out, total) = quant_c_u8(&words(&c), bits).expect("non-empty input");
    let out = reals(&out);
    let true_min = c.iter().copied().fold(f64::INFINITY, f64::min);
    let true_max = c.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let c_min = h16(true_min);
    let c_max = h16(true_max);
    let span = (c_max - c_min).max(1e-12);
    let levels = f64::from((1_u32 << bits) - 1);
    let slack = (c_min - true_min).max(0.0) + (true_max - c_max).max(0.0);
    assert_eq!(total, u64::from(bits) * c.len() as u64 + 32);
    // Single-rounding endpoints may sit one binary16 ulp beyond the oracle's, so the grid point
    // moves by at most half a step plus that ulp; the Q32 quantum adds `2⁻³²`.
    let tol = span / (2.0 * levels) + slack + f16_ulp(&c) + 2f64.powi(-32);
    for i in 0..c.len() {
        let e = (out[i] - c[i]).abs();
        assert!(e <= tol, "col {i}: {e} > {tol}");
    }
}

/// `quant_c_u8` reproduces a grid value under re-quantization to within the same bound: the second
/// pass re-derives the endpoint words from the output extremes, which move by at most one binary16
/// ulp, so the grid itself moves by at most that much.
#[test]
fn quant_c_u8_requantises_to_itself() {
    let mut rng = Rng::new(23);
    let c: Vec<f64> = (0..150).map(|_| rng.next_f64() * 2.0 + 0.5).collect();
    let (out, _) = quant_c_u8(&words(&c), 8).expect("non-empty input");
    let (out2, _) = quant_c_u8(&out, 8).expect("non-empty input");
    let out = reals(&out);
    let out2 = reals(&out2);
    let tol = 2.0 * f16_ulp(&out) + 4.0 * 2f64.powi(-32);
    for i in 0..out.len() {
        let e = (out[i] - out2[i]).abs();
        assert!(e <= tol, "column {i}: {e} > {tol}");
    }
}

/// `quant_c_u8` agrees with the retired f64 body on fixed LCG input within the derived endpoint
/// bound: one binary16 ulp `2⁻¹⁰·max|c|` per endpoint and one Q32 quantum per rounding.
#[test]
fn quant_c_u8_agrees_with_the_f64_oracle() {
    let mut rng = Rng::new(0x5EED_1234);
    let c: Vec<f64> = (0..180).map(|_| rng.next_f64() * 2.5 - 0.5).collect();
    let (out, out_bits) = quant_c_u8(&words(&c), 7).expect("non-empty input");
    let (want, want_bits) = ref_quant_c_u8(&c, 7);
    assert_eq!(out_bits, want_bits);
    let tol = 3.0 * f16_ulp(&c) + 2f64.powi(-31);
    for i in 0..c.len() {
        let e = (from_grid(out[i]) - want[i]).abs();
        assert!(e <= tol, "col {i}: {e} > {tol}");
    }
}

/// `quant_c_u8` refusals: empty input, an out-of-range bit width, and an f16 endpoint at the binary16
/// infinity the grid cannot carry. The non-finite / out-of-grid arm moved to the caller's
/// `to_grid` boundary and is pinned here on that map.
#[test]
fn quant_c_u8_refusals() {
    assert!(matches!(quant_c_u8(&[], 8), Err(QuantError::Empty)));
    assert!(matches!(quant_c_u8(&[1], 0), Err(QuantError::BadBits)));
    assert!(matches!(quant_c_u8(&[1], 33), Err(QuantError::BadBits)));
    assert_eq!(to_grid(f64::NAN, GRID_BOUND), None);
    assert_eq!(to_grid(1e12, GRID_BOUND), None);
    let big = to_grid(65520.0, GRID_BOUND).expect("grid");
    assert!(matches!(
        quant_c_u8(&[big], 8),
        Err(QuantError::LengthOverflow)
    ));
}

/// `quant_r_hier` reconstructs `rq` from the stored per-row grid and 4-bit codes: the stored fields
/// are exact Q32.32 words, so the decode is bit-identical, and an f64 evaluation of the stored
/// formula agrees within the two roundings of that formula (two Q32 quanta).
#[test]
fn quant_r_hier_reconstructs_bitwise() {
    let (o, g) = (6, 32);
    let r = make_rows(o, g, 3);
    let h = quant_r_hier(&words(&r), o, g).expect("well-shaped input");
    let rec = h.reconstruct();
    assert_eq!(rec.len(), h.rq.len());
    for i in 0..rec.len() {
        assert_eq!(rec[i], h.rq[i], "element {i}");
        // The decode is the stored formula `r_lo + (k / 15) * r_sp`; the f64 evaluation rounds the
        // quotient and the product-add, the Q32 decode rounds the product once.
        let row = i / g;
        let k = f64::from(h.k[i]);
        let direct = from_grid(h.r_lo[row]) + (k / 15.0) * from_grid(h.r_sp[row]);
        let e = (from_grid(rec[i]) - direct).abs();
        assert!(e <= 2.0 * 2f64.powi(-32), "element {i}: {e} > 2 quanta");
    }
}

/// The Q32.32 body agrees with the retired f64 body on fixed LCG input: the stored codes are equal
/// and every reconstructed value moves by at most two binary16 ulps plus two Q32 quanta.
#[test]
fn quant_r_hier_agrees_with_the_f64_oracle() {
    let (o, g) = (8, 24);
    for seed in [29_u64, 101, 0xABCD_1234] {
        let r = make_rows(o, g, seed);
        let got = quant_r_hier(&words(&r), o, g).expect("well-shaped input");
        let want = ref_quant_r_hier(&r, o, g);
        assert_eq!(got.bits, want.bits, "seed {seed}: bit count");
        assert_eq!(got.u6, want.u6, "seed {seed}: 6-bit mean codes");
        assert_eq!(got.sp_idx, want.sp_idx, "seed {seed}: spread indices");
        assert_eq!(got.k, want.k, "seed {seed}: 4-bit codes");
        let tol = 2.0 * f16_ulp(&r) + 4.0 * 2f64.powi(-32);
        for i in 0..got.rq.len() {
            let e = (from_grid(got.rq[i]) - want.rq[i]).abs();
            assert!(e <= tol, "seed {seed} element {i}: {e} > {tol}");
        }
    }
}

/// The chosen candidate's per-row squared error is no worse than the `(du = 0, sp_max)` RTN grid:
/// 12 candidates include that grid, and the Q32 squared error differs from the f64 oracle by at
/// most `2·|d|·δ·g` with `δ = 2·2⁻³²` on the grid value, `|d| ≤ 4` and `g ≤ 32`, i.e. `5·10⁻⁹`.
#[test]
fn quant_r_hier_error_within_rtn_bound() {
    let (o, g) = (6, 32);
    let r = make_rows(o, g, 29);
    let h = quant_r_hier(&words(&r), o, g).expect("well-shaped input");
    for i in 0..o {
        let mut err = 0.0_f64;
        for j in 0..g {
            let d = r[i * g + j] - from_grid(h.rq[i * g + j]);
            err += d * d;
        }
        let ref_err = ref_row_err(&r, o, g, i);
        assert!(
            err <= ref_err + 1e-7 * (1.0 + ref_err),
            "row {i}: {err} > {ref_err}"
        );
    }
}

/// `quant_r_hier`: the bit-count formula and the storage-code ranges.
#[test]
fn quant_r_hier_bit_count_and_code_ranges() {
    let (o, g) = (4, 16);
    let r = make_rows(o, g, 101);
    let h = quant_r_hier(&words(&r), o, g).expect("well-shaped input");
    assert_eq!(h.bits, (4 * o * g + 8 * o + 64) as u64);
    assert_eq!(h.u6.len(), o);
    assert_eq!(h.sp_idx.len(), o);
    assert_eq!(h.k.len(), o * g);
    assert!(h.u6.iter().all(|&v| v <= 63), "u6 out of range");
    assert!(h.sp_idx.iter().all(|&v| v <= 3), "sp_idx out of range");
    assert!(h.k.iter().all(|&v| v <= 15), "k out of range");
}

/// `quant_r_hier` refusals: empty, ragged shapes, and an f16 row-mean endpoint at the binary16
/// infinity the grid cannot carry.
#[test]
fn quant_r_hier_refusals() {
    assert!(matches!(quant_r_hier(&[], 0, 4), Err(QuantError::Empty)));
    assert!(matches!(quant_r_hier(&[], 0, 0), Err(QuantError::Empty)));
    assert!(matches!(
        quant_r_hier(&[1, 2], 1, 3),
        Err(QuantError::ShapeMismatch)
    ));
    assert!(matches!(
        quant_r_hier(&[1, 2, 3, 4], 2, 3),
        Err(QuantError::ShapeMismatch)
    ));
    let big = to_grid(65520.0, GRID_BOUND).expect("grid");
    assert!(matches!(
        quant_r_hier(&[big], 1, 1),
        Err(QuantError::LengthOverflow)
    ));
}

/// FNV-1a over a word sequence, the bit-identity witness's digest.
fn words_hash(ws: &[i64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &w in ws {
        for b in w.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

#[test]
fn quant_words_are_bit_identical_to_the_recorded_f64_wrapper() {
    // The digests were recorded from the retired `&[f64]` API on the same seeded inputs before the
    // flip, where the wrapper was `to_grid` -> word body -> `from_grid` and its outputs are exact
    // grid words. Equality here is word-for-word, so the API flip moves no bit of either carrier.
    let golden = [
        (7_u64, 0x92d2_87d4_0692_4754, 0xe8cd_62c1_3455_4f5a),
        (23, 0x98b2_e561_84ce_c3ef, 0x1a03_11e9_b942_f696),
        (0x51ed, 0x8f57_fffb_6bf5_06c5, 0x6a71_9ddb_fa12_4dd8),
    ];
    for (seed, want_c, want_r) in golden {
        let mut rng = Rng::new(seed);
        let c: Vec<f64> = (0..64).map(|_| rng.next_f64() * 2.5 - 0.5).collect();
        let (q, _) = quant_c_u8(&words(&c), 7).expect("quant_c_u8");
        assert_eq!(words_hash(&q), want_c, "seed {seed}: quant_c_u8 words");

        let h = quant_r_hier(&words(&make_rows(6, 16, seed)), 6, 16).expect("quant_r_hier");
        let mut mix: Vec<i64> = h.rq.iter().chain(&h.r_lo).chain(&h.r_sp).copied().collect();
        mix.extend(h.k.iter().map(|&v| i64::from(v)));
        mix.extend(h.u6.iter().map(|&v| i64::from(v)));
        mix.extend(h.sp_idx.iter().map(|&v| i64::from(v)));
        assert_eq!(words_hash(&mix), want_r, "seed {seed}: quant_r_hier words");
    }
}
