//! The shared `f64` ↔ Q32.32 boundary (rounding, refusals, the power-of-two fill and panel shift)
//! and the pure-integer carriers the solve primitives reuse: the floor-log2, the round-to-nearest-
//! even right shift, the `Fx`/`Bfp` block-floating-point normalisers, the `SpdQ` diagonal
//! equilibration and the single-rounding binary16 `round_f16`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::float_cmp,
    reason = "the tests build the f16 reference through f64 and compare exact f16-representable \
              values, and index row-major buffers by i/j; the pedantic lints stay on for the \
              crate's non-test code."
)]

use apeiron::format::quant::f16_to_f32;
use apeiron::learn::solve::q32::{
    fill_scale, from_grid, ilog2_pos, narrow, panel_shift, round_f16, round_q32_even, rshift_rne,
    to_grid, Bfp, Fx, Q32Error, SpdQ, GRID_BOUND, GRID_SCALE, INPUT_BOUND,
};

#[test]
fn narrow_is_the_checked_i64_cast_at_both_edges() {
    assert_eq!(narrow(i128::from(i64::MAX)), Some(i64::MAX));
    assert_eq!(narrow(i128::from(i64::MIN)), Some(i64::MIN));
    assert_eq!(narrow(i128::from(i64::MAX) + 1), None);
    assert_eq!(narrow(i128::from(i64::MIN) - 1), None);
    // The deleted copies spelled the same predicate `i64::try_from(v).ok()` (`plane`, `parts`),
    // `…map_err(..)` (`lw`) and `i64::MAX as i128` (`dense`); this pins the survivor to that form.
    for &v in &[
        i128::MIN,
        i128::from(i64::MIN),
        i128::from(i64::MIN) - 1,
        -1,
        0,
        1,
        i128::from(i64::MAX),
        i128::from(i64::MAX) + 1,
        i128::MAX,
    ] {
        assert_eq!(narrow(v), i64::try_from(v).ok(), "v={v}");
    }
}

#[test]
fn round_trip_is_within_half_ulp_of_the_grid() {
    for &x in &[0.0, 1.0, -1.0, 0.1, -3.25e-7, 1234.5678, 15.999] {
        let q = to_grid(x, GRID_BOUND).expect("in range");
        assert!((from_grid(q) - x).abs() <= 0.5 / GRID_SCALE, "x={x}");
    }
}

#[test]
fn refuses_non_finite_and_out_of_bound() {
    for &x in &[
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        16.0,
        -16.0,
        1e30,
    ] {
        assert_eq!(to_grid(x, INPUT_BOUND), None, "x={x}");
    }
    assert!(to_grid(16.0, GRID_BOUND).is_some());
    assert_eq!(to_grid(GRID_BOUND, GRID_BOUND), None);
    assert_eq!(to_grid(-GRID_BOUND, GRID_BOUND), None);
}

#[test]
fn fill_scale_is_an_exact_power_of_two_filling_the_range() {
    assert!((fill_scale(0.0) - 1.0).abs() < f64::EPSILON);
    for &m in &[1e-12, 1e-3, 0.5, 1.0, 7.0, 1e3, 1e9, 3e12] {
        let s = fill_scale(m);
        assert_eq!(
            s.to_bits() & ((1u64 << 52) - 1),
            0,
            "m={m}: not a power of two"
        );
        let scaled = m * s;
        assert!(
            scaled > 1_073_741_823.0 && scaled <= 2_147_483_647.0,
            "m={m}: scaled={scaled}"
        );
        assert!(to_grid(scaled, GRID_BOUND).is_some());
    }
}

#[test]
fn panel_shift_brings_the_maximum_under_eight() {
    assert_eq!(panel_shift(0.0), 0);
    assert_eq!(panel_shift(8.0), 0);
    for &m in &[8.5, 16.0, 100.0, 1e6] {
        let k = panel_shift(m);
        assert!(k >= 1 && m * 0.5f64.powi(k) <= 8.0, "m={m} k={k}");
        assert!(m * 0.5f64.powi(k - 1) > 8.0, "m={m} k={k} not least");
    }
}

/// Deterministic 64-bit LCG (Numerical Recipes constants): the tests' only randomness source.
struct Lcg(u64);

impl Lcg {
    /// A generator seeded with `seed`.
    fn new(seed: u64) -> Lcg {
        Lcg(seed)
    }

    /// The next 64-bit word.
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

/// A wide-magnitude signed sample: two `u64` words, a variable right shift and a random sign.
fn sample_i128(lcg: &mut Lcg) -> i128 {
    let hi = u128::from(lcg.next() >> 11);
    let lo = u128::from(lcg.next());
    let mag = ((hi << 64) | lo) >> (lcg.next() % 60) as u32;
    if lcg.next() & 1 == 0 {
        mag as i128
    } else {
        -(mag as i128)
    }
}

#[test]
fn ilog2_pos_is_the_high_bit_index() {
    assert_eq!(ilog2_pos(0), None);
    assert_eq!(ilog2_pos(1), Some(0));
    assert_eq!(ilog2_pos(2), Some(1));
    assert_eq!(ilog2_pos(3), Some(1));
    assert_eq!(ilog2_pos(1_u128 << 127), Some(127));
    assert_eq!(ilog2_pos(u128::MAX), Some(127));
}

#[test]
fn rshift_rne_rounds_ties_to_even() {
    assert_eq!(rshift_rne(0, 0), Some(0));
    // s = 1: the half-integers round to the even neighbour
    assert_eq!(rshift_rne(1, 1), Some(0));
    assert_eq!(rshift_rne(-1, 1), Some(0));
    assert_eq!(rshift_rne(3, 1), Some(2));
    assert_eq!(rshift_rne(-3, 1), Some(-2));
    assert_eq!(rshift_rne(5, 1), Some(2));
    assert_eq!(rshift_rne(-5, 1), Some(-2));
    assert_eq!(rshift_rne(7, 1), Some(4));
    assert_eq!(rshift_rne(-7, 1), Some(-4));
    // non-ties
    assert_eq!(rshift_rne(5, 2), Some(1));
    assert_eq!(rshift_rne(6, 2), Some(2));
    assert_eq!(rshift_rne(-6, 2), Some(-2));
    // large shifts
    assert_eq!(rshift_rne(1_i128 << 40, 40), Some(1));
    assert_eq!(rshift_rne(3 * (1_i128 << 99), 100), Some(2));
    assert_eq!(rshift_rne(1_i128 << 99, 100), Some(0));
    assert_eq!(rshift_rne(1_i128 << 126, 126), Some(1));
    assert_eq!(rshift_rne(1, 127), None);
}

#[test]
fn round_q32_even_is_the_q64_64_narrowing() {
    assert_eq!(round_q32_even(0), Ok(0));
    assert_eq!(round_q32_even(1_i128 << 31), Ok(0));
    assert_eq!(round_q32_even(3 * (1_i128 << 31)), Ok(2));
    assert_eq!(round_q32_even(1_i128 << 62), Ok(1 << 30));
    assert_eq!(round_q32_even(i128::from(i64::MAX) << 32), Ok(i64::MAX));
    assert_eq!(
        round_q32_even((i128::from(i64::MAX) + 1) << 32),
        Err(Q32Error::Overflow)
    );
}

#[test]
fn fx_normalises_and_preserves_small_values() {
    assert_eq!(Fx::from_i128(0, 9), Ok(Fx { m: 0, e: 0 }));
    assert_eq!(Fx::from_i128(1234, -5), Ok(Fx { m: 1234, e: -5 }));
    assert_eq!(Fx::from_i128(-42, 7), Ok(Fx { m: -42, e: 7 }));
    // a 62-bit value is exact (shift 0)
    let exact = (1_i128 << 62) - 1;
    assert_eq!(
        Fx::from_i128(exact, 3),
        Ok(Fx {
            m: i64::MAX >> 1,
            e: 3
        })
    );
    let mut lcg = Lcg::new(0xdead_beef_1234_5678);
    for _ in 0..4000 {
        let v = sample_i128(&mut lcg);
        let e = (lcg.next() % 200) as i32 - 100;
        let fx = Fx::from_i128(v, e).expect("i64 range");
        if v == 0 {
            assert_eq!(fx, Fx { m: 0, e: 0 });
            continue;
        }
        let m = i128::from(fx.m);
        assert!(m.abs() < (1_i128 << 62), "v={v} e={e} m={m}");
        assert!(fx.e >= e, "v={v} e={e} e'={}", fx.e);
    }
}

#[test]
fn bfp_normalises_into_range() {
    // all-zero input returns zeros with `e` unchanged
    assert_eq!(
        Bfp::from_i128(&[0, 0, 0], 7),
        Ok(Bfp {
            m: vec![0, 0, 0],
            e: 7
        })
    );
    // a maximum already in [2^34, 2^35) is left exact (shift 0)
    let v = [i128::from((1_i64 << 34) + 3), -i128::from(1_i64 << 34), 1];
    assert_eq!(
        Bfp::from_i128(&v, -4),
        Ok(Bfp {
            m: vec![(1_i64 << 34) + 3, -(1_i64 << 34), 1],
            e: -4
        })
    );
    let mut lcg = Lcg::new(0x0123_4567_89ab_cdef);
    for _ in 0..500 {
        let n = 1 + (lcg.next() % 6) as usize;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push(sample_i128(&mut lcg));
        }
        let e = (lcg.next() % 200) as i32 - 100;
        let b = Bfp::from_i128(&v, e).expect("i64 range");
        let mut maxm: u128 = 0;
        for &m in &b.m {
            let a = if m < 0 {
                (-i128::from(m)) as u128
            } else {
                m as u128
            };
            if a > maxm {
                maxm = a;
            }
        }
        if maxm == 0 {
            assert_eq!(b.e, e);
        } else {
            assert!(
                (1_u128 << 34..1_u128 << 35).contains(&maxm),
                "n={n} e={e} maxm={maxm}"
            );
        }
    }
}

#[test]
fn spdq_equilibrates_and_round_trips() {
    let n = 6_usize;
    let mut h = vec![0_i64; n * n];
    for i in 0..n {
        let bit = 12 + 8 * i; // 12..=52: real 2^-20..=2^20
        h[i * n + i] = 1_i64 << bit;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let val = ((i as i64 + 1) * 7 - j as i64 * 3) * (i as i64 * 5 + 1);
            h[i * n + j] = val;
            h[j * n + i] = val;
        }
    }
    let s = SpdQ::from_q32(&h, n).expect("spd");
    for i in 0..n {
        let hh = s.hat[i * n + i];
        assert!((1_i64 << 32..1_i64 << 34).contains(&hh), "i={i} hh={hh}");
    }
    let back = s.to_q32().expect("invert");
    for i in 0..n {
        assert_eq!(back[i * n + i], h[i * n + i], "diagonal i={i}");
    }
    for idx in 0..n * n {
        let i = idx / n;
        let j = idx % n;
        let sum = s.d[i] + s.d[j];
        let diff = (i128::from(back[idx]) - i128::from(h[idx])).abs();
        if sum <= 0 {
            assert_eq!(diff, 0, "idx={idx}");
        } else {
            assert!(diff <= 1_i128 << sum, "idx={idx} diff={diff} sum={sum}");
        }
    }
}

#[test]
fn spdq_refuses_bad_shapes_and_symmetry() {
    assert_eq!(SpdQ::from_q32(&[], 0), Err(Q32Error::ShapeMismatch));
    assert_eq!(
        SpdQ::from_q32(&[1, 0, 0, 1], 3),
        Err(Q32Error::ShapeMismatch)
    );
    let zero_diag = [0_i64, 0, 0, 1];
    assert_eq!(SpdQ::from_q32(&zero_diag, 2), Err(Q32Error::NotSpd));
    let asym = [1_i64 << 32, 5, 7, 1_i64 << 32];
    assert_eq!(SpdQ::from_q32(&asym, 2), Err(Q32Error::Asymmetric));
}

#[test]
fn round_f16_is_identity_on_finite_binary16_patterns() {
    let mut finite = 0_u32;
    for bits in 0_u32..=0xFFFF {
        if bits & 0x7FFF >= 0x7C00 {
            continue; // infinity or NaN
        }
        let exp = (bits >> 10) & 0x1F;
        let frac = bits & 0x3FF;
        let mut x = if exp == 0 {
            if frac == 0 {
                Fx { m: 0, e: 0 }
            } else {
                Fx {
                    m: i64::from(frac),
                    e: -24,
                }
            }
        } else {
            Fx {
                m: 1024 + i64::from(frac),
                e: exp as i32 - 25,
            }
        };
        if bits >> 15 == 1 {
            x.m = -x.m;
        }
        let expected = x.clone();
        assert_eq!(round_f16(x), Ok(expected), "bits={bits:#06x}");
        finite += 1;
    }
    assert_eq!(finite, 63_488);
}

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

/// The retired `quant::f16r`: the `f64 → f32 → f16 → f64` double rounding.
fn ref_f16r(x: f64) -> f64 {
    f64::from(f16_to_f32(ref_f16_bits(x as f32)))
}

#[test]
fn round_f16_matches_the_f64_reference_on_24_bit_significands() {
    // The retired f64 reference, kept here: `f64 → f32 → f16` double rounding. A 24-bit
    // significand is exact in `f32`, so this is the single-rounding reference the migrated
    // `round_f16` must equal.
    let mut lcg = Lcg::new(0x9e37_79b9_7f4a_7c15);
    for _ in 0..20_000 {
        let m = (lcg.next() % (1_u64 << 24)) as i64;
        let m = if lcg.next() & 1 == 0 { m } else { -m };
        let e = (lcg.next() % 81) as i32 - 60; // -60..=20
        let x = Fx { m, e };
        let value = m as f64 * 2_f64.powi(e);
        let want = ref_f16r(value);
        match round_f16(x) {
            Ok(r) => {
                let got = r.m as f64 * 2_f64.powi(r.e);
                assert_eq!(got, want, "m={m} e={e}");
                assert!(want.is_finite(), "m={m} e={e}");
            }
            Err(Q32Error::Overflow) => assert!(want.is_infinite(), "m={m} e={e}"),
            Err(other) => panic!("unexpected {other:?}"),
        }
    }
}

#[test]
fn round_f16_boundary_ties_and_overflow() {
    // half of the smallest subnormal 2^-24: ties to even -> zero
    let zero = round_f16(Fx { m: 1, e: -25 }).expect("finite");
    assert_eq!(zero.m as f64 * 2_f64.powi(zero.e), 0.0);
    // 1.5 * 2^-24 rounds to 2 * 2^-24
    assert_eq!(round_f16(Fx { m: 3, e: -25 }), Ok(Fx { m: 2, e: -24 }));
    // the largest finite binary16 is the identity
    assert_eq!(round_f16(Fx { m: 65504, e: 0 }), Ok(Fx { m: 2047, e: 5 }));
    // 65519 rounds down to 65504; 65520 is the tie to infinity
    assert_eq!(round_f16(Fx { m: 65519, e: 0 }), Ok(Fx { m: 2047, e: 5 }));
    assert_eq!(round_f16(Fx { m: 65520, e: 0 }), Err(Q32Error::Overflow));
}
