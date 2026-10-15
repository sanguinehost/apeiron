//! IEEE 754 binary32 as a bit pattern, so the extraction can state float arithmetic as axioms.
//!
//! The pinned Aeneas translates no `f32`/`f64` arithmetic, comparison, constant or int-float cast
//! (`/home/socol/oc-tmp/escalation/FORM-FLOAT.verdict.md`). [`F32`] carries the 32 bits as a `u32`;
//! every operation that is a pure bit operation (`from_bits`, `to_bits`, `neg`, `abs`, `is_nan`,
//! `is_finite`) is ordinary extracted code. Every operation that rounds or orders is a function of
//! [`prim`], which the extraction treats as opaque; each has one entry in
//! `proof/boundary-axioms.txt` stating it over the Lean decoding `v : F32 → {±0, ±∞, NaN, ℚ}` and
//! `rne : ℚ → F32` (round to binary32 nearest-even).
//!
//! The operators are the thin wrappers at the bottom of this file, so call sites read as float code.
//! `PartialEq`/`PartialOrd` are not implemented: bitwise equality is not IEEE equality (`-0`, NaN), so
//! comparisons are the named [`F32::lt`], [`F32::le`], [`F32::gt`], [`F32::ge`], [`F32::eq_ieee`].
//!
//! # Trust boundary
//! The Lean axioms describe IEEE 754-2008 `roundTiesToEven`. The CPU and rustc match that
//! specification at grade `tested` only: the `prim_*` tests below compare each primitive with the
//! hardware result over a fixed set of normal, subnormal, zero, infinite and NaN patterns. MXCSR
//! (FTZ, DAZ, rounding mode) is process state outside the model; `mxcsr_is_default` checks it.

#![allow(
    clippy::inline_always,
    reason = "each wrapper is one instruction or one bit operation; a call frame costs more than the body"
)]

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A binary32 value, stored as its bit pattern.
///
/// `#[repr(transparent)]` over `u32`: a `&[F32]` is a `&[f32]` for the SIMD twins.
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct F32(pub u32);

/// A binary64 value, stored as its bit pattern. Config values (`rope_theta`, `rms_eps`) carry it so
/// the extraction sees an integer; the host `f64` arithmetic lives in [`prim`].
/// `PartialEq` is bitwise (`-0 != +0`, `NaN == NaN` on equal bits), as for a stored config word.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct F64(pub u64);

impl F64 {
    #[inline(always)]
    pub const fn from_bits(bits: u64) -> F64 {
        F64(bits)
    }

    #[inline(always)]
    pub const fn to_bits(self) -> u64 {
        self.0
    }

    /// Host conversion for tests, the GPU upload and the learn crate; no extracted body calls it.
    #[inline(always)]
    pub fn from_f64(v: f64) -> F64 {
        F64(v.to_bits())
    }

    /// Host conversion; see [`F64::from_f64`].
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}

const SIGN: u32 = 0x8000_0000;
const ABS: u32 = 0x7fff_ffff;
const EXP: u32 = 0x7f80_0000;

impl F32 {
    pub const ZERO: F32 = F32(0);
    pub const NEG_ZERO: F32 = F32(SIGN);
    pub const ONE: F32 = F32(0x3f80_0000);
    pub const NEG_ONE: F32 = F32(0xbf80_0000);
    pub const HALF: F32 = F32(0x3f00_0000);
    pub const TWO: F32 = F32(0x4000_0000);
    pub const INFINITY: F32 = F32(EXP);
    pub const NEG_INFINITY: F32 = F32(EXP | SIGN);
    pub const NAN: F32 = F32(0x7fc0_0000);
    pub const MAX: F32 = F32(0x7f7f_ffff);
    pub const MIN_POSITIVE: F32 = F32(0x0080_0000);
    /// `2^32`, the Q32.32 scale.
    pub const TWO32: F32 = F32(0x4f80_0000);

    #[inline(always)]
    pub const fn from_bits(bits: u32) -> F32 {
        F32(bits)
    }

    #[inline(always)]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// `|self|`: clears the sign bit.
    #[inline(always)]
    pub const fn abs(self) -> F32 {
        F32(self.0 & ABS)
    }

    /// `self` with the sign of `sign`.
    #[inline(always)]
    pub const fn copysign(self, sign: F32) -> F32 {
        F32((self.0 & ABS) | (sign.0 & SIGN))
    }

    #[inline(always)]
    pub const fn is_nan(self) -> bool {
        (self.0 & ABS) > EXP
    }

    #[inline(always)]
    pub const fn is_finite(self) -> bool {
        (self.0 & EXP) != EXP
    }

    /// `self * b + c` with one rounding.
    #[inline(always)]
    pub fn fma(self, b: F32, c: F32) -> F32 {
        prim::fma(self, b, c)
    }

    #[inline(always)]
    pub fn sqrt(self) -> F32 {
        prim::sqrt(self)
    }

    /// IEEE ordered `<`: false when either operand is NaN.
    #[inline(always)]
    pub fn lt(self, o: F32) -> bool {
        prim::lt(self, o)
    }

    #[inline(always)]
    pub fn le(self, o: F32) -> bool {
        prim::le(self, o)
    }

    #[inline(always)]
    pub fn gt(self, o: F32) -> bool {
        prim::lt(o, self)
    }

    #[inline(always)]
    pub fn ge(self, o: F32) -> bool {
        prim::le(o, self)
    }

    /// IEEE `==`: `+0 == -0`, NaN unequal to everything.
    #[inline(always)]
    pub fn eq_ieee(self, o: F32) -> bool {
        prim::eq(self, o)
    }

    /// IEEE maxNum: the other operand when one is NaN.
    #[inline(always)]
    pub fn max(self, o: F32) -> F32 {
        prim::max(self, o)
    }

    /// IEEE minNum: the other operand when one is NaN.
    #[inline(always)]
    pub fn min(self, o: F32) -> F32 {
        prim::min(self, o)
    }

    /// `rne(i)`.
    #[inline(always)]
    pub fn from_i64(i: i64) -> F32 {
        prim::from_i64(i)
    }

    /// Rust `as i64`: truncate toward zero, saturate, NaN to 0.
    #[inline(always)]
    pub fn to_i64_sat(self) -> i64 {
        prim::to_i64_sat(self)
    }

    /// Rust `as i32`: truncate toward zero, saturate, NaN to 0.
    #[inline(always)]
    pub fn to_i32_sat(self) -> i32 {
        prim::to_i32_sat(self)
    }

    /// The Q32.32 word `x / 2^32` as binary32: `rne(rne(x) / 2^32)`.
    #[inline(always)]
    pub fn from_q32(x: i64) -> F32 {
        prim::from_i64(x) / F32::TWO32
    }

    /// `round_half_away(self * 2^32)` as an `i64`; `None` for NaN, infinity, or a result outside
    /// `[-2^63, 2^63)`. Pure integer decode of the exponent and mantissa, no rounding in a wider
    /// float. Equals `(f64::from(v) * 2^32).round() as i64` on every finite binary32.
    ///
    /// Complexity: O(1) time and space (a fixed number of integer operations).
    #[inline]
    pub fn to_q32(self) -> Option<i64> {
        let bits = self.0;
        let e = (bits >> 23) & 0xff;
        if e == 0xff {
            return None;
        }
        let frac = bits & 0x007f_ffff;
        let m: u64 = if e == 0 {
            u64::from(frac)
        } else {
            u64::from(frac | 0x0080_0000)
        };
        let neg = (bits & SIGN) != 0;
        let ee = if e == 0 { 1 } else { e };
        // |value| * 2^32 = m * 2^(ee - 118).
        let mag: u128 = if ee >= 118 {
            let k = ee - 118;
            if k > 40 {
                return None;
            }
            u128::from(m) << k
        } else {
            let sh = 118 - ee;
            if sh > 24 {
                0
            } else {
                u128::from((m + (1u64 << (sh - 1))) >> sh)
            }
        };
        let lim: u128 = 1u128 << 63;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "mag < 2^63 on both branches that cast"
        )]
        if neg {
            match mag.cmp(&lim) {
                core::cmp::Ordering::Greater => None,
                core::cmp::Ordering::Equal => Some(i64::MIN),
                core::cmp::Ordering::Less => Some(-(mag as i64)),
            }
        } else if mag >= lim {
            None
        } else {
            Some(mag as i64)
        }
    }

    /// Round to the nearest integer, ties to even.
    #[inline(always)]
    pub fn round_ties_even(self) -> F32 {
        prim::round_ties_even(self)
    }

    /// Largest integer value not above `self`.
    #[inline(always)]
    pub fn floor(self) -> F32 {
        prim::floor(self)
    }
}

/// The boundary primitives: opaque to the extraction, one `proof/boundary-axioms.txt` entry each.
///
/// Each body is the hardware instruction through the native `f32` operator. All are O(1) in time and
/// space (one instruction).
pub mod prim {
    use super::{F32, F64};

    #[inline(always)]
    fn f(a: F32) -> f32 {
        f32::from_bits(a.0)
    }

    #[inline(always)]
    fn g(x: f32) -> F32 {
        F32(x.to_bits())
    }

    /// `rne(v a + v b)`.
    #[inline(always)]
    pub fn add(a: F32, b: F32) -> F32 {
        g(f(a) + f(b))
    }

    /// `rne(v a - v b)`.
    #[inline(always)]
    pub fn sub(a: F32, b: F32) -> F32 {
        g(f(a) - f(b))
    }

    /// `rne(v a * v b)`.
    #[inline(always)]
    pub fn mul(a: F32, b: F32) -> F32 {
        g(f(a) * f(b))
    }

    /// `rne(v a / v b)`.
    #[inline(always)]
    pub fn div(a: F32, b: F32) -> F32 {
        g(f(a) / f(b))
    }

    /// `rne(v a * v b + v c)`, one rounding.
    #[inline(always)]
    pub fn fma(a: F32, b: F32, c: F32) -> F32 {
        g(f(a).mul_add(f(b), f(c)))
    }

    /// `rne(sqrt(v a))`; NaN for `v a < 0`.
    #[inline(always)]
    pub fn sqrt(a: F32) -> F32 {
        g(f(a).sqrt())
    }

    /// IEEE ordered `<`.
    #[inline(always)]
    pub fn lt(a: F32, b: F32) -> bool {
        f(a) < f(b)
    }

    /// IEEE ordered `<=`.
    #[inline(always)]
    pub fn le(a: F32, b: F32) -> bool {
        f(a) <= f(b)
    }

    /// IEEE `==`.
    #[inline(always)]
    pub fn eq(a: F32, b: F32) -> bool {
        f(a) == f(b)
    }

    /// IEEE maxNum.
    #[inline(always)]
    pub fn max(a: F32, b: F32) -> F32 {
        g(f(a).max(f(b)))
    }

    /// IEEE minNum.
    #[inline(always)]
    pub fn min(a: F32, b: F32) -> F32 {
        g(f(a).min(f(b)))
    }

    /// `rne(i)`.
    #[inline(always)]
    pub fn from_i64(i: i64) -> F32 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "the rounding is the specification"
        )]
        g(i as f32)
    }

    /// Rust `as i64`.
    #[inline(always)]
    pub fn to_i64_sat(a: F32) -> i64 {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "saturating cast is the specification"
        )]
        {
            f(a) as i64
        }
    }

    /// Rust `as i32`.
    #[inline(always)]
    pub fn to_i32_sat(a: F32) -> i32 {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "saturating cast is the specification"
        )]
        {
            f(a) as i32
        }
    }

    /// Round to nearest integer, ties to even.
    #[inline(always)]
    pub fn round_ties_even(a: F32) -> F32 {
        g(f(a).round_ties_even())
    }

    /// Floor.
    #[inline(always)]
    pub fn floor(a: F32) -> F32 {
        g(f(a).floor())
    }

    /// Load-time `f64` config check: `v` is finite and strictly positive.
    #[inline]
    pub fn f64_positive_finite(v: F64) -> bool {
        let v = v.to_f64();
        v.is_finite() && v > 0.0
    }

    /// Load-time `RoPE` phase `round(theta^(-2j/head_dim) / (2 pi) * 2^64)` as a `u64` turn count
    /// (host `f64` `powf` and rounding); `None` when the value is non-finite or reaches `2^64`.
    /// O(1) per call; the table build calls it once per rotary dimension.
    #[inline]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the host f64 powf/round boundary is declared; the range check keeps the cast in the \
                  u64 phase domain, and `j`/`head_dim` are tiny"
    )]
    pub fn rope_turn(theta: F64, j: usize, head_dim: usize) -> Option<u64> {
        let theta = theta.to_f64();
        let unit = 18_446_744_073_709_551_616.0_f64; // 2^64: one full turn in phases
        let two_pi = 2.0 * core::f64::consts::PI;
        let exponent = -2.0 * (j as f64) / (head_dim as f64);
        let q = (theta.powf(exponent) * unit / two_pi).round();
        if q.is_finite() && (0.0..unit).contains(&q) {
            Some(q as u64)
        } else {
            None
        }
    }

    /// `round(v * 2^32)` for a finite non-negative `v` (the norm epsilon as Q32.32); `None` when `v`
    /// is negative, non-finite or the product leaves `[0, 2^63)`. O(1).
    #[inline]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the finite range guard above the cast"
    )]
    pub fn eps_q32(v: F64) -> Option<i64> {
        let v = v.to_f64();
        if !v.is_finite() || v < 0.0 {
            return None;
        }
        let q = (v * 4_294_967_296.0).round();
        if q.is_finite() && (0.0..9_223_372_036_854_775_808.0).contains(&q) {
            Some(q as i64)
        } else {
            None
        }
    }

    /// CPU feature oracle: AVX2 is available (same contract as [`cpu_avx2_fma`]).
    #[allow(
        dead_code,
        reason = "the bare avx2 feature-detect sites route through it as their lanes land"
    )]
    #[inline]
    pub fn cpu_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// CPU feature oracle: AVX2 and FMA are both available. Process-constant after the first call;
    /// the Lean model treats the value as arbitrary, since every SIMD twin equals its scalar twin.
    #[inline]
    pub fn cpu_avx2_fma() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

impl Add for F32 {
    type Output = F32;
    #[inline(always)]
    fn add(self, o: F32) -> F32 {
        prim::add(self, o)
    }
}

impl Sub for F32 {
    type Output = F32;
    #[inline(always)]
    fn sub(self, o: F32) -> F32 {
        prim::sub(self, o)
    }
}

impl Mul for F32 {
    type Output = F32;
    #[inline(always)]
    fn mul(self, o: F32) -> F32 {
        prim::mul(self, o)
    }
}

impl Div for F32 {
    type Output = F32;
    #[inline(always)]
    fn div(self, o: F32) -> F32 {
        prim::div(self, o)
    }
}

impl Neg for F32 {
    type Output = F32;
    #[inline(always)]
    fn neg(self) -> F32 {
        F32(self.0 ^ SIGN)
    }
}

impl AddAssign for F32 {
    #[inline(always)]
    fn add_assign(&mut self, o: F32) {
        *self = prim::add(*self, o);
    }
}

impl SubAssign for F32 {
    #[inline(always)]
    fn sub_assign(&mut self, o: F32) {
        *self = prim::sub(*self, o);
    }
}

impl MulAssign for F32 {
    #[inline(always)]
    fn mul_assign(&mut self, o: F32) {
        *self = prim::mul(*self, o);
    }
}

impl DivAssign for F32 {
    #[inline(always)]
    fn div_assign(&mut self, o: F32) {
        *self = prim::div(*self, o);
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::nonminimal_bool,
    reason = "tests compare each primitive with the hardware cast, comparison and constant it models"
)]
mod tests {
    use super::{prim, F32, F64};

    /// Normal, subnormal, zero, infinite and NaN patterns, both signs.
    const PATTERNS: [u32; 24] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x0080_0001,
        0x3f80_0000,
        0xbf80_0000,
        0x3f00_0000,
        0x4000_0000,
        0x4049_0fdb,
        0xc2f6_e979,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_0001,
        0x3eaa_aaab,
        0x4b00_0000,
        0x4b7f_ffff,
        0x3fc0_0000,
    ];

    fn same(a: F32, b: f32) -> bool {
        let n = f32::from_bits(a.0);
        if n.is_nan() || b.is_nan() {
            n.is_nan() && b.is_nan()
        } else {
            a.0 == b.to_bits()
        }
    }

    fn nat(a: u32) -> f32 {
        f32::from_bits(a)
    }

    #[test]
    fn prim_binary_matches_hardware() {
        for &x in &PATTERNS {
            for &y in &PATTERNS {
                let (a, b) = (F32(x), F32(y));
                let (p, q) = (nat(x), nat(y));
                assert!(same(prim::add(a, b), p + q), "add {x:#x} {y:#x}");
                assert!(same(prim::sub(a, b), p - q), "sub {x:#x} {y:#x}");
                assert!(same(prim::mul(a, b), p * q), "mul {x:#x} {y:#x}");
                assert!(same(prim::div(a, b), p / q), "div {x:#x} {y:#x}");
                assert!(same(prim::max(a, b), p.max(q)), "max {x:#x} {y:#x}");
                assert!(same(prim::min(a, b), p.min(q)), "min {x:#x} {y:#x}");
                assert_eq!(prim::lt(a, b), p < q, "lt {x:#x} {y:#x}");
                assert_eq!(prim::le(a, b), p <= q, "le {x:#x} {y:#x}");
                assert_eq!(prim::eq(a, b), p == q, "eq {x:#x} {y:#x}");
                for &z in &PATTERNS {
                    let (c, r) = (F32(z), nat(z));
                    assert!(
                        same(prim::fma(a, b, c), p.mul_add(q, r)),
                        "fma {x:#x} {y:#x} {z:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn prim_unary_matches_hardware() {
        for &x in &PATTERNS {
            let (a, p) = (F32(x), nat(x));
            assert!(same(prim::sqrt(a), p.sqrt()), "sqrt {x:#x}");
            assert!(
                same(prim::round_ties_even(a), p.round_ties_even()),
                "rte {x:#x}"
            );
            assert!(same(prim::floor(a), p.floor()), "floor {x:#x}");
            assert_eq!(prim::to_i64_sat(a), p as i64, "to_i64 {x:#x}");
            assert_eq!(prim::to_i32_sat(a), p as i32, "to_i32 {x:#x}");
            assert!(same(-a, -p), "neg {x:#x}");
            assert!(same(a.abs(), p.abs()), "abs {x:#x}");
            assert_eq!(a.is_nan(), p.is_nan(), "is_nan {x:#x}");
            assert_eq!(a.is_finite(), p.is_finite(), "is_finite {x:#x}");
            for &y in &PATTERNS {
                assert!(
                    same(a.copysign(F32(y)), p.copysign(nat(y))),
                    "copysign {x:#x} {y:#x}"
                );
            }
        }
    }

    #[test]
    fn prim_from_i64_rounds_to_nearest_even() {
        for i in [
            0i64,
            1,
            -1,
            16_777_216,
            16_777_217,
            16_777_219,
            -16_777_217,
            i64::MAX,
            i64::MIN,
            (1 << 40) + (1 << 16),
            (1 << 40) + (1 << 16) + 1,
            123_456_789_012_345,
        ] {
            assert!(same(prim::from_i64(i), i as f32), "from_i64 {i}");
        }
    }

    #[test]
    fn to_q32_matches_f64_round() {
        let old = |v: f32| -> Option<i64> {
            if !v.is_finite() {
                return None;
            }
            let q = (f64::from(v) * 4_294_967_296.0).round();
            if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&q) {
                return None;
            }
            Some(q as i64)
        };
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let check = |b: u32| assert_eq!(F32(b).to_q32(), old(f32::from_bits(b)), "to_q32 {b:#x}");
        for &b in &PATTERNS {
            check(b);
        }
        for e in 0u32..=255 {
            for sign in [0u32, 0x8000_0000] {
                for frac in [0u32, 1, 0x40_0000, 0x7f_ffff, 0x12_3457] {
                    check(sign | (e << 23) | frac);
                }
            }
        }
        for _ in 0..2_000_000 {
            check(next() as u32);
        }
    }

    #[test]
    fn from_q32_is_x_over_2_pow_32() {
        for x in [
            0i64,
            1,
            -1,
            4_294_967_296,
            i64::MAX,
            i64::MIN,
            123_456_789_012,
            -987_654_321,
        ] {
            assert!(
                same(F32::from_q32(x), (x as f32) / 4_294_967_296.0_f32),
                "from_q32 {x}"
            );
        }
    }

    /// MXCSR flush-to-zero, denormals-are-zero and rounding mode are process state outside the
    /// model; the operations above are specified for the default control word.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn mxcsr_is_default() {
        let tiny = F32(0x0000_0001);
        assert!(
            same(prim::add(tiny, F32::ZERO), f32::from_bits(1)),
            "DAZ/FTZ must be off"
        );
        assert!(same(
            prim::mul(F32(0x0080_0000), F32::HALF),
            f32::from_bits(0x0040_0000)
        ));
        assert!(
            same(prim::add(F32::ONE, F32(0x3380_0000)), 1.0),
            "round-to-nearest-even"
        );
    }

    /// Test inputs: edges plus a deterministic xorshift sweep of bit patterns.
    fn f64_inputs() -> Vec<f64> {
        let mut v = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            1e-300,
            5e-324,
            1e300,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1e-9,
            1e-6,
            1e-5,
            1e4,
            1e6,
            1e7,
            500_000.0,
            10_000.0,
            2147483648.0,
            2147483647.5,
            4294967296.0,
            9.223372036854776e9,
            2.0_f64.powi(31),
        ];
        let mut s = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v.push(f64::from_bits(s));
            v.push((s >> 11) as f64 / (1u64 << 53) as f64 * 1e7);
        }
        v
    }

    #[test]
    fn f64_positive_finite_matches_old_guard() {
        for v in f64_inputs() {
            let old = !(!v.is_finite() || v <= 0.0);
            assert_eq!(prim::f64_positive_finite(F64::from_f64(v)), old, "{v:e}");
        }
    }

    #[test]
    fn eps_q32_matches_old_step_body() {
        for v in f64_inputs() {
            let old = if !v.is_finite() || v < 0.0 {
                None
            } else {
                let q = (v * 4_294_967_296.0).round();
                if q.is_finite() && (0.0..9_223_372_036_854_775_808.0).contains(&q) {
                    Some(q as i64)
                } else {
                    None
                }
            };
            assert_eq!(prim::eps_q32(F64::from_f64(v)), old, "{v:e}");
        }
    }

    #[test]
    fn rope_turn_matches_old_table_body() {
        for theta in [
            10_000.0_f64,
            1e6,
            500_000.0,
            1e7,
            1.5,
            1e-3,
            1e300,
            1e-300,
            2.0,
        ] {
            for hd in [2usize, 64, 128, 256] {
                for j in 0..hd / 2 {
                    let unit = 18_446_744_073_709_551_616.0_f64;
                    let two_pi = 2.0 * core::f64::consts::PI;
                    let q = (theta.powf(-2.0 * (j as f64) / (hd as f64)) * unit / two_pi).round();
                    let old = if q.is_finite() && (0.0..unit).contains(&q) {
                        Some(q as u64)
                    } else {
                        None
                    };
                    assert_eq!(
                        prim::rope_turn(F64::from_f64(theta), j, hd),
                        old,
                        "{theta:e} {j} {hd}"
                    );
                }
            }
        }
    }

    #[test]
    fn cpu_oracles_agree_with_std() {
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(prim::cpu_avx2(), std::is_x86_feature_detected!("avx2"));
            assert_eq!(
                prim::cpu_avx2_fma(),
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
            );
        }
    }
}
