//! Tests for the binary16 carrier: the exact conversions, and the F16 tile/group cores against
//! the committed binary32 cores over the same plane materialised in binary32.
//!
//! The carrier's contract is that a stored half converts to the same binary32 word at every read,
//! so every F16 read equals the F32 read of that materialised plane bit for bit. These tests
//! check that equality over the geometries the engine uses, and compare the scalar conversion
//! against the F16C instruction over the whole finite half domain and random binary32 patterns.

#![allow(
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    reason = "test harnesses, random generators and numerical verification"
)]

use super::*;
use crate::engine::attn::{KvCarrier, KvState};

#[allow(unused_imports, reason = "the RNG comes from the sibling test module")]
use super::tests::SimpleRng;

#[test]
fn test_f16_to_f32_is_exact_on_every_finite_half() {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("f16c") {
        use std::arch::x86_64::{_mm_cvtph_ps, _mm_set1_epi16, _mm_storeu_ps};
        let mut bits = 0u32;
        while bits <= 0xffff {
            let h = bits as u16;
            bits += 1;
            if h & 0x7c00 == 0x7c00 {
                // Infinity and NaN are refused at the write, and their payload policy is the
                // conversion's own (below); the finite domain is the carrier's.
                continue;
            }
            // SAFETY: F16C was detected on this processor; the intrinsic reads one register.
            let want = unsafe {
                let v = _mm_cvtph_ps(_mm_set1_epi16(h as i16));
                let mut lanes = [0.0f32; 4];
                _mm_storeu_ps(lanes.as_mut_ptr(), v);
                lanes[0]
            };
            let got = f16_to_f32(h);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "half {h:#06x}: got {:#010x}, want {:#010x}",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

#[test]
fn test_f16_to_f32_handles_infinity_and_nan_by_payload() {
    assert_eq!(f16_to_f32(0x7c00).to_bits(), 0x7f80_0000);
    assert_eq!(f16_to_f32(0xfc00).to_bits(), 0xff80_0000);
    // A NaN keeps its significand, shifted into the binary32 field.
    assert_eq!(f16_to_f32(0x7e00).to_bits(), 0x7fc0_0000);
    assert_eq!(f16_to_f32(0xfe01).to_bits(), 0xffc0_2000);
}

/// One value checked against the hardware rounding: the conversion and the comparison both live
/// in this `#[target_feature]` body, so release codegen cannot emit the intrinsic from a caller
/// compiled for a narrower feature set (which loses the sign of zero and the subnormal bit).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline(never)]
unsafe fn check_hardware_half(x: f32) {
    use std::arch::x86_64::{_mm256_cvtps_ph, _mm256_set1_ps, _mm_storeu_si128};
    let v = _mm256_cvtps_ph(_mm256_set1_ps(x), 0);
    let mut lanes = [0u16; 8];
    _mm_storeu_si128(lanes.as_mut_ptr().cast(), v);
    let want = lanes[0];
    let got = f32_to_f16_bits(F32::from_bits(x.to_bits()));
    let hw_special = want & 0x7c00 == 0x7c00;
    let hw_nan = hw_special && want.trailing_zeros() < 10;
    let hw_infinite = hw_special && !hw_nan;
    if hw_infinite || hw_nan {
        assert_eq!(got, None, "{x} ({:#010x}) must refuse", x.to_bits());
    } else {
        assert_eq!(
            got,
            Some(want),
            "{x} ({:#010x}): got {got:x?}, want {want:#06x}",
            x.to_bits()
        );
    }
}

#[test]
fn test_f32_to_f16_bits_equals_the_hardware_rounding() {
    #[cfg(target_arch = "x86_64")]
    if !(std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx2"))
    {
        return;
    }
    // Boundaries: signed zero, both normal/subnormal edges, the largest finite half, the first
    // value that rounds to infinity, and the tie at the overflow.
    let mut rng = SimpleRng::new(0x1337_c0de_5eed_00f6);
    let fixed: Vec<u32> = vec![
        0x0000_0000,
        0x8000_0000,
        0x3380_0000, // 2^-24, smallest half subnormal
        0x3300_0000, // half of it: rounds to even zero
        0x387f_ffff, // just below the smallest normal half
        0x3880_0000, // smallest normal half
        0x477f_e000, // 65504, largest finite half
        0x477f_efff, // just below the 65520 tie
        0x477f_f000, // 65520: rounds to infinity, refused
        0x4780_0000, // 65536, refused
    ];
    #[cfg(target_arch = "x86_64")]
    for &bits in &fixed {
        // SAFETY: F16C and AVX2 were detected on this processor.
        unsafe { check_hardware_half(f32::from_bits(bits)) };
    }
    let mut i = 0usize;
    while i < 200_000 {
        // Uniform random binary32 bit patterns cover normals, subnormals, NaN and infinity.
        let bits = (rng.next_u64() >> 32) as u32;
        #[cfg(target_arch = "x86_64")]
        // SAFETY: F16C and AVX2 were detected on this processor.
        unsafe {
            check_hardware_half(f32::from_bits(bits));
        };
        i += 1;
    }
}

#[test]
fn test_attn_core_f16_group_equals_the_f32_group() {
    let mut rng = SimpleRng::new(0x1337_c0de_5eed_00f7);
    let configs = [
        (16usize, 5 * 16, 21usize, 3usize),
        (64, 64, 0, 4),
        (128, 1024, 31, 4),
        (128, 1024, 71, 1),
    ];
    for &(hd, stride, pos, group) in &configs {
        let count = pos + 1;
        let total_len = pos * stride + hd;
        let mut k = vec![0u16; total_len];
        let mut v = vec![0u16; total_len];
        let mut kf = vec![F32::ZERO; total_len];
        let mut vf = vec![F32::ZERO; total_len];
        for s in 0..count {
            let base = s * stride;
            for i in 0..hd {
                let kr = rng.next_f32_range(-1.0, 1.0);
                let vr = rng.next_f32_range(-4.0, 4.0);
                let kh = f32_to_f16_bits(F32::from_bits(kr.to_bits())).expect("finite");
                let vh = f32_to_f16_bits(F32::from_bits(vr.to_bits())).expect("finite");
                k[base + i] = kh;
                v[base + i] = vh;
                kf[base + i] = f16_to_f32(kh);
                vf[base + i] = f16_to_f32(vh);
            }
        }
        let mut q = vec![F32::ZERO; group * hd];
        for item in &mut q {
            *item = F32::from_bits(rng.next_f32_range(-1.0, 1.0).to_bits());
        }
        let scale = F32::from_bits((1.0f32 / (hd as f32).sqrt()).to_bits());
        let mut want = vec![F32::ZERO; group * hd];
        attn_core_f32_group(&q, &kf, &vf, group, hd, stride, pos, scale, &mut want)
            .expect("f32 group");
        let mut got = vec![F32::ZERO; group * hd];
        attn_core_f16_group(&q, &k, &v, group, hd, stride, pos, scale, &mut got)
            .expect("f16 group");
        for i in 0..got.len() {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "hd {hd} stride {stride} pos {pos} group {group} word {i}"
            );
        }
    }
}

#[test]
fn test_attn_core_f16_tile_equals_the_f32_tile() {
    let mut rng = SimpleRng::new(0x1337_c0de_5eed_00f8);
    let configs = [
        (8usize, 8usize, 0usize),
        (16, 16, 1),
        (16, 24, 7),
        (64, 64, 65),
        (128, 128, 129),
    ];
    for &(hd, stride, first) in &configs {
        for query_heads in [1usize, 4] {
            for nq in [1usize, 3, 8] {
                let last = first + nq - 1;
                let total = last * stride + hd;
                let width = query_heads * hd;
                let queries: Vec<F32> = (0..nq * width)
                    .map(|_| F32::from_bits(rng.next_f32_range(-1.0, 1.0).to_bits()))
                    .collect();
                let mut k = vec![0u16; total];
                let mut v = vec![0u16; total];
                let mut kf = vec![F32::ZERO; total];
                let mut vf = vec![F32::ZERO; total];
                for s in 0..=last {
                    for i in 0..hd {
                        let kr = rng.next_f32_range(-1.0, 1.0);
                        let vr = rng.next_f32_range(-2.0, 2.0);
                        let kh = f32_to_f16_bits(F32::from_bits(kr.to_bits())).expect("finite");
                        let vh = f32_to_f16_bits(F32::from_bits(vr.to_bits())).expect("finite");
                        k[s * stride + i] = kh;
                        v[s * stride + i] = vh;
                        kf[s * stride + i] = f16_to_f32(kh);
                        vf[s * stride + i] = f16_to_f32(vh);
                    }
                }
                let positions: Vec<usize> = (first..=last).collect();
                let scale = F32::from_bits((1.0f32 / (hd as f32).sqrt()).to_bits());
                let mut want = vec![F32::ZERO; nq * width];
                attn_core_f32_tile(
                    &queries,
                    &kf,
                    &vf,
                    hd,
                    query_heads,
                    stride,
                    &positions,
                    scale,
                    &mut want,
                )
                .expect("f32 tile");
                let mut got = vec![F32::ZERO; nq * width];
                attn_core_f16_tile(
                    &queries,
                    &k,
                    &v,
                    hd,
                    query_heads,
                    stride,
                    &positions,
                    scale,
                    &mut got,
                )
                .expect("f16 tile");
                for i in 0..got.len() {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "hd {hd} stride {stride} first {first} nq {nq} heads {query_heads} word {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_attn_core_f16_refuses_bad_geometry_and_non_finite_values() {
    let hd = 8usize;
    let stride = 16usize;
    let pos = 1usize;
    let len = pos * stride + hd;
    let k = vec![0u16; len];
    let mut v = vec![0u16; len];
    let q = vec![F32::ZERO; 2 * hd];
    let scale = F32::from_bits(1.0f32.to_bits());
    let mut out = vec![F32::ZERO; 2 * hd];
    // A half-plane with a NaN value row refuses like the binary32 core's non-finite row, for
    // every head, even at a position a query in the tile does not reach.
    v[stride] = 0x7e00;
    assert_eq!(
        attn_core_f16_group(&q, &k, &v, 2, hd, stride, pos, scale, &mut out),
        Err(AttnError::Overflow)
    );
    v[stride] = 0;
    let short_q = vec![F32::ZERO; 2 * hd - 1];
    assert_eq!(
        attn_core_f16_group(&short_q, &k, &v, 2, hd, stride, pos, scale, &mut out),
        Err(AttnError::BadGeometry)
    );
    let mut short_out = vec![F32::ZERO; 2 * hd - 1];
    assert_eq!(
        attn_core_f16_group(&q, &k, &v, 2, hd, stride, pos, scale, &mut short_out),
        Err(AttnError::BadGeometry)
    );
    assert_eq!(
        attn_core_f16_group(&q, &k, &v, 0, hd, stride, pos, scale, &mut []),
        Err(AttnError::BadGeometry)
    );
    assert_eq!(
        attn_core_f16_group(&q, &k[..len - 1], &v, 2, hd, stride, pos, scale, &mut out),
        Err(AttnError::BadGeometry)
    );
    // The tile refuses a lane whose half is not finite at any row up to the last column.
    let mut vt = v.clone();
    vt[0] = 0x7c00;
    let positions = vec![0usize, 1];
    let qt = vec![F32::ZERO; positions.len() * 2 * hd];
    let mut outt = vec![F32::ZERO; qt.len()];
    assert_eq!(
        attn_core_f16_tile(&qt, &k, &vt, hd, 2, stride, &positions, scale, &mut outt),
        Err(AttnError::Overflow)
    );
}

#[test]
fn test_f16_append_rounds_and_refuses_above_the_largest_half() {
    let one = 1i64 << 32;
    let mut kv = KvState::with_carrier(1, 1, 1, 2, 4, KvCarrier::F16).expect("geometry");
    // 0.5 is exact in binary16; the value row reads back bit for bit at one position.
    assert_eq!(kv.append(0, &[one, 0], &[0, one / 2]), Ok(0));
    let mut out = [i64::MIN; 2];
    kv.attn_head(0, 0, &[one, 0], 0, &mut out)
        .expect("attend pos 0");
    assert_eq!(out, [0, one / 2]);
    // A Q32.32 word above the largest finite half refuses on the write. 70,000 · 2^32 fits i64.
    let big = 70_000i64 << 32;
    assert_eq!(kv.append(0, &[big, 0], &[0, 0]), Err(AttnError::Overflow));
    assert_eq!(kv.occupied(0), Ok(1));
    // A subnormal half round-trips through the read as its exact binary32 value: the smallest
    // positive half (2^-24) is `1 << 8` in Q32.32.
    let mut kv = KvState::with_carrier(1, 1, 1, 2, 4, KvCarrier::F16).expect("geometry");
    assert_eq!(kv.append(0, &[0, 0], &[0, 1 << 8]), Ok(0));
    let mut out = [i64::MIN; 2];
    kv.attn_head(0, 0, &[one, 0], 0, &mut out)
        .expect("attend pos 0");
    // Rounding to Q32.32 of 2^-24 is exact at zero: the readout is `round(f · 2^32)` = 2^8.
    assert_eq!(out[1], 1 << 8);
}
