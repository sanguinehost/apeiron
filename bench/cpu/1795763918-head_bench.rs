//! Standalone AVX2 Q8 head-kernel bench: current per-block-scale row kernel versus a
//! per-row-scale carrier whose dots accumulate in i32 lanes across all 32 blocks.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
use std::arch::x86_64::*;
use std::time::Instant;

const N_OUT: usize = 151936;
const N_IN: usize = 1024;
const BLOCKS: usize = N_IN / 32;
const K: usize = 7;

#[target_feature(enable = "avx2")]
unsafe fn dot32(w: __m256i, x: __m256i) -> __m256i {
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w)),
        _mm256_set1_epi16(1),
    )
}

#[target_feature(enable = "avx2")]
unsafe fn hsum8(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x4e>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0xb1>(s));
    _mm_cvtsi128_si32(s)
}

fn f16_from_f32(v: f32) -> u16 {
    // Minimal f32 -> f16 (round-to-nearest) for plausible scales.
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mut mant = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        exp = 30;
    }
    mant |= 0x400;
    let _ = mant;
    (sign | ((exp as u16) << 10) | (((bits >> 13) & 0x3ff) as u16))
}

fn fill_weights() -> Vec<u8> {
    let row_bytes = BLOCKS * 34;
    let mut w = vec![0u8; N_OUT * row_bytes];
    let d = f16_from_f32(0.0217);
    let db = d.to_le_bytes();
    for r in 0..N_OUT {
        let row = &mut w[r * row_bytes..(r + 1) * row_bytes];
        for b in 0..BLOCKS {
            row[b * 34] = db[0];
            row[b * 34 + 1] = db[1];
            for l in 0..32 {
                // Deterministic signed codes from a cheap LCG.
                let v = ((r as u64)
                    .wrapping_mul(1103515245)
                    .wrapping_add((b as u64).wrapping_mul(12345))
                    .wrapping_add((l as u64).wrapping_mul(2654435761)))
                    % 255;
                row[b * 34 + 2 + l] = (v as u8).wrapping_sub(127);
            }
        }
    }
    w
}

fn activations() -> (Vec<i8>, Vec<i64>) {
    let mut xs = vec![0i8; K * N_IN];
    let mut ss = vec![0i64; K * BLOCKS];
    for c in 0..K {
        for i in 0..N_IN {
            let v = (c * 977 + i * 131) % 255;
            xs[c * N_IN + i] = (v as i32 - 127) as i8;
        }
        for b in 0..BLOCKS {
            ss[c * BLOCKS + b] = 1i64 << 30;
        }
    }
    (xs, ss)
}

/// The tree's current structure: 4-block groups, per-block scale epilogue in i128.
#[target_feature(enable = "avx2")]
unsafe fn kernel_block_scale(w: &[u8], xs: &[i8], ss: &[i64], out: &mut [i64; K]) -> u64 {
    let row_bytes = BLOCKS * 34;
    let ones = _mm256_set1_epi16(1);
    let mut sink = 0u64;
    for r in 0..N_OUT {
        let row = w.as_ptr().add(r * row_bytes);
        let mut acc = [0i128; K];
        let mut blk = 0usize;
        while blk + 4 <= BLOCKS {
            let off = blk * 34;
            let ds = [
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off) as *const __m128i), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 34) as *const __m128i), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 68) as *const __m128i), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 102) as *const __m128i), 0) as u16),
            ];
            let codes = [
                _mm256_loadu_si256(row.add(off + 2) as *const __m256i),
                _mm256_loadu_si256(row.add(off + 36) as *const __m256i),
                _mm256_loadu_si256(row.add(off + 70) as *const __m256i),
                _mm256_loadu_si256(row.add(off + 104) as *const __m256i),
            ];
            let mut c = 0usize;
            while c < K {
                let x = xs.as_ptr().add(c * N_IN + blk * 32);
                let p0 = dot32(codes[0], _mm256_loadu_si256(x as *const __m256i));
                let p1 = dot32(codes[1], _mm256_loadu_si256(x.add(32) as *const __m256i));
                let p2 = dot32(codes[2], _mm256_loadu_si256(x.add(64) as *const __m256i));
                let p3 = dot32(codes[3], _mm256_loadu_si256(x.add(96) as *const __m256i));
                // 4 block dots from the lane sums.
                let mut d = [0i32; 4];
                for (i, p) in [p0, p1, p2, p3].iter().enumerate() {
                    d[i] = hsum8(*p);
                }
                let s = ss.as_ptr().add(c * BLOCKS + blk);
                let mut i = 0usize;
                while i < 4 {
                    let scaled = *s.add(i) * i64::from(d[i]);
                    acc[c] += i128::from(ds[i]) * i128::from(scaled);
                    i += 1;
                }
                c += 1;
            }
            blk += 4;
        }
        while blk < BLOCKS {
            let off = blk * 34;
            let d0 = f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off) as *const __m128i), 0) as u16);
            let mut c = 0usize;
            while c < K {
                let x = xs.as_ptr().add(c * N_IN + blk * 32);
                let p = dot32(
                    _mm256_loadu_si256(row.add(off + 2) as *const __m256i),
                    _mm256_loadu_si256(x as *const __m256i),
                );
                let s = ss[c * BLOCKS + blk] * i64::from(hsum8(p));
                acc[c] += i128::from(d0) * i128::from(s);
                c += 1;
            }
            blk += 1;
        }
        let _ = ones;
        let mut c = 0usize;
        while c < K {
            sink = sink.wrapping_add(acc[c] as u64);
            c += 1;
        }
    }
    sink
}

/// Per-row scale: the 32 block dots accumulate in i32 lanes, one scale at the end.
#[target_feature(enable = "avx2")]
unsafe fn kernel_row_scale(w: &[u8], xs: &[i8], ss: &[i64], ds: &[i64], out: &mut [i64; K]) -> u64 {
    let row_bytes = BLOCKS * 34;
    let mut sink = 0u64;
    for r in 0..N_OUT {
        let row = w.as_ptr().add(r * row_bytes);
        let d = ds[r];
        let mut acc = [_mm256_setzero_si256(); K];
        let mut blk = 0usize;
        while blk < BLOCKS {
            let off = blk * 34;
            let wv = _mm256_loadu_si256(row.add(off + 2) as *const __m256i);
            let wa = _mm256_abs_epi8(wv);
            let mut c = 0usize;
            while c < K {
                let x = xs.as_ptr().add(c * N_IN + blk * 32);
                let p = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(wa, _mm256_sign_epi8(_mm256_loadu_si256(x as *const __m256i), wv)),
                    _mm256_set1_epi16(1),
                );
                acc[c] = _mm256_add_epi32(acc[c], p);
                c += 1;
            }
            blk += 1;
        }
        let mut c = 0usize;
        while c < K {
            let dot = i64::from(hsum8(acc[c]));
            out[c] = (i128::from(d) * i128::from(ss[c]) * i128::from(dot) >> 32) as i64;
            sink = sink.wrapping_add(out[c] as u64);
            c += 1;
        }
    }
    sink
}

fn f16_to_fixed(bits: u16) -> i64 {
    let sign = if bits & 0x8000 != 0 { -1i64 } else { 1 };
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x3ff) as i64;
    if exp == 0 {
        sign * (mant << (32 - 24))
    } else {
        sign * ((mant | 0x400) << (32 + exp as i64 - 25))
    }
}

fn run(name: &str, mut f: impl FnMut() -> u64) {
    // Warm-up, then three timed runs.
    let _ = f();
    let mut best = f64::INFINITY;
    let mut sink = 0u64;
    for _ in 0..3 {
        let t = Instant::now();
        sink = f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
    }
    let macs = N_OUT as f64 * N_IN as f64 * K as f64;
    let gmac = macs / 1e9;
    eprintln!(
        "{name:>10}: {best:8.3} ms  {:.1} GMAC/s  sink {sink}",
        gmac / (best / 1e3)
    );
}

fn main() {
    let w = fill_weights();
    let (xs, ss) = activations();
    let mut ds = vec![0i64; N_OUT];
    for (r, d) in ds.iter_mut().enumerate() {
        *d = f16_to_fixed(f16_from_f32(0.0217 + (r % 7) as f32 * 1e-5));
    }
    let mut out = [0i64; K];
    run("block", || unsafe {
        kernel_block_scale(&w, &xs, &ss, &mut out)
    });
    run("row", || unsafe {
        kernel_row_scale(&w, &xs, &ss, &ds, &mut out)
    });
}
