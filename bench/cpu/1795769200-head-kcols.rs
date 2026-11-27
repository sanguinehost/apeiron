//! Head `Q8_0` row kernel: fold the activation scale out of the block loop.
//!
//! The head's 1024-element row is one quantizer group, so one activation scale covers all 32
//! blocks. Forming `Σ_b A·D_b·dot_b` as `A·Σ_b D_b·dot_b` removes one of the two multiplies per
//! block and keeps the `i128` accumulator bit-identical (integer multiplication is associative).
//! `abs` of the four weight code vectors is hoisted out of the column loop as well.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
use std::arch::x86_64::*;
use std::time::Instant;

const N_OUT: usize = 151936;
const N_IN: usize = 1024;
const BLOCKS: usize = N_IN / 32;
const K: usize = 7;

struct Q {
    codes: Vec<i8>,
    scales: Vec<i64>,
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

fn f16_from_f32(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        exp = 30;
    }
    sign | ((exp as u16) << 10) | (((bits >> 13) & 0x3ff) as u16)
}

fn fill_weights() -> Vec<u8> {
    let row_bytes = BLOCKS * 34;
    let mut w = vec![0u8; N_OUT * row_bytes];
    for r in 0..N_OUT {
        let row = &mut w[r * row_bytes..(r + 1) * row_bytes];
        for b in 0..BLOCKS {
            let d = f16_from_f32(0.018 + ((r + b) % 13) as f32 * 1e-4);
            row[b * 34..b * 34 + 2].copy_from_slice(&d.to_le_bytes());
            for l in 0..32 {
                let v = ((r as u64)
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add((b as u64).wrapping_mul(12_345))
                    .wrapping_add((l as u64).wrapping_mul(2_654_435_761)))
                    % 255;
                row[b * 34 + 2 + l] = (v as u8).wrapping_sub(127);
            }
        }
    }
    w
}

fn activations() -> Vec<Q> {
    let mut out = Vec::new();
    for c in 0..K {
        let mut codes = vec![0i8; N_IN];
        for (i, x) in codes.iter_mut().enumerate() {
            let v = (c * 977 + i * 131) % 255;
            *x = (v as i32 - 127) as i8;
        }
        let scales: Vec<i64> = (0..BLOCKS).map(|b| 1i64 << (28 + (b % 3))).collect();
        out.push(Q { codes, scales });
    }
    out
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot32(c: __m256i, x: __m256i) -> __m256i {
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_abs_epi8(c), _mm256_sign_epi8(x, c)),
        _mm256_set1_epi16(1),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn reduce4(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> [i32; 4] {
    let h01 = _mm256_hadd_epi32(p0, p1);
    let h23 = _mm256_hadd_epi32(p2, p3);
    let h = _mm256_hadd_epi32(h01, h23);
    let r = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256::<1>(h));
    let mut d = [0i32; 4];
    _mm_storeu_si128(d.as_mut_ptr().cast(), r);
    d
}

/// The shipped shape, inlined per column.
#[target_feature(enable = "avx2")]
unsafe fn dyn_k(w: &[u8], qs: &[Q], k: usize, acc: &mut [i128]) {
    let row_bytes = BLOCKS * 34;
    let ones = _mm256_set1_epi16(1);
    for r in 0..N_OUT {
        let row = w.as_ptr().add(r * row_bytes);
        acc[..k].fill(0);
        let mut blk = 0usize;
        while blk + 4 <= BLOCKS {
            let off = blk * 34;
            let ds = [
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 34).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 68).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 102).cast()), 0) as u16),
            ];
            let w0 = _mm256_loadu_si256(row.add(off + 2).cast());
            let w1 = _mm256_loadu_si256(row.add(off + 36).cast());
            let w2 = _mm256_loadu_si256(row.add(off + 70).cast());
            let w3 = _mm256_loadu_si256(row.add(off + 104).cast());
            let mut c = 0usize;
            while c < k {
                let q = &qs[c];
                let x = q.codes.as_ptr().add(blk * 32);
                let x0 = _mm256_loadu_si256(x.cast());
                let x1 = _mm256_loadu_si256(x.add(32).cast());
                let x2 = _mm256_loadu_si256(x.add(64).cast());
                let x3 = _mm256_loadu_si256(x.add(96).cast());
                let d = reduce4(
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            _mm256_abs_epi8(w0),
                            _mm256_sign_epi8(x0, w0),
                        ),
                        ones,
                    ),
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            _mm256_abs_epi8(w1),
                            _mm256_sign_epi8(x1, w1),
                        ),
                        ones,
                    ),
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            _mm256_abs_epi8(w2),
                            _mm256_sign_epi8(x2, w2),
                        ),
                        ones,
                    ),
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(
                            _mm256_abs_epi8(w3),
                            _mm256_sign_epi8(x3, w3),
                        ),
                        ones,
                    ),
                );
                let s = q.scales.as_ptr().add(blk);
                let mut i = 0usize;
                while i < 4 {
                    acc[c] += i128::from(ds[i]) * i128::from(*s.add(i) * i64::from(d[i]));
                    i += 1;
                }
                c += 1;
            }
            blk += 4;
        }
    }
}

/// Activation scale folded out of the block loop and `abs` hoisted: one multiply per block.
#[target_feature(enable = "avx2")]
unsafe fn afold(w: &[u8], qs: &[Q], a: &[i64; K], acc: &mut [i128]) {
    let row_bytes = BLOCKS * 34;
    let ones = _mm256_set1_epi16(1);
    let codes: [*const i8; K] = std::array::from_fn(|c| qs[c].codes.as_ptr());
    for r in 0..N_OUT {
        let row = w.as_ptr().add(r * row_bytes);
        let mut v = [0i128; K];
        let mut blk = 0usize;
        while blk + 4 <= BLOCKS {
            let off = blk * 34;
            let ds = [
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 34).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 68).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 102).cast()), 0) as u16),
            ];
            let w0 = _mm256_loadu_si256(row.add(off + 2).cast());
            let w1 = _mm256_loadu_si256(row.add(off + 36).cast());
            let w2 = _mm256_loadu_si256(row.add(off + 70).cast());
            let w3 = _mm256_loadu_si256(row.add(off + 104).cast());
            let a0 = _mm256_abs_epi8(w0);
            let a1 = _mm256_abs_epi8(w1);
            let a2 = _mm256_abs_epi8(w2);
            let a3 = _mm256_abs_epi8(w3);
            let mut c = 0usize;
            while c < K {
                let x0 = _mm256_loadu_si256(codes[c].add(blk * 32).cast());
                let x1 = _mm256_loadu_si256(codes[c].add(blk * 32 + 32).cast());
                let x2 = _mm256_loadu_si256(codes[c].add(blk * 32 + 64).cast());
                let x3 = _mm256_loadu_si256(codes[c].add(blk * 32 + 96).cast());
                let d = reduce4(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(x0, w0)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(x1, w1)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a2, _mm256_sign_epi8(x2, w2)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a3, _mm256_sign_epi8(x3, w3)), ones),
                );
                v[c] += i128::from(ds[0]) * i128::from(d[0])
                    + i128::from(ds[1]) * i128::from(d[1])
                    + i128::from(ds[2]) * i128::from(d[2])
                    + i128::from(ds[3]) * i128::from(d[3]);
                c += 1;
            }
            blk += 4;
        }
        let mut c = 0usize;
        while c < K {
            acc[c] = i128::from(a[c]) * v[c];
            c += 1;
        }
    }
}


/// Per-block activation scale applied as a shift: `quant_act`'s scale is `2^k` by construction, so
/// `S_b·dot_b` is the dot shifted and the multiplier port carries only the `D_b` product.
#[target_feature(enable = "avx2")]
unsafe fn shift_k(w: &[u8], qs: &[Q], sh: &[[u32; BLOCKS]; K], acc: &mut [i128]) {
    let row_bytes = BLOCKS * 34;
    let ones = _mm256_set1_epi16(1);
    let codes: [*const i8; K] = std::array::from_fn(|c| qs[c].codes.as_ptr());
    for r in 0..N_OUT {
        let row = w.as_ptr().add(r * row_bytes);
        let mut v = [0i128; K];
        let mut blk = 0usize;
        while blk + 4 <= BLOCKS {
            let off = blk * 34;
            let ds = [
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 34).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 68).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 102).cast()), 0) as u16),
            ];
            let w0 = _mm256_loadu_si256(row.add(off + 2).cast());
            let w1 = _mm256_loadu_si256(row.add(off + 36).cast());
            let w2 = _mm256_loadu_si256(row.add(off + 70).cast());
            let w3 = _mm256_loadu_si256(row.add(off + 104).cast());
            let a0 = _mm256_abs_epi8(w0);
            let a1 = _mm256_abs_epi8(w1);
            let a2 = _mm256_abs_epi8(w2);
            let a3 = _mm256_abs_epi8(w3);
            let mut c = 0usize;
            while c < K {
                let x0 = _mm256_loadu_si256(codes[c].add(blk * 32).cast());
                let x1 = _mm256_loadu_si256(codes[c].add(blk * 32 + 32).cast());
                let x2 = _mm256_loadu_si256(codes[c].add(blk * 32 + 64).cast());
                let x3 = _mm256_loadu_si256(codes[c].add(blk * 32 + 96).cast());
                let h01 = _mm256_hadd_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(x0, w0)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(x1, w1)), ones),
                );
                let h23 = _mm256_hadd_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a2, _mm256_sign_epi8(x2, w2)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a3, _mm256_sign_epi8(x3, w3)), ones),
                );
                let h = _mm256_hadd_epi32(h01, h23);
                let d = _mm_add_epi32(
                    _mm256_castsi256_si128(h),
                    _mm256_extracti128_si256::<1>(h),
                );
                let mut q = [0i32; 4];
                _mm_storeu_si128(q.as_mut_ptr().cast(), d);
                let s = sh[c].as_ptr().add(blk);
                v[c] += i128::from(ds[0]) * i128::from(i64::from(q[0]).wrapping_shl(*s))
                    + i128::from(ds[1]) * i128::from(i64::from(q[1]).wrapping_shl(*s.add(1)))
                    + i128::from(ds[2]) * i128::from(i64::from(q[2]).wrapping_shl(*s.add(2)))
                    + i128::from(ds[3]) * i128::from(i64::from(q[3]).wrapping_shl(*s.add(3)));
                c += 1;
            }
            blk += 4;
        }
        acc[..K].copy_from_slice(&v);
    }
}


fn f16_split(h: u16) -> Option<(i32, u32)> {
    let e = u32::from((h >> 10) & 0x1f);
    let f = (h & 0x3ff) as i32;
    if e == 31 {
        return None;
    }
    let (m, k) = if e == 0 { (f, 8) } else { (f + 1024, e + 7) };
    if h >> 15 == 1 {
        Some((-m, k))
    } else {
        Some((m, k))
    }
}

/// Per-block activation exponent as a lane shift: `D_b·S_b·dot_b = (m_b·dot_b) << (k_b − k_min +
/// e_b − e_min)`, then one shift by `k_min + e_min` per row and column. The gemm's own mechanism,
/// on the pooled row loop.
#[target_feature(enable = "avx2")]
unsafe fn shiftlane(
    w: &[u8],
    qs: &[Q],
    col_shift: &[[i64; BLOCKS]; K],
    emin: &[u32; K],
    acc: &mut [i128],
) {
    let row_bytes = BLOCKS * 34;
    let ones = _mm256_set1_epi16(1);
    let codes: [*const i8; K] = std::array::from_fn(|c| qs[c].codes.as_ptr());
    for r in 0..N_OUT {
        let row = w.as_ptr().add(r * row_bytes);
        let mut mv = [0i32; BLOCKS];
        let mut kk = [0u32; BLOCKS];
        for b in 0..BLOCKS {
            let word = u16::from_le_bytes([*row.add(b * 34), *row.add(b * 34 + 1)]);
            let (m, k) = f16_split(word).unwrap();
            mv[b] = m;
            kk[b] = k;
        }
        let kmin = *kk.iter().min().unwrap();
        let mut v = [_mm256_setzero_si256(); K];
        let mut blk = 0usize;
        while blk + 4 <= BLOCKS {
            let off = blk * 34;
            let w0 = _mm256_loadu_si256(row.add(off + 2).cast());
            let w1 = _mm256_loadu_si256(row.add(off + 36).cast());
            let w2 = _mm256_loadu_si256(row.add(off + 70).cast());
            let w3 = _mm256_loadu_si256(row.add(off + 104).cast());
            let a0 = _mm256_abs_epi8(w0);
            let a1 = _mm256_abs_epi8(w1);
            let a2 = _mm256_abs_epi8(w2);
            let a3 = _mm256_abs_epi8(w3);
            let mvv = _mm256_cvtepi32_epi64(_mm_set_epi32(
                mv[blk + 3],
                mv[blk + 2],
                mv[blk + 1],
                mv[blk],
            ));
            let rv = _mm256_cvtepi32_epi64(_mm_set_epi32(
                (kk[blk + 3] - kmin) as i32,
                (kk[blk + 2] - kmin) as i32,
                (kk[blk + 1] - kmin) as i32,
                (kk[blk] - kmin) as i32,
            ));
            let mut c = 0usize;
            while c < K {
                let x0 = _mm256_loadu_si256(codes[c].add(blk * 32).cast());
                let x1 = _mm256_loadu_si256(codes[c].add(blk * 32 + 32).cast());
                let x2 = _mm256_loadu_si256(codes[c].add(blk * 32 + 64).cast());
                let x3 = _mm256_loadu_si256(codes[c].add(blk * 32 + 96).cast());
                let h01 = _mm256_hadd_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(x0, w0)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(x1, w1)), ones),
                );
                let h23 = _mm256_hadd_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a2, _mm256_sign_epi8(x2, w2)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a3, _mm256_sign_epi8(x3, w3)), ones),
                );
                let h = _mm256_hadd_epi32(h01, h23);
                let d = _mm_add_epi32(
                    _mm256_castsi256_si128(h),
                    _mm256_extracti128_si256::<1>(h),
                );
                let dv = _mm256_cvtepi32_epi64(d);
                let p = _mm256_mul_epi32(dv, mvv);
                let cs = _mm256_loadu_si256(col_shift[c].as_ptr().add(blk).cast());
                let s = _mm256_add_epi64(rv, cs);
                v[c] = _mm256_add_epi64(v[c], _mm256_sllv_epi64(p, s));
                c += 1;
            }
            blk += 4;
        }
        let mut c = 0usize;
        while c < K {
            let lo = _mm256_castsi256_si128(v[c]);
            let hi = _mm256_extracti128_si256::<1>(v[c]);
            let s = _mm_add_epi64(lo, hi);
            let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
            acc[c] = i128::from(_mm_cvtsi128_si64(s)) << (kmin + emin[c]);
            c += 1;
        }
    }
}

fn checksum(v: &[i128]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in v {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn run(name: &str, f: impl Fn(&mut [i128]) -> u64) -> (f64, u64) {
    let mut acc = vec![0i128; K];
    let mut best = f64::INFINITY;
    let mut cs = 0u64;
    for _ in 0..4 {
        acc.fill(0);
        let t = Instant::now();
        cs = f(&mut acc);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best {
            best = ms;
        }
    }
    let gmac = N_OUT as f64 * N_IN as f64 * K as f64 / 1e9;
    eprintln!(
        "{name:>7}: {best:7.3} ms  {:5.1} GMAC/s  {:4.1} cycles/unit  checksum {cs:016x}",
        gmac / (best / 1e3),
        best * 1e-3 * 3.6e9 / (N_OUT as f64 * (BLOCKS / 4) as f64 * K as f64)
    );
    (best, cs)
}

fn main() {
    let w = fill_weights();
    let qs = activations();
    let a = [1i64 << 30; K];
    let mut ref_acc = vec![0i128; K];
    unsafe {
        dyn_k(&w, &qs, K, &mut ref_acc);
    }
    let ref_cs = checksum(&ref_acc);
    let (d, c0) = run("dyn", |acc| unsafe {
        dyn_k(&w, &qs, K, acc);
        checksum(acc)
    });
    let (f, c1) = run("afold", |acc| unsafe {
        afold(&w, &qs, &a, acc);
        checksum(acc)
    });
    let sh: [[u32; BLOCKS]; K] = std::array::from_fn(|c| {
        let mut a = [0u32; BLOCKS];
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = qs[c].scales[i].trailing_zeros();
        }
        a
    });
    let mut col_shift = [[0i64; BLOCKS]; K];
    let mut emin = [0u32; K];
    for c in 0..K {
        let e: Vec<u32> = qs[c].scales.iter().map(|s| u64::from(*s as u64).trailing_zeros()).collect();
        let m = *e.iter().min().unwrap();
        emin[c] = m;
        for b in 0..BLOCKS {
            col_shift[c][b] = i64::from(e[b] - m);
        }
    }
    let (f4, c4) = run("shlane", |acc| unsafe {
        shiftlane(&w, &qs, &col_shift, &emin, acc);
        checksum(acc)
    });
    let (f3, c3) = run("shift", |acc| unsafe {
        shift_k(&w, &qs, &sh, acc);
        checksum(acc)
    });
    let (f2, c2) = run("afold64", |acc| unsafe {
        afold64(&w, &qs, &a, acc);
        checksum(acc)
    });
    eprintln!(
        "exact: dyn={} afold={} afold64={} shift={} shlane={}",
        c0 == ref_cs,
        c1 == ref_cs,
        c2 == ref_cs,
        c3 == ref_cs,
        c4 == ref_cs
    );
    let (dd, _) = run("dyn1", |acc| unsafe {
        dyn_k(&w, &qs, 1, &mut acc[..1]);
        checksum(&acc[..1])
    });
    eprintln!(
        "afold speedup {:.2}x; k=1 floor {:.3} ms",
        d / f,
        dd
    );
}

/// Vector epilogue: the four block dots sign-extend to i64 lanes, one `vpmuldq` forms the four
/// `D_b·dot_b` products in `i64`, and the four lanes accumulate across every group. One
/// horizontal sum per row, then the activation scale in `i128`.
///
/// The lanes hold `8` blocks each, so the lane bound is
/// `8·|D|max·2^19`; with `|D|max < 2^31` it is below `2^54`, and `A·Σ` is the reference's own
/// accumulator, so the value is the reference's. `D` is the fixed `f16` block scale.
#[target_feature(enable = "avx2")]
unsafe fn afold64(w: &[u8], qs: &[Q], a: &[i64; K], acc: &mut [i128]) {
    let row_bytes = BLOCKS * 34;
    let ones = _mm256_set1_epi16(1);
    let codes: [*const i8; K] = std::array::from_fn(|c| qs[c].codes.as_ptr());
    for r in 0..N_OUT {
        let row = w.as_ptr().add(r * row_bytes);
        let mut v = [_mm256_setzero_si256(); K];
        let mut blk = 0usize;
        while blk + 4 <= BLOCKS {
            let off = blk * 34;
            let ds = [
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 34).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 68).cast()), 0) as u16),
                f16_to_fixed(_mm_extract_epi16(_mm_loadu_si128(row.add(off + 102).cast()), 0) as u16),
            ];
            let sv = _mm256_cvtepi32_epi64(_mm_set_epi32(
                ds[3] as i32,
                ds[2] as i32,
                ds[1] as i32,
                ds[0] as i32,
            ));
            let w0 = _mm256_loadu_si256(row.add(off + 2).cast());
            let w1 = _mm256_loadu_si256(row.add(off + 36).cast());
            let w2 = _mm256_loadu_si256(row.add(off + 70).cast());
            let w3 = _mm256_loadu_si256(row.add(off + 104).cast());
            let a0 = _mm256_abs_epi8(w0);
            let a1 = _mm256_abs_epi8(w1);
            let a2 = _mm256_abs_epi8(w2);
            let a3 = _mm256_abs_epi8(w3);
            let mut c = 0usize;
            while c < K {
                let x0 = _mm256_loadu_si256(codes[c].add(blk * 32).cast());
                let x1 = _mm256_loadu_si256(codes[c].add(blk * 32 + 32).cast());
                let x2 = _mm256_loadu_si256(codes[c].add(blk * 32 + 64).cast());
                let x3 = _mm256_loadu_si256(codes[c].add(blk * 32 + 96).cast());
                let h01 = _mm256_hadd_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(x0, w0)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(x1, w1)), ones),
                );
                let h23 = _mm256_hadd_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a2, _mm256_sign_epi8(x2, w2)), ones),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(a3, _mm256_sign_epi8(x3, w3)), ones),
                );
                let h = _mm256_hadd_epi32(h01, h23);
                let d = _mm_add_epi32(
                    _mm256_castsi256_si128(h),
                    _mm256_extracti128_si256::<1>(h),
                );
                let dv = _mm256_cvtepi32_epi64(d);
                v[c] = _mm256_add_epi64(v[c], _mm256_mul_epi32(dv, sv));
                c += 1;
            }
            blk += 4;
        }
        let mut c = 0usize;
        while c < K {
            let lo = _mm256_castsi256_si128(v[c]);
            let hi = _mm256_extracti128_si256::<1>(v[c]);
            let s = _mm_add_epi64(lo, hi);
            let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
            acc[c] = i128::from(a[c]) * i128::from(_mm_cvtsi128_si64(s));
            c += 1;
        }
    }
}
