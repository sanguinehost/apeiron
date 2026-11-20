// Scratch microbenchmark, hot fixture: the activation codes stay in L1 and the weight
// words cycle over a small buffer, so the loop is instruction-bound rather than DRAM-bound.
// Usage: bench2 [shuffle|min|sd|pop|all] [reps]
use std::arch::x86_64::*;
use std::time::Instant;

const INN: usize = 4096;
const WORDS: usize = INN / 32;
const ROWS: usize = 16;
const DOTS: usize = 4096;

#[inline(always)]
unsafe fn hsum_i32(acc: __m256i) -> i64 {
    let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc));
    let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(acc));
    let quad = _mm256_add_epi64(lo, hi);
    let half = _mm_add_epi64(
        _mm256_castsi256_si128(quad),
        _mm256_extracti128_si256::<1>(quad),
    );
    let half = _mm_add_epi64(half, _mm_unpackhi_epi64(half, half));
    _mm_cvtsi128_si64(half)
}

#[inline(always)]
unsafe fn hsum_i64(acc: __m256i) -> i64 {
    let half = _mm_add_epi64(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let half = _mm_add_epi64(half, _mm_unpackhi_epi64(half, half));
    _mm_cvtsi128_si64(half)
}



#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn ces_shuffle(codes: &[i8], shifts: &[i32], words: &[u32], len: usize) -> i64 {
    let digit_mask = _mm256_set1_epi32(0x0804_0201u32 as i32);
    let bit_at = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    let repeat = _mm256_setr_epi8(
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8,
        12, 12, 12, 12,
    );
    let select_one = _mm256_setr_epi8(
        0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0,
    );
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    let full = len / 32;
    for w in 0..full {
        let spread = _mm256_srlv_epi32(_mm256_set1_epi32(words[w] as i32), bit_at);
        let nibble = _mm256_shuffle_epi8(spread, repeat);
        let select = _mm256_shuffle_epi8(select_one, _mm256_and_si256(nibble, digit_mask));
        let plane = _mm256_loadu_si256(codes.as_ptr().add(w * 32).cast::<__m256i>());
        let pair = _mm256_maddubs_epi16(select, plane);
        let part = _mm256_madd_epi16(pair, ones);
        let sv = _mm256_loadu_si256(shifts.as_ptr().add(w * 8).cast::<__m256i>());
        acc = _mm256_add_epi32(acc, _mm256_sllv_epi32(part, sv));
    }
    hsum_i32(acc)
}

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn ces_min(codes: &[i8], shifts: &[i32], words: &[u32], len: usize) -> i64 {
    let digit_mask = _mm256_set1_epi32(0x0804_0201u32 as i32);
    let bit_at = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    let repeat = _mm256_setr_epi8(
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8,
        12, 12, 12, 12,
    );
    let one_byte = _mm256_set1_epi8(1);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    let full = len / 32;
    for w in 0..full {
        let spread = _mm256_srlv_epi32(_mm256_set1_epi32(words[w] as i32), bit_at);
        let nibble = _mm256_shuffle_epi8(spread, repeat);
        let select = _mm256_min_epu8(_mm256_and_si256(nibble, digit_mask), one_byte);
        let plane = _mm256_loadu_si256(codes.as_ptr().add(w * 32).cast::<__m256i>());
        let pair = _mm256_maddubs_epi16(select, plane);
        let part = _mm256_madd_epi16(pair, ones);
        let sv = _mm256_loadu_si256(shifts.as_ptr().add(w * 8).cast::<__m256i>());
        acc = _mm256_add_epi32(acc, _mm256_sllv_epi32(part, sv));
    }
    hsum_i32(acc)
}

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn sd_current(t: &[i64], words: &[u32], len: usize) -> i64 {
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi64x(1);
    let lane = _mm256_setr_epi64x(0, 1, 2, 3);
    let mut acc = zero;
    let mut i = 0usize;
    for &word in words {
        if i + 32 > len {
            break;
        }
        let mut bits = _mm256_set1_epi64x(i64::from(word));
        let mut g = 0usize;
        while g < 8 {
            let at = i + g * 4;
            let v = _mm256_loadu_si256(t.as_ptr().add(at).cast::<__m256i>());
            let bit = _mm256_and_si256(_mm256_srlv_epi64(bits, lane), one);
            let mask = _mm256_sub_epi64(zero, bit);
            acc = _mm256_add_epi64(acc, _mm256_sub_epi64(_mm256_xor_si256(v, mask), mask));
            bits = _mm256_srli_epi64::<4>(bits);
            g += 1;
        }
        i += 32;
    }
    let total = hsum_i64(acc);
    if i < len {
        let mut tail = 0i64;
        let mut j = i;
        while j < len {
            let bit = (words[j / 32] >> (j % 32)) & 1;
            tail += if bit == 0 { t[j] } else { -t[j] };
            j += 1;
        }
        total + tail
    } else {
        total
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
unsafe fn sd_popcount(t: &[i64], words: &[u32], len: usize) -> i64 {
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi64x(1);
    let lane = _mm256_setr_epi64x(0, 1, 2, 3);
    let mut acc = zero;
    let mut i = 0usize;
    let mut bits_total = 0i64;
    for &word in words {
        if i + 32 > len {
            break;
        }
        bits_total += i64::from(word.count_ones());
        let mut bits = _mm256_set1_epi64x(i64::from(word));
        let mut g = 0usize;
        while g < 8 {
            let at = i + g * 4;
            let v = _mm256_loadu_si256(t.as_ptr().add(at).cast::<__m256i>());
            let bit = _mm256_and_si256(_mm256_srlv_epi64(bits, lane), one);
            let mask = _mm256_sub_epi64(zero, bit);
            acc = _mm256_add_epi64(acc, _mm256_xor_si256(v, mask));
            bits = _mm256_srli_epi64::<4>(bits);
            g += 1;
        }
        i += 32;
    }
    let total = hsum_i64(acc) + bits_total;
    if i < len {
        let mut tail = 0i64;
        let mut j = i;
        while j < len {
            let bit = (words[j / 32] >> (j % 32)) & 1;
            tail += if bit == 0 { t[j] } else { -t[j] };
            j += 1;
        }
        total + tail
    } else {
        total
    }
}

fn next_u32(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let mut state = 0x00c0_ffee_0bad_f00du64;
    let t: Vec<i64> = (0..INN)
        .map(|_| i64::from(next_u32(&mut state) as i32))
        .collect();
    let codes: Vec<i8> = (0..INN * 2)
        .map(|_| (next_u32(&mut state) % 255) as i8)
        .collect();
    let words: Vec<u32> = (0..ROWS * WORDS).map(|_| next_u32(&mut state)).collect();
    let shifts: Vec<i32> = (0..WORDS * 8)
        .map(|_| (next_u32(&mut state) % 8) as i32)
        .collect();

    let run_ces = |f: unsafe fn(&[i8], &[i32], &[u32], usize) -> i64| -> i64 {
        let mut acc = 0i64;
        for d in 0..DOTS {
            let r = d % ROWS;
            acc = acc.wrapping_add(unsafe {
                f(&codes[..INN], &shifts, &words[r * WORDS..(r + 1) * WORDS], INN)
            });
        }
        acc
    };
    let run_sd = |f: unsafe fn(&[i64], &[u32], usize) -> i64| -> i64 {
        let mut acc = 0i64;
        for d in 0..DOTS {
            let r = d % ROWS;
            acc = acc.wrapping_add(unsafe { f(&t, &words[r * WORDS..(r + 1) * WORDS], INN) });
        }
        acc
    };

    let a = run_ces(ces_shuffle);
    let b = run_ces(ces_min);
    let c = run_sd(sd_current);
    let e = run_sd(sd_popcount);
    assert_eq!(a, b, "min differs from shuffle");
    assert_eq!(c, e, "popcount dot differs");
    if mode != "all" {
        let pick = match mode.as_str() {
            "shuffle" => run_ces(ces_shuffle),
            "min" => run_ces(ces_min),
            "sd" => run_sd(sd_current),
            "pop" => run_sd(sd_popcount),
            other => panic!("unknown mode {other}"),
        };
        println!("{mode} {pick}");
        return;
    }
    println!("equality ok: ces {a}=={b} sd {c}=={e}");

    // Interleaved samples; the minimum is the estimator, so a scheduler slice on any one
    // sample does not move the result.
    let sample = |f: &dyn Fn() -> i64, reps: usize| -> f64 {
        let start = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(f());
        }
        start.elapsed().as_nanos() as f64 / reps as f64
    };
    let (mut a, mut b, mut c, mut e) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
    for _ in 0..15 {
        a = a.min(sample(&|| run_ces(ces_shuffle), 20));
        b = b.min(sample(&|| run_ces(ces_min), 20));
        c = c.min(sample(&|| run_sd(sd_current), 5));
        e = e.min(sample(&|| run_sd(sd_popcount), 5));
    }
    let blocks = (DOTS * WORDS) as f64;
    for (name, ns) in [
        ("code_select shuffle", a),
        ("code_select min    ", b),
        ("sign_dot    current", c),
        ("sign_dot    popcnt ", e),
    ] {
        println!(
            "{name} {ns:9.0} ns/pass  {:.3} ns/word",
            ns * 1e3 / blocks
        );
    }
    println!("min/shuffle {:.4}   pop/current {:.4}", b / a, e / c);
}
