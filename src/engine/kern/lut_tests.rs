#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "test-only fixture arithmetic on bounded small values; the timing test divides \
                  nanosecond counts by a weight count for a reported ns/weight, not a value used \
                  in a proof"
)]

use super::*;
use crate::engine::chart::quant_act;
use crate::format::quant::GGML_TYPE_PQ2_0;
use std::time::Instant;

/// xorshift64: a deterministic code/activation stream for the equality tests.
fn next_rand(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// A `rows × n_in` `PQ2_0` payload with random code bytes and every block scale overwritten by
/// `d_bits`, so each block is well formed while all four codes appear.
fn random_payload(n_out: usize, n_in: usize, d_bits: u16, seed: u64) -> Vec<u8> {
    let blocks = n_in / PQ2_0_QK;
    let row_bytes = blocks * PQ2_0_BLOCK_BYTES;
    let mut raw = vec![0u8; n_out * row_bytes];
    let mut state = seed;
    let mut i = 0usize;
    while i < raw.len() {
        raw[i] = (next_rand(&mut state) & 0xff) as u8;
        i += 1;
    }
    let mut row = 0usize;
    while row < n_out {
        let mut b = 0usize;
        while b < blocks {
            let o = row * row_bytes + b * PQ2_0_BLOCK_BYTES;
            raw[o..o + 2].copy_from_slice(&d_bits.to_le_bytes());
            b += 1;
        }
        row += 1;
    }
    raw
}

/// `n` small Q32.32 activations in `[-2048, 2047]`, from a deterministic stream.
fn random_x(n: usize, seed: u64) -> Vec<i64> {
    let mut state = seed;
    let mut x: Vec<i64> = Vec::with_capacity(n);
    while x.len() < n {
        x.push((next_rand(&mut state) % 4096) as i64 - 2048);
    }
    x
}

/// The lookup kernel equals `matvec_fixed` on the dequantized activation for random payloads
/// (all four codes), several widths and several output counts including ragged 32-row tiles.
#[test]
fn matvec_lut_matches_fixed_on_random_payloads() {
    const N_OUT: [usize; 5] = [1, 31, 32, 33, 100];
    const N_IN: [usize; 3] = [128, 256, 1024];
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for &n_in in &N_IN {
        for &n_out in &N_OUT {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let raw = random_payload(n_out, n_in, 0x3800, seed);
            let q = quant_act(&random_x(n_in, seed ^ 0xdead_beef)).expect("dequantizes");
            let deq = q.dequant().expect("quant_act guarantees dequant");
            let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
            assert_eq!(
                matvec_lut(&lut, &q),
                matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq),
                "shape ({n_out},{n_in}) diverged"
            );
        }
    }
}

/// Extreme activation codes `±127` (unit scale) against the largest finite and the smallest
/// positive block scales: the lookup value is exact in both cases.
#[test]
fn matvec_lut_matches_fixed_on_extreme_codes_and_scales() {
    for &(d_bits, n_in) in &[(0x7BFFu16, 128usize), (0x3C00, 1024), (0x0001, 256)] {
        let n_out = 33usize;
        let raw = random_payload(n_out, n_in, d_bits, 0x1234 ^ n_in as u64);
        let x: Vec<i64> = (0..n_in)
            .map(|j| if j % 2 == 0 { 127 } else { -127 })
            .collect();
        let q = quant_act(&x).expect("±127 dequantizes");
        assert_eq!(q.scales()[0], 1, "unit scale expected");
        assert_eq!(q.codes()[0], 127);
        assert_eq!(q.codes()[1], -127);
        let deq = q.dequant().expect("unit scale cannot overflow");
        let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
        assert_eq!(
            matvec_lut(&lut, &q),
            matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq),
            "extreme d={d_bits:#06x} n_in={n_in} diverged"
        );
    }
}

/// A zero activation quantizes to zero codes and zero scales; the lookup result is the exact
/// all-zero projection the reference produces.
#[test]
fn matvec_lut_matches_fixed_on_zero_activations() {
    let n_in = 256usize;
    let n_out = 33usize;
    let raw = random_payload(n_out, n_in, 0x3800, 0x77);
    let q = quant_act(&vec![0i64; n_in]).expect("zero quantizes");
    assert!(q.codes().iter().all(|&c| c == 0));
    assert!(q.scales().iter().all(|&s| s == 0));
    let deq = q.dequant().expect("zero dequantizes");
    let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
    assert_eq!(
        matvec_lut(&lut, &q),
        matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq)
    );
}

/// The `AVX2` `vpshufb` core returns exactly the portable twin's rows, including ragged
/// 32-row tiles.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_and_portable_lut_agree() {
    const SHAPES: [(usize, usize); 6] = [
        (1, 128),
        (31, 128),
        (32, 128),
        (33, 256),
        (100, 1024),
        (64, 1024),
    ];
    if !avx2() {
        return;
    }
    for &(n_out, n_in) in &SHAPES {
        let raw = random_payload(n_out, n_in, 0x7BFF, 0xabc ^ n_in as u64);
        let q = quant_act(&random_x(n_in, 0x5eed ^ n_out as u64)).expect("dequantizes");
        let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
        assert_eq!(
            matvec_lut_with(&lut, &q, true),
            matvec_lut_with(&lut, &q, false),
            "shape ({n_out},{n_in}) diverged"
        );
    }
}

/// When the call-level bound cannot prove the `i128` accumulation stays in range, the kernel
/// takes the reference fallback and reproduces its `Ok`/`Err` value exactly.
#[test]
fn matvec_lut_falls_back_when_the_bound_is_exceeded() {
    let n_in = 32768usize;
    let n_out = 1usize;
    let raw = random_payload(n_out, n_in, 0x7BFF, 0x1a2b);
    let q = quant_act(&vec![127i64 << 56; n_in])
        .expect("127·2^56 dequantizes on the power-of-two grid");
    let deq = q.dequant().expect("quant_act guarantees dequant");
    let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
    assert!(
        !lut_bound_ok(n_in, q.scales()),
        "the test shape must exceed the bound"
    );
    assert_eq!(
        matvec_lut(&lut, &q),
        matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq)
    );
}

/// The constructor refuses a non-finite scale, a short payload and a width that is not a whole
/// number of 128-weight blocks.
#[test]
fn from_pq2_0_refuses_bad_inputs() {
    let n_in = 128usize;
    let n_out = 2usize;
    let mut raw = random_payload(n_out, n_in, 0x3800, 9);
    raw[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
    assert_eq!(
        Pq2Lut::from_pq2_0(&raw, n_out, n_in).unwrap_err(),
        MatvecError::Dequant(EngineError::NonFiniteScale { bits: 0x7C00 })
    );
    assert_eq!(
        Pq2Lut::from_pq2_0(&raw[..raw.len() - 1], n_out, n_in).unwrap_err(),
        MatvecError::ShortPayload {
            need: raw.len(),
            have: raw.len() - 1
        }
    );
    assert!(matches!(
        Pq2Lut::from_pq2_0(&[], 1, 127),
        Err(MatvecError::RowLength { .. })
    ));
}

/// An activation whose largest block scale leaves the `i64` lane bound but stays inside the
/// `i128` bound takes the checked portable path and still reproduces `matvec_fixed` bit for bit;
/// the `AVX2` and portable twins agree there too.
#[test]
fn matvec_lut_wide_path_matches_fixed() {
    let n_in = 128usize;
    let n_out = 33usize;
    let raw = random_payload(n_out, n_in, 0x3800, 0x55aa);
    let x: Vec<i64> = (0..n_in)
        .map(|j| {
            if j % 3 == 0 {
                20_000_000_000_000_000
            } else {
                -19_999_999_999_999_999
            }
        })
        .collect();
    let q = quant_act(&x).expect("dequantizes");
    assert!(
        !lut_i64_bound_ok(q.scales()),
        "the scale must leave the i64 lane bound"
    );
    assert!(
        lut_bound_ok(n_in, q.scales()),
        "the scale must stay inside the i128 bound"
    );
    let deq = q.dequant().expect("quant_act guarantees dequant");
    let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
    assert_eq!(
        matvec_lut(&lut, &q),
        matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq)
    );
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        assert_eq!(
            matvec_lut_with(&lut, &q, true),
            matvec_lut_with(&lut, &q, false),
            "the wide path's AVX2 and portable twins diverged"
        );
    }
}

/// The `Pool`-dispatched, 32-row-aligned chunked projection equals the unchunked
/// [`matvec_lut`] and `matvec_fixed(142, raw, n_out, n_in, &dequant(quant_act(x)))` bit for
/// bit, on ragged row counts and several chunk budgets.
///
/// The chunks start on [`TILE_ROWS`] boundaries (the last is the ragged remainder), so a
/// misaligned start is a seeded defect this test catches.
#[test]
fn pooled_row_chunked_lut_equals_the_unchunked_kernel() {
    const SHAPES: [(usize, usize); 5] = [(33, 128), (100, 256), (64, 1024), (35, 128), (32, 384)];
    for &(n_out, n_in) in &SHAPES {
        let raw = random_payload(n_out, n_in, 0x3800, 0x00C0_FFEE ^ n_out as u64);
        let x = random_x(n_in, 0x0000_ABCD ^ n_in as u64);
        let q = quant_act(&x).expect("dequantizes");
        let deq = q.dequant().expect("quant_act guarantees dequant");
        let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
        let want_fixed = matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq).expect("fixed");
        let want_lut = matvec_lut(&lut, &q).expect("unchunked lut");
        assert_eq!(want_lut, want_fixed, "shape ({n_out},{n_in}) lut vs fixed");
        for chunks in [1usize, 2, 3, 8, 16, 64] {
            let got = matvec_lut_chunked(&lut, &x, chunks)
                .unwrap_or_else(|e| panic!("shape ({n_out},{n_in}) chunks {chunks}: {e:?}"));
            assert_eq!(got, want_lut, "shape ({n_out},{n_in}) chunks {chunks}");
        }
        // The row-range entry equals the whole-row kernel on its 32-row-aligned window.
        let rows = 32.min(n_out);
        let mut out = vec![0i64; rows];
        matvec_lut_rows(&lut, &q, 0, rows, &mut out).expect("rows");
        assert!(
            out == want_lut[..rows],
            "shape ({n_out},{n_in}) rows[0..{rows}]"
        );
    }
}

/// A comparison of the lookup kernel against the reference on the `2048 × 2048` shape, best of
/// `REPS` runs so a single scheduler hiccup cannot set the number. Not part of the acceptance
/// gate; run with `--ignored --nocapture` and read the ns/weight.
#[test]
#[ignore = "timing measurement; run with --ignored --nocapture"]
fn timing_lut_vs_fixed_2048() {
    const REPS: usize = 20;
    let n_out = 2048usize;
    let n_in = 2048usize;
    let raw = random_payload(n_out, n_in, 0x3800, 0xfeed);
    let q = quant_act(&random_x(n_in, 0xbeef)).expect("dequantizes");
    let deq = q.dequant().expect("quant_act guarantees dequant");
    let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
    assert_eq!(
        matvec_lut(&lut, &q),
        matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq)
    );
    assert!(
        lut_i64_bound_ok(q.scales()),
        "the timing shape must take the i64 fast path"
    );
    let weights = (n_out * n_in) as f64;
    std::hint::black_box(matvec_lut(&lut, &q).expect("lut"));
    std::hint::black_box(matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq).expect("fixed"));
    let mut lut_ns = f64::MAX;
    let mut fixed_ns = f64::MAX;
    let mut rep = 0usize;
    while rep < REPS {
        let t = Instant::now();
        std::hint::black_box(matvec_lut(&lut, &q).expect("lut"));
        lut_ns = lut_ns.min(t.elapsed().as_nanos() as f64);
        let t = Instant::now();
        std::hint::black_box(
            matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq).expect("fixed"),
        );
        fixed_ns = fixed_ns.min(t.elapsed().as_nanos() as f64);
        rep += 1;
    }
    println!(
        "matvec_lut   {:.4} ns/weight  ({:.4} ms, best of {REPS})",
        lut_ns / weights,
        lut_ns / 1e6
    );
    println!(
        "matvec_fixed {:.4} ns/weight  ({:.4} ms, best of {REPS})",
        fixed_ns / weights,
        fixed_ns / 1e6
    );
}

/// A `rows × n_in` `F16` payload (`n_in` arbitrary) drawn from a small finite palette.
fn f16_payload(rows: usize, n_in: usize) -> Vec<u8> {
    const VALS: [u16; 8] = [
        0x0000, 0x3000, 0xB000, 0x3400, 0xB400, 0x3800, 0xB800, 0x3C00,
    ];
    let mut out: Vec<u8> = Vec::with_capacity(rows * n_in * 2);
    let mut i = 0usize;
    while i < rows * n_in {
        out.extend_from_slice(&VALS[(i * 3 + 1) % 8].to_le_bytes());
        i += 1;
    }
    out
}

/// One N-projection dispatch over one activation equals the N separate unchunked projections bit
/// for bit — a `PQ2_0` LUT projection mixed with the `F16` reference chart plan, across chunk
/// budgets — and a short payload refuses in projection-then-row order.
#[test]
fn n_projection_dispatch_matches_separate_calls_and_refuses() {
    use crate::engine::chart::ffn::{projections_q8act, ProjSource, ProjSpec};
    use crate::engine::chart::matvec_q8act;
    use crate::format::quant::GGML_TYPE_F16;

    const N_IN: usize = 256;
    const PQ_ROWS: usize = 96;
    const CH_ROWS: usize = 40;
    let x = random_x(N_IN, 0x51ed);
    let q = quant_act(&x).expect("dequantizes");
    let pq_raw = random_payload(PQ_ROWS, N_IN, 0x3C00, 0xfeed);
    let lut = Pq2Lut::from_pq2_0(&pq_raw, PQ_ROWS, N_IN).expect("well-formed");
    let ch_raw = f16_payload(CH_ROWS, N_IN);
    let specs = [
        ProjSpec {
            n_out: PQ_ROWS,
            row_align: TILE_ROWS,
            source: ProjSource::Rows(&lut),
        },
        ProjSpec {
            n_out: CH_ROWS,
            row_align: 1,
            source: ProjSource::Chart {
                ggml_type: GGML_TYPE_F16,
                raw: &ch_raw,
            },
        },
    ];
    let want = [
        matvec_lut(&lut, &q).expect("lut"),
        matvec_q8act(GGML_TYPE_F16, &ch_raw, CH_ROWS, N_IN, &q).expect("chart"),
    ];
    for chunks in [2usize, 8, 32] {
        let got = projections_q8act(N_IN, &x, &specs, chunks).expect("dispatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "projection {i}, chunks {chunks}");
        }
    }
    // Projection 0 succeeds; the short projection 1 refuses with the same value the separate
    // unchunked call returns, so the failure is the first in projection order.
    let short = &ch_raw[..ch_raw.len() - 1];
    let bad = [
        ProjSpec {
            n_out: PQ_ROWS,
            row_align: TILE_ROWS,
            source: ProjSource::Rows(&lut),
        },
        ProjSpec {
            n_out: CH_ROWS,
            row_align: 1,
            source: ProjSource::Chart {
                ggml_type: GGML_TYPE_F16,
                raw: short,
            },
        },
    ];
    let want_err = matvec_q8act(GGML_TYPE_F16, short, CH_ROWS, N_IN, &q).expect_err("short");
    assert!(
        matches!(want_err, MatvecError::ShortPayload { .. }),
        "{want_err:?}"
    );
    assert_eq!(projections_q8act(N_IN, &x, &bad, 8), Err(want_err));
}

/// Per-call cost of the serial and pooled table builds, min of many reps. Not part of the
/// acceptance gate; run with `--ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run with --ignored --nocapture"]
fn timing_build_tables_serial_vs_pooled() {
    for n_in in [1024usize, 2048, 6144] {
        let x = random_x(n_in, 0x1234 ^ n_in as u64);
        let q = quant_act(&x).expect("dequantizes");
        let codes = q.codes();
        for _ in 0..200 {
            std::hint::black_box(build_tables(codes));
            std::hint::black_box(build_tables_pooled(codes));
        }
        let mut serial = f64::MAX;
        let mut pooled = f64::MAX;
        let mut s = 0usize;
        while s < 1000 {
            let t = Instant::now();
            std::hint::black_box(build_tables(codes));
            serial = serial.min(t.elapsed().as_nanos() as f64);
            let t = Instant::now();
            std::hint::black_box(build_tables_pooled(codes));
            pooled = pooled.min(t.elapsed().as_nanos() as f64);
            s += 1;
        }
        println!(
            "n_in={n_in} groups={} serial={serial:.0} ns pooled={pooled:.0} ns",
            n_in / 4
        );
    }
}

/// The independent table oracle: `P_g[n] = Σ_{k<4} bit_k(n)·code_{4g+k}` (the `planeSum` identity
/// of `Numeric/LutPQ2.lean`), split as `P + 508 = lo + 256·hi` with `xs[g] = P_g[15]`.
fn plane_sum_oracle(codes: &[i8]) -> (Vec<u8>, Vec<u8>, Vec<i16>) {
    let n_groups = codes.len() / 4;
    let mut lo = vec![0u8; n_groups * 16];
    let mut hi = vec![0u8; n_groups * 16];
    let mut xs = vec![0i16; n_groups];
    let mut g = 0usize;
    while g < n_groups {
        let mut n = 0usize;
        while n < 16 {
            let mut p: i32 = 0;
            let mut k = 0usize;
            while k < 4 {
                if (n >> k) & 1 == 1 {
                    p += i32::from(codes[4 * g + k]);
                }
                k += 1;
            }
            let enc = (p + 508) as u32;
            lo[g * 16 + n] = (enc & 0xff) as u8;
            hi[g * 16 + n] = (enc >> 8) as u8;
            if n == 15 {
                xs[g] = p as i16;
            }
            n += 1;
        }
        g += 1;
    }
    (lo, hi, xs)
}

/// The build's two planes and group sums are exactly the `planeSum` oracle, for every code
/// combination over the extremes and a long random stream; the pooled build equals the serial one
/// byte for byte on the long stream, so the group split and its disjoint writes are exact.
#[test]
fn build_tables_matches_the_plane_sum_oracle() {
    const VALUES: [i8; 5] = [-127, -1, 0, 1, 127];
    for &a in &VALUES {
        for &b in &VALUES {
            for &c in &VALUES {
                for &d in &VALUES {
                    let codes = [a, b, c, d];
                    let (lo, hi, xs) = plane_sum_oracle(&codes);
                    let t = build_tables(&codes);
                    assert_eq!((t.lo, t.hi, t.xs), (lo, hi, xs), "codes {codes:?} diverged");
                }
            }
        }
    }
    // A stream long enough to split across the pool, so the parallel range writes are compared too.
    let n_in = 2048usize;
    let codes: Vec<i8> = {
        let mut state = 0x5eed_1234_9876u64;
        (0..n_in)
            .map(|_| {
                state = next_rand(&mut state);
                ((state % 255) as i64 - 127) as i8
            })
            .collect()
    };
    let (lo, hi, xs) = plane_sum_oracle(&codes);
    let serial = build_tables(&codes);
    let pooled = build_tables_pooled(&codes);
    assert_eq!(
        (&serial.lo, &serial.hi, &serial.xs),
        (&lo, &hi, &xs),
        "serial diverged"
    );
    assert_eq!(
        (&pooled.lo, &pooled.hi, &pooled.xs),
        (&serial.lo, &serial.hi, &serial.xs),
        "pooled build diverged from serial"
    );
}

/// The proved width choice on an ordinary shape: the block inner fits `i64` (`lut_i64_bound_ok`)
/// and the row split fits two `i64` halves (`lut_row_i64_bound_ok`), so the kernel takes the
/// two-`i64` path and still reproduces `matvec_fixed` bit for bit.
#[test]
fn matvec_lut_takes_the_two_i64_row_path_when_the_bound_holds() {
    let n_in = 1024usize;
    let n_out = 33usize;
    let raw = random_payload(n_out, n_in, 0x7BFF, 0x0dd1);
    let q = quant_act(&random_x(n_in, 0x1bad_c0de)).expect("dequantizes");
    let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
    assert!(lut_i64_bound_ok(q.scales()), "block inner must fit i64");
    assert!(
        lut_row_i64_bound_ok(&lut, q.scales()),
        "this shape must take the two-i64 row path"
    );
    let deq = q.dequant().expect("quant_act guarantees dequant");
    assert_eq!(
        matvec_lut(&lut, &q),
        matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq)
    );
}

/// The boundary the two-`i64` width rests on: for a shape whose row bound
/// (`n_blocks·d_max·S_max·65_024 > i64::MAX·2^32`) fails while the block-inner and `i128` bounds
/// hold, the kernel must keep the checked `i128` row accumulation and still reproduce
/// `matvec_fixed`. Forcing the two-`i64` path on this input truncates `(d·v) >> 32` and this test
/// goes red (the mutation control).
#[test]
fn matvec_lut_row_bound_keeps_i128_on_an_oversized_row() {
    let n_in = 128usize;
    let n_out = 33usize;
    // The largest finite `f16` scale (`0x7BFF` => `65_504·2^32` in Q32.32) is the weight factor
    // that makes the row bound fail; `absmax = 2^47` gives `S_max ≈ 2^40`, so
    // `65_024·2^48·2^40 = 2^104 > i64::MAX·2^32 = 2^95` while `S_max·65_024 = 2^56 ≤ i64::MAX`
    // and the `i128` bound `n_in·65_504·2^32·S_max·508 = 2^104 < i128::MAX` both hold.
    let mut x = vec![0i64; n_in];
    x[0] = 1i64 << 47;
    let q = quant_act(&x).expect("dequantizes");
    let raw = random_payload(n_out, n_in, 0x7BFF, 0x7e57);
    let lut = Pq2Lut::from_pq2_0(&raw, n_out, n_in).expect("well-formed payload");
    assert!(
        !lut_row_i64_bound_ok(&lut, q.scales()),
        "the test shape must leave the two-i64 row bound"
    );
    assert!(
        lut_i64_bound_ok(q.scales()),
        "the block inner must still fit i64"
    );
    assert!(
        lut_bound_ok(n_in, q.scales()),
        "the i128 bound must still hold, so the checked path runs"
    );
    let deq = q.dequant().expect("quant_act guarantees dequant");
    assert_eq!(
        matvec_lut(&lut, &q),
        matvec_fixed(GGML_TYPE_PQ2_0, &raw, n_out, n_in, &deq)
    );
}
