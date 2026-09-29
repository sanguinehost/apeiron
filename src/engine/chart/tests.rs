#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    reason = "test-only fixture arithmetic on bounded small values; not extracted"
)]

use super::*;

/// xorshift64: a deterministic code/activation stream for the equality tests.
fn next_rand(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// A `rows × n_in` payload of random bytes with every block's `f16` scale(s) overwritten by
/// `d` (and `dmin`), so each block is well formed while its codes and packed scales are random.
fn random_payload(
    ggml_type: u32,
    rows: usize,
    n_in: usize,
    d: u16,
    dmin: u16,
    seed: u64,
) -> Vec<u8> {
    let (qk, bb) = quant::block_geom(ggml_type).expect("supported geometry");
    let blocks_per_row = n_in.div_ceil(qk);
    let row_bytes = blocks_per_row * bb;
    let total_blocks = rows * blocks_per_row;
    let mut raw = vec![0u8; rows * row_bytes];
    let mut state = seed;
    let mut i = 0usize;
    while i < raw.len() {
        raw[i] = (next_rand(&mut state) & 0xff) as u8;
        i += 1;
    }
    let mut block = 0usize;
    while block < total_blocks {
        let o = block * bb;
        match ggml_type {
            quant::GGML_TYPE_Q8_0 | quant::GGML_TYPE_Q5_0 => {
                raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
            }
            quant::GGML_TYPE_Q2_K => {
                raw[o + 80..o + 82].copy_from_slice(&d.to_le_bytes());
                raw[o + 82..o + 84].copy_from_slice(&dmin.to_le_bytes());
            }
            quant::GGML_TYPE_Q3_K => {
                raw[o + 108..o + 110].copy_from_slice(&d.to_le_bytes());
            }
            quant::GGML_TYPE_Q4_K => {
                raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
                raw[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
            }
            quant::GGML_TYPE_Q6_K => {
                raw[o + 208..o + 210].copy_from_slice(&d.to_le_bytes());
            }
            _ => {}
        }
        block += 1;
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

/// The bit-manipulation `f16` decode equals the reference on all `65536` patterns; every
/// `Inf`/`NaN` exponent takes the fallback (`None` against the reference's refusal).
#[test]
fn fast_f16_decode_matches_reference_for_all_patterns() {
    let mut bits = 0u32;
    while bits <= 0xffff {
        let h = bits as u16;
        match f16_to_fixed_fast(h) {
            Some(v) => assert_eq!(
                crate::engine::f16_to_fixed(h),
                Ok(v),
                "finite pattern {h:#06x} diverged"
            ),
            None => assert!(
                crate::engine::f16_to_fixed(h).is_err(),
                "pattern {h:#06x} must refuse in the reference"
            ),
        }
        bits += 1;
    }
}

/// The AVX2 fast-row cores return the same exact `i32` as their portable twins on random
/// `Q8_0` code bytes and random `Q5_0` `qh`/`qs` bytes.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_fast_row_cores_match_the_portable_twins() {
    if !x86::avx2() {
        return;
    }
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut round = 0usize;
    while round < 512 {
        let codes: Vec<u8> = (0..32)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        let xq: Vec<i8> = (0..32)
            .map(|_| ((next_rand(&mut state) % 255) as i64 - 127) as i8)
            .collect();
        // SAFETY: AVX2 was detected and both slices hold 32 lanes.
        let avx8 = unsafe { x86::dot32_u8_i8(&codes, &xq) };
        assert_eq!(
            avx8,
            portable_u8_i8_dot(&codes, &xq),
            "q8_0 fast core diverged"
        );

        let qh: Vec<u8> = (0..4)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        let qs: Vec<u8> = (0..16)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        // SAFETY: AVX2 was detected; `qh` holds 4 bytes, `qs` 16 bytes and `xq` 32 lanes.
        let avx5 = unsafe { x86::q5_0_dot32(&qh, &qs, &xq) };
        assert_eq!(
            avx5,
            portable_q5_0_dot(&qh, &qs, &xq),
            "q5_0 fast core diverged"
        );
        round += 1;
    }
}

/// The four whole-row fast bodies agree bit for bit between the portable lane loop and the
/// AVX2 body, on random payloads and activations for each family's geometry — including a
/// `Q4_K` payload whose pair count is not a whole 256-weight block, the case where the pair
/// loop reuses the block's cached `d`, `dmin` and packed scales.
#[cfg(target_arch = "x86_64")]
#[test]
fn fast_row_bodies_match_their_portable_twins() {
    const CASES: [(u32, usize, usize); 8] = [
        (quant::GGML_TYPE_Q8_0, 3, 96),
        (quant::GGML_TYPE_Q5_0, 3, 96),
        (quant::GGML_TYPE_Q8_0, 2, 256),
        (quant::GGML_TYPE_Q5_0, 2, 256),
        (quant::GGML_TYPE_Q4_K, 2, 512),
        (quant::GGML_TYPE_Q4_K, 1, 192),
        (quant::GGML_TYPE_Q6_K, 2, 512),
        (quant::GGML_TYPE_Q5_0, 1, 32),
    ];
    if !x86::avx2() {
        return;
    }
    let mut seed = 0x0bad_c0de_1234_5678u64;
    let mut ci = 0usize;
    while ci < CASES.len() {
        let (ggml_type, rows, n_in) = CASES[ci];
        let (qk, bb) = quant::block_geom(ggml_type).expect("supported geometry");
        let row_bytes = n_in.div_ceil(qk) * bb;
        let raw = random_payload(ggml_type, rows, n_in, 0x3c00, 0x3c00, seed);
        let q = quant_act(&random_x(n_in, seed ^ 0x1357_9bdf)).expect("dequantizable");
        let xsums = block_xsums(&q.codes);
        let mut r = 0usize;
        while r < rows {
            let row = &raw[r * row_bytes..(r + 1) * row_bytes];
            // SAFETY: AVX2 was detected above; the row extent and the code/scale lengths are
            // the caller-side checks each fast body documents.
            let (portable, avx2) = unsafe {
                match ggml_type {
                    quant::GGML_TYPE_Q8_0 => {
                        let b = n_in / ACT_QK;
                        (
                            q8_0_row_portable(row, b, &q.codes, &q.scales),
                            q8_0_row_avx2(row, b, &q.codes, &q.scales),
                        )
                    }
                    quant::GGML_TYPE_Q5_0 => {
                        let b = n_in / ACT_QK;
                        (
                            q5_0_row_portable(row, b, &q.codes, &q.scales),
                            q5_0_row_avx2(row, b, &q.codes, &q.scales),
                        )
                    }
                    quant::GGML_TYPE_Q4_K => {
                        let pairs = n_in / 64;
                        (
                            q4_k_row_portable(row, pairs, &q.codes, &q.scales, &xsums),
                            q4_k_row_avx2(row, pairs, &q.codes, &q.scales, &xsums),
                        )
                    }
                    _ => {
                        let b = n_in / QK_K;
                        (
                            q6_k_row_portable(row, b, &q.codes, &q.scales),
                            q6_k_row_avx2(row, b, &q.codes, &q.scales),
                        )
                    }
                }
            };
            assert_eq!(portable, avx2, "type {ggml_type} row {r} diverged");
            r += 1;
        }
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ci += 1;
    }
}

/// Every supported family, several shapes (including partial last blocks, a zero width and a
/// whole block), and random codes/packed scales agree exactly with `matvec_fixed` on the
/// dequantized activation.
#[test]
fn chart_matches_fixed_on_random_payloads_for_every_family() {
    const TYPES: [u32; 6] = [
        quant::GGML_TYPE_Q8_0,
        quant::GGML_TYPE_Q5_0,
        quant::GGML_TYPE_Q2_K,
        quant::GGML_TYPE_Q3_K,
        quant::GGML_TYPE_Q4_K,
        quant::GGML_TYPE_Q6_K,
    ];
    const SHAPES: [(usize, usize); 9] = [
        (4, 32),
        (3, 48),
        (2, 33),
        (2, 300),
        (1, 64),
        (5, 0),
        (1, 256),
        (2, 512),
        (2, 256),
    ];
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut t = 0usize;
    while t < TYPES.len() {
        let ggml_type = TYPES[t];
        let mut s = 0usize;
        while s < SHAPES.len() {
            let (rows, n_in) = SHAPES[s];
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let raw = random_payload(ggml_type, rows, n_in, 0x3800, 0x3800, seed);
            let x = random_x(n_in, seed ^ 0xdead_beef);
            let quant = quant_act(&x).expect("small activations dequantize");
            let deq = quant.dequant().expect("quant_act guarantees dequant");
            assert_eq!(
                matvec_q8act(ggml_type, &raw, rows, n_in, &quant),
                matvec_fixed(ggml_type, &raw, rows, n_in, &deq),
                "type {ggml_type} shape ({rows},{n_in}) diverged"
            );
            s += 1;
        }
        t += 1;
    }
}

/// Extremes: activations whose codes are `±127` everywhere against maximal codes and the
/// largest finite `f16` scales, which forces the fallback as well as the fast path. Equality
/// holds in both.
#[test]
fn chart_matches_fixed_on_extreme_codes_and_scales() {
    const TYPES: [u32; 6] = [
        quant::GGML_TYPE_Q8_0,
        quant::GGML_TYPE_Q5_0,
        quant::GGML_TYPE_Q2_K,
        quant::GGML_TYPE_Q3_K,
        quant::GGML_TYPE_Q4_K,
        quant::GGML_TYPE_Q6_K,
    ];
    for (bi, &ggml_type) in TYPES.iter().enumerate() {
        let n_in = 256usize;
        let raw = random_payload(ggml_type, 2, n_in, 0x7BFF, 0x7BFF, 0x51ff_0000 + bi as u64);
        // codes ±127 with unit scale: x̂ = ±127 exactly, no fallback from the magnitude bound.
        let codes: Vec<i8> = (0..n_in)
            .map(|j| if j % 2 == 0 { 127 } else { -127 })
            .collect();
        let scales = vec![1i64; n_in.div_ceil(ACT_QK)];
        let quant = QuantAct { scales, codes };
        let deq = quant.dequant().expect("unit scale cannot overflow");
        assert_eq!(
            matvec_q8act(ggml_type, &raw, 2, n_in, &quant),
            matvec_fixed(ggml_type, &raw, 2, n_in, &deq),
            "extreme type {ggml_type} diverged"
        );
        // and the same activation as `quant_act` produces it from a ±127-scaled vector.
        let dequantized: Vec<i64> = deq.clone();
        let from_quant = quant_act(&dequantized).expect("±127 dequantizes");
        assert_eq!(
            from_quant.codes, quant.codes,
            "quant_act of a dequantized extreme changed the codes"
        );
    }
}

/// A ragged width is not a multiple of 32: the last activation block and the last weight
/// group are both partial, and the result still matches bit for bit.
#[test]
fn chart_matches_fixed_on_ragged_widths() {
    const TYPES: [u32; 6] = [
        quant::GGML_TYPE_Q8_0,
        quant::GGML_TYPE_Q5_0,
        quant::GGML_TYPE_Q2_K,
        quant::GGML_TYPE_Q3_K,
        quant::GGML_TYPE_Q4_K,
        quant::GGML_TYPE_Q6_K,
    ];
    const WIDTHS: [usize; 6] = [33, 47, 65, 257, 300, 271];
    for (bi, &ggml_type) in TYPES.iter().enumerate() {
        for &n_in in &WIDTHS {
            let raw = random_payload(ggml_type, 2, n_in, 0x3800, 0x3800, 0x7000 + bi as u64);
            let x = random_x(n_in, 0xabc0 ^ n_in as u64);
            let quant = quant_act(&x).expect("small activations dequantize");
            let deq = quant.dequant().expect("quant_act guarantees dequant");
            assert_eq!(
                matvec_q8act(ggml_type, &raw, 2, n_in, &quant),
                matvec_fixed(ggml_type, &raw, 2, n_in, &deq),
                "type {ggml_type} ragged width {n_in} diverged"
            );
        }
    }
}

/// An all-zero activation block quantizes to zero codes and a zero scale, and a mixture of zero
/// and nonzero blocks still matches.
#[test]
fn chart_matches_fixed_on_zero_activation_blocks() {
    let mut x = vec![0i64; 96];
    let mut j = 64usize;
    while j < 96 {
        x[j] = 1000 - 100 * (j as i64);
        j += 1;
    }
    let quant = quant_act(&x).expect("zero blocks dequantize");
    assert!(
        quant.codes[..64].iter().all(|&c| c == 0),
        "a zero block must quantize to zero codes"
    );
    assert_eq!(quant.scales[0], 0, "a zero block must have scale 0");
    assert_eq!(quant.scales[1], 0, "a zero block must have scale 0");
    let deq = quant.dequant().expect("quant_act guarantees dequant");
    for (bi, &ggml_type) in [
        quant::GGML_TYPE_Q8_0,
        quant::GGML_TYPE_Q4_K,
        quant::GGML_TYPE_Q6_K,
    ]
    .iter()
    .enumerate()
    {
        let raw = random_payload(ggml_type, 1, 96, 0x3800, 0x3800, 0x9000 + bi as u64);
        assert_eq!(
            matvec_q8act(ggml_type, &raw, 1, 96, &quant),
            matvec_fixed(ggml_type, &raw, 1, 96, &deq),
            "zero-block type {ggml_type} diverged"
        );
    }
}

/// `quant_act`'s stated properties: `|xq| ≤ 127`, the `S/2` rounding bound, zero in for zero
/// out, ties away from zero, and the exact `Some`/`None` contract with `dequant`.
#[test]
fn quant_act_bounds_and_dequant_contract() {
    for seed in 0..16u64 {
        let x: Vec<i64> = random_x(200, 0x5eed_1000 + seed)
            .into_iter()
            .map(|v| v * 1_000_003)
            .collect();
        let quant = quant_act(&x).expect("scaled small activations dequantize");
        assert_eq!(quant.codes.len(), x.len());
        assert_eq!(quant.scales.len(), x.len().div_ceil(ACT_QK));
        let deq = quant.dequant().expect("quant_act guarantees dequant");
        for (j, &xv) in x.iter().enumerate() {
            let s = quant.scales[j / ACT_QK];
            let code = i64::from(quant.codes[j]);
            assert!(code.abs() <= 127, "code {code} out of range");
            if s == 0 {
                assert_eq!(code, 0, "zero scale must give zero code");
            }
            // 2·|x − S·xq| ≤ S  ⟺  |x − x̂| ≤ S/2
            let err = (i128::from(xv) - i128::from(deq[j])).abs();
            assert!(2 * err <= i128::from(s), "rounding bound broken at {j}");
        }
    }

    // Ties away from zero: |x|/S = 1.5 exactly must round to 2.
    let tied = quant_act(&[3, 130, -3]).expect("tied values dequantize");
    assert_eq!(tied.codes[0], 2, "positive tie must round away from zero");
    assert_eq!(tied.codes[2], -2, "negative tie must round away from zero");

    // Zero in, zero out.
    let zero = quant_act(&[0i64; 64]).expect("zero quantizes");
    assert!(zero.codes.iter().all(|&c| c == 0));
    assert!(zero.scales.iter().all(|&s| s == 0));
    assert_eq!(zero.dequant(), Some(vec![0i64; 64]));

    // The `None` contract under power-of-two scales: `i64::MAX` needs `S = 2^57` and rounds to the
    // code 64, whose product `2^63` leaves `i64`; `i64::MIN = −64·2^57` lies on that grid and
    // dequantizes exactly, as does `127·2^56`.
    assert!(quant_act(&[i64::MAX; 32]).is_none());
    let min = quant_act(&[i64::MIN; 32]).expect("i64::MIN lies on the 2^57 grid");
    assert_eq!(min.scales, vec![1i64 << 57]);
    assert_eq!(min.dequant(), Some(vec![i64::MIN; 32]));
    let huge = quant_act(&[127i64 << 56; 32]).expect("127·2^56 dequantizes exactly");
    assert_eq!(huge.dequant(), Some(vec![127i64 << 56; 32]));
}

/// The integer identity behind the power-of-two scale: for `s = 2^e` the shift/mask quotient and
/// remainder equal `m / s` and `m % s` at the boundary magnitudes a `u64` activation can reach.
/// A zero and a non-power-of-two scale keep the checked-division fallback.
#[test]
fn act_quot_rem_shift_equals_division_for_power_of_two_scales() {
    let ms: [u64; 6] = [0, 1, 1 << 31, 1 << 32, 1 << 62, i64::MIN.unsigned_abs()];
    let ss: [u64; 6] = [0, 1, 2, 1 << 31, 1 << 57, 1 << 62];
    for &m in &ms {
        for &s in &ss {
            let pow2 = s.is_power_of_two();
            let shift = s.trailing_zeros();
            let mask = s.wrapping_sub(1);
            let (quotient, remainder) = act_quot_rem(m, s, pow2, shift, mask);
            if s == 0 {
                // `pow2` is false; the checked division's zero-divisor failure is `(0, 0)`.
                assert!(!pow2);
                assert_eq!((quotient, remainder), (0, 0), "m {m} s 0");
            } else {
                assert!(pow2, "fixture scale {s} must be a power of two");
                assert_eq!((quotient, remainder), (m / s, m % s), "m {m} s {s}");
            }
        }
    }
    // A non-power-of-two scale takes the same checked-division fallback.
    assert_eq!(act_quot_rem(7, 3, false, 0, 2), (2, 1));
    assert_eq!(
        act_quot_rem(i64::MIN.unsigned_abs(), 3, false, 0, 2),
        (i64::MIN.unsigned_abs() / 3, i64::MIN.unsigned_abs() % 3)
    );
}

/// The previous per-element `dequantizes` body, as the replica the block fit check must match: it
/// reads `scales[idx / ACT_QK]` per element and returns `false` at the first missing scale or
/// overflowing `S_b·xq_j`.
fn dequantizes_per_element(scales: &[i64], codes: &[i8]) -> bool {
    let mut idx = 0usize;
    while idx < codes.len() {
        let Some(&scale) = scales.get(idx / ACT_QK) else {
            return false;
        };
        if scale.checked_mul(i64::from(codes[idx])).is_none() {
            return false;
        }
        idx += 1;
    }
    true
}

/// The block-constant fit check agrees with the per-element loop on scales around the `i64`
/// boundary. `S·127` fits for `0, 1, 2, 2^31, i64::MAX/127, i64::MAX/127 + 1`, so those blocks skip
/// their elements; `S·127` leaves `i64` for `2^62` and for `i64::MAX/126`, whose `S·126` still fits
/// and which therefore separates the block bound from any smaller one; those blocks fall back to
/// the per-element check.
#[test]
fn dequantizes_block_fit_matches_the_per_element_loop() {
    let scales: [i64; 8] = [
        0,
        1,
        2,
        1 << 31,
        1 << 62,
        i64::MAX / 127,
        i64::MAX / 127 + 1,
        i64::MAX / 126,
    ];
    let code_values: [i8; 5] = [0, 1, -1, 127, -127];
    // One block per scale, each block's codes cycling through the boundary code set.
    let mut codes: Vec<i8> = Vec::with_capacity(scales.len() * ACT_QK);
    for _ in &scales {
        for lane in 0..ACT_QK {
            codes.push(code_values[lane % code_values.len()]);
        }
    }
    let combined = QuantAct::from_parts(scales.to_vec(), codes.clone());
    assert_eq!(
        combined.dequantizes(),
        dequantizes_per_element(&scales, &codes)
    );

    // Each scale on its own, so a refusing block cannot hide the blocks after it.
    for &scale in &scales {
        let one: Vec<i8> = (0..ACT_QK)
            .map(|lane| code_values[lane % code_values.len()])
            .collect();
        let qa = QuantAct::from_parts(vec![scale], one.clone());
        assert_eq!(
            qa.dequantizes(),
            dequantizes_per_element(&[scale], &one),
            "scale {scale}"
        );
    }

    // A block the fit check clears accepts every code, including the `±127` extremes.
    let clears = QuantAct::from_parts(vec![i64::MAX / 127], vec![127i8; ACT_QK]);
    assert!(clears.dequantizes());
    // `2^62·127` leaves `i64`: the fallback accepts code `1` and rejects the `±127` extremes.
    let falls_back = QuantAct::from_parts(vec![1i64 << 62], vec![1i8; ACT_QK]);
    assert!(falls_back.dequantizes());
    let falls_back_top = QuantAct::from_parts(vec![1i64 << 62], vec![127i8; ACT_QK]);
    assert!(!falls_back_top.dequantizes());

    // A short `scales` vector refuses, as the per-element `scales.get` did.
    let short = QuantAct::from_parts(vec![1], vec![0i8; ACT_QK + 1]);
    assert!(!short.dequantizes());
    assert!(!dequantizes_per_element(&[1], &[0i8; ACT_QK + 1]));
}

/// Every refusal class agrees exactly with the reference, including the refusal value.
#[test]
fn chart_agrees_with_fixed_on_every_refusal_class() {
    let q8 = quant::GGML_TYPE_Q8_0;
    let raw = random_payload(q8, 2, 32, 0x3800, 0x3800, 1);
    let quant = quant_act(&random_x(32, 7)).expect("small activations dequantize");
    let deq = quant.dequant().expect("quant_act guarantees dequant");

    // Shape mismatch: the dequantized activation length differs from `n_in`.
    assert_eq!(
        matvec_q8act(q8, &raw, 2, 31, &quant),
        matvec_fixed(q8, &raw, 2, 31, &deq)
    );

    // Short payload.
    let short = &raw[..raw.len() - 1];
    assert_eq!(
        matvec_q8act(q8, short, 2, 32, &quant),
        matvec_fixed(q8, short, 2, 32, &deq)
    );

    // Unsupported family.
    assert_eq!(
        matvec_q8act(quant::GGML_TYPE_F32, &[], 1, 1, &quant),
        matvec_fixed(quant::GGML_TYPE_F32, &[], 1, 1, &deq)
    );

    // Row-stride / payload length overflow.
    assert_eq!(
        matvec_q8act(q8, &raw, usize::MAX, 32, &quant),
        matvec_fixed(q8, &raw, usize::MAX, 32, &deq)
    );

    // Non-finite weight scale falls back to the reference's `Dequant(NonFiniteScale)`.
    let mut inf = random_payload(q8, 2, 32, 0x3800, 0x3800, 2);
    inf[0..2].copy_from_slice(&0x7C00u16.to_le_bytes());
    let non_finite = matvec_q8act(q8, &inf, 2, 32, &quant);
    assert!(matches!(
        non_finite,
        Err(MatvecError::Dequant(EngineError::NonFiniteScale {
            bits: 0x7C00
        }))
    ));
    assert_eq!(non_finite, matvec_fixed(q8, &inf, 2, 32, &deq));

    // Output overflow: weight 1.0 against `i64::MAX` rounds past `i64`.
    let mut one = random_payload(q8, 1, 32, 0x3C00, 0x3800, 3);
    let mut l = 0usize;
    while l < 32 {
        one[2 + l] = 1;
        l += 1;
    }
    let top = 127i64 << 56;
    let max_quant = quant_act(&vec![top; 32]).expect("127·2^56 dequantizes");
    assert_eq!(max_quant.dequant(), Some(vec![top; 32]));
    let output = matvec_q8act(q8, &one, 1, 32, &max_quant);
    assert_eq!(output, Err(MatvecError::OutputOverflow));
    assert_eq!(output, matvec_fixed(q8, &one, 1, 32, &vec![top; 32]));

    // A `QuantAct` whose dequantized activation cannot exist refuses rather than guesses.
    let impossible = QuantAct {
        scales: vec![i64::MAX],
        codes: vec![127],
    };
    assert_eq!(
        matvec_q8act(q8, &raw, 1, 1, &impossible),
        Err(MatvecError::Dequant(EngineError::FixedOverflow))
    );
}

/// The AVX2 cores return the same exact `i32` as the portable lane loop.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_group_cores_match_the_portable_loop() {
    if !x86::avx2() {
        return;
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut round = 0usize;
    while round < 128 {
        let codes32: Vec<i8> = (0..32).map(|_| next_rand(&mut state) as i8).collect();
        let xq32: Vec<i8> = (0..32)
            .map(|_| ((next_rand(&mut state) % 255) as i64 - 127) as i8)
            .collect();
        // SAFETY: AVX2 was detected and both slices hold 32 lanes.
        let avx = unsafe { x86::dot32_i8(&codes32, &xq32) };
        assert_eq!(avx, portable_i8_dot(&codes32, &xq32), "dot32_i8 diverged");

        let codes16 = &codes32[..16];
        let xq16 = &xq32[..16];
        // SAFETY: AVX2 was detected and both slices hold 16 lanes.
        let avx16 = unsafe { x86::dot16_i8(codes16, xq16) };
        assert_eq!(avx16, portable_i8_dot(codes16, xq16), "dot16_i8 diverged");

        let qs: Vec<u8> = (0..32)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        let lo_codes: Vec<i8> = qs.iter().map(|b| (b & 0x0F) as i8).collect();
        let hi_codes: Vec<i8> = qs.iter().map(|b| (b >> 4) as i8).collect();
        let xq_lo: Vec<i8> = (0..32)
            .map(|_| ((next_rand(&mut state) % 255) as i64 - 127) as i8)
            .collect();
        let xq_hi: Vec<i8> = (0..32)
            .map(|_| ((next_rand(&mut state) % 255) as i64 - 127) as i8)
            .collect();
        // SAFETY: AVX2 was detected, `qs` holds 32 bytes and each activation window 32 lanes.
        let (dot_lo, dot_hi) = unsafe { x86::q4_k_window_dot(&qs, &xq_lo, &xq_hi) };
        assert_eq!(
            dot_lo,
            portable_i8_dot(&lo_codes, &xq_lo),
            "q4 low plane diverged"
        );
        assert_eq!(
            dot_hi,
            portable_i8_dot(&hi_codes, &xq_hi),
            "q4 high plane diverged"
        );
        round += 1;
    }
}

/// The VNNI-accelerated fast row cores match the portable twins and AVX2 fallback.
#[cfg(target_arch = "x86_64")]
#[test]
fn vnni_and_avx2_cores_match_the_portable_twins() {
    if !x86::avx2() {
        return;
    }
    let mut state = 0x517c_c1b7_2722_0a95u64;
    let mut round = 0usize;
    while round < 512 {
        let codes: Vec<u8> = (0..32)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        let xq: Vec<i8> = (0..32)
            .map(|_| ((next_rand(&mut state) % 255) as i64 - 127) as i8)
            .collect();
        // SAFETY: AVX2 was detected and both slices hold 32 lanes.
        let core_dot = unsafe { x86::dot32_u8_i8(&codes, &xq) };
        assert_eq!(
            core_dot,
            portable_u8_i8_dot(&codes, &xq),
            "dot32 fast core diverged from portable twin"
        );

        let qh: Vec<u8> = (0..4)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        let qs: Vec<u8> = (0..16)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        // SAFETY: AVX2 was detected; `qh` holds 4 bytes, `qs` 16 bytes and `xq` 32 lanes.
        let core_q5 = unsafe { x86::q5_0_dot32(&qh, &qs, &xq) };
        assert_eq!(
            core_q5,
            portable_q5_0_dot(&qh, &qs, &xq),
            "q5_0 fast core diverged from portable twin"
        );

        let ql: Vec<u8> = (0..64)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        let qh_k: Vec<u8> = (0..32)
            .map(|_| (next_rand(&mut state) & 0xff) as u8)
            .collect();
        let xq_k: Vec<i8> = (0..128)
            .map(|_| ((next_rand(&mut state) % 255) as i64 - 127) as i8)
            .collect();
        // SAFETY: AVX2 was detected; `ql` holds 64 bytes, `qh_k` 32 bytes and `xq_k` 128 lanes.
        let half_dots = unsafe { x86::q6_k_half_dots(ql.as_ptr(), qh_k.as_ptr(), xq_k.as_ptr()) };
        let mut blk = vec![0u8; quant::Q6_K_BLOCK_BYTES];
        blk[..64].copy_from_slice(&ql);
        blk[128..160].copy_from_slice(&qh_k);
        let codes256 = quant::q6_k_codes(&blk).expect("q6_k_codes");
        let port_dot0 = portable_i8_dot(&codes256[0..16], &xq_k[0..16]);
        let port_dot1 = portable_i8_dot(&codes256[16..32], &xq_k[16..32]);
        assert_eq!(half_dots[0], port_dot0, "q6_k group 0 low diverged");
        assert_eq!(half_dots[1], port_dot1, "q6_k group 0 high diverged");

        round += 1;
    }
}

/// Throwaway microbench profiling fast row kernels single-threaded over Qwen-shaped matrices
/// with DRAM buffers > 64 MB (exceeding L3 cache).
#[test]
#[ignore = "microbenchmark for Step 1 profiling"]
#[allow(
    clippy::too_many_lines,
    reason = "throwaway benchmark measuring three quantized formats"
)]
#[allow(
    clippy::uninlined_format_args,
    reason = "explicit format arguments matching surrounding benchmark tests"
)]
#[allow(
    clippy::cast_precision_loss,
    reason = "byte counts fit within f64 mantissa exactly"
)]
fn bench_qwen_row_kernels() {
    use std::time::Instant;

    println!("\n=== Single-Thread Fast Row Kernels Profile (DRAM > 64 MB) ===");

    // 1. Q5_0: 4864 x 896 -> 23 reps = 111,872 rows = 68.91 MB
    {
        let ggml_type = quant::GGML_TYPE_Q5_0;
        let n_in = 896;
        let rows = 4864 * 23; // 111,872 rows
        let (_, bb) = quant::block_geom(ggml_type).unwrap();
        let row_bytes = (n_in / 32) * bb; // 28 * 22 = 616 bytes
        let raw = random_payload(ggml_type, rows, n_in, 0x3800, 0x3800, 12345);
        let x = random_x(n_in, 54321);
        let quant = quant_act(&x).expect("quant_act");
        let blocks = n_in / 32;

        let _ = q8act_q5_0_row_fast(&raw[0..row_bytes], blocks, &quant, true);

        let start = Instant::now();
        let mut sum: i128 = 0;
        let mut r = 0;
        while r < rows {
            let row = &raw[r * row_bytes..(r + 1) * row_bytes];
            sum ^= q8act_q5_0_row_fast(row, blocks, &quant, true).unwrap();
            r += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let bytes = rows * row_bytes;
        let gb = bytes as f64 / 1e9;
        let gbs = gb / elapsed;
        let mb = bytes as f64 / 1e6;
        println!(
            "Q5_0 (4864x896 x23 = {} rows, {:.2} MB): elapsed={:.4} s, GB/s={:.2}, checksum={}",
            rows, mb, elapsed, gbs, sum
        );
    }

    // 2. Q6_K: 896 x 4864 -> 19 reps = 17,024 rows = 67.93 MB
    {
        let ggml_type = quant::GGML_TYPE_Q6_K;
        let n_in = 4864;
        let rows = 896 * 19; // 17,024 rows
        let (_, bb) = quant::block_geom(ggml_type).unwrap();
        let row_bytes = (n_in / 256) * bb; // 19 * 210 = 3990 bytes
        let raw = random_payload(ggml_type, rows, n_in, 0x3800, 0x3800, 23456);
        let x = random_x(n_in, 65432);
        let quant = quant_act(&x).expect("quant_act");
        let blocks = n_in / 256;

        let _ = q8act_q6_k_row_fast(&raw[0..row_bytes], blocks, &quant, true);

        let start = Instant::now();
        let mut sum: i128 = 0;
        let mut r = 0;
        while r < rows {
            let row = &raw[r * row_bytes..(r + 1) * row_bytes];
            sum ^= q8act_q6_k_row_fast(row, blocks, &quant, true).unwrap();
            r += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let bytes = rows * row_bytes;
        let gb = bytes as f64 / 1e9;
        let gbs = gb / elapsed;
        let mb = bytes as f64 / 1e6;
        println!(
            "Q6_K (896x4864 x19 = {} rows, {:.2} MB): elapsed={:.4} s, GB/s={:.2}, checksum={}",
            rows, mb, elapsed, gbs, sum
        );
    }

    // 3. Q8_0: 151936 x 896 = 144.64 MB
    {
        let ggml_type = quant::GGML_TYPE_Q8_0;
        let n_in = 896;
        let rows = 151_936;
        let (_, bb) = quant::block_geom(ggml_type).unwrap();
        let row_bytes = (n_in / 32) * bb; // 28 * 34 = 952 bytes
        let raw = random_payload(ggml_type, rows, n_in, 0x3800, 0x3800, 34567);
        let x = random_x(n_in, 76543);
        let quant = quant_act(&x).expect("quant_act");
        let blocks = n_in / 32;

        let _ = q8act_q8_0_row_fast(&raw[0..row_bytes], blocks, &quant, true);

        let start = Instant::now();
        let mut sum: i128 = 0;
        let mut r = 0;
        while r < rows {
            let row = &raw[r * row_bytes..(r + 1) * row_bytes];
            sum ^= q8act_q8_0_row_fast(row, blocks, &quant, true).unwrap();
            r += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let bytes = rows * row_bytes;
        let gb = bytes as f64 / 1e9;
        let gbs = gb / elapsed;
        let mb = bytes as f64 / 1e6;
        println!(
            "Q8_0 (151936x896 = {} rows, {:.2} MB): elapsed={:.4} s, GB/s={:.2}, checksum={}",
            rows, mb, elapsed, gbs, sum
        );
    }
}

/// Pooled `Q8Act` matvec bandwidth per real projection shape (Qwen3.5-0.8B `Q8_0`, Qwen2.5-0.5B
/// `Q4_K`), each payload well past the last-level cache. Indicative on a shared host: run with
/// `cargo test --release --lib engine::chart::tests::bench_pooled_shapes -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark"]
#[allow(clippy::cast_precision_loss, reason = "byte counts fit f64 exactly")]
fn bench_pooled_shapes() {
    use std::time::Instant;
    let shapes: [(&str, u32, usize, usize); 7] = [
        (
            "q35 attn_qkv 6144x1024 Q8_0",
            quant::GGML_TYPE_Q8_0,
            6144,
            1024,
        ),
        (
            "q35 ffn_gate 3584x1024 Q8_0",
            quant::GGML_TYPE_Q8_0,
            3584,
            1024,
        ),
        (
            "q35 ffn_down 1024x3584 Q8_0",
            quant::GGML_TYPE_Q8_0,
            1024,
            3584,
        ),
        (
            "q35 ssm_out  1024x2048 Q8_0",
            quant::GGML_TYPE_Q8_0,
            1024,
            2048,
        ),
        (
            "q35 head 248320x1024 Q8_0",
            quant::GGML_TYPE_Q8_0,
            248_320,
            1024,
        ),
        (
            "q25 ffn_gate 4864x896 Q4_K",
            quant::GGML_TYPE_Q4_K,
            4864,
            896 / 256 * 256,
        ),
        (
            "q25 ffn_down 896x4864 Q4_K",
            quant::GGML_TYPE_Q4_K,
            896,
            4864 / 256 * 256,
        ),
    ];
    for (name, t, n_out, n_in) in shapes {
        let (qk, bb) = quant::block_geom(t).expect("geometry");
        let row_bytes = n_in.div_ceil(qk) * bb;
        // Enough copies of the matrix to exceed 256 MB, cycled so every call streams from DRAM.
        let copies = (256_000_000 / (n_out * row_bytes)).max(1);
        let mats: Vec<Vec<u8>> = (0..copies)
            .map(|c| random_payload(t, n_out, n_in, 0x2000, 0x2000, 7 + c as u64))
            .collect();
        let x = random_x(n_in, 99);
        let chunks = crate::engine::par::row_chunk_budget(n_out);
        let _ = super::ffn::matvec_q8act_chunked(t, &mats[0], n_out, n_in, &x, chunks);
        let reps = 3 * copies;
        let start = Instant::now();
        let mut i = 0usize;
        while i < reps {
            let out =
                super::ffn::matvec_q8act_chunked(t, &mats[i % copies], n_out, n_in, &x, chunks)
                    .expect("matvec");
            std::hint::black_box(out);
            i += 1;
        }
        let secs = start.elapsed().as_secs_f64();
        let bytes = (reps * n_out * row_bytes) as f64;
        println!(
            "{name}: {:.1} µs/call, {:.1} GB/s (chunks {chunks}, {:.1} MB/matrix)",
            secs * 1e6 / reps as f64,
            bytes / secs / 1e9,
            (n_out * row_bytes) as f64 / 1e6
        );
    }
}
