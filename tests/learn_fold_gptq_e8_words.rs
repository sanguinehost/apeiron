//! `gptq_e8` on Q32.32 words: bit-identity against the digests recorded from the f64 wrapper.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::type_complexity,
    reason = "the test setup mirrors the row-major indices and the Q32.32 casts of the ported kernel; \
              the f64 comparisons are exactness claims (`s` decodes to codeword/ONE, the act order \
              round-trips bit for bit, the diagonal Hessian reproduces RTN exactly), and the golden \
              literals are exact captured bit patterns, not tolerances"
)]

use apeiron::learn::fold::fit::gptq_e8::{gptq_e8, E8_SCALE};
use apeiron::learn::solve::q32::{to_grid, GRID_BOUND};

fn word(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("value inside the Q32.32 grid range")
}

/// The words of a real buffer: the boundary conversion the callers of the flipped API apply. `O(len)`.
fn words(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| word(v)).collect()
}

/// FNV-1a 64 of a word buffer, each word as eight little-endian bytes.
fn hash_words(v: &[i64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &w in v {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// FNV-1a 64 of the code indices, each as two little-endian bytes.
fn hash_idx(v: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &w in v {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// A Toeplitz Gram `H[i,j] = 0.4^|i−j|`, SPD with unit diagonal.
fn witness_h(n: usize) -> Vec<f64> {
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = 0.4f64.powi(i32::try_from(i.abs_diff(j)).expect("n < 2^31"));
        }
    }
    h
}

/// The witness: the three `idx`/`s`/`scale` word digests of [`gptq_e8`] on three seeds, recorded
/// through the declared-`f64` API before the Q32.32 boundary flip (`to_grid` of the old `s`/`scale`
/// and the old `idx`), and re-checked here on the word API that replaced it. Any movement is a
/// crossing that rounded.
///
/// The seeds are `(w, h)` pairs on the word grid: `w` from the LCG below, `h` the Toeplitz Gram,
/// both rounded by [`word`]; `o = 4`, `n = 256`, `damp = 0.01`, `act_order = wscale = true`, at
/// `beam = 1` and `beam = 4`. `s` is a function of `idx` (the codeword word), so the three digests
/// together pin the whole returned plane.
#[test]
fn gptq_e8_words_are_bit_identical_to_the_recorded_f64_wrapper() {
    const RECORDED: [((u64, u64, u64), (u64, u64, u64)); 3] = [
        (
            (
                0x227e_621e_f7f1_58c5,
                0xcd1d_cb1a_cf75_adfb,
                0x1a1d_08cc_49c2_f671,
            ),
            (
                0x3fc8_fca7_154e_f672,
                0xaabf_bb6f_df71_6d75,
                0x5efe_8c10_b1ba_4b62,
            ),
        ),
        (
            (
                0x0ea6_fc07_2b73_1662,
                0xef41_b679_b938_e45d,
                0xad7e_976c_07ee_6b29,
            ),
            (
                0x5ce6_72c0_ba7f_baed,
                0x066b_1875_fa93_1928,
                0x4d7f_1850_ae98_4e31,
            ),
        ),
        (
            (
                0xd258_2eb6_f362_f312,
                0x59fc_5d37_fdb4_82f9,
                0x2a3e_5909_dd58_31ec,
            ),
            (
                0x5865_9fb8_fd63_762d,
                0x8476_ea44_14b6_5cf7,
                0xc58a_0d53_32b2_c9c0,
            ),
        ),
    ];
    // `RNE(0.828·2^32)`, the `E8_SCALE` word the recorded run used.
    assert_eq!(E8_SCALE, 3_556_232_921);
    let (o, n) = (4usize, 256usize);
    let h = words(&witness_h(n));
    for (i, seed) in [0x2C2C_0001u64, 0x2C2C_0002, 0x2C2C_0003]
        .into_iter()
        .enumerate()
    {
        let mut state = seed;
        let w: Vec<i64> = (0..o * n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                word(u * 2.0 - 1.0)
            })
            .collect();
        for (k, beam) in [1usize, 4].into_iter().enumerate() {
            let r = gptq_e8(&w, &h, o, n, word(0.01), true, true, beam).expect("gptq_e8");
            let (hs, hc, hi) = (hash_words(&r.s), hash_words(&r.scale), hash_idx(&r.idx));
            let want = if k == 0 { RECORDED[i].0 } else { RECORDED[i].1 };
            assert_eq!(
                (hs, hc, hi),
                want,
                "seed {i} beam {beam}: s {hs:#018x} scale {hc:#018x} idx {hi:#018x}"
            );
        }
    }
}
