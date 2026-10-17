//! `tests/learn_fold_gptq_pool.rs` — the pooled row split of `gptq_e8`'s `beam_pass` and the pooled
//! column split of `chol_solve`'s `solve_rhs` reproduce the serial order's exact output bits.
//! Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::too_many_lines,
    reason = "the fixtures mirror the row-major indices and Q32.32 casts of the ported kernel, and \
              the golden literals are exact captured digests"
)]

use apeiron::learn::fold::fit::gptq_e8::{gptq_e8, BLOCK};
use apeiron::learn::solve::q32::{to_grid, GRID_BOUND};

/// The Q32.32 word of `x`: `round(x·2³²)`. `O(1)`.
fn word(x: f64) -> i64 {
    to_grid(x, GRID_BOUND).expect("value inside the Q32.32 grid range")
}

/// The words of a real buffer. `O(len)`.
fn words(x: &[f64]) -> Vec<i64> {
    x.iter().map(|&v| word(v)).collect()
}

/// A deterministic 64-bit linear congruential generator (MMIX constants), uniform in `[−1, 1)` from
/// the high bits.
fn lcg_uniform(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// FNV-1a 64 over the exact bit patterns of `gptq_e8`'s `s` and `scale` words (little-endian `i64`)
/// then its `idx` (little-endian `u16`); no float comparison.
fn fnv_words(s: &[i64], scale: &[i64], idx: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let byte = |b: u8, h: &mut u64| {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for v in s.iter().chain(scale) {
        for b in v.to_le_bytes() {
            byte(b, &mut h);
        }
    }
    for v in idx {
        for b in v.to_le_bytes() {
            byte(b, &mut h);
        }
    }
    h
}

/// FNV-1a 64 over a `chol_solve` solution (little-endian `i64`) and its `residual_max`.
fn fnv_solution(x: &[i64], residual_max: i64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let byte = |b: u8, h: &mut u64| {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for v in x.iter().chain(core::iter::once(&residual_max)) {
        for b in v.to_le_bytes() {
            byte(b, &mut h);
        }
    }
    h
}

/// `H = (1/t)·XᵀX + 0.25·I` from an LCG `X` (`t = n`) with a per-8-block column gain, the
/// [`correlated_h`] fixture shape on the LCG stream.
fn lcg_correlated_h(n: usize, seed: u64) -> Vec<i64> {
    let mut st = seed;
    let gain: Vec<f64> = (0..n).map(|j| 1.0 + (j / BLOCK) as f64 * 0.05).collect();
    let x: Vec<f64> = (0..n * n).map(|_| lcg_uniform(&mut st)).collect();
    let mut h = vec![0.0f64; n * n];
    for r in 0..n {
        for i in 0..n {
            let xi = x[r * n + i] * gain[i];
            for j in i..n {
                h[i * n + j] += xi * x[r * n + j] * gain[j];
            }
        }
    }
    let s = 1.0 / n as f64;
    for i in 0..n {
        for j in i..n {
            let v = h[i * n + j] * s;
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
        h[i * n + i] += 0.25;
    }
    words(&h)
}

/// An SPD `[n, n]` Q32.32 fixture `XᵀX/t + I` from an LCG `X`, `t = n`.
fn lcg_spd(n: usize, seed: u64) -> Vec<i64> {
    let mut st = seed;
    let x: Vec<f64> = (0..n * n).map(|_| lcg_uniform(&mut st)).collect();
    let mut a = vec![0.0f64; n * n];
    for r in 0..n {
        for i in 0..n {
            let xi = x[r * n + i];
            for j in i..n {
                a[i * n + j] += xi * x[r * n + j];
            }
        }
    }
    let s = 1.0 / n as f64;
    for i in 0..n {
        for j in i..n {
            let v = a[i * n + j] * s;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] += 1.0;
    }
    words(&a)
}

/// The pooled row split of `beam_pass` (three `beam = 4` LCG fixtures) and the pooled column split
/// of `chol_solve`'s `solve_rhs` (one multi-RHS SPD fixture, `n = 512`, `k = 1536`) reproduce the
/// serial order's exact output bits: the digests below were captured from the serial body before
/// the pool dispatch moved the rows/columns to `width_global()` chunks. The elapsed wall of each
/// call is printed (three runs are averaged outside the test); it is not asserted.
#[test]
fn pooled_splits_reproduce_the_serial_digests() {
    use apeiron::learn::solve::chol::chol_solve;
    use std::time::Instant;
    // The serial (pre-pool) digests: `(o, n, act_order, wscale, seed)`, `beam = 4`, `damp = 0.01`.
    const GPTQ: [(usize, usize, bool, bool, u64, u64); 3] = [
        (192, 256, false, false, 0x00CC_0001, 0x86f4_42b7_65c8_9589),
        (192, 256, true, true, 0x00CC_0002, 0x3a92_7154_d1b9_d649),
        (160, 384, true, true, 0x00CC_0003, 0xa8bc_6d85_477a_4f76),
    ];
    for (i, &(o, n, act_order, wscale, seed, want)) in GPTQ.iter().enumerate() {
        let mut st = seed;
        let w: Vec<f64> = (0..o * n).map(|_| lcg_uniform(&mut st)).collect();
        let h = lcg_correlated_h(n, seed ^ 0x5A5A_5A5A);
        let t = Instant::now();
        let r = gptq_e8(&words(&w), &h, o, n, word(0.01), act_order, wscale, 4).expect("gptq_e8");
        let got = fnv_words(&r.s, &r.scale, &r.idx);
        println!(
            "gptq fixture {i} (o={o} n={n}): digest {got:#018x} wall {:.3}s",
            t.elapsed().as_secs_f64()
        );
        assert_eq!(
            got, want,
            "fixture {i}: pooled rows changed the output bits"
        );
    }
    let (n, k) = (512usize, 1536usize);
    let a = lcg_spd(n, 0x00CC_1001);
    let mut st = 0x00CC_2001u64;
    let b: Vec<f64> = (0..n * k).map(|_| lcg_uniform(&mut st)).collect();
    let t = Instant::now();
    let solved = chol_solve(&a, &words(&b), n, k).expect("chol_solve");
    let got = fnv_solution(&solved.x, solved.residual_max);
    println!(
        "chol n={n} k={k}: digest {got:#018x} residual {:#x} wall {:.3}s",
        solved.residual_max,
        t.elapsed().as_secs_f64()
    );
    assert_eq!(
        got, 0x1011_bbd5_03ad_9402,
        "chol_solve: pooled columns changed the solution bits"
    );
}
