//! `tests/learn_gram.rs` — the exact-integer Gram fold: `gram_fold_i8` equals the naive `i64`
//! triple loop, folding in two batches equals one batch (the executable `gram_stack` law), and the
//! shape and overflow refusals hold. Grade `tested`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    reason = "the fixed-seed LCG maps to an `i8` palette and the short `r/i/j` indices are the \
              reference arithmetic's vocabulary; the pedantic lints stay on for the crate."
)]

use apeiron::learn::gram::{gram_fold_i8, GramError};

/// Deterministic 64-bit LCG (Knuth's MMIX constants), test-only.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// One `i8` drawn from the LCG, uniform over `[-128, 127]`.
fn pick_i8(state: &mut u64) -> i8 {
    (lcg(state) % 256) as i32 as i8
}

/// The naive reference: `g[i*d+j] + Σ_r x[r*d+i]·x[r*d+j]` in `i64`.
fn naive(g: &[i64], x: &[i8], rows: usize, d: usize) -> Vec<i64> {
    let mut out = g.to_vec();
    for r in 0..rows {
        for i in 0..d {
            for j in 0..d {
                let a = i64::from(x[r * d + i]);
                let b = i64::from(x[r * d + j]);
                out[i * d + j] += a * b;
            }
        }
    }
    out
}

#[test]
fn gram_fold_i8_exact() {
    const ROWS: usize = 37;
    const D: usize = 9;

    let mut state = 0x1234_5678_9abc_def0u64;
    let x: Vec<i8> = (0..ROWS * D).map(|_| pick_i8(&mut state)).collect();
    let g0 = vec![0i64; D * D];

    // One batch equals the naive triple loop.
    let want = naive(&g0, &x, ROWS, D);
    let got = gram_fold_i8(g0.clone(), &x, ROWS, D).expect("valid shape");
    assert_eq!(got, want, "one-batch fold differs from the naive reference");

    // Two batches equal one batch: fold X₁ then X₂ from the accumulator (the `gram_stack` law).
    let n1 = 20usize;
    let n2 = ROWS - n1;
    let after_first = gram_fold_i8(g0.clone(), &x[..n1 * D], n1, D).expect("first batch");
    let after_both = gram_fold_i8(after_first, &x[n1 * D..], n2, D).expect("second batch");
    assert_eq!(after_both, want, "two-batch fold differs from one batch");

    // Shape mismatch refuses: wrong `g` length and wrong `x` length.
    assert_eq!(
        gram_fold_i8(vec![0i64; D * D], &x, ROWS, D + 1),
        Err(GramError::ShapeMismatch)
    );
    assert_eq!(
        gram_fold_i8(vec![0i64; D * D - 1], &x, ROWS, D),
        Err(GramError::ShapeMismatch)
    );
    assert_eq!(
        gram_fold_i8(vec![0i64; D * D], &x[..x.len() - 1], ROWS, D),
        Err(GramError::ShapeMismatch)
    );

    // An accumulator pre-filled at `i64::MAX - 1` refuses with `Overflow` when a positive square
    // would leave the range. `x[0] = 2` gives `x[0]·x[0] = 4`, so `i64::MAX - 1 + 4` overflows.
    let small_x: [i8; 2] = [2, 0];
    let mut big = vec![0i64; 4];
    big[0] = i64::MAX - 1;
    assert_eq!(
        gram_fold_i8(big, &small_x, 1, 2),
        Err(GramError::Overflow),
        "a near-max accumulator must refuse an overflowing square"
    );
}
