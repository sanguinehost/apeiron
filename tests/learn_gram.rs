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

use apeiron::learn::gram::{gram_fold_i8, gram_fold_i8_kern, GramError};

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

/// The kernel route is bit-exact against the extracted body on the same inputs, carries the same
/// shape and overflow refusals, and decides the near-`i64` boundary through the extracted body
/// (where the scalar fold's first-overflowing-prefix rule differs from the final-total rule).
#[test]
fn gram_fold_i8_kern_exact() {
    const ROWS: usize = 37;
    const D: usize = 9;

    let mut state = 0x1234_5678_9abc_def0u64;
    let x: Vec<i8> = (0..ROWS * D).map(|_| pick_i8(&mut state)).collect();
    let g0 = vec![0i64; D * D];
    let want = naive(&g0, &x, ROWS, D);

    for g in [g0.clone(), vec![7i64; D * D], vec![-3i64; D * D]] {
        let exact = gram_fold_i8(g.clone(), &x, ROWS, D).expect("valid shape");
        let kern = gram_fold_i8_kern(g, &x, ROWS, D).expect("valid shape");
        assert_eq!(kern, exact, "kernel route differs from the extracted body");
    }
    let kern = gram_fold_i8_kern(g0.clone(), &x, ROWS, D).expect("valid shape");
    assert_eq!(kern, want, "kernel route differs from the naive reference");

    // Two batches equal one batch through the kernel route (the `gram_stack` law).
    let n1 = 20usize;
    let n2 = ROWS - n1;
    let after_first = gram_fold_i8_kern(g0.clone(), &x[..n1 * D], n1, D).expect("first batch");
    let after_both = gram_fold_i8_kern(after_first, &x[n1 * D..], n2, D).expect("second batch");
    assert_eq!(
        after_both, want,
        "two-batch kernel fold differs from one batch"
    );

    // A palette saturated at `-128` (the code the `Q8_0` fast bound admits at `|xq| = 128`) and at
    // `127`, with `rows` and `d` both off the 32-block boundary.
    for fill in [-128i8, 127] {
        let flat = vec![fill; ROWS * D];
        assert_eq!(
            gram_fold_i8_kern(g0.clone(), &flat, ROWS, D),
            gram_fold_i8(g0.clone(), &flat, ROWS, D),
            "saturated palette `{fill}` differs from the extracted body"
        );
    }

    // Shape mismatch refuses identically.
    assert_eq!(
        gram_fold_i8_kern(vec![0i64; D * D], &x, ROWS, D + 1),
        Err(GramError::ShapeMismatch)
    );
    assert_eq!(
        gram_fold_i8_kern(vec![0i64; D * D - 1], &x, ROWS, D),
        Err(GramError::ShapeMismatch)
    );
    assert_eq!(
        gram_fold_i8_kern(vec![0i64; D * D], &x[..x.len() - 1], ROWS, D),
        Err(GramError::ShapeMismatch)
    );

    // Empty folds succeed.
    assert_eq!(gram_fold_i8_kern(Vec::new(), &[], 0, 0), Ok(Vec::new()));
    assert_eq!(
        gram_fold_i8_kern(vec![5i64; 4], &[], 0, 2),
        Ok(vec![5i64; 4])
    );

    // A near-max accumulator refuses on the first overflowing term, as the extracted body does.
    let small_x: [i8; 2] = [2, 0];
    let mut big = vec![0i64; 4];
    big[0] = i64::MAX - 1;
    assert_eq!(
        gram_fold_i8_kern(big, &small_x, 1, 2),
        Err(GramError::Overflow)
    );

    // A prefix that overflows and a later term that returns the total in range: the extracted body
    // refuses, and the kernel route must refuse with it (the band sends the call to that body).
    let straddle: [i8; 4] = [127, 127, 127, -127];
    let mut near = vec![0i64; 4];
    near[0] = i64::MAX - 1;
    near[1] = i64::MAX - 1;
    assert_eq!(
        gram_fold_i8(near.clone(), &straddle, 2, 2),
        Err(GramError::Overflow)
    );
    assert_eq!(
        gram_fold_i8_kern(near, &straddle, 2, 2),
        Err(GramError::Overflow),
        "the kernel route must carry the extracted body's prefix refusal"
    );
}

/// R8: paired microbench of the ported product at the production shape (`d = 896`, `rows = 512`),
/// five alternating old/new runs. Run with
/// `cargo test --test learn_gram -- --ignored --nocapture gram_fold_i8_kern_microbench`.
#[test]
#[ignore = "paired microbench; needs --ignored --nocapture"]
#[allow(
    clippy::cast_precision_loss,
    reason = "the reported means and ns/MAC are f64 statistics of nanosecond counts; the counts are \
              far below 2^53, so the conversion is exact here"
)]
fn gram_fold_i8_kern_microbench() {
    use std::time::Instant;

    const ROWS: usize = 512;
    const D: usize = 896;

    let mut state = 0x0ddc_0ffe_e123_4567u64;
    let x: Vec<i8> = (0..ROWS * D).map(|_| pick_i8(&mut state)).collect();
    let g0 = vec![0i64; D * D];
    let macs = (ROWS as f64) * (D as f64) * (D as f64);

    // Warm both routes (page-in, kernel feature detection) before the paired runs.
    assert_eq!(
        gram_fold_i8(g0.clone(), &x, ROWS, D).expect("valid shape"),
        gram_fold_i8_kern(g0.clone(), &x, ROWS, D).expect("valid shape")
    );

    let mut old_ns = Vec::new();
    let mut new_ns = Vec::new();
    for run in 0..5 {
        let t = Instant::now();
        let old = gram_fold_i8(g0.clone(), &x, ROWS, D).expect("valid shape");
        old_ns.push(t.elapsed().as_secs_f64() * 1e9);

        let t = Instant::now();
        let new = gram_fold_i8_kern(g0.clone(), &x, ROWS, D).expect("valid shape");
        new_ns.push(t.elapsed().as_secs_f64() * 1e9);

        assert_eq!(old, new, "run {run}: kernel route differs");
    }

    let summary = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (mean, lo, hi)
    };
    let (old_mean, old_lo, old_hi) = summary(&old_ns);
    let (new_mean, new_lo, new_hi) = summary(&new_ns);
    println!(
        "gram d={D} rows={ROWS} macs={macs:.0}\n  old  mean {:.3} ns/MAC ({:.1} ms, spread {:.1}..{:.1} ms)\n  new  mean {:.3} ns/MAC ({:.1} ms, spread {:.1}..{:.1} ms)\n  ratio old/new {:.2}x",
        old_mean / macs,
        old_mean / 1e6,
        old_lo / 1e6,
        old_hi / 1e6,
        new_mean / macs,
        new_mean / 1e6,
        new_lo / 1e6,
        new_hi / 1e6,
        old_mean / new_mean,
    );
}
