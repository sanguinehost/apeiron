//! `tests/learn_solve_syrk.rs` — the weighted Q32.32 SYRK fold [`wsyrk`] and the elementwise
//! hadamard factor [`hadamard_into`].
//!
//! Positive: bit-equal agreement with a naive test-local `i128` triple loop (same single rounding)
//! over five shapes, weighted and unweighted; exact symmetry; deterministic output on the engine
//! pool; agreement to relative `1e-9` with an f64 evaluation of the `c_step` formula including
//! the `hl` factor. Boundary: every refusal arm (shape, `n == 0`, `o == 0`, input bound, overflow,
//! hadamard shape/overflow). Timing (`#[ignore]`): `o = 896`, `n = 4864`, three pooled
//! runs each, against the scalar `c_step` `A`-formation loop copied here.
//!
//! Grade `tested`; nothing here is `proved`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixture's xorshift draws map a 64-bit field to fixed-point `i64` entries through \
              a bounded modulo, and the f64 reference converts raw Q32.32 to reals; the kernels \
              index row-major `[o, n]` and `[n, n]` buffers by `i/j/oo`, the names of the `c_step` \
              arithmetic they mirror."
)]

use std::hint::black_box;
use std::time::Instant;

use apeiron::learn::solve::syrk::{hadamard_into, wsyrk, SyrkError};

/// Raw Q32.32 `1.0` (the unit weight, `w = None`).
const ONE_RAW: i64 = 1 << 32;

/// Deterministic xorshift64 (the tests' only randomness source).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// An `i64` in `[lo, hi)` for `0 < hi − lo ≤ 2⁴⁴`; the high 44 bits are used so a power-of-two
    /// span does not read a short LCG cycle.
    fn next_range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo) as u64;
        lo + ((self.next_u64() >> 20) % span) as i64
    }
}

/// The naive test-local reference: the `i128` triple loop with the same single round-to-nearest
/// shift by 64 that [`wsyrk`] applies.
fn naive_wsyrk(x: &[i64], w: Option<&[i64]>, o: usize, n: usize) -> Vec<i64> {
    let mut g = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc: i128 = 0;
            for oo in 0..o {
                let ww = match w {
                    Some(ws) => i128::from(ws[oo]),
                    None => i128::from(ONE_RAW),
                };
                acc += ww * i128::from(x[oo * n + i]) * i128::from(x[oo * n + j]);
            }
            g[i * n + j] = i64::try_from((acc + (1i128 << 63)) >> 64).expect("i64");
        }
    }
    g
}

#[test]
fn wsyrk_matches_naive_triple_loop_bit_equal() {
    let shapes: [(usize, usize); 5] = [(1, 1), (3, 5), (8, 64), (17, 70), (64, 130)];
    let mut lcg = Lcg::new(0x1234_5678_9abc_def0);
    for (o, n) in shapes {
        let x: Vec<i64> = (0..o * n)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        let w: Vec<i64> = (0..o)
            .map(|_| lcg.next_range(-(1i64 << 34), 1i64 << 34))
            .collect();
        let want = naive_wsyrk(&x, Some(&w), o, n);
        let got = wsyrk(&x, Some(&w), o, n).expect("weighted wsyrk");
        assert_eq!(got, want, "weighted shape ({o},{n})");
        let want_none = naive_wsyrk(&x, None, o, n);
        let got_none = wsyrk(&x, None, o, n).expect("unweighted wsyrk");
        assert_eq!(got_none, want_none, "unweighted shape ({o},{n})");
    }
}

#[test]
fn wsyrk_result_is_bit_symmetric() {
    let (o, n) = (24usize, 96usize);
    let mut lcg = Lcg::new(0xdead_beef_1234_5678);
    let x: Vec<i64> = (0..o * n)
        .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
        .collect();
    let w: Vec<i64> = (0..o).map(|_| lcg.next_range(0, 1i64 << 34)).collect();
    let g = wsyrk(&x, Some(&w), o, n).expect("wsyrk");
    for i in 0..n {
        for j in 0..n {
            assert_eq!(g[i * n + j], g[j * n + i], "mirror ({i},{j})");
        }
    }
}

#[test]
fn wsyrk_matches_f64_c_step_formula() {
    // The `c_step` metric `A[i,j] = hl[i,j]·Σ_o w_o·x_{o,i}·x_{o,j}` with `g_row`-like weights.
    // The draws are positive so every term is non-negative and the relative comparison is
    // well-posed (a cancelling sum would make a fixed-point `1e-9` relative bound meaningless).
    let (o, n) = (512usize, 8usize);
    let mut lcg = Lcg::new(0x0f0f_1e1e_2d2d_3c3c);
    let x: Vec<i64> = (0..o * n).map(|_| lcg.next_range(1, 1i64 << 31)).collect();
    let w: Vec<i64> = (0..o)
        .map(|_| lcg.next_range(1i64 << 28, 1i64 << 35))
        .collect();
    let hl: Vec<i64> = (0..n * n)
        .map(|_| lcg.next_range(1i64 << 31, 3i64 << 30))
        .collect();
    let g = wsyrk(&x, Some(&w), o, n).expect("wsyrk");
    let mut a = g.clone();
    hadamard_into(&mut a, &hl).expect("hadamard");
    let pow = |v: i64| (v as f64) * 2f64.powi(-32);
    let mut max_rel = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for oo in 0..o {
                acc += pow(w[oo]) * pow(x[oo * n + i]) * pow(x[oo * n + j]);
            }
            let want = pow(hl[i * n + j]) * acc;
            let got = pow(a[i * n + j]);
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel <= 1e-9,
                "A[{i},{j}]: got {got:e} want {want:e} rel {rel:e}"
            );
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    println!("wsyrk vs f64 c_step formula: max relative error {max_rel:e}");
}

#[test]
fn wsyrk_refusals_reach_every_arm() {
    // Shape: `x.len() != o·n`, then `w.len() != o`.
    assert_eq!(
        wsyrk(&[ONE_RAW; 5], None, 2, 3),
        Err(SyrkError::ShapeMismatch)
    );
    assert_eq!(
        wsyrk(&[ONE_RAW; 6], Some(&[ONE_RAW; 3]), 2, 3),
        Err(SyrkError::ShapeMismatch)
    );
    // `n == 0`, `o == 0`.
    assert_eq!(wsyrk(&[], None, 3, 0), Err(SyrkError::ZeroN));
    assert_eq!(wsyrk(&[], None, 0, 3), Err(SyrkError::ZeroO));
    // Input bound `2³⁶` (strict) on `x` and on `w`.
    assert_eq!(wsyrk(&[1i64 << 36], None, 1, 1), Err(SyrkError::InputBound));
    assert_eq!(
        wsyrk(&[-(1i64 << 36)], None, 1, 1),
        Err(SyrkError::InputBound)
    );
    let mut xb = vec![0i64; 3];
    xb[1] = 1i64 << 36;
    assert_eq!(wsyrk(&xb, None, 1, 3), Err(SyrkError::InputBound));
    assert_eq!(
        wsyrk(&[ONE_RAW], Some(&[1i64 << 36]), 1, 1),
        Err(SyrkError::InputBound)
    );
    // Overflow: `o` at the `2¹⁸` domain bound (`x` is a valid `o × 1` panel).
    let xo = vec![0i64; 1 << 18];
    assert_eq!(wsyrk(&xo, None, 1 << 18, 1), Err(SyrkError::Overflow));
    // Overflow: `n·n` leaves `usize` before any slice is touched.
    assert_eq!(
        wsyrk(&[], None, 1, usize::MAX / 2),
        Err(SyrkError::Overflow)
    );
    // `hadamard_into`: shape and overflow arms.
    let mut a = [ONE_RAW, ONE_RAW, ONE_RAW];
    assert_eq!(
        hadamard_into(&mut a, &[ONE_RAW; 2]),
        Err(SyrkError::ShapeMismatch)
    );
    assert_eq!(hadamard_into(&mut [], &[]), Ok(()));
    let mut big = [i64::MAX, 0];
    assert_eq!(
        hadamard_into(&mut big, &[i64::MAX, ONE_RAW]),
        Err(SyrkError::Overflow)
    );
}

/// The `c_step` metric formation (`steps.rs:288-341`, `A[i,j] = hl[i,j]·Σ_o b_sh[o,i]·b_sh[o,j]·
/// weight(oo)`) as the scalar f64 loop it is today, on the raw panel converted to reals; the `hl`
/// factor is a separate elementwise multiply and does not change the loop cost.
fn scalar_c_step_a(bsh: &[f64], wr: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for oo in 0..o {
                acc += bsh[oo * n + i] * bsh[oo * n + j] * wr[oo];
            }
            a[i * n + j] = acc;
        }
    }
    a
}

#[test]
#[ignore = "timing: o=896 n=4864, three runs per thread count plus the scalar loop; run with \
            --release --ignored --nocapture"]
fn timing_o896_n4864() {
    let (o, n) = (896usize, 4864usize);
    let mut lcg = Lcg::new(0x5eed_0f0f_a5a5_1234);
    let x: Vec<i64> = (0..o * n)
        .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
        .collect();
    let w: Vec<i64> = (0..o)
        .map(|_| lcg.next_range(1i64 << 28, 1i64 << 35))
        .collect();
    let upper = (o as f64) * (n as f64) * ((n + 1) as f64) / 2.0;
    let mut times = [0.0f64; 3];
    for t in &mut times {
        let start = Instant::now();
        let g = wsyrk(&x, Some(&w), o, n).expect("wsyrk");
        *t = start.elapsed().as_secs_f64();
        black_box(&g);
    }
    let mean = times.iter().sum::<f64>() / 3.0;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "wsyrk pooled: runs [{}] mean {mean:.3}s min {min:.3}s max {max:.3}s {:.3} GMAC/s \
         (upper-triangle MACs)",
        times.map(|v| format!("{v:.3}")).join(", "),
        upper / mean / 1e9
    );
    let bsh: Vec<f64> = x.iter().map(|&v| (v as f64) * 2f64.powi(-32)).collect();
    let wr: Vec<f64> = w.iter().map(|&v| (v as f64) * 2f64.powi(-32)).collect();
    let start = Instant::now();
    let a = scalar_c_step_a(&bsh, &wr, o, n);
    let scalar = start.elapsed().as_secs_f64();
    let full = (o as f64) * (n as f64) * (n as f64);
    println!(
        "scalar c_step A-formation loop (f64, full n×n): {scalar:.3}s {:.3} GMAC/s",
        full / scalar / 1e9
    );
    println!("ratio scalar/wsyrk {:.2}x", scalar / mean);
    // Instrument check: the two computations must agree before either timing is quoted.
    let g8 = wsyrk(&x, Some(&w), o, n).expect("wsyrk check");
    let mut max_abs = 0.0f64;
    let mut scale = 0.0f64;
    for k in 0..n * n {
        let d = ((g8[k] as f64) * 2f64.powi(-32) - a[k]).abs();
        if d > max_abs {
            max_abs = d;
        }
        if a[k].abs() > scale {
            scale = a[k].abs();
        }
    }
    println!("instrument check: max |wsyrk − scalar| {max_abs:e}, max |A| {scale:e}");
    assert!(
        max_abs <= 1e-6 * scale.max(1.0),
        "instrument mismatch {max_abs:e} at scale {scale:e}"
    );
}
