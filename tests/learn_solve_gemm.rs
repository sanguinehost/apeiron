//! `tests/learn_solve_gemm.rs` — the blocked, threaded Q32.32 matrix products [`gemm_nn`] and
//! [`gemm_nt`].
//!
//! Positive: bit-equal agreement with a naive test-local `i128` triple loop (the same single
//! round-to-nearest shift by 32) over four shapes for both products; `gemm_nt(a, b) ==
//! gemm_nn(a, bᵀ)` bit-equal; bit-identical output for 1, 2 and 8 threads; and bit-equal agreement
//! with `learn::grad::dense::linear_fwd` under the stated mapping. Boundary: every refusal arm
//! (zero dimension, shape, input bound, `k ≥ MAX_K`, dimension-product overflow). Timing
//! (`#[ignore]`): `nt` at `(896, 4864, 896)` and `nn` at `(896, 4864, 4864)`, threads 1 and 8,
//! three runs each, plus the scalar `f64` `eigen_parts` `w1 = ΔW·Hl` loop at `(896, 4864, 4864)`
//! once as the reference.
//!
//! Grade `tested`; nothing here is `proved`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixture's xorshift draws map a 64-bit field to fixed-point `i64` entries through \
              a bounded modulo, and the timing reference converts raw Q32.32 to reals; the kernels \
              and the naive oracle index row-major `[m, k]`, `[k, n]`, `[n, k]` buffers by \
              `i/j/l`, the names of the products they mirror."
)]

use std::hint::black_box;
use std::time::Instant;

use apeiron::learn::grad::dense::linear_fwd;
use apeiron::learn::solve::gemm::{gemm_nn, gemm_nt, GemmError, MAX_K};

/// Raw Q32.32 `1.0`.
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

/// `(v + 2³¹) >> 32`, the single round-to-nearest the kernel applies; the naive oracle shares it.
fn round32(v: i128) -> i128 {
    (v + (1i128 << 31)) >> 32
}

/// The naive test-local `i128` oracle for `C = A·B` (`A` `[m, k]`, `B` `[k, n]`).
fn naive_nn(a: &[i64], b: &[i64], m: usize, k: usize, n: usize) -> Vec<i64> {
    let mut c = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i128 = 0;
            for l in 0..k {
                acc += i128::from(a[i * k + l]) * i128::from(b[l * n + j]);
            }
            c[i * n + j] = i64::try_from(round32(acc)).expect("i64");
        }
    }
    c
}

/// The naive test-local `i128` oracle for `C = A·Bᵀ` (`A` `[m, k]`, `B` `[n, k]`).
fn naive_nt(a: &[i64], b: &[i64], m: usize, k: usize, n: usize) -> Vec<i64> {
    let mut c = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i128 = 0;
            for l in 0..k {
                acc += i128::from(a[i * k + l]) * i128::from(b[j * k + l]);
            }
            c[i * n + j] = i64::try_from(round32(acc)).expect("i64");
        }
    }
    c
}

/// The `nt` operand `B` (`[n, k]`) transposed into the `nn` layout `[k, n]`:
/// `out[l·n + j] = b[j·k + l]`.
fn transpose_nk_to_kn(b: &[i64], k: usize, n: usize) -> Vec<i64> {
    let mut out = vec![0i64; k * n];
    for l in 0..k {
        for j in 0..n {
            out[l * n + j] = b[j * k + l];
        }
    }
    out
}

#[test]
fn gemm_nn_and_nt_match_the_naive_i128_triple_loop() {
    let shapes: [(usize, usize, usize); 4] = [(1, 1, 1), (3, 5, 7), (17, 70, 33), (64, 130, 65)];
    let mut lcg = Lcg::new(0x1234_5678_9abc_def0);
    for (m, k, n) in shapes {
        let a: Vec<i64> = (0..m * k)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        let b: Vec<i64> = (0..k * n)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        assert_eq!(
            gemm_nn(&a, &b, m, k, n, 3).expect("nn"),
            naive_nn(&a, &b, m, k, n),
            "nn shape ({m},{k},{n})"
        );
        assert_eq!(
            gemm_nt(&a, &b, m, k, n, 3).expect("nt"),
            naive_nt(&a, &b, m, k, n),
            "nt shape ({m},{k},{n})"
        );
    }
}

#[test]
fn gemm_nt_equals_gemm_nn_of_the_transpose() {
    let shapes: [(usize, usize, usize); 3] = [(3, 5, 7), (17, 70, 33), (64, 130, 65)];
    let mut lcg = Lcg::new(0x0f0f_1e1e_2d2d_3c3c);
    for (m, k, n) in shapes {
        let a: Vec<i64> = (0..m * k)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        // `b` is the `[n, k]` operand of `nt`; `bt` is the same map in the `[k, n]` layout.
        let b: Vec<i64> = (0..n * k)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        let bt = transpose_nk_to_kn(&b, k, n);
        assert_eq!(
            gemm_nt(&a, &b, m, k, n, 2).expect("nt"),
            gemm_nn(&a, &bt, m, k, n, 5).expect("nn of transpose"),
            "nt vs nn∘transpose shape ({m},{k},{n})"
        );
    }
}

#[test]
fn gemm_thread_counts_are_bit_identical() {
    let shapes: [(usize, usize, usize); 2] = [(17, 70, 33), (64, 130, 65)];
    let mut lcg = Lcg::new(0x0bad_f00d_5eed_1234);
    for (m, k, n) in shapes {
        let a: Vec<i64> = (0..m * k)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        let b: Vec<i64> = (0..k * n)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        let nn1 = gemm_nn(&a, &b, m, k, n, 1).expect("nn 1");
        assert_eq!(
            nn1,
            gemm_nn(&a, &b, m, k, n, 2).expect("nn 2"),
            "nn threads 1 vs 2 ({m},{k},{n})"
        );
        assert_eq!(
            nn1,
            gemm_nn(&a, &b, m, k, n, 8).expect("nn 8"),
            "nn threads 1 vs 8 ({m},{k},{n})"
        );
        let nt1 = gemm_nt(&a, &b, m, k, n, 1).expect("nt 1");
        assert_eq!(
            nt1,
            gemm_nt(&a, &b, m, k, n, 2).expect("nt 2"),
            "nt threads 1 vs 2 ({m},{k},{n})"
        );
        assert_eq!(
            nt1,
            gemm_nt(&a, &b, m, k, n, 8).expect("nt 8"),
            "nt threads 1 vs 8 ({m},{k},{n})"
        );
    }
}

#[test]
fn gemm_nt_agrees_bit_equal_with_dense_linear_fwd() {
    // `linear_fwd(x, w, n, i, o)` computes `y[r,c] = round32(Σ_k x[r,k]·w[c,k])` with `x` `n×i` and
    // `w` `o×i`. That is `gemm_nt(a = x, b = w, m = n, k = i, n = o)`: the row count is `x`'s row
    // count, the contraction is `x`'s column count, and `gemm_nt`'s output column count is `w`'s row
    // count. Same product, same `(v + 2³¹) >> 32` rounding, same `k`-ascending accumulation.
    let shapes: [(usize, usize, usize); 2] = [(4, 7, 5), (33, 130, 17)];
    let mut lcg = Lcg::new(0x5a5a_3c3c_9696_0f0f);
    for (rows, inner, out) in shapes {
        let x: Vec<i64> = (0..rows * inner)
            .map(|_| lcg.next_range(-(1i64 << 34), 1i64 << 34))
            .collect();
        let w: Vec<i64> = (0..out * inner)
            .map(|_| lcg.next_range(-(1i64 << 34), 1i64 << 34))
            .collect();
        let want = linear_fwd(&x, &w, rows, inner, out).expect("linear_fwd");
        for threads in [1usize, 3, 8] {
            let got = gemm_nt(&x, &w, rows, inner, out, threads).expect("gemm_nt");
            assert_eq!(
                got, want,
                "linear_fwd vs gemm_nt ({rows},{inner},{out}) threads {threads}"
            );
        }
    }
}

#[test]
fn gemm_refusals_reach_every_arm() {
    // Zero dimension, on each of `m`, `k`, `n`.
    assert_eq!(gemm_nn(&[], &[], 0, 1, 1, 1), Err(GemmError::ZeroDim));
    assert_eq!(gemm_nn(&[], &[], 1, 0, 1, 1), Err(GemmError::ZeroDim));
    assert_eq!(gemm_nn(&[], &[], 1, 1, 0, 1), Err(GemmError::ZeroDim));
    assert_eq!(gemm_nt(&[], &[], 1, 1, 0, 1), Err(GemmError::ZeroDim));
    // Contraction-domain bound `k < MAX_K`, refused before any slice is materialised.
    assert_eq!(gemm_nn(&[], &[], 1, MAX_K, 1, 1), Err(GemmError::Overflow));
    assert_eq!(gemm_nt(&[], &[], 1, MAX_K, 1, 1), Err(GemmError::Overflow));
    // A dimension product that leaves `usize` (`m·k`).
    assert_eq!(
        gemm_nn(&[], &[], usize::MAX / 2, 4, 1, 1),
        Err(GemmError::Overflow)
    );
    // Shape: `a.len() != m·k`, then `b.len() != k·n`, in either layout.
    assert_eq!(
        gemm_nn(&[ONE_RAW; 5], &[ONE_RAW; 3], 2, 3, 1, 1),
        Err(GemmError::ShapeMismatch)
    );
    assert_eq!(
        gemm_nn(&[ONE_RAW; 6], &[ONE_RAW; 2], 2, 3, 1, 1),
        Err(GemmError::ShapeMismatch)
    );
    assert_eq!(
        gemm_nt(&[ONE_RAW; 6], &[ONE_RAW; 2], 2, 3, 1, 1),
        Err(GemmError::ShapeMismatch)
    );
    // Input bound `2³⁶` (strict) on `a` and on `b`.
    assert_eq!(
        gemm_nn(&[1i64 << 36], &[ONE_RAW], 1, 1, 1, 1),
        Err(GemmError::InputBound)
    );
    assert_eq!(
        gemm_nn(&[-(1i64 << 36)], &[ONE_RAW], 1, 1, 1, 1),
        Err(GemmError::InputBound)
    );
    assert_eq!(
        gemm_nn(&[ONE_RAW], &[1i64 << 36], 1, 1, 1, 1),
        Err(GemmError::InputBound)
    );
    assert_eq!(
        gemm_nt(&[ONE_RAW], &[-(1i64 << 36)], 1, 1, 1, 1),
        Err(GemmError::InputBound)
    );
    // One entry at the bound inside a longer panel is still refused.
    let mut ab = vec![0i64; 4];
    ab[3] = 1i64 << 36;
    assert_eq!(
        gemm_nn(&ab, &[ONE_RAW; 4], 2, 2, 2, 1),
        Err(GemmError::InputBound)
    );
}

/// The `eigen_parts` `w1 = ΔW·Hl` loop (`corrector.rs:268-277`) exactly as the scalar `f64` it is
/// today: `w1[i·n+k] = Σ_j delta_w[i·n+j]·hl[j·n+k]` over `o = 896`, `n = 4864`, with `hl` read
/// column-wise.
fn scalar_eigen_w1(delta_w: &[f64], hl: &[f64], o: usize, n: usize) -> Vec<f64> {
    let mut w1 = vec![0.0f64; o * n];
    for i in 0..o {
        for k in 0..n {
            let mut acc = 0.0f64;
            for j in 0..n {
                acc += delta_w[i * n + j] * hl[j * n + k];
            }
            w1[i * n + k] = acc;
        }
    }
    w1
}

#[test]
#[ignore = "timing: nt (896,4864,896) and nn (896,4864,4864), threads 1 and 8, three runs each, \
            plus the scalar f64 eigen_parts w1 loop; run with --release --ignored --nocapture"]
fn timing_nt_and_nn() {
    let mut lcg = Lcg::new(0x5eed_0f0f_a5a5_1234);
    let (m, k, n) = (896usize, 4864usize, 896usize);
    let a: Vec<i64> = (0..m * k)
        .map(|_| lcg.next_range(-(1i64 << 33), 1i64 << 33))
        .collect();
    let b_nt: Vec<i64> = (0..n * k)
        .map(|_| lcg.next_range(-(1i64 << 33), 1i64 << 33))
        .collect();
    let macs_nt = (m as f64) * (k as f64) * (n as f64);
    let mut means_nt = [0.0f64; 2];
    for (idx, &threads) in [1usize, 8usize].iter().enumerate() {
        let mut times = [0.0f64; 3];
        for t in &mut times {
            let start = Instant::now();
            let c = gemm_nt(&a, &b_nt, m, k, n, threads).expect("nt");
            *t = start.elapsed().as_secs_f64();
            black_box(&c);
        }
        let mean = times.iter().sum::<f64>() / 3.0;
        let min = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        means_nt[idx] = mean;
        println!(
            "gemm_nt ({m},{k},{n}) threads={threads}: runs [{}] mean {mean:.3}s min {min:.3}s max \
             {max:.3}s {:.3} GMAC/s",
            times.map(|v| format!("{v:.3}")).join(", "),
            macs_nt / mean / 1e9
        );
    }
    drop(b_nt);

    let (m2, k2, n2) = (896usize, 4864usize, 4864usize);
    let b_nn: Vec<i64> = (0..k2 * n2)
        .map(|_| lcg.next_range(-(1i64 << 33), 1i64 << 33))
        .collect();
    let macs_nn = (m2 as f64) * (k2 as f64) * (n2 as f64);
    let mut means_nn = [0.0f64; 2];
    for (idx, &threads) in [1usize, 8usize].iter().enumerate() {
        let mut times = [0.0f64; 3];
        for t in &mut times {
            let start = Instant::now();
            let c = gemm_nn(&a, &b_nn, m2, k2, n2, threads).expect("nn");
            *t = start.elapsed().as_secs_f64();
            black_box(&c);
        }
        let mean = times.iter().sum::<f64>() / 3.0;
        let min = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        means_nn[idx] = mean;
        println!(
            "gemm_nn ({m2},{k2},{n2}) threads={threads}: runs [{}] mean {mean:.3}s min {min:.3}s max \
             {max:.3}s {:.3} GMAC/s",
            times.map(|v| format!("{v:.3}")).join(", "),
            macs_nn / mean / 1e9
        );
    }

    // The scalar f64 `eigen_parts` `w1 = ΔW·Hl` reference at the same shape.
    let dw: Vec<f64> = a.iter().map(|&v| (v as f64) * 2f64.powi(-32)).collect();
    let hl: Vec<f64> = b_nn.iter().map(|&v| (v as f64) * 2f64.powi(-32)).collect();
    let start = Instant::now();
    let w1 = scalar_eigen_w1(&dw, &hl, m2, n2);
    let scalar = start.elapsed().as_secs_f64();
    println!(
        "scalar f64 eigen_parts w1 loop ({m2},{k2},{n2}): {scalar:.3}s {:.3} GMAC/s",
        macs_nn / scalar / 1e9
    );
    println!(
        "ratio scalar/gemm_nn(1) {:.2}x  scalar/gemm_nn(8) {:.2}x",
        scalar / means_nn[0],
        scalar / means_nn[1]
    );
    println!(
        "nt/gemm_nn: nt(1) {:.3} GMAC/s, nn(1) {:.3} GMAC/s",
        macs_nt / means_nt[0] / 1e9,
        macs_nn / means_nn[0] / 1e9
    );

    // Instrument check: the two computations must agree before either timing is quoted.
    let c_nn = gemm_nn(&a, &b_nn, m2, k2, n2, 8).expect("nn check");
    let mut max_abs = 0.0f64;
    let mut scale = 0.0f64;
    for idx in 0..m2 * n2 {
        let d = ((c_nn[idx] as f64) * 2f64.powi(-32) - w1[idx]).abs();
        if d > max_abs {
            max_abs = d;
        }
        if w1[idx].abs() > scale {
            scale = w1[idx].abs();
        }
    }
    println!("instrument check: max |gemm_nn − scalar| {max_abs:e}, max |w1| {scale:e}");
    assert!(
        max_abs <= 1e-6 * scale.max(1.0),
        "instrument mismatch {max_abs:e} at scale {scale:e}"
    );
}
