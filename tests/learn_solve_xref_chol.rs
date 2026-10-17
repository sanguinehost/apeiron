//! Cross-reference bit-identity gate for `learn::solve::chol` (lane `q-xchol`).
//!
//! Drives the module's public entry points (`chol_solve`, `equilibrate`) over deterministic inline
//! LCG fixtures -- sizes `1`, `2`, `7`, `63`, `64`, `65`, `129`, `130` (around `CHOL_BLOCK = 64`),
//! an SPD matrix from a Gram plus ridge, thread counts `1`, `2`, `8`, and every refusal arm -- and
//! folds every output word and every error variant into an FNV-1a 64 hash, printed on one line. The
//! asserted constants are `taken from the pre-change code at 29ae4148`; the control-flow
//! restructure for Aeneas extraction must leave them unchanged. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the single test converts the Q32.32 grid to/from f64 to build deterministic SPD \
              fixtures and indexes row-major buffers by i/j/t; the pedantic lints stay on for the \
              crate's non-test code."
)]

use apeiron::learn::solve::chol::{chol_solve, equilibrate, SolveError};

/// The Q32.32 unit.
const ONE: i64 = 1 << 32;

/// `2^32`, the Q32.32 scale factor.
const SCALE: f64 = 4_294_967_296.0;

/// FNV-1a 64 offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64 prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Deterministic 64-bit LCG (Numerical Recipes constants): the test's only randomness source.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }
}

/// Q32.32 grid of an `f64` (round to nearest).
fn to_q32(x: f64) -> i64 {
    (x * SCALE).round() as i64
}

/// Fold one `u64` word into the FNV-1a 64 byte stream.
fn fnv_u64(mut h: u64, v: u64) -> u64 {
    for b in v.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Fold one `i64` word.
fn fnv_i64(h: u64, v: i64) -> u64 {
    fnv_u64(h, v as u64)
}

/// Fold a slice of `i64` words, preceded by its length.
fn fnv_slice(h: u64, s: &[i64]) -> u64 {
    let mut h = fnv_u64(h, s.len() as u64);
    for &v in s {
        h = fnv_i64(h, v);
    }
    h
}

/// Fold an error into its variant discriminant `1..=5`.
fn fnv_err(h: u64, e: SolveError) -> u64 {
    let code = match e {
        SolveError::ShapeMismatch => 1u64,
        SolveError::Empty => 2,
        SolveError::NonPositiveDiagonal => 3,
        SolveError::NotPositiveDefinite => 4,
        SolveError::Overflow => 5,
    };
    fnv_u64(h, code)
}

/// Fold a `chol_solve` outcome: `Ok` as `(x, residual_max)`, `Err` as its variant.
fn fnv_solved(h: u64, r: &Result<apeiron::learn::solve::chol::Solved, SolveError>) -> u64 {
    match r {
        Ok(s) => fnv_slice(fnv_i64(h, s.residual_max), &s.x),
        Err(e) => fnv_err(h, *e),
    }
}

/// Fold an `equilibrate` outcome.
fn fnv_equil(h: u64, r: &Result<(Vec<i64>, Vec<i64>), SolveError>) -> u64 {
    match r {
        Ok((scaled, d)) => fnv_slice(fnv_slice(h, scaled), d),
        Err(e) => fnv_err(h, *e),
    }
}

/// A well-conditioned SPD `A = M·Mᵀ/n + 0.1·I` (`n × n`) and an `n × k` right-hand side `B`, both
/// `f64`, from a fixed seed.
fn build_system(seed: u64, n: usize, k: usize) -> (Vec<i64>, Vec<i64>) {
    let mut lcg = Lcg::new(seed);
    let m: Vec<f64> = (0..n * n).map(|_| lcg.sym()).collect();
    let mut a = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for t in 0..n {
                s += m[i * n + t] * m[j * n + t];
            }
            a[i * n + j] = s / (n as f64) + if i == j { 0.1 } else { 0.0 };
        }
    }
    let b: Vec<f64> = (0..n * k).map(|_| lcg.sym()).collect();
    (
        a.iter().map(|&x| to_q32(x)).collect(),
        b.iter().map(|&x| to_q32(x)).collect(),
    )
}

/// The frozen constants the restructure must reproduce, taken from the pre-change code at 29ae4148.
const HASH_CHOL_SOLVE: u64 = 0xd5c0_b76f_4f37_8772;
const HASH_EQUILIBRATE: u64 = 0x5c78_71c8_341b_697b;
const HASH_REFUSAL: u64 = 0xbbf8_4411_c7a0_cc64;

/// The full fold: SPD solves over sizes around `CHOL_BLOCK` at `threads ∈ {1, 2, 8}`, direct
/// `equilibrate` calls, and every refusal arm. Prints the three hashes.
#[test]
fn xref_bit_identity_holds() {
    let mut hs = FNV_OFFSET;
    for &n in &[1usize, 2, 7, 63, 64, 65, 129, 130] {
        for &k in &[1usize, 3] {
            let (a, b) = build_system(0x9e37 + (n as u64) * 31 + k as u64, n, k);
            for &threads in &[1usize, 2, 8] {
                hs = fnv_u64(hs, n as u64);
                hs = fnv_u64(hs, k as u64);
                hs = fnv_u64(hs, threads as u64);
                let r = chol_solve(&a, &b, n, k);
                hs = fnv_solved(hs, &r);
            }
        }
    }

    let mut he = FNV_OFFSET;
    for &n in &[1usize, 2, 7, 63, 64, 65, 129, 130] {
        let (a, _b) = build_system(0x51ed + n as u64, n, 1);
        he = fnv_u64(he, n as u64);
        he = fnv_equil(he, &equilibrate(&a, n));
    }

    // Every refusal arm, at least once: Empty, ShapeMismatch, NonPositiveDiagonal,
    // NotPositiveDefinite, Overflow.
    let mut hr = FNV_OFFSET;
    hr = fnv_equil(hr, &equilibrate(&[], 0));
    hr = fnv_solved(hr, &chol_solve(&[], &[], 0, 0));
    hr = fnv_equil(hr, &equilibrate(&[ONE], 2));
    hr = fnv_solved(hr, &chol_solve(&[ONE], &[ONE], 2, 1));
    hr = fnv_solved(hr, &chol_solve(&[ONE, 0, 0, ONE], &[ONE, ONE], 2, 2));
    hr = fnv_solved(hr, &chol_solve(&[-ONE, 0, 0, ONE], &[ONE, 0], 2, 1));
    hr = fnv_solved(hr, &chol_solve(&[0, 0, 0, ONE], &[ONE, 0], 2, 1));
    hr = fnv_solved(
        hr,
        &chol_solve(&[ONE, 2 * ONE, 2 * ONE, ONE], &[ONE, ONE], 2, 1),
    );
    let big = 1i64 << 62;
    hr = fnv_solved(hr, &chol_solve(&[1, big, big, 1], &[ONE, ONE], 2, 1));
    // A 1-entry SPD solve that must succeed, so a failing refusal fold is distinguishable.
    hr = fnv_solved(hr, &chol_solve(&[4 * ONE], &[2 * ONE], 1, 1));

    println!("CHOL_XREF solve={hs:#018x} equilibrate={he:#018x} refusal={hr:#018x}");
    assert_eq!(hs, HASH_CHOL_SOLVE, "chol_solve bit-identity");
    assert_eq!(he, HASH_EQUILIBRATE, "equilibrate bit-identity");
    assert_eq!(hr, HASH_REFUSAL, "refusal bit-identity");
}
