//! `tests/learn_solve_xref_inv.rs` — cross-reference bit-identity oracle for
//! [`inv_factor_upper`](apeiron::learn::solve::inv::inv_factor_upper) (lane `q-xinv`).
//!
//! The Aeneas control-flow restructure of `src/learn/solve/inv.rs` must leave the executed words
//! and the typed refusals unchanged. This test folds every output word and every error variant of
//! `inv_factor_upper` over deterministic fixtures — sizes including `1`, odd, and the values around
//! `INV_BLOCK = 64`, an SPD matrix over three repeated calls, and the refusal inputs — into an
//! FNV-1a 64 hash and asserts the constant. The constant is taken from the pre-change code at
//! `29ae4148`; the restructured code must reproduce it bit for bit.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    reason = "the fixtures build the Q32.32 grid from an `f64` LCG and index row-major buffers by \
              i/j/t; the pedantic lints stay on for the crate's non-test code."
)]

use apeiron::learn::solve::chol::SolveError;
use apeiron::learn::solve::inv::inv_factor_upper;

/// `2^32`, the Q32.32 scale factor.
const SCALE: f64 = 4_294_967_296.0;

/// FNV-1a 64 over a byte string.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// One stable byte per [`SolveError`] variant.
fn tag(e: SolveError) -> u8 {
    match e {
        SolveError::ShapeMismatch => 1,
        SolveError::Empty => 2,
        SolveError::NonPositiveDiagonal => 3,
        SolveError::NotPositiveDefinite => 4,
        SolveError::Overflow => 5,
    }
}

/// Fold one call's result: every output word, or the error's variant tag.
fn fold(mut h: u64, r: &Result<Vec<i64>, SolveError>) -> u64 {
    match r {
        Ok(v) => {
            h = fnv1a(h, &(v.len() as u64).to_le_bytes());
            for &w in v {
                h = fnv1a(h, &w.to_le_bytes());
            }
        }
        Err(e) => h = fnv1a(h, &[tag(*e)]),
    }
    h
}

/// Deterministic 64-bit LCG (Numerical Recipes constants).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// A deterministic SPD `[n, n]` on the Q32.32 grid: `A = M Mᵀ / n + I`, so `A` is symmetric,
/// diagonally dominant and positive-definite with a positive diagonal.
fn spd(n: usize, seed: u64) -> Vec<i64> {
    let mut lcg = Lcg::new(seed);
    let mut m = vec![0.0f64; n * n];
    for x in &mut m {
        *x = lcg.sym();
    }
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for t in 0..n {
                s += m[i * n + t] * m[j * n + t];
            }
            h[i * n + j] = s / n as f64 + if i == j { 1.0 } else { 0.0 };
        }
    }
    h.iter().map(|&x| (x * SCALE).round() as i64).collect()
}

/// Hash `inv_factor_upper` over the fixture set; the constant is the pre-change value.
fn xref_hash() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    // SPD matrices: `1`, odd, and around `INV_BLOCK = 64`, folded over three repeated calls.
    for &n in &[1usize, 3, 63, 64, 65, 128, 130] {
        let hq = spd(n, 0x51ED_0000 ^ n as u64);
        for _ in 0..3 {
            h = fold(h, &inv_factor_upper(&hq, n));
        }
    }
    // Refusals: empty, wrong shape, non-positive diagonal.
    h = fold(h, &inv_factor_upper(&[], 0));
    h = fold(h, &inv_factor_upper(&[1i64 << 32], 2));
    let nneg = 16usize;
    let mut neg = vec![0i64; nneg * nneg];
    for i in 0..nneg {
        neg[i * nneg + i] = -(1i64 << 32);
    }
    h = fold(h, &inv_factor_upper(&neg, nneg));
    // A non-SPD (indefinite) symmetric matrix: `[[1, 2], [2, 1]]` on the grid.
    let indef = vec![1i64 << 32, 2i64 << 32, 2i64 << 32, 1i64 << 32];
    h = fold(h, &inv_factor_upper(&indef, 2));
    // Large-magnitude SPD: exercises the `i128` path near the narrowing bound.
    let big = vec![1i64 << 40, (1i64 << 40) - 1, (1i64 << 40) - 1, 1i64 << 40];
    h = fold(h, &inv_factor_upper(&big, 2));
    h
}

/// The executed words and typed refusals of `inv_factor_upper` are unchanged by the control-flow
/// restructure: the hash is taken from the pre-change code at `29ae4148`.
#[test]
fn inv_factor_upper_is_bit_identical_to_the_pre_change_code() {
    let h = xref_hash();
    println!("inv_factor_upper xref hash = {h:#018x}");
    assert_eq!(
        h, 0x8c38_0c14_a025_ea8b,
        "inv_factor_upper output/refusal hash changed from the pre-change constant"
    );
}
