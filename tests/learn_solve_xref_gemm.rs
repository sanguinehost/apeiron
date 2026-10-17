//! `tests/learn_solve_xref_gemm.rs` — cross-reference bit-identity oracle for the Q32.32 blocked,
//! threaded matrix products [`gemm_nn`] and [`gemm_nt`] of `src/learn/solve/gemm.rs`.
//!
//! Purpose: pin the exact output words and refusal variants of the two public entry points under a
//! control-flow-only restructure (the Aeneas extraction rework of `block_rows`). Deterministic
//! inline xorshift fixtures feed shapes below and above the `64`-wide output block (`mc < 64`,
//! `mc > 64`) whose `nc` is never a multiple of `4`, plus one `k ≥ MAX_K` overflow refusal; every
//! output word is folded into one FNV-1a 64 hash over two repeated calls. The same hash must hold
//! before and after the restructure.
//!
//! Grade `tested`; nothing here is `proved`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names,
    reason = "the fixture's xorshift draws map a 64-bit field to bounded fixed-point `i64` entries; \
              the shapes are indexed by their `m`/`k`/`n` recurrence symbols."
)]

use apeiron::learn::solve::gemm::{gemm_nn, gemm_nt, GemmError, MAX_K};

/// Deterministic xorshift64 (the test's only randomness source).
struct Lcg(u64);

impl Lcg {
    const fn new(seed: u64) -> Self {
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

/// FNV-1a 64 fold of `bytes` into `h`.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A stable tag per [`GemmError`] variant, so every refusal arm is folded distinctly.
fn tag(e: GemmError) -> u8 {
    match e {
        GemmError::ZeroDim => 0,
        GemmError::ShapeMismatch => 1,
        GemmError::InputBound => 2,
        GemmError::Overflow => 3,
    }
}

/// Fold a `gemm_nn`/`gemm_nt` result: `0` tag plus every output word on `Ok`, `1` tag plus the
/// variant on `Err`.
fn fold(h: &mut u64, got: &Result<Vec<i64>, GemmError>) {
    match got {
        Ok(v) => {
            *h = fnv1a(*h, &[0]);
            for &word in v {
                *h = fnv1a(*h, &word.to_le_bytes());
            }
        }
        Err(e) => *h = fnv1a(*h, &[1, tag(*e)]),
    }
}

#[test]
fn gemm_xref_hash() {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;

    // `mc < 64` (5), `mc > 64` (67, 130); `nc` in {9, 70, 66} is never a multiple of 4; `k` spans
    // below, at and above the `128`-deep contraction block.
    let shapes: [(usize, usize, usize); 3] = [(5, 7, 9), (67, 131, 70), (130, 260, 66)];
    let mut lcg = Lcg::new(0x9e37_79b9_7f4a_7c15);
    for (m, k, n) in shapes {
        h = fnv1a(h, &(m as u64).to_le_bytes());
        h = fnv1a(h, &(k as u64).to_le_bytes());
        h = fnv1a(h, &(n as u64).to_le_bytes());
        let a: Vec<i64> = (0..m * k)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        // `b` is the `[k, n]` operand of `nn`; `bt` is the same map in the `[n, k]` layout of `nt`.
        let b_nn: Vec<i64> = (0..k * n)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        let b_nt: Vec<i64> = (0..n * k)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        // Repeated calls are part of the pinned behaviour: the pooled kernel is deterministic.
        for _ in 0..2 {
            fold(&mut h, &gemm_nn(&a, &b_nn, m, k, n));
            fold(&mut h, &gemm_nt(&a, &b_nt, m, k, n));
        }
    }

    // Overflow refusal: `k ≥ MAX_K`, refused before any slice is materialised, in either layout.
    fold(&mut h, &gemm_nn(&[], &[], 1, MAX_K, 1));
    fold(&mut h, &gemm_nt(&[], &[], 1, MAX_K, 1));

    println!("gemm xref hash = 0x{h:016x}");
    // taken from the pre-change code at d775efc7
    assert_eq!(h, 0x09f7_a5df_6f2a_6e87);
}
