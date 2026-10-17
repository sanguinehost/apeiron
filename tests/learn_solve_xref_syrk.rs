//! `tests/learn_solve_xref_syrk.rs` — cross-reference bit-identity oracle for the Q32.32 weighted
//! SYRK fold [`wsyrk`] and the elementwise hadamard factor [`hadamard_into`].
//!
//! Purpose: pin the exact output words and refusal variants of the two public entry points of
//! `src/learn/solve/syrk.rs` under a control-flow-only restructure (the Aeneas extraction rework).
//! Deterministic inline xorshift fixtures feed shapes `1`, odd, at and around `BLOCK`, and an
//! `o ≥ n` SPD panel; every output word and every error variant is folded into one FNV-1a 64 hash.
//! The same hash must hold before and after the restructure.
//!
//! Grade `tested`; nothing here is `proved`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::similar_names,
    reason = "the fixture's xorshift draws map a 64-bit field to bounded fixed-point `i64` entries; \
              the shapes are indexed by their `o`/`n` recurrence symbols."
)]

use apeiron::learn::solve::syrk::{hadamard_into, wsyrk, SyrkError};

/// Raw Q32.32 `1.0` (the unit weight, `w = None`).
const ONE_RAW: i64 = 1 << 32;
/// Column-block width of the restructured body (`syrk::BLOCK`); fixtures land at and around it.
const BLOCK: usize = 64;

/// Deterministic xorshift64.
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

    /// An `i64` in `[lo, hi)` for `0 < hi − lo ≤ 2⁴⁴`.
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

/// A stable tag per [`SyrkError`] variant, so every refusal arm is folded distinctly.
fn tag(e: SyrkError) -> u8 {
    match e {
        SyrkError::ShapeMismatch => 0,
        SyrkError::ZeroN => 1,
        SyrkError::ZeroO => 2,
        SyrkError::InputBound => 3,
        SyrkError::Overflow => 4,
    }
}

/// Fold a `wsyrk` result: `0` tag plus every output word on `Ok`, `1` tag plus the variant on `Err`.
fn fold_wsyrk(h: &mut u64, got: &Result<Vec<i64>, SyrkError>) {
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

/// Fold a `hadamard_into` call: the result tag/variant followed by the post-call buffer state.
fn fold_had(h: &mut u64, a: &[i64], got: Result<(), SyrkError>) {
    match got {
        Ok(()) => *h = fnv1a(*h, &[0]),
        Err(e) => *h = fnv1a(*h, &[1, tag(e)]),
    }
    for &word in a {
        *h = fnv1a(*h, &word.to_le_bytes());
    }
}

#[test]
fn syrk_xref_hash() {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;

    // Deterministic panels: `1`, odd, at `BLOCK`, just past `BLOCK`, and an `o ≥ n` SPD panel.
    let shapes: [(usize, usize, bool); 7] = [
        (1, 1, false),
        (3, 5, true),
        (5, 3, false),
        (8, BLOCK, true),
        (33, BLOCK + 1, true),
        (64, 130, false),
        (16, 6, true),
    ];
    let mut lcg = Lcg::new(0x5171_c0de_1234_abcd);
    for (o, n, weighted) in shapes {
        h = fnv1a(h, &(o as u64).to_le_bytes());
        h = fnv1a(h, &(n as u64).to_le_bytes());
        let x: Vec<i64> = (0..o * n)
            .map(|_| lcg.next_range(-(1i64 << 35), 1i64 << 35))
            .collect();
        let w: Vec<i64> = (0..o)
            .map(|_| lcg.next_range(-(1i64 << 34), 1i64 << 34))
            .collect();
        let wref = if weighted { Some(w.as_slice()) } else { None };
        fold_wsyrk(&mut h, &wsyrk(&x, wref, o, n));
        // Thread-count invariance is part of the pinned behaviour.
        fold_wsyrk(&mut h, &wsyrk(&x, wref, o, n));
        fold_wsyrk(&mut h, &wsyrk(&x, wref, o, n));
    }

    // Refusal arms: shape (x and w), zero n, zero o, input bound (x and w), domain overflow, n·n.
    let mut h2 = h;
    fold_wsyrk(&mut h2, &wsyrk(&[ONE_RAW; 5], None, 2, 3));
    fold_wsyrk(&mut h2, &wsyrk(&[ONE_RAW; 6], Some(&[ONE_RAW; 3]), 2, 3));
    fold_wsyrk(&mut h2, &wsyrk(&[], None, 3, 0));
    fold_wsyrk(&mut h2, &wsyrk(&[], None, 0, 3));
    fold_wsyrk(&mut h2, &wsyrk(&[1i64 << 36], None, 1, 1));
    fold_wsyrk(&mut h2, &wsyrk(&[-(1i64 << 36)], None, 1, 1));
    fold_wsyrk(&mut h2, &wsyrk(&[ONE_RAW], Some(&[1i64 << 36]), 1, 1));
    fold_wsyrk(&mut h2, &wsyrk(&vec![0i64; 1 << 18], None, 1 << 18, 1));
    fold_wsyrk(&mut h2, &wsyrk(&[], None, 1, usize::MAX / 2));
    h = h2;

    // hadamard: positive elementwise factor, then shape and overflow refusals.
    let mut a: Vec<i64> = (0..9)
        .map(|_| lcg.next_range(-(1i64 << 34), 1i64 << 34))
        .collect();
    let hl: Vec<i64> = (0..9)
        .map(|_| lcg.next_range(1i64 << 31, 3i64 << 30))
        .collect();
    let got = hadamard_into(&mut a, &hl);
    fold_had(&mut h, &a, got);
    let mut short = [ONE_RAW, ONE_RAW, ONE_RAW];
    let got = hadamard_into(&mut short, &[ONE_RAW; 2]);
    fold_had(&mut h, &short, got);
    let mut empty: [i64; 0] = [];
    let got = hadamard_into(&mut empty, &[]);
    fold_had(&mut h, &empty, got);
    let mut big = [i64::MAX, 0];
    let got = hadamard_into(&mut big, &[i64::MAX, ONE_RAW]);
    fold_had(&mut h, &big, got);

    println!("syrk xref hash = 0x{h:016x}");
    // taken from the pre-change code at 29ae4148
    assert_eq!(h, 0xc4b8_f760_0001_fe7b);
}
