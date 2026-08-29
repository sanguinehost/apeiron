//! The address function alone, on a dependent chain -- which is how a probe consumes it.
//!
//! Each iteration feeds the previous result back in, so this measures *latency*, not throughput.
//! That is the number that matters: the group index addresses the tag-line load, so nothing in the
//! lookup can start until the address is finished.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::inline_always,
    reason = "this is a measurement harness, not library code, and each lint below is listed because this \
              file actually triggers it: sizes and indices narrowed to build keys, bounded by the loop that \
              produced them; counts converted to floats to report rates; a harness would need 2^53 samples \
              to lose a bit; the attribute being measured. Listed individually rather than suppressed \
              wholesale, so a pedantic lint this file does not yet trigger still surfaces"
)]

use std::hint::black_box;
use std::time::Instant;

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;
const PSI: u64 = 0x94D0_49BB_1331_11EB;

#[inline(always)] fn two_serial(x: u64) -> u64 { let h = x.wrapping_mul(PHI); let h = h ^ (h >> 48); h.wrapping_mul(PSI) }
#[inline(always)] fn mulx_rot(x: u64) -> u64 { let w = u128::from(x).wrapping_mul(u128::from(PHI)); let (lo, hi) = (w as u64, (w >> 64) as u64); hi.rotate_left(32) ^ lo }
#[inline(always)] fn mulx_fold_mul(x: u64) -> u64 { let w = u128::from(x).wrapping_mul(u128::from(PHI)); (((w >> 64) as u64) ^ (w as u64)).wrapping_mul(PSI) }
#[inline(always)] fn one_mul(x: u64) -> u64 { x.wrapping_mul(PHI) }

fn chain(name: &str, f: impl Fn(u64) -> u64) {
    const N: u64 = 40_000_000;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut x = black_box(1u64);
        let t = Instant::now();
        for _ in 0..N { x = f(x) | 1; }
        let ns = t.elapsed().as_secs_f64() * 1e9 / N as f64;
        black_box(x);
        best = best.min(ns);
    }
    // 5.7 GHz boost on this part; report cycles at the measured clock-independent ratio instead.
    println!("  {name:<22} {best:>6.2} ns/link   ~{:>4.1} cycles", best * 5.7);
}

fn main() {
    println!("  dependent-chain latency of the address function (lower is better)");
    chain("1 multiply", one_mul);
    chain("2 serial muls (now)", two_serial);
    chain("mulx + rotl + xor", mulx_rot);
    chain("mulx + xor + mul", mulx_fold_mul);
}
