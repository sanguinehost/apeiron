//! Why the fold's top bit leaks, and what fixes it.
//!
//! `2^63` is a zero divisor in `Z/2^64` with annihilator `2Z`, so flipping the top bit of a `mix2`
//! operand moves the product by `2^63 * (b & 1)` -- a function of the *other* operand's low bit,
//! not of the flipped one. The differential distribution collapses, which is what `SMHasher`'s
//! `DiffDist` reports at bit 63.
//!
//! Multiplication in this ring carries information upward only, so no further multiply recovers
//! it. The repair has to come from the other structure -- a `GF(2)`-linear shift/xor, which is
//! what `coord::finish` and `coord::spread` are.
//!
//! Run: `cargo run --release --example top_bit`

use apeiron::coord::finish;
use apeiron::fold::fold_bytes;
fn main() {
    // 1. Is 2^63 a zero divisor in Z/2^64, and how big is its annihilator?
    let t = 1u64 << 63;
    let ann = (0..1000u64).filter(|&x| t.wrapping_mul(x) == 0).count();
    println!("  2^63 * x == 0 for {ann}/1000 small x  (every even x: the annihilator is 2Z)");
    println!("  so a difference of 2^63 in one operand moves the product by 2^63*(b & 1) only\n");

    // 2. DiffDist's bit-63 measurement, local: flip the top bit, look at the difference.
    for (name, f) in [
        ("fold (as shipped)", (|k: &[u8]| fold_bytes(k)) as fn(&[u8]) -> u64),
        ("fold then finish", |k: &[u8]| finish(fold_bytes(k))),
    ] {
        for bit in [32u32, 62, 63] {
            let mut diffs = std::collections::HashSet::new();
            let mut hi32 = std::collections::HashSet::new();
            let mut s = 0x243F_6A88_85A3_08D3u64;
            for _ in 0..200_000 {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                let a = f(&s.to_le_bytes());
                let b = f(&(s ^ (1u64 << bit)).to_le_bytes());
                diffs.insert(a ^ b);
                hi32.insert((a ^ b) >> 32);
            }
            println!("  {name:<18} bit {bit:>2}: {:>6} distinct differences, {:>6} distinct high-32",
                diffs.len(), hi32.len());
        }
    }
}
