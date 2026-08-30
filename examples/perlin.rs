//! `SMHasher`'s `PerlinNoise` keyset: one key under many seeds.
//!
//! 4,096 keys x 4,096 seeds. It found the fold's worst defect -- at a two-byte key the second
//! `mix2` operand was zero, so the hash was a function of `key + seed` and the two were
//! interchangeable. 14,741,760 collisions of 16,777,216, which this reproduces exactly.
//!
//! Run: `cargo run --release --example perlin`
#![expect(clippy::cast_possible_truncation, reason = "counts are bounded by the loop")]

use apeiron::fold::fold_bytes_with;
use std::collections::HashSet;
fn main() {
    for input_len in [1usize, 2, 3, 4, 8] {
        let mut all: HashSet<u64> = HashSet::new();
        let mut n = 0u64;
        for x in 0..4096u64 {
            let key = &x.to_le_bytes()[..input_len];
            for y in 0..4096u64 {
                all.insert(fold_bytes_with(y, key));
                n += 1;
            }
        }
        println!("  inputLen={input_len:<3} {n} hashes -> {} distinct  ({} collisions)", all.len(), n as usize - all.len());
    }
}
