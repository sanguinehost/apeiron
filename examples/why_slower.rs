//! Where the serial version's time goes: hashing, then the flat unpartitioned build, then the probe.
//!
//! The partitioned build is in `examples/phases.rs`; this one is the shape it replaced.
#![expect(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "this is a measurement harness, not library code, and each lint below is listed because this \
              file actually triggers it: sizes and indices narrowed to build keys, bounded by the loop that \
              produced them; the algebra's own names. Listed individually rather than suppressed wholesale, \
              so a pedantic lint this file does not yet trigger still surfaces"
)]

use apeiron::coord::{bucket, i21, Packed};
use std::time::Instant;

fn main() {
    let n = 1_000_000usize;
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r };
    let keys: Vec<u64> = (0..n)
        .map(|_| {
            let v = next();
            let c = |s: u32| ((v >> s) as i32 % 400) - 200;
            (i21::new(c(0)).unwrap(), i21::new(c(20)).unwrap(), i21::new(c(40)).unwrap()).packed()
        })
        .collect();
    let mut occ = keys.clone(); occ.sort_unstable(); occ.dedup();
    let bits = (occ.len() * 2).next_power_of_two().trailing_zeros();
    let m = 1usize << bits;
    println!("n={n}  distinct={}  table=2^{bits}", occ.len());

    let t = Instant::now();
    let addrs: Vec<u64> = keys.iter().map(|&a| bucket(a, bits)).collect();
    println!("  hash only            {:>8.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    std::hint::black_box(&addrs);

    let t = Instant::now();
    let mut slots = vec![(u64::MAX, 0i64); m];
    for &a in &keys {
        let mut i = bucket(a, bits) as usize;
        loop {
            if slots[i].0 == u64::MAX { slots[i] = (a, 1); break; }
            if slots[i].0 == a { slots[i].1 += 1; break; }
            i = (i + 1) & (m - 1);
        }
    }
    println!("  build (AoS)          {:>8.2} ms", t.elapsed().as_secs_f64() * 1000.0);

    let t = Instant::now();
    let mut acc = 0i64;
    for &a in &occ {
        let mut i = bucket(a, bits) as usize;
        loop {
            if slots[i].0 == u64::MAX { break; }
            if slots[i].0 == a { acc += slots[i].1; break; }
            i = (i + 1) & (m - 1);
        }
    }
    println!("  probe                {:>8.2} ms   (acc {acc})", t.elapsed().as_secs_f64() * 1000.0);
}
