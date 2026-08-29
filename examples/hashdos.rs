//! Can an attacker aim keys at one bucket?
//!
//! `finish` is splitmix64's finalizer: public constants, no seed, and a bijection -- so it has an
//! inverse, and anyone can compute the key that lands wherever they choose.

use apeiron::coord::{bucket, finish};
use apeiron::table::{shard_of, CountTable};
use std::time::Instant;

/// Invert `h ^= h >> s`.
fn unxorshift(mut v: u64, s: u32) -> u64 {
    let mut shift = s;
    let orig = v;
    let _ = orig;
    let mut acc = v;
    while shift < 64 {
        acc ^= v >> shift;
        let _ = acc; // the fold's last value is not read; only the bucket it lands in is
        shift += s;
        // recompute from scratch each round is unnecessary; the loop converges in ceil(64/s)
        v = acc;
    }
    acc
}

/// The inverse of `finish`, by inverting each step in reverse order.
fn finish_inv(mut h: u64) -> u64 {
    const INV_A: u64 = 0x319642b2d24d8ec3; // inverse of 0x94D049BB133111EB mod 2^64
    const INV_B: u64 = 0x96de1b173f119089; // inverse of 0xBF58476D1CE4E5B9 mod 2^64
    h = unxorshift(h, 31);
    h = h.wrapping_mul(INV_A);
    h = unxorshift(h, 27);
    h = h.wrapping_mul(INV_B);
    unxorshift(h, 30)
}

fn main() {
    // Confirm the inverse before relying on it.
    for probe in [0u64, 1, 0xDEAD_BEEF, u64::MAX, 0x0123_4567_89AB_CDEF] {
        assert_eq!(finish_inv(finish(probe)), probe, "inverse is wrong for {probe:#x}");
    }
    println!("finish is invertible: verified on 5 probes");

    let bits = 21u32;
    let n = 200_000usize;

    // Honest keys: whatever the world produces.
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; r };
    let honest: Vec<u64> = (0..n).map(|_| next() & 0x7FFF_FFFF_FFFF_FFFF).collect();

    // Attack keys: every one aimed at bucket zero, by inverting `finish`.
    let mut attack: Vec<u64> = Vec::with_capacity(n);
    let mut k = 0u64;
    while attack.len() < n {
        let target = k; // top `bits` are zero, so bucket() is zero
        let a = finish_inv(target);
        if a < (1 << 63) {
            attack.push(a);
        }
        k += 1;
    }

    let hb: Vec<usize> = honest.iter().map(|&a| bucket(a, bits) as usize).collect();
    let ab: Vec<usize> = attack.iter().map(|&a| bucket(a, bits) as usize).collect();
    println!(
        "  honest keys -> {} distinct buckets, {} distinct shards",
        hb.iter().collect::<std::collections::HashSet<_>>().len(),
        honest.iter().map(|&a| shard_of(a)).collect::<std::collections::HashSet<_>>().len()
    );
    println!(
        "  attack keys -> {} distinct buckets, {} distinct shards",
        ab.iter().collect::<std::collections::HashSet<_>>().len(),
        attack.iter().map(|&a| shard_of(a)).collect::<std::collections::HashSet<_>>().len()
    );

    for (label, keys) in [("honest", &honest), ("attack", &attack)] {
        let t = Instant::now();
        let table = CountTable::build(keys, bits);
        let build = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let acc: i64 = keys.iter().map(|&a| table.get(a)).sum();
        let probe = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {label:<8} build {build:>8.2} ms   probe {probe:>8.2} ms   acc {acc}"
        );
    }
}
