//! Head to head: hashtrinity's fold, apeiron's discard, apeiron's bijection.
//!
//! Same key sets, same method, three folds. Reported two ways, because a fold can separate keys
//! in 64 bits and still pile them into the same bucket: a table masks some of those bits away,
//! and which end it takes from decides whether a multiplicative hash works at all.

use apeiron::coord::{bucket, i21, Coord, Packed, PHI};
use std::collections::HashSet;

fn xor2(a: i64, b: i64) -> u64 {
    (a as u64).wrapping_mul(PHI) ^ (b as u64)
}
fn xor3(a: i64, b: i64, c: i64) -> u64 {
    ((a as u64).wrapping_mul(PHI) ^ (b as u64))
        .wrapping_mul(PHI)
        ^ (c as u64)
}

/// Collisions among full 64-bit addresses.
fn full(addrs: &[u64]) -> usize {
    addrs.len() - addrs.iter().copied().collect::<HashSet<_>>().len()
}

/// Collisions after a table keeps `bits` of the address. `high` takes the top bits, which is what
/// multiplicative hashing requires; `low` masks, which is what `% 2^n` does.
fn bucketed(addrs: &[u64], bits: u32, avalanche: bool) -> usize {
    let f = |a: u64| if avalanche { bucket(a, bits) } else { a >> (64 - bits) };
    addrs.len() - addrs.iter().map(|&a| f(a)).collect::<HashSet<_>>().len()
}

/// What the birthday bound gives for `n` keys in `2^bits` buckets -- the floor, not a defect.
fn floor(n: usize, bits: u32) -> String {
    let m = 2f64.powi(bits as i32);
    let expected_distinct = m * (1.0 - (-(n as f64) / m).exp());
    format!("{:>6.2}%", 100.0 * (n as f64 - expected_distinct) / n as f64)
}

fn pct(n: usize, of: usize) -> String {
    format!("{:>6.2}%", 100.0 * n as f64 / of as f64)
}

fn row(name: &str, addrs: &[u64], bits: u32) {
    let n = addrs.len();
    println!(
        "  {name:<26} {:>9} keys   full {:>8}   raw-top {:>8}   bucket() {:>8}   floor {}",
        n,
        pct(full(addrs), n),
        pct(bucketed(addrs, bits, false), n),
        pct(bucketed(addrs, bits, true), n),
        floor(n, bits),
    );
}

fn main() {
    // ── the grids hashtrinity was measured on ────────────────────────────────
    for (label, lo, hi, bits) in [("signed 128x128", -64i64, 64i64, 14u32), ("signed 1024x1024", -512, 512, 20)] {
        println!("\n{label}");
        let keys: Vec<(i64, i64)> = (lo..hi).flat_map(|x| (lo..hi).map(move |y| (x, y))).collect();
        row("hashtrinity xor", &keys.iter().map(|&(a, b)| xor2(a, b)).collect::<Vec<_>>(), bits);
        row("apeiron discard", &keys.iter().map(|&(a, b)| (a, b).coord()).collect::<Vec<_>>(), bits);
        let packed: Vec<u64> = keys
            .iter()
            .map(|&(a, b)| (a as i32, b as i32).packed())
            .collect();
        row("apeiron packed (B)", &packed, bits);
    }

    println!("\nsigned 32x32x32");
    let keys: Vec<(i64, i64, i64)> = (-16..16)
        .flat_map(|x| (-16..16).flat_map(move |y| (-16..16).map(move |z| (x, y, z))))
        .collect();
    row("hashtrinity xor", &keys.iter().map(|&(a, b, c)| xor3(a, b, c)).collect::<Vec<_>>(), 15);
    row("apeiron discard", &keys.iter().map(|&(a, b, c)| (a, b, c).coord()).collect::<Vec<_>>(), 15);
    let packed: Vec<u64> = keys
        .iter()
        .map(|&(a, b, c)| {
            (
                i21::new(a as i32).unwrap(),
                i21::new(b as i32).unwrap(),
                i21::new(c as i32).unwrap(),
            )
                .packed()
        })
        .collect();
    row("apeiron packed (B)", &packed, 15);

    // ── the shape a game actually produces: clustered, sparse, signed ────────
    println!("\nclustered world cells (8 clusters, 4096 cells each)");
    let mut keys: Vec<(i64, i64)> = Vec::new();
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for c in 0..8 {
        let (cx, cy) = ((c as i64 - 4) * 5000, (c as i64 % 3 - 1) * 5000);
        for _ in 0..4096 {
            let r = next();
            keys.push((
                cx + ((r >> 3) % 64) as i64 - 32,
                cy + ((r >> 32) % 64) as i64 - 32,
            ));
        }
    }
    keys.sort_unstable();
    keys.dedup();
    row("hashtrinity xor", &keys.iter().map(|&(a, b)| xor2(a, b)).collect::<Vec<_>>(), 16);
    row("apeiron discard", &keys.iter().map(|&(a, b)| (a, b).coord()).collect::<Vec<_>>(), 16);
    row(
        "apeiron packed (B)",
        &keys.iter().map(|&(a, b)| (a as i32, b as i32).packed()).collect::<Vec<_>>(),
        16,
    );

    // ── the known weak spot for a golden-ratio multiplier ────────────────────
    println!("\nfibonacci-offset keys (the multiplier's own blind spot)");
    let fibs: Vec<i64> = {
        let (mut a, mut b) = (1i64, 1i64);
        let mut v = vec![];
        while b < 1 << 31 {
            v.push(b);
            let t = a + b;
            a = b;
            b = t;
        }
        v
    };
    let keys: Vec<(i64, i64)> = fibs
        .iter()
        .flat_map(|&f| (0..512).map(move |k| (f + k, -k)))
        .collect();
    row("hashtrinity xor", &keys.iter().map(|&(a, b)| xor2(a, b)).collect::<Vec<_>>(), 14);
    row("apeiron discard", &keys.iter().map(|&(a, b)| (a, b).coord()).collect::<Vec<_>>(), 14);
    row(
        "apeiron packed (B)",
        &keys.iter().map(|&(a, b)| (a as i32, b as i32).packed()).collect::<Vec<_>>(),
        14,
    );
}
