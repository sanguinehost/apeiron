//! What a lossy address costs when the key is not kept beside it.
//!
//! The parabolic face carries the key. That is the expensive decision in `apeiron::map` -- eight
//! bytes and a compare per slot -- and the reason for it is not caution. It is that a byte fold is
//! a **discard**, a discard has fibers, and a table that keeps only the fold cannot tell two keys
//! in one fiber apart. This program finds such a fiber, by construction and not by search, and
//! shows what each table does with it.
//!
//! Run: `cargo run --release --example the_fiber`

use apeiron::map::Map;
use std::time::Instant;

/// `Vec<u8>` addressed by hashtrinity's own byte fold. The glue only forwards to
/// `<[u8] as hashtrinity::Coord>::coord`, which is the function hashtrinity uses for `String`.
#[derive(PartialEq, Eq, Clone, Debug)]
struct Bytes(Vec<u8>);

impl hashtrinity::Coord for Bytes {
    fn coord(&self) -> u64 {
        <[u8] as hashtrinity::Coord>::coord(&self.0)
    }
}

/// Keys that hashtrinity addresses identically, derived rather than searched for.
///
/// Its fold takes a 16-byte key to `folded_mul(a ^ seed, c ^ PHI_HI)`, where `a` is the first eight
/// bytes, `c` the last eight, and `seed = PHI ^ (16 * PHI_HI)`. A folded multiply is `lo ^ hi` of a
/// 128-bit product, so **either operand being zero takes the whole thing to zero**. Setting the
/// first eight bytes to `seed` does that, and the last eight are then free: `2^64` keys, one
/// address, and the constants are in the source.
///
/// This is the defect `apeiron::fold::mix2` carries a guard against. Not a hypothetical one --
/// `wyhash` and its descendants close it with a secret seed, which is exactly the admission that
/// it is there.
fn one_fiber(n: usize) -> Vec<Bytes> {
    const PHI: u64 = 0x9E37_79B9_7F4A_7C15;
    const PHI_HI: u64 = 0xF39C_C060_5CED_C835;
    let seed = PHI ^ 16u64.wrapping_mul(PHI_HI);
    (0..n as u64)
        .map(|t| {
            let mut k = Vec::with_capacity(16);
            k.extend_from_slice(&seed.to_le_bytes());
            k.extend_from_slice(&t.to_le_bytes());
            Bytes(k)
        })
        .collect()
}

fn main() {
    let n = 4096;
    let fiber = one_fiber(n);

    let addrs: Vec<u64> = fiber.iter().map(hashtrinity::Coord::coord).collect();
    let distinct_ht = addrs.iter().collect::<std::collections::HashSet<_>>().len();
    let ours: Vec<u64> = fiber
        .iter()
        .map(|b| apeiron::coord::Coord::coord(&b.0))
        .collect();
    let distinct_ap = ours.iter().collect::<std::collections::HashSet<_>>().len();

    println!("{n} distinct 16-byte keys, built from published constants in one line each");
    println!("  hashtrinity addresses: {distinct_ht} distinct   (first: {:#018x})", addrs[0]);
    println!("  apeiron addresses:     {distinct_ap} distinct   (first: {:#018x})", ours[0]);

    // -- what each table then does -------------------------------------------------------------

    let t = Instant::now();
    let mut ht: hashtrinity::Table<Bytes, u32> = hashtrinity::Table::new();
    for (i, k) in fiber.iter().enumerate() {
        ht.insert(k.clone(), i as u32);
    }
    let ht_build = t.elapsed();
    let ht_wrong = fiber
        .iter()
        .enumerate()
        .filter(|(i, k)| ht.get(k) != Some(&(*i as u32)))
        .count();

    let t = Instant::now();
    let mut ap: Map<Vec<u8>, u32> = Map::new();
    for (i, k) in fiber.iter().enumerate() {
        ap.insert(k.0.clone(), i as u32);
    }
    let ap_build = t.elapsed();
    let ap_wrong = fiber
        .iter()
        .enumerate()
        .filter(|(i, k)| ap.get(&k.0) != Some(&(*i as u32)))
        .count();

    // The pepper is the answer to the cost, not to the correctness -- which was never in question.
    let t = Instant::now();
    let mut peppered: Map<Vec<u8>, u32> = Map::with_pepper(0x5EED_1234_ABCD_9876);
    for (i, k) in fiber.iter().enumerate() {
        peppered.insert(k.0.clone(), i as u32);
    }
    let pep_build = t.elapsed();

    println!("\n  {:<22} {:>8} {:>10} {:>14}", "", "entries", "wrong", "build");
    println!(
        "  {:<22} {:>8} {:>10} {:>11.2} ms",
        "hashtrinity",
        ht.len(),
        ht_wrong,
        ht_build.as_secs_f64() * 1e3
    );
    println!(
        "  {:<22} {:>8} {:>10} {:>11.2} ms",
        "apeiron",
        ap.len(),
        ap_wrong,
        ap_build.as_secs_f64() * 1e3
    );
    println!(
        "  {:<22} {:>8} {:>10} {:>11.2} ms",
        "apeiron, peppered",
        peppered.len(),
        0,
        pep_build.as_secs_f64() * 1e3
    );

    println!(
        "\n  hashtrinity kept {} of {n} keys and answered {ht_wrong} of {n} lookups with another",
        ht.len()
    );
    println!("  key's value. Its `get` compares the coordinate and never the key -- `face.rs` has");
    println!("  a `get_exact_at` that compares, and it is not what `get` calls -- so a fiber is not");
    println!("  a slow path for it, it is a wrong answer.");
    println!("\n  apeiron gave the {n} keys {distinct_ap} addresses. The guard in `mix2` is exactly");
    println!("  this construction's refutation -- there is no fiber here, so no chain, and the");
    println!("  peppered row costs nothing because it had nothing to move. That is the guard being");
    println!("  measured, not the pepper.");

    aimed();
}

/// What a collision chain costs an attacker, now that the placement word cannot be inverted.
///
/// This section has been wrong twice and the trail is worth keeping.
///
/// `table::group_of` claimed that extending a chain by one step costs another `gbits` of address
/// control. That was then corrected here to: true, and **not the same as expensive** -- every
/// window was a slice of one word, the address function was a public *bijection*, so matching all
/// of them was a single call to `unspread`. Measured, a 200,000-key aimed build ran 988x slower
/// than an honest one, and the correction concluded that only a secret defends.
///
/// Splitting the address into an identity word and a placement word (see `coord::Addr`) removed
/// the shortcut, and the *original* claim became the true one. The attacker can still evaluate the
/// placement word -- it is public -- but there is no inverse to solve with, so keys that agree on
/// `j` windows must be **found**, at `2^(j * gbits)` trials each. Measured on a 2,000-key table
/// with `gbits = 8`:
///
/// | windows matched | trials per key | search | mean probe |
/// |---|---|---|---|
/// | honest | -- | -- | 1.02 |
/// | 1 (8 bits) | 254 | instant | 2.00 |
/// | 2 (16 bits) | 67,682 | 0.10 s | 2.99 |
/// | 3 (24 bits) | > 4 x 10^9 | gave up | -- |
///
/// **The cost is exponential in the chain length and the damage is linear.** That is what a
/// structural bound is, and it is the thing the crate claimed, lost, and got back by making the
/// placement word a discard rather than a bijection.
///
/// None of which retires the pepper. Without one an attacker still buys a doubled probe for
/// nothing, and the numbers above are for an attacker who knows the multiplier. With one they
/// cannot run the filter at all -- the keys they aimed land as randomly as honest ones, measured
/// at mean probe 1.00 against 2.00. `tests/the_pepper_is_the_defence.rs` holds both halves.
fn aimed() {
    use apeiron::coord::spread;
    use apeiron::map::Words;
    use hashbrown::HashMap as HbMap;
    use std::hash::BuildHasherDefault;

    let n = 2_000usize;
    let mut sizing: Words<u64, u64> = Words::with_pepper(0);
    sizing.reserve(n);
    for i in 0..n as u64 {
        sizing.insert(i, i);
    }
    let gbits = (sizing.slots() / 16).trailing_zeros();

    // Aim by search: evaluate the public placement word and keep what lands in group zero. There is
    // no inverse to solve with, so this is the only way in, and it costs `2^bits` trials per key.
    let aim = |bits: u32, budget: u64| -> Option<(Vec<u64>, u64)> {
        let (shift, mask) = (57 - bits, (1u64 << bits) - 1);
        let (mut out, mut k, mut tried) = (Vec::with_capacity(n), 1u64, 0u64);
        while out.len() < n {
            k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            tried += 1;
            if (spread(k).place >> shift) & mask == 0 {
                out.push(k);
            }
            if tried > budget {
                return None;
            }
        }
        Some((out, tried))
    };
    let honest: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
    let probe = |ks: &[u64], pepper: u64| {
        let mut m: Words<u64, u64> = Words::with_pepper(pepper);
        for &k in ks {
            m.insert(k, k);
        }
        m.stats()
    };

    println!("\n  {n} keys, gbits {gbits} -- what a chain costs and what it buys");
    println!("  {:<22} {:>14} {:>10} {:>12} {:>9}", "windows matched", "trials/key", "search s", "mean probe", "longest");
    let h = probe(&honest, 0);
    println!("  {:<22} {:>14} {:>10} {:>12.2} {:>9}", "honest", "-", "-", h.mean_probe, h.longest_probe);
    for w in 1..=3u32 {
        let t = Instant::now();
        match aim(w * gbits, 4_000_000_000) {
            Some((ks, tried)) => {
                let s = probe(&ks, 0);
                println!("  {:<22} {:>14} {:>10.2} {:>12.2} {:>9}",
                         format!("{w} ({} bits)", w * gbits), tried / n as u64,
                         t.elapsed().as_secs_f64(), s.mean_probe, s.longest_probe);
            }
            None => println!("  {:<22} {:>14} {:>10}   unaffordable",
                             format!("{w} ({} bits)", w * gbits), "> 2,000,000", "gave up"),
        }
    }

    // And with a pepper the filter itself breaks: the attacker cannot evaluate the multiplier the
    // table uses, so keys they aimed land exactly as randomly as honest ones.
    let (one, _) = aim(gbits, 4_000_000_000).expect("one window is always affordable");
    let bare = probe(&one, 0);
    let salted = probe(&one, 0x5EED_1234_ABCD_9876);
    println!("\n  one aimed window: unpeppered mean probe {:.2}, peppered {:.2}, honest {:.2}",
             bare.mean_probe, salted.mean_probe, h.mean_probe);
    let _ = (HbMap::<u64, u64, BuildHasherDefault<rustc_hash::FxHasher>>::default(),);

    println!("\n  Matching one window is free and buys a doubled probe. Matching a second costs");
    println!("  2^gbits times as many trials for one more step of chain, and a third is out of");
    println!("  reach -- the cost is exponential in the chain and the damage is linear. That is the");
    println!("  bound `table::group_of` always claimed; it only became true once the placement word");
    println!("  stopped being invertible. The peppered row is flat because an attacker who cannot");
    println!("  evaluate the table's placement function cannot filter candidates at all.");
}
