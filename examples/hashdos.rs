//! What a collision chain costs an attacker of a `CountTable`.
//!
//! This example used to invert `finish`. That was the right attack against the table as it stood:
//! the address was splitmix64's finalizer, public constants, no seed, and a **bijection** -- so
//! anyone could compute the key that lands wherever they chose, and matching every window of the
//! descent cost one call.
//!
//! `CountTable` now addresses through `coord::spread`, which splits into an identity word (a
//! bijection, because a count table's key *is* its address and the slot compares nothing else) and
//! a placement word (a discard, which is what the shard, the descent and the tag read). The
//! placement word has no inverse. The attacker can still evaluate it -- it is public -- but they
//! can no longer solve it, so keys agreeing on `j` windows have to be **found**, at `2^(j*gbits)`
//! trials each.
//!
//! Run: `cargo run --release --example hashdos`

use apeiron::map::Salt;
use apeiron::table::{shard_of, CountTable};
use std::time::Instant;

/// `table::SHARDS`, which is private; the count is fixed at 256.
const SHARDS_TOTAL: usize = 256;

/// The shard a placement word lands in, without going back through the address.
fn home_bits(place: u64) -> usize {
    (place as usize) & 0xFF
}

/// The mask for the first `w` windows of the descent, each `gbits` wide, taken from bit 57 down.
fn mask_for(w: u32, gbits: u32) -> u64 {
    if w == 0 {
        0
    } else {
        (1u64 << (w * gbits)) - 1
    }
}

fn main() {
    let salt = Salt::default();
    let n = 200_000usize;

    // Honest traffic: a clustered spatial scene, the workload this table is for.
    let mut r = 0x243F_6A88_85A3_08D3u64;
    let mut next = || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    let honest: Vec<u64> = (0..n).map(|_| next()).collect();

    // Aiming a *shard* is the wrong attack and always was: a shard's region is sized to the keys
    // that land in it, so crowding one gets a bigger region and finishes faster than honest
    // traffic. That is why the old version of this example measured the attack as a speed-up. The
    // thing worth aiming is the descent.
    let one_shard: Vec<u64> = {
        let mut out = Vec::with_capacity(n);
        let mut k = 1u64;
        while out.len() < n {
            k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            if shard_of(salt, k) == 0 {
                out.push(k);
            }
        }
        out
    };

    let build = |ks: &[u64]| {
        let t = Instant::now();
        let table = CountTable::build(ks, 20);
        let hits = ks.iter().filter(|&&k| table.get(k) > 0).count();
        assert_eq!(hits, ks.len(), "the table lost keys");
        t.elapsed().as_secs_f64() * 1e3
    };

    let base = build(&honest);
    println!("  {n} keys, built then read back");
    println!("  {:<26} {:>10} {:>9}", "keys", "ms", "vs honest");
    println!("  {:<26} {:>10.2} {:>8.2}x", "honest", base, 1.0);
    println!(
        "  {:<26} {:>10.2} {:>8.2}x   <- the old attack, and not one",
        "all in one shard",
        build(&one_shard),
        build(&one_shard) / base
    );

    // The real attack, and it has to aim at **both** ends of the placement word. A `CountTable`
    // shards on its bottom eight bits and descends from bit 57, so crowding one region takes the
    // shard *and* the leading windows: `8 + j*gbits` bits of a word that cannot be inverted, hence
    // `2^(8 + j*gbits)` trials per key. Smaller `n` here than above, because the point is where the
    // wall is and the wall does not move with the key count.
    let n = 2_000usize;
    let honest: Vec<u64> = (0..n).map(|_| next()).collect();
    let base = build(&honest);
    let gbits = 8u32; // a 2,000-key region's window width
    println!("\n  {n} keys, aiming the shard *and* the descent");
    println!(
        "  {:<26} {:>14} {:>10} {:>9}",
        "windows matched", "trials/key", "search s", "vs honest"
    );
    println!("  {:<26} {:>14} {:>10} {:>8.2}x", "honest", "-", "-", 1.0);
    for w in 0..=2u32 {
        let bits = 8 + w * gbits;
        let t = Instant::now();
        let (mut aimed, mut k, mut tried) = (Vec::with_capacity(n), 1u64, 0u64);
        // Ten million trials per key is far past what an attacker would spend and still not enough
        // for the third row -- which is the result, not a limitation of the harness.
        let budget = 10_000_000u64 * n as u64;
        while aimed.len() < n {
            k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            tried += 1;
            let place = salt.spread(k).place;
            // The shard is the bottom eight bits; the descent's first `w` windows are the
            // `w * gbits` bits below 57. Both must hit, and neither can be solved for.
            let shard_hit = home_bits(place) == 0;
            let windows_hit = w == 0 || (place >> (57 - w * gbits)) & mask_for(w, gbits) == 0;
            if shard_hit && windows_hit {
                aimed.push(k);
            }
            if tried > budget {
                break;
            }
        }
        if aimed.len() < n {
            println!(
                "  {:<26} {:>14} {:>10} {:>9}",
                format!("shard + {w} ({bits} bits)"),
                format!("> {}", budget / n as u64),
                "gave up",
                "unaffordable"
            );
            continue;
        }
        let secs = t.elapsed().as_secs_f64();
        let ms = build(&aimed);
        println!(
            "  {:<26} {:>14} {:>10.2} {:>8.2}x",
            format!("shard + {w} ({bits} bits)"),
            tried / n as u64,
            secs,
            ms / base
        );
    }

    // And the pepper, which does not make the attack dearer -- it makes the filter impossible.
    // Measured structurally rather than by the clock: how many of the 256 shards the aimed keys
    // reach. A timing comparison at two thousand keys is microseconds and says nothing.
    let aimed_public: Vec<u64> = {
        let (mut out, mut k) = (Vec::with_capacity(n), 1u64);
        while out.len() < n {
            k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let place = salt.spread(k).place;
            if home_bits(place) == 0 && (place >> (57 - gbits)) & mask_for(1, gbits) == 0 {
                out.push(k);
            }
        }
        out
    };
    let secret = Salt::new(0x5EED_1234_ABCD_9876);
    let spread_over = |s: Salt, ks: &[u64]| {
        ks.iter()
            .map(|&k| shard_of(s, k))
            .collect::<std::collections::HashSet<_>>()
            .len()
    };
    println!("\n  shards reached, of {SHARDS_TOTAL}");
    println!("  {:<34} {:>8}", "honest keys", spread_over(salt, &honest));
    println!(
        "  {:<34} {:>8}   <- the attack, against the salt it was aimed at",
        "aimed keys, public salt",
        spread_over(salt, &aimed_public)
    );
    println!(
        "  {:<34} {:>8}   <- the same keys, against a pepper they could not evaluate",
        "aimed keys, peppered table",
        spread_over(secret, &aimed_public)
    );

    println!("\n  Exponential cost, linear damage -- which is what `table::group_of` always claimed");
    println!("  and could not deliver while the address was one invertible word. It is not a");
    println!("  substitute for a pepper: `CountTable::with_pepper` makes the filter itself");
    println!("  impossible, because an attacker who cannot evaluate the table's placement word");
    println!("  cannot keep the candidates that land where they want.");
}
