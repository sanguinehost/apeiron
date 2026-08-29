//! What a collision chain costs an attacker, and what a pepper adds on top of that.
//!
//! A *fiber* is a set of keys the placement word cannot tell apart. The descent reads successive
//! windows of that word, so a chain is a fiber of the first `j` windows, and the question this
//! program answers is what one costs to build and what it buys once built.
//!
//! This used to open by constructing a fiber of `hashtrinity`'s fold from published constants --
//! 4,096 keys, one address, of which its table kept 1 and answered 4,095 lookups with another
//! key's value. That dependency is gone; the finding is kept as prose in `apeiron::fold`, where
//! the guard against it lives.
//!
//! Run: `cargo run --release --example the_fiber`

use std::time::Instant;

fn main() {
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
