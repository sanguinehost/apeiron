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

/// The attack apeiron **is** exposed to, and what the descent does about it.
///
/// The construction above fails against `mix2`. What does not fail is aiming at the *index*: the
/// address function is public and invertible, so an attacker can compute keys whose addresses agree
/// on the top bits and land them all in one group. Every open-addressed table with a fixed hash has
/// this, including `hashbrown` with `fxhash`. What differs is what a chain then costs.
///
/// A linear probe walks one slot at a time from the bucket, so `m` aimed keys cost `O(m^2)` and the
/// attacker needs only the index's bits. The window descent's second step reads the *next* `gbits`
/// of the address, which the attacker also has to match to keep the chain going -- so a chain of
/// `j` steps costs `j * gbits` bits of control, and an unbounded one costs all 64, at which point
/// the two addresses are equal and the fold, not the table, is what was broken.
fn aimed() {
    use apeiron::coord::unspread;
    use apeiron::map::Words;
    use hashbrown::HashMap as HbMap;
    use std::hash::BuildHasherDefault;

    let n = 200_000usize;
    // Keys whose addresses share their top 20 bits: one group of a table of any size up to a
    // million. `unspread` is public and exact, so this is arithmetic, not search.
    let aimed: Vec<u64> = (0..n as u64).map(|i| unspread(0xABCD_E000_0000_0000 | i)).collect();
    let honest: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();

    let run = |pepper: u64| {
        move |ks: &[u64]| {
            let t = Instant::now();
            let mut m: Words<u64, u64> = Words::with_pepper(pepper);
            for &k in ks {
                m.insert(k, k);
            }
            let hits = ks.iter().filter(|&&k| m.get(&k).is_some()).count();
            assert_eq!(hits, ks.len());
            t.elapsed().as_secs_f64() * 1e3
        }
    };
    let run_fx = |ks: &[u64]| {
        type Fx = BuildHasherDefault<rustc_hash::FxHasher>;
        let t = Instant::now();
        let mut m: HbMap<u64, u64, Fx> = HbMap::default();
        for &k in ks {
            m.insert(k, k);
        }
        let hits = ks.iter().filter(|&&k| m.get(&k).is_some()).count();
        assert_eq!(hits, ks.len());
        t.elapsed().as_secs_f64() * 1e3
    };
    let run_fold = |ks: &[u64]| {
        let t = Instant::now();
        let mut m: HbMap<u64, u64, foldhash::fast::RandomState> = HbMap::default();
        for &k in ks {
            m.insert(k, k);
        }
        let hits = ks.iter().filter(|&&k| m.get(&k).is_some()).count();
        assert_eq!(hits, ks.len());
        t.elapsed().as_secs_f64() * 1e3
    };

    let plain = run(0);
    let peppered = run(0x5EED_1234_ABCD_9876);
    println!("\n  {n} keys, built then read back (ms)");
    println!("  {:<26} {:>10} {:>10} {:>10}", "", "honest", "aimed", "ratio");
    for (name, f) in [
        ("apeiron", &plain as &dyn Fn(&[u64]) -> f64),
        ("apeiron, peppered", &peppered),
        ("hashbrown + fxhash", &run_fx),
        ("hashbrown + foldhash", &run_fold),
    ] {
        let (h, a) = (f(&honest), f(&aimed));
        println!("  {name:<26} {h:>10.2} {a:>10.2} {:>9.1}x", a / h);
    }
    println!("\n  The unpeppered row is the correction. `table.rs` says a chain of j steps costs the");
    println!("  attacker j*gbits bits of address -- true, and not the same as expensive. Every");
    println!("  window is a slice of ONE public, invertible word, so matching all of them costs one");
    println!("  call to `unspread`. The descent bounds a chain in bits of address; it cannot make");
    println!("  those bits dear. **Only a secret can**, which is why SipHash is keyed and why the");
    println!("  peppered row is flat. fxhash is unmoved here only because this attack was aimed at");
    println!("  apeiron's address function; the same construction against its multiplier moves it.");
}
