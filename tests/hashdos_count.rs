//! Red team for the count face's capped group probe and deterministic overflow, plus the
//! reserved-frequency identity fix.
//!
//! `FreqLine` (behind `FreqMap::add` and `count_par`) examines at most [`PROBE_CAP`] primary groups
//! per lookup, insert or update; a key whose capped prefix is fully occupied routes to a
//! deterministic AVL overflow tree ([`Overflow`], the same store the map face uses). Separately, the
//! reserved frequencies `EMPTY`/`TOMB` (`2^64-1`, `2^64-2`) are held out of band, so every `u64` key
//! — including `2^64-3`, `2^64-2`, `2^64-1` and `0` — has an exact, distinct count.
//!
//! The negative controls: restoring the old `min(k, 2^64-3)` fold reddens
//! [`reserved_frequencies_count_exactly`]; removing the probe cap (`probe_limit` returning a huge
//! bound) reddens [`crafted_group_flood_stays_capped`].

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::many_single_char_names
)]

use apeiron::map::freq::{FreqMap, PROBE_CAP};
use std::collections::HashMap;

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

/// `PHI⁻¹ mod 2⁶⁴` (Newton iteration; `PHI` is odd ⇒ a unit). Lets a test name the product
/// `k·PHI` exactly, so it can aim a key at a chosen home group.
fn phi_inv() -> u64 {
    let mut x = 1u64;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(PHI.wrapping_mul(x)));
    }
    assert_eq!(PHI.wrapping_mul(x), 1, "PHI⁻¹");
    x
}

/// The home group of a frequency's product, replicated from `FreqLine::home_group`.
fn home_group(bits: u32, p: u64) -> usize {
    let gbits = bits - 4;
    let gmask = (1usize << gbits) - 1;
    (p >> (64 - gbits)) as usize & gmask
}

/// The group at descent step `j`, replicated from `FreqLine::descend_group`.
fn descend_group(bits: u32, p: u64, j: u32) -> usize {
    let gbits = bits - 4;
    let gmask = (1usize << gbits) - 1;
    if gbits == 0 {
        return 0;
    }
    let used = gbits * (j + 1);
    if used <= 64 {
        (p >> (64 - used)) as usize & gmask
    } else {
        let full = 64 / gbits;
        ((p >> (64 - gbits * full)) as usize + (j - full) as usize + 1) & gmask
    }
}

/// The slot count the crafted flood pins: `FreqMap::with_capacity(256)` is 512 slots ⇒ 32 groups,
/// so the victim's first `PROBE_CAP` groups are distinguishable from the groups the (negative
/// control's) uncapped descent reaches past the cap.
const FLOOD_CAP: usize = 256;

/// A crafted flood aimed at one victim's home-group path: for each of the victim's first
/// `PROBE_CAP` group addresses, 16 keys whose home is that group. Every target group ends full, so
/// the victim is blocked past the cap (routed to the overflow); without the cap its probe would
/// continue past `PROBE_CAP`. Returns `(blocker keys, victim)`.
fn crafted_group_flood() -> (Vec<u64>, u64) {
    let n = FreqMap::<u64, i64>::with_capacity(FLOOD_CAP).capacity();
    assert_eq!(n, 512, "FreqMap::with_capacity(256) is 512 slots");
    let bits = n.trailing_zeros();
    let phiinv = phi_inv();
    let victim: u64 = 0x0123_4567_89AB_CDEF;
    let p_victim = victim.wrapping_mul(PHI);

    let mut flood: Vec<u64> = Vec::new();
    for j in 0..PROBE_CAP {
        let g = if j == 0 {
            home_group(bits, p_victim)
        } else {
            descend_group(bits, p_victim, j)
        };
        for _ in 0..16u64 {
            // Product with the top `gbits = 5` bits = `g` (home group `g`); the low bits make each
            // blocker's frequency distinct.
            let low = (flood.len() as u64) + 1;
            let product = ((g as u64) << 59) | low;
            let k = product.wrapping_mul(phiinv);
            assert_ne!(k, u64::MAX, "a blocker must not own a reserved frequency");
            assert_ne!(
                k,
                u64::MAX - 1,
                "a blocker must not own a reserved frequency"
            );
            flood.push(k);
        }
    }
    assert_eq!(
        flood.len(),
        PROBE_CAP as usize * 16,
        "one group of lanes per step"
    );
    (flood, victim)
}

/// The reserved frequencies (`2^64-3`, `2^64-2`, `2^64-1`) and `0` are four distinct keys, each
/// counted exactly; this is the identity fix — the old `min(k, 2^64-3)` fold merged the top three.
#[test]
fn reserved_frequencies_count_exactly() {
    let keys = [u64::MAX - 2, u64::MAX - 1, u64::MAX, 0];
    let mut m: FreqMap<u64, i64> = FreqMap::with_capacity(8);
    for &k in &keys {
        m.add(&k, 1);
    }
    m.add(&u64::MAX, 4);
    m.add(&(u64::MAX - 1), 2);

    assert_eq!(m.len(), 4, "four distinct keys ⇒ four entries");
    assert_eq!(m.get(&u64::MAX), Some(&5), "EMPTY key folded away");
    assert_eq!(m.get(&(u64::MAX - 1)), Some(&3), "TOMB key folded away");
    assert_eq!(m.get(&(u64::MAX - 2)), Some(&1), "TOMB−1 key folded away");
    assert_eq!(m.get(&0), Some(&1));

    // Force a grow with enough other keys: the reserved counts survive it.
    for k in 1..200u64 {
        m.add(&k, 1);
    }
    assert!(m.capacity() > 16, "expected a grow");
    assert_eq!(m.get(&u64::MAX), Some(&5), "reserved key lost across grow");
    assert_eq!(m.get(&(u64::MAX - 1)), Some(&3));
    assert_eq!(m.get(&(u64::MAX - 2)), Some(&1));
    assert_eq!(m.get(&0), Some(&1));
    assert_eq!(m.len(), 203, "4 reserved/zero + 199 fillers");

    // count_par routes reserved frequencies through the same side store.
    let stream = [u64::MAX, u64::MAX, u64::MAX - 1, u64::MAX - 2, 0];
    let maps = FreqMap::<u64, i64>::count_par(&stream);
    let got = |k: u64| -> i64 { maps.iter().filter_map(|m| m.get(&k).copied()).sum() };
    assert_eq!(got(u64::MAX), 2);
    assert_eq!(got(u64::MAX - 1), 1);
    assert_eq!(got(u64::MAX - 2), 1);
    assert_eq!(got(0), 1);
}

/// The crafted flood resolves every key exactly, and every operation (insert via `add`, lookup via
/// `get`) examines at most [`PROBE_CAP`] primary groups. Restoring the uncapped probe reddens this.
#[test]
fn crafted_group_flood_stays_capped() {
    let (flood, victim) = crafted_group_flood();
    let mut m: FreqMap<u64, i64> = FreqMap::with_capacity(FLOOD_CAP);
    assert_eq!(m.capacity(), 512);

    for &k in &flood {
        m.add(&k, 1);
    }
    m.add(&victim, 42);

    assert_eq!(m.len(), flood.len() + 1, "every distinct key stored");
    for &k in &flood {
        assert_eq!(m.get(&k), Some(&1), "flood key {k:#x} lost");
        assert!(
            m.primary_examinations(&k) <= PROBE_CAP,
            "flood key {k:#x}: {} group probes > PROBE_CAP {PROBE_CAP}",
            m.primary_examinations(&k)
        );
    }
    // The victim's whole capped prefix was full ⇒ it must be found, capped, from the overflow.
    // The bound is asserted FIRST: the negative control that removes the cap must redden here
    // (the uncapped probe reaches past `PROBE_CAP`), not on a downstream consequence.
    assert_eq!(m.get(&victim), Some(&42), "blocked victim lost");
    assert!(
        m.primary_examinations(&victim) <= PROBE_CAP,
        "victim: {} group probes > PROBE_CAP {PROBE_CAP}",
        m.primary_examinations(&victim)
    );
    assert!(
        m.longest_probe() <= PROBE_CAP as usize,
        "primary probe bound: {} > PROBE_CAP {PROBE_CAP}",
        m.longest_probe()
    );
    assert!(
        m.overflow_len() >= 1,
        "an exhausted capped prefix must route the victim to the overflow"
    );
}

/// `count_par` agrees with a `HashMap` oracle on the flood, reserved frequencies included.
#[test]
fn count_par_matches_oracle_on_the_group_flood() {
    let (flood, victim) = crafted_group_flood();
    let mut keys: Vec<u64> = flood.clone();
    keys.extend_from_slice(&flood); // duplicate every blocker ⇒ counts of 2
    keys.push(victim);
    keys.push(victim);
    keys.push(u64::MAX); // reserved frequencies flow through the side store
    keys.push(u64::MAX - 1);
    keys.push(u64::MAX - 2);
    keys.push(0);

    let mut oracle: HashMap<u64, i64> = HashMap::new();
    for &k in &keys {
        *oracle.entry(k).or_insert(0) += 1;
    }

    let maps = FreqMap::<u64, i64>::count_par(&keys);
    assert_eq!(
        maps.iter().map(FreqMap::len).sum::<usize>(),
        oracle.len(),
        "distinct keys"
    );
    let total: i64 = maps.iter().flat_map(FreqMap::values).sum();
    assert_eq!(total, keys.len() as i64, "counts sum to n");
    for (k, &c) in &oracle {
        let got: i64 = maps.iter().filter_map(|m| m.get(k).copied()).sum();
        assert_eq!(got, c, "key {k:#x}");
    }
}
