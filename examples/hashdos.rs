//! Board APEIRON-HASHDOS-CAP-COUNT-W1-T01.
//! HashDOS instrument: what the capped count-face probe costs, what it saves against its own
//! pre-cap behaviour, and what an uncapped hashbrown-layout table costs on the same class of craft.
//!
//! Not a gate, and not evidence about hashbrown. The crate carries no `hashbrown` dependency, so
//! `hb` is a local reference table with hashbrown's layout: 16-lane groups, 7-bit control tags in
//! the hash's high bits, linear stepping to the next group, first empty lane, 7/8-load resize, and
//! a fixed multiply-xor spread. It is an instrument for "an uncapped open-addressed table with a
//! group probe", not a measurement of that crate.
//!
//! Populations, each crafted for the structure it is measured on:
//!
//! - `ap flood`: `FreqMap::with_capacity` sized so the table has `--groups` groups; every one of
//!   the victim's first `--groups` probe groups is filled (16 keys each, the shape
//!   `tests/hashdos_count.rs::crafted_group_flood` uses), then the victim is inserted. The capped
//!   probe stops at `PROBE_CAP` groups and consults the overflow tree; the pre-cap build walks the
//!   saturated run to the first empty group.
//! - `hb flood`: the same shape against the reference table's own hash — `--groups` consecutive
//!   probe groups on the victim's path filled, then the victim.
//! - `benign`: SplitMix64 keys at 50 % load for `ap`, the same key set for `hb`.
//!
//! The pre-cap arm is this binary built against a count face whose `probe_limit` returns
//! `u32::MAX`; `--tag` labels the binary so two builds can be interleaved and paired.
//!
//! One row per measurement:
//!
//! ```text
//! <tag> <arm> <metric> <ns_per_op> digest=<d> probe=<p> overflow=<o>
//! ```
//!
//! `digest` is the sum of the answers taken in one untimed pass, so rows with equal digests
//! measured the same work; `probe` is the per-lookup primary-group count (`FreqMap::longest_probe`
//! / the reference's own walk) and `overflow` the overflow population (`.` where not applicable).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    reason = "measurement harness: `usize`/`u64` casts are population sizes, group widths and \
              digests, and the short names are the probe's own (`g`, `j`, `k`)"
)]

use std::time::Instant;

use apeiron::map::freq::{FreqMap, PROBE_CAP};

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;
const GROUP: usize = 16;

/// `PHI⁻¹ mod 2⁶⁴` (Newton iteration; `PHI` is odd ⇒ a unit), so a crafted product can be named
/// exactly and inverted back into a key.
fn phi_inv() -> u64 {
    let mut x = 1u64;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(PHI.wrapping_mul(x)));
    }
    assert_eq!(PHI.wrapping_mul(x), 1, "PHI^-1");
    x
}

/// `FreqLine::home_group`, replicated.
fn home_group(bits: u32, p: u64) -> usize {
    let gbits = bits - 4;
    let gmask = (1usize << gbits) - 1;
    (p >> (64 - gbits)) as usize & gmask
}

/// `FreqLine::descend_group`, replicated.
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

/// A count-face flood: for each of `groups` probe groups on `victim`'s path, 16 keys whose home is
/// that group, in probe order. Every targeted group ends exactly full.
fn count_flood(capacity: usize, groups: usize, victim: u64) -> Vec<u64> {
    let bits = capacity.trailing_zeros();
    let phiinv = phi_inv();
    let p = victim.wrapping_mul(PHI);
    let mut keys = Vec::with_capacity(groups * GROUP);
    for j in 0..groups as u32 {
        let g = if j == 0 {
            home_group(bits, p)
        } else {
            descend_group(bits, p, j)
        };
        for _ in 0..GROUP {
            // Product with the top `gbits` bits = `g` (so its home group is `g`); the low bits make
            // each blocker's frequency distinct.
            let low = (keys.len() as u64) + 1;
            let product = ((g as u64) << (64 - (bits - 4))) | low;
            let k = product.wrapping_mul(phiinv);
            if k != u64::MAX && k != u64::MAX - 1 {
                keys.push(k);
            }
        }
    }
    keys
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The hashbrown-layout reference: control tags in the hash's top 7 bits (`| 0x80`), the group
/// index in the next bits, linear stepping to the next group, first empty lane, resize at 7/8.
struct HbStyle {
    tags: Vec<u8>,
    keys: Vec<u64>,
    vals: Vec<i64>,
    groups: usize,
    len: usize,
}

impl HbStyle {
    fn new(groups: usize) -> Self {
        let groups = groups.max(2).next_power_of_two();
        Self {
            tags: vec![0; groups * GROUP],
            keys: vec![0; groups * GROUP],
            vals: vec![0; groups * GROUP],
            groups,
            len: 0,
        }
    }

    #[inline]
    fn spread(k: u64) -> u64 {
        k.wrapping_mul(0x51_7c_c1_b7_27_22_0a_95) ^ (k >> 32)
    }

    #[inline]
    fn tag(h: u64) -> u8 {
        (h >> 57) as u8 | 0x80
    }

    #[inline]
    fn home(h: u64, groups: usize) -> usize {
        ((h >> 7) as usize) & (groups - 1)
    }

    fn get(&self, k: u64) -> Option<i64> {
        self.get_walk(k).0
    }

    /// `(value, groups scrutinised)`.
    fn get_walk(&self, k: u64) -> (Option<i64>, usize) {
        let h = Self::spread(k);
        let tag = Self::tag(h);
        let mut g = Self::home(h, self.groups);
        for walked in 1..=self.groups {
            let base = g * GROUP;
            let mut empty = false;
            for lane in 0..GROUP {
                let t = self.tags[base + lane];
                if t == 0 {
                    empty = true;
                } else if t == tag && self.keys[base + lane] == k {
                    return (Some(self.vals[base + lane]), walked);
                }
            }
            if empty {
                return (None, walked);
            }
            g = (g + 1) & (self.groups - 1);
        }
        (None, self.groups)
    }

    fn insert(&mut self, k: u64, v: i64) -> usize {
        if (self.len + 1) * 8 >= self.groups * GROUP * 7 {
            self.grow();
        }
        let h = Self::spread(k);
        let tag = Self::tag(h);
        let mut g = Self::home(h, self.groups);
        for walked in 1..=self.groups {
            let base = g * GROUP;
            for lane in 0..GROUP {
                if self.tags[base + lane] == 0 {
                    self.tags[base + lane] = tag;
                    self.keys[base + lane] = k;
                    self.vals[base + lane] = v;
                    self.len += 1;
                    return walked;
                }
            }
            g = (g + 1) & (self.groups - 1);
        }
        panic!("reference table full");
    }

    fn grow(&mut self) {
        let old_tags = std::mem::take(&mut self.tags);
        let old_keys = std::mem::take(&mut self.keys);
        let old_vals = std::mem::take(&mut self.vals);
        self.groups *= 2;
        self.tags = vec![0; self.groups * GROUP];
        self.keys = vec![0; self.groups * GROUP];
        self.vals = vec![0; self.groups * GROUP];
        self.len = 0;
        for i in 0..old_tags.len() {
            if old_tags[i] != 0 {
                self.insert(old_keys[i], old_vals[i]);
            }
        }
    }
}

/// Brute-force `n` distinct keys whose reference home group is `g`, for the reference's own flood.
fn hb_flood_to(groups: usize, g: usize, n: usize, seed: u64) -> Vec<u64> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let k = splitmix(&mut state);
        if HbStyle::home(HbStyle::spread(k), groups) == g {
            out.push(k);
        }
    }
    out
}

/// ns per call over `calls` repetitions, and the answer's sum taken once (untimed).
fn bench<F: FnMut() -> Option<i64>>(calls: usize, mut f: F) -> (f64, i64) {
    let digest: i64 = f().unwrap_or(0);
    let t = Instant::now();
    let mut sink = 0i64;
    for _ in 0..calls {
        sink = sink.wrapping_add(f().unwrap_or(0));
    }
    let ns = t.elapsed().as_nanos() as f64 / calls as f64;
    std::hint::black_box(sink);
    (ns, digest)
}

fn emit(tag: &str, arm: &str, metric: &str, round: usize, r: (f64, i64), probe: usize, ov: &str) {
    println!(
        "{tag} {arm} {metric} r{round} {:.2} digest={} probe={probe} overflow={ov}",
        r.0, r.1
    );
}

fn main() {
    let mut tag = String::from("unnamed");
    let mut groups = 512usize;
    let mut benign = 100_000usize;
    let mut calls = 20_000usize;
    let mut rounds = 5usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--tag" => tag.clone_from(&args[i + 1]),
            "--groups" => groups = args[i + 1].parse().expect("--groups usize"),
            "--benign" => benign = args[i + 1].parse().expect("--benign usize"),
            "--calls" => calls = args[i + 1].parse().expect("--calls usize"),
            "--rounds" => rounds = args[i + 1].parse().expect("--rounds usize"),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    let victim: u64 = 0x0123_4567_89AB_CDEF;
    // An absent key on the victim's own group path: its product differs from the victim's in the
    // low two bits only, and the probe reads only the consumed high windows (the first 16 groups
    // depend on the top 62 bits and up), so it examines the same saturated groups and then misses.
    let absent_ap: u64 = victim
        .wrapping_mul(PHI)
        .wrapping_sub(2)
        .wrapping_mul(phi_inv());

    // ---- count face: its own crafted flood, built once, untimed -------------------------
    // `with_capacity` allocates at ~50 % load, so this is about `2 * groups * GROUP` slots: the
    // saturated run of `groups` groups fills half the table, leaving as many groups empty. That
    // spare room is what makes the insertion stay under the grow threshold (a grow would re-place
    // every crafted key) and what lets the pre-cap build terminate at the first empty group.
    let mut m: FreqMap<u64, i64> = FreqMap::with_capacity(groups * GROUP);
    let capacity = m.capacity();
    let ckeys = count_flood(capacity, groups, victim);
    let t0 = Instant::now();
    for &k in &ckeys {
        m.add(&k, 1);
    }
    m.add(&victim, 42);
    let ap_insert = t0.elapsed().as_nanos() as f64 / ckeys.len() as f64;
    let ap_probe = m.longest_probe();
    let ap_overflow = m.overflow_len();

    // ---- reference table: its own crafted flood -----------------------------------------
    let mut t = HbStyle::new(groups.next_power_of_two() * 8);
    let h_groups = t.groups;
    let h_victim = u64::from(victim as u32) | 0x0123_4567_0000_0000;
    let vhome = HbStyle::home(HbStyle::spread(h_victim), h_groups);
    let mut hkeys = Vec::new();
    for step in 0..groups {
        let g = (vhome + step) & (h_groups - 1);
        hkeys.extend(hb_flood_to(h_groups, g, GROUP, 0x5EED_0000 + step as u64));
    }
    let t1 = Instant::now();
    for &k in &hkeys {
        t.insert(k, 1);
    }
    t.insert(h_victim, 42);
    let hb_insert = t1.elapsed().as_nanos() as f64 / hkeys.len() as f64;
    let (_, hb_probe) = t.get_walk(h_victim);
    // An absent key homed at the same reference group: it walks the whole saturated run.
    let hb_absent = hb_flood_to(h_groups, vhome, 1, 0xDEAD_BEEF)[0];
    let (_, hb_absent_probe) = t.get_walk(hb_absent);

    // ---- benign population: the cap's steady-state cost ---------------------------------
    let mut state = 0x5EED_u64;
    let bkeys: Vec<u64> = (0..benign).map(|_| splitmix(&mut state)).collect();
    let mut bm: FreqMap<u64, i64> = FreqMap::with_capacity(benign * 2);
    for &k in &bkeys {
        bm.add(&k, 1);
    }
    let bm_probe = bm.longest_probe();
    let mut bt = HbStyle::new(benign * 2 / GROUP);
    for &k in &bkeys {
        bt.insert(k, 1);
    }

    println!(
        "# hashdos instrument: tag={tag} PROBE_CAP={PROBE_CAP} capacity={capacity} groups={groups} benign={benign} calls={calls} rounds={rounds}"
    );

    for round in 1..=rounds {
        // Alternate which arm goes first, so a monotone drift in the box does not load one arm.
        let ap_first = round % 2 == 1;
        let ap = |round: usize| {
            emit(
                &tag,
                "ap",
                "flood_victim_get",
                round,
                bench(calls, || m.get(&victim).copied()),
                ap_probe,
                &ap_overflow.to_string(),
            );
            emit(
                &tag,
                "ap",
                "flood_absent_get",
                round,
                bench(calls, || m.get(&absent_ap).copied()),
                ap_probe,
                &ap_overflow.to_string(),
            );
            let mut it = bkeys.iter();
            let d = bench(calls, || it.next().and_then(|&k| bm.get(&k).copied()));
            emit(&tag, "ap", "benign_get", round, d, bm_probe, ".");
            emit(
                &tag,
                "ap",
                "flood_insert",
                round,
                (ap_insert, ckeys.len() as i64 + 1),
                ap_probe,
                &ap_overflow.to_string(),
            );
        };
        let hb = |round: usize| {
            emit(
                &tag,
                "hb",
                "flood_victim_get",
                round,
                bench(calls, || t.get(h_victim)),
                hb_probe,
                ".",
            );
            emit(
                &tag,
                "hb",
                "flood_absent_get",
                round,
                bench(calls, || t.get(hb_absent)),
                hb_absent_probe,
                ".",
            );
            let mut it = bkeys.iter();
            let d = bench(calls, || it.next().and_then(|&k| bt.get(k)));
            emit(&tag, "hb", "benign_get", round, d, 0, ".");
            emit(
                &tag,
                "hb",
                "flood_insert",
                round,
                (hb_insert, hkeys.len() as i64 + 1),
                hb_probe,
                ".",
            );
        };
        if ap_first {
            ap(round);
            hb(round);
        } else {
            hb(round);
            ap(round);
        }
    }
}
