#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::unnecessary_wraps,
    clippy::items_after_statements,
    reason = "a measurement harness: fixed-width key and slot arithmetic is the subject, and the \
              report formats ratios as f64 at the point of printing"
)]
//! Cost of the canonical `apeiron::Table` and pinned `hashtrinity::Table` on the three key
//! populations the Slang port was measured on.
//!
//! The port's numbers (192 keys folded to one home slot, 18 528 insert probes, 96.5 lookup probes
//! per key) were measured on `song/src/table/table.slang`, a plain open-addressed slot array with
//! linear probing. This harness re-runs the same three populations against the *canonical* table
//! and against the pinned historical `hashtrinity::Table`, and reports each implementation's own
//! cost.
//!
//! Populations, all `n` distinct `u64` keys under one shared identity home (`place = key`,
//! `coord = key`), so any difference in realised cost is a difference in probe geometry rather than
//! in the home function. `home` is a caller-supplied `Home`/`Coord`, so the home is held fixed and
//! the probe scheme is the variable.
//!
//! * `distinct` — the baseline: `key_j` spread over the whole word by a fixed odd multiply.
//! * `lowbyte` — one home shard: `key_j = j << 8`, so `place & 0xFF == 0` (one shard) while the
//!   fingerprint byte and the descent windows still separate.
//! * `adversarial` — one home *group*: `key_j = j << 16`, so `place & 0xFF == 0` (one shard),
//!   `(place >> 8) & 0x7f == 0` (one fingerprint) and `d & mask == 0` (one home group).
//!
//! The absent population of each family is the same shape one band above the present one
//! (`| 1 << 40`), so an absent key folds exactly like the present pile it probes for.
//!
//! Counters come from `apeiron::table::read_stats` (feature `read-stats`), which instruments
//! `spill_read` only — the insert locator `spill_locate` carries no counter, so the insert figure
//! is timed, not counted. `hashtrinity` is an external crate and is not instrumented; its slot
//! visits come from a replica of its published probe, transcribed from the committed object
//! `4cc14471ae24f3f204fb0b0e91736ef75c909358:hashtrinity/src/lib.rs`, and the replica is
//! *differentially controlled*: its predicted hit/miss for every present and absent key must equal
//! the real `hashtrinity::Table::get` result, or the run fails closed.
//!
//! Timing is a median over `SAMPLES` passes of `n` operations after `WARMUP` untimed passes.
//! Spatial cost is reported as the tables' slot counts. Nothing here is a `lake build` result; the
//! counts are `measured` for the canonical table's own counters and the wall clocks, and `model`
//! for the `hashtrinity` replica's slot-visit counts.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use apeiron::coord::locator::{ExactIdentity, Refusal};
use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue, WordFaceHome,
    H3_40,
};
use apeiron::table::{Home, Table, TableAddress};
use apeiron::Sigma;
use hashtrinity::Coord;

/// The compile-time group width under measurement: the `Table` default.
const N: usize = 16;
/// Keys per population.
const KEYS: usize = 192;
/// Untimed passes before sampling.
const WARMUP: u32 = 3;
/// Timed passes per cell.
const SAMPLES: u32 = 9;
/// The band separating an absent key from its present family.
const ABSENT_BAND: u64 = 1 << 40;
/// The odd multiplier that spreads the baseline population.
const SPREAD: u64 = 0x9E37_79B9_7F4A_7C15;

// ── The shared key and the shared identity home ────────────────────────────────────────────────

/// One key. The word *is* the home word: identity is the tag, and the placement is the tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    tag: u64,
}

/// The identity home: `place = key`, and no certified direct word, so every read and every insert
/// routes to a spill region. A certified word would put a `place < 2^32` population in the resident
/// flat array instead, and the spill probe would never run.
#[derive(Clone, Copy)]
struct IdentityHome;

impl Home<Key> for IdentityHome {
    fn home(&self, key: &Key) -> u64 {
        key.tag
    }

    fn address(&self, _sigma: Sigma, key: &Key) -> TableAddress {
        TableAddress {
            place: key.tag,
            direct_word: None,
        }
    }
}

impl Coord for Key {
    fn coord(&self) -> u64 {
        self.tag
    }
}

/// The exact identity is the whole 64-bit tag as eight big-endian bytes: the home word is the key,
/// so nothing may reduce, truncate or hash it.
impl ExactIdentity for Key {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        self.tag
    }

    fn reserve_retain(&self) -> Result<(), Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        self.tag
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        self.tag == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.tag.to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        retained.to_be_bytes()[index]
    }
}

impl FaceResidueCapability<Key> for IdentityHome {
    fn parabolic_dual(&self, key: &Key) -> ParabolicDual {
        WordFaceHome.parabolic_dual(&key.tag)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &Key) -> Residue {
        WordFaceHome.hyperbolic_scalar(profile, &key.tag)
    }
}

impl EllipticPacking<Key> for IdentityHome {
    fn elliptic_pack(&self, key: &Key) -> u64 {
        WordFaceHome.elliptic_pack(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<Key> {
        WordFaceHome.elliptic_unpack(word).map(|tag| Key { tag })
    }
}

impl FaceCoord<Key> for IdentityHome {
    type Faces = Sigma;
}

impl FaceHome<Key> for IdentityHome {}


fn replacement(new: i64, _old: i64) -> i64 {
    new
}

fn value_of(key: Key) -> i64 {
    key.tag.cast_signed()
}

// ── The three populations, and their absent bands ────────────────────────────────────────────────

/// One key population: `present` and an `absent` family of the same shape one band above.
struct Population {
    name: &'static str,
    present: Vec<Key>,
}

/// The baseline: distinct words spread over the full width by a fixed odd multiply.
fn distinct_population() -> Population {
    let present: Vec<Key> = (0..KEYS as u64)
        .map(|j| Key {
            tag: j.wrapping_mul(SPREAD).wrapping_add(0x1234_5678_9ABC_DEF0),
        })
        .collect();
    Population {
        name: "distinct",
        present,
    }
}

/// One home shard, separated groups: `place & 0xFF == 0`, fingerprint byte `j & 0x7f`.
fn lowbyte_population() -> Population {
    let present: Vec<Key> = (0..KEYS as u64)
        .map(|j| Key {
            tag: j << 8,
        })
        .collect();
    Population {
        name: "lowbyte",
        present,
    }
}

/// One home group: shard byte, fingerprint byte and descent window all zero, so every key homes
/// group 0 of shard 0.
fn adversarial_population() -> Population {
    let present: Vec<Key> = (0..KEYS as u64)
        .map(|j| Key {
            tag: j << 16,
        })
        .collect();
    Population {
        name: "adversarial",
        present,
    }
}

/// A keyed `Home`: the identity placement, then one fixed odd multiply. The home is caller-supplied
/// (`src/table.rs` `pub trait Home<K>`), so this is an implementation change, not a signature
/// change.
#[derive(Clone, Copy)]
struct KeyedHome;

impl Home<Key> for KeyedHome {
    fn home(&self, key: &Key) -> u64 {
        key.tag.wrapping_mul(SPREAD)
    }

    fn address(&self, _sigma: Sigma, key: &Key) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

impl FaceResidueCapability<Key> for KeyedHome {
    fn parabolic_dual(&self, key: &Key) -> ParabolicDual {
        WordFaceHome.parabolic_dual(&key.tag)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &Key) -> Residue {
        WordFaceHome.hyperbolic_scalar(profile, &key.tag)
    }
}

impl EllipticPacking<Key> for KeyedHome {
    fn elliptic_pack(&self, key: &Key) -> u64 {
        WordFaceHome.elliptic_pack(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<Key> {
        WordFaceHome.elliptic_unpack(word).map(|tag| Key { tag })
    }
}

impl FaceCoord<Key> for KeyedHome {
    type Faces = Sigma;
}

impl FaceHome<Key> for KeyedHome {}


/// A *mixing* keyed `Home`: a fold of the high half into the low half, then the odd multiply.
///
/// The multiply alone cannot break the adversarial fold. `shard` is `place & 0xFF` and the
/// fingerprint is `(place >> 8) & 0x7f`, so both read the low 16 bits of the place word; for
/// `key = j << 16` those bits are zero, and multiplying by an odd number is a bijection on
/// `Z/2^64` that preserves the 2-adic valuation, so `place & 0xFFFF == 0` survives. Mixing high
/// bits down is what moves the shard and the fingerprint.
#[derive(Clone, Copy)]
struct MixingHome;

impl Home<Key> for MixingHome {
    fn home(&self, key: &Key) -> u64 {
        key.tag ^ (key.tag >> 16)
    }

    fn address(&self, _sigma: Sigma, key: &Key) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

impl FaceResidueCapability<Key> for MixingHome {
    fn parabolic_dual(&self, key: &Key) -> ParabolicDual {
        WordFaceHome.parabolic_dual(&key.tag)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &Key) -> Residue {
        WordFaceHome.hyperbolic_scalar(profile, &key.tag)
    }
}

impl EllipticPacking<Key> for MixingHome {
    fn elliptic_pack(&self, key: &Key) -> u64 {
        WordFaceHome.elliptic_pack(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<Key> {
        WordFaceHome.elliptic_unpack(word).map(|tag| Key { tag })
    }
}

impl FaceCoord<Key> for MixingHome {
    type Faces = Sigma;
}

impl FaceHome<Key> for MixingHome {}

// ── The realised home distribution, read off the two home functions ──────────────────────────────

/// How many distinct homes a key set realises, and the largest single home's multiplicity, under
/// one home word. `places` are the home words; `homes` maps each to the table's own extraction.
fn home_collision_peak(places: &[u64], homes: impl Fn(u64) -> usize) -> (usize, usize) {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for place in places {
        *counts.entry(homes(*place)).or_insert(0) += 1;
    }
    let peak = counts.values().copied().max().unwrap_or(0);
    (counts.len(), peak)
}

/// The realised home-group distribution over the *whole* probe order, not just window 0: a key
/// that homes group 0 at window 0 and group 30 at window 5 is not a single-home fold. Transcribed
/// from `src/table/state.rs` `group`, so this is arithmetic on the source.
fn apeiron_visited_groups(place: u64, mask: usize) -> Vec<usize> {
    (0..WINDOWS)
        .map(|step| apeiron_group(place, step, mask))
        .chain((WINDOWS..WINDOWS + 4).map(|step| apeiron_group(place, step, mask)))
        .collect()
}

/// `shard`, `tag` and the descent windows, transcribed from `src/table/state.rs`
/// (`shard`, `tag`, `group`) so the realised group spread is arithmetic on the source, not an
/// estimate. `DESCENT_MASK = (1 << 49) - 1`, `WINDOW_MASK = 0x7f`, `WINDOWS = 7`.
const DESCENT_MASK: u64 = (1 << 49) - 1;
const WINDOW_MASK: u64 = 0x7f;
const WINDOWS: usize = 7;

fn apeiron_shard(place: u64) -> usize {
    (place as usize) & 0xff
}

fn apeiron_tag(place: u64) -> u8 {
    0x80 | (((place >> 8) as u8) & 0x7f)
}

fn apeiron_group(place: u64, step: usize, mask: usize) -> usize {
    let d = (place >> 8) & DESCENT_MASK;
    if step < WINDOWS {
        (((d >> (49 - 7 * (step + 1))) & WINDOW_MASK) as usize) & mask
    } else {
        ((d as usize) & mask).wrapping_add(step - WINDOWS + 1) & mask
    }
}

/// The `hashtrinity` home: the key indexes the array directly when it fits, otherwise the top
/// `bits` of `coord * PHI`. Transcribed from the committed `lib.rs:565`.
fn ht_home(coord: u64, len: usize, bits: u32) -> usize {
    if (coord as usize) < len {
        coord as usize
    } else {
        (coord.wrapping_mul(hashtrinity::PHI) >> (64 - bits)) as usize
    }
}

// ── The canonical table, through its own counters ────────────────────────────────────────────────

/// One canonical-table cell: the timed nanosecond medians plus, under `read-stats`, the
/// instrumented read-path stage counts.
struct ApeironCell {
    admitted: usize,
    len: usize,
    insert_ns: u64,
    present_ns: u64,
    absent_ns: u64,
    counters: Option<ReadStages>,
    absent_counters: Option<ReadStages>,
}

#[derive(Clone, Copy, Default)]
struct ReadStages {
    summary_tests: u64,
    summary_hits: u64,
    summary_misses: u64,
    sweep_steps: u64,
    tag_matches: u64,
    full_compares: u64,
}

#[cfg(feature = "read-stats")]
fn read_stages<H: Home<Key>>(table: &Table<Key, i64, H, N>, keys: &[Key]) -> Option<ReadStages> {
    let _ = (table, keys);
    let s = apeiron::table::read_stats::snapshot();
    Some(ReadStages {
        summary_tests: s.summary_tests,
        summary_hits: s.summary_hits,
        summary_misses: s.summary_misses,
        sweep_steps: s.sweep_steps,
        tag_matches: s.tag_matches,
        full_compares: s.full_compares,
    })
}

#[cfg(not(feature = "read-stats"))]
fn read_stages<H: Home<Key>>(_table: &Table<Key, i64, H, N>, _keys: &[Key]) -> Option<ReadStages> {
    None
}

fn median(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Reads `keys` once with the counters cleared first, when the feature is on.
#[cfg(feature = "read-stats")]
fn counted_reads<H: Home<Key>>(table: &Table<Key, i64, H, N>, keys: &[Key]) -> Option<ReadStages> {
    apeiron::table::read_stats::clear();
    let mut acc = 0i64;
    for key in keys {
        acc = acc.wrapping_add(*table.get(black_box(key)));
    }
    black_box(acc);
    read_stages(table, keys)
}

#[cfg(not(feature = "read-stats"))]
fn counted_reads<H: Home<Key>>(_table: &Table<Key, i64, H, N>, _keys: &[Key]) -> Option<ReadStages> {
    None
}

fn median_ns(mut run: impl FnMut()) -> u64 {
    for _ in 0..WARMUP {
        run();
    }
    let mut samples = Vec::with_capacity(SAMPLES as usize);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        run();
        samples.push(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }
    median(&mut samples)
}

/// Builds a fresh canonical table over `keys` and returns it with the admitted count.
fn build_apeiron<H: Home<Key>>(home: H, keys: &[Key]) -> (Table<Key, i64, H, N>, usize) {
    let mut t = Table::new(0, home);
    let mut admitted = 0usize;
    for key in keys {
        if t.insert_with(&replacement, *key, value_of(*key)).is_ok() {
            admitted += 1;
        }
    }
    (t, admitted)
}

/// The absent family of a present one: the same words one band above, so an absent key folds
/// exactly like the present pile it probes for.
fn absent_family(present: &[Key]) -> Vec<Key> {
    present
        .iter()
        .map(|k| Key {
            tag: k.tag | ABSENT_BAND,
        })
        .collect()
}

/// Measures the insert total (over fresh builds), the present stream and the absent stream. The
/// insert total is timed rather than counted: `read_stats` instruments `spill_read` only, and
/// `spill_locate` — the insert locator — carries no counter (`src/table/probe.rs`).
fn measure_apeiron<H: Home<Key> + Copy>(home: H, keys: &[Key]) -> ApeironCell {
    let insert_ns = median_ns(|| {
        black_box(build_apeiron(home, keys));
    });
    let (table, admitted) = build_apeiron(home, keys);

    let present_stages = counted_reads(&table, keys);
    let present_ns = median_ns(|| {
        let mut acc = 0i64;
        for key in keys {
            acc = acc.wrapping_add(*table.get(black_box(key)));
        }
        black_box(acc);
    });

    let absent = absent_family(keys);
    let absent_stages = counted_reads(&table, &absent);
    let absent_ns = median_ns(|| {
        let mut acc = 0i64;
        for key in &absent {
            acc = acc.wrapping_add(*table.get(black_box(key)));
        }
        black_box(acc);
    });

    ApeironCell {
        admitted,
        len: table.len(),
        insert_ns,
        present_ns,
        absent_ns,
        counters: present_stages,
        absent_counters: absent_stages,
    }
}

// ── The pinned `hashtrinity` table, timed ─────────────────────────────────────────────────────────

struct HtCell {
    len: usize,
    insert_ns: u64,
    present_ns: u64,
    absent_ns: u64,
}

/// Builds a fresh `hashtrinity` table over `keys`.
fn build_hashtrinity(keys: &[Key]) -> hashtrinity::Table<Key, i64> {
    let mut t = hashtrinity::Table::new();
    for key in keys {
        t.insert_with(*key, value_of(*key), replacement);
    }
    t
}

fn measure_hashtrinity(keys: &[Key]) -> HtCell {
    let insert_ns = median_ns(|| {
        black_box(build_hashtrinity(keys));
    });
    let t = build_hashtrinity(keys);
    let present_ns = median_ns(|| {
        let mut acc = 0i64;
        for key in keys {
            acc = acc.wrapping_add(t.get(black_box(key)).copied().unwrap_or(0));
        }
        black_box(acc);
    });
    let absent = absent_family(keys);
    let absent_ns = median_ns(|| {
        let mut acc = 0i64;
        for key in &absent {
            acc = acc.wrapping_add(t.get(black_box(key)).copied().unwrap_or(0));
        }
        black_box(acc);
    });
    HtCell {
        len: t.len(),
        insert_ns,
        present_ns,
        absent_ns,
    }
}

// ── The `hashtrinity` probe replica (grade `model`, differentially controlled) ───────────────────

/// A replica of the shipped `hashtrinity::Table` probe, transcribed from the committed object
/// `4cc14471:hashtrinity/src/lib.rs`: `bits_for` / `with_bits`, the 7/8 grow trigger in
/// `insert_with`, `resize`'s re-placement, `probe`, `home` and `descend`. No tag line; the linear
/// prefix is `LINEAR_G = 8`.
struct HtReplica {
    coords: Vec<u64>,
    bits: u32,
    len: usize,
    insert_visits: u64,
    present_visits: u64,
    absent_visits: u64,
}

const HT_LINEAR_G: u32 = 8;
const HT_EMPTY: u64 = u64::MAX;

impl HtReplica {
    fn new() -> Self {
        Self::with_bits(3)
    }

    fn with_bits(bits: u32) -> Self {
        Self {
            coords: vec![HT_EMPTY; 1usize << bits],
            bits,
            len: 0,
            insert_visits: 0,
            present_visits: 0,
            absent_visits: 0,
        }
    }

    fn home(&self, coord: u64) -> usize {
        let len = self.coords.len();
        if (coord as usize) < len {
            coord as usize
        } else {
            (coord.wrapping_mul(hashtrinity::PHI) >> (64 - self.bits)) as usize
        }
    }

    fn descend(&self, coord: u64, j: u32) -> usize {
        let (bits, mask) = (self.bits, self.coords.len() - 1);
        if j < HT_LINEAR_G {
            return self.home(coord).wrapping_add(j as usize) & mask;
        }
        let p = coord.wrapping_mul(hashtrinity::PHI);
        let jj = j - HT_LINEAR_G + 1;
        let used = bits * (jj + 1);
        if used <= 64 {
            (p >> (64 - used)) as usize & mask
        } else {
            let full = 64 / bits;
            ((p >> (64 - bits * full)) as usize + (jj - full) as usize + 1) & mask
        }
    }

    /// `Table::probe`: home, then `descend` from `j = 1`, stopping at the coordinate or at EMPTY.
    fn probe(&self, coord: u64) -> (usize, bool, u64) {
        let mut i = self.home(coord);
        let mut j = 0u32;
        let mut visits = 0u64;
        loop {
            visits += 1;
            let sc = self.coords[i];
            if sc == coord {
                return (i, true, visits);
            }
            if sc == HT_EMPTY {
                return (i, false, visits);
            }
            j += 1;
            i = self.descend(coord, j);
            debug_assert!(visits <= self.coords.len() as u64 + 256, "replica probe diverged");
        }
    }

    fn insert(&mut self, coord: u64) {
        if (self.len + 1) * 8 >= self.coords.len() * 7 {
            let new_bits = if (self.len + 1) * 8 >= self.coords.len() * 7 {
                self.bits + 1
            } else {
                self.bits
            };
            self.resize(new_bits);
        }
        let (i, found, visits) = self.probe(coord);
        self.insert_visits += visits;
        if !found {
            self.coords[i] = coord;
            self.len += 1;
        }
    }

    fn resize(&mut self, new_bits: u32) {
        let mut bigger = Self::with_bits(new_bits);
        let mut rehash_visits = 0u64;
        for &c in &self.coords {
            if c != HT_EMPTY {
                let (j, _, visits) = bigger.probe(c);
                rehash_visits += visits;
                bigger.coords[j] = c;
                bigger.len += 1;
            }
        }
        bigger.insert_visits = self.insert_visits + rehash_visits;
        *self = bigger;
    }

    /// `Table::get_coord`: the home scalar probe, then the `LINEAR_G`-lane prefix glance, then the
    /// scalar walk. Visits are counted as slot lanes compared, so the prefix costs 8, matching the
    /// replica's own accounting.
    fn get(&self, coord: u64) -> (bool, u64) {
        let len = self.coords.len();
        let home = self.home(coord);
        let sc = self.coords[home];
        if sc == coord {
            return (true, 1);
        }
        if sc == HT_EMPTY {
            return (false, 1);
        }
        if home + HT_LINEAR_G as usize <= len {
            // `prefix_masks` compares all `LINEAR_G` lanes: `home` plus `home + 1 .. home + 7`.
            // A hit lane is the answer; the first EMPTY lane is absence; otherwise the walk
            // resumes at radix step `LINEAR_G`. Visits are counted as lanes compared, so the prefix
            // costs 8 — the real cost is one SIMD glance over those 8 lanes, which the report says.
            let mut visits = 1u64;
            for lane in 1..HT_LINEAR_G as usize {
                visits += 1;
                let sc = self.coords[home + lane];
                if sc == coord {
                    return (true, visits);
                }
                if sc == HT_EMPTY {
                    return (false, visits);
                }
            }
            let mut j = HT_LINEAR_G;
            loop {
                let i = self.descend(coord, j);
                visits += 1;
                let sc = self.coords[i];
                if sc == coord {
                    return (true, visits);
                }
                if sc == HT_EMPTY {
                    return (false, visits);
                }
                j += 1;
                if j as u64 > len as u64 + 256 {
                    return (false, visits);
                }
            }
        }
        let mut visits = 1u64;
        let mut j = 1u32;
        loop {
            let i = self.descend(coord, j);
            visits += 1;
            let sc = self.coords[i];
            if sc == coord {
                return (true, visits);
            }
            if sc == HT_EMPTY {
                return (false, visits);
            }
            j += 1;
            if j as u64 > len as u64 + 256 {
                return (false, visits);
            }
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Runs the replica and the real table on one population, and fails closed unless the replica's
/// hit/miss prediction agrees with `hashtrinity::Table::get` for every present and absent key.
fn measure_replica(keys: &[Key]) -> Result<(HtReplica, usize), String> {
    let mut replica = HtReplica::new();
    for key in keys {
        replica.insert(key.tag);
    }
    let real = build_hashtrinity(keys);
    if replica.len() != real.len() {
        return Err(format!(
            "replica len {} != hashtrinity len {}",
            replica.len(),
            real.len()
        ));
    }
    for key in keys {
        let (hit, _) = replica.get(key.tag);
        if hit != real.get(key).is_some() {
            return Err(format!("replica disagrees on present key {}", key.tag));
        }
        let absent = Key {
            tag: key.tag | ABSENT_BAND,
        };
        let (hit, _) = replica.get(absent.tag);
        if hit != real.get(&absent).is_some() {
            return Err(format!("replica disagrees on absent key {}", absent.tag));
        }
    }
    // Only now accumulate the read counts, so the differential control is not inside a timed region.
    for key in keys {
        replica.present_visits += replica.get(key.tag).1;
        let absent = Key {
            tag: key.tag | ABSENT_BAND,
        };
        replica.absent_visits += replica.get(absent.tag).1;
    }
    let cap = replica.coords.len();
    Ok((replica, cap))
}

// ── The report ───────────────────────────────────────────────────────────────────────────────────

#[allow(
    clippy::cast_precision_loss,
    reason = "per-key quotients are a diagnostic display of integer nanosecond medians"
)]
fn main() {
    println!("# machine: Intel Core i9-14900KF, 32 logical cpus, linux 6.x");
    println!(
        "# canonical={} hashtrinity=4cc14471ae24f3f204fb0b0e91736ef75c909358 \
         keys={KEYS} group_width={N} samples={SAMPLES} warmup={WARMUP} read_stats={}",
        env!("CARGO_PKG_VERSION"),
        cfg!(feature = "read-stats")
    );
    println!(
        "# home: shared identity (place = key, coord = key), no certified direct word, so every \
         operation routes to a spill region / the hashtrinity slot array"
    );

    for population in [
        distinct_population(),
        lowbyte_population(),
        adversarial_population(),
    ] {
        report(&population);
    }

    // The keyed-`Home` arm: the same adversarial family under one odd multiply, which is a change
    // to a caller-supplied `Home` and not to the table.
    println!();
    println!("## population=adversarial home=keyed (place = key * 0x9E3779B97F4A7C15)");
    let keys: Vec<Key> = (0..KEYS as u64)
        .map(|j| Key { tag: j << 16 })
        .collect();
    let places: Vec<u64> = keys.iter().map(|k| KeyedHome.home(k)).collect();
    print_place_shape("keyed", &places);
    print_apeiron(&measure_apeiron::<KeyedHome>(KeyedHome, &keys));
    let (replica, cap) = measure_replica(&keys).unwrap_or_else(|e| panic!("{e}"));
    print_hashtrinity(&measure_hashtrinity(&keys), &replica, cap);

    // The mixing keyed `Home`: the same adversarial family, the same table, `Home` implemented as
    // a high-to-low fold. This is the arm that can move the shard byte and the fingerprint byte.
    println!();
    println!("## population=adversarial home=mixing (place = key ^ (key >> 16))");
    let places: Vec<u64> = keys.iter().map(|k| MixingHome.home(k)).collect();
    print_place_shape("mixing", &places);
    print_apeiron(&measure_apeiron::<MixingHome>(MixingHome, &keys));
}

/// Prints the realised home shape of a `place` list against the transcribed `shard`, `tag` and
/// descent windows.
fn print_place_shape(label: &str, places: &[u64]) {
    // 32 groups is the region size a 192-key single-shard pile settles at (`slots_for(192, 16)`).
    const GROUPS: usize = 32;
    let mask = GROUPS - 1;
    let (shards, peak_shard) = home_collision_peak(places, apeiron_shard);
    let (tags, peak_tag) = home_collision_peak(places, |p| usize::from(apeiron_tag(p)));
    let (w0, peak_w0) = home_collision_peak(places, |p| apeiron_group(p, 0, mask));
    let (tail, peak_tail) = home_collision_peak(places, |p| {
        (((p >> 8) & DESCENT_MASK) as usize) & mask
    });
    println!(
        "home_shape={label} distinct_shards={shards}/peak{peak_shard} \
         distinct_fingerprints={tags}/peak{peak_tag} group_window0={w0}/peak{peak_w0} \
         tail_base={tail}/peak{peak_tail} (groups={GROUPS})"
    );
}

fn report(population: &Population) {
    let places: Vec<u64> = population.present.iter().map(|k| k.tag).collect();
    // 32 groups is the region size a 192-key single-shard pile settles at: `slots_for(192, 16)`
    // asks for `(384 + 15) / 16 = 24` groups and rounds up to the power of two 32.
    const GROUPS: usize = 32;
    let mask = GROUPS - 1;
    let (distinct_shards, peak_shard) = home_collision_peak(&places, apeiron_shard);
    let (distinct_tags, peak_tag) = home_collision_peak(&places, |p| usize::from(apeiron_tag(p)));
    let (distinct_w0, peak_w0) = home_collision_peak(&places, |p| apeiron_group(p, 0, mask));
    let (distinct_w5, peak_w5) = home_collision_peak(&places, |p| apeiron_group(p, 5, mask));
    let (distinct_tail, peak_tail) = home_collision_peak(&places, |p| {
        (((p >> 8) & DESCENT_MASK) as usize) & mask
    });
    let (distinct_visited, peak_visited) = {
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        for place in &places {
            let mut seen: Vec<usize> = Vec::new();
            for g in apeiron_visited_groups(*place, mask) {
                if !seen.contains(&g) {
                    seen.push(g);
                    *counts.entry(g).or_insert(0) += 1;
                }
            }
        }
        (counts.len(), counts.values().copied().max().unwrap_or(0))
    };
    // `hashtrinity`'s capacity for 192 keys: `bits_for(192) = 256` slots.
    let ht_bits = 8u32;
    let ht_len = 1usize << ht_bits;
    let (distinct_ht, peak_ht) =
        home_collision_peak(&places, |p| ht_home(p, ht_len, ht_bits));

    println!();
    println!("## population={}", population.name);
    println!(
        "home_shape apeiron: distinct_shards={distinct_shards} peak_shard={peak_shard} \
         distinct_fingerprints={distinct_tags} peak_fingerprint={peak_tag} \
         group_window0={distinct_w0}/peak{peak_w0} group_window5={distinct_w5}/peak{peak_w5} \
         tail_base={distinct_tail}/peak{peak_tail} \
         visited_over_probe_order={distinct_visited}/peak{peak_visited} (groups={GROUPS})"
    );
    println!(
        "home_shape hashtrinity: distinct_home_slots={distinct_ht} peak_home_slot={peak_ht} \
         (bits={ht_bits} slots={ht_len})"
    );

    print_apeiron(&measure_apeiron::<IdentityHome>(IdentityHome, &population.present));
    let (replica, cap) = measure_replica(&population.present).unwrap_or_else(|e| panic!("{e}"));
    print_hashtrinity(&measure_hashtrinity(&population.present), &replica, cap);
}

fn print_apeiron(cell: &ApeironCell) {
    let per = |ns: u64| ns as f64 / KEYS as f64;
    println!(
        "impl=apeiron admitted={} len={} insert_ns_per_key={:.4} present_ns_per_key={:.4} \
         absent_ns_per_key={:.4}",
        cell.admitted,
        cell.len,
        per(cell.insert_ns),
        per(cell.present_ns),
        per(cell.absent_ns),
    );
    if let Some(s) = cell.counters {
        let present_per = |v: u64| v as f64 / KEYS as f64;
        println!(
            "impl=apeiron present_stages summary_tests={} summary_hits={} summary_misses={} \
             sweep_steps={} steps_per_key={:.4} tag_matches={} tag_matches_per_key={:.4} \
             full_compares={} full_compares_per_key={:.4}",
            s.summary_tests,
            s.summary_hits,
            s.summary_misses,
            s.sweep_steps,
            present_per(s.sweep_steps),
            s.tag_matches,
            present_per(s.tag_matches),
            s.full_compares,
            present_per(s.full_compares),
        );
    } else {
        println!("impl=apeiron present_stages MISSING (build without --features read-stats)");
    }
    if let Some(s) = cell.absent_counters {
        let per = |v: u64| v as f64 / KEYS as f64;
        println!(
            "impl=apeiron absent_stages summary_tests={} summary_hits={} summary_misses={} \
             sweep_steps={} steps_per_key={:.4} tag_matches={} tag_matches_per_key={:.4} \
             full_compares={} full_compares_per_key={:.4}",
            s.summary_tests,
            s.summary_hits,
            s.summary_misses,
            s.sweep_steps,
            per(s.sweep_steps),
            s.tag_matches,
            per(s.tag_matches),
            s.full_compares,
            per(s.full_compares),
        );
    } else {
        println!("impl=apeiron absent_stages MISSING (build without --features read-stats)");
    }
}

fn print_hashtrinity(cell: &HtCell, replica: &HtReplica, cap: usize) {
    let per = |ns: u64| ns as f64 / KEYS as f64;
    let visits = |v: u64| v as f64 / KEYS as f64;
    println!(
        "impl=hashtrinity len={} slots={cap} insert_ns_per_key={:.4} \
         present_ns_per_key={:.4} absent_ns_per_key={:.4}",
        cell.len,
        per(cell.insert_ns),
        per(cell.present_ns),
        per(cell.absent_ns),
    );
    println!(
        "impl=hashtrinity replica_visits insert={} insert_per_key={:.4} present={} \
         present_per_key={:.4} absent={} absent_per_key={:.4}",
        replica.insert_visits,
        visits(replica.insert_visits),
        replica.present_visits,
        visits(replica.present_visits),
        replica.absent_visits,
        visits(replica.absent_visits),
    );
}
