//! Focused release micro-benchmark for the pathological spill cells.
//!
//! Reproduces the `occ16-seeded-random-absent-miss-replacement`,
//! `occ16-direct-index-resident-growth-insert-replacement`,
//! `occ16-direct-index-resident-present-hit-replacement` and
//! `occ16-offset-beyond-direct-present-hit-replacement` schedules from
//! `examples/reg1/strata.rs` on the real `apeiron::Table`, without the hashtrinity
//! baseline. The point is a fast before/after number for the spill probe, the
//! resident read and the growth path, not REG-1 acceptance.
//!
//! Each cell prints `cell=... ops=... median_ns=... ns_per_op=...`; the timed region
//! is exactly the 4096-op schedule the strata harness times.

use std::hint::black_box;
use std::time::Instant;

use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue,
    WordFaceHome, H3_40,
};
use apeiron::coord::locator::{ExactIdentity, Refusal};
use apeiron::table::{Home, Table, TableAddress};
use apeiron::Sigma;

/// The fixed shard count of `apeiron::Table`.
const SHARDS: usize = 256;
/// The compile-time group width under measurement.
const N: usize = 16;
/// Scheduled operations in one trace, matching `STRATA_OPS`.
const OPS: usize = 4096;
/// Warmup traces before sampling.
const WARMUP: u32 = 2;
/// Timed samples per cell.
const SAMPLES: u32 = 15;

/// The deterministic workload seed, matching `STRATA_SEED`.
const STRATA_SEED: u64 = 0x5EED_1234_ABCD_0001;
/// The base of the offset-beyond-direct coordinate band, matching `OFFSET_BASE`.
const OFFSET_BASE: u64 = 1 << 24;
/// The base of the seeded-random coordinate band.
const RAND_BASE: u64 = 1 << 25;
/// The 23-bit mask of the seeded-random displacement.
const RAND_MASK: u64 = (1 << 23) - 1;
/// The odd seeded-random multiplier.
const RAND_ODD: u64 = 0x9E37_79B9_7F4A_7C15;

/// One coordinate key; identity is the raw tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    tag: u64,
}

/// The serial raw-coordinate home: `place = tag`, direct word certified below `2^32`.
struct RawHome;

impl Home<Key> for RawHome {
    fn home(&self, key: &Key) -> u64 {
        key.tag
    }

    fn address(&self, _sigma: Sigma, key: &Key) -> TableAddress {
        TableAddress {
            place: key.tag,
            direct_word: u32::try_from(key.tag).ok(),
        }
    }
}

/// The exact identity of a diagnostic coordinate key: the raw 64-bit tag as eight big-endian
/// bytes.
///
/// The tag *is* the key, so the encoding is the original bit pattern, most-significant byte first.
/// No residue and no physical placement participates, so distinct tags have distinct encodings.
/// [`tests::distinct_diagnostic_keys_encode_distinctly`] asserts that; the I3-E rejecting control
/// keeps only the low byte and must turn that assertion red.
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

/// The explicit I3-A face capability for the raw-coordinate home.
///
/// The face map is derived from the key's tag, never from the physical placement (`place = tag`),
/// so the collision-only physical route cannot leak into a face encoding. `Faces = Sigma` because a
/// `u64` tag has an exact [`Packed`](apeiron::coord::Packed) word; the parabolic and hyperbolic
/// maps delegate to the reviewed [`WordFaceHome`] word instance.
impl FaceResidueCapability<Key> for RawHome {
    fn parabolic_dual(&self, key: &Key) -> ParabolicDual {
        WordFaceHome.parabolic_dual(&key.tag)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &Key) -> Residue {
        WordFaceHome.hyperbolic_scalar(profile, &key.tag)
    }
}

impl EllipticPacking<Key> for RawHome {
    fn elliptic_pack(&self, key: &Key) -> u64 {
        WordFaceHome.elliptic_pack(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<Key> {
        WordFaceHome.elliptic_unpack(word).map(|tag| Key { tag })
    }
}

impl FaceCoord<Key> for RawHome {
    type Faces = Sigma;
}

impl FaceHome<Key> for RawHome {}

type Table16 = Table<Key, i64, RawHome, N>;

fn replacement_merge(new: i64, _old: i64) -> i64 {
    new
}

fn initial_value(key: Key) -> i64 {
    key.tag.cast_signed().wrapping_add(1)
}

fn direct_key(shard: usize, slot: usize) -> Key {
    Key {
        tag: shard as u64 + 256 * slot as u64,
    }
}

/// An offset-beyond-direct coordinate: `OFFSET_BASE + shard + 256 * slot`, matching
/// `strata::offset_key` at occupancy 16.
fn offset_key(shard: usize, slot: usize) -> Key {
    Key {
        tag: OFFSET_BASE + shard as u64 + 256 * slot as u64,
    }
}

fn random_key(shard: usize, index: u64) -> Key {
    let shard = shard as u64;
    let word = index
        .wrapping_add(STRATA_SEED)
        .wrapping_mul(RAND_ODD)
        .wrapping_add(shard)
        & RAND_MASK;
    Key {
        tag: RAND_BASE + shard + 256 * word,
    }
}

fn class_offset(class: u8) -> u64 {
    u64::from(class) * 1_000_000
}

/// The present/absent/fresh key classes for one distribution.
struct Plan {
    present: Vec<Key>,
    absent: Vec<Key>,
    fresh: Vec<Key>,
}

fn seeded_random_plan() -> Plan {
    let mut present = Vec::with_capacity(SHARDS * N);
    let mut absent = Vec::with_capacity(SHARDS * N);
    let mut fresh = Vec::with_capacity(SHARDS * N);
    for shard in 0..SHARDS {
        for slot in 0..N {
            let base = (shard * 16 + slot) as u64;
            present.push(random_key(shard, class_offset(0) + base));
            absent.push(random_key(shard, class_offset(1) + base + 16));
            fresh.push(random_key(shard, class_offset(2) + base + 16));
        }
    }
    Plan {
        present,
        absent,
        fresh,
    }
}

fn direct_plan() -> Plan {
    let mut present = Vec::with_capacity(SHARDS * N);
    let mut absent = Vec::with_capacity(SHARDS * N);
    let mut fresh = Vec::with_capacity(SHARDS * N);
    for shard in 0..SHARDS {
        for slot in 0..N {
            present.push(direct_key(shard, slot));
            absent.push(direct_key(shard, N + slot));
            fresh.push(direct_key(shard, N + slot));
        }
    }
    Plan {
        present,
        absent,
        fresh,
    }
}

/// The `offset-beyond-direct` present set at occupancy 16: `OFFSET_BASE + shard + 256 * slot`.
///
/// These coordinates are certified direct words (`< 2^32`) whose direct lane is far beyond the
/// resident byte budget, so every present key routes to its shard's spill region.
fn offset_plan() -> Plan {
    let mut present = Vec::with_capacity(SHARDS * N);
    for shard in 0..SHARDS {
        for slot in 0..N {
            present.push(offset_key(shard, slot));
        }
    }
    Plan {
        present,
        absent: Vec::new(),
        fresh: Vec::new(),
    }
}

fn build(keys: &[Key]) -> (Table16, usize) {
    let mut table = Table16::new(0, RawHome);
    let mut admitted = 0usize;
    for key in keys {
        if table
            .insert_with(&replacement_merge, *key, initial_value(*key))
            .is_ok()
        {
            admitted += 1;
        }
    }
    (table, admitted)
}

fn median(samples: &[u64]) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

/// Time 4096 `get` calls over `keys`; returns median elapsed ns.
fn time_reads(table: &Table16, keys: &[Key]) -> u64 {
    let mut samples = Vec::with_capacity(SAMPLES as usize);
    for _ in 0..WARMUP {
        let mut acc = 0i64;
        for key in keys.iter().take(OPS) {
            acc = acc.wrapping_add(*table.get(black_box(key)));
        }
        black_box(acc);
    }
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let mut acc = 0i64;
        for key in keys.iter().take(OPS) {
            acc = acc.wrapping_add(*table.get(black_box(key)));
        }
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        black_box(acc);
        samples.push(elapsed);
    }
    median(&samples)
}

/// Time 4096 fresh `insert_with` calls from a freshly built admitted table; returns
/// median elapsed ns. The admitted build is outside the timed region.
fn time_growth(present: &[Key], fresh: &[Key]) -> u64 {
    let mut samples = Vec::with_capacity(SAMPLES as usize);
    for _ in 0..WARMUP {
        let (mut sample, _) = build(present);
        for (i, key) in fresh.iter().enumerate().take(OPS) {
            let value = i64::try_from(i)
                .unwrap_or(i64::MAX)
                .wrapping_mul(3)
                .wrapping_add(7);
            let _ = sample.insert_with(&replacement_merge, black_box(*key), black_box(value));
        }
    }
    for _ in 0..SAMPLES {
        let (mut sample, _) = build(present);
        let start = Instant::now();
        for (i, key) in fresh.iter().enumerate().take(OPS) {
            let value = i64::try_from(i)
                .unwrap_or(i64::MAX)
                .wrapping_mul(3)
                .wrapping_add(7);
            let _ = sample.insert_with(&replacement_merge, black_box(*key), black_box(value));
        }
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        black_box(&sample);
        samples.push(elapsed);
    }
    median(&samples)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the ns/op quotient is a diagnostic display of an integer nanosecond median"
)]
fn main() {
    // Cell 1: seeded-random absent miss. Build the 4096 admitted keys, then time the
    // 4096 absent lookups.
    let plan = seeded_random_plan();
    let (table, admitted) = build(&plan.present);
    let absent_ns = time_reads(&table, &plan.absent);
    println!(
        "cell=occ16-seeded-random-absent-miss-replacement admitted={admitted} len={} ops={OPS} \
         median_ns={absent_ns} ns_per_op={:.4}",
        table.len(),
        absent_ns as f64 / OPS as f64,
    );

    // Cell 2: direct-index-resident growth insert. Build the 4096 admitted keys, then
    // time 4096 fresh inserts that cross the resident extent.
    let plan = direct_plan();
    let growth_ns = time_growth(&plan.present, &plan.fresh);
    let (table, admitted) = build(&plan.present);
    println!(
        "cell=occ16-direct-index-resident-growth-insert-replacement admitted={admitted} len={} \
         ops={OPS} median_ns={growth_ns} ns_per_op={:.4}",
        table.len(),
        growth_ns as f64 / OPS as f64,
    );

    // Cell 3: direct-index-resident present hit. Build the 4096 admitted keys, then time the
    // 4096 present lookups; every key is a certified direct word inside the resident extent.
    let plan = direct_plan();
    let (table, admitted) = build(&plan.present);
    let resident_ns = time_reads(&table, &plan.present);
    println!(
        "cell=occ16-direct-index-resident-present-hit-replacement admitted={admitted} len={} \
         ops={OPS} median_ns={resident_ns} ns_per_op={:.4}",
        table.len(),
        resident_ns as f64 / OPS as f64,
    );

    // Cell 4: offset-beyond-direct present hit. Build the 4096 admitted keys, then time the
    // 4096 present lookups; every key is in its shard's spill region.
    let plan = offset_plan();
    let (table, admitted) = build(&plan.present);
    let present_ns = time_reads(&table, &plan.present);
    println!(
        "cell=occ16-offset-beyond-direct-present-hit-replacement admitted={admitted} len={} \
         ops={OPS} median_ns={present_ns} ns_per_op={:.4}",
        table.len(),
        present_ns as f64 / OPS as f64,
    );

    // Attribution (only under `--features read-stats`): the release path times a build with the
    // counters compiled out, so an attributed run of the same schedules reads the counters that
    // split the read cost into summary tests, sweep steps, fingerprint matches and full
    // `(Sigma, K)` compares.
    #[cfg(feature = "read-stats")]
    attribute(&plan.present, &seeded_random_plan().absent);
}

/// Prints the read-path stage counters for one present-hit schedule and one absent-miss schedule.
///
/// Feature `read-stats` only: the counters are `#[cfg(feature = "read-stats")]` in the crate, so
/// this function and the counters vanish from the timed release build together.
#[cfg(feature = "read-stats")]
#[allow(
    clippy::cast_precision_loss,
    reason = "the sweep-per-op quotient is a diagnostic display of an integer count"
)]
fn attribute(offset_present: &[Key], absent: &[Key]) {
    let (table, _) = build(offset_present);
    apeiron::table::read_stats::clear();
    let mut acc = 0i64;
    for key in offset_present.iter().take(OPS) {
        acc = acc.wrapping_add(*table.get(black_box(key)));
    }
    black_box(acc);
    let s = apeiron::table::read_stats::snapshot();
    println!(
        "attribute=occ16-offset-present-hit ops={OPS} summary_tests={} summary_hits={} \
         summary_misses={} sweep_steps={} steps_per_op={:.3} tag_matches={} full_compares={}",
        s.summary_tests,
        s.summary_hits,
        s.summary_misses,
        s.sweep_steps,
        s.sweep_steps as f64 / OPS as f64,
        s.tag_matches,
        s.full_compares,
    );

    let plan = seeded_random_plan();
    let (table, _) = build(&plan.present);
    apeiron::table::read_stats::clear();
    let mut acc = 0i64;
    for key in absent.iter().take(OPS) {
        acc = acc.wrapping_add(*table.get(black_box(key)));
    }
    black_box(acc);
    let s = apeiron::table::read_stats::snapshot();
    println!(
        "attribute=occ16-seeded-random-absent-miss ops={OPS} summary_tests={} summary_hits={} \
         summary_misses={} sweep_steps={} tag_matches={} full_compares={}",
        s.summary_tests,
        s.summary_hits,
        s.summary_misses,
        s.sweep_steps,
        s.tag_matches,
        s.full_compares,
    );
}

/// The inline I3-E identity and face round-trip assertions.
///
/// These run under `cargo test --all-targets`, not only when the example binary is executed, so
/// the semantic encoding is checked on the same diagnostic keys the workload uses.
#[cfg(test)]
mod tests {
    use super::{direct_key, offset_key, random_key, RawHome, H3_40, N};
    use apeiron::coord::address::{AddressValue, EllipticPacking, FaceCoord};
    use apeiron::coord::locator::ExactIdentity;
    use apeiron::Sigma;

    /// The frozen profile the face assertions encode against.
    fn profile() -> H3_40 {
        H3_40::new().expect("the frozen H3_40 constants validate")
    }

    /// The exact identity bytes of a diagnostic key are its tag's big-endian bytes, and the
    /// retained identity round-trips to the same key.
    #[test]
    fn diagnostic_key_identity_bytes_are_the_original_tag() {
        for key in [
            direct_key(0, 0),
            direct_key(3, 7),
            offset_key(255, 15),
            random_key(7, 0),
        ] {
            assert_eq!(key.encoding_len(), 8);
            let bytes: Vec<u8> = (0..key.encoding_len())
                .map(|index| key.encoding_byte(index))
                .collect();
            assert_eq!(bytes, key.tag.to_be_bytes().to_vec());
            assert!(key.matches_retained(&key.retain()));
        }
    }

    /// Distinct diagnostic keys must encode to distinct byte sequences.
    ///
    /// This is the assertion the I3-E rejecting control mutates: keeping only the low byte makes
    /// `direct_key(0, 0)` and `offset_key(0, 0)` collide on `[0]`.
    #[test]
    fn distinct_diagnostic_keys_encode_distinctly() {
        let keys = [
            direct_key(0, 0),
            direct_key(0, N),
            offset_key(0, 0),
            random_key(0, 0),
        ];
        for (i, a) in keys.iter().enumerate() {
            for (j, b) in keys.iter().enumerate() {
                if i == j {
                    continue;
                }
                let ea: Vec<u8> = (0..a.encoding_len())
                    .map(|index| a.encoding_byte(index))
                    .collect();
                let eb: Vec<u8> = (0..b.encoding_len())
                    .map(|index| b.encoding_byte(index))
                    .collect();
                assert_ne!(ea, eb, "distinct diagnostic keys share identity bytes");
            }
        }
    }

    /// The elliptic face round-trips exactly and all three faces separate distinct keys.
    #[test]
    fn face_encodings_round_trip_and_separate_distinct_keys() {
        let profile = profile();
        let home = RawHome;
        let a = direct_key(0, 0);
        let b = direct_key(0, N);

        assert_eq!(home.elliptic_unpack(home.elliptic_pack(&a)), Some(a));

        for face in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
            let fa = FaceCoord::coord(&home, profile, face, &a);
            let fb = FaceCoord::coord(&home, profile, face, &b);
            assert_ne!(fa, fb, "distinct keys share a face address");
            assert_eq!(fa.face(), face);
        }

        // The elliptic address is the exact word, not the physical placement.
        assert_eq!(
            FaceCoord::coord(&home, profile, Sigma::Elliptic, &a).value(),
            AddressValue::EllipticWord(a.tag)
        );
    }
}
