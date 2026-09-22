//! Focused release micro-benchmark for the packed spatial build+probe face.
//!
//! Reproduces the registered `71942db scene(n)` packed-u64 build+probe workload that
//! `examples/reg1/adapters/spatial.rs` measures on the real `apeiron::Table`, and splits the timed
//! region into its real phases: input qualification, partition, scatter validation, schedule,
//! scoped per-shard build, and assembly; then the same for the grouped probe. The point is a fast
//! before/after number and the dominant phase, not REG-1 acceptance.
//!
//! Each size prints one `spatial_micro` line per phase group plus the end-to-end medians. The
//! independent oracle is a `std::collections::HashMap` count folded over the probe set in query
//! order, so a run whose selected digest disagrees is reported and not trusted.

use std::collections::HashMap as StdHashMap;
use std::hint::black_box;
use std::time::Instant;

use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue,
    WordFaceHome, H3_40,
};
use apeiron::coord::locator::{ExactIdentity, Refusal};
use apeiron::coord::{i21, Packed};
use apeiron::table::{Home, Table, TableAddress};
use apeiron::Sigma;

/// The compile-time per-shard override width, matching the spatial selected table.
const N: usize = 16;
/// The worker width the micro harness schedules at.
const WORKERS: usize = 16;
/// Warmup build+probe passes before sampling.
const WARMUP: u32 = 2;
/// Timed samples per size.
const SAMPLES: u32 = 15;
/// The registered packed spatial sizes.
const SIZES: [usize; 2] = [100_000, 1_000_000];

/// The `71942db` scene seed and cell size, matching the registered workload.
const SCENE_SEED: u64 = 0x243F_6A88_85A3_08D3;
const CELL: f32 = 0.25;

/// FNV-1a 64-bit offset basis and prime, matching the strata digest.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One packed world key; identity is the raw tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    tag: u64,
}

/// The registered spatial home: `place` is the coordinate spread, `direct_word` is certified below
/// `2^32`.
struct KeyHome;

impl Home<Key> for KeyHome {
    fn home(&self, key: &Key) -> u64 {
        apeiron::coord::spread(key.tag).place
    }

    fn address(&self, _sigma: Sigma, key: &Key) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: u32::try_from(key.tag).ok(),
        }
    }
}

/// The exact identity of a packed spatial key: the raw 64-bit tag as eight big-endian bytes.
///
/// The packed tag *is* the key, so the encoding is the original bit pattern, most-significant byte
/// first. No residue and no physical placement participates, so distinct packed cells have distinct
/// encodings. [`tests::distinct_packed_keys_encode_distinctly`] asserts that; the I3-E rejecting
/// control keeps only the low byte and must turn that assertion red.
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

/// The explicit I3-A face capability for the registered spatial home.
///
/// The face map is derived from the packed tag, never from the physical `spread` placement, so the
/// collision-only physical route cannot leak into a face encoding. `Faces = Sigma` because the tag
/// has an exact [`Packed`] word; the parabolic and hyperbolic maps delegate to the reviewed
/// [`WordFaceHome`] word instance.
impl FaceResidueCapability<Key> for KeyHome {
    fn parabolic_dual(&self, key: &Key) -> ParabolicDual {
        WordFaceHome.parabolic_dual(&key.tag)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &Key) -> Residue {
        WordFaceHome.hyperbolic_scalar(profile, &key.tag)
    }
}

impl EllipticPacking<Key> for KeyHome {
    fn elliptic_pack(&self, key: &Key) -> u64 {
        WordFaceHome.elliptic_pack(&key.tag)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<Key> {
        WordFaceHome.elliptic_unpack(word).map(|tag| Key { tag })
    }
}

impl FaceCoord<Key> for KeyHome {
    type Faces = Sigma;
}

impl FaceHome<Key> for KeyHome {}

type SpatialTable = Table<Key, i64, KeyHome, N>;

fn merge(new: i64, old: i64) -> i64 {
    new.wrapping_add(old)
}

fn fold(hash: &mut u64, value: i64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// A deterministic xorshift scene generator, copied from `71942db:benches/spatial_binning.rs`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "the scene is the registered historical float workload; the cast is the workload"
    )]
    fn f(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 * 2000.0 - 1000.0
    }
}

/// A clustered scene, as entities actually sit. Reproduces the registered `71942db` `scene(n)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the registered spatial workload is this exact float scene; the casts are its definition"
)]
fn scene(n: usize) -> Vec<(i32, i32, i32)> {
    let mut r = Rng(SCENE_SEED);
    let clusters: Vec<(f32, f32, f32)> = (0..32).map(|_| (r.f(), r.f() * 0.1, r.f())).collect();
    (0..n)
        .map(|i| {
            let c = clusters[i % clusters.len()];
            let j = |v: f32, r: &mut Rng| v + ((r.next() >> 40) as f32 / 16777.0 - 50.0);
            (
                (j(c.0, &mut r) * CELL) as i32,
                (j(c.1, &mut r) * CELL) as i32,
                (j(c.2, &mut r) * CELL) as i32,
            )
        })
        .collect()
}

/// The registered packed key: three `i21` world cells packed into one word.
fn pack(k: (i32, i32, i32)) -> u64 {
    (
        i21::new(k.0).expect("scene coordinate fits i21"),
        i21::new(k.1).expect("scene coordinate fits i21"),
        i21::new(k.2).expect("scene coordinate fits i21"),
    )
        .packed()
}

/// The packed scene entities and the packed distinct occupied-cell probe set.
fn spatial_input(n: usize) -> (Vec<(Key, i64)>, Vec<Key>) {
    let coords = scene(n);
    let mut occupied = coords.clone();
    occupied.sort_unstable();
    occupied.dedup();
    (
        coords.iter().map(|&k| (Key { tag: pack(k) }, 1)).collect(),
        occupied.iter().map(|&k| Key { tag: pack(k) }).collect(),
    )
}

/// The independent count oracle: count every packed key, then fold the probe counts in query order.
fn oracle_digest(keys: &[(Key, i64)], probe: &[Key]) -> u64 {
    let mut counts: StdHashMap<u64, i64> = StdHashMap::with_capacity(keys.len());
    for (key, _) in keys {
        *counts.entry(key.tag).or_insert(0) += 1;
    }
    let mut hash = FNV_OFFSET;
    for key in probe {
        fold(&mut hash, counts.get(&key.tag).copied().unwrap_or(0));
    }
    hash
}

fn median(values: &[u64]) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

/// The phase medians for one phase group.
struct Split {
    qualify: u64,
    partition: u64,
    validate: u64,
    schedule: u64,
    reserve: u64,
    kernel: u64,
    assemble: u64,
}

impl Split {
    fn from_medians(samples: &[apeiron::table::parallel::PhaseNanos]) -> Self {
        let field = |f: fn(&apeiron::table::parallel::PhaseNanos) -> u64| {
            median(&samples.iter().map(f).collect::<Vec<_>>())
        };
        Self {
            qualify: field(|p| p.qualify),
            partition: field(|p| p.partition),
            validate: field(|p| p.validate),
            schedule: field(|p| p.schedule),
            reserve: field(|p| p.reserve),
            kernel: field(|p| p.kernel),
            assemble: field(|p| p.assemble),
        }
    }

    fn total(&self) -> u64 {
        self.qualify
            + self.partition
            + self.validate
            + self.schedule
            + self.reserve
            + self.kernel
            + self.assemble
    }

    /// Input qualification, stable partition, scatter validation, LPT schedule and serial
    /// reservation: the work done before any probe worker starts.
    fn routing(&self) -> u64 {
        self.qualify + self.partition + self.validate + self.schedule + self.reserve
    }

    fn print(&self, cell: &str, group: &str) {
        let total = self.total();
        #[allow(
            clippy::cast_precision_loss,
            reason = "phase shares are a diagnostic display of integer nanosecond medians"
        )]
        let pct = |v: u64| v as f64 * 100.0 / total.max(1) as f64;
        println!(
            "spatial_micro cell={cell} group={group} qualify={} ({:.1}%) partition={} ({:.1}%) \
             validate={} ({:.1}%) schedule={} ({:.1}%) reserve={} ({:.1}%) kernel={} ({:.1}%) \
             assemble={} ({:.1}%) total={total}",
            self.qualify,
            pct(self.qualify),
            self.partition,
            pct(self.partition),
            self.validate,
            pct(self.validate),
            self.schedule,
            pct(self.schedule),
            self.reserve,
            pct(self.reserve),
            self.kernel,
            pct(self.kernel),
            self.assemble,
            pct(self.assemble),
        );
    }

    /// The probe attribution the task names: routing (qualification, partition, validation,
    /// schedule, readback reservation), the scoped per-shard parallel probe, and the grouped
    /// query-order readback/assemble.
    fn print_probe_split(&self, cell: &str) {
        let total = self.total();
        #[allow(
            clippy::cast_precision_loss,
            reason = "phase shares are a diagnostic display of integer nanosecond medians"
        )]
        let pct = |v: u64| v as f64 * 100.0 / total.max(1) as f64;
        println!(
            "spatial_micro cell={cell} group=probe_split routing={} ({:.1}%) \
             per_shard_probe={} ({:.1}%) readback_assemble={} ({:.1}%) total={total}",
            self.routing(),
            pct(self.routing()),
            self.kernel,
            pct(self.kernel),
            self.assemble,
            pct(self.assemble),
        );
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the ns/op quotient is a diagnostic display of an integer nanosecond median"
)]
fn main() {
    println!(
        "spatial_micro protocol=\"71942db scene(n) packed u64, build+probe\" sizes={SIZES:?} \
         workers={WORKERS} warmup={WARMUP} samples={SAMPLES} metric=elapsed_ns_median \
         oracle=std::HashMap count folded in query order"
    );
    for size in SIZES {
        let (keys, probe) = spatial_input(size);
        let oracle = oracle_digest(&keys, &probe);
        let cell = format!("spatial/n{size}");

        let mut build_phases = Vec::with_capacity(SAMPLES as usize);
        let mut probe_phases = Vec::with_capacity(SAMPLES as usize);
        let mut build_ns = Vec::with_capacity(SAMPLES as usize);
        let mut probe_ns = Vec::with_capacity(SAMPLES as usize);
        let mut end_ns = Vec::with_capacity(SAMPLES as usize);
        let mut digest_ok = true;

        for pass in 0..(WARMUP + SAMPLES) {
            let start = Instant::now();
            let (table, build_phase) =
                SpatialTable::build_parallel_phased(&merge, 0, &keys, KeyHome, WORKERS)
                    .expect("spatial parallel build must fit");
            let build_elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let start = Instant::now();
            let (values, probe_phase) =
                table.get_each_parallel_phased(&probe, WORKERS, |value| *value);
            let probe_elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let end_elapsed = build_elapsed + probe_elapsed;
            let mut hash = FNV_OFFSET;
            for value in &values {
                fold(&mut hash, *value);
            }
            digest_ok &= hash == oracle;
            black_box(&table);
            if pass >= WARMUP {
                build_phases.push(build_phase);
                probe_phases.push(probe_phase);
                build_ns.push(build_elapsed);
                probe_ns.push(probe_elapsed);
                end_ns.push(end_elapsed);
            }
        }

        let build_split = Split::from_medians(&build_phases);
        let probe_split = Split::from_medians(&probe_phases);
        let build_med = median(&build_ns);
        let probe_med = median(&probe_ns);
        let end_med = median(&end_ns);
        println!(
            "spatial_micro cell={cell} population={} ops={} digest_ok={digest_ok} \
             build_median_ns={build_med} probe_median_ns={probe_med} end_to_end_median_ns={end_med} \
             ns_per_op={:.4}",
            keys.len(),
            probe.len(),
            end_med as f64 / keys.len() as f64,
        );
        build_split.print(&cell, "build");
        probe_split.print(&cell, "probe");
        probe_split.print_probe_split(&cell);
    }
}

/// The inline I3-E identity and face round-trip assertions.
///
/// These run under `cargo test --all-targets`, not only when the example binary is executed, so
/// the semantic encoding is checked on the same packed keys the workload uses.
#[cfg(test)]
mod tests {
    use super::{pack, Key, KeyHome, H3_40};
    use apeiron::coord::address::{AddressValue, EllipticPacking, FaceCoord};
    use apeiron::coord::locator::ExactIdentity;
    use apeiron::Sigma;

    /// The frozen profile the face assertions encode against.
    fn profile() -> H3_40 {
        H3_40::new().expect("the frozen H3_40 constants validate")
    }

    /// Four distinct packed world cells.
    fn packed_cells() -> [Key; 4] {
        [
            Key {
                tag: pack((0, 0, 0)),
            },
            Key {
                tag: pack((1, 0, 0)),
            },
            Key {
                tag: pack((0, 1, 0)),
            },
            Key {
                tag: pack((0, 0, 1)),
            },
        ]
    }

    /// The exact identity bytes of a packed key are its tag's big-endian bytes, and the retained
    /// identity round-trips to the same key.
    #[test]
    fn packed_key_identity_bytes_are_the_original_tag() {
        for key in packed_cells() {
            assert_eq!(key.encoding_len(), 8);
            let bytes: Vec<u8> = (0..key.encoding_len())
                .map(|index| key.encoding_byte(index))
                .collect();
            assert_eq!(bytes, key.tag.to_be_bytes().to_vec());
            assert!(key.matches_retained(&key.retain()));
        }
    }

    /// Distinct packed keys must encode to distinct byte sequences.
    ///
    /// This is the assertion the I3-E rejecting control mutates: keeping only the low byte makes
    /// distinct cells collide.
    #[test]
    fn distinct_packed_keys_encode_distinctly() {
        let keys = packed_cells();
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
                assert_ne!(ea, eb, "distinct packed keys share identity bytes");
            }
        }
    }

    /// The elliptic face round-trips exactly and all three faces separate distinct packed keys.
    #[test]
    fn face_encodings_round_trip_and_separate_distinct_keys() {
        let profile = profile();
        let home = KeyHome;
        let [a, b, ..] = packed_cells();

        assert_eq!(home.elliptic_unpack(home.elliptic_pack(&a)), Some(a));

        for face in [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic] {
            let fa = FaceCoord::coord(&home, profile, face, &a);
            let fb = FaceCoord::coord(&home, profile, face, &b);
            assert_ne!(fa, fb, "distinct packed keys share a face address");
            assert_eq!(fa.face(), face);
        }

        // The elliptic address is the exact packed word, not the physical spread placement.
        assert_eq!(
            FaceCoord::coord(&home, profile, Sigma::Elliptic, &a).value(),
            AddressValue::EllipticWord(a.tag)
        );
    }
}
