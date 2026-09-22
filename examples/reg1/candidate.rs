//! Benchmark-only candidate representation for the REG-1 shared fixed-16 measurement.
//!
//! This module is **not** the production `apeiron::Table`, is **not** proof-covered, and makes no
//! correctness claim beyond the benchmark's own controls. It exists to measure one alternative
//! physical representation before any production migration decision.
//!
//! The candidate keeps the authoritative `[Option<(K, V)>; N]` slots of the production table and
//! adds, per shard, a compact occupancy prefix length and a `[u64; N]` cache of full `Home`
//! outputs. `get`/`insert` compute the home word once, reduce it `% 256` to select the shard, scan
//! the compact word prefix, and accept a slot only after a full `K::eq` comparison. The cached word
//! is a filter, never an identity: two unequal keys with the same full home word stay distinct
//! while capacity permits, and the collision witness observes exactly that.

#![allow(
    clippy::module_name_repetitions,
    reason = "the module owns the benchmark-only candidate and its collision witness"
)]

use apeiron::coord::address::{
    EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, ParabolicDual, Residue, H3_40,
};
use apeiron::table::{Home, Merge, ReplacementMerge, StorageError, TableAddress};
use apeiron::Sigma;

use crate::strata::{StrataHome, StrataKey};

/// The fixed shard count, matching the production table.
pub(crate) const SHARD_COUNT: usize = 256;

/// The full home word every collision-witness key returns.
pub(crate) const COLLISION_WORD: u64 = 0x0000_0000_DEAD_BEEF;

/// Reduce a full home word to its shard index. This is the production `% 256` reduction, applied
/// after the full word is cached.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a residue below 256 fits usize"
)]
const fn shard_index(word: u64) -> usize {
    (word % 256) as usize
}

/// One shard: an occupancy prefix length, a compact cache of full home words, and the
/// authoritative slots. The production table never removes, so slots `0..occupancy` are exactly
/// the occupied set.
struct Shard<K, V, const N: usize> {
    occupancy: usize,
    words: [u64; N],
    slots: [Option<(K, V)>; N],
}

impl<K, V, const N: usize> Shard<K, V, N> {
    const fn new() -> Self {
        Self {
            occupancy: 0,
            words: [0; N],
            slots: [const { None }; N],
        }
    }
}

/// The benchmark-only candidate table.
///
/// `FULL_KEY` selects the accept rule. `true` compares the full key after a home-word match and is
/// the measured candidate. `false` accepts the cached home word alone and is the deliberately
/// broken control variant; it must never be reported as the candidate.
pub(crate) struct CandidateTable<K, V, H, const N: usize, const FULL_KEY: bool = true> {
    default: V,
    home: H,
    shards: [Shard<K, V, N>; SHARD_COUNT],
}

impl<K, V, H: Home<K>, const N: usize, const FULL_KEY: bool> CandidateTable<K, V, H, N, FULL_KEY> {
    /// Creates an empty candidate whose total function initially returns `default`.
    #[must_use]
    pub(crate) fn new(default: V, home: H) -> Self {
        Self {
            default,
            home,
            shards: [const { Shard::new() }; SHARD_COUNT],
        }
    }

    /// The occupied prefix length of the shard selected by `word`.
    #[must_use]
    pub(crate) fn shard_occupancy(&self, word: u64) -> usize {
        self.shards[shard_index(word)].occupancy
    }
}

impl<K: Eq, V, H: Home<K>, const N: usize, const FULL_KEY: bool>
    CandidateTable<K, V, H, N, FULL_KEY>
{
    // Temporal O(occupancy): at most N cached-word comparisons and, per word match, one full
    // `K::eq`. Spatial O(1): index arithmetic on a borrowed shard, no allocation.
    fn find(&self, shard: usize, word: u64, key: &K) -> Option<usize> {
        let shard_slots = &self.shards[shard];
        let mut i = 0;
        while i < shard_slots.occupancy {
            if shard_slots.words[i] == word {
                if !FULL_KEY {
                    return Some(i);
                }
                if let Some((stored, _)) = &shard_slots.slots[i] {
                    if stored == key {
                        return Some(i);
                    }
                }
            }
            i += 1;
        }
        None
    }

    /// Returns the stored override, or the total default for an unwritten key.
    ///
    /// Temporal O(occupancy) for the shard selected by one home evaluation; spatial O(1); the
    /// result is a borrow, not a copy.
    #[must_use]
    pub(crate) fn get(&self, key: &K) -> &V {
        let word = Home::home(&self.home, key);
        let shard = shard_index(word);
        if let Some(i) = self.find(shard, word, key) {
            if let Some((_, value)) = &self.shards[shard].slots[i] {
                return value;
            }
        }
        &self.default
    }
}

impl<K: Eq, V: Clone, H: Home<K>, const N: usize, const FULL_KEY: bool>
    CandidateTable<K, V, H, N, FULL_KEY>
{
    /// Applies `merge(new, old)`, including the default on the first write.
    ///
    /// Temporal O(occupancy) for the shard selected by one home evaluation, plus the caller's
    /// `merge`/`clone`; spatial O(1) and no allocation. The shard is stored inline.
    ///
    /// # Errors
    /// Returns [`StorageError::ShardFull`] exactly when the key is absent and its shard has no
    /// vacant slot. Refusal invokes neither clone nor merge and changes no field.
    pub(crate) fn insert_with<F: Merge<V>>(
        &mut self,
        merge: &F,
        key: K,
        value: V,
    ) -> Result<(), StorageError> {
        let word = Home::home(&self.home, &key);
        let shard = shard_index(word);
        if let Some(i) = self.find(shard, word, &key) {
            if let Some((_, stored)) = &mut self.shards[shard].slots[i] {
                *stored = merge.merge(value, stored.clone());
                return Ok(());
            }
        }
        if N == 0 {
            return Err(StorageError::ShardFull);
        }
        let occupancy = self.shards[shard].occupancy;
        if occupancy == N {
            return Err(StorageError::ShardFull);
        }
        let merged = merge.merge(value, self.default.clone());
        let target = &mut self.shards[shard];
        target.slots[occupancy] = Some((key, merged));
        target.words[occupancy] = word;
        target.occupancy = occupancy + 1;
        Ok(())
    }
}

/// The collision fixture reuses the migrated word-wrapper key rather than defining another one.
///
/// `StrataKey` is a `u64` tag whose equality, exact identity and face capabilities already live in
/// `adapters/native_strata.rs`; the candidate's collision home only overrides the physical
/// placement.
pub(crate) type CollisionKey = StrataKey;

/// A home that returns [`COLLISION_WORD`] for every key.
///
/// `Home` is the degenerate physical placement adapter: it certifies no coordinate word, so
/// `direct_word` is `None`. The semantic face map is [`FaceHome`], delegated to the migrated
/// [`StrataHome`], and is deliberately separate from that constant placement. This is a benchmark
/// fixture, not a production home.
pub(crate) struct CollisionHome;

impl Home<StrataKey> for CollisionHome {
    fn home(&self, _key: &StrataKey) -> u64 {
        COLLISION_WORD
    }

    /// Physical placement only: the constant collision word, with no certified coordinate word.
    fn address(&self, _sigma: Sigma, _key: &StrataKey) -> TableAddress {
        TableAddress {
            place: COLLISION_WORD,
            direct_word: None,
        }
    }
}

/// The semantic face capability of [`CollisionHome`], delegated to the migrated [`StrataHome`] so
/// the face map is exactly the word-wrapper encoding and not a second implementation.
impl FaceResidueCapability<StrataKey> for CollisionHome {
    fn parabolic_dual(&self, key: &StrataKey) -> ParabolicDual {
        FaceResidueCapability::parabolic_dual(&StrataHome, key)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &StrataKey) -> Residue {
        FaceResidueCapability::hyperbolic_scalar(&StrataHome, profile, key)
    }
}

impl EllipticPacking<StrataKey> for CollisionHome {
    fn elliptic_pack(&self, key: &StrataKey) -> u64 {
        EllipticPacking::elliptic_pack(&StrataHome, key)
    }

    fn elliptic_unpack(&self, word: u64) -> Option<StrataKey> {
        EllipticPacking::elliptic_unpack(&StrataHome, word)
    }
}

impl FaceCoord<StrataKey> for CollisionHome {
    type Faces = Sigma;
}

impl FaceHome<StrataKey> for CollisionHome {}

/// The collision observation: two unequal keys that share one full home word.
#[allow(
    clippy::struct_excessive_bools,
    reason = "each boolean is an independent observation of the collision fixture"
)]
pub(crate) struct CollisionWitness {
    pub(crate) keys: usize,
    pub(crate) capacity: usize,
    pub(crate) word_equal: bool,
    pub(crate) occupancy: usize,
    pub(crate) first_value: i64,
    pub(crate) second_value: i64,
    pub(crate) first_read_ok: bool,
    pub(crate) second_read_ok: bool,
    pub(crate) zero_capacity_refused: bool,
    pub(crate) zero_read_is_default: bool,
}

/// Run the collision fixture against the measured candidate (`FULL_KEY = true`).
#[must_use]
pub(crate) fn run_collision_witness() -> CollisionWitness {
    collision_witness::<true>()
}

/// Run the collision fixture against the deliberately broken word-only variant
/// (`FULL_KEY = false`). The same validator must reject it.
#[must_use]
pub(crate) fn run_word_only_collision_witness() -> CollisionWitness {
    collision_witness::<false>()
}

/// Build the collision observation with a chosen accept rule.
fn collision_witness<const FULL_KEY: bool>() -> CollisionWitness {
    const N: usize = 4;
    const ZERO_DEFAULT: i64 = 7;
    let first = CollisionKey { tag: 1 };
    let second = CollisionKey { tag: 2 };
    let mut table: CandidateTable<CollisionKey, i64, CollisionHome, N, FULL_KEY> =
        CandidateTable::new(0, CollisionHome);
    let first_ok = table.insert_with(&ReplacementMerge, first, 11).is_ok();
    let second_ok = table.insert_with(&ReplacementMerge, second, 22).is_ok();
    let first_value = *table.get(&first);
    let second_value = *table.get(&second);
    let home = CollisionHome;

    let mut zero: CandidateTable<CollisionKey, i64, CollisionHome, 0> =
        CandidateTable::new(ZERO_DEFAULT, CollisionHome);
    let zero_capacity_refused =
        zero.insert_with(&ReplacementMerge, first, 1) == Err(StorageError::ShardFull);
    let zero_read_is_default = *zero.get(&first) == ZERO_DEFAULT;

    CollisionWitness {
        keys: 2,
        capacity: N,
        word_equal: Home::home(&home, &first) == Home::home(&home, &second),
        occupancy: table.shard_occupancy(COLLISION_WORD),
        first_value,
        second_value,
        first_read_ok: first_ok && first_value == 11,
        second_read_ok: second_ok && second_value == 22,
        zero_capacity_refused,
        zero_read_is_default,
    }
}

/// Validate the collision witness: the two unequal keys share one full home word and, because
/// capacity permits, both remain stored and read back distinctly. This is the evidence that a
/// cached home word is a filter and not an identity, and that `N = 0` still refuses.
///
/// # Errors
/// Returns a description of the first observation that fails.
pub(crate) fn validate_collision_witness(w: &CollisionWitness) -> Result<(), String> {
    if w.keys != 2 || w.capacity < 2 {
        return Err(format!(
            "collision fixture keys={} capacity={} cannot hold two distinct keys",
            w.keys, w.capacity
        ));
    }
    if !w.word_equal {
        return Err("collision fixture keys do not share a full home word".to_string());
    }
    if w.occupancy != w.keys {
        return Err(format!(
            "cached-home acceptance coalesced {} distinct keys into occupancy {}",
            w.keys, w.occupancy
        ));
    }
    if !w.first_read_ok || !w.second_read_ok {
        return Err(format!(
            "collision readback failed: first={} second={}",
            w.first_value, w.second_value
        ));
    }
    if !w.zero_capacity_refused {
        return Err("N=0 candidate did not refuse an insert".to_string());
    }
    if !w.zero_read_is_default {
        return Err("N=0 candidate read is not the default".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CollisionHome, COLLISION_WORD};
    use crate::strata::{StrataHome, StrataKey};
    use apeiron::coord::address::{FaceHome, H3_40};
    use apeiron::table::Home;
    use apeiron::Sigma;

    /// Audit: the collision home's physical placement is not its semantic face address.
    ///
    /// `Home::address` is the degenerate constant placement with no coordinate certificate;
    /// `FaceHome::address` is the migrated word-wrapper map and is unaffected by it. The candidate
    /// routes on the physical placement only, so it never consumes the semantic map.
    #[test]
    fn collision_home_physical_and_semantic_are_separate() {
        let key = StrataKey { tag: 7 };
        let profile = H3_40::new().expect("the frozen H3_40 token validates");

        let physical = Home::address(&CollisionHome, Sigma::Elliptic, &key);
        assert_eq!(physical.place, COLLISION_WORD);
        assert_eq!(physical.direct_word, None);

        let semantic = FaceHome::address(&CollisionHome, profile, Sigma::Elliptic, &key);
        assert_eq!(semantic.face(), Sigma::Elliptic);
        assert_eq!(semantic.value().elliptic_word(), Some(7));
        // The delegated semantic map is exactly the migrated `StrataHome`'s.
        assert_eq!(
            semantic,
            FaceHome::address(&StrataHome, profile, Sigma::Elliptic, &key)
        );
    }
}
