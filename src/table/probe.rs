//! Probe, read and placement helpers factored out of `src/table.rs` without behavior change.
//!
//! Declared as the private child module `table::probe`; `src/table.rs` re-exports each moved
//! name so existing callers and tests keep the same `table.*` path. The spill region is one flat
//! open-addressed array: the probe is bounded linear probing over one cycle, the zero tag
//! terminates it, and a completed cycle with no vacancy is the growth signal.

use super::{matches, tag, valid_n, KeyRep, Region, Sigma, SpillSite, TableAddress};

/// Walks one bounded linear-probe cycle and classifies the query.
///
/// Returns the unique full `(Sigma,K)` match, the first vacant slot at or after the query's home,
/// or [`SpillSite::Grow`] when the whole cycle was visited and every slot was occupied — the growth
/// signal, and equally the absence certificate for the read path.
///
/// **Probe sequence and bound.** Step `k` visits `(home + k) & (capacity - 1)` for
/// `k = 0 .. capacity`, so a cycle visits each of the `capacity` slots exactly once. A tag
/// fingerprint mismatch advances; a matching tag runs the full `(Sigma,K)` compare; a zero tag
/// stops with a vacancy, because nothing is ever removed from a region (no tombstones).
///
/// **Growth.** `Grow` means a complete probe cycle found every slot occupied, which is what forces
/// `grow_spill` to reserve a larger region and relocate.
///
/// Temporal `O(capacity)` slot visits counted in tag compares plus the full-key compares of the
/// lanes whose fingerprint matches; spatial `O(1)`.
pub(super) fn spill_locate<K, H, R: KeyRep<K, H>, V>(
    r: &Region<R, V>,
    sigma: Sigma,
    address: &TableAddress,
    key: &K,
    n: usize,
) -> SpillSite {
    if !valid_n(n) || r.bound() == 0 {
        return SpillSite::Grow;
    }
    let bound = r.bound();
    let home = r.home(address.place);
    let wanted = tag(address.place);
    let mut step = 0;
    while step < bound {
        let slot = (home + step) & (bound - 1);
        let found = r.slots[slot].tag;
        if found == 0 {
            return SpillSite::Vacant(slot);
        }
        if found == wanted {
            #[cfg(feature = "read-stats")]
            read_stats::record_full_compare();
            if matches::<K, H, R, V>(&r.slots[slot], sigma, key, address) {
                return SpillSite::Found(slot);
            }
        }
        step += 1;
    }
    SpillSite::Grow
}

/// Walks one bounded linear-probe cycle against an already-held carrier.
///
/// Same sequence, bound and growth meaning as [`spill_locate`]; the compare runs on the held
/// carrier without materializing the key, which is the relocation path's requirement.
pub(super) fn spill_locate_held<K, H, R: KeyRep<K, H>, V>(
    r: &Region<R, V>,
    sigma: Sigma,
    address: &TableAddress,
    carrier: &R,
    n: usize,
) -> SpillSite {
    if !valid_n(n) || r.bound() == 0 {
        return SpillSite::Grow;
    }
    let bound = r.bound();
    let home = r.home(address.place);
    let wanted = tag(address.place);
    let mut step = 0;
    while step < bound {
        let slot = (home + step) & (bound - 1);
        let found = r.slots[slot].tag;
        if found == 0 {
            return SpillSite::Vacant(slot);
        }
        if found == wanted
            && r.slots[slot].sigma == sigma
            && R::matches_held(sigma, &r.slots[slot].held, carrier)
        {
            return SpillSite::Found(slot);
        }
        step += 1;
    }
    SpillSite::Grow
}

/// Test-only counters for the read path, so a read's probe cost is attributable. Inert outside
/// `cfg(test)`.
#[cfg(test)]
pub(super) mod read_trace {
    use std::cell::Cell;

    thread_local! {
        static ABSENCE_CERTIFICATES: Cell<usize> = const { Cell::new(0) };
        static SWEEP_STEPS: Cell<usize> = const { Cell::new(0) };
    }

    /// Clears both counters on this thread.
    pub(crate) fn clear() {
        ABSENCE_CERTIFICATES.with(|count| count.set(0));
        SWEEP_STEPS.with(|count| count.set(0));
    }

    /// Records one read that returned absence on a zero tag.
    pub(crate) fn record_absence_certificate() {
        ABSENCE_CERTIFICATES.with(|count| count.set(count.get() + 1));
    }

    /// Records one probe step of a read's bounded linear cycle.
    pub(crate) fn record_sweep_step() {
        SWEEP_STEPS.with(|count| count.set(count.get() + 1));
    }

    /// Returns the absence-certificate count on this thread.
    pub(crate) fn absence_certificates() -> usize {
        ABSENCE_CERTIFICATES.with(Cell::get)
    }

    /// Returns the bounded-cycle step count on this thread.
    pub(crate) fn sweep_steps() -> usize {
        SWEEP_STEPS.with(Cell::get)
    }
}

/// Attribution counters for the spill read path.
///
/// Compiled only with the `read-stats` feature, so the release path that `examples/micro_spill.rs`
/// times is byte-identical to the un-instrumented build and the Charon/Aeneas extraction unit (a
/// debug profile, where `debug_assertions` is on) never sees the `thread_local`; an attributed run
/// reads the counters to split the present-hit cost into its stages. They are diagnostics, not a
/// gate: a counter is a count of real executions, never a substitute for the timed release number.
#[cfg(feature = "read-stats")]
pub mod read_stats {
    use core::cell::Cell;

    /// Stage counts accumulated over the reads on one thread since the last [`clear`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ReadStats {
        /// `spill_read` calls that entered the bounded cycle.
        pub summary_tests: u64,
        /// Reads that stopped on a zero tag, certified absent by the no-deletion policy.
        pub summary_hits: u64,
        /// Reads that completed a full cycle without a vacancy, certified absent by the bound.
        pub summary_misses: u64,
        /// Probe steps of the bounded linear cycle.
        pub sweep_steps: u64,
        /// Tag lanes equal to the query fingerprint.
        pub tag_matches: u64,
        /// Full `(Sigma, K)` compares reached through a fingerprint match.
        pub full_compares: u64,
    }

    thread_local! {
        static STATS: Cell<ReadStats> = const { Cell::new(ReadStats {
            summary_tests: 0,
            summary_hits: 0,
            summary_misses: 0,
            sweep_steps: 0,
            tag_matches: 0,
            full_compares: 0,
        }) };
    }

    /// Clears the counters on this thread.
    pub fn clear() {
        STATS.with(|stats| stats.set(ReadStats::default()));
    }

    /// Records one entry into the bounded cycle.
    pub(super) fn record_summary(_hit: bool) {
        STATS.with(|stats| {
            let mut s = stats.get();
            s.summary_tests += 1;
            stats.set(s);
        });
    }

    /// Records one read that returned absence, and which certificate ended it.
    ///
    /// `by_vacancy` is `true` for the zero-tag stop and `false` for a completed full cycle.
    pub(super) fn record_absence(by_vacancy: bool) {
        STATS.with(|stats| {
            let mut s = stats.get();
            if by_vacancy {
                s.summary_hits += 1;
            } else {
                s.summary_misses += 1;
            }
            stats.set(s);
        });
    }

    /// Records one step of the bounded cycle.
    pub(super) fn record_step() {
        STATS.with(|stats| {
            let mut s = stats.get();
            s.sweep_steps += 1;
            stats.set(s);
        });
    }

    /// Records the fingerprint-match count of one cycle.
    pub(super) fn record_tag_matches(count: u64) {
        STATS.with(|stats| {
            let mut s = stats.get();
            s.tag_matches += count;
            stats.set(s);
        });
    }

    /// Records one full `(Sigma, K)` compare.
    pub(super) fn record_full_compare() {
        STATS.with(|stats| {
            let mut s = stats.get();
            s.full_compares += 1;
            stats.set(s);
        });
    }

    /// Returns the counters accumulated on this thread.
    pub fn snapshot() -> ReadStats {
        STATS.with(Cell::get)
    }
}

/// Reads one routed spill region without any insertion bookkeeping.
///
/// Returns the slot index of the unique full `(Sigma,K)` match, or `None` for absence. **Absence is
/// certified by the bounded probe**: either a zero tag terminates the cycle, which is sound because
/// no entry is ever removed from a region, or the cycle visits all `capacity` slots without a
/// match, in which case every slot was visited and the bound `SpillProbeBoundAbsentSound` states
/// applies. This function never searches for a vacancy and never computes a first fit.
///
/// Temporal `O(capacity)` slot visits counted in tag compares plus the full-key compares of the
/// lanes whose fingerprint matches; spatial `O(1)`.
#[allow(
    clippy::len_zero,
    reason = "len avoids the unmodeled Vec::is_empty extraction"
)]
pub(super) fn spill_read<K, H, R: KeyRep<K, H>, V>(
    r: &Region<R, V>,
    sigma: Sigma,
    address: &TableAddress,
    key: &K,
    n: usize,
) -> Option<usize> {
    let mut hit = None;
    if r.slots.len() != 0 && valid_n(n) {
        #[cfg(feature = "read-stats")]
        read_stats::record_summary(true);
        let bound = r.bound();
        let mask = bound - 1;
        let home = r.home(address.place);
        let wanted = tag(address.place);
        let mut step = 0;
        while step < bound && hit.is_none() {
            #[cfg(test)]
            read_trace::record_sweep_step();
            #[cfg(feature = "read-stats")]
            read_stats::record_step();
            let slot = (home + step) & mask;
            let found = r.slots[slot].tag;
            if found == 0 {
                // Absence certificate: nothing is ever removed, so a live key cannot sit past a
                // vacancy.
                #[cfg(test)]
                read_trace::record_absence_certificate();
                #[cfg(feature = "read-stats")]
                read_stats::record_absence(true);
                step = bound;
            } else {
                if found == wanted {
                    #[cfg(feature = "read-stats")]
                    read_stats::record_tag_matches(1);
                    #[cfg(feature = "read-stats")]
                    read_stats::record_full_compare();
                    if matches::<K, H, R, V>(&r.slots[slot], sigma, key, address) {
                        hit = Some(slot);
                    }
                }
                step += 1;
            }
        }
        if hit.is_none() {
            // Absence certificate: the cycle visited every slot and found no match.
            #[cfg(feature = "read-stats")]
            read_stats::record_absence(false);
        }
    }
    hit
}

/// Finite regression fixtures for the real `spill_read`, `spill_locate` and `spill_locate_held`
/// bodies on the flat open-addressed region.
///
/// These tests run the real helpers, `Region`, the nonzero tag marker and `Stored<i64>` comparison
/// at `n = 2` over a four-slot region whose home is slot 0 (placement 0). The fixtures are built in
/// first-fit order, which is the only state a linear insert can produce: a live slot never sits
/// past a vacancy. They cover named concrete cases only. They do not establish whole-`Table`
/// construction, production geometry, canonical refinement or universal behavior.
#[cfg(test)]
mod tests {
    use crate::table::keyrep::{KeyRep, Stored};
    use crate::table::{Region, TableAddress, WordHome};
    use crate::Sigma;

    use super::{spill_locate, spill_locate_held, spill_read, tag, SpillSite};
    use crate::table::TAG_FINGERPRINT_BITS;

    const N: usize = 2;
    const LEN: usize = 4;

    fn held(key: i64, address: &TableAddress) -> Stored<i64> {
        <Stored<i64> as KeyRep<i64, WordHome>>::hold(Sigma::Parabolic, &key, address)
    }

    fn address() -> TableAddress {
        TableAddress {
            place: 0,
            direct_word: None,
        }
    }

    /// Four slots, home 0, published in first-fit order with keys `k0, k1, k2` in slots 0, 1, 2
    /// and slot 3 vacant. Every published slot carries the query fingerprint (placement 0), so a
    /// walk past any of them costs a full-key reject.
    fn fixture(k0: i64, k1: i64, k2: i64) -> Region<Stored<i64>, i64> {
        let address = address();
        let mut region = Region::reserve_empty(LEN).expect("four slots");
        region.initialize(Sigma::Parabolic, &held(k0, &address), &0);
        region.publish_live(0, (Sigma::Parabolic, held(k0, &address), 41), address.place);
        region.publish_live(1, (Sigma::Parabolic, held(k1, &address), 43), address.place);
        region.publish_live(2, (Sigma::Parabolic, held(k2, &address), 7), address.place);
        region
    }

    /// The same region with the fourth slot published: every slot occupied, so a query with no
    /// full-key match must exhaust the bounded cycle.
    fn full_fixture(k0: i64, k1: i64, k2: i64) -> Region<Stored<i64>, i64> {
        let address = address();
        let mut region = fixture(k0, k1, k2);
        region.publish_live(3, (Sigma::Parabolic, held(11, &address), 13), address.place);
        region
    }

    /// The real insertion locator classifies a match past two same-fingerprint rejects, the first
    /// vacancy, and exhaustion on assumption-free, initialized flat regions.
    #[test]
    fn spill_locate_classifies_a_fixed_initialized_region() {
        let region = fixture(7, 8, 9);
        assert_eq!(
            spill_locate::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &9,
                N,
            ),
            SpillSite::Found(2),
            "two same-fingerprint rejects must not hide the full-key match at slot 2"
        );
        assert_eq!(
            spill_locate::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &10,
                N,
            ),
            SpillSite::Vacant(3),
            "same-fingerprint key collisions must retain the first vacancy"
        );
        assert_eq!(
            spill_locate::<i64, WordHome, Stored<i64>, i64>(
                &full_fixture(7, 8, 9),
                Sigma::Parabolic,
                &address(),
                &10,
                N,
            ),
            SpillSite::Grow,
            "a full region with no full-key match exhausts the cycle and requires growth"
        );
        assert_eq!(
            spill_locate::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Elliptic,
                &address(),
                &9,
                N,
            ),
            SpillSite::Vacant(3),
            "an equal key under a different Sigma is not a match, so the walk continues to the \
             vacancy"
        );
        assert_eq!(
            spill_locate_held::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &held(9, &address()),
                N,
            ),
            SpillSite::Found(2),
            "the held carrier walk finds the same slot"
        );
    }

    #[test]
    fn spill_read_returns_the_matching_slot() {
        let region = fixture(7, 8, 9);
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &9,
                N,
            ),
            Some(2),
            "the two slot-0 and slot-1 fingerprint collisions must not stop the walk to slot 2"
        );
    }

    #[test]
    fn spill_read_reports_absence_without_a_full_key_match() {
        let region = fixture(7, 8, 9);
        for query in [10, i64::MIN, i64::MAX] {
            assert_eq!(
                spill_read::<i64, WordHome, Stored<i64>, i64>(
                    &region,
                    Sigma::Parabolic,
                    &address(),
                    &query,
                    N,
                ),
                None,
                "a fingerprint collision on every occupied slot is still absence without a full match"
            );
        }
    }

    #[test]
    fn spill_read_certifies_absence_on_a_zero_tag() {
        let region = fixture(7, 8, 9);
        let other = TableAddress {
            place: 1 << 40,
            direct_word: None,
        };
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &other,
                &99,
                N,
            ),
            None,
            "a different placement homes elsewhere; its zero tag certifies absence"
        );
    }

    #[test]
    fn spill_read_certifies_absence_on_a_completed_cycle() {
        let region = full_fixture(7, 8, 9);
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &10,
                N,
            ),
            None,
            "a full region with no match exhausts the bound, which certifies absence"
        );
    }

    /// The match at the **last** step of the cycle.
    ///
    /// `full_fixture` publishes key 11 into slot 3 of a four-slot region homed at slot 0, and
    /// `spill_read` visits `(home + step) & mask` for `step = 0 .. slots.len()`, so the match sits at
    /// `step = slots.len() - 1`. A cycle stopped one step early never reaches it and returns the
    /// caller's default, which is what distinguishes this fixture from the interior matches above:
    /// those are reachable by any cycle long enough to reach their own slot, so they cannot
    /// discriminate the bound.
    #[test]
    fn spill_read_finds_a_match_at_the_last_step_of_the_cycle() {
        let region = full_fixture(7, 8, 9);
        let last = region.slots.len() - 1;
        assert!(
            region.slots[last].live(),
            "the fixture's match occupies the last slot, so the region is full"
        );
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &11,
                N,
            ),
            Some(last),
            "a cycle one step short of the last slot would return the default here"
        );
    }

    /// The base placement the forced-collision fixture perturbs: a nonzero word, so its published
    /// tag is nonzero and every `base + i` neighbour is a distinct placement.
    fn collision_base() -> u64 {
        0x1234
    }

    /// Two distinct placements that share a tag and home elsewhere.
    ///
    /// The shared tag is derived from [`tag`], not assumed. [`tag`] reads the top
    /// [`TAG_FINGERPRINT_BITS`] bits of `place * TAG_SPREAD`, so the tag takes
    /// `2^TAG_FINGERPRINT_BITS` values and any `2 * 2^TAG_FINGERPRINT_BITS` consecutive placements
    /// contain a colliding pair by the pigeonhole principle; the scan below covers twice that
    /// width. The folding class is therefore a residue class of the spread product, not an
    /// arithmetic family of `place`: shifting `place` by a whole fingerprint window changes the
    /// spread product's top bits, because `TAG_SPREAD` is odd, so `place = j << k` *separates*
    /// fingerprints rather than folding them. The scan runs over a fixed sequence, so the fixture is
    /// deterministic, and it reports the pair it found rather than the tag alone.
    fn same_tag_other_home(region: &Region<Stored<i64>, i64>) -> (u64, u64) {
        let base = collision_base();
        let mut seen: Vec<(u8, u64)> = Vec::new();
        for step in 0..(2u64 << TAG_FINGERPRINT_BITS) {
            let place = base + step;
            let fingerprint = tag(place);
            if let Some((_, other)) = seen.iter().find(|(seen_tag, _)| *seen_tag == fingerprint) {
                let other = *other;
                if other != place && region.home(other) != region.home(place) {
                    return (other, place);
                }
            }
            seen.push((fingerprint, place));
        }
        panic!(
            "no colliding pair in 2 * 2^TAG_FINGERPRINT_BITS consecutive placements homes apart"
        );
    }

    /// Publishes `key` at the slot the real locator selects: the first vacancy at or after its home,
    /// which is exactly where a linear insert places it.
    fn publish_first_fit(region: &mut Region<Stored<i64>, i64>, key: i64, place: u64) -> usize {
        let address = TableAddress {
            place,
            direct_word: None,
        };
        let slot = match spill_locate::<i64, WordHome, Stored<i64>, i64>(
            region,
            Sigma::Parabolic,
            &address,
            &key,
            N,
        ) {
            SpillSite::Vacant(slot) => slot,
            other => panic!("a fresh key selects a vacancy, not {other:?}"),
        };
        region.publish_live(
            slot,
            (Sigma::Parabolic, held(key, &address), 100 + key),
            place,
        );
        slot
    }

    /// Two distinct keys under one forced same-fingerprint collision: every published slot carries
    /// the same nonzero tag, so the walk must resolve each key by its full compare and must not
    /// stop on the tag, merge the keys, or report absence for a stored one.
    ///
    /// The pair is derived, not assumed: [`same_tag_other_home`] scans a fixed window for two
    /// placements whose published tags agree and whose homes differ, which the pigeonhole argument
    /// in its own contract guarantees within the scanned width, and the fixture then fills the
    /// region through the production locator, so every state is one a linear insert produces. This
    /// covers distinct placements, distinct homes and a shared tag, which the single-placement
    /// fixtures above cannot reach.
    #[test]
    fn spill_read_resolves_a_forced_same_fingerprint_collision() {
        const WIDE: usize = 16;
        let mut region = Region::reserve_empty(WIDE).expect("sixteen slots");
        region.initialize(Sigma::Parabolic, &held(0, &address()), &0);
        let (base, other) = same_tag_other_home(&region);
        assert_eq!(
            tag(base),
            tag(other),
            "the forced pair shares one nonzero tag"
        );
        assert_ne!(base, other, "the forced pair is two distinct placements");
        assert_ne!(
            region.home(base),
            region.home(other),
            "the forced pair homes elsewhere, so the two probe orders differ"
        );

        // Eight keys alternate between the two placements: every slot carries the shared tag and
        // every walk runs a full compare at each one.
        let mut placed: Vec<(i64, u64, usize)> = Vec::new();
        for key in 1..=8i64 {
            let place = if key % 2 == 0 { other } else { base };
            let slot = publish_first_fit(&mut region, key, place);
            placed.push((key, place, slot));
        }
        let live = region.slots.iter().filter(|slot| slot.live()).count();
        assert_eq!(live, 8, "one live slot per key");
        assert!(
            region
                .slots
                .iter()
                .filter(|slot| slot.live())
                .all(|slot| slot.tag == tag(base)),
            "every live slot carries the shared tag, so no tag alone can resolve the walk"
        );

        // No false negative: every stored key is found at the slot its own insert published.
        for (key, place, slot) in &placed {
            let at = TableAddress {
                place: *place,
                direct_word: None,
            };
            assert_eq!(
                spill_read::<i64, WordHome, Stored<i64>, i64>(
                    &region,
                    Sigma::Parabolic,
                    &at,
                    key,
                    N,
                ),
                Some(*slot),
                "a same-tag neighbour must not hide the full-key match"
            );
        }

        // No false positive and no merge: the shared tag is not identity.
        let base_at = TableAddress {
            place: base,
            direct_word: None,
        };
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &base_at,
                &99,
                N,
            ),
            None,
            "an absent key under the shared tag is absence, not the neighbour's entry"
        );
        for (key, _place, slot) in &placed {
            assert_eq!(
                region.slots[*slot].value,
                100 + key,
                "one key owns one slot; a shared tag merges nothing"
            );
        }
    }
}
