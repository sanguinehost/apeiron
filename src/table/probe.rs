//! Probe, read and placement helpers factored out of `src/table.rs` without behavior change.
//!
//! Declared as the private child module `table::probe`; `src/table.rs` re-exports each moved
//! name so existing callers and tests keep the same `table.*` path. Bodies, signatures and the
//! summary branches are unchanged.

use super::{
    group, least_set_lane, matches, scan, tag, valid_n, KeyRep, Region, Sigma, SpillSite,
    TableAddress, WINDOWS,
};

#[allow(
    clippy::len_zero,
    clippy::similar_names,
    reason = "the tag line and its selected lane are conventional probe terms; len avoids the unmodeled Vec::is_empty extraction"
)]
pub(super) fn spill_locate<K, H, R: KeyRep<K, H>, V>(
    r: &Region<R, V>,
    sigma: Sigma,
    address: &TableAddress,
    key: &K,
    n: usize,
) -> SpillSite {
    let mut result = SpillSite::Grow;
    if r.slots.len() != 0 {
        let groups = r.slots.len() / n;
        if groups >= 2 && groups * n == r.slots.len() && groups.is_power_of_two() {
            let mut vacant = (false, 0);
            let mut found = (false, 0);
            let mask = groups - 1;
            let wanted = tag(address.place);
            // A summary miss proves the query absent, so the match scan can be skipped and the walk
            // may stop at the first vacancy. A summary hit (or an absent summary) runs the tag
            // scan, which rejects every lane whose fingerprint differs from `wanted` before the
            // full-key compare; a fingerprint or summary collision only costs that compare.
            let may_match = r.summary_maybe(address.place);
            let mut step = 0;
            while step < WINDOWS + groups && !found.0 && (may_match || !vacant.0) {
                let base = group(address.place, step, mask) * n;
                if may_match {
                    let mut hits = scan::match_tag(&r.tags[base..base + n], wanted);
                    while hits != 0 && !found.0 {
                        let lane = least_set_lane(hits);
                        let slot = base + lane;
                        let matched = matches::<K, H, R, V>(&r.slots[slot], sigma, key, address);
                        if matched {
                            found = (true, base + lane);
                        }
                        hits &= hits - 1;
                    }
                }
                if !vacant.0 {
                    let empty = scan::match_empty(&r.tags[base..base + n]);
                    if empty != 0 {
                        vacant = (true, base + least_set_lane(empty));
                    }
                }
                step += 1;
            }
            if found.0 {
                result = SpillSite::Found(found.1);
            } else if vacant.0 {
                result = SpillSite::Vacant(vacant.1);
            }
        }
    }
    result
}

/// Test-only counters for the read path, so a read can be shown to skip the summary-negative
/// sweep and to run complete summary-positive coverage. Inert outside `cfg(test)`.
#[cfg(test)]
pub(super) mod read_trace {
    use std::cell::Cell;

    thread_local! {
        static SUMMARY_MISSES: Cell<usize> = const { Cell::new(0) };
        static SWEEP_STEPS: Cell<usize> = const { Cell::new(0) };
    }

    /// Clears both counters on this thread.
    pub(crate) fn clear() {
        SUMMARY_MISSES.with(|count| count.set(0));
        SWEEP_STEPS.with(|count| count.set(0));
    }

    /// Records one read that returned absence on a summary miss.
    pub(crate) fn record_summary_miss() {
        SUMMARY_MISSES.with(|count| count.set(count.get() + 1));
    }

    /// Records one group step of a read's coverage sweep.
    pub(crate) fn record_sweep_step() {
        SWEEP_STEPS.with(|count| count.set(count.get() + 1));
    }

    /// Returns the summary-miss count on this thread.
    pub(crate) fn summary_misses() -> usize {
        SUMMARY_MISSES.with(Cell::get)
    }

    /// Returns the coverage-sweep step count on this thread.
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
        /// `spill_read` calls that reached the summary test.
        pub summary_tests: u64,
        /// Summary hits that entered the tag sweep.
        pub summary_hits: u64,
        /// Summary misses that returned absence without a sweep.
        pub summary_misses: u64,
        /// Group steps of the tag sweep, the window iteration.
        pub sweep_steps: u64,
        /// Tag lanes equal to the query fingerprint, the SWAR `match_tag` result popcount.
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

    /// Records one summary test and whether it hit.
    pub(super) fn record_summary(hit: bool) {
        STATS.with(|stats| {
            let mut s = stats.get();
            s.summary_tests += 1;
            if hit {
                s.summary_hits += 1;
            } else {
                s.summary_misses += 1;
            }
            stats.set(s);
        });
    }

    /// Records one group step of a tag sweep.
    pub(super) fn record_step() {
        STATS.with(|stats| {
            let mut s = stats.get();
            s.sweep_steps += 1;
            stats.set(s);
        });
    }

    /// Records the fingerprint-match count of one tag scan.
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
/// Returns the slot index of the unique full `(Sigma,K)` match, or `None` for absence.  Absence is
/// certified by the caller's direct exclusion plus a summary miss, or by complete finite match
/// coverage; this function never searches for a vacancy and never computes a first fit.
///
/// Temporal O(`WINDOWS + groups`) group steps, each O(`n`) tag lanes plus the full-key compares of
/// the lanes whose fingerprint matches, counted in tag-lane comparisons; spatial O(1).  Every group
/// is visited in probe order, so a matching key after an earlier empty lane is still found.
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
        let groups = r.slots.len() / n;
        if groups >= 2 && groups * n == r.slots.len() && groups.is_power_of_two() {
            if r.summary_maybe(address.place) {
                #[cfg(feature = "read-stats")]
                read_stats::record_summary(true);
                let mask = groups - 1;
                let wanted = tag(address.place);
                let mut step = 0;
                while step < WINDOWS + groups && hit.is_none() {
                    #[cfg(test)]
                    read_trace::record_sweep_step();
                    #[cfg(feature = "read-stats")]
                    read_stats::record_step();
                    let base = group(address.place, step, mask) * n;
                    let mut hits = scan::match_tag(&r.tags[base..base + n], wanted);
                    #[cfg(feature = "read-stats")]
                    read_stats::record_tag_matches(u64::from(hits.count_ones()));
                    while hits != 0 && hit.is_none() {
                        let lane = scan::least_set_lane(hits);
                        let slot = base + lane;
                        #[cfg(feature = "read-stats")]
                        read_stats::record_full_compare();
                        if matches::<K, H, R, V>(&r.slots[slot], sigma, key, address) {
                            hit = Some(slot);
                        }
                        hits &= hits - 1;
                    }
                    step += 1;
                }
            } else {
                #[cfg(test)]
                read_trace::record_summary_miss();
                #[cfg(feature = "read-stats")]
                read_stats::record_summary(false);
            }
        }
    }
    hit
}

#[allow(
    clippy::len_zero,
    reason = "len avoids the unmodeled Vec::is_empty extraction"
)]
pub(super) fn spill_locate_held<K, H, R: KeyRep<K, H>, V>(
    r: &Region<R, V>,
    sigma: Sigma,
    address: &TableAddress,
    carrier: &R,
    n: usize,
) -> SpillSite {
    let mut result = SpillSite::Grow;
    if r.slots.len() != 0 {
        let groups = r.slots.len() / n;
        if groups >= 2 && groups * n == r.slots.len() && groups.is_power_of_two() {
            let mut vacant = (false, 0);
            let mut found = (false, 0);
            let mask = groups - 1;
            let wanted = tag(address.place);
            let may_match = r.summary_maybe(address.place);
            let mut step = 0;
            while step < WINDOWS + groups && !found.0 && (may_match || !vacant.0) {
                let base = group(address.place, step, mask) * n;
                if may_match {
                    let mut hits = scan::match_tag(&r.tags[base..base + n], wanted);
                    while hits != 0 && !found.0 {
                        let lane = least_set_lane(hits);
                        let slot = base + lane;
                        if r.slots[slot].0 == sigma
                            && R::matches_held(sigma, &r.slots[slot].1, carrier)
                        {
                            found = (true, slot);
                        }
                        hits &= hits - 1;
                    }
                }
                if !vacant.0 {
                    let empty = scan::match_empty(&r.tags[base..base + n]);
                    if empty != 0 {
                        vacant = (true, base + least_set_lane(empty));
                    }
                }
                step += 1;
            }
            if found.0 {
                result = SpillSite::Found(found.1);
            } else if vacant.0 {
                result = SpillSite::Vacant(vacant.1);
            }
        }
    }
    result
}

/// Kani-only fixtures for the real `spill_read` body.
///
/// A whole-`Table` harness cannot decide this helper: its loops are bounded by a symbolic `Vec`
/// length and a data-dependent bitmask (`docs/evidence/kani-table-harness-audit.md`). These run the
/// real `spill_read`, the real `Region`, tags, summary and the real `Stored<i64>` full-key
/// comparison, on a fixed placement so the probe order is deterministic, over a symbolic query key.
/// Bounded at `n = 2`, four slots, two groups. Not a whole-`Table`, production-geometry, canonical
/// or universal-refinement result; the `Table` construction/preservation of these facts is a
/// separate blocked obligation.
#[cfg(kani)]
mod harnesses {
    use crate::table::keyrep::{KeyRep, Stored};
    use crate::table::{Region, TableAddress, WordHome};
    use crate::Sigma;

    use super::spill_read;

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

    /// Two groups of two. Group 0: lane 0 a same-fingerprint live slot whose key differs, lane 1
    /// vacant. Group 1: lane 0 the matching key, lane 1 a same-fingerprint live slot whose key
    /// differs. Placement `0` sends every prefix window to group 0, so group 1 is only reached in
    /// the cyclic sweep.
    fn fixture(stored2: i64, stored0: i64, stored3: i64) -> Region<Stored<i64>, i64> {
        let address = address();
        let mut region = Region::reserve_empty(LEN).expect("two groups of two");
        region.initialize(Sigma::Parabolic, &held(stored2, &address), &0);
        region.publish_live(
            0,
            (Sigma::Parabolic, held(stored0, &address), 41),
            address.place,
        );
        region.publish_live(
            2,
            (Sigma::Parabolic, held(stored2, &address), 7),
            address.place,
        );
        region.publish_live(
            3,
            (Sigma::Parabolic, held(stored3, &address), 11),
            address.place,
        );
        region
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn spill_read_returns_the_late_matching_slot() {
        let stored2: i64 = kani::any();
        let stored0: i64 = kani::any();
        let stored3: i64 = kani::any();
        kani::assume(stored0 != stored2 && stored3 != stored2);
        let region = fixture(stored2, stored0, stored3);
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &stored2,
                N,
            ),
            Some(2),
            "the slot-0 collision and the slot-1 vacancy must not stop the sweep to slot 2"
        );
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn spill_read_reports_absence_without_a_full_key_match() {
        let query: i64 = kani::any();
        let stored2: i64 = kani::any();
        let stored0: i64 = kani::any();
        let stored3: i64 = kani::any();
        kani::assume(query != stored2 && query != stored0 && query != stored3);
        let region = fixture(stored2, stored0, stored3);
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &query,
                N,
            ),
            None,
            "a fingerprint collision on every live lane is still absence without a full match"
        );
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn spill_read_returns_absence_on_a_summary_miss() {
        let stored2: i64 = kani::any();
        let stored0: i64 = kani::any();
        let stored3: i64 = kani::any();
        let region = fixture(stored2, stored0, stored3);
        let miss = TableAddress {
            place: 1 << 40,
            direct_word: None,
        };
        assert!(
            !region.summary_maybe(miss.place),
            "the miss address must not be recorded in the summary"
        );
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &miss,
                &stored2,
                N,
            ),
            None,
            "a summary miss certifies absence without a sweep"
        );
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn spill_read_sweeps_when_the_summary_is_unavailable() {
        let stored2: i64 = kani::any();
        let stored0: i64 = kani::any();
        let stored3: i64 = kani::any();
        kani::assume(stored0 != stored2 && stored3 != stored2);
        let mut region = fixture(stored2, stored0, stored3);
        region.summary.clear();
        assert_eq!(
            spill_read::<i64, WordHome, Stored<i64>, i64>(
                &region,
                Sigma::Parabolic,
                &address(),
                &stored2,
                N,
            ),
            Some(2),
            "an unavailable summary disables filtering; the sweep still finds the match"
        );
    }
}
