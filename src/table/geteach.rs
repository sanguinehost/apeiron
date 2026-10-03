//! Batched multi-key read: each key's resident slot line is requested [`STAGE_DEPTH`] keys ahead of
//! the read that consumes it.
//!
//! Ported from `fold hashtrinity/src/lib.rs:746-778` (`get_each`, disassembled as
//! `probe_get_each`, 319 instructions at the pinned revision `4cc14471`, `src/lib.rs` sha256
//! `f8c8fd12fc2b7a69c0a045611d70e3a02fc43310e56c69de75e163341ed19e87`; pinned disassembly
//! `/home/socol/oc-tmp/ht-llvm/full.asm`, sha256
//! `b68a950e375e5516288236bf1a70a75d9abf32471149590838b3644a1a57fb97`).
//!
//! What transferred is the schedule, not the instruction count. The incumbent computes each key's
//! coordinate, requests its home line, and resolves the lookup `P` keys later, so one key's cache
//! miss overlaps the reads of the keys ahead of it. The mechanism is memory-level parallelism. That
//! benefit is `measured` for the incumbent and is a measurement here too; it is **not** a proved
//! claim, and `proof/corpus/Growable/Batched.lean` proves the refinement only.
//!
//! # What this port does not carry over
//!
//! The incumbent keeps the computed coordinate in a `P`-deep ring and resolves from the staged
//! value (`coords[i & (P - 1)]` at `fold hashtrinity/src/lib.rs:765`), so a key is addressed once.
//! Here every read is the ordinary [`Table::get_at`], which addresses the key itself, so the staged
//! value would be carried and never read: a ring with no reader is dead code. Dropping it leaves
//! exactly the transferred mechanism — the line request, issued `STAGE_DEPTH` keys early — and
//! loses the address reuse, which is a handful of arithmetic instructions against a cache miss
//! measured in hundreds of cycles.
//!
//! Resolving through [`Table::get_at`] is also what keeps this module free of a second copy of the
//! resident-read body. A local `resolve(address)` would have to repeat `get_at`'s resident arm, and
//! a copy of that body is a parallel implementation of the table's read path. The other way to
//! share it — extracting one `resolve(address)` that both call — would add a root to the
//! committed extraction and require `proof/ApeironWord.lean` to be regenerated, which is outside
//! this port's scope; `get_at` is therefore left byte-identical.
//!
//! # Why the staged read is the serial read
//!
//! Staging moves a line request forward; it does not change which slot a key resolves to. The read
//! for key `i` is the ordinary [`Table::get_at`] on `keys[i]` at `face`, and `&self` and `keys` are
//! shared for the whole loop, so no cell, counter or slot can change between the request and the
//! read. A line request has no effect on program semantics. That is the whole argument, and
//! `Growable.Batched` states its addressable content as a theorem.

use super::{Home, Table};
use crate::cell::Sigma;
use crate::coord::address::FaceDomain;

/// How many keys ahead a resident slot line is requested.
///
/// A power of two, so an advance is a mask rather than a divide. The incumbent's depth is
/// `P = 16` (`fold hashtrinity/src/lib.rs:750`). A core's fill-buffer width bounds how many misses
/// can be in flight, and 16 exceeds the ~14 loads / ~9 stores the incumbent's own comment reports
/// for one core.
pub const STAGE_DEPTH: usize = 16;

/// Inputs at or below this many keys skip the pipeline and read one key at a time.
///
/// The incumbent's `n <= P` arm (`fold hashtrinity/src/lib.rs:753`): below the pipeline depth a
/// prologue would request every key's line and then read them back in order, which is the serial
/// loop with a request pass bolted on.
pub const PIPELINE_MIN_KEYS: usize = STAGE_DEPTH;

/// A table at or below this many estimated bytes reads without staging.
///
/// The incumbent's `footprint_bytes() <= (1 << 19)` arm (`fold hashtrinity/src/lib.rs:753`, and
/// `fold hashtrinity/src/lib.rs:1092-1094` for the measure): a table that fits in cache has no
/// miss to overlap, so staging is pure overhead.
pub const CACHE_RESIDENT_BYTES: usize = 1 << 19;

/// Bytes charged per coordinate word in the footprint estimate.
///
/// A coordinate is a `u64`. The bound is named rather than `size_of::<u64>()`, because
/// `core::mem::size_of` is an unmodeled Aeneas external and the fail-closed audit rejects it.
const ASSUMED_COORD_BYTES: usize = 8;

impl<K, V, H: Home<K>, const N: usize, R: super::KeyRep<K, H>> Table<K, V, H, N, R> {
    /// Reads a stream of keys at one face, requesting each resident slot line ahead of its read.
    ///
    /// `visit(key, value)` runs exactly once per key, in the order of `keys`, and `value` is what
    /// [`Table::get_at`] returns for that `(face, key)`. There is no absence case: the read is
    /// total and answers the table default, as `get_at` does.
    ///
    /// # Complexity
    ///
    /// Temporal `O(n)` reads plus `O(n)` line requests, counted in slot visits and requests: the
    /// prologue issues [`STAGE_DEPTH`] requests and each of the `n` iterations issues one read and
    /// at most one further request, so the requests total `STAGE_DEPTH + (n - STAGE_DEPTH) = n`
    /// once `n > STAGE_DEPTH`. The fallback arm is `n` reads and no requests. Spatial `O(1)` — no
    /// ring is retained.
    pub fn get_each_at<'a, F>(&'a self, face: H::Faces, keys: &'a [K], mut visit: F)
    where
        F: FnMut(&'a K, &'a V),
    {
        let n = keys.len();
        if n <= PIPELINE_MIN_KEYS || self.cache_resident() {
            for key in keys {
                visit(key, self.get_at(face, key));
            }
            return;
        }
        let sigma = face.sigma();
        for key in &keys[..STAGE_DEPTH] {
            self.request_resident_line(sigma, key);
        }
        for i in 0..n {
            visit(&keys[i], self.get_at(face, &keys[i]));
            let ahead = i + STAGE_DEPTH;
            if ahead < n {
                self.request_resident_line(sigma, &keys[ahead]);
            }
        }
    }

    /// Reads a stream of keys at the home's default face.
    ///
    /// The default-face batch of [`Table::get_each_at`], matching [`Table::get`]'s default face.
    #[inline]
    pub fn get_each<'a, F>(&'a self, keys: &'a [K], visit: F)
    where
        F: FnMut(&'a K, &'a V),
    {
        self.get_each_at(<H::Faces as FaceDomain<K, H>>::DEFAULT, keys, visit);
    }

    /// Requests the cache line holding the certified resident slot this key addresses.
    ///
    /// `Flat::index` yields `shard * lanes + lane` with `shard < SHARDS` and `lane < lanes`, and a
    /// reserved resident area holds exactly `lanes * SHARDS` slots, so the `Some` arm indexes inside
    /// `slots`. The request is advisory: the pointer is handed to the architecture's hint and never
    /// dereferenced, so it is safe to issue before the read that consumes it, and a line evicted by
    /// then costs a dropped hint. A `None` arm addresses nothing and issues nothing, which is the
    /// absent-key case.
    ///
    /// Temporal and spatial `O(1)`: one address computation, one `Option` word test, one lane range
    /// test, one address multiply-add and one line request.
    #[inline]
    fn request_resident_line(&self, sigma: Sigma, key: &K) {
        let address = Home::address(&self.home, sigma, key);
        if let Some(index) = address
            .direct_word
            .and_then(|word| self.resident.index(word))
        {
            #[cfg(test)]
            request_trace::record();
            request_slot_line(self.resident.slots.as_ptr().wrapping_add(index));
        }
    }

    /// Whether the whole table is small enough that staging cannot pay.
    ///
    /// The incumbent compares one threshold against its whole-table coordinate array. Here the
    /// certified resident area is charged at `Flat::byte_len_for` — the same estimate the
    /// reservation budget uses (`src/table.rs:469`) — and every spill region at
    /// `ASSUMED_COORD_BYTES + ASSUMED_SLOT_BYTES` per slot. A `None` resident estimate means the
    /// checked product overflowed, which the reservation rules reject; `usize::MAX` there keeps the
    /// answer conservative and total.
    ///
    /// The resident term alone can never reach [`CACHE_RESIDENT_BYTES`], because
    /// `DEFAULT_BUDGET_BYTES = 256 * 1024` caps the resident area at `32 * 256 * 32` bytes. The
    /// spill regions carry a large table past the threshold, so they are charged; without that term
    /// this predicate would be vacuously true and the staging arm unreachable.
    fn cache_resident(&self) -> bool {
        self.footprint_bytes() <= CACHE_RESIDENT_BYTES
    }

    /// The whole-table footprint estimate the threshold is compared against.
    ///
    /// Saturating throughout: an overflow saturates to `usize::MAX`, which is the conservative
    /// answer for both arms, so the estimate stays total.
    fn footprint_bytes(&self) -> usize {
        let resident = super::Flat::<R, V>::byte_len_for(self.resident.lanes).unwrap_or(usize::MAX);
        let slot_bytes = ASSUMED_COORD_BYTES + super::ASSUMED_SLOT_BYTES;
        let spill = self.shards.iter().fold(0usize, |acc, region| {
            acc.saturating_add(region.coords.len().saturating_mul(slot_bytes))
        });
        resident.saturating_add(spill)
    }
}

/// Requests one cache line without reading it.
///
/// Advisory by construction: the pointer is only passed to the architecture's prefetch hint, which
/// has no observable effect on program semantics. On a target without the hint this is the empty
/// function and the batched read degrades to an unstaged serial read with no overlap.
#[inline]
fn request_slot_line<T>(slot: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` accepts any address and never faults; it does not read through
    // `slot`, so the pointer needs no provenance beyond being derived from a live allocation. The
    // line may be stale when the request is issued, which is the point.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(slot.cast::<i8>());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = slot;
}

#[cfg(test)]
mod request_trace {
    use core::cell::Cell;

    // Count of resident slot lines the batched read actually requested on this thread.
    //
    // Inert outside `cfg(test)`, as `table::state::publication_trace` is. It exists so the
    // staging-arm tests can assert they took the staged path rather than the fallback: without it
    // a test would pass identically whether or not any line was ever requested.
    thread_local! {
        static REQUESTS: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn clear() {
        REQUESTS.with(|count| count.set(0));
    }

    pub(super) fn record() {
        REQUESTS.with(|count| count.set(count.get() + 1));
    }

    pub(super) fn count() -> usize {
        REQUESTS.with(Cell::get)
    }
}

#[cfg(test)]
mod tests {
    use super::request_trace;
    use super::{Table, CACHE_RESIDENT_BYTES, PIPELINE_MIN_KEYS};
    use crate::cell::Sigma;
    use crate::table::WordHome;

    /// The lowest key whose lane is outside every resident width the budget admits.
    ///
    /// `DEFAULT_BUDGET_BYTES = 256 * 1024` caps the resident area at `32 * 256 * 32` bytes, so
    /// `Flat::index` admits a lane below `32` and `Flat::index` computes `lane = word >> 8`. A word
    /// at or above `32 * 256` therefore has `lane >= 32` and routes to spill.
    const FIRST_SPILL_KEY: i64 = 32 * 256;

    /// Live entries per spill region, enough to force the region past capacity `32`.
    ///
    /// A region grows only when a probe cycle finds every slot occupied (`src/table.rs:494`), so the
    /// 33rd entry in a region relocates it to `slots_for(33, 16) = 128`. The footprint estimate is
    /// then `256 * 128 * (8 + 32) = 1310720` bytes, above the `524288`-byte threshold.
    const SPILL_LOAD_PER_SHARD: i64 = 33;

    /// Resident keys, all with lane `0`, so the certified resident area admits them.
    const RESIDENT_KEYS: i64 = 256;

    /// Every face the default `Sigma` domain admits.
    const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

    /// Builds a table whose spill regions carry it past [`CACHE_RESIDENT_BYTES`], so the staged arm
    /// is reachable.
    ///
    /// [`table::shard`] routes a certified word on its low byte, and `FIRST_SPILL_KEY + offset` has
    /// low byte `offset`, so [`SPILL_LOAD_PER_SHARD`] keys per `offset` gives each of the `256`
    /// regions that load. See the constant for why `33` and not `18`.
    fn table_past_the_cache_threshold() -> Table<i64, i64, WordHome, 16> {
        let mut table = Table::<i64, i64, WordHome, 16>::new(-1, WordHome);
        // Keys `0..RESIDENT_KEYS` all have lane `0`, so this grows the resident area to lane `1`
        // and they stay findable there. Without them every key in the table spills, there is no
        // resident slot to request, and the staged arm would issue no request at all.
        for key in 0..RESIDENT_KEYS {
            table
                .insert_with_at(&crate::table::WordMerge, Sigma::Parabolic, key, key * 2)
                .expect("a fresh insert into an empty table is admitted");
        }
        for offset in 0..256 {
            for step in 0..SPILL_LOAD_PER_SHARD {
                let key = FIRST_SPILL_KEY + offset + 256 * step;
                table
                    .insert_with_at(&crate::table::WordMerge, Sigma::Parabolic, key, key * 2)
                    .expect("a fresh insert into an empty table is admitted");
            }
        }
        assert!(
            !table.cache_resident(),
            "the staging arm is unreachable: estimated footprint {} bytes is not above the \
             threshold {CACHE_RESIDENT_BYTES}",
            table.footprint_bytes()
        );
        table
    }

    /// The batched read is the serial read, index by index, at every face.
    ///
    /// This is the test the batched-address mutation control reddens. It compares against
    /// `get_at` per key rather than against a fixed table of expectations, so it holds for every
    /// value the table can hold.
    #[test]
    fn batched_read_matches_the_serial_read_at_every_face() {
        let table = table_past_the_cache_threshold();
        // Resident keys, spill keys and absent keys, interleaved, and more than the pipeline depth
        // so the staged arm is the one under test.
        let mut keys: Vec<i64> = (0..24).collect();
        keys.extend(FIRST_SPILL_KEY..FIRST_SPILL_KEY + 24);
        keys.extend((FIRST_SPILL_KEY + 100_000)..(FIRST_SPILL_KEY + 100_024));
        assert!(keys.len() > PIPELINE_MIN_KEYS);

        for face in FACES {
            request_trace::clear();
            let mut batched: Vec<i64> = Vec::with_capacity(keys.len());
            table.get_each_at(face, &keys, |_key, value| batched.push(*value));
            let requests = request_trace::count();
            let serial: Vec<i64> = keys.iter().map(|key| *table.get_at(face, key)).collect();
            assert_eq!(
                batched, serial,
                "the batched read diverged from the serial read at face {face:?}"
            );
            assert!(
                requests > 0,
                "no resident slot line was requested at face {face:?}: the fallback arm ran, so \
                 this test did not exercise the staged arm"
            );
        }
    }

    /// The default-face batch agrees with [`Table::get`] key by key, and visits in query order.
    #[test]
    fn default_face_batch_matches_get_and_preserves_order() {
        let table = table_past_the_cache_threshold();
        let keys: Vec<i64> = (0..40)
            .chain(FIRST_SPILL_KEY..FIRST_SPILL_KEY + 40)
            .collect();
        let mut seen: Vec<(i64, i64)> = Vec::with_capacity(keys.len());
        table.get_each(&keys, |key, value| seen.push((*key, *value)));
        let expected: Vec<(i64, i64)> = keys.iter().map(|key| (*key, *table.get(key))).collect();
        assert_eq!(seen, expected);
    }

    /// A short batch takes the fallback arm and issues no request, so the staged arm never claims a
    /// benefit it did not take.
    #[test]
    fn a_batch_at_or_below_the_depth_reads_without_staging() {
        let table = table_past_the_cache_threshold();
        let keys: Vec<i64> =
            (0..i64::try_from(PIPELINE_MIN_KEYS).expect("depth fits in i64")).collect();
        request_trace::clear();
        let mut batched: Vec<i64> = Vec::with_capacity(keys.len());
        table.get_each(&keys, |_key, value| batched.push(*value));
        assert_eq!(
            request_trace::count(),
            0,
            "the fallback arm issued a line request"
        );
        let serial: Vec<i64> = keys.iter().map(|key| *table.get(key)).collect();
        assert_eq!(batched, serial);
    }
}
