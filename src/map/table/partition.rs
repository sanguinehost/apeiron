#![allow(
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    clippy::manual_map,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::len_zero,
    clippy::bool_to_int_with_if,
    reason = "extraction-facing scalar forms: the pinned extractor emits an `axiom` for \
              `is_empty`, `div_ceil`, `Option::map`, `let-else`, integer `TryFrom` and the \
              narrowing casts, so the explicit forms are kept and these lints stand"
)]
//! Stable key partitioning for the sequential batch-build executor.
//!
//! The scatter writes one contiguous buffer: shard `s` owns
//! `buf[ends[s]..ends[s + 1]]`, in input order.  Counting sort is stable, so a
//! logical `(Sigma, K)` sees exactly its original value subsequence.  The whole
//! partition is one allocation and two linear passes; the output does not depend
//! on the worker count the caller later schedules over the shards.

use crate::map::coord::address::FaceDomain;

use super::resources::try_reserve_exact;
use super::{shard, Home, StorageError};

/// Shard-address width in bits.
///
/// The low byte selects one of 256 shards. The fingerprint/descent filter reads higher placement
/// bits and is followed by full identity comparison.
pub(crate) const SHARD_BITS: u32 = 8;

/// Number of home shards. A power of two, `1 << SHARD_BITS`.
pub(crate) const SHARDS: usize = 1 << SHARD_BITS;

/// Mask selecting the shard bits of a direct word.
pub(crate) const SHARD_MASK: usize = SHARDS - 1;

const _: () = {
    assert!(SHARDS > 0);
    assert!(SHARDS.is_power_of_two());
    assert!(SHARD_MASK == SHARDS - 1);
};
const _: () = assert!(SHARD_BITS == 8 && SHARDS == 256 && SHARD_MASK == 255);

/// Input indices routed into one contiguous per-shard buffer.
///
/// Shard `s` owns `buf[ends[s]..ends[s + 1]]`; the shard ranges tile `buf`
/// without gaps, so a shard's build reads one contiguous run and the whole
/// partition is a single allocation rather than one buffer per worker chunk.
pub(crate) struct Parts {
    buf: Vec<usize>,
    ends: [usize; SHARDS + 1],
}

impl Parts {
    /// Count, prefix, and stable-scatter the whole input into one per-shard buffer.
    ///
    /// `_workers` is the caller's later schedule width, not a partition parameter:
    /// the output is worker-independent, so a shard's run does not depend on which
    /// worker builds it.  Temporal O(n) over two linear passes; spatial one `usize`
    /// buffer of the input length plus the fixed `ends` array.
    pub(crate) fn of<K, V, H: Home<K>>(
        home: &H,
        input: &[((H::Faces, K), V)],
        _workers: usize,
    ) -> Self {
        if input.len() == 0 {
            return Self {
                buf: Vec::new(),
                ends: [0; SHARDS + 1],
            };
        }
        let (buf, ends) = partition_chunk(home, input, 0, input.len());
        Self { buf, ends }
    }

    /// Temporal O(SHARDS); spatial O(1).
    #[must_use]
    pub(crate) fn loads(&self) -> [usize; SHARDS] {
        let mut loads = [0; SHARDS];
        let mut s = 0;
        while s < SHARDS {
            loads[s] = self.ends[s + 1] - self.ends[s];
            s += 1;
        }
        loads
    }

    /// The one contiguous run of a shard, in input order.
    pub(crate) fn runs(&self, shard: usize) -> impl Iterator<Item = &[usize]> {
        core::iter::once(&self.buf[self.ends[shard]..self.ends[shard + 1]])
    }

    /// The partition is one chunk, so a caller's chunk loop runs once.
    pub(crate) fn len(&self) -> usize {
        if self.buf.len() == 0 {
            0
        } else {
            1
        }
    }

    /// The single chunk's run for `shard`; any other chunk is empty.
    pub(crate) fn run(&self, worker: usize, shard: usize) -> &[usize] {
        let start = if worker == 0 { self.ends[shard] } else { 0 };
        let end = if worker == 0 { self.ends[shard + 1] } else { 0 };
        &self.buf[start..end]
    }

    pub(crate) fn first(&self, shard: usize) -> Option<usize> {
        let lo = self.ends[shard];
        if lo < self.ends[shard + 1] {
            match self.buf.get(lo) {
                Some(value) => Some(*value),
                None => None,
            }
        } else {
            None
        }
    }

    /// Rejects malformed, duplicated, omitted, or misrouted scatter output.
    ///
    /// Temporal O(`input.len() + SHARDS`) in comparisons and `home.address` calls;
    /// spatial O(1) auxiliary.  The endpoints must tile `buf` and `buf` must have the
    /// input length; every shard range must be monotone and in range; within a shard
    /// the indexes must be in `input` and strictly ascending; each index must reroute
    /// by a fresh `home.address` to its own shard.  Strict local order makes the
    /// indexes injective, and the length equation makes the in-range injection a
    /// bijection, so no index is duplicated or omitted.  No loop contains `return`,
    /// `break` or `continue`, so the body is Aeneas-shaped.
    pub(crate) fn is_intact<K, V, H: Home<K>>(
        &self,
        home: &H,
        input: &[((H::Faces, K), V)],
    ) -> bool {
        // Each conjunct is bound on its own line and combined by single-exit accumulation, so
        // every statement reads one pair of fields: the multi-borrow `&&` initializer has no
        // Aeneas context to match.
        let starts_at_zero = self.ends[0] == 0;
        let ends_tile_the_buffer = self.ends[SHARDS] == self.buf.len();
        let buffer_matches_input = self.buf.len() == input.len();
        let mut intact = starts_at_zero;
        intact = intact && ends_tile_the_buffer;
        intact = intact && buffer_matches_input;
        let mut s = 0;
        while s < SHARDS && intact {
            let lo = self.ends[s];
            let hi = self.ends[s + 1];
            if lo > hi || hi > self.buf.len() {
                intact = false;
            } else {
                let mut at = lo;
                let mut prev = 0usize;
                while at < hi && intact {
                    let index = self.buf[at];
                    let first = at == lo;
                    let ascending = index > prev;
                    let ordered = first | ascending;
                    let matched = partition_entry_matches(home, input, index, s, ordered);
                    prev = index;
                    intact = intact && matched;
                    at += 1;
                }
            }
            s += 1;
        }
        intact
    }
}

/// Checks bounds, strict local order, and fresh routing with at most one `Home::address` call.
fn partition_entry_matches<K, V, H: Home<K>>(
    home: &H,
    input: &[((H::Faces, K), V)],
    index: usize,
    expected_shard: usize,
    ordered: bool,
) -> bool {
    if !ordered || index >= input.len() {
        return false;
    }
    let ((face, key), _) = &input[index];
    let address = Home::address(home, face.sigma(), key);
    shard(&address) == expected_shard
}

/// Count, prefix-sum, stable-scatter one contiguous slice into per-shard runs.
///
/// The two passes recompute the address rather than cache a per-element shard:
/// the cache is one more byte written and read per element on a memory-bound
/// loop, and the historical measurement was 12% slower with it.  Temporal
/// O(`last - first`) over two linear passes; spatial one `usize` per element of
/// the slice plus the two fixed 256-entry cursors.
fn partition_chunk<K, V, H: Home<K>>(
    home: &H,
    input: &[((H::Faces, K), V)],
    first: usize,
    last: usize,
) -> (Vec<usize>, [usize; SHARDS + 1]) {
    let mut counts = [0usize; SHARDS];
    let mut i = first;
    while i < last {
        let ((face, key), _) = &input[i];
        let address = Home::address(home, face.sigma(), key);
        let routed = shard(&address);
        counts[routed] += 1;
        i += 1;
    }
    let mut ends = [0usize; SHARDS + 1];
    let mut cursor = [0usize; SHARDS];
    let mut total = 0;
    let mut s = 0;
    while s < SHARDS {
        ends[s] = total;
        cursor[s] = total;
        total += counts[s];
        s += 1;
    }
    ends[SHARDS] = total;
    let mut buf = vec![0usize; last - first];
    i = first;
    while i < last {
        let ((face, key), _) = &input[i];
        let address = Home::address(home, face.sigma(), key);
        let s = shard(&address);
        buf[cursor[s]] = i;
        cursor[s] += 1;
        i += 1;
    }
    (buf, ends)
}

/// Checked contiguous chunk geometry and per-chunk shard prefixes for the parallel scatter.
///
/// The calling thread builds the plan before any worker starts. `starts` holds the `p + 1`
/// contiguous chunk boundaries; `local_starts` holds, for chunk `c`, its `SHARDS + 1`-entry shard
/// prefix, so `local_start(c, s)..local_end(c, s)` is shard `s`'s run in that chunk; `loads` and
/// `ends` are the global shard load and endpoint vectors. Every boundary, prefix and load addition
/// is checked, so a sizing overflow refuses before any reservation or worker start.
///
/// Temporal O(`input.len() + p * SHARDS`) counted in routing calls and prefix steps; spatial
/// `p * (SHARDS + 1)` words. The chunk geometry is balanced: the first `n % p` chunks carry one
/// extra index.
pub(crate) struct ScatterPlan {
    starts: Vec<usize>,
    local_starts: Vec<usize>,
    loads: [usize; SHARDS],
    ends: [usize; SHARDS + 1],
}

/// The balanced chunk boundary `floor(c * n / p)` without overflow.
///
/// `(n / p) * c <= n` and `min(c, n % p) <= n`, so the sum never exceeds `n`. Temporal O(1);
/// spatial O(1).
fn chunk_boundary(n: usize, c: usize, p: usize) -> usize {
    let base = n / p;
    let extra = n % p;
    c * base + c.min(extra)
}

impl ScatterPlan {
    /// Counts each chunk's routed loads and derives the checked local prefixes and global ends.
    ///
    /// `tasks` is clamped to `1..=input.len()`; an empty input plans one empty chunk. Temporal
    /// O(`input.len() + p * SHARDS`) counted in routing calls and prefix steps; spatial
    /// `p * (SHARDS + 1)` words, all reserved here on the calling thread.
    ///
    /// # Errors
    /// Returns [`StorageError::SizeOverflow`] when a boundary, prefix or load addition overflows,
    /// and [`StorageError::AllocationFailed`] when the plan metadata cannot be reserved.
    #[allow(
        clippy::many_single_char_names,
        reason = "the plan names mirror the input length, task count, chunk, index and shard axes"
    )]
    pub(crate) fn of<K, V, H: Home<K>>(
        home: &H,
        input: &[((H::Faces, K), V)],
        tasks: usize,
    ) -> Result<Self, StorageError> {
        let n = input.len();
        let p = tasks.max(1).min(n.max(1));
        let stride = SHARDS + 1;
        let boundaries = p.checked_add(1).ok_or(StorageError::SizeOverflow)?;
        let metadata = p.checked_mul(stride).ok_or(StorageError::SizeOverflow)?;
        let mut starts = Vec::new();
        try_reserve_exact(&mut starts, boundaries)?;
        starts.resize(boundaries, 0);
        let mut local_starts = Vec::new();
        try_reserve_exact(&mut local_starts, metadata)?;
        local_starts.resize(metadata, 0);
        let mut c = 0;
        while c <= p {
            starts[c] = chunk_boundary(n, c, p);
            c += 1;
        }
        let mut loads = [0usize; SHARDS];
        c = 0;
        while c < p {
            let first = starts[c];
            let last = starts[c + 1];
            let mut counts = [0usize; SHARDS];
            let mut i = first;
            while i < last {
                let ((face, key), _) = &input[i];
                let address = Home::address(home, face.sigma(), key);
                counts[shard(&address)] += 1;
                i += 1;
            }
            let base = c * stride;
            let mut total = 0usize;
            let mut s = 0;
            while s < SHARDS {
                local_starts[base + s] = total;
                total = total
                    .checked_add(counts[s])
                    .ok_or(StorageError::SizeOverflow)?;
                loads[s] = loads[s]
                    .checked_add(counts[s])
                    .ok_or(StorageError::SizeOverflow)?;
                s += 1;
            }
            local_starts[base + SHARDS] = total;
            c += 1;
        }
        let mut ends = [0usize; SHARDS + 1];
        let mut total = 0usize;
        let mut s = 0;
        while s < SHARDS {
            ends[s] = total;
            total = total
                .checked_add(loads[s])
                .ok_or(StorageError::SizeOverflow)?;
            s += 1;
        }
        ends[SHARDS] = total;
        Ok(Self {
            starts,
            local_starts,
            loads,
            ends,
        })
    }

    /// The number of contiguous input chunks.
    pub(crate) fn chunk_count(&self) -> usize {
        self.starts.len() - 1
    }

    /// The first global input index of chunk `c`.
    pub(crate) fn chunk_start(&self, c: usize) -> usize {
        self.starts[c]
    }

    /// One past the last global input index of chunk `c`.
    pub(crate) fn chunk_end(&self, c: usize) -> usize {
        self.starts[c + 1]
    }

    /// The number of input indexes in chunk `c`.
    pub(crate) fn chunk_len(&self, c: usize) -> usize {
        self.starts[c + 1] - self.starts[c]
    }

    /// The offset of shard `s`'s run within chunk `c`'s private buffer.
    pub(crate) fn local_start(&self, c: usize, s: usize) -> usize {
        self.local_starts[c * (SHARDS + 1) + s]
    }

    /// One past the offset of shard `s`'s run within chunk `c`'s private buffer.
    pub(crate) fn local_end(&self, c: usize, s: usize) -> usize {
        self.local_starts[c * (SHARDS + 1) + s + 1]
    }

    /// The global per-shard routed load vector.
    pub(crate) fn loads(&self) -> &[usize; SHARDS] {
        &self.loads
    }

    /// The global per-shard endpoint vector.
    pub(crate) fn ends(&self) -> &[usize; SHARDS + 1] {
        &self.ends
    }

    /// The planned input length.
    pub(crate) fn len(&self) -> usize {
        self.starts[self.starts.len() - 1]
    }
}

/// Pre-sized private per-chunk index buffers and the final destination buffer.
///
/// [`Self::reserve`] allocates both on the calling thread before the first scatter worker; the
/// worker bodies only index these initialized slices. Spatial `n` private index words plus `n`
/// destination words; with the plan's `p * (SHARDS + 1)` prefix words the whole scatter scratch is
/// `O(n + 257p)` words beyond the final `Parts`, with no key or value copy and no per-element heap
/// allocation.
pub(crate) struct ScatterScratch {
    private: Vec<usize>,
    buf: Vec<usize>,
}

impl ScatterScratch {
    /// Reserves and materializes the private and destination buffers.
    ///
    /// # Errors
    /// Returns [`StorageError::AllocationFailed`] when either reservation is refused; no worker has
    /// started and no table exists.
    pub(crate) fn reserve(plan: &ScatterPlan) -> Result<Self, StorageError> {
        let n = plan.len();
        let mut private = Vec::new();
        try_reserve_exact(&mut private, n)?;
        private.resize(n, 0);
        let mut buf = Vec::new();
        try_reserve_exact(&mut buf, n)?;
        buf.resize(n, 0);
        Ok(Self { private, buf })
    }

    /// Scatters every chunk into its private pre-sized shard runs with one scoped task per chunk.
    ///
    /// Each task revisits its chunk in increasing global input-index order and writes into the
    /// disjoint private slice the caller reserved; no task allocates, holds, clones or merges.
    /// Temporal O(`input.len()`) counted in routing calls and index writes; spatial O(`SHARDS`)
    /// stack words per task.
    pub(crate) fn scatter<K, V, H>(
        &mut self,
        home: &H,
        input: &[((H::Faces, K), V)],
        plan: &ScatterPlan,
    ) where
        K: Sync,
        V: Sync,
        H: Home<K> + Sync,
        H::Faces: Send + Sync,
    {
        let p = plan.chunk_count();
        let private = &mut self.private;
        let home_ref = home;
        let input_ref = input;
        let plan_ref = plan;
        std::thread::scope(|scope| {
            let mut rest: &mut [usize] = private.as_mut_slice();
            let mut c = 0;
            while c < p {
                let len = plan_ref.chunk_len(c);
                let (chunk, tail) = rest.split_at_mut(len);
                rest = tail;
                scope.spawn(move || {
                    let mut cursor = [0usize; SHARDS];
                    let mut s = 0;
                    while s < SHARDS {
                        cursor[s] = plan_ref.local_start(c, s);
                        s += 1;
                    }
                    let mut i = plan_ref.chunk_start(c);
                    let last = plan_ref.chunk_end(c);
                    while i < last {
                        let ((face, key), _) = &input_ref[i];
                        let address = Home::address(home_ref, face.sigma(), key);
                        let s = shard(&address);
                        chunk[cursor[s]] = i;
                        cursor[s] += 1;
                        i += 1;
                    }
                });
                c += 1;
            }
        });
    }

    /// Concatenates each shard's local runs in increasing input-chunk order into the final buffer.
    ///
    /// After every scatter task joins, the destination ranges are split into at most `p` disjoint
    /// initialized groups; each scoped task copies its shards' runs in chunk order and allocates
    /// nothing. Temporal O(`input.len()`) counted in index copies; spatial O(`SHARDS`) stack words
    /// per task.
    #[must_use]
    #[allow(
        clippy::many_single_char_names,
        reason = "the assembly names mirror the chunk, group, shard and index axes"
    )]
    pub(crate) fn assemble(self, plan: &ScatterPlan) -> Parts {
        let Self { private, mut buf } = self;
        let p = plan.chunk_count();
        let groups = p.min(SHARDS);
        let ends = *plan.ends();
        let private_ref = &private;
        let plan_ref = plan;
        std::thread::scope(|scope| {
            let mut rest: &mut [usize] = &mut buf;
            let mut g = 0;
            while g < groups {
                let lo_shard = g * SHARDS / groups;
                let hi_shard = (g + 1) * SHARDS / groups;
                let len = ends[hi_shard] - ends[lo_shard];
                let (dest, tail) = rest.split_at_mut(len);
                rest = tail;
                scope.spawn(move || {
                    let base = ends[lo_shard];
                    let mut s = lo_shard;
                    while s < hi_shard {
                        let mut at = ends[s] - base;
                        let mut c = 0;
                        while c < p {
                            let lo = plan_ref.local_start(c, s);
                            let hi = plan_ref.local_end(c, s);
                            let chunk_base = plan_ref.chunk_start(c);
                            let mut k = lo;
                            while k < hi {
                                dest[at] = private_ref[chunk_base + k];
                                at += 1;
                                k += 1;
                            }
                            c += 1;
                        }
                        s += 1;
                    }
                });
                g += 1;
            }
        });
        Parts { buf, ends }
    }
}

/// A deterministic LPT assignment of every shard to exactly one worker.
pub(crate) struct Schedule {
    bins: Vec<Vec<usize>>,
}

impl Schedule {
    #[must_use]
    pub(crate) fn lpt(loads: &[usize; SHARDS], workers: usize) -> Self {
        let workers = if workers == 0 { 1 } else { workers };
        let mut order = Vec::new();
        let mut shard = 0;
        while shard < SHARDS {
            order.push(shard);
            shard += 1;
        }
        let mut first = 0;
        while first < SHARDS {
            let mut best = first;
            let mut candidate = first + 1;
            while candidate < SHARDS {
                let candidate_shard = order[candidate];
                let best_shard = order[best];
                if loads[candidate_shard] > loads[best_shard]
                    || (loads[candidate_shard] == loads[best_shard] && candidate_shard < best_shard)
                {
                    best = candidate;
                }
                candidate += 1;
            }
            if best != first {
                order.swap(first, best);
            }
            first += 1;
        }
        let mut bins = Vec::new();
        let mut filled: Vec<usize> = Vec::new();
        let mut worker = 0;
        while worker < workers {
            bins.push(Vec::new());
            filled.push(0);
            worker += 1;
        }
        let mut order_at = 0;
        while order_at < order.len() {
            let s = order[order_at];
            let mut chosen = 0;
            let mut worker = 1;
            while worker < workers {
                if filled[worker] < filled[chosen]
                    || (filled[worker] == filled[chosen] && worker < chosen)
                {
                    chosen = worker;
                }
                worker += 1;
            }
            bins[chosen].push(s);
            filled[chosen] = filled[chosen].saturating_add(loads[s]);
            order_at += 1;
        }
        Self { bins }
    }

    pub(crate) fn bins(&self) -> impl Iterator<Item = &[usize]> {
        self.bins.iter().map(Vec::as_slice)
    }

    pub(crate) fn len(&self) -> usize {
        self.bins.len()
    }

    pub(crate) fn bin(&self, worker: usize) -> &[usize] {
        &self.bins[worker]
    }

    /// Rejects a schedule that omits or assigns any shard more than once.
    #[must_use]
    pub(crate) fn is_complete(&self) -> bool {
        let mut seen = [false; SHARDS];
        let mut complete = true;
        let mut worker = 0;
        while worker < self.bins.len() && complete {
            let bin = &self.bins[worker];
            let mut at = 0;
            while at < bin.len() && complete {
                let s = bin[at];
                if s >= SHARDS || seen[s] {
                    complete = false;
                } else {
                    seen[s] = true;
                }
                at += 1;
            }
            worker += 1;
        }
        let mut shard = 0;
        while shard < SHARDS && complete {
            if !seen[shard] {
                complete = false;
            }
            shard += 1;
        }
        complete
    }
}

#[cfg(test)]
mod tests {
    use super::{Parts, ScatterPlan, ScatterScratch, Schedule, SHARDS};
    use crate::map::coord::address::{
        AddressValue, EllipticPacking, FaceCoord, FaceHome, FaceResidueCapability, LocatorSigma,
        ParabolicDual, Residue, H3_40,
    };
    use crate::map::coord::{spread, Packed};
    use crate::map::table::{shard, Home, ReplacementMerge, Table, TableAddress};
    use crate::Sigma;

    /// The named inline-test home: the physical low-byte placement plus the explicit I3-A
    /// semantic capability. Its face map is derived from the key, never from the placement.
    struct HomeByLowByte;

    impl Home<i64> for HomeByLowByte {
        fn home(&self, key: &i64) -> u64 {
            key.cast_unsigned()
        }

        /// Placement-only: reproduces the former default (`home`-derived place, no direct word).
        fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
            TableAddress {
                place: self.home(key),
                direct_word: None,
            }
        }
    }

    impl FaceResidueCapability<i64> for HomeByLowByte {
        fn parabolic_dual(&self, key: &i64) -> ParabolicDual {
            ParabolicDual::new(spread(key.cast_unsigned()).place, 0)
        }

        fn hyperbolic_scalar(&self, profile: H3_40, key: &i64) -> Residue {
            Residue::reduce(profile, key.cast_unsigned())
        }
    }

    impl EllipticPacking<i64> for HomeByLowByte {
        fn elliptic_pack(&self, key: &i64) -> u64 {
            <i64 as Packed>::packed(key)
        }

        fn elliptic_unpack(&self, word: u64) -> Option<i64> {
            Some(<i64 as Packed>::unpack(word))
        }
    }

    impl FaceCoord<i64> for HomeByLowByte {
        type Faces = Sigma;
    }

    impl FaceHome<i64> for HomeByLowByte {}

    /// A locator-only home: `Faces = LocatorSigma`, so `EllipticPacking` is not implemented and the
    /// elliptic face cannot be named on any of this table's public paths.
    #[derive(Clone, Copy)]
    struct LocatorOnlyHome;

    impl FaceResidueCapability<i64> for LocatorOnlyHome {
        fn parabolic_dual(&self, key: &i64) -> ParabolicDual {
            ParabolicDual::new(spread(key.cast_unsigned()).place, 0)
        }

        fn hyperbolic_scalar(&self, profile: H3_40, key: &i64) -> Residue {
            Residue::reduce(profile, key.cast_unsigned())
        }
    }

    impl FaceCoord<i64> for LocatorOnlyHome {
        type Faces = LocatorSigma;
    }

    impl FaceHome<i64> for LocatorOnlyHome {}

    impl Home<i64> for LocatorOnlyHome {
        fn home(&self, key: &i64) -> u64 {
            key.cast_unsigned()
        }

        /// Placement-only: reproduces the former default (`home`-derived place, no direct word).
        fn address(&self, _sigma: Sigma, key: &i64) -> TableAddress {
            TableAddress {
                place: self.home(key),
                direct_word: None,
            }
        }
    }

    /// A thread-local allocation counter, so `Parts::of` can be checked for a
    /// constant allocation count rather than one allocation per element.
    mod counting {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static ALLOCS: Cell<usize> = const { Cell::new(0) };
        }

        struct Counting;

        // SAFETY: every method forwards to `System` unchanged; the counter is a
        // thread-local `Cell<usize>` with no destructor, so counting cannot
        // recurse and cannot run during thread teardown.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                ALLOCS.with(|c| c.set(c.get() + 1));
                unsafe { System.alloc(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                unsafe { System.dealloc(ptr, layout) }
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                ALLOCS.with(|c| c.set(c.get() + 1));
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }

        #[global_allocator]
        static GLOBAL: Counting = Counting;

        pub(super) fn count<T>(f: impl FnOnce() -> T) -> (T, usize) {
            let before = ALLOCS.with(Cell::get);
            let out = f();
            let after = ALLOCS.with(Cell::get);
            (out, after - before)
        }
    }

    /// `n` keys over 251 shards, so most shards hold several indices.
    fn mixed(n: i64) -> Vec<((Sigma, i64), i64)> {
        (0..n).map(|i| ((Sigma::Parabolic, i % 251), i)).collect()
    }

    #[test]
    fn every_input_index_appears_exactly_once_in_output() {
        let input = mixed(1000);
        let parts = Parts::of(&HomeByLowByte, &input, 7);
        let mut indices = Vec::with_capacity(input.len());
        for s in 0..SHARDS {
            for run in parts.runs(s) {
                indices.extend_from_slice(run);
            }
        }
        assert_eq!(indices.len(), input.len());
        indices.sort_unstable();
        assert_eq!(indices, (0..input.len()).collect::<Vec<_>>());
        assert!(parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn runs_preserve_input_order_within_each_shard() {
        let input = [
            ((Sigma::Parabolic, 1), 10),
            ((Sigma::Parabolic, 2), 20),
            ((Sigma::Parabolic, 1), 11),
            ((Sigma::Parabolic, 2), 21),
        ];
        let parts = Parts::of(&HomeByLowByte, &input, 2);
        let indices: Vec<_> = parts.runs(1).flatten().copied().collect();
        assert_eq!(indices, [0, 2]);
        for s in 0..SHARDS {
            let run: Vec<usize> = parts.runs(s).flatten().copied().collect();
            assert!(run.windows(2).all(|pair| pair[0] < pair[1]));
        }
        assert!(parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn shard_sizes_match_the_loads_and_lpt_bins_partition_the_shards() {
        let input = mixed(512);
        let parts = Parts::of(&HomeByLowByte, &input, 8);
        let loads = parts.loads();
        let mut routed = [0usize; SHARDS];
        for ((sigma, key), _) in &input {
            let address = Home::address(&HomeByLowByte, *sigma, key);
            routed[crate::map::table::shard(&address)] += 1;
        }
        assert_eq!(loads, routed);
        for (s, &load) in loads.iter().enumerate() {
            let run_len: usize = parts.runs(s).map(<[usize]>::len).sum();
            assert_eq!(run_len, load, "shard {s} run must match its LPT load");
        }
        let schedule = Schedule::lpt(&loads, 8);
        assert!(schedule.is_complete());
        let bin_total: usize = schedule
            .bins()
            .flat_map(|bin| bin.iter())
            .map(|&s| loads[s])
            .sum();
        assert_eq!(bin_total, input.len());
    }

    #[test]
    fn partition_is_one_contiguous_buffer_not_one_allocation_per_element() {
        for n in [64usize, 4096] {
            let input = mixed(i64::try_from(n).expect("test size fits i64"));
            let (parts, allocations) = counting::count(|| Parts::of(&HomeByLowByte, &input, 4));
            assert!(
                allocations <= 1,
                "n={n}: {allocations} allocations for {n} elements"
            );
            assert_eq!(parts.buf.len(), n);
            assert_eq!(parts.ends[0], 0);
            assert_eq!(parts.ends[SHARDS], n);
            assert_eq!(parts.len(), 1);
            assert!(parts.run(1, 0).is_empty());
        }
    }

    /// The production two-phase scatter over `tasks` chunks, assembled back into one `Parts`.
    fn parallel_parts(input: &[((Sigma, i64), i64)], tasks: usize) -> Parts {
        let home = HomeByLowByte;
        let plan = ScatterPlan::of(&home, input, tasks).expect("scatter plan fits");
        let mut scratch = ScatterScratch::reserve(&plan).expect("scatter scratch fits");
        scratch.scatter(&home, input, &plan);
        scratch.assemble(&plan)
    }

    #[test]
    fn parallel_scatter_equals_serial_parts_across_worker_counts() {
        let cases = [Vec::new(), mixed(1), mixed(64), mixed(4097)];
        for input in &cases {
            let serial = Parts::of(&HomeByLowByte, input, 1);
            for tasks in [1usize, 2, 3, 7, 16, 64, 300] {
                let parallel = parallel_parts(input, tasks);
                assert_eq!(
                    parallel.buf,
                    serial.buf,
                    "n={} tasks={tasks}: index buffer differs",
                    input.len()
                );
                assert_eq!(
                    parallel.ends,
                    serial.ends,
                    "n={} tasks={tasks}: endpoints differ",
                    input.len()
                );
            }
        }
    }

    #[test]
    fn parallel_scatter_preserves_chunk_order_for_a_single_hot_shard() {
        // Every key routes to shard 7, so its one run is the whole input in input order and a
        // chunk-order concatenation error changes it.
        let input: Vec<((Sigma, i64), i64)> = (0..4097)
            .map(|i| ((Sigma::Parabolic, 7 + 256 * i), i))
            .collect();
        let serial = Parts::of(&HomeByLowByte, &input, 1);
        assert_eq!(serial.ends[8] - serial.ends[7], input.len());
        for tasks in [2usize, 3, 7, 16, 64, 300] {
            let parallel = parallel_parts(&input, tasks);
            assert_eq!(
                parallel.buf, serial.buf,
                "tasks={tasks}: hot-shard order differs"
            );
            assert_eq!(
                parallel.ends, serial.ends,
                "tasks={tasks}: endpoints differ"
            );
            assert!(parallel.is_intact(&HomeByLowByte, &input));
        }
    }

    #[test]
    fn parallel_scatter_matches_serial_for_interleaved_repeated_keys() {
        // 32 hot keys interleaved over the whole input, so each shard's run spans every chunk.
        let input: Vec<((Sigma, i64), i64)> =
            (0..8192).map(|i| ((Sigma::Parabolic, i % 32), i)).collect();
        let serial = Parts::of(&HomeByLowByte, &input, 1);
        for tasks in [2usize, 3, 7, 16, 64, 300] {
            let parallel = parallel_parts(&input, tasks);
            assert_eq!(
                parallel.buf, serial.buf,
                "tasks={tasks}: repeated-key order differs"
            );
            assert_eq!(
                parallel.ends, serial.ends,
                "tasks={tasks}: endpoints differ"
            );
            assert!(parallel.is_intact(&HomeByLowByte, &input));
        }
    }

    #[test]
    fn parallel_scatter_output_rejects_a_same_shard_reversal() {
        // A shard with at least two indexes whose run is reversed must fail the strict local order
        // check, exactly as the serial output does.
        let input = mixed(4097);
        let mut parts = parallel_parts(&input, 4);
        let lo = parts.ends[1];
        let hi = parts.ends[2];
        assert!(hi - lo >= 2, "shard 1 must carry at least two indexes");
        parts.buf[lo..hi].reverse();
        assert!(!parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn partition_integrity_rejects_a_duplicate_assignment() {
        let input = [((Sigma::Parabolic, 1), 10), ((Sigma::Parabolic, 2), 20)];
        let mut parts = Parts::of(&HomeByLowByte, &input, 1);
        parts.buf[parts.ends[2]] = 0;
        assert!(!parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn partition_integrity_rejects_an_omitted_assignment() {
        let input = mixed(64);
        let mut parts = Parts::of(&HomeByLowByte, &input, 1);
        parts.buf.pop();
        assert!(!parts.is_intact(&HomeByLowByte, &input));
    }

    /// Keys 1 and 257 both route to shard 1, so its run holds two indexes.
    fn two_in_one_shard() -> [((Sigma, i64), i64); 2] {
        [((Sigma::Parabolic, 1), 10), ((Sigma::Parabolic, 257), 20)]
    }

    /// The named test home's semantic face is derived from the key, not from the physical
    /// placement. The two fixture keys keep their one-shard physical collision while their
    /// semantic faces separate them; a mutation that routes the semantic face through the
    /// placement leaves the partition tests green and turns these assertions red.
    #[test]
    fn named_test_home_semantic_face_is_not_the_physical_placement() {
        let home = HomeByLowByte;
        let profile = H3_40::new().expect("the frozen H3_40 token validates");
        let [((_, one), _), ((_, two_fifty_seven), _)] = two_in_one_shard();

        // The physical collision fixture is unchanged: both keys route to one shard.
        assert_eq!(
            shard(&Home::address(&home, Sigma::Parabolic, &one)),
            shard(&Home::address(&home, Sigma::Parabolic, &two_fifty_seven)),
            "the two fixture keys must keep their one-shard physical collision"
        );

        // The semantic face is the key's specified encoding, not the placement word.
        for key in [one, two_fifty_seven] {
            let word = key.cast_unsigned();
            assert_eq!(
                FaceCoord::coord(&home, profile, Sigma::Elliptic, &key).value(),
                AddressValue::EllipticWord(word),
                "elliptic face must be the exact packed word"
            );
            assert_eq!(
                FaceCoord::coord(&home, profile, Sigma::Parabolic, &key).value(),
                AddressValue::ParabolicWord(spread(word).place),
                "parabolic face must be the spreading fold, not the placement"
            );
            assert_eq!(
                FaceCoord::coord(&home, profile, Sigma::Hyperbolic, &key).value(),
                AddressValue::HyperbolicResidue(profile, Residue::reduce(profile, word)),
                "hyperbolic face must be the canonical residue"
            );
        }
        assert_ne!(
            FaceCoord::coord(&home, profile, Sigma::Parabolic, &one),
            FaceCoord::coord(&home, profile, Sigma::Parabolic, &two_fifty_seven),
            "the semantic parabolic face separates the physically colliding keys"
        );
    }

    #[test]
    fn partition_integrity_rejects_a_same_shard_duplicate() {
        let input = two_in_one_shard();
        let mut parts = Parts::of(&HomeByLowByte, &input, 1);
        let lo = parts.ends[1];
        assert_eq!(parts.ends[2] - lo, 2, "both keys must share shard 1");
        parts.buf[lo + 1] = parts.buf[lo];
        assert!(!parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn partition_integrity_rejects_a_same_shard_reversal() {
        let input = two_in_one_shard();
        let mut parts = Parts::of(&HomeByLowByte, &input, 1);
        let lo = parts.ends[1];
        assert_eq!(parts.ends[2] - lo, 2, "both keys must share shard 1");
        parts.buf.swap(lo, lo + 1);
        assert!(!parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn partition_integrity_rejects_a_misrouted_index() {
        // Keys 1 and 2 route to shards 1 and 2; swapping the indexes puts each in
        // the other's shard while both stay in range.
        let input = [((Sigma::Parabolic, 1), 10), ((Sigma::Parabolic, 2), 20)];
        let mut parts = Parts::of(&HomeByLowByte, &input, 1);
        parts.buf.swap(0, 1);
        assert!(!parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn partition_integrity_rejects_out_of_range_bounds() {
        // `ends[SHARDS]` still equals `buf.len()`, but shard 1's end exceeds it.
        let input = [((Sigma::Parabolic, 1), 10), ((Sigma::Parabolic, 2), 20)];
        let mut parts = Parts::of(&HomeByLowByte, &input, 1);
        parts.ends[2] = parts.buf.len() + 1;
        assert!(!parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn partition_integrity_rejects_nonmonotone_bounds() {
        // Keys 0, 1, 2 route to shards 0, 1, 2; shard 1's end falls below its start.
        let input = mixed(3);
        let mut parts = Parts::of(&HomeByLowByte, &input, 1);
        parts.ends[2] = 0;
        assert!(!parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn partition_integrity_accepts_a_single_shard_load_skew() {
        // Every key routes to shard 7, so its run is the whole buffer in input
        // order and every other shard is empty.
        let input: Vec<((Sigma, i64), i64)> = (0..64)
            .map(|i| ((Sigma::Parabolic, 7 + 256 * i), i))
            .collect();
        let parts = Parts::of(&HomeByLowByte, &input, 4);
        assert_eq!(parts.loads()[7], 64);
        assert!(parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn partition_integrity_accepts_clustered_repeated_keys() {
        // A scene-like clustered workload: 32 hot keys, each repeated and
        // interleaved, so most shards hold several strictly ascending indexes.
        let input: Vec<((Sigma, i64), i64)> =
            (0..512).map(|i| ((Sigma::Parabolic, i % 32), i)).collect();
        let parts = Parts::of(&HomeByLowByte, &input, 8);
        assert!(parts.is_intact(&HomeByLowByte, &input));
    }

    #[test]
    fn lpt_is_load_ordered_and_deterministic() {
        let mut loads = [0usize; SHARDS];
        loads[3] = 100;
        loads[9] = 80;
        loads[1] = 70;
        let first: Vec<Vec<_>> = Schedule::lpt(&loads, 2)
            .bins()
            .map(ToOwned::to_owned)
            .collect();
        let second: Vec<Vec<_>> = Schedule::lpt(&loads, 2)
            .bins()
            .map(ToOwned::to_owned)
            .collect();
        assert_eq!(first, second);
        assert_eq!(first[0][0], 3);
        assert_eq!(first[1][0], 9);
        let duplicate = Schedule {
            bins: vec![vec![0], vec![0]],
        };
        assert!(!duplicate.is_complete());
    }

    /// Same key, three faces: each `(Sigma, K)` is a distinct representative, so the serial
    /// readback under one face must not return another face's value. Erasing the token (for
    /// example Hyperbolic as Parabolic) collides the two inserts and reddens this test.
    #[test]
    fn mixed_face_serial_roundtrip_distinguishes_faces() {
        let mut table = Table::<i64, i64, HomeByLowByte, 16>::new(0, HomeByLowByte);
        table
            .insert_with_at(&ReplacementMerge, Sigma::Parabolic, 7, 41)
            .expect("parabolic insert fits");
        table
            .insert_with_at(&ReplacementMerge, Sigma::Hyperbolic, 7, 42)
            .expect("hyperbolic insert fits");
        table
            .insert_with_at(&ReplacementMerge, Sigma::Elliptic, 7, 43)
            .expect("elliptic insert fits");
        assert_eq!(
            table.len(),
            3,
            "three faces of one key are three representatives"
        );
        assert_eq!(*table.get_at(Sigma::Parabolic, &7), 41);
        assert_eq!(*table.get_at(Sigma::Hyperbolic, &7), 42);
        assert_eq!(*table.get_at(Sigma::Elliptic, &7), 43);
        assert_eq!(*table.get(&7), 41, "the default face is parabolic");
    }

    /// The parallel `_at` entry points carry the same face tokens through partition, scatter,
    /// reservation and assembly. Above the crossover the workers run, so a token dropped on a
    /// worker collides the faces and this readback rejects.
    #[test]
    fn mixed_face_parallel_roundtrip_distinguishes_faces() {
        const KEYS: i64 = 1500;
        let count = usize::try_from(KEYS).expect("test size fits usize") * 3;
        let mut input: Vec<((Sigma, i64), i64)> = Vec::with_capacity(count);
        let mut key = 0i64;
        while key < KEYS {
            input.push(((Sigma::Parabolic, key), 1));
            input.push(((Sigma::Hyperbolic, key), 2));
            input.push(((Sigma::Elliptic, key), 3));
            key += 1;
        }
        let table = Table::<i64, i64, HomeByLowByte, 16>::build_at_parallel(
            &ReplacementMerge,
            0,
            &input,
            HomeByLowByte,
            8,
        )
        .expect("mixed-face parallel build fits");
        assert_eq!(table.len(), count);
        let queries: Vec<(Sigma, i64)> =
            input.iter().map(|((face, key), _)| (*face, *key)).collect();
        let values = table.get_each_at_parallel(&queries, 8, |value| *value);
        for (at, ((face, key), _)) in input.iter().enumerate() {
            let expected = match face {
                Sigma::Parabolic => 1,
                Sigma::Hyperbolic => 2,
                Sigma::Elliptic => 3,
            };
            assert_eq!(
                values[at], expected,
                "face {face:?} key {key} parallel readback"
            );
        }
    }

    /// The locator-only domain admits Parabolic and Hyperbolic on the same public paths; the
    /// compile-fail doctest on `Table::get_at` is the elliptic refusal. A token erased to one face
    /// collides these two and this readback rejects.
    #[test]
    fn locator_only_serial_and_parallel_roundtrip_distinguishes_faces() {
        const KEYS: i64 = 2100;
        let mut table = Table::<i64, i64, LocatorOnlyHome, 16>::new(0, LocatorOnlyHome);
        table
            .insert_with_at(&ReplacementMerge, LocatorSigma::Parabolic, 7, 41)
            .expect("parabolic insert fits");
        table
            .insert_with_at(&ReplacementMerge, LocatorSigma::Hyperbolic, 7, 42)
            .expect("hyperbolic insert fits");
        assert_eq!(table.len(), 2);
        assert_eq!(*table.get_at(LocatorSigma::Parabolic, &7), 41);
        assert_eq!(*table.get_at(LocatorSigma::Hyperbolic, &7), 42);
        assert_eq!(*table.get(&7), 41, "the locator-only default is parabolic");

        let count = usize::try_from(KEYS).expect("test size fits usize") * 2;
        let mut input: Vec<((LocatorSigma, i64), i64)> = Vec::with_capacity(count);
        let mut key = 0i64;
        while key < KEYS {
            input.push(((LocatorSigma::Parabolic, key), 1));
            input.push(((LocatorSigma::Hyperbolic, key), 2));
            key += 1;
        }
        let built = Table::<i64, i64, LocatorOnlyHome, 16>::build_at_parallel(
            &ReplacementMerge,
            0,
            &input,
            LocatorOnlyHome,
            8,
        )
        .expect("locator-only parallel build fits");
        assert_eq!(built.len(), count);
        let queries: Vec<(LocatorSigma, i64)> =
            input.iter().map(|((face, key), _)| (*face, *key)).collect();
        let values = built.get_each_at_parallel(&queries, 8, |value| *value);
        for (at, ((face, key), _)) in input.iter().enumerate() {
            let expected = match face {
                LocatorSigma::Parabolic => 1,
                LocatorSigma::Hyperbolic => 2,
            };
            assert_eq!(
                values[at], expected,
                "face {face:?} key {key} parallel readback"
            );
        }
    }
}
