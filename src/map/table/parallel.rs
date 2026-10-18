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
//! Scoped parallel execution for the table's already-partitioned sequential kernels.
//!
//! `std::thread::scope` is deliberately confined to this unextracted executor boundary. The
//! partition, schedule, per-shard build, and probe/assembly mechanics remain separate from thread
//! realization; spawn/join memory semantics, panic behavior, and liveness are not refined here.
//!
//! Every project-owned container a scoped phase writes into - each worker's result buffer, the
//! joined-results sink, and the probe readback vectors - is reserved or allocated on the calling
//! thread before the first scatter worker, so a refusal that can return `Err` never follows a worker
//! start. Executor and runtime memory, and allocation inside a caller's `each` callback, are outside
//! this boundary.

use std::time::Instant;

use super::resources::try_reserve_exact;
use super::{
    apply_prepared_shard_placed, build_prepared_shard, slots_for, valid_n, Home, KeyRep, Merge,
    PayloadLocation, Region, StorageError, Table, BUILD_PARTITION_CROSSOVER, SHARDS,
};
use crate::cell::Sigma;
use crate::map::coord::address::FaceDomain;

/// Fills the caller's pre-reserved `values` sink with every scoped task's result, then propagates
/// the first panic outside the scope.
///
/// The sink is reserved by the caller before the tasks start, so joining adds no project-owned
/// allocation to the post-start phase.
fn join_into<T>(handles: Vec<std::thread::ScopedJoinHandle<'_, T>>, mut values: Vec<T>) -> Vec<T> {
    let mut panic = None;
    for handle in handles {
        match handle.join() {
            Ok(value) => values.push(value),
            Err(payload) if panic.is_none() => panic = Some(payload),
            Err(_) => {}
        }
    }
    if let Some(payload) = panic {
        std::panic::resume_unwind(payload);
    }
    values
}

/// Wall-clock nanoseconds spent in each phase of one parallel build or grouped probe.
///
/// Measurement only. The production entry points run the identical code and discard this; the
/// phase boundaries are the real calls, so a harness reads where the time went instead of
/// inferring it. Zero on a serial fallback.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseNanos {
    /// The default-face `(Sigma, K)` widening copy (`build_parallel` / `get_each_parallel`).
    pub qualify: u64,
    /// The checked chunk plan, scatter-scratch reservation, stable scatter and stable
    /// concatenation of the parallel partition.
    pub partition: u64,
    /// `Parts::is_intact`: the scatter validation pass.
    pub validate: u64,
    /// `Parts::loads` plus `Schedule::lpt`; now the shard schedule and worker bins derived from the
    /// scatter plan's loads, before the first scatter worker.
    pub schedule: u64,
    /// The serial reserve-all: every nonempty shard's checked region, the batch's aggregate locator
    /// reverse-link extent, every input entry's admitted handle, one placement-record buffer per
    /// nonempty shard, and every worker's outcome buffer and joined-results sink, reserved before
    /// any worker starts. The per-key directory path is not walked here; the cold
    /// [`Self::assemble_directory`] builds it after the workers join.
    pub reserve: u64,
    /// The scoped per-shard build or grouped probe.
    pub kernel: u64,
    /// Joining worker results into the table or the query-order output, including the parallel
    /// build's recorded-location finalization.
    pub assemble: u64,
}

impl PhaseNanos {
    pub(crate) fn absorb(&mut self, other: Self) {
        self.qualify += other.qualify;
        self.partition += other.partition;
        self.validate += other.validate;
        self.schedule += other.schedule;
        self.reserve += other.reserve;
        self.kernel += other.kernel;
        self.assemble += other.assemble;
    }

    /// The sum of every phase.
    #[must_use]
    pub fn total(self) -> u64 {
        self.qualify
            + self.partition
            + self.validate
            + self.schedule
            + self.reserve
            + self.kernel
            + self.assemble
    }
}

/// Elapsed nanoseconds since `start`, saturating at `u64::MAX`.
#[inline]
fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Deterministic one-shot reservation-failure injection for the serial phase.
///
/// This is an inherent seam, not a `#[cfg(test)]` one: the cross-worker zero-effects tests live in
/// `tests/parallel/main.rs`, an external integration crate that cannot observe `#[cfg(test)]` items.
/// The thread-local arm names one shard, one worker outcome buffer, one placement-record buffer or
/// one [`FinalizeArm`] refusal arm of the recorded-location finalization; the serial phase consumes
/// it at that reservation or check and returns `Err` before any hold, clone, merge, relocation or
/// worker start, and the finalization consumes it at its own arm with that arm's exact error. A
/// production call never arms it, and the arm is one-shot, so it cannot leak into a later operation
/// on the same thread.
#[doc(hidden)]
#[cfg(feature = "injection")]
pub mod reservation_failure {
    use std::cell::Cell;

    /// One checked refusal arm of [`Table::finalize_recorded_locations`].
    ///
    /// Arming a variant forces that arm's refusal at its own check on an otherwise valid build;
    /// the forced refusal returns the same `StorageError` the real arm returns, so a witness
    /// observes the arm's exact refusal semantics. Without the seam the arms are unreachable from
    /// a valid build by construction: the fill records exactly the entries whose handles the
    /// preassembly set.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FinalizeArm {
        /// A record's input index is out of range of the per-input handle table.
        IndexOutOfRange,
        /// A record's input index has no unconsumed handle.
        MissingHandle,
        /// A record's handle names no recorded link in the locator.
        LinkOutOfRange,
        /// The record pass left an unconsumed `Some` handle: the record set is incomplete.
        UnconsumedHandle,
    }

    thread_local! {
        static ARMED_SHARD: Cell<Option<usize>> = const { Cell::new(None) };
        static ARMED_OUTPUT: Cell<Option<usize>> = const { Cell::new(None) };
        static ARMED_RECORDS: Cell<Option<usize>> = const { Cell::new(None) };
        static ARMED_FINALIZE: Cell<Option<FinalizeArm>> = const { Cell::new(None) };
    }

    /// Arms a one-shot failure for the reservation of `shard` on the calling thread.
    pub fn arm(shard: usize) {
        ARMED_SHARD.with(|cell| cell.set(Some(shard)));
    }

    /// Arms a one-shot failure for finalization refusal arm `arm` on the calling thread.
    pub fn arm_finalize(arm: FinalizeArm) {
        ARMED_FINALIZE.with(|cell| cell.set(Some(arm)));
    }

    /// Arms a one-shot failure for the reservation of worker outcome buffer `bin` on the calling
    /// thread, before any worker starts.
    pub fn arm_output(bin: usize) {
        ARMED_OUTPUT.with(|cell| cell.set(Some(bin)));
    }

    /// Arms a one-shot failure for the reservation of shard `shard`'s placement-record buffer on
    /// the calling thread, before any worker starts.
    pub fn arm_records(shard: usize) {
        ARMED_RECORDS.with(|cell| cell.set(Some(shard)));
    }

    /// Clears every armed failure on the calling thread.
    pub fn disarm() {
        ARMED_SHARD.with(|cell| cell.set(None));
        ARMED_OUTPUT.with(|cell| cell.set(None));
        ARMED_RECORDS.with(|cell| cell.set(None));
        ARMED_FINALIZE.with(|cell| cell.set(None));
    }

    /// Consumes the arm exactly when `shard` matches; reports whether it fired.
    pub(crate) fn consume(shard: usize) -> bool {
        ARMED_SHARD.with(|cell| match cell.get() {
            Some(armed) if armed == shard => {
                cell.set(None);
                true
            }
            _ => false,
        })
    }

    /// Consumes the output-buffer arm exactly when `bin` matches; reports whether it fired.
    pub(crate) fn consume_output(bin: usize) -> bool {
        ARMED_OUTPUT.with(|cell| match cell.get() {
            Some(armed) if armed == bin => {
                cell.set(None);
                true
            }
            _ => false,
        })
    }

    /// Consumes the placement-record-buffer arm exactly when `shard` matches; reports whether it
    /// fired.
    pub(crate) fn consume_records(shard: usize) -> bool {
        ARMED_RECORDS.with(|cell| match cell.get() {
            Some(armed) if armed == shard => {
                cell.set(None);
                true
            }
            _ => false,
        })
    }

    /// Consumes the finalization arm exactly when it names `arm`; reports whether it fired.
    ///
    /// A mismatched variant stays armed, so the arm fires at its own check and exactly once: the
    /// finalization runs once per build on the calling thread.
    pub(crate) fn consume_finalize(arm: FinalizeArm) -> bool {
        ARMED_FINALIZE.with(|cell| match cell.get() {
            Some(armed) if armed == arm => {
                cell.set(None);
                true
            }
            _ => false,
        })
    }
}

/// Fills one already-reserved shard region from its stable partition runs, recording each placed
/// locator-face entry's `(input index, slot)` placement.
///
/// The region was reserved by the serial reserve-all before any worker started, so this body
/// performs no fallible allocation. It initializes the region (one `hold` and the default clone)
/// and applies each run in input order through [`apply_prepared_shard_placed`], returning the
/// region's live-tag count. Every placed entry records its slot - a repeat key records the slot it
/// was found at - except locator-free elliptic entries, so every record pairs with a `Some` handle
/// in the finalize. The caller's record buffer is pre-reserved to the shard's routed load, which is
/// an upper bound on these pushes.
#[allow(
    clippy::too_many_arguments,
    reason = "the fill signature mirrors the prepared shard core plus its shard and record sink"
)]
fn fill_reserved_shard<K, V, H, R, F, const N: usize>(
    region: &mut Region<R, V>,
    home: &H,
    merge: &F,
    default: &V,
    input: &[((H::Faces, K), V)],
    parts: &super::partition::Parts,
    shard_at: usize,
    records: &mut ShardRecords,
) -> Result<usize, StorageError>
where
    K: Clone,
    V: Clone,
    H: Home<K>,
    R: KeyRep<K, H>,
    F: Merge<V>,
{
    let first = parts
        .runs(shard_at)
        .find_map(|run| match run.first() {
            Some(value) => Some(*value),
            None => None,
        })
        .ok_or(StorageError::SizeOverflow)?;
    let ((face, key), _) = &input[first];
    let sigma = face.sigma();
    let address = Home::address(home, sigma, key);
    let held = R::hold(sigma, key, &address);
    region.initialize(sigma, &held, default);
    for run in parts.runs(shard_at) {
        for &index in run {
            let ((entry_face, key), value) = &input[index];
            let entry_sigma = entry_face.sigma();
            let address = Home::address(home, entry_sigma, key);
            let (_, slot) = apply_prepared_shard_placed::<K, V, H, R, F, N>(
                region,
                merge,
                default,
                *entry_face,
                key,
                value.clone(),
                &address,
            )?;
            if !matches!(entry_sigma, Sigma::Elliptic) {
                records.push((index, slot));
            }
        }
    }
    Ok(region.slots.iter().filter(|slot| slot.live()).count())
}

/// A shard's reserved region, or `None` for a zero-load shard.
type ReservedRegion<R, V> = Option<Region<R, V>>;

/// One worker-recorded placement: the input entry's index and the slot its payload occupies in its
/// shard's assembled region.
type PlacementRecord = (usize, usize);

/// One shard's pre-reserved placement-record buffer; empty for a zero-load shard.
type ShardRecords = Vec<PlacementRecord>;

/// One owned shard: its shard index, its reserved region and its placement-record buffer.
type OwnedShard<R, V> = (usize, ReservedRegion<R, V>, ShardRecords);

/// One worker bin: each owned shard carried with its reserved region and record buffer.
type WorkerRegions<R, V> = Vec<OwnedShard<R, V>>;

/// One worker's per-shard outcome: the shard index, its built or refused region, its live-tag count
/// and the placement records its fill recorded.
type ShardOutcome<R, V> = (
    usize,
    Result<Region<R, V>, StorageError>,
    usize,
    ShardRecords,
);

/// One worker's outcome buffer, reserved on the calling thread before the worker starts.
type WorkerOutcomes<R, V> = Vec<ShardOutcome<R, V>>;

/// The pre-reserved worker storage: one outcome buffer per bin plus the joined-results sink.
type WorkerStorage<R, V> = (Vec<WorkerOutcomes<R, V>>, Vec<WorkerOutcomes<R, V>>);

/// Serial reserve-all: reserves every nonempty shard's checked region before any worker starts.
///
/// Returns `None` for a zero-load shard; its empty region is built by the shared
/// [`build_prepared_shard`] entry on the worker, which reserves nothing and holds nothing for zero
/// load. A seam hit or a checked sizing/allocation refusal returns `Err` with no hold, clone,
/// merge, relocation or spawned worker.
pub(crate) fn reserve_all_shards<R: Clone, V: Clone, const N: usize>(
    loads: &[usize; SHARDS],
) -> Result<Vec<ReservedRegion<R, V>>, StorageError> {
    let mut reserved = Vec::with_capacity(SHARDS);
    // The first refusal is carried beside the index: no `?` or early return in the loop.
    let mut refusal = None;
    let mut s = 0;
    while s < SHARDS {
        #[cfg(feature = "injection")]
        let armed = refusal.is_none() && reservation_failure::consume(s);
        #[cfg(not(feature = "injection"))]
        let armed = false;
        if armed {
            refusal = Some(StorageError::AllocationFailed);
        }
        if refusal.is_none() {
            if loads[s] == 0 {
                reserved.push(None);
            } else {
                match slots_for(loads[s], N) {
                    Ok(slots) => match Region::reserve_empty(slots) {
                        Ok(region) => reserved.push(Some(region)),
                        Err(error) => refusal = Some(error),
                    },
                    Err(error) => refusal = Some(error),
                }
            }
        }
        s += 1;
    }
    match refusal {
        Some(error) => Err(error),
        None => Ok(reserved),
    }
}

/// Reserves the per-input terminal-handle table before any worker starts.
///
/// Entry `i` later holds the locator handle of input `kvs[i]`'s key: `Some` for every locator-face
/// entry (repeated keys share their handle) and `None` for the locator-free elliptic face. The
/// entries start `None` and are overwritten in order by the preassembly pass.
pub(crate) fn reserve_handle_table(len: usize) -> Result<Vec<Option<u32>>, StorageError> {
    let mut handles: Vec<Option<u32>> = Vec::new();
    try_reserve_exact(&mut handles, len)?;
    handles.resize(len, None);
    Ok(handles)
}

/// Serial reserve-all: one placement-record buffer per nonempty shard before any worker starts.
///
/// A nonempty shard's buffer carries up to `loads[shard]` `(input index, slot)` records - one push
/// per routed input entry at most, since a repeat key records on each of its input occurrences. A
/// zero-load shard carries no buffer. The buffers move with their shard's region into its one
/// worker. A seam hit or a reservation refusal returns `Err` with no worker started and no hold,
/// clone, merge or relocation run.
pub(crate) fn reserve_placement_records(
    loads: &[usize; SHARDS],
) -> Result<Vec<ShardRecords>, StorageError> {
    let mut record_bufs: Vec<ShardRecords> = Vec::new();
    try_reserve_exact(&mut record_bufs, SHARDS)?;
    // The first refusal is carried beside the index: no `?` or early return in the loop.
    let mut refusal = None;
    let mut s = 0;
    while s < SHARDS {
        #[cfg(feature = "injection")]
        let armed = refusal.is_none() && reservation_failure::consume_records(s);
        #[cfg(not(feature = "injection"))]
        let armed = false;
        if armed {
            refusal = Some(StorageError::AllocationFailed);
        }
        if refusal.is_none() {
            let mut records = Vec::new();
            if loads[s] > 0 {
                match try_reserve_exact(&mut records, loads[s]) {
                    Ok(()) => record_bufs.push(records),
                    Err(error) => refusal = Some(error),
                }
            } else {
                record_bufs.push(records);
            }
        }
        s += 1;
    }
    match refusal {
        Some(error) => Err(error),
        None => Ok(record_bufs),
    }
}

/// Derives the shard schedule from the scatter plan's loads and hands each reserved region and its
/// placement-record buffer to the one bin that owns its shard.
///
/// Runs on the calling thread before the first scatter worker; the plan's loads are the assembled
/// partition's per-shard counts, and `Schedule::is_complete` checks that the bins partition the
/// shards, so every region and record buffer is taken exactly once.
pub(crate) fn schedule_bins_with_regions<R, V>(
    loads: &[usize; SHARDS],
    tasks: usize,
    reserved: &mut [ReservedRegion<R, V>],
    record_bufs: &mut [ShardRecords],
) -> Result<Vec<WorkerRegions<R, V>>, StorageError> {
    let schedule = super::partition::Schedule::lpt(loads, tasks);
    if !schedule.is_complete() {
        return Err(StorageError::SizeOverflow);
    }
    // The schedule's bins are copied by index: `bins()` is an iterator adapter the extraction
    // fragment does not carry.
    let mut bins: Vec<Vec<usize>> = Vec::new();
    let mut worker = 0;
    while worker < schedule.len() {
        let bin = schedule.bin(worker);
        let mut row: Vec<usize> = Vec::new();
        let mut at = 0;
        while at < bin.len() {
            row.push(bin[at]);
            at += 1;
        }
        bins.push(row);
        worker += 1;
    }
    Ok(distribute_regions(&bins, reserved, record_bufs))
}

/// Reserves one outcome buffer per worker bin and the joined-results sink on the calling thread.
///
/// Runs before the first scatter worker: the injectable seam consumes at each bin before its
/// reservation, so a refusal starts no worker, and the workers only append to storage that already
/// exists.
pub(crate) fn reserve_worker_outcomes<R, V>(
    bins_with_regions: &[WorkerRegions<R, V>],
) -> Result<WorkerStorage<R, V>, StorageError> {
    let mut outcomes: Vec<WorkerOutcomes<R, V>> = Vec::new();
    try_reserve_exact(&mut outcomes, bins_with_regions.len())?;
    // The first refusal is carried beside the index: no `?` or early return in the loop.
    let mut refusal = None;
    let mut bin_index = 0;
    while bin_index < bins_with_regions.len() {
        #[cfg(feature = "injection")]
        let armed = refusal.is_none() && reservation_failure::consume_output(bin_index);
        #[cfg(not(feature = "injection"))]
        let armed = false;
        if armed {
            refusal = Some(StorageError::AllocationFailed);
        }
        if refusal.is_none() {
            let mut buffer: WorkerOutcomes<R, V> = Vec::new();
            match try_reserve_exact(&mut buffer, bins_with_regions[bin_index].len()) {
                Ok(()) => outcomes.push(buffer),
                Err(error) => refusal = Some(error),
            }
        }
        bin_index += 1;
    }
    if let Some(error) = refusal {
        return Err(error);
    }
    let mut sink: Vec<WorkerOutcomes<R, V>> = Vec::new();
    try_reserve_exact(&mut sink, outcomes.len())?;
    Ok((outcomes, sink))
}

/// Moves each reserved region and its placement-record buffer into the one worker bin that owns its
/// shard.
pub(crate) fn distribute_regions<R, V>(
    bins: &[Vec<usize>],
    reserved: &mut [ReservedRegion<R, V>],
    record_bufs: &mut [ShardRecords],
) -> Vec<WorkerRegions<R, V>> {
    // Index loops, not the iterator chain: the extraction fragment carries neither `into_iter`
    // closures nor `collect`.
    let mut regions: Vec<WorkerRegions<R, V>> = Vec::new();
    let mut worker = 0;
    while worker < bins.len() {
        let bin = &bins[worker];
        let mut owned: WorkerRegions<R, V> = Vec::new();
        let mut at = 0;
        while at < bin.len() {
            let shard_at = bin[at];
            owned.push((
                shard_at,
                reserved[shard_at].take(),
                core::mem::take(&mut record_bufs[shard_at]),
            ));
            at += 1;
        }
        regions.push(owned);
        worker += 1;
    }
    regions
}

impl<K, V, H, const N: usize, R> Table<K, V, H, N, R>
where
    K: Clone + Sync,
    V: Clone + Send + Sync,
    H: Home<K> + Sync,
    H::Faces: Send + Sync,
    R: KeyRep<K, H>,
    R: Send + Sync,
{
    /// Builds with deterministic LPT shard assignment using at most `workers` scoped tasks.
    ///
    /// A shard is built by exactly one task from its chunk-ordered runs. Thus each logical key's
    /// values retain input order even when shards are assigned to different workers.
    ///
    /// # Errors
    /// Returns a sizing or reservation error without returning a partially assembled table.
    pub fn build_at_parallel<F: Merge<V> + Sync>(
        merge: &F,
        default: V,
        kvs: &[((H::Faces, K), V)],
        home: H,
        workers: usize,
    ) -> Result<Self, StorageError> {
        Self::build_at_parallel_phased(merge, default, kvs, home, workers).map(|(table, _)| table)
    }

    /// [`Self::build_at_parallel`] with its per-phase wall-clock split.
    ///
    /// # Errors
    /// Returns a sizing or reservation error without returning a partially assembled table.
    #[doc(hidden)]
    pub fn build_at_parallel_phased<F: Merge<V> + Sync>(
        merge: &F,
        default: V,
        kvs: &[((H::Faces, K), V)],
        home: H,
        workers: usize,
    ) -> Result<(Self, PhaseNanos), StorageError> {
        let mut phases = PhaseNanos::default();
        let workers = workers.max(1);
        if workers == 1 || kvs.len() < BUILD_PARTITION_CROSSOVER {
            return Self::build_at_sequential(merge, default, kvs, home).map(|t| (t, phases));
        }
        if !valid_n(N) {
            return Err(StorageError::SizeOverflow);
        }
        let tasks = workers.min(kvs.len());
        if tasks == 1 {
            return Self::build_at_sequential(merge, default, kvs, home).map(|t| (t, phases));
        }
        // The table owns the reserved shard regions and the directory from here on. It is created
        // before the first worker so the directory reservation below precedes every hold, clone,
        // merge and relocation.
        let mut table = Self::new(default, home);
        // Checked chunk/load plan and scatter-scratch reservation on this thread, then the
        // table-shard reservation, the directory reservation, the shard schedule and every worker's
        // outcome buffer, all before the first scatter worker starts. A sizing or reservation
        // refusal here starts no worker, runs no hold, clone, merge or relocation and returns no
        // table. This supplies the cross-worker half of the lifecycle contract that
        // `build_prepared_shard` already satisfies per shard.
        let mark = Instant::now();
        let plan = super::partition::ScatterPlan::of(&table.home, kvs, tasks)?;
        let mut scratch = super::partition::ScatterScratch::reserve(&plan)?;
        phases.partition = elapsed_ns(mark);
        let loads = *plan.loads();
        let mark = Instant::now();
        let mut reserved = reserve_all_shards::<R, V, N>(&loads)?;
        phases.reserve = elapsed_ns(mark);
        // The recording structures and the directory: the per-input handle table and one
        // placement-record buffer per nonempty shard, then the aggregate locator extent for every
        // input entry, all before any worker starts. Each input entry's handle is recorded without
        // a per-key directory walk: its admitted reverse-link slot is written into the reserved
        // extent, and the cold `assemble_directory` resolves the directory after the workers join.
        let mark = Instant::now();
        let mut handles = reserve_handle_table(kvs.len())?;
        let mut record_bufs = reserve_placement_records(&loads)?;
        table.reserve_locator_batch(kvs.len())?;
        table.admit_parallel_handles(kvs, &mut handles)?;
        phases.reserve += elapsed_ns(mark);
        // The scatter plan's loads are the assembled partition's per-shard counts; derive the shard
        // schedule, the worker bins and every worker's outcome buffer on the calling thread before
        // the first scatter worker, so the whole reservation boundary - shard regions, directory
        // paths, recording buffers, schedule and outcome buffers - completes with no worker running.
        let mark = Instant::now();
        let bins_with_regions =
            schedule_bins_with_regions(&loads, tasks, &mut reserved, &mut record_bufs)?;
        phases.schedule = elapsed_ns(mark);
        let mark = Instant::now();
        let (outcomes, sink) = reserve_worker_outcomes(&bins_with_regions)?;
        // The finalize staging: one slot per shard outcome, filled as the joined regions are
        // installed and read by the finalization. Reserved here, so the post-worker passes add no
        // project-owned allocation.
        let mut placed: Vec<(usize, ShardRecords)> = Vec::new();
        try_reserve_exact(&mut placed, SHARDS)?;
        phases.reserve += elapsed_ns(mark);
        // Scatter each chunk into its private pre-sized shard runs, then concatenate the runs in
        // increasing input-chunk order into the final per-shard buffer. Neither scope allocates.
        let mark = Instant::now();
        scratch.scatter(&table.home, kvs, &plan);
        let parts = scratch.assemble(&plan);
        phases.partition += elapsed_ns(mark);
        let mark = Instant::now();
        if !parts.is_intact(&table.home, kvs) {
            return Err(StorageError::SizeOverflow);
        }
        // The schedule was derived from the plan's loads before the workers; the validated
        // assembled partition must carry exactly those loads.
        debug_assert_eq!(parts.loads(), loads);
        phases.validate = elapsed_ns(mark);
        let home_ref = &table.home;
        let default_ref = &table.default;
        let input = kvs;
        let parts_ref = &parts;
        let mark = Instant::now();
        let built = std::thread::scope(|scope| {
            let handles = bins_with_regions
                .into_iter()
                .zip(outcomes)
                .map(|(bin, mut mine)| {
                    scope.spawn(move || {
                        for (s, region, mut records) in bin {
                            // A zero-load shard has no reserved region and no record buffer; the
                            // shared per-shard builder returns the empty region without reserving
                            // or holding and places no entry, so the record buffer stays empty.
                            let outcome = match region {
                                Some(mut region) => fill_reserved_shard::<K, V, H, R, F, N>(
                                    &mut region,
                                    home_ref,
                                    merge,
                                    default_ref,
                                    input,
                                    parts_ref,
                                    s,
                                    &mut records,
                                )
                                .map(|live| (region, live)),
                                None => build_prepared_shard::<K, V, H, R, F, N>(
                                    home_ref,
                                    merge,
                                    default_ref,
                                    input,
                                    parts_ref,
                                    s,
                                    0,
                                )
                                .map(|region| (region, 0)),
                            };
                            // The live-tag count is one linear pass over the shard; running it
                            // here keeps it on the worker that already owns the region instead of
                            // on the serial assembly thread.
                            match outcome {
                                Ok((region, live)) => mine.push((s, Ok(region), live, records)),
                                Err(error) => mine.push((s, Err(error), 0, records)),
                            }
                        }
                        mine
                    })
                })
                .collect();
            join_into(handles, sink)
        });
        phases.kernel = elapsed_ns(mark);
        let mark = Instant::now();
        table.install_and_finalize(built, &mut handles, &mut placed)?;
        phases.assemble = elapsed_ns(mark);
        Ok((table, phases))
    }

    /// Installs every joined shard region, finalizes the worker-recorded payload locations and
    /// assembles the locator directory.
    ///
    /// Runs after every worker has joined. Each outcome's region moves into its shard slot and its
    /// placement records into the caller's pre-reserved staging; when all regions are installed,
    /// [`Self::finalize_recorded_locations`] consumes the recorded handles and then
    /// [`Self::assemble_directory`] links each live payload's complete path cold, resolving the
    /// reverse links the reserve phase admitted. The directory build is the only fallible part; the
    /// region install and the finalization add no reservation.
    fn install_and_finalize(
        &mut self,
        built: Vec<WorkerOutcomes<R, V>>,
        handles: &mut [Option<u32>],
        placed: &mut Vec<(usize, ShardRecords)>,
    ) -> Result<(), StorageError> {
        for worker in built {
            for (s, region, live, records) in worker {
                let region = region?;
                self.len += live;
                self.shards[s] = region;
                placed.push((s, records));
            }
        }
        self.finalize_recorded_locations(handles, placed)?;
        self.assemble_directory()
    }

    /// Installs every worker-recorded payload location as its input entry's reverse link.
    ///
    /// Runs after the joined regions are installed. It consumes the calling thread's handle table
    /// and the placement records the fill workers returned, setting each consumed handle's recorded
    /// location to its record's `Spill(shard, slot)`: no `Home::address`, directory walk, payload
    /// probe, reservation, clone or merge. A record whose input index is out of range, a record
    /// with no unconsumed handle (elliptic, missing or already recorded) or a handle naming no
    /// recorded link is a checked refusal, and an unconsumed `Some` handle left at the end is an
    /// incomplete record set: no case leaves a silent placeholder.
    ///
    /// Temporal `O(records + |handles|)` copied pairs, scalar comparisons and whole-pair link
    /// writes; spatial `O(1)` beyond the buffers the reserve phase already reserved.
    fn finalize_recorded_locations(
        &mut self,
        handles: &mut [Option<u32>],
        placed: &[(usize, ShardRecords)],
    ) -> Result<(), StorageError> {
        let mut at = 0;
        while at < placed.len() {
            let (shard, records) = &placed[at];
            let mut r = 0;
            while r < records.len() {
                let (input_index, slot) = records[r];
                // Each checked arm merges its one-shot injection variant into the real condition,
                // so the forced refusal takes the same `return Err` the real arm takes and a
                // witness observes the arm's exact refusal; without the feature the conditions
                // are the original ones.
                let out_of_range = input_index >= handles.len();
                #[cfg(feature = "injection")]
                let out_of_range = out_of_range
                    || reservation_failure::consume_finalize(
                        reservation_failure::FinalizeArm::IndexOutOfRange,
                    );
                if out_of_range {
                    return Err(StorageError::SizeOverflow);
                }
                let missing = handles[input_index].is_none();
                #[cfg(feature = "injection")]
                let missing = missing
                    || reservation_failure::consume_finalize(
                        reservation_failure::FinalizeArm::MissingHandle,
                    );
                if missing {
                    return Err(StorageError::AllocationFailed);
                }
                let handle = match handles[input_index] {
                    Some(handle) => handle,
                    None => return Err(StorageError::AllocationFailed),
                };
                let link_out_of_range = handle as usize >= self.locator.links.len();
                #[cfg(feature = "injection")]
                let link_out_of_range = link_out_of_range
                    || reservation_failure::consume_finalize(
                        reservation_failure::FinalizeArm::LinkOutOfRange,
                    );
                if link_out_of_range {
                    return Err(StorageError::AllocationFailed);
                }
                handles[input_index] = None;
                self.locator
                    .set_link_location(handle, PayloadLocation::Spill(*shard, slot));
                r += 1;
            }
            at += 1;
        }
        let mut i = 0;
        while i < handles.len() {
            let unconsumed = handles[i].is_some();
            #[cfg(feature = "injection")]
            let unconsumed = unconsumed
                || reservation_failure::consume_finalize(
                    reservation_failure::FinalizeArm::UnconsumedHandle,
                );
            if unconsumed {
                return Err(StorageError::AllocationFailed);
            }
            i += 1;
        }
        // The finalized registry is the admitted extent: a later merge reserves against its length.
        self.locator.admitted = self.locator.links.len();
        Ok(())
    }

    /// Default-face parallel build. Below the crossover and for one worker it uses the ordered
    /// sequential build path.
    ///
    /// # Errors
    /// Returns a sizing or reservation error without returning a partially assembled table.
    pub fn build_parallel<F: Merge<V> + Sync>(
        merge: &F,
        default: V,
        kvs: &[(K, V)],
        home: H,
        workers: usize,
    ) -> Result<Self, StorageError> {
        Self::build_parallel_phased(merge, default, kvs, home, workers).map(|(table, _)| table)
    }

    /// [`Self::build_parallel`] with its per-phase wall-clock split.
    ///
    /// # Errors
    /// Returns a sizing or reservation error without returning a partially assembled table.
    #[doc(hidden)]
    pub fn build_parallel_phased<F: Merge<V> + Sync>(
        merge: &F,
        default: V,
        kvs: &[(K, V)],
        home: H,
        workers: usize,
    ) -> Result<(Self, PhaseNanos), StorageError> {
        let mut phases = PhaseNanos::default();
        if workers == 1 || kvs.len() < BUILD_PARTITION_CROSSOVER {
            return Self::build(merge, default, kvs, home).map(|t| (t, phases));
        }
        let mark = Instant::now();
        let default_face = <H::Faces as FaceDomain<K, H>>::DEFAULT;
        let qualified: Vec<_> = kvs
            .iter()
            .map(|(key, value)| ((default_face, key.clone()), value.clone()))
            .collect();
        phases.qualify = elapsed_ns(mark);
        let (table, inner) =
            Self::build_at_parallel_phased(merge, default, &qualified, home, workers)?;
        phases.absorb(inner);
        Ok((table, phases))
    }

    /// Groups probes by shard but invokes `each` only after results are restored to query order.
    ///
    /// # Panics
    /// Propagates a panic from addressing, identity comparison, or `each`; a task panic is resumed
    /// only after every scoped task has joined.
    pub fn get_each_at_parallel<T, F>(
        &self,
        queries: &[(H::Faces, K)],
        workers: usize,
        each: F,
    ) -> Vec<T>
    where
        F: FnMut(&V) -> T,
    {
        self.get_each_at_parallel_phased(queries, workers, each).0
    }

    /// [`Self::get_each_at_parallel`] with its per-phase wall-clock split.
    ///
    /// # Panics
    /// Propagates a panic from addressing, identity comparison, or `each`; a task panic is resumed
    /// only after every scoped task has joined.
    #[doc(hidden)]
    pub fn get_each_at_parallel_phased<T, F>(
        &self,
        queries: &[(H::Faces, K)],
        workers: usize,
        mut each: F,
    ) -> (Vec<T>, PhaseNanos)
    where
        F: FnMut(&V) -> T,
    {
        let mut phases = PhaseNanos::default();
        let workers = workers.max(1);
        if workers == 1 || queries.len() < BUILD_PARTITION_CROSSOVER {
            let out = queries
                .iter()
                .map(|(face, key)| each(self.get_at(*face, key)))
                .collect();
            return (out, phases);
        }
        let mark = Instant::now();
        let grouped: Vec<_> = queries
            .iter()
            .map(|(face, key)| ((*face, key.clone()), ()))
            .collect();
        phases.qualify = elapsed_ns(mark);
        let (out, inner) = self.get_each_grouped_phased(&grouped, workers, each);
        phases.absorb(inner);
        (out, phases)
    }

    /// Grouped probe over an already `(Sigma, K)`-qualified stream.
    ///
    /// [`Self::get_each_at_parallel_phased`] and [`Self::get_each_parallel_phased`] qualify the
    /// default `Sigma` face differently; sharing this body means a key is cloned once for the
    /// default face instead of once per entry point.
    ///
    /// # Panics
    /// Propagates a panic from addressing, identity comparison, or `each`; a task panic is resumed
    /// only after every scoped task has joined.
    fn get_each_grouped_phased<T, F>(
        &self,
        grouped: &[((H::Faces, K), ())],
        workers: usize,
        mut each: F,
    ) -> (Vec<T>, PhaseNanos)
    where
        F: FnMut(&V) -> T,
    {
        let mut phases = PhaseNanos::default();
        let queries_len = grouped.len();
        let tasks = workers.min(queries_len);
        let mark = Instant::now();
        let parts = super::partition::Parts::of(&self.home, grouped, tasks);
        phases.partition = elapsed_ns(mark);
        let mark = Instant::now();
        if !parts.is_intact(&self.home, grouped) {
            let out = grouped
                .iter()
                .map(|((face, key), ())| each(self.get_at(*face, key)))
                .collect();
            return (out, phases);
        }
        phases.validate = elapsed_ns(mark);
        let mark = Instant::now();
        let loads = parts.loads();
        let schedule = super::partition::Schedule::lpt(&loads, tasks);
        let bins: Vec<Vec<usize>> = schedule.bins().map(ToOwned::to_owned).collect();
        phases.schedule = elapsed_ns(mark);
        // Reserve every worker's readback buffer, the joined-results sink, the query-order slot
        // vector and the output vector before the first probe worker starts: each probe produces
        // exactly one result, so the bin's routed load is the exact result count and the scoped
        // phase and readback add no project-owned container.
        let mut buffers: Vec<Vec<(usize, V)>> = Vec::with_capacity(bins.len());
        for bin in &bins {
            let reserve: usize = bin.iter().map(|&s| loads[s]).sum();
            buffers.push(Vec::with_capacity(reserve));
        }
        let probed_sink: Vec<Vec<(usize, V)>> = Vec::with_capacity(bins.len());
        let mut ordered: Vec<Option<V>> = Vec::with_capacity(queries_len);
        ordered.resize_with(queries_len, || None);
        let mut out: Vec<T> = Vec::with_capacity(queries_len);
        let parts_ref = &parts;
        let query_ref = grouped;
        let table_ref = self;
        let mark = Instant::now();
        let probed = std::thread::scope(|scope| {
            let handles = bins
                .iter()
                .zip(buffers)
                .map(|(bin, mut mine)| {
                    scope.spawn(move || {
                        for &s in bin {
                            for run in parts_ref.runs(s) {
                                for &index in run {
                                    let ((face, key), ()) = &query_ref[index];
                                    mine.push((index, table_ref.get_at(*face, key).clone()));
                                }
                            }
                        }
                        mine
                    })
                })
                .collect();
            join_into(handles, probed_sink)
        });
        phases.kernel = elapsed_ns(mark);
        let mark = Instant::now();
        for worker in probed {
            for (index, value) in worker {
                ordered[index] = Some(value);
            }
        }
        for value in ordered {
            out.push(each(&value.expect("complete grouped probe schedule")));
        }
        phases.assemble = elapsed_ns(mark);
        (out, phases)
    }

    /// Default-face grouped probe with query-order callback execution.
    pub fn get_each_parallel<T, F>(&self, queries: &[K], workers: usize, each: F) -> Vec<T>
    where
        F: FnMut(&V) -> T,
    {
        self.get_each_parallel_phased(queries, workers, each).0
    }

    /// [`Self::get_each_parallel`] with its per-phase wall-clock split.
    #[doc(hidden)]
    pub fn get_each_parallel_phased<T, F>(
        &self,
        queries: &[K],
        workers: usize,
        mut each: F,
    ) -> (Vec<T>, PhaseNanos)
    where
        F: FnMut(&V) -> T,
    {
        let mut phases = PhaseNanos::default();
        let workers = workers.max(1);
        if workers == 1 || queries.len() < BUILD_PARTITION_CROSSOVER {
            let out = queries.iter().map(|key| each(self.get(key))).collect();
            return (out, phases);
        }
        // Qualify straight into the grouped stream: the former path built a `(Sigma, K)` copy and
        // then a `((Sigma, K), ())` copy, cloning every key twice.
        let mark = Instant::now();
        let default_face = <H::Faces as FaceDomain<K, H>>::DEFAULT;
        let grouped: Vec<_> = queries
            .iter()
            .cloned()
            .map(|key| ((default_face, key), ()))
            .collect();
        phases.qualify = elapsed_ns(mark);
        let (out, inner) = self.get_each_grouped_phased(&grouped, workers, each);
        phases.absorb(inner);
        (out, phases)
    }
}

impl<H, const N: usize, R> Table<i64, i64, H, N, R>
where
    H: Home<i64> + Sync,
    H::Faces: Send + Sync,
    R: KeyRep<i64, H>,
    R: Send + Sync,
{
    /// Commutative grouped reduction over default-face probes.
    #[must_use]
    pub fn sum_parallel(&self, queries: &[i64], workers: usize) -> i64 {
        self.get_each_parallel(queries, workers, |value| *value)
            .into_iter()
            .sum()
    }
}
