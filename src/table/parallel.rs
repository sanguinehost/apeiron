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

use std::time::Instant;

use super::{
    apply_prepared_shard, build_prepared_shard, slots_for, valid_n, Home, KeyRep, Merge, Region,
    StorageError, Table, BUILD_PARTITION_CROSSOVER, SHARDS,
};
use crate::coord::address::FaceDomain;

/// Join every scoped task before propagating the first panic outside the scope.
fn join_or_panic<T>(handles: Vec<std::thread::ScopedJoinHandle<'_, T>>) -> Vec<T> {
    let mut values = Vec::with_capacity(handles.len());
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
    /// `Parts::loads` plus `Schedule::lpt`.
    pub schedule: u64,
    /// The serial reserve-all: every nonempty shard's checked region and every distinct key's
    /// complete directory path reserved before any worker starts.
    pub reserve: u64,
    /// The scoped per-shard build or grouped probe.
    pub kernel: u64,
    /// Joining worker results into the table or the query-order output, including the parallel
    /// build's directory rebase.
    pub assemble: u64,
}

impl PhaseNanos {
    fn absorb(&mut self, other: Self) {
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

/// Deterministic one-shot reservation-failure injection for the serial reserve-all.
///
/// This is an inherent seam, not a `#[cfg(test)]` one: the cross-worker zero-effects test lives in
/// `tests/parallel.rs`, an external integration crate that cannot observe `#[cfg(test)]` items.
/// The thread-local arm names one shard; the serial reserve-all consumes it at that shard's
/// reservation and returns `Err` before any hold, clone, merge, relocation or worker start. A
/// production call never arms it, and the arm is one-shot, so it cannot leak into a later
/// operation on the same thread.
#[doc(hidden)]
#[cfg(feature = "injection")]
pub mod reservation_failure {
    use std::cell::Cell;

    thread_local! {
        static ARMED_SHARD: Cell<Option<usize>> = const { Cell::new(None) };
    }

    /// Arms a one-shot failure for the reservation of `shard` on the calling thread.
    pub fn arm(shard: usize) {
        ARMED_SHARD.with(|cell| cell.set(Some(shard)));
    }

    /// Clears any armed failure on the calling thread.
    pub fn disarm() {
        ARMED_SHARD.with(|cell| cell.set(None));
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
}

/// Fills one already-reserved shard region from its stable partition runs.
///
/// The region was reserved by the serial reserve-all before any worker started, so this body
/// performs no fallible allocation. It initializes the region (one `hold` and the default clone)
/// and applies each run in input order, returning the region's live-tag count.
fn fill_reserved_shard<K, V, H, R, F, const N: usize>(
    region: &mut Region<R, V>,
    home: &H,
    merge: &F,
    default: &V,
    input: &[((H::Faces, K), V)],
    parts: &super::partition::Parts,
    shard_at: usize,
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
            let address = Home::address(home, entry_face.sigma(), key);
            apply_prepared_shard::<K, V, H, R, F, N>(
                region,
                merge,
                default,
                *entry_face,
                key,
                value.clone(),
                &address,
            )?;
        }
    }
    Ok(region.tags.iter().filter(|&&tag| tag != 0).count())
}

/// A shard's reserved region, or `None` for a zero-load shard.
type ReservedRegion<R, V> = Option<Region<R, V>>;

/// One worker bin: each owned shard paired with its reserved region.
type WorkerRegions<R, V> = Vec<(usize, ReservedRegion<R, V>)>;

/// Serial reserve-all: reserves every nonempty shard's checked region before any worker starts.
///
/// Returns `None` for a zero-load shard; its empty region is built by the shared
/// [`build_prepared_shard`] entry on the worker, which reserves nothing and holds nothing for zero
/// load. A seam hit or a checked sizing/allocation refusal returns `Err` with no hold, clone,
/// merge, relocation or spawned worker.
fn reserve_all_shards<R: Clone, V: Clone, const N: usize>(
    loads: &[usize; SHARDS],
) -> Result<Vec<ReservedRegion<R, V>>, StorageError> {
    let mut reserved = Vec::with_capacity(SHARDS);
    let mut s = 0;
    while s < SHARDS {
        #[cfg(feature = "injection")]
        if reservation_failure::consume(s) {
            return Err(StorageError::AllocationFailed);
        }
        if loads[s] == 0 {
            reserved.push(None);
        } else {
            let slots = slots_for(loads[s], N)?;
            let region = Region::reserve_empty(slots)?;
            reserved.push(Some(region));
        }
        s += 1;
    }
    Ok(reserved)
}

/// Moves each reserved region into the one worker bin that owns its shard.
fn distribute_regions<R, V>(
    bins: Vec<Vec<usize>>,
    reserved: &mut [ReservedRegion<R, V>],
) -> Vec<WorkerRegions<R, V>> {
    bins.into_iter()
        .map(|bin| {
            bin.into_iter()
                .map(|shard_at| (shard_at, reserved[shard_at].take()))
                .collect()
        })
        .collect()
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
        // table-shard reservation, then the directory reservation, all before the first scatter
        // worker starts. A sizing or reservation refusal here starts no worker, runs no hold,
        // clone, merge or relocation and returns no table. This supplies the cross-worker half of
        // the lifecycle contract that `build_prepared_shard` already satisfies per shard.
        let mark = Instant::now();
        let plan = super::partition::ScatterPlan::of(&table.home, kvs, tasks)?;
        let mut scratch = super::partition::ScatterScratch::reserve(&plan)?;
        phases.partition = elapsed_ns(mark);
        let loads = *plan.loads();
        let mark = Instant::now();
        let mut reserved = reserve_all_shards::<R, V, N>(&loads)?;
        phases.reserve = elapsed_ns(mark);
        // Reserve and link every distinct key's complete directory path before any worker callback;
        // the recorded payload locations are placeholders rebased after assembly.
        let mark = Instant::now();
        table.preassemble_directory(kvs)?;
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
        phases.validate = elapsed_ns(mark);
        let mark = Instant::now();
        let loads = parts.loads();
        let schedule = super::partition::Schedule::lpt(&loads, tasks);
        if !schedule.is_complete() {
            return Err(StorageError::SizeOverflow);
        }
        let bins: Vec<Vec<usize>> = schedule.bins().map(ToOwned::to_owned).collect();
        phases.schedule = elapsed_ns(mark);
        // Hand each reserved region to the one worker that owns its shard. `Schedule::is_complete`
        // already checked that the bins partition the shards, so every region is taken exactly once.
        let bins_with_regions = distribute_regions(bins, &mut reserved);
        let home_ref = &table.home;
        let default_ref = &table.default;
        let input = kvs;
        let parts_ref = &parts;
        let mark = Instant::now();
        let built = std::thread::scope(|scope| {
            let handles = bins_with_regions
                .into_iter()
                .map(|bin| {
                    scope.spawn(move || {
                        let mut mine = Vec::with_capacity(bin.len());
                        for (s, region) in bin {
                            // A zero-load shard has no reserved region; the shared per-shard builder
                            // returns the empty region without reserving or holding.
                            let outcome = match region {
                                Some(mut region) => fill_reserved_shard::<K, V, H, R, F, N>(
                                    &mut region,
                                    home_ref,
                                    merge,
                                    default_ref,
                                    input,
                                    parts_ref,
                                    s,
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
                                Ok((region, live)) => mine.push((s, Ok(region), live)),
                                Err(error) => mine.push((s, Err(error), 0)),
                            }
                        }
                        mine
                    })
                })
                .collect();
            join_or_panic(handles)
        });
        phases.kernel = elapsed_ns(mark);
        let mark = Instant::now();
        for worker in built {
            for (s, region, live) in worker {
                let region = region?;
                table.len += live;
                table.shards[s] = region;
            }
        }
        // Rebase every preassembled terminal to its assembled payload location. The directory was
        // fully reserved and linked before the workers, so this step performs no fallible
        // reservation and no duplicate merge.
        table.rebase_directory(kvs)?;
        phases.assemble = elapsed_ns(mark);
        Ok((table, phases))
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
        let parts_ref = &parts;
        let query_ref = grouped;
        let table_ref = self;
        let mark = Instant::now();
        let probed = std::thread::scope(|scope| {
            let handles = bins
                .iter()
                .map(|bin| {
                    scope.spawn(move || {
                        // Each probe produces exactly one result, so the bin's routed load is the
                        // exact result count; reserving it removes the growth copies from the
                        // worker's readback buffer.
                        let reserve: usize = bin.iter().map(|&s| loads[s]).sum();
                        let mut mine = Vec::with_capacity(reserve);
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
            join_or_panic(handles)
        });
        phases.kernel = elapsed_ns(mark);
        let mark = Instant::now();
        let mut ordered: Vec<Option<V>> = (0..queries_len).map(|_| None).collect();
        for worker in probed {
            for (index, value) in worker {
                ordered[index] = Some(value);
            }
        }
        let out = ordered
            .into_iter()
            .map(|value| each(&value.expect("complete grouped probe schedule")))
            .collect();
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
