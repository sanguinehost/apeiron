//! Extraction unit for the table's scoped-parallel scheduling and reservation seams
//! (`src/map/table/parallel.rs`).
//!
//! `src/map/table.rs` already declares the table module in this unit:
//!
//! ```text
//! #[path = "../../src/map/table.rs"]
//! pub mod table;
//! ```
//!
//! so no `#[path]` declaration is needed here. Each `word_table_parallel_*` wrapper fixes the word
//! monomorphization (`Stored<i64>`, `i64`, `WordHome`, `WordMerge`, `N = 16`) and calls the real
//! function; no wrapper restates a formula.
//!
//! # Boundary
//!
//! The thread-realization boundary is not rooted: `std::thread::scope`, the worker closures and
//! `Instant` phase timing stay outside the extraction fragment, as does the
//! `#[cfg(feature = "injection")]` seam, which the extraction unit does not compile. The roots
//! below are the serial partition, schedule, reservation and fill kernels those workers call.

use crate::table::keyrep::Stored;
use crate::table::{parallel, Region, StorageError};

/// The shard count of the table's fixed partition.
const SHARDS: usize = 256;

/// The table width the word roots are extracted at.
const N: usize = 16;

/// A shard's reserved region, or `None` for a zero-load shard.
type ReservedRegion = Option<Region<Stored<i64>, i64>>;

/// One worker-recorded placement: the input entry's index and its slot.
type PlacementRecord = (usize, usize);

/// One shard's placement-record buffer.
type ShardRecords = Vec<PlacementRecord>;

/// One owned shard: its index, its region and its placement-record buffer.
type OwnedShard = (usize, ReservedRegion, ShardRecords);

/// One worker bin of owned shards.
type WorkerRegions = Vec<OwnedShard>;

/// One worker's per-shard outcome.
type ShardOutcome = (
    usize,
    Result<Region<Stored<i64>, i64>, StorageError>,
    usize,
    ShardRecords,
);

/// One worker's outcome buffer.
type WorkerOutcomes = Vec<ShardOutcome>;

/// The pre-reserved worker storage: one outcome buffer per bin plus the joined-results sink.
type WorkerStorage = (Vec<WorkerOutcomes>, Vec<WorkerOutcomes>);

// ── `table::parallel::PhaseNanos` ───────────────────────────────────────────────────────────

/// Adds one measured phase split into another.
pub fn word_table_parallel_phase_absorb(
    mut phases: parallel::PhaseNanos,
    other: parallel::PhaseNanos,
) -> parallel::PhaseNanos {
    phases.absorb(other);
    phases
}

/// The sum of every phase.
pub fn word_table_parallel_phase_total(phases: parallel::PhaseNanos) -> u64 {
    phases.total()
}

// ── `table::parallel` serial reservation and scheduling ─────────────────────────────────────

/// Reserves the per-input terminal-handle table.
pub fn word_table_parallel_reserve_handle_table(
    len: usize,
) -> Result<Vec<Option<u32>>, StorageError> {
    parallel::reserve_handle_table(len)
}

/// Reserves every nonempty shard's checked region.
pub fn word_table_parallel_reserve_all_shards(
    loads: &[usize; SHARDS],
) -> Result<Vec<ReservedRegion>, StorageError> {
    parallel::reserve_all_shards::<Stored<i64>, i64, N>(loads)
}

/// Reserves one placement-record buffer per nonempty shard.
pub fn word_table_parallel_reserve_placement_records(
    loads: &[usize; SHARDS],
) -> Result<Vec<ShardRecords>, StorageError> {
    parallel::reserve_placement_records(loads)
}

/// Derives the shard schedule and hands each region and record buffer to its bin.
pub fn word_table_parallel_schedule_bins_with_regions(
    loads: &[usize; SHARDS],
    tasks: usize,
    reserved: &mut [ReservedRegion],
    record_bufs: &mut [ShardRecords],
) -> Result<Vec<WorkerRegions>, StorageError> {
    parallel::schedule_bins_with_regions::<Stored<i64>, i64>(loads, tasks, reserved, record_bufs)
}

/// Reserves one outcome buffer per worker bin and the joined-results sink.
pub fn word_table_parallel_reserve_worker_outcomes(
    bins_with_regions: &[WorkerRegions],
) -> Result<WorkerStorage, StorageError> {
    parallel::reserve_worker_outcomes::<Stored<i64>, i64>(bins_with_regions)
}

/// Moves each reserved region and record buffer into the bin that owns its shard.
pub fn word_table_parallel_distribute_regions(
    bins: Vec<Vec<usize>>,
    reserved: &mut [ReservedRegion],
    record_bufs: &mut [ShardRecords],
) -> Vec<WorkerRegions> {
    parallel::distribute_regions::<Stored<i64>, i64>(&bins, reserved, record_bufs)
}

// ── Boundary ────────────────────────────────────────────────────────────────────────────────

// `table::parallel::fill_reserved_shard` is NOT rooted: the pinned extractor refuses it with
// `[Error] Could not match the contexts` at its `records.push` inside the index loop, under three
// rewrites (in-place match arm, carried `Option<usize>` slot, typed `PlacementRecord` local), and
// the body's own `for`-over-`runs` shape is outside the fragment. Its emitted definition is a
// `sorry`, which the audit refuses, so the root is dropped and the body stays untranslated.
