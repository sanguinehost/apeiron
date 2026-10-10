//! A persistent spin-waiting worker pool for the `Q8Act` projections.
//!
//! `std::thread::scope` spawns and joins its threads on every call. Measured on the i9-14900KF
//! (empty scope, 2000 reps, `rustc -O`), that costs 86.5 µs at 8 threads and 291.6 µs at 32. At
//! ≈169 projections per token this is 14.6–49.3 ms per token, while llama.cpp decodes a whole
//! token in 8.8 ms. This pool keeps its workers alive and hands
//! them a job through one atomic store per worker, so a dispatch costs a store and the workers'
//! wake-up is a spin on their own cache line.
//!
//! **Protocol.** `slots[w]` packs `(epoch: 44 bits, n: 20 bits)` into one `u64` and is written
//! only by the dispatcher, one cache-line-padded word per worker; `done[w]` is the epoch worker `w`
//! last finished. A dispatch stores the job pointer and `panicked = false`, then publishes the
//! `(epoch, n)` word to each worker that owns a task (`Release`), then runs its own partition and
//! spins until every published worker's `done[w]` names the epoch. The caller always owns task 0.
//!
//! **Assignment.** Participant `k` (0 is the calling thread, `w + 1` is worker `w`) runs tasks
//! `k, k + width, k + 2·width, …` below `n`. The mapping is fixed and shared by construction, so
//! there is no claim word to compare-and-swap: the per-dispatch cost is one padded store per live
//! worker and one padded store back. For the equal-cost row chunks the callers build (they split
//! to `2·width` tasks) a round-robin split is balanced; the caller's share is claimed with no
//! wake-up, so a dispatch never depends on a worker's start for its first task. No two tasks
//! alias because the index sets are disjoint.
//!
//! **Heterogeneous cores.** By default, when the host provides efficiency cores (listed in
//! `/sys/devices/cpu_atom/cpus`), the pool includes all detected efficiency CPUs as workers pinned
//! one per efficiency core. On an equal-weight pool the round-robin above is used. On a
//! heterogeneous pool the dispatcher picks, per dispatch, whichever of the two candidate
//! assignments has the smaller modelled makespan `max_i work_i/speed_i` (`Shared::weighted`): the
//! contiguous split below, or the original round-robin. The contiguous split is weighted by
//! measured single-core speed, participant `k` owning
//! `[round(n·Σ_{j<k}w_j/Σw), round(n·Σ_{j≤k}w_j/Σw))`. Performance weight is [`PERF_WEIGHT`] and
//! efficiency weight `round(PERF_WEIGHT/ρ)` for the ratio ρ of performance to efficiency
//! single-core speed; the makespan model is equalised when `work_i ∝ w_i` and `speed_i ∝ w_i`.
//! Both the dispatcher and each worker compute the same choice and boundary from the immutable
//! `prefix` table and `n`, so no claim word is needed. Below the participant count the round-robin
//! keeps whole tasks on the performance participants, which the contiguous split can hand to an
//! efficiency worker; the modelled comparison chooses whichever balances the two. Setting
//! `APEIRON_ECORES=0` (or when the host has no efficiency cores / `APEIRON_PIN=0`) restores the
//! performance-cores-only pool where weights are all equal and the round-robin path is
//! byte-for-byte the pre-existing one; an explicit `k` caps efficiency workers at `k`.
//!
//! **Stale-snapshot safety.** `slots[w]` is single-writer (the dispatcher), single-reader (worker
//! `w`), and strictly increasing. A worker acts only on an epoch different from the one it last
//! served, read with `Acquire`, which pairs with the dispatcher's `Release` store of the same
//! word; the job pointer and `n` for that epoch were stored before it. The dispatcher does not
//! publish the next epoch until every worker it published to has stored `done[w] == epoch`, so the
//! job pointer a worker reads is always its own epoch's. `fold/hashtrinity/src/par.rs` needed an
//! epoch-tagged CAS and a separate publish-generation word because job, task count and claim word
//! were three independently loaded atomics; here the epoch and `n` live in the worker's own slot,
//! behind one acquire, so that class cannot arise.
//!
//! **Park.** A worker whose slot is unchanged spins for [`SPIN_BEFORE_PARK`] and then parks; the
//! dispatcher unparks the workers it published to only when `sleepers` is non-zero.
//!
//! **Output.** [`Pool::chunks_mut`] splits a mutable slice into disjoint chunks. Each task
//! receives exactly one chunk, so no two tasks alias. The result equals a sequential loop over the
//! same chunks for every job whose tasks are independent, which is how the projections use it.

use std::cell::UnsafeCell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// Bits of `n` in a `(epoch, n)` slot word; at most `2^20 − 1` tasks per dispatch.
const IDX_BITS: u32 = 20;
const IDX_MASK: u64 = (1 << IDX_BITS) - 1;
/// The largest task count one dispatch accepts.
pub(crate) const MAX_TASKS: usize = (1 << IDX_BITS) - 1;
/// Idle time after which a worker parks instead of spinning.
const SPIN_BEFORE_PARK: Duration = Duration::from_millis(5);

/// The speed weight of a performance-core participant in the heterogeneous partition. The
/// efficiency weight is `round(PERF_WEIGHT / ρ)`; the scale is small enough that `n·Σw` stays a
/// `u64` for every reachable `n` and core count.
const PERF_WEIGHT: u64 = 100;
/// Performance-to-efficiency single-core speed ratio ρ used when `APEIRON_ECORE_RATIO` is unset.
///
/// Measured on the i9-14900KF by a scratch bench (single-thread; P
/// pinned to cpu 0, E pinned to cpu 16): the `Q5_0` 4864×896 chart projection ran 20.76 Gweight/s
/// on P against 9.83 on E (ρ = 2.11) and the `PQ2_0` 4096×2048 lookup projection 46.34 against 13.45
/// (ρ = 3.45); the geometric mean is 2.70. Overridden at runtime by `APEIRON_ECORE_RATIO` (a finite
/// value ≥ 1).
const DEFAULT_ECORE_RATIO: f64 = 2.7;

/// One word per cache line: a worker's slot never shares a line with its neighbours.
#[repr(align(64))]
struct CachePadded<T>(T);

/// Which core class a worker is pinned to, and therefore its weight in the heterogeneous split.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkerClass {
    /// A performance (P) core.
    Performance,
    /// An efficiency (E) core.
    Efficiency,
}

/// The largest task count for which `weighted(n)` is precomputed in [`Split`].
pub(crate) const WEIGHTED_TABLE_MAX: usize = 1024;

/// Contiguous task range `[lo, hi)` under a speed-weighted split with `prefix` weights:
/// `lo = round(n·prefix[k]/W)`, `hi = round(n·prefix[k+1]/W)` with `W = prefix[last]`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "lo and hi are bounded by n ≤ MAX_TASKS < 2^20, so the u64 quotient fits usize"
)]
fn range_with_prefix(prefix: &[u64], k: usize, n: usize) -> (usize, usize) {
    let Some(&total) = prefix.last() else {
        return (0, 0);
    };
    if total == 0 {
        return (0, 0);
    }
    let n = n as u64;
    let half = total / 2;
    let lo = (n * prefix[k] + half) / total;
    let hi = (n * prefix[k + 1] + half) / total;
    (lo as usize, hi as usize)
}

/// Pure makespan model decision: whether the weighted contiguous split beats round-robin for `n`.
fn compute_weighted(prefix: &[u64], width: usize, heterogeneous: bool, n: usize) -> bool {
    if n <= 1 || !heterogeneous {
        return false;
    }
    let parts = prefix.len().saturating_sub(1);
    let Some(&total) = prefix.last() else {
        return false;
    };
    if parts == 0 || total == 0 {
        return false;
    }
    let (mut w_num, mut w_den) = (0u128, 1u128);
    let (mut r_num, mut r_den) = (0u128, 1u128);
    let total = u128::from(total);
    for k in 0..parts {
        let weight = u128::from(prefix[k + 1] - prefix[k]);
        if weight == 0 {
            continue;
        }
        let (lo, hi) = range_with_prefix(prefix, k, n);
        let w_hits = (hi - lo) as u128 * total;
        if w_hits * w_den > w_num * weight {
            w_num = w_hits;
            w_den = weight;
        }
        let r_hits = round_robin_range(k, n, width).1 as u128 * total;
        if r_hits * r_den > r_num * weight {
            r_num = r_hits;
            r_den = weight;
        }
    }
    r_num * w_den >= w_num * r_den
}

/// The immutable speed-weighted split shared by the dispatcher and every worker.
struct Split {
    /// `prefix[k]` is the total weight of participants `0..k`; `prefix[participants]` is the whole
    /// weight.
    prefix: Vec<u64>,
    /// Whether the participant weights differ; `false` selects the round-robin path, which is the
    /// pre-`APEIRON_ECORES` behaviour.
    heterogeneous: bool,
    /// Precomputed `weighted(n)` decisions for `n` in `0..=WEIGHTED_TABLE_MAX`.
    weighted_table: Box<[bool]>,
}

impl Split {
    /// The split of `participants` participants whose weights are `weights`.
    fn new(weights: &[u64]) -> Self {
        let mut prefix = Vec::with_capacity(weights.len() + 1);
        prefix.push(0);
        let mut acc = 0u64;
        for w in weights {
            acc += w;
            prefix.push(acc);
        }
        let heterogeneous = weights
            .first()
            .is_some_and(|first| weights.iter().any(|w| w != first));
        let width = weights.len();
        let mut table = Vec::with_capacity(WEIGHTED_TABLE_MAX + 1);
        for n in 0..=WEIGHTED_TABLE_MAX {
            table.push(compute_weighted(&prefix, width, heterogeneous, n));
        }
        Self {
            prefix,
            heterogeneous,
            weighted_table: table.into_boxed_slice(),
        }
    }
}

/// Participant `k`'s share of the homogeneous round-robin split of `n` tasks across `width`
/// participants: `(start, count)` with `start = k` and `count = ceil((n − k)/width)`, i.e. the
/// `count` indices `k, k + width, …` below `n`; `(k, 0)` when `k ≥ n`. This is the mapping the
/// pool used before the weighted split and the fallback [`Shared::weighted`] compares against.
///
/// # Complexity
///
/// `O(1)`.
fn round_robin_range(k: usize, n: usize, width: usize) -> (usize, usize) {
    if width == 0 || k >= n {
        return (k, 0);
    }
    (k, (n - k).div_ceil(width))
}

/// The type-erased job: a `&(dyn Fn(usize) + Sync)` with its lifetime removed. It is read only by
/// a participant of the epoch that published it, before that epoch's `done` store.
type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    /// `slots[w] = (epoch << IDX_BITS) | n`, written only by the dispatcher.
    slots: Vec<CachePadded<AtomicU64>>,
    /// Completion counter for the active dispatch: initialized to live worker count, decremented by
    /// finishing workers (`Release`), dispatcher spins on this line (`Acquire`).
    remaining: CachePadded<AtomicUsize>,
    /// The job of the live epoch; read by every participant, written only between epochs.
    /// Cache-padded to isolate per-dispatch dispatcher writes from spinning workers.
    job: CachePadded<UnsafeCell<Option<Job>>>,
    /// Whether any task panicked; cache-padded to avoid false sharing with spin fields.
    panicked: CachePadded<AtomicBool>,
    /// Participants in a dispatch (workers plus the caller); written once, before the first
    /// dispatch.
    width: AtomicUsize,
    /// The speed-weighted split, set after the workers that actually spawned are known and before
    /// the first dispatch. `None` (only reachable before that store) selects the round-robin path.
    split: OnceLock<Split>,
    sleepers: AtomicUsize,
    shutdown: AtomicBool,
}

// SAFETY: `job` is written only by the dispatcher between epochs (no participant can be reading:
// each decrements `remaining` before the dispatcher proceeds) and read only by participants of
// the live epoch; every other field is atomic.
unsafe impl Sync for Shared {}
// SAFETY: as above; the raw job pointer is only dereferenced under the epoch protocol.
unsafe impl Send for Shared {}

impl Shared {
    /// Participant `k`'s contiguous task range `[lo, hi)` under the speed-weighted split:
    /// `lo = round(n·prefix[k]/W)`, `hi = round(n·prefix[k+1]/W)` with `W = prefix[last]`. The
    /// boundaries are non-decreasing, `lo(0) = 0` and `hi(last) = n`, so the ranges partition
    /// `0..n`; an empty range is skipped by its participant.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    fn range(&self, k: usize, n: usize) -> (usize, usize) {
        let Some(split) = self.split.get() else {
            return (0, 0);
        };
        range_with_prefix(&split.prefix, k, n)
    }

    /// Whether the weighted contiguous assignment beats the homogeneous round-robin for an `n`-task
    /// dispatch. Looked up in `O(1)` from precomputed table for `n <= WEIGHTED_TABLE_MAX`, evaluated
    /// dynamically otherwise.
    ///
    /// # Complexity
    ///
    /// `O(1)` table lookup for `n ≤ 1024`; `O(participants)` integer operations otherwise.
    fn weighted(&self, n: usize) -> bool {
        let Some(split) = self.split.get() else {
            return false;
        };
        if let Some(&choice) = split.weighted_table.get(n) {
            choice
        } else {
            compute_weighted(
                &split.prefix,
                self.width.load(Ordering::Relaxed),
                split.heterogeneous,
                n,
            )
        }
    }

    /// Run participant `k`'s share of the `n`-task epoch: the contiguous weighted range when
    /// `weighted` selects it, indices `k, k + width, …` otherwise.
    ///
    /// # Complexity
    ///
    /// `O(tasks owned)` job calls, each wrapped in `catch_unwind`.
    fn run_partition(&self, k: usize, n: usize, weighted: bool) {
        let job = unsafe { *self.job.0.get() };
        let Some(job) = job else {
            return;
        };
        // SAFETY: the job outlives the dispatch that published it (see `run`).
        let f = unsafe { &*job };
        let serve = |i: usize| {
            if catch_unwind(AssertUnwindSafe(|| f(i))).is_err() {
                self.panicked.0.store(true, Ordering::Relaxed);
            }
        };
        if weighted {
            let (lo, hi) = self.range(k, n);
            let mut i = lo;
            while i < hi {
                serve(i);
                i += 1;
            }
        } else {
            let width = self.width.load(Ordering::Relaxed);
            let (start, count) = round_robin_range(k, n, width);
            let mut i = start;
            for _ in 0..count {
                serve(i);
                i += width;
            }
        }
    }
}

/// A fixed set of spin-waiting worker threads.
pub(crate) struct Pool {
    shared: Arc<Shared>,
    threads: Vec<Thread>,
    epoch: AtomicU64,
    /// Serializes dispatches: one job is live at a time (tests share the global pool across
    /// threads). A task must not dispatch on its own pool.
    dispatch: Mutex<()>,
}

impl Pool {
    /// A pool with `workers` equal-weight performance-core background threads (the caller is one
    /// more participant).
    pub(crate) fn new(workers: usize) -> Self {
        Self::with_classes(&vec![WorkerClass::Performance; workers])
    }

    /// A pool with one background thread per entry of `classes`, pinned to a core of that class and
    /// weighted by it; the caller is one more performance-core participant.
    ///
    /// A performance worker takes the next P core after the caller's, an efficiency worker the next
    /// E core. A core class whose `/sys` list is empty leaves its workers unpinned but weighted as
    /// configured. A failed spawn truncates the pool; the split is then built from the workers that
    /// did spawn, so the task ranges still cover every index.
    pub(crate) fn with_classes(classes: &[WorkerClass]) -> Self {
        let p_cpus = performance_cpus();
        let e_cpus = efficiency_cpus();
        let eff_weight = efficiency_weight();
        let slots = (0..classes.len())
            .map(|_| CachePadded(AtomicU64::new(0)))
            .collect();
        let shared = Arc::new(Shared {
            slots,
            remaining: CachePadded(AtomicUsize::new(0)),
            job: CachePadded(UnsafeCell::new(None)),
            width: AtomicUsize::new(1),
            split: OnceLock::new(),
            panicked: CachePadded(AtomicBool::new(false)),
            sleepers: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        });
        let mut threads = Vec::with_capacity(classes.len());
        let mut spawned: Vec<WorkerClass> = Vec::with_capacity(classes.len());
        let mut p_next = 0usize;
        let mut e_next = 0usize;
        for (w, &class) in classes.iter().enumerate() {
            let sh = Arc::clone(&shared);
            // A performance worker `w` takes the (`w`+1)-th performance core, leaving the first for
            // the caller; an efficiency worker the next E core. Each list cycles if it runs out.
            let pin = match class {
                WorkerClass::Performance => {
                    if p_cpus.is_empty() {
                        None
                    } else {
                        let cpu = p_cpus[(p_next + 1) % p_cpus.len()];
                        p_next += 1;
                        Some(cpu)
                    }
                }
                WorkerClass::Efficiency => {
                    if e_cpus.is_empty() {
                        None
                    } else {
                        let cpu = e_cpus[e_next % e_cpus.len()];
                        e_next += 1;
                        Some(cpu)
                    }
                }
            };
            let handle = thread::Builder::new()
                .name("apeiron-q8act".into())
                .spawn(move || {
                    if let Some(cpu) = pin {
                        pin_current_thread(cpu);
                    }
                    worker(&sh, w);
                });
            match handle {
                Ok(h) => {
                    threads.push(h.thread().clone());
                    spawned.push(class);
                }
                Err(_) => break,
            }
        }
        // The caller is a performance participant; the split covers exactly the spawned workers.
        let mut weights = Vec::with_capacity(spawned.len() + 1);
        weights.push(PERF_WEIGHT);
        weights.extend(spawned.iter().map(|&c| match c {
            WorkerClass::Performance => PERF_WEIGHT,
            WorkerClass::Efficiency => eff_weight,
        }));
        let _ = shared.split.set(Split::new(&weights));
        shared.width.store(threads.len() + 1, Ordering::Release);
        Self {
            shared,
            threads,
            epoch: AtomicU64::new(0),
            dispatch: Mutex::new(()),
        }
    }

    /// The process-wide pool. Its performance width is `APEIRON_THREADS` participants when that
    /// variable parses, else 8 (llama.cpp's fastest setting on this host), capped at the pinnable performance cores. Efficiency
    /// cores are ON by default when the host has them: with `APEIRON_ECORES` unset, all detected
    /// efficiency CPUs are used. `APEIRON_ECORES=0` restores the performance-cores-only pool,
    /// while an explicit `k` uses at most `k` efficiency workers (capped at pinnable efficiency
    /// cores). All efficiency workers are disabled when `APEIRON_PIN=0` or when
    /// `/sys/devices/cpu_atom/cpus` is absent.
    pub(crate) fn global() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(|| {
            let n = std::env::var("APEIRON_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(8);
            // Two spinners on one core starve each other (measured: 8 → 78 tok/s, 12 → 28, 16 →
            // 21 with 8 performance cores), so the width never exceeds the pinnable cores.
            let cpus = performance_cpus();
            let n = if cpus.is_empty() {
                n
            } else {
                n.min(cpus.len())
            };
            let detected = efficiency_cpus().len();
            let e_workers =
                resolve_ecores(std::env::var("APEIRON_ECORES").ok().as_deref(), detected);
            if e_workers == 0 {
                return Pool::new(n.saturating_sub(1));
            }
            let mut classes = vec![WorkerClass::Performance; n.saturating_sub(1)];
            classes.extend(std::iter::repeat_n(WorkerClass::Efficiency, e_workers));
            Pool::with_classes(&classes)
        })
    }

    /// Participants in a dispatch: the workers plus the caller.
    pub(crate) fn width(&self) -> usize {
        self.threads.len() + 1
    }

    /// Run `f(0) … f(n − 1)` across the pool and the calling thread, returning when all are done.
    /// Concurrent callers are serialized on an internal lock.
    ///
    /// # Panics
    ///
    /// Re-raises a panic from any task after all tasks have finished; panics if `n > MAX_TASKS`.
    ///
    /// # Complexity
    ///
    /// `n` job calls; `O(1)` dispatch cost plus one wake-up per parked worker.
    pub(crate) fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        assert!(
            n <= MAX_TASKS,
            "pool dispatch of {n} tasks exceeds {MAX_TASKS}"
        );
        if n == 0 {
            return;
        }
        let _serial = self
            .dispatch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        CALLER_PINNED.with(|done| {
            if !done.get() {
                done.set(true);
                if let Some(&cpu) = performance_cpus().first() {
                    pin_current_thread(cpu);
                }
            }
        });
        let sh = &*self.shared;
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed) + 1;
        // SAFETY: no participant is running (the previous epoch finished with every published
        // `done[w] == old epoch`, and the new one is not published), so nothing reads `job`
        // concurrently. The lifetime is erased; `run` does not return before every published
        // worker has finished, so `f` outlives its use.
        unsafe {
            let erased: Job = std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f);
            *sh.job.0.get() = Some(erased);
        }
        sh.panicked.0.store(false, Ordering::Relaxed);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "n ≤ MAX_TASKS < 2^20 fits u64"
        )]
        let word = (epoch << IDX_BITS) | n as u64;
        // One choice per dispatch, computed here and identically inside every worker: a weighted
        // dispatch publishes to the workers whose weighted range is non-empty and joins exactly
        // those; the round-robin branch is the pre-`APEIRON_ECORES` code, unchanged.
        let weighted = sh.weighted(n);
        let live = if weighted {
            let mut count = 0usize;
            for w in 0..self.threads.len() {
                let (lo, hi) = sh.range(w + 1, n);
                if lo < hi {
                    count += 1;
                }
            }
            count
        } else {
            self.threads.len().min(n.saturating_sub(1))
        };
        sh.remaining.0.store(live, Ordering::Release);
        if live > 0 {
            let wake = sh.sleepers.load(Ordering::Acquire) > 0;
            if weighted {
                for (w, slot) in sh.slots.iter().enumerate().take(self.threads.len()) {
                    let (lo, hi) = sh.range(w + 1, n);
                    if lo < hi {
                        slot.0.store(word, Ordering::Release);
                        if wake {
                            self.threads[w].unpark();
                        }
                    }
                }
            } else {
                for slot in sh.slots.iter().take(live) {
                    slot.0.store(word, Ordering::Release);
                }
                if wake {
                    for t in self.threads.iter().take(live) {
                        t.unpark();
                    }
                }
            }
        }
        // The caller is already awake: run its own share before waiting on anyone.
        sh.run_partition(0, n, weighted);
        let _wait = crate::engine::trace::span(crate::engine::trace::Phase::PoolWait, n, 0, 0);
        if live > 0 {
            while sh.remaining.0.load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
        }
        // SAFETY: every published participant finished; no worker can claim in this epoch any more.
        unsafe {
            *sh.job.0.get() = None;
        }
        if sh.panicked.0.load(Ordering::Relaxed) {
            resume_unwind(Box::new("a Q8Act pool task panicked"));
        }
    }

    /// Split `out` into consecutive chunks of `chunk` elements (the last may be shorter) and run
    /// `f(chunk_index, start, chunk_slice)` on each, across the pool.
    ///
    /// # Complexity
    ///
    /// `⌈out.len() / chunk⌉` tasks; `O(1)` extra space.
    pub(crate) fn chunks_mut<T: Send>(
        &self,
        out: &mut [T],
        chunk: usize,
        f: &(dyn Fn(usize, usize, &mut [T]) + Sync),
    ) {
        let chunk = chunk.max(1);
        let len = out.len();
        let tasks = len.div_ceil(chunk);
        let base = SendPtr(out.as_mut_ptr());
        self.run(tasks, &move |i| {
            let start = i * chunk;
            let end = (start + chunk).min(len);
            // SAFETY: task `i` owns exactly `[start, end)`; tasks partition `0..len` disjointly and
            // `out` is mutably borrowed for the whole dispatch.
            let part =
                unsafe { std::slice::from_raw_parts_mut(base.get().add(start), end - start) };
            f(i, start, part);
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        for t in &self.threads {
            t.unpark();
        }
    }
}

/// A raw pointer that may cross threads; the disjoint-chunk argument in `chunks_mut` makes it
/// sound.
struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
// SAFETY: see `chunks_mut`: each task dereferences only its own disjoint range.
unsafe impl<T: Send> Send for SendPtr<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// One logical CPU per physical performance core, when the host says which cores those are.
///
/// Hybrid Intel parts list their performance cores in `/sys/devices/cpu_core/cpus` (for example
/// `0-15`); a CPU is kept when it is the first of its `thread_siblings_list`. On the i9-14900KF
/// this is `0, 2, …, 14`. An absent file or `APEIRON_PIN=0` yields no pinning. T6B measured the
/// `Q8Act` decode at 47 tok/s on P-cores against 32–35 unpinned.
///
/// # Complexity
///
/// `O(c)` file reads for `c` listed CPUs, once per pool.
fn performance_cpus() -> Vec<usize> {
    if pinned_off() {
        return Vec::new();
    }
    cpus_from_sysfs("/sys/devices/cpu_core/cpus")
}

/// One logical CPU per physical efficiency core, when the host says which cores those are.
///
/// Hybrid Intel parts list their efficiency cores in `/sys/devices/cpu_atom/cpus` (i9-14900KF:
/// `16-31`, one thread each). The sibling filter is the same as [`performance_cpus`]; an absent
/// file or `APEIRON_PIN=0` yields no pinning and therefore no efficiency workers.
///
/// # Complexity
///
/// `O(c)` file reads for `c` listed CPUs, once per pool.
fn efficiency_cpus() -> Vec<usize> {
    if pinned_off() {
        return Vec::new();
    }
    cpus_from_sysfs("/sys/devices/cpu_atom/cpus")
}

/// Whether `APEIRON_PIN=0` disables all affinity handling.
fn pinned_off() -> bool {
    std::env::var("APEIRON_PIN").as_deref() == Ok("0")
}

/// The physical CPUs listed in a `/sys/devices/.../cpus` `a-b,c` list, keeping the first thread of
/// each `thread_siblings_list`.
///
/// # Complexity
///
/// `O(c)` file reads for `c` listed CPUs.
fn cpus_from_sysfs(path: &str) -> Vec<usize> {
    let Ok(list) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for part in list.trim().split(',') {
        let mut ends = part.split('-').map(|v| v.trim().parse::<usize>());
        let (Some(Ok(a)), b) = (ends.next(), ends.next()) else {
            continue;
        };
        let b = match b {
            Some(Ok(b)) => b,
            _ => a,
        };
        for cpu in a..=b {
            let sib = std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ));
            let first = sib.ok().and_then(|t| {
                t.trim()
                    .split([',', '-'])
                    .next()
                    .and_then(|v| v.parse::<usize>().ok())
            });
            if first.is_none_or(|f| f == cpu) {
                out.push(cpu);
            }
        }
    }
    out
}

/// The efficiency-core weight in the speed-weighted split: `round(PERF_WEIGHT / ρ)` for the
/// performance-to-efficiency single-core speed ratio ρ, at least 1.
///
/// ρ is `APEIRON_ECORE_RATIO` when it parses to a finite value ≥ 1, else [`DEFAULT_ECORE_RATIO`].
/// The model behind the weight is `makespan = max_i work_i / speed_i`: a worker whose single-core
/// speed is `1/ρ` of a performance core receives `1/ρ` of a performance core's work.
///
/// # Complexity
///
/// `O(1)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "PERF_WEIGHT = 100 is exactly representable as f64, and the quotient is clamped into \
              [1, PERF_WEIGHT], so the f64 → u64 cast is exact and non-negative"
)]
fn efficiency_weight() -> u64 {
    let ratio = std::env::var("APEIRON_ECORE_RATIO")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|r| r.is_finite() && *r >= 1.0)
        .unwrap_or(DEFAULT_ECORE_RATIO);
    let weight = (PERF_WEIGHT as f64 / ratio)
        .round()
        .clamp(1.0, PERF_WEIGHT as f64);
    weight as u64
}

/// Resolves the number of efficiency-core workers to spawn.
///
/// When `var` is `None` (environment variable unset), all `detected` efficiency cores are used.
/// An explicit `"0"` restores the performance-cores-only pool (0 efficiency workers).
/// An explicit count `k` is capped at `detected`.
/// Invalid or non-numeric strings fall back to 0.
///
/// # Complexity
///
/// `O(1)`.
pub(crate) fn resolve_ecores(var: Option<&str>, detected: usize) -> usize {
    match var {
        None => detected,
        Some(s) => match s.trim().parse::<usize>() {
            Ok(k) => k.min(detected),
            Err(_) => 0,
        },
    }
}

/// Restrict the calling thread to one CPU; a refusal leaves it unpinned.
#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) {
    // SAFETY: `set` is a zero-initialized `cpu_set_t` owned by this frame; `CPU_SET` writes one bit
    // inside it (`cpu` is bounds-checked against its capacity), and `sched_setaffinity(0, …)`
    // reads it for the calling thread only.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if cpu < 8 * std::mem::size_of::<libc::cpu_set_t>() {
            libc::CPU_SET(cpu, &mut set);
            let _ =
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw const set);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: usize) {}

thread_local! {
    /// Whether this dispatching thread has been pinned to the first performance core.
    static CALLER_PINNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Worker loop: spin on this worker's slot, serve a new epoch, park after `SPIN_BEFORE_PARK`.
fn worker(sh: &Shared, w: usize) {
    let mut seen = 0u64;
    let mut idle_since = Instant::now();
    let mut spins: u32 = 0;
    loop {
        let word = sh.slots[w].0.load(Ordering::Acquire);
        let epoch = word >> IDX_BITS;
        if epoch != seen && epoch != 0 {
            seen = epoch;
            // SAFETY-adjacent: `k < n` holds because the dispatcher publishes only slots with an
            // owned task; the guard is defensive.
            let k = w + 1;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "n ≤ MAX_TASKS < 2^20 fits usize"
            )]
            let n = (word & IDX_MASK) as usize;
            let weighted = sh.weighted(n);
            if weighted || k < n {
                sh.run_partition(k, n, weighted);
            }
            sh.remaining.0.fetch_sub(1, Ordering::Release);
            idle_since = Instant::now();
            spins = 0;
            continue;
        }
        if spins.is_multiple_of(1024) && sh.shutdown.load(Ordering::Acquire) {
            return;
        }
        std::hint::spin_loop();
        spins = spins.wrapping_add(1);
        if spins.is_multiple_of(1024) && idle_since.elapsed() > SPIN_BEFORE_PARK {
            sh.sleepers.fetch_add(1, Ordering::AcqRel);
            let again = sh.slots[w].0.load(Ordering::Acquire);
            if again >> IDX_BITS == seen && !sh.shutdown.load(Ordering::Acquire) {
                thread::park_timeout(Duration::from_millis(50));
            }
            sh.sleepers.fetch_sub(1, Ordering::AcqRel);
            // `idle_since` is reset only by served work: resetting it here made an idle worker spin
            // 5 ms after every 50 ms park, a 9% duty cycle per worker (≈2 of 32 CPUs busy with 23
            // idle workers), which loaded the host under every concurrent measurement.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn every_task_runs_exactly_once() {
        let pool = Pool::new(5);
        for n in [1usize, 2, 7, 64, 1000] {
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "n = {n}"
            );
        }
    }

    #[test]
    fn chunks_equal_sequential_and_back_to_back_dispatches_do_not_mix() {
        let pool = Pool::new(3);
        for round in 0..500u64 {
            let mut out = vec![0u64; 97];
            pool.chunks_mut(&mut out, 8, &|_, start, part| {
                for (k, v) in part.iter_mut().enumerate() {
                    *v = (start + k) as u64 * 3 + round;
                }
            });
            let want: Vec<u64> = (0..97u64).map(|i| i * 3 + round).collect();
            assert_eq!(out, want, "round {round}");
        }
    }

    #[test]
    fn zero_workers_is_sequential() {
        let pool = Pool::new(0);
        let mut out = vec![0u32; 10];
        pool.chunks_mut(&mut out, 3, &|i, _, part| {
            part.fill(u32::try_from(i).unwrap_or(u32::MAX));
        });
        assert_eq!(out, vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3]);
    }

    #[test]
    fn a_task_panic_is_reraised_after_the_dispatch() {
        let pool = Pool::new(2);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(16, &|i| assert!(i != 9, "seeded"));
        }));
        assert!(r.is_err());
        let hits = AtomicU64::new(0);
        pool.run(4, &|_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 4);
    }

    /// A panic in a *worker's* partition (task 1 belongs to participant 1) is also re-raised, and
    /// the pool keeps dispatching afterwards.
    #[test]
    fn a_worker_panic_is_reraised_and_the_pool_survives() {
        let pool = Pool::new(2);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(16, &|i| assert!(i != 1, "seeded-worker"));
        }));
        assert!(r.is_err());
        let hits = AtomicU64::new(0);
        pool.run(9, &|_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 9);
    }

    /// 10^5 dispatches of a random task count: the per-index sum and the call count must both
    /// equal the closed forms for every `n`, which catches a duplicated, dropped or crossed index.
    #[test]
    fn stress_100k_random_dispatches_checksum() {
        let pool = Pool::new(7);
        let sum = AtomicU64::new(0);
        let calls = AtomicU64::new(0);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut acc = 0u64;
        let mut want = 0u64;
        for _ in 0..100_000u32 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let n = (seed >> 33) as usize % 65;
            sum.store(0, Ordering::Relaxed);
            calls.store(0, Ordering::Relaxed);
            pool.run(n, &|i| {
                sum.fetch_add((i as u64) + 1, Ordering::Relaxed);
                calls.fetch_add(1, Ordering::Relaxed);
            });
            let n = n as u64;
            acc += sum.load(Ordering::Relaxed);
            want += n * (n + 1) / 2;
            assert_eq!(calls.load(Ordering::Relaxed), n, "call count at n = {n}");
        }
        assert_eq!(acc, want);
    }

    /// An explicit 2P+2E topology exercises the speed-weighted contiguous split: every index runs
    /// exactly once for every size, then a 10^5-dispatch checksum over random sizes catches a
    /// duplicated, dropped or crossed index in the weighted ranges — including one an efficiency
    /// worker would drop.
    #[test]
    fn heterogeneous_stress_100k_dispatches_checksum() {
        let pool = Pool::with_classes(&[
            WorkerClass::Performance,
            WorkerClass::Performance,
            WorkerClass::Efficiency,
            WorkerClass::Efficiency,
        ]);
        for n in 1..=64usize {
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "n = {n}"
            );
        }
        let sum = AtomicU64::new(0);
        let calls = AtomicU64::new(0);
        let mut seed = 0x517c_c1b7_2722_0a95u64;
        let mut acc = 0u64;
        let mut want = 0u64;
        for _ in 0..100_000u32 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let n = (seed >> 33) as usize % 65;
            sum.store(0, Ordering::Relaxed);
            calls.store(0, Ordering::Relaxed);
            pool.run(n, &|i| {
                sum.fetch_add((i as u64) + 1, Ordering::Relaxed);
                calls.fetch_add(1, Ordering::Relaxed);
            });
            let n = n as u64;
            acc += sum.load(Ordering::Relaxed);
            want += n * (n + 1) / 2;
            assert_eq!(calls.load(Ordering::Relaxed), n, "call count at n = {n}");
        }
        assert_eq!(acc, want);
    }

    #[test]
    fn ecores_resolution_default_and_overrides() {
        // Unset env var uses all detected E-cores.
        assert_eq!(resolve_ecores(None, 16), 16);
        assert_eq!(resolve_ecores(None, 8), 8);
        assert_eq!(resolve_ecores(None, 0), 0);

        // APEIRON_ECORES=0 restores P-cores only.
        assert_eq!(resolve_ecores(Some("0"), 16), 0);
        assert_eq!(resolve_ecores(Some("0"), 0), 0);

        // Explicit k caps at detected E-cores.
        assert_eq!(resolve_ecores(Some("4"), 16), 4);
        assert_eq!(resolve_ecores(Some("16"), 16), 16);
        assert_eq!(resolve_ecores(Some("32"), 16), 16);
        assert_eq!(resolve_ecores(Some("8"), 4), 4);

        // Non-numeric or empty string falls back to 0.
        assert_eq!(resolve_ecores(Some(""), 16), 0);
        assert_eq!(resolve_ecores(Some("invalid"), 16), 0);
    }
}

/// Pins for the per-dispatch assignment choice: the two candidate mappings are exact partitions of
/// the task indices, the chosen one never has the larger modelled makespan, the caller's
/// participant 0 is never left idle, and the weighted split still wins at the large `n` the
/// pre-existing behaviour relies on.
#[cfg(test)]
mod weighted_split_tests {
    use super::*;

    /// The real two-class table `Pool::global` builds on this host: the caller plus 7 performance
    /// workers at [`PERF_WEIGHT`], and 16 efficiency workers at `round(100/2.7) = 37`.
    fn real_weights() -> Vec<u64> {
        let mut weights = vec![PERF_WEIGHT; 8];
        weights.extend(std::iter::repeat_n(37, 16));
        weights
    }

    /// A `Shared` with `weights` and no worker threads: enough to exercise the pure `range` and
    /// `weighted` decisions without spawning a pool.
    fn shared_with(weights: &[u64]) -> Shared {
        let workers = weights.len().saturating_sub(1);
        let split = OnceLock::new();
        let _ = split.set(Split::new(weights));
        Shared {
            slots: (0..workers)
                .map(|_| CachePadded(AtomicU64::new(0)))
                .collect(),
            remaining: CachePadded(AtomicUsize::new(0)),
            job: CachePadded(UnsafeCell::new(None::<Job>)),
            width: AtomicUsize::new(weights.len()),
            split,
            panicked: CachePadded(AtomicBool::new(false)),
            sleepers: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    #[test]
    fn table_equals_function_over_n_0_to_1024() {
        let weights = real_weights();
        let sh = shared_with(&weights);
        let split = sh.split.get().expect("split");
        for n in 0..=WEIGHTED_TABLE_MAX {
            let func_val = compute_weighted(&split.prefix, weights.len(), split.heterogeneous, n);
            let tab_val = split.weighted_table[n];
            assert_eq!(
                tab_val, func_val,
                "table differs from compute_weighted at n={n}"
            );
            assert_eq!(
                sh.weighted(n),
                func_val,
                "sh.weighted differs from compute_weighted at n={n}"
            );
        }
    }

    /// The owner of every index `0..n` under one candidate mapping, `usize::MAX` if unowned; also
    /// asserts the candidate's own ranges are pairwise disjoint.
    fn coverage(sh: &Shared, n: usize, weighted: bool, width: usize) -> Vec<usize> {
        let parts = sh.split.get().map_or(0, |s| s.prefix.len() - 1);
        let mut owner = vec![usize::MAX; n];
        for k in 0..parts {
            if weighted {
                let (lo, hi) = sh.range(k, n);
                for (i, o) in owner.iter_mut().enumerate().take(hi).skip(lo) {
                    assert_eq!(*o, usize::MAX, "weighted overlap at {i} (n={n})");
                    *o = k;
                }
            } else {
                let (start, count) = round_robin_range(k, n, width);
                let mut i = start;
                for _ in 0..count {
                    assert!(i < n, "round-robin index {i} >= n={n}");
                    assert_eq!(owner[i], usize::MAX, "round-robin overlap at {i} (n={n})");
                    owner[i] = k;
                    i += width;
                }
            }
        }
        owner
    }

    /// The modelled makespan `max_i (count_i · W) / weight_i` of one candidate, as an exact
    /// `(numerator, denominator)` fraction. Computed independently of [`Shared::weighted`] so the
    /// pin tests the selection rather than restating it.
    fn makespan(sh: &Shared, n: usize, width: usize, weighted: bool) -> (u128, u128) {
        let split = sh.split.get().expect("split");
        let total = u128::from(*split.prefix.last().expect("non-empty prefix"));
        let parts = split.prefix.len() - 1;
        let (mut num, mut den) = (0u128, 1u128);
        for k in 0..parts {
            let weight = u128::from(split.prefix[k + 1] - split.prefix[k]);
            if weight == 0 {
                continue;
            }
            let count = if weighted {
                let (lo, hi) = sh.range(k, n);
                (hi - lo) as u128
            } else {
                round_robin_range(k, n, width).1 as u128
            };
            let cand = count * total;
            if cand * den > num * weight {
                num = cand;
                den = weight;
            }
        }
        (num, den)
    }

    #[test]
    fn weighted_choice_is_exact_and_preserves_large_n() {
        let weights = real_weights();
        let width = weights.len();
        let sh = shared_with(&weights);
        for n in [
            1usize, 2, 4, 8, 16, 24, 32, 48, 64, 70, 96, 128, 192, 256, 384, 1024,
        ] {
            // (a) both candidates are disjoint partitions of `0..n`.
            let w_cov = coverage(&sh, n, true, width);
            let r_cov = coverage(&sh, n, false, width);
            assert!(
                w_cov.iter().all(|&o| o != usize::MAX),
                "weighted leaves an index unowned at n={n}"
            );
            assert!(
                r_cov.iter().all(|&o| o != usize::MAX),
                "round-robin leaves an index unowned at n={n}"
            );

            // (b) the chosen assignment is never the slower one under the exact model.
            let choose_w = sh.weighted(n);
            let (wn, wd) = makespan(&sh, n, width, true);
            let (rn, rd) = makespan(&sh, n, width, false);
            if choose_w {
                assert!(
                    wn * rd <= rn * wd,
                    "chose weighted but it is slower at n={n}: {wn}/{wd} vs {rn}/{rd}"
                );
            } else {
                assert!(
                    rn * wd <= wn * rd,
                    "chose round-robin but it is slower at n={n}: {rn}/{rd} vs {wn}/{wd}"
                );
            }

            // (c) participant 0 (the caller) always owns task 0.
            if choose_w {
                let (lo, hi) = sh.range(0, n);
                assert!(
                    lo < hi,
                    "caller idle under the chosen weighted split at n={n}"
                );
            } else {
                assert!(
                    round_robin_range(0, n, width).1 > 0,
                    "caller idle under the chosen round-robin at n={n}"
                );
            }
        }

        // (d) the weighted split is still chosen for the large dispatches it always served.
        assert!(
            sh.weighted(1024),
            "weighted split must be chosen at n = 1024"
        );
    }

    #[test]
    fn round_robin_count_at_width_24() {
        let width = 24usize;
        for n in 0..=128usize {
            let mut total = 0usize;
            for k in 0..width {
                let (start, count) = round_robin_range(k, n, width);
                if count > 0 {
                    assert!(start < n, "n={n} k={k} count={count}");
                    assert_eq!(start, k);
                    assert!(start + (count - 1) * width < n, "stride overruns n={n}");
                } else {
                    assert_eq!(start, k);
                    assert!(k >= n, "n={n} k={k} dropped");
                }
                total += count;
            }
            assert_eq!(total, n, "round-robin count does not cover n={n}");
        }
        assert_eq!(round_robin_range(0, 10, width), (0, 1));
        assert_eq!(round_robin_range(8, 33, width), (8, 2));
        assert_eq!(round_robin_range(23, 24, width), (23, 1));
        assert_eq!(round_robin_range(23, 23, width), (23, 0));
        assert_eq!(round_robin_range(24, 100, width), (24, 4));
    }

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_lossless,
        reason = "timing benchmark nanosecond conversions"
    )]
    #[ignore = "release-only pool microbench: cargo test --release --lib bench_pool_dispatch_latency -- --ignored --nocapture"]
    fn bench_pool_dispatch_latency() {
        let pool = Pool::global();
        println!("Pool width: {}", pool.width());

        // Warm up
        for _ in 0..5000 {
            pool.run(1, &|_| {});
            pool.run(24, &|_| {});
            pool.run(48, &|_| {});
        }

        let runs = 5;
        let reps = 20_000;

        let mut best_empty_1 = f64::INFINITY;
        let mut best_empty_24 = f64::INFINITY;
        let mut best_48 = f64::INFINITY;

        for r in 0..runs {
            let t0 = Instant::now();
            for _ in 0..reps {
                pool.run(1, &|_| {});
            }
            let el_1 = t0.elapsed().as_nanos() as f64 / reps as f64;
            if el_1 < best_empty_1 {
                best_empty_1 = el_1;
            }

            let t0 = Instant::now();
            for _ in 0..reps {
                pool.run(24, &|_| {});
            }
            let el_24 = t0.elapsed().as_nanos() as f64 / reps as f64;
            if el_24 < best_empty_24 {
                best_empty_24 = el_24;
            }

            let t0 = Instant::now();
            for _ in 0..reps {
                pool.run(48, &|_| {});
            }
            let el_48 = t0.elapsed().as_nanos() as f64 / reps as f64;
            if el_48 < best_48 {
                best_48 = el_48;
            }
            println!(
                "Run {r}: n=1: {:.2} ns ({:.3} us), n=24: {:.2} ns ({:.3} us), n=48: {:.2} ns ({:.3} us)",
                el_1,
                el_1 / 1000.0,
                el_24,
                el_24 / 1000.0,
                el_48,
                el_48 / 1000.0
            );
        }
        println!(
            "BEST of 5: n=1: {:.2} ns ({:.3} us), n=24: {:.2} ns ({:.3} us), n=48: {:.2} ns ({:.3} us)",
            best_empty_1,
            best_empty_1 / 1000.0,
            best_empty_24,
            best_empty_24 / 1000.0,
            best_48,
            best_48 / 1000.0
        );
    }
}
