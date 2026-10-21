//! A persistent worker pool for static-partition parallel passes over a table.
//!
//! The reduce is **schedule-free** — `Web.preduceAt_schedule_free` proves the answer is invariant
//! under any bijection relabeling which unit handles which task — so an equal partition needs no
//! work-stealing scheduler. The pool's only job is to *amortize thread creation*: `std::thread::scope`
//! spawns and joins fresh OS threads on every call (the measured ~4× CPU burn vs a warm pool), so we
//! park a fixed set of workers once and hand them a batch of task indices per dispatch.
//!
//! **The spin knee** (song `scope: encircle spin knee`, measured 92µs → 30µs/dispatch there; and
//! the decode regression that forced it here: 90 → 27 tok/s under pure condvar parking, ~97
//! dispatches/token each paying a futex wake-storm). Workers SPIN on the atomic epoch for
//! [`SPIN_KNEE`] pauses before parking on the condvar: back-to-back dispatches land in the spin
//! window (no futex, ~µs latency); an idle pool still parks (no burnt cores at rest). The hot
//! path is all atomics — epoch (`aepoch`), an epoch-tagged claim word (`aclaim`), a done counter
//! (`adone`) — the mutex/condvar pair survives only for parking and the panic stash.
//!
//! The claim word packs `(epoch << CLAIM_BITS) | next_index`, claimed by CAS: a stale executor
//! (parked through a whole batch, or preempted mid-claim across a batch boundary) fails the CAS
//! on the epoch bits and can never claim into a batch whose job pointer it did not acquire — the
//! use-after-free a bare `fetch_add` claim would allow.
//!
//! **And the claim tag alone was NOT enough — the 2026-08-07 stale-snapshot hole (core PID
//! 1283745).** `run_batch` loads `job` and `tasks` as separate atomics, and nothing tied them to
//! `epoch`: a worker that acquired epoch E and then stalled could resume after batch E completed,
//! load `ajob` from the post-batch scrub (`[0,0]`) or from batch E+1's publish, load `atasks`
//! from E+1, and still find `aclaim` at E's REST value `(E<<20)|tasks_E` — three individually
//! valid reads of three different instants. With `tasks_{E+1} > tasks_E` the claim passes
//! (`tasks_E < tasks_{E+1}`), the CAS on the stale rest word succeeds, and the worker calls a
//! null/franken fat pointer (the captured core: `call *0x28(0)`, job `[0,0]`, tag `0x6229`,
//! cur `0x6229<<20|24`, tasks 36) — or runs a task of the WRONG batch whose spurious `adone`
//! increment releases the next barrier one task early, leaving a straggler running on a dead
//! frame and freed buffers: garbage indices at the panic sites, silent corruption everywhere
//! else. Closed by `abegin`, a publish-generation word bumped BEFORE the batch is published and
//! validated in `run_batch` AFTER the job/tasks loads: any load that observed a newer batch's
//! store implies (release/acquire on that store, coherence on `abegin`) the validation read sees
//! the newer generation and refuses. The barrier then re-checks `adone == tasks` and the claim
//! word's final value exactly — a release-build refusal, not a `debug_assert`, because release
//! builds are the only place this class ever fired.
//!
//! [`Pool::dispatch`] runs `f(i)` for every `i in 0..tasks` across the workers **and the calling
//! thread**, and blocks until all complete. That barrier is what makes the lifetime erasure sound:
//! `f` and everything it borrows are alive for the whole call, so a worker holding a lifetime-erased
//! pointer to `f` never outlives it.
//!
//! [`Table::get_each`] is the single-core pipeline and does NOT use this pool (it already wins by
//! spawning nothing); the pool is only for the multi-core faces.
//!
//! **Panic safety.** A task panic is caught at the task boundary, stashed, and re-raised out of
//! [`Pool::dispatch`] only after the barrier — never unwound through a worker or the dispatcher while
//! the batch is live. So a panicking task can neither deadlock the barrier (a worker that unwound
//! past the done-increment would leave `done < tasks` forever) nor free the borrowed closure under
//! the other executors (the dispatcher unwinding out early is a use-after-free). The workers survive
//! the panic and park for the next batch; the caller of `dispatch` sees the first panic, and the
//! pool stays usable. This is the "unrecoverable panic across the shared address space" hazard,
//! closed.

use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

/// Spin iterations before a worker parks (and between the dispatcher's barrier yields). Song's
/// encircle measured the knee's worth; tens of µs of `spin_loop` covers the µs-scale gaps
/// between a decode's back-to-back dispatches without burning cores on an idle pool.
const SPIN_KNEE: u32 = 1 << 15;

/// Low bits of the claim word holding the next task index; the rest is the epoch tag.
/// `tasks` per dispatch must fit (dispatch asserts) — partitions are ≤ pool width or a few
/// thousand chunks, nowhere near 2^20.
const CLAIM_BITS: u32 = 20;
const CLAIM_MASK: u64 = (1 << CLAIM_BITS) - 1;

/// A raw `*mut T` made shareable across the pool's tasks. The caller is responsible for handing each
/// task a **disjoint** window (distinct task indices ⇒ non-overlapping writes), so the aliasing rules
/// are upheld by the partition, and [`Pool::dispatch`]'s barrier orders every write before the buffer
/// is read.
pub struct SendPtr<T>(pub *mut T);
// Manual Copy/Clone with no `T: Copy` bound (`*mut T` is always Copy, whatever T is).
impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
// SAFETY: soundness rests on the caller's disjoint-window discipline (see the type doc), not on T.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Offset the wrapped pointer by `i` elements. Written as a method so a task closure captures the
    /// whole (`Sync`) `SendPtr` rather than the raw field.
    ///
    /// # Safety
    /// `i` must index the buffer, and the disjointness contract of [`SendPtr`] must hold.
    #[inline]
    pub unsafe fn add(self, i: usize) -> *mut T {
        self.0.add(i)
    }
}

/// Cold-path state under the mutex: parking bookkeeping and the panic stash. The hot path never
/// takes this lock (executors in the spin window touch only the atomics).
struct Batch {
    epoch: u64, // mirror of `aepoch` for the condvar predicate
    shutdown: bool,
    // First task panic of the batch, stashed by whichever executor caught it and re-raised by the
    // dispatcher AFTER the barrier (see the module doc's panic-safety paragraph).
    panic: Option<Box<dyn Any + Send>>,
}

struct Shared {
    batch: Mutex<Batch>,
    work_ready: Condvar, // parked workers wait here for a new epoch (or shutdown)
    // Serializes DISPATCHERS: only one batch is ever live. Two threads calling `dispatch` at once
    // would clobber each other's epoch/tasks/job and corrupt both results. Held for the whole
    // dispatch, so concurrent callers queue. (The pool is flat data-parallel; a task must not
    // itself call `dispatch` — that would self-deadlock here, and is not a supported pattern.)
    dispatching: Mutex<()>,
    // ── the hot path ──
    aepoch: AtomicU64, // bumped per dispatch (Release); workers gate on it (Acquire)
    // Publish generation, stored (Release) BEFORE ajob/atasks/aclaim. `run_batch` validates it
    // AFTER loading job/tasks: a snapshot containing any newer batch's value cannot pass (the
    // newer publish stored `abegin` first, so reading its later stores makes the newer `abegin`
    // visible to the validation read). This is what ties job/tasks to the claim tag's epoch —
    // the 2026-08-07 stale-snapshot class, closed by construction.
    abegin: AtomicU64,
    ajob: [AtomicUsize; 2], // erased fat pointer to the closure, published before aepoch
    atasks: AtomicUsize,    // batch size, published before aepoch
    aclaim: AtomicU64,      // (epoch << CLAIM_BITS) | next-index — the epoch-tagged CAS claim
    adone: AtomicUsize,     // completed tasks; the dispatcher spins to `tasks`
    ashutdown: AtomicBool,  // spin-window shutdown check (parked workers use Batch.shutdown)
}

/// A fixed pool of parked worker threads plus the calling thread as one more executor. `shared` is an
/// `Arc` so workers hold clones and `drop` reclaims it once they join — no leak, no unsafe free.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    width: usize, // total executors = spawned workers + 1 (the dispatcher)
}

/// Reconstruct and invoke the erased closure for task `i`.
///
/// # Safety
/// `job` must be the two halves of a live `*const (dyn Fn(usize) + Sync)` — i.e. called only inside a
/// batch whose dispatcher is still blocked in [`Pool::dispatch`]. The `+ Sync` bound on the original
/// closure is what makes sharing `&f` across threads sound.
#[inline]
unsafe fn run_task(job: [usize; 2], i: usize) {
    let ptr: *const (dyn Fn(usize) + Sync) = std::mem::transmute(job);
    (*ptr)(i);
}

/// Run task `i`, catching a panic so it can be re-raised out of `dispatch` after the barrier instead
/// of unwound through this executor's frame (which would deadlock the barrier or free the shared
/// closure). `AssertUnwindSafe` is sound here: the pool's own state is behind atomics/the mutex and
/// the panic is faithfully propagated, so no observer sees a broken invariant across the unwind.
///
/// # Safety
/// Same contract as [`run_task`]: `job` must be a live erased `Fn(usize) + Sync`.
#[inline]
unsafe fn run_task_caught(job: [usize; 2], i: usize) -> Result<(), Box<dyn Any + Send>> {
    catch_unwind(AssertUnwindSafe(|| unsafe { run_task(job, i) }))
}

/// Claim-and-run every remaining task of epoch `epoch`. Shared by workers and the dispatcher.
/// The CAS on the epoch-tagged claim word is what makes a stale executor harmless: its epoch
/// bits no longer match, every CAS fails, it claims nothing.
fn run_batch(shared: &Shared, epoch: u64) {
    let job = [
        shared.ajob[0].load(Ordering::Acquire),
        shared.ajob[1].load(Ordering::Acquire),
    ];
    let tasks = shared.atasks.load(Ordering::Acquire) as u64;
    // THE STALE-SNAPSHOT GATE (2026-08-07, core PID 1283745). The three loads above and the
    // claim word below are four separate atomics; only this check proves they all belong to
    // `epoch`. If any load observed a batch newer than `epoch`, that batch stored `abegin`
    // first, so this read sees it and refuses. If `abegin` still reads `epoch`, then job and
    // tasks are `epoch`'s own (the scrub does not bump `abegin`, but a scrubbed job is only
    // observable with `epoch`'s own `tasks`, whose claim word is at rest ⇒ the loop below
    // claims nothing). Without this, a stalled worker claimed E's rest word with E+1's task
    // count and the scrubbed null job — `call *0x28(0)` — or double-ran a task and released
    // the next barrier early.
    if shared.abegin.load(Ordering::Acquire) != epoch {
        return;
    }
    let tag = epoch << CLAIM_BITS;
    loop {
        let cur = shared.aclaim.load(Ordering::Acquire);
        if cur & !CLAIM_MASK != tag {
            return; // a newer batch took over; this executor re-gates on the epoch
        }
        let i = cur & CLAIM_MASK;
        if i >= tasks {
            return;
        }
        if shared
            .aclaim
            .compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            continue;
        }
        // SAFETY: the dispatcher blocks until done == tasks, so `job` is live; the epoch-tagged
        // CAS above proves this claim belongs to the batch that published `job`.
        let r = unsafe { run_task_caught(job, i as usize) };
        if let Err(p) = r {
            let mut b = shared
                .batch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if b.panic.is_none() {
                b.panic = Some(p);
            }
        }
        // The panic (if any) is stashed BEFORE the increment, so when the dispatcher sees
        // done == tasks every panic is already visible.
        shared.adone.fetch_add(1, Ordering::AcqRel);
    }
}

/// One worker: spin on the epoch for the knee, park if nothing arrives, then run the batch.
fn worker(shared: &Shared) {
    let mut seen = 0u64;
    loop {
        let mut spins = 0u32;
        let epoch = loop {
            let e = shared.aepoch.load(Ordering::Acquire);
            if e != seen {
                break e;
            }
            if shared.ashutdown.load(Ordering::Acquire) {
                return;
            }
            std::hint::spin_loop();
            spins += 1;
            if spins >= SPIN_KNEE {
                // ── park (the cold path) ──
                let mut b = shared
                    .batch
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                while b.epoch == seen && !b.shutdown {
                    b = shared
                        .work_ready
                        .wait(b)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                if b.shutdown {
                    return;
                }
                break b.epoch;
            }
        };
        seen = epoch;
        run_batch(shared, epoch);
    }
}

/// Physical core count via sysfs (Linux): distinct `core_cpus_list` strings across cpus — HT
/// siblings share the string, so the distinct count is the physical count. `None` elsewhere
/// (the caller falls back to `available_parallelism`).
fn physical_cores() -> Option<usize> {
    let dir = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut cores: Vec<String> = Vec::new();
    for e in dir.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let p = e.path().join("topology/core_cpus_list");
        if let Ok(list) = std::fs::read_to_string(&p) {
            let list = list.trim().to_string();
            if !cores.contains(&list) {
                cores.push(list);
            }
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

impl Pool {
    /// The process-wide pool, sized to the physical core count (see [`Pool::new`]).
    pub fn global() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(Pool::new)
    }

    fn new() -> Pool {
        // Default width = PHYSICAL cores, not available_parallelism: the OS count includes HT
        // siblings, and a spinner parked on the sibling of a busy P-core steals real throughput
        // from it (measured on i9-14900KF: width 24 = 103 tok/s decode, width 32 = 90).
        // HASHTRINITY_POOL_WIDTH overrides either way.
        let width = std::env::var("HASHTRINITY_POOL_WIDTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&w| w >= 1)
            .or_else(physical_cores)
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
            });
        let shared = Arc::new(Shared {
            batch: Mutex::new(Batch {
                epoch: 0,
                shutdown: false,
                panic: None,
            }),
            work_ready: Condvar::new(),
            dispatching: Mutex::new(()),
            aepoch: AtomicU64::new(0),
            abegin: AtomicU64::new(0),
            ajob: [AtomicUsize::new(0), AtomicUsize::new(0)],
            atasks: AtomicUsize::new(0),
            aclaim: AtomicU64::new(0),
            adone: AtomicUsize::new(0),
            ashutdown: AtomicBool::new(false),
        });
        // width-1 spawned workers; the dispatching (calling) thread is the width-th executor.
        let workers = (0..width.saturating_sub(1))
            .map(|_| {
                let s = Arc::clone(&shared);
                std::thread::spawn(move || worker(&s))
            })
            .collect();
        Pool {
            shared,
            workers,
            width,
        }
    }

    /// The number of executors (spawned workers + the calling thread).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Run `f(i)` for every `i in 0..tasks` across the pool and the calling thread; block until all
    /// complete. `f` may borrow freely — it only has to outlive this call, which the barrier enforces.
    pub fn dispatch<F: Fn(usize) + Sync>(&self, tasks: usize, f: F) {
        if tasks == 0 {
            return;
        }
        if self.width <= 1 || tasks == 1 {
            for i in 0..tasks {
                f(i);
            }
            return;
        }
        assert!(
            (tasks as u64) < CLAIM_MASK,
            "dispatch batch too large for the claim word"
        );
        // Only one dispatcher at a time (see Shared.dispatching). Poison-tolerant: it guards
        // mutual exclusion, not data, and is released before any panic re-raise.
        let dispatching_guard = self
            .shared
            .dispatching
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Erase the closure to a lifetime-free fat-pointer pair. Sound because we block below until
        // done == tasks, so no worker observes `f` after this call returns.
        let ptr: *const (dyn Fn(usize) + Sync) = &f;
        let job: [usize; 2] = unsafe { std::mem::transmute(ptr) };

        let epoch = self.shared.aepoch.load(Ordering::Relaxed) + 1;
        // Publish the batch: the generation word FIRST (what makes a torn cross-batch snapshot
        // detectable in `run_batch`), then job/tasks/done/claim, epoch LAST (Release) — an
        // executor that acquires the new epoch therefore sees the whole batch.
        self.shared.abegin.store(epoch, Ordering::Release);
        self.shared.ajob[0].store(job[0], Ordering::Release);
        self.shared.ajob[1].store(job[1], Ordering::Release);
        self.shared.atasks.store(tasks, Ordering::Release);
        self.shared.adone.store(0, Ordering::Release);
        self.shared
            .aclaim
            .store(epoch << CLAIM_BITS, Ordering::Release);
        self.shared.aepoch.store(epoch, Ordering::Release);
        // Wake any PARKED workers (spinners saw the epoch already; an empty wake is cheap).
        {
            let mut b = self
                .shared
                .batch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            b.epoch = epoch;
            self.shared.work_ready.notify_all();
        }
        // The dispatcher is one executor too.
        run_batch(&self.shared, epoch);
        // ── the barrier: spin (with yields past the knee) until every task completed ──
        let mut spins = 0u32;
        while self.shared.adone.load(Ordering::Acquire) < tasks {
            std::hint::spin_loop();
            spins += 1;
            if spins >= SPIN_KNEE {
                std::thread::yield_now();
                spins = 0;
            }
        }
        // THE BARRIER'S OWN REFUSAL (release-mode, deliberately not a debug_assert — release
        // builds are the only place this class ever fired). With the stale-snapshot gate above,
        // exactly `tasks` claims occur and each increments `adone` once, so both words have one
        // legal final value. Anything else is a cross-batch claim or a straggler and the
        // dispatch that suffered it names itself here, before its buffers are freed.
        let done = self.shared.adone.load(Ordering::Acquire);
        let claim = self.shared.aclaim.load(Ordering::Acquire);
        assert!(
            done == tasks && claim == (epoch << CLAIM_BITS) + tasks as u64,
            "pool batch accounting corrupt after the barrier: done {done} vs tasks {tasks}, \
             claim {claim:#x} vs {:#x} — a stale executor claimed across batches (the \
             2026-08-07 class); this dispatch's buffers must not be trusted",
            (epoch << CLAIM_BITS) + tasks as u64
        );
        // Scrub the job pointer (stale executors CAS-fail on the epoch anyway; this is hygiene).
        self.shared.ajob[0].store(0, Ordering::Release);
        self.shared.ajob[1].store(0, Ordering::Release);
        // Every task completed and no worker still touches `f`; only now is it sound to re-raise
        // a stashed task panic. Release dispatcher exclusion first so the lock is never held
        // across the unwind and a queued dispatcher may proceed.
        let panic = {
            let mut b = self
                .shared
                .batch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            b.panic.take()
        };
        drop(dispatching_guard);
        if let Some(p) = panic {
            resume_unwind(p);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.ashutdown.store(true, Ordering::Release);
        {
            let mut b = self
                .shared
                .batch
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            b.shutdown = true;
            self.shared.work_ready.notify_all();
        }
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
        // Workers joined ⇒ their `Arc<Shared>` clones are gone; the pool's own `Arc` drops with the
        // struct, freeing `Shared`. The global pool lives in a `OnceLock` static and is never dropped
        // (process-lived); only explicitly-dropped pools (e.g. tests) reach here.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every task index runs exactly once, results land in the right (disjoint) slots, and repeated
    /// dispatches on the same pool reuse the workers. Small enough for Miri (real threads).
    #[test]
    fn dispatch_runs_each_task_once_into_disjoint_slots() {
        let pool = Pool::new();
        for &tasks in &[0usize, 1, 2, 7, 32] {
            // count total invocations
            let hits = AtomicUsize::new(0);
            // per-task output written through a raw pointer, one slot each (disjoint)
            let mut out = vec![0usize; tasks.max(1)];
            let p = SendPtr(out.as_mut_ptr());
            pool.dispatch(tasks, |t| {
                hits.fetch_add(1, Ordering::Relaxed);
                // SAFETY: task t writes only slot t; disjoint; barrier orders before the read below.
                unsafe { *p.add(t) = t * t };
            });
            assert_eq!(
                hits.load(Ordering::Relaxed),
                tasks,
                "each task once (tasks={tasks})"
            );
            for (t, &v) in out.iter().enumerate().take(tasks) {
                assert_eq!(v, t * t, "task {t} wrote its own slot");
            }
        }
        // a second batch on the same (warm) pool still works
        let hits = AtomicUsize::new(0);
        pool.dispatch(16, |_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 16, "warm pool reused");
    }

    /// Concurrent dispatchers on ONE pool do not corrupt each other's batch. Many threads each run a
    /// dispatch that sums 0..100 through disjoint per-task slots and checks the total; without the
    /// dispatching lock the epoch/job clobber gives wrong sums (the latent race the encircle test
    /// caught intermittently).
    #[test]
    fn concurrent_dispatchers_do_not_clobber() {
        let pool = Pool::new();
        if pool.width() <= 1 {
            return;
        }
        let pool = &pool;
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(move || {
                    for _ in 0..50 {
                        let n = 100usize;
                        let mut out = vec![0usize; n];
                        let p = SendPtr(out.as_mut_ptr());
                        pool.dispatch(n, |t| {
                            // SAFETY: task t writes only slot t; disjoint.
                            unsafe { *p.add(t) = t };
                        });
                        assert_eq!(
                            out.iter().sum::<usize>(),
                            n * (n - 1) / 2,
                            "batch not clobbered"
                        );
                    }
                });
            }
        });
    }

    /// Reproduces the 2026-08-07 stale-claim TOCTOU (core PID 1283745, `SEGV_MAPERR` at
    /// `call *0x28(0)` inside `run_batch`): a worker that acquired epoch E can stall between
    /// batches, then load `ajob` from the post-E scrub (or from E+1's publish), `atasks` from
    /// E+1, and the claim word from E's REST value — the three loads are individually valid
    /// snapshots of different instants, and nothing ties them to `epoch`. If E+1 has more tasks
    /// than E, `i = tasks_E < tasks_{E+1}` passes, the CAS on E's rest word succeeds, and the
    /// worker either calls a null/franken fat pointer (the captured core: job = [0,0], tag
    /// 0x6229, cur = 0x6229<<20|24, tasks = 36) or runs a task of the WRONG batch, whose
    /// spurious `adone` increment releases the next barrier one task early — the straggler
    /// then reads a dead frame (the `2^63+25` garbage-index panics) or writes freed buffers
    /// (silent corruption).
    ///
    /// The shape is `GpuPrefillStack::run`'s own: a 24-task readback copy followed by a
    /// 36-task KV fold, back-to-back, forever. Oversubscribing the pool keeps stragglers
    /// entering `run_batch` late on every batch. Detection: `hits != tasks` for any batch
    /// (an over-count is a cross-batch claim; an under-count is an early barrier release),
    /// a per-slot total that does not match the batch schedule, or the process dying on the
    /// null fat pointer. Pre-fix this fails in seconds; post-fix it must be silent.
    #[test]
    fn pool_publish_claim_consistency_under_batch_churn() {
        // SAFETY(env): set before this test's own pool is built; other tests in this module
        // build their own pools and are width-agnostic.
        unsafe { std::env::set_var("HASHTRINITY_POOL_WIDTH", "64") };
        let pool = Pool::new();
        if pool.width() <= 2 {
            return;
        }
        let (small, large) = (24usize, 36usize); // the readback -> fold pattern from the core
        let iters: usize = std::env::var("HASHTRINITY_POOL_STRESS_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(400_000);
        let slots: Vec<AtomicUsize> = (0..large).map(|_| AtomicUsize::new(0)).collect();
        let slots = &slots;
        for k in 0..iters {
            let tasks = if k % 2 == 0 { small } else { large };
            let hits = AtomicUsize::new(0);
            pool.dispatch(tasks, |i| {
                hits.fetch_add(1, Ordering::Relaxed);
                slots[i].fetch_add(1, Ordering::Relaxed);
            });
            let h = hits.load(Ordering::Relaxed);
            assert_eq!(
                h, tasks,
                "batch {k}: {h} hits for {tasks} tasks — a cross-batch claim (over) or an \
                 early barrier release (under); the 2026-08-07 stale-claim class"
            );
        }
        // Slot audit: index i is hit once per batch whose task count exceeds i.
        let halves = iters / 2; // small batches at even k, large at odd
        for (i, s) in slots.iter().enumerate() {
            let want = if i < small { iters } else { halves };
            assert_eq!(
                s.load(Ordering::Relaxed),
                want,
                "slot {i} total wrong — a task ran under the wrong batch"
            );
        }
    }

    /// A panicking task neither deadlocks the barrier nor frees the shared closure: the panic
    /// propagates out of `dispatch`, and the pool is still usable afterward. Repeated so the panicking
    /// index is claimed by a worker in some runs and by the dispatcher in others (both the
    /// deadlock and the use-after-free paths). Runs under Miri.
    #[test]
    fn task_panic_propagates_and_pool_survives() {
        let pool = Pool::new();
        if pool.width() <= 1 {
            return; // the panic paths only exist on the pooled (multi-executor) route
        }
        // Silence the default panic hook: we deliberately provoke caught panics.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for round in 0..8 {
            let n = 64;
            let victim = (round * 7 + 3) % n; // vary which task panics across rounds
            let ran = AtomicUsize::new(0);
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                pool.dispatch(n, |t| {
                    ran.fetch_add(1, Ordering::Relaxed);
                    assert_ne!(t, victim, "boom");
                });
            }));
            assert!(
                r.is_err(),
                "the task panic propagates out of dispatch (round {round})"
            );
            // The barrier still completed — otherwise dispatch would have hung, not returned Err.
            // The pool survived: a subsequent normal dispatch runs every task.
            let hits = AtomicUsize::new(0);
            pool.dispatch(48, |_| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(
                hits.load(Ordering::Relaxed),
                48,
                "pool usable after a task panic (round {round})"
            );
        }
        std::panic::set_hook(prev);
    }
}
