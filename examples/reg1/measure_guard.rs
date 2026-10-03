//! The measurement precondition for `--check performance`: an exclusive advisory lock held across
//! the whole measurement run, and a quiet-box check that refuses by name instead of reporting.
//!
//! A `reg1` performance run is a timing claim. On a contended box the timing is a different
//! quantity, not a noisier estimate of the same one, so a contended run is **invalid for
//! measurement** and must not be printed as meeting or missing the floor. Reporting it anyway
//! invites exactly the wrong reading: a failing pass tally is then indistinguishable from a
//! failing floor. This module therefore has two parts.
//!
//! 1. **The lock.** `flock(LOCK_EX)` on one named file, taken before any timed region and released
//!    when the run ends. Cooperating invocations serialize. `flock` is attached to the open file
//!    description, so the kernel releases it when the process exits, abnormally as well as
//!    normally: there is no stale-lock state to clean up and no lease to expire.
//! 2. **The quiet-box check.** Four named readings taken *after* the lock is held and *before* any
//!    sample: the 1-minute load average from `/proc/loadavg` against `ncpu * 0.25`; the count of
//!    live processes other than this one and its ancestry whose command line carries a registered
//!    build/measurement token; and the host's CPU busy fraction over a one-second `/proc/stat`
//!    window against `0.05`, the same instrument and threshold `examples/engine_profile.rs` and
//!    the perf-attribution-prefill note already use. Any reading over its limit refuses
//!    by name. An instrument that cannot be read also refuses: a missing reading is not a quiet
//!    box.
//!
//! Each limit is overridable by environment variable so a deliberately contended run can be
//! reproduced, and every reading is printed on both dispositions, so an override cannot be
//! invisible in a receipt.
//!
//! Refusal is machine-visible on **stdout**, so a `--json` run carries it, and it exits
//! [`REFUSED_EXIT`], which is distinct from 0 (a valid report), 1 (an acceptance or validation
//! failure) and 64 (a usage error). A refusal prints no pass tally, no HEADLINE and no
//! `strict_acceptance` line at all: the run aborts before the first sample, so it has no tally to
//! report, and emitting any of those keys would invite a reader to take it for a verdict.
//!
//! Correctness is untouched. `--check bootstrap|correctness|coverage|history` takes no lock and
//! runs no quiet-box check, because their results do not depend on machine speed.
//!
//! Cost: `flock` is one non-blocking syscall, plus one per [`LOCK_POLL`] until the wait budget
//! expires; `/proc/loadavg` and the `/proc` scan are one directory read each; the busy fraction is
//! one [`BUSY_WINDOW`] sleep and two `/proc/stat` reads. All of it is wall-clock and process-scan
//! cost, incurred before the measurement and outside every timed sample region.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::performance::Profile;

/// The default named lock file. Overridable with [`ENV_LOCK`].
const DEFAULT_LOCK_PATH: &str = "/tmp/apeiron/reg1.lock";

/// The lock file path.
pub(crate) const ENV_LOCK: &str = "REG1_MEASURE_LOCK";
/// Seconds to wait for a held lock before refusing. Default `0.0`: refuse immediately rather than
/// queue behind a campaign of unknown duration.
pub(crate) const ENV_LOCK_WAIT_S: &str = "REG1_MEASURE_LOCK_WAIT_S";
/// The absolute `load1` limit, replacing `ncpu * 0.25`.
pub(crate) const ENV_LOAD1_LIMIT: &str = "REG1_LOAD1_LIMIT";
/// The host busy-fraction limit, replacing `0.05`.
pub(crate) const ENV_BUSY_LIMIT: &str = "REG1_HOST_BUSY_LIMIT";
/// `|`-separated command-line tokens that count as contention, replacing [`DEFAULT_TOKENS`]. Empty
/// disables the token check, and the printed `contention_tokens` field says so either way.
pub(crate) const ENV_TOKENS: &str = "REG1_CONTENTION_TOKENS";

/// The exit status of a refused measurement. Distinct from 0 (valid), 1 (acceptance or validation
/// failure) and 64 (usage), so no caller can read a refusal as an acceptance verdict.
pub(crate) const REFUSED_EXIT: i32 = 3;

/// The registered contention tokens: build, extraction and measurement processes that compete for
/// the same cores and memory bandwidth as the campaign. `lean4` rather than `lean`, because `lean`
/// is a substring of unrelated names such as `btrfs-cleaner`.
const DEFAULT_TOKENS: &[&str] = &[
    "reg1",
    "cargo",
    "rustc",
    "clippy-driver",
    "lake",
    "lean4",
    "cc1plus",
    "ninja",
    "make",
    "perf",
];

/// The registered `load1` limit as a fraction of the online CPUs: a box carrying more than a
/// quarter of its width in run-queue depth is not a quiet box.
const DEFAULT_LOAD1_FRACTION: f64 = 0.25;

/// The registered host busy-fraction limit, matching the quiet gate already used by
/// `examples/engine_profile.rs` and the perf-attribution-prefill note.
const DEFAULT_BUSY_LIMIT: f64 = 0.05;

/// The `/proc/stat` sampling window for the busy fraction. One second, as in
/// `examples/engine_profile.rs`.
const BUSY_WINDOW: Duration = Duration::from_secs(1);

/// The lock-wait poll interval.
const LOCK_POLL: Duration = Duration::from_millis(100);

/// A refusal: a named reason, the readings that produced it, and a human-readable detail.
pub(crate) struct Refusal {
    reason: &'static str,
    detail: String,
    reading: Reading,
}

impl Refusal {
    /// Print the refusal and return [`REFUSED_EXIT`]. The machine line goes to **stdout** so a
    /// `--json` run carries it; the human lines go to stderr so an unredirected terminal says why
    /// the run stopped.
    pub(crate) fn report(&self) -> i32 {
        println!(
            "{}",
            self.reading
                .line("refused", Some((self.reason, &self.detail)))
        );
        eprintln!(
            "reg1 refused to report a measurement: reason={} detail=\"{}\"",
            self.reason, self.detail
        );
        eprintln!(
            "no pass tally, HEADLINE or strict_acceptance line is reported; acceptance was not \
             evaluated and this is not an acceptance verdict"
        );
        REFUSED_EXIT
    }
}

/// The held lock. Dropping it, or calling [`Guard::release`], closes the descriptor, which releases
/// the `flock`.
pub(crate) struct Guard {
    file: File,
}

impl Guard {
    /// Release the lock now rather than at end of scope.
    pub(crate) fn release(self) {
        drop(self.file);
    }
}

/// Every preflight observation. All of it is printed on both dispositions, so a reader sees which
/// instrument was near its limit and which one fired.
struct Reading {
    profile: &'static str,
    accept: bool,
    lock_path: PathBuf,
    lock_state: &'static str,
    lock_wait_s: f64,
    ncpu: usize,
    load1: f64,
    load1_limit: f64,
    host_busy: f64,
    host_busy_limit: f64,
    contending: usize,
    tokens: String,
}

impl Reading {
    /// The one machine line. `disposition` is `ok` or `refused`; `reason` is present only on a
    /// refusal.
    fn line(&self, disposition: &'static str, reason: Option<(&'static str, &str)>) -> String {
        let mut line = format!(
            "measurement_precondition={disposition} measurement_valid={} profile={} accept={} \
             lock={} lock_state={} lock_wait_s={:.3} ncpu={} load1={:.2} load1_limit={:.2} \
             host_busy={:.4} host_busy_limit={:.4} contending_processes={} contention_tokens={}",
            if reason.is_some() { "false" } else { "true" },
            self.profile,
            self.accept,
            self.lock_path.display(),
            self.lock_state,
            self.lock_wait_s,
            self.ncpu,
            self.load1,
            self.load1_limit,
            self.host_busy,
            self.host_busy_limit,
            self.contending,
            self.tokens,
        );
        match reason {
            None => line
                .push_str(" measurement_verdict=measured exit_code=0 acceptance_evaluated=later"),
            Some((name, detail)) => {
                let _ = write!(
                    line,
                    " measurement_verdict=refused reason={name} detail=\"{}\" exit_code={} \
                     acceptance_evaluated=false",
                    crate::escape(detail),
                    REFUSED_EXIT
                );
            }
        }
        line
    }
}

/// Take the lock, take the quiet-box readings, and either return the held lock or a refusal. On
/// `Err` nothing has been timed and no measurement output exists.
pub(crate) fn acquire(profile: Profile, accept: bool) -> Result<Guard, Box<Refusal>> {
    let path = lock_path();
    let file = open_lock(&path)
        .map_err(|detail| Box::new(instrument_error(profile, accept, &path, detail)))?;
    let wait_s = match acquire_lock(&file) {
        Ok(wait_s) => wait_s,
        Err(LockFailure::Error(detail)) => {
            return Err(Box::new(instrument_error(profile, accept, &path, detail)));
        }
        Err(LockFailure::Contended(wait_s)) => {
            return Err(Box::new(Refusal {
                reason: "lock_contended",
                detail: format!(
                    "another cooperating reg1 measurement holds the exclusive lock {} after \
                     waiting {wait_s:.1}s of a {:.1}s budget; the holder's campaign perturbs this \
                     one, so this run is invalid for measurement",
                    path.display(),
                    lock_wait_budget_s()
                ),
                reading: Reading {
                    lock_state: "blocked",
                    lock_wait_s: wait_s,
                    ..unread(profile, accept, &path)
                },
            }));
        }
    };
    let reading = read_box(profile, accept, &path, wait_s)?;
    if let Some((reason, detail)) = verdict(&reading) {
        return Err(Box::new(Refusal {
            reason,
            detail,
            reading,
        }));
    }
    println!("{}", reading.line("ok", None));
    let _ = std::io::stdout().flush();
    Ok(Guard { file })
}

/// The named reason this reading set refuses, or `None` when the box is quiet. The order is fixed
/// and cheapest-instrument-first, so a refusal two instruments would both explain names the one
/// that is cheapest to re-check.
fn verdict(reading: &Reading) -> Option<(&'static str, String)> {
    if reading.load1 > reading.load1_limit {
        return Some((
            "load1_above_limit",
            format!(
                "the 1-minute load average {:.2} exceeds the registered limit {:.2} on {} online \
                 CPUs; a timing run on this box does not measure the table",
                reading.load1, reading.load1_limit, reading.ncpu
            ),
        ));
    }
    if reading.contending > 0 && !reading.tokens.is_empty() && reading.tokens != "none" {
        return Some((
            "contending_processes",
            format!(
                "{} live process(es) other than this one and its ancestry carry a registered \
                 build/measurement token ({}); a concurrent lane perturbs this campaign",
                reading.contending, reading.tokens
            ),
        ));
    }
    if reading.host_busy > reading.host_busy_limit {
        return Some((
            "host_busy_above_limit",
            format!(
                "the host CPU busy fraction {:.4} over a {}s window exceeds the registered limit \
                 {:.4}",
                reading.host_busy,
                BUSY_WINDOW.as_secs(),
                reading.host_busy_limit
            ),
        ));
    }
    None
}

/// Take every quiet-box reading. Each one fails closed: an instrument that cannot be read refuses
/// the run rather than passing it.
fn read_box(
    profile: Profile,
    accept: bool,
    path: &Path,
    wait_s: f64,
) -> Result<Reading, Box<Refusal>> {
    let fail = |detail: String| -> Box<Refusal> {
        Box::new(instrument_error(profile, accept, path, detail))
    };
    let ncpu = std::thread::available_parallelism()
        .map_err(|e| fail(format!("available_parallelism: {e}")))?
        .get();
    let load1_limit = match env_f64(ENV_LOAD1_LIMIT) {
        Ok(Some(v)) => v,
        Ok(None) => {
            #[allow(
                clippy::cast_precision_loss,
                reason = "a CPU count scaled by a registered fraction, compared as a limit"
            )]
            let derived = ncpu as f64 * DEFAULT_LOAD1_FRACTION;
            derived
        }
        Err(detail) => return Err(fail(detail)),
    };
    let busy_limit = match env_f64(ENV_BUSY_LIMIT) {
        Ok(Some(v)) => v,
        Ok(None) => DEFAULT_BUSY_LIMIT,
        Err(detail) => return Err(fail(detail)),
    };
    let (tokens, token_list) = env_tokens();
    let load1 = load1().map_err(fail)?;
    let contending = scan_contending(&tokens, &ancestry()).map_err(fail)?;
    let Some(host_busy) = host_busy() else {
        return Err(fail("/proc/stat has no cpu aggregate line".to_string()));
    };
    Ok(Reading {
        profile: profile.name(),
        accept,
        lock_path: path.to_path_buf(),
        lock_state: "held",
        lock_wait_s: wait_s,
        ncpu,
        load1,
        load1_limit,
        host_busy,
        host_busy_limit: busy_limit,
        contending,
        tokens: token_list,
    })
}

/// A reading for a failure that happened before, or instead of, a full set of instruments. The
/// unknown readings print as `nan` rather than `0`, so a reader cannot mistake "not measured" for
/// "measured and quiet".
fn unread(profile: Profile, accept: bool, path: &Path) -> Reading {
    Reading {
        profile: profile.name(),
        accept,
        lock_path: path.to_path_buf(),
        lock_state: "unread",
        lock_wait_s: 0.0,
        ncpu: 0,
        load1: f64::NAN,
        load1_limit: f64::NAN,
        host_busy: f64::NAN,
        host_busy_limit: f64::NAN,
        contending: 0,
        tokens: String::new(),
    }
}

/// A refusal from an instrument that could not be read or locked. The lock state is reported as
/// `unread` because the failure may have come before or after the lock was taken.
fn instrument_error(profile: Profile, accept: bool, path: &Path, detail: String) -> Refusal {
    Refusal {
        reason: "precondition_error",
        detail,
        reading: unread(profile, accept, path),
    }
}

/// Why the lock could not be taken.
enum LockFailure {
    /// Still held by another invocation after `f64` seconds of waiting.
    Contended(f64),
    /// The lock file could not be locked at all.
    Error(String),
}

/// The configured wait budget in seconds; `0.0` unless [`ENV_LOCK_WAIT_S`] says otherwise. An
/// unparsable budget is reported by `read_box` as a `precondition_error`; treating it as zero here
/// only decides how long to wait before that report is made.
fn lock_wait_budget_s() -> f64 {
    match env_f64(ENV_LOCK_WAIT_S) {
        Ok(Some(v)) => v.max(0.0),
        Ok(None) | Err(_) => 0.0,
    }
}

/// Take `LOCK_EX`, polling until the wait budget expires. Returns the seconds spent waiting, so a
/// run that queued behind another one records the wait in its report line rather than looking like
/// a run that never contended.
fn acquire_lock(file: &File) -> Result<f64, LockFailure> {
    let budget = Duration::from_secs_f64(lock_wait_budget_s());
    let started = Instant::now();
    loop {
        // SAFETY: `flock` reads only the descriptor and the lock state in the kernel.
        let rc = unsafe {
            libc::flock(
                std::os::fd::AsRawFd::as_raw_fd(file),
                libc::LOCK_EX | libc::LOCK_NB,
            )
        };
        if rc == 0 {
            return Ok(started.elapsed().as_secs_f64());
        }
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() != Some(libc::EWOULDBLOCK) {
            return Err(LockFailure::Error(format!("flock: {e}")));
        }
        if started.elapsed() >= budget {
            return Err(LockFailure::Contended(started.elapsed().as_secs_f64()));
        }
        std::thread::sleep(LOCK_POLL);
    }
}

/// The lock file path, from [`ENV_LOCK`] or the default.
fn lock_path() -> PathBuf {
    match std::env::var_os(ENV_LOCK) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => PathBuf::from(DEFAULT_LOCK_PATH),
    }
}

/// Open (creating) the lock file read-write. The file carries no data; `flock` is on the
/// descriptor, so its contents are irrelevant and no writer ever truncates it.
fn open_lock(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| format!("open lock {}: {e}", path.display()))
}

/// The 1-minute load average from `/proc/loadavg`.
fn load1() -> Result<f64, String> {
    let text =
        std::fs::read_to_string("/proc/loadavg").map_err(|e| format!("/proc/loadavg: {e}"))?;
    text.split_whitespace()
        .next()
        .ok_or_else(|| "/proc/loadavg has no first field".to_string())?
        .parse::<f64>()
        .map_err(|e| format!("/proc/loadavg first field: {e}"))
}

/// This process's pid and its whole ancestry. They are excluded from the contention scan, because
/// `cargo run --example reg1` and the shell that started it both carry a registered token and are
/// not competing with the campaign.
fn ancestry() -> BTreeSet<u32> {
    let mut set = BTreeSet::new();
    let mut pid = std::process::id();
    set.insert(pid);
    while let Some(parent) = parent_of(pid) {
        if !set.insert(parent) {
            break;
        }
        pid = parent;
    }
    set
}

/// The parent pid from `/proc/<pid>/stat`. `comm` is parenthesised and may contain spaces, so the
/// fields are read after the last `)`: `state` first, then `ppid`.
fn parent_of(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_comm = stat.rsplit_once(')')?.1;
    after_comm.split_whitespace().nth(1)?.parse().ok()
}

/// The number of live processes outside `skip` whose command line carries a registered token.
/// Kernel threads have an empty command line and never match.
fn scan_contending(tokens: &[&str], skip: &BTreeSet<u32>) -> Result<usize, String> {
    let mut count = 0usize;
    let entries = std::fs::read_dir("/proc").map_err(|e| format!("/proc: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("/proc entry: {e}"))?;
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        if skip.contains(&pid) {
            continue;
        }
        let Ok(raw) = std::fs::read(format!("/proc/{pid}/cmdline")) else {
            continue;
        };
        let cmd = String::from_utf8_lossy(&raw).replace('\0', " ");
        if tokens.iter().any(|t| cmd.contains(t)) {
            count += 1;
        }
    }
    Ok(count)
}

/// `(total, idle)` jiffies from the first `/proc/stat` line.
fn cpu_jiffies() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/stat").ok()?;
    let line = text.lines().next()?;
    let values: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|v| v.parse().ok())
        .collect();
    let idle = values.get(3)? + values.get(4).copied().unwrap_or(0);
    Some((values.iter().sum(), idle))
}

/// The host's CPU busy fraction over [`BUSY_WINDOW`], measured while this process sleeps.
#[allow(
    clippy::cast_precision_loss,
    reason = "a jiffy ratio over a window, used as a gate and printed to four places"
)]
fn host_busy() -> Option<f64> {
    let (t0, i0) = cpu_jiffies()?;
    std::thread::sleep(BUSY_WINDOW);
    let (t1, i1) = cpu_jiffies()?;
    let dt = t1.saturating_sub(t0).max(1);
    Some(1.0 - (i1.saturating_sub(i0) as f64 / dt as f64))
}

/// The registered tokens, overridden by [`ENV_TOKENS`], plus their printable form.
fn env_tokens() -> (Vec<&'static str>, String) {
    let Some(raw) = std::env::var_os(ENV_TOKENS) else {
        return (DEFAULT_TOKENS.to_vec(), DEFAULT_TOKENS.join("|"));
    };
    let text = raw.to_string_lossy().into_owned();
    if text.trim().is_empty() {
        return (Vec::new(), "none".to_string());
    }
    // The override is read once per invocation and must outlive the `Vec` the scan borrows, so the
    // trimmed tokens are leaked once. An empty override yields no tokens and the printed field
    // reads `none`, which is the report's statement that the check was disabled.
    let owned: Vec<&'static str> = text
        .split('|')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| &*Box::leak(t.to_string().into_boxed_str()))
        .collect();
    let list = owned.join("|");
    (owned, list)
}

/// Read a `f64` from the environment. `Ok(None)` for unset or blank, `Err` for set-and-unparsable.
fn env_f64(name: &str) -> Result<Option<f64>, String> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(None);
    };
    let text = raw.to_string_lossy().trim().to_string();
    if text.is_empty() {
        return Ok(None);
    }
    text.parse::<f64>()
        .map(Some)
        .map_err(|e| format!("{name}={text:?}: {e}"))
}
