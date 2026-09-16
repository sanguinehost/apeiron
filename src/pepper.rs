//! The process-wide address constant.
//!
//! [`set_process_pepper`] once at startup; every `Map` and `CountTable` made afterwards picks it
//! up. [`process_pepper`] reads and commits it, [`peek_process_pepper`] reads without committing.
//! Setting it after any table has been built is an error, not a warning: those tables keep the old
//! constant and the process ends up with two address functions.
//!
//! **Why a floor.** `Map::with_pepper` is useless if nobody calls it, and the maps that need one
//! are the ones nobody thinks about -- a resource index keyed by names out of an archive header.
//!
//! **Why it is still needed** now that `place` is non-invertible. Without a pepper the placement
//! is a public deterministic function, so an attacker searches once, offline, and reuses the key
//! set against every such process forever. `examples/hashdos.rs` puts that search at 63,839 trials
//! per key for sixteen bits -- a real cost, paid once by the attacker, never by a victim. A pepper
//! makes it un-amortisable: they cannot tell which candidates to keep.
//!
//! ```
//! // once, in main, before anything builds a table
//! let pepper: u64 = 0x5EED_1234_ABCD_9876; // from config, or the OS at first run
//! apeiron::pepper::set_process_pepper(pepper);
//! println!("address salt: {pepper:#018x}"); // write it down
//! ```
//!
//! **Log it.** A pepper is a constant, not a per-run seed, which is what lets a recorded session
//! replay exactly. One nobody wrote down is a random seed with extra steps.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

static PEPPER: AtomicU64 = AtomicU64::new(0);

/// The pepper's lifecycle, as one word so it can be moved atomically.
///
/// It was two -- an `AtomicU64` and an `AtomicBool` -- and the setter loaded the flag, then stored
/// the pepper, with nothing between them. Two threads racing at startup could both read the flag as
/// clear and both store, and the loser's pepper would be the one in force with no error reported.
/// The comment that used to sit here claimed an ordering guarantee the code did not implement; the
/// state is now one word and every transition is a `compare_exchange`, so the claim and the code
/// are the same thing.
static STATE: AtomicU8 = AtomicU8::new(UNSET);

/// Nobody has set or read it.
const UNSET: u8 = 0;
/// A setter has won the race and is writing `PEPPER`. Held for exactly one store.
const SETTING: u8 = 1;
/// Set and readable, not yet read.
const READY: u8 = 2;
/// A table has read it; it can no longer change.
const COMMITTED: u8 = 3;

/// Set the pepper every later `Map::new` will use.
///
/// Temporal and spatial O(1): one [`try_set_process_pepper`], which is one `compare_exchange` and
/// two atomic stores on the winning path; a loss panics rather than retrying, and the `PepperInUse`
/// it formats carries one `u64`. No allocation of its own.
///
/// # Panics
///
/// If a table has already read it. See the module docs: a pepper that changes mid-process splits
/// the address space in two.
pub fn set_process_pepper(pepper: u64) {
    assert!(
        try_set_process_pepper(pepper).is_ok(),
        "the process pepper was already set or read; changing it now would give the tables built \
         before this call a different address for every key than the ones built after"
    );
}

/// Set the pepper, reporting rather than panicking if it is already too late.
///
/// Temporal O(1) expected: one `compare_exchange` and, for the winner, two atomic stores; a loser
/// builds [`PepperInUse`] through [`peek_process_pepper`], which spins only while another thread
/// holds `SETTING` for one store, so the worst case is bounded by that store and not by the number
/// of contenders. Spatial O(1): one `u64` in the error value, no allocation.
///
/// Succeeds for exactly one caller. A second setter loses whether or not a table has read yet,
/// which is stricter than the old behaviour and is the right rule: two calls disagreeing about the
/// pepper is a bug in the caller either way.
///
/// # Errors
///
/// [`PepperInUse`] when the pepper is already set or already read.
pub fn try_set_process_pepper(pepper: u64) -> Result<(), PepperInUse> {
    // Only one thread can move UNSET -> SETTING, so only one can reach the store.
    match STATE.compare_exchange(UNSET, SETTING, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {
            PEPPER.store(pepper, Ordering::Release);
            // Publishing READY is what makes the store visible to a reader; until then a reader
            // spins rather than latching the zero it would otherwise see.
            STATE.store(READY, Ordering::Release);
            Ok(())
        }
        Err(_) => Err(PepperInUse {
            current: peek_process_pepper(),
        }),
    }
}

/// The pepper, and a note that it is now committed.
///
/// Temporal O(1) expected: one atomic load, then on `UNSET` one `compare_exchange`, on `READY` one
/// `compare_exchange` and one load, and one load on `COMMITTED`. The `loop` spins only on `SETTING`,
/// which a setter holds for exactly one store, so a pass is bounded by that store and occurs only
/// during startup. Spatial O(1): no allocation.
///
/// Called by `Map::new`. Reading is what freezes it -- after this, `set_process_pepper` fails.
#[must_use]
pub fn process_pepper() -> u64 {
    loop {
        match STATE.load(Ordering::Acquire) {
            // Nobody set one. Commit to zero, so a setter arriving later loses rather than
            // splitting the process between peppered and unpeppered tables.
            UNSET => {
                if STATE
                    .compare_exchange(UNSET, COMMITTED, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return 0;
                }
            }
            // A setter holds the window for one store. Spinning here is bounded by that store and
            // can only happen during startup.
            SETTING => std::hint::spin_loop(),
            READY => {
                let _ =
                    STATE.compare_exchange(READY, COMMITTED, Ordering::AcqRel, Ordering::Acquire);
                return PEPPER.load(Ordering::Acquire);
            }
            _ => return PEPPER.load(Ordering::Acquire),
        }
    }
}

/// The pepper, without committing to it. For diagnostics and for logging at startup.
///
/// Temporal O(1) expected: one atomic load; the `loop` spins only while a setter holds `SETTING` for
/// one store, so a pass is bounded by that store. Spatial O(1): no allocation.
#[must_use]
pub fn peek_process_pepper() -> u64 {
    loop {
        match STATE.load(Ordering::Acquire) {
            UNSET => return 0,
            SETTING => std::hint::spin_loop(),
            _ => return PEPPER.load(Ordering::Acquire),
        }
    }
}

/// Whether the pepper can still change.
#[must_use]
pub fn pepper_is_committed() -> bool {
    matches!(STATE.load(Ordering::Acquire), READY | COMMITTED)
}

/// The pepper was already in use when someone tried to change it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PepperInUse {
    /// The value that is already committed.
    pub current: u64,
}

impl std::fmt::Display for PepperInUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the process pepper is already committed to {:#018x}; tables built before and after a \
             change would disagree about where every key lives",
            self.current
        )
    }
}

impl std::error::Error for PepperInUse {}
