//! The process-wide address constant.
//!
//! [`crate::map::Salt`] explains what a pepper does and why it has to enter the address
//! *multiplicatively*. This module answers the other half: how it gets set without every call site
//! having to remember.
//!
//! # Why a floor is needed at all
//!
//! `Map::with_pepper` has existed since the salt did, and it is useless if nobody calls it. The
//! maps that need one are exactly the maps nobody thinks about -- a resource index keyed by names
//! out of an archive header, a field table keyed by labels read verbatim from a file someone
//! downloaded. Those are written once, early, by whoever was parsing the format, and they do not
//! look like security code. Measured, an aimed key set costs a `Map` **988x** its honest build
//! time, and the constants needed to aim it are in this crate's source.
//!
//! So the default is a floor rather than a footgun: one call at startup peppers every map the
//! process will ever make.
//!
//! ```
//! // once, in main, before anything builds a table
//! let pepper: u64 = 0x5EED_1234_ABCD_9876; // from config, or the OS at first run
//! apeiron::pepper::set_process_pepper(pepper);
//! println!("address salt: {pepper:#018x}"); // or `tracing::info!` -- but write it down
//! ```
//!
//! **Log it.** A pepper is a constant, not a per-run seed, and that is the whole reason this crate
//! can have one at all: a recorded pepper replays a recorded session exactly. A pepper nobody wrote
//! down is a random seed with extra steps.
//!
//! # Why setting it late is an error and not a warning
//!
//! Two maps in one process must address a key the same way, or a value moved from one to the other
//! is lost and a key looked up in the wrong one is silently absent. If the pepper changed after the
//! first map was built, half the process disagrees with the other half about where every key lives.
//! That is not a degraded mode, it is a corrupted one, and it would show up as an intermittent
//! missing resource a long way from the cause. So [`set_process_pepper`] panics rather than let it
//! happen, and [`try_set_process_pepper`] is there for a caller that wants to decide.

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
                let _ = STATE.compare_exchange(READY, COMMITTED, Ordering::AcqRel, Ordering::Acquire);
                return PEPPER.load(Ordering::Acquire);
            }
            _ => return PEPPER.load(Ordering::Acquire),
        }
    }
}

/// The pepper, without committing to it. For diagnostics and for logging at startup.
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
