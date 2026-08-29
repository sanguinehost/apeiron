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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static PEPPER: AtomicU64 = AtomicU64::new(0);
static IN_USE: AtomicBool = AtomicBool::new(false);

/// Set the pepper every later `Map::new` will use.
///
/// # Panics
///
/// If a table has already read it. See the module docs: a pepper that changes mid-process splits
/// the address space in two.
pub fn set_process_pepper(pepper: u64) {
    assert!(
        try_set_process_pepper(pepper).is_ok(),
        "the process pepper was already read by a table; setting it now would give the tables \
         built before this call a different address for every key than the ones built after"
    );
}

/// Set the pepper, reporting rather than panicking if it is already too late.
///
/// # Errors
///
/// [`PepperInUse`] when a table has already read the pepper.
pub fn try_set_process_pepper(pepper: u64) -> Result<(), PepperInUse> {
    if IN_USE.load(Ordering::Acquire) {
        return Err(PepperInUse { current: PEPPER.load(Ordering::Acquire) });
    }
    PEPPER.store(pepper, Ordering::Release);
    // Ordered after the store, so a racing reader either sees `IN_USE` false and has not read yet,
    // or sees the pepper this call set. It can never latch the old value and report it as in use.
    Ok(())
}

/// The pepper, and a note that it is now committed.
///
/// Called by `Map::new`. Reading is what freezes it -- after this, `set_process_pepper` fails.
#[must_use]
pub fn process_pepper() -> u64 {
    IN_USE.store(true, Ordering::Release);
    PEPPER.load(Ordering::Acquire)
}

/// The pepper, without committing to it. For diagnostics and for logging at startup.
#[must_use]
pub fn peek_process_pepper() -> u64 {
    PEPPER.load(Ordering::Acquire)
}

/// Whether any table has read the pepper yet.
#[must_use]
pub fn pepper_is_committed() -> bool {
    IN_USE.load(Ordering::Acquire)
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
