//! Exactly one caller may set the process pepper, whoever else is racing.
//!
//! Its own test binary, like `the_pepper_is_the_defence.rs`, because the pepper is a pair of
//! statics and a test that sets it cannot share a process with one that does not.
//!
//! The bug this pins: the setter used to load an `AtomicBool`, then store an `AtomicU64`, with
//! nothing between them. Two threads arriving together could both read the flag as clear and both
//! store, and the loser's pepper would be the one in force with no error reported anywhere. On a
//! server that initialises subsystems in parallel that is not a hypothetical ordering -- it is the
//! ordinary one. The state is one word now and every transition is a `compare_exchange`.

#![expect(
    clippy::items_after_statements,
    reason = "this is a test, not library code, and each lint below is listed because this file actually \
              triggers it: constants declared next to the loop that uses them. Listed individually rather \
              than suppressed wholesale, so a pedantic lint this file does not yet trigger still surfaces"
)]

use apeiron::map::Map;
use apeiron::pepper::{
    peek_process_pepper, pepper_is_committed, process_pepper, try_set_process_pepper,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

#[test]
fn exactly_one_setter_wins_a_race_and_everyone_reads_the_winner() {
    assert!(!pepper_is_committed(), "something touched the pepper before this test");
    assert_eq!(peek_process_pepper(), 0);

    const THREADS: usize = 32;
    let barrier = Barrier::new(THREADS);
    let winners = AtomicUsize::new(0);
    // Every candidate is nonzero and distinct, so the value that ends up in force names its setter.
    let candidate = |i: usize| (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;

    let won: Vec<Option<u64>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                let barrier = &barrier;
                let winners = &winners;
                s.spawn(move || {
                    // Release them all at once, so the window between the check and the store is
                    // actually contended rather than stepped through.
                    barrier.wait();
                    let mine = candidate(i);
                    if try_set_process_pepper(mine).is_ok() {
                        winners.fetch_add(1, Ordering::Relaxed);
                        Some(mine)
                    } else {
                        None
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread")).collect()
    });

    assert_eq!(
        winners.load(Ordering::Relaxed),
        1,
        "{} setters succeeded; the transition is not atomic",
        winners.load(Ordering::Relaxed)
    );
    let winner = won.into_iter().flatten().next().expect("someone must win");

    // And the value in force is the winner's, not a later store that raced past the check.
    assert_eq!(peek_process_pepper(), winner);
    assert_eq!(process_pepper(), winner);
    assert!(pepper_is_committed());

    // Every table built after the race agrees, which is the property the whole thing exists for:
    // two maps in one process must address a key the same way.
    let a: Map<String, u32> = Map::new();
    let b: Map<String, u32> = Map::new();
    assert_eq!(a.pepper(), winner);
    assert_eq!(b.pepper(), winner);

    // A setter arriving after the race loses, and changes nothing.
    assert!(try_set_process_pepper(0xDEAD_BEEF).is_err());
    assert_eq!(peek_process_pepper(), winner);
}
