//! The wait histogram is cleared by `trace::reset`, not only by its internal helper.
//!
//! The unit tests cover the bucket and class helpers; this test is an integration target so the
//! process has no other test recording `PoolWait` spans, which lets it assert exact emptiness
//! after `reset`. A mutation that drops the `clear_wait_table` call from `reset` turns it red.

use apeiron::engine::trace::{self, Phase};

/// A recorded `PoolWait` span is visible, and the next `reset` zeroes every bucket and max.
#[test]
fn reset_zeroes_the_wait_table() {
    trace::set_enabled(true);
    trace::reset();
    {
        let _wait = trace::span(Phase::PoolWait, 8, 0, 0);
    }
    let before = trace::pool_wait_histogram();
    assert!(
        before.buckets.iter().flatten().any(|&count| count > 0),
        "the seeding span counted nothing"
    );

    trace::reset();
    let after = trace::pool_wait_histogram();
    for (class, row) in after.buckets.iter().enumerate() {
        for (bucket, &count) in row.iter().enumerate() {
            assert_eq!(count, 0, "class {class} bucket {bucket}");
        }
    }
    for (class, &max) in after.max_ns.iter().enumerate() {
        assert_eq!(max, 0, "class {class} max");
    }
    trace::set_enabled(false);
}
