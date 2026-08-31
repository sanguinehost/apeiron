//! Checks the in-process counter against `perf stat` on the same work.
//!
//! Runs one loop whose instruction count is known by construction, reports what
//! `examples/support/counter.rs` read for it, and leaves the process doing nothing else so an
//! external `perf stat` on the whole process can be compared against the printed number.
//!
//! Run: `taskset -c 2 perf stat -e cpu_core/instructions/,cpu_core/cycles/ cargo run --release --example counter_check`

#[path = "support/counter.rs"]
mod counter;

use std::hint::black_box;

/// One `add` and one loop-carried dependency per iteration, out of line so the loop stays a loop.
#[inline(never)]
fn spin(n: u64) -> u64 {
    let mut a = 0u64;
    for i in 0..n {
        a = a.wrapping_add(black_box(i));
    }
    a
}

fn main() {
    let Some(c) = counter::Counter::open() else {
        println!("counters unavailable");
        return;
    };
    // Warm: the first count pays for page faults and the branch predictor.
    let _ = c.count(|| spin(1_000_000));
    let n = 100_000_000u64;
    let (v, counts) = c.count(|| spin(n));
    println!("value {v}");
    println!("in-process: {} instructions, {} cycles", counts.instructions, counts.cycles);
    #[expect(clippy::cast_precision_loss, reason = "a harness printing a rate")]
    let per = counts.instructions as f64 / n as f64;
    println!("per iteration: {per:.3} instructions");
}
