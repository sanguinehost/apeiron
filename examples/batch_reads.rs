//! Does pipelining a batch of lookups beat looping over them?
//!
//! No. This harness is the evidence for that, and it is kept because the answer is not the
//! intuitive one.
//!
//! `Map::get_each` was written as a software pipeline: prefetch the group of the key sixteen
//! positions ahead, so the misses of independent lookups overlap instead of queueing. It lost at
//! every size from 1,024 entries to 8.4 million -- 16.17 ns against 20.19 at the top -- and still
//! lost with every address hoisted out of the loop so the prefetch cost only its instruction.
//! Nothing in the loop depends on the previous iteration, so an out-of-order core already has
//! several probes in flight; there was no queue to unblock. `get_each` is now a plain loop and
//! this should read about 1.00x.
//!
//! Best of many rounds, arms alternated inside each round: a loaded machine can only make a round
//! slower, so the minimum is the closest a contended machine gets to the uncontended cost.
//!
//! Run: `cargo run --release --example batch_reads`
#![expect(
    clippy::cast_precision_loss,
    reason = "this is a measurement harness, not library code: counts become floats to report \
              nanoseconds per lookup, and it would need 2^53 samples to lose a bit"
)]

use apeiron::map::Map;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(9);
    println!("  {:>10} {:>9} {:>11} {:>11} {:>9}", "entries", "MiB", "loop ns", "get_each ns", "ratio");

    for bits in [10u32, 14, 16, 18, 20, 22, 23] {
        let n = 1usize << bits;
        let m: Map<u64, u64> = (0..n as u64).map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15), i)).collect();
        // Probe in an order unrelated to insertion, so the walk is not accidentally sequential.
        let keys: Vec<u64> = (0..n as u64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .rev()
            .collect();

        let (mut best_loop, mut best_each) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..rounds {
            let t = Instant::now();
            let mut acc = 0u64;
            for k in &keys {
                acc += m.get(k).copied().unwrap_or(0);
            }
            black_box(acc);
            best_loop = best_loop.min(t.elapsed().as_secs_f64() * 1e9 / n as f64);

            let t = Instant::now();
            let mut acc = 0u64;
            m.get_each(&keys, |_, v| acc += v.copied().unwrap_or(0));
            black_box(acc);
            best_each = best_each.min(t.elapsed().as_secs_f64() * 1e9 / n as f64);

        }
        // 40 bytes a slot at the 7/8 fill the table keeps.
        let mib = (m.slots() * 40) as f64 / (1024.0 * 1024.0);
        println!(
            "  {n:>10} {mib:>9.1} {best_loop:>11.2} {best_each:>11.2} {:>8.2}x",
            best_loop / best_each
        );
    }
}
