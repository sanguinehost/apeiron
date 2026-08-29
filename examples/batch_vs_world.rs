//! The batch readers against the competition, not against ourselves.
//!
//! `get_all` and `encircle` are new API. Parity with apeiron's own loop says nothing about whether
//! they beat `std`, or `hashbrown` with `foldhash`, doing the same work the way a caller would
//! write it. This measures exactly that.
//!
//! Best of many rounds, arms alternated inside each round.
//!
//! Run: `cargo run --release --example batch_vs_world`
#![expect(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "this is a measurement harness: counts become floats to report nanoseconds per key, \
              and the arms are named for the libraries they time -- `a`, `w`, `s`, `f`, `x` line \
              the table up in the source the way it prints"
)]

use apeiron::map::{Map, Words};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::hint::black_box;
use std::time::Instant;

type Fold = foldhash::fast::RandomState;
type Fx = BuildHasherDefault<rustc_hash::FxHasher>;
type HbMap<K, V, S> = hashbrown::HashMap<K, V, S>;

fn best(rounds: usize, mut f: impl FnMut() -> u64) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        black_box(f());
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() {
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(9);
    println!("ns per key, best of {rounds}; * = best\n");

    for bits in [12u32, 16, 20, 22] {
        let n = 1usize << bits;
        let ks: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        let probe: Vec<u64> = ks.iter().rev().copied().collect();

        let ap: Map<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
        let aw: Words<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
        let st: HashMap<u64, u64> = ks.iter().map(|&k| (k, k)).collect();
        let hf: HbMap<u64, u64, Fold> = ks.iter().map(|&k| (k, k)).collect();
        let hx: HbMap<u64, u64, Fx> = ks.iter().map(|&k| (k, k)).collect();

        // -- bulk lookup, summed: get_all's job, written the way each library makes you write it.
        let d = n as f64;
        let a = best(rounds, || ap.get_reduce(&probe, 0u64, |v| *v, u64::wrapping_add));
        let w = best(rounds, || aw.get_reduce(&probe, 0u64, |v| *v, u64::wrapping_add));
        let s = best(rounds, || probe.iter().filter_map(|k| st.get(k)).sum());
        let f = best(rounds, || probe.iter().filter_map(|k| hf.get(k)).sum());
        let x = best(rounds, || probe.iter().filter_map(|k| hx.get(k)).sum());
        row("bulk lookup", n, &[a, w, s, f, x], d);

        // -- fold every value: encircle's job.
        let a = best(rounds, || ap.encircle(0u64, |v| *v, u64::wrapping_add));
        let w = best(rounds, || aw.encircle(0u64, |v| *v, u64::wrapping_add));
        let s = best(rounds, || st.values().fold(0u64, |a, v| a.wrapping_add(*v)));
        let f = best(rounds, || hf.values().fold(0u64, |a, v| a.wrapping_add(*v)));
        let x = best(rounds, || hx.values().fold(0u64, |a, v| a.wrapping_add(*v)));
        row("fold values", n, &[a, w, s, f, x], d);
    }
}

fn row(what: &str, n: usize, secs: &[f64], per: f64) {
    let ns: Vec<f64> = secs.iter().map(|s| s * 1e9 / per).collect();
    let best = ns.iter().copied().fold(f64::INFINITY, f64::min);
    print!("  {what:<13} {n:>9}");
    for v in &ns {
        let mark = if (v - best).abs() < f64::EPSILON { '*' } else { ' ' };
        print!(" {v:>9.2}{mark}");
    }
    println!("   (Map, Words, std, hb+fold, hb+fx)");
}
