//! The fold alone, no table -- foldhash's "hashonly" context.
//!
//! A lookup is a fold plus a probe plus a compare. If apeiron's String lookup is behind, this says
//! whether the fold is the reason or whether the table is.
//!
//! # This is not a length sweep, and must not be read as one
//!
//! The four corpora are named for their *shape* -- `short`, `resref`, `path`, `long` -- and land
//! at roughly 2, 13, 45 and 231 bytes. `fold::absorb` routes on length at **0, 4, 8, 16, 32 and
//! 64**, and these four samples hit none of those. That blind spot hid a 2.17x gap at exactly
//! sixteen bytes -- the length `ResRef` pads to, and the key shape this crate's only consumer
//! indexes 113,489 resources by -- while the corpus *named* `resref` measured thirteen.
//!
//! It also **dilutes**: these are heap `String`s hashed in a tight loop, so the per-call ratio
//! here reads well below what the fold actually costs. It reported 1.20x at the resref corpus
//! where `examples/length_sweep.rs` counts 1.70x in instructions.
//!
//! Use this file for the question it answers -- fold versus table, on realistic key shapes -- and
//! `length_sweep.rs` for how the fold behaves across its branches.
#![expect(
    clippy::cast_precision_loss,
    reason = "this is a measurement harness, not library code, and each lint below is listed because this \
              file actually triggers it: counts converted to floats to report rates; a harness would need \
              2^53 samples to lose a bit. Listed individually rather than suppressed wholesale, so a \
              pedantic lint this file does not yet trigger still surfaces"
)]

use apeiron::fold::fold_bytes;
use std::hash::BuildHasher;
use std::hint::black_box;
use std::time::Instant;

fn time(reps: usize, mut f: impl FnMut() -> u64) -> f64 {
    for _ in 0..3 { black_box(f()); }
    let mut v: Vec<f64> = (0..9).map(|_| {
        let t = Instant::now();
        for _ in 0..reps { black_box(f()); }
        t.elapsed().as_secs_f64() * 1e9 / reps as f64
    }).collect();
    v.sort_by(f64::total_cmp);
    v[4]
}

fn corpus(kind: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| match kind {
        "resref" => format!("nw_wswmg{i:05}"),          // 13 bytes, the real shape
        "short"  => format!("{i}"),                     // 1-5
        "path"   => format!("data/meshes/creature/c_human_male_{i:04}.mdl"), // ~45
        _        => "x".repeat(200 + i % 64),           // long
    }).collect()
}

fn main() {
    let fold = foldhash::fast::RandomState::default();
    println!("  {:<10} {:>6} {:>12} {:>12} {:>12}", "corpus", "bytes", "apeiron ns", "foldhash ns", "ratio");
    for kind in ["short", "resref", "path", "long"] {
        let ks = corpus(kind, 1000);
        let avg: usize = ks.iter().map(String::len).sum::<usize>() / ks.len();
        let a = time(2000, || ks.iter().map(|s| fold_bytes(s.as_bytes())).fold(0, u64::wrapping_add)) / ks.len() as f64;
        let f = time(2000, || ks.iter().map(|s| fold.hash_one(s)).fold(0, u64::wrapping_add))
            / ks.len() as f64;
        println!("  {kind:<10} {avg:>6} {a:>12.2} {f:>12.2} {:>11.2}x", a / f);
    }
}
