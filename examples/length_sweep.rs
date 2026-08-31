//! Every length across every branch boundary, against foldhash, counted in instructions.
//!
//! **This file exists because `hash_only.rs` was not a sweep.** It samples four corpora named for
//! their shape -- `short`, `resref`, `path`, `long` -- at roughly 2, 13, 45 and 231 bytes. The
//! fold branches on length at **0, 4, 8, 16, 32 and 64**, and those four samples hit *none* of
//! them. A 2.5x gap sat at exactly sixteen bytes, the length `ResRef` pads to and the most
//! important key shape in this crate's only consumer, while the corpus named `resref` tested
//! thirteen. A function that routes on length must be measured at every boundary and on both
//! sides of it.
//!
//! One length and one arm per process, selected by `LEN` and `ARM`, driven under `perf stat` --
//! **not** wall time. The first version of this file timed a closure per length and reported
//! 0.98 ns at eight bytes against 2.24 at fifteen, which take the *same branch* and cannot
//! differ; at this scale `Instant` measures scheduling, not the fold. Instructions are exact and
//! reproduce to two decimal places across runs.
use std::hash::{BuildHasher, Hash, Hasher};
use std::hint::black_box;

#[inline(never)]
fn apeiron_arm(ks: &[Vec<u8>], reps: usize) -> u64 {
    let mut acc = 0u64;
    for _ in 0..reps {
        for k in ks {
            acc = acc.wrapping_add(apeiron::fold::fold_bytes(k));
        }
    }
    acc
}

#[inline(never)]
fn foldhash_arm(ks: &[Vec<u8>], reps: usize, st: foldhash::fast::FixedState) -> u64 {
    let mut acc = 0u64;
    for _ in 0..reps {
        for k in ks {
            let mut h = st.build_hasher();
            Hash::hash_slice(k, &mut h);
            acc = acc.wrapping_add(h.finish());
        }
    }
    acc
}

fn main() {
    let len: usize = std::env::var("LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let ks: Vec<Vec<u8>> = (0..256)
        .map(|_| {
            (0..len)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    b'a' + (s % 26) as u8
                })
                .collect()
        })
        .collect();
    let reps = 40_000;
    let out = if std::env::var("ARM").as_deref() == Ok("foldhash") {
        let st = foldhash::fast::FixedState::with_seed(black_box(0x1234_5678));
        foldhash_arm(&ks, reps, st)
    } else {
        apeiron_arm(&ks, reps)
    };
    println!("{}", black_box(out));
}
