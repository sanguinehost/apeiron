//! Every key distribution anyone has found a defect with, against the real map, with bounds.
//!
//! Two defects shipped because this file did not exist. The original address read its tag off the
//! bottom of a word a multiply never varied, so `highbits` keys shared one tag and every scan
//! compared sixteen addresses -- invisible to probe length, which stayed at 1.0. The fix moved
//! the tag up and dropped the fold, and a slice-scoring experiment blessed it because it scored
//! (window, tag) independence and never played out the placement: with nothing varying below bit
//! 48, every overflow collapsed into group zero and the mean probe reached 136 groups. So this
//! test asserts the three numbers the placement simulation says a healthy table holds at once --
//! probe mean, probe max, and address compares -- on every distribution that broke something,
//! plus the controls. The bounds are wide: the measured worst honest cell is mean 1.030, longest
//! 8, compares 1.051, and the defects sat at 2.0 to 137, so anything that trips this is a real
//! regression and nothing that passes it is worth tuning for.

#![expect(
    clippy::cast_possible_wrap,
    reason = "this is a test, not library code, and each lint below is listed because this file actually \
              triggers it: the reverse of that reinterpretation. Listed individually rather than suppressed \
              wholesale, so a pedantic lint this file does not yet trigger still surfaces"
)]

use apeiron::coord::i21;
use apeiron::map::{Map, Stats, Words};

const MEAN_PROBE_BOUND: f64 = 1.5;
const LONGEST_BOUND: u32 = 16;
const MEAN_COMPARES_BOUND: f64 = 1.5;

fn check(label: &str, s: &Stats) {
    assert!(
        s.mean_probe <= MEAN_PROBE_BOUND,
        "{label}: mean probe {:.3} exceeds {MEAN_PROBE_BOUND} -- the descent is collapsing ({s:?})",
        s.mean_probe
    );
    assert!(
        s.longest_probe <= LONGEST_BOUND,
        "{label}: longest probe {} exceeds {LONGEST_BOUND} ({s:?})",
        s.longest_probe
    );
    assert!(
        s.mean_compares <= MEAN_COMPARES_BOUND,
        "{label}: {:.3} address compares per hit exceeds {MEAN_COMPARES_BOUND} -- the tag has \
         stopped discriminating ({s:?})",
        s.mean_compares
    );
}

/// A named key distribution: a label, and the map from a counter to a key.
type Distribution = (&'static str, fn(u64) -> u64);

/// The integer distributions: everything that has bitten, at three sizes, because the failure
/// mode changes with `gbits` -- what is flat at 5,000 keys (9 window bits) can collapse at
/// 1,500,000 (17), and did.
fn u64_distributions() -> Vec<Distribution> {
    vec![
        ("serial", |i| i),
        // All entropy in the top of the word: hashbrown's `highbits`. Broke the low tag, then
        // broke the foldless address.
        ("highbits", u64::swap_bytes),
        // The same shape out of this crate's own coordinate packing: bits 42..63.
        ("i21hi", |i| (i & 0x1F_FFFF) << 42),
        ("align8", |i| i * 8),
        ("pairlike", |i| ((i % 80) << 32) | (i / 80)),
        ("random", |i| {
            // splitmix-free scramble so the control shares no structure with the address function
            let mut x = i.wrapping_add(0x0123_4567_89AB_CDEF);
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x.rotate_left(23).wrapping_mul(0xD605_1C87_EE0C_59DB) ^ x
        }),
    ]
}

#[test]
fn every_distribution_probes_flat_at_every_size() {
    // Debug builds insert at ~100 ns; the full grid is ~7M inserts and stays under a minute.
    // Under Miri, sample instead of sweep.
    #[cfg(miri)]
    let sizes: &[usize] = &[700];
    #[cfg(not(miri))]
    // Miri cannot walk a million and a half keys. The smallest size still exercises the descent
    // and the flatness bounds; the two large ones are what make the *distribution* argument, and
    // they run natively, which is where that argument is made.
    let sizes: &[usize] =
        if cfg!(miri) { &[2_000] } else { &[5_000, 200_000, 1_500_000] };
    for &n in sizes {
        for (name, f) in u64_distributions() {
            let mut m: Words<u64, u64> = Words::new();
            for i in 0..n as u64 {
                m.insert(f(i), i);
            }
            assert_eq!(m.len(), n, "{name}: generator is not injective; dedup before measuring");
            check(&format!("{name} n={n}"), &m.stats());
        }
    }
}

/// Only 65,536 distinct values, deliberately: a map holds distinct keys, and measuring the raw
/// stream made `i & 0xFFFF` look like a hash defect when it was only a small key set. (BLAKE2
/// scored identically, which is how that mistake was caught.)
#[test]
fn low_entropy_keys_probe_flat_after_dedup() {
    #[cfg(miri)]
    let n = 700u64;
    #[cfg(not(miri))]
    let n = if cfg!(miri) { 2_048u64 } else { 65_536 };
    let mut m: Words<u64, u64> = Words::new();
    for i in 0..n {
        m.insert(i & 0xFFFF, i);
    }
    assert_eq!(m.len() as u64, n);
    check("low16", &m.stats());
}

/// The distribution `i21hi` is a proxy for, run for real: a world whose objects vary along one
/// axis, keyed by this crate's own packed coordinates on the elliptic face.
#[test]
fn one_axis_worlds_probe_flat() {
    #[cfg(miri)]
    let half = 350i64;
    #[cfg(not(miri))]
    let half = if cfg!(miri) { 2_000i64 } else { 750_000 };
    let mut m: Words<(i21, i21, i21), u32> = Words::new();
    let zero = i21::new(0).unwrap();
    for v in -half..half {
        #[allow(clippy::cast_possible_truncation)]
        m.insert((i21::new(v as i32).unwrap(), zero, zero), 1);
    }
    assert_eq!(m.len() as i64, 2 * half);
    check("one-axis (i21,i21,i21)", &m.stats());
}

/// String keys: the parabolic face, addresses out of `fold_bytes` and then the same bijection.
#[test]
fn string_keys_probe_flat() {
    #[cfg(miri)]
    let n = 500;
    #[cfg(not(miri))]
    let n = if cfg!(miri) { 2_000 } else { 50_000 };
    let mut m: Map<String, u32> = Map::new();
    for i in 0..n {
        m.insert(format!("key{i}"), i);
    }
    assert_eq!(m.len(), n as usize);
    check("strings", &m.stats());
}
