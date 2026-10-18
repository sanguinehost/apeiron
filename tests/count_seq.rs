//! The sequential count helpers and their oracle.
//! The sequential steps of `FreqMap::count_par`: oracle equality, hand-written helper outputs,
//! single-threaded composition, and the checked admission refusals.

#![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]

use apeiron::map::freq::FreqMap;
use std::collections::HashMap;

type F = FreqMap<u64, i64>;
const B: usize = F::COUNT_BUCKETS;
const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

/// Independent statement of the home bucket: the 8 bits of `c * PHI` just below the probe's home
/// window (bits `[40, 48)`), disjoint from `FreqLine::home_group`'s top window and the tag window.
fn home(c: u64) -> usize {
    ((c.wrapping_mul(PHI) >> 40) & 0xFF) as usize
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn keys_of(n: usize, modulus: u64, seed: u64) -> Vec<u64> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.next() % modulus).collect()
}

fn oracle(keys: &[u64]) -> HashMap<u64, i64> {
    let mut m = HashMap::new();
    for &k in keys {
        *m.entry(k).or_insert(0i64) += 1;
    }
    m
}

fn assert_matches(maps: &[F], keys: &[u64], what: &str) {
    let o = oracle(keys);
    let live: usize = maps.iter().map(F::len).sum();
    assert_eq!(live, o.len(), "{what}: distinct keys");
    let total: i64 = maps.iter().flat_map(F::values).sum();
    assert_eq!(total, keys.len() as i64, "{what}: counts sum to n");
    // Totals above cover every key; the per-key probe (256 maps each) runs on every `stride`-th key.
    let stride = o.len() / 20_000 + 1;
    for (k, &c) in o.iter().step_by(stride) {
        let got: i64 = maps.iter().filter_map(|m| m.get(k).copied()).sum();
        assert_eq!(got, c, "{what}: key {k}");
        let holders = maps.iter().filter(|m| m.get(k).is_some()).count();
        assert_eq!(holders, 1, "{what}: key {k} lives in exactly one map");
    }
}

#[test]
fn count_par_equals_hashmap_oracle() {
    for (n, modulus) in [
        (0usize, 1u64),
        (1, 1),
        (2, 1),
        (1000, 100),
        (40_000, 1000),
        (40_000, u64::MAX),
        (1_000_000, 100_000),
        (1_000_000, u64::MAX),
    ] {
        let keys = keys_of(n, modulus, 0x00C0_FFEE ^ n as u64);
        assert_matches(&F::count_par(&keys), &keys, &format!("n={n} mod={modulus}"));
        assert_matches(
            &F::try_count_par(&keys).expect("admitted"),
            &keys,
            &format!("try n={n}"),
        );
    }
}

#[test]
fn home_matches_the_below_home_window() {
    for c in [0u64, 1, 2, 255, 1 << 40, u64::MAX, 0xDEAD_BEEF_CAFE_F00D] {
        assert_eq!(F::count_home(c), home(c));
    }
    assert_eq!(F::count_home(0), 0);
}

#[test]
fn histogram_hand_input() {
    // 0 homes to bucket 0; 1 and 2 to their independently computed buckets.
    let keys = [0u64, 1, 0, 2, 0, 1];
    let mut homes = [0xFFu8; 6];
    let mut hist = vec![0u32; B];
    assert!(F::count_histogram(&keys, &mut homes, &mut hist));
    let want_homes: Vec<u8> = keys.iter().map(|&k| home(k) as u8).collect();
    assert_eq!(homes.to_vec(), want_homes);
    let mut want = vec![0u32; B];
    for &k in &keys {
        want[home(k)] += 1;
    }
    assert_eq!(hist, want);
    assert_eq!(hist[0], 3);
    assert_eq!(hist.iter().sum::<u32>(), 6);
    // wrong lengths refuse
    assert!(!F::count_histogram(&keys, &mut homes[..5], &mut hist));
    assert!(!F::count_histogram(&keys, &mut homes, &mut hist[..B - 1]));
}

#[test]
fn offsets_hand_input() {
    // w = 2 workers; worker 0 holds {bucket0: 2, bucket1: 1}, worker 1 {bucket0: 1, bucket2: 3}.
    let mut counts = vec![0u32; 2 * B];
    counts[0] = 2;
    counts[1] = 1;
    counts[B] = 1;
    counts[B + 2] = 3;
    let mut bstart = vec![0usize; B + 1];
    let mut woff = vec![0usize; 2 * B];
    assert!(F::count_offsets(&counts, 2, 7, &mut bstart, &mut woff));
    assert_eq!(&bstart[..5], &[0, 3, 4, 7, 7]);
    assert!(bstart[3..].iter().all(|&s| s == 7));
    assert_eq!((woff[0], woff[B]), (0, 2)); // bucket 0: worker 0 then worker 1
    assert_eq!((woff[1], woff[B + 1]), (3, 4)); // bucket 1
    assert_eq!((woff[2], woff[B + 2]), (4, 4)); // bucket 2: worker 0 empty, worker 1 at 4
}

#[test]
fn slot_hand_input() {
    let mut off = vec![0usize; B];
    off[0] = 5;
    off[7] = 9;
    assert_eq!(F::count_slot(&mut off, 0), 5);
    assert_eq!(F::count_slot(&mut off, 0), 6);
    assert_eq!(F::count_slot(&mut off, 7), 9);
    assert_eq!((off[0], off[7], off[1]), (7, 10, 0));
}

#[test]
fn aggregate_hand_input() {
    let routed = [7u64, 7, 9, 7];
    let m = F::count_aggregate(&routed, 0, 3);
    assert_eq!(
        (m.get(&7).copied(), m.get(&9).copied(), m.len()),
        (Some(2), Some(1), 2)
    );
    let m = F::count_aggregate(&routed, 3, 4);
    assert_eq!((m.get(&7).copied(), m.len()), (Some(1), 1));
    let m = F::count_aggregate(&routed, 2, 99); // clipped to the slice
    assert_eq!((m.get(&7).copied(), m.get(&9).copied()), (Some(1), Some(1)));
    assert!(F::count_aggregate(&routed, 3, 1).is_empty());
}

/// The helpers composed on one thread (`w` shards run in sequence) equal `count_par`.
fn compose(keys: &[u64], w: usize) -> Vec<F> {
    let n = keys.len();
    let chunk = n / w + usize::from(!n.is_multiple_of(w));
    let mut homes = vec![0u8; n];
    let mut counts = vec![0u32; w * B];
    for t in 0..w {
        let (lo, hi) = ((t * chunk).min(n), ((t + 1) * chunk).min(n));
        assert!(F::count_histogram(
            &keys[lo..hi],
            &mut homes[lo..hi],
            &mut counts[t * B..(t + 1) * B]
        ));
    }
    let mut bstart = vec![0usize; B + 1];
    let mut woff = vec![0usize; w * B];
    assert!(F::count_offsets(&counts, w, n, &mut bstart, &mut woff));
    let mut routed = vec![0u64; n];
    for t in 0..w {
        let (lo, hi) = ((t * chunk).min(n), ((t + 1) * chunk).min(n));
        let mut off = woff[t * B..(t + 1) * B].to_vec();
        for i in lo..hi {
            routed[F::count_slot(&mut off, homes[i])] = keys[i];
        }
    }
    (0..B)
        .map(|j| F::count_aggregate(&routed, bstart[j], bstart[j + 1]))
        .collect()
}

#[test]
fn composition_single_threaded_equals_count_par() {
    for (n, w) in [
        (0usize, 1usize),
        (1, 1),
        (5, 3),
        (50_000, 1),
        (50_000, 7),
        (200_000, 16),
    ] {
        let keys = keys_of(n, 5000, 77 + n as u64);
        let composed = compose(&keys, w);
        assert_matches(&composed, &keys, &format!("compose n={n} w={w}"));
        let par = F::count_par(&keys);
        // Both are exact counts: compare per-key totals.
        let o = oracle(&keys);
        for (k, &c) in &o {
            let a: i64 = composed.iter().filter_map(|m| m.get(k).copied()).sum();
            let b: i64 = par.iter().filter_map(|m| m.get(k).copied()).sum();
            assert_eq!((a, b), (c, c), "key {k}");
        }
        // Same bucket partition as the production path: home bucket j holds exactly its keys.
        for (j, cj) in composed.iter().enumerate() {
            for &k in &keys {
                let in_j = cj.get(&k).is_some();
                assert_eq!(in_j, home(k) == j, "key {k} bucket {j}");
            }
            if n > 20_000 {
                break; // full bucket check on the small sizes only
            }
        }
    }
}

#[test]
fn admission_refuses_at_the_boundary() {
    // i64 count: n <= count_max admitted, n = count_max + 1 refused (parameterised boundary).
    assert!(F::count_admit(10, 10));
    assert!(!F::count_admit(11, 10));
    assert!(F::count_admit(0, 0));
    assert!(!F::count_admit(usize::MAX, i64::MAX as u64));
    assert!(F::count_admit(1 << 40, i64::MAX as u64));

    // u32 histogram: an entry at u32::MAX refuses the next increment (no wrap).
    let keys = [5u64, 5];
    let mut homes = [0u8; 2];
    let mut hist = vec![0u32; B];
    hist[home(5)] = u32::MAX;
    assert!(!F::count_histogram(&keys, &mut homes, &mut hist));
    assert_eq!(hist[home(5)], u32::MAX, "refused, not wrapped");
    hist[home(5)] = u32::MAX - 2;
    assert!(F::count_histogram(&keys, &mut homes, &mut hist));
    assert_eq!(hist[home(5)], u32::MAX);
    assert!(!F::count_histogram(&keys[..1], &mut homes[..1], &mut hist));

    // offsets: a histogram whose total is not n refuses (the routed buffer would be underwritten).
    let mut counts = vec![0u32; B];
    counts[3] = 4;
    let mut bstart = vec![0usize; B + 1];
    let mut woff = vec![0usize; B];
    assert!(F::count_offsets(&counts, 1, 4, &mut bstart, &mut woff));
    assert!(!F::count_offsets(&counts, 1, 5, &mut bstart, &mut woff));
    assert!(!F::count_offsets(&counts, 1, 3, &mut bstart, &mut woff));
    assert!(!F::count_offsets(&counts, 2, 4, &mut bstart, &mut woff)); // wrong length
    assert!(!F::count_offsets(
        &counts,
        usize::MAX,
        4,
        &mut bstart,
        &mut woff
    )); // w*b overflows
}
