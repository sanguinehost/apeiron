//! A counting table, partitioned so each shard's working set fits in cache.
//!
//! Binning a million keys into one table is not compute-bound: hashing costs 0.7 ms and the
//! random access costs 21. A table of `2^21` sixteen-byte slots is 32 MB, so nearly every insert
//! is a miss, and adding threads to a miss-bound loop buys much less than the thread count -- 32
//! threads move it 2.35x.
//!
//! So the build is partitioned first. Keys are sorted into shards by the top bits of their
//! avalanched address, each shard's sub-table is small enough to sit in L2, and the scatter that
//! gets them there is sequential. What the algebra contributes is the licence:
//! `Table.build_eq_of_key_streams_eq` says a build depends on the stream only through each key's
//! values in their stream order, for **every** merge -- so sharding by key is legal, and the
//! shards need no merge step because a key is in exactly one of them.
//!
//! The same theorem licenses the *second* cut. A worker takes a contiguous chunk of the stream
//! and partitions only that; a shard's stream is then the workers' runs concatenated in worker
//! order, which is the original order of that key's occurrences. Not a reordering that would need
//! `build_gauge_invariance` and a commutative merge -- the per-key sequences are untouched, so
//! `build_eq_of_key_streams_eq` covers it as it stands.
//!
//! # What is on the critical path, and what was taken off it
//!
//! Measured with `examples/phases.rs`, `n = 1_000_000`, 16 workers. Three costs ran on one thread
//! while fifteen waited:
//!
//! | was | ms | now |
//! |---|---|---|
//! | partition, twice (build and probe) | 1.53 each | cut into worker chunks |
//! | `vec![(u64::MAX, 0); ..]` over 33.6 MB of regions | 1.81 | zero-initialised, inside the worker |
//! | `finish` recomputed four times per key | 0.69 each | twice, both inside the partition |
//!
//! `apeiron_par16` on `bin_and_probe/1000000` went from 9.47 ms to 3.46, against 4.09 for fxhash
//! and 4.69 for foldhash sharded and threaded the same way.
//!
//! The last row is the one that has to be argued rather than measured. A slot holds `finish(a)`, not
//! `a`. `finish` is splitmix64's finalizer, a bijection on `Z/2^64` -- `a_xor_shift_step_is_injective`
//! and `multiplying_by_an_odd_constant_is_injective` in `proofs.rs` check its steps -- so
//! `finish(x) == finish(y)` exactly when `x == y`, and comparing finished addresses *is*
//! comparing keys. `tests/the_counts_are_exact.rs` asserts that against a reference count rather
//! than leaving it as an argument.

//!
//! # Measured and not taken
//!
//! The workload is memory-bound, so each of these trades traffic for arithmetic in the direction
//! that looks obviously right and is not. Same harness, criterion mean, each alternated against
//! the current code so the two share the machine's mood.
//!
//! | change | ms | against |
//! |---|---|---|
//! | keep `finish`'s output in a scratch buffer instead of recomputing it in the scatter | 3.95 | 3.48 |
//! | `_mm_prefetch` the next group's tag line, eight keys ahead, in both loops | 4.00 | 3.78 |
//! | drop the probe's partition, chunk the probe stream contiguously instead | 5.14 | 3.44 |
//! | size regions at 3x load, and at 4x | 4.64, 4.27 | 3.44 |
//!
//! The scratch buffer is 8 MB written and 8 MB read back, to save two multiplies. The prefetch
//! pays a hint for latency that a 128 KB L2-resident region and an out-of-order engine were
//! already hiding. Dropping the probe's grouping saves a whole thread scope and a whole partition
//! pass and costs 49%, which is the measurement that says the grouping is the point rather than
//! the bookkeeping around it. And more slots is more misses: occupancy is 47.4% already.
//!
//! Sizing at 1x load is faster still, 2.63, and is not available. `next_power_of_two` can land
//! exactly on the load; a full region has no empty slot; `locate` looks for one. So an absent
//! key's probe would not return -- a hang, at a load an attacker chooses.
//!
//! `SHARD_BITS` at 6 and 7 rather than 8 measured 3.79 and 3.43 against 3.44: no signal.
//!
use crate::coord::finish;
use std::cmp::Reverse;

/// How many shards to cut the key space into. Chosen so a shard's sub-table stays inside L2.
const SHARD_BITS: u32 = 8;
const SHARDS: usize = 1 << SHARD_BITS;

/// A key's shard: the top bits of its avalanched address, which is also where the group descent
/// starts, so a shard owns a contiguous run of the descent's first window and the two never
/// disagree.
#[inline]
#[must_use]
pub fn shard_of(address: u64) -> usize {
    home(finish(address))
}

/// The same thing, for an address that has already been through `finish`.
#[inline]
const fn home(p: u64) -> usize {
    (p >> (64 - SHARD_BITS)) as usize
}

/// Counts per distinct address, built by partitioning and probed the same way.
pub struct CountTable {
    /// One region per shard, each sized to the keys that actually landed in it.
    ///
    /// Not one array cut into equal parts. A fixed share assumes the shard function spreads, and
    /// an attacker who aims keys at one shard then overflows a region that cannot grow: the probe
    /// looks for an empty slot, there is none, and the build does not terminate. Sizing after
    /// partitioning removes the assumption -- a shard holding every key gets a region for every
    /// key.
    shards: Vec<Region>,
}

impl CountTable {
    /// Build from a key stream on this thread. `bits` is a hint for the whole table; each shard
    /// is sized to its own load, so the hint never bounds a shard.
    #[must_use]
    pub fn build(addresses: &[u64], bits: u32) -> Self {
        let _ = bits;
        Self::assemble(&Parts::of(addresses, 1), &[(0..SHARDS).collect()])
    }

    /// Build across `threads` workers. Shards hold disjoint keys, so each worker owns a run of
    /// buckets outright and there is no merge and no synchronisation on the hot path.
    #[must_use]
    pub fn build_parallel(addresses: &[u64], bits: u32, threads: usize) -> Self {
        let _ = bits;
        let workers = threads.max(1);
        let parts = Parts::of(addresses, workers);
        Self::assemble(&parts, &schedule(&parts.loads(), workers))
    }

    /// Build the shards each worker was given, then place them. A worker creates its own regions,
    /// so the 33.6 MB of zeroing that a million keys' worth of regions costs is spread over the
    /// workers instead of run on the main thread before any of them start.
    fn assemble(parts: &Parts, bins: &[Vec<usize>]) -> Self {
        let loads = parts.loads();
        let build_bin = |mine: &Vec<usize>| -> Vec<(usize, Region)> {
            mine.iter()
                .map(|&s| {
                    let mut region = region_for(loads[s]);
                    for run in parts.runs(s) {
                        for &p in run {
                            insert_into(&mut region, p);
                        }
                    }
                    (s, region)
                })
                .collect()
        };

        let built: Vec<Vec<(usize, Region)>> = if bins.len() <= 1 {
            bins.iter().map(build_bin).collect()
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = bins
                    .iter()
                    .map(|mine| scope.spawn(|| build_bin(mine)))
                    .collect();
                handles.into_iter().map(join_or_panic).collect()
            })
        };

        let mut shards: Vec<Region> = (0..SHARDS).map(|_| Region::new(0)).collect();
        for group in built {
            for (s, region) in group {
                shards[s] = region;
            }
        }
        Self { shards }
    }

    /// The count for an address, or zero.
    #[must_use]
    pub fn get(&self, a: u64) -> i64 {
        self.get_finished(finish(a))
    }

    #[inline]
    fn get_finished(&self, p: u64) -> i64 {
        let region = &self.shards[home(p)];
        if region.is_empty() {
            return 0;
        }
        let (i, found) = locate(region, p);
        if found {
            region.slots[i].1
        } else {
            0
        }
    }

    /// Total slots across every shard.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.shards.iter().map(Region::len).sum()
    }

    /// Sum the counts for `addresses`, across `threads` workers.
    ///
    /// Partitioned the same way the build was, so a worker touches one contiguous region and the
    /// probe is as cache-local as the insert. Probing without grouping is what makes a sharded
    /// table look slow: the key is in exactly one shard and visiting the others measures nothing.
    #[must_use]
    pub fn sum_parallel(&self, addresses: &[u64], threads: usize) -> i64 {
        let workers = threads.max(1);
        let parts = Parts::of(addresses, workers);
        let bins = schedule(&parts.loads(), workers);
        let sum_bin = |mine: &Vec<usize>| -> i64 {
            let mut acc = 0i64;
            for &s in mine {
                for run in parts.runs(s) {
                    for &p in run {
                        acc += self.get_finished(p);
                    }
                }
            }
            acc
        };
        if workers == 1 {
            return bins.iter().map(sum_bin).sum();
        }
        std::thread::scope(|scope| {
            let handles: Vec<_> = bins
                .iter()
                .map(|mine| scope.spawn(|| sum_bin(mine)))
                .collect();
            handles.into_iter().map(join_or_panic).sum()
        })
    }
}

/// A worker that panicked has not counted its keys, and a table that quietly returns fewer than
/// it was given is worse than one that does not return. `filter_map(.., ok)` and `unwrap_or(0)`
/// both turned a panicking worker into a silent undercount.
fn join_or_panic<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    match handle.join() {
        Ok(v) => v,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

// ── partitioning ────────────────────────────────────────────────────────────────────────────

/// A stream cut into shards, as **avalanched** addresses, one buffer per worker.
///
/// Not one `Vec` per shard. That was 256 allocations a call and 256 live write cursors in 256
/// separate allocations; this is one buffer per worker with its shard runs laid end to end, so
/// the scatter writes into a single object and the whole partition is `workers` allocations.
struct Parts {
    /// `bufs[w][ends[w][s] .. ends[w][s + 1]]` is worker `w`'s run for shard `s`.
    bufs: Vec<Vec<u64>>,
    ends: Vec<[usize; SHARDS + 1]>,
}

impl Parts {
    /// Below this many keys a worker is not worth raising: a scope of sixteen costs 0.11 ms to
    /// start and stop, and partitioning a key costs about 1.6 ns.
    const PER_WORKER: usize = 4096;

    fn of(addresses: &[u64], workers: usize) -> Self {
        let workers = workers.clamp(1, (addresses.len() / Self::PER_WORKER).max(1));
        if addresses.is_empty() {
            return Self {
                bufs: Vec::new(),
                ends: Vec::new(),
            };
        }
        let chunk = addresses.len().div_ceil(workers).max(1);
        let done: Vec<Chunk> = if workers == 1 || addresses.len() <= chunk {
            addresses.chunks(chunk).map(partition_chunk).collect()
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = addresses
                    .chunks(chunk)
                    .map(|c| scope.spawn(move || partition_chunk(c)))
                    .collect();
                handles.into_iter().map(join_or_panic).collect()
            })
        };
        let (bufs, ends) = done.into_iter().unzip();
        Self { bufs, ends }
    }

    /// Shard `s`'s stream: each worker's run, in worker order, which is stream order.
    #[inline]
    fn runs(&self, s: usize) -> impl Iterator<Item = &[u64]> {
        self.bufs
            .iter()
            .zip(self.ends.iter())
            .map(move |(b, e)| &b[e[s]..e[s + 1]])
    }

    fn loads(&self) -> Vec<usize> {
        let mut out = vec![0usize; SHARDS];
        for e in &self.ends {
            for (s, l) in out.iter_mut().enumerate() {
                *l += e[s + 1] - e[s];
            }
        }
        out
    }
}

type Chunk = (Vec<u64>, [usize; SHARDS + 1]);

/// Count, prefix-sum, scatter -- over one worker's slice, into one buffer it owns outright.
///
/// The *avalanched* address is what gets written, so nothing downstream recomputes it: not the
/// group descent, not the tag, not the slot comparison. It used to run four times on the way to a
/// slot and three more on the way back.
///
/// It does run twice here, once in each pass, and that is deliberate: keeping the first pass's
/// results in a scratch buffer to spare the second pass a recomputation was measured 12% *slower*
/// -- 3.95 ms against 3.48 -- because the buffer is 8 MB written and 8 MB read back and two
/// multiplies are not. The workload is memory-bound; arithmetic is the cheap side of that trade.
fn partition_chunk(keys: &[u64]) -> Chunk {
    let mut counts = [0usize; SHARDS];
    for &a in keys {
        counts[home(finish(a))] += 1;
    }
    let mut ends = [0usize; SHARDS + 1];
    let mut cursor = [0usize; SHARDS];
    let mut acc = 0usize;
    for s in 0..SHARDS {
        ends[s] = acc;
        cursor[s] = acc;
        acc += counts[s];
    }
    ends[SHARDS] = acc;

    let mut buf = vec![0u64; keys.len()];
    for &a in keys {
        let p = finish(a);
        let s = home(p);
        buf[cursor[s]] = p;
        cursor[s] += 1;
    }
    (buf, ends)
}

/// Which worker builds which shards: longest-processing-time-first onto the emptiest worker.
///
/// A fixed run of `SHARDS / workers` assumes the shard loads are equal, and a build finishes when
/// its *slowest* worker does, so a skewed stream -- which is what an attacker sends -- pays its
/// skew in wall time. The loads are known before any region is built, so the assignment can just
/// use them. Deterministic: the sort key is `(load descending, shard)`, a total order, and ties
/// among workers go to the lowest index.
fn schedule(loads: &[usize], workers: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..SHARDS).collect();
    order.sort_unstable_by_key(|&s| (Reverse(loads[s]), s));
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); workers];
    let mut filled = vec![0usize; workers];
    for s in order {
        let w = filled
            .iter()
            .enumerate()
            .min_by_key(|&(i, &l)| (l, i))
            .map_or(0, |(i, _)| i);
        bins[w].push(s);
        filled[w] += loads[s];
    }
    bins
}

// ── the probe ───────────────────────────────────────────────────────────────────────────────

/// Slots per group. One group is scanned before the probe moves on, and it is sixteen because
/// that is one 128-bit tag compare.
const GROUP: usize = 16;

/// The probe sequence: successive windows of the avalanched address, not a linear walk.
///
/// This is what makes the table structurally hard to attack, and it is the difference between
/// bounded and unbounded collision chains.
///
/// A linear probe uses only the `bits` of the bucket index. Two keys that agree on those bits
/// chain for as long as the attacker likes, and with an unseeded hash those bits are cheap to
/// aim: inverting `finish` -- splitmix64's finalizer, public constants -- puts twenty thousand
/// keys in one bucket and costs ten thousand probe steps each, against one.
///
/// A window descent uses the whole word. To extend a chain by one step an attacker must match
/// another `gbits` of the address, so a chain of `j` steps costs `j * gbits` bits of control, and
/// an unbounded chain requires matching all 64 -- at which point the two addresses are equal and
/// the table is right to treat them as one key. The cost of an attack is therefore bounded by the
/// address width rather than by the index width.
///
/// Seedless deliberately. A random per-table seed would also defeat aiming, and would break
/// deterministic replay: same inputs must give the same state, which a per-run seed denies.
#[inline]
fn group_of(p: u64, j: u32, gbits: u32, gmask: usize) -> usize {
    if gbits == 0 {
        return 0;
    }
    let used = gbits * (j + 1);
    if used <= 64 {
        (p >> (64 - used)) as usize & gmask
    } else {
        // The word is spent; sweep, offset so a step never re-scans the group just examined.
        let full = 64 / gbits;
        ((p >> (64 - gbits * full)) as usize + (j - full) as usize + 1) & gmask
    }
}

/// A slot's seven-bit tag, with the top bit marking it occupied. Zero means empty.
///
/// Taken from the *low* bits of the avalanched address, which the descent never looks at -- it
/// walks windows down from the top -- so a tag hit and a group hit are independent evidence.
#[inline]
const fn tag_of(p: u64) -> u8 {
    0x80 | (p & 0x7F) as u8
}

/// Sixteen lanes at a time, two ways.
///
/// A group is sixteen slots, so a group scan is one 128-bit compare on x86_64 and a pair of SWAR
/// words everywhere else. Both are compiled on every target and `tests/both_tag_paths_agree.rs`
/// asserts they are the same function, because the one this build does not select is exactly the
/// code that rots.
///
/// Public so the test can reach both. Nothing else should: this is the inside of `locate`.
pub mod scan {
    /// Eight marker bits, gathered into the low eight bits of the result.
    ///
    /// The SWAR compare leaves `0x80` at bit `7 + 8i` of each matching lane `i`. Multiplying by
    /// `GATHER` shifts lane `i`'s marker by `49 - 7i`, landing it at bit `56 + i`. All 64 cross
    /// terms land on distinct positions -- `8(i - i') = 7(j - j')` forces `i = i'` when both
    /// differences are at most seven -- so the multiply carries nowhere and is an exact
    /// permutation of the marker bits rather than an approximation of one.
    #[inline]
    const fn gather(marked: u64) -> u8 {
        const GATHER: u64 = 0x0002_0408_1020_4081;
        #[allow(clippy::cast_possible_truncation)]
        {
            (marked.wrapping_mul(GATHER) >> 56) as u8
        }
    }

    /// `0x80` at each lane of `word` equal to `byte`, and nowhere else.
    ///
    /// Exactly, not approximately. The textbook form `(x - LO) & !x & HI` is a *superset*: a lane
    /// that matches borrows out of the subtraction, and a neighbour whose difference from `byte`
    /// is exactly one is then flagged as well. `[.., 0xB1, 0xB0, ..]` against `0xB1` reports two
    /// lanes; SSE2 reports one. Setting every lane's top bit before subtracting stops the borrow
    /// at the lane boundary, and or-ing `x` back in restores the lanes whose own top bit the
    /// forcing overwrote, so no lane's answer depends on any other lane.
    ///
    /// The looser form would in fact be sound here -- `locate` compares the stored key before it
    /// believes a tag, and an empty lane can never be a false positive because its tag is zero
    /// while every real tag has bit seven set, so the `!x & HI` term rejects it. But then the two
    /// scans would be *different functions that happen to agree on the answer*, and a test that
    /// says so is a test of the argument rather than of the code.
    #[inline]
    const fn eq_swar(word: u64, byte: u8) -> u64 {
        const LO: u64 = 0x0101_0101_0101_0101;
        const HI: u64 = 0x8080_8080_8080_8080;
        let x = word ^ LO.wrapping_mul(byte as u64);
        // Every lane of `x | HI` is at least 0x80, so subtracting one never borrows out of it.
        !(x | (x | HI).wrapping_sub(LO)) & HI
    }

    #[inline]
    fn halves(line: &[u8; 16]) -> (u64, u64) {
        let mut lo = [0u8; 8];
        let mut hi = [0u8; 8];
        lo.copy_from_slice(&line[..8]);
        hi.copy_from_slice(&line[8..]);
        (u64::from_le_bytes(lo), u64::from_le_bytes(hi))
    }

    /// Lanes of `line` equal to `tag`, one bit per lane.
    #[inline]
    #[must_use]
    pub fn match_tag_swar(line: &[u8; 16], tag: u8) -> u16 {
        let (lo, hi) = halves(line);
        u16::from(gather(eq_swar(lo, tag))) | (u16::from(gather(eq_swar(hi, tag))) << 8)
    }

    /// Lanes of `line` with the top bit set, one bit per lane. `tag_of` always sets that bit and
    /// an empty slot's tag is zero, so this is exactly "not empty" -- and it is one instruction
    /// rather than a compare against zero.
    #[inline]
    #[must_use]
    pub fn occupied_swar(line: &[u8; 16]) -> u16 {
        const HI: u64 = 0x8080_8080_8080_8080;
        let (lo, hi) = halves(line);
        u16::from(gather(lo & HI)) | (u16::from(gather(hi & HI)) << 8)
    }

    /// SSE2, which is in the x86_64 baseline: no target feature, no runtime check, and no path
    /// that only some machines take.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn match_tag_simd(line: &[u8; 16], tag: u8) -> u16 {
        use std::arch::x86_64::{
            __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
        };
        // SAFETY: SSE2 is unconditionally available on x86_64, and `line` is exactly the sixteen
        // bytes the load reads. `loadu` does not require alignment.
        unsafe {
            let v = _mm_loadu_si128(line.as_ptr().cast::<__m128i>());
            _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_set1_epi8(tag as i8))) as u16
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn occupied_simd(line: &[u8; 16]) -> u16 {
        use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_movemask_epi8};
        // SAFETY: as `match_tag_simd`.
        unsafe { _mm_movemask_epi8(_mm_loadu_si128(line.as_ptr().cast::<__m128i>())) as u16 }
    }

    /// Whichever of the two this target compiled to.
    #[inline]
    #[must_use]
    pub fn match_tag(line: &[u8; 16], tag: u8) -> u16 {
        #[cfg(target_arch = "x86_64")]
        {
            match_tag_simd(line, tag)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            match_tag_swar(line, tag)
        }
    }

    #[inline]
    #[must_use]
    pub fn occupied(line: &[u8; 16]) -> u16 {
        #[cfg(target_arch = "x86_64")]
        {
            occupied_simd(line)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            occupied_swar(line)
        }
    }
}

/// Find `p`'s slot, or the first empty slot in probe order. `p` is already avalanched.
///
/// Two reads per group in the common case: the tag line, then one address for each lane whose tag
/// matched. Scanning all sixteen addresses instead costs 256 bytes a group against 16, and
/// measured 2.3x on the whole workload -- the tag line is not an optimisation on top of the
/// descent, it is what makes the descent affordable.
///
/// Stops at the first group holding an empty slot: placement is first-fit and occupancy only
/// grows, so an empty slot means the key would have been placed at or before here, and is absent.
#[inline]
fn locate(region: &Region, p: u64) -> (usize, bool) {
    let groups = region.slots.len() / GROUP;
    let gbits = groups.trailing_zeros();
    let gmask = groups - 1;
    let tag = tag_of(p);
    let mut j = 0u32;
    loop {
        let base = group_of(p, j, gbits, gmask) * GROUP;
        let line: &[u8; 16] = region.tags[base..base + GROUP]
            .try_into()
            .unwrap_or(&EMPTY_LINE);
        let mut hits = scan::match_tag(line, tag);
        while hits != 0 {
            let lane = base + hits.trailing_zeros() as usize;
            if region.slots[lane].0 == p {
                return (lane, true);
            }
            hits &= hits - 1;
        }
        // Only now look for an empty lane: the hit path above never reads this.
        let empties = !scan::occupied(line);
        if empties != 0 {
            return (base + empties.trailing_zeros() as usize, false);
        }
        j += 1;
    }
}

/// Unreachable: `tags.len()` is `slots.len()` and `base` is a group start, so the slice is always
/// sixteen bytes. Present so `locate` has no panicking branch on the hot path.
const EMPTY_LINE: [u8; 16] = [0; 16];

/// A shard's storage: a tag line beside the slots, so a group scan touches one cache line.
#[derive(Clone)]
pub struct Region {
    /// One tag byte per slot, so a group is sixteen contiguous bytes and one vector load.
    tags: Vec<u8>,
    /// The **avalanched** address and its count. An empty slot is `(0, 0)` and is told apart by
    /// its tag, never by a sentinel value -- so zero is an ordinary key, and the whole region can
    /// be allocated already zeroed instead of written over. That sentinel cost 1.81 ms of
    /// `memset` on the main thread at a million keys.
    slots: Vec<(u64, i64)>,
}

impl Region {
    fn new(slots: usize) -> Self {
        Self {
            tags: vec![0u8; slots],
            slots: vec![(0, 0); slots],
        }
    }

    fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    fn len(&self) -> usize {
        self.slots.len()
    }
}

/// A region for `load` keys: at least two groups, and at most half full, rounded to a power of
/// two so the group index is a mask.
fn region_for(load: usize) -> Region {
    if load == 0 {
        return Region::new(0);
    }
    Region::new((load * 2).max(GROUP * 2).next_power_of_two())
}

#[inline]
fn insert_into(region: &mut Region, p: u64) {
    let (i, found) = locate(region, p);
    if found {
        region.slots[i].1 += 1;
    } else {
        region.slots[i] = (p, 1);
        region.tags[i] = tag_of(p);
    }
}
