//! A counting table, partitioned so each shard's working set fits in cache.
//!
//! Binning a million keys into one table is not compute-bound: hashing costs 1.5 ms and the
//! random access costs 21. A table of `2^21` sixteen-byte slots is 32 MB, so nearly every insert
//! is a miss, and adding threads to a miss-bound loop buys much less than the thread count -- 32
//! threads move it 2.35x.
//!
//! So the build is partitioned first. Keys are sorted into shards by the top bits of their bucket
//! index, each shard's sub-table is small enough to sit in L2, and the scatter that gets them
//! there is sequential. What the algebra contributes is the licence: `Table.build_eq_of_key_streams_eq`
//! says a build depends on the stream only through each key's values in their stream order, for
//! **every** merge -- so sharding by key is legal, and the shards need no merge step because a key
//! is in exactly one of them.

use crate::coord::finish;

/// How many shards to cut the key space into. Chosen so a shard's sub-table stays inside L2.
const SHARD_BITS: u32 = 8;
const SHARDS: usize = 1 << SHARD_BITS;

/// A key's shard: the top bits of its avalanched address, which is also where `bucket` reads from,
/// so a shard owns a contiguous run of buckets and the two never disagree.
#[inline]
#[must_use]
pub fn shard_of(address: u64) -> usize {
    (finish(address) >> (64 - SHARD_BITS)) as usize
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
    /// Build from a key stream. `bits` is a hint for the whole table; each shard is sized to its
    /// own load, so the hint never bounds a shard.
    #[must_use]
    pub fn build(addresses: &[u64], bits: u32) -> Self {
        let parts = partition(addresses);
        let shards: Vec<Region> = parts
            .iter()
            .map(|keys| {
                let mut region = region_for(keys.len(), bits);
                for &a in keys {
                    insert_into(&mut region, a);
                }
                region
            })
            .collect();
        Self { shards }
    }

    /// Build across `threads` workers. Shards hold disjoint keys, so each worker owns a run of
    /// buckets outright and there is no merge and no synchronisation on the hot path.
    #[must_use]
    pub fn build_parallel(addresses: &[u64], bits: u32, threads: usize) -> Self {
        let parts = partition(addresses);
        let mut shards: Vec<Region> = parts.iter().map(|k| region_for(k.len(), bits)).collect();

        // A shard owns its own region outright, so the split is by ownership rather than by lock.
        std::thread::scope(|scope| {
            let chunk = SHARDS.div_ceil(threads.max(1));
            let mut rest: Vec<(usize, &mut Region)> = shards.iter_mut().enumerate().collect();
            while !rest.is_empty() {
                let take = chunk.min(rest.len());
                let mine: Vec<(usize, &mut Region)> = rest.drain(..take).collect();
                let parts = &parts;
                scope.spawn(move || {
                    for (s, region) in mine {
                        for &a in &parts[s] {
                            insert_into(region, a);
                        }
                    }
                });
            }
        });

        Self { shards }
    }

    /// The count for an address, or zero.
    #[must_use]
    pub fn get(&self, a: u64) -> i64 {
        let region = &self.shards[shard_of(a)];
        if region.is_empty() {
            return 0;
        }
        let (i, found) = locate(region, a);
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
        let shards = partition(addresses);
        let chunk = SHARDS.div_ceil(threads.max(1));
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for group in (0..SHARDS).step_by(chunk) {
                let hi = (group + chunk).min(SHARDS);
                let shards = &shards;
                handles.push(scope.spawn(move || {
                    let mut acc = 0i64;
                    for s in group..hi {
                        for &a in &shards[s] {
                            acc += self.get(a);
                        }
                    }
                    acc
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
        })
    }
}

/// Slots per group. One group is scanned before the probe moves on.
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
#[inline]
const fn tag_of(p: u64) -> u8 {
    0x80 | (p & 0x7F) as u8
}

/// Byte positions in `word` equal to `byte`, as a mask with `0x80` set at each match.
///
/// SWAR rather than an intrinsic: eight lanes per instruction with no `unsafe`, no target
/// feature and no fallback path to keep correct. The point of a tag line is to make the common
/// case one cache line instead of sixteen scattered addresses; getting eight lanes rather than
/// sixteen keeps most of that.
#[inline]
const fn match_byte(word: u64, byte: u8) -> u64 {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let x = word ^ (LO.wrapping_mul(byte as u64));
    x.wrapping_sub(LO) & !x & HI
}

/// Find `a`'s slot, or the first empty slot in probe order.
///
/// Two reads per group in the common case: the tag line, then one address for each lane whose tag
/// matched. Scanning all sixteen addresses instead costs 128 bytes a group against 16, and
/// measured 2.3x on the whole workload -- the tag line is not an optimisation on top of the
/// descent, it is what makes the descent affordable.
///
/// Stops at the first group holding an empty slot: placement is first-fit and occupancy only
/// grows, so an empty slot means the key would have been placed at or before here, and is absent.
#[inline]
fn locate(region: &Region, a: u64) -> (usize, bool) {
    let groups = region.slots.len() / GROUP;
    let gbits = groups.trailing_zeros();
    let gmask = groups - 1;
    let p = finish(a);
    let tag = tag_of(p);
    let mut j = 0u32;
    loop {
        let g = group_of(p, j, gbits, gmask);
        let base = g * GROUP;
        for half in 0..2 {
            let word = region.tags[g * 2 + half];
            let mut hits = match_byte(word, tag);
            while hits != 0 {
                let lane = base + half * 8 + (hits.trailing_zeros() as usize / 8);
                if region.slots[lane].0 == a {
                    return (lane, true);
                }
                hits &= hits - 1;
            }
        }
        // Only now look for an empty lane: the hit path above never reads this.
        for half in 0..2 {
            let empties = match_byte(region.tags[g * 2 + half], 0);
            if empties != 0 {
                return (
                    base + half * 8 + (empties.trailing_zeros() as usize / 8),
                    false,
                );
            }
        }
        j += 1;
    }
}

/// A shard's storage: a tag line beside the slots, so a group scan touches one cache line.
#[derive(Clone)]
pub struct Region {
    /// Two `u64` of eight tag bytes per group.
    tags: Vec<u64>,
    slots: Vec<(u64, i64)>,
}

impl Region {
    fn new(slots: usize) -> Self {
        Self {
            tags: vec![0; slots / 8],
            slots: vec![(u64::MAX, 0); slots],
        }
    }

    fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn set_tag(&mut self, lane: usize, tag: u8) {
        let (w, b) = (lane / 8, lane % 8);
        self.tags[w] = (self.tags[w] & !(0xFFu64 << (b * 8))) | ((tag as u64) << (b * 8));
    }
}

/// A region for `load` keys: at least two groups, and at most half full, rounded to a power of
/// two so the group index is a mask.
fn region_for(load: usize, hint_bits: u32) -> Region {
    if load == 0 {
        return Region::new(0);
    }
    let _ = hint_bits;
    Region::new((load * 2).max(GROUP * 2).next_power_of_two())
}

#[inline]
fn insert_into(region: &mut Region, a: u64) {
    let (i, found) = locate(region, a);
    if found {
        region.slots[i].1 += 1;
    } else {
        region.slots[i] = (a, 1);
        region.set_tag(i, tag_of(finish(a)));
    }
}

/// Sort addresses into shards. Counting pass, prefix sum, sequential scatter.
fn partition(addresses: &[u64]) -> Vec<Vec<u64>> {
    let mut counts = [0usize; SHARDS];
    for &a in addresses {
        counts[shard_of(a)] += 1;
    }
    let mut shards: Vec<Vec<u64>> = counts.iter().map(|&c| Vec::with_capacity(c)).collect();
    for &a in addresses {
        shards[shard_of(a)].push(a);
    }
    shards
}
