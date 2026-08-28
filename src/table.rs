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

use crate::coord::{bucket, finish};

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
    slots: Vec<(u64, i64)>,
    bits: u32,
}

impl CountTable {
    /// Build from a key stream. `bits` is the table's log2 size and must exceed `SHARD_BITS`.
    #[must_use]
    pub fn build(addresses: &[u64], bits: u32) -> Self {
        assert!(bits > SHARD_BITS, "table must be larger than one shard");
        let mut table = Self {
            slots: vec![(u64::MAX, 0); 1 << bits],
            bits,
        };
        let shards = partition(addresses);
        for keys in &shards {
            for &a in keys {
                table.insert(a);
            }
        }
        table
    }

    /// Build across `threads` workers. Shards hold disjoint keys, so each worker owns a run of
    /// buckets outright and there is no merge and no synchronisation on the hot path.
    #[must_use]
    pub fn build_parallel(addresses: &[u64], bits: u32, threads: usize) -> Self {
        assert!(bits > SHARD_BITS);
        let shards = partition(addresses);
        let n = 1usize << bits;
        let per_shard = n >> SHARD_BITS;
        let mut slots = vec![(u64::MAX, 0i64); n];

        // A shard owns buckets [s * per_shard, (s+1) * per_shard), which is a contiguous slice --
        // so the split is by ownership rather than by lock, and safe by construction.
        let mut regions: Vec<&mut [(u64, i64)]> = slots.chunks_mut(per_shard).collect();
        let work: Vec<(usize, &mut [(u64, i64)])> =
            regions.drain(..).enumerate().collect();

        std::thread::scope(|scope| {
            let chunk = work.len().div_ceil(threads.max(1));
            let mut rest = work;
            while !rest.is_empty() {
                let take = chunk.min(rest.len());
                let mine: Vec<(usize, &mut [(u64, i64)])> = rest.drain(..take).collect();
                let shards = &shards;
                scope.spawn(move || {
                    for (s, region) in mine {
                        for &a in &shards[s] {
                            insert_into(region, a, per_shard);
                        }
                    }
                });
            }
        });

        Self { slots, bits }
    }

    #[inline]
    fn insert(&mut self, a: u64) {
        let per_shard = self.slots.len() >> SHARD_BITS;
        let s = shard_of(a);
        let base = s * per_shard;
        insert_into(&mut self.slots[base..base + per_shard], a, per_shard);
    }

    /// The count for an address, or zero.
    #[must_use]
    pub fn get(&self, a: u64) -> i64 {
        let per_shard = self.slots.len() >> SHARD_BITS;
        let base = shard_of(a) * per_shard;
        let mut i = local_index(a, self.bits, per_shard);
        loop {
            let slot = self.slots[base + i];
            if slot.0 == u64::MAX {
                return 0;
            }
            if slot.0 == a {
                return slot.1;
            }
            i = (i + 1) & (per_shard - 1);
        }
    }

    #[must_use]
    pub fn bits(&self) -> u32 {
        self.bits
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

/// Where an address sits inside its own shard's region.
#[inline]
fn local_index(a: u64, bits: u32, per_shard: usize) -> usize {
    bucket(a, bits) as usize & (per_shard - 1)
}

#[inline]
fn insert_into(region: &mut [(u64, i64)], a: u64, per_shard: usize) {
    let mut i = {
        // `bits` is recoverable from the region: per_shard = 2^(bits - SHARD_BITS).
        let bits = per_shard.trailing_zeros() + SHARD_BITS;
        local_index(a, bits, per_shard)
    };
    loop {
        if region[i].0 == u64::MAX {
            region[i] = (a, 1);
            return;
        }
        if region[i].0 == a {
            region[i].1 += 1;
            return;
        }
        i = (i + 1) & (per_shard - 1);
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
