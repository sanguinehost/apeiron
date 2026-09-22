//! The single storage representation for `Apeiron.Table`.
//!
//! A total default plus growable, home-partitioned regions. A key's home selects one of
//! [`SHARD_COUNT`] regions; a region grows by reallocating and re-placing every resident entry, so
//! there is no fixed total capacity. Reads are total: an unwritten key reads the default.
//!
//! The region shape is the extractable one: a `Vec<u8>` tag line beside a `Vec` of owned slots, an
//! explicit search-state [`Locate`] with no `return` inside its nested loops, and a bounded probe.
//! Placement identity is decided by full-key equality, never by the tag or the home word.

/// Number of home regions. A key's home word selects its region by residue.
const SHARD_COUNT: usize = 256;

/// Bit mask for the 7-bit slot tag. The tag's top bit marks an occupied slot; a zero tag is empty.
const TAG_MASK: u64 = 0x7F;

/// A checked resource failure, not a logic error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// A region's target slot count overflowed `usize`, or the const group width `N` is zero.
    SizeOverflow,
    /// Slot or tag storage could not be reserved. The table is unchanged.
    AllocationFailed,
    /// Retained for compatibility with the bounded fixed-shard surface. The growable
    /// representation does not produce this: an absent key grows its home region instead.
    ShardFull,
}

/// Residue of a placement word into a home region.
fn shard_index(home: u64) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "residue below 256 fits usize"
    )]
    {
        (home % SHARD_COUNT as u64) as usize
    }
}

/// Selects placement, not identity. Must be stable and respect key equality.
pub trait Home<K> {
    /// Returns the placement word for `key`.
    fn home(&self, key: &K) -> u64;
}

impl<K, F: Fn(&K) -> u64> Home<K> for F {
    fn home(&self, key: &K) -> u64 {
        self(key)
    }
}

/// New-first value combination. Total, pure instances are required for refinement.
pub trait Merge<V> {
    /// Combines the new contribution with the previous value.
    fn merge(&self, new: V, old: V) -> V;
}

impl<V, F: Fn(V, V) -> V> Merge<V> for F {
    fn merge(&self, new: V, old: V) -> V {
        self(new, old)
    }
}

/// Word placement by bit pattern; truncation to shards never replaces full-key equality.
pub struct WordHome;

impl Home<i64> for WordHome {
    fn home(&self, key: &i64) -> u64 {
        #[allow(clippy::cast_sign_loss, reason = "placement uses the word bit pattern")]
        {
            *key as u64
        }
    }
}

/// Wrapping word addition, usable without a stored function pointer.
pub struct WordMerge;

impl Merge<i64> for WordMerge {
    fn merge(&self, new: i64, old: i64) -> i64 {
        new.wrapping_add(old)
    }
}

/// Word replacement with explicit new-first merge order.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplacementMerge;

impl Merge<i64> for ReplacementMerge {
    fn merge(&self, new: i64, _old: i64) -> i64 {
        new
    }
}

/// The outcome of a probe over one home region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Locate {
    /// The key is stored at this slot index.
    Found(usize),
    /// The key is absent; this slot index is the first vacant lane in probe order.
    Vacant(usize),
    /// No vacant lane and no match: the region has no room for an absent key.
    Exhausted,
}

/// One home region: a tag byte per slot beside the owned slot.
struct Region<K, V> {
    tags: Vec<u8>,
    slots: Vec<Option<(K, V)>>,
}

impl<K, V> Region<K, V> {
    fn empty() -> Self {
        Self {
            tags: Vec::new(),
            slots: Vec::new(),
        }
    }
}

/// A slot's occupancy tag: seven bits of the placement word with the top bit set. Zero is empty.
#[inline]
const fn tag_of(home: u64) -> u8 {
    0x80 | (home & TAG_MASK) as u8
}

/// Probes one region for `key`, whose placement word is `home`, using a group width of `n`.
///
/// The search visits every tag group exactly once, starting at the group selected by the high bits
/// of `home` and stepping by one with wraparound, so a vacant lane is found whenever the region is
/// not full and a live match is never concealed by an earlier vacancy. There is no `return` inside
/// the nested loops, so the body is Aeneas-translatable.
///
/// Temporal `O(groups * n)` `u8`/key comparisons in the worst case and `O(hit rate * groups * n)`
/// on a present key, counted in comparisons; `groups = slots / n`. Spatial `O(1)`: a borrow over
/// the region, no allocation. Derivation: the outer `while` advances `k` by one for `groups`
/// iterations, the inner `while` advances `lane` by one for `n` iterations, and both are bounded by
/// the region geometry.
fn locate<K: Eq, V>(region: &Region<K, V>, home: u64, key: &K, n: usize) -> Locate {
    let slots = region.slots.len();
    if slots == 0 || n == 0 {
        return Locate::Exhausted;
    }
    let groups = slots / n;
    let gmask = groups - 1;
    let tag = tag_of(home);
    let start = ((home >> 32) as usize) & gmask;
    let mut vacant: Option<usize> = None;
    let mut found: Option<usize> = None;
    let mut k = 0usize;
    while k < groups && found.is_none() {
        let base = ((start + k) & gmask) * n;
        let mut lane = 0usize;
        while lane < n && found.is_none() {
            let idx = base + lane;
            if region.tags[idx] == tag {
                if let Some((stored, _)) = &region.slots[idx] {
                    if stored == key {
                        found = Some(idx);
                    }
                }
            }
            if found.is_none() && vacant.is_none() && region.tags[idx] == 0 {
                vacant = Some(idx);
            }
            lane += 1;
        }
        k += 1;
    }
    match found {
        Some(i) => Locate::Found(i),
        None => match vacant {
            Some(i) => Locate::Vacant(i),
            None => Locate::Exhausted,
        },
    }
}

/// A total default plus growable, home-partitioned regions.
///
/// `N` is the tag-group width in lanes; a positive `N` bounds neither the keys per home nor the
/// total key count. `N = 0` has no admissible insertion and refuses with
/// [`StorageError::SizeOverflow`]. Full regions grow; they do not refuse.
pub struct Table<K, V, H, const N: usize = 16> {
    default: V,
    home: H,
    shards: Vec<Region<K, V>>,
    len: usize,
}

impl<K, V, H: Home<K>, const N: usize> Table<K, V, H, N> {
    /// Creates an empty table whose total function initially returns `default`.
    #[must_use]
    pub fn new(default: V, home: H) -> Self {
        Self {
            default,
            home,
            shards: (0..SHARD_COUNT).map(|_| Region::empty()).collect(),
            len: 0,
        }
    }
}

impl<K: Eq, V, H: Home<K>, const N: usize> Table<K, V, H, N> {
    /// Returns the stored override, or the total default for an unwritten key.
    ///
    /// Temporal `O(groups * n)` key comparisons in the worst case, from [`locate`]; `O(1)` other
    /// work. Spatial `O(1)`; the result is a borrow, not a copy.
    #[must_use]
    pub fn get(&self, key: &K) -> &V {
        let home = self.home.home(key);
        let shard = shard_index(home);
        if let Some(region) = self.shards.get(shard) {
            if let Locate::Found(i) = locate(region, home, key, N) {
                if let Some((_, value)) = &region.slots[i] {
                    return value;
                }
            }
        }
        &self.default
    }
}

impl<K: Eq + Clone, V: Clone, H: Home<K>, const N: usize> Table<K, V, H, N> {
    /// Number of stored override slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no override is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Applies `merge(new, old)`, including the default on the first write.
    ///
    /// Temporal: one [`locate`] (`O(groups * n)`) plus, on an absent key, a placement; a full
    /// region triggers one [`Self::grow`]. Amortized over an insertion sequence the reinsertion
    /// cost is `O(1)` per key because a region doubles. Spatial `O(1)` of its own; the home region
    /// owns its storage and is reallocated only on growth.
    ///
    /// # Errors
    /// Returns [`StorageError::SizeOverflow`] when `N = 0` or a region's target size overflows
    /// `usize`, and [`StorageError::AllocationFailed`] when slot/tag storage cannot be reserved.
    /// On error the table is unchanged and neither clone nor merge is invoked.
    pub fn insert_with<F: Merge<V>>(
        &mut self,
        merge: &F,
        key: K,
        value: V,
    ) -> Result<(), StorageError> {
        if N == 0 {
            return Err(StorageError::SizeOverflow);
        }
        let home = self.home.home(&key);
        let shard = shard_index(home);
        loop {
            match locate(&self.shards[shard], home, &key, N) {
                Locate::Found(i) => {
                    let old = self.shards[shard].slots[i]
                        .as_ref()
                        .map_or_else(|| self.default.clone(), |(_, old)| old.clone());
                    let merged = merge.merge(value, old);
                    self.shards[shard].slots[i] = Some((key, merged));
                    return Ok(());
                }
                Locate::Vacant(i) => {
                    let merged = merge.merge(value, self.default.clone());
                    self.shards[shard].slots[i] = Some((key, merged));
                    self.shards[shard].tags[i] = tag_of(home);
                    self.len += 1;
                    return Ok(());
                }
                Locate::Exhausted => self.grow(shard)?,
            }
        }
    }

    /// Doubles a region's capacity and re-places every resident entry.
    ///
    /// Temporal `O(capacity)` key comparisons and clones: one full reinsertion pass; counted in
    /// resident entries and comparisons. Spatial `O(capacity)` slots and bytes for the new region,
    /// held until the old region is dropped. Called only when [`locate`] reports the region full.
    ///
    /// # Errors
    /// Returns the checked sizing or reservation failure; the region is replaced only after the new
    /// storage and every reinsertion succeed, so a failure leaves it unchanged.
    fn grow(&mut self, shard: usize) -> Result<(), StorageError> {
        let old = &self.shards[shard];
        let old_len = old.slots.len();
        let min_len = N.max(1) * 2;
        let target = if old_len == 0 {
            min_len
        } else {
            old_len.checked_mul(2).ok_or(StorageError::SizeOverflow)?
        };
        let new_len = target.max(min_len).next_power_of_two();

        let mut tags: Vec<u8> = Vec::new();
        let mut slots: Vec<Option<(K, V)>> = Vec::new();
        tags.try_reserve_exact(new_len)
            .map_err(|_| StorageError::AllocationFailed)?;
        slots
            .try_reserve_exact(new_len)
            .map_err(|_| StorageError::AllocationFailed)?;
        tags.resize(new_len, 0u8);
        slots.resize_with(new_len, || None);
        let mut fresh = Region { tags, slots };

        for index in 0..old_len {
            if let Some((key, value)) = &old.slots[index] {
                let home = self.home.home(key);
                match locate(&fresh, home, key, N) {
                    Locate::Vacant(i) => {
                        fresh.slots[i] = Some((key.clone(), value.clone()));
                        fresh.tags[i] = tag_of(home);
                    }
                    Locate::Found(_) | Locate::Exhausted => {
                        return Err(StorageError::AllocationFailed);
                    }
                }
            }
        }
        self.shards[shard] = fresh;
        Ok(())
    }

    /// Builds from a borrowed stream, preserving per-key order without a partial table.
    ///
    /// Temporal `O(m * groups * n)` for `m` pairs: one [`Self::insert_with`] per pair, amortized by
    /// region growth. Spatial `O(1)` beyond the table itself and any allocation inside the caller's
    /// `merge`/`clone`.
    ///
    /// # Errors
    /// Returns the first insertion error; no partial table is returned.
    pub fn build<F: Merge<V>>(
        merge: &F,
        default: V,
        kvs: &[(K, V)],
        home: H,
    ) -> Result<Self, StorageError> {
        let mut state = Self::new(default, home);
        let mut i = 0;
        while i < kvs.len() {
            let (key, value) = &kvs[i];
            state.insert_with(merge, key.clone(), value.clone())?;
            i += 1;
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::{StorageError, Table, WordHome, WordMerge};

    #[test]
    fn absent_key_grows_the_region_instead_of_refusing() {
        let mut table = Table::<i64, i64, _, 1>::new(0, WordHome);
        for k in 0..1000 {
            table.insert_with(&WordMerge, k, 1).unwrap();
        }
        assert_eq!(table.len(), 1000);
        for k in 0..1000 {
            assert_eq!(*table.get(&k), 1);
        }
    }

    #[test]
    fn zero_group_width_has_no_admissible_insertion() {
        let mut table = Table::<i64, i64, _, 0>::new(7, WordHome);
        assert_eq!(*table.get(&99), 7);
        assert_eq!(
            table.insert_with(&WordMerge, 99, 1),
            Err(StorageError::SizeOverflow)
        );
        assert!(Table::<i64, i64, _, 0>::build(&WordMerge, 7, &[], WordHome).is_ok());
    }

    #[test]
    fn refusal_path_does_not_invoke_merge() {
        let mut table = Table::<i64, i64, _, 0>::new(7, WordHome);
        let result = table.insert_with(&|_, _| panic!("merge on refusal"), 0, 1);
        assert_eq!(result, Err(StorageError::SizeOverflow));
        assert_eq!(table.len(), 0);
    }
}
