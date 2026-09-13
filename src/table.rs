//! The single storage representation for `Apeiron.Table`.
//!
//! A total default plus bounded, full-key overrides. Table storage never allocates;
//! key/value operations and caller-supplied callbacks may allocate or panic.

const SHARD_COUNT: usize = 256;

/// A checked storage admission failure, not an allocator error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// A new key's home shard has no vacant slot. The table is unchanged.
    ShardFull,
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

/// A constant default plus at most `N` overrides per home shard.
///
/// `N = 0` is valid: reads return the default and all insertions refuse.
/// Full shards still allow updates to existing keys. No key/value `Copy` bound is needed.
pub struct Table<K, V, H, const N: usize = 16> {
    default: V,
    home: H,
    #[allow(
        clippy::type_complexity,
        reason = "explicit inline shard and slot layout"
    )]
    shards: [Option<[Option<(K, V)>; N]>; SHARD_COUNT],
}

impl<K, V, H: Home<K>, const N: usize> Table<K, V, H, N> {
    /// Creates an empty table whose total function initially returns `default`.
    #[must_use]
    pub fn new(default: V, home: H) -> Self {
        Self {
            default,
            home,
            shards: [const { None }; SHARD_COUNT],
        }
    }

    fn shard_index(&self, key: &K) -> usize {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "residue below 256 fits usize"
        )]
        {
            (self.home.home(key) % 256) as usize
        }
    }
}

impl<K: Eq, V, H: Home<K>, const N: usize> Table<K, V, H, N> {
    fn find(&self, shard: usize, key: &K) -> Option<usize> {
        if let Some(slots) = &self.shards[shard] {
            return Self::find_slots(slots, key);
        }
        None
    }

    fn find_slots(slots: &[Option<(K, V)>; N], key: &K) -> Option<usize> {
        let mut i = 0;
        while i < N {
            if let Some((stored, _)) = &slots[i] {
                if stored == key {
                    return Some(i);
                }
            }
            i += 1;
        }
        None
    }

    /// Returns the stored override, or the total default for an unwritten key.
    #[must_use]
    pub fn get(&self, key: &K) -> &V {
        let shard = self.shard_index(key);
        if let Some(i) = self.find(shard, key) {
            if let Some(slots) = &self.shards[shard] {
                if let Some((_, value)) = &slots[i] {
                    return value;
                }
            }
        }
        &self.default
    }
}

impl<K: Eq, V: Clone, H: Home<K>, const N: usize> Table<K, V, H, N> {
    /// Applies `merge(new, old)`, including the default on the first write.
    ///
    /// # Errors
    /// Returns [`StorageError::ShardFull`] exactly when the key is absent and its
    /// shard has no vacant slot. Refusal invokes neither clone nor merge and changes no field.
    pub fn insert_with<F: Merge<V>>(
        &mut self,
        merge: &F,
        key: K,
        value: V,
    ) -> Result<(), StorageError> {
        let shard = self.shard_index(&key);
        if let Some(i) = self.find(shard, &key) {
            if let Some(slots) = &mut self.shards[shard] {
                if let Some((_, stored)) = &mut slots[i] {
                    *stored = merge.merge(value, stored.clone());
                    return Ok(());
                }
            }
        }
        if N == 0 {
            return Err(StorageError::ShardFull);
        }
        if self.shards[shard].is_none() {
            let mut slots = [const { None }; N];
            let merged = merge.merge(value, self.default.clone());
            slots[0] = Some((key, merged));
            self.shards[shard] = Some(slots);
            return Ok(());
        }
        if let Some(slots) = &mut self.shards[shard] {
            let mut i = 0;
            while i < N {
                if slots[i].is_none() {
                    let merged = merge.merge(value, self.default.clone());
                    slots[i] = Some((key, merged));
                    return Ok(());
                }
                i += 1;
            }
        }
        Err(StorageError::ShardFull)
    }

    /// Builds from a borrowed stream, preserving per-key order without allocating table storage.
    ///
    /// # Errors
    /// Returns [`StorageError::ShardFull`] at the first refused insertion; no partial table is
    /// returned. Key/value clones and callbacks retain their own resource and panic behavior.
    pub fn build<F: Merge<V>>(
        merge: &F,
        default: V,
        kvs: &[(K, V)],
        home: H,
    ) -> Result<Self, StorageError>
    where
        K: Clone,
    {
        let mut state = Self::new(default, home);
        let mut i = 0;
        while i < kvs.len() {
            let (key, value) = &kvs[i];
            match state.insert_with(merge, key.clone(), value.clone()) {
                Ok(()) => (),
                Err(error) => return Err(error),
            }
            i += 1;
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::{StorageError, Table, WordHome, WordMerge};

    #[test]
    fn full_shard_refuses_unchanged_but_existing_key_still_updates() {
        let mut table = Table::<_, _, _, 1>::new(10, WordHome);
        table.insert_with(&WordMerge, 0, 3).unwrap();
        assert_eq!(
            table.insert_with(&WordMerge, 256, 9),
            Err(StorageError::ShardFull)
        );
        assert_eq!(*table.get(&0), 13);
        assert_eq!(*table.get(&256), 10);
        table.insert_with(&WordMerge, 0, 5).unwrap();
        assert_eq!(*table.get(&0), 18);
        table.insert_with(&WordMerge, 1, 1).unwrap();
        assert_eq!(*table.get(&1), 11);
    }

    #[test]
    fn zero_capacity_and_build_refusal_are_explicit() {
        let mut table = Table::<_, _, _, 0>::new(7, WordHome);
        assert_eq!(*table.get(&99), 7);
        assert_eq!(
            table.insert_with(&WordMerge, 99, 1),
            Err(StorageError::ShardFull)
        );
        assert!(Table::<_, _, _, 0>::build(&WordMerge, 7, &[], WordHome).is_ok());
        assert!(matches!(
            Table::<_, _, _, 1>::build(&WordMerge, 7, &[(0, 1), (256, 2)], WordHome),
            Err(StorageError::ShardFull)
        ));
    }

    #[test]
    fn refusal_does_not_invoke_merge() {
        let mut table = Table::<_, _, _, 0>::new(7, WordHome);
        let result = table.insert_with(&|_, _| panic!("merge on refusal"), 0, 1);
        assert_eq!(result, Err(StorageError::ShardFull));
    }
}
