//! The single storage representation for `Apeiron.Table`.
//!
//! The representation is a total default plus finite, full-key overrides.

const SHARD_COUNT: usize = 256;

/// A physical resource failure while changing a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageError {
    /// The requested capacity cannot be represented by `usize`.
    CapacityOverflow,
    /// The allocator declined a requested reservation.
    AllocationFailed,
}

/// A constant default plus finite overrides, partitioned by home address.
///
/// Every lookup must return a value, including the default for an unwritten key.
/// Home addresses select storage, never key identity; collisions require full key equality.
pub struct Table<K, V> {
    default: V,
    home: fn(&K) -> u64,
    shards: Vec<Vec<(K, V)>>,
}

impl<K, V> Table<K, V> {
    /// Creates an empty table whose total function initially returns `default`.
    #[must_use]
    pub fn new(default: V, home: fn(&K) -> u64) -> Self {
        Self {
            default,
            home,
            shards: Vec::new(),
        }
    }

    fn shard_index(&self, key: &K) -> usize {
        usize::try_from((self.home)(key) % (SHARD_COUNT as u64)).unwrap_or_default()
    }

    fn initialize_shards(&mut self) -> Result<(), StorageError> {
        if self.shards.is_empty() {
            self.shards
                .try_reserve_exact(SHARD_COUNT)
                .map_err(|_| StorageError::AllocationFailed)?;
            self.shards.resize_with(SHARD_COUNT, Vec::new);
        }
        Ok(())
    }

    fn additional_capacity(len: usize, capacity: usize) -> Result<usize, StorageError> {
        let required = len.checked_add(1).ok_or(StorageError::CapacityOverflow)?;
        Ok(required.saturating_sub(capacity))
    }

    fn reserve_entry(shard: &mut Vec<(K, V)>) -> Result<(), StorageError> {
        let additional = Self::additional_capacity(shard.len(), shard.capacity())?;
        shard
            .try_reserve_exact(additional)
            .map_err(|_| StorageError::AllocationFailed)
    }
}

impl<K: Eq, V> Table<K, V> {
    /// Returns the stored override for `key`, or the total default when no override exists.
    #[must_use]
    pub fn get(&self, key: &K) -> &V {
        self.shards
            .get(self.shard_index(key))
            .and_then(|shard| shard.iter().find(|(stored, _)| stored == key))
            .map_or(&self.default, |(_, value)| value)
    }
}

impl<K: Eq, V: Clone> Table<K, V> {
    /// Inserts `value` at `key`, applying `merge(new, old)` even on the first write.
    ///
    /// A failed reservation leaves the logical mapping unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when capacity arithmetic overflows or allocation fails.
    pub fn insert_with<F>(&mut self, merge: F, key: K, value: V) -> Result<(), StorageError>
    where
        F: Fn(V, V) -> V,
    {
        let shard_index = self.shard_index(&key);
        if let Some((_, stored)) = self
            .shards
            .get_mut(shard_index)
            .and_then(|shard| shard.iter_mut().find(|(stored_key, _)| stored_key == &key))
        {
            *stored = merge(value, stored.clone());
            return Ok(());
        }

        let merged = merge(value, self.default.clone());
        self.initialize_shards()?;
        let shard = &mut self.shards[shard_index];
        Self::reserve_entry(shard)?;
        shard.push((key, merged));
        Ok(())
    }

    /// Builds a table by applying [`Self::insert_with`] to `kvs` in stream order.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when any insertion cannot reserve storage. No partial table is
    /// returned.
    pub fn build<F>(
        merge: F,
        default: V,
        kvs: Vec<(K, V)>,
        home: fn(&K) -> u64,
    ) -> Result<Self, StorageError>
    where
        F: Fn(V, V) -> V,
    {
        let mut table = Self::new(default, home);
        for (key, value) in kvs {
            table.insert_with(&merge, key, value)?;
        }
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::{StorageError, Table};

    #[test]
    fn reads_the_default_before_any_allocation() {
        let table = Table::new(7_u32, |key: &u64| *key);
        assert_eq!(*table.get(&99), 7);
        assert!(table.shards.is_empty());
    }

    #[test]
    fn first_write_merges_against_the_default_and_replacement_merges_against_the_override() {
        let mut table = Table::new(10_i32, |key: &u64| *key);
        table.insert_with(|new, old| new - old, 1, 3).unwrap();
        assert_eq!(*table.get(&1), -7);
        table.insert_with(|new, old| new - old, 1, 5).unwrap();
        assert_eq!(*table.get(&1), 12);
    }

    #[test]
    fn full_keys_disambiguate_home_collisions() {
        let mut table = Table::new(0_u32, |_| 0);
        table.insert_with(|new, _| new, 1, 11).unwrap();
        table.insert_with(|new, _| new, 2, 22).unwrap();
        assert_eq!(*table.get(&1), 11);
        assert_eq!(*table.get(&2), 22);
    }

    #[test]
    fn unrepresentable_entry_capacity_is_an_explicit_resource_failure() {
        assert_eq!(
            Table::<u8, u8>::additional_capacity(usize::MAX, usize::MAX),
            Err(StorageError::CapacityOverflow)
        );
    }
}
