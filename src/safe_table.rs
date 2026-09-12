//! A safe fixed-capacity table semantics.
//!
//! This is the executable reference for the core keyed transitions: lookup, replace and removal.
//! Storage is an inline array of initialized `Option` values, so there is no `unsafe`, allocation,
//! tag state, tombstone or placement order in its meaning. [`crate::map::Map`] is compared against
//! this surface before any relation to its optimized representation is claimed.

/// A refused insertion into a full table.
#[derive(Debug, PartialEq, Eq)]
pub struct TableFull<K, V> {
    /// The key that was not inserted.
    pub key: K,
    /// The value that was not inserted.
    pub value: V,
}

/// A safe table with room for at most `N` distinct keys.
///
/// Replacement of a present key succeeds even when all slots are occupied. Insertion of an absent
/// key into a full table returns [`TableFull`] and leaves the table unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTable<K, V, const N: usize> {
    slots: [Option<(K, V)>; N],
    len: usize,
}

impl<K, V, const N: usize> SafeTable<K, V, N> {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            len: 0,
        }
    }

    /// Number of stored keys.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether no key is stored.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of distinct keys.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Stored entries in physical slot order.
    ///
    /// Consumers that require canonical observation must order by key rather than use this order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.slots
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(key, value)| (key, value)))
    }
}

impl<K, V, const N: usize> Default for SafeTable<K, V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq, V, const N: usize> SafeTable<K, V, N> {
    /// Value stored at `key`.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.slots.iter().find_map(|slot| match slot {
            Some((stored, value)) if stored == key => Some(value),
            _ => None,
        })
    }

    /// Whether `key` is stored.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Insert `value`, returning the displaced value or an explicit full refusal.
    ///
    /// A replacement keeps the stored key, matching [`crate::map::Map::insert`].
    ///
    /// # Errors
    ///
    /// Returns [`TableFull`] with ownership of `key` and `value` when the key is absent and every
    /// slot is occupied. The table is unchanged in that case.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<Option<V>, TableFull<K, V>> {
        for (stored, old) in self.slots.iter_mut().flatten() {
            if stored == &key {
                return Ok(Some(std::mem::replace(old, value)));
            }
        }

        for slot in &mut self.slots {
            if slot.is_none() {
                *slot = Some((key, value));
                self.len += 1;
                return Ok(None);
            }
        }

        Err(TableFull { key, value })
    }

    /// Remove `key`, returning its value when present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        for slot in &mut self.slots {
            let matches = slot.as_ref().is_some_and(|(stored, _)| stored == key);
            if matches {
                self.len -= 1;
                return slot.take().map(|(_, value)| value);
            }
        }
        None
    }
}
