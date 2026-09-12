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
            slots: [const { None }; N],
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
    fn find_from(&self, key: &K, start: usize) -> Option<usize> {
        let mut index = start;
        while index < N {
            match &self.slots[index] {
                Some((stored, _)) if stored == key => return Some(index),
                _ => index += 1,
            }
        }
        None
    }

    fn empty_from(&self, start: usize) -> Option<usize> {
        let mut index = start;
        while index < N {
            match &self.slots[index] {
                None => return Some(index),
                Some(_) => index += 1,
            }
        }
        None
    }

    /// Value stored at `key`.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        match self.find_from(key, 0) {
            Some(index) => match &self.slots[index] {
                Some((_, value)) => Some(value),
                None => None,
            },
            None => None,
        }
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
        if let Some(index) = self.find_from(&key, 0) {
            return match &mut self.slots[index] {
                Some((_, old)) => Ok(Some(std::mem::replace(old, value))),
                None => Err(TableFull { key, value }),
            };
        }

        if let Some(index) = self.empty_from(0) {
            self.slots[index] = Some((key, value));
            self.len += 1;
            Ok(None)
        } else {
            Err(TableFull { key, value })
        }
    }

    /// Remove `key`, returning its value when present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.find_from(key, 0) {
            Some(index) => match self.slots[index].take() {
                Some((_, value)) => {
                    self.len -= 1;
                    Some(value)
                }
                None => None,
            },
            None => None,
        }
    }
}
