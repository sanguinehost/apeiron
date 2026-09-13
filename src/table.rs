//! The single storage representation for `Apeiron.Table`.
//!
//! Boundary declaration only: construction and transitions are not implemented yet.
//! The required API and refinement obligations are in `docs/kernel-boundary.md`.
//! No legacy container or compatibility implementation is retained.

/// A constant default plus finite overrides, partitioned by home address.
///
/// Every lookup must return a value, including the default for an unwritten key.
/// Home addresses select storage, never key identity; collisions require full key equality.
/// This declaration supplies no constructor or operations until their implementation lands.
pub struct Table<K, V> {
    default: V,
    home: fn(&K) -> u64,
    shards: Vec<Vec<(K, V)>>,
}
