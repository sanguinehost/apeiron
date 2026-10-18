//! Extraction unit for the hashtrinity map body (`src/map/hashtrinity/`).
//!
//! The module tree is already rooted in the extraction unit as `crate::hashtrinity`
//! (`proof/extract/lib.rs`, the `#[path = "../../src/map/hashtrinity/mod.rs"] pub mod hashtrinity`
//! declaration), so this file declares no second `#[path]` root: a second inclusion would translate
//! every body twice. Each `word_*` wrapper below fixes the monomorphic key/value words and calls the
//! real body, restating no formula.
//!
//! The reachable set is the first-order surface: `Option`/`Result` and `match`, no `Fn`/`FnOnce`
//! argument, no `impl Iterator` return, no thread dispatch, no `_mm_prefetch`. The faces outside it
//! are named in the lane return, not silently dropped.
//!
//! `Table::remove` / `CoordMap::remove` are NOT rooted here. Their blocked-overflow path reaches
//! `alloc.vec.Vec.remove` (`overflow.rs:450,473`) and `core.slice.Slice.last` (`overflow.rs:616,647`),
//! neither of which the pinned Lean model defines, so the extraction emits them as `axiom`s and
//! `proof/regen.sh`'s audit rejects the file. Two roots and ~10 bodies wait on that model entry.

use crate::hashtrinity::seed;
use crate::hashtrinity::{Coord, FreqMap, Table};

// ── The coordinate seed (`seed.rs`): the stored-residual addressing ────────────────────────────

/// Real `CoordSeed::new`, rooted at the concrete `u32` window width.
pub fn word_seed_new(bits: u32) -> seed::CoordSeed {
    seed::CoordSeed::new(bits)
}

/// Real `seed::fold`, rooted at the concrete `u64` coordinate.
pub fn word_seed_fold(c: u64) -> u64 {
    seed::fold(c)
}

/// Real `seed::unfold`, rooted at the concrete `u64` folded word.
pub fn word_seed_unfold(p: u64) -> u64 {
    seed::unfold(p)
}

/// Real `CoordSeed::home`, the slot index of `c`.
pub fn word_seed_home(s: seed::CoordSeed, c: u64) -> u64 {
    s.home(c)
}

/// Real `CoordSeed::residual`, the stored part of `fold(c)`.
pub fn word_seed_residual(s: seed::CoordSeed, c: u64) -> u64 {
    s.residual(c)
}

/// Real `CoordSeed::regrow_coord`, the coordinate rebuilt from `(home, residual)`.
pub fn word_seed_regrow_coord(s: seed::CoordSeed, home: u64, residual: u64) -> u64 {
    s.regrow_coord(home, residual)
}

/// Real `CoordSeed::regrow::<u64>`, the key rebuilt from `(home, residual)`.
pub fn word_seed_regrow(s: seed::CoordSeed, home: u64, residual: u64) -> u64 {
    s.regrow::<u64>(home, residual)
}

/// Real `CoordSeed::seed_of::<u64>`, the `(home, residual)` pair a readout stores.
pub fn word_seed_seed_of(s: seed::CoordSeed, k: u64) -> (u64, u64) {
    s.seed_of::<u64>(&k)
}

// ── The coordinate folds (`algebra.rs`): the key-to-address halves ────────────────────────────

/// Real `<u128 as Coord>::coord` — the golden-pair fold.
pub fn word_coord_u128(v: u128) -> u64 {
    <u128 as Coord>::coord(&v)
}

/// Real `<i128 as Coord>::coord`.
pub fn word_coord_i128(v: i128) -> u64 {
    <i128 as Coord>::coord(&v)
}

/// Real `<bool as Coord>::coord`.
pub fn word_coord_bool(v: bool) -> u64 {
    <bool as Coord>::coord(&v)
}

/// Real `<[u8] as Coord>::coord` — the golden-128 byte fold (`fold_bytes`, `folded_mul`).
pub fn word_coord_bytes(v: &[u8]) -> u64 {
    <[u8] as Coord>::coord(v)
}

// ── The selected map body (`mod.rs`): shape, removal, exact read ──────────────────────────────

/// Real `Table::len` — primary, side and overflow entries counted.
pub fn word_table_len(state: &Table<u64, i64>) -> usize {
    state.len()
}

/// Real `Table::is_empty`.
pub fn word_table_is_empty(state: &Table<u64, i64>) -> bool {
    state.is_empty()
}

/// Real `Table::capacity` — the slot count.
pub fn word_table_capacity(state: &Table<u64, i64>) -> usize {
    state.capacity()
}

/// Real `Table::tombstones` — the `TOMB` slots a rebuild has yet to reclaim.
pub fn word_table_tombstones(state: &Table<u64, i64>) -> usize {
    state.tombstones()
}

/// Real `Table::coord_seed` — the seed matched to this table's slot count.
pub fn word_table_coord_seed(state: &Table<u64, i64>) -> seed::CoordSeed {
    state.coord_seed()
}

/// Real `Table::get_exact` — bit-exact read through `get_coord`.
pub fn word_table_get_exact(state: &Table<u64, i64>, key: u64) -> Option<i64> {
    match state.get_exact(&key) {
        Some(v) => Some(*v),
        None => None,
    }
}

// ── The count face (`freq.rs`): the group probe's read and shape ──────────────────────────────

/// Real `FreqMap::<u64, i64>::get` — the group probe's read (`get_p`, `probe_tagged`).
pub fn word_freq_get(state: &FreqMap<u64, i64>, key: u64) -> Option<i64> {
    match state.get(&key) {
        Some(v) => Some(*v),
        None => None,
    }
}

/// Real `FreqMap::len`.
pub fn word_freq_len(state: &FreqMap<u64, i64>) -> usize {
    state.len()
}

/// Real `FreqMap::is_empty`.
pub fn word_freq_is_empty(state: &FreqMap<u64, i64>) -> bool {
    state.is_empty()
}

/// Real `FreqMap::capacity` — the frequency-line length.
pub fn word_freq_capacity(state: &FreqMap<u64, i64>) -> usize {
    state.capacity()
}

/// Real `FreqMap::overflow_len`.
pub fn word_freq_overflow_len(state: &FreqMap<u64, i64>) -> usize {
    state.overflow_len()
}

/// Real `FreqMap::overflow_height`.
pub fn word_freq_overflow_height(state: &FreqMap<u64, i64>) -> u32 {
    state.overflow_height()
}
