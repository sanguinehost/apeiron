//! KV prefix cache: O(1) table lookup instead of O(prompt) prefill on the crate's Table.
//!
//! # Causality and Exactness
//!
//! In causal autoregressive transformer attention, the key/value state rows for sequence
//! positions `< m` depend exclusively on prefix tokens `t[0..m]`. Specifically, position `p`
//! consumes `t[p]` and previous keys `0..=p` through the causal lower-triangular mask; tokens at
//! positions `≥ m` have no backward impact on the keys, values, or normalization states at positions
//! `< m`. Therefore, a [`KvState`] snapshot captured after stepping tokens `t[0..m]` is bit-for-bit
//! identical to the state that a cold prefill reaches at position `m`.
//!
//! # Architecture and Full-Key Identity
//!
//! [`PrefixCache`] keys snapshots at block boundaries (every `B` tokens, default `B = 16`).
//! Probing checks prefixes from the longest boundary down: `m = (n / B) * B, ..., B`.
//! For prompt length `n`, this requires at most `n / B` lookups.
//!
//! Under Apeiron's Table architecture, every key lookup must respect the full-key identity rule:
//! a 64-bit hash hit alone is never trusted. Two distinct prefix token sequences could collide on
//! the 64-bit hash. A hit is accepted if and only if the full stored token sequence equals the
//! query prefix `t[0..m]`. A hash hit with mismatched tokens is treated as a miss.
//!
//! # Memory Footprint and O(n) Storage
//!
//! To avoid O(n²) memory overhead from cloning an independent [`KvState`] at every block boundary,
//! the cache retains exactly ONE [`KvState`] per distinct cached prompt in an arena. The Apeiron
//! [`Table`] maps 64-bit boundary keys to [`BoundaryRef`] `(entry_id, m)`.
//!
//! On a hit at boundary `m ≤ n`:
//! - If `m == entry.tokens.len()`, the stored state already holds exactly `m` occupied positions.
//! - If `m < entry.tokens.len()`, a clone of the prompt's state is truncated in-place to `m`
//!   positions via [`truncate_kv_state`], which sets the layer occupied counters to `m`.
//!
//! Snapshot memory is dominated by the key and value matrices. For a prompt at `n` positions:
//! - Per position, the KV footprint is `2 · kv_out · 8 · n_layer` bytes, where
//!   `kv_out = n_head_kv · head_dim`.
//! - For Qwen2.5-0.5B (`n_layer = 24`, `n_head_kv = 2`, `head_dim = 64`, `kv_out = 128`):
//!   `2 · 128 · 8 · 24 = 49,152` bytes/position (~49 KB/position).
//! - For Bonsai-8B (`n_layer = 32`, `n_head_kv = 8`, `head_dim = 128`, `kv_out = 1024`):
//!   `2 · 1024 · 8 · 32 = 524,288` bytes/position (~512 KB/position, or ~458 KB/position with non-padded layout).
//!
//! # Eviction
//!
//! [`PrefixCache::insert`] admits a snapshot only while `current_bytes + needed ≤ byte_budget`. When
//! a snapshot would exceed the budget, the cache evicts whole prompt entries in least-recently-used
//! order until it fits. Recency is a monotone `u64` tick stored per arena entry; both a successful
//! [`PrefixCache::insert`] and a confirmed [`PrefixCache::longest_prefix`] hit advance the tick. A
//! snapshot whose own footprint exceeds the whole budget is refused with
//! [`PrefixCacheError::BudgetExceeded`], and the cache is left unchanged.
//!
//! The Apeiron [`Table`] exposes no removal operation, so eviction frees an entry's arena slot and
//! its bytes but leaves that entry's boundary keys in the table. Stale keys are safe misses:
//! [`PrefixCache::longest_prefix`] confirms every candidate by comparing the stored token sequence
//! with the queried prefix `t[0..m]`, so a key whose slot is freed or reused by a *different* prompt
//! misses. When the reused slot's prompt shares the first `m` tokens the state is causally identical,
//! so the hit is correct.
//!
//! # Complexity
//!
//! - `PrefixCache::longest_prefix`: `O(n / B)` probes down boundary candidates, each probe an `O(1)`
//!   table lookup with `O(m)` full-key token comparison on hit, where `m ≤ n`, plus one `O(1)` tick
//!   update on a confirmed hit.
//! - `PrefixCache::insert`: `O(m)` hashing and token cloning, `O(m / B)` boundary insertions, plus
//!   eviction: one `O(entries)` LRU scan per evicted entry, so `O(entries²)` when the whole cache is
//!   recycled.
//! - `prefill_with_cache`: resumes from prefix `m` (saving `m` steps of `O(n_layer · ...)` compute),
//!   then steps the remaining `n - m` tokens. Overall speedup is proportional to `m / n`.

use crate::cell::Sigma;
use crate::coord::address::{
    FaceCoord, FaceHome, FaceResidueCapability, LocatorSigma, ParabolicDual, Residue, H3_40,
};
use crate::coord::locator::ExactIdentity;
use crate::coord::spread;
use crate::engine::attn::KvState;
use crate::engine::loader::Engine;
use crate::engine::step::{StepError, StepResult};
use crate::table::keyrep::Stored;
use crate::table::{Home, Merge, Table, TableAddress};

/// Default block boundary step in tokens.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// A hasher abstraction for 64-bit prefix hashing.
pub trait PrefixHasher: Send + Sync {
    /// Hashes a slice of token ids into a 64-bit key.
    fn hash_tokens(&self, tokens: &[u32]) -> u64;
}

/// FNV-1a 64-bit hasher: standard deterministic default.
#[derive(Debug, Clone, Copy, Default)]
pub struct FnvPrefixHasher;

impl PrefixHasher for FnvPrefixHasher {
    /// Temporal complexity: O(len) byte iterations; spatial: O(1) in registers.
    fn hash_tokens(&self, tokens: &[u32]) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0100_0000_01b3;

        let mut hash = FNV_OFFSET_BASIS;
        for &tok in tokens {
            for b in tok.to_le_bytes() {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash
    }
}

/// A reference from a block boundary to a cached prompt entry and token count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BoundaryRef {
    /// Arena slot index of the parent prompt entry.
    pub entry_id: u32,
    /// Token boundary length.
    pub m: u32,
}

/// Table merge policy for boundary references: replacement (newest snapshot wins).
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryMerge;

impl Merge<Option<BoundaryRef>> for BoundaryMerge {
    fn merge(&self, new: Option<BoundaryRef>, _old: Option<BoundaryRef>) -> Option<BoundaryRef> {
        new
    }
}

/// A stored entry for a distinct cached prompt in the arena.
#[derive(Clone)]
pub struct PromptEntry {
    /// Full token sequence for full-key identity confirmation.
    pub tokens: Vec<u32>,
    /// The captured KV snapshot after prefilling `tokens`.
    pub state: KvState,
    /// Estimated memory footprint in bytes.
    pub byte_size: usize,
    /// Monotone recency tick of the last use (insert or confirmed hit); lower means older.
    pub last_used: u64,
}

/// Key type for the Apeiron Table: carries the primary 64-bit hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrefixKey(pub u64);

impl ExactIdentity for PrefixKey {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        self.0
    }

    fn reserve_retain(&self) -> Result<(), crate::coord::locator::Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        self.0
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        self.0 == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.0.to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        retained.to_be_bytes()[index]
    }
}

/// Physical and semantic home for [`PrefixKey`] in the Apeiron Table.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefixHome;

impl FaceResidueCapability<PrefixKey> for PrefixHome {
    fn parabolic_dual(&self, key: &PrefixKey) -> ParabolicDual {
        ParabolicDual::new(spread(key.0).place, 0)
    }

    fn hyperbolic_scalar(&self, profile: H3_40, key: &PrefixKey) -> Residue {
        Residue::reduce(profile, key.0)
    }
}

impl FaceCoord<PrefixKey> for PrefixHome {
    type Faces = LocatorSigma;
}

impl FaceHome<PrefixKey> for PrefixHome {}

impl Home<PrefixKey> for PrefixHome {
    fn home(&self, key: &PrefixKey) -> u64 {
        key.0
    }

    fn address(&self, _sigma: Sigma, key: &PrefixKey) -> TableAddress {
        TableAddress {
            place: self.home(key),
            direct_word: None,
        }
    }
}

/// Refusal and runtime errors for prefix cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixCacheError {
    /// Inserting the snapshot would exceed the configured byte budget.
    BudgetExceeded {
        /// Requested snapshot footprint in bytes.
        needed: usize,
        /// Current cache usage in bytes.
        current: usize,
        /// Maximum allowed bytes.
        budget: usize,
    },
    /// The underlying Apeiron Table refused storage allocation.
    TableStorage(crate::table::StorageError),
    /// Prefill decode execution error.
    Step(StepError),
    /// Prompt tokens slice was empty.
    EmptyPrompt,
    /// Cache snapshot capacity was smaller than required prompt length.
    CapacityExceeded,
}

impl From<StepError> for PrefixCacheError {
    fn from(err: StepError) -> Self {
        Self::Step(err)
    }
}

/// The prefix cache using Apeiron's growable Table primitive.
///
/// Stores ONE [`KvState`] per distinct cached prompt in an arena, and indexes all
/// block boundaries in Apeiron's [`Table`] pointing to `(entry_id, m)`.
pub struct PrefixCache<H: PrefixHasher = FnvPrefixHasher> {
    /// Boundary block size in tokens.
    block_size: usize,
    /// Maximum allowed byte footprint across all retained snapshots.
    byte_budget: usize,
    /// Currently committed byte footprint.
    current_bytes: usize,
    /// Prompt arena holding one entry per distinct prompt.
    arena: Vec<Option<PromptEntry>>,
    /// Free slot indices in the arena.
    free_slots: Vec<usize>,
    /// Monotone recency clock; each use stamps an entry with the next value.
    tick: u64,
    /// Apeiron's Table mapping 64-bit hash keys to boundary references.
    table: Table<PrefixKey, Option<BoundaryRef>, PrefixHome, 16, Stored<PrefixKey>>,
    /// Pluggable hasher for token slices.
    hasher: H,
}

impl PrefixCache<FnvPrefixHasher> {
    /// Creates a new prefix cache with default FNV-1a hasher and default block size (16).
    #[must_use]
    pub fn new(byte_budget: usize) -> Self {
        Self::with_block_size_and_hasher(byte_budget, DEFAULT_BLOCK_SIZE, FnvPrefixHasher)
    }

    /// Creates a new prefix cache with custom block size and default FNV-1a hasher.
    #[must_use]
    pub fn with_block_size(byte_budget: usize, block_size: usize) -> Self {
        Self::with_block_size_and_hasher(byte_budget, block_size, FnvPrefixHasher)
    }
}

impl<H: PrefixHasher> PrefixCache<H> {
    /// Creates a new prefix cache with custom budget, block size, and hasher.
    #[must_use]
    pub fn with_block_size_and_hasher(byte_budget: usize, block_size: usize, hasher: H) -> Self {
        let block_size = block_size.max(1);
        Self {
            block_size,
            byte_budget,
            current_bytes: 0,
            arena: Vec::new(),
            free_slots: Vec::new(),
            tick: 0,
            table: Table::new(None, PrefixHome),
            hasher,
        }
    }

    /// Returns the configured block size in tokens.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the configured byte budget.
    #[must_use]
    pub fn byte_budget(&self) -> usize {
        self.byte_budget
    }

    /// Returns the current total byte usage of stored snapshots.
    #[must_use]
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Returns the number of distinct prompt entries stored in the arena.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.arena.iter().filter(|e| e.is_some()).count()
    }

    /// Estimates the byte size of a [`KvState`] snapshot at `m` positions.
    ///
    /// Per position: `2 · (n_head_kv · head_dim) · 8 · n_layer` bytes.
    /// Temporal: O(1); spatial: O(1).
    #[must_use]
    pub fn estimate_snapshot_bytes(kv: &KvState, m: usize) -> usize {
        let kv_out = kv.n_head_kv() * kv.head_dim();
        // 2 (k and v) * kv_out elements * 8 bytes (i64) * n_layer * m positions
        2 * kv_out * 8 * kv.n_layer() * m
    }

    /// Probes for the longest cached prefix matching `tokens`.
    ///
    /// Checks boundaries `m = (len / B) * B` down to `B`.
    /// On a 64-bit key hit in the table, verifies the full token sequence `t[0..m]`.
    /// Returns `Some((m, &KvState))` on confirmed match, or `None`. A confirmed hit counts as a use
    /// and advances the entry's recency tick.
    ///
    /// # Complexity
    ///
    /// `O(len / B)` Table lookups. Each lookup is `O(1)` in probe width, followed by
    /// an `O(m)` full-key token comparison on hit, plus one `O(1)` tick update on a confirmed hit.
    #[must_use]
    pub fn longest_prefix(&mut self, tokens: &[u32]) -> Option<(usize, &KvState)> {
        let n = tokens.len();
        if n < self.block_size {
            return None;
        }

        let max_blocks = n / self.block_size;
        let mut b = max_blocks;
        let mut hit: Option<(usize, usize)> = None;
        while b > 0 {
            let m = b * self.block_size;
            let prefix = &tokens[..m];
            let hash = self.hasher.hash_tokens(prefix);
            let key = PrefixKey(hash);

            if let Some(b_ref) = *self.table.get(&key) {
                if let Some(Some(entry)) = self.arena.get_mut(b_ref.entry_id as usize) {
                    // Full-key identity confirmation: hash alone is never trusted!
                    if entry.tokens.len() >= m && &entry.tokens[..m] == prefix {
                        let tick = self.tick.wrapping_add(1);
                        entry.last_used = tick;
                        self.tick = tick;
                        hit = Some((m, b_ref.entry_id as usize));
                        break;
                    }
                }
            }
            b -= 1;
        }

        let (m, slot) = hit?;
        Some((m, &self.arena[slot].as_ref()?.state))
    }

    /// Returns the arena slot of the least-recently-used entry, if any.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(entries)` full scan of the arena; spatial: `O(1)`.
    fn lru_victim(&self) -> Option<usize> {
        let mut victim: Option<(usize, u64)> = None;
        for (slot, entry) in self.arena.iter().enumerate() {
            if let Some(e) = entry {
                if victim.is_none_or(|(_, t)| e.last_used < t) {
                    victim = Some((slot, e.last_used));
                }
            }
        }
        victim.map(|(slot, _)| slot)
    }

    /// Evicts the entry in `slot`, dropping its state and releasing its bytes.
    ///
    /// The entry's boundary keys stay in the [`Table`] (it exposes no removal); a later probe
    /// resolves them through full-key token confirmation, so a stale key is a miss unless a reused
    /// slot holds an entry with the same prefix tokens.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(bytes)` drop of the [`KvState`]; spatial: `O(1)` beyond the freed entry.
    fn evict_entry(&mut self, slot: usize) {
        if let Some(entry) = self.arena[slot].take() {
            self.current_bytes = self.current_bytes.saturating_sub(entry.byte_size);
            self.free_slots.push(slot);
        }
    }

    /// Inserts a snapshot of `kv` corresponding to `tokens` into the cache.
    ///
    /// Stores exactly ONE [`KvState`] in the prompt arena and indexes each block boundary
    /// `b · B ≤ m` into Apeiron's Table as `(entry_id, b · B)`.
    ///
    /// When the snapshot would exceed `byte_budget`, whole prompt entries are evicted in
    /// least-recently-used order until it fits.
    ///
    /// # Errors
    ///
    /// - Returns [`PrefixCacheError::BudgetExceeded`] when the snapshot alone exceeds `byte_budget`,
    ///   leaving the cache completely unchanged.
    /// - Returns [`PrefixCacheError::TableStorage`] if the underlying table refuses storage.
    /// - Returns [`PrefixCacheError::CapacityExceeded`] if boundary length exceeds `u32::MAX`.
    ///
    /// # Complexity
    ///
    /// `O(m)` where `m = tokens.len()` for hashing and token cloning, plus `O(m / B)`
    /// Table boundary insertions each taking `O(1)` time, plus eviction at `O(entries)` per evicted
    /// entry (`O(entries²)` when the whole cache is recycled).
    pub fn insert(&mut self, tokens: &[u32], kv: &KvState) -> Result<(), PrefixCacheError> {
        let m = tokens.len();
        if m == 0 {
            return Ok(());
        }

        let needed = Self::estimate_snapshot_bytes(kv, m);

        // A snapshot that cannot fit even in an empty cache is refused before any mutation.
        if needed > self.byte_budget {
            return Err(PrefixCacheError::BudgetExceeded {
                needed,
                current: self.current_bytes,
                budget: self.byte_budget,
            });
        }

        // Evict least-recently-used whole entries until the snapshot fits. The loop terminates
        // because `needed ≤ byte_budget`: after evicting every entry the free space is the budget.
        while needed > self.byte_budget.saturating_sub(self.current_bytes) {
            let Some(victim) = self.lru_victim() else {
                break;
            };
            self.evict_entry(victim);
        }

        let slot = if let Some(free_slot) = self.free_slots.pop() {
            free_slot
        } else {
            let s = self.arena.len();
            self.arena.push(None);
            s
        };

        let entry_id = u32::try_from(slot).map_err(|_| PrefixCacheError::CapacityExceeded)?;

        let tick = self.tick.wrapping_add(1);
        let entry = PromptEntry {
            tokens: tokens.to_vec(),
            state: kv.clone(),
            byte_size: needed,
            last_used: tick,
        };

        self.arena[slot] = Some(entry);
        self.current_bytes += needed;
        self.tick = tick;

        let max_b = m / self.block_size;
        for b in 1..=max_b {
            let boundary_m = b * self.block_size;
            let m_u32 =
                u32::try_from(boundary_m).map_err(|_| PrefixCacheError::CapacityExceeded)?;
            let prefix = &tokens[..boundary_m];
            let hash = self.hasher.hash_tokens(prefix);
            let key = PrefixKey(hash);
            let b_ref = BoundaryRef { entry_id, m: m_u32 };

            if let Err(err) = self.table.insert_with(&BoundaryMerge, key, Some(b_ref)) {
                // Roll back allocation on table failure to preserve unmodified cache invariant
                self.arena[slot] = None;
                self.current_bytes = self.current_bytes.saturating_sub(needed);
                self.free_slots.push(slot);
                return Err(PrefixCacheError::TableStorage(err));
            }
        }

        Ok(())
    }
}

/// Truncates a cloned [`KvState`] to `m` occupied positions through [`KvState::truncate`];
/// `false` when a layer holds fewer than `m` positions.
///
/// # Complexity
///
/// `O(n_layer)`, as [`KvState::truncate`].
pub fn truncate_kv_state(kv: &mut KvState, m: usize) -> bool {
    kv.truncate(m).is_ok()
}

/// Prefills a sequence of tokens with prefix caching.
///
/// 1. Queries `cache` for the longest cached prefix of `tokens` strictly less than `tokens.len()`.
/// 2. If a prefix of length `m` is found, clones its `KvState` as the starting state,
///    and truncates it in-place to `m` positions via [`truncate_kv_state`].
///    Otherwise, creates a fresh cold `KvState` with capacity `tokens.len() + extra_capacity`.
/// 3. Steps through the remaining tokens `tokens[m..]`, advancing the KV state.
/// 4. If `insert_boundaries` is true, stores the completed prompt snapshot into the cache.
/// 5. Returns the final `KvState` and the last `StepResult`.
///
/// # Errors
///
/// - Returns [`PrefixCacheError::EmptyPrompt`] if `tokens` is empty.
/// - Returns [`PrefixCacheError::CapacityExceeded`] if cache snapshot capacity is smaller
///   than needed.
/// - Returns [`PrefixCacheError::Step`] on decode step failure.
///
/// # Complexity
///
/// `O(n / B)` lookup time + `O((n - m) · cost_per_step)`. When `m == n - 1`, only 1 decode step
/// is executed.
pub fn prefill_with_cache<H: PrefixHasher>(
    engine: &Engine,
    cache: &mut PrefixCache<H>,
    tokens: &[u32],
    extra_capacity: usize,
    insert_boundaries: bool,
) -> Result<(KvState, StepResult), PrefixCacheError> {
    let n = tokens.len();
    if n == 0 {
        return Err(PrefixCacheError::EmptyPrompt);
    }

    let total_cap = n
        .checked_add(extra_capacity)
        .ok_or(PrefixCacheError::Step(StepError::LengthOverflow))?;

    // Step 1: probe cache for longest prefix strictly less than n tokens so that at least
    // the last token tokens[n - 1] is stepped to produce the step result (logits / prediction).
    let prefix_bound = &tokens[..n.saturating_sub(1)];
    let (start_pos, mut kv) = if let Some((m, cached_state)) = cache.longest_prefix(prefix_bound) {
        if cached_state.cap() < total_cap {
            return Err(PrefixCacheError::CapacityExceeded);
        }
        let mut kv = cached_state.clone();
        if let Ok(occ) = kv.occupied(0) {
            if occ > m && !truncate_kv_state(&mut kv, m) {
                return Err(PrefixCacheError::CapacityExceeded);
            }
        }
        (m, kv)
    } else {
        (
            0,
            engine
                .decode_state(total_cap)
                .map_err(PrefixCacheError::Step)?,
        )
    };

    // Step 2: prefill remaining tokens
    let mut last_result = None;
    let mut pos = start_pos;

    while pos < n {
        let token_id = tokens[pos];
        let step_res = engine
            .step_with_kv(&mut kv, token_id)
            .map_err(PrefixCacheError::Step)?;
        last_result = Some(step_res);
        pos += 1;
    }

    // Step 3: if insert_boundaries is true, insert the prompt snapshot into cache
    if insert_boundaries && n >= cache.block_size() && start_pos < n {
        let _ = cache.insert(tokens, &kv);
    }

    let Some(result) = last_result else {
        return Err(PrefixCacheError::EmptyPrompt);
    };

    Ok((kv, result))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
