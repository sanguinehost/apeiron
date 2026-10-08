//! Forward-only finetuning adapter (`FoldAdapter`).
//!
//! Subsumed from `../fold/modelfold/src/train.rs` @ `d497161f` (`pub mod finetune`)
//! and `../spinor-ttl/src/spinor_ttl/finetune.py` @ `a05bb1b` (`FoldAdapter`).
//! Count storage is the crate's exact-key map [`Table`] (`Stored` keys, full-key equality on
//! equal coordinates). The vendored `FreqMap` is not admissible here: its `freq(k) = k.coord()`
//! is a lossy 64-bit fold, so distinct structured keys with equal coordinates merge.
//!
//! # Mathematical Foundations and Citations
//!
//! - **Kneser–Ney Floor Refinement**:
//!   One absolute-discount lane over an arbitrary base distribution conserves probability mass
//!   exactly, remains nonnegative, and lands domain evidence regardless of base floor.
//!   Refines `Sanguine.MLPrimitivesBridge.kn_floor_refinement`
//!   (`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:385`, grade: tested).
//! - **Chart Addition and Grouping-Free Merge**:
//!   Accumulation of observation counts (`fold`) and adapter merge (`merge`) form commutative
//!   monoids over $\mathbb{Z}$, with exact unlearning (`unfold`) as the additive group inverse:
//!   $(M + A) - A = M$.
//!   Refines `Sanguine.MLPrimitivesBridge.osFuse_is_add`
//!   (`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:174`, grade: tested) and
//!   `Sanguine.MLPrimitivesBridge.osFuse_grouping_free`
//!   (`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:182`, grade: tested).
//! - **Doorway Participation-Ratio Cut**:
//!   Context update fires only when continuation collapse satisfies $\text{PR} \le 2.0$
//!   ($\text{PR} = (\sum c)^2 / \sum c^2$), preserving diffuse out-of-domain distributions.
#![allow(
    clippy::doc_markdown,
    reason = "mathematical formulas, variable names and citations in doc comments"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "context orders k <= 4 fit within u8 by construction"
)]
#![allow(
    clippy::cast_precision_loss,
    reason = "integer counts converted to f64 for probability normalisation"
)]
#![allow(
    clippy::must_use_candidate,
    reason = "idiomatic state querying and inference API"
)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::freq::Table;
use crate::hashtrinity::Entry;
use crate::learn::countlm::kn_discounts_cg;

/// Maximum context order supported by the compact lossless representation ($k \le 4$).
pub const MAX_ORDER: usize = 4;

/// Default context order for finetuning adapter ($k = 3$).
pub const DEFAULT_ORDER: usize = 3;

/// Derived anti-degeneration budget: participation ratio threshold $\text{PR} \le 2.0$.
pub const DOORWAY: f64 = 2.0;

/// Error type for fold adapter construction, configuration, and inference operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldAdaptError {
    /// Context order must be in `1..=MAX_ORDER` (1..=4).
    InvalidOrder(usize),
    /// Base distribution must not be empty.
    EmptyBase,
    /// Context orders must match for adapter merge.
    OrderMismatch {
        /// Expected order (self).
        expected: usize,
        /// Actual order (other).
        actual: usize,
    },
}

impl std::fmt::Display for FoldAdaptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOrder(k) => {
                write!(f, "invalid context order {k}: must be in 1..={MAX_ORDER}")
            }
            Self::EmptyBase => write!(f, "base distribution must not be empty"),
            Self::OrderMismatch { expected, actual } => {
                write!(
                    f,
                    "context order mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for FoldAdaptError {}

/// Context-order, packed-context, and next-token key: `(order, packed_context, token)`.
pub type Key = (u8, u128, u32);

/// Flat count state: `(order, packed-context, token) -> occurrence count`.
/// Stored in the crate's exact-key map [`Table`] (the `Stored` key store), whose full-key
/// equality keeps distinct `(order, packed-context, token)` keys apart even when their 64-bit
/// coordinates coincide — unlike the lossy frequency fold `FreqMap`.
type Flat = Table<Key, i64>;

/// Context key: `(order, packed-context)`.
type CtxKey = (u8, u128);

/// Per-context continuation head: `(order, packed-context) -> arena index of the most recently
/// appended continuation`. Exact-keyed so the continuation lists of distinct contexts (for example
/// `(a,b,c)` and its reversal) never merge even when their 64-bit coordinates coincide.
type CtxHead = Table<CtxKey, u32>;

/// Continuation arena, one node per live `(order, context, token)` key: `(token, prev)` where
/// `prev` is the arena index of the older node that follows in the same context's list, `u32::MAX`
/// at the oldest. An unlinked node joins a free list threaded through the same `prev` field and is
/// reused by the next append, so the arena holds at most the peak live-key count, not every key
/// ever folded. A context's list is walked newest-to-oldest through `prev` and reversed to
/// recover insertion order. Threading the list through the arena replaces the per-context
/// `Vec<u32>`: a new key still costs one `ctx_head` probe, but its insert is a sequential 8-byte
/// arena append instead of a per-context heap allocation and a pointer dereference.
type Arena = Vec<(u32, u32)>;

/// The empty-list arena index (the `prev` sentinel at a context's oldest node).
const NIL: u32 = u32::MAX;

/// A context's continuations in insertion order with their live counts, built once per context
/// between mutations: `(token, count)` pairs read contiguously, in the summation order `refine`
/// depends on for bit-identity. Shared through [`Rc`] so a cache hit does not copy the list.
type Continuations = Rc<Vec<(u32, i64)>>;

/// Pack up to 4 token ids into a `u128`, oldest id highest.
///
/// # Domain
///
/// `ctx.len() <= MAX_ORDER` (up to 4 tokens).
///
/// # Complexity
///
/// Temporal: $O(1)$ operations ($\text{ctx.len()} \le 4$).
/// Spatial: $O(1)$ words.
#[inline]
#[must_use]
pub fn pack(ctx: &[u32]) -> u128 {
    ctx.iter().fold(0u128, |p, &id| (p << 32) | u128::from(id))
}

/// Forward-only finetuning adapter over token contexts.
///
/// Holds n-gram occurrence counts over `u32` token contexts of order $1 \le k \le 4$.
/// Modifies base model predictions using Kneser–Ney count refinement.
#[derive(Clone)]
pub struct FoldAdapter {
    flat: Flat,
    keys: Vec<Key>,
    ctx_head: CtxHead,
    arena: Arena,
    /// Head of the free-node list inside `arena` (`NIL` when empty).
    free: u32,
    /// Contiguous continuation cache: `(order, packed-context) -> insertion-ordered `(token,
    /// count)` pairs`. `refine_gated` walks each context's `prev` chain once to build it, then reads
    /// it contiguously on every later refine of the same context, so scoring does not pay one cache
    /// miss per arena node per query. Built lazily (only the queried contexts), and cleared by every
    /// mutation (`fold`, `unfold`, `merge`), so it never serves a stale count. Interior mutability
    /// keeps `refine` on `&self`.
    cont_cache: RefCell<Table<CtxKey, Continuations>>,
    k: usize,
}

impl Default for FoldAdapter {
    fn default() -> Self {
        Self::new(DEFAULT_ORDER).expect("default order is within 1..=MAX_ORDER")
    }
}

impl FoldAdapter {
    /// Create a new fold adapter of context order $k$ ($1 \le k \le 4$).
    ///
    /// # Domain
    ///
    /// $1 \le k \le \text{MAX\_ORDER}$. Returns [`FoldAdaptError::InvalidOrder`] if out of domain.
    ///
    /// # Errors
    ///
    /// Returns [`FoldAdaptError::InvalidOrder`] if `k == 0` or `k > MAX_ORDER`.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(1)$ operations.
    /// Spatial: $O(1)$ allocations.
    pub fn new(k: usize) -> Result<Self, FoldAdaptError> {
        if k == 0 || k > MAX_ORDER {
            return Err(FoldAdaptError::InvalidOrder(k));
        }
        Ok(FoldAdapter {
            flat: Table::with_capacity(16),
            keys: Vec::new(),
            ctx_head: Table::with_capacity(16),
            arena: Vec::new(),
            free: NIL,
            cont_cache: RefCell::new(Table::with_capacity(16)),
            k,
        })
    }

    /// Pre-sized adapter for an estimated key capacity.
    ///
    /// # Domain
    ///
    /// $1 \le k \le \text{MAX\_ORDER}$. Returns [`FoldAdaptError::InvalidOrder`] if out of domain.
    ///
    /// # Errors
    ///
    /// Returns [`FoldAdaptError::InvalidOrder`] if `k == 0` or `k > MAX_ORDER`.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(1)$ table initialization.
    /// Spatial: allocates table of capacity in [`Table`].
    pub fn with_capacity(k: usize, cap: usize) -> Result<Self, FoldAdaptError> {
        if k == 0 || k > MAX_ORDER {
            return Err(FoldAdaptError::InvalidOrder(k));
        }
        Ok(FoldAdapter {
            flat: Table::with_capacity(cap),
            keys: Vec::new(),
            ctx_head: Table::with_capacity(cap),
            arena: Vec::with_capacity(cap),
            free: NIL,
            cont_cache: RefCell::new(Table::with_capacity(16)),
            k,
        })
    }

    /// Context order $k$ of this adapter.
    #[inline]
    pub fn order(&self) -> usize {
        self.k
    }

    /// Whether this adapter holds no live keys.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Number of active distinct `(order, context, token)` entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Slice of active `(order, packed_context, token)` keys.
    #[inline]
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    /// Finetune: fold the token stream into the count table in one pass.
    ///
    /// Accumulates occurrence counts across all context orders $1 \le o \le \min(k, i)$
    /// for each position $i \in 1..\text{ids.len()}$.
    ///
    /// # Domain
    ///
    /// Any token slice `ids`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(N \cdot k)$ operations, where $N = \text{ids.len()}$ and $k = \text{self.k} \le 4$.
    /// Derivation: outer loop iterates $i \in 1..N$; inner loop iterates $o \in 1..\min(k, i)$.
    /// Each step performs $O(1)$ pack and one $O(1)$ [`Table::entry`] probe (increment, or insert on
    /// a new key). A new key additionally pushes to `keys`, one `ctx_head` probe, and one sequential
    /// arena append, amortized $O(1)$. No per-step allocation and no per-context `Vec`. The
    /// continuation cache is cleared first (its entries bind the pre-fold counts).
    /// Spatial: $O(U)$ words where $U \le N \cdot k$ is the number of distinct n-grams observed.
    pub fn fold(&mut self, ids: &[u32]) {
        // Every position adds at most one distinct `(order, context, token)` key, so `ids.len() * k`
        // bounds the new keys this fold can add. Half that bound is reserved once, against the live
        // keys already held: the `foldadapt_race` stream holds 2,050,121 distinct keys against a
        // bound of 5,116,416 (40%), and the halved reserve was faster (0.29 s to 0.23 s together
        // with the arena; the split is unmeasured). A stream whose distinct keys exceed the half
        // grows once more; the counts are unaffected.
        *self.cont_cache.borrow_mut() = Table::with_capacity(16);
        let bound = ids.len().saturating_mul(self.k);
        let additional = bound.saturating_sub(self.flat.len());
        self.flat.reserve(additional / 2);
        self.keys.reserve(additional / 2);
        self.ctx_head.reserve(additional / 2);
        self.arena.reserve(additional / 2);
        self.fold_signed(ids, 1);
    }

    /// Exact unlearning: subtract the same token stream (the group inverse over $\mathbb{Z}$).
    ///
    /// Leaves no trace behind: entries whose counts return to zero are completely removed
    /// from key tracking and sparse context indices.
    ///
    /// # Domain
    ///
    /// Any token slice `ids`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(N \cdot k + K)$ operations, where $N = \text{ids.len()}$, $k = \text{self.k} \le 4$,
    /// and $K = \text{self.keys.len()}$.
    /// Derivation: loops over tokens in $\Theta(N \cdot k)$ subtracting counts, followed by $O(K)$ cleanup
    /// removing zero-count keys and unlinking empty continuations from their context's arena list.
    /// The continuation cache is cleared first (its entries bind the pre-unfold counts).
    /// Spatial: $O(1)$ auxiliary space.
    pub fn unfold(&mut self, ids: &[u32]) {
        *self.cont_cache.borrow_mut() = Table::with_capacity(16);
        self.fold_signed(ids, -1);
    }

    /// Append `token` as the newest node of `ctx_key`'s continuation list.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(1)$ amortized (one `ctx_head` probe, then a free-node pop or a sequential
    /// arena append).
    /// Spatial: one 8-byte arena node amortized, reused from the free list when one is available.
    #[inline]
    fn ctx_append(&mut self, ctx_key: CtxKey, token: u32) {
        let new_idx = if self.free == NIL {
            self.arena.push((token, NIL));
            (self.arena.len() - 1) as u32
        } else {
            let idx = self.free;
            self.free = self.arena[idx as usize].1;
            idx
        };
        let prev = match self.ctx_head.entry(ctx_key) {
            Entry::Vacant(e) => {
                e.insert(new_idx);
                NIL
            }
            Entry::Occupied(mut e) => std::mem::replace(e.get_mut(), new_idx),
        };
        self.arena[new_idx as usize] = (token, prev);
    }

    /// Unlink `token` from `ctx_key`'s continuation list; drop the context entry when its list
    /// becomes empty.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(L)$ where $L$ is `ctx_key`'s continuation-list length (linear walk from the
    /// newest node through the `prev` links).
    /// Spatial: $O(1)$.
    fn ctx_unlink(&mut self, ctx_key: CtxKey, token: u32) {
        let Some(&head) = self.ctx_head.get(&ctx_key) else {
            return;
        };
        // Walk newest-to-oldest; `newer` is the node whose `prev` points at `cur` (`NIL` at the head).
        let mut cur = head;
        let mut newer = NIL;
        loop {
            let (t, prev) = self.arena[cur as usize];
            if t == token {
                if newer == NIL {
                    if prev == NIL {
                        self.ctx_head.remove(&ctx_key);
                    } else if let Some(h) = self.ctx_head.get_mut(&ctx_key) {
                        *h = prev;
                    }
                } else {
                    self.arena[newer as usize].1 = prev;
                }
                self.arena[cur as usize].1 = self.free;
                self.free = cur;
                return;
            }
            if prev == NIL {
                return;
            }
            newer = cur;
            cur = prev;
        }
    }

    /// Shared signed fold/unfold core: `sign > 0` accumulates counts and threads each new
    /// continuation onto its context's arena list; `sign < 0` subtracts and unlinks continuations
    /// whose count reaches zero, then drops zero-count keys from `keys`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(N \cdot k + U + R + K)$ operations, where $N = \text{ids.len()}$,
    /// $k = \text{self.k} \le 4$, $U$ is the number of new keys (one `flat` probe, one `ctx_head`
    /// probe and one arena append each), $R$ the number of continuations unlinked (an $O(L)$ walk of
    /// the context's list each) and $K = \text{self.keys.len()}$ on the unfold cleanup pass.
    /// Spatial: $O(U)$ arena nodes amortized; no other auxiliary allocation.
    fn fold_signed(&mut self, ids: &[u32], sign: i32) {
        let k = self.k;
        if ids.len() < 2 {
            return;
        }
        for i in 1..ids.len() {
            let next_id = ids[i];
            for o in 1..=k.min(i) {
                let p = pack(&ids[i - o..i]);
                let key = (o as u8, p, next_id);
                let ctx_key = (o as u8, p);
                if sign > 0 {
                    // One probe, one merge: increment through the entry face, or take the vacant
                    // slot. A pre-existing zero (left in `flat` by an unfold) is logically absent.
                    let was_absent = match self.flat.entry(key) {
                        Entry::Vacant(e) => {
                            e.insert(1);
                            true
                        }
                        Entry::Occupied(mut e) => {
                            let c = e.get_mut();
                            let absent = *c == 0;
                            *c += 1;
                            absent
                        }
                    };
                    if was_absent {
                        self.keys.push(key);
                        // A genuinely new `(order, context, token)` is absent from the context's
                        // continuation list (the zero-count/zero-token invariant), so no scan.
                        self.ctx_append(ctx_key, next_id);
                    }
                } else {
                    // One probe: decrement a live count; a count that hits zero unlinks its
                    // continuation from the context's arena list, with no per-call `Vec`.
                    let became_zero = match self.flat.entry(key) {
                        Entry::Vacant(_) => false,
                        Entry::Occupied(mut e) => {
                            let c = e.get_mut();
                            if *c > 0 {
                                *c -= 1;
                                *c == 0
                            } else {
                                false
                            }
                        }
                    };
                    if became_zero {
                        self.ctx_unlink(ctx_key, next_id);
                    }
                }
            }
        }
        if sign < 0 {
            self.keys
                .retain(|k| self.flat.get(k).copied().unwrap_or(0) != 0);
        }
    }

    /// Merge counts from another adapter into `self` (exact, order-free addition over $\mathbb{Z}$).
    ///
    /// Refines `Sanguine.MLPrimitivesBridge.osFuse_is_add` and
    /// `Sanguine.MLPrimitivesBridge.osFuse_grouping_free`.
    ///
    /// # Domain
    ///
    /// `self.k == other.k`. Returns [`FoldAdaptError::OrderMismatch`] if context orders differ.
    ///
    /// # Errors
    ///
    /// Returns [`FoldAdaptError::OrderMismatch`] if `self.k != other.k`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K + R \cdot L)$ operations where $K = \text{other.keys.len()}$ and each
    /// zero-count continuation is unlinked in $O(L)$ ($L$ = its context's list length).
    /// Derivation: single pass over `other.keys`; each key is looked up in `other.flat` in $O(1)$ time and added to `self.flat` in $O(1)$ time, with one `ctx_head` probe and one arena append on a new key; a second pass unlinks the merged-to-zero continuations. The continuation cache is cleared first (its entries bind the pre-merge counts).
    /// Spatial: up to $K$ new key words in `self.keys` and `self.flat`.
    pub fn merge(&mut self, other: &Self) -> Result<(), FoldAdaptError> {
        if self.k != other.k {
            return Err(FoldAdaptError::OrderMismatch {
                expected: self.k,
                actual: other.k,
            });
        }
        *self.cont_cache.borrow_mut() = Table::with_capacity(16);
        for &k in &other.keys {
            let Some(&c) = other.flat.get(&k) else {
                continue;
            };
            if c == 0 {
                continue;
            }
            // One probe per key: add through the entry face; a pre-existing zero is logically absent.
            let was_absent = match self.flat.entry(k) {
                Entry::Vacant(e) => {
                    e.insert(c);
                    true
                }
                Entry::Occupied(mut e) => {
                    let v = e.get_mut();
                    let absent = *v == 0;
                    *v += c;
                    absent
                }
            };
            if was_absent {
                self.keys.push(k);
                let ctx_key = (k.0, k.1);
                let next_id = k.2;
                // The key is newly live, so its continuation is absent from the list by invariant.
                self.ctx_append(ctx_key, next_id);
            }
        }
        self.keys
            .retain(|k| self.flat.get(k).copied().unwrap_or(0) != 0);
        for &k in &other.keys {
            let (o, p, next_id) = k;
            let c = self.flat.get(&k).copied().unwrap_or(0);
            if c == 0 {
                self.ctx_unlink((o, p), next_id);
            }
        }
        Ok(())
    }

    /// Return a new adapter with `other` merged into `self`.
    ///
    /// # Domain
    ///
    /// `self.k == other.k`. Returns [`FoldAdaptError::OrderMismatch`] if context orders differ.
    ///
    /// # Errors
    ///
    /// Returns [`FoldAdaptError::OrderMismatch`] if `self.k != other.k`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K_1 + K_2)$ operations where $K_1 = \text{self.keys.len()}$ and $K_2 = \text{other.keys.len()}$.
    /// Derivation: clones `self` ($O(K_1)$) and merges `other` ($O(K_2)$).
    /// Spatial: $O(K_1 + K_2)$ words for the merged adapter.
    pub fn merged(&self, other: &Self) -> Result<Self, FoldAdaptError> {
        let mut res = self.clone();
        res.merge(other)?;
        Ok(res)
    }

    /// Modified-KN absolute discounts derived from top-order counts-of-counts.
    ///
    /// Calls [`crate::learn::countlm::kn_discounts_cg`].
    ///
    /// # Domain
    ///
    /// Total.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K)$ operations where $K = \text{self.keys.len()}$.
    /// Derivation: single pass over `self.keys` counting occurrences at top order $k$.
    /// Spatial: $O(1)$ words.
    #[must_use]
    pub fn discounts(&self) -> (f64, f64, f64) {
        let (mut n1, mut n2, mut n3, mut n4) = (0u64, 0u64, 0u64, 0u64);
        for &k in &self.keys {
            if k.0 as usize == self.k {
                let c = self.flat.get(&k).copied().unwrap_or(0);
                match c {
                    1 => n1 += 1,
                    2 => n2 += 1,
                    3 => n3 += 1,
                    4 => n4 += 1,
                    _ => {}
                }
            }
        }
        kn_discounts_cg(n1, n2, n3, n4)
    }

    /// Refine a base distribution `base` at `ctx` using default doorway gating ([`DOORWAY`]).
    ///
    /// Applies domain evidence only when participation ratio $\text{PR} \le 2.0$.
    /// Refines `Sanguine.MLPrimitivesBridge.kn_floor_refinement`.
    ///
    /// # Domain
    ///
    /// Non-empty `base`. Returns [`FoldAdaptError::EmptyBase`] if `base.is_empty()`.
    ///
    /// # Errors
    ///
    /// Returns [`FoldAdaptError::EmptyBase`] if `base.is_empty()`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(V + k \cdot D)$ operations where $V = \text{base.len()}$, $k = \text{self.k} \le 4$,
    /// and $D \le V$ is maximum continuation count per queried context order.
    /// Derivation: clones `base` ($O(V)$), then queries at most $k$ orders. Each order evaluates $D$
    /// continuation counts and scales/adds into $p$ in $O(V)$ operations. A queried context's
    /// continuation list is built once and cached until the next mutation, so repeated refines at
    /// the same context amortize the $O(D)$ build to zero.
    /// Spatial: allocates vector of length $V$.
    pub fn refine(&self, base: &[f64], ctx: &[u32]) -> Result<Vec<f64>, FoldAdaptError> {
        let d = self.discounts();
        self.refine_gated(base, ctx, d, DOORWAY)
    }

    /// Refine with pre-computed discounts and unconstrained participation ratio ($\text{PR} \le \infty$).
    ///
    /// # Domain
    ///
    /// Non-empty `base`. Returns [`FoldAdaptError::EmptyBase`] if `base.is_empty()`.
    ///
    /// # Errors
    ///
    /// Returns [`FoldAdaptError::EmptyBase`] if `base.is_empty()`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(V + k \cdot D)$ operations, with the $O(D)$ continuation-list build
    /// amortized across repeated refines at the same context by the cache.
    /// Spatial: allocates vector of length $V$.
    pub fn refine_with(
        &self,
        base: &[f64],
        ctx: &[u32],
        d: (f64, f64, f64),
    ) -> Result<Vec<f64>, FoldAdaptError> {
        self.refine_gated(base, ctx, d, f64::INFINITY)
    }

    /// Sparse refinement with explicit participation ratio ceiling `pr_max`.
    ///
    /// # Domain
    ///
    /// Non-empty `base`. Returns [`FoldAdaptError::EmptyBase`] if `base.is_empty()`.
    ///
    /// # Errors
    ///
    /// Returns [`FoldAdaptError::EmptyBase`] if `base.is_empty()`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(V + k \cdot D)$ operations, with the $O(D)$ continuation-list build
    /// amortized across repeated refines at the same context by the cache.
    /// Spatial: allocates vector of length $V$.
    pub fn refine_sparse(
        &self,
        base: &[f64],
        ctx: &[u32],
        d: (f64, f64, f64),
        pr_max: f64,
    ) -> Result<Vec<f64>, FoldAdaptError> {
        self.refine_gated(base, ctx, d, pr_max)
    }

    /// The continuations of `ctx_key` in insertion order with their live counts, served from the
    /// cache or built by one walk of `ctx_key`'s arena chain and cached until the next mutation.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(1)$ amortized on a cache hit; on a miss $O(D)$ (one `prev`-chain walk and one
    /// `flat` probe per node), where $D$ is the context's continuation-list length.
    /// Spatial: $O(D)$ for the built list, retained until the next mutation.
    fn continuations(&self, ctx_key: CtxKey) -> Continuations {
        if let Some(cached) = self.cont_cache.borrow().get(&ctx_key) {
            return Rc::clone(cached);
        }
        let (o, p_ctx) = ctx_key;
        let mut cnts: Vec<(u32, i64)> = Vec::new();
        if let Some(&head) = self.ctx_head.get(&ctx_key) {
            // Walk newest-to-oldest, then reverse to the insertion order the summation in
            // `refine_gated` depends on for bit-identity.
            let mut idx = head;
            while idx != NIL {
                let (next_id, prev) = self.arena[idx as usize];
                let key = (o, p_ctx, next_id);
                if let Some(&c) = self.flat.get(&key) {
                    if c > 0 {
                        cnts.push((next_id, c));
                    }
                }
                idx = prev;
            }
            cnts.reverse();
        }
        let cnts = Rc::new(cnts);
        self.cont_cache
            .borrow_mut()
            .insert(ctx_key, Rc::clone(&cnts));
        cnts
    }

    fn refine_gated(
        &self,
        base: &[f64],
        ctx: &[u32],
        d: (f64, f64, f64),
        pr_max: f64,
    ) -> Result<Vec<f64>, FoldAdaptError> {
        if base.is_empty() {
            return Err(FoldAdaptError::EmptyBase);
        }
        let mut p = base.to_vec();
        let n = ctx.len();
        for o in 1..=self.k {
            if o > n {
                continue;
            }
            let p_ctx = pack(&ctx[n - o..n]);
            let ctx_key = (o as u8, p_ctx);
            // Insertion-ordered, contiguous, from the cache after the first query of this context.
            let cnts = self.continuations(ctx_key);
            if cnts.is_empty() {
                continue;
            }
            let (mut tot, mut sumsq, mut nn1, mut nn2, mut nn3) =
                (0i64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &(_, c) in cnts.iter() {
                tot += c;
                let fc = c as f64;
                sumsq += fc * fc;
                match c {
                    1 => nn1 += 1.0,
                    2 => nn2 += 1.0,
                    _ => nn3 += 1.0,
                }
            }
            if tot <= 0 {
                continue;
            }
            let ftot = tot as f64;
            let pr = ftot * ftot / sumsq.max(1e-30);
            if pr > pr_max {
                continue;
            }
            let gamma = (d.0 * nn1 + d.1 * nn2 + d.2 * nn3) / ftot;
            for pj in &mut p {
                *pj *= gamma;
            }
            for &(j, c) in cnts.iter() {
                if (j as usize) < p.len() {
                    let dc = match c {
                        1 => d.0,
                        2 => d.1,
                        _ => d.2,
                    };
                    p[j as usize] += (c as f64 - dc).max(0.0) / ftot;
                }
            }
        }
        Ok(p)
    }

    /// Total count and number of non-zero entries: `(total_count, non_zero_entries)`.
    ///
    /// # Domain
    ///
    /// Total.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K)$ operations where $K = \text{self.keys.len()}$.
    /// Derivation: single pass over `self.keys`.
    /// Spatial: $O(1)$ words.
    pub fn signature(&self) -> (i64, usize) {
        let mut tot = 0i64;
        let mut ne = 0usize;
        for &k in &self.keys {
            if let Some(&c) = self.flat.get(&k) {
                if c != 0 {
                    tot += c;
                    ne += 1;
                }
            }
        }
        (tot, ne)
    }

    /// Exact non-zero count ledger sorted by key for bit-for-bit equivalence comparison.
    ///
    /// # Domain
    ///
    /// Total.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(M \log M)$ comparisons where $M \le \text{self.keys.len()}$ is the non-zero entry count.
    /// Derivation: loops over `self.keys` ($O(K)$), then sorts $M$ tuples ($O(M \log M)$).
    /// Spatial: allocates vector of $M$ `((order, packed_ctx, token), count)` tuples.
    pub fn count_state(&self) -> Vec<(Key, i64)> {
        let mut state: Vec<(Key, i64)> = self
            .keys
            .iter()
            .filter_map(|&k| {
                let c = self.flat.get(&k).copied().unwrap_or(0);
                if c != 0 {
                    Some((k, c))
                } else {
                    None
                }
            })
            .collect();
        state.sort_unstable_by_key(|a| a.0);
        state
    }

    /// Counts-of-counts $[N_1, N_2, N_3, N_4]$ across all context keys.
    ///
    /// # Domain
    ///
    /// Total.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K)$ operations where $K = \text{self.keys.len()}$.
    /// Derivation: single pass over `self.keys`.
    /// Spatial: $O(1)$ words.
    #[must_use]
    pub fn count_of_counts(&self) -> [u64; 4] {
        let mut n = [0u64; 4];
        for &k in &self.keys {
            let c = self.flat.get(&k).copied().unwrap_or(0);
            match c {
                1 => n[0] += 1,
                2 => n[1] += 1,
                3 => n[2] += 1,
                4 => n[3] += 1,
                _ => {}
            }
        }
        n
    }
}

impl PartialEq for FoldAdapter {
    fn eq(&self, other: &Self) -> bool {
        self.k == other.k && self.count_state() == other.count_state()
    }
}

impl Eq for FoldAdapter {}

impl std::fmt::Debug for FoldAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoldAdapter")
            .field("order", &self.k)
            .field("keys_count", &self.keys.len())
            .field("signature", &self.signature())
            .finish_non_exhaustive()
    }
}
