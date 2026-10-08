//! Kneser–Ney byte count language model (`CountLM`).
//!
//! Subsumed from `../fold/modelfold/src/countlm.rs` @ `d497161f` and `../spinor-ttl/src/spinor_ttl/countlm.py` @ `a05bb1b`.
//! Count storage is the vendored counting face [`FreqMap`], keyed by an **injective** `u64` word:
//! `(order, packed-context, label)` is packed into disjoint bit fields ([`pack_key`]) so no two
//! distinct n-gram keys share a coordinate. A structured tuple key would instead be addressed by the
//! lossy tuple `Coord` fold (`src/hashtrinity/algebra.rs:465-476`), which merges distinct keys on a
//! coordinate clash. `u64::coord` is the identity and `u64::INJECTIVE` is `true`
//! (`src/hashtrinity/algebra.rs:60-66`), so `FreqMap<u64, …>` — whose `freq(k) = k.coord()`
//! (`src/hashtrinity/freq.rs:1257-1259`) — is exact on the packed words.
//!
//! # Mathematical Foundations
//!
//! - **Kneser–Ney Smoothing**: Kneser & Ney (1995), *Improved backing-off for M-gram language modeling*;
//!   Chen & Goodman (1998), *An empirical study of smoothing techniques for language modeling*.
//!   Discounts $D_k$ are derived from corpus counts-of-counts $N_1 \dots N_4$ via Good–Turing matching.
//!   Interpolation refines lower-order continuation distributions with backoff weight $\gamma = \sum_c D_c N_c / \text{total}$.
//!   Refines `KneserNey.kn_normalizes` (`proof/corpus/Substrate/Algebra/Neural/KneserNey.lean:98`).
//! - **Order-Free Merge and Exact Retention**: Memory state is a commutative group over $\mathbb{Z}$.
//!   Observation accumulation (`fold`) and model merge (`merge`) are commutative monoids.
//!   Unlearning (`unfold`) is the exact group inverse: $(M + A) - A = M$.
//!   Refines `MirasCountLM.hebb_comm` (`proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:89`),
//!   `MirasCountLM.forget_hebb` (`proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:101`), and
//!   `MirasCountLM.countlm_is_exact_miras` (`proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:138`).
#![allow(
    clippy::doc_markdown,
    reason = "mathematical formulas, variable names and citations in doc comments"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "context orders k <= 4 and byte labels 0..256 fit within u8 and u16 by construction"
)]
#![allow(
    clippy::cast_precision_loss,
    reason = "integer counts and stream lengths converted to f64 for probability normalisation and entropy"
)]
#![allow(
    clippy::must_use_candidate,
    reason = "idiomatic state querying and inference API"
)]

use crate::freq::FreqMap;

/// Maximum context order supported by the compact lossless representation ($k \le 4$).
pub const MAX_ORDER: usize = 4;

/// Bit width of the order field. Orders `1..=MAX_ORDER` (`MAX_ORDER = 4`) fit in 3 bits.
const ORDER_BITS: u32 = 3;
/// Bit width of the packed-context field. `pack_canon` folds one byte per `(p << 8) | g[b]` step
/// with `g[b] < 256`, and [`fold_range`](CountLM::fold_range) never packs more than `MAX_ORDER = 4`
/// bytes, so a packed context occupies at most `4 bytes × 8 bits = 32` bits.
const CTX_BITS: u32 = 32;
/// Bit width of the byte-label field. A label in `0..256` needs 9 bits.
const LABEL_BITS: u32 = 9;
/// Shift of the packed-context field in the packed key.
const CTX_SHIFT: u32 = ORDER_BITS;
/// Shift of the byte-label field in the packed key. `3 + 32 + 9 = 44 ≤ 64`.
const LABEL_SHIFT: u32 = ORDER_BITS + CTX_BITS;
/// Mask of the packed-context field.
const CTX_MASK: u64 = (1u64 << CTX_BITS) - 1;
/// Mask of the byte-label field.
const LABEL_MASK: u64 = (1u64 << LABEL_BITS) - 1;
/// Mask of the order field.
const ORDER_MASK: u64 = (1u64 << ORDER_BITS) - 1;

/// Pack an n-gram key `(order, packed-context, label)` into one `u64` with disjoint bit fields, so
/// the packing is injective on the used domain (`order ≤ 7`, `context < 2^32`, `label < 2^9`).
///
/// The 44-bit word is `< 2^44`, so it never equals a reserved occupancy marker (`EMPTY = u64::MAX`,
/// `TOMB = u64::MAX - 1`, `src/hashtrinity/mod.rs:229,286`): every packed key lands in the primary
/// lanes of [`FreqMap`] and its coordinate (the word itself, `u64::coord`) is exact.
///
/// # Panics
///
/// Panics if `order > 7`, `ctx ≥ 2^32` or `label ≥ 2^9`: an out-of-domain field would otherwise be
/// truncated and alias another key.
#[inline]
#[must_use]
pub fn pack_key(order: u8, ctx: u64, label: u16) -> u64 {
    assert!(
        u64::from(order) <= ORDER_MASK && ctx <= CTX_MASK && u64::from(label) <= LABEL_MASK,
        "pack_key: field out of domain"
    );
    (u64::from(order) & ORDER_MASK)
        | ((ctx & CTX_MASK) << CTX_SHIFT)
        | ((u64::from(label) & LABEL_MASK) << LABEL_SHIFT)
}

/// Inverse of [`pack_key`]: recover `(order, packed-context, label)` exactly.
#[inline]
#[must_use]
pub fn unpack_key(key: u64) -> (u8, u64, u16) {
    (
        (key & ORDER_MASK) as u8,
        (key >> CTX_SHIFT) & CTX_MASK,
        ((key >> LABEL_SHIFT) & LABEL_MASK) as u16,
    )
}

/// Pack a `(order, packed-context)` distribution key into one `u64`, the low 35 bits of
/// [`pack_key`]'s layout. Injective on the used domain by the same disjoint-field argument.
///
/// # Panics
///
/// Panics if `order > 7` or `ctx ≥ 2^32`.
#[inline]
#[must_use]
pub fn pack_ctx(order: u8, ctx: u64) -> u64 {
    assert!(
        u64::from(order) <= ORDER_MASK && ctx <= CTX_MASK,
        "pack_ctx: field out of domain"
    );
    (u64::from(order) & ORDER_MASK) | ((ctx & CTX_MASK) << CTX_SHIFT)
}

/// Flat count state: `pack_key(order, packed-context, byte) -> occurrence count`.
///
/// Keyed by an injective `u64` word rather than a `(u8, u64, u16)` tuple, whose tuple `Coord` fold is
/// lossy and merges distinct keys on a coordinate clash.
type Flat = FreqMap<u64, i64>;

/// Sparse distribution for reader inference:
/// `pack_ctx(order, packed-context) -> [(byte, count)]`.
type Dist = FreqMap<u64, Vec<(u16, i32)>>;

/// The identity canonicalizer mapping each byte to itself.
pub const CANON_IDENTITY: [u8; 256] = {
    let mut c = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        c[i] = i as u8;
        i += 1;
    }
    c
};

/// Pack up to 8 context bytes into a `u64`, oldest byte highest.
/// Lossless bijection for contexts of length $\le 8$.
///
/// `CountLM` caps a context at `MAX_ORDER = 4` bytes ([`fold_range`](CountLM::fold_range)), so a
/// packed context produced here occupies at most `4 × 8 = 32` bits (`CTX_BITS`), the width
/// [`pack_key`] reserves for it.
#[inline]
fn pack_canon(ctx: &[u8], g: &[u8; 256]) -> u64 {
    ctx.iter()
        .fold(0u64, |p, &b| (p << 8) | u64::from(g[b as usize]))
}

/// Drop the oldest byte from an order-$o$ packed context, yielding the order-$(o-1)$ continuation context.
#[inline]
fn drop_oldest(p: u64, o: u8) -> u64 {
    if o <= 1 {
        0
    } else {
        p & ((1u64 << (8 * (u32::from(o) - 1))) - 1)
    }
}

/// Merge new `(byte, count)` pairs into an existing sparse distribution list.
fn bump(mut old: Vec<(u16, i32)>, new: Vec<(u16, i32)>) -> Vec<(u16, i32)> {
    for (b, c) in new {
        match old.iter_mut().find(|(bb, _)| *bb == b) {
            Some(e) => e.1 += c,
            None => old.push((b, c)),
        }
    }
    old
}

/// Floor a probability without laundering NaN into a finite number.
#[inline]
fn floor_keep_nan(x: f64, floor: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else {
        x.max(floor)
    }
}

/// Chen & Goodman (1998) modified Kneser–Ney absolute discounts from counts-of-counts $N_1 \dots N_4$.
///
/// # Domain
///
/// Inputs: counts-of-counts $n_1, n_2, n_3, n_4 \in \mathbb{N}$.
/// Output: absolute discounts $(D_1, D_2, D_3)$ with $0 \le D_c \le c$.
///
/// # Complexity
///
/// Temporal: $O(1)$ arithmetic operations.
/// Spatial: $O(1)$ words.
#[must_use]
pub fn kn_discounts_cg(n1: u64, n2: u64, n3: u64, n4: u64) -> (f64, f64, f64) {
    if n1 == 0 || n2 == 0 {
        return (0.75, 0.75, 0.75);
    }
    let (f1, f2, f3, f4) = (n1 as f64, n2 as f64, n3 as f64, n4 as f64);
    let y = f1 / (f1 + 2.0 * f2);
    let d1 = 1.0 - 2.0 * y * f2 / f1;
    let d2 = if n3 == 0 { d1 } else { 2.0 - 3.0 * y * f3 / f2 };
    let d3 = if n4 == 0 { d2 } else { 3.0 - 4.0 * y * f4 / f3 };
    (
        d1.clamp(0.0, 0.95),
        d2.clamp(0.0, 1.95),
        d3.clamp(0.0, 2.95),
    )
}

/// One filtration step: discounted higher-order counts plus reserved backoff mass onto distribution `p`.
///
/// Refines `KneserNey.kn_normalizes`:
/// $$p_{new}(w) = \gamma \cdot p_{old}(w) + \frac{\max(c(w) - D_c, 0)}{\sum_v c(v)}$$
/// where $\gamma = \frac{\sum_c D_c N_c}{\sum_v c(v)}$.
fn kn_refine(d: (f64, f64, f64), p: &mut [f64; 256], t: &[(u16, i32)]) {
    if t.is_empty() {
        return;
    }
    let (mut tot, mut nn1, mut nn2, mut nn3) = (0i64, 0f64, 0f64, 0f64);
    for &(_, c) in t {
        tot += i64::from(c);
        match c {
            1 => nn1 += 1.0,
            2 => nn2 += 1.0,
            _ => nn3 += 1.0,
        }
    }
    let ftot = tot as f64;
    let gamma = (d.0 * nn1 + d.1 * nn2 + d.2 * nn3) / ftot;
    for pj in p.iter_mut() {
        *pj *= gamma;
    }
    for &(j, c) in t {
        if (j as usize) < 256 {
            let dc = match c {
                1 => d.0,
                2 => d.1,
                _ => d.2,
            };
            p[j as usize] += (f64::from(c) - dc).max(0.0) / ftot;
        }
    }
}

/// Mutable count language model holding n-gram occurrence counts over byte contexts.
///
/// Counts form an exact group over $\mathbb{Z}$, enabling order-free merge and exact unlearning.
#[derive(Clone)]
pub struct CountLM {
    flat: Flat,
    keys: Vec<u64>,
    order: usize,
    canon: [u8; 256],
}

impl CountLM {
    /// Create a new model of given context order ($1 \le \text{order} \le 4$).
    ///
    /// # Domain
    ///
    /// $1 \le \text{order} \le \text{MAX\_ORDER}$. Panics if out of domain.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(1)$ operations.
    /// Spatial: $O(1)$ allocations.
    pub fn new(order: usize) -> Self {
        Self::with_canon(order, CANON_IDENTITY)
    }

    /// Pre-sized model for estimated corpus volume in bytes.
    ///
    /// # Domain
    ///
    /// $1 \le \text{order} \le \text{MAX\_ORDER}$; `bytes` is estimated key capacity.
    ///
    /// # Panics
    ///
    /// Panics if `order` is not in `1..=MAX_ORDER`.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(1)$ table initialization.
    /// Spatial: allocates table of capacity $\approx 1.5 \cdot \text{bytes}$ slots in [`FreqMap`], $O(\text{bytes})$ words.
    pub fn with_capacity(order: usize, bytes: usize) -> Self {
        assert!(
            (1..=MAX_ORDER).contains(&order),
            "order 1..=4 supported by byte context pack"
        );
        CountLM {
            flat: FreqMap::with_capacity(bytes),
            keys: Vec::new(),
            order,
            canon: CANON_IDENTITY,
        }
    }

    /// Model with resonance canonicalizer `g` mapping surface bytes to pivot bytes.
    ///
    /// # Domain
    ///
    /// $1 \le \text{order} \le \text{MAX\_ORDER}$.
    ///
    /// # Panics
    ///
    /// Panics if `order` is not in `1..=MAX_ORDER`.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(1)$ operations.
    /// Spatial: $O(1)$ allocations.
    pub fn with_canon(order: usize, canon: [u8; 256]) -> Self {
        assert!(
            (1..=MAX_ORDER).contains(&order),
            "order 1..=4 supported by byte context pack"
        );
        CountLM {
            flat: FreqMap::with_capacity(16),
            keys: Vec::new(),
            order,
            canon,
        }
    }

    /// Context order $k$ of this model.
    #[inline]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Train by counting suffix-to-byte occurrences over `bytes` in one pass.
    ///
    /// # Domain
    ///
    /// Any byte slice `bytes`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(N \cdot k)$ operations, where $N = \text{bytes.len()}$ and $k = \text{self.order} \le 4$.
    /// Derivation: outer loop iterates $i \in 1..N$; inner loop iterates $o \in 1..\min(k, i)$.
    /// Each step performs $O(1)$ pack, $O(1)$ [`FreqMap::get`], $O(1)$ [`FreqMap::add`], and amortized $O(1)$ key tracking.
    /// Spatial: $O(U)$ words where $U \le N \cdot k$ is the number of distinct n-grams observed.
    pub fn fold(&mut self, bytes: &[u8]) {
        self.fold_range(bytes, 0, bytes.len());
    }

    /// Fold positions $[start, end)$ of `bytes`.
    ///
    /// Context reaches into the full slice $[i - o \dots i)$.
    ///
    /// # Domain
    ///
    /// $0 \le start \le end \le bytes.len()$.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta((end - start) \cdot k)$ operations where $k \le 4$.
    /// Spatial: $O(U)$ distinct n-gram words added.
    pub fn fold_range(&mut self, bytes: &[u8], start: usize, end: usize) {
        let k = self.order;
        for i in start.max(1)..end {
            let label = u16::from(bytes[i]);
            for o in 1..=k.min(i) {
                let p = pack_canon(&bytes[i - o..i], &self.canon);
                let key = pack_key(o as u8, p, label);
                if self.flat.get(&key).is_none() {
                    self.keys.push(key);
                }
                self.flat.add(&key, 1);
            }
        }
    }

    /// Merge counts from another model into `self` (exact, order-free addition over $\mathbb{Z}$).
    ///
    /// Refines `MirasCountLM.hebb_comm`.
    ///
    /// # Domain
    ///
    /// `self.order == other.order`. Panics if orders mismatch.
    ///
    /// # Panics
    ///
    /// Panics if `self.order != other.order`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K)$ operations where $K = \text{other.keys.len()}$.
    /// Derivation: single pass over `other.keys`; each key is looked up in `other.flat` in $O(1)$ time and added to `self.flat` in $O(1)$ time.
    /// Spatial: up to $K$ new key words in `self.keys` and `self.flat`.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(self.order, other.order, "orders must match");
        for &k in &other.keys {
            if let Some(&c) = other.flat.get(&k) {
                if c != 0 {
                    if self.flat.get(&k).is_none() {
                        self.keys.push(k);
                    }
                    self.flat.add(&k, c);
                }
            }
        }
    }

    /// Unfold counts of another model from `self` (exact unlearning, the $\mathbb{Z}$-group inverse).
    ///
    /// Refines `MirasCountLM.forget_hebb`.
    ///
    /// # Domain
    ///
    /// `self.order == other.order`. Panics if orders mismatch.
    ///
    /// # Panics
    ///
    /// Panics if `self.order != other.order`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K)$ operations where $K = \text{other.keys.len()}$.
    /// Derivation: single pass over `other.keys`; each count is subtracted from `self.flat` in $O(1)$ time.
    /// Spatial: $O(1)$ auxiliary space beyond possible residue key tracking.
    pub fn unfold(&mut self, other: &Self) {
        assert_eq!(self.order, other.order, "orders must match");
        for &k in &other.keys {
            if let Some(&c) = other.flat.get(&k) {
                if c != 0 {
                    if self.flat.get(&k).is_none() {
                        self.keys.push(k);
                    }
                    self.flat.add(&k, -c);
                }
            }
        }
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
    /// Spatial: $O(1)$ words.
    pub fn signature(&self) -> (i64, usize) {
        let (mut tot, mut ne) = (0i64, 0usize);
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
    /// Each packed key is unpacked back to its `(order, packed-context, byte)` triple.
    ///
    /// # Domain
    ///
    /// Total.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(M \log M)$ comparisons where $M \le \text{self.keys.len()}$ is the non-zero entry count.
    /// Derivation: loops over `self.keys` ($O(K)$), then sorts $M$ tuples ($O(M \log M)$).
    /// Spatial: allocates vector of $M$ `((order, packed_ctx, byte), count)` tuples.
    pub fn count_state(&self) -> Vec<((u8, u64, u16), i64)> {
        let mut state: Vec<((u8, u64, u16), i64)> = self
            .keys
            .iter()
            .filter_map(|&k| {
                let c = self.flat.get(&k).copied().unwrap_or(0);
                if c != 0 {
                    Some((unpack_key(k), c))
                } else {
                    None
                }
            })
            .collect();
        state.sort_unstable_by_key(|a| a.0);
        state
    }

    /// Counts-of-counts $[N_1, N_2, N_3, N_4]$ estimating the Chen–Goodman discounts.
    ///
    /// # Domain
    ///
    /// Total.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(S)$ operations where $S$ is the number of live entries in [`FreqMap`].
    /// Derivation: single pass over `self.flat.values()`.
    /// Spatial: $O(1)$ words.
    #[must_use]
    pub fn count_of_counts(&self) -> [u64; 4] {
        let mut n = [0u64; 4];
        for &c in self.flat.values() {
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

    /// Freeze the mutable count table into an immutable inference [`Reader`].
    ///
    /// Regroups occurrence counts into per-context distributions, builds continuation counts,
    /// computes the unigram floor, and estimates discounts.
    ///
    /// # Domain
    ///
    /// Total.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K)$ operations where $K = \text{self.keys.len()}$.
    /// Derivation: single loop over `self.keys`; each non-zero entry performs $O(1)$ insertion into `dist` and `cont`.
    /// Spatial: allocates `Reader` holding $O(K)$ entries across `dist` and `cont`.
    pub fn reader(&self) -> Reader {
        let cap = self.keys.len().max(16);
        let mut dist: Dist = FreqMap::with_capacity(cap);
        let mut cont: Dist = FreqMap::with_capacity(cap);
        let mut uni = [0u64; 256];
        let (mut n1, mut n2, mut n3, mut n4) = (0u64, 0u64, 0u64, 0u64);

        for &k in &self.keys {
            let (o, p, b) = unpack_key(k);
            let c = match self.flat.get(&k) {
                Some(&c_val) if c_val > 0 => c_val,
                _ => continue,
            };

            match c {
                1 => n1 += 1,
                2 => n2 += 1,
                3 => n3 += 1,
                4 => n4 += 1,
                _ => {}
            }

            dist.insert_with(&pack_ctx(o, p), vec![(b, c as i32)], |new, old| {
                bump(old, new)
            });
            if o == 1 {
                uni[b as usize] += 1;
            } else if (o as usize) <= MAX_ORDER {
                cont.insert_with(
                    &pack_ctx(o - 1, drop_oldest(p, o)),
                    vec![(b, 1)],
                    |new, old| bump(old, new),
                );
            }
        }

        let tot: u64 = uni.iter().sum();
        let denom = (tot + 256) as f64;
        let floor = std::array::from_fn(|j| (uni[j] + 1) as f64 / denom);

        Reader {
            dist,
            cont,
            floor,
            d: kn_discounts_cg(n1, n2, n3, n4),
            order: self.order,
            canon: self.canon,
        }
    }

    /// Next-byte probability distribution $P(\cdot \mid context) \in \mathbb{R}^{256}$ given trailing `context`.
    ///
    /// Refines `KneserNey.kn_normalizes`: $\sum_{w=0}^{255} P(w \mid context) = 1.0$.
    ///
    /// # Domain
    ///
    /// Any byte slice `context`.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(K + k \cdot 256)$ where $K = \text{self.keys.len()}$ to freeze reader and query $k \le 4$ lanes.
    /// Spatial: allocates `[f64; 256]`.
    pub fn dist(&self, context: &[u8]) -> [f64; 256] {
        self.reader().dist(context)
    }

    /// Average bits per byte (log-loss / cross-entropy) on held-out `stream`:
    /// $$\frac{1}{N-1} \sum_{i=1}^{N-1} -\log_2 P(\text{stream}[i] \mid \text{stream}[\max(0, i-k) \dots i))$$
    ///
    /// # Domain
    ///
    /// Any non-empty byte slice `stream`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(K + N \cdot k \cdot 256)$ operations where $K = \text{self.keys.len()}$, $N = \text{stream.len()}$, $k \le 4$.
    /// Derivation: builds `Reader` once in $\Theta(K)$, then evaluates each position in $O(k \cdot 256)$.
    /// Spatial: $O(K)$ words for frozen reader.
    pub fn bits_per_byte(&self, stream: &[u8]) -> f64 {
        self.reader().bits_per_byte(stream)
    }
}

impl std::iter::Sum for CountLM {
    fn sum<I: Iterator<Item = CountLM>>(iter: I) -> CountLM {
        let mut it = iter;
        match it.next() {
            Some(mut acc) => {
                for m in it {
                    acc.merge(&m);
                }
                acc
            }
            None => CountLM::new(MAX_ORDER),
        }
    }
}

/// Frozen Kneser–Ney inference reader over counts.
#[derive(Clone)]
pub struct Reader {
    dist: Dist,
    cont: Dist,
    floor: [f64; 256],
    d: (f64, f64, f64),
    order: usize,
    canon: [u8; 256],
}

impl Reader {
    /// Frozen absolute discounts $(D_1, D_2, D_3)$ applied by this reader.
    #[inline]
    #[must_use]
    pub fn discounts(&self) -> (f64, f64, f64) {
        self.d
    }

    /// Next-byte probability distribution given context: unigram floor refined up continuation lanes $1 \dots k-1$ and top lane $k$.
    ///
    /// Refines `KneserNey.kn_normalizes`: $\sum_{w=0}^{255} P(w \mid context) = 1.0$.
    ///
    /// # Domain
    ///
    /// Any byte slice `ctx`.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(k \cdot 256)$ operations where $k = \text{self.order} \le 4$.
    /// Derivation: iterates $o \in 1..k$; looks up context in `cont` or `dist` in $O(1)$ time, and calls `kn_refine` over at most 256 entries.
    /// Spatial: $O(1)$ allocations (`[f64; 256]`).
    pub fn dist(&self, ctx: &[u8]) -> [f64; 256] {
        let mut p = self.floor;
        let (n, k) = (ctx.len(), self.order);
        let kn = k.min(MAX_ORDER);

        for o in 1..kn {
            if o <= n {
                if let Some(t) = self
                    .cont
                    .get(&pack_ctx(o as u8, pack_canon(&ctx[n - o..], &self.canon)))
                {
                    kn_refine(self.d, &mut p, t);
                }
            }
        }

        if kn <= n {
            if let Some(t) = self
                .dist
                .get(&pack_ctx(kn as u8, pack_canon(&ctx[n - kn..], &self.canon)))
            {
                kn_refine(self.d, &mut p, t);
            }
        }

        p
    }

    /// Most likely next byte $\text{argmax}_{w} P(w \mid context)$.
    ///
    /// # Domain
    ///
    /// Total.
    ///
    /// # Complexity
    ///
    /// Temporal: $O(k \cdot 256)$ operations.
    /// Spatial: $O(1)$.
    pub fn argmax(&self, ctx: &[u8]) -> u8 {
        let p = self.dist(ctx);
        (0..256).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(0) as u8
    }

    /// Average bits per byte on held-out text: $\frac{1}{N-1} \sum_{i=1}^{N-1} -\log_2 P(\text{bytes}[i] \mid \text{bytes}[\max(0, i-k) \dots i))$.
    ///
    /// # Domain
    ///
    /// Any non-empty byte slice `bytes`.
    ///
    /// # Complexity
    ///
    /// Temporal: $\Theta(N \cdot k \cdot 256)$ operations where $N = \text{bytes.len()}$, $k \le 4$.
    /// Derivation: iterates $i \in 1..N$, calling `dist` at each position.
    /// Spatial: $O(1)$ auxiliary allocations.
    pub fn bits_per_byte(&self, bytes: &[u8]) -> f64 {
        if bytes.len() <= 1 {
            return 0.0;
        }
        let mut bits = 0.0;
        for i in 1..bytes.len() {
            let p = self.dist(&bytes[i.saturating_sub(self.order)..i]);
            let s: f64 = floor_keep_nan(p.iter().sum::<f64>(), 1e-30);
            bits += -floor_keep_nan(p[bytes[i] as usize] / s, 1e-30).log2();
        }
        bits / (bytes.len() - 1) as f64
    }
}
