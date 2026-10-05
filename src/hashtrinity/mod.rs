// Row: `apeiron-extract-insert-4-7NjO1ZbPTGkqjZ3afaKeu`.
// Row: `apeiron-vendor-purge-B7P4Q0hmPKDkYftLsbxHIm`: only the table core, its extraction reach and webfold's three names remain.
//! hashtrinity — a `K → V` table where the key is its own address.
//!
//! Not a wrapper over `std::HashMap`. One flat power-of-two array, one insert, and four merge
//! functions that pick what kind of table you get. No random seed, so the layout is deterministic.
//!
//! ## Feature flags
//!
//! As a *module* of apeiron these names resolve against apeiron's own feature table, so the
//! upstream feature names are renamed with an `ht-` prefix; they are declared as expected cfg
//! values in `Cargo.toml`, not as features, because as features `--all-features` would switch the
//! whole vendored surface on. Only the retained `codec` module (with its `compress`/`syntax`
//! submodules and its `hex` syntax bridge) still carries such a gate; the table below documents
//! upstream's names.
//!
//! | Feature | Deps | What it adds |
//! |---------|------|-------------|
//! | `codec` | — | Zero-copy wire format ([`codec::web`]), universal Term, integer column codec |
//! | `compress` | — | Context-mixing entropy coder ([`codec::cm`]) — subsumes gzip -9 |
//! | `syntax` | — | Collapsed syntax AST ([`codec::syntax`]) — JSON/BAML/CSV/TOON/POML/YAML surfaces |
//!
//! # One structure, four merges
//!
//! [`Table::insert_with`] is the only mutator: it merges the new value with whatever was already
//! there. The merge function is the *only* difference between a set, a map, a counter, and a graph:
//!
//! | table | methods                     | merge      | absent reads as |
//! |-------|-----------------------------|------------|-----------------|
//! | Set   | [`Table::mark`]/`has`       | `\|\|`     | `false`         |
//! | Map   | [`Table::put`]/`get`        | last write | `None`          |
//! | Count | [`Table::add`]/`count`      | `+`        | `0`             |
//! | Graph | [`Table::push`]/`neighbors` | append     | `[]`            |
//!
//! An absent key reads as the merge's identity, not a sentinel. Commutative merges (Set, Count) are
//! order-free, so their tables can be split across threads and recombined ([`Table::fuse`]); the
//! directional merges (Map, Graph) cannot.
//!
//! [`Table`] keeps the key (exact). Two keyless variants trade that for speed and footprint:
//! [`CoordMap`] for coordinate keys (the address regenerates the key), and [`FreqMap`] for opaque
//! keys it routes by fold (`2⁻⁶⁴` false-positive on non-members) — dedup / count / index.
//!
//! # Why it's fast
//!
//! The key is its own address when it can be. An integer key smaller than the array indexes directly
//! — no hash, no collision, one probe. Larger or byte keys fold to a 64-bit coordinate via `k·PHI`
//! (Fibonacci hashing) and take the top `bits` as the home slot; a collision re-addresses by the next
//! window of `k·PHI`. Multiply-by-odd is a bijection on 2⁶⁴, so distinct keys always separate within
//! `⌈64/bits⌉` steps — no two keys can alias forever. Coordinate keys are HashDOS-immune by
//! construction (`home(c) = c`: the only key that can occupy slot `s` is `s`). On the FOLD path the
//! immunity is measured, not proved: separation bounds aliasing, not probe length, and
//! `AggregationBound.descent_can_be_wholly_blocked` refutes the worst-case reading — `PHI` is public,
//! so `PHI⁻¹` aims one crafted key at each slot a victim would try. `tests/hashdos.rs` is the red
//! team and carries the numbers. No SipHash tax.
//!
//! # Which face, and when — see `FACES.md`
//!
//! Two choices, every time: the **keystore** and the **face**. **Bulk work is parallel by default.**
//!
//! **Map (the trinity `x = B/U`), by what you need from a key:**
//!
//! | map | face | when | confirm |
//! |-----|------|------|---------|
//! | [`FreqMap`] | **NB** | join / count / dedup / cache / index — the analytical work | trusts the fold (`2^-64` false-positive) — the group probe |
//! | [`CoordMap`] | **B** | dense *coordinate* keys you must resolve EXACTLY, worst-probe-1 | the address round-trips (regenerates the key) |
//! | [`Table`] `Stored` | **B/U** | opaque keys you must resolve exactly OR read back | stores + compares the key |
//!
//! Default to **`FreqMap`** for the analytical faces (GROUP BY, JOIN, DISTINCT, counters): it is the
//! proven group-probe engine (`GroupProbe.lean` — uniform fold, 16-wide tag glance miss filter, bounded
//! terminal sweep), and `2^-64` is "never" on those workloads. Reach for `CoordMap`/`Table` only when a
//! `2^-64` false positive is unacceptable or you must recover the key.
//!
//! **Operation shape** (on any of them):
//!
//! | you have          | you want                | face |
//! |-------------------|-------------------------|------|
//! | one key           | its value               | [`get`](Table::get) / [`FreqMap::get`] |
//! | a stream of keys  | each value (materialize)| [`get_all`](Table::get_all) (threaded) |
//! | a stream of keys  | each value (visit)      | [`get_each`](Table::get_each) (pipelined) |
//! | a stream of keys  | one aggregate           | [`get_reduce`](Table::get_reduce) (threaded, fused) |
//! | the whole table   | one aggregate / argmin  | [`encircle`](Table::encircle) / [`encircle_floor`](Table::encircle_floor) |
//! | a stream of rows  | a per-key table (GROUP BY) | [`FreqMap::count_par`] (two-level parallel, group probe) |
//! | one row at a time | update a key            | [`add`](Table::add) / [`FreqMap::add`] |
//!
//! **Anti-patterns** (each produced a false "we lose"): `get_all(keys)…sum()` (materialize-then-fold) —
//! use [`get_reduce`](Table::get_reduce); a serial `for k { add(k,1) }` (or the [`FoldExt`] builders) as a *bulk* GROUP BY —
//! use [`FreqMap::count_par`]; `CoordMap` for a miss-heavy join — use [`FreqMap`] (its uniform fold has
//! no identity-home block); benchmarking any bulk pass single-threaded. If you're about to report a loss,
//! you have probably picked the wrong face — check the parallel group-probe one first. Full map: `FACES.md`.
//!
//! ```ignore
//! use apeiron::hashtrinity::FoldExt;
//! let freq = ["a", "b", "a"].iter().copied().count_by(|w| w);
//! assert_eq!(freq.count(&"a"), 2);
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

// ---------------------------------------------------------------------------
// Host-policy allowance (apeiron). Upstream's own `#![allow]` list is below and is left byte-
// identical to rev `4cc14471`; this block is separate because these five entries exist only to
// reconcile the vendored tree with *apeiron's* crate-level policy, which upstream never saw.
//
// `clippy::pedantic` + `clippy::style` — apeiron raises pedantic crate-wide through
// `[lints.clippy]` in `Cargo.toml`, so it reaches this tree too; `style` is on by default in every
// build. Upstream instead tuned against its own 18-entry list. Measured on clippy 0.1.96 against
// this tree, the host's two groups account for 38 distinct lints (28 pedantic + 10 style:
// `explicit_iter_loop` … `items_after_test_module`). Suppressing the two groups rather than 38
// names is what keeps a future upstream re-sync from re-deriving this. `dead_code` and
// `unused_imports` are separate because they are rustc lints, not clippy groups.
// `manual_is_multiple_of` and `manual_memcpy` are named one by one because they are in neither
// group — `manual_is_multiple_of` is `complexity`, and allowing that whole group for two hits is a
// wider hammer than naming them. Measured: 6 and 2 sites respectively.
// `dead_code`/`unused_imports` — this is a whole upstream crate, imported whole at a pinned
// revision so the count face can be read against its real source. Only `FreqMap` is wired
// (`src/freq.rs`); the rest is unreachable from this crate by construction, not by oversight, and
// deleting it would break the re-sync identity. Scoped to this module: it silences nothing outside
// `src/hashtrinity/`.
// `clippy::allow_attributes_without_reason` — apeiron requires a reason on every `#[allow]`. The 19
// bare `#[allow]`s here are upstream's, in files this task does not edit; requiring a reason would
// mean editing vendored source, so the lint is silenced for the subtree instead.
// ---------------------------------------------------------------------------
#![allow(
    clippy::pedantic,
    clippy::style,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    dead_code,
    unused_imports,
    clippy::allow_attributes_without_reason
)]

// The truncating/wrapping casts are the mechanism, not a bug: a fold *is* a `u64`→slot truncation and
// the coordinate math wraps mod 2⁶⁴ on purpose. The doc/style lints below are noise for this crate.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::must_use_candidate,
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::inline_always,
    clippy::wildcard_imports,
    clippy::many_single_char_names,
    clippy::cast_ptr_alignment
)]

use std::mem::MaybeUninit;

// codec is a secondary, non-default feature (the folded-in cerealize surface); its own doc pass is
// deferred, so the crate-wide missing_docs lint is relaxed for it rather than warn on internals.
#[cfg(feature = "ht-codec")]
#[allow(missing_docs)]
pub mod codec;

/// The `hex` masquerade — encode/decode hex strings, the same public API as the real crate.
/// Zero dependencies; see the module doc for what's ported.
#[allow(missing_docs)]
pub mod hex;

/// The `base64` masquerade — the same `Engine` API as the real crate. Zero dependencies; see the
/// module doc for what's ported.
#[allow(missing_docs)]
pub mod base64;

pub mod seed;
pub mod tregex;
pub use seed::{CoordSeed, PHI_INV};

// The ML primitives (emit·carrier·fold — simt/simt_nn/gpu, gptq/seal, foldtrain/foldnet/ridgefold/
// finetune/countlm) and the serving surface were carved out of this core into the `modelfold` and
// `servefold` crates. hashtrinity is the K→V Table and its faces, nothing model-specific.

pub mod par;

/// The std-shaped faces — `entry`/`get_mut`/`insert`/`retain`/iteration, and borrowed-key probes
/// ([`Locate`]) so `&str` reads a `Table<String, _>` and `(&str, &str)` reads a
/// `Table<(String, String), _>` with no key materialized. Purely additive; see [`face`].
pub mod face;
pub use face::{Absorb, Entry, IntoIter, Iter, Locate, OccupiedEntry, VacantEntry};

// ═══════════════════════════════════════════════════════════════════
// SUBSUMED — hashtrinity-crypto is now a module of the one crate:
//   hashtrinity-crypto   → crypto    (the standard hashes as the fold)
pub mod crypto;

// ── The two files this root was split into. ────────────────────────────────────────────────────
//
// PRIVATE modules, glob-re-exported, so every path stays exactly where it was: `Coord`, `Monoid`,
// `FreqMap` and friends are `hashtrinity::Coord`/`Monoid`/`FreqMap` as before, and no NEW public
// module path (`hashtrinity::algebra::…`) is created. `Table` itself, the sentinels, `is_live`,
// `prefix_masks` and `bits_for` stay HERE: `face` reaches them as `crate::hashtrinity::is_live` etc.,
// which is legal only because they are private items of the crate ROOT and those modules are its
// descendants. Moving them down one level would force `pub(crate)` on all of them.
//
// algebra — the trait vocabulary `Table<K, V, KS>` is parameterized by: how a key becomes an
//   address (`Coord`/`CoordInv`/`KeyStore`/`Stored`/`Regen`, the golden-128 byte fold) and how two
//   values merge (`Face`/`Species`/`keep`·`set`·`count`·`link`/`Monoid`/`FoldExt`).
// freq — the NB face: `FreqMap` and its `FreqLine` group-probe engine, a whole separate structure
//   (SwissTable control bytes, no key column) that shares nothing with `Table`'s probe.
mod side;
use side::SideStore;
mod algebra;
mod freq;
pub use algebra::*;
pub use freq::*;

/// Golden ratio · 2⁶⁴ — the Fibonacci-hashing multiplier. Odd, so `c ↦ c·PHI` is a bijection on 2⁶⁴
/// and loses nothing: splitting `c·PHI` into (top `bits` = home slot | low bits = residual) is
/// reversible. Public so the [`seed`] codec can store just the residual and regrow the coordinate.
pub const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

// ── The table: coordinates + values + (for opaque keys) a tag line and a key store. ────────────

/// The empty-slot marker. A live coordinate never equals it: a key whose own coordinate is `EMPTY`
/// (or `TOMB`) is held out of band in the [`SideStore`], so `coord == EMPTY` unambiguously means
/// "empty".
const EMPTY: u64 = u64::MAX;

/// Linear-prefix width of the probe ([`Table::descend`]): 8 `u64` coordinates = one 64-byte cache
/// line. The collision walk's first `LINEAR_G` steps are contiguous (`home + j`), so a miss or short
/// chain touches one coordinate line instead of one scattered line per radix step
/// (`RadixDescent.linear_prefix_chain_separates`).
const LINEAR_G: u32 = 8;

// The AVX2 prefix scan below consumes the `LINEAR_G` lanes as two 256-bit (4×u64) loads; if the
// prefix width ever changes, the scan must change with it, so pin the coupling at compile time.
const _: () = assert!(LINEAR_G == 8, "prefix_masks scans exactly 8 lanes");

/// The linear-prefix glance: compare the `LINEAR_G` contiguous coordinates `coords[home + k]` against
/// `coord` (a candidate hit) and against `EMPTY` (a chain terminator) in one branchless pass,
/// returning `(hits, empties)` as lane-bit masks. This is the read-side analogue of
/// [`group_tag_mask`], but on the coordinates *themselves*. The coordinate is identity over the live
/// space ([`coord_of`] is the raw coordinate; the reserved markers live in the [`SideStore`]), but two
/// distinct keys may share a coordinate, so a lane match is confirmed against the stored key (free on
/// `Regen`) before it is a hit. No second structure: the tag line THE CUT removed is not reintroduced;
/// the collision tail's own contiguous cache line is filtered in place, turning the mispredict-prone
/// scalar walk into one movemask. The hot home hit never calls this.
///
/// The prefix is addressed as a range of the coordinate slice (`home + k`), not as a raw pointer: the
/// scalar tail reads an in-bounds element, so the fallback needs no `unsafe` and the extraction sees no
/// raw-pointer dereference. The caller ensures `home + LINEAR_G ≤ coords.len()`.
#[inline]
fn prefix_masks(coords: &[u64], home: usize, coord: u64) -> (u32, u32) {
    // The shipping target is x86-64-v3, which mandates AVX2 (see `.cargo/config.toml`), so the common
    // build resolves the glance at *compile* time — a bare inlined `_mm256` pair on the hot read, no
    // per-call feature probe. Only a portable (sub-v3) build pays the runtime check, then falls back
    // to a branchless scalar loop the compiler still autovectorizes.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: caller guarantees `home + LINEAR_G ≤ len`; AVX2 is statically enabled here.
    return unsafe { prefix_masks_avx2(coords.as_ptr().add(home), coord) };

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: caller's read-length contract is exactly the two 4-lane loads avx2 does.
            return unsafe { prefix_masks_avx2(coords.as_ptr().add(home), coord) };
        }
        let (mut hits, mut empties) = (0u32, 0u32);
        // A `while`, not `for k in 0..LINEAR_G`: a range loop monomorphizes `Range<usize>::next`,
        // which resolves the `core::iter::range::Step` trait, and the extraction's pinned model
        // carries no `Step::forward_overflowing`. Bound, order, indexing and body are unchanged.
        let mut k = 0usize;
        while k < LINEAR_G as usize {
            let sc = coords[home + k];
            hits |= u32::from(sc == coord) << k;
            empties |= u32::from(sc == EMPTY) << k;
            k += 1;
        }
        (hits, empties)
    }
}

/// AVX2 body of [`prefix_masks`]: two `_mm256_cmpeq_epi64` + `_mm256_movemask_pd` pairs give one bit
/// per 64-bit lane, so the eight coordinates fold to two 4-bit masks each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prefix_masks_avx2(base: *const u64, coord: u64) -> (u32, u32) {
    use std::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_loadu_si256, _mm256_movemask_pd,
        _mm256_set1_epi64x,
    };
    // SAFETY: caller guarantees 8 readable lanes; the two loads cover lanes 0..4 and 4..8.
    let (lo, hi) = unsafe {
        (
            _mm256_loadu_si256(base.cast::<__m256i>()),
            _mm256_loadu_si256(base.add(4).cast::<__m256i>()),
        )
    };
    let key = _mm256_set1_epi64x(coord as i64);
    let emp = _mm256_set1_epi64x(EMPTY as i64);
    let mask = |v: __m256i, k: __m256i| -> u32 {
        _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v, k))) as u32
    };
    let hits = mask(lo, key) | (mask(hi, key) << 4);
    let empties = mask(lo, emp) | (mask(hi, emp) << 4);
    (hits, empties)
}

/// Deleted-slot marker (a tombstone), distinct from `EMPTY`. A removed slot's coordinate is set to
/// `TOMB`: it is not a live coordinate (a key whose own coordinate is `EMPTY` or `TOMB` is held out
/// of band in the [`SideStore`]) and not `EMPTY`, so every probe walks *through* it (chains stay
/// intact — see `probe`) while a resize reclaims it. Its value is uninitialized, exactly like an
/// `EMPTY` slot — the liveness invariant below excludes it. There is no tag line for a tombstone to
/// mark (see [`Table`]'s doc): the coordinate alone carries it.
const TOMB: u64 = EMPTY - 1;

/// A slot holds an initialized value iff its coordinate is neither `EMPTY` nor `TOMB`. This is the one
/// invariant every `unsafe` value read rests on; with deletions it replaces the old `!= EMPTY` test.
#[inline]
fn is_live(c: u64) -> bool {
    c != EMPTY && c != TOMB
}

/// Whether `c` is one of the two coordinates reserved as occupancy markers. Such a coordinate is
/// never written into [`Table`]'s `coords`; the key that owns it is held in the [`SideStore`].
#[inline]
fn is_sentinel(c: u64) -> bool {
    c == EMPTY || c == TOMB
}

/// `K -> V`. The slot's coordinate is the identity, the filter, and the occupancy flag at once:
/// `coord == wanted` is a hit (no key deref, no string compare) and `coord == EMPTY` is empty.
/// Values live in a parallel array so a probe can scan a tight coordinate line without dragging
/// payloads through cache; the key (when stored) lives in a third array, off the hot path.
///
/// No `Option` anywhere — the coordinate already says whether the slot is live, so each slot is
/// 8 bytes of coordinate, and value/key are touched only after a match. `Drop`/`Clone` walk the
/// live slots by hand. All the `unsafe` rests on ONE invariant, upheld at every write:
/// **`vals[i]` (and, for `Stored`, `keys[i]`) is initialized iff `coords[i]` is live**
/// ([`is_live`] — neither `EMPTY` nor a [`remove`](Self::remove)d slot's `TOMB` tombstone).
///
/// There is no tag line. The lossy 7-bit tag (SwissTable h2) is argued against by construction
/// (`CoordinateReadout.readout_beats_compare_by_construction`: a non-injective fingerprint cannot
/// confirm a hit; the full coordinate can, confirmed by the stored key at one compare) and its one
/// measured win — the opaque-key miss stream — is not a workload the framework runs. The probe is
/// the pure radix descent; its first two steps are the spinor readout (see [`Table::get`]).
///
/// The reserved coordinates (`EMPTY`/`TOMB`) never appear in `coords`: keys that own them live in
/// `side`, whose value is initialized iff `side.live[i]` (cleared only while a merge owns it).
pub struct Table<K, V, KS = Stored<K>>
where
    K: Coord,
    KS: KeyStore<K>,
{
    coords: Box<[u64]>, // identity + occupancy; coord == EMPTY ⇒ empty, value uninit
    vals: Box<[MaybeUninit<V>]>, // payloads; initialized exactly where coords[i] != EMPTY
    keys: KS,           // the identity store — a vector (Stored) or nothing (Regen)
    side: SideStore<K, V, KS>, // keys whose coordinate is a reserved marker (EMPTY/TOMB)
    bits: u32,
    len: usize,   // live entries (excludes tombstones)
    tombs: usize, // deleted-but-not-yet-reclaimed slots; cleared on resize
    _k: std::marker::PhantomData<K>,
}

/// Values AND stored keys are `MaybeUninit`, so drop exactly the live ones (`coord != EMPTY`).
/// No-op for types without destructors, the common integer case.
impl<K: Coord, V, KS: KeyStore<K>> Drop for Table<K, V, KS> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.coords.iter().zip(self.vals.iter_mut()) {
                if is_live(*c) {
                    // SAFETY: the invariant — live coord ⇒ value initialized.
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        // the key store rides the same invariant; internally gated on needs_drop::<K>
        let coords = std::mem::take(&mut self.coords);
        self.keys.drop_live(&coords);
        // Side entries: `coords` holds reserved markers (never `is_live`), so their keys are moved
        // out and dropped one by one, and only the `live` values are dropped.
        if std::mem::needs_drop::<V>() {
            for (v, live) in self.side.vals.iter_mut().zip(self.side.live.iter()) {
                if *live {
                    // SAFETY: `live[i]` ⇒ value initialized.
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        for i in 0..self.side.coords.len() {
            let c = self.side.coords[i];
            drop(self.side.keys.take_key(i, c));
        }
    }
}

/// Copy live values and keys through `Clone`, leave empty slots uninitialized.
impl<K: Coord + Clone, V: Clone, KS: KeyStore<K>> Clone for Table<K, V, KS> {
    fn clone(&self) -> Self {
        let coords = self.coords.clone();
        let vals = self
            .coords
            .iter()
            .zip(self.vals.iter())
            .map(|(c, v)| {
                if is_live(*c) {
                    // SAFETY: the invariant — live coord ⇒ value initialized.
                    MaybeUninit::new(unsafe { v.assume_init_ref() }.clone())
                } else {
                    MaybeUninit::uninit()
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        // Side entries: liveness rides `side.live` (their coordinates are reserved markers, so
        // `clone_live` needs an all-live mask), and only live values are cloned.
        let side_vals = self
            .side
            .vals
            .iter()
            .zip(self.side.live.iter())
            .map(|(v, live)| {
                if *live {
                    // SAFETY: `live[i]` ⇒ value initialized.
                    MaybeUninit::new(unsafe { v.assume_init_ref() }.clone())
                } else {
                    MaybeUninit::uninit()
                }
            })
            .collect::<Vec<_>>();
        let side_live_mask = vec![0u64; self.side.coords.len()]; // 0 is live (not EMPTY/TOMB)
        let side = SideStore {
            coords: self.side.coords.clone(),
            vals: side_vals,
            live: self.side.live.clone(),
            keys: self.side.keys.clone_live(&side_live_mask),
            cap: self.side.coords.len(),
            _k: std::marker::PhantomData,
        };
        Table {
            keys: self.keys.clone_live(&self.coords),
            coords,
            vals,
            side,
            bits: self.bits,
            len: self.len,
            tombs: self.tombs,
            _k: std::marker::PhantomData,
        }
    }
}

/// Show the shape (bits/len/capacity), not the raw slots.
impl<K: Coord, V, KS: KeyStore<K>> std::fmt::Debug for Table<K, V, KS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("bits", &self.bits)
            .field("len", &self.len)
            .field("capacity", &self.coords.len())
            .finish_non_exhaustive()
    }
}

/// A table for coordinate keys ([`CoordInv`], integers): the key regenerates from the address, so no
/// key array. Builds a big map faster than hashbrown (no key writes) and uses less memory.
pub type CoordMap<K, V> = Table<K, V, Regen<K>>;

impl<K: Eq + Coord, V, KS: KeyStore<K>> Default for Table<K, V, KS> {
    fn default() -> Self {
        Self::with_bits(3)
    }
}

/// THE one sizing: the grow law (7/8 headroom) applied at construction — the next power of two
/// holding `cap` keys under the grow threshold. No extra doubling, no second load factor:
/// `LoadFactorFrontier` proves there is no magic load factor, and the measured frontier
/// (BENCHMARK.md §Discharged inequalities) showed the old `+1` doubling losing the 1M build AND
/// the 1M hit stream (footprint beats probe length out of cache, `CoordinateReadout.footprint_mono`)
/// while winning only sub-millisecond in-cache hits. One law, stated once, same for both key species.
#[inline]
fn bits_for(cap: usize) -> u32 {
    let need = cap.saturating_mul(8) / 7 + 1;
    (usize::BITS - need.max(8).next_power_of_two().leading_zeros() - 1).max(3)
}

/// Bits for a slot line of at least `slots` (the next power of two ≥ `slots`). Used by [`FreqMap`],
/// which sizes its oversize directly — the oversize is its own headroom, no extra grow factor.
#[inline]
fn bits_for_slots(slots: usize) -> u32 {
    (usize::BITS - slots.max(8).next_power_of_two().leading_zeros() - 1).max(3)
}

/// Constructors for the default (opaque-key) table — keys stored. Concrete `Stored` so `Table::new()`
/// is unambiguous; for coordinate keys use [`CoordMap`]'s constructors, which store no key.
impl<K: Eq + Coord, V> Table<K, V, Stored<K>> {
    /// An empty table (8 slots; grows as needed).
    pub fn new() -> Self {
        Self::with_bits(3) // 8 slots
    }

    /// An empty table presized to hold `cap` keys without growing.
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

/// Constructors for a [`CoordMap`] — coordinate keys, no key array (regenerated from the address).
impl<K: CoordInv + Eq, V> Table<K, V, Regen<K>> {
    /// An empty coordinate map. The key regenerates from the address — nothing but the coordinate stored.
    pub fn coordinate() -> Self {
        Self::with_bits(3)
    }
    /// A coordinate map sized for `cap` keys — the same one sizing as [`Table::with_capacity`].
    pub fn coordinate_with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    fn with_bits(bits: u32) -> Self {
        let n = 1usize << bits;
        Table {
            coords: vec![EMPTY; n].into_boxed_slice(),
            vals: alloc_uninit_boxed(n),
            keys: KS::alloc(n),
            side: SideStore::empty(),
            bits,
            len: 0,
            tombs: 0,
            _k: std::marker::PhantomData,
        }
    }

    /// A key's coordinate — the identity on every coordinate. The two values reserved as occupancy
    /// markers (`EMPTY`, `TOMB`) are held out of band (see [`SideStore`]), so no non-reserved
    /// coordinate is remapped and `coord_of(U64::MAX)` is not folded onto another key's coordinate.
    #[inline]
    fn coord_of(k: &K) -> u64 {
        k.coord()
    }

    /// The home slot. A key that fits the array indexes directly (no hash); a larger key takes the
    /// top `bits` of `coord·PHI`.
    #[inline]
    fn home(&self, coord: u64) -> usize {
        let len = self.coords.len();
        if (coord as usize) < len {
            coord as usize
        } else {
            (coord.wrapping_mul(PHI) >> (64 - self.bits)) as usize
        }
    }

    /// The `j`-th probe slot for a collision — the shipped `seqLin` probe
    /// (`RadixDescent.linear_prefix_chain_separates`). Steps `1..LINEAR_G` are a **linear prefix**:
    /// `home + j`, contiguous slots that stay in the home cache line, so a miss (or a short chain)
    /// reads one 64-byte line of coordinates instead of one scattered line per radix step — the cache
    /// locality THE CUT gave up when it reverted to the pure scattered descent. Steps `≥ LINEAR_G`
    /// resume the MSB-first radix windows of `coord·PHI`, which break any linear cluster: distinct
    /// keys still separate within `LINEAR_G + D − 1` steps — a constant fixed by the word and the
    /// group width, never `n` (the prefix is an injective translation mod `2^b`; the digits after it
    /// are the proven walk re-indexed). That is ALIASING, not probe length: corrected 2026-08-06,
    /// this comment read "so flood immunity is preserved", and
    /// `AggregationBound.descent_can_be_wholly_blocked` refutes that step — `PHI⁻¹` aims one crafted
    /// key at each slot a victim would try. The worst case here is the terminal sweep below, one
    /// period of the slot space; the bounds in `tests/hashdos.rs` are measured. No tag line, no
    /// extra array — only the probe *order* changes.
    #[cold]
    fn descend(&self, coord: u64, j: u32) -> usize {
        let (bits, mask) = (self.bits, self.coords.len() - 1);
        if j < LINEAR_G {
            // seqLin, j < G: the home slot walked linearly, staying in the home cache line.
            return self.home(coord).wrapping_add(j as usize) & mask;
        }
        // seqLin, j ≥ G: fresh radix windows, digit index shifted past the prefix.
        let p = coord.wrapping_mul(PHI);
        let jj = j - LINEAR_G + 1; // radix step 1, 2, … (window 0 is home, already probed)
        let used = bits * (jj + 1);
        if used <= 64 {
            // A window of the first product `coord·PHI` — the proven radix digits
            // (`linear_prefix_chain_separates`, separation within D+G−1).
            (p >> (64 - used)) as usize & mask
        } else {
            // Past the first product's digits: the proven TERMINAL SWEEP (`GroupProbe.group_descent_dichotomy`
            // at slot granularity). `sweep_period_no_repeat`: the `+1` offset makes `1..2^bits` a
            // permutation of the slots, so a miss finds EMPTY within one period (guaranteed
            // termination); `zero_offset_duplicates` proves the `+1` is required (offset 0 re-glances
            // the last digit's slot). Base carries the low product bits (the mirror of `pile_card`'s
            // residual) where keys that collide every high window still differ. The dense-block scan is
            // routed away by construction: joins/counts/mixed keys go to the NB face (`FreqMap`), whose
            // uniform fold has no identity-home block; this B-face serves dense coordinate keys whose
            // misses hit EMPTY at their own slot and never reach here.
            let full = 64 / bits; // radix windows per product
            ((p >> (64 - bits * full)) as usize + (jj - full) as usize + 1) & mask
        }
    }

    /// Walk `coord`'s probe sequence; return its slot and whether it was found. Coordinate-only — the
    /// key is never touched. `descend` is `#[cold]`, keeping the collision walk off the hot path.
    ///
    /// Indexing is unchecked: `home`/`descend` both mask with `& (len - 1)` on a power-of-two array,
    /// so `i` is always in bounds by construction and the checks would be pure overhead.
    ///
    /// A reserved coordinate (`EMPTY`/`TOMB`) is never stored in `coords` — its key lives in the
    /// [`SideStore`] — so a coordinate-only probe of one returns not-found at its home slot without
    /// dereferencing a slot (a `TOMB` slot's value is uninitialized). The keyed [`probe_keyed`] is
    /// the probe the storage faces use.
    #[inline]
    fn probe(&self, coord: u64) -> (usize, bool) {
        let i = self.home(coord);
        if is_sentinel(coord) {
            return (i, false);
        }
        let mut i = i;
        let mut j = 0;
        loop {
            // SAFETY: i ∈ [0, len) — home() and descend() mask to the array.
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord {
                return (i, true); // hit — the address is the key
            }
            if sc == EMPTY {
                return (i, false); // empty — coord was never here, or the chain past it is empty
            }
            // occupied, non-matching: a live collision OR a tombstone (`TOMB`) — either way the
            // chain continues, so a removed slot never breaks reachability for keys probed past it.
            j += 1;
            // Safety net: the double-hashing tail visits every slot within `cap` steps (odd stride on a
            // 2^bits table), so a non-full table always terminates well under this. A blow-past is a
            // real regression (a broken descent), not adversarial cost — cheap vs the miss it guards.
            debug_assert!(
                (j as usize) <= self.coords.len() + 256,
                "PROBE NON-TERM: coord={coord:#x} bits={} home={} cap={} j={j}",
                self.bits,
                self.home(coord),
                self.coords.len()
            );
            i = self.descend(coord, j);
        }
    }

    /// [`probe`](Self::probe) with full-key confirmation: a coordinate match is a hit only when the
    /// stored key equals `k`. Two distinct keys with equal coordinates therefore take two slots, and
    /// a mismatch keeps probing exactly as a different coordinate would. `Regen`'s `matches` is the
    /// coordinate round-trip, so for a [`CoordMap`] this is the coordinate probe unchanged.
    ///
    /// Only ever called with a non-reserved `coord` (the sentinel keys are resolved in the
    /// [`SideStore`]); a coordinate match on a live slot therefore implies an initialized key.
    #[inline]
    fn probe_keyed(&self, coord: u64, k: &K) -> (usize, bool) {
        let mut i = self.home(coord);
        let mut j = 0;
        loop {
            // SAFETY: i ∈ [0, len) — home()/descend() mask to the array; `coord` is non-reserved, so
            // `sc == coord` names a live slot whose key is initialized (confirmed by `matches`).
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord && self.keys.matches(i, coord, k) {
                return (i, true);
            }
            if sc == EMPTY {
                return (i, false);
            }
            j += 1;
            debug_assert!(
                (j as usize) <= self.coords.len() + 256,
                "PROBE NON-TERM: coord={coord:#x} bits={} home={} cap={} j={j}",
                self.bits,
                self.home(coord),
                self.coords.len()
            );
            i = self.descend(coord, j);
        }
    }

    /// Read one key. `None` = absent. ONE algorithm for both key stores: the home coordinate, then
    /// (on a collision) the `seqLin` walk — a linear prefix in the home cache line, then radix windows
    /// (`RadixDescent.linear_prefix_chain_separates`). The near-certain home hit reads one coordinate,
    /// confirms the stored key, and reads its value; a chain or miss stays in the home line. `KS`
    /// selects only key *storage* (regenerated vs kept), never the algorithm.
    ///
    /// The stored key is confirmed on a coordinate match (free on `Regen`, where the coordinate
    /// round-trips), so a coordinate collision returns `None` instead of another key's value;
    /// [`get_exact`](Self::get_exact) is the same read under its historical name.
    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.get_coord(k)
    }

    /// The coordinate read: one scalar coordinate at `home`, a predicted-taken compare, then the
    /// collision tail only on the rare miss. The scalar home probe is what keeps the *coordinate*-key
    /// path (CoordMap: near-certain home hit) free of the SIMD glance's overhead — the glance is paid
    /// only where it wins, in the collision tail that opaque hashed keys actually reach.
    ///
    /// A coordinate match is confirmed against the stored key (free on `Regen`), so two distinct keys
    /// with equal coordinates are two entries; a reserved coordinate is resolved in the [`SideStore`].
    #[inline]
    fn get_coord(&self, k: &K) -> Option<&V> {
        let coord = Self::coord_of(k);
        if is_sentinel(coord) {
            let i = self.side.find(coord, k)?;
            // SAFETY: `find` only returns an entry whose `live` marker is set.
            return Some(unsafe { self.side.vals.get_unchecked(i).assume_init_ref() });
        }
        let home = self.home(coord);
        // SAFETY: home ∈ [0, len). The home hit reads one coord, confirms the key, then one value.
        let sc = unsafe { *self.coords.get_unchecked(home) };
        if sc == coord && self.keys.matches(home, coord, k) {
            return Some(unsafe { self.vals.get_unchecked(home).assume_init_ref() });
        }
        if sc == EMPTY {
            return None;
        }
        self.get_coord_cold_keyed(coord, home, k)
    }

    /// Collision tail of [`get_coord`](Self::get_coord). The linear prefix
    /// (`home .. home + LINEAR_G`) is one contiguous 64-byte cache line, so filter it in a single
    /// branchless glance ([`prefix_masks`]) instead of the mispredict-prone scalar compare-per-slot:
    /// an EMPTY lane terminates the chain, and — since a key is placed before any EMPTY in its own
    /// probe sequence — a present key is at a matching lane before the first EMPTY. All-occupied-no-
    /// match resumes at the radix windows. Only the last `LINEAR_G − 1` homes wrap the array end →
    /// the scalar walk (which masks the wrap). Lane 0 is `home`, already probed by the caller.
    ///
    /// The keyed tail: every coordinate-match lane is confirmed against the stored key, so two
    /// distinct keys with equal coordinates are not conflated (`Regen`'s `matches` is the coordinate
    /// round-trip).
    #[cold]
    #[inline(never)]
    fn get_coord_cold_keyed(&self, coord: u64, home: usize, k: &K) -> Option<&V> {
        self.get_coord_cold_with(coord, home, |i| self.keys.matches(i, coord, k))
    }

    /// Shared body of the collision tail. `confirm(slot)` is the hit test on a lane whose coordinate
    /// matched — the stored-key compare for the keyed caller.
    #[cold]
    #[inline(never)]
    fn get_coord_cold_with(
        &self,
        coord: u64,
        home: usize,
        confirm: impl Fn(usize) -> bool,
    ) -> Option<&V> {
        let len = self.coords.len();
        if home + LINEAR_G as usize <= len {
            // `home + LINEAR_G ≤ len`, so the `LINEAR_G`-lane read stays in bounds.
            let (mut hits, empties) = prefix_masks(&self.coords, home, coord);
            while hits != 0 {
                let slot = home + hits.trailing_zeros() as usize;
                if confirm(slot) {
                    // SAFETY: slot < home + LINEAR_G ≤ len; a confirmed match ⇒ value initialized.
                    return Some(unsafe { self.vals.get_unchecked(slot).assume_init_ref() });
                }
                hits &= hits - 1;
            }
            if empties != 0 {
                return None; // the chain terminated inside the prefix
            }
            // All LINEAR_G slots occupied, no match: resume at the first radix window.
            return self.get_coord_walk_with(coord, LINEAR_G, &confirm);
        }
        // Wrap tail (rare): the scalar walk from step 1 — `home` was already probed by the caller.
        self.get_coord_walk_with(coord, 1, &confirm)
    }

    /// Scalar continuation of [`get_coord_cold`](Self::get_coord_cold) from probe step `j`: the radix
    /// windows past the SIMD-scanned linear prefix (`j = LINEAR_G`), or the whole walk on the rare
    /// array-end wrap (`j = 1`). One coordinate compare (plus the confirmation) per step.
    #[cold]
    #[inline(never)]
    fn get_coord_walk(&self, coord: u64, j: u32) -> Option<&V> {
        self.get_coord_walk_with(coord, j, &|_| true)
    }

    /// Keyed scalar continuation — see [`get_coord_walk`](Self::get_coord_walk).
    #[cold]
    #[inline(never)]
    fn get_coord_walk_keyed(&self, coord: u64, j: u32, k: &K) -> Option<&V> {
        self.get_coord_walk_with(coord, j, &|i| self.keys.matches(i, coord, k))
    }

    fn get_coord_walk_with(
        &self,
        coord: u64,
        mut j: u32,
        confirm: &impl Fn(usize) -> bool,
    ) -> Option<&V> {
        loop {
            let i = self.descend(coord, j);
            // SAFETY: i ∈ [0, len) — descend() masks to the array.
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord && confirm(i) {
                // SAFETY: a confirmed coord match ⇒ value initialized.
                return Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() });
            }
            if sc == EMPTY {
                return None;
            }
            j += 1;
        }
    }

    /// Resolve a stream of keys, software-pipelined. Compute each key's coordinate, prefetch its home
    /// line, and resolve the lookup `P` keys later — so each key's cache miss overlaps the probes of
    /// the keys ahead of it. A plain [`get`](Self::get) loop stalls on each miss with nothing else in
    /// flight; pipelining a stream lets the misses run concurrently (memory-level parallelism). For a
    /// lookup-all pass over a table larger than cache this is ~2× a `get` loop **and ~2× hashbrown**;
    /// a small (cache-resident) table falls back to the plain loop, so it never loses.
    ///
    /// `visit(key, value)` runs once per key in order; absence is `None`.
    #[inline]
    pub fn get_each<'a, F>(&'a self, keys: &'a [K], mut visit: F)
    where
        F: FnMut(&'a K, Option<&'a V>),
    {
        const P: usize = 16; // power of two ⇒ ring index is a mask, not a divide
        let n = keys.len();
        // Small / cache-resident: no miss to hide, so skip the pipeline.
        if n <= P || self.footprint_bytes() <= (1 << 19) {
            for k in keys {
                visit(k, self.get(k));
            }
            return;
        }
        let mut coords = [0u64; P];
        for (i, c) in coords.iter_mut().enumerate() {
            *c = Self::coord_of(&keys[i]);
            self.prefetch_coord(*c);
        }
        for i in 0..n {
            let c = coords[i & (P - 1)];
            let v = self.get_with_coord(c, &keys[i]);
            visit(&keys[i], v);
            let j = i + P;
            if j < n {
                let cj = Self::coord_of(&keys[j]);
                coords[j & (P - 1)] = cj; // j ≡ i (mod P): the slot we just consumed
                self.prefetch_coord(cj);
            }
        }
    }

    /// Resolve a stream of keys across every core — the threaded version of
    /// [`get_each`](Self::get_each). Split the keys into one slice per core, resolve each on its own
    /// thread into a disjoint output slice; each lane still pipelines internally, so this stacks core
    /// parallelism on top of per-core memory-level parallelism. A read of one key can't affect
    /// another, so the split is safe, and `out[i] = get(keys[i])` with order preserved.
    ///
    /// Small/serial inputs just pipeline on this thread (spawn isn't worth it), so it never loses.
    pub fn get_all<'a>(&'a self, keys: &'a [K]) -> Vec<Option<&'a V>>
    where
        K: Sync,
        V: Sync,
        KS: Sync,
    {
        let n = keys.len();
        let mut out: Vec<Option<&'a V>> = vec![None; n];
        let pool = par::Pool::global();
        // Below ~a page of keys per executor the dispatch/partition cost dominates — pipeline serially.
        if pool.width() <= 1 || n < 1 << 15 {
            let mut w = out.iter_mut();
            self.get_each(keys, |_, v| {
                if let Some(slot) = w.next() {
                    *slot = v;
                }
            });
            return out;
        }
        // Static equal partition, self-balanced by the pool's task counter (the reduce is
        // schedule-free — `Web.preduceAt_schedule_free`). Each task runs the *pipelined* `get_each`
        // over its window, so core parallelism stacks on per-core memory-level parallelism. Measured
        // (examples/machine.rs, i9-14900KF): saturating DRAM (~66 GB/s) needs ~L·BW/line ≈ 100
        // outstanding misses; one miss per thread gives 32 (28% of ceiling, measured), while
        // threads × pipeline reaches ~95% of ceiling — deep per-thread MLP is required at every
        // thread count, and N×P does NOT over-saturate the controller (the fabric window is the
        // ceiling, `MemoryLevelParallelism.latency_hidden`, not a cliff). Per-core depth is capped
        // by the core's fill-buffer width (~14 P / ~9 E), which `get_each`'s P=16 already covers.
        let tasks = (pool.width() * 4).min(n);
        let chunk = n.div_ceil(tasks);
        // Disjoint output windows keyed by task index: task t writes out[t*chunk .. (t+1)*chunk].
        let out_ptr = par::SendPtr(out.as_mut_ptr());
        pool.dispatch(tasks, |t| {
            let start = t * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: task windows are disjoint (distinct `t` ⇒ distinct [start,end)), so no two tasks
            // touch the same `out` slot; the dispatch barrier joins before `out` is read/returned.
            let oc = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(start), end - start) };
            let mut w = oc.iter_mut();
            self.get_each(&keys[start..end], |_, v| {
                if let Some(slot) = w.next() {
                    *slot = v;
                }
            });
        });
        out
    }

    /// Reduce a **stream of keys** in parallel: look each key up, `map` its result (`None` if absent),
    /// and `combine` across all cores — without materializing the per-key results. This is the fused
    /// face of [`get_all`](Self::get_all): where `get_all` writes an `N`-vector the caller then folds,
    /// `get_reduce` folds inline. Each worker runs the pipelined [`get_each`](Self::get_each) over its
    /// key-chunk into a local accumulator, and the partials combine. `combine` must be a commutative
    /// monoid with identity `id`, so the split is schedule-free (`Web.preduceAt_schedule_free`) — any
    /// task→core assignment and merge order give the same answer. The honest parallel-aggregate face:
    /// core parallelism over per-core memory-level parallelism, no `N`-alloc, no scatter-back.
    pub fn get_reduce<'a, M, Rf, C>(&'a self, keys: &'a [K], id: M, map: Rf, combine: C) -> M
    where
        M: Send + Sync + Clone,
        Rf: Fn(Option<&'a V>) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
        K: Sync,
        V: Sync,
        KS: Sync,
    {
        let reduce_range = |start: usize, end: usize| -> M {
            let mut acc = id.clone();
            // pipelined lookups (get_each) folded into the local accumulator, in stream order.
            self.get_each(&keys[start..end], |_, v| {
                acc = combine(core::mem::replace(&mut acc, id.clone()), map(v));
            });
            acc
        };
        let n = keys.len();
        let pool = par::Pool::global();
        // Below a page of keys per executor the dispatch cost dominates — reduce serially (still
        // pipelined by get_each).
        if pool.width() <= 1 || n < 1 << 15 {
            return reduce_range(0, n);
        }
        // One partial per task; more tasks than executors so uneven work self-balances. Schedule-free,
        // so the assignment and merge order are free.
        let tasks = (pool.width() * 4).min(n);
        let chunk = n.div_ceil(tasks);
        let mut partials: Vec<M> = (0..tasks).map(|_| id.clone()).collect();
        let pptr = par::SendPtr(partials.as_mut_ptr());
        pool.dispatch(tasks, |t| {
            let start = t * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: task `t` writes only `partials[t]`; distinct tasks ⇒ disjoint slots, and the
            // dispatch barrier orders the write before the merge below.
            unsafe { *pptr.add(t) = reduce_range(start, end) };
        });
        partials.into_iter().fold(id, &combine)
    }

    /// Reduce the whole table in parallel: apply `residual` to every value and `combine` the results
    /// across all cores. Where [`get`](Self::get) reads one key and [`get_each`](Self::get_each) a
    /// stream, this reads the entire table — the aggregate a point-lookup map makes you write a serial
    /// loop for. `combine` must be a commutative monoid with identity `id`, so the split across cores
    /// gives the same answer regardless of how the work is partitioned.
    pub fn encircle<M, R, C>(&self, id: M, residual: R, combine: C) -> M
    where
        M: Send + Sync + Clone,
        R: Fn(&V) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
        K: Sync,
        V: Sync,
        KS: Sync,
    {
        let fold_range = |start: usize, end: usize| -> M {
            let mut acc = id.clone();
            for i in start..end {
                if is_live(self.coords[i]) {
                    // SAFETY: live coord ⇒ value initialized.
                    acc = combine(
                        acc,
                        residual(unsafe { self.vals.get_unchecked(i).assume_init_ref() }),
                    );
                }
            }
            acc
        };
        let n = self.coords.len();
        let pool = par::Pool::global();
        let main = if pool.width() <= 1 || n < 1 << 14 {
            fold_range(0, n) // no core to spread onto — one pass
        } else {
            // One partial per task; more tasks than executors so uneven live-density self-balances. The
            // reduce is schedule-free (`Web.preduceAt_schedule_free`), so any task→executor assignment and
            // any merge order give the same answer.
            let tasks = (pool.width() * 4).min(n);
            let chunk = n.div_ceil(tasks);
            let mut partials: Vec<M> = (0..tasks).map(|_| id.clone()).collect();
            let pptr = par::SendPtr(partials.as_mut_ptr());
            pool.dispatch(tasks, |t| {
                let start = t * chunk;
                if start >= n {
                    return;
                }
                let end = (start + chunk).min(n);
                // SAFETY: task `t` writes only `partials[t]`; distinct tasks ⇒ disjoint slots, and the
                // dispatch barrier orders the write before the merge below.
                unsafe { *pptr.add(t) = fold_range(start, end) };
            });
            partials.into_iter().fold(id, &combine)
        };
        // Side entries (the reserved-coordinate keys): O(1) on ordinary tables, folded onto the result.
        let mut acc = main;
        for (i, live) in self.side.live.iter().enumerate() {
            if *live {
                // SAFETY: `live[i]` ⇒ value initialized.
                acc = combine(acc, residual(unsafe {
                    self.side.vals.get_unchecked(i).assume_init_ref()
                }));
            }
        }
        acc
    }

}

/// The key-readback faces — the only operations that project [`KeyReadout::Key`], so the only ones
/// that require the extension trait. Storage, probe, insert and growth above stay on the GAT-free
/// [`KeyStore`], which is what lets Aeneas translate those executing bodies.
impl<K: Eq + Coord, V, KS: KeyReadout<K>> Table<K, V, KS> {
    /// Like [`encircle`](Self::encircle), but return *where* the minimum is: the entry with the
    /// smallest `residual`, and its key. Ties break toward the lower slot, so the result is
    /// deterministic. `None` for an empty table.
    pub fn encircle_floor<M, R>(&self, residual: R) -> Option<(KS::Key<'_>, &V)>
    where
        M: Ord + Copy + Send,
        R: Fn(&V) -> M + Sync,
        K: Sync,
        V: Sync,
        KS: Sync,
    {
        // argmin over one chunk: (residual, absolute slot), strict `<` so the lower slot wins ties.
        let range_argmin = |start: usize, end: usize| -> Option<(M, usize)> {
            let mut best: Option<(M, usize)> = None;
            for i in start..end {
                if is_live(self.coords[i]) {
                    // SAFETY: live coord ⇒ value initialized.
                    let m = residual(unsafe { self.vals.get_unchecked(i).assume_init_ref() });
                    if best.map_or(true, |(bm, _)| m < bm) {
                        best = Some((m, i));
                    }
                }
            }
            best
        };
        let n = self.coords.len();
        let pool = par::Pool::global();
        let floor = if pool.width() <= 1 || n < 1 << 14 {
            range_argmin(0, n)
        } else {
            let tasks = (pool.width() * 4).min(n);
            let chunk = n.div_ceil(tasks);
            let mut partials: Vec<Option<(M, usize)>> = vec![None; tasks];
            let pptr = par::SendPtr(partials.as_mut_ptr());
            pool.dispatch(tasks, |t| {
                let start = t * chunk;
                if start >= n {
                    return;
                }
                let end = (start + chunk).min(n);
                // SAFETY: task `t` writes only `partials[t]`; disjoint across tasks, barrier-ordered.
                unsafe { *pptr.add(t) = range_argmin(start, end) };
            });
            // combine partials: least residual, then least slot — deterministic (independent of the
            // task→executor assignment, since `min_by` over the disjoint ranges is order-free).
            partials
                .into_iter()
                .flatten()
                .min_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)))
        };
        // Side entries: the same argmin at pseudo-slots `n + i` (past every main slot, so a main tie
        // still wins; among side entries the lower index wins). O(1) on ordinary tables.
        let mut floor = floor;
        for (i, live) in self.side.live.iter().enumerate() {
            if *live {
                // SAFETY: `live[i]` ⇒ value initialized.
                let m = residual(unsafe { self.side.vals.get_unchecked(i).assume_init_ref() });
                let s = n + i;
                if floor.map_or(true, |(bm, bs)| m < bm || (m == bm && s < bs)) {
                    floor = Some((m, s));
                }
            }
        }
        // a returned slot is occupied ⇒ value initialized; the key is read/regenerated.
        floor.map(|(_, s)| {
            if s < n {
                let c = self.coords[s];
                // SAFETY: the argmin ranged over occupied slots only ⇒ value initialized.
                (self.keys.key(s, c), unsafe {
                    self.vals.get_unchecked(s).assume_init_ref()
                })
            } else {
                let i = s - n;
                let c = self.side.coords[i];
                // SAFETY: side entries are occupied ⇒ value initialized.
                (self.side.keys.key(i, c), unsafe {
                    self.side.vals.get_unchecked(i).assume_init_ref()
                })
            }
        })
    }
}

/// A landed insert: the slot is blank and the caller owns the merge between `new` and the value
/// `old` that was already there. Returned by [`Table::insert_core`] instead of writing, so every
/// insert face shares one probe/reserve core and differs only in what it does with the two values.
/// The caller MUST write `slot` back and re-occupy it with `coord`.
struct InsertHit<V> {
    /// The probe's slot; in bounds by construction (`probe` masks to the power-of-two array).
    slot: usize,
    /// The slot's coordinate, to restore after the write — identity is unchanged by an overwrite.
    /// Unused when `side` is set (the side entry's coordinate is fixed).
    coord: u64,
    /// The hit is an entry of the [`SideStore`], not a main slot. The caller writes `side`'s arrays
    /// and re-sets `side.live[slot]` instead of `coords[slot]`.
    side: bool,
    /// The incoming value, handed back because a hit does not consume it.
    new: V,
    /// The value that was in the slot, moved out so exactly one of the two faces owns its drop.
    old: V,
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// Remove `k` and return its value, or `None` if absent. The bijection face — inverse of `put`.
    ///
    /// A removed main slot becomes a **tombstone** (`TOMB`), not `EMPTY`: because the probe re-addresses
    /// by windows of `coord·PHI` (not a contiguous run), blanking a slot to `EMPTY` mid-chain would make
    /// every key that probed *past* it unreachable. A tombstone reads as occupied-but-non-matching, so
    /// chains stay intact; the next `resize` (triggered on live + tombstone occupancy) reclaims it. A
    /// reserved-coordinate key is removed from the [`SideStore`].
    #[inline]
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let coord = Self::coord_of(k);
        if is_sentinel(coord) {
            let i = self.side.find(coord, k)?;
            // SAFETY: `find` only returns an entry whose `live` marker is set.
            let old = unsafe { self.side.vals.get_unchecked_mut(i).assume_init_read() };
            self.side.remove_entry(i);
            return Some(old);
        }
        let (i, found) = self.probe_keyed(coord, k);
        if found {
            // SAFETY: found ⇒ value initialized. Read it out and drop the key, then tombstone the
            // slot: its value is now uninitialized, and `TOMB` (like `EMPTY`) fails the liveness test,
            // so `Drop`/`Clone`/iteration skip it — no double free, no chain break.
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            self.keys.take_key(i, coord);
            unsafe { *self.coords.get_unchecked_mut(i) = TOMB };
            self.len -= 1;
            self.tombs += 1;
            Some(old)
        } else {
            None
        }
    }

    /// The reserve/probe/store core every insert face shares. On a miss `new` is stored and this
    /// returns `None`. On a hit the old value is read out and the slot is blanked, and the caller
    /// receives the [`InsertHit`] — so [`insert_with`](Self::insert_with) (merge applied by the
    /// caller) and [`insert_merge_last_write`](Self::insert_merge_last_write) (merge specialised to
    /// its own body) are the same probe over one body, not two copies of it.
    ///
    /// Blanking before the merge is what keeps a panicking merge sound: the slot is logically empty
    /// for its duration, so the unwinding `old` is dropped once and `Drop` skips the slot — no
    /// double free, no chain break. A key whose coordinate is a reserved marker (`EMPTY`/`TOMB`) is
    /// inserted into the side store instead of the slot arrays.
    #[inline]
    fn insert_core(&mut self, k: K, new: V) -> Option<InsertHit<V>> {
        let coord = Self::coord_of(&k);
        if is_sentinel(coord) {
            if let Some(i) = self.side.find(coord, &k) {
                // SAFETY: `find` only returns an entry whose `live` marker is set. Blank the `live`
                // marker (keep the key) so a panicking merge drops the value once.
                let old = unsafe { self.side.vals.get_unchecked_mut(i).assume_init_read() };
                self.side.live[i] = false;
                return Some(InsertHit {
                    slot: i,
                    coord,
                    side: true,
                    new,
                    old,
                });
            }
            self.side.push(coord, k, new);
            return None;
        }
        // Occupancy that bounds the probe (live + tombstones, since probe walks through both) must stay
        // under the load limit — otherwise a churn workload could fill the table with tombstones and a
        // not-found probe would never reach an `EMPTY`. With no deletions `tombs == 0`, so this is the
        // original trigger unchanged.
        if (self.len + self.tombs + 1) * 8 >= self.coords.len() * 7 {
            // Genuinely full of live keys ⇒ grow; merely crowded by tombstones ⇒ same-size rebuild
            // clears them (reclaims the space without doubling).
            let new_bits = if (self.len + 1) * 8 >= self.coords.len() * 7 {
                self.bits + 1
            } else {
                self.bits
            };
            self.resize(new_bits);
        }
        let (i, found) = self.probe_keyed(coord, &k);
        // SAFETY: probe returns an in-bounds slot; coords, vals, and keys share the index space.
        if found {
            // SAFETY: found ⇒ value initialized. Read it out, then blank the coord: the slot is
            // logically empty for the duration of the merge, so if the merge panics the value is
            // dropped once (by the unwinding `old`) and `Drop` skips this slot — no double free.
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            unsafe { *self.coords.get_unchecked_mut(i) = EMPTY };
            Some(InsertHit {
                slot: i,
                coord,
                side: false,
                new,
                old,
            })
        } else {
            unsafe { *self.coords.get_unchecked_mut(i) = coord };
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
            self.keys.set(i, k); // Stored: write the key; Regen: no-op
            self.len += 1;
            None
        }
    }

    /// The one mutator: set `k` to `f(new, old)`. An absent key has no old value, so you get
    /// `f(new, identity) = new`. Every table face is just this with a different `f`.
    #[inline]
    pub fn insert_with(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) {
        if let Some(hit) = self.insert_core(k, new) {
            let merged = f(hit.new, hit.old);
            if hit.side {
                // SAFETY: `insert_core` cleared `side.live[slot]`, so the value is uninitialized;
                // re-writing and re-marking restores the invariant. The entry's key is untouched.
                unsafe { *self.side.vals.get_unchecked_mut(hit.slot) = MaybeUninit::new(merged) };
                self.side.live[hit.slot] = true;
            } else {
                // SAFETY: `insert_core` blanked the slot, so `vals[slot]` is uninitialized and
                // `coords[slot]` is `EMPTY`; writing the merged value and re-occupying restores the
                // invariant.
                unsafe { *self.vals.get_unchecked_mut(hit.slot) = MaybeUninit::new(merged) };
                unsafe { *self.coords.get_unchecked_mut(hit.slot) = hit.coord }; // re-occupy — identity unchanged
            }
        }
    }

    /// The Map face's merge, concretely: set `k` to `new`, overwriting whatever was there — the same
    /// semantics as [`put`](Self::put) (`insert_with(k, v, set)`, with [`set`](crate::hashtrinity::set)
    /// `= new`), specialised into its own body instead of taking the merge as a `fn` item.
    ///
    /// Merge-free by construction, not by duplication: it is [`insert_core`](Self::insert_core) plus
    /// the merge `set(new, old) = new`, which consumes `old` and keeps `new`. A first-order translator
    /// cannot take an arrow-typed merge argument, so the concrete face is what the extraction roots.
    #[inline]
    pub fn insert_merge_last_write(&mut self, k: K, new: V) {
        if let Some(hit) = self.insert_core(k, new) {
            drop(hit.old); // `set(new, old) = new` ignores `old`; this is where it is dropped.
            if hit.side {
                // SAFETY: as in `insert_with` — `insert_core` cleared the side entry's `live` marker.
                unsafe { *self.side.vals.get_unchecked_mut(hit.slot) = MaybeUninit::new(hit.new) };
                self.side.live[hit.slot] = true;
            } else {
                // SAFETY: as in `insert_with` — `insert_core` blanked the slot; this restores it.
                unsafe { *self.vals.get_unchecked_mut(hit.slot) = MaybeUninit::new(hit.new) };
                unsafe { *self.coords.get_unchecked_mut(hit.slot) = hit.coord };
            }
        }
    }

    /// Rebuild into a fresh table of `new_bits` slots, carrying only live entries — so tombstones
    /// vanish. `new_bits > bits` grows; `new_bits == bits` is an in-place compaction that reclaims
    /// tombstones without doubling. The old `grow` is `resize(bits + 1)` with no tombstones present.
    /// The [`SideStore`] is carried over unchanged (it does not depend on `bits`).
    #[cold]
    fn resize(&mut self, new_bits: u32) {
        let mut bigger = Self::with_bits(new_bits);
        for i in 0..self.coords.len() {
            let c = self.coords[i];
            if is_live(c) {
                // Move the key out first so the keyed probe can confirm it: two distinct keys with
                // equal coordinates must land on two slots of the rebuilt table, not one.
                let key = self.keys.take_key(i, c);
                let (j, _) = bigger.probe_keyed(c, &key); // re-place along the same probe sequence
                                                          // SAFETY: live coord ⇒ value initialized; bigger.probe_keyed returns an
                                                          // in-bounds slot. Move the value out and blank the source coord so the old
                                                          // table's Drop skips it.
                let v = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
                self.coords[i] = EMPTY;
                unsafe { *bigger.coords.get_unchecked_mut(j) = c };
                unsafe { *bigger.vals.get_unchecked_mut(j) = MaybeUninit::new(v) };
                bigger.keys.set(j, key); // Stored: write the key; Regen: no-op
                bigger.len += 1;
            }
        }
        // Carry the side store over; the rebuilt table is fresh (empty side store).
        bigger.side = std::mem::replace(&mut self.side, SideStore::empty());
        *self = bigger; // tombstones dropped: bigger.tombs == 0
    }

    /// The number of live keys, main slots plus side entries.
    pub fn len(&self) -> usize {
        self.len + self.side.coords.len()
    }
    /// Whether the table holds no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    fn footprint_bytes(&self) -> usize {
        self.coords.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<MaybeUninit<V>>())
    }
    /// Slot count (the power-of-two array length). `len / capacity` is the load factor.
    pub fn capacity(&self) -> usize {
        self.coords.len()
    }
}

/// The key-readback face of iteration — projects [`KeyReadout::Key`], so it requires the extension
/// trait; the storage and probe bodies stay on [`KeyStore`].
impl<K: Eq + Coord, V, KS: KeyReadout<K>> Table<K, V, KS> {
    /// Iterate the live `(key, &value)` pairs in arbitrary order — the main slots then the side
    /// entries.
    pub fn iter(&self) -> impl Iterator<Item = (KS::Key<'_>, &V)> {
        let main = self.coords.iter().enumerate().filter_map(|(i, c)| {
            if is_live(*c) {
                // SAFETY: live coord ⇒ value initialized.
                Some((self.keys.key(i, *c), unsafe {
                    self.vals.get_unchecked(i).assume_init_ref()
                }))
            } else {
                None
            }
        });
        let side = self
            .side
            .coords
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                if self.side.live[i] {
                    // SAFETY: `live[i]` ⇒ value initialized.
                    Some((self.side.keys.key(i, *c), unsafe {
                        self.side.vals.get_unchecked(i).assume_init_ref()
                    }))
                } else {
                    None
                }
            });
        main.chain(side)
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// Bit-exact read — never returns a wrong value. The stored key is compared on a coordinate
    /// match (free on `Regen`, where the coordinate round-trips), so a coordinate collision returns
    /// `None` instead of someone else's value; a reserved-coordinate key is resolved in the
    /// [`SideStore`]. Same read as [`get`](Self::get).
    #[inline]
    pub fn get_exact(&self, k: &K) -> Option<&V> {
        self.get_coord(k)
    }

    /// Worst-case probe length, in steps. Coordinate keys: **1** (address = key, nothing to flood).
    /// Folded keys: collisions descend fresh windows, and `coord·PHI` is a bijection, so distinct
    /// keys can't fully collide — a pinned home slot builds no pile of ALIASES. That is not a bound
    /// on this number (`AggregationBound.descent_can_be_wholly_blocked`); past the `⌈64/bits⌉`
    /// windows the ceiling is the terminal sweep, one period of the slot space. Read it as a
    /// measurement. Diagnostic — not part of the stable surface.
    #[doc(hidden)]
    pub fn longest_probe(&self) -> usize {
        let mut worst = 0;
        for (slot, sc) in self.coords.iter().enumerate() {
            if is_live(*sc) {
                let mut i = self.home(*sc);
                let mut j = 0u32;
                while i != slot {
                    j += 1;
                    i = self.descend(*sc, j);
                }
                worst = worst.max(j as usize + 1);
            }
        }
        worst
    }

    /// Prefetch (T0) the coordinate home line for the scalar pipeline. Issued for a key `P` steps
    /// ahead so its cache miss overlaps the probes already in flight.
    #[inline]
    fn prefetch_coord(&self, coord: u64) {
        let i = self.home(coord);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(self.coords.as_ptr().add(i).cast::<i8>(), _MM_HINT_T0);
        }
        let _ = i;
    }

    /// The coordinate seed matched to this table's slot count (see the [`seed`] module). `seed.home(c)`
    /// is this table's own home slot and `seed.residual(c)` is the rest of `c·PHI` the slot doesn't
    /// carry — what a compact readout would store instead of the full coordinate. Diagnostic — not
    /// part of the stable surface.
    #[doc(hidden)]
    #[inline]
    pub fn coord_seed(&self) -> CoordSeed {
        CoordSeed::new(self.bits)
    }

    /// `get` from a pre-computed coordinate — the pipelined path computes the coordinate during
    /// prefetch and resolves `k` here. A reserved coordinate is resolved in the [`SideStore`].
    #[inline]
    fn get_with_coord(&self, coord: u64, k: &K) -> Option<&V> {
        if is_sentinel(coord) {
            let i = self.side.find(coord, k)?;
            // SAFETY: `find` only returns an entry whose `live` marker is set.
            return Some(unsafe { self.side.vals.get_unchecked(i).assume_init_ref() });
        }
        // Same shape as `get_coord`: the near-certain home hit, then the shared SIMD collision tail.
        let home = self.home(coord);
        // SAFETY: home ∈ [0, len).
        let sc = unsafe { *self.coords.get_unchecked(home) };
        if sc == coord && self.keys.matches(home, coord, k) {
            return Some(unsafe { self.vals.get_unchecked(home).assume_init_ref() });
        }
        if sc == EMPTY {
            return None;
        }
        self.get_coord_cold_keyed(coord, home, k)
    }
}

/// Equal iff the same entries: main slots by coordinate, side entries by coordinate and value (the
/// coordinate identity the main comparison has always used).
impl<K: Eq + Coord, V: PartialEq, KS: KeyStore<K>> PartialEq for Table<K, V, KS> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for (i, c) in self.coords.iter().enumerate() {
            if !is_live(*c) {
                continue;
            }
            let (j, found) = other.probe(*c);
            if !found {
                return false;
            }
            // SAFETY: live coord ⇒ this value initialized; found ⇒ other's value initialized.
            if unsafe { other.vals.get_unchecked(j).assume_init_ref() }
                != unsafe { self.vals.get_unchecked(i).assume_init_ref() }
            {
                return false;
            }
        }
        // Side entries participate by `(coordinate, value)`, not through `len()` alone: two tables
        // differing only in a reserved-coordinate entry must compare unequal. O(s²) with `s` the
        // side-entry count (0 on ordinary tables).
        if self.side.coords.len() != other.side.coords.len() {
            return false;
        }
        let mut matched = vec![false; other.side.coords.len()];
        for i in 0..self.side.coords.len() {
            let c = self.side.coords[i];
            // SAFETY: side entries are occupied ⇒ values initialized.
            let v = unsafe { self.side.vals.get_unchecked(i).assume_init_ref() };
            let mut hit = false;
            for j in 0..other.side.coords.len() {
                if !matched[j]
                    && other.side.coords[j] == c
                    && unsafe { other.side.vals.get_unchecked(j).assume_init_ref() } == v
                {
                    matched[j] = true;
                    hit = true;
                    break;
                }
            }
            if !hit {
                return false;
            }
        }
        true
    }
}
impl<K: Eq + Coord, V: Eq, KS: KeyStore<K>> Eq for Table<K, V, KS> {}
