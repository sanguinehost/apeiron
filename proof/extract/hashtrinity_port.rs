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
use crate::hashtrinity::{Coord, Face, FreqMap, Species, Table};

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

// ── The std-shaped faces and the regex engine, rooted here to keep proof/extract at 12 files. ──
use crate::hashtrinity::face::{Absorb, Entry, Locate};
use crate::hashtrinity::tregex::{term, Match};

// ── The borrowed-key views (`face.rs`): `Locate`/`Absorb` over `&K`, `&str`, tuples ─────────────

/// Real `<&u64 as Locate<u64>>::coord` — the identity view's coordinate.
pub fn word_locate_ref_coord(k: u64) -> u64 {
    let view: &u64 = &k;
    <&u64 as Locate<u64>>::coord(&view)
}

/// Real `<&u64 as Locate<u64>>::same` — the exact key compare of the identity view.
pub fn word_locate_ref_same(k: u64, other: u64) -> bool {
    let view: &u64 = &k;
    <&u64 as Locate<u64>>::same(&view, &other)
}

/// Real `<&u64 as Absorb<u64>>::absorb` — the identity view's key materialization.
pub fn word_absorb_ref(k: u64) -> u64 {
    <&u64 as Absorb<u64>>::absorb(&k)
}

/// Real `<&str as Locate<String>>::coord` — the `fold_bytes` coordinate of a borrowed string.
pub fn word_locate_str_coord(s: &str) -> u64 {
    let view: &str = s;
    <&str as Locate<String>>::coord(&view)
}

/// Real `<&str as Locate<String>>::same` — the borrowed-string exact compare.
pub fn word_locate_str_same(s: &str, k: &String) -> bool {
    let view: &str = s;
    <&str as Locate<String>>::same(&view, k)
}

/// Real `<&str as Absorb<String>>::absorb` — materialize the owned key.
pub fn word_absorb_str(s: &str) -> String {
    <&str as Absorb<String>>::absorb(s)
}

/// Real `<(&u64, &u64) as Locate<(u64, u64)>>::coord` — the tuple fold on borrowed parts.
pub fn word_locate_pair_coord(a: u64, b: u64) -> u64 {
    let view: (&u64, &u64) = (&a, &b);
    <(&u64, &u64) as Locate<(u64, u64)>>::coord(&view)
}

/// Real `<(&u64, &u64) as Locate<(u64, u64)>>::same` — the composite exact compare.
pub fn word_locate_pair_same(a: u64, b: u64, k: (u64, u64)) -> bool {
    let view: (&u64, &u64) = (&a, &b);
    <(&u64, &u64) as Locate<(u64, u64)>>::same(&view, &k)
}

/// Real `<(&u64, &u64) as Absorb<(u64, u64)>>::absorb` — materialize both components.
pub fn word_absorb_pair(a: u64, b: u64) -> (u64, u64) {
    <(&u64, &u64) as Absorb<(u64, u64)>>::absorb((&a, &b))
}

/// Real `<(&u64, &u64, &u64) as Locate<(u64, u64, u64)>>::coord` — the three-part tuple fold.
pub fn word_locate_triple_coord(a: u64, b: u64, c: u64) -> u64 {
    let view: (&u64, &u64, &u64) = (&a, &b, &c);
    <(&u64, &u64, &u64) as Locate<(u64, u64, u64)>>::coord(&view)
}

/// Real `<(&u64, &u64, &u64) as Locate<(u64, u64, u64)>>::same` — the three-part exact compare.
pub fn word_locate_triple_same(a: u64, b: u64, c: u64, k: (u64, u64, u64)) -> bool {
    let view: (&u64, &u64, &u64) = (&a, &b, &c);
    <(&u64, &u64, &u64) as Locate<(u64, u64, u64)>>::same(&view, &k)
}

/// Real `<(&u64, &u64, &u64) as Absorb<(u64, u64, u64)>>::absorb` — materialize all three.
pub fn word_absorb_triple(a: u64, b: u64, c: u64) -> (u64, u64, u64) {
    <(&u64, &u64, &u64) as Absorb<(u64, u64, u64)>>::absorb((&a, &b, &c))
}

// ── The std-shaped reads (`face.rs`), one probe each ───────────────────────────────────────────

// The borrowed-view reads (`get_at`, `get_mut_at`, `contains_at`, `get_exact_at`) are NOT rooted:
// each is bounded by `KS: KeyReadout<K>`, whose `type Key<'a>` makes the pinned Aeneas
// `translate_fwd_ty` abort on the trait declaration (algebra.rs:126-148, `TTraitType`) before any
// body. `get_mut` is likewise not rootable at this pin: its value write calls
// `MaybeUninit::assume_init_mut`, which the model does not define (the read `assume_init_ref` is
// modeled). The read `contains_key` is bounded by `KeyStore` alone and translates.

/// Real `Table::<u64, i64>::contains_key` — presence, no value touched.
pub fn word_face_contains_key(state: &Table<u64, i64>, key: u64) -> bool {
    state.contains_key(&key)
}

// `insert` and `entry(...).insert` are NOT rooted: both call an `OccupiedEntry`/`VacantEntry`
// write that returns `&mut V` through `MaybeUninit::assume_init_mut`, which the pinned Aeneas model
// does not define (`assume_init_ref` is modeled, `assume_init_mut` is not). `Table::entry`'s
// classification is rooted on its own, which is the grow/probe/overflow-lookup path; the writes it
// leads to stay outside.

/// Real `Table::<u64, i64>::entry` — the one-probe occupied/vacant classification (the body that
/// grows, probes and consults the overflow), without the entry writes.
pub fn word_face_entry_is_occupied(
    mut state: Table<u64, i64>,
    key: u64,
) -> (Table<u64, i64>, bool) {
    let occupied = match state.entry(key) {
        Entry::Occupied(_) => true,
        Entry::Vacant(_) => false,
    };
    (state, occupied)
}

/// Real `Table::<u64, i64>::reserve` — size for `additional` more keys, one rebuild at most.
pub fn word_face_reserve(mut state: Table<u64, i64>, additional: usize) -> Table<u64, i64> {
    state.reserve(additional);
    state
}

// ── The regex span type and matcher (`tregex.rs`) ─────────────────────────────────────────────

/// Real `Match::start` — the span's start offset.
pub fn word_tregex_match_start(m: Match) -> usize {
    m.start()
}

/// Real `Match::end` — the span's exclusive end offset.
pub fn word_tregex_match_end(m: Match) -> usize {
    m.end()
}

/// Real `Match::range` — the span as a `Range<usize>`.
pub fn word_tregex_match_range(m: Match) -> std::ops::Range<usize> {
    m.range()
}

/// Real `term::is_word` — the `\b` byte predicate.
pub fn word_tregex_is_word(b: u8) -> bool {
    term::is_word(b)
}

/// Real `match_segs` — the linear segment program at `pos`, the engine's verify step.
pub fn word_tregex_match_segs(segs: &[term::Seg], hay: &[u8], pos: usize) -> Option<usize> {
    crate::hashtrinity::tregex::match_segs(segs, hay, pos)
}

// ── The face algebra (`algebra.rs`): the merges, the four value faces and the total read ──────
//
// The trait-declaration items (`Coord::coord`, `KeyStore::set`, `KeyReadout::key` and the rest)
// carry no body, so no wrapper can root them; the gate's name matcher also cannot see the
// monomorphized impl names (`U64.Insts.Apeiron_wordHashtrinityAlgebraCoord.coord`). The
// `FoldExt` collectors and `count_each` stay unrooted: their iterator/`_mm_prefetch` bodies are
// outside the pinned fragment. `mark`, `add`, `accumulate`, `push` and graph `link` stay
// unrooted: each calls `Table::insert_with` with a `fn` item or a closure, and the pinned
// translator answers "Arrow types are not supported yet" for every `impl FnOnce` argument.
// `neighbors` stays unrooted in both forms: `Vec::as_slice` as a `map_or` argument is an arrow
// type, and the `match` form's `None => &[]` arm is a bottom value. `fuse` stays unrooted: its
// `Table::insert_with` call instantiates `side::MergeGuard`, whose unwind `Drop` calls
// `alloc.vec.Vec.remove` through `Overflow::discard_entry`, and that external is not modeled.
// `FreqMap::default` stays unrooted: the derive reaches `PhantomData::default`, an unmodeled
// external the audit rejects.

/// Real `Face::species` — commutative (fusible) vs directional.
pub fn word_alg_species(f: Face) -> Species {
    f.species()
}

/// Real merge `keep` — set membership union.
pub fn word_alg_keep(new: bool, old: bool) -> bool {
    crate::hashtrinity::keep(new, old)
}

/// Real merge `set` — last write wins, rooted at the `u64` word.
pub fn word_alg_set(new: u64, old: u64) -> u64 {
    crate::hashtrinity::set::<u64>(new, old)
}

/// Real merge `count` — monoid combine, rooted at `i64`.
pub fn word_alg_merge_count(new: i64, old: i64) -> i64 {
    crate::hashtrinity::count::<i64>(new, old)
}

/// Real merge `link` — prepend the new list to the old.
pub fn word_alg_merge_link(new: Vec<u64>, old: Vec<u64>) -> Vec<u64> {
    crate::hashtrinity::link::<u64>(new, old)
}

/// Real `Table::<u64, i64>::put` — the map face's last-write-wins insert.
pub fn word_alg_put(mut state: Table<u64, i64>, key: u64, v: i64) -> Table<u64, i64> {
    state.put(key, v);
    state
}

/// Real `Table::<u64, bool>::has` — the set face's read.
pub fn word_alg_has(state: &Table<u64, bool>, key: u64) -> bool {
    state.has(&key)
}
/// Real `Table::<u64, i64>::count` — the count face's read.
pub fn word_alg_table_count(state: &Table<u64, i64>, key: u64) -> i64 {
    state.count(&key)
}

/// Real `Table::<u64, i64>::fold` — total read falling back to the monoid identity.
pub fn word_alg_fold(state: &Table<u64, i64>, key: u64) -> i64 {
    state.fold(&key)
}
// ── The count face's live surface (`freq.rs`): writes, sizing, and the probe diagnostic ───────

/// Real `FreqMap::<u64, i64>::stream_with_capacity` — the dense sizing alias.
pub fn word_freq_stream_with_capacity(cap: usize) -> FreqMap<u64, i64> {
    FreqMap::<u64, i64>::stream_with_capacity(cap)
}

/// Real `FreqMap::<u64, i64>::insert_with` — resolve a frequency clash with addition.
pub fn word_freq_insert_with(
    mut state: FreqMap<u64, i64>,
    key: u64,
    new: i64,
) -> FreqMap<u64, i64> {
    state.insert_with(&key, new, |a, b| a + b);
    state
}

/// Real `FreqMap::<u64, i64>::insert` — last write wins.
pub fn word_freq_insert(
    mut state: FreqMap<u64, i64>,
    key: u64,
    v: i64,
) -> FreqMap<u64, i64> {
    state.insert(&key, v);
    state
}
/// Real `FreqMap::<u64, i64>::longest_probe` — the worst primary group probe over live entries.
pub fn word_freq_longest_probe(state: &FreqMap<u64, i64>) -> usize {
    state.longest_probe()
}

/// Real `FreqMap::<u64, i64>::primary_examinations` — the diagnostic probe count for one key.
pub fn word_freq_primary_examinations(state: &FreqMap<u64, i64>, key: u64) -> u32 {
    state.primary_examinations(&key)
}
