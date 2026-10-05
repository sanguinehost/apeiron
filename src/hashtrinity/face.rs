//! The std-shaped faces of [`Table`] — the surface a *mechanical* `HashMap` port lands on.
//!
//! [`Table::insert_with`] is the whole algebra: one probe, one merge, four faces. That is the right
//! primitive and it is not what a porter reaches for. A porter reaches for `entry`, `get_mut`,
//! `insert`, `retain`, `for (k, v) in &map`, and — most expensively — `map.get(s)` where `s: &str`
//! and the key type is `String`. When those are absent the port still *compiles*, because the porter
//! writes the workaround: two probes instead of one, a `to_string()` per lookup, a rebuild instead
//! of a `retain`. Every one of those is slower than the `HashMap` it replaced, and none of them is a
//! user error — a primitive that is only fast in expert hands has failed at the layer that matters.
//!
//! Everything here is additive and expressed in the existing fold. Nothing re-implements the probe.
//!
//! # The two cliffs
//!
//! **Read-modify-write.** `*m.entry(k).or_insert(0) += 1` is ONE probe in std because `entry` hands
//! back the `&mut`. [`Table::entry`] does the same: probe once, hand back the slot.
//! [`Table::insert_fetch`] is the merge-shaped form of it.
//!
//! **Borrowed and composite keys.** `HashMap<String, V>` is probed with `&str` through
//! `Borrow<Q>`; `Table<String, V>`'s [`get`](Table::get) takes `&K`, so the mechanical port writes
//! `t.get(&s.to_string())` — an allocation on every lookup. Worse, `Table<(String, String), V>`
//! *compiles* when probed with an owned tuple, so the cost is two `String` clones per probe and no
//! compiler error to notice it. [`Locate`] is the fix: a borrowed view that addresses the same
//! coordinate as the owned key, so `(&str, &str)` probes a `(String, String)` table with **zero**
//! allocation. It extends the crate's existing tuple fold (`c ← c·PHI ⊕ xᵢ`) rather than adding a
//! second scheme — see [`Locate`] for the identity that makes that exact.
//!
//! ```ignore
//! use apeiron::hashtrinity::Table;
//! let mut t: Table<(String, String), i64> = Table::new();
//! t.put(("lo".into(), "hi".into()), 7);
//! assert_eq!(t.get_at(("lo", "hi")), Some(&7)); // no String materialized
//! *t.entry_at(("lo", "hi")).or_insert(0) += 1;  // one probe, no String materialized
//! assert_eq!(t.get_at(("lo", "hi")), Some(&8));
//! ```

// A `Locate` view IS the borrow — `&K`, `&str`, `(&str, &str)`. Taking it by value is the whole
// point (it is one or two machine words, and `Absorb` consumes it on the vacant path); taking it by
// reference would put a second indirection on the hot probe.
#![allow(clippy::needless_pass_by_value)]

use crate::hashtrinity::{
    bits_for, is_live, Coord, KeyReadout, KeyStore, Stored, Table, EMPTY, PHI, TOMB,
};
use std::mem::MaybeUninit;

// ── Locate / Absorb: a borrowed key that addresses the same slot as the owned one. ─────────────

/// A borrowed view of a key that lands on the **same coordinate** as the owned key `K`.
///
/// This is `Borrow<Q>`'s job in `std`, done through the fold instead of through `Hash`. The
/// contract is one equation:
///
/// > `q.coord() == owned.coord()` whenever `q` locates `owned`
///
/// and it holds *by construction*, not by convention, because every impl reduces to the coordinate
/// the crate already computes for that shape:
///
/// * `&K` → `K::coord` (the identity impl).
/// * `&str` → `fold_bytes(bytes)`, which is *literally* `<String as Coord>::coord`'s body.
/// * `(PA, PB)` → `pa.coord()·PHI ⊕ pb.coord()`, the SAME expression as
///   `<(A, B) as Coord>::coord`, with each component's coordinate supplied by that component's own
///   `Locate`. By induction the composite equals `<(A, B) as Coord>::coord` of the absorbed key.
///
/// So this is the existing tuple fold, not a second scheme, and nothing downstream of the `u64`
/// changes: the radix descent, the `⌈64/bits⌉` separation bound, the HashDOS immunity, and the
/// `2⁻⁶⁴` coordinate-identity risk are all untouched. `tests/locate.rs` pins the equation.   CITATION-EXEMPT: deleted by `d84bda3`, 25 crates → 12
///
/// [`same`](Locate::same) is the exact key compare, used only by the `_exact` faces — the ordinary
/// borrowed read trusts the coordinate exactly as [`Table::get`] does, and costs exactly the same.
pub trait Locate<K: Coord + Eq> {
    /// This view's coordinate — equal to `K::coord` of the key it locates.
    fn coord(&self) -> u64;
    /// Whether this view names exactly `k` (the exact compare; `_exact` faces only).
    fn same(&self, k: &K) -> bool;
}

/// A borrowed view that can be turned into the owned key — paid **only** when a probe misses and the
/// table must take ownership. On a hit, nothing is allocated.
pub trait Absorb<K: Coord + Eq>: Locate<K> {
    /// Materialize the owned key. Called on the vacant path and nowhere else.
    fn absorb(self) -> K;
}

impl<K: Coord + Eq> Locate<K> for &K {
    #[inline]
    fn coord(&self) -> u64 {
        (**self).coord()
    }
    #[inline]
    fn same(&self, k: &K) -> bool {
        *self == k
    }
}
impl<K: Coord + Eq + Clone> Absorb<K> for &K {
    #[inline]
    fn absorb(self) -> K {
        self.clone()
    }
}

impl Locate<String> for &str {
    #[inline]
    fn coord(&self) -> u64 {
        // `<str as Coord>::coord` and `<String as Coord>::coord` are the same `fold_bytes` call.
        <str as Coord>::coord(self)
    }
    #[inline]
    fn same(&self, k: &String) -> bool {
        *self == k.as_str()
    }
}
impl Absorb<String> for &str {
    #[inline]
    fn absorb(self) -> String {
        self.to_owned()
    }
}

impl<A: Coord + Eq, B: Coord + Eq, PA: Locate<A>, PB: Locate<B>> Locate<(A, B)> for (PA, PB) {
    #[inline]
    fn coord(&self) -> u64 {
        // Identical to `<(A, B) as Coord>::coord` — the crate's tuple fold, on borrowed parts.
        self.0.coord().wrapping_mul(PHI) ^ self.1.coord()
    }
    #[inline]
    fn same(&self, k: &(A, B)) -> bool {
        self.0.same(&k.0) && self.1.same(&k.1)
    }
}
impl<A: Coord + Eq, B: Coord + Eq, PA: Absorb<A>, PB: Absorb<B>> Absorb<(A, B)> for (PA, PB) {
    #[inline]
    fn absorb(self) -> (A, B) {
        (self.0.absorb(), self.1.absorb())
    }
}

impl<A: Coord + Eq, B: Coord + Eq, C: Coord + Eq, PA: Locate<A>, PB: Locate<B>, PC: Locate<C>>
    Locate<(A, B, C)> for (PA, PB, PC)
{
    #[inline]
    fn coord(&self) -> u64 {
        (self.0.coord().wrapping_mul(PHI) ^ self.1.coord()).wrapping_mul(PHI) ^ self.2.coord()
    }
    #[inline]
    fn same(&self, k: &(A, B, C)) -> bool {
        self.0.same(&k.0) && self.1.same(&k.1) && self.2.same(&k.2)
    }
}
impl<A: Coord + Eq, B: Coord + Eq, C: Coord + Eq, PA: Absorb<A>, PB: Absorb<B>, PC: Absorb<C>>
    Absorb<(A, B, C)> for (PA, PB, PC)
{
    #[inline]
    fn absorb(self) -> (A, B, C) {
        (self.0.absorb(), self.1.absorb(), self.2.absorb())
    }
}

/// The `EMPTY`/`TOMB` nudge, on a raw coordinate — the borrowed twin of `Table::coord_of`, which
/// takes a `&K` we deliberately never materialize here.
#[inline]
fn nudge(c: u64) -> u64 {
    if c == EMPTY || c == TOMB {
        TOMB - 1
    } else {
        c
    }
}

// ── Entry: the one-probe read-modify-write. ────────────────────────────────────────────────────

/// A single probe's result, held open so the caller can decide what to do with the slot. The std
/// `Entry` shape, on the fold's one probe: [`Table::entry`] walks the table exactly once, and
/// nothing here walks it again.
pub enum Entry<'a, K: Coord + Eq, V, KS: KeyStore<K>> {
    /// The key is present; the slot is live.
    Occupied(OccupiedEntry<'a, K, V, KS>),
    /// The key is absent; the slot is where it would go.
    Vacant(VacantEntry<'a, K, V, KS>),
}

/// A live slot, already located. Every method here is zero further probes.
pub struct OccupiedEntry<'a, K: Coord + Eq, V, KS: KeyStore<K>> {
    t: &'a mut Table<K, V, KS>,
    slot: usize,
    coord: u64,
}

/// The slot an absent key would occupy, already located, holding the key that would go there.
pub struct VacantEntry<'a, K: Coord + Eq, V, KS: KeyStore<K>> {
    t: &'a mut Table<K, V, KS>,
    slot: usize,
    coord: u64,
    key: K,
}

impl<'a, K: Coord + Eq, V, KS: KeyStore<K>> OccupiedEntry<'a, K, V, KS> {
    /// This slot's key — borrowed when stored, regenerated from the address when not.
    #[inline]
    pub fn key(&self) -> KS::Key<'_>
    where
        KS: KeyReadout<K>,
    {
        self.t.keys.key(self.slot, self.coord)
    }
    /// The value in the slot.
    #[inline]
    pub fn get(&self) -> &V {
        // SAFETY: an occupied entry names a live slot ⇒ its value is initialized.
        unsafe { self.t.vals.get_unchecked(self.slot).assume_init_ref() }
    }
    /// The value in the slot, mutably.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: an occupied entry names a live slot ⇒ its value is initialized.
        unsafe { self.t.vals.get_unchecked_mut(self.slot).assume_init_mut() }
    }
    /// The value in the slot, mutably, for as long as the table borrow lives.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        // SAFETY: an occupied entry names a live slot ⇒ its value is initialized.
        unsafe { self.t.vals.get_unchecked_mut(self.slot).assume_init_mut() }
    }
    /// Overwrite the value and return the old one. The `Set` face, at a located slot.
    #[inline]
    pub fn insert(&mut self, v: V) -> V {
        std::mem::replace(self.get_mut(), v)
    }
    /// Take the value out, tombstoning the slot (see [`Table::remove`]).
    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
    /// Take the key and the value out, tombstoning the slot.
    #[inline]
    pub fn remove_entry(self) -> (K, V) {
        let (i, c) = (self.slot, self.coord);
        // SAFETY: live slot ⇒ value initialized. Read it out and take the key, then tombstone: the
        // slot's value is now uninitialized and `TOMB` fails the liveness test, so Drop/Clone/iter
        // skip it — no double free, and the chain past it stays intact.
        let v = unsafe { self.t.vals.get_unchecked_mut(i).assume_init_read() };
        let k = self.t.keys.take_key(i, c);
        unsafe { *self.t.coords.get_unchecked_mut(i) = TOMB };
        self.t.len -= 1;
        self.t.tombs += 1;
        (k, v)
    }
}

impl<'a, K: Coord + Eq, V, KS: KeyStore<K>> VacantEntry<'a, K, V, KS> {
    /// The key that would be inserted here.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }
    /// Take the key back out (the entry is dropped).
    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }
    /// Fill the slot and hand back the value. The table was already sized for this by
    /// [`Table::entry`], so the slot index is still the right one — no second probe.
    #[inline]
    pub fn insert(self, v: V) -> &'a mut V {
        let (i, c) = (self.slot, self.coord);
        // SAFETY: `entry` grew the table BEFORE probing, so `i` is a valid EMPTY-or-TOMB slot in the
        // current arrays; writing the coordinate last would be the panic-safe order, but nothing
        // between here and there can panic.
        unsafe { *self.t.coords.get_unchecked_mut(i) = c };
        unsafe { *self.t.vals.get_unchecked_mut(i) = MaybeUninit::new(v) };
        self.t.keys.set(i, self.key); // Stored: write the key; Regen: no-op
        self.t.len += 1;
        // SAFETY: just written ⇒ initialized.
        unsafe { self.t.vals.get_unchecked_mut(i).assume_init_mut() }
    }
}

impl<'a, K: Coord + Eq, V, KS: KeyStore<K>> Entry<'a, K, V, KS> {
    /// The value, inserting `v` if the key was absent. `*t.entry(k).or_insert(0) += 1` — the RMW a
    /// porter writes, at ONE probe.
    #[inline]
    pub fn or_insert(self, v: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(v),
        }
    }
    /// The value, inserting `f()` if the key was absent (the default is not built on a hit).
    #[inline]
    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(f()),
        }
    }
    /// The value, inserting `f(&key)` if the key was absent.
    #[inline]
    pub fn or_insert_with_key(self, f: impl FnOnce(&K) -> V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let v = f(&e.key);
                e.insert(v)
            }
        }
    }
    /// The value, inserting `V::default()` if the key was absent.
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
    /// Run `f` on the value if the key was present, then carry on.
    #[must_use]
    #[inline]
    pub fn and_modify(mut self, f: impl FnOnce(&mut V)) -> Self {
        if let Entry::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }
}

// ── The table faces. ───────────────────────────────────────────────────────────────────────────

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// The grow law, hoisted out of `insert_slot` so [`entry`](Self::entry) can apply it BEFORE it
    /// probes — which is what makes the vacant fill a zero-probe operation instead of a re-probe.
    #[inline]
    fn grow_for_one(&mut self) {
        if (self.len + self.tombs + 1) * 8 >= self.coords.len() * 7 {
            let new_bits = if (self.len + 1) * 8 >= self.coords.len() * 7 {
                self.bits + 1
            } else {
                self.bits // merely crowded by tombstones: a same-size rebuild reclaims them
            };
            self.resize(new_bits);
        }
    }

    /// Locate `k` in ONE probe and hold the slot open. The std `entry` face:
    /// `*t.entry(k).or_insert(0) += 1` is one probe, one merge, no second lookup.
    ///
    /// Grows first, then probes, so [`VacantEntry::insert`] never has to re-probe.
    #[inline]
    pub fn entry(&mut self, k: K) -> Entry<'_, K, V, KS> {
        self.grow_for_one();
        let coord = Self::coord_of(&k);
        let (slot, found) = self.probe(coord);
        if found {
            Entry::Occupied(OccupiedEntry {
                t: self,
                slot,
                coord,
            })
        } else {
            Entry::Vacant(VacantEntry {
                t: self,
                slot,
                coord,
                key: k,
            })
        }
    }

    /// [`entry`](Self::entry) from a BORROWED key — the owned key is materialized only if the probe
    /// misses and the table has to take ownership. `t.entry_at(("a", "b")).or_insert(0)` on a
    /// `Table<(String, String), _>` allocates nothing on the hit path.
    #[inline]
    pub fn entry_at<Q: Absorb<K>>(&mut self, q: Q) -> Entry<'_, K, V, KS> {
        self.grow_for_one();
        let coord = nudge(q.coord());
        let (slot, found) = self.probe(coord);
        if found {
            Entry::Occupied(OccupiedEntry {
                t: self,
                slot,
                coord,
            })
        } else {
            Entry::Vacant(VacantEntry {
                t: self,
                slot,
                coord,
                key: q.absorb(), // the ONLY allocation, and only when the key is genuinely new
            })
        }
    }

    /// The value for `k`, mutably. One probe.
    #[inline]
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let coord = Self::coord_of(k);
        let (i, found) = self.probe(coord);
        // SAFETY: found ⇒ value initialized.
        found.then(|| unsafe { self.vals.get_unchecked_mut(i).assume_init_mut() })
    }

    /// [`get`](Self::get) from a borrowed key — `t.get_at("word")` on a `Table<String, _>`,
    /// `t.get_at((a, b))` on a `Table<(String, String), _>`. Zero allocation, one probe, and
    /// bit-for-bit the same slot the owned key would reach.
    #[inline]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        self.get_by_coord(nudge(q.coord()))
    }

    /// [`get_mut`](Self::get_mut) from a borrowed key.
    #[inline]
    pub fn get_mut_at<Q: Locate<K>>(&mut self, q: Q) -> Option<&mut V> {
        let (i, found) = self.probe(nudge(q.coord()));
        // SAFETY: found ⇒ value initialized.
        found.then(|| unsafe { self.vals.get_unchecked_mut(i).assume_init_mut() })
    }

    /// Whether `k` is present. One probe, no value touched.
    #[inline]
    pub fn contains_key(&self, k: &K) -> bool {
        self.probe(Self::coord_of(k)).1
    }

    /// Whether a borrowed key is present. One probe, no allocation.
    #[inline]
    pub fn contains_at<Q: Locate<K>>(&self, q: Q) -> bool {
        self.probe(nudge(q.coord())).1
    }

    /// Set `k` to `v` and return the value it replaced. The `Set` face with its old value handed
    /// back — std's `insert`. One probe.
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        match self.entry(k) {
            Entry::Occupied(mut e) => Some(e.insert(v)),
            Entry::Vacant(e) => {
                e.insert(v);
                None
            }
        }
    }

    /// [`insert`](Self::insert) from a borrowed key — no allocation unless the key is new.
    #[inline]
    pub fn insert_at<Q: Absorb<K>>(&mut self, q: Q, v: V) -> Option<V> {
        match self.entry_at(q) {
            Entry::Occupied(mut e) => Some(e.insert(v)),
            Entry::Vacant(e) => {
                e.insert(v);
                None
            }
        }
    }

    /// [`insert_fetch`](Self::insert_fetch) from a borrowed key: merge and hand back the result, in
    /// ONE probe, materializing the key only when it is new. This is the pair-counter shape —
    /// `counts[(a, b)] += 1; push(counts[(a, b)])` — with neither a second probe nor a `String`.
    #[inline]
    pub fn insert_fetch_at<Q: Absorb<K>>(&mut self, q: Q, new: V, f: impl FnOnce(V, V) -> V) -> &V {
        self.grow_for_one();
        let coord = nudge(q.coord());
        let (i, found) = self.probe(coord);
        if found {
            // SAFETY: found ⇒ value initialized. Read it out, then blank the coord: the slot is
            // logically empty for the duration of `f`, so a panic in `f` drops the value once (by
            // the unwinding `old`) and `Drop` skips this slot — the same order as `insert_slot`.
            unsafe {
                let old = self.vals.get_unchecked_mut(i).assume_init_read();
                *self.coords.get_unchecked_mut(i) = EMPTY;
                *self.vals.get_unchecked_mut(i) = MaybeUninit::new(f(new, old));
                *self.coords.get_unchecked_mut(i) = coord;
            }
        } else {
            // SAFETY: `grow_for_one` ran, so `i` is an EMPTY/TOMB slot in the current arrays.
            unsafe { *self.coords.get_unchecked_mut(i) = coord };
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
            self.keys.set(i, q.absorb()); // the ONLY allocation, and only for a genuinely new key
            self.len += 1;
        }
        // SAFETY: the slot is live either way ⇒ its value is initialized.
        unsafe { self.vals.get_unchecked(i).assume_init_ref() }
    }

    /// [`remove`](Self::remove) from a borrowed key.
    #[inline]
    pub fn remove_at<Q: Locate<K>>(&mut self, q: Q) -> Option<V> {
        let coord = nudge(q.coord());
        let (i, found) = self.probe(coord);
        if !found {
            return None;
        }
        // SAFETY: found ⇒ value initialized; see `Table::remove` for the tombstone argument.
        let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
        drop(self.keys.take_key(i, coord));
        unsafe { *self.coords.get_unchecked_mut(i) = TOMB };
        self.len -= 1;
        self.tombs += 1;
        Some(old)
    }

    /// Remove `k` and return both the key and the value.
    #[inline]
    pub fn remove_entry(&mut self, k: &K) -> Option<(K, V)> {
        self.entry_ref(k).map(OccupiedEntry::remove_entry)
    }

    /// The stored key and its value. `Regen` regenerates the key from the address.
    #[inline]
    pub fn get_key_value(&self, k: &K) -> Option<(KS::Key<'_>, &V)>
    where
        KS: KeyReadout<K>,
    {
        let coord = Self::coord_of(k);
        let (i, found) = self.probe(coord);
        if !found {
            return None;
        }
        // SAFETY: found ⇒ value initialized.
        Some((self.keys.key(i, coord), unsafe {
            self.vals.get_unchecked(i).assume_init_ref()
        }))
    }

    /// The occupied entry for `k`, if present — the read side of [`entry`](Self::entry) that never
    /// grows the table (no key is being added).
    #[inline]
    fn entry_ref(&mut self, k: &K) -> Option<OccupiedEntry<'_, K, V, KS>> {
        let coord = Self::coord_of(k);
        let (slot, found) = self.probe(coord);
        if found {
            Some(OccupiedEntry {
                t: self,
                slot,
                coord,
            })
        } else {
            None
        }
    }

    /// Iterate the live keys.
    pub fn keys(&self) -> impl Iterator<Item = KS::Key<'_>>
    where
        KS: KeyReadout<K>,
    {
        Iter::new(self).map(|(k, _)| k)
    }

    /// Iterate the live values.
    pub fn values(&self) -> impl Iterator<Item = &V>
    where
        KS: KeyReadout<K>,
    {
        Iter::new(self).map(|(_, v)| v)
    }

    /// Iterate the live values mutably.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        let coords = &self.coords;
        self.vals
            .iter_mut()
            .zip(coords.iter())
            .filter(|(_, c)| is_live(**c))
            // SAFETY: live coord ⇒ value initialized.
            .map(|(v, _)| unsafe { v.assume_init_mut() })
    }

    /// Iterate the live `(key, &mut value)` pairs.
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (KS::Key<'a>, &'a mut V)>
    where
        KS: KeyReadout<K> + 'a,
        K: 'a,
    {
        let (coords, keys) = (&self.coords, &self.keys);
        self.vals
            .iter_mut()
            .enumerate()
            .filter(move |(i, _)| is_live(coords[*i]))
            // SAFETY: live coord ⇒ value initialized.
            .map(move |(i, v)| (keys.key(i, coords[i]), unsafe { v.assume_init_mut() }))
    }

    /// Keep only the entries `keep` returns true for. Tombstones the rest — one pass, no rebuild.
    pub fn retain(&mut self, mut keep: impl FnMut(KS::Key<'_>, &mut V) -> bool)
    where
        KS: KeyReadout<K>,
    {
        for i in 0..self.coords.len() {
            let c = self.coords[i];
            if !is_live(c) {
                continue;
            }
            let alive = {
                // Disjoint field borrows: the key store is read, the value array is written.
                let k = self.keys.key(i, c);
                // SAFETY: live coord ⇒ value initialized.
                let v = unsafe { self.vals.get_unchecked_mut(i).assume_init_mut() };
                keep(k, v)
            };
            if !alive {
                // SAFETY: live coord ⇒ value initialized; dropped exactly once, then the slot is
                // tombstoned so nothing reads it again.
                unsafe { self.vals.get_unchecked_mut(i).assume_init_drop() };
                drop(self.keys.take_key(i, c));
                self.coords[i] = TOMB;
                self.len -= 1;
                self.tombs += 1;
            }
        }
    }

    /// Drop every entry, keeping the allocation.
    pub fn clear(&mut self) {
        if self.len == 0 && self.tombs == 0 {
            return;
        }
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.coords.iter().zip(self.vals.iter_mut()) {
                if is_live(*c) {
                    // SAFETY: live coord ⇒ value initialized.
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        // The key store rides the same liveness line, which it does not own — hand it over.
        let mut coords = std::mem::take(&mut self.coords);
        self.keys.drop_live(&coords);
        coords.fill(EMPTY);
        self.coords = coords;
        self.len = 0;
        self.tombs = 0;
    }

    /// Make room for `additional` more keys without growing again. The same one sizing as
    /// [`Table::with_capacity`] — a no-op when the table is already big enough.
    pub fn reserve(&mut self, additional: usize) {
        let want = bits_for(self.len + additional);
        if want > self.bits {
            self.resize(want);
        }
    }

    /// Empty the table into an iterator of its `(key, value)` pairs, keeping the table usable.
    pub fn drain(&mut self) -> IntoIter<K, V, KS> {
        std::mem::replace(self, Self::with_bits(3)).into_iter()
    }

    /// `get` from an already-nudged coordinate — the shared body of the borrowed reads. Same shape
    /// as the owned path: the near-certain home hit, then the shared collision tail.
    #[inline]
    fn get_by_coord(&self, coord: u64) -> Option<&V> {
        let home = self.home(coord);
        // SAFETY: home ∈ [0, len).
        let sc = unsafe { *self.coords.get_unchecked(home) };
        if sc == coord {
            // SAFETY: a coord match ⇒ value initialized.
            return Some(unsafe { self.vals.get_unchecked(home).assume_init_ref() });
        }
        if sc == EMPTY {
            return None;
        }
        self.get_coord_cold(coord, home)
    }
}

/// The exact borrowed read — only on the [`Stored`] key store, which is the only one that HAS a key
/// to compare. On [`Regen`](crate::hashtrinity::Regen) the coordinate round-trips to the key, so
/// [`get_at`](Table::get_at) already IS the exact read
/// (`CoordinateReadout.exact_read_is_read`) — except at the `EMPTY`/`TOMB` nudge, and no borrowed
/// view can help there either.
impl<K: Eq + Coord, V> Table<K, V, Stored<K>> {
    /// [`get_exact`](Table::get_exact) from a borrowed key — also compares the stored key, so a
    /// `2⁻⁶⁴` coordinate collision reads as absent instead of returning someone else's value.
    #[inline]
    pub fn get_exact_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        let coord = nudge(q.coord());
        let (i, found) = self.probe(coord);
        if found && q.same(self.keys.key(i, coord)) {
            // SAFETY: found ⇒ value initialized.
            return Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() });
        }
        None
    }
}

// ── The live-lane scan: the read side of iteration, branchless. ────────────────────────────────
//
// A drain (`for (k, v) in &map`) walks every SLOT, not every entry, and at the grow law's steady
// state half of them are empty. The obvious loop — `if is_live(coords[i])` per slot — puts a
// ~50/50 unpredictable branch on that walk, and a mispredict costs ~15 cycles where the load costs
// ~0.5. That is the whole of the measured 4.4x iteration loss to std/hashbrown, and it is NOT a
// bandwidth story: the coordinate line is 8 B/slot against SwissTable's 1 B/slot control array, a
// factor 8 in traffic that would explain 8x if traffic were the binding constraint — it explains
// almost none of it, because the scan is branch-bound, not memory-bound (measured: 2.4 ns/slot,
// ~10 cycles, where a streaming 8 B/slot read is ~0.5).
//
// So take SwissTable's ACTUAL trick, which is not the byte array — it is the movemask. Glance eight
// coordinates (one 64-byte cache line) against `EMPTY` and `TOMB` in two AVX2 compares, invert to a
// live-lane bitmask, and walk the SET BITS. One branch per eight slots instead of one per slot, and
// the empty slots cost a shift each. No new array, no control byte, no layout change, so `iter()`
// keeps its signature, its slot order, and its footprint.

/// The live-lane bitmask for the eight coordinates at `base` — bit `k` set iff slot `base + k` holds
/// a live entry. `bits >= 3` and the slot count is a power of two, so a table is always a whole
/// number of eight-slot blocks and there is no tail case.
#[inline]
fn live_mask8(coords: &[u64], base: usize) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: the caller only calls this with `base + 8 <= coords.len()`.
    return unsafe { live_mask8_avx2(coords.as_ptr().add(base)) };

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the caller only calls this with `base + 8 <= coords.len()`.
            return unsafe { live_mask8_avx2(coords.as_ptr().add(base)) };
        }
        let mut m = 0u32;
        for k in 0..8 {
            // branchless by construction — the compiler autovectorizes this shape
            m |= u32::from(is_live(coords[base + k])) << k;
        }
        m
    }
}

/// AVX2 body of [`live_mask8`]: two 256-bit loads (one 64-byte cache line), and ONE `vpcmpgtq` each.
///
/// `is_live(c)` is `c < TOMB` (the sentinels are the top two `u64`s), so a lane is live iff its
/// coordinate is *below* `TOMB`. AVX2 has no unsigned 64-bit compare, so flip the sign bit — an
/// order-preserving `u64 → i64` map — and compare signed against `TOMB` flipped: live ⟺
/// `flip(TOMB) > flip(c)`. Two vector ops per four slots (`vpxor`, `vpcmpgtq`) where the
/// equality form needed four (`vpcmpeqq` ×2, `vpor`, and the final invert).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn live_mask8_avx2(p: *const u64) -> u32 {
    use std::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_movemask_pd,
        _mm256_set1_epi64x, _mm256_xor_si256,
    };
    // SAFETY: caller guarantees 8 readable lanes at `p`.
    let (lo, hi) = unsafe {
        (
            _mm256_loadu_si256(p.cast::<__m256i>()),
            _mm256_loadu_si256(p.add(4).cast::<__m256i>()),
        )
    };
    let sign = _mm256_set1_epi64x(i64::MIN);
    let bound = _mm256_set1_epi64x((TOMB ^ (1u64 << 63)) as i64); // flip(TOMB)
    let live = |v: __m256i| -> u32 {
        let f = _mm256_xor_si256(v, sign);
        _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(bound, f))) as u32
    };
    live(lo) | (live(hi) << 4)
}

/// The eight-slot cursor: `base` is the block whose `mask` is loaded, `next` the block after it.
#[derive(Clone, Copy)]
struct Lanes {
    base: usize,
    next: usize,
    mask: u32,
}

impl Lanes {
    const START: Lanes = Lanes {
        base: 0,
        next: 0,
        mask: 0,
    };
    /// The next live slot index, or `None` at the end. One branch per eight slots.
    #[inline(always)]
    fn step(&mut self, coords: &[u64]) -> Option<usize> {
        while self.mask == 0 {
            if self.next >= coords.len() {
                return None;
            }
            self.base = self.next;
            self.next += 8;
            self.mask = live_mask8(coords, self.base);
        }
        let k = self.mask.trailing_zeros() as usize;
        self.mask &= self.mask - 1; // clear the lowest set bit
        Some(self.base + k)
    }
    /// How many live slots are left at or after the cursor — for `size_hint`/`Drop`.
    fn remaining(&self, coords: &[u64]) -> usize {
        let mut c = self.mask.count_ones() as usize;
        let mut b = self.next;
        while b < coords.len() {
            c += live_mask8(coords, b).count_ones() as usize;
            b += 8;
        }
        c
    }
}

// ── Iteration: `for (k, v) in &t`, `for (k, v) in t`, `t.extend(..)`. ──────────────────────────

/// `(key, &value)` over the live slots — the `IntoIterator` form of [`Table::iter`], on the
/// eight-slot live-lane glance.
pub struct Iter<'a, K: Coord + Eq, V, KS: KeyStore<K>> {
    t: &'a Table<K, V, KS>,
    lanes: Lanes,
    seen: usize,
}

impl<'a, K: Coord + Eq, V, KS: KeyStore<K>> Iter<'a, K, V, KS> {
    /// The walker for `t` — used by [`Table::iter`] itself.
    #[inline]
    pub(crate) fn new(t: &'a Table<K, V, KS>) -> Self {
        Iter {
            t,
            lanes: Lanes::START,
            seen: 0,
        }
    }
}

impl<'a, K: Coord + Eq, V, KS: KeyReadout<K>> Iterator for Iter<'a, K, V, KS>
where
    KS: 'a,
    K: 'a,
{
    type Item = (KS::Key<'a>, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let t: &'a Table<K, V, KS> = self.t; // copy the borrow so the yielded key lives for 'a
        let i = self.lanes.step(&t.coords)?;
        self.seen += 1;
        // SAFETY: `step` only yields in-bounds live lanes ⇒ this slot's value (and, for `Stored`,
        // its key) is initialized, and the index needs no bounds check.
        let c = unsafe { *t.coords.get_unchecked(i) };
        Some((t.keys.key(i, c), unsafe {
            t.vals.get_unchecked(i).assume_init_ref()
        }))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.t.len - self.seen;
        (left, Some(left))
    }
}

impl<K: Coord + Eq, V, KS: KeyReadout<K>> ExactSizeIterator for Iter<'_, K, V, KS> {}

impl<'a, K: Coord + Eq, V, KS: KeyReadout<K> + 'a> IntoIterator for &'a Table<K, V, KS>
where
    K: 'a,
{
    type Item = (KS::Key<'a>, &'a V);
    type IntoIter = Iter<'a, K, V, KS>;
    fn into_iter(self) -> Iter<'a, K, V, KS> {
        Iter::new(self)
    }
}

/// Owned `(key, value)` pairs — [`Table::drain`] and `for (k, v) in table`.
pub struct IntoIter<K: Coord + Eq, V, KS: KeyStore<K>> {
    coords: Box<[u64]>,
    vals: Box<[MaybeUninit<V>]>,
    keys: KS,
    lanes: Lanes,
    _k: std::marker::PhantomData<K>,
}

impl<K: Coord + Eq, V, KS: KeyStore<K>> Iterator for IntoIter<K, V, KS> {
    type Item = (K, V);
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        let i = self.lanes.step(&self.coords)?;
        // SAFETY: `step` only yields in-bounds live lanes.
        let c = unsafe { *self.coords.get_unchecked(i) };
        // SAFETY: a live lane ⇒ value and (for Stored) key initialized. Both are moved out exactly
        // once, then the slot is blanked so this iterator's Drop skips it.
        let v = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
        let k = self.keys.take_key(i, c);
        self.coords[i] = EMPTY;
        Some((k, v))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.lanes.remaining(&self.coords);
        (left, Some(left))
    }
}

impl<K: Coord + Eq, V, KS: KeyStore<K>> Drop for IntoIter<K, V, KS> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.coords.iter().zip(self.vals.iter_mut()) {
                if is_live(*c) {
                    // SAFETY: still-live coord ⇒ never yielded ⇒ value initialized.
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        let coords = std::mem::take(&mut self.coords);
        self.keys.drop_live(&coords);
    }
}

impl<K: Coord + Eq, V, KS: KeyStore<K>> IntoIterator for Table<K, V, KS> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, KS>;
    fn into_iter(self) -> IntoIter<K, V, KS> {
        // Move the three arrays out and suppress the table's own Drop: the iterator owns them now.
        let mut t = std::mem::ManuallyDrop::new(self);
        let coords = std::mem::take(&mut t.coords);
        let vals = std::mem::replace(&mut t.vals, Box::new([]));
        let keys = std::mem::replace(&mut t.keys, KS::alloc(0));
        IntoIter {
            coords,
            vals,
            keys,
            lanes: Lanes::START,
            _k: std::marker::PhantomData,
        }
    }
}

impl<K: Coord + Eq, V, KS: KeyStore<K>> Extend<(K, V)> for Table<K, V, KS> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let it = iter.into_iter();
        self.reserve(it.size_hint().0);
        for (k, v) in it {
            self.put(k, v);
        }
    }
}

impl<'x, K: Coord + Eq + Clone, V: Clone, KS: KeyStore<K>> Extend<(&'x K, &'x V)>
    for Table<K, V, KS>
{
    fn extend<I: IntoIterator<Item = (&'x K, &'x V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.put(k.clone(), v.clone());
        }
    }
}

impl<Q: Locate<K>, K: Coord + Eq, V, KS: KeyStore<K>> std::ops::Index<Q> for Table<K, V, KS> {
    type Output = V;
    /// # Panics
    /// If the key is absent — std `HashMap`'s `Index` contract.
    fn index(&self, q: Q) -> &V {
        self.get_at(q).expect("no entry found for key")
    }
}
