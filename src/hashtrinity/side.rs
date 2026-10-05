//! The side store for reserved-coordinate keys (`EMPTY`/`TOMB`), moved out of `mod.rs` to keep it
//! under the 1500-line bound. Row: `apeiron-ht-identity-BvOqNvKHHfxZS7sswjc6j6`.

use super::*;

/// The side store for keys whose own coordinate is a reserved marker (`EMPTY` or `TOMB`). Such a key
/// cannot live in `coords` — its coordinate would alias the occupancy flag — so its identity is held
/// here, outside the slot arrays. `coords[i]` is the reserved coordinate, `vals[i]` the value
/// (initialized iff `live[i]`; `live[i]` is false only while a merge owns the value), and `keys` the
/// identity exactly as the main store carries it: the full key for `Stored`, nothing for `Regen`
/// (which regenerates the key from `coords[i]`, and is admitted only under an injective coordinate —
/// [`CoordInv`]). The store is packed and scanned linearly; it holds only the keys that literally ask
/// for a sentinel, so it is empty (one zero-length `Vec` and a zero-sized key store) on every
/// ordinary table. Capacity is O(side entries), never O(table slots).
pub(super) struct SideStore<K: Coord, V, KS: KeyStore<K>> {
    pub(super) coords: Vec<u64>,
    pub(super) vals: Vec<MaybeUninit<V>>,
    pub(super) live: Vec<bool>,
    pub(super) keys: KS, // allocated to `cap`; entries `0..coords.len()` initialized
    pub(super) cap: usize,
    pub(super) _k: std::marker::PhantomData<K>,
}

impl<K: Coord, V, KS: KeyStore<K>> SideStore<K, V, KS> {
    pub(super) fn empty() -> Self {
        SideStore {
            coords: Vec::new(),
            vals: Vec::new(),
            live: Vec::new(),
            keys: KS::alloc(0),
            cap: 0,
            _k: std::marker::PhantomData,
        }
    }

    /// Append `(coord, k, v)`, growing the key store's fixed-capacity allocation by doubling. The
    /// values and coordinates live in plain `Vec`s and grow themselves.
    pub(super) fn push(&mut self, coord: u64, k: K, v: V) {
        let need = self.coords.len() + 1;
        if need > self.cap {
            let new_cap = (self.cap * 2).max(4).max(need);
            let mut grown = KS::alloc(new_cap);
            for i in 0..self.coords.len() {
                let c = self.coords[i];
                let key = self.keys.take_key(i, c);
                grown.set(i, key);
            }
            self.keys = grown;
            self.cap = new_cap;
        }
        let i = self.coords.len();
        self.keys.set(i, k);
        self.coords.push(coord);
        self.vals.push(MaybeUninit::new(v));
        self.live.push(true);
    }

    /// The entry holding key `k`: coordinate equality plus the key compare (`matches` is the
    /// coordinate round-trip for `Regen`), so distinct keys with equal reserved coordinates stay
    /// distinct here exactly as they do in the main array.
    pub(super) fn find(&self, coord: u64, k: &K) -> Option<usize> {
        let mut i = 0;
        while i < self.coords.len() {
            if self.coords[i] == coord && self.keys.matches(i, coord, k) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Remove entry `i`; its value must already have been moved out by the caller. The removed key is
    /// dropped, and the last entry is swapped into `i` — its key explicitly, since `KS` has no swap.
    pub(super) fn remove_entry(&mut self, i: usize) {
        let gone = self.keys.take_key(i, self.coords[i]);
        let last = self.coords.len() - 1;
        if i != last {
            let moved = self.keys.take_key(last, self.coords[last]);
            self.keys.set(i, moved);
            self.coords.swap(i, last);
            self.live.swap(i, last);
            self.vals.swap(i, last);
        }
        self.coords.pop();
        self.live.pop();
        self.vals.pop();
        drop(gone);
    }

    /// Move entry `i`'s `(coord, key, value)` out and swap-remove it. The caller owns all three;
    /// nothing is dropped here. The owner-consuming faces (`fuse`, `drain`, `IntoIterator`) take an
    /// entry apart with this rather than leave it in the store.
    pub(super) fn take_entry(&mut self, i: usize) -> (u64, K, V) {
        let c = self.coords[i];
        // SAFETY: entry `i` is occupied ⇒ its key and value are initialized.
        let key = self.keys.take_key(i, c);
        let val = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
        let last = self.coords.len() - 1;
        if i != last {
            let moved = self.keys.take_key(last, self.coords[last]);
            self.keys.set(i, moved);
        }
        self.coords.swap(i, last);
        self.live.swap(i, last);
        self.vals.swap(i, last);
        self.coords.pop();
        self.live.pop();
        self.vals.pop();
        (c, key, val)
    }

    /// Drop every entry, keeping the key store's allocation — the side twin of `Table::clear`.
    pub(super) fn clear(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (v, live) in self.vals.iter_mut().zip(self.live.iter()) {
                if *live {
                    // SAFETY: `live[i]` ⇒ value initialized.
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        for i in 0..self.coords.len() {
            let c = self.coords[i];
            drop(self.keys.take_key(i, c));
        }
        self.coords.clear();
        self.live.clear();
        self.vals.clear();
    }
}
