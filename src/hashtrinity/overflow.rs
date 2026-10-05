// Row: `apeiron-hashdos-cap-B8tcM9Je2duKmuNmBjvHzM`: the deterministic overflow behind the probe cap.
//! The deterministic overflow store for [`Table`](super::Table) — one table-wide AVL tree in an
//! arena, keyed by a key's *coordinate*.
//!
//! Row: `apeiron-hashdos-cap-B8tcM9Je2duKmuNmBjvHzM`.
//!
//! The primary probe is capped at [`PROBE_CAP`](super::PROBE_CAP) slot examinations
//! ([`Table::probe_limit`](super::Table)). A key whose whole capped prefix held no `EMPTY` slot and
//! no identity is *blocked*: it is stored here instead, and every capped miss consults this store.
//! Placement is deterministic (no seed); the tree is balanced so the depth is `O(log n)` even when
//! an adversary aims every key at one home.
//!
//! **The invariant.** A key is here iff, when it was placed, its `PROBE_CAP` primary slots held no
//! `EMPTY` and no slot carrying its identity. Primary slots only ever change from `EMPTY` to
//! occupied (`TOMB`, not `EMPTY`, is what `remove` leaves; `resize` re-places everything through
//! this same capped placement), so the property holds for the life of a table generation. Hence on
//! lookup: a hit in the capped prefix is found; an `EMPTY` in it means absent (no lookup here); the
//! prefix exhausted means consult this store.
//!
//! **Arena, not `Box`.** Nodes live in one `Vec` indexed by `u32`; children are indices and a free
//! list recycles removed nodes. No `unsafe`, no recursive `Box`. The tree is AVL: every node's
//! subtree heights differ by at most one, so [`height`](Overflow::height) is `O(log len)`.
//!
//! **Coordinate key, full-key identity.** Entries with equal coordinates share a node (a `Coord`
//! that is not injective makes two distinct keys collide here exactly as they do in the primary
//! array; `Regen` keys, whose coordinate round-trips, have at most one entry per node).

use super::{
    is_sentinel, prefix_masks, Capped, Coord, KeyStore, Locate, Table, EMPTY, LINEAR_G, PROBE_CAP,
};

/// The `NIL` child index — no node. `u32::MAX` is not a valid arena index (the arena would need
/// `2^32` live nodes).
const NIL: u32 = u32::MAX;

/// One AVL node: a coordinate, the entries sharing it, and the child indices.
#[derive(Clone)]
pub(super) struct Node<K, V> {
    pub(super) key: u64,
    /// The entries at this coordinate. The `Option` is `None` only for the duration of a merge that
    /// has moved the value out (panic safety, exactly as the primary slot blanks its coordinate);
    /// every steady-state entry is `Some`.
    pub(super) entries: Vec<(K, Option<V>)>,
    pub(super) left: u32,
    pub(super) right: u32,
    /// Subtree height: `0` for `NIL`, `1` for a leaf.
    pub(super) height: i32,
    /// Whether this arena slot currently holds a tree node (a freed slot is reused).
    pub(super) alive: bool,
}

/// The overflow store: an AVL tree in a `Vec` arena. No bounds on `K`/`V` at the type level so it
/// can be a field of [`Table<K, V, KS>`](super::Table) under the table's own `K: Coord` bound; the
/// coordinate-comparing operations carry `K: Eq`.
#[derive(Clone)]
pub(super) struct Overflow<K: Coord, V> {
    pub(super) nodes: Vec<Node<K, V>>,
    free: Vec<u32>,
    root: u32,
    len: usize,
}

impl<K: Coord, V> Default for Overflow<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Coord, V> Overflow<K, V> {
    pub(super) fn new() -> Self {
        Overflow {
            nodes: Vec::new(),
            free: Vec::new(),
            root: NIL,
            len: 0,
        }
    }

    /// The number of stored entries (not nodes): `len()` for the table.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// The tree height: `0` empty, `1` a single node. `O(log len)` by the AVL invariant.
    pub(super) fn height(&self) -> u32 {
        self.h(self.root).max(0) as u32
    }

    fn h(&self, i: u32) -> i32 {
        if i == NIL {
            0
        } else {
            self.nodes[i as usize].height
        }
    }

    fn update(&mut self, i: u32) {
        let l = self.nodes[i as usize].left;
        let r = self.nodes[i as usize].right;
        let hl = self.h(l);
        let hr = self.h(r);
        self.nodes[i as usize].height = 1 + hl.max(hr);
    }

    fn rotate_right(&mut self, y: u32) -> u32 {
        let x = self.nodes[y as usize].left;
        let t = self.nodes[x as usize].right;
        self.nodes[x as usize].right = y;
        self.nodes[y as usize].left = t;
        self.update(y);
        self.update(x);
        x
    }

    fn rotate_left(&mut self, x: u32) -> u32 {
        let y = self.nodes[x as usize].right;
        let t = self.nodes[y as usize].left;
        self.nodes[y as usize].left = x;
        self.nodes[x as usize].right = t;
        self.update(x);
        self.update(y);
        y
    }

    /// Restore the AVL invariant at `i` (whose children's heights are current), returning the root
    /// of the rebalanced subtree.
    fn rebalance(&mut self, i: u32) -> u32 {
        self.update(i);
        let l = self.nodes[i as usize].left;
        let r = self.nodes[i as usize].right;
        let bf = self.h(l) - self.h(r);
        if bf > 1 {
            let lr = self.nodes[l as usize].right;
            if self.h(lr) > self.h(self.nodes[l as usize].left) {
                let nl = self.rotate_left(l);
                self.nodes[i as usize].left = nl;
            }
            return self.rotate_right(i);
        }
        if bf < -1 {
            let rl = self.nodes[r as usize].left;
            if self.h(rl) > self.h(self.nodes[r as usize].right) {
                let nr = self.rotate_right(r);
                self.nodes[i as usize].right = nr;
            }
            return self.rotate_left(i);
        }
        i
    }

    /// Rebalance a root-to-deepest path bottom-up, re-linking each parent to the rotated subtree
    /// root. `chain[0]` is the tree root (its new root is written to `self.root`).
    fn rebalance_chain(&mut self, chain: &[u32]) {
        for idx in (0..chain.len()).rev() {
            let n = chain[idx];
            let newroot = self.rebalance(n);
            if idx == 0 {
                self.root = newroot;
            } else {
                let p = chain[idx - 1];
                if self.nodes[p as usize].left == n {
                    self.nodes[p as usize].left = newroot;
                } else {
                    self.nodes[p as usize].right = newroot;
                }
            }
        }
    }

    fn alloc(&mut self, key: u64, k: K, v: V) -> u32 {
        let node = Node {
            key,
            entries: vec![(k, Some(v))],
            left: NIL,
            right: NIL,
            height: 1,
            alive: true,
        };
        if let Some(i) = self.free.pop() {
            self.nodes[i as usize] = node;
            i
        } else {
            self.nodes.push(node);
            (self.nodes.len() - 1) as u32
        }
    }

    fn free_node(&mut self, i: u32) {
        let n = &mut self.nodes[i as usize];
        n.entries.clear();
        n.left = NIL;
        n.right = NIL;
        n.height = 0;
        n.alive = false;
        self.free.push(i);
    }

    fn find_node(&self, coord: u64) -> u32 {
        let mut cur = self.root;
        while cur != NIL {
            let ck = self.nodes[cur as usize].key;
            if coord == ck {
                return cur;
            }
            cur = if coord < ck {
                self.nodes[cur as usize].left
            } else {
                self.nodes[cur as usize].right
            };
        }
        NIL
    }

    /// The entry with coordinate `coord` and full key `k` — coordinate equality plus the exact key
    /// compare (a `Coord` that is not injective keeps its two distinct keys distinct here).
    pub(super) fn find_entry(&self, coord: u64, k: &K) -> Option<(u32, u32)>
    where
        K: Eq,
    {
        let n = self.find_node(coord);
        if n == NIL {
            return None;
        }
        self.nodes[n as usize]
            .entries
            .iter()
            .position(|(ek, _)| ek == k)
            .map(|e| (n, e as u32))
    }

    /// The entry a borrowed view `q` names (coordinate plus the view's own compare).
    pub(super) fn find_locate<Q: Locate<K>>(&self, coord: u64, q: &Q) -> Option<(u32, u32)>
    where
        K: Eq,
    {
        let n = self.find_node(coord);
        if n == NIL {
            return None;
        }
        self.nodes[n as usize]
            .entries
            .iter()
            .position(|(ek, _)| q.same(ek))
            .map(|e| (n, e as u32))
    }

    pub(super) fn get(&self, coord: u64, k: &K) -> Option<&V>
    where
        K: Eq,
    {
        let (n, e) = self.find_entry(coord, k)?;
        self.nodes[n as usize].entries[e as usize].1.as_ref()
    }

    pub(super) fn get_mut(&mut self, coord: u64, k: &K) -> Option<&mut V>
    where
        K: Eq,
    {
        let (n, e) = self.find_entry(coord, k)?;
        self.nodes[n as usize].entries[e as usize].1.as_mut()
    }

    /// Insert a genuinely new entry (the caller has established absence). Returns a borrow of the
    /// stored value.
    pub(super) fn insert_new(&mut self, coord: u64, k: K, v: V) -> &mut V {
        self.len += 1;
        let n = self.find_node(coord);
        let node = if n == NIL {
            let i = self.alloc(coord, k, v);
            self.insert_node(i);
            i
        } else {
            self.nodes[n as usize].entries.push((k, Some(v)));
            n
        };
        let e = self.nodes[node as usize].entries.len() - 1;
        self.nodes[node as usize].entries[e].1.as_mut().unwrap()
    }

    /// Upsert: merge `new` into the running value at `(coord, k)` when present, else insert.
    pub(super) fn upsert(&mut self, coord: u64, k: K, new: V, f: impl FnOnce(V, V) -> V) -> &mut V
    where
        K: Eq,
    {
        if let Some((n, e)) = self.find_entry(coord, &k) {
            let old = self.nodes[n as usize].entries[e as usize]
                .1
                .take()
                .expect("overflow: live entry");
            let merged = f(new, old);
            self.nodes[n as usize].entries[e as usize].1 = Some(merged);
            return self.nodes[n as usize].entries[e as usize].1.as_mut().unwrap();
        }
        self.insert_new(coord, k, new)
    }

    /// Move the value out of an existing entry, leaving it `None` for the duration of a merge.
    pub(super) fn take_value(&mut self, n: u32, e: u32) -> V {
        self.nodes[n as usize].entries[e as usize]
            .1
            .take()
            .expect("overflow: live entry")
    }

    /// Write a merged value back into an entry blanked by [`take_value`](Self::take_value).
    pub(super) fn write_back(&mut self, n: u32, e: u32, v: V) {
        self.nodes[n as usize].entries[e as usize].1 = Some(v);
    }

    /// Remove the entry at `(n, e)` and return it. Removes the node when it becomes empty.
    pub(super) fn take_entry(&mut self, n: u32, e: u32) -> (K, V) {
        let (k, v) = self.nodes[n as usize].entries.remove(e as usize);
        self.len -= 1;
        if self.nodes[n as usize].entries.is_empty() {
            self.delete_node(n);
        }
        (k, v.expect("overflow: live entry"))
    }

    /// Remove `(coord, k)` and return its value.
    pub(super) fn remove(&mut self, coord: u64, k: &K) -> Option<V>
    where
        K: Eq,
    {
        let (n, e) = self.find_entry(coord, k)?;
        Some(self.take_entry(n, e).1)
    }

    /// Whether any entry at `coord` has value equal to `value` — the `PartialEq` scan.
    pub(super) fn contains_value(&self, coord: u64, value: &V) -> bool
    where
        V: PartialEq,
    {
        let n = self.find_node(coord);
        if n == NIL {
            return false;
        }
        self.nodes[n as usize]
            .entries
            .iter()
            .any(|(_, v)| v.as_ref() == Some(value))
    }

    /// Every live entry in node-key order, as `(coordinate, &key, &value)`.
    pub(super) fn iter(&self) -> impl Iterator<Item = (u64, &K, &V)> + '_ {
        let mut order = Vec::new();
        self.in_order(&mut order);
        order.into_iter().flat_map(move |n| {
            let node = &self.nodes[n as usize];
            node.entries
                .iter()
                .filter_map(move |(k, v)| v.as_ref().map(|v| (node.key, k, v)))
        })
    }

    /// Consume the store, yielding every live `(coordinate, key, value)`.
    pub(super) fn into_entries(mut self) -> Vec<(u64, K, V)> {
        let mut out = Vec::with_capacity(self.len);
        let mut order = Vec::new();
        self.in_order(&mut order);
        for n in order {
            let node = &mut self.nodes[n as usize];
            let key = node.key;
            for (k, v) in node.entries.drain(..) {
                if let Some(v) = v {
                    out.push((key, k, v));
                }
            }
        }
        out
    }

    /// Keep only the entries `keep` returns true for. Rebuilds the tree (not a hot path).
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        let old = std::mem::replace(self, Overflow::new());
        for (coord, k, mut v) in old.into_entries() {
            if keep(&k, &mut v) {
                self.insert_new(coord, k, v);
            }
        }
    }

    /// The in-order node indices, iteratively (no recursion).
    pub(super) fn in_order(&self, out: &mut Vec<u32>) {
        let mut stack: Vec<u32> = Vec::new();
        let mut cur = self.root;
        while cur != NIL || !stack.is_empty() {
            while cur != NIL {
                stack.push(cur);
                cur = self.nodes[cur as usize].left;
            }
            let n = stack.pop().expect("nonempty stack");
            if self.nodes[n as usize].alive {
                out.push(n);
            }
            cur = self.nodes[n as usize].right;
        }
    }

    /// BST insertion of the fresh node `i` (its key is not yet in the tree) followed by a bottom-up
    /// rebalance.
    fn insert_node(&mut self, i: u32) {
        if self.root == NIL {
            self.root = i;
            return;
        }
        let key = self.nodes[i as usize].key;
        let mut chain: Vec<u32> = Vec::with_capacity(64);
        let mut cur = self.root;
        loop {
            chain.push(cur);
            let ck = self.nodes[cur as usize].key;
            if key < ck {
                let l = self.nodes[cur as usize].left;
                if l == NIL {
                    self.nodes[cur as usize].left = i;
                    break;
                }
                cur = l;
            } else {
                let r = self.nodes[cur as usize].right;
                if r == NIL {
                    self.nodes[cur as usize].right = i;
                    break;
                }
                cur = r;
            }
        }
        self.rebalance_chain(&chain);
    }

    /// Physically remove node `target` from the tree (its entries are already empty) and rebalance.
    fn delete_node(&mut self, target: u32) {
        // Ancestors of `target`, root first.
        let mut chain: Vec<u32> = Vec::new();
        let tkey = self.nodes[target as usize].key;
        let mut cur = self.root;
        while cur != target {
            chain.push(cur);
            let ck = self.nodes[cur as usize].key;
            cur = if tkey < ck {
                self.nodes[cur as usize].left
            } else {
                self.nodes[cur as usize].right
            };
        }
        let l = self.nodes[target as usize].left;
        let r = self.nodes[target as usize].right;
        if l == NIL || r == NIL {
            let replacement = if l != NIL { l } else { r };
            if chain.is_empty() {
                self.root = replacement;
            } else {
                let p = *chain.last().unwrap();
                if self.nodes[p as usize].left == target {
                    self.nodes[p as usize].left = replacement;
                } else {
                    self.nodes[p as usize].right = replacement;
                }
            }
            self.free_node(target);
            self.rebalance_chain(&chain);
            return;
        }
        // Two children: the in-order successor is the leftmost of the right subtree. Move its
        // payload up into `target`, then physically delete the successor (which has no left child).
        let mut succ_chain: Vec<u32> = Vec::new();
        let mut s = r;
        while self.nodes[s as usize].left != NIL {
            succ_chain.push(s);
            s = self.nodes[s as usize].left;
        }
        let succ = s;
        let (skey, sentries) = {
            let n = &mut self.nodes[succ as usize];
            (n.key, std::mem::take(&mut n.entries))
        };
        self.nodes[target as usize].key = skey;
        self.nodes[target as usize].entries = sentries;
        let succ_right = self.nodes[succ as usize].right;
        if succ_chain.is_empty() {
            // The successor was `target`'s right child.
            self.nodes[target as usize].right = succ_right;
        } else {
            let sp = *succ_chain.last().unwrap();
            self.nodes[sp as usize].left = succ_right;
        }
        self.free_node(succ);
        chain.push(target);
        chain.extend_from_slice(&succ_chain);
        self.rebalance_chain(&chain);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// The capped primary probe, on `Table`. Split out of `mod.rs` to keep that file under the 1,500-line
// bound; the executing probe and diagnostics are the same bodies.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// The number of primary probe steps the radix walk offers at this table width: the home slot,
    /// the `LINEAR_G` linear-prefix slots, then the fresh radix windows of `coord·PHI`. Steps at or
    /// past this index are the removed terminal sweep — `descend`'s last arm — and the capped probe
    /// never takes them; such a key is either placed within the walk or routed to the overflow.
    #[inline]
    fn window_steps(&self) -> u32 {
        LINEAR_G + 64 / self.bits - 1
    }

    /// The primary probe limit in slot examinations: [`PROBE_CAP`], or the whole radix walk when the
    /// word is spent sooner (large `bits`). At most `PROBE_CAP`.
    #[inline]
    fn probe_limit(&self) -> u32 {
        PROBE_CAP.min(self.window_steps())
    }

    /// The scalar tail of the capped probe, from step `j` to `limit`: one coordinate compare (plus
    /// the caller's confirmation) per step. Used past the SIMD-scanned linear prefix, and for the
    /// whole walk on the rare array-end wrap.
    #[cold]
    fn probe_rest_with(
        &self,
        coord: u64,
        mut j: u32,
        limit: u32,
        confirm: &impl Fn(usize) -> bool,
    ) -> Capped {
        while j < limit {
            let i = self.descend(coord, j);
            // SAFETY: i ∈ [0, len) — descend() masks to the power-of-two array.
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord && confirm(i) {
                return Capped::Hit(i);
            }
            if sc == EMPTY {
                return Capped::Vacant(i);
            }
            j += 1;
        }
        Capped::Blocked
    }

    /// The capped primary probe from the caller-probed `home` slot: the linear prefix is scanned in
    /// one branchless glance (`prefix_masks`), then the radix windows walk scalar up to
    /// [`probe_limit`](Self::probe_limit). No `EMPTY` within the cap and no identity ⇒
    /// [`Capped::Blocked`](super::Capped::Blocked), and the caller consults the overflow. Indexing is
    /// unchecked (`home`/`descend` mask to the array); a reserved coordinate never reaches here.
    #[cold]
    #[inline(never)]
    pub(super) fn probe_capped_with(
        &self,
        coord: u64,
        home: usize,
        confirm: &impl Fn(usize) -> bool,
    ) -> Capped {
        let limit = self.probe_limit();
        let len = self.coords.len();
        if limit >= LINEAR_G && home + LINEAR_G as usize <= len {
            // `home + LINEAR_G ≤ len`, so the `LINEAR_G`-lane read stays in bounds. Lane 0 is
            // `home`, already probed by the caller; a match there cannot confirm, but scanning it
            // is free.
            let (mut hits, empties) = prefix_masks(&self.coords, home, coord);
            while hits != 0 {
                let slot = home + hits.trailing_zeros() as usize;
                if confirm(slot) {
                    return Capped::Hit(slot);
                }
                hits &= hits - 1;
            }
            if empties != 0 {
                return Capped::Vacant(home + empties.trailing_zeros() as usize);
            }
            self.probe_rest_with(coord, LINEAR_G, limit, confirm)
        } else {
            // Wrap tail (rare) or a cap below the prefix width: the scalar walk from step 0. The
            // home slot is examined here — this probe is also reached directly from
            // `probe_capped_keyed`, which has not probed `home` yet.
            self.probe_rest_with(coord, 0, limit, confirm)
        }
    }

    /// [`probe_capped_with`](Self::probe_capped_with) with full-key confirmation: a coordinate match
    /// is a hit only when the stored key equals `k`, so two distinct keys with equal coordinates are
    /// two entries. For `Regen`, `matches` is the coordinate round-trip.
    ///
    /// The home slot is probed with one scalar compare first — the near-certain case for every
    /// insert and a home hit — so the `#[cold]` SIMD prefix glance is paid only on a collision. The
    /// cold body re-examines `home` in its lane 0; that redundant read is on the cold path.
    #[inline]
    pub(super) fn probe_capped_keyed(&self, coord: u64, k: &K) -> Capped {
        let home = self.home(coord);
        // SAFETY: home ∈ [0, len).
        let sc = unsafe { *self.coords.get_unchecked(home) };
        if sc == coord && self.keys.matches(home, coord, k) {
            return Capped::Hit(home);
        }
        if sc == EMPTY {
            return Capped::Vacant(home);
        }
        self.probe_capped_with(coord, home, &|i| self.keys.matches(i, coord, k))
    }

    /// Whether this table holds `value` at `coord`, by coordinate: a live primary slot with
    /// `coord == c` and an equal value, or an overflow entry at `c` with an equal value.
    /// Coordinate-only — equal coordinates conflate, exactly as the pre-overflow comparison did.
    pub(super) fn holds_value_at(&self, coord: u64, value: &V) -> bool
    where
        V: PartialEq,
    {
        let limit = self.probe_limit();
        let mut i = self.home(coord);
        let mut j = 0u32;
        loop {
            // SAFETY: i ∈ [0, len) — home()/descend() mask to the array; `sc == coord` names a live
            // slot whose value is initialized.
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord && unsafe { self.vals.get_unchecked(i).assume_init_ref() } == value {
                return true;
            }
            if sc == EMPTY || j + 1 >= limit {
                break;
            }
            j += 1;
            i = self.descend(coord, j);
        }
        self.overflow
            .as_ref()
            .is_some_and(|o| o.contains_value(coord, value))
    }

    /// Worst-case probe length, in steps. Coordinate keys: **1** (address = key, nothing to flood).
    /// Folded keys: collisions descend fresh windows, and `coord·PHI` is a bijection, so distinct
    /// keys can't fully collide — a pinned home slot builds no pile of ALIASES. That is not a bound
    /// on this number (`AggregationBound.descent_can_be_wholly_blocked`); the probe is capped at
    /// [`PROBE_CAP`] primary slots, and every key unresolved within the cap lives in the overflow.
    /// Read it as a measurement. Diagnostic — not part of the stable surface.
    #[doc(hidden)]
    pub fn longest_probe(&self) -> usize {
        let limit = self.probe_limit() as usize;
        let mut worst = 0;
        for (slot, sc) in self.coords.iter().enumerate() {
            if super::is_live(*sc) {
                let mut i = self.home(*sc);
                let mut j = 0u32;
                while i != slot && (j as usize) + 1 < limit {
                    j += 1;
                    i = self.descend(*sc, j);
                }
                worst = worst.max(((j as usize) + 1).min(limit));
            }
        }
        worst
    }

    /// The number of primary slots a capped probe of `k` examines: `1` for a home hit or an empty
    /// home; otherwise the step index (≤ [`PROBE_CAP`]) of the hit, the terminating `EMPTY`, or the
    /// exhausted cap. Diagnostic — not part of the stable surface.
    #[doc(hidden)]
    pub fn primary_examinations(&self, k: &K) -> u32 {
        let coord = Self::coord_of(k);
        if is_sentinel(coord) {
            return 0; // resolved in the side store, not a primary slot
        }
        let limit = self.probe_limit();
        let mut i = self.home(coord);
        let mut j = 0u32;
        loop {
            // SAFETY: i ∈ [0, len) — home()/descend() mask to the array.
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == EMPTY || (sc == coord && self.keys.matches(i, coord, k)) || j + 1 >= limit {
                return j + 1;
            }
            j += 1;
            i = self.descend(coord, j);
        }
    }

    /// The overflow tree's height: `0` when no key has been blocked. Diagnostic — the AVL bound is
    /// `height ≤ 2·log2(len + 1) + 1`.
    #[doc(hidden)]
    pub fn overflow_height(&self) -> u32 {
        self.overflow.as_ref().map_or(0, Overflow::height)
    }

    /// The number of keys held in the overflow tree. Diagnostic.
    #[doc(hidden)]
    pub fn overflow_len(&self) -> usize {
        self.overflow.as_ref().map_or(0, Overflow::len)
    }
}
