//! **PROTOTYPE — not the shipped layout, not adopted.** Two alternative slot layouts for
//! [`crate::hashtrinity::Table`], built so the current layout and the alternatives can be timed side by side in
//! ONE binary. Purely additive: nothing here is referenced by `Table`, `CoordMap`, `FreqMap` or any
//! converted call site, and the default layout is unchanged.
//!
//! The question being priced: `Table` decides occupancy by reading an 8-byte coordinate
//! (`is_live(c)` ⟺ `c < TOMB`). A full drain therefore streams 8 B/slot where hashbrown streams
//! 1 B/slot from a dense control array — the measured 0.48–0.82× iteration loss. Two ways to fix
//! that, in increasing order of what they touch:
//!
//! * [`TableB`] — **the live bitmap.** 1 BIT per slot, occupancy only, no hash bits. The probe is
//!   byte-for-byte `Table`'s (coordinate readout, unchanged); the bitmap is written on
//!   insert/remove/resize and read ONLY by iteration, where a `u64` word clears 64 slots.
//!   `CoordinateReadout.readout_beats_compare_by_construction` is about the *confirm* at a located
//!   slot and about a tag used for *identity*; an occupancy bit is neither, so this arm leaves the
//!   theorem untouched.
//! * [`TableC`] — **the SwissTable control byte.** 1 BYTE per slot carrying `{EMPTY, DELETED, 7-bit
//!   h2 tag}`, group-probed 16 lanes at a time with `_mm_movemask_epi8`. This one DOES carry hash
//!   bits used to reject a slot without touching the coordinate, i.e. a lossy tag in the theorem's
//!   sense — it engages the doctrine directly. The coordinate is still the confirm (a tag match is
//!   followed by `coords[i] == coord`), so identity is unchanged; what the tag buys is skipping the
//!   coordinate READ on a non-matching lane.
//!
//! Both mirror `Table`'s identity semantics exactly: the same `coord_of` nudge, the same 7/8 grow
//! law, the same tombstone-on-remove + reclaim-on-resize, the same `Stored`/`Regen` key stores.
//!
//! **Named cuts** (things `Table` has that these do not, and how each biases a number):
//! * no `entry`, no `get_each` pipeline, no threaded build, no `Absorb` insert-from-borrow. The
//!   sweep drives all three arms through `put`/`get_at`/`remove`/`iter`, which all three have, so
//!   no arm is measured on a path another lacks.
//! * `TableC` allocates `max(n, 16)` control bytes so a 16-lane group read is in bounds at the
//!   8-slot minimum; the padding lanes are permanently `EMPTY` and masked out. This over-counts
//!   `TableC`'s memory at `n = 8` by 8 bytes and nowhere else.
//! * `get` here is the coordinate-trusting read (`Table::get`, 2⁻⁶⁴), not `get_exact`.
//!
//! A third arm, [`TableZ`], changes NO width at all: it only moves the reserved sentinels from the
//! top of the `u64` range to the bottom (`EMPTY = 0`, `TOMB = 1`), which lets the coordinate array
//! come from `alloc_zeroed` instead of an 8 B/slot memset. See its own doc for the semantic price
//! that move carries for integer coordinate keys.

#![allow(clippy::missing_safety_doc)]

use crate::hashtrinity::face::Locate;
use crate::hashtrinity::{
    bits_for, is_live, Coord, CoordInv, KeyReadout, KeyStore, Regen, Stored, EMPTY, PHI, TOMB,
};
use std::mem::MaybeUninit;

/// Bytes the key store really allocates for `n` slots: `Stored` keeps one `K` per slot, `Regen`
/// keeps nothing. Derived from `KS::REGEN`, the same discriminant the crate already exposes — no
/// change to the `KeyStore` signature.
#[inline]
pub fn key_bytes<K: Coord, KS: KeyStore<K>>(n: usize) -> usize {
    if KS::REGEN {
        0
    } else {
        n * std::mem::size_of::<MaybeUninit<K>>()
    }
}

/// The `EMPTY`/`TOMB` nudge — identical to `Table::coord_of`, so the two layouts agree on identity
/// at every key including the reserved coordinates.
#[inline]
fn nudge(c: u64) -> u64 {
    if c == EMPTY || c == TOMB {
        TOMB - 1
    } else {
        c
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// ARM B — THE LIVE BITMAP. 1 bit/slot, occupancy only. The probe is unchanged.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// `Table`'s layout plus a 1-bit-per-slot occupancy bitmap, read only by iteration.
///
/// `live[i >> 6] >> (i & 63) & 1 == 1` ⟺ `is_live(coords[i])`. The invariant is maintained at the
/// four places occupancy changes: fill a vacant slot, tombstone a slot, allocate, resize.
pub struct TableB<K, V, KS = Stored<K>>
where
    K: Coord,
    KS: KeyStore<K>,
{
    coords: Box<[u64]>,
    live: Box<[u64]>,
    vals: Box<[MaybeUninit<V>]>,
    keys: KS,
    bits: u32,
    len: usize,
    tombs: usize,
    _k: std::marker::PhantomData<K>,
}

impl<K: Coord, V, KS: KeyStore<K>> Drop for TableB<K, V, KS> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.coords.iter().zip(self.vals.iter_mut()) {
                if is_live(*c) {
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        let coords = std::mem::take(&mut self.coords);
        self.keys.drop_live(&coords);
    }
}

impl<K: Eq + Coord, V> TableB<K, V, Stored<K>> {
    pub fn new() -> Self {
        Self::with_bits(3)
    }
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

impl<K: Eq + Coord, V> Default for TableB<K, V, Stored<K>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: CoordInv + Eq, V> TableB<K, V, Regen<K>> {
    pub fn coordinate() -> Self {
        Self::with_bits(3)
    }
    pub fn coordinate_with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> TableB<K, V, KS> {
    fn with_bits(bits: u32) -> Self {
        let n = 1usize << bits;
        TableB {
            coords: vec![EMPTY; n].into_boxed_slice(),
            live: vec![0u64; n.div_ceil(64)].into_boxed_slice(),
            vals: std::iter::repeat_with(MaybeUninit::uninit)
                .take(n)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            keys: KS::alloc(n),
            bits,
            len: 0,
            tombs: 0,
            _k: std::marker::PhantomData,
        }
    }

    #[inline]
    fn set_live(&mut self, i: usize) {
        unsafe { *self.live.get_unchecked_mut(i >> 6) |= 1u64 << (i & 63) };
    }
    #[inline]
    fn clear_live(&mut self, i: usize) {
        unsafe { *self.live.get_unchecked_mut(i >> 6) &= !(1u64 << (i & 63)) };
    }

    /// `Table::home`, verbatim.
    #[inline]
    fn home(&self, coord: u64) -> usize {
        let len = self.coords.len();
        if (coord as usize) < len {
            coord as usize
        } else {
            (coord.wrapping_mul(PHI) >> (64 - self.bits)) as usize
        }
    }

    /// `Table::descend`, verbatim (seqLin: linear prefix then radix windows).
    #[cold]
    fn descend(&self, coord: u64, j: u32) -> usize {
        const LINEAR_G: u32 = 8;
        let (bits, mask) = (self.bits, self.coords.len() - 1);
        if j < LINEAR_G {
            return self.home(coord).wrapping_add(j as usize) & mask;
        }
        let p = coord.wrapping_mul(PHI);
        let jj = j - LINEAR_G + 1;
        let used = bits * (jj + 1);
        if used <= 64 {
            (p >> (64 - used)) as usize & mask
        } else {
            let full = 64 / bits;
            ((p >> (64 - bits * full)) as usize + (jj - full) as usize + 1) & mask
        }
    }

    #[inline]
    fn probe(&self, coord: u64) -> (usize, bool) {
        let mut i = self.home(coord);
        let mut j = 0;
        loop {
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord {
                return (i, true);
            }
            if sc == EMPTY {
                return (i, false);
            }
            j += 1;
            i = self.descend(coord, j);
        }
    }

    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.get_with_coord(nudge(k.coord()))
    }

    /// Probe from a borrowed view — the `&str`-into-`Table<String,_>` shape.
    #[inline]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        self.get_with_coord(nudge(q.coord()))
    }

    /// `Table::get_coord`, verbatim — including the `#[cold]` collision tail and its `prefix_masks`
    /// SIMD glance. The bitmap is NOT read here: the read path is byte-for-byte the shipped one, so
    /// the `bmap` column of the GET table is a same-code control, and any difference it shows is
    /// code layout, not the bitmap.
    #[inline]
    fn get_with_coord(&self, coord: u64) -> Option<&V> {
        let home = self.home(coord);
        let sc = unsafe { *self.coords.get_unchecked(home) };
        if sc == coord {
            return Some(unsafe { self.vals.get_unchecked(home).assume_init_ref() });
        }
        if sc == EMPTY {
            return None;
        }
        self.get_cold(coord, home)
    }

    #[cold]
    #[inline(never)]
    fn get_cold(&self, coord: u64, home: usize) -> Option<&V> {
        const LINEAR_G: usize = 8;
        let len = self.coords.len();
        if home + LINEAR_G <= len {
            let s1 = unsafe { *self.coords.get_unchecked(home + 1) };
            if s1 == coord {
                return Some(unsafe { self.vals.get_unchecked(home + 1).assume_init_ref() });
            }
            if s1 == EMPTY {
                return None;
            }
            let (hits, empties) =
                crate::hashtrinity::prefix_masks(&self.coords, home, coord);
            if hits != 0 {
                let slot = home + hits.trailing_zeros() as usize;
                return Some(unsafe { self.vals.get_unchecked(slot).assume_init_ref() });
            }
            if empties != 0 {
                return None;
            }
            return self.get_walk(coord, LINEAR_G as u32);
        }
        self.get_walk(coord, 1)
    }

    #[cold]
    #[inline(never)]
    fn get_walk(&self, coord: u64, mut j: u32) -> Option<&V> {
        loop {
            let i = self.descend(coord, j);
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord {
                return Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() });
            }
            if sc == EMPTY {
                return None;
            }
            j += 1;
        }
    }

    #[inline]
    pub fn put(&mut self, k: K, v: V) {
        self.insert_with(k, v, |new, _old| new);
    }

    #[inline]
    pub fn insert_with(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) {
        self.insert_slot(k, new, f);
    }

    #[inline]
    pub fn insert_fetch(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) -> &V {
        let i = self.insert_slot(k, new, f);
        unsafe { self.vals.get_unchecked(i).assume_init_ref() }
    }

    #[inline]
    fn insert_slot(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) -> usize {
        if (self.len + self.tombs + 1) * 8 >= self.coords.len() * 7 {
            let new_bits = if (self.len + 1) * 8 >= self.coords.len() * 7 {
                self.bits + 1
            } else {
                self.bits
            };
            self.resize(new_bits);
        }
        let coord = nudge(k.coord());
        let (i, found) = self.probe(coord);
        if found {
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            unsafe { *self.coords.get_unchecked_mut(i) = EMPTY };
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(f(new, old)) };
            unsafe { *self.coords.get_unchecked_mut(i) = coord };
        } else {
            unsafe { *self.coords.get_unchecked_mut(i) = coord };
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
            self.keys.set(i, k);
            self.set_live(i); // ← THE PAID COST: one RMW on the bitmap word per fill
            self.len += 1;
        }
        i
    }

    #[inline]
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let coord = nudge(k.coord());
        let (i, found) = self.probe(coord);
        if found {
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            self.keys.take_key(i, coord);
            unsafe { *self.coords.get_unchecked_mut(i) = TOMB };
            self.clear_live(i); // ← THE PAID COST: one RMW on the bitmap word per remove
            self.len -= 1;
            self.tombs += 1;
            Some(old)
        } else {
            None
        }
    }

    #[cold]
    fn resize(&mut self, new_bits: u32) {
        let mut bigger = Self::with_bits(new_bits);
        let old_coords = std::mem::take(&mut self.coords);
        let mut old_vals = std::mem::replace(&mut self.vals, Box::new([]));
        let mut old_keys = std::mem::replace(&mut self.keys, KS::alloc(0));
        for (i, &c) in old_coords.iter().enumerate() {
            if is_live(c) {
                let (j, _) = bigger.probe(c);
                let v = unsafe { old_vals.get_unchecked_mut(i).assume_init_read() };
                unsafe { *bigger.coords.get_unchecked_mut(j) = c };
                unsafe { *bigger.vals.get_unchecked_mut(j) = MaybeUninit::new(v) };
                old_keys.migrate(i, &mut bigger.keys, j);
                bigger.set_live(j);
                bigger.len += 1;
            }
        }
        *self = bigger;
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.coords.len()
    }
    /// Bytes of slot line actually allocated: coords + bitmap + values (+ key store).
    pub fn slot_bytes(&self) -> usize {
        let n = self.coords.len();
        n * 8
            + self.live.len() * 8
            + n * std::mem::size_of::<MaybeUninit<V>>()
            + key_bytes::<K, KS>(n)
    }

    /// THE BOUGHT BENEFIT: the drain walks `n/8` bytes of bitmap instead of `n·8` bytes of
    /// coordinate. One `u64` load clears 64 slots; `trailing_zeros` + `w &= w-1` yields each live
    /// lane with no branch per empty slot and no coordinate read at all on the `Regen` path.
    pub fn iter(&self) -> IterB<'_, K, V, KS> {
        IterB {
            t: self,
            word: if self.live.is_empty() {
                0
            } else {
                self.live[0]
            },
            wi: 0,
        }
    }
}

pub struct IterB<'a, K: Coord, V, KS: KeyStore<K>> {
    t: &'a TableB<K, V, KS>,
    word: u64,
    wi: usize,
}

impl<'a, K: Coord, V, KS: KeyReadout<K>> Iterator for IterB<'a, K, V, KS> {
    type Item = (KS::Key<'a>, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.word != 0 {
                let b = self.word.trailing_zeros() as usize;
                self.word &= self.word - 1;
                let i = (self.wi << 6) | b;
                let c = unsafe { *self.t.coords.get_unchecked(i) };
                return Some((self.t.keys.key(i, c), unsafe {
                    self.t.vals.get_unchecked(i).assume_init_ref()
                }));
            }
            self.wi += 1;
            if self.wi >= self.t.live.len() {
                return None;
            }
            self.word = unsafe { *self.t.live.get_unchecked(self.wi) };
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.t.len, Some(self.t.len))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// ARM C — THE SWISSTABLE CONTROL BYTE. 1 byte/slot: {EMPTY, DELETED, 7-bit h2}. Group probe.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

const GROUP: usize = 16;
const C_EMPTY: u8 = 0xFF;
const C_DEL: u8 = 0x80;

/// The 7-bit fingerprint — the LOW byte of `coord·PHI`, a window DISJOINT from the top-`bits`
/// window `home` uses. A top-window tag would be home-determined and reject nothing
/// (`GroupProbe.top_window_is_home_determined`).
#[inline]
fn h2(coord: u64) -> u8 {
    (coord.wrapping_mul(PHI) as u8) & 0x7F
}

/// 16 control bytes → a bitmask of lanes equal to `tag`. `tag < 0x80`, so `EMPTY`/`DELETED` never
/// match.
#[inline]
unsafe fn tag_mask(ctrl: *const u8, tag: u8) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let v = _mm_loadu_si128(ctrl.cast());
        _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_set1_epi8(tag as i8))) as u16
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut m = 0u16;
        for k in 0..GROUP {
            m |= u16::from(*ctrl.add(k) == tag) << k;
        }
        m
    }
}

/// Lanes that are exactly `EMPTY` (0xFF) — where a probe stops and an insert lands.
#[inline]
unsafe fn empty_mask(ctrl: *const u8) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let v = _mm_loadu_si128(ctrl.cast());
        _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_set1_epi8(C_EMPTY as i8))) as u16
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut m = 0u16;
        for k in 0..GROUP {
            m |= u16::from(*ctrl.add(k) == C_EMPTY) << k;
        }
        m
    }
}

/// Lanes that are FULL — the sign bit is set on both sentinels and clear on every tag, so this is
/// a bare `movemask`, no compare. THIS is the drain: one instruction per 16 slots, 1 B/slot read.
#[inline]
unsafe fn full_mask(ctrl: *const u8) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let v = _mm_loadu_si128(ctrl.cast());
        !(_mm_movemask_epi8(v) as u16)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut m = 0u16;
        for k in 0..GROUP {
            m |= u16::from(*ctrl.add(k) < 0x80) << k;
        }
        m
    }
}

/// `Table`'s layout plus a dense SwissTable control byte per slot, group-probed.
pub struct TableC<K, V, KS = Stored<K>>
where
    K: Coord,
    KS: KeyStore<K>,
{
    coords: Box<[u64]>,
    /// `max(n, 16)` bytes. Lanes `n..16` (only when `n = 8`) stay `EMPTY` and are masked out.
    ctrl: Box<[u8]>,
    vals: Box<[MaybeUninit<V>]>,
    keys: KS,
    bits: u32,
    /// `(1 << min(n,16)) - 1` — the valid-lane mask for the `n = 8` group.
    gmask: u16,
    len: usize,
    tombs: usize,
    _k: std::marker::PhantomData<K>,
}

impl<K: Coord, V, KS: KeyStore<K>> Drop for TableC<K, V, KS> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.coords.iter().zip(self.vals.iter_mut()) {
                if is_live(*c) {
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        let coords = std::mem::take(&mut self.coords);
        self.keys.drop_live(&coords);
    }
}

impl<K: Eq + Coord, V> TableC<K, V, Stored<K>> {
    pub fn new() -> Self {
        Self::with_bits(3)
    }
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

impl<K: Eq + Coord, V> Default for TableC<K, V, Stored<K>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: CoordInv + Eq, V> TableC<K, V, Regen<K>> {
    pub fn coordinate() -> Self {
        Self::with_bits(3)
    }
    pub fn coordinate_with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> TableC<K, V, KS> {
    fn with_bits(bits: u32) -> Self {
        let n = 1usize << bits;
        TableC {
            coords: vec![EMPTY; n].into_boxed_slice(),
            ctrl: vec![C_EMPTY; n.max(GROUP)].into_boxed_slice(),
            vals: std::iter::repeat_with(MaybeUninit::uninit)
                .take(n)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            keys: KS::alloc(n),
            bits,
            gmask: if n >= GROUP {
                u16::MAX
            } else {
                ((1u32 << n) - 1) as u16
            },
            len: 0,
            tombs: 0,
            _k: std::marker::PhantomData,
        }
    }

    #[inline]
    fn ngroups(&self) -> usize {
        (self.coords.len() / GROUP).max(1)
    }

    /// `Table::home`, verbatim — so the two layouts place dense coordinate keys identically.
    #[inline]
    fn home(&self, coord: u64) -> usize {
        let len = self.coords.len();
        if (coord as usize) < len {
            coord as usize
        } else {
            (coord.wrapping_mul(PHI) >> (64 - self.bits)) as usize
        }
    }

    /// The group probe: glance 16 tags, confirm each candidate against the COORDINATE (the tag is a
    /// filter, never identity), stop at the first `EMPTY` lane. Groups walk linearly and the walk
    /// terminates because the 7/8 grow law keeps at least one `EMPTY` lane in the table.
    #[inline]
    fn probe(&self, coord: u64) -> (usize, bool) {
        let tag = h2(coord);
        let ng = self.ngroups();
        let mut g = (self.home(coord) / GROUP) % ng;
        loop {
            let base = g * GROUP;
            let p = unsafe { self.ctrl.as_ptr().add(base) };
            let mut m = unsafe { tag_mask(p, tag) } & self.gmask;
            while m != 0 {
                let lane = m.trailing_zeros() as usize;
                let i = base + lane;
                if unsafe { *self.coords.get_unchecked(i) } == coord {
                    return (i, true);
                }
                m &= m - 1;
            }
            let e = unsafe { empty_mask(p) } & self.gmask;
            if e != 0 {
                return (base + e.trailing_zeros() as usize, false);
            }
            g += 1;
            if g == ng {
                g = 0;
            }
        }
    }

    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.get_with_coord(nudge(k.coord()))
    }
    #[inline]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        self.get_with_coord(nudge(q.coord()))
    }
    #[inline]
    fn get_with_coord(&self, coord: u64) -> Option<&V> {
        let (i, found) = self.probe(coord);
        if found {
            Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() })
        } else {
            None
        }
    }

    #[inline]
    pub fn put(&mut self, k: K, v: V) {
        self.insert_with(k, v, |new, _old| new);
    }

    #[inline]
    pub fn insert_with(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) {
        self.insert_slot(k, new, f);
    }

    #[inline]
    pub fn insert_fetch(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) -> &V {
        let i = self.insert_slot(k, new, f);
        unsafe { self.vals.get_unchecked(i).assume_init_ref() }
    }

    #[inline]
    fn insert_slot(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) -> usize {
        if (self.len + self.tombs + 1) * 8 >= self.coords.len() * 7 {
            let new_bits = if (self.len + 1) * 8 >= self.coords.len() * 7 {
                self.bits + 1
            } else {
                self.bits
            };
            self.resize(new_bits);
        }
        let coord = nudge(k.coord());
        let (i, found) = self.probe(coord);
        if found {
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            unsafe { *self.coords.get_unchecked_mut(i) = EMPTY };
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(f(new, old)) };
            unsafe { *self.coords.get_unchecked_mut(i) = coord };
        } else {
            unsafe { *self.coords.get_unchecked_mut(i) = coord };
            unsafe { *self.ctrl.get_unchecked_mut(i) = h2(coord) }; // ← THE PAID COST
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
            self.keys.set(i, k);
            self.len += 1;
        }
        i
    }

    #[inline]
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let coord = nudge(k.coord());
        let (i, found) = self.probe(coord);
        if found {
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            self.keys.take_key(i, coord);
            unsafe { *self.coords.get_unchecked_mut(i) = TOMB };
            unsafe { *self.ctrl.get_unchecked_mut(i) = C_DEL }; // ← THE PAID COST
            self.len -= 1;
            self.tombs += 1;
            Some(old)
        } else {
            None
        }
    }

    #[cold]
    fn resize(&mut self, new_bits: u32) {
        let mut bigger = Self::with_bits(new_bits);
        let old_coords = std::mem::take(&mut self.coords);
        let mut old_vals = std::mem::replace(&mut self.vals, Box::new([]));
        let mut old_keys = std::mem::replace(&mut self.keys, KS::alloc(0));
        for (i, &c) in old_coords.iter().enumerate() {
            if is_live(c) {
                let (j, _) = bigger.probe(c);
                let v = unsafe { old_vals.get_unchecked_mut(i).assume_init_read() };
                unsafe { *bigger.coords.get_unchecked_mut(j) = c };
                unsafe { *bigger.ctrl.get_unchecked_mut(j) = h2(c) };
                unsafe { *bigger.vals.get_unchecked_mut(j) = MaybeUninit::new(v) };
                old_keys.migrate(i, &mut bigger.keys, j);
                bigger.len += 1;
            }
        }
        *self = bigger;
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.coords.len()
    }
    /// Bytes of slot line actually allocated: coords + ctrl + values (+ key store).
    pub fn slot_bytes(&self) -> usize {
        let n = self.coords.len();
        n * 8 + self.ctrl.len() + n * std::mem::size_of::<MaybeUninit<V>>() + key_bytes::<K, KS>(n)
    }

    /// THE BOUGHT BENEFIT: one `_mm_movemask_epi8` clears 16 slots, reading 1 B/slot.
    pub fn iter(&self) -> IterC<'_, K, V, KS> {
        let m = unsafe { full_mask(self.ctrl.as_ptr()) } & self.gmask;
        IterC {
            t: self,
            mask: m,
            g: 0,
            ng: self.ngroups(),
        }
    }
}

pub struct IterC<'a, K: Coord, V, KS: KeyStore<K>> {
    t: &'a TableC<K, V, KS>,
    mask: u16,
    g: usize,
    ng: usize,
}

impl<'a, K: Coord, V, KS: KeyReadout<K>> Iterator for IterC<'a, K, V, KS> {
    type Item = (KS::Key<'a>, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.mask != 0 {
                let lane = self.mask.trailing_zeros() as usize;
                self.mask &= self.mask - 1;
                let i = self.g * GROUP + lane;
                let c = unsafe { *self.t.coords.get_unchecked(i) };
                return Some((self.t.keys.key(i, c), unsafe {
                    self.t.vals.get_unchecked(i).assume_init_ref()
                }));
            }
            self.g += 1;
            if self.g >= self.ng {
                return None;
            }
            self.mask = unsafe { full_mask(self.t.ctrl.as_ptr().add(self.g * GROUP)) };
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.t.len, Some(self.t.len))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// ARM Z — THE ZERO SENTINEL. Same 8-byte coordinate line, no sideband at all: only the reserved
// values move, from the TOP of the u64 range to the BOTTOM.
// ═══════════════════════════════════════════════════════════════════════════════════════════════
//
//   shipped   EMPTY = u64::MAX, TOMB = MAX-1, is_live(c) = c < TOMB, coord_of folds
//             {MAX, MAX-1} → MAX-2, allocation is `vec![u64::MAX; n]` — a real memset of 8 B/slot.
//   arm Z     Z_EMPTY = 0,      Z_TOMB = 1,   is_live(c) = c > 1,     coord_of folds {0, 1} → 2,
//             allocation is `vec![0u64; n]` — which `RawVec`'s `IsZero` specialization lowers to
//             `alloc_zeroed`, i.e. `calloc`, i.e. fresh pages the kernel already guarantees zero.
//
// Every property is uniform in the reserved triple: same 3-element defect set, same SINGLE unsigned
// compare for liveness (`1 < c` instead of `c < TOMB`), same confirm, same probe. The only thing
// that changes is who pays to write the empty state.
//
// This arm exists to price the GROW face. Its `iter` is present for the correctness gate but is a
// plain scalar scan, NOT the shipped 8-wide `vpcmpgtq` glance, so its DRAIN number is not
// comparable and is not reported.

const Z_EMPTY: u64 = 0;
const Z_TOMB: u64 = 1;

#[inline]
fn z_live(c: u64) -> bool {
    c > Z_TOMB
}

#[inline]
fn z_nudge(c: u64) -> u64 {
    if c <= Z_TOMB {
        Z_TOMB + 1
    } else {
        c
    }
}

/// `Table` with the reserved sentinels moved to the bottom of the `u64` range, so the empty state
/// is all-zero and the coordinate array can come from `alloc_zeroed`.
pub struct TableZ<K, V, KS = Stored<K>>
where
    K: Coord,
    KS: KeyStore<K>,
{
    coords: Box<[u64]>,
    vals: Box<[MaybeUninit<V>]>,
    keys: KS,
    bits: u32,
    len: usize,
    tombs: usize,
    _k: std::marker::PhantomData<K>,
}

impl<K: Coord, V, KS: KeyStore<K>> Drop for TableZ<K, V, KS> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.coords.iter().zip(self.vals.iter_mut()) {
                if z_live(*c) {
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        // `KS::drop_live` reads the CALLER's liveness convention (`is_live`, i.e. `c < TOMB`), so
        // hand it a translated copy rather than this arm's zero-sentinel line.
        let translated: Box<[u64]> = self
            .coords
            .iter()
            .map(|&c| if z_live(c) { c.min(TOMB - 1) } else { EMPTY })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.coords = Box::new([]);
        self.keys.drop_live(&translated);
    }
}

impl<K: Eq + Coord, V> TableZ<K, V, Stored<K>> {
    pub fn new() -> Self {
        Self::with_bits(3)
    }
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

impl<K: Eq + Coord, V> Default for TableZ<K, V, Stored<K>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: CoordInv + Eq, V> TableZ<K, V, Regen<K>> {
    pub fn coordinate() -> Self {
        Self::with_bits(3)
    }
    pub fn coordinate_with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> TableZ<K, V, KS> {
    fn with_bits(bits: u32) -> Self {
        let n = 1usize << bits;
        TableZ {
            // THE POINT OF THIS ARM: `vec![0u64; n]` hits `RawVec`'s `IsZero` specialization and
            // lowers to `alloc_zeroed`; `vec![u64::MAX; n]` cannot and lowers to a memset.
            coords: vec![Z_EMPTY; n].into_boxed_slice(),
            vals: std::iter::repeat_with(MaybeUninit::uninit)
                .take(n)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            keys: KS::alloc(n),
            bits,
            len: 0,
            tombs: 0,
            _k: std::marker::PhantomData,
        }
    }

    #[inline]
    fn home(&self, coord: u64) -> usize {
        let len = self.coords.len();
        if (coord as usize) < len {
            coord as usize
        } else {
            (coord.wrapping_mul(PHI) >> (64 - self.bits)) as usize
        }
    }

    #[cold]
    fn descend(&self, coord: u64, j: u32) -> usize {
        const LINEAR_G: u32 = 8;
        let (bits, mask) = (self.bits, self.coords.len() - 1);
        if j < LINEAR_G {
            return self.home(coord).wrapping_add(j as usize) & mask;
        }
        let p = coord.wrapping_mul(PHI);
        let jj = j - LINEAR_G + 1;
        let used = bits * (jj + 1);
        if used <= 64 {
            (p >> (64 - used)) as usize & mask
        } else {
            let full = 64 / bits;
            ((p >> (64 - bits * full)) as usize + (jj - full) as usize + 1) & mask
        }
    }

    #[inline]
    fn probe(&self, coord: u64) -> (usize, bool) {
        let mut i = self.home(coord);
        let mut j = 0;
        loop {
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord {
                return (i, true);
            }
            if sc == Z_EMPTY {
                return (i, false);
            }
            j += 1;
            i = self.descend(coord, j);
        }
    }

    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.get_with_coord(z_nudge(k.coord()))
    }
    #[inline]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        self.get_with_coord(z_nudge(q.coord()))
    }
    #[inline]
    fn get_with_coord(&self, coord: u64) -> Option<&V> {
        let home = self.home(coord);
        let sc = unsafe { *self.coords.get_unchecked(home) };
        if sc == coord {
            return Some(unsafe { self.vals.get_unchecked(home).assume_init_ref() });
        }
        if sc == Z_EMPTY {
            return None;
        }
        self.get_walk(coord, 1)
    }
    #[cold]
    #[inline(never)]
    fn get_walk(&self, coord: u64, mut j: u32) -> Option<&V> {
        loop {
            let i = self.descend(coord, j);
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord {
                return Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() });
            }
            if sc == Z_EMPTY {
                return None;
            }
            j += 1;
        }
    }

    #[inline]
    pub fn put(&mut self, k: K, v: V) {
        self.insert_with(k, v, |new, _old| new);
    }

    #[inline]
    pub fn insert_with(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) {
        if (self.len + self.tombs + 1) * 8 >= self.coords.len() * 7 {
            let new_bits = if (self.len + 1) * 8 >= self.coords.len() * 7 {
                self.bits + 1
            } else {
                self.bits
            };
            self.resize(new_bits);
        }
        let coord = z_nudge(k.coord());
        let (i, found) = self.probe(coord);
        if found {
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            unsafe { *self.coords.get_unchecked_mut(i) = Z_EMPTY };
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(f(new, old)) };
            unsafe { *self.coords.get_unchecked_mut(i) = coord };
        } else {
            unsafe { *self.coords.get_unchecked_mut(i) = coord };
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
            self.keys.set(i, k);
            self.len += 1;
        }
    }

    #[inline]
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let coord = z_nudge(k.coord());
        let (i, found) = self.probe(coord);
        if found {
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            self.keys.take_key(i, coord);
            unsafe { *self.coords.get_unchecked_mut(i) = Z_TOMB };
            self.len -= 1;
            self.tombs += 1;
            Some(old)
        } else {
            None
        }
    }

    #[cold]
    fn resize(&mut self, new_bits: u32) {
        let mut bigger = Self::with_bits(new_bits);
        let old_coords = std::mem::take(&mut self.coords);
        let mut old_vals = std::mem::replace(&mut self.vals, Box::new([]));
        let mut old_keys = std::mem::replace(&mut self.keys, KS::alloc(0));
        for (i, &c) in old_coords.iter().enumerate() {
            if z_live(c) {
                let (j, _) = bigger.probe(c);
                let v = unsafe { old_vals.get_unchecked_mut(i).assume_init_read() };
                unsafe { *bigger.coords.get_unchecked_mut(j) = c };
                unsafe { *bigger.vals.get_unchecked_mut(j) = MaybeUninit::new(v) };
                old_keys.migrate(i, &mut bigger.keys, j);
                bigger.len += 1;
            }
        }
        *self = bigger;
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.coords.len()
    }

    /// Correctness only — a plain scalar scan, NOT the shipped glance. Do not time this.
    pub fn iter(&self) -> impl Iterator<Item = (KS::Key<'_>, &V)>
    where
        KS: KeyReadout<K>,
    {
        self.coords
            .iter()
            .enumerate()
            .filter(|(_, c)| z_live(**c))
            .map(move |(i, &c)| {
                (self.keys.key(i, c), unsafe {
                    self.vals.get_unchecked(i).assume_init_ref()
                })
            })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// GATE — the prototypes must agree with `Table` on every operation, or their numbers mean nothing.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashtrinity::{CoordMap, Table};

    fn multiset<T: Ord>(mut v: Vec<T>) -> Vec<T> {
        v.sort();
        v
    }

    #[test]
    fn coord_arms_agree_with_table() {
        for n in [1usize, 7, 8, 9, 100, 1000, 5000] {
            let keys: Vec<u32> = (0..n as u32)
                .map(|i| i.wrapping_mul(2_654_435_761).rotate_left(13))
                .collect();
            let mut a: CoordMap<u32, i64> = CoordMap::coordinate();
            let mut b: TableB<u32, i64, Regen<u32>> = TableB::coordinate();
            let mut c: TableC<u32, i64, Regen<u32>> = TableC::coordinate();
            for (i, &k) in keys.iter().enumerate() {
                a.put(k, i as i64);
                b.put(k, i as i64);
                c.put(k, i as i64);
            }
            // overwrite half
            for (i, &k) in keys.iter().enumerate().filter(|(i, _)| i % 2 == 0) {
                a.put(k, -(i as i64));
                b.put(k, -(i as i64));
                c.put(k, -(i as i64));
            }
            // remove a third
            for &k in keys.iter().step_by(3) {
                let (ra, rb, rc) = (a.remove(&k), b.remove(&k), c.remove(&k));
                assert_eq!(ra, rb, "remove B n={n}");
                assert_eq!(ra, rc, "remove C n={n}");
            }
            // re-insert some, forcing tombstone reuse / same-size rebuild
            for (i, &k) in keys.iter().step_by(3).enumerate() {
                a.put(k, 777 + i as i64);
                b.put(k, 777 + i as i64);
                c.put(k, 777 + i as i64);
            }
            assert_eq!(a.len(), b.len(), "len B n={n}");
            assert_eq!(a.len(), c.len(), "len C n={n}");
            for &k in &keys {
                assert_eq!(a.get(&k), b.get(&k), "get B n={n} k={k}");
                assert_eq!(a.get(&k), c.get(&k), "get C n={n} k={k}");
            }
            // absent keys
            for k in [u32::MAX, u32::MAX - 1, 0xDEAD_BEEF] {
                assert_eq!(a.get(&k), b.get(&k));
                assert_eq!(a.get(&k), c.get(&k));
            }
            let ia = multiset(a.iter().map(|(k, v)| (k, *v)).collect::<Vec<_>>());
            let ib = multiset(b.iter().map(|(k, v)| (k, *v)).collect::<Vec<_>>());
            let ic = multiset(c.iter().map(|(k, v)| (k, *v)).collect::<Vec<_>>());
            assert_eq!(ia, ib, "iter B n={n}");
            assert_eq!(ia, ic, "iter C n={n}");
        }
    }

    #[test]
    fn stored_arms_agree_with_table() {
        for n in [1usize, 8, 9, 100, 3000] {
            let strs: Vec<String> = (0..n).map(|i| format!("key-{i:08}-{}", i * 7919)).collect();
            let mut a: Table<String, i64> = Table::new();
            let mut b: TableB<String, i64> = TableB::new();
            let mut c: TableC<String, i64> = TableC::new();
            for (i, s) in strs.iter().enumerate() {
                a.put(s.clone(), i as i64);
                b.put(s.clone(), i as i64);
                c.put(s.clone(), i as i64);
            }
            for s in strs.iter().step_by(4) {
                let (ra, rb, rc) = (a.remove(s), b.remove(s), c.remove(s));
                assert_eq!(ra, rb);
                assert_eq!(ra, rc);
            }
            assert_eq!(a.len(), b.len());
            assert_eq!(a.len(), c.len());
            for s in &strs {
                assert_eq!(a.get(s), b.get(s));
                assert_eq!(a.get(s), c.get(s));
                assert_eq!(a.get_at(s.as_str()), b.get_at(s.as_str()));
                assert_eq!(a.get_at(s.as_str()), c.get_at(s.as_str()));
            }
            let ia = multiset(a.iter().map(|(k, v)| (k.clone(), *v)).collect::<Vec<_>>());
            let ib = multiset(b.iter().map(|(k, v)| (k.clone(), *v)).collect::<Vec<_>>());
            let ic = multiset(c.iter().map(|(k, v)| (k.clone(), *v)).collect::<Vec<_>>());
            assert_eq!(ia, ib);
            assert_eq!(ia, ic);
        }
    }

    /// `TableZ` agrees with `Table` on every key whose coordinate avoids the moved defect set.
    #[test]
    fn zero_sentinel_agrees_away_from_its_defect() {
        for n in [1usize, 8, 100, 5000] {
            // every coordinate ≥ 3, so {0,1,2} is untouched
            let keys: Vec<u32> = (0..n as u32)
                .map(|i| i.wrapping_mul(2_654_435_761).rotate_left(13) | 4)
                .collect();
            let mut a: CoordMap<u32, i64> = CoordMap::coordinate();
            let mut z: TableZ<u32, i64, Regen<u32>> = TableZ::coordinate();
            for (i, &k) in keys.iter().enumerate() {
                a.put(k, i as i64);
                z.put(k, i as i64);
            }
            for &k in keys.iter().step_by(3) {
                let (ra, rz) = (a.remove(&k), z.remove(&k));
                assert_eq!(ra, rz, "remove Z n={n}");
            }
            assert_eq!(a.len(), z.len());
            for &k in &keys {
                assert_eq!(a.get(&k), z.get(&k), "get Z n={n} k={k}");
            }
            let mut ia: Vec<(u32, i64)> = a.iter().map(|(k, v)| (k, *v)).collect();
            let mut iz: Vec<(u32, i64)> = z.iter().map(|(k, v)| (k, *v)).collect();
            ia.sort();
            iz.sort();
            assert_eq!(ia, iz, "iter Z n={n}");
        }
    }

    /// **The zero sentinel is NOT semantically free for integer coordinate keys.** The shipped
    /// table reserves the TOP of the `u64` range, so its `coord_of` nudge collapses
    /// `{MAX, MAX-1, MAX-2}` — coordinates a `u32`/`u16`/`u8` key can never produce. Moving the
    /// sentinels to the BOTTOM collapses `{0, 1, 2}` instead, and those are the first three keys of
    /// every dense integer map. This test PINS the difference rather than asserting it away.
    #[test]
    fn zero_sentinel_defect_is_reachable_for_integer_keys() {
        let mut a: CoordMap<u32, i64> = CoordMap::coordinate();
        let mut z: TableZ<u32, i64, Regen<u32>> = TableZ::coordinate();
        for k in [0u32, 1, 2] {
            a.put(k, i64::from(k) + 100);
            z.put(k, i64::from(k) + 100);
        }
        // shipped: three distinct keys, three distinct entries
        assert_eq!(a.len(), 3);
        assert_eq!(a.get(&0), Some(&100));
        assert_eq!(a.get(&1), Some(&101));
        assert_eq!(a.get(&2), Some(&102));
        // zero sentinel: 0 and 1 are nudged onto 2, so all three alias to ONE entry
        assert_eq!(z.len(), 1, "keys 0,1,2 collapse under EMPTY=0/TOMB=1");
        assert_eq!(z.get(&0), Some(&102));
        assert_eq!(z.get(&1), Some(&102));
        // and the regenerated key comes back as 2, not 0
        let regen: Vec<u32> = z.iter().map(|(k, _)| k).collect();
        assert_eq!(regen, vec![2u32]);
    }

    /// The values must really be dropped exactly once — the arms carry `MaybeUninit` by hand.
    #[test]
    fn drop_exactly_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        struct D(#[allow(dead_code)] u32);
        impl Drop for D {
            fn drop(&mut self) {
                N.fetch_add(1, Ordering::Relaxed);
            }
        }
        N.store(0, Ordering::Relaxed);
        {
            let mut b: TableB<u32, D, Regen<u32>> = TableB::coordinate();
            for i in 0..500u32 {
                b.put(i, D(i));
            }
            for i in (0..500u32).step_by(5) {
                drop(b.remove(&i));
            }
        }
        assert_eq!(N.load(Ordering::Relaxed), 500);
        N.store(0, Ordering::Relaxed);
        {
            let mut c: TableC<u32, D, Regen<u32>> = TableC::coordinate();
            for i in 0..500u32 {
                c.put(i, D(i));
            }
            for i in (0..500u32).step_by(5) {
                drop(c.remove(&i));
            }
        }
        assert_eq!(N.load(Ordering::Relaxed), 500);
    }
}
