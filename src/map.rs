//! The two keyed faces: **B** (the address is the key) and **B/U** (the key rides beside it).
//!
//! [`crate::table`] is the third face. It keeps a fold and a count and throws the key away, which
//! is what hyperbolic licenses and nothing else does. A map cannot do that -- it has to hand the
//! value back to whoever names the key -- so a map lives on one of the two faces that keep
//! something, and *which* one is decided by the key type rather than by preference:
//!
//! | key | its fold | face | a slot holds | a hit costs |
//! |---|---|---|---|---|
//! | fits in a word ([`Packed`]) | a bijection | elliptic, **B** | the address | one compare |
//! | wider ([`Coord`] only) | a discard | parabolic, **B/U** | address **and** key | a compare, then the key |
//!
//! The second row is not a safety margin. `str -> u64` has fibers by counting; two keys in one
//! fiber get the same address and therefore the *same probe sequence*, and the only thing that can
//! tell them apart is the key itself. `Projection.discards_iff_disc_pos` says the parabolic face
//! has no discard available, so the key cannot be thrown away -- the same sentence read from the
//! algebra's side.
//!
//! The first row is not a micro-optimisation either. [`Regen`] stores no key at all, so a
//! `u64 -> u32` table is eight bytes a slot lighter than the same table on the parabolic face,
//! growth never touches a key, and a hit never reads past the slot it landed on. That is what
//! "the address determines the key" *buys*, stated as bytes.
//!
//! ```
//! use apeiron::map::{Map, Words};
//! let mut resrefs: Map<String, u32> = Map::new();       // parabolic: the key is carried
//! resrefs.insert("nw_wswmg001".into(), 7);
//! assert_eq!(resrefs.get_at("nw_wswmg001"), Some(&7));  // no String materialised
//!
//! let mut ids: Words<u64, u32> = Words::new();          // elliptic: the address is the key
//! ids.insert(4096, 1);
//! assert_eq!(ids.get(&4096), Some(&1));
//! ```
//!
//! # The probe
//!
//! Unchanged from [`crate::table`]: successive `gbits`-wide windows of the whole avalanched word,
//! not a linear walk, so extending a collision chain by one step costs another `gbits` bits of
//! control rather than being free. What is new here is removal, and removal is where a descent and
//! a tombstone have to be made to agree -- see [`Map::remove`].
//!
//! # Why the borrowed-key API is not a convenience
//!
//! `HashMap<String, V>` is probed with `&str` through `Borrow`. A map whose `get` takes `&K` is
//! probed, by every mechanical port, with `map.get(&s.to_string())` -- which compiles, runs, and
//! allocates once per lookup. [`Locate`] is the same idea expressed through the fold: a borrowed
//! view that lands on the address its owned key lands on, so `(&str, &str)` probes a
//! `Map<(String, String), V>` with no allocation at all. It is offered on the parabolic face only,
//! because the elliptic face's keys fit in a word and there is nothing to borrow.

#![allow(clippy::needless_pass_by_value)] // a `Locate` view IS the borrow; taking it by reference
                                          // would put a second indirection on the probe.

use crate::coord::{finish, inverse_odd, Coord, Packed, PHI, PHI_INV};
use crate::keystore::Keystore;
use crate::table::{group_at, scan, tag_of, GROUP};
use std::marker::PhantomData;
use std::mem::MaybeUninit;

/// An unused slot. Zero, so a fresh region is a `calloc` the kernel may hand over lazily rather
/// than a pass of writes.
const EMPTY: u8 = 0;
/// A slot whose entry was removed. Distinct from [`EMPTY`] because a probe that stopped here would
/// lose every key placed beyond it while it was live, and distinct from a live tag because
/// `tag_of` always sets the top bit and this does not.
const TOMB: u8 = 1;
/// A live slot's tag has this bit; `EMPTY` and `TOMB` do not. One `movemask` separates them.
const LIVE: u8 = 0x80;

/// Fill limit, as a fraction. Seven eighths, where `hashbrown` sits, so a memory comparison
/// against it compares layouts rather than load factors.
const FILL_NUM: usize = 7;
const FILL_DEN: usize = 8;

/// The smallest non-empty table: one group, one tag line, one vector compare.
const MIN_SLOTS: usize = GROUP;

/// How many entries a region of `slots` slots holds before it must grow.
///
/// Never equal to `slots`. `locate` walks until it finds an empty slot, so a region with none does
/// not terminate -- a hang, at a load an attacker would choose. This is the precondition, split out
/// so it can be proved rather than believed.
#[inline]
#[must_use]
pub const fn limit(slots: usize) -> usize {
    slots / FILL_DEN * FILL_NUM
}

/// The probe order: successive `gbits`-wide windows of the whole avalanched word.
///
/// [`crate::table::group_at`] computes the `j`-th window from scratch, which costs two branches and
/// a variable shift on every step of every probe. A probe walks its windows in order, so the state
/// can just be *carried*: each step takes the top `gbits` and shifts the word left by that much.
/// The two agree step for step -- `tests/the_descent_is_the_descent.rs` checks it over the whole
/// range -- and this one is a shift and a mask.
///
/// Once the word is spent the two must still agree, and there the carried form has nothing left to
/// shift, so it falls back to the function. That path is the one an attacker with a long collision
/// chain reaches, and it is deliberately the slow one: the fast path is the honest case.
/// The address bijection's secret half.
///
/// A pepper cannot be *added* to a key and do any good, and finding that out is what this type is
/// for. `spread` is `x -> (x * M) ^ ((x * M) >> 32)`; the top 32 bits of the answer are the top 32
/// bits of `x * M` untouched. Adding a constant to `x` before the multiply is adding `c * M` after
/// it, a translation -- and a set of keys chosen to agree on their high bits still agrees on them
/// after a translation. Measured: an additive pepper left a 200,000-key aimed build 996x slower
/// than an honest one, against 1008x with no pepper at all. It bought nothing, and it looked like
/// it should have.
///
/// So the pepper becomes **the multiplier**. An attacker who does not know `M` cannot choose keys
/// whose products have chosen high bits, because that is inverting a multiplication they cannot
/// name. `M` is forced odd, so the map stays a bijection on `Z/2^64` and the elliptic face can
/// still run it backwards -- with [`inverse_odd`], which is why that function exists.
///
/// A pepper of zero gives `M = PHI` and the ordinary deterministic, aimable table. That is the
/// right default for a table whose keys the process chose itself, and the wrong one for a table
/// fed from a socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Salt {
    pepper: u64,
    mul: u64,
    inv: u64,
}

impl Salt {
    #[must_use]
    /// The salt a pepper produces. Zero gives the plain, deterministic, aimable bijection.
    pub const fn new(pepper: u64) -> Self {
        if pepper == 0 {
            return Self {
                pepper: 0,
                mul: PHI,
                inv: PHI_INV,
            };
        }
        // Through `finish` first: a pepper read from config might be 1, or a date, and a multiplier
        // has to be spread across the whole word to spread anything else.
        let mul = finish(pepper) | 1;
        Self {
            pepper,
            mul,
            inv: inverse_odd(mul),
        }
    }

    /// Avalanche: one multiply and one xor-fold. The top bits spread because multiplication by an
    /// odd constant carries low bits upward; the low seven -- the tag -- spread because the fold
    /// brings the product's high half down to meet them.
    #[inline]
    #[must_use]
    pub const fn spread(self, x: u64) -> u64 {
        let h = x.wrapping_mul(self.mul);
        h ^ (h >> 32)
    }

    /// Exactly undone. `x ^ (x >> 32)` is an involution and the multiply is odd.
    #[inline]
    #[must_use]
    pub const fn unspread(self, h: u64) -> u64 {
        (h ^ (h >> 32)).wrapping_mul(self.inv)
    }

    #[must_use]
    /// The pepper this salt was built from.
    pub const fn pepper(self) -> u64 {
        self.pepper
    }
}

impl Default for Salt {
    fn default() -> Self {
        Self::new(0)
    }
}

/// What one group's scan concluded.
enum Step {
    /// The key is here.
    Found(usize),
    /// This group holds an empty slot, so the key is nowhere: placement is first-fit in probe
    /// order and a live entry is never moved.
    Absent,
    /// Full of live entries and tombstones; the probe continues.
    Onward,
}

#[derive(Clone, Copy, Debug, Default)]
struct Geom {
    /// `log2(groups)`.
    gbits: u32,
    /// `64 - gbits`, and 64 when a region is one group -- a shift *by* 64 is undefined, so it is
    /// taken with `wrapping_shr`, which then shifts by nothing and the mask (zero) makes the answer
    /// group zero, which is right because group zero is the whole region.
    gshift: u32,
    /// `groups - 1`.
    gmask: usize,
}

impl Geom {
    fn of(groups: usize) -> Self {
        let gbits = groups.trailing_zeros();
        Self {
            gbits,
            gshift: 64 - gbits,
            gmask: groups.saturating_sub(1),
        }
    }

    /// The first group, straight-line: one shift and one mask, from geometry the table already
    /// knows. Recomputing `groups.trailing_zeros()` per lookup put a `tzcnt` and a variable-shift
    /// setup in front of every probe; caching it is the difference between five instructions before
    /// the tag load and two.
    #[inline]
    fn first(self, p: u64) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        {
            (p.wrapping_shr(self.gshift) as usize & self.gmask) * GROUP
        }
    }
}

/// The probe order: successive `gbits`-wide windows of the whole avalanched word.
///
/// [`crate::table::group_at`] computes the `j`-th window from scratch. A probe walks its windows in
/// order, so the state can be *carried* instead: each step takes the top `gbits` of what is left
/// and shifts. The two agree step for step -- `tests/the_descent_is_the_descent.rs` checks it over
/// the range -- and this one is a shift and a mask.
///
/// Once the word is spent the two must still agree, and there the carried form has nothing left to
/// shift, so it falls back to the function. That path is the one an attacker with a long collision
/// chain reaches, and it is deliberately the slow one: the fast path is the honest case.
struct Descent {
    p: u64,
    /// What is left of the word; the next window is its top `gbits`.
    window: u64,
    j: u32,
    /// Bits of `p` consumed. Compared against 64 rather than against `64 / gbits`, because that
    /// quotient is an integer division on a runtime value and would sit on every probe.
    spent: u32,
    g: Geom,
}

impl Descent {
    /// Starts *after* the first group, which the caller took straight-line from [`Geom::first`].
    #[inline]
    fn after_first(p: u64, g: Geom) -> Self {
        Self {
            p,
            window: p.wrapping_shl(g.gbits),
            j: 1,
            spent: g.gbits,
            g,
        }
    }

    /// The next group's first slot.
    #[inline]
    fn next(&mut self) -> usize {
        let j = self.j;
        self.j += 1;
        self.spent += self.g.gbits;
        if self.spent > 64 || self.g.gbits == 0 {
            // The word is spent, or the region is a single group and every window is group zero.
            return group_at(self.p, j, self.g.gbits, self.g.gmask) * GROUP;
        }
        #[allow(clippy::cast_possible_truncation)]
        let g = (self.window >> self.g.gshift) as usize & self.g.gmask;
        self.window <<= self.g.gbits;
        g * GROUP
    }
}

// -- what a slot remembers of its key -----------------------------------------------------------

/// How a live slot holds its key's identity.
///
/// The two implementations are the two keyed faces of [`Keystore`], and they differ in exactly the
/// way the discriminant predicts: [`Regen`] keeps nothing, because the address determines the key;
/// [`Stored`] keeps the key, because it does not.
pub trait Held<K: Coord + Eq>: Sized {
    /// The face this is the storage of.
    const FACE: Keystore;

    /// The key as read back: a borrow where it was stored, a fresh value where it is regenerated.
    type Ref<'a>
    where
        Self: 'a,
        K: 'a;

    /// The avalanched address of `k` under `pepper`.
    ///
    /// Not the same function on the two faces, and it must not be: the elliptic face addresses by
    /// [`Packed::packed`], a bijection, while the parabolic face addresses by [`Coord::coord_with`],
    /// a discard. For an integer these coincide; for `(i21, i21, i21)` they do not, and reading the
    /// wrong one would silently give the elliptic face a lossy address and with it a key compare it
    /// does not perform.
    fn address(salt: Salt, k: &K) -> u64;

    /// Build a slot's key half from the address and the key that produced it.
    fn hold(p: u64, k: K) -> Self;

    /// The avalanched address this slot was placed at.
    fn addr(&self) -> u64;

    /// Whether this slot holds `k`, given `k`'s address.
    fn holds(&self, p: u64, k: &K) -> bool;

    /// Whether two slots hold the same key. Used by equality, which has a slot and not a key.
    fn same_slot(&self, other: &Self) -> bool;

    /// The key.
    ///
    /// Takes the pepper because the elliptic face **regenerates** rather than stores, and the
    /// pepper is part of the bijection it inverts. Storing it per slot instead would cost the eight
    /// bytes the face exists to save.
    fn key(&self, salt: Salt) -> Self::Ref<'_>;

    /// The key, owned, consuming the slot.
    fn into_key(self, salt: Salt) -> K;
}

/// **B/U, the parabolic face.** The key is carried beside its address.
///
/// Both are kept and both are read, in that order: the address first because it is already in the
/// slot the tag line pointed at, the key only when the address agreed. A seven-bit tag has a
/// 1-in-128 false-hit rate, so on the ordinary miss the key is never touched -- which is what keeps
/// a `String` table's probe off a second allocation.
pub struct Stored<K> {
    p: u64,
    k: K,
}

impl<K: Coord + Eq> Held<K> for Stored<K> {
    const FACE: Keystore = Keystore::BeingOverUnbeing;
    type Ref<'a>
        = &'a K
    where
        K: 'a;

    #[inline]
    fn address(salt: Salt, k: &K) -> u64 {
        salt.spread(k.coord_with(salt.pepper()))
    }
    #[inline]
    fn hold(p: u64, k: K) -> Self {
        Self { p, k }
    }
    #[inline]
    fn addr(&self) -> u64 {
        self.p
    }
    #[inline]
    fn holds(&self, p: u64, k: &K) -> bool {
        self.p == p && self.k == *k
    }
    #[inline]
    fn same_slot(&self, other: &Self) -> bool {
        self.p == other.p && self.k == other.k
    }
    #[inline]
    fn key(&self, _salt: Salt) -> &K {
        &self.k
    }
    #[inline]
    fn into_key(self, _salt: Salt) -> K {
        self.k
    }
}

/// **B, the elliptic face.** The address *is* the key, so the slot keeps only the address.
///
/// `spread` is a bijection and so is [`Packed::packed`], so the composite is one: equal addresses
/// are equal keys, exactly rather than probably. The key compare is not skipped as an optimisation,
/// it is **absent**, because there is no second thing to compare -- and the key is read back by
/// running the bijection the other way rather than by storing it.
///
/// The pepper is folded in *before* the bijection and taken back out on the way home, so this face
/// carries one without storing one. It is not needed for correctness here -- a bijection has no
/// fibers, so there is nothing an attacker can make collide -- but it is needed for the other half
/// of the problem, which is aiming. See [`Map::with_pepper`].
pub struct Regen<K>(u64, PhantomData<K>);

impl<K: Coord + Packed + Eq> Held<K> for Regen<K> {
    const FACE: Keystore = Keystore::Being;
    type Ref<'a>
        = K
    where
        K: 'a;

    #[inline]
    fn address(salt: Salt, k: &K) -> u64 {
        salt.spread(k.packed())
    }
    #[inline]
    fn hold(p: u64, _k: K) -> Self {
        Self(p, PhantomData)
    }
    #[inline]
    fn addr(&self) -> u64 {
        self.0
    }
    #[inline]
    fn holds(&self, p: u64, _k: &K) -> bool {
        self.0 == p
    }
    #[inline]
    fn same_slot(&self, other: &Self) -> bool {
        self.0 == other.0
    }
    #[inline]
    fn key(&self, salt: Salt) -> K {
        K::unpack(salt.unspread(self.0))
    }
    #[inline]
    fn into_key(self, salt: Salt) -> K {
        K::unpack(salt.unspread(self.0))
    }
}

// -- borrowed keys, on the parabolic face --------------------------------------------------------

/// A borrowed view of a key that lands on the **same address** as the owned key.
///
/// What `std` does with `Borrow<Q>` and `Hash`, done through the fold instead. The contract is one
/// equation --
///
/// > `q.coord_at(pepper) == owned.coord_with(pepper)` whenever `q` names `owned`
///
/// -- and every implementation satisfies it by *being* the owned key's fold rather than by agreeing
/// with it: `&str` calls the same `fold_bytes` a `String` calls, and a tuple of views calls the same
/// `mix` a tuple of keys calls. `tests/a_borrowed_key_is_the_same_key.rs` pins it.
pub trait Locate<K: Coord + Eq>: Copy {
    /// This view's address under `pepper`.
    fn coord_at(&self, pepper: u64) -> u64;
    /// Whether this view names exactly `k`.
    fn same(&self, k: &K) -> bool;
}

/// A view that can become the owned key -- paid only when a probe misses and the table must take
/// ownership. A hit allocates nothing.
pub trait Absorb<K: Coord + Eq>: Locate<K> {
    /// Materialise the owned key. Called on the vacant path and nowhere else.
    fn absorb(self) -> K;
}

impl<K: Coord + Eq> Locate<K> for &K {
    #[inline]
    fn coord_at(&self, pepper: u64) -> u64 {
        (**self).coord_with(pepper)
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
    fn coord_at(&self, pepper: u64) -> u64 {
        <str as Coord>::coord_with(self, pepper)
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

impl Locate<Vec<u8>> for &[u8] {
    #[inline]
    fn coord_at(&self, pepper: u64) -> u64 {
        <[u8] as Coord>::coord_with(self, pepper)
    }
    #[inline]
    fn same(&self, k: &Vec<u8>) -> bool {
        *self == k.as_slice()
    }
}
impl Absorb<Vec<u8>> for &[u8] {
    #[inline]
    fn absorb(self) -> Vec<u8> {
        self.to_vec()
    }
}

impl<A: Coord + Eq, B: Coord + Eq, PA: Locate<A>, PB: Locate<B>> Locate<(A, B)> for (PA, PB) {
    #[inline]
    fn coord_at(&self, pepper: u64) -> u64 {
        // Literally `<(A, B) as Coord>::coord_with`, with each component's address supplied by that
        // component's own view -- not a second scheme that happens to agree.
        crate::coord::finish(crate::coord::mix(
            self.0.coord_at(pepper),
            self.1.coord_at(pepper),
        ))
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

// -- the map ---------------------------------------------------------------------------------------

/// A map on one of the two keyed faces.
///
/// `Map<K, V>` is the parabolic default and takes any `K: Coord + Eq`. [`Words`] is the elliptic
/// alias, for keys that fit in a word.
pub struct Map<K, V, H = Stored<K>> {
    /// One byte per slot: [`EMPTY`], [`TOMB`], or a live slot's seven-bit tag with [`LIVE`] set.
    /// Sixteen of these are one vector compare, which is what makes a group scan one instruction
    /// rather than sixteen loads.
    tags: Box<[u8]>,
    /// The entries. **`slots[i]` is initialised exactly when `tags[i] & LIVE != 0`.** That is the
    /// whole safety invariant of this module; every `unsafe` below cites it and nothing else.
    slots: Box<[MaybeUninit<(H, V)>]>,
    /// Live entries.
    live: usize,
    /// Live entries plus tombstones -- what the fill limit is measured against, because a tombstone
    /// occupies probe order even though it holds nothing.
    used: usize,
    /// The region's probe geometry, recomputed only when the region is.
    geom: Geom,
    salt: Salt,
    _k: PhantomData<K>,
}

/// The elliptic face: keys that fit in a word, addressed bijectively, with no key stored.
pub type Words<K, V> = Map<K, V, Regen<K>>;

// SAFETY: the map owns its keys and values outright; there is no interior mutability and no shared
// state. `MaybeUninit` is what makes the auto-impls not apply on their own.
unsafe impl<K: Send, V: Send, H: Send> Send for Map<K, V, H> {}
unsafe impl<K: Sync, V: Sync, H: Sync> Sync for Map<K, V, H> {}

impl<K: Coord + Eq, V, H: Held<K>> Default for Map<K, V, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Coord + Eq, V, H: Held<K>> Map<K, V, H> {
    /// An empty map. Allocates nothing, so one that is declared and never filled costs a few words.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tags: Box::new([]),
            slots: Box::new([]),
            live: 0,
            used: 0,
            geom: Geom::default(),
            salt: Salt::new(0),
            _k: PhantomData,
        }
    }

    /// An empty map that holds `n` entries without growing.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        let mut m = Self::new();
        m.reserve(n);
        m
    }

    /// An empty map with an address pepper. **Set one on any table fed from outside the process.**
    ///
    /// It closes two different holes, and only one of them is the obvious one.
    ///
    /// *Fibers.* A byte fold is a discard, so two keys can share an address; they then share a
    /// probe sequence and can only be separated by the key compare, which is `O(m)` for a fiber of
    /// size `m`. A pepper moves the fibers somewhere an attacker cannot precompute.
    ///
    /// *Aiming.* This one is worse and applies to **both** faces, integer keys included. The
    /// address function is a public bijection, so keys whose addresses agree on any number of bits
    /// are one inversion away -- and every window of the descent is a slice of that one word.
    /// Measured in `examples/the_fiber.rs`: two hundred thousand aimed `u64` keys build **988x**
    /// slower than an honest set, and **0.9x** with a pepper. The window descent bounds a chain in
    /// bits of address and cannot make those bits expensive; only a secret can.
    ///
    /// A pepper is a process-wide **constant**, not a per-run random seed. Choose it once at
    /// startup, log it, and a recorded session replays exactly -- determinism and unpredictability
    /// are not in conflict, because only reproducibility of *a recorded run* was ever needed.
    #[must_use]
    pub fn with_pepper(pepper: u64) -> Self {
        let mut m = Self::new();
        m.salt = Salt::new(pepper);
        m
    }

    /// This map's pepper.
    #[must_use]
    pub const fn pepper(&self) -> u64 {
        self.salt.pepper()
    }

    /// The address bijection this map is using.
    #[must_use]
    pub const fn salt(&self) -> Salt {
        self.salt
    }

    /// Which face this map is on -- surfaced so a caller can assert it rather than infer it.
    #[must_use]
    pub fn face(&self) -> Keystore {
        H::FACE
    }

    /// Live entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live
    }

    /// Whether the map holds nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Entries this map holds before it must grow.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        limit(self.slots.len())
    }

    /// Slots allocated, live or not.
    #[must_use]
    pub const fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Slots holding a removed entry. Nonzero means the next growth may be a rebuild in place.
    #[must_use]
    pub const fn tombstones(&self) -> usize {
        self.used - self.live
    }

    // -- probing -------------------------------------------------------------------------------

    /// The slot `is` accepts, if any.
    ///
    /// Stops at the first group containing a genuinely empty slot: placement is first-fit in probe
    /// order and a live entry is never moved, so an empty slot in probe order means the key would
    /// have been placed at or before it. A **tombstone is not such a slot**, which is the whole
    /// reason it is distinct from empty -- stopping there would lose every key placed past it while
    /// it was live.
    #[inline]
    fn find(&self, p: u64, is: impl Fn(&H) -> bool) -> Option<usize> {
        if self.slots.len() < MIN_SLOTS {
            return None;
        }
        let tag = tag_of(p);
        // The first group is peeled out of the loop deliberately. Left inside it, the compiler
        // hoists the whole descent's loop-invariant setup -- the shifted window, the step counter,
        // the masked shift amount -- ahead of the *first* tag load, which is seven instructions
        // paid on every lookup to serve the case where the first group misses. Almost none do: at
        // a 7/8 fill an occupied group holds the key it was probed for.
        let base = self.geom.first(p);
        match self.scan_group(base, p, tag, &is) {
            Step::Found(i) => return Some(i),
            Step::Absent => return None,
            Step::Onward => {}
        }
        let mut d = Descent::after_first(p, self.geom);
        loop {
            match self.scan_group(d.next(), p, tag, &is) {
                Step::Found(i) => return Some(i),
                Step::Absent => return None,
                Step::Onward => {}
            }
        }
    }

    /// One group: the tag compare, the candidates it names, and whether the probe is over.
    #[inline]
    fn scan_group(&self, base: usize, p: u64, tag: u8, is: &impl Fn(&H) -> bool) -> Step {
        let _ = p;
        let line = self.line(base);
        let mut hits = scan::match_tag(line, tag);
        while hits != 0 {
            let lane = base + hits.trailing_zeros() as usize;
            // SAFETY: a matched tag has `LIVE` set (`tag_of` sets it; `EMPTY` and `TOMB` do not),
            // so by the slot invariant this entry is initialised, and `lane` is inside the line,
            // which is inside the region.
            let (h, _) = unsafe { self.slots.get_unchecked(lane).assume_init_ref() };
            if is(h) {
                return Step::Found(lane);
            }
            hits &= hits - 1;
        }
        if scan::match_tag(line, EMPTY) != 0 {
            Step::Absent
        } else {
            Step::Onward
        }
    }

    /// `Ok(slot)` when present, `Err(slot)` naming where it would go.
    ///
    /// The vacant answer prefers the earliest **tombstone** over the first empty slot. Reusing one
    /// is what keeps a table that is repeatedly filled and drained from growing without bound, and
    /// it is sound because the key's own probe order passed through that tombstone on the way here.
    #[inline]
    fn site(&self, p: u64, is: impl Fn(&H) -> bool) -> Result<usize, usize> {
        debug_assert!(self.slots.len() >= MIN_SLOTS, "probing an unallocated table");
        let tag = tag_of(p);
        let mut vacant: Option<usize> = None;
        let mut base = self.geom.first(p);
        let mut d = Descent::after_first(p, self.geom);
        loop {
            let line = self.line(base);
            let mut hits = scan::match_tag(line, tag);
            while hits != 0 {
                let lane = base + hits.trailing_zeros() as usize;
                // SAFETY: as `find`.
                let (h, _) = unsafe { self.slots.get_unchecked(lane).assume_init_ref() };
                if is(h) {
                    return Ok(lane);
                }
                hits &= hits - 1;
            }
            if vacant.is_none() {
                // Empty or tombstoned: both are free, and the earliest in probe order wins.
                let free = !scan::occupied(line);
                if free != 0 {
                    vacant = Some(base + free.trailing_zeros() as usize);
                }
            }
            if scan::match_tag(line, EMPTY) != 0 {
                // An empty slot ends the search, and it is itself free, so `vacant` is set.
                return Err(vacant.unwrap_or(base));
            }
            base = d.next();
        }
    }

    #[inline]
    fn line(&self, base: usize) -> &[u8; GROUP] {
        // SAFETY: `Descent` yields a group start below `tags.len() / GROUP`, so the sixteen bytes
        // from there are inside the array. `tags.len() == slots.len()` and both are a multiple of
        // `GROUP` -- `rebuild` is the only thing that sizes them and it allocates in groups.
        unsafe { &*self.tags.as_ptr().add(base).cast::<[u8; GROUP]>() }
    }

    // -- reading -------------------------------------------------------------------------------

    #[must_use]
    /// The value at `k`.
    pub fn get(&self, k: &K) -> Option<&V> {
        let p = H::address(self.salt, k);
        let i = self.find(p, |h| h.holds(p, k))?;
        // SAFETY: `find` returns only live slots.
        Some(unsafe { &self.slots[i].assume_init_ref().1 })
    }

    /// The value at `k`, mutably.
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let p = H::address(self.salt, k);
        let i = self.find(p, |h| h.holds(p, k))?;
        // SAFETY: as `get`.
        Some(unsafe { &mut self.slots[i].assume_init_mut().1 })
    }

    #[must_use]
    /// Whether `k` is present.
    pub fn contains_key(&self, k: &K) -> bool {
        let p = H::address(self.salt, k);
        self.find(p, |h| h.holds(p, k)).is_some()
    }

    /// The stored key and its value.
    #[must_use]
    pub fn get_key_value(&self, k: &K) -> Option<(H::Ref<'_>, &V)> {
        let p = H::address(self.salt, k);
        let i = self.find(p, |h| h.holds(p, k))?;
        // SAFETY: as `get`.
        let (h, v) = unsafe { self.slots[i].assume_init_ref() };
        Some((h.key(self.salt), v))
    }

    // -- writing -------------------------------------------------------------------------------

    /// Insert, returning what was displaced.
    ///
    /// Takes the key by value and clones nothing: the probe borrows it, and the borrow is over
    /// before either arm needs to own it.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.reserve(1);
        let p = H::address(self.salt, &k);
        match self.site(p, |h| h.holds(p, &k)) {
            Ok(i) => {
                // SAFETY: `site`'s `Ok` names a live slot.
                let slot = unsafe { self.slots[i].assume_init_mut() };
                Some(std::mem::replace(&mut slot.1, v))
            }
            Err(i) => {
                self.place(i, p, k, v);
                None
            }
        }
    }

    /// **The merge.** Combine `v` with whatever is already at `k`, in one probe.
    ///
    /// This is the primitive; every other writer here is a special case of it. It is also the one
    /// the algebra has something to say about: `fold`'s commutativity and associativity hold on
    /// every face with no hypothesis on the dial, so which merge is chosen never depends on the
    /// keystore. **One merge, three keystores** -- see [`crate::keystore`], and [`crate::merge`]
    /// for the four that a table is ordinarily built with.
    pub fn insert_with(&mut self, k: K, v: V, f: impl FnOnce(V, V) -> V) {
        self.reserve(1);
        let p = H::address(self.salt, &k);
        match self.site(p, |h| h.holds(p, &k)) {
            Ok(i) => self.merge_at(i, p, v, f),
            Err(i) => self.place(i, p, k, v),
        }
    }

    /// Write a fresh entry into a vacant slot.
    #[inline]
    fn place(&mut self, i: usize, p: u64, k: K, v: V) {
        if self.tags[i] == EMPTY {
            self.used += 1;
        }
        self.live += 1;
        self.slots[i] = MaybeUninit::new((H::hold(p, k), v));
        self.tags[i] = tag_of(p);
    }

    /// Replace slot `i`'s value with `f(new, old)`.
    ///
    /// The tag is cleared for the duration. `f` is caller code and may panic; if it does, the slot
    /// holds a value that has been moved out of, and the cleared tag is what stops the unwind from
    /// dropping it or a later probe from reading it. The entry is lost -- a tombstone in its place
    /// -- which is the honest outcome for a merge that did not complete.
    #[inline]
    fn merge_at(&mut self, i: usize, p: u64, v: V, f: impl FnOnce(V, V) -> V) {
        self.tags[i] = TOMB;
        self.live -= 1;
        let slot = self.slots[i].as_mut_ptr();
        // SAFETY: the slot was live, so it is initialised; the tag now says it is not, so nothing
        // else will read or drop it while the value is out.
        let old = unsafe { std::ptr::read(std::ptr::addr_of!((*slot).1)) };
        let merged = f(v, old);
        // SAFETY: writing back into the slot the value came out of.
        unsafe { std::ptr::write(std::ptr::addr_of_mut!((*slot).1), merged) };
        self.tags[i] = tag_of(p);
        self.live += 1;
    }

    /// Remove, returning the value.
    ///
    /// The slot becomes a **tombstone**, not empty. A window descent has the same requirement a
    /// linear probe does: a key placed later in probe order was placed *because* this slot was
    /// occupied, so emptying it would make that key unreachable while it is still in the table. The
    /// tombstone keeps the probe walking, and the next insert that passes it takes it, so a table
    /// filled and drained repeatedly does not grow.
    pub fn remove(&mut self, k: &K) -> Option<V> {
        Some(self.remove_entry(k)?.1)
    }

    /// Remove, returning the key as well.
    pub fn remove_entry(&mut self, k: &K) -> Option<(K, V)> {
        let p = H::address(self.salt, k);
        let i = self.find(p, |h| h.holds(p, k))?;
        Some(self.evict(i))
    }

    #[inline]
    fn evict(&mut self, i: usize) -> (K, V) {
        self.tags[i] = TOMB;
        self.live -= 1;
        // SAFETY: the slot was live and its tag now says it is not, so nothing else will read or
        // drop it.
        let (h, v) = unsafe { self.slots[i].assume_init_read() };
        (h.into_key(self.salt), v)
    }

    /// Drop every entry, keeping the allocation.
    pub fn clear(&mut self) {
        for i in 0..self.tags.len() {
            let live = self.tags[i] & LIVE != 0;
            self.tags[i] = EMPTY;
            if live {
                // SAFETY: the slot was live and its tag now says it is not.
                unsafe { self.slots[i].assume_init_drop() };
            }
        }
        self.live = 0;
        self.used = 0;
    }

    /// Keep the entries `keep` returns true for.
    pub fn retain(&mut self, mut keep: impl FnMut(H::Ref<'_>, &mut V) -> bool) {
        for i in 0..self.tags.len() {
            if self.tags[i] & LIVE == 0 {
                continue;
            }
            // SAFETY: the tag says live.
            let entry = unsafe { self.slots[i].assume_init_mut() };
            // Disjoint fields of the same entry: the key is read while the value is written.
            let (h, v) = (&entry.0, &mut entry.1);
            if keep(h.key(self.salt), v) {
                continue;
            }
            self.tags[i] = TOMB;
            self.live -= 1;
            // SAFETY: the tag now says otherwise; nothing else will read or drop it.
            unsafe { self.slots[i].assume_init_drop() };
        }
    }

    /// Make room for `additional` more entries.
    ///
    /// Grows when the fill limit would be crossed. When most of what fills the table is tombstones
    /// rather than entries it rebuilds at the *same* size instead of doubling -- a table churned in
    /// place would otherwise double on garbage.
    pub fn reserve(&mut self, additional: usize) {
        if self.used + additional <= limit(self.slots.len()) {
            return;
        }
        let want = self.live + additional;
        let mut slots = self.slots.len().max(MIN_SLOTS);
        while limit(slots) < want {
            slots *= 2;
        }
        self.rebuild(slots);
    }

    /// Move every live entry into a fresh region of `slots` slots.
    ///
    /// Nothing is re-folded. The address is in the slot, and it is the *avalanched* address, so the
    /// new placement is a descent over a word that already exists: growth costs a probe per entry
    /// and not a hash per entry. For a hundred thousand strings that is the difference between
    /// re-reading megabytes of bytes and reading a word.
    fn rebuild(&mut self, slots: usize) {
        let mut tags = vec![EMPTY; slots].into_boxed_slice();
        let mut fresh: Box<[MaybeUninit<(H, V)>]> =
            (0..slots).map(|_| MaybeUninit::uninit()).collect();
        let geometry = Geom::of(slots / GROUP);
        let (gbits, gmask) = (geometry.gbits, geometry.gmask);

        let old_tags = std::mem::replace(&mut self.tags, Box::new([]));
        let old_slots = std::mem::replace(&mut self.slots, Box::new([]));
        for (i, &t) in old_tags.iter().enumerate() {
            if t & LIVE == 0 {
                continue;
            }
            // SAFETY: the tag says live. `MaybeUninit` has no drop glue, so moving the entry out
            // here is the only move -- the old array is dropped as raw storage below.
            let entry = unsafe { old_slots[i].assume_init_read() };
            let p = entry.0.addr();
            let mut j = 0u32;
            let at = loop {
                let base = group_at(p, j, gbits, gmask) * GROUP;
                // A fresh region holds only empty slots, so this is plain first-fit with no
                // tombstone bookkeeping.
                let line: &[u8; GROUP] = tags[base..base + GROUP]
                    .try_into()
                    .unwrap_or(&[EMPTY; GROUP]);
                let free = !scan::occupied(line);
                if free != 0 {
                    break base + free.trailing_zeros() as usize;
                }
                j += 1;
            };
            tags[at] = tag_of(p);
            fresh[at] = MaybeUninit::new(entry);
        }
        self.tags = tags;
        self.slots = fresh;
        self.geom = geometry;
        self.used = self.live;
    }

    // -- iteration -----------------------------------------------------------------------------

    /// Every entry, in slot order -- which is address order, so neither insertion order nor sorted.
    /// It is however the *same* order for the same keys under the same pepper, on every machine and
    /// every run: nothing here is seeded and nothing is randomised.
    pub fn iter(&self) -> impl Iterator<Item = (H::Ref<'_>, &V)> {
        self.tags
            .iter()
            .enumerate()
            .filter(|(_, t)| **t & LIVE != 0)
            .map(move |(i, _)| {
                // SAFETY: the tag says live.
                let (h, v) = unsafe { self.slots[i].assume_init_ref() };
                (h.key(self.salt), v)
            })
    }

    /// Every entry, in slot order, with the value mutable.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, H> {
        IterMut {
            n: self.tags.len(),
            i: 0,
            map: std::ptr::from_mut(self),
            _m: PhantomData,
        }
    }

    /// The keys, in slot order.
    pub fn keys(&self) -> impl Iterator<Item = H::Ref<'_>> {
        self.iter().map(|(k, _)| k)
    }

    /// The values, in slot order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// The values, in slot order, mutably.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }

    /// Take every entry out, keeping the allocation.
    pub fn drain(&mut self) -> Vec<(K, V)> {
        let mut out = Vec::with_capacity(self.live);
        for i in 0..self.tags.len() {
            let live = self.tags[i] & LIVE != 0;
            self.tags[i] = EMPTY;
            if live {
                // SAFETY: the slot was live and its tag now says it is empty.
                let (h, v) = unsafe { self.slots[i].assume_init_read() };
                out.push((h.into_key(self.salt), v));
            }
        }
        self.live = 0;
        self.used = 0;
        out
    }
}

/// `iter_mut`, as a named type because the key and the value come out of the same entry.
///
/// `std` can hand back `(&K, &mut V)` from safe code because its key and value live in separate
/// arrays. Here they are two fields of one tuple, which is just as disjoint but not expressible
/// through a single borrow, so the split is made through a pointer and stated here rather than
/// hidden in a closure.
pub struct IterMut<'a, K: Coord + Eq, V, H: Held<K>> {
    map: *mut Map<K, V, H>,
    i: usize,
    n: usize,
    _m: PhantomData<&'a mut Map<K, V, H>>,
}

impl<'a, K: Coord + Eq, V, H: Held<K>> Iterator for IterMut<'a, K, V, H> {
    type Item = (H::Ref<'a>, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.i < self.n {
            let i = self.i;
            self.i += 1;
            // SAFETY: `map` came from a `&'a mut` and no other reference to it exists for `'a`.
            // Each index is visited exactly once and the two references are to disjoint fields of
            // one entry, so nothing handed out ever aliases anything else handed out.
            unsafe {
                let m = &mut *self.map;
                let salt = m.salt;
                if m.tags[i] & LIVE == 0 {
                    continue;
                }
                let entry = m.slots[i].as_mut_ptr();
                let h: &'a H = &*std::ptr::addr_of!((*entry).0);
                let v: &'a mut V = &mut *std::ptr::addr_of_mut!((*entry).1);
                return Some((h.key(salt), v));
            }
        }
        None
    }
}

// -- the borrowed-key surface, parabolic face only ------------------------------------------------

impl<K: Coord + Eq, V> Map<K, V, Stored<K>> {
    /// Read through a borrowed view: `map.get_at("resref")` on a `Map<String, V>`.
    #[must_use]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        let p = self.salt.spread(q.coord_at(self.salt.pepper()));
        let i = self.find(p, |h| h.p == p && q.same(&h.k))?;
        // SAFETY: `find` returns only live slots.
        Some(unsafe { &self.slots[i].assume_init_ref().1 })
    }

    /// The value a view names, mutably.
    pub fn get_mut_at<Q: Locate<K>>(&mut self, q: Q) -> Option<&mut V> {
        let p = self.salt.spread(q.coord_at(self.salt.pepper()));
        let i = self.find(p, |h| h.p == p && q.same(&h.k))?;
        // SAFETY: as `get_at`.
        Some(unsafe { &mut self.slots[i].assume_init_mut().1 })
    }

    #[must_use]
    /// Whether a view's key is present.
    pub fn contains_at<Q: Locate<K>>(&self, q: Q) -> bool {
        let p = self.salt.spread(q.coord_at(self.salt.pepper()));
        self.find(p, |h| h.p == p && q.same(&h.k)).is_some()
    }

    /// Insert through a view, taking ownership of the key only on a miss.
    pub fn insert_at<Q: Absorb<K>>(&mut self, q: Q, v: V) -> Option<V> {
        self.reserve(1);
        let p = self.salt.spread(q.coord_at(self.salt.pepper()));
        match self.site(p, |h| h.p == p && q.same(&h.k)) {
            Ok(i) => {
                // SAFETY: `site`'s `Ok` names a live slot.
                let slot = unsafe { self.slots[i].assume_init_mut() };
                Some(std::mem::replace(&mut slot.1, v))
            }
            Err(i) => {
                self.place(i, p, q.absorb(), v);
                None
            }
        }
    }

    /// The merge, through a view.
    /// The merge, through a view.
    pub fn insert_with_at<Q: Absorb<K>>(&mut self, q: Q, v: V, f: impl FnOnce(V, V) -> V) {
        self.reserve(1);
        let p = self.salt.spread(q.coord_at(self.salt.pepper()));
        match self.site(p, |h| h.p == p && q.same(&h.k)) {
            Ok(i) => self.merge_at(i, p, v, f),
            Err(i) => self.place(i, p, q.absorb(), v),
        }
    }

    /// Remove through a view, returning the value.
    pub fn remove_at<Q: Locate<K>>(&mut self, q: Q) -> Option<V> {
        Some(self.remove_entry_at(q)?.1)
    }

    /// Remove through a view, returning the key as well.
    pub fn remove_entry_at<Q: Locate<K>>(&mut self, q: Q) -> Option<(K, V)> {
        let p = self.salt.spread(q.coord_at(self.salt.pepper()));
        let i = self.find(p, |h| h.p == p && q.same(&h.k))?;
        Some(self.evict(i))
    }

    /// One probe, held open, through a view: the owned key is built only on a miss.
    pub fn entry_at<Q: Absorb<K>>(&mut self, q: Q) -> Entry<'_, K, V, Stored<K>> {
        self.reserve(1);
        let p = self.salt.spread(q.coord_at(self.salt.pepper()));
        match self.site(p, |h| h.p == p && q.same(&h.k)) {
            Ok(at) => Entry::Occupied(OccupiedEntry { map: self, at }),
            Err(at) => Entry::Vacant(VacantEntry {
                map: self,
                at,
                p,
                key: q.absorb(),
            }),
        }
    }
}

// -- entry -----------------------------------------------------------------------------------------

/// A located slot, occupied or not: one probe, held open.
///
/// `*m.entry(k).or_insert(0) += 1` is one probe here as it is in `std`. Written with `get` and then
/// `insert` it would be two, and the port that writes it that way is slower than the map it
/// replaced with nothing in the code to say so.
pub enum Entry<'a, K: Coord + Eq, V, H: Held<K>> {
    /// The key is present.
    Occupied(OccupiedEntry<'a, K, V, H>),
    /// The key is absent, and this is where it would go.
    Vacant(VacantEntry<'a, K, V, H>),
}

/// A slot the probe found occupied.
pub struct OccupiedEntry<'a, K: Coord + Eq, V, H: Held<K>> {
    map: &'a mut Map<K, V, H>,
    at: usize,
}

/// A slot the probe found free, and the key that would fill it.
pub struct VacantEntry<'a, K: Coord + Eq, V, H: Held<K>> {
    map: &'a mut Map<K, V, H>,
    at: usize,
    p: u64,
    key: K,
}

impl<'a, K: Coord + Eq, V, H: Held<K>> OccupiedEntry<'a, K, V, H> {
    #[must_use]
    /// The key stored here.
    pub fn key(&self) -> H::Ref<'_> {
        // SAFETY: an occupied entry names a live slot.
        unsafe { self.map.slots[self.at].assume_init_ref().0.key(self.map.salt) }
    }
    #[must_use]
    /// The value.
    pub fn get(&self) -> &V {
        // SAFETY: as above.
        unsafe { &self.map.slots[self.at].assume_init_ref().1 }
    }
    /// The value, mutably.
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: as above.
        unsafe { &mut self.map.slots[self.at].assume_init_mut().1 }
    }
    #[must_use]
    /// The value, borrowed for as long as the map is.
    pub fn into_mut(self) -> &'a mut V {
        // SAFETY: as above.
        unsafe { &mut self.map.slots[self.at].assume_init_mut().1 }
    }
    /// Replace the value, returning the old one.
    pub fn insert(&mut self, v: V) -> V {
        std::mem::replace(self.get_mut(), v)
    }
    /// Take the entry out, returning the value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
    /// Take the entry out, key and value.
    pub fn remove_entry(self) -> (K, V) {
        let at = self.at;
        self.map.evict(at)
    }
}

impl<'a, K: Coord + Eq, V, H: Held<K>> VacantEntry<'a, K, V, H> {
    #[must_use]
    /// The key that will be inserted.
    pub const fn key(&self) -> &K {
        &self.key
    }
    #[must_use]
    /// The key, owned.
    pub fn into_key(self) -> K {
        self.key
    }
    /// Fill the slot.
    pub fn insert(self, v: V) -> &'a mut V {
        let Self { map, at, p, key } = self;
        map.place(at, p, key, v);
        // SAFETY: just written.
        unsafe { &mut map.slots[at].assume_init_mut().1 }
    }
}

impl<'a, K: Coord + Eq, V, H: Held<K>> Entry<'a, K, V, H> {
    /// The value, inserting `v` if the slot is vacant.
    pub fn or_insert(self, v: V) -> &'a mut V {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(v),
        }
    }
    /// The value, calling `f` for it only if the slot is vacant.
    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(f()),
        }
    }
    /// The value, built from the key only if the slot is vacant.
    pub fn or_insert_with_key(self, f: impl FnOnce(&K) -> V) -> &'a mut V {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => {
                let v = f(&e.key);
                e.insert(v)
            }
        }
    }
    /// The value, defaulted if the slot is vacant.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
    #[must_use]
    /// Run `f` over the value if it is already there.
    pub fn and_modify(mut self, f: impl FnOnce(&mut V)) -> Self {
        if let Self::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }
}

impl<K: Coord + Eq, V, H: Held<K>> Map<K, V, H> {
    /// One probe, held open, so a read-modify-write costs one probe and not two.
    pub fn entry(&mut self, k: K) -> Entry<'_, K, V, H> {
        self.reserve(1);
        let p = H::address(self.salt, &k);
        match self.site(p, |h| h.holds(p, &k)) {
            Ok(at) => Entry::Occupied(OccupiedEntry { map: self, at }),
            Err(at) => Entry::Vacant(VacantEntry {
                map: self,
                at,
                p,
                key: k,
            }),
        }
    }
}

// -- the rest of the shape --------------------------------------------------------------------------

/// Unbounded, because a `Drop` impl may not ask for more than the type does -- and it needs
/// nothing more: the slot invariant is carried by the tag line, not by any trait.
impl<K, V, H> Drop for Map<K, V, H> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<(H, V)>() {
            for (i, &t) in self.tags.iter().enumerate() {
                if t & LIVE != 0 {
                    // SAFETY: the tag says live, and this runs once, at the end.
                    unsafe { self.slots[i].assume_init_drop() };
                }
            }
        }
    }
}

impl<K: Coord + Eq, V: Clone, H: Held<K> + Clone> Clone for Map<K, V, H> {
    /// The copy holds every entry in the *same slot* the original does, so the two iterate in the
    /// same order and probe identically. Rebuilding by reinsertion would not guarantee that once
    /// tombstones are in play.
    fn clone(&self) -> Self {
        let n = self.slots.len();
        let mut slots: Box<[MaybeUninit<(H, V)>]> = (0..n).map(|_| MaybeUninit::uninit()).collect();
        for (i, &t) in self.tags.iter().enumerate() {
            if t & LIVE != 0 {
                // SAFETY: the tag says live.
                let (h, v) = unsafe { self.slots[i].assume_init_ref() };
                slots[i] = MaybeUninit::new((h.clone(), v.clone()));
            }
        }
        Self {
            tags: self.tags.clone(),
            slots,
            live: self.live,
            used: self.used,
            geom: self.geom,
            salt: self.salt,
            _k: PhantomData,
        }
    }
}

impl<K: Clone> Clone for Stored<K> {
    fn clone(&self) -> Self {
        Self {
            p: self.p,
            k: self.k.clone(),
        }
    }
}

impl<K> Clone for Regen<K> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<K: Coord + Eq, V, H: Held<K>> std::fmt::Debug for Map<K, V, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Map<{:?}>({} live, {} tombstoned, {} slots)",
            H::FACE,
            self.live,
            self.tombstones(),
            self.slots.len()
        )
    }
}

impl<K: Coord + Eq, V, H: Held<K>> FromIterator<(K, V)> for Map<K, V, H> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(it: T) -> Self {
        let it = it.into_iter();
        let mut m = Self::with_capacity(it.size_hint().0);
        for (k, v) in it {
            m.insert(k, v);
        }
        m
    }
}

impl<K: Coord + Eq, V, H: Held<K>> Extend<(K, V)> for Map<K, V, H> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, it: T) {
        for (k, v) in it {
            self.insert(k, v);
        }
    }
}

impl<K: Coord + Eq, V: PartialEq, H: Held<K>> PartialEq for Map<K, V, H> {
    /// Set equality, not slot equality. Two maps built from the same pairs hold them in the same
    /// slots, but one that has been churned carries tombstones the other does not, and those must
    /// not make it a different map.
    fn eq(&self, other: &Self) -> bool {
        if self.live != other.live || self.salt != other.salt {
            return false;
        }
        for (i, &t) in self.tags.iter().enumerate() {
            if t & LIVE == 0 {
                continue;
            }
            // SAFETY: the tag says live.
            let (h, v) = unsafe { self.slots[i].assume_init_ref() };
            let p = h.addr();
            let Some(j) = other.find(p, |o| o.same_slot(h)) else {
                return false;
            };
            // SAFETY: `find` returns a live slot.
            if unsafe { &other.slots[j].assume_init_ref().1 } != v {
                return false;
            }
        }
        true
    }
}

impl<K: Coord + Eq, V: Eq, H: Held<K>> Eq for Map<K, V, H> {}

impl<K: Coord + Eq, V, H: Held<K>> IntoIterator for Map<K, V, H> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(mut self) -> Self::IntoIter {
        self.drain().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The carried descent and the computed one are the same function.
    ///
    /// This is the load-bearing claim of [`Descent`], and it is checked rather than argued because
    /// the two are written differently on purpose: one shifts a word it is carrying, the other
    /// recomputes the window from scratch, and they must agree at every step -- including past the
    /// point where the word runs out and the carried form has nothing left to shift.
    #[test]
    fn the_carried_descent_is_the_computed_one() {
        let mut r = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r
        };
        for gbits in 0..=16u32 {
            let groups = 1usize << gbits;
            let g = Geom::of(groups);
            for _ in 0..200 {
                let p = next();
                assert_eq!(
                    g.first(p),
                    group_at(p, 0, gbits, groups - 1) * GROUP,
                    "first group, gbits {gbits}"
                );
                let mut d = Descent::after_first(p, g);
                // Well past the point the word is spent, which is where the two implementations
                // stop being the same code and have to still be the same function.
                for j in 1..(64 / gbits.max(1) + 8) {
                    assert_eq!(
                        d.next(),
                        group_at(p, j, gbits, groups - 1) * GROUP,
                        "step {j}, gbits {gbits}, p {p:#018x}"
                    );
                }
            }
        }
    }

    /// The slot invariant, checked directly: a tag says live exactly when a slot holds an entry.
    /// Every `unsafe` in this module cites it, so it is worth a test that does not go through the
    /// public surface to look.
    #[test]
    fn a_tag_says_live_exactly_when_a_slot_holds_something() {
        let mut m: Map<String, u32> = Map::new();
        for i in 0..400u32 {
            m.insert(format!("k{i}"), i);
        }
        for i in (0..400u32).step_by(3) {
            m.remove(&format!("k{i}"));
        }
        let live = m.tags.iter().filter(|t| **t & LIVE != 0).count();
        let tombs = m.tags.iter().filter(|t| **t == TOMB).count();
        let empty = m.tags.iter().filter(|t| **t == EMPTY).count();
        assert_eq!(live, m.live);
        assert_eq!(live + tombs, m.used);
        assert_eq!(live + tombs + empty, m.tags.len());
        assert_eq!(m.tags.len(), m.slots.len());
        // Every live slot's stored address is the one the map would compute for its key.
        for (i, &t) in m.tags.iter().enumerate() {
            if t & LIVE == 0 {
                continue;
            }
            let (h, _) = unsafe { m.slots[i].assume_init_ref() };
            assert_eq!(h.addr(), Stored::<String>::address(m.salt, h.key(m.salt)));
            assert_eq!(t, tag_of(h.addr()));
        }
    }

    /// A pepper of zero must reproduce the plain bijection exactly, or every table that never sets
    /// one silently changes layout the day this type is touched.
    #[test]
    fn no_pepper_is_the_plain_bijection() {
        let s = Salt::new(0);
        for x in [0u64, 1, u64::MAX, PHI, 0x0123_4567_89AB_CDEF] {
            assert_eq!(s.spread(x), crate::coord::spread(x));
            assert_eq!(s.unspread(s.spread(x)), x);
        }
    }

    /// The multiplier a pepper produces must be odd, or the address map is not a bijection and the
    /// elliptic face regenerates the wrong key.
    #[test]
    fn every_pepper_gives_an_invertible_multiplier() {
        let mut r = 0x9E37_79B9u64;
        for _ in 0..20_000 {
            r = r.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let s = Salt::new(r);
            assert_eq!(s.mul & 1, 1, "pepper {r:#x} gave an even multiplier");
            assert_eq!(s.mul.wrapping_mul(s.inv), 1, "pepper {r:#x}");
            for x in [0u64, 1, r, u64::MAX] {
                assert_eq!(s.unspread(s.spread(x)), x, "pepper {r:#x}, x {x:#x}");
            }
        }
    }
}
