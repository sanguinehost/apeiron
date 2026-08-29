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
//! # What the probe costs, measured
//!
//! `perf stat` on a 694-key L1-resident hit loop, one core, against `hashbrown` with `foldhash`
//! -- the flat competitor, since `hashbrown` with `fxhash` is 209 ns on `highbits` against this
//! crate's 1.35 and is therefore not a comparison -- and `std` (SipHash, seeded, not reproducible).
//!
//! Instructions retired, and **not** cycles: a retired-instruction count is per-process and does
//! not move with machine load, while cycles do. This table is the load-independent column and can
//! be reproduced on a busy machine. The cycle findings are in prose underneath, each with the
//! condition it needs.
//!
//! | | instructions/lookup |
//! |---|---|
//! | apeiron, `u64` keys | **34.8** |
//! | hashbrown + foldhash | 35.3 |
//! | hashbrown + fxhash | 34.2 |
//! | std | 128.2 |
//! | apeiron, `String` keys | 93.4 |
//! | hashbrown + foldhash, `String` | 83.9 |
//! | hashbrown + fxhash, `String` | 92.3 |
//!
//! On `u64` keys the probe is half an instruction *under* the flat competitor, and within one of
//! `fxhash`, which buys its count by having no resistance to pay for. On a quiet machine that came
//! out at 6.84 cycles against foldhash's 6.62: parity in time off fewer instructions, which places
//! what remains in chain latency rather than in work. Reproducing that number needs an unloaded
//! machine, and a reading taken under load is not evidence about it either way.
//!
//! Three findings live in this table, from three rounds of measurement.
//!
//! The first was outlining. `get` and `find` were `#[inline]` but the compiler outlined them
//! anyway, so every lookup paid a call, a return, and the register save/restore around both:
//! **70 instructions and 12.0 cycles**, against hashbrown's 32 -- at an IPC already as high as
//! hashbrown's, which said the gap was instruction count and not stalls. `#[inline(always)]` on
//! the probe internals and `#[inline]` on the public readers took it to 35.9 and 6.5. Guessing
//! would not have found that; three earlier guesses at the arithmetic each bought a few percent,
//! and the whole gap was a call instruction.
//!
//! The second was the address's own latency, and it has since been retired rather than paid.
//! [`crate::coord::spread`] used to be two serial multiplies, and IPC *fell* to 4.2 when that
//! landed: an L1-resident hit is a latency chain, and the second multiply sat at the head of it
//! before the first tag line could load, for 2.3 cycles a lookup. It was paid knowingly, because
//! the placement simulation broke every cheaper address on some honest distribution. The
//! identity/placement split then removed the payment instead of accepting it. One **widening**
//! multiply produces both words at once -- `id = x * mul`, the elliptic bijection, from the low
//! half, and `place = rotl(hi, 32) ^ lo`, a hyperbolic discard, from both -- so the two words
//! now cost one multiply's latency between them rather than two in series, and the placement is
//! a function of the *full* 128-bit product rather than of a second pass over the first. The
//! staged-probe experiment that this latency kept inviting stays dead for the reason in the table
//! below: the split got the cycles back without spending any placement quality to do it.
//!
//! The third finding is that the outlining lesson had to be learned twice. The `String` row sat
//! at 112.3 for a release because `fold_bytes_with` carried no `#[inline]` -- a non-generic
//! function is codegenned once in the rlib, so every cross-crate `String` lookup paid a call and
//! the register traffic around it while fxhash inlined. One attribute took it to 97.3, and the
//! address split and the wider fold round have since taken it to 93.4, which is a shade under
//! fxhash on the same keys.
//!
//! The 9.4 instructions that remain against foldhash on `String` keys are **not in the fold**, and
//! their cause is not yet located. `examples/hash_only.rs` puts the two folds at instruction
//! parity on this corpus -- 23.1 against 24.1 on a 13-byte resref -- so the difference is
//! somewhere in the table. The obvious suspect was tested and did not survive: the slot carries
//! the address beside the key, 40 bytes against hashbrown's 32, but a map built with a
//! zero-sized value, which brings the slot back to 32, is **no faster** -- 93.2 instructions
//! against 93.4, a difference of 0.13 (`LOOKUP_ARM=apstrzst`, beside `apstr` in
//! `examples/probe_cost.rs`, so the comparison stays runnable). Slot width is therefore
//! not the explanation, and the honest position is that this gap is open. It is recorded here
//! rather than attributed, because a wrong cause in a doc comment forecloses the next
//! measurement.
//!
//! What the 40-byte slot does buy is stated where it belongs: growth is a probe per entry rather
//! than a re-fold of every key's bytes. That trade stands on its own measurement, not on this row.
//!
//! # Where the cycles sit, and what was tried against them
//!
//! `perf`'s top-down accounting on the `u64` hit loop, pinned to one P-core: 67% of issue slots
//! retire, 13% are lost to branch misprediction (one extra tag lane in ~2% of lookups; the lanes
//! an attacker cannot aim are also the lanes a predictor cannot learn), 17% to the backend --
//! and no execution port is saturated (the busiest, port 6, carries 5.9 uops against 8.8 cycles).
//! So the loop is bounded by its own uop count first (36.5 / 6-wide allocation is 6.1 cycles),
//! misprediction second, chain latency third. Measured against that ranking:
//!
//! | change | cycles/lookup | verdict |
//! |---|---|---|
//! | baseline, this table's parent commit | 8.8 | |
//! | group mask pre-scaled to bytes (`Geom`) | 8.4 | **taken**: one `shl` off the load-address chain |
//! | first window off the *first* multiply, tag off the second | 8.0 | **dead**: `the_probe_stays_flat` fails -- `highbits` mean probe 1.95 at n=5,000, one-axis `(i21,i21,i21)` 1.93 at 1.5M. The first product's window is a rank-deficient image of exactly the structured key sets the second multiply exists to spread |
//! | tag broadcast as a `u64` multiply instead of `_mm_set1_epi8` | 8.4 | no-op: LLVM canonicalises the splat and emits the identical five instructions |
//!
//! The middle row is the experiment the second multiply's latency keeps inviting: the tag-line
//! load could issue five cycles earlier, and the placement simulation said, correctly, that the
//! placement would collapse. The broadcast row says the five-instruction `movd`/`punpcklbw`/`pshuflw`/
//! `pshufd` sequence is off the critical path anyway -- the timeline has it ready two cycles
//! before the tag line arrives -- so there is nothing there to recover.
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

use crate::coord::{finish, inverse_odd, Addr, Coord, Packed, PHI, PHI_INV};
use crate::keystore::Keystore;
use crate::table::{group_at, scan, tag_of, DESCENT_BITS, GROUP};
use std::marker::PhantomData;
use std::num::NonZeroU64;
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

/// The address bijection's secret half.
///
/// A pepper cannot be *added* to a key and do any good, and finding that out is what this type is
/// for. `spread` opens with `x * M`, and every later step is a public bijection. Adding a constant
/// to `x` before the multiply is adding `c * M` after it, a translation -- and a set of keys
/// chosen to agree on their high bits still agrees on them after a translation. Measured: an
/// additive pepper left a 200,000-key aimed build 996x slower than an honest one, against 1008x
/// with no pepper at all. It bought nothing, and it looked like it should have.
///
/// (Both figures are from the pre-split design, where aiming was one `unspread` call; the same
/// quantity reads 988x in `crate::table` and in the README from a different run of it. The
/// disagreement is run-to-run noise on a ~1000x effect, and the effect is what the argument uses.
/// The closed-form attack itself is gone -- see `coord::Addr` -- but the conclusion about
/// *additive* peppering is a fact about translations and does not depend on that.)
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

    /// The address, split into identity and placement -- see [`crate::coord::Addr`] and
    /// [`crate::coord::spread`]. One widening multiply by the peppered multiplier: the low half is
    /// the identity bijection `x * mul`, the folded halves `rotl(hi, 32) ^ lo` are the placement.
    /// The pepper is the multiplier, so an attacker who does not know it cannot choose the high
    /// half `hi` that the placement fold mixes in -- which is what leaves the descent unaimable.
    #[inline]
    #[must_use]
    pub const fn spread(self, x: u64) -> Addr {
        let w = (x as u128).wrapping_mul(self.mul as u128);
        let lo = w as u64;
        let hi = (w >> 64) as u64;
        Addr { place: hi.rotate_left(32) ^ lo, id: lo }
    }

    /// The identity word undone: `id * inv`, one multiply, where `inv` is the peppered multiplier's
    /// inverse mod `2^64`. Inverts [`Addr::id`] only -- the placement fold is one-way.
    #[inline]
    #[must_use]
    pub const fn unspread(self, id: u64) -> u64 {
        id.wrapping_mul(self.inv)
    }

    /// The placement word recovered from a stored identity word. Two multiplies: undo the identity
    /// bijection to get the key's coordinate, then redo the widening multiply. Used off the hot
    /// path -- growth and diagnostics walk stored slots, which keep only the identity.
    #[inline]
    #[must_use]
    pub const fn placement_of(self, id: u64) -> u64 {
        self.spread(self.unspread(id)).place
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
    /// `DESCENT_BITS - gbits - 4`: shifts a window straight to a **byte offset** (group * GROUP),
    /// so the load address is one shift and one mask with no scale step after the mask.
    gshift: u32,
    /// `(groups - 1) * GROUP`: the group mask, pre-scaled to bytes for the same reason.
    gmask: usize,
}

impl Geom {
    fn of(groups: usize) -> Self {
        let gbits = groups.trailing_zeros();
        Self {
            gbits,
            gshift: (DESCENT_BITS - 4).saturating_sub(gbits),
            gmask: groups.saturating_sub(1) * GROUP,
        }
    }

    /// The unscaled group mask, for the sweep fallback and the rebuild, which use [`group_at`].
    fn mask(self) -> usize {
        self.gmask / GROUP
    }

    /// The first group, straight-line: one shift and one mask, from geometry the table already
    /// knows. Recomputing `groups.trailing_zeros()` per lookup put a `tzcnt` and a variable-shift
    /// setup in front of every probe; caching it is the difference between five instructions before
    /// the tag load and two.
    #[inline(always)]
    fn first(self, p: u64) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        {
            p.wrapping_shr(self.gshift) as usize & self.gmask
        }
    }
}

/// The probe order: successive `gbits`-wide windows of the whole avalanched word.
///
/// [`crate::table::group_at`] computes the `j`-th window from scratch. A probe walks its windows in
/// order, so the state can be *carried* instead: each step takes the top `gbits` of what is left
/// and shifts. The two agree step for step -- `the_carried_descent_is_the_computed_one`, in this
/// module's own tests -- and this one is a shift and a mask.
///
/// Once the word is spent the two must still agree, and there the carried form has nothing left to
/// shift, so it falls back to the function. That path is the one an attacker with a long collision
/// chain reaches, and it is deliberately the slow one: the fast path is the honest case.
struct Descent {
    p: u64,
    /// What is left of the word; the next window is its top `gbits`.
    window: u64,
    j: u32,
    /// Bits of `p` consumed. Compared against [`DESCENT_BITS`] rather than against
    /// `DESCENT_BITS / gbits`, because that quotient is an integer division on a runtime value and
    /// would sit on every probe.
    spent: u32,
    g: Geom,
}

impl Descent {
    /// Starts *after* the first group, which the caller took straight-line from [`Geom::first`].
    #[inline]
    fn after_first(p: u64, g: Geom) -> Self {
        Self {
            p,
            // Shifting the tag's bits off the top is not needed and would be wrong: `gshift` is
            // `DESCENT_BITS - gbits`, so the mask already keeps only bits below 57.
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
        if self.spent > DESCENT_BITS || self.g.gbits == 0 {
            // The word is spent, or the region is a single group and every window is group zero.
            return group_at(self.p, j, self.g.gbits, self.g.mask()) * GROUP;
        }
        #[allow(clippy::cast_possible_truncation)]
        let g = (self.window >> self.g.gshift) as usize & self.g.gmask;
        self.window <<= self.g.gbits;
        g
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

    /// The address of `k` under `pepper` -- both its words, [`Addr::place`] to descend and
    /// [`Addr::id`] to compare, from the one widening multiply.
    ///
    /// Not the same function on the two faces, and it must not be: the elliptic face addresses by
    /// [`Packed::packed`], a bijection, while the parabolic face addresses by [`Coord::coord_with`],
    /// a discard. For an integer these coincide; for `(i21, i21, i21)` they do not, and reading the
    /// wrong one would silently give the elliptic face a lossy address and with it a key compare it
    /// does not perform.
    fn address(salt: Salt, k: &K) -> Addr;

    /// Build a slot's key half from the address's **identity** word and the key that produced it.
    /// The placement word is not stored -- it is recovered from the identity when growth needs it.
    fn hold(id: u64, k: K) -> Self;

    /// The **identity** word this slot stored: `x * mul`, the bijection. *Not* the placement word
    /// -- growth and diagnostics recover placement from this through [`Salt::placement_of`]. Named
    /// apart from placement deliberately: the two used to be one word and are now different jobs.
    fn id(&self) -> u64;

    /// Whether this slot holds `k`, given `k`'s **identity** word.
    fn holds(&self, id: u64, k: &K) -> bool;

    /// Whether two slots hold the same key. Used by equality, which has a slot and not a key.
    fn same_slot(&self, other: &Self) -> bool;

    /// The placement word for a slot that only remembers its identity word.
    ///
    /// Growth and diagnostics walk stored slots and need the word the descent reads, which is not
    /// the word the slot keeps. On the elliptic face -- and on the parabolic face for a key whose
    /// fold is not already mixed -- it is recovered by running the bijection backwards and forwards
    /// again. For a key that declares [`Coord::SPREAD`] the two words are the same and there is
    /// nothing to recover; recovering anyway is what broke
    /// `a_tag_says_live_exactly_when_a_slot_holds_something`, which is why this is a trait method
    /// and not a `Salt` one.
    fn placement(&self, salt: Salt) -> u64;

    /// The key.
    ///
    /// Takes the pepper because the elliptic face **regenerates** rather than stores, and the
    /// pepper is part of the bijection it inverts. Storing it per slot instead would cost the eight
    /// bytes the face exists to save.
    fn key(&self, salt: Salt) -> Self::Ref<'_>;

    /// The key, owned, consuming the slot.
    fn into_key(self, salt: Salt) -> K;

    /// Run `f` on this slot's key as a plain `&K`, whatever the face stores.
    ///
    /// The two faces disagree about what a key *is* on the way out -- a borrow on one, a value
    /// rebuilt from the address on the other -- and a caller that wants to probe another table with
    /// it needs one shape. A callback rather than a return type is what makes that possible without
    /// allocating: the elliptic face's key is a temporary, so it can be lent but not returned.
    fn with_key<R>(&self, salt: Salt, f: impl FnOnce(&K) -> R) -> R;
}

/// **B/U, the parabolic face.** The key is carried beside its address.
///
/// Both are kept and both are read, in that order: the address first because it is already in the
/// slot the tag line pointed at, the key only when the address agreed. A seven-bit tag has a
/// 1-in-128 false-hit rate, so on the ordinary miss the key is never touched -- which is what keeps
/// a `String` table's probe off a second allocation.
pub struct Stored<K> {
    id: u64,
    k: K,
}

impl<K: Coord + Eq> Held<K> for Stored<K> {
    const FACE: Keystore = Keystore::BeingOverUnbeing;
    type Ref<'a>
        = &'a K
    where
        K: 'a;

    #[inline]
    fn address(salt: Salt, k: &K) -> Addr {
        already_mixed::<K>(salt, k.coord_with(salt.pepper()))
    }
    #[inline]
    fn hold(id: u64, k: K) -> Self {
        Self { id, k }
    }
    #[inline]
    fn id(&self) -> u64 {
        self.id
    }
    #[inline]
    fn holds(&self, id: u64, k: &K) -> bool {
        self.id == id && self.k == *k
    }
    #[inline]
    fn same_slot(&self, other: &Self) -> bool {
        self.id == other.id && self.k == other.k
    }
    #[inline]
    fn placement(&self, salt: Salt) -> u64 {
        if K::SPREAD {
            self.id
        } else {
            salt.placement_of(self.id)
        }
    }
    #[inline]
    fn with_key<R>(&self, _salt: Salt, f: impl FnOnce(&K) -> R) -> R {
        f(&self.k)
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

/// **B, the elliptic face.** The address *is* the key, so the slot keeps only the identity word.
///
/// The identity `x * mul` is a bijection and so is [`Packed::packed`], so the composite is one:
/// equal identity words are equal keys, exactly rather than probably. The key compare is not
/// skipped as an optimisation, it is **absent**, because there is no second thing to compare -- and
/// the key is read back by running the bijection the other way ([`Salt::unspread`], one multiply)
/// rather than by storing it. The placement word is not kept: `rehome` recovers it from the
/// identity when it has to re-place the slot.
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
    fn address(salt: Salt, k: &K) -> Addr {
        salt.spread(k.packed())
    }
    #[inline]
    fn hold(id: u64, _k: K) -> Self {
        Self(id, PhantomData)
    }
    #[inline]
    fn id(&self) -> u64 {
        self.0
    }
    #[inline]
    fn holds(&self, id: u64, _k: &K) -> bool {
        self.0 == id
    }
    #[inline]
    fn same_slot(&self, other: &Self) -> bool {
        self.0 == other.0
    }
    #[inline]
    fn placement(&self, salt: Salt) -> u64 {
        // The elliptic address is always the bijection, whatever the key type says about its fold.
        salt.placement_of(self.0)
    }
    #[inline]
    fn with_key<R>(&self, salt: Salt, f: impl FnOnce(&K) -> R) -> R {
        f(&K::unpack(salt.unspread(self.0)))
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

/// A small key **by value**, so a composite view need not reference every component.
///
/// `(&str, u16)` is the shape a `(String, u16)` index is actually probed with -- Lyrium keys every
/// archive by `(resref, restype)` and the type is a bare `u16` at the call site, six times over.
/// Without this the caller writes `(&str, &u16)`, which compiles and reads like a mistake.
///
/// A blanket `impl<K: Copy> Locate<K> for K` would be the obvious way to write it and does not
/// compile: it overlaps the tuple impl below, because a tuple of `Copy` views is itself `Copy` and
/// could be its own key. So it is a list, and the list is the primitives that are plausibly one
/// component of a composite key.
macro_rules! locate_by_value {
    ($($t:ty),*) => {$(
        impl Locate<$t> for $t {
            #[inline]
            fn coord_at(&self, pepper: u64) -> u64 {
                Coord::coord_with(self, pepper)
            }
            #[inline]
            fn same(&self, k: &$t) -> bool {
                self == k
            }
        }
        impl Absorb<$t> for $t {
            #[inline]
            fn absorb(self) -> $t {
                self
            }
        }
    )*};
}
locate_by_value!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, char, bool);

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

/// The parabolic face's address, skipping a second mix the key does not need.
///
/// `Salt::spread` exists to turn an unmixed word into a spread one, which is what an integer key
/// needs and a byte key does not: `fold_bytes` ends in `mix2` -- a widening multiply with its
/// halves folded and a guard laid over -- so its output is already as mixed as `spread` would make
/// it. Doing it twice puts a second widening multiply on the critical path of every string lookup
/// and buys nothing. [`Coord::SPREAD`] is how a key type says so.
///
/// Only the parabolic face may take this shortcut. The elliptic face's address must be the
/// bijection whatever the key type, because it reads the key back off it -- and `fold_bytes` is a
/// discard, so there is nothing to read back.
///
/// Both words come out equal here, and that is right rather than a shortcut: on this face the
/// identity word is an early-reject hint before the key compare, not a key in its own right, so it
/// has no injectivity to preserve.
#[inline]
fn already_mixed<K: Coord + Eq + ?Sized>(salt: Salt, c: u64) -> Addr {
    if K::SPREAD {
        Addr { place: c, id: c }
    } else {
        salt.spread(c)
    }
}

/// A map on one of the two keyed faces.
///
/// `Map<K, V>` is the parabolic default and takes any `K: Coord + Eq`. [`Words`] is the elliptic
/// alias, for keys that fit in a word.
///
/// # The header is 88 bytes, against `std`'s 48
///
/// Two boxed slices (32), the counters (16), `Geom` (16) and [`Salt`] (24). The size is not
/// free: the same map-of-maps workload the field order was measured on runs 25% slower with the
/// header padded to 128, so a struct holding many maps does pay for these bytes. What 88 buys is
/// on the probe: `Geom` cached is two instructions before the tag load against five recomputed
/// (`trailing_zeros` plus variable-shift setup, measured when it was added), and `Salt::inv` is
/// what lets the elliptic face read keys back without a 21-instruction Newton inversion per
/// entry. 64 is reachable -- derive the lengths from `gmask`, recompute `inv` on the readback
/// path -- but it is thin-pointer surgery through every slice access, and the field order below
/// already claimed the cheap part of the win; measured as worth revisiting, not taken yet.
pub struct Map<K, V, H = Stored<K>> {
    /// One byte per slot: [`EMPTY`], [`TOMB`], or a live slot's seven-bit tag with [`LIVE`] set.
    /// Sixteen of these are one vector compare, which is what makes a group scan one instruction
    /// rather than sixteen loads.
    tags: Box<[u8]>,
    /// The entries. **`slots[i]` is initialised exactly when `tags[i] & LIVE != 0`.** That is the
    /// whole safety invariant of this module; every `unsafe` below cites it and nothing else.
    slots: Box<[MaybeUninit<(H, V)>]>,
    /// The region's probe geometry, recomputed only when the region is.
    ///
    /// Sits here, not after the counters: a lookup reads `tags`, `slots`, `geom` and `salt.mul`
    /// and nothing else, and in this order the four fit the first 64 bytes of the struct -- one
    /// cache line instead of two when the map itself is cold. Measured on 65,536 four-entry maps
    /// probed at random: 57.0 against 61.7 cycles a lookup, interleaved runs. The counters are
    /// write-path fields and ride behind.
    geom: Geom,
    salt: Salt,
    /// Live entries.
    live: usize,
    /// Live entries plus tombstones -- what the fill limit is measured against, because a tombstone
    /// occupies probe order even though it holds nothing.
    used: usize,
    _k: PhantomData<K>,
}

/// The elliptic face: keys that fit in a word, addressed bijectively, with no key stored.
pub type Words<K, V> = Map<K, V, Regen<K>>;

// `Send` and `Sync` are left to the automatic impls. This once carried a hand-written pair on the
// belief that `MaybeUninit` blocks them; it does not -- `MaybeUninit<T>` is `Send`/`Sync` when `T`
// is, so the derived bounds already read the real field types. Two `unsafe` assertions that agreed
// with the compiler were two more than the crate needed.

impl<K: Coord + Eq, V, H: Held<K>> Default for Map<K, V, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Coord + Eq, V, H: Held<K>> Map<K, V, H> {
    /// An empty map. Allocates nothing, so one that is declared and never filled costs a few words.
    ///
    /// Takes the pepper from [`crate::pepper`], which is zero unless the process set one at
    /// startup. That is the floor: a map written by someone who was thinking about parsing rather
    /// than about addressing still gets whatever protection the process asked for, without the
    /// call site mentioning it. [`Map::with_pepper`] overrides it, and [`Exposed`] refuses to exist
    /// without one.
    #[must_use]
    pub fn new() -> Self {
        Self::with_salt(Salt::new(crate::pepper::process_pepper()))
    }

    /// An empty map with a given address bijection, ignoring the process pepper.
    ///
    /// The analogue of `HashMap::with_hasher`. There is no `BuildHasher` here because there is no
    /// hasher to build -- the address function is fixed and the [`Salt`] is the only thing that
    /// varies, which is what makes the layout reproducible from a recorded constant.
    #[must_use]
    pub fn with_salt(salt: Salt) -> Self {
        Self {
            tags: Box::new([]),
            slots: Box::new([]),
            live: 0,
            used: 0,
            geom: Geom::default(),
            salt,
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
    /// *Aiming.* An attacker who can choose keys can try to crowd them into one group. They cannot
    /// *solve* for it -- the placement word is a discard, see [`crate::coord::Addr`] -- but the
    /// function is public, so they can evaluate it and keep what lands where they want. That costs
    /// `2^(j * gbits)` trials to match `j` windows of the descent, which is exponential in the
    /// chain length for damage that is only linear in it: measured, one window is free and doubles
    /// the mean probe, two cost 67,682 trials per key for a third step, and three are unaffordable.
    ///
    /// A pepper breaks the filter itself. The multiplier is the secret, so an attacker cannot
    /// evaluate the placement word the table will use, and the keys they aimed land as randomly as
    /// honest ones -- measured at mean probe 1.00 against 2.00 unpeppered.
    ///
    /// A pepper is a process-wide **constant**, not a per-run random seed. Choose it once at
    /// startup, log it, and a recorded session replays exactly -- determinism and unpredictability
    /// are not in conflict, because only reproducibility of *a recorded run* was ever needed.
    #[must_use]
    pub fn with_pepper(pepper: u64) -> Self {
        Self::with_salt(Salt::new(pepper))
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

    /// What this table currently looks like, measured rather than tracked.
    ///
    /// The probe lengths are computed by re-probing every live entry, so this is `O(n)` and costs
    /// **nothing** on the insert and lookup paths. That is the deliberate trade: a counter
    /// maintained on the hot path would be a store per insert forever, to serve a question that
    /// gets asked when someone is already worried.
    ///
    /// `longest_probe` is the number to watch. An honest key set keeps it in the low single digits
    /// whatever the size; an aimed one drives it to hundreds or thousands, because every aimed key
    /// walks past every other one. That is what makes an attack *visible* instead of merely slow --
    /// see [`crate::map::Salt`] for what to do about it.
    #[must_use]
    pub fn stats(&self) -> Stats {
        let mut longest = 0u32;
        let mut total = 0u64;
        let mut compared = 0u64;
        for (i, &t) in self.tags.iter().enumerate() {
            if t & LIVE == 0 {
                continue;
            }
            // SAFETY: the tag says live.
            let (h, _) = unsafe { self.slots[i].assume_init_ref() };
            let (steps, compares) = self.probe_cost(h.placement(self.salt), i);
            longest = longest.max(steps);
            total += u64::from(steps);
            compared += u64::from(compares);
        }
        Stats {
            face: H::FACE,
            live: self.live,
            tombstones: self.used - self.live,
            slots: self.slots.len(),
            peppered: self.salt.pepper() != 0,
            longest_probe: longest,
            mean_probe: if self.live == 0 {
                0.0
            } else {
                total as f64 / self.live as f64
            },
            mean_compares: if self.live == 0 {
                0.0
            } else {
                compared as f64 / self.live as f64
            },
        }
    }

    /// How many groups the descent visits before reaching slot `at`, and how many full addresses
    /// the tag lines make it compare on the way.
    ///
    /// One rather than zero for an entry found in the first group, so the means read as "groups
    /// touched per lookup" and "addresses compared per lookup", and a perfect table reads 1.0 in
    /// both. The compare count replays what `find` does: every matching tag in a group it passes
    /// through, and in the group it stops at, the matching lanes the scan visits up to its own.
    fn probe_cost(&self, p: u64, at: usize) -> (u32, u32) {
        let tag = tag_of(p);
        let mut base = self.geom.first(p);
        let mut d = Descent::after_first(p, self.geom);
        let mut steps = 1u32;
        let mut compares = 0u32;
        // Bounded by the number of groups: the descent's sweep visits each at most once before
        // repeating, and the entry is certainly in one of them.
        let cap = (self.slots.len() / GROUP).max(1) as u32 + 1;
        while steps <= cap {
            if at >= base && at < base + GROUP {
                for lane in base..=at {
                    compares += u32::from(self.tags[lane] == tag);
                }
                return (steps, compares);
            }
            for lane in base..base + GROUP {
                compares += u32::from(self.tags[lane] == tag);
            }
            base = d.next();
            steps += 1;
        }
        (steps, compares)
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
    fn site(&self, p: u64, is: impl Fn(&H) -> bool) -> Result<usize, usize> {
        // A real assertion, not a `debug_assert`. `line` reads sixteen tag bytes without a bounds
        // check, so an unallocated table would read past an empty box -- and the only thing
        // standing between this and that was every caller happening to call `reserve(1)` first.
        // One predictable branch on a path that already allocates is the right price for turning
        // undefined behaviour into a panic.
        assert!(self.slots.len() >= MIN_SLOTS, "probing an unallocated table");
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

    #[inline(always)]
    fn line(&self, base: usize) -> &[u8; GROUP] {
        debug_assert!(base + GROUP <= self.tags.len(), "group {base} is outside the region");
        // SAFETY: `Descent` yields a group start below `tags.len() / GROUP`, so the sixteen bytes
        // from there are inside the array. `tags.len() == slots.len()` and both are a multiple of
        // `GROUP` -- `rebuild` is the only thing that sizes them and it allocates in groups.
        unsafe { &*self.tags.as_ptr().add(base).cast::<[u8; GROUP]>() }
    }

    // -- reading -------------------------------------------------------------------------------

    #[must_use]
    /// The value at `k`.
    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        let a = H::address(self.salt, k);
        let i = self.find(a.place, |h| h.holds(a.id, k))?;
        // SAFETY: `find` returns only live slots.
        Some(unsafe { &self.slots[i].assume_init_ref().1 })
    }

    /// The value at `k`, mutably.
    #[inline]
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let a = H::address(self.salt, k);
        let i = self.find(a.place, |h| h.holds(a.id, k))?;
        // SAFETY: as `get`.
        Some(unsafe { &mut self.slots[i].assume_init_mut().1 })
    }

    #[must_use]
    /// Whether `k` is present.
    #[inline]
    pub fn contains_key(&self, k: &K) -> bool {
        let a = H::address(self.salt, k);
        self.find(a.place, |h| h.holds(a.id, k)).is_some()
    }

    /// The stored key and its value.
    #[must_use]
    pub fn get_key_value(&self, k: &K) -> Option<(H::Ref<'_>, &V)> {
        let a = H::address(self.salt, k);
        let i = self.find(a.place, |h| h.holds(a.id, k))?;
        // SAFETY: as `get`.
        let (h, v) = unsafe { self.slots[i].assume_init_ref() };
        Some((h.key(self.salt), v))
    }

    // -- writing -------------------------------------------------------------------------------

    /// Insert, returning what was displaced.
    ///
    /// Takes the key by value and clones nothing: the probe borrows it, and the borrow is over
    /// before either arm needs to own it.
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.reserve(1);
        let a = H::address(self.salt, &k);
        match self.site(a.place, |h| h.holds(a.id, &k)) {
            Ok(i) => {
                // SAFETY: `site`'s `Ok` names a live slot.
                let slot = unsafe { self.slots[i].assume_init_mut() };
                Some(std::mem::replace(&mut slot.1, v))
            }
            Err(i) => {
                self.place(i, a, k, v);
                None
            }
        }
    }

    /// **The merge.** Combine `v` into whatever is already at `k`, in one probe.
    ///
    /// `f` takes the slot **by reference**, not by value. That is what makes it panic-safe rather
    /// than panic-tolerant: the earlier form was `FnOnce(V, V) -> V`, which had to move the old
    /// value out of the slot to call `f`, and a panic inside `f` left an initialised-looking slot
    /// holding nothing. The entry was lost. Here the value never leaves, so there is no window to
    /// unwind out of -- the failure mode is gone by construction, not handled.
    ///
    /// This is the primitive; every other writer here is a special case of it. It is also the one
    /// the algebra has something to say about: `fold`'s commutativity and associativity hold on
    /// every face with no hypothesis on the dial, so which merge is chosen never depends on the
    /// keystore. **One merge, three keystores** -- see [`crate::keystore`], and [`crate::merge`]
    /// for the four that a table is ordinarily built with.
    pub fn insert_with(&mut self, k: K, v: V, f: impl FnOnce(&mut V, V)) {
        self.reserve(1);
        let a = H::address(self.salt, &k);
        match self.site(a.place, |h| h.holds(a.id, &k)) {
            Ok(i) => {
                // SAFETY: `site`'s `Ok` names a live slot.
                let slot = unsafe { self.slots[i].assume_init_mut() };
                f(&mut slot.1, v);
            }
            Err(i) => self.place(i, a, k, v),
        }
    }

    /// Write a fresh entry into a vacant slot. Stores the identity word; the tag comes off the
    /// placement word. Both are already in `a` from the one address multiply.
    #[inline]
    fn place(&mut self, i: usize, a: Addr, k: K, v: V) {
        if self.tags[i] == EMPTY {
            self.used += 1;
        }
        self.live += 1;
        self.slots[i] = MaybeUninit::new((H::hold(a.id, k), v));
        self.tags[i] = tag_of(a.place);
    }

    /// Remove, returning the value.
    ///
    /// The slot becomes a **tombstone**, not empty. A window descent has the same requirement a
    /// linear probe does: a key placed later in probe order was placed *because* this slot was
    /// occupied, so emptying it would make that key unreachable while it is still in the table. The
    /// tombstone keeps the probe walking, and the next insert that passes it takes it, so a table
    /// filled and drained repeatedly does not grow.
    #[inline]
    pub fn remove(&mut self, k: &K) -> Option<V> {
        Some(self.remove_entry(k)?.1)
    }

    /// Remove, returning the key as well.
    pub fn remove_entry(&mut self, k: &K) -> Option<(K, V)> {
        let a = H::address(self.salt, k);
        let i = self.find(a.place, |h| h.holds(a.id, k))?;
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
        self.rebuild(Self::region_for(want).max(self.slots.len()));
    }

    /// Move every live entry into a fresh region of `slots` slots.
    ///
    /// Nothing is re-folded. The address is in the slot, and it is the *avalanched* address, so the
    /// new placement is a descent over a word that already exists: growth costs a probe per entry
    /// and not a hash per entry. For a hundred thousand strings that is the difference between
    /// re-reading megabytes of bytes and reading a word.
    fn rebuild(&mut self, slots: usize) {
        let tags = vec![EMPTY; slots].into_boxed_slice();
        let fresh: Box<[MaybeUninit<(H, V)>]> = (0..slots).map(|_| MaybeUninit::uninit()).collect();
        self.rehome(tags, fresh);
    }

    /// Move every live entry into arrays the caller allocated.
    ///
    /// Split from [`Self::rebuild`] so [`Self::try_reserve`] can allocate fallibly and then hand
    /// the result here -- the placement is the same either way, and having one copy of it means a
    /// fallible path cannot drift from the infallible one.
    ///
    /// Nothing is re-folded. The address is in the slot, and it is the *avalanched* address, so a
    /// move costs a probe per entry and not a hash per entry. For a hundred thousand strings that
    /// is the difference between re-reading megabytes of bytes and reading a word.
    fn rehome(&mut self, mut tags: Box<[u8]>, mut fresh: Box<[MaybeUninit<(H, V)>]>) {
        let slots = tags.len();
        debug_assert!(slots >= MIN_SLOTS && slots.is_power_of_two());
        debug_assert_eq!(slots, fresh.len());
        debug_assert!(limit(slots) >= self.live, "rehoming into a region that cannot hold it");
        let geometry = Geom::of(slots / GROUP);
        let (gbits, gmask) = (geometry.gbits, geometry.mask());

        // `self` is handed empty arrays now, so from here until the swap at the end it owns
        // nothing -- `Map::drop` would find a zero-length tag line and free none of the entries
        // still sitting in the old region. `Salvage` is what covers that gap.
        let mut old = Salvage {
            tags: std::mem::replace(&mut self.tags, Box::new([])),
            slots: std::mem::replace(&mut self.slots, Box::new([])),
            taken: 0,
        };
        for i in 0..old.tags.len() {
            old.taken = i + 1;
            if old.tags[i] & LIVE == 0 {
                continue;
            }
            // SAFETY: the tag says live, and `taken` now covers this index, so the salvage guard
            // will not drop it again if anything below unwinds.
            let entry = unsafe { old.slots[i].assume_init_read() };
            // The slot stored only the identity word; the descent runs on the placement word.
            // Recover it -- one multiply back to the coordinate, one widening multiply forward --
            // rather than re-folding the key, so growth stays a probe per entry and not a hash per
            // entry even though the placement is no longer sitting in the slot.
            let p = entry.0.placement(self.salt);
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

    /// The smallest region that holds `n` entries.
    fn region_for(n: usize) -> usize {
        region_for(n)
    }

    /// [`Self::reserve`], reporting an allocation failure instead of aborting on it.
    ///
    /// # Errors
    ///
    /// [`AllocFailed`] when the region cannot be allocated. A persistent world that has just been
    /// handed an archive header claiming four billion resources would rather refuse the file than
    /// take the process down, and `reserve` gives it no way to.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), AllocFailed> {
        if self.used + additional <= limit(self.slots.len()) {
            return Ok(());
        }
        let want = self.live + additional;
        let slots = Self::region_for(want).max(self.slots.len());

        let mut tags: Vec<u8> = Vec::new();
        tags.try_reserve_exact(slots).map_err(|_| AllocFailed { slots })?;
        tags.resize(slots, EMPTY);

        let mut fresh: Vec<MaybeUninit<(H, V)>> = Vec::new();
        fresh.try_reserve_exact(slots).map_err(|_| AllocFailed { slots })?;
        // SAFETY: the capacity was just reserved, and `MaybeUninit<T>` needs no initialisation --
        // that is the whole reason a slot array can be `calloc`-cheap.
        unsafe { fresh.set_len(slots) };

        self.rehome(tags.into_boxed_slice(), fresh.into_boxed_slice());
        Ok(())
    }

    /// Give back every slot the entries do not need.
    ///
    /// Also drops every tombstone, since a rehome rebuilds placement from scratch.
    pub fn shrink_to_fit(&mut self) {
        self.shrink_to(0);
    }

    /// Shrink, but keep room for at least `min` entries.
    pub fn shrink_to(&mut self, min: usize) {
        if self.slots.is_empty() {
            return;
        }
        let want = Self::region_for(self.live.max(min));
        if want < self.slots.len() {
            self.rebuild(want);
        }
    }

    /// Every entry, in slot order -- which is address order, so neither insertion order nor sorted.
    /// It is however the *same* order for the same keys under the same pepper, on every machine and
    /// every run: nothing here is seeded and nothing is randomised. That is the property 13 of
    /// Lyrium's 14 `BTreeSet`s were reaching for when they paid for an ordering they never queried.
    pub fn iter(&self) -> Iter<'_, K, V, H> {
        Iter { map: self, i: 0, left: self.live }
    }

    /// Every entry, with the value mutable.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, H> {
        let (n, left) = (self.tags.len(), self.live);
        IterMut { map: std::ptr::from_mut(self), i: 0, n, left, _m: PhantomData }
    }

    /// The keys, in slot order.
    pub fn keys(&self) -> Keys<'_, K, V, H> {
        Keys(self.iter())
    }

    /// The values, in slot order.
    pub fn values(&self) -> Values<'_, K, V, H> {
        Values(self.iter())
    }

    /// The values, in slot order, mutably.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V, H> {
        ValuesMut(self.iter_mut())
    }

    /// Take every entry out, keeping the allocation.
    ///
    /// A real iterator, not a `Vec`. Draining a hundred thousand entries to look at them one at a
    /// time used to allocate a hundred thousand entries' worth of somewhere else to look at them.
    /// Dropping it early still empties the map, which is what makes it a drain rather than an
    /// iterator that happens to remove things.
    pub fn drain(&mut self) -> Drain<'_, K, V, H> {
        Drain { map: self, i: 0 }
    }

    /// The keys, consuming the map.
    pub fn into_keys(self) -> IntoKeys<K, V, H> {
        IntoKeys(self.into_iter())
    }

    /// The values, consuming the map.
    pub fn into_values(self) -> IntoValues<K, V, H> {
        IntoValues(self.into_iter())
    }

    /// Take out every entry `pred` accepts, leaving the rest.
    ///
    /// Removal is by tombstone, as everywhere else here. Dropping the iterator early stops the scan
    /// and leaves what it has not reached, matching `std`.
    pub fn extract_if<F: FnMut(H::Ref<'_>, &mut V) -> bool>(
        &mut self,
        pred: F,
    ) -> ExtractIf<'_, K, V, H, F> {
        ExtractIf { map: self, i: 0, pred }
    }

    /// Whether `f` holds for every key. Stops at the first that fails.
    ///
    /// Takes a `&K` on both faces, via [`Held::with_key`], which is what makes it usable to probe
    /// a *second* table -- the thing set algebra is made of.
    pub fn all_keys(&self, mut f: impl FnMut(&K) -> bool) -> bool {
        for (i, &t) in self.tags.iter().enumerate() {
            if t & LIVE == 0 {
                continue;
            }
            // SAFETY: the tag says live.
            let (h, _) = unsafe { self.slots[i].assume_init_ref() };
            if !h.with_key(self.salt, &mut f) {
                return false;
            }
        }
        true
    }

    /// Visit every key as a `&K`.
    pub fn for_each_key(&self, mut f: impl FnMut(&K)) {
        self.all_keys(|k| {
            f(k);
            true
        });
    }

    /// Several values at once, mutably, from keys that must be distinct.
    ///
    /// # Panics
    ///
    /// If two of `ks` name the same slot, as `std` does. Handing out two `&mut` to one value is the
    /// thing this exists to make impossible, so it is a bug in the caller rather than a `None` to
    /// thread through.
    pub fn get_disjoint_mut<const N: usize>(&mut self, ks: [&K; N]) -> [Option<&mut V>; N] {
        let mut at: [Option<usize>; N] = [None; N];
        for (slot, k) in at.iter_mut().zip(ks) {
            let a = H::address(self.salt, k);
            *slot = self.find(a.place, |h| h.holds(a.id, k));
        }
        for i in 0..N {
            for j in 0..i {
                assert!(
                    at[i].is_none() || at[i] != at[j],
                    "get_disjoint_mut was given the same key twice"
                );
            }
        }
        let base = self.slots.as_mut_ptr();
        at.map(|found| {
            found.map(|i| {
                // SAFETY: `find` returns live slots; the indices were just checked pairwise
                // distinct; each is yielded once. So no two of the returned references alias, which
                // is the whole precondition -- the raw round trip is only what lets N of them live
                // at the same time.
                unsafe { &mut (*base.add(i)).assume_init_mut().1 }
            })
        })
    }
}

// -- iterators -------------------------------------------------------------------------------------
//
// All of these walk the tag line and read the slots it says are live, which is the slot invariant
// and the only thing their `unsafe` rests on. They are named types rather than `impl Iterator`
// because `std`'s are, because a named type can carry `ExactSizeIterator`, and because `Drain` has
// to have a `Drop`.

/// [`Map::iter`].
pub struct Iter<'a, K: Coord + Eq, V, H: Held<K>> {
    map: &'a Map<K, V, H>,
    i: usize,
    left: usize,
}

impl<'a, K: Coord + Eq, V, H: Held<K>> Iterator for Iter<'a, K, V, H> {
    type Item = (H::Ref<'a>, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.i < self.map.tags.len() {
            let i = self.i;
            self.i += 1;
            if self.map.tags[i] & LIVE == 0 {
                continue;
            }
            self.left -= 1;
            // SAFETY: the tag says live.
            let (h, v) = unsafe { self.map.slots[i].assume_init_ref() };
            return Some((h.key(self.map.salt), v));
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for Iter<'_, K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> std::iter::FusedIterator for Iter<'_, K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> Clone for Iter<'_, K, V, H> {
    fn clone(&self) -> Self {
        Self { map: self.map, i: self.i, left: self.left }
    }
}

/// [`Map::keys`].
pub struct Keys<'a, K: Coord + Eq, V, H: Held<K>>(Iter<'a, K, V, H>);
impl<'a, K: Coord + Eq, V, H: Held<K>> Iterator for Keys<'a, K, V, H> {
    type Item = H::Ref<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, _)| k)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for Keys<'_, K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> std::iter::FusedIterator for Keys<'_, K, V, H> {}

/// [`Map::values`].
pub struct Values<'a, K: Coord + Eq, V, H: Held<K>>(Iter<'a, K, V, H>);
impl<'a, K: Coord + Eq, V, H: Held<K>> Iterator for Values<'a, K, V, H> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for Values<'_, K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> std::iter::FusedIterator for Values<'_, K, V, H> {}

/// [`Map::values_mut`].
pub struct ValuesMut<'a, K: Coord + Eq, V, H: Held<K>>(IterMut<'a, K, V, H>);
impl<'a, K: Coord + Eq, V, H: Held<K>> Iterator for ValuesMut<'a, K, V, H> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(_, v)| v)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for ValuesMut<'_, K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> std::iter::FusedIterator for ValuesMut<'_, K, V, H> {}

/// [`Map::drain`].
pub struct Drain<'a, K: Coord + Eq, V, H: Held<K>> {
    map: &'a mut Map<K, V, H>,
    i: usize,
}

impl<K: Coord + Eq, V, H: Held<K>> Drain<'_, K, V, H> {
    /// Take the next entry out, clearing tombstones on the way past.
    ///
    /// The counters move **per entry**, not at the end. A `Drain` that is forgotten rather than
    /// dropped then leaves the map holding exactly what the iterator did not reach -- a smaller
    /// map, not a corrupt one.
    fn take_next(&mut self) -> Option<(K, V)> {
        while self.i < self.map.tags.len() {
            let i = self.i;
            self.i += 1;
            let tag = self.map.tags[i];
            if tag == EMPTY {
                continue;
            }
            self.map.tags[i] = EMPTY;
            self.map.used -= 1;
            if tag & LIVE == 0 {
                continue; // a tombstone: it occupied `used`, it holds nothing
            }
            self.map.live -= 1;
            // SAFETY: the tag said live and now says empty, so nothing else will read or drop it.
            let (h, v) = unsafe { self.map.slots[i].assume_init_read() };
            return Some((h.into_key(self.map.salt), v));
        }
        None
    }
}

impl<K: Coord + Eq, V, H: Held<K>> Iterator for Drain<'_, K, V, H> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        self.take_next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.map.live))
    }
}
impl<K: Coord + Eq, V, H: Held<K>> std::iter::FusedIterator for Drain<'_, K, V, H> {}

impl<K: Coord + Eq, V, H: Held<K>> Drop for Drain<'_, K, V, H> {
    fn drop(&mut self) {
        while self.take_next().is_some() {}
    }
}

/// [`Map::extract_if`].
pub struct ExtractIf<'a, K: Coord + Eq, V, H: Held<K>, F> {
    map: &'a mut Map<K, V, H>,
    i: usize,
    pred: F,
}

impl<K: Coord + Eq, V, H: Held<K>, F> Iterator for ExtractIf<'_, K, V, H, F>
where
    F: FnMut(H::Ref<'_>, &mut V) -> bool,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.i < self.map.tags.len() {
            let i = self.i;
            self.i += 1;
            if self.map.tags[i] & LIVE == 0 {
                continue;
            }
            let salt = self.map.salt;
            // SAFETY: the tag says live. The two references are to disjoint fields of one entry.
            let entry = unsafe { self.map.slots[i].assume_init_mut() };
            let (h, v) = (&entry.0, &mut entry.1);
            if (self.pred)(h.key(salt), v) {
                return Some(self.map.evict(i));
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.map.live))
    }
}
impl<K: Coord + Eq, V, H: Held<K>, F> std::iter::FusedIterator for ExtractIf<'_, K, V, H, F> where
    F: FnMut(H::Ref<'_>, &mut V) -> bool
{
}

/// [`Map::into_iter`]. Owns the region, so a partially-consumed one still drops the rest.
pub struct IntoIter<K: Coord + Eq, V, H: Held<K>> {
    tags: Box<[u8]>,
    slots: Box<[MaybeUninit<(H, V)>]>,
    salt: Salt,
    i: usize,
    left: usize,
    _k: PhantomData<K>,
}

impl<K: Coord + Eq, V, H: Held<K>> Iterator for IntoIter<K, V, H> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        while self.i < self.tags.len() {
            let i = self.i;
            self.i += 1;
            if self.tags[i] & LIVE == 0 {
                continue;
            }
            self.tags[i] = EMPTY;
            self.left -= 1;
            // SAFETY: the tag said live and now says empty, so `Drop` will not take it again.
            let (h, v) = unsafe { self.slots[i].assume_init_read() };
            return Some((h.into_key(self.salt), v));
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for IntoIter<K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> std::iter::FusedIterator for IntoIter<K, V, H> {}

impl<K: Coord + Eq, V, H: Held<K>> Drop for IntoIter<K, V, H> {
    fn drop(&mut self) {
        if !std::mem::needs_drop::<(H, V)>() {
            return;
        }
        for i in self.i..self.tags.len() {
            if self.tags[i] & LIVE != 0 {
                // SAFETY: past the cursor, so `next` has not taken it.
                unsafe { self.slots[i].assume_init_drop() };
            }
        }
    }
}

/// [`Map::into_keys`].
pub struct IntoKeys<K: Coord + Eq, V, H: Held<K>>(IntoIter<K, V, H>);
impl<K: Coord + Eq, V, H: Held<K>> Iterator for IntoKeys<K, V, H> {
    type Item = K;
    fn next(&mut self) -> Option<K> {
        self.0.next().map(|(k, _)| k)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for IntoKeys<K, V, H> {}

/// [`Map::into_values`].
pub struct IntoValues<K: Coord + Eq, V, H: Held<K>>(IntoIter<K, V, H>);
impl<K: Coord + Eq, V, H: Held<K>> Iterator for IntoValues<K, V, H> {
    type Item = V;
    fn next(&mut self) -> Option<V> {
        self.0.next().map(|(_, v)| v)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for IntoValues<K, V, H> {}

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
    left: usize,
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
                self.left -= 1;
                let entry = m.slots[i].as_mut_ptr();
                let h: &'a H = &*std::ptr::addr_of!((*entry).0);
                let v: &'a mut V = &mut *std::ptr::addr_of_mut!((*entry).1);
                return Some((h.key(salt), v));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for IterMut<'_, K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> std::iter::FusedIterator for IterMut<'_, K, V, H> {}

/// The smallest region that holds `n` entries.
///
/// A free function so it can be proved on its own. It is the sizing rule behind `reserve`,
/// `try_reserve` and -- new, and the reason this is worth pulling out -- `shrink_to`, which is the
/// first caller that makes the region *smaller*. Growing into a region that is too small is caught
/// by the next insert; shrinking into one is not, so the rule now has to be right in a direction it
/// was never exercised in.
#[must_use]
pub fn region_for(n: usize) -> usize {
    let mut slots = MIN_SLOTS;
    while limit(slots) < n {
        slots *= 2;
    }
    slots
}

/// A region could not be allocated.
///
/// `std`'s `TryReserveError` has no public constructor, so this is the crate's own. It carries the
/// slot count that failed, because "we could not allocate" is not actionable and "we could not
/// allocate 2^31 slots" names the input that asked for it -- which, for a parser handed an archive
/// header claiming four billion resources, is the whole diagnosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocFailed {
    /// Slots the failed region would have held.
    pub slots: usize,
}

impl std::fmt::Display for AllocFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "could not allocate a region of {} slots", self.slots)
    }
}

impl std::error::Error for AllocFailed {}

/// The old region, while [`Map::rehome`] is walking it.
///
/// `rehome` takes the arrays off the map before it starts moving entries across, which means that
/// for the length of the loop the map owns nothing and its `Drop` would free nothing. If the loop
/// unwound there, every entry not yet moved would leak. This owns them for that window and drops
/// whatever is left.
///
/// `taken` is the number of slots already accounted for -- moved out, or skipped as not live -- so
/// the guard drops exactly the tail it is still responsible for and never an entry twice.
struct Salvage<H, V> {
    tags: Box<[u8]>,
    slots: Box<[MaybeUninit<(H, V)>]>,
    taken: usize,
}

impl<H, V> Drop for Salvage<H, V> {
    fn drop(&mut self) {
        if !std::mem::needs_drop::<(H, V)>() {
            return;
        }
        for i in self.taken..self.tags.len() {
            if self.tags[i] & LIVE != 0 {
                // SAFETY: the tag says live and this index is past `taken`, so nothing has moved
                // it out. Each index is visited once.
                unsafe { self.slots[i].assume_init_drop() };
            }
        }
    }
}

// -- the borrowed-key surface, parabolic face only ------------------------------------------------

impl<K: Coord + Eq, V> Map<K, V, Stored<K>> {
    /// Read through a borrowed view: `map.get_at("resref")` on a `Map<String, V>`.
    #[must_use]
    #[inline]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        let i = self.find(a.place, |h| h.id == a.id && q.same(&h.k))?;
        // SAFETY: `find` returns only live slots.
        Some(unsafe { &self.slots[i].assume_init_ref().1 })
    }

    /// The value a view names, mutably.
    #[inline]
    pub fn get_mut_at<Q: Locate<K>>(&mut self, q: Q) -> Option<&mut V> {
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        let i = self.find(a.place, |h| h.id == a.id && q.same(&h.k))?;
        // SAFETY: as `get_at`.
        Some(unsafe { &mut self.slots[i].assume_init_mut().1 })
    }

    #[must_use]
    /// Whether a view's key is present.
    #[inline]
    pub fn contains_at<Q: Locate<K>>(&self, q: Q) -> bool {
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        self.find(a.place, |h| h.id == a.id && q.same(&h.k)).is_some()
    }

    /// Insert through a view, taking ownership of the key only on a miss.
    #[inline]
    pub fn insert_at<Q: Absorb<K>>(&mut self, q: Q, v: V) -> Option<V> {
        self.reserve(1);
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        match self.site(a.place, |h| h.id == a.id && q.same(&h.k)) {
            Ok(i) => {
                // SAFETY: `site`'s `Ok` names a live slot.
                let slot = unsafe { self.slots[i].assume_init_mut() };
                Some(std::mem::replace(&mut slot.1, v))
            }
            Err(i) => {
                self.place(i, a, q.absorb(), v);
                None
            }
        }
    }

    /// The merge, through a view.
    /// The merge, through a view.
    pub fn insert_with_at<Q: Absorb<K>>(&mut self, q: Q, v: V, f: impl FnOnce(&mut V, V)) {
        self.reserve(1);
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        match self.site(a.place, |h| h.id == a.id && q.same(&h.k)) {
            Ok(i) => {
                // SAFETY: `site`'s `Ok` names a live slot.
                let slot = unsafe { self.slots[i].assume_init_mut() };
                f(&mut slot.1, v);
            }
            Err(i) => self.place(i, a, q.absorb(), v),
        }
    }

    /// Remove through a view, returning the value.
    pub fn remove_at<Q: Locate<K>>(&mut self, q: Q) -> Option<V> {
        Some(self.remove_entry_at(q)?.1)
    }

    /// Remove through a view, returning the key as well.
    pub fn remove_entry_at<Q: Locate<K>>(&mut self, q: Q) -> Option<(K, V)> {
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        let i = self.find(a.place, |h| h.id == a.id && q.same(&h.k))?;
        Some(self.evict(i))
    }

    /// One probe, held open, through a view: the owned key is built only on a miss.
    pub fn entry_at<Q: Absorb<K>>(&mut self, q: Q) -> Entry<'_, K, V, Stored<K>> {
        self.reserve(1);
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        match self.site(a.place, |h| h.id == a.id && q.same(&h.k)) {
            Ok(at) => Entry::Occupied(OccupiedEntry { map: self, at }),
            Err(at) => Entry::Vacant(VacantEntry {
                map: self,
                at,
                a,
                key: q.absorb(),
            }),
        }
    }
}

// -- entry -----------------------------------------------------------------------------------------

/// What a table looks like right now: the report [`Map::stats`] returns.
///
/// Lyrium's own ARCHITECTURE.md asks every data-driven subsystem for a report that answers in
/// counts and reasons rather than prose, so that "absent", "present but degraded" and "present and
/// fine" are distinguishable without a screenshot. This is that, for a table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Which keyed face this table is on.
    pub face: Keystore,
    /// Live entries.
    pub live: usize,
    /// Slots holding a removed entry: they occupy probe order but hold nothing.
    pub tombstones: usize,
    /// Slots allocated.
    pub slots: usize,
    /// Whether this table's addresses are peppered. **A `false` here on a table fed from outside
    /// the process is the finding**, whatever the probe lengths currently say.
    pub peppered: bool,
    /// Groups the worst entry's probe touches. Low single digits on an honest key set at any size.
    pub longest_probe: u32,
    /// Groups an average entry's probe touches. 1.0 is perfect.
    pub mean_probe: f64,
    /// Full addresses an average hit compares. 1.0 is perfect; the birthday floor for a seven-bit
    /// tag at seven-eighths load is about 1.05. This is the statistic `mean_probe` cannot see: a
    /// tag that stops discriminating leaves every probe one group long and every scan sixteen
    /// compares wide, which is how the original low-tag defect stayed invisible.
    pub mean_compares: f64,
}

impl Stats {
    /// Live entries over slots.
    #[must_use]
    pub fn load(&self) -> f64 {
        if self.slots == 0 {
            0.0
        } else {
            self.live as f64 / self.slots as f64
        }
    }

    /// Whether the probe lengths look like an attack rather than a workload.
    ///
    /// Deliberately a wide bar: an honest key set measures mean 1.03, longest 8 and 1.05 compares
    /// at worst across every distribution the placement simulation covers, so anything that trips
    /// this is worth a log line, and nothing that does not trip it is worth waking anyone for.
    ///
    /// Three statistics, because an attacker chooses which one to spend their bits on. Keys aimed
    /// at the descent's windows pile into shared groups -- that is `mean_probe`, and with enough
    /// agreed bits, `longest_probe`. Keys that agree on the whole top of the address also share
    /// their **tag**, and sixteen full-width compares per group is a cost `mean_probe` never
    /// sees: the measured 20,000-key aimed build sits at 69 compares per hit against an honest
    /// 1.04, while its longest probe is seven.
    #[must_use]
    pub fn looks_aimed(&self) -> bool {
        self.longest_probe > 32 || self.mean_probe > 4.0 || self.mean_compares > 8.0
    }
}

/// A map that **cannot** be built without a pepper.
///
/// [`Map::new`] takes the process pepper, which is the floor. This is the ceiling: a table whose
/// keys come from outside the process says so at its declaration, and the compiler holds it to
/// that. The pepper is a [`NonZeroU64`], so there is no constructor that takes zero and no runtime
/// check to forget -- an unpeppered `Exposed` is not a value that exists.
///
/// Use it wherever an attacker picks the keys: a resource index built from an archive header, a
/// field table keyed by labels read out of a file, anything arriving on a socket.
///
/// ```
/// use apeiron::map::Exposed;
/// use std::num::NonZeroU64;
///
/// let pepper = NonZeroU64::new(0x5EED_1234_ABCD_9876).unwrap();
/// let mut index: Exposed<String, u32> = Exposed::new(pepper);
/// index.insert("nw_wswmg001".into(), 7);          // Deref gives the whole `Map` surface
/// assert_eq!(index.get_at("nw_wswmg001"), Some(&7));
/// ```
pub struct Exposed<K, V, H = Stored<K>>(Map<K, V, H>);

/// [`Exposed`] on the elliptic face, for keys that fit in a word.
pub type ExposedWords<K, V> = Exposed<K, V, Regen<K>>;

impl<K: Coord + Eq, V, H: Held<K>> Exposed<K, V, H> {
    /// An empty table with a pepper that cannot be zero.
    #[must_use]
    pub fn new(pepper: NonZeroU64) -> Self {
        Self(Map::with_pepper(pepper.get()))
    }

    /// The same, sized for `n` entries.
    #[must_use]
    pub fn with_capacity(pepper: NonZeroU64, n: usize) -> Self {
        let mut m = Map::with_pepper(pepper.get());
        m.reserve(n);
        Self(m)
    }

    /// Take the process pepper, if the process set one.
    ///
    /// `None` when it is still zero -- which is the point. A caller that reaches for this and gets
    /// `None` has found out at startup that nobody called
    /// [`set_process_pepper`](crate::pepper::set_process_pepper), rather than finding out under
    /// load.
    #[must_use]
    pub fn from_process_pepper() -> Option<Self> {
        NonZeroU64::new(crate::pepper::process_pepper()).map(Self::new)
    }

    /// The plain map underneath.
    pub fn into_inner(self) -> Map<K, V, H> {
        self.0
    }
}

impl<K: Coord + Eq, V, H: Held<K>> std::ops::Deref for Exposed<K, V, H> {
    type Target = Map<K, V, H>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K: Coord + Eq, V, H: Held<K>> std::ops::DerefMut for Exposed<K, V, H> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<K: Coord + Eq, V, H: Held<K>> std::fmt::Debug for Exposed<K, V, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exposed({})", self.0.summary())
    }
}

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
    a: Addr,
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
        let Self { map, at, a, key } = self;
        map.place(at, a, key, v);
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

impl<'a, K: Coord + Eq, V> Entry<'a, K, V, Stored<K>> {
    /// The key this entry is for, occupied or not.
    ///
    /// Offered on the parabolic face only. There `H::Ref` is `&K`, so an occupied slot and a vacant
    /// one hand back the same type; on the elliptic face an occupied entry *regenerates* its key as
    /// a value while a vacant one owns it, and there is no single type that is both.
    #[must_use]
    pub fn key(&self) -> &K {
        match self {
            Self::Occupied(e) => e.key(),
            Self::Vacant(e) => e.key(),
        }
    }
}

impl<K: Coord + Eq, V, H: Held<K>> Map<K, V, H> {
    /// One probe, held open, so a read-modify-write costs one probe and not two.
    #[inline]
    pub fn entry(&mut self, k: K) -> Entry<'_, K, V, H> {
        self.reserve(1);
        let a = H::address(self.salt, &k);
        match self.site(a.place, |h| h.holds(a.id, &k)) {
            Ok(at) => Entry::Occupied(OccupiedEntry { map: self, at }),
            Err(at) => Entry::Vacant(VacantEntry {
                map: self,
                at,
                a,
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
            id: self.id,
            k: self.k.clone(),
        }
    }
}

impl<K> Clone for Regen<K> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

/// Prints the entries, like `std`'s.
///
/// The `for<'a> H::Ref<'a>: Debug` bound is what lets one impl serve both faces: on the parabolic
/// one that is `&K: Debug`, on the elliptic one it is `K: Debug`. A map whose types do not satisfy
/// it still prints -- see [`Map::summary`] -- it just prints the shape rather than the contents.
impl<K: Coord + Eq, V: std::fmt::Debug, H: Held<K>> std::fmt::Debug for Map<K, V, H>
where
    for<'a> H::Ref<'a>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Coord + Eq, V, H: Held<K>> Map<K, V, H> {
    /// The shape, not the contents: face, occupancy, size.
    ///
    /// What `Debug` used to print. Kept as a method because it is the useful thing in a log line --
    /// a hundred thousand entries in a tracing field is not a diagnostic -- and because it works
    /// for every `V`, including the ones that are not `Debug`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
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
            // Salts are equal (checked above), so the placement `other` would descend on is the one
            // recovered from this slot's identity word.
            let p = h.placement(self.salt);
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

/// `map["resref"]`, panicking on an absent key, as `std` does.
///
/// Indexed by a [`Locate`] view rather than by `&K`, which is what makes `m["a"]` work on a
/// `Map<String, V>` -- the same thing `std` gets from `Borrow<Q>`. It matters more here than for
/// the other readers because an operator cannot be spelled a second way: `get` has `get_at` beside
/// it, and `[]` has nothing.
impl<K: Coord + Eq, V, Q: Locate<K>> std::ops::Index<Q> for Map<K, V, Stored<K>> {
    type Output = V;
    fn index(&self, q: Q) -> &V {
        self.get_at(q).expect("no entry found for key")
    }
}

/// The elliptic face indexes by the key itself: its address is a bijection of `Packed::packed`, not
/// of `Coord::coord_with`, so a `Locate` view would compute a different address and miss.
impl<K: Coord + Packed + Eq, V> std::ops::Index<&K> for Map<K, V, Regen<K>> {
    type Output = V;
    fn index(&self, k: &K) -> &V {
        self.get(k).expect("no entry found for key")
    }
}

impl<K: Coord + Eq, V, H: Held<K>, const N: usize> From<[(K, V); N]> for Map<K, V, H> {
    fn from(entries: [(K, V); N]) -> Self {
        let mut m = Self::with_capacity(N);
        for (k, v) in entries {
            m.insert(k, v);
        }
        m
    }
}

/// Copies out of a borrowed stream, the way `std` lets you extend from `&other`.
impl<'a, K: Coord + Eq + Copy + 'a, V: Copy + 'a, H: Held<K>> Extend<(&'a K, &'a V)>
    for Map<K, V, H>
{
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, it: T) {
        for (k, v) in it {
            self.insert(*k, *v);
        }
    }
}

impl<K: Coord + Eq, V, H: Held<K>> IntoIterator for Map<K, V, H> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, H>;

    /// Takes the region with it, rather than draining into a `Vec` first. A partially consumed
    /// `IntoIter` still drops what is left, which is why it owns the arrays instead of borrowing.
    fn into_iter(self) -> Self::IntoIter {
        // The map must not run its own `Drop` after the arrays are taken -- it would find them
        // empty and free nothing, but `ManuallyDrop` says so rather than relying on that.
        let mut m = std::mem::ManuallyDrop::new(self);
        IntoIter {
            tags: std::mem::replace(&mut m.tags, Box::new([])),
            slots: std::mem::replace(&mut m.slots, Box::new([])),
            salt: m.salt,
            i: 0,
            left: m.live,
            _k: PhantomData,
        }
    }
}

impl<'a, K: Coord + Eq, V, H: Held<K>> IntoIterator for &'a Map<K, V, H> {
    type Item = (H::Ref<'a>, &'a V);
    type IntoIter = Iter<'a, K, V, H>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Coord + Eq, V, H: Held<K>> IntoIterator for &'a mut Map<K, V, H> {
    type Item = (H::Ref<'a>, &'a mut V);
    type IntoIter = IterMut<'a, K, V, H>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
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
        // Every live slot's stored identity word is the one the map would compute for its key, and
        // its tag is the one the placement word carries.
        for (i, &t) in m.tags.iter().enumerate() {
            if t & LIVE == 0 {
                continue;
            }
            let (h, _) = unsafe { m.slots[i].assume_init_ref() };
            let a = Stored::<String>::address(m.salt, h.key(m.salt));
            assert_eq!(h.id(), a.id);
            assert_eq!(t, tag_of(a.place));
        }
    }

    /// A pepper of zero must reproduce the plain bijection exactly, or every table that never sets
    /// one silently changes layout the day this type is touched.
    #[test]
    fn no_pepper_is_the_plain_bijection() {
        let s = Salt::new(0);
        for x in [0u64, 1, u64::MAX, PHI, 0x0123_4567_89AB_CDEF] {
            assert_eq!(s.spread(x), crate::coord::spread(x));
            assert_eq!(s.unspread(s.spread(x).id), x);
        }
    }

    /// The identity bijection, run over the inputs Kani reaches as one formula but which are worth
    /// the wider spray anyway.
    ///
    /// The bijection is now a single odd multiply each way -- `spread(x).id == x * mul`,
    /// `unspread(id) == id * inv` -- so `proofs.rs` decides the whole composite as a concrete
    /// equation (`mul * inv == 1`) rather than leaving it to CBMC's 64-bit multiplier reasoning,
    /// which is what stalled on the old two-multiply-plus-fold word. This runs the composite both
    /// ways over the edges and a spray of random words, under the default multiplier and under
    /// pepper-derived ones. The elliptic face regenerates every key through `unspread`, so "almost
    /// a bijection" is a map that hands back wrong keys.
    #[test]
    fn the_address_bijection_round_trips() {
        let mut inputs = vec![0u64, 1, 2, u64::MAX, PHI, PHI_INV];
        for b in 0..64 {
            inputs.push(1u64 << b);
            inputs.push((1u64 << b).wrapping_sub(1));
            inputs.push(!(1u64 << b));
        }
        let mut r = 0x243F_6A88_85A3_08D3u64;
        for _ in 0..10_000 {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            inputs.push(r);
        }
        for pepper in [0u64, 1, 0xDEAD_BEEF, 20_260_829, u64::MAX] {
            let s = Salt::new(pepper);
            for &x in &inputs {
                assert_eq!(s.unspread(s.spread(x).id), x, "pepper {pepper:#x}, x {x:#x}");
                assert_eq!(s.spread(s.unspread(x)).id, x, "pepper {pepper:#x}, h {x:#x}");
            }
        }
        for &x in &inputs {
            assert_eq!(crate::coord::unspread(crate::coord::spread(x).id), x, "x {x:#x}");
            assert_eq!(crate::coord::spread(crate::coord::unspread(x)).id, x, "h {x:#x}");
        }
    }

    /// The multiplier a pepper produces must be odd, or the address map is not a bijection and the
    /// elliptic face regenerates the wrong key.
    #[test]
    fn every_pepper_gives_an_invertible_multiplier() {
        let mut r = 0x9E37_79B9u64;
        #[cfg(miri)]
        let rounds = 200;
        #[cfg(not(miri))]
        let rounds = 20_000;
        for _ in 0..rounds {
            r = r.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let s = Salt::new(r);
            assert_eq!(s.mul & 1, 1, "pepper {r:#x} gave an even multiplier");
            assert_eq!(s.mul.wrapping_mul(s.inv), 1, "pepper {r:#x}");
            for x in [0u64, 1, r, u64::MAX] {
                assert_eq!(s.unspread(s.spread(x).id), x, "pepper {r:#x}, x {x:#x}");
            }
        }
    }
}
