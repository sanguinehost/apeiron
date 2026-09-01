//! Keyed hash maps on the two keyed faces.
//!
//! [`Map<K, V>`] carries the key beside its address (parabolic; any `K: Coord + Eq`).
//! [`Words<K, V>`] stores only the address and regenerates the key from it (elliptic;
//! `K: Packed`). Both are open-addressed, 7/8 fill, sixteen-slot groups with a one-byte tag each.
//!
//! **Probe.** One widening multiply gives `Addr { id, place }`. `place` picks the first group and
//! the seven-bit tag; a group is one vector compare of sixteen tags; candidates are settled by
//! comparing `id`, then the key on the parabolic face. Misses walk successive `gbits`-wide windows
//! of `place` from bit 57 down (the window descent), so extending a collision chain by a step costs the
//! attacker another `gbits`.
//!
//! **Cost**, instructions per lookup, 694 keys, L1-resident, one `#[inline(never)]` loop per arm
//! with the build in its own frame (`examples/probe_cost.rs`, whose doc says why both of those
//! matter and what they were worth):
//!
//! | | `u64` | 13-byte resref | 42-byte path |
//! |---|---|---|---|
//! | apeiron `Words` | **31.8** | -- | -- |
//! | apeiron `Map` | **32.8** | 92.5 | 128.6 |
//! | hashbrown + foldhash | 33.1 | **79.6** | **127.4** |
//! | hashbrown + fxhash | 32.2 | 94.9 | 120.8 |
//! | std | 122.2 | -- | -- |
//!
//! The word column is ours on both faces. The `String` column is not, and the 42-byte shape --
//! the one a Lyrium `VPath` actually is -- is 1.2 instructions behind rather than 22.
//!
//! # Where the `String` gap is
//!
//! At 13 bytes it is 12.9 instructions. **8.0 of that is the fold**, measured on its own in
//! `examples/fold_instr.rs`:
//!
//! | corpus | bytes | apeiron | foldhash | |
//! |---|---|---|---|---|
//! | short | 2 | 43.1 | 34.1 | +9.0 |
//! | **resref** | **13** | **29.1** | **22.1** | **+7.0** |
//! | path | 42 | **69.1** | 73.1 | -4.0 |
//! | long | 231 | 202.2 | 179.6 | +22.6 |
//!
//! The two named steps, priced by stubbing each out and re-measuring:
//!
//! | | resref | path |
//! |---|---|---|
//! | `fold::settle`, the finalizer | **4.00** | 4.00 |
//! | `fold::mix2`'s zero guard | **1.00** | 16.00 |
//!
//! Both are bought deliberately. `settle` closes `SMHasher`'s `DiffDist` failure at bit 63, where
//! `2^63` is a zero divisor and no further multiply repairs it. The guard is what stops `a` or `b`
//! equal to zero annihilating the product -- the defect `hashtrinity` shipped -- and at pepper zero
//! the seed is a known constant, so that operand is reachable by choice rather than by luck.
//! `foldhash` needs neither because it is seeded from process randomness and has no deterministic
//! mode to defend. That is the trade, and it is a trade rather than a deficit.
//!
//! The guard went from 2.00 to 1.00 here, and from 18.00 to 16.00 on the path shape, by becoming
//! `a - b` instead of `a + rotl(b, 32)`. That was not a speed change looking for a justification:
//! the old guard swaps for every pair of words with the same half difference -- classes of size
//! `2^32` -- and the new one on exactly two points. See `fold::mix2` and the four proofs beside it.
//!
//! The remaining **4.9** is the table and the compare, and it is register pressure: the inlined
//! length ladder leaves the probe spilling its accumulator, its iterator and two `xmm` registers
//! per lookup. Ruled out by measurement: slot width (`LOOKUP_ARM=apstrzst` brings the stride to a
//! power of two and reads 92.9 against 92.5, which is the wrong sign as well as no size), the
//! out-of-line threshold (64 beats both 32 and 16 on both key shapes), and sinking the fold's
//! seeds into each length branch (bit-identical output, 92.5 either way -- LLVM was already
//! doing it).
//!
//! # What the remaining `u64` gap is, instruction by instruction
//!
//! `examples/two_gets.rs` puts `Map::get` and `hashbrown::HashMap::get` in one binary behind
//! `#[inline(never)] extern "C"`, so `objdump` shows each design's whole address-to-load path with
//! nothing else in it. Both compute one widening multiply. Lined up to the first tag load:
//!
//! | | hashbrown + foldhash | apeiron |
//! |---|---|---|
//! | guard | `cmpq $0,len` / `je` | `cmpq $16,len` / `jb` (hoisted out of the caller's loop on the borrowed surface -- see `probeable`) |
//! | mix | `xor` seed, `mulq`, `xor` hi into lo | `mulq`, `rol $32`, `xor` hi into lo |
//! | tag | `mov`, `shr $57` | `mov`, `shr $57` |
//! | group | `mov` mask, `and` | `mov` gshift, **`mov` to `cl`**, `shr %cl`, `mov` mask, `and` |
//! | total | 12 | 13 |
//!
//! So the split's *mixing* is free: our `rol` + `xor` is two instructions and foldhash's seed
//! `xor` + fold `xor` is two. Of the three that once differed, two are gone and neither was the
//! algebra:
//!
//! - **`or $0x80`**, our tag's live bit -- **removed** by flipping the empty sentinel from `0` to
//!   `0xFF`, so a live tag now has the top bit clear and `tag_of` puts nothing back. It took
//!   **two** instructions off the read path, not one, and the second is the interesting one: `or
//!   $0x80,%r9b` writes eight bits and leaves the rest of the register undefined, so LLVM had to
//!   `movzbl` before broadcasting the tag. `shr $57` alone already leaves a value below 128 in a
//!   full register, so the broadcast reads it directly. The write path separately loses the `not`
//!   behind `free = !occupied(line)`, since the vacancy scan is now the raw `movemask`. The
//!   sentinel was zero so a fresh region could be a
//!   lazy `calloc`; `examples/growcost.rs` measures that argument and refutes it -- 2^22 with
//!   growth, five rounds, 113.7 ms against **105.8 ms**. The fill is real and it is smaller than
//!   what the other polarity cost per insert.
//! - **`mov` to `cl`** -- a variable shift needs its amount in `cl` on the `x86_64` baseline.
//!   Under `-C target-cpu=native` it is `shrx` and the `mov` is gone. ISA, not design.
//! - **one extra `mov`** -- `place` stays live across the whole probe because every descent step
//!   reads fresh bits of it. hashbrown destroys its hash into the group index at the first group.
//!   That one is real, and it is the *only* one of the three the descent causes. See below.
//!
//! A fourth, `and $0x3f` in front of the variable shift, was `wrapping_shr` honouring the `% 64`
//! it promises; `Geom::first` now tells LLVM the range and it is gone. It was hoisted out of the
//! read loop already (`&self`, loop-invariant) but not the write loop, so it showed up as exactly
//! one instruction on each of `insert` (73.74 -> 72.74) and `entry` (75.73 -> 74.74) and zero on
//! `get`.
//!
//! # Why the last instruction is structural
//!
//! hashbrown's probe after the first group is `i + 16(k+1) mod G` -- triangular probing, which
//! consumes **no** further hash bits. Its whole probe order is a function of `h mod 2^gbits`
//! alone; everything above `gbits` feeds only the tag. Ours reads a fresh disjoint `gbits`-wide
//! window per step, so the probe order is a function of all 57 descent bits.
//!
//! That is the entire trade, stated exactly: **hashbrown's probe sequence uses `gbits` bits of the
//! address and ours uses 57.** Keys agreeing on `j` of our windows need `2^(j * gbits)` trials to
//! find; under triangular probing one solve for the first group makes the rest of the chain free.
//! The cost of holding 57 bits rather than `gbits` is that the word must stay in a register for
//! the length of the probe -- one `mov`.
//!
//! Measured by deleting the split (`place = lo`, no `rotl`, no `xor`) and re-running: `get` 36.65
//! -> 34.55 instr, `insert` 73.74 -> 70.74, `entry` 75.73 -> 71.74. That delta is the two mixing
//! instructions plus the register copies, and the table above says the mixing half of it is
//! matched instruction for instruction by anything else that mixes. What is left is the descent.
//!
//! The cheaper placements were measured and rejected on distribution, in `coord::spread`'s table:
//! `hi ^ lo` and `hi + lo` both fail `align4096`, and `(hi ^ lo) * PSI` needs a second multiply.
//!
//! **Rejected**, cycles per `u64` lookup, baseline 8.8:
//!
//! | change | cyc | |
//! |---|---|---|
//! | group mask pre-scaled to bytes | 8.4 | taken |
//! | first window off the first multiply | 8.0 | dead: `highbits` mean probe 1.95 at n=5,000 |
//! | tag broadcast as a multiply | 8.4 | no-op: LLVM emits the same five instructions |
//!
//! **Borrowed keys.** `get_at`/`insert_at` probe `Map<String, V>` with `&str` and
//! `Map<(String, u16), V>` with `(&str, u16)`, allocating nothing on a hit. Required, not
//! convenient: the parabolic face compares the key on every hit, so without it every lookup would
//! build one.

#![expect(
    clippy::inline_always,
    reason = "measured, not assumed. `get` and `find` were `#[inline]` and the compiler outlined \
              them anyway, so every lookup paid a call and the register traffic around it: 70 \
              instructions and 12.0 cycles against hashbrown's 32. `#[inline(always)]` on the \
              probe internals took it to 35.9 and 6.5. The numbers are in this module's doc"
)]
#![expect(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "`p` is the placement word, `g` the group, `j` the descent step, `lo`/`hi` the \
              halves of the widening product. These are the names the algebra and the papers use, \
              and expanding them to `placement_word` would make the probe harder to check against \
              the equations it implements, not easier"
)]


use crate::coord::{finish, inverse_odd, Addr, Coord, Packed, PHI, PHI_INV};
use crate::cell::Sigma;
use crate::table::{group_at, scan, tag_of, DESCENT_BITS, GROUP};
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::mem::MaybeUninit;

/// An unused slot. All ones, so its top bit is set and [`FREE`] finds it in one `movemask`.
///
/// It was zero, so that a fresh region could be a `calloc` the kernel hands over lazily rather
/// than a pass of writes. That is true of the allocation and false of the total: the polarity buys
/// an `or` off `tag_of` and a `not` off the write path's vacancy scan, and `examples/growcost.rs`
/// prices the whole build *including* the fill at 2^22 with growth, five rounds each -- 113.7 ms
/// zeroed against **105.8 ms** filled. See [`crate::table::tag_of`] for the polarity itself, and
/// [`crate::table::EMPTY`], which is the same constant a second time because `Region` allocates
/// its own tags.
pub(crate) const EMPTY: u8 = 0xFF;
/// A slot whose entry was removed. Distinct from [`EMPTY`] because a probe that stopped here would
/// lose every key placed beyond it while it was live, and distinct from a live tag because both
/// sentinels set the top bit and `tag_of` never does.
pub(crate) const TOMB: u8 = 0xFE;
/// `EMPTY` and `TOMB` have this bit; a live slot's tag does not. One `movemask` separates them.
pub(crate) const FREE: u8 = 0x80;

/// Fill limit, as a fraction. Seven eighths, where `hashbrown` sits, so a memory comparison
/// against it compares layouts rather than load factors.
/// `site`'s "no vacancy seen yet". Not a valid slot index on any target.
const NONE: usize = usize::MAX;

const FILL_NUM: usize = 7;
const FILL_DEN: usize = 8;

/// The smallest non-empty table: one group, one tag line, one vector compare.
pub(crate) const MIN_SLOTS: usize = GROUP;

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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the truncations ARE the split, as in `coord::spread`: `lo` is the low half of \
                  the 128-bit product and `hi` the high half"
    )]
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
/// [`Step`], but naming the slot by reference rather than by index.
///
/// The two exist for one measured reason. `find` returns an index, and a reader then has to turn
/// that index back into an address -- `shl $0x4` and an add -- even though the scan computed
/// exactly that address a moment earlier to read the slot's key half. LLVM keeps the index live
/// and redoes the arithmetic instead of reusing it, so `get` costs four instructions more than
/// `contains_key` while `hashbrown`'s two are identical. Handing back the reference the scan
/// already formed removes the recomputation.
///
/// The index form stays for insertion and removal, which need a *position* rather than a value.
enum Probe<'a, H, V> {
    /// The key is here.
    Hit(&'a (H, V)),
    /// This group holds an empty slot, so the key is nowhere.
    Absent,
    /// Full of live entries and tombstones; the probe continues.
    Onward,
}

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
        // `trailing_zeros()` is **64** for an empty region, and 64 is not a shift amount. Zero is
        // the value the rest of this file already means by "one group, nothing to choose":
        // `group_of` returns 0 for it, `Descent::next` takes the fallback either way, and
        // `p.wrapping_shl(64)` and `p.wrapping_shl(0)` are both `p`. So the empty region folds onto
        // the sentinel that already exists and nothing downstream can tell the two apart -- while
        // `gbits < 64` becomes a fact the shift sites can hand to LLVM.
        //
        // What that fact is worth: `wrapping_shl` has to honour the `% 64` it promises, so
        // `Descent::after_first` emitted `and $0x3f` on `gbits` and LLVM hoisted the result into a
        // **stack slot held for the whole probe loop** -- cold descent state occupying a hot-path
        // register. `Geom::first` had already been given the same fact about `gshift`;
        // `after_first` had not. `a_geometry_never_asks_for_a_shift_of_sixty_four` pins it.
        let gbits = if groups == 0 { 0 } else { groups.trailing_zeros() };
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
    ///
    /// # The floor on this path, and why a constant shift does not reach it
    ///
    /// The chain from the key to the tag load is `mul(4) -> rol(1) -> xor(1) -> shr(1) -> and(1)`,
    /// eight cycles, against `hashbrown`'s `xor(1) -> mul(4) -> xor(1) -> and(1)`, seven. Measured
    /// isolated, 8.9 cycles against 8.2. **The one cycle is the `rol`**, and the `rol` is what
    /// makes `place` non-invertible. There is nothing else left on the path.
    ///
    /// Anchoring the window's *bottom* at a constant instead of its top was built and measured, and
    /// it is the only thing that moved the rest. `gshift` leaves the hot loop entirely, so the
    /// address seed stops being spilled -- `mov -0x10(%rsp),%rax` before the multiply becomes
    /// `mov %r8,%rax` -- and the index becomes `shr $0x21` plus `and %ebx` where it was a `mov` to
    /// `cl`, a three-uop `shr %cl` and an `and` from the stack. It wins the instruction count:
    /// `get` 33.49 to **32.55** against `hashbrown`'s 33.12, `get miss` 31.91 to 31.26 with cycles
    /// at parity, `insert reserved` 57.4 to 50.4, `extend` 57.4 to 51.4. Every test passes,
    /// including all four of `the_probe_stays_flat`, because `[37, 57)` is inside the region the
    /// address proofs establish is well mixed.
    ///
    /// **It is still not landed, and `growth_splits_each_group_in_two` is why.** With the window's
    /// top anchored, doubling appends a bit at the *bottom* of the index and a group splits into
    /// `2g` and `2g + 1`, which are adjacent, so `rehome` walking the old region in ascending order
    /// writes the new one in ascending order. Anchoring the bottom moves the new bit to the *top*:
    /// `g` splits into `g` and `g + 2^gbits`, half a table apart, which is `hashbrown`'s locality
    /// and the thing that proof was written to say this crate does not have. The cost shows up
    /// where it should, on the growth-heavy row: `insert+drop` at a million keys goes from 84.7
    /// cycles to **92.7**.
    ///
    /// So it trades a cycle loss on `get hit` that it does not fix for a cycle loss on
    /// `insert+drop` that it causes. The Kani harness caught it; the flatness gate did not, and
    /// could not, because the partition is fine and only its *order* changed.
    ///
    /// # Why it is not one mask, like `hashbrown`'s
    ///
    /// `hashbrown` reaches its first group with a single `and` -- `hash & bucket_mask` -- while
    /// this pays a *variable* shift whose amount the compiler reloads from the `Geom` on every
    /// lookup (`mov 0x30(%rsp),%rcx`, `shr %cl,%rax`). Taking the window from the bottom of the
    /// placement word instead would be one `and` and was tried.
    ///
    /// It **fails the flatness gate**: `pairlike` keys at n = 5,000 gave a mean probe of 1.672
    /// against the 1.5 bound. The reason is the first thing this crate's own `coord` doc says --
    /// a multiply carries entropy upward and never downward, so a product's low bits are its
    /// least mixed. `place` is `rotl(hi, 32) ^ lo`, and its low half is `hi`'s *top* half xored
    /// with `lo`'s bottom half; for a packed key small enough that `x * mul` does not fill 128
    /// bits, `hi`'s top half is zero, and the low window collapses to the raw low bits of a
    /// product. The high window is chosen, not arbitrary.
    ///
    /// The identity `first == group_of(p, 0)` is what makes this checkable: get it wrong and
    /// `rebuild` places entries where `find` will not look. `the_carried_descent_is_the_computed_one`
    /// fails immediately, which is how the first attempt at this was caught.
    #[inline(always)]
    fn first(self, p: u64) -> usize {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "`gshift` is chosen so the result is already below `gmask`, which is at most `slots * \
                          GROUP` and therefore a `usize`. The mask makes the truncation unobservable."
        )]
        {
            // SAFETY: `gshift` is `(DESCENT_BITS - 4).saturating_sub(gbits)`, so at most 53. LLVM
            // cannot see that across the struct load, so `wrapping_shr` emits an `and $0x3f` to
            // implement the `% 64` it promises. Telling it the range deletes that instruction.
            // `Geom::of` is the only constructor and `the_shift_is_in_range` pins the bound.
            unsafe { std::hint::assert_unchecked(self.gshift < 64) };
            (p >> self.gshift) as usize & self.gmask
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
    ///
    /// # Four ways to stop `gbits` holding a hot register, and what each costs
    ///
    /// `gbits` is read here and nowhere on the first-group path, and the descent is entered on
    /// fewer than one lookup in a thousand (`Map::stats`, 1.0005 mean probe at a million keys).
    /// It is still loop-invariant across a caller's lookup loop, so LLVM hoists the load into that
    /// loop's preheader and holds it live across every iteration. The register that costs is one
    /// the hot path wanted: `objdump` on `examples/op_asm.rs` shows `gshift`, which every lookup
    /// needs, spilled to `-0x18(%rsp)` while `gbits` sits in `%r10d`.
    ///
    /// Instructions per operation, which are exact where cycles here carry about 7% run-to-run
    /// spread (`hashbrown`'s arm, unchanged code, read 9.81, 9.55 and 9.10 cycles on three runs):
    ///
    /// | attempt | `get` | `get` under `filter_map` | `get miss` | `remove` |
    /// |---|---|---|---|---|
    /// | as written | **33.49** | 36.47 | **31.91** | 73.28 |
    /// | derive `gbits` from `gmask` (`shr`, `inc`, `tzcnt`) | 33.49 | 37.47 | | |
    /// | derive `gbits` from `gshift` (one `sub`) | 33.49 | 36.47 | 31.91 | |
    /// | outline this constructor | 34.60 | **32.56** | 33.99 | 73.35 |
    /// | outline the whole descent loop | 37.60 | 37.61 | | **73.28** |
    ///
    /// Both derivations are hoisted the same way the load was: LLVM lifts `mov $0x35,%r10d` and
    /// `sub %r8d,%r10d` into the preheader and keeps the result live, so the live count is
    /// unchanged and the arithmetic is added. Outlining is the only thing that works, because a
    /// call has nothing to hoist, and it costs about one instruction per lookup on `get` and two
    /// on `get miss` for the few percent of probes that do enter the descent. It buys four on the
    /// adapter shape and it is not worth them.
    ///
    /// `find_onward` is where outlining did pay, and the reason it pays there and not here is on
    /// that function.
    #[inline]
    fn after_first(p: u64, g: Geom) -> Self {
        // SAFETY: `Geom::of` is the only constructor and it never stores 64 -- see the comment
        // there. Without this, `wrapping_shl`'s `% 64` becomes an `and $0x3f` that LLVM hoists into
        // a stack slot live across the whole probe.
        unsafe { std::hint::assert_unchecked(g.gbits < 64) };
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
        // SAFETY: as `Geom::first`.
        unsafe { std::hint::assert_unchecked(self.g.gshift < 64) };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "As `Geom::first`: masked to `gmask` immediately, so the discarded bits cannot reach \
                          the group index."
        )]
        let g = (self.window >> self.g.gshift) as usize & self.g.gmask;
        self.window <<= self.g.gbits;
        g
    }
}

// -- what a slot remembers of its key -----------------------------------------------------------

/// The face constants are **checked against what the impls do**, at compile time.
///
/// `Held::FACE` was an annotation: two hand-written constants that no code consulted. Writing
/// `Sigma::Hyperbolic` on [`Regen`] -- a flat lie, the hyperbolic face discards the key and
/// `Regen` regenerates it exactly -- would have compiled and passed every test. A label nothing
/// can falsify is the thing this crate exists not to ship.
///
/// [`Sigma::stores_the_key`] is the claim that the key must sit beside the address, so it is the
/// claim that the slot is wider than the address alone. That is decidable here, so it is decided
/// here: change either constant and the crate stops building.
///
/// The other half -- that a face claiming [`Sigma::key_survives`] can actually reproduce the key
/// from what it stored -- is a statement about values, not sizes, and lives in
/// `tests/the_faces_keep_their_promises.rs`.
const _: () = {
    assert!(<Stored<u64> as Held<u64>>::FACE.stores_the_key());
    assert!(size_of::<Stored<u64>>() > size_of::<u64>(), "a carried key needs room for one");

    assert!(!<Regen<u64> as Held<u64>>::FACE.stores_the_key());
    assert!(size_of::<Regen<u64>>() == size_of::<u64>(), "a regenerated key stores nothing");

    // Neither keyed face may claim the discard: both reproduce the key, one from the address and
    // one from the slot. The hyperbolic face has no `Held` impl at all.
    assert!(<Stored<u64> as Held<u64>>::FACE.key_survives());
    assert!(<Regen<u64> as Held<u64>>::FACE.key_survives());
};

/// How a live slot holds its key's identity.
///
/// The two implementations are the two keyed faces of [`Sigma`], and they differ in exactly the
/// way the discriminant predicts: [`Regen`] keeps nothing, because the address determines the key;
/// [`Stored`] keeps the key, because it does not.
/// **There are two impls and there can only be two.**
///
/// [`Held::into_key`] hands back the key, so a face that *discards* it cannot implement this
/// trait -- the hyperbolic one has nowhere to get the return value from. `Held` is therefore the
/// trait of the key-surviving faces, and [`Sigma::key_survives`] is false exactly where an impl is
/// impossible. The `const` block above asserts that correspondence rather than trusting it.
///
/// So the third face is absent from this module by construction, not by omission. Where it lives
/// is [`crate::table`], which keys on the address itself.
pub trait Held<K: Coord + Eq>: Sized {
    /// The face this is the storage of.
    const FACE: Sigma;

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

/// **The parabolic face.** The key is carried beside its address.
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
    const FACE: Sigma = Sigma::Parabolic;
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
        // The key compare is the decision; the identity word is one early reject in front of it,
        // worth its two instructions only when the compare it skips is a `memcmp`. See
        // `Coord::CHEAP_EQ`, which carries the measurement both ways.
        if K::CHEAP_EQ {
            self.k == *k
        } else {
            self.id == id && self.k == *k
        }
    }
    #[inline]
    fn same_slot(&self, other: &Self) -> bool {
        self.id == other.id && self.k == other.k
    }
    #[inline]
    fn placement(&self, salt: Salt) -> u64 {
        if K::SPREAD {
            // The two words are the same on this face; there is nothing to recover.
            self.id
        } else if K::CHEAP_EQ {
            // **This face has the key, so it can re-address instead of running the bijection
            // backwards and forwards.** `placement_of` is `spread(unspread(id))` -- two multiplies
            // -- where `Self::address` is one widening multiply and a fold. It is the right trade
            // exactly when the key is cheap to turn back into a coordinate, which is what
            // `CHEAP_EQ` already names: a word key re-addresses in one multiply, a byte key would
            // have to re-fold and `placement_of` wins. `Regen` cannot take this branch at all --
            // it stores no key, and that is the eight bytes the elliptic face exists to save.
            Self::address(salt, &self.k).place
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
    const FACE: Sigma = Sigma::Elliptic;
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
    #[inline(always)]
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
/// Every `coord_at` here is `#[inline(always)]`, not `#[inline]`.
///
/// They are one-line forwards to `Coord::coord_with`, so an outlined one is pure loss -- a call, a
/// frame, and the loop-invariant part of the fold's seed recomputed per key because it can no
/// longer be hoisted. LLVM outlined the `&str` one anyway: it was **23% of the `String` lookup
/// profile** as its own symbol, and the plain `#[inline]` on the *callee* did not reach it, because
/// the refusal is at this boundary and not at the fold's. Making this one attribute
/// `#[inline(always)]` took a 13-byte resref lookup from 114.75 instructions to 107.73. The rest
/// are the same shape and carry it for the same reason, so the next key type does not have to
/// rediscover it.
macro_rules! locate_by_value {
    ($($t:ty),*) => {$(
        impl Locate<$t> for $t {
            #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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

// A `Locate<[u8; N]> for &str` view was written, tested and **removed**. It let a call site say
// `map.get_at("nw_wswmg001")` against a `Map<[u8; 16], V>` without building the array by hand, and
// it was correct -- five tests, including the one that matters, that a view longer than `N` pads to
// the same array as its own first `N` bytes and so must be a miss rather than a truncation.
//
// It was also **seven times slower than the array it stands for**. The stored key is zero-padded to
// `N`, so the view has to materialise that array before it can fold it, and the array is memory:
// `copy_from_slice` with a runtime length is a call to `memcpy` (103.64 instructions and 87.01
// cycles a lookup, against 50.80 and 10.84 probing the same table with the array directly), and
// rewriting it as the fold's own overlapping fixed-size reads only moved LLVM to a byte-at-a-time
// copy into a stack slot that the fold then read back as two `u64`s -- 75.64 instructions and
// **79.28 cycles**, a store-to-load forwarding stall on every lookup.
//
// The fast version exists: fold the `&str`'s bytes directly into the two words the padded array
// would have produced, with a masked overlapping read and no array at all. It is a second copy of
// `fold::absorb`'s length ladder that has to stay bit-identical to it forever -- exactly the
// "second scheme that happens to agree" that the tuple view below is written to avoid.
//
// And the performance would depend on something the caller cannot see: with a literal the length is
// a constant and the pad is a register move, with a `String` it is not. A view that is fast or 7x
// slow depending on whether the length is known at compile time is worse than no view. Hold the
// `[u8; N]` key.

impl<A: Coord + Eq, B: Coord + Eq, PA: Locate<A>, PB: Locate<B>> Locate<(A, B)> for (PA, PB) {
    #[inline(always)]
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
    /// One byte per slot: [`EMPTY`], [`TOMB`], or a live slot's seven-bit tag, which has [`FREE`] clear.
    /// Sixteen of these are one vector compare, which is what makes a group scan one instruction
    /// rather than sixteen loads.
    tags: Box<[u8]>,
    /// The entries. **`slots[i]` is initialised exactly when `tags[i] & FREE == 0`.** That is the
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
    /// Occupied slots the region can still take before it must grow: `limit(slots) - used`, where
    /// `used` is live entries plus tombstones (a tombstone occupies probe order even though it
    /// holds nothing).
    ///
    /// Stored this way round because of what `reserve` costs otherwise. `used + additional <=
    /// limit(slots.len())` has to **recompute the limit on every insert** -- load `slots`, load
    /// `used`, `shr $3`, `and $-8`, `sub`, `add`, `cmp`, `ja`: nine instructions to decide not to
    /// grow, against `hashbrown`'s `cmp growth_left, 0` and a branch. Carrying the difference makes
    /// it `cmp additional, room`, and the subtraction that maintains it happens only on the vacant
    /// path, where a tag and a slot are being written anyway.
    ///
    /// Nothing is lost: `used` is only ever read to derive [`Self::tombstones`] and to make that
    /// comparison, and `limit(slots) - room - live` recovers it exactly, on a path that is cold.
    room: usize,
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
            room: 0,
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
    pub fn face(&self) -> Sigma {
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
    #[expect(
        clippy::cast_precision_loss,
        reason = "a mean over probe counts, for a diagnostic report. The numerator would have to \
                  exceed 2^53 for the conversion to lose anything, which is more entries than the \
                  table can hold"
    )]
    pub fn stats(&self) -> Stats {
        let mut longest = 0u32;
        let mut total = 0u64;
        let mut compared = 0u64;
        for (i, &t) in self.tags.iter().enumerate() {
            if t & FREE != 0 {
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
            tombstones: self.tombstones(),
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
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a table with more than 2^32 groups is 2^36 slots and 2^41 bytes; the \
                      allocation fails long before the count does"
        )]
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
        limit(self.slots.len()) - self.room - self.live
    }

    // -- probing -------------------------------------------------------------------------------

    /// The table has slots to probe. **Checked before the address is computed, not after.**
    ///
    /// `find` guards this too, and has to -- it is reachable from `site` and from the rebuild. But
    /// a guard *after* the address lands in the middle of the loop body, where LLVM cannot unswitch
    /// it out, and it is then a compare and a branch on every lookup. `hashbrown` puts its
    /// `items == 0` check at the top of `contains_key` and gets it hoisted out of the caller's loop
    /// for free; this is the same check in the same place. Measured on 13-byte resrefs: 94.70
    /// instructions and 20.86 cycles a lookup with the guard after the fold, **92.71 and 19.87**
    /// with it before.
    ///
    /// # Only the borrowed surface gets it
    ///
    /// The same guard on `get`/`contains_key`/`remove_entry`, where the address is `H::address` --
    /// one multiply -- is a **loss**, and a large one. Measured, instructions per lookup:
    ///
    /// | | without | with |
    /// |---|---|---|
    /// | `Words<u64, u64>::contains_key` | **31.82** | 33.79 |
    /// | `Map<u64, u64>::contains_key` | **34.82** | 37.82 |
    /// | `Map<String, u64>::contains_key` | **94.76** | 115.71 |
    ///
    /// A branch is only worth hoisting past work worth skipping. In front of a fold it buys the two
    /// instructions above; in front of a multiply it is two instructions that buy nothing and cost
    /// the scheduler a block boundary it was using.
    #[inline(always)]
    fn probeable(&self) -> bool {
        self.slots.len() >= MIN_SLOTS
    }

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
            Step::Found(i) => Some(i),
            Step::Absent => None,
            Step::Onward => self.find_onward(p, tag, &is),
        }
    }

    /// The probe past the first group, out of line so its state cannot reach the caller's loop.
    ///
    /// Peeling the first group out of the loop was already here, for the reason above it. What
    /// peeling does not stop is **loop-invariant code motion in the caller**: `Descent::after_first`
    /// reads `gbits`, and `Descent::next` reads `mask()`, and both are invariant across a lookup
    /// loop, so LLVM hoists the loads into that loop's preheader and holds the values live across
    /// every iteration. Neither is on the first-group path. Behind a `#[cold] #[inline(never)]`
    /// call there is nothing for the caller to hoist.
    ///
    /// # It pays here and on neither of the other two probes, and the reason is the traffic
    ///
    /// The three probes were each tried both ways. Measured on `examples/op_asm.rs` at 20,000 keys
    /// and `examples/whole_surface.rs` across six sizes:
    ///
    /// | probe | callers | outlined | inline |
    /// |---|---|---|---|
    /// | `find` | `remove`, `get_mut` | **73.3 instr, 13.9 cyc** | 77.2, 15.4 |
    /// | `find_ref` | `get` | 37.6, 9.7 | **33.5, 8.7** |
    /// | `site` | `insert`, `entry` | `entry` at 1M **0.58x**, 79.0 instr | **0.84x, 68.0** |
    ///
    /// `#[cold]` is a claim about traffic and it is true for exactly one of the three. A lookup at
    /// a 7/8 fill finds its key in the first group almost always, so `find`'s descent is cold and
    /// moving it out is free. An *insert* is hunting a vacancy rather than a key, and a table at
    /// 7/8 leaves the first group often, so the same annotation on `site` is a false statement
    /// about where the time goes and costs eleven instructions for it. `find_ref` is the
    /// interesting one: its descent is as cold as `find`'s, and outlining still loses, because
    /// `get` is the pure-read path with registers to spare and it pays the call setup without
    /// buying relief it needed. `probe_cost.rs` recorded "+5.4 instructions, cycles unmoved" for
    /// this once already; that measurement was right and this is what it was measuring.
    #[cold]
    #[inline(never)]
    fn find_onward(&self, p: u64, tag: u8, is: &impl Fn(&H) -> bool) -> Option<usize> {
        let mut d = Descent::after_first(p, self.geom);
        loop {
            match self.scan_group(d.next(), p, tag, is) {
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
            // SAFETY: a matched tag has `FREE` clear (`tag_of` never sets it; `EMPTY` and `TOMB` do),
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

    /// [`Self::scan_group`], returning the slot itself. See [`Probe`] for why both exist.
    #[inline(always)]
    fn scan_group_ref(&self, base: usize, tag: u8, is: &impl Fn(&H) -> bool) -> Probe<'_, H, V> {
        let line = self.line(base);
        let mut hits = scan::match_tag(line, tag);
        while hits != 0 {
            let lane = base + hits.trailing_zeros() as usize;
            // SAFETY: as `scan_group` -- a matched tag has `FREE` clear, so the slot is initialised,
            // and `lane` is inside the line, which is inside the region.
            let slot = unsafe { self.slots.get_unchecked(lane).assume_init_ref() };
            if is(&slot.0) {
                return Probe::Hit(slot);
            }
            hits &= hits - 1;
        }
        if scan::match_tag(line, EMPTY) != 0 {
            Probe::Absent
        } else {
            Probe::Onward
        }
    }

    /// [`Self::find`], returning the slot itself. See [`Probe`].
    #[inline(always)]
    fn find_ref(&self, p: u64, is: impl Fn(&H) -> bool) -> Option<&(H, V)> {
        if self.slots.len() < MIN_SLOTS {
            return None;
        }
        let tag = tag_of(p);
        let base = self.geom.first(p);
        match self.scan_group_ref(base, tag, &is) {
            Probe::Hit(s) => return Some(s),
            Probe::Absent => return None,
            Probe::Onward => {}
        }
        let mut d = Descent::after_first(p, self.geom);
        loop {
            match self.scan_group_ref(d.next(), tag, &is) {
                Probe::Hit(s) => return Some(s),
                Probe::Absent => return None,
                Probe::Onward => {}
            }
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
        // `usize::MAX` as "no vacancy seen", not `Option<usize>`: the option is sixteen bytes, so
        // it lived on the stack and cost a store, a test and a reload per group of every insert.
        // No slot index can reach it -- the region's byte size has to fit in an allocation.
        let mut vacant = NONE;
        let tag = tag_of(p);
        let mut base = self.geom.first(p);

        // The first group is peeled, as it is in `find`, and for the same reason: at a 7/8 fill
        // almost every insert finds its empty slot here and never touches the descent. Left inside
        // the loop, `Descent::after_first` is built -- five fields written -- before the first tag
        // line is even loaded, on every insert, to serve the case that almost never happens.
        match self.scan_site(base, tag, &is, &mut vacant) {
            Ok(found) => return Ok(found),
            Err(true) => return Err(if vacant == NONE { base } else { vacant }),
            Err(false) => {}
        }

        // **The descent is not where the time goes, and `Map::stats` says so.** Mean probe depth
        // on a `Words<u64, u64>` built from a random corpus: 1.0008 at 16,384 entries, 1.0283 at
        // 100,000, **1.0005 at 1,000,000**, longest 3. Fewer than one lookup in a thousand leaves
        // the first group at a million keys. Any hypothesis that prices this crate's remaining
        // gaps on the probe order has to get past that number first, and the next paragraph is one
        // that did not.
        //
        // **Prefetching the next group here does not pay, and it was the obvious thing to try.**
        // `Descent::next` reads bits of the placement word and nothing else, so the address of the
        // group after this one is known before this one is looked at: the load can be started
        // against the scan rather than after it, which is the one thing a fresh window per step
        // can buy back against a stride that stays in the page. Measured as the minimum of two
        // full sweeps either side, `_mm_prefetch` on the group ahead is a **net loss**: `entry`
        // -5.7% at 100k and -6.7% at 1M, `extend` -3% to -4%, `remove all` -4% to -9%, and three
        // instructions added to every insert whether the descent is entered or not. Running the
        // descent a step ahead to have an address to prefetch is what costs them, and the win it
        // was meant to buy did not appear at any size.
        //
        // Deliberately **not** outlined, where `find_onward` is. On a lookup the descent is cold:
        // at a 7/8 fill an occupied group holds the key it was probed for, so almost nothing
        // reaches it. On an insert the descent is hunting a *vacancy*, and a table at 7/8 leaves
        // the first group often, so `#[cold]` would be a false claim about the traffic. Measured,
        // it is: outlining this took `entry` at a million keys from 0.84x to **0.58x** and its
        // instructions from 68.0 to 79.0, and `insert reserved` from 57.4 to 66.4.
        let mut d = Descent::after_first(p, self.geom);
        loop {
            base = d.next();
            match self.scan_site(base, tag, &is, &mut vacant) {
                Ok(found) => return Ok(found),
                Err(true) => return Err(if vacant == NONE { base } else { vacant }),
                Err(false) => {}
            }
        }
    }

    /// One group of [`Self::site`]: `Ok(slot)` on a match, `Err(true)` when this group ends the
    /// search, `Err(false)` to keep descending. Records the earliest free slot in `vacant`.
    #[inline(always)]
    fn scan_site(
        &self,
        base: usize,
        tag: u8,
        is: &impl Fn(&H) -> bool,
        vacant: &mut usize,
    ) -> Result<usize, bool> {
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
        if *vacant == NONE {
            // Empty or tombstoned: both are free, and the earliest in probe order wins.
            let free = scan::free(line);
            if free != 0 {
                *vacant = base + free.trailing_zeros() as usize;
            }
        }
        // An empty slot ends the search, and it is itself free, so `vacant` is set.
        Err(scan::match_tag(line, EMPTY) != 0)
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

    /// Look every key up, visiting `(key, value)` in order.
    ///
    /// A batch reader, for the shape `hashtrinity` spelled `get_each`: it says "these lookups are
    /// independent" in one call rather than leaving the caller to write the loop.
    ///
    /// # It is a plain loop, and that is a measurement
    ///
    /// This was written as a software pipeline -- prefetch the group of the key sixteen positions
    /// ahead, so the misses of independent lookups overlap instead of queueing. The reasoning is
    /// sound and the implementation was **slower at every size tested**, from 1,024 entries to
    /// 8.4 million (640 MiB, far past any cache):
    ///
    /// | entries | plain loop | prefetching |
    /// |---|---|---|
    /// | 65,536 | **2.78 ns** | 4.37 |
    /// | 1,048,576 | **9.49** | 14.49 |
    /// | 8,388,608 | **16.17** | 20.19 |
    ///
    /// The first suspicion was that the pipeline folded each key twice, once as the lookahead and
    /// once as the probe. Hoisting every address out of the loop first, so the prefetch costs only
    /// the instruction, still lost: 18.53 ns against 16.17 at 8.4 million.
    ///
    /// The reason is that there was never a queue to unblock. Nothing in the loop depends on the
    /// previous iteration, so an out-of-order core already has several probes in flight and the
    /// misses already overlap; an explicit prefetch adds instructions to a loop that was not
    /// stalled on the thing it prefetches. `examples/batch_reads.rs` is the harness, kept so this
    /// is not re-derived.
    ///
    /// ```
    /// use apeiron::map::Map;
    ///
    /// let m: Map<u64, u64> = (0u64..1000).map(|i| (i, i * 2)).collect();
    /// let keys: Vec<u64> = (0..1000).collect();
    /// let mut sum = 0;
    /// m.get_each(&keys, |_, v| sum += v.copied().unwrap_or(0));
    /// assert_eq!(sum, (0..1000u64).map(|i| i * 2).sum::<u64>());
    /// ```
    pub fn get_each<'a>(&'a self, keys: &'a [K], mut visit: impl FnMut(&'a K, Option<&'a V>)) {
        for k in keys {
            visit(k, self.get(k));
        }
    }

    /// Every key's value, in order. [`Map::get_each`] collected.
    ///
    /// ```
    /// use apeiron::map::Map;
    ///
    /// let m: Map<u64, u64> = [(1u64, 10), (3, 30)].into_iter().collect();
    /// assert_eq!(m.get_all(&[1, 2, 3]), vec![Some(&10), None, Some(&30)]);
    /// ```
    #[must_use]
    pub fn get_all<'a>(&'a self, keys: &'a [K]) -> Vec<Option<&'a V>> {
        let mut out = Vec::with_capacity(keys.len());
        self.get_each(keys, |_, v| out.push(v));
        out
    }

    /// Look every key up and fold the results, without materialising them.
    ///
    /// `get_all` when you want the vector, this when you want a number: the pipeline is the same
    /// and the allocation is not.
    ///
    /// ```
    /// use apeiron::map::Map;
    ///
    /// let m: Map<u64, i64> = [(1u64, 10), (3, 30)].into_iter().collect();
    /// let total = m.get_reduce(&[1, 2, 3], 0i64, |v| *v, |a, b| a + b);
    /// assert_eq!(total, 40);
    /// ```
    pub fn get_reduce<'a, M>(
        &'a self,
        keys: &'a [K],
        id: M,
        map: impl Fn(&'a V) -> M,
        combine: impl Fn(M, M) -> M,
    ) -> M {
        // `id` is held back rather than seeded into the accumulator, so the fold needs no
        // `Option::take` dance and has no unreachable branch to document. For a monoid the two are
        // the same value: `id` combined with the first residual *is* the first residual.
        let mut acc: Option<M> = None;
        self.get_each(keys, |_, v| {
            if let Some(v) = v {
                acc = Some(match acc.take() {
                    Some(a) => combine(a, map(v)),
                    None => map(v),
                });
            }
        });
        acc.unwrap_or(id)
    }

    /// Fold every live value in the table. The whole surface, not a set of keys.
    ///
    /// Order is the table's, which is the address order and therefore not the insertion order --
    /// so `combine` should be associative and commutative if the result is to mean anything
    /// stable. That is the same condition [`crate::merge::Species::Free`] names.
    ///
    /// ```
    /// use apeiron::map::Map;
    ///
    /// let m: Map<u64, i64> = (0u64..10).map(|i| (i, i as i64)).collect();
    /// assert_eq!(m.encircle(0i64, |v| *v, |a, b| a + b), 45);
    /// ```
    pub fn encircle<M>(
        &self,
        id: M,
        residual: impl Fn(&V) -> M,
        combine: impl Fn(M, M) -> M,
    ) -> M {
        let mut acc = id;
        for v in self.values() {
            acc = combine(acc, residual(v));
        }
        acc
    }

    #[must_use]
    /// The value at `k`.
    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        let a = H::address(self.salt, k);
        self.find_ref(a.place, |h| h.holds(a.id, k)).map(|s| &s.1)
    }

    /// The value at `k`, mutably.
    #[inline]
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let a = H::address(self.salt, k);
        let i = self.find(a.place, |h| h.holds(a.id, k))?;
        // SAFETY: as `get` -- `find`'s index is in bounds by construction.
        Some(unsafe { &mut self.slots.get_unchecked_mut(i).assume_init_mut().1 })
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
        // SAFETY: as `get` -- `find`'s index is in bounds by construction.
        let (h, v) = unsafe { self.slots.get_unchecked(i).assume_init_ref() };
        Some((h.key(self.salt), v))
    }

    // -- writing -------------------------------------------------------------------------------

    /// Insert, returning what was displaced.
    ///
    /// Takes the key by value and clones nothing: the probe borrows it, and the borrow is over
    /// before either arm needs to own it.
    ///
    /// # `inline(always)` and not `inline`
    ///
    /// LLVM outlined it. On a `Map<[u8; 16], u64>` build it was **94.85% of the profile as its own
    /// symbol**, while `hashbrown`'s insert inlined into the caller entirely -- and the growth path
    /// is not the reason, `grow_for` has been `#[cold] #[inline(never)]` for as long as it has
    /// existed. It is the monomorphised probe and `place` together, and a plain `#[inline]` does
    /// not carry them.
    ///
    /// Measured, instructions per key, 694 keys, table pre-sized:
    ///
    /// | build | `#[inline]` | `#[inline(always)]` |
    /// |---|---|---|
    /// | `Map<[u8; 16], u64>` | 101.93 | **86.87** |
    /// | `Map<[u8; 16], u64>`, growing | 171.38 | **156.95** |
    /// | `Map<String, u64>` | 500.91 | **492.94** |
    /// | `Map<u64, u64>` | 67.05 | **66.11** |
    ///
    /// Cycles agree on every row. The cost is 592 bytes on a binary with three `Map` instantiations
    /// and nothing on the lookup, whose instruction counts are byte-identical either way.
    #[inline(always)]
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
        // SAFETY: `i` came from `site`, which returns a group base plus a lane -- both bounded by
        // the region, so it indexes both arrays. Three bounds checks on the write path otherwise,
        // for an index the probe already proved.
        unsafe {
            if *self.tags.get_unchecked(i) == EMPTY {
                self.room -= 1;
            }
            self.live += 1;
            *self.slots.get_unchecked_mut(i) = MaybeUninit::new((H::hold(a.id, k), v));
            *self.tags.get_unchecked_mut(i) = tag_of(a.place);
        }
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
        let slots = &mut self.slots;
        for_each_live(&self.tags, |i| {
            // SAFETY: the tag carries the live bit, so the slot is initialised. Every tag is
            // blanked immediately below, so nothing can observe a dropped slot as live.
            unsafe { slots[i].assume_init_drop() };
        });
        self.tags.fill(EMPTY);
        self.live = 0;
        self.room = limit(self.tags.len());
    }

    /// Keep the entries `keep` returns true for.
    pub fn retain(&mut self, mut keep: impl FnMut(H::Ref<'_>, &mut V) -> bool) {
        for i in 0..self.tags.len() {
            if self.tags[i] & FREE != 0 {
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
    /// # Why the split
    ///
    /// Every `insert` calls this, and in the overwhelming case it is a compare and a return. It
    /// was one function and not `#[inline]`, so that compare cost a call, a return and the
    /// register traffic around both -- `perf` put **13% of insert's time inside `reserve`**, a
    /// function that in the measured loop never grew anything. It is the same defect the module
    /// doc records on the read path, on the write path, found the same way.
    ///
    /// The test is now inline and the growth is `#[cold]`, so the branch predictor sees a
    /// never-taken branch instead of a call.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        if additional <= self.room {
            return;
        }
        self.grow_for(additional);
    }

    /// The growth `reserve` guards, kept out of line so the guard can be inlined.
    #[cold]
    #[inline(never)]
    fn grow_for(&mut self, additional: usize) {
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
        let fresh: Box<[MaybeUninit<(H, V)>]> = uninit_slots(slots);
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
        // A group at a time, like every other walk of the tag array. `taken` still moves per
        // entry and the indices still ascend, so the salvage guard's contract is unchanged: on an
        // unwind it drops from `taken` onward, and the slots skipped here were never live.
        let mut group = 0;
        while group + GROUP <= old.tags.len() {
            // SAFETY: bounded by the loop condition.
            let line: &[u8; GROUP] =
                unsafe { &*old.tags.as_ptr().add(group).cast::<[u8; GROUP]>() };
            let mut live = scan::occupied(line);
            while live != 0 {
                let i = group + live.trailing_zeros() as usize;
                live &= live - 1;
                old.taken = i + 1;
            // SAFETY: the tag says live, and `taken` now covers this index, so the salvage guard
            // will not drop it again if anything below unwinds.
            let entry = unsafe { old.slots.get_unchecked_mut(i).assume_init_read() };
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
                let free = scan::free(line);
                if free != 0 {
                    break base + free.trailing_zeros() as usize;
                }
                j += 1;
            };
            tags[at] = tag_of(p);
            fresh[at] = MaybeUninit::new(entry);
            }
            group += GROUP;
        }
        self.tags = tags;
        self.slots = fresh;
        self.geom = geometry;
        self.room = limit(self.slots.len()) - self.live;
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
        if additional <= self.room {
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
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K, V, H> {
        Iter { map: self, base: 0, mask: 0, ahead: 0, left: self.live }
    }

    /// Every entry, with the value mutable.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, H> {
        let (n, left) = (self.tags.len(), self.live);
        IterMut { map: std::ptr::from_mut(self), i: 0, n, left, _m: PhantomData }
    }

    /// The keys, in slot order.
    #[must_use]
    pub fn keys(&self) -> Keys<'_, K, V, H> {
        Keys(self.iter())
    }

    /// The values, in slot order.
    #[must_use]
    pub fn values(&self) -> Values<'_, K, V, H> {
        Values { map: self, base: 0, mask: 0, ahead: 0, left: self.live }
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
        Drain { map: self, base: 0, mask: 0, ahead: 0 }
    }

    /// The keys, consuming the map.
    #[must_use]
    pub fn into_keys(self) -> IntoKeys<K, V, H> {
        IntoKeys(self.into_iter())
    }

    /// The values, consuming the map.
    #[must_use]
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
            if t & FREE != 0 {
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
    /// Start of the group `mask` describes.
    base: usize,
    /// Live lanes of that group not yet yielded, one bit per lane.
    mask: u16,
    /// The next group to load.
    ahead: usize,
    left: usize,
}

impl<K: Coord + Eq, V, H: Held<K>> Iter<'_, K, V, H> {
    /// A walk that reads sixteen tags at a time instead of one.
    ///
    /// The obvious loop tests `tags[i] & FREE` per slot. At the table's 7/8 fill that is a
    /// predictable branch, but it is still one bounds-checked load and one test per slot, and the
    /// tombstones and empties are walked over one at a time. The scan the *probe* already uses
    /// answers the same question for a whole group in one instruction:
    /// [`crate::table::scan::occupied`] gathers the top bit of sixteen tags into a `u16`, and
    /// `EMPTY` is `0` and `TOMB` is `1`, so the mask is exactly the live lanes.
    ///
    /// Iterating the set bits of that mask is then `trailing_zeros` and `m & (m - 1)`, and an
    /// empty group costs one load and one test rather than sixteen.
    #[inline]
    fn refill(&mut self) -> bool {
        while self.ahead + GROUP <= self.map.tags.len() {
            self.base = self.ahead;
            self.ahead += GROUP;
            self.mask = crate::table::scan::occupied(self.map.line(self.base));
            if self.mask != 0 {
                return true;
            }
        }
        false
    }
}

impl<'a, K: Coord + Eq, V, H: Held<K>> Iterator for Iter<'a, K, V, H> {
    type Item = (H::Ref<'a>, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 && !self.refill() {
            return None;
        }
        let lane = self.mask.trailing_zeros() as usize;
        self.mask &= self.mask - 1;
        let i = self.base + lane;
        self.left -= 1;
        // SAFETY: the mask bit is set, which is the tag's live bit, which is the same fact the
        // per-slot test used to establish.
        let (h, v) = unsafe { self.map.slots[i].assume_init_ref() };
        Some((h.key(self.map.salt), v))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}
impl<K: Coord + Eq, V, H: Held<K>> ExactSizeIterator for Iter<'_, K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> std::iter::FusedIterator for Iter<'_, K, V, H> {}
impl<K: Coord + Eq, V, H: Held<K>> Clone for Iter<'_, K, V, H> {
    fn clone(&self) -> Self {
        Self {
            map: self.map,
            base: self.base,
            mask: self.mask,
            ahead: self.ahead,
            left: self.left,
        }
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
/// The values, and **only** the values.
///
/// This wrapped [`Iter`] and threw the key half away. On the elliptic face a key is not stored, it
/// is regenerated from the address by a multiply, so wrapping the pair iterator meant computing a
/// key per entry for a caller that had already said it did not want one. Walking the groups
/// directly is the same scan without that.
pub struct Values<'a, K: Coord + Eq, V, H: Held<K>> {
    map: &'a Map<K, V, H>,
    base: usize,
    mask: u16,
    ahead: usize,
    left: usize,
}

impl<'a, K: Coord + Eq, V, H: Held<K>> Iterator for Values<'a, K, V, H> {
    type Item = &'a V;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.mask != 0 {
                let i = self.base + self.mask.trailing_zeros() as usize;
                self.mask &= self.mask - 1;
                self.left -= 1;
                // SAFETY: the bit came from `occupied`, so the tag carries the live bit and the
                // slot invariant makes the entry initialised.
                return Some(unsafe { &self.map.slots.get_unchecked(i).assume_init_ref().1 });
            }
            if self.ahead + GROUP > self.map.tags.len() {
                return None;
            }
            self.base = self.ahead;
            self.ahead += GROUP;
            self.mask = scan::occupied(self.map.line(self.base));
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
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
    /// Start of the group `mask` describes. Its tags are already blanked.
    base: usize,
    /// Live lanes of that group whose values have not been handed out yet.
    mask: u16,
    /// The next group to claim.
    ahead: usize,
}

impl<K: Coord + Eq, V, H: Held<K>> Drain<'_, K, V, H> {
    /// Take the next entry out, clearing tombstones on the way past.
    ///
    /// The tags and the counters move a **group** at a time, in [`Self::claim_group`], and the map
    /// is consistent after every one: the sixteen tags read `EMPTY` and `live`/`used` agree with
    /// them. Dropping a `Drain` early still empties the map -- that is what makes it a drain, and
    /// `Drop` walks the rest.
    ///
    /// The one thing this gives up against the older per-entry form is the behaviour of
    /// `mem::forget` mid-group: the values of that group's unvisited lanes are leaked rather than
    /// left in the map. The map stays valid either way. `std`'s own `Drain` leaks on `forget`, and
    /// `forget` is a safe function precisely because a leak is not unsoundness.
    fn take_next(&mut self) -> Option<(K, V)> {
        loop {
            if self.mask != 0 {
                let i = self.base + self.mask.trailing_zeros() as usize;
                self.mask &= self.mask - 1;
                // SAFETY: the bit came from `occupied`, so this slot's tag carried the live bit
                // when the group was claimed and the slot invariant makes it initialised. Its tag
                // now reads `EMPTY`, so nothing else will read or drop the entry.
                let (h, v) = unsafe { self.map.slots.get_unchecked(i).assume_init_read() };
                return Some((h.into_key(self.map.salt), v));
            }
            if !self.claim_group() {
                return None;
            }
        }
    }

    /// Claim the next group holding anything: blank its sixteen tags in one store, move the
    /// counters by that group's counts, and keep its live lanes to yield from.
    ///
    /// A group at a time, not an entry at a time. Writing a tag and moving two counters per
    /// element made `drain` five times slower than `std`'s, which clears its control bytes in
    /// bulk.
    #[inline]
    fn claim_group(&mut self) -> bool {
        while self.ahead + GROUP <= self.map.tags.len() {
            self.base = self.ahead;
            self.ahead += GROUP;
            let (live, filled) = {
                // SAFETY: the loop condition bounds these sixteen bytes to the tag array.
                let line: &[u8; GROUP] =
                    unsafe { &*self.map.tags.as_ptr().add(self.base).cast::<[u8; GROUP]>() };
                (scan::occupied(line), !scan::match_tag(line, EMPTY))
            };
            if filled == 0 {
                continue;
            }
            // Written as a whole `[u8; GROUP]`, not `slice::fill`: the slice form does not know
            // its length at compile time and can come out as a call to `memset` for sixteen
            // bytes. The array form is one store.
            //
            // SAFETY: `self.base + GROUP <= tags.len()`, checked by the loop condition above.
            unsafe {
                *self.map.tags.as_mut_ptr().add(self.base).cast::<[u8; GROUP]>() = [EMPTY; GROUP];
            }
            self.map.room += filled.count_ones() as usize;
            self.map.live -= live.count_ones() as usize;
            self.mask = live;
            if live != 0 {
                return true;
            }
        }
        false
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
            if self.map.tags[i] & FREE != 0 {
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
            if self.tags[i] & FREE != 0 {
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
            if self.tags[i] & FREE == 0 {
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
                if m.tags[i] & FREE != 0 {
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
            if self.tags[i] & FREE == 0 {
                // SAFETY: the tag says live and this index is past `taken`, so nothing has moved
                // it out. Each index is visited once.
                unsafe { self.slots[i].assume_init_drop() };
            }
        }
    }
}

// -- the borrowed-key surface, parabolic face only ------------------------------------------------

impl<K: Coord + Eq, V> Map<K, V, Stored<K>> {
    /// The stored key and its value, from a view. [`Self::get_key_value`] without building a key.
    #[must_use]
    #[inline]
    pub fn get_key_value_at<Q: Locate<K>>(&self, q: Q) -> Option<(&K, &V)> {
        if !self.probeable() {
            return None;
        }
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        self.find_ref(a.place, |h| h.id == a.id && q.same(&h.k)).map(|s| (&s.0.k, &s.1))
    }

    /// Read through a borrowed view: `map.get_at("resref")` on a `Map<String, V>`.
    #[must_use]
    #[inline]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        if !self.probeable() {
            return None;
        }
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        let i = self.find(a.place, |h| h.id == a.id && q.same(&h.k))?;
        // SAFETY: `find` returns only live slots, and a live slot's index is a group base plus a
        // lane -- both bounded by the region, so it is inside `slots` by construction. The bounds
        // check the indexing operator adds is provably redundant here, and it is not free: it is
        // what made `get` cost four instructions more than `contains_key`, which never re-indexes.
        Some(unsafe { &self.slots.get_unchecked(i).assume_init_ref().1 })
    }

    /// The value a view names, mutably.
    #[inline]
    pub fn get_mut_at<Q: Locate<K>>(&mut self, q: Q) -> Option<&mut V> {
        if !self.probeable() {
            return None;
        }
        let a = already_mixed::<K>(self.salt, q.coord_at(self.salt.pepper()));
        let i = self.find(a.place, |h| h.id == a.id && q.same(&h.k))?;
        // SAFETY: as `get_at`.
        Some(unsafe { &mut self.slots[i].assume_init_mut().1 })
    }

    #[must_use]
    /// Whether a view's key is present.
    #[inline]
    pub fn contains_at<Q: Locate<K>>(&self, q: Q) -> bool {
        if !self.probeable() {
            return false;
        }
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
        if !self.probeable() {
            return None;
        }
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
    pub face: Sigma,
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
    #[expect(
        clippy::cast_precision_loss,
        reason = "a mean over probe counts, for a diagnostic report. The numerator would have to \
                  exceed 2^53 for the conversion to lose anything, which is more entries than the \
                  table can hold"
    )]
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
    #[must_use]
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
    #[inline]
    pub fn key(&self) -> H::Ref<'_> {
        // SAFETY: an occupied entry names a live slot.
        unsafe { self.map.slots.get_unchecked(self.at).assume_init_ref().0.key(self.map.salt) }
    }
    #[must_use]
    /// The value.
    #[inline]
    pub fn get(&self) -> &V {
        // SAFETY: as above.
        unsafe { &self.map.slots.get_unchecked(self.at).assume_init_ref().1 }
    }
    /// The value, mutably.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        // SAFETY: as above.
        unsafe { &mut self.map.slots.get_unchecked_mut(self.at).assume_init_mut().1 }
    }
    #[must_use]
    /// The value, borrowed for as long as the map is.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        // SAFETY: as above.
        unsafe { &mut self.map.slots.get_unchecked_mut(self.at).assume_init_mut().1 }
    }
    /// Replace the value, returning the old one.
    #[inline]
    pub fn insert(&mut self, v: V) -> V {
        std::mem::replace(self.get_mut(), v)
    }
    /// Take the entry out, returning the value.
    #[must_use]
    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
    /// Take the entry out, key and value.
    #[must_use]
    #[inline]
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
    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }
    /// Fill the slot.
    #[inline]
    pub fn insert(self, v: V) -> &'a mut V {
        let Self { map, at, a, key } = self;
        map.place(at, a, key, v);
        // SAFETY: just written.
        unsafe { &mut map.slots.get_unchecked_mut(at).assume_init_mut().1 }
    }
}

impl<'a, K: Coord + Eq, V, H: Held<K>> Entry<'a, K, V, H> {
    /// The value, inserting `v` if the slot is vacant.
    #[inline]
    pub fn or_insert(self, v: V) -> &'a mut V {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(v),
        }
    }
    /// The value, calling `f` for it only if the slot is vacant.
    #[inline]
    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Self::Occupied(e) => e.into_mut(),
            Self::Vacant(e) => e.insert(f()),
        }
    }
    /// The value, built from the key only if the slot is vacant.
    #[inline]
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
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
    #[must_use]
    /// Run `f` over the value if it is already there.
    #[inline]
    pub fn and_modify(mut self, f: impl FnOnce(&mut V)) -> Self {
        if let Self::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }
}

impl<K: Coord + Eq, V> Entry<'_, K, V, Stored<K>> {
    /// The key this entry is for, occupied or not.
    ///
    /// Offered on the parabolic face only. There `H::Ref` is `&K`, so an occupied slot and a vacant
    /// one hand back the same type; on the elliptic face an occupied entry *regenerates* its key as
    /// a value while a vacant one owns it, and there is no single type that is both.
    #[must_use]
    #[inline]
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
            let slots = &mut self.slots;
            for_each_live(&self.tags, |i| {
                // SAFETY: the tag carries the live bit, so the slot is initialised. This runs once,
                // at the end, so nothing observes the map afterwards.
                unsafe { slots[i].assume_init_drop() };
            });
        }
    }
}
/// Visit every live slot index, sixteen tags at a time.
///
/// The same scan the probe uses, factored out because `clone` and the rest walked the tag array
/// one byte at a time -- a load, a mask and a branch per slot, including for the empties, which at
/// a fresh 7/8 fill are an eighth of the array and after a `retain` can be most of it.
#[inline]
fn for_each_live(tags: &[u8], mut f: impl FnMut(usize)) {
    let mut base = 0;
    while base + GROUP <= tags.len() {
        // SAFETY: `base + GROUP <= tags.len()` is the loop condition.
        let line: &[u8; GROUP] = unsafe { &*tags.as_ptr().add(base).cast::<[u8; GROUP]>() };
        let mut m = scan::occupied(line);
        while m != 0 {
            f(base + m.trailing_zeros() as usize);
            m &= m - 1;
        }
        base += GROUP;
    }
}

/// `n` uninitialised slots, without touching them.
///
/// `(0..n).map(|_| MaybeUninit::uninit()).collect()` walks the whole range to build a box of
/// values that are by definition not worth writing, and at a million entries that loop was most of
/// what `clone` cost.
fn uninit_slots<T>(n: usize) -> Box<[MaybeUninit<T>]> {
    let mut v: Vec<MaybeUninit<T>> = Vec::with_capacity(n);
    // SAFETY: `MaybeUninit<T>` has no validity requirement, so every element of the reserved
    // capacity is already a valid `MaybeUninit<T>`. `len == capacity`, so no reallocation happens.
    unsafe { v.set_len(n) };
    v.into_boxed_slice()
}

impl<K: Coord + Eq, V: Clone, H: Held<K> + Clone> Clone for Map<K, V, H> {
    /// The copy holds every entry in the *same slot* the original does, so the two iterate in the
    /// same order and probe identically. Rebuilding by reinsertion would not guarantee that once
    /// tombstones are in play.
    ///
    /// # The one operation still behind `std` on word keys, and what is not the cause
    ///
    /// Against `std::collections::HashMap<u64, u64>`, this is 1.12x ahead at 2^10 entries, **0.71x
    /// at 2^14**, 0.90x at 2^17 and 1.04x at 2^20. Winning at both ends and losing in the middle
    /// is the shape of a fixed cost, not of a bad loop: at 2^10 there is nothing to copy, and at
    /// 2^20 bandwidth dominates and skipping the empty eighth pays for itself.
    ///
    /// **Rejected, measured:** a straight-line copy of all sixteen lanes for full groups, on the
    /// theory that a fixed-length form vectorises where a bitmask walk cannot -- and at the 7/8
    /// fill most groups are full, so it would have covered most of the array. It changed nothing,
    /// 0.63x against 0.63x. The per-lane branching is not the cost.
    ///
    /// The open suspect is structural: `hashbrown` puts its control bytes and its buckets in
    /// **one** allocation, and this map has two boxes -- two allocations and two memcpys against
    /// one. That is the kind of fixed cost that hides at both ends and shows in the middle. It is
    /// stated as a suspect and not a finding, because the microbenchmark written to price it
    /// compared two `calloc`s against one and so measured lazy zeroing instead. Merging the two
    /// boxes is the same thin-pointer change that would take `size_of::<Map>()` from 88 to 64.
    fn clone(&self) -> Self {
        let mut slots: Box<[MaybeUninit<(H, V)>]> = uninit_slots(self.slots.len());
        for_each_live(&self.tags, |i| {
            // SAFETY: `for_each_live` yields only lanes whose tag carries the live bit, and the
            // slot invariant says such a slot is initialised.
            let (h, v) = unsafe { self.slots.get_unchecked(i).assume_init_ref() };
            slots[i] = MaybeUninit::new((h.clone(), v.clone()));
        });
        Self {
            tags: self.tags.clone(),
            slots,
            live: self.live,
            room: self.room,
            geom: self.geom,
            salt: self.salt,
            _k: PhantomData,
        }
    }

    /// Overwrite `self` with `source`, reusing this map's allocation when the two are the same
    /// size.
    ///
    /// The default `clone_from` is `*self = source.clone()`: a fresh pair of arrays allocated and
    /// the old ones freed, every call. `hashbrown` specialises this and benchmarks it separately
    /// (`clone_from_small`, `clone_from_large`); not having it left a whole operation slower for
    /// no reason.
    fn clone_from(&mut self, source: &Self) {
        if self.slots.len() != source.slots.len() {
            *self = source.clone();
            return;
        }
        self.clear();
        let slots = &mut self.slots;
        for_each_live(&source.tags, |i| {
            // SAFETY: `for_each_live` yields live lanes of `source`, so that entry is initialised.
            // `self` was just cleared, so the destination slot holds nothing to drop.
            let (h, v) = unsafe { source.slots.get_unchecked(i).assume_init_ref() };
            slots[i] = MaybeUninit::new((h.clone(), v.clone()));
        });
        self.tags.copy_from_slice(&source.tags);
        self.live = source.live;
        self.room = source.room;
        self.geom = source.geom;
        self.salt = source.salt;
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
    /// Reserves from the iterator's lower bound, then inserts.
    ///
    /// Without the reserve this grows geometrically from whatever capacity it had, and the cost of
    /// that is not small: measured on `examples/whole_surface.rs` at a million pairs into an empty
    /// map, **150.5 instructions a pair against `hashbrown`'s 84.4**, for the same work `insert
    /// reserved` does in 56.4. Every doubling rehomes everything already placed, and a bare
    /// `for` loop pays for all of them. `from_iter` has always reserved; this did not.
    ///
    /// Half the hint when the map is not empty, because an extend into a populated map is the
    /// shape that overlaps -- reserving the full hint there buys capacity for pairs that turn out
    /// to be updates. `hashbrown` splits it the same way and for the same reason.
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, it: T) {
        let it = it.into_iter();
        let hint = it.size_hint().0;
        self.reserve(if self.live == 0 { hint } else { hint / 2 });
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
            if t & FREE != 0 {
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
    /// Reserves from the iterator's lower bound, then inserts. See the owning impl above.
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, it: T) {
        let it = it.into_iter();
        let hint = it.size_hint().0;
        self.reserve(if self.live == 0 { hint } else { hint / 2 });
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
    /// `Geom::first` and `Descent::next` shift by `gshift` under `assert_unchecked(gshift < 64)`,
    /// which is unsound if any reachable geometry exceeds it. `Geom::of` is the only constructor.
    #[test]
    fn the_shift_amount_is_always_in_range() {
        for gbits in 0..=48u32 {
            let g = Geom::of(1usize << gbits);
            assert!(g.gshift < 64, "gbits {gbits} gave gshift {}", g.gshift);
        }
        assert!(Geom::of(0).gshift < 64);
        assert!(Geom::default().gshift < 64);
    }

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
    /// **The two ways to recover a placement word agree.**
    ///
    /// `Stored::placement` re-addresses from the stored key when `K::CHEAP_EQ`, instead of running
    /// `spread(unspread(id))`. That is sound only because the two are the same word, and `rehome`
    /// places every live entry by it -- get it wrong and growth scatters entries where `find` will
    /// not look, which a lookup test on a table that never grew would not catch.
    #[test]
    fn re_addressing_a_stored_key_gives_the_word_the_bijection_would() {
        for pepper in [0u64, 1, 0x5EED_1234_ABCD_9876, u64::MAX] {
            let salt = Salt::new(pepper);
            for k in [0u64, 1, 2, 7, 0xFFFF_FFFF, 0x8000_0000_0000_0000, u64::MAX] {
                let a = <Stored<u64> as Held<u64>>::address(salt, &k);
                let held = <Stored<u64> as Held<u64>>::hold(a.id, k);
                assert_eq!(
                    held.placement(salt),
                    salt.placement_of(a.id),
                    "re-addressed and round-tripped placement disagree at key {k} pepper {pepper}"
                );
                assert_eq!(held.placement(salt), a.place, "neither is the word it was placed by");
            }
        }
    }

    #[test]
    #[expect(
        clippy::naive_bytecount,
        reason = "clippy suggests the `bytecount` crate. This crate has no dependencies and that \
                  is a deliberate property of it; a test counting a few hundred tags is not the \
                  reason to acquire the first one"
    )]
    fn a_tag_says_live_exactly_when_a_slot_holds_something() {
        let mut m: Map<String, u32> = Map::new();
        for i in 0..400u32 {
            m.insert(format!("k{i}"), i);
        }
        for i in (0..400u32).step_by(3) {
            m.remove(&format!("k{i}"));
        }
        let live = m.tags.iter().filter(|t| **t & FREE == 0).count();
        let tombs = m.tags.iter().filter(|t| **t == TOMB).count();
        let empty = m.tags.iter().filter(|t| **t == EMPTY).count();
        assert_eq!(live, m.live);
        // `room` carries `limit(slots) - used`, so this is the same assertion the other way round.
        assert_eq!(
            live + tombs,
            limit(m.slots.len()) - m.room,
            "room must account for every occupied slot"
        );
        assert_eq!(tombs, m.tombstones());
        assert_eq!(live + tombs + empty, m.tags.len());
        assert_eq!(m.tags.len(), m.slots.len());
        // Every live slot's stored identity word is the one the map would compute for its key, and
        // its tag is the one the placement word carries.
        for (i, &t) in m.tags.iter().enumerate() {
            if t & FREE != 0 {
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
        for _ in 0..if cfg!(miri) { 200 } else { 10_000 } {
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
