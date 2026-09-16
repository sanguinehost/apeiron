//! Turning a key into an address, without leaving the ring.
//!
//! A coordinate fold is arithmetic in `Z/2^64`, which is a commutative ring. So is the two-slot
//! algebra. Mixing in an operation from a *different* structure -- XOR, which is addition in
//! `GF(2)^64` -- is not a smaller version of the same thing; it is a different algebra, and the
//! multiplicative spread that made the fold work does not survive the crossing.
//!
//! Measured on hashtrinity's `(A, B)` fold, `a.wrapping_mul(PHI) ^ b`:
//!
//! | key set | XOR | `wrapping_add` |
//! |---|---|---|
//! | 2-tuple, `-64..63` | 49.2% collide | 0% |
//! | 2-tuple, `-512..511` | 49.9% collide | 0% |
//! | 3-tuple, `-16..15` | 71.8% collide | 0% |
//!
//! The cause is exact. A signed integer widens by sign extension, so a negative coordinate has
//! its whole high half set, and `PHI ^ -PHI` is `0xFFFF_FFFF_FFFF_FFFE` -- for any odd `v`,
//! `v ^ -v` is `!1`. XOR is self-inverse and linear over `GF(2)`, so it cannot tell that flip
//! apart from the matching flip in the other coordinate, and the two cancel. Addition carries,
//! so it cannot cancel: `(a1 - a2) * PHI = b2 - b1` has no small solutions when `PHI` is a large
//! odd multiplier.
//!
//! It is worth being clear about what this is *not*. The collisions do not lie on the null cone
//! `|a| = |b|` -- 126 of 8064 do, against 7938 that do not -- so this is not the algebra showing
//! through as a bug. It is a plain ring/field mismatch.
//!
//! # One fold is the wrong number of folds
//!
//! Staying in the ring is necessary and not sufficient, and the earlier claim that multiply-add
//! collides on 0% of keys was a property of the grids it was measured on. It is **not injective**
//! on `u32 x u32`: `f(a, b) = a * PHI + b` collides whenever `d * PHI = -e (mod 2^64)` for
//! `|d|, |e| < 2^32`, and the golden ratio's convergents are Fibonacci numbers, so
//! `d = 2_971_215_073` gives `e = 50_920_843` and `(2971215073, 50920843)` folds to the same
//! address as `(0, 0)`.
//!
//! The algebra says that is the wrong question to ask once, because the faces want different
//! things from an address:
//!
//! | face | wants | so the fold is |
//! |---|---|---|
//! | elliptic (*regenerated*) | the address *determines* the key -- nothing is degenerate | a bijection: [`pack`], and it refuses keys too wide to fit |
//! | parabolic (*carried*) | the address need not determine the key, which is carried beside it | any spreading fold; misses are caught by comparing the stored key |
//! | hyperbolic (*discarded*) | the address is a discard, and the key is gone | a scrambler, judged on *distribution*, never on injectivity |
//!
//! So injectivity is a requirement on one face, a non-requirement on the second, and impossible
//! on the third. A single fold cannot be right for all three, and measuring "the" collision rate
//! is measuring three different things at once.

/// The golden-ratio odd multiplier. Odd, so multiplication by it is a bijection on `Z/2^64`.
pub const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

/// A key's 64-bit address.
pub trait Coord {
    /// This key's address. Lossy for anything wider than a word.
    fn coord(&self) -> u64;

    /// Whether [`coord_with`](Coord::coord_with) already returns a well-spread word.
    ///
    /// **False by default, and it must be**: an integer's address is the integer, which is not
    /// spread at all, and a table that believed otherwise would put every key in one group.
    ///
    /// A byte fold overrides it. `fold_bytes` finishes with `mix2` -- a widening multiply, its
    /// halves folded together, and a guard term -- so its output is already as mixed as
    /// [`spread`] would make it. Running `spread` on top is a second widening multiply on the
    /// critical path of every string lookup, buying nothing: measured, that redundant mix is most
    /// of a 0.87 ns per-lookup constant that separated a `Map<String, _>` from `hashbrown` even at
    /// eight keys, where nothing can be a cache effect.
    ///
    /// Only the **parabolic** face may act on this. The elliptic face's address has to be the
    /// bijection whatever the key type, because it reads the key back off it.
    const SPREAD: bool = false;

    /// Whether comparing this key is as cheap as comparing a word.
    ///
    /// **False by default, and the default is the safe one**: a type that claims this wrongly
    /// loses speed and never correctness, because the key compare is the decision either way --
    /// see `proofs::the_parabolic_face_decides_on_the_key`, which says `Stored::holds` is true
    /// exactly when the keys are equal, whatever the identity words say.
    ///
    /// What it controls is one early reject. `Stored` keeps an identity word beside the key, and
    /// comparing it first skips a `memcmp` on the roughly one tag collision in twenty that a
    /// seven-bit tag lets through. For a `String` that is worth paying for; for a `u64` it is a
    /// second compare of the same information. Measured, `Map<K, V>::contains_key`:
    ///
    /// | key | with the early reject | without |
    /// |---|---|---|
    /// | `u64` | 34.82 instr, 7.28 cyc | **32.82, 6.14** |
    /// | `String`, 13-byte resref | 94.58, **20.61** | **92.84**, 20.65 |
    /// | `String`, 42-byte shared-prefix path | 131.65, **33.73** | **129.73**, 34.18 |
    ///
    /// The two shapes disagree, and they disagree in the way the split predicts: for a word key
    /// the reject costs two instructions and a cycle and cannot pay, and for a byte key it costs
    /// two instructions and *buys* a cycle, because it short-circuits ahead of the pointer and
    /// length loads and the `bcmp` call rather than behind them. `hashbrown` with `foldhash` is
    /// 33.15 on the `u64` row, so this const is also the difference between losing that row and
    /// winning it.
    ///
    /// It is not the hash-flooding defence and must not be mistaken for one. An attacker who can
    /// aim at the tag can already aim at the whole address; what stops that is the pepper, and
    /// `Map::with_pepper` is where it lives.
    const CHEAP_EQ: bool = false;

    /// The address, displaced by a process-wide constant.
    ///
    /// Defaults to ignoring the pepper, which is correct for every key that *is* its own address:
    /// an integer's fold has no fiber, so there is nothing an attacker could aim at and nothing
    /// for a pepper to move. Only a lossy fold overrides this -- see [`crate::fold`], where the
    /// pepper enters the seed rather than being applied to the finished address, because a
    /// bijection applied afterwards maps a collision to a collision and would buy nothing.
    #[inline]
    fn coord_with(&self, _pepper: u64) -> u64 {
        self.coord()
    }
}

macro_rules! int_coord {
    ($($t:ty),*) => {$(
        impl Coord for $t {
            const CHEAP_EQ: bool = true;
            #[inline]
            fn coord(&self) -> u64 {
                // `allow` and not `expect`: this macro expands for ten types and the lints fire
                // only for some of them, so an expectation would be unfulfilled on the rest.
                #[allow(
                    clippy::cast_sign_loss,
                    clippy::cast_lossless,
                    reason = "sign extension is deliberate, and is what makes `from_coord` round-trip"
                )]
                { *self as u64 }
            }
        }
    )*};
}
int_coord!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// Avalanche a finished address, so every input bit reaches every output bit.
///
/// An accumulating fold and a bucket index are different jobs, and running one function for both
/// is what breaks them. Horner's `acc * PHI + next` puts `next` into the *low* bits only, so the
/// high bits carry `acc` and nothing else -- and a table that follows the multiplicative-hashing
/// advice to take the top bits then recovers almost nothing. Measured on a signed 128x128 grid:
/// 99.2% of keys share their top fourteen bits before this, 36.9% after, which is the birthday
/// rate for 16384 keys in 16384 buckets and therefore the floor rather than a defect.
///
/// This is `splitmix64`'s finalizer: two xor-shifts and two odd multiplies, a bijection on
/// `Z/2^64` -- so it changes where a key lands and never whether two keys are the same key.
#[inline]
#[must_use]
pub const fn finish(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// The two words an address splits into -- one for identity, one for placement.
///
/// The crate's own trichotomy, applied to its own internals. A table asks two questions of a key,
/// and they are the elliptic and the hyperbolic question wearing table clothes:
///
/// - **identity** -- *is this the same key?* Answered by [`id`](Addr::id), the **elliptic**
///   bijection `x * PHI`. Exact, regenerable, injective. The elliptic face reads its key back off
///   it; the parabolic face compares it before the key.
/// - **placement** -- *where does it go?* Answered by [`place`](Addr::place), a **hyperbolic**
///   discard. It feeds the group descent and the tag, and it is judged on distribution and never
///   on injectivity -- there is deliberately no way to run it backwards.
///
/// One word did both jobs until 2026-08-29, because `(x * PHI ^ ...) * PSI` happened to be a
/// bijection that also spread. Separating them is what buys the lookup two cycles: the identity is
/// now `x * PHI` straight out of a **widening** multiply, and the placement is the *other* half of
/// that same multiply folded back in -- `rotl(hi, 32) ^ lo` -- so both come off one `mulx` instead
/// of a two-multiply serial chain. See [`spread`] for the shape and the measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Addr {
    /// The **placement** word: the group descent and the tag read this. A fold of the two halves
    /// of `x * PHI`, judged only on how evenly it spreads -- not injective, and it must not be
    /// mistaken for the identity. See [`spread`].
    pub place: u64,
    /// The **identity** word: `x * PHI`, the low half of the widening multiply. A bijection (PHI
    /// odd), so equal `id` means equal key, exactly. Stored in the slot; inverted by [`unspread`]
    /// to regenerate a key on the elliptic face.
    pub id: u64,
}

/// The map's address: one widening multiply, split into an identity word and a placement word.
///
/// Takes a coordinate, returns [`Addr`]. `id = x * PHI` is the low half -- a bijection, so
/// `unspread(spread(x).id) == x`, which is how the elliptic face reads its key back.
/// `place = rotl(hi, 32) ^ lo` folds both halves -- a discard, with no inverse.
///
/// Not [`finish`], which is two multiplies and three xor-shifts on a serial chain: ~14 cycles
/// before the first tag byte can be compared. A table build is memory-bound and pays that
/// happily; an L1-resident lookup *is* the latency chain.
///
/// | address | ns/link | ~cycles |
/// |---|---|---|
/// | two serial multiplies (the old `spread`) | 1.69 | 9.6 |
/// | `mulx` + `rotl` + `xor` (this) | **1.31** | **7.5** |
///
/// # What the two halves are
///
/// `lo` is a ring homomorphism composed with a unit, so it is additive exactly:
/// `lo(x + y) == lo(x) + lo(y)` in `Z/2^64`. `hi` is additive only up to the carry out of `lo`:
///
/// ```text
/// hi(x + y) == hi(x) + hi(y) + carry(lo(x), lo(y))
/// ```
///
/// Both checked over two million pairs, half of them with a live carry. That carry term is the
/// whole nonlinearity of the address, and it is why the wide product needs no second multiply.
///
/// It also says which face each half is on -- and here the doc used to reach upstream for a
/// licence the upstream theorems do not give it over this ring. What follows replaces that.
///
/// # The trichotomy is a theorem over the reals, and it does not survive the base change
///
/// The Lean development in `sanguine-apeiron/proof/Apeiron/Trichotomy` characterises the three
/// faces by three properties, each of which holds at exactly one sign of `sigma`. This crate's
/// arithmetic is `Z/2^64`. **All three characterising properties are constant in `sigma` there**,
/// so none of them can single a face out, and `proofs.rs` says so at full width or over `Z/2^8`
/// where the argument is uniform in the width:
///
/// | upstream, over `R` | over `Z/2^64` | harness |
/// |---|---|---|
/// | a nontrivial idempotent exists iff `0 < sigma` | **none exists, at any dial** | `the_word_ring_has_no_nontrivial_idempotent_at_any_dial` |
/// | no zero divisors iff `sigma < 0` | **a nilpotent at every dial** (`2^32`) | `the_word_ring_has_a_nilpotent_at_every_dial` |
/// | a discarding projection exists iff `0 < sigma` | **none exists, at any dial** | `no_word_projection_both_kills_and_keeps_at_any_dial` |
///
/// The first one fails for a reason worth stating, because it is not that the equivalence flips.
/// Both proofs reach `b * (2a - 1) = 0`. Over `R` that is split by having no zero divisors, and
/// the `a = 1/2` branch is where every nontrivial idempotent comes from. Over `Z/2^64` it is split
/// by `2a - 1` being **odd, hence a unit** -- a different argument reaching a *stronger*
/// conclusion, `b = 0` outright, with the dial never consulted. And `1/2` is not there to be
/// reached: `the_half_the_real_idempotents_need_is_not_in_the_word_ring`. [`crate::scalar::Halve`]
/// already refuses to be implemented for a word; these say the objects it would have built are
/// absent and not merely unconstructible.
///
/// **So the face names on `place` and `id` are an analogy with the real case, not a transfer.**
/// Saying otherwise was the thing this crate exists not to do. What is actually true about the two
/// halves is true, is proved directly, and needs none of the above:
///
/// - `id` is a bijection because `PHI` is **odd**, hence a unit -- `multiplying_by_an_odd_constant_is_injective`
///   and `the_elliptic_address_has_a_trivial_kernel`. That is the whole reason the elliptic face
///   can read its key back, and it is a fact about units in `Z/2^64` rather than about `sigma`.
/// - `place` has no closed-form inverse, which is what the security argument needs: matching `j`
///   windows of it costs `2^(j * gbits)` trials. That is a statement about the *windows*
///   (`the_group_index_cannot_separate_keys`) and not about whether the whole word is injective.
///   This doc used to say `place` "is not injective, by construction". **That is false over every
///   key a caller will ever use.** `place_separates_keys_below_two_to_the_thirty_two` verifies in
///   32s that `place` separates *every* pair of keys below `2^32` -- four billion of them, with no
///   collision. A search first found none over 2^26 dense and 2^26 scattered keys, which settles
///   nothing on its own (a uniformly random 64-bit function has about 0.0001 expected collisions
///   at that sample size); the model checker settles it. Whether `place` is injective on the full
///   64-bit domain is open and does not matter here: what the security argument needs is that it
///   has no closed-form inverse, and injectivity would not give an attacker one.
///
/// # What does transfer, and it is now a theorem upstream rather than an argument here
///
/// `sanguine-apeiron/proof/Apeiron/Trichotomy/AnyRing.lean` carries the construction over an
/// **arbitrary commutative ring**, so `Z/2^64` is an instance of it rather than an analogy for it.
/// Two results cross:
///
/// - `AnyRing.ringNormForm_mul`: `N(zw) = N(z) N(w)`. An identity between two polynomials with
///   whole-number coefficients holds in every commutative ring or in none.
/// - `AnyRing.exists_inverse_iff_isUnit_ringNormForm`: a point can be divided by exactly when its
///   norm form can. Over `R` that reads as lying off the cone, because the divisors of `R` are the
///   non-zero reals. Over `Z/2^64` the same theorem reads as **the norm form being odd**. The
///   statement does not change; what counts as a divisor does.
///
/// And the idempotent result crosses with its shape changed, which is the useful part:
/// `AnyRing.ringIdempotent_trivial` says a point unchanged by squaring is one of the two every
/// ring has, given that `2a - 1` can be divided by and that one of `a`, `a - 1` can. Over `R`
/// the first is supplied by having no zero divisors and the `a = 1/2` branch is what escapes it;
/// over `Z/2^n` both hypotheses hold at *every* element, because the divisors there are the odd
/// elements and `2a - 1` is always odd. So the conclusion is unconditional here and the dial is
/// never consulted, which is the same fact `the_word_ring_has_no_nontrivial_idempotent_at_any_dial`
/// checks at machine width.
///
/// Those four Lean theorems depend on `[propext, Quot.sound]` and not on `Classical.choice`.
///
/// **`Addr` is not a point of the algebra at any one `sigma`, and an earlier version of this doc
/// said it was a `sigma = 0` pair. That is refuted upstream**: `NullBasis.upper_eq_lower_of_disc_zero`
/// is `upper 0 z = lower 0 z` -- at zero the two channels *agree*, so the parabolic face has no
/// second channel to be independent. `id` and `place` are not the same read; one is injective and
/// the other has no closed-form inverse. So the pair cannot sit at `sigma = 0` -- and by the table above it
/// does not sit at the other two either, over this ring.
///
/// What it is: **two reads at two different faces, taken off one multiply.** The widening product
/// is what makes that cheap, and the three-face table at the top of this module is what says why a
/// table wants both at once -- the parabolic face carries the key beside the address precisely
/// because one word cannot be a bijection and a discard together.
///
/// The `^` is a step out of `Z/2^64` and into `GF(2)^64`, the crossing this module opens by
/// warning about. It is safe *here* and was not safe there for a stateable reason: hashtrinity
/// xored two ring elements whose sum still had to be injective, and XOR cannot carry so the two
/// sign extensions cancelled. This xors a cocycle value into a ring element to produce `place`,
/// which is judged on distribution and never on injectivity. Different job, different obligation.
///
/// # Which bits of `place` are which
///
/// `rotate_left(32)` sends bit `j` of `hi` to bit `j + 32`, so the whole word is
///
/// ```text
/// place_i == lo_i ^ hi_((i - 32) mod 64)
/// ```
///
/// -- verified bit for bit over two million keys. Sixty-four independent `GF(2)` functionals of
/// the 128 product bits, so *disjoint slices of `place` are independent evidence*, which is the
/// property [`crate::table::tag_of`] needs and the reason the tag may sit directly above the
/// descent.
///
/// It also derives the low-window rejection that `crate::map::Geom::first` records as a
/// measurement. The tag slice is `lo_57..64 ^ hi_25..32`. A low window would be
/// `lo_0..32 ^ hi_32..64` -- and `hi_32..64` is zero for every key whose product does not fill 128
/// bits: over `x < 200_000`, `hi`'s top half took **one** value and the low 32 bits of `place`
/// equalled the raw low bits of `lo` in 200,000 of 200,000 cases, while the tag took all 128
/// values. The high window is forced, not preferred.
///
/// | fold | worst mean | worst max | worst compares | fails |
/// |---|---|---|---|---|
/// | `hi ^ lo` | 1.547 | 6 | 1.367 | `align4096` |
/// | `hi ^ rotl(lo, 32)` | 2.675 | 13 | 20.6 | six cells |
/// | `hi + lo` | 1.547 | 6 | 1.362 | `align4096` |
/// | `(hi ^ lo) * PSI` | 1.028 | 7 | 1.057 | none, but two multiplies again |
/// | **`rotl(hi, 32) ^ lo`** (this) | **1.075** | **8** | **1.083** | **none** |
///
/// **The rotate is bought by that table and not by non-invertibility, which is what it is easy to
/// say instead.** `hi ^ lo` is no more invertible than this: given a target, either one leaves you
/// iterating on a guess of `hi` through a multiply, and neither iteration contracts. What the
/// rotate does is route `hi`'s **low** bits into the top window, and that matters because for the
/// keys that fail without it -- page-aligned ones, `i * 4096` -- `hi` is about `i * PHI / 2^52` and
/// so occupies only its bottom thirty bits. Without the rotate the window sees `lo` alone.
///
/// Re-measured against the gate rather than taken from the table: `hi ^ lo` gives `align4096` a
/// mean probe of **1.547** against the 1.5 bound, which is the number this table records, to three
/// decimals. `align4096` was **not in `tests/the_probe_stays_flat.rs`** until it was put there, so
/// the gate could not have caught the fold it rejects.
///
/// A single window of the 128-bit product, `(w >> 32) as u64`, was tried as a one-instruction
/// replacement for the rotate and the xor together. It fails worse and in a way worth recording:
/// `serial` reaches 36.37 mean compares, because the tag then reads the product's bits 89 to 96,
/// which are `hi`'s bits 25 to 32 and are zero for every key below about 5.4e7. The tag dies before
/// the window does.
///
/// `place` is deliberately not invertible. The old single word was, so aiming a collision chain
/// was one `unspread` call; now keys agreeing on `j` windows must be found at `2^(j * gbits)`
/// trials each.
#[inline]
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the two truncations ARE the split: `lo` is the low half of the 128-bit product and \
              `hi` the high half, and taking each is what makes one widening multiply yield both \
              an identity word and a placement word"
)]
pub const fn spread(x: u64) -> Addr {
    let w = (x as u128).wrapping_mul(PHI as u128);
    let lo = w as u64;
    let hi = (w >> 64) as u64;
    Addr {
        place: hi.rotate_left(32) ^ lo,
        id: lo,
    }
}

/// The identity word undone: `id * PHI_INV`, one multiply. The elliptic face regenerates its key
/// through this.
///
/// `spread(x).id` is `x * PHI`, so this is the odd-multiply inverse and nothing else -- the whole
/// bijection is now a single multiply each way, which is why `proofs.rs` can decide it as a
/// concrete equation rather than leaving the composite to CBMC's 64-bit multiplier reasoning. It
/// does **not** invert `place`: the placement fold has fibers and is deliberately one-way.
#[inline]
#[must_use]
pub const fn unspread(id: u64) -> u64 {
    id.wrapping_mul(PHI_INV)
}

/// `PHI * PHI_INV == 1` in `Z/2^64`. Checked by `proofs.rs`, not taken on faith.
pub const PHI_INV: u64 = 0xF1DE_83E1_9937_733D;

/// splitmix64's second multiplier, the one [`finish`] applies between its xor-shifts (written
/// there as a literal). Odd, so it is a bijection on `Z/2^64`.
///
/// It was [`spread`]'s second multiplier too, until the identity/placement split made `spread` a
/// single widening multiply on 2026-08-29. It is kept because `finish` still uses the value and
/// because `proofs.rs` checks [`inverse_odd`] against it as a known-answer fixture.
pub const PSI: u64 = 0x94D0_49BB_1331_11EB;

/// `PSI * PSI_INV == 1` in `Z/2^64`. Checked by `proofs.rs`, not taken on faith.
pub const PSI_INV: u64 = 0x3196_42B2_D24D_8EC3;

/// The inverse of an odd number mod `2^64`.
///
/// Newton's iteration on `f(x) = 1/x - a`: each step doubles the number of correct low bits, and
/// the seed `3a xor 2` is already correct to five, so five steps cover 64. Exists because
/// [`crate::map`]'s pepper becomes the multiplier of its address bijection, and a bijection the
/// table cannot run backwards is not one the elliptic face can regenerate a key through.
///
/// Temporal and spatial O(1): the `while` loop is a fixed five iterations, independent of `a`, and
/// each iteration is three word operations (two `wrapping_mul` and one `wrapping_sub`), so the count
/// does not grow with the input. Spatial O(1): two `u64` locals, no allocation. The five is the
/// derivation and not a guess -- Newton's step doubles the number of correct low bits and the seed
/// is already correct to five, so five steps carry that past the 64 bits of the word.
///
/// Undefined for even `a`, which has no inverse. Callers force the low bit.
#[must_use]
pub const fn inverse_odd(a: u64) -> u64 {
    let mut x = a.wrapping_mul(3) ^ 2;
    let mut i = 0;
    while i < 5 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        i += 1;
    }
    x
}

/// Which bucket an address falls in, for a table of `2^bits` buckets.
///
/// Takes the top bits of the avalanched address. Separated from the address itself because the
/// elliptic face needs the *address* to determine the key -- a bijection, whose entropy sits
/// wherever the packing put it -- while the *bucket* needs the entropy spread. `Packed::packed`
/// keeps its low bits meaningful and its high bits nearly constant, so bucketing on a slice of it
/// directly is the same mistake in the other direction.
#[inline]
#[must_use]
pub const fn bucket(address: u64, bits: u32) -> u64 {
    finish(address) >> (64 - bits)
}

/// Mix one more coordinate into a running address, losing information.
///
/// Multiply then add, both in `Z/2^64`. Each step is a bijection on a single word, but the
/// composite maps two words to one and therefore cannot be injective: it is a **discard**, and
/// belongs to the hyperbolic face. Judge it on how evenly it spreads, never on whether it
/// separates -- there are `2^64` addresses and `2^128` pairs, so it does not.
#[inline]
#[must_use]
pub fn mix(acc: u64, next: u64) -> u64 {
    acc.wrapping_mul(PHI).wrapping_add(next)
}

/// A bijective address for keys narrow enough to have one.
///
/// Concatenation, not arithmetic: `hi` in the top `shift` bits, `lo` beneath. Exactly invertible
/// by construction, which is what the elliptic face requires and what no multiply-add fold of two
/// full words can provide. `shift` is the width of `lo`, and the caller is responsible for `hi`
/// fitting in the rest -- see [`Packed`] for the types where that is checked by construction.
#[inline]
#[must_use]
pub const fn pack(hi: u64, lo: u64, shift: u32) -> u64 {
    (hi << shift) | lo
}

/// A key whose address determines it exactly: `unpack(k.packed()) == k`.
///
/// The elliptic face. Deliberately a **separate trait from [`Coord`]**, not a refinement of it,
/// because the two are different operations that give different answers: `packed` is a bijection
/// and `coord` is a discard. A key can have both, and it should not be possible to reach for one
/// while meaning the other.
///
/// Implemented only for keys that fit in 64 bits, because that is exactly when the bijection
/// exists. There is no honest `Packed` for two full `u64`s.
pub trait Packed: Sized {
    /// The exact address.
    fn packed(&self) -> u64;
    /// The key that address came from.
    fn unpack(c: u64) -> Self;
}

macro_rules! int_packed {
    ($($t:ty),*) => {$(
        impl Packed for $t {
            #[inline]
            // `allow` and not `expect`: ten expansions, and the lints fire for only some.
            #[allow(
                clippy::cast_sign_loss,
                clippy::cast_lossless,
                reason = "half of a round trip, not a conversion: the sign bit travels as data, \
                          `unpack` restores it exactly, and `tests/coord_is_injective.rs` checks \
                          that round trip over the full range of every type this macro covers"
            )]
            fn packed(&self) -> u64 { *self as u64 }
            #[inline]
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                reason = "the other half: `c` came from `packed`, so the bits dropped here are \
                          exactly the sign extension it produced -- for any other `c` this claims \
                          nothing"
            )]
            fn unpack(c: u64) -> Self { c as Self }
        }
    )*};
}
int_packed!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// Two 32-bit halves fill one word exactly.
impl Packed for (u32, u32) {
    #[inline]
    fn packed(&self) -> u64 {
        pack(u64::from(self.0), u64::from(self.1), 32)
    }
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`pack` put the two halves at 32 and 0; recovering them is a shift and a truncation, \
                      which is that composition run backwards."
    )]
    fn unpack(c: u64) -> Self {
        ((c >> 32) as u32, c as u32)
    }
}

/// Signed halves, bit-cast. Two's complement makes the round-trip exact.
impl Packed for (i32, i32) {
    #[inline]
    fn packed(&self) -> u64 {
        // `cast_unsigned` rather than `as u32`: the sign bit is being carried as data, and saying
        // so in the type system is better than silencing a lint that is right to ask.
        pack(
            u64::from(self.0.cast_unsigned()),
            u64::from(self.1.cast_unsigned()),
            32,
        )
    }
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`pack` put the halves at 32 and 0, so recovering them is that composition run \
                  backwards; `cast_signed` carries the sign back without a second suppression"
    )]
    fn unpack(c: u64) -> Self {
        (((c >> 32) as u32).cast_signed(), (c as u32).cast_signed())
    }
}

/// Three signed 21-bit coordinates and a spare bit: a world of +/- one million units a side,
/// larger than any Aurora area and most worlds built out of them.
impl Packed for (i21, i21, i21) {
    #[inline]
    fn packed(&self) -> u64 {
        pack(pack(self.0.bits(), self.1.bits(), 21), self.2.bits(), 21)
    }
    #[inline]
    fn unpack(c: u64) -> Self {
        (
            i21::from_bits((c >> 42) & 0x1F_FFFF),
            i21::from_bits((c >> 21) & 0x1F_FFFF),
            i21::from_bits(c & 0x1F_FFFF),
        )
    }
}

/// A 21-bit signed integer, so three fit in a word with a bit to spare.
///
/// Exists because the elliptic face's bijection is only available when the key actually fits, and
/// a type that cannot hold an out-of-range value is how that is enforced rather than hoped for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[expect(
    non_camel_case_types,
    reason = "Named for the width it holds, the way `i32` and `u8` are. Camel case would hide that this is \
                  an integer type and not a struct that happens to wrap one."
)]
pub struct i21(i32);

impl i21 {
    /// The smallest representable value.
    pub const MIN: i32 = -(1 << 20);
    /// The largest representable value.
    pub const MAX: i32 = (1 << 20) - 1;

    /// `None` if `v` will not fit, which is the whole point of the type.
    #[must_use]
    pub const fn new(v: i32) -> Option<Self> {
        if v >= Self::MIN && v <= Self::MAX {
            Some(Self(v))
        } else {
            None
        }
    }

    #[must_use]
    /// The value, widened.
    pub const fn get(self) -> i32 {
        self.0
    }

    #[inline]
    const fn bits(self) -> u64 {
        // The sign bit travels into the 21-bit field as data; `from_bits` sign-extends it back,
        // and the constructor has already bounded the value to `MIN..=MAX`.
        (self.0.cast_unsigned() as u64) & 0x1F_FFFF
    }

    #[inline]
    const fn from_bits(b: u64) -> Self {
        // Sign-extend from bit 20. `cast_signed` before the arithmetic shift leaves a value in
        // `i21::MIN..=i21::MAX`, which is why the narrowing needs no suppression: clippy can see
        // the range. It used to carry two.
        Self(((b << 43).cast_signed() >> 43) as i32)
    }
}

impl Coord for i21 {
    const CHEAP_EQ: bool = true;
    #[inline]
    fn coord(&self) -> u64 {
        self.bits()
    }
}

/// The lossy folds, for the faces that do not need a bijection. Generic over any `Coord`, and
/// deliberately *not* [`Packed`]: a fold of two words into one cannot round-trip.
impl<A: Coord, B: Coord> Coord for (A, B) {
    /// A composite is cheap to compare exactly when every component is.
    const CHEAP_EQ: bool = A::CHEAP_EQ && B::CHEAP_EQ;
    /// Temporal O(1) of its own -- one [`mix`] then one [`finish`], a fixed count of word
    /// operations -- plus the components' own `coord`, which is O(1) for fixed-width components and
    /// O(key length) for the byte folds, so the composite's cost is inherited and not visible from
    /// `(A, B)`. Spatial O(1): two word temporaries, no allocation.
    #[inline]
    fn coord(&self) -> u64 {
        finish(mix(self.0.coord(), self.1.coord()))
    }
    /// As [`Coord::coord`]: one [`mix`] and one [`finish`], plus the components' `coord_with`.
    /// Spatial O(1).
    #[inline]
    fn coord_with(&self, pepper: u64) -> u64 {
        finish(mix(self.0.coord_with(pepper), self.1.coord_with(pepper)))
    }
}

impl<A: Coord, B: Coord, C: Coord> Coord for (A, B, C) {
    const CHEAP_EQ: bool = A::CHEAP_EQ && B::CHEAP_EQ && C::CHEAP_EQ;
    /// Temporal O(1) of its own -- two [`mix`] calls then one [`finish`], one `mix` per extra
    /// component, a fixed count of word operations -- plus the components' own `coord`, which is
    /// O(1) for fixed-width components and O(key length) for the byte folds. Spatial O(1): word
    /// temporaries, no allocation.
    #[inline]
    fn coord(&self) -> u64 {
        finish(mix(mix(self.0.coord(), self.1.coord()), self.2.coord()))
    }
    /// As [`Coord::coord`], through the components' `coord_with`. Spatial O(1).
    #[inline]
    fn coord_with(&self, p: u64) -> u64 {
        finish(mix(
            mix(self.0.coord_with(p), self.1.coord_with(p)),
            self.2.coord_with(p),
        ))
    }
}

impl<A: Coord, B: Coord, C: Coord, D: Coord> Coord for (A, B, C, D) {
    const CHEAP_EQ: bool = A::CHEAP_EQ && B::CHEAP_EQ && C::CHEAP_EQ && D::CHEAP_EQ;
    /// Temporal O(1) of its own -- three [`mix`] calls then one [`finish`], one `mix` per extra
    /// component, a fixed count of word operations -- plus the components' own `coord`, which is
    /// O(1) for fixed-width components and O(key length) for the byte folds. Spatial O(1): word
    /// temporaries, no allocation.
    #[inline]
    fn coord(&self) -> u64 {
        finish(mix(
            mix(mix(self.0.coord(), self.1.coord()), self.2.coord()),
            self.3.coord(),
        ))
    }
    /// As [`Coord::coord`], through the components' `coord_with`. Spatial O(1).
    #[inline]
    fn coord_with(&self, p: u64) -> u64 {
        finish(mix(
            mix(
                mix(self.0.coord_with(p), self.1.coord_with(p)),
                self.2.coord_with(p),
            ),
            self.3.coord_with(p),
        ))
    }
}
