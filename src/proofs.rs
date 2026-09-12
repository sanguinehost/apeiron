//! Bounded proofs. `cfg(kani)` only.

#[cfg(kani)]
mod harnesses {
    use crate::coord::{bucket, i21, Packed};

    /// The elliptic face's requirement: the address determines the key, over the whole domain.
    #[kani::proof]
    fn a_packed_i32_pair_round_trips() {
        let x: i32 = kani::any();
        let y: i32 = kani::any();
        assert_eq!(<(i32, i32)>::unpack((x, y).packed()), (x, y));
    }

    #[kani::proof]
    fn a_packed_u32_pair_round_trips() {
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        assert_eq!(<(u32, u32)>::unpack((x, y).packed()), (x, y));
    }

    /// Injectivity directly: two distinct keys never share an address.
    #[kani::proof]
    fn packing_is_injective() {
        let a: (i32, i32) = (kani::any(), kani::any());
        let b: (i32, i32) = (kani::any(), kani::any());
        kani::assume(a != b);
        assert_ne!(a.packed(), b.packed());
    }

    /// The 21-bit coordinate refuses what will not fit, and round-trips what does.
    #[kani::proof]
    fn a_narrow_coordinate_is_exact_or_refused() {
        let v: i32 = kani::any();
        match i21::new(v) {
            Some(n) => {
                assert_eq!(n.get(), v);
                assert!(v >= i21::MIN && v <= i21::MAX);
            }
            None => assert!(v < i21::MIN || v > i21::MAX),
        }
    }

    #[kani::proof]
    fn a_packed_triple_round_trips() {
        let (x, y, z): (i32, i32, i32) = (kani::any(), kani::any(), kani::any());
        kani::assume(i21::new(x).is_some() && i21::new(y).is_some() && i21::new(z).is_some());
        let k = (
            i21::new(x).unwrap(),
            i21::new(y).unwrap(),
            i21::new(z).unwrap(),
        );
        assert_eq!(<(i21, i21, i21)>::unpack(k.packed()), k);
    }

    /// `finish` must permute, not fold: it may move a key, never merge two. If it folded,
    /// `bucket` would lose keys that `packed` had separated and the elliptic face's guarantee
    /// would end at the table's door.
    ///
    /// Proved by its steps rather than end to end. `finish` is four operations, and the whole
    /// composition against two symbolic words is two pairs of 64-bit multiplies -- the solver does
    /// not return. Each step alone does, in milliseconds, and a composition of bijections is a
    /// bijection. That is the decomposition, stated so the gap is visible: what is machine-checked
    /// is that each step permutes, and the composition is an argument on top of it.
    #[kani::proof]
    fn a_xor_shift_step_is_injective() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        kani::assume(a != b);
        // The two shift distances `finish` uses on the way in, and the one on the way out.
        assert_ne!(a ^ (a >> 30), b ^ (b >> 30));
        assert_ne!(a ^ (a >> 27), b ^ (b >> 27));
        assert_ne!(a ^ (a >> 31), b ^ (b >> 31));
        // `fold::settle`'s, which is the other half of its bijection: a `PHI` multiply, covered by
        // `multiplying_by_an_odd_constant_is_injective`, then this.
        assert_ne!(a ^ (a >> 32), b ^ (b >> 32));
    }

    /// Multiplication by an odd constant permutes `Z/2^64`, for both of `finish`'s multipliers
    /// and for `PHI`.
    /// Stated as the difference rather than as two products: `a * C == b * C` iff
    /// `(a - b) * C == 0`, and for odd `C` that forces `a - b == 0`. One multiply instead of two,
    /// which is the difference between deciding and not.
    #[kani::proof]
    fn multiplying_by_an_odd_constant_is_injective() {
        let d: u64 = kani::any();
        kani::assume(d != 0);
        assert_ne!(d.wrapping_mul(0xBF58_476D_1CE4_E5B9), 0);
        assert_ne!(d.wrapping_mul(0x94D0_49BB_1331_11EB), 0);
        assert_ne!(d.wrapping_mul(crate::coord::PHI), 0);
    }

    /// The SWAR group scan is byte equality, on every line and every tag.
    ///
    /// `tests/both_tag_paths_agree.rs` checks this against SSE2 and against a scalar reference on
    /// four thousand sampled lines. Sampling is the wrong instrument for a bit trick: the flaw it
    /// did catch -- a matching lane borrowing into its neighbour -- needed two specific adjacent
    /// bytes, and the next one might need three. Here the line and the tag are both symbolic, so
    /// the sixteen lanes and all 256 tags are covered at once rather than visited.
    #[kani::proof]
    fn the_swar_group_scan_is_byte_equality() {
        let line: [u8; 16] = kani::any();
        let tag: u8 = kani::any();
        let got = crate::table::scan::match_tag_swar(&line, tag);
        let mut want = 0u16;
        let mut i = 0usize;
        while i < 16 {
            if line[i] == tag {
                want |= 1 << i;
            }
            i += 1;
        }
        assert_eq!(got, want);
    }

    /// And the vacancy scan is the top bit, which is what tells a free slot from a full one.
    ///
    /// The top bit marks **free**, not occupied: both sentinels set it and `tag_of` never does, so
    /// the write path -- which asks for free lanes -- gets its answer straight out of the
    /// `movemask` with no `not`, and `tag_of` needs no `or` to put the bit back. `scan::occupied`
    /// is the complement and only `rebuild` calls it.
    #[kani::proof]
    fn the_swar_vacancy_scan_is_the_top_bit() {
        let line: [u8; 16] = kani::any();
        let got = crate::table::scan::free_swar(&line);
        let mut want = 0u16;
        let mut i = 0usize;
        while i < 16 {
            if line[i] & 0x80 != 0 {
                want |= 1 << i;
            }
            i += 1;
        }
        assert_eq!(got, want);
        // And the two are exact complements, which is the property `rebuild` leans on.
        assert_eq!(crate::table::scan::occupied(&line), !got);
    }

    /// A bucket index is in range for its table.
    #[kani::proof]
    fn a_bucket_index_fits_its_table() {
        let address: u64 = kani::any();
        let bits: u32 = kani::any();
        kani::assume(bits >= 1 && bits <= 63);
        assert!(bucket(address, bits) < (1u64 << bits));
    }

    // ── the table ────────────────────────────────────────────────────────────────────────────
    //
    // Everything above is about turning a key into an address. These are about the table that
    // holds it, which is where both hangs in this crate's history came from: a shard region that
    // could overflow, and a load factor that could fill a region exactly. Neither was a wrong
    // answer -- each was a probe that never returned.

    /// A region always has room. This is `locate`'s termination precondition.
    ///
    /// `locate` walks until it finds an empty slot. If a region can be exactly filled there is no
    /// empty slot to find and an absent key loops forever -- a hang an attacker can aim, since the
    /// shard a key lands in is a function of the key. Measured once at load factor 1x, where
    /// `next_power_of_two` can land exactly on the load.
    #[kani::proof]
    fn a_region_always_has_room_for_its_load() {
        let load: usize = kani::any();
        // Bounded: the proof is about the arithmetic, and a larger load cannot be allocated on any
        // machine this runs on.
        kani::assume(load <= (1 << 40));

        let slots = crate::table::region_slots_for(load);
        if load == 0 {
            assert_eq!(slots, 0);
        } else {
            assert!(slots >= 2 * load, "a region holds at least twice its load");
            assert!(
                slots.is_power_of_two(),
                "the group index is a mask, so this must be exact"
            );
            assert!(
                slots >= 32,
                "at least two groups, so the descent has somewhere to go"
            );
        }
    }

    /// Once the address is spent the descent sweeps by exactly one group a step.
    ///
    /// This is what makes `locate` terminate: the sweep is a cycle over every group, so a region
    /// with any empty slot is found within one lap. If a step could stall or repeat, the walk
    /// could miss the only empty group forever.
    #[kani::proof]
    fn the_spent_descent_advances_by_exactly_one_group() {
        let p: u64 = kani::any();
        let gbits: u32 = kani::any();
        kani::assume(gbits >= 1 && gbits <= 12);
        let gmask: usize = (1usize << gbits) - 1;

        let full = crate::table::DESCENT_BITS / gbits;
        let j: u32 = kani::any();
        // Both steps past the point where the word is consumed.
        kani::assume(j >= full && j < full + 4096);

        let here = crate::table::group_at(p, j, gbits, gmask);
        let next = crate::table::group_at(p, j + 1, gbits, gmask);
        assert_eq!(next, (here + 1) & gmask, "the sweep advances one group");
        assert!(here <= gmask, "and never leaves the table");
    }

    /// A group index is always inside the region, spent word or not.
    #[kani::proof]
    fn a_group_index_is_always_in_range() {
        let p: u64 = kani::any();
        let j: u32 = kani::any();
        let gbits: u32 = kani::any();
        kani::assume(gbits <= 12);
        kani::assume(j < 100_000);
        let gmask: usize = if gbits == 0 { 0 } else { (1usize << gbits) - 1 };
        assert!(crate::table::group_at(p, j, gbits, gmask) <= gmask);
    }

    // ── the root: is this actually the split-complex algebra? ────────────────────────────────
    //
    // The laws below are currently *tested* over a 7x7 grid of small integers. That samples the
    // structure; it does not establish it. These prove them over symbolic `Dyadic` -- the scalar
    // that holds the idempotents -- which is the path on which this is the algebra
    // `sanguine/Trichotomy/SplitComplex.lean` defines rather than something shaped like it.
    //
    // Bounded, and the bound is stated rather than implied: `Dyadic::mul` is an i128 multiply plus
    // a normalising `trailing_zeros`, and the solver does not return on the unbounded form. What
    // is proved is the law over that range, not over all of Z[1/2].

    use crate::cell::{Cell, Sigma};
    use crate::scalar::{Dyadic, Halve, Ring};

    const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

    fn any_dyadic() -> Dyadic {
        let m: i8 = kani::any();
        let e: u32 = kani::any();
        kani::assume(e <= 2);
        Dyadic::new(i128::from(m), e)
    }

    fn any_cell() -> Cell<Dyadic> {
        Cell::new(any_dyadic(), any_dyadic())
    }

    /// `Trichotomy.Basic.one_mul`, on the dyadic path.
    ///
    /// Kept over `Dyadic` where the cheaper cell laws are over `i64`, because the unit is the one
    /// place a scale bug would hide: `Dyadic::ONE` is `1 / 2^0`, and a normalisation that lost the
    /// scale would still pass every whole-number test.
    #[kani::proof]
    fn one_is_the_unit_over_the_dyadics() {
        let z = any_cell();
        for sigma in FACES {
            assert_eq!(Cell::<Dyadic>::one().mul(sigma, z), z);
        }
    }

    /// The scalar first: `Dyadic` is a commutative ring.
    ///
    /// Two symbolic values rather than four, which is the difference between deciding and not.
    /// Every `Cell` law is this law lifted -- `mul_comm` on cells is `mul_comm` on the scalar
    /// applied componentwise -- so proving the scalar and then the lift separately is cheaper than
    /// proving the composite, and says the same thing.
    #[kani::proof]
    fn the_dyadics_are_a_commutative_ring() {
        let (a, b) = (any_dyadic(), any_dyadic());

        assert_eq!(a.add(b), b.add(a), "addition commutes");
        assert_eq!(a.mul(b), b.mul(a), "multiplication commutes");
        assert_eq!(a.add(Dyadic::ZERO), a, "zero is the additive identity");
        assert_eq!(a.mul(Dyadic::ONE), a, "one is the multiplicative identity");
        assert_eq!(a.mul(Dyadic::ZERO), Dyadic::ZERO, "zero absorbs");
        assert_eq!(
            a.sub(a),
            Dyadic::ZERO,
            "subtraction is the inverse of addition"
        );
        assert_eq!(a.add(b).sub(b), a);
    }

    /// Halving is exact and invertible, which is what makes `(1 +/- j)/2` representable at all.
    #[kani::proof]
    fn halving_is_exact_over_the_dyadics() {
        let a = any_dyadic();
        let h = a.halve();
        assert_eq!(h.add(h), a, "a half doubles back exactly");
    }

    // The identities that make this split-complex rather than merely two-dimensional. Each names
    // the theorem in `sanguine/proof/Apeiron/Trichotomy/` it mirrors.

    /// `SplitComplex.j_sq`: the generator squares to one, which is what `sigma = +1` means.
    /// `Idempotents`: `e+` and `e-` are idempotent, orthogonal, and sum to the unit.
    /// `SplitComplex.nullPair_mul_eq_zero`: the null directions annihilate.
    ///
    /// Concrete values, so this decides instantly -- and it is the part that would still be true
    /// of an implementation that had the constants wrong everywhere else.
    #[kani::proof]
    fn the_split_complex_identities_hold() {
        let s = Sigma::Hyperbolic;
        let j = Cell::<Dyadic>::new(Dyadic::ZERO, Dyadic::ONE);
        assert_eq!(j.mul(s, j), Cell::one(), "j^2 = 1");

        let (p, m) = (Cell::<Dyadic>::idem_plus(), Cell::<Dyadic>::idem_minus());
        assert_eq!(p.mul(s, p), p, "e+ is idempotent");
        assert_eq!(m.mul(s, m), m, "e- is idempotent");
        assert_eq!(p.mul(s, m), Cell::zero(), "orthogonal");
        assert_eq!(p.add(m), Cell::one(), "they sum to the unit");
        assert_ne!(p, Cell::zero());
        assert_ne!(p, Cell::one());

        // The null cone: each null direction is degenerate, and the pair annihilates.
        let (np, nm) = (Cell::<Dyadic>::of(1, 1), Cell::<Dyadic>::of(1, -1));
        assert_eq!(np.mul(s, nm), Cell::zero());
        assert_eq!(np.norm_form(s), Dyadic::ZERO);
        assert_eq!(nm.norm_form(s), Dyadic::ZERO);

        // And in the null basis the idempotents are the unit vectors -- the halves are an artefact
        // of the coordinates, not of the algebra.
        assert_eq!(p.upper(s), Some(Dyadic::ONE));
        assert_eq!(p.lower(s), Some(Dyadic::ZERO));
        assert_eq!(m.upper(s), Some(Dyadic::ZERO));
        assert_eq!(m.lower(s), Some(Dyadic::ONE));
    }

    /// `Idempotents.idempotent_iff_trivial_of_disc_nonpos`: the SAME cell is idempotent above zero
    /// and is not at or below it. The dial is the only thing that changed.
    #[kani::proof]
    fn the_idempotent_exists_only_above_zero() {
        let e = Cell::<Dyadic>::idem_plus();
        for sigma in FACES {
            assert_eq!(e.mul(sigma, e) == e, sigma.discards());
        }
    }

    /// `NullBasis.normForm_eq_upper_mul_lower`: the form factors into the channels, so it vanishes
    /// exactly when a lane dies. That is what makes "degenerate" mean "a channel is gone".
    #[kani::proof]
    fn the_norm_factors_into_the_channels() {
        let s = Sigma::Hyperbolic;
        let w: [i8; 2] = kani::any();
        let z = Cell::<Dyadic>::of(i64::from(w[0]), i64::from(w[1]));
        assert_eq!(z.norm_form(s), z.upper(s).unwrap().mul(z.lower(s).unwrap()));
    }

    // -- the map's address bijection and its probe geometry --------------------------------------

    use crate::map::{limit, region_for, Held, Salt, Stored};

    /// The elliptic face reads its key back off the address, so the bijection has to be exact --
    /// otherwise the face is not elliptic, it is a lossy fold pretending to be one and every key it
    /// hands back is a guess. The bijection is `x -> ((x*M) ^ ((x*M) >> 48)) * PSI`, so it is
    /// exact when its three steps are, and the steps are proved apart because the whole is not
    /// decidable here.
    ///
    /// **The whole is not decidable here, and that is a tooling boundary rather than a discovery.**
    /// `unspread(spread(x)) == x` asks CBMC to show `(x * M) * M_inv == x` over all `x` twice
    /// over, which is full 64-bit multiplier reasoning; it did not return in four minutes, the
    /// same wall this crate hit before on `a_ring_multiplication_is_commutative`. What follows
    /// instead is the steps, each of which decides in under a second, plus `map::tests` running
    /// the composite over twenty thousand peppers and their inverses, and
    /// `the_address_bijection_round_trips` running it over the edges.
    ///
    /// The xor-fold step: `x ^ (x >> 48)` is an involution -- the shift is at least half the
    /// width. Pure shifts and xors, no multiply.
    #[kani::proof]
    fn the_xor_fold_is_its_own_inverse() {
        let x: u64 = kani::any();
        let folded = x ^ (x >> 48);
        assert_eq!(folded ^ (folded >> 48), x);
    }

    /// The multiply steps, at the only constants the pepper-free path uses. Concrete, so each
    /// multiplier is a computation rather than a relation, and it decides at once.
    #[kani::proof]
    fn the_default_multiplier_has_the_inverse_it_claims() {
        assert_eq!(crate::coord::PHI.wrapping_mul(crate::coord::PHI_INV), 1);
        assert_eq!(
            crate::coord::inverse_odd(crate::coord::PHI),
            crate::coord::PHI_INV
        );
        assert_eq!(crate::coord::PSI.wrapping_mul(crate::coord::PSI_INV), 1);
        assert_eq!(
            crate::coord::inverse_odd(crate::coord::PSI),
            crate::coord::PSI_INV
        );
    }

    /// The fill limit is the probe's termination precondition: a region with no empty slot is one
    /// the probe walks forever. It has been wrong once already, in `table::region_slots_for`, where
    /// `next_power_of_two` could land exactly on the load.
    #[kani::proof]
    fn a_filled_region_always_keeps_an_empty_slot() {
        let slots: usize = kani::any();
        kani::assume(slots >= 16 && slots <= 1 << 40 && slots.is_power_of_two());
        assert!(limit(slots) < slots);
        assert!(limit(slots) >= slots / 2);
    }

    /// The sizing rule, in both directions.
    ///
    /// `shrink_to` is the first caller that asks for a region *smaller* than the current one, and a
    /// rule that is too generous when growing is caught by the next insert while one that is too
    /// mean when shrinking is not -- it rehomes into a region that cannot hold what it was given.
    /// So the three things it must be are proved rather than tested: a power of two (the group
    /// index is a mask), at least the minimum (a probe reads sixteen tag bytes), and big enough
    /// (`limit` must clear `n`).
    #[kani::proof]
    fn a_region_is_a_power_of_two_that_actually_holds_what_it_was_sized_for() {
        let n: usize = kani::any();
        kani::assume(n <= 1 << 20);
        let slots = region_for(n);
        assert!(slots.is_power_of_two());
        assert!(slots >= 16);
        assert!(limit(slots) >= n);
        // And minimal: one size down does not fit, unless it is already the floor.
        assert!(slots == 16 || limit(slots / 2) < n);
    }

    /// A tag must never be mistaken for an empty slot or a tombstone, or the probe stops at a live
    /// entry -- or reads one that is not there.
    ///
    /// **This harness was stale for four commits and nothing said so.** The sentinel flip inverted
    /// the polarity -- `EMPTY` went from `0` to `0xFF`, `TOMB` from `1` to `0xFE`, and a live tag
    /// from top-bit-set to top-bit-clear -- and this file is `#[cfg(kani)]`, so it does not compile
    /// under `cargo build`, `cargo test` or `cargo clippy`. Three green gates and a full `miri` run
    /// said nothing; only `cargo kani` did, and it had not been run since. The lesson is the one
    /// `table::scan`'s rename already taught in this file: a proof behind a `cfg` is only as
    /// current as the last time its tool ran.
    ///
    /// The constants are named rather than written out, so the next flip breaks the build here
    /// instead of passing a wrong assertion.
    #[kani::proof]
    fn a_live_tag_is_never_empty_and_never_a_tombstone() {
        let p: u64 = kani::any();
        let t = crate::table::tag_of(p);
        assert!(
            t & crate::map::FREE == 0,
            "a live tag has the free bit clear"
        );
        assert!(t != crate::map::EMPTY && t != crate::map::TOMB);
    }

    // -- the fold's guard, as laws ----------------------------------------------------------------

    /// **The shipped guard is symmetric on a class of size `2^32`.**
    ///
    /// `mix2(a, b)` is `(hi ^ lo) ^ g(a, b)` with `g(a, b) = a + rotl(b, 32)`. The product fold is
    /// symmetric, so `mix2(u, v) == mix2(v, u)` exactly when `g(u, v) == g(v, u)`, which is exactly
    /// when `f(u) == f(v)` for `f(x) = x - rotl(x, 32)`.
    ///
    /// Write `x = h * 2^32 + l`. Then
    ///
    /// ```text
    /// f(x) = (h - l) * 2^32 + (l - h) = (h - l) * (2^32 - 1)   (mod 2^64)
    /// ```
    ///
    /// and `2^32 - 1` is **odd**, hence invertible on `Z/2^64`. So `f` collapses every word with
    /// the same half difference `h - l` onto one value: fibers of size `2^32`, not a coincidence
    /// between two unlucky constants.
    ///
    /// That is a **constructible collision** in `fold_long`, whose round is
    /// `mix2(x_acc + a0, NU + a1)`: a pepper-zero attacker knows `x_acc`, picks `a0` freely, and
    /// solves for the `a1` that makes the two operands a swapped pair. `the_guard_is_not_symmetric`
    /// checks a handful of values and passes, which is why this went unnoticed.
    #[kani::proof]
    fn the_shipped_guard_swaps_whenever_the_half_differences_agree() {
        let x: u64 = kani::any();
        let y: u64 = kani::any();
        let g = |a: u64, b: u64| a.wrapping_add(b.rotate_left(32));
        let f = |v: u64| v.wrapping_sub(v.rotate_left(32));
        // Symmetry of the guard is exactly equality of `f`, for every pair.
        assert_eq!(g(x, y) == g(y, x), f(x) == f(y));
    }

    /// The fiber of that symmetry, exhibited: **every word has a swap partner**, so the attacker
    /// never has to search.
    ///
    /// Incrementing *both* halves leaves `h - l` fixed and therefore leaves `f` fixed. The carry is
    /// the only thing to be careful of, and it is excluded rather than glossed: at `l == u32::MAX`
    /// the increment crosses into the high half and is not the map being described. Kani found that
    /// case before this comment existed.
    #[kani::proof]
    fn every_word_has_a_swap_partner_under_the_shipped_guard() {
        let h: u32 = kani::any();
        let l: u32 = kani::any();
        kani::assume(h < u32::MAX && l < u32::MAX);
        let g = |a: u64, b: u64| a.wrapping_add(b.rotate_left(32));
        let f = |v: u64| v.wrapping_sub(v.rotate_left(32));
        let x = (u64::from(h) << 32) | u64::from(l);
        let y = (u64::from(h + 1) << 32) | u64::from(l + 1);
        assert_ne!(x, y);
        assert_eq!(f(x), f(y), "equal half differences must give equal `f`");
        assert!(g(x, y) == g(y, x), "so the pair swaps");
    }

    /// **The `a + 3b` guard swaps on exactly two points, and keeps everything else.**
    ///
    /// For a linear guard `g(a, b) = alpha * a + beta * b` on `Z/2^64` the three properties the
    /// fold wants are each a parity condition, and they are **not** in tension the way the
    /// measured `MomentChi2` failure first suggested:
    ///
    /// | property | condition |
    /// |---|---|
    /// | injective in each argument separately | `alpha` and `beta` both odd |
    /// | `g(x,y) == g(y,x)` only on a small set | `ker(alpha - beta)` small |
    /// | the diagonal `t -> g(t+s1, t+s2)` is **not** injective | `alpha + beta` even |
    ///
    /// Two odds sum to an even, so per-argument injectivity *implies* the diagonal collapse that
    /// `MomentChi2`'s derivative needs -- they are the same condition, not opposed ones. What they
    /// cost is a swap kernel of size `2^v2(alpha-beta)`, which for `(1, 3)` is **two**: `y == x` and
    /// `y == x + 2^63`. Against the shipped guard's `2^32` that is not a regression, it is thirty-one
    /// bits of improvement. `(1, -1)` is the pair the fold ships: its diagonal is the *constant*
    /// `s1 - s2`, the strongest form of the collapse, and `a - b` is a single `sub` where
    /// `a + rotl(b, 32)` was a `rol` and an `add`.
    #[kani::proof]
    fn the_odd_pair_guard_swaps_only_on_a_two_element_coset() {
        let x: u64 = kani::any();
        let y: u64 = kani::any();
        let g = |a: u64, b: u64| a.wrapping_sub(b);
        let d = y.wrapping_sub(x);
        assert_eq!(g(x, y) == g(y, x), d == 0 || d == 1 << 63);
    }

    /// **`a + 3b` is injective in each argument**, which is `mix2`'s stated contract.
    ///
    /// Both coefficients are odd and an odd multiple is a bijection on `Z/2^64`. Stated as the
    /// trivial kernel rather than pairwise, for the reason
    /// `the_elliptic_address_has_a_trivial_kernel` records: the pairwise form asks CBMC to
    /// bit-blast two multiplies against each other and it does not return.
    #[kani::proof]
    fn the_odd_pair_guard_is_injective_in_each_argument() {
        let d: u64 = kani::any();
        kani::assume(d != 0);
        // Second argument: the coefficient is `-1`, odd, and negation is a bijection.
        assert_ne!(0u64.wrapping_sub(d), 0, "negation must be injective");
        // First argument: the coefficient is 1.
        assert_ne!(d, 0);
    }

    /// **The diagonal collapses, which is what the moment test needs.**
    ///
    /// At four and eight bytes the fold reads the same bytes twice, so both `mix2` operands are
    /// `t + seed` and the guard restricted to that line is `alpha*(t+s1) + beta*(t+s2)`, i.e.
    /// `(alpha+beta) * t + k`. With both coefficients odd the multiplier is even and the line is
    /// four-to-one: the guard cannot carry the input faithfully, which is exactly the property
    /// every variant that passed `MomentChi2` shares and every variant that failed lacks.
    #[kani::proof]
    fn the_odd_pair_guard_loses_the_diagonal() {
        let t: u64 = kani::any();
        let s1: u64 = kani::any();
        let s2: u64 = kani::any();
        let phi = |v: u64| v.wrapping_add(s1).wrapping_sub(v.wrapping_add(s2));
        // `alpha + beta == 0`, so the diagonal is the **constant** `s1 - s2`: it carries nothing
        // about the input at all, which is the strongest form of the property the moment needs.
        let other: u64 = kani::any();
        assert_eq!(
            phi(t),
            phi(other),
            "the diagonal must not depend on the input"
        );
        assert_eq!(phi(t), s1.wrapping_sub(s2));
    }

    /// **A zeroed operand still moves the output.** The guard's whole job.
    ///
    /// The product fold `hi ^ lo` is zero whenever either operand is, so the guard is all that is
    /// left. `0 + 3b` is `3b`, zero only at `b == 0`; `a + 0` is `a`.
    #[kani::proof]
    fn the_odd_pair_guard_survives_an_annihilated_product() {
        let v: u64 = kani::any();
        kani::assume(v != 0);
        let g = |a: u64, b: u64| a.wrapping_sub(b);
        assert_ne!(g(0, v), 0, "a zero first operand must not zero the guard");
        assert_ne!(g(v, 0), 0, "a zero second operand must not zero the guard");
    }

    // -- the unaligned first group ----------------------------------------------------------------

    /// **An unaligned first window reads sixteen bytes that exist.**
    ///
    /// `Geom::first` returns a byte offset into `tags` and `Map::line` reads `base..base + GROUP`
    /// with no wraparound. Today the offset is a multiple of `GROUP` and at most `gmask`, so the
    /// last window starts exactly `GROUP` bytes from the end and the read is in bounds by
    /// construction.
    ///
    /// Letting the window start at *any* slot is worth about 3.4 cycles of the 150k-400k gap
    /// (`examples/probe_cost.rs`), and it breaks that argument: an offset of `slots - 1` would read
    /// fifteen bytes past the array. Two designs fix it -- `hashbrown` duplicates its first fifteen
    /// control bytes after the end, or the offset is **clamped** to `slots - GROUP`. Clamping keeps
    /// `tags.len() == slots`, so no iterator, `retain`, `Drain` or `stats` walk has to learn about
    /// a tail; this proves the clamp is what it claims.
    #[kani::proof]
    fn a_clamped_unaligned_window_stays_inside_the_region() {
        let p: u64 = kani::any();
        let gbits: u32 = kani::any();
        kani::assume(gbits >= 1 && gbits <= 16);
        let slots = (1usize << gbits) * crate::table::GROUP;
        let smask = slots - 1;
        let slast = slots - crate::table::GROUP;
        let raw = (p >> (crate::table::DESCENT_BITS - 4 - gbits)) as usize & smask;
        let base = if raw < slast { raw } else { slast };
        assert!(base + crate::table::GROUP <= slots, "the window must fit");
        // Clamping only ever moves a window backwards, so it never skips the slot it was aiming at.
        assert!(base <= raw);
        assert!(raw < base + crate::table::GROUP || base == slast);
    }

    /// **The unaligned window reads the aligned window's bits, plus offset bits below them.**
    ///
    /// This is the obligation `Geom::first`'s doc exists to enforce, and it is easy to fail by
    /// accident -- the first benchmark of this change shifted by the same amount and dropped the
    /// `* GROUP`, which reads a window **four bits lower** rather than the same window unaligned.
    /// A product's low bits are its least mixed, which is why the low-bit window failed the
    /// flatness gate; the fix has to keep the top bits and add offset *below* them.
    ///
    /// Stated as: the aligned group index is the unaligned slot index divided by `GROUP`.
    #[kani::proof]
    fn the_unaligned_window_is_the_aligned_one_plus_an_offset() {
        let p: u64 = kani::any();
        let gbits: u32 = kani::any();
        kani::assume(gbits >= 1 && gbits <= 16);
        let groups = 1usize << gbits;
        let slots = groups * crate::table::GROUP;
        // The unaligned form: four bits further down, masked to a slot.
        let unaligned = (p >> (crate::table::DESCENT_BITS - 4 - gbits)) as usize & (slots - 1);
        // The aligned form, as it is today.
        let aligned = (p >> (crate::table::DESCENT_BITS - gbits)) as usize & (groups - 1);
        assert_eq!(
            unaligned / crate::table::GROUP,
            aligned,
            "the unaligned window must sit inside the group the aligned one names"
        );
    }

    /// **Growth is a dyadic refinement, and that is why a rehome re-reads instead of re-hashes.**
    ///
    /// Read the placement word as a point in the 2-adic interval `0.b56 b55 b54 ...`. A region of
    /// `2^d` groups is the dyadic partition of that interval at depth `d`, and a key's home is the
    /// interval containing its point:
    ///
    /// ```text
    /// group_d(p)     = p[57 - d .. 57)
    /// group_{d+1}(p) = p[56 - d .. 57)  =  2 * group_d(p) + p[56 - d]
    /// ```
    ///
    /// So doubling the region **splits every group in two** and each entry follows one bit it
    /// already carries. Nothing is recomputed from the key, which is the whole reason `rehome`
    /// costs a probe per entry and not a hash per entry -- and it is why `Map<String, V>` grows
    /// 83.9 instructions a key against `hashbrown`'s 101.0, where the key must be re-folded.
    ///
    /// It is also a **locality** claim, and the opposite of `hashbrown`'s. Its index is
    /// `hash & (n - 1)`, the low bits, so doubling sends `i` to `i` or `i + n` -- two destinations
    /// half a table apart. Ours are `2g` and `2g + 1`, adjacent, so a `rehome` that walks the old
    /// groups in ascending order writes the new region in ascending order too. `rehome` does walk
    /// them that way, so the crate already collects this; the law was simply never written down,
    /// which is why nothing downstream could depend on it.
    ///
    /// Stated over the *computed* window `group_at` rather than the carried one, because
    /// `the_carried_descent_is_the_computed_one` already ties the two together.
    #[kani::proof]
    fn growth_splits_each_group_in_two() {
        let p: u64 = kani::any();
        let gbits: u32 = kani::any();
        // A region of `2^gbits` groups doubling to `2^(gbits+1)`. The upper bound keeps both
        // masks inside `usize` on every target and covers every region this crate can allocate.
        kani::assume(gbits >= 1 && gbits <= 20);
        let here = crate::table::group_at(p, 0, gbits, (1usize << gbits) - 1);
        let there = crate::table::group_at(p, 0, gbits + 1, (1usize << (gbits + 1)) - 1);
        // The one bit the entry follows: the next one down the strand.
        let bit = (p >> (crate::table::DESCENT_BITS - gbits - 1)) & 1;
        assert_eq!(
            there,
            2 * here + bit as usize,
            "growth is not a refinement of the partition"
        );
        // And a refinement is exactly a two-to-one map back: both halves return to `here`.
        assert_eq!(
            there / 2,
            here,
            "a split group does not return to the group it split from"
        );
    }

    /// The descent must stay inside the region at every step, for every address and every size.
    /// Leaving it is the one thing `map::line`'s unchecked tag load cannot survive.
    #[kani::proof]
    fn the_descent_never_leaves_the_region() {
        let p: u64 = kani::any();
        let gbits: u32 = kani::any();
        let j: u32 = kani::any();
        kani::assume(gbits >= 1 && gbits <= 16);
        kani::assume(j < 64);
        let groups = 1usize << gbits;
        assert!(crate::table::group_at(p, j, gbits, groups - 1) < groups);
    }

    // -- the two keyed faces, as laws rather than as prose ----------------------------------------

    /// **The elliptic face's safety property.** Distinct keys get distinct identity words.
    ///
    /// `Regen::holds` compares the identity word and *nothing else* -- that is what "the address is
    /// the key" means and why the face stores no key. If this fails, `Words` does not merely slow
    /// down: it returns another key's value, silently, for one pair in the corpus.
    ///
    /// Stated as a **trivial kernel** rather than as `id(a) != id(b)`, which is the same theorem
    /// and is not decidable here. `id` is `x * mul`, so `id(a) - id(b) = id(a - b)`, and injectivity
    /// is exactly "nothing but zero maps to zero". Asking CBMC for the pairwise form makes it
    /// bit-blast two 64-bit multiplies against each other and it does not return -- measured, 400
    /// seconds without a verdict, the same wall this crate hit on the old
    /// `the_address_bijection_round_trips`. The kernel form decides at once, and it is the better
    /// statement to have written down anyway: it names *why* the face is exact rather than checking
    /// that it happens to be.
    #[kani::proof]
    fn the_elliptic_address_has_a_trivial_kernel() {
        let d: u64 = kani::any();
        kani::assume(d != 0);
        let s = Salt::new(0);
        assert_ne!(
            s.spread(d).id,
            0,
            "a nonzero key maps to the same word as zero"
        );
        assert_eq!(
            s.spread(0).id,
            0,
            "the map is not linear, so the kernel argument does not apply"
        );
    }

    /// The same under an arbitrary pepper. Peppering changes the multiplier, and if it could
    /// produce an even one the elliptic face would stop being exact -- two keys would share an
    /// identity word and the face compares nothing else. `Salt::new` forces the low bit; this is
    /// the consequence, checked over every pepper rather than the few a test can reach.
    #[kani::proof]
    fn a_peppered_elliptic_address_still_has_a_trivial_kernel() {
        let pepper: u64 = kani::any();
        let d: u64 = kani::any();
        kani::assume(d != 0);
        let s = Salt::new(pepper);
        assert_ne!(s.spread(d).id, 0);
        assert_eq!(s.spread(0).id, 0);
    }

    /// It runs backwards too -- that is how the face regenerates a key it never stored -- and the
    /// composite `unspread(spread(x).id) == x` is **not decidable here**: two constant 64-bit
    /// multiplies, 400 seconds without a verdict. It is proved in pieces instead, which is this
    /// crate's standing rule for this checker.
    /// `the_default_multiplier_has_the_inverse_it_claims` pins `mul * inv == 1` concretely, the
    /// kernel harnesses above pin injectivity, and
    /// `map::tests::every_pepper_gives_an_invertible_multiplier` runs the composite over twenty
    /// thousand peppers and their inverses. The mathematics joining those is one line and does not
    /// need a SAT solver.

    /// **The placement side is a discard, and that is why identity is a separate question.**
    ///
    /// The first attempt at this asked whether the *placement word* is non-injective, by looking
    /// for two keys that share one. That is undecidable here for the same reason the round trip is
    /// -- pairwise 64-bit multiplies -- and it was the wrong question anyway. Nothing in the design
    /// needs the placement word to collide; what it needs is that the thing the descent actually
    /// reads, the **group index**, cannot separate keys. That is pigeonhole and needs no search:
    /// `gbits` bits cannot distinguish `2^64` addresses, and here is a pair that proves it for
    /// every table size at once.
    ///
    /// So a table cannot answer "is this the same key?" from where the key landed, on any face.
    /// The elliptic face answers it from the identity word (proved injective above), the parabolic
    /// face from the key itself, and the hyperbolic face declines to ask. Those are the three
    /// options and there is no fourth.
    #[kani::proof]
    fn the_group_index_cannot_separate_keys() {
        let p: u64 = kani::any();
        let gbits: u32 = kani::any();
        kani::assume(gbits >= 1 && gbits <= 16);
        let gmask = (1usize << gbits) - 1;

        // Two distinct addresses differing only below the descent's first window. The window reads
        // bits [DESCENT_BITS - gbits, DESCENT_BITS), so flipping bit 0 cannot move it.
        let q = p ^ 1;
        assert_ne!(p, q);
        assert_eq!(
            crate::table::group_at(p, 0, gbits, gmask),
            crate::table::group_at(q, 0, gbits, gmask),
            "the first window separated two addresses it cannot see apart"
        );
        // And the tag is no help either: it reads the top seven bits, which bit 0 also does not
        // reach. Placement carries no identity at all.
        assert_eq!(crate::table::tag_of(p), crate::table::tag_of(q));
    }

    /// **The parabolic face is exact whatever its fold does.**
    ///
    /// This is the law that licenses `Coord::SPREAD`. On this face the identity word is an
    /// early-reject hint and the *key* is the decision, so a fold with fibers costs probe length
    /// and never correctness. The elliptic face has no such licence, which is why `Regen::address`
    /// does not consult `SPREAD`.
    #[kani::proof]
    fn the_parabolic_face_decides_on_the_key() {
        let stored: u32 = kani::any();
        let asked: u32 = kani::any();
        let id: u64 = kani::any();
        let held = <Stored<u32> as Held<u32>>::hold(id, stored);
        // Whatever the identity words say, `holds` is true only when the keys are equal.
        assert_eq!(held.holds(id, &asked), stored == asked);
    }

    // -- the trichotomy, as a law ------------------------------------------------------------------

    /// **Three faces, three genuinely different storage policies.**
    ///
    /// This harness used to prove that a `Keystore` enum and [`Sigma`] were in bijection, via a
    /// `Keystore::of` and a `Keystore::sigma` written as each other's inverse. That decided
    /// nothing: it compared a relabelling against the thing it relabelled, and it could not fail.
    /// The enum is gone, the face is the discriminant, and the claim worth deciding is that the
    /// policy actually varies -- that no two faces could share a storage implementation, so a
    /// fourth would need a fourth algebra, which `classification` says does not exist.
    #[kani::proof]
    fn the_three_faces_license_three_different_policies() {
        let policy = |s: Sigma| (s.stores_the_key(), s.key_survives());
        let (a, b, c) = (
            policy(Sigma::Elliptic),
            policy(Sigma::Parabolic),
            policy(Sigma::Hyperbolic),
        );
        assert!(a != b);
        assert!(b != c);
        assert!(a != c);
    }

    /// What each face does with a key follows from its discriminant and from nothing else.
    #[kani::proof]
    fn only_the_middle_face_carries_a_key() {
        for sigma in FACES {
            // Parabolic, and only parabolic, has to carry the key beside the address.
            assert_eq!(sigma.stores_the_key(), matches!(sigma, Sigma::Parabolic));
            // Hyperbolic, and only hyperbolic, destroys it -- which is `discards_iff_disc_pos`.
            assert_eq!(sigma.key_survives(), !sigma.discards());
        }
    }

    /// **Liveness.** The probe cannot spin: past the point where the word is spent the descent
    /// walks one group at a time, so within `groups` further steps it has visited every group --
    /// and `a_filled_region_always_keeps_an_empty_slot` says one of them holds an empty slot.
    ///
    /// Stated as coverage rather than as "it advances", because advancing is what
    /// `the_spent_descent_advances_by_exactly_one_group` proves and termination is what a table fed
    /// hostile keys actually needs.
    #[kani::proof]
    fn the_descent_visits_every_group() {
        let p: u64 = kani::any();
        let gbits: u32 = kani::any();
        kani::assume(gbits >= 1 && gbits <= 5);
        let groups = 1usize << gbits;
        let gmask = groups - 1;
        let full = crate::table::DESCENT_BITS / gbits;

        // Any group we care to name.
        let target: usize = kani::any();
        kani::assume(target <= gmask);

        // The sweep starts somewhere and advances by one each step, so the offset that reaches
        // `target` is inside one lap.
        let start = crate::table::group_at(p, full, gbits, gmask);
        let lap = (target + groups - start) & gmask;
        assert!(lap < groups);
        assert_eq!(
            crate::table::group_at(p, full + lap as u32, gbits, gmask),
            target
        );
    }

    /// `Geom` never stores a shift amount of 64.
    ///
    /// `groups.trailing_zeros()` is 64 for an empty region, and `Descent::after_first` hands
    /// `gbits < 64` to LLVM so `wrapping_shl`'s `% 64` stops emitting an `and $0x3f` that gets
    /// hoisted into a stack slot live across the whole probe. That `assert_unchecked` is only
    /// sound because `Geom::of` maps the empty region onto `gbits = 0` instead, which every
    /// consumer already treats as "one group, nothing to choose".
    #[kani::proof]
    fn a_geometry_never_asks_for_a_shift_of_sixty_four() {
        let groups: usize = kani::any();
        let gbits: u32 = if groups == 0 {
            0
        } else {
            groups.trailing_zeros()
        };
        assert!(gbits < 64, "a shift amount must be a shift amount");
        // And the empty region lands on the sentinel rather than on a distinguishable value:
        // `wrapping_shl(0)` and `wrapping_shl(64)` are both the identity, which is what makes
        // folding the two cases together unobservable.
        let p: u64 = kani::any();
        if groups == 0 {
            assert_eq!(p.wrapping_shl(gbits), p.wrapping_shl(64));
        }
    }

    // -- why the group window is taken from the top of the placement word ------------------------
    //
    // `Geom::first`'s doc argues that a low window fails because "a multiply carries entropy
    // upward and never downward". That was prose with a measurement attached. These five decide
    // it as arithmetic.

    /// A key below `2^32` never reaches the top quarter of the widening product.
    ///
    /// `x < 2^32` and `PHI < 2^64` give `x * PHI < 2^96`, so bits 96..128 are zero -- which is to
    /// say the top half of `hi` is zero for every such key. Everything below rests on this.
    #[kani::proof]
    fn a_small_key_never_reaches_the_top_of_the_product() {
        let x: u32 = kani::any();
        let w = u128::from(x) * u128::from(crate::coord::PHI);
        assert_eq!(w >> 96, 0, "a key below 2^32 cannot fill 128 bits");
    }

    /// The low half of `place` **is** the low half of the identity word, for every key below
    /// `2^32`.
    ///
    /// `place = rotl(hi, 32) ^ lo`, and `rotl(hi, 32)`'s low half is `hi`'s *top* half, which
    /// `a_small_key_never_reaches_the_top_of_the_product` shows is zero. So the xor contributes
    /// nothing there and `place`'s low 32 bits are `lo`'s low 32 bits exactly.
    ///
    /// This is the collapse `Geom::first` rejects a low window over, stated without reference to
    /// any measurement: a low window on the placement word is a window on the identity word.
    #[kani::proof]
    fn the_low_half_of_place_is_the_low_half_of_the_identity_word() {
        let x: u32 = kani::any();
        let a = crate::coord::spread(u64::from(x));
        assert_eq!(
            a.place as u32, a.id as u32,
            "the low window sees only the identity word"
        );
    }

    /// A difference confined **above** bit `k` is still above bit `k` after the fold.
    ///
    /// `id = x * PHI` in `Z/2^64`. If two keys differ only above bit `k` their difference is
    /// `d << k`, and `(d << k) * PHI` is divisible by `2^k` because `d * PHI` is an integer -- so
    /// the two identity words agree on their low `k` bits. Multiplication carries information from
    /// bit `i` to bits at or above `i`, and to no bit below `i`.
    ///
    /// This is "a multiply carries entropy upward and never downward" as a theorem rather than as
    /// a slogan, and with `the_low_half_of_place_is_the_low_half_of_the_identity_word` it says a
    /// low group window separates keys only by bits on which they already differ.
    ///
    /// # Why this shape and not the two-key one
    ///
    /// The direct statement -- two symbolic keys assumed equal below `k`, their identity words
    /// asserted equal below `k` -- puts **two** symbolic 64-bit multiplies in front of CBMC and
    /// does not decide in ten minutes. The difference form has one, and says the same thing given
    /// that `id` is additive, which `the_identity_word_is_additive` is.
    #[kani::proof]
    fn a_difference_above_a_bit_stays_above_it() {
        let d: u64 = kani::any();
        let k: u32 = kani::any();
        kani::assume(k >= 1 && k <= 63);
        let m = (1u64 << k) - 1;
        assert_eq!(
            d.wrapping_shl(k).wrapping_mul(crate::coord::PHI) & m,
            0,
            "a difference above bit k cannot reach a bit below it"
        );
    }

    /// Two keys agreeing on their low `k` bits have identity words agreeing on their low `k` bits.
    ///
    /// The two-key form of `a_difference_above_a_bit_stays_above_it`, stated where the low-window
    /// argument actually lives: keys below `2^32`, which are the ones whose product does not fill
    /// 128 bits. Two symbolic 64-bit multiplies do not decide; two with 32 known-zero bits in each
    /// operand do.
    #[kani::proof]
    fn two_keys_agreeing_low_down_have_identity_words_agreeing_low_down() {
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        let k: u32 = kani::any();
        kani::assume(k >= 1 && k <= 31);
        let m = (1u64 << k) - 1;
        kani::assume(u64::from(x) & m == u64::from(y) & m);
        assert_eq!(
            crate::coord::spread(u64::from(x)).id & m,
            crate::coord::spread(u64::from(y)).id & m,
            "the identity word's low bits cannot see a key bit above them"
        );
    }

    /// A low group window separates two keys only if their low bits already differ.
    ///
    /// `Geom::first` is `(p >> gshift) & gmask` with `gmask = (groups - 1) * GROUP`, so the mask
    /// keeps bits 4..gbits+4 of the shifted word and the result is already a byte offset. The
    /// low-window form of the same expression is `(p << 4) & gmask`, which keeps bits 0..gbits of
    /// `p` scaled by `GROUP`.
    ///
    /// Two keys below `2^32` agreeing on their low `gbits` bits then land in the same group for
    /// every region size -- so an attacker who fixes `gbits` key bits fixes the group at no cost,
    /// and the descent never begins.
    ///
    /// # This harness failed first, and the design was not what was wrong
    ///
    /// Written as `(p >> 4) & gmask` against an assumption on the low `gbits + 4` bits, it is
    /// **false**, and Kani said so in 0.1s: that expression reads bits 8..gbits+8, four above what
    /// the assumption pins. The off-by-four was mine. It is the reason to write the window as the
    /// shift the code actually performs rather than as a paraphrase of it.
    #[kani::proof]
    fn a_low_group_window_cannot_separate_keys_that_agree_low_down() {
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        let gbits: u32 = kani::any();
        kani::assume(gbits >= 1 && gbits <= 20);
        let m = (1u64 << gbits) - 1;
        kani::assume(u64::from(x) & m == u64::from(y) & m);
        let gmask = ((1usize << gbits) - 1) * crate::table::GROUP;
        let low = |k: u32| (crate::coord::spread(u64::from(k)).place << 4) as usize & gmask;
        assert_eq!(
            low(x),
            low(y),
            "a low window is decided by bits the keys already share"
        );
    }

    /// The **top** window is not a function of those bits.
    ///
    /// A witness, which is what an existence claim needs: two keys agreeing on their whole low
    /// half and landing on different tags. `tag_of` reads bits 57..64 of `place`, which are
    /// `lo`'s bits 57..64 xored with `hi`'s bits 25..32 -- both of which move when a key bit above
    /// the window moves. So the asymmetry between the two ends of `place` is real and not an
    /// artifact of the corpus the flatness gate happened to run on.
    #[kani::proof]
    fn the_top_window_does_see_bits_a_low_window_cannot() {
        let x: u64 = 1;
        let y: u64 = 1 | (1u64 << 32);
        assert_eq!(
            x & 0xFFFF_FFFF,
            y & 0xFFFF_FFFF,
            "the witnesses agree on their low half"
        );
        assert_ne!(
            crate::table::tag_of(crate::coord::spread(x).place),
            crate::table::tag_of(crate::coord::spread(y).place),
            "the top window must move when a high key bit moves"
        );
    }

    /// A left rotation by 32 sends bit `(i + 32) mod 64` to bit `i`.
    ///
    /// `coord::spread`'s doc states `place_i == lo_i ^ hi_((i - 32) mod 64)` and attaches
    /// "verified bit for bit over two million keys". The multiply is not what that claim is about:
    /// `place` is `rotl(hi, 32) ^ lo` **by definition**, and xor is bitwise, so the whole content
    /// is where the rotation puts each bit. Stated without the multiply it decides instantly;
    /// stated with one it did not decide in twenty minutes, which is the crate's rule for
    /// restating rather than grinding.
    ///
    /// This is what lets the tag sit directly above the descent: disjoint slices of `place` are
    /// disjoint sets of `GF(2)` functionals of the 128 product bits, so neither reads the other's
    /// evidence.
    ///
    /// The other half -- that `spread` computes `rotl(hi, 32) ^ lo` from the two halves of the
    /// widening product -- is `spread`'s body, and a harness asserting it re-types the source. It
    /// was written, did not decide (CBMC will not fold two symbolic 128-bit multiplies), and was
    /// dropped rather than ground on, because what it would have bought is reading four lines.
    #[kani::proof]
    fn a_rotation_by_thirty_two_lands_each_bit_where_the_address_doc_says() {
        let hi: u64 = kani::any();
        let lo: u64 = kani::any();
        let i: u32 = kani::any();
        kani::assume(i < 64);
        let place = hi.rotate_left(32) ^ lo;
        assert_eq!(
            (place >> i) & 1,
            ((lo >> i) & 1) ^ ((hi >> ((i + 32) % 64)) & 1),
            "bit i of place is bit i of lo xored with bit (i - 32) of hi"
        );
    }

    /// `gbits` is recoverable from `gmask`, so storing both is storing one thing twice.
    ///
    /// `gmask` is `(groups - 1) * GROUP` for a power-of-two `groups`, so `mask() + 1` is `groups`
    /// and its `trailing_zeros` is `gbits`. The empty region has `gmask == 0`, giving 0, which is
    /// the sentinel `Geom::of` folds it onto.
    ///
    /// Removing the field was **tried and reverted**: LLVM hoists the recomputation into the same
    /// loop preheader the load was in, so the hot path keeps the live value and gains the
    /// arithmetic. The redundancy is real; removing it is not free. The theorem stands either way.
    #[kani::proof]
    fn gbits_is_recoverable_from_the_group_mask() {
        let gbits: u32 = kani::any();
        kani::assume(gbits <= 32);
        let groups = 1usize << gbits;
        let gmask = (groups - 1) * crate::table::GROUP;
        assert_eq!((gmask / crate::table::GROUP + 1).trailing_zeros(), gbits);
    }

    // -- what the upstream trichotomy does and does not survive into Z/2^64 -----------------------
    //
    // `coord.rs` names `place` hyperbolic and `id` elliptic and cites the Lean development for
    // both -- `Projection.discards_iff_disc_pos`, `Basic.noZeroDivisors_iff_disc_neg`,
    // `Idempotents.exists_nontrivial_idempotent_iff_disc_pos`. Those are theorems over the reals,
    // and this crate's arithmetic is `Z/2^64`. The harnesses above check them on `Cell<Dyadic>`,
    // which has halves and is the real path. `Cell<i64>` -- `Ring for i64` is `wrapping_*`, so it
    // **is** `Z/2^64` -- is the path the address actually takes, and these say what happens there.
    //
    // Two of the three characterisations do not transfer, and they fail for the same reason from
    // opposite directions: over the reals `b * (2a - 1) = 0` is split by having no zero divisors,
    // and over `Z/2^64` it is split by `2a - 1` being a unit. The conclusion is *stronger* here
    // and the trichotomy is *gone*, because the property it classified by is now constant in the
    // dial.

    /// Two is not invertible in `Z/2^64`, so the point the real idempotents sit at is not there.
    ///
    /// `Idempotents.exists_nontrivial_idempotent_iff_disc_pos` produces its witnesses at
    /// `(1/2, +/- 1/(2 sqrt(sigma)))`. The `a = 1/2` branch of that proof is the entire source of
    /// the nontrivial idempotents, and `2h = 1` has no solution here: `2h` is even and `1` is not.
    ///
    /// This is `Halve` as a theorem rather than as a trait bound. The bound already says a scalar
    /// without halves cannot build `idem_plus`; this says the object it would have built is absent
    /// and not merely unconstructible.
    #[kani::proof]
    fn the_half_the_real_idempotents_need_is_not_in_the_word_ring() {
        let h: u64 = kani::any();
        assert_ne!(h.wrapping_mul(2), 1, "two has no inverse in Z/2^64");
    }

    /// A word times an **odd** word is zero only when the word is.
    ///
    /// `Z/2^64` has zero divisors, so `mul_eq_zero` -- the step the Lean proof of
    /// `exists_nontrivial_idempotent_iff_disc_pos` takes over the reals -- is unavailable. What
    /// replaces it is narrower and enough: an odd factor is a unit, so it can be cancelled even
    /// though the ring is not a domain.
    #[kani::proof]
    fn an_odd_factor_can_be_cancelled_in_the_word_ring() {
        let b: u64 = kani::any();
        let u: u64 = kani::any();
        kani::assume(u & 1 == 1);
        kani::assume(b.wrapping_mul(u) == 0);
        assert_eq!(b, 0, "an odd multiplier is a unit, so it cannot annihilate");
    }

    /// **The trichotomy's idempotent characterisation does not transfer.** Over `Z/2^64` the only
    /// idempotents are `0` and `1`, at *every* dial -- elliptic, parabolic and hyperbolic alike.
    ///
    /// `(a + bx)^2 = (a^2 + sigma b^2) + 2ab x`, so idempotence is `2ab = b` and
    /// `a^2 + sigma b^2 = a`. The first is `b (2a - 1) = 0`, and `2a - 1` is odd, so
    /// `an_odd_factor_can_be_cancelled_in_the_word_ring` forces `b = 0` **without the dial being
    /// consulted at all**. Then `a^2 = a`, and one of `a`, `a - 1` is odd, so
    /// `the_only_word_solutions_of_a_squared_equals_a` forces `a` into `{0, 1}`.
    ///
    /// So `exists_nontrivial_idempotent_iff_disc_pos` has a left-hand side that is constantly
    /// false here. It is not that the equivalence flips; it is that the property it classifies by
    /// stops depending on the dial. Naming a `Z/2^64` object hyperbolic because it has the
    /// property that theorem characterises is naming it after something nothing here has.
    ///
    /// # Width
    ///
    /// Over `Z/2^8`, which is what decides. The argument uses exactly two facts -- `2a - 1` is odd
    /// and one of `a`, `a - 1` is odd -- and both hold in `Z/2^n` for every `n >= 1`, so it is
    /// uniform in the width in the way `Ring for i8`'s doc describes. The two cancellation steps
    /// it rests on are proved at full width beside it.
    #[kani::proof]
    fn the_word_ring_has_no_nontrivial_idempotent_at_any_dial() {
        let a: i8 = kani::any();
        let b: i8 = kani::any();
        let z = Cell::<i8>::new(a, b);
        for sigma in FACES {
            if z.mul(sigma, z) == z {
                assert!(
                    z == Cell::<i8>::zero() || z == Cell::<i8>::one(),
                    "the word ring has only the two idempotents every ring has"
                );
            }
        }
    }

    /// The scalar half of the above, at full width: `a^2 = a` only at `0` and `1`.
    #[kani::proof]
    fn the_only_word_solutions_of_a_squared_equals_a() {
        let a: u64 = kani::any();
        kani::assume(a.wrapping_mul(a) == a);
        assert!(
            a == 0 || a == 1,
            "Boole's equation has two roots in Z/2^64 as well"
        );
    }

    /// **The no-zero-divisors characterisation does not transfer either.** `Z/2^64` has a nonzero
    /// nilpotent at *every* dial, so no dial is a domain.
    ///
    /// `Basic.noZeroDivisors_iff_disc_neg` makes "has no zero divisors" the defining property of
    /// the elliptic face over the reals. Here `2^32` squares to zero in the scalar ring already,
    /// so `(2^32, 0)` is a nonzero cell whose square is zero whatever the dial is -- the dial never
    /// enters, because the second slot is empty. The elliptic face's characterising property is
    /// constantly false rather than dial-dependent.
    ///
    /// `coord::spread`'s doc reads `noZeroDivisors_iff_disc_neg` as saying the identity word
    /// "discards nothing". That conclusion is still right, and it does not come from this theorem:
    /// it comes from `PHI` being **odd**, hence a unit, which
    /// `the_elliptic_address_has_a_trivial_kernel` proves directly and without the analogy.
    #[kani::proof]
    fn the_word_ring_has_a_nilpotent_at_every_dial() {
        let z = Cell::<i64>::new(1i64 << 32, 0);
        assert_ne!(z, Cell::<i64>::zero(), "the witness is not zero");
        for sigma in FACES {
            assert_eq!(
                z.mul(sigma, z),
                Cell::<i64>::zero(),
                "and it squares to zero"
            );
        }
    }

    /// What **does** transfer: the norm form is multiplicative, at every dial.
    ///
    /// `N(a + bx) = a^2 - sigma b^2` and `N(zw) = N(z) N(w)` is a polynomial identity with integer
    /// coefficients, so it holds in every commutative ring or in none -- which is why it survives
    /// the base change that the idempotent, domain and projection characterisations do not.
    ///
    /// **The mathematical statement is proved upstream and universally**, over an arbitrary
    /// commutative ring, by `Trichotomy.AnyRing.ringNormForm_mul` in `sanguine-apeiron`. No
    /// bit-precise checker can reach that: it decides one width at a time and the theorem is about
    /// every width at once. What this harness adds is the half Lean cannot see -- that `Cell::mul`
    /// and `Cell::norm_form` as *written here* compute the thing the theorem is about. A proof of
    /// the algebra and a check of the code are two obligations, not one.
    ///
    /// # Why the operands are bounded
    ///
    /// `N(zw)` is degree **four** in the four scalars, so a bit-precise checker bit-blasts a
    /// multiply of two multiplies. Over unrestricted `i8` this did not return in twenty-five
    /// minutes, where the degree-two laws beside it decide in under a second: the width is not what
    /// costs, the nesting is. Bounded to a 4-bit window it decides. That bound is why the Lean
    /// theorem is the one to cite for the mathematics, and this is the one to cite for the code.
    #[kani::proof]
    fn the_norm_form_is_multiplicative_in_the_word_ring() {
        let (a, b, c, d): (i8, i8, i8, i8) = (kani::any(), kani::any(), kani::any(), kani::any());
        for v in [a, b, c, d] {
            kani::assume((-8..8).contains(&v));
        }
        let z = Cell::<i8>::new(a, b);
        let w = Cell::<i8>::new(c, d);
        for sigma in FACES {
            assert_eq!(
                z.mul(sigma, w).norm_form(sigma),
                z.norm_form(sigma).mul(w.norm_form(sigma)),
                "the norm is multiplicative"
            );
        }
    }

    /// **And the projection characterisation does not transfer.** Over `Z/2^n` no projection both
    /// kills a nonzero point and keeps one, at any dial.
    ///
    /// `Projection.Discards sigma` is `exists e, Idempotent sigma e and (exists z != 0, z e = 0)
    /// and (exists w != 0, w e = w)`, and `proj sigma e z` is `mul sigma z e` -- so a projection
    /// here is multiplication by an idempotent, and nothing else.
    /// `the_word_ring_has_no_nontrivial_idempotent_at_any_dial` leaves only `e = 0`, which keeps
    /// nothing, and `e = 1`, which kills nothing. `Discards` is therefore **false at every dial**,
    /// where over the reals it is exactly `0 < sigma`.
    ///
    /// # What this costs the address doc
    ///
    /// `coord::spread`'s doc says naming `place` hyperbolic "is forced by what it does", citing
    /// `Projection.discards_iff_disc_pos`. It is not forced by that theorem: the property the
    /// theorem characterises is one no dial has over `Z/2^64`, so the citation cannot single out
    /// the hyperbolic one. What is true about `place` -- that it is not injective and has no
    /// inverse -- is true, is proved elsewhere, and is not this. The name is an analogy with the
    /// real case, and the doc now says so.
    ///
    /// Non-vacuous by construction: the premise is satisfied by `e = 1` with `z = 0`, which the
    /// harness checks before assuming anything, so a passing run is the implication and not an
    /// empty hypothesis. The positive control is `the_split_complex_identities_hold`, where
    /// `Cell<Dyadic>` -- which has halves -- does exhibit exactly the pair this rules out.
    #[kani::proof]
    fn no_word_projection_both_kills_and_keeps_at_any_dial() {
        // The premise is reachable, so the implication below is not vacuous.
        let one = Cell::<i8>::one();
        assert_eq!(one.mul(Sigma::Hyperbolic, one), one);
        assert_eq!(
            Cell::<i8>::zero().mul(Sigma::Hyperbolic, one),
            Cell::<i8>::zero()
        );

        let e = Cell::<i8>::new(kani::any(), kani::any());
        let z = Cell::<i8>::new(kani::any(), kani::any());
        let w = Cell::<i8>::new(kani::any(), kani::any());
        for sigma in FACES {
            if e.mul(sigma, e) == e && z.mul(sigma, e) == Cell::<i8>::zero() && w.mul(sigma, e) == w
            {
                assert!(
                    z == Cell::<i8>::zero() || w == Cell::<i8>::zero(),
                    "a projection over Z/2^n cannot both kill a nonzero point and keep one"
                );
            }
        }
    }

    /// Is `place` injective on 32-bit keys?
    ///
    /// `coord::spread`'s doc says `place` "is not injective, by construction". Nothing proved it.
    /// A search over 2^26 dense keys and 2^26 scattered ones found zero collisions, which settles
    /// nothing in either direction -- a uniformly random function on 64 bits has about 0.0001
    /// expected collisions at that sample size, so the search cannot tell a bijection from a
    /// scrambler.
    ///
    /// Asked where it can be decided, the answer is **yes**: `place` separates every pair of keys
    /// below `2^32`, four billion of them, verified in 32s. So the doc's claim was false over the
    /// whole range a caller will ever use, and it has been withdrawn. Whether `place` is injective
    /// on the full 64-bit domain is open.
    ///
    /// Either way the crate's security argument is untouched, and that is the point of writing it
    /// down: the aiming cost is `2^(j * gbits)` trials to match `j` *windows* of `place`, which is
    /// a statement about the windows and not about whether the whole word is injective.
    /// `the_group_index_cannot_separate_keys` is where that lives.
    #[kani::proof]
    fn place_separates_keys_below_two_to_the_thirty_two() {
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        kani::assume(x != y);
        assert_ne!(
            crate::coord::spread(u64::from(x)).place,
            crate::coord::spread(u64::from(y)).place,
            "place collides on two keys below 2^32"
        );
    }

    // -- narrowing the 16,384 anomaly -------------------------------------------------------------
    //
    // `iterate` and `clone+drop` at 16,384 entries read 0.49x to 0.62x against `hashbrown` and swing
    // that far run to run, while the same map measured in isolation reads a 17% win. Until that is
    // explained no verdict from the surface harness is worth anything, so the explanations get
    // closed one at a time. These are the structural ones, which decide here rather than on a clock.

    /// The region a load is given is the smallest power of two that can hold it at the fill limit.
    ///
    /// Both arrays are scanned end to end by `iterate`, `clone` and `drain`, so their cost is per
    /// **slot** and not per entry. If this crate sized a region differently from `hashbrown` at some
    /// load, those rows would be comparing a longer scan against a shorter one and the ratio would
    /// mean nothing.
    ///
    /// It does not: measured, `capacity()` is 1792, 28672, 114688 and 1835008 for apeiron,
    /// `hashbrown` and `std` alike at 1,000, 16,384, 100,000 and 1,000,000 entries. This pins the
    /// rule that produces those numbers so the agreement cannot drift, and it settles that the
    /// 16,384 anomaly is not a difference in how much there is to scan.
    #[kani::proof]
    fn a_region_is_the_smallest_power_of_two_that_holds_its_load() {
        let n: usize = kani::any();
        // Up to a load that needs 2^20 slots. Beyond it the doubling is the same statement again.
        kani::assume(n <= (1 << 19));
        let slots = crate::map::region_for(n);
        assert!(slots.is_power_of_two(), "a region is a power of two");
        assert!(
            slots >= crate::map::MIN_SLOTS,
            "a region is never smaller than one group"
        );
        assert!(
            crate::map::limit(slots) >= n,
            "the region must hold the load at the fill limit"
        );
        assert!(
            slots == crate::map::MIN_SLOTS || crate::map::limit(slots / 2) < n,
            "and it must be the smallest such region"
        );
    }

    /// The fill limit is seven eighths, exactly, at every region size.
    ///
    /// `limit` is `slots / FILL_DEN * FILL_NUM`, an integer division before the multiply, so it is
    /// not obviously `7 * slots / 8`. On a power of two it is: the division is exact.
    #[kani::proof]
    fn the_fill_limit_is_seven_eighths_of_a_power_of_two() {
        let gbits: u32 = kani::any();
        kani::assume(gbits >= 4 && gbits <= 40);
        let slots = 1usize << gbits;
        assert_eq!(
            crate::map::limit(slots) * 8,
            slots * 7,
            "the fill limit is exactly 7/8"
        );
    }
}
