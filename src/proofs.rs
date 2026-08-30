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
            assert!(slots.is_power_of_two(), "the group index is a mask, so this must be exact");
            assert!(slots >= 32, "at least two groups, so the descent has somewhere to go");
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
        assert_eq!(a.sub(a), Dyadic::ZERO, "subtraction is the inverse of addition");
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
        assert_eq!(crate::coord::inverse_odd(crate::coord::PHI), crate::coord::PHI_INV);
        assert_eq!(crate::coord::PSI.wrapping_mul(crate::coord::PSI_INV), 1);
        assert_eq!(crate::coord::inverse_odd(crate::coord::PSI), crate::coord::PSI_INV);
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
    #[kani::proof]
    fn a_live_tag_is_never_empty_and_never_a_tombstone() {
        let p: u64 = kani::any();
        let t = crate::table::tag_of(p);
        assert!(t & 0x80 != 0);
        assert!(t != 0 && t != 1);
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
        assert_ne!(s.spread(d).id, 0, "a nonzero key maps to the same word as zero");
        assert_eq!(s.spread(0).id, 0, "the map is not linear, so the kernel argument does not apply");
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
        assert_eq!(crate::table::group_at(p, full + lap as u32, gbits, gmask), target);
    }

}
