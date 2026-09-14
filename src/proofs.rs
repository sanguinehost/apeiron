//! Bounded proofs. `cfg(kani)` only.
//!
//! Each harness below is a `#[kani::proof]` over a symbolic or bounded input, checked by
//! `cargo kani`. Three bounds are stated where the tool forces them rather than hidden:
//!
//! - **Width.** CBMC bit-blasts a multiply to the width of the type whatever the operands hold, so
//!   the structural ring laws are stated over `i8` (`scalar.rs` says why that is the structure) and
//!   the `i64` path is checked where the solver returns.
//! - **Window.** A degree-four identity is checked on the small window its harness names rather
//!   than ground on. The bound is a tractability device, not a domain restriction of the algebra.
//! - **Shards.** `Table`'s shard index is `home(key) % 256`, and a symbolic index into the
//!   256-entry `shards` array makes CBMC build a 256-way mux. The symbolic-key table coverage is
//!   split between the real routing arithmetic, the real bodies at concrete shard indices (reduced
//!   key width for the multi-key case) and a reduced-shard replica; the table section states which
//!   harness covers what and what is not covered.
//!
//! `cargo build`, `cargo test` and `cargo clippy` do not set `cfg(kani)`, so none of this is
//! compiled by them: a harness that drifts from the kernel is caught only when `cargo kani` runs.

#[cfg(kani)]
mod harnesses {
    use crate::carrier::{conservative_density_step, diffusive_flux_half};
    use crate::cell::{Cell, Sigma};
    use crate::coord::{bucket, i21, Packed};
    use crate::exchange::{accept_derived_impulse, elastic_exchange, exchange_impulse};
    use crate::scalar::{Dyadic, Halve, Ring};
    use crate::table::{Home, Merge, StorageError, Table, WordHome, WordMerge};

    const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

    // ── the algebra over the dyadics: the scalar that holds the idempotents ───────────────────

    fn any_dyadic() -> Dyadic {
        let m: i8 = kani::any();
        let e: u32 = kani::any();
        kani::assume(e <= 2);
        Dyadic::new(i128::from(m), e)
    }

    fn any_dyadic_cell() -> Cell<Dyadic> {
        Cell::new(any_dyadic(), any_dyadic())
    }

    /// `Trichotomy.Basic.one_mul`, on the dyadic path.
    ///
    /// Kept over `Dyadic` where the cheaper cell laws are over `i64`, because the unit is the one
    /// place a scale bug would hide: `Dyadic::ONE` is `1 / 2^0`, and a normalisation that lost the
    /// scale would still pass every whole-number test.
    #[kani::proof]
    fn one_is_the_unit_over_the_dyadics() {
        let z = any_dyadic_cell();
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
        assert_eq!(a.sub(a), Dyadic::ZERO, "subtraction inverts addition");
        assert_eq!(a.add(b).sub(b), a);
    }

    /// Halving is exact and invertible, which is what makes `(1 +/- j)/2` representable at all.
    #[kani::proof]
    fn halving_is_exact_over_the_dyadics() {
        let a = any_dyadic();
        let h = a.halve();
        assert_eq!(h.add(h), a, "a half doubles back exactly");
    }

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

    // ── the word path: `Cell<i64>` is `Z/2^64` under wrapping arithmetic ──────────────────────

    fn bounded_word() -> i64 {
        let v: i64 = kani::any();
        kani::assume(-8 <= v && v <= 8);
        v
    }

    fn bounded_word_cell() -> Cell<i64> {
        Cell::new(bounded_word(), bounded_word())
    }

    /// A byte in a small window, for the identities whose bit-blasted form is SAT-hard at full
    /// `i8` width.
    fn small_byte() -> i8 {
        let v: i8 = kani::any();
        kani::assume(-4 <= v && v <= 4);
        v
    }

    fn small_byte_cell() -> Cell<i8> {
        Cell::new(small_byte(), small_byte())
    }

    /// Componentwise addition is the scalar's wrapping addition, at full width.
    ///
    /// `Ring for i64` is `wrapping_add`, so `Cell::add` must be exactly that componentwise. A
    /// component that subtracted, or a carry that escaped, shows here for every `i64` pair.
    #[kani::proof]
    fn word_cell_add_matches_native_wrapping() {
        let (a, b, c, d): (i64, i64, i64, i64) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        let z = Cell::new(a, b);
        let w = Cell::new(c, d);
        assert_eq!(
            z.add(w),
            Cell::new(a.wrapping_add(c), b.wrapping_add(d)),
            "addition is not componentwise wrapping addition"
        );
    }

    /// Multiplication is the documented formula in `wrapping_*`, on the window where CBMC returns.
    ///
    /// The dial's `-1/0/1` enters only through `sigma.value()`, so a mutation of the formula -- a
    /// dropped `sigma`, a swapped slot, an add where the cross term needs a multiply -- shows here.
    #[kani::proof]
    fn word_cell_mul_matches_native_wrapping() {
        let z = bounded_word_cell();
        let w = bounded_word_cell();
        for sigma in FACES {
            let s = sigma.value();
            let want = Cell::new(
                z.a.wrapping_mul(w.a)
                    .wrapping_add(s.wrapping_mul(z.b).wrapping_mul(w.b)),
                z.a.wrapping_mul(w.b).wrapping_add(z.b.wrapping_mul(w.a)),
            );
            assert_eq!(z.mul(sigma, w), want, "mul is not the documented formula");
        }
    }

    /// Addition commutes on `Z/2^64`, at full width.
    #[kani::proof]
    fn word_cell_add_is_commutative() {
        let (a, b, c, d): (i64, i64, i64, i64) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        let (z, w) = (Cell::new(a, b), Cell::new(c, d));
        assert_eq!(z.add(w), w.add(z));
    }

    /// Addition associates on `Z/2^64`, at full width.
    #[kani::proof]
    fn word_cell_add_is_associative() {
        let (a, b, c, d, e, f): (i64, i64, i64, i64, i64, i64) = (
            kani::any(),
            kani::any(),
            kani::any(),
            kani::any(),
            kani::any(),
            kani::any(),
        );
        let (z, w, v) = (Cell::new(a, b), Cell::new(c, d), Cell::new(e, f));
        assert_eq!(z.add(w).add(v), z.add(w.add(v)));
    }

    /// `mul_comm` on the hyperbolic face, over `Z/2^8`.
    ///
    /// Stated over `i8` rather than the bounded `i64` window: `Ring for i8` is `Z/2^8`, a
    /// commutative ring, and CBMC bit-blasts an 8-bit multiply in milliseconds where the same
    /// identity over 64 bits did not return in thirty minutes. `scalar.rs` says why that is the
    /// right width for the structure; the `i64` code path is checked by the addition laws and the
    /// exchange harnesses, which are cheap at full width.
    ///
    /// One face, because the only dial dependence is the `sigma * b * b` term and the hyperbolic
    /// face has it nonzero. The `FACES` loop used elsewhere generates an unwinding assertion this
    /// formula's second property does not discharge.
    #[kani::proof]
    fn byte_cell_mul_is_commutative() {
        let sigma = Sigma::Hyperbolic;
        let z = Cell::<i8>::new(kani::any(), kani::any());
        let w = Cell::<i8>::new(kani::any(), kani::any());
        assert_eq!(z.mul(sigma, w), w.mul(sigma, z));
    }

    /// `mul_assoc` on the hyperbolic face, over a bounded window of `Z/2^8`.
    #[kani::proof]
    fn byte_cell_mul_is_associative() {
        let sigma = Sigma::Hyperbolic;
        let z = small_byte_cell();
        let w = small_byte_cell();
        let v = small_byte_cell();
        assert_eq!(z.mul(sigma, w).mul(sigma, v), z.mul(sigma, w.mul(sigma, v)));
    }

    /// Distributivity on the hyperbolic face, over a bounded window of `Z/2^8`.
    #[kani::proof]
    fn byte_cell_mul_distributes_over_add() {
        let sigma = Sigma::Hyperbolic;
        let z = small_byte_cell();
        let w = small_byte_cell();
        let v = small_byte_cell();
        let lhs = z.mul(sigma, w.add(v));
        let rhs = z.mul(sigma, w).add(z.mul(sigma, v));
        assert!(lhs.a == rhs.a, "distributivity fails in the first slot");
        assert!(lhs.b == rhs.b, "distributivity fails in the second slot");
    }

    /// `(1, 0)` is the multiplicative unit on the hyperbolic face, over `Z/2^8`.
    #[kani::proof]
    fn byte_cell_one_is_the_unit() {
        let sigma = Sigma::Hyperbolic;
        let z = Cell::<i8>::new(kani::any(), kani::any());
        assert_eq!(Cell::<i8>::one().mul(sigma, z), z);
    }

    // ── what the upstream trichotomy does and does not survive into Z/2^64 ────────────────────
    //
    // `coord.rs` names `place` hyperbolic and `id` elliptic and cites the Lean development for
    // both -- `Projection.discards_iff_disc_pos`, `Basic.noZeroDivisors_iff_disc_neg`,
    // `Idempotents.exists_nontrivial_idempotent_iff_disc_pos`. Those are theorems over the reals,
    // and this crate's arithmetic is `Z/2^64`. The harnesses above check them on `Cell<Dyadic>`,
    // which has halves and is the real path. `Cell<i64>` -- `Ring for i64` is `wrapping_*`, so it
    // **is** `Z/2^64` -- is the path the address actually takes, and these say what happens there.

    /// Two is not invertible in `Z/2^64`, so the point the real idempotents sit at is not there.
    #[kani::proof]
    fn the_half_the_real_idempotents_need_is_not_in_the_word_ring() {
        let h: u64 = kani::any();
        assert_ne!(h.wrapping_mul(2), 1, "two has no inverse in Z/2^64");
    }

    /// A word times an **odd** word is zero only when the word is.
    ///
    /// `Z/2^64` has zero divisors, so `mul_eq_zero` is unavailable. What replaces it is narrower
    /// and enough: an odd factor is a unit, so it can be cancelled even though the ring is not a
    /// domain.
    #[kani::proof]
    fn an_odd_factor_can_be_cancelled_in_the_word_ring() {
        let b: u64 = kani::any();
        let u: u64 = kani::any();
        kani::assume(u & 1 == 1);
        kani::assume(b.wrapping_mul(u) == 0);
        assert_eq!(b, 0, "an odd multiplier is a unit, so it cannot annihilate");
    }

    /// **The trichotomy's idempotent characterisation does not transfer.** Over `Z/2^64` the only
    /// idempotents are `0` and `1`, at *every* dial.
    ///
    /// `(a + bx)^2 = (a^2 + sigma b^2) + 2ab x`, so idempotence is `2ab = b` and
    /// `a^2 + sigma b^2 = a`. The first is `b (2a - 1) = 0`, and `2a - 1` is odd, so
    /// `an_odd_factor_can_be_cancelled_in_the_word_ring` forces `b = 0` without the dial being
    /// consulted at all. Then `a^2 = a`, and one of `a`, `a - 1` is odd, so
    /// `the_only_word_solutions_of_a_squared_equals_a` forces `a` into `{0, 1}`.
    ///
    /// Over `Z/2^8`, which is what decides: the argument uses exactly two facts -- `2a - 1` is odd
    /// and one of `a`, `a - 1` is odd -- and both hold in `Z/2^n` for every `n >= 1`, so it is
    /// uniform in the width in the way `Ring for i8`'s doc describes.
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
    /// `2^32` squares to zero in the scalar ring already, so `(2^32, 0)` is a nonzero cell whose
    /// square is zero whatever the dial is -- the dial never enters, because the second slot is
    /// empty.
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
    /// coefficients, so it holds in every commutative ring or in none. The mathematical statement
    /// is proved upstream over an arbitrary commutative ring; what this harness adds is that
    /// `Cell::mul` and `Cell::norm_form` as *written here* compute that thing.
    ///
    /// # Why the operands are bounded
    ///
    /// `N(zw)` is degree **four** in the four scalars, so a bit-precise checker bit-blasts a
    /// multiply of two multiplies. Over unrestricted `i8` this did not return in twenty-five
    /// minutes, where the degree-two laws beside it decide in under a second: the width is not what
    /// costs, the nesting is. Bounded to a 4-bit window it decides.
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
    /// A projection here is multiplication by an idempotent, and nothing else.
    /// `the_word_ring_has_no_nontrivial_idempotent_at_any_dial` leaves only `e = 0`, which keeps
    /// nothing, and `e = 1`, which kills nothing. `Discards` is therefore **false at every dial**,
    /// where over the reals it is exactly `0 < sigma`.
    ///
    /// Non-vacuous by construction: the premise is satisfied by `e = 1` with `z = 0`, which the
    /// harness checks before assuming anything, so a passing run is the implication and not an
    /// empty hypothesis.
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

    // ── packing and the address bijection ─────────────────────────────────────────────────────

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

    /// The elliptic face's safety property, as a **trivial kernel** rather than as
    /// `id(a) != id(b)`.
    ///
    /// `id` is `x * PHI`, so `id(a) - id(b) = id(a - b)`, and injectivity is exactly "nothing but
    /// zero maps to zero". Asking CBMC for the pairwise form makes it bit-blast two 64-bit
    /// multiplies against each other and it does not return. The kernel form decides at once, and
    /// it names *why* the face is exact rather than checking that it happens to be.
    #[kani::proof]
    fn the_elliptic_address_has_a_trivial_kernel() {
        let d: u64 = kani::any();
        kani::assume(d != 0);
        assert_ne!(
            crate::coord::spread(d).id,
            0,
            "a nonzero key maps to the same word as zero"
        );
        assert_eq!(
            crate::coord::spread(0).id,
            0,
            "the map is not linear, so the kernel argument does not apply"
        );
    }

    /// `finish` must permute, not fold: it may move a key, never merge two.
    ///
    /// Proved by its steps rather than end to end. `finish` is four operations, and the whole
    /// composition against two symbolic words is two pairs of 64-bit multiplies -- the solver does
    /// not return. Each step alone does, in milliseconds, and a composition of bijections is a
    /// bijection. What is machine-checked is that each step permutes; the composition is an
    /// argument on top of it.
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

    /// Multiplication by an odd constant permutes `Z/2^64`, for both of `finish`'s multipliers and
    /// for `PHI`.
    ///
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

    /// A bucket index is in range for its table.
    #[kani::proof]
    fn a_bucket_index_fits_its_table() {
        let address: u64 = kani::any();
        let bits: u32 = kani::any();
        kani::assume(bits >= 1 && bits <= 63);
        assert!(bucket(address, bits) < (1u64 << bits));
    }

    /// The xor-fold step `x ^ (x >> 48)` is an involution -- the shift is at least half the width.
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

    // ── the fold's guard, as laws ─────────────────────────────────────────────────────────────

    /// **The rejected shipped guard is symmetric on a class of size `2^32`.**
    ///
    /// `mix2(a, b)` was `(hi ^ lo) ^ g(a, b)` with `g(a, b) = a + rotl(b, 32)`. The product fold is
    /// symmetric, so `mix2(u, v) == mix2(v, u)` exactly when `g(u, v) == g(v, u)`, which is exactly
    /// when `f(u) == f(v)` for `f(x) = x - rotl(x, 32)`. Writing `x = h * 2^32 + l` gives
    /// `f(x) = (h - l) * (2^32 - 1)`, and `2^32 - 1` is **odd**, hence invertible: `f` collapses
    /// every word with the same half difference onto one value, fibers of size `2^32`.
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
    /// excluded rather than glossed: at `l == u32::MAX` the increment crosses into the high half.
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

    /// **The shipped `a - b` guard swaps on exactly two points, and keeps everything else.**
    ///
    /// For a linear guard `g(a, b) = alpha * a + beta * b` the three properties the fold wants are
    /// each a parity condition, and they are not in tension: two odds sum to an even, so
    /// per-argument injectivity implies the diagonal collapse that the moment test needs. The swap
    /// kernel is `2^v2(alpha - beta)`, which for `(1, -1)` is **two**: `y == x` and `y == x + 2^63`.
    /// Against the rejected guard's `2^32` that is thirty-one bits of improvement.
    #[kani::proof]
    fn the_odd_pair_guard_swaps_only_on_a_two_element_coset() {
        let x: u64 = kani::any();
        let y: u64 = kani::any();
        let g = |a: u64, b: u64| a.wrapping_sub(b);
        let d = y.wrapping_sub(x);
        assert_eq!(g(x, y) == g(y, x), d == 0 || d == 1 << 63);
    }

    /// **`a - b` is injective in each argument**, which is `mix2`'s stated contract.
    ///
    /// Stated as the trivial kernel rather than pairwise, for the reason
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
    /// `t + seed` and the guard restricted to that line is `(alpha + beta) * t + k`. With
    /// `alpha + beta == 0` the diagonal is the **constant** `s1 - s2`: it carries nothing about
    /// the input at all, the strongest form of the collapse.
    #[kani::proof]
    fn the_odd_pair_guard_loses_the_diagonal() {
        let t: u64 = kani::any();
        let s1: u64 = kani::any();
        let s2: u64 = kani::any();
        let phi = |v: u64| v.wrapping_add(s1).wrapping_sub(v.wrapping_add(s2));
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
    /// left. `a - b` is zero only when the operands are equal.
    #[kani::proof]
    fn the_odd_pair_guard_survives_an_annihilated_product() {
        let v: u64 = kani::any();
        kani::assume(v != 0);
        let g = |a: u64, b: u64| a.wrapping_sub(b);
        assert_ne!(g(0, v), 0, "a zero first operand must not zero the guard");
        assert_ne!(g(v, 0), 0, "a zero second operand must not zero the guard");
    }

    // ── why the group window is taken from the top of the placement word ──────────────────────

    /// A key below `2^32` never reaches the top quarter of the widening product.
    ///
    /// `x < 2^32` and `PHI < 2^64` give `x * PHI < 2^96`, so bits 96..128 are zero.
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
    /// `a_small_key_never_reaches_the_top_of_the_product` shows is zero.
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
    /// `d << k`, and `(d << k) * PHI` is divisible by `2^k` because `d * PHI` is an integer.
    /// Multiplication carries information from bit `i` to bits at or above `i`, and to no bit
    /// below `i`.
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
    /// 128 bits.
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

    /// A left rotation by 32 sends bit `(i + 32) mod 64` to bit `i`.
    ///
    /// `coord::spread`'s doc states `place_i == lo_i ^ hi_((i - 32) mod 64)` and attaches
    /// "verified bit for bit over two million keys". `place` is `rotl(hi, 32) ^ lo` **by
    /// definition**, and xor is bitwise, so the whole content is where the rotation puts each bit.
    /// Stated without the multiply it decides instantly.
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

    /// Is `place` injective on 32-bit keys?
    ///
    /// A search over `2^26` dense keys and `2^26` scattered ones found zero collisions, which
    /// settles nothing in either direction -- a uniformly random function on 64 bits has about
    /// 0.0001 expected collisions at that sample size. Asked where it can be decided, the answer is
    /// **yes**: `place` separates every pair of keys below `2^32`. Whether `place` is injective on
    /// the full 64-bit domain is open and does not matter here: the security argument is about
    /// matching `j` *windows* of `place`, which is `the_group_index_cannot_separate_keys`'s
    /// subject and not this.
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

    // ── the fixed-shard table ─────────────────────────────────────────────────────────────────
    //
    // The storage kernel is a constant default plus at most `N` overrides per home shard, selected
    // by `home(key) % 256`. These harnesses pin the read relation, the merge, the refusal path and
    // the bounds of the index.
    //
    // `shard_index` is private to `table`, and a *symbolic* index into the 256-element shard array
    // makes CBMC build a 256-way mux per access (718s and 1072s in the solver for two earlier
    // harnesses). The symbolic-key coverage is therefore split by what each layer can decide:
    //
    // - `the_word_home_route_is_the_low_byte` states the real routing arithmetic over a symbolic
    //   `i64` key, sharing the expression `WordHome::home` with the kernel.
    // - `a_reduced_shard_table_searches_symbolic_keys` runs the access path -- `home % SHARDS`,
    //   first-match slot scan, full-key equality, new-first merge -- in `MiniTable`, a reduced-width
    //   replica of the kernel. It is a model: a `table.rs` mutation is not observed there.
    // - `table_symbolic_key_round_trips_a_present_key` and
    //   `table_symbolic_byte_keys_fill_one_shard` run the *real* kernel with symbolic keys and a
    //   constant home, so the shard index is concrete and no mux is built; the slot search,
    //   full-key equality and merge are the extracted body's.
    // - `table_build_folds_a_multi_key_stream_in_order` runs the real `build` over two shards with
    //   a repeated key and an order-sensitive merge, so the fold order is observable.
    //
    // **Not covered:** a symbolic shard index against the real 256-entry array, hence the real
    // `% 256` route combined with a symbolic key in one harness. The `% SHARDS` model and the
    // `% 256` real arithmetic are joined only by the shared expression and the concrete-key
    // behavioural harnesses below.

    /// The residue of a home word below the shard count, for **every** home word.
    ///
    /// This is the arithmetic `Table::shard_index` performs before the array access. A modulus that
    /// let the residue reach the array length would make the behavioural harnesses below panic, and
    /// this states the bound itself.
    #[kani::proof]
    fn the_shard_index_is_a_residue_below_the_shard_count() {
        let home: u64 = kani::any();
        assert!(((home % 256) as usize) < 256);
    }

    /// The real `WordHome` route is the low byte of the key word, for every key.
    ///
    /// `Table::shard_index` is `home(key) % 256`, and `WordHome::home` is the key's bit pattern.
    /// Over two symbolic `i64` keys this states the routing arithmetic the real kernel performs:
    /// the residue is the low byte, it is below the shard count, and two keys route together
    /// exactly when their low bytes agree. It is the `% 256` expression that `MiniTable`'s
    /// `% SHARDS` reduces and that the concrete-key table harnesses exercise end to end.
    #[kani::proof]
    fn the_word_home_route_is_the_low_byte() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        assert_eq!(WordHome.home(&a) % 256, (a as u64) & 0xff);
        assert!((WordHome.home(&a) % 256) < 256);
        assert_eq!(
            WordHome.home(&a) % 256 == WordHome.home(&b) % 256,
            (a as u64) & 0xff == (b as u64) & 0xff,
            "two keys route together exactly when their low bytes agree"
        );
    }

    /// A constant home: every key routes to shard `0`.
    ///
    /// A test fixture, not the kernel's `WordHome`. It makes `shard_index` a concrete `0` so the
    /// real `Table` bodies can be checked with symbolic keys without a symbolic index into the
    /// 256-entry `shards` array.
    struct ConstZeroHome;

    impl Home<i64> for ConstZeroHome {
        fn home(&self, _key: &i64) -> u64 {
            0
        }
    }

    /// A real-table read/write round-trip over a **symbolic** key in one concrete shard.
    ///
    /// The home is `ConstZeroHome`, so `shard_index` is concrete and CBMC builds no mux; the key,
    /// the slot scan and the full-key equality are symbolic and the bodies are `table.rs`'s. A
    /// present symbolic key reads the value it was written with, and a distinct concrete key reads
    /// the default. One key rather than several because the solver cost of symbolic `i64` keys in
    /// the real slot scan is superlinear (measured: one key 0.3s, two 56s, four 1148s); the
    /// multi-key symbolic case is `table_symbolic_byte_keys_fill_one_shard` at reduced key width and
    /// `a_reduced_shard_table_searches_symbolic_keys` at reduced shard width.
    ///
    /// **Not covered:** the routing from a symbolic key to its shard, which needs a symbolic index
    /// into the real 256-entry array. It is stated arithmetically by
    /// `the_word_home_route_is_the_low_byte` and modelled by
    /// `a_reduced_shard_table_searches_symbolic_keys`.
    #[kani::proof]
    fn table_symbolic_key_round_trips_a_present_key() {
        let key: i64 = kani::any();
        kani::assume(key != 0);
        let mut table = Table::<i64, i64, ConstZeroHome, 1>::new(0, ConstZeroHome);
        table.insert_with(&WordMerge, key, 5).unwrap();
        assert_eq!(*table.get(&key), 5, "the symbolic key reads its value");
        assert_eq!(
            *table.get(&0),
            0,
            "the distinct absent key reads the default"
        );
    }

    /// A constant home over byte keys, and byte wrapping addition, for the reduced-key harness.
    struct ConstZeroHomeI8;

    impl Home<i8> for ConstZeroHomeI8 {
        fn home(&self, _key: &i8) -> u64 {
            0
        }
    }

    struct ByteMerge;

    impl Merge<i8> for ByteMerge {
        fn merge(&self, new: i8, old: i8) -> i8 {
            new.wrapping_add(old)
        }
    }

    /// A real-table read/write round-trip over four **symbolic** byte keys in one concrete shard.
    ///
    /// The real `Table` bodies are monomorphised at `K = i8`, so the slot scan, the full-key
    /// equality and the first-free-slot placement are the extracted code; only the key width is
    /// reduced, and an 8-bit key makes four symbolic keys decide where four `i64` keys did not
    /// (1148s). Four distinct keys fill the four slots and each reads back its own value, so the
    /// scan must match the full key and not merely the shard. The home is `ConstZeroHomeI8`, so the
    /// shard index is concrete and no mux is built.
    ///
    /// **Not covered:** the routing from a symbolic key to its shard, and any key wider than a
    /// byte; the `i64` path is `table_symbolic_key_round_trips_a_present_key` (one symbolic key).
    #[kani::proof]
    fn table_symbolic_byte_keys_fill_one_shard() {
        let (k0, k1, k2, k3): (i8, i8, i8, i8) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        kani::assume(k0 != k1 && k0 != k2 && k0 != k3 && k1 != k2 && k1 != k3 && k2 != k3);
        let mut table = Table::<i8, i8, ConstZeroHomeI8, 4>::new(0, ConstZeroHomeI8);
        table.insert_with(&ByteMerge, k0, 1).unwrap();
        table.insert_with(&ByteMerge, k1, 2).unwrap();
        table.insert_with(&ByteMerge, k2, 3).unwrap();
        table.insert_with(&ByteMerge, k3, 4).unwrap();
        assert_eq!(*table.get(&k0), 1);
        assert_eq!(*table.get(&k1), 2);
        assert_eq!(*table.get(&k2), 3);
        assert_eq!(*table.get(&k3), 4);
    }

    /// A reduced-width replica of `Table`'s shard/slot kernel, for symbolic-key checking.
    ///
    /// `SHARDS` and `SLOTS` stand where `table.rs`'s `256` and `N` stand, and every expression is
    /// the one `Table` writes: `home(key) % SHARDS` for the shard, the first-match scan over the
    /// slot array, the full-key equality that decides a hit, and `merge(new, old)` including the
    /// default on the first write.
    ///
    /// **It is a model, not the extracted body.** A mutation in `table.rs` is not observed here;
    /// the real bodies carry the same properties through
    /// `table_symbolic_key_round_trips_a_present_key` and
    /// `table_symbolic_byte_keys_fill_one_shard` (symbolic keys, one concrete shard) and
    /// `table_build_folds_a_multi_key_stream_in_order` (two concrete shards, a repeated key). It
    /// exists because a symbolic index into the real 256-entry `shards` array makes CBMC build a
    /// 256-way mux per access; at `SHARDS = 4` the same access path is decided.
    struct MiniTable<K, V, H, const SHARDS: usize, const SLOTS: usize> {
        default: V,
        home: H,
        shards: [Option<[Option<(K, V)>; SLOTS]>; SHARDS],
    }

    impl<K, V, H: Home<K>, const SHARDS: usize, const SLOTS: usize> MiniTable<K, V, H, SHARDS, SLOTS> {
        fn new(default: V, home: H) -> Self {
            Self {
                default,
                home,
                shards: [const { None }; SHARDS],
            }
        }

        fn shard_index(&self, key: &K) -> usize {
            (self.home.home(key) % SHARDS as u64) as usize
        }
    }

    impl<K: Eq, V, H: Home<K>, const SHARDS: usize, const SLOTS: usize>
        MiniTable<K, V, H, SHARDS, SLOTS>
    {
        fn find(&self, shard: usize, key: &K) -> Option<usize> {
            if let Some(slots) = &self.shards[shard] {
                return Self::find_slots(slots, key);
            }
            None
        }

        fn find_slots(slots: &[Option<(K, V)>; SLOTS], key: &K) -> Option<usize> {
            let mut i = 0;
            while i < SLOTS {
                if let Some((stored, _)) = &slots[i] {
                    if stored == key {
                        return Some(i);
                    }
                }
                i += 1;
            }
            None
        }

        fn get(&self, key: &K) -> &V {
            let shard = self.shard_index(key);
            if let Some(i) = self.find(shard, key) {
                if let Some(slots) = &self.shards[shard] {
                    if let Some((_, value)) = &slots[i] {
                        return value;
                    }
                }
            }
            &self.default
        }
    }

    impl<K: Eq, V: Clone, H: Home<K>, const SHARDS: usize, const SLOTS: usize>
        MiniTable<K, V, H, SHARDS, SLOTS>
    {
        fn insert_with<F: Merge<V>>(
            &mut self,
            merge: &F,
            key: K,
            value: V,
        ) -> Result<(), StorageError> {
            let shard = self.shard_index(&key);
            if let Some(i) = self.find(shard, &key) {
                if let Some(slots) = &mut self.shards[shard] {
                    if let Some((_, stored)) = &mut slots[i] {
                        *stored = merge.merge(value, stored.clone());
                        return Ok(());
                    }
                }
            }
            if SLOTS == 0 {
                return Err(StorageError::ShardFull);
            }
            if self.shards[shard].is_none() {
                let mut slots = [const { None }; SLOTS];
                let merged = merge.merge(value, self.default.clone());
                slots[0] = Some((key, merged));
                self.shards[shard] = Some(slots);
                return Ok(());
            }
            if let Some(slots) = &mut self.shards[shard] {
                let mut i = 0;
                while i < SLOTS {
                    if slots[i].is_none() {
                        let merged = merge.merge(value, self.default.clone());
                        slots[i] = Some((key, merged));
                        return Ok(());
                    }
                    i += 1;
                }
            }
            Err(StorageError::ShardFull)
        }
    }

    /// The reduced-width table routes and searches **symbolic** keys at `% SHARDS` width.
    ///
    /// This is `MiniTable`'s coverage, not `Table`'s: a symbolic shard index against the real
    /// 256-entry array is the cost the model avoids. What it decides is the access path itself,
    /// over symbolic keys: `home(key) % 4` is below the shard count; a written key reads its value
    /// (the slot scan matches the full key, not the shard); a distinct key routed to the same shard
    /// reads the default; and a repeated write merges new-first, with the absent shard-mate still
    /// reading the default. `WordHome` and `WordMerge` are the real instances, so the expressions
    /// are the kernel's with `SHARDS` in place of `256`.
    #[kani::proof]
    fn a_reduced_shard_table_searches_symbolic_keys() {
        let key: i64 = kani::any();
        let mate: i64 = kani::any();
        let value: i64 = kani::any();
        let second: i64 = kani::any();
        kani::assume(mate != key);
        kani::assume(WordHome.home(&mate) % 4 == WordHome.home(&key) % 4);

        let mut table = MiniTable::<i64, i64, _, 4, 2>::new(0, WordHome);
        assert!(
            table.shard_index(&key) < 4,
            "the route is a residue below the shard count"
        );
        table.insert_with(&WordMerge, key, value).unwrap();
        assert_eq!(*table.get(&key), value, "the written key reads its value");
        assert_eq!(
            *table.get(&mate),
            0,
            "a same-shard key that was never written reads the default"
        );

        table.insert_with(&WordMerge, key, second).unwrap();
        assert_eq!(
            *table.get(&key),
            second.wrapping_add(value),
            "a repeated key merges new-first"
        );
        assert_eq!(
            *table.get(&mate),
            0,
            "the update did not disturb the absent shard-mate"
        );
    }

    /// A fresh table reads its total default for every key, across the shard boundaries.
    ///
    /// The keys are concrete but chosen to reach both ends of the array and the wrap: `-1` and
    /// `i64::MAX` land on shard 255, `i64::MIN` and `256` on shard 0. A key is not symbolic here,
    /// so this is a bounded check of the access path rather than a universal one; the universal
    /// statement is `the_shard_index_is_a_residue_below_the_shard_count`.
    #[kani::proof]
    fn table_fresh_read_is_the_default_for_every_key() {
        let table = Table::<i64, i64, WordHome, 4>::new(7, WordHome);
        for key in [0i64, 1, 255, 256, 299, -1, i64::MAX, i64::MIN] {
            assert_eq!(*table.get(&key), 7, "fresh read at {key}");
        }
    }

    /// The read relation for a present key: what a successful insert stored is what `get` returns.
    ///
    /// The default is zero and the merge is wrapping addition, so the first write stores exactly
    /// the value. A mutation of the slot index, the stored key, or the merge breaks this. One key
    /// with a symbolic value: a loop of keys multiplies the symbolic table state and the solver
    /// cost with it, and the same-shard read is `table_default_is_returned_for_a_key_never_written`.
    #[kani::proof]
    fn table_insert_then_get_round_trips_a_present_key() {
        let value: i64 = kani::any();
        let mut table = Table::<i64, i64, WordHome, 4>::new(0, WordHome);
        table.insert_with(&WordMerge, 7, value).unwrap();
        assert_eq!(*table.get(&7), value);
    }

    /// A second write to the same key is `merge(new, stored)`, including the default on the first.
    #[kani::proof]
    fn table_update_merges_new_with_the_stored_value() {
        let (first, second): (i64, i64) = (kani::any(), kani::any());
        let mut table = Table::<i64, i64, WordHome, 4>::new(0, WordHome);
        table.insert_with(&WordMerge, 7, first).unwrap();
        table.insert_with(&WordMerge, 7, second).unwrap();
        assert_eq!(*table.get(&7), second.wrapping_add(first));
    }

    /// A key that was never written reads the default, even when it shares the written key's shard.
    ///
    /// Full-key equality, not shard equality, is the decision; a table that answered from the shard
    /// alone would return the wrong value here. `0` and `256` share shard 0.
    #[kani::proof]
    fn table_default_is_returned_for_a_key_never_written() {
        let mut table = Table::<i64, i64, WordHome, 4>::new(9, WordHome);
        table.insert_with(&WordMerge, 0, 1).unwrap();
        assert_eq!(
            *table.get(&0),
            10,
            "the written key is merged with the default"
        );
        assert_eq!(*table.get(&256), 9, "a shard-mate that was never written");
        assert_eq!(*table.get(&1), 9, "a key on another shard");
    }

    /// A refusal changes nothing: the resident key keeps its value and the refused key reads the
    /// default.
    ///
    /// `0` and `256` have the same low eight bits, hence the same shard; with one slot per shard
    /// the second insert is refused.
    #[kani::proof]
    fn table_refusal_leaves_the_table_unchanged() {
        let mut table = Table::<i64, i64, WordHome, 1>::new(5, WordHome);
        table.insert_with(&WordMerge, 0, 2).unwrap();
        let result = table.insert_with(&WordMerge, 256, 3);
        assert_eq!(result, Err(StorageError::ShardFull));
        assert_eq!(*table.get(&0), 7);
        assert_eq!(*table.get(&256), 5);
    }

    /// Zero capacity refuses every insertion, without invoking the merge.
    #[kani::proof]
    fn table_zero_capacity_refuses_without_merging() {
        let mut table = Table::<i64, i64, WordHome, 0>::new(4, WordHome);
        let result = table.insert_with(&|_, _| panic!("merge on refusal"), 0, 1);
        assert_eq!(result, Err(StorageError::ShardFull));
        assert_eq!(*table.get(&0), 4);
    }

    /// `merge(new, old) = new - old`, non-commutative so the build fold's order is observable.
    ///
    /// `WordMerge` is wrapping addition, which is commutative: a build that reversed its stream
    /// would produce the same word and no harness could see it. This instance makes the order a
    /// value, so `table_build_folds_a_multi_key_stream_in_order` fails on a reversed fold or a
    /// swapped `insert_with` merge.
    struct NewFirstDifference;

    impl Merge<i64> for NewFirstDifference {
        fn merge(&self, new: i64, old: i64) -> i64 {
            new.wrapping_sub(old)
        }
    }

    /// `Table::build` folds a small stream in order across two shards, with a repeated key.
    ///
    /// Keys `0` and `1` land on shards `0` and `1`, so the stream exercises more than one shard
    /// without a symbolic index. Key `0` appears twice, so its second write takes the update path
    /// and merges with the first write, not the default. The merge is `new - old`, non-commutative,
    /// so the fold order is observable: a build that reversed its stream, or an `insert_with` that
    /// merged `old` before `new`, gives a different word. The `get` assertions are the
    /// insert-then-get round-trip for both a repeated and a distinct key.
    ///
    /// **Not covered:** a symbolic key, a symbolic shard count or a refusal inside `build`; the
    /// two-shard layout and the stream are concrete so the ordered fold is checked exactly.
    #[kani::proof]
    fn table_build_folds_a_multi_key_stream_in_order() {
        let (a, b, c): (i64, i64, i64) = (kani::any(), kani::any(), kani::any());
        let stream = [(0i64, a), (1i64, b), (0i64, c)];
        let table =
            Table::<i64, i64, _, 2>::build(&NewFirstDifference, 10, &stream, WordHome).unwrap();
        assert_eq!(
            *table.get(&0),
            c.wrapping_sub(a.wrapping_sub(10)),
            "the repeated key must fold new-first in stream order"
        );
        assert_eq!(
            *table.get(&1),
            b.wrapping_sub(10),
            "the distinct key must fold once against the default"
        );
    }

    // ── the exchange and the carrier step ─────────────────────────────────────────────────────

    /// The derived impulse is the componentwise difference, and nothing is supplied.
    #[kani::proof]
    fn the_derived_impulse_is_the_componentwise_difference() {
        let (a, b, c, d): (i64, i64, i64, i64) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        let p = Cell::new(a, b);
        let q = Cell::new(c, d);
        assert_eq!(
            exchange_impulse(p, q),
            Cell::new(a.wrapping_sub(c), b.wrapping_sub(d))
        );
    }

    /// The equal-mass exchange swaps the two momenta: the carrier ends at `q`, the body at `p`.
    #[kani::proof]
    fn the_elastic_exchange_swaps_the_two_momenta() {
        let p = Cell::<i64>::new(kani::any(), kani::any());
        let q = Cell::<i64>::new(kani::any(), kani::any());
        let x = elastic_exchange(p, q);
        assert_eq!(x.impulse, exchange_impulse(p, q));
        assert_eq!(x.carrier_after, q);
        assert_eq!(x.body_after, p);
    }

    /// An impulse is accepted exactly when it is the derived one.
    #[kani::proof]
    fn an_impulse_is_accepted_exactly_when_it_is_the_derived_one() {
        let p = Cell::<i64>::new(kani::any(), kani::any());
        let q = Cell::<i64>::new(kani::any(), kani::any());
        let claimed = Cell::<i64>::new(kani::any(), kani::any());
        let accepted = accept_derived_impulse(p, q, claimed);
        assert_eq!(accepted.is_some(), claimed == exchange_impulse(p, q));
    }

    /// A successful half-flux is exactly half the negative gradient, under wrapping.
    ///
    /// `diffusive_flux_half(left, right)` is `-(right - left) / 2` when that half exists, so
    /// `2 * half == left - right` in `Z/2^64`; the wrapping multiplication also covers the
    /// `i64::MIN` difference, where the negation is the same residue.
    #[kani::proof]
    fn the_half_flux_is_exactly_half_the_negative_gradient() {
        let (left, right): (i64, i64) = (kani::any(), kani::any());
        if let Some(half) = diffusive_flux_half(left, right) {
            assert_eq!(half.wrapping_mul(2), left.wrapping_sub(right));
        }
    }

    /// A half-flux exists exactly when the difference is representable and even.
    #[kani::proof]
    fn a_half_flux_exists_exactly_for_an_even_representable_difference() {
        let (left, right): (i64, i64) = (kani::any(), kani::any());
        match right.checked_sub(left) {
            Some(difference) if difference % 2 == 0 => {
                assert!(diffusive_flux_half(left, right).is_some());
            }
            Some(_) => assert!(diffusive_flux_half(left, right).is_none()),
            None => assert!(diffusive_flux_half(left, right).is_none()),
        }
    }

    /// A successful conservative step agrees with the wrapping formula for the same update.
    ///
    /// The checked kernel returns the exact integer `density - (outgoing - incoming) + source`
    /// when it is representable, and `Z/2^64` makes that equal to the wrapping expression. A
    /// mutation of the order of the terms, or of a sign, breaks the agreement.
    #[kani::proof]
    fn the_conservative_step_agrees_with_the_wrapping_formula() {
        let density: i64 = kani::any();
        let incoming: i64 = kani::any();
        let outgoing: i64 = kani::any();
        let source: i64 = kani::any();
        if let Some(updated) = conservative_density_step(density, incoming, outgoing, source) {
            assert_eq!(
                updated,
                density
                    .wrapping_sub(outgoing)
                    .wrapping_add(incoming)
                    .wrapping_add(source)
            );
        }
    }
}
