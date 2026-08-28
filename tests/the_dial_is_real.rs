//! The trichotomy, as behaviour rather than vocabulary.
//!
//! Each test mirrors one theorem in `../sanguine-apeiron/proof/Apeiron/Trichotomy/`. The reason
//! they exist as tests and not as comments is the thing this crate was written to fix: hashtrinity
//! names its three storage faces after the trichotomy and branches on no discriminant anywhere, so
//! nothing would fail if the trichotomy were wrong, absent, or replaced by three other words.

use apeiron::cell::{fold, fold_is_order_free, Cell, Sigma};

const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

fn sample() -> Vec<Cell> {
    (-3..4)
        .flat_map(|a| (-3..4).map(move |b| Cell::new(a, b)))
        .collect()
}

/// `mul_comm`, `mul_assoc`, `one_mul` -- on every face, which is why folding is well defined
/// before the dial is even chosen.
#[test]
fn multiplication_is_commutative_associative_and_unital_on_every_face() {
    for sigma in FACES {
        for z in sample() {
            assert_eq!(Cell::one().mul(sigma, z), z, "one_mul at {sigma:?}");
            for w in sample() {
                assert_eq!(z.mul(sigma, w), w.mul(sigma, z), "mul_comm at {sigma:?}");
                for v in [Cell::new(2, 1), Cell::new(-1, 3)] {
                    assert_eq!(
                        z.mul(sigma, w).mul(sigma, v),
                        z.mul(sigma, w.mul(sigma, v)),
                        "mul_assoc at {sigma:?}"
                    );
                }
            }
        }
    }
}

/// `normForm_mul`: the form is multiplicative on every face. This is what makes "degenerate" a
/// property closed under multiplication -- once a lane is dead it stays dead.
#[test]
fn the_norm_is_multiplicative_on_every_face() {
    for sigma in FACES {
        for z in sample() {
            for w in sample() {
                assert_eq!(
                    z.mul(sigma, w).norm_form(sigma),
                    z.norm_form(sigma) * w.norm_form(sigma),
                    "normForm_mul at {sigma:?} for {z:?} {w:?}"
                );
            }
        }
    }
}

/// `two_channels_iff_disc_pos`, `channel_iff_fst_of_disc_zero`, `no_channel_of_disc_neg`.
///
/// The count is not a label: the accessors that return the channels are `None` off the face that
/// has them, so code cannot read a lane that does not exist.
#[test]
fn the_channel_count_is_the_sign_of_the_discriminant() {
    assert_eq!(Sigma::Elliptic.channels(), 0);
    assert_eq!(Sigma::Parabolic.channels(), 1);
    assert_eq!(Sigma::Hyperbolic.channels(), 2);

    let z = Cell::new(3, 2);
    assert!(z.upper(Sigma::Elliptic).is_none() && z.lower(Sigma::Elliptic).is_none());
    assert!(z.channel(Sigma::Elliptic).is_none());

    assert_eq!(z.channel(Sigma::Parabolic), Some(3));
    assert!(z.upper(Sigma::Parabolic).is_none());

    assert_eq!(z.upper(Sigma::Hyperbolic), Some(5));
    assert_eq!(z.lower(Sigma::Hyperbolic), Some(1));
    assert!(z.channel(Sigma::Hyperbolic).is_none());
}

/// `upper_mul` and `lower_mul`: above zero, multiplication is componentwise in the null basis.
///
/// This is the property that makes two lanes accumulate without coordinating, and it is the one
/// real engineering consequence of the whole trichotomy.
#[test]
fn the_two_channels_multiply_independently() {
    let sigma = Sigma::Hyperbolic;
    for z in sample() {
        for w in sample() {
            let zw = z.mul(sigma, w);
            assert_eq!(
                zw.upper(sigma).unwrap(),
                z.upper(sigma).unwrap() * w.upper(sigma).unwrap(),
                "upper_mul for {z:?} {w:?}"
            );
            assert_eq!(
                zw.lower(sigma).unwrap(),
                z.lower(sigma).unwrap() * w.lower(sigma).unwrap(),
                "lower_mul for {z:?} {w:?}"
            );
        }
    }
}

/// `normForm_eq_upper_mul_lower`: the form factors into the two channels, so it vanishes exactly
/// when a lane does. Degeneracy is "a channel died", stated precisely.
#[test]
fn the_norm_factors_into_the_channels() {
    let sigma = Sigma::Hyperbolic;
    for z in sample() {
        assert_eq!(
            z.norm_form(sigma),
            z.upper(sigma).unwrap() * z.lower(sigma).unwrap(),
            "normForm = upper * lower for {z:?}"
        );
    }
}

/// `discards_iff_disc_pos`: a projection that keeps one thing and destroys another exists only
/// above zero.
///
/// Every lossy thing a table does is a discard -- collapsing a key to a fold, culling by interest,
/// demoting detail. Each is therefore a hyperbolic operation, and the crate says so where it can
/// be checked rather than in a comment.
#[test]
fn only_the_hyperbolic_face_can_discard() {
    assert!(!Sigma::Elliptic.discards());
    assert!(!Sigma::Parabolic.discards());
    assert!(Sigma::Hyperbolic.discards());

    // The witness: an idempotent that is neither 0 nor 1 projects onto one lane and kills the
    // other. `2e = (1, 1)` is the doubled idempotent, scaled to stay in the integers.
    let sigma = Sigma::Hyperbolic;
    let e = Cell::new(1, 1);
    assert_eq!(e.lower(sigma), Some(0), "the projector kills the lower lane");
    for z in sample() {
        let projected = z.mul(sigma, e);
        assert_eq!(projected.lower(sigma), Some(0), "the lane stays dead");
        assert_eq!(projected.norm_form(sigma), 0, "and the result is degenerate");
    }
}

/// `fold_perm`: for `sigma >= 0` the fold does not depend on the order of its input.
///
/// The licence for accumulating from many threads and combining once. Elliptic is excluded
/// because the theorem excludes it, not because it was tested and failed.
#[test]
fn folding_is_order_free_exactly_where_the_theorem_says() {
    assert!(!fold_is_order_free(Sigma::Elliptic));
    assert!(fold_is_order_free(Sigma::Parabolic));
    assert!(fold_is_order_free(Sigma::Hyperbolic));

    let cells = [
        Cell::new(1, 1),
        Cell::new(2, -1),
        Cell::new(0, 3),
        Cell::new(-2, 1),
    ];
    for sigma in [Sigma::Parabolic, Sigma::Hyperbolic] {
        let forward = fold(sigma, &cells);
        let mut reversed = cells;
        reversed.reverse();
        assert_eq!(forward, fold(sigma, &reversed), "fold_perm at {sigma:?}");

        let rotated = [cells[2], cells[3], cells[0], cells[1]];
        assert_eq!(forward, fold(sigma, &rotated), "fold_perm at {sigma:?}");
    }
}

/// Two properties that are easy to conflate, and the base ring that separates them further.
///
/// The corpus is over the reals; this crate is over wrapping `i64`. That gap is real and is
/// stated rather than glossed:
///
/// - **Nontrivial idempotents.** Over the reals they exist exactly above zero
///   (`exists_nontrivial_idempotent_iff_disc_pos`), and they are `(1 +/- j)/2`. Over the integers
///   they exist on **no** face, because they are half-integral. So the theorem does not transfer,
///   and this test asserts the integer fact instead of pretending it does.
/// - **Zero divisors.** These do transfer. None below zero (the elliptic face is a field), the
///   nilpotent line `(0, b)` at zero, the null cone above it.
///
/// What survives to the integers is the part the table actually uses: `(1, 1)` is `2e+`, and
/// multiplying by it kills the lower lane exactly as the idempotent would. The discard is
/// available above zero and nowhere else, which is
/// `only_the_hyperbolic_face_can_discard`.
#[test]
fn idempotents_and_zero_divisors_over_the_integers() {
    // Over Z, no face has a nontrivial idempotent -- including the hyperbolic one, whose real
    // idempotents need a factor of one half.
    for sigma in FACES {
        let nontrivial: Vec<Cell> = sample()
            .into_iter()
            .filter(|z| z.mul(sigma, *z) == *z && *z != Cell::zero() && *z != Cell::one())
            .collect();
        assert!(
            nontrivial.is_empty(),
            "{sigma:?}: over Z the idempotents are 0 and 1 only, found {nontrivial:?}"
        );
    }

    // Elliptic is a field: the norm a^2 + b^2 vanishes only at the origin.
    for z in sample() {
        assert_eq!(
            z.norm_form(Sigma::Elliptic) == 0,
            z == Cell::zero(),
            "elliptic has no zero divisors, {z:?}"
        );
    }

    // Parabolic: the nilpotent line. (0, b) squares to zero for every b.
    for b in 1..4 {
        let n = Cell::new(0, b);
        assert_eq!(n.mul(Sigma::Parabolic, n), Cell::zero(), "nilpotent {n:?}");
        assert_ne!(n, Cell::zero());
    }

    // Hyperbolic: the null cone. The two null directions annihilate each other, and each is
    // degenerate on its own.
    let (np, nm) = (Cell::new(1, 1), Cell::new(1, -1));
    assert_eq!(np.mul(Sigma::Hyperbolic, nm), Cell::zero());
    assert_eq!(np.norm_form(Sigma::Hyperbolic), 0);
    assert_eq!(nm.norm_form(Sigma::Hyperbolic), 0);

    // And a zero divisor exists above zero but not below it, which is the transferable half.
    for sigma in FACES {
        let divisors = sample()
            .into_iter()
            .filter(|z| *z != Cell::zero() && z.norm_form(sigma) == 0)
            .count();
        assert_eq!(
            divisors > 0,
            sigma != Sigma::Elliptic,
            "{sigma:?}: zero divisors exist iff the discriminant is not negative"
        );
    }
}
