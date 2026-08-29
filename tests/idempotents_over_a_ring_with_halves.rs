//! `exists_nontrivial_idempotent_iff_disc_pos`, over a scalar that can hold the idempotents.
//!
//! This was first written off as "does not transfer to the integers". It does transfer -- the
//! obstruction was the scalar, not the theorem. `(1 +/- j)/2` needs one bit of fraction, and a
//! dyadic rational is two integers, exactly as a float is three. Choosing `i64` and concluding the
//! algebra does not fit is choosing the wrong container and blaming the contents.
//!
//! What the type system now says, and the comment used to: `Cell::<i64>::idem_plus()` does not
//! compile, because `i64` is a `Ring` and not `Halve`.

use apeiron::cell::{Cell, Sigma};
use apeiron::scalar::{Dyadic, Ring};

const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

/// `idempotent_idemPlus` / `idempotent_idemMinus`: both are idempotent above zero.
#[cfg_attr(miri, ignore = "pure algebra on `Cell`: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn the_idempotents_are_idempotent_above_zero() {
    let e_plus = Cell::<Dyadic>::idem_plus();
    let e_minus = Cell::<Dyadic>::idem_minus();
    let sigma = Sigma::Hyperbolic;

    assert_eq!(e_plus.mul(sigma, e_plus), e_plus, "e+ is idempotent");
    assert_eq!(e_minus.mul(sigma, e_minus), e_minus, "e- is idempotent");
}

/// `idemPlus_nontrivial`: neither is `0` or `1`, which is the whole content of "nontrivial".
#[cfg_attr(miri, ignore = "pure algebra on `Cell`: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn the_idempotents_are_neither_zero_nor_one() {
    for e in [Cell::<Dyadic>::idem_plus(), Cell::<Dyadic>::idem_minus()] {
        assert_ne!(e, Cell::<Dyadic>::zero());
        assert_ne!(e, Cell::<Dyadic>::one());
    }
}

/// They are orthogonal and sum to one: `e+ e- = 0`, `e+ + e- = 1`.
///
/// This is the decomposition itself. A cell above zero is `upper * e+ + lower * e-`, which is why
/// the two lanes never interact.
#[cfg_attr(miri, ignore = "pure algebra on `Cell`: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn the_idempotents_split_the_unit() {
    let sigma = Sigma::Hyperbolic;
    let (p, m) = (Cell::<Dyadic>::idem_plus(), Cell::<Dyadic>::idem_minus());

    assert_eq!(p.mul(sigma, m), Cell::zero(), "orthogonal");
    assert_eq!(p.add(m), Cell::one(), "they sum to the unit");
    assert_eq!(p.norm_form(sigma), Dyadic::ZERO, "each is degenerate");
    assert_eq!(m.norm_form(sigma), Dyadic::ZERO);
}

/// `halves_idemPlus` / `halves_idemMinus`: in the null basis the idempotents are the unit vectors.
///
/// This is why they look non-integral and are not: `(1/2, 1/2)` in the standard basis is `(1, 0)`
/// in the null basis. The halves are an artefact of the coordinates, not of the algebra.
#[cfg_attr(miri, ignore = "pure algebra on `Cell`: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn in_the_null_basis_the_idempotents_are_the_unit_vectors() {
    let sigma = Sigma::Hyperbolic;
    let p = Cell::<Dyadic>::idem_plus();
    let m = Cell::<Dyadic>::idem_minus();

    assert_eq!(p.upper(sigma), Some(Dyadic::ONE));
    assert_eq!(p.lower(sigma), Some(Dyadic::ZERO));
    assert_eq!(m.upper(sigma), Some(Dyadic::ZERO));
    assert_eq!(m.lower(sigma), Some(Dyadic::ONE));
}

/// `proj_realises_K`: projecting by `e+` keeps the upper channel and destroys the lower.
///
/// The K combinator, in the algebra: of two things given, return one and discard the other. This
/// is the operation `discards_iff_disc_pos` says exists only above zero.
#[cfg_attr(miri, ignore = "pure algebra on `Cell`: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn projection_keeps_one_channel_and_destroys_the_other() {
    let sigma = Sigma::Hyperbolic;
    let e = Cell::<Dyadic>::idem_plus();

    for u in -3..4 {
        for l in -3..4 {
            let z = Cell::<Dyadic>::of_halves(sigma, Dyadic::whole(u), Dyadic::whole(l))
                .expect("hyperbolic has halves");
            assert_eq!(z.upper(sigma), Some(Dyadic::whole(u)), "round-trip upper");
            assert_eq!(z.lower(sigma), Some(Dyadic::whole(l)), "round-trip lower");

            let kept = z.mul(sigma, e);
            assert_eq!(kept.upper(sigma), Some(Dyadic::whole(u)), "upper survives");
            assert_eq!(kept.lower(sigma), Some(Dyadic::ZERO), "lower is destroyed");
        }
    }
}

/// `idempotent_iff_trivial_of_disc_nonpos`: at or below zero, `e+` stops being idempotent.
///
/// The same cell, the same scalar, a different dial -- so the difference is the discriminant and
/// nothing else. This is the test that would fail if `Sigma` were decoration.
#[cfg_attr(miri, ignore = "pure algebra on `Cell`: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn the_same_cell_is_idempotent_only_above_zero() {
    let e = Cell::<Dyadic>::idem_plus();
    for sigma in FACES {
        let is_idem = e.mul(sigma, e) == e;
        assert_eq!(
            is_idem,
            sigma.discards(),
            "{sigma:?}: e+ is idempotent iff the discriminant is positive"
        );
    }
}

/// And the scope, stated as a compiling fact: the null basis needs halves too.
#[cfg_attr(miri, ignore = "pure algebra on `Cell`: no unsafe for Miri to check, and too slow to interpret")]
#[test]
fn the_null_basis_change_needs_halves() {
    // `of_halves` is on `Cell<R: Halve>`, so this is available over Dyadic...
    assert!(Cell::<Dyadic>::of_halves(Sigma::Hyperbolic, Dyadic::ONE, Dyadic::ZERO).is_some());
    // ...and returns None off the face that has no channels to convert.
    assert!(Cell::<Dyadic>::of_halves(Sigma::Parabolic, Dyadic::ONE, Dyadic::ZERO).is_none());
    assert!(Cell::<Dyadic>::of_halves(Sigma::Elliptic, Dyadic::ONE, Dyadic::ZERO).is_none());
}
