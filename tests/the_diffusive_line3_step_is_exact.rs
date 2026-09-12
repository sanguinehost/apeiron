//! The three-cell diffusive step is exact, synchronous and transactional.
//!
//! `diffusive_line3_step` is the smallest closed line: two interior faces read once each from the
//! old state, two endpoint faces fixed at zero, and three conservative cell updates. What is at
//! stake is that the composition does not introduce a second rounding or a partial result: an odd
//! interior difference or an unrepresentable intermediate must refuse the whole line, and the
//! shared faces must move density down the gradient without creating it.
//!
//! The tests assert concrete values and conservation, not a restatement of the production
//! composition.

use apeiron::carrier::{conservative_density_step, diffusive_flux_half, diffusive_line3_step};
use apeiron::DensityLine3;

/// The zero source and the zero line.
fn zero() -> DensityLine3 {
    DensityLine3 {
        left: 0,
        center: 0,
        right: 0,
    }
}

/// The sum of the three cells.
fn total(line: DensityLine3) -> i64 {
    line.left + line.center + line.right
}

/// A level line moves by exactly its source: the two interior faces are zero.
#[test]
fn a_level_line_moves_by_exactly_its_source() {
    let old = DensityLine3 {
        left: 5,
        center: 5,
        right: 5,
    };
    let source = DensityLine3 {
        left: 1,
        center: -2,
        right: 3,
    };
    let next = diffusive_line3_step(old, source).expect("a level line has zero faces");
    assert_eq!(
        next,
        DensityLine3 {
            left: 6,
            center: 3,
            right: 8,
        }
    );
    assert_eq!(next.left - old.left, source.left);
    assert_eq!(next.center - old.center, source.center);
    assert_eq!(next.right - old.right, source.right);
}

/// A nontrivial spike levels: `(0, 4, 0)` becomes `(2, 0, 2)` at `nu = 1/2`.
#[test]
fn a_closed_spike_levels_toward_its_neighbors() {
    let old = DensityLine3 {
        left: 0,
        center: 4,
        right: 0,
    };
    let next = diffusive_line3_step(old, zero()).expect("even faces");
    assert_eq!(
        next,
        DensityLine3 {
            left: 2,
            center: 0,
            right: 2,
        }
    );
    assert!(next.center < old.center);
    assert!(next.left > old.left && next.right > old.right);
}

/// With no source, the closed line conserves the total exactly.
#[test]
fn a_closed_line_with_no_source_conserves_the_total() {
    let old = DensityLine3 {
        left: 7,
        center: 3,
        right: 11,
    };
    let next = diffusive_line3_step(old, zero()).expect("even faces");
    assert_eq!(
        next,
        DensityLine3 {
            left: 5,
            center: 9,
            right: 7,
        }
    );
    assert_eq!(total(next), total(old));
}

/// The shared face moves density without creating it: the source is the only net addition.
#[test]
fn the_source_is_the_only_net_addition() {
    let old = DensityLine3 {
        left: 10,
        center: 20,
        right: 30,
    };
    let source = DensityLine3 {
        left: 1,
        center: -2,
        right: 3,
    };
    let next = diffusive_line3_step(old, source).expect("even faces");
    assert_eq!(total(next), total(old) + total(source));
}

/// An odd interior difference has no exact half, so the whole line is refused rather than rounded.
#[test]
fn an_odd_interior_difference_refuses_the_whole_line() {
    // The first face `(0, 1)` is odd.
    assert_eq!(
        diffusive_line3_step(
            DensityLine3 {
                left: 0,
                center: 1,
                right: 0,
            },
            zero(),
        ),
        None
    );
    // The first face is level; the second face `(0, 1)` is odd.
    assert_eq!(
        diffusive_line3_step(
            DensityLine3 {
                left: 0,
                center: 0,
                right: 1,
            },
            zero(),
        ),
        None
    );
}

/// An unrepresentable face difference or cell intermediate refuses the whole line.
#[test]
fn an_unrepresentable_intermediate_refuses_the_whole_line() {
    // `i64::MIN - i64::MAX` does not fit, so the first face does not exist.
    assert_eq!(
        diffusive_line3_step(
            DensityLine3 {
                left: i64::MAX,
                center: i64::MIN,
                right: 0,
            },
            zero(),
        ),
        None
    );

    // Both faces exist and the left and center updates fit; the right update overflows.
    let old = DensityLine3 {
        left: 0,
        center: i64::MIN,
        right: i64::MIN,
    };
    assert_eq!(
        diffusive_line3_step(
            old,
            DensityLine3 {
                left: 0,
                center: 0,
                right: -1,
            },
        ),
        None
    );
    // The same line with the failing source removed succeeds, so the refusal above is the right
    // cell and not an earlier one.
    let next = diffusive_line3_step(old, zero()).expect("the faces exist");
    let half = 1i64 << 62;
    assert_eq!(
        next,
        DensityLine3 {
            left: -half,
            center: -half,
            right: i64::MIN,
        }
    );
}

/// A late refusal leaves no partial result: the failing line above returns `None`, not the two
/// successful updates that preceded the failure.
#[test]
fn a_late_refusal_is_transactional() {
    let old = DensityLine3 {
        left: 0,
        center: i64::MIN,
        right: i64::MIN,
    };
    let failing = diffusive_line3_step(
        old,
        DensityLine3 {
            left: 0,
            center: 0,
            right: -1,
        },
    );
    assert!(failing.is_none());
    // The left and center updates that the kernel computed before the failure are not observable.
    let succeeding = diffusive_line3_step(old, zero()).expect("faces exist");
    let half = 1i64 << 62;
    assert_eq!(succeeding.left, -half);
    assert_eq!(succeeding.center, -half);
}

/// Orientation negative control: on an asymmetric line the two faces are not interchangeable. The
/// kernel runs down the gradient (`(0, 8, 4)` to `(4, 2, 6)`); feeding the negated faces back
/// sharpens the line instead, which is the anti-diffusive answer the kernel does not produce.
#[test]
fn the_orientation_is_down_gradient_not_anti_diffusive() {
    let old = DensityLine3 {
        left: 0,
        center: 8,
        right: 4,
    };
    let next = diffusive_line3_step(old, zero()).expect("even faces");
    assert_eq!(
        next,
        DensityLine3 {
            left: 4,
            center: 2,
            right: 6,
        }
    );

    // The same two face values with the opposite sign, assembled by hand, sharpens the spike.
    let face_left = diffusive_flux_half(old.left, old.center).unwrap();
    let face_right = diffusive_flux_half(old.center, old.right).unwrap();
    assert_eq!((face_left, face_right), (-4, 2));
    let anti = DensityLine3 {
        left: conservative_density_step(old.left, 0, -face_left, 0).unwrap(),
        center: conservative_density_step(old.center, -face_left, -face_right, 0).unwrap(),
        right: conservative_density_step(old.right, -face_right, 0, 0).unwrap(),
    };
    assert_eq!(
        anti,
        DensityLine3 {
            left: -4,
            center: 14,
            right: 2,
        }
    );
    assert!(anti.center - anti.left > old.center - old.left);
}

/// A multi-step run eventually refuses: `(0, 4, 0)` levels twice and the third step meets an odd
/// interior difference, so the kernel is not closed over repeated application.
#[test]
fn a_multi_step_run_eventually_refuses() {
    let zero = zero();
    let first = diffusive_line3_step(
        DensityLine3 {
            left: 0,
            center: 4,
            right: 0,
        },
        zero,
    )
    .expect("first step");
    assert_eq!(
        first,
        DensityLine3 {
            left: 2,
            center: 0,
            right: 2,
        }
    );
    let second = diffusive_line3_step(first, zero).expect("second step");
    assert_eq!(
        second,
        DensityLine3 {
            left: 1,
            center: 2,
            right: 1,
        }
    );
    assert_eq!(diffusive_line3_step(second, zero), None);
}
