//! The carrier step is exact, checked and conservative.
//!
//! `conservative_density_step` and `diffusive_flux_half` are the nondimensional integer kernel
//! for one carrier update: whole units move between a cell and its faces, an unrepresentable
//! result is refused rather than wrapped, and a half-unit diffusive flux exists only when the
//! difference is exactly even. What is at stake here is the difference between a refusal and a
//! silently wrong number: every edge below is a place where wrapping or rounding would have
//! produced a plausible value.
//!
//! The tests assert concrete values and conservation properties, not a restatement of the
//! production formula.

use apeiron::carrier::{conservative_density_step, diffusive_flux_half};

/// The conservative update is `density - (outgoing - incoming) + source`, signed throughout.
#[test]
fn a_step_balances_incoming_outgoing_and_source() {
    assert_eq!(conservative_density_step(10, 3, 5, 2), Some(10)); // 10 - 2 + 2
    assert_eq!(conservative_density_step(10, 3, 5, 7), Some(15)); // 10 - 2 + 7
    assert_eq!(conservative_density_step(10, 5, 3, 0), Some(12)); // 10 + 2
    assert_eq!(conservative_density_step(10, 5, 3, -2), Some(10)); // 10 + 2 - 2
    assert_eq!(conservative_density_step(-10, -3, -5, -2), Some(-10));
}

/// The representable edges still produce an answer, including the cancellation that keeps the
/// intermediate net at zero.
#[test]
fn a_step_at_the_representable_edges() {
    assert_eq!(conservative_density_step(i64::MAX, 0, 0, 0), Some(i64::MAX));
    assert_eq!(conservative_density_step(i64::MIN, 0, 0, 0), Some(i64::MIN));
    assert_eq!(
        conservative_density_step(i64::MAX, i64::MAX, i64::MAX, 0),
        Some(i64::MAX)
    );
    assert_eq!(
        conservative_density_step(i64::MIN, i64::MIN, i64::MIN, 0),
        Some(i64::MIN)
    );
}

/// Every intermediate is checked, and a later overflow refuses the whole update rather than
/// leaving the earlier steps as a partial result.
#[test]
fn an_overflowing_step_is_refused() {
    // The net `outgoing - incoming` itself does not fit.
    assert_eq!(conservative_density_step(0, i64::MIN, i64::MAX, 0), None);
    assert_eq!(conservative_density_step(0, i64::MAX, i64::MIN, 0), None);

    // `density - net` does not fit, in either direction.
    assert_eq!(conservative_density_step(i64::MIN, 0, i64::MAX, 0), None);
    assert_eq!(conservative_density_step(i64::MAX, i64::MAX, 0, 0), None);

    // The earlier intermediates fit; only the final `+ source` overflows.
    assert_eq!(conservative_density_step(i64::MAX, 0, 0, 1), None);
    assert_eq!(conservative_density_step(i64::MIN, 0, 0, -1), None);
}

/// Two cells sharing one face: the flux that leaves one is exactly the flux that enters the
/// other, so the face moves density without creating it.
#[test]
fn a_shared_face_moves_density_without_creating_it() {
    let (a_before, b_before) = (100i64, 40i64);
    for flux in [-7i64, -1, 0, 1, 7] {
        let a_after = conservative_density_step(a_before, 0, flux, 0).unwrap();
        let b_after = conservative_density_step(b_before, flux, 0, 0).unwrap();
        assert_eq!(a_after - a_before, -flux, "flux {flux} left a");
        assert_eq!(b_after - b_before, flux, "flux {flux} entered b");
        assert_eq!(
            a_after + b_after,
            a_before + b_before,
            "the shared face changed the total"
        );
    }
}

/// Only the source adds to the total; the face terms cancel.
#[test]
fn the_source_is_the_only_net_addition() {
    let (a, b) = (10i64, 20i64);
    let a_after = conservative_density_step(a, 0, 0, 3).unwrap();
    let b_after = conservative_density_step(b, 0, 0, -1).unwrap();
    assert_eq!(a_after + b_after, a + b + 2);
}

/// The signed left-to-right half-flux runs down the gradient and vanishes on a level face.
#[test]
fn diffusion_runs_down_the_gradient_and_stops_when_level() {
    assert_eq!(diffusive_flux_half(5, 5), Some(0));
    assert_eq!(diffusive_flux_half(-3, -3), Some(0));
    assert_eq!(diffusive_flux_half(0, 4), Some(-2)); // right denser, flows left
    assert_eq!(diffusive_flux_half(4, 0), Some(2)); // left denser, flows right
    assert_eq!(diffusive_flux_half(2, 4), Some(-1)); // one whole face share
}

/// A stable two-cell fixture: a spike levels out and the total is unchanged.
#[test]
fn a_diffusive_step_moves_a_spike_toward_level() {
    let (left, right) = (10i64, 4i64);
    let flux = diffusive_flux_half(left, right).unwrap();
    assert_eq!(flux, 3);
    let left_after = conservative_density_step(left, 0, flux, 0).unwrap();
    let right_after = conservative_density_step(right, flux, 0, 0).unwrap();
    assert_eq!((left_after, right_after), (7, 7));
    assert_eq!(left_after + right_after, left + right);
}

/// An odd difference has no exact half, so it is refused rather than rounded.
#[test]
fn an_odd_difference_has_no_exact_half() {
    assert_eq!(diffusive_flux_half(0, 1), None);
    assert_eq!(diffusive_flux_half(0, -1), None);
    assert_eq!(diffusive_flux_half(3, 4), None);
    assert_eq!(diffusive_flux_half(-4, -3), None);
    // An even difference next to the same edge does have a half.
    assert_eq!(diffusive_flux_half(i64::MAX, i64::MAX - 2), Some(1));
}

/// The difference itself is checked: an overlarge one is refused rather than wrapped.
#[test]
fn an_overlarge_difference_has_no_half() {
    assert_eq!(diffusive_flux_half(i64::MAX, i64::MIN), None);
    assert_eq!(diffusive_flux_half(i64::MIN, i64::MAX), None);
    assert_eq!(diffusive_flux_half(i64::MIN, 0), None);
    // The even extreme that does fit still halves exactly.
    assert_eq!(diffusive_flux_half(0, i64::MIN), Some(-(i64::MIN / 2)));
}

/// The exact half is not closed over `i64`, and the kernel negates the half rather than the raw
/// difference. At the extreme the difference `right - left = i64::MIN` is representable while its
/// negation `left - right = 2^63` is not, so a kernel that negated the difference first would
/// refuse a face that has a perfectly good half. This is the edge the source's `0 - half` handles.
#[test]
fn the_exact_half_negates_the_half_not_the_difference() {
    // `right - left` fits; `left - right` does not.
    assert_eq!(i64::MIN.checked_sub(0), Some(i64::MIN));
    assert_eq!(0i64.checked_sub(i64::MIN), None);
    // The half is `2^62`, and it is what the kernel returns.
    assert_eq!(diffusive_flux_half(0, i64::MIN), Some(1i64 << 62));
    // An odd difference has no exact half; an unrepresentable one is refused.
    assert_eq!(diffusive_flux_half(0, 1), None);
    assert_eq!(diffusive_flux_half(1, 0), None);
    assert_eq!(diffusive_flux_half(i64::MIN, i64::MAX), None);
}

/// Negative control: feeding the flux back with the opposite sign is anti-diffusive and sharpens
/// the fixture. The kernel does not forbid it; the sign convention is the caller's contract.
#[test]
fn an_anti_diffusive_orientation_sharpens_instead_of_levelling() {
    let (left, right) = (10i64, 4i64);
    let down = diffusive_flux_half(left, right).unwrap();
    assert_eq!(down, 3);
    let left_after = conservative_density_step(left, 0, -down, 0).unwrap();
    let right_after = conservative_density_step(right, -down, 0, 0).unwrap();
    assert_eq!((left_after, right_after), (13, 1));
    assert!(left_after - right_after > left - right);
}
