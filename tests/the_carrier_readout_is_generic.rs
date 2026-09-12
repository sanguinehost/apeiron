//! The carrier readout is one generic kernel, not a third formula.
//!
//! [`point_density`] and [`point_flux`] are the Rust mirror of
//! `Apeiron.Physics.CarrierAnyRing`, and the machine-word readouts are their unit-direction
//! instantiation. This file checks the structural identity, the dial independence at the unit
//! direction, and the exact integer values the carrier domain relies on.

use apeiron::carrier::{point_density, point_flux};
use apeiron::cell::{Cell, Sigma};
use apeiron::scalar::Ring;

const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

/// `pointDensity σ a u = norm_form σ ((a, 0) * u)`, at directions that are not just the unit.
#[test]
fn density_is_the_norm_form_of_the_stretched_cell() {
    let directions = [
        Cell::new(1i64, 0),
        Cell::new(0, 1),
        Cell::new(2, -3),
        Cell::new(-4, 5),
    ];
    for sigma in FACES {
        for a in -6i64..=6 {
            for u in directions {
                let expected = Cell::new(a, 0).mul(sigma, u).norm_form(sigma);
                assert_eq!(point_density(sigma, a, u), expected, "sigma {sigma:?}");
            }
        }
    }
}

/// `pointFlux σ a u p n = pointDensity σ a u * (n - p)`, with the forward order retained.
#[test]
fn flux_is_the_density_times_the_forward_difference() {
    let u = Cell::new(2i64, -3);
    for sigma in FACES {
        for a in -6i64..=6 {
            for p in -3i64..=3 {
                for n in -3i64..=3 {
                    let expected = point_density(sigma, a, u).mul(n.sub(p));
                    assert_eq!(point_flux(sigma, a, u, p, n), expected);
                }
            }
        }
    }
}

/// At the unit direction the density is the amplitude's square on every face, so the dial a
/// machine-word kernel fixes is immaterial.
#[test]
fn the_unit_direction_density_is_the_amplitude_square_on_every_face() {
    for sigma in FACES {
        for a in -10i64..=10 {
            assert_eq!(point_density(sigma, a, Cell::one()), a * a);
        }
    }
}

/// The concrete carrier domain (`u16` amplitudes, `i16` phases) lands on the machine-word values
/// without wrapping, which is the no-wrap premise the Lean `site_momentum_r22_no_wrap` records.
#[test]
fn the_carrier_domain_matches_the_machine_word_readout() {
    for amplitude in [0u16, 1, 3, 1024, 65535] {
        let a = i64::from(amplitude);
        for (p, n) in [(0i16, 0i16), (0, 2), (2, 0), (-2, 2), (2, -2), (-2, -2)] {
            let (pi, ni) = (i64::from(p), i64::from(n));
            assert_eq!(point_density(Sigma::Parabolic, a, Cell::one()), a * a);
            assert_eq!(
                point_flux(Sigma::Parabolic, a, Cell::one(), pi, ni),
                a * a * (ni - pi)
            );
        }
    }
}
