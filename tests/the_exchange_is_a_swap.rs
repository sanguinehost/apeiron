//! The canonical equal-mass exchange kernel: swap, derived impulse, acceptance, ledger.
//!
//! The Lean development is `Apeiron.Physics.CarrierBodyWork`. This is the Rust kernel it
//! mirrors: [`apeiron::exchange_impulse`] is `p - q`, [`apeiron::elastic_exchange`] performs
//! the equal-mass swap through that impulse, and [`apeiron::accept_derived_impulse`] admits a
//! claimed impulse only when it is exactly the derived one.
//!
//! SCOPE: small exact integers and the `i64` wrapping extremes. There are no positions, no
//! contact detection and no time integration. The norm-form ledger is conserved because the
//! exchange swaps the two cells; that is a tested accounting, not a Lean proof, and the
//! wrapping case is a ring identity over `Z/2^64` rather than a transfer of the real-number
//! theorem.

use apeiron::{accept_derived_impulse, elastic_exchange, exchange_impulse, Cell, Sigma};

/// The three dial settings, `-1`, `0` and `1`.
const FACES: [Sigma; 3] = [Sigma::Elliptic, Sigma::Parabolic, Sigma::Hyperbolic];

/// Small exact momenta, including pairs that change under the exchange.
const PAIRS: [(Cell<i64>, Cell<i64>); 3] = [
    (Cell::new(3, 1), Cell::new(1, 2)),
    (Cell::new(-4, 7), Cell::new(2, -5)),
    (Cell::new(0, 0), Cell::new(6, -1)),
];

/// The norm-form ledger: the carrier norm form plus the body norm form.
fn norm_form_ledger(sigma: Sigma, p: Cell<i64>, q: Cell<i64>) -> i64 {
    p.norm_form(sigma).wrapping_add(q.norm_form(sigma))
}

#[test]
fn the_derived_impulse_is_the_difference() {
    for (p, q) in PAIRS {
        assert_eq!(
            exchange_impulse(p, q),
            Cell::new(p.a.wrapping_sub(q.a), p.b.wrapping_sub(q.b))
        );
        let forward = exchange_impulse(p, q);
        let backward = exchange_impulse(q, p);
        assert_eq!(forward.a, backward.a.wrapping_neg());
        assert_eq!(forward.b, backward.b.wrapping_neg());
    }
}

#[test]
fn the_exchange_swaps_the_two_momenta() {
    for (p, q) in PAIRS {
        let exchange = elastic_exchange(p, q);
        assert_eq!(exchange.impulse, exchange_impulse(p, q));
        assert_eq!(exchange.carrier_after, q);
        assert_eq!(exchange.body_after, p);
    }
}

#[test]
fn total_momentum_is_conserved() {
    for (p, q) in PAIRS {
        let exchange = elastic_exchange(p, q);
        assert_eq!(exchange.carrier_after.add(exchange.body_after), p.add(q));
    }
}

#[test]
fn the_norm_form_ledger_is_conserved_on_all_three_faces() {
    for sigma in FACES {
        for (p, q) in PAIRS {
            let exchange = elastic_exchange(p, q);
            assert_eq!(
                norm_form_ledger(sigma, exchange.carrier_after, exchange.body_after),
                norm_form_ledger(sigma, p, q)
            );
        }
    }
}

#[test]
fn the_exact_claim_is_accepted_and_others_are_rejected() {
    for (p, q) in PAIRS {
        let derived = exchange_impulse(p, q);
        assert_eq!(
            accept_derived_impulse(p, q, derived),
            Some(elastic_exchange(p, q))
        );

        // The Lean negative control, `(0, 1)`, is never the derived impulse here.
        assert_ne!(derived, Cell::new(0, 1));
        assert_eq!(accept_derived_impulse(p, q, Cell::new(0, 1)), None);

        // A one-slot near miss is rejected just as exactly.
        let near_miss = Cell::new(derived.a.wrapping_add(1), derived.b);
        assert_ne!(near_miss, derived);
        assert_eq!(accept_derived_impulse(p, q, near_miss), None);
    }
}

#[test]
fn the_exchange_swaps_even_at_the_wrapping_extremes() {
    let p = Cell::new(i64::MIN, i64::MAX);
    let q = Cell::new(i64::MAX, i64::MIN);

    let exchange = elastic_exchange(p, q);
    assert_eq!(exchange.impulse, Cell::new(1, -1));
    assert_eq!(exchange.carrier_after, q);
    assert_eq!(exchange.body_after, p);
    assert_eq!(
        accept_derived_impulse(p, q, exchange.impulse),
        Some(exchange)
    );
    assert_eq!(accept_derived_impulse(p, q, Cell::new(0, 0)), None);

    for sigma in FACES {
        assert_eq!(
            norm_form_ledger(sigma, exchange.carrier_after, exchange.body_after),
            norm_form_ledger(sigma, p, q),
            "the wrapping ledger moved at sigma = {}",
            sigma.value()
        );
    }
}
