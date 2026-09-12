//! Extraction root for Aeneas.
//!
//! This is a development-only translation unit, not a crate in the Apeiron workspace.
//! Charon drives rustc over this single file; the modules it pulls in by `#[path]` are the
//! real `src/scalar.rs`, `src/cell.rs` and `src/exchange.rs`, so the extracted definitions track
//! the source the library builds. The wrappers below exist only to give Charon concrete,
//! monomorphic roots and to call the real methods; they do not restate any formula.
//!
//! Regenerate with `proof/regen.sh`.

#[path = "../../src/scalar.rs"]
pub mod scalar;
#[path = "../../src/cell.rs"]
pub mod cell;
#[path = "../../src/exchange.rs"]
pub mod exchange;
#[path = "../../src/carrier.rs"]
pub mod carrier;

/// `Cell::<i64>::mul`, rooted so Charon does not have to translate the whole crate.
pub fn word_mul(a: cell::Cell<i64>, sigma: cell::Sigma, b: cell::Cell<i64>) -> cell::Cell<i64> {
    a.mul(sigma, b)
}

/// `Cell::<i64>::norm_form`, rooted for the same reason.
pub fn word_norm(a: cell::Cell<i64>, sigma: cell::Sigma) -> i64 {
    a.norm_form(sigma)
}

/// `carrier::point_density`, the phase-cell density, rooted at the concrete `i64` scalar.
///
/// This calls the real `src/carrier.rs` generic kernel with `R = i64`; the `Ring` instance is the
/// same `i64` instance `word_mul`/`word_norm` use, so the extracted `carrier.point_density` is the
/// production body rather than a restatement.
pub fn word_point_density(sigma: cell::Sigma, amplitude: i64, direction: cell::Cell<i64>) -> i64 {
    carrier::point_density(sigma, amplitude, direction)
}

/// `carrier::point_flux`, the phase-cell flux, rooted at the concrete `i64` scalar.
pub fn word_point_flux(
    sigma: cell::Sigma,
    amplitude: i64,
    direction: cell::Cell<i64>,
    previous: i64,
    next: i64,
) -> i64 {
    carrier::point_flux(sigma, amplitude, direction, previous, next)
}

/// `carrier::conservative_density_step`, the exact checked scalar update, at the concrete `i64`
/// scalar. This calls the real `src/carrier.rs` body; it does not restate the formula.
pub fn word_conservative_density_step(
    density: i64,
    incoming: i64,
    outgoing: i64,
    source: i64,
) -> Option<i64> {
    carrier::conservative_density_step(density, incoming, outgoing, source)
}

/// `carrier::diffusive_flux_half`, the exact checked face half-flux, at the concrete `i64` scalar.
pub fn word_diffusive_flux_half(left: i64, right: i64) -> Option<i64> {
    carrier::diffusive_flux_half(left, right)
}

/// The derived impulse `p - q`, rooted at the concrete `i64` scalar.
pub fn word_exchange_impulse(p: cell::Cell<i64>, q: cell::Cell<i64>) -> cell::Cell<i64> {
    exchange::exchange_impulse(p, q)
}

/// The equal-mass elastic exchange, through the derived impulse.
pub fn word_elastic_exchange(p: cell::Cell<i64>, q: cell::Cell<i64>) -> exchange::Exchange<i64> {
    exchange::elastic_exchange(p, q)
}

/// The checked-claim kernel, rooted at the concrete `i64` scalar.
///
/// `accept_derived_impulse` compares the claim against the derived impulse with an explicit `if`,
/// so Charon translates the branch and the derived `Cell` equality rather than the external
/// `core.bool.Bool.then` combinator. `regen.sh`'s audit still rejects any generated `axiom`.
pub fn word_accept_derived_impulse(
    p: cell::Cell<i64>,
    q: cell::Cell<i64>,
    claimed: cell::Cell<i64>,
) -> Option<exchange::Exchange<i64>> {
    exchange::accept_derived_impulse(p, q, claimed)
}
