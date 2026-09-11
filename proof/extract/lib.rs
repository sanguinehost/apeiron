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

/// `Cell::<i64>::mul`, rooted so Charon does not have to translate the whole crate.
pub fn word_mul(a: cell::Cell<i64>, sigma: cell::Sigma, b: cell::Cell<i64>) -> cell::Cell<i64> {
    a.mul(sigma, b)
}

/// `Cell::<i64>::norm_form`, rooted for the same reason.
pub fn word_norm(a: cell::Cell<i64>, sigma: cell::Sigma) -> i64 {
    a.norm_form(sigma)
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
