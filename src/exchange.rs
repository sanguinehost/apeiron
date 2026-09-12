//! Equal-mass elastic exchange of two basis momenta.
//!
//! This is the Rust kernel for `Apeiron.Physics.CarrierBodyWork`: a local, instantaneous
//! exchange between a carrier momentum `p` and a body momentum `q`, both [`Cell`]s of the
//! two-slot algebra. The impulse is **derived** from the two momenta alone,
//! [`exchange_impulse(p, q)`](exchange_impulse) `= p - q`; no impulse is supplied
//! independently. The carrier removes the impulse and the body adds it, so the two momenta
//! swap exactly.
//!
//! # Scope
//!
//! Equal masses, one instantaneous local exchange, no positions, no contact detection and no
//! time integration. There is no dissipation, forcing or circulation. The norm-form ledger is
//! conserved because the exchange merely swaps the two cells; that conservation follows
//! mathematically from the swap and is not established by a runtime proof here.
//!
//! # What this proves, and what it does not
//!
//! The functions are generic over the crate's [`Ring`] contract. The type bound does not itself
//! prove that an implementation satisfies the ring laws, and a partial implementation such as
//! [`Dyadic`](crate::scalar::Dyadic) can panic when a result is unrepresentable. For lawful,
//! successful operations the swap, impulse antisymmetry and norm-form ledger are ring identities;
//! `Apeiron.Physics.CarrierBodyWorkAnyRing` states them over an arbitrary commutative ring. The
//! concrete `i64` implementation is **wrapping** (`Z/2^64`), and its full-width Aeneas refinement
//! is proved: the `apeironWordBridge` package (`proof/corpus/WordRefinement.lean`) proves
//! `word_exchange_impulse_denote` and `word_elastic_exchange_denote` against
//! `CarrierBodyWorkAnyRing` instantiated at `ZMod (2^64)`. That is a statement about the
//! Aeneas-extracted kernel; it establishes nothing about rustc/LLVM code generation or the
//! generated machine code.

use crate::cell::Cell;
use crate::scalar::Ring;

/// The derived impulse and the two momenta it produces.
///
/// Returned by [`elastic_exchange`] and [`accept_derived_impulse`]. All three fields are
/// computed from the derived impulse, never from a caller-supplied claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exchange<R> {
    /// The derived impulse `p - q`.
    pub impulse: Cell<R>,
    /// The carrier momentum after the exchange: `p - impulse`.
    pub carrier_after: Cell<R>,
    /// The body momentum after the exchange: `q + impulse`.
    pub body_after: Cell<R>,
}

/// The derived impulse, `p - q`.
///
/// This is the whole authority: no impulse parameter is supplied. The subtraction is the
/// crate [`Ring`] subtraction, which wraps for the `i64` scalar.
#[must_use]
pub fn exchange_impulse<R: Ring>(p: Cell<R>, q: Cell<R>) -> Cell<R> {
    Cell::new(p.a.sub(q.a), p.b.sub(q.b))
}

/// The equal-mass elastic exchange, computed through the derived impulse.
///
/// The carrier ends at `q` and the body ends at `p`. See the module documentation for the
/// scope and for the real-number versus wrapping distinction.
#[must_use]
pub fn elastic_exchange<R: Ring>(p: Cell<R>, q: Cell<R>) -> Exchange<R> {
    let impulse = exchange_impulse(p, q);
    Exchange {
        impulse,
        carrier_after: Cell::new(p.a.sub(impulse.a), p.b.sub(impulse.b)),
        body_after: Cell::new(q.a.add(impulse.a), q.b.add(impulse.b)),
    }
}

/// Accept a claimed impulse only when it is exactly the derived `p - q`.
///
/// On acceptance the returned [`Exchange`] is recomputed from the derived impulse, so the
/// result cannot depend on the claim beyond its equality with the derived value. On rejection
/// this returns `None` and produces no state. Scalar arithmetic can still panic when the selected
/// [`Ring`] implementation is partial, so this is transactional claim checking, not a general
/// panic-free arithmetic API.
#[must_use]
pub fn accept_derived_impulse<R: Ring>(
    p: Cell<R>,
    q: Cell<R>,
    claimed: Cell<R>,
) -> Option<Exchange<R>> {
    if claimed == exchange_impulse(p, q) {
        Some(elastic_exchange(p, q))
    } else {
        None
    }
}
