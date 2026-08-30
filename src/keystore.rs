//! Where the three storage faces come from.
//!
//! A table has to answer one question about every key: **given the address, what of the key
//! survives?** Write the key as a two-slot cell -- one slot the address, one slot whatever the
//! address does not determine -- and the answer is a property of the dial, not a design choice.
//!
//! | face | channels | discard | what survives | the storage it licenses |
//! |---|---|---|---|---|
//! | elliptic | 0 | no | everything; nothing is degenerate | the address *is* the key |
//! | parabolic | 1 | no | one lane readable, one carried | keep the key beside the address |
//! | hyperbolic | 2 | **yes** | one lane; the other is destroyed | keep the fold, lose the key |
//!
//! The count is not chosen. `classification` proves every 2-dimensional unital commutative
//! associative real algebra is `mul sigma` for `sigma` in `{-1, 0, 1}`, so there are exactly three
//! faces, hence exactly three storage strategies -- and a fourth would have to be a fourth
//! algebra, which the classification says does not exist.
//!
//! The merge does not appear in that table, and that is the other half of the result. Order-free
//! accumulation needs commutativity and associativity, which hold on **every** face with no
//! hypothesis (see [`crate::cell::fold`]). So the merge does not vary with the dial while the
//! keystore does: **one merge, three keystores**, and the asymmetry is the algebra's.
//!
//! # There is no second enum, and there used to be
//!
//! This module carried a `Keystore` enum -- `Being`, `BeingOverUnbeing`, `NonBeing` -- with
//! `Keystore::of(sigma)` and `Keystore::sigma()` as its derivation, and a test proving those two
//! were mutually inverse. A type isomorphic to [`Sigma`] is not a derivation from `Sigma`; it is
//! `Sigma` under different nouns, and the nouns were **`hashtrinity`'s** -- its `freq.rs` names
//! "the NB (Non-Being) face of the storage trinity". Importing the ontology of the crate whose
//! stated defect is that its trichotomy is vocabulary is the one mistake this crate cannot make.
//!
//! So the face *is* the discriminant. `Sigma` is what the operations take, the predicates below
//! are properties of `Sigma`, and there is nothing left to relabel.

use crate::cell::{Cell, Sigma};
use crate::scalar::Ring;

impl Sigma {
    /// Whether a key must be stored alongside its address.
    ///
    /// Exactly the parabolic face: elliptic regenerates the key from the address, hyperbolic
    /// discards it, and only the middle face has to carry it.
    #[must_use]
    pub const fn stores_the_key(self) -> bool {
        matches!(self, Self::Parabolic)
    }

    /// Whether the key survives at all.
    ///
    /// False only on the hyperbolic face, which is the only one with a discard
    /// (`Projection.discards_iff_disc_pos`).
    #[must_use]
    pub const fn key_survives(self) -> bool {
        !matches!(self, Self::Hyperbolic)
    }
}

/// A key, as a cell: the address in one slot, what the address does not determine in the other.
///
/// This is the reading the whole module rests on, and it is a reading -- the algebra does not
/// mention tables. What it does is fix the shape: two slots, and a dial deciding what a projection
/// can do to them.
#[must_use]
pub fn key_cell<R: Ring>(address: R, residue: R) -> Cell<R> {
    Cell::new(address, residue)
}

/// What is left of a key after storing it on this face.
///
/// - Elliptic: the whole cell, because nothing was thrown away.
/// - Parabolic: the whole cell, because both slots are carried.
/// - Hyperbolic: the cell projected onto the upper channel, its lower lane destroyed.
///
/// The third case is the only one that loses anything, and it is the only one whose face has a
/// discard. The `2e+` form `(1, 1)` is used rather than the exact idempotent so this works over
/// `i64` as well as over an exact scalar; it scales the survivor by two and kills the other lane,
/// which is the same projection up to that factor.
#[must_use]
pub fn after_storing<R: Ring>(face: Sigma, key: Cell<R>) -> Cell<R> {
    match face {
        Sigma::Elliptic | Sigma::Parabolic => key,
        Sigma::Hyperbolic => key.mul(Sigma::Hyperbolic, Cell::of(1, 1)),
    }
}
