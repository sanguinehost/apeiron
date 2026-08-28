//! Where the three storage faces come from.
//!
//! A table has to answer one question about every key: **given the address, what of the key
//! survives?** Write the key as a two-slot cell -- one slot the address, one slot whatever the
//! address does not determine -- and the answer is a property of the dial, not a design choice.
//!
//! | face | channels | discard | what survives | the storage face |
//! |---|---|---|---|---|
//! | elliptic | 0 | no | everything; nothing is degenerate | **B** -- the address *is* the key |
//! | parabolic | 1 | no | one lane readable, one carried | **B/U** -- keep the key beside the address |
//! | hyperbolic | 2 | **yes** | one lane; the other is destroyed | **NB** -- keep the fold, lose the key |
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

use crate::cell::{Cell, Sigma};
use crate::scalar::Ring;

/// How a key's identity is held, once its address is known.
///
/// Each variant is the face whose discriminant licenses it, and [`Keystore::of`] is the whole
/// derivation: the face decides, nothing else does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keystore {
    /// **Being.** The address determines the key, so nothing is stored. Elliptic: no cell is
    /// degenerate, so no information can be lost on the way to an address.
    Being,
    /// **Being over Unbeing.** The address does not determine the key, and nothing may be thrown
    /// away, so both are carried. Parabolic: one readable channel and one nilpotent slot -- the
    /// second is present and cannot be read off the first.
    BeingOverUnbeing,
    /// **Non-Being.** One channel is kept and the other destroyed, so the key does not survive at
    /// all -- only its fold. Hyperbolic, and only there, because
    /// `Projection.discards_iff_disc_pos` says a discard exists nowhere else.
    NonBeing,
}

impl Keystore {
    /// The storage face this discriminant licenses. This is the derivation.
    #[must_use]
    pub const fn of(sigma: Sigma) -> Self {
        match sigma {
            Sigma::Elliptic => Self::Being,
            Sigma::Parabolic => Self::BeingOverUnbeing,
            Sigma::Hyperbolic => Self::NonBeing,
        }
    }

    /// The face it came from.
    #[must_use]
    pub const fn sigma(self) -> Sigma {
        match self {
            Self::Being => Sigma::Elliptic,
            Self::BeingOverUnbeing => Sigma::Parabolic,
            Self::NonBeing => Sigma::Hyperbolic,
        }
    }

    /// Whether a key must be stored alongside its address.
    ///
    /// Exactly the parabolic face: elliptic regenerates the key from the address, hyperbolic
    /// discards it, and only the middle face has to carry it.
    #[must_use]
    pub const fn stores_the_key(self) -> bool {
        matches!(self, Self::BeingOverUnbeing)
    }

    /// Whether the key survives at all.
    #[must_use]
    pub const fn key_survives(self) -> bool {
        !matches!(self, Self::NonBeing)
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
/// - Being: the whole cell, because nothing was thrown away.
/// - `BeingOverUnbeing`: the whole cell, because both slots are carried.
/// - `NonBeing`: the cell projected onto the upper channel, its lower lane destroyed.
///
/// The third case is the only one that loses anything, and it is the only one whose face has a
/// discard. The `2e+` form `(1, 1)` is used rather than the exact idempotent so this works over
/// `i64` as well as over an exact scalar; it scales the survivor by two and kills the other lane,
/// which is the same projection up to that factor.
#[must_use]
pub fn after_storing<R: Ring>(face: Keystore, key: Cell<R>) -> Cell<R> {
    match face {
        Keystore::Being | Keystore::BeingOverUnbeing => key,
        Keystore::NonBeing => key.mul(Sigma::Hyperbolic, Cell::of(1, 1)),
    }
}
