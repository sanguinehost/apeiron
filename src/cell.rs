//! `R[x]/(x^2 - sigma)`: one ordered pair, one dial.
//!
//! Every operation here takes the dial. That is the whole difference between implementing the
//! trichotomy and naming it: `Sigma` is an argument, the channel accessors exist only where the
//! channels do, and a discard cannot be constructed on a face that has none.
//!
//! Theorem names in the doc comments resolve against `../sanguine-apeiron/proof/Apeiron/`.

use crate::scalar::{Halve, Ring};

/// The dial, by the sign of the discriminant.
///
/// Only the sign matters. `classification` proves every 2-dimensional unital commutative
/// associative real algebra is `mul sigma` for `sigma` in `{-1, 0, 1}`, so these three are the
/// whole family and not three cases picked out of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Sigma {
    /// `sigma < 0`. The complex numbers. No channel exists at all
    /// (`Channels.no_channel_of_disc_neg`), the only idempotents are `0` and `1`
    /// (`Idempotents.idempotent_iff_trivial_of_disc_nonpos`), and nothing can be discarded.
    /// Every non-zero cell is invertible: this is the face that loses nothing.
    Elliptic,
    /// `sigma = 0`. The dual numbers. Exactly one channel, the first slot
    /// (`Channels.channel_iff_fst_of_disc_zero`). The second slot is nilpotent -- carried, never
    /// readable as an independent lane.
    #[default]
    Parabolic,
    /// `sigma > 0`. The split-complex numbers. Two independent channels
    /// (`Channels.two_channels_iff_disc_pos`), a nontrivial idempotent
    /// (`Idempotents.exists_nontrivial_idempotent_iff_disc_pos`), and -- the one that matters for
    /// a table -- **a projection that keeps one thing and destroys another**
    /// (`Projection.discards_iff_disc_pos`).
    Hyperbolic,
}

impl Sigma {
    /// The dial as a number: `-1`, `0`, `1`.
    #[must_use]
    pub const fn value(self) -> i64 {
        match self {
            Self::Elliptic => -1,
            Self::Parabolic => 0,
            Self::Hyperbolic => 1,
        }
    }

    /// How many independent channels this face has: 0, 1 or 2.
    ///
    /// A channel is a ring map to the base -- a lane that multiplies componentwise and so can be
    /// accumulated without coordinating with the other. This count is what decides whether work
    /// can be split in two and merged for free.
    #[must_use]
    pub const fn channels(self) -> u8 {
        match self {
            Self::Elliptic => 0,
            Self::Parabolic => 1,
            Self::Hyperbolic => 2,
        }
    }

    /// Whether a projection exists that keeps one thing and destroys another.
    ///
    /// `Projection.discards_iff_disc_pos`: discard is available **only** above zero. Every lossy
    /// operation a table performs -- collapsing a key to a fold, culling by interest, demoting a
    /// level of detail -- is a discard, so each of them is a hyperbolic operation and is refused
    /// on the other two faces rather than silently permitted.
    #[must_use]
    pub const fn discards(self) -> bool {
        matches!(self, Self::Hyperbolic)
    }
}

/// An ordered pair holding two values, each readable back independently.
///
/// `Apeiron/Emergence.TwoSlot`: one thing holding two. Generic over the scalar because which ring
/// the slots live in decides what the cell can hold -- `i64` is the fast lattice, `Dyadic` holds
/// the idempotents. See [`crate::scalar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Cell<R> {
    /// The first slot: the real part, and the one a projection keeps.
    pub a: R,
    /// The second slot: the coefficient of `x`, whose square is `sigma`.
    pub b: R,
}

impl<R: Ring> Cell<R> {
    #[must_use]
    /// The cell `a + b*x`.
    pub const fn new(a: R, b: R) -> Self {
        Self { a, b }
    }

    /// From two whole numbers, whatever the scalar.
    #[must_use]
    pub fn of(a: i64, b: i64) -> Self {
        Self {
            a: R::from_i64(a),
            b: R::from_i64(b),
        }
    }

    /// The multiplicative unit, `(1, 0)`, on every face.
    #[must_use]
    pub fn one() -> Self {
        Self {
            a: R::ONE,
            b: R::ZERO,
        }
    }

    #[must_use]
    /// The additive identity.
    pub fn zero() -> Self {
        Self {
            a: R::ZERO,
            b: R::ZERO,
        }
    }

    /// `(a1 a2 + sigma b1 b2, a1 b2 + b1 a2)`.
    ///
    /// Temporal and spatial O(1): a fixed count of scalar `mul`/`add` on the two slots.
    ///
    /// `Trichotomy.Basic.mul`, and it is commutative and associative on every face
    /// (`mul_comm`, `mul_assoc`) -- which is what licenses folding a list of these in any order.
    #[must_use]
    pub fn mul(self, sigma: Sigma, other: Self) -> Self {
        let s = R::from_i64(sigma.value());
        Self {
            a: self.a.mul(other.a).add(s.mul(self.b).mul(other.b)),
            b: self.a.mul(other.b).add(self.b.mul(other.a)),
        }
    }

    /// Componentwise addition. Named rather than the `Add` operator because `mul` cannot be
    /// `Mul` -- it takes the dial -- and one of the pair being an operator and the other a method
    /// would read as though they were different kinds of thing.
    #[expect(
        clippy::should_implement_trait,
        reason = "`mul` cannot be `Mul` because it takes the dial, and a pair where one half is \
                  an operator and the other a method reads as two different kinds of thing"
    )]
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self {
            a: self.a.add(other.a),
            b: self.b.add(other.b),
        }
    }

    /// `a^2 - sigma b^2`. `Trichotomy.Basic.normForm`.
    ///
    /// Temporal and spatial O(1): a fixed count of scalar `mul`/`sub` on the two slots.
    ///
    /// Multiplicative (`normForm_mul`), so it is zero exactly on the non-invertible cells. Above
    /// zero it factors as the product of the two channels
    /// (`NullBasis.normForm_eq_upper_mul_lower`), so a cell is degenerate exactly when one lane
    /// has gone to zero -- which is the precise sense in which a lossy fold sits on the null cone.
    #[must_use]
    pub fn norm_form(self, sigma: Sigma) -> R {
        let s = R::from_i64(sigma.value());
        self.a.mul(self.a).sub(s.mul(self.b).mul(self.b))
    }

    /// The upper channel, `a + b`. Hyperbolic only.
    ///
    /// `NullBasis.upper_mul`: `upper(z * w) = upper(z) * upper(w)`. Returns `None` off the
    /// hyperbolic face because there is no second channel to be the upper one -- the type refuses
    /// what `no_channel_of_disc_neg` and `channel_iff_fst_of_disc_zero` say does not exist.
    #[must_use]
    pub fn upper(self, sigma: Sigma) -> Option<R> {
        match sigma {
            Sigma::Hyperbolic => Some(self.a.add(self.b)),
            Sigma::Parabolic | Sigma::Elliptic => None,
        }
    }

    /// The lower channel, `a - b`. Hyperbolic only. `NullBasis.lower_mul`.
    #[must_use]
    pub fn lower(self, sigma: Sigma) -> Option<R> {
        match sigma {
            Sigma::Hyperbolic => Some(self.a.sub(self.b)),
            Sigma::Parabolic | Sigma::Elliptic => None,
        }
    }

    /// The one channel a parabolic cell has: the first slot.
    /// `Channels.channel_iff_fst_of_disc_zero`.
    #[must_use]
    pub fn channel(self, sigma: Sigma) -> Option<R> {
        match sigma {
            Sigma::Parabolic => Some(self.a),
            Sigma::Hyperbolic | Sigma::Elliptic => None,
        }
    }
}

/// Fold a list of cells into one.
///
/// Temporal O(cells.len()): one `mul` per element, each a fixed number of scalar operations
/// (O(1)). Spatial O(1): a single accumulator, no allocation.
///
/// Permutation-invariant on **every** face, which is the licence for accumulating from many
/// threads and combining at a sync point with no ordering.
///
/// `Gates.fold_perm` states this with `0 <= sigma`, and that hypothesis is an artefact of how it
/// is proved -- through the channel decomposition, which needs a non-negative discriminant to
/// have channels at all. The fact does not need it: order-freedom follows from `mul_comm` and
/// `mul_assoc`, and both hold for every `sigma` with no hypothesis. Measured over all 120
/// permutations of five cells on each of the three faces, elliptic included
/// (`tests/three_and_one.rs`).
///
/// This matters for what a table can be. Order-free accumulation is not a hyperbolic privilege
/// that the merge has to earn -- it is available everywhere, so the merge does not vary with the
/// dial. What varies with the dial is the discard, which is the *keystore*. One merge, three
/// keystores, and the asymmetry is the algebra's rather than a design choice.
#[must_use]
pub fn fold<R: Ring>(sigma: Sigma, cells: &[Cell<R>]) -> Cell<R> {
    cells.iter().fold(Cell::one(), |acc, c| acc.mul(sigma, *c))
}

/// Whether folding is order-free on this face. It is, on all of them.
///
/// Kept as a function rather than deleted because the answer is a claim worth being able to point
/// at: it was `!matches!(sigma, Elliptic)` first, transcribing `fold_perm`'s hypothesis as though
/// it bounded the fact.
#[must_use]
pub const fn fold_is_order_free(_sigma: Sigma) -> bool {
    true
}

impl<R: Halve> Cell<R> {
    /// `e+ = (1 + j)/2`, the idempotent that projects onto the upper channel.
    ///
    /// `Idempotents.idemPlus`, and `exists_nontrivial_idempotent_iff_disc_pos` says it is
    /// nontrivial exactly above zero. Its existence needs a half, which is why this lives behind
    /// [`Halve`]: over `i64` the construction does not compile, and that is the theorem's scope
    /// enforced by the type rather than described in a comment.
    #[must_use]
    pub fn idem_plus() -> Self {
        Self {
            a: R::ONE.halve(),
            b: R::ONE.halve(),
        }
    }

    /// `e- = (1 - j)/2`, projecting onto the lower channel. `Idempotents.idemMinus`.
    #[must_use]
    pub fn idem_minus() -> Self {
        Self {
            a: R::ONE.halve(),
            b: R::ZERO.sub(R::ONE.halve()),
        }
    }

    /// Back from the null basis: `a = (u + l)/2`, `b = (u - l)/2`. `NullBasis.ofHalves`.
    ///
    /// Only meaningful above zero, where the two channels exist; `None` elsewhere, matching
    /// [`Cell::upper`].
    #[must_use]
    pub fn of_halves(sigma: Sigma, upper: R, lower: R) -> Option<Self> {
        match sigma {
            Sigma::Hyperbolic => Some(Self {
                a: upper.add(lower).halve(),
                b: upper.sub(lower).halve(),
            }),
            Sigma::Parabolic | Sigma::Elliptic => None,
        }
    }
}
