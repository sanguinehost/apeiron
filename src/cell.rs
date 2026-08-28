//! `R[x]/(x^2 - sigma)`: one ordered pair, one dial.
//!
//! Every operation here takes the dial. That is the whole difference between implementing the
//! trichotomy and naming it: `Sigma` is an argument, the channel accessors exist only where the
//! channels do, and a discard cannot be constructed on a face that has none.
//!
//! Theorem names in the doc comments resolve against `../sanguine-apeiron/proof/Apeiron/`.

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
/// `Apeiron/Emergence.TwoSlot`: one thing holding two. The algebra realises it above zero and the
/// combinator calculus realises it too, which is why this type carries both readings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Cell {
    pub a: i64,
    pub b: i64,
}

impl Cell {
    #[must_use]
    pub const fn new(a: i64, b: i64) -> Self {
        Self { a, b }
    }

    /// The multiplicative unit, `(1, 0)`, on every face.
    #[must_use]
    pub const fn one() -> Self {
        Self { a: 1, b: 0 }
    }

    #[must_use]
    pub const fn zero() -> Self {
        Self { a: 0, b: 0 }
    }

    /// `(a1 a2 + sigma b1 b2, a1 b2 + b1 a2)`.
    ///
    /// `Trichotomy.Basic.mul`, and it is commutative and associative on every face
    /// (`mul_comm`, `mul_assoc`) -- which is what licenses folding a list of these in any order.
    #[must_use]
    pub fn mul(self, sigma: Sigma, other: Self) -> Self {
        let s = sigma.value();
        Self {
            a: self
                .a
                .wrapping_mul(other.a)
                .wrapping_add(s.wrapping_mul(self.b).wrapping_mul(other.b)),
            b: self
                .a
                .wrapping_mul(other.b)
                .wrapping_add(self.b.wrapping_mul(other.a)),
        }
    }

    #[must_use]
    pub const fn add(self, other: Self) -> Self {
        Self {
            a: self.a.wrapping_add(other.a),
            b: self.b.wrapping_add(other.b),
        }
    }

    /// `a^2 - sigma b^2`. `Trichotomy.Basic.normForm`.
    ///
    /// Multiplicative (`normForm_mul`), so it is zero exactly on the non-invertible cells. Above
    /// zero it factors as the product of the two channels
    /// (`NullBasis.normForm_eq_upper_mul_lower`), so a cell is degenerate exactly when one lane
    /// has gone to zero -- which is the precise sense in which a lossy fold sits on the null cone.
    #[must_use]
    pub fn norm_form(self, sigma: Sigma) -> i64 {
        self.a
            .wrapping_mul(self.a)
            .wrapping_sub(sigma.value().wrapping_mul(self.b).wrapping_mul(self.b))
    }

    /// The upper channel, `a + b`. Hyperbolic only.
    ///
    /// `NullBasis.upper_mul`: `upper(z * w) = upper(z) * upper(w)`. Returns `None` off the
    /// hyperbolic face because there is no second channel to be the upper one -- the type refuses
    /// what `no_channel_of_disc_neg` and `channel_iff_fst_of_disc_zero` say does not exist.
    #[must_use]
    pub fn upper(self, sigma: Sigma) -> Option<i64> {
        match sigma {
            Sigma::Hyperbolic => Some(self.a.wrapping_add(self.b)),
            Sigma::Parabolic | Sigma::Elliptic => None,
        }
    }

    /// The lower channel, `a - b`. Hyperbolic only. `NullBasis.lower_mul`.
    #[must_use]
    pub fn lower(self, sigma: Sigma) -> Option<i64> {
        match sigma {
            Sigma::Hyperbolic => Some(self.a.wrapping_sub(self.b)),
            Sigma::Parabolic | Sigma::Elliptic => None,
        }
    }

    /// The one channel a parabolic cell has: the first slot.
    /// `Channels.channel_iff_fst_of_disc_zero`.
    #[must_use]
    pub fn channel(self, sigma: Sigma) -> Option<i64> {
        match sigma {
            Sigma::Parabolic => Some(self.a),
            Sigma::Hyperbolic | Sigma::Elliptic => None,
        }
    }
}

/// Fold a list of cells into one.
///
/// `Gates.fold`, and `Gates.fold_perm`: for `sigma >= 0` the result is invariant under
/// permutation of the input. That is the licence for accumulating from many threads and combining
/// at a sync point with no ordering -- and it is stated as a property of the dial, so the elliptic
/// face does not get it by accident.
#[must_use]
pub fn fold(sigma: Sigma, cells: &[Cell]) -> Cell {
    cells.iter().fold(Cell::one(), |acc, c| acc.mul(sigma, *c))
}

/// Whether folding is order-free on this face. `Gates.fold_perm` requires `0 <= sigma`.
#[must_use]
pub const fn fold_is_order_free(sigma: Sigma) -> bool {
    !matches!(sigma, Sigma::Elliptic)
}
