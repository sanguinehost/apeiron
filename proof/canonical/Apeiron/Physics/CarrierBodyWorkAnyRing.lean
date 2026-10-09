import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing

/-!
# Equal-mass elastic carrier and body exchange over any commutative ring

@name Carrier Body Work Accounting Over Any Commutative Ring
@grade proved

`Apeiron.Physics.CarrierBodyWork` states the equal-mass elastic exchange for `Cell`, which is
`ℝ × ℝ`. That reading does not transfer to a machine word: `Z/2^64` has no order, two is not
invertible and `normForm` is not a nonnegative energy. This file carries the same exchange to an
arbitrary commutative ring, so `Z/2^64` is an instance rather than an analogy.

The exchange is additive, so it never consults the dial `σ` or the multiplication.
`ringExchangeImpulse` is `p - q` on `RingCell R`; the carrier removes the impulse and the body adds
it, so the two momenta swap exactly and the `ringNormForm` ledger is only permuted. The names carry
the `ring` prefix to keep them distinct from the real-valued `Cell` development, exactly as
`Apeiron.Trichotomy.AnyRing` keeps `ringMul` and `ringNormForm` distinct from `mul` and `normForm`.

This is the norm-form ledger, not kinetic energy. `ringNormForm` is indefinite in a general ring and
there is no `1/2` and no nonnegativity claim here; the kinetic-energy reading belongs to the
nonpositive-`σ` faces of the real development.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- The impulse the carrier hands the body, derived from the two basis momenta.

    (p q : RingCell R)              [the carrier momentum and the body momentum]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    p - q                           [the carrier momentum minus the body momentum]

    No impulse parameter is supplied; the difference of the two points is the whole authority. -/
def ringExchangeImpulse (p q : RingCell R) : RingCell R := p - q

/-- The carrier momentum after the derived exchange.

    (p q : RingCell R)              [the carrier momentum and the body momentum]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    p - ringExchangeImpulse p q     [the carrier momentum with the derived impulse removed]

    The update is defined from the impulse, not the other way round. -/
def ringCarrierMomentumAfter (p q : RingCell R) : RingCell R := p - ringExchangeImpulse p q

/-- The body momentum after the derived exchange.

    (p q : RingCell R)              [the carrier momentum and the body momentum]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    q + ringExchangeImpulse p q     [the body momentum with the derived impulse added]

    The body reaction is the same impulse with the opposite sign. -/
def ringBodyMomentumAfter (p q : RingCell R) : RingCell R := q + ringExchangeImpulse p q

/-- The norm-form ledger of the two momenta.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (p q : RingCell R)              [the carrier momentum and the body momentum]
    R                               [the commutative ring the points are built over]
    ringNormForm σ p + ringNormForm σ q
                                    [the carrier norm form plus the body norm form]

    This is universal accounting: `ringNormForm` may be negative or non-invertible in a general
    ring. -/
def ringNormFormLedger (σ : R) (p q : RingCell R) : R :=
  ringNormForm σ p + ringNormForm σ q

/-- The carrier update returns the body momentum exactly.

    (p q : RingCell R)              [the carrier momentum and the body momentum]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    ringCarrierMomentumAfter p q = q
                                    [the carrier ends holding the body momentum]

    Equal masses make the exchange a swap. -/
theorem ringCarrierMomentumAfter_eq (p q : RingCell R) :
    ringCarrierMomentumAfter p q = q := by
  simp only [ringCarrierMomentumAfter, ringExchangeImpulse]
  abel

/-- The body reaction returns the carrier momentum exactly.

    (p q : RingCell R)              [the carrier momentum and the body momentum]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    ringBodyMomentumAfter p q = p   [the body ends holding the carrier momentum]

    This is the other half of the swap. -/
theorem ringBodyMomentumAfter_eq (p q : RingCell R) :
    ringBodyMomentumAfter p q = p := by
  simp only [ringBodyMomentumAfter, ringExchangeImpulse]
  abel

/-- The impulse the body receives is the negative of the impulse the carrier receives.

    (p q : RingCell R)              [the carrier momentum and the body momentum]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    ringExchangeImpulse p q = - ringExchangeImpulse q p
                                    [swapping the roles negates the derived impulse]

    Equal and opposite, componentwise, from the one definition. -/
theorem ringExchangeImpulse_antisymm (p q : RingCell R) :
    ringExchangeImpulse p q = - ringExchangeImpulse q p := by
  simp only [ringExchangeImpulse]
  abel

/-- The two updates conserve the total basis momentum.

    (p q : RingCell R)              [the carrier momentum and the body momentum]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    ringCarrierMomentumAfter p q + ringBodyMomentumAfter p q = p + q
                                    [the total after the exchange is the total before it]

    The total is an equality of basis points, not a fitted scalar. -/
theorem ringTotalMomentum_conserved (p q : RingCell R) :
    ringCarrierMomentumAfter p q + ringBodyMomentumAfter p q = p + q := by
  simp only [ringCarrierMomentumAfter, ringBodyMomentumAfter, ringExchangeImpulse]
  abel

/-- The norm-form ledger is conserved by the derived exchange at every dial value.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (p q : RingCell R)              [the carrier momentum and the body momentum]
    R                               [the commutative ring the points are built over]
    ringNormFormLedger σ (ringCarrierMomentumAfter p q) (ringBodyMomentumAfter p q)
      = ringNormFormLedger σ p q
                                    [the ledger after the exchange equals the ledger before it]

    The two norm forms are only permuted, so the sum is unchanged. -/
theorem ringNormFormLedger_conserved (σ : R) (p q : RingCell R) :
    ringNormFormLedger σ (ringCarrierMomentumAfter p q) (ringBodyMomentumAfter p q)
      = ringNormFormLedger σ p q := by
  rw [ringCarrierMomentumAfter_eq, ringBodyMomentumAfter_eq]
  exact add_comm _ _

end Physics
end Apeiron
