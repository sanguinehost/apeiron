import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.Basic
import Apeiron.Physics.CarrierBodyWorkAnyRing

/-!
# Equal-mass elastic carrier and body exchange

@name Carrier Body Work Accounting
@grade proved

A local elastic exchange between a carrier basis momentum `p` and a body basis momentum `q`, both
points of the trichotomy algebra. The impulse is derived from the two cells alone,
`exchangeImpulse p q = p - q`, with no independently authored impulse parameter. The carrier
update removes the impulse and the body reaction adds it, so the two momenta swap exactly. Every
ledger below is a consequence of that single definition.

The universal accounting is the sum of norm forms. `normForm σ` is indefinite on hyperbolic
faces, so the conserved sum is named a norm-form ledger rather than a kinetic energy. The
kinetic-energy reading, with its explicit `1/2`, is reserved for the nonpositive-`σ` faces, where
`kineticEnergy_nonnegative_of_nonpos` proves the reading is never negative. The mass is one
throughout, so the ledger is twice the unit-mass kinetic energy on those faces.

## Supported regime, assumptions and scope

Equal masses, an instantaneous local elastic exchange, no positions and no contact detection. The
slice has no dissipation, no forcing and no circulation evolution. It is a local collision
constitutive law, not a body-fluid coupling: there is no shared fluid velocity, no pressure, no
boundary or volume term, and no time integration.

## Root

`exchangeImpulse` is `p - q`, the difference of two `Trichotomy.Cell` momenta.
`carrierMomentumAfter_eq` and `bodyMomentumAfter_eq` are the swap, and `totalMomentum_conserved`
with `normFormLedger_conserved` accounts for the two conserved totals. `carrierWork_eq_neg_bodyWork`
and `totalWork_zero` are the work ledger, and `accepts_derivedImpulse` with
`independentImpulse_rejected` is the acceptance relation.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- The impulse the carrier hands the body, derived from the two basis momenta.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    p - q                           [the carrier momentum minus the body momentum]

    No impulse parameter is supplied; the difference of the two cells is the whole authority.

    This is the real specialization of `ringExchangeImpulse`, the ring-universal definition, so
    the impulse formula is authored once. -/
abbrev exchangeImpulse (p q : Cell) : Cell := ringExchangeImpulse (R := ℝ) p q

/-- The carrier momentum after the derived exchange.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    p - exchangeImpulse p q         [the carrier momentum with the derived impulse removed]

    The update is defined from the impulse, not the other way round.

    This is the real specialization of `ringCarrierMomentumAfter`. -/
abbrev carrierMomentumAfter (p q : Cell) : Cell := ringCarrierMomentumAfter (R := ℝ) p q

/-- The body momentum after the derived exchange.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    q + exchangeImpulse p q         [the body momentum with the derived impulse added]

    The body reaction is the same impulse with the opposite sign.

    This is the real specialization of `ringBodyMomentumAfter`. -/
abbrev bodyMomentumAfter (p q : Cell) : Cell := ringBodyMomentumAfter (R := ℝ) p q

/-- The carrier update returns the body momentum exactly.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    carrierMomentumAfter p q = q    [the carrier ends holding the body momentum]

    Equal masses make the exchange a swap. This is the real specialization of
    `ringCarrierMomentumAfter_eq`. -/
theorem carrierMomentumAfter_eq (p q : Cell) : carrierMomentumAfter p q = q :=
  ringCarrierMomentumAfter_eq (R := ℝ) p q

/-- The body reaction returns the carrier momentum exactly.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    bodyMomentumAfter p q = p       [the body ends holding the carrier momentum]

    This is the other half of the swap, and the real specialization of
    `ringBodyMomentumAfter_eq`. -/
theorem bodyMomentumAfter_eq (p q : Cell) : bodyMomentumAfter p q = p :=
  ringBodyMomentumAfter_eq (R := ℝ) p q

/-- The impulse the body receives is the negative of the impulse the carrier receives.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    exchangeImpulse p q = - exchangeImpulse q p
                                    [swapping the roles negates the derived impulse]

    Equal and opposite, componentwise, from the one definition, and the real specialization of
    `ringExchangeImpulse_antisymm`. -/
theorem exchangeImpulse_antisymm (p q : Cell) :
    exchangeImpulse p q = - exchangeImpulse q p :=
  ringExchangeImpulse_antisymm (R := ℝ) p q

/-- The two updates conserve the total basis momentum.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    carrierMomentumAfter p q + bodyMomentumAfter p q = p + q
                                    [the total after the exchange is the total before it]

    The total is an equality of basis cells, not a fitted scalar, and the real specialization of
    `ringTotalMomentum_conserved`. -/
theorem totalMomentum_conserved (p q : Cell) :
    carrierMomentumAfter p q + bodyMomentumAfter p q = p + q :=
  ringTotalMomentum_conserved (R := ℝ) p q

/-- The norm-form ledger of the two momenta.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (p q : Cell)                    [the carrier momentum and the body momentum]
    ℝ                               [the real numbers: every number on the number line]
    normForm σ p + normForm σ q     [the carrier norm form plus the body norm form]

    This is universal accounting: `normForm` may be negative on hyperbolic faces. This is the
    real specialization of `ringNormFormLedger`. -/
abbrev normFormLedger (σ : ℝ) (p q : Cell) : ℝ := ringNormFormLedger (R := ℝ) σ p q

/-- The norm-form ledger is conserved by the derived exchange at every σ.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (p q : Cell)                    [the carrier momentum and the body momentum]
    ℝ                               [the real numbers: every number on the number line]
    normFormLedger σ (carrierMomentumAfter p q) (bodyMomentumAfter p q)
      = normFormLedger σ p q
                                    [the ledger after the exchange equals the ledger before it]

    The two norm forms are only permuted, so the sum is unchanged. This is the real
    specialization of `ringNormFormLedger_conserved`. -/
theorem normFormLedger_conserved (σ : ℝ) (p q : Cell) :
    normFormLedger σ (carrierMomentumAfter p q) (bodyMomentumAfter p q)
      = normFormLedger σ p q :=
  ringNormFormLedger_conserved (R := ℝ) σ p q

/-- The carrier's norm-form change under the derived exchange.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (p q : Cell)                    [the carrier momentum and the body momentum]
    ℝ                               [the real numbers: every number on the number line]
    normForm σ (carrierMomentumAfter p q) - normForm σ p
                                    [the carrier norm form after minus before]

    On the nonpositive-σ faces this is twice the carrier kinetic-energy change. -/
def carrierWork (σ : ℝ) (p q : Cell) : ℝ :=
  normForm σ (carrierMomentumAfter p q) - normForm σ p

/-- The body's norm-form change under the derived exchange.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (p q : Cell)                    [the carrier momentum and the body momentum]
    ℝ                               [the real numbers: every number on the number line]
    normForm σ (bodyMomentumAfter p q) - normForm σ q
                                    [the body norm form after minus before]

    The reaction's accounting, defined the same way as the carrier's. -/
def bodyWork (σ : ℝ) (p q : Cell) : ℝ :=
  normForm σ (bodyMomentumAfter p q) - normForm σ q

/-- The carrier work is the exact opposite of the body work at every σ.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (p q : Cell)                    [the carrier momentum and the body momentum]
    ℝ                               [the real numbers: every number on the number line]
    carrierWork σ p q = - bodyWork σ p q
                                    [the two norm-form changes are exact opposites]

    Equal masses make the work ledger antisymmetric. -/
theorem carrierWork_eq_neg_bodyWork (σ : ℝ) (p q : Cell) :
    carrierWork σ p q = - bodyWork σ p q := by
  simp only [carrierWork, bodyWork, carrierMomentumAfter_eq, bodyMomentumAfter_eq]
  ring

/-- The total work of the derived exchange is zero at every σ.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (p q : Cell)                    [the carrier momentum and the body momentum]
    ℝ                               [the real numbers: every number on the number line]
    carrierWork σ p q + bodyWork σ p q = 0
                                    [the carrier work plus the body work vanishes]

    This is the accounting law that licenses the additional work authority. -/
theorem totalWork_zero (σ : ℝ) (p q : Cell) :
    carrierWork σ p q + bodyWork σ p q = 0 := by
  rw [carrierWork_eq_neg_bodyWork]
  ring

/-- The kinetic-energy reading of a basis momentum, at unit mass.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (z : Cell)                      [a basis momentum]
    ℝ                               [the real numbers: every number on the number line]
    normForm σ z / 2                [half the norm form, the kinetic energy at unit mass]

    The `1/2` is present and the mass is one. The reading is nonnegative only on the
    nonpositive-σ faces; on hyperbolic faces the norm form can be negative. -/
noncomputable def kineticEnergy (σ : ℝ) (z : Cell) : ℝ := normForm σ z / 2

/-- On the nonpositive-σ faces the kinetic-energy reading is never negative.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (hσ : σ ≤ 0)                    [the parameter is at or below zero]
    (z : Cell)                      [a basis momentum]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    0 ≤ kineticEnergy σ z           [the kinetic energy is at or above zero]

    The proof splits the norm form into a square plus `-σ` times a square. -/
theorem kineticEnergy_nonnegative_of_nonpos {σ : ℝ} (hσ : σ ≤ 0) (z : Cell) :
    0 ≤ kineticEnergy σ z := by
  have h : 0 ≤ normForm σ z := by
    have hsplit : normForm σ z = z.1 ^ 2 + (-σ * z.2 ^ 2) := by
      simp only [normForm]
      ring
    rw [hsplit]
    exact add_nonneg (sq_nonneg z.1) (mul_nonneg (neg_nonneg.mpr hσ) (sq_nonneg z.2))
  exact div_nonneg h (by norm_num)

/-- The carrier work is twice the carrier kinetic-energy change.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (p q : Cell)                    [the carrier momentum and the body momentum]
    ℝ                               [the real numbers: every number on the number line]
    carrierWork σ p q = 2 * (kineticEnergy σ (carrierMomentumAfter p q) - kineticEnergy σ p)
                                    [the norm-form work is twice the unit-mass kinetic change]

    Algebraic at every σ; the kinetic reading is a nonnegative energy only for σ at or below
    zero. -/
theorem carrierWork_eq_two_mul_kineticEnergyChange (σ : ℝ) (p q : Cell) :
    carrierWork σ p q
      = 2 * (kineticEnergy σ (carrierMomentumAfter p q) - kineticEnergy σ p) := by
  simp only [carrierWork, kineticEnergy, carrierMomentumAfter_eq]
  ring

/-- The body work is twice the body kinetic-energy change.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (p q : Cell)                    [the carrier momentum and the body momentum]
    ℝ                               [the real numbers: every number on the number line]
    bodyWork σ p q = 2 * (kineticEnergy σ (bodyMomentumAfter p q) - kineticEnergy σ q)
                                    [the body norm-form work is twice its unit-mass kinetic change]

    The same identity on the reaction side. -/
theorem bodyWork_eq_two_mul_kineticEnergyChange (σ : ℝ) (p q : Cell) :
    bodyWork σ p q
      = 2 * (kineticEnergy σ (bodyMomentumAfter p q) - kineticEnergy σ q) := by
  simp only [bodyWork, kineticEnergy, bodyMomentumAfter_eq]
  ring

/-- Acceptance of a claimed impulse only when it is the derived impulse.

    (p q r : Cell)                  [the carrier momentum, the body momentum and the claim]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    r = exchangeImpulse p q         [the claim equals the derived impulse]

    The predicate is the whole control: a claim that is not the derived impulse is rejected. -/
def acceptsImpulse (p q r : Cell) : Prop := r = exchangeImpulse p q

/-- The derived impulse is accepted.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    acceptsImpulse p q (exchangeImpulse p q)
                                    [the derived claim passes the acceptance relation]

    Immediate from the definition. -/
theorem accepts_derivedImpulse (p q : Cell) :
    acceptsImpulse p q (exchangeImpulse p q) := rfl

/-- An independently supplied impulse, not derived from the two momenta.

    Cell                            [a point: two real numbers written side by side, a + b·x]
    (0, 1)                          [the second basis direction, chosen by hand]

    This is the negative control for the acceptance relation. -/
def independentImpulse : Cell := (0, 1)

/-- Any independently supplied impulse unequal to the derived impulse is rejected.

    (p q : Cell)                    [the carrier momentum and the body momentum]
    (h : exchangeImpulse p q ≠ independentImpulse)
                                    [the derived impulse differs from the independent claim]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ≠                               [is not equal to]
    ¬                               [this is not true]
    ¬ acceptsImpulse p q independentImpulse
                                    [the independent claim is rejected]

    No impulse is accepted for being supplied alone. -/
theorem independentImpulse_rejected (p q : Cell)
    (h : exchangeImpulse p q ≠ independentImpulse) :
    ¬ acceptsImpulse p q independentImpulse := by
  intro hac
  exact h hac.symm

/-- The carrier momentum of the concrete exchange witness.

    Cell                            [a point: two real numbers written side by side, a + b·x]
    (3, 1)                          [three in the first slot and one in the second]

    The two momenta differ, so the exchange is nontrivial. -/
def witnessP : Cell := (3, 1)

/-- The body momentum of the concrete exchange witness.

    Cell                            [a point: two real numbers written side by side, a + b·x]
    (1, 2)                          [one in the first slot and two in the second]

    The witness is an exact small-integer pair. -/
def witnessQ : Cell := (1, 2)

/-- The witness impulse evaluates to a nonzero basis cell.

    exchangeImpulse witnessP witnessQ = (2, -1)
                                    [the derived impulse is two then minus one]

    The impulse is not the zero cell, so the exchange moves both momenta. -/
theorem exchangeImpulse_witness :
    exchangeImpulse witnessP witnessQ = (2, -1) := by
  norm_num [ringExchangeImpulse, witnessP, witnessQ]

/-- The witness exchange changes both momenta.

    carrierMomentumAfter witnessP witnessQ ≠ witnessP
                                    [the carrier ends holding a different momentum]
    bodyMomentumAfter witnessP witnessQ ≠ witnessQ
                                    [the body ends holding a different momentum]
    ∧                               [both are true]
    ≠                               [is not equal to]

    A swap of two distinct cells. -/
theorem witness_exchange_changes_both :
    carrierMomentumAfter witnessP witnessQ ≠ witnessP ∧
      bodyMomentumAfter witnessP witnessQ ≠ witnessQ := by
  constructor
  · rw [carrierMomentumAfter_eq]
    intro h
    have hf : (1 : ℝ) = 3 := congrArg Prod.fst h
    norm_num at hf
  · rw [bodyMomentumAfter_eq]
    intro h
    have hf : (3 : ℝ) = 1 := congrArg Prod.fst h
    norm_num at hf

/-- The witness conserves the total basis momentum.

    carrierMomentumAfter witnessP witnessQ + bodyMomentumAfter witnessP witnessQ
      = witnessP + witnessQ
                                    [the witness total is unchanged]

    The concrete instance of `totalMomentum_conserved`. -/
theorem witness_totalMomentum :
    carrierMomentumAfter witnessP witnessQ + bodyMomentumAfter witnessP witnessQ
      = witnessP + witnessQ :=
  totalMomentum_conserved witnessP witnessQ

/-- The witness conserves the norm-form ledger at every σ.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    ℝ                               [the real numbers: every number on the number line]
    normFormLedger σ (carrierMomentumAfter witnessP witnessQ)
        (bodyMomentumAfter witnessP witnessQ) = normFormLedger σ witnessP witnessQ
                                    [the witness ledger is unchanged]

    The concrete instance of `normFormLedger_conserved`. -/
theorem witness_energyLedger (σ : ℝ) :
    normFormLedger σ (carrierMomentumAfter witnessP witnessQ)
        (bodyMomentumAfter witnessP witnessQ)
      = normFormLedger σ witnessP witnessQ :=
  normFormLedger_conserved σ witnessP witnessQ

/-- The witness carrier work is a fixed linear function of σ.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    ℝ                               [the real numbers: every number on the number line]
    carrierWork σ witnessP witnessQ = -8 - 3 * σ
                                    [the carrier norm-form change evaluates to this expression]

    The ledger evaluates, not just balances. -/
theorem witness_carrierWork (σ : ℝ) :
    carrierWork σ witnessP witnessQ = -8 - 3 * σ := by
  unfold carrierWork witnessP witnessQ
  rw [carrierMomentumAfter_eq]
  unfold normForm
  norm_num
  ring

/-- The witness body work is the exact opposite expression.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    ℝ                               [the real numbers: every number on the number line]
    bodyWork σ witnessP witnessQ = 8 + 3 * σ
                                    [the body norm-form change evaluates to this expression]

    The two witness expressions sum to zero. -/
theorem witness_bodyWork (σ : ℝ) :
    bodyWork σ witnessP witnessQ = 8 + 3 * σ := by
  unfold bodyWork witnessP witnessQ
  rw [bodyMomentumAfter_eq]
  unfold normForm
  norm_num
  ring

/-- The witness rejects an independently supplied impulse.

    ¬                               [this is not true]
    ¬ acceptsImpulse witnessP witnessQ independentImpulse
                                    [the independent claim fails at the concrete witness]

    The concrete instance of `independentImpulse_rejected`. -/
theorem witness_independentImpulse_rejected :
    ¬ acceptsImpulse witnessP witnessQ independentImpulse := by
  apply independentImpulse_rejected
  rw [exchangeImpulse_witness, independentImpulse]
  intro h
  have hf : (2 : ℝ) = 0 := congrArg Prod.fst h
  norm_num at hf

end Physics
end Apeiron
