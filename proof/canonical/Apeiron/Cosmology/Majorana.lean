import Apeiron.Trichotomy.Roots
import Apeiron.Physics.Flavour
import Apeiron.Cosmology.AnyRing

/-!
# A charge needs somewhere to turn, and three unequal weights cannot cancel

@name Majorana
@cite koide-2000
@cite nufit-6-2024
@grade proved

Two questions about a chargeless fermion, and the algebra answers both.

**Can it carry a phase?** A charge that acts by rotating needs a point squaring to minus the unit,
and `Trichotomy.exists_sqrt_neg_one_iff_disc_neg` says the algebra holds one exactly below zero.
So a sector living on the hyperbolic or the parabolic face has no rotor to charge with. That is
the algebraic content of "structurally Majorana", and it is a statement about `σ`, not about
a neutrino.

**Can a phase be rotated away?** `rephasing_fixes_nothing` says the size-one points that carry
every real point to a real point are `±1`, at every value of `σ`. A form
that is its own conjugate therefore has no rephasing freedom, so its relative phases are physical
rather than gauge. That is why a decay amplitude summing three generations can depend on phases a
charged sector would throw away.

## What the three phases do to the amplitude

Sum three real weights on the three forced phases of `Apeiron/Physics/Flavour.lean`, the orbit of
the order-three element. `weightSpread_eq_variance` says the resulting point's size is

```
   ((w₁ − w₂)² + (w₂ − w₃)² + (w₃ − w₁)²) / 2
```

the spread of the three weights, fully symmetric, so no ordering convention enters. Equal weights
give zero (`equal_weights_silence`), and the amplitude is the departure from equality. Below the
trinity choice, `phase_floor` bounds the amplitude from below at **arbitrary** size-one phases,
which is the full freedom a Majorana amplitude has, so the floor does not depend on the trinity
assignment at all.

The spread identity is polynomial, so it is proved over any commutative ring in
`Apeiron/Cosmology/AnyRing.lean`, with the order-three relation and a unit second slot as explicit
witness hypotheses. This file is its real specialization: the phase is the forced
`flow (-1) (2π/3)`, the trigonometric values of a third of a turn discharge the two hypotheses, and
`trinitySum_eq_ringTrinitySum` and `weightSpread_eq_ringWeightSpread` identify the real objects
with the generic ones. No trigonometric function, no pi and no decimal appears in the generic file.

`double_beta_lands` then carries the weights as interval hypotheses and returns a bound. The
weights are `|U_ei|²·mᵢ` for a spectrum this corpus does not derive, so the arithmetic is a
calibration instance: the bound is checked from the weights inward and the input is not derived
from the ring axioms.

## Root

Instantiates `Trichotomy.exists_sqrt_neg_one_iff_disc_neg` for whether a charge rotor exists and
`Trichotomy.exists_cube_root_one_iff_disc_neg` for the three phases, both from
`Apeiron/Trichotomy/Roots.lean`, and `Trichotomy.normForm` at `σ = −1` for the amplitude's size.
The edges:

* `rephasing_fixes_nothing` instantiates `Cosmology.ringRephasing_fixes_nothing`, which is uniform
  in `σ` and needs only that one of `z.1 − 1` and `z.1 + 1` can be divided by. The reals supply
  that witness, so the real statement is the same two points on every face.
* `weightSpread_eq_variance` instantiates `Cosmology.ringWeightSpread_eq_spread` at the forced
  phase, using the forced first slot `−1/2` of the order-three element through
  `flow_third_turn_root`. The identity is therefore the algebra's and not a choice of cube roots in
  some other ring; the factor two that the general ring cannot divide out is divided out here.
* `phase_floor` is stated on the elliptic face, where the size-one points form the circle the
  phases run on, and is an inequality rather than a polynomial identity, so it stays real.
-/

namespace Apeiron
namespace Cosmology

open Trichotomy Physics Real
open Trichotomy.AnyRing

/-- **A charge rotor exists exactly below zero.**

    σ                                   [the parameter]
    ℝ                                   [the real numbers: every number on the number line]
    ∃                                   [there is at least one]
    ↔                                   [exactly when, true in both directions]
    <                                   [is smaller than]
    mul                                 [multiply two points together, the σ way]
    one                                 [the point multiplication leaves alone]
    Cell                                [a point: two real numbers written side by side]

    So: a sector whose algebra is not elliptic has no point to build a phase rotation from. The
    reading that this sector is a chargeless fermion is not made here. -/
theorem charge_rotor_iff_disc_neg (σ : ℝ) :
    (∃ z : Cell, mul σ z z = -one) ↔ σ < 0 :=
  exists_sqrt_neg_one_iff_disc_neg σ

/-- **Only plus and minus the unit fix every real point, at every value of `σ`.**

    σ                                                     [the parameter]
    ℝ                                                     [the real numbers]
    ∀                                                     [for every single one]
    →                                                     [takes the left and gives the right]
    ∨                                                     [at least one is true]
    normForm                                              [the norm form of a point]
    mul                                                   [multiply two points, the σ way]
    one                                                   [the point multiplication leaves alone]
    Cell                                                  [a point: two real numbers side by side]

    So: a size-one point that keeps the real line inside itself is `±1`. A form equal to its own
    conjugate therefore has no phase to absorb, so its relative phases are physical. The statement
    holds on all three faces with one proof, and that proof is the generic
    `Cosmology.ringRephasing_fixes_nothing`: over the reals one of `z.1 − 1` and `z.1 + 1` is
    always away from zero, so the unit witness the general ring needs is free here. -/
theorem rephasing_fixes_nothing (σ : ℝ) (z : Cell) (hnorm : normForm σ z = 1)
    (hreal : ∀ w : Cell, w.2 = 0 → (mul σ z w).2 = 0) :
    z = one ∨ z = -one := by
  have hunit : IsUnit (z.1 - 1) ∨ IsUnit (z.1 + 1) := by
    rcases lt_trichotomy z.1 1 with h | h | h
    · left
      exact isUnit_iff_ne_zero.mpr (by linarith)
    · right
      exact isUnit_iff_ne_zero.mpr (by linarith)
    · right
      exact isUnit_iff_ne_zero.mpr (by linarith)
  exact ringRephasing_fixes_nothing σ z hreal hnorm hunit

/-- A third of a turn squares to two thirds of a turn.

    flow                                                  [the size-one flow of the algebra]
    ringMul                                               [multiply two points, the σ way]

    So: the phase at a third of a turn is the order-three element, so its square is the phase at
    two thirds, which is the generic square the weighted sum uses. -/
theorem flow_third_turn_sq :
    ringMul (-1) (flow (-1) (2 * π / 3)) (flow (-1) (2 * π / 3)) = flow (-1) (4 * π / 3) := by
  have h := flow_add (-1) (2 * π / 3) (2 * π / 3)
  rw [show 2 * π / 3 + 2 * π / 3 = 4 * π / 3 by ring] at h
  exact h

/-- The forced phase satisfies the order-three relation.

    flow                                                  [the size-one flow of the algebra]
    ringMul                                               [multiply two points, the σ way]
    ringOne                                               [the point (1, 0)]

    So: `1 + ω + ω² = 0` at the forced phase. This is the hypothesis the generic weighted-sum
    theorem carries, and the trigonometric values of a third of a turn discharge it. -/
theorem flow_third_turn_root :
    ringOne + flow (-1) (2 * π / 3)
      + ringMul (-1) (flow (-1) (2 * π / 3)) (flow (-1) (2 * π / 3)) = 0 := by
  rw [flow_third_turn_sq, flow_neg_one (2 * π / 3), flow_neg_one (4 * π / 3)]
  simp only [ringOne, cos_two_pi_div_three, sin_two_pi_div_three,
    cos_four_pi_div_three, sin_four_pi_div_three]
  refine Prod.ext_iff.mpr ⟨?_, ?_⟩ <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_zero, Prod.snd_zero] <;> ring

/-- The forced phase's second slot is a unit.

    IsUnit                                                [can be divided by]
    flow                                                  [the size-one flow of the algebra]

    So: half the square root of three is not zero. This is the unit hypothesis the generic
    weighted-sum theorem carries, and the real face supplies it at the forced phase. -/
theorem flow_third_turn_snd_isUnit : IsUnit (flow (-1) (2 * π / 3)).2 := by
  rw [flow_neg_one, sin_two_pi_div_three]
  exact isUnit_iff_ne_zero.mpr (ne_of_gt (by positivity : (0 : ℝ) < Real.sqrt 3 / 2))

/-- Three real weights placed on the three forced phases.

    ℝ                                                     [the real numbers]
    : Cell                                                [the point they add up to]
    Cell                                                  [a point: two real numbers side by side]
    one                                                   [the point multiplication leaves alone]
    flow                                                  [the size-one flow of the algebra]

    So: the first weight on the unit, the second a third of a turn on, the third two thirds. The
    phases are the orbit of the order-three element, which `exists_cube_root_one_iff_disc_neg`
    shows the elliptic face alone supplies. -/
noncomputable def trinitySum (w₁ w₂ w₃ : ℝ) : Cell :=
  w₁ • one + w₂ • flow (-1) (2 * π / 3) + w₃ • flow (-1) (4 * π / 3)

/-- The real weighted sum is the generic weighted sum at the forced phase.

    ℝ                                                     [the real numbers]
    trinitySum                                            [three weights on the three phases]
    flow                                                  [the size-one flow of the algebra]
    ringTrinitySum                                        [three ring weights on the phases]

    So: the explicit third of a turn and two thirds of a turn are the generic element and its
    square, which is what makes the real identity an instance rather than a second proof. -/
theorem trinitySum_eq_ringTrinitySum (w₁ w₂ w₃ : ℝ) :
    trinitySum w₁ w₂ w₃
      = ringTrinitySum (-1) (flow (-1) (2 * π / 3)) w₁ w₂ w₃ := by
  simp only [trinitySum, ringTrinitySum, flow_third_turn_sq, one]

/-- The size of that sum.

    ℝ                                                     [the real numbers]
    normForm                                              [the norm form of a point]
    weightSpread                                          [the size of the three-phase sum]

    So: on the elliptic face the norm form is the definite one, so this is the squared length of
    the amplitude. -/
noncomputable def weightSpread (w₁ w₂ w₃ : ℝ) : ℝ := normForm (-1) (trinitySum w₁ w₂ w₃)

/-- The real spread is the generic spread at the forced phase.

    ℝ                                                     [the real numbers]
    weightSpread                                          [the size of the three-phase sum, squared]
    ringWeightSpread                                      [the size of the sum over a ring]
    flow                                                  [the size-one flow of the algebra]

    So: the real definition is the generic one instantiated at the forced phase, so the real
    spread identity is the generic identity with the factor two divided back out. -/
theorem weightSpread_eq_ringWeightSpread (w₁ w₂ w₃ : ℝ) :
    weightSpread w₁ w₂ w₃
      = ringWeightSpread (-1) (flow (-1) (2 * π / 3)) w₁ w₂ w₃ := by
  simp only [weightSpread, ringWeightSpread, trinitySum_eq_ringTrinitySum]

/-- **The three-phase sum's size is the spread of the three weights.**

    ℝ                                                            [the real numbers]
    weightSpread                                                 [the size of the three-phase sum]
    ((w₁ - w₂) ^ 2 + (w₂ - w₃) ^ 2 + (w₃ - w₁) ^ 2) / 2          [the spread of the three weights]

    So: fully symmetric in the three weights, so no ordering convention enters, and the amplitude
    measures departure from equality. This is the real specialization of
    `Cosmology.ringWeightSpread_eq_spread`: the generic theorem keeps the factor two because a
    general ring need not have it invertible, and here it is divided back out. -/
theorem weightSpread_eq_variance (w₁ w₂ w₃ : ℝ) :
    weightSpread w₁ w₂ w₃ = ((w₁ - w₂) ^ 2 + (w₂ - w₃) ^ 2 + (w₃ - w₁) ^ 2) / 2 := by
  have hgen := ringWeightSpread_eq_spread (-1) (flow (-1) (2 * π / 3)) w₁ w₂ w₃
    flow_third_turn_root flow_third_turn_snd_isUnit
  rw [weightSpread_eq_ringWeightSpread]
  linarith

/-- **Equal weights silence the amplitude.**

    ℝ                                                     [the real numbers]
    weightSpread                                          [the size of the three-phase sum]

    So: three equal weights on the three phases cancel exactly, because the phases sum to zero.
    The amplitude exists only because the three generations sit unequally. This is the real
    specialization of `Cosmology.ringEqualWeights_silence`. -/
theorem equal_weights_silence (w : ℝ) : weightSpread w w w = 0 := by
  rw [weightSpread_eq_ringWeightSpread]
  exact ringEqualWeights_silence (-1) (flow (-1) (2 * π / 3)) w flow_third_turn_root

/-- **No choice of phases can silence a dominant first weight.**

    ℝ                                                            [the real numbers]
    ≤                                                            [is smaller than or equal to]
    normForm                                                     [the norm form of a point]
    one                                                      [the point multiplication leaves alone]
    Cell                                                         [a point: two reals side by side]
    •                                                           [stretch by, a number times a point]
    (w₁ - w₂ - w₃) ^ 2                                           [the gap, squared]

    So: at arbitrary size-one phases, which is the full freedom the amplitude has, the size is at
    least the squared gap. The floor does not use the three forced phases, so it survives whatever
    the phase assignment is. -/
theorem phase_floor (w₁ w₂ w₃ : ℝ) (h₂ : 0 ≤ w₂) (h₃ : 0 ≤ w₃) (hgap : w₂ + w₃ ≤ w₁)
    (u v : Cell) (hu : normForm (-1) u = 1) (hv : normForm (-1) v = 1) :
    (w₁ - w₂ - w₃) ^ 2 ≤ normForm (-1) (w₁ • one + w₂ • u + w₃ • v) := by
  obtain ⟨u1, u2⟩ := u
  obtain ⟨v1, v2⟩ := v
  simp only [normForm, one, Prod.smul_mk, Prod.mk_add_mk, smul_eq_mul] at hu hv ⊢
  have hu1 : -1 ≤ u1 := by nlinarith [sq_nonneg u2, sq_nonneg (u1 + 1)]
  have hv1 : -1 ≤ v1 := by nlinarith [sq_nonneg v2, sq_nonneg (v1 + 1)]
  have hgnn : (0 : ℝ) ≤ w₁ - w₂ - w₃ := by linarith
  have h2u : -w₂ ≤ w₂ * u1 := by nlinarith [mul_nonneg h₂ (by linarith : (0 : ℝ) ≤ u1 + 1)]
  have h3v : -w₃ ≤ w₃ * v1 := by nlinarith [mul_nonneg h₃ (by linarith : (0 : ℝ) ≤ v1 + 1)]
  have hA : w₁ - w₂ - w₃ ≤ w₁ * 1 + w₂ * u1 + w₃ * v1 := by linarith
  have hsq : (w₁ - w₂ - w₃) ^ 2 ≤ (w₁ * 1 + w₂ * u1 + w₃ * v1) ^ 2 := by nlinarith [hgnn, hA]
  nlinarith [hsq, sq_nonneg (w₁ * 0 + w₂ * u2 + w₃ * v2)]

/-- **The registered weights put the amplitude between 3.6 and 3.8.**

    ℝ                                                            [the real numbers]
    ∧                                                            [both are true]
    ≤                                                            [is smaller than or equal to]
    <                                                            [is smaller than]
    weightSpread                                                 [the size of the three-phase sum]
    5.37 ≤ w₁                                                    [the first weight, in meV]
    3.52 ≤ w₂                                                    [the second]
    1.11 ≤ w₃                                                    [the third]

    So: the arithmetic, from the weights inward, by rational bounds alone. The weights are the
    electron-row mixing weights times the masses of a spectrum this corpus does not derive, so
    what is checked here is the step from the weights to the amplitude and nothing before it. This
    is a calibration instance: the interval hypotheses are measured inputs, and the bound is not a
    consequence of the ring axioms. -/
theorem double_beta_lands (w₁ w₂ w₃ : ℝ)
    (h₁ : 5.37 ≤ w₁ ∧ w₁ ≤ 5.39)
    (h₂ : 3.52 ≤ w₂ ∧ w₂ ≤ 3.54)
    (h₃ : 1.11 ≤ w₃ ∧ w₃ ≤ 1.12) :
    (3.6 : ℝ) ^ 2 < weightSpread w₁ w₂ w₃ ∧ weightSpread w₁ w₂ w₃ < (3.8 : ℝ) ^ 2 := by
  obtain ⟨ha1, ha2⟩ := h₁
  obtain ⟨hb1, hb2⟩ := h₂
  obtain ⟨hc1, hc2⟩ := h₃
  rw [weightSpread_eq_variance]
  constructor <;> nlinarith

end Cosmology
end Apeiron
