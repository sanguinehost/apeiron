--//@ file: Apeiron/Physics/Mixing.lean
--//@ does: Gatto relation and maximal atmospheric mixing angle from null channel balance
--//@ ring: R0
--//@ onto: observation:physics:gatto_mixing
--//@ cite: Gatto, Sartori & Tonin (1968), NuFIT (2024)
--//@ grade: proved

import Apeiron.Trichotomy.Flow
import Apeiron.Trichotomy.NullBasis
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Two angles the corpus does not derive, and what it does with them instead

@name Mixing
@cite gatto-sartori-tonin-1968
@cite nufit-6-2024
@cite pdg-2024
@grade proved

Two flavour-mixing statements are carried by no theorem in the predecessor and by none here. They
are the maximal atmospheric angle `θ₂₃ = π/4`, and the Gatto relation, that a mixing angle is the
square root of a mass ratio. Both are ported as **named rules with the prediction conditional on
them**, so the posited step is visible in the type rather than buried in a definition or a
docstring.

## The maximal angle

`BalancedChannels θ` says the elliptic flow at `θ` has size zero for the hyperbolic norm form:
the point sits on the `σ = 1` cone. `Trichotomy.normForm_eq_upper_mul_lower` says that norm form
is the product of the two null channel readings, so a point on the cone is one whose two channels
do not both survive, and squaring gives the balanced case `cos²θ = sin²θ`. That is the content of
`maximal_mixing_iff_on_hyperbolic_cone`: the elliptic flow crosses the hyperbolic cone exactly at
the balanced angles.

The rule posited is that the mixing angle of a sector with nothing to distinguish its two channels
is such a crossing. Given it, `balanced_forces_half` returns `sin²θ = 1/2` with no freedom left,
which is the quarter turn. NuFIT 6.0, arXiv:2410.05380, Table 1, measures `sin²θ₂₃ = 0.470` with a
three-sigma range `0.407` to `0.620` in normal ordering, so a half lies inside the measured range
and the rule is not yet decided by the data.

What the corpus does **not** do is derive that the atmospheric sector is balanced. The value `π/4`
follows from the rule; the rule does not follow from anything here.

## The Gatto relation

`MixingIsRootRatio θ mLight mHeavy` says `sin²θ · mHeavy = mLight`. Given it and the measured
quark mass ratio, `gatto_predicts_cabibbo` returns a band for the sine, and
`cabibbo_measured_in_band` says the measured Cabibbo sine sits in it. The band's width is the mass
ratio's, which is why this is a consistency check rather than a sharp prediction.

Gatto, Sartori and Tonin published the relation in 1968, Phys. Lett. B 28, 128, so nothing here is
new about it; what is here is that the rule is a hypothesis with a name, and the arithmetic that
turns it into a number is checked.

## Root

Instantiates `Trichotomy.flow` at `σ = −1` for the angle and `Trichotomy.normForm` at `σ = 1` for
the balance condition, and rides `Trichotomy.normForm_eq_upper_mul_lower` for why a point on the
hyperbolic cone is one whose two channel readings cannot both stand. The edge:

* `maximal_mixing_iff_on_hyperbolic_cone` is the crossing of the elliptic flow with the hyperbolic
  cone, and it is where the quarter turn comes from; the statement names both faces and neither is
  a spectator.

The Gatto rule is rooted in nothing: it is a hypothesis, and `theory/DEAD_ENDS.md` records the
three routes tried to derive it.
-/

namespace Apeiron
namespace Physics

open Trichotomy Real

/-- **The elliptic flow meets the hyperbolic cone exactly at the balanced angles.**

    ℝ                                                    [the real numbers]
    ↔                                                    [exactly when, true in both directions]
    normForm                                            [the norm form: a point times its conjugate]
    flow                                                 [the size-one flow of the algebra]
    Real.cos θ ^ 2 = Real.sin θ ^ 2                      [the two channels weigh the same]

    So: two faces in one statement, and neither is a spectator. The turn is elliptic and the cone
    is hyperbolic, and the crossing is where the two coordinates of the turning point balance. -/
theorem maximal_mixing_iff_on_hyperbolic_cone (θ : ℝ) :
    normForm 1 (flow (-1) θ) = 0 ↔ Real.cos θ ^ 2 = Real.sin θ ^ 2 := by
  rw [flow_neg_one]
  simp only [normForm]
  constructor <;> intro h <;> linarith

/-- The posited rule for a sector with nothing to tell its two channels apart.

    ℝ                                                    [the real numbers]
    normForm                                             [the norm form of a point]
    flow                                                 [the size-one flow of the algebra]
    BalancedChannels                                 [the turning point sits on the hyperbolic cone]

    So: the rule, named. Nothing in this corpus says any physical sector satisfies it. -/
def BalancedChannels (θ : ℝ) : Prop := normForm 1 (flow (-1) θ) = 0

/-- **Balanced channels force a half, with no freedom left.**

    ℝ                                                    [the real numbers]
    BalancedChannels                                 [the turning point sits on the hyperbolic cone]
    Real.sin θ ^ 2 = 1 / 2                               [the mixing weight is a half]

    So: the quarter turn, conditional on the named rule. NuFIT 6.0 Table 1 measures the
    atmospheric weight at `0.470`, three-sigma range `0.407` to `0.620` in normal ordering, so a
    half is inside the measured range and the rule is undecided by present data. -/
theorem balanced_forces_half (θ : ℝ) (h : BalancedChannels θ) : Real.sin θ ^ 2 = 1 / 2 := by
  have hbal := (maximal_mixing_iff_on_hyperbolic_cone θ).mp h
  have pyth := Real.sin_sq_add_cos_sq θ
  linarith

/-- **The quarter turn is balanced.**

    BalancedChannels                                 [the turning point sits on the hyperbolic cone]

    So: the rule is satisfiable, and the angle that satisfies it is the quarter turn, so
    `balanced_forces_half` is not about an empty condition. -/
theorem quarter_turn_is_balanced : BalancedChannels (π / 4) := by
  unfold BalancedChannels
  rw [flow_neg_one]
  simp only [normForm, Real.cos_pi_div_four, Real.sin_pi_div_four]
  ring

/-- The posited rule that a mixing weight is a mass ratio.

    ℝ                                                    [the real numbers]
    Real.sin θ ^ 2 * mHeavy = mLight                     [the mixing weight times the heavier mass]
    MixingIsRootRatio                                    [the mixing weight is the mass ratio]

    So: Gatto, Sartori and Tonin's relation, stated as a hypothesis rather than proved. -/
def MixingIsRootRatio (θ mLight mHeavy : ℝ) : Prop := Real.sin θ ^ 2 * mHeavy = mLight

/-- **The rule and the measured quark mass ratio give a band for the Cabibbo sine.**

    ℝ                                                    [the real numbers]
    ∧                                                    [both are true]
    ≤                                                    [is smaller than or equal to]
    <                                                    [is smaller than]
    MixingIsRootRatio                                    [the mixing weight is the mass ratio]
    0.0454 * mHeavy ≤ mLight                             [the down-to-strange ratio's lower end]
    mLight ≤ 0.0589 * mHeavy                             [and its upper end]

    So: the particle data ratio of the strange to the down mass runs from about seventeen to about
    twenty-two, so the ratio the rule consumes lies between the two bounds above, and the sine
    lands between `0.213` and `0.243`. The band's width is the mass ratio's, so this is a
    consistency check and not a sharp prediction. -/
theorem gatto_predicts_cabibbo (θ mLight mHeavy : ℝ)
    (hrule : MixingIsRootRatio θ mLight mHeavy) (hpos : 0 < mHeavy)
    (hlo : 0.0454 * mHeavy ≤ mLight) (hhi : mLight ≤ 0.0589 * mHeavy)
    (hsin : 0 ≤ Real.sin θ) :
    0.213 ≤ Real.sin θ ∧ Real.sin θ ≤ 0.243 := by
  unfold MixingIsRootRatio at hrule
  have hsq_lo : 0.0454 ≤ Real.sin θ ^ 2 := by
    have : 0.0454 * mHeavy ≤ Real.sin θ ^ 2 * mHeavy := by linarith
    exact le_of_mul_le_mul_right (by linarith) hpos
  have hsq_hi : Real.sin θ ^ 2 ≤ 0.0589 := by
    have : Real.sin θ ^ 2 * mHeavy ≤ 0.0589 * mHeavy := by linarith
    exact le_of_mul_le_mul_right (by linarith) hpos
  constructor
  · nlinarith [hsq_lo, hsin]
  · nlinarith [hsq_hi, hsin]

/-- **The measured Cabibbo sine lies in that band.**

    ∧                                                    [both are true]
    ≤                                                    [is smaller than or equal to]
    ℝ                                                   [the real numbers: every number on the line]

    So: the particle data value of the first row's second entry of the quark mixing grid is
    `0.2243`, which sits inside the band the rule predicts. The rule survives the comparison; it
    is not confirmed by it, because the band is wide. -/
theorem cabibbo_measured_in_band : (0.213 : ℝ) ≤ 0.2243 ∧ (0.2243 : ℝ) ≤ 0.243 := by
  constructor <;> norm_num

end Physics
end Apeiron
