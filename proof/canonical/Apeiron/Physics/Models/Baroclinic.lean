import Apeiron.Trichotomy.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Algebra.BigOperators.Fin

/-!
# Baroclinic instability

@name Growing Modes
@grade proved

A normal mode of a layered rotating flow solves a quadratic. Whether it grows or wobbles and dies is
decided by the sign of that quadratic's discriminant, and that is the same number this corpus
calls `σ`.

`growth_iff_noZeroDivisors` states it: the mode grows exactly when the algebra at its own
discriminant has no zero divisors. Not an analogy. The `σ` in `mul σ` and the discriminant of the
mode are one number, and the trichotomy's elliptic face and the growing mode are one condition.

## What is a theorem here and what is a reading

The growth criterion, the sign-change algebra behind the Charney-Stern condition, and the balance
that fixes the deformation radius are theorems. That `reducedDisc` **is** a normal mode's
discriminant, and that `N·H/f` **is** the length an instability selects, are readings: no fluid
equation appears anywhere in this file, and nothing here derives either from one. The predecessor
graded them the same way and it was right to.

## A citation that names nothing

`../whispering-tides/WORKLOG.md:317` and `../whispering-tides/src/moisture.slang:185` both cite
`bandCount` as what a generator computes instead of copying Earth's three. **There is no such
declaration, and there never was.** The predecessor names it in one prose line and its own `Open`
bullet says it "is declared nowhere". What exists is `eddyCount`, a ratio of lengths, and the
predecessor's docstring warns the two must not be confused: an eddy count is not a cell count,
and no theorem here or there derives a cell count.
-/

noncomputable section

namespace Apeiron
namespace Physics

open Trichotomy

/-- The reduced discriminant of a normal mode's quadratic.

    (b d : ℝ)                   [the two coefficients of the reduced quadratic]
    : ℝ                         [its discriminant] -/
def reducedDisc (b d : ℝ) : ℝ := b ^ 2 - d

/-- How fast a mode grows.

    (b d : ℝ)                   [the two coefficients]
    : ℝ                         [the growth rate, zero when the mode does not grow] -/
def growthRate (b d : ℝ) : ℝ := Real.sqrt (d - b ^ 2)

/-- A mode grows exactly on the negative-discriminant branch.

    (b d : ℝ)                                    [the two coefficients]
    0 < growthRate b d                           [it grows]
    ↔ reducedDisc b d < 0                        [exactly when the discriminant is below zero] -/
theorem grows_iff_disc_neg (b d : ℝ) : 0 < growthRate b d ↔ reducedDisc b d < 0 := by
  unfold growthRate reducedDisc
  rw [Real.sqrt_pos]
  constructor <;> intro h <;> linarith

/-- A mode is neutral exactly on the other branch.

    (b d : ℝ)                                    [the two coefficients]
    growthRate b d = 0                           [it does not grow]
    ↔ 0 ≤ reducedDisc b d                        [exactly when the discriminant is not negative] -/
theorem neutral_iff_disc_nonneg (b d : ℝ) : growthRate b d = 0 ↔ 0 ≤ reducedDisc b d := by
  unfold growthRate reducedDisc
  rw [Real.sqrt_eq_zero']
  constructor <;> intro h <;> linarith

/-- A mode grows exactly when the algebra at its own discriminant has no zero divisors.

    (b d : ℝ)                                                          [the two coefficients]
    0 < growthRate b d                                                 [the mode grows]
    ↔                                                                  [exactly when]
    ∀ z w : Cell, mul (reducedDisc b d) z w = 0 → z = 0 ∨ w = 0        [nothing divides to zero]

    So: the discriminant of the mode and the algebra's own `σ` are one number. A growing mode
    sits on the elliptic face, where the algebra is a field and division is total; a neutral mode
    sits at or above zero, where it is not and there are zero divisors.

    The predecessor proved one direction of this, from growth to the absence of zero divisors,
    and called it `growth_is_the_being_cell`. Both directions hold because
    `noZeroDivisors_iff_disc_neg` is itself an equivalence. -/
theorem growth_iff_noZeroDivisors (b d : ℝ) :
    0 < growthRate b d ↔ (∀ z w : Cell, mul (reducedDisc b d) z w = 0 → z = 0 ∨ w = 0) := by
  rw [grows_iff_disc_neg, ← noZeroDivisors_iff_disc_neg]

-- ═══════════════════════════════════════════════════════════════════
-- A VANISHING WEIGHTED SUM FORCES A SIGN CHANGE
-- ═══════════════════════════════════════════════════════════════════

/-- A positively weighted sum that vanishes forbids a one-signed summand.

    {n : ℕ} (w g : Fin n → ℝ)                     [weights and a quantity, indexed]
    (i₀ : Fin n)                                  [there is at least one index]
    (hw : ∀ i, 0 < w i)                           [every weight is above zero]
    (hsum : ∑ i, w i * g i = 0)                   [the weighted sum vanishes]
    ¬ (∀ i, 0 < g i) ∧ ¬ (∀ i, g i < 0)           [so the quantity is not one-signed]

    So: this is the algebra of the Charney-Stern condition and only that. The integral identity
    that supplies the vanishing sum is not proved here and is not assumed to be free: the
    predecessor withdrew a claim that its necessary condition rested on no unproved step, because
    the summed relation enters as a hypothesis there too. -/
theorem sign_change_of_weighted_zero {n : ℕ} (w g : Fin n → ℝ) (i₀ : Fin n)
    (hw : ∀ i, 0 < w i) (hsum : ∑ i, w i * g i = 0) :
    ¬ (∀ i, 0 < g i) ∧ ¬ (∀ i, g i < 0) := by
  constructor
  · intro hg
    have : 0 < ∑ i, w i * g i :=
      Finset.sum_pos (fun i _ => mul_pos (hw i) (hg i)) ⟨i₀, Finset.mem_univ i₀⟩
    linarith
  · intro hg
    have : ∑ i, w i * g i < 0 :=
      Finset.sum_neg (fun i _ => mul_neg_of_pos_of_neg (hw i) (hg i)) ⟨i₀, Finset.mem_univ i₀⟩
    linarith

-- ═══════════════════════════════════════════════════════════════════
-- THE LENGTH THE ROTATION AND THE LAYERING AGREE ON
-- ═══════════════════════════════════════════════════════════════════

/-- The length at which rotation and stratification balance.

    (N H f : ℝ)                 [a layering strength, a depth, and a rotation rate]
    : ℝ                         [the length they agree on] -/
def deformationRadius (N H f : ℝ) : ℝ := N * H / f

/-- It is the unique length at which the two terms balance.

    (N H f L : ℝ) (hf : f ≠ 0)                        [the three, a length, and spin not zero]
    f * L = N * H ↔ L = deformationRadius N H f        [the balance holds exactly at that length]

    So: the selection is a uniqueness, not a definition dressed as one. That this length is what a
    baroclinic instability picks out is the reading, and no theorem here supplies it. -/
theorem deformation_balances (N H f L : ℝ) (hf : f ≠ 0) :
    f * L = N * H ↔ L = deformationRadius N H f := by
  unfold deformationRadius
  constructor
  · intro h
    field_simp
    linarith
  · intro h
    subst h
    field_simp

end Physics
end Apeiron

end
