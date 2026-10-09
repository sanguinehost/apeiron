import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing
import Mathlib.Tactic

/-!
# The order-three weighted sum over any commutative ring

@name The Order-Three Weighted Sum Over Any Commutative Ring
@cite koide-2000
@cite brannen-2006
@cite gauss-1831
@grade proved

`Apeiron/Cosmology/Majorana.lean` places three real weights on the three phases of the order-three
element of the elliptic algebra and proves that the size of the sum is the spread of the weights.
That identity is polynomial, so it does not need the reals, the order, the trigonometric functions
or the number pi. This file carries it over an arbitrary commutative ring, so a finite ring is an
instance rather than an analogy.

The witness is explicit. `ringTrinitySum σ ω w₁ w₂ w₃` is `w₁ • 1 + w₂ • ω + w₃ • ω²`, where `ω`
satisfies the order-three relation `1 + ω + ω² = 0` in the algebra whose dial is `σ`, and the
second slot of `ω` is a unit. From those two hypotheses:

* `ringWeightSpread_eq_spread` says twice the size of the sum is the sum of the three squared
  differences of the weights, the polynomial form of the variance identity. The factor two is kept
  because two is not invertible in a general ring.
* `ringEqualWeights_silence` says three equal weights cancel.
* `ringRephasing_factor` says a size-one point that keeps the real axis inside itself satisfies
  `(z.1 - 1)(z.1 + 1) = 0`, and `ringRephasing_fixes_nothing` turns that into `z = ±1` once one of
  the two factors is a unit. Over the reals both are units away from zero, which is the case
  `Apeiron/Cosmology/Majorana.lean` needs; over a ring with zero divisors the unit hypothesis is
  the missing witness.

What is deliberately absent is the real face: no `Real.cos`, no pi, no square root and no decimal
literal appears below. The trigonometric formulas and the registered numerical bounds stay in
`Majorana.lean` and `NeutrinoSum.lean`, as the real and calibration instances they are.

## Root

`Trichotomy/AnyRing.lean` supplies the algebra this file computes in: `ringMul`, `ringNormForm` and
`ringOne`. The edge derived from it is `ringWeightSpread_eq_spread`, which needs the order-three
relation and a unit second slot as hypotheses because neither follows from the ring axioms. The
real specialization in `Majorana.lean` discharges those hypotheses with the forced phase
`flow (-1) (2π/3)` and the trigonometric values of a third of a turn, and the existence of that
phase is `Trichotomy.exists_cube_root_one_iff_disc_neg`. The order-three relation is a hypothesis
here rather than a theorem because no ring axiom forces a point of order three to exist; the
missing primitive is a ring in which `X² + X + 1` has a root, and the reals at the elliptic dial
are one such ring.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Cosmology

open Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- Three ring weights placed on the three order-three phases.

    σ                                       [the parameter: what the adjoined direction squares to]
    RingCell R                              [a point: two ring elements side by side, a + b·x]

    So: the first weight on the unit, the second on the order-three element, the third on its
    square. No trigonometric function and no real number appears. -/
def ringTrinitySum (σ : R) (ω : RingCell R) (w₁ w₂ w₃ : R) : RingCell R :=
  w₁ • ringOne + w₂ • ω + w₃ • ringMul σ ω ω

/-- The size of the three-phase sum over a ring.

    σ                                       [the parameter: what the adjoined direction squares to]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]

    So: the norm form of `ringTrinitySum`, the squared length of the amplitude. The factor two
    that the variance identity needs is left to the theorem, because two may not be invertible. -/
def ringWeightSpread (σ : R) (ω : RingCell R) (w₁ w₂ w₃ : R) : R :=
  ringNormForm σ (ringTrinitySum σ ω w₁ w₂ w₃)

/-- The order-three relation solves for the square.

    σ                                       [the parameter: what the adjoined direction squares to]
    ringMul                                 [multiply two points together, the σ way]
    ringOne                                 [the point (1, 0), which changes nothing]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]

    So: from `1 + ω + ω² = 0` the square is `−(1 + ω)`. This is the only shape the relation is
    used in below. -/
theorem ringRoot_sq (σ : R) (ω : RingCell R)
    (hroot : ringOne + ω + ringMul σ ω ω = 0) :
    ringMul σ ω ω = -(ringOne + ω) := by
  rw [eq_neg_iff_add_eq_zero]
  simpa only [add_assoc, add_comm, add_left_comm] using hroot

/-- The second slot of the order-three relation is a multiple of the first slot's defect.

    σ                                       [the parameter: what the adjoined direction squares to]
    ringMul                                 [multiply two points together, the σ way]
    ringOne                                 [the point (1, 0), which changes nothing]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]

    So: the second slot of `1 + ω + ω² = 0` reads `ω.2 · (1 + 2·ω.1) = 0`. A unit second slot
    therefore forces `1 + 2·ω.1 = 0`, which is the half-turn value the real phase carries. -/
theorem ringRoot_snd (σ : R) (ω : RingCell R)
    (hroot : ringOne + ω + ringMul σ ω ω = 0) :
    ω.2 * (1 + 2 * ω.1) = 0 := by
  have h := (Prod.ext_iff.mp hroot).2
  change 0 + ω.2 + (ω.1 * ω.2 + ω.2 * ω.1) = 0 at h
  linear_combination h

/-- A unit second slot pins the first slot to the half-turn value.

    σ                                       [the parameter: what the adjoined direction squares to]
    IsUnit                                  [can be divided by]
    ringMul                                 [multiply two points together, the σ way]
    ringOne                                 [the point (1, 0), which changes nothing]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]

    So: the missing witness of the general ring is exactly that `ω.2` can be divided by. The
    reals supply it at the forced phase, where the second slot is half the square root of three. -/
theorem ringRoot_two_mul_add_one (σ : R) (ω : RingCell R)
    (hroot : ringOne + ω + ringMul σ ω ω = 0) (hb : IsUnit ω.2) :
    1 + 2 * ω.1 = 0 :=
  hb.mul_right_eq_zero.mp (ringRoot_snd σ ω hroot)

/-- The first slot of the order-three relation gives the size of the element.

    σ                                       [the parameter: what the adjoined direction squares to]
    ringMul                                 [multiply two points together, the σ way]
    ringOne                                 [the point (1, 0), which changes nothing]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]

    So: the first slot of `1 + ω + ω² = 0` reads `ω.1² + σ·ω.2² = −(1 + ω.1)`, the value the
    real phase satisfies at `σ = −1`. -/
theorem ringRoot_fst (σ : R) (ω : RingCell R)
    (hroot : ringOne + ω + ringMul σ ω ω = 0) :
    ω.1 ^ 2 + σ * ω.2 ^ 2 = -(1 + ω.1) := by
  have h := (Prod.ext_iff.mp hroot).1
  change 1 + ω.1 + (ω.1 * ω.1 + σ * ω.2 * ω.2) = 0 at h
  linear_combination h

/-- **Twice the size of the three-phase sum is the spread of the weights.**

    σ                                                            [the parameter]
    IsUnit                                                       [can be divided by]
    ringMul                                                      [multiply two points, the σ way]
    ringOne                                                      [the point (1, 0)]
    ((w₁ - w₂) ^ 2 + (w₂ - w₃) ^ 2 + (w₃ - w₁) ^ 2)              [the spread of the three weights]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]
    `ringWeightSpread`          [the size of the three-phase sum over a ring]

    So: fully symmetric in the three weights, so no ordering convention enters. The factor two is
    carried rather than divided out, because two need not be invertible in a general ring; over
    the reals the real specialization divides it back out. The hypotheses are the order-three
    relation and a unit second slot, and neither follows from the ring axioms. -/
theorem ringWeightSpread_eq_spread (σ : R) (ω : RingCell R) (w₁ w₂ w₃ : R)
    (hroot : ringOne + ω + ringMul σ ω ω = 0) (hb : IsUnit ω.2) :
    2 * ringWeightSpread σ ω w₁ w₂ w₃
      = (w₁ - w₂) ^ 2 + (w₂ - w₃) ^ 2 + (w₃ - w₁) ^ 2 := by
  have h2a := ringRoot_two_mul_add_one σ ω hroot hb
  have hfst := ringRoot_fst σ ω hroot
  have hsq := ringRoot_sq σ ω hroot
  have hsum : ringTrinitySum σ ω w₁ w₂ w₃
      = (w₁ - w₃ + ω.1 * (w₂ - w₃), ω.2 * (w₂ - w₃)) := by
    simp only [ringTrinitySum]
    rw [hsq]
    refine Prod.ext_iff.mpr ⟨?_, ?_⟩ <;>
      simp only [ringOne, Prod.fst_add, Prod.snd_add,
        Prod.fst_neg, Prod.snd_neg, Prod.smul_fst, Prod.smul_snd, smul_eq_mul] <;>
      ring
  rw [ringWeightSpread, hsum]
  simp only [ringNormForm]
  linear_combination (-2 * (w₂ - w₃) ^ 2) * hfst
    + (2 * ω.1 * (w₂ - w₃) ^ 2 + 2 * (w₂ - w₃) * (w₁ - w₃)) * h2a

/-- **Three equal weights cancel over any ring.**

    σ                                       [the parameter: what the adjoined direction squares to]
    ringMul                                 [multiply two points together, the σ way]
    ringOne                                 [the point (1, 0), which changes nothing]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]
    `ringWeightSpread`          [the size of the three-phase sum over a ring]

    So: three equal weights on the three phases sum to zero, because the phases sum to zero. The
    amplitude exists only because the weights differ. -/
theorem ringEqualWeights_silence (σ : R) (ω : RingCell R) (w : R)
    (hroot : ringOne + ω + ringMul σ ω ω = 0) :
    ringWeightSpread σ ω w w w = 0 := by
  have hsum : ringTrinitySum σ ω w w w = 0 := by
    simp only [ringTrinitySum]
    rw [← smul_add, ← smul_add, hroot, smul_zero]
  rw [ringWeightSpread, hsum]
  simp [ringNormForm]

/-- A point that keeps the real axis inside itself has an empty second slot.

    σ                                       [the parameter: what the adjoined direction squares to]
    ∀                                       [for every single one]
    →                                       [takes the left and gives the right]
    ringMul                                 [multiply two points together, the σ way]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]

    So: the second slot of `z` multiplying any real point back to the real axis is zero, because
    the unit is real. This is the hypothesis that a form equal to its own conjugate supplies. -/
theorem ringFixedOfRealAxis (σ : R) (z : RingCell R)
    (hreal : ∀ w : RingCell R, w.2 = 0 → (ringMul σ z w).2 = 0) : z.2 = 0 := by
  have h := hreal ringOne rfl
  change z.1 * 0 + z.2 * 1 = 0 at h
  simpa using h

/-- A size-one point fixing the real axis factors its first slot.

    σ                                       [the parameter: what the adjoined direction squares to]
    ∀                                       [for every single one]
    →                                       [takes the left and gives the right]
    ringMul                                 [multiply two points together, the σ way]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    So: with the second slot gone, `normForm σ z = 1` reads `z.1² = 1`, which factors as
    `(z.1 − 1)(z.1 + 1) = 0`. The factorisation is polynomial and needs no unit. -/
theorem ringRephasing_factor (σ : R) (z : RingCell R)
    (hreal : ∀ w : RingCell R, w.2 = 0 → (ringMul σ z w).2 = 0)
    (hnorm : ringNormForm σ z = 1) :
    (z.1 - 1) * (z.1 + 1) = 0 := by
  have hz2 := ringFixedOfRealAxis σ z hreal
  have hsq : z.1 ^ 2 = 1 := by
    have h := hnorm
    simp only [ringNormForm] at h
    rw [hz2] at h
    simpa using h
  linear_combination hsq

/-- **A unit factor turns the fixed-point identity into plus or minus the unit.**

    σ                                                     [the parameter]
    ∀                                                     [for every single one]
    →                                                     [takes the left and gives the right]
    ∨                                                     [at least one is true]
    IsUnit                                                [can be divided by]
    ringMul                                               [multiply two points, the σ way]
    ringOne                                               [the point (1, 0)]
    `RingCell`                  [a point: two ring elements side by side, a + b·x]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    So: a size-one point that keeps the real axis inside itself is `±1` once one of the two
    factors can be divided by. Over the reals that hypothesis always holds away from zero, which
    is the case `Majorana.lean` uses; over a ring with zero divisors it is the missing witness,
    and without it the conclusion is false. -/
theorem ringRephasing_fixes_nothing (σ : R) (z : RingCell R)
    (hreal : ∀ w : RingCell R, w.2 = 0 → (ringMul σ z w).2 = 0)
    (hnorm : ringNormForm σ z = 1)
    (hunit : IsUnit (z.1 - 1) ∨ IsUnit (z.1 + 1)) :
    z = ringOne ∨ z = -ringOne := by
  have hz2 := ringFixedOfRealAxis σ z hreal
  have hfac := ringRephasing_factor σ z hreal hnorm
  rcases hunit with h | h
  · right
    have hz1 : z.1 + 1 = 0 := h.mul_right_eq_zero.mp hfac
    refine Prod.ext_iff.mpr ⟨?_, ?_⟩
    · change z.1 = -1
      linear_combination hz1
    · simpa [ringOne] using hz2
  · left
    have hz1 : z.1 - 1 = 0 := h.mul_left_eq_zero.mp hfac
    refine Prod.ext_iff.mpr ⟨?_, ?_⟩
    · change z.1 = 1
      linear_combination hz1
    · simpa [ringOne] using hz2

/- The generic weighted sum and norm form reduce at the integer instance. -/
#guard ringWeightSpread (R := Int) 1 (0, 1) 1 2 3 == 12

end Cosmology
end Apeiron
