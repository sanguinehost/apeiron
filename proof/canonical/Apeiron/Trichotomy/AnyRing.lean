import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Mathlib.Tactic
import Mathlib.Algebra.Group.Units.Basic
import Mathlib.Data.Matrix.Mul

/-!
# The Two-Slot Algebra Over Any Commutative Ring

@name The Two-Slot Algebra Over Any Commutative Ring
@cite euclid-elements-ii-5
@cite boole-1847
@cite peirce-1870
@cite cockle-1849
@grade proved

Every other file here builds on `Cell`, which is `ℝ × ℝ`. `σ` ranges over the reals, and the
classification runs by the sign of `σ`. A consumer computing in a finite ring gets none of that.
`../apeiron` addresses a hash table with arithmetic in `Z/2^64`, where two is not invertible and
the sign of `σ` is not a thing that exists.

This file carries the same construction over an arbitrary commutative ring, so such a ring is an
instance rather than an analogy. Reusing this corpus's names would have been the wrong move: the
registry's meaning for `Cell` says two real numbers, and a general carrier is a different object
even though the formulas coincide. The `Ring` prefix marks the difference.

What survives leaving `ℝ` splits cleanly.

**The norm form is still multiplicative.** `ringNormForm σ (z · w) = ringNormForm σ z ·
ringNormForm σ w` is a polynomial identity with integer coefficients, so it holds in every
commutative ring or in none. Euclid II.5 is the `σ = 1` case of it.

**Divisibility is still the norm form's question.** `exists_inverse_iff_isUnit_ringNormForm` says
a point can be divided by exactly when its norm form can. Over `ℝ` this reads as lying off the
cone, which is `Basic.isUnit_iff_normForm_ne_zero`; over `Z/2^64` it reads as the norm form being
odd. One theorem, two readings, because in each ring the units are whatever they are.

**The idempotent argument changes shape.** Both proofs reach `b · (2a − 1) = 0`. Over `ℝ` that is
split by having no zero divisors, and the `a = 1/2` branch is where every nontrivial idempotent
comes from, which is the whole content of
`Idempotents.exists_nontrivial_idempotent_iff_disc_pos`. Neither step is available in a general
ring. What replaces the first is narrower and enough: a factor that can be divided by can be
cancelled, whether or not the ring is a domain. What replaces the second is a hypothesis, because
`a · (a − 1) = 0` has more roots in some rings than in `ℝ`.

The two hypotheses are not exotic. In `Z/2^n` every odd element can be divided by, `2a − 1` is odd
for every `a`, and one of `a` and `a − 1` is odd, so both hold at once and the conclusion is that
the only idempotents are the two every ring has. `../apeiron`'s `proofs.rs` discharges exactly
those two facts at machine width, which is the division of labour: the ring-universal shape is
here, and the width-specific facts are checked where the width is.

## Root

The affine-step declarations add a monoid of two-slot steps over the ring. Each law is the
matching `ringMul` law read through a step's coefficient and injection, and the integer witness
pins the chronological order and the explicit `sigma`.

| from | here |
|---|---|
| `ringMul_assoc` | `ringAffineCompose_assoc` |
| `ringMul_distrib_add` | `ringAffineCompose_apply` |
| `ringMul_one`, `ringMul_zero` | `ringAffineCompose_id` |
| `ringOne_mul` | `ringAffineId_compose` |
| `ringAffineCompose_apply` | `ringAffineFold_apply` |
| `ringAffineCompose_assoc` | `ringAffineFold_accum` |
| `ringAffineFold_accum` | `ringAffineFold_append` |

The matrix declarations carry the same monoid over a semiring, with a grid in place of the
coefficient and a vector in place of the point:

| from | here |
|---|---|
| `ringAffineCompose_apply` | `matrixStep_apply` |
| `ringAffineCompose_assoc` | `matrixStep_comp` |
| `ringAffineId_compose`, `ringAffineCompose_id` | `matrixStepId_comp`, `matrixStepComp_id` |
| `ringAffineFold_apply` | `matrixStepFold_apply` |
| `ringAffineFold_append` | `matrixStepFold_append` |

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Trichotomy
namespace AnyRing

variable {R : Type*} [CommRing R]

/-- A point over a commutative ring: two ring elements side by side.

    RingCell R   [the name for one point over the ring R]
    R × R        [an element of R, then another element of R]

    So: written `a + b·x`, the same shape as `Cell` with the reals replaced. -/
abbrev RingCell (R : Type*) [CommRing R] : Type _ := R × R

/-- How to multiply two points.

    (σ : R)                        [the parameter, now an element of the ring itself]
    (z w : RingCell R)             [the two points being multiplied]
    z.1 * w.1 + σ * z.2 * w.2      [the first slot of the answer]
    z.1 * w.2 + z.2 * w.1          [the second slot of the answer]

    So: `(a+bx)(c+dx) = (ac + σbd) + (ad+bc)x`, with `x·x = σ`. `σ` enters at one place
    only, the `σbd` term, and it is a ring element rather than a real number. -/
@[simp] abbrev ringMul (σ : R) (z w : RingCell R) : RingCell R :=
  (z.1 * w.1 + σ * z.2 * w.2, z.1 * w.2 + z.2 * w.1)

/-- The conjugate: the second slot negated, first slot kept.

    (z : RingCell R)        [the point]
    (z.1, -z.2)             [first slot, and the negation of the second]

    So: `a - b·x`, the reflection that turns a product into its size. -/
@[simp] abbrev ringConj (z : RingCell R) : RingCell R := (z.1, -z.2)

/-- The size of a point.

    (σ : R)                  [the parameter]
    (z : RingCell R)         [the point being measured]
    z.1 ^ 2 - σ * z.2 ^ 2    [first slot squared, less σ times the second squared]
    : R                      [an element of the ring]

    So: `N(a + bx) = a² − σb²`, the left-hand side of Euclid II.5 at `σ = 1`. -/
@[simp] abbrev ringNormForm (σ : R) (z : RingCell R) : R := z.1 ^ 2 - σ * z.2 ^ 2

/-- The bilinear form: two points paired.

    (σ : R)                    [the parameter]
    (z w : RingCell R)         [the two points]
    z.1 * w.1 - σ * z.2 * w.2  [their pairing]
    : R                        [an element of the ring]

    So: `ringNormForm` opened out to take two different points. Given the same point
    twice it returns that point's norm form. -/
abbrev ringForm (σ : R) (z w : RingCell R) : R := z.1 * w.1 - σ * z.2 * w.2

/-- The point that changes nothing under multiplication.

    ringOne      [the name for this point]
    (1, 0)       [one in the first slot, zero in the second]
    : RingCell R [a point over the ring] -/
@[simp] abbrev ringOne : RingCell R := (1, 0)

/-- A point unchanged by squaring.

    (σ : R)                  [the parameter]
    (z : RingCell R)         [the point]
    ringMul σ z z = z        [multiplying it by itself gives it back]

    So: Boole's `x² = x`, asked over a ring rather than over the reals. -/
def RingIdempotent (σ : R) (z : RingCell R) : Prop := ringMul σ z z = z

/-- The size of a product is the product of the sizes.

    (σ : R)                        [the parameter]
    (z w : RingCell R)             [the two points]
    ringNormForm σ (ringMul σ z w) [the size of their product]
    =                              [is equal to]
    ringNormForm σ z * ringNormForm σ w   [their sizes multiplied]

    So: the one law that crosses every commutative ring, because it is an identity between
    two polynomials with whole-number coefficients. At `σ = 1` it is Euclid II.5 and at
    `σ = −1` it is the sum-of-two-squares identity. -/
theorem ringNormForm_mul (σ : R) (z w : RingCell R) :
    ringNormForm σ (ringMul σ z w) = ringNormForm σ z * ringNormForm σ w := by
  simp only [ringNormForm]
  ring

/-- Multiplying two points does not depend on their order.

    (σ : R)                    [the parameter]
    (z w : RingCell R)         [the two points]
    ringMul σ z w = ringMul σ w z   [multiplying either way round gives the same point]

    So: each slot commutes, so the whole product does. -/
theorem ringMul_comm (σ : R) (z w : RingCell R) : ringMul σ z w = ringMul σ w z := by
  simp only [ringMul, Prod.ext_iff]
  constructor <;> ring

/-- Multiplying three points does not depend on the bracketing.

    (σ : R)                    [the parameter]
    (z w v : RingCell R)       [three points]
    ringMul σ (ringMul σ z w) v = ringMul σ z (ringMul σ w v)   [the same point either way]

    So: each slot associates, so the whole product does. -/
theorem ringMul_assoc (σ : R) (z w v : RingCell R) :
    ringMul σ (ringMul σ z w) v = ringMul σ z (ringMul σ w v) := by
  simp only [ringMul, Prod.ext_iff]
  constructor <;> ring

/-- The unit changes nothing under multiplication.

    (σ : R)                    [the parameter]
    (z : RingCell R)           [any point]
    ringMul σ ringOne z = z    [multiply the unit by z, get z back]

    So: one in the first slot and zero in the second acts as the identity. -/
theorem ringOne_mul (σ : R) (z : RingCell R) : ringMul σ ringOne z = z := by
  simp [ringMul]

/-- The unit changes nothing under multiplication on the other side.

    (σ : R)                    [the parameter]
    (z : RingCell R)           [any point]
    ringMul σ z ringOne = z    [multiply z by the unit, get z back]

    So: `ringOne_mul` read through commutativity. -/
theorem ringMul_one (σ : R) (z : RingCell R) : ringMul σ z ringOne = z := by
  rw [ringMul_comm, ringOne_mul]

/-- The origin absorbs on the right.

    (σ : R)                    [the parameter]
    (z : RingCell R)           [any point]
    ringMul σ z 0 = 0          [the origin times z is the origin]

    So: both slots of the product are zero. -/
theorem ringMul_zero (σ : R) (z : RingCell R) : ringMul σ z 0 = 0 := by
  simp [ringMul]

/-- Multiplication spreads across a sum on the right.

    (σ : R)                    [the parameter]
    (z w v : RingCell R)       [three points]
    ringMul σ z (w + v) = ringMul σ z w + ringMul σ z v   [multiply each half and add]

    So: the coordinate formula makes this an identity in the four slots. -/
theorem ringMul_distrib_add (σ : R) (z w v : RingCell R) :
    ringMul σ z (w + v) = ringMul σ z w + ringMul σ z v := by
  simp only [ringMul, Prod.ext_iff, Prod.fst_add, Prod.snd_add]
  constructor <;> ring

/-- Multiplication spreads across a difference on the right.

    (σ : R)                    [the parameter]
    (z w v : RingCell R)       [three points]
    ringMul σ z (w - v) = ringMul σ z w - ringMul σ z v   [multiply each half and subtract]

    So: the sum law with a sign. -/
theorem ringMul_distrib_sub (σ : R) (z w v : RingCell R) :
    ringMul σ z (w - v) = ringMul σ z w - ringMul σ z v := by
  simp only [ringMul, Prod.ext_iff, Prod.fst_sub, Prod.snd_sub]
  constructor <;> ring

/-- The bilinear form on a point against itself is the norm form.

    (σ : R)                    [the parameter]
    (z : RingCell R)           [the point]
    ringNormForm σ z = ringForm σ z z   [the norm form is the point paired with itself]

    So: the two definitions agree when the second is given the same point twice. -/
theorem ringForm_self (σ : R) (z : RingCell R) : ringNormForm σ z = ringForm σ z z := by
  simp only [ringForm, ringNormForm]
  ring

/-- Multiplying both arguments scales the pairing by the multiplier's norm form.

    (σ : R)                        [the parameter]
    (u z w : RingCell R)           [three points]
    ringForm σ (ringMul σ u z) (ringMul σ u w)   [pair the two products]
    = ringNormForm σ u * ringForm σ z w          [the size times the old pairing]

    So: multiplication is a similarity of the form, and at size one an isometry. -/
theorem ringForm_mul (σ : R) (u z w : RingCell R) :
    ringForm σ (ringMul σ u z) (ringMul σ u w) = ringNormForm σ u * ringForm σ z w := by
  simp only [ringForm, ringNormForm]
  ring

/-- Conjugating twice returns the point.

    (z : RingCell R)           [the point]
    ringConj (ringConj z) = z  [conjugating twice changes nothing]

    So: negation is an involution on the second slot. -/
theorem ringConj_conj (z : RingCell R) : ringConj (ringConj z) = z := by
  simp [ringConj]

/-- Stretching a point squares into its norm form.

    (σ r : R)                    [the parameter, and a stretch factor]
    (z : RingCell R)             [the point]
    ringNormForm σ (r • z)       [stretch by r, then measure]
    = r ^ 2 * ringNormForm σ z   [the factor squared, times the size]

    So: the norm form is quadratic, so a stretch factor comes out squared. -/
theorem ringNormForm_smul (σ r : R) (z : RingCell R) :
    ringNormForm σ (r • z) = r ^ 2 * ringNormForm σ z := by
  simp only [ringNormForm, Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
  ring

/-- Multiplying a point by its conjugate gives its size.

    (σ : R)                            [the parameter]
    (z : RingCell R)                   [the point]
    ringMul σ z (z.1, -z.2)            [the point times its second slot negated]
    =                                  [is equal to]
    (ringNormForm σ z, 0)              [its size, sitting in the first slot]
    `ringConj`                  [the same point with its second slot negated]

    So: `(a+bx)(a−bx) = a² − σb²`. This is what turns a divisible size into a divisible
    point, with no square root and no order anywhere in it. -/
theorem ringMul_conj (σ : R) (z : RingCell R) :
    ringMul σ z (ringConj z) = (ringNormForm σ z, 0) := by
  simp only [ringNormForm, ringMul, Prod.mk.injEq]
  constructor <;> ring

/-- Conjugating a point leaves its norm form alone.

    (σ : R)                        [the parameter]
    (z : RingCell R)               [the point]
    ringNormForm σ (ringConj z)    [the size of the point with its second slot negated]
    = ringNormForm σ z             [the size of the point]

    So: `ringMul_conj` at `z` and at `ringConj z` sends both `z · conj z` and
    `conj z · conj (conj z)` to `(size, 0)`, and `ringMul_comm` makes the two products one,
    so the two sizes agree. Negating a slot cannot change a square, and no inverse appears. -/
theorem ringNormForm_conj (σ : R) (z : RingCell R) :
    ringNormForm σ (ringConj z) = ringNormForm σ z := by
  have h1 : (ringMul σ z (ringConj z)).1 = ringNormForm σ z := by
    simpa using congrArg Prod.fst (ringMul_conj σ z)
  have h2 : (ringMul σ (ringConj z) z).1 = ringNormForm σ (ringConj z) := by
    simpa [ringConj_conj] using congrArg Prod.fst (ringMul_conj σ (ringConj z))
  rw [← h2, ← h1, ringMul_comm]

/-- The scalar Cayley factor has norm form one.

    (σ : R)                        [the parameter]
    (θ : R)                        [the factor's angle, a ring element]
    (w : RingCell R)               [a point claimed to invert `1 + θ·x`]
    ringMul σ ((1, θ)) w = ringOne [the claimed inverse multiplies out to the unit]
    ringNormForm σ (ringMul σ (ringConj (1, θ)) w) = 1
                                   [the factor's size is one]
    `ringMul`                      [multiply two points together, the σ way]
    `ringConj`                     [the same point with its second slot negated]
    `ringNormForm`                 [the norm form: a point times its conjugate, negative or not]
    `ringOne`                      [the point (1, 0), which changes nothing when multiplied by]

    So: `(1 − θ·x)` times the inverse of `(1 + θ·x)`, the ring form of
    `(1 − θ·i)/(1 + θ·i)`. `ringNormForm_mul` splits the product's size,
    `ringNormForm_conj` makes numerator and denominator share a size, and `hw` says that size
    times the inverse's size is one. This is the scalar Cayley factor's size being one, with
    the reals replaced by `R` and `σ` free. -/
theorem ringCayley_normForm (σ θ : R) (w : RingCell R)
    (hw : ringMul σ ((1 : R), θ) w = ringOne) :
    ringNormForm σ (ringMul σ (ringConj ((1 : R), θ)) w) = 1 := by
  have hN : ringNormForm σ ((1 : R), θ) * ringNormForm σ w = 1 := by
    have h := congrArg (ringNormForm σ) hw
    rw [ringNormForm_mul] at h
    simpa [ringNormForm, ringOne] using h
  rw [ringNormForm_mul, ringNormForm_conj, hN]

/-- A point can be divided by exactly when its size can.

    (σ : R)                                  [the parameter]
    (z : RingCell R)                         [the point]
    ∃ w : RingCell R, ringMul σ z w = ringOne   [there is at least one point z multiplies to one]
    ↔                                        [exactly when, true in both directions]
    IsUnit (ringNormForm σ z)                [something multiplies with its size to give one]

    So: `Basic.isUnit_iff_normForm_ne_zero` over `ℝ` reads its right-hand side as lying off the
    cone, because the divisors of `ℝ` are the non-zero reals. Over `Z/2^n` the same statement
    reads as the size being odd. The theorem does not change; what can be divided by does.

    The inverse is written out rather than asserted: it is the conjugate scaled by the size's
    own inverse, which is why no square root and no order appear. Stated with an explicit
    witness because `R × R` already carries a slot-by-slot product from the library, and that
    is not this one. -/
theorem exists_inverse_iff_isUnit_ringNormForm (σ : R) (z : RingCell R) :
    (∃ w : RingCell R, ringMul σ z w = ringOne) ↔ IsUnit (ringNormForm σ z) := by
  constructor
  · rintro ⟨w, hw⟩
    refine IsUnit.of_mul_eq_one (ringNormForm σ w) ?_
    rw [← ringNormForm_mul, hw]
    simp [ringNormForm]
  · rintro ⟨u, hu⟩
    refine ⟨((↑u⁻¹ : R) * z.1, -((↑u⁻¹ : R) * z.2)), ?_⟩
    have hN : (↑u⁻¹ : R) * (z.1 ^ 2 - σ * z.2 ^ 2) = 1 := by
      have : (↑u⁻¹ : R) * ringNormForm σ z = 1 := by rw [← hu]; exact u.inv_mul
      simpa [ringNormForm] using this
    simp only [ringMul, ringOne, Prod.mk.injEq]
    refine ⟨?_, by ring⟩
    linear_combination hN

/-- An idempotent's second slot vanishes whenever twice its first slot, less one, can be
divided by.

    (σ : R)                        [the parameter]
    (z : RingCell R)               [the point]
    IsUnit (2 * z.1 - 1)           [twice the first slot less one can be divided by]
    RingIdempotent σ z             [the point is unchanged by squaring]
    z.2 = 0                        [its second slot is zero]

    So: the second slot of `z·z = z` says `2·a·b = b`, which is `b·(2a − 1) = 0`. Over `ℝ`
    that is split by having no zero divisors. Here it is split by cancelling a divisor, which
    a ring with zero divisors still allows. `σ` is never consulted. -/
theorem ringIdempotent_snd_eq_zero {σ : R} {z : RingCell R}
    (hu : IsUnit (2 * z.1 - 1)) (h : RingIdempotent σ z) : z.2 = 0 := by
  have h2 : z.1 * z.2 + z.2 * z.1 = z.2 := by
    have := congrArg Prod.snd h
    simpa [ringMul] using this
  have hz : z.2 * (2 * z.1 - 1) = 0 := by linear_combination h2
  exact hu.mul_left_eq_zero.mp hz

/-- With one further divisor, an idempotent is one of the two every ring has.

    (σ : R)                                [the parameter]
    (z : RingCell R)                       [the point]
    IsUnit (2 * z.1 - 1)                   [twice the first slot less one can be divided by]
    IsUnit z.1 ∨ IsUnit (z.1 - 1)          [at least one is true of the first slot and one less]
    RingIdempotent σ z                     [the point is unchanged by squaring]
    z = (0, 0) ∨ z = ringOne               [the point is the origin or the unit]

    So: with the second slot gone the first satisfies `a·(a − 1) = 0`, and cancelling whichever
    factor can be divided by leaves the other at zero. Both hypotheses hold at every `a` in
    `Z/2^n`, where the divisors are the odd elements, so there the conclusion is unconditional
    and holds at every `σ`. That is the sense in which
    `Idempotents.exists_nontrivial_idempotent_iff_disc_pos` does not cross: over such a ring
    its left-hand side is false everywhere, so it stops telling the three faces apart. -/
theorem ringIdempotent_trivial {σ : R} {z : RingCell R}
    (hu : IsUnit (2 * z.1 - 1)) (hfst : IsUnit z.1 ∨ IsUnit (z.1 - 1))
    (h : RingIdempotent σ z) : z = (0, 0) ∨ z = ringOne := by
  have hb : z.2 = 0 := ringIdempotent_snd_eq_zero hu h
  have h1 : z.1 * z.1 + σ * z.2 * z.2 = z.1 := by
    have := congrArg Prod.fst h
    simpa [ringMul] using this
  have hsq : z.1 * (z.1 - 1) = 0 := by rw [hb] at h1; linear_combination h1
  rcases hfst with ha | ha
  · right
    have h0 : z.1 - 1 = 0 := ha.mul_right_eq_zero.mp hsq
    exact Prod.ext (sub_eq_zero.mp h0) hb
  · left
    exact Prod.ext (ha.mul_left_eq_zero.mp hsq) hb

/-- **One affine step over a commutative ring: a coefficient and an injection.**

    RingAffineStep R   [the name for one such step over the ring R]
    RingCell R         [a point: two ring elements side by side, a + b·x]
    coefficient        [the point the step multiplies by]
    injection          [the point the step adds afterwards]

    So: the step sends `z` to `coefficient · z + injection`, the two-slot form of `S ↦ a·S + b`
    over the ring. The coefficient is an ordinary weight, not a rotor. -/
structure RingAffineStep (R : Type*) [CommRing R] where
  coefficient : RingCell R
  injection   : RingCell R

/-- **The identity affine step: coefficient one, injection zero.**

    ringAffineId R    [the step that changes nothing, over the ring R]
    RingAffineStep R  [a step over the ring]

    So: `ringAffineApply σ (ringAffineId R) z = z` for every point `z`. -/
def ringAffineId (R : Type*) [CommRing R] : RingAffineStep R := ⟨ringOne, 0⟩

/-- **Applying one affine step: multiply by the coefficient, then add the injection.**

    (sigma : R)               [the parameter: what the adjoined direction squares to]
    (step : RingAffineStep R) [the step being applied]
    (z : RingCell R)          [the point the step is applied to]
    RingCell R                [a point: two ring elements side by side, a + b·x]

    So: the step sends `z` to `coefficient · z + injection`, with `sigma` explicit in the
    multiplication. -/
def ringAffineApply (sigma : R) (step : RingAffineStep R) (z : RingCell R) : RingCell R :=
  ringMul sigma step.coefficient z + step.injection

/-- **Composing two affine steps: apply the inner step, then the outer step.**

    (sigma : R)                      [the parameter]
    (outer inner : RingAffineStep R) [the step applied second, and the step applied first]
    RingAffineStep R                 [the composed step]

    So: the composed coefficient is `outer.coefficient · inner.coefficient` and the composed
    injection is `outer.coefficient · inner.injection + outer.injection`, the inner injection
    carried forward by the outer coefficient. -/
def ringAffineCompose (sigma : R) (outer inner : RingAffineStep R) : RingAffineStep R where
  coefficient := ringMul sigma outer.coefficient inner.coefficient
  injection   := ringMul sigma outer.coefficient inner.injection + outer.injection

/-- **The left fold of a list of steps into one step.**

    (sigma : R)                       [the parameter]
    (steps : List (RingAffineStep R)) [the steps in the order they are applied]
    List                              [a finite sequence of things, in order]
    RingAffineStep R                  [the single step the list composes to]

    So: start from the identity and compose each step on the outside, so the first step in the
    list is applied first.

    Complexity: `n` list steps cost two `ringMul` and one cell addition each, so `O(n)` cell
    operations in the list length and `O(1)` retained accumulator cells for the tail fold,
    excluding the input list and the scalar representation. No proof-elaboration or
    hardware-timing claim is made. -/
def ringAffineFold (sigma : R) (steps : List (RingAffineStep R)) : RingAffineStep R :=
  steps.foldl (fun acc step => ringAffineCompose sigma step acc) (ringAffineId R)

/-- **The identity step changes nothing.**

    (sigma : R)      [the parameter]
    (z : RingCell R) [the point]
    RingCell R       [a point: two ring elements side by side, a + b·x]

    So: the coefficient is one and the injection is zero, closed by `ringOne_mul` and
    `add_zero`. -/
theorem ringAffineId_apply (sigma : R) (z : RingCell R) :
    ringAffineApply sigma (ringAffineId R) z = z := by
  simp only [ringAffineApply, ringAffineId]
  rw [ringOne_mul, add_zero]

/-- **The identity step composed on the outside changes nothing.**

    (sigma : R)               [the parameter]
    (step : RingAffineStep R) [the step]
    RingAffineStep R          [a step over the ring]

    So: the left unit of the step monoid, closed by `ringOne_mul` and `add_zero`. -/
theorem ringAffineId_compose (sigma : R) (step : RingAffineStep R) :
    ringAffineCompose sigma (ringAffineId R) step = step := by
  cases step with
  | mk coefficient injection =>
    simp only [ringAffineCompose, ringAffineId, RingAffineStep.mk.injEq]
    constructor
    · rw [ringOne_mul]
    · rw [ringOne_mul, add_zero]

/-- **The identity step composed on the inside changes nothing.**

    (sigma : R)               [the parameter]
    (step : RingAffineStep R) [the step]
    RingAffineStep R          [a step over the ring]

    So: the right unit of the step monoid, closed by `ringMul_one` and `ringMul_zero`. -/
theorem ringAffineCompose_id (sigma : R) (step : RingAffineStep R) :
    ringAffineCompose sigma step (ringAffineId R) = step := by
  cases step with
  | mk coefficient injection =>
    simp only [ringAffineCompose, ringAffineId, RingAffineStep.mk.injEq]
    constructor
    · rw [ringMul_one]
    · rw [ringMul_zero, zero_add]

/-- **Applying a composition is applying the inner step then the outer step.**

    (sigma : R)                      [the parameter]
    (outer inner : RingAffineStep R) [the step applied second, and the step applied first]
    (z : RingCell R)                 [the point the composed step is applied to]
    RingCell R                       [a point: two ring elements side by side, a + b·x]

    So: the two sides agree by `ringMul_assoc`, `ringMul_distrib_add` and `add_assoc`, with no
    polynomial expansion. -/
theorem ringAffineCompose_apply (sigma : R) (outer inner : RingAffineStep R) (z : RingCell R) :
    ringAffineApply sigma (ringAffineCompose sigma outer inner) z =
      ringAffineApply sigma outer (ringAffineApply sigma inner z) := by
  simp only [ringAffineApply, ringAffineCompose]
  rw [ringMul_assoc, ringMul_distrib_add, add_assoc]

/-- **Composing three steps does not depend on the bracketing.**

    (sigma : R)                [the parameter]
    (a b c : RingAffineStep R) [the three steps]
    RingAffineStep R           [a step over the ring]

    So: each slot associates by `ringMul_assoc` and the injection by `ringMul_distrib_add`, so
    the whole step associates. -/
theorem ringAffineCompose_assoc (sigma : R) (a b c : RingAffineStep R) :
    ringAffineCompose sigma (ringAffineCompose sigma a b) c =
      ringAffineCompose sigma a (ringAffineCompose sigma b c) := by
  cases a with | mk a1 a2 =>
  cases b with | mk b1 b2 =>
  cases c with | mk c1 c2 =>
  simp only [ringAffineCompose, RingAffineStep.mk.injEq]
  constructor
  · rw [ringMul_assoc]
  · rw [ringMul_assoc, ringMul_distrib_add, add_assoc]

/-- **The fold accumulator: composing from an arbitrary starting step.**

    (sigma : R)                       [the parameter]
    (steps : List (RingAffineStep R)) [the steps]
    List                              [a finite sequence of things, in order]
    (acc : RingAffineStep R)          [the step the fold starts from]

    So: folding from `acc` is the fold from the identity composed on the outside of `acc`, the
    accumulator form `ringAffineFold_append` needs. -/
private theorem ringAffineFold_accum (sigma : R) (steps : List (RingAffineStep R))
    (acc : RingAffineStep R) :
    steps.foldl (fun acc step => ringAffineCompose sigma step acc) acc =
      ringAffineCompose sigma
        (steps.foldl (fun acc step => ringAffineCompose sigma step acc) (ringAffineId R)) acc := by
  induction steps generalizing acc with
  | nil => simp [ringAffineId_compose]
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (ringAffineCompose sigma s acc),
      ih (ringAffineCompose sigma s (ringAffineId R))]
    rw [ringAffineCompose_id, ringAffineCompose_assoc]

/-- **The fold applied to a point, with an arbitrary starting step.**

    (sigma : R)                       [the parameter]
    (steps : List (RingAffineStep R)) [the steps]
    List                              [a finite sequence of things, in order]
    (acc : RingAffineStep R)          [the step the fold starts from]
    (z : RingCell R)                  [the point]
    RingCell R                        [a point: two ring elements side by side, a + b·x]

    So: the accumulator form of `ringAffineFold_apply`, by `ringAffineCompose_apply`. -/
private theorem ringAffineApply_fold (sigma : R) (steps : List (RingAffineStep R))
    (acc : RingAffineStep R) (z : RingCell R) :
    ringAffineApply sigma
        (steps.foldl (fun acc step => ringAffineCompose sigma step acc) acc) z =
      steps.foldl (fun z step => ringAffineApply sigma step z) (ringAffineApply sigma acc z) := by
  induction steps generalizing acc z with
  | nil => rfl
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (ringAffineCompose sigma s acc) z, ringAffineCompose_apply]

/-- **Applying the fold is the left fold of the applies.**

    (sigma : R)                       [the parameter]
    (steps : List (RingAffineStep R)) [the steps in the order they are applied]
    List                              [a finite sequence of things, in order]
    (z : RingCell R)                  [the point]
    RingCell R                        [a point: two ring elements side by side, a + b·x]

    So: the composed step applied once equals applying the steps one after another, so a prefix
    scan and a sequential loop agree. -/
theorem ringAffineFold_apply (sigma : R) (steps : List (RingAffineStep R)) (z : RingCell R) :
    ringAffineApply sigma (ringAffineFold sigma steps) z =
      steps.foldl (fun acc step => ringAffineApply sigma step acc) z := by
  rw [ringAffineFold, ringAffineApply_fold, ringAffineId_apply]

/-- **Appending two step lists composes the two folds.**

    (sigma : R)                           [the parameter]
    (steps more : List (RingAffineStep R)) [the first list, then the list appended after it]
    List                                  [a finite sequence of things, in order]
    ++                                    [the two lists joined, the left one's entries first]

    So: the fold of the appended list is the fold of the second list composed on the outside of
    the fold of the first. -/
theorem ringAffineFold_append (sigma : R) (steps more : List (RingAffineStep R)) :
    ringAffineFold sigma (steps ++ more) =
      ringAffineCompose sigma (ringAffineFold sigma more) (ringAffineFold sigma steps) := by
  rw [ringAffineFold, List.foldl_append, ringAffineFold_accum]
  rfl

/-- **The integer witness: the chronological order and the explicit `sigma` both show.**

    (1 : ℤ)      [the parameter at the split face, one]
    weighted     [the step with coefficient (3, 1) and zero injection]
    inject       [the step with coefficient (1, 1) and injection (1, 1)]
    ∧            [both are true]
    ≠            [is not equal to]

    So: folding `[inject, weighted]` and applying to the origin at `sigma = 1` gives `(4, 4)`,
    the reverse order gives `(1, 1)`, and the same forward fold at `sigma = 0` gives `(3, 4)`, so
    the fold is chronological and reads `sigma` in the coefficient's second slot times the
    payload's second slot. `weighted` is an ordinary weighted step, not an SO(2) rotor. -/
theorem ringAffine_hyperbolic_witness :
    (let weighted : RingAffineStep ℤ := ⟨((3 : ℤ), (1 : ℤ)), 0⟩
     let inject : RingAffineStep ℤ := ⟨((1 : ℤ), (1 : ℤ)), ((1 : ℤ), (1 : ℤ))⟩
     ringAffineApply (1 : ℤ) (ringAffineFold (1 : ℤ) [inject, weighted]) 0 = ((4 : ℤ), (4 : ℤ)) ∧
     ringAffineApply (1 : ℤ) (ringAffineFold (1 : ℤ) [weighted, inject]) 0 = ((1 : ℤ), (1 : ℤ)) ∧
     ringAffineApply (0 : ℤ) (ringAffineFold (0 : ℤ) [inject, weighted]) 0 = ((3 : ℤ), (4 : ℤ)) ∧
     ringAffineFold (1 : ℤ) [inject, weighted] ≠ ringAffineFold (1 : ℤ) [weighted, inject]) := by
  dsimp only
  have h1 : ringAffineApply (1 : ℤ)
      (ringAffineFold (1 : ℤ) [⟨((1 : ℤ), (1 : ℤ)), ((1 : ℤ), (1 : ℤ))⟩,
        ⟨((3 : ℤ), (1 : ℤ)), 0⟩]) 0 = ((4 : ℤ), (4 : ℤ)) := by
    rw [ringAffineFold_apply]
    simp only [List.foldl_cons, List.foldl_nil]
    simp only [ringAffineApply, ringMul]
    norm_num
  have h2 : ringAffineApply (1 : ℤ)
      (ringAffineFold (1 : ℤ) [⟨((3 : ℤ), (1 : ℤ)), 0⟩,
        ⟨((1 : ℤ), (1 : ℤ)), ((1 : ℤ), (1 : ℤ))⟩]) 0 = ((1 : ℤ), (1 : ℤ)) := by
    rw [ringAffineFold_apply]
    simp only [List.foldl_cons, List.foldl_nil]
    simp only [ringAffineApply, ringMul]
    norm_num
  have h3 : ringAffineApply (0 : ℤ)
      (ringAffineFold (0 : ℤ) [⟨((1 : ℤ), (1 : ℤ)), ((1 : ℤ), (1 : ℤ))⟩,
        ⟨((3 : ℤ), (1 : ℤ)), 0⟩]) 0 = ((3 : ℤ), (4 : ℤ)) := by
    rw [ringAffineFold_apply]
    simp only [List.foldl_cons, List.foldl_nil]
    simp only [ringAffineApply, ringMul]
    norm_num
  refine ⟨h1, h2, h3, ?_⟩
  intro h
  have h4 := congrArg (fun s : RingAffineStep ℤ => ringAffineApply (1 : ℤ) s 0) h
  rw [h1, h2] at h4
  norm_num at h4

section MatrixAffine

variable {ι : Type*} [Fintype ι] [DecidableEq ι]
variable {S : Type*} [Semiring S]

open Matrix

/-- **One affine step over a semiring: a grid and an injection.**

    MatrixAffineStep ι S   [the name for one such step over the index set ι and the ring S]
    Matrix ι ι S           [a rectangle of ring elements, indexed by row and column]
    linear                 [the grid the step multiplies by]
    injection              [the vector the step adds afterwards]
    ι → S                  [the index set, then an element of S for each one]

    So: the step sends `x` to `linear *ᵥ x + injection`, the matrix form of `x ↦ A·x + b` over
    any semiring. The `→` in the injection's type [takes the left and gives the right]. -/
structure MatrixAffineStep (ι : Type*) [Fintype ι] [DecidableEq ι]
    (S : Type*) [Semiring S] where
  linear : Matrix ι ι S
  injection : ι → S

/-- **Applying one affine step: multiply by the grid, then add the injection.**

    (s : MatrixAffineStep ι S)   [the step being applied]
    (x : ι → S)                  [the vector the step is applied to]
    ι → S                        [the index set, then an element of S for each one]

    So: the step sends `x` to `linear *ᵥ x + injection`. The `→` in the type
    [takes the left and gives the right]. -/
def matrixAffine (s : MatrixAffineStep ι S) (x : ι → S) : ι → S :=
  s.linear *ᵥ x + s.injection

/-- **Composing two affine steps: apply the inner step, then the outer step.**

    (outer inner : MatrixAffineStep ι S)   [the step applied second, and the step applied first]
    MatrixAffineStep ι S                   [the composed step]

    So: the composed grid is `outer.linear * inner.linear` and the composed injection is
    `outer.linear *ᵥ inner.injection + outer.injection`, the inner injection carried forward by
    the outer grid. -/
def matrixStepComp (outer inner : MatrixAffineStep ι S) : MatrixAffineStep ι S where
  linear := outer.linear * inner.linear
  injection := outer.linear *ᵥ inner.injection + outer.injection

/-- **Applying a composition is applying the inner step then the outer step.**

    (outer inner : MatrixAffineStep ι S)   [the step applied second, and the step applied first]
    (x : ι → S)                            [the vector the composed step is applied to]
    ι → S                                  [the index set, then an element of S for each one]

    So: the two sides agree by `mulVec_mulVec`, `mulVec_add` and `add_assoc`, with no coordinate
    expansion. The `→` in the type [takes the left and gives the right]. -/
theorem matrixStep_apply (outer inner : MatrixAffineStep ι S) (x : ι → S) :
    matrixAffine (matrixStepComp outer inner) x = matrixAffine outer (matrixAffine inner x) := by
  simp only [matrixAffine, matrixStepComp, Matrix.mulVec_add]
  rw [← Matrix.mulVec_mulVec, add_assoc]

/-- **Composing three steps does not depend on the bracketing.**

    (a b c : MatrixAffineStep ι S)   [the three steps]
    MatrixAffineStep ι S             [the composed step]

    So: the grids associate by `mul_assoc` and the injections by `mulVec_mulVec`, `mulVec_add`
    and `add_assoc`, so the whole step associates. -/
theorem matrixStep_comp (a b c : MatrixAffineStep ι S) :
    matrixStepComp (matrixStepComp a b) c = matrixStepComp a (matrixStepComp b c) := by
  cases a with | mk a1 a2 =>
  cases b with | mk b1 b2 =>
  cases c with | mk c1 c2 =>
  simp only [matrixStepComp, MatrixAffineStep.mk.injEq]
  constructor
  · rw [Matrix.mul_assoc]
  · rw [← Matrix.mulVec_mulVec, Matrix.mulVec_add, add_assoc]

/-- **The identity affine step: grid one, injection zero.**

    MatrixAffineStep ι S   [the step that changes nothing]
    1                      [the grid that changes nothing when multiplying]
    0                      [the vector all of whose entries are zero]

    So: `matrixAffine matrixStepId x = x` for every vector `x`. -/
def matrixStepId : MatrixAffineStep ι S := ⟨1, 0⟩

/-- **The identity step changes nothing.**

    (x : ι → S)   [the vector]
    ι → S         [the index set, then an element of S for each one]

    So: the grid is one and the injection is zero, closed by `one_mulVec` and `add_zero`. The
    `→` in the type [takes the left and gives the right]. -/
theorem matrixStepId_apply (x : ι → S) : matrixAffine matrixStepId x = x := by
  simp [matrixAffine, matrixStepId]

/-- **The identity step composed on the outside changes nothing.**

    (s : MatrixAffineStep ι S)   [the step]
    MatrixAffineStep ι S         [a step]

    So: the left unit of the step monoid, closed by `one_mul`, `one_mulVec` and `add_zero`. -/
theorem matrixStepId_comp (s : MatrixAffineStep ι S) :
    matrixStepComp matrixStepId s = s := by
  cases s with | mk l i =>
  simp only [matrixStepComp, matrixStepId, MatrixAffineStep.mk.injEq]
  constructor
  · rw [Matrix.one_mul]
  · rw [Matrix.one_mulVec, add_zero]

/-- **The identity step composed on the inside changes nothing.**

    (s : MatrixAffineStep ι S)   [the step]
    MatrixAffineStep ι S         [a step]

    So: the right unit of the step monoid, closed by `mul_one`, `mulVec_zero` and `zero_add`. -/
theorem matrixStepComp_id (s : MatrixAffineStep ι S) :
    matrixStepComp s matrixStepId = s := by
  cases s with | mk l i =>
  simp only [matrixStepComp, matrixStepId, MatrixAffineStep.mk.injEq]
  constructor
  · rw [Matrix.mul_one]
  · rw [Matrix.mulVec_zero, zero_add]

/-- **The left fold of a list of steps into one step.**

    (steps : List (MatrixAffineStep ι S))   [the steps in the order they are applied]
    List                                    [a finite sequence of things, in order]
    MatrixAffineStep ι S                    [the single step the list composes to]

    So: start from the identity and compose each step on the outside, so the first step in the
    list is applied first.

    Complexity: `n` steps cost one `d`-by-`d` grid product and one grid-vector product each,
    where `d` is the size of `ι`, so `O(n·d^3)` ring operations and `O(1)` retained accumulator
    cells for the tail fold, excluding the input list and the scalar representation. No
    proof-elaboration or hardware-timing claim is made. -/
def matrixStepFold (steps : List (MatrixAffineStep ι S)) : MatrixAffineStep ι S :=
  steps.foldl (fun acc step => matrixStepComp step acc) matrixStepId

/-- **The fold accumulator: composing from an arbitrary starting step.**

    (steps : List (MatrixAffineStep ι S))   [the steps]
    List                                    [a finite sequence of things, in order]
    (acc : MatrixAffineStep ι S)            [the step the fold starts from]

    So: folding from `acc` is the fold from the identity composed on the outside of `acc`, the
    accumulator form `matrixStepFold_append` needs. -/
private theorem matrixStepFold_accum (steps : List (MatrixAffineStep ι S))
    (acc : MatrixAffineStep ι S) :
    steps.foldl (fun acc step => matrixStepComp step acc) acc =
      matrixStepComp
        (steps.foldl (fun acc step => matrixStepComp step acc) matrixStepId) acc := by
  induction steps generalizing acc with
  | nil => simp [matrixStepId_comp]
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (matrixStepComp s acc), ih (matrixStepComp s matrixStepId)]
    rw [matrixStepComp_id, matrixStep_comp]

/-- **The fold applied to a vector, with an arbitrary starting step.**

    (steps : List (MatrixAffineStep ι S))   [the steps]
    List                                    [a finite sequence of things, in order]
    (acc : MatrixAffineStep ι S)            [the step the fold starts from]
    (x : ι → S)                             [the vector]

    So: the accumulator form of `matrixStepFold_apply`, by `matrixStep_apply`. -/
private theorem matrixStepApply_fold (steps : List (MatrixAffineStep ι S))
    (acc : MatrixAffineStep ι S) (x : ι → S) :
    matrixAffine (steps.foldl (fun acc step => matrixStepComp step acc) acc) x =
      steps.foldl (fun z step => matrixAffine step z) (matrixAffine acc x) := by
  induction steps generalizing acc x with
  | nil => rfl
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (matrixStepComp s acc) x, matrixStep_apply]

/-- **Applying the fold is the left fold of the applies.**

    (steps : List (MatrixAffineStep ι S))   [the steps in the order they are applied]
    List                                    [a finite sequence of things, in order]
    (x : ι → S)                             [the vector]

    So: the composed step applied once equals applying the steps one after another, so a prefix
    scan and a sequential loop agree. -/
theorem matrixStepFold_apply (steps : List (MatrixAffineStep ι S)) (x : ι → S) :
    matrixAffine (matrixStepFold steps) x =
      steps.foldl (fun z step => matrixAffine step z) x := by
  rw [matrixStepFold, matrixStepApply_fold, matrixStepId_apply]

/-- **Appending two step lists composes the two folds.**

    (steps more : List (MatrixAffineStep ι S))  [the first list, then the list appended after]
    List                                        [a finite sequence of things, in order]
    ++                                          [the two lists joined, the left one's entries first]

    So: the fold of the appended list is the fold of the second list composed on the outside of
    the fold of the first. -/
theorem matrixStepFold_append (steps more : List (MatrixAffineStep ι S)) :
    matrixStepFold (steps ++ more) =
      matrixStepComp (matrixStepFold more) (matrixStepFold steps) := by
  rw [matrixStepFold, List.foldl_append, matrixStepFold_accum]
  rfl

end MatrixAffine

end AnyRing
end Trichotomy
end Apeiron
