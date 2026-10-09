import Apeiron.Trichotomy.IdempotentsAnyRing

/-!
# Channels and gates over any commutative ring

@name Channels Over Any Commutative Ring
@cite weierstrass-1884
@cite peirce-1870
@cite boole-1847
@cite stone-1936

`Channels` proves that a read of the real algebra is forced by the sign of `σ`: none below zero,
the first slot at zero, two above. The forcing has two halves. The first is algebraic and holds
over every commutative ring: a read is fixed by its value on the generator, and that value
squares to `σ`. The second half, the classification of the square roots of `σ`, is where the
order of the reals enters, and it stays in `Channels`.

A read here is a map `RingCell R → R` respecting addition, stretching, multiplication and the
unit. Such a map is a ring homomorphism in all but name, and the name it is missing is the ring
structure on `R × R` under `ringMul`, which the library does not install.

Given a root `r` of `σ`, the two coordinate reads are `z ↦ z.1 + r·z.2` and `z ↦ z.1 − r·z.2`.
They are channels by `r·r = σ` alone, with no order and no half. The null basis they define is
multiplicative (`ringUpper_mul`, `ringLower_mul`) and the norm form is their product
(`ringNormForm_eq_upper_mul_lower`). An affine step reads on each channel as the ordinary scalar
affine step (`ringUpper_ringAffineApply`, `ringLower_ringAffineApply`), and a folded list of
steps reads as the scalar left fold of the individual reads (`ringUpper_ringAffineFold`,
`ringLower_ringAffineFold`).

Folding a list of points multiplies them, and each channel reads the fold as the product of the
individual reads (`ringUpper_fold`, `ringLower_fold`). Carrying a truth value into a corner and
folding the list computes AND on the first channel and NOR on the second
(`ringFold_is_and_and_nor`), with the two extra idempotents built from a root, its inverse and a
half.

## Root

| from | here |
|---|---|
| `AnyRing.RingCell`, `AnyRing.ringMul`, `AnyRing.ringOne` | `RingChannel`, `ringChannel_apply` |
| `AnyRing.ringMul` | `ringUpper_mul`, `ringLower_mul`, `ringUpper_fold` |
| `ringUpper_mul`, `ringUpper_add`, `AnyRing.ringAffineApply` | `ringUpper_ringAffineApply` |
| `ringLower_mul`, `ringLower_add`, `AnyRing.ringAffineApply` | `ringLower_ringAffineApply` |
| `ringUpper_ringAffineApply`, `AnyRing.ringAffineFold_apply` | `ringUpper_ringAffineFold` |
| `ringLower_ringAffineApply`, `AnyRing.ringAffineFold_apply` | `ringLower_ringAffineFold` |
| `AnyRing.ringNormForm` | `ringNormForm_eq_upper_mul_lower` |
| `IdempotentsAnyRing.ringIdemPlus`, `ringIdemMinus` | `ringFold_is_and_and_nor` |

A read of the ring is defined by the four clauses of `RingChannel`, and `ringChannel_gen_sq`
forces its value at the generator to square to `σ`. Given a root `r` with `r * r = σ`, the
coordinate reads `ringUpper r` and `ringLower r` are channels, their product is the norm form,
and `ringFold` multiplies the reads down a list. The construction never uses the order. What
stays in `Channels.lean` is the classification of the square roots of `σ` by its sign, which is
where `Real.sqrt` and the two-channel case above zero live. Here the root is a hypothesis, so the
number of channels is whatever the ring's roots supply.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy
namespace AnyRing

variable {R : Type*} [CommRing R]

/-- A read of the ring: one element off each point, respecting the operations.

    (σ : R)                                  [the parameter]
    (φ : RingCell R → R)                     [a map from points to ring elements]
    φ (z + w) = φ z + φ w                    [adding points adds the outputs]
    φ (c • z) = c * φ z                      [stretching a point stretches the output]
    φ (ringMul σ z w) = φ z * φ w            [multiplying points multiplies the outputs]
    φ ringOne = 1                            [the unit goes to one]

    So: the four clauses say the read tracks how points combine. What it forgets is only where in
    the ring a point sits. Over the reals this is `Channel`. -/
def RingChannel (σ : R) (φ : RingCell R → R) : Prop :=
  (∀ z w : RingCell R, φ (z + w) = φ z + φ w) ∧
  (∀ (c : R) (z : RingCell R), φ (c • z) = c * φ z) ∧
  (∀ z w : RingCell R, φ (ringMul σ z w) = φ z * φ w) ∧
  φ ringOne = 1

/-- A channel is fixed by its value on the generator.

    {σ : R}                                    [the parameter]
    {φ : RingCell R → R}                       [a map from points to ring elements]
    (h : RingChannel σ φ)                      [which is a read of the ring]
    (z : RingCell R)                           [any point]
    φ z = z.1 + z.2 * φ ((0, 1) : RingCell R)  [its read is fixed by its value at (0,1)]

    So: the point is its first slot times the unit plus its second slot times the generator, and
    the read respects both operations. -/
theorem ringChannel_apply {σ : R} {φ : RingCell R → R} (h : RingChannel σ φ)
    (z : RingCell R) : φ z = z.1 + z.2 * φ ((0, 1) : RingCell R) := by
  obtain ⟨hadd, hsmul, -, hone⟩ := h
  have hz : z = z.1 • ringOne + z.2 • ((0, 1) : RingCell R) := by
    simp only [ringOne, Prod.smul_mk, smul_eq_mul, Prod.mk_add_mk, mul_one, mul_zero,
      add_zero, zero_add]
  calc φ z = φ (z.1 • ringOne + z.2 • ((0, 1) : RingCell R)) := by rw [← hz]
    _ = φ (z.1 • ringOne) + φ (z.2 • ((0, 1) : RingCell R)) := hadd _ _
    _ = z.1 * φ ringOne + z.2 * φ ((0, 1) : RingCell R) := by rw [hsmul, hsmul]
    _ = z.1 + z.2 * φ ((0, 1) : RingCell R) := by rw [hone, mul_one]

/-- What a channel gives the generator, squared, is the parameter.

    {σ : R}                                              [the parameter]
    {φ : RingCell R → R}                                 [a map from points to ring elements]
    (h : RingChannel σ φ)                                [which is a read of the ring]
    φ ((0, 1) : RingCell R) * φ ((0, 1) : RingCell R) = σ
                                                         [its value at (0,1), squared, is σ]

    So: the generator squares to `σ` times the unit, and the read respects multiplying.
    Everything the real classification says follows from this one equation. -/
theorem ringChannel_gen_sq {σ : R} {φ : RingCell R → R} (h : RingChannel σ φ) :
    φ ((0, 1) : RingCell R) * φ ((0, 1) : RingCell R) = σ := by
  obtain ⟨-, hsmul, hmul, hone⟩ := h
  have hgen : ringMul σ ((0, 1) : RingCell R) ((0, 1) : RingCell R) = σ • ringOne := by
    simp only [ringMul, ringOne, Prod.ext_iff, Prod.smul_mk, smul_eq_mul]
    constructor <;> ring
  have hx := hmul ((0, 1) : RingCell R) ((0, 1) : RingCell R)
  rw [hgen, hsmul, hone, mul_one] at hx
  exact hx.symm

/-- How far a point lies along the first direction of size zero.

    (r : R)                        [a root: an element whose square is the parameter]
    (z : RingCell R)               [the point]
    z.1 + r * z.2                  [first slot, plus the root times the second]
    : R                            [a ring element]

    So: the coordinate along the direction `(r, -1)`, for any root `r` of the parameter. No
    order and no half is used. -/
def ringUpper (r : R) (z : RingCell R) : R := z.1 + r * z.2

/-- How far a point lies along the other direction of size zero.

    (r : R)                        [a root: an element whose square is the parameter]
    (z : RingCell R)               [the point]
    z.1 - r * z.2                  [first slot, minus the root times the second]
    : R                            [a ring element]

    So: the coordinate along the direction `(r, 1)`, the mirror of `ringUpper`. -/
def ringLower (r : R) (z : RingCell R) : R := z.1 - r * z.2

/-- The first coordinate of a product is the product of the first coordinates.

    {σ r : R}                              [the parameter and one of its roots]
    r ^ 2 = σ                              [the root squares to the parameter]
    (z w : RingCell R)                     [two points]
    ringUpper r (ringMul σ z w)            [multiply, then take the coordinate]
    = ringUpper r z * ringUpper r w        [take each coordinate, then multiply]

    So: the two channels are independent. The one step is `r² = σ`, with no order anywhere. -/
theorem ringUpper_mul {σ r : R} (hr : r ^ 2 = σ) (z w : RingCell R) :
    ringUpper r (ringMul σ z w) = ringUpper r z * ringUpper r w := by
  simp only [ringUpper]
  linear_combination (-(z.2 * w.2)) * hr

/-- The second coordinate of a product is the product of the second coordinates.

    {σ r : R}                              [the parameter and one of its roots]
    r ^ 2 = σ                              [the root squares to the parameter]
    (z w : RingCell R)                     [two points]
    ringLower r (ringMul σ z w)            [multiply, then take the coordinate]
    = ringLower r z * ringLower r w        [take each coordinate, then multiply] -/
theorem ringLower_mul {σ r : R} (hr : r ^ 2 = σ) (z w : RingCell R) :
    ringLower r (ringMul σ z w) = ringLower r z * ringLower r w := by
  simp only [ringLower]
  linear_combination (-(z.2 * w.2)) * hr

/-- A point's size is the product of its two coordinates.

    {σ r : R}                                [the parameter and one of its roots]
    r ^ 2 = σ                                [the root squares to the parameter]
    (z : RingCell R)                         [the point]
    ringNormForm σ z                         [its size]
    = ringUpper r z * ringLower r z          [the two coordinates multiplied]

    So: size zero means one coordinate is zero, the cone being the two coordinate axes. This is
    where the real `normForm_eq_upper_mul_lower` gets its statement. -/
theorem ringNormForm_eq_upper_mul_lower {σ r : R} (hr : r ^ 2 = σ) (z : RingCell R) :
    ringNormForm σ z = ringUpper r z * ringLower r z := by
  simp only [ringNormForm, ringUpper, ringLower]
  linear_combination (z.2 ^ 2) * hr

/-- The unit has both coordinates one.

    (r : R)                      [a root of the parameter]
    ringUpper r ringOne = 1      [the first coordinate]
    ∧                            [and]
    ringLower r ringOne = 1      [the second coordinate] -/
theorem ringUpper_ringOne (r : R) : ringUpper r ringOne = 1 := by
  simp [ringUpper]

/-- The unit has both coordinates one, read the other way.

    (r : R)                      [a root of the parameter]
    ringLower r ringOne = 1      [the second coordinate] -/
theorem ringLower_ringOne (r : R) : ringLower r ringOne = 1 := by
  simp [ringLower]

/-- The first coordinate respects addition.

    (r : R) (z w : RingCell R)                       [a root and two points]
    ringUpper r (z + w) = ringUpper r z + ringUpper r w  [add, then read; or read, then add] -/
theorem ringUpper_add (r : R) (z w : RingCell R) :
    ringUpper r (z + w) = ringUpper r z + ringUpper r w := by
  simp only [ringUpper, Prod.fst_add, Prod.snd_add]
  ring

/-- The second coordinate respects addition.

    (r : R) (z w : RingCell R)                       [a root and two points]
    ringLower r (z + w) = ringLower r z + ringLower r w  [add, then read; or read, then add] -/
theorem ringLower_add (r : R) (z w : RingCell R) :
    ringLower r (z + w) = ringLower r z + ringLower r w := by
  simp only [ringLower, Prod.fst_add, Prod.snd_add]
  ring

/-- The first coordinate respects stretching.

    (r c : R) (z : RingCell R)                       [a root, a ring element, a point]
    ringUpper r (c • z) = c * ringUpper r z          [stretch, then read; or read, then stretch] -/
theorem ringUpper_smul (r c : R) (z : RingCell R) :
    ringUpper r (c • z) = c * ringUpper r z := by
  simp only [ringUpper, Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
  ring

/-- The second coordinate respects stretching.

    (r c : R) (z : RingCell R)                       [a root, a ring element, a point]
    ringLower r (c • z) = c * ringLower r z          [stretch, then read; or read, then stretch] -/
theorem ringLower_smul (r c : R) (z : RingCell R) :
    ringLower r (c • z) = c * ringLower r z := by
  simp only [ringLower, Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
  ring

/-- A root of the parameter makes the first coordinate a channel.

    {σ r : R}                                  [the parameter and one of its roots]
    r ^ 2 = σ                                  [the root squares to the parameter]
    RingChannel σ (ringUpper r)                [the first coordinate is a read of the ring]

    So: addition, stretching, multiplication and the unit are the four clauses, each proved by
    its own lemma. -/
theorem ringUpper_channel {σ r : R} (hr : r ^ 2 = σ) : RingChannel σ (ringUpper r) :=
  ⟨fun z w => ringUpper_add r z w,
   fun c z => ringUpper_smul r c z,
   fun z w => ringUpper_mul hr z w,
   ringUpper_ringOne r⟩

/-- A root of the parameter makes the second coordinate a channel.

    {σ r : R}                                  [the parameter and one of its roots]
    r ^ 2 = σ                                  [the root squares to the parameter]
    RingChannel σ (ringLower r)                [the second coordinate is a read of the ring] -/
theorem ringLower_channel {σ r : R} (hr : r ^ 2 = σ) : RingChannel σ (ringLower r) :=
  ⟨fun z w => ringLower_add r z w,
   fun c z => ringLower_smul r c z,
   fun z w => ringLower_mul hr z w,
   ringLower_ringOne r⟩

/-- An affine step read on the first channel is the scalar affine step on that channel.

    {sigma r : R}                               [the parameter and one of its roots]
    r ^ 2 = sigma                               [the root squares to the parameter]
    (step : RingAffineStep R)                   [the affine step being applied]
    (z : RingCell R)                            [the point the step is applied to]
    ringUpper r (ringAffineApply sigma step z)  [apply the step, then read the first channel]
    = ringUpper r step.coefficient * ringUpper r z + ringUpper r step.injection
                                                [or read each part, then combine]
    `ringUpper`                                 [the first-coordinate read over a ring]
    `ringAffineApply`                           [apply one affine step: multiply by the coefficient]
    `RingAffineStep`                            [one affine step: a coefficient and an injection]
    `RingCell`                                  [a point: two ring elements side by side, a + b·x]

    So: `ringUpper_mul` and `ringUpper_add` make the read multiplicative and additive, so the
    read of `coefficient · z + injection` is the scalar step on the channel. -/
theorem ringUpper_ringAffineApply {sigma r : R} (hr : r ^ 2 = sigma)
    (step : RingAffineStep R) (z : RingCell R) :
    ringUpper r (ringAffineApply sigma step z) =
      ringUpper r step.coefficient * ringUpper r z + ringUpper r step.injection := by
  simp only [ringAffineApply, ringUpper_add, ringUpper_mul hr]

/-- An affine step read on the second channel is the scalar affine step on that channel.

    {sigma r : R}                               [the parameter and one of its roots]
    r ^ 2 = sigma                               [the root squares to the parameter]
    (step : RingAffineStep R)                   [the affine step being applied]
    (z : RingCell R)                            [the point the step is applied to]
    ringLower r (ringAffineApply sigma step z)  [apply the step, then read the second channel]
    = ringLower r step.coefficient * ringLower r z + ringLower r step.injection
                                                [or read each part, then combine]
    `ringLower`                                 [the second-coordinate read over a ring]
    `ringAffineApply`                           [apply one affine step: multiply by the coefficient]
    `RingAffineStep`                            [one affine step: a coefficient and an injection]
    `RingCell`                                  [a point: two ring elements side by side, a + b·x]

    So: `ringLower_mul` and `ringLower_add` make the read multiplicative and additive, so the
    read of `coefficient · z + injection` is the scalar step on the channel. -/
theorem ringLower_ringAffineApply {sigma r : R} (hr : r ^ 2 = sigma)
    (step : RingAffineStep R) (z : RingCell R) :
    ringLower r (ringAffineApply sigma step z) =
      ringLower r step.coefficient * ringLower r z + ringLower r step.injection := by
  simp only [ringAffineApply, ringLower_add, ringLower_mul hr]

/-- The first coordinate of a folded step list is the scalar fold of the coordinates.

    {sigma r : R}                              [the parameter and one of its roots]
    r ^ 2 = sigma                              [the root squares to the parameter]
    (steps : List (RingAffineStep R))          [the steps in the order they are applied]
    (z : RingCell R)                           [the seed the composed step is applied to]
    ringUpper r (ringAffineApply sigma (ringAffineFold sigma steps) z)
                                               [apply the composed step, then read channel one]
    = steps.foldl (fun acc step => ringUpper r step.coefficient * acc +
        ringUpper r step.injection) (ringUpper r z)
                                               [fold the scalar steps over the seed's reading]
    `ringUpper`                                [the first-coordinate read over a ring]
    `List`                                     [a finite sequence of things, in order]

    So: `ringAffineFold_apply` turns the composed step into the sequential applies, and
    `ringUpper_ringAffineApply` reads each apply as the scalar affine step. -/
theorem ringUpper_ringAffineFold {sigma r : R} (hr : r ^ 2 = sigma)
    (steps : List (RingAffineStep R)) (z : RingCell R) :
    ringUpper r (ringAffineApply sigma (ringAffineFold sigma steps) z) =
      steps.foldl (fun acc step => ringUpper r step.coefficient * acc +
        ringUpper r step.injection) (ringUpper r z) := by
  rw [ringAffineFold_apply]
  induction steps generalizing z with
  | nil => rfl
  | cons step rest ih =>
    simp only [List.foldl_cons]
    rw [ih (ringAffineApply sigma step z), ringUpper_ringAffineApply hr]

/-- The second coordinate of a folded step list is the scalar fold of the coordinates.

    {sigma r : R}                              [the parameter and one of its roots]
    r ^ 2 = sigma                              [the root squares to the parameter]
    (steps : List (RingAffineStep R))          [the steps in the order they are applied]
    (z : RingCell R)                           [the seed the composed step is applied to]
    ringLower r (ringAffineApply sigma (ringAffineFold sigma steps) z)
                                               [apply the composed step, then read channel two]
    = steps.foldl (fun acc step => ringLower r step.coefficient * acc +
        ringLower r step.injection) (ringLower r z)
                                               [fold the scalar steps over the seed's reading]
    `ringLower`                                [the second-coordinate read over a ring]
    `List`                                     [a finite sequence of things, in order]

    So: `ringAffineFold_apply` turns the composed step into the sequential applies, and
    `ringLower_ringAffineApply` reads each apply as the scalar affine step. -/
theorem ringLower_ringAffineFold {sigma r : R} (hr : r ^ 2 = sigma)
    (steps : List (RingAffineStep R)) (z : RingCell R) :
    ringLower r (ringAffineApply sigma (ringAffineFold sigma steps) z) =
      steps.foldl (fun acc step => ringLower r step.coefficient * acc +
        ringLower r step.injection) (ringLower r z) := by
  rw [ringAffineFold_apply]
  induction steps generalizing z with
  | nil => rfl
  | cons step rest ih =>
    simp only [List.foldl_cons]
    rw [ih (ringAffineApply sigma step z), ringLower_ringAffineApply hr]

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO CONSTRUCTED IDEMPOTENTS IN CHANNEL COORDINATES
-- ═══════════════════════════════════════════════════════════════════

/-- The first coordinate of `ringIdemPlus` is one.

    {σ r s h : R}                              [the parameter, root, inverse, half]
    r * s = 1                                  [the root times its inverse is one]
    2 * h = 1                                  [twice the half is one]
    →                                          [then]
    ringUpper r (ringIdemPlus σ r s h) = 1     [all of it is in the first channel] -/
theorem ringUpper_ringIdemPlus {σ r s h : R} (hrs : r * s = 1) (hh : 2 * h = 1) :
    ringUpper r (ringIdemPlus σ r s h) = 1 := by
  simp only [ringUpper, ringIdemPlus]
  rw [show r * (h * s) = h * (r * s) by ring, hrs, mul_one]
  linear_combination hh

/-- The second coordinate of `ringIdemPlus` is zero.

    {σ r s h : R}                              [the parameter, root, inverse, half]
    r * s = 1                                  [the root times its inverse is one]
    →                                          [then]
    ringLower r (ringIdemPlus σ r s h) = 0     [none of it is in the second channel] -/
theorem ringLower_ringIdemPlus {σ r s h : R} (hrs : r * s = 1) :
    ringLower r (ringIdemPlus σ r s h) = 0 := by
  simp only [ringLower, ringIdemPlus]
  rw [show r * (h * s) = h * (r * s) by ring, hrs, mul_one]
  ring

/-- The first coordinate of `ringIdemMinus` is zero.

    {σ r s h : R}                               [the parameter, root, inverse, half]
    r * s = 1                                   [the root times its inverse is one]
    →                                           [then]
    ringUpper r (ringIdemMinus σ r s h) = 0     [none of it is in the first channel] -/
theorem ringUpper_ringIdemMinus {σ r s h : R} (hrs : r * s = 1) :
    ringUpper r (ringIdemMinus σ r s h) = 0 := by
  simp only [ringUpper, ringIdemMinus]
  rw [show r * (-(h * s)) = -(h * (r * s)) by ring, hrs, mul_one]
  ring

/-- The second coordinate of `ringIdemMinus` is one.

    {σ r s h : R}                              [the parameter, root, inverse, half]
    r * s = 1                                  [the root times its inverse is one]
    2 * h = 1                                  [twice the half is one]
    →                                          [then]
    ringLower r (ringIdemMinus σ r s h) = 1    [all of it is in the second channel] -/
theorem ringLower_ringIdemMinus {σ r s h : R} (hrs : r * s = 1) (hh : 2 * h = 1) :
    ringLower r (ringIdemMinus σ r s h) = 1 := by
  simp only [ringLower, ringIdemMinus]
  rw [show r * (-(h * s)) = -(h * (r * s)) by ring, hrs, mul_one]
  linear_combination hh

-- ═══════════════════════════════════════════════════════════════════
-- THE POINT WITH GIVEN COORDINATES
-- ═══════════════════════════════════════════════════════════════════

/-- The point with the given pair of coordinates.

    (u v s h : R)                    [the two coordinates, an inverse root, and a half]
    ((u + v) * h, (u - v) * h * s)   [solve the two coordinate equations]
    : RingCell R                     [a point over the ring]

    So: `r·s = 1` and `2·h = 1` make this the point whose first coordinate is `u` and second is
    `v`. Over the reals `s` is `1/√σ` and `h` is `1/2`, which is why `NullBasis.ofHalves`
    divides. -/
def ringOfHalves (u v s h : R) : RingCell R := ((u + v) * h, (u - v) * h * s)

/-- Building from coordinates and reading the first one back returns it.

    {r u v s h : R}                        [the root, the two coordinates, inverse and half]
    r * s = 1                              [the root times its inverse is one]
    2 * h = 1                              [twice the half is one]
    →                                      [then]
    ringUpper r (ringOfHalves u v s h) = u [reading back gives what went in] -/
theorem ringUpper_ringOfHalves {r u v s h : R} (hrs : r * s = 1) (hh : 2 * h = 1) :
    ringUpper r (ringOfHalves u v s h) = u := by
  simp only [ringUpper, ringOfHalves]
  linear_combination ((u - v) * h) * hrs + u * hh

/-- Building from coordinates and reading the second one back returns it.

    {r u v s h : R}                        [the root, the two coordinates, inverse and half]
    r * s = 1                              [the root times its inverse is one]
    2 * h = 1                              [twice the half is one]
    →                                      [then]
    ringLower r (ringOfHalves u v s h) = v [reading back gives what went in] -/
theorem ringLower_ringOfHalves {r u v s h : R} (hrs : r * s = 1) (hh : 2 * h = 1) :
    ringLower r (ringOfHalves u v s h) = v := by
  simp only [ringLower, ringOfHalves]
  linear_combination (-((u - v) * h)) * hrs + v * hh

/-- Reading a point's coordinates and rebuilding from them returns the point.

    {r s h : R}                                        [the root, its inverse, the half]
    r * s = 1                                          [the root times its inverse is one]
    2 * h = 1                                          [twice the half is one]
    (z : RingCell R)                                   [any point]
    ringOfHalves (ringUpper r z) (ringLower r z) s h = z [the round trip is the identity]

    So: where the root and the half exist, the two coordinates carry everything the two slots
    did, and the ring is two copies of itself side by side. -/
theorem ringOfHalves_upper_lower {r s h : R} (hrs : r * s = 1) (hh : 2 * h = 1)
    (z : RingCell R) :
    ringOfHalves (ringUpper r z) (ringLower r z) s h = z := by
  apply Prod.ext
  · show (ringUpper r z + ringLower r z) * h = z.1
    simp only [ringUpper, ringLower]
    rw [show (z.1 + r * z.2 + (z.1 - r * z.2)) * h = z.1 * (2 * h) by ring, hh]
    ring
  · show (ringUpper r z - ringLower r z) * h * s = z.2
    simp only [ringUpper, ringLower]
    rw [show (z.1 + r * z.2 - (z.1 - r * z.2)) * h * s = z.2 * (2 * h) * (r * s) by ring,
      hh, hrs]
    ring

-- ═══════════════════════════════════════════════════════════════════
-- FOLDING A LIST OF POINTS
-- ═══════════════════════════════════════════════════════════════════

/-- Multiply every point in a list together, starting from the unit.

    (σ : R)                          [the parameter]
    (zs : List (RingCell R))         [a list of points]
    zs.foldr (ringMul σ) ringOne     [multiply them all, right to left, starting at the unit]
    : RingCell R                     [a point of the ring]

    So: over the reals this is `Gates.fold`, and the multiplication is the algebra's own. -/
def ringFold (σ : R) (zs : List (RingCell R)) : RingCell R := zs.foldr (ringMul σ) ringOne

/-- The first coordinate of a product is the product of the first coordinates.

    {σ r : R}                                  [the parameter and one of its roots]
    r ^ 2 = σ                                  [the root squares to the parameter]
    (zs : List (RingCell R))                   [a list of points]
    ringUpper r (ringFold σ zs)                [multiply everything, then read channel one]
    = (zs.map (ringUpper r)).prod              [read channel one of each, then multiply]

    So: `ringUpper_mul` carried through the whole list by induction. -/
theorem ringUpper_fold {σ r : R} (hr : r ^ 2 = σ) (zs : List (RingCell R)) :
    ringUpper r (ringFold σ zs) = (zs.map (ringUpper r)).prod := by
  induction zs with
  | nil => simp [ringFold, ringUpper_ringOne]
  | cons z zs ih =>
    simp only [ringFold, List.foldr_cons] at ih ⊢
    rw [ringUpper_mul hr, ih, List.map_cons, List.prod_cons]

/-- The second coordinate of a product is the product of the second coordinates.

    {σ r : R}                                  [the parameter and one of its roots]
    r ^ 2 = σ                                  [the root squares to the parameter]
    (zs : List (RingCell R))                   [a list of points]
    ringLower r (ringFold σ zs)                [multiply everything, then read channel two]
    = (zs.map (ringLower r)).prod              [read channel two of each, then multiply] -/
theorem ringLower_fold {σ r : R} (hr : r ^ 2 = σ) (zs : List (RingCell R)) :
    ringLower r (ringFold σ zs) = (zs.map (ringLower r)).prod := by
  induction zs with
  | nil => simp [ringFold, ringLower_ringOne]
  | cons z zs ih =>
    simp only [ringFold, List.foldr_cons] at ih ⊢
    rw [ringLower_mul hr, ih, List.map_cons, List.prod_cons]

/-- The size of a whole product is the two channel products, closed by one multiplication.

    {σ r : R}                                  [the parameter and one of its roots]
    r ^ 2 = σ                                  [the root squares to the parameter]
    (zs : List (RingCell R))                   [a list of points]
    ringNormForm σ (ringFold σ zs)             [the size of the product]
    = (zs.map (ringUpper r)).prod * (zs.map (ringLower r)).prod
                                               [the two channels, multiplied]
    `ringUpper`                 [the first-coordinate read over a ring]
    `ringLower`                 [the second-coordinate read over a ring]

    So: no point in the list is ever squared. Two running products and one final
    multiplication. -/
theorem ringNormForm_fold {σ r : R} (hr : r ^ 2 = σ) (zs : List (RingCell R)) :
    ringNormForm σ (ringFold σ zs) =
      (zs.map (ringUpper r)).prod * (zs.map (ringLower r)).prod := by
  rw [ringNormForm_eq_upper_mul_lower hr, ringUpper_fold hr, ringLower_fold hr]

/-- Reordering the list cannot change the product.

    {σ r : R}                                  [the parameter and one of its roots]
    r ^ 2 = σ                                  [the root squares to the parameter]
    {zs ws : List (RingCell R)}                [two lists]
    (h : zs.Perm ws)                           [one is a reordering of the other]
    ringUpper r (ringFold σ zs) = ringUpper r (ringFold σ ws)  [same first channel]
    ∧                                          [and]
    ringLower r (ringFold σ zs) = ringLower r (ringFold σ ws)  [same second channel]

    So: each channel is an ordinary product of ring elements, and multiplication does not care
    about order. Nothing here had to be assumed about scheduling. -/
theorem ringFold_perm {σ r : R} (hr : r ^ 2 = σ) {zs ws : List (RingCell R)}
    (h : zs.Perm ws) :
    ringUpper r (ringFold σ zs) = ringUpper r (ringFold σ ws) ∧
      ringLower r (ringFold σ zs) = ringLower r (ringFold σ ws) := by
  rw [ringUpper_fold hr, ringUpper_fold hr, ringLower_fold hr, ringLower_fold hr]
  exact ⟨(h.map (ringUpper r)).prod_eq, (h.map (ringLower r)).prod_eq⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO GATES
-- ═══════════════════════════════════════════════════════════════════

/-- A truth value as a ring element.

    (b : Bool)              [a truth value]
    if b then 1 else 0      [one for true, zero for false]
    : R                     [a ring element]

    So: over the reals this is `Gates.boolVal`. -/
def ringBoolVal (R : Type*) [CommRing R] (b : Bool) : R := if b then 1 else 0

/-- Multiplying two truth values is AND.

    (b c : Bool)                                     [two truth values]
    ringBoolVal b * ringBoolVal c                    [their values, multiplied]
    = ringBoolVal (b && c)                           [the value of their conjunction]

    So: `1·1 = 1` and every other product has a zero in it, over any ring. -/
theorem ringBoolVal_mul (R : Type*) [CommRing R] (b c : Bool) :
    ringBoolVal R b * ringBoolVal R c = ringBoolVal R (b && c) := by
  cases b <;> cases c <;> simp [ringBoolVal]

/-- A truth value carried by the ring: true is one corner, false is the other.

    (σ r s h : R)                              [the parameter, the root, its inverse, the half]
    (b : Bool)                                 [a truth value]
    if b then ringIdemPlus σ r s h else ringIdemMinus σ r s h
                                               [true goes to one corner, false the other]
    : RingCell R                               [a point of the ring] -/
def ringCellOfBool (σ r s h : R) (b : Bool) : RingCell R :=
  if b then ringIdemPlus σ r s h else ringIdemMinus σ r s h

/-- The first coordinate of a carried truth value is its value.

    {σ r s h : R}                                    [the parameter, root, inverse, half]
    r * s = 1                                        [the root times its inverse is one]
    2 * h = 1                                        [twice the half is one]
    (b : Bool)                                       [a truth value]
    ringUpper r (ringCellOfBool σ r s h b) = ringBoolVal R b
                                                     [the first channel carries it directly]
    `ringUpper`                 [the first-coordinate read over a ring]
    `ringBoolVal`               [a truth value as a ring element: one for true, zero for false]
    `ringCellOfBool`            [a truth value carried by the ring: true is one corner] -/
theorem ringUpper_ringCellOfBool {σ r s h : R} (hrs : r * s = 1) (hh : 2 * h = 1)
    (b : Bool) : ringUpper r (ringCellOfBool σ r s h b) = ringBoolVal R b := by
  cases b
  · simp [ringCellOfBool, ringBoolVal, ringUpper_ringIdemMinus hrs]
  · simp [ringCellOfBool, ringBoolVal, ringUpper_ringIdemPlus hrs hh]

/-- The second coordinate of a carried truth value is the value of its negation.

    {σ r s h : R}                                    [the parameter, root, inverse, half]
    r * s = 1                                        [the root times its inverse is one]
    2 * h = 1                                        [twice the half is one]
    (b : Bool)                                       [a truth value]
    ringLower r (ringCellOfBool σ r s h b) = ringBoolVal R (!b)
                                                     [the second channel carries the negation]
    `ringLower`                 [the second-coordinate read over a ring]
    `ringBoolVal`               [a truth value as a ring element: one for true, zero for false]
    `ringCellOfBool`            [a truth value carried by the ring: true is one corner] -/
theorem ringLower_ringCellOfBool {σ r s h : R} (hrs : r * s = 1) (hh : 2 * h = 1)
    (b : Bool) : ringLower r (ringCellOfBool σ r s h b) = ringBoolVal R (!b) := by
  cases b
  · simp [ringCellOfBool, ringBoolVal, ringLower_ringIdemMinus hrs hh]
  · simp [ringCellOfBool, ringBoolVal, ringLower_ringIdemPlus hrs]

/-- The first channel of a fold over truth values is AND.

    {σ r s h : R}                                     [the parameter, root, inverse, half]
    r ^ 2 = σ                                         [the root squares to the parameter]
    r * s = 1                                         [the root times its inverse is one]
    2 * h = 1                                         [twice the half is one]
    (bs : List Bool)                                  [a list of truth values]
    ringUpper r (ringFold σ (bs.map (ringCellOfBool σ r s h)))
                                                      [carry them, multiply, read channel one]
    = ringBoolVal R (bs.all id)                         [one exactly when all are true]
    `ringUpper`                 [the first-coordinate read over a ring]
    `ringFold`                  [multiply every point in a list together over a ring]
    `ringCellOfBool`            [a truth value carried by the ring: true is one corner] -/
theorem ringUpper_fold_bool {σ r s h : R} (hr : r ^ 2 = σ) (hrs : r * s = 1)
    (hh : 2 * h = 1) (bs : List Bool) :
    ringUpper r (ringFold σ (bs.map (ringCellOfBool σ r s h))) = ringBoolVal R (bs.all id) := by
  rw [ringUpper_fold hr]
  induction bs with
  | nil => simp [ringBoolVal]
  | cons b bs ih =>
    rw [List.map_cons, List.map_cons, List.prod_cons, ringUpper_ringCellOfBool hrs hh, ih,
      List.all_cons, ringBoolVal_mul R]
    simp

/-- The second channel of a fold over truth values is NOR.

    {σ r s h : R}                                     [the parameter, root, inverse, half]
    r ^ 2 = σ                                         [the root squares to the parameter]
    r * s = 1                                         [the root times its inverse is one]
    2 * h = 1                                         [twice the half is one]
    (bs : List Bool)                                  [a list of truth values]
    ringLower r (ringFold σ (bs.map (ringCellOfBool σ r s h)))
                                                      [carry them, multiply, read channel two]
    = ringBoolVal R (!(bs.any id))                      [one exactly when none is true]
    `ringLower`                 [the second-coordinate read over a ring]
    `ringFold`                  [multiply every point in a list together over a ring]
    `ringCellOfBool`            [a truth value carried by the ring: true is one corner] -/
theorem ringLower_fold_bool {σ r s h : R} (hr : r ^ 2 = σ) (hrs : r * s = 1)
    (hh : 2 * h = 1) (bs : List Bool) :
    ringLower r (ringFold σ (bs.map (ringCellOfBool σ r s h))) =
      ringBoolVal R (!(bs.any id)) := by
  rw [ringLower_fold hr]
  induction bs with
  | nil => simp [ringBoolVal]
  | cons b bs ih =>
    rw [List.map_cons, List.map_cons, List.prod_cons, ringLower_ringCellOfBool hrs hh, ih,
      List.any_cons, ringBoolVal_mul R]
    cases b <;> simp [ringBoolVal]

/-- One pass over the list computes both gates at once.

    {σ r s h : R}                                     [the parameter, root, inverse, half]
    r ^ 2 = σ                                         [the root squares to the parameter]
    r * s = 1                                         [the root times its inverse is one]
    2 * h = 1                                         [twice the half is one]
    (bs : List Bool)                                  [a list of truth values]
    ringUpper r (ringFold σ (bs.map (ringCellOfBool σ r s h))) = ringBoolVal R (bs.all id)
                                                      [channel one is AND]
    ∧                                                 [and, at the same time]
    ringLower r (ringFold σ (bs.map (ringCellOfBool σ r s h))) = ringBoolVal R (!(bs.any id))
                                                      [channel two is NOR]
    `ringUpper`                 [the first-coordinate read over a ring]
    `ringLower`                 [the second-coordinate read over a ring]
    `ringFold`                  [multiply every point in a list together over a ring]
    `ringBoolVal`               [a truth value as a ring element: one for true, zero for false]
    `ringCellOfBool`            [a truth value carried by the ring: true is one corner]

    So: one multiplication per input, two answers. The two channels are independent, so neither
    gate costs anything the other did not already pay for. -/
theorem ringFold_is_and_and_nor {σ r s h : R} (hr : r ^ 2 = σ) (hrs : r * s = 1)
    (hh : 2 * h = 1) (bs : List Bool) :
    ringUpper r (ringFold σ (bs.map (ringCellOfBool σ r s h))) = ringBoolVal R (bs.all id) ∧
      ringLower r (ringFold σ (bs.map (ringCellOfBool σ r s h))) =
        ringBoolVal R (!(bs.any id)) :=
  ⟨ringUpper_fold_bool hr hrs hh bs, ringLower_fold_bool hr hrs hh bs⟩

end AnyRing
end Trichotomy
end Apeiron

end
