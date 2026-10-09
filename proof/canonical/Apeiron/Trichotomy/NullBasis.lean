import Apeiron.Trichotomy.Idempotents
import Apeiron.Trichotomy.ChannelsAnyRing
import Mathlib.NumberTheory.SumFourSquares

/-!
# The null basis

@name The Null Basis
@cite peirce-1870
@cite boole-1847
@cite mathlib-sqrt
@cite artin-schreier-1927
@cite kaucher-1973
@cite shary-2022
@grade proved

## What the channels carry besides numbers

Each channel is a copy of a ring, and that ring fixes an order without anything being added
from outside. Over the reals one square does it (`le_iff_exists_sq`, Artin and Schreier), over
the whole numbers it takes four and no fewer (`int_le_iff_exists_sum_four_sq`, with
`two_not_sq` and `seven_not_sum_three_sq` giving the count). The channels therefore carry a
lattice, `meet` intersects two of them, and `inCell_meet_iff` says the intersection holds
exactly what both held.

This is what `../song/src/prover/cone/cell.slang` names under C3 and assigns to the spec side:
its `ncone0` holds a variable's feasible set as a pair of whole numbers read as the two null
coordinates, and its propagation step is max on one channel and min on the other. Neither is a
ring operation, so the question is whether the order was smuggled in. It was not.

Multiplying two points is four real multiplications and a choice of sigma:

```
   (a + b·x)(c + d·x) = (ac + σbd) + (ad + bc)·x
```

The two slots are tangled. Changing coordinates untangles them. Measure a point not by its two
slots but by how far it lies along the two directions of size zero:

```
   u = a + √σ·b            v = a − √σ·b
```

In those coordinates multiplication is componentwise. The two coordinates are independent:

```
   u(zw) = u(z)·u(w)             v(zw) = v(z)·v(w)
```

which is `upper_mul` and `lower_mul`. The size of a point is the product of its two
coordinates (`normForm_eq_upper_mul_lower`), so a point has size zero exactly when one of its
coordinates does.

Above zero the change of coordinates is reversible (`ofHalves_upper_lower`), so the algebra
*is* two independent copies of `ℝ` side by side. At exactly zero it is not: `√0 = 0` makes
`u = v = a`, both coordinates collapse onto the first slot, and the two channels become one.
That collapse is what the degenerate setting means here.

## What it buys

Being unchanged by squaring becomes two separate scalar questions, `u² = u` and `v² = v`,
and over the reals each has only the answers `0` and `1`. So above zero there are exactly
four such points, no more (`idempotent_locus`), and their coordinates run over the four
corners of `{0,1} × {0,1}`.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

/-- How far a point lies along the first direction of size zero.

    (σ : ℝ)                        [the parameter]
    (z : Cell)                     [the point]
    z.1 + Real.sqrt σ * z.2        [first slot, plus the square root of sigma times the second]
    : ℝ                            [a plain real number] -/
def upper (σ : ℝ) (z : Cell) : ℝ := z.1 + Real.sqrt σ * z.2

/-- How far a point lies along the other direction of size zero.

    (σ : ℝ)                        [the parameter]
    (z : Cell)                     [the point]
    z.1 - Real.sqrt σ * z.2        [first slot, minus the square root of sigma times the second]
    : ℝ                            [a plain real number] -/
def lower (σ : ℝ) (z : Cell) : ℝ := z.1 - Real.sqrt σ * z.2

/-- The first coordinate of a product is the product of the first coordinates.

    (σ : ℝ) (hσ : 0 ≤ σ)                       [the parameter, at zero or above]
    (z w : Cell)                               [two points]
    upper σ (mul σ z w)                        [multiply, then take the coordinate]
    = upper σ z * upper σ w                    [take each coordinate, then multiply]

    So: the two channels are independent. The step needing `0 ≤ σ` is `(√σ)² = σ`. This is
    `AnyRing.ringUpper_mul` at the real root, with `Real.sq_sqrt` supplying the witness. -/
theorem upper_mul {σ : ℝ} (hσ : 0 ≤ σ) (z w : Cell) :
    upper σ (mul σ z w) = upper σ z * upper σ w :=
  AnyRing.ringUpper_mul (Real.sq_sqrt hσ) z w

/-- The second coordinate of a product is the product of the second coordinates.

    (σ : ℝ) (hσ : 0 ≤ σ)                       [the parameter, at zero or above]
    (z w : Cell)                               [two points]
    lower σ (mul σ z w)                        [multiply, then take the coordinate]
    = lower σ z * lower σ w                    [take each coordinate, then multiply]

    So: `AnyRing.ringLower_mul` at the real root. -/
theorem lower_mul {σ : ℝ} (hσ : 0 ≤ σ) (z w : Cell) :
    lower σ (mul σ z w) = lower σ z * lower σ w :=
  AnyRing.ringLower_mul (Real.sq_sqrt hσ) z w

/-- A point's size is the product of its two coordinates.

    (σ : ℝ) (hσ : 0 ≤ σ)                    [the parameter, at zero or above]
    (z : Cell)                              [the point]
    normForm σ z                                [its size]
    = upper σ z * lower σ z                 [the two coordinates multiplied]

    So: size zero means one coordinate is zero. The cone is the two coordinate axes. This is
    `AnyRing.ringNormForm_eq_upper_mul_lower` at the real root. -/
theorem normForm_eq_upper_mul_lower {σ : ℝ} (hσ : 0 ≤ σ) (z : Cell) :
    normForm σ z = upper σ z * lower σ z :=
  AnyRing.ringNormForm_eq_upper_mul_lower (Real.sq_sqrt hσ) z

-- ═══════════════════════════════════════════════════════════════════
-- ABOVE ZERO THE COORDINATES ARE REVERSIBLE
-- ═══════════════════════════════════════════════════════════════════

/-- The point with the given pair of coordinates.

    (σ u v : ℝ)                                    [the parameter and the two coordinates]
    ((u + v) / 2, (u - v) / (2 * Real.sqrt σ))     [solve the two coordinate equations]
    : Cell                                         [a point of the algebra] -/
def ofHalves (σ u v : ℝ) : Cell := ((u + v) / 2, (u - v) / (2 * Real.sqrt σ))

/-- The real point with given coordinates is the generic one at the real inverse root and half.

    (σ u v : ℝ)                [the parameter and the two coordinates]
    ofHalves σ u v             [the real point with these coordinates]
    Real.sqrt σ                [the number that squares to give sigma]
    ofHalves σ u v = AnyRing.ringOfHalves u v (1 / Real.sqrt σ) (1 / 2)
                               [the real point is the generic point at these witnesses]
    `ringOfHalves`             [the ring point having the two given coordinates]

    So: the two real divisions are the generic inverse root and half. This is what makes the
    three round-trip theorems below specialisations of `AnyRing.ringOfHalves`. -/
theorem ofHalves_eq_ringOfHalves (σ u v : ℝ) :
    ofHalves σ u v = AnyRing.ringOfHalves u v (1 / Real.sqrt σ) (1 / 2) := by
  simp only [ofHalves, AnyRing.ringOfHalves]
  apply Prod.ext
  · ring
  · ring

/-- Building from coordinates and reading the first one back returns it.

    {σ : ℝ} (hσ : 0 < σ)                  [the parameter, above zero]
    (u v : ℝ)                             [two coordinates]
    upper σ (ofHalves σ u v) = u          [reading back gives what went in]

    So: `AnyRing.ringUpper_ringOfHalves` at the real witnesses. -/
theorem upper_ofHalves {σ : ℝ} (hσ : 0 < σ) (u v : ℝ) :
    upper σ (ofHalves σ u v) = u := by
  rw [ofHalves_eq_ringOfHalves]
  exact AnyRing.ringUpper_ringOfHalves
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num)

/-- Building from coordinates and reading the second one back returns it.

    {σ : ℝ} (hσ : 0 < σ)                  [the parameter, above zero]
    (u v : ℝ)                             [two coordinates]
    lower σ (ofHalves σ u v) = v          [reading back gives what went in]

    So: `AnyRing.ringLower_ringOfHalves` at the real witnesses. -/
theorem lower_ofHalves {σ : ℝ} (hσ : 0 < σ) (u v : ℝ) :
    lower σ (ofHalves σ u v) = v := by
  rw [ofHalves_eq_ringOfHalves]
  exact AnyRing.ringLower_ringOfHalves
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num)

/-- Reading a point's coordinates and rebuilding from them returns the point.

    {σ : ℝ} (hσ : 0 < σ)                            [the parameter, above zero]
    (z : Cell)                                      [any point]
    ofHalves σ (upper σ z) (lower σ z) = z          [the round trip is the identity]

    So: above zero the two coordinates carry everything the two slots did. This is
    `AnyRing.ringOfHalves_upper_lower` at the real witnesses. -/
theorem ofHalves_upper_lower {σ : ℝ} (hσ : 0 < σ) (z : Cell) :
    ofHalves σ (upper σ z) (lower σ z) = z := by
  rw [ofHalves_eq_ringOfHalves]
  exact AnyRing.ringOfHalves_upper_lower
    (by rw [mul_one_div]; exact div_self (ne_of_gt (Real.sqrt_pos.mpr hσ)))
    (by norm_num) z

/-- Two points agreeing on both coordinates are the same point.

    {σ : ℝ} (hσ : 0 < σ)                     [the parameter, above zero]
    {z w : Cell}                             [two points]
    upper σ z = upper σ w                    [same first coordinate]
    lower σ z = lower σ w                    [same second coordinate]
    →                                        [forces]
    z = w                                    [they are the same point] -/
theorem halves_ext {σ : ℝ} (hσ : 0 < σ) {z w : Cell}
    (hu : upper σ z = upper σ w) (hl : lower σ z = lower σ w) : z = w := by
  calc z = ofHalves σ (upper σ z) (lower σ z) := (ofHalves_upper_lower hσ z).symm
    _ = ofHalves σ (upper σ w) (lower σ w) := by rw [hu, hl]
    _ = w := ofHalves_upper_lower hσ w

-- ═══════════════════════════════════════════════════════════════════
-- THE COORDINATES OF THE FOUR SPECIAL POINTS
-- ═══════════════════════════════════════════════════════════════════

/-- The origin has both coordinates zero.

    (σ : ℝ)                       [the parameter]
    upper σ 0 = 0                 [first coordinate]
    ∧                             [and]
    lower σ 0 = 0                 [second coordinate] -/
theorem halves_zero (σ : ℝ) : upper σ 0 = 0 ∧ lower σ 0 = 0 := by
  simp [upper, lower]

/-- The unit has both coordinates one.

    (σ : ℝ)                         [the parameter]
    upper σ one = 1                 [first coordinate]
    ∧                               [and]
    lower σ one = 1                 [second coordinate] -/
theorem halves_one (σ : ℝ) : upper σ one = 1 ∧ lower σ one = 1 := by
  simp [upper, lower]

/-- `idemPlus` sits at coordinate one on the first channel and zero on the second.

    {σ : ℝ} (hσ : 0 < σ)                       [the parameter, above zero]
    upper σ (idemPlus σ) = 1                   [all of it is in the first channel]
    ∧                                          [and]
    lower σ (idemPlus σ) = 0                   [none of it in the second] -/
theorem halves_idemPlus {σ : ℝ} (hσ : 0 < σ) :
    upper σ (idemPlus σ) = 1 ∧ lower σ (idemPlus σ) = 0 := by
  have hsne : Real.sqrt σ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hσ)
  have key : Real.sqrt σ * (1 / (2 * Real.sqrt σ)) = 1 / 2 := by field_simp
  refine ⟨?_, ?_⟩
  · simp only [upper, idemPlus, key]; norm_num
  · simp only [lower, idemPlus, key]; norm_num

/-- `idemMinus` sits at zero on the first channel and one on the second.

    {σ : ℝ} (hσ : 0 < σ)                        [the parameter, above zero]
    upper σ (idemMinus σ) = 0                   [none of it in the first channel]
    ∧                                           [and]
    lower σ (idemMinus σ) = 1                   [all of it in the second] -/
theorem halves_idemMinus {σ : ℝ} (hσ : 0 < σ) :
    upper σ (idemMinus σ) = 0 ∧ lower σ (idemMinus σ) = 1 := by
  have hsne : Real.sqrt σ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hσ)
  have key : Real.sqrt σ * (1 / (2 * Real.sqrt σ)) = 1 / 2 := by field_simp
  refine ⟨?_, ?_⟩
  · simp only [upper, idemMinus, mul_neg, key]; norm_num
  · simp only [lower, idemMinus, mul_neg, key]; norm_num

-- ═══════════════════════════════════════════════════════════════════
-- THE IDEMPOTENT LOCUS IS FOUR POINTS
-- ═══════════════════════════════════════════════════════════════════

/-- Over the reals, `a² = a` has exactly two answers.

    (a : ℝ)                     [any real number]
    a * a = a                   [unchanged by squaring]
    ↔                           [exactly when]
    a = 0 ∨ a = 1               [it is zero or one]

    So: `a(a − 1) = 0`, and the reals have no zero divisors. -/
theorem real_idem (a : ℝ) : a * a = a ↔ a = 0 ∨ a = 1 := by
  constructor
  · intro h
    have hz : a * (a - 1) = 0 := by linear_combination h
    rcases mul_eq_zero.mp hz with h0 | h1
    · exact Or.inl h0
    · exact Or.inr (by linarith)
  · rintro (rfl | rfl) <;> ring

/-- Being unchanged by squaring is two separate scalar questions.

    {σ : ℝ} (hσ : 0 < σ)                                   [the parameter, above zero]
    (z : Cell)                                             [the point]
    Idempotent σ z                                         [unchanged by squaring]
    ↔                                                      [exactly when]
    upper σ z * upper σ z = upper σ z                      [the first channel fixes itself]
    ∧                                                      [and]
    lower σ z * lower σ z = lower σ z                      [the second channel fixes itself]

    So: multiplication is componentwise here, so squaring is too, and the question splits. -/
theorem idempotent_iff_halves {σ : ℝ} (hσ : 0 < σ) (z : Cell) :
    Idempotent σ z ↔
      (upper σ z * upper σ z = upper σ z ∧ lower σ z * lower σ z = lower σ z) := by
  constructor
  · intro h
    exact ⟨by rw [← upper_mul hσ.le, show mul σ z z = z from h],
      by rw [← lower_mul hσ.le, show mul σ z z = z from h]⟩
  · rintro ⟨hu, hl⟩
    exact halves_ext hσ (by rw [upper_mul hσ.le, hu]) (by rw [lower_mul hσ.le, hl])

/-- Above zero there are exactly four points unchanged by squaring.

    {σ : ℝ} (hσ : 0 < σ)                    [the parameter, above zero]
    (z : Cell)                              [any point]
    Idempotent σ z                          [unchanged by squaring]
    ↔                                       [exactly when]
    z = 0                                   [it is the origin]
    ∨                                       [or]
    z = idemPlus σ                          [it is the first channel's unit]
    ∨                                       [or]
    z = idemMinus σ                         [it is the second channel's unit]
    ∨                                       [or]
    z = one                                 [it is the unit]

    So: the two coordinates each answer `0` or `1` independently, giving four combinations.
    `exists_nontrivial_idempotent_iff_disc_pos` says two of these four appear only above
    zero; this says there are never more than four. -/
theorem idempotent_locus {σ : ℝ} (hσ : 0 < σ) (z : Cell) :
    Idempotent σ z ↔ (z = 0 ∨ z = idemPlus σ ∨ z = idemMinus σ ∨ z = one) := by
  rw [idempotent_iff_halves hσ, real_idem, real_idem]
  obtain ⟨hp1, hp0⟩ := halves_idemPlus hσ
  obtain ⟨hm0, hm1⟩ := halves_idemMinus hσ
  obtain ⟨hz1, hz2⟩ := halves_zero σ
  obtain ⟨ho1, ho2⟩ := halves_one σ
  constructor
  · rintro ⟨hu | hu, hl | hl⟩
    · exact Or.inl (halves_ext hσ (by rw [hu, hz1]) (by rw [hl, hz2]))
    · exact Or.inr (Or.inr (Or.inl (halves_ext hσ (by rw [hu, hm0]) (by rw [hl, hm1]))))
    · exact Or.inr (Or.inl (halves_ext hσ (by rw [hu, hp1]) (by rw [hl, hp0])))
    · exact Or.inr (Or.inr (Or.inr (halves_ext hσ (by rw [hu, ho1]) (by rw [hl, ho2]))))
  · rintro (rfl | rfl | rfl | rfl)
    · exact ⟨Or.inl hz1, Or.inl hz2⟩
    · exact ⟨Or.inr hp1, Or.inl hp0⟩
    · exact ⟨Or.inl hm0, Or.inr hm1⟩
    · exact ⟨Or.inr ho1, Or.inr ho2⟩

/-- On points unchanged by squaring, each coordinate is zero or one.

    {σ : ℝ} (hσ : 0 < σ)                            [the parameter, above zero]
    {z : Cell} (hz : Idempotent σ z)                [a point unchanged by squaring]
    (upper σ z = 0 ∨ upper σ z = 1)                 [its first coordinate is zero or one]
    ∧                                               [and]
    (lower σ z = 0 ∨ lower σ z = 1)                 [so is its second] -/
theorem halves_boolean_of_idempotent {σ : ℝ} (hσ : 0 < σ) {z : Cell}
    (hz : Idempotent σ z) :
    (upper σ z = 0 ∨ upper σ z = 1) ∧ (lower σ z = 0 ∨ lower σ z = 1) := by
  rw [idempotent_iff_halves hσ, real_idem, real_idem] at hz
  exact hz

/-- How many points survive squaring, at every sigma.

    (σ : ℝ)                                                       [the parameter]
    (z : Cell)                                                    [any point of the algebra]
    idemPlus σ                                                    [one extra corner]
    idemMinus σ                                                   [the other extra corner]
    0 < σ → ∀ z, Idempotent σ z ↔                                 [above zero, four points]
      (z = 0 ∨ z = idemPlus σ ∨ z = idemMinus σ ∨ z = one)
    ∧                                                             [and]
    σ ≤ 0 → ∀ z, Idempotent σ z ↔ (z = 0 ∨ z = one)               [otherwise, two points]

    So: the sign of sigma fixes the size of the idempotent monoid, four above zero and two
    at or below. Above zero those four are the corners of `{0,1} × {0,1}` in null-basis
    coordinates, so multiplication on them is AND on each coordinate separately. -/
theorem idempotent_classification (σ : ℝ) :
    (0 < σ → ∀ z : Cell,
        Idempotent σ z ↔ (z = 0 ∨ z = idemPlus σ ∨ z = idemMinus σ ∨ z = one)) ∧
      (σ ≤ 0 → ∀ z : Cell, Idempotent σ z ↔ (z = 0 ∨ z = one)) :=
  ⟨fun hσ z => idempotent_locus hσ z, fun hσ z => idempotent_iff_trivial_of_disc_nonpos hσ z⟩

/-- Above zero the quotient has exactly four elements unchanged by squaring.

    {σ : ℝ} (hσ : 0 < σ)                        [the parameter, above zero]
    (p : AdjoinRoot (rel σ))                    [any element of the quotient ring]
    p * p = p                                   [unchanged by squaring, in the ring]
    ↔                                           [exactly when]
    p = 0                                       [it is zero]
    ∨                                           [or]
    p = toQuot σ (idemPlus σ)                   [one of the two extra elements]
    ∨                                           [or]
    p = toQuot σ (idemMinus σ)                  [the other]
    ∨                                           [or]
    p = 1                                       [the unit]

    So: `idempotent_locus` on the ring. The count is a fact about `ℝ[X]/(X² − σ)`. -/
theorem quot_idempotent_locus {σ : ℝ} (hσ : 0 < σ) (p : AdjoinRoot (rel σ)) :
    p * p = p ↔
      (p = 0 ∨ p = toQuot σ (idemPlus σ) ∨ p = toQuot σ (idemMinus σ) ∨ p = 1) := by
  obtain ⟨z, rfl⟩ := toQuot_surjective σ p
  rw [mul_eq_quotient_mul]
  constructor
  · intro h
    have hz : Idempotent σ z := toQuot_injective σ h
    rcases (idempotent_locus hσ z).mp hz with h0 | hp | hm | h1
    · exact Or.inl ((toQuot_eq_zero_iff σ z).mpr h0)
    · exact Or.inr (Or.inl (congrArg (toQuot σ) hp))
    · exact Or.inr (Or.inr (Or.inl (congrArg (toQuot σ) hm)))
    · exact Or.inr (Or.inr (Or.inr (by rw [h1]; exact toQuot_one σ)))
  · intro h
    refine congrArg (toQuot σ) ((idempotent_locus hσ z).mpr ?_)
    rcases h with h0 | hp | hm | h1
    · exact Or.inl ((toQuot_eq_zero_iff σ z).mp h0)
    · exact Or.inr (Or.inl (toQuot_injective σ hp))
    · exact Or.inr (Or.inr (Or.inl (toQuot_injective σ hm)))
    · exact Or.inr (Or.inr (Or.inr (toQuot_injective σ (by rw [h1]; exact (toQuot_one σ).symm))))

/-- Above zero the two channels are ring maps out of the quotient onto the reals.

    {σ : ℝ} (hσ : 0 ≤ σ)                                       [the parameter, at zero or above]
    (p q : AdjoinRoot (rel σ))                                 [two elements of the ring]
    upper σ ((toQuotEquiv σ).symm (p * q))                     [read channel one of a product]
    = upper σ ((toQuotEquiv σ).symm p) * upper σ ((toQuotEquiv σ).symm q)
                                                               [is the product of the reads]

    So: `upper_mul` on the ring. The two channels split `ℝ[X]/(X² − σ)` into two copies of
    the reals above zero, and this is one half of that splitting. -/
theorem quot_upper_mul {σ : ℝ} (hσ : 0 ≤ σ) (p q : AdjoinRoot (rel σ)) :
    upper σ ((toQuotEquiv σ).symm (p * q))
      = upper σ ((toQuotEquiv σ).symm p) * upper σ ((toQuotEquiv σ).symm q) := by
  obtain ⟨z, rfl⟩ := toQuot_surjective σ p
  obtain ⟨w, rfl⟩ := toQuot_surjective σ q
  have hz : (toQuotEquiv σ).symm (toQuot σ z) = z := by
    rw [← toQuotEquiv_apply]; exact (toQuotEquiv σ).symm_apply_apply z
  have hw : (toQuotEquiv σ).symm (toQuot σ w) = w := by
    rw [← toQuotEquiv_apply]; exact (toQuotEquiv σ).symm_apply_apply w
  rw [mul_eq_quotient_mul]
  have hzw : (toQuotEquiv σ).symm (toQuot σ (mul σ z w)) = mul σ z w := by
    rw [← toQuotEquiv_apply]; exact (toQuotEquiv σ).symm_apply_apply _
  rw [hz, hw, hzw, upper_mul hσ]

-- ═══════════════════════════════════════════════════════════════════
-- THE ORDER ON A CHANNEL, AND WHERE IT COMES FROM
-- ═══════════════════════════════════════════════════════════════════

/-- One real number is below another exactly when the gap between them is a square.

    (a b : ℝ)                              [two real numbers]
    a ≤ b                                  [the first is below the second]
    ↔                                      [exactly when]
    ∃ s : ℝ, b - a = s ^ 2                 [the gap is something squared]

    So: real-closedness. Which way round two numbers are is settled by the ring operations and
    nothing else, so an order built on this adds no structure to the algebra. -/
theorem le_iff_exists_sq (a b : ℝ) : a ≤ b ↔ ∃ s : ℝ, b - a = s ^ 2 := by
  constructor
  · intro h
    exact ⟨Real.sqrt (b - a), (Real.sq_sqrt (sub_nonneg.mpr h)).symm⟩
  · rintro ⟨s, hs⟩
    have hnn : (0 : ℝ) ≤ b - a := hs ▸ sq_nonneg s
    linarith

/-- Every real number is some point's upper reading.

    {σ : ℝ} (hσ : 0 < σ)                   [the parameter, above zero]
    (u : ℝ)                                [any real number]
    ∃ z : Cell, upper σ z = u              [some point reads that way on the upper channel] -/
theorem exists_upper {σ : ℝ} (hσ : 0 < σ) (u : ℝ) : ∃ z : Cell, upper σ z = u :=
  ⟨ofHalves σ u 0, upper_ofHalves hσ u 0⟩

/-- Every real number is some point's lower reading.

    {σ : ℝ} (hσ : 0 < σ)                   [the parameter, above zero]
    (v : ℝ)                                [any real number]
    ∃ z : Cell, lower σ z = v              [some point reads that way on the lower channel] -/
theorem exists_lower {σ : ℝ} (hσ : 0 < σ) (v : ℝ) : ∃ z : Cell, lower σ z = v :=
  ⟨ofHalves σ 0 v, lower_ofHalves hσ 0 v⟩

/-- The upper channel's order is the algebra's own.

    {σ : ℝ} (hσ : 0 < σ)                              [the parameter, above zero]
    (z w : Cell)                                      [two points]
    upper σ z ≤ upper σ w                             [one reads below the other]
    ↔                                                 [exactly when]
    ∃ s : Cell, upper σ w - upper σ z = upper σ s ^ 2 [the gap is a squared reading]

    So: the comparison a propagator makes on this channel is definable from the algebra, with
    no order supplied from outside. This is what `../song/src/prover/cone/cell.slang` names as
    owed to the spec side under C3. -/
theorem upper_le_iff_exists_sq {σ : ℝ} (hσ : 0 < σ) (z w : Cell) :
    upper σ z ≤ upper σ w ↔ ∃ s : Cell, upper σ w - upper σ z = upper σ s ^ 2 := by
  rw [le_iff_exists_sq]
  constructor
  · rintro ⟨t, ht⟩
    obtain ⟨s, hs⟩ := exists_upper hσ t
    exact ⟨s, by rw [hs]; exact ht⟩
  · rintro ⟨s, hs⟩
    exact ⟨upper σ s, hs⟩

/-- The lower channel's order is the algebra's own, by the same argument.

    {σ : ℝ} (hσ : 0 < σ)                              [the parameter, above zero]
    (z w : Cell)                                      [two points]
    lower σ z ≤ lower σ w                             [one reads below the other]
    ↔                                                 [exactly when]
    ∃ s : Cell, lower σ w - lower σ z = lower σ s ^ 2 [the gap is a squared reading] -/
theorem lower_le_iff_exists_sq {σ : ℝ} (hσ : 0 < σ) (z w : Cell) :
    lower σ z ≤ lower σ w ↔ ∃ s : Cell, lower σ w - lower σ z = lower σ s ^ 2 := by
  rw [le_iff_exists_sq]
  constructor
  · rintro ⟨t, ht⟩
    obtain ⟨s, hs⟩ := exists_lower hσ t
    exact ⟨s, by rw [hs]; exact ht⟩
  · rintro ⟨s, hs⟩
    exact ⟨lower σ s, hs⟩

/-- Over the whole numbers one square is not enough: two is not a square.

    ¬ ∃ s : ℤ, (2 : ℤ) = s ^ 2        [no whole number squares to two]

    So: `le_iff_exists_sq` does not transport. The channel a decider carries holds whole
    numbers, not reals, and there the gap between two readings need not be one square. -/
theorem two_not_sq : ¬ ∃ s : ℤ, (2 : ℤ) = s ^ 2 := by
  rintro ⟨s, hs⟩
  have h1 : s ^ 2 = 2 := hs.symm
  have hl : (-2 : ℤ) < s := by nlinarith
  have hr : s < 2 := by nlinarith
  interval_cases s <;> norm_num at h1

/-- Three squares are not enough either: seven is not a sum of three.

    ¬ ∃ x y z : ℤ, (7 : ℤ) = x ^ 2 + y ^ 2 + z ^ 2    [no three whole squares sum to seven]

    So: with `two_not_sq` this pins the count at four. A square is `0`, `1` or `4` modulo eight,
    and no three of those sum to seven modulo eight. Mathlib carries Lagrange but not Legendre's
    three-square theorem, so the bound is proved here rather than cited. -/
theorem seven_not_sum_three_sq : ¬ ∃ x y z : ℤ, (7 : ℤ) = x ^ 2 + y ^ 2 + z ^ 2 := by
  rintro ⟨x, y, z, h⟩
  have h8 : ((7 : ℤ) : ZMod 8) = ((x ^ 2 + y ^ 2 + z ^ 2 : ℤ) : ZMod 8) := by rw [h]
  push_cast at h8
  revert h8
  generalize ((x : ℤ) : ZMod 8) = a
  generalize ((y : ℤ) : ZMod 8) = b
  generalize ((z : ℤ) : ZMod 8) = c
  revert a b c
  decide

/-- Over the whole numbers the order is still the ring's, at four squares instead of one.

@cite artin-schreier-1927

    (a b : ℤ)                                              [two whole numbers]
    a ≤ b                                                  [the first is below the second]
    ↔                                                      [exactly when]
    ∃ w x y z : ℤ, b - a = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2   [the gap is four squares summed]

    So: Lagrange's four squares carries over the integers what real-closedness carries over the
    reals. `two_not_sq` and `seven_not_sum_three_sq` say one and three will not do it, so four
    is the count and this is a different theorem rather than the real one restated. Both
    directions matter: sums of squares are never negative, which makes the comparison sound,
    and every non-negative whole number is such a sum, which makes it complete.

    This is the second half of what `../song/src/prover/cone/cell.slang` names under C3. Its
    `ncone0` propagates over whole numbers, so the real statement above does not reach it. -/
theorem int_le_iff_exists_sum_four_sq (a b : ℤ) :
    a ≤ b ↔ ∃ w x y z : ℤ, b - a = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2 := by
  constructor
  · intro h
    obtain ⟨p, q, r, s, hpqrs⟩ := Nat.sum_four_squares (b - a).toNat
    refine ⟨(p : ℤ), (q : ℤ), (r : ℤ), (s : ℤ), ?_⟩
    have hc : ((p : ℤ)) ^ 2 + (q : ℤ) ^ 2 + (r : ℤ) ^ 2 + (s : ℤ) ^ 2
        = ((b - a).toNat : ℤ) := by exact_mod_cast congrArg (fun n : ℕ => (n : ℤ)) hpqrs
    rw [Int.toNat_of_nonneg (sub_nonneg.mpr h)] at hc
    exact hc.symm
  · rintro ⟨w, x, y, z, hwxyz⟩
    nlinarith [sq_nonneg w, sq_nonneg x, sq_nonneg y, sq_nonneg z]

/-- The cone is exactly the two rays where a channel reads zero.

    {σ : ℝ} (hσ : 0 ≤ σ)                                [the parameter, at zero or above]
    (z : Cell)                                          [a point]
    normForm σ z = 0                                        [it has size zero]
    ↔                                                   [exactly when]
    upper σ z = 0 ∨ lower σ z = 0                       [one of its two channels reads zero]

    So: the size is the product of the two readings
    (`normForm_eq_upper_mul_lower`), and a product of
    reals is zero exactly when a factor is. The degenerate locus is not a vague region, it is the
    union of two lines, one per channel. -/
theorem normForm_zero_iff_channel_zero {σ : ℝ} (hσ : 0 ≤ σ) (z : Cell) :
    normForm σ z = 0 ↔ upper σ z = 0 ∨ lower σ z = 0 := by
  rw [normForm_eq_upper_mul_lower hσ, mul_eq_zero]

/-- At zero the two channels are one channel.

    (z : Cell)                       [a point]
    upper 0 z = lower 0 z            [both readings agree]

    So: the parabolic face has no second channel to be independent. Whatever distinguishes two
    points above zero is unavailable here, and `no_channel_of_disc_neg` says the elliptic face
    has no channel reading at all. -/
theorem upper_eq_lower_of_disc_zero (z : Cell) : upper 0 z = lower 0 z := by
  simp [upper, lower]

-- ═══════════════════════════════════════════════════════════════════
-- THE LATTICE THE CHANNELS CARRY
-- ═══════════════════════════════════════════════════════════════════

/-- The numbers a point stands for, read as an interval between its two channels.

    (σ : ℝ)                                        [the parameter]
    (t : ℝ)                                        [a number]
    (z : Cell)                                     [a point]
    lower σ z ≤ t ∧ t ≤ upper σ z                  [the number sits between the two readings]

    So: this is how `ncone0` holds a variable's feasible set, with the pair `(lo, hi)` being the
    point's two null coordinates. -/
def InCell (σ : ℝ) (t : ℝ) (z : Cell) : Prop := lower σ z ≤ t ∧ t ≤ upper σ z

/-- Meeting two points channel by channel.

    (σ : ℝ)                                        [the parameter]
    (z w : Cell)                                   [two points]
    : Cell                                         [the point holding the tighter of each channel]

    So: the propagation step. The lower channel takes the larger of the two and the upper channel
    the smaller, which is what intersecting two intervals does. Neither is a ring operation, and
    `upper_le_iff_exists_sq` is why taking them is not smuggling structure in. -/
noncomputable def meet (σ : ℝ) (z w : Cell) : Cell :=
  ofHalves σ (min (upper σ z) (upper σ w)) (max (lower σ z) (lower σ w))

/-- The meet's upper channel is the smaller of the two.

    {σ : ℝ} (hσ : 0 < σ)                                          [the parameter, above zero]
    (z w : Cell)                                                  [two points]
    upper σ (meet σ z w) = min (upper σ z) (upper σ w)            [the tighter upper bound] -/
theorem upper_meet {σ : ℝ} (hσ : 0 < σ) (z w : Cell) :
    upper σ (meet σ z w) = min (upper σ z) (upper σ w) :=
  upper_ofHalves hσ _ _

/-- The meet's lower channel is the larger of the two.

    {σ : ℝ} (hσ : 0 < σ)                                          [the parameter, above zero]
    (z w : Cell)                                                  [two points]
    lower σ (meet σ z w) = max (lower σ z) (lower σ w)            [the tighter lower bound] -/
theorem lower_meet {σ : ℝ} (hσ : 0 < σ) (z w : Cell) :
    lower σ (meet σ z w) = max (lower σ z) (lower σ w) :=
  lower_ofHalves hσ _ _

/-- The meet holds exactly the numbers both points hold.

    {σ : ℝ} (hσ : 0 < σ)                                   [the parameter, above zero]
    (t : ℝ) (z w : Cell)                                   [a number and two points]
    InCell σ t (meet σ z w)                                [the meet holds it]
    ↔                                                      [exactly when]
    InCell σ t z ∧ InCell σ t w                            [both points hold it]

    So: propagation loses no solution and keeps no non-solution. The two directions are what
    `../song/src/prover/cone/cell.slang` proves separately over the integers as soundness and
    tightness; here they are one statement over the reals, against `Cell σ`. -/
theorem inCell_meet_iff {σ : ℝ} (hσ : 0 < σ) (t : ℝ) (z w : Cell) :
    InCell σ t (meet σ z w) ↔ InCell σ t z ∧ InCell σ t w := by
  simp only [InCell, upper_meet hσ, lower_meet hσ, max_le_iff, le_min_iff]
  constructor
  · rintro ⟨⟨h1, h2⟩, h3, h4⟩
    exact ⟨⟨h1, h3⟩, h2, h4⟩
  · rintro ⟨⟨h1, h2⟩, h3, h4⟩
    exact ⟨⟨h1, h3⟩, h2, h4⟩

/-- A point is a crossing when its lower reading has overtaken its upper one.

    (σ : ℝ)                            [the parameter]
    (z : Cell)                         [a point]
    upper σ z < lower σ z              [the upper reading sits below the lower one]

    So: an empty interval, and the condition a propagator reports as unsatisfiable. In Kaucher's
    completed interval arithmetic these are the improper intervals and are elements in good
    standing rather than a failure state. -/
def Crossing (σ : ℝ) (z : Cell) : Prop := upper σ z < lower σ z

/-- Crossing and lying on the null cone are independent conditions.

    {σ : ℝ} (hσ : 0 < σ)                                  [the parameter, above zero]
    ∃ z : Cell, Crossing σ z ∧ normForm σ z ≠ 0               [a crossing off the cone]
    ∃ z : Cell, ¬ Crossing σ z ∧ normForm σ z = 0             [a non-crossing on it]
    ∧                                                     [both together]

    So: the two loci are different sets, and neither contains the other. A propagator's verdict
    rides the first and the algebra's degeneracy rides the second, so reading one as the other
    is a mistake about which set is which. This is what
    `../song/src/prover/cone/cell.slang` establishes under C4 over the whole numbers, stated
    here over the reals against `Cell σ`. -/
theorem crossing_independent_of_cone {σ : ℝ} (hσ : 0 < σ) :
    (∃ z : Cell, Crossing σ z ∧ normForm σ z ≠ 0) ∧
      (∃ z : Cell, ¬ Crossing σ z ∧ normForm σ z = 0) := by
  constructor
  · refine ⟨ofHalves σ 1 2, ?_, ?_⟩
    · simp only [Crossing, upper_ofHalves hσ, lower_ofHalves hσ]; norm_num
    · rw [normForm_eq_upper_mul_lower hσ.le, upper_ofHalves hσ, lower_ofHalves hσ]; norm_num
  · refine ⟨ofHalves σ 1 0, ?_, ?_⟩
    · simp only [Crossing, upper_ofHalves hσ, lower_ofHalves hσ]; norm_num
    · rw [normForm_eq_upper_mul_lower hσ.le, upper_ofHalves hσ, lower_ofHalves hσ]; norm_num

end Trichotomy
end Apeiron

end
