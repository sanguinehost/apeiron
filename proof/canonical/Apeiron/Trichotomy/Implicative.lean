import Apeiron.Trichotomy.Derived.Application
import Mathlib.Order.CompleteLattice.Basic

/-!
# The face, its order, and the implicative structure that order carries

@name Implicative
@cite artin-schreier-1927
@cite boole-1847
@cite stone-1936
@cite miquel-2018
@cite palmieri-2026
@cite hofstra-vanoosten-2003
@cite hyland-johnstone-pitts-1980
@cite rodenburg-2023
@grade proved

## Root

| from | here |
|---|---|
| `two_channels_iff_disc_pos`, `channel_iff_of_disc_pos` | `two_boolean_channels_iff_disc_pos` |
| `upper_le_iff_exists_sq`, `lower_le_iff_exists_sq` | `precedes_iff_exists_sq`, `precedes_no_top` |
| `idempotent_classification`, `halves_boolean_of_idempotent` | `idem_mul_is_meet`,
  `idem_compl_partition`, `conj_is_swap_not_complement` |
| `ratio_ofHalves`, `ratio_ofHalves_of_disc_nonpos` | `nodeOp_faces`, `nodeOp_no_K` |
| `idempotent_idemPlus`, `idempotent_iff_trivial_of_disc_nonpos` | `no_iso_across_the_sign` |

Two distinct reads of a point exist above zero and nowhere else, so a point surviving squaring
whose two channels carry independent truth values exists on that face alone. That is the face
every construction here sits on. The comparison on each channel is an existence statement about
squares, so the order below is the algebra's own; and that order has no greatest point, so the
channels are not a complete lattice and Miquel's Definition 2.1 does not reach them without a
completion. The four survivors above zero are where it does reach: they are a Boolean algebra
whose meet is `mul σ`, whose complement is subtraction from the unit, and whose channel swap is
a different symmetry from that complement.

The face itself is a theorem rather than a background assumption. `positive_face_is_one_face`
gives a bijection carrying `mul σ` onto `mul τ` for any two positive values of sigma, keeping
the unit and both channel readings, and `ordered_positive_face_is_one_face` adds that it
preserves the order both ways. `no_iso_across_the_sign` says no injective multiplicative map
runs from a positive sigma to a nonpositive one. So every construction here reads which face it
is on, every positive sigma gives the same structure because they are the same face, and a
construction separating `σ = 1` from `σ = 4` would need a primitive not invariant under that
stretch, meaning one carrying an absolute scale on the second slot rather than a ratio of
readings.

## What the face carries

`Precedes` compares two points channel by channel. `precedes_iff_exists_sq` says each
comparison is an existence statement about squares, which is `upper_le_iff_exists_sq`, so the
order is the algebra's. `precedes_no_top` says the order has no greatest point, so the channels
are not a complete lattice and Miquel's Definition 2.1 does not apply to them as they stand.

The idempotents are where it does apply. Above zero there are four, `idempotent_classification`,
their two channels each read `0` or `1`, and `idem_mul_min` says multiplying them takes the
smaller reading on each channel. So `mul σ` restricted to the idempotents is the meet of the
channel order, and the four points are the corners of a two-element Boolean algebra.

## Where the route stops

Miquel's implicative structure derives application from the order: `a b` is the meet of every
`c` with `a ≼ (b → c)`. `implApply_ofAdjunction` computes that meet for an implication with an
adjunction, and the answer is `a ⊓ b`. A meet is commutative and associative, and
`no_K_of_comm_assoc` says no such operation has a discarding element unless the carrier is a
single point. So the structure exists on this face and its derived application is not the
calculus's application. This is Miquel's Remark 2.25 in the concrete case, and it is why the
tripos such a structure generates is a forcing tripos rather than a realizability one.

`node_op_no_K` closes the other candidate the corpus already had. `ratio` packed over
`ofHalves` is the Tower's node rule, it is division above zero, and division has no discarding
element either.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- EVERY POSITIVE SIGMA IS ONE FACE
-- ═══════════════════════════════════════════════════════════════════

/-- Stretch the second slot so one positive value of sigma reads as another.

    (σ τ : ℝ)                                     [two values of sigma]
    (z : Cell)                                    [a point]
    (z.1, Real.sqrt σ / Real.sqrt τ * z.2)        [first slot kept, second slot stretched]
    : Cell                                        [a point of the other algebra]

    So: the second slot is measured in units of the generator, and the generator squares to
    sigma, so varying sigma is a change of unit on that slot. -/
def rescaleFace (σ τ : ℝ) (z : Cell) : Cell :=
  (z.1, Real.sqrt σ / Real.sqrt τ * z.2)

/-- The stretch carries one multiplication onto the other.

    {σ τ : ℝ} (hσ : 0 ≤ σ)                          [the source value of sigma, at zero or above]
    (hτ : 0 < τ)                                       [the target value of sigma, above zero]
    (z w : Cell)                                       [two points]
    rescaleFace σ τ (mul σ z w)                        [multiply the σ way, then stretch]
    = mul τ (rescaleFace σ τ z) (rescaleFace σ τ w)    [stretch both, then multiply the τ way]

    So: the first slot picks up `τ` times the square of the stretch, which is `σ`. -/
theorem rescaleFace_mul {σ τ : ℝ} (hσ : 0 ≤ σ) (hτ : 0 < τ) (z w : Cell) :
    rescaleFace σ τ (mul σ z w) = mul τ (rescaleFace σ τ z) (rescaleFace σ τ w) := by
  have hcc : τ * (Real.sqrt σ / Real.sqrt τ) ^ 2 = σ := by
    rw [div_pow, Real.sq_sqrt hσ, Real.sq_sqrt hτ.le]
    field_simp
  simp only [rescaleFace, mul, Prod.ext_iff]
  refine ⟨?_, by ring⟩
  linear_combination (-(z.2 * w.2)) * hcc

/-- The stretch keeps the unit.

    (σ τ : ℝ)                        [two values of sigma]
    rescaleFace σ τ one = one        [the unit goes to the unit] -/
theorem rescaleFace_one (σ τ : ℝ) : rescaleFace σ τ one = one := by
  simp [rescaleFace, one]

/-- The stretch keeps the first channel's reading.

    {σ τ : ℝ} (hτ : 0 < τ)                        [two values of sigma, the target above zero]
    (z : Cell)                                    [a point]
    upper τ (rescaleFace σ τ z) = upper σ z       [the reading is unchanged] -/
theorem upper_rescaleFace {σ τ : ℝ} (hτ : 0 < τ) (z : Cell) :
    upper τ (rescaleFace σ τ z) = upper σ z := by
  have hsτ : Real.sqrt τ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hτ)
  simp only [upper, rescaleFace]
  field_simp

/-- The stretch keeps the second channel's reading.

    {σ τ : ℝ} (hτ : 0 < τ)                        [two values of sigma, the target above zero]
    (z : Cell)                                    [a point]
    lower τ (rescaleFace σ τ z) = lower σ z       [the reading is unchanged] -/
theorem lower_rescaleFace {σ τ : ℝ} (hτ : 0 < τ) (z : Cell) :
    lower τ (rescaleFace σ τ z) = lower σ z := by
  have hsτ : Real.sqrt τ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hτ)
  simp only [lower, rescaleFace]
  field_simp

/-- Stretching one way and back is the identity.

    {σ τ : ℝ} (hσ : 0 < σ) (hτ : 0 < τ)                  [two values of sigma, both above zero]
    (z : Cell)                                           [a point]
    rescaleFace τ σ (rescaleFace σ τ z) = z              [the round trip changes nothing] -/
theorem rescaleFace_left_inverse {σ τ : ℝ} (hσ : 0 < σ) (hτ : 0 < τ) (z : Cell) :
    rescaleFace τ σ (rescaleFace σ τ z) = z := by
  have hsσ : Real.sqrt σ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hσ)
  have hsτ : Real.sqrt τ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hτ)
  rw [Prod.ext_iff]
  refine ⟨rfl, ?_⟩
  simp only [rescaleFace]
  field_simp

/-- Above zero every sigma gives the same algebra, with the same two channels.

@cite weierstrass-1884

    {σ τ : ℝ} (hσ : 0 < σ) (hτ : 0 < τ)                   [two values of sigma, both above zero]
    ∃ f : Cell → Cell                                     [there is a map on points]
    Function.Bijective f                                  [it pairs the two algebras off]
    ∀ z w, f (mul σ z w) = mul τ (f z) (f w)              [it carries one product to the other]
    f one = one                                           [it keeps the unit]
    ∀ z, upper τ (f z) = upper σ z                        [it keeps the first reading]
    ∀ z, lower τ (f z) = lower σ z                        [it keeps the second reading]
    ∧                                                     [all five together]

    So: `σ = 1` and `σ = 4` are the same object relabelled, and every construction built from
    `mul`, the unit and the two channels reads the face rather than the number. What a
    construction would need in order to separate two positive values of sigma is a primitive
    that is not invariant under this stretch, which means one carrying an absolute scale on the
    second slot rather than a ratio of readings. -/
theorem positive_face_is_one_face {σ τ : ℝ} (hσ : 0 < σ) (hτ : 0 < τ) :
    ∃ f : Cell → Cell, Function.Bijective f ∧
      (∀ z w : Cell, f (mul σ z w) = mul τ (f z) (f w)) ∧
      f one = one ∧
      (∀ z : Cell, upper τ (f z) = upper σ z) ∧
      (∀ z : Cell, lower τ (f z) = lower σ z) := by
  refine ⟨rescaleFace σ τ, ⟨?_, ?_⟩, rescaleFace_mul hσ.le hτ, rescaleFace_one σ τ,
    upper_rescaleFace hτ, lower_rescaleFace hτ⟩
  · intro z w h
    have := congrArg (rescaleFace τ σ) h
    rwa [rescaleFace_left_inverse hσ hτ, rescaleFace_left_inverse hσ hτ] at this
  · intro w
    exact ⟨rescaleFace τ σ w, rescaleFace_left_inverse hτ hσ w⟩

/-- No such map crosses the sign of sigma.

    {σ τ : ℝ} (hσ : 0 < σ) (hτ : τ ≤ 0)                   [σ above zero, τ at or below]
    ¬ ∃ f : Cell → Cell                                   [there is no map on points]
    Function.Injective f                                  [pairing distinct points off]
    ∀ z w, f (mul σ z w) = mul τ (f z) (f w)              [carrying one product to the other]
    f one = one                                           [keeping the unit]
    f 0 = 0                                               [keeping the origin]
    ∧                                                     [all four together]

    So: `idemPlus σ` survives squaring and is neither the origin nor the unit, its image
    survives squaring at the other value of sigma, and at or below zero the only two survivors
    are the origin and the unit. So the sign of sigma is exactly what the algebra exports to
    any construction that sees only multiplication. -/
theorem no_iso_across_the_sign {σ τ : ℝ} (hσ : 0 < σ) (hτ : τ ≤ 0) :
    ¬ ∃ f : Cell → Cell, Function.Injective f ∧
      (∀ z w : Cell, f (mul σ z w) = mul τ (f z) (f w)) ∧ f one = one ∧ f 0 = 0 := by
  rintro ⟨f, hinj, hmul, hone, hzero⟩
  have hidem : Idempotent τ (f (idemPlus σ)) := by
    have := hmul (idemPlus σ) (idemPlus σ)
    rw [idempotent_idemPlus hσ] at this
    exact this.symm
  obtain ⟨hne0, hne1⟩ := idemPlus_nontrivial σ
  rcases (idempotent_iff_trivial_of_disc_nonpos hτ _).mp hidem with h | h
  · exact hne0 (hinj (h.trans hzero.symm))
  · exact hne1 (hinj (h.trans hone.symm))

-- ═══════════════════════════════════════════════════════════════════
-- THE ORDER THE CHANNELS CARRY
-- ═══════════════════════════════════════════════════════════════════

/-- One point sits below another when both its readings do.

    (σ : ℝ)                                            [the parameter]
    (z w : Cell)                                       [two points]
    upper σ z ≤ upper σ w ∧ lower σ z ≤ lower σ w      [both readings compare that way]

    So: the product order on the two channels. `precedes_iff_exists_sq` says it is the
    algebra's own order rather than one supplied from outside. -/
def Precedes (σ : ℝ) (z w : Cell) : Prop :=
  upper σ z ≤ upper σ w ∧ lower σ z ≤ lower σ w

/-- The order is definable from multiplying and adding.

@cite artin-schreier-1927

    {σ : ℝ} (hσ : 0 < σ)                                       [the parameter, above zero]
    (z w : Cell)                                               [two points]
    Precedes σ z w                                             [one sits below the other]
    ↔                                                          [exactly when]
    (∃ s : Cell, upper σ w - upper σ z = upper σ s ^ 2)         [the first gap is a square]
    ∧ (∃ s : Cell, lower σ w - lower σ z = lower σ s ^ 2)       [and so is the second]

    So: real-closedness on each channel separately. This is `upper_le_iff_exists_sq` and its
    mirror, and it is what makes an order-based semantics over this face honest. -/
theorem precedes_iff_exists_sq {σ : ℝ} (hσ : 0 < σ) (z w : Cell) :
    Precedes σ z w ↔
      (∃ s : Cell, upper σ w - upper σ z = upper σ s ^ 2) ∧
      (∃ s : Cell, lower σ w - lower σ z = lower σ s ^ 2) := by
  simp only [Precedes, upper_le_iff_exists_sq hσ, lower_le_iff_exists_sq hσ]

/-- Every point sits below itself.

    (σ : ℝ) (z : Cell)             [the parameter and a point]
    Precedes σ z z                 [it sits below itself] -/
theorem precedes_refl (σ : ℝ) (z : Cell) : Precedes σ z z := ⟨le_refl _, le_refl _⟩

/-- The order chains.

    {σ : ℝ} {z w v : Cell}                       [the parameter and three points]
    Precedes σ z w                               [the first sits below the second]
    Precedes σ w v                               [the second below the third]
    Precedes σ z v                               [so the first sits below the third] -/
theorem precedes_trans {σ : ℝ} {z w v : Cell} (h₁ : Precedes σ z w) (h₂ : Precedes σ w v) :
    Precedes σ z v := ⟨h₁.1.trans h₂.1, h₁.2.trans h₂.2⟩

/-- Above zero, two points below each other are the same point.

    {σ : ℝ} (hσ : 0 < σ)                         [the parameter, above zero]
    {z w : Cell}                                 [two points]
    Precedes σ z w                               [each sits below the other]
    Precedes σ w z
    z = w                                        [so they are the same point]

    So: the two readings determine the point on this face, `halves_ext`. At or below zero the
    readings coincide and the order is blind to the second slot. -/
theorem precedes_antisymm {σ : ℝ} (hσ : 0 < σ) {z w : Cell}
    (h₁ : Precedes σ z w) (h₂ : Precedes σ w z) : z = w :=
  halves_ext hσ (le_antisymm h₁.1 h₂.1) (le_antisymm h₁.2 h₂.2)

/-- The order on the channels has no greatest point.

    {σ : ℝ} (hσ : 0 < σ)                         [the parameter, above zero]
    ¬ ∃ t : Cell, ∀ z : Cell, Precedes σ z t     [no point has every point below it]

    So: the first channel takes every real value, `exists_upper`, and the reals have no
    greatest element. A complete lattice needs a greatest point, so the channels are not one,
    and Miquel's Definition 2.1 does not apply to them without a completion. The idempotents
    below are the sublattice where it does apply. -/
theorem precedes_no_top {σ : ℝ} (hσ : 0 < σ) : ¬ ∃ t : Cell, ∀ z : Cell, Precedes σ z t := by
  rintro ⟨t, ht⟩
  obtain ⟨z, hz⟩ := exists_upper hσ (upper σ t + 1)
  have := (ht z).1
  rw [hz] at this
  linarith

/-- Above zero, every value of sigma carries the same ordered structure.

    {σ τ : ℝ} (hσ : 0 < σ) (hτ : 0 < τ)             [two values of sigma, both above zero]
    Function.Bijective (rescaleFace σ τ)            [the stretch pairs the two algebras off]
    ∀ z w : Cell                                    [for every two points of the algebra]
    Precedes σ z w                                  [one sits below the other]
    ↔                                               [exactly when]
    Precedes τ (rescaleFace σ τ z) (rescaleFace σ τ w)      [their two stretches do]
    ∧                                               [both together]

    So: an order-based semantics over the channels reads which face it is on and reads no
    finer. Two positive values of sigma are the same ordered structure relabelled, so
    `σ = 1` and `σ = 4` are indistinguishable to anything built from the order. -/
theorem ordered_positive_face_is_one_face {σ τ : ℝ} (hσ : 0 < σ) (hτ : 0 < τ) :
    Function.Bijective (rescaleFace σ τ) ∧
      ∀ z w : Cell, (Precedes σ z w ↔
        Precedes τ (rescaleFace σ τ z) (rescaleFace σ τ w)) := by
  refine ⟨⟨fun z w h => ?_, fun w => ⟨rescaleFace τ σ w, rescaleFace_left_inverse hτ hσ w⟩⟩,
    fun z w => ?_⟩
  · have := congrArg (rescaleFace τ σ) h
    rwa [rescaleFace_left_inverse hσ hτ, rescaleFace_left_inverse hσ hτ] at this
  · simp only [Precedes, upper_rescaleFace hτ, lower_rescaleFace hτ]

-- ═══════════════════════════════════════════════════════════════════
-- THE IDEMPOTENTS ARE THE SUBLATTICE, AND MULTIPLYING IS THE MEET
-- ═══════════════════════════════════════════════════════════════════

/-- On points that survive squaring, multiplying takes the smaller first reading.

@cite peirce-1870

    {σ : ℝ} (hσ : 0 < σ)                                       [the parameter, above zero]
    {z w : Cell}                                               [two points]
    (hz : Idempotent σ z) (hw : Idempotent σ w)                [each survives squaring]
    upper σ (mul σ z w) = min (upper σ z) (upper σ w)          [multiplying meets that channel]

    So: each reading is `0` or `1`, and on those two values a product is a minimum. -/
theorem idem_mul_min_upper {σ : ℝ} (hσ : 0 < σ) {z w : Cell}
    (hz : Idempotent σ z) (hw : Idempotent σ w) :
    upper σ (mul σ z w) = min (upper σ z) (upper σ w) := by
  rw [upper_mul hσ.le]
  rcases (halves_boolean_of_idempotent hσ hz).1 with h | h <;>
    rcases (halves_boolean_of_idempotent hσ hw).1 with h' | h' <;>
      rw [h, h'] <;> norm_num

/-- On points that survive squaring, multiplying takes the smaller second reading.

    {σ : ℝ} (hσ : 0 < σ)                                       [the parameter, above zero]
    {z w : Cell}                                               [two points]
    (hz : Idempotent σ z) (hw : Idempotent σ w)                [each survives squaring]
    lower σ (mul σ z w) = min (lower σ z) (lower σ w)          [multiplying meets that channel] -/
theorem idem_mul_min_lower {σ : ℝ} (hσ : 0 < σ) {z w : Cell}
    (hz : Idempotent σ z) (hw : Idempotent σ w) :
    lower σ (mul σ z w) = min (lower σ z) (lower σ w) := by
  rw [lower_mul hσ.le]
  rcases (halves_boolean_of_idempotent hσ hz).2 with h | h <;>
    rcases (halves_boolean_of_idempotent hσ hw).2 with h' | h' <;>
      rw [h, h'] <;> norm_num

/-- Multiplying two survivors is their meet in the channel order.

@cite boole-1847
@cite stone-1936

    {σ : ℝ} (hσ : 0 < σ)                                     [the parameter, above zero]
    {z w : Cell}                                             [two points]
    (hz : Idempotent σ z) (hw : Idempotent σ w)              [each survives squaring]
    Precedes σ (mul σ z w) z                                 [the product sits below the first]
    Precedes σ (mul σ z w) w                                 [and below the second]
    ∀ v, Idempotent σ v → Precedes σ v z → Precedes σ v w →  [and any survivor below both]
      Precedes σ v (mul σ z w)                               [sits below the product]
    ∧                                                        [all three together]

    So: `mul σ` restricted to the four survivors is the meet of the order the channels carry.
    That is the operation `implApply_ofAdjunction` below computes. -/
theorem idem_mul_is_meet {σ : ℝ} (hσ : 0 < σ) {z w : Cell}
    (hz : Idempotent σ z) (hw : Idempotent σ w) :
    Precedes σ (mul σ z w) z ∧ Precedes σ (mul σ z w) w ∧
      ∀ v : Cell, Idempotent σ v → Precedes σ v z → Precedes σ v w →
        Precedes σ v (mul σ z w) := by
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ?_⟩
  · rw [idem_mul_min_upper hσ hz hw]; exact min_le_left _ _
  · rw [idem_mul_min_lower hσ hz hw]; exact min_le_left _ _
  · rw [idem_mul_min_upper hσ hz hw]; exact min_le_right _ _
  · rw [idem_mul_min_lower hσ hz hw]; exact min_le_right _ _
  · intro v _ hvz hvw
    exact ⟨by rw [idem_mul_min_upper hσ hz hw]; exact le_min hvz.1 hvw.1,
      by rw [idem_mul_min_lower hσ hz hw]; exact le_min hvz.2 hvw.2⟩

/-- The complement of a survivor is a survivor.

    {σ : ℝ} (hσ : 0 < σ)                       [the parameter, above zero]
    {z : Cell} (hz : Idempotent σ z)           [a point surviving squaring]
    Idempotent σ (one - z)                     [the unit minus it also survives]

    So: each reading of `one - z` is one minus that reading of `z`, and `1 - 0` and `1 - 1`
    are again `0` and `1`. -/
theorem idem_compl {σ : ℝ} (hσ : 0 < σ) {z : Cell} (hz : Idempotent σ z) :
    Idempotent σ (one - z) := by
  have hu : upper σ (one - z) = 1 - upper σ z := by
    simp only [upper, one, Prod.fst_sub, Prod.snd_sub]; ring
  have hl : lower σ (one - z) = 1 - lower σ z := by
    simp only [lower, one, Prod.fst_sub, Prod.snd_sub]; ring
  rw [idempotent_iff_halves hσ, hu, hl]
  obtain ⟨hbu, hbl⟩ := halves_boolean_of_idempotent hσ hz
  constructor
  · rcases hbu with h | h <;> rw [h] <;> ring
  · rcases hbl with h | h <;> rw [h] <;> ring

/-- A survivor and its complement multiply to the origin and add to the unit.

@cite peirce-1870

    {σ : ℝ} (hσ : 0 < σ)                       [the parameter, above zero]
    {z : Cell} (hz : Idempotent σ z)           [a point surviving squaring]
    mul σ z (one - z) = 0                      [the two meet at the origin]
    z + (one - z) = one                        [and join at the unit]
    ∧                                          [both together]

    So: the four survivors are a Boolean algebra, and complementation is subtraction from the
    unit. This is Peirce's resolution of the unit read on this face. -/
theorem idem_compl_partition {σ : ℝ} (hσ : 0 < σ) {z : Cell} (hz : Idempotent σ z) :
    mul σ z (one - z) = 0 ∧ z + (one - z) = one := by
  have hu : upper σ (one - z) = 1 - upper σ z := by
    simp only [upper, one, Prod.fst_sub, Prod.snd_sub]; ring
  have hl : lower σ (one - z) = 1 - lower σ z := by
    simp only [lower, one, Prod.fst_sub, Prod.snd_sub]; ring
  obtain ⟨hbu, hbl⟩ := halves_boolean_of_idempotent hσ hz
  refine ⟨halves_ext hσ ?_ ?_, by ring⟩
  · rw [upper_mul hσ.le, hu, (halves_zero σ).1]
    rcases hbu with h | h <;> rw [h] <;> ring
  · rw [lower_mul hσ.le, hl, (halves_zero σ).2]
    rcases hbl with h | h <;> rw [h] <;> ring

/-- Conjugating swaps what the two channels read.

    (σ : ℝ) (z : Cell)                       [the parameter and a point]
    upper σ (conj z) = lower σ z             [the first reading of the conjugate is the second]
    lower σ (conj z) = upper σ z             [and the other way round]
    ∧                                        [both together] -/
theorem conj_swaps_channels (σ : ℝ) (z : Cell) :
    upper σ (conj z) = lower σ z ∧ lower σ (conj z) = upper σ z := by
  constructor <;> simp only [upper, lower] <;> ring

/-- The channel swap is an order-preserving symmetry of the survivors, and is not
complementation.

    {σ : ℝ} (hσ : 0 < σ)                            [the parameter, above zero]
    ∀ z : Cell                                      [for every point of the algebra]
    Idempotent σ z → Idempotent σ (conj z)          [swapping keeps a survivor one]
    conj (idemPlus σ) = idemMinus σ                 [it exchanges the two extra corners]
    conj one ≠ one - one                            [and it is not complementation]
    ∧                                               [all three together]

    So: the four survivors carry two different symmetries and they must not be conflated. The
    swap `conj` exchanges the two nontrivial corners and fixes the origin and the unit.
    Complementation exchanges the origin and the unit as well. -/
theorem conj_is_swap_not_complement {σ : ℝ} (hσ : 0 < σ) :
    (∀ z : Cell, Idempotent σ z → Idempotent σ (conj z)) ∧
      conj (idemPlus σ) = idemMinus σ ∧ conj one ≠ one - one := by
  refine ⟨fun z hz => ?_, ?_, ?_⟩
  · rw [idempotent_iff_halves hσ, (conj_swaps_channels σ z).1, (conj_swaps_channels σ z).2]
    exact ⟨(idempotent_iff_halves hσ z).mp hz |>.2, (idempotent_iff_halves hσ z).mp hz |>.1⟩
  · simp [conj, idemPlus, idemMinus]
  · intro h
    have := congrArg Prod.fst h
    simp only [one, Prod.fst_sub] at this
    norm_num at this

/-- The two Boolean channels exist exactly above zero.

    (σ : ℝ)                                                    [the parameter]
    ∃ z : Cell                                                 [there is a point]
    Idempotent σ z                                             [surviving squaring]
    upper σ z ≠ lower σ z                                      [whose two readings disagree]
    ∧                                                          [both together]
    ↔                                                          [exactly when]
    0 < σ                                                      [sigma is above zero]

    So: a survivor whose two channels carry independent truth values exists on the hyperbolic
    face and on no other. At or below zero the square root of sigma is zero, so the two readings
    of every point agree and there is one Boolean channel rather than two. This is
    `idempotent_classification` read as a statement about how many wires the face supplies. -/
theorem two_boolean_channels_iff_disc_pos (σ : ℝ) :
    (∃ z : Cell, Idempotent σ z ∧ upper σ z ≠ lower σ z) ↔ 0 < σ := by
  constructor
  · rintro ⟨z, -, hne⟩
    by_cases hσ : 0 < σ
    · exact hσ
    · exact absurd (by simp [upper, lower, Real.sqrt_eq_zero_of_nonpos (not_lt.mp hσ)]) hne
  · intro hσ
    refine ⟨idemPlus σ, idempotent_idemPlus hσ, ?_⟩
    rw [(halves_idemPlus hσ).1, (halves_idemPlus hσ).2]
    norm_num

-- ═══════════════════════════════════════════════════════════════════
-- MIQUEL'S DEFINITION 2.1, AND WHAT ITS APPLICATION COMES OUT TO BE
-- ═══════════════════════════════════════════════════════════════════

/-- An implicative structure: an order with everything meetable, and one implication.

@cite miquel-2018

    (α : Type*) [CompleteLattice α]                        [a carrier where every set has a meet]
    imp : α → α → α                                        [the implication]
    ∀ b b' c, b' ≤ b → imp b c ≤ imp b' c                  [weakening what is assumed raises it]
    ∀ b c c', c ≤ c' → imp b c ≤ imp b c'                  [raising what is concluded raises it]
    ∀ b C, imp b (sInf C) = sInf (imp b '' C)              [it meets what it concludes]

    So: Miquel's Definition 2.1, with the two axioms as fields. Application is not a field: it
    is derived from the order by `implApply` below, which is how this family of structures
    escapes the requirement that application be an algebraic operation. -/
structure Implicative (α : Type*) [CompleteLattice α] where
  /-- The implication of the structure. -/
  imp : α → α → α
  /-- Weakening the assumption raises the implication. -/
  imp_anti : ∀ b b' c : α, b' ≤ b → imp b c ≤ imp b' c
  /-- Raising the conclusion raises the implication. -/
  imp_mono : ∀ b c c' : α, c ≤ c' → imp b c ≤ imp b c'
  /-- The implication carries meets in its conclusion. -/
  imp_sInf : ∀ (b : α) (C : Set α), imp b (sInf C) = sInf (imp b '' C)

/-- Application, derived from the order rather than given.

@cite miquel-2018

    {α : Type*} [CompleteLattice α]           [a carrier where every set has a meet]
    (I : Implicative α)                       [an implicative structure on it]
    (a b : α)                                 [two elements]
    sInf {c | a ≤ I.imp b c}                  [the meet of every conclusion the two support]
    : α                                       [an element]

    So: Miquel's derived application. Nothing forces it to be associative, which is why the
    obstruction recorded for algebraic operations does not apply to the definition itself. -/
def implApply {α : Type*} [CompleteLattice α] (I : Implicative α) (a b : α) : α :=
  sInf {c | a ≤ I.imp b c}

/-- An implication with an adjunction is an implicative structure.

@cite miquel-2018

    {α : Type*} [CompleteLattice α]                      [a carrier where every set has a meet]
    (imp : α → α → α)                                    [a candidate implication]
    (h : ∀ a b c, a ≤ imp b c ↔ a ⊓ b ≤ c)               [it is adjoint to meeting]
    : Implicative α                                      [it satisfies Miquel's two axioms]

    So: every complete Heyting algebra is an implicative structure, which is Miquel's first
    family of examples and the one a forcing tripos comes from. The four survivors of squaring
    above zero are such an algebra, by `idem_mul_is_meet` and `idem_compl_partition`. -/
def Implicative.ofAdjunction {α : Type*} [CompleteLattice α] (imp : α → α → α)
    (h : ∀ a b c : α, a ≤ imp b c ↔ a ⊓ b ≤ c) : Implicative α where
  imp := imp
  imp_anti := by
    intro b b' c hb
    rw [h]
    exact le_trans (inf_le_inf_left _ hb) ((h _ _ _).mp (le_refl _))
  imp_mono := by
    intro b c c' hc
    rw [h]
    exact le_trans ((h _ _ _).mp (le_refl _)) hc
  imp_sInf := by
    intro b C
    refine le_antisymm (le_sInf ?_) ?_
    · rintro x ⟨c, hc, rfl⟩
      rw [h]
      exact le_trans ((h _ _ _).mp (le_refl _)) (sInf_le hc)
    · rw [h]
      refine le_sInf ?_
      intro c hc
      exact le_trans (inf_le_inf_right _ (sInf_le ⟨c, hc, rfl⟩)) ((h _ _ _).mp (le_refl _))

/-- The derived application of such a structure is the meet.

@cite miquel-2018

    {α : Type*} [CompleteLattice α]                          [a carrier with all meets]
    (imp : α → α → α)                                        [a candidate implication]
    (h : ∀ a b c, a ≤ imp b c ↔ a ⊓ b ≤ c)                   [adjoint to meeting]
    (a b : α)                                                [two elements]
    implApply (Implicative.ofAdjunction imp h) a b = a ⊓ b   [applying is meeting]

    So: the set whose meet is taken is everything above `a ⊓ b`, and the meet of that is
    `a ⊓ b`. This is what Miquel's Remark 2.25 is about: on this family the derived
    application is a lattice operation, and the next theorem says a lattice operation is not
    the calculus's application. -/
theorem implApply_ofAdjunction {α : Type*} [CompleteLattice α] (imp : α → α → α)
    (h : ∀ a b c : α, a ≤ imp b c ↔ a ⊓ b ≤ c) (a b : α) :
    implApply (Implicative.ofAdjunction imp h) a b = a ⊓ b := by
  have hset : {c : α | a ≤ imp b c} = {c : α | a ⊓ b ≤ c} := by
    ext c
    exact h a b c
  simp only [implApply, Implicative.ofAdjunction, hset]
  exact le_antisymm (sInf_le (le_refl _)) (le_sInf fun _ hc => hc)

/-- No operation that is commutative and associative discards.

@cite palmieri-2026

    {α : Type*} (op : α → α → α)                        [a binary operation on some carrier]
    (hcomm : ∀ a b, op a b = op b a)                    [order does not matter]
    (hassoc : ∀ a b c, op (op a b) c = op a (op b c))   [bracketing does not matter]
    (k : α) (hk : ∀ x y, op (op k x) y = x)             [k throws its second argument away]
    (x y : α)                                           [any two elements]
    x = y                                               [the carrier is a single point]

    So: `k x y = k y x` by commutativity under the regrouping, so `x = y` for every pair.
    Palmieri's Theorem 3.6 with Theorem 3.5 is the general form, and states the consequence:
    no semigroup, monoid, group or ring carries a classifier alongside a retraction pair. -/
theorem no_K_of_comm_assoc {α : Type*} (op : α → α → α)
    (hcomm : ∀ a b : α, op a b = op b a)
    (hassoc : ∀ a b c : α, op (op a b) c = op a (op b c))
    (k : α) (hk : ∀ x y : α, op (op k x) y = x) (x y : α) : x = y := by
  calc x = op (op k x) y := (hk x y).symm
    _ = op k (op x y) := hassoc k x y
    _ = op k (op y x) := by rw [hcomm x y]
    _ = op (op k y) x := (hassoc k y x).symm
    _ = y := hk y x

/-- At no value of sigma does the meet of the survivors discard.

@cite palmieri-2026
@cite miquel-2018

    (σ : ℝ)                                                  [the parameter, at any value]
    ¬ ∃ k : Cell                                             [there is no point]
    ∀ z w : Cell, Idempotent σ z → Idempotent σ w →          [such that for all survivors]
      mul σ (mul σ k z) w = z                                [it throws the second away]

    So: `idem_mul_is_meet` says that above zero multiplying survivors is the meet of the
    channel order, and `implApply_ofAdjunction` says the derived application of an implicative
    structure built from a Heyting implication is that same meet. A meet is commutative and
    associative, so `no_K_of_comm_assoc` applies. Sigma is quantified rather than
    restricted because the origin and the unit are survivors at every setting, so no face
    supplies a discarding element and the four survivors above zero are no better off than the
    two below. The implicative structure exists on the split face; its application is not the
    calculus's application on any face. -/
theorem idem_no_K (σ : ℝ) :
    ¬ ∃ k : Cell, ∀ z w : Cell, Idempotent σ z → Idempotent σ w →
      mul σ (mul σ k z) w = z := by
  rintro ⟨k, hk⟩
  have h := hk one 0 (idempotent_one σ) (idempotent_zero σ)
  rw [show mul σ (mul σ k one) 0 = 0 by simp [mul]] at h
  have : (0 : Cell).1 = (one : Cell).1 := congrArg Prod.fst h
  norm_num at this

-- ═══════════════════════════════════════════════════════════════════
-- THE TOWER'S NODE RULE, AND WHAT IT ALSO IS NOT
-- ═══════════════════════════════════════════════════════════════════

/-- The Tower's node rule read as a binary operation on plain numbers.

    (σ : ℝ)                                 [the parameter]
    (u v : ℝ)                               [two numbers]
    ratio σ (ofHalves σ u v)                [pack them into the two channels, read the fraction]
    : ℝ                                     [a plain number]

    So: this is what `state_node` does at a branch, isolated from the tree. -/
def nodeOp (σ : ℝ) (u v : ℝ) : ℝ := ratio σ (ofHalves σ u v)

/-- Above zero the node rule is division, and at or below it carries neither input.

    {σ : ℝ}                                                [the parameter]
    (u v : ℝ)                                              [two numbers]
    0 < σ → nodeOp σ u v = u / v                           [above zero it divides them]
    σ ≤ 0 → nodeOp σ u v = 1 ∨ nodeOp σ u v = 0            [at or below it reads one or zero]
    ∧                                                      [both together]

    So: `ratio_ofHalves` and `ratio_ofHalves_of_disc_nonpos`. The node is faithful to its two
    inputs on the hyperbolic face and on no other, which is `node_faithful_iff_disc_pos`. -/
theorem nodeOp_faces (σ : ℝ) (u v : ℝ) :
    (0 < σ → nodeOp σ u v = u / v) ∧
      (σ ≤ 0 → nodeOp σ u v = 1 ∨ nodeOp σ u v = 0) :=
  ⟨fun hσ => ratio_ofHalves hσ u v, fun hσ => ratio_ofHalves_of_disc_nonpos hσ u v⟩

/-- Above zero the node rule has no discarding element either.

@cite rodenburg-2023
@cite palmieri-2026

    {σ : ℝ} (hσ : 0 < σ)                                       [the parameter, above zero]
    ¬ ∃ k : ℝ                                                  [there is no number]
    ∀ u v : ℝ, u ≠ 0 → v ≠ 0 →                                 [such that for all nonzero pairs]
      nodeOp σ (nodeOp σ k u) v = u                            [it throws the second away]

    So: the node rule divides, and `k / u / v = k / (u v) = u` forces `k = u² v`, which is not
    one number. Taking `u = v = 1` gives `k = 1` and `u = 2`, `v = 1` gives `k = 4`. The
    partial reading, with the two arguments required nonzero, is the one Rodenburg's
    characterisation of combinatorially complete partial groupoids asks about, and the
    obstruction survives it. -/
theorem nodeOp_no_K {σ : ℝ} (hσ : 0 < σ) :
    ¬ ∃ k : ℝ, ∀ u v : ℝ, u ≠ 0 → v ≠ 0 → nodeOp σ (nodeOp σ k u) v = u := by
  rintro ⟨k, hk⟩
  have h1 := hk 1 1 one_ne_zero one_ne_zero
  have h2 := hk 2 1 two_ne_zero one_ne_zero
  simp only [nodeOp, ratio_ofHalves hσ] at h1 h2
  norm_num at h1 h2
  rw [h1] at h2
  norm_num at h2

end Trichotomy
end Apeiron

end
