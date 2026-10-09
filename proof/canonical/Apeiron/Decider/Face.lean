import Apeiron.Bridge.Cell
import Apeiron.Trichotomy.Channels
import Apeiron.Trichotomy.SplitComplex

/-!
# The face the decider lives on

@name The Decider's Face
@grade proved

A decider for linear arithmetic over the whole numbers holds one interval per column and
tightens it. An interval is two numbers that can be compared, and the pair has to be readable
one number at a time or the tightening has nothing to act on. This file shows that requirement
picks the algebra out rather than being satisfied by it, and then states the whole of interval
propagation as theorems about `Cell σ` with σ free.

## Two ordered reads exist only above zero

`Channel σ φ` is a read: one number off each point, respecting adding, stretching, multiplying
and the unit. `two_channels_iff_disc_pos` says two distinct reads exist exactly when `σ > 0`, and
`channel_iff_of_disc_pos` says that above zero there are exactly two and they are `upper σ` and
`lower σ`. So an interval representation is available on the hyperbolic face and on neither of
the others: below zero there is no read at all, and at zero there is one, which sees half the
point. `face_is_forced` is those two theorems instantiated at the decider's requirement.

## And the whole numbers pick out one setting on that face

Above zero the coordinates of the point whose readings are `2·hi` and `2·lo` are
`(hi + lo, (hi - lo)/√σ)`. `../song/src/prover/cone/cell.slang` doubles for exactly this reason,
so that the change of basis closes inside the whole numbers in both directions, and
`disc_one_of_integer_basis` shows both directions closing forces `σ = 1`: reading forward makes
`1/√σ` a whole number, reading back makes `√σ` one, and two whole numbers whose product is one
are both one. `Trichotomy.SplitComplex.disc_eq_one` is the same setting reached from the other
side, by running `classification` on the split-complex carrier rather than by choosing it.

## Root

| from | here |
|---|---|
| `two_channels_iff_disc_pos`, `channel_iff_of_disc_pos` | `face_is_forced` |
| `SplitComplex.disc_eq_one` (the carrier's own σ) | `disc_one_of_integer_basis` |
| `ofHalves`, `upper_ofHalves`, `lower_ofHalves` | `intervalOf`, `mem_intervalOf` |
| `inCell_meet_iff`, `upper_meet`, `lower_meet` | `meet_intervalOf`, `mem_meet_intervalOf` |
| `Crossing`, `crossing_independent_of_cone` | `crossing_intervalOf`, `crossing_empty` |
| `conj`, `Bridge.conj_ofBounds` | `conj_intervalOf`, `upper_conj`, `pick_is_conj` |
| `Bridge.ofBounds` | `intervalOf_one` |

Everything above is stated with σ free and `0 < σ`, then instantiated at `σ = 1`, which
is where `Bridge.ofBounds` and the decider live. The propagation and certificate files below
use only the instantiated forms, so deleting any theorem in this table breaks them.
-/

noncomputable section

namespace Apeiron
namespace Decider

open Trichotomy

/-- The point whose two readings are the doubled endpoints of an interval.

    (σ : ℝ)                             [the parameter]
    (lo hi : ℤ)                         [the two endpoints, as whole numbers]
    : Cell                              [the point of the algebra they name]
    ofHalves                            [the point having the two given readings]

    So: `Bridge.ofBounds` with σ left free. The doubling is
    `../song/src/prover/cone/cell.slang`'s, and it is there so the change of basis closes
    inside the whole numbers. -/
def intervalOf (σ : ℝ) (lo hi : ℤ) : Cell := ofHalves σ (2 * (hi : ℝ)) (2 * (lo : ℝ))

/-- The upper reading of an interval's point is twice its top endpoint.

    {σ : ℝ} (hσ : 0 < σ)                [the parameter, above zero]
    (lo hi : ℤ)                         [the two endpoints, as whole numbers]
    upper σ (intervalOf σ lo hi)        [the top read of the interval's point]
    = 2 * (hi : ℝ)                      [is the top endpoint, doubled] -/
theorem upper_intervalOf {σ : ℝ} (hσ : 0 < σ) (lo hi : ℤ) :
    upper σ (intervalOf σ lo hi) = 2 * (hi : ℝ) := upper_ofHalves hσ _ _

/-- The lower reading of an interval's point is twice its bottom endpoint.

    {σ : ℝ} (hσ : 0 < σ)                [the parameter, above zero]
    (lo hi : ℤ)                         [the two endpoints, as whole numbers]
    lower σ (intervalOf σ lo hi)        [the bottom read of the interval's point]
    = 2 * (lo : ℝ)                      [is the bottom endpoint, doubled] -/
theorem lower_intervalOf {σ : ℝ} (hσ : 0 < σ) (lo hi : ℤ) :
    lower σ (intervalOf σ lo hi) = 2 * (lo : ℝ) := lower_ofHalves hσ _ _

/-- At σ = 1 the point is the one song's decider holds.

    (lo hi : ℤ)                         [the two endpoints, as whole numbers]
    intervalOf 1 lo hi                  [the interval's point at σ = 1]
    = Bridge.ofBounds lo hi             [is the point the bridge file names]

    So: the general construction, at the setting the section below forces, is the bridge
    file's `ofBounds`. Everything the decider uses is this instance. -/
theorem intervalOf_one (lo hi : ℤ) : intervalOf 1 lo hi = Bridge.ofBounds lo hi := by
  simp only [intervalOf, ofHalves, Bridge.ofBounds, Real.sqrt_one, Prod.ext_iff]
  constructor <;> · push_cast; ring

/-- An interval holds a whole number exactly when the number is between its endpoints.

    {σ : ℝ} (hσ : 0 < σ)                            [the parameter, above zero]
    (lo hi x : ℤ)                                   [two endpoints and a whole number]
    InCell σ (2 * (x : ℝ)) (intervalOf σ lo hi)     [the point holds the doubled number]
    ↔                                               [exactly when, true in both directions]
    lo ≤ x ∧ x ≤ hi                                 [it sits between the two endpoints]

    So: the bridge from the algebra's membership to the arithmetic every bound in the decider
    is stated with. The doubling cancels on both sides, so it costs nothing. -/
theorem mem_intervalOf {σ : ℝ} (hσ : 0 < σ) (lo hi x : ℤ) :
    InCell σ (2 * (x : ℝ)) (intervalOf σ lo hi) ↔ lo ≤ x ∧ x ≤ hi := by
  simp only [InCell, upper_intervalOf hσ, lower_intervalOf hσ]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨by exact_mod_cast (by linarith : (lo : ℝ) ≤ (x : ℝ)),
      by exact_mod_cast (by linarith : (x : ℝ) ≤ (hi : ℝ))⟩
  · rintro ⟨h1, h2⟩
    have h1' : (lo : ℝ) ≤ (x : ℝ) := by exact_mod_cast h1
    have h2' : (x : ℝ) ≤ (hi : ℝ) := by exact_mod_cast h2
    constructor <;> linarith

/-- Meeting two intervals takes the larger bottom and the smaller top.

    {σ : ℝ} (hσ : 0 < σ)                                     [the parameter, above zero]
    (l₁ h₁ l₂ h₂ : ℤ)                                        [two intervals, by their endpoints]
    meet σ (intervalOf σ l₁ h₁) (intervalOf σ l₂ h₂)         [meeting the two points]
    = intervalOf σ (max l₁ l₂) (min h₁ h₂)                   [is the interval of tighter bounds]

    So: `Bridge.meet_ofBounds` with σ free. This equation is the whole of the
    propagation step: raising the bottom is the maximum on one channel and lowering the top is
    the minimum on the other, and the two together are one operation of the algebra rather
    than two sign cases inside a loop. -/
theorem meet_intervalOf {σ : ℝ} (hσ : 0 < σ) (l₁ h₁ l₂ h₂ : ℤ) :
    meet σ (intervalOf σ l₁ h₁) (intervalOf σ l₂ h₂)
      = intervalOf σ (max l₁ l₂) (min h₁ h₂) := by
  have hu : upper σ (meet σ (intervalOf σ l₁ h₁) (intervalOf σ l₂ h₂))
      = 2 * ((min h₁ h₂ : ℤ) : ℝ) := by
    rw [upper_meet hσ, upper_intervalOf hσ, upper_intervalOf hσ]
    rcases le_total h₁ h₂ with h | h
    · have hr : (h₁ : ℝ) ≤ (h₂ : ℝ) := by exact_mod_cast h
      rw [min_eq_left h, min_eq_left (by linarith : 2 * (h₁ : ℝ) ≤ 2 * (h₂ : ℝ))]
    · have hr : (h₂ : ℝ) ≤ (h₁ : ℝ) := by exact_mod_cast h
      rw [min_eq_right h, min_eq_right (by linarith : 2 * (h₂ : ℝ) ≤ 2 * (h₁ : ℝ))]
  have hl : lower σ (meet σ (intervalOf σ l₁ h₁) (intervalOf σ l₂ h₂))
      = 2 * ((max l₁ l₂ : ℤ) : ℝ) := by
    rw [lower_meet hσ, lower_intervalOf hσ, lower_intervalOf hσ]
    rcases le_total l₁ l₂ with h | h
    · have hr : (l₁ : ℝ) ≤ (l₂ : ℝ) := by exact_mod_cast h
      rw [max_eq_right h, max_eq_right (by linarith : 2 * (l₁ : ℝ) ≤ 2 * (l₂ : ℝ))]
    · have hr : (l₂ : ℝ) ≤ (l₁ : ℝ) := by exact_mod_cast h
      rw [max_eq_left h, max_eq_left (by linarith : 2 * (l₂ : ℝ) ≤ 2 * (l₁ : ℝ))]
  have hrt := ofHalves_upper_lower hσ (meet σ (intervalOf σ l₁ h₁) (intervalOf σ l₂ h₂))
  rw [hu, hl] at hrt
  rw [← hrt, intervalOf]

/-- The tightened interval holds exactly the numbers both intervals held.

    {σ : ℝ} (hσ : 0 < σ)                                  [the parameter, above zero]
    (l₁ h₁ l₂ h₂ x : ℤ)                                   [two intervals and a whole number]
    (max l₁ l₂ ≤ x ∧ x ≤ min h₁ h₂)                       [the tighter interval holds it]
    ↔                                                     [exactly when, true both directions]
    (l₁ ≤ x ∧ x ≤ h₁) ∧ (l₂ ≤ x ∧ x ≤ h₂)                 [both intervals hold it]

    So: `inCell_meet_iff` transported to the whole numbers through `mem_intervalOf` and
    `meet_intervalOf`. Left to right is why propagation never loses a solution, which is the
    entire soundness of the first phase; right to left is why it never keeps a non-solution,
    which is what makes the operation the meet rather than some other common bound. -/
theorem mem_meet_intervalOf {σ : ℝ} (hσ : 0 < σ) (l₁ h₁ l₂ h₂ x : ℤ) :
    (max l₁ l₂ ≤ x ∧ x ≤ min h₁ h₂) ↔ (l₁ ≤ x ∧ x ≤ h₁) ∧ (l₂ ≤ x ∧ x ≤ h₂) := by
  rw [← mem_intervalOf hσ, ← mem_intervalOf hσ, ← mem_intervalOf hσ, ← meet_intervalOf hσ]
  exact inCell_meet_iff hσ _ _ _

/-- Conjugating an interval swaps its endpoints.

    {σ : ℝ}                                     [the parameter]
    (lo hi : ℤ)                                 [the two endpoints, as whole numbers]
    conj (intervalOf σ lo hi)                   [the conjugate of the interval's point]
    = intervalOf σ hi lo                        [is the interval with its endpoints exchanged]

    So: `Bridge.conj_ofBounds` with σ free. The propagator's sign test is this
    involution: a positive coefficient reads the interval as it stands and a negative one
    reads its conjugate. -/
theorem conj_intervalOf {σ : ℝ} (lo hi : ℤ) :
    conj (intervalOf σ lo hi) = intervalOf σ hi lo := by
  simp only [conj, intervalOf, ofHalves, Prod.ext_iff]
  constructor <;> ring

/-- The upper reading of a conjugate is the lower reading of the point.

    {σ : ℝ}                                     [the parameter]
    (z : Cell)                                  [a point of the algebra]
    upper σ (conj z)                            [the top read of the conjugated point]
    = lower σ z                                 [is the bottom read of the point itself]

    So: what makes the sign test in `propPick` an operation of the algebra instead of a case
    split. σ is free, so this is a fact about conjugation on every face. -/
theorem upper_conj {σ : ℝ} (z : Cell) : upper σ (conj z) = lower σ z := by
  simp only [upper, lower]
  ring

/-- Which endpoint a coefficient's sign reads, as one conjugation.

    {σ : ℝ} (hσ : 0 < σ)                        [the parameter, above zero]
    (a lo hi : ℤ)                               [a coefficient and an interval]
    upper σ (intervalOf σ lo hi)                [a positive coefficient reads the top]
    upper σ (conj (intervalOf σ lo hi))         [a negative one reads the conjugate's top]
    = 2 * ((if 0 < a then hi else lo : ℤ) : ℝ)  [which is the top, or the bottom, doubled]

    So: `../song/src/prover/cone/ncone0.slang`'s `propPick`, which reads `getHi` for a
    positive coefficient and `getLo` for a negative one, is one read of the algebra applied
    either to the point or to its conjugate. `prop_pick_upper` and `prop_pick_lower` are the
    two arms of this equation on the running decider. -/
theorem pick_is_conj {σ : ℝ} (hσ : 0 < σ) (a lo hi : ℤ) :
    (if 0 < a then upper σ (intervalOf σ lo hi)
      else upper σ (conj (intervalOf σ lo hi)))
      = 2 * ((if 0 < a then hi else lo : ℤ) : ℝ) := by
  by_cases ha : 0 < a
  · rw [if_pos ha, if_pos ha, upper_intervalOf hσ]
  · rw [if_neg ha, if_neg ha, upper_conj, lower_intervalOf hσ]

/-- An interval is a crossing exactly when its top has fallen below its bottom.

    {σ : ℝ} (hσ : 0 < σ)                        [the parameter, above zero]
    (lo hi : ℤ)                                 [the two endpoints, as whole numbers]
    Crossing σ (intervalOf σ lo hi)             [the interval's point is a crossing]
    ↔ hi < lo                                   [exactly when the endpoints have passed]

    So: `Bridge.crossing_ofBounds` with σ free. This is the order, and
    `crossing_independent_of_cone` says it is not the null cone: the decider's verdict rides
    the order and the algebra's degeneracy rides the cone, and neither predicts the other. -/
theorem crossing_intervalOf {σ : ℝ} (hσ : 0 < σ) (lo hi : ℤ) :
    Crossing σ (intervalOf σ lo hi) ↔ hi < lo := by
  simp only [Crossing, upper_intervalOf hσ, lower_intervalOf hσ]
  constructor
  · intro h
    exact_mod_cast (by linarith : (hi : ℝ) < (lo : ℝ))
  · intro h
    have hc : (hi : ℝ) < (lo : ℝ) := by exact_mod_cast h
    linarith

/-- A crossing holds no whole number at all.

    {σ : ℝ} (hσ : 0 < σ)                              [the parameter, above zero]
    (lo hi x : ℤ)                                     [an interval and a whole number]
    Crossing σ (intervalOf σ lo hi)                   [the interval's point is a crossing]
    ¬ InCell σ (2 * (x : ℝ)) (intervalOf σ lo hi)     [so the point holds nothing]

    So: the step that turns a crossing into a verdict. Every UNSAT this decider reports by
    propagation is this theorem applied once, at the column whose interval crossed. -/
theorem crossing_empty {σ : ℝ} (hσ : 0 < σ) (lo hi x : ℤ)
    (h : Crossing σ (intervalOf σ lo hi)) :
    ¬ InCell σ (2 * (x : ℝ)) (intervalOf σ lo hi) := by
  intro hmem
  rw [mem_intervalOf hσ] at hmem
  rw [crossing_intervalOf hσ] at h
  omega

/-- Two distinct ordered reads exist exactly on the hyperbolic face, and there they are two.

    (σ : ℝ)                                               [the parameter]
    ∃ φ ψ : Cell → ℝ, Channel σ φ ∧ Channel σ ψ ∧ φ ≠ ψ   [there are two different reads]
    ↔ 0 < σ                                               [exactly when σ is above zero]
    ∀ φ : Cell → ℝ                                        [and for every map to plain numbers]
    (∀ z : Cell, φ z = upper σ z)                         [it is the top read]
    ∨ (∀ z : Cell, φ z = lower σ z)                       [or it is the bottom read]

    So: the interval representation is not a modelling choice. A pair of endpoints that can be
    read and compared one at a time is a pair of channels, `two_channels_iff_disc_pos` says a
    second channel exists only above zero, and `channel_iff_of_disc_pos` says there is no
    third. Below zero the decider has no representation at all and at zero it has half of one.
    Which setting above zero is then fixed by `disc_one_of_integer_basis`. -/
theorem face_is_forced (σ : ℝ) :
    ((∃ φ ψ : Cell → ℝ, Channel σ φ ∧ Channel σ ψ ∧ φ ≠ ψ) ↔ 0 < σ) ∧
      ∀ φ : Cell → ℝ, 0 < σ → Channel σ φ →
        (∀ z : Cell, φ z = upper σ z) ∨ (∀ z : Cell, φ z = lower σ z) :=
  ⟨two_channels_iff_disc_pos σ, fun φ hσ hφ => (channel_iff_of_disc_pos hσ φ).mp hφ⟩

/-- The coordinates of an interval's point, with σ free.

    {σ : ℝ} (hσ : 0 < σ)                            [the parameter, above zero]
    (lo hi : ℤ)                                     [the two endpoints, as whole numbers]
    intervalOf σ lo hi                              [the interval's point]
    = ((hi : ℝ) + lo, ((hi : ℝ) - lo) / Real.sqrt σ)   [its two coordinates]

    So: the sum of the endpoints, and their difference divided by the root of σ. The
    second coordinate is where the whole numbers get their say. -/
theorem intervalOf_coords {σ : ℝ} (hσ : 0 < σ) (lo hi : ℤ) :
    intervalOf σ lo hi = ((hi : ℝ) + lo, ((hi : ℝ) - lo) / Real.sqrt σ) := by
  have hs : Real.sqrt σ ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hσ)
  simp only [intervalOf, ofHalves, Prod.ext_iff]
  constructor
  · ring
  · field_simp

/-- Closing the change of basis inside the whole numbers forces σ to one.

    {σ : ℝ} (hσ : 0 < σ)                            [the parameter, above zero]
    ∀ lo hi : ℤ                                     [for every pair of endpoints]
    ∃ m r : ℤ                                       [there are two whole coordinates]
    ∀ m r : ℤ                                       [and for every pair of whole coordinates]
    ∃ lo hi : ℤ                                     [there is an interval with them]
    intervalOf σ lo hi = ((m : ℝ), (r : ℝ))         [the point and its coordinates]
    σ = 1                                           [then σ = 1]

    So: `../song/src/prover/cone/cell.slang` doubles its coordinates so that "the inverse
    closes inside the integers", and closing in both directions is what the words mean.
    Reading forward at the interval from zero to one makes `1/√σ` a whole number; reading back
    at the coordinate pair `(0, 1)` makes `√σ` one. Two whole numbers whose product is one and
    which are not negative are both one. `Trichotomy.SplitComplex.disc_eq_one` reaches the same
    setting from the other side, by running the classification on the split-complex carrier
    instead of choosing it, so the decider's face is settled twice over and chosen neither
    time. -/
theorem disc_one_of_integer_basis {σ : ℝ} (hσ : 0 < σ)
    (hinto : ∀ lo hi : ℤ, ∃ m r : ℤ, intervalOf σ lo hi = ((m : ℝ), (r : ℝ)))
    (honto : ∀ m r : ℤ, ∃ lo hi : ℤ, intervalOf σ lo hi = ((m : ℝ), (r : ℝ))) :
    σ = 1 := by
  have hspos : 0 < Real.sqrt σ := Real.sqrt_pos.mpr hσ
  have hsne : Real.sqrt σ ≠ 0 := ne_of_gt hspos
  obtain ⟨m₀, r₀, h₀⟩ := hinto 0 1
  rw [intervalOf_coords hσ, Prod.ext_iff] at h₀
  have hr₀ : ((r₀ : ℤ) : ℝ) * Real.sqrt σ = 1 := by
    have h := h₀.2
    push_cast at h
    field_simp at h
    linarith [h]
  obtain ⟨lo, hi, h₁⟩ := honto 0 1
  rw [intervalOf_coords hσ, Prod.ext_iff] at h₁
  have hd : (hi : ℝ) - (lo : ℝ) = Real.sqrt σ := by
    have h := h₁.2
    push_cast at h
    field_simp at h
    linarith [h]
  have hprod : ((hi - lo : ℤ) : ℝ) * ((r₀ : ℤ) : ℝ) = 1 := by
    push_cast
    rw [hd]
    linarith [hr₀]
  have hprodz : (hi - lo) * r₀ = 1 := by exact_mod_cast hprod
  have hpos : (0 : ℝ) < ((hi - lo : ℤ) : ℝ) := by push_cast; rw [hd]; exact hspos
  have hposz : (0 : ℤ) ≤ hi - lo := by exact_mod_cast hpos.le
  have hone : hi - lo = 1 := Int.eq_one_of_mul_eq_one_right hposz hprodz
  have hs1 : Real.sqrt σ = 1 := by
    rw [← hd, ← Int.cast_sub, hone, Int.cast_one]
  have hsq := Real.sq_sqrt hσ.le
  rw [hs1] at hsq
  linarith [hsq]

end Decider
end Apeiron

end
