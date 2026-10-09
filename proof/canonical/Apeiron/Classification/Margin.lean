import Apeiron.Trichotomy.ChannelsAnyRing
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

/-!
# Executable split-face margin classifier

The classifier reads `AnyRing.RingCell K` through a supplied positive root of its parameter.
All score, cut, and box operations reduce over an ordered field with decidable comparisons. The
theorems state sufficient stability conditions and equality-boundary witnesses only.

## Root

`AnyRing.ringUpper_channel` and `AnyRing.ringLower_channel` make the supplied root reads of
`ringMul F.sigma` channels. `SplitFace.root_sq` is the premise of those theorems. The interval
and classifier layer below uses those reads. Labels have no domain interpretation.
-/

namespace Apeiron
namespace Classification
namespace Margin

open Trichotomy AnyRing Finset

/-- A positive selected root of an algebra parameter.

    K [an ordered field]
    sigma root : K [the parameter and its selected positive root]
    root_pos : 0 < root [the root is positive]
    root_sq : root ^ 2 = sigma [the root squares to the parameter]
    `≤` [is smaller than or equal to]
    `<` [is smaller than] -/
structure SplitFace (K : Type*) [Field K] [LinearOrder K] [IsStrictOrderedRing K] where
  sigma : K
  root : K
  root_pos : 0 < root
  root_sq : root ^ 2 = sigma

section Ordered

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The selected upper read is a multiplicative channel for the split face.

    F : SplitFace K [a parameter and positive root]
    RingChannel F.sigma (ringUpper F.root) [the upper read respects ring operations]
    `RingChannel` [a read that respects addition, stretching, multiplication, and one]
    `ringUpper` [the first root coordinate] -/
theorem upper_channel (F : SplitFace K) : RingChannel F.sigma (ringUpper F.root) :=
  ringUpper_channel F.root_sq

/-- The selected lower read is a multiplicative channel for the split face.

    F : SplitFace K [a parameter and positive root]
    RingChannel F.sigma (ringLower F.root) [the lower read respects ring operations]
    `RingChannel` [a read that respects addition, stretching, multiplication, and one]
    `ringLower` [the second root coordinate] -/
theorem lower_channel (F : SplitFace K) : RingChannel F.sigma (ringLower F.root) :=
  ringLower_channel F.root_sq

/-- The ratio of the selected upper and lower reads.

    F : SplitFace K [a parameter and positive root]
    z : RingCell K [a two-slot point]
    ringRatio F z [the upper read divided by the lower read]
    `RingCell` [a point: two ring elements written side by side]
    `ringUpper` [the first root coordinate]
    `ringLower` [the second root coordinate] -/
def ringRatio (F : SplitFace K) (z : RingCell K) : K :=
  ringUpper F.root z / ringLower F.root z

/-- A cell reconstructed from its two selected reads.

    F : SplitFace K [a parameter and positive root]
    u v : K [the requested upper and lower reads]
    ofChannels F u v [the reconstructed ring cell]
    `RingCell` [a point: two ring elements written side by side]
    `ringOfHalves` [the ring point having the two given coordinates] -/
def ofChannels (F : SplitFace K) (u v : K) : RingCell K :=
  ringOfHalves u v F.root⁻¹ (2 : K)⁻¹

/-- The upper read of `ofChannels` is its requested upper value.

    F : SplitFace K [a parameter and positive root]
    u v : K [the requested reads]
    ringUpper F.root (ofChannels F u v) = u [the upper read is reconstructed]
    `ringUpper` [the first root coordinate]
    `ringOfHalves` [the ring point having the two given coordinates] -/
theorem ringUpper_ofChannels (F : SplitFace K) (u v : K) :
    ringUpper F.root (ofChannels F u v) = u := by
  apply ringUpper_ringOfHalves
  · exact mul_inv_cancel₀ (ne_of_gt F.root_pos)
  · norm_num

/-- The lower read of `ofChannels` is its requested lower value.

    F : SplitFace K [a parameter and positive root]
    u v : K [the requested reads]
    ringLower F.root (ofChannels F u v) = v [the lower read is reconstructed]
    `ringLower` [the second root coordinate]
    `ringOfHalves` [the ring point having the two given coordinates] -/
theorem ringLower_ofChannels (F : SplitFace K) (u v : K) :
    ringLower F.root (ofChannels F u v) = v := by
  apply ringLower_ringOfHalves
  · exact mul_inv_cancel₀ (ne_of_gt F.root_pos)
  · norm_num

/-- A scalar lies between a cell's selected lower and upper reads.

    F : SplitFace K [a parameter and positive root]
    t : K [the scalar under test]
    z : RingCell K [the bounding cell]
    RingInCell F t z [the scalar is between the two reads]
    `RingCell` [a point: two ring elements written side by side]
    `∧` [both are true]
    `≤` [is smaller than or equal to]
    `ringLower` [the second root coordinate]
    `ringUpper` [the first root coordinate] -/
def RingInCell (F : SplitFace K) (t : K) (z : RingCell K) : Prop :=
  ringLower F.root z ≤ t ∧ t ≤ ringUpper F.root z

/-- The cell containing the intersection of two selected-read intervals.

    F : SplitFace K [a parameter and positive root]
    z w : RingCell K [two bounding cells]
    ringMeet F z w [the cell with the minimum upper and maximum lower reads]
    `RingCell` [a point: two ring elements written side by side] -/
def ringMeet (F : SplitFace K) (z w : RingCell K) : RingCell K :=
  ofChannels F (min (ringUpper F.root z) (ringUpper F.root w))
    (max (ringLower F.root z) (ringLower F.root w))

/-- Membership in a selected-read meet is membership in both input cells.

    F : SplitFace K [a parameter and positive root]
    t : K [the scalar under test]
    z w : RingCell K [two bounding cells]
    RingInCell F t (ringMeet F z w) ↔ RingInCell F t z ∧ RingInCell F t w [the meet is both]
    `RingInCell` [the scalar is between the two reads]
    `∧` [both are true]
    `↔` [exactly when, true in both directions] -/
theorem ringInCell_meet_iff (F : SplitFace K) (t : K) (z w : RingCell K) :
    RingInCell F t (ringMeet F z w) ↔ RingInCell F t z ∧ RingInCell F t w := by
  unfold ringMeet
  rw [RingInCell, ringLower_ofChannels, ringUpper_ofChannels]
  constructor
  · rintro ⟨hl, hu⟩
    exact ⟨⟨le_trans (le_max_left _ _) hl, le_trans hu (min_le_left _ _)⟩,
      ⟨le_trans (le_max_right _ _) hl, le_trans hu (min_le_right _ _)⟩⟩
  · rintro ⟨⟨hlz, huz⟩, ⟨hlw, huw⟩⟩
    exact ⟨max_le hlz hlw, le_min huz huw⟩

/-- A ratio interval centred at a cell's selected ratio.

    F : SplitFace K [a parameter and positive root]
    epsilon : K [the ratio error bound]
    z : RingCell K [the reference cell]
    ratioBand F epsilon z [the cell from ratio plus epsilon and ratio minus epsilon]
    `RingCell` [a point: two ring elements written side by side] -/
def ratioBand (F : SplitFace K) (epsilon : K) (z : RingCell K) : RingCell K :=
  ofChannels F (ringRatio F z + epsilon) (ringRatio F z - epsilon)

/-- Membership in a ratio band is an absolute ratio-error bound.

    F : SplitFace K [a parameter and positive root]
    epsilon t : K [the error bound and scalar under test]
    z : RingCell K [the reference cell]
    RingInCell F t (ratioBand F epsilon z) ↔ |t - ringRatio F z| ≤ epsilon [the ratio is bounded]
    `RingInCell` [the scalar is between the two reads]
    `RingCell` [a point: two ring elements written side by side]
    `↔` [exactly when, true in both directions]
    `≤` [is smaller than or equal to] -/
theorem ringInCell_ratioBand_iff (F : SplitFace K) (epsilon t : K) (z : RingCell K) :
    RingInCell F t (ratioBand F epsilon z) ↔ |t - ringRatio F z| ≤ epsilon := by
  unfold ratioBand
  rw [RingInCell, ringLower_ofChannels, ringUpper_ofChannels]
  constructor
  · rintro ⟨hl, hu⟩
    rw [abs_le]
    constructor <;> linarith
  · intro h
    rw [abs_le] at h
    constructor <;> linarith

/-- Membership in two ratio bands is membership in their meet.

    F : SplitFace K [a parameter and positive root]
    t epsilon delta : K [a scalar and two error bounds]
    z w : RingCell K [two reference cells]
    RingInCell F t (ringMeet F (ratioBand F epsilon z) (ratioBand F delta w))
      ↔ |t - ringRatio F z| ≤ epsilon ∧ |t - ringRatio F w| ≤ delta [both bounds hold]
    `RingInCell` [the scalar is between the two reads]
    `RingCell` [a point: two ring elements written side by side]
    `∧` [both are true]
    `↔` [exactly when, true in both directions]
    `≤` [is smaller than or equal to] -/
theorem ringInCell_meet_ratioBand_iff (F : SplitFace K) (t epsilon delta : K)
    (z w : RingCell K) :
    RingInCell F t (ringMeet F (ratioBand F epsilon z) (ratioBand F delta w)) ↔
      |t - ringRatio F z| ≤ epsilon ∧ |t - ringRatio F w| ≤ delta := by
  rw [ringInCell_meet_iff, ringInCell_ratioBand_iff, ringInCell_ratioBand_iff]

/-- A channel-reconstructed cell with ratio `q`.

    F : SplitFace K [a parameter and positive root]
    q : K [the requested ratio]
    ringRatio F (ofChannels F q 1) = q [the constructed cell reads back to q]
    `ringRatio` [the upper read divided by the lower read]
    `ringOfHalves` [the ring point having the two given coordinates] -/
theorem ringRatio_ofChannels_one (F : SplitFace K) (q : K) :
    ringRatio F (ofChannels F q 1) = q := by
  rw [ringRatio, ringUpper_ofChannels, ringLower_ofChannels, div_one]

/-- A strict winner under selected cell ratios.

    F : SplitFace K [a parameter and positive root]
    r : Fin k → RingCell K [one cell score for each label]
    w : Fin k [the proposed winner]
    IsWinner F r w [every distinct label has a lower ratio]
    `RingCell` [a point: two ring elements written side by side]
    `∀` [for every single one]
    `→` [if the left holds, the right holds]
    `<` [is smaller than]
    `≠` [is not equal to] -/
def IsWinner {k : Nat} (F : SplitFace K) (r : Fin k → RingCell K) (w : Fin k) : Prop :=
  ∀ j, j ≠ w → ringRatio F (r j) < ringRatio F (r w)

/-- A winner survives cell perturbations constrained by its ratio bands.

    F : SplitFace K [a parameter and positive root]
    r s : Fin k → RingCell K [reference and perturbed cell scores]
    w : Fin k [the reference winner]
    epsilon : K [the ratio error bound]
    hpert : ∀ i, RingInCell F (ringRatio F (s i)) (ratioBand F epsilon (r i))
                                                       [each ratio is in band]
    hmargin : ∀ j, j ≠ w → 2 * epsilon < ringRatio F (r w) - ringRatio F (r j)
                                                       [each gap clears twice]
    IsWinner F s w [the perturbed winner remains strict]
    `RingInCell` [the scalar is between the two reads]
    `RingCell` [a point: two ring elements written side by side]
    `∀` [for every single one]
    `→` [if the left holds, the right holds]
    `<` [is smaller than]
    `≠` [is not equal to]
    `j` [a finite-label rival]
    -/
theorem margin_preserves_label {k : Nat} (F : SplitFace K) (r s : Fin k → RingCell K)
    (w : Fin k) (epsilon : K)
    (hpert : ∀ i, RingInCell F (ringRatio F (s i)) (ratioBand F epsilon (r i)))
    (hmargin : ∀ j, j ≠ w → 2 * epsilon < ringRatio F (r w) - ringRatio F (r j)) :
    IsWinner F s w := by
  intro j hj
  have hw := (ringInCell_ratioBand_iff F epsilon (ringRatio F (s w)) (r w)).mp (hpert w)
  have hjj := (ringInCell_ratioBand_iff F epsilon (ringRatio F (s j)) (r j)).mp (hpert j)
  rw [abs_le] at hw hjj
  linarith [hmargin j hj]

/-- Equality at twice epsilon admits cell perturbations that lose strict winning order.

    F : SplitFace K [a parameter and positive root]
    r : Fin k → RingCell K [reference cell scores]
    w j : Fin k [a winner and a distinct rival]
    epsilon : K [the nonnegative ratio error bound]
    ∃ s : Fin k → RingCell K, (∀ i, RingInCell F (ringRatio F (s i))
      (ratioBand F epsilon (r i))) ∧ ¬ IsWinner F s w [a bounded perturbation loses strictness]
    `∃` [there is at least one]
    `∀` [for every single one]
    `∧` [both are true]
    `¬` [this is not true]
    `RingInCell` [the scalar is between the two reads]
    `RingCell` [a point: two ring elements written side by side]
    `≤` [is smaller than or equal to]
    `≠` [is not equal to] -/
theorem margin_sharp {k : Nat} (F : SplitFace K) (r : Fin k → RingCell K) (w j : Fin k)
    (epsilon : K) (hepsilon : 0 ≤ epsilon) (hj : j ≠ w)
    (hgap : ringRatio F (r w) - ringRatio F (r j) = 2 * epsilon) :
    ∃ s : Fin k → RingCell K,
      (∀ i, RingInCell F (ringRatio F (s i)) (ratioBand F epsilon (r i))) ∧ ¬ IsWinner F s w := by
  let s : Fin k → RingCell K := Function.update
    (fun i => ofChannels F (ringRatio F (r i) + epsilon) 1)
    w (ofChannels F (ringRatio F (r w) - epsilon) 1)
  refine ⟨s, ?_, ?_⟩
  · intro i
    by_cases hi : i = w
    · subst i
      rw [show s w = ofChannels F (ringRatio F (r w) - epsilon) 1 by simp [s],
        ringInCell_ratioBand_iff, ringRatio_ofChannels_one]
      rw [show ringRatio F (r w) - epsilon - ringRatio F (r w) = -epsilon by ring,
        abs_neg, abs_of_nonneg hepsilon]
    · rw [show s i = ofChannels F (ringRatio F (r i) + epsilon) 1 by simp [s, hi],
        ringInCell_ratioBand_iff, ringRatio_ofChannels_one]
      rw [show ringRatio F (r i) + epsilon - ringRatio F (r i) = epsilon by ring,
        abs_of_nonneg hepsilon]
  · intro hs
    have hsj := hs j hj
    rw [show s j = ofChannels F (ringRatio F (r j) + epsilon) 1 by simp [s, hj],
      show s w = ofChannels F (ringRatio F (r w) - epsilon) 1 by simp [s],
      ringRatio_ofChannels_one, ringRatio_ofChannels_one] at hsj
    linarith

/-- The executable side of a cut at one cell ratio.

    F : SplitFace K [a parameter and positive root]
    theta x : RingCell K [a cut cell and a measured cell]
    cut F theta x [true exactly when the cut ratio is at most the measured ratio]
    `Bool` [true or false, nothing else]
    `RingCell` [a point: two ring elements written side by side]
    `ringRatio` [the upper read divided by the lower read]
    `≤` [is smaller than or equal to] -/
def cut (F : SplitFace K) (theta x : RingCell K) : Bool :=
  decide (ringRatio F theta ≤ ringRatio F x)

/-- The count of cuts whose executable side is true.

    F : SplitFace K [a parameter and positive root]
    theta : Fin m → RingCell K [one cut cell for each finite cut]
    x : RingCell K [the measured cell]
    axisCell F theta x [the number of satisfied cut sides]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `RingCell` [a point: two ring elements written side by side] -/
def axisCell {m : Nat} (F : SplitFace K) (theta : Fin m → RingCell K) (x : RingCell K) : Nat :=
  (Finset.univ.filter fun i => cut F (theta i) x).card

/-- Every cut ratio is farther than epsilon from a cell ratio.

    F : SplitFace K [a parameter and positive root]
    epsilon : K [the required clearance]
    theta : Fin m → RingCell K [the cut cells]
    x : RingCell K [the measured cell]
    ClearsCuts F epsilon theta x [every cut clears epsilon]
    `RingCell` [a point: two ring elements written side by side]
    `∀` [for every single one]
    `<` [is smaller than] -/
def ClearsCuts {m : Nat} (F : SplitFace K) (epsilon : K) (theta : Fin m → RingCell K)
    (x : RingCell K) : Prop :=
  ∀ i, epsilon < |ringRatio F x - ringRatio F (theta i)|

/-- A cleared cut retains its executable side under a ratio-band perturbation.

    F : SplitFace K [a parameter and positive root]
    epsilon : K [the error bound]
    theta : Fin m → RingCell K [the cut cells]
    x y : RingCell K [the reference and perturbed cells]
    i : Fin m [one cut]
    cut F (theta i) y = cut F (theta i) x [the cut side is unchanged]
    `RingInCell` [the scalar is between the two reads]
    `RingCell` [a point: two ring elements written side by side]
    `=` [is equal to] -/
theorem cut_side_stable {m : Nat} (F : SplitFace K) (epsilon : K) (theta : Fin m → RingCell K)
    (x y : RingCell K) (hclear : ClearsCuts F epsilon theta x)
    (hpert : RingInCell F (ringRatio F y) (ratioBand F epsilon x)) (i : Fin m) :
    cut F (theta i) y = cut F (theta i) x := by
  unfold cut
  apply decide_eq_decide.mpr
  have hp := (ringInCell_ratioBand_iff F epsilon (ringRatio F y) x).mp hpert
  have hc := hclear i
  rw [abs_le] at hp
  constructor
  · intro h
    by_contra hx
    push Not at hx
    rw [abs_of_neg (by linarith : ringRatio F x - ringRatio F (theta i) < 0)] at hc
    linarith [hp.2]
  · intro h
    rw [abs_of_nonneg (by linarith : 0 ≤ ringRatio F x - ringRatio F (theta i))] at hc
    linarith [hp.1]

/-- A cleared axis keeps its executable cut count under a ratio-band perturbation.

    F : SplitFace K [a parameter and positive root]
    epsilon : K [the error bound]
    theta : Fin m → RingCell K [the cut cells]
    x y : RingCell K [the reference and perturbed cells]
    axisCell F theta y = axisCell F theta x [the cut count is unchanged]
    `RingCell` [a point: two ring elements written side by side]
    `=` [is equal to] -/
theorem axisCell_stable {m : Nat} (F : SplitFace K) (epsilon : K) (theta : Fin m → RingCell K)
    (x y : RingCell K) (hclear : ClearsCuts F epsilon theta x)
    (hpert : RingInCell F (ringRatio F y) (ratioBand F epsilon x)) :
    axisCell F theta y = axisCell F theta x := by
  unfold axisCell
  congr 1
  apply Finset.filter_congr
  intro i _
  rw [cut_side_stable F epsilon theta x y hclear hpert i]

/-- Equality one epsilon below a cut changes its executable side at the boundary.

    F : SplitFace K [a parameter and positive root]
    theta : RingCell K [the cut cell]
    epsilon : K [a positive error bound]
    ∃ x y : RingCell K, RingInCell F (ringRatio F y) (ratioBand F epsilon x)
      ∧ axisCell F (fun _ : Fin 1 => theta) y ≠ axisCell F (fun _ : Fin 1 => theta) x
                                                       [a changed axis]
    `∃` [there is at least one]
    `∧` [both are true]
    `≠` [is not equal to]
    `<` [is smaller than]
    `RingInCell` [the scalar is between the two reads]
    `RingCell` [a point: two ring elements written side by side] -/
theorem axisCell_sharp (F : SplitFace K) (theta : RingCell K) (epsilon : K)
    (hepsilon : 0 < epsilon) :
    ∃ x y : RingCell K, RingInCell F (ringRatio F y) (ratioBand F epsilon x) ∧
      axisCell F (fun _ : Fin 1 => theta) y ≠ axisCell F (fun _ : Fin 1 => theta) x := by
  let x := ofChannels F (ringRatio F theta - epsilon) 1
  let y := ofChannels F (ringRatio F theta) 1
  refine ⟨x, y, ?_, ?_⟩
  · have hyRatio : ringRatio F y = ringRatio F theta := by
      simp [y, ringRatio_ofChannels_one]
    have hxRatio : ringRatio F x = ringRatio F theta - epsilon := by
      simp [x, ringRatio_ofChannels_one]
    rw [ringInCell_ratioBand_iff, hyRatio, hxRatio,
      show ringRatio F theta - (ringRatio F theta - epsilon) = epsilon by ring,
      abs_of_pos hepsilon]
  · have hy : cut F theta y = true := by
      simp [cut, y, ringRatio_ofChannels_one]
    have hx : cut F theta x = false := by
      have h : ¬ ringRatio F theta ≤ ringRatio F theta - epsilon := by linarith
      simp [cut, x, ringRatio_ofChannels_one, h]
    simp [axisCell, hy, hx]

/-- The coordinatewise executable label of a box partition.

    F : SplitFace K [a parameter and positive root]
    theta : (a : Fin d) → Fin (m a) → RingCell K [the cut cells for each coordinate]
    x : Fin d → RingCell K [one measured cell per coordinate]
    boxLabel F theta x [the cut count for every coordinate]
    `RingCell` [a point: two ring elements written side by side]
    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `→` [if the left holds, the right holds] -/
def boxLabel {d : Nat} {m : Fin d → Nat} (F : SplitFace K)
    (theta : (a : Fin d) → Fin (m a) → RingCell K) (x : Fin d → RingCell K) : Fin d → Nat :=
  fun a => axisCell F (theta a) (x a)

/-- A box label is stable when every coordinate clears every cut and stays in its ratio band.

    F : SplitFace K [a parameter and positive root]
    epsilon : K [the common error bound]
    theta : (a : Fin d) → Fin (m a) → RingCell K [the cut cells]
    x y : Fin d → RingCell K [reference and perturbed cells]
    boxLabel F theta y = boxLabel F theta x [the coordinatewise label is unchanged]
    `RingCell` [a point: two ring elements written side by side]
    `=` [is equal to]
    `∀` [for every single one] -/
theorem boxLabel_stable {d : Nat} {m : Fin d → Nat} (F : SplitFace K) (epsilon : K)
    (theta : (a : Fin d) → Fin (m a) → RingCell K) (x y : Fin d → RingCell K)
    (hclear : ∀ a, ClearsCuts F epsilon (theta a) (x a))
    (hpert : ∀ a, RingInCell F (ringRatio F (y a)) (ratioBand F epsilon (x a))) :
    boxLabel F theta y = boxLabel F theta x := by
  funext a
  exact axisCell_stable F epsilon (theta a) (x a) (y a) (hclear a) (hpert a)

/-- The rational split face at sigma and root equal to one. -/
def rationalFace : SplitFace Rat :=
  { sigma := 1, root := 1, root_pos := by norm_num, root_sq := by norm_num }

/-- A joint rational fixture whose axis count reduces to one. -/
example :
    axisCell rationalFace (fun _ : Fin 1 => ofChannels rationalFace 0 1)
      (ofChannels rationalFace 1 1) = 1 := by
  norm_num [rationalFace, axisCell, cut, ringRatio, ofChannels, ringUpper, ringLower,
    ringOfHalves]

#guard axisCell rationalFace (fun _ : Fin 1 => ofChannels rationalFace 0 1)
  (ofChannels rationalFace 1 1) = 1

end Ordered
end Margin
end Classification
end Apeiron
