import Apeiron.Classification.Margin
import Mathlib.Tactic

/-!
# Scale invariance of split-face labels

The selected ratio is unchanged by multiplying both slots of a cell by a nonzero scalar. Ratio
bands and finite labels therefore preserve their value under that scale. A cut made from the
absolute upper read does not have this invariance.

## Root

`AnyRing.ringUpper_smul` and `AnyRing.ringLower_smul` provide the two readback identities.
Their common scale factor cancels in `ringRatio`; the cancellation requires a nonzero scale.
-/

namespace Apeiron
namespace Classification
namespace ScaleInvariant

open Trichotomy AnyRing Margin Finset

section Ordered

variable {K : Type*} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- A selected ratio is invariant under a nonzero common scale.

    F : SplitFace K [a parameter and positive root]
    a : K [the common scale]
    z : RingCell K [the observed cell]
    a ≠ 0 [the scale is nonzero]
    ringRatio F (a • z) = ringRatio F z [the selected ratio is unchanged]
    `RingCell` [a point: two ring elements written side by side]
    `ringRatio` [the upper read divided by the lower read]
    `≠` [is not equal to]
    `=` [is equal to]
    `•` [stretch by, a number times a point] -/
theorem ringRatio_scale_invariant (F : SplitFace K) (a : K) (z : RingCell K) (ha : a ≠ 0) :
    ringRatio F (a • z) = ringRatio F z := by
  unfold ringRatio
  rw [ringUpper_smul, ringLower_smul]
  exact mul_div_mul_left _ _ ha

/-- A ratio band is invariant under a nonzero common scale of its reference cell.

    F : SplitFace K [a parameter and positive root]
    epsilon a : K [the band radius and common scale]
    z : RingCell K [the reference cell]
    a ≠ 0 [the scale is nonzero]
    ratioBand F epsilon (a • z) = ratioBand F epsilon z [the reconstructed band is unchanged]
    `RingCell` [a point: two ring elements written side by side]
    `ratioBand` [a cell from ratio plus and minus an error bound]
    `≠` [is not equal to]
    `=` [is equal to]
    `•` [stretch by, a number times a point] -/
theorem ratioBand_scale_invariant (F : SplitFace K) (epsilon a : K) (z : RingCell K)
    (ha : a ≠ 0) : ratioBand F epsilon (a • z) = ratioBand F epsilon z := by
  unfold ratioBand
  rw [ringRatio_scale_invariant F a z ha]

/-- A ratio cut is invariant under a nonzero common scale of its observation.

    F : SplitFace K [a parameter and positive root]
    theta z : RingCell K [a cut cell and observed cell]
    a : K [the common scale]
    a ≠ 0 [the scale is nonzero]
    cut F theta (a • z) = cut F theta z [the executable cut is unchanged]
    `RingCell` [a point: two ring elements written side by side]
    `≠` [is not equal to]
    `=` [is equal to]
    `•` [stretch by, a number times a point] -/
theorem cut_scale_invariant (F : SplitFace K) (theta z : RingCell K) (a : K) (ha : a ≠ 0) :
    cut F theta (a • z) = cut F theta z := by
  unfold cut
  rw [ringRatio_scale_invariant F a z ha]

/-- An axis label is invariant under a nonzero common scale of its observation.

    F : SplitFace K [a parameter and positive root]
    theta : Fin m → RingCell K [the cut cells]
    z : RingCell K [the observed cell]
    a : K [the common scale]
    a ≠ 0 [the scale is nonzero]
    axisCell F theta (a • z) = axisCell F theta z [the executable axis label is unchanged]
    `RingCell` [a point: two ring elements written side by side]
    `=` [is equal to]
    `•` [stretch by, a number times a point] -/
theorem axisCell_scale_invariant {m : Nat} (F : SplitFace K) (theta : Fin m → RingCell K)
    (z : RingCell K) (a : K) (ha : a ≠ 0) : axisCell F theta (a • z) = axisCell F theta z := by
  unfold axisCell
  congr 1
  apply Finset.filter_congr
  intro i _
  rw [cut_scale_invariant F (theta i) z a ha]

/-- A box label is invariant when every observed cell has the same nonzero scale.

    F : SplitFace K [a parameter and positive root]
    theta : (i : Fin d) → Fin (m i) → RingCell K [the cut cells for each coordinate]
    z : Fin d → RingCell K [the observed cells]
    a : K [the common scale]
    a ≠ 0 [the scale is nonzero]
    boxLabel F theta (fun i => a • z i) = boxLabel F theta z [the executable box label is unchanged]
    `RingCell` [a point: two ring elements written side by side]
    `=` [is equal to]
    `→` [if the left holds, the right holds]
    `≠` [is not equal to]
    `•` [stretch by, a number times a point] -/
theorem boxLabel_scale_invariant {d : Nat} {m : Fin d → Nat} (F : SplitFace K)
    (theta : (i : Fin d) → Fin (m i) → RingCell K) (z : Fin d → RingCell K)
    (a : K) (ha : a ≠ 0) :
    boxLabel F theta (fun i => a • z i) = boxLabel F theta z := by
  funext i
  exact axisCell_scale_invariant F (theta i) (z i) a ha

/-- The executable cut made from an absolute upper threshold cell.

    F : SplitFace K [a parameter and positive root]
    threshold observation : RingCell K [the threshold cell and observed cell]
    absoluteUpperCut F threshold observation [whether threshold upper is at most observation upper]
    `Bool` [true or false, nothing else]
    `RingCell` [a point: two ring elements written side by side]
    `ringUpper` [the first root coordinate]
    `≤` [is smaller than or equal to] -/
def absoluteUpperCut (F : SplitFace K) (threshold observation : RingCell K) : Bool :=
  decide (ringUpper F.root threshold ≤ ringUpper F.root observation)

/-- A positive upper threshold can change under a positive contraction.

    F : SplitFace K [a parameter and positive root]
    t : K [the positive threshold]
    z : RingCell K [the observed cell]
    0 < t [the threshold is positive]
    t < ringUpper F.root z [the observation begins above the threshold]
    ∃ a : K, 0 < a ∧ a < 1 ∧
      absoluteUpperCut F (ofChannels F t 1) (a • z) = false ∧
      absoluteUpperCut F (ofChannels F t 1) z = true [a contraction moves the absolute cut]
    `∃` [there is at least one]
    `∧` [both are true]
    `<` [is smaller than]
    `=` [is equal to]
    `•` [stretch by, a number times a point] -/
theorem absolute_upper_threshold_moves (F : SplitFace K) (t : K) (z : RingCell K)
    (ht0 : 0 < t) (ht : t < ringUpper F.root z) :
    ∃ a : K, 0 < a ∧ a < 1 ∧
      absoluteUpperCut F (ofChannels F t 1) (a • z) = false ∧
      absoluteUpperCut F (ofChannels F t 1) z = true := by
  have hu : 0 < ringUpper F.root z := lt_trans ht0 ht
  have htwo : 0 < (2 : K) * ringUpper F.root z := mul_pos (by norm_num) hu
  let a : K := t / ((2 : K) * ringUpper F.root z)
  refine ⟨a, div_pos ht0 htwo, (div_lt_one htwo).mpr (by nlinarith), ?_, ?_⟩
  · unfold absoluteUpperCut
    rw [ringUpper_ofChannels, ringUpper_smul]
    have ha : a * ringUpper F.root z = t / 2 := by
      dsimp [a]
      field_simp
    simp [show ¬ t ≤ a * ringUpper F.root z by rw [ha]; linarith]
  · unfold absoluteUpperCut
    rw [ringUpper_ofChannels]
    simp [le_of_lt ht]

/-- Zero scaling changes a concrete rational selected ratio.

    ringRatio Margin.rationalFace ((0 : Rat) • ofChannels Margin.rationalFace 2 1)
      ≠ ringRatio Margin.rationalFace (ofChannels Margin.rationalFace 2 1)
                                                       [zero scaling changes the ratio]
    `RingCell` [a point: two ring elements written side by side]
    `ringRatio` [the upper read divided by the lower read]
    `≠` [is not equal to]
    `•` [stretch by, a number times a point] -/
theorem zero_scale_changes_ratio :
    ringRatio rationalFace ((0 : Rat) • ofChannels rationalFace 2 1) ≠
      ringRatio rationalFace (ofChannels rationalFace 2 1) := by
  norm_num [rationalFace, ringRatio, ofChannels, ringUpper, ringLower, ringOfHalves]

#guard boxLabel rationalFace (fun _ : Fin 1 => fun _ : Fin 1 => ofChannels rationalFace 0 1)
  (fun _ : Fin 1 => (2 : Rat) • ofChannels rationalFace 1 1) =
  boxLabel rationalFace (fun _ : Fin 1 => fun _ : Fin 1 => ofChannels rationalFace 0 1)
    (fun _ : Fin 1 => ofChannels rationalFace 1 1)

end Ordered
end ScaleInvariant
end Classification
end Apeiron
