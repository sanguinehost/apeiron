import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Numeric.Scale
import Numeric.Rope.Part2

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Fold-constant displacement

The fold subtracts one rounded constant per level: `PI_Q32` at the first level, `HALF_PI_Q32` at the
second. Each is within half a Q32.32 unit of the exact value, so the folded argument is displaced by
at most `trig_fold_delta = 1/2^32` (`2 * (1/2)/2^32`). The integer branch thresholds are the floors
of the exact thresholds (`6746518852 = floor(π/2·2^32)`, `3373259426 = floor(π/4·2^32)`), so the
integer and exact branches coincide. The composed displacement and the `sin`/`cos` transfer are the
next slice; the three constant defects below are exact. -/

/-- `PI_Q32 = 13493037705` is within half a Q32.32 unit of `π`. -/
theorem pi_q32_defect : |(13493037705 : ℝ) / q32 - Real.pi| ≤ 1 / (2 * q32) := by
  have h1 : (3.14159265358979323846 : ℝ) < Real.pi := Real.pi_gt_d20
  have h2 : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  unfold q32
  rw [abs_le]
  constructor <;> nlinarith

/-- `HALF_PI_Q32 = 6746518852` is within half a Q32.32 unit of `π/2`. -/
theorem half_pi_q32_defect : |(6746518852 : ℝ) / q32 - Real.pi / 2| ≤ 1 / (2 * q32) := by
  have h1 : (3.14159265358979323846 : ℝ) < Real.pi := Real.pi_gt_d20
  have h2 : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  unfold q32
  rw [abs_le]
  constructor <;> nlinarith

/-- `QUARTER_PI_Q32 = 3373259426` is within half a Q32.32 unit of `π/4`. -/
theorem quarter_pi_q32_defect : |(3373259426 : ℝ) / q32 - Real.pi / 4| ≤ 1 / (2 * q32) := by
  have h1 : (3.14159265358979323846 : ℝ) < Real.pi := Real.pi_gt_d20
  have h2 : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  unfold q32
  rw [abs_le]
  constructor <;> nlinarith

/-- Fold displacement budget `1/2^32` (one rounded constant per fold level). -/
def trig_fold_delta : ℝ := 1 / q32


/-- Exact-threshold `sin`-fold on `ℝ`, for the displacement comparison. -/
def fold1T (a : ℝ) : ℝ :=
  if Real.pi / 2 < a then a - Real.pi else if a < -(Real.pi / 2) then a + Real.pi else a

/-- Exact-threshold quadrant fold on `ℝ`, for the displacement comparison. -/
def fold2T (a : ℝ) : ℝ :=
  if Real.pi / 4 < a then Real.pi / 2 - a
  else if a < -(Real.pi / 4) then a + Real.pi / 2 else a

/-- `6746518852 = floor(π/2·2^32)`: the integer and exact first-fold thresholds coincide. -/
theorem half_pi_q32_floor : (6746518852 : ℝ) / q32 < Real.pi / 2 ∧
    Real.pi / 2 < (6746518853 : ℝ) / q32 := by
  have h1 : (3.14159265358979323846 : ℝ) < Real.pi := Real.pi_gt_d20
  have h2 : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  unfold q32
  constructor <;> nlinarith

/-- `3373259426 = floor(π/4·2^32)`: the integer and exact second-fold thresholds coincide. -/
theorem quarter_pi_q32_floor : (3373259426 : ℝ) / q32 < Real.pi / 4 ∧
    Real.pi / 4 < (3373259427 : ℝ) / q32 := by
  have h1 : (3.14159265358979323846 : ℝ) < Real.pi := Real.pi_gt_d20
  have h2 : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  unfold q32
  constructor <;> nlinarith

/-- First-fold displacement: the integer `sin`-fold differs from the exact real fold by at most
half a Q32.32 unit. -/
theorem fold1_displacement (r : I64) :
    |(((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 - fold1T (toReal r)| ≤ 1 / (2 * q32) := by
  obtain ⟨hp1, hp2⟩ := half_pi_q32_floor
  unfold fixed_trig.fold1 fold1T toReal
  by_cases h1 : (6746518852 : ℤ) < r.val
  · have hge : (6746518853 : ℝ) ≤ (r.val : ℝ) := by
      exact_mod_cast (show (6746518853 : ℤ) ≤ r.val by omega)
    have hthis : (6746518853 : ℝ) / q32 ≤ (r.val : ℝ) / q32 :=
      div_le_div_of_nonneg_right hge (le_of_lt q32_pos)
    have hr : Real.pi / 2 < (r.val : ℝ) / q32 := lt_of_lt_of_le hp2 hthis
    simp only [if_pos h1, if_pos hr]
    rw [show (((r.val - 13493037705 : ℤ) : ℝ)) / q32 - ((r.val : ℝ) / q32 - Real.pi)
        = Real.pi - (13493037705 : ℝ) / q32 by push_cast; ring]
    rw [abs_sub_comm]; exact pi_q32_defect
  · by_cases h2 : r.val < (-6746518852 : ℤ)
    · have hle : (r.val : ℝ) ≤ (-(6746518853 : ℝ)) := by
        exact_mod_cast (show r.val ≤ (-6746518853 : ℤ) by omega)
      have hthis : (r.val : ℝ) / q32 ≤ (-(6746518853 : ℝ)) / q32 :=
        div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
      have hm : (-(6746518853 : ℝ)) / q32 < -(Real.pi / 2) := by
        simpa [neg_div] using neg_lt_neg hp2
      have hr : (r.val : ℝ) / q32 < -(Real.pi / 2) := lt_of_le_of_lt hthis hm
      have hr0 : ¬ (Real.pi / 2 < (r.val : ℝ) / q32) := by
        refine not_lt.mpr (le_of_lt (lt_of_lt_of_le hr ?_))
        linarith [Real.pi_pos]
      simp only [if_neg h1, if_pos h2]
      rw [if_neg hr0, if_pos hr]
      rw [show (((r.val + 13493037705 : ℤ) : ℝ)) / q32 - ((r.val : ℝ) / q32 + Real.pi)
          = (13493037705 : ℝ) / q32 - Real.pi by push_cast; ring]
      exact pi_q32_defect
    · have hle : (r.val : ℝ) ≤ 6746518852 := by
        exact_mod_cast (show r.val ≤ (6746518852 : ℤ) by omega)
      have hthis : (r.val : ℝ) / q32 ≤ 6746518852 / q32 :=
        div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
      have hr : ¬ (Real.pi / 2 < (r.val : ℝ) / q32) :=
        not_lt.mpr (le_of_lt (lt_of_le_of_lt hthis hp1))
      have hle2 : (-6746518852 : ℝ) ≤ (r.val : ℝ) := by
        exact_mod_cast (show (-6746518852 : ℤ) ≤ r.val by omega)
      have hthis2 : (-6746518852 : ℝ) / q32 ≤ (r.val : ℝ) / q32 :=
        div_le_div_of_nonneg_right hle2 (le_of_lt q32_pos)
      have hm : -(Real.pi / 2) < (-6746518852 : ℝ) / q32 := by
        simpa [neg_div] using neg_lt_neg hp1
      have hr2 : ¬ ((r.val : ℝ) / q32 < -(Real.pi / 2)) :=
        not_lt.mpr (le_of_lt (lt_of_lt_of_le hm hthis2))
      simp only [if_neg h1, if_neg h2, if_neg hr, if_neg hr2]
      rw [sub_self, abs_zero]; unfold q32; norm_num

/-- Second-fold (quadrant) displacement: the integer quadrant fold differs from the exact real
quadrant fold by at most half a Q32.32 unit. -/
theorem fold2_displacement (s : I64) :
    |(((fixed_trig.fold2 s.val : ℤ) : ℝ)) / q32 - fold2T (toReal s)| ≤ 1 / (2 * q32) := by
  obtain ⟨hq1, hq2⟩ := quarter_pi_q32_floor
  unfold fixed_trig.fold2 fold2T toReal
  by_cases h1 : (3373259426 : ℤ) < s.val
  · have hge : (3373259427 : ℝ) ≤ (s.val : ℝ) := by
      exact_mod_cast (show (3373259427 : ℤ) ≤ s.val by omega)
    have hthis : (3373259427 : ℝ) / q32 ≤ (s.val : ℝ) / q32 :=
      div_le_div_of_nonneg_right hge (le_of_lt q32_pos)
    have hr : Real.pi / 4 < (s.val : ℝ) / q32 := lt_of_lt_of_le hq2 hthis
    simp only [if_pos h1, if_pos hr]
    rw [show (((6746518852 - s.val : ℤ) : ℝ)) / q32 - (Real.pi / 2 - (s.val : ℝ) / q32)
        = (6746518852 : ℝ) / q32 - Real.pi / 2 by push_cast; ring]
    exact half_pi_q32_defect
  · by_cases h2 : s.val < (-3373259426 : ℤ)
    · have hle : (s.val : ℝ) ≤ (-(3373259427 : ℝ)) := by
        exact_mod_cast (show s.val ≤ (-3373259427 : ℤ) by omega)
      have hthis : (s.val : ℝ) / q32 ≤ (-(3373259427 : ℝ)) / q32 :=
        div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
      have hm : (-(3373259427 : ℝ)) / q32 < -(Real.pi / 4) := by
        simpa [neg_div] using neg_lt_neg hq2
      have hr : (s.val : ℝ) / q32 < -(Real.pi / 4) := lt_of_le_of_lt hthis hm
      have hr0 : ¬ (Real.pi / 4 < (s.val : ℝ) / q32) := by
        refine not_lt.mpr (le_of_lt (lt_of_lt_of_le hr ?_))
        linarith [Real.pi_pos]
      simp only [if_neg h1, if_pos h2]
      rw [if_neg hr0, if_pos hr]
      rw [show (((s.val + 6746518852 : ℤ) : ℝ)) / q32 - ((s.val : ℝ) / q32 + Real.pi / 2)
          = (6746518852 : ℝ) / q32 - Real.pi / 2 by push_cast; ring]
      exact half_pi_q32_defect
    · have hle : (s.val : ℝ) ≤ 3373259426 := by
        exact_mod_cast (show s.val ≤ (3373259426 : ℤ) by omega)
      have hthis : (s.val : ℝ) / q32 ≤ 3373259426 / q32 :=
        div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
      have hr : ¬ (Real.pi / 4 < (s.val : ℝ) / q32) :=
        not_lt.mpr (le_of_lt (lt_of_le_of_lt hthis hq1))
      have hle2 : (-3373259426 : ℝ) ≤ (s.val : ℝ) := by
        exact_mod_cast (show (-3373259426 : ℤ) ≤ s.val by omega)
      have hthis2 : (-3373259426 : ℝ) / q32 ≤ (s.val : ℝ) / q32 :=
        div_le_div_of_nonneg_right hle2 (le_of_lt q32_pos)
      have hm : -(Real.pi / 4) < (-3373259426 : ℝ) / q32 := by
        simpa [neg_div] using neg_lt_neg hq1
      have hr2 : ¬ ((s.val : ℝ) / q32 < -(Real.pi / 4)) :=
        not_lt.mpr (le_of_lt (lt_of_lt_of_le hm hthis2))
      simp only [if_neg h1, if_neg h2, if_neg hr, if_neg hr2]
      rw [sub_self, abs_zero]; unfold q32; norm_num


/-- Exact phase identity: `sin` of the first fold, with the recorded sign flip. -/
theorem sin_fold1T (a : ℝ) :
    Real.sin (fold1T a) =
      (if Real.pi / 2 < a then -Real.sin a
       else if a < -(Real.pi / 2) then -Real.sin a else Real.sin a) := by
  unfold fold1T
  by_cases h1 : Real.pi / 2 < a
  · simp [h1, Real.sin_sub_pi]
  · by_cases h2 : a < -(Real.pi / 2)
    · simp [h1, h2, Real.sin_add_pi]
    · simp [h1, h2]

/-- Exact phase identity: `cos` of the first fold, with the recorded sign flip. -/
theorem cos_fold1T (a : ℝ) :
    Real.cos (fold1T a) =
      (if Real.pi / 2 < a then -Real.cos a
       else if a < -(Real.pi / 2) then -Real.cos a else Real.cos a) := by
  unfold fold1T
  by_cases h1 : Real.pi / 2 < a
  · simp [h1, Real.cos_sub_pi]
  · by_cases h2 : a < -(Real.pi / 2)
    · simp [h1, h2, Real.cos_add_pi]
    · simp [h1, h2]

/-- Exact phase identity: `sin` of the quadrant fold, with the recorded `sin`/`cos` swap. -/
theorem sin_fold2T (a : ℝ) :
    Real.sin (fold2T a) =
      (if Real.pi / 4 < a then Real.cos a
       else if a < -(Real.pi / 4) then Real.cos a else Real.sin a) := by
  unfold fold2T
  by_cases h1 : Real.pi / 4 < a
  · simp [h1, Real.sin_pi_div_two_sub]
  · by_cases h2 : a < -(Real.pi / 4)
    · simp [h1, h2, Real.sin_add_pi_div_two]
    · simp [h1, h2]

/-- Exact phase identity: `cos` of the quadrant fold, with the recorded `sin`/`cos` swap. -/
theorem cos_fold2T (a : ℝ) :
    Real.cos (fold2T a) =
      (if Real.pi / 4 < a then Real.sin a
       else if a < -(Real.pi / 4) then -Real.sin a else Real.cos a) := by
  unfold fold2T
  by_cases h1 : Real.pi / 4 < a
  · simp [h1, Real.cos_pi_div_two_sub]
  · by_cases h2 : a < -(Real.pi / 4)
    · simp [h1, h2, Real.cos_add_pi_div_two]
    · simp [h1, h2]


/-- Phase selection for the `sin` output: the `θ`-fold `negate` flag (first level) and `swap` flag
(second level) select `±sin`/`±cos` of the exact folded angle `fold2T (fold1T θ)`. Together with
`sin_fold2T`/`cos_fold2T` this is the flag algebra the extracted `word_fixed_sin_cos_value` performs. -/
theorem sin_phase_select (θ : ℝ) :
    Real.sin θ =
      (if Real.pi / 2 < θ then -1 else if θ < -(Real.pi / 2) then -1 else 1) *
        (if Real.pi / 4 < fold1T θ then Real.cos (fold2T (fold1T θ))
         else if fold1T θ < -(Real.pi / 4) then -Real.cos (fold2T (fold1T θ))
         else Real.sin (fold2T (fold1T θ))) := by
  rw [sin_fold2T (fold1T θ), cos_fold2T (fold1T θ), sin_fold1T θ, cos_fold1T θ]
  split_ifs <;> ring

/-- Phase selection for the `cos` output, dual to `sin_phase_select`. -/
theorem cos_phase_select (θ : ℝ) :
    Real.cos θ =
      (if Real.pi / 2 < θ then -1 else if θ < -(Real.pi / 2) then -1 else 1) *
        (if Real.pi / 4 < fold1T θ then Real.sin (fold2T (fold1T θ))
         else if fold1T θ < -(Real.pi / 4) then Real.sin (fold2T (fold1T θ))
         else Real.cos (fold2T (fold1T θ))) := by
  rw [sin_fold2T (fold1T θ), cos_fold2T (fold1T θ), sin_fold1T θ, cos_fold1T θ]
  split_ifs <;> ring


/-- The extracted flagged `sin` output as a real function of the folded angle `Φ` (swaps with `cos`
for `swap = 1`/`2`, sign flips for `negate`). -/
def flagSin (negate : Bool) (swap : ℤ) (Φ : ℝ) : ℝ :=
  (if negate then -1 else 1) *
    (if swap = 1 then Real.cos Φ else if swap = 2 then -Real.cos Φ else Real.sin Φ)

/-- The extracted flagged `cos` output as a real function of the folded angle `Φ`. -/
def flagCos (negate : Bool) (swap : ℤ) (Φ : ℝ) : ℝ :=
  (if negate then -1 else 1) *
    (if swap = 1 then Real.sin Φ else if swap = 2 then Real.sin Φ else Real.cos Φ)

/-- Pre-Taylor bound for the flagged `sin` output: the extracted `swap`/`negate` flags applied to
`sin`/`cos` of the folded angle are within `trig_fold_delta` of `sin (toReal r)`. The `negate` factor
absorbs the first-fold jump; the `swap` cases absorb the quadrant jump (continuity of the flagged
combination). -/
theorem flag_sin_fold_bound (r : I64) :
    |flagSin (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
        (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.sin (toReal r)| ≤
      trig_fold_delta := by
  obtain ⟨hp1, hp2⟩ := half_pi_q32_floor
  obtain ⟨hq1, hq2⟩ := quarter_pi_q32_floor
  have h1disp : |(((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 - fold1T (toReal r)| ≤ 1 / (2 * q32) :=
    fold1_displacement r
  have hsig : (if fixed_trig.fold1_negate r.val then (-1 : ℝ) else 1) *
      Real.sin (fold1T (toReal r)) = Real.sin (toReal r) := by
    rw [sin_fold1T]
    by_cases h1 : (6746518852 : ℤ) < r.val
    · have hra : Real.pi / 2 < toReal r := by
        have hge : (6746518853 : ℝ) ≤ (r.val : ℝ) := by
          exact_mod_cast (show (6746518853 : ℤ) ≤ r.val by omega)
        have := div_le_div_of_nonneg_right hge (le_of_lt q32_pos)
        exact lt_of_lt_of_le hp2 this
      have hneg : fixed_trig.fold1_negate r.val = true := by simp [fixed_trig.fold1_negate, h1]
      simp [hneg, hra]
    · by_cases h2 : r.val < (-6746518852 : ℤ)
      · have hra : toReal r < -(Real.pi / 2) := by
          have hle : (r.val : ℝ) ≤ (-(6746518853 : ℝ)) := by
            exact_mod_cast (show r.val ≤ (-6746518853 : ℤ) by omega)
          have := div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
          have hm : (-(6746518853 : ℝ)) / q32 < -(Real.pi / 2) := by
            simpa [neg_div] using neg_lt_neg hp2
          exact lt_of_le_of_lt this hm
        have hra2 : ¬ (Real.pi / 2 < toReal r) := by
          refine not_lt.mpr (le_of_lt (lt_of_lt_of_le hra ?_)); linarith [Real.pi_pos]
        have hneg : fixed_trig.fold1_negate r.val = true := by
          simp [fixed_trig.fold1_negate, h1, h2]
        simp [hneg, hra2, hra]
      · have hle : (r.val : ℝ) ≤ 6746518852 := by
          exact_mod_cast (show r.val ≤ (6746518852 : ℤ) by omega)
        have hthis : (r.val : ℝ) / q32 ≤ 6746518852 / q32 :=
          div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
        have hra : ¬ (Real.pi / 2 < toReal r) :=
          not_lt.mpr (le_of_lt (lt_of_le_of_lt hthis hp1))
        have hle2 : (-6746518852 : ℝ) ≤ (r.val : ℝ) := by
          exact_mod_cast (show (-6746518852 : ℤ) ≤ r.val by omega)
        have hthis2 : (-6746518852 : ℝ) / q32 ≤ (r.val : ℝ) / q32 :=
          div_le_div_of_nonneg_right hle2 (le_of_lt q32_pos)
        have hm : -(Real.pi / 2) < (-6746518852 : ℝ) / q32 := by
          simpa [neg_div] using neg_lt_neg hp1
        have hra2 : ¬ (toReal r < -(Real.pi / 2)) :=
          not_lt.mpr (le_of_lt (lt_of_lt_of_le hm hthis2))
        have hneg : fixed_trig.fold1_negate r.val = false := by
          simp [fixed_trig.fold1_negate, h1, h2]
        simp [hneg, hra, hra2]
  have h2b : |(if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          -Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
      Real.sin ((((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32)| ≤ 1 / (2 * q32) := by
    by_cases ht1 : (3373259426 : ℤ) < fixed_trig.fold1 r.val
    · have hswap : fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 := by
        simp [fixed_trig.fold2_swap, ht1]
      have harg : fixed_trig.fold2 (fixed_trig.fold1 r.val) =
          6746518852 - fixed_trig.fold1 r.val := by simp [fixed_trig.fold2, ht1]
      rw [hswap, harg]
      rw [show (((6746518852 - fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 =
          (6746518852 : ℝ) / q32 - (((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 by push_cast; ring]
      set X : ℝ := (((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 with hX
      calc |Real.cos ((6746518852 : ℝ) / q32 - X) - Real.sin X|
          = |Real.cos ((6746518852 : ℝ) / q32 - X) - Real.cos (Real.pi / 2 - X)| := by
              rw [Real.cos_pi_div_two_sub]
        _ ≤ |((6746518852 : ℝ) / q32 - X) - (Real.pi / 2 - X)| := Real.abs_cos_sub_cos_le _ _
        _ = |(6746518852 : ℝ) / q32 - Real.pi / 2| := by ring_nf
        _ ≤ 1 / (2 * q32) := half_pi_q32_defect
    · by_cases ht2 : fixed_trig.fold1 r.val < (-3373259426 : ℤ)
      · have hswap : fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 := by
          simp [fixed_trig.fold2_swap, ht1, ht2]
        have harg : fixed_trig.fold2 (fixed_trig.fold1 r.val) =
            fixed_trig.fold1 r.val + 6746518852 := by simp [fixed_trig.fold2, ht1, ht2]
        rw [hswap, harg]
        rw [show (((fixed_trig.fold1 r.val + 6746518852 : ℤ) : ℝ)) / q32 =
            (((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 + (6746518852 : ℝ) / q32 by push_cast; ring]
        set X : ℝ := (((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 with hX
        set A : ℝ := X + (6746518852 : ℝ) / q32 with hA
        calc |(-Real.cos A) - Real.sin X|
            = |Real.cos A + Real.sin X| := by
                rw [show (-Real.cos A) - Real.sin X = -(Real.cos A + Real.sin X) by ring, abs_neg]
          _ = |Real.cos A - Real.cos (X + Real.pi / 2)| := by
                rw [Real.cos_add_pi_div_two]
                congr 1
                ring
          _ ≤ |A - (X + Real.pi / 2)| := Real.abs_cos_sub_cos_le _ _
          _ = |(6746518852 : ℝ) / q32 - Real.pi / 2| := by rw [hA]; ring_nf
          _ ≤ 1 / (2 * q32) := half_pi_q32_defect
      · have hswap : fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 0 := by
          simp [fixed_trig.fold2_swap, ht1, ht2]
        have harg : fixed_trig.fold2 (fixed_trig.fold1 r.val) = fixed_trig.fold1 r.val := by
          simp [fixed_trig.fold2, ht1, ht2]
        rw [hswap, harg]
        rw [if_neg (by norm_num : ¬ ((0 : ℤ) = 1)), if_neg (by norm_num : ¬ ((0 : ℤ) = 2)),
            sub_self, abs_zero]
        have hq : (0 : ℝ) < q32 := q32_pos
        positivity
  have htri : |(if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          -Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
      Real.sin (fold1T (toReal r))| ≤ 1 / (2 * q32) + 1 / (2 * q32) := by
    have hs2 : |Real.sin ((((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32) -
        Real.sin (fold1T (toReal r))| ≤
        |(((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 - fold1T (toReal r)| :=
      Real.abs_sin_sub_sin_le _ _
    calc |(if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          -Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
        Real.sin (fold1T (toReal r))|
        ≤ |(if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
         else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          -Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
         else Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
          Real.sin ((((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32)| +
          |Real.sin ((((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32) - Real.sin (fold1T (toReal r))| :=
            abs_sub_le _ _ _
      _ ≤ 1 / (2 * q32) + 1 / (2 * q32) := add_le_add h2b (le_trans hs2 h1disp)
  have hσ : |(if fixed_trig.fold1_negate r.val then (-1 : ℝ) else 1)| = 1 := by
    by_cases h : fixed_trig.fold1_negate r.val <;> simp [h]
  have hkey : flagSin (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
        (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.sin (toReal r)
      = (if fixed_trig.fold1_negate r.val then (-1 : ℝ) else 1) *
        ((if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
         else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          -Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
         else Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
          Real.sin (fold1T (toReal r))) := by
    rw [flagSin, ← hsig]; ring
  rw [hkey, abs_mul, hσ, one_mul]
  rw [show trig_fold_delta = 1 / q32 from rfl]
  exact le_trans htri (le_of_eq (by ring))

/-- Pre-Taylor bound for the flagged `cos` output: the extracted `swap`/`negate` flags applied to
`sin`/`cos` of the folded angle are within `trig_fold_delta` of `cos (toReal r)`. -/
theorem flag_cos_fold_bound (r : I64) :
    |flagCos (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
        (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.cos (toReal r)| ≤
      trig_fold_delta := by
  obtain ⟨hp1, hp2⟩ := half_pi_q32_floor
  obtain ⟨hq1, hq2⟩ := quarter_pi_q32_floor
  have h1disp : |(((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 - fold1T (toReal r)| ≤ 1 / (2 * q32) :=
    fold1_displacement r
  have hsig : (if fixed_trig.fold1_negate r.val then (-1 : ℝ) else 1) *
      Real.cos (fold1T (toReal r)) = Real.cos (toReal r) := by
    rw [cos_fold1T]
    by_cases h1 : (6746518852 : ℤ) < r.val
    · have hra : Real.pi / 2 < toReal r := by
        have hge : (6746518853 : ℝ) ≤ (r.val : ℝ) := by
          exact_mod_cast (show (6746518853 : ℤ) ≤ r.val by omega)
        have := div_le_div_of_nonneg_right hge (le_of_lt q32_pos)
        exact lt_of_lt_of_le hp2 this
      have hneg : fixed_trig.fold1_negate r.val = true := by simp [fixed_trig.fold1_negate, h1]
      simp [hneg, hra]
    · by_cases h2 : r.val < (-6746518852 : ℤ)
      · have hra : toReal r < -(Real.pi / 2) := by
          have hle : (r.val : ℝ) ≤ (-(6746518853 : ℝ)) := by
            exact_mod_cast (show r.val ≤ (-6746518853 : ℤ) by omega)
          have := div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
          have hm : (-(6746518853 : ℝ)) / q32 < -(Real.pi / 2) := by
            simpa [neg_div] using neg_lt_neg hp2
          exact lt_of_le_of_lt this hm
        have hra2 : ¬ (Real.pi / 2 < toReal r) := by
          refine not_lt.mpr (le_of_lt (lt_of_lt_of_le hra ?_)); linarith [Real.pi_pos]
        have hneg : fixed_trig.fold1_negate r.val = true := by
          simp [fixed_trig.fold1_negate, h1, h2]
        simp [hneg, hra2, hra]
      · have hle : (r.val : ℝ) ≤ 6746518852 := by
          exact_mod_cast (show r.val ≤ (6746518852 : ℤ) by omega)
        have hthis : (r.val : ℝ) / q32 ≤ 6746518852 / q32 :=
          div_le_div_of_nonneg_right hle (le_of_lt q32_pos)
        have hra : ¬ (Real.pi / 2 < toReal r) :=
          not_lt.mpr (le_of_lt (lt_of_le_of_lt hthis hp1))
        have hle2 : (-6746518852 : ℝ) ≤ (r.val : ℝ) := by
          exact_mod_cast (show (-6746518852 : ℤ) ≤ r.val by omega)
        have hthis2 : (-6746518852 : ℝ) / q32 ≤ (r.val : ℝ) / q32 :=
          div_le_div_of_nonneg_right hle2 (le_of_lt q32_pos)
        have hm : -(Real.pi / 2) < (-6746518852 : ℝ) / q32 := by
          simpa [neg_div] using neg_lt_neg hp1
        have hra2 : ¬ (toReal r < -(Real.pi / 2)) :=
          not_lt.mpr (le_of_lt (lt_of_lt_of_le hm hthis2))
        have hneg : fixed_trig.fold1_negate r.val = false := by
          simp [fixed_trig.fold1_negate, h1, h2]
        simp [hneg, hra, hra2]
  have h2b : |(if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
      Real.cos ((((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32)| ≤ 1 / (2 * q32) := by
    by_cases ht1 : (3373259426 : ℤ) < fixed_trig.fold1 r.val
    · have hswap : fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 := by
        simp [fixed_trig.fold2_swap, ht1]
      have harg : fixed_trig.fold2 (fixed_trig.fold1 r.val) =
          6746518852 - fixed_trig.fold1 r.val := by simp [fixed_trig.fold2, ht1]
      rw [hswap, harg]
      rw [show (((6746518852 - fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 =
          (6746518852 : ℝ) / q32 - (((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 by push_cast; ring]
      set X : ℝ := (((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 with hX
      calc |Real.sin ((6746518852 : ℝ) / q32 - X) - Real.cos X|
          = |Real.sin ((6746518852 : ℝ) / q32 - X) - Real.sin (Real.pi / 2 - X)| := by
              rw [Real.sin_pi_div_two_sub]
        _ ≤ |((6746518852 : ℝ) / q32 - X) - (Real.pi / 2 - X)| := Real.abs_sin_sub_sin_le _ _
        _ = |(6746518852 : ℝ) / q32 - Real.pi / 2| := by ring_nf
        _ ≤ 1 / (2 * q32) := half_pi_q32_defect
    · by_cases ht2 : fixed_trig.fold1 r.val < (-3373259426 : ℤ)
      · have hswap : fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 := by
          simp [fixed_trig.fold2_swap, ht1, ht2]
        have harg : fixed_trig.fold2 (fixed_trig.fold1 r.val) =
            fixed_trig.fold1 r.val + 6746518852 := by simp [fixed_trig.fold2, ht1, ht2]
        rw [hswap, harg]
        rw [show (((fixed_trig.fold1 r.val + 6746518852 : ℤ) : ℝ)) / q32 =
            (((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 + (6746518852 : ℝ) / q32 by push_cast; ring]
        set X : ℝ := (((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 with hX
        calc |Real.sin (X + (6746518852 : ℝ) / q32) - Real.cos X|
            = |Real.sin (X + (6746518852 : ℝ) / q32) - Real.sin (X + Real.pi / 2)| := by
                rw [Real.sin_add_pi_div_two]
          _ ≤ |(X + (6746518852 : ℝ) / q32) - (X + Real.pi / 2)| := Real.abs_sin_sub_sin_le _ _
          _ = |(6746518852 : ℝ) / q32 - Real.pi / 2| := by ring_nf
          _ ≤ 1 / (2 * q32) := half_pi_q32_defect
      · have hswap : fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 0 := by
          simp [fixed_trig.fold2_swap, ht1, ht2]
        have harg : fixed_trig.fold2 (fixed_trig.fold1 r.val) = fixed_trig.fold1 r.val := by
          simp [fixed_trig.fold2, ht1, ht2]
        rw [hswap, harg]
        rw [if_neg (by norm_num : ¬ ((0 : ℤ) = 1)), if_neg (by norm_num : ¬ ((0 : ℤ) = 2)),
            sub_self, abs_zero]
        have hq : (0 : ℝ) < q32 := q32_pos
        positivity
  have htri : |(if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
      Real.cos (fold1T (toReal r))| ≤ 1 / (2 * q32) + 1 / (2 * q32) := by
    have hs2 : |Real.cos ((((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32) -
        Real.cos (fold1T (toReal r))| ≤
        |(((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32 - fold1T (toReal r)| :=
      Real.abs_cos_sub_cos_le _ _
    calc |(if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
        else Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
        Real.cos (fold1T (toReal r))|
        ≤ |(if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
         else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
         else Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
          Real.cos ((((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32)| +
          |Real.cos ((((fixed_trig.fold1 r.val : ℤ) : ℝ)) / q32) - Real.cos (fold1T (toReal r))| :=
            abs_sub_le _ _ _
      _ ≤ 1 / (2 * q32) + 1 / (2 * q32) := add_le_add h2b (le_trans hs2 h1disp)
  have hσ : |(if fixed_trig.fold1_negate r.val then (-1 : ℝ) else 1)| = 1 := by
    by_cases h : fixed_trig.fold1_negate r.val <;> simp [h]
  have hkey : flagCos (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
        (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.cos (toReal r)
      = (if fixed_trig.fold1_negate r.val then (-1 : ℝ) else 1) *
        ((if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
         else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then
          Real.sin (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)
         else Real.cos (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32)) -
          Real.cos (fold1T (toReal r))) := by
    rw [flagCos, ← hsig]; ring
  rw [hkey, abs_mul, hσ, one_mul]
  rw [show trig_fold_delta = 1 / q32 from rfl]
  exact le_trans htri (le_of_eq (by ring))

/-- Pre-Taylor flagged `sin` bound against the original angle: the flagged extracted combination is
within the total displacement `trig_delta + trig_fold_delta` of `sin (toReal x)` on `trig_window x`.
`trig_delta = 326/q32` is the reduction-remainder displacement; `trig_fold_delta = 1/q32` is the
two-constant fold displacement (`2 * (1/2)/q32`). -/
theorem flag_sin_reduction_bound (x r k : I64) (hx : trig_window x)
    (hk : k.val = (x.val * 683565276 + 2 ^ 63) >>> 64)
    (hr : r.val = x.val - k.val * 26986075409) :
    |flagSin (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
        (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.sin (toReal x)| ≤
      trig_delta + trig_fold_delta := by
  have h1 := flag_sin_fold_bound r
  have h2 := trig_sin_reduction_transfer x r k hx hk hr
  calc |flagSin (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
        (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.sin (toReal x)|
      ≤ |flagSin (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
          (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.sin (toReal r)| +
        |Real.sin (toReal r) - Real.sin (toReal x)| := abs_sub_le _ _ _
      _ ≤ trig_fold_delta + trig_delta := add_le_add h1 h2
      _ = trig_delta + trig_fold_delta := by ring

/-- Pre-Taylor flagged `cos` bound against the original angle: within `trig_delta + trig_fold_delta`
of `cos (toReal x)` on `trig_window x`. -/
theorem flag_cos_reduction_bound (x r k : I64) (hx : trig_window x)
    (hk : k.val = (x.val * 683565276 + 2 ^ 63) >>> 64)
    (hr : r.val = x.val - k.val * 26986075409) :
    |flagCos (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
        (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.cos (toReal x)| ≤
      trig_delta + trig_fold_delta := by
  have h1 := flag_cos_fold_bound r
  have h2 := trig_cos_reduction_transfer x r k hx hk hr
  calc |flagCos (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
        (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.cos (toReal x)|
      ≤ |flagCos (fixed_trig.fold1_negate r.val) (fixed_trig.fold2_swap (fixed_trig.fold1 r.val))
          (((fixed_trig.fold2 (fixed_trig.fold1 r.val) : ℤ) : ℝ) / q32) - Real.cos (toReal r)| +
        |Real.cos (toReal r) - Real.cos (toReal x)| := abs_sub_le _ _ _
      _ ≤ trig_fold_delta + trig_delta := add_le_add h1 h2
      _ = trig_delta + trig_fold_delta := by ring

/-- Real Horner target of the extracted `cos_poly`, in `s = t/2^32`. -/
def fixed_trig.cosHorner (s : ℝ) : ℝ :=
  ((((106522 : ℝ) / q32 * s ^ 2 - 5965232 / q32) * s ^ 2 + 178956971 / q32) * s ^ 2 -
      2147483648 / q32) * s ^ 2 + 1

/-- Real Horner target of the extracted `sin_poly`, in `s = t/2^32`. -/
def fixed_trig.sinHorner (s : ℝ) : ℝ :=
  s * ((((-852176 : ℝ) / q32 * s ^ 2 + 35791394 / q32) * s ^ 2 - 715827883 / q32) * s ^ 2 +
    4294967296 / q32)

theorem fixed_trig.sin_poly_horner (t : I64) (ht : |t.val| ≤ 2 ^ 32) :
    |((fixed_trig.sin_poly t.val : ℤ) : ℝ) / q32 - fixed_trig.sinHorner ((t.val : ℝ) / q32)| ≤
      15 / (2 * q32) := by
  have hd : (0 : ℝ) < q32 := q32_pos
  unfold fixed_trig.sin_poly
  dsimp only
  set T : ℤ := t.val with hT
  have hT32 : |T| ≤ 2 ^ 32 := by rw [hT]; exact ht
  set s : ℝ := (T : ℝ) / q32 with hs
  have hs1 : |s| ≤ 1 := by
    have hTr : |(T : ℝ)| ≤ (2 : ℝ) ^ 32 := by exact_mod_cast hT32
    rw [hs, abs_div, abs_of_pos hd, div_le_one hd]
    simpa [q32] using hTr
  have hs2 : s ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one s).mpr hs1
  have hs2n : 0 ≤ s ^ 2 := sq_nonneg s
  set u : ℤ := fixed_trig.round_q32 (T * T) with hu
  set a0 : ℤ := fixed_trig.round_q32 ((-852176) * u) + 35791394 with ha0
  set a1 : ℤ := fixed_trig.round_q32 (a0 * u) - 715827883 with ha1
  set a2 : ℤ := fixed_trig.round_q32 (a1 * u) + 4294967296 with ha2
  set U : ℝ := (u : ℝ) / q32 with hUdef
  set A0 : ℝ := (a0 : ℝ) / q32 with hA0def
  set A1 : ℝ := (a1 : ℝ) / q32 with hA1def
  set A2 : ℝ := (a2 : ℝ) / q32 with hA2def
  set S : ℝ := ((fixed_trig.round_q32 (T * a2) : ℤ) : ℝ) / q32 with hSdef
  set D : ℝ := 1 / (2 * q32) with hD
  have hDpos : 0 < D := by rw [hD]; positivity
  have hD1 : D ≤ 1 := by rw [hD]; norm_num [q32]
  have h852 : (852176 : ℝ) / q32 ≤ 1 / 5000 := by norm_num [q32]
  have h357 : (35791394 : ℝ) / q32 ≤ 1 / 120 := by norm_num [q32]
  have h715 : (715827883 : ℝ) / q32 ≤ 1 / 5 := by norm_num [q32]
  have h42 : (4294967296 : ℝ) / q32 = 1 := by norm_num [q32]
  have hc852 : |(-852176 : ℝ) / q32| = 852176 / q32 := by
    rw [abs_div, abs_of_neg (by norm_num : (-852176 : ℝ) < 0), abs_of_pos hd]
    ring
  -- stage ledgers
  have hU : |U - s ^ 2| ≤ D := by
    have h := fixed_trig.round_q32_ledger (T * T)
    rw [← hu, ← hD] at h
    have hsq : (((T * T : ℤ) : ℝ)) / q32 / q32 = s ^ 2 := by rw [hs]; push_cast; ring
    rw [hsq] at h
    simpa [hUdef] using h
  have hA0 : |A0 - ((-852176 : ℝ) / q32 * U + 35791394 / q32)| ≤ D := by
    have h := fixed_trig.round_q32_ledger ((-852176) * u)
    have h0 : ((((-852176) * u : ℤ) : ℝ)) / q32 / q32 = (-852176 : ℝ) / q32 * U := by
      rw [hUdef]; push_cast; ring
    rw [h0, ← hD] at h
    rw [show A0 - ((-852176 : ℝ) / q32 * U + 35791394 / q32) =
        ((fixed_trig.round_q32 ((-852176) * u) : ℤ) : ℝ) / q32 - (-852176 : ℝ) / q32 * U by
      rw [hA0def, ha0]; push_cast; ring]
    exact h
  have hA1 : |A1 - (A0 * U - 715827883 / q32)| ≤ D := by
    have h := fixed_trig.round_q32_ledger (a0 * u)
    have h0 : (((a0 * u : ℤ) : ℝ)) / q32 / q32 = A0 * U := by
      rw [hA0def, hUdef]; push_cast; ring
    rw [h0, ← hD] at h
    rw [show A1 - (A0 * U - 715827883 / q32) =
        ((fixed_trig.round_q32 (a0 * u) : ℤ) : ℝ) / q32 - A0 * U by
      rw [hA1def, ha1]; push_cast; ring]
    exact h
  have hA2 : |A2 - (A1 * U + 1)| ≤ D := by
    have h := fixed_trig.round_q32_ledger (a1 * u)
    have h0 : (((a1 * u : ℤ) : ℝ)) / q32 / q32 = A1 * U := by
      rw [hA1def, hUdef]; push_cast; ring
    rw [h0, ← hD] at h
    rw [show A2 - (A1 * U + 1) =
        ((fixed_trig.round_q32 (a1 * u) : ℤ) : ℝ) / q32 - A1 * U by
      rw [hA2def, ha2]; push_cast; rw [add_div, h42]; ring]
    exact h
  have hS : |S - s * A2| ≤ D := by
    have h := fixed_trig.round_q32_ledger (T * a2)
    have h0 : (((T * a2 : ℤ) : ℝ)) / q32 / q32 = s * A2 := by
      rw [hs, hA2def]; push_cast; ring
    rw [h0, ← hD] at h
    rw [show S - s * A2 = ((fixed_trig.round_q32 (T * a2) : ℤ) : ℝ) / q32 - s * A2 by
      rw [hSdef]]
    exact h
  -- E chain
  set E0 : ℝ := (-852176 : ℝ) / q32 * s ^ 2 + 35791394 / q32 with hE0
  set E1 : ℝ := E0 * s ^ 2 - 715827883 / q32 with hE1
  set E2 : ℝ := E1 * s ^ 2 + 1 with hE2
  have hE0b : |E0| ≤ 1 / 100 := by
    have hc : (-852176 : ℝ) / q32 ≤ 0 := by norm_num [q32]
    have hterm : (-852176 : ℝ) / q32 * s ^ 2 ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hc hs2n
    have hterm2 : (-852176 : ℝ) / q32 ≤ (-852176 : ℝ) / q32 * s ^ 2 := by
      have := mul_le_mul_of_nonpos_left hs2 hc
      simpa [mul_comm] using this
    have hlow : (-1 / 5000 : ℝ) ≤ (-852176 : ℝ) / q32 := by norm_num [q32]
    have h357n : 0 ≤ (35791394 : ℝ) / q32 := by norm_num [q32]
    rw [hE0, abs_le]
    constructor <;> linarith only [hterm, hterm2, hlow, h357, h357n, hDpos]
  have hE1b : |E1| ≤ 1 / 4 := by
    have h1 : |E0 * s ^ 2| ≤ 1 / 100 := by
      rw [abs_mul, abs_of_nonneg hs2n]
      calc |E0| * s ^ 2 ≤ (1 / 100) * s ^ 2 := mul_le_mul_of_nonneg_right hE0b hs2n
        _ ≤ (1 / 100) * 1 := mul_le_mul_of_nonneg_left hs2 (by norm_num)
        _ = 1 / 100 := by ring
    have h2 : |(715827883 : ℝ) / q32| ≤ 1 / 5 := by
      rw [abs_of_pos (by positivity)]; exact h715
    have h3 : |E0 * s ^ 2 - 715827883 / q32| ≤ |E0 * s ^ 2| + |715827883 / q32| := by
      rw [abs_sub_le_iff]
      constructor <;>
        linarith [le_abs_self (E0 * s ^ 2), neg_abs_le (E0 * s ^ 2),
          le_abs_self (715827883 / q32 : ℝ), neg_abs_le (715827883 / q32 : ℝ)]
    rw [hE1]
    linarith only [h1, h2, h3]
  have hUb : |U| ≤ 2 := by
    have h : |U| ≤ D + 1 := by
      calc |U| = |(U - s ^ 2) + s ^ 2| := by ring_nf
        _ ≤ |U - s ^ 2| + |s ^ 2| := abs_add_le _ _
        _ ≤ D + 1 := add_le_add hU (by rw [abs_of_nonneg hs2n]; exact hs2)
    linarith only [h, hD1]
  -- propagation
  have hmul : ∀ {a b A B : ℝ}, |a| ≤ A → |b| ≤ B → 0 ≤ A → |a * b| ≤ A * B := by
    intro a b A B ha hb hA
    rw [abs_mul]
    exact mul_le_mul ha hb (abs_nonneg b) hA
  have he0 : |A0 - E0| ≤ 2 * D := by
    have hX : |((-852176 : ℝ) / q32 * U + 35791394 / q32) - E0| ≤ (1 / 5000) * D := by
      have he : ((-852176 : ℝ) / q32 * U + 35791394 / q32) - E0 =
          (-852176 : ℝ) / q32 * (U - s ^ 2) := by rw [hE0]; ring
      rw [he, abs_mul, hc852]
      exact mul_le_mul h852 hU (abs_nonneg _) (by positivity)
    calc |A0 - E0| ≤ |A0 - ((-852176 : ℝ) / q32 * U + 35791394 / q32)| +
          |((-852176 : ℝ) / q32 * U + 35791394 / q32) - E0| :=
          abs_sub_le A0 ((-852176 : ℝ) / q32 * U + 35791394 / q32) E0
      _ ≤ D + (1 / 5000) * D := add_le_add hA0 hX
      _ ≤ 2 * D := by linarith only [hDpos]
  have he1 : |A1 - E1| ≤ 6 * D := by
    have hY : |(A0 * U - 715827883 / q32) - E1| ≤ 2 * D * 2 + (1 / 100) * D := by
      have he : (A0 * U - 715827883 / q32) - E1 = (A0 - E0) * U + E0 * (U - s ^ 2) := by
        rw [hE1]; ring
      rw [he]
      calc |(A0 - E0) * U + E0 * (U - s ^ 2)|
          ≤ |(A0 - E0) * U| + |E0 * (U - s ^ 2)| := abs_add_le _ _
        _ ≤ 2 * D * 2 + (1 / 100) * D :=
            add_le_add (hmul he0 hUb (by positivity)) (hmul hE0b hU (by positivity))
    calc |A1 - E1| ≤ |A1 - (A0 * U - 715827883 / q32)| + |(A0 * U - 715827883 / q32) - E1| :=
          abs_sub_le A1 (A0 * U - 715827883 / q32) E1
      _ ≤ D + (2 * D * 2 + (1 / 100) * D) := add_le_add hA1 hY
      _ ≤ 6 * D := by linarith only [hDpos]
  have he2 : |A2 - E2| ≤ 14 * D := by
    have hZ : |(A1 * U + 1) - E2| ≤ 6 * D * 2 + (1 / 4) * D := by
      have he : (A1 * U + 1) - E2 = (A1 - E1) * U + E1 * (U - s ^ 2) := by
        rw [hE2]; ring
      rw [he]
      calc |(A1 - E1) * U + E1 * (U - s ^ 2)|
          ≤ |(A1 - E1) * U| + |E1 * (U - s ^ 2)| := abs_add_le _ _
        _ ≤ 6 * D * 2 + (1 / 4) * D :=
            add_le_add (hmul he1 hUb (by positivity)) (hmul hE1b hU (by positivity))
    calc |A2 - E2| ≤ |A2 - (A1 * U + 1)| + |(A1 * U + 1) - E2| :=
          abs_sub_le A2 (A1 * U + 1) E2
      _ ≤ D + (6 * D * 2 + (1 / 4) * D) := add_le_add hA2 hZ
      _ ≤ 14 * D := by linarith only [hDpos]
  have he3 : |S - s * E2| ≤ 15 * D := by
    have hW : |s * A2 - s * E2| ≤ 14 * D := by
      have he : s * A2 - s * E2 = s * (A2 - E2) := by ring
      rw [he]
      calc |s * (A2 - E2)| = |s| * |A2 - E2| := by rw [abs_mul]
        _ ≤ 1 * (14 * D) := mul_le_mul hs1 he2 (abs_nonneg _) (by norm_num)
        _ = 14 * D := by ring
    calc |S - s * E2| ≤ |S - s * A2| + |s * A2 - s * E2| := abs_sub_le S (s * A2) (s * E2)
      _ ≤ D + 14 * D := add_le_add hS hW
      _ ≤ 15 * D := by linarith only []
  rw [show fixed_trig.sinHorner ((T : ℝ) / q32) = s * E2 by
    rw [← hs]; unfold fixed_trig.sinHorner; rw [← hE0, ← hE1, h42, ← hE2]]
  have hfin : (15 : ℝ) / (2 * q32) = 15 * D := by rw [hD]; ring
  rw [hfin]
  exact he3


theorem fixed_trig.cos_poly_horner (t : I64) (ht : |t.val| ≤ 2 ^ 32) :
    |((fixed_trig.cos_poly t.val : ℤ) : ℝ) / q32 - fixed_trig.cosHorner ((t.val : ℝ) / q32)| ≤
      40 / (2 * q32) := by
  have hd : (0 : ℝ) < q32 := q32_pos
  unfold fixed_trig.cos_poly
  dsimp only
  set T : ℤ := t.val with hT
  have hT32 : |T| ≤ 2 ^ 32 := by rw [hT]; exact ht
  set s : ℝ := (T : ℝ) / q32 with hs
  have hs1 : |s| ≤ 1 := by
    have hTr : |(T : ℝ)| ≤ (2 : ℝ) ^ 32 := by exact_mod_cast hT32
    rw [hs, abs_div, abs_of_pos hd, div_le_one hd]
    simpa [q32] using hTr
  have hs2 : s ^ 2 ≤ 1 := (sq_le_one_iff_abs_le_one s).mpr hs1
  have hs2n : 0 ≤ s ^ 2 := sq_nonneg s
  set u : ℤ := fixed_trig.round_q32 (T * T) with hu
  set b0 : ℤ := fixed_trig.round_q32 (106522 * u) - 5965232 with hb0
  set b1 : ℤ := fixed_trig.round_q32 (b0 * u) + 178956971 with hb1
  set b2 : ℤ := fixed_trig.round_q32 (b1 * u) - 2147483648 with hb2
  set U : ℝ := (u : ℝ) / q32 with hUdef
  set B0 : ℝ := (b0 : ℝ) / q32 with hB0def
  set B1 : ℝ := (b1 : ℝ) / q32 with hB1def
  set B2 : ℝ := (b2 : ℝ) / q32 with hB2def
  set SR : ℝ := ((fixed_trig.round_q32 (b2 * u) : ℤ) : ℝ) / q32 + 1 with hSRdef
  set D : ℝ := 1 / (2 * q32) with hD
  have hDpos : 0 < D := by rw [hD]; positivity
  have hD1 : D ≤ 1 := by rw [hD]; norm_num [q32]
  have h106 : (106522 : ℝ) / q32 ≤ 1 / 2 := by norm_num [q32]
  have h596 : (5965232 : ℝ) / q32 ≤ 1 / 2 := by norm_num [q32]
  have h596n : 0 ≤ (5965232 : ℝ) / q32 := by norm_num [q32]
  have h178 : (178956971 : ℝ) / q32 ≤ 1 / 10 := by norm_num [q32]
  have h214 : (2147483648 : ℝ) / q32 = 1 / 2 := by norm_num [q32]
  have h42c : (4294967296 : ℝ) / q32 = 1 := by norm_num [q32]
  have hU : |U - s ^ 2| ≤ D := by
    have h := fixed_trig.round_q32_ledger (T * T)
    rw [← hu, ← hD] at h
    have hsq : (((T * T : ℤ) : ℝ)) / q32 / q32 = s ^ 2 := by rw [hs]; push_cast; ring
    rw [hsq] at h
    simpa [hUdef] using h
  have hB0 : |B0 - ((106522 : ℝ) / q32 * U - 5965232 / q32)| ≤ D := by
    have h := fixed_trig.round_q32_ledger (106522 * u)
    have h0 : (((106522 * u : ℤ) : ℝ)) / q32 / q32 = (106522 : ℝ) / q32 * U := by
      rw [hUdef]; push_cast; ring
    rw [h0, ← hD] at h
    rw [show B0 - ((106522 : ℝ) / q32 * U - 5965232 / q32) =
        ((fixed_trig.round_q32 (106522 * u) : ℤ) : ℝ) / q32 - (106522 : ℝ) / q32 * U by
      rw [hB0def, hb0]; push_cast; ring]
    exact h
  have hB1 : |B1 - (B0 * U + 178956971 / q32)| ≤ D := by
    have h := fixed_trig.round_q32_ledger (b0 * u)
    have h0 : (((b0 * u : ℤ) : ℝ)) / q32 / q32 = B0 * U := by
      rw [hB0def, hUdef]; push_cast; ring
    rw [h0, ← hD] at h
    rw [show B1 - (B0 * U + 178956971 / q32) =
        ((fixed_trig.round_q32 (b0 * u) : ℤ) : ℝ) / q32 - B0 * U by
      rw [hB1def, hb1]; push_cast; ring]
    exact h
  have hB2 : |B2 - (B1 * U - 2147483648 / q32)| ≤ D := by
    have h := fixed_trig.round_q32_ledger (b1 * u)
    have h0 : (((b1 * u : ℤ) : ℝ)) / q32 / q32 = B1 * U := by
      rw [hB1def, hUdef]; push_cast; ring
    rw [h0, ← hD] at h
    rw [show B2 - (B1 * U - 2147483648 / q32) =
        ((fixed_trig.round_q32 (b1 * u) : ℤ) : ℝ) / q32 - B1 * U by
      rw [hB2def, hb2]; push_cast; ring]
    exact h
  have hSR : |SR - (B2 * U + 1)| ≤ D := by
    have h := fixed_trig.round_q32_ledger (b2 * u)
    have h0 : (((b2 * u : ℤ) : ℝ)) / q32 / q32 = B2 * U := by
      rw [hB2def, hUdef]; push_cast; ring
    rw [h0, ← hD] at h
    rw [show SR - (B2 * U + 1) =
        ((fixed_trig.round_q32 (b2 * u) : ℤ) : ℝ) / q32 - B2 * U by
      rw [hSRdef]; ring]
    exact h
  set F0 : ℝ := (106522 : ℝ) / q32 * s ^ 2 - 5965232 / q32 with hF0
  set F1 : ℝ := F0 * s ^ 2 + 178956971 / q32 with hF1
  set F2 : ℝ := F1 * s ^ 2 - 2147483648 / q32 with hF2
  have hF0b : |F0| ≤ 1 := by
    have hc : (0 : ℝ) ≤ (106522 : ℝ) / q32 := by norm_num [q32]
    have hterm : (106522 : ℝ) / q32 * s ^ 2 ≤ 1 / 2 := by
      calc (106522 : ℝ) / q32 * s ^ 2 ≤ (1 / 2) * s ^ 2 := mul_le_mul_of_nonneg_right h106 hs2n
        _ ≤ (1 / 2) * 1 := mul_le_mul_of_nonneg_left hs2 (by norm_num)
        _ = 1 / 2 := by ring
    have hterm2 : -(1 / 2) ≤ (106522 : ℝ) / q32 * s ^ 2 := by
      have := mul_nonneg hc hs2n; linarith
    rw [hF0, abs_le]
    constructor <;> linarith only [hterm, hterm2, h596, h596n]
  have hF1b : |F1| ≤ 2 := by
    have h1 : |F0 * s ^ 2| ≤ 1 := by
      rw [abs_mul, abs_of_nonneg hs2n]
      calc |F0| * s ^ 2 ≤ 1 * s ^ 2 := mul_le_mul_of_nonneg_right hF0b hs2n
        _ ≤ 1 * 1 := mul_le_mul_of_nonneg_left hs2 (by norm_num)
        _ = 1 := by ring
    have h2 : |(178956971 : ℝ) / q32| ≤ 1 / 10 := by
      rw [abs_of_pos (by positivity)]; exact h178
    rw [hF1]
    linarith only [h1, h2, abs_add_le (F0 * s ^ 2) (178956971 / q32)]
  have hF2b : |F2| ≤ 3 := by
    have h1 : |F1 * s ^ 2| ≤ 2 := by
      rw [abs_mul, abs_of_nonneg hs2n]
      calc |F1| * s ^ 2 ≤ 2 * s ^ 2 := mul_le_mul_of_nonneg_right hF1b hs2n
        _ ≤ 2 * 1 := mul_le_mul_of_nonneg_left hs2 (by norm_num)
        _ = 2 := by ring
    have h2 : |(2147483648 : ℝ) / q32| ≤ 1 / 2 := by
      rw [abs_of_pos (by positivity), h214]
    have h3 : |F1 * s ^ 2 - 2147483648 / q32| ≤ |F1 * s ^ 2| + |2147483648 / q32| := by
      rw [abs_sub_le_iff]
      constructor <;>
        linarith [le_abs_self (F1 * s ^ 2), neg_abs_le (F1 * s ^ 2),
          le_abs_self (2147483648 / q32 : ℝ), neg_abs_le (2147483648 / q32 : ℝ)]
    rw [hF2]
    linarith only [h1, h2, h3]
  have hUb : |U| ≤ 2 := by
    have h : |U| ≤ D + 1 := by
      calc |U| = |(U - s ^ 2) + s ^ 2| := by ring_nf
        _ ≤ |U - s ^ 2| + |s ^ 2| := abs_add_le _ _
        _ ≤ D + 1 := add_le_add hU (by rw [abs_of_nonneg hs2n]; exact hs2)
    linarith only [h, hD1]
  have hmul : ∀ {a b A B : ℝ}, |a| ≤ A → |b| ≤ B → 0 ≤ A → |a * b| ≤ A * B := by
    intro a b A B ha hb hA
    rw [abs_mul]
    exact mul_le_mul ha hb (abs_nonneg b) hA
  have hc106 : |(106522 : ℝ) / q32| = 106522 / q32 := by
    rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 106522), abs_of_pos hd]
  have he0 : |B0 - F0| ≤ 2 * D := by
    have hX : |((106522 : ℝ) / q32 * U - 5965232 / q32) - F0| ≤ (1 / 2) * D := by
      have he : ((106522 : ℝ) / q32 * U - 5965232 / q32) - F0 =
          (106522 : ℝ) / q32 * (U - s ^ 2) := by rw [hF0]; ring
      rw [he, abs_mul, hc106]
      exact mul_le_mul h106 hU (abs_nonneg _) (by positivity)
    calc |B0 - F0| ≤ |B0 - ((106522 : ℝ) / q32 * U - 5965232 / q32)| +
          |((106522 : ℝ) / q32 * U - 5965232 / q32) - F0| :=
          abs_sub_le B0 ((106522 : ℝ) / q32 * U - 5965232 / q32) F0
      _ ≤ D + (1 / 2) * D := add_le_add hB0 hX
      _ ≤ 2 * D := by linarith only [hDpos]
  have he1 : |B1 - F1| ≤ 6 * D := by
    have hY : |(B0 * U + 178956971 / q32) - F1| ≤ 2 * D * 2 + 1 * D := by
      have he : (B0 * U + 178956971 / q32) - F1 = (B0 - F0) * U + F0 * (U - s ^ 2) := by
        rw [hF1]; ring
      rw [he]
      calc |(B0 - F0) * U + F0 * (U - s ^ 2)|
          ≤ |(B0 - F0) * U| + |F0 * (U - s ^ 2)| := abs_add_le _ _
        _ ≤ 2 * D * 2 + 1 * D :=
            add_le_add (hmul he0 hUb (by positivity)) (hmul hF0b hU (by positivity))
    calc |B1 - F1| ≤ |B1 - (B0 * U + 178956971 / q32)| + |(B0 * U + 178956971 / q32) - F1| :=
          abs_sub_le B1 (B0 * U + 178956971 / q32) F1
      _ ≤ D + (2 * D * 2 + 1 * D) := add_le_add hB1 hY
      _ ≤ 6 * D := by linarith only [hDpos]
  have he2 : |B2 - F2| ≤ 15 * D := by
    have hZ : |(B1 * U - 2147483648 / q32) - F2| ≤ 6 * D * 2 + 2 * D := by
      have he : (B1 * U - 2147483648 / q32) - F2 = (B1 - F1) * U + F1 * (U - s ^ 2) := by
        rw [hF2]; ring
      rw [he]
      calc |(B1 - F1) * U + F1 * (U - s ^ 2)|
          ≤ |(B1 - F1) * U| + |F1 * (U - s ^ 2)| := abs_add_le _ _
        _ ≤ 6 * D * 2 + 2 * D :=
            add_le_add (hmul he1 hUb (by positivity)) (hmul hF1b hU (by positivity))
    calc |B2 - F2| ≤ |B2 - (B1 * U - 2147483648 / q32)| + |(B1 * U - 2147483648 / q32) - F2| :=
          abs_sub_le B2 (B1 * U - 2147483648 / q32) F2
      _ ≤ D + (6 * D * 2 + 2 * D) := add_le_add hB2 hZ
      _ ≤ 15 * D := by linarith only [hDpos]
  have he3 : |SR - (F2 * s ^ 2 + 1)| ≤ 40 * D := by
    have hW : |(B2 * U + 1) - (F2 * s ^ 2 + 1)| ≤ 15 * D * 2 + 3 * D := by
      have he : (B2 * U + 1) - (F2 * s ^ 2 + 1) = (B2 - F2) * U + F2 * (U - s ^ 2) := by
        ring
      rw [he]
      calc |(B2 - F2) * U + F2 * (U - s ^ 2)|
          ≤ |(B2 - F2) * U| + |F2 * (U - s ^ 2)| := abs_add_le _ _
        _ ≤ 15 * D * 2 + 3 * D :=
            add_le_add (hmul he2 hUb (by positivity)) (hmul hF2b hU (by positivity))
    calc |SR - (F2 * s ^ 2 + 1)| ≤ |SR - (B2 * U + 1)| + |(B2 * U + 1) - (F2 * s ^ 2 + 1)| :=
          abs_sub_le SR (B2 * U + 1) (F2 * s ^ 2 + 1)
      _ ≤ D + (15 * D * 2 + 3 * D) := add_le_add hSR hW
      _ ≤ 40 * D := by linarith only [hDpos]
  rw [show fixed_trig.cosHorner ((T : ℝ) / q32) = F2 * s ^ 2 + 1 by
    rw [← hs]; unfold fixed_trig.cosHorner; rw [← hF0, ← hF1, ← hF2]]
  have hSRlhs : ((fixed_trig.round_q32 (b2 * u) + 4294967296 : ℤ) : ℝ) / q32 = SR := by
    rw [hSRdef]; push_cast; rw [add_div, h42c]
  have hfin : (40 : ℝ) / (2 * q32) = 40 * D := by rw [hD]; ring
  rw [hfin, hSRlhs]
  exact he3
/-- Degree-7 truncation of `sin`, the real polynomial `sinHorner` targets. -/
def fixed_trig.sinTaylor (s : ℝ) : ℝ := s - s ^ 3 / 6 + s ^ 5 / 120 - s ^ 7 / 5040

/-- Degree-8 truncation of `cos`, the real polynomial `cosHorner` targets. -/
def fixed_trig.cosTaylor (s : ℝ) : ℝ := 1 - s ^ 2 / 2 + s ^ 4 / 24 - s ^ 6 / 720 + s ^ 8 / 40320

theorem fixed_trig.sinHorner_taylor (s : ℝ) (hs : |s| ≤ 1) :
    |fixed_trig.sinHorner s - fixed_trig.sinTaylor s| ≤ 1 / q32 := by
  have h1 : (1 / 6 : ℝ) - 715827883 / q32 = -1 / (3 * q32) := by norm_num [q32]
  have h2 : (35791394 : ℝ) / q32 - 1 / 120 = -2 / (15 * q32) := by norm_num [q32]
  have h3 : (1 / 5040 : ℝ) - 852176 / q32 = 16 / (315 * q32) := by norm_num [q32]
  have e : fixed_trig.sinHorner s - fixed_trig.sinTaylor s =
      ((1 / 6 : ℝ) - 715827883 / q32) * s ^ 3 + ((35791394 : ℝ) / q32 - 1 / 120) * s ^ 5 +
        ((1 / 5040 : ℝ) - 852176 / q32) * s ^ 7 := by
    unfold fixed_trig.sinHorner fixed_trig.sinTaylor
    rw [show (4294967296 : ℝ) / q32 = 1 by norm_num [q32]]
    ring
  have hs3 : |s| ^ 3 ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
  have hs5 : |s| ^ 5 ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
  have hs7 : |s| ^ 7 ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
  have hb1 : |((1 / 6 : ℝ) - 715827883 / q32)| = 1 / (3 * q32) := by
    rw [h1, abs_of_neg (by norm_num [q32])]; ring
  have hb2 : |((35791394 : ℝ) / q32 - 1 / 120)| = 2 / (15 * q32) := by
    rw [h2, abs_of_neg (by norm_num [q32])]; ring
  have hb3 : |((1 / 5040 : ℝ) - 852176 / q32)| = 16 / (315 * q32) := by
    rw [h3, abs_of_pos (by norm_num [q32])]
  rw [e]
  have hm1 : |((1 / 6 : ℝ) - 715827883 / q32) * s ^ 3| ≤ (1 / (3 * q32)) * 1 := by
    rw [abs_mul, abs_pow, hb1]
    exact mul_le_mul_of_nonneg_left hs3 (by norm_num [q32])
  have hm2 : |((35791394 : ℝ) / q32 - 1 / 120) * s ^ 5| ≤ (2 / (15 * q32)) * 1 := by
    rw [abs_mul, abs_pow, hb2]
    exact mul_le_mul_of_nonneg_left hs5 (by norm_num [q32])
  have hm3 : |((1 / 5040 : ℝ) - 852176 / q32) * s ^ 7| ≤ (16 / (315 * q32)) * 1 := by
    rw [abs_mul, abs_pow, hb3]
    exact mul_le_mul_of_nonneg_left hs7 (by norm_num [q32])
  calc |((1 / 6 : ℝ) - 715827883 / q32) * s ^ 3 +
        ((35791394 : ℝ) / q32 - 1 / 120) * s ^ 5 + ((1 / 5040 : ℝ) - 852176 / q32) * s ^ 7|
      ≤ |((1 / 6 : ℝ) - 715827883 / q32) * s ^ 3 +
          ((35791394 : ℝ) / q32 - 1 / 120) * s ^ 5| +
        |((1 / 5040 : ℝ) - 852176 / q32) * s ^ 7| := abs_add_le _ _
    _ ≤ |((1 / 6 : ℝ) - 715827883 / q32) * s ^ 3| +
          |((35791394 : ℝ) / q32 - 1 / 120) * s ^ 5| +
        |((1 / 5040 : ℝ) - 852176 / q32) * s ^ 7| := by
        linarith [abs_add_le (((1 / 6 : ℝ) - 715827883 / q32) * s ^ 3)
          (((35791394 : ℝ) / q32 - 1 / 120) * s ^ 5)]
    _ ≤ (1 / (3 * q32)) * 1 + (2 / (15 * q32)) * 1 + (16 / (315 * q32)) * 1 := by
        linarith [hm1, hm2, hm3]
    _ ≤ 1 / q32 := by
        rw [show (1 / (3 * q32)) * 1 + (2 / (15 * q32)) * 1 + (16 / (315 * q32)) * 1 =
          (163 / 315 : ℝ) / q32 by ring]
        rw [div_le_div_iff_of_pos_right q32_pos]
        norm_num

theorem fixed_trig.cosHorner_taylor (s : ℝ) (hs : |s| ≤ 1) :
    |fixed_trig.cosHorner s - fixed_trig.cosTaylor s| ≤ 1 / q32 := by
  have h1 : (178956971 : ℝ) / q32 - 1 / 24 = 1 / (3 * q32) := by norm_num [q32]
  have h2 : (1 / 720 : ℝ) - 5965232 / q32 = 16 / (45 * q32) := by norm_num [q32]
  have h3 : (106522 : ℝ) / q32 - 1 / 40320 = -2 / (315 * q32) := by norm_num [q32]
  have e : fixed_trig.cosHorner s - fixed_trig.cosTaylor s =
      ((178956971 : ℝ) / q32 - 1 / 24) * s ^ 4 + ((1 / 720 : ℝ) - 5965232 / q32) * s ^ 6 +
        ((106522 : ℝ) / q32 - 1 / 40320) * s ^ 8 := by
    unfold fixed_trig.cosHorner fixed_trig.cosTaylor
    rw [show (2147483648 : ℝ) / q32 = 1 / 2 by norm_num [q32]]
    ring
  have hs4 : |s| ^ 4 ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
  have hs6 : |s| ^ 6 ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
  have hs8 : |s| ^ 8 ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
  have hb1 : |((178956971 : ℝ) / q32 - 1 / 24)| = 1 / (3 * q32) := by
    rw [h1, abs_of_pos (by norm_num [q32])]
  have hb2 : |((1 / 720 : ℝ) - 5965232 / q32)| = 16 / (45 * q32) := by
    rw [h2, abs_of_pos (by norm_num [q32])]
  have hb3 : |((106522 : ℝ) / q32 - 1 / 40320)| = 2 / (315 * q32) := by
    rw [h3, abs_of_neg (by norm_num [q32])]; ring
  rw [e]
  have hm1 : |((178956971 : ℝ) / q32 - 1 / 24) * s ^ 4| ≤ (1 / (3 * q32)) * 1 := by
    rw [abs_mul, abs_pow, hb1]
    exact mul_le_mul_of_nonneg_left hs4 (by norm_num [q32])
  have hm2 : |((1 / 720 : ℝ) - 5965232 / q32) * s ^ 6| ≤ (16 / (45 * q32)) * 1 := by
    rw [abs_mul, abs_pow, hb2]
    exact mul_le_mul_of_nonneg_left hs6 (by norm_num [q32])
  have hm3 : |((106522 : ℝ) / q32 - 1 / 40320) * s ^ 8| ≤ (2 / (315 * q32)) * 1 := by
    rw [abs_mul, abs_pow, hb3]
    exact mul_le_mul_of_nonneg_left hs8 (by norm_num [q32])
  calc |((178956971 : ℝ) / q32 - 1 / 24) * s ^ 4 +
        ((1 / 720 : ℝ) - 5965232 / q32) * s ^ 6 + ((106522 : ℝ) / q32 - 1 / 40320) * s ^ 8|
      ≤ |((178956971 : ℝ) / q32 - 1 / 24) * s ^ 4 +
          ((1 / 720 : ℝ) - 5965232 / q32) * s ^ 6| +
        |((106522 : ℝ) / q32 - 1 / 40320) * s ^ 8| := abs_add_le _ _
    _ ≤ |((178956971 : ℝ) / q32 - 1 / 24) * s ^ 4| +
          |((1 / 720 : ℝ) - 5965232 / q32) * s ^ 6| +
        |((106522 : ℝ) / q32 - 1 / 40320) * s ^ 8| := by
        linarith [abs_add_le (((178956971 : ℝ) / q32 - 1 / 24) * s ^ 4)
          (((1 / 720 : ℝ) - 5965232 / q32) * s ^ 6)]
    _ ≤ (1 / (3 * q32)) * 1 + (16 / (45 * q32)) * 1 + (2 / (315 * q32)) * 1 := by
        linarith [hm1, hm2, hm3]
    _ ≤ 1 / q32 := by
        rw [show (1 / (3 * q32)) * 1 + (16 / (45 * q32)) * 1 + (2 / (315 * q32)) * 1 =
          (219 / 315 : ℝ) / q32 by ring]
        rw [div_le_div_iff_of_pos_right q32_pos]
        norm_num


theorem fixed_trig.sinTaylor_remainder (s : ℝ) (hs : |s| ≤ 1) :
    |fixed_trig.sinTaylor s - Real.sin s| ≤ |s| ^ 9 * (10 / 3265920) := by
  have hzn : ‖(s : ℂ) * Complex.I‖ = |s| := by
    rw [norm_mul, Complex.norm_I, mul_one, Complex.norm_def, Complex.normSq_ofReal, ← sq,
      Real.sqrt_sq_eq_abs]
  have hz : ‖(s : ℂ) * Complex.I‖ ≤ 1 := by rw [hzn]; exact hs
  have hsin : ((∑ m ∈ Finset.range 9, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)).im) =
      fixed_trig.sinTaylor s := by
    unfold fixed_trig.sinTaylor
    norm_num [Finset.sum_range_succ, Complex.I_mul_I]
    simp only [mul_pow, ← Complex.ofReal_pow, Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im,
      zero_mul, add_zero]
    norm_num [pow_succ, pow_two, Complex.I_mul_I]
    ring
  have hb := Complex.exp_bound hz (n := 9) (by norm_num)
  calc |fixed_trig.sinTaylor s - Real.sin s|
      = |((∑ m ∈ Finset.range 9, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) -
            Complex.exp ((s : ℂ) * Complex.I)).im| := by
          rw [← Complex.exp_ofReal_mul_I_im, ← hsin]
          simp only [← Complex.sub_im]
    _ ≤ ‖(∑ m ∈ Finset.range 9, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) -
            Complex.exp ((s : ℂ) * Complex.I)‖ := Complex.abs_im_le_norm _
    _ = ‖Complex.exp ((s : ℂ) * Complex.I) -
            (∑ m ∈ Finset.range 9, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ))‖ :=
          norm_sub_rev _ _
    _ ≤ ‖(s : ℂ) * Complex.I‖ ^ 9 * (10 * (362880 * 9)⁻¹) := hb
    _ = |s| ^ 9 * (10 / 3265920) := by rw [hzn]; norm_num

theorem fixed_trig.cosTaylor_remainder (s : ℝ) (hs : |s| ≤ 1) :
    |fixed_trig.cosTaylor s - Real.cos s| ≤ |s| ^ 10 * (11 / 36288000) := by
  have hzn : ‖(s : ℂ) * Complex.I‖ = |s| := by
    rw [norm_mul, Complex.norm_I, mul_one, Complex.norm_def, Complex.normSq_ofReal, ← sq,
      Real.sqrt_sq_eq_abs]
  have hz : ‖(s : ℂ) * Complex.I‖ ≤ 1 := by rw [hzn]; exact hs
  have hcos : ((∑ m ∈ Finset.range 10, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)).re) =
      fixed_trig.cosTaylor s := by
    unfold fixed_trig.cosTaylor
    norm_num [Finset.sum_range_succ, Complex.I_mul_I]
    simp only [mul_pow, ← Complex.ofReal_pow, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
      zero_mul]
    norm_num [pow_succ, pow_two, Complex.I_mul_I]
    ring
  have hb := Complex.exp_bound hz (n := 10) (by norm_num)
  calc |fixed_trig.cosTaylor s - Real.cos s|
      = |((∑ m ∈ Finset.range 10, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) -
            Complex.exp ((s : ℂ) * Complex.I)).re| := by
          rw [← Complex.exp_ofReal_mul_I_re, ← hcos]
          simp only [← Complex.sub_re]
    _ ≤ ‖(∑ m ∈ Finset.range 10, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) -
            Complex.exp ((s : ℂ) * Complex.I)‖ := Complex.abs_re_le_norm _
    _ = ‖Complex.exp ((s : ℂ) * Complex.I) -
            (∑ m ∈ Finset.range 10, ((s : ℂ) * Complex.I) ^ m / (m.factorial : ℂ))‖ :=
          norm_sub_rev _ _
    _ ≤ ‖(s : ℂ) * Complex.I‖ ^ 10 * (11 * (3628800 * 10)⁻¹) := hb
    _ = |s| ^ 10 * (11 / 36288000) := by rw [hzn]; norm_num
/-! ## The past-guards body: direct value specification

`word_fixed_sin_cos_window_past_guards` rewrites the extracted body to its past-guards program on
`trig_window x`. The theorem below is the value specification of that program with
`word_fixed_sin_cos_value`'s post: the reduction witnesses `k`/`r`, the fold witnesses
`s`/`negate`/`arg`/`swap`, and the output equations in the
`fold1`/`fold2`/`fold1_negate`/`fold2_swap` form. -/


end

end apeiron_word
