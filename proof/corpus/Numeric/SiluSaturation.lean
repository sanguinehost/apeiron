/-
SiLU saturation arms and the total (all-`i64`) error statement
(`APEIRON-ALGEBRA-G3-SILU`; plan `docs/epics/model-algebra.md`, gap G3).

`EngineSiluRefinement` proves the two saturation arms structurally
(`word_silu_fixed_high`, `word_silu_fixed_low`) and the in-window refinement
(`silu_window_refines`) against `siluReal`. This module supplies the missing real-error halves of
the two saturation arms and composes the three arms into one total statement:

* `silu_high_real`: `16 ≤ toReal v` gives `|toReal v - siluReal (toReal v)| ≤ 16·exp(-16)`;
* `silu_low_real`: `toReal v ≤ -21` gives `|toReal v - siluReal (toReal v)| ≤ 21·exp(-21)`;
* `silu_refines_total`: every `I64` is covered by exactly one of the three arms with error bounded
  by `siluErrTotal`, whose constant part dominates both saturation bounds and whose scaled part is
  the in-window composition.

Both saturation bounds rest on `x·exp(-x) ≤ c·exp(-c)` for `x ≥ c ≥ 1`, from
`exp(x-c) ≥ 1 + (x-c) ≥ x/c`. The case split is the two saturation guards versus their complement,
which is exactly `silu_window v`.
-/
import Numeric.Engine.Silu

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Exponential decay bound on the tail

`x·exp(-x)` decays monotonically on `[1, ∞)`. The elementary crossing estimate
`exp(x-c) ≥ 1 + (x-c) ≥ x/c` holds whenever `c ≥ 1` and `x ≥ c`; the rest is cancellation of
`exp(x-c)·exp(-x) = exp(-c)`. -/

/-- For `x ≥ c ≥ 1`, `x·exp(-x) ≤ c·exp(-c)`. -/
private theorem mul_exp_neg_le (c x : ℝ) (hc : 1 ≤ c) (hx : c ≤ x) :
    x * Real.exp (-x) ≤ c * Real.exp (-c) := by
  have hcpos : 0 < c := by linarith
  have h1 : x / c ≤ Real.exp (x - c) := by
    have h2 : x / c ≤ (x - c) + 1 := by
      rw [div_le_iff₀ hcpos]
      nlinarith [mul_nonneg (show (0 : ℝ) ≤ c - 1 by linarith)
        (show (0 : ℝ) ≤ x - c by linarith)]
    exact le_trans h2 (Real.add_one_le_exp (x - c))
  have h3 : (x / c) * (c * Real.exp (-x)) ≤ Real.exp (x - c) * (c * Real.exp (-x)) :=
    mul_le_mul_of_nonneg_right h1 (mul_nonneg (le_of_lt hcpos) (le_of_lt (Real.exp_pos _)))
  have hR : Real.exp (x - c) * Real.exp (-x) = Real.exp (-c) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc x * Real.exp (-x) = (x / c) * (c * Real.exp (-x)) := by
        field_simp
    _ ≤ Real.exp (x - c) * (c * Real.exp (-x)) := h3
    _ = c * Real.exp (-c) := by
        rw [mul_comm c (Real.exp (-x)), ← mul_assoc, hR]
        ring

/-- For `a ≥ 0`, `a/(1 + exp a) ≤ a·exp(-a)`. -/
private theorem div_one_add_exp_le (a : ℝ) (ha : 0 ≤ a) :
    a / (1 + Real.exp a) ≤ a * Real.exp (-a) := by
  have hpos : 0 < 1 + Real.exp a := by positivity
  rw [div_le_iff₀ hpos]
  have he : Real.exp (-a) * Real.exp a = 1 := by
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  have hnn : 0 ≤ a * Real.exp (-a) := mul_nonneg ha (le_of_lt (Real.exp_pos _))
  nlinarith [he, hnn]

/-! ## The high saturation real-error bound -/

/-- **Saturation arm 1, real error.** For `v ≥ 16·2^32`, the exact returned value `v` is within
`16·exp(-16)` of `siluReal (toReal v)`. -/
theorem silu_high_real (v : I64) (h : (68719476736 : ℤ) ≤ v.val) :
    |toReal v - siluReal (toReal v)| ≤ 16 * Real.exp (-16) := by
  have hx16 : (16 : ℝ) ≤ toReal v := by
    unfold toReal
    rw [le_div_iff₀ q32_pos]
    rw [show q32 = (2 : ℝ) ^ (32 : ℕ) from rfl]
    have h16 : (16 : ℝ) * (2 : ℝ) ^ (32 : ℕ) = (68719476736 : ℤ) := by norm_num
    rw [h16]
    exact_mod_cast h
  have htxnn : 0 ≤ toReal v := by linarith
  have hs : toReal v - siluReal (toReal v)
      = toReal v * Real.exp (-toReal v) / (1 + Real.exp (-toReal v)) := by
    unfold siluReal
    field_simp
    ring
  have hnn : 0 ≤ toReal v * Real.exp (-toReal v) / (1 + Real.exp (-toReal v)) :=
    div_nonneg (mul_nonneg htxnn (le_of_lt (Real.exp_pos _))) (by positivity)
  have hden1 : (1 : ℝ) ≤ 1 + Real.exp (-toReal v) := by
    have := Real.exp_pos (-toReal v)
    linarith
  have hA : 0 ≤ toReal v * Real.exp (-toReal v) :=
    mul_nonneg htxnn (le_of_lt (Real.exp_pos _))
  have hle : toReal v * Real.exp (-toReal v) / (1 + Real.exp (-toReal v))
      ≤ toReal v * Real.exp (-toReal v) := by
    rw [div_le_iff₀ (by positivity)]
    have := mul_le_mul_of_nonneg_left hden1 hA
    simpa using this
  rw [hs, abs_of_nonneg hnn]
  exact le_trans hle (mul_exp_neg_le 16 (toReal v) (by norm_num) hx16)

/-! ## The low saturation real-error bound -/

/-- **Saturation arm 2, real error.** For `v ≤ -21·2^32`, the exact returned value `0` is within
`21·exp(-21)` of `siluReal (toReal v)`. -/
theorem silu_low_real (v : I64) (h : v.val ≤ (-90194313216 : ℤ)) :
    |toReal 0#i64 - siluReal (toReal v)| ≤ 21 * Real.exp (-21) := by
  have hx21 : toReal v ≤ -21 := by
    unfold toReal
    rw [div_le_iff₀ q32_pos]
    rw [show q32 = (2 : ℝ) ^ (32 : ℕ) from rfl]
    have h21 : (-21 : ℝ) * (2 : ℝ) ^ (32 : ℕ) = (-90194313216 : ℤ) := by norm_num
    rw [h21]
    exact_mod_cast h
  have hxneg : toReal v < 0 := by linarith
  have hspos : 0 < 1 + Real.exp (-toReal v) := by positivity
  have hneg : siluReal (toReal v) < 0 := by
    unfold siluReal
    exact div_neg_of_neg_of_pos hxneg hspos
  have h0 : toReal (0#i64) = 0 := by
    unfold toReal
    norm_num
  have habs : |toReal 0#i64 - siluReal (toReal v)|
      = (-toReal v) / (1 + Real.exp (-toReal v)) := by
    rw [h0, zero_sub, abs_neg, abs_of_neg hneg]
    unfold siluReal
    ring
  have hdiv : (-toReal v) / (1 + Real.exp (-toReal v))
      ≤ (-toReal v) * Real.exp (toReal v) := by
    have hkey := div_one_add_exp_le (-toReal v) (by linarith)
    rwa [neg_neg] at hkey
  have hmul : (-toReal v) * Real.exp (toReal v) ≤ 21 * Real.exp (-21) := by
    have hkey := mul_exp_neg_le 21 (-toReal v) (by norm_num) (by linarith)
    rwa [neg_neg] at hkey
  rw [habs]
  exact le_trans hdiv hmul

/-! ## Total error statement

`siluErrTotal` takes the maximum of the in-window composition and the larger of the two saturation
constants; the three arms are exhaustive because `silu_window` is the complement of the two
guards. -/

/-- The total error target: in-window scaled term against the dominant saturation constant. -/
def siluErrTotal (x : ℝ) : ℝ :=
  max (epsExp * |x| + (4 / q32) * |x| + 1 / (2 * q32))
    (max (16 * Real.exp (-16)) (21 * Real.exp (-21)))

/-- **Total refinement.** Every `I64` input lands in one of the three arms and its returned value
is within `siluErrTotal (toReal v)` of `siluReal (toReal v)`. -/
theorem silu_refines_total (v : I64) :
    ∃ y : I64, word_silu_fixed v = ok (core.result.Result.Ok y) ∧
      |toReal y - siluReal (toReal v)| ≤ siluErrTotal (toReal v) := by
  by_cases hhi : (68719476736 : ℤ) ≤ v.val
  · refine ⟨v, word_silu_fixed_high v hhi, ?_⟩
    exact le_trans (silu_high_real v hhi)
      (le_trans (le_max_left _ _) (le_max_right _ _))
  · by_cases hlo : v.val ≤ (-90194313216 : ℤ)
    · refine ⟨0#i64, word_silu_fixed_low v hlo, ?_⟩
      exact le_trans (silu_low_real v hlo)
        (le_trans (le_max_right _ _) (le_max_right _ _))
    · have hwin : silu_window v := by
        rw [silu_window]
        constructor <;> omega
      obtain ⟨y, hy, hb⟩ := silu_window_refines v hwin
      exact ⟨y, hy, le_trans hb (le_max_left _ _)⟩

end

end apeiron_word
