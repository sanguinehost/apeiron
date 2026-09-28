import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic

/-!
# Speculative-decoding acceptance preserves the target distribution

A finite vocabulary `V`. The target law `p : V → ℝ` and the draft law `q : V → ℝ` are nonnegative
and sum to one. One speculative step draws `x ~ q`, accepts with probability `min 1 (p x / q x)`,
and on rejection draws from the residual `r y = max 0 (p y - q y) / Z` with the rejection mass
`Z = ∑ y, max 0 (p y - q y)` (Leviathan, Kalman and Matias, *Fast Inference from Transformers via
Speculative Decoding*, arXiv 2211.17192).

`specStep_eq_target` proves the output law `specStep p q` equals `p` for every nonnegative draft,
so the draft's error costs work but not quality. `accept_rate` identifies the acceptance
probability `∑ min (p x) (q x)` with `1 - (1/2) ∑ |p x - q x|`, the total-variation overlap.

Every definition is total: Lean's division by zero is zero, so `q x = 0` gives acceptance
probability `0` (the symbol is never drawn and its acceptance mass is `0`), and `Z = 0` gives
residual `0`. The identity `rejectMass p q * residual p q y = max 0 (p y - q y)` therefore holds
at `Z = 0` as well, because then `p ≤ q` pointwise.
-/

open Finset

namespace Nous
namespace Model

variable {V : Type*} [Fintype V]

/-- Acceptance probability of the draft token `x`: `min 1 (p x / q x)`. When `q x = 0` the ratio
`p x / 0` is `0` by Lean's total division, so the acceptance probability is `0`; `x` is never drawn
and its acceptance mass `q x * acceptProb p q x` is `0`. -/
noncomputable def acceptProb (p q : V → ℝ) (x : V) : ℝ := min 1 (p x / q x)

/-- Rejection mass `Z = ∑ y, max 0 (p y - q y)`. Temporal cost `|V|` additions, one per element
of the fold over `univ`; space `O(1)`. -/
noncomputable def rejectMass (p q : V → ℝ) : ℝ := ∑ y, max 0 (p y - q y)

/-- Residual law `r y = max 0 (p y - q y) / Z`. Total: at `Z = 0` division by zero returns `0`. -/
noncomputable def residual (p q : V → ℝ) (y : V) : ℝ := max 0 (p y - q y) / rejectMass p q

/-- Acceptance mass `∑ x, q x * acceptProb p q x`. Temporal cost `|V|` multiplications and
additions, one per element of the fold over `univ`; space `O(1)`. -/
noncomputable def acceptMass (p q : V → ℝ) : ℝ := ∑ x, q x * acceptProb p q x

/-- The output distribution of one speculative step: accepted draft mass `q y * acceptProb p q y`
plus the rejection mass `1 - acceptMass p q` times the residual law `residual p q y`. Temporal cost
`O(|V|)` per call through the `acceptMass` fold; space `O(1)`. -/
noncomputable def specStep (p q : V → ℝ) (y : V) : ℝ :=
  q y * acceptProb p q y + (1 - acceptMass p q) * residual p q y

omit [Fintype V] in
/-- **Acceptance identity.** For nonnegative `q`, `q x * min 1 (p x / q x) = min (q x) (p x)`:
the draft mass at `x` times its acceptance probability is the overlap of draft and target at `x`. -/
theorem q_mul_acceptProb_eq_min (p q : V → ℝ) (hp : ∀ x, 0 ≤ p x) (hq : ∀ x, 0 ≤ q x) (x : V) :
    q x * acceptProb p q x = min (q x) (p x) := by
  unfold acceptProb
  rcases eq_or_lt_of_le (hq x) with hq0 | hqpos
  · rw [← hq0, zero_mul, min_eq_left (hp x)]
  · rw [mul_min_of_nonneg 1 (p x / q x) (le_of_lt hqpos), mul_one,
      mul_div_cancel₀ (p x) (ne_of_gt hqpos)]

/-- **Pointwise reconstruction.** `min (q x) (p x) = p x - max 0 (p x - q x)`: the overlap is the
target minus the target's excess over the draft. -/
theorem min_swap_eq_sub_max (a b : ℝ) : min b a = a - max 0 (a - b) := by
  rcases le_total a b with h | h
  · rw [min_eq_right h, max_eq_left (sub_nonpos.mpr h)]; ring
  · rw [min_eq_left h, max_eq_right (sub_nonneg.mpr h)]; ring

/-- Summing pointwise reconstruction against `∑ p = 1` gives
`∑ min (q x) (p x) = 1 - rejectMass p q`. -/
theorem sum_min_eq_one_sub_rejectMass (p q : V → ℝ) (hps : ∑ x, p x = 1) :
    (∑ x, min (q x) (p x)) = 1 - rejectMass p q := by
  unfold rejectMass
  rw [sum_congr rfl (fun x _ => min_swap_eq_sub_max (p x) (q x)), sum_sub_distrib, hps]

/-- The acceptance mass is the overlap: `acceptMass p q = ∑ x, min (q x) (p x)`. -/
theorem acceptMass_eq_sum_min (p q : V → ℝ) (hp : ∀ x, 0 ≤ p x) (hq : ∀ x, 0 ≤ q x) :
    acceptMass p q = ∑ x, min (q x) (p x) := by
  unfold acceptMass
  exact sum_congr rfl (fun x _ => q_mul_acceptProb_eq_min p q hp hq x)

/-- **Acceptance mass.** `acceptMass p q = 1 - rejectMass p q`. -/
theorem acceptMass_eq_one_sub_rejectMass (p q : V → ℝ) (hp : ∀ x, 0 ≤ p x) (hq : ∀ x, 0 ≤ q x)
    (hps : ∑ x, p x = 1) : acceptMass p q = 1 - rejectMass p q := by
  rw [acceptMass_eq_sum_min p q hp hq, sum_min_eq_one_sub_rejectMass p q hps]

/-- **Residual scaling.** `rejectMass p q * residual p q y = max 0 (p y - q y)`, including at
`rejectMass p q = 0`, where the sum of the nonnegative excesses forces `p ≤ q` pointwise. -/
theorem residual_scale (p q : V → ℝ) (y : V) :
    rejectMass p q * residual p q y = max 0 (p y - q y) := by
  by_cases hZ : rejectMass p q = 0
  · have hnonneg : ∀ x ∈ (univ : Finset V), 0 ≤ max 0 (p x - q x) :=
      fun x _ => le_max_left 0 (p x - q x)
    have hle : max 0 (p y - q y) ≤ rejectMass p q := by
      unfold rejectMass
      exact single_le_sum hnonneg (mem_univ y)
    have hzero : max 0 (p y - q y) = 0 :=
      le_antisymm (hZ ▸ hle) (le_max_left 0 (p y - q y))
    rw [hZ, residual, hzero, zero_div, mul_zero]
  · rw [residual]
    exact mul_div_cancel₀ (max 0 (p y - q y)) hZ

/-- **Exactness.** The speculative step reproduces the target distribution for every nonnegative
draft `q`: `specStep p q = p`. -/
theorem specStep_eq_target (p q : V → ℝ) (hp : ∀ x, 0 ≤ p x) (hq : ∀ x, 0 ≤ q x)
    (hps : ∑ x, p x = 1) : specStep p q = p := by
  funext y
  unfold specStep
  rw [q_mul_acceptProb_eq_min p q hp hq y,
    acceptMass_eq_one_sub_rejectMass p q hp hq hps,
    min_swap_eq_sub_max (p y) (q y),
    show (1 : ℝ) - (1 - rejectMass p q) = rejectMass p q from by ring,
    residual_scale p q y]
  ring

/-- **Acceptance rate.** The acceptance probability is `1` minus the total variation:
`∑ x, min (p x) (q x) = 1 - (1 / 2) * ∑ x, |p x - q x|`. -/
theorem accept_rate (p q : V → ℝ) (hps : ∑ x, p x = 1) (hqs : ∑ x, q x = 1) :
    (∑ x, min (p x) (q x)) = 1 - (1 / 2) * ∑ x, |p x - q x| := by
  have hkey : ∀ x, max 0 (p x - q x) - max 0 (q x - p x) = p x - q x := by
    intro x
    rcases le_total (p x) (q x) with h | h
    · rw [max_eq_left (sub_nonpos.mpr h), max_eq_right (sub_nonneg.mpr h)]; ring
    · rw [max_eq_right (sub_nonneg.mpr h), max_eq_left (sub_nonpos.mpr h)]; ring
  have hsym : (∑ x, max 0 (p x - q x)) = ∑ x, max 0 (q x - p x) := by
    have hsub : (∑ x, max 0 (p x - q x)) - (∑ x, max 0 (q x - p x)) = 0 := by
      rw [← sum_sub_distrib, sum_congr rfl (fun x _ => hkey x), sum_sub_distrib, hps, hqs]
      ring
    linarith
  have habs : (∑ x, |p x - q x|) = 2 * rejectMass p q := by
    have hpt : ∀ x, |p x - q x| = max 0 (p x - q x) + max 0 (q x - p x) := by
      intro x
      rcases le_total (p x) (q x) with h | h
      · rw [abs_of_nonpos (sub_nonpos.mpr h), max_eq_left (sub_nonpos.mpr h),
            max_eq_right (sub_nonneg.mpr h)]; ring
      · rw [abs_of_nonneg (sub_nonneg.mpr h), max_eq_right (sub_nonneg.mpr h),
            max_eq_left (sub_nonpos.mpr h)]; ring
    rw [sum_congr rfl (fun x _ => hpt x), sum_add_distrib, ← hsym]
    unfold rejectMass
    ring
  have hmin : (∑ x, min (p x) (q x)) = 1 - rejectMass p q := by
    rw [← sum_min_eq_one_sub_rejectMass p q hps]
    exact sum_congr rfl (fun x _ => min_comm (p x) (q x))
  rw [hmin, habs]
  ring

/-! ## Two-token rational witness -/

namespace Witness

/-- Target law on two tokens: `(1/2, 1/2)`. -/
noncomputable def p2 : Fin 2 → ℝ := ![1 / 2, 1 / 2]

/-- Draft law on two tokens: `(3/4, 1/4)`. -/
noncomputable def q2 : Fin 2 → ℝ := ![3 / 4, 1 / 4]

/-- **Witness of exactness.** For the closed rational pair `p2 = (1/2, 1/2)`,
`q2 = (3/4, 1/4)`, the speculative step returns `p2`. Here the acceptance mass is `3/4`, the
rejection mass is `1/4`, the residual is `(0, 1)`, and `specStep p2 q2 = (1/2, 1/2)`. -/
theorem specStep_witness : specStep p2 q2 = p2 := by
  funext i
  fin_cases i <;>
    simp [specStep, acceptProb, acceptMass, residual, rejectMass, p2, q2, Fin.sum_univ_two] <;>
    norm_num

end Witness

end Model
end Nous
