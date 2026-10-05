import Mathlib

/-!
# Complementarity of two candidates

LACE measures the complementarity of two candidates by the mean absolute difference of their
per-instance ranks. `rankDistance` is that absolute difference summed over the instances; the
mean is this quantity divided by the instance count. It is a pseudometric: symmetric, zero on the
diagonal, satisfying the triangle inequality, and zero exactly between candidates that order every
instance identically.
-/

namespace Nous.Core

variable {m n : Nat}

/-- Sum over instances of the absolute difference of the ranks of candidates `a` and `b`. -/
def rankDistance (R : Fin m → Fin n → ℤ) (a b : Fin n) : Nat :=
  ∑ ξ, (R ξ a - R ξ b).natAbs

theorem rankDistance_symm (R : Fin m → Fin n → ℤ) (a b : Fin n) :
    rankDistance R a b = rankDistance R b a := by
  simp only [rankDistance]
  apply Finset.sum_congr rfl
  intro ξ _
  rw [← Int.natAbs_neg (R ξ a - R ξ b), neg_sub]

theorem rankDistance_self (R : Fin m → Fin n → ℤ) (a : Fin n) : rankDistance R a a = 0 := by
  simp [rankDistance]

theorem rankDistance_triangle (R : Fin m → Fin n → ℤ) (a b c : Fin n) :
    rankDistance R a c ≤ rankDistance R a b + rankDistance R b c := by
  simp only [rankDistance]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro ξ _
  have h : R ξ a - R ξ c = (R ξ a - R ξ b) + (R ξ b - R ξ c) := by ring
  rw [h]
  exact Int.natAbs_add_le _ _

theorem rankDistance_eq_zero_iff (R : Fin m → Fin n → ℤ) (a b : Fin n) :
    rankDistance R a b = 0 ↔ ∀ ξ, R ξ a = R ξ b := by
  simp only [rankDistance, Finset.sum_eq_zero_iff, Finset.mem_univ, true_implies,
    Int.natAbs_eq_zero, sub_eq_zero]

end Nous.Core
