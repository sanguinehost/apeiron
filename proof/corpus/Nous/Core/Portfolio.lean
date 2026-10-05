import Mathlib

/-!
# Complementary portfolio coverage

LACE's complementary portfolio cost (its CPI) is the mean over instances of the cost of the best
candidate in the selected portfolio. `cpi` is that quantity summed rather than averaged, which
keeps the statement in `Nat`. Adding a candidate never worsens coverage; equal-cost candidates are
interchangeable; and a candidate that is strictly cheapest on some instance strictly improves
coverage. The empty portfolio is excluded by an explicit `Nonempty` hypothesis, never by a
sentinel.
-/

namespace Nous.Core

variable {m n : Nat}

/-- The complementary portfolio cost: over each instance, the cost of the best selected
candidate, summed over the instances. -/
def cpi (R : Fin m → Fin n → Nat) (S : Finset (Fin n)) (hS : S.Nonempty) : Nat :=
  ∑ ξ, (S.image (fun h => R ξ h)).min' (hS.image _)

/-- Adding a candidate never worsens coverage. -/
theorem cpi_le_of_subset (R : Fin m → Fin n → Nat) {S T : Finset (Fin n)}
    (hS : S.Nonempty) (hT : T.Nonempty) (hST : S ⊆ T) : cpi R T hT ≤ cpi R S hS := by
  apply Finset.sum_le_sum
  intro ξ _
  obtain ⟨h₀, hh₀, hval⟩ :=
    Finset.mem_image.mp (Finset.min'_mem (S.image (fun h => R ξ h)) (hS.image _))
  exact Finset.min'_le _ _ (Finset.mem_image.mpr ⟨h₀, hST hh₀, hval⟩)

/-- Candidates whose columns agree on every instance are interchangeable in the portfolio. -/
theorem cpi_insert_eq_of_equal_columns (R : Fin m → Fin n → Nat) {S : Finset (Fin n)}
    {a b : Fin n} (h : ∀ ξ, R ξ a = R ξ b) :
    cpi R (insert a S) ⟨a, Finset.mem_insert_self a S⟩ =
      cpi R (insert b S) ⟨b, Finset.mem_insert_self b S⟩ := by
  apply Finset.sum_congr rfl
  intro ξ _
  have himg : (insert a S).image (fun h => R ξ h) = (insert b S).image (fun h => R ξ h) := by
    rw [Finset.image_insert, Finset.image_insert, h ξ]
  simp only [himg]

/-- A candidate that is strictly cheapest on some instance strictly improves coverage. -/
theorem cpi_insert_lt_of_strict (R : Fin m → Fin n → Nat) {S : Finset (Fin n)}
    (hS : S.Nonempty) {a : Fin n} (ξ₀ : Fin m)
    (h : R ξ₀ a < (S.image (fun h => R ξ₀ h)).min' (hS.image _)) :
    cpi R (insert a S) ⟨a, Finset.mem_insert_self a S⟩ < cpi R S hS := by
  apply Finset.sum_lt_sum
  · intro ξ _
    obtain ⟨h₀, hh₀, hval⟩ :=
      Finset.mem_image.mp (Finset.min'_mem (S.image (fun h => R ξ h)) (hS.image _))
    exact Finset.min'_le _ _ (Finset.mem_image.mpr ⟨h₀, Finset.mem_insert_of_mem hh₀, hval⟩)
  · refine ⟨ξ₀, Finset.mem_univ _, ?_⟩
    have hmem : R ξ₀ a ∈ (insert a S).image (fun h => R ξ₀ h) :=
      Finset.mem_image.mpr ⟨a, Finset.mem_insert_self a S, rfl⟩
    exact lt_of_le_of_lt (Finset.min'_le _ _ hmem) h

end Nous.Core
