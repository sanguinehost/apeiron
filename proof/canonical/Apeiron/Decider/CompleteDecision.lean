import Apeiron.Decider.Residue
import Apeiron.Decider.UnitProjection

/-!
# Complete integer-row decision by residue projection

@name Complete Integer-Row Decision
@grade proved

The equation `2x + y = 1` has solutions in the odd residue class of its second coordinate.
Trying only the zero residue would lose those solutions. `residueChoices` enumerates every
residue of every remaining column, and `completeSat` tests the projection in each branch.

Each branch normalizes the first coefficient to minus one, zero, or one, then removes that
column with exact integer Fourier-Motzkin projection. Recursion decreases the number of columns.
At zero columns the rows are constant inequalities, checked directly. The procedure is total
over arbitrary integer coefficients; the size of its finite residue search has no stated bound.

## Root

`exists_satRows_residue` preserves the integer row predicate through residue substitution.
`satRows_unitProject_iff` removes one normalized column without losing solutions, using
`Fragment.unit_elimination_exact`. The resulting decision theorem uses the existing
`Fragment.Row.Holds` semantics rooted in `Fragment.holds_iff_four_squares`.
-/

namespace Apeiron.Decider

open Fragment

/-- Enumerate all residue assignments with coordinates from zero through modulus minus one.

    (m : ℤ)                               [the common modulus]
    (n : ℕ)                               [the number of remaining columns]
    : List (Fin n → ℤ)                    [every residue assignment] -/
def residueChoices (m : ℤ) : (n : ℕ) → List (Fin n → ℤ)
  | 0 => [Fin.elim0]
  | n + 1 => (List.range m.toNat).flatMap fun (k : ℕ) =>
      (residueChoices m n).map (Fin.cons (k : ℤ))

/-- Membership in the residue enumeration is exactly the coordinate bound.

    (m : ℤ) (n : ℕ) (s : Fin n → ℤ)       [a modulus, dimension, and residue assignment]
    s ∈ residueChoices m n                 [the enumeration contains the assignment]
    ↔ ∀ i, 0 ≤ s i ∧ s i < m             [exactly when every coordinate is a residue] -/
theorem mem_residueChoices_iff (m : ℤ) (n : ℕ) (s : Fin n → ℤ) :
    s ∈ residueChoices m n ↔ ∀ i, 0 ≤ s i ∧ s i < m := by
  induction n with
  | zero =>
      simp only [residueChoices, List.mem_singleton]
      constructor
      · intro _ i
        exact i.elim0
      · intro _
        funext i
        exact i.elim0
  | succ n ih =>
      simp only [residueChoices, List.mem_flatMap, List.mem_map, List.mem_range]
      constructor
      · rintro ⟨k, hk, t, ht, rfl⟩ i
        refine Fin.cases ?_ (fun j => ?_) i
        · simp only [Fin.cons_zero]
          constructor
          · exact Int.natCast_nonneg _
          · omega
        · simpa only [Fin.cons_succ] using (ih t).mp ht j
      · intro h
        have hzero := h 0
        refine ⟨(s 0).toNat, by omega, (fun i => s i.succ), ?_, ?_⟩
        · exact (ih _).mpr fun i => h i.succ
        · funext i
          refine Fin.cases ?_ (fun j => ?_) i
          · simpa only [Fin.cons_zero] using Int.toNat_of_nonneg hzero.1
          · rfl

/-- Decide integer-row satisfiability by reducing the number of columns at every branch.

    (n : ℕ)                               [the number of columns]
    List (Row n) → Bool                    [the input rows and their satisfiability result] -/
def completeSat : (n : ℕ) → List (Row n) → Bool
  | 0, rs => rs.all fun r => decide (0 ≤ r.const)
  | n + 1, rs => (residueChoices (residueModulus rs) n).any fun s =>
      completeSat n (unitProject (rs.map (residueRow (residueModulus rs) s)))

/-- Every residue-normalized system satisfies the unit projection's joint hypothesis.

    {n : ℕ}                               [the remaining number of columns]
    (m : ℤ) (s : Fin n → ℤ)               [a modulus and a residue assignment]
    (rs : List (Row (n + 1)))              [the input system]
    UnitHeads (rs.map (residueRow m s))    [every first coefficient is minus one, zero, or one] -/
theorem unitHeads_residueRows {n : ℕ} (m : ℤ) (s : Fin n → ℤ)
    (rs : List (Row (n + 1))) : UnitHeads (rs.map (residueRow m s)) := by
  intro r hr
  obtain ⟨q, _, rfl⟩ := List.mem_map.mp hr
  exact residueRow_unit m s q

/-- The executing decision is exact for every finite integer-row system.

    (n : ℕ) (rs : List (Row n))            [the number of columns and input system]
    completeSat n rs = true                 [the procedure reports satisfiable]
    ↔ ∃ ρ : Fin n → ℤ, SatRows ρ rs       [exactly when some integer assignment satisfies it] -/
theorem completeSat_eq_true_iff (n : ℕ) (rs : List (Row n)) :
    completeSat n rs = true ↔ ∃ ρ : Fin n → ℤ, SatRows ρ rs := by
  induction n with
  | zero =>
      simp only [completeSat, List.all_eq_true, decide_eq_true_eq]
      constructor
      · intro h
        refine ⟨Fin.elim0, fun r hr => ?_⟩
        simpa [Row.Holds, Row.value] using h r hr
      · rintro ⟨ρ, h⟩ r hr
        simpa [Row.Holds, Row.value] using h r hr
  | succ n ih =>
      rw [completeSat, List.any_eq_true, exists_satRows_residue]
      constructor
      · rintro ⟨s, hs, hsat⟩
        obtain ⟨z, hz⟩ := (ih _).mp hsat
        obtain ⟨x, hx⟩ := (satRows_unitProject_iff (unitHeads_residueRows _ s rs)).mpr hz
        exact ⟨s, (mem_residueChoices_iff _ _ s).mp hs, x, z, hx⟩
      · rintro ⟨s, hs, x, z, hsat⟩
        refine ⟨s, (mem_residueChoices_iff _ _ s).mpr hs, (ih _).mpr ?_⟩
        exact ⟨z, (satRows_unitProject_iff (unitHeads_residueRows _ s rs)).mp ⟨x, hsat⟩⟩

/-- A negative result is exact integer infeasibility.

    (n : ℕ) (rs : List (Row n))            [the number of columns and input system]
    completeSat n rs = false                [the procedure reports unsatisfiable]
    ↔ ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs     [exactly when no integer assignment satisfies it] -/
theorem completeSat_eq_false_iff (n : ℕ) (rs : List (Row n)) :
    completeSat n rs = false ↔ ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs := by
  rw [← completeSat_eq_true_iff]
  exact Bool.eq_false_iff

-- The equality requires the nonzero residue of the second coordinate.
#guard completeSat 2 [⟨-1, ![2, 1]⟩, ⟨1, ![-2, -1]⟩]
-- An odd right-hand side cannot equal twice an integer.
#guard !completeSat 1 [⟨-1, ![2]⟩, ⟨1, ![-2]⟩]
#guard completeSat 1 [⟨-2, ![2]⟩, ⟨2, ![-2]⟩]
-- The earlier propagation procedure's false-SAT witness has no integer solution.
#guard !completeSat 2 [⟨-3, ![4, 6]⟩, ⟨3, ![-4, -6]⟩]
#guard completeSat 0 []
#guard !completeSat 0 [⟨-1, Fin.elim0⟩]

end Apeiron.Decider
