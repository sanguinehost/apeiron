import Apeiron.Decider.Wire

/-!
# Exact unit-coefficient projection

`unitProject` removes the first column of a row system when every coefficient in that
column is `-1`, `0`, or `1`.  Zero rows are copied without their first coefficient; every
lower/upper pair is added after that coefficient cancels.  The pair rows are necessary: they
state that every lower bound is below every upper bound.  Since those bounds are integral, a
maximal lower bound is an integral witness whenever one exists; with no lower bound, a minimal
upper bound (or zero) is a witness.
-/

namespace Apeiron
namespace Decider

open Fragment

variable {n : ℕ}

/-- Remove the first coefficient of a row.

    Row [a whole-number linear inequality row] -/
def tailRow (r : Row (n + 1)) : Row n :=
  ⟨r.const, fun i => r.coeff i.succ⟩

/-- The value of a row at a head/tail assignment.

    Row [a whole-number linear inequality row]
    → [takes the left and gives the right] -/
theorem value_cons (x : ℤ) (ρ : Fin n → ℤ) (r : Row (n + 1)) :
    Row.value (Fin.cons x ρ) r = r.coeff 0 * x + Row.value ρ (tailRow r) := by
  simp only [Row.value, tailRow, Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
  ring

/-- Rows whose first coefficient supplies a lower bound on the eliminated variable.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row] -/
def unitLowers (rs : List (Row (n + 1))) : List (Row (n + 1)) :=
  rs.filter fun r => decide (r.coeff 0 = 1)

/-- Rows whose first coefficient supplies an upper bound on the eliminated variable.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row] -/
def unitUppers (rs : List (Row (n + 1))) : List (Row (n + 1)) :=
  rs.filter fun r => decide (r.coeff 0 = -1)

/-- Rows that do not mention the eliminated variable.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row] -/
def unitZeros (rs : List (Row (n + 1))) : List (Row (n + 1)) :=
  rs.filter fun r => decide (r.coeff 0 = 0)

/-- The first coefficient has exactly the unit cases supported by this projection.

    Row [a whole-number linear inequality row] -/
def UnitHead (r : Row (n + 1)) : Prop :=
  r.coeff 0 = -1 ∨ r.coeff 0 = 0 ∨ r.coeff 0 = 1

/-- A system lies in the exact unit-coefficient projection fragment.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row] -/
def UnitHeads (rs : List (Row (n + 1))) : Prop := ∀ r ∈ rs, UnitHead r

/-- The lower/upper pair with the first coefficient cancelled.

    Row [a whole-number linear inequality row] -/
def unitPair (l u : Row (n + 1)) : Row n := Row.add (tailRow l) (tailRow u)

/-- Exact Fourier--Motzkin projection for a unit first coefficient.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row] -/
def unitProject (rs : List (Row (n + 1))) : List (Row n) :=
  (unitZeros rs).map tailRow ++
    (unitLowers rs).flatMap fun l => (unitUppers rs).map fun u => unitPair l u

/-- Membership in the lower partition is membership in the source system with head `1`.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row]
    ↔ [exactly when, true both directions]
    ∧ [both are true]
    ∈ [is one of the things in] -/
theorem mem_unitLowers_iff {rs : List (Row (n + 1))} {r : Row (n + 1)} :
    r ∈ unitLowers rs ↔ r ∈ rs ∧ r.coeff 0 = 1 := by
  simp [unitLowers]

/-- Membership in the upper partition is membership in the source system with head `-1`.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row]
    ↔ [exactly when, true both directions]
    ∧ [both are true]
    ∈ [is one of the things in] -/
theorem mem_unitUppers_iff {rs : List (Row (n + 1))} {r : Row (n + 1)} :
    r ∈ unitUppers rs ↔ r ∈ rs ∧ r.coeff 0 = -1 := by
  simp [unitUppers]

/-- Membership in the zero partition is membership in the source system with head `0`.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row]
    ↔ [exactly when, true both directions]
    ∧ [both are true]
    ∈ [is one of the things in] -/
theorem mem_unitZeros_iff {rs : List (Row (n + 1))} {r : Row (n + 1)} :
    r ∈ unitZeros rs ↔ r ∈ rs ∧ r.coeff 0 = 0 := by
  simp [unitZeros]

/-- A solution of the original system satisfies its unit projection.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row]
    → [takes the left and gives the right] -/
theorem satRows_unitProject {x : ℤ} {ρ : Fin n → ℤ} {rs : List (Row (n + 1))}
    (hs : SatRows (Fin.cons x ρ) rs) : SatRows ρ (unitProject rs) := by
  intro q hq
  rw [unitProject, List.mem_append] at hq
  rcases hq with hzero | hpair
  · rw [List.mem_map] at hzero
    obtain ⟨r, hr, rfl⟩ := hzero
    have hhead : r.coeff 0 = 0 := (mem_unitZeros_iff.mp hr).2
    have horig := hs r (mem_unitZeros_iff.mp hr).1
    rw [Row.Holds, value_cons] at horig
    simpa [Row.Holds, hhead] using horig
  · simp only [List.mem_flatMap, List.mem_map] at hpair
    obtain ⟨l, hl, u, hu, rfl⟩ := hpair
    have hlhead : l.coeff 0 = 1 := (mem_unitLowers_iff.mp hl).2
    have huhead : u.coeff 0 = -1 := (mem_unitUppers_iff.mp hu).2
    have hlorig := hs l (mem_unitLowers_iff.mp hl).1
    have huorig := hs u (mem_unitUppers_iff.mp hu).1
    rw [Row.Holds, value_cons] at hlorig huorig
    simp only [hlhead, huhead, one_mul, neg_one_mul] at hlorig huorig
    rw [Row.Holds, unitPair, Row.value_add]
    omega

/-- The lower bound a positive-head row places on the eliminated variable.

    Row [a whole-number linear inequality row]
    → [takes the left and gives the right] -/
def lowerBound (ρ : Fin n → ℤ) (r : Row (n + 1)) : ℤ := -Row.value ρ (tailRow r)

/-- The upper bound a negative-head row places on the eliminated variable.

    Row [a whole-number linear inequality row]
    → [takes the left and gives the right] -/
def upperBound (ρ : Fin n → ℤ) (r : Row (n + 1)) : ℤ := Row.value ρ (tailRow r)

/-- Every emitted lower/upper pair orders its two bounds.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row]
    → [takes the left and gives the right]
    ∈ [is one of the things in]
    ≤ [is smaller than or equal to] -/
theorem pair_bound {ρ : Fin n → ℤ} {rs : List (Row (n + 1))}
    (hs : SatRows ρ (unitProject rs)) {l u : Row (n + 1)}
    (hl : l ∈ unitLowers rs) (hu : u ∈ unitUppers rs) :
    lowerBound ρ l ≤ upperBound ρ u := by
  have hp : unitPair l u ∈ unitProject rs := by
    rw [unitProject, List.mem_append]
    right
    rw [List.mem_flatMap]
    exact ⟨l, hl, List.mem_map.mpr ⟨u, hu, rfl⟩⟩
  have hh := hs (unitPair l u) hp
  rw [Row.Holds, unitPair, Row.value_add] at hh
  simp only [lowerBound, upperBound]
  omega

/-- A copied zero-head row holds for every extension of a projected solution.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row]
    → [takes the left and gives the right]
    ∈ [is one of the things in] -/
theorem sat_zero_of_unitProject {x : ℤ} {ρ : Fin n → ℤ} {rs : List (Row (n + 1))}
    (hs : SatRows ρ (unitProject rs)) {r : Row (n + 1)}
    (hr : r ∈ rs) (hhead : r.coeff 0 = 0) : Row.Holds (Fin.cons x ρ) r := by
  have hz : tailRow r ∈ unitProject rs := by
    rw [unitProject, List.mem_append]
    left
    rw [List.mem_map]
    exact ⟨r, mem_unitZeros_iff.mpr ⟨hr, hhead⟩, rfl⟩
  have hh := hs (tailRow r) hz
  rw [Row.Holds, value_cons]
  simpa [Row.Holds, hhead] using hh

/-- A lower bound supplies a satisfying extension for a positive-head row.

    Row [a whole-number linear inequality row]
    → [takes the left and gives the right]
    ≤ [is smaller than or equal to] -/
theorem sat_lower_of_bound {x : ℤ} {ρ : Fin n → ℤ} {r : Row (n + 1)}
    (hhead : r.coeff 0 = 1) (hbound : lowerBound ρ r ≤ x) :
    Row.Holds (Fin.cons x ρ) r := by
  rw [Row.Holds, value_cons]
  simp only [hhead, one_mul]
  simp only [lowerBound] at hbound
  omega

/-- An upper bound supplies a satisfying extension for a negative-head row.

    Row [a whole-number linear inequality row]
    → [takes the left and gives the right]
    ≤ [is smaller than or equal to] -/
theorem sat_upper_of_bound {x : ℤ} {ρ : Fin n → ℤ} {r : Row (n + 1)}
    (hhead : r.coeff 0 = -1) (hbound : x ≤ upperBound ρ r) :
    Row.Holds (Fin.cons x ρ) r := by
  rw [Row.Holds, value_cons]
  simp only [hhead, neg_one_mul]
  simp only [upperBound] at hbound
  omega

/-- A projected solution extends to a solution of every unit-head input system.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row]
    ∃ [there is at least one]
    → [takes the left and gives the right] -/
theorem satRows_extend_unitProject {ρ : Fin n → ℤ} {rs : List (Row (n + 1))}
    (hunit : UnitHeads rs) (hs : SatRows ρ (unitProject rs)) :
    ∃ x : ℤ, SatRows (Fin.cons x ρ) rs := by
  by_cases hl : unitLowers rs = []
  · by_cases hu : unitUppers rs = []
    · refine ⟨0, ?_⟩
      intro r hr
      rcases hunit r hr with hneg | hzero | hpos
      · have : r ∈ unitUppers rs := mem_unitUppers_iff.mpr ⟨hr, hneg⟩
        rw [hu] at this
        simp at this
      · exact sat_zero_of_unitProject hs hr hzero
      · have : r ∈ unitLowers rs := mem_unitLowers_iff.mpr ⟨hr, hpos⟩
        rw [hl] at this
        simp at this
    · let ns : List ℤ := (unitUppers rs).map fun r => -upperBound ρ r
      have hns : ns.toFinset.Nonempty := by
        obtain ⟨u, hu'⟩ : ∃ u, u ∈ unitUppers rs := by
          cases hU : unitUppers rs with
          | nil => exact False.elim (hu hU)
          | cons u us => exact ⟨u, by simp⟩
        refine ⟨-upperBound ρ u, ?_⟩
        exact List.mem_toFinset.mpr (List.mem_map.mpr ⟨u, hu', rfl⟩)
      obtain ⟨y, hy, _⟩ := (Fragment.unit_elimination_exact ns.toFinset ∅ hns).mpr (by simp)
      refine ⟨-y, ?_⟩
      intro r hr
      rcases hunit r hr with hneg | hzero | hpos
      · have hurow : r ∈ unitUppers rs := mem_unitUppers_iff.mpr ⟨hr, hneg⟩
        have hymem : -upperBound ρ r ∈ ns.toFinset := by
          exact List.mem_toFinset.mpr (List.mem_map.mpr ⟨r, hurow, rfl⟩)
        have hyu : -upperBound ρ r ≤ y := hy _ hymem
        exact sat_upper_of_bound hneg (by omega : -y ≤ upperBound ρ r)
      · exact sat_zero_of_unitProject hs hr hzero
      · have : r ∈ unitLowers rs := mem_unitLowers_iff.mpr ⟨hr, hpos⟩
        rw [hl] at this
        simp at this
  · let ls : List ℤ := (unitLowers rs).map (lowerBound ρ)
    let us : List ℤ := (unitUppers rs).map (upperBound ρ)
    have hls : ls.toFinset.Nonempty := by
      obtain ⟨l, hl'⟩ : ∃ l, l ∈ unitLowers rs := by
        cases hL : unitLowers rs with
        | nil => exact False.elim (hl hL)
        | cons l ls => exact ⟨l, by simp⟩
      refine ⟨lowerBound ρ l, ?_⟩
      exact List.mem_toFinset.mpr (List.mem_map.mpr ⟨l, hl', rfl⟩)
    have hbetween : ∀ l ∈ ls.toFinset, ∀ u ∈ us.toFinset, l ≤ u := by
      intro l hlmem u humem
      rw [List.mem_toFinset] at hlmem humem
      obtain ⟨lrow, hll, rfl⟩ := List.mem_map.mp hlmem
      obtain ⟨urow, huu, rfl⟩ := List.mem_map.mp humem
      exact pair_bound hs hll huu
    obtain ⟨x, hxlo, hxup⟩ :=
      (Fragment.unit_elimination_exact ls.toFinset us.toFinset hls).mpr hbetween
    refine ⟨x, ?_⟩
    intro r hr
    rcases hunit r hr with hneg | hzero | hpos
    · have hurow : r ∈ unitUppers rs := mem_unitUppers_iff.mpr ⟨hr, hneg⟩
      have humem : upperBound ρ r ∈ us.toFinset := by
        exact List.mem_toFinset.mpr (List.mem_map.mpr ⟨r, hurow, rfl⟩)
      exact sat_upper_of_bound hneg (hxup _ humem)
    · exact sat_zero_of_unitProject hs hr hzero
    · have hlrow : r ∈ unitLowers rs := mem_unitLowers_iff.mpr ⟨hr, hpos⟩
      have hlmem : lowerBound ρ r ∈ ls.toFinset := by
        exact List.mem_toFinset.mpr (List.mem_map.mpr ⟨r, hlrow, rfl⟩)
      exact sat_lower_of_bound hpos (hxlo _ hlmem)

/-- Unit-coefficient projection preserves and reflects integer satisfiability exactly.

    List [a finite sequence of things, in order]
    Row [a whole-number linear inequality row]
    ↔ [exactly when, true both directions]
    ∃ [there is at least one]
    → [takes the left and gives the right] -/
theorem satRows_unitProject_iff {ρ : Fin n → ℤ} {rs : List (Row (n + 1))}
    (hunit : UnitHeads rs) :
    (∃ x : ℤ, SatRows (Fin.cons x ρ) rs) ↔ SatRows ρ (unitProject rs) := by
  constructor
  · rintro ⟨x, hs⟩
    exact satRows_unitProject hs
  · exact satRows_extend_unitProject hunit

end Decider
end Apeiron
