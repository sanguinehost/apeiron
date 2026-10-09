import Apeiron.Decider.Wire

/-!
# Integer rows inside a residue class

@name Residue Normalization
@grade proved

For a row `2x + 3y - 1 ≥ 0`, write `y = 2z + s` with `s` equal to zero or one.
Inside either residue class the row becomes `x + 3z + (3s - 1) / 2 ≥ 0`.
The division is integer floor division, so the transformation is exact over integers.

The common modulus is divisible by every nonzero coefficient of the column to eliminate.
Substituting that modulus in all other columns lets each row divide by its own coefficient.
The resulting first coefficient is minus one, zero, or one. These are the coefficients for
which integer Fourier-Motzkin elimination is exact.

## Root

`Fragment.Row.Holds` supplies the integer row semantics rooted in
`Fragment.holds_iff_four_squares`. The transformation below preserves that same predicate in both
directions. It changes coordinates and divides integers; it introduces no algebraic primitive.
-/

namespace Apeiron.Decider

open Fragment

variable {n : ℕ}

/-- The magnitude of the first coefficient, using one for an absent coefficient.

    (r : Row (n + 1))                      [a row with a first column]
    : ℤ                                    [its positive normalization divisor] -/
def rowDivisor (r : Row (n + 1)) : ℤ := max 1 |r.coeff 0|

/-- The normalization divisor is positive.

    (r : Row (n + 1))                      [a row with a first column]
    0 < rowDivisor r                        [its divisor is above zero] -/
theorem rowDivisor_pos (r : Row (n + 1)) : 0 < rowDivisor r := by
  have := le_max_left (1 : ℤ) |r.coeff 0|
  unfold rowDivisor
  omega

/-- The normalization divisor divides the first coefficient.

    (r : Row (n + 1))                      [a row with a first column]
    rowDivisor r ∣ r.coeff 0                [the first coefficient divides exactly] -/
theorem rowDivisor_dvd (r : Row (n + 1)) : rowDivisor r ∣ r.coeff 0 := by
  by_cases h : r.coeff 0 = 0
  · simp [rowDivisor, h]
  · have hpos : (1 : ℤ) ≤ |r.coeff 0| := by
      have := abs_pos.mpr h
      omega
    simp only [rowDivisor, max_eq_right hpos]
    exact abs_dvd_self _

/-- Floor division preserves a bound whose variable part is divisible by the divisor.

    (c d z : ℤ)                            [a constant, divisor, and integer variable part]
    (hd : 0 < d)                           [the divisor is positive]
    0 ≤ c + d * z ↔ 0 ≤ c / d + z        [the bound before and after division agrees] -/
theorem floor_bound_iff (c d z : ℤ) (hd : 0 < d) :
    0 ≤ c + d * z ↔ 0 ≤ c / d + z := by
  have h := Int.le_ediv_iff_mul_le hd (a := -z) (b := c)
  constructor
  · intro hc
    have : -z * d ≤ c := by nlinarith
    have := h.mpr this
    omega
  · intro hc
    have : -z ≤ c / d := by omega
    have := h.mp this
    nlinarith

/-- Substitute a residue class in all tail columns and normalize the first coefficient.

    (m : ℤ)                               [the common modulus]
    (s : Fin n → ℤ)                       [one residue per remaining column]
    (r : Row (n + 1))                     [the input row]
    : Row (n + 1)                         [the normalized row] -/
def residueRow (m : ℤ) (s : Fin n → ℤ) (r : Row (n + 1)) : Row (n + 1) :=
  ⟨(r.const + ∑ i, r.coeff i.succ * s i) / rowDivisor r,
    Fin.cons (r.coeff 0 / rowDivisor r)
      (fun i => r.coeff i.succ * m / rowDivisor r)⟩

/-- The normalized first coefficient is minus one, zero, or one.

    (m : ℤ) (s : Fin n → ℤ)               [a modulus and a residue assignment]
    (r : Row (n + 1))                     [an input row]
    (residueRow m s r).coeff 0 = -1 ∨     [the result bounds the first column from above]
      (residueRow m s r).coeff 0 = 0 ∨    [or does not mention that column]
      (residueRow m s r).coeff 0 = 1      [or bounds it from below] -/
theorem residueRow_unit (m : ℤ) (s : Fin n → ℤ) (r : Row (n + 1)) :
    (residueRow m s r).coeff 0 = -1 ∨
      (residueRow m s r).coeff 0 = 0 ∨ (residueRow m s r).coeff 0 = 1 := by
  simp only [residueRow, Fin.cons_zero]
  rcases lt_trichotomy (r.coeff 0) 0 with h | h | h
  · left
    have hpos : (1 : ℤ) ≤ -(r.coeff 0) := by omega
    simp only [rowDivisor, abs_of_neg h, max_eq_right hpos]
    exact Int.ediv_eq_neg_one_of_neg_of_le h le_rfl
  · right; left
    simp [rowDivisor, h]
  · right; right
    have hpos : (1 : ℤ) ≤ r.coeff 0 := by omega
    simp [rowDivisor, abs_of_pos h, max_eq_right hpos, ne_of_gt h]

/-- Residue substitution and normalization preserve the exact row predicate.

    (m x : ℤ)                            [a common modulus and the first coordinate]
    (s z : Fin n → ℤ)                    [residues and quotient coordinates]
    (r : Row (n + 1))                    [the row being normalized]
    (hm : rowDivisor r ∣ m)              [the modulus divides by the row divisor]
    Row.Holds                            [the original row holds at the represented assignment]
      (Fin.cons x (fun i => m * z i + s i)) r
    ↔ Row.Holds (Fin.cons x z)           [exactly when the normalized row holds]
      (residueRow m s r) -/
theorem residueRow_holds_iff (m x : ℤ) (s z : Fin n → ℤ) (r : Row (n + 1))
    (hm : rowDivisor r ∣ m) :
    Row.Holds (Fin.cons x (fun i => m * z i + s i)) r ↔
      Row.Holds (Fin.cons x z) (residueRow m s r) := by
  let d := rowDivisor r
  let c := r.const + ∑ i, r.coeff i.succ * s i
  let v := r.coeff 0 / d * x + ∑ i, r.coeff i.succ * m / d * z i
  have hfirst : d * (r.coeff 0 / d) = r.coeff 0 :=
    Int.mul_ediv_cancel' (rowDivisor_dvd r)
  have htail (i : Fin n) : d * (r.coeff i.succ * m / d) = r.coeff i.succ * m :=
    Int.mul_ediv_cancel' (dvd_mul_of_dvd_right hm _)
  have hsum : (∑ i, r.coeff i.succ * (m * z i + s i)) =
      d * (∑ i, r.coeff i.succ * m / d * z i) + ∑ i, r.coeff i.succ * s i := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i _
    rw [← mul_assoc, htail]
    ring
  have hvalue : Row.value (Fin.cons x (fun i => m * z i + s i)) r = c + d * v := by
    simp only [Row.value, Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ, c, v]
    rw [hsum, mul_add, ← mul_assoc, hfirst]
    ring
  have hnormal : Row.value (Fin.cons x z) (residueRow m s r) = c / d + v := by
    simp [Row.value, residueRow, Fin.sum_univ_succ, c, d, v]
  simp only [Row.Holds, hvalue, hnormal]
  exact floor_bound_iff c d v (rowDivisor_pos r)

/-- A common positive multiple of all first-column normalization divisors.

    List (Row (n + 1)) → ℤ                [the rows and their common modulus] -/
def residueModulus : List (Row (n + 1)) → ℤ
  | [] => 1
  | r :: rs => rowDivisor r * residueModulus rs

/-- The common modulus is positive, including for an empty system.

    (rs : List (Row (n + 1)))              [a system with a first column]
    0 < residueModulus rs                   [the common modulus is above zero] -/
theorem residueModulus_pos (rs : List (Row (n + 1))) : 0 < residueModulus rs := by
  induction rs with
  | nil => simp [residueModulus]
  | cons r rs ih => exact mul_pos (rowDivisor_pos r) ih

/-- Every system row divides the common modulus by its normalization divisor.

    (rs : List (Row (n + 1))) (r : Row (n + 1)) [the system and one row]
    (hr : r ∈ rs)                             [the row belongs to the system]
    rowDivisor r ∣ residueModulus rs           [its divisor divides the common modulus] -/
theorem rowDivisor_dvd_residueModulus (rs : List (Row (n + 1))) (r : Row (n + 1))
    (hr : r ∈ rs) : rowDivisor r ∣ residueModulus rs := by
  induction rs with
  | nil => simp at hr
  | cons t ts ih =>
      rcases List.mem_cons.mp hr with h | h
      · subst t
        exact dvd_mul_right _ _
      · exact dvd_mul_of_dvd_right (ih h) _

/-- Normalizing every row preserves satisfaction inside the chosen residue class.

    (rs : List (Row (n + 1)))              [a system with a first column]
    (x : ℤ) (s z : Fin n → ℤ)             [the first coordinate, residues, and quotients]
    SatRows                               [the represented assignment satisfies the input]
      (Fin.cons x (fun i => residueModulus rs * z i + s i)) rs
    ↔ SatRows (Fin.cons x z)              [exactly when the quotients satisfy the normalized rows]
      (rs.map (residueRow (residueModulus rs) s)) -/
theorem satRows_residue_iff (rs : List (Row (n + 1))) (x : ℤ) (s z : Fin n → ℤ) :
    SatRows (Fin.cons x (fun i => residueModulus rs * z i + s i)) rs ↔
      SatRows (Fin.cons x z) (rs.map (residueRow (residueModulus rs) s)) := by
  constructor
  · intro h t ht
    obtain ⟨r, hr, rfl⟩ := List.mem_map.mp ht
    exact (residueRow_holds_iff _ _ _ _ r (rowDivisor_dvd_residueModulus rs r hr)).mp
      (h r hr)
  · intro h r hr
    exact (residueRow_holds_iff _ _ _ _ r (rowDivisor_dvd_residueModulus rs r hr)).mpr
      (h _ (List.mem_map.mpr ⟨r, hr, rfl⟩))

/-- All integer assignments are covered by the finitely bounded residue assignments.

    (rs : List (Row (n + 1)))              [a system with a first column]
    (∃ ρ, SatRows ρ rs)                    [some integer assignment satisfies the input]
    ↔ ∃ s : Fin n → ℤ,                   [exactly when some residue assignment]
      (∀ i, 0 ≤ s i ∧ s i < residueModulus rs) [lies within the common modulus]
      ∧ ∃ x z, SatRows (Fin.cons x z)     [and some first coordinate and quotients satisfy]
        (rs.map (residueRow (residueModulus rs) s)) [the normalized rows] -/
theorem exists_satRows_residue (rs : List (Row (n + 1))) :
    (∃ ρ, SatRows ρ rs) ↔ ∃ s : Fin n → ℤ,
      (∀ i, 0 ≤ s i ∧ s i < residueModulus rs) ∧
      ∃ x z, SatRows (Fin.cons x z) (rs.map (residueRow (residueModulus rs) s)) := by
  constructor
  · rintro ⟨ρ, hρ⟩
    let m := residueModulus rs
    let s : Fin n → ℤ := fun i => ρ i.succ % m
    let z : Fin n → ℤ := fun i => ρ i.succ / m
    have hm : 0 < m := residueModulus_pos rs
    refine ⟨s, fun i => ⟨Int.emod_nonneg _ (ne_of_gt hm), Int.emod_lt_of_pos _ hm⟩,
      ρ 0, z, ?_⟩
    apply (satRows_residue_iff rs (ρ 0) s z).mp
    have heq : Fin.cons (ρ 0) (fun i => m * z i + s i) = ρ := by
      funext i
      refine Fin.cases rfl (fun j => ?_) i
      simp only [Fin.cons_succ, z, s]
      exact Int.mul_ediv_add_emod (ρ j.succ) m
    change SatRows (Fin.cons (ρ 0) (fun i => m * z i + s i)) rs
    rw [heq]
    exact hρ
  · rintro ⟨s, _, x, z, h⟩
    exact ⟨_, (satRows_residue_iff rs x s z).mpr h⟩

-- Both residues are needed: odd values of the tail coordinate change the rounded constant.
#guard (residueRow 2 (fun _ : Fin 1 => 0)
  (⟨-1, ![2, 3]⟩ : Row 2)).const = -1
#guard (residueRow 2 (fun _ : Fin 1 => 1)
  (⟨-1, ![2, 3]⟩ : Row 2)).const = 1
#guard (residueRow 2 (fun _ : Fin 1 => 0)
  (⟨-1, ![-2, 3]⟩ : Row 2)).coeff 0 = -1

end Apeiron.Decider
