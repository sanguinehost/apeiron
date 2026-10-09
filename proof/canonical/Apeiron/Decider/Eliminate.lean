import Apeiron.Decider.Propagate

/-!
# Phase two: Fourier-Motzkin with integer tightening

@name Fourier-Motzkin
@grade proved
@cite farkas-1902

When propagation settles without a crossing, `../song/src/prover/cone/ncone0.slang` picks a
column and projects it away. Every row bounding the column from below is paired with every row
bounding it from above, the pair is combined so the column cancels, and the result is tightened
by the greatest common divisor of its coefficients. This file is that phase, and the soundness of
each operation: every derived row is implied by its parents over the whole numbers.

## The combination is Farkas, and it is already proved

A lower row has a positive coefficient `p` on the column and an upper row a negative one `-q`.
Multiplying the first by `q` and the second by `p` and adding cancels the column. Both multipliers
are positive, so the result is a nonnegative combination of two rows that hold, which is
`Fragment.combo_holds`, which is `../song/src/prover/cone/arith.slang`'s `combovalid_nonneg` and
`farkas_bound`. `combineFM` is written as `Fragment.combo` of a two-element list for exactly that
reason: the soundness is not restated here, it is applied. ncone0 divides `p` and `q` by their
own greatest common divisor first, which keeps intermediates small and changes nothing about the
cancellation; `combineFM_cancels` proves the column leaves.

## The tightening is a cutting plane, and it is not a combination

If every coefficient of `c + a·x ≥ 0` is divisible by `g`, then `a·x` is `g` times a whole number
at every integer assignment, so the row may be divided through and the constant rounded inward.
That step is sound over the whole numbers and false over the rationals, which is why it is not a
Farkas combination and why a certificate needs a `t` operation of its own. `tidyRow_holds` is the
step and `tidy_four_squares` puts its conclusion in the channel's own ring.

## Root

| from | here |
|---|---|
| `Fragment.combo_holds` (from `combovalid_nonneg`, `farkas_bound`) | `combineFM_holds` |
| `int_le_iff_exists_sum_four_sq` | `tidy_four_squares` |
| `Fragment.holds_iff_four_squares` | `eliminate_four_squares` |
| `face_is_forced`, `disc_one_of_integer_basis` | the face these rows are read on |

The cutting plane is where the whole numbers, rather than the reals, decide the answer, and
`int_le_iff_exists_sum_four_sq` is the corpus theorem that says the integer channel carries its
own order: `a ≤ b` exactly when `b - a` is four squares summed. `two_not_sq` and
`seven_not_sum_three_sq` are why the count is four, so this is the integer statement and not the
real one restated. `tidy_four_squares` states the tightened row's conclusion in that form, with
no order symbol in it.
-/

namespace Apeiron
namespace Decider

open Trichotomy Fragment

variable {n : ℕ}

/-- The Fourier-Motzkin combination of a lower row and an upper row.

    (v : Fin n)                              [the column being projected away]
    (l u : Row n)                            [a row bounding it below, and one bounding it above]
    : Row n                                  [the row with that column cancelled]

    So: `combineOne` in ncone0, multipliers reduced by their own greatest common divisor before
    multiplying. Written as `Fragment.combo` of two weighted rows, because that is what it is. -/
def combineFM (v : Fin n) (l u : Row n) : Row n :=
  let p := l.coeff v
  let q := -(u.coeff v)
  let g : ℤ := (Int.gcd p q : ℕ)
  Fragment.combo [(q / g, l), (p / g, u)]

/-- The combination of two rows that hold is a row that holds.

    {ρ : Fin n → ℤ} {l u : Row n} {v : Fin n}    [an assignment, two rows, a column]
    0 < l.coeff v                                [the first bounds the column from below]
    u.coeff v < 0                                [the second bounds it from above]
    Row.Holds ρ l                                [the first holds there]
    Row.Holds ρ u                                [the second holds there]
    Row.Holds ρ (combineFM v l u)                [so does their combination]

    So: `Fragment.combo_holds` applied, not restated. Both multipliers come out of a division of
    a positive number by a positive divisor, so both are at least zero, which is the only thing
    Farkas needs. -/
theorem combineFM_holds {ρ : Fin n → ℤ} {l u : Row n} {v : Fin n}
    (hl : 0 < l.coeff v) (hu : u.coeff v < 0)
    (hlh : Row.Holds ρ l) (huh : Row.Holds ρ u) :
    Row.Holds ρ (combineFM v l u) := by
  have hg : (0 : ℤ) < (Int.gcd (l.coeff v) (-(u.coeff v)) : ℕ) := by
    have : Int.gcd (l.coeff v) (-(u.coeff v)) ≠ 0 := by
      simp only [ne_eq, Int.gcd_eq_zero_iff, not_and]
      intro h
      omega
    omega
  refine Fragment.combo_holds ρ _ ?_ ?_
  · intro pr hpr
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hpr
    rcases hpr with rfl | rfl
    · exact Int.ediv_nonneg (by omega) (by omega)
    · exact Int.ediv_nonneg (by omega) (by omega)
  · intro pr hpr
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hpr
    rcases hpr with rfl | rfl
    · exact hlh
    · exact huh

/-- Two numbers divided by their common factor, crossed and subtracted, cancel.

    (p q : ℤ)                                            [two whole numbers]
    (Int.gcd p q : ℕ)                                    [their greatest common divisor]
    q / (Int.gcd p q) * p - p / (Int.gcd p q) * q = 0     [the cross difference is zero]

    So: the arithmetic behind the elimination. `combineOne` in ncone0 divides both multipliers
    by their greatest common divisor before multiplying, which keeps intermediates clear of the
    overflow guard and leaves the cancellation exact. -/
theorem gcd_cross (p q : ℤ) :
    (q / ((Int.gcd p q : ℕ) : ℤ)) * p - (p / ((Int.gcd p q : ℕ) : ℤ)) * q = 0 := by
  by_cases hg : ((Int.gcd p q : ℕ) : ℤ) = 0
  · have hz : Int.gcd p q = 0 := by exact_mod_cast hg
    obtain ⟨hp0, hq0⟩ := Int.gcd_eq_zero_iff.mp hz
    rw [hp0, hq0]
    simp
  · have hqg : ((Int.gcd p q : ℕ) : ℤ) * (q / ((Int.gcd p q : ℕ) : ℤ)) = q :=
      Int.mul_ediv_cancel' (Int.gcd_dvd_right p q)
    have hpg : ((Int.gcd p q : ℕ) : ℤ) * (p / ((Int.gcd p q : ℕ) : ℤ)) = p :=
      Int.mul_ediv_cancel' (Int.gcd_dvd_left p q)
    have key : ((Int.gcd p q : ℕ) : ℤ) * ((q / ((Int.gcd p q : ℕ) : ℤ)) * p
        - (p / ((Int.gcd p q : ℕ) : ℤ)) * q) = 0 := by
      have hring : ((Int.gcd p q : ℕ) : ℤ) * ((q / ((Int.gcd p q : ℕ) : ℤ)) * p
          - (p / ((Int.gcd p q : ℕ) : ℤ)) * q)
          = (((Int.gcd p q : ℕ) : ℤ) * (q / ((Int.gcd p q : ℕ) : ℤ))) * p
            - (((Int.gcd p q : ℕ) : ℤ) * (p / ((Int.gcd p q : ℕ) : ℤ))) * q := by ring
      rw [hring, hqg, hpg]
      ring
    exact (mul_eq_zero.mp key).resolve_left hg

/-- The combined row no longer mentions the column.

    (v : Fin n) (l u : Row n)                [a column, a lower row and an upper row]
    (combineFM v l u).coeff v = 0            [the column has left the combination]

    So: what makes the step an elimination. The two multipliers are the opposite coefficients
    divided by their common factor, so the two contributions are equal and opposite. -/
theorem combineFM_cancels (v : Fin n) (l u : Row n) :
    (combineFM v l u).coeff v = 0 := by
  simp only [combineFM, Fragment.combo, Row.add, Row.scale, Row.ofConst, add_zero]
  linarith [gcd_cross (l.coeff v) (-(u.coeff v))]

/-- The greatest common divisor of a row's coefficients.

    (r : Row n)                              [a row over n columns]
    : ℤ                                      [the largest number dividing every coefficient]

    So: `rowGcd` in ncone0. It comes out zero exactly when every coefficient is zero. -/
def rowGcd (r : Row n) : ℤ :=
  ((List.finRange n).foldr (fun i g => Nat.gcd (r.coeff i).natAbs g) 0 : ℕ)

/-- The greatest common divisor divides every coefficient.

    (r : Row n) (i : Fin n)                  [a row and one of its columns]
    rowGcd r ∣ r.coeff i                     [the divisor divides that coefficient] -/
theorem rowGcd_dvd (r : Row n) (i : Fin n) : rowGcd r ∣ r.coeff i := by
  have hgo : ∀ (l : List (Fin n)), i ∈ l →
      (((l.foldr (fun k g => Nat.gcd (r.coeff k).natAbs g) 0 : ℕ) : ℤ)) ∣ r.coeff i := by
    intro l
    induction l with
    | nil => intro h; simp at h
    | cons k l ih =>
        intro h
        simp only [List.foldr_cons]
        rcases List.mem_cons.mp h with rfl | hmem
        · exact dvd_trans (Int.natCast_dvd_natCast.mpr (Nat.gcd_dvd_left _ _))
            (Int.natAbs_dvd.mpr dvd_rfl)
        · exact dvd_trans (Int.natCast_dvd_natCast.mpr (Nat.gcd_dvd_right _ _)) (ih hmem)
  exact hgo (List.finRange n) (List.mem_finRange i)

/-- The greatest common divisor is never negative.

    (r : Row n)                              [a row over n columns]
    0 ≤ rowGcd r                             [the divisor is at zero or above] -/
theorem rowGcd_nonneg (r : Row n) : 0 ≤ rowGcd r := Int.natCast_nonneg _

/-- A row divided through by a common divisor of its coefficients.

    (r : Row n)                              [a row over n columns]
    : Row n                                  [the row divided through, with the constant rounded]

    So: the greatest-common-divisor cutting plane, `ccTidy` in certcheck and the tightening
    branch of `combineOne` in ncone0. Below a divisor of two the row is left alone. -/
def tidyRow (r : Row n) : Row n :=
  if 1 < rowGcd r then ⟨r.const / rowGcd r, fun i => r.coeff i / rowGcd r⟩ else r

/-- The cutting plane keeps every whole-number solution.

    {ρ : Fin n → ℤ} {r : Row n}              [an assignment and a row]
    Row.Holds ρ r                            [the row holds there]
    Row.Holds ρ (tidyRow r)                  [so does the tightened row]

    So: this is the one step that is sound over the whole numbers and false over the rationals.
    Every coefficient is divisible by the divisor, so the row's column part is that divisor
    times a whole number, and the constant may be rounded inward without losing an integer
    point. Rounding it the other way is the red control `../song/plans/REFLECT.md` records:
    a ceiling that is off by one refutes. -/
theorem tidyRow_holds {ρ : Fin n → ℤ} {r : Row n} (h : Row.Holds ρ r) :
    Row.Holds ρ (tidyRow r) := by
  rw [tidyRow]
  split
  · next hg =>
      have hgpos : (0 : ℤ) < rowGcd r := by omega
      have hsum : (∑ i, r.coeff i / rowGcd r * ρ i) * rowGcd r = ∑ i, r.coeff i * ρ i := by
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl fun i _ => ?_
        calc r.coeff i / rowGcd r * ρ i * rowGcd r
            = r.coeff i / rowGcd r * rowGcd r * ρ i := by ring
          _ = r.coeff i * ρ i := by rw [Int.ediv_mul_cancel (rowGcd_dvd r i)]
      rw [Row.Holds, Row.value] at h ⊢
      have hle : -(∑ i, r.coeff i / rowGcd r * ρ i) ≤ r.const / rowGcd r := by
        rw [Int.le_ediv_iff_mul_le hgpos]
        have hexp : -(∑ i, r.coeff i / rowGcd r * ρ i) * rowGcd r
            = -((∑ i, r.coeff i / rowGcd r * ρ i) * rowGcd r) := by ring
        rw [hexp, hsum]
        linarith
      simp only
      linarith
  · exact h

/-- The tightened row's conclusion, in the channel's own ring.

    {ρ : Fin n → ℤ} {r : Row n}                          [an assignment and a row]
    Row.Holds ρ r                                        [the row holds there]
    ∃ w x y z : ℤ                                        [there are four whole numbers]
    Row.value ρ (tidyRow r) = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2
                                                         [whose squares add to the tightened
                                                          row's value]

    So: `int_le_iff_exists_sum_four_sq` at the cutting plane. The step is the one place the
    decider uses that its variables are whole numbers rather than rationals, and this is the
    corpus theorem that says the whole numbers carry their own order inside the ring, at four
    squares rather than the one the reals need. -/
theorem tidy_four_squares {ρ : Fin n → ℤ} {r : Row n} (h : Row.Holds ρ r) :
    ∃ w x y z : ℤ, Row.value ρ (tidyRow r) = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2 :=
  (holds_iff_four_squares ρ (tidyRow r)).mp (tidyRow_holds h)

/-- The rows bounding a column from below.

    (v : Fin n) (rs : List (Row n))          [a column and a system]
    : List (Row n)                           [the rows whose coefficient there is above zero] -/
def lowersOf (v : Fin n) (rs : List (Row n)) : List (Row n) :=
  rs.filter (fun r => decide (0 < r.coeff v))

/-- The rows bounding a column from above.

    (v : Fin n) (rs : List (Row n))          [a column and a system]
    : List (Row n)                           [the rows whose coefficient there is below zero] -/
def uppersOf (v : Fin n) (rs : List (Row n)) : List (Row n) :=
  rs.filter (fun r => decide (r.coeff v < 0))

/-- The rows that do not mention a column.

    (v : Fin n) (rs : List (Row n))          [a column and a system]
    : List (Row n)                           [the rows whose coefficient there is zero]

    So: `elimIrrel` in ncone0, which copies them across untouched. -/
def irrelOf (v : Fin n) (rs : List (Row n)) : List (Row n) :=
  rs.filter (fun r => decide (r.coeff v = 0))

/-- One column projected away.

    (v : Fin n) (rs : List (Row n))          [the column to remove and the system]
    : List (Row n)                           [the projected system]

    So: one turn of `foldLoop`. The rows not mentioning the column are kept, and every
    lower row is combined with every upper row and the result tightened. -/
def eliminate (v : Fin n) (rs : List (Row n)) : List (Row n) :=
  irrelOf v rs ++
    (lowersOf v rs).flatMap (fun l => (uppersOf v rs).map (fun u => tidyRow (combineFM v l u)))

/-- Projecting a column away keeps every solution.

    {ρ : Fin n → ℤ} {rs : List (Row n)}      [an assignment and a system]
    SatRows ρ rs                             [the assignment satisfies every row]
    (v : Fin n)                              [any column]
    SatRows ρ (eliminate v rs)               [it satisfies every row of the projection]

    So: phase two's soundness. Kept rows are rows of the system, and every new row is a
    tightened positive combination of two of them, so `combineFM_holds` and `tidyRow_holds`
    between them cover the whole projection. The converse fails over the whole numbers, which
    is why a SAT verdict is claimed only on an all-exact run and MAYBE otherwise. -/
theorem eliminate_sound {ρ : Fin n → ℤ} {rs : List (Row n)} (h : SatRows ρ rs) (v : Fin n) :
    SatRows ρ (eliminate v rs) := by
  intro r hr
  rw [eliminate, List.mem_append] at hr
  rcases hr with hkept | hnew
  · exact h r (List.mem_of_mem_filter hkept)
  · simp only [List.mem_flatMap, List.mem_map] at hnew
    obtain ⟨l, hlm, u, hum, rfl⟩ := hnew
    have hl : 0 < l.coeff v := by
      have := List.of_mem_filter hlm
      simpa using this
    have hu : u.coeff v < 0 := by
      have := List.of_mem_filter hum
      simpa using this
    exact tidyRow_holds
      (combineFM_holds hl hu (h l (List.mem_of_mem_filter hlm))
        (h u (List.mem_of_mem_filter hum)))

/-- The projected system, stated with no order symbol in it.

    {ρ : Fin n → ℤ} {rs : List (Row n)}                  [an assignment and a system]
    SatRows ρ rs                                         [the assignment satisfies every row]
    (v : Fin n)                                          [any column]
    ∀ r ∈ eliminate v rs                                 [for every row of the projection]
    ∃ w x y z : ℤ                                        [there are four whole numbers]
    Row.value ρ r = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2        [whose squares add to its value]

    So: `Fragment.holds_iff_four_squares` on the output of the projection. Both phases and both
    operations stay inside the channel's own ring, which is `cell.slang`'s C3 entry condition
    read at the end of the pipeline rather than at the start. -/
theorem eliminate_four_squares {ρ : Fin n → ℤ} {rs : List (Row n)} (h : SatRows ρ rs)
    (v : Fin n) :
    ∀ r ∈ eliminate v rs, ∃ w x y z : ℤ,
      Row.value ρ r = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2 :=
  fun r hr => (holds_iff_four_squares ρ r).mp (eliminate_sound h v r hr)

end Decider
end Apeiron
