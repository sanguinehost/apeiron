import Apeiron.Decider.Cert

/-!
# Capped certificates and cutting-plane arithmetic

@name Capped Certificates and Cut Arithmetic
@grade proved

A finite box permits a certificate that names one failed system row at every assignment in the
box. The checker validates both parts: the named row belongs to the system, and that row fails at
the assignment. It does not trust the function that supplied the row.

For example, a one-column box from zero through two has three assignments. A certificate for an
unsatisfiable system names a failed row at zero, at one, and at two. `checkCappedCert` walks exactly
those three assignments. `capped_certificate_complete` proves that such a certificate exists
exactly when no assignment in the box satisfies the system.

`rescaled_floor_le` and `rescaled_ceil_ge` show that division by a larger common factor followed
by scaling gives bounds at least as strong as division by the smaller factor. `certStep_scale`
implements the scaling with an existing combination step.

## Root

`cappedAssignments` recursively enumerates each inclusive integer coordinate range. The certificate
records the first failed row from that concrete list, and `capped_certificate_complete` derives both
directions from its membership theorem.
The cut arithmetic uses integer division, and `certStep_scale` specializes `crCombo` to two
occurrences of one parent. These facts support certificate construction; they do not supply a
finite cutting-plane derivation for an arbitrary unsatisfiable system.

## Boundary

This certificate is for the capped fragment. It is not a derivation in `CertOp`, and the theorem
does not prove that every uncapped unsatisfiable system has a log accepted by `checkCert`.
-/

namespace Apeiron
namespace Decider

open Fragment

variable {n : ℕ}

/-- One claimed failing row index for every assignment in the box.

    {n : ℕ}                                 [the number of columns]
    (rs : List (Row n))                     [the rows of the system]
    CappedCertificate rs                    [a certificate for that system]
    (Fin n → ℤ) → Option (Fin rs.length)     [maps assignments to optional row indices] -/
abbrev CappedCertificate (rs : List (Row n)) := (Fin n → ℤ) → Option (Fin rs.length)

/-- Enumerate the declared finite integer box by structural recursion on its dimensions.

    (n : ℕ)                               [the number of coordinates]
    (Fin n → ℤ)                           [one whole-number bound at every coordinate]
    List                                  [a finite sequence of things, in order]
    →                                     [takes the left and gives the right] -/
def cappedAssignments : (n : ℕ) → (Fin n → ℤ) → (Fin n → ℤ) → List (Fin n → ℤ)
  | 0, _, _ => [fun i => Fin.elim0 i]
  | n + 1, lo, hi =>
    (Int.range (lo 0) (hi 0 + 1)).flatMap fun x =>
      (cappedAssignments n (fun i => lo i.succ) (fun i => hi i.succ)).map (Fin.cons x)

/-- The first system row that fails at an assignment, if there is one.

    (rs : List (Row n))                   [the finite sequence of system rows]
    (ρ : Fin n → ℤ)                       [the whole-number assignment]
    List                                  [a finite sequence of things, in order]
    Row                                   [a constant-plus-coefficients inequality]
    →                                     [takes the left and gives the right] -/
def firstFailingRow? (rs : List (Row n)) (ρ : Fin n → ℤ) : Option (Fin rs.length) :=
  (List.finRange rs.length).find? fun k => decide (¬ Row.Holds ρ (rs.get k))

/-- The deterministic certificate reader selects the first failing system row.

    (rs : List (Row n))                   [the finite sequence of system rows]
    List                                  [a finite sequence of things, in order]
    Row                                   [a constant-plus-coefficients inequality]
    →                                     [takes the left and gives the right] -/
def synthesizeCappedCert (rs : List (Row n)) : CappedCertificate rs :=
  firstFailingRow? rs

/-- Membership in the executable assignment list is exactly the declared box condition.

    (n : ℕ)                               [the number of coordinates]
    (lo hi : Fin n → ℤ)                   [the lower and upper coordinate bounds]
    (ρ : Fin n → ℤ)                       [the assignment]
    List                                  [a finite sequence of things, in order]
    Box                                   [the coordinatewise cap condition]
    ∈                                     [is one of the things in]
    ↔                                     [exactly when, true in both directions] -/
theorem mem_cappedAssignments_iff (n : ℕ) (lo hi : Fin n → ℤ) (ρ : Fin n → ℤ) :
    ρ ∈ cappedAssignments n lo hi ↔ Box lo hi ρ := by
  induction n with
  | zero =>
      simp only [cappedAssignments, List.mem_singleton]
      constructor
      · intro _ i
        exact i.elim0
      · intro _
        funext i
        exact i.elim0
  | succ n ih =>
      simp only [cappedAssignments, List.mem_flatMap, List.mem_map]
      constructor
      · rintro ⟨x, hx, t, ht, rfl⟩ i
        refine Fin.cases ?_ (fun j => ?_) i
        · have hrange := Int.mem_range_iff.mp hx
          simpa only [Fin.cons_zero] using ⟨hrange.1, by omega⟩
        · simpa only [Fin.cons_succ] using (ih _ _ _).mp ht j
      · intro h
        have hzero := h 0
        refine ⟨ρ 0, Int.mem_range_iff.mpr ⟨hzero.1, by omega⟩,
          (fun i => ρ i.succ), ?_, ?_⟩
        · exact (ih _ _ _).mpr fun i => h i.succ
        · funext i
          refine Fin.cases ?_ (fun j => ?_) i
          · rfl
          · rfl

/-- Check a certificate at every assignment inside declared caps.

    (lo hi : Fin n → ℤ)                     [a bottom and top cap for every column]
    (rs : List (Row n))                     [the rows of the system]
    (cert : CappedCertificate rs)            [one claimed failing row per assignment]
    : Bool                                   [whether every in-box claim checks]

    So: acceptance checks row membership and failure over the complete finite box. -/
def checkCappedCert (lo hi : Fin n → ℤ) (rs : List (Row n))
    (cert : CappedCertificate rs) : Bool :=
  (cappedAssignments n lo hi).all fun ρ =>
    match cert ρ with
    | some k => decide (¬ Row.Holds ρ (rs.get k))
    | none => false

/-- **Capped certificates exist exactly for capped-unsatisfiable systems.**

    (lo hi : Fin n → ℤ)                                  [a bottom and top cap per column]
    (rs : List (Row n))                                  [the rows of a system]
    (∃ cert : CappedCertificate rs,                      [some checked certificate exists]
      checkCappedCert lo hi rs cert = true)              [and the checker accepts it]
    ↔                                                    [exactly when, true both directions]
    ¬ ∃ ρ : Fin n → ℤ, Box lo hi ρ ∧ SatRows ρ rs        [no in-box assignment satisfies it]

    So: finite search supplies a failed system row at each assignment, and checking those rows
    covers the whole declared box. Assignments outside the caps are not claimed. -/
theorem capped_certificate_complete (lo hi : Fin n → ℤ) (rs : List (Row n)) :
    (∃ cert : CappedCertificate rs, checkCappedCert lo hi rs cert = true) ↔
      ¬ ∃ ρ : Fin n → ℤ, Box lo hi ρ ∧ SatRows ρ rs := by
  constructor
  · rintro ⟨cert, hcert⟩ hsat
    obtain ⟨ρ, hρ, hrs⟩ := hsat
    have hchecked := List.all_eq_true.mp hcert ρ ((mem_cappedAssignments_iff n lo hi ρ).mpr hρ)
    cases hc : cert ρ with
    | none => simp [hc] at hchecked
    | some k =>
        simp only [hc] at hchecked
        have hfailed : ¬ Row.Holds ρ (rs.get k) := by
          exact of_decide_eq_true hchecked
        exact hfailed (hrs (rs.get k) (List.get_mem rs k))
  · intro hsat
    refine ⟨synthesizeCappedCert rs, ?_⟩
    apply List.all_eq_true.mpr
    intro ρ hρ
    have hbox : Box lo hi ρ := (mem_cappedAssignments_iff n lo hi ρ).mp hρ
    cases hfail : firstFailingRow? rs ρ with
    | none =>
        have hall : SatRows ρ rs := by
          intro r hr
          by_contra hnot
          obtain ⟨k, hk⟩ :=
            (List.exists_mem_iff_get (l := rs) (p := fun r => ¬ Row.Holds ρ r)).mp
              ⟨r, hr, hnot⟩
          exact (List.find?_eq_none.mp hfail k (List.mem_finRange k)) (decide_eq_true hk)
        exact (hsat ⟨ρ, hbox, hall⟩).elim
    | some k =>
        simpa only [synthesizeCappedCert, hfail] using List.find?_some hfail

/-- An accepted capped certificate exposes a failed system row at each in-box assignment.

    (lo hi : Fin n → ℤ)                    [the declared lower and upper caps]
    (rs : List (Row n))                    [the system rows]
    (cert : CappedCertificate rs)          [the claimed failed-row reader]
    (ρ : Fin n → ℤ)                        [one assignment inside the box]
    checkCappedCert lo hi rs cert = true   [the exhaustive checker accepted]
    Box lo hi ρ                            [the assignment is within its caps]
    ∃ k : Fin rs.length,                  [one valid system-row index]
      cert ρ = some k ∧                   [the reader names that index]
      ¬ Row.Holds ρ (rs.get k)            [and the named row fails]

    So: the capped object is an exhaustive pointwise reader, not a Farkas vector.  The uncapped
    derivation-to-log extraction is `CuttingPlaneDerivation.compile_accepts`; a global rational
    Farkas extraction is false for genuine `crTidy` cuts. -/
theorem capped_certificate_extract (lo hi : Fin n → ℤ) (rs : List (Row n))
    (cert : CappedCertificate rs) (hcert : checkCappedCert lo hi rs cert = true)
    (ρ : Fin n → ℤ) (hbox : Box lo hi ρ) :
    ∃ k : Fin rs.length, cert ρ = some k ∧ ¬ Row.Holds ρ (rs.get k) := by
  have hchecked := List.all_eq_true.mp hcert ρ
    ((mem_cappedAssignments_iff n lo hi ρ).mpr hbox)
  cases hc : cert ρ with
  | none => simp [hc] at hchecked
  | some k =>
      refine ⟨k, rfl, ?_⟩
      exact of_decide_eq_true (by simpa only [hc] using hchecked)

/-- Dividing by a larger common factor and scaling back tightens an upper bound.

    (u d k : ℤ)                             [a bound, a divisor, and a scaling factor]
    (hd : 0 < d)                            [the divisor is positive]
    (hk : 0 < k)                            [the scaling factor is positive]
    k * (u / (d * k)) ≤ u / d              [the scaled bound is no larger]

    So: gcd division can replace division by any positive common divisor without weakening
    the resulting upper bound. -/
theorem rescaled_floor_le (u d k : ℤ) (hd : 0 < d) (hk : 0 < k) :
    k * (u / (d * k)) ≤ u / d := by
  apply (Int.le_ediv_iff_mul_le hd).mpr
  have h := Int.ediv_mul_le u (ne_of_gt (mul_pos hd hk))
  nlinarith

/-- Dividing by a larger common factor and scaling back tightens a lower bound.

    (l d k : ℤ)                             [a bound, a divisor, and a scaling factor]
    (hd : 0 < d)                            [the divisor is positive]
    (hk : 0 < k)                            [the scaling factor is positive]
    -((-l) / d) ≤ k * -((-l) / (d * k))    [the scaled bound is no smaller]

    So: this is the inward rounding used by `crTidy` on a lower bound. -/
theorem rescaled_ceil_ge (l d k : ℤ) (hd : 0 < d) (hk : 0 < k) :
    -((-l) / d) ≤ k * -((-l) / (d * k)) := by
  have h := rescaled_floor_le (-l) d k hd hk
  nlinarith

/-- A positive integer multiple is available using the existing combination operation.

    (raw : List (Line n)) (rs : List (CRow n)) [the input system and replay rows]
    (i : ℕ) (x : CRow n)                      [a row index and the row at that index]
    (hi : rs[i]? = some x)                    [the row has already been emitted]
    (k : ℤ) (hk : 2 ≤ k)                     [the desired integer multiple]
    certStep raw rs (.combo i i 1 (k - 1))    [combining the row with itself]
      = some (rs ++ [⟨                       [appends one scaled row]
        x.lo.map (fun l => k * l),           [the lower bound is multiplied]
        x.up.map (fun u => k * u),           [the upper bound is multiplied]
        fun j => k * x.coeff j⟩])            [every coefficient is multiplied]

    So: scaling needs no new certificate operation; a multiplier of one reuses the index. -/
theorem certStep_scale (raw : List (Line n)) (rs : List (CRow n))
    (i : ℕ) (x : CRow n) (hi : rs[i]? = some x) (k : ℤ) (hk : 2 ≤ k) :
    certStep raw rs (.combo i i 1 (k - 1)) =
      some (rs ++ [⟨x.lo.map (fun l => k * l),
        x.up.map (fun u => k * u), fun j => k * x.coeff j⟩]) := by
  have hguard : 1 ≤ (1 : ℤ) ∧ 1 ≤ k - 1 := by omega
  simp only [certStep, if_pos hguard, hi]
  apply congrArg (fun y : CRow n => some (rs ++ [y]))
  cases x with
  | mk lo up coeff =>
      cases lo <;> cases up <;>
        simp only [crCombo, Option.map_none, Option.map_some, CRow.mk.injEq,
          Option.some.injEq, true_and] <;>
        (try constructor) <;> (try constructor) <;> (try funext j) <;> ring

-- The deterministic reader accepts a nonempty box whose sole row fails everywhere.
#guard checkCappedCert (fun _ : Fin 1 => 0) (fun _ : Fin 1 => 2)
  [⟨-1, fun _ : Fin 1 => 0⟩]
  (synthesizeCappedCert [⟨-1, fun _ : Fin 1 => 0⟩])

-- The only satisfying assignment is the inclusive upper endpoint, so no certificate is accepted.
#guard !checkCappedCert (fun _ : Fin 1 => 0) (fun _ : Fin 1 => 2)
  [⟨-2, fun _ : Fin 1 => 1⟩, ⟨2, fun _ : Fin 1 => -1⟩]
  (synthesizeCappedCert [⟨-2, fun _ : Fin 1 => 1⟩, ⟨2, fun _ : Fin 1 => -1⟩])

-- A reader that names no failed row is rejected on a nonempty box.
#guard !checkCappedCert (fun _ : Fin 1 => 0) (fun _ : Fin 1 => 0)
  [⟨-1, fun _ : Fin 1 => 0⟩] (fun _ => none)

-- Inverted caps enumerate no assignments and therefore accept the empty reader vacuously.
#guard checkCappedCert (fun _ : Fin 1 => 2) (fun _ : Fin 1 => 0) [] (fun _ => none)

-- Dimension zero has its unique empty assignment.
#guard (cappedAssignments 0 (fun i => Fin.elim0 i) (fun i => Fin.elim0 i)).length == 1

-- The cut turns `4x = 2` into `1 ≤ x ≤ 0`; a subsequent scaling preserves the crossing.
#guard checkCert [Line.eq (⟨-2, fun _ : Fin 1 => 4⟩ : Row 1)] []
  [.load 0, .tidy 0, .combo 1 1 1 1, .close 2]

end Decider
end Apeiron
