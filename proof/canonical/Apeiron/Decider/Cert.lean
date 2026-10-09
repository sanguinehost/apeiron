import Apeiron.Decider.Eliminate
import Apeiron.Decider.CompleteCertificate

/-!
# The certificate, and the checker that needs no trust in the decider

@name The Certificate
@grade proved
@cite farkas-1902

An UNSAT verdict from `../song/src/prover/cone/ncone0.slang` is not trusted until
`../song/src/prover/cone/certcheck.slang` replays it. The replay is arithmetic: five row
operations, a log of them, and a last row that cannot hold. The canonical language also carries
a finite residue-refutation tree as a terminal operation. `checkCert_sound` proves that every
accepted log refutes the system. `certificate_complete` proves the converse for the extended
language: every integer-infeasible system has an accepted log, constructed by `synthesizeCert`.

## The row a certificate carries

`certcheck` holds a two-sided row: a lower bound, an upper bound, either of which may be absent,
and one coefficient per column. A wire line `g c a…` enters as `-c ≤ a·x` with no upper bound;
an `e` line enters two-sided, which is why the equality kind exists on the wire at all.

## Row operations and terminal certificates

```
   r k        input line k of the system, verbatim          ccStepR
   c i j a b  a·row i + b·row j, with a ≥ 1 and b ≥ 1       ccStepC / ccCombo
   n i        the negation of row i                          ccStepN / ccNegate
   m i j      the meet of two rows with equal coefficients   ccStepM / ccMeet
   t i        row i tidied by the coefficient gcd            ccStepT / ccTidy
   x i        row i is impossible, and it ends the log       ccStepX
```

Rows are numbered in emission order and a step may only name a row already built. Every operation
that emits a row maps its parents to a row they imply. The additional `.residue cert` terminal
checks every finite residue branch against the original input, using `checkResidueCertificate`.
Neither terminal permits a following operation. The residue constructor is canonical proof data;
the existing Song text checker has no corresponding opcode or tree decoder.

## Root

| from | here |
|---|---|
| `Fragment.combo_holds` (from `combovalid_nonneg`, `farkas_bound`) | `combo_is_farkas` |
| the same, applied | `crCombo_holds` |
| `inCell_meet_iff`, `meet_ofBounds` | `crMeet_holds` |
| `crossing_ofBounds`, `crossing_independent_of_cone` | `impossible_is_crossing` |
| the same, applied | `crImpossible_empty` |
| `int_le_iff_exists_sum_four_sq` | `crTidy_holds`, `cert_four_squares` |
| `Fragment.unsat_of_negative_combo` | `negative_combo_refutes` |
| `synthesizeResidueCertificate_accepts`, `completeSat_eq_false_iff` | `certificate_complete` |

`combo_is_farkas` is the identity that makes the `c` operation an instance rather than an
analogy: the lower bound of a combined row, read as a row of the fragment, is literally
`Fragment.combo` of the two parents' lower rows with the certificate's own multipliers. The
soundness of the operation is then `Fragment.combo_holds` applied, which is
`../song/src/prover/cone/arith.slang`'s `combovalid_nonneg` and `farkas_bound`. The `x`
operation closes on a crossing, which `Bridge.crossing_ofBounds` says is the order and
`crossing_independent_of_cone` says is not the cone.

## Integer cuts and residue branches

`t` is the greatest-common-divisor cutting plane. It is sound over the whole numbers and false
over the rationals, so it is not a nonnegative combination of its parent and a Farkas extraction
of a whole certificate is not available in general. `negative_combo_refutes` is the Farkas
statement itself, and its docstring says how much of the certificate it reaches: the `c` chain
is a combination by `combo_is_farkas`, and turning a whole log into one combination is an
extraction this file does not perform. `checkCert_sound`'s induction is the meta-composition
instead, and it covers logs whose `t` is a genuine cutting plane, which have no Farkas
combination at all.

The residue terminal also uses integer semantics: exact residue splitting followed by unit
projection. `synthesizeCert` extracts its finite tree for every integer-infeasible input. It is
not a rational Farkas vector.

## Delta against the running checker, declared

`certcheck` stores a row's bounds as a sign and three sixty-four-bit limbs, and `ccTidy` peels
whole zero limbs before the single-word gcd runs. That widening is arithmetic bookkeeping for a
machine word and carries no law of its own in that file either; the model computes in `ℤ`, where
the peel is the identity. What the model does not carry is `certcheck`'s refusals on a
multi-limb gcd, which cost coverage and never soundness.
-/

namespace Apeiron
namespace Decider

open Trichotomy Fragment

variable {n : ℕ}

/-- A row of a certificate: a bound from each side, and one coefficient per column.

    CRow (n : ℕ)                             [a certificate row over n columns]

    So: `certcheck`'s `F_HASLO`/`F_LO`/`F_HASUP`/`F_UP`/`F_C` layout, with the flags folded
    into the absence of a bound. -/
structure CRow (n : ℕ) where
  /-- The lower bound, if the row carries one. -/
  lo : Option ℤ
  /-- The upper bound, if the row carries one. -/
  up : Option ℤ
  /-- One coefficient per column. -/
  coeff : Fin n → ℤ

/-- What a certificate row's columns work out to at an assignment.

    (x : CRow n)                             [a certificate row]
    (ρ : Fin n → ℤ)                          [a whole number for each column]
    : ℤ                                      [the value the row's bounds bound] -/
def CRow.dotv (x : CRow n) (ρ : Fin n → ℤ) : ℤ := ∑ i, x.coeff i * ρ i

/-- A certificate row holds when its value meets whichever bounds are present.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    (x : CRow n)                             [a certificate row]
    : Prop                                   [a statement that is true or false] -/
def CRow.Holds (ρ : Fin n → ℤ) (x : CRow n) : Prop :=
  x.lo.elim True (fun l => l ≤ x.dotv ρ) ∧ x.up.elim True (fun u => x.dotv ρ ≤ u)

/-- The fragment row saying a certificate row's value is at least a given bound.

    (x : CRow n) (l : ℤ)                     [a certificate row and a lower bound]
    : Row n                                  [the fragment row asserting the bound]

    So: the bridge from `certcheck`'s two-sided row to `Apeiron/Fragment/Basic.lean`'s row,
    which is where Farkas already lives. -/
def loRow (x : CRow n) (l : ℤ) : Row n := ⟨-l, x.coeff⟩

/-- The fragment row saying a certificate row's value is at most a given bound.

    (x : CRow n) (u : ℤ)                     [a certificate row and an upper bound]
    : Row n                                  [the fragment row asserting the bound] -/
def upRow (x : CRow n) (u : ℤ) : Row n := ⟨u, fun i => -x.coeff i⟩

/-- A lower bound holds exactly when its fragment row does.

    (ρ : Fin n → ℤ) (x : CRow n) (l : ℤ)     [an assignment, a row, a lower bound]
    Row.Holds ρ (loRow x l)                  [the fragment row holds]
    ↔ l ≤ x.dotv ρ                           [exactly when the bound is met] -/
theorem loRow_holds_iff (ρ : Fin n → ℤ) (x : CRow n) (l : ℤ) :
    Row.Holds ρ (loRow x l) ↔ l ≤ x.dotv ρ := by
  simp only [Row.Holds, Row.value, loRow, CRow.dotv]
  omega

/-- An upper bound holds exactly when its fragment row does.

    (ρ : Fin n → ℤ) (x : CRow n) (u : ℤ)     [an assignment, a row, an upper bound]
    Row.Holds ρ (upRow x u)                  [the fragment row holds]
    ↔ x.dotv ρ ≤ u                           [exactly when the bound is met] -/
theorem upRow_holds_iff (ρ : Fin n → ℤ) (x : CRow n) (u : ℤ) :
    Row.Holds ρ (upRow x u) ↔ x.dotv ρ ≤ u := by
  have hneg : ∑ i, -x.coeff i * ρ i = -(∑ i, x.coeff i * ρ i) := by
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  simp only [Row.Holds, Row.value, upRow, CRow.dotv, hneg]
  omega

/-- One line of the wire, as the certificate's first operation reads it.

    Line n → CRow n                          [a wire line, and the certificate row it becomes]

    So: `ccRawRow`. A `g` line gives a lower bound only; an `e` line gives both, which is how
    the two-sided shape enters the replay. -/
def certRaw : Line n → CRow n
  | .ge r => ⟨some (-r.const), none, r.coeff⟩
  | .eq r => ⟨some (-r.const), some (-r.const), r.coeff⟩

/-- Reading a satisfied line gives a certificate row that holds.

    (ρ : Fin n → ℤ) (l : Line n)             [an assignment and a wire line]
    Line.Sat ρ l                             [the assignment satisfies the line]
    CRow.Holds ρ (certRaw l)                 [so the row read off it holds] -/
theorem certRaw_holds (ρ : Fin n → ℤ) (l : Line n) (h : Line.Sat ρ l) :
    CRow.Holds ρ (certRaw l) := by
  cases l with
  | ge r =>
      rw [Line.Sat, Row.Holds, Row.value] at h
      exact ⟨by simp only [certRaw, CRow.dotv, Option.elim]; omega, trivial⟩
  | eq r =>
      rw [Line.Sat, Row.value] at h
      exact ⟨by simp only [certRaw, CRow.dotv, Option.elim]; omega,
        by simp only [certRaw, CRow.dotv, Option.elim]; omega⟩

/-- The `c` operation: a positive combination of two rows.

    (a b : ℤ) (x y : CRow n)                 [two multipliers and two certificate rows]
    : CRow n                                 [their combination]

    So: `ccCombo`. A bound survives only when both parents carry it, which is `ccComboLo`'s and
    `ccComboUp`'s own condition. -/
def crCombo (a b : ℤ) (x y : CRow n) : CRow n :=
  { lo := match x.lo, y.lo with
      | some l₁, some l₂ => some (a * l₁ + b * l₂)
      | _, _ => none
    up := match x.up, y.up with
      | some u₁, some u₂ => some (a * u₁ + b * u₂)
      | _, _ => none
    coeff := fun i => a * x.coeff i + b * y.coeff i }

/-- The combined row's lower bound is the Farkas combination of its parents'.

    (a b l₁ l₂ : ℤ) (x y : CRow n)                       [multipliers, bounds, two rows]
    loRow (crCombo a b x y) (a * l₁ + b * l₂)            [the combined row's lower assertion]
    = Fragment.combo [(a, loRow x l₁), (b, loRow y l₂)]  [is the weighted sum of the parents']

    So: the `c` operation is an instance of `Fragment.combo` and not an analogy to it. This
    identity is why `crCombo_holds` below is `Fragment.combo_holds` applied, which is
    `../song/src/prover/cone/arith.slang`'s `combovalid_nonneg` together with `farkas_bound`,
    the meta-composition that says a nonnegative weighted sum of rows that hold holds. -/
theorem combo_is_farkas (a b l₁ l₂ : ℤ) (x y : CRow n) :
    loRow (crCombo a b x y) (a * l₁ + b * l₂)
      = Fragment.combo [(a, loRow x l₁), (b, loRow y l₂)] := by
  simp only [loRow, crCombo, Fragment.combo, Row.add, Row.scale, Row.ofConst, Row.mk.injEq]
  constructor
  · ring
  · funext i
    ring

/-- The `c` operation keeps every solution.

    {ρ : Fin n → ℤ} {a b : ℤ} {x y : CRow n} [an assignment, two multipliers, two rows]
    1 ≤ a                                    [the first multiplier is at least one]
    1 ≤ b                                    [the second multiplier is at least one]
    CRow.Holds ρ x                           [the first row holds]
    CRow.Holds ρ y                           [the second row holds]
    CRow.Holds ρ (crCombo a b x y)           [so does their combination]

    So: `Fragment.combo_holds` on the lower side through `combo_is_farkas`, and the same
    argument on the upper side through `upRow`. Nothing here is proved twice. -/
theorem crCombo_holds {ρ : Fin n → ℤ} {a b : ℤ} {x y : CRow n}
    (ha : 1 ≤ a) (hb : 1 ≤ b) (hx : CRow.Holds ρ x) (hy : CRow.Holds ρ y) :
    CRow.Holds ρ (crCombo a b x y) := by
  refine ⟨?_, ?_⟩
  · cases hlx : x.lo with
    | none => simp [crCombo, hlx]
    | some l₁ =>
        cases hly : y.lo with
        | none => simp [crCombo, hlx, hly]
        | some l₂ =>
            have h₁ : Row.Holds ρ (loRow x l₁) := by
              rw [loRow_holds_iff]
              have := hx.1
              rw [hlx] at this
              exact this
            have h₂ : Row.Holds ρ (loRow y l₂) := by
              rw [loRow_holds_iff]
              have := hy.1
              rw [hly] at this
              exact this
            have hfark : Row.Holds ρ (Fragment.combo [(a, loRow x l₁), (b, loRow y l₂)]) := by
              refine Fragment.combo_holds ρ _ ?_ ?_
              · intro pr hpr
                simp only [List.mem_cons, List.not_mem_nil, or_false] at hpr
                rcases hpr with rfl | rfl
                · omega
                · omega
              · intro pr hpr
                simp only [List.mem_cons, List.not_mem_nil, or_false] at hpr
                rcases hpr with rfl | rfl
                · exact h₁
                · exact h₂
            rw [← combo_is_farkas, loRow_holds_iff] at hfark
            simpa [crCombo, hlx, hly] using hfark
  · cases hux : x.up with
    | none => simp [crCombo, hux]
    | some u₁ =>
        cases huy : y.up with
        | none => simp [crCombo, hux, huy]
        | some u₂ =>
            have h₁ : x.dotv ρ ≤ u₁ := by have := hx.2; rw [hux] at this; exact this
            have h₂ : y.dotv ρ ≤ u₂ := by have := hy.2; rw [huy] at this; exact this
            have hgoal : (crCombo a b x y).dotv ρ ≤ a * u₁ + b * u₂ := by
              have hsplit : (crCombo a b x y).dotv ρ = a * x.dotv ρ + b * y.dotv ρ := by
                simp only [CRow.dotv, crCombo]
                rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
                exact Finset.sum_congr rfl fun i _ => by ring
              rw [hsplit]
              have hA : a * x.dotv ρ ≤ a * u₁ := by
                exact mul_le_mul_of_nonneg_left h₁ (by omega)
              have hB : b * y.dotv ρ ≤ b * u₂ := by
                exact mul_le_mul_of_nonneg_left h₂ (by omega)
              omega
            simpa [crCombo, hux, huy] using hgoal

/-- The `n` operation: a row with every sign flipped.

    (x : CRow n)                             [a certificate row]
    : CRow n                                 [the same assertion, negated]

    So: `ccNegate`. The two bounds swap sides, which is why negation is an equivalence rather
    than a weakening, and why it lets the `c` operation keep both multipliers positive. -/
def crNeg (x : CRow n) : CRow n :=
  { lo := x.up.map (fun u => -u)
    up := x.lo.map (fun l => -l)
    coeff := fun i => -x.coeff i }

/-- The `n` operation keeps every solution.

    {ρ : Fin n → ℤ} {x : CRow n}             [an assignment and a certificate row]
    CRow.Holds ρ x                           [the row holds]
    CRow.Holds ρ (crNeg x)                   [so does its negation] -/
theorem crNeg_holds {ρ : Fin n → ℤ} {x : CRow n} (h : CRow.Holds ρ x) :
    CRow.Holds ρ (crNeg x) := by
  have hd : ∑ i, -x.coeff i * ρ i = -(x.dotv ρ) := by
    simp only [CRow.dotv, ← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  obtain ⟨hl, hu⟩ := h
  refine ⟨?_, ?_⟩
  · cases hux : x.up with
    | none => simp [crNeg, hux]
    | some u =>
        rw [hux] at hu
        simp only [Option.elim, CRow.dotv] at hu
        simp only [crNeg, hux, Option.map_some, Option.elim, CRow.dotv, hd]
        omega
  · cases hlx : x.lo with
    | none => simp [crNeg, hlx]
    | some l =>
        rw [hlx] at hl
        simp only [Option.elim, CRow.dotv] at hl
        simp only [crNeg, hlx, Option.map_some, Option.elim, CRow.dotv, hd]
        omega

/-- The `m` operation: two rows over the same columns, met.

    (x y : CRow n)                           [two certificate rows]
    : CRow n                                 [the row of the tighter bounds]

    So: `ccMeet`, which the checker allows only after `ccSameCoefs` agrees the two rows bound
    the same quantity. -/
def crMeet (x y : CRow n) : CRow n :=
  { lo := maxOpt x.lo y.lo, up := minOpt x.up y.up, coeff := x.coeff }

/-- The `m` operation keeps every solution.

    {ρ : Fin n → ℤ} {x y : CRow n}           [an assignment and two certificate rows]
    x.coeff = y.coeff                        [the two rows bound the same quantity]
    CRow.Holds ρ x                           [the first holds]
    CRow.Holds ρ y                           [the second holds]
    CRow.Holds ρ (crMeet x y)                [so does their meet]

    So: `meet_lo_two_sided` and `meet_hi_two_sided`, which are `inCell_meet_iff` transported to
    the whole numbers, so the checker's meet is the algebra's meet and not a separate notion.
    The equal-coefficient side condition is what makes the two bounds bounds on one number. -/
theorem crMeet_holds {ρ : Fin n → ℤ} {x y : CRow n} (hc : x.coeff = y.coeff)
    (hx : CRow.Holds ρ x) (hy : CRow.Holds ρ y) : CRow.Holds ρ (crMeet x y) := by
  have hd : (crMeet x y).dotv ρ = x.dotv ρ := rfl
  have hdy : y.dotv ρ = x.dotv ρ := by simp only [CRow.dotv, hc]
  obtain ⟨hxl, hxu⟩ := hx
  obtain ⟨hyl, hyu⟩ := hy
  rw [hdy] at hyl hyu
  refine ⟨?_, ?_⟩
  · cases hlx : x.lo with
    | none => cases hly : y.lo <;> simp_all [crMeet, maxOpt]
    | some l₁ =>
        cases hly : y.lo with
        | none => simp_all [crMeet, maxOpt]
        | some l₂ =>
            rw [hlx] at hxl
            rw [hly] at hyl
            simp only [crMeet, maxOpt, hlx, hly, propNewLo, Option.elim]
            exact meet_lo_two_sided hxl hyl
  · cases hux : x.up with
    | none => cases huy : y.up <;> simp_all [crMeet, minOpt]
    | some u₁ =>
        cases huy : y.up with
        | none => simp_all [crMeet, minOpt]
        | some u₂ =>
            rw [hux] at hxu
            rw [huy] at hyu
            simp only [crMeet, minOpt, hux, huy, propNewHi, Option.elim]
            exact meet_hi_two_sided hxu hyu

/-- The first coefficient that is not zero, walking the columns in order.

    List (Fin n) → CRow n → ℤ                [the columns to walk, a row, the leading value]

    So: `ccLead`, which `ccTidy` uses to decide whether to negate before dividing. -/
def crLead : List (Fin n) → CRow n → ℤ
  | [], _ => 0
  | i :: is, x => if x.coeff i = 0 then crLead is x else x.coeff i

/-- The greatest common divisor of a certificate row's coefficients.

    (x : CRow n)                             [a certificate row]
    : ℤ                                      [the largest number dividing every coefficient]

    So: `ccGcd`. It comes out zero exactly when the row is a constant, which is the branch
    `ccGround` takes. -/
def crGcd (x : CRow n) : ℤ :=
  ((List.finRange n).foldr (fun i g => Nat.gcd (x.coeff i).natAbs g) 0 : ℕ)

/-- The divisor divides every coefficient.

    (x : CRow n) (i : Fin n)                 [a certificate row and one of its columns]
    crGcd x ∣ x.coeff i                      [the divisor divides that coefficient] -/
theorem crGcd_dvd (x : CRow n) (i : Fin n) : crGcd x ∣ x.coeff i :=
  rowGcd_dvd (⟨0, x.coeff⟩ : Row n) i

/-- The `t` operation: negate if the lead is negative, then divide by the coefficient divisor.

    (x : CRow n)                             [a certificate row]
    : CRow n                                 [the tightened row]

    So: `ccTidy`. A row with no columns left is grounded: if its bounds already exclude zero it
    becomes the impossible interval the `x` operation accepts, and otherwise it loses both
    bounds. Otherwise the coefficients are divided through and the bounds rounded inward, which
    is the cutting plane. -/
def crTidy (x : CRow n) : CRow n :=
  let y := if crLead (List.finRange n) x < 0 then crNeg x else x
  let g := crGcd y
  if g = 0 then
    (if y.lo.elim false (fun l => 0 < l) || y.up.elim false (fun u => u < 0) then
      ⟨some 1, some 0, y.coeff⟩
    else ⟨none, none, y.coeff⟩)
  else if g = 1 then y
  else ⟨y.lo.map (fun l => -((-l) / g)), y.up.map (fun u => u / g), fun i => y.coeff i / g⟩

/-- The `t` operation keeps every whole-number solution.

    {ρ : Fin n → ℤ} {x : CRow n}             [an assignment and a certificate row]
    CRow.Holds ρ x                           [the row holds]
    CRow.Holds ρ (crTidy x)                  [so does the tightened row]

    The cutting-plane row operation is sound over the whole numbers and false over the rationals.
    The ground branch is the other half: a constant row whose bounds already
    exclude zero cannot hold, so replacing it by the impossible interval loses nothing, and a
    constant row whose bounds allow zero is dropped. -/
theorem crTidy_holds {ρ : Fin n → ℤ} {x : CRow n} (h : CRow.Holds ρ x) :
    CRow.Holds ρ (crTidy x) := by
  set y : CRow n := if crLead (List.finRange n) x < 0 then crNeg x else x with hy
  have hyh : CRow.Holds ρ y := by
    rw [hy]; split
    · exact crNeg_holds h
    · exact h
  rw [crTidy, ← hy]
  split
  · next hg0 =>
      have hzero : y.dotv ρ = 0 := by
        have hcz : ∀ i, y.coeff i = 0 := by
          intro i
          have hd : crGcd y ∣ y.coeff i := crGcd_dvd y i
          rw [hg0] at hd
          exact zero_dvd_iff.mp hd
        simp [CRow.dotv, hcz]
      split
      · next hbad =>
          exfalso
          obtain ⟨hl, hu⟩ := hyh
          rcases Bool.or_eq_true _ _ |>.mp hbad with hb | hb
          · cases hly : y.lo with
            | none => rw [hly] at hb; simp at hb
            | some l =>
                rw [hly] at hb hl
                simp only [Option.elim, decide_eq_true_eq] at hb hl
                rw [hzero] at hl
                omega
          · cases huy : y.up with
            | none => rw [huy] at hb; simp at hb
            | some u =>
                rw [huy] at hb hu
                simp only [Option.elim, decide_eq_true_eq] at hb hu
                rw [hzero] at hu
                omega
      · exact ⟨trivial, trivial⟩
  · next hg0 =>
      split
      · exact hyh
      · next hg1 =>
          have hgpos : 0 < crGcd y := by
            have := rowGcd_nonneg (⟨0, y.coeff⟩ : Row n)
            simp only [crGcd, rowGcd] at *
            omega
          have hdv : (∑ i, y.coeff i / crGcd y * ρ i) * crGcd y = ∑ i, y.coeff i * ρ i := by
            rw [Finset.sum_mul]
            refine Finset.sum_congr rfl fun i _ => ?_
            calc y.coeff i / crGcd y * ρ i * crGcd y
                = y.coeff i / crGcd y * crGcd y * ρ i := by ring
              _ = y.coeff i * ρ i := by rw [Int.ediv_mul_cancel (crGcd_dvd y i)]
          obtain ⟨hl, hu⟩ := hyh
          refine ⟨?_, ?_⟩
          · cases hly : y.lo with
            | none => simp
            | some l =>
                rw [hly] at hl
                simp only [Option.elim, CRow.dotv] at hl
                simp only [Option.map_some, Option.elim, CRow.dotv]
                rw [neg_le, Int.le_ediv_iff_mul_le hgpos]
                have hexp : -(∑ i, y.coeff i / crGcd y * ρ i) * crGcd y
                    = -((∑ i, y.coeff i / crGcd y * ρ i) * crGcd y) := by ring
                rw [hexp, hdv]
                omega
          · cases huy : y.up with
            | none => simp
            | some u =>
                rw [huy] at hu
                simp only [Option.elim, CRow.dotv] at hu
                simp only [Option.map_some, Option.elim, CRow.dotv]
                rw [Int.le_ediv_iff_mul_le hgpos, hdv]
                exact hu

/-- A row whose bounds have passed each other.

    (x : CRow n)                             [a certificate row]
    : Bool                                   [true exactly when its interval is a crossing]

    So: what `ccStepX` requires before it accepts. -/
def crImpossible (x : CRow n) : Bool :=
  match x.lo, x.up with
  | some l, some u => decide (u < l)
  | _, _ => false

/-- The closing test is the algebra's crossing.

    (l u : ℤ) (c : Fin n → ℤ)                                [two bounds and the coefficients]
    crImpossible ⟨some l, some u, c⟩ = true                  [the row reads as impossible]
    ↔                                                        [exactly when, both directions]
    Crossing 1 (intervalOf 1 l u)                            [that point of the algebra crosses]

    So: `Bridge.crossing_ofBounds` at the certificate's own closing step. The verdict rides the
    order `u < l`, which `crossing_independent_of_cone` shows is a different locus from the null
    cone, and `Fragment.norm_not_a_row` shows the cone is not expressible as a row at all. -/
theorem impossible_is_crossing (l u : ℤ) (c : Fin n → ℤ) :
    crImpossible ⟨some l, some u, c⟩ = true ↔ Crossing 1 (intervalOf 1 l u) := by
  rw [crossing_intervalOf (by norm_num : (0 : ℝ) < 1)]
  simp [crImpossible]

/-- An impossible row holds at no assignment.

    {ρ : Fin n → ℤ} {x : CRow n}             [an assignment and a certificate row]
    crImpossible x = true                    [the row's interval is a crossing]
    ¬ CRow.Holds ρ x                         [so it holds at no assignment]

    So: `Face.crossing_empty`, which turns a crossing into a refutation. -/
theorem crImpossible_empty {ρ : Fin n → ℤ} {x : CRow n} (h : crImpossible x = true) :
    ¬ CRow.Holds ρ x := by
  intro hx
  obtain ⟨xl, xu, xc⟩ := x
  cases xl with
  | none => simp [crImpossible] at h
  | some l =>
      cases xu with
      | none => simp [crImpossible] at h
      | some u =>
          refine crossing_empty (σ := 1) (by norm_num) l u
            ((⟨some l, some u, xc⟩ : CRow n).dotv ρ)
            ((impossible_is_crossing l u xc).mp h) ?_
          rw [mem_intervalOf (by norm_num)]
          exact ⟨hx.1, hx.2⟩

/-- One row operation or terminal refutation in the canonical certificate language.

    CertOp                                   [one line of a certificate]

    The row fragment uses `r`, `c`, `n`, `m`, `t`, and `x`. The residue terminal carries a finite
    tree checked against the original system, extending that fragment with complete refutations. -/
inductive CertOp : Type
  /-- `r k`: input line k of the system, verbatim. -/
  | load (k : ℕ)
  /-- `c i j a b`: a times row i plus b times row j, both multipliers at least one. -/
  | combo (i j : ℕ) (a b : ℤ)
  /-- `n i`: the negation of row i. -/
  | neg (i : ℕ)
  /-- `m i j`: the meet of two rows with equal coefficients. -/
  | meet (i j : ℕ)
  /-- `t i`: row i tidied by the coefficient divisor. -/
  | tidy (i : ℕ)
  /-- `x i`: row i is impossible, and it ends the certificate. -/
  | close (i : ℕ)
  /-- A finite residue-refutation tree for the input system, ending the certificate. -/
  | residue (cert : ResidueCertificate)
  deriving DecidableEq, Repr

/-- Every row a replay has built holds at an assignment satisfying the system.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    (rs : List (CRow n))                     [the rows the replay has built so far]
    : Prop                                   [a statement that is true or false] -/
def CertHolds (ρ : Fin n → ℤ) (rs : List (CRow n)) : Prop := ∀ x ∈ rs, CRow.Holds ρ x

/-- One replay step: extend the row list, or refuse.

    (raw : List (Line n))                    [the system, as it arrived on the wire]
    (rs : List (CRow n))                     [the rows built so far]
    CertOp → Option (List (CRow n))          [one step, and the rows after it]

    So: `ccStepR`, `ccStepC`, `ccStepN`, `ccStepM` and `ccStepT`. A step naming a row that does
    not exist refuses, a `c` with a multiplier below one refuses, and an `m` over unequal
    coefficients refuses, each exactly as `certcheck` refuses it. Both terminal constructors are
    handled by `checkCert` and never append a row here. -/
def certStep (raw : List (Line n)) (rs : List (CRow n)) : CertOp → Option (List (CRow n))
  | .load k =>
      match raw[k]? with
      | some l => some (rs ++ [certRaw l])
      | none => none
  | .combo i j a b =>
      if 1 ≤ a ∧ 1 ≤ b then
        match rs[i]?, rs[j]? with
        | some x, some y => some (rs ++ [crCombo a b x y])
        | _, _ => none
      else none
  | .neg i =>
      match rs[i]? with
      | some x => some (rs ++ [crNeg x])
      | none => none
  | .meet i j =>
      match rs[i]?, rs[j]? with
      | some x, some y => if x.coeff = y.coeff then some (rs ++ [crMeet x y]) else none
      | _, _ => none
  | .tidy i =>
      match rs[i]? with
      | some x => some (rs ++ [crTidy x])
      | none => none
  | .close _ => none
  | .residue _ => none

/-- Replay the log and accept only at a checked terminal with no remaining operations.

    (raw : List (Line n))                    [the system, as it arrived on the wire]
    (rs : List (CRow n))                     [the rows built so far]
    List CertOp → Bool                       [the log, and whether it is accepted]

    The `close` terminal names a row whose bounds cross. The `residue` terminal independently
    checks a finite tree against the loaded input rows. Nothing may follow either terminal. -/
def checkCert (raw : List (Line n)) (rs : List (CRow n)) : List CertOp → Bool
  | [] => false
  | .close i :: rest =>
      match rs[i]? with
      | some x => crImpossible x && rest.isEmpty
      | none => false
  | .residue cert :: rest =>
      checkResidueCertificate n (loadRows raw) cert && rest.isEmpty
  | s :: rest =>
      match certStep raw rs s with
      | some rs' => checkCert raw rs' rest
      | none => false

/-- Adding a row that holds to a list of rows that hold keeps them all holding.

    {ρ : Fin n → ℤ} {rs : List (CRow n)} {x : CRow n}   [an assignment, some rows, one more]
    CertHolds ρ rs                                      [the rows all hold]
    CRow.Holds ρ x                                      [and so does the new one]
    CertHolds ρ (rs ++ [x])                             [so do all of them together] -/
theorem certHolds_snoc {ρ : Fin n → ℤ} {rs : List (CRow n)} {x : CRow n}
    (h : CertHolds ρ rs) (hx : CRow.Holds ρ x) : CertHolds ρ (rs ++ [x]) := by
  intro y hy
  rcases List.mem_append.mp hy with hm | hm
  · exact h y hm
  · rw [List.mem_singleton.mp hm]; exact hx

/-- A row the replay names is one of the rows it has built.

    {ρ : Fin n → ℤ} {rs : List (CRow n)} {i : ℕ} {x : CRow n}  [an assignment, rows, an index]
    CertHolds ρ rs                                             [every built row holds]
    rs[i]? = some x                                            [the index names one of them]
    CRow.Holds ρ x                                             [so that one holds] -/
theorem holds_of_getElem {ρ : Fin n → ℤ} {rs : List (CRow n)} {i : ℕ} {x : CRow n}
    (h : CertHolds ρ rs) (hi : rs[i]? = some x) : CRow.Holds ρ x :=
  h x (List.mem_of_getElem? hi)

/-- Every replay step maps rows that hold to rows that hold.

    {ρ : Fin n → ℤ}                                       [an assignment]
    {raw : List (Line n)} {rs rs' : List (CRow n)}         [the system and the rows]
    {s : CertOp}                                           [one step of the certificate]
    SatLines ρ raw                                         [the assignment satisfies the system]
    CertHolds ρ rs                                         [every built row holds]
    certStep raw rs s = some rs'                           [the step is taken]
    CertHolds ρ rs'                                        [every row after it holds]

    So: the five operation soundness theorems in one place. `certRaw_holds` for `r`,
    `crCombo_holds` for `c`, which is `Fragment.combo_holds` through `combo_is_farkas`,
    `crNeg_holds` for `n`, `crMeet_holds` for `m`, which is `inCell_meet_iff` transported, and
    `crTidy_holds` for `t`, the cutting plane. -/
theorem certStep_holds {ρ : Fin n → ℤ} {raw : List (Line n)} {rs rs' : List (CRow n)}
    {s : CertOp} (hraw : SatLines ρ raw) (h : CertHolds ρ rs)
    (he : certStep raw rs s = some rs') : CertHolds ρ rs' := by
  cases s with
  | load k =>
      simp only [certStep] at he
      split at he
      · next l hk =>
          rw [Option.some.injEq] at he
          subst he
          exact certHolds_snoc h (certRaw_holds ρ l (hraw l (List.mem_of_getElem? hk)))
      · simp at he
  | combo i j a b =>
      simp only [certStep] at he
      split at he
      · next hab =>
          split at he
          · next x y hx hy =>
              rw [Option.some.injEq] at he
              subst he
              exact certHolds_snoc h
                (crCombo_holds hab.1 hab.2 (holds_of_getElem h hx) (holds_of_getElem h hy))
          · simp at he
      · simp at he
  | neg i =>
      simp only [certStep] at he
      split at he
      · next x hx =>
          rw [Option.some.injEq] at he
          subst he
          exact certHolds_snoc h (crNeg_holds (holds_of_getElem h hx))
      · simp at he
  | meet i j =>
      simp only [certStep] at he
      split at he
      · next x y hx hy =>
          split at he
          · next hc =>
              rw [Option.some.injEq] at he
              subst he
              exact certHolds_snoc h
                (crMeet_holds hc (holds_of_getElem h hx) (holds_of_getElem h hy))
          · simp at he
      · simp at he
  | tidy i =>
      simp only [certStep] at he
      split at he
      · next x hx =>
          rw [Option.some.injEq] at he
          subst he
          exact certHolds_snoc h (crTidy_holds (holds_of_getElem h hx))
      · simp at he
  | close i => simp [certStep] at he
  | residue cert => simp [certStep] at he

/-- A replay that starts from rows that hold never accepts.

    {ρ : Fin n → ℤ} {raw : List (Line n)}                 [an assignment and the system]
    SatLines ρ raw                                        [the assignment satisfies the system]
    ∀ (log : List CertOp) (rs : List (CRow n))            [for every log and every row list]
    CertHolds ρ rs                                        [whose rows all hold]
    checkCert raw rs log = false                          [the checker refuses the log]

    Row operations preserve every solution. The crossing terminal contradicts a preserved row;
    the residue terminal contradicts satisfaction of the original loaded system. -/
theorem checkCert_refutes {ρ : Fin n → ℤ} {raw : List (Line n)} (hraw : SatLines ρ raw) :
    ∀ (log : List CertOp) (rs : List (CRow n)), CertHolds ρ rs → checkCert raw rs log = false
  | [], _, _ => rfl
  | .close i :: rest, rs, h => by
      show (match rs[i]? with
            | some x => crImpossible x && rest.isEmpty
            | none => false) = false
      split
      · next x hx =>
          cases hb : crImpossible x with
          | true => exact absurd (holds_of_getElem h hx) (crImpossible_empty hb)
          | false => rfl
      · rfl
  | .residue cert :: rest, _, _ => by
      have hfalse : checkResidueCertificate n (loadRows raw) cert = false := by
        apply Bool.eq_false_iff.mpr
        intro hcert
        exact checkResidueCertificate_sound n (loadRows raw) cert hcert
          ⟨ρ, satRows_loadRows hraw⟩
      simp only [checkCert, hfalse, Bool.false_and]
  | .load k :: rest, rs, h => by
      show (match certStep raw rs (.load k) with
            | some rs' => checkCert raw rs' rest
            | none => false) = false
      split
      · next rs' he => exact checkCert_refutes hraw rest rs' (certStep_holds hraw h he)
      · rfl
  | .combo i j a b :: rest, rs, h => by
      show (match certStep raw rs (.combo i j a b) with
            | some rs' => checkCert raw rs' rest
            | none => false) = false
      split
      · next rs' he => exact checkCert_refutes hraw rest rs' (certStep_holds hraw h he)
      · rfl
  | .neg i :: rest, rs, h => by
      show (match certStep raw rs (.neg i) with
            | some rs' => checkCert raw rs' rest
            | none => false) = false
      split
      · next rs' he => exact checkCert_refutes hraw rest rs' (certStep_holds hraw h he)
      · rfl
  | .meet i j :: rest, rs, h => by
      show (match certStep raw rs (.meet i j) with
            | some rs' => checkCert raw rs' rest
            | none => false) = false
      split
      · next rs' he => exact checkCert_refutes hraw rest rs' (certStep_holds hraw h he)
      · rfl
  | .tidy i :: rest, rs, h => by
      show (match certStep raw rs (.tidy i) with
            | some rs' => checkCert raw rs' rest
            | none => false) = false
      split
      · next rs' he => exact checkCert_refutes hraw rest rs' (certStep_holds hraw h he)
      · rfl

/-- **A certificate the checker accepts refutes the system.**

    (raw : List (Line n)) (log : List CertOp)             [a wire system and a certificate]
    checkCert raw [] log = true                           [the checker accepts the certificate]
    ¬ ∃ ρ : Fin n → ℤ, SatLines ρ raw                     [no assignment satisfies the system]

    So: the theorem the whole trust argument rests on. No property of
    `../song/src/prover/cone/ncone0.slang` is used: the log and the system it names are the
    only inputs, so a decider that emits a wrong certificate is caught rather than believed.
    The row-operation fragment uses `Fragment.Row` and the trichotomy's meet and crossing.
    The residue terminal uses the independently checked finite projection tree. -/
theorem checkCert_sound (raw : List (Line n)) (log : List CertOp)
    (h : checkCert raw [] log = true) : ¬ ∃ ρ : Fin n → ℤ, SatLines ρ raw := by
  rintro ⟨ρ, hsat⟩
  have hfalse := checkCert_refutes hsat log [] (fun x hx => absurd hx (List.not_mem_nil))
  rw [hfalse] at h
  exact Bool.noConfusion h

/-- Construct a canonical log containing the finite residue-refutation candidate.

    (raw : List (Line n))                  [the input wire system]
    : List CertOp                          [the finite certificate candidate]

    The checker accepts this candidate exactly on infeasible inputs. -/
def synthesizeCert (raw : List (Line n)) : List CertOp :=
  [.residue (synthesizeResidueCertificate n (loadRows raw))]

/-- Every integer-infeasible input has an accepted executable certificate construction.

    (raw : List (Line n))                  [the input wire system]
    (h : ¬ ∃ ρ : Fin n → ℤ, SatLines ρ raw) [no integer assignment satisfies it]
    checkCert raw [] (synthesizeCert raw) = true [the constructed log passes replay] -/
theorem synthesizeCert_accepts (raw : List (Line n))
    (h : ¬ ∃ ρ : Fin n → ℤ, SatLines ρ raw) :
    checkCert raw [] (synthesizeCert raw) = true := by
  have hloaded : ¬ ∃ ρ : Fin n → ℤ, SatRows ρ (loadRows raw) := by
    rintro ⟨ρ, hρ⟩
    apply h
    refine ⟨ρ, (satLines_iff_four_squares ρ raw).mpr ?_⟩
    intro r hr
    exact (holds_iff_four_squares ρ r).mp (hρ r hr)
  have hcert := synthesizeResidueCertificate_accepts n (loadRows raw)
    ((completeSat_eq_false_iff n (loadRows raw)).mpr hloaded)
  simpa only [synthesizeCert, checkCert, List.isEmpty_nil, Bool.and_true] using hcert

/-- **Canonical certificates exist exactly for integer-infeasible input systems.**

    (raw : List (Line n))                  [the uncapped input wire system]
    (∃ log : List CertOp,                 [some finite canonical log exists]
      checkCert raw [] log = true)         [and passes replay from no derived rows]
    ↔ ¬ ∃ ρ : Fin n → ℤ, SatLines ρ raw  [exactly when no integer assignment satisfies the input] -/
theorem certificate_complete (raw : List (Line n)) :
    (∃ log : List CertOp, checkCert raw [] log = true) ↔
      ¬ ∃ ρ : Fin n → ℤ, SatLines ρ raw := by
  constructor
  · rintro ⟨log, h⟩
    exact checkCert_sound raw log h
  · intro h
    exact ⟨synthesizeCert raw, synthesizeCert_accepts raw h⟩

#guard checkCert [Line.eq (⟨-1, ![2]⟩ : Row 1)] []
  (synthesizeCert [Line.eq (⟨-1, ![2]⟩ : Row 1)])
#guard !checkCert [Line.eq (⟨-1, ![2]⟩ : Row 1)] []
  (synthesizeCert [Line.eq (⟨-1, ![2]⟩ : Row 1)] ++ [.load 0])
#guard !checkCert [Line.eq (⟨-2, ![2]⟩ : Row 1)] []
  (synthesizeCert [Line.eq (⟨-2, ![2]⟩ : Row 1)])

/-- Every row a replay builds is four squares away from its own bounds.

    {ρ : Fin n → ℤ} {raw : List (Line n)} {rs : List (CRow n)}   [assignment, system, rows]
    SatLines ρ raw                                               [the system is satisfied]
    CertHolds ρ rs                                               [every built row holds]
    ∀ x ∈ rs, ∀ l ∈ x.lo, ∃ w y z t : ℤ                          [for every row and lower bound]
    Row.value ρ (loRow x l) = w ^ 2 + y ^ 2 + z ^ 2 + t ^ 2      [there are four such squares]

    So: `Fragment.holds_iff_four_squares` at every row of a replay. What the checker asserts at
    each step is a statement in the null channel's own ring, which is `cell.slang`'s C3 entry
    condition read along the certificate rather than at one operation. -/
theorem cert_four_squares {ρ : Fin n → ℤ} {rs : List (CRow n)} (h : CertHolds ρ rs) :
    ∀ x ∈ rs, ∀ l ∈ x.lo, ∃ w y z t : ℤ,
      Row.value ρ (loRow x l) = w ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 := by
  intro x hx l hl
  refine (holds_iff_four_squares ρ (loRow x l)).mp ?_
  rw [loRow_holds_iff]
  have := (h x hx).1
  rw [hl] at this
  exact this

/-- A nonnegative combination that grounds to a negative constant refutes the system.

    (rs : List (Row n)) (ws : List (ℤ × Row n))          [a system and weighted rows from it]
    (∀ p ∈ ws, 0 ≤ p.1)                                  [every weight is at least zero]
    (∀ p ∈ ws, p.2 ∈ rs)                                 [every weighted row is the system's]
    (∀ i, (Fragment.combo ws).coeff i = 0)               [the combination has no columns left]
    ((Fragment.combo ws).const < 0)                      [and its constant is below zero]
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs                      [no assignment satisfies the system]

    So: `Fragment.unsat_of_negative_combo`, restated at this file's `SatRows` so the
    certificate files can reach it. What it says is about a combination, and what it does not
    say is that any particular certificate produces one.

    The relation between the two is exact on one side and prose on the other, and the split is
    worth stating. `combo_is_farkas` proves the `c` operation is one `Fragment.combo` step, so a
    log of `r` lines followed by `c` lines does accumulate a combination of this shape, and in
    the certificates `../song/src/prover/cone/ncone0.slang` prints, the closing `t` grounds a
    row that already has no columns, where the tidy is a canonicalisation rather than a cutting
    plane. Turning that observation into a theorem needs an induction extracting the
    weights from a whole log, and that induction is not here: `checkCert_sound` refutes the
    system by carrying "every built row holds" along the log instead, which covers logs whose
    `t` is a genuine cutting plane as well, and those have no Farkas combination at all. -/
theorem negative_combo_refutes (rs : List (Row n)) (ws : List (ℤ × Row n))
    (hw : ∀ p ∈ ws, 0 ≤ p.1) (hm : ∀ p ∈ ws, p.2 ∈ rs)
    (hz : ∀ i, (Fragment.combo ws).coeff i = 0) (hn : (Fragment.combo ws).const < 0) :
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs :=
  Fragment.unsat_of_negative_combo rs ws hw hm hz hn

end Decider
end Apeiron
