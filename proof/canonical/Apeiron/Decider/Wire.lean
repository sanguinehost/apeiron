import Apeiron.Fragment.Basic
import Apeiron.Decider.Face

/-!
# The wire format ncone0 reads

@name The Wire Format
@grade proved

`Apeiron/Fragment/Basic.lean` fixes what a row is and proves the format cannot state a product.
This file is the layer above it: what arrives on `../song/src/prover/cone/ncone0.slang`'s own
standard input, and how those lines become rows.

## One line, one or two rows

The wire is text. A line beginning `g` carries a constant and then one coefficient per column, a
line beginning `e` the same but asserted equal to zero:

```
   g -1 1 0        means      -1 + 1·x₀ + 0·x₁ ≥ 0
   e  0 1 -1       means       0 + 1·x₀ - 1·x₁ = 0
```

`Line` is what arrives and `Fragment.Row` is what the decider holds, because an equality is one
line and two rows: `loadLoop` splits it at load time into `c + a·x ≥ 0` and `-c - a·x ≥ 0`, and
`loadRows` performs the same split. That split is why a certificate's `r k` names an input line
rather than a row, and why `certcheck` keeps a two-sided interval for a line that arrived as `e`.

## Root

| from | here |
|---|---|
| `Fragment.two_caps_iff_disc_pos` (from `two_channels_iff_disc_pos`) | `eq_line_is_two_caps` |
| `Fragment.holds_iff_four_squares` (from the four-squares order) | `satLines_iff_four_squares` |
| `Decider.face_is_forced`, `Decider.disc_one_of_integer_basis` | the face these lines are read on |

An `e` line is the two-sided cap, and `eq_line_is_two_caps` says so: it holds exactly when the
row and its negation both hold, which is one order from each side. `Fragment.two_caps_iff_disc_pos`
says two orders on one point need two independent channels, and those exist exactly above zero,
so the wire's own second line kind is what puts the decider on the split face.
`satLines_iff_four_squares` then restates a whole wire system with no order symbol in it at all,
by `Fragment.holds_iff_four_squares`, so the relation every step of the decider tests is one the
null channel's ring already defines.

## The bound that is not decoration

ncone0's row arithmetic runs in machine words and refuses to wrap: `mulChk` and `addChk` set an
overflow flag the moment a product or sum passes `NC_MAG`, and a set flag downgrades the verdict
to MAYBE. `NC_MAG` is stated here as `2 ^ 63 - 1`, and `sum_can_leave_range` exhibits two in-range
rows whose sum is out of range, so the guard is doing work. This model computes in `ℤ`, which does
not wrap, so the guard is not part of its soundness argument; it is part of the argument that the
compiled program agrees with the model, which is the refinement gap named in
`Apeiron/Decider/Model.lean`.
-/

namespace Apeiron
namespace Decider

open Trichotomy Fragment

variable {n : ℕ}

/-- A row with every sign flipped.

    (r : Row n)                              [a row over n columns]
    : Row n                                  [the same row multiplied through by minus one]

    So: the second half of an equality split, and the certificate's `n` operation. -/
def negRow (r : Row n) : Row n := r.scale (-1)

/-- Negating a row negates its value.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    (r : Row n)                              [a row over n columns]
    Row.value ρ (negRow r) = - Row.value ρ r [the value flips sign] -/
@[simp] theorem value_negRow (ρ : Fin n → ℤ) (r : Row n) :
    Row.value ρ (negRow r) = - Row.value ρ r := by
  simp [negRow]

/-- One line of the wire: an inequality, or an equality.

    Line (n : ℕ)                             [a wire line over n columns]

    So: `g c a₁ … aₙ` and `e c a₁ … aₙ`, the two line kinds `loadLoop` accepts. A certificate's
    `r k` names the k-th of these, which is why the type exists separately from `Row`. -/
inductive Line (n : ℕ) : Type
  /-- `g c as`: the inequality `c + a·x ≥ 0`. -/
  | ge (r : Row n)
  /-- `e c as`: the equality `c + a·x = 0`, which loads as two rows. -/
  | eq (r : Row n)

/-- The rows one wire line loads as.

    Line n → List (Row n)                    [one wire line, and the rows it becomes]

    So: an inequality is one row and an equality is two, exactly `loadEqNeg`'s split in
    ncone0. Nothing downstream sees an equality again. -/
def Line.rows : Line n → List (Row n)
  | .ge r => [r]
  | .eq r => [r, negRow r]

/-- An assignment satisfies a line when it satisfies what the line says.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    Line n → Prop                            [each line kind, and whether it is met]

    So: `Holds` for `g`, and exactly zero for `e`. -/
def Line.Sat (ρ : Fin n → ℤ) : Line n → Prop
  | .ge r => Row.Holds ρ r
  | .eq r => Row.value ρ r = 0

/-- An equality line is a cap from each side.

    (ρ : Fin n → ℤ)                                       [a whole number for each column]
    (r : Row n)                                           [a row over n columns]
    Line.Sat ρ (Line.eq r)                                [the equality line is met]
    ↔                                                     [exactly when, true both directions]
    Row.Holds ρ r ∧ Row.Holds ρ (negRow r)                [the row and its negation both hold]

    So: the wire's second line kind is a two-sided cap, and `Fragment.two_caps_iff_disc_pos`
    says a two-sided cap needs two independent readings of a point, which exist exactly when
    σ is above zero. The format's own equality line is what puts the decider on the
    split face, before any propagation happens. -/
theorem eq_line_is_two_caps (ρ : Fin n → ℤ) (r : Row n) :
    Line.Sat ρ (Line.eq r) ↔ Row.Holds ρ r ∧ Row.Holds ρ (negRow r) := by
  simp only [Line.Sat, Row.Holds, value_negRow]
  omega

/-- An assignment satisfies a wire system when it satisfies every line of it.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    (ls : List (Line n))                     [the system as it arrives on the wire]
    : Prop                                   [a statement that is true or false]

    So: the property a verdict of UNSAT denies, for every assignment at once. -/
def SatLines (ρ : Fin n → ℤ) (ls : List (Line n)) : Prop := ∀ l ∈ ls, Line.Sat ρ l

/-- The rows a whole wire system loads as.

    (ls : List (Line n))                     [the system as it arrives on the wire]
    : List (Row n)                           [the rows the decider works on]

    So: `loadLoop`, with equalities split. -/
def loadRows (ls : List (Line n)) : List (Row n) := ls.flatMap Line.rows

/-- Every row of a system holds when the system's lines are met.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    (rs : List (Row n))                      [a list of rows]
    : Prop                                   [a statement that is true or false] -/
def SatRows (ρ : Fin n → ℤ) (rs : List (Row n)) : Prop := ∀ r ∈ rs, Row.Holds ρ r

/-- Loading preserves satisfaction.

    {ρ : Fin n → ℤ} {ls : List (Line n)}     [an assignment and a wire system]
    SatLines ρ ls                            [the assignment meets every line]
    SatRows ρ (loadRows ls)                  [so it meets every loaded row]

    So: the equality split loses nothing, so refuting the loaded rows refutes the wire system.
    This is the first link of the soundness chain and the only place the split is used. -/
theorem satRows_loadRows {ρ : Fin n → ℤ} {ls : List (Line n)} (h : SatLines ρ ls) :
    SatRows ρ (loadRows ls) := by
  intro r hr
  simp only [loadRows, List.mem_flatMap] at hr
  obtain ⟨l, hl, hrl⟩ := hr
  have hls := h l hl
  cases l with
  | ge s =>
      have hcase : r = s := by simpa [Line.rows] using hrl
      subst hcase
      exact hls
  | eq s =>
      have hcase : r = s ∨ r = negRow s := by simpa [Line.rows] using hrl
      rcases (eq_line_is_two_caps ρ s).mp hls with ⟨h1, h2⟩
      rcases hcase with rfl | rfl
      · exact h1
      · exact h2

/-- A whole wire system, stated with no order symbol in it.

    (ρ : Fin n → ℤ) (ls : List (Line n))                    [an assignment and a wire system]
    SatLines ρ ls                                           [every line is met]
    ↔                                                       [exactly when, both directions]
    ∀ r ∈ loadRows ls                                       [for every loaded row]
    ∃ w x y z : ℤ                                           [there are four whole numbers]
    Row.value ρ r = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2           [whose squares add to its value]

    So: `Fragment.holds_iff_four_squares` lifted from one row to the wire. The property a
    verdict of UNSAT denies is a statement in the null channel's own ring, using multiplication
    and addition alone. -/
theorem satLines_iff_four_squares (ρ : Fin n → ℤ) (ls : List (Line n)) :
    SatLines ρ ls ↔
      ∀ r ∈ loadRows ls, ∃ w x y z : ℤ,
        Row.value ρ r = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2 := by
  constructor
  · intro h r hr
    exact (holds_iff_four_squares ρ r).mp (satRows_loadRows h r hr)
  · intro h l hl
    have hmem : ∀ s ∈ Line.rows l, Row.Holds ρ s := by
      intro s hs
      refine (holds_iff_four_squares ρ s).mpr (h s ?_)
      simp only [loadRows, List.mem_flatMap]
      exact ⟨l, hl, hs⟩
    cases l with
    | ge s => exact hmem s (by simp [Line.rows])
    | eq s =>
        refine (eq_line_is_two_caps ρ s).mpr ⟨hmem s (by simp [Line.rows]), ?_⟩
        exact hmem (negRow s) (by simp [Line.rows])

/-- The magnitude ncone0's own arithmetic refuses to pass.

    : ℤ                                      [one whole number, the guard's ceiling]

    So: `NC_MAG` in `../song/src/prover/cone/ncone0.slang`, the largest signed machine word.
    `mulChk` and `addChk` set the overflow flag rather than wrap once a product or sum leaves
    this range, and a set flag turns UNSAT and SAT alike into MAYBE. -/
def NC_MAG : ℤ := 2 ^ 63 - 1

/-- The ceiling is the literal ncone0 writes.

    NC_MAG = 9223372036854775807             [the guard's ceiling, spelled out]

    So: the model's constant and the decider's constant are the same number, checked rather
    than asserted. -/
theorem nc_mag_eq : NC_MAG = 9223372036854775807 := by norm_num [NC_MAG]

/-- A row is in range when its constant and every coefficient fit under the ceiling.

    (r : Row n)                              [a row over n columns]
    : Prop                                   [a statement that is true or false]

    So: the condition every row the compiled decider holds must meet for its arithmetic to be
    exact. This model works in `ℤ` and never needs it; the compiled program does. -/
def InRange (r : Row n) : Prop :=
  |r.const| ≤ NC_MAG ∧ ∀ i, |r.coeff i| ≤ NC_MAG

/-- Two rows in range can add to a row out of range.

    ∃ r s : Row n                            [there are two rows over n columns]
    InRange r ∧ InRange s                    [both fit under the ceiling]
    ¬ InRange (Row.add r s)                  [their sum does not]

    So: the overflow guard is load-bearing rather than defensive. A model over `ℤ` has no
    ceiling and needs none; the compiled decider computes in machine words, and this is the
    case its guard exists for. It is why an overflow downgrades the verdict to MAYBE rather
    than being ignored, and why `certcheck` replays a certificate at a wider domain than the
    decider derived it in. -/
theorem sum_can_leave_range :
    ∃ r s : Row n, InRange r ∧ InRange s ∧ ¬ InRange (Row.add r s) := by
  have hone : |NC_MAG| ≤ NC_MAG := le_of_eq (abs_of_nonneg (by norm_num [NC_MAG]))
  have hzero : |(0 : ℤ)| ≤ NC_MAG := by norm_num [NC_MAG]
  refine ⟨Row.ofConst NC_MAG, Row.ofConst NC_MAG, ⟨hone, fun _ => hzero⟩,
    ⟨hone, fun _ => hzero⟩, ?_⟩
  intro hcon
  have h := hcon.1
  simp only [Row.add, Row.ofConst] at h
  rw [abs_le] at h
  have h2 := h.2
  norm_num [NC_MAG] at h2

end Decider
end Apeiron
