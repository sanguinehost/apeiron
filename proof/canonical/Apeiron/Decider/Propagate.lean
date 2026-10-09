import Apeiron.Decider.Wire

/-!
# Phase one: bound propagation to a fixed point

@name Bound Propagation
@grade proved

`../song/src/prover/cone/ncone0.slang` decides most systems without eliminating anything. It
holds one interval per column, walks each row, and for each column derives what that row forces
given the intervals the other columns already carry. This file is that phase, and its soundness:
every solution of the system survives every step, so a column whose interval crosses refutes the
system.

## The step, in three pieces

For a row `c + a·x ≥ 0` and a column `j` with `a_j ≠ 0`, move every other column to the other
side. Each term `a_k · x_k` is bounded by `a_k` times an endpoint of `x_k`'s interval, and which
endpoint depends only on the sign of `a_k`: the top for a positive coefficient, the bottom for a
negative one. That is `propPick`. The right-hand side that comes out is `rhsOf`. Dividing by
`a_j`, rounding away from the feasible side, gives a new bound, and the column's interval becomes
the meet of what it held and what the row just derived. That is `Bnd.meet`, whose two channels
are `propNewLo` taking the larger bottom and `propNewHi` taking the smaller top.

## Root

| from | here |
|---|---|
| `conj_ofBounds`, `upper_conj` | `propPick_is_conj` |
| `meet_ofBounds`, `inCell_meet_iff` | `bnd_meet_holds` (two-sided case), `meet_is_the_algebra` |
| `crossing_ofBounds`, `crossing_independent_of_cone` | `crossed_iff_crossing`, `crossed_empty` |
| `Fragment.holds_iff_four_squares` | `derived_bound_four_squares` |
| `face_is_forced`, `disc_one_of_integer_basis` | the face the interval is held on |

Sigma stays free wherever the statement allows it. `propPick_is_conj`, `meet_is_the_algebra` and
`crossed_iff_crossing` are proved by instantiating theorems of `Apeiron/Decider/Face.lean` that
are themselves stated over `Cell σ` with `0 < σ`, at the setting `disc_one_of_integer_basis`
forces. Nothing here re-proves a fact about the algebra; it applies one.

## The missing primitive, named

A column whose interval is bounded on one side only has no point of `Cell σ` behind it. The
algebra's two channels both read a real number, so an interval with no bottom would need a
channel reading minus infinity, and the corpus has no compactified channel. ncone0 stands in the
same place and uses the same stand-in: `BND_FLO` and `BND_FHI` are flags, and
`../song/src/prover/cone/cell.slang` says in its own words that a flag stands in for Kaucher's
infinity. So `bnd_meet_holds` splits: the case where both intervals are two-sided is
`Face.mem_meet_intervalOf`, which is `inCell_meet_iff` transported, and the one-sided cases are
discharged by hand because there is nothing to transport them to.

The primitive that would close this is a null channel completed with two points at infinity,
which is the projective line `Apeiron/Projective/Basic.lean` already builds on each channel. With
it, `Bnd` is a cell of the completed algebra outright, `maxOpt` and `minOpt` are the meet with no
cases, and the flags disappear. Not built here.
-/

namespace Apeiron
namespace Decider

open Trichotomy Fragment

variable {n : ℕ}

/-- What one column's interval holds, as ncone0 holds it.

    Bnd                                      [a bottom and a top, each of which may be absent]

    So: `BND_LO` and `BND_HI` with their `BND_FLO` and `BND_FHI` flags folded in. An absent
    endpoint is a column the propagation has not bounded on that side yet. -/
structure Bnd where
  /-- The bottom endpoint, if there is one. -/
  lo : Option ℤ
  /-- The top endpoint, if there is one. -/
  hi : Option ℤ
  deriving DecidableEq, Repr

/-- The interval that bounds nothing.

    Bnd                                      [a bottom and a top, both absent]

    So: what `clearBounds` leaves behind at the start of every propagation pass. -/
def Bnd.top : Bnd := ⟨none, none⟩

/-- A whole number lies in an interval when it meets whichever endpoints are present.

    (b : Bnd)                                [one column's interval]
    (x : ℤ)                                  [a whole number]
    : Prop                                   [a statement that is true or false] -/
def BndHolds (b : Bnd) (x : ℤ) : Prop :=
  b.lo.elim True (fun l => l ≤ x) ∧ b.hi.elim True (fun h => x ≤ h)

/-- An assignment lies in a box when every column lies in its own interval.

    (b : Fin n → Bnd)                        [one interval per column]
    (ρ : Fin n → ℤ)                          [a whole number for each column]
    : Prop                                   [a statement that is true or false] -/
def BoxHolds (b : Fin n → Bnd) (ρ : Fin n → ℤ) : Prop := ∀ i, BndHolds (b i) (ρ i)

/-- The empty box holds every assignment.

    (ρ : Fin n → ℤ)                          [a whole number for each column]
    BoxHolds (fun _ => Bnd.top) ρ            [the box that bounds nothing holds it] -/
theorem boxHolds_top (ρ : Fin n → ℤ) : BoxHolds (fun _ => Bnd.top) ρ :=
  fun _ => ⟨trivial, trivial⟩

/-- Which endpoint a coefficient's sign reads.

    (a : ℤ)                                  [a coefficient]
    (b : Bnd)                                [the column's interval]
    : Option ℤ                               [the endpoint that bounds this term, if present]

    So: `propPick` in ncone0, which reads `getHi` for a positive coefficient and `getLo` for a
    negative one. A zero coefficient contributes nothing, so the endpoint it reads is zero and
    is always present. -/
def propPick (a : ℤ) (b : Bnd) : Option ℤ :=
  if 0 < a then b.hi else if a < 0 then b.lo else some 0

/-- The larger of two bottom endpoints.

    (cur nv : ℤ)                             [the endpoint held, and the one derived]
    : ℤ                                      [the larger of the two]

    So: `propNewLo`, which only ever raises a bottom endpoint. -/
def propNewLo (cur nv : ℤ) : ℤ := max cur nv

/-- The smaller of two top endpoints.

    (cur nv : ℤ)                             [the endpoint held, and the one derived]
    : ℤ                                      [the smaller of the two]

    So: `propNewHi`, which only ever lowers a top endpoint. -/
def propNewHi (cur nv : ℤ) : ℤ := min cur nv

/-- Reading an endpoint by a coefficient's sign is the algebra's conjugation.

    (a lo hi : ℤ)                                              [a coefficient and an interval]
    a ≠ 0                                                      [the coefficient is not zero]
    upper 1 (intervalOf 1 lo hi)                               [the top read of the interval]
    conj (intervalOf 1 lo hi)                                  [the conjugate of that point]
    2 * ((if 0 < a then hi else lo : ℤ) : ℝ)                   [the endpoint read, doubled]
    ∧                                                          [both halves together]
    upper 1 (intervalOf 1 lo hi)                               [a positive coefficient reads
                                                                the top]
    upper 1 (conj (intervalOf 1 lo hi))                        [a negative one reads the
                                                                conjugate's top]
    propPick a ⟨some lo, some hi⟩ = some (if 0 < a then hi else lo)
                                                               [and that is what propPick gives]

    So: `Bridge.conj_ofBounds` reaching the running propagator. The sign test is not a case
    split bolted on to a loop; it is one read of the algebra applied to the point or to its
    conjugate, and `Face.pick_is_conj` is the equation that says so. -/
theorem propPick_is_conj (a lo hi : ℤ) (ha0 : a ≠ 0) :
    (if 0 < a then upper 1 (intervalOf 1 lo hi)
      else upper 1 (conj (intervalOf 1 lo hi)))
      = 2 * ((if 0 < a then hi else lo : ℤ) : ℝ)
    ∧ propPick a ⟨some lo, some hi⟩ = some (if 0 < a then hi else lo) := by
  refine ⟨pick_is_conj (by norm_num) a lo hi, ?_⟩
  by_cases ha : 0 < a
  · simp [propPick, ha]
  · have ha' : a < 0 := by omega
    simp [propPick, ha, ha']

/-- Two bottoms met, one of which may be absent.

    Option ℤ → Option ℤ → Option ℤ           [two bottoms, and the one that survives]

    So: `propNewLo` when both are present, and the present one when only one is. -/
def maxOpt : Option ℤ → Option ℤ → Option ℤ
  | none, y => y
  | x, none => x
  | some a, some b => some (propNewLo a b)

/-- Two tops met, one of which may be absent.

    Option ℤ → Option ℤ → Option ℤ           [two tops, and the one that survives]

    So: `propNewHi` when both are present, and the present one when only one is. -/
def minOpt : Option ℤ → Option ℤ → Option ℤ
  | none, y => y
  | x, none => x
  | some a, some b => some (propNewHi a b)

/-- Two intervals met channel by channel.

    (a c : Bnd)                              [two intervals for one column]
    : Bnd                                    [the interval of the tighter endpoints]

    So: the propagation step. The bottom takes the larger and the top the smaller, which is
    `meet` of the algebra when both are present. -/
def Bnd.meet (a c : Bnd) : Bnd := ⟨maxOpt a.lo c.lo, minOpt a.hi c.hi⟩

/-- The meet of two two-sided intervals is the algebra's meet.

    (l₁ h₁ l₂ h₂ : ℤ)                                                [two intervals]
    Bnd.meet ⟨some l₁, some h₁⟩ ⟨some l₂, some h₂⟩                    [meeting them here]
    = ⟨some (max l₁ l₂), some (min h₁ h₂)⟩                            [is the tighter interval]
    ∧                                                                 [and, on top of that]
    meet 1 (intervalOf 1 l₁ h₁) (intervalOf 1 l₂ h₂)                  [meeting the two points]
    = intervalOf 1 (max l₁ l₂) (min h₁ h₂)                            [gives the same interval]

    So: `Bridge.meet_ofBounds` with the propagator's own operation beside it. The two channels
    of one ring-level operation are exactly `propNewLo` and `propNewHi`, which ncone0 carries as
    four separate laws because its spec faces cannot be linked. -/
theorem meet_is_the_algebra (l₁ h₁ l₂ h₂ : ℤ) :
    Bnd.meet ⟨some l₁, some h₁⟩ ⟨some l₂, some h₂⟩
        = ⟨some (max l₁ l₂), some (min h₁ h₂)⟩ ∧
      meet 1 (intervalOf 1 l₁ h₁) (intervalOf 1 l₂ h₂)
        = intervalOf 1 (max l₁ l₂) (min h₁ h₂) :=
  ⟨by simp [Bnd.meet, maxOpt, minOpt, propNewLo, propNewHi], meet_intervalOf (by norm_num) _ _ _ _⟩


/-- Two bottoms met keep every number both allowed.

    {l₁ l₂ x : ℤ}                            [two bottom endpoints and a whole number]
    l₁ ≤ x                                   [the first allows it]
    l₂ ≤ x                                   [the second allows it]
    max l₁ l₂ ≤ x                            [so does the larger of the two]

    So: the lower channel of `inCell_meet_iff`, read off `Face.mem_meet_intervalOf` at a pair
    of intervals whose tops are the number itself. -/
theorem meet_lo_two_sided {l₁ l₂ x : ℤ} (h1 : l₁ ≤ x) (h2 : l₂ ≤ x) : max l₁ l₂ ≤ x :=
  ((mem_meet_intervalOf (σ := 1) (by norm_num) l₁ x l₂ x x).mpr
    ⟨⟨h1, le_refl x⟩, ⟨h2, le_refl x⟩⟩).1

/-- Two tops met keep every number both allowed.

    {h₁ h₂ x : ℤ}                            [two top endpoints and a whole number]
    x ≤ h₁                                   [the first allows it]
    x ≤ h₂                                   [the second allows it]
    x ≤ min h₁ h₂                            [so does the smaller of the two]

    So: the upper channel of the same theorem. -/
theorem meet_hi_two_sided {h₁ h₂ x : ℤ} (h1 : x ≤ h₁) (h2 : x ≤ h₂) : x ≤ min h₁ h₂ :=
  ((mem_meet_intervalOf (σ := 1) (by norm_num) x h₁ x h₂ x).mpr
    ⟨⟨le_refl x, h1⟩, ⟨le_refl x, h2⟩⟩).2

/-- Meeting two intervals keeps every whole number both held.

    {a c : Bnd} {x : ℤ}                      [two intervals and a whole number]
    BndHolds a x                             [the first holds it]
    BndHolds c x                             [the second holds it]
    BndHolds (Bnd.meet a c) x                [so does their meet]

    So: propagation loses no solution. Each channel goes through `meet_lo_two_sided` or
    `meet_hi_two_sided`, which are `inCell_meet_iff` transported to the whole numbers, so the
    step is an operation of the algebra rather than two comparisons. An absent endpoint has no
    point of `Cell σ` behind it and is discharged directly; the header names the primitive that
    would remove the split. -/
theorem bnd_meet_holds {a c : Bnd} {x : ℤ} (ha : BndHolds a x) (hc : BndHolds c x) :
    BndHolds (Bnd.meet a c) x := by
  obtain ⟨al, ah⟩ := a
  obtain ⟨cl, ch⟩ := c
  obtain ⟨hal, hah⟩ := ha
  obtain ⟨hcl, hch⟩ := hc
  simp only [BndHolds, Bnd.meet] at *
  refine ⟨?_, ?_⟩
  · cases al with
    | none => cases cl <;> simp_all [maxOpt]
    | some l₁ =>
        cases cl with
        | none => simpa [maxOpt] using hal
        | some l₂ =>
            simp only [maxOpt, propNewLo, Option.elim] at *
            exact meet_lo_two_sided hal hcl
  · cases ah with
    | none => cases ch <;> simp_all [minOpt]
    | some h₁ =>
        cases ch with
        | none => simpa [minOpt] using hah
        | some h₂ =>
            simp only [minOpt, propNewHi, Option.elim] at *
            exact meet_hi_two_sided hah hch

/-- An interval whose top has fallen below its bottom.

    (b : Bnd)                                [one column's interval]
    : Bool                                   [true exactly when the endpoints have passed]

    So: the UNSAT test. `trailCross` in ncone0 fires on this condition. -/
def Bnd.crossed (b : Bnd) : Bool :=
  match b.lo, b.hi with
  | some l, some h => decide (h < l)
  | _, _ => false

/-- A crossed interval is the algebra's crossing.

    (l h : ℤ)                                        [a bottom and a top endpoint]
    Bnd.crossed ⟨some l, some h⟩ = true              [the interval reads as crossed]
    ↔                                                [exactly when, true both directions]
    Crossing 1 (intervalOf 1 l h)                    [the point of the algebra is a crossing]

    So: `Bridge.crossing_ofBounds`. The verdict rides the order and not the null cone, which
    `crossing_independent_of_cone` shows is a different locus: `Fragment.norm_not_a_row` says
    the cone is not expressible as a row at all, and `Fragment.verdict_needs_no_norm` says the
    verdict never asks for it. -/
theorem crossed_iff_crossing (l h : ℤ) :
    Bnd.crossed ⟨some l, some h⟩ = true ↔ Crossing 1 (intervalOf 1 l h) := by
  rw [crossing_intervalOf (by norm_num : (0 : ℝ) < 1)]
  simp [Bnd.crossed]

/-- A crossed interval holds nothing.

    {b : Bnd} {x : ℤ}                        [an interval and a whole number]
    Bnd.crossed b = true                     [the interval is crossed]
    ¬ BndHolds b x                           [so it does not hold the number]

    So: `Face.crossing_empty`, where a crossing becomes a verdict. Every UNSAT this phase
    reports is this theorem applied once, at the column whose interval crossed. -/
theorem crossed_empty {b : Bnd} {x : ℤ} (h : Bnd.crossed b = true) : ¬ BndHolds b x := by
  intro hb
  obtain ⟨bl, bh⟩ := b
  cases bl with
  | none => simp [Bnd.crossed] at h
  | some l =>
      cases bh with
      | none => simp [Bnd.crossed] at h
      | some hh =>
          refine crossing_empty (σ := 1) (by norm_num) l hh x
            ((crossed_iff_crossing l hh).mp h) ?_
          rw [mem_intervalOf (by norm_num)]
          exact ⟨hb.1, hb.2⟩

/-- The endpoint one column contributes to another column's bound.

    (b : Fin n → Bnd)                        [one interval per column]
    (r : Row n)                              [a row over n columns]
    (k : Fin n)                              [which column contributes]
    : ℤ                                      [the endpoint its coefficient's sign reads]

    So: `propEnd` in ncone0, with the absent case defaulted; `rhsOf` below only reads it where
    the endpoint is present. -/
def endVal (b : Fin n → Bnd) (r : Row n) (k : Fin n) : ℤ := (propPick (r.coeff k) (b k)).getD 0

/-- The other side of a row, once every other column is replaced by an endpoint.

    (b : Fin n → Bnd)                        [one interval per column]
    (r : Row n)                              [a row over n columns]
    (j : Fin n)                              [the column a bound is wanted for]
    : Option ℤ                               [the bound on that column's term, if derivable]

    So: `propRhs`, which returns its sentinel when some other column has no usable endpoint. -/
def rhsOf (b : Fin n → Bnd) (r : Row n) (j : Fin n) : Option ℤ :=
  if ∀ k ∈ Finset.univ.erase j, (propPick (r.coeff k) (b k)).isSome = true then
    some (-r.const - ∑ k ∈ Finset.univ.erase j, r.coeff k * endVal b r k)
  else none

/-- One column's term is no larger with its endpoint than with its value.

    {b : Fin n → Bnd} {ρ : Fin n → ℤ}        [a box and an assignment inside it]
    (r : Row n) (k : Fin n)                  [a row and one of its columns]
    r.coeff k * ρ k ≤ r.coeff k * endVal b r k   [the term is at most the endpoint's term]
                                             [the term is bounded by the endpoint it reads]

    So: the arithmetic behind `propPick`. A positive coefficient reads the top and a negative
    one reads the bottom, and either way the product only goes up. -/
theorem term_le {b : Fin n → Bnd} {ρ : Fin n → ℤ} (hb : BoxHolds b ρ) (r : Row n) (k : Fin n)
    (h : (propPick (r.coeff k) (b k)).isSome = true) :
    r.coeff k * ρ k ≤ r.coeff k * endVal b r k := by
  rcases lt_trichotomy 0 (r.coeff k) with hpos | hzero | hneg
  · have hpick : propPick (r.coeff k) (b k) = (b k).hi := by simp [propPick, hpos]
    rw [hpick] at h
    obtain ⟨e, he⟩ := Option.isSome_iff_exists.mp h
    have hval : endVal b r k = e := by simp [endVal, hpick, he]
    have hle : ρ k ≤ e := by
      have := (hb k).2
      rw [he] at this
      exact this
    rw [hval]
    exact mul_le_mul_of_nonneg_left hle hpos.le
  · rw [← hzero]
    simp
  · have hnp : ¬ (0 : ℤ) < r.coeff k := by omega
    have hpick : propPick (r.coeff k) (b k) = (b k).lo := by simp [propPick, hnp, hneg]
    rw [hpick] at h
    obtain ⟨e, he⟩ := Option.isSome_iff_exists.mp h
    have hval : endVal b r k = e := by simp [endVal, hpick, he]
    have hle : e ≤ ρ k := by
      have := (hb k).1
      rw [he] at this
      exact this
    rw [hval]
    exact mul_le_mul_of_nonpos_left hle hneg.le

/-- The derived right-hand side is a genuine bound on the column's term.

    {b : Fin n → Bnd} {ρ : Fin n → ℤ}        [a box and an assignment]
    {r : Row n} {j : Fin n} {rhs : ℤ}        [a row, a column, the derived bound]
    BoxHolds b ρ                             [the assignment lies in the box]
    Row.Holds ρ r                            [and satisfies the row]
    rhsOf b r j = some rhs                   [and the row derives a bound for the column]
    rhs ≤ r.coeff j * ρ j                    [then the term meets it]

    So: the whole of phase one's soundness, before the division. Every other column's
    contribution is replaced by an endpoint that only makes the sum larger, so what is left for
    this column can only be smaller than the truth. -/
theorem rhsOf_le {b : Fin n → Bnd} {ρ : Fin n → ℤ} {r : Row n} {j : Fin n} {rhs : ℤ}
    (hb : BoxHolds b ρ) (hr : Row.Holds ρ r) (h : rhsOf b r j = some rhs) :
    rhs ≤ r.coeff j * ρ j := by
  rw [rhsOf] at h
  split at h
  · next hall =>
      rw [Option.some.injEq] at h
      subst h
      have hsplit : r.coeff j * ρ j + ∑ k ∈ Finset.univ.erase j, r.coeff k * ρ k
          = ∑ k, r.coeff k * ρ k :=
        Finset.add_sum_erase Finset.univ (fun k => r.coeff k * ρ k) (Finset.mem_univ j)
      have hmono : ∑ k ∈ Finset.univ.erase j, r.coeff k * ρ k
          ≤ ∑ k ∈ Finset.univ.erase j, r.coeff k * endVal b r k :=
        Finset.sum_le_sum fun k hk => term_le hb r k (hall k hk)
      rw [Row.Holds, Row.value] at hr
      omega
  · exact absurd h (by simp)

/-- Rounding a bound up, for a positive coefficient.

    (a p : ℤ)                                [the bound and the coefficient]
    : ℤ                                      [the smallest whole number the column may take]

    So: `ceildiv` in ncone0, written so the divisor stays positive. -/
def ceilDivPos (a p : ℤ) : ℤ := -((-a) / p)

/-- Rounding a bound down, for a negative coefficient.

    (a p : ℤ)                                [the bound and the coefficient]
    : ℤ                                      [the largest whole number the column may take]

    So: `floordiv` in ncone0, written so the divisor stays positive. -/
def floorDivNeg (a p : ℤ) : ℤ := (-a) / (-p)

/-- A positive coefficient's bound rounds up to a bottom endpoint.

    {a p x : ℤ}                              [a bound, a coefficient, a value]
    0 < p                                    [the coefficient is above zero]
    a ≤ p * x                                [the term meets the bound]
    ceilDivPos a p ≤ x                       [so the value is at or above the rounded bound] -/
theorem ceilDivPos_le {a p x : ℤ} (hp : 0 < p) (h : a ≤ p * x) : ceilDivPos a p ≤ x := by
  rw [ceilDivPos, neg_le, Int.le_ediv_iff_mul_le hp]
  have hc : -x * p = -(p * x) := by ring
  omega

/-- A negative coefficient's bound rounds down to a top endpoint.

    {a p x : ℤ}                              [a bound, a coefficient, a value]
    p < 0                                    [the coefficient is below zero]
    a ≤ p * x                                [the term meets the bound]
    x ≤ floorDivNeg a p                      [so the value is at or below the rounded bound] -/
theorem le_floorDivNeg {a p x : ℤ} (hp : p < 0) (h : a ≤ p * x) : x ≤ floorDivNeg a p := by
  rw [floorDivNeg, Int.le_ediv_iff_mul_le (by omega : (0 : ℤ) < -p)]
  have hc : x * -p = -(p * x) := by ring
  omega

/-- The derived bound, in the channel's own ring.

    {b : Fin n → Bnd} {ρ : Fin n → ℤ}        [a box and an assignment]
    {r : Row n} {j : Fin n} {rhs : ℤ}        [a row, a column, the derived bound]
    BoxHolds b ρ                             [the assignment lies in the box]
    Row.Holds ρ r                            [and satisfies the row]
    rhsOf b r j = some rhs                   [and the row derives a bound]
    ∃ w x y z : ℤ                            [there are four whole numbers]
    r.coeff j * ρ j - rhs = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2
                                             [whose squares add to the slack of that bound]

    So: `int_le_iff_exists_sum_four_sq` at the bound `rhsOf_le` derives. The comparison the
    propagator performs is a statement in the null channel's ring, using multiplication and
    addition alone, which is `cell.slang`'s C3 entry condition on the operation the propagator
    runs rather than on one described beside it. -/
theorem derived_bound_four_squares {b : Fin n → Bnd} {ρ : Fin n → ℤ} {r : Row n} {j : Fin n}
    {rhs : ℤ} (hb : BoxHolds b ρ) (hr : Row.Holds ρ r) (h : rhsOf b r j = some rhs) :
    ∃ w x y z : ℤ, r.coeff j * ρ j - rhs = w ^ 2 + x ^ 2 + y ^ 2 + z ^ 2 :=
  (int_le_iff_exists_sum_four_sq rhs (r.coeff j * ρ j)).mp (rhsOf_le hb hr h)

/-- The interval one row derives for one column, before it is met with what is held.

    (r : Row n) (j : Fin n) (rhs : ℤ)        [a row, a column, and the derived bound]
    : Bnd                                    [one column's interval, bounded on one side]

    So: the two arms of `propOne`, split out so the meet below reads as one operation. -/
def propBnd (r : Row n) (j : Fin n) (rhs : ℤ) : Bnd :=
  if 0 < r.coeff j then ⟨some (ceilDivPos rhs (r.coeff j)), none⟩
  else ⟨none, some (floorDivNeg rhs (r.coeff j))⟩

/-- One propagation step: one row deriving a bound for one column.

    (b : Fin n → Bnd)                        [one interval per column]
    (r : Row n)                              [a row over n columns]
    (j : Fin n)                              [the column to tighten]
    : Option (Fin n → Bnd)                   [the tightened box, or nothing if it crossed]

    So: `propOne` in ncone0. Absent means the column's interval crossed, which is the UNSAT
    report; present means the box, possibly tightened at that column and nowhere else. -/
def propOne (b : Fin n → Bnd) (r : Row n) (j : Fin n) : Option (Fin n → Bnd) :=
  if r.coeff j = 0 then some b
  else
    match rhsOf b r j with
    | none => some b
    | some rhs =>
        if ((b j).meet (propBnd r j rhs)).crossed then none
        else some (Function.update b j ((b j).meet (propBnd r j rhs)))

/-- One propagation step keeps every solution.

    {b : Fin n → Bnd} {ρ : Fin n → ℤ}        [a box and an assignment]
    {r : Row n}                              [a row over n columns]
    BoxHolds b ρ                             [the assignment lies in the box]
    Row.Holds ρ r                            [and satisfies the row]
    (j : Fin n)                              [any column]
    ∃ b', propOne b r j = some b'            [the step succeeds, giving some box]
    ∧ BoxHolds b' ρ                          [and the assignment survives into it]

    So: two statements in one. The step cannot report a crossing while a solution is in the
    box, and the box it returns still holds that solution. `bnd_meet_holds` supplies the second
    and `crossed_empty` the first. -/
theorem propOne_sound {b : Fin n → Bnd} {ρ : Fin n → ℤ} {r : Row n}
    (hb : BoxHolds b ρ) (hr : Row.Holds ρ r) (j : Fin n) :
    ∃ b', propOne b r j = some b' ∧ BoxHolds b' ρ := by
  by_cases hz : r.coeff j = 0
  · exact ⟨b, by simp [propOne, hz], hb⟩
  cases hrhs : rhsOf b r j with
  | none => exact ⟨b, by simp [propOne, hz, hrhs], hb⟩
  | some rhs =>
      have hle : rhs ≤ r.coeff j * ρ j := rhsOf_le hb hr hrhs
      have hnbh : BndHolds (propBnd r j rhs) (ρ j) := by
        by_cases hpos : 0 < r.coeff j
        · rw [propBnd, if_pos hpos]
          exact ⟨ceilDivPos_le hpos hle, trivial⟩
        · have hneg : r.coeff j < 0 := by omega
          rw [propBnd, if_neg hpos]
          exact ⟨trivial, le_floorDivNeg hneg hle⟩
      have hmh : BndHolds ((b j).meet (propBnd r j rhs)) (ρ j) := bnd_meet_holds (hb j) hnbh
      have hnc : ((b j).meet (propBnd r j rhs)).crossed = false := by
        by_contra hc
        exact crossed_empty (by simpa using hc) hmh
      refine ⟨Function.update b j ((b j).meet (propBnd r j rhs)), ?_, ?_⟩
      · simp only [propOne, if_neg hz, hrhs, hnc, Bool.false_eq_true, if_false]
      · intro i
        by_cases hij : i = j
        · subst hij
          simpa using hmh
        · simpa [Function.update_of_ne hij] using hb i

/-- One row propagated across a list of columns.

    (b : Fin n → Bnd)                        [one interval per column]
    (r : Row n)                              [a row over n columns]
    List (Fin n) → Option (Fin n → Bnd)      [the columns to walk, and the box that comes out]

    So: `propRowVars`, which stops at the first crossing. -/
def propRowCols (b : Fin n → Bnd) (r : Row n) : List (Fin n) → Option (Fin n → Bnd)
  | [] => some b
  | j :: js => (propOne b r j).bind (fun b' => propRowCols b' r js)

/-- Walking a row's columns keeps every solution.

    {ρ : Fin n → ℤ} {r : Row n}              [an assignment and a row]
    Row.Holds ρ r                            [the assignment satisfies the row]
    ∀ (js : List (Fin n))                    [for every list of columns]
    {b : Fin n → Bnd}                        [and every box]
    BoxHolds b ρ                             [holding the assignment]
    ∃ b', propRowCols b r js = some b'       [the walk succeeds, giving some box]
    ∧ BoxHolds b' ρ                          [and the assignment survives into it] -/
theorem propRowCols_sound {ρ : Fin n → ℤ} {r : Row n} (hr : Row.Holds ρ r) :
    ∀ (js : List (Fin n)) {b : Fin n → Bnd}, BoxHolds b ρ →
      ∃ b', propRowCols b r js = some b' ∧ BoxHolds b' ρ
  | [], b, hb => ⟨b, rfl, hb⟩
  | j :: js, b, hb => by
      obtain ⟨b1, h1, hb1⟩ := propOne_sound hb hr j
      obtain ⟨b2, h2, hb2⟩ := propRowCols_sound hr js hb1
      exact ⟨b2, by simp [propRowCols, h1, h2], hb2⟩

/-- One row propagated across every column.

    (b : Fin n → Bnd)                        [one interval per column]
    (r : Row n)                              [a row over n columns]
    : Option (Fin n → Bnd)                   [the box that comes out, or nothing] -/
def propRow (b : Fin n → Bnd) (r : Row n) : Option (Fin n → Bnd) :=
  propRowCols b r (List.finRange n)

/-- Every row of a system propagated in turn.

    (b : Fin n → Bnd)                        [one interval per column]
    List (Row n) → Option (Fin n → Bnd)      [the rows to walk, and the box that comes out]

    So: `propAllRows`, one pass over the bank. -/
def propAll (b : Fin n → Bnd) : List (Row n) → Option (Fin n → Bnd)
  | [] => some b
  | r :: rs => (propRow b r).bind (fun b' => propAll b' rs)

/-- One pass over the system keeps every solution.

    {ρ : Fin n → ℤ}                          [an assignment]
    ∀ (rs : List (Row n))                    [for every system of rows]
    SatRows ρ rs                             [the assignment satisfies every row]
    {b : Fin n → Bnd}                        [and for every box]
    BoxHolds b ρ                             [holding the assignment]
    ∃ b', propAll b rs = some b'             [the pass succeeds, giving some box]
    ∧ BoxHolds b' ρ                          [and the assignment survives into it] -/
theorem propAll_sound {ρ : Fin n → ℤ} :
    ∀ (rs : List (Row n)), SatRows ρ rs → ∀ {b : Fin n → Bnd}, BoxHolds b ρ →
      ∃ b', propAll b rs = some b' ∧ BoxHolds b' ρ
  | [], _, b, hb => ⟨b, rfl, hb⟩
  | r :: rs, hs, b, hb => by
      obtain ⟨b1, h1, hb1⟩ :=
        propRowCols_sound (hs r (by simp)) (List.finRange n) hb
      obtain ⟨b2, h2, hb2⟩ :=
        propAll_sound rs (fun s hsm => hs s (by simp [hsm])) hb1
      exact ⟨b2, by simp [propAll, propRow, h1, h2], hb2⟩

/-- Passes repeated until nothing moves, or the fuel runs out.

    ℕ → (Fin n → Bnd)                        [how many passes, and one interval per column]
    → List (Row n)                           [the rows of a system]
    → Option (Fin n → Bnd)                   [the box that settles, or nothing]

    So: `propagate` with its fuel of two hundred. Absent means some column's interval
    crossed. -/
def propFix : ℕ → (Fin n → Bnd) → List (Row n) → Option (Fin n → Bnd)
  | 0, b, _ => some b
  | fuel + 1, b, rs =>
      (propAll b rs).bind (fun b' => if b' = b then some b' else propFix fuel b' rs)

/-- Propagating to a fixed point keeps every solution.

    {ρ : Fin n → ℤ} (rs : List (Row n))      [an assignment and a system]
    SatRows ρ rs                             [the assignment satisfies every row]
    ∀ (fuel : ℕ) {b : Fin n → Bnd}           [for every fuel and every box]
    BoxHolds b ρ                             [holding the assignment]
    ∃ b', propFix fuel b rs = some b'        [the fixed point is reached]
    ∧ BoxHolds b' ρ                          [with the assignment still inside it] -/
theorem propFix_sound {ρ : Fin n → ℤ} (rs : List (Row n)) (hs : SatRows ρ rs) :
    ∀ (fuel : ℕ) {b : Fin n → Bnd}, BoxHolds b ρ →
      ∃ b', propFix fuel b rs = some b' ∧ BoxHolds b' ρ
  | 0, b, hb => ⟨b, rfl, hb⟩
  | fuel + 1, b, hb => by
      obtain ⟨b1, h1, hb1⟩ := propAll_sound rs hs hb
      by_cases heq : b1 = b
      · exact ⟨b1, by simp [propFix, h1, heq], hb1⟩
      · obtain ⟨b2, h2, hb2⟩ := propFix_sound rs hs fuel hb1
        exact ⟨b2, by simp [propFix, h1, heq, h2], hb2⟩

/-- A row with no columns left in it.

    (r : Row n)                              [a row over n columns]
    : Bool                                   [true when every coefficient is zero]

    So: `rowIsGroundLoop`. Propagation cannot tighten anything from such a row, so it is
    checked on its own. -/
def groundRow (r : Row n) : Bool := decide (∀ i, r.coeff i = 0)

/-- A system holding a ground row whose constant is below zero.

    (rs : List (Row n))                      [the rows of a system]
    : Bool                                   [true when one of them is ground and negative]

    So: `groundCheck`. -/
def groundBad (rs : List (Row n)) : Bool :=
  rs.any (fun r => groundRow r && decide (r.const < 0))

/-- A ground row below zero refutes the system outright.

    {rs : List (Row n)}                      [the rows of a system]
    groundBad rs = true                      [one row is ground with a negative constant]
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs          [no assignment satisfies the system]

    So: the cheapest refutation there is, and the one every certificate ends on after `t`
    grounds the accumulated combination. -/
theorem groundBad_sound {rs : List (Row n)} (h : groundBad rs = true) :
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs := by
  rintro ⟨ρ, hρ⟩
  rw [groundBad, List.any_eq_true] at h
  obtain ⟨r, hmem, hcond⟩ := h
  rw [Bool.and_eq_true, groundRow, decide_eq_true_eq, decide_eq_true_eq] at hcond
  have hv := hρ r hmem
  rw [Row.Holds, Row.value] at hv
  simp only [hcond.1, zero_mul, Finset.sum_const_zero, add_zero] at hv
  omega

/-- The verdict phase one reports.

    (fuel : ℕ)                               [how many passes to allow]
    (rs : List (Row n))                      [the rows of a system]
    : Bool                                   [true when the system is refuted]

    So: `propagate`'s own return, ground check and all. -/
def propUnsat (fuel : ℕ) (rs : List (Row n)) : Bool :=
  groundBad rs || (propFix fuel (fun _ => Bnd.top) rs).isNone

/-- Phase one never reports a refutation that is not one.

    {fuel : ℕ} {rs : List (Row n)}           [a fuel and a system]
    propUnsat fuel rs = true                 [propagation reports the system refuted]
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs          [then no assignment satisfies it]

    So: the headline of phase one. Every solution survives every step, by `propFix_sound`, so
    a box that crosses had no solution to lose. `../song/src/prover/cone/ncone0.slang` reports
    this as UNSAT and emits the trail that becomes the certificate. -/
theorem propUnsat_sound {fuel : ℕ} {rs : List (Row n)} (h : propUnsat fuel rs = true) :
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs := by
  rintro ⟨ρ, hρ⟩
  rw [propUnsat, Bool.or_eq_true] at h
  rcases h with hg | hn
  · exact groundBad_sound hg ⟨ρ, hρ⟩
  · obtain ⟨b', hb', -⟩ := propFix_sound rs hρ fuel (boxHolds_top ρ)
    rw [hb'] at hn
    simp at hn

end Decider
end Apeiron
