import Apeiron.Combinator.Linear.Substitution

/-!
# The three equations about the exponential

@name Exponential
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite curry-1930
@grade proved

`Combinator/Substitution.lean` closed the five wiring equations of a linear combinatory algebra
and `S` besides. Three remain, and they are the ones which speak about the exponential rather
than about the wiring: dereliction, which reads one copy out of a reusable route; digging,
which turns a reusable route into a reusable copy of a reusable copy; and functoriality, which
carries an application under the exponential. Abramsky, Haghverdi and Scott write them
`D !x = x`, `delta !x = !!x` and `F !x !y = !(x y)`. This file takes the first two;
`Combinator/Functorial.lean` takes the third. The middle one is named `digging` throughout
rather than by its Greek letter, which `Physics/Chaos.lean` already spends on Feigenbaum's
constant.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | `d_leave`, `digging_leave`, `entry_leave_d` |

Both routings are permutations of the counting numbers weighted by the unit, so the same routing
serves every setting and `theory/NOVELTY.md`'s A1 spectator test applies as it has since
`Combinator/Interaction.lean`. The parameter weights a path.

## What each one is

`bang z` runs `z` inside every one of the countably many copies Cantor's pairing supplies, so
`bang z (Nat.pair n u)` is `z u` with the tag `n` kept. Read that way the two equations here are
arithmetic on the tags.

**Dereliction** picks one copy. `dRoute` sends an external position `i` into the loop at
`Nat.pair 0 i`, which is copy zero, and sends what comes back out of copy zero to the external
position it names. Nothing else comes back, so the loop is empty and the trace is one pass. This
is what `Combinator/Duplication.lean`'s `capp_kRoute` recorded as owed: read through
`x · y := x (!y)` the discarding routing returns `bang x` where `x` was wanted, and `D` is
exactly the map which undoes that.

**Digging** re-indexes. `bang (bang x)` reads a position as a tag and a tagged position,
`bang x` reads it as a pair of tags and a position, and `dig` is the bijection between the two,
the associativity of Cantor's pairing. `diggingRoute` sends an external position through `dig`
into the loop and back through its inverse, so the trace is `bang x` conjugated by `dig`, and
`conj_dig_bang` says that conjugate is `bang (bang x)`.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- DERELICTION: READ ONE COPY
-- ═══════════════════════════════════════════════════════════════════

/-- Reading copy zero back gives the position it names.

    (j : ℕ)                                        [a position]
    uncopy 0 (Nat.pair 0 j) = some (j, one)        [copy zero is where the payload was put] -/
theorem uncopy_zero_pair (j : ℕ) : uncopy 0 (Nat.pair 0 j) = some (j, one) := by
  simp [uncopy, Nat.unpair_pair]

/-- The route that reads one copy of the exponential and hands back what is inside.

@cite abramsky-haghverdi-scott-2002

    Route                              [a weighted wire map]

    So: an external position goes into the loop at copy zero, and what comes back out of copy
    zero goes to the external position it names. Every other copy is dropped, which is what makes
    the loop empty. -/
def dRoute : Route := fun j =>
  Sum.elim
    (fun k => some (Equiv.natSumNatEquivNat (Sum.inr (Nat.pair 0 k)), one))
    (fun v => (uncopy 0 v).map fun q => (Equiv.natSumNatEquivNat (Sum.inl q.1), q.2))
    (Equiv.natSumNatEquivNat.symm j)

/-- The dereliction routing sends an external position into copy zero.

    (k : ℕ)                                                            [a position]
    dRoute (Equiv.natSumNatEquivNat (Sum.inl k))                       [the routing there]
    = some (Equiv.natSumNatEquivNat (Sum.inr (Nat.pair 0 k)), one)     [enters at copy zero] -/
theorem dRoute_inl (k : ℕ) :
    dRoute (Equiv.natSumNatEquivNat (Sum.inl k))
      = some (Equiv.natSumNatEquivNat (Sum.inr (Nat.pair 0 k)), one) := by
  simp [dRoute]

/-- The dereliction routing lets copy zero out and drops the rest.

    (v : ℕ)                                                            [a loop position]
    dRoute (Equiv.natSumNatEquivNat (Sum.inr v))                       [the routing there]
    = (uncopy 0 v).map into the even half          [uncopy lets copy zero out, stops elsewhere] -/
theorem dRoute_inr (v : ℕ) :
    dRoute (Equiv.natSumNatEquivNat (Sum.inr v))
      = (uncopy 0 v).map fun q => (Equiv.natSumNatEquivNat (Sum.inl q.1), q.2) := by
  simp [dRoute]

/-- Nothing crosses straight over.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    direct σ (plug σ dRoute (bang x)) i = none         [no straight crossing] -/
theorem d_direct (σ : ℝ) (x : Route) (i : ℕ) :
    direct σ (plug σ dRoute (bang x)) i = none := by
  rw [direct_apply]
  simp only [plug, rcomp, dRoute_inl, Option.bind_some, rtensor_inr]
  cases hb : bang x (Nat.pair 0 i) with
  | none => simp
  | some p => simp only [Option.map_some, Option.bind_some, outL_inr, Option.map_none]

/-- Entering the loop reads copy zero of the exponential.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    entry σ (plug σ dRoute (bang x)) i                 [the way in]
    = bang x (Nat.pair 0 i)                            [is the exponential at copy zero] -/
theorem d_entry (σ : ℝ) (x : Route) (i : ℕ) :
    entry σ (plug σ dRoute (bang x)) i = bang x (Nat.pair 0 i) := by
  rw [entry_apply]
  simp only [plug, rcomp, dRoute_inl, Option.bind_some, rtensor_inr]
  cases hb : bang x (Nat.pair 0 i) with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, outR_inr, one_mul, mul_one_right,
      Prod.mk.eta]

/-- The loop is never taken twice.

    (σ : ℝ) (x : Route) (u : ℕ)                        [the parameter, a route, a position]
    body σ (plug σ dRoute (bang x)) u = none           [one trip and no more] -/
theorem d_body (σ : ℝ) (x : Route) (u : ℕ) :
    body σ (plug σ dRoute (bang x)) u = none := by
  rw [body_apply]
  simp only [plug, rcomp, dRoute_inr]
  cases hu : uncopy 0 u with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, rtensor_inl, rid_apply, Option.map_some,
      outR_inl, Option.map_none]

/-- Leaving the loop strips the tag.

    (σ : ℝ) (x : Route) (u : ℕ)                        [the parameter, a route, a position]
    leave σ (plug σ dRoute (bang x)) u = uncopy 0 u    [the way out reads copy zero] -/
theorem d_leave (σ : ℝ) (x : Route) (u : ℕ) :
    leave σ (plug σ dRoute (bang x)) u = uncopy 0 u := by
  rw [leave_apply]
  simp only [plug, rcomp, dRoute_inr]
  cases hu : uncopy 0 u with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, rtensor_inl, rid_apply, Option.map_some,
      outL_inl, one_mul, mul_one_right, Prod.mk.eta]

/-- One pass through the loop is the route itself.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    rcomp σ (entry σ (plug σ dRoute (bang x)))         [in through copy zero]
      (leave σ (plug σ dRoute (bang x))) i             [and out again]
    = x i                                              [is the route, untagged] -/
theorem entry_leave_d (σ : ℝ) (x : Route) (i : ℕ) :
    rcomp σ (entry σ (plug σ dRoute (bang x))) (leave σ (plug σ dRoute (bang x))) i = x i := by
  rw [rcomp, d_entry, bang_tag]
  cases hx : x i with
  | none => simp
  | some q =>
    simp only [Option.map_some, Option.bind_some, d_leave, uncopy_zero_pair, Option.map_some,
      mul_one_right, Prod.mk.eta]

/-- **Dereliction.** Applying the reading routing to an exponential returns the route inside.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ (plug σ dRoute (bang x)) i j w        [a path of the applied routing]
    ↔                                              [exactly when]
    x i = some (j, w)                              [the route takes i there] -/
theorem dRoute_app (σ : ℝ) (x : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ dRoute (bang x)) i j w ↔ x i = some (j, w) := by
  have hbody : body σ (plug σ dRoute (bang x)) = rnone := funext (d_body σ x)
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · rw [d_direct] at hd; exact absurd hd (by simp)
    · cases n with
      | zero =>
        rw [rpow, rcomp_rid_left] at hn
        rw [entry_leave_d] at hn
        exact hn
      | succ n =>
        rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
        simp [rnone] at hn
  · intro h
    refine Or.inr ⟨0, ?_⟩
    rw [rpow, rcomp_rid_left, entry_leave_d]
    exact h

/-- **D !x = x**, as an equation between routes.

@cite abramsky-haghverdi-scott-2002
@cite girard-1989

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    rapp σ dRoute (bang x) = x                     [reading one copy returns the route]

    So: the sixth of the eight equations. `Combinator/Duplication.lean`'s `capp_kRoute` said the
    translation from the linear algebra to the ordinary one owed exactly this map, and this is
    it, with the loop empty because every copy but one is dropped. -/
theorem rapp_dRoute (σ : ℝ) (x : Route) : rapp σ dRoute (bang x) = x :=
  rapp_eq (dRoute_app σ x)

/-- Dereliction stated against application with the exponential.

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    capp σ dRoute x = x                            [the ordinary application of the reader] -/
theorem capp_dRoute (σ : ℝ) (x : Route) : capp σ dRoute x = x := rapp_dRoute σ x

-- ═══════════════════════════════════════════════════════════════════
-- DIGGING: RE-INDEX ALONG THE PAIRING OF PAIRINGS
-- ═══════════════════════════════════════════════════════════════════

/-- Read a tag and a tagged position as a pair of tags and a position.

    (i : ℕ)                                        [a position of the double exponential]
    ℕ                                              [its place in the single one] -/
def digFun (i : ℕ) : ℕ :=
  Nat.pair (Nat.pair (Nat.unpair i).1 (Nat.unpair (Nat.unpair i).2).1)
    (Nat.unpair (Nat.unpair i).2).2

/-- Read a pair of tags and a position as a tag and a tagged position.

    (i : ℕ)                                        [a position of the single exponential]
    ℕ                                              [its place in the double one] -/
def digInv (i : ℕ) : ℕ :=
  Nat.pair (Nat.unpair (Nat.unpair i).1).1
    (Nat.pair (Nat.unpair (Nat.unpair i).1).2 (Nat.unpair i).2)

/-- The re-indexing on a nested pair.

    (s t u : ℕ)                                    [a tag, a tag and a position]
    digFun (Nat.pair s (Nat.pair t u))             [read outside in]
    = Nat.pair (Nat.pair s t) u                    [is read inside out] -/
theorem digFun_pair (s t u : ℕ) :
    digFun (Nat.pair s (Nat.pair t u)) = Nat.pair (Nat.pair s t) u := by
  simp [digFun, Nat.unpair_pair]

/-- The inverse re-indexing on a nested pair.

    (s t u : ℕ)                                    [two tags and a position]
    digInv (Nat.pair (Nat.pair s t) u)             [read inside out]
    = Nat.pair s (Nat.pair t u)                    [is read outside in] -/
theorem digInv_pair (s t u : ℕ) :
    digInv (Nat.pair (Nat.pair s t) u) = Nat.pair s (Nat.pair t u) := by
  simp [digInv, Nat.unpair_pair]

/-- The re-indexing is a bijection.

    ℕ ≃ ℕ                                          [the associativity of Cantor's pairing] -/
def dig : ℕ ≃ ℕ where
  toFun := digFun
  invFun := digInv
  left_inv := by
    intro i
    simp [digFun, digInv, Nat.unpair_pair, Nat.pair_unpair]
  right_inv := by
    intro i
    simp [digFun, digInv, Nat.unpair_pair, Nat.pair_unpair]

/-- The bijection is the map.

    (i : ℕ)                                        [a position]
    dig i = digFun i                               [the same re-indexing] -/
theorem dig_apply (i : ℕ) : dig i = digFun i := rfl

/-- The inverse bijection is the inverse map.

    (i : ℕ)                                        [a position]
    dig.symm i = digInv i                          [the same re-indexing back] -/
theorem dig_symm_apply (i : ℕ) : dig.symm i = digInv i := rfl

/-- The route that re-indexes the exponential into the double one.

@cite abramsky-haghverdi-scott-2002

    Route                              [a weighted wire map]

    So: an external position goes into the loop re-indexed, and what comes back out is
    re-indexed the other way. The loop is empty because the routing sends every loop position
    straight out. -/
def diggingRoute : Route := fun j =>
  Sum.elim
    (fun k => some (Equiv.natSumNatEquivNat (Sum.inr (digFun k)), one))
    (fun v => some (Equiv.natSumNatEquivNat (Sum.inl (digInv v)), one))
    (Equiv.natSumNatEquivNat.symm j)

/-- The digging routing sends an external position into the loop, re-indexed.

    (k : ℕ)                                                            [a position]
    diggingRoute (Equiv.natSumNatEquivNat (Sum.inl k))                   [the routing there]
    = some (Equiv.natSumNatEquivNat (Sum.inr (digFun k)), one)         [enters re-indexed] -/
theorem diggingRoute_inl (k : ℕ) :
    diggingRoute (Equiv.natSumNatEquivNat (Sum.inl k))
      = some (Equiv.natSumNatEquivNat (Sum.inr (digFun k)), one) := by
  simp [diggingRoute]

/-- The digging routing sends a loop position out, re-indexed back.

    (v : ℕ)                                                            [a loop position]
    diggingRoute (Equiv.natSumNatEquivNat (Sum.inr v))                   [the routing there]
    = some (Equiv.natSumNatEquivNat (Sum.inl (digInv v)), one)         [leaves re-indexed] -/
theorem diggingRoute_inr (v : ℕ) :
    diggingRoute (Equiv.natSumNatEquivNat (Sum.inr v))
      = some (Equiv.natSumNatEquivNat (Sum.inl (digInv v)), one) := by
  simp [diggingRoute]

/-- Nothing crosses straight over.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    direct σ (plug σ diggingRoute (bang x)) i = none     [no straight crossing] -/
theorem digging_direct (σ : ℝ) (x : Route) (i : ℕ) :
    direct σ (plug σ diggingRoute (bang x)) i = none := by
  rw [direct_apply]
  simp only [plug, rcomp, diggingRoute_inl, Option.bind_some, rtensor_inr]
  cases hb : bang x (digFun i) with
  | none => simp
  | some p => simp only [Option.map_some, Option.bind_some, outL_inr, Option.map_none]

/-- Entering the loop reads the exponential at the re-indexed position.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    entry σ (plug σ diggingRoute (bang x)) i             [the way in]
    = bang x (digFun i)                                [is the exponential, re-indexed] -/
theorem digging_entry (σ : ℝ) (x : Route) (i : ℕ) :
    entry σ (plug σ diggingRoute (bang x)) i = bang x (digFun i) := by
  rw [entry_apply]
  simp only [plug, rcomp, diggingRoute_inl, Option.bind_some, rtensor_inr]
  cases hb : bang x (digFun i) with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, outR_inr, one_mul, mul_one_right,
      Prod.mk.eta]

/-- The loop is never taken twice.

    (σ : ℝ) (x : Route) (u : ℕ)                        [the parameter, a route, a position]
    body σ (plug σ diggingRoute (bang x)) u = none       [one trip and no more] -/
theorem digging_body (σ : ℝ) (x : Route) (u : ℕ) :
    body σ (plug σ diggingRoute (bang x)) u = none := by
  rw [body_apply]
  simp only [plug, rcomp, diggingRoute_inr, Option.bind_some, rtensor_inl, rid_apply,
    Option.map_some, outR_inl, Option.map_none]

/-- Leaving the loop re-indexes back.

    (σ : ℝ) (x : Route) (u : ℕ)                        [the parameter, a route, a position]
    leave σ (plug σ diggingRoute (bang x)) u             [the way out]
    = some (digInv u, one)                             [re-indexes the other way] -/
theorem digging_leave (σ : ℝ) (x : Route) (u : ℕ) :
    leave σ (plug σ diggingRoute (bang x)) u = some (digInv u, one) := by
  rw [leave_apply]
  simp only [plug, rcomp, diggingRoute_inr, Option.bind_some, rtensor_inl, rid_apply,
    Option.map_some, outL_inl, one_mul]

/-- **The exponential re-indexed is the double exponential.**

@cite abramsky-haghverdi-scott-2002

    (x : Route)                                    [a route]
    conjRoute dig (bang x) = bang (bang x)         [reading the tags either way agrees]

    So: this is the whole arithmetic content of digging. A position of `!!x` is a tag and a
    tagged position, a position of `!x` is a pair of tags and a position, and the associativity
    of Cantor's pairing carries one to the other. -/
theorem conj_dig_bang (x : Route) : conjRoute dig (bang x) = bang (bang x) := by
  funext i
  obtain ⟨s, m, rfl⟩ : ∃ s m, i = Nat.pair s m := ⟨_, _, (Nat.pair_unpair i).symm⟩
  obtain ⟨t, u, rfl⟩ : ∃ t u, m = Nat.pair t u := ⟨_, _, (Nat.pair_unpair m).symm⟩
  rw [conjRoute, dig_apply, digFun_pair, bang_tag, bang_tag, bang_tag]
  cases hx : x u with
  | none => rfl
  | some q =>
    simp only [Option.map_some, dig_symm_apply, digInv_pair]

/-- One pass through the loop is the exponential, re-indexed.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    rcomp σ (entry σ (plug σ diggingRoute (bang x)))     [in re-indexed]
      (leave σ (plug σ diggingRoute (bang x))) i         [and out again]
    = bang (bang x) i                                  [is the double exponential] -/
theorem entry_leave_digging (σ : ℝ) (x : Route) (i : ℕ) :
    rcomp σ (entry σ (plug σ diggingRoute (bang x))) (leave σ (plug σ diggingRoute (bang x))) i
      = bang (bang x) i := by
  rw [rcomp, digging_entry, ← conj_dig_bang, conjRoute, dig_apply]
  cases hb : bang x (digFun i) with
  | none => simp
  | some p =>
    simp only [Option.bind_some, digging_leave, Option.map_some, mul_one_right, dig_symm_apply]

/-- Applying the re-indexing routing to an exponential gives the double one.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ (plug σ diggingRoute (bang x)) i j w    [a path of the applied routing]
    ↔                                              [exactly when]
    bang (bang x) i = some (j, w)                  [the double exponential takes i there] -/
theorem diggingRoute_app (σ : ℝ) (x : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ diggingRoute (bang x)) i j w ↔ bang (bang x) i = some (j, w) := by
  have hbody : body σ (plug σ diggingRoute (bang x)) = rnone := funext (digging_body σ x)
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · rw [digging_direct] at hd; exact absurd hd (by simp)
    · cases n with
      | zero =>
        rw [rpow, rcomp_rid_left] at hn
        rw [entry_leave_digging] at hn
        exact hn
      | succ n =>
        rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
        simp [rnone] at hn
  · intro h
    refine Or.inr ⟨0, ?_⟩
    rw [rpow, rcomp_rid_left, entry_leave_digging]
    exact h

/-- **Digging**, `bang x` turned into `bang (bang x)`, as an equation between routes.

@cite abramsky-haghverdi-scott-2002
@cite girard-1989

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    rapp σ diggingRoute (bang x) = bang (bang x)     [re-indexing gives the double exponential]

    So: the seventh of the eight equations, and it is a relabelling and no more. The two ways of
    reading a position, as a tag on a tagged position or as a pair of tags on a position, are
    carried to each other by the associativity of Cantor's pairing, and the routing is that
    bijection sent into the loop and back. -/
theorem rapp_diggingRoute (σ : ℝ) (x : Route) : rapp σ diggingRoute (bang x) = bang (bang x) :=
  rapp_eq (diggingRoute_app σ x)

end Combinator
end Apeiron

end
