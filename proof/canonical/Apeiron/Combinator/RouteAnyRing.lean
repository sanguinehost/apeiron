import Apeiron.Trichotomy.AnyRing
import Mathlib.Data.Nat.Pairing
import Mathlib.Logic.Equiv.Nat

/-!
# The weighted route core over any commutative ring

@name Weighted Routes Over Any Commutative Ring
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite haghverdi-scott-2010
@cite joyal-street-verity-1996
@grade proved

`Combinator/Interaction.lean` builds weighted wire maps over the reals: a partial injection on
the counting numbers with a point of the two-slot algebra on every wire. The wiring is free of
the parameter, and the only place the weights are read is the multiplication along a path. This
file installs that construction once over an arbitrary commutative ring, so the real routes and
their evaluators are specialisations rather than restatements.

Everything here is `RingCell R` and `ringMul σ`, the generic point and product of
`Trichotomy/AnyRing.lean`. The two-track retraction is `Equiv.natSumNatEquivNat`, the exponential
is Cantor's pairing, and neither mentions the ring.

The **relational** semantics is `Traces`, the union over every path length, and `Runs`, a path
staying inside a set until it leaves. The **executable** semantics is the fuel-bounded
`traceWithin` and `runsWithin`: the budget is an explicit input, a value found inside it is
`.found`, and running out is `.exhausted`. Neither bounded evaluator uses `Classical.choice`:
`traceWithin` matches on `Option`, and `runsWithin` decides loop membership through an explicit
`DecidablePred`. An exhausted budget is not a proof of divergence, and this file makes no such
claim. The relational union stays in place, and `traceWithin_iff_traces` and
`runsWithin_iff_Runs` say the bounded evaluator and the relation agree wherever the evaluator
answers.

## Root

| from | here |
|---|---|
| `AnyRing.RingCell` | `Route` |
| `AnyRing.ringMul` | `rcomp`, `traceWithin`, `runsWithin` |
| `AnyRing.ringOne` | `rid`, `inL`, `inR` |
| `AnyRing.ringMul_assoc` | `rcomp_assoc` |

`Route R` is a partial injection on the counting numbers with one `RingCell R` weight on every
wire, `rcomp` multiplies the weights along a path by `ringMul`, and `rid` weights every wire by
`ringOne`. The wiring is free of the parameter: the two-track retraction is
`Equiv.natSumNatEquivNat` and the exponential is Cantor's pairing. `traces_functional` makes the
relational trace single-valued, and `traceWithin_iff_traces` and `runsWithin_iff_Runs` say the
fuel-bounded evaluator agrees with it. What does not transfer is the sign classification of the
loop weight, `Interaction.weight_blocks_the_loop_iff_disc_nonneg`, which reads `ringNormForm` and
the order of `ℝ`. Over a general ring there is no such classification, and an exhausted budget is
not a proof of divergence.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Combinator
namespace RouteAnyRing

open Apeiron.Trichotomy.AnyRing

universe u

variable {R : Type u} [CommRing R]

-- ═══════════════════════════════════════════════════════════════════
-- THE WEIGHTED WIRE, AND THE MONOID IT CARRIES
-- ═══════════════════════════════════════════════════════════════════

/-- A bounded search result: a value found inside the budget, or the budget exhausted.

    Outcome α        [one bounded search result]
    found (a : α)    [a value was found before the budget ran out]
    exhausted        [the budget ran out with no value found] -/
inductive Outcome (α : Type u) : Type u where
  | found (a : α) : Outcome α
  | exhausted : Outcome α

/-- A weighted wire map: at most one target for each position, carrying a point as its weight.

    Route R                      [a weighted wire map over the ring R]
    ℕ                            [the counting numbers 0, 1, 2, 3, and so on]
    →                            [takes the left and gives the right]
    Option (ℕ × RingCell R)      [each position goes nowhere, or to one with one weight]
    ×                            [paired with, so the position and the weight sit side by side] -/
abbrev Route (R : Type u) [CommRing R] : Type u := ℕ → Option (ℕ × RingCell R)

/-- The route that connects nothing.

    Route R    [a weighted wire map over the ring R]
    rnone      [the route that joins nothing] -/
abbrev rnone : Route R := fun _ => none

/-- The route that connects every position to itself, with the unit weight.

    Route R    [a weighted wire map over the ring R]
    rid        [the route joining every position to itself, weighted by the unit] -/
abbrev rid : Route R := fun i => some (i, ringOne)

/-- Follow one route and then another, multiplying the weights.

    (σ : R)              [the parameter: what the adjoined direction squares to]
    (a b : Route R)      [two weighted wire maps]
    rcomp                [follow one route and then another, multiplying the weights]
    Route R              [the composite, first a and then b] -/
def rcomp (σ : R) (a b : Route R) : Route R :=
  fun i => (a i).bind fun p => (b p.1).map fun q => (q.1, ringMul σ p.2 q.2)

/-- The identity route is a left unit for composing.

    (σ : R) (a : Route R)            [the parameter and a route]
    rcomp                            [follow one route and then another, multiplying the weights]
    rid                              [the route joining every position to itself]
    rcomp σ rid a = a                [going nowhere first changes nothing] -/
theorem rcomp_rid_left (σ : R) (a : Route R) : rcomp σ rid a = a := by
  funext i
  cases h : a i with
  | none => simp [rcomp, rid, h]
  | some p => simp [rcomp, rid, h]

/-- The identity route is a right unit for composing.

    (σ : R) (a : Route R)            [the parameter and a route]
    rcomp                            [follow one route and then another, multiplying the weights]
    rid                              [the route joining every position to itself]
    rcomp σ a rid = a                [going nowhere after changes nothing] -/
theorem rcomp_rid_right (σ : R) (a : Route R) : rcomp σ a rid = a := by
  funext i
  cases h : a i with
  | none => simp [rcomp, rid, h]
  | some p => simp [rcomp, rid, h]

/-- Composing routes does not depend on the bracketing.

    (σ : R) (a b c : Route R)                          [the parameter and three routes]
    rcomp σ (rcomp σ a b) c = rcomp σ a (rcomp σ b c)  [either bracketing gives the same]
    rcomp                                              [follow one route and then another] -/
theorem rcomp_assoc (σ : R) (a b c : Route R) :
    rcomp σ (rcomp σ a b) c = rcomp σ a (rcomp σ b c) := by
  funext i
  cases h : a i with
  | none => simp [rcomp, h]
  | some p =>
    cases h2 : b p.1 with
    | none => simp [rcomp, h, h2]
    | some q =>
      cases h3 : c q.1 with
      | none => simp [rcomp, h, h2, h3]
      | some r => simp [rcomp, h, h2, h3]; constructor <;> ring

/-- The empty route absorbs on the left.

    (σ : R) (a : Route R)            [the parameter and a route]
    rnone                            [the route that joins nothing]
    rcomp σ rnone a = rnone          [nothing in, nothing out] -/
theorem rcomp_rnone_left (σ : R) (a : Route R) : rcomp σ rnone a = rnone := by
  funext i
  simp [rcomp, rnone]

/-- The empty route absorbs on the right.

    (σ : R) (a : Route R)            [the parameter and a route]
    rnone                            [the route that joins nothing]
    rcomp σ a rnone = rnone          [nothing out, nothing in] -/
theorem rcomp_rnone_right (σ : R) (a : Route R) : rcomp σ a rnone = rnone := by
  funext i
  cases h : a i with
  | none => simp [rcomp, rnone, h]
  | some p => simp [rcomp, rnone, h]

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO WIRES INSIDE ONE WIRE
-- ═══════════════════════════════════════════════════════════════════

/-- Place a wire in the even positions.

    Route R    [a weighted wire map over the ring R]
    inL        [place a wire in the even positions] -/
abbrev inL : Route R := fun i => some (Equiv.natSumNatEquivNat (Sum.inl i), ringOne)

/-- Place a wire in the odd positions.

    Route R    [a weighted wire map over the ring R]
    inR        [place a wire in the odd positions] -/
abbrev inR : Route R := fun i => some (Equiv.natSumNatEquivNat (Sum.inr i), ringOne)

/-- Read the even positions back.

    Route R    [a weighted wire map over the ring R]
    outL       [read the even positions back] -/
abbrev outL : Route R := fun j =>
  Sum.elim (fun k => some (k, ringOne)) (fun _ => none) (Equiv.natSumNatEquivNat.symm j)

/-- Read the odd positions back.

    Route R    [a weighted wire map over the ring R]
    outR       [read the odd positions back] -/
abbrev outR : Route R := fun j =>
  Sum.elim (fun _ => none) (fun k => some (k, ringOne)) (Equiv.natSumNatEquivNat.symm j)

-- The six nullary wires are abbreviations, so a `simp [outL]` at the real carrier would stop at
-- `RouteAnyRing.outL`. Tagging them puts their bodies in the default simp set, so the reduction
-- reaches the raw `Sum.elim` the original definitions exposed.
attribute [simp] rnone rid inL inR outL outR

/-- Placing in the even positions and reading them back is the identity.

    (σ : R)                            [the parameter: what the adjoined direction squares to]
    rcomp                              [follow one route and then another]
    inL                                [place a wire in the even positions]
    outL                               [read the even positions back]
    rid                                [the route joining every position to itself]
    rcomp σ inL outL = rid             [the round trip changes nothing] -/
theorem outL_inL (σ : R) : rcomp σ (inL (R := R)) (outL (R := R)) = rid (R := R) := by
  funext i
  simp [rcomp, inL, outL, rid]

/-- Placing in the odd positions and reading them back is the identity.

    (σ : R)                            [the parameter: what the adjoined direction squares to]
    rcomp                              [follow one route and then another]
    inR                                [place a wire in the odd positions]
    outR                               [read the odd positions back]
    rid                                [the route joining every position to itself]
    rcomp σ inR outR = rid             [the round trip changes nothing] -/
theorem outR_inR (σ : R) : rcomp σ (inR (R := R)) (outR (R := R)) = rid (R := R) := by
  funext i
  simp [rcomp, inR, outR, rid]

/-- The even positions carry nothing that was placed in the odd ones.

    (σ : R)                            [the parameter: what the adjoined direction squares to]
    rcomp                              [follow one route and then another]
    inR                                [place a wire in the odd positions]
    outL                               [read the even positions back]
    rnone                              [the route that joins nothing]
    rcomp σ inR outL = rnone           [no wire crosses] -/
theorem outL_inR (σ : R) : rcomp σ (inR (R := R)) (outL (R := R)) = rnone (R := R) := by
  funext i
  simp [rcomp, inR, outL, rnone]

/-- The odd positions carry nothing that was placed in the even ones.

    (σ : R)                            [the parameter: what the adjoined direction squares to]
    rcomp                              [follow one route and then another]
    inL                                [place a wire in the even positions]
    outR                               [read the odd positions back]
    rnone                              [the route that joins nothing]
    rcomp σ inL outR = rnone           [no wire crosses] -/
theorem outR_inL (σ : R) : rcomp σ (inL (R := R)) (outR (R := R)) = rnone (R := R) := by
  funext i
  simp [rcomp, inL, outR, rnone]

/-- Two routes side by side, the first on the even positions and the second on the odd ones.

    (a b : Route R)      [two weighted wire maps]
    rtensor              [two routes side by side, one on the even positions and one on the odd]
    Route R              [the two of them, running in parallel] -/
def rtensor (a b : Route R) : Route R := fun j =>
  Sum.elim (fun k => (a k).map fun q => (Equiv.natSumNatEquivNat (Sum.inl q.1), q.2))
    (fun k => (b k).map fun q => (Equiv.natSumNatEquivNat (Sum.inr q.1), q.2))
    (Equiv.natSumNatEquivNat.symm j)

-- ═══════════════════════════════════════════════════════════════════
-- GOING ROUND THE LOOP
-- ═══════════════════════════════════════════════════════════════════

/-- Going round a route n times.

    (σ : R) (a : Route R)            [the parameter and a route]
    rpow                             [a route composed with itself that many times]
    ℕ → Route R                      [a counting number to that many trips round] -/
def rpow (σ : R) (a : Route R) : ℕ → Route R
  | 0 => rid (R := R)
  | n + 1 => rcomp σ a (rpow σ a n)

/-- A wiring that runs out stays run out.

    (σ : R) (a : Route R) {n : ℕ}                    [the parameter, a route, a depth]
    (h : rpow σ a n = rnone)                         [n trips connect nothing]
    rpow σ a (n + 1) = rnone                         [neither does one more]
    rpow                                             [a route composed with itself]
    rnone                                            [the route that joins nothing] -/
theorem rpow_rnone_succ (σ : R) (a : Route R) {n : ℕ} (h : rpow σ a n = rnone (R := R)) :
    rpow σ a (n + 1) = rnone (R := R) := by
  rw [rpow, h, rcomp_rnone_right]

/-- Going round n plus m times is going round n times and then m times.

    (σ : R) (a : Route R) (n m : ℕ)                            [σ, a route, two counts]
    rpow σ a (n + m) = rcomp σ (rpow σ a n) (rpow σ a m)       [the trips split]
    rpow                                                       [a route composed with itself] -/
theorem rpow_split (σ : R) (a : Route R) (n m : ℕ) :
    rpow σ a (n + m) = rcomp σ (rpow σ a n) (rpow σ a m) := by
  induction n with
  | zero => rw [Nat.zero_add, rpow, rcomp_rid_left]
  | succ n ih =>
    rw [Nat.succ_add, rpow, ih, rpow, rcomp_assoc]

/-- One trip of a route that connects nothing connects nothing.

    (σ : R) (a : Route R) (n i : ℕ)                            [σ, a route, a count, a start]
    (h : a i = none)                                           [the route stops here]
    rpow σ a (n + 1) i = none                                  [so no trip from here lands]
    rpow                                                       [a route composed with itself] -/
theorem rpow_succ_none (σ : R) (a : Route R) (n i : ℕ) (h : a i = none) :
    rpow σ a (n + 1) i = none := by
  simp [rpow, rcomp, h]

-- ═══════════════════════════════════════════════════════════════════
-- THE TRACE: ONE PATH, WHATEVER ITS LENGTH
-- ═══════════════════════════════════════════════════════════════════

/-- The part of a route that goes straight across, from the even wire to the even wire.

    (σ : R) (f : Route R)            [the parameter and a route on the doubled wire]
    direct                           [what the route does without entering the loop]
    Route R                          [the straight crossing] -/
def direct (σ : R) (f : Route R) : Route R :=
  rcomp σ (inL (R := R)) (rcomp σ f (outL (R := R)))

/-- The part of a route that enters the loop.

    (σ : R) (f : Route R)            [the parameter and a route on the doubled wire]
    entry                            [the way from the even wire into the odd one]
    Route R                          [the way in] -/
def entry (σ : R) (f : Route R) : Route R :=
  rcomp σ (inL (R := R)) (rcomp σ f (outR (R := R)))

/-- The part of a route that leaves the loop.

    (σ : R) (f : Route R)            [the parameter and a route on the doubled wire]
    leave                            [the way from the odd wire out to the even one]
    Route R                          [the way out] -/
def leave (σ : R) (f : Route R) : Route R :=
  rcomp σ (inR (R := R)) (rcomp σ f (outL (R := R)))

/-- The part of a route that goes once round the loop.

    (σ : R) (f : Route R)            [the parameter and a route on the doubled wire]
    body                             [the way from the odd wire back to the odd wire]
    Route R                          [one trip round] -/
def body (σ : R) (f : Route R) : Route R :=
  rcomp σ (inR (R := R)) (rcomp σ f (outR (R := R)))

/-- What one trip round the loop does, written out.

    (σ : R) (f : Route R) (u : ℕ)                    [the parameter, a route, a loop position]
    body σ f u                                       [one trip round]
    outR                                             [kept only if the read lands odd]
    ringMul σ ringOne                                [with the weights multiplied]
    rcomp                                            [follow one route and then another] -/
theorem body_apply (σ : R) (f : Route R) (u : ℕ) :
    body σ f u = ((f (Equiv.natSumNatEquivNat (Sum.inr u))).bind fun p =>
      (outR (R := R) p.1).map fun q => (q.1, ringMul σ p.2 q.2)).map
        fun q => (q.1, ringMul σ ringOne q.2) := rfl

/-- What leaving the loop does, written out.

    (σ : R) (f : Route R) (u : ℕ)                    [the parameter, a route, a loop position]
    leave σ f u                                      [the way out]
    outL                                             [kept only if the read lands even]
    ringMul σ ringOne                                [with the weights multiplied]
    rcomp                                            [follow one route and then another] -/
theorem leave_apply (σ : R) (f : Route R) (u : ℕ) :
    leave σ f u = ((f (Equiv.natSumNatEquivNat (Sum.inr u))).bind fun p =>
      (outL (R := R) p.1).map fun q => (q.1, ringMul σ p.2 q.2)).map
        fun q => (q.1, ringMul σ ringOne q.2) := rfl

/-- What crossing straight over does, written out.

    (σ : R) (f : Route R) (i : ℕ)                    [the parameter, a route, a position]
    direct σ f i                                     [the straight crossing]
    outL                                             [kept only if the read lands even]
    ringMul σ ringOne                                [with the weights multiplied]
    rcomp                                            [follow one route and then another] -/
theorem direct_apply (σ : R) (f : Route R) (i : ℕ) :
    direct σ f i = ((f (Equiv.natSumNatEquivNat (Sum.inl i))).bind fun p =>
      (outL (R := R) p.1).map fun q => (q.1, ringMul σ p.2 q.2)).map
        fun q => (q.1, ringMul σ ringOne q.2) := rfl

/-- What entering the loop does, written out.

    (σ : R) (f : Route R) (i : ℕ)                    [the parameter, a route, a position]
    entry σ f i                                      [the way in]
    outR                                             [kept only if the read lands odd]
    ringMul σ ringOne                                [with the weights multiplied]
    rcomp                                            [follow one route and then another] -/
theorem entry_apply (σ : R) (f : Route R) (i : ℕ) :
    entry σ f i = ((f (Equiv.natSumNatEquivNat (Sum.inl i))).bind fun p =>
      (outR (R := R) p.1).map fun q => (q.1, ringMul σ p.2 q.2)).map
        fun q => (q.1, ringMul σ ringOne q.2) := rfl

/-- A position lands in one half of the wire, so one of the two reads gives nothing.

    (j : ℕ)                              [a position of the doubled wire]
    outL                                 [read the even positions back]
    outR                                 [read the odd positions back]
    outL j = none ∨ outR j = none        [it is odd, or it is even] -/
theorem outL_outR_disjoint (j : ℕ) :
    outL (R := R) j = none ∨ outR (R := R) j = none := by
  cases hs : Equiv.natSumNatEquivNat.symm j with
  | inl k => right; simp only [outR, hs, Sum.elim_inl]
  | inr k => left; simp only [outL, hs, Sum.elim_inr]

/-- A position in the loop either goes round again or leaves, never both.

    (σ : R) (f : Route R) (u : ℕ)                      [the parameter, a route, a loop position]
    body σ f u = none ∨ leave σ f u = none              [at most one of the two is defined]
    body                                                [one trip round the loop]
    leave                                               [the way out] -/
theorem body_leave_disjoint (σ : R) (f : Route R) (u : ℕ) :
    body σ f u = none ∨ leave σ f u = none := by
  rw [body_apply, leave_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inr u)) with
  | none => left; simp
  | some p =>
    rcases outL_outR_disjoint (R := R) p.1 with h | h
    · right; simp only [Option.bind_some, h, Option.map_none]
    · left; simp only [Option.bind_some, h, Option.map_none]

/-- A position on the way in either crosses straight over or enters the loop, never both.

    (σ : R) (f : Route R) (i : ℕ)                      [the parameter, a route, a position]
    direct σ f i = none ∨ entry σ f i = none            [at most one of the two is defined]
    direct                                              [what the route does without entering]
    entry                                               [the way from the even wire into the odd] -/
theorem direct_entry_disjoint (σ : R) (f : Route R) (i : ℕ) :
    direct σ f i = none ∨ entry σ f i = none := by
  rw [direct_apply, entry_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inl i)) with
  | none => left; simp
  | some p =>
    rcases outL_outR_disjoint (R := R) p.1 with h | h
    · left; simp only [Option.bind_some, h, Option.map_none]
    · right; simp only [Option.bind_some, h, Option.map_none]

/-- A composite that lands somewhere came through a first leg that landed somewhere.

    {σ : R} {a b : Route R} {i : ℕ}                        [the parameter, two routes, a position]
    (h : rcomp σ a b i ≠ none)                             [the composite goes somewhere]
    ∃ p, a i = some p ∧ b p.1 ≠ none                       [so each leg does]
    rcomp                                                  [follow one route and then another] -/
theorem rcomp_ne_none {σ : R} {a b : Route R} {i : ℕ} (h : rcomp σ a b i ≠ none) :
    ∃ p, a i = some p ∧ b p.1 ≠ none := by
  cases ha : a i with
  | none => exact absurd (by simp [rcomp, ha]) h
  | some p =>
    refine ⟨p, rfl, fun hb => h ?_⟩
    simp [rcomp, ha, hb]

/-- A path cannot leave the loop at two different depths.

    (σ : R) (f : Route R) (u : ℕ) {n k : ℕ}                      [the parameter, a route, a start]
    (hn : rcomp σ (rpow σ (body σ f) n) (leave σ f) u ≠ none)    [it leaves after n trips]
    (hm : rcomp σ (rpow σ (body σ f) (n + k + 1))                [and also after more]
      (leave σ f) u ≠ none)
    False                                                        [which cannot happen]
    rcomp                                                        [follow one route and then another]
    rpow                                                         [a route composed with itself]
    body                                                         [one trip round the loop]
    leave                                                        [the way out] -/
theorem no_two_exit_depths (σ : R) (f : Route R) (u : ℕ) {n k : ℕ}
    (hn : rcomp σ (rpow σ (body σ f) n) (leave σ f) u ≠ none)
    (hm : rcomp σ (rpow σ (body σ f) (n + k + 1)) (leave σ f) u ≠ none) : False := by
  obtain ⟨p, hp, hl⟩ := rcomp_ne_none hn
  obtain ⟨q, hq, -⟩ := rcomp_ne_none hm
  have hsplit : rcomp σ (rpow σ (body σ f) n) (rpow σ (body σ f) (k + 1)) u = some q := by
    rw [← rpow_split]; exact hq
  have hne : rcomp σ (rpow σ (body σ f) n) (rpow σ (body σ f) (k + 1)) u ≠ none := by
    rw [hsplit]; simp
  obtain ⟨p', hp', hb⟩ := rcomp_ne_none hne
  have hpp : p = p' := by rw [hp] at hp'; exact Option.some.inj hp'
  subst hpp
  rw [rpow] at hb
  obtain ⟨r, hr, -⟩ := rcomp_ne_none hb
  rcases body_leave_disjoint σ f p.1 with h | h
  · simp [h] at hr
  · exact hl h

/-- The depth at which a path leaves the loop is unique.

    (σ : R) (f : Route R) (u : ℕ) {n m : ℕ}                      [the parameter, a route, a start]
    (hn : rcomp σ (rpow σ (body σ f) n) (leave σ f) u ≠ none)    [it leaves after n trips]
    (hm : rcomp σ (rpow σ (body σ f) m) (leave σ f) u ≠ none)    [and after m trips]
    n = m                                                        [so the two counts agree]
    rcomp                                                        [follow one route and then another]
    rpow                                                         [a route composed with itself] -/
theorem exit_depth_unique (σ : R) (f : Route R) (u : ℕ) {n m : ℕ}
    (hn : rcomp σ (rpow σ (body σ f) n) (leave σ f) u ≠ none)
    (hm : rcomp σ (rpow σ (body σ f) m) (leave σ f) u ≠ none) : n = m := by
  by_contra hne
  rcases Nat.lt_or_ge n m with h | h
  · obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_lt h
    exact no_two_exit_depths σ f u hn (by rw [← hk]; exact hm)
  · obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_lt (Nat.lt_of_le_of_ne h fun hc => hne hc.symm)
    exact no_two_exit_depths σ f u hm (by rw [← hk]; exact hn)

/-- The trace: cross straight over, or go in, round any number of times, and out.

    (σ : R) (f : Route R)                          [the parameter and a route]
    (i j : ℕ) (w : RingCell R)                     [two positions and a weight]
    direct σ f i = some (j, w)                     [either it crosses straight]
    ∃ n, rcomp σ (entry σ f)                       [or it goes round n times]
      (rcomp σ (rpow σ (body σ f) n) (leave σ f)) i = some (j, w)
    ∨                                              [one or the other]
    Traces                                         [the union over every path length]
    direct                                         [what the route does without entering]
    entry                                          [the way from the even wire into the odd]
    rpow                                           [a route composed with itself]
    body                                           [one trip round the loop]
    leave                                          [the way out]
    rcomp                                          [follow one route and then another] -/
def Traces (σ : R) (f : Route R) (i j : ℕ) (w : RingCell R) : Prop :=
  direct σ f i = some (j, w) ∨
    ∃ n : ℕ, rcomp σ (entry σ f) (rcomp σ (rpow σ (body σ f) n) (leave σ f)) i = some (j, w)

/-- The union over path lengths is single-valued.

    (σ : R) (f : Route R)                          [the parameter and a route]
    {i j₁ j₂ : ℕ} {w₁ w₂ : RingCell R}             [one start, two results, two weights]
    (h₁ : Traces σ f i j₁ w₁)                      [the trace relates one]
    (h₂ : Traces σ f i j₂ w₂)                      [and the other]
    (j₁, w₁) = (j₂, w₂)                            [so they are the same]
    Traces                                         [the union over every path length]
    direct                                         [what the route does without entering]
    entry                                          [the way from the even wire into the odd] -/
theorem traces_functional (σ : R) (f : Route R) {i j₁ j₂ : ℕ} {w₁ w₂ : RingCell R}
    (h₁ : Traces σ f i j₁ w₁) (h₂ : Traces σ f i j₂ w₂) : (j₁, w₁) = (j₂, w₂) := by
  rcases h₁ with hd₁ | ⟨n, hn⟩
  · rcases h₂ with hd₂ | ⟨m, hm⟩
    · exact Option.some.inj (hd₁.symm.trans hd₂)
    · exfalso
      have hmn : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) m) (leave σ f)) i ≠ none := by rw [hm]; simp
      obtain ⟨p, hp, -⟩ := rcomp_ne_none hmn
      rcases direct_entry_disjoint σ f i with h | h
      · simp [h] at hd₁
      · simp [h] at hp
  · rcases h₂ with hd₂ | ⟨m, hm⟩
    · exfalso
      have hnn : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) n) (leave σ f)) i ≠ none := by rw [hn]; simp
      obtain ⟨p, hp, -⟩ := rcomp_ne_none hnn
      rcases direct_entry_disjoint σ f i with h | h
      · simp [h] at hd₂
      · simp [h] at hp
    · have hnn : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) n) (leave σ f)) i ≠ none := by rw [hn]; simp
      have hmn : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) m) (leave σ f)) i ≠ none := by rw [hm]; simp
      obtain ⟨p, hp, hne1⟩ := rcomp_ne_none hnn
      obtain ⟨p', hp', hne2⟩ := rcomp_ne_none hmn
      have hpp : p = p' := by rw [hp] at hp'; exact Option.some.inj hp'
      subst hpp
      have hnm : n = m := exit_depth_unique σ f p.1 hne1 hne2
      subst hnm
      exact Option.some.inj (hn.symm.trans hm)

-- ═══════════════════════════════════════════════════════════════════
-- A PATH THROUGH A ROUTE, STAYING INSIDE A SET
-- ═══════════════════════════════════════════════════════════════════

/-- A path of a given length through a route, staying inside a set until it leaves.

    (σ : R) (f : Route R) (S : Set ℕ)              [the parameter, a route, the inside]
    ℕ → ℕ → ℕ → RingCell R → Prop                 [length, start, end, weight]
    RunsN                                          [a path of a given length through a route]
    ∈                                              [is one of the things in]
    ringMul                                        [multiply two points together, the σ way] -/
inductive RunsN (σ : R) (f : Route R) (S : Set ℕ) : ℕ → ℕ → ℕ → RingCell R → Prop where
  /-- One step, landing outside the set. -/
  | exit {i j : ℕ} {w : RingCell R} : f i = some (j, w) → j ∉ S → RunsN σ f S 0 i j w
  /-- One step landing inside the set, and then the rest of the path. -/
  | step {n i k j : ℕ} {w v : RingCell R} : f i = some (k, w) → k ∈ S →
      RunsN σ f S n k j v → RunsN σ f S (n + 1) i j (ringMul σ w v)

/-- A path through a route, staying inside a set until it leaves.

    (σ : R) (f : Route R) (S : Set ℕ)              [the parameter, a route, the inside]
    (i j : ℕ) (w : RingCell R)                     [a start, an end and a weight]
    ∃ n, RunsN σ f S n i j w                       [some path of some length]
    RunsN                                          [a path of a given length through a route] -/
def Runs (σ : R) (f : Route R) (S : Set ℕ) (i j : ℕ) (w : RingCell R) : Prop :=
  ∃ n, RunsN σ f S n i j w

-- ═══════════════════════════════════════════════════════════════════
-- A FINITE BUDGET FOR THE TRACE
-- ═══════════════════════════════════════════════════════════════════

/-- Search the trace of a route up to a bound on the number of trips round the loop.

    (σ : R) (f : Route R)                    [the parameter and a route]
    ℕ → ℕ → Outcome (ℕ × RingCell R)         [a budget and a start, to a bounded result]
    Outcome                                  [one bounded search result]
    direct                                   [what the route does without entering the loop]
    entry                                    [the way from the even wire into the odd]
    rpow                                     [a route composed with itself]
    body                                     [one trip round the loop]
    leave                                    [the way out]
    rcomp                                    [follow one route and then another] -/
def traceWithin (σ : R) (f : Route R) : ℕ → ℕ → Outcome (ℕ × RingCell R)
  | 0, i =>
    match direct σ f i with
    | some p => .found p
    | none => .exhausted
  | fuel + 1, i =>
    match direct σ f i with
    | some p => .found p
    | none =>
      match rcomp σ (entry σ f) (rcomp σ (rpow σ (body σ f) fuel) (leave σ f)) i with
      | some p => .found p
      | none => traceWithin σ f fuel i

/-- A result found inside the budget is a trace.

    (σ : R) (f : Route R)                                [the parameter and a route]
    ∀ {fuel i j w}                                       [every budget, start, end and weight]
    traceWithin σ f fuel i = .found (j, w)               [the bounded search found them]
    → Traces σ f i j w                                   [so the trace relates them]
    traceWithin                                          [the bounded search for the trace]
    Traces                                               [the union over every path length] -/
theorem traceWithin_sound (σ : R) (f : Route R) :
    ∀ {fuel i j w}, traceWithin σ f fuel i = .found (j, w) → Traces σ f i j w := by
  intro fuel
  induction fuel with
  | zero =>
    intro i j w h
    cases hd : direct σ f i with
    | some p =>
      simp only [traceWithin, hd] at h
      injection h with hp
      subst hp
      exact Or.inl hd
    | none =>
      simp only [traceWithin, hd] at h
      cases h
  | succ fuel ih =>
    intro i j w h
    cases hd : direct σ f i with
    | some p =>
      simp only [traceWithin, hd] at h
      injection h with hp
      subst hp
      exact Or.inl hd
    | none =>
      simp only [traceWithin, hd] at h
      cases hb : rcomp σ (entry σ f)
          (rcomp σ (rpow σ (body σ f) fuel) (leave σ f)) i with
      | some p =>
        simp only [hb] at h
        injection h with hp
        subst hp
        exact Or.inr ⟨fuel, hb⟩
      | none =>
        simp only [hb] at h
        exact ih h

/-- Every trace is found at some finite budget.

    (σ : R) (f : Route R)                                [the parameter and a route]
    {i j : ℕ} {w : RingCell R}                           [a start, an end and a weight]
    Traces σ f i j w                                     [the trace relates them]
    ∃ fuel, traceWithin σ f fuel i = .found (j, w)       [so some budget finds them]
    Traces                                               [the union over every path length]
    traceWithin                                          [the bounded search for the trace]
    ∃                                                    [there is at least one] -/
theorem traces_exists_fuel (σ : R) (f : Route R) {i j : ℕ} {w : RingCell R}
    (h : Traces σ f i j w) : ∃ fuel, traceWithin σ f fuel i = .found (j, w) := by
  rcases h with hd | ⟨n, hn⟩
  · exact ⟨0, by simp only [traceWithin, hd]⟩
  · cases hdir : direct σ f i with
    | some p =>
      have hp : p = (j, w) := traces_functional σ f (Or.inl hdir) (Or.inr ⟨n, hn⟩)
      subst hp
      exact ⟨0, by simp only [traceWithin, hdir]⟩
    | none =>
      refine ⟨n + 1, ?_⟩
      simp only [traceWithin, hdir]
      rw [hn]

/-- The bounded search covers exactly the relational trace.

    (σ : R) (f : Route R) (i j : ℕ) (w : RingCell R)      [the parameter, a route, a start]
    (∃ fuel, traceWithin σ f fuel i = .found (j, w))       [some budget finds them]
    ↔                                                      [exactly when]
    Traces σ f i j w                                       [the trace relates them]
    traceWithin                                            [the bounded search for the trace]
    Traces                                                 [the union over every path length] -/
theorem traceWithin_iff_traces (σ : R) (f : Route R) (i j : ℕ) (w : RingCell R) :
    (∃ fuel, traceWithin σ f fuel i = .found (j, w)) ↔ Traces σ f i j w :=
  ⟨fun ⟨_, h⟩ => traceWithin_sound σ f h, fun h => traces_exists_fuel σ f h⟩

-- ═══════════════════════════════════════════════════════════════════
-- A FINITE BUDGET FOR A PATH
-- ═══════════════════════════════════════════════════════════════════

/-- Search a path through a route up to a bound on the steps taken.

    (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]   [the parameter, a route, the inside]
    ℕ → ℕ → Outcome (ℕ × RingCell R)                      [a budget and a start, to a result]
    Outcome                                               [one bounded search result]
    RunsN                                                 [a path of a given length]
    ringMul                                               [multiply two points together] -/
def runsWithin (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S] :
    ℕ → ℕ → Outcome (ℕ × RingCell R)
  | 0, _ => .exhausted
  | fuel + 1, i =>
    match f i with
    | none => .exhausted
    | some (k, w) =>
      if _ : S k then
        match runsWithin σ f S fuel k with
        | .found p => .found (p.1, ringMul σ w p.2)
        | .exhausted => .exhausted
      else .found (k, w)

/-- One step of the bounded path search, written out.

    (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]   [the parameter, a route, the inside]
    (fuel i : ℕ)                                          [a budget and a start]
    runsWithin σ f S (fuel + 1) i                         [the search with one unit spent]
    ringMul σ w p.2                                       [the step's weight, multiplied on]
    runsWithin                                            [the bounded search for a path]
    ∈                                                     [is one of the things in] -/
theorem runsWithin_succ (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]
    (fuel i : ℕ) :
    runsWithin σ f S (fuel + 1) i =
      (match f i with
       | none => .exhausted
       | some (k, w) =>
         if _ : S k then
           match runsWithin σ f S fuel k with
           | .found p => .found (p.1, ringMul σ w p.2)
           | .exhausted => .exhausted
         else .found (k, w)) := rfl

/-- A result found inside the budget is a path.

    (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]   [the parameter, a route, the inside]
    ∀ {fuel i j w}                                        [every budget, start, end and weight]
    runsWithin σ f S fuel i = .found (j, w)               [the bounded search found them]
    → Runs σ f S i j w                                    [so a path relates them]
    runsWithin                                            [the bounded search for a path]
    Runs                                                  [a path through a route] -/
theorem runsWithin_sound (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S] :
    ∀ {fuel i j w}, runsWithin σ f S fuel i = .found (j, w) → Runs σ f S i j w := by
  intro fuel
  induction fuel with
  | zero =>
    intro i j w h
    simp only [runsWithin] at h
    cases h
  | succ fuel ih =>
    intro i j w h
    rw [runsWithin_succ] at h
    cases hf : f i with
    | none =>
      simp only [hf] at h
      cases h
    | some p =>
      obtain ⟨k, w0⟩ := p
      by_cases hs : S k
      · simp only [hf, dif_pos hs] at h
        cases hr : runsWithin σ f S fuel k with
        | found q =>
          simp only [hr] at h
          injection h with hq
          have hq1 : q.1 = j := congrArg Prod.fst hq
          have hq2 : ringMul σ w0 q.2 = w := congrArg Prod.snd hq
          subst hq1
          subst hq2
          obtain ⟨n, hn⟩ := ih hr
          exact ⟨n + 1, RunsN.step hf hs hn⟩
        | exhausted =>
          simp only [hr] at h
          cases h
      · simp only [hf, dif_neg hs] at h
        injection h with hq
        have hq1 : k = j := congrArg Prod.fst hq
        have hq2 : w0 = w := congrArg Prod.snd hq
        subst hq1
        subst hq2
        exact ⟨0, RunsN.exit hf hs⟩

/-- Every path is found at some finite budget.

    (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]   [the parameter, a route, the inside]
    ∀ {n i j w}                                           [every length, start, end and weight]
    RunsN σ f S n i j w                                   [a path of that length relates them]
    → runsWithin σ f S (n + 1) i = .found (j, w)          [so one past the length finds them]
    RunsN                                                 [a path of a given length]
    runsWithin                                            [the bounded search for a path] -/
theorem runsWithin_complete (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S] :
    ∀ {n i j w}, RunsN σ f S n i j w → runsWithin σ f S (n + 1) i = .found (j, w) := by
  intro n i j w h
  induction h with
  | @exit i j w hf hns =>
    rw [runsWithin_succ]
    simp only [hf]
    rw [dif_neg (show ¬ S j from hns)]
  | @step n i k j w v hf hs hr ih =>
    rw [runsWithin_succ]
    simp only [hf]
    rw [dif_pos (show S k from hs)]
    rw [ih]

/-- The bounded search covers exactly the relational path.

    (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]   [the parameter, a route, the inside]
    (i j : ℕ) (w : RingCell R)                            [a start, an end and a weight]
    (∃ fuel, runsWithin σ f S fuel i = .found (j, w))     [some budget finds them]
    ↔                                                     [exactly when]
    Runs σ f S i j w                                      [a path relates them]
    runsWithin                                            [the bounded search for a path]
    Runs                                                  [a path through a route] -/
theorem runsWithin_iff_Runs (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]
    (i j : ℕ) (w : RingCell R) :
    (∃ fuel, runsWithin σ f S fuel i = .found (j, w)) ↔ Runs σ f S i j w := by
  constructor
  · rintro ⟨fuel, h⟩
    exact runsWithin_sound σ f S h
  · rintro ⟨n, hn⟩
    exact ⟨n + 1, runsWithin_complete σ f S hn⟩

end RouteAnyRing
end Combinator
end Apeiron
