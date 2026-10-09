import Apeiron.Combinator.Linear.Core.Interaction
import Apeiron.Combinator.RouteAnyRing

/-!
# Using an argument twice: the exponential, and what it takes to reach S

@name Duplication
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924
@cite curry-1930
@grade proved

`Combinator/Interaction.lean` proves `I x = x` and `K x y = x` with explicit routings, and both
work because the loop is empty: the argument is consulted once and the path leaves. `S` is the
combinator that consults its argument twice, so it is the one that needs the exponential rather
than only the two-track retraction. This file builds the pieces that need, in order.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | `bang_copy`, `rapp_iRoute`, `rapp_kRoute` |
| `nilpotent_iff_disc_zero`, `divergent_loop_of_disc_pos` | the face reading in the last
  section below |

The wiring is free of the parameter for the same reason it was in `Interaction.lean`: a partial
injection carries at most one path out of each position, so composing multiplies one weight along
one path and the retractions are equalities at every setting. Where it enters is stated in
`Interaction.lean`'s `weight_blocks_the_loop_iff_disc_nonneg` and, for the loop `S` opens, in
the last section below.

## The four rungs

**One.** `TracesTo` names the situation where a route's trace is itself a route, and
`tracesTo_unique` says such a route is unique. `tracesTo_of_body_rnone` extracts what `I x = x`
and `K x y = x` both used: when the loop is empty, the trace is the straight crossing joined with
one pass, and no induction over path length is needed.

**Two.** `bang` is the exponential applied to a route: the same route running inside every one of
the countably many copies that `Nat.pair` supplies. `bang_copy` says every consumer that reads
copy `n` gets the whole of `z` back, and `bang_serves_two` says two consumers reading different
copies each get the whole of `z` while their channels do not meet. That is duplication, as an
equation on routes rather than as an assumption.

**Three.** The loop that runs more than once. The only channel to the argument is the loop, so
consulting the argument twice is the loop running once: `tracesTo_of_body_sq_rnone` is that case,
with three terms instead of two, and `body_leave_disjoint` is what keeps them from overlapping.
`rcomp_rplus` is the distribution that lets the terms be collected; the mirror law on the left is
false and is not used.

**Four.** `traceRoute` makes the trace total. Its value at a position is the unique result the
trace relates it to, which exists by `traces_functional`, so no bound on path length is needed
and a path that runs for ever simply leaves the route undefined there. With `rapp` for
application, `rapp_iRoute` and `rapp_kRoute` restate `I x = x` and `K x y = x` as equations
between routes rather than as biconditionals about a relation.

## What `S` still needs, exactly

`S x y z = x z (y z)` is stated in this setting as
`rapp σ (rapp σ (rapp σ s x) y) z = rapp σ (rapp σ x z) (rapp σ y z)` for a routing `s`. Rung
four removes the obstruction `Combinator/Interaction.lean` recorded, that the union over path
lengths might not be a route: it is one, always, so the routing `s` is now writable, and rung
five supplies the tagging that keeps two consumers of one argument apart.

What is left is one property of the trace, and the reason it is needed is worth stating because
it also explains why `I` and `K` came without it.

In `rapp σ a b` the applied route's odd half faces the argument, and one step of that route is
one traversal of it. A combinator that has to **compose** its arguments, feeding one argument's
output into another argument's input, cannot do that in one step: the transition between the two
is neither an exit nor a query to the outer argument, so it has to ride a loop of its own,
internal to the stored route. A route is a single partial injection and has no loop closure, so
that internal loop has to be a trace, and the stored route becomes `traceRoute` of a larger
wiring. Verifying its equation then needs

> the trace of a route built from traces is the trace of one larger route,

which is Joyal, Street and Verity's vanishing over a pair at the level of `Traces`.
`Trichotomy/Trace.lean`'s `feedback_vanishing_pair` is the algebraic form of the same axiom and
is proved there; the route-level form is not, and it is the single thing standing between this
file and `S`.

**Which combinators that argument reaches.** `I` composes nothing, and `K` discards its second
argument, so neither needs an internal loop, and both came out of rungs one to four. Every other
combinator of the standard bases composes: `B x y z = x (y z)`, `C x y z = x z y`,
`W x y = x y y` and `S x y z = x z (y z)` all feed one argument to another. So the combinators
reachable here without the vanishing property are exactly the two that are proved, and the
boundary is not an accident of how the routings were chosen.

**Where `S` already exists in this corpus.** `Combinator/Model.lean` proves `feed_sPoint` and
`feed_meaning_bracket`: the duplicating equation as an equality of sets, and combinatory
completeness, in Engeler's graph model built over the algebra's idempotents. So the corpus is not
without `S`. What this file is doing is different and is the harder half: getting the same
combinator in the feedback setting, where the algebra's weights ride along the wires and the
parameter has somewhere to enter.

**Where the parameter enters, if it does.** Whether the trace is defined at a position is a
question about the wiring alone, and `traceRoute` answers it at every setting, so the parameter
does not gate the statement of `S`. What it decides is whether a loop the wiring leaves open can
be closed by the weight: `Combinator/Interaction.lean`'s
`weight_blocks_the_loop_iff_disc_nonneg` says a weight that neither dies out nor closes exists at
zero and above and nowhere else, and `Trichotomy/Trace.lean`'s `nilpotent_iff_disc_zero` says a
weight that dies out exists at zero alone. So the degenerate face is the one on which a weight
can end a loop the wiring would run for ever. Neither fact is used by `I` or `K`, whose loops are
empty or run once, and neither would be needed for `S` itself: they bear on `S` applied to
arguments whose own loops do not terminate.

**What the bounded kernel adds.** `traceRoute` is total because `Classical.choice` picks a witness
from the unbounded existential in `Traces`, which makes it a relation rather than a program: no
algorithm decides an unbounded union. Rung eight adds `traceWithin`, which takes the path length
as an explicit budget and returns a value or an explicit exhaustion, and
`traceWithin_iff_traceRoute` says the two agree wherever the bounded one answers. An exhausted
budget is not a claim that the route diverges, and the relational `Traces` is left in place.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

open scoped Classical

-- ═══════════════════════════════════════════════════════════════════
-- RUNG ONE: THE TRACE AS A ROUTE, AND WHEN IT IS ONE PASS
-- ═══════════════════════════════════════════════════════════════════

/-- The trace of one route is another route.

    (σ : ℝ) (f r : Route)                                  [the parameter and two routes]
    ∀ i j w, Traces σ f i j w ↔ r i = some (j, w)          [the trace of f is exactly r]

    So: `Interaction.lean` states its combinator equations in this shape, and naming it lets
    them be chained. -/
def TracesTo (σ : ℝ) (f r : Route) : Prop :=
  ∀ i j w, Traces σ f i j w ↔ r i = some (j, w)

/-- A route has at most one trace.

    {σ : ℝ} {f r r' : Route}                               [the parameter and three routes]
    (h : TracesTo σ f r) (h' : TracesTo σ f r')            [both are the trace of f]
    r = r'                                                 [so they are the same route]

    So: `traces_functional` says the trace relates each position to at most one result, and this
    is the consequence for routes. `TracesTo` is a partial function, which is what makes an
    applicative structure out of it. -/
theorem tracesTo_unique {σ : ℝ} {f r r' : Route} (h : TracesTo σ f r) (h' : TracesTo σ f r') :
    r = r' := by
  funext i
  cases hr : r i with
  | none =>
    cases hr' : r' i with
    | none => rfl
    | some p =>
      have hp : r' i = some (p.1, p.2) := by simp [hr']
      have hc := (h i p.1 p.2).mp ((h' i p.1 p.2).mpr hp)
      rw [hr] at hc
      exact absurd hc (by simp)
  | some p =>
    have hp : r i = some (p.1, p.2) := by simp [hr]
    have hc := (h' i p.1 p.2).mp ((h i p.1 p.2).mpr hp)
    rw [hc]

/-- Join two routes, preferring the first.

    (a b : Route)                          [two routes]
    Route                                  [the first where it is defined, else the second]

    So: the union of two wirings. Where the two are disjoint, as the straight crossing and the
    way into the loop are (`direct_entry_disjoint`), this is their disjoint union. -/
def rplus (a b : Route) : Route := fun i =>
  match a i with
  | some p => some p
  | none => b i

/-- Where the first route is defined the join follows it.

    {a b : Route} {i : ℕ} {p : ℕ × Cell}           [two routes, a position, a result]
    (h : a i = some p)                             [the first route lands there]
    rplus a b i = some p                           [so the join does too] -/
theorem rplus_of_some {a b : Route} {i : ℕ} {p : ℕ × Cell} (h : a i = some p) :
    rplus a b i = some p := by
  simp [rplus, h]

/-- Where the first route is undefined the join is the second.

    {a b : Route} {i : ℕ}                          [two routes and a position]
    (h : a i = none)                               [the first route stops there]
    rplus a b i = b i                              [so the join is the second route] -/
theorem rplus_of_none {a b : Route} {i : ℕ} (h : a i = none) :
    rplus a b i = b i := by
  simp [rplus, h]

/-- **When the loop is empty the trace is one pass.**

@cite girard-1989

    (σ : ℝ) (f : Route)                                          [the parameter and a route]
    (hb : body σ f = rnone)                        [the loop closes on itself nowhere]
    TracesTo σ f (rplus (direct σ f)               [so the trace is the straight crossing]
      (rcomp σ (entry σ f) (leave σ f)))           [joined with one pass through the loop]

    So: the execution formula with one term. Both `iRoute_app` and `kRoute_app` are instances,
    and this is the lemma that spares the next construction their path induction. -/
theorem tracesTo_of_body_rnone (σ : ℝ) (f : Route) (hb : body σ f = rnone) :
    TracesTo σ f (rplus (direct σ f) (rcomp σ (entry σ f) (leave σ f))) := by
  intro i j w
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · exact rplus_of_some hd
    · cases n with
      | zero =>
        rw [rpow, rcomp_rid_left] at hn
        rcases direct_entry_disjoint σ f i with h | h
        · rw [rplus_of_none h]; exact hn
        · rw [rcomp, h] at hn
          simp at hn
      | succ n =>
        rw [hb, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
        simp [rnone] at hn
  · intro h
    cases hd : direct σ f i with
    | some p =>
      refine Or.inl ?_
      rw [rplus_of_some hd] at h
      simpa [hd] using h
    | none =>
      refine Or.inr ⟨0, ?_⟩
      rw [rpow, rcomp_rid_left]
      rw [rplus_of_none hd] at h
      exact h

-- ═══════════════════════════════════════════════════════════════════
-- RUNG TWO: THE EXPONENTIAL HANDS ONE ARGUMENT TO TWO CONSUMERS
-- ═══════════════════════════════════════════════════════════════════

/-- A route running inside every one of the countably many copies.

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (z : Route)                            [a route]
    Route                                  [the same route, once in each copy]

    So: the exponential applied to a route. `Interaction.lean`'s `copy` and `uncopy` place a
    wire in one copy and read it back; this runs `z` in all of them at once, keeping the copy
    index untouched. -/
def bang (z : Route) : Route := fun j =>
  (z (Nat.unpair j).2).map fun q => (Nat.pair (Nat.unpair j).1 q.1, q.2)

/-- **Every consumer of the exponential gets the whole route back.**

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (z : Route) (n : ℕ)                    [the parameter, a route, one copy]
    rcomp σ (copy n) (rcomp σ (bang z) (uncopy n)) = z   [reading copy n returns z]

    So: place a wire in copy `n`, run the exponential, read copy `n` back, and the composite is
    `z` itself, weights included. This is the duplication `S` needs, as an equation. -/
theorem bang_copy (σ : ℝ) (z : Route) (n : ℕ) :
    rcomp σ (copy n) (rcomp σ (bang z) (uncopy n)) = z := by
  funext i
  simp only [rcomp, copy, Option.bind_some, bang, Nat.unpair_pair]
  cases hz : z i with
  | none => simp
  | some p =>
    simp [uncopy, Nat.unpair_pair]

/-- **Two consumers, one argument, no interference.**

@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924

    (σ : ℝ) (z : Route) {n m : ℕ} (h : n ≠ m)      [the parameter, a route, two copies]
    rcomp σ (copy n) (rcomp σ (bang z) (uncopy n)) = z   [the first consumer gets z]
    rcomp σ (copy m) (rcomp σ (bang z) (uncopy m)) = z   [the second gets z too]
    rcomp σ (copy n) (uncopy m) = rnone            [and their channels do not meet]
    ∧                                              [all three together]

    So: the same `z` is handed to two consumers in full, and nothing crosses between them.
    `S x y z = x z (y z)` needs exactly this and nothing weaker: `y` and `x` must each receive
    the whole of `z`, and the two readings must not interfere. -/
theorem bang_serves_two (σ : ℝ) (z : Route) {n m : ℕ} (h : n ≠ m) :
    rcomp σ (copy n) (rcomp σ (bang z) (uncopy n)) = z ∧
      rcomp σ (copy m) (rcomp σ (bang z) (uncopy m)) = z ∧
      rcomp σ (copy n) (uncopy m) = rnone :=
  ⟨bang_copy σ z n, bang_copy σ z m, uncopy_copy_of_ne σ h⟩

-- ═══════════════════════════════════════════════════════════════════
-- RUNG THREE: THE LOOP THAT RUNS MORE THAN ONCE
-- ═══════════════════════════════════════════════════════════════════

/-- Composing distributes over a join on the right.

    (σ : ℝ) (a b c : Route)                        [the parameter and three routes]
    rcomp σ a (rplus b c) = rplus (rcomp σ a b) (rcomp σ a c)   [either order is the same]

    So: one path leaves each position, so following it and then choosing is choosing and then
    following. The mirror statement on the left is false and is not used: joining first can
    hide a wire that the second leg would have carried. -/
theorem rcomp_rplus (σ : ℝ) (a b c : Route) :
    rcomp σ a (rplus b c) = rplus (rcomp σ a b) (rcomp σ a c) := by
  funext i
  cases ha : a i with
  | none => simp [rcomp, rplus, ha]
  | some p =>
    cases hb : b p.1 with
    | none => simp [rcomp, rplus, ha, hb]
    | some q => simp [rcomp, rplus, ha, hb]

/-- One trip is the loop itself.

    (σ : ℝ) (b : Route)                    [the parameter and a route]
    rpow σ b 1 = b                         [going round once is the route] -/
theorem rpow_one (σ : ℝ) (b : Route) : rpow σ b 1 = b := by
  show rcomp σ b (rpow σ b 0) = b
  rw [rpow, rcomp_rid_right]

/-- A loop that dies after two trips is dead at every greater depth.

    (σ : ℝ) (b : Route)                                          [the parameter and a route]
    (hb : rcomp σ b b = rnone)                                   [two trips connect nothing]
    (n : ℕ)                                                      [any further depth]
    rpow σ b (n + 2) = rnone                                     [connects nothing either] -/
theorem rpow_rnone_of_sq (σ : ℝ) (b : Route) (hb : rcomp σ b b = rnone) (n : ℕ) :
    rpow σ b (n + 2) = rnone := by
  induction n with
  | zero =>
    show rcomp σ b (rcomp σ b (rpow σ b 0)) = rnone
    rw [rpow, rcomp_rid_right, hb]
  | succ n ih =>
    show rcomp σ b (rpow σ b (n + 2)) = rnone
    rw [ih, rcomp_rnone_right]

/-- **When the loop runs at most once the trace has three terms.**

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    (hb : rcomp σ (body σ f) (body σ f) = rnone)   [no path goes round twice]
    TracesTo σ f (rplus (direct σ f)               [the trace is the straight crossing]
      (rcomp σ (entry σ f) (rplus (leave σ f)      [joined with in and straight out]
        (rcomp σ (body σ f) (leave σ f)))))        [joined with in, once round, and out]

    So: the execution formula with two terms rather than one. This is the first case in which the
    argument is consulted twice, because the only channel to the argument is the loop: one trip
    is one consultation. `body_leave_disjoint` is what makes the join a disjoint one, so the
    three terms do not overlap and the trace stays single-valued. -/
theorem tracesTo_of_body_sq_rnone (σ : ℝ) (f : Route)
    (hb : rcomp σ (body σ f) (body σ f) = rnone) :
    TracesTo σ f (rplus (direct σ f)
      (rcomp σ (entry σ f) (rplus (leave σ f) (rcomp σ (body σ f) (leave σ f))))) := by
  intro i j w
  rw [rcomp_rplus]
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · exact rplus_of_some hd
    · match n, hn with
      | 0, hn =>
        rw [rpow, rcomp_rid_left] at hn
        rcases direct_entry_disjoint σ f i with h | h
        · rw [rplus_of_none h, rplus_of_some hn]
        · rw [rcomp, h] at hn; simp at hn
      | 1, hn =>
        rw [rpow_one] at hn
        rcases direct_entry_disjoint σ f i with hde | hde
        · have hnone : rcomp σ (entry σ f) (leave σ f) i = none := by
            cases he : entry σ f i with
            | none => simp [rcomp, he]
            | some p =>
              have hbne : body σ f p.1 ≠ none := by
                intro hc
                simp [rcomp, he, hc] at hn
              rcases body_leave_disjoint σ f p.1 with h | h
              · exact absurd h hbne
              · simp [rcomp, he, h]
          rw [rplus_of_none hde, rplus_of_none hnone]
          exact hn
        · rw [rcomp, hde] at hn; simp at hn
      | (n + 2), hn =>
        rw [rpow_rnone_of_sq σ _ hb, rcomp_rnone_left, rcomp_rnone_right] at hn
        simp [rnone] at hn
  · intro h
    cases hd : direct σ f i with
    | some p =>
      refine Or.inl ?_
      rw [rplus_of_some hd] at h
      simpa [hd] using h
    | none =>
      rw [rplus_of_none hd] at h
      cases h0 : rcomp σ (entry σ f) (leave σ f) i with
      | some q =>
        refine Or.inr ⟨0, ?_⟩
        rw [rpow, rcomp_rid_left, h0]
        rw [rplus_of_some h0] at h
        exact h
      | none =>
        refine Or.inr ⟨1, ?_⟩
        rw [rpow_one]
        rw [rplus_of_none h0] at h
        exact h

-- ═══════════════════════════════════════════════════════════════════
-- RUNG FOUR: APPLICATION AS A TOTAL OPERATION ON ROUTES
-- ═══════════════════════════════════════════════════════════════════

/-- The trace of a route, as a route.

@cite girard-1989

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    Route                                          [the route its trace is]

    So: `traces_functional` says the trace relates each position to at most one result, so
    picking that result where it exists gives a route. The choice is the only place this file
    uses `Classical.choice`, and it picks from a subsingleton. -/
def traceRoute (σ : ℝ) (f : Route) : Route := fun i =>
  if h : ∃ p : ℕ × Cell, Traces σ f i p.1 p.2 then some (Classical.choose h) else none

/-- **Every route has a trace, and it is a route.**

@cite girard-1989
@cite haghverdi-scott-2010

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    TracesTo σ f (traceRoute σ f)                  [its trace is that route]

    So: the trace is total as an operation on routes, with no bound on path length needed. A
    path may run for ever and then the route is undefined at that position, which is what a
    partial trace means. With `tracesTo_unique` this makes `TracesTo` a total function and lets
    the combinator equations be written as equations between routes rather than as biconditionals
    about the trace relation. -/
theorem tracesTo_traceRoute (σ : ℝ) (f : Route) : TracesTo σ f (traceRoute σ f) := by
  intro i j w
  constructor
  · intro ht
    have hex : ∃ p : ℕ × Cell, Traces σ f i p.1 p.2 := ⟨(j, w), ht⟩
    have hsp := Classical.choose_spec hex
    have heq := traces_functional σ f hsp ht
    simp only [traceRoute, dif_pos hex]
    rw [← heq]
  · intro h
    by_cases hex : ∃ p : ℕ × Cell, Traces σ f i p.1 p.2
    · have hsp := Classical.choose_spec hex
      simp only [traceRoute, dif_pos hex, Option.some.injEq] at h
      rw [h] at hsp
      exact hsp
    · simp only [traceRoute, dif_neg hex] at h
      exact absurd h (by simp)

/-- Applying one route to another.

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (a b : Route)                          [the parameter and two routes]
    traceRoute σ (plug σ a b)                      [plug the argument in and close the loop]
    : Route                                        [one route]

    So: composition with feedback, taken as an operation. This is the applicative structure the
    combinator equations live in. -/
def rapp (σ : ℝ) (a b : Route) : Route := traceRoute σ (plug σ a b)

/-- Reading off an applicative equation from a trace.

    {σ : ℝ} {a b r : Route}                        [the parameter, two routes and a third]
    (h : TracesTo σ (plug σ a b) r)                [the trace of the plug is r]
    rapp σ a b = r                                 [so applying gives r]

    So: `tracesTo_unique` in the shape the combinator equations need. -/
theorem rapp_eq {σ : ℝ} {a b r : Route} (h : TracesTo σ (plug σ a b) r) : rapp σ a b = r :=
  tracesTo_unique (tracesTo_traceRoute σ _) h

/-- **I x = x**, as an equation between routes.

@cite schonfinkel-1924

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    rapp σ iRoute x = x                            [the swap applied to x is x]

    So: `iRoute_app` with application taken as an operation rather than as a relation. -/
theorem rapp_iRoute (σ : ℝ) (x : Route) : rapp σ iRoute x = x :=
  rapp_eq (iRoute_app σ x)

/-- **K x y = x**, as an equation between routes.

@cite schonfinkel-1924
@cite curry-1930

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    rapp σ (rapp σ kRoute x) y = x                 [the discarding routing throws y away]

    So: `kRoute_app` and `emb_app` chained, which is what `tracesTo_unique` was for. The first
    application stores `x` beside the empty route and the second hands it back. -/
theorem rapp_kRoute (σ : ℝ) (x y : Route) : rapp σ (rapp σ kRoute x) y = x := by
  rw [rapp_eq (kRoute_app σ x)]
  exact rapp_eq (emb_app σ x y)

-- ═══════════════════════════════════════════════════════════════════
-- RUNG FIVE: MANY CONSUMERS, ONE ARGUMENT, TAGS KEPT APART
-- ═══════════════════════════════════════════════════════════════════

/-- The exponential keeps the requester's tag and works on the payload.

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (z : Route) (n u : ℕ)                          [a route, a tag and a position]
    bang z (Nat.pair n u)                          [the exponential at a tagged position]
    = (z u).map fun q => (Nat.pair n q.1, q.2)     [runs z on the payload, tag untouched]

    So: a request tagged `n` comes back tagged `n`, so answers reach the requester that asked.
    A route holds no state, so one `z` can serve unboundedly many requests; what the exponential
    supplies is not extra copies but the bookkeeping that keeps the requests apart. That is what
    `S` needs from it. -/
theorem bang_tag (z : Route) (n u : ℕ) :
    bang z (Nat.pair n u) = (z u).map fun q => (Nat.pair n q.1, q.2) := by
  simp [bang, Nat.unpair_pair]

/-- **The weakening equation of a linear combinatory algebra**, `K x !y = x`.

@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    rapp σ (rapp σ kRoute x) (bang y) = x          [the discarded argument may be banged]

    So: Abramsky, Haghverdi and Scott's weakening combinator, in their notation `K x !y = x`.
    The discarding routing reaches no wire of its second argument, so banging that argument
    changes nothing, and the equation follows from `rapp_kRoute` with no further work. This is
    the first of their eight combinator equations to hold here. -/
theorem rapp_kRoute_bang (σ : ℝ) (x y : Route) :
    rapp σ (rapp σ kRoute x) (bang y) = x :=
  rapp_kRoute σ x (bang y)

-- ═══════════════════════════════════════════════════════════════════
-- RUNG SIX: THE EXPONENTIAL IS A FUNCTOR, AND THE MERGE IT PERFORMS
-- ═══════════════════════════════════════════════════════════════════

/-- The exponential keeps the identity.

@cite abramsky-haghverdi-scott-2002

    bang rid = rid                                 [the identity in every copy is itself] -/
theorem bang_rid : bang rid = rid := by
  funext j
  simp [bang, rid, Nat.pair_unpair]

/-- The exponential keeps the empty route.

    bang rnone = rnone                             [running nothing in every copy is nothing] -/
theorem bang_rnone : bang rnone = rnone := by
  funext j
  simp [bang, rnone]

/-- **The exponential carries composition.**

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (a b : Route)                          [the parameter and two routes]
    bang (rcomp σ a b) = rcomp σ (bang a) (bang b) [a composite in every copy is each]

    So: this is Abramsky, Haghverdi and Scott's `F !x !y = !(x y)`, the functoriality of the
    exponential, in the form a carrier of routes takes. With `bang_rid` it says `bang` is a
    functor on the monoid `rcomp` makes, which is what lets application-with-bang below compose
    at all. -/
theorem bang_rcomp (σ : ℝ) (a b : Route) :
    bang (rcomp σ a b) = rcomp σ (bang a) (bang b) := by
  funext j
  cases ha : a (Nat.unpair j).2 with
  | none => simp [bang, rcomp, ha]
  | some p =>
    cases hb : b p.1 with
    | none => simp [bang, rcomp, ha, hb]
    | some q => simp [bang, rcomp, ha, hb]

/-- **The merge, one direction.** One tag of the exponential is the route itself.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (z : Route) (n v : ℕ)                  [the parameter, a route, a tag, a position]
    bang z (Nat.pair n v)                          [the exponential at that tag]
    = rcomp σ (uncopy n) (rcomp σ z (copy n)) (Nat.pair n v)   [is enter the tag, run z, leave]

    So: what `bang z` does inside one tag is exactly `z`, wrapped in entering and leaving that
    tag. Together with `bang_copy` this is the merge: two argument-facing loops on one side of a
    combinator equation become one loop on the other, told apart by their tags, and neither
    consumer can see the other's because `uncopy_copy_of_ne` says the channels do not meet. -/
theorem bang_restrict (σ : ℝ) (z : Route) (n v : ℕ) :
    bang z (Nat.pair n v) = rcomp σ (uncopy n) (rcomp σ z (copy n)) (Nat.pair n v) := by
  rw [bang_tag]
  cases hz : z v with
  | none => simp [rcomp, uncopy, Nat.unpair_pair, hz]
  | some p => simp [rcomp, uncopy, copy, Nat.unpair_pair, hz]

/-- **The merge, other direction.** Outside its own tag the exponential is invisible.

    (σ : ℝ) (z : Route) {n m : ℕ} (h : n ≠ m)      [the parameter, a route, two tags]
    (v : ℕ)                                        [a position, read at the wrong tag]
    rcomp σ (uncopy m) (rcomp σ z (copy m)) (Nat.pair n v) = none   [nothing comes back]

    So: a consumer reading tag `m` sees nothing of a request made at tag `n`. With
    `bang_restrict` the two together say the exponential splits into independent tagged copies,
    which is what makes two consumers of one argument safe. -/
theorem bang_restrict_of_ne (σ : ℝ) (z : Route) {n m : ℕ} (h : n ≠ m) (v : ℕ) :
    rcomp σ (uncopy m) (rcomp σ z (copy m)) (Nat.pair n v) = none := by
  simp [rcomp, uncopy, Nat.unpair_pair, h]

-- ═══════════════════════════════════════════════════════════════════
-- RUNG SEVEN: APPLICATION WITH THE EXPONENTIAL
-- ═══════════════════════════════════════════════════════════════════

/-- Applying a route to the exponential of another.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (a b : Route)                          [the parameter and two routes]
    rapp σ a (bang b)                              [feed the argument once for every consumer]
    : Route                                        [one route]

    So: Abramsky, Haghverdi and Scott reach the ordinary combinatory algebra from the linear one
    by `x · y := x (!y)`, and this is that operation. `rapp` on its own is linear and cannot
    carry a combinator that uses an argument twice; this can. -/
def capp (σ : ℝ) (a b : Route) : Route := rapp σ a (bang b)

/-- The discarding routing under application-with-bang returns the exponential of its argument.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    capp σ (capp σ kRoute x) y = bang x            [it hands back the banged first argument]

    So: this is `K x !y = x` read through `x · y := x (!y)`, and it comes out as `bang x` rather
    than `x`. That is not a defect of the routing, it is why Abramsky, Haghverdi and Scott's
    translation from the linear algebra to the ordinary one needs a dereliction `D !x = x`, and
    why the ordinary discarding combinator is `B K D` and not `K`. Recorded rather than hidden,
    because it is the exact shape of what the translation still owes. -/
theorem capp_kRoute (σ : ℝ) (x y : Route) : capp σ (capp σ kRoute x) y = bang x :=
  rapp_kRoute σ (bang x) (bang y)

-- ═══════════════════════════════════════════════════════════════════
-- RUNG EIGHT: A FINITE BUDGET, AND WHAT IT CANNOT DECIDE
-- ═══════════════════════════════════════════════════════════════════

/-- The result of a bounded search: a value found inside the budget, or the budget exhausted.

    Outcome α                       [one bounded search result]
    found (a : α)                   [a value was found before the budget ran out]
    exhausted                       [the budget ran out with no value found] -/
abbrev Outcome (α : Type) : Type := RouteAnyRing.Outcome α

/-- Search the trace of a route up to a bound on the number of trips round the loop.

    (σ : ℝ) (f : Route)             [the parameter and a route]
    ℕ → ℕ → Outcome (ℕ × Cell)      [a budget and a start, to a bounded result]

    So: `Traces` is a union over every path length, and no finite program can decide it, so the
    length is an explicit budget here. At budget zero only the straight crossing is checked; at
    `fuel + 1` one more trip is checked than at `fuel`, so the trips below `fuel` are all tried.
    An exhausted budget is not a proof of divergence: it says only that nothing was found within
    the bound, which is why the result is not an `Option`. The `match` on `Option` is the whole
    control flow, so the budget adds no `Classical.choice`. -/
abbrev traceWithin (σ : ℝ) (f : Route) : ℕ → ℕ → Outcome (ℕ × Cell) :=
  RouteAnyRing.traceWithin (R := ℝ) σ f

/-- **A result found inside the budget is a trace.**

    (σ : ℝ) (f : Route)                                  [the parameter and a route]
    ∀ {fuel i j w}                                       [every budget, start, end and weight]
    traceWithin σ f fuel i = .found (j, w)               [the bounded search found them]
    → Traces σ f i j w                                   [so the trace relates them]

    So: soundness against the relational trace. The proof is the generic one at `ℝ`,
    `RouteAnyRing.traceWithin_sound`: the budget induction over the ring-generic evaluator, with
    the trip count the evaluator has just tried as the witness. -/
theorem traceWithin_sound (σ : ℝ) (f : Route) :
    ∀ {fuel i j w}, traceWithin σ f fuel i = .found (j, w) → Traces σ f i j w :=
  RouteAnyRing.traceWithin_sound (R := ℝ) σ f

/-- **Every trace is found at some finite budget.**

    (σ : ℝ) (f : Route)                                  [the parameter and a route]
    {i j : ℕ} {w : Cell}                                 [a start, an end and a weight]
    Traces σ f i j w                                     [the trace relates them]
    ∃ fuel, traceWithin σ f fuel i = .found (j, w)       [so some budget finds them]

    So: completeness against the relational trace, and the honest form of it. The budget is an
    output, not an input: the statement says a witness exists, not that it can be computed ahead
    of time. That gap is the halting question, and this file does not claim to close it. The
    proof is the generic one at `ℝ`, `RouteAnyRing.traces_exists_fuel`. -/
theorem traces_exists_fuel (σ : ℝ) (f : Route) {i j : ℕ} {w : Cell}
    (h : Traces σ f i j w) : ∃ fuel, traceWithin σ f fuel i = .found (j, w) :=
  RouteAnyRing.traces_exists_fuel (R := ℝ) σ f h

/-- **The bounded search covers exactly the total trace route.**

    (σ : ℝ) (f : Route) (i j : ℕ) (w : Cell)   [the parameter, a route, a start and an end]
    (∃ fuel, traceWithin σ f fuel i = .found (j, w))     [some budget finds them]
    ↔                                                     [exactly when]
    traceRoute σ f i = some (j, w)                        [the total route returns them]

    So: `traceRoute` is the union over every budget, which is why it needs `Classical.choice` to
    be total. This says the bounded evaluator agrees with it wherever the evaluator answers, and
    that every value of the total route is reached at some finite budget. -/
theorem traceWithin_iff_traceRoute (σ : ℝ) (f : Route) (i j : ℕ) (w : Cell) :
    (∃ fuel, traceWithin σ f fuel i = .found (j, w)) ↔ traceRoute σ f i = some (j, w) := by
  constructor
  · rintro ⟨fuel, h⟩
    exact (tracesTo_traceRoute σ f i j w).mp (traceWithin_sound σ f h)
  · intro h
    exact traces_exists_fuel σ f ((tracesTo_traceRoute σ f i j w).mpr h)

end Combinator
end Apeiron

end
