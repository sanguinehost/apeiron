import Apeiron.Combinator.Linear.Basis

/-!
# What an application costs, and what vanishing does to the bill

@name Cost
@cite joyal-street-verity-1996
@cite abramsky-haghverdi-scott-2002
@cite girard-1989
@cite haghverdi-scott-2010
@grade proved

`Combinator/Interaction.lean` writes the trace as a straight crossing joined with a sum over how
many times the path goes round the feedback loop. That count is a cost: it is the number of trips
the machine makes before it answers. This file makes it a definition, `AppTrips`, proves it is
determined by where the path starts, and settles what an abstraction on this carrier costs.

## Root

| from | here |
|---|---|
| `exit_depth_unique` | `appTrips_unique` |
| `rcomp_rnone_left`, `rcomp_rnone_right` | `appTrips_zero_of_body_rnone` |
| `store_body`, `k_body`, `i_body`, `d_body`, `b_body` | the five zero-cost combinators |
| `tracesTo_of_body_rnone`, `tracesTo_unique` | `trace_eq_of_body_rnone` |
| `runs_collapse`, `runs_append`, `runsN_widen` | `staged_never_exceeds_raw` |
| `runs_collapse`, `runs_append`, `runsN_widen` | `staged_path_expands_no_shorter` |
| `body_arg`, `bStored_ae`, `xPos_arg` | `bStored_loops` |

## Zero cost is a relabelling

`appTrips_zero_of_body_rnone` says that a route whose loop body connects nothing is answered
without going round at all. `trace_eq_of_body_rnone` says the same thing in closed form: the trace
is then `direct` joined with one pass in and out, a two-term expression with no series in it.
Every combinator built by `storeRoute` is of that kind, and so are the discarding, identity,
dereliction and composing routings at their first application. A relabelling does cost
nothing, and the statement is not an observation about the definitions but a consequence of the
loop body being empty.

That also settles where the parameter can enter. A trace with no series in it is a finite
expression at every σ, so σ cannot decide whether a zero-cost application terminates. The moment
the loop body is inhabited the trace is a geometric series in the loop weight, and then
`loop_solvable_iff_off_cone` and `loop_always_solvable_iff_disc_neg` are what decide whether it
closes: every loop weight has a resolvent exactly below zero, and
`weight_blocks_the_loop_iff_disc_nonneg` exhibits one that neither dies out nor closes at zero and
above. So termination and the parameter are on one axis after all, but only for the routes that
loop at all.

## Zero overhead, and why not less than zero

@cite joyal-street-verity-1996

Two questions hide under "what does the abstraction cost", and they have different answers.

**Do you pay for the abstraction?** No, and that is a theorem rather than an observation.
`storeRoute_costs_nothing` says an application of a storing combinator makes no trip round the
loop, and `rapp_storeRoute` says what it does instead: it relabels. The combinator is applied
through the general mechanism, the trace, and contributes nothing to the count that mechanism
charges for. Seven of the nine are of that kind, and the eighth, the composing one, is free at its
first application and stops being free at its second, which `bStored_loops` pins to a position.
This is the zero-overhead principle in the form it is usually stated, that you do not pay for what
you do not use, and here the thing not used is the loop.

**Do you get back more than you paid, so the composed form beats the hand-rolled one?** Not here,
and the reason is structural rather than a failure to find the witness. The mechanism that would
have to supply it is vanishing, `runs_vanishing`, and vanishing is an equivalence of path sets:
`runs_collapse` and `runs_expand` go both ways, so tracing one loop and then the other visits the
same positions in the same order with the same weights as tracing them together. Nothing becomes
known by composing that was not known before composing, and a saving of the kind a compiler gets
from inlining through an abstraction boundary comes from information the un-abstracted
form did not have. There is no such information on this carrier, so vanishing cannot be the source
of one.

`staged_never_exceeds_raw` does say the staged count is at most the raw count, and it is tempting
to read that as the composed form being cheaper. It is not, and `staged_path_expands_no_shorter`
is why: one step of the collapsed route is a whole path of the inner trace, so the two counts are
in different units and the staged one is smaller only because it stops counting. The pair is kept
because together they are what makes the units explicit, and a cost claim that does not fix its
unit is not a cost claim.

**Where a statement of that kind would live, if there is one.** It needs two machines carrying the
same measure, computing the same route, one of them strictly cheaper. The corpus already has such
pairs and does not yet measure them. `bRoute_bracket` gives `B x y z` and `x (y z)` as one route
reached by two machines. `Combinator/Fusing.lean` compares `plug σ (bStored x) y` against
`bStore σ x y`, two machines whose paths agree and which differ in where a single step falls, one
fusing at the start and paying it back at the end. What is on offer at that pair is zero overhead
again, to within one step, and not less than zero. Recorded so the next attempt starts at the
measure rather than at the hope.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy
open RouteAnyRing

-- ═══════════════════════════════════════════════════════════════════
-- THE COST OF AN APPLICATION
-- ═══════════════════════════════════════════════════════════════════

/-- **The cost of applying one route to another**: how many trips round the loop it takes.

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (a b : Route)                          [the parameter and two routes]
    (i n : ℕ)                                      [a starting position and a count]
    Prop                                           [the path from i goes round n times]

    So: this is the second disjunct of `Traces` with the count exposed. `Traces` says a path
    either crosses straight over, which is cost nothing at all, or enters the loop, goes round
    some number of times and leaves. That number is the length of the path the trace runs, and
    naming it is what lets one application be compared with another. -/
def AppTrips (σ : ℝ) (a b : Route) (i n : ℕ) : Prop :=
  ∃ (j : ℕ) (w : Cell),
    rcomp σ (entry σ (plug σ a b))
      (rcomp σ (rpow σ (body σ (plug σ a b)) n) (leave σ (plug σ a b))) i = some (j, w)

/-- **An application has one cost.** The trip count is determined by where the path starts.

    (σ : ℝ) (a b : Route) (i : ℕ)                  [the parameter, two routes, a position]
    {n m : ℕ}                                      [two counts]
    (hn : AppTrips σ a b i n)                      [the path goes round n times]
    (hm : AppTrips σ a b i m)                      [and round m times]
    n = m                                          [so the two counts agree]

    So: `exit_depth_unique` says a path leaves the loop at one depth only, and the entry step is a
    partial map, so the position the loop is entered at is fixed by `i`. -/
theorem appTrips_unique (σ : ℝ) (a b : Route) (i : ℕ) {n m : ℕ}
    (hn : AppTrips σ a b i n) (hm : AppTrips σ a b i m) : n = m := by
  obtain ⟨j₁, w₁, h₁⟩ := hn
  obtain ⟨j₂, w₂, h₂⟩ := hm
  rw [rcomp] at h₁ h₂
  cases he : entry σ (plug σ a b) i with
  | none => rw [he] at h₁; simp at h₁
  | some p =>
    rw [he] at h₁ h₂
    simp only [Option.bind_some] at h₁ h₂
    refine exit_depth_unique σ (plug σ a b) p.1 ?_ ?_
    · intro hc; rw [hc] at h₁; simp at h₁
    · intro hc; rw [hc] at h₂; simp at h₂

/-- **A route with an empty loop is answered without going round it.**

    (σ : ℝ) (a b : Route)                          [the parameter and two routes]
    (hb : ∀ u, body σ (plug σ a b) u = none)       [the loop body connects nothing]
    {i n : ℕ}                                      [a position and a count]
    (h : AppTrips σ a b i n)                       [the path goes round n times]
    n = 0                                          [then it does not go round at all]

    So: one trip and no more. This is the zero-cost statement, and it is a consequence of the
    wiring rather than of the weights, so it holds at every setting of the parameter. -/
theorem appTrips_zero_of_body_rnone (σ : ℝ) (a b : Route)
    (hb : ∀ u, body σ (plug σ a b) u = none) {i n : ℕ} (h : AppTrips σ a b i n) : n = 0 := by
  cases n with
  | zero => rfl
  | succ m =>
    exfalso
    obtain ⟨j, w, hj⟩ := h
    have hbn : body σ (plug σ a b) = rnone := funext hb
    rw [hbn, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hj
    simp [rnone] at hj

/-- **The trace of a route with an empty loop, in closed form.**

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    (hb : body σ f = rnone)                        [the loop body connects nothing]
    traceRoute σ f                                 [the trace]
    = rplus (direct σ f) (rcomp σ (entry σ f) (leave σ f))   [is two terms and no series]

    So: no geometric series is summed, so no convergence question arises and the parameter cannot
    enter through the loop. This is what "costs nothing" means as an equation rather than as a
    count. -/
theorem trace_eq_of_body_rnone (σ : ℝ) (f : Route) (hb : body σ f = rnone) :
    traceRoute σ f = rplus (direct σ f) (rcomp σ (entry σ f) (leave σ f)) :=
  tracesTo_unique (tracesTo_traceRoute σ f) (tracesTo_of_body_rnone σ f hb)

-- ═══════════════════════════════════════════════════════════════════
-- THE COMBINATORS THAT COST NOTHING
-- ═══════════════════════════════════════════════════════════════════

/-- **A storing routing costs nothing.** Applying it can only relabel.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route)                [the parameter, a relabelling, a route]
    {i n : ℕ}                                      [a position and a count]
    (h : AppTrips σ (storeRoute e) x i n)          [the path goes round n times]
    n = 0                                          [then it does not go round at all]

    So: with `rapp_storeRoute`, which says the result is the argument read through the
    relabelling, this is a zero-cost abstraction as a theorem. The identity, the exchanging
    combinator, functoriality, the contracting combinator and the comultiplication are all of this
    form, so all five cost nothing. -/
theorem storeRoute_costs_nothing (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) {i n : ℕ}
    (h : AppTrips σ (storeRoute e) x i n) : n = 0 :=
  appTrips_zero_of_body_rnone σ _ _ (store_body σ e x) h

/-- **The discarding routing costs nothing.**

    (σ : ℝ) (x : Route) {i n : ℕ}                  [the parameter, a route, a position, a count]
    (h : AppTrips σ kRoute x i n)                  [the path goes round n times]
    n = 0                                          [then it does not go round at all] -/
theorem kRoute_costs_nothing (σ : ℝ) (x : Route) {i n : ℕ}
    (h : AppTrips σ kRoute x i n) : n = 0 :=
  appTrips_zero_of_body_rnone σ _ _ (k_body σ x) h

/-- **The identity routing costs nothing.**

    (σ : ℝ) (x : Route) {i n : ℕ}                  [the parameter, a route, a position, a count]
    (h : AppTrips σ iRoute x i n)                  [the path goes round n times]
    n = 0                                          [then it does not go round at all] -/
theorem iRoute_costs_nothing (σ : ℝ) (x : Route) {i n : ℕ}
    (h : AppTrips σ iRoute x i n) : n = 0 :=
  appTrips_zero_of_body_rnone σ _ _ (i_body σ x) h

/-- **The dereliction costs nothing.**

    (σ : ℝ) (x : Route) {i n : ℕ}                  [the parameter, a route, a position, a count]
    (h : AppTrips σ dRoute (bang x) i n)           [the path goes round n times]
    n = 0                                          [then it does not go round at all] -/
theorem dRoute_costs_nothing (σ : ℝ) (x : Route) {i n : ℕ}
    (h : AppTrips σ dRoute (bang x) i n) : n = 0 :=
  appTrips_zero_of_body_rnone σ _ _ (d_body σ x) h

/-- **The composing routing costs nothing at its first application.**

    (σ : ℝ) (x : Route) {i n : ℕ}                  [the parameter, a route, a position, a count]
    (h : AppTrips σ bRoute x i n)                  [the path goes round n times]
    n = 0                                          [then it does not go round at all]

    So: storing the first route is free. What is not free is the second application, where the
    loop re-enters the stored route, and `bStored_loops` below exhibits a position at which that
    loop is inhabited. That is the exact place where the composing combinator starts to cost. -/
theorem bRoute_costs_nothing (σ : ℝ) (x : Route) {i n : ℕ}
    (h : AppTrips σ bRoute x i n) : n = 0 :=
  appTrips_zero_of_body_rnone σ _ _ (b_body σ x) h

-- ═══════════════════════════════════════════════════════════════════
-- VANISHING, AND WHAT IT DOES NOT SAVE
-- ═══════════════════════════════════════════════════════════════════

/-- Collapsing the inner loop first, with the path length carried along.

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}                    [the parameter, a route, two sets]
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂)                            [the two sets do not meet]
    ∀ (n : ℕ) {i j : ℕ} {w : Cell}                         [a length, two positions, a weight]
    RunsN σ f (S₁ ∪ S₂) n i j w                            [a path through the union]
    →                                                      [if the left holds so does the right]
    ∃ p, p ≤ n ∧ RunsN σ (runsRoute σ f S₂) S₁ p i j w     [is a no longer staged path]

    So: `runs_collapse` with the bound kept. Each step of the staged path consumes at least one
    step of the raw path, which is what the strong induction on `runsN_split` already
    produced. -/
theorem runs_collapse_le {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} (hd : ∀ x, x ∈ S₁ → x ∉ S₂) :
    ∀ (n : ℕ) {i j : ℕ} {w : Cell}, RunsN σ f (S₁ ∪ S₂) n i j w →
      ∃ p, p ≤ n ∧ RunsN σ (runsRoute σ f S₂) S₁ p i j w := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro i j w h
    rcases runsN_split hd h with ⟨hr, hj⟩ | ⟨k, w₁, w₂, m, hr, hk, hrest, hm, hw⟩
    · exact ⟨0, Nat.zero_le _, RunsN.exit ((runsRoute_eq σ f S₂ i j w).mpr hr) hj⟩
    · obtain ⟨p, hple, hp⟩ := ih m hm hrest
      refine ⟨p + 1, ?_, ?_⟩
      · exact Nat.succ_le_of_lt (Nat.lt_of_le_of_lt hple hm)
      · rw [hw]
        exact RunsN.step ((runsRoute_eq σ f S₂ i k w₁).mpr hr) hk hp

/-- Gluing an inner path in front of an outer one, with the lengths added.

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}                    [the parameter, a route, two sets]
    ∀ {n i k : ℕ} {w : Cell}                               [a length, two positions, a weight]
    RunsN σ f S₂ n i k w                                   [a path inside the second set]
    →                                                      [if the left holds so does the right]
    k ∈ S₁                                                 [ending at a position of the first]
    RunsN σ f (S₁ ∪ S₂) m k j v                            [with a path onward from there]
    RunsN σ f (S₁ ∪ S₂) (n + 1 + m) i j (mul σ w v)        [glue, and the lengths add]

    So: `runs_append` with the arithmetic kept. This is the bookkeeping that shows no step is
    dropped when the two loops are traced together. -/
theorem runsN_append {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} :
    ∀ {n i k : ℕ} {w : Cell}, RunsN σ f S₂ n i k w → k ∈ S₁ →
      ∀ {m j : ℕ} {v : Cell}, RunsN σ f (S₁ ∪ S₂) m k j v →
        RunsN σ f (S₁ ∪ S₂) (n + 1 + m) i j (mul σ w v) := by
  intro n i k w h
  induction h with
  | @exit i k w hf hns =>
    intro hk m j v hv
    have h0 : 0 + 1 + m = m + 1 := by omega
    rw [h0]
    exact RunsN.step hf (Or.inl hk) hv
  | @step n i k' k w v' hf hs hr ih =>
    intro hk m j v hv
    have hstep := RunsN.step (σ := σ) (S := S₁ ∪ S₂) hf (Or.inr hs) (ih hk hv)
    simp only [mul] at hstep ⊢
    rw [← AnyRing.ringMul_assoc] at hstep
    have harith : n + 1 + m + 1 = n + 1 + 1 + m := by omega
    rwa [harith] at hstep

/-- **Every staged step is real work.** A path of the collapsed route expands to one at least as
    long.

@cite joyal-street-verity-1996

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}                    [the parameter, a route, two sets]
    ∀ {p i j : ℕ} {w : Cell}                               [a length, two positions, a weight]
    RunsN σ (runsRoute σ f S₂) S₁ p i j w                  [a path of the collapsed route]
    →                                                      [if the left holds so does the right]
    ∃ n, p ≤ n ∧ RunsN σ f (S₁ ∪ S₂) n i j w               [expands to one no shorter]

    So: this is the theorem that fixes the unit. Each step of the staged path is a whole path of
    the inner trace, and `runsN_append` puts those steps back, so the raw path is never shorter
    than the staged one. A path that vanishes from the outer count has not been avoided; it has
    been moved into the inner one, and any reading of `staged_never_exceeds_raw` as a saving has
    to answer this first. -/
theorem staged_path_expands_no_shorter {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} :
    ∀ {p i j : ℕ} {w : Cell}, RunsN σ (runsRoute σ f S₂) S₁ p i j w →
      ∃ n, p ≤ n ∧ RunsN σ f (S₁ ∪ S₂) n i j w := by
  intro p i j w h
  induction h with
  | @exit i j w hf hns =>
    obtain ⟨n, hn⟩ := (runsRoute_eq σ f S₂ i j w).mp hf
    exact ⟨n, Nat.zero_le _, runsN_widen hn hns⟩
  | @step p i k j w v hf hs _ ih =>
    obtain ⟨n₁, hn₁⟩ := (runsRoute_eq σ f S₂ i k w).mp hf
    obtain ⟨m, hple, hm⟩ := ih
    exact ⟨n₁ + 1 + m, by omega, runsN_append hn₁ hs hm⟩

/-- **The staged count never exceeds the raw count.**

@cite joyal-street-verity-1996

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}                    [the parameter, a route, two sets]
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂)                            [the two sets do not meet]
    {p n i j : ℕ} {w : Cell}                               [two lengths, two positions, a weight]
    (hstaged : RunsN σ (runsRoute σ f S₂) S₁ p i j w)      [the staged path]
    (hraw : RunsN σ f (S₁ ∪ S₂) n i j w)                   [and the raw one]
    p ≤ n                                                  [the staged one is no longer]

    So: read alone this looks like the composed form being cheaper than the flat one, and it is
    not. `staged_path_expands_no_shorter` is the other half: the staged count is smaller because
    it does not count the inner steps, and those steps are still taken. The two counts are in
    different units, which is the whole content of the pair. -/
theorem staged_never_exceeds_raw {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂) {p n i j : ℕ} {w : Cell}
    (hstaged : RunsN σ (runsRoute σ f S₂) S₁ p i j w)
    (hraw : RunsN σ f (S₁ ∪ S₂) n i j w) : p ≤ n := by
  obtain ⟨p', hp'le, hp'⟩ := runs_collapse_le hd n hraw
  obtain ⟨heq, -, -⟩ := runsN_functional hstaged hp'
  omega

-- ═══════════════════════════════════════════════════════════════════
-- WHERE THE COMPOSING COMBINATOR STARTS TO COST
-- ═══════════════════════════════════════════════════════════════════

/-- **The composing routing's second application does loop.**

    (σ : ℝ) (u : ℕ)                                            [the parameter and a position]
    body σ (plug σ (bStored rid) rid) (extPos u)               [the loop body of B's composite]
    = some (extPos u, one)                                     [connects a position to itself]

    So: the stored route sends the asking track back onto its own tracks, and the argument sends
    it back again, so the loop body is inhabited and `appTrips_zero_of_body_rnone` does not apply.
    This is the exact place where the composing combinator stops being a relabelling: storing is
    free, `bRoute_costs_nothing`, and composing is not.

    The witness is a fixed point of weight `one`, so by `body_leave_disjoint` the path from that
    position never leaves the loop. That is a concrete divergence on this carrier and it lives
    where `rapp_rnone` said it would, in the route being absent at a position rather than in the
    application failing to exist. -/
theorem bStored_loops (σ : ℝ) (u : ℕ) :
    body σ (plug σ (bStored rid) rid) (extPos u) = some (extPos u, one) := by
  rw [body_arg, plug, rcomp, bStored_ae]
  simp only [rid_apply, Option.map_some, xPos_arg, outPos_eq, Option.bind_some, rtensor_arg,
    outR_arg, one_mul]

/-- **So the zero-cost theorem does not reach the composition.**

    (σ : ℝ)                                                    [the parameter]
    body σ (plug σ (bStored rid) rid) ≠ rnone                  [that loop body is not empty] -/
theorem bStored_body_ne_rnone (σ : ℝ) : body σ (plug σ (bStored rid) rid) ≠ rnone := by
  intro h
  have := congrFun h (extPos 0)
  rw [bStored_loops σ 0] at this
  simp [rnone] at this

end Combinator
end Apeiron

end
