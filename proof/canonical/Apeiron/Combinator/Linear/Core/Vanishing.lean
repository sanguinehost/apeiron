import Apeiron.Combinator.Linear.Core.Duplication
import Apeiron.Combinator.RouteAnyRing

/-!
# Tracing two loops one at a time is tracing them together

@name Vanishing
@cite joyal-street-verity-1996
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@grade proved

`Combinator/Duplication.lean` proves `I x = x` and `K x y = x` in the feedback setting and says
why they are the only two reachable there: a route applied to an argument has an exit channel and
an argument channel and no third one, so a combinator that feeds one argument's output into
another argument's input has to trace a loop internal to the stored route. Verifying such a
combinator needs Joyal, Street and Verity's vanishing over a pair, at the level of paths rather
than of elements. This file proves it.

## Root

| from | here |
|---|---|
| `mul_assoc`, `one_mul` | `runs_append`, `runsN_split` |

The statement is about wiring and not about the parameter, and `theory/NOVELTY.md`'s A1
spectator test says so plainly: `runs_vanishing` holds at every setting, because a path is a path
whatever the weights on it are. The parameter enters this layer only through
`Combinator/Interaction.lean`'s `weight_blocks_the_loop_iff_disc_nonneg`, which is about whether
a weight can close a loop the wiring leaves open. What the parameter does supply here is the
multiplication that accumulates along a path, and `mul_assoc` is what lets a path be cut in two
and the two weights recombined, which is the whole of `runs_append`.

## The trace, read as a path

`Traces` in `Combinator/Interaction.lean` is written as a straight crossing joined with a sum
over how many times a path goes round the loop. That shape is convenient for computing but hides
the object: a path. `Runs σ f S i j w` says there is a path following `f` from `i` to `j` whose
intermediate positions all lie in `S` and whose end does not, carrying the product of the weights
along it. `RunsN` is the same with the length exposed, which is what the inductions need.

Read that way the trace is `Runs` with `S` the loop positions, and vanishing is a statement about
splitting `S`.

## Vanishing

`runs_vanishing` says that for disjoint `S₁` and `S₂`,

```
   Runs σ f (S₁ ∪ S₂)   =   Runs σ (runsRoute σ f S₂) S₁
```

where `runsRoute σ f S₂` is `f` with its `S₂`-paths collapsed to single steps. Tracing out `S₂`
first and then `S₁` gives the same paths as tracing out both at once. The two directions are
proved separately and by the same disjointness style that carried `traces_functional`:

- `runsN_split` cuts a path through `S₁ ∪ S₂` at its first `S₁` position, which is what makes the
  collapsed route's steps well defined, and the tail is strictly shorter so the recursion is on
  the length;
- `runs_append` glues a collapsed step back onto a path, and `mul_assoc` is what makes the two
  weights recombine.

## What this unlocks, and what is left

`B x y z = x (y z)`, `C x y z = x z y`, `W x y = x y y` and `S x y z = x z (y z)` all need a loop
internal to the stored route, and all four now have the theorem that flattens such a loop into
the outer one. `traces_iff_runs` puts the trace those combinators are stated against inside the
reach of `runs_vanishing`, and `traceRoute_eq_runsRoute` says the two collapses,
`Combinator/Duplication.lean`'s and this file's, are one construction seen twice.

None of the four combinators is built here, and what is left for them is a construction rather
than a theorem. A combinator that composes needs its stored route laid out on three tracks at
once, external, outer loop and inner loop, which is a bijection of the counting numbers with
`ℕ ⊕ (ℕ ⊕ ℕ)` and the routing written against it. The two loop sets are then disjoint, which is
what `runs_vanishing` asks for, and the flattening follows. That layout and that routing are the
next piece of work; nothing further about the trace is needed.

Worth keeping visible: the corpus already has `S`. `Combinator/Model.lean` proves `feed_sPoint`
and `feed_meaning_bracket` in Engeler's graph model over the algebra's idempotents. What this line
of files chases is `S` in the feedback setting, where the algebra's weights ride the wires.

**A bounded path evaluator.** `runsRoute` is total by the same `Classical.choice` as `traceRoute`,
over the unbounded path length in `Runs`. The section at the end adds `runsWithin`, which takes
the step count as an explicit budget and returns a value or an explicit exhaustion, and
`runsWithin_iff_runsRoute` says the two agree wherever the bounded one answers. Membership in the
loop set is decided by an explicit `DecidablePred`, so the budget adds no classical choice.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy
open RouteAnyRing

open scoped Classical

-- ═══════════════════════════════════════════════════════════════════
-- A PATH THROUGH A ROUTE, STAYING INSIDE A SET
-- ═══════════════════════════════════════════════════════════════════

-- The path relation is the generic one, exported at the real carrier. `RouteAnyRing.lean`
-- carries the inductive and its docstring; `open RouteAnyRing` above is what makes the
-- constructors `RunsN.exit` and `RunsN.step` resolve at `ℝ`, and `σ : ℝ` fixes the ring.
export RouteAnyRing (RunsN Runs)

/-- A path is determined by where it starts.

    {σ : ℝ} {f : Route} {S : Set ℕ}                        [the parameter, a route, the inside]
    {n i j₁ : ℕ} {w₁ : Cell}                               [a length, a start, an end, a weight]
    (h₁ : RunsN σ f S n i j₁ w₁)                           [one path from i]
    ∀ {m j₂ w₂}, RunsN σ f S m i j₂ w₂ →                   [and another from i]
      n = m ∧ j₁ = j₂ ∧ w₁ = w₂                            [same length, end and weight]

    So: a route gives one step out of each position, so a path has no choices to make. This is
    `traces_functional` in its natural form. -/
theorem runsN_functional {σ : ℝ} {f : Route} {S : Set ℕ} {n i j₁ : ℕ} {w₁ : Cell}
    (h₁ : RunsN σ f S n i j₁ w₁) :
    ∀ {m j₂ : ℕ} {w₂ : Cell}, RunsN σ f S m i j₂ w₂ → n = m ∧ j₁ = j₂ ∧ w₁ = w₂ := by
  induction h₁ with
  | @exit i j w hf hns =>
    intro m j₂ w₂ h₂
    cases h₂ with
    | exit hf2 hns2 =>
      rw [hf] at hf2
      injection hf2 with hpair
      injection hpair with hj hw
      exact ⟨rfl, hj, hw⟩
    | step hf2 hs2 hr2 =>
      rw [hf] at hf2
      injection hf2 with hpair
      injection hpair with hj hw
      cases hj
      exact absurd hs2 hns
  | @step n i k j w v hf hs hr ih =>
    intro m j₂ w₂ h₂
    cases h₂ with
    | exit hf2 hns2 =>
      rw [hf] at hf2
      injection hf2 with hpair
      injection hpair with hj hw
      cases hj
      exact absurd hs hns2
    | @step m' i' k' j' w' v' hf2 hs2 hr2 =>
      rw [hf] at hf2
      injection hf2 with hpair
      injection hpair with hk hw
      cases hk
      obtain ⟨hn, hj, hv⟩ := ih hr2
      exact ⟨by rw [hn], hj, by rw [hw, hv]⟩

/-- The endpoint and weight of a path are determined by its start.

    {σ : ℝ} {f : Route} {S : Set ℕ} {i j₁ j₂ : ℕ} {w₁ w₂ : Cell}   [two paths from one start]
    (h₁ : Runs σ f S i j₁ w₁) (h₂ : Runs σ f S i j₂ w₂)            [both from i]
    (j₁, w₁) = (j₂, w₂)                                            [so they agree] -/
theorem runs_functional {σ : ℝ} {f : Route} {S : Set ℕ} {i j₁ j₂ : ℕ} {w₁ w₂ : Cell}
    (h₁ : Runs σ f S i j₁ w₁) (h₂ : Runs σ f S i j₂ w₂) : (j₁, w₁) = (j₂, w₂) := by
  obtain ⟨n, h₁⟩ := h₁
  obtain ⟨m, h₂⟩ := h₂
  obtain ⟨-, hj, hw⟩ := runsN_functional h₁ h₂
  rw [hj, hw]

-- ═══════════════════════════════════════════════════════════════════
-- COLLAPSING THE PATHS THROUGH ONE SET INTO SINGLE STEPS
-- ═══════════════════════════════════════════════════════════════════

/-- A route with its paths through a set collapsed to single steps.

@cite girard-1989

    (σ : ℝ) (f : Route) (S : Set ℕ)                [the parameter, a route, the inside]
    Route                                          [one step for each whole path]

    So: this is the trace of `f` over `S`, as a route. `runs_functional` says each position has
    at most one path out of it, so the choice below picks from a subsingleton. -/
def runsRoute (σ : ℝ) (f : Route) (S : Set ℕ) : Route := fun i =>
  if h : ∃ p : ℕ × Cell, Runs σ f S i p.1 p.2 then some (Classical.choose h) else none

/-- The collapsed route's steps are exactly the paths.

    (σ : ℝ) (f : Route) (S : Set ℕ)                [the parameter, a route, the inside]
    (i j : ℕ) (w : Cell)                           [a start, an end and a weight]
    runsRoute σ f S i = some (j, w) ↔ Runs σ f S i j w     [a step is a path] -/
theorem runsRoute_eq (σ : ℝ) (f : Route) (S : Set ℕ) (i j : ℕ) (w : Cell) :
    runsRoute σ f S i = some (j, w) ↔ Runs σ f S i j w := by
  constructor
  · intro h
    by_cases hex : ∃ p : ℕ × Cell, Runs σ f S i p.1 p.2
    · have hsp := Classical.choose_spec hex
      simp only [runsRoute, dif_pos hex, Option.some.injEq] at h
      rw [h] at hsp
      exact hsp
    · simp only [runsRoute, dif_neg hex] at h
      exact absurd h (by simp)
  · intro hr
    have hex : ∃ p : ℕ × Cell, Runs σ f S i p.1 p.2 := ⟨(j, w), hr⟩
    have hsp := Classical.choose_spec hex
    have heq := runs_functional hsp hr
    simp only [runsRoute, dif_pos hex]
    rw [← heq]

-- ═══════════════════════════════════════════════════════════════════
-- CUTTING A PATH, AND GLUING ONE BACK
-- ═══════════════════════════════════════════════════════════════════

/-- A path through the smaller set, glued in front of a path through the union.

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}            [the parameter, a route, two sets]
    ∀ {n i k : ℕ} {w : Cell}                       [a length, two positions, a weight]
    RunsN σ f S₂ n i k w                           [a path inside the second set]
    →                                              [and, if it]
    k ∈ S₁                                         [ends at a position of the first]
    Runs σ f (S₁ ∪ S₂) k j v                       [with a path onward from there]
    Runs σ f (S₁ ∪ S₂) i j (mul σ w v)             [glue to one path, weights multiplied]

    So: the second set sits inside the union, so every position the first path passes through is
    allowed in the second, and the joint at `k` is allowed because `k` lies in the first set.
    `mul_assoc` is what makes the weights recombine. -/
theorem runs_append {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} :
    ∀ {n i k : ℕ} {w : Cell}, RunsN σ f S₂ n i k w → k ∈ S₁ →
      ∀ {j : ℕ} {v : Cell}, Runs σ f (S₁ ∪ S₂) k j v → Runs σ f (S₁ ∪ S₂) i j (mul σ w v) := by
  intro n i k w h
  induction h with
  | @exit i k w hf hns =>
    intro hk j v hv
    obtain ⟨m, hv⟩ := hv
    exact ⟨m + 1, RunsN.step hf (Or.inl hk) hv⟩
  | @step n i k' k w v' hf hs hr ih =>
    intro hk j v hv
    obtain ⟨m, hm⟩ := ih hk hv
    refine ⟨m + 1, ?_⟩
    have := RunsN.step (σ := σ) (S := S₁ ∪ S₂) hf (Or.inr hs) hm
    simp only [mul] at this ⊢
    rwa [← AnyRing.ringMul_assoc] at this

/-- A path through the second set alone is a path through the union, when it ends outside both.

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}            [the parameter, a route, two sets]
    ∀ {n i j : ℕ} {w : Cell}                       [a length, two positions, a weight]
    RunsN σ f S₂ n i j w                           [a path inside the second set]
    →                                              [and, if it]
    j ∉ S₁                                         [ends outside the first]
    RunsN σ f (S₁ ∪ S₂) n i j w                    [it is a path inside the union] -/
theorem runsN_widen {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} :
    ∀ {n i j : ℕ} {w : Cell}, RunsN σ f S₂ n i j w → j ∉ S₁ →
      RunsN σ f (S₁ ∪ S₂) n i j w := by
  intro n i j w h
  induction h with
  | @exit i j w hf hns =>
    intro hj
    exact RunsN.exit hf (fun hc => hc.elim hj hns)
  | @step n i k j w v hf hs hr ih =>
    intro hj
    exact RunsN.step hf (Or.inr hs) (ih hj)

/-- **Cutting a path at its first position of the first set.**

@cite joyal-street-verity-1996

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}                    [the parameter, a route, two sets]
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂)                            [the two sets do not meet]
    {n i j : ℕ} {w : Cell}                                 [a length, two positions, a weight]
    RunsN σ f (S₁ ∪ S₂) n i j w                            [a path through the union]
    (Runs σ f S₂ i j w ∧ j ∉ S₁)                           [is one path inside the second set]
    ∨ (∃ k w₁ w₂ m, Runs σ f S₂ i k w₁ ∧ k ∈ S₁ ∧          [or one such path reaching the first]
      RunsN σ f (S₁ ∪ S₂) m k j w₂ ∧ m < n ∧               [followed by a strictly shorter rest]
      w = mul σ w₁ w₂)                                     [with the weights multiplied]

    So: follow the path until it first lands in the first set. Everything before that lies in the
    second, so it is one step of the collapsed route, and what follows is shorter than what
    started, which is what lets the recursion below terminate. -/
theorem runsN_split {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} (hd : ∀ x, x ∈ S₁ → x ∉ S₂) :
    ∀ {n i j : ℕ} {w : Cell}, RunsN σ f (S₁ ∪ S₂) n i j w →
      (Runs σ f S₂ i j w ∧ j ∉ S₁) ∨
      (∃ (k : ℕ) (w₁ w₂ : Cell) (m : ℕ), Runs σ f S₂ i k w₁ ∧ k ∈ S₁ ∧
        RunsN σ f (S₁ ∪ S₂) m k j w₂ ∧ m < n ∧ w = mul σ w₁ w₂) := by
  intro n i j w h
  induction h with
  | @exit i j w hf hns =>
    exact Or.inl ⟨⟨0, RunsN.exit hf (fun hc => hns (Or.inr hc))⟩, fun hc => hns (Or.inl hc)⟩
  | @step n i k j w v hf hs hr ih =>
    rcases hs with hk | hk
    · exact Or.inr ⟨k, w, v, n, ⟨0, RunsN.exit hf (hd k hk)⟩, hk, hr, Nat.lt_succ_self n, rfl⟩
    · rcases ih with ⟨⟨p, hp⟩, hj⟩ | ⟨k', w₁, w₂, m, ⟨p, hp⟩, hk', hrest, hm, hw⟩
      · exact Or.inl ⟨⟨p + 1, RunsN.step hf hk hp⟩, hj⟩
      · refine Or.inr ⟨k', mul σ w w₁, w₂, m, ⟨p + 1, RunsN.step hf hk hp⟩, hk', hrest,
          Nat.lt_succ_of_lt hm, ?_⟩
        rw [hw, mul_assoc]

-- ═══════════════════════════════════════════════════════════════════
-- VANISHING OVER A PAIR
-- ═══════════════════════════════════════════════════════════════════

/-- Tracing the second set first gives paths of the collapsed route.

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}                    [the parameter, a route, two sets]
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂)                            [the two sets do not meet]
    (n : ℕ) {i j : ℕ} {w : Cell}                           [a length, two positions, a weight]
    RunsN σ f (S₁ ∪ S₂) n i j w                            [a path through the union]
    Runs σ (runsRoute σ f S₂) S₁ i j w                     [is a path of the collapsed route] -/
theorem runs_collapse {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} (hd : ∀ x, x ∈ S₁ → x ∉ S₂) :
    ∀ (n : ℕ) {i j : ℕ} {w : Cell}, RunsN σ f (S₁ ∪ S₂) n i j w →
      Runs σ (runsRoute σ f S₂) S₁ i j w := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro i j w h
    rcases runsN_split hd h with ⟨hr, hj⟩ | ⟨k, w₁, w₂, m, hr, hk, hrest, hm, hw⟩
    · exact ⟨0, RunsN.exit ((runsRoute_eq σ f S₂ i j w).mpr hr) hj⟩
    · obtain ⟨p, hp⟩ := ih m hm hrest
      refine ⟨p + 1, ?_⟩
      rw [hw]
      exact RunsN.step ((runsRoute_eq σ f S₂ i k w₁).mpr hr) hk hp

/-- Paths of the collapsed route expand to paths through the union.

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}            [the parameter, a route, two sets]
    ∀ {n i j : ℕ} {w : Cell}                       [a length, two positions, a weight]
    RunsN σ (runsRoute σ f S₂) S₁ n i j w          [a path of the collapsed route]
    →                                              [then it]
    Runs σ f (S₁ ∪ S₂) i j w                       [is a path through the union] -/
theorem runs_expand {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} :
    ∀ {n i j : ℕ} {w : Cell}, RunsN σ (runsRoute σ f S₂) S₁ n i j w →
      Runs σ f (S₁ ∪ S₂) i j w := by
  intro n i j w h
  induction h with
  | @exit i j w hf hns =>
    obtain ⟨p, hp⟩ := (runsRoute_eq σ f S₂ i j w).mp hf
    exact ⟨p, runsN_widen hp hns⟩
  | @step n i k j w v hf hs hr ih =>
    obtain ⟨p, hp⟩ := (runsRoute_eq σ f S₂ i k w).mp hf
    exact runs_append hp hs ih

/-- **Vanishing over a pair.** Tracing two loops one at a time is tracing them together.

@cite joyal-street-verity-1996
@cite abramsky-haghverdi-scott-2002

    {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ}                    [the parameter, a route, two sets]
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂)                            [the two sets do not meet]
    (i j : ℕ) (w : Cell)                                   [a start, an end and a weight]
    Runs σ f (S₁ ∪ S₂) i j w                               [a path through both loops at once]
    ↔                                                      [exactly when]
    Runs σ (runsRoute σ f S₂) S₁ i j w                     [a path through one, then the other]

    So: Joyal, Street and Verity's `tr^{X⊗Y}(f) = tr^X(tr^Y(f))`, at the level of paths.
    `Trichotomy/Trace.lean`'s `feedback_vanishing_pair` is the same axiom for the algebraic
    feedback of a two-by-two block; this is the form a route needs, and it is what a combinator
    that composes its arguments requires, since such a combinator has to trace a loop internal to
    the route it stores. The parameter is not used: a path is a path whatever the weights on it
    are, and all `mul σ` does is accumulate along it. -/
theorem runs_vanishing {σ : ℝ} {f : Route} {S₁ S₂ : Set ℕ} (hd : ∀ x, x ∈ S₁ → x ∉ S₂)
    (i j : ℕ) (w : Cell) :
    Runs σ f (S₁ ∪ S₂) i j w ↔ Runs σ (runsRoute σ f S₂) S₁ i j w := by
  constructor
  · rintro ⟨n, h⟩
    exact runs_collapse hd n h
  · rintro ⟨n, h⟩
    exact runs_expand h

-- ═══════════════════════════════════════════════════════════════════
-- THE TRACE OF INTERACTION.LEAN IS ONE OF THESE PATHS
-- ═══════════════════════════════════════════════════════════════════

/-- The loop positions: the odd half of the doubled wire.

    Set ℕ                                          [the positions a path may pass through] -/
def oddPos : Set ℕ := {p | ∃ u, p = Equiv.natSumNatEquivNat (Sum.inr u)}

/-- An odd position is a loop position.

    (u : ℕ)                                                [a position]
    Equiv.natSumNatEquivNat (Sum.inr u) ∈ oddPos           [its odd copy is in the loop] -/
theorem mem_oddPos (u : ℕ) : Equiv.natSumNatEquivNat (Sum.inr u) ∈ oddPos := ⟨u, rfl⟩

/-- An even position is not a loop position.

    (k : ℕ)                                                [a position]
    Equiv.natSumNatEquivNat (Sum.inl k) ∉ oddPos           [its even copy is outside the loop] -/
theorem not_mem_oddPos (k : ℕ) : Equiv.natSumNatEquivNat (Sum.inl k) ∉ oddPos := by
  rintro ⟨u, hu⟩
  have hc := Equiv.natSumNatEquivNat.injective hu
  exact absurd hc (by simp)

/-- Crossing straight over is one even-to-even step.

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    direct σ f i = some (j, w)                             [the straight crossing lands there]
    ↔                                                      [exactly when]
    f (Equiv.natSumNatEquivNat (Sum.inl i))                [the route takes the even copy of i]
      = some (Equiv.natSumNatEquivNat (Sum.inl j), w)      [to the even copy of j] -/
theorem direct_iff (σ : ℝ) (f : Route) (i j : ℕ) (w : Cell) :
    direct σ f i = some (j, w) ↔
      f (Equiv.natSumNatEquivNat (Sum.inl i))
        = some (Equiv.natSumNatEquivNat (Sum.inl j), w) := by
  rw [direct_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inl i)) with
  | none => simp
  | some p =>
    obtain ⟨p1, p2⟩ := p
    cases hs : Equiv.natSumNatEquivNat.symm p1 with
    | inl k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inl k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst hp
      simp only [Option.bind_some, outL_inl, Option.map_some, Trichotomy.one_mul,
        mul_one_right]
      simp
    | inr k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inr k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst hp
      simp only [Option.bind_some, outL_inr, Option.map_none]
      simp
      omega

/-- Entering the loop is one even-to-odd step.

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    (i u : ℕ) (w : Cell)                           [two positions and a weight]
    entry σ f i = some (u, w)                              [the way in lands there]
    ↔                                                      [exactly when]
    f (Equiv.natSumNatEquivNat (Sum.inl i))                [the route takes the even copy of i]
      = some (Equiv.natSumNatEquivNat (Sum.inr u), w)      [to the odd copy of u] -/
theorem entry_iff (σ : ℝ) (f : Route) (i u : ℕ) (w : Cell) :
    entry σ f i = some (u, w) ↔
      f (Equiv.natSumNatEquivNat (Sum.inl i))
        = some (Equiv.natSumNatEquivNat (Sum.inr u), w) := by
  rw [entry_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inl i)) with
  | none => simp
  | some p =>
    obtain ⟨p1, p2⟩ := p
    cases hs : Equiv.natSumNatEquivNat.symm p1 with
    | inl k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inl k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst hp
      simp only [Option.bind_some, outR_inl, Option.map_none]
      simp
      omega
    | inr k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inr k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst hp
      simp only [Option.bind_some, outR_inr, Option.map_some, Trichotomy.one_mul,
        mul_one_right]
      simp

/-- Leaving the loop is one odd-to-even step.

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    (u j : ℕ) (w : Cell)                           [two positions and a weight]
    leave σ f u = some (j, w)                              [the way out lands there]
    ↔                                                      [exactly when]
    f (Equiv.natSumNatEquivNat (Sum.inr u))                [the route takes the odd copy of u]
      = some (Equiv.natSumNatEquivNat (Sum.inl j), w)      [to the even copy of j] -/
theorem leave_iff (σ : ℝ) (f : Route) (u j : ℕ) (w : Cell) :
    leave σ f u = some (j, w) ↔
      f (Equiv.natSumNatEquivNat (Sum.inr u))
        = some (Equiv.natSumNatEquivNat (Sum.inl j), w) := by
  rw [leave_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inr u)) with
  | none => simp
  | some p =>
    obtain ⟨p1, p2⟩ := p
    cases hs : Equiv.natSumNatEquivNat.symm p1 with
    | inl k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inl k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst hp
      simp only [Option.bind_some, outL_inl, Option.map_some, Trichotomy.one_mul,
        mul_one_right]
      simp
    | inr k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inr k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst hp
      simp only [Option.bind_some, outL_inr, Option.map_none]
      simp
      omega

/-- Going once round the loop is one odd-to-odd step.

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    (u v : ℕ) (w : Cell)                           [two positions and a weight]
    body σ f u = some (v, w)                               [one trip lands there]
    ↔                                                      [exactly when]
    f (Equiv.natSumNatEquivNat (Sum.inr u))                [the route takes the odd copy of u]
      = some (Equiv.natSumNatEquivNat (Sum.inr v), w)      [to the odd copy of v] -/
theorem body_iff (σ : ℝ) (f : Route) (u v : ℕ) (w : Cell) :
    body σ f u = some (v, w) ↔
      f (Equiv.natSumNatEquivNat (Sum.inr u))
        = some (Equiv.natSumNatEquivNat (Sum.inr v), w) := by
  rw [body_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inr u)) with
  | none => simp
  | some p =>
    obtain ⟨p1, p2⟩ := p
    cases hs : Equiv.natSumNatEquivNat.symm p1 with
    | inl k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inl k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst hp
      simp only [Option.bind_some, outR_inl, Option.map_none]
      simp
      omega
    | inr k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inr k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst hp
      simp only [Option.bind_some, outR_inr, Option.map_some, Trichotomy.one_mul,
        mul_one_right]
      simp

/-- Going round the loop n times and leaving is a path of length n.

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    ∀ (n u j : ℕ) (w : Cell)                       [a count, two positions and a weight]
    rcomp σ (rpow σ (body σ f) n) (leave σ f) u = some (j, w)   [n trips and out]
    ↔                                              [exactly when]
    RunsN σ f oddPos n (odd copy of u) (even copy of j) w       [a path of that length] -/
theorem loop_iff (σ : ℝ) (f : Route) : ∀ (n u j : ℕ) (w : Cell),
    rcomp σ (rpow σ (body σ f) n) (leave σ f) u = some (j, w) ↔
      RunsN σ f oddPos n (Equiv.natSumNatEquivNat (Sum.inr u))
        (Equiv.natSumNatEquivNat (Sum.inl j)) w := by
  intro n
  induction n with
  | zero =>
    intro u j w
    rw [rpow, rcomp_rid_left]
    constructor
    · intro h
      exact RunsN.exit ((leave_iff σ f u j w).mp h) (not_mem_oddPos j)
    · intro h
      cases h with
      | exit hf hns => exact (leave_iff σ f u j w).mpr hf
  | succ n ih =>
    intro u j w
    rw [rpow, rcomp_assoc]
    constructor
    · intro h
      cases hb : body σ f u with
      | none => rw [rcomp, hb] at h; simp at h
      | some q =>
        rw [rcomp, hb] at h
        simp only [Option.bind_some] at h
        cases hr : rcomp σ (rpow σ (body σ f) n) (leave σ f) q.1 with
        | none => rw [hr] at h; simp at h
        | some r =>
          rw [hr] at h
          simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨hj, hw⟩ := h
          subst hj
          subst hw
          exact RunsN.step ((body_iff σ f u q.1 q.2).mp hb) (mem_oddPos q.1)
            ((ih q.1 r.1 r.2).mp hr)
    · intro h
      cases h with
      | step hf hs hr =>
        obtain ⟨u2, hu2⟩ := hs
        subst hu2
        rw [rcomp, (body_iff σ f u u2 _).mpr hf]
        simp [(ih u2 j _).mpr hr]

/-- **The trace is a path.**

@cite girard-1989
@cite joyal-street-verity-1996

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ f i j w                               [the trace relates them]
    ↔                                              [exactly when]
    Runs σ f oddPos (even copy of i) (even copy of j) w    [a path through the loop positions]

    So: `Combinator/Interaction.lean` writes the trace as a straight crossing joined with a sum
    over how many times the loop is gone round, which is convenient for computing and hides the
    object. This says the two descriptions are the same, so `runs_vanishing` applies to the
    trace: `direct_iff`, `entry_iff`, `leave_iff` and `body_iff` are the four one-step
    translations it rests on. -/
theorem traces_iff_runs (σ : ℝ) (f : Route) (i j : ℕ) (w : Cell) :
    Traces σ f i j w ↔
      Runs σ f oddPos (Equiv.natSumNatEquivNat (Sum.inl i))
        (Equiv.natSumNatEquivNat (Sum.inl j)) w := by
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · exact ⟨0, RunsN.exit ((direct_iff σ f i j w).mp hd) (not_mem_oddPos j)⟩
    · cases he : entry σ f i with
      | none => rw [rcomp, he] at hn; simp at hn
      | some q =>
        rw [rcomp, he] at hn
        simp only [Option.bind_some] at hn
        cases hr : rcomp σ (rpow σ (body σ f) n) (leave σ f) q.1 with
        | none => rw [hr] at hn; simp at hn
        | some r =>
          rw [hr] at hn
          simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hn
          obtain ⟨hj, hw⟩ := hn
          subst hj
          subst hw
          exact ⟨n + 1, RunsN.step ((entry_iff σ f i q.1 q.2).mp he) (mem_oddPos q.1)
            ((loop_iff σ f n q.1 r.1 r.2).mp hr)⟩
  · rintro ⟨n, h⟩
    cases n with
    | zero =>
      cases h with
      | exit hf hns => exact Or.inl ((direct_iff σ f i j w).mpr hf)
    | succ n =>
      cases h with
      | step hf hs hr =>
        obtain ⟨u, hu⟩ := hs
        subst hu
        refine Or.inr ⟨n, ?_⟩
        rw [rcomp, (entry_iff σ f i u _).mpr hf]
        simp [(loop_iff σ f n u j _).mpr hr]

/-- **The two collapses are one.**

@cite girard-1989
@cite joyal-street-verity-1996

    (σ : ℝ) (g : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    traceRoute σ g i = some (j, w)                 [the trace, taken as a route]
    ↔                                              [exactly when]
    runsRoute σ g oddPos (even copy of i)          [the path collapse, at the even copies]
      = some (even copy of j, w)

    So: `Combinator/Duplication.lean`'s `traceRoute`, which makes application total, and this
    file's `runsRoute`, which collapses the paths through a set, are the same construction seen
    twice. With `runs_vanishing` that is what lets a combinator's stored route be a trace and
    still be reasoned about: the inner loop and the outer one may be traced together. -/
theorem traceRoute_eq_runsRoute (σ : ℝ) (g : Route) (i j : ℕ) (w : Cell) :
    traceRoute σ g i = some (j, w) ↔
      runsRoute σ g oddPos (Equiv.natSumNatEquivNat (Sum.inl i))
        = some (Equiv.natSumNatEquivNat (Sum.inl j), w) := by
  rw [runsRoute_eq, ← traces_iff_runs]
  exact (tracesTo_traceRoute σ g i j w).symm

-- ═══════════════════════════════════════════════════════════════════
-- A FINITE BUDGET FOR A PATH
-- ═══════════════════════════════════════════════════════════════════

-- The bounded path evaluator and its one-step equation are the generic ones, exported at the
-- real carrier. `RouteAnyRing.lean` carries the authoritative definitions and docstrings: the
-- evaluator spends one unit on each step landing inside `S`, decides membership through the
-- explicit `DecidablePred`, and reports `.exhausted` rather than claiming divergence. The ring
-- is fixed to `ℝ` by the parameter `σ` at every use.
export RouteAnyRing (runsWithin runsWithin_succ)

/-- **A result found inside the budget is a path.**

    (σ : ℝ) (f : Route) (S : Set ℕ) [DecidablePred S]    [the parameter, a route, the inside,
      and a decision that membership is computable]
    ∀ {fuel i j w}                                       [every budget, start, end and weight]
    runsWithin σ f S fuel i = .found (j, w)              [the bounded search found them]
    → Runs σ f S i j w                                   [so a path relates them]

    So: soundness against the relational path. The proof is the generic one at `ℝ`,
    `RouteAnyRing.runsWithin_sound`: the budget induction over the ring-generic evaluator. -/
theorem runsWithin_sound (σ : ℝ) (f : Route) (S : Set ℕ) [DecidablePred S] :
    ∀ {fuel i j w}, runsWithin σ f S fuel i = .found (j, w) → Runs σ f S i j w :=
  RouteAnyRing.runsWithin_sound (R := ℝ) σ f S

/-- **Every path is found at some finite budget.**

    (σ : ℝ) (f : Route) (S : Set ℕ) [DecidablePred S]    [the parameter, a route, the inside,
      and a decision that membership is computable]
    ∀ {n i j w}                                          [every length, start, end and weight]
    RunsN σ f S n i j w                                  [a path of that length relates them]
    → runsWithin σ f S (n + 1) i = .found (j, w)   [so one past the length finds them]

    So: completeness against the relational path. The budget is the length plus one, because the
    path spends one step leaving the set after its inside steps. The proof is the generic one at
    `ℝ`, `RouteAnyRing.runsWithin_complete`. -/
theorem runsWithin_complete (σ : ℝ) (f : Route) (S : Set ℕ) [DecidablePred S] :
    ∀ {n i j w}, RunsN σ f S n i j w → runsWithin σ f S (n + 1) i = .found (j, w) :=
  RouteAnyRing.runsWithin_complete (R := ℝ) σ f S

/-- **The bounded search covers exactly the collapsed route.**

    (σ : ℝ) (f : Route) (S : Set ℕ) [DecidablePred S]    [the parameter, a route, the inside,
      and a decision that membership is computable]
    (i j : ℕ) (w : Cell)                                  [a start, an end and a weight]
    (∃ fuel, runsWithin σ f S fuel i = .found (j, w))     [some budget finds them]
    ↔                                                     [exactly when]
    runsRoute σ f S i = some (j, w)                       [the collapsed route returns them]

    So: `runsRoute` is the union over every budget, which is why it needs `Classical.choice` to
    be total. This says the bounded evaluator agrees with it wherever the evaluator answers, and
    that every value of the collapsed route is reached at some finite budget. -/
theorem runsWithin_iff_runsRoute (σ : ℝ) (f : Route) (S : Set ℕ) [DecidablePred S]
    (i j : ℕ) (w : Cell) :
    (∃ fuel, runsWithin σ f S fuel i = .found (j, w)) ↔ runsRoute σ f S i = some (j, w) := by
  constructor
  · rintro ⟨fuel, h⟩
    exact (runsRoute_eq σ f S i j w).mpr (runsWithin_sound σ f S h)
  · intro h
    obtain ⟨n, hn⟩ := (runsRoute_eq σ f S i j w).mp h
    exact ⟨n + 1, runsWithin_complete σ f S hn⟩

end Combinator
end Apeiron

end
