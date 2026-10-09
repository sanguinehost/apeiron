import Apeiron.Combinator.RouteAnyRing
import Apeiron.Combinator.RelationalPaths

/-!
# Budget-certified route materialization

The caller supplies an input-indexed budget and a proof that it is sufficient for every related
output.  Materialization then remains executable while its relation is exact pointwise.
-/

namespace Apeiron
namespace Combinator
namespace BoundedRoutes

open Apeiron.Trichotomy.AnyRing
open RouteAnyRing

universe u

variable {R : Type u} [CommRing R]

/-- An explicit input-indexed bound sufficient for every trace output.

    Route R          [a weighted wire map over the ring R]
    Traces           [the union over every path length]
    traceWithin      [the bounded search for the trace]
    Outcome.found    [a value found before the budget ran out]
    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    ∀                [for every single one]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on] -/
structure TraceBudget (σ : R) (f : Route R) where
  fuel : ℕ → ℕ
  complete : ∀ {i j w}, Traces σ f i j w →
    traceWithin σ f (fuel i) i = .found (j, w)

/-- Materialize a route by evaluating the supplied trace budget at each input.

    TraceBudget      [an explicit input-indexed bound sufficient for every trace output]
    traceWithin      [the bounded search for the trace]
    Outcome.found    [a value found before the budget ran out]
    Outcome.exhausted [the budget ran out with no value found]
    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    Route R          [a weighted wire map over the ring R] -/
def traceRouteOfBudget (σ : R) (f : Route R) (b : TraceBudget σ f) : Route R := fun i =>
  match traceWithin σ f (b.fuel i) i with
  | .found p => some p
  | .exhausted => none

/-- The materialized trace route is pointwise exactly the trace relation.

    Traces           [the union over every path length]
    traceRouteOfBudget [materialize a route by evaluating an explicit trace budget]
    Route R          [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell         [a point: two ring elements side by side, a + b·x]
    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    (j : ℕ)          [the final wire]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    ↔                [exactly when, true in both directions] -/
theorem traces_iff_traceRouteOfBudget (σ : R) (f : Route R) (b : TraceBudget σ f)
    (i j : ℕ) (w : RingCell R) :
    Traces σ f i j w ↔ traceRouteOfBudget σ f b i = some (j, w) := by
  constructor
  · intro h
    simp [traceRouteOfBudget, b.complete h]
  · intro h
    cases he : traceWithin σ f (b.fuel i) i with
    | found p =>
      simp only [traceRouteOfBudget, he, Option.some.injEq] at h
      cases h
      exact traceWithin_sound σ f he
    | exhausted => simp [traceRouteOfBudget, he] at h

/-- A materialized trace route is sound without using the budget completeness field.

    traceRouteOfBudget [materialize a route by evaluating an explicit trace budget]
    Traces           [the union over every path length]
    Route R          [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell         [a point: two ring elements side by side, a + b·x]
    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    (j : ℕ)          [the final wire]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    →                [implication] -/
theorem traceRouteOfBudget_sound (σ : R) (f : Route R) (b : TraceBudget σ f)
    {i j : ℕ} {w : RingCell R} :
    traceRouteOfBudget σ f b i = some (j, w) → Traces σ f i j w := by
  intro h
  cases he : traceWithin σ f (b.fuel i) i with
  | found p =>
    simp only [traceRouteOfBudget, he, Option.some.injEq] at h
    cases h
    exact traceWithin_sound σ f he
  | exhausted => simp [traceRouteOfBudget, he] at h

/-- An explicit input-indexed bound sufficient for every path through `S`.

    Route R          [a weighted wire map over the ring R]
    Runs             [a path through a route, staying inside a set until it leaves]
    runsWithin       [the bounded search for a path]
    Outcome.found    [a value found before the budget ran out]
    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    ∀                [for every single one]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on] -/
structure RunsBudget (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S] where
  fuel : ℕ → ℕ
  complete : ∀ {i j w}, Runs σ f S i j w →
    runsWithin σ f S (fuel i) i = .found (j, w)

/-- Materialize a route by evaluating the supplied inside-set path budget at each input.

    RunsBudget       [an explicit input-indexed bound sufficient for every path output]
    runsWithin       [the bounded search for a path]
    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    Route R          [a weighted wire map over the ring R] -/
def runsRouteOfBudget (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]
    (b : RunsBudget σ f S) : Route R := fun i =>
  match runsWithin σ f S (b.fuel i) i with
  | .found p => some p
  | .exhausted => none

/-- The materialized path route is pointwise exactly the path relation.

    Runs             [a path through a route, staying inside a set until it leaves]
    runsRouteOfBudget [materialize a route by evaluating an explicit path budget]
    Route R          [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell         [a point: two ring elements side by side, a + b·x]
    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    (j : ℕ)          [the final wire]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    ↔                [exactly when, true in both directions] -/
theorem runs_iff_runsRouteOfBudget (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]
    (b : RunsBudget σ f S) (i j : ℕ) (w : RingCell R) :
    Runs σ f S i j w ↔ runsRouteOfBudget σ f S b i = some (j, w) := by
  constructor
  · intro h
    simp [runsRouteOfBudget, b.complete h]
  · intro h
    cases he : runsWithin σ f S (b.fuel i) i with
    | found p =>
      simp only [runsRouteOfBudget, he, Option.some.injEq] at h
      cases h
      exact runsWithin_sound σ f S he
    | exhausted => simp [runsRouteOfBudget, he] at h

/-- A materialized path route is sound without using the budget completeness field.

    runsRouteOfBudget [materialize a route by evaluating an explicit path budget]
    Runs             [a path through a route, staying inside a set until it leaves]
    Route R          [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell         [a point: two ring elements side by side, a + b·x]
    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    (j : ℕ)          [the final wire]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    →                [implication] -/
theorem runsRouteOfBudget_sound (σ : R) (f : Route R) (S : Set ℕ) [DecidablePred S]
    (b : RunsBudget σ f S) {i j : ℕ} {w : RingCell R} :
    runsRouteOfBudget σ f S b i = some (j, w) → Runs σ f S i j w := by
  intro h
  cases he : runsWithin σ f S (b.fuel i) i with
  | found p =>
    simp only [runsRouteOfBudget, he, Option.some.injEq] at h
    cases h
    exact runsWithin_sound σ f S he
  | exhausted => simp [runsRouteOfBudget, he] at h

open RelationalPaths

#guard match runsWithin 2 twoStepRoute (fun x : ℕ => x = 1) 2 0 with
  | Outcome.found (2, (8, 5)) => true
  | _ => false
#guard match runsWithin 2 twoStepRoute (fun x : ℕ => x = 1) 1 0 with
  | Outcome.exhausted => true
  | _ => false

end BoundedRoutes
end Combinator
end Apeiron
