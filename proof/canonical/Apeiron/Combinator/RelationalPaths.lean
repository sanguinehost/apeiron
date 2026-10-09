import Apeiron.Combinator.RouteAnyRing

/-!
# Constructive relational paths over the two-slot algebra

`RouteAnyRing.RunsN` fixes its transition system to a partial weighted route.  This module
factors the path relation through an arbitrary constructive step predicate, then identifies the
route predicate with the existing route semantics.  The weight accumulated at each interior step
is always `Trichotomy.AnyRing.ringMul σ`; no ambient multiplication is introduced here.
-/

namespace Apeiron
namespace Combinator
namespace RelationalPaths

open Apeiron.Trichotomy.AnyRing
open RouteAnyRing

universe u v w

variable {R : Type u} [CommRing R]

/-- A weighted constructive transition relation on vertices of type `α`. -/
abbrev StepRel (α : Type v) (R : Type u) [CommRing R] : Type max u v :=
  α → α → RingCell R → Prop

/-- The relation presented by a partial weighted route.

    Route R          [a weighted wire map: each position goes nowhere, or to one with one weight]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    StepRel          [a weighted constructive transition relation] -/
def routeGraph (f : Route R) : StepRel ℕ R :=
  fun i j w => f i = some (j, w)

/-- A fixed-length path which remains in `S` until its final, external vertex.

    (σ : R)          [the square parameter]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    →                [implication]
    RingCell          [a point: two ring elements side by side, a + b·x] -/
inductive RelRunsN {α : Type v} (σ : R) (g : StepRel α R) (S : Set α) :
    ℕ → α → α → RingCell R → Prop where
  | exit {i j : α} {w : RingCell R} :
      g i j w → j ∉ S → RelRunsN σ g S 0 i j w
  | step {n : ℕ} {i k j : α} {w v : RingCell R} :
      g i k w → k ∈ S → RelRunsN σ g S n k j v →
        RelRunsN σ g S (n + 1) i j (ringMul σ w v)

/-- An unbounded finite path, represented by its explicit number of interior vertices.

    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    (j : α)          [the final vertex]
    RingCell          [a point: two ring elements side by side, a + b·x] -/
def RelRuns {α : Type v} (σ : R) (g : StepRel α R) (S : Set α)
    (i j : α) (w : RingCell R) : Prop :=
  ∃ n, RelRunsN σ g S n i j w

/-- A relation has at most one weighted outgoing edge from any source vertex. -/
def StepFunctional {α : Type v} (g : StepRel α R) : Prop :=
  ∀ {i j₁ j₂ : α} {w₁ w₂ : RingCell R},
    g i j₁ w₁ → g i j₂ w₂ → (j₁, w₁) = (j₂, w₂)

/-- Every partial route presents a functional step relation.

    Route R          [a weighted wire map: each position goes nowhere, or to one with one weight]
    StepFunctional   [a relation with at most one weighted outgoing edge] -/
theorem routeGraph_functional (f : Route R) : StepFunctional (routeGraph f) := by
  intro i j₁ j₂ w₁ w₂ h₁ h₂
  change f i = some (j₁, w₁) at h₁
  change f i = some (j₂, w₂) at h₂
  exact Option.some.inj (h₁.symm.trans h₂)

/-- The abstract path relation specializes definitionally to the route path relation.

    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    Route R          [a weighted wire map: each position goes nowhere, or to one with one weight]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    RingCell          [a point: two ring elements side by side, a + b·x]
    (j : ℕ)          [the final wire]
    RunsN            [a path of given length through a route, staying inside a set until it leaves]
    ↔                [exactly when, true in both directions] -/
theorem relRunsN_routeGraph_iff (σ : R) (f : Route R) (S : Set ℕ)
    (n i j : ℕ) (w : RingCell R) :
    RelRunsN σ (routeGraph f) S n i j w ↔ RunsN σ f S n i j w := by
  constructor
  · intro h
    induction h with
    | exit hEdge hOutside =>
      exact .exit hEdge hOutside
    | step hEdge hInside _ ih =>
      exact .step hEdge hInside ih
  · intro h
    induction h with
    | exit hEdge hOutside =>
      exact .exit hEdge hOutside
    | step hEdge hInside _ ih =>
      exact .step hEdge hInside ih

/-- The unbounded abstract path relation specializes to `RouteAnyRing.Runs`.

    (σ : R)          [the parameter: what the adjoined direction squares to, below, zero, or above]
    (j : ℕ)          [the final wire]
    Route R          [a weighted wire map: each position goes nowhere, or to one with one weight]
    ℕ                [the counting numbers 0, 1, 2, 3, and so on]
    RingCell          [a point: two ring elements side by side, a + b·x]
    Runs             [a path through a route, staying inside a set until it leaves]
    ↔                [exactly when, true in both directions] -/
theorem relRuns_routeGraph_iff (σ : R) (f : Route R) (S : Set ℕ)
    (i j : ℕ) (w : RingCell R) :
    RelRuns σ (routeGraph f) S i j w ↔ Runs σ f S i j w := by
  constructor
  · rintro ⟨n, h⟩
    exact ⟨n, (relRunsN_routeGraph_iff σ f S n i j w).mp h⟩
  · rintro ⟨n, h⟩
    exact ⟨n, (relRunsN_routeGraph_iff σ f S n i j w).mpr h⟩

/-- The loop positions are the odd half of the doubled wire.

    Set ℕ [the positions a path may pass through] -/
def oddPos : Set ℕ := {p | ∃ u, p = Equiv.natSumNatEquivNat (Sum.inr u)}

/-- The odd copy is a loop position.

    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∈ [is one of the things in] -/
private theorem mem_oddPos (u : ℕ) :
    Equiv.natSumNatEquivNat (Sum.inr u) ∈ oddPos := ⟨u, rfl⟩

/-- The even copy is outside the loop.

    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem not_mem_oddPos (k : ℕ) :
    Equiv.natSumNatEquivNat (Sum.inl k) ∉ oddPos := by
  rintro ⟨u, hu⟩
  have hc := Equiv.natSumNatEquivNat.injective hu
  exact absurd hc (by simp)

/-- The even reader at an even wire returns that wire.

    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    outL [read the even positions back]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem outL_inl (k : ℕ) :
    outL (R := R) (Equiv.natSumNatEquivNat (Sum.inl k)) = some (k, ringOne) := by
  rw [RouteAnyRing.outL, Equiv.symm_apply_apply]
  rfl

/-- The even reader at an odd wire returns nothing.

    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    outL [read the even positions back] -/
private theorem outL_inr (k : ℕ) :
    outL (R := R) (Equiv.natSumNatEquivNat (Sum.inr k)) = none := by
  rw [RouteAnyRing.outL, Equiv.symm_apply_apply]
  rfl

/-- The odd reader at an even wire returns nothing.

    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    outR [read the odd positions back] -/
private theorem outR_inl (k : ℕ) :
    outR (R := R) (Equiv.natSumNatEquivNat (Sum.inl k)) = none := by
  rw [RouteAnyRing.outR, Equiv.symm_apply_apply]
  rfl

/-- The odd reader at an odd wire returns that wire.

    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    outR [read the odd positions back]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem outR_inr (k : ℕ) :
    outR (R := R) (Equiv.natSumNatEquivNat (Sum.inr k)) = some (k, ringOne) := by
  rw [RouteAnyRing.outR, Equiv.symm_apply_apply]
  rfl

/-- An odd natural number is not an even natural number.

    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    j [the special direction that squares to give 1, not −1 the way i does]
    ≠ [is not equal to] -/
@[simp] private theorem odd_ne_even (k j : ℕ) : 2 * k + 1 ≠ 2 * j := by
  intro h
  have hb := congrArg Nat.bodd h
  simp at hb

/-- An even natural number is not an odd natural number.

    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    j [the special direction that squares to give 1, not −1 the way i does]
    ≠ [is not equal to] -/
@[simp] private theorem even_ne_odd (k j : ℕ) : 2 * k ≠ 2 * j + 1 := by
  intro h
  exact odd_ne_even j k h.symm

/-- A direct trace edge is one even-to-even route edge.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    direct  [what the route does without entering the loop]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    j       [the special direction that squares to give 1, not −1 the way i does]
    ↔       [exactly when, true in both directions] -/
private theorem direct_iff (σ : R) (f : Route R) (i j : ℕ) (w : RingCell R) :
    direct σ f i = some (j, w) ↔
      f (Equiv.natSumNatEquivNat (Sum.inl i)) =
        some (Equiv.natSumNatEquivNat (Sum.inl j), w) := by
  rw [RouteAnyRing.direct_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inl i)) with
  | none => simp
  | some p =>
    obtain ⟨p1, p2⟩ := p
    cases hs : Equiv.natSumNatEquivNat.symm p1 with
    | inl k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inl k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst p1
      simp only [Option.bind_some, outL_inl, Option.map_some, ringOne_mul, ringMul_one]
      simp only [Option.some.injEq, Prod.mk.injEq, Equiv.apply_eq_iff_eq, Sum.inl.injEq]
    | inr k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inr k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst p1
      simp only [Option.bind_some, outL_inr, Option.map_none]
      simp only [Option.some.injEq, Prod.mk.injEq, Equiv.apply_eq_iff_eq]
      constructor
      · intro h
        cases h
      · rintro ⟨hp, _⟩
        exact nomatch hp

/-- An entry trace edge is one even-to-odd route edge.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    entry   [the way from the even wire into the odd one]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ↔       [exactly when, true in both directions] -/
private theorem entry_iff (σ : R) (f : Route R) (i u : ℕ) (w : RingCell R) :
    entry σ f i = some (u, w) ↔
      f (Equiv.natSumNatEquivNat (Sum.inl i)) =
        some (Equiv.natSumNatEquivNat (Sum.inr u), w) := by
  rw [RouteAnyRing.entry_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inl i)) with
  | none => simp
  | some p =>
    obtain ⟨p1, p2⟩ := p
    cases hs : Equiv.natSumNatEquivNat.symm p1 with
    | inl k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inl k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst p1
      simp only [Option.bind_some, outR_inl, Option.map_none]
      simp only [Option.some.injEq, Prod.mk.injEq, Equiv.apply_eq_iff_eq]
      constructor
      · intro h
        cases h
      · rintro ⟨hp, _⟩
        exact nomatch hp
    | inr k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inr k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst p1
      simp only [Option.bind_some, outR_inr, Option.map_some, ringOne_mul, ringMul_one]
      simp only [Option.some.injEq, Prod.mk.injEq, Equiv.apply_eq_iff_eq, Sum.inr.injEq]

/-- A leave trace edge is one odd-to-even route edge.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    leave   [the way from the odd wire out to the even one]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    j       [the special direction that squares to give 1, not −1 the way i does]
    ↔       [exactly when, true in both directions] -/
private theorem leave_iff (σ : R) (f : Route R) (u j : ℕ) (w : RingCell R) :
    leave σ f u = some (j, w) ↔
      f (Equiv.natSumNatEquivNat (Sum.inr u)) =
        some (Equiv.natSumNatEquivNat (Sum.inl j), w) := by
  rw [RouteAnyRing.leave_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inr u)) with
  | none => simp
  | some p =>
    obtain ⟨p1, p2⟩ := p
    cases hs : Equiv.natSumNatEquivNat.symm p1 with
    | inl k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inl k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst p1
      simp only [Option.bind_some, outL_inl, Option.map_some, ringOne_mul, ringMul_one]
      simp only [Option.some.injEq, Prod.mk.injEq, Equiv.apply_eq_iff_eq, Sum.inl.injEq]
    | inr k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inr k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst p1
      simp only [Option.bind_some, outL_inr, Option.map_none]
      simp only [Option.some.injEq, Prod.mk.injEq, Equiv.apply_eq_iff_eq]
      constructor
      · intro h
        cases h
      · rintro ⟨hp, _⟩
        exact nomatch hp

/-- A body trace edge is one odd-to-odd route edge.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    body    [the way from the odd wire back to the odd wire, one trip round]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ↔       [exactly when, true in both directions] -/
private theorem body_iff (σ : R) (f : Route R) (u v : ℕ) (w : RingCell R) :
    body σ f u = some (v, w) ↔
      f (Equiv.natSumNatEquivNat (Sum.inr u)) =
        some (Equiv.natSumNatEquivNat (Sum.inr v), w) := by
  rw [RouteAnyRing.body_apply]
  cases hf : f (Equiv.natSumNatEquivNat (Sum.inr u)) with
  | none => simp
  | some p =>
    obtain ⟨p1, p2⟩ := p
    cases hs : Equiv.natSumNatEquivNat.symm p1 with
    | inl k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inl k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst p1
      simp only [Option.bind_some, outR_inl, Option.map_none]
      simp only [Option.some.injEq, Prod.mk.injEq, Equiv.apply_eq_iff_eq]
      constructor
      · intro h
        cases h
      · rintro ⟨hp, _⟩
        exact nomatch hp
    | inr k =>
      have hp : p1 = Equiv.natSumNatEquivNat (Sum.inr k) := by
        rw [← hs, Equiv.apply_symm_apply]
      subst p1
      simp only [Option.bind_some, outR_inr, Option.map_some, ringOne_mul, ringMul_one]
      simp only [Option.some.injEq, Prod.mk.injEq, Equiv.apply_eq_iff_eq, Sum.inr.injEq]

/-- Loop iterations followed by a leave are exactly relational loop paths.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    RunsN    [a path of a given length through a route, staying inside a set until it leaves]
    rcomp   [follow one route and then another, multiplying the weights]
    rpow    [a route composed with itself that many times]
    body    [the way from the odd wire back to the odd wire, one trip round]
    leave   [the way from the odd wire out to the even one]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    j       [the special direction that squares to give 1, not −1 the way i does]
    ∀       [for every single one]
    ↔       [exactly when, true in both directions] -/
private theorem loop_iff (σ : R) (f : Route R) : ∀ (n u j : ℕ) (w : RingCell R),
    rcomp σ (rpow σ (body σ f) n) (leave σ f) u = some (j, w) ↔
      RunsN σ f oddPos n
        (Equiv.natSumNatEquivNat (Sum.inr u))
        (Equiv.natSumNatEquivNat (Sum.inl j)) w := by
  intro n
  induction n with
  | zero =>
    intro u j w
    rw [RouteAnyRing.rpow, RouteAnyRing.rcomp_rid_left]
    constructor
    · intro h
      exact .exit ((leave_iff σ f u j w).mp h) (not_mem_oddPos j)
    · intro h
      cases h with
      | exit hf _ => exact (leave_iff σ f u j w).mpr hf
  | succ n ih =>
    intro u j w
    rw [RouteAnyRing.rpow, RouteAnyRing.rcomp_assoc]
    constructor
    · intro h
      cases hb : body σ f u with
      | none => rw [RouteAnyRing.rcomp, hb] at h; simp at h
      | some q =>
        rw [RouteAnyRing.rcomp, hb] at h
        simp only [Option.bind_some] at h
        cases hr : rcomp σ (rpow σ (body σ f) n) (leave σ f) q.1 with
        | none => rw [hr] at h; simp at h
        | some r =>
          rw [hr] at h
          simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨hj, hw⟩ := h
          subst hj
          subst hw
          exact .step ((body_iff σ f u q.1 q.2).mp hb) (mem_oddPos q.1)
            ((ih q.1 r.1 r.2).mp hr)
    · intro h
      cases h with
      | step hf hs hr =>
        obtain ⟨u2, hu2⟩ := hs
        subst hu2
        rw [RouteAnyRing.rcomp, (body_iff σ f u u2 _).mpr hf]
        simp [(ih u2 j _).mpr hr]

/-- A generic route trace is a path through its odd loop positions.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    Traces  [the union over every path length through a route]
    Runs    [a path through a route, staying inside a set until it leaves]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    j       [the special direction that squares to give 1, not −1 the way i does]
    ↔       [exactly when, true in both directions] -/
private theorem traces_iff_runs (σ : R) (f : Route R) (i j : ℕ) (w : RingCell R) :
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
          exact ⟨n + 1, RunsN.step ((entry_iff σ f i q.1 q.2).mp he)
            (mem_oddPos q.1) ((loop_iff σ f n q.1 r.1 r.2).mp hr)⟩
  · rintro ⟨n, h⟩
    cases n with
    | zero =>
      cases h with
      | exit hf _ => exact Or.inl ((direct_iff σ f i j w).mpr hf)
    | succ n =>
      cases h with
      | step hf hs hr =>
        obtain ⟨u, hu⟩ := hs
        subst hu
        refine Or.inr ⟨n, ?_⟩
        rw [rcomp, (entry_iff σ f i u _).mpr hf]
        simp [(loop_iff σ f n u j _).mpr hr]

/-- A route trace is exactly an abstract relational path through the odd loop.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    Traces  [the union over every path length through a route]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    j       [the special direction that squares to give 1, not −1 the way i does]
    ↔       [exactly when, true in both directions] -/
theorem traces_iff_relRuns (σ : R) (f : Route R) (i j : ℕ) (w : RingCell R) :
    Traces σ f i j w ↔ RelRuns σ (routeGraph f) oddPos
      (Equiv.natSumNatEquivNat (Sum.inl i))
      (Equiv.natSumNatEquivNat (Sum.inl j)) w := by
  rw [traces_iff_runs, ← relRuns_routeGraph_iff]

/-- A two-edge integer loop enters and leaves with non-unit weights.

    Route Int [a weighted wire map over the integer ring] -/
private def nonUnitLoopRoute : Route Int
  | 0 => some (1, (1, 1))
  | 1 => some (2, (2, 3))
  | _ => none

/-- The integer fixture has one non-unit loop entry and one non-unit loop exit.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
theorem nonUnit_loop_entry_exit :
    RelRuns 2 (routeGraph nonUnitLoopRoute) oddPos 0 2 (8, 5) := by
  have h : RelRunsN 2 (routeGraph nonUnitLoopRoute) oddPos 1 0 2
      (ringMul 2 (1, 1) (2, 3)) := by
    exact .step (by rfl) (mem_oddPos 0) (.exit (by rfl) (not_mem_oddPos 1))
  norm_num [ringMul] at h
  exact ⟨1, h⟩

/-- A finite image path has an endpoint in the source relation.

    RelRunsN [a fixed-length path which remains inside until its final vertex]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ∀ [for every single one]
    → [takes the left and gives the right; between statements, if the left holds so does the right]
    ↔ [exactly when, true in both directions]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    j [the special direction that squares to give 1, not −1 the way i does] -/
private theorem relRunsN_embed_fwd {α : Type v} {β : Type w} {σ : R}
    {e : α → β} (he : Function.Injective e) {f : StepRel β R} {g : StepRel α R}
    (hEdge : ∀ a c w, f (e a) c w ↔ ∃ b, c = e b ∧ g a b w) {S : Set α} :
    ∀ {n a c w}, RelRunsN σ f (e '' S) n a c w → ∀ i, a = e i →
      ∃ j, c = e j ∧ RelRunsN σ g S n i j w := by
  intro n a c w h
  induction h with
  | @exit a c w hfc hOutside =>
    intro i hai
    subst a
    obtain ⟨j, hc, hgj⟩ := (hEdge i c w).mp hfc
    refine ⟨j, hc, .exit hgj ?_⟩
    intro hj
    exact hOutside ⟨j, hj, hc.symm⟩
  | @step n a c j w v hfc hInside hrun ih =>
    intro i hai
    subst a
    obtain ⟨k, hc, hgk⟩ := (hEdge i c w).mp hfc
    subst c
    obtain ⟨t, ht, het⟩ := hInside
    have hkt : k = t := he (het.symm)
    subst t
    obtain ⟨q, hq, hrun'⟩ := ih k rfl
    exact ⟨q, hq, .step hgk ht hrun'⟩

/-- A finite source path embeds into the image relation.

    RelRunsN [a fixed-length path which remains inside until its final vertex]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ∀ [for every single one]
    ∃ [there is at least one]
    ∧ [both statements hold]
    → [takes the left and gives the right; between statements, if the left holds so does the right]
    ↔ [exactly when, true in both directions]
    j [the special direction that squares to give 1, not −1 the way i does] -/
private theorem relRunsN_embed_bwd {α : Type v} {β : Type w} {σ : R}
    {e : α → β} (he : Function.Injective e) {f : StepRel β R} {g : StepRel α R}
    (hEdge : ∀ a c w, f (e a) c w ↔ ∃ b, c = e b ∧ g a b w) {S : Set α} :
    ∀ {n i j w}, RelRunsN σ g S n i j w → RelRunsN σ f (e '' S) n (e i) (e j) w := by
  intro n i j w h
  induction h with
  | @exit i j w hgj hOutside =>
    refine .exit ((hEdge i (e j) w).mpr ⟨j, rfl, hgj⟩) ?_
    rintro ⟨t, ht, htj⟩
    exact hOutside (he htj ▸ ht)
  | @step n i k j w v hgk hInside hrun ih =>
    exact .step ((hEdge i (e k) w).mpr ⟨k, rfl, hgk⟩) ⟨k, hInside, rfl⟩ ih

/-- Finite weighted paths transport exactly along an injective embedding.

    RelRunsN [a fixed-length path which remains inside until its final vertex]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every single one]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    → [takes the left and gives the right; between statements, if the left holds so does the right]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ↔ [exactly when, true in both directions]
    j [the special direction that squares to give 1, not −1 the way i does] -/
theorem relRunsN_embed_iff {α : Type v} {β : Type w} {σ : R}
    {e : α → β} (he : Function.Injective e) {f : StepRel β R} {g : StepRel α R}
    (hEdge : ∀ a c w, f (e a) c w ↔ ∃ b, c = e b ∧ g a b w) {S : Set α}
    {n : ℕ} {i j : α} {w : RingCell R} :
    RelRunsN σ f (e '' S) n (e i) (e j) w ↔ RelRunsN σ g S n i j w := by
  constructor
  · intro h
    obtain ⟨j', hj', hrun⟩ := relRunsN_embed_fwd he hEdge h i rfl
    have hj : j = j' := he hj'
    subst j'
    exact hrun
  · exact relRunsN_embed_bwd he hEdge

/-- Unbounded weighted paths transport exactly along an injective embedding.

    RelRuns [an unbounded finite path]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    ∀ [for every single one]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    → [takes the left and gives the right; between statements, if the left holds so does the right]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ↔ [exactly when, true in both directions]
    j [the special direction that squares to give 1, not −1 the way i does] -/
theorem relRuns_embed_iff {α : Type v} {β : Type w} {σ : R}
    {e : α → β} (he : Function.Injective e) {f : StepRel β R} {g : StepRel α R}
    (hEdge : ∀ a c w, f (e a) c w ↔ ∃ b, c = e b ∧ g a b w) {S : Set α}
    {i j : α} {w : RingCell R} :
    RelRuns σ f (e '' S) (e i) (e j) w ↔ RelRuns σ g S i j w := by
  constructor
  · rintro ⟨n, h⟩
    exact ⟨n, (relRunsN_embed_iff he hEdge).mp h⟩
  · rintro ⟨n, h⟩
    exact ⟨n, (relRunsN_embed_iff he hEdge).mpr h⟩

/-- An embedded path has an image endpoint exactly when it comes from a source path.

    RelRuns [an unbounded finite path]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    Function.Injective [a map with no merged vertices]
    Set [the positions permitted before the final exit]
    ∀ [for every single one]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    → [takes the left and gives the right; between statements, if the left holds so does the right]
    ↔ [exactly when, true in both directions]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    α [the source vertex type]
    β [the embedded vertex type]
    R [a commutative coefficient ring]
    e [the injective vertex embedding]
    f [the target transition relation]
    g [the source transition relation]
    S [the source interior positions]
    i [the initial source vertex]
    c [the arbitrary final target vertex]
    j [the final source vertex]
    w [the accumulated path weight] -/
theorem relRuns_embed_exists_iff {α : Type v} {β : Type w} {σ : R}
    {e : α → β} (he : Function.Injective e) {f : StepRel β R} {g : StepRel α R}
    (hEdge : ∀ a c w, f (e a) c w ↔ ∃ b, c = e b ∧ g a b w) {S : Set α}
    {i : α} {c : β} {w : RingCell R} :
    RelRuns σ f (e '' S) (e i) c w ↔ ∃ j, c = e j ∧ RelRuns σ g S i j w := by
  constructor
  · rintro ⟨n, h⟩
    obtain ⟨j, hc, hrun⟩ := relRunsN_embed_fwd he hEdge h i rfl
    exact ⟨j, hc, ⟨n, hrun⟩⟩
  · rintro ⟨j, rfl, hrun⟩
    exact (relRuns_embed_iff he hEdge).mpr hrun

/-- A weighted relation read through an equivalence of its vertices.

    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x] -/
def relConj {α : Type v} {β : Type w} (e : α ≃ β) (g : StepRel α R) : StepRel β R :=
  fun a b w => g (e.symm a) (e.symm b) w

/-- Paths through a relation and its equivalence conjugate agree exactly.

    RelRuns [an unbounded finite path]
    relConj [a weighted relation read through an equivalence of its vertices]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    ∀ [for every single one]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ↔ [exactly when, true in both directions]
    j [the special direction that squares to give 1, not −1 the way i does] -/
theorem relRuns_conj_iff {α : Type v} {β : Type w} {σ : R}
    (e : α ≃ β) (g : StepRel α R) (S : Set α) (i j : α) (w : RingCell R) :
    RelRuns σ (relConj e g) (e '' S) (e i) (e j) w ↔ RelRuns σ g S i j w := by
  apply relRuns_embed_iff e.injective
  intro a c w
  constructor
  · intro h
    refine ⟨e.symm c, (e.apply_symm_apply c).symm, ?_⟩
    simpa [relConj] using h
  · rintro ⟨b, rfl, h⟩
    simpa [relConj] using h

/-- A path through `S₂` followed by one through `S₁ ∪ S₂` is a path through the union.

    RelRunsN          [a fixed-length path which remains inside until its final vertex]
    RelRuns           [an unbounded finite path]
    ringMul           [multiply two points together, the σ way]
    RingCell          [a point: two ring elements side by side, a + b·x]
    (σ : R)           [the square parameter]
    (j : α)           [the final vertex]
    ℕ                 [the counting numbers 0, 1, 2, 3, and so on]
    ∈                 [is one of the things in]
    ∀                 [for every single one]
    →                 [implication]
    <                 [is smaller than]
    →                 [implication] -/
theorem relRuns_append {α : Type v} {σ : R} {g : StepRel α R} {S₁ S₂ : Set α} :
    ∀ {n : ℕ} {i k : α} {w : RingCell R}, RelRunsN σ g S₂ n i k w → k ∈ S₁ →
      ∀ {j : α} {v : RingCell R}, RelRuns σ g (S₁ ∪ S₂) k j v →
        RelRuns σ g (S₁ ∪ S₂) i j (ringMul σ w v) := by
  intro n i k w h
  induction h with
  | exit hEdge hOutside =>
    intro hIn j v hTail
    obtain ⟨m, hTail⟩ := hTail
    exact ⟨m + 1, .step hEdge (Or.inl hIn) hTail⟩
  | step hEdge hInside _ ih =>
    intro hIn j v hTail
    obtain ⟨m, hTail⟩ := ih hIn hTail
    refine ⟨m + 1, ?_⟩
    have hStep := RelRunsN.step (σ := σ) (g := g) (S := S₁ ∪ S₂)
      hEdge (Or.inr hInside) hTail
    simpa only [ringMul_assoc] using hStep

/-- A path in `S₂` is a path in `S₁ ∪ S₂` when its final vertex avoids `S₁`.

    RelRunsN          [a fixed-length path which remains inside until its final vertex]
    RingCell          [a point: two ring elements side by side, a + b·x]
    (σ : R)           [the square parameter]
    (j : α)           [the final vertex]
    ℕ                 [the counting numbers 0, 1, 2, 3, and so on]
    ∀                 [for every single one]
    →                 [implication] -/
theorem relRunsN_widen {α : Type v} {σ : R} {g : StepRel α R} {S₁ S₂ : Set α} :
    ∀ {n : ℕ} {i j : α} {w : RingCell R}, RelRunsN σ g S₂ n i j w → j ∉ S₁ →
      RelRunsN σ g (S₁ ∪ S₂) n i j w := by
  intro n i j w h
  induction h with
  | exit hEdge hOutside =>
    intro hNotIn
    exact .exit hEdge (fun hUnion => hUnion.elim hNotIn hOutside)
  | step hEdge hInside _ ih =>
    intro hNotIn
    exact .step hEdge (Or.inr hInside) (ih hNotIn)

/-- A union path either stays in `S₂`, or first reaches a vertex of disjoint `S₁`.

    RelRunsN          [a fixed-length path which remains inside until its final vertex]
    RelRuns           [an unbounded finite path]
    ringMul           [multiply two points together, the σ way]
    RingCell          [a point: two ring elements side by side, a + b·x]
    (σ : R)           [the square parameter]
    (j : α)           [the final vertex]
    ℕ                 [the counting numbers 0, 1, 2, 3, and so on]
    ∈                 [is one of the things in]
    ∀                 [for every single one]
    →                 [implication]
    <                 [is smaller than]
    ∨                 [or: at least one of the two statements holds]
    ∃                 [there is at least one]
    ∧                 [and: both statements hold] -/
theorem relRunsN_split {α : Type v} {σ : R} {g : StepRel α R} {S₁ S₂ : Set α}
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂) :
    ∀ {n : ℕ} {i j : α} {w : RingCell R}, RelRunsN σ g (S₁ ∪ S₂) n i j w →
      (RelRuns σ g S₂ i j w ∧ j ∉ S₁) ∨
      (∃ (k : α) (w₁ w₂ : RingCell R) (m : ℕ), RelRuns σ g S₂ i k w₁ ∧ k ∈ S₁ ∧
        RelRunsN σ g (S₁ ∪ S₂) m k j w₂ ∧ m < n ∧ w = ringMul σ w₁ w₂) := by
  intro n i j w h
  induction h with
  | exit hEdge hOutside =>
    exact Or.inl ⟨⟨0, .exit hEdge (fun hIn => hOutside (Or.inr hIn))⟩,
      fun hIn => hOutside (Or.inl hIn)⟩
  | step hEdge hUnion hRest ih =>
    rcases hUnion with hIn₁ | hIn₂
    · exact Or.inr ⟨_, _, _, _, ⟨0, .exit hEdge (hd _ hIn₁)⟩, hIn₁, hRest,
        Nat.lt_succ_self _, rfl⟩
    · rcases ih with ⟨⟨p, hPrefix⟩, hNotIn₁⟩ |
        ⟨k, w₁, w₂, m, ⟨p, hPrefix⟩, hIn₁, hRest, hm, hw⟩
      · exact Or.inl ⟨⟨p + 1, .step hEdge hIn₂ hPrefix⟩, hNotIn₁⟩
      · refine Or.inr ⟨k, ringMul σ _ w₁, w₂, m,
          ⟨p + 1, .step hEdge hIn₂ hPrefix⟩, hIn₁, hRest, Nat.lt_succ_of_lt hm, ?_⟩
        rw [hw, ringMul_assoc]

/-- A union path collapses to a path whose steps are paths through the second set.

    RelRunsN          [a fixed-length path which remains inside until its final vertex]
    RelRuns           [an unbounded finite path]
    RingCell          [a point: two ring elements side by side, a + b·x]
    (σ : R)           [the square parameter]
    (j : α)           [the final vertex]
    ∈                 [is one of the things in]
    ∀                 [for every]
    ℕ                 [the counting numbers 0, 1, 2, 3, and so on]
    →                 [implication] -/
theorem relRuns_collapse {α : Type v} {σ : R} {g : StepRel α R} {S₁ S₂ : Set α}
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂) :
    ∀ (n : ℕ) {i j : α} {w : RingCell R}, RelRunsN σ g (S₁ ∪ S₂) n i j w →
      RelRuns σ (RelRuns σ g S₂) S₁ i j w := by
  intro n
  induction n using Nat.strong_induction_on with
  | h n ih =>
    intro i j w h
    rcases relRunsN_split hd h with ⟨hSmall, hNotIn₁⟩ |
        ⟨k, w₁, w₂, m, hSmall, hIn₁, hRest, hm, hw⟩
    · exact ⟨0, .exit hSmall hNotIn₁⟩
    · obtain ⟨p, hCollapsed⟩ := ih m hm hRest
      refine ⟨p + 1, ?_⟩
      rw [hw]
      exact .step hSmall hIn₁ hCollapsed

/-- A path whose steps are second-set paths expands to a path through the union.

    RelRunsN          [a fixed-length path which remains inside until its final vertex]
    RelRuns           [an unbounded finite path]
    RingCell          [a point: two ring elements side by side, a + b·x]
    (σ : R)           [the square parameter]
    (j : α)           [the final vertex]
    ℕ                 [the counting numbers 0, 1, 2, 3, and so on]
    ∀                 [for every single one]
    →                 [implication] -/
theorem relRuns_expand {α : Type v} {σ : R} {g : StepRel α R} {S₁ S₂ : Set α} :
    ∀ {n : ℕ} {i j : α} {w : RingCell R}, RelRunsN σ (RelRuns σ g S₂) S₁ n i j w →
      RelRuns σ g (S₁ ∪ S₂) i j w := by
  intro n i j w h
  induction h with
  | exit hSmall hNotIn₁ =>
    obtain ⟨p, hSmall⟩ := hSmall
    exact ⟨p, relRunsN_widen hSmall hNotIn₁⟩
  | step hSmall hIn₁ _ ih =>
    obtain ⟨p, hSmall⟩ := hSmall
    exact relRuns_append hSmall hIn₁ ih

/-- Disjoint path layers factor exactly: run `S₂` first, then run `S₁`.

    RelRuns           [an unbounded finite path]
    RingCell          [a point: two ring elements side by side, a + b·x]
    (σ : R)           [the square parameter]
    (j : α)           [the final vertex]
    ∀                 [for every single one]
    ∈                 [is one of the things in]
    →                 [implication]
    ↔                 [exactly when, true in both directions]
    ∪                 [the union of two sets] -/
theorem relRuns_vanishing {α : Type v} {σ : R} {g : StepRel α R} {S₁ S₂ : Set α}
    (hd : ∀ x, x ∈ S₁ → x ∉ S₂) (i j : α) (w : RingCell R) :
    RelRuns σ g (S₁ ∪ S₂) i j w ↔ RelRuns σ (RelRuns σ g S₂) S₁ i j w := by
  constructor
  · rintro ⟨n, h⟩
    exact relRuns_collapse hd n h
  · rintro ⟨n, h⟩
    exact relRuns_expand h

/-- The first edge of the executable two-edge integer fixture.

    Route Int        [a weighted wire map over the integer ring] -/
def twoStepFirst : Route Int
  | 0 => some (1, (1, 1))
  | _ => none

/-- The second edge of the executable two-edge integer fixture.

    Route Int        [a weighted wire map over the integer ring] -/
def twoStepSecond : Route Int
  | 1 => some (2, (2, 3))
  | _ => none

#guard rcomp 2 twoStepFirst twoStepSecond 0 == some (2, (8, 5))

/-- The two-edge integer route used to witness the relational constructors. 

    Route Int        [a weighted wire map over the integer ring] -/
def twoStepRoute : Route Int
  | 0 => some (1, (1, 1))
  | 1 => some (2, (2, 3))
  | _ => none

/-- The two-edge fixture accumulates `(1, 1)` and `(2, 3)` to `(8, 5)` at `σ = 2`.

    RelRunsN          [a fixed-length path which remains inside until its final vertex]
    routeGraph         [the transition relation presented by a partial weighted route]
    ℕ                  [the counting numbers 0, 1, 2, 3, and so on] -/
example : RelRunsN 2 (routeGraph twoStepRoute) ({1} : Set ℕ) 1 0 2 (8, 5) := by
  have h : RelRunsN 2 (routeGraph twoStepRoute) ({1} : Set ℕ) (0 + 1) 0 2
      (ringMul 2 (1, 1) (2, 3)) := by
    exact .step (by rfl) (by simp) (.exit (by rfl) (by simp))
  norm_num [ringMul] at h
  exact h

/-- Splitting the two-edge fixture at vertex one preserves its accumulated weight.

    RelRuns           [an unbounded finite path]
    ℕ                 [the counting numbers 0, 1, 2, 3, and so on] -/
example : RelRuns 2 (RelRuns 2 (routeGraph twoStepRoute) (∅ : Set ℕ))
    ({1} : Set ℕ) 0 2 (8, 5) := by
  have h : RelRuns (R := Int) 2 (routeGraph twoStepRoute)
      (({1} : Set ℕ) ∪ ∅) 0 2 (8, 5) := by
    have hPath : RelRunsN (R := Int) 2 (routeGraph twoStepRoute)
        (({1} : Set ℕ) ∪ ∅) 1 0 2 (ringMul 2 (1, 1) (2, 3)) := by
      exact RelRunsN.step (by rfl) (by simp) (RelRunsN.exit (by rfl) (by simp))
    refine ⟨1, ?_⟩
    norm_num [ringMul] at hPath ⊢
    exact hPath
  exact (relRuns_vanishing (S₁ := ({1} : Set ℕ)) (S₂ := ∅)
    (fun _ _ hEmpty => hEmpty) 0 2 (8, 5)).mp h

/-- The integer relation on the successor image has one weighted edge.

    StepRel [a weighted constructive transition relation]
    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
private def successorStep : StepRel ℕ Int :=
  fun a c w => a = 1 ∧ c = 2 ∧ w = (2, 3)

/-- The source integer relation has the corresponding unembedded edge.

    StepRel [a weighted constructive transition relation]
    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
private def sourceStep : StepRel ℕ Int :=
  fun a c w => a = 0 ∧ c = 1 ∧ w = (2, 3)

/-- Successor exactly carries source edges into the integer image relation.

    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every single one]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions] -/
private theorem successorStep_edge (a c : ℕ) (w : RingCell Int) :
    successorStep (Nat.succ a) c w ↔ ∃ b, c = Nat.succ b ∧ sourceStep a b w := by
  cases a with
  | zero =>
    constructor
    · rintro ⟨_, hc, hw⟩
      exact ⟨1, hc, rfl, rfl, hw⟩
    · rintro ⟨b, hc, _, hb, hw⟩
      subst b
      exact ⟨rfl, hc, hw⟩
  | succ a =>
    constructor
    · rintro ⟨h, _, _⟩
      exact nomatch h
    · rintro ⟨b, _, h, _, _⟩
      exact nomatch h

/-- A nonidentity successor embedding preserves the integer edge and its weight.

    RelRuns [an unbounded finite path]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
theorem relRuns_embed_succ_nonUnit :
    RelRuns 2 successorStep (Nat.succ '' (∅ : Set ℕ)) 1 2 (2, 3) := by
  have hsource : sourceStep 0 1 (2, 3) := by
    change 0 = 0 ∧ 1 = 1 ∧ (2, 3) = (2, 3)
    exact ⟨rfl, rfl, rfl⟩
  have hbase : RelRuns 2 sourceStep (∅ : Set ℕ) 0 1 (2, 3) :=
    ⟨0, .exit hsource (by simp)⟩
  exact (relRuns_embed_iff Nat.succ_injective successorStep_edge).mpr hbase

end RelationalPaths
end Combinator
end Apeiron
