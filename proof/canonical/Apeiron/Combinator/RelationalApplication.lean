import Apeiron.Combinator.RelationalPaths

/-!
# Relation-first combinator application

Application is specified by the trace relation of the plugged route.  This file intentionally
does not select a route from that relation: a selected route is a separate construction, whereas
`RApp` and `CApp` retain the constructive relation itself.
-/

namespace Apeiron
namespace Combinator

open RouteAnyRing

universe u

namespace ApplicationWiring

variable {R : Type u} [CommRing R]

/-- Plug a route into its feedback wiring.

    Route [a weighted wire map: each position goes nowhere, or to one with one weight]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above] -/
abbrev plug (σ : R) (a b : Route R) : Route R :=
  rcomp σ a (rtensor (rid (R := R)) b)

end ApplicationWiring

variable {R : Type u} [CommRing R]

/-- One relation-first application step: either an external step or a queried argument step.

    RelPlug [the relation-first application wiring]
    StepRel [a weighted constructive transition relation]
    ringMul [multiply two points together, the σ way]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    ∨ [either statement holds]
    = [is equal to] -/
def RelPlug (σ : R) (a b : RelationalPaths.StepRel ℕ R) (p q : ℕ)
    (w : Trichotomy.AnyRing.RingCell R) : Prop :=
  (∃ k, a p (2 * k) w ∧ q = 2 * k) ∨
    ∃ u v w₁ w₂, a p (2 * u + 1) w₁ ∧ b u v w₂ ∧ q = 2 * v + 1 ∧
      w = Trichotomy.AnyRing.ringMul σ w₁ w₂

/-- Relation-first application is the finite-path closure of relation-first plugging.

    RelApp [the relation-first application relation]
    RelPlug [one relation-first application step]
    RelRuns [an unbounded finite path]
    oddPos [the odd loop positions]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
def RelApp (σ : R) (a b : RelationalPaths.StepRel ℕ R) (i j : ℕ)
    (w : Trichotomy.AnyRing.RingCell R) : Prop :=
  RelationalPaths.RelRuns σ (RelPlug σ a b) RelationalPaths.oddPos (2 * i) (2 * j) w

/-- The tensor routes an even position through its identity component.

    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem rtensor_even (b : RouteAnyRing.Route R) (k : ℕ) :
    RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) b (2 * k) =
      some (2 * k, Trichotomy.AnyRing.ringOne) := by
  change RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) b
    (Equiv.natSumNatEquivNat (Sum.inl k)) = some (2 * k, Trichotomy.AnyRing.ringOne)
  rw [RouteAnyRing.rtensor, Equiv.symm_apply_apply]
  rfl

/-- The tensor routes an odd position through its argument component.

    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem rtensor_odd (b : RouteAnyRing.Route R) (u : ℕ) :
    RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) b (2 * u + 1) =
      (b u).map fun q => (2 * q.1 + 1, q.2) := by
  change RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) b
    (Equiv.natSumNatEquivNat (Sum.inr u)) = (b u).map fun q => (2 * q.1 + 1, q.2)
  rw [RouteAnyRing.rtensor, Equiv.symm_apply_apply]
  rfl

/-- Relation-first plugging of route graphs is exactly the plugged route graph.

    RelPlug [the relation-first application wiring]
    routeGraph [the transition relation presented by a partial weighted route]
    plug [the relation-first application wiring on routes]
    ringMul [multiply two points together, the σ way]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ↔ [exactly when, true in both directions]
    p [the source position]
    q [the final position]
    w [the edge weight] -/
theorem relPlug_routeGraph (σ : R) (a b : RouteAnyRing.Route R) (p q : ℕ)
    (w : Trichotomy.AnyRing.RingCell R) :
    RelPlug σ (RelationalPaths.routeGraph a) (RelationalPaths.routeGraph b) p q w ↔
      ApplicationWiring.plug σ a b p = some (q, w) := by
  constructor
  · rintro (⟨k, ha, rfl⟩ | ⟨u, v, w₁, w₂, ha, hb, rfl, hw⟩)
    · change a p = some (2 * k, w) at ha
      rw [ApplicationWiring.plug, RouteAnyRing.rcomp, ha, Option.bind_some, rtensor_even]
      simp only [Option.map_some,
        Trichotomy.AnyRing.ringMul_one]
    · change a p = some (2 * u + 1, w₁) at ha
      change b u = some (v, w₂) at hb
      rw [ApplicationWiring.plug, RouteAnyRing.rcomp, ha, Option.bind_some, rtensor_odd, hb]
      simp only [Option.map_some, hw]
  · intro h
    rw [ApplicationWiring.plug, RouteAnyRing.rcomp] at h
    cases ha : a p with
    | none =>
      simp [ha] at h
    | some r =>
      obtain ⟨m, w₁⟩ := r
      simp only [ha, Option.bind_some] at h
      rcases hm : Equiv.natSumNatEquivNat.symm m with k | u
      · have hmk : m = 2 * k := by
          calc m = Equiv.natSumNatEquivNat (Equiv.natSumNatEquivNat.symm m) :=
            (Equiv.apply_symm_apply Equiv.natSumNatEquivNat m).symm
          _ = 2 * k := by rw [hm]; rfl
        subst m
        rw [rtensor_even] at h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq,
          Trichotomy.AnyRing.ringMul_one] at h
        obtain ⟨hq, hw⟩ := h
        subst q
        subst w
        exact Or.inl ⟨k, ha, rfl⟩
      · have hmu : m = 2 * u + 1 := by
          calc m = Equiv.natSumNatEquivNat (Equiv.natSumNatEquivNat.symm m) :=
            (Equiv.apply_symm_apply Equiv.natSumNatEquivNat m).symm
          _ = 2 * u + 1 := by rw [hm]; rfl
        subst m
        rw [rtensor_odd] at h
        cases hb : b u with
        | none =>
          simp [hb] at h
        | some r =>
          obtain ⟨v, w₂⟩ := r
          simp only [hb, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨hq, hw⟩ := h
          subst q
          exact Or.inr ⟨u, v, w₁, w₂, ha, hb, rfl, hw.symm⟩

/-- Relation-first application of route graphs is exactly the plugged route trace.

    RelApp [the relation-first application relation]
    RelPlug [one relation-first application step]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    Traces [the union over finite paths through a route]
    plug [the relation-first application wiring on routes]
    oddPos [the odd loop positions]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ↔ [exactly when, true in both directions]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relApp_routeGraph (σ : R) (a b : RouteAnyRing.Route R) (i j : ℕ)
    (w : Trichotomy.AnyRing.RingCell R) :
    RelApp σ (RelationalPaths.routeGraph a) (RelationalPaths.routeGraph b) i j w ↔
      RouteAnyRing.Traces σ (ApplicationWiring.plug σ a b) i j w := by
  have hplug : RelPlug σ (RelationalPaths.routeGraph a) (RelationalPaths.routeGraph b) =
      RelationalPaths.routeGraph (ApplicationWiring.plug σ a b) := by
    funext p q w
    exact propext (relPlug_routeGraph σ a b p q w)
  rw [RelApp, hplug]
  exact (RelationalPaths.traces_iff_relRuns σ (ApplicationWiring.plug σ a b) i j w).symm

namespace ApplicationWiring

/-- Put a route in each countable copy.

    bang  [the same route running inside every one of the countably many copies]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight] -/
def bang (z : Route R) : Route R := fun j =>
  (z (Nat.unpair j).2).map fun q => (Nat.pair (Nat.unpair j).1 q.1, q.2)

/-- The routing that exchanges the two tracks.

    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
def iRoute : Route R := fun j =>
  Sum.elim (fun k => some (Equiv.natSumNatEquivNat (Sum.inr k), Trichotomy.AnyRing.ringOne))
    (fun k => some (Equiv.natSumNatEquivNat (Sum.inl k), Trichotomy.AnyRing.ringOne))
    (Equiv.natSumNatEquivNat.symm j)

/-- The routing that keeps the first argument and drops the second.

    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
def kRoute : Route R := fun j =>
  Sum.elim
    (fun a => Sum.elim (fun k => some (Equiv.natSumNatEquivNat (Sum.inr k),
      Trichotomy.AnyRing.ringOne))
      (fun _ => none) (Equiv.natSumNatEquivNat.symm a))
    (fun m => some (Equiv.natSumNatEquivNat
      (Sum.inl (Equiv.natSumNatEquivNat (Sum.inl m))), Trichotomy.AnyRing.ringOne))
    (Equiv.natSumNatEquivNat.symm j)

end ApplicationWiring

open ApplicationWiring

variable {R : Type u} [CommRing R]

/-- Relation-first application by the plugged trace.

    σ      [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route  [a weighted wire map: each position goes nowhere, or to one with one weight]
    Traces [the union over every path length through a route]
    ∀      [for every single one]
    ↔      [exactly when, true in both directions] -/
def RApp (σ : R) (a b r : RouteAnyRing.Route R) : Prop :=
  ∀ i j w, RouteAnyRing.Traces σ (plug σ a b) i j w ↔ r i = some (j, w)

/-- Relation-first copying application by the plugged trace.

    σ      [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route  [a weighted wire map: each position goes nowhere, or to one with one weight]
    Traces [the union over every path length through a route]
    bang   [the same route running inside every one of the countably many copies]
    ∀      [for every single one]
    ↔      [exactly when, true in both directions] -/
def CApp (σ : R) (a b r : RouteAnyRing.Route R) : Prop :=
  ∀ i j w, RouteAnyRing.Traces σ (plug σ a (bang b)) i j w ↔ r i = some (j, w)

/-- A relational application has at most one route representative.

    σ     [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight] -/
theorem rApp_unique {σ : R} {a b r r' : RouteAnyRing.Route R}
    (h : RApp σ a b r) (h' : RApp σ a b r') : r = r' := by
  funext i
  cases hr : r i with
  | none =>
    cases hr' : r' i with
    | none => rfl
    | some p =>
      have ht := (h' i p.1 p.2).mpr hr'
      have hc := (h i p.1 p.2).mp ht
      simp [hr] at hc
  | some p =>
    cases hr' : r' i with
    | none =>
      have ht := (h i p.1 p.2).mpr hr
      have hc := (h' i p.1 p.2).mp ht
      simp [hr'] at hc
    | some q =>
      have ht := (h i p.1 p.2).mpr hr
      have ht' := (h' i q.1 q.2).mpr hr'
      have hpq := RouteAnyRing.traces_functional σ (plug σ a b) ht ht'
      exact congrArg some hpq

/-- A copying relational application has at most one route representative.

    σ     [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight] -/
theorem cApp_unique {σ : R} {a b r r' : RouteAnyRing.Route R}
    (h : CApp σ a b r) (h' : CApp σ a b r') : r = r' := by
  apply rApp_unique (a := a) (b := bang b)
  · exact h
  · exact h'

/-- Relational application agrees with bounded trace search at every input.

    σ           [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route       [a weighted wire map: each position goes nowhere, or to one with one weight]
    traceWithin [the bounded search for the trace, up to a trip budget]
    j           [the special direction that squares to give 1, not −1 the way i does]
    ∀           [for every single one]
    ∃           [there is at least one]
    ↔           [exactly when, true in both directions] -/
theorem rApp_iff_traceWithin (σ : R) (a b r : RouteAnyRing.Route R) :
    RApp σ a b r ↔ ∀ i j w, r i = some (j, w) ↔
      ∃ fuel, RouteAnyRing.traceWithin σ (plug σ a b) fuel i = .found (j, w) := by
  constructor
  · intro h i j w
    constructor
    · intro hr
      exact RouteAnyRing.traces_exists_fuel σ (plug σ a b) ((h i j w).mpr hr)
    · rintro ⟨fuel, hFound⟩
      exact (h i j w).mp (RouteAnyRing.traceWithin_sound σ (plug σ a b) hFound)
  · intro h i j w
    constructor
    · intro ht
      exact (h i j w).mpr (RouteAnyRing.traces_exists_fuel σ (plug σ a b) ht)
    · intro hr
      obtain ⟨fuel, hFound⟩ := (h i j w).mp hr
      exact RouteAnyRing.traceWithin_sound σ (plug σ a b) hFound

/-- Copying relational application agrees with bounded trace search.

    σ           [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route       [a weighted wire map: each position goes nowhere, or to one with one weight]
    bang        [the same route running inside every one of the countably many copies]
    traceWithin [the bounded search for the trace, up to a trip budget]
    j           [the special direction that squares to give 1, not −1 the way i does]
    ∀           [for every single one]
    ∃           [there is at least one]
    ↔           [exactly when, true in both directions] -/
theorem cApp_iff_traceWithin (σ : R) (a b r : RouteAnyRing.Route R) :
    CApp σ a b r ↔ ∀ i j w, r i = some (j, w) ↔
      ∃ fuel, RouteAnyRing.traceWithin σ (plug σ a (bang b)) fuel i = .found (j, w) :=
  rApp_iff_traceWithin σ a (bang b) r

/-- The generic identity route at one wire.

    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem rid_apply (i : ℕ) :
    RouteAnyRing.rid (R := R) i = some (i, Trichotomy.AnyRing.ringOne) := rfl

/-- Evaluate the left track of a parallel route.

    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem rtensor_inl (a b : RouteAnyRing.Route R) (k : ℕ) :
    RouteAnyRing.rtensor a b (Equiv.natSumNatEquivNat (Sum.inl k)) =
      (a k).map fun q => (Equiv.natSumNatEquivNat (Sum.inl q.1), q.2) := by
  rw [RouteAnyRing.rtensor, Equiv.symm_apply_apply]
  rfl

/-- Evaluate the right track of a parallel route.

    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem rtensor_inr (a b : RouteAnyRing.Route R) (k : ℕ) :
    RouteAnyRing.rtensor a b (Equiv.natSumNatEquivNat (Sum.inr k)) =
      (b k).map fun q => (Equiv.natSumNatEquivNat (Sum.inr q.1), q.2) := by
  rw [RouteAnyRing.rtensor, Equiv.symm_apply_apply]
  rfl

/-- The even reader at an even wire.

    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem outL_inl (k : ℕ) :
    RouteAnyRing.outL (R := R) (Equiv.natSumNatEquivNat (Sum.inl k)) =
      some (k, Trichotomy.AnyRing.ringOne) := by
  rw [RouteAnyRing.outL, Equiv.symm_apply_apply]
  rfl

/-- The even reader at an odd wire.

    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem outL_inr (k : ℕ) :
    RouteAnyRing.outL (R := R) (Equiv.natSumNatEquivNat (Sum.inr k)) = none := by
  rw [RouteAnyRing.outL, Equiv.symm_apply_apply]
  rfl

/-- The odd reader at an even wire.

    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem outR_inl (k : ℕ) :
    RouteAnyRing.outR (R := R) (Equiv.natSumNatEquivNat (Sum.inl k)) = none := by
  rw [RouteAnyRing.outR, Equiv.symm_apply_apply]
  rfl

/-- The odd reader at an odd wire.

    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem outR_inr (k : ℕ) :
    RouteAnyRing.outR (R := R) (Equiv.natSumNatEquivNat (Sum.inr k)) =
      some (k, Trichotomy.AnyRing.ringOne) := by
  rw [RouteAnyRing.outR, Equiv.symm_apply_apply]
  rfl

/-- An even-only route crosses straight over.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    direct  [what the route does without entering the loop]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rnone   [the route that joins nothing]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem emb_direct (σ : R) (a b : RouteAnyRing.Route R) (i : ℕ) :
    RouteAnyRing.direct σ
      (plug σ (RouteAnyRing.rtensor a (RouteAnyRing.rnone (R := R))) b) i = a i := by
  rw [RouteAnyRing.direct_apply]
  simp only [plug, RouteAnyRing.rcomp, rtensor_inl]
  cases ha : a i with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, rtensor_inl,
      rid_apply, outL_inl, Trichotomy.AnyRing.ringOne_mul,
      Trichotomy.AnyRing.ringMul_one, Prod.mk.eta]

/-- An even-only route has no loop entry.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    entry   [the way from the even wire into the odd one]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rnone   [the route that joins nothing]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem emb_entry (σ : R) (a b : RouteAnyRing.Route R) (i : ℕ) :
    RouteAnyRing.entry σ
      (plug σ (RouteAnyRing.rtensor a (RouteAnyRing.rnone (R := R))) b) i = none := by
  rw [RouteAnyRing.entry_apply]
  simp only [plug, RouteAnyRing.rcomp, rtensor_inl]
  cases ha : a i with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, rtensor_inl,
      rid_apply, outR_inl, Option.map_none]

/-- Applying an even-only route discards its argument.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    Traces  [the union over every path length through a route]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rnone   [the route that joins nothing]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    j       [the special direction that squares to give 1, not −1 the way i does]
    ↔       [exactly when, true in both directions] -/
private theorem emb_app (σ : R) (x y : RouteAnyRing.Route R) (i j : ℕ)
    (w : Trichotomy.AnyRing.RingCell R) :
    RouteAnyRing.Traces σ (plug σ (RouteAnyRing.rtensor x (RouteAnyRing.rnone (R := R))) y) i j w ↔
      x i = some (j, w) := by
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · rwa [emb_direct] at hd
    · exfalso
      rw [RouteAnyRing.rcomp, emb_entry] at hn
      simp at hn
  · intro h
    exact Or.inl (by rw [emb_direct]; exact h)

/-- The identity routing sends an even wire to the matching odd wire.

    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem iRoute_inl (k : ℕ) :
    iRoute (R := R) (Equiv.natSumNatEquivNat (Sum.inl k)) =
      some (Equiv.natSumNatEquivNat (Sum.inr k), Trichotomy.AnyRing.ringOne) := by
  rw [iRoute, Equiv.symm_apply_apply]
  rfl

/-- The identity routing sends an odd wire to the matching even wire.

    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem iRoute_inr (k : ℕ) :
    iRoute (R := R) (Equiv.natSumNatEquivNat (Sum.inr k)) =
      some (Equiv.natSumNatEquivNat (Sum.inl k), Trichotomy.AnyRing.ringOne) := by
  rw [iRoute, Equiv.symm_apply_apply]
  rfl

/-- The identity routing has no straight crossing.

    σ      [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route  [a weighted wire map: each position goes nowhere, or to one with one weight]
    direct [what the route does without entering the loop]
    ℕ      [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem i_direct (σ : R) (x : RouteAnyRing.Route R) (i : ℕ) :
    RouteAnyRing.direct σ (plug σ iRoute x) i = none := by
  rw [RouteAnyRing.direct_apply]
  simp only [plug, RouteAnyRing.rcomp, iRoute_inl, Option.bind_some, rtensor_inr]
  cases hx : x i with
  | none => simp
  | some p => simp only [Option.map_some, Option.bind_some, outL_inr, Option.map_none]

/-- The identity routing enters the loop through its argument.

    σ     [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight]
    entry [the way from the even wire into the odd one]
    ℕ     [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem i_entry (σ : R) (x : RouteAnyRing.Route R) (i : ℕ) :
    RouteAnyRing.entry σ (plug σ iRoute x) i = x i := by
  rw [RouteAnyRing.entry_apply]
  simp only [plug, RouteAnyRing.rcomp, iRoute_inl, Option.bind_some, rtensor_inr]
  cases hx : x i with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, outR_inr,
      Trichotomy.AnyRing.ringOne_mul, Trichotomy.AnyRing.ringMul_one, Prod.mk.eta]

/-- The identity routing cannot take the loop twice.

    σ    [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight]
    body [the way from the odd wire back to the odd wire, one trip round]
    ℕ    [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem i_body (σ : R) (x : RouteAnyRing.Route R) (u : ℕ) :
    RouteAnyRing.body σ (plug σ iRoute x) u = none := by
  rw [RouteAnyRing.body_apply]
  simp only [plug, RouteAnyRing.rcomp, iRoute_inr, Option.bind_some, rtensor_inl,
    rid_apply, Option.map_some, outR_inl, Option.map_none]

/-- The identity routing leaves at the same wire with the unit weight.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    leave   [the way from the odd wire out to the even one]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem i_leave (σ : R) (x : RouteAnyRing.Route R) (u : ℕ) :
    RouteAnyRing.leave σ (plug σ iRoute x) u =
      some (u, Trichotomy.AnyRing.ringOne) := by
  rw [RouteAnyRing.leave_apply]
  simp only [plug, RouteAnyRing.rcomp, iRoute_inr, Option.bind_some, rtensor_inl,
    rid_apply, Option.map_some, outL_inl, Trichotomy.AnyRing.ringOne_mul]

/-- Applying the identity routing returns its argument.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    Traces  [the union over every path length through a route]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    j       [the special direction that squares to give 1, not −1 the way i does]
    ↔       [exactly when, true in both directions] -/
private theorem iRoute_app (σ : R) (x : RouteAnyRing.Route R) (i j : ℕ)
    (w : Trichotomy.AnyRing.RingCell R) :
    RouteAnyRing.Traces σ (plug σ iRoute x) i j w ↔ x i = some (j, w) := by
  have hbody : RouteAnyRing.body σ (plug σ iRoute x) = RouteAnyRing.rnone (R := R) :=
    funext (i_body σ x)
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · rw [i_direct] at hd; exact absurd hd (by simp)
    · cases n with
      | zero =>
        rw [RouteAnyRing.rpow, RouteAnyRing.rcomp_rid_left, RouteAnyRing.rcomp, i_entry] at hn
        cases hx : x i with
        | none => rw [hx] at hn; simp at hn
        | some p =>
          rw [hx] at hn
          simp only [Option.bind_some, i_leave, Option.map_some,
            Trichotomy.AnyRing.ringMul_one, Prod.mk.eta] at hn
          exact hn
      | succ n =>
        rw [hbody, RouteAnyRing.rpow, RouteAnyRing.rcomp_rnone_left,
          RouteAnyRing.rcomp_rnone_left, RouteAnyRing.rcomp_rnone_right] at hn
        simp [RouteAnyRing.rnone] at hn
  · intro h
    refine Or.inr ⟨0, ?_⟩
    rw [RouteAnyRing.rpow, RouteAnyRing.rcomp_rid_left, RouteAnyRing.rcomp, i_entry, h]
    simp only [Option.bind_some, i_leave, Option.map_some, Trichotomy.AnyRing.ringMul_one]

/-- The first K slot enters the loop.

    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem kRoute_inl_inl (k : ℕ) :
    kRoute (R := R) (Equiv.natSumNatEquivNat (Sum.inl (Equiv.natSumNatEquivNat (Sum.inl k)))) =
      some (Equiv.natSumNatEquivNat (Sum.inr k), Trichotomy.AnyRing.ringOne) := by
  rw [kRoute, Equiv.symm_apply_apply]
  change Sum.elim _ _ (Equiv.natSumNatEquivNat.symm
    (Equiv.natSumNatEquivNat (Sum.inl k))) = _
  rw [Equiv.symm_apply_apply]
  rfl

/-- The second K slot is unwired.

    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem kRoute_inl_inr (k : ℕ) :
    kRoute (R := R)
      (Equiv.natSumNatEquivNat (Sum.inl (Equiv.natSumNatEquivNat (Sum.inr k)))) = none := by
  rw [kRoute, Equiv.symm_apply_apply]
  change Sum.elim _ _ (Equiv.natSumNatEquivNat.symm
    (Equiv.natSumNatEquivNat (Sum.inr k))) = _
  rw [Equiv.symm_apply_apply]
  rfl

/-- The K return wire lands in the first result slot.

    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem kRoute_inr (m : ℕ) :
    kRoute (R := R) (Equiv.natSumNatEquivNat (Sum.inr m)) =
      some (Equiv.natSumNatEquivNat (Sum.inl (Equiv.natSumNatEquivNat (Sum.inl m))),
        Trichotomy.AnyRing.ringOne) := by
  rw [kRoute, Equiv.symm_apply_apply]
  rfl

/-- K has no straight crossing.

    σ      [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route  [a weighted wire map: each position goes nowhere, or to one with one weight]
    direct [what the route does without entering the loop]
    ℕ      [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem k_direct (σ : R) (x : RouteAnyRing.Route R) (i : ℕ) :
    RouteAnyRing.direct σ (plug σ kRoute x) i = none := by
  rw [RouteAnyRing.direct_apply]
  cases hs : Equiv.natSumNatEquivNat.symm i with
  | inl k =>
    have hi : i = Equiv.natSumNatEquivNat (Sum.inl k) := by rw [← hs, Equiv.apply_symm_apply]
    rw [hi]
    simp only [plug, RouteAnyRing.rcomp, kRoute_inl_inl, Option.bind_some, rtensor_inr]
    cases hx : x k with
    | none => simp
    | some p => simp only [Option.map_some, Option.bind_some, outL_inr, Option.map_none]
  | inr k =>
    have hi : i = Equiv.natSumNatEquivNat (Sum.inr k) := by rw [← hs, Equiv.apply_symm_apply]
    rw [hi]
    simp only [plug, RouteAnyRing.rcomp, kRoute_inl_inr, Option.bind_none, Option.map_none]

/-- K cannot take the loop twice.

    σ    [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight]
    body [the way from the odd wire back to the odd wire, one trip round]
    ℕ    [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem k_body (σ : R) (x : RouteAnyRing.Route R) (u : ℕ) :
    RouteAnyRing.body σ (plug σ kRoute x) u = none := by
  rw [RouteAnyRing.body_apply]
  simp only [plug, RouteAnyRing.rcomp, kRoute_inr, Option.bind_some, rtensor_inl,
    rid_apply, Option.map_some, outR_inl, Option.map_none]

/-- K leaves the loop at the first result slot.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    leave   [the way from the odd wire out to the even one]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    ringOne [the point (1, 0), which changes nothing when you multiply by it] -/
private theorem k_leave (σ : R) (x : RouteAnyRing.Route R) (u : ℕ) :
    RouteAnyRing.leave σ (plug σ kRoute x) u =
      some (Equiv.natSumNatEquivNat (Sum.inl u), Trichotomy.AnyRing.ringOne) := by
  rw [RouteAnyRing.leave_apply]
  simp only [plug, RouteAnyRing.rcomp, kRoute_inr, Option.bind_some, rtensor_inl,
    rid_apply, Option.map_some, outL_inl, Trichotomy.AnyRing.ringOne_mul]

/-- K's first input slot enters the loop through its argument.

    σ     [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight]
    entry [the way from the even wire into the odd one]
    ℕ     [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem k_entry_inl (σ : R) (x : RouteAnyRing.Route R) (k : ℕ) :
    RouteAnyRing.entry σ (plug σ kRoute x) (Equiv.natSumNatEquivNat (Sum.inl k)) = x k := by
  rw [RouteAnyRing.entry_apply]
  simp only [plug, RouteAnyRing.rcomp, kRoute_inl_inl, Option.bind_some, rtensor_inr]
  cases hx : x k with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, outR_inr,
      Trichotomy.AnyRing.ringOne_mul, Trichotomy.AnyRing.ringMul_one, Prod.mk.eta]

/-- K's second input slot has no loop entry.

    σ     [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight]
    entry [the way from the even wire into the odd one]
    ℕ     [the counting numbers 0, 1, 2, 3, and so on] -/
private theorem k_entry_inr (σ : R) (x : RouteAnyRing.Route R) (k : ℕ) :
    RouteAnyRing.entry σ (plug σ kRoute x) (Equiv.natSumNatEquivNat (Sum.inr k)) = none := by
  rw [RouteAnyRing.entry_apply]
  simp only [plug, RouteAnyRing.rcomp, kRoute_inl_inr, Option.bind_none, Option.map_none]

/-- Applying K once produces its explicit even-track intermediate route.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    Traces  [the union over every path length through a route]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rnone   [the route that joins nothing]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    j       [the special direction that squares to give 1, not −1 the way i does]
    ↔       [exactly when, true in both directions] -/
private theorem kRoute_app (σ : R) (x : RouteAnyRing.Route R) (i j : ℕ)
    (w : Trichotomy.AnyRing.RingCell R) :
    RouteAnyRing.Traces σ (plug σ kRoute x) i j w ↔
      RouteAnyRing.rtensor x (RouteAnyRing.rnone (R := R)) i = some (j, w) := by
  have hbody : RouteAnyRing.body σ (plug σ kRoute x) = RouteAnyRing.rnone (R := R) :=
    funext (k_body σ x)
  cases hs : Equiv.natSumNatEquivNat.symm i with
  | inl k =>
    have hi : i = Equiv.natSumNatEquivNat (Sum.inl k) := by rw [← hs, Equiv.apply_symm_apply]
    subst i
    rw [rtensor_inl]
    constructor
    · rintro (hd | ⟨n, hn⟩)
      · rw [k_direct] at hd; exact absurd hd (by simp)
      · cases n with
        | zero =>
          rw [RouteAnyRing.rpow, RouteAnyRing.rcomp_rid_left, RouteAnyRing.rcomp, k_entry_inl] at hn
          cases hx : x k with
          | none => rw [hx] at hn; simp at hn
          | some p =>
            rw [hx] at hn
            simp only [Option.bind_some, k_leave, Option.map_some,
              Trichotomy.AnyRing.ringMul_one] at hn
            simp only [Option.map_some]
            exact hn
        | succ n =>
          rw [hbody, RouteAnyRing.rpow, RouteAnyRing.rcomp_rnone_left,
            RouteAnyRing.rcomp_rnone_left, RouteAnyRing.rcomp_rnone_right] at hn
          simp [RouteAnyRing.rnone] at hn
    · intro h
      refine Or.inr ⟨0, ?_⟩
      rw [RouteAnyRing.rpow, RouteAnyRing.rcomp_rid_left, RouteAnyRing.rcomp, k_entry_inl]
      cases hx : x k with
      | none => rw [hx] at h; simp at h
      | some p =>
        rw [hx] at h
        simp only [Option.map_some] at h
        simp only [Option.bind_some, k_leave, Option.map_some, Trichotomy.AnyRing.ringMul_one]
        exact h
  | inr k =>
    have hi : i = Equiv.natSumNatEquivNat (Sum.inr k) := by rw [← hs, Equiv.apply_symm_apply]
    subst i
    rw [rtensor_inr]
    simp only [RouteAnyRing.rnone, Option.map_none]
    constructor
    · rintro (hd | ⟨n, hn⟩)
      · rw [k_direct] at hd; exact absurd hd (by simp)
      · rw [RouteAnyRing.rcomp, k_entry_inr] at hn; simp at hn
    · intro h
      exact absurd h (by simp)

/-- The identity route is a relation-first application witness.

    σ     [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route [a weighted wire map: each position goes nowhere, or to one with one weight] -/
theorem rApp_iRoute (σ : R) (x : RouteAnyRing.Route R) : RApp σ iRoute x x :=
  fun i j w => iRoute_app σ x i j w

/-- The first K application has the embedded intermediate route.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rnone   [the route that joins nothing] -/
theorem rApp_kRoute_first (σ : R) (x : RouteAnyRing.Route R) :
    RApp σ kRoute x (RouteAnyRing.rtensor x (RouteAnyRing.rnone (R := R))) :=
  fun i j w => kRoute_app σ x i j w

/-- The embedded K intermediate route discards its second argument.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rnone   [the route that joins nothing] -/
theorem rApp_kRoute_second (σ : R) (x y : RouteAnyRing.Route R) :
    RApp σ (RouteAnyRing.rtensor x (RouteAnyRing.rnone (R := R))) y x :=
  fun i j w => emb_app σ x y i j w

/-- K has a two-stage relation-first witness.

    σ       [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route   [a weighted wire map: each position goes nowhere, or to one with one weight]
    ∃       [there is at least one]
    ∧       [both are true] -/
theorem rApp_kRoute_twoStage (σ : R) (x y : RouteAnyRing.Route R) :
    ∃ z : RouteAnyRing.Route R, RApp σ kRoute x z ∧ RApp σ z y x :=
  ⟨RouteAnyRing.rtensor x (RouteAnyRing.rnone (R := R)), rApp_kRoute_first σ x,
    rApp_kRoute_second σ x y⟩

end Combinator
end Apeiron
