import Apeiron.Combinator.Linear.Core.Contraction
import Apeiron.Combinator.BoundedRoutes

/-!
# A stored route holding two routes wired to each other

@name Composition
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite joyal-street-verity-1996
@grade proved

Every combinator proved so far stores a route with no trace inside it: the identity stores the
swap, the discarding one stores its argument beside the empty route, and the contraction stores
its argument relabelled. `B x y z = x (y z)` cannot. It has to store **two** routes wired to each
other, and the wire between them is neither an exit nor a query to the outer argument, so it has
to be a loop internal to the stored route. That makes the stored route a trace, which is what
`Combinator/Tracks.lean` built `traces_flatten` and `runs_vanishing_three` for.

This file lays out that structure and proves the small general facts the equations need. The
equations themselves are in `Combinator/Compositor.lean`.

## Root

| from | here |
|---|---|
| `AnyRing.ringOne_mul`, `AnyRing.ringMul_one` | `swap_arg_comm`, `bPlug_x`, `bPlug_y` |

The layout is arithmetic on the counting numbers and the wire is a permutation of them, so
`theory/NOVELTY.md`'s A1 spectator test applies as it did to the contraction: the same wire
serves every setting. The parameter supplies the multiplication along a path and enters through
`Combinator/Vanishing.lean`, where a path's weight is the product along it.

## The four tracks

A position of the wire splits twice, and the four pieces get one job each:

```
   extPos (extPos k)     the outside, which is the whole composite's outside
   argPos (extPos u)     the first route asking its argument
   argPos (argPos u)     the second route answering
   extPos (argPos v)     the second route asking the outer argument
```

The middle two are the odd positions, so they are the ones a trace collapses, and the outer two
are the even ones, so they survive as the composite's own outside and argument channel. That is
the reason for this order rather than the obvious one: with it, the stored route's trace is a
route on the composite's own wire with no relabelling in between, and `bComb` below is a bare
`traceRoute`.

`xPos` lays the first route on the outside and the asking track, `yPos` lays the second on the
answering track and the outer argument track. `bSwapFun` exchanges the two middle tracks and
fixes the outer two: what the first route asks for is what the second route is asked, and what
the second answers is what the first receives. It is an involution, so its round trips are free.

`pairRoute` is the two routes side by side in that layout, `bStore` follows it with the wire,
and `bComb` is the trace of that. Feeding the outer argument is `bArg`, which acts on the fourth
track alone, and `bBig` is the whole machine: both routes, the wire, and the argument.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy
open RouteAnyRing

universe u

-- ═══════════════════════════════════════════════════════════════════
-- SMALL GENERAL FACTS ABOUT ROUTES AND COLLAPSES
-- ═══════════════════════════════════════════════════════════════════

/-- Two routes with the same steps are the same route.

    {a b : Route}                                  [two weighted wire maps]
    (h : ∀ i j w, a i = some (j, w) ↔ b i = some (j, w))    [with the same steps]
    a = b                                                   [are equal] -/
theorem route_ext {a b : Route} (h : ∀ i j w, a i = some (j, w) ↔ b i = some (j, w)) : a = b := by
  funext i
  cases ha : a i with
  | none =>
    cases hb : b i with
    | none => rfl
    | some p =>
      have hc : a i = some (p.1, p.2) := (h i p.1 p.2).mpr (by rw [hb])
      rw [ha] at hc
      exact absurd hc (by simp)
  | some p =>
    have hb : b i = some (p.1, p.2) := (h i p.1 p.2).mp (by rw [ha])
    rw [hb]

/-- A path takes a first step.

    {σ : ℝ} {f : Route} {S : Set ℕ} {i j : ℕ} {w : Cell}   [parameter, route, set, path]
    (h : Runs σ f S i j w)                                 [a path]
    ∃ m v, f i = some (m, v)                               [so the route moves at its start] -/
theorem runs_start_some {σ : ℝ} {f : Route} {S : Set ℕ} {i j : ℕ} {w : Cell}
    (h : Runs σ f S i j w) : ∃ m v, f i = some (m, v) := by
  obtain ⟨n, h⟩ := h
  cases h with
  | exit hf _ => exact ⟨_, _, hf⟩
  | step hf _ _ => exact ⟨_, _, hf⟩

/-- Collapsing changes nothing where the route leaves at once.

    {σ : ℝ} {f : Route} {S : Set ℕ} {i : ℕ}        [parameter, route, loop, position]
    (h : ∀ j w, f i = some (j, w) → j ∉ S)         [every step from here leaves the loop]
    runsRoute σ f S i = f i                        [so the collapse agrees with the route] -/
theorem runsRoute_eq_self {σ : ℝ} {f : Route} {S : Set ℕ} {i : ℕ}
    (h : ∀ j w, f i = some (j, w) → j ∉ S) : runsRoute σ f S i = f i := by
  cases hf : f i with
  | none =>
    cases hr : runsRoute σ f S i with
    | none => rfl
    | some p =>
      obtain ⟨m, v, hm⟩ := runs_start_some ((runsRoute_eq σ f S i p.1 p.2).mp (by rw [hr]))
      rw [hf] at hm
      exact absurd hm (by simp)
  | some p =>
    exact (runsRoute_eq σ f S i p.1 p.2).mpr
      ⟨0, RunsN.exit (by rw [hf]) (h p.1 p.2 (by rw [hf]))⟩

/-- A relational path is its one route edge when every edge from its start leaves the loop.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    Route [a weighted wire map]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Set [a collection of positions]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every]
    ∉ [is not one of the things in]
    → [implies]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    j [the final position] -/
theorem relRuns_eq_self {R : Type u} [CommRing R] {σ : R} {f : RouteAnyRing.Route R}
    {S : Set ℕ} {i : ℕ} (h : ∀ j w, f i = some (j, w) → j ∉ S) (j : ℕ)
    (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph f) S i j w ↔ f i = some (j, w) := by
  constructor
  · rintro ⟨n, hrun⟩
    cases hrun with
    | exit hEdge _ => exact hEdge
    | step hEdge hInside _ => exact False.elim (h _ _ hEdge hInside)
  · intro hEdge
    exact ⟨0, .exit hEdge (h _ _ hEdge)⟩

/-- A relational path through a loop-fixing postcomposition is a first path and final edge.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rcomp [follow one route and then another, multiplying the weights]
    Route [a weighted wire map]
    RingCell [a point: two ring elements side by side, a + b·x]
    ringOne [the point (1, 0), which changes nothing when you multiply by it]
    ringMul [multiply two points together, the σ way]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Set [a collection of positions]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every]
    ∈ [is one of the things in]
    → [implies]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    j [an output position of E] -/
theorem relRuns_rcomp_id {R : Type u} [CommRing R] {σ : R}
    {F E : RouteAnyRing.Route R} {S : Set ℕ}
    (hEid : ∀ s, s ∈ S → E s = some (s, AnyRing.ringOne))
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S) (i c : ℕ)
    (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (RouteAnyRing.rcomp σ F E)) S i c w ↔
      ∃ m u v, RelationalPaths.RelRuns σ (RelationalPaths.routeGraph F) S i m u ∧
        E m = some (c, v) ∧ w = AnyRing.ringMul σ u v := by
  rw [RelationalPaths.relRuns_routeGraph_iff, runs_rcomp_id_on hEid hEout]
  constructor
  · rintro ⟨m, u, v, hrun, hE, hw⟩
    exact ⟨m, u, v, (RelationalPaths.relRuns_routeGraph_iff σ F S i m u).mpr hrun, hE, hw⟩
  · rintro ⟨m, u, v, hrun, hE, hw⟩
    exact ⟨m, u, v, (RelationalPaths.relRuns_routeGraph_iff σ F S i m u).mp hrun, hE, hw⟩

/-- A relational route path from an embedded input has an embedded output and source path.

    RelRuns [an unbounded finite path]
    RelRunsN [a fixed-length path which remains in a set until it leaves]
    routeGraph [the transition relation presented by a partial weighted route]
    Route [a weighted wire map]
    RingCell [a point: two ring elements side by side, a + b·x]
    Set [a collection of positions]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Function.Injective [different inputs have different outputs]
    ∀ [for every]
    ∃ [there is at least one]
    ∧ [both statements hold]
    → [implies]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    c [the final position] -/
theorem relRuns_route_embed {R : Type u} [CommRing R] {σ : R} {ι : ℕ → ℕ}
    (hι : Function.Injective ι) {f g : RouteAnyRing.Route R}
    (hfg : ∀ q, f (ι q) = (g q).map fun p => (ι p.1, p.2)) {S : Set ℕ}
    (a c : ℕ) (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph f) (ι '' S) (ι a) c w ↔
      ∃ b, c = ι b ∧ RelationalPaths.RelRuns σ (RelationalPaths.routeGraph g) S a b w := by
  have hEdge : ∀ q d z, RelationalPaths.routeGraph f (ι q) d z ↔
      ∃ b, d = ι b ∧ RelationalPaths.routeGraph g q b z := by
    intro q d z
    change f (ι q) = some (d, z) ↔ _
    rw [hfg q]
    constructor
    · intro h
      obtain ⟨⟨b, v⟩, hg, hd⟩ := Option.map_eq_some_iff.mp h
      injection hd with hdb hvz
      subst d
      subst z
      exact ⟨b, rfl, hg⟩
    · rintro ⟨b, rfl, hg⟩
      exact Option.map_eq_some_iff.mpr ⟨(b, z), hg, rfl⟩
  have hImage : ∀ {n d e z},
      RelationalPaths.RelRunsN σ (RelationalPaths.routeGraph f) (ι '' S) n d e z →
        ∀ q, d = ι q → ∃ b, e = ι b := by
    intro n d e z hrun
    induction hrun with
    | @exit d e z hEdge' _ =>
      intro q hd
      subst d
      obtain ⟨b, he, _⟩ := (hEdge q e z).mp hEdge'
      exact ⟨b, he⟩
    | @step n d k e z v hEdge' hInside htail ih =>
      intro q hd
      subst d
      obtain ⟨b, hk, _⟩ := (hEdge q k z).mp hEdge'
      subst k
      obtain ⟨t, ht, hbt⟩ := hInside
      have hbt' : b = t := hι hbt.symm
      subst t
      exact ih b rfl
  constructor
  · rintro ⟨n, hrun⟩
    obtain ⟨b, hc⟩ := hImage hrun a rfl
    subst c
    exact ⟨b, rfl, (RelationalPaths.relRuns_embed_iff hι hEdge).mp ⟨n, hrun⟩⟩
  · rintro ⟨b, rfl, hrun⟩
    exact (RelationalPaths.relRuns_embed_iff hι hEdge).mpr hrun

/-- A relational path over the odd loop exits at an external position exactly when it is a trace.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    Traces [the union over every path length through a route]
    oddPos [the odd half of a doubled wire]
    extPos [a position of the wire outside both loops]
    Route [a weighted wire map]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    c [the final position] -/
theorem relRuns_oddPos_iff_traces {R : Type u} [CommRing R] {σ : R}
    {f : RouteAnyRing.Route R} (a c : ℕ) (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph f) RelationalPaths.oddPos
      (extPos a) c w ↔ ∃ b, c = extPos b ∧ RouteAnyRing.Traces σ f a b w := by
  constructor
  · intro h
    rw [RelationalPaths.relRuns_routeGraph_iff] at h
    change RouteAnyRing.Runs σ f oddPos (extPos a) c w at h
    obtain ⟨b, hc⟩ := not_oddPos_ext (runs_end_not_mem h)
    subst c
    refine ⟨b, rfl, ?_⟩
    rw [RelationalPaths.traces_iff_relRuns]
    exact (RelationalPaths.relRuns_routeGraph_iff σ f RelationalPaths.oddPos
      (extPos a) (extPos b) w).mpr h
  · rintro ⟨b, rfl, htrace⟩
    exact (RelationalPaths.traces_iff_relRuns σ f a b w).mp htrace

/-- **Collapsing a loop commutes with a route that leaves the loop alone.**

@cite joyal-street-verity-1996

    {σ : ℝ} {F E : Route} {S : Set ℕ}              [parameter, two routes, loop positions]
    (hEid : ∀ s, s ∈ S → E s = some (s, one))      [E does nothing on the loop]
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S)   [and cannot enter it]
    runsRoute σ (rcomp σ F E) S = rcomp σ (runsRoute σ F S) E   [collapse first or last] -/
theorem runsRoute_rcomp_id {σ : ℝ} {F E : Route} {S : Set ℕ}
    (hEid : ∀ s, s ∈ S → E s = some (s, one))
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S) :
    runsRoute σ (rcomp σ F E) S = rcomp σ (runsRoute σ F S) E := by
  refine route_ext fun i c w => ?_
  rw [runsRoute_eq, runs_rcomp_id_on hEid hEout, rcomp_eq_some]
  constructor
  · rintro ⟨m, w₁, w₂, hr, hE, hw⟩
    exact ⟨m, w₁, w₂, (runsRoute_eq σ F S i m w₁).mpr hr, hE, hw⟩
  · rintro ⟨m, w₁, w₂, hr, hE, hw⟩
    exact ⟨m, w₁, w₂, (runsRoute_eq σ F S i m w₁).mp hr, hE, hw⟩

/-- **A collapse transports along an embedding.**

@cite girard-1989

    {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)      [the parameter and an embedding]
    {f g : Route}                                        [a wide route and a narrow one]
    (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))   [agreeing on the image]
    {S : Set ℕ} (a : ℕ)                                  [a loop and a position]
    runsRoute σ f (ι '' S) (ι a)                         [collapse the wide loop]
    = (runsRoute σ g S a).map fun q => (ι q.1, q.2)      [is the narrow collapse, embedded] -/
theorem runsRoute_embed {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι) {f g : Route}
    (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2)) {S : Set ℕ} (a : ℕ) :
    runsRoute σ f (ι '' S) (ι a) = (runsRoute σ g S a).map fun q => (ι q.1, q.2) := by
  cases hg : runsRoute σ g S a with
  | none =>
    cases hf : runsRoute σ f (ι '' S) (ι a) with
    | none => simp
    | some p =>
      obtain ⟨n, hn⟩ := (runsRoute_eq σ f (ι '' S) (ι a) p.1 p.2).mp (by rw [hf])
      obtain ⟨j, _, hrun⟩ := runsN_embed_fwd hι hfg hn a rfl
      rw [(runsRoute_eq σ g S a j p.2).mpr ⟨n, hrun⟩] at hg
      exact absurd hg (by simp)
  | some p =>
    have hrun := (runsRoute_eq σ g S a p.1 p.2).mp (by rw [hg])
    rw [(runsRoute_eq σ f (ι '' S) (ι a) (ι p.1) p.2).mpr ((runs_embed hι hfg a p.1 p.2).mpr hrun)]
    rfl

/-- Composing two routes that both embed is the embedded composite.

    {R : Type u} [CommRing R] {σ : R} {ι : ℕ → ℕ}         [ring, parameter, embedding]
    {f g f' g' : Route R}                                 [four routes]
    (h1 : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))     [the first pair agrees]
    (h2 : ∀ a, f' (ι a) = (g' a).map fun q => (ι q.1, q.2))   [and so does the second]
    (a : ℕ)                                                   [at a position]
    rcomp σ f f' (ι a) = (rcomp σ g g' a).map fun q => (ι q.1, q.2)   [the composites agree] -/
theorem rcomp_embed {R : Type u} [CommRing R] {σ : R} {ι : ℕ → ℕ}
    {f g f' g' : RouteAnyRing.Route R}
    (h1 : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))
    (h2 : ∀ a, f' (ι a) = (g' a).map fun q => (ι q.1, q.2)) (a : ℕ) :
    RouteAnyRing.rcomp σ f f' (ι a) =
      (RouteAnyRing.rcomp σ g g' a).map fun q => (ι q.1, q.2) := by
  rw [RouteAnyRing.rcomp, h1 a, RouteAnyRing.rcomp]
  cases hg : g a with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, h2 p.1]
    cases hg' : g' p.1 with
    | none => simp
    | some q => simp

/-- The trace as a collapse, stated in the named position.

    (σ : ℝ) (g : Route) (i j : ℕ) (w : Cell)       [the parameter, a route, a step]
    traceRoute σ g i = some (j, w)                 [the trace lands there]
    ↔                                              [exactly when]
    runsRoute σ g oddPos (extPos i) = some (extPos j, w)   [so does the collapse] -/
theorem traceRoute_eq_runsRoute_ext (σ : ℝ) (g : Route) (i j : ℕ) (w : Cell) :
    traceRoute σ g i = some (j, w) ↔ runsRoute σ g oddPos (extPos i) = some (extPos j, w) :=
  traceRoute_eq_runsRoute σ g i j w

/-- **The trace of a route is its collapse over the odd positions, read at the even ones.**

@cite girard-1989

    (σ : ℝ) (f : Route) (a : ℕ)                    [the parameter, a route, a position]
    runsRoute σ f oddPos (extPos a)                [collapsing the loop at an even position]
    = (traceRoute σ f a).map fun q => (extPos q.1, q.2)   [is the trace, laid on the even track]

    So: the two constructions which have been used side by side since `Combinator/Tracks.lean`
    are one map apart, and this is the map. It is what lets a combinator whose stored route is a
    trace be plugged into the collapse machinery without a separate account of either. -/
theorem runsRoute_oddPos_traceRoute (σ : ℝ) (f : Route) (a : ℕ) :
    runsRoute σ f oddPos (extPos a) = (traceRoute σ f a).map fun q => (extPos q.1, q.2) := by
  cases ht : traceRoute σ f a with
  | none =>
    cases hr : runsRoute σ f oddPos (extPos a) with
    | none => simp
    | some p =>
      obtain ⟨p1, p2⟩ := p
      have hrun := (runsRoute_eq σ f oddPos (extPos a) p1 p2).mp (by rw [hr])
      obtain ⟨k, hk⟩ := not_oddPos_ext (runs_end_not_mem hrun)
      subst hk
      rw [(traceRoute_eq_runsRoute_ext σ f a k p2).mpr
        ((runsRoute_eq σ f oddPos (extPos a) (extPos k) p2).mpr hrun)] at ht
      exact absurd ht (by simp)
  | some p =>
    rw [(traceRoute_eq_runsRoute_ext σ f a p.1 p.2).mp (by rw [ht])]
    rfl

-- ═══════════════════════════════════════════════════════════════════
-- THE FOUR TRACKS
-- ═══════════════════════════════════════════════════════════════════

/-- Every position is even or odd.

    (p : ℕ)                                        [any position]
    (∃ k, p = extPos k) ∨ (∃ u, p = argPos u)      [it is on one of the two tracks] -/
theorem pos_split2 (p : ℕ) : (∃ k, p = extPos k) ∨ (∃ u, p = argPos u) := by
  rcases hp : Equiv.natSumNatEquivNat.symm p with k | u
  · exact Or.inl ⟨k, by rw [extPos, ← hp, Equiv.apply_symm_apply]⟩
  · exact Or.inr ⟨u, by rw [argPos, ← hp, Equiv.apply_symm_apply]⟩

/-- The odd positions are distinct.

    Function.Injective argPos                      [distinct positions stay distinct] -/
theorem argPos_injective : Function.Injective argPos := by
  intro a b h
  rw [argPos, argPos] at h
  exact Sum.inr.inj (Equiv.natSumNatEquivNat.injective h)

/-- The two tracks never meet.

    (k v : ℕ)                                      [two positions]
    extPos k ≠ argPos v                            [an even copy is never an odd one] -/
theorem extPos_ne_argPos (k v : ℕ) : extPos k ≠ argPos v := by
  intro h
  exact absurd (Equiv.natSumNatEquivNat.injective h) (by simp)

/-- The track on which the second route asks the outer argument.

    (v : ℕ)                                        [a position]
    the odd copy of v, made even                   [its place on the argument track]
    : ℕ                                            [a position of the wire] -/
def zPos (v : ℕ) : ℕ := extPos (argPos v)

/-- The argument track named twice.

    (v : ℕ)                                        [a position]
    zPos v = extPos (argPos v)                     [the same position] -/
theorem zPos_eq (v : ℕ) : zPos v = extPos (argPos v) := rfl

/-- The positions of the outer argument channel.

    Set ℕ                                          [the track the composite offers outwards] -/
def zSet : Set ℕ := {p | ∃ v, p = zPos v}

/-- An argument position is on the argument track.

    (v : ℕ)                                        [a position]
    zPos v ∈ zSet                                  [it lies on the track] -/
theorem mem_zSet (v : ℕ) : zPos v ∈ zSet := ⟨v, rfl⟩

/-- The argument track is even, so no trace of the composite collapses it.

    ∀ x, x ∈ zSet → x ∉ oddPos                     [the argument track is not a loop] -/
theorem zSet_not_oddPos : ∀ x, x ∈ zSet → x ∉ oddPos := by
  rintro x ⟨v, rfl⟩
  exact not_mem_oddPos (argPos v)

/-- The first route's place on the wire.

    (p : ℕ)                                        [one of the first route's positions]
    the outside track if p is even, the asking track if p is odd   [where it sits]
    : ℕ                                            [a position of the wire] -/
def xPos (p : ℕ) : ℕ :=
  Equiv.natSumNatEquivNat (Sum.map extPos extPos (Equiv.natSumNatEquivNat.symm p))

/-- The second route's place on the wire.

    (p : ℕ)                                        [one of the second route's positions]
    the answering track if p is even, the argument track if p is odd   [where it sits]
    : ℕ                                            [a position of the wire] -/
def yPos (p : ℕ) : ℕ :=
  Equiv.natSumNatEquivNat (Sum.swap (Sum.map argPos argPos (Equiv.natSumNatEquivNat.symm p)))

/-- The first route's outside is the composite's outside.

    (k : ℕ)                                        [a position]
    xPos (extPos k) = extPos (extPos k)            [the first route's even track is the outside] -/
theorem xPos_ext (k : ℕ) : xPos (extPos k) = extPos (extPos k) := by
  simp only [xPos, extPos, Equiv.symm_apply_apply, Sum.map_inl]

/-- The first route's argument channel is the asking track.

    (u : ℕ)                                        [a position]
    xPos (argPos u) = outPos u                     [the first route's odd track asks] -/
theorem xPos_arg (u : ℕ) : xPos (argPos u) = outPos u := by
  simp only [xPos, argPos, extPos, outPos, Equiv.symm_apply_apply, Sum.map_inr]

/-- The second route's outside is the answering track.

    (u : ℕ)                                        [a position]
    yPos (extPos u) = innPos u                     [the second route's even track answers] -/
theorem yPos_ext (u : ℕ) : yPos (extPos u) = innPos u := by
  simp only [yPos, extPos, argPos, innPos, Equiv.symm_apply_apply, Sum.map_inl, Sum.swap_inl]

/-- The second route's argument channel is the composite's.

    (v : ℕ)                                        [a position]
    yPos (argPos v) = zPos v                       [the second route asks outwards] -/
theorem yPos_arg (v : ℕ) : yPos (argPos v) = zPos v := by
  simp only [yPos, argPos, extPos, zPos, Equiv.symm_apply_apply, Sum.map_inr, Sum.swap_inr]

/-- The first route's places are distinct.

    Function.Injective xPos                        [distinct positions stay distinct] -/
theorem xPos_injective : Function.Injective xPos := by
  intro a b h
  rcases pos_split2 a with ⟨k, rfl⟩ | ⟨u, rfl⟩ <;> rcases pos_split2 b with ⟨k', rfl⟩ | ⟨u', rfl⟩
  · rw [xPos_ext, xPos_ext] at h
    rw [extPos_injective (extPos_injective h)]
  · rw [xPos_ext, xPos_arg] at h
    exact absurd (h.symm ▸ mem_outerSet u') (extPos_not_mem (extPos k)).1
  · rw [xPos_arg, xPos_ext] at h
    exact absurd (h ▸ mem_outerSet u) (extPos_not_mem (extPos k')).1
  · rw [xPos_arg, xPos_arg, outPos_eq, outPos_eq] at h
    rw [extPos_injective (argPos_injective h)]

/-- The second route's places are distinct.

    Function.Injective yPos                        [distinct positions stay distinct] -/
theorem yPos_injective : Function.Injective yPos := by
  intro a b h
  rcases pos_split2 a with ⟨k, rfl⟩ | ⟨u, rfl⟩ <;> rcases pos_split2 b with ⟨k', rfl⟩ | ⟨u', rfl⟩
  · rw [yPos_ext, yPos_ext, innPos_eq, innPos_eq] at h
    rw [argPos_injective (argPos_injective h)]
  · rw [yPos_ext, yPos_arg, zPos_eq] at h
    exact absurd (h ▸ mem_innerSet k) (extPos_not_mem (argPos u')).2
  · rw [yPos_arg, yPos_ext, zPos_eq] at h
    exact absurd (h.symm ▸ mem_innerSet k') (extPos_not_mem (argPos u)).2
  · rw [yPos_arg, yPos_arg, zPos_eq, zPos_eq] at h
    exact extPos_injective h

/-- The first route never sits on the outer argument track.

    (p : ℕ)                                        [one of the first route's positions]
    xPos p ∉ zSet                                  [it is not on the argument track] -/
theorem xPos_not_zSet (p : ℕ) : xPos p ∉ zSet := by
  rintro ⟨v, hv⟩
  rcases pos_split2 p with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · rw [xPos_ext, zPos_eq] at hv
    exact extPos_ne_argPos k v (extPos_injective hv)
  · rw [xPos_arg, zPos_eq] at hv
    exact absurd (hv ▸ mem_outerSet u) (extPos_not_mem (argPos v)).1

/-- The first route's argument channel is the asking track.

    xPos '' oddPos = outerSet                      [its odd positions are the first loop] -/
theorem xPos_image_oddPos : xPos '' oddPos = outerSet := by
  ext p
  constructor
  · rintro ⟨q, ⟨u, rfl⟩, rfl⟩
    exact ⟨u, (xPos_arg u).symm ▸ rfl⟩
  · rintro ⟨u, rfl⟩
    exact ⟨argPos u, ⟨u, rfl⟩, xPos_arg u⟩

/-- The second route's argument channel is the composite's.

    yPos '' oddPos = zSet                          [its odd positions are the argument track] -/
theorem yPos_image_oddPos : yPos '' oddPos = zSet := by
  ext p
  constructor
  · rintro ⟨q, ⟨v, rfl⟩, rfl⟩
    exact ⟨v, (yPos_arg v).symm ▸ rfl⟩
  · rintro ⟨v, rfl⟩
    exact ⟨argPos v, ⟨v, rfl⟩, yPos_arg v⟩

/-- The composite's argument channel is the argument track.

    extPos '' oddPos = zSet                        [an even copy of an odd position] -/
theorem extPos_image_oddPos : extPos '' oddPos = zSet := by
  ext p
  constructor
  · rintro ⟨q, ⟨v, rfl⟩, rfl⟩
    exact ⟨v, rfl⟩
  · rintro ⟨v, rfl⟩
    exact ⟨argPos v, ⟨v, rfl⟩, rfl⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE WIRE
-- ═══════════════════════════════════════════════════════════════════

/-- Exchange the two halves of a position.

    (q : ℕ)                                        [a position]
    the even copy made odd and the odd made even   [read the other way]
    : ℕ                                            [a position] -/
def flipPos (q : ℕ) : ℕ :=
  Equiv.natSumNatEquivNat (Sum.swap (Equiv.natSumNatEquivNat.symm q))

/-- An even position flips to the odd one.

    (u : ℕ)                                        [a position]
    flipPos (extPos u) = argPos u                  [even becomes odd] -/
theorem flipPos_ext (u : ℕ) : flipPos (extPos u) = argPos u := by
  simp only [flipPos, extPos, argPos, Equiv.symm_apply_apply, Sum.swap_inl]

/-- An odd position flips to the even one.

    (v : ℕ)                                        [a position]
    flipPos (argPos v) = extPos v                  [odd becomes even] -/
theorem flipPos_arg (v : ℕ) : flipPos (argPos v) = extPos v := by
  simp only [flipPos, extPos, argPos, Equiv.symm_apply_apply, Sum.swap_inr]

/-- The wire: exchange the two middle tracks and fix the outer two.

    (i : ℕ)                                        [a position of the wire]
    ℕ                                              [where the wire sends it] -/
def bSwapFun (i : ℕ) : ℕ :=
  Sum.elim (fun k => extPos k) (fun q => argPos (flipPos q)) (Equiv.natSumNatEquivNat.symm i)

/-- The wire fixes the even half.

    (k : ℕ)                                        [a position]
    bSwapFun (extPos k) = extPos k                 [the outside and the argument track stay] -/
theorem bSwapFun_ext (k : ℕ) : bSwapFun (extPos k) = extPos k := by
  simp only [bSwapFun, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The wire flips inside the odd half.

    (q : ℕ)                                        [a position]
    bSwapFun (argPos q) = argPos (flipPos q)       [asking and answering exchange] -/
theorem bSwapFun_arg (q : ℕ) : bSwapFun (argPos q) = argPos (flipPos q) := by
  simp only [bSwapFun, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- What the first route asks is what the second is asked.

    (u : ℕ)                                        [a position]
    bSwapFun (outPos u) = innPos u                 [asking becomes answering] -/
theorem bSwapFun_out (u : ℕ) : bSwapFun (outPos u) = innPos u := by
  rw [outPos_eq, bSwapFun_arg, flipPos_ext, innPos_eq]

/-- What the second route answers is what the first receives.

    (u : ℕ)                                        [a position]
    bSwapFun (innPos u) = outPos u                 [answering becomes asking] -/
theorem bSwapFun_inn (u : ℕ) : bSwapFun (innPos u) = outPos u := by
  rw [innPos_eq, bSwapFun_arg, flipPos_arg, outPos_eq]

/-- The wire is its own inverse.

    Function.Involutive bSwapFun                   [running it twice changes nothing] -/
theorem bSwapFun_involutive : Function.Involutive bSwapFun := by
  intro p
  rcases pos_split2 p with ⟨k, rfl⟩ | ⟨q, rfl⟩
  · rw [bSwapFun_ext, bSwapFun_ext]
  · rcases pos_split2 q with ⟨u, rfl⟩ | ⟨v, rfl⟩
    · rw [← outPos_eq, bSwapFun_out, bSwapFun_inn]
    · rw [← innPos_eq, bSwapFun_inn, bSwapFun_out]

/-- The wire as a route, weighted by the unit.

    Route R                                        [a weighted wire map over the ring]
    ringOne                                        [the point (1, 0), which changes nothing] -/
def swapRoute {R : Type u} [CommRing R] : RouteAnyRing.Route R :=
  fun i => some (bSwapFun i, AnyRing.ringOne)

/-- The wire leaves the outer argument track alone.

    ∀ s, s ∈ zSet → swapRoute s = some (s, ringOne) [it does nothing there] -/
theorem swapRoute_id_zSet {R : Type u} [CommRing R] :
    ∀ s, s ∈ zSet → swapRoute (R := R) s = some (s, AnyRing.ringOne) := by
  rintro s ⟨v, rfl⟩
  rw [swapRoute, zPos, bSwapFun_ext]

/-- The wire cannot bring a path onto the outer argument track.

    ∀ m j w, swapRoute m = some (j, w) → j ∈ zSet → m ∈ zSet [it does not enter] -/
theorem swapRoute_not_enter_zSet {R : Type u} [CommRing R] :
    ∀ m j w, swapRoute (R := R) m = some (j, w) → j ∈ zSet → m ∈ zSet := by
  intro m j w hm hj
  rw [swapRoute] at hm
  injection hm with hq
  injection hq with h1 _
  subst h1
  obtain ⟨v, hv⟩ := hj
  rw [← bSwapFun_involutive m, hv, zPos, bSwapFun_ext]
  exact mem_zSet v

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO ROUTES SIDE BY SIDE, AND THE WIRE BETWEEN THEM
-- ═══════════════════════════════════════════════════════════════════

/-- Two routes laid on the four tracks, the first on the outer pair, the second on the inner.

    (x y : Route R)                                [two weighted wire maps over the ring]
    Route R                                        [both at once, on one wire] -/
def pairRoute {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) :
    RouteAnyRing.Route R := fun i =>
  Sum.elim
    (fun m => Sum.elim (fun k => (x (extPos k)).map fun q => (xPos q.1, q.2))
      (fun v => (y (argPos v)).map fun q => (yPos q.1, q.2))
      (Equiv.natSumNatEquivNat.symm m))
    (fun m => Sum.elim (fun u => (x (argPos u)).map fun q => (xPos q.1, q.2))
      (fun u => (y (extPos u)).map fun q => (yPos q.1, q.2))
      (Equiv.natSumNatEquivNat.symm m))
    (Equiv.natSumNatEquivNat.symm i)

/-- On the outside the first route runs.

    (x y : Route R) (k : ℕ)                        [two routes and a position]
    pairRoute x y (extPos (extPos k))              [a step on the outside]
    = (x (extPos k)).map fun q => (xPos q.1, q.2)  [is the first route, laid out] -/
theorem pairRoute_ee {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (k : ℕ) :
    pairRoute x y (extPos (extPos k)) = (x (extPos k)).map fun q => (xPos q.1, q.2) := by
  simp only [pairRoute, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- On the argument track the second route runs.

    (x y : Route R) (v : ℕ)                        [two routes and a position]
    pairRoute x y (extPos (argPos v))              [a step on the argument track]
    = (y (argPos v)).map fun q => (yPos q.1, q.2)  [is the second route, laid out] -/
theorem pairRoute_ea {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (v : ℕ) :
    pairRoute x y (extPos (argPos v)) = (y (argPos v)).map fun q => (yPos q.1, q.2) := by
  simp only [pairRoute, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- On the asking track the first route runs.

    (x y : Route R) (u : ℕ)                        [two routes and a position]
    pairRoute x y (argPos (extPos u))              [a step on the asking track]
    = (x (argPos u)).map fun q => (xPos q.1, q.2)  [is the first route, laid out] -/
theorem pairRoute_ae {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (u : ℕ) :
    pairRoute x y (argPos (extPos u)) = (x (argPos u)).map fun q => (xPos q.1, q.2) := by
  simp only [pairRoute, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- On the answering track the second route runs.

    (x y : Route R) (u : ℕ)                        [two routes and a position]
    pairRoute x y (argPos (argPos u))              [a step on the answering track]
    = (y (extPos u)).map fun q => (yPos q.1, q.2)  [is the second route, laid out] -/
theorem pairRoute_aa {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (u : ℕ) :
    pairRoute x y (argPos (argPos u)) = (y (extPos u)).map fun q => (yPos q.1, q.2) := by
  simp only [pairRoute, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- **The pair is the first route on its own two tracks.**

    (x y : Route R) (p : ℕ)                        [two routes and a position]
    pairRoute x y (xPos p)                         [a step where the first route sits]
    = (x p).map fun q => (xPos q.1, q.2)           [is that route, laid out] -/
theorem pairRoute_x {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (p : ℕ) :
    pairRoute x y (xPos p) = (x p).map fun q => (xPos q.1, q.2) := by
  rcases pos_split2 p with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · rw [xPos_ext, pairRoute_ee]
  · rw [xPos_arg, outPos_eq, pairRoute_ae]

/-- **The pair is the second route on its own two tracks.**

    (x y : Route R) (p : ℕ)                        [two routes and a position]
    pairRoute x y (yPos p)                         [a step where the second route sits]
    = (y p).map fun q => (yPos q.1, q.2)           [is that route, laid out] -/
theorem pairRoute_y {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (p : ℕ) :
    pairRoute x y (yPos p) = (y p).map fun q => (yPos q.1, q.2) := by
  rcases pos_split2 p with ⟨u, rfl⟩ | ⟨v, rfl⟩
  · rw [yPos_ext, innPos_eq, pairRoute_aa]
  · rw [yPos_arg, zPos, pairRoute_ea]

/-- **The stored route: two routes side by side with the wire between them.**

@cite girard-1989

    (σ : R) (x y : Route R)                        [the parameter and two routes over the ring]
    rcomp σ (pairRoute x y) swapRoute              [run whichever the position belongs to, then
    rewire]                                        [so a question becomes the other's input]
    : Route R                                      [one route on the combined wire]

    So: this is what a composing combinator has to store. It is not a relabelling of one route,
    which is all `I`, `K` and the contraction needed, but two routes with a wire joining them,
    and that wire is the loop the stored route traces out. -/
def bStore {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R) :
    RouteAnyRing.Route R :=
  RouteAnyRing.rcomp σ (pairRoute x y) (swapRoute (R := R))

/-- The unrestricted relation realized by the stored composition route.

    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    Route R [a weighted wire map over the ring]
    Traces [the union over every path length through a route]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    j [the final position] -/
def BCompose {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R)
    (i j : ℕ) (w : AnyRing.RingCell R) : Prop :=
  RouteAnyRing.Traces σ (bStore σ x y) i j w

/-- A stored-composition relation holds exactly when some finite trace search finds it.

    BCompose [the unrestricted stored-composition relation]
    bStore [two routes side by side with the wire between them]
    traceWithin [the bounded search for the trace]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ↔ [exactly when, true in both directions]
    j [the final position] -/
theorem bCompose_iff_traceWithin {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R)
    (i j : ℕ) (w : AnyRing.RingCell R) :
    BCompose σ x y i j w ↔
      ∃ fuel, RouteAnyRing.traceWithin σ (bStore σ x y) fuel i = .found (j, w) :=
  (RouteAnyRing.traceWithin_iff_traces σ (bStore σ x y) i j w).symm

/-- Materialize a stored composition from an explicitly complete trace budget.

    TraceBudget [an input-indexed finite-search budget complete for every trace]
    traceRouteOfBudget [materialize a route by evaluating an explicit trace budget]
    Route R [a weighted wire map over the ring]
    bStore [two routes side by side with the wire between them]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above] -/
def bCombOfBudget {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R)
    (b : BoundedRoutes.TraceBudget σ (bStore σ x y)) : RouteAnyRing.Route R :=
  BoundedRoutes.traceRouteOfBudget σ (bStore σ x y) b

/-- A supplied complete budget materializes exactly the stored-composition relation.

    BCompose [the unrestricted stored-composition relation]
    bCombOfBudget [the route materialized from a supplied complete budget]
    bStore [two routes side by side with the wire between them]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every]
    ↔ [exactly when, true in both directions]
    j [the final position] -/
theorem bCompose_iff_bCombOfBudget {R : Type u} [CommRing R] (σ : R)
    (x y : RouteAnyRing.Route R) (b : BoundedRoutes.TraceBudget σ (bStore σ x y))
    (i j : ℕ) (w : AnyRing.RingCell R) :
    BCompose σ x y i j w ↔ bCombOfBudget σ x y b i = some (j, w) :=
  BoundedRoutes.traces_iff_traceRouteOfBudget σ (bStore σ x y) b i j w

/-- **The composite of two routes, as a route on the composite's own wire.**

@cite girard-1989
@cite curry-1930

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    traceRoute σ (bStore σ x y)                    [collapse the wire between them]
    : Route                                        [what is left is the composite]

    So: the layout is chosen so that no relabelling stands between the trace and the composite's
    wire. The two middle tracks are odd, so the trace collapses them, and the outer two are even,
    so they survive as the composite's outside and its argument channel. -/
def bComb (σ : ℝ) (x y : Route) : Route := traceRoute σ (bStore σ x y)

-- ═══════════════════════════════════════════════════════════════════
-- FEEDING THE OUTER ARGUMENT
-- ═══════════════════════════════════════════════════════════════════

/-- The outer argument, laid on the fourth track alone.

    (z : Route R)                                  [a weighted wire map over the ring]
    rtensor (rtensor rid z) rid                    [act on the argument track and nowhere else]
    : Route R                                      [the argument layer] -/
def bArg {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) : RouteAnyRing.Route R :=
  RouteAnyRing.rtensor (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z)
    (RouteAnyRing.rid (R := R))

/-- The argument layer on an even position.

    (z : Route R) (p : ℕ)                          [a route and a position]
    bArg z (extPos p)                              [a step on the even half]
    = ((rtensor rid z) p).map fun q => (extPos q.1, q.2)   [is the plug's own argument layer] -/
theorem bArg_ext {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) (p : ℕ) :
    bArg z (extPos p) =
      ((RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z) p).map fun q => (extPos q.1, q.2) :=
  rtensor_ext _ _ p

/-- The argument layer does nothing on the odd half.

    (z : Route R) (q : ℕ)                          [a route and a position]
    bArg z (argPos q) = some (argPos q, ringOne)   [the two middle tracks pass through] -/
theorem bArg_arg {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) (q : ℕ) :
    bArg z (argPos q) = some (argPos q, AnyRing.ringOne) := by
  rw [bArg, rtensor_arg, RouteAnyRing.rid, Option.map_some]

/-- The argument layer does nothing where the first route sits.

    (z : Route R) (p : ℕ)                          [a route and a position]
    bArg z (xPos p) = some (xPos p, ringOne)       [the first route's tracks pass through] -/
theorem bArg_xPos {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) (p : ℕ) :
    bArg z (xPos p) = some (xPos p, AnyRing.ringOne) := by
  rcases pos_split2 p with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · rw [xPos_ext, bArg_ext, rtensor_ext_id, Option.map_some]
  · rw [xPos_arg, outPos_eq, bArg_arg]

/-- The argument layer is the second route's own argument layer.

    (z : Route R) (p : ℕ)                          [a route and a position]
    bArg z (yPos p)                                [a step where the second route sits]
    = ((rtensor rid z) p).map fun q => (yPos q.1, q.2)   [is that route's argument layer] -/
theorem bArg_yPos {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) (p : ℕ) :
    bArg z (yPos p) =
      ((RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z) p).map fun q => (yPos q.1, q.2) := by
  rcases pos_split2 p with ⟨u, rfl⟩ | ⟨v, rfl⟩
  · rw [yPos_ext, innPos_eq, bArg_arg, rtensor_ext_id, Option.map_some, yPos_ext, innPos_eq]
  · rw [yPos_arg, zPos_eq, bArg_ext, rtensor_arg]
    cases hz : z v with
    | none => rfl
    | some q => rw [Option.map_some, Option.map_some, Option.map_some, yPos_arg, zPos_eq]

/-- The argument layer does nothing on a loop of the composite.

    (z : Route R)                                  [a route]
    ∀ s, s ∈ oddPos → bArg z s = some (s, ringOne) [the middle tracks pass through] -/
theorem bArg_id_oddPos {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) :
    ∀ s, s ∈ oddPos → bArg z s = some (s, AnyRing.ringOne) := by
  rintro s ⟨q, rfl⟩
  exact bArg_arg z q

/-- The argument layer cannot bring a path onto a loop of the composite.

    (z : Route R)                                            [a route]
    ∀ m j w, bArg z m = some (j, w) → j ∈ oddPos → m ∈ oddPos   [it does not enter] -/
theorem bArg_not_enter_oddPos {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) :
    ∀ m j w, bArg z m = some (j, w) → j ∈ oddPos → m ∈ oddPos := by
  intro m j w hm hj
  rcases pos_split2 m with ⟨k, rfl⟩ | ⟨q, rfl⟩
  · rw [bArg_ext] at hm
    obtain ⟨⟨c1, c2⟩, _, hq⟩ := Option.map_eq_some_iff.mp hm
    simp only [Prod.mk.injEq] at hq
    obtain ⟨h1, _⟩ := hq
    subst h1
    exact absurd hj (not_mem_oddPos c1)
  · exact ⟨q, rfl⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE WHOLE MACHINE
-- ═══════════════════════════════════════════════════════════════════

/-- The pair with the outer argument, before the wire.

    (σ : R) (x y z : Route R)                       [the parameter and three routes over the ring]
    rcomp σ (pairRoute x y) (bArg z)               [both routes, then the argument]
    : Route R                                      [the machine without its wire] -/
def bPlug {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) :
    RouteAnyRing.Route R :=
  RouteAnyRing.rcomp σ (pairRoute x y) (bArg z)

/-- **The whole machine: both routes, the wire, and the outer argument.**

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (σ : R) (x y z : Route R)                      [the parameter and three routes over the ring]
    rcomp σ (bStore σ x y) (bArg z)                [the stored route with its argument plugged]
    : Route R                                      [one route carrying every loop at once]

    So: both readings of `B x y z = x (y z)` are paths of this one route, differing in which
    loops are collapsed first. The left side collapses the wire and then the argument channel,
    the right side collapses the argument channel and the answering track and then the asking
    track, and the three loops are the same three either way. -/
def bBig {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) :
    RouteAnyRing.Route R :=
  RouteAnyRing.rcomp σ (bStore σ x y) (bArg z)

/-- The wire and the argument layer commute.

    (σ : R) (z : Route R)                          [the parameter and a route over the ring]
    rcomp σ swapRoute (bArg z) = rcomp σ (bArg z) swapRoute   [either order gives the same]

    So: the wire moves only the two middle tracks and the argument layer acts only on the fourth,
    so neither sees what the other does. That is what lets the machine be read with the wire
    last, which is the reading the composite needs, or with the wire first, which is the reading
    the argument needs. -/
theorem swap_arg_comm {R : Type u} [CommRing R] (σ : R) (z : RouteAnyRing.Route R) :
    RouteAnyRing.rcomp σ (swapRoute (R := R)) (bArg z) =
      RouteAnyRing.rcomp σ (bArg z) (swapRoute (R := R)) := by
  funext i
  rcases pos_split i with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨q, rfl⟩
  · have h1 : bArg z (extPos (extPos k)) =
      some (extPos (extPos k), AnyRing.ringOne) := by
      rw [bArg_ext, rtensor_ext_id, Option.map_some]
    rw [RouteAnyRing.rcomp, RouteAnyRing.rcomp, swapRoute, bSwapFun_ext, Option.bind_some, h1,
      Option.map_some, Option.bind_some, swapRoute, bSwapFun_ext, Option.map_some]
  · have h1 : bArg z (extPos (argPos v)) = (z v).map fun r => (extPos (argPos r.1), r.2) := by
      rw [bArg_ext, rtensor_arg]
      cases z v <;> rfl
    rw [RouteAnyRing.rcomp, RouteAnyRing.rcomp, swapRoute, bSwapFun_ext, Option.bind_some, h1]
    cases hz : z v with
    | none => rfl
    | some r =>
      simp only [Option.map_some, Option.bind_some, swapRoute, bSwapFun_ext,
        AnyRing.ringOne_mul, AnyRing.ringMul_one]
  · rw [RouteAnyRing.rcomp, RouteAnyRing.rcomp, swapRoute, bSwapFun_arg, Option.bind_some,
      bArg_arg, Option.map_some, bArg_arg, Option.bind_some, swapRoute, bSwapFun_arg,
      Option.map_some]

/-- **The machine with its wire last.**

    (σ : R) (x y z : Route R)                      [the parameter and three routes over the ring]
    bBig σ x y z = rcomp σ (bPlug σ x y z) swapRoute   [both routes, the argument, then the wire]

    So: with the wire last, `runs_rcomp_id_on` applies to it wherever the wire fixes the loop
    being collapsed, which is what the right hand reading needs on the argument channel. -/
theorem bBig_eq {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) :
    bBig σ x y z = RouteAnyRing.rcomp σ (bPlug σ x y z) (swapRoute (R := R)) := by
  rw [bBig, bStore, bPlug, RouteAnyRing.rcomp_assoc, swap_arg_comm, RouteAnyRing.rcomp_assoc]

/-- The machine on the first route's tracks.

    (σ : R) (x y z : Route R) (p : ℕ)              [the parameter, three routes, a position]
    bPlug σ x y z (xPos p)                         [a step where the first route sits]
    = (x p).map fun q => (xPos q.1, q.2)           [is that route, laid out] -/
theorem bPlug_x {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) (p : ℕ) :
    bPlug σ x y z (xPos p) = (x p).map fun q => (xPos q.1, q.2) := by
  have h2 : ∀ a, bArg z (xPos a) =
      (RouteAnyRing.rid (R := R) a).map fun q => (xPos q.1, q.2) := by
    intro a
    rw [bArg_xPos, RouteAnyRing.rid, Option.map_some]
  rw [bPlug, rcomp_embed (pairRoute_x x y) h2 p, RouteAnyRing.rcomp_rid_right]

/-- The machine on the second route's tracks is that route with its argument plugged.

    (σ : R) (x y z : Route R) (p : ℕ)              [the parameter, three routes, a position]
    bPlug σ x y z (yPos p)                         [a step where the second route sits]
    = (ApplicationWiring.plug σ y z p).map fun q => (yPos q.1, q.2)
                                                        [is the relation-first plug laid out]

    So: the second route and the outer argument sit on the machine exactly as they sit in their
    own `ApplicationWiring.plug σ y z`, with no wire between them. Thus every relation-first
    trace of that plug transports to the second-route tracks. -/
theorem bPlug_y {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) (p : ℕ) :
    bPlug σ x y z (yPos p) = (ApplicationWiring.plug σ y z p).map fun q => (yPos q.1, q.2) := by
  rw [bPlug, rcomp_embed (pairRoute_y x y) (bArg_yPos z) p, ApplicationWiring.plug]

end Combinator
end Apeiron

end
