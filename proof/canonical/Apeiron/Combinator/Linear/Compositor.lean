import Apeiron.Combinator.Linear.Composition
import Apeiron.Combinator.RelationalApplication

/-!
# The composing combinator

@name Compositor
@cite girard-1989
@cite schonfinkel-1924
@cite curry-1930
@cite joyal-street-verity-1996
@grade proved

`B x y z = x (y z)` at the carrier of `Combinator/Interaction.lean`, where a route is a weighted
partial injection of the counting numbers and application is the trace. The stored route is
`bComb`, built in `Combinator/Composition.lean`, and the equation is `bracket` below.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | `relFed_x`, `relFed_inn`, `relApplied_step` |

The wiring is free of the parameter for the reason it has been free since
`Combinator/Interaction.lean`: a partial injection carries at most one path out of each position,
so composing multiplies one weight along one path. The parameter is the multiplication that
weights a path, and where it decides anything is stated in
`Combinator/Interaction.lean`'s `weight_blocks_the_loop_iff_disc_nonneg` and in
`Trichotomy/Trace.lean`'s reading of the three faces.

## The argument

Both sides of `B x y z = x (y z)` are readings of one route, `bBig`, which carries all three
loops at once: the wire between the two stored routes, the answering track, and the outer
argument's channel. They differ in the order the loops are collapsed.

**The left side** collapses the wire first, which is what `bComb` is, and then the argument's
channel, which is what applying `bComb` to `z` does. `left_reading` says so.

**The right side** collapses the argument's channel first, then the answering track, which
together are `rapp σ y z` sitting inside the machine, and then the asking track, which is the
outer application. `right_reading` says so, and `relApplied_step` records the corresponding
relation: after the first two collapses, a step from the first route's track either leaves at once
or crosses to the answering track and comes straight back.

Since the three loops are pairwise disjoint and their union is the same either way, the two
readings agree, which is `bracket`.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy
open RouteAnyRing

universe u

-- ═══════════════════════════════════════════════════════════════════
-- WHERE THE FIRST ROUTE SENDS A PATH
-- ═══════════════════════════════════════════════════════════════════

/-- Where the first route's step lands after the wire.

    (q : ℕ)                                        [one of the first route's positions]
    bSwapFun (xPos q)                              [its place, then the wire]
    : ℕ                                            [a position of the combined wire] -/
def xOut (q : ℕ) : ℕ := bSwapFun (xPos q)

/-- The first route's outside is the composite's outside.

    (k : ℕ)                                        [a position]
    xOut (extPos k) = extPos (extPos k)            [an exit stays on the outside] -/
theorem xOut_ext (k : ℕ) : xOut (extPos k) = extPos (extPos k) := by
  rw [xOut, xPos_ext, bSwapFun_ext]

/-- The first route's question reaches the second route.

    (u : ℕ)                                        [a position]
    xOut (argPos u) = innPos u                     [asking crosses to the answering track] -/
theorem xOut_arg (u : ℕ) : xOut (argPos u) = innPos u := by
  rw [xOut, xPos_arg, bSwapFun_out]

/-- A step that does not cross was an exit.

    {q : ℕ} (h : xOut q ∉ innerSet)                [a step off the answering track]
    ∃ k, q = extPos k                              [came from the first route's outside] -/
theorem xOut_not_inner {q : ℕ} (h : xOut q ∉ innerSet) : ∃ k, q = extPos k := by
  rcases pos_split2 q with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · exact ⟨k, rfl⟩
  · rw [xOut_arg] at h
    exact absurd (mem_innerSet u) h

/-- A step that crosses was a question.

    {q : ℕ} (h : xOut q ∈ innerSet)                [a step onto the answering track]
    ∃ u, q = argPos u                              [came from the first route's argument] -/
theorem xOut_inner {q : ℕ} (h : xOut q ∈ innerSet) : ∃ u, q = argPos u := by
  rcases pos_split2 q with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · rw [xOut_ext] at h
    exact absurd h (extPos_not_mem (extPos k)).2
  · exact ⟨u, rfl⟩

-- ═══════════════════════════════════════════════════════════════════
-- ONE STEP OF AN APPLICATION
-- ═══════════════════════════════════════════════════════════════════

/-- A step that leaves is a step of the application.

    (σ : ℝ) (a b : Route) {p k : ℕ} {w : Cell}     [parameter, two routes, a step]
    (h : a p = some (extPos k, w))                 [the route leaves]
    plug σ a b p = some (extPos k, w)              [and the argument is not consulted] -/
theorem plug_ext_step (σ : ℝ) (a b : Route) {p k : ℕ} {w : Cell}
    (h : a p = some (extPos k, w)) : plug σ a b p = some (extPos k, w) := by
  rw [plug, rcomp_eq_some]
  exact ⟨extPos k, w, one, h, rtensor_ext_id b k, (mul_one_right σ w).symm⟩

/-- A step that asks is a step of the application, with the argument's answer.

    (σ : ℝ) (a b : Route) {p u u' : ℕ} {w₁ w₂ : Cell}   [parameter, two routes, a step]
    (h : a p = some (argPos u, w₁))                     [the route asks]
    (hb : b u = some (u', w₂))                          [and the argument answers]
    plug σ a b p = some (argPos u', mul σ w₁ w₂)        [one step of the application] -/
theorem plug_arg_step (σ : ℝ) (a b : Route) {p u u' : ℕ} {w₁ w₂ : Cell}
    (h : a p = some (argPos u, w₁)) (hb : b u = some (u', w₂)) :
    plug σ a b p = some (argPos u', mul σ w₁ w₂) := by
  rw [plug, rcomp_eq_some]
  exact ⟨argPos u, w₁, w₂, h, by rw [rtensor_arg, hb, Option.map_some], rfl⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE MACHINE WITH THE ARGUMENT'S CHANNEL COLLAPSED
-- ═══════════════════════════════════════════════════════════════════

/-- A machine after the argument's own channel is collapsed.

    (σ : ℝ) (M : Route)                            [the parameter and a machine]
    runsRoute σ (rcomp σ M swapRoute) zSet         [wire it, then collapse the argument channel]
    : Route                                        [what is left carries the two other loops] -/
def fedRoute (σ : ℝ) (M : Route) : Route := runsRoute σ (rcomp σ M swapRoute) zSet

/-- The relational machine after its argument channel is collapsed.

    FedRel [an unbounded weighted relation after the channel is collapsed]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rcomp [follow one route and then another, multiplying the weights]
    swapRoute [the middle-track wire exchange]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    zSet [the argument channel positions]
    ℕ [the counting numbers 0, 1, 2, 3, and so on] -/
def FedRel {R : Type u} [CommRing R] (σ : R) (M : RouteAnyRing.Route R) :
    RelationalPaths.StepRel ℕ R :=
  RelationalPaths.RelRuns σ
    (RelationalPaths.routeGraph (RouteAnyRing.rcomp σ M (swapRoute (R := R)))) zSet

/-- A collapsed relational machine path is a machine path followed by the middle-track swap.

    FedRel [the relational machine after its argument channel is collapsed]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rcomp [follow one route and then another, multiplying the weights]
    swapRoute [the middle-track wire exchange]
    bSwapFun [the position map which exchanges the middle tracks]
    ringOne [the point (1, 0), which changes nothing when you multiply by it]
    ringMul [multiply two points together, the σ way]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    zSet [the argument channel positions]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    i [the initial position]
    c [the final position]
    w [the accumulated path weight] -/
theorem fedRel_iff {R : Type u} [CommRing R] (σ : R) (M : RouteAnyRing.Route R)
    (i c : ℕ) (w : AnyRing.RingCell R) :
    FedRel σ M i c w ↔ ∃ m,
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph M) zSet i m w ∧ c = bSwapFun m := by
  rw [FedRel, relRuns_rcomp_id swapRoute_id_zSet swapRoute_not_enter_zSet]
  constructor
  · rintro ⟨m, u, v, hM, hswap, hw⟩
    rw [swapRoute] at hswap
    simp only [Option.some.injEq, Prod.mk.injEq] at hswap
    obtain ⟨hc, hv⟩ := hswap
    subst v
    refine ⟨m, ?_, hc.symm⟩
    rw [hw, AnyRing.ringMul_one]
    exact hM
  · rintro ⟨m, hM, rfl⟩
    refine ⟨m, w, AnyRing.ringOne, hM, ?_, ?_⟩
    · simp only [swapRoute]
    · exact (AnyRing.ringMul_one σ w).symm

/-- The fed relation on the first track is the supplied relation followed by the middle swap.

    FedRel [the relational machine after its argument channel is collapsed]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    xPos [the first route's position in the machine]
    xOut [the first route's output after the middle swap]
    zSet [the argument channel positions]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    p [the source position]
    c [the final position]
    w [the accumulated path weight] -/
theorem relFed_x {R : Type u} [CommRing R] (σ : R) (M : RouteAnyRing.Route R)
    (a : RelationalPaths.StepRel ℕ R)
    (hx : ∀ p c w, RelationalPaths.RelRuns σ (RelationalPaths.routeGraph M) zSet (xPos p) c w ↔
      ∃ q, c = xPos q ∧ a p q w)
    (p c : ℕ) (w : AnyRing.RingCell R) :
    FedRel σ M (xPos p) c w ↔ ∃ q, c = xOut q ∧ a p q w := by
  rw [fedRel_iff]
  constructor
  · rintro ⟨m, hM, hc⟩
    obtain ⟨q, hm, ha⟩ := (hx p m w).mp hM
    subst m
    exact ⟨q, by rw [hc, xOut], ha⟩
  · rintro ⟨q, hc, ha⟩
    refine ⟨xPos q, (hx p (xPos q) w).mpr ⟨q, rfl, ha⟩, ?_⟩
    rw [hc, xOut]

/-- The fed relation on the answering track is the supplied relation followed by the middle swap.

    FedRel [the relational machine after its argument channel is collapsed]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    innPos [the answering-track position]
    outPos [the asking-track position]
    zSet [the argument channel positions]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    n [the source position]
    c [the final position]
    w [the accumulated path weight] -/
theorem relFed_inn {R : Type u} [CommRing R] (σ : R) (M : RouteAnyRing.Route R)
    (b : RelationalPaths.StepRel ℕ R)
    (hy : ∀ n c w, RelationalPaths.RelRuns σ (RelationalPaths.routeGraph M) zSet (innPos n) c w ↔
      ∃ t, c = innPos t ∧ b n t w)
    (n c : ℕ) (w : AnyRing.RingCell R) :
    FedRel σ M (innPos n) c w ↔ ∃ t, c = outPos t ∧ b n t w := by
  rw [fedRel_iff]
  constructor
  · rintro ⟨m, hM, hc⟩
    obtain ⟨t, hm, hb⟩ := (hy n m w).mp hM
    subst m
    exact ⟨t, by rw [hc, bSwapFun_inn], hb⟩
  · rintro ⟨t, hc, hb⟩
    refine ⟨innPos t, (hy n (innPos t) w).mpr ⟨t, rfl, hb⟩, ?_⟩
    rw [hc, bSwapFun_inn]

/-- A relational path through the fed machine is one relation-first application step.

    RelRuns [an unbounded finite path]
    FedRel [the relational machine after its argument channel is collapsed]
    RelPlug [one relation-first application step]
    routeGraph [the transition relation presented by a partial weighted route]
    xPos [the first route's position in the machine]
    xOut [the first route's output after the middle swap]
    innPos [the answering-track position]
    outPos [the asking-track position]
    innerSet [the answering-track loop positions]
    zSet [the argument channel positions]
    StepRel [a weighted constructive transition relation]
    ringMul [multiply two points together, the σ way]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    p [the source position]
    c [the final position]
    w [the accumulated path weight] -/
theorem relApplied_step {R : Type u} [CommRing R] (σ : R) (M : RouteAnyRing.Route R)
    (a b : RelationalPaths.StepRel ℕ R)
    (hx : ∀ p c w, RelationalPaths.RelRuns σ (RelationalPaths.routeGraph M) zSet (xPos p) c w ↔
      ∃ q, c = xPos q ∧ a p q w)
    (hy : ∀ n c w, RelationalPaths.RelRuns σ (RelationalPaths.routeGraph M) zSet (innPos n) c w ↔
      ∃ t, c = innPos t ∧ b n t w)
    (p c : ℕ) (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (FedRel σ M) innerSet (xPos p) c w ↔
      ∃ q, RelPlug σ a b p q w ∧ c = xPos q := by
  constructor
  · rintro ⟨n, hrun⟩
    cases hrun with
    | exit hEdge hOutside =>
      obtain ⟨q, hc, ha⟩ := (relFed_x σ M a hx p _ _).mp hEdge
      rcases pos_split2 q with ⟨k, rfl⟩ | ⟨u, rfl⟩
      · refine ⟨extPos k, ?_, ?_⟩
        · left
          exact ⟨k, ha, rfl⟩
        · rw [hc, xOut_ext, xPos_ext]
      · rw [xOut_arg] at hc
        exact False.elim (hOutside (hc ▸ mem_innerSet u))
    | step hEdge hInside hrest =>
      obtain ⟨q, hc, ha⟩ := (relFed_x σ M a hx p _ _).mp hEdge
      rcases pos_split2 q with ⟨k, rfl⟩ | ⟨u, rfl⟩
      · rw [xOut_ext] at hc
        exact False.elim ((extPos_not_mem (extPos k)).2 (hc.symm ▸ hInside))
      · rw [xOut_arg] at hc
        rw [hc] at hrest
        cases hrest with
        | exit hEdge' _ =>
          obtain ⟨t, hc', hb⟩ := (relFed_inn σ M b hy u _ _).mp hEdge'
          refine ⟨argPos t, ?_, ?_⟩
          · right
            refine ⟨u, t, _, _, ?_, hb, rfl, rfl⟩
            exact ha
          · rw [hc', xPos_arg]
        | step hEdge' hInside' _ =>
          obtain ⟨t, hc', _⟩ := (relFed_inn σ M b hy u _ _).mp hEdge'
          exact False.elim (outer_disjoint_inner _ (mem_outerSet t) (hc'.symm ▸ hInside'))
  · rintro ⟨q, hstep, rfl⟩
    cases hstep with
    | inl hExt =>
      obtain ⟨k, ha, hq⟩ := hExt
      subst q
      refine ⟨0, .exit ?_ ?_⟩
      · apply (relFed_x σ M a hx p _ _).mpr
        refine ⟨extPos k, ?_, ha⟩
        change xPos (extPos k) = xOut (extPos k)
        rw [xPos_ext, xOut_ext]
      · change xPos (extPos k) ∉ innerSet
        rw [xPos_ext]
        exact (extPos_not_mem (extPos k)).2
    | inr hQuery =>
      obtain ⟨u, v, w₁, w₂, ha, hb, hq, hw⟩ := hQuery
      subst q
      subst w
      refine ⟨1, .step ?_ (mem_innerSet u) (.exit ?_ ?_)⟩
      · apply (relFed_x σ M a hx p _ _).mpr
        refine ⟨argPos u, ?_, ha⟩
        rw [xOut_arg]
      · apply (relFed_inn σ M b hy u _ _).mpr
        refine ⟨v, ?_, hb⟩
        change xPos (argPos v) = outPos v
        rw [xPos_arg]
      · change xPos (argPos v) ∉ innerSet
        rw [xPos_arg]
        exact outer_disjoint_inner _ (mem_outerSet v)

/-- The answering track does not meet the argument's channel.

    ∀ x, x ∈ innerSet → x ∉ zSet                   [one is odd, the other even] -/
theorem inner_disjoint_zSet : ∀ x, x ∈ innerSet → x ∉ zSet := by
  rintro x ⟨v, rfl⟩ ⟨v', hv⟩
  rw [innPos_eq, zPos_eq] at hv
  exact extPos_ne_argPos (argPos v') (argPos v) hv.symm

/-- The asking track meets neither of the other two loops.

    ∀ x, x ∈ outerSet → x ∉ innerSet ∪ zSet        [the three loops are pairwise apart] -/
theorem outer_disjoint_rest : ∀ x, x ∈ outerSet → x ∉ innerSet ∪ zSet := by
  rintro x hx (hi | ⟨v, hv⟩)
  · exact outer_disjoint_inner x hx hi
  · obtain ⟨u, rfl⟩ := hx
    rw [outPos_eq, zPos_eq] at hv
    exact extPos_ne_argPos (argPos v) (extPos u) hv.symm

/-- The three loops are the same three either way.

    zSet ∪ oddPos = outerSet ∪ (innerSet ∪ zSet)   [the wire, the answering track, the argument] -/
theorem bracket_sets : zSet ∪ oddPos = outerSet ∪ (innerSet ∪ zSet) := by
  rw [oddPos_eq]
  ext p
  constructor
  · rintro (hz | ho | hi)
    · exact Or.inr (Or.inr hz)
    · exact Or.inl ho
    · exact Or.inr (Or.inl hi)
  · rintro (ho | hi | hz)
    · exact Or.inr (Or.inl ho)
    · exact Or.inr (Or.inr hi)
    · exact Or.inl hz

/-- The relation-first application is the three-loop reading of the wired machine.

    RelApp [the relation-first application relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rcomp [compose weighted routes]
    swapRoute [the middle-track exchange]
    outerSet [the asking-track loop positions]
    innerSet [the answering-track loop positions]
    zSet [the argument channel positions]
    xPos [the first route's position on the combined wire]
    innPos [the stored route's answering-track position]
    extPos [the external wire position]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relApplied_reading {R : Type u} [CommRing R] (σ : R) (M : RouteAnyRing.Route R)
    (a b : RelationalPaths.StepRel ℕ R)
    (hx : ∀ p c w, RelationalPaths.RelRuns σ (RelationalPaths.routeGraph M) zSet (xPos p) c w ↔
      ∃ q, c = xPos q ∧ a p q w)
    (hy : ∀ n c w, RelationalPaths.RelRuns σ (RelationalPaths.routeGraph M) zSet (innPos n) c w ↔
      ∃ t, c = innPos t ∧ b n t w)
    (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ a b i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph
        (RouteAnyRing.rcomp σ M (swapRoute (R := R))))
        (outerSet ∪ (innerSet ∪ zSet))
        (extPos (extPos i)) (extPos (extPos j)) w := by
  have hInner : RelationalPaths.RelRuns σ (FedRel σ M) innerSet = RelationalPaths.RelRuns σ
      (RelationalPaths.routeGraph (RouteAnyRing.rcomp σ M (swapRoute (R := R))))
      (innerSet ∪ zSet) := by
    funext p q z
    apply propext
    change RelationalPaths.RelRuns σ
      (RelationalPaths.RelRuns σ (RelationalPaths.routeGraph
        (RouteAnyRing.rcomp σ M (swapRoute (R := R)))) zSet) innerSet p q z ↔ _
    exact (RelationalPaths.relRuns_vanishing inner_disjoint_zSet p q z).symm
  have hStep : ∀ p c z, RelationalPaths.RelRuns σ (FedRel σ M) innerSet (xPos p) c z ↔
      ∃ q, c = xPos q ∧ RelPlug σ a b p q z := by
    intro p c z
    constructor
    · rintro h
      obtain ⟨q, hq, hc⟩ := (relApplied_step σ M a b hx hy p c z).mp h
      exact ⟨q, hc, hq⟩
    · rintro ⟨q, hc, hq⟩
      exact (relApplied_step σ M a b hx hy p c z).mpr ⟨q, hq, hc⟩
  rw [RelApp, ← RelationalPaths.relRuns_embed_iff xPos_injective hStep]
  change RelationalPaths.RelRuns σ (RelationalPaths.RelRuns σ (FedRel σ M) innerSet)
    (xPos '' oddPos) (xPos (extPos i)) (xPos (extPos j)) w ↔ _
  rw [xPos_image_oddPos, xPos_ext, xPos_ext, hInner,
    ← RelationalPaths.relRuns_vanishing outer_disjoint_rest]

/-- Reading a collapsed stored path through the outer argument relation.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rcomp [compose weighted routes]
    bArg [the fourth-track argument layer]
    RelPlug [one relation-first application step]
    Traces [the union over finite paths through a route]
    rtensor [put two routes side by side]
    ringMul [multiply two points together, the σ way]
    oddPos [the odd loop positions]
    extPos [the external wire position]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    ∨ [either statement holds]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    i [the source position]
    j [the collapsed application output]
    c [the final position]
    w [the accumulated path weight] -/
private theorem relPlug_graph_right_iff {R : Type u} [CommRing R] (σ : R)
    (g z : RouteAnyRing.Route R) (i c : ℕ) (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph
      (RouteAnyRing.rcomp σ g (bArg z))) oddPos (extPos i) c w ↔
      ∃ j, c = extPos j ∧ RelPlug σ (RouteAnyRing.Traces σ g)
        (RelationalPaths.routeGraph z) i j w := by
  rw [relRuns_rcomp_id (bArg_id_oddPos z) (bArg_not_enter_oddPos z)]
  constructor
  · rintro ⟨m, u, v, hm, hb, hw⟩
    obtain ⟨k, hm, htrace⟩ := (relRuns_oddPos_iff_traces i m u).mp hm
    subst m
    rcases pos_split2 k with ⟨l, rfl⟩ | ⟨q, rfl⟩
    · rw [bArg_ext, rtensor_ext_id] at hb
      obtain ⟨hc, hv⟩ := Option.some.inj hb
      change RouteAnyRing.Traces σ g i (2 * l) u at htrace
      rw [AnyRing.ringMul_one] at hw
      subst w
      refine ⟨extPos l, rfl, Or.inl ⟨l, ?_, ?_⟩⟩
      · exact htrace
      · change Equiv.natSumNatEquivNat (Sum.inl l) = 2 * l
        rfl
    · rw [bArg_ext, rtensor_arg] at hb
      cases hz : z q with
      | none => simp [hz] at hb
      | some pair =>
        obtain ⟨t, wt⟩ := pair
        simp only [hz, Option.map_some, Option.some.injEq, Prod.mk.injEq] at hb
        obtain ⟨hc, hv⟩ := hb
        change RouteAnyRing.Traces σ g i (2 * q + 1) u at htrace
        have hw' : w = AnyRing.ringMul σ u wt := by
          rw [hv]
          exact hw
        refine ⟨argPos t, hc.symm, Or.inr ⟨q, t, u, wt, htrace, hz, ?_, hw'⟩⟩
        change Equiv.natSumNatEquivNat (Sum.inr t) = 2 * t + 1
        rfl
  · rintro ⟨j, hc, hstep⟩
    subst c
    cases hstep with
    | inl hExt =>
      obtain ⟨l, htrace, hj⟩ := hExt
      subst j
      change RouteAnyRing.Traces σ g i (extPos l) w at htrace
      refine ⟨extPos (extPos l), w, AnyRing.ringOne, ?_, ?_, ?_⟩
      · exact (relRuns_oddPos_iff_traces i (extPos (extPos l)) w).mpr
          ⟨extPos l, rfl, htrace⟩
      · rw [bArg_ext, rtensor_ext_id]
        change some (extPos (extPos l), AnyRing.ringOne) =
          some (extPos (extPos l), AnyRing.ringOne)
        rfl
      · rw [AnyRing.ringMul_one]
    | inr hArg =>
      obtain ⟨q, t, u, v, htrace, hz, hj, hw⟩ := hArg
      subst j
      subst w
      refine ⟨extPos (argPos q), u, v, ?_, ?_, rfl⟩
      · exact (relRuns_oddPos_iff_traces i (extPos (argPos q)) u).mpr
          ⟨argPos q, rfl, htrace⟩
      · rw [bArg_ext, rtensor_arg, hz, Option.map_some]
        change some (extPos (argPos t), v) = some (extPos (argPos t), v)
        rfl

/-- Applying an already-traced route is the two-loop reading of its outer-argument machine.

    RelApp [the relation-first application relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rcomp [compose weighted routes]
    bArg [the fourth-track argument layer]
    Traces [the union over finite paths through a route]
    zSet [the argument channel positions]
    oddPos [the odd loop positions]
    extPos [the external wire position]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∧ [both statements hold]
    ∪ [one of the two loop layers]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relOuter_reading {R : Type u} [CommRing R] (σ : R)
    (g z : RouteAnyRing.Route R) (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (RouteAnyRing.Traces σ g) (RelationalPaths.routeGraph z) i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph
        (RouteAnyRing.rcomp σ g (bArg z))) (zSet ∪ RelationalPaths.oddPos)
        (extPos (extPos i)) (extPos (extPos j)) w := by
  rw [RelApp, ← RelationalPaths.relRuns_embed_iff extPos_injective
    (relPlug_graph_right_iff σ g z)]
  change RelationalPaths.RelRuns σ (RelationalPaths.RelRuns σ
    (RelationalPaths.routeGraph (RouteAnyRing.rcomp σ g (bArg z))) oddPos)
    (extPos '' oddPos) (extPos (extPos i)) (extPos (extPos j)) w ↔ _
  rw [extPos_image_oddPos, ← RelationalPaths.relRuns_vanishing zSet_not_oddPos]
  change RelationalPaths.RelRuns σ (RelationalPaths.routeGraph
    (RouteAnyRing.rcomp σ g (bArg z))) (zSet ∪ RelationalPaths.oddPos)
    (extPos (extPos i)) (extPos (extPos j)) w ↔ _
  rfl

/-- A relational path from the first stored route's track leaves the argument channel at once.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    bPlug [the stored machine with its outer argument connected]
    xPos [the first route's position in the machine]
    zSet [the argument channel positions]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    p [the source position in the first route]
    c [the final position]
    w [the accumulated path weight] -/
theorem relBPlug_collapse_x {R : Type u} [CommRing R] (σ : R)
    (x y z : RouteAnyRing.Route R) (p c : ℕ) (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (bPlug σ x y z)) zSet (xPos p) c w ↔
      ∃ q, c = xPos q ∧ x p = some (q, w) := by
  have hExit : ∀ j v, bPlug σ x y z (xPos p) = some (j, v) → j ∉ zSet := by
    intro j v hj
    rw [bPlug_x] at hj
    obtain ⟨⟨q, t⟩, _, heq⟩ := Option.map_eq_some_iff.mp hj
    simp only [Prod.mk.injEq] at heq
    exact heq.1 ▸ xPos_not_zSet q
  rw [relRuns_eq_self hExit, bPlug_x]
  constructor
  · rintro h
    obtain ⟨⟨q, t⟩, hx, hc⟩ := Option.map_eq_some_iff.mp h
    simp only [Prod.mk.injEq] at hc
    obtain ⟨hpos, hweight⟩ := hc
    exact ⟨q, hpos.symm, hweight ▸ hx⟩
  · rintro ⟨q, rfl, hx⟩
    rw [hx, Option.map_some]

/-- A relational path from the second stored route's track is a trace of its plugged argument.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    Traces [the union over finite paths through a route]
    bPlug [the stored machine with its outer argument connected]
    plug [the relation-first application wiring]
    yPos [the second route's position in the machine]
    innPos [the answering-track position]
    zSet [the argument channel positions]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    u [the source position in the second route]
    c [the final position]
    w [the accumulated path weight] -/
theorem relBPlug_collapse_y {R : Type u} [CommRing R] (σ : R)
    (x y z : RouteAnyRing.Route R) (u c : ℕ) (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (bPlug σ x y z)) zSet (innPos u) c w ↔
      ∃ v, c = innPos v ∧ RouteAnyRing.Traces σ (ApplicationWiring.plug σ y z) u v w := by
  have h := relRuns_route_embed (σ := σ) yPos_injective (bPlug_y σ x y z)
    (S := RelationalPaths.oddPos) (extPos u) c w
  rw [← yPos_image_oddPos, ← yPos_ext]
  constructor
  · intro hrun
    obtain ⟨b, hc, hpath⟩ := h.mp hrun
    obtain ⟨v, hb, htrace⟩ :=
      (relRuns_oddPos_iff_traces (σ := σ) (f := ApplicationWiring.plug σ y z) u b w).mp hpath
    subst b
    exact ⟨v, by simpa only [yPos_ext] using hc, htrace⟩
  · rintro ⟨v, hc, htrace⟩
    apply h.mpr
    refine ⟨extPos v, ?_, ?_⟩
    · simpa only [yPos_ext] using hc
    · exact (relRuns_oddPos_iff_traces (σ := σ) (f := ApplicationWiring.plug σ y z) u
        (extPos v) w).mpr ⟨v, rfl, htrace⟩

/-- The stored-composition application is the reading that collapses the wire before its argument.

    RelApp [the relation-first application relation]
    BCompose [the unrestricted stored-composition relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    bBig [the whole four-track composing machine]
    zSet [the argument channel positions]
    oddPos [the remaining application loop positions]
    extPos [the external wire position]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∪ [the union of loop layers]
    ↔ [exactly when, true in both directions]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relLeft_reading {R : Type u} [CommRing R] (σ : R)
    (x y z : RouteAnyRing.Route R) (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (BCompose σ x y) (RelationalPaths.routeGraph z) i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (bBig σ x y z))
        (zSet ∪ oddPos) (extPos (extPos i)) (extPos (extPos j)) w := by
  change RelApp σ (RouteAnyRing.Traces σ (bStore σ x y))
    (RelationalPaths.routeGraph z) i j w ↔ _
  rw [bBig]
  exact relOuter_reading σ (bStore σ x y) z i j w

/-- The nested relation-first application is the reading that collapses all three machine loops.

    RelApp [the relation-first application relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    bBig [the whole four-track composing machine]
    outerSet [the asking-track loop positions]
    innerSet [the answering-track loop positions]
    zSet [the argument channel positions]
    extPos [the external wire position]
    StepRel [a weighted constructive transition relation]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∀ [for every]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relRight_reading {R : Type u} [CommRing R] (σ : R)
    (x y z : RouteAnyRing.Route R) (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (RelationalPaths.routeGraph x)
      (RelApp σ (RelationalPaths.routeGraph y) (RelationalPaths.routeGraph z)) i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (bBig σ x y z))
        (outerSet ∪ (innerSet ∪ zSet)) (extPos (extPos i)) (extPos (extPos j)) w := by
  have hx : ∀ p c w, RelationalPaths.RelRuns σ
      (RelationalPaths.routeGraph (bPlug σ x y z)) zSet (xPos p) c w ↔
      ∃ q, c = xPos q ∧ RelationalPaths.routeGraph x p q w := by
    simpa only [RelationalPaths.routeGraph] using (relBPlug_collapse_x σ x y z)
  have hy : ∀ n c w, RelationalPaths.RelRuns σ
      (RelationalPaths.routeGraph (bPlug σ x y z)) zSet (innPos n) c w ↔
      ∃ t, c = innPos t ∧ RelApp σ (RelationalPaths.routeGraph y)
        (RelationalPaths.routeGraph z) n t w := by
    intro n c w
    rw [relBPlug_collapse_y]
    constructor
    · rintro ⟨t, hc, htrace⟩
      exact ⟨t, hc, (relApp_routeGraph σ y z n t w).mpr htrace⟩
    · rintro ⟨t, hc, happ⟩
      exact ⟨t, hc, (relApp_routeGraph σ y z n t w).mp happ⟩
  rw [bBig_eq]
  exact relApplied_reading σ (bPlug σ x y z) (RelationalPaths.routeGraph x)
    (RelApp σ (RelationalPaths.routeGraph y) (RelationalPaths.routeGraph z)) hx hy i j w

/-- The wire-first and nested relation-first readings of the composing machine agree.

    RelApp [the relation-first application relation]
    BCompose [the unrestricted stored-composition relation]
    routeGraph [the transition relation presented by a partial weighted route]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ↔ [exactly when, true in both directions]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relBracket {R : Type u} [CommRing R] (σ : R)
    (x y z : RouteAnyRing.Route R) (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (BCompose σ x y) (RelationalPaths.routeGraph z) i j w ↔
      RelApp σ (RelationalPaths.routeGraph x)
        (RelApp σ (RelationalPaths.routeGraph y) (RelationalPaths.routeGraph z)) i j w := by
  rw [relLeft_reading, relRight_reading, bracket_sets]

/-- The collapsed machine on the first route's tracks.

    {σ : ℝ} {M a : Route}                          [parameter, machine, route]
    (hx : ∀ p, runsRoute σ M zSet (xPos p)         [for every position of the first route]
      = (a p).map fun q => (xPos q.1, q.2))        [collapsing the channel gives that route]
    (p : ℕ)                                        [a position]
    fedRoute σ M (xPos p)                          [a step of the collapsed machine there]
    = (a p).map fun q => (xOut q.1, q.2)           [is that route, then the wire] -/
theorem fed_x {σ : ℝ} {M a : Route}
    (hx : ∀ p, runsRoute σ M zSet (xPos p) = (a p).map fun q => (xPos q.1, q.2)) (p : ℕ) :
    fedRoute σ M (xPos p) = (a p).map fun q => (xOut q.1, q.2) := by
  rw [fedRoute, runsRoute_rcomp_id swapRoute_id_zSet swapRoute_not_enter_zSet, rcomp, hx]
  cases ha : a p with
  | none => rfl
  | some q => simp only [Option.map_some, Option.bind_some, swapRoute, xOut, mul_one_right]

/-- The collapsed machine on the answering track hands its answer back.

    {σ : ℝ} {M b : Route}                          [parameter, machine, route]
    (hy : ∀ n, runsRoute σ M zSet (innPos n)       [for every position of the answering track]
      = (b n).map fun t => (innPos t.1, t.2))      [collapsing the channel gives that route]
    (n : ℕ)                                        [a position]
    fedRoute σ M (innPos n)                        [a step of the collapsed machine there]
    = (b n).map fun t => (outPos t.1, t.2)         [answers onto the asking track] -/
theorem fed_inn {σ : ℝ} {M b : Route}
    (hy : ∀ n, runsRoute σ M zSet (innPos n) = (b n).map fun t => (innPos t.1, t.2)) (n : ℕ) :
    fedRoute σ M (innPos n) = (b n).map fun t => (outPos t.1, t.2) := by
  rw [fedRoute, runsRoute_rcomp_id swapRoute_id_zSet swapRoute_not_enter_zSet, rcomp, hy]
  cases hb : b n with
  | none => rfl
  | some t =>
    simp only [Option.map_some, Option.bind_some, swapRoute, bSwapFun_inn, mul_one_right]

-- ═══════════════════════════════════════════════════════════════════
-- THE OUTER STEP
-- ═══════════════════════════════════════════════════════════════════

/-- **One collapsed step is one step of the first route applied to the second.**

@cite joyal-street-verity-1996

    {σ : ℝ} {M a b : Route}                        [the parameter, a machine, two routes]
    (hx : ∀ p, runsRoute σ M zSet (xPos p)         [for every position of the first route]
      = (a p).map fun q => (xPos q.1, q.2))        [collapsing the channel gives that route]
    (hy : ∀ n, runsRoute σ M zSet (innPos n)       [for every position of the answering track]
      = (b n).map fun t => (innPos t.1, t.2))      [collapsing the channel gives that route]
    (p c : ℕ) (w : Cell)                           [a start, an end and a weight]
    Runs σ (fedRoute σ M) innerSet (xPos p) c w    [a path over the answering track]
    ↔                                              [exactly when]
    ∃ q, plug σ a b p = some (q, w) ∧ c = xPos q   [one step of the application]

    So: from the first route's track a path either leaves at once, which is the case where the
    first route does not consult its argument, or it crosses to the answering track and comes
    straight back, which is the case where it does. There is no third case and no longer path,
    because the answering track hands a path back to the asking track, which is outside the
    loop being collapsed. That is the whole content of a combinator which composes: what the
    machine holds on the answering track, already applied, answers the first route's question in
    one step. -/
theorem applied_step {σ : ℝ} {M a b : Route}
    (hx : ∀ p, runsRoute σ M zSet (xPos p) = (a p).map fun q => (xPos q.1, q.2))
    (hy : ∀ n, runsRoute σ M zSet (innPos n) = (b n).map fun t => (innPos t.1, t.2))
    (p c : ℕ) (w : Cell) :
    Runs σ (fedRoute σ M) innerSet (xPos p) c w
      ↔ ∃ q, plug σ a b p = some (q, w) ∧ c = xPos q := by
  constructor
  · rintro ⟨n, h⟩
    cases h with
    | exit hf hns =>
      rw [fed_x hx] at hf
      obtain ⟨⟨q1, q2⟩, ha, hq⟩ := Option.map_eq_some_iff.mp hf
      simp only [Prod.mk.injEq] at hq
      obtain ⟨e1, e2⟩ := hq
      subst e1
      subst e2
      obtain ⟨k, rfl⟩ := xOut_not_inner hns
      exact ⟨extPos k, plug_ext_step σ a _ ha, by rw [xOut_ext, xPos_ext]⟩
    | step hf hsm hr =>
      rw [fed_x hx] at hf
      obtain ⟨⟨q1, q2⟩, ha, hq⟩ := Option.map_eq_some_iff.mp hf
      simp only [Prod.mk.injEq] at hq
      obtain ⟨e1, e2⟩ := hq
      subst e1
      subst e2
      obtain ⟨u, rfl⟩ := xOut_inner hsm
      rw [xOut_arg] at hr
      cases hr with
      | exit hf' hns' =>
        rw [fed_inn hy] at hf'
        obtain ⟨⟨t1, t2⟩, ht, hqq⟩ := Option.map_eq_some_iff.mp hf'
        simp only [Prod.mk.injEq] at hqq
        obtain ⟨f1, f2⟩ := hqq
        subst f1
        subst f2
        exact ⟨argPos t1, plug_arg_step σ a _ ha ht, by rw [xPos_arg]⟩
      | step hf' hsm' _ =>
        exfalso
        rw [fed_inn hy] at hf'
        obtain ⟨⟨t1, t2⟩, _, hqq⟩ := Option.map_eq_some_iff.mp hf'
        simp only [Prod.mk.injEq] at hqq
        obtain ⟨f1, _⟩ := hqq
        subst f1
        exact outer_disjoint_inner _ (mem_outerSet t1) hsm'
  · rintro ⟨q, hq, rfl⟩
    rw [plug, rcomp_eq_some] at hq
    obtain ⟨m, w₁, w₂, ha, hb, rfl⟩ := hq
    rcases pos_split2 m with ⟨k, rfl⟩ | ⟨u, rfl⟩
    · rw [rtensor_ext_id] at hb
      simp only [Option.some.injEq, Prod.mk.injEq] at hb
      obtain ⟨hb2, hb3⟩ := hb
      subst hb2
      subst hb3
      refine ⟨0, RunsN.exit ?_ ?_⟩
      · rw [fed_x hx, ha, Option.map_some, xOut_ext, xPos_ext,
          Trichotomy.AnyRing.ringMul_one]
      · rw [xPos_ext]
        exact (extPos_not_mem (extPos k)).2
    · rw [rtensor_arg] at hb
      obtain ⟨⟨t1, t2⟩, hr, ht⟩ := Option.map_eq_some_iff.mp hb
      simp only [Prod.mk.injEq] at ht
      obtain ⟨ht1, ht2⟩ := ht
      subst ht1
      subst ht2
      refine ⟨1, RunsN.step ?_ (mem_innerSet u) (RunsN.exit ?_ ?_)⟩
      · rw [fed_x hx, ha, Option.map_some, xOut_arg]
      · rw [fed_inn hy, hr, Option.map_some, xPos_arg]
      · rw [xPos_arg]
        exact outer_disjoint_inner _ (mem_outerSet t1)

/-- The twice collapsed machine is the first route with the second plugged in.

    {σ : ℝ} {M a b : Route}                        [the parameter, a machine, two routes]
    (hx : ∀ p, runsRoute σ M zSet (xPos p)         [for every position of the first route]
      = (a p).map fun q => (xPos q.1, q.2))        [collapsing the channel gives that route]
    (hy : ∀ n, runsRoute σ M zSet (innPos n)       [for every position of the answering track]
      = (b n).map fun t => (innPos t.1, t.2))      [collapsing the channel gives that route]
    (p : ℕ)                                        [a position]
    runsRoute σ (fedRoute σ M) innerSet (xPos p)   [two loops collapsed]
    = (plug σ a b p).map fun q => (xPos q.1, q.2)  [is the outer application] -/
theorem fed_collapse {σ : ℝ} {M a b : Route}
    (hx : ∀ p, runsRoute σ M zSet (xPos p) = (a p).map fun q => (xPos q.1, q.2))
    (hy : ∀ n, runsRoute σ M zSet (innPos n) = (b n).map fun t => (innPos t.1, t.2)) (p : ℕ) :
    runsRoute σ (fedRoute σ M) innerSet (xPos p)
      = (plug σ a b p).map fun q => (xPos q.1, q.2) := by
  cases hp : plug σ a b p with
  | none =>
    cases hq : runsRoute σ (fedRoute σ M) innerSet (xPos p) with
    | none => rfl
    | some t =>
      obtain ⟨q, hq1, _⟩ := (applied_step hx hy p t.1 t.2).mp
        ((runsRoute_eq σ (fedRoute σ M) innerSet (xPos p) t.1 t.2).mp (by rw [hq]))
      rw [hp] at hq1
      exact absurd hq1 (by simp)
  | some t =>
    rw [(runsRoute_eq σ (fedRoute σ M) innerSet (xPos p) (xPos t.1) t.2).mpr
      ((applied_step hx hy p (xPos t.1) t.2).mpr ⟨t.1, by rw [hp], rfl⟩)]
    rfl

/-- **The applied reading: collapse the argument's channel, the answering track, then the asking.**

@cite joyal-street-verity-1996

    {σ : ℝ} {M a b : Route}                        [the parameter, a machine, two routes]
    (hx : ∀ p, runsRoute σ M zSet (xPos p)         [for every position of the first route]
      = (a p).map fun q => (xPos q.1, q.2))        [collapsing the channel gives that route]
    (hy : ∀ n, runsRoute σ M zSet (innPos n)       [for every position of the answering track]
      = (b n).map fun t => (innPos t.1, t.2))      [collapsing the channel gives that route]
    (i j : ℕ) (w : Cell)                           [a start, an end and a weight]
    rapp σ a b i = some (j, w)                     [the first route applied to the second]
    ↔                                              [exactly when]
    Runs σ (rcomp σ M swapRoute) (outerSet ∪ (innerSet ∪ zSet))   [a path through all three loops]
      (extPos (extPos i)) (extPos (extPos j)) w                   [outside to outside]

    So: whatever machine holds the first route on its own tracks and the second, already applied,
    on the answering track, applying one to the other is a path of that machine. What each of the
    combinators below supplies is the pair of collapses, and this reading is then the same. -/
theorem applied_reading {σ : ℝ} {M a b : Route}
    (hx : ∀ p, runsRoute σ M zSet (xPos p) = (a p).map fun q => (xPos q.1, q.2))
    (hy : ∀ n, runsRoute σ M zSet (innPos n) = (b n).map fun t => (innPos t.1, t.2))
    (i j : ℕ) (w : Cell) :
    rapp σ a b i = some (j, w)
      ↔ Runs σ (rcomp σ M swapRoute) (outerSet ∪ (innerSet ∪ zSet))
          (extPos (extPos i)) (extPos (extPos j)) w := by
  rw [rapp, ← tracesTo_traceRoute σ (plug σ a b) i j w, traces_iff_runs_ext,
    ← runs_embed xPos_injective (fed_collapse hx hy) (extPos i) (extPos j) w,
    xPos_image_oddPos, xPos_ext, xPos_ext, fedRoute,
    ← runsRoute_union inner_disjoint_zSet, ← runs_vanishing outer_disjoint_rest]

/-- Collapsing the loops of an application, one machine at a time.

    (σ : ℝ) (g z : Route) (a : ℕ)                  [the parameter, a machine, an argument]
    runsRoute σ (rcomp σ g (bArg z)) oddPos (extPos a)         [collapse the machine's own loop]
    = (plug σ (traceRoute σ g) z a).map fun q => (extPos q.1, q.2)   [is the stored route, plugged]

    So: a machine which stores something and then offers a channel to an argument collapses to
    that stored route with the argument plugged in, because the argument's layer does nothing on
    the loop being collapsed. Every combinator here is of that shape, which is why the reading
    below serves all of them. -/
theorem plug_collapse (σ : ℝ) (g z : Route) (a : ℕ) :
    runsRoute σ (rcomp σ g (bArg z)) oddPos (extPos a)
      = (plug σ (traceRoute σ g) z a).map fun q => (extPos q.1, q.2) := by
  rw [runsRoute_rcomp_id (bArg_id_oddPos z) (bArg_not_enter_oddPos z),
    rcomp_embed (fun b => runsRoute_oddPos_traceRoute σ g b) (bArg_ext z) a, plug]

/-- **Applying a stored route is a path of its machine through both loops.**

@cite joyal-street-verity-1996
@cite girard-1989

    (σ : ℝ) (g z : Route) (i j : ℕ) (w : Cell)     [the parameter, a machine, an argument, a step]
    rapp σ (traceRoute σ g) z i = some (j, w)      [apply what the machine stores to the argument]
    ↔                                              [exactly when]
    Runs σ (rcomp σ g (bArg z)) (zSet ∪ oddPos)              [a path of the machine]
      (extPos (extPos i)) (extPos (extPos j)) w              [from the outside to the outside]

    So: collapse the machine's own loop first and the argument's channel second, or collapse
    both at once. This is the reading every stored side of every equation below takes, with a
    different machine each time. -/
theorem outer_reading (σ : ℝ) (g z : Route) (i j : ℕ) (w : Cell) :
    rapp σ (traceRoute σ g) z i = some (j, w)
      ↔ Runs σ (rcomp σ g (bArg z)) (zSet ∪ oddPos)
          (extPos (extPos i)) (extPos (extPos j)) w := by
  rw [rapp, ← tracesTo_traceRoute σ (plug σ (traceRoute σ g) z) i j w, traces_iff_runs_ext,
    ← runs_embed extPos_injective (plug_collapse σ g z) (extPos i) (extPos j) w,
    extPos_image_oddPos, ← runs_vanishing zSet_not_oddPos]

-- ═══════════════════════════════════════════════════════════════════
-- THE COMPOSING COMBINATOR
-- ═══════════════════════════════════════════════════════════════════

/-- Collapsing the argument's channel leaves the first route alone.

    (σ : ℝ) (x y z : Route) (p : ℕ)                [parameter, three routes, a position]
    runsRoute σ (bPlug σ x y z) zSet (xPos p)      [a collapsed step on the first route]
    = (x p).map fun q => (xPos q.1, q.2)           [is the first route's own step] -/
theorem bPlug_collapse_x (σ : ℝ) (x y z : Route) (p : ℕ) :
    runsRoute σ (bPlug σ x y z) zSet (xPos p) = (x p).map fun q => (xPos q.1, q.2) := by
  have h : ∀ j w, bPlug σ x y z (xPos p) = some (j, w) → j ∉ zSet := by
    intro j w hj
    rw [bPlug_x] at hj
    obtain ⟨⟨c1, c2⟩, _, hq⟩ := Option.map_eq_some_iff.mp hj
    simp only [Prod.mk.injEq] at hq
    obtain ⟨h1, _⟩ := hq
    subst h1
    exact xPos_not_zSet c1
  rw [runsRoute_eq_self h, bPlug_x]

/-- **Collapsing the argument's channel is applying the second route to the argument.**

@cite girard-1989

    (σ : ℝ) (x y z : Route) (u : ℕ)                [parameter, three routes, a position]
    runsRoute σ (bPlug σ x y z) zSet (innPos u)    [a collapsed step on the answering track]
    = (rapp σ y z u).map fun t => (innPos t.1, t.2)   [is a step of the applied second route]

    So: the second route and the outer argument sit inside the machine exactly as they sit in
    their own application, and collapsing that one channel is taking their trace. This is where
    `rapp σ y z` enters, whole, without being taken apart. -/
theorem bPlug_collapse_y (σ : ℝ) (x y z : Route) (u : ℕ) :
    runsRoute σ (bPlug σ x y z) zSet (innPos u)
      = (rapp σ y z u).map fun t => (innPos t.1, t.2) := by
  have h := runsRoute_embed (σ := σ) yPos_injective (bPlug_y σ x y z) (S := oddPos) (extPos u)
  rw [yPos_image_oddPos, yPos_ext] at h
  rw [h, runsRoute_oddPos_traceRoute, rapp]
  rw [plug]
  cases ht : traceRoute σ (ApplicationWiring.plug σ y z) u with
  | none => rfl
  | some t => rw [Option.map_some, Option.map_some, Option.map_some, yPos_ext]

/-- **The left reading: collapse the wire, then the argument's channel.**

    (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell)   [parameter, three routes, a step]
    rapp σ (bComb σ x y) z i = some (j, w)         [the stored composite, applied]
    ↔                                              [exactly when]
    Runs σ (bBig σ x y z) (zSet ∪ oddPos)          [a path of the machine through all three loops]
      (extPos (extPos i)) (extPos (extPos j)) w    [from the outside to the outside] -/
theorem left_reading (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell) :
    rapp σ (bComb σ x y) z i = some (j, w)
      ↔ Runs σ (bBig σ x y z) (zSet ∪ oddPos) (extPos (extPos i)) (extPos (extPos j)) w :=
  outer_reading σ (bStore σ x y) z i j w

/-- **The right reading: collapse the argument's channel, the answering track, then the asking.**

    (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell)   [parameter, three routes, a step]
    rapp σ x (rapp σ y z) i = some (j, w)          [the first route applied to the second's value]
    ↔                                              [exactly when]
    Runs σ (bBig σ x y z) (outerSet ∪ (innerSet ∪ zSet))   [a path through all three loops]
      (extPos (extPos i)) (extPos (extPos j)) w            [from the outside to the outside] -/
theorem right_reading (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell) :
    rapp σ x (rapp σ y z) i = some (j, w)
      ↔ Runs σ (bBig σ x y z) (outerSet ∪ (innerSet ∪ zSet))
          (extPos (extPos i)) (extPos (extPos j)) w := by
  rw [bBig_eq]
  exact applied_reading (bPlug_collapse_x σ x y z) (bPlug_collapse_y σ x y z) i j w

/-- **B x y z = x (y z)**, as an equation between routes.

@cite schonfinkel-1924
@cite curry-1930
@cite girard-1989

    (σ : ℝ) (x y z : Route)                        [the parameter and three routes]
    rapp σ (bComb σ x y) z = rapp σ x (rapp σ y z) [composing then applying is applying twice]

    So: the composing combinator holds at this carrier, and it is the first one whose stored
    route is itself a trace. The two sides are one machine read in two orders, and the equality
    is the vanishing axiom of a traced monoidal category, which `Combinator/Vanishing.lean`
    proves for these routes rather than assuming. -/
theorem bracket (σ : ℝ) (x y z : Route) :
    rapp σ (bComb σ x y) z = rapp σ x (rapp σ y z) := by
  refine route_ext fun i j w => ?_
  rw [left_reading, right_reading, bracket_sets]

end Combinator
end Apeiron

end
