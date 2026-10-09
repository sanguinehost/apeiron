import Apeiron.Combinator.Linear.Compositor

/-!
# The exchanging combinator

@name Exchange
@cite girard-1989
@cite schonfinkel-1924
@cite curry-1930
@grade proved

`C x y z = x z y` at the carrier of `Combinator/Interaction.lean`. The stored route is `cComb`
and the equation is `commutator` below.

## Root

| from | here |
|---|---|
| `AnyRing.ringOne_mul`, `AnyRing.ringMul_one`, `AnyRing.ringMul_comm` | `bArg_rtensor_comm` |
| `AnyRing.ringOne_mul`, `AnyRing.ringMul_one`, `AnyRing.ringMul_comm` | `cMach_eq` |

The layout and the relabelling are arithmetic on the counting numbers, so
`theory/NOVELTY.md`'s A1 spectator test applies as it did to `Combinator/Compositor.lean`. The
parameter weights a path and enters through `Combinator/Vanishing.lean`.

## The argument

`C` needs two routes stored, as `B` does, but the wire between them is different. Where `B` puts
the second route between the first and its argument, `C` puts the second route at the first
route's **second** argument and leaves the first argument facing outwards. Written on the first
route's own wire that reads:

```
   extPos (extPos k)     the outside, after both arguments are taken
   extPos (argPos v)     the second argument, where the stored route sits
   argPos u              the first argument, which is the one the composite offers
```

So the stored route is `cStore`, the first route with the second plugged on the middle track,
and the loop to collapse is the middle track rather than the odd half. `cSwap` exchanges those
two, which puts the loop where a trace collapses it, so `cComb` is a `traceRoute` again.

The two sides are then the same machine `cMach` read twice, exactly as in `Compositor.lean`, and
`outer_reading` serves both. On the left the relabelling stands in the way, and it is removed by
`runs_conj`, since `cSwap` carries the pair of loops onto itself. On the right the machine is the
first route applied to the outer argument, then the stored route, which is `cMach_eq`.
-/

noncomputable section

universe u

namespace Apeiron
namespace Combinator

open Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- THE RELABELLING
-- ═══════════════════════════════════════════════════════════════════

/-- Exchange the second argument's track with the first argument's, fixing the outside.

    (i : ℕ)                                        [a position of the wire]
    ℕ                                              [where the relabelling sends it] -/
def cSwapFun (i : ℕ) : ℕ :=
  Sum.elim
    (fun m => Sum.elim (fun k => extPos (extPos k)) (fun v => argPos v)
      (Equiv.natSumNatEquivNat.symm m))
    (fun u => extPos (argPos u))
    (Equiv.natSumNatEquivNat.symm i)

/-- The relabelling fixes the outside.

    (k : ℕ)                                                        [a position]
    cSwapFun (extPos (extPos k)) = extPos (extPos k)               [the outside stays] -/
theorem cSwapFun_ee (k : ℕ) : cSwapFun (extPos (extPos k)) = extPos (extPos k) := by
  simp only [cSwapFun, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The relabelling sends the second argument's track to the odd half.

    (v : ℕ)                                                        [a position]
    cSwapFun (extPos (argPos v)) = argPos v                        [the stored route's track] -/
theorem cSwapFun_ea (v : ℕ) : cSwapFun (extPos (argPos v)) = argPos v := by
  simp only [cSwapFun, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- The relabelling sends the odd half to the second argument's track.

    (u : ℕ)                                                        [a position]
    cSwapFun (argPos u) = extPos (argPos u)                        [the outer argument's track] -/
theorem cSwapFun_arg (u : ℕ) : cSwapFun (argPos u) = extPos (argPos u) := by
  simp only [cSwapFun, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- The relabelling is its own inverse.

    Function.Involutive cSwapFun                   [running it twice changes nothing] -/
theorem cSwapFun_involutive : Function.Involutive cSwapFun := by
  intro p
  rcases pos_split p with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨u, rfl⟩
  · rw [cSwapFun_ee, cSwapFun_ee]
  · rw [cSwapFun_ea, cSwapFun_arg]
  · rw [cSwapFun_arg, cSwapFun_ea]

/-- The relabelling as a permutation.

    Equiv.Perm ℕ                                   [the positions, rearranged] -/
def cSwap : Equiv.Perm ℕ := Function.Involutive.toPerm cSwapFun cSwapFun_involutive

/-- The permutation is the map.

    (i : ℕ)                                        [a position]
    cSwap i = cSwapFun i                           [the same relabelling] -/
theorem cSwap_apply (i : ℕ) : cSwap i = cSwapFun i := rfl

/-- The permutation's inverse is the map.

    (i : ℕ)                                        [a position]
    cSwap.symm i = cSwapFun i                      [an involution is its own inverse] -/
theorem cSwap_symm_apply (i : ℕ) : cSwap.symm i = cSwapFun i := rfl

/-- The outside is on neither loop.

    (k : ℕ)                                        [a position]
    extPos (extPos k) ∉ zSet ∪ oddPos              [the outside is not collapsed] -/
theorem aPos_not_mem (k : ℕ) : extPos (extPos k) ∉ zSet ∪ oddPos := by
  rintro (⟨v, hv⟩ | hv)
  · rw [zPos_eq] at hv
    exact extPos_ne_argPos k v (extPos_injective hv)
  · exact not_mem_oddPos (extPos k) hv

/-- **The relabelling carries the pair of loops onto itself.**

    cSwap ⁻¹' (zSet ∪ oddPos) = zSet ∪ oddPos      [it exchanges the two and keeps the pair]

    So: the two loops the exchanging combinator collapses are the ones the relabelling swaps, so
    a path through both is a path through both either way. That is what lets `runs_conj` take
    the relabelling off the left side without anything else moving. -/
theorem cSwap_preimage : cSwap ⁻¹' (zSet ∪ oddPos) = zSet ∪ oddPos := by
  ext p
  simp only [Set.mem_preimage]
  rcases pos_split p with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨u, rfl⟩
  · rw [cSwap_apply, cSwapFun_ee]
  · rw [cSwap_apply, cSwapFun_ea]
    exact iff_of_true (Or.inr ⟨v, rfl⟩) (Or.inl (mem_zSet v))
  · rw [cSwap_apply, cSwapFun_arg]
    exact iff_of_true (Or.inl (mem_zSet u)) (Or.inr ⟨u, rfl⟩)

/-- The relabelling carries the pair of loops onto itself.

    '' [the image of a set under a map]
    ∪ [one of two position sets]
    = [is equal to]
    cSwap [the exchange relabelling of wire positions]
    zSet [the outer argument channel]
    oddPos [the stored route's loop] -/
theorem cSwap_image : cSwap '' (zSet ∪ RelationalPaths.oddPos) =
    zSet ∪ RelationalPaths.oddPos := by
  change cSwap '' (zSet ∪ oddPos) = zSet ∪ oddPos
  apply Set.Subset.antisymm
  · rintro p ⟨q, hq, rfl⟩
    have hpre : q ∈ cSwap ⁻¹' (zSet ∪ oddPos) := by
      rwa [cSwap_preimage]
    exact hpre
  · intro p hp
    refine ⟨cSwap.symm p, ?_, cSwap.apply_symm_apply p⟩
    have hpre : cSwap.symm p ∈ cSwap ⁻¹' (zSet ∪ oddPos) := by
      change cSwap (cSwap.symm p) ∈ zSet ∪ oddPos
      simpa using hp
    rwa [cSwap_preimage] at hpre

/-- **The argument layer is the plug's own layer, relabelled.**

    (z : Route R)                                  [a route]
    bArg z = conjRoute cSwap (rtensor rid z)       [feeding on the middle track is feeding, moved]

    So: the two ways of offering a channel to an argument, the one an application uses and the
    one this layout uses, differ by the relabelling alone. -/
theorem bArg_conj {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) :
    bArg z = conjRoute cSwap (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z) := by
  funext i
  rcases pos_split i with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨u, rfl⟩
  · rw [conjRoute, cSwap_apply, cSwapFun_ee, rtensor_ext_id, Option.map_some, cSwap_symm_apply,
      cSwapFun_ee, bArg_ext, rtensor_ext_id, Option.map_some]
  · rw [conjRoute, cSwap_apply, cSwapFun_ea, rtensor_arg, bArg_ext, rtensor_arg]
    cases hz : z v with
    | none => rfl
    | some q =>
      rw [Option.map_some, Option.map_some, Option.map_some, cSwap_symm_apply, cSwapFun_arg]
  · rw [conjRoute, cSwap_apply, cSwapFun_arg, rtensor_ext_id, Option.map_some, cSwap_symm_apply,
      cSwapFun_ea, bArg_arg]

/-- The two argument layers act on different tracks, so they commute.

    (σ : R) (y z : Route R)                        [the parameter and two routes]
    rcomp σ (bArg y) (rtensor rid z)               [the stored route, then the outer argument]
    = rcomp σ (rtensor rid z) (bArg y)             [is the outer argument, then the stored route] -/
theorem bArg_rtensor_comm {R : Type u} [CommRing R] (σ : R) (y z : RouteAnyRing.Route R) :
    RouteAnyRing.rcomp σ (bArg y) (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z) =
      RouteAnyRing.rcomp σ (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z) (bArg y) := by
  funext i
  rcases pos_split i with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨u, rfl⟩
  · simp only [RouteAnyRing.rcomp, bArg_ext, rtensor_ext_id, Option.map_some, Option.bind_some]
  · rw [RouteAnyRing.rcomp, RouteAnyRing.rcomp, bArg_ext, rtensor_arg, rtensor_ext_id,
      Option.bind_some, bArg_ext, rtensor_arg]
    cases hy : y v with
    | none => rfl
    | some q =>
      rw [Option.map_some, Option.map_some, Option.bind_some, Option.map_some, rtensor_ext_id,
        Option.map_some, AnyRing.ringOne_mul, AnyRing.ringMul_one]
  · rw [RouteAnyRing.rcomp, RouteAnyRing.rcomp, bArg_arg, Option.bind_some, rtensor_arg]
    cases hz : z u with
    | none => rfl
    | some q =>
      simp only [Option.map_some, Option.bind_some, bArg_arg, AnyRing.ringOne_mul,
        AnyRing.ringMul_one]

-- ═══════════════════════════════════════════════════════════════════
-- THE STORED ROUTE AND ITS MACHINE
-- ═══════════════════════════════════════════════════════════════════

/-- The first route with the second plugged at its second argument.

    (σ : R) (x y : Route R)                        [the parameter and two routes]
    rcomp σ x (bArg y)                             [run the first, then the stored second]
    : Route R                                      [one route with one loop left open] -/
def cStore {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R) : RouteAnyRing.Route R :=
  RouteAnyRing.rcomp σ x (bArg y)

/-- The exchange relation: trace the relabelled stored route without choosing a route value.

    Traces [the union over finite paths through a route]
    conjRoute [a route read through a relabelling of the wire]
    cStore [the first route with the second on its middle argument track]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    CExchange [the unbounded exchange relation]
    cSwap [the exchange relabelling of wire positions]
    x [the first route]
    y [the stored second route] -/
def CExchange {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R) :=
  RouteAnyRing.Traces σ (conjRoute cSwap (cStore σ x y))

/-- **The exchanging combinator's stored route.**

@cite curry-1930

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    traceRoute σ (conjRoute cSwap (cStore σ x y))  [collapse the second argument's track]
    : Route                                        [what is left offers the first argument]

    So: the relabelling is there to put the loop where the trace collapses it. What the trace
    leaves is the outside and the first route's first argument, which is the channel the
    composite offers, and that is the whole of the exchanging combinator. -/
def cComb (σ : ℝ) (x y : Route) : Route := traceRoute σ (conjRoute cSwap (cStore σ x y))

/-- The machine carrying both loops at once.

    (σ : R) (x y z : Route R)                      [the parameter and three routes]
    rcomp σ (cStore σ x y) (rtensor rid z)         [the stored route with the argument plugged]
    : Route R                                      [one route carrying both loops] -/
def cMach {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) :
    RouteAnyRing.Route R :=
  RouteAnyRing.rcomp σ (cStore σ x y) (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z)

/-- The machine read the other way: apply the outer argument first.

    (σ : R) (x y z : Route R)                      [the parameter and three routes]
    cMach σ x y z = rcomp σ (plug σ x z) (bArg y)  [the first route applied, then the stored one]

    So: the two argument layers act on different tracks, so which is written first is a choice of
    reading and not a choice of machine. -/
theorem cMach_eq {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) :
    cMach σ x y z = RouteAnyRing.rcomp σ (ApplicationWiring.plug σ x z) (bArg y) := by
  rw [cMach, cStore, ApplicationWiring.plug, RouteAnyRing.rcomp_assoc,
    RouteAnyRing.rcomp_assoc, bArg_rtensor_comm]

/-- The relational reading that applies the first route to the outer argument before the stored
second route.

    RelApp [the relation-first application relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    Traces [the union over finite paths through a route]
    cMach [the exchange machine carrying the two loops]
    zSet [the outer argument channel]
    oddPos [the stored route's loop]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ↔ [exactly when, true in both directions]
    extPos [a position of the wire outside both loops]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relCRight_reading {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R)
    (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (RouteAnyRing.Traces σ (ApplicationWiring.plug σ x z))
      (RelationalPaths.routeGraph y) i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (cMach σ x y z))
        (zSet ∪ RelationalPaths.oddPos) (extPos (extPos i)) (extPos (extPos j)) w := by
  have h := relOuter_reading σ (ApplicationWiring.plug σ x z) y i j w
  rw [← cMach_eq] at h
  exact h

/-- The relation-first left exchange reading reaches the same two-loop machine as the right one.

    RelApp [the relation-first application relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    CExchange [the unbounded exchange relation]
    cMach [the exchange machine carrying the two loops]
    zSet [the outer argument channel]
    oddPos [the stored route's loop]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∪ [one of two loop layers]
    ↔ [exactly when, true in both directions]
    extPos [a position of the wire outside both loops]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relCLeft_reading {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R)
    (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (CExchange σ x y) (RelationalPaths.routeGraph z) i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (cMach σ x y z))
        (zSet ∪ RelationalPaths.oddPos) (extPos (extPos i)) (extPos (extPos j)) w := by
  have h := relOuter_reading σ (conjRoute cSwap (cStore σ x y)) z i j w
  change RelApp σ (RouteAnyRing.Traces σ (conjRoute cSwap (cStore σ x y)))
    (RelationalPaths.routeGraph z) i j w ↔ _
  rw [bArg_conj, ← conjRoute_rcomp] at h
  have hconj := relRuns_conj σ cSwap (cMach σ x y z) (zSet ∪ RelationalPaths.oddPos)
    (extPos (extPos i)) (extPos (extPos j)) w
  rw [cSwap_image, cSwap_apply, cSwapFun_ee, cSwap_apply, cSwapFun_ee] at hconj
  exact h.trans hconj

/-- The two relation-first exchange readings agree without selecting a traced route.

    RelApp [the relation-first application relation]
    routeGraph [the transition relation presented by a partial weighted route]
    CExchange [the unbounded exchange relation]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ↔ [exactly when, true in both directions]
    x [the first route]
    y [the stored second route]
    z [the outer argument route]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relCommutator {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R)
    (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (CExchange σ x y) (RelationalPaths.routeGraph z) i j w ↔
      RelApp σ (RelApp σ (RelationalPaths.routeGraph x) (RelationalPaths.routeGraph z))
        (RelationalPaths.routeGraph y) i j w := by
  have hinner : RelApp σ (RelationalPaths.routeGraph x) (RelationalPaths.routeGraph z) =
      RouteAnyRing.Traces σ (ApplicationWiring.plug σ x z) := by
    funext p q u
    exact propext (relApp_routeGraph σ x z p q u)
  rw [hinner, relCLeft_reading, relCRight_reading]

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO READINGS
-- ═══════════════════════════════════════════════════════════════════

/-- **The left reading: collapse the second argument's track, then the first's.**

    (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell)   [parameter, three routes, a step]
    rapp σ (cComb σ x y) z i = some (j, w)         [the stored route, applied]
    ↔                                              [exactly when]
    Runs σ (cMach σ x y z) (zSet ∪ oddPos)                   [a path through both loops]
      (extPos (extPos i)) (extPos (extPos j)) w              [from the outside to the outside] -/
theorem cLeft_reading (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell) :
    rapp σ (cComb σ x y) z i = some (j, w)
      ↔ Runs σ (cMach σ x y z) (zSet ∪ oddPos)
          (extPos (extPos i)) (extPos (extPos j)) w := by
  have h := outer_reading σ (conjRoute cSwap (cStore σ x y)) z i j w
  rw [bArg_conj, ← conjRoute_rcomp] at h
  have h2 := runs_conj (σ := σ) cSwap (cMach σ x y z) (zSet ∪ oddPos)
    (extPos (extPos i)) (extPos (extPos j)) w
  rw [cSwap_preimage, cSwap_apply, cSwapFun_ee, cSwap_apply, cSwapFun_ee] at h2
  exact h.trans h2

/-- **The right reading: apply the outer argument first, then the stored route.**

    (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell)   [parameter, three routes, a step]
    rapp σ (rapp σ x z) y i = some (j, w)          [the first route applied twice]
    ↔                                              [exactly when]
    Runs σ (cMach σ x y z) (zSet ∪ oddPos)                   [a path through both loops]
      (extPos (extPos i)) (extPos (extPos j)) w              [from the outside to the outside] -/
theorem cRight_reading (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell) :
    rapp σ (rapp σ x z) y i = some (j, w)
      ↔ Runs σ (cMach σ x y z) (zSet ∪ oddPos)
          (extPos (extPos i)) (extPos (extPos j)) w := by
  have h := outer_reading σ (plug σ x z) y i j w
  rw [cMach_eq]
  exact h

/-- **C x y z = x z y**, as an equation between routes.

@cite schonfinkel-1924
@cite curry-1930

    (σ : ℝ) (x y z : Route)                        [the parameter and three routes]
    rapp σ (cComb σ x y) z = rapp σ (rapp σ x z) y [exchanging the two arguments]

    So: the exchanging combinator holds at this carrier. Its stored route is a trace, as the
    composing combinator's is, and the two loops it carries are the same two either way, so the
    equation is the vanishing axiom again with a different pair of loops. -/
theorem commutator (σ : ℝ) (x y z : Route) :
    rapp σ (cComb σ x y) z = rapp σ (rapp σ x z) y := by
  refine route_ext fun i j w => ?_
  rw [cLeft_reading, cRight_reading]

end Combinator
end Apeiron

end
