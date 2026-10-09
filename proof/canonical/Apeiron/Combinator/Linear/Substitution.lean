import Apeiron.Combinator.Linear.Compositor

/-!
# The substituting combinator

@name Substitution
@cite girard-1989
@cite schonfinkel-1924
@cite curry-1930
@cite abramsky-haghverdi-scott-2002
@grade proved

`S x y z = x z (y z)` at the carrier of `Combinator/Interaction.lean`, with the argument arriving
as the two copies its two uses consume, which is the form `Combinator/Contraction.lean` fixed for
`W x y = x y y`. The stored route is `sComb` and the equation is `substitution` below.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | `sPlug_x`, `sPlug_y`, `sBig_eq` |

The layout is arithmetic on the counting numbers, so `theory/NOVELTY.md`'s A1 spectator test
applies as it did in `Combinator/Compositor.lean`. The parameter weights a path and enters
through `Combinator/Vanishing.lean`.

## The five tracks

`S` is the composing combinator with both routes given the argument. Written out, a position of
the wire falls into one of five tracks:

```
   extPos (extPos k)                the outside, after both arguments are taken
   argPos (extPos m)                the first route asking for the second route's value
   argPos (argPos n)                the second route answering
   extPos (argPos (extPos a))       the first route asking the outer argument
   extPos (argPos (argPos b))       the second route asking the outer argument
```

The middle two are the odd positions, so a trace collapses them and `sComb` is a bare
`traceRoute` again. The outer three are even, so they survive as the composite's outside and its
argument channel, and that channel is split in two: the copy the first route consumes and the
copy the second consumes. That is why the argument arrives as `rtensor z z`, and it is the same
splitting `Combinator/Contraction.lean` uses.

## The argument

The wire is the same wire as in `Combinator/Compositor.lean`, `swapRoute`, and the argument layer
is the same layer, `bArg`, applied to the doubled argument. So the left side is
`outer_reading` at this machine, and the right side is `applied_reading` at this machine, once
the two collapses it asks for are supplied. Those are `sPlug_collapse_x` and `sPlug_collapse_y`:
collapsing the argument channel gives the first route applied to the argument on its own tracks
and the second route applied to the argument on the answering track. Each is a single embedding,
because each route reaches only its own copy of the argument, and the other copy is collapsed
first by `runsRoute_union` without either route noticing.
-/

noncomputable section

universe u

namespace Apeiron
namespace Combinator

open Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO COPIES OF THE ARGUMENT CHANNEL
-- ═══════════════════════════════════════════════════════════════════

/-- Distinct argument positions stay distinct.

    Function.Injective zPos                        [the argument track is a copy of the wire] -/
theorem zPos_injective : Function.Injective zPos := by
  intro a b h
  rw [zPos_eq, zPos_eq] at h
  exact argPos_injective (extPos_injective h)

/-- The copy of the argument the first route consumes.

    Set ℕ                                          [the even half of the argument channel] -/
def z1Set : Set ℕ := {p | ∃ a, p = zPos (extPos a)}

/-- The copy of the argument the second route consumes.

    Set ℕ                                          [the odd half of the argument channel] -/
def z2Set : Set ℕ := {p | ∃ b, p = zPos (argPos b)}

/-- The argument channel is its two copies.

    zSet = z1Set ∪ z2Set                           [one channel, split in two] -/
theorem zSet_split : zSet = z1Set ∪ z2Set := by
  ext p
  constructor
  · rintro ⟨d, rfl⟩
    rcases pos_split2 d with ⟨a, rfl⟩ | ⟨b, rfl⟩
    · exact Or.inl ⟨a, rfl⟩
    · exact Or.inr ⟨b, rfl⟩
  · rintro (⟨a, rfl⟩ | ⟨b, rfl⟩)
    · exact ⟨extPos a, rfl⟩
    · exact ⟨argPos b, rfl⟩

/-- The argument channel is its two copies, the other way round.

    zSet = z2Set ∪ z1Set                           [the same channel] -/
theorem zSet_split' : zSet = z2Set ∪ z1Set := by
  rw [zSet_split, Set.union_comm]

/-- The two copies do not meet.

    ∀ p, p ∈ z1Set → p ∉ z2Set                     [each route reaches one copy] -/
theorem z1_disjoint_z2 : ∀ p, p ∈ z1Set → p ∉ z2Set := by
  rintro p ⟨a, rfl⟩ ⟨b, hb⟩
  exact extPos_ne_argPos a b (zPos_injective hb)

/-- The two copies do not meet, the other way round.

    ∀ p, p ∈ z2Set → p ∉ z1Set                     [each route reaches one copy] -/
theorem z2_disjoint_z1 : ∀ p, p ∈ z2Set → p ∉ z1Set := by
  rintro p ⟨b, rfl⟩ ⟨a, ha⟩
  exact extPos_ne_argPos a b (zPos_injective ha).symm

-- ═══════════════════════════════════════════════════════════════════
-- WHERE THE TWO ROUTES SIT
-- ═══════════════════════════════════════════════════════════════════

/-- The first route's place on the wire.

    (p : ℕ)                                        [one of the first route's positions]
    its place after one application if p is even, its own copy of the argument if p is odd
    : ℕ                                            [a position of the wire] -/
def sxPos (p : ℕ) : ℕ :=
  Sum.elim (fun e => xPos e) (fun a => zPos (extPos a)) (Equiv.natSumNatEquivNat.symm p)

/-- The second route's place on the wire.

    (p : ℕ)                                        [one of the second route's positions]
    the answering track if p is even, its own copy of the argument if p is odd
    : ℕ                                            [a position of the wire] -/
def syPos (p : ℕ) : ℕ :=
  Sum.elim (fun n => innPos n) (fun b => zPos (argPos b)) (Equiv.natSumNatEquivNat.symm p)

/-- The first route, once applied, sits where the composing combinator puts it.

    (e : ℕ)                                        [a position]
    sxPos (extPos e) = xPos e                      [the outside and the asking track] -/
theorem sxPos_ext (e : ℕ) : sxPos (extPos e) = xPos e := by
  simp only [sxPos, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The first route's argument is its own copy.

    (a : ℕ)                                        [a position]
    sxPos (argPos a) = zPos (extPos a)             [the first copy of the argument channel] -/
theorem sxPos_arg (a : ℕ) : sxPos (argPos a) = zPos (extPos a) := by
  simp only [sxPos, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- The second route answers on the answering track.

    (n : ℕ)                                        [a position]
    syPos (extPos n) = innPos n                    [where the first route's question arrives] -/
theorem syPos_ext (n : ℕ) : syPos (extPos n) = innPos n := by
  simp only [syPos, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The second route's argument is its own copy.

    (b : ℕ)                                        [a position]
    syPos (argPos b) = zPos (argPos b)             [the second copy of the argument channel] -/
theorem syPos_arg (b : ℕ) : syPos (argPos b) = zPos (argPos b) := by
  simp only [syPos, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- The first route's places are distinct.

    Function.Injective sxPos                       [distinct positions stay distinct] -/
theorem sxPos_injective : Function.Injective sxPos := by
  intro p q h
  rcases pos_split2 p with ⟨e, rfl⟩ | ⟨a, rfl⟩ <;> rcases pos_split2 q with ⟨e', rfl⟩ | ⟨a', rfl⟩
  · rw [sxPos_ext, sxPos_ext] at h
    rw [xPos_injective h]
  · rw [sxPos_ext, sxPos_arg] at h
    exact absurd ⟨extPos a', h⟩ (xPos_not_zSet e)
  · rw [sxPos_arg, sxPos_ext] at h
    exact absurd ⟨extPos a, h.symm⟩ (xPos_not_zSet e')
  · rw [sxPos_arg, sxPos_arg] at h
    rw [extPos_injective (zPos_injective h)]

/-- The second route's places are distinct.

    Function.Injective syPos                       [distinct positions stay distinct] -/
theorem syPos_injective : Function.Injective syPos := by
  intro p q h
  rcases pos_split2 p with ⟨n, rfl⟩ | ⟨b, rfl⟩ <;> rcases pos_split2 q with ⟨n', rfl⟩ | ⟨b', rfl⟩
  · rw [syPos_ext, syPos_ext, innPos_eq, innPos_eq] at h
    rw [argPos_injective (argPos_injective h)]
  · rw [syPos_ext, syPos_arg, innPos_eq, zPos_eq] at h
    exact absurd h.symm (extPos_ne_argPos (argPos (argPos b')) (argPos n))
  · rw [syPos_arg, syPos_ext, innPos_eq, zPos_eq] at h
    exact absurd h (extPos_ne_argPos (argPos (argPos b)) (argPos n'))
  · rw [syPos_arg, syPos_arg] at h
    rw [argPos_injective (zPos_injective h)]

/-- The first route never reaches the second route's copy of the argument.

    (p : ℕ)                                        [one of the first route's positions]
    sxPos p ∉ z2Set                                [it reaches its own copy alone] -/
theorem sxPos_not_z2Set (p : ℕ) : sxPos p ∉ z2Set := by
  rcases pos_split2 p with ⟨e, rfl⟩ | ⟨a, rfl⟩
  · rw [sxPos_ext]
    rintro ⟨b, hb⟩
    exact xPos_not_zSet e ⟨argPos b, hb⟩
  · rw [sxPos_arg]
    rintro ⟨b, hb⟩
    exact extPos_ne_argPos a b (zPos_injective hb)

/-- The second route never reaches the first route's copy of the argument.

    (q : ℕ)                                        [one of the second route's positions]
    syPos q ∉ z1Set                                [it reaches its own copy alone] -/
theorem syPos_not_z1Set (q : ℕ) : syPos q ∉ z1Set := by
  rcases pos_split2 q with ⟨n, rfl⟩ | ⟨b, rfl⟩
  · rw [syPos_ext]
    rintro ⟨a, ha⟩
    rw [innPos_eq, zPos_eq] at ha
    exact extPos_ne_argPos (argPos (extPos a)) (argPos n) ha.symm
  · rw [syPos_arg]
    rintro ⟨a, ha⟩
    exact extPos_ne_argPos a b (zPos_injective ha).symm

/-- The first route's argument channel is the first copy.

    sxPos '' oddPos = z1Set                        [its odd positions are one copy] -/
theorem sxPos_image_oddPos : sxPos '' oddPos = z1Set := by
  ext p
  constructor
  · rintro ⟨q, ⟨a, rfl⟩, rfl⟩
    exact ⟨a, (sxPos_arg a).symm ▸ rfl⟩
  · rintro ⟨a, rfl⟩
    exact ⟨argPos a, ⟨a, rfl⟩, sxPos_arg a⟩

/-- The second route's argument channel is the second copy.

    syPos '' oddPos = z2Set                        [its odd positions are the other copy] -/
theorem syPos_image_oddPos : syPos '' oddPos = z2Set := by
  ext p
  constructor
  · rintro ⟨q, ⟨b, rfl⟩, rfl⟩
    exact ⟨b, (syPos_arg b).symm ▸ rfl⟩
  · rintro ⟨b, rfl⟩
    exact ⟨argPos b, ⟨b, rfl⟩, syPos_arg b⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO ROUTES SIDE BY SIDE
-- ═══════════════════════════════════════════════════════════════════

/-- Two routes laid on the five tracks, each with its own copy of the argument channel.

    (x y : Route R)                                [two weighted wire maps]
    Route R                                        [both at once, on one wire] -/
def sPair {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) : RouteAnyRing.Route R := fun i =>
  Sum.elim
    (fun e => Sum.elim
      (fun k => (x (extPos (extPos k))).map fun q => (sxPos q.1, q.2))
      (fun d => Sum.elim
        (fun a => (x (argPos a)).map fun q => (sxPos q.1, q.2))
        (fun b => (y (argPos b)).map fun q => (syPos q.1, q.2))
        (Equiv.natSumNatEquivNat.symm d))
      (Equiv.natSumNatEquivNat.symm e))
    (fun o => Sum.elim
      (fun m => (x (extPos (argPos m))).map fun q => (sxPos q.1, q.2))
      (fun n => (y (extPos n)).map fun q => (syPos q.1, q.2))
      (Equiv.natSumNatEquivNat.symm o))
    (Equiv.natSumNatEquivNat.symm i)

/-- On the outside the first route runs.

    (x y : Route R) (k : ℕ)                        [two routes and a position]
    sPair x y (extPos (extPos k))                                    [a step on the outside]
    = (x (extPos (extPos k))).map fun q => (sxPos q.1, q.2)          [is the first route] -/
theorem sPair_ee {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (k : ℕ) :
    sPair x y (extPos (extPos k))
      = (x (extPos (extPos k))).map fun q => (sxPos q.1, q.2) := by
  simp only [sPair, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- On the first copy of the argument channel the first route runs.

    (x y : Route R) (a : ℕ)                        [two routes and a position]
    sPair x y (extPos (argPos (extPos a)))                           [a step on the first copy]
    = (x (argPos a)).map fun q => (sxPos q.1, q.2)                   [is the first route] -/
theorem sPair_eae {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (a : ℕ) :
    sPair x y (extPos (argPos (extPos a)))
      = (x (argPos a)).map fun q => (sxPos q.1, q.2) := by
  simp only [sPair, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- On the second copy of the argument channel the second route runs.

    (x y : Route R) (b : ℕ)                        [two routes and a position]
    sPair x y (extPos (argPos (argPos b)))                           [a step on the second copy]
    = (y (argPos b)).map fun q => (syPos q.1, q.2)                   [is the second route] -/
theorem sPair_eaa {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (b : ℕ) :
    sPair x y (extPos (argPos (argPos b)))
      = (y (argPos b)).map fun q => (syPos q.1, q.2) := by
  simp only [sPair, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- On the asking track the first route runs.

    (x y : Route R) (m : ℕ)                        [two routes and a position]
    sPair x y (argPos (extPos m))                                    [a step on the asking track]
    = (x (extPos (argPos m))).map fun q => (sxPos q.1, q.2)          [is the first route] -/
theorem sPair_ae {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (m : ℕ) :
    sPair x y (argPos (extPos m))
      = (x (extPos (argPos m))).map fun q => (sxPos q.1, q.2) := by
  simp only [sPair, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- On the answering track the second route runs.

    (x y : Route R) (n : ℕ)                        [two routes and a position]
    sPair x y (argPos (argPos n))                  [a step on the answering track]
    = (y (extPos n)).map fun q => (syPos q.1, q.2) [is the second route] -/
theorem sPair_aa {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (n : ℕ) :
    sPair x y (argPos (argPos n)) = (y (extPos n)).map fun q => (syPos q.1, q.2) := by
  simp only [sPair, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- **The pair is the first route on its own three tracks.**

    (x y : Route R) (p : ℕ)                        [two routes and a position]
    sPair x y (sxPos p) = (x p).map fun q => (sxPos q.1, q.2)   [that route, laid out] -/
theorem sPair_x {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (p : ℕ) :
    sPair x y (sxPos p) = (x p).map fun q => (sxPos q.1, q.2) := by
  rcases pos_split p with ⟨k, rfl⟩ | ⟨m, rfl⟩ | ⟨a, rfl⟩
  · rw [sxPos_ext, xPos_ext, sPair_ee]
  · rw [sxPos_ext, xPos_arg, outPos_eq, sPair_ae]
  · rw [sxPos_arg, zPos_eq, sPair_eae]

/-- **The pair is the second route on its own two tracks.**

    (x y : Route R) (q : ℕ)                        [two routes and a position]
    sPair x y (syPos q) = (y q).map fun t => (syPos t.1, t.2)   [that route, laid out] -/
theorem sPair_y {R : Type u} [CommRing R] (x y : RouteAnyRing.Route R) (q : ℕ) :
    sPair x y (syPos q) = (y q).map fun t => (syPos t.1, t.2) := by
  rcases pos_split2 q with ⟨n, rfl⟩ | ⟨b, rfl⟩
  · rw [syPos_ext, innPos_eq, sPair_aa]
  · rw [syPos_arg, zPos_eq, sPair_eaa]

-- ═══════════════════════════════════════════════════════════════════
-- THE DOUBLED ARGUMENT
-- ═══════════════════════════════════════════════════════════════════

/-- The argument layer on the argument channel.

    (W : Route R) (d : ℕ)                          [a route and a position]
    bArg W (zPos d) = (W d).map fun q => (zPos q.1, q.2)   [the argument runs there] -/
theorem bArg_zPos {R : Type u} [CommRing R] (W : RouteAnyRing.Route R) (d : ℕ) :
    bArg W (zPos d) = (W d).map fun q => (zPos q.1, q.2) := by
  rw [zPos_eq, bArg_ext, rtensor_arg]
  cases hw : W d with
  | none => rfl
  | some q => simp only [Option.map_some, zPos_eq]

/-- The doubled argument is the first route's own argument, on its own tracks.

    (z : Route R) (p : ℕ)                          [a route and a position]
    bArg (rtensor z z) (sxPos p)                                     [the layer where it sits]
    = ((rtensor rid z) p).map fun q => (sxPos q.1, q.2)              [is its own plug's layer] -/
theorem bArg_sxPos {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) (p : ℕ) :
    bArg (RouteAnyRing.rtensor z z) (sxPos p) =
      ((RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z) p).map fun q => (sxPos q.1, q.2) := by
  rcases pos_split2 p with ⟨e, rfl⟩ | ⟨a, rfl⟩
  · rw [sxPos_ext, bArg_xPos, rtensor_ext_id, Option.map_some, sxPos_ext]
  · rw [sxPos_arg, bArg_zPos, rtensor_ext, rtensor_arg]
    cases hz : z a with
    | none => rfl
    | some q => simp only [Option.map_some, sxPos_arg]

/-- The doubled argument is the second route's own argument, on its own tracks.

    (z : Route R) (q : ℕ)                          [a route and a position]
    bArg (rtensor z z) (syPos q)                                     [the layer where it sits]
    = ((rtensor rid z) q).map fun t => (syPos t.1, t.2)              [is its own plug's layer] -/
theorem bArg_syPos {R : Type u} [CommRing R] (z : RouteAnyRing.Route R) (q : ℕ) :
    bArg (RouteAnyRing.rtensor z z) (syPos q) =
      ((RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) z) q).map fun t => (syPos t.1, t.2) := by
  rcases pos_split2 q with ⟨n, rfl⟩ | ⟨b, rfl⟩
  · rw [syPos_ext, innPos_eq, bArg_arg, rtensor_ext_id, Option.map_some, syPos_ext, innPos_eq]
  · rw [syPos_arg, bArg_zPos, rtensor_arg, rtensor_arg]
    cases hz : z b with
    | none => rfl
    | some t => simp only [Option.map_some, syPos_arg]

-- ═══════════════════════════════════════════════════════════════════
-- THE MACHINE
-- ═══════════════════════════════════════════════════════════════════

/-- The pair with the doubled argument, before the wire.

    (σ : R) (x y z : Route R)                      [the parameter and three routes]
    rcomp σ (sPair x y) (bArg (rtensor z z))       [both routes, then both copies]
    : Route R                                      [the machine without its wire] -/
def sPlug {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) :
    RouteAnyRing.Route R :=
  RouteAnyRing.rcomp σ (sPair x y) (bArg (RouteAnyRing.rtensor z z))

/-- The machine on the first route's tracks is that route with the argument plugged.

    (σ : R) (x y z : Route R) (p : ℕ)              [the parameter, three routes, a position]
    sPlug σ x y z (sxPos p)                                          [a step where it sits]
    = (plug σ x z p).map fun q => (sxPos q.1, q.2)                   [is its own application] -/
theorem sPlug_x {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) (p : ℕ) :
    sPlug σ x y z (sxPos p) =
      (ApplicationWiring.plug σ x z p).map fun q => (sxPos q.1, q.2) := by
  rw [sPlug, rcomp_embed (sPair_x x y) (bArg_sxPos z) p, ApplicationWiring.plug]

/-- The machine on the second route's tracks is that route with the argument plugged.

    (σ : R) (x y z : Route R) (q : ℕ)              [the parameter, three routes, a position]
    sPlug σ x y z (syPos q)                                          [a step where it sits]
    = (plug σ y z q).map fun t => (syPos t.1, t.2)                   [is its own application] -/
theorem sPlug_y {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) (q : ℕ) :
    sPlug σ x y z (syPos q) =
      (ApplicationWiring.plug σ y z q).map fun t => (syPos t.1, t.2) := by
  rw [sPlug, rcomp_embed (sPair_y x y) (bArg_syPos z) q, ApplicationWiring.plug]

/-- Collapsing the doubled argument on the first route's tracks is its relation-first trace.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    sPlug [the paired substitution machine with both argument copies]
    sPair [two stored routes on their five tracks]
    bArg [the outer argument layer]
    rtensor [put the outer argument into both copies]
    ApplicationWiring.plug [run a route, then its supplied argument]
    Traces [the union over every path length through a route]
    relRuns_embed_exists_iff [embedding transport with an arbitrary endpoint]
    relRuns_oddPos_iff_traces [an odd-loop path is a route trace]
    zSet [both copies of the outer argument channel]
    xPos [the first stored route's wire positions]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    x [the first stored route]
    y [the second stored route]
    z [the duplicated outer argument]
    p [the initial first-route position]
    c [the arbitrary final machine position]
    q [the final first-route position]
    w [the accumulated path weight] -/
theorem relSPlug_collapse_x {R : Type u} [CommRing R] (σ : R)
    (x y z : RouteAnyRing.Route R) (p c : ℕ) (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (sPlug σ x y z)) zSet (xPos p) c w ↔
      ∃ q, c = xPos q ∧ RouteAnyRing.Traces σ (ApplicationWiring.plug σ x z) p q w := by
  have hEdge : ∀ a c w,
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (sPlug σ x y z)) z2Set
        (sxPos a) c w ↔
        ∃ b, c = sxPos b ∧ RelationalPaths.routeGraph (ApplicationWiring.plug σ x z) a b w := by
    intro a c w
    have hExit : ∀ j v, sPlug σ x y z (sxPos a) = some (j, v) → j ∉ z2Set := by
      intro j v hj
      rw [sPlug_x] at hj
      obtain ⟨⟨b, u⟩, _, hcb⟩ := Option.map_eq_some_iff.mp hj
      simp only [Prod.mk.injEq] at hcb
      exact hcb.1 ▸ sxPos_not_z2Set b
    rw [relRuns_eq_self hExit, sPlug_x]
    constructor
    · rintro h
      obtain ⟨⟨b, v⟩, hplug, hcb⟩ := Option.map_eq_some_iff.mp h
      simp only [Prod.mk.injEq] at hcb
      exact ⟨b, hcb.1.symm, hcb.2 ▸ hplug⟩
    · rintro ⟨b, rfl, hplug⟩
      rw [hplug, Option.map_some]
  have hEmbed := RelationalPaths.relRuns_embed_exists_iff (σ := σ) sxPos_injective hEdge
    (S := RelationalPaths.oddPos) (i := extPos p) (c := c) (w := w)
  have hOdd : RelationalPaths.oddPos = oddPos := by rfl
  rw [hOdd, sxPos_image_oddPos, sxPos_ext,
    ← RelationalPaths.relRuns_vanishing z1_disjoint_z2, ← zSet_split] at hEmbed
  constructor
  · intro h
    obtain ⟨q, hc, hq⟩ := hEmbed.mp h
    obtain ⟨b, hqb, htrace⟩ := (relRuns_oddPos_iff_traces (σ := σ)
      (f := ApplicationWiring.plug σ x z) p q w).mp hq
    rw [hqb, sxPos_ext] at hc
    exact ⟨b, hc, htrace⟩
  · rintro ⟨b, hc, htrace⟩
    apply hEmbed.mpr
    refine ⟨extPos b, ?_, ?_⟩
    · rw [sxPos_ext]
      exact hc
    · exact (relRuns_oddPos_iff_traces (σ := σ)
        (f := ApplicationWiring.plug σ x z) p (extPos b) w).mpr ⟨b, rfl, htrace⟩

/-- Collapsing the doubled argument on the second route's tracks is its relation-first trace.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    sPlug [the paired substitution machine with both argument copies]
    sPair [two stored routes on their five tracks]
    bArg [the outer argument layer]
    rtensor [put the outer argument into both copies]
    ApplicationWiring.plug [run a route, then its supplied argument]
    Traces [the union over every path length through a route]
    relRuns_embed_exists_iff [embedding transport with an arbitrary endpoint]
    relRuns_oddPos_iff_traces [an odd-loop path is a route trace]
    zSet [both copies of the outer argument channel]
    innPos [the second stored route's wire positions]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions]
    x [the first stored route]
    y [the second stored route]
    z [the duplicated outer argument]
    n [the initial second-route position]
    c [the arbitrary final machine position]
    q [the final second-route position]
    w [the accumulated path weight] -/
theorem relSPlug_collapse_y {R : Type u} [CommRing R] (σ : R)
    (x y z : RouteAnyRing.Route R) (n c : ℕ) (w : AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (sPlug σ x y z)) zSet (innPos n) c w ↔
      ∃ q, c = innPos q ∧ RouteAnyRing.Traces σ (ApplicationWiring.plug σ y z) n q w := by
  have hEdge : ∀ a c w,
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (sPlug σ x y z)) z1Set
        (syPos a) c w ↔
        ∃ b, c = syPos b ∧ RelationalPaths.routeGraph (ApplicationWiring.plug σ y z) a b w := by
    intro a c w
    have hExit : ∀ j v, sPlug σ x y z (syPos a) = some (j, v) → j ∉ z1Set := by
      intro j v hj
      rw [sPlug_y] at hj
      obtain ⟨⟨b, u⟩, _, hcb⟩ := Option.map_eq_some_iff.mp hj
      simp only [Prod.mk.injEq] at hcb
      exact hcb.1 ▸ syPos_not_z1Set b
    rw [relRuns_eq_self hExit, sPlug_y]
    constructor
    · rintro h
      obtain ⟨⟨b, v⟩, hplug, hcb⟩ := Option.map_eq_some_iff.mp h
      simp only [Prod.mk.injEq] at hcb
      exact ⟨b, hcb.1.symm, hcb.2 ▸ hplug⟩
    · rintro ⟨b, rfl, hplug⟩
      rw [hplug, Option.map_some]
  have hEmbed := RelationalPaths.relRuns_embed_exists_iff (σ := σ) syPos_injective hEdge
    (S := RelationalPaths.oddPos) (i := extPos n) (c := c) (w := w)
  have hOdd : RelationalPaths.oddPos = oddPos := by rfl
  rw [hOdd, syPos_image_oddPos, syPos_ext,
    ← RelationalPaths.relRuns_vanishing z2_disjoint_z1, ← zSet_split'] at hEmbed
  constructor
  · intro h
    obtain ⟨q, hc, hq⟩ := hEmbed.mp h
    obtain ⟨b, hqb, htrace⟩ := (relRuns_oddPos_iff_traces (σ := σ)
      (f := ApplicationWiring.plug σ y z) n q w).mp hq
    rw [hqb, syPos_ext] at hc
    exact ⟨b, hc, htrace⟩
  · rintro ⟨b, hc, htrace⟩
    apply hEmbed.mpr
    refine ⟨extPos b, ?_, ?_⟩
    · rw [syPos_ext]
      exact hc
    · exact (relRuns_oddPos_iff_traces (σ := σ)
        (f := ApplicationWiring.plug σ y z) n (extPos b) w).mpr ⟨b, rfl, htrace⟩

/-- **Collapsing the argument channel applies the argument to the first route.**

@cite girard-1989

    (σ : ℝ) (x y z : Route) (p : ℕ)                [the parameter, three routes, a position]
    runsRoute σ (sPlug σ x y z) zSet (xPos p)                        [collapse both copies]
    = (rapp σ x z p).map fun q => (xPos q.1, q.2)                    [is the first route applied]

    So: the other copy of the argument is collapsed first and the first route does not notice,
    because it never reaches that copy. What is left is a single embedding, and collapsing along
    it is taking the trace of the first route's own application. -/
theorem sPlug_collapse_x (σ : ℝ) (x y z : Route) (p : ℕ) :
    runsRoute σ (sPlug σ x y z) zSet (xPos p)
      = (rapp σ x z p).map fun q => (xPos q.1, q.2) := by
  have hB : ∀ a, runsRoute σ (sPlug σ x y z) z2Set (sxPos a)
      = (ApplicationWiring.plug σ x z a).map fun q => (sxPos q.1, q.2) := by
    intro a
    have h : ∀ j w, sPlug σ x y z (sxPos a) = some (j, w) → j ∉ z2Set := by
      intro j w hj
      rw [sPlug_x (R := ℝ)] at hj
      obtain ⟨⟨c1, c2⟩, _, hq⟩ := Option.map_eq_some_iff.mp hj
      simp only [Prod.mk.injEq] at hq
      obtain ⟨h1, _⟩ := hq
      subst h1
      exact sxPos_not_z2Set c1
    rw [runsRoute_eq_self h, sPlug_x (R := ℝ)]
  have h := runsRoute_embed (σ := σ) sxPos_injective hB (S := oddPos) (extPos p)
  rw [sxPos_image_oddPos, sxPos_ext] at h
  rw [zSet_split, runsRoute_union z1_disjoint_z2, h, runsRoute_oddPos_traceRoute, rapp]
  rw [plug]
  cases ht : traceRoute σ (ApplicationWiring.plug σ x z) p with
  | none => rfl
  | some t => rw [Option.map_some, Option.map_some, Option.map_some, sxPos_ext]

/-- **Collapsing the argument channel applies the argument to the second route.**

@cite girard-1989

    (σ : ℝ) (x y z : Route) (n : ℕ)                [the parameter, three routes, a position]
    runsRoute σ (sPlug σ x y z) zSet (innPos n)                      [collapse both copies]
    = (rapp σ y z n).map fun t => (innPos t.1, t.2)                  [is the second route applied]

    So: the same again with the copies exchanged, which is where the two uses of the argument
    are kept apart. -/
theorem sPlug_collapse_y (σ : ℝ) (x y z : Route) (n : ℕ) :
    runsRoute σ (sPlug σ x y z) zSet (innPos n)
      = (rapp σ y z n).map fun t => (innPos t.1, t.2) := by
  have hB : ∀ b, runsRoute σ (sPlug σ x y z) z1Set (syPos b)
      = (ApplicationWiring.plug σ y z b).map fun t => (syPos t.1, t.2) := by
    intro b
    have h : ∀ j w, sPlug σ x y z (syPos b) = some (j, w) → j ∉ z1Set := by
      intro j w hj
      rw [sPlug_y (R := ℝ)] at hj
      obtain ⟨⟨c1, c2⟩, _, hq⟩ := Option.map_eq_some_iff.mp hj
      simp only [Prod.mk.injEq] at hq
      obtain ⟨h1, _⟩ := hq
      subst h1
      exact syPos_not_z1Set c1
    rw [runsRoute_eq_self h, sPlug_y (R := ℝ)]
  have h := runsRoute_embed (σ := σ) syPos_injective hB (S := oddPos) (extPos n)
  rw [syPos_image_oddPos, syPos_ext] at h
  rw [zSet_split', runsRoute_union z2_disjoint_z1, h, runsRoute_oddPos_traceRoute, rapp]
  rw [plug]
  cases ht : traceRoute σ (ApplicationWiring.plug σ y z) n with
  | none => rfl
  | some t => rw [Option.map_some, Option.map_some, Option.map_some, syPos_ext]

-- ═══════════════════════════════════════════════════════════════════
-- THE STORED ROUTE AND THE EQUATION
-- ═══════════════════════════════════════════════════════════════════

/-- The two routes side by side with the wire between them.

    (σ : R) (x y : Route R)                        [the parameter and two routes]
    rcomp σ (sPair x y) swapRoute                  [both routes, then the wire]
    : Route R                                      [one route on the combined wire] -/
def sStore {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R) : RouteAnyRing.Route R :=
  RouteAnyRing.rcomp σ (sPair x y) (swapRoute (R := R))

/-- The unrestricted substitution relation before its outer argument is supplied.

    SSubstitute [the unbounded finite-path substitution relation]
    Traces [the union over every path length through a route]
    sStore [the stored substitution route]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    x [the first stored route]
    y [the second stored route] -/
def SSubstitute {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R) :=
  RouteAnyRing.Traces σ (sStore σ x y)

/-- **The substituting combinator's stored route.**

@cite curry-1930

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    traceRoute σ (sStore σ x y)                    [collapse the wire between them]
    : Route                                        [what is left offers both copies at once] -/
def sComb (σ : ℝ) (x y : Route) : Route := traceRoute σ (sStore σ x y)

/-- The machine carrying all four loops at once.

    (σ : R) (x y z : Route R)                      [the parameter and three routes]
    rcomp σ (sStore σ x y) (bArg (rtensor z z))    [the stored route with both copies plugged]
    : Route R                                      [one route carrying every loop] -/
def sBig {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) : RouteAnyRing.Route R :=
  RouteAnyRing.rcomp σ (sStore σ x y) (bArg (RouteAnyRing.rtensor z z))

/-- The machine with its wire last.

    (σ : R) (x y z : Route R)                      [the parameter and three routes]
    sBig σ x y z = rcomp σ (sPlug σ x y z) swapRoute   [both routes, both copies, then the wire] -/
theorem sBig_eq {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R) :
    sBig σ x y z = RouteAnyRing.rcomp σ (sPlug σ x y z) (swapRoute (R := R)) := by
  rw [sBig, sStore, sPlug, RouteAnyRing.rcomp_assoc, swap_arg_comm,
    RouteAnyRing.rcomp_assoc]

/-- The relation-first left substitution reading is a path through the full stored machine.

    RelApp [the relation-first application relation]
    SSubstitute [the unbounded finite-path substitution relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rtensor [put the outer argument into both copies]
    sBig [the stored substitution machine with both arguments]
    zSet [the outer argument channel]
    oddPos [the stored middle-track loop]
    extPos [a position of the wire outside both loops]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∪ [one of the two loop layers]
    ↔ [exactly when, true in both directions]
    x [the first stored route]
    y [the second stored route]
    z [the duplicated outer argument]
    i [the initial position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relSLeft_reading {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R)
    (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (SSubstitute σ x y) (RelationalPaths.routeGraph (RouteAnyRing.rtensor z z)) i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (sBig σ x y z))
        (zSet ∪ RelationalPaths.oddPos) (extPos (extPos i)) (extPos (extPos j)) w :=
  relOuter_reading σ (sStore σ x y) (RouteAnyRing.rtensor z z) i j w

/-- The relation-first right substitution reading is a path through the full stored machine.

    RelApp [the relation-first application relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    Traces [the union over every path length through a route]
    ApplicationWiring.plug [run a route, then its supplied argument]
    relApplied_reading [the two stored applications as one machine path]
    relSPlug_collapse_x [the first collapsed substitution track]
    relSPlug_collapse_y [the second collapsed substitution track]
    sBig [the stored substitution machine with both arguments]
    outerSet [the asking loop positions]
    innerSet [the answering loop positions]
    zSet [the outer argument channel]
    extPos [a position of the wire outside both loops]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∪ [one of the three loop layers]
    ↔ [exactly when, true in both directions]
    x [the first stored route]
    y [the second stored route]
    z [the duplicated outer argument]
    i [the initial position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relSRight_reading {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R)
    (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (RouteAnyRing.Traces σ (ApplicationWiring.plug σ x z))
      (RouteAnyRing.Traces σ (ApplicationWiring.plug σ y z)) i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (sBig σ x y z))
        (outerSet ∪ (innerSet ∪ zSet)) (extPos (extPos i)) (extPos (extPos j)) w := by
  rw [sBig_eq]
  exact relApplied_reading σ (sPlug σ x y z)
    (RouteAnyRing.Traces σ (ApplicationWiring.plug σ x z))
    (RouteAnyRing.Traces σ (ApplicationWiring.plug σ y z))
    (relSPlug_collapse_x σ x y z) (relSPlug_collapse_y σ x y z) i j w

/-- The two relation-first substitution readings agree without choosing traced routes.

    RelApp [the relation-first application relation]
    SSubstitute [the unbounded finite-path substitution relation]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rtensor [put the outer argument into both copies]
    ApplicationWiring.plug [run a route, then its supplied argument]
    relApp_routeGraph [relation-first application of two route graphs]
    relSLeft_reading [the stored relation followed by the doubled argument]
    relSRight_reading [the two independently applied route relations]
    bracket_sets [the equality of the two loop-union orders]
    Route R [a weighted wire map over the coefficient ring]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ↔ [exactly when, true in both directions]
    x [the first stored route]
    y [the second stored route]
    z [the duplicated outer argument]
    i [the initial position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relSubstitution {R : Type u} [CommRing R] (σ : R) (x y z : RouteAnyRing.Route R)
    (i j : ℕ) (w : AnyRing.RingCell R) :
    RelApp σ (SSubstitute σ x y) (RelationalPaths.routeGraph (RouteAnyRing.rtensor z z)) i j w ↔
      RelApp σ
        (RelApp σ (RelationalPaths.routeGraph x) (RelationalPaths.routeGraph z))
        (RelApp σ (RelationalPaths.routeGraph y) (RelationalPaths.routeGraph z)) i j w := by
  have hx : RelApp σ (RelationalPaths.routeGraph x) (RelationalPaths.routeGraph z) =
      RouteAnyRing.Traces σ (ApplicationWiring.plug σ x z) := by
    funext p q u
    exact propext (relApp_routeGraph σ x z p q u)
  have hy : RelApp σ (RelationalPaths.routeGraph y) (RelationalPaths.routeGraph z) =
      RouteAnyRing.Traces σ (ApplicationWiring.plug σ y z) := by
    funext p q u
    exact propext (relApp_routeGraph σ y z p q u)
  have hOdd : RelationalPaths.oddPos = oddPos := by rfl
  rw [hx, hy, relSLeft_reading, relSRight_reading, hOdd, bracket_sets]

/-- **The left reading: collapse the wire, then both copies of the argument.**

    (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell)   [parameter, three routes, a step]
    rapp σ (sComb σ x y) (rtensor z z) i = some (j, w)   [the stored route, given both copies]
    ↔                                                    [exactly when]
    Runs σ (sBig σ x y z) (zSet ∪ oddPos)                [a path through every loop]
      (extPos (extPos i)) (extPos (extPos j)) w          [from the outside to the outside] -/
theorem sLeft_reading (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell) :
    rapp σ (sComb σ x y) (rtensor z z) i = some (j, w)
      ↔ Runs σ (sBig σ x y z) (zSet ∪ oddPos) (extPos (extPos i)) (extPos (extPos j)) w :=
  outer_reading σ (sStore σ x y) (rtensor z z) i j w

/-- **The right reading: collapse the copies, the answering track, then the asking track.**

    (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell)   [parameter, three routes, a step]
    rapp σ (rapp σ x z) (rapp σ y z) i = some (j, w)     [each applied, then one to the other]
    ↔                                                    [exactly when]
    Runs σ (sBig σ x y z) (outerSet ∪ (innerSet ∪ zSet)) [a path through every loop]
      (extPos (extPos i)) (extPos (extPos j)) w          [from the outside to the outside] -/
theorem sRight_reading (σ : ℝ) (x y z : Route) (i j : ℕ) (w : Cell) :
    rapp σ (rapp σ x z) (rapp σ y z) i = some (j, w)
      ↔ Runs σ (sBig σ x y z) (outerSet ∪ (innerSet ∪ zSet))
          (extPos (extPos i)) (extPos (extPos j)) w := by
  rw [sBig_eq]
  exact applied_reading (sPlug_collapse_x σ x y z) (sPlug_collapse_y σ x y z) i j w

/-- **S x y z = x z (y z)**, as an equation between routes.

@cite schonfinkel-1924
@cite curry-1930
@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x y z : Route)                        [the parameter and three routes]
    rapp σ (sComb σ x y) (rtensor z z)             [the stored route given the argument twice]
    = rapp σ (rapp σ x z) (rapp σ y z)             [is each route applied, then one to the other]

    So: the substituting combinator holds at this carrier, with the argument arriving as the two
    copies its two uses consume, which is the form `Combinator/Contraction.lean` fixed for the
    contraction. With `Combinator/Interaction.lean`'s `I` and `K`, that file's contraction, and
    `Combinator/Compositor.lean` and `Combinator/Exchange.lean`, the five wiring combinators
    `B`, `C`, `I`, `K` and `W` of a linear combinatory algebra hold at this carrier, and so does
    `S`. What is not proved here is the other three, `D`, `delta` and `F`, which speak about the
    exponential rather than about the wiring. -/
theorem substitution (σ : ℝ) (x y z : Route) :
    rapp σ (sComb σ x y) (rtensor z z) = rapp σ (rapp σ x z) (rapp σ y z) := by
  refine route_ext fun i j w => ?_
  rw [sLeft_reading, sRight_reading, bracket_sets]

end Combinator
end Apeiron

end
