import Apeiron.Combinator.Linear.Core.Tracks
import Apeiron.Combinator.RelationalApplication

/-!
# Using an argument twice, with the copies coming from the tensor

@name Contraction
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite joyal-street-verity-1996
@grade proved

`Combinator/Tracks.lean` finished the general machinery and recorded the finding this file acts
on: a combinator that uses an argument twice needs two copies of it, and two copies is what the
tensor gives. `rtensor y y` is two copies of `y` side by side, so the contraction combinator can
be stated without the exponential at all.

## Root

| from | here |
|---|---|
| `mul_assoc`, `one_mul` through `runs_conj` and `traces_flatten` | `wIdx_comp`, `plug_conj` |

The layout and the relabelling are arithmetic on the counting numbers and say nothing about the
parameter, which is `theory/NOVELTY.md`'s A1 spectator test again. What the parameter supplies is
the multiplication accumulating along each path, and it enters through the machinery of
`Combinator/Vanishing.lean` rather than here.

## The relabelling

A stored route that uses its argument twice has one argument channel, and the route it stores has
two. `wIdx` is the change of layout, sending the stored route's positions to the argument's:

```
   even k            to   even (even k)          the outside
   odd (even v)      to   odd v                  the first use
   odd (odd u)       to   even (odd u)           the second use
```

It is built by composing `Equiv.sumAssoc`, `Equiv.sumComm` and the conversions
`Equiv.natSumNatEquivNat` gives, so the round trips come from Mathlib rather than by hand.
Splitting the one channel by even and odd rather than by `Nat.pair` is what makes the argument
`rtensor y y` and not `bang y`.

## What is proved here

`contraction` is `W x y = x y y`: a stored route that uses its argument twice, applied to two
copies of that argument, is the argument applied twice in turn. Four steps carry it.
`contraction_left` turns the left into one path relation over the two loops the two uses run on.
`runs_vanishing` collapses the first use. `collapse_step` says what that collapse is, and it is
the right side's own applied route: the collapsed route is not `plug σ x y`'s own collapse, since
the doubled argument applies `y` at the second use where the ordinary one does not, so the
collapsed step ends one application of `y` further on, which is the same place the right side
applies its outer copy. `runs_embed` then carries the result down to the second use, which is the
first pushed one level out.

The parameter is not used anywhere in the argument. Every weight that arises is the argument's
own or the unit, so the equation holds at every setting, and what the parameter decides stays in
`Combinator/Interaction.lean`'s `weight_blocks_the_loop_iff_disc_nonneg`.

## Which of the eight equations hold at this carrier

Proved: `I x = x` and `K x y = x` in `Combinator/Duplication.lean`, `K x !y = x` there too, and
`W x y = x y y` here, in its binary form with the two copies from the tensor.

Not proved: `B x y z = x (y z)`, `C x y z = x z y` and `S x y z = x z (y z)`. `B` and `C` are
linear and need no contraction, so they need this file's relabelling and nothing from `dupArg`.
`S` composes and contracts at once, so it needs both.

-/

namespace Apeiron
namespace Combinator

open Trichotomy

universe u

-- ═══════════════════════════════════════════════════════════════════
-- THE RELABELLING
-- ═══════════════════════════════════════════════════════════════════

/-- A position of the argument channel.

    (u : ℕ)                                        [a position]
    Equiv.natSumNatEquivNat (Sum.inr u)            [its odd copy]
    : ℕ                                            [a position of the wire]

    So: `extPos` names the even half and this names the odd one, which is the channel a route
    faces its argument through. `outPos u` is `argPos (extPos u)` and `innPos v` is
    `argPos (argPos v)`. -/
def argPos (u : ℕ) : ℕ := Equiv.natSumNatEquivNat (Sum.inr u)

/-- The wire read as one outside track and two loop tracks.

    ℕ ≃ ℕ ⊕ (ℕ ⊕ ℕ)                               [outside, first use, second use] -/
def tripleSplit : ℕ ≃ ℕ ⊕ (ℕ ⊕ ℕ) :=
  Equiv.natSumNatEquivNat.symm.trans
    (Equiv.sumCongr (Equiv.refl ℕ) Equiv.natSumNatEquivNat.symm)

/-- The wire read as two outside tracks and one loop track.

    (ℕ ⊕ ℕ) ⊕ ℕ ≃ ℕ                               [the argument's own layout] -/
def pairJoin : (ℕ ⊕ ℕ) ⊕ ℕ ≃ ℕ :=
  (Equiv.sumCongr Equiv.natSumNatEquivNat (Equiv.refl ℕ)).trans Equiv.natSumNatEquivNat

/-- The change of layout a contraction needs.

@cite girard-1989

    ℕ ≃ ℕ                                          [the stored route's labels to the argument's]

    So: the stored route has one argument channel split by even and odd, and the route it stores
    has two, one on its own odd half and one on the odd half of its even half. This exchanges
    them. -/
def wIdx : ℕ ≃ ℕ :=
  tripleSplit.trans
    (((Equiv.sumCongr (Equiv.refl ℕ) (Equiv.sumComm ℕ ℕ)).trans
      (Equiv.sumAssoc ℕ ℕ ℕ).symm).trans pairJoin)

/-- The outside track goes to the outside of the outside.

    (k : ℕ)                                                    [a position]
    wIdx (extPos k) = extPos (extPos k)                        [it lands twice outside] -/
theorem wIdx_ext (k : ℕ) : wIdx (extPos k) = extPos (extPos k) := by
  simp [wIdx, tripleSplit, pairJoin, extPos]

/-- The first loop track goes to the argument's own loop.

    (v : ℕ)                                                    [a position]
    wIdx (outPos v) = argPos v                                 [it lands on the argument's loop] -/
theorem wIdx_out (v : ℕ) : wIdx (outPos v) = argPos v := by
  simp [wIdx, tripleSplit, pairJoin, outPos, argPos]

/-- The second loop track goes to the loop inside the argument's outside.

    (u : ℕ)                                                    [a position]
    wIdx (innPos u) = extPos (argPos u)                        [it lands there] -/
theorem wIdx_inn (u : ℕ) : wIdx (innPos u) = extPos (argPos u) := by
  simp [wIdx, tripleSplit, pairJoin, innPos, extPos, argPos]

/-- The relabelling read backwards, on the outside.

    (k : ℕ)                                        [a position]
    wIdx.symm (extPos (extPos k)) = extPos k       [reading back] -/
theorem wIdx_symm_ext (k : ℕ) : wIdx.symm (extPos (extPos k)) = extPos k :=
  (Equiv.symm_apply_eq wIdx).mpr (wIdx_ext k).symm

/-- The relabelling read backwards, on the argument's loop.

    (v : ℕ)                                        [a position]
    wIdx.symm (argPos v) = outPos v                [reading back] -/
theorem wIdx_symm_arg (v : ℕ) : wIdx.symm (argPos v) = outPos v :=
  (Equiv.symm_apply_eq wIdx).mpr (wIdx_out v).symm

/-- The relabelling read backwards, on the loop inside the outside.

    (u : ℕ)                                        [a position]
    wIdx.symm (extPos (argPos u)) = innPos u       [reading back] -/
theorem wIdx_symm_inn (u : ℕ) : wIdx.symm (extPos (argPos u)) = innPos u :=
  (Equiv.symm_apply_eq wIdx).mpr (wIdx_inn u).symm

-- ═══════════════════════════════════════════════════════════════════
-- PLUGGING THROUGH A RELABELLING
-- ═══════════════════════════════════════════════════════════════════

/-- Plugging a relabelled route is relabelling a plug.

@cite girard-1989

    (σ : ℝ) (e : ℕ ≃ ℕ) (a A : Route)              [parameter, relabelling, route, argument]
    plug σ (conjRoute e a) A                       [plug the relabelled route]
    = conjRoute e (rcomp σ a                       [is the relabelling of the route composed]
        (conjRoute e.symm (rtensor rid A)))        [with the argument in the route's own labels]

    So: `conjRoute_rcomp` and `conjRoute_symm` together. This is what puts a combinator's stored
    route back into its argument's labels, where `traces_conj` can read its trace. -/
theorem plug_conj (σ : ℝ) (e : ℕ ≃ ℕ) (a A : Route) :
    plug σ (conjRoute e a) A
      = conjRoute e (rcomp σ a (conjRoute e.symm (rtensor rid A))) := by
  rw [conjRoute_rcomp, conjRoute_symm, plug]

-- ═══════════════════════════════════════════════════════════════════
-- THE NAMED POSITIONS, SO NOTHING NORMALISES AWAY
-- ═══════════════════════════════════════════════════════════════════

/-- The tensor at an outside position.

    (a b : Route R) (k : ℕ)                                    [two routes and a position]
    rtensor a b (extPos k) = (a k).map fun q => (extPos q.1, q.2)   [the first route runs] -/
theorem rtensor_ext {R : Type u} [CommRing R] (a b : RouteAnyRing.Route R) (k : ℕ) :
    RouteAnyRing.rtensor a b (extPos k) = (a k).map fun q => (extPos q.1, q.2) := by
  rw [RouteAnyRing.rtensor, extPos, Equiv.symm_apply_apply]
  rfl

/-- The tensor at an argument position.

    (a b : Route R) (k : ℕ)                                    [two routes and a position]
    rtensor a b (argPos k) = (b k).map fun q => (argPos q.1, q.2)   [the second route runs] -/
theorem rtensor_arg {R : Type u} [CommRing R] (a b : RouteAnyRing.Route R) (k : ℕ) :
    RouteAnyRing.rtensor a b (argPos k) = (b k).map fun q => (argPos q.1, q.2) := by
  rw [RouteAnyRing.rtensor, argPos, Equiv.symm_apply_apply]
  rfl

/-- The first loop track is the argument's own, one level down.

    (v : ℕ)                                        [a position]
    outPos v = argPos (extPos v)                   [odd of even] -/
theorem outPos_eq (v : ℕ) : outPos v = argPos (extPos v) := rfl

/-- The second loop track is the argument's own, twice.

    (u : ℕ)                                        [a position]
    innPos u = argPos (argPos u)                   [odd of odd] -/
theorem innPos_eq (u : ℕ) : innPos u = argPos (argPos u) := rfl

-- ═══════════════════════════════════════════════════════════════════
-- THE ARGUMENT A CONTRACTION SEES
-- ═══════════════════════════════════════════════════════════════════

/-- The argument a contraction's stored route sees, in the argument's own labels.

@cite girard-1989

    (y : Route R)                                  [a route]
    Route R                                        [the identity outside, y on each loop track]

    So: the stored route uses its argument twice, and this is what those two uses look like from
    inside the argument's own labels: the same `y` on the argument's own loop and on the loop
    sitting inside the argument's outside. `dupArg_is_tensor` below says this is exactly two
    copies of `y` side by side, read through the relabelling, which is why no exponential
    appears. -/
def dupArg {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) : RouteAnyRing.Route R := fun i =>
  Sum.elim
    (fun p => Sum.elim (fun k => some (extPos (extPos k), Trichotomy.AnyRing.ringOne))
      (fun u => (y u).map fun q => (extPos (argPos q.1), q.2))
      (Equiv.natSumNatEquivNat.symm p))
    (fun v => (y v).map fun q => (argPos q.1, q.2))
    (Equiv.natSumNatEquivNat.symm i)

/-- Outside, the argument does nothing.

    (y : Route R) (k : ℕ)                                          [a route and a position]
    dupArg y (extPos (extPos k)) = some (extPos (extPos k), ringOne) [the outside passes through] -/
theorem dupArg_ext {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) (k : ℕ) :
    dupArg y (extPos (extPos k)) = some (extPos (extPos k), Trichotomy.AnyRing.ringOne) := by
  simp only [dupArg, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- On the first loop track the argument is the route.

    (y : Route R) (v : ℕ)                                      [a route and a position]
    dupArg y (argPos v) = (y v).map fun q => (argPos q.1, q.2) [it is y there] -/
theorem dupArg_arg {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) (v : ℕ) :
    dupArg y (argPos v) = (y v).map fun q => (argPos q.1, q.2) := by
  simp only [dupArg, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- On the second loop track the argument is the same route again.

    (y : Route R) (u : ℕ)                                              [a route and a position]
    dupArg y (extPos (argPos u))                                       [the argument there]
    = (y u).map fun q => (extPos (argPos q.1), q.2)                    [is y again] -/
theorem dupArg_inn {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) (u : ℕ) :
    dupArg y (extPos (argPos u)) = (y u).map fun q => (extPos (argPos q.1), q.2) := by
  simp only [dupArg, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- **The argument a contraction sees is two copies of its own, and no exponential.**

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    (y : Route)                                                  [a route]
    conjRoute wIdx (dupArg y)                      [read through the relabelling]
    = rtensor rid (rtensor y y)                    [it is two copies beside the identity]

    So: this is the finding `Combinator/Tracks.lean` recorded, now as an equation. A combinator
    that uses an argument twice needs two copies, `rtensor y y` is two copies, and the change of
    layout carries one into the other. The exponential is for keeping a duplicated argument
    itself duplicable and is not needed here. -/
theorem dupArg_is_tensor (y : Route) :
    conjRoute wIdx (dupArg y) = rtensor rid (rtensor y y) := by
  funext i
  rcases pos_trichotomy i with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨u, rfl⟩
  · rw [conjRoute, wIdx_ext, dupArg_ext, rtensor_ext, rid_apply]
    simp only [Option.map_some, wIdx_symm_ext]
  · rw [conjRoute, wIdx_out, dupArg_arg, outPos_eq, rtensor_arg, rtensor_ext]
    cases hy : y v with
    | none => simp
    | some p => simp only [Option.map_some, wIdx_symm_arg, outPos_eq]
  · rw [conjRoute, wIdx_inn, dupArg_inn, innPos_eq, rtensor_arg, rtensor_arg]
    cases hy : y u with
    | none => simp
    | some p => simp only [Option.map_some, wIdx_symm_inn, innPos_eq]

-- ═══════════════════════════════════════════════════════════════════
-- THE STORED ROUTE, PLUGGED WITH TWO COPIES
-- ═══════════════════════════════════════════════════════════════════

/-- The tensor read back into the stored route's labels is the doubled argument.

    (y : Route)                                                        [a route]
    conjRoute wIdx.symm (rtensor rid (rtensor y y)) = dupArg y         [reading back] -/
theorem conj_symm_tensor (y : Route) :
    conjRoute wIdx.symm (rtensor rid (rtensor y y)) = dupArg y := by
  rw [← dupArg_is_tensor y]
  have h := conjRoute_symm wIdx.symm (dupArg y)
  rwa [Equiv.symm_symm] at h

/-- **The stored route plugged with two copies, in the argument's own labels.**

@cite girard-1989

    (σ : ℝ) (x y : Route)                                        [the parameter and two routes]
    plug σ (conjRoute wIdx x) (rtensor y y)        [relabel x, plug two copies]
    = conjRoute wIdx (rcomp σ x (dupArg y))        [is x then the doubled argument, relabelled]

    So: `plug_conj` and `conj_symm_tensor` together. The left of `W x y = x y y` is now a single
    relabelled route built from `x` and `y` alone, which is what `traces_conj` needs
    before its trace can be read in `x`'s labels. -/
theorem plug_dup (σ : ℝ) (x y : Route) :
    plug σ (conjRoute wIdx x) (rtensor y y) = conjRoute wIdx (rcomp σ x (dupArg y)) := by
  rw [plug_conj, conj_symm_tensor]

/-- The positions of the second use, in the argument's labels.

    Set ℕ                                          [the loop the second use runs on] -/
def innArgSet : Set ℕ := {p | ∃ u, p = extPos (argPos u)}

/-- **The relabelling carries the loop onto the two uses.**

    wIdx '' oddPos = oddPos ∪ innArgSet            [one loop becomes the two uses]

    So: the stored route has one loop and the argument has two uses, and this says the change of
    layout takes the first onto the second. The two on the right are disjoint, one being odd and
    the other even, so `runs_vanishing` applies to them. -/
theorem wIdx_image_oddPos : wIdx '' oddPos = oddPos ∪ innArgSet := by
  ext p
  constructor
  · rintro ⟨q, hq, rfl⟩
    rw [oddPos_eq] at hq
    rcases hq with ⟨v, rfl⟩ | ⟨u, rfl⟩
    · exact Or.inl ⟨v, wIdx_out v⟩
    · exact Or.inr ⟨u, wIdx_inn u⟩
  · rintro (⟨v, rfl⟩ | ⟨u, rfl⟩)
    · exact ⟨outPos v, by rw [oddPos_eq]; exact Or.inl ⟨v, rfl⟩, wIdx_out v⟩
    · exact ⟨innPos u, by rw [oddPos_eq]; exact Or.inr ⟨u, rfl⟩, wIdx_inn u⟩

/-- The two uses do not meet.

    ∀ x, x ∈ innArgSet → x ∉ oddPos                [the second use is even, the first odd] -/
theorem innArgSet_disjoint_oddPos : ∀ x, x ∈ innArgSet → x ∉ oddPos := by
  rintro x ⟨u, rfl⟩ ⟨v, hv⟩
  rw [extPos, argPos] at hv
  exact absurd (Equiv.natSumNatEquivNat.injective hv) (by simp)

/-- **The left of the contraction equation, as a path relation.**

@cite girard-1989
@cite joyal-street-verity-1996

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ (plug σ (conjRoute wIdx x) (rtensor y y)) i j w   [the stored route on two copies]
    ↔                                              [exactly when]
    Runs σ (rcomp σ x (dupArg y))                  [there is a path through x]
      (oddPos ∪ innArgSet)                         [over the two loops the uses run on]
      (extPos (extPos i)) (extPos (extPos j)) w    [from the outside to the outside]

    So: `plug_dup`, `traces_conj` and `wIdx_image_oddPos` chained. The left of `W x y = x y y` is
    one path relation through `x`, over the two loops the two uses run on, with `y` supplied on
    each. What remains for the equation is that the right of it is the same relation, which is
    `runs_vanishing` on those two loops together with the identification of the inner collapse
    with `rapp σ x y`. -/
theorem contraction_left (σ : ℝ) (x y : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ (conjRoute wIdx x) (rtensor y y)) i j w ↔
      Runs σ (rcomp σ x (dupArg y)) (oddPos ∪ innArgSet)
        (extPos (extPos i)) (extPos (extPos j)) w := by
  rw [plug_dup, traces_conj, wIdx_image_oddPos, wIdx_ext, wIdx_ext]

-- ═══════════════════════════════════════════════════════════════════
-- SPLITTING THE DOUBLED ARGUMENT INTO THE FIRST USE AND THE SECOND
-- ═══════════════════════════════════════════════════════════════════

/-- Every position is outside twice, outside then a use, or a use.

    (p : ℕ)                                        [any position]
    ∃ k, p = extPos (extPos k)                     [it is outside twice]
    ∨ ∃ u, p = extPos (argPos u)                   [or outside then a use]
    ∨ ∃ v, p = argPos v                            [or a use] -/
theorem pos_split (p : ℕ) :
    (∃ k, p = extPos (extPos k)) ∨ (∃ u, p = extPos (argPos u)) ∨ (∃ v, p = argPos v) := by
  rcases hp : Equiv.natSumNatEquivNat.symm p with m | v
  · rcases hm : Equiv.natSumNatEquivNat.symm m with k | u
    · left
      refine ⟨k, ?_⟩
      rw [extPos, extPos, ← hm, Equiv.apply_symm_apply, ← hp, Equiv.apply_symm_apply]
    · right; left
      refine ⟨u, ?_⟩
      rw [extPos, argPos, ← hm, Equiv.apply_symm_apply, ← hp, Equiv.apply_symm_apply]
  · right; right
    refine ⟨v, ?_⟩
    rw [argPos, ← hp, Equiv.apply_symm_apply]

/-- Outside the loop is outside.

    (p : ℕ) (h : p ∉ oddPos)                       [a position off the loop]
    ∃ k, p = extPos k                              [it is an outside one] -/
theorem not_oddPos_ext {p : ℕ} (h : p ∉ oddPos) : ∃ k, p = extPos k := by
  rcases hp : Equiv.natSumNatEquivNat.symm p with k | v
  · exact ⟨k, by rw [extPos, ← hp, Equiv.apply_symm_apply]⟩
  · exact absurd ⟨v, by rw [← hp, Equiv.apply_symm_apply]⟩ h

/-- The tensor with the identity, at an outside position.

    (y : Route R) (m : ℕ)                                              [a route and a position]
    rtensor rid y (extPos m) = some (extPos m, ringOne)                [the outside passes] -/
theorem rtensor_ext_id {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) (m : ℕ) :
    RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) y (extPos m) =
      some (extPos m, Trichotomy.AnyRing.ringOne) := by
  rw [rtensor_ext, RouteAnyRing.rid, Option.map_some]

/-- The part of the doubled argument that acts on the second use alone.

@cite girard-1989

    (y : Route R)                                  [a route]
    Route R                                        [the identity on the first use, y on the
    second, read one level out]                    [so it fires only after the first is done]

    So: `dupArg_eq` says the doubled argument is the ordinary one followed by this. That is the
    whole of the difference between the two sides of the contraction equation: the left applies
    the second copy one step earlier than the right does, and the right applies it as its outer
    copy, which is the same place. -/
def outerArg {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) : RouteAnyRing.Route R := fun i =>
  Sum.elim (fun c => (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) y c).map
      fun q => (extPos q.1, q.2))
    (fun v => some (argPos v, Trichotomy.AnyRing.ringOne))
    (Equiv.natSumNatEquivNat.symm i)

/-- Outside, the second-use part is the ordinary argument one level out.

    (y : Route R) (c : ℕ)                                              [a route and a position]
    outerArg y (extPos c)                                              [the second-use part]
    = (rtensor rid y c).map fun q => (extPos q.1, q.2)                 [is the ordinary argument] -/
theorem outerArg_ext {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) (c : ℕ) :
    outerArg y (extPos c) =
      (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) y c).map fun q => (extPos q.1, q.2) := by
  simp only [outerArg, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- On the first use the second-use part does nothing.

    (y : Route R) (v : ℕ)                                      [a route and a position]
    outerArg y (argPos v) = some (argPos v, ringOne)           [the first use passes through] -/
theorem outerArg_arg {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) (v : ℕ) :
    outerArg y (argPos v) = some (argPos v, Trichotomy.AnyRing.ringOne) := by
  simp only [outerArg, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- **The doubled argument is the ordinary one, then the second use.**

@cite girard-1989

    (σ : R) (y : Route R)                                      [the parameter and a route]
    dupArg y = rcomp σ (rtensor rid y) (outerArg y)            [the two uses come in order]

    So: this is the bookkeeping the contraction equation turns on, as an equation. The doubled
    argument does the ordinary argument's job and then, one level out, the second copy. -/
theorem dupArg_eq {R : Type u} [CommRing R] (σ : R) (y : RouteAnyRing.Route R) :
    dupArg y = RouteAnyRing.rcomp σ (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) y)
      (outerArg y) := by
  funext i
  rcases pos_split i with ⟨k, rfl⟩ | ⟨u, rfl⟩ | ⟨v, rfl⟩
  · rw [dupArg_ext, RouteAnyRing.rcomp, rtensor_ext_id, Option.bind_some, outerArg_ext,
      rtensor_ext, RouteAnyRing.rid]
    simp
  · rw [dupArg_inn, RouteAnyRing.rcomp, rtensor_ext_id, Option.bind_some, outerArg_ext,
      rtensor_arg]
    cases hy : y u with
    | none => simp
    | some p => simp only [Option.map_some, Trichotomy.AnyRing.ringOne_mul]
  · rw [dupArg_arg, RouteAnyRing.rcomp, rtensor_arg]
    cases hy : y v with
    | none => simp
    | some p =>
      simp only [Option.map_some, Option.bind_some, outerArg_arg,
        Trichotomy.AnyRing.ringMul_one]

-- ═══════════════════════════════════════════════════════════════════
-- THE COLLAPSE, AND THE CONTRACTION EQUATION
-- ═══════════════════════════════════════════════════════════════════

/-- The trace as a path, in the named outside positions.

    (σ : ℝ) (f : Route) (i j : ℕ) (w : Cell)                   [parameter, route, path]
    Traces σ f i j w ↔ Runs σ f oddPos (extPos i) (extPos j) w [the same, named] -/
theorem traces_iff_runs_ext (σ : ℝ) (f : Route) (i j : ℕ) (w : Cell) :
    Traces σ f i j w ↔ Runs σ f oddPos (extPos i) (extPos j) w :=
  traces_iff_runs σ f i j w

/-- The second-use part does nothing on the first use.

    (y : Route R)                                  [a route]
    ∀ s, s ∈ oddPos → outerArg y s = some (s, ringOne) [it is the identity there] -/
theorem outerArg_id_on_oddPos {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) :
    ∀ s, s ∈ oddPos → outerArg y s = some (s, Trichotomy.AnyRing.ringOne) := by
  rintro s ⟨v, rfl⟩
  exact outerArg_arg y v

/-- The second-use part cannot put a path back on the first use.

    (y : Route R)                                              [a route]
    ∀ m j w, outerArg y m = some (j, w) → j ∈ oddPos → m ∈ oddPos   [it cannot enter] -/
theorem outerArg_not_enter {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) :
    ∀ m j w, outerArg y m = some (j, w) → j ∈ oddPos → m ∈ oddPos := by
  intro m j w hE hj
  rcases hm : Equiv.natSumNatEquivNat.symm m with c | v
  · exfalso
    have hmc : m = extPos c := by rw [extPos, ← hm, Equiv.apply_symm_apply]
    rw [hmc, outerArg_ext] at hE
    obtain ⟨⟨b1, b2⟩, _, hq⟩ := Option.map_eq_some_iff.mp hE
    injection hq with h1 _
    subst h1
    exact not_mem_oddPos b1 hj
  · exact ⟨v, by rw [← hm, Equiv.apply_symm_apply]⟩

/-- The stored route is the ordinary plug followed by the second use.

    (σ : R) (x y : Route R)                                        [the parameter, two routes]
    rcomp σ x (dupArg y) = rcomp σ (plug σ x y) (outerArg y)       [regrouping the composite] -/
theorem stored_eq {R : Type u} [CommRing R] (σ : R)
    (x y : RouteAnyRing.Route R) :
    RouteAnyRing.rcomp σ x (dupArg y) =
      RouteAnyRing.rcomp σ (ApplicationWiring.plug σ x y) (outerArg y) := by
  rw [dupArg_eq σ y, ← RouteAnyRing.rcomp_assoc, ApplicationWiring.plug]

/-- Conjugating a composite route is the composite of its conjugates.

    conjRoute [a route read through a relabelling of the wire]
    rcomp [follow one route and then another, multiplying the weights]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    = [is equal to] -/
private theorem conjRoute_rcomp_generic {R : Type u} [CommRing R] (σ : R) (e : ℕ ≃ ℕ)
    (a b : RouteAnyRing.Route R) :
    conjRoute e (RouteAnyRing.rcomp σ a b) =
      RouteAnyRing.rcomp σ (conjRoute e a) (conjRoute e b) := by
  funext i
  cases ha : a (e i) with
  | none => simp [conjRoute, RouteAnyRing.rcomp, ha]
  | some p =>
    cases hb : b p.1 with
    | none => simp [conjRoute, RouteAnyRing.rcomp, ha, hb, Equiv.apply_symm_apply]
    | some q => simp [conjRoute, RouteAnyRing.rcomp, ha, hb, Equiv.apply_symm_apply]

/-- Conjugating by inverse wire equivalences returns the original route.

    conjRoute [a route read through a relabelling of the wire]
    RingCell [a point: two ring elements side by side, a + b·x]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    = [is equal to] -/
private theorem conjRoute_symm_generic {R : Type u} [CommRing R] (e : ℕ ≃ ℕ)
    (f : RouteAnyRing.Route R) : conjRoute e (conjRoute e.symm f) = f := by
  funext i
  cases hf : f i with
  | none => simp [conjRoute, hf]
  | some p => simp [conjRoute, hf]

/-- The contraction layout sends the first loop track to the argument's loop.

    wIdx [the change of layout a contraction needs, its labels to its argument's]
    outPos [a position of the loop that faces the argument]
    argPos [a position of the channel a route faces its argument through]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    = [is equal to] -/
private theorem wIdx_out_constructive (v : ℕ) : wIdx (outPos v) = argPos v := by
  simp only [wIdx, tripleSplit, pairJoin, outPos, argPos, Equiv.trans_apply,
    Equiv.sumCongr_apply, Equiv.coe_refl, Equiv.symm_apply_apply, Sum.map_inr,
    Equiv.sumComm_apply, Sum.swap_inl, Equiv.sumAssoc_symm_apply_inr_inr, id_eq]

/-- The contraction layout sends the second loop track to the nested argument loop.

    wIdx [the change of layout a contraction needs, its labels to its argument's]
    innPos [a position of the loop a stored route keeps to itself]
    extPos [a position of the wire outside both loops]
    argPos [a position of the channel a route faces its argument through]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    = [is equal to] -/
private theorem wIdx_inn_constructive (u : ℕ) : wIdx (innPos u) = extPos (argPos u) := by
  simp only [wIdx, tripleSplit, pairJoin, innPos, extPos, argPos, Equiv.trans_apply,
    Equiv.sumCongr_apply, Equiv.coe_refl, Equiv.symm_apply_apply, Sum.map_inr,
    Equiv.sumComm_apply, Sum.swap_inr, Equiv.sumAssoc_symm_apply_inr_inl, Sum.map_inl]

/-- The inverse contraction layout maps the doubled outside position to the outside track.

    wIdx [the change of layout a contraction needs, its labels to its argument's]
    extPos [a position of the wire outside both loops]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    = [is equal to] -/
private theorem wIdx_symm_ext_constructive (k : ℕ) :
    wIdx.symm (extPos (extPos k)) = extPos k := by
  rw [← wIdx_ext]
  exact wIdx.symm_apply_apply _

/-- The inverse contraction layout maps the argument loop to the first loop track.

    wIdx [the change of layout a contraction needs, its labels to its argument's]
    argPos [a position of the channel a route faces its argument through]
    outPos [a position of the loop that faces the argument]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    = [is equal to] -/
private theorem wIdx_symm_arg_constructive (v : ℕ) : wIdx.symm (argPos v) = outPos v := by
  rw [← wIdx_out_constructive]
  exact wIdx.symm_apply_apply _

/-- The inverse contraction layout maps the nested argument loop to the second loop track.

    wIdx [the change of layout a contraction needs, its labels to its argument's]
    extPos [a position of the wire outside both loops]
    argPos [a position of the channel a route faces its argument through]
    innPos [a position of the loop a stored route keeps to itself]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    = [is equal to] -/
private theorem wIdx_symm_inn_constructive (u : ℕ) :
    wIdx.symm (extPos (argPos u)) = innPos u := by
  rw [← wIdx_inn_constructive]
  exact wIdx.symm_apply_apply _

/-- The doubled argument is the tensor of its two copies at every ring carrier.

    dupArg [the route that supplies two copies of an argument]
    conjRoute [a route read through a relabelling of the wire]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rid [the route joining every position to itself, weighted by the unit]
    wIdx [the change of layout a contraction needs, its labels to its argument's]
    RingCell [a point: two ring elements side by side, a + b·x]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    = [is equal to] -/
private theorem dupArg_is_tensor_generic {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) :
    conjRoute wIdx (dupArg y) = RouteAnyRing.rtensor (RouteAnyRing.rid (R := R))
      (RouteAnyRing.rtensor y y) := by
  funext i
  rcases pos_trichotomy i with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨u, rfl⟩
  · rw [conjRoute, wIdx_ext, dupArg_ext, rtensor_ext, RouteAnyRing.rid]
    simp only [Option.map_some, wIdx_symm_ext_constructive]
  · rw [conjRoute, wIdx_out_constructive, dupArg_arg, outPos_eq, rtensor_arg, rtensor_ext]
    cases hy : y v with
    | none => simp
    | some p => simp only [Option.map_some, wIdx_symm_arg_constructive, outPos_eq]
  · rw [conjRoute, wIdx_inn_constructive, dupArg_inn, innPos_eq, rtensor_arg, rtensor_arg]
    cases hy : y u with
    | none => simp
    | some p => simp only [Option.map_some, wIdx_symm_inn_constructive, innPos_eq]

/-- Reading the two-copy tensor back through the contraction layout gives the doubled argument.

    dupArg [the route that supplies two copies of an argument]
    conjRoute [a route read through a relabelling of the wire]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rid [the route joining every position to itself, weighted by the unit]
    wIdx [the change of layout a contraction needs, its labels to its argument's]
    RingCell [a point: two ring elements side by side, a + b·x]
    = [is equal to] -/
private theorem conj_symm_tensor_generic {R : Type u} [CommRing R] (y : RouteAnyRing.Route R) :
    conjRoute wIdx.symm (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R))
      (RouteAnyRing.rtensor y y)) = dupArg y := by
  rw [← dupArg_is_tensor_generic y]
  have h := conjRoute_symm_generic wIdx.symm (dupArg y)
  rwa [Equiv.symm_symm] at h

/-- The contraction layout carries its loop exactly onto the two argument uses.

    wIdx [the change of layout a contraction needs, its labels to its argument's]
    oddPos [the odd half of a doubled wire]
    innArgSet [the positions of the second argument use]
    Set [a collection of positions]
    '' [the image of a set under a map]
    ∪ [either of two sets]
    = [is equal to] -/
private theorem wIdx_image_oddPos_constructive :
    wIdx '' oddPos = oddPos ∪ innArgSet := by
  ext p
  constructor
  · rintro ⟨q, hq, rfl⟩
    rw [oddPos_eq] at hq
    rcases hq with ⟨v, rfl⟩ | ⟨u, rfl⟩
    · exact Or.inl ⟨v, wIdx_out_constructive v⟩
    · exact Or.inr ⟨u, wIdx_inn_constructive u⟩
  · rintro (⟨v, rfl⟩ | ⟨u, rfl⟩)
    · exact ⟨outPos v, by rw [oddPos_eq]; exact Or.inl ⟨v, rfl⟩,
        wIdx_out_constructive v⟩
    · exact ⟨innPos u, by rw [oddPos_eq]; exact Or.inr ⟨u, rfl⟩,
        wIdx_inn_constructive u⟩

/-- Plugging the relabelled stored route with two copies is the relabelled doubled composite.

    plug [put an argument route into a stored route's loop]
    conjRoute [a route read through a relabelling of the wire]
    wIdx [the change of layout a contraction needs, its labels to its argument's]
    rcomp [follow one route and then another, multiplying the weights]
    dupArg [the route that supplies two copies of an argument]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    = [is equal to] -/
private theorem plug_dup_generic {R : Type u} [CommRing R] (σ : R)
    (x y : RouteAnyRing.Route R) :
    ApplicationWiring.plug σ (conjRoute wIdx x) (RouteAnyRing.rtensor y y) =
      conjRoute wIdx (RouteAnyRing.rcomp σ x (dupArg y)) := by
  have hconj := conjRoute_symm_generic wIdx
    (RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) (RouteAnyRing.rtensor y y))
  rw [ApplicationWiring.plug, ← hconj, ← conjRoute_rcomp_generic,
    conj_symm_tensor_generic]

/-- A relational collapsed step is one trace followed by the second argument copy.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    rcomp [follow one route and then another, multiplying the weights]
    dupArg [the route that supplies two copies of an argument]
    extPos [a position of the wire outside both loops]
    Traces [the union over every path length through a route]
    plug [put an argument route into a stored route's loop]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rid [the route that sends each wire to itself]
    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul [multiply two points together, the σ way]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions] -/
theorem relCollapse_step {R : Type u} [CommRing R] (σ : R)
    (x y : RouteAnyRing.Route R) (a c : ℕ) (w : Trichotomy.AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ
      (RelationalPaths.routeGraph (RouteAnyRing.rcomp σ x (dupArg y))) RelationalPaths.oddPos
      (extPos a) c w ↔
      ∃ b k u v, c = extPos b ∧
        RouteAnyRing.Traces σ (ApplicationWiring.plug σ x y) a k u ∧
        RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) y k = some (b, v) ∧
        w = Trichotomy.AnyRing.ringMul σ u v := by
  rw [RelationalPaths.relRuns_routeGraph_iff, stored_eq]
  change RouteAnyRing.Runs σ
    (RouteAnyRing.rcomp σ (ApplicationWiring.plug σ x y) (outerArg y)) oddPos
    (extPos a) c w ↔ _
  rw [runs_rcomp_id_on (outerArg_id_on_oddPos y) (outerArg_not_enter y)]
  constructor
  · rintro ⟨m, u, v, hrun, houter, rfl⟩
    obtain ⟨k, rfl⟩ := not_oddPos_ext (runs_end_not_mem hrun)
    rw [outerArg_ext] at houter
    obtain ⟨⟨b, v⟩, hsecond, heq⟩ := Option.map_eq_some_iff.mp houter
    injection heq with hc hv
    subst hc
    subst hv
    refine ⟨b, k, u, v, rfl, ?_, hsecond, rfl⟩
    rw [RelationalPaths.traces_iff_relRuns, RelationalPaths.relRuns_routeGraph_iff]
    change RouteAnyRing.Runs σ (ApplicationWiring.plug σ x y) oddPos (extPos a) (extPos k) u
    exact hrun
  · rintro ⟨b, k, u, v, rfl, htrace, hsecond, rfl⟩
    refine ⟨extPos k, u, v, ?_, ?_, rfl⟩
    · rw [RelationalPaths.traces_iff_relRuns, RelationalPaths.relRuns_routeGraph_iff] at htrace
      change RouteAnyRing.Runs σ (ApplicationWiring.plug σ x y) oddPos (extPos a) (extPos k) u
        at htrace
      exact htrace
    · rw [outerArg_ext, hsecond, Option.map_some]

/-- One application step is a traced first copy followed by the second argument copy.

    Traces [the union over every path length through a route]
    plug [put an argument route into a stored route's loop]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    rid [the route joining every position to itself, weighted by the unit]
    RingCell [a point: two ring elements side by side, a + b·x]
    ringMul [multiply two points together, the σ way]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to] -/
def appliedStep {R : Type u} [CommRing R] (σ : R) (x y : RouteAnyRing.Route R)
    (a b : ℕ) (w : Trichotomy.AnyRing.RingCell R) : Prop :=
  ∃ k u v, RouteAnyRing.Traces σ (ApplicationWiring.plug σ x y) a k u ∧
    RouteAnyRing.rtensor (RouteAnyRing.rid (R := R)) y k = some (b, v) ∧
    w = Trichotomy.AnyRing.ringMul σ u v

/-- The contraction wiring is exactly paths of its traced-and-applied step relation.

    Traces [the union over every path length through a route]
    RelRuns [an unbounded finite path]
    appliedStep [one application step: a trace followed by the second argument copy]
    plug [put an argument route into a stored route's loop]
    conjRoute [a route read through a relabelling of the wire]
    wIdx [the change of layout a contraction needs, its labels to its argument's]
    rtensor [two routes side by side, one on the even positions and one on the odd]
    extPos [a position of the wire outside both loops]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ↔ [exactly when, true in both directions]
    j [the final position] -/
theorem traces_appliedStep {R : Type u} [CommRing R] (σ : R)
    (x y : RouteAnyRing.Route R) (i j : ℕ) (w : Trichotomy.AnyRing.RingCell R) :
    RouteAnyRing.Traces σ
      (ApplicationWiring.plug σ (conjRoute wIdx x) (RouteAnyRing.rtensor y y)) i j w ↔
      RelationalPaths.RelRuns σ (appliedStep σ x y) RelationalPaths.oddPos
        (extPos i) (extPos j) w := by
  rw [plug_dup_generic, relTraces_conj]
  have hImage : wIdx '' RelationalPaths.oddPos = oddPos ∪ innArgSet := by
    change wIdx '' oddPos = oddPos ∪ innArgSet
    exact wIdx_image_oddPos_constructive
  have hInner : innArgSet = extPos '' oddPos := by
    ext p
    constructor
    · rintro ⟨u, rfl⟩
      exact ⟨argPos u, ⟨u, rfl⟩, rfl⟩
    · rintro ⟨q, ⟨u, rfl⟩, rfl⟩
      exact ⟨u, rfl⟩
  rw [hImage, wIdx_ext, wIdx_ext, Set.union_comm,
    RelationalPaths.relRuns_vanishing innArgSet_disjoint_oddPos, hInner]
  apply RelationalPaths.relRuns_embed_iff extPos_injective
  intro a c z
  change RelationalPaths.RelRuns σ
      (RelationalPaths.routeGraph (RouteAnyRing.rcomp σ x (dupArg y))) RelationalPaths.oddPos
      (extPos a) c z ↔
    ∃ b, c = extPos b ∧ appliedStep σ x y a b z
  constructor
  · intro h
    obtain ⟨b, k, u, v, hc, htrace, hsecond, hw⟩ :=
      (relCollapse_step σ x y a c z).mp h
    exact ⟨b, hc, k, u, v, htrace, hsecond, hw⟩
  · rintro ⟨b, hc, k, u, v, htrace, hsecond, hw⟩
    exact (relCollapse_step σ x y a c z).mpr ⟨b, k, u, v, hc, htrace, hsecond, hw⟩

/-- The second use is the first pushed one level out.

    innArgSet = extPos '' oddPos                   [the same positions, embedded] -/
theorem innArgSet_eq : innArgSet = extPos '' oddPos := by
  ext p
  constructor
  · rintro ⟨u, rfl⟩
    exact ⟨argPos u, ⟨u, rfl⟩, rfl⟩
  · rintro ⟨q, ⟨u, rfl⟩, rfl⟩
    exact ⟨u, rfl⟩

/-- **The collapse.** Collapsing the first use leaves the right side's own applied route.

@cite girard-1989
@cite joyal-street-verity-1996

    (σ : ℝ) (x y : Route) (a : ℕ)                  [parameter, two routes, a position]
    runsRoute σ (rcomp σ x (dupArg y)) oddPos (extPos a)       [collapse the first use]
    = (plug σ (rapp σ x y) y a).map fun q => (extPos q.1, q.2) [and the right side appears]

    So: this is the identification the equation turns on. The collapsed route is not
    `plug σ x y`'s own collapse: the two agree on the first use and on the far outside and differ
    at the second use, where the doubled argument applies `y` and the ordinary one does not, so
    the collapsed step ends one application of `y` further on. That is the same place the right
    side applies its outer copy, which is why they agree. -/
theorem collapse_step (σ : ℝ) (x y : Route) (a : ℕ) :
    runsRoute σ (rcomp σ x (dupArg y)) oddPos (extPos a)
      = (plug σ (rapp σ x y) y a).map fun q => (extPos q.1, q.2) := by
  have hchar : ∀ c w, Runs σ (rcomp σ x (dupArg y)) oddPos (extPos a) c w ↔
      ∃ b, c = extPos b ∧ plug σ (rapp σ x y) y a = some (b, w) := by
    intro c w
    rw [stored_eq, runs_rcomp_id_on (outerArg_id_on_oddPos y) (outerArg_not_enter y)]
    constructor
    · rintro ⟨m, w₁, w₂, hrun, hE, rfl⟩
      obtain ⟨c', rfl⟩ := not_oddPos_ext (runs_end_not_mem hrun)
      rw [outerArg_ext] at hE
      obtain ⟨⟨b1, b2⟩, hT, hq⟩ := Option.map_eq_some_iff.mp hE
      injection hq with h1 h2
      subst h1
      subst h2
      refine ⟨b1, rfl, ?_⟩
      refine rcomp_eq_some.mpr ⟨c', w₁, b2, ?_, hT, rfl⟩
      exact (tracesTo_traceRoute σ (plug σ x y) a c' w₁).mp
        ((traces_iff_runs σ (plug σ x y) a c' w₁).mpr hrun)
    · rintro ⟨b, rfl, hp⟩
      obtain ⟨c', w₁, w₂, hP, hT, rfl⟩ := rcomp_eq_some.mp hp
      refine ⟨extPos c', w₁, w₂, ?_, ?_, rfl⟩
      · exact (traces_iff_runs σ (plug σ x y) a c' w₁).mp
          ((tracesTo_traceRoute σ (plug σ x y) a c' w₁).mpr hP)
      · rw [outerArg_ext, hT, Option.map_some]
  cases hp : plug σ (rapp σ x y) y a with
  | none =>
    simp only [Option.map_none]
    cases hr : runsRoute σ (rcomp σ x (dupArg y)) oddPos (extPos a) with
    | none => rfl
    | some q =>
      obtain ⟨q1, q2⟩ := q
      obtain ⟨b, -, hpb⟩ :=
        (hchar q1 q2).mp ((runsRoute_eq σ _ oddPos (extPos a) q1 q2).mp hr)
      rw [hp] at hpb
      exact absurd hpb (by simp)
  | some p =>
    obtain ⟨p1, p2⟩ := p
    simp only [Option.map_some]
    exact (runsRoute_eq σ _ oddPos (extPos a) (extPos p1) p2).mpr
      ((hchar (extPos p1) p2).mpr ⟨p1, rfl, hp⟩)

/-- **The contraction equation.** A route that uses its argument twice, with two copies.

@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924
@cite girard-1989

    (σ : ℝ) (x y : Route)                                      [the parameter and two routes]
    rapp σ (conjRoute wIdx x) (rtensor y y)                    [the stored route on two copies]
    = rapp σ (rapp σ x y) y                                    [is x applied to y twice]

    So: `W x y = x y y`, with the two copies coming from the tensor and not the exponential.
    `contraction_left` turns the left into one path relation over the two loops the two uses run
    on, `runs_vanishing` collapses the first, `collapse_step` says what the collapse is, and
    `runs_embed` carries the result down to the second use, which is the first pushed one level
    out. The parameter is not used anywhere in the argument: every weight that arises is the
    argument's own or the unit. -/
theorem contraction (σ : ℝ) (x y : Route) :
    rapp σ (conjRoute wIdx x) (rtensor y y) = rapp σ (rapp σ x y) y := by
  refine rapp_eq ?_
  intro i j w
  rw [contraction_left, Set.union_comm,
    runs_vanishing innArgSet_disjoint_oddPos, innArgSet_eq,
    runs_embed extPos_injective (collapse_step σ x y), ← traces_iff_runs_ext]
  exact tracesTo_traceRoute σ (plug σ (rapp σ x y) y) i j w

end Combinator
end Apeiron
