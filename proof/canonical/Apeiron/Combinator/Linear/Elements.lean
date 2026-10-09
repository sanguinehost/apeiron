import Apeiron.Combinator.Linear.Comultiply
import Apeiron.Combinator.Linear.Exchange

/-!
# Combinators as constant routes

@name Elements
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924
@grade proved

The second caveat. Five of the nine equations proved so far are stated with a construction on the
stored routes, or a relabelling of one, rather than with a fixed route applied to its arguments.
A linear combinatory algebra wants the combinators to be **elements**: one route `R`, and the
equation read off `rapp σ (rapp σ R x) y`. This file supplies the machinery for that and uses it
on three of the five.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | `store_leave`, `entry_leave_store` |

Every route built here is a permutation of the counting numbers weighted by the unit, so the same
routing serves every setting; `theory/NOVELTY.md`'s A1 spectator test applies as throughout.

## The one observation

A route whose loop is empty and which is a bijection on each half does one thing when applied: it
relabels. `storeRoute e` sends an external position `i` into the loop at `e i` and sends a loop
position `v` out at `e.symm v`, and `rapp_storeRoute` says applying it to `x` gives
`conjRoute e x`, which is the whole of its effect. So every equation whose left side is a
relabelling of a stored route is already an equation about a constant route, once the
relabelling is named.

`Combinator/Interaction.lean`'s `iRoute` and `Combinator/Exponential.lean`'s `diggingRoute` are
both of this shape, which `iRoute_eq_store` and `diggingRoute_eq_store` record.

## The second observation

Relabelling the argument is the same as relabelling the stored route's argument channel.
`argExt e` acts as `e` on the odd half and does nothing on the even half, and `rapp_conj_arg`
says `rapp σ a (conjRoute e b) = rapp σ (conjRoute (argExt e).symm a) b`. With `conjRoute_trans`
that turns any equation of the shape "relabel the stored route, relabel the argument" into one
constant route.

## What this closes and what it does not

`cRoute`, `wRoute` and `fRoute` below are constant routes for the exchange, the contraction and
functoriality. The composition and the substitution are **not** here. The reason is sharp: for a
pure wiring, `rapp_storeRoute` says applying it can only relabel, and the exchange survives that
because `cComb_eq` shows its store is one application of a relabelled route. The composing
combinator's store is not. It holds two routes side by side with a wire between them, and its
composite must still offer the second route's argument channel outwards, which a relabelling of
a single stored route cannot produce. So `bRoute` and `sRoute` need a routing with a loop that
re-enters its argument, which is a different construction and not this one.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- APPLYING A PURE WIRING IS RELABELLING
-- ═══════════════════════════════════════════════════════════════════

/-- The route that stores its argument under a relabelling.

@cite girard-1989

    (e : ℕ ≃ ℕ)                        [a relabelling of the wire]
    Route                              [a weighted wire map]

    So: an external position goes into the loop relabelled, and a loop position comes out
    relabelled back. The loop is empty, because every loop position is sent straight out. -/
def storeRoute (e : ℕ ≃ ℕ) : Route := fun j =>
  Sum.elim
    (fun k => some (Equiv.natSumNatEquivNat (Sum.inr (e k)), one))
    (fun v => some (Equiv.natSumNatEquivNat (Sum.inl (e.symm v)), one))
    (Equiv.natSumNatEquivNat.symm j)

/-- The storing routing sends an external position into the loop.

    (e : ℕ ≃ ℕ) (k : ℕ)                                                [a relabelling, a position]
    storeRoute e (Equiv.natSumNatEquivNat (Sum.inl k))                 [the routing there]
    = some (Equiv.natSumNatEquivNat (Sum.inr (e k)), one)              [enters relabelled] -/
theorem storeRoute_inl (e : ℕ ≃ ℕ) (k : ℕ) :
    storeRoute e (Equiv.natSumNatEquivNat (Sum.inl k))
      = some (Equiv.natSumNatEquivNat (Sum.inr (e k)), one) := by
  simp [storeRoute]

/-- The storing routing sends a loop position out.

    (e : ℕ ≃ ℕ) (v : ℕ)                                                [a relabelling, a position]
    storeRoute e (Equiv.natSumNatEquivNat (Sum.inr v))                 [the routing there]
    = some (Equiv.natSumNatEquivNat (Sum.inl (e.symm v)), one)         [leaves relabelled back] -/
theorem storeRoute_inr (e : ℕ ≃ ℕ) (v : ℕ) :
    storeRoute e (Equiv.natSumNatEquivNat (Sum.inr v))
      = some (Equiv.natSumNatEquivNat (Sum.inl (e.symm v)), one) := by
  simp [storeRoute]

/-- Nothing crosses straight over.

    (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (i : ℕ)            [parameter, relabelling, route, position]
    direct σ (plug σ (storeRoute e) x) i = none        [no straight crossing] -/
theorem store_direct (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (i : ℕ) :
    direct σ (plug σ (storeRoute e) x) i = none := by
  rw [direct_apply]
  simp only [plug, rcomp, storeRoute_inl, Option.bind_some, rtensor_inr]
  cases hx : x (e i) with
  | none => simp
  | some p => simp only [Option.map_some, Option.bind_some, outL_inr, Option.map_none]

/-- Entering the loop reads the argument at the relabelled position.

    (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (i : ℕ)            [parameter, relabelling, route, position]
    entry σ (plug σ (storeRoute e) x) i = x (e i)      [the way in is the argument, relabelled] -/
theorem store_entry (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (i : ℕ) :
    entry σ (plug σ (storeRoute e) x) i = x (e i) := by
  rw [entry_apply]
  simp only [plug, rcomp, storeRoute_inl, Option.bind_some, rtensor_inr]
  cases hx : x (e i) with
  | none => simp
  | some p =>
    simp only [Option.map_some, Option.bind_some, outR_inr, one_mul, mul_one_right,
      Prod.mk.eta]

/-- The loop is never taken twice.

    (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (u : ℕ)            [parameter, relabelling, route, position]
    body σ (plug σ (storeRoute e) x) u = none          [one trip and no more] -/
theorem store_body (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (u : ℕ) :
    body σ (plug σ (storeRoute e) x) u = none := by
  rw [body_apply]
  simp only [plug, rcomp, storeRoute_inr, Option.bind_some, rtensor_inl, rid_apply,
    Option.map_some, outR_inl, Option.map_none]

/-- Leaving the loop relabels back.

    (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (u : ℕ)            [parameter, relabelling, route, position]
    leave σ (plug σ (storeRoute e) x) u                [the way out]
    = some (e.symm u, one)                             [relabels the other way] -/
theorem store_leave (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (u : ℕ) :
    leave σ (plug σ (storeRoute e) x) u = some (e.symm u, one) := by
  rw [leave_apply]
  simp only [plug, rcomp, storeRoute_inr, Option.bind_some, rtensor_inl, rid_apply,
    Option.map_some, outL_inl, one_mul]

/-- One pass through the loop is the argument, relabelled.

    (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (i : ℕ)            [parameter, relabelling, route, position]
    rcomp σ (entry σ (plug σ (storeRoute e) x))        [in relabelled]
      (leave σ (plug σ (storeRoute e) x)) i            [and out again]
    = conjRoute e x i                                  [is the argument, relabelled] -/
theorem entry_leave_store (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (i : ℕ) :
    rcomp σ (entry σ (plug σ (storeRoute e) x)) (leave σ (plug σ (storeRoute e) x)) i
      = conjRoute e x i := by
  rw [rcomp, store_entry, conjRoute]
  cases hx : x (e i) with
  | none => simp
  | some p =>
    simp only [Option.bind_some, store_leave, Option.map_some, mul_one_right]

/-- Applying the storing routing is a path of length one.

    (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route)                    [parameter, relabelling, route]
    (i j : ℕ) (w : Cell)                               [two positions and a weight]
    Traces σ (plug σ (storeRoute e) x) i j w           [a path of the applied routing]
    ↔                                                  [exactly when]
    conjRoute e x i = some (j, w)                      [the relabelled route takes i there] -/
theorem storeRoute_app (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ (storeRoute e) x) i j w ↔ conjRoute e x i = some (j, w) := by
  have hbody : body σ (plug σ (storeRoute e) x) = rnone := funext (store_body σ e x)
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · rw [store_direct] at hd; exact absurd hd (by simp)
    · cases n with
      | zero =>
        rw [rpow, rcomp_rid_left] at hn
        rw [entry_leave_store] at hn
        exact hn
      | succ n =>
        rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
        simp [rnone] at hn
  · intro h
    refine Or.inr ⟨0, ?_⟩
    rw [rpow, rcomp_rid_left, entry_leave_store]
    exact h

/-- **Applying a pure wiring relabels and does nothing else.**

@cite girard-1989

    (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route)                [the parameter, a relabelling, a route]
    rapp σ (storeRoute e) x = conjRoute e x        [the stored route comes back relabelled]

    So: every combinator equation whose left side is a relabelling of a stored route is an
    equation about a constant route, and this is the dictionary between them. `iRoute` is the
    case of the identity relabelling, `diggingRoute` the pairing's associativity. -/
theorem rapp_storeRoute (σ : ℝ) (e : ℕ ≃ ℕ) (x : Route) :
    rapp σ (storeRoute e) x = conjRoute e x :=
  rapp_eq (storeRoute_app σ e x)

/-- The identity routing stores under the identity relabelling.

    iRoute = storeRoute (Equiv.refl ℕ)             [the swap is the trivial store] -/
theorem iRoute_eq_store : iRoute = storeRoute (Equiv.refl ℕ) := rfl

/-- The digging routing stores under the pairing's associativity.

    diggingRoute = storeRoute dig                  [digging is the associativity, stored] -/
theorem diggingRoute_eq_store : diggingRoute = storeRoute dig := rfl

-- ═══════════════════════════════════════════════════════════════════
-- RELABELLING THE ARGUMENT IS RELABELLING THE ARGUMENT CHANNEL
-- ═══════════════════════════════════════════════════════════════════

/-- A relabelling of the loop half, doing nothing on the outside.

    (e : ℕ ≃ ℕ) (n : ℕ)                            [a relabelling and a position]
    ℕ                                              [relabelled if odd, untouched if even] -/
def argExtFun (e : ℕ ≃ ℕ) (n : ℕ) : ℕ :=
  Sum.elim (fun k => extPos k) (fun u => argPos (e u)) (Equiv.natSumNatEquivNat.symm n)

/-- The extended relabelling does nothing on the outside.

    (e : ℕ ≃ ℕ) (k : ℕ)                            [a relabelling and a position]
    argExtFun e (extPos k) = extPos k              [the even half is untouched] -/
theorem argExtFun_ext (e : ℕ ≃ ℕ) (k : ℕ) : argExtFun e (extPos k) = extPos k := by
  simp only [argExtFun, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The extended relabelling is the relabelling on the loop.

    (e : ℕ ≃ ℕ) (u : ℕ)                            [a relabelling and a position]
    argExtFun e (argPos u) = argPos (e u)          [the odd half is relabelled] -/
theorem argExtFun_arg (e : ℕ ≃ ℕ) (u : ℕ) : argExtFun e (argPos u) = argPos (e u) := by
  simp only [argExtFun, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- A relabelling of the wire which acts on the argument channel alone.

    (e : ℕ ≃ ℕ)                                    [a relabelling of the argument channel]
    ℕ ≃ ℕ                                          [a relabelling of the whole wire] -/
def argExt (e : ℕ ≃ ℕ) : ℕ ≃ ℕ where
  toFun := argExtFun e
  invFun := argExtFun e.symm
  left_inv := by
    intro n
    rcases pos_split2 n with ⟨k, rfl⟩ | ⟨u, rfl⟩
    · rw [argExtFun_ext, argExtFun_ext]
    · rw [argExtFun_arg, argExtFun_arg, Equiv.symm_apply_apply]
  right_inv := by
    intro n
    rcases pos_split2 n with ⟨k, rfl⟩ | ⟨u, rfl⟩
    · rw [argExtFun_ext, argExtFun_ext]
    · rw [argExtFun_arg, argExtFun_arg, Equiv.apply_symm_apply]

/-- The extended relabelling on the outside.

    (e : ℕ ≃ ℕ) (k : ℕ)                            [a relabelling and a position]
    argExt e (extPos k) = extPos k                 [the even half is untouched] -/
theorem argExt_ext (e : ℕ ≃ ℕ) (k : ℕ) : argExt e (extPos k) = extPos k := argExtFun_ext e k

/-- The extended relabelling on the loop.

    (e : ℕ ≃ ℕ) (u : ℕ)                            [a relabelling and a position]
    argExt e (argPos u) = argPos (e u)             [the odd half is relabelled] -/
theorem argExt_arg (e : ℕ ≃ ℕ) (u : ℕ) : argExt e (argPos u) = argPos (e u) := argExtFun_arg e u

/-- The inverse extended relabelling on the outside.

    (e : ℕ ≃ ℕ) (k : ℕ)                            [a relabelling and a position]
    (argExt e).symm (extPos k) = extPos k          [the even half is untouched] -/
theorem argExt_symm_ext (e : ℕ ≃ ℕ) (k : ℕ) : (argExt e).symm (extPos k) = extPos k :=
  argExtFun_ext e.symm k

/-- The inverse extended relabelling on the loop.

    (e : ℕ ≃ ℕ) (u : ℕ)                            [a relabelling and a position]
    (argExt e).symm (argPos u) = argPos (e.symm u) [the odd half is relabelled back] -/
theorem argExt_symm_arg (e : ℕ ≃ ℕ) (u : ℕ) :
    (argExt e).symm (argPos u) = argPos (e.symm u) := argExtFun_arg e.symm u

/-- The extended relabelling keeps the loop.

    (e : ℕ ≃ ℕ)                                    [a relabelling]
    argExt e '' oddPos = oddPos                    [it permutes the loop positions] -/
theorem argExt_image_oddPos (e : ℕ ≃ ℕ) : argExt e '' oddPos = oddPos := by
  ext p
  constructor
  · rintro ⟨q, ⟨u, rfl⟩, rfl⟩
    exact ⟨e u, argExt_arg e u⟩
  · rintro ⟨u, rfl⟩
    refine ⟨argPos (e.symm u), ⟨e.symm u, rfl⟩, ?_⟩
    show argExt e (argPos (e.symm u)) = argPos u
    rw [argExt_arg, Equiv.apply_symm_apply]

/-- Relabelling twice is relabelling once.

    (f g : ℕ ≃ ℕ) (x : Route)                      [two relabellings and a route]
    conjRoute f (conjRoute g x) = conjRoute (f.trans g) x   [the relabellings compose] -/
theorem conjRoute_trans (f g : ℕ ≃ ℕ) (x : Route) :
    conjRoute f (conjRoute g x) = conjRoute (f.trans g) x := by
  funext i
  simp only [conjRoute, Equiv.trans_apply, Equiv.symm_trans_apply, Option.map_map]
  cases hx : x (g (f i)) with
  | none => rfl
  | some q => rfl

/-- A relabelled argument is the argument channel relabelled.

    (e : ℕ ≃ ℕ) (b : Route)                        [a relabelling and a route]
    rtensor rid (conjRoute e b)                    [the argument layer of a relabelled route]
    = conjRoute (argExt e) (rtensor rid b)         [is the relabelled argument layer] -/
theorem rtensor_conj (e : ℕ ≃ ℕ) (b : Route) :
    rtensor rid (conjRoute e b) = conjRoute (argExt e) (rtensor rid b) := by
  funext i
  rcases pos_split2 i with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · rw [rtensor_ext_id, conjRoute, argExt_ext, rtensor_ext_id, Option.map_some,
      argExt_symm_ext]
  · rw [rtensor_arg, conjRoute, conjRoute, argExt_arg, rtensor_arg]
    cases hb : b (e u) with
    | none => rfl
    | some q =>
      rw [Option.map_some, Option.map_some, Option.map_some, Option.map_some, argExt_symm_arg]

/-- The applied machine with a relabelled argument.

    (σ : ℝ) (a b : Route) (e : ℕ ≃ ℕ)              [the parameter, two routes, a relabelling]
    plug σ a (conjRoute e b)                                            [the machine]
    = conjRoute (argExt e) (plug σ (conjRoute (argExt e).symm a) b)     [is a relabelled machine] -/
theorem plug_conj_arg (σ : ℝ) (a b : Route) (e : ℕ ≃ ℕ) :
    plug σ a (conjRoute e b)
      = conjRoute (argExt e) (plug σ (conjRoute (argExt e).symm a) b) := by
  rw [plug, plug, conjRoute_rcomp, conjRoute_symm, rtensor_conj]

/-- **Relabelling the argument is relabelling the stored route's argument channel.**

@cite girard-1989

    (σ : ℝ) (a b : Route) (e : ℕ ≃ ℕ)              [the parameter, two routes, a relabelling]
    rapp σ a (conjRoute e b)                       [apply to a relabelled argument]
    = rapp σ (conjRoute (argExt e).symm a) b       [is applying a relabelled route to it]

    So: whatever a combinator does to its argument by a relabelling, it can do to itself
    instead. With `conjRoute_trans` and `rapp_storeRoute` that turns an equation of the shape
    "relabel the stored route, relabel the argument" into one constant route applied twice. -/
theorem rapp_conj_arg (σ : ℝ) (a b : Route) (e : ℕ ≃ ℕ) :
    rapp σ a (conjRoute e b) = rapp σ (conjRoute (argExt e).symm a) b := by
  refine route_ext fun i j w => ?_
  rw [rapp, ← tracesTo_traceRoute σ (plug σ a (conjRoute e b)) i j w, rapp,
    ← tracesTo_traceRoute σ (plug σ (conjRoute (argExt e).symm a) b) i j w,
    plug_conj_arg, traces_conj, argExt_image_oddPos, argExt_ext, argExt_ext,
    ← traces_iff_runs_ext]

-- ═══════════════════════════════════════════════════════════════════
-- THREE OF THE FIVE, AS CONSTANT ROUTES
-- ═══════════════════════════════════════════════════════════════════

/-- The exchange relabelling is its own inverse.

    cSwap.symm = cSwap                             [an involution is its own inverse] -/
theorem cSwap_symm : cSwap.symm = cSwap := rfl

/-- Relabelling by the exchange twice changes nothing.

    (f : Route)                                    [a route]
    conjRoute cSwap (conjRoute cSwap f) = f        [the round trip is the identity] -/
theorem conjRoute_cSwap_twice (f : Route) : conjRoute cSwap (conjRoute cSwap f) = f :=
  conjRoute_symm cSwap f

/-- **The exchanging combinator's store is an application.**

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    cComb σ x y = rapp σ (conjRoute cSwap x) y     [storing y is applying to y]

    So: the exchange plugs its second route into the first route's second argument channel,
    which is a composition and not a tensor, and `bArg_conj` says that channel is the ordinary
    argument channel relabelled. So the whole construction is one application of a relabelled
    stored route, and a constant route follows. -/
theorem cComb_eq (σ : ℝ) (x y : Route) : cComb σ x y = rapp σ (conjRoute cSwap x) y := by
  rw [cComb, cStore, conjRoute_rcomp, bArg_conj, conjRoute_cSwap_twice, rapp, plug]

/-- The exchanging combinator as a constant route.

@cite schonfinkel-1924
@cite curry-1930

    Route                              [a weighted wire map] -/
def cRoute : Route := storeRoute cSwap

/-- **C as an element.** The exchanging combinator is one fixed route.

@cite schonfinkel-1924

    (σ : ℝ) (x y z : Route)                        [the parameter and three routes]
    rapp σ (rapp σ (rapp σ cRoute x) y) z          [one constant route, applied three times]
    = rapp σ (rapp σ x z) y                        [exchanges the two arguments] -/
theorem cRoute_app (σ : ℝ) (x y z : Route) :
    rapp σ (rapp σ (rapp σ cRoute x) y) z = rapp σ (rapp σ x z) y := by
  rw [cRoute, rapp_storeRoute, ← cComb_eq]
  exact commutator σ x y z

/-- Functoriality as a constant route.

@cite abramsky-haghverdi-scott-2002

    Route                              [a weighted wire map] -/
def fRoute : Route := storeRoute copySplit

/-- **F as an element.** Functoriality is one fixed route.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    rapp σ (rapp σ fRoute (bang x)) (bang y)       [one constant route, applied to two copies]
    = bang (rapp σ x y)                            [is a copy of the application] -/
theorem fRoute_app (σ : ℝ) (x y : Route) :
    rapp σ (rapp σ fRoute (bang x)) (bang y) = bang (rapp σ x y) := by
  rw [fRoute, rapp_storeRoute]
  exact functoriality σ x y

/-- The contracting combinator as a constant route.

@cite curry-1930
@cite abramsky-haghverdi-scott-2002

    Route                              [a weighted wire map] -/
def wRoute : Route := storeRoute ((argExt contractSplit).symm.trans wIdx)

/-- **W as an element.** The contracting combinator is one fixed route.

@cite curry-1930
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    rapp σ (rapp σ wRoute x) (bang y)              [one constant route, applied twice]
    = rapp σ (rapp σ x (bang y)) (bang y)          [uses the reusable argument twice]

    So: the combinator carries both relabellings at once, the one the contraction puts on the
    stored route and the one the comultiplication puts on the argument, because
    `rapp_conj_arg` moves the second onto the first. -/
theorem wRoute_app (σ : ℝ) (x y : Route) :
    rapp σ (rapp σ wRoute x) (bang y) = rapp σ (rapp σ x (bang y)) (bang y) := by
  rw [wRoute, rapp_storeRoute, ← conjRoute_trans, ← rapp_conj_arg]
  exact contraction_bang σ x y

end Combinator
end Apeiron

end
