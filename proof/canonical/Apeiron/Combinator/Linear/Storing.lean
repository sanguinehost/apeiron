import Apeiron.Combinator.Linear.Elements

/-!
# The composing combinator's routing

@name Storing
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite curry-1930
@grade proved

`Combinator/Elements.lean` closed the exchange, the contraction and functoriality as constant
routes, and recorded why the same observation cannot reach the composition: applying consumes the
whole of the argument's wire, while `B x y z = x (y z)` has to keep offering the second route's
argument channel outwards. So the composition needs a routing whose loop re-enters its argument
rather than passing through it once. This file builds that routing and computes what applying it
gives.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | `b_direct_z`, `b_leave`, `bRoute_app` |

The routing is an involution of the counting numbers weighted by the unit, so the same routing
serves every setting and `theory/NOVELTY.md`'s A1 spectator test applies as throughout.

## The six families

A position of the wire falls into one of six families, and `bFun` pairs them off:

```
   extPos (extPos (extPos k))   <->  argPos (extPos k)
   argPos (argPos u)            <->  extPos (argPos (extPos u))
   extPos (argPos (argPos v))   <->  extPos (extPos (argPos v))
```

the outside against the first route's outside, the first route asking against the second route
answering, and the second route asking against the outer argument. Read through `xPos` the first
two collapse into one statement, `bFun (extPos (xPos p)) = argPos p`, which says the even half
carries the first route's whole wire one level out and the odd half carries it plain.

No odd position is sent to an odd one, so the loop is empty and `Traces` reduces to a straight
crossing joined with one pass. `bStored` is what that comes to, and `rapp_bRoute` says applying
the routing to a route gives it.

## Where this leaves the equation

`bStored x` stands on the four tracks `Combinator/Composition.lean` uses, with the same labels:
the outside and the outer argument channel are its even half, the asking and answering tracks its
odd half. What remains for `B` as an element is that `rapp σ (bStored x) y` and `bComb σ x y`
agree.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- THE ROUTING
-- ═══════════════════════════════════════════════════════════════════

/-- The composing combinator's wiring, as a map of the counting numbers.

@cite curry-1930

    (i : ℕ)                            [a position of the wire]
    ℕ                                  [the position it is wired to] -/
def bFun (i : ℕ) : ℕ :=
  Sum.elim
    (fun c => Sum.elim
      (fun d => Sum.elim
        (fun k => argPos (extPos k))
        (fun v => extPos (argPos (argPos v)))
        (Equiv.natSumNatEquivNat.symm d))
      (fun d => Sum.elim
        (fun u => argPos (argPos u))
        (fun v => extPos (extPos (argPos v)))
        (Equiv.natSumNatEquivNat.symm d))
      (Equiv.natSumNatEquivNat.symm c))
    (fun c => extPos (xPos c))
    (Equiv.natSumNatEquivNat.symm i)

/-- The outside is wired to the first route's outside.

    (k : ℕ)                                                            [a position]
    bFun (extPos (extPos (extPos k))) = argPos (extPos k)              [the outside enters] -/
theorem bFun_ee (k : ℕ) : bFun (extPos (extPos (extPos k))) = argPos (extPos k) := by
  simp only [bFun, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The outer argument is wired to the second route asking.

    (v : ℕ)                                                            [a position]
    bFun (extPos (extPos (argPos v))) = extPos (argPos (argPos v))     [the argument answers] -/
theorem bFun_ez (v : ℕ) :
    bFun (extPos (extPos (argPos v))) = extPos (argPos (argPos v)) := by
  simp only [bFun, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- The asking track is wired to the first route's argument.

    (u : ℕ)                                                            [a position]
    bFun (extPos (argPos (extPos u))) = argPos (argPos u)              [the question enters] -/
theorem bFun_eo (u : ℕ) : bFun (extPos (argPos (extPos u))) = argPos (argPos u) := by
  simp only [bFun, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- The answering track is wired to the outer argument.

    (v : ℕ)                                                            [a position]
    bFun (extPos (argPos (argPos v))) = extPos (extPos (argPos v))     [the question goes out] -/
theorem bFun_ei (v : ℕ) :
    bFun (extPos (argPos (argPos v))) = extPos (extPos (argPos v)) := by
  simp only [bFun, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- The first route's wire comes back one level out.

    (c : ℕ)                                        [one of the first route's positions]
    bFun (argPos c) = extPos (xPos c)              [it lands on that route's tracks] -/
theorem bFun_odd (c : ℕ) : bFun (argPos c) = extPos (xPos c) := by
  simp only [bFun, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- The first route's wire goes in plain.

    (p : ℕ)                                        [one of the first route's positions]
    bFun (extPos (xPos p)) = argPos p              [its track enters the loop at that position] -/
theorem bFun_xPos (p : ℕ) : bFun (extPos (xPos p)) = argPos p := by
  rcases pos_split2 p with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · rw [xPos_ext, bFun_ee]
  · rw [xPos_arg, outPos_eq, bFun_eo]

/-- The outer argument track and the answering track are wired to each other.

    (v : ℕ)                                        [a position]
    bFun (extPos (zPos v)) = extPos (innPos v)     [the argument faces the second route] -/
theorem bFun_z (v : ℕ) : bFun (extPos (zPos v)) = extPos (innPos v) := by
  rw [zPos_eq, bFun_ez, innPos_eq]

/-- The answering track and the outer argument track are wired to each other.

    (v : ℕ)                                        [a position]
    bFun (extPos (innPos v)) = extPos (zPos v)     [the second route faces the argument] -/
theorem bFun_inn (v : ℕ) : bFun (extPos (innPos v)) = extPos (zPos v) := by
  rw [innPos_eq, bFun_ei, zPos_eq]

/-- The wiring is its own inverse.

    Function.Involutive bFun                       [running it twice changes nothing] -/
theorem bFun_involutive : Function.Involutive bFun := by
  intro i
  rcases pos_split2 i with ⟨c, rfl⟩ | ⟨c, rfl⟩
  · rcases pos_split c with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨d, rfl⟩
    · rw [bFun_ee, bFun_odd, xPos_ext]
    · rw [← zPos_eq, bFun_z, bFun_inn]
    · rcases pos_split2 d with ⟨u, rfl⟩ | ⟨v, rfl⟩
      · rw [← outPos_eq, ← xPos_arg, bFun_xPos, bFun_odd]
      · rw [← innPos_eq, bFun_inn, bFun_z]
  · rw [bFun_odd, bFun_xPos]

/-- The composing combinator as a constant route.

@cite curry-1930
@cite abramsky-haghverdi-scott-2002

    Route                              [a weighted wire map] -/
def bRoute : Route := fun i => some (bFun i, one)

-- ═══════════════════════════════════════════════════════════════════
-- WHAT APPLYING IT GIVES
-- ═══════════════════════════════════════════════════════════════════

/-- The first route stored on its tracks, with the argument channel folded back.

    (x : Route)                                    [a weighted wire map]
    Route                                          [that route, stored] -/
def bStored (x : Route) : Route := fun i =>
  Sum.elim
    (fun c => Sum.elim
      (fun k => (x (extPos k)).map fun q => (xPos q.1, q.2))
      (fun v => some (innPos v, one))
      (Equiv.natSumNatEquivNat.symm c))
    (fun c => Sum.elim
      (fun u => (x (argPos u)).map fun q => (xPos q.1, q.2))
      (fun v => some (zPos v, one))
      (Equiv.natSumNatEquivNat.symm c))
    (Equiv.natSumNatEquivNat.symm i)

/-- The stored route on the outside.

    (x : Route) (k : ℕ)                            [a route and a position]
    bStored x (extPos (extPos k))                                    [a step on the outside]
    = (x (extPos k)).map fun q => (xPos q.1, q.2)                    [is the route, laid out] -/
theorem bStored_ee (x : Route) (k : ℕ) :
    bStored x (extPos (extPos k)) = (x (extPos k)).map fun q => (xPos q.1, q.2) := by
  simp only [bStored, extPos, Equiv.symm_apply_apply, Sum.elim_inl]

/-- The stored route on the asking track.

    (x : Route) (u : ℕ)                            [a route and a position]
    bStored x (argPos (extPos u))                                    [a step on the asking track]
    = (x (argPos u)).map fun q => (xPos q.1, q.2)                    [is the route, laid out] -/
theorem bStored_ae (x : Route) (u : ℕ) :
    bStored x (argPos (extPos u)) = (x (argPos u)).map fun q => (xPos q.1, q.2) := by
  simp only [bStored, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- The stored route on the first route's tracks.

    (x : Route) (p : ℕ)                            [a route and a position]
    bStored x (xPos p) = (x p).map fun q => (xPos q.1, q.2)          [the route, laid out] -/
theorem bStored_x (x : Route) (p : ℕ) :
    bStored x (xPos p) = (x p).map fun q => (xPos q.1, q.2) := by
  rcases pos_split2 p with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · rw [xPos_ext, bStored_ee]
  · rw [xPos_arg, outPos_eq, bStored_ae]

/-- The stored route sends the outer argument to the answering track.

    (x : Route) (v : ℕ)                            [a route and a position]
    bStored x (zPos v) = some (innPos v, one)      [the argument faces the second route] -/
theorem bStored_z (x : Route) (v : ℕ) : bStored x (zPos v) = some (innPos v, one) := by
  simp only [bStored, zPos, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inl, Sum.elim_inr]

/-- The stored route sends the answering track to the outer argument.

    (x : Route) (v : ℕ)                            [a route and a position]
    bStored x (innPos v) = some (zPos v, one)      [the second route faces the argument] -/
theorem bStored_inn (x : Route) (v : ℕ) : bStored x (innPos v) = some (zPos v, one) := by
  simp only [bStored, innPos, extPos, argPos, Equiv.symm_apply_apply, Sum.elim_inr]

/-- One step of the applied routing, written out.

    (σ : ℝ) (x : Route) (i : ℕ)                        [the parameter, a route, a position]
    plug σ bRoute x i                                  [a step of the applied routing]
    = ((rtensor rid x) (bFun i)).map fun q => (q.1, mul σ one q.2)  [the wire, then the route] -/
theorem plug_bRoute (σ : ℝ) (x : Route) (i : ℕ) :
    plug σ bRoute x i = ((rtensor rid x) (bFun i)).map fun q => (q.1, mul σ one q.2) := by
  simp only [plug, rcomp, bRoute, Option.bind_some]

/-- The straight crossing, at an even position named as such.

    (σ : ℝ) (f : Route) (i : ℕ)                    [the parameter, a route, a position]
    direct σ f i = (f (extPos i)).bind fun p =>    [one step from the even copy of i]
      (outL p.1).map fun q => (q.1, mul σ p.2 q.2) [kept if it lands even, weights multiplied]
    then .map fun q => (q.1, mul σ one q.2)        [and weighted by one on the way in] -/
theorem direct_ext (σ : ℝ) (f : Route) (i : ℕ) :
    direct σ f i = ((f (extPos i)).bind fun p =>
      (outL p.1).map fun q => (q.1, mul σ p.2 q.2)).map fun q => (q.1, mul σ one q.2) :=
  direct_apply σ f i

/-- The way into the loop, at an even position named as such.

    (σ : ℝ) (f : Route) (i : ℕ)                    [the parameter, a route, a position]
    entry σ f i = (f (extPos i)).bind fun p =>     [one step from the even copy of i]
      (outR p.1).map fun q => (q.1, mul σ p.2 q.2) [kept if it lands odd, weights multiplied]
    then .map fun q => (q.1, mul σ one q.2)        [and weighted by one on the way in] -/
theorem entry_ext (σ : ℝ) (f : Route) (i : ℕ) :
    entry σ f i = ((f (extPos i)).bind fun p =>
      (outR p.1).map fun q => (q.1, mul σ p.2 q.2)).map fun q => (q.1, mul σ one q.2) :=
  entry_apply σ f i

/-- The way out of the loop, at an odd position named as such.

    (σ : ℝ) (f : Route) (u : ℕ)                    [the parameter, a route, a position]
    leave σ f u = (f (argPos u)).bind fun p =>     [one step from the odd copy of u]
      (outL p.1).map fun q => (q.1, mul σ p.2 q.2) [kept if it lands even, weights multiplied]
    then .map fun q => (q.1, mul σ one q.2)        [and weighted by one on the way in] -/
theorem leave_arg (σ : ℝ) (f : Route) (u : ℕ) :
    leave σ f u = ((f (argPos u)).bind fun p =>
      (outL p.1).map fun q => (q.1, mul σ p.2 q.2)).map fun q => (q.1, mul σ one q.2) :=
  leave_apply σ f u

/-- One trip round the loop, at an odd position named as such.

    (σ : ℝ) (f : Route) (u : ℕ)                    [the parameter, a route, a position]
    body σ f u = (f (argPos u)).bind fun p =>      [one step from the odd copy of u]
      (outR p.1).map fun q => (q.1, mul σ p.2 q.2) [kept if it lands odd, weights multiplied]
    then .map fun q => (q.1, mul σ one q.2)        [and weighted by one on the way in] -/
theorem body_arg (σ : ℝ) (f : Route) (u : ℕ) :
    body σ f u = ((f (argPos u)).bind fun p =>
      (outR p.1).map fun q => (q.1, mul σ p.2 q.2)).map fun q => (q.1, mul σ one q.2) :=
  body_apply σ f u

/-- Reading the even half at an even position, named.

    (k : ℕ)                                        [a position]
    outL (extPos k) = some (k, one)                [the even copy is read] -/
theorem outL_ext (k : ℕ) : outL (extPos k) = some (k, one) := outL_inl k

/-- Reading the even half at an odd position, named.

    (u : ℕ)                                        [a position]
    outL (argPos u) = none                         [the odd copy is not read] -/
theorem outL_arg (u : ℕ) : outL (argPos u) = none := outL_inr u

/-- Reading the odd half at an even position, named.

    (k : ℕ)                                        [a position]
    outR (extPos k) = none                         [the even copy is not read] -/
theorem outR_ext (k : ℕ) : outR (extPos k) = none := outR_inl k

/-- Reading the odd half at an odd position, named.

    (u : ℕ)                                        [a position]
    outR (argPos u) = some (u, one)                [the odd copy is read] -/
theorem outR_arg (u : ℕ) : outR (argPos u) = some (u, one) := outR_inr u

/-- The first route's tracks cross nowhere.

    (σ : ℝ) (x : Route) (p : ℕ)                        [the parameter, a route, a position]
    direct σ (plug σ bRoute x) (xPos p) = none         [no straight crossing there] -/
theorem b_direct_x (σ : ℝ) (x : Route) (p : ℕ) :
    direct σ (plug σ bRoute x) (xPos p) = none := by
  rw [direct_ext, plug_bRoute, bFun_xPos, rtensor_arg]
  cases hx : x p with
  | none => rfl
  | some q =>
    rw [Option.map_some, Option.map_some, Option.bind_some, outL_arg]
    rfl

/-- The outer argument track crosses straight to the answering track.

    (σ : ℝ) (x : Route) (v : ℕ)                        [the parameter, a route, a position]
    direct σ (plug σ bRoute x) (zPos v)                [the straight crossing there]
    = some (innPos v, one)                             [reaches the answering track] -/
theorem b_direct_z (σ : ℝ) (x : Route) (v : ℕ) :
    direct σ (plug σ bRoute x) (zPos v) = some (innPos v, one) := by
  rw [direct_ext, plug_bRoute, bFun_z, rtensor_ext_id, Option.map_some, Option.bind_some,
    outL_ext, Option.map_some, Option.map_some]
  simp only [mul_one_right]

/-- The answering track crosses straight to the outer argument track.

    (σ : ℝ) (x : Route) (v : ℕ)                        [the parameter, a route, a position]
    direct σ (plug σ bRoute x) (innPos v)              [the straight crossing there]
    = some (zPos v, one)                               [reaches the outer argument] -/
theorem b_direct_inn (σ : ℝ) (x : Route) (v : ℕ) :
    direct σ (plug σ bRoute x) (innPos v) = some (zPos v, one) := by
  rw [direct_ext, plug_bRoute, bFun_inn, rtensor_ext_id, Option.map_some, Option.bind_some,
    outL_ext, Option.map_some, Option.map_some]
  simp only [mul_one_right]

/-- Entering the loop from the first route's tracks reads that route.

    (σ : ℝ) (x : Route) (p : ℕ)                        [the parameter, a route, a position]
    entry σ (plug σ bRoute x) (xPos p) = x p           [the way in is the route itself] -/
theorem b_entry_x (σ : ℝ) (x : Route) (p : ℕ) :
    entry σ (plug σ bRoute x) (xPos p) = x p := by
  rw [entry_ext, plug_bRoute, bFun_xPos, rtensor_arg]
  cases hx : x p with
  | none => rfl
  | some q =>
    rw [Option.map_some, Option.map_some, Option.bind_some, outR_arg, Option.map_some,
      Option.map_some]
    simp only [Trichotomy.one_mul, mul_one_right]

/-- The outer argument track does not enter the loop.

    (σ : ℝ) (x : Route) (v : ℕ)                        [the parameter, a route, a position]
    entry σ (plug σ bRoute x) (zPos v) = none          [it crosses instead] -/
theorem b_entry_z (σ : ℝ) (x : Route) (v : ℕ) :
    entry σ (plug σ bRoute x) (zPos v) = none := by
  rw [entry_ext, plug_bRoute, bFun_z, rtensor_ext_id, Option.map_some, Option.bind_some,
    outR_ext]
  rfl

/-- The answering track does not enter the loop.

    (σ : ℝ) (x : Route) (v : ℕ)                        [the parameter, a route, a position]
    entry σ (plug σ bRoute x) (innPos v) = none        [it crosses instead] -/
theorem b_entry_inn (σ : ℝ) (x : Route) (v : ℕ) :
    entry σ (plug σ bRoute x) (innPos v) = none := by
  rw [entry_ext, plug_bRoute, bFun_inn, rtensor_ext_id, Option.map_some, Option.bind_some,
    outR_ext]
  rfl

/-- The loop is never taken twice.

    (σ : ℝ) (x : Route) (c : ℕ)                        [the parameter, a route, a position]
    body σ (plug σ bRoute x) c = none                  [one trip and no more] -/
theorem b_body (σ : ℝ) (x : Route) (c : ℕ) :
    body σ (plug σ bRoute x) c = none := by
  rw [body_arg, plug_bRoute, bFun_odd, rtensor_ext_id, Option.map_some, Option.bind_some,
    outR_ext]
  rfl

/-- Leaving the loop lands on the first route's tracks.

    (σ : ℝ) (x : Route) (c : ℕ)                        [the parameter, a route, a position]
    leave σ (plug σ bRoute x) c = some (xPos c, one)   [the way out lays it on those tracks] -/
theorem b_leave (σ : ℝ) (x : Route) (c : ℕ) :
    leave σ (plug σ bRoute x) c = some (xPos c, one) := by
  rw [leave_arg, plug_bRoute, bFun_odd, rtensor_ext_id, Option.map_some, Option.bind_some,
    outL_ext, Option.map_some, Option.map_some]
  simp only [mul_one_right]

/-- One pass through the loop is the first route, laid on its tracks.

    (σ : ℝ) (x : Route) (p : ℕ)                        [the parameter, a route, a position]
    rcomp σ (entry σ (plug σ bRoute x)) (leave σ (plug σ bRoute x)) (xPos p)   [in and out]
    = (x p).map fun q => (xPos q.1, q.2)                                       [is that route] -/
theorem entry_leave_b (σ : ℝ) (x : Route) (p : ℕ) :
    rcomp σ (entry σ (plug σ bRoute x)) (leave σ (plug σ bRoute x)) (xPos p)
      = (x p).map fun q => (xPos q.1, q.2) := by
  rw [rcomp, b_entry_x]
  cases hx : x p with
  | none => rfl
  | some q =>
    rw [Option.bind_some, b_leave, Option.map_some, Option.map_some, AnyRing.ringMul_one]

/-- **Applying the composing routing stores the route.**

@cite curry-1930

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ (plug σ bRoute x) i j w               [a path of the applied routing]
    ↔                                              [exactly when]
    bStored x i = some (j, w)                      [the stored route takes i there] -/
theorem bRoute_app (σ : ℝ) (x : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ bRoute x) i j w ↔ bStored x i = some (j, w) := by
  have hbody : body σ (plug σ bRoute x) = rnone := funext (b_body σ x)
  rcases pos_split i with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨c, rfl⟩
  · rw [← xPos_ext, bStored_x]
    constructor
    · rintro (hd | ⟨n, hn⟩)
      · rw [b_direct_x] at hd; exact absurd hd (by simp)
      · cases n with
        | zero =>
          rw [rpow, rcomp_rid_left, entry_leave_b] at hn
          exact hn
        | succ n =>
          rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
          simp [rnone] at hn
    · intro h
      refine Or.inr ⟨0, ?_⟩
      rw [rpow, rcomp_rid_left, entry_leave_b]
      exact h
  · rw [← zPos_eq, bStored_z]
    constructor
    · rintro (hd | ⟨n, hn⟩)
      · rw [b_direct_z] at hd; exact hd
      · cases n with
        | zero =>
          rw [rpow, rcomp_rid_left, rcomp, b_entry_z] at hn
          simp at hn
        | succ n =>
          rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
          simp [rnone] at hn
    · intro h
      refine Or.inl ?_
      rw [b_direct_z]
      exact h
  · rcases pos_split2 c with ⟨u, rfl⟩ | ⟨v, rfl⟩
    · rw [← outPos_eq, ← xPos_arg, bStored_x]
      constructor
      · rintro (hd | ⟨n, hn⟩)
        · rw [b_direct_x] at hd; exact absurd hd (by simp)
        · cases n with
          | zero =>
            rw [rpow, rcomp_rid_left, entry_leave_b] at hn
            exact hn
          | succ n =>
            rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
            simp [rnone] at hn
      · intro h
        refine Or.inr ⟨0, ?_⟩
        rw [rpow, rcomp_rid_left, entry_leave_b]
        exact h
    · rw [← innPos_eq, bStored_inn]
      constructor
      · rintro (hd | ⟨n, hn⟩)
        · rw [b_direct_inn] at hd; exact hd
        · cases n with
          | zero =>
            rw [rpow, rcomp_rid_left, rcomp, b_entry_inn] at hn
            simp at hn
          | succ n =>
            rw [hbody, rpow, rcomp_rnone_left, rcomp_rnone_left, rcomp_rnone_right] at hn
            simp [rnone] at hn
      · intro h
        refine Or.inl ?_
        rw [b_direct_inn]
        exact h

/-- **The composing routing applied to a route stores it.**

@cite curry-1930
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    rapp σ bRoute x = bStored x                    [one constant route, applied once]

    So: the loop is empty because the routing sends every loop position straight out, and what
    the trace leaves is the first route on the two tracks `Combinator/Composition.lean` gives it,
    together with a crossing between the outer argument track and the answering track. That
    crossing is the part no relabelling could supply, and it is what will let the second route
    reach the outer argument. -/
theorem rapp_bRoute (σ : ℝ) (x : Route) : rapp σ bRoute x = bStored x :=
  rapp_eq (bRoute_app σ x)

end Combinator
end Apeiron

end
