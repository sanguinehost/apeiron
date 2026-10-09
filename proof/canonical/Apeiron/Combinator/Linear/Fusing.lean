import Apeiron.Combinator.Linear.Storing

/-!
# One machine with the second route's step fused in

@name Fusing
@cite girard-1989
@cite joyal-street-verity-1996
@cite curry-1930
@grade proved

`Combinator/Storing.lean` computes `rapp σ bRoute x` as `bStored x`. What remains for the
composing combinator as an element is that applying `bStored x` to the second route gives
`bComb σ x y`, and that is not the comparison `Combinator/Compositor.lean` runs.

## The correction

An earlier estimate said the remaining step was the two readings comparison already in hand. It
is not, and the reason is worth stating because it is forced rather than incidental. In the
element form the second route is fed through `plug`, which applies it at **every** odd position.
So the position where the first route asks and the position where the second route answers are
the same position, and one step of `plug σ (bStored x) y` runs the first route and then the
second. In `bStore σ x y` they are two positions and two steps. The two machines therefore have
different step counts, and no relabelling repairs that, because the fusion is what `plug` does.

## What does work

Collapse the answering track on both sides first. `traces_flatten` says a trace is the collapse
of the inner track followed by a run along the outer one, and after that collapse the two
machines become the **same route** on every surviving position:

```
   from the first route's tracks   run the first route, then the second, then out
   from the outer argument track   run the second route, then out
```

One side reaches that by fusing at the start and taking an extra step at the end, the other by
taking an extra step at the start and fusing at the end, and the endpoints and weights agree.
`merge_x` and `merge_z` are those two computations, and `runs_agree_off` carries them back to a
statement about traces, since a path along the outer track never visits the inner one.

## Root

| from | here |
|---|---|
| `one_mul`, `mul_comm` through `mul_one_right` | `gone_z`, `merge_x`, `merge_z` |

The routing is arithmetic on the counting numbers, so the A1 spectator test applies as throughout.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy
open RouteAnyRing

-- ═══════════════════════════════════════════════════════════════════
-- READING A PATH OFF ONE STEP
-- ═══════════════════════════════════════════════════════════════════

/-- A route that stops has no paths.

    {σ : ℝ} {f : Route} {S : Set ℕ} {i : ℕ}        [parameter, route, loop, position]
    (h : f i = none) (c : ℕ) (w : Cell)            [the route stops there]
    ¬ Runs σ f S i c w                             [so no path starts there] -/
theorem runs_none {σ : ℝ} {f : Route} {S : Set ℕ} {i : ℕ} (h : f i = none) (c : ℕ) (w : Cell) :
    ¬ Runs σ f S i c w := by
  intro hr
  obtain ⟨m, u, hm⟩ := runs_start_some hr
  rw [h] at hm
  exact absurd hm (by simp)

/-- A step that leaves the loop is the whole path.

    {σ : ℝ} {f : Route} {S : Set ℕ} {i m : ℕ} {u : Cell}   [parameter, route, loop, a step]
    (h : f i = some (m, u)) (hm : m ∉ S)                   [one step, landing outside]
    (c : ℕ) (w : Cell)                                     [an end and a weight]
    Runs σ f S i c w ↔ (c = m ∧ w = u)                     [the path is that step] -/
theorem runs_exit_iff {σ : ℝ} {f : Route} {S : Set ℕ} {i m : ℕ} {u : Cell}
    (h : f i = some (m, u)) (hm : m ∉ S) (c : ℕ) (w : Cell) :
    Runs σ f S i c w ↔ (c = m ∧ w = u) := by
  constructor
  · rintro ⟨n, hn⟩
    cases hn with
    | exit hf _ =>
      rw [h] at hf
      injection hf with hq
      injection hq with h1 h2
      exact ⟨h1.symm, h2.symm⟩
    | step hf hs _ =>
      rw [h] at hf
      injection hf with hq
      injection hq with h1 _
      exact absurd (h1 ▸ hs) hm
  · rintro ⟨rfl, rfl⟩
    exact ⟨0, RunsN.exit h hm⟩

/-- A step that stays in the loop is the first step of the path.

    {σ : ℝ} {f : Route} {S : Set ℕ} {i m : ℕ} {u : Cell}   [parameter, route, loop, a step]
    (h : f i = some (m, u)) (hm : m ∈ S)                   [one step, landing inside]
    (c : ℕ) (w : Cell)                                     [an end and a weight]
    Runs σ f S i c w                                       [a path from i]
    ↔ ∃ w', Runs σ f S m c w' ∧ w = mul σ u w'             [is that step and a path from m] -/
theorem runs_step_iff {σ : ℝ} {f : Route} {S : Set ℕ} {i m : ℕ} {u : Cell}
    (h : f i = some (m, u)) (hm : m ∈ S) (c : ℕ) (w : Cell) :
    Runs σ f S i c w ↔ ∃ w', Runs σ f S m c w' ∧ w = mul σ u w' := by
  constructor
  · rintro ⟨n, hn⟩
    cases hn with
    | exit hf hns =>
      rw [h] at hf
      injection hf with hq
      injection hq with h1 _
      exact absurd (h1 ▸ hm) hns
    | @step n _ k _ _ v hf hs hr =>
      rw [h] at hf
      injection hf with hq
      injection hq with h1 h2
      subst h1
      subst h2
      exact ⟨_, ⟨n, hr⟩, rfl⟩
  · rintro ⟨w', ⟨n, hn⟩, rfl⟩
    exact ⟨n + 1, RunsN.step h hm hn⟩

-- ═══════════════════════════════════════════════════════════════════
-- ROUTES THAT AGREE OFF A TRACK HAVE THE SAME PATHS THROUGH ANOTHER
-- ═══════════════════════════════════════════════════════════════════

/-- Agreement off one track carries paths through a track disjoint from it.

    {σ : ℝ} {f g : Route} {S T : Set ℕ}                    [parameter, two routes, two tracks]
    (hd : ∀ p, p ∈ S → p ∉ T) (hfg : ∀ p, p ∉ T → f p = g p)   [disjoint, and agreeing off T]
    ∀ {n i j : ℕ} {w : Cell}                              [a length, two positions, a weight]
    RunsN σ f S n i j w → i ∉ T →                          [a path of one, starting off T]
      RunsN σ g S n i j w                                  [is a path of the other] -/
theorem runsN_agree_off {σ : ℝ} {f g : Route} {S T : Set ℕ}
    (hd : ∀ p, p ∈ S → p ∉ T) (hfg : ∀ p, p ∉ T → f p = g p) :
    ∀ {n i j : ℕ} {w : Cell}, RunsN σ f S n i j w → i ∉ T → RunsN σ g S n i j w := by
  intro n i j w h
  induction h with
  | @exit i j w hf hns =>
    intro hi
    exact RunsN.exit (by rw [← hfg i hi]; exact hf) hns
  | @step n i k j w v hf hs hr ih =>
    intro hi
    exact RunsN.step (by rw [← hfg i hi]; exact hf) hs (ih (hd k hs))

/-- **Routes agreeing off one track have the same paths through a disjoint one.**

    {σ : ℝ} {f g : Route} {S T : Set ℕ}                    [parameter, two routes, two tracks]
    (hd : ∀ p, p ∈ S → p ∉ T) (hfg : ∀ p, p ∉ T → f p = g p)   [disjoint, and agreeing off T]
    {i j : ℕ} {w : Cell} (hi : i ∉ T)                      [a path starting off T]
    Runs σ f S i j w ↔ Runs σ g S i j w                    [the two agree on it]

    So: a path which stays inside `S` only ever reads the route at positions outside `T`, so what
    the two routes do on `T` cannot be seen. That is what lets two machines which differ only on
    the answering track be compared along the asking one. -/
theorem runs_agree_off {σ : ℝ} {f g : Route} {S T : Set ℕ}
    (hd : ∀ p, p ∈ S → p ∉ T) (hfg : ∀ p, p ∉ T → f p = g p) {i j : ℕ} {w : Cell}
    (hi : i ∉ T) : Runs σ f S i j w ↔ Runs σ g S i j w :=
  ⟨fun ⟨n, h⟩ => ⟨n, runsN_agree_off hd hfg h hi⟩,
    fun ⟨n, h⟩ => ⟨n, runsN_agree_off hd (fun p hp => (hfg p hp).symm) h hi⟩⟩

/-- Two collapses with the same paths at a position agree there.

    {σ : ℝ} {f g : Route} {S T : Set ℕ} {i : ℕ}                  [parameter, routes, sets, spot]
    (h : ∀ c w, Runs σ f S i c w ↔ Runs σ g T i c w)             [the same paths from i]
    runsRoute σ f S i = runsRoute σ g T i                        [so the same collapsed step] -/
theorem runsRoute_ext_at {σ : ℝ} {f g : Route} {S T : Set ℕ} {i : ℕ}
    (h : ∀ c w, Runs σ f S i c w ↔ Runs σ g T i c w) :
    runsRoute σ f S i = runsRoute σ g T i := by
  cases hf : runsRoute σ f S i with
  | none =>
    cases hg : runsRoute σ g T i with
    | none => rfl
    | some q =>
      have hr := (h q.1 q.2).mpr ((runsRoute_eq σ g T i q.1 q.2).mp (by rw [hg]))
      rw [(runsRoute_eq σ f S i q.1 q.2).mpr hr] at hf
      exact absurd hf (by simp)
  | some q =>
    have hr := (h q.1 q.2).mp ((runsRoute_eq σ f S i q.1 q.2).mp (by rw [hf]))
    rw [(runsRoute_eq σ g T i q.1 q.2).mpr hr]

-- ═══════════════════════════════════════════════════════════════════
-- ONE STEP OF EACH MACHINE
-- ═══════════════════════════════════════════════════════════════════

/-- Where the second route's step lands after the wire.

    (r : ℕ)                                        [one of the second route's positions]
    bSwapFun (yPos r)                              [its place, then the wire]
    : ℕ                                            [a position of the combined wire] -/
def yOut (r : ℕ) : ℕ := bSwapFun (yPos r)

/-- The second route's answer reaches the asking track.

    (n : ℕ)                                        [a position]
    yOut (extPos n) = outPos n                     [an answer crosses to where the first waits] -/
theorem yOut_ext (n : ℕ) : yOut (extPos n) = outPos n := by
  rw [yOut, yPos_ext, bSwapFun_inn]

/-- The second route's question reaches the outer argument.

    (v : ℕ)                                        [a position]
    yOut (argPos v) = zPos v                       [a question crosses to the argument track] -/
theorem yOut_arg (v : ℕ) : yOut (argPos v) = zPos v := by
  rw [yOut, yPos_arg, zPos_eq, bSwapFun_ext]

/-- The second route's places are never on the answering track's exit side.

    (r : ℕ)                                        [one of the second route's positions]
    yOut r ∉ innerSet                              [an answer or a question leaves the track] -/
theorem yOut_not_inner (r : ℕ) : yOut r ∉ innerSet := by
  rcases pos_split2 r with ⟨n, rfl⟩ | ⟨v, rfl⟩
  · rw [yOut_ext]
    exact outer_disjoint_inner _ (mem_outerSet n)
  · rw [yOut_arg, zPos_eq]
    exact (extPos_not_mem (argPos v)).2

/-- The first route's places are off the answering track.

    (p : ℕ)                                        [one of the first route's positions]
    xPos p ∉ innerSet                              [the first route never sits there] -/
theorem xPos_not_inner (p : ℕ) : xPos p ∉ innerSet := by
  rcases pos_split2 p with ⟨k, rfl⟩ | ⟨u, rfl⟩
  · rw [xPos_ext]
    exact (extPos_not_mem (extPos k)).2
  · rw [xPos_arg]
    exact outer_disjoint_inner _ (mem_outerSet u)

/-- The stored pair on the first route's tracks.

    (σ : ℝ) (x y : Route) (p : ℕ)                  [the parameter, two routes, a position]
    bStore σ x y (xPos p)                                        [a step where the first sits]
    = (x p).map fun q => (xOut q.1, q.2)                         [is that route, then the wire] -/
theorem bStore_xPos (σ : ℝ) (x y : Route) (p : ℕ) :
    bStore σ x y (xPos p) = (x p).map fun q => (xOut q.1, q.2) := by
  rw [bStore, rcomp, pairRoute_x]
  cases hx : x p with
  | none => rfl
  | some q =>
    simp only [Option.map_some, Option.bind_some, swapRoute, xOut, mul_one_right]

/-- The stored pair on the second route's tracks.

    (σ : ℝ) (x y : Route) (r : ℕ)                  [the parameter, two routes, a position]
    bStore σ x y (yPos r)                                        [a step where the second sits]
    = (y r).map fun s => (yOut s.1, s.2)                         [is that route, then the wire] -/
theorem bStore_yPos (σ : ℝ) (x y : Route) (r : ℕ) :
    bStore σ x y (yPos r) = (y r).map fun s => (yOut s.1, s.2) := by
  rw [bStore, rcomp, pairRoute_y]
  cases hy : y r with
  | none => rfl
  | some s =>
    simp only [Option.map_some, Option.bind_some, swapRoute, yOut, mul_one_right]

/-- One step of the element machine on the first route's tracks.

    (σ : ℝ) (x y : Route) (p : ℕ)                  [the parameter, two routes, a position]
    plug σ (bStored x) y (xPos p)                                     [a step of the machine]
    = (x p).bind fun q =>                                             [run the first route]
      ((rtensor rid y) (xPos q.1)).map fun r => (r.1, mul σ q.2 r.2)  [then the second] -/
theorem gone_x (σ : ℝ) (x y : Route) (p : ℕ) :
    plug σ (bStored x) y (xPos p)
      = (x p).bind fun q => ((rtensor rid y) (xPos q.1)).map fun r => (r.1, mul σ q.2 r.2) := by
  rw [plug, rcomp, bStored_x]
  cases hx : x p with
  | none => rfl
  | some q => rw [Option.map_some, Option.bind_some, Option.bind_some]

/-- The element machine when the first route leaves.

    (σ : ℝ) (x y : Route) {p k : ℕ} {w : Cell}     [the parameter, two routes, a step]
    (h : x p = some (extPos k, w))                 [the first route leaves]
    plug σ (bStored x) y (xPos p) = some (xPos (extPos k), w)   [and the second is not consulted] -/
theorem gone_x_ext (σ : ℝ) (x y : Route) {p k : ℕ} {w : Cell} (h : x p = some (extPos k, w)) :
    plug σ (bStored x) y (xPos p) = some (xPos (extPos k), w) := by
  rw [gone_x, h, Option.bind_some, xPos_ext, rtensor_ext_id, Option.map_some, mul_one_right]

/-- The element machine when the first route asks.

    (σ : ℝ) (x y : Route) {p u : ℕ} {w : Cell}     [the parameter, two routes, a step]
    (h : x p = some (argPos u, w))                 [the first route asks]
    plug σ (bStored x) y (xPos p)                                     [the step continues]
    = (y (extPos u)).map fun r => (argPos r.1, mul σ w r.2)           [into the second route] -/
theorem gone_x_arg (σ : ℝ) (x y : Route) {p u : ℕ} {w : Cell} (h : x p = some (argPos u, w)) :
    plug σ (bStored x) y (xPos p) = (y (extPos u)).map fun r => (argPos r.1, mul σ w r.2) := by
  rw [gone_x, h, Option.bind_some, xPos_arg, outPos_eq, rtensor_arg]
  cases hy : y (extPos u) with
  | none => rfl
  | some r => rw [Option.map_some, Option.map_some, Option.map_some]

/-- One step of the element machine on the outer argument track.

    (σ : ℝ) (x y : Route) (v : ℕ)                  [the parameter, two routes, a position]
    plug σ (bStored x) y (zPos v)                                [a step from the argument]
    = (y (argPos v)).map fun r => (argPos r.1, r.2)              [runs the second route] -/
theorem gone_z (σ : ℝ) (x y : Route) (v : ℕ) :
    plug σ (bStored x) y (zPos v) = (y (argPos v)).map fun r => (argPos r.1, r.2) := by
  rw [plug, rcomp, bStored_z, Option.bind_some, innPos_eq, rtensor_arg]
  cases hy : y (argPos v) with
  | none => rfl
  | some r =>
    simp only [Option.map_some, Option.map_some, Trichotomy.one_mul]

/-- One step of the element machine on the answering track.

    (σ : ℝ) (x y : Route) (v : ℕ)                  [the parameter, two routes, a position]
    plug σ (bStored x) y (innPos v) = some (zPos v, one)   [crosses to the argument track] -/
theorem gone_inn (σ : ℝ) (x y : Route) (v : ℕ) :
    plug σ (bStored x) y (innPos v) = some (zPos v, one) := by
  rw [plug, rcomp, bStored_inn, Option.bind_some, zPos_eq, rtensor_ext_id, Option.map_some,
    ← zPos_eq]
  simp only [Trichotomy.one_mul]


/-- The stored pair on the answering track.

    (σ : ℝ) (x y : Route) (n : ℕ)                  [the parameter, two routes, a position]
    bStore σ x y (innPos n)                                      [a step on the answering track]
    = (y (extPos n)).map fun s => (yOut s.1, s.2)                [runs the second route] -/
theorem bStore_inn (σ : ℝ) (x y : Route) (n : ℕ) :
    bStore σ x y (innPos n) = (y (extPos n)).map fun s => (yOut s.1, s.2) := by
  rw [← yPos_ext, bStore_yPos]

/-- The stored pair on the outer argument track.

    (σ : ℝ) (x y : Route) (v : ℕ)                  [the parameter, two routes, a position]
    bStore σ x y (zPos v)                                        [a step from the argument]
    = (y (argPos v)).map fun s => (yOut s.1, s.2)                [runs the second route] -/
theorem bStore_zPos (σ : ℝ) (x y : Route) (v : ℕ) :
    bStore σ x y (zPos v) = (y (argPos v)).map fun s => (yOut s.1, s.2) := by
  rw [← yPos_arg, bStore_yPos]

/-- The outer argument track is not the answering track.

    (v : ℕ)                                        [a position]
    zPos v ∉ innerSet                              [an even track is not an odd one] -/
theorem zPos_not_inner (v : ℕ) : zPos v ∉ innerSet := by
  rw [zPos_eq]
  exact (extPos_not_mem (argPos v)).2

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO MACHINES AGREE ONCE THE ANSWERING TRACK IS COLLAPSED
-- ═══════════════════════════════════════════════════════════════════

/-- **The collapsed machines agree on the first route's tracks.**

@cite joyal-street-verity-1996

    (σ : ℝ) (x y : Route) (p c : ℕ) (w : Cell)     [the parameter, two routes, a path]
    Runs σ (plug σ (bStored x) y) innerSet (xPos p) c w      [a path of the element machine]
    ↔                                                        [exactly when]
    Runs σ (bStore σ x y) innerSet (xPos p) c w              [one of the stored pair]

    So: the element machine runs both routes in one step and then takes an extra step when the
    second route asks; the stored pair takes an extra step to reach the second route and then
    fuses the question into it. The endpoints and the weights agree, which is the whole of what
    the trace sees. -/
theorem merge_x (σ : ℝ) (x y : Route) (p c : ℕ) (w : Cell) :
    Runs σ (plug σ (bStored x) y) innerSet (xPos p) c w
      ↔ Runs σ (bStore σ x y) innerSet (xPos p) c w := by
  cases hx : x p with
  | none =>
    have hL : plug σ (bStored x) y (xPos p) = none := by rw [gone_x, hx]; rfl
    have hR : bStore σ x y (xPos p) = none := by rw [bStore_xPos, hx]; rfl
    exact ⟨fun h => absurd h (runs_none hL c w), fun h => absurd h (runs_none hR c w)⟩
  | some q =>
    obtain ⟨q1, w1⟩ := q
    rcases pos_split2 q1 with ⟨k, rfl⟩ | ⟨u, rfl⟩
    · have hL : plug σ (bStored x) y (xPos p) = some (extPos (extPos k), w1) := by
        rw [gone_x_ext σ x y hx, xPos_ext]
      have hR : bStore σ x y (xPos p) = some (extPos (extPos k), w1) := by
        rw [bStore_xPos, hx, Option.map_some, xOut_ext]
      rw [runs_exit_iff hL (extPos_not_mem (extPos k)).2 c w,
        runs_exit_iff hR (extPos_not_mem (extPos k)).2 c w]
    · have hR : bStore σ x y (xPos p) = some (innPos u, w1) := by
        rw [bStore_xPos, hx, Option.map_some, xOut_arg]
      cases hy : y (extPos u) with
      | none =>
        have hL : plug σ (bStored x) y (xPos p) = none := by
          rw [gone_x_arg σ x y hx, hy]; rfl
        have hI : bStore σ x y (innPos u) = none := by rw [bStore_inn, hy]; rfl
        constructor
        · intro h
          exact absurd h (runs_none hL c w)
        · intro h
          obtain ⟨w', hw', -⟩ := (runs_step_iff hR (mem_innerSet u) c w).mp h
          exact absurd hw' (runs_none hI c w')
      | some r =>
        obtain ⟨r1, w2⟩ := r
        have hI : bStore σ x y (innPos u) = some (yOut r1, w2) := by
          rw [bStore_inn, hy, Option.map_some]
        have hRfull : Runs σ (bStore σ x y) innerSet (xPos p) c w
            ↔ (c = yOut r1 ∧ w = mul σ w1 w2) := by
          rw [runs_step_iff hR (mem_innerSet u) c w]
          constructor
          · rintro ⟨w', hw', rfl⟩
            obtain ⟨rfl, rfl⟩ := (runs_exit_iff hI (yOut_not_inner r1) c w').mp hw'
            exact ⟨rfl, rfl⟩
          · rintro ⟨rfl, rfl⟩
            exact ⟨w2, (runs_exit_iff hI (yOut_not_inner r1) _ _).mpr ⟨rfl, rfl⟩, rfl⟩
        rw [hRfull]
        rcases pos_split2 r1 with ⟨n, rfl⟩ | ⟨v, rfl⟩
        · have hL : plug σ (bStored x) y (xPos p) = some (outPos n, mul σ w1 w2) := by
            rw [gone_x_arg σ x y hx, hy, Option.map_some, outPos_eq]
          rw [runs_exit_iff hL (outer_disjoint_inner _ (mem_outerSet n)) c w, yOut_ext]
        · have hL : plug σ (bStored x) y (xPos p) = some (innPos v, mul σ w1 w2) := by
            rw [gone_x_arg σ x y hx, hy, Option.map_some, innPos_eq]
          rw [yOut_arg]
          constructor
          · intro h
            obtain ⟨w'', hw'', rfl⟩ := (runs_step_iff hL (mem_innerSet v) c w).mp h
            obtain ⟨rfl, rfl⟩ :=
              (runs_exit_iff (gone_inn σ x y v) (zPos_not_inner v) c w'').mp hw''
            exact ⟨rfl, mul_one_right σ (mul σ w1 w2)⟩
          · rintro ⟨rfl, rfl⟩
            refine (runs_step_iff hL (mem_innerSet v) _ _).mpr ⟨one, ?_, ?_⟩
            · exact (runs_exit_iff (gone_inn σ x y v) (zPos_not_inner v) _ _).mpr ⟨rfl, rfl⟩
            · rw [mul_one_right]

/-- **The collapsed machines agree on the outer argument track.**

@cite joyal-street-verity-1996

    (σ : ℝ) (x y : Route) (v c : ℕ) (w : Cell)     [the parameter, two routes, a path]
    Runs σ (plug σ (bStored x) y) innerSet (zPos v) c w      [a path of the element machine]
    ↔                                                        [exactly when]
    Runs σ (bStore σ x y) innerSet (zPos v) c w              [one of the stored pair] -/
theorem merge_z (σ : ℝ) (x y : Route) (v c : ℕ) (w : Cell) :
    Runs σ (plug σ (bStored x) y) innerSet (zPos v) c w
      ↔ Runs σ (bStore σ x y) innerSet (zPos v) c w := by
  cases hy : y (argPos v) with
  | none =>
    have hL : plug σ (bStored x) y (zPos v) = none := by rw [gone_z, hy]; rfl
    have hR : bStore σ x y (zPos v) = none := by rw [bStore_zPos, hy]; rfl
    exact ⟨fun h => absurd h (runs_none hL c w), fun h => absurd h (runs_none hR c w)⟩
  | some r =>
    obtain ⟨r1, w2⟩ := r
    have hR : bStore σ x y (zPos v) = some (yOut r1, w2) := by
      rw [bStore_zPos, hy, Option.map_some]
    rw [runs_exit_iff hR (yOut_not_inner r1) c w]
    rcases pos_split2 r1 with ⟨n, rfl⟩ | ⟨v', rfl⟩
    · have hL : plug σ (bStored x) y (zPos v) = some (outPos n, w2) := by
        rw [gone_z, hy, Option.map_some, outPos_eq]
      rw [runs_exit_iff hL (outer_disjoint_inner _ (mem_outerSet n)) c w, yOut_ext]
    · have hL : plug σ (bStored x) y (zPos v) = some (innPos v', w2) := by
        rw [gone_z, hy, Option.map_some, innPos_eq]
      rw [yOut_arg]
      constructor
      · intro h
        obtain ⟨w'', hw'', rfl⟩ := (runs_step_iff hL (mem_innerSet v') c w).mp h
        obtain ⟨rfl, rfl⟩ :=
          (runs_exit_iff (gone_inn σ x y v') (zPos_not_inner v') c w'').mp hw''
        exact ⟨rfl, mul_one_right σ w2⟩
      · rintro ⟨rfl, rfl⟩
        refine (runs_step_iff hL (mem_innerSet v') _ _).mpr ⟨one, ?_, ?_⟩
        · exact (runs_exit_iff (gone_inn σ x y v') (zPos_not_inner v') _ _).mpr ⟨rfl, rfl⟩
        · rw [mul_one_right]

-- ═══════════════════════════════════════════════════════════════════
-- THE COMPOSING COMBINATOR AS AN ELEMENT
-- ═══════════════════════════════════════════════════════════════════

/-- **The element machine and the stored pair have the same trace.**

@cite joyal-street-verity-1996
@cite girard-1989

    (σ : ℝ) (x y : Route) (i j : ℕ) (w : Cell)     [the parameter, two routes, a step]
    Traces σ (plug σ (bStored x) y) i j w          [a path of the element machine]
    ↔                                              [exactly when]
    Traces σ (bStore σ x y) i j w                  [one of the stored pair]

    So: collapse the answering track on both sides, which `traces_flatten` licenses, and the two
    machines become the same route on every surviving position. A path along the asking track
    never visits the answering one, so `runs_agree_off` carries the agreement through. -/
theorem bStored_traces (σ : ℝ) (x y : Route) (i j : ℕ) (w : Cell) :
    Traces σ (plug σ (bStored x) y) i j w ↔ Traces σ (bStore σ x y) i j w := by
  rw [traces_flatten, traces_flatten]
  refine runs_agree_off outer_disjoint_inner ?_ (extPos_not_mem i).2
  intro q hq
  refine runsRoute_ext_at ?_
  intro c w'
  rcases pos_split q with ⟨k, rfl⟩ | ⟨v, rfl⟩ | ⟨d, rfl⟩
  · rw [← xPos_ext]
    exact merge_x σ x y (extPos k) c w'
  · rw [← zPos_eq]
    exact merge_z σ x y v c w'
  · rcases pos_split2 d with ⟨u, rfl⟩ | ⟨v, rfl⟩
    · rw [← outPos_eq, ← xPos_arg]
      exact merge_x σ x y (argPos u) c w'
    · rw [← innPos_eq] at hq
      exact absurd (mem_innerSet v) hq

/-- **Applying the stored first route to the second gives the composite.**

@cite curry-1930

    (σ : ℝ) (x y : Route)                          [the parameter and two routes]
    rapp σ (bStored x) y = bComb σ x y             [the element form is the construction] -/
theorem rapp_bStored (σ : ℝ) (x y : Route) : rapp σ (bStored x) y = bComb σ x y := by
  refine rapp_eq ?_
  intro i j w
  rw [bStored_traces, bComb]
  exact tracesTo_traceRoute σ (bStore σ x y) i j w

/-- **B as an element.** The composing combinator is one fixed route.

@cite schonfinkel-1924
@cite curry-1930
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x y z : Route)                        [the parameter and three routes]
    rapp σ (rapp σ (rapp σ bRoute x) y) z          [one constant route, applied three times]
    = rapp σ x (rapp σ y z)                        [composes its first two arguments]

    So: the composing combinator is an element of this carrier, not only a construction on the
    routes it stores. What made it available is a routing whose loop re-enters its argument, and
    what made the equation checkable is that fusing the second route's step into the first
    route's does not change the trace. -/
theorem bRoute_bracket (σ : ℝ) (x y z : Route) :
    rapp σ (rapp σ (rapp σ bRoute x) y) z = rapp σ x (rapp σ y z) := by
  rw [rapp_bRoute, rapp_bStored]
  exact bracket σ x y z

end Combinator
end Apeiron

end
