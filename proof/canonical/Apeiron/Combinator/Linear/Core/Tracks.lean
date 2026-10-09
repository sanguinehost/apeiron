import Apeiron.Combinator.Linear.Core.Vanishing
import Apeiron.Combinator.RelationalPaths

/-!
# Three tracks on one wire

@name Tracks
@cite joyal-street-verity-1996
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@grade proved

`Combinator/Vanishing.lean` proves that tracing two disjoint loops one at a time is tracing them
together. To use that on a combinator, its stored route has to carry two loops at once, and this
file lays them out: one wire, three tracks, external and outer loop and inner loop.

## Root

| from | here |
|---|---|
| `mul_assoc`, `one_mul` through `runs_vanishing` | `traces_flatten` |

The layout is arithmetic on the counting numbers and says nothing about the parameter, which
`theory/NOVELTY.md`'s A1 spectator test settles: the same three tracks serve every setting. What
the parameter supplies is the multiplication that accumulates along each path, and that enters
through `runs_vanishing`, not here.

## The layout

Cantor's pairing is not needed: nesting the even and odd split twice is enough. A position is

```
   extPos k  =  even k                      the outside
   outPos u  =  odd (even u)                the loop that faces the argument
   innPos v  =  odd (odd v)                 the loop the stored route keeps to itself
```

`pos_trichotomy` says every position is exactly one of the three, `outer_disjoint_inner` says the
two loop tracks do not meet, and `oddPos_eq` says together they are the whole loop. Those three
facts are what `runs_vanishing` asks for.

## The flattening

`traces_flatten` is the payoff and it is one line of chaining: the trace of a route over its whole
loop is the trace, over the outer track alone, of the route whose inner-track paths have been
collapsed to single steps. That is what lets a combinator store a route which is itself a trace,
which `Combinator/Duplication.lean` shows is forced for any combinator that composes its
arguments. `runs_vanishing_three` is the same twice over, for three loops.

## What `S` needs, corrected

The report that asked for this file estimated the rest of `S` as index plumbing with no
conceptual risk. That estimate was wrong and the correction is worth recording, because it is
about `S` and not about the plumbing.

Count the loops on each side of `S x y z = x z (y z)`.

- The left has two. The stored route carries one loop of its own, by which `x` and `y` talk to
  each other, and the application to `z` opens one more, which both of them consult through.
- The right has three: the loop by which `x` consults its argument, the loop by which `y`
  consults its argument, and the loop by which `x z` is applied to `y z`.

`runs_vanishing_three` handles three loops, so that is not the difficulty. The difficulty is that
the two argument-facing loops on the right have to become the one argument-facing loop on the
left, and merging them is a bijection between position spaces, not a further property of the
trace. What performs that merge is the tagging of `Combinator/Duplication.lean`'s `bang`, whose
`bang_tag` keeps a requester's tag and works on the payload, so two consumers of one argument stay
apart.

That is the linear-logic content rather than an accident of this encoding. `S` uses its third
argument twice, so it is not linear, and Abramsky, Haghverdi and Scott do not put it in the linear
combinatory algebra: they obtain the ordinary one from it by `x · y := x (!y)`. The equation to
aim at here is therefore the banged one, and what it needs on top of this file is the routing and
the merge, not another theorem about tracing.

## Which of the four are reachable now, and which are not

The count above separates them, and the line it draws is linearity.

| combinator | each argument used | state |
|---|---|---|
| `I x = x` | once | proved, `Combinator/Duplication.lean` |
| `K x !y = x` | second discarded | proved, discarding costs nothing here |
| `B x y z = x (y z)` | once each | reachable: two loops each side |
| `C x y z = x z y` | once each | reachable: two loops each side |
| `W x !y = x !y !y` | second twice | needs the merge |
| `S x y z = x z (y z)` | third twice | needs the merge |

## The two copies may come from the tensor, not the exponential

Working `W x y = x y y` out far enough to see its shape turned up something worth recording
before anyone builds it.

The stored route `W x` must send `x`'s two argument-facing channels into the one channel it has,
which asks for a relabelling `ℕ ≃ ℕ` sending its own positions to `x`'s:

```
   even k            to   even (even k)          the outside
   odd (even v)      to   odd v                  x's first argument
   odd (odd u)       to   even (odd u)           x's second argument
```

That is a bijection of the whole of `ℕ`, and it is Mathlib's `Equiv.sumAssoc` and
`Equiv.sumComm` composed with the conversions `Equiv.natSumNatEquivNat` gives, so it needs no
round-trip proofs written by hand. The point is what it forces on the argument. Splitting the
channel by even and odd rather than by `Nat.pair` means the argument the stored route sees is
`rtensor y y`, two copies of `y` side by side, and **not** `bang y`.

So the binary contraction wants the tensor and not the exponential. `bang` is needed when a
duplicated argument must itself stay duplicable, which is what makes `!` reusable in a full
linear combinatory algebra; for `W x y = x y y`, and by the same count for
`S x y z = x z (y z)`, two copies is all that is asked and `rtensor` supplies exactly two. That
is a cheaper target than the banged form and it is the one to try first.

All the general machinery a composing combinator needs is now in the tree: `traces_conj` puts a
relabelled route's trace back in its argument's labels, `traces_flatten` and
`runs_vanishing_three` collapse the loops the layout leaves, and
`Combinator/Duplication.lean`'s `bang_restrict` performs the merge for an argument used twice.
What each combinator still needs is its own index bijection and its own routing, and no further
general theory.

`B` and `C` are linear, so their two sides carry the same two loops and only the labelling of
which is inner and which is outer differs, which `runs_vanishing` does not care about: it reads
in both directions. Neither needs `bang`. `W` and `S` duplicate an argument, so one loop on one
side must become two on the other, and that is the merge. Whoever carries this on should take `B`
and `C` first, since they need nothing beyond this file, and treat `W` and `S` as the separate
piece of work they are.

Worth keeping visible: the corpus already has `S`. `Combinator/Model.lean` proves `feed_sPoint`
and `feed_meaning_bracket` in Engeler's graph model over the algebra's idempotents. What this line
of files chases is `S` in the feedback setting, where the algebra's weights ride the wires.
-/

namespace Apeiron
namespace Combinator

open Trichotomy
open RouteAnyRing

universe u

-- ═══════════════════════════════════════════════════════════════════
-- THE THREE TRACKS
-- ═══════════════════════════════════════════════════════════════════

/-- A position outside both loops.

    (k : ℕ)                                        [a position]
    Equiv.natSumNatEquivNat (Sum.inl k)            [its even copy]
    : ℕ                                            [a position of the wire] -/
def extPos (k : ℕ) : ℕ := Equiv.natSumNatEquivNat (Sum.inl k)

/-- A position of the loop that faces the argument.

    (u : ℕ)                                        [a position]
    the even copy of u, made odd                   [its place on the outer track]
    : ℕ                                            [a position of the wire] -/
def outPos (u : ℕ) : ℕ :=
  Equiv.natSumNatEquivNat (Sum.inr (Equiv.natSumNatEquivNat (Sum.inl u)))

/-- A position of the loop the stored route keeps to itself.

    (v : ℕ)                                        [a position]
    the odd copy of v, made odd                    [its place on the inner track]
    : ℕ                                            [a position of the wire] -/
def innPos (v : ℕ) : ℕ :=
  Equiv.natSumNatEquivNat (Sum.inr (Equiv.natSumNatEquivNat (Sum.inr v)))

/-- The positions of the outer loop.

    Set ℕ                                          [the track that faces the argument] -/
def outerSet : Set ℕ := {p | ∃ u, p = outPos u}

/-- The positions of the inner loop.

    Set ℕ                                          [the track the stored route keeps] -/
def innerSet : Set ℕ := {p | ∃ v, p = innPos v}

/-- An outer position is on the outer track.

    (u : ℕ)                                        [a position]
    outPos u ∈ outerSet                            [its outer copy is on that track] -/
theorem mem_outerSet (u : ℕ) : outPos u ∈ outerSet := ⟨u, rfl⟩

/-- An inner position is on the inner track.

    (v : ℕ)                                        [a position]
    innPos v ∈ innerSet                            [its inner copy is on that track] -/
theorem mem_innerSet (v : ℕ) : innPos v ∈ innerSet := ⟨v, rfl⟩

/-- Every position lies on exactly one of the three tracks.

    (p : ℕ)                                        [any position of the wire]
    (∃ k, p = extPos k) ∨ (∃ u, p = outPos u) ∨ (∃ v, p = innPos v)   [one of the three]

    So: the wire is split once into outside and loop, and the loop is split again into the two
    tracks. Nothing else is needed to lay a combinator's stored route out. -/
theorem pos_trichotomy (p : ℕ) :
    (∃ k, p = extPos k) ∨ (∃ u, p = outPos u) ∨ (∃ v, p = innPos v) := by
  rcases hs : Equiv.natSumNatEquivNat.symm p with k | q
  · left
    exact ⟨k, by rw [extPos, ← hs, Equiv.apply_symm_apply]⟩
  · rcases hq : Equiv.natSumNatEquivNat.symm q with u | v
    · right; left
      refine ⟨u, ?_⟩
      rw [outPos, ← hq, Equiv.apply_symm_apply, ← hs, Equiv.apply_symm_apply]
    · right; right
      refine ⟨v, ?_⟩
      rw [innPos, ← hq, Equiv.apply_symm_apply, ← hs, Equiv.apply_symm_apply]

/-- The two loop tracks do not meet.

    ∀ x, x ∈ outerSet → x ∉ innerSet               [nothing is on both tracks]

    So: the inner split is by the even and odd halves, and a number is not both. This is the
    disjointness `runs_vanishing` asks for. -/
theorem outer_disjoint_inner : ∀ x, x ∈ outerSet → x ∉ innerSet := by
  rintro x ⟨u, rfl⟩ ⟨v, hv⟩
  rw [outPos, innPos] at hv
  have h1 := Equiv.natSumNatEquivNat.injective hv
  injection h1 with h2
  have h3 := Equiv.natSumNatEquivNat.injective h2
  exact absurd h3 (by simp)

/-- An outside position is on neither loop track.

    (k : ℕ)                                        [a position]
    extPos k ∉ outerSet ∧ extPos k ∉ innerSet      [its outside copy is on neither] -/
theorem extPos_not_mem (k : ℕ) : extPos k ∉ outerSet ∧ extPos k ∉ innerSet := by
  constructor
  · rintro ⟨u, hu⟩
    rw [extPos, outPos] at hu
    exact absurd (Equiv.natSumNatEquivNat.injective hu) (by simp)
  · rintro ⟨v, hv⟩
    rw [extPos, innPos] at hv
    exact absurd (Equiv.natSumNatEquivNat.injective hv) (by simp)

/-- The two tracks together are the whole loop.

    oddPos = outerSet ∪ innerSet                   [the loop is the two tracks and nothing more]

    So: `Combinator/Vanishing.lean` traces over `oddPos`, and this says that tracing is tracing
    over the union of two disjoint sets, which is the shape `runs_vanishing` takes. -/
theorem oddPos_eq : oddPos = outerSet ∪ innerSet := by
  ext p
  constructor
  · rintro ⟨u, rfl⟩
    rcases hq : Equiv.natSumNatEquivNat.symm u with a | b
    · left
      exact ⟨a, by rw [outPos, ← hq, Equiv.apply_symm_apply]⟩
    · right
      exact ⟨b, by rw [innPos, ← hq, Equiv.apply_symm_apply]⟩
  · rintro (⟨u, rfl⟩ | ⟨v, rfl⟩)
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE FLATTENING
-- ═══════════════════════════════════════════════════════════════════

/-- **The trace of a route is the trace of its inner-collapsed form over the outer track.**

@cite joyal-street-verity-1996
@cite girard-1989

    (σ : ℝ) (f : Route)                            [the parameter and a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ f i j w                               [the trace over the whole loop]
    ↔                                              [exactly when]
    Runs σ (runsRoute σ f innerSet) outerSet       [collapse the inner track first]
      (extPos i) (extPos j) w                      [then run along the outer one]

    So: a combinator may store a route which is itself a trace, and its equation may still be
    checked, because the inner loop and the outer one can be traced in either order. That is what
    every combinator which composes its arguments needs, and `Combinator/Duplication.lean` shows
    all of them do. -/
theorem traces_flatten (σ : ℝ) (f : Route) (i j : ℕ) (w : Cell) :
    Traces σ f i j w ↔
      Runs σ (runsRoute σ f innerSet) outerSet (extPos i) (extPos j) w := by
  rw [traces_iff_runs, oddPos_eq]
  exact runs_vanishing outer_disjoint_inner _ _ _

/-- Arbitrary relational paths over both loop tracks factor through inner-track paths.

    RelRuns  [an unbounded finite path]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ        [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ        [the counting numbers 0, 1, 2, 3, and so on]
    outerSet [the positions of the loop that faces the argument]
    innerSet [the positions of the loop a stored route keeps to itself]
    extPos   [a position of the wire outside both loops]
    j        [the special direction that squares to give 1, not −1 the way i does]
    ↔        [exactly when, true in both directions]
    ∪        [the union of two sets] -/
theorem relRuns_flatten {R : Type u} [CommRing R] (σ : R)
    (g : RelationalPaths.StepRel ℕ R) (i j : ℕ) (w : Trichotomy.AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ g (outerSet ∪ innerSet) (extPos i) (extPos j) w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.RelRuns σ g innerSet) outerSet
        (extPos i) (extPos j) w := by
  exact RelationalPaths.relRuns_vanishing (R := R) (σ := σ) (g := g)
    (S₁ := outerSet) (S₂ := innerSet) outer_disjoint_inner _ _ _

/-- A generic route trace factors through its inner-track relational paths.

    Traces   [the union over every path length through a route]
    RelRuns  [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    Route    [a weighted wire map: each position goes nowhere, or to one with one weight]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ        [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ        [the counting numbers 0, 1, 2, 3, and so on]
    outerSet [the positions of the loop that faces the argument]
    innerSet [the positions of the loop a stored route keeps to itself]
    extPos   [a position of the wire outside both loops]
    j        [the special direction that squares to give 1, not −1 the way i does]
    ↔        [exactly when, true in both directions] -/
theorem relTraces_flatten {R : Type u} [CommRing R] (σ : R)
    (f : RouteAnyRing.Route R) (i j : ℕ) (w : Trichotomy.AnyRing.RingCell R) :
    RouteAnyRing.Traces σ f i j w ↔
      RelationalPaths.RelRuns σ
        (RelationalPaths.RelRuns σ (RelationalPaths.routeGraph f) innerSet) outerSet
        (extPos i) (extPos j) w := by
  have hodd : RelationalPaths.oddPos = outerSet ∪ innerSet := by
    change oddPos = outerSet ∪ innerSet
    exact oddPos_eq
  rw [RelationalPaths.traces_iff_relRuns, hodd]
  exact RelationalPaths.relRuns_vanishing (R := R) (σ := σ)
    (g := RelationalPaths.routeGraph f) (S₁ := outerSet) (S₂ := innerSet)
    outer_disjoint_inner _ _ _

/-- A concrete integer route enters the inner track, then the outer track, before exiting.

    Route Int [a weighted wire map over the integer ring] -/
private def innerOuterRoute : RouteAnyRing.Route Int
  | 0 => some (3, (1, 1))
  | 3 => some (1, (2, 3))
  | 1 => some (2, (3, 1))
  | _ => none

/-- Inner and outer relational collapse retains the integer route's non-unit weight.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    outerSet [the positions of the loop that faces the argument]
    innerSet [the positions of the loop a stored route keeps to itself]
    extPos [a position of the wire outside both loops] -/
theorem relTraces_flatten_nonUnit :
    RelationalPaths.RelRuns 2
      (RelationalPaths.RelRuns 2 (RelationalPaths.routeGraph innerOuterRoute) innerSet) outerSet
      (extPos 0) (extPos 1) (34, 23) := by
  have hinter : RelationalPaths.RelRuns 2 (RelationalPaths.routeGraph innerOuterRoute) innerSet
      (extPos 0) (outPos 0) (8, 5) := by
    have h : RelationalPaths.RelRunsN 2 (RelationalPaths.routeGraph innerOuterRoute) innerSet 1
        (extPos 0) (outPos 0) (Trichotomy.AnyRing.ringMul 2 (1, 1) (2, 3)) := by
      exact .step (by change innerOuterRoute 0 = some (3, (1, 1)); rfl) (mem_innerSet 0)
        (.exit (by change innerOuterRoute 3 = some (1, (2, 3)); rfl)
          (outer_disjoint_inner _ (mem_outerSet 0)))
    norm_num [Trichotomy.AnyRing.ringMul] at h
    exact ⟨1, h⟩
  have houter : RelationalPaths.RelRuns 2 (RelationalPaths.routeGraph innerOuterRoute) innerSet
      (outPos 0) (extPos 1) (3, 1) := by
    exact ⟨0, .exit (by change innerOuterRoute 1 = some (2, (3, 1)); rfl)
      (extPos_not_mem 1).2⟩
  have h : RelationalPaths.RelRunsN 2
      (RelationalPaths.RelRuns 2 (RelationalPaths.routeGraph innerOuterRoute) innerSet) outerSet 1
      (extPos 0) (extPos 1) (Trichotomy.AnyRing.ringMul 2 (8, 5) (3, 1)) := by
    exact .step hinter (mem_outerSet 0) (.exit houter (extPos_not_mem 1).1)
  norm_num [Trichotomy.AnyRing.ringMul] at h
  exact ⟨1, h⟩

-- ═══════════════════════════════════════════════════════════════════
-- COLLAPSING MORE THAN ONE LOOP
-- ═══════════════════════════════════════════════════════════════════

/-- Routes with the same paths collapse to the same route.

    {σ : ℝ} {f g : Route} {S T : Set ℕ}            [the parameter, two routes, two sets]
    (h : ∀ i j w, Runs σ f S i j w ↔ Runs σ g T i j w)           [the same paths]
    runsRoute σ f S = runsRoute σ g T                            [so the same collapse] -/
theorem runsRoute_congr {σ : ℝ} {f g : Route} {S T : Set ℕ}
    (h : ∀ i j w, Runs σ f S i j w ↔ Runs σ g T i j w) :
    runsRoute σ f S = runsRoute σ g T := by
  funext i
  cases hr : runsRoute σ f S i with
  | none =>
    cases hr2 : runsRoute σ g T i with
    | none => rfl
    | some p =>
      have hp : runsRoute σ g T i = some (p.1, p.2) := by simp [hr2]
      have hrun := (h i p.1 p.2).mpr ((runsRoute_eq σ g T i p.1 p.2).mp hp)
      rw [(runsRoute_eq σ f S i p.1 p.2).mpr hrun] at hr
      exact absurd hr (by simp)
  | some p =>
    have hp : runsRoute σ f S i = some (p.1, p.2) := by simp [hr]
    have hrun := (h i p.1 p.2).mp ((runsRoute_eq σ f S i p.1 p.2).mp hp)
    rw [(runsRoute_eq σ g T i p.1 p.2).mpr hrun]

/-- Collapsing two loops at once is collapsing them one at a time.

    {σ : ℝ} {f : Route} {S₂ S₃ : Set ℕ}            [the parameter, a route, two loops]
    (h23 : ∀ x, x ∈ S₂ → x ∉ S₃)                                 [which do not meet]
    runsRoute σ f (S₂ ∪ S₃) = runsRoute σ (runsRoute σ f S₃) S₂  [either order gives the same] -/
theorem runsRoute_union {σ : ℝ} {f : Route} {S₂ S₃ : Set ℕ} (h23 : ∀ x, x ∈ S₂ → x ∉ S₃) :
    runsRoute σ f (S₂ ∪ S₃) = runsRoute σ (runsRoute σ f S₃) S₂ :=
  runsRoute_congr (runs_vanishing h23)

/-- **Vanishing over three loops.**

@cite joyal-street-verity-1996

    {σ : ℝ} {f : Route} {S₁ S₂ S₃ : Set ℕ}         [the parameter, a route, three loops]
    (h12 : ∀ x, x ∈ S₁ → x ∉ S₂)                                 [pairwise]
    (h13 : ∀ x, x ∈ S₁ → x ∉ S₃)                                 [they do]
    (h23 : ∀ x, x ∈ S₂ → x ∉ S₃)                                 [not meet]
    (i j : ℕ) (w : Cell)                                         [two positions and a weight]
    Runs σ f (S₁ ∪ (S₂ ∪ S₃)) i j w                              [a path through all three]
    ↔                                                            [exactly when]
    Runs σ (runsRoute σ (runsRoute σ f S₃) S₂) S₁ i j w          [a path through them in turn]

    So: `runs_vanishing` twice. This is the shape the right-hand side of `S x y z = x z (y z)`
    takes, because that side carries three loops: the one by which `x` consults its argument, the
    one by which `y` consults its argument, and the one by which `x z` is applied to `y z`. -/
theorem runs_vanishing_three {σ : ℝ} {f : Route} {S₁ S₂ S₃ : Set ℕ}
    (h12 : ∀ x, x ∈ S₁ → x ∉ S₂) (h13 : ∀ x, x ∈ S₁ → x ∉ S₃)
    (h23 : ∀ x, x ∈ S₂ → x ∉ S₃) (i j : ℕ) (w : Cell) :
    Runs σ f (S₁ ∪ (S₂ ∪ S₃)) i j w ↔
      Runs σ (runsRoute σ (runsRoute σ f S₃) S₂) S₁ i j w := by
  rw [runs_vanishing (fun x hx hc => hc.elim (h12 x hx) (h13 x hx)) i j w, runsRoute_union h23]

/-- Relational paths through three pairwise-disjoint loops factor in sequence.

    RelRuns  [an unbounded finite path]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ        [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ        [the counting numbers 0, 1, 2, 3, and so on]
    j        [the special direction that squares to give 1, not −1 the way i does]
    ∀        [for every single one]
    ∈        [is one of the things in]
    →        [takes the left and gives the right; between statements, left implies right]
    ∉        [is not one of the things in]
    ↔        [exactly when, true in both directions]
    ∪        [the union of two sets] -/
theorem relRuns_vanishing_three {R : Type u} [CommRing R] {σ : R}
    {g : RelationalPaths.StepRel ℕ R} {S₁ S₂ S₃ : Set ℕ}
    (h12 : ∀ x, x ∈ S₁ → x ∉ S₂) (h13 : ∀ x, x ∈ S₁ → x ∉ S₃)
    (h23 : ∀ x, x ∈ S₂ → x ∉ S₃) (i j : ℕ) (w : Trichotomy.AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ g (S₁ ∪ (S₂ ∪ S₃)) i j w ↔
      RelationalPaths.RelRuns σ
        (RelationalPaths.RelRuns σ (RelationalPaths.RelRuns σ g S₃) S₂) S₁ i j w := by
  rw [RelationalPaths.relRuns_vanishing
    (fun x hx hc => hc.elim (h12 x hx) (h13 x hx)) i j w]
  have h23' : RelationalPaths.RelRuns σ g (S₂ ∪ S₃) =
      RelationalPaths.RelRuns σ (RelationalPaths.RelRuns σ g S₃) S₂ := by
    funext i j w
    apply propext
    exact RelationalPaths.relRuns_vanishing h23 i j w
  rw [h23']

-- ═══════════════════════════════════════════════════════════════════
-- RELABELLING THE WIRE
-- ═══════════════════════════════════════════════════════════════════

/-- A route read through a relabelling of the wire.

@cite girard-1989

    (e : ℕ ≃ ℕ) (f : Route)                        [a relabelling and a route]
    fun i => (f (e i)).map fun q => (e.symm q.1, q.2)   [relabel in, run f, relabel out]
    : Route                                        [the route seen in the new labels]

    So: a combinator stores its argument with the wires laid out differently from how the
    argument itself uses them, and this is that change of layout. Every combinator that composes
    needs one, because the tracks its stored route needs are not the tracks its arguments came
    with. -/
def conjRoute {R : Type u} [CommRing R] (e : ℕ ≃ ℕ) (f : RouteAnyRing.Route R) :
    RouteAnyRing.Route R :=
  fun i => (f (e i)).map fun q => (e.symm q.1, q.2)

/-- A conjugated route edge is exactly an original edge between corresponding vertices.

    routeGraph [the transition relation presented by a partial weighted route]
    conjRoute [a route read through a relabelling of the wire]
    RingCell [a point: two ring elements side by side, a + b·x]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    ∃ [there is at least one]
    ∧ [both statements hold]
    = [is equal to]
    ↔ [exactly when, true in both directions] -/
private theorem routeGraph_conj_edge {R : Type u} [CommRing R] (e : ℕ ≃ ℕ)
    (f : RouteAnyRing.Route R) (a c : ℕ) (w : Trichotomy.AnyRing.RingCell R) :
    RelationalPaths.routeGraph f (e a) c w ↔
      ∃ b, c = e b ∧ RelationalPaths.routeGraph (conjRoute e f) a b w := by
  change f (e a) = some (c, w) ↔
    ∃ b, c = e b ∧ conjRoute e f a = some (b, w)
  constructor
  · intro h
    refine ⟨e.symm c, (e.apply_symm_apply c).symm, ?_⟩
    rw [conjRoute, h, Option.map_some]
  · rintro ⟨b, rfl, h⟩
    simp only [conjRoute, Option.map_eq_some_iff] at h
    obtain ⟨⟨d, u⟩, hfd, hq⟩ := h
    injection hq with hd hu
    subst hu
    have hdb : d = e b := by
      calc
        d = e (e.symm d) := (e.apply_symm_apply d).symm
        _ = e b := congrArg e hd
    rw [hdb] at hfd
    exact hfd

/-- A generic route trace transports to a relational path through the relabelled loop.

    Traces [the union over every path length through a route]
    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    conjRoute [a route read through a relabelling of the wire]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to, below zero, zero, or above]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    extPos [a position of the wire outside both loops]
    ↔ [exactly when, true in both directions]
    j [the special direction that squares to give 1, not −1 the way i does] -/
theorem relTraces_conj {R : Type u} [CommRing R] (σ : R) (e : ℕ ≃ ℕ)
    (f : RouteAnyRing.Route R) (i j : ℕ) (w : Trichotomy.AnyRing.RingCell R) :
    RouteAnyRing.Traces σ (conjRoute e f) i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph f) (e '' RelationalPaths.oddPos)
        (e (extPos i)) (e (extPos j)) w := by
  rw [RelationalPaths.traces_iff_relRuns]
  change RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (conjRoute e f))
    RelationalPaths.oddPos (extPos i) (extPos j) w ↔ _
  exact (RelationalPaths.relRuns_embed_iff e.injective (routeGraph_conj_edge e f)
    (S := RelationalPaths.oddPos)).symm

/-- A path of a relabelled route is a path of the route.

    {σ : ℝ} {e : ℕ ≃ ℕ} {f : Route} {T : Set ℕ}    [parameter, relabelling, route, positions]
    ∀ {n i j : ℕ} {w : Cell}                       [a length, two positions, a weight]
    RunsN σ (conjRoute e f) (e ⁻¹' T) n i j w      [a path in the new labels]
    →                                              [gives]
    RunsN σ f T n (e i) (e j) w                    [the same path in the old ones] -/
theorem runsN_conj_fwd {σ : ℝ} {e : ℕ ≃ ℕ} {f : Route} {T : Set ℕ} :
    ∀ {n i j : ℕ} {w : Cell}, RunsN σ (conjRoute e f) (e ⁻¹' T) n i j w →
      RunsN σ f T n (e i) (e j) w := by
  intro n i j w h
  induction h with
  | @exit i j w hf hns =>
    refine RunsN.exit ?_ hns
    simp only [conjRoute, Option.map_eq_some_iff] at hf
    obtain ⟨⟨p1, p2⟩, hp, hq⟩ := hf
    injection hq with h1 h2
    subst h2
    rw [← h1, Equiv.apply_symm_apply]
    exact hp
  | @step n i k j w v hf hs hr ih =>
    refine RunsN.step ?_ hs ih
    simp only [conjRoute, Option.map_eq_some_iff] at hf
    obtain ⟨⟨p1, p2⟩, hp, hq⟩ := hf
    injection hq with h1 h2
    subst h2
    rw [← h1, Equiv.apply_symm_apply]
    exact hp

/-- A path of a route is a path of its relabelling.

    {σ : ℝ} {e : ℕ ≃ ℕ} {f : Route} {T : Set ℕ}    [parameter, relabelling, route, positions]
    ∀ {n a b : ℕ} {w : Cell}                       [a length, two positions, a weight]
    RunsN σ f T n a b w                            [a path in the old labels]
    →                                              [gives]
    RunsN σ (conjRoute e f) (e ⁻¹' T) n (e.symm a) (e.symm b) w   [the same in the new ones] -/
theorem runsN_conj_bwd {σ : ℝ} {e : ℕ ≃ ℕ} {f : Route} {T : Set ℕ} :
    ∀ {n a b : ℕ} {w : Cell}, RunsN σ f T n a b w →
      RunsN σ (conjRoute e f) (e ⁻¹' T) n (e.symm a) (e.symm b) w := by
  intro n a b w h
  induction h with
  | @exit a b w hf hns =>
    refine RunsN.exit ?_ ?_
    · simp [conjRoute, Equiv.apply_symm_apply, hf]
    · simpa [Set.mem_preimage, Equiv.apply_symm_apply] using hns
  | @step n a k b w v hf hs hr ih =>
    refine RunsN.step ?_ ?_ ih
    · simp [conjRoute, Equiv.apply_symm_apply, hf]
    · simpa [Set.mem_preimage, Equiv.apply_symm_apply] using hs

/-- **Relabelling the wire does not change the paths.**

@cite girard-1989

    {σ : ℝ} (e : ℕ ≃ ℕ) (f : Route) (T : Set ℕ)    [parameter, relabelling, route, positions]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Runs σ (conjRoute e f) (e ⁻¹' T) i j w         [a path in the new labels]
    ↔                                              [exactly when]
    Runs σ f T (e i) (e j) w                       [the same path in the old ones]

    So: a combinator may lay its stored route out on whatever tracks it needs, and the paths are
    the argument's own. With `runs_vanishing` and `traces_flatten` this is the last general tool
    a composing combinator asks for: relabel, flatten, and the two sides of its equation meet. -/
theorem runs_conj {σ : ℝ} (e : ℕ ≃ ℕ) (f : Route) (T : Set ℕ) (i j : ℕ) (w : Cell) :
    Runs σ (conjRoute e f) (e ⁻¹' T) i j w ↔ Runs σ f T (e i) (e j) w := by
  constructor
  · rintro ⟨n, h⟩
    exact ⟨n, runsN_conj_fwd h⟩
  · rintro ⟨n, h⟩
    refine ⟨n, ?_⟩
    have := runsN_conj_bwd (e := e) (T := T) h
    rwa [Equiv.symm_apply_apply, Equiv.symm_apply_apply] at this

/-- Relabelling twice by inverse relabellings changes nothing.

    (e : ℕ ≃ ℕ) (f : Route)                        [a relabelling and a route]
    conjRoute e (conjRoute e.symm f) = f           [the round trip is the identity] -/
theorem conjRoute_symm (e : ℕ ≃ ℕ) (f : Route) : conjRoute e (conjRoute e.symm f) = f := by
  funext i
  cases hf : f i with
  | none => simp [conjRoute, hf]
  | some p => simp [conjRoute, hf]

/-- **Relabelling carries composition.**

    (σ : R) (e : ℕ ≃ ℕ) (a b : Route R)            [parameter, relabelling, two routes]
    conjRoute e (rcomp σ a b)                                    [relabel the composite]
    = rcomp σ (conjRoute e a) (conjRoute e b)                    [is compose the relabellings]

    So: `conjRoute` is a functor on the monoid `rcomp` makes, as `bang` is. That is what lets a
    combinator's stored route be relabelled and plugged in either order, which every proof about
    a composing combinator needs before `traces_conj` can be applied to it. -/
theorem conjRoute_rcomp {R : Type u} [CommRing R] (σ : R) (e : ℕ ≃ ℕ)
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

/-- A path through a conjugated route transports to the original route under the relabelling.

    RelRuns [an unbounded finite path]
    routeGraph [the transition relation presented by a partial weighted route]
    conjRoute [a route read through a relabelling of the wire]
    RingCell [a point: two ring elements side by side, a + b·x]
    σ [the parameter: what the adjoined direction squares to]
    R [a commutative coefficient ring]
    ℕ [the counting numbers 0, 1, 2, 3, and so on]
    '' [the image of a set under a map]
    ↔ [exactly when, true in both directions]
    e [a relabelling of wire positions]
    f [the route before relabelling]
    S [the positions that may be visited before exit]
    i [the source position]
    j [the final position]
    w [the accumulated path weight] -/
theorem relRuns_conj {R : Type u} [CommRing R] (σ : R) (e : ℕ ≃ ℕ)
    (f : RouteAnyRing.Route R) (S : Set ℕ) (i j : ℕ) (w : Trichotomy.AnyRing.RingCell R) :
    RelationalPaths.RelRuns σ (RelationalPaths.routeGraph (conjRoute e f)) S i j w ↔
      RelationalPaths.RelRuns σ (RelationalPaths.routeGraph f) (e '' S) (e i) (e j) w :=
  (RelationalPaths.relRuns_embed_iff e.injective (routeGraph_conj_edge e f) (S := S)).symm

/-- **The trace of a relabelled route, in the argument's own labels.**

@cite girard-1989
@cite joyal-street-verity-1996

    (σ : ℝ) (e : ℕ ≃ ℕ) (f : Route)                [the parameter, a relabelling, a route]
    (i j : ℕ) (w : Cell)                           [two positions and a weight]
    Traces σ (conjRoute e f) i j w                 [the trace of the relabelled route]
    ↔                                              [exactly when]
    Runs σ f (e '' oddPos) (e (extPos i)) (e (extPos j)) w   [a path of the route itself]

    So: this is where a proof about a composing combinator starts. The combinator stores its
    argument on tracks of its own choosing, this puts the resulting trace back in the argument's
    labels, and `traces_flatten` with `runs_vanishing_three` then collapse whatever loops the
    layout has left. Those three, with `Combinator/Duplication.lean`'s `bang_restrict` for
    arguments used twice, are the whole of the general machinery. -/
theorem traces_conj (σ : ℝ) (e : ℕ ≃ ℕ) (f : Route) (i j : ℕ) (w : Cell) :
    Traces σ (conjRoute e f) i j w ↔
      Runs σ f (e '' oddPos) (e (extPos i)) (e (extPos j)) w := by
  rw [traces_iff_runs]
  have h := runs_conj (σ := σ) e f (e '' oddPos) (extPos i) (extPos j) w
  rwa [Equiv.preimage_image] at h

-- ═══════════════════════════════════════════════════════════════════
-- PUSHING A WIRE ONE LEVEL OUT
-- ═══════════════════════════════════════════════════════════════════

/-- A path of the wider route, started at an embedded position, is a path of the narrower one.

    {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)      [the parameter and an embedding]
    {f g : Route}                                        [a wide route and a narrow one]
    (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))   [agreeing on the image]
    ∀ {n a b : ℕ} {w : Cell}                             [a length, two positions, a weight]
    RunsN σ f (ι '' S) n a b w → ∀ i, a = ι i →          [a wide path from an embedded start]
      ∃ j, b = ι j ∧ RunsN σ g S n i j w                 [is a narrow one, ending embedded] -/
theorem runsN_embed_fwd {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)
    {f g : Route} (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2)) {S : Set ℕ} :
    ∀ {n a b : ℕ} {w : Cell}, RunsN σ f (ι '' S) n a b w → ∀ i, a = ι i →
      ∃ j, b = ι j ∧ RunsN σ g S n i j w := by
  intro n a b w h
  induction h with
  | @exit a b w hf hns =>
    intro i hai
    subst hai
    rw [hfg i] at hf
    obtain ⟨⟨c1, c2⟩, hg, hq⟩ := Option.map_eq_some_iff.mp hf
    injection hq with h1 h2
    subst h1
    subst h2
    exact ⟨c1, rfl, RunsN.exit hg fun hc => hns ⟨c1, hc, rfl⟩⟩
  | @step n a k b w v hf hs hr ih =>
    intro i hai
    subst hai
    rw [hfg i] at hf
    obtain ⟨⟨c1, c2⟩, hg, hq⟩ := Option.map_eq_some_iff.mp hf
    injection hq with h1 h2
    subst h1
    subst h2
    obtain ⟨t, htS, hti⟩ := hs
    have htc : t = c1 := hι hti
    subst htc
    obtain ⟨j, hbj, hrun⟩ := ih t rfl
    exact ⟨j, hbj, RunsN.step hg htS hrun⟩

/-- A path of the narrower route is a path of the wider one.

    {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)      [the parameter and an embedding]
    {f g : Route}                                        [a wide route and a narrow one]
    (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))   [agreeing on the image]
    ∀ {n i j : ℕ} {w : Cell}                             [a length, two positions, a weight]
    RunsN σ g S n i j w                                  [a narrow path]
    →                                                    [gives]
    RunsN σ f (ι '' S) n (ι i) (ι j) w                   [the same path, embedded] -/
theorem runsN_embed_bwd {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)
    {f g : Route} (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2)) {S : Set ℕ} :
    ∀ {n i j : ℕ} {w : Cell}, RunsN σ g S n i j w → RunsN σ f (ι '' S) n (ι i) (ι j) w := by
  intro n i j w h
  induction h with
  | @exit i j w hg hns =>
    refine RunsN.exit ?_ ?_
    · rw [hfg i, hg]; rfl
    · rintro ⟨t, htS, htj⟩
      exact hns (hι htj ▸ htS)
  | @step n i k j w v hg hs hr ih =>
    refine RunsN.step ?_ ⟨k, hs, rfl⟩ ih
    rw [hfg i, hg]; rfl

/-- **A path relation transports along an embedding.**

@cite girard-1989

    {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)      [the parameter and an embedding]
    {f g : Route}                                        [a wide route and a narrow one]
    (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2))   [agreeing on the image]
    (i j : ℕ) (w : Cell)                                 [two positions and a weight]
    Runs σ f (ι '' S) (ι i) (ι j) w                      [a wide path between embedded points]
    ↔                                                    [exactly when]
    Runs σ g S i j w                                     [a narrow path between the points]

    So: `runs_conj` does this for a relabelling of the whole wire, and this does it for a wire
    pushed one level out. That is the shape a combinator's second use of an argument takes,
    since it lives on the first use embedded once, and it will not be the last place that
    happens. Only the forward direction needs the embedding to be injective, which is where the
    inverse would have gone. -/
theorem runs_embed {σ : ℝ} {ι : ℕ → ℕ} (hι : Function.Injective ι)
    {f g : Route} (hfg : ∀ a, f (ι a) = (g a).map fun q => (ι q.1, q.2)) {S : Set ℕ}
    (i j : ℕ) (w : Cell) :
    Runs σ f (ι '' S) (ι i) (ι j) w ↔ Runs σ g S i j w := by
  constructor
  · rintro ⟨n, h⟩
    obtain ⟨j', hj', hrun⟩ := runsN_embed_fwd hι hfg h i rfl
    have hjj : j = j' := hι hj'
    subst hjj
    exact ⟨n, hrun⟩
  · rintro ⟨n, h⟩
    exact ⟨n, runsN_embed_bwd hι hfg h⟩

/-- The outside embedding is injective.

    Function.Injective extPos                      [distinct positions stay distinct] -/
theorem extPos_injective : Function.Injective extPos := by
  intro a b h
  rw [extPos, extPos] at h
  exact Sum.inl.inj (Equiv.natSumNatEquivNat.injective h)

/-- A path ends outside the set it stayed in.

    Route R, σ : R [a weighted route and its ring parameter]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    Runs    [a path through a route, staying inside a set until it leaves]
    RingCell [a point: two ring elements side by side, a + b·x]
    j ∉ S   [the path ends outside its interior set] -/
theorem runs_end_not_mem {R : Type u} [CommRing R] {σ : R} {f : RouteAnyRing.Route R}
    {S : Set ℕ} {i j : ℕ} {w : Trichotomy.AnyRing.RingCell R}
    (h : RouteAnyRing.Runs σ f S i j w) : j ∉ S := by
  obtain ⟨n, h⟩ := h
  induction h with
  | exit _ hns => exact hns
  | step _ _ _ ih => exact ih

-- ═══════════════════════════════════════════════════════════════════
-- A ROUTE THAT ONLY ACTS WHEN THE PATH LEAVES
-- ═══════════════════════════════════════════════════════════════════

/-- What one step of a composite is.

    Route R, σ : R [a weighted route and its ring parameter]
    ℕ       [the counting numbers 0, 1, 2, 3, and so on]
    RingCell [a point: two ring elements side by side, a + b·x]
    rcomp σ a b i = some (j, w)                    [the composite lands there]
    ↔                                              [exactly when]
    ∃ m w₁ w₂, a i = some (m, w₁) ∧                [each leg lands]
      b m = some (j, w₂) ∧ w = ringMul σ w₁ w₂     [and the weights multiply] -/
theorem rcomp_eq_some {R : Type u} [CommRing R] {σ : R}
    {a b : RouteAnyRing.Route R} {i j : ℕ} {w : Trichotomy.AnyRing.RingCell R} :
    RouteAnyRing.rcomp σ a b i = some (j, w) ↔
      ∃ m w₁ w₂, a i = some (m, w₁) ∧ b m = some (j, w₂) ∧
        w = Trichotomy.AnyRing.ringMul σ w₁ w₂ := by
  constructor
  · intro h
    cases ha : a i with
    | none => rw [RouteAnyRing.rcomp, ha] at h; simp at h
    | some p =>
      obtain ⟨p1, p2⟩ := p
      rw [RouteAnyRing.rcomp, ha] at h
      simp only [Option.bind_some] at h
      cases hb : b p1 with
      | none => rw [hb] at h; simp at h
      | some q =>
        obtain ⟨q1, q2⟩ := q
        rw [hb] at h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        exact ⟨p1, p2, q2, rfl, by rw [hb, h.1], h.2.symm⟩
  · rintro ⟨m, w₁, w₂, ha, hb, rfl⟩
    rw [RouteAnyRing.rcomp, ha, Option.bind_some, hb, Option.map_some]

/-- A composite path, cut at its last step.

    Route R, σ : R [a weighted route and its ring parameter]
    (hEid : ∀ s, s ∈ S → E s = some (s, ringOne))  [E does nothing on the loop]
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S)   [and cannot enter it]
    ∀ {n i c : ℕ} {w : RingCell R}                       [a length, two positions, a weight]
    RunsN σ (rcomp σ F E) S n i c w →                            [a path of the composite]
      ∃ m w₁ w₂, RunsN σ F S n i m w₁ ∧                          [is a path of the first leg]
        E m = some (c, w₂) ∧ w = ringMul σ w₁ w₂                 [then one step of the second] -/
theorem runsN_rcomp_id_fwd {R : Type u} [CommRing R] {σ : R}
    {F E : RouteAnyRing.Route R} {S : Set ℕ}
    (hEid : ∀ s, s ∈ S → E s = some (s, Trichotomy.AnyRing.ringOne))
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S) :
    ∀ {n i c : ℕ} {w : Trichotomy.AnyRing.RingCell R},
      RouteAnyRing.RunsN σ (RouteAnyRing.rcomp σ F E) S n i c w →
      ∃ m w₁ w₂, RouteAnyRing.RunsN σ F S n i m w₁ ∧ E m = some (c, w₂) ∧
        w = Trichotomy.AnyRing.ringMul σ w₁ w₂ := by
  intro n i c w h
  induction h with
  | @exit i c w hf hns =>
    obtain ⟨m, w₁, w₂, hF, hE, rfl⟩ := rcomp_eq_some.mp hf
    refine ⟨m, w₁, w₂, RouteAnyRing.RunsN.exit hF ?_, hE, rfl⟩
    intro hmS
    rw [hEid m hmS] at hE
    injection hE with hq
    injection hq with hq1 _
    exact hns (hq1 ▸ hmS)
  | @step n i k c w v hf hs hr ih =>
    obtain ⟨m, w₁, w₂, hF, hE, hw⟩ := rcomp_eq_some.mp hf
    have hmS : m ∈ S := hEout m k w₂ hE hs
    rw [hEid m hmS] at hE
    injection hE with hq
    injection hq with hq1 hq2
    subst hq1
    obtain ⟨m', u₁, u₂, hrun, hE', hv⟩ := ih
    refine ⟨m', Trichotomy.AnyRing.ringMul σ w₁ u₁, u₂,
      RouteAnyRing.RunsN.step hF hmS hrun, hE', ?_⟩
    rw [hw, hv, ← hq2, Trichotomy.AnyRing.ringMul_one,
      Trichotomy.AnyRing.ringMul_assoc]

/-- A path of the first leg, with one step of the second, is a path of the composite.

    Route R, σ : R [a weighted route and its ring parameter]
    (hEid : ∀ s, s ∈ S → E s = some (s, ringOne))  [E does nothing on the loop]
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S)   [and cannot enter it]
    ∀ {n i m : ℕ} {w₁ : RingCell R}, RunsN σ F S n i m w₁ →      [a path of the first leg]
      ∀ {c : ℕ} {w₂ : RingCell R}, E m = some (c, w₂) →                [and one step of the second]
        RunsN σ (rcomp σ F E) S n i c (ringMul σ w₁ w₂)          [is a path of the composite] -/
theorem runsN_rcomp_id_bwd {R : Type u} [CommRing R] {σ : R}
    {F E : RouteAnyRing.Route R} {S : Set ℕ}
    (hEid : ∀ s, s ∈ S → E s = some (s, Trichotomy.AnyRing.ringOne))
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S) :
    ∀ {n i m : ℕ} {w₁ : Trichotomy.AnyRing.RingCell R}, RouteAnyRing.RunsN σ F S n i m w₁ →
      ∀ {c : ℕ} {w₂ : Trichotomy.AnyRing.RingCell R}, E m = some (c, w₂) →
        RouteAnyRing.RunsN σ (RouteAnyRing.rcomp σ F E) S n i c
          (Trichotomy.AnyRing.ringMul σ w₁ w₂) := by
  intro n i m w₁ h
  induction h with
  | @exit i m w hF hns =>
    intro c w₂ hE
    refine RouteAnyRing.RunsN.exit (rcomp_eq_some.mpr ⟨m, w, w₂, hF, hE, rfl⟩) ?_
    intro hcS
    exact hns (hEout m c w₂ hE hcS)
  | @step n i k m w v hF hs hr ih =>
    intro c w₂ hE
    have hstep : RouteAnyRing.rcomp σ F E i = some
        (k, Trichotomy.AnyRing.ringMul σ w Trichotomy.AnyRing.ringOne) :=
      rcomp_eq_some.mpr ⟨k, w, Trichotomy.AnyRing.ringOne, hF, hEid k hs, rfl⟩
    have hrest := ih hE
    have hfin := RouteAnyRing.RunsN.step hstep hs hrest
    simpa only [Trichotomy.AnyRing.ringMul_one, Trichotomy.AnyRing.ringMul_assoc] using hfin

/-- **A route that does nothing on the loop only acts when the path leaves.**

@cite joyal-street-verity-1996

    Route R, σ : R [a weighted route and its ring parameter]
    (hEid : ∀ s, s ∈ S → E s = some (s, ringOne))  [E does nothing on the loop]
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S)   [and cannot enter it]
    (i c : ℕ) (w : RingCell R)                           [two positions and a weight]
    Runs σ (rcomp σ F E) S i c w                                 [a path of the composite]
    ↔                                                            [exactly when]
    ∃ m w₁ w₂, Runs σ F S i m w₁ ∧                               [a path of the first leg]
      E m = some (c, w₂) ∧ w = ringMul σ w₁ w₂                   [and one step of the second]

    So: post-composing with a route that leaves the loop alone changes only where a path comes
    out. That is exactly the difference between the two sides of a contraction equation, where
    one side applies the second copy of its argument a step earlier than the other. -/
theorem runs_rcomp_id_on {R : Type u} [CommRing R] {σ : R}
    {F E : RouteAnyRing.Route R} {S : Set ℕ}
    (hEid : ∀ s, s ∈ S → E s = some (s, Trichotomy.AnyRing.ringOne))
    (hEout : ∀ m j w, E m = some (j, w) → j ∈ S → m ∈ S) (i c : ℕ)
    (w : Trichotomy.AnyRing.RingCell R) :
    RouteAnyRing.Runs σ (RouteAnyRing.rcomp σ F E) S i c w ↔
      ∃ m w₁ w₂, RouteAnyRing.Runs σ F S i m w₁ ∧ E m = some (c, w₂) ∧
        w = Trichotomy.AnyRing.ringMul σ w₁ w₂ := by
  constructor
  · rintro ⟨n, h⟩
    obtain ⟨m, w₁, w₂, hrun, hE, hw⟩ := runsN_rcomp_id_fwd hEid hEout h
    exact ⟨m, w₁, w₂, ⟨n, hrun⟩, hE, hw⟩
  · rintro ⟨m, w₁, w₂, ⟨n, hrun⟩, hE, rfl⟩
    exact ⟨n, runsN_rcomp_id_bwd hEid hEout hrun hE⟩

end Combinator
end Apeiron
