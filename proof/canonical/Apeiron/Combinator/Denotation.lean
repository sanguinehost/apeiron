import Apeiron.Combinator.Linear.Basis
import Apeiron.Combinator.Confluence

/-!
# Terms as routes: a denotation of the calculus into the wires

@name Denotation
@cite abramsky-haghverdi-scott-2002
@cite girard-1989
@cite curry-1930
@cite schonfinkel-1924
@cite barendregt-1984
@cite engeler-1981
@cite vanoosten-2008
@grade proved

`Combinator/Basic.lean` has the term calculus, `Comb`, with its two rewriting rules.
`Combinator/Basis.lean` has the routes, weighted partial injections of the counting numbers,
with `rapp` for application, and proves that the nine combinators are elements of that carrier.
Until this file no declaration mentioned both carriers. This one sends a term to a route and
proves that rewriting does not move it.

## Root

| from | here |
|---|---|
| `rapp_kRoute`, `rapp_dRoute`, `bRoute_bracket` | `capp_kCombinator` |
| `wRoute_app`, `cRoute_app`, `fRoute_app`, `rapp_diggingRoute` | `capp_sCombinator` |
| `capp_kCombinator`, `capp_sCombinator` | `denote_step`, `denote_reduces` |

## Which application, and which combinators

The naive choice sends `app f x` to `rapp σ` of the two and sends `S` to `sRoute σ`. It does not
work, and the reason is exact rather than vague. `rapp` is the application of a **linear**
combinatory algebra: `rapp_kRoute` gives `K x y = x` for a bare `y`, but `sRoute_app` gives
`S x y !z` with the third argument banged, because the contraction inside `S` uses its argument
twice and a route has at most one wire out of each position. `linearDenote` below is that naive
denotation, `linearDenote_step_k` is the K rule holding for it unrestrictedly, and
`linearDenote_step_s_of_bang` is the S rule holding for it exactly when the argument is already
an exponential.

The asymmetry is the carrier speaking, and it says something precise about which structural rule
is free here.

- **Weakening is free.** `kRoute_reads_nothing_of_its_argument` says `K x y` does not depend on
  `y` at all: the second argument may be replaced by any other route, the empty one included.
  Discarding costs nothing because a route is a *partial* map, so a wire that goes nowhere is a
  value and not a failure. `rapp_rnone` is the same fact from the other side: the empty route
  absorbs.
- **Contraction is not free.** A route sends each position to at most one place, which is what
  `traces_functional` rests on, so there is no branching to duplicate a wire with. Two consumers
  need two disjoint sets of positions, and `bang` is what supplies them: `bang_serves_two` says
  the copies at distinct tags do not meet.

So the fix is not a restriction on terms. It is to use the application the exponential already
provides, `capp σ a b = rapp σ a (bang b)`, which `Combinator/Duplication.lean` defines and which
is Abramsky, Haghverdi and Scott's `x · y := x (!y)`. Under it the *combinators change too*:
`capp_kRoute` records that `kRoute` under `capp` returns `bang x` rather than `x`, so the
discarding combinator of the ordinary algebra is `B K D` and not `K`. That is `kCombinator`.
The substituting one is longer, because the copy that `S` hands to its second argument has to be
made reusable again before it is passed on:

```
   K_ca  =  B K D                             K_ca !x !y  =  K (D !x) !y  =  x
   S_ca  =  B (B (B W) G) D                   S_ca !x !y !z  =  (x !z) (F !y (δ !z))
```

with `G` built from `B` and `C` alone, in three named pieces below. The `F` and the `δ` are
where the second argument's result is promoted back to an exponential, and they are the exact
price of the duplication.

**Preservation then holds for every term, with no fragment.** `denote_step` is stated for
arbitrary `{a b : Comb}` with `Comb.Step a b` as its only hypothesis. There is no halting
obstruction to this and there never was: `Combinator/Model.lean` already denotes all of `Comb`
into Engeler graph sets with `meaning_reduces` unrestricted, and nothing follows from that
because equality in the target is not decidable. Equality of routes is equality of two functions
`ℕ → Option (ℕ × Cell)`, which is no more decidable, and divergence has a home inside the value:
`traceRoute` is total as an operation on routes, and a path that runs for ever leaves the route
undefined at that position rather than leaving the route undefined.

## What this decides, and what it does not

`denote_reduces` gives one direction and only one: convertible terms have equal routes. The
useful reading is the contrapositive, `not_join_of_denote_ne`: **two terms with different routes
are not convertible.** That is a certificate of DISTINCT, and it is available without
normalising, so it can speak where a fuel-bounded normaliser cannot.

The other direction is not proved and is not claimed. Route equality is a congruence on `Comb`
containing reduction, so it is never *finer* than convertibility; whether it is strictly coarser
is open, and `theory/DEAD_ENDS.md` records what was tried. Withholding it is deliberate: a
denotation that identified two non-convertible normal forms would be a real theorem and a
denotation that did not would be another, and neither is in hand.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- THE EMPTY ROUTE ABSORBS, AND THE DISCARDING ONE READS NOTHING
-- ═══════════════════════════════════════════════════════════════════

/-- **The empty route absorbs.** Applying it to anything gives the empty route.

    (σ : ℝ) (x : Route)                            [the parameter and a route]
    rapp σ rnone x = rnone                         [nothing applied to anything is nothing]

    So: this is where divergence lives on this carrier. `rapp` is total as an operation, and a
    position at which no path terminates is a position where the value is absent. A route being
    empty is a value, not a failure of the denotation to exist, which is why the denotation below
    needs no restriction on terms. -/
theorem rapp_rnone (σ : ℝ) (x : Route) : rapp σ rnone x = rnone := by
  have hplug : plug σ rnone x = rnone := by
    rw [plug, rcomp_rnone_left]
  refine rapp_eq ?_
  intro i j w
  rw [hplug]
  constructor
  · rintro (hd | ⟨n, hn⟩)
    · rw [direct, rcomp_rnone_left, rcomp_rnone_right] at hd
      simp [rnone] at hd
    · rw [entry, rcomp_rnone_left, rcomp_rnone_right, rcomp_rnone_left] at hn
      simp [rnone] at hn
  · intro h
    simp [rnone] at h

/-- **The discarding combinator reads nothing of its second argument.**

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x y y' : Route)                       [the parameter and three routes]
    rapp σ (rapp σ kRoute x) y                     [K applied to x and then to y]
    = rapp σ (rapp σ kRoute x) y'                  [is the same for any other second argument]

    So: weakening is free on this carrier, and this is why. The routing of `K` puts no wire into
    the track its second argument sits on, so replacing that argument by any route at all,
    `rnone` included, changes nothing. That is what a partial map buys, and it is the exact
    counterpart of the fact that contraction is *not* free: a route has at most one wire out of
    each position, so nothing here can branch, and two consumers of one argument need the
    disjoint copies `bang` supplies. -/
theorem kRoute_reads_nothing_of_its_argument (σ : ℝ) (x y y' : Route) :
    rapp σ (rapp σ kRoute x) y = rapp σ (rapp σ kRoute x) y' := by
  rw [rapp_kRoute, rapp_kRoute]

-- ═══════════════════════════════════════════════════════════════════
-- THE COMBINATORS OF THE ORDINARY ALGEBRA
-- ═══════════════════════════════════════════════════════════════════

/-- The discarding combinator of the ordinary algebra: `B K D`.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ)                                        [the parameter]
    Route                                          [a weighted wire map]

    So: `capp_kRoute` records that the linear `K` under `capp` hands back `bang x` rather than
    `x`. Composing a dereliction in front of it strips that exponential, which is why the
    ordinary discarding combinator is this and not `kRoute`. -/
def kCombinator (σ : ℝ) : Route := rapp σ (rapp σ bRoute kRoute) dRoute

/-- The piece of the substituting combinator that makes the copy reusable again: `B (C B δ) F`.

@cite abramsky-haghverdi-scott-2002
@cite girard-1989

    (σ : ℝ)                                        [the parameter]
    Route                                          [a weighted wire map]

    So: `sPromote_app` says this route takes `!y` and `!z` to `!(y !z)`. Functoriality supplies
    the exponential of an application and the comultiplication supplies the second bang the
    functoriality asks for. This is Girard's promotion, written as an element. -/
def sPromote (σ : ℝ) : Route :=
  rapp σ (rapp σ bRoute (rapp σ (rapp σ cRoute bRoute) diggingRoute)) fRoute

/-- The piece that holds the promotion aside while the first argument is served: `B (C B) P`.

    (σ : ℝ)                                        [the parameter]
    Route                                          [a weighted wire map] -/
def sShare (σ : ℝ) : Route :=
  rapp σ (rapp σ bRoute (rapp σ cRoute bRoute)) (sPromote σ)

/-- The piece that hands one copy to each half: `B (C B M) (C B)`.

    (σ : ℝ)                                        [the parameter]
    Route                                          [a weighted wire map]

    So: `sSplit_app` says this takes four arguments and gives `(a c) (P b d)`, with the third and
    the fourth still separate. The contraction that follows is what puts the same route in both
    places. -/
def sSplit (σ : ℝ) : Route :=
  rapp σ (rapp σ bRoute (rapp σ (rapp σ cRoute bRoute) (sShare σ))) (rapp σ cRoute bRoute)

/-- The substituting combinator of the ordinary algebra: `B (B (B W) G) D`.

@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924
@cite curry-1930

    (σ : ℝ)                                        [the parameter]
    Route                                          [a weighted wire map]

    So: the dereliction on the outside strips the exponential from the first argument, which the
    ordinary application put there and which the left branch does not want; the contraction on
    the inside is the one place the third argument is used twice, and it is legal because that
    argument arrives banged. -/
def sCombinator (σ : ℝ) : Route :=
  rapp σ (rapp σ bRoute (rapp σ (rapp σ bRoute (rapp σ bRoute wRoute)) (sSplit σ))) dRoute

/-- **Promotion.** The promoting piece takes two exponentials to the exponential of the
    application.

@cite abramsky-haghverdi-scott-2002
@cite girard-1989

    (σ : ℝ) (y z : Route)                          [the parameter and two routes]
    rapp σ (rapp σ (sPromote σ) (bang y)) (bang z) [the piece, applied twice]
    = bang (rapp σ y (bang z))                     [is the reusable copy of the application] -/
theorem sPromote_app (σ : ℝ) (y z : Route) :
    rapp σ (rapp σ (sPromote σ) (bang y)) (bang z) = bang (rapp σ y (bang z)) := by
  rw [sPromote, bRoute_bracket, cRoute_app, bRoute_bracket, rapp_diggingRoute, fRoute_app]

/-- **The split.** Four arguments in, the first pair and the second pair served separately.

    (σ : ℝ) (a b c d : Route)                              [the parameter and four routes]
    rapp σ (rapp σ (rapp σ (rapp σ (sSplit σ) a) b) c) d   [the piece, applied four times]
    = rapp σ (rapp σ a c) (rapp σ (rapp σ (sPromote σ) b) d)   [first to third, second to fourth]

    So: every step is one of `bRoute_bracket` and `cRoute_app`, so this uses no exponential and
    no property of the parameter. It is pure rewiring. -/
theorem sSplit_app (σ : ℝ) (a b c d : Route) :
    rapp σ (rapp σ (rapp σ (rapp σ (sSplit σ) a) b) c) d
      = rapp σ (rapp σ a c) (rapp σ (rapp σ (sPromote σ) b) d) := by
  rw [sSplit, bRoute_bracket, cRoute_app, bRoute_bracket, sShare, bRoute_bracket, cRoute_app,
    bRoute_bracket, cRoute_app, bRoute_bracket]

/-- **K, for the ordinary application.** The discarding combinator discards.

@cite abramsky-haghverdi-scott-2002
@cite curry-1930

    (σ : ℝ) (x y : Route)                                  [the parameter and two routes]
    capp σ (capp σ (kCombinator σ) x) y = x                [it hands back the first, bare]

    So: `capp_kRoute` gave `bang x` and this gives `x`, which is the whole of what the
    dereliction was inserted for. -/
theorem capp_kCombinator (σ : ℝ) (x y : Route) :
    capp σ (capp σ (kCombinator σ) x) y = x := by
  simp only [capp, kCombinator]
  rw [bRoute_bracket, rapp_dRoute, rapp_kRoute]

/-- **S, for the ordinary application.** The substituting combinator substitutes.

@cite abramsky-haghverdi-scott-2002
@cite schonfinkel-1924
@cite curry-1930

    (σ : ℝ) (x y z : Route)                                [the parameter and three routes]
    capp σ (capp σ (capp σ (sCombinator σ) x) y) z         [the combinator, applied three times]
    = capp σ (capp σ x z) (capp σ y z)                     [gives the argument to each]

    So: with `capp_kCombinator` this says the routes under `capp σ` are a combinatory algebra,
    at every setting of the parameter, which is Abramsky, Haghverdi and Scott's theorem that a
    linear combinatory algebra yields an ordinary one, instantiated at their own partial
    injections. `Combinator/Basis.lean` had the linear half; this is the other. -/
theorem capp_sCombinator (σ : ℝ) (x y z : Route) :
    capp σ (capp σ (capp σ (sCombinator σ) x) y) z = capp σ (capp σ x z) (capp σ y z) := by
  simp only [capp, sCombinator]
  rw [bRoute_bracket, rapp_dRoute, bRoute_bracket, bRoute_bracket, wRoute_app, sSplit_app,
    sPromote_app]

/-- **The routes are a combinatory algebra.** Both defining equations, at every setting.

@cite abramsky-haghverdi-scott-2002
@cite vanoosten-2008

    (σ : ℝ)                                                [the parameter]
    ∃ k s : Route                                          [there are two routes]
    ∀ x y, capp σ (capp σ k x) y = x                       [the first discards]
    ∀ x y z, capp σ (capp σ (capp σ s x) y) z              [the second]
      = capp σ (capp σ x z) (capp σ y z)                   [substitutes]
    ∧                                                      [both together]

    So: `theory/LITERATURE.md` recorded that the corpus had an applicative structure and no
    combinatory algebra over it. This is the combinatory algebra, and it is the one the
    exponential forces: the operation is `capp`, not `rapp`, and the two elements are the
    corrected combinators rather than `kRoute` and `sRoute`. -/
theorem routes_are_a_combinatory_algebra (σ : ℝ) :
    ∃ k s : Route, (∀ x y : Route, capp σ (capp σ k x) y = x) ∧
      (∀ x y z : Route, capp σ (capp σ (capp σ s x) y) z
        = capp σ (capp σ x z) (capp σ y z)) :=
  ⟨kCombinator σ, sCombinator σ, capp_kCombinator σ, capp_sCombinator σ⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE DENOTATION, AND PRESERVATION
-- ═══════════════════════════════════════════════════════════════════

/-- **A term as a route.**

@cite abramsky-haghverdi-scott-2002
@cite engeler-1981

    (σ : ℝ) (env : ℕ → Route)                      [the parameter, and a route for each variable]
    Comb → Route                                   [a term becomes a weighted wire map]

    So: the two constants go to the corrected combinators, a variable goes to whatever the
    environment holds for it, and an application goes to `capp`, which feeds the argument once
    for every consumer. Total on every term, with no side condition. -/
def denote (σ : ℝ) (env : ℕ → Route) : Comb → Route
  | Comb.S => sCombinator σ
  | Comb.K => kCombinator σ
  | Comb.var n => env n
  | Comb.app f x => capp σ (denote σ env f) (denote σ env x)

/-- **Preservation.** One rewriting step does not change the route.

@cite abramsky-haghverdi-scott-2002
@cite curry-1930

    {σ : ℝ} {env : ℕ → Route}                      [the parameter and an environment]
    {a b : Comb}                                   [two terms]
    (h : Comb.Step a b)                            [the first rewrites to the second]
    denote σ env a = denote σ env b                [and they are the same route]

    So: no fragment and no hypothesis on the term. The two rule cases are exactly
    `capp_kCombinator` and `capp_sCombinator`, and the two congruence cases are the recursion of
    `denote` on an application. -/
theorem denote_step {σ : ℝ} {env : ℕ → Route} {a b : Comb} (h : Comb.Step a b) :
    denote σ env a = denote σ env b := by
  induction h with
  | k x y => exact capp_kCombinator σ (denote σ env x) (denote σ env y)
  | s f g x => exact capp_sCombinator σ (denote σ env f) (denote σ env g) (denote σ env x)
  | left b _ ih => simp only [denote, ih]
  | right a _ ih => simp only [denote, ih]

/-- **Preservation, for any number of steps.**

    {σ : ℝ} {env : ℕ → Route}                      [the parameter and an environment]
    {a b : Comb}                                   [two terms]
    (h : Comb.Reduces a b)                         [the first reaches the second]
    denote σ env a = denote σ env b                [and they are the same route] -/
theorem denote_reduces {σ : ℝ} {env : ℕ → Route} {a b : Comb} (h : Comb.Reduces a b) :
    denote σ env a = denote σ env b := by
  induction h with
  | refl => rfl
  | tail _ hstep ih => exact ih.trans (denote_step hstep)

/-- **Two terms that meet have the same route.**

    {σ : ℝ} {env : ℕ → Route}                      [the parameter and an environment]
    {a b c : Comb}                                 [three terms]
    (ha : Comb.Reduces a c)                        [the first reaches the third]
    (hb : Comb.Reduces b c)                        [so does the second]
    denote σ env a = denote σ env b                [then the first two are the same route]

    So: by `confluence`, two terms are convertible exactly when they meet, so this is
    convertibility implying equality of routes. -/
theorem denote_join {σ : ℝ} {env : ℕ → Route} {a b c : Comb} (ha : Comb.Reduces a c)
    (hb : Comb.Reduces b c) : denote σ env a = denote σ env b :=
  (denote_reduces ha).trans (denote_reduces hb).symm

/-- **Different routes certify that two terms are not convertible.**

@cite barendregt-1984

    {σ : ℝ} {env : ℕ → Route}                      [the parameter and an environment]
    {a b : Comb}                                   [two terms]
    (h : denote σ env a ≠ denote σ env b)          [their routes differ]
    ¬ ∃ c, Comb.Reduces a c ∧ Comb.Reduces b c     [then they never meet]

    So: this is the direction that is worth something to a decision procedure. A normaliser with
    a fuel bound reports neither EQUAL nor DISTINCT when the bound runs out; a difference between
    the two routes answers DISTINCT outright, without normalising either term, because by
    `confluence` meeting is exactly convertibility. The converse is not proved and is not
    claimed. -/
theorem not_join_of_denote_ne {σ : ℝ} {env : ℕ → Route} {a b : Comb}
    (h : denote σ env a ≠ denote σ env b) :
    ¬ ∃ c : Comb, Comb.Reduces a c ∧ Comb.Reduces b c := by
  rintro ⟨c, ha, hb⟩
  exact h (denote_join ha hb)

-- ═══════════════════════════════════════════════════════════════════
-- HOW FAR THE OTHER DIRECTION IS FROM HOLDING
-- ═══════════════════════════════════════════════════════════════════

/-- A variable takes no step.

    {n : ℕ} {t : Comb}                             [a variable name and a term]
    ¬ Comb.Step (Comb.var n) t                     [nothing rewrites a bare variable] -/
theorem no_step_var {n : ℕ} {t : Comb} : ¬ Comb.Step (Comb.var n) t := by
  intro h
  cases h

/-- A variable applied to a variable is a normal form.

    (m n : ℕ)                                                       [two variable names]
    Comb.NormalForm (Comb.app (Comb.var m) (Comb.var n))            [no rule applies to it] -/
theorem varApp_normalForm (m n : ℕ) :
    Comb.NormalForm (Comb.app (Comb.var m) (Comb.var n)) := by
  rintro ⟨t, ht⟩
  cases ht with
  | left _ h => exact no_step_var h
  | right _ h => exact no_step_var h

/-- **The denotation is not faithful.** Two distinct normal forms can be one route.

@cite barendregt-1984

    (σ : ℝ)                                                [the parameter]
    ∃ env : ℕ → Route                                      [there is an environment]
    ∃ a b : Comb                                           [and two terms]
    denote σ env a = denote σ env b                        [with the same route]
    ¬ ∃ c, Comb.Reduces a c ∧ Comb.Reduces b c             [that never meet]
    ∧                                                      [both together]

    So: route equality is strictly coarser than convertibility, and the witness says exactly
    where the coarseness comes from. Send a variable to the empty route; by `rapp_rnone` the empty
    route absorbs, so every application headed by that variable is the empty route whatever its
    argument, while the terms themselves are distinct normal forms. This is bottom behaving like
    bottom, which is what a domain-theoretic model does and not a defect.

    The scope is exactly this. It is an environment sending a variable to the empty route, so it
    says nothing about closed terms, and nothing about whether route equality on closed terms is
    strictly coarser than convertibility. `theory/DEAD_ENDS.md` records what was tried on that
    question and why none of it closed. -/
theorem denote_not_faithful (σ : ℝ) :
    ∃ (env : ℕ → Route) (a b : Comb), denote σ env a = denote σ env b ∧
      ¬ ∃ c : Comb, Comb.Reduces a c ∧ Comb.Reduces b c := by
  refine ⟨fun _ => rnone, Comb.app (Comb.var 0) (Comb.var 1),
    Comb.app (Comb.var 0) (Comb.var 2), ?_, ?_⟩
  · simp only [denote, capp, rapp_rnone]
  · rintro ⟨c, ha, hb⟩
    have h1 := Comb.eq_of_reduces_normalForm (varApp_normalForm 0 1) ha
    have h2 := Comb.eq_of_reduces_normalForm (varApp_normalForm 0 2) hb
    exact absurd (h1.trans h2.symm) (by simp)

-- ═══════════════════════════════════════════════════════════════════
-- THE LINEAR DENOTATION, AND WHERE IT STOPS
-- ═══════════════════════════════════════════════════════════════════

/-- The naive denotation: application is `rapp` and the constants are the linear combinators.

@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (env : ℕ → Route)                      [the parameter, and a route for each variable]
    Comb → Route                                   [a term becomes a weighted wire map]

    So: this is what the calculus looks like read into the *linear* algebra, with no exponential
    anywhere. It is kept because the two statements below are the sharpest description of what
    the exponential is buying. -/
def linearDenote (σ : ℝ) (env : ℕ → Route) : Comb → Route
  | Comb.S => sRoute σ
  | Comb.K => kRoute
  | Comb.var n => env n
  | Comb.app f x => rapp σ (linearDenote σ env f) (linearDenote σ env x)

/-- **The discarding rule survives the linear denotation, unrestrictedly.**

    (σ : ℝ) (env : ℕ → Route) (x y : Comb)                     [parameter, environment, terms]
    linearDenote σ env (Comb.app (Comb.app Comb.K x) y)        [K applied to both]
    = linearDenote σ env x                                     [is the first, whatever y is]

    So: weakening needs no exponential, because a route may be absent at a position. -/
theorem linearDenote_step_k (σ : ℝ) (env : ℕ → Route) (x y : Comb) :
    linearDenote σ env (Comb.app (Comb.app Comb.K x) y) = linearDenote σ env x :=
  rapp_kRoute σ (linearDenote σ env x) (linearDenote σ env y)

/-- **The substituting rule survives the linear denotation exactly when the argument is
    already an exponential.**

@cite girard-1989

    (σ : ℝ) (env : ℕ → Route) (f g x : Comb)                   [parameter, environment, terms]
    {z : Route}                                                [a route]
    (hz : linearDenote σ env x = bang z)                       [the argument is an exponential]
    linearDenote σ env (Comb.app (Comb.app (Comb.app Comb.S f) g) x)   [S applied to all three]
    = linearDenote σ env (Comb.app (Comb.app f x) (Comb.app g x))      [is the substituted term]

    So: the hypothesis is not a restriction that can be pushed onto terms and left there. The
    right-hand side puts `Comb.app g x` into argument position, and nothing makes the route of an
    application an exponential, so the same hypothesis is not available at the next step. That is
    why the fix is `capp`, which puts the exponential on at every application rather than asking
    the term to have brought one. -/
theorem linearDenote_step_s_of_bang (σ : ℝ) (env : ℕ → Route) (f g x : Comb) {z : Route}
    (hz : linearDenote σ env x = bang z) :
    linearDenote σ env (Comb.app (Comb.app (Comb.app Comb.S f) g) x)
      = linearDenote σ env (Comb.app (Comb.app f x) (Comb.app g x)) := by
  simp only [linearDenote, hz]
  exact sRoute_app σ (linearDenote σ env f) (linearDenote σ env g) z

end Combinator
end Apeiron

end
