import Apeiron.Combinator.Linear.Fusing

/-!
# The substituting combinator from the basis

@name Basis
@cite schonfinkel-1924
@cite curry-1930
@cite abramsky-haghverdi-scott-2002
@grade proved

With `Combinator/Fusing.lean` the composing combinator is an element, and with
`Combinator/Elements.lean` so are the exchanging and contracting ones. Combinatory logic then
gives the substituting combinator over that basis, `S = B (B (B W) C) (B B)`, and applying
elements to elements is an operation this carrier already has. So `sRoute` needs no routing of its
own and no trace argument: it is a term in the combinators already built, and the equation follows
from theirs by rewriting.

## Root

| from | here |
|---|---|
| nothing new | the equation is `bRoute_bracket`, `cRoute_app` and `wRoute_app` composed |

## The one thing to check

The classical identity is stated for the untyped calculus, where an argument may be used twice
without ceremony. Here the algebra is linear and the contraction is the one place a route is used
twice, so `wRoute_app` asks for its argument banged. Following the derivation through, the
argument that reaches the contraction is the third one, so the equation comes out as

```
   S x y !z = (x !z) (y !z)
```

which is the substituting combinator read through `x · y := x (!y)`, exactly as Abramsky,
Haghverdi and Scott's translation from the linear algebra to the ordinary one requires. Nothing
else needs a bang, because `B` and `C` are linear.

## Where the parameter is

`sRoute` is built by applying routes to routes, so it depends on the parameter and on nothing
else. That is what an element is at this carrier: the routings `bRoute`, `cRoute` and `wRoute`
are themselves free of it, and the parameter enters only through the weight multiplication that
`rapp` performs.

## What now stands

With this file the eight combinators of a linear combinatory algebra are all elements of this
carrier. Let a route be a weighted partial injection of the counting numbers, let application be
the trace, `rapp sigma a b = traceRoute sigma (plug sigma a b)`, which is total, and let the
exponential be `bang`. Then for every real `sigma` the triple `(Route, rapp sigma, bang)` is a
linear combinatory algebra in the sense of Abramsky, Haghverdi and Scott, *Geometry of Interaction
and linear combinatory algebras*, MSCS 12 (2002) 625-665, with

```
   B   bRoute        Combinator/Storing.lean        bRoute_bracket      Combinator/Fusing.lean
   C   cRoute        Combinator/Elements.lean       cRoute_app          Combinator/Elements.lean
   I   iRoute        Combinator/Interaction.lean    rapp_iRoute         Combinator/Duplication.lean
   K   kRoute        Combinator/Interaction.lean    rapp_kRoute_bang    Combinator/Duplication.lean
   W   wRoute        Combinator/Elements.lean       wRoute_app          Combinator/Elements.lean
   D   dRoute        Combinator/Exponential.lean    rapp_dRoute         Combinator/Exponential.lean
   dig diggingRoute  Combinator/Exponential.lean    rapp_diggingRoute   Combinator/Exponential.lean
   F   fRoute        Combinator/Elements.lean       fRoute_app          Combinator/Elements.lean
```

and `sRoute` here besides. This instantiates their theorem that a Geometry of Interaction
situation yields such an algebra, at their own partial injections example, `U` the counting
numbers with `T X = U times X`, which `Combinator/Interaction.lean` builds with Cantor's pairing
supplying both retractions.

Two things sharpen it. The eight routes are free of the parameter, so one family of wirings serves
every algebra in the family and the parameter enters only through the weight multiplication
`rapp` performs. And the contraction and the substitution have forms which do not mention the
exponential at all, `contraction` and `substitution`, taking the duplicated argument as its two
copies; those are the linear facts, and `conj_contract_bang` is the relabelling which turns them
into the forms the literature states.
-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

/-- The substituting combinator as a term in the basis.

@cite schonfinkel-1924
@cite curry-1930

    (σ : ℝ)                            [the parameter]
    B (B (B W) C) (B B)                [the classical definition over the basis]
    : Route                            [a weighted wire map] -/
def sRoute (σ : ℝ) : Route :=
  rapp σ (rapp σ bRoute (rapp σ (rapp σ bRoute (rapp σ bRoute wRoute)) cRoute))
    (rapp σ bRoute bRoute)

/-- **S as an element.** The substituting combinator is one route built from the others.

@cite schonfinkel-1924
@cite curry-1930
@cite abramsky-haghverdi-scott-2002

    (σ : ℝ) (x y z : Route)                        [the parameter and three routes]
    rapp σ (rapp σ (rapp σ (sRoute σ) x) y) (bang z)     [the combinator, applied three times]
    = rapp σ (rapp σ x (bang z)) (rapp σ y (bang z))     [gives the argument to each]

    So: no new routing and no new trace argument. Each step is one of `bRoute_bracket`,
    `cRoute_app` or `wRoute_app`, and the argument arrives banged because the contraction is the
    one place a route is used twice. -/
theorem sRoute_app (σ : ℝ) (x y z : Route) :
    rapp σ (rapp σ (rapp σ (sRoute σ) x) y) (bang z)
      = rapp σ (rapp σ x (bang z)) (rapp σ y (bang z)) := by
  have e1 : rapp σ (sRoute σ) x
      = rapp σ (rapp σ (rapp σ bRoute (rapp σ bRoute wRoute)) cRoute)
          (rapp σ (rapp σ bRoute bRoute) x) :=
    bRoute_bracket σ _ _ x
  have e2 : rapp σ (rapp σ (rapp σ bRoute (rapp σ bRoute wRoute)) cRoute)
          (rapp σ (rapp σ bRoute bRoute) x)
      = rapp σ (rapp σ bRoute wRoute)
          (rapp σ cRoute (rapp σ (rapp σ bRoute bRoute) x)) :=
    bRoute_bracket σ _ _ _
  have e3 : rapp σ (rapp σ (rapp σ bRoute wRoute)
          (rapp σ cRoute (rapp σ (rapp σ bRoute bRoute) x))) y
      = rapp σ wRoute (rapp σ (rapp σ cRoute (rapp σ (rapp σ bRoute bRoute) x)) y) :=
    bRoute_bracket σ _ _ y
  have e4 : rapp σ (rapp σ wRoute
          (rapp σ (rapp σ cRoute (rapp σ (rapp σ bRoute bRoute) x)) y)) (bang z)
      = rapp σ (rapp σ (rapp σ (rapp σ cRoute (rapp σ (rapp σ bRoute bRoute) x)) y) (bang z))
          (bang z) :=
    wRoute_app σ _ z
  have e5 : rapp σ (rapp σ (rapp σ cRoute (rapp σ (rapp σ bRoute bRoute) x)) y) (bang z)
      = rapp σ (rapp σ (rapp σ (rapp σ bRoute bRoute) x) (bang z)) y :=
    cRoute_app σ _ y (bang z)
  have e6 : rapp σ (rapp σ (rapp σ bRoute bRoute) x) (bang z)
      = rapp σ bRoute (rapp σ x (bang z)) :=
    bRoute_bracket σ bRoute x (bang z)
  rw [e1, e2, e3, e4, e5, e6, bRoute_bracket]

end Combinator
end Apeiron

end
