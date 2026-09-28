import Substrate.Algebra.Bulk.BulkTower
import Substrate.Ski.IotaTermAlgebra
import Substrate.Ski.TermCore
import Substrate.Algebra.Translation.UniverseSyntax

/-!
# The Polymorphic Iota Term Algebra
@name One Shape Holds Everything
@child Every program, sentence, picture, or equation can be written as the same little tree built from three pieces. This is that tree. Once everything is the same tree, turning one thing into another is just reading the tree a different way.
@scientist The free bulk algebra forced by x = B/U: a three-constructor term over ANY carrier α, evaluated under ANY binary operation. Languages, formats, opcodes, modalities are different applicatives over the one Term — meaning is the applicative you choose, never the structure.
@engineer `Polymorphic.Term α` with `state app`: `state (K B U) = app B U`; `state (S_routeF prev outer) = app (state prev) outer`; mirror for `S_routeG`. The bulk's setting is the α = (ℝ→ℝ), app = pointwiseDiv specialisation — but the same three-constructor shape lives at every applicative.

`IotaTermAlgebra` specialises the combinator-named recursive
structure to α = (ℝ → ℝ) with pointwise division as the
applicative. The framework's bulk setting is one specialisation;
the same three-constructor shape lives at every applicative.

This file pulls the type parameter and the applicative out.

## The polymorphic term algebra

For any carrier type `α : Type u`:

* `Polymorphic.Term α` — the three-constructor inductive type
  (`K B U`, `S_routeF prev outer`, `S_routeG outer prev`), with
  every leaf and edge carrying values of `α`.
* `Polymorphic.Term.state app` — evaluation under any binary
  operation `app : α → α → α`. The state of `K B U` is `app B U`;
  the state of `S_routeF prev outer` is `app (state prev) outer`;
  the state of `S_routeG outer prev` is `app outer (state prev)`.

The bulk's setting is recovered as:

* `α := (ℝ → ℝ)`,
* `app := pointwiseDiv` (pointwise division of functions).

At this specialisation, `Polymorphic.Term (ℝ → ℝ)` is canonically
isomorphic to both `BulkTower` and `IotaTermAlgebra.IotaTerm`, with
state preserved by the canonical bijections.

## What this captures

The combinator triad (K, S_routeF, S_routeG) is **applicative-
parametric**. The specific applicative determines the *content* of
each constructor's state; the *shape* of the recursion is the same
at every applicative. This is the "as above, so below" pattern at
the applicative layer.

## What this does NOT directly capture

Standard SKI Iota's reduction calculus (`K x y → x`, `S x y z →
x z (y z)`, `ι x → x S K`) imposes specific computational meaning
on juxtaposition that goes beyond a binary operation `app`. The
present file gives the *structural* polymorphic equivalence: the
three-constructor shape and the recursive state evaluation under
any applicative. Connecting that to the specific SKI reduction
calculus is a separate question (the bulk's applicative is
division; standard SKI's would be juxtaposition with the standard
reductions).

## What the polymorphism is proved to buy

* **Proved (native):** `map_state`, the naturality of the fold under `Polymorphic.map`, and
  `state_congr` — changing the applicative is the only freedom, the constructors and the
  recursion are forced. `state_congr` is one proof under two names: it rides
  `UniverseSyntax.universe_syntax_evaluation_parametric` and stays here because the citations
  point here.
* **Proved (native):** at the bulk specialisation `α = (ℝ → ℝ)`, `app = pointwiseDiv`, the
  term algebra is canonically isomorphic to `BulkTower` and to `IotaTermAlgebra.IotaTerm`
  with `state` preserved — `fromBulkTower_toBulkTower`, `toBulkTower_fromBulkTower`,
  `fromIotaTerm_toIotaTerm`, `toIotaTerm_fromIotaTerm`,
  `state_pointwiseDiv_eq_BulkTower_state` and `state_pointwiseDiv_eq_IotaTerm_state`,
  conjoined as `triple_equivalence`.
* **Proved (riding):** the carrier itself is not declared here. The three-constructor
  `Polymorphic.Term α` and `Polymorphic.Term.state` live in the Mathlib-free `TermCore`, with
  `TermCore.state_K`, `TermCore.state_S_routeF` and `TermCore.state_S_routeG` as the
  evaluation clauses; this
  file is the bulk specialisation and the equivalences on top of them.
* **Anchor:** that "every program, sentence, picture or equation" is this tree,
  and that "languages, formats, opcodes, modalities are different applicatives over the one
  `Term`". What is proved is parametricity over a carrier and a binary operation; that those
  carriers are languages and modalities is the framework's reading, not a statement here.
* **Open:** the standard SKI reduction calculus — the file's own "What this does NOT directly
  capture". `SKIReductionBridge` ties `Term.state` to constructor dispatch on
  `Polymorphic.Term` (`SKIReductionBridge.ski_reduction_bridge`), but its own scope note says
  it does NOT prove normalisation of raw SKI/`Comb` expressions, and cannot: `Term` has no
  application node and no substitution, hence no redex, so the rewrite rules `K x y → x`,
  `S x y z → x z (y z)`, `ι x → x S K` are not in it. That calculus lives on `Comb` in
  `SkiReductionContextual` / `Confluence`, no map takes `Term α` to `Comb`, and no statement
  connects the two.

-/

noncomputable section

universe u

namespace Sanguine
namespace Polymorphic

-- The polymorphic `Term` and its `state` now live in the pure `Substrate.Ski.TermCore`
-- (Mathlib-free, the execution-path carrier). This file re-imports it and proves the parity
-- with the bulk's `ℝ → ℝ` applicative below.

-- ═══════════════════════════════════════════════════════════════════
-- THE BULK'S APPLICATIVE: POINTWISE DIVISION ON (ℝ → ℝ)
-- ═══════════════════════════════════════════════════════════════════

/-- Pointwise division of functions `ℝ → ℝ`. This is the explicit
    applicative under which the bulk algebra's combinator triad
    lives. -/
def pointwiseDiv : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ) :=
  fun B U p => B p / U p

@[simp] theorem pointwiseDiv_apply (B U : ℝ → ℝ) (p : ℝ) :
    pointwiseDiv B U p = B p / U p := rfl

/-- The bulk's algebra read at one point: `/` on `ℝ` with every leaf evaluated at `p`. Every
    "state at `p`" map out of `Term (ℝ → ℝ)` respects it, so the bridges below are
    `fromTerm_ext` here rather than six inductions. -/
def evalAlgebra (p : ℝ) : UniverseSyntax.AlgebraStructure (ℝ → ℝ) ℝ where
  inj := fun B U => B p / U p
  extL := fun x outer => x / outer p
  extR := fun outer x => outer p / x

-- ═══════════════════════════════════════════════════════════════════
-- BRIDGE: BULKTOWER ↔ TERM (ℝ → ℝ)
-- ═══════════════════════════════════════════════════════════════════

/-- Send a `BulkTower` to the corresponding polymorphic term over
    `(ℝ → ℝ)`. The constructor renaming is exactly the same as
    `IotaTermAlgebra.fromBulkTower`. -/
def fromBulkTower : BulkTower → Term (ℝ → ℝ)
  | .base e => .K e.B e.U
  | .stepB prev outer => .S_routeF (fromBulkTower prev) outer
  | .stepU outer prev => .S_routeG outer (fromBulkTower prev)

/-- Send a polymorphic term over `(ℝ → ℝ)` back to a `BulkTower`. -/
def toBulkTower : Term (ℝ → ℝ) → BulkTower
  | .K B U => .base ⟨B, U⟩
  | .S_routeF prev outer => .stepB (toBulkTower prev) outer
  | .S_routeG outer prev => .stepU outer (toBulkTower prev)

/-- Round trip via `toBulkTower` then `fromBulkTower` is identity. Both legs respect `Term`'s own
    algebra, so this is `fromTerm_ext` at `termAlgebra`. -/
theorem fromBulkTower_toBulkTower (t : Term (ℝ → ℝ)) :
    fromBulkTower (toBulkTower t) = t :=
  UniverseSyntax.fromTerm_ext (UniverseSyntax.termAlgebra (ℝ → ℝ))
    (fun t => fromBulkTower (toBulkTower t)) (fun t => t)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) t

/-- Round trip via `fromBulkTower` then `toBulkTower` is identity.

    This one is NOT `Term`'s initiality: the recursion is over `BulkTower`, so it is `BulkTower`'s
    own freeness (`FreeBulkAlgebra.hom_toFun_unique`). One induction per free algebra is the floor;
    everything below transports through this round trip instead of re-inducting. -/
theorem toBulkTower_fromBulkTower (t : BulkTower) :
    toBulkTower (fromBulkTower t) = t := by
  induction t with
  | base e => rfl
  | stepB prev outer ih =>
      simp only [fromBulkTower, toBulkTower, ih]
  | stepU outer prev ih =>
      simp only [fromBulkTower, toBulkTower, ih]

/-- `Term.state pointwiseDiv` on a polymorphic term over `(ℝ → ℝ)` equals `BulkTower.state` on its
    `toBulkTower` image. Both sides read `Term (ℝ → ℝ)` into `ℝ` respecting `evalAlgebra p`. -/
theorem BulkTower_state_eq_state_pointwiseDiv
    (t : Term (ℝ → ℝ)) (p : ℝ) :
    BulkTower.state (toBulkTower t) p = Term.state pointwiseDiv t p :=
  UniverseSyntax.fromTerm_ext (evalAlgebra p)
    (fun t => BulkTower.state (toBulkTower t) p)
    (fun t => Term.state pointwiseDiv t p)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) t

/-- The bulk's `BulkTower.state` equals the polymorphic `Term.state` under `pointwiseDiv`. The
    polymorphic structure plus the bulk's applicative IS the bulk algebra's state. Transported
    along the round trip, not re-inducted. -/
theorem state_pointwiseDiv_eq_BulkTower_state
    (t : BulkTower) (p : ℝ) :
    Term.state pointwiseDiv (fromBulkTower t) p = BulkTower.state t p := by
  rw [← BulkTower_state_eq_state_pointwiseDiv (fromBulkTower t) p, toBulkTower_fromBulkTower]

-- ═══════════════════════════════════════════════════════════════════
-- BRIDGE: IOTATERM ↔ TERM (ℝ → ℝ)
-- ═══════════════════════════════════════════════════════════════════

/-- Send an `IotaTerm` to the corresponding polymorphic term over
    `(ℝ → ℝ)`. The constructor names already match — this is the
    canonical type-erasure of the `(ℝ → ℝ)`-baked specialisation. -/
def fromIotaTerm : IotaTermAlgebra.IotaTerm → Term (ℝ → ℝ)
  | .K B U => .K B U
  | .S_routeF prev outer => .S_routeF (fromIotaTerm prev) outer
  | .S_routeG outer prev => .S_routeG outer (fromIotaTerm prev)

/-- Send a polymorphic term over `(ℝ → ℝ)` back to an `IotaTerm`. -/
def toIotaTerm : Term (ℝ → ℝ) → IotaTermAlgebra.IotaTerm
  | .K B U => .K B U
  | .S_routeF prev outer => .S_routeF (toIotaTerm prev) outer
  | .S_routeG outer prev => .S_routeG outer (toIotaTerm prev)

theorem fromIotaTerm_toIotaTerm (t : Term (ℝ → ℝ)) :
    fromIotaTerm (toIotaTerm t) = t :=
  UniverseSyntax.fromTerm_ext (UniverseSyntax.termAlgebra (ℝ → ℝ))
    (fun t => fromIotaTerm (toIotaTerm t)) (fun t => t)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) t

/-- `IotaTerm`'s own initiality — the recursion is over `IotaTerm`, not `Term`, so `fromTerm_ext`
    does not reach it. `IotaTerm` has no freeness theorem of its own; this induction is it. -/
theorem toIotaTerm_fromIotaTerm (t : IotaTermAlgebra.IotaTerm) :
    toIotaTerm (fromIotaTerm t) = t := by
  induction t with
  | K B U => rfl
  | S_routeF prev outer ih =>
      simp only [fromIotaTerm, toIotaTerm, ih]
  | S_routeG outer prev ih =>
      simp only [fromIotaTerm, toIotaTerm, ih]

theorem IotaTerm_state_eq_state_pointwiseDiv
    (t : Term (ℝ → ℝ)) (p : ℝ) :
    IotaTermAlgebra.state (toIotaTerm t) p = Term.state pointwiseDiv t p :=
  UniverseSyntax.fromTerm_ext (evalAlgebra p)
    (fun t => IotaTermAlgebra.state (toIotaTerm t) p)
    (fun t => Term.state pointwiseDiv t p)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) t

/-- The Iota term's state equals the polymorphic term's state at
    `pointwiseDiv`. The combinator-named view and the applicative-
    parametric view coincide at the bulk's specialisation. -/
theorem state_pointwiseDiv_eq_IotaTerm_state
    (t : IotaTermAlgebra.IotaTerm) (p : ℝ) :
    Term.state pointwiseDiv (fromIotaTerm t) p = IotaTermAlgebra.state t p := by
  rw [← IotaTerm_state_eq_state_pointwiseDiv (fromIotaTerm t) p, toIotaTerm_fromIotaTerm]

-- ═══════════════════════════════════════════════════════════════════
-- THE TRIPLE EQUIVALENCE AT THE BULK'S APPLICATIVE
-- ═══════════════════════════════════════════════════════════════════

/-- **Triple equivalence at the bulk's applicative.** At
    `α = (ℝ → ℝ)`, `app = pointwiseDiv`, the three presentations
    `BulkTower`, `IotaTermAlgebra.IotaTerm`, and `Term (ℝ → ℝ)` are
    canonically isomorphic with state preserved.

    This shows the bulk's combinator-named algebra is one *named*
    specialisation of an applicative-parametric structure; the
    polymorphic `Term α` is the same shape at any applicative. -/
theorem triple_equivalence :
    -- BulkTower ↔ Term (ℝ → ℝ)
    (∀ t : BulkTower,
        toBulkTower (fromBulkTower t) = t) ∧
    (∀ t : Term (ℝ → ℝ),
        fromBulkTower (toBulkTower t) = t) ∧
    -- IotaTerm ↔ Term (ℝ → ℝ)
    (∀ t : IotaTermAlgebra.IotaTerm,
        toIotaTerm (fromIotaTerm t) = t) ∧
    (∀ t : Term (ℝ → ℝ),
        fromIotaTerm (toIotaTerm t) = t) ∧
    -- State preserved in every direction
    (∀ (t : BulkTower) (p : ℝ),
        Term.state pointwiseDiv (fromBulkTower t) p = BulkTower.state t p) ∧
    (∀ (t : IotaTermAlgebra.IotaTerm) (p : ℝ),
        Term.state pointwiseDiv (fromIotaTerm t) p = IotaTermAlgebra.state t p) :=
  ⟨toBulkTower_fromBulkTower,
   fromBulkTower_toBulkTower,
   toIotaTerm_fromIotaTerm,
   fromIotaTerm_toIotaTerm,
   state_pointwiseDiv_eq_BulkTower_state,
   state_pointwiseDiv_eq_IotaTerm_state⟩

-- ═══════════════════════════════════════════════════════════════════
-- INVARIANCE OF THE SHAPE UNDER APPLICATIVE CHANGE
-- ═══════════════════════════════════════════════════════════════════

/-- **Applicative congruence.** If two applicatives `app₁`, `app₂ :
    α → α → α` agree pointwise on the leaves and edges of a term,
    they give the same state. The shape is fixed; the content
    follows the applicative.

    This is the formal statement of "the shape is applicative-
    parametric": changing the applicative is the only freedom; the
    constructors and recursion are forced.

    One proof, two names: this is `UniverseSyntax`'s requirement 3
    (`universe_syntax_evaluation_parametric`, initiality at the `app₂` state-algebra).
    `state_congr` is the corpus's name for it and stays here, where the citations point. -/
theorem state_congr {α : Type u}
    {app₁ app₂ : α → α → α}
    (h : ∀ x y, app₁ x y = app₂ x y)
    (t : Term α) :
    Term.state app₁ t = Term.state app₂ t :=
  UniverseSyntax.universe_syntax_evaluation_parametric t app₁ app₂ h

-- `Polymorphic.map` + its simp lemmas now live in the pure `TermCore` (the execution-path carrier);
-- this file re-imports them and adds the naturality theorem below.

/-- **Naturality.** If `f : α → β` intertwines the applicatives —
    i.e. `f (app_α x y) = app_β (f x) (f y)` — then transport
    commutes with state.

    The square is not a square. Both `state app_β ∘ map f` and `f ∘ state app_α` are maps
    `Term α → β` respecting the ONE mixed-carrier algebra `transportAlgebra f app_β`, so
    naturality is `fromTerm_ext` there: uniqueness applied twice, no commuting diagram to chase.
    The left column is rfl (that is `map`'s definition); `intertwine` is the whole right column. -/
theorem map_state {α : Type u} {β : Type u}
    (f : α → β)
    (app_α : α → α → α) (app_β : β → β → β)
    (intertwine : ∀ x y, f (app_α x y) = app_β (f x) (f y))
    (t : Term α) :
    Term.state app_β (map f t) = f (Term.state app_α t) :=
  UniverseSyntax.fromTerm_ext (UniverseSyntax.transportAlgebra f app_β)
    (fun t => Term.state app_β (map f t)) (fun t => f (Term.state app_α t))
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun B U => intertwine B U) (fun _ outer => intertwine _ outer)
    (fun outer _ => intertwine outer _) t

end Polymorphic
end Sanguine
