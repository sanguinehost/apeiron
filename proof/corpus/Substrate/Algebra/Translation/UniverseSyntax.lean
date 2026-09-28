import Substrate.Ski.TermCore

/-!
# Universe Syntax — what any bulk universe must support
@name The Smallest Possible Grammar
@child What's the least a language needs to say anything at all? Just three things: name something, join two things together, and remember which one came first. That's the whole grammar — everything else is built from it.
@scientist The minimal syntax any bulk universe needs falls out of three requirements from the bulk's axioms alone — no presupposition of dimension, observer count, or applicative: inject α-values (one atomic constructor), compose under the binary `/` (structural, content-blind), and distinguish routing (since `/` is non-commutative).
@engineer One atom constructor + binary composition + a routing distinction = the three-constructor Term. Evaluation knows only that the applicative is a binary function, never its content — so the one syntax serves every universe.

The bulk's operator is `/`, binary. The minimal syntax any bulk
universe needs falls out of three requirements, derived from the
bulk's axioms alone — no presupposition about dimension, observer
count, or applicative.

1. **Inject α-values.** One atomic constructor, or the syntax says
   nothing.
2. **Compose under the binary `/`.** Composition is binary and
   *structural* — evaluation knows only that the applicative is a
   binary function, never its content.
3. **Distinguish routing.** `/` is non-commutative, so the syntax
   needs two recursive constructors: recurse-left and recurse-right.

Three constructors, forced — two lose routing, four are redundant.
`Polymorphic.Term` is exactly this syntax; each requirement is
proven against it, and the routing distinction shown non-vacuous
under the bulk's own applicative.

## Proven

* `universe_syntax_can_introduce` — atomic introduction is realisable
  (`Term.state_K`, ∃-wrapped).
* `universe_syntax_can_route_left` / `_right` — both routing
  directions exist (`Term.state_S_routeF` / `Term.state_S_routeG`).
* `fromTerm_unique` — initiality: exactly one structure-respecting map
  out of `Term α` into any target algebra.
* `fromTerm_ext` — **the headline.** Two structure-respecting maps into
  the same algebra are equal. Every three-case induction over `Term`
  whose statement is an equation between two such maps is this theorem
  at some algebra; choosing the algebra is the whole proof.
* `universe_syntax_evaluation_parametric` (the corpus calls it
  `Polymorphic.state_congr`) — requirement 3, evaluation depends on the
  applicative only as a function. A corollary of initiality, not a
  fourth induction.
* `universe_syntax_route_distinction_pointwiseDiv` — under the bulk's
  applicative the two routing directions give different states; the
  third requirement is not vacuous. (In `UniverseSyntaxParity`.)

Requirements 1 and 2 are `TermCore`'s three equation lemmas; requirement 3
is initiality. The module's "three requirements" are its own §2, restated
as the fields of `AlgebraStructure`.

## Open

* **Uniqueness up to isomorphism** — any syntax meeting the three
  requirements is equivalent to `Polymorphic.Term`. Tractable via
  `FreeBulkAlgebra`; not attempted here.
* **Constructor-count minimality as a meta-theorem** — no syntax with
  fewer than 3 constructors meets the requirements. Informal, not
  formalised.

## What is proved here, and what the initiality theorem does not say

* **Proved (native):** `fromTerm_unique` — initiality, the one three-case induction this module
  owns — and everything downstream of it in this file: `fromTerm_ext`,
  `state_eq_fromTerm_stateAlgebra`, `universe_syntax_evaluation_parametric` and
  `fromTerm_homomorphism_factorization` are each `fromTerm_unique` applied once or twice, with no
  further recursion.
* **Proved (native):** read what `fromTerm_unique` is about. It is a **uniqueness theorem about a
  structure-respecting map out of `Term α`**: once `a.inj`, `a.extL`, `a.extR` are fixed, exactly
  one `f` satisfies the three recurrences. It is **not a theorem about semantics**, and it does
  not become one by being composed — `AlgebraStructure` constrains nothing about the three
  operations, so a target algebra is free to inspect any α-value and branch on it. What is forced
  is the recursion; what remains free is everything the three fields do.
* **Proved (riding):** requirements 1 and 2. `universe_syntax_can_introduce`,
  `universe_syntax_can_route_left` and `universe_syntax_can_route_right` are `TermCore`'s
  `Term.state_K` / `Term.state_S_routeF` / `Term.state_S_routeG` ∃-wrapped or returned unchanged.
  The §2 comment above already calls them restatements, not proofs.
* **Open:** *"Three constructors, forced — two lose routing, four are
  redundant."* No statement in this file quantifies over candidate syntaxes. The two facts the
  `AlgebraStructure` docstring leans on — `BulkExhaustiveness.Node.singleRec_trichotomy` for slot
  occupancy and `BulkRoutingForced.routing_selector_forced` for the routing bit — are real
  theorems, but each is about a different type and neither states minimality of a constructor
  count.
* **Open:** uniqueness up to isomorphism — that any syntax meeting the three requirements is
  equivalent to `Term α`. Still open: `FreeBulkAlgebra.hom_toFun_unique` is `BulkTower`'s own
  initiality, a different type's theorem, and nothing in the corpus transports it here.
* **Open:** constructor-count minimality as a meta-theorem. Not formalised anywhere.

-/

noncomputable section

universe u v

namespace Sanguine
namespace UniverseSyntax

open Polymorphic

-- ═══════════════════════════════════════════════════════════════════
-- THE THREE REQUIREMENTS, PROVEN AGAINST Polymorphic.Term
-- ═══════════════════════════════════════════════════════════════════

-- Requirements 1 and 2 are `TermCore`'s three equation lemmas, ∃-wrapped or applied. They are
-- restatements, not proofs: the content is `Term.state`'s own defining equations.

/-- **Requirement 1: atomic introduction.** For any α-values B and U,
    there exists a syntactic term whose state under any applicative
    is the result of applying that applicative to B and U. -/
theorem universe_syntax_can_introduce (α : Type u) (B U : α) :
    ∃ t : Term α, ∀ app : α → α → α, Term.state app t = app B U :=
  ⟨Term.K B U, fun app => Term.state_K app B U⟩

/-- **Requirement 2a: left-routing composition.** A syntactic node
    can represent "recurse in the left operand, leaf in the right." -/
theorem universe_syntax_can_route_left {α : Type u}
    (prev : Term α) (outer : α) (app : α → α → α) :
    Term.state app (Term.S_routeF prev outer)
      = app (Term.state app prev) outer :=
  Term.state_S_routeF app prev outer

/-- **Requirement 2b: right-routing composition.** A syntactic node
    can represent "leaf in the left, recurse in the right operand." -/
theorem universe_syntax_can_route_right {α : Type u}
    (outer : α) (prev : Term α) (app : α → α → α) :
    Term.state app (Term.S_routeG outer prev)
      = app outer (Term.state app prev) :=
  Term.state_S_routeG app outer prev

-- Requirement 3 (`universe_syntax_evaluation_parametric`) is below §2: it is a corollary of
-- initiality, not an independent induction. Its non-vacuity witness under the bulk's `ℝ → ℝ`
-- applicative is a Mathlib proof and lives in `UniverseSyntaxParity.lean`
-- (`universe_syntax_route_distinction_pointwiseDiv`) so this lift file stays Lean-core only.

-- ═══════════════════════════════════════════════════════════════════
-- UNIVERSAL TRANSLATOR: POLYMORPHIC.TERM IS THE FREE ALGEBRA
-- ═══════════════════════════════════════════════════════════════════
--
-- The categorical universal property: `Polymorphic.Term α` is the
-- INITIAL OBJECT in the category of algebras with one binary atom
-- constructor and two recursive route constructors. Any other
-- structure with the same algebraic operations is the image of a
-- UNIQUE function from `Polymorphic.Term`.
--
-- This is the mathematical content of "universal translator." Given
-- any target structure (any format, any universe, any substrate)
-- equipped with the same algebraic operations as Polymorphic.Term,
-- translation from term to target is uniquely determined — no
-- hardcoding required; the translation function is forced by the
-- target's algebraic structure alone.

/-- The algebraic interface a target structure must expose to be a
    valid translation target — three operations matching `Term`'s
    three constructors:

    * `inj` — atomic introduction (`Term.K`)
    * `extL` — left-recursive extension (`Term.S_routeF`)
    * `extR` — right-recursive extension (`Term.S_routeG`)

    Any format, language, or substrate supporting these three is a
    valid target. No hardcoded format knowledge.

    The three are not a design choice. They are the slot-occupancy
    trichotomy of a binary node with at most one recursive slot —
    `(leaf, leaf)` / `(rec, leaf)` / `(leaf, rec)`, proven exhaustive in
    `BulkExhaustiveness.Node.singleRec_trichotomy` — and the `extL`/`extR`
    split is the single forced bit of `BulkRoutingForced`
    (`routing_selector_forced`), which is the non-commutativity of `/`.
    So the module's "three requirements" and these three fields are the
    same three.

    ## What this does NOT say

    This is not the trinity `{NB, B, B/U}` of `TrinityCodex`. That is a
    signature trichotomy on a norm; this is a slot-occupancy trichotomy
    on a node. Same count, different level — the corpus's own fractal law
    is that face labels do not transport across levels, and nothing here
    proves they do. -/
structure AlgebraStructure (α : Type u) (T : Type v) where
  inj : α → α → T
  extL : T → α → T
  extR : α → T → T

/-- **The universal translator.** The translation from
    `Polymorphic.Term α` into any target algebra structure, by
    structural recursion, depending only on the target's three
    operations — never on its format, language, or any other
    property. -/
def fromTerm {α : Type u} {T : Type v}
    (a : AlgebraStructure α T) : Term α → T
  | .K B U => a.inj B U
  | .S_routeF prev outer => a.extL (fromTerm a prev) outer
  | .S_routeG outer prev => a.extR outer (fromTerm a prev)

/-- **Initiality — the translation is unique.** Any `f : Term α → T`
    satisfying the three structural recurrences for a target algebra
    `a` must equal `fromTerm a` everywhere. So every universe-syntax
    expression has exactly one translation into any structure with
    compatible operations — the translation is forced by the target's
    algebra, not chosen. -/
theorem fromTerm_unique {α : Type u} {T : Type v}
    (a : AlgebraStructure α T) (f : Term α → T)
    (hK : ∀ B U, f (Term.K B U) = a.inj B U)
    (hF : ∀ prev outer,
            f (Term.S_routeF prev outer) = a.extL (f prev) outer)
    (hG : ∀ outer prev,
            f (Term.S_routeG outer prev) = a.extR outer (f prev)) :
    ∀ t : Term α, f t = fromTerm a t := by
  intro t
  induction t with
  | K B U => exact hK B U
  | S_routeF prev outer ih =>
      show f (Term.S_routeF prev outer) = a.extL (fromTerm a prev) outer
      rw [hF, ih]
  | S_routeG outer prev ih =>
      show f (Term.S_routeG outer prev) = a.extR outer (fromTerm a prev)
      rw [hG, ih]

/-- **The theorem every three-case induction over `Term` is a copy of.** Two maps out of
    `Term α` respecting the *same* target algebra are equal — initiality applied twice, with
    `fromTerm a` the mediator that cancels. Picking the algebra is the whole proof; there is no
    recursion left to write.

    Read as a law: a `Term` has no content beyond its three constructors, so agreeing on the three
    constructors is agreeing everywhere. Naturality squares are the sharp case — both sides of
    `state (map f t) = f (state t)` are `fromTerm` of one mixed-carrier algebra, so the square is
    not a square, it is uniqueness twice (`Polymorphic.map_state`).

    ## What this does NOT say

    Nothing about recursions over *other* types that happen to share the constructor names: an
    induction over `BulkTower` or `IotaTermAlgebra.IotaTerm` is that type's own initiality
    (`FreeBulkAlgebra.hom_toFun_unique` for the former), a different theorem. And nothing about
    non-algebraic goals — a decidability, a `Prop`-valued predicate that is not an equation
    between two structure-respecting maps, still needs its own induction. -/
theorem fromTerm_ext {α : Type u} {T : Type v}
    (a : AlgebraStructure α T) (f g : Term α → T)
    (fK : ∀ B U, f (Term.K B U) = a.inj B U)
    (fF : ∀ prev outer, f (Term.S_routeF prev outer) = a.extL (f prev) outer)
    (fG : ∀ outer prev, f (Term.S_routeG outer prev) = a.extR outer (f prev))
    (gK : ∀ B U, g (Term.K B U) = a.inj B U)
    (gF : ∀ prev outer, g (Term.S_routeF prev outer) = a.extL (g prev) outer)
    (gG : ∀ outer prev, g (Term.S_routeG outer prev) = a.extR outer (g prev))
    (t : Term α) :
    f t = g t :=
  (fromTerm_unique a f fK fF fG t).trans (fromTerm_unique a g gK gF gG t).symm

/-- **`Term α` is an algebra over itself** — its own three constructors. Any renaming of the
    constructors that round-trips is `fromTerm_ext` at this algebra. -/
def termAlgebra (α : Type u) : AlgebraStructure α (Term α) where
  inj := Term.K
  extL := Term.S_routeF
  extR := Term.S_routeG

/-- **The mixed-carrier algebra.** Read the `α`-leaves through `f : α → β` and compose in `β`.
    `Term α → β` maps that respect it are exactly the ones that transport; this is the algebra at
    which naturality collapses into initiality. -/
def transportAlgebra {α β : Type u} (f : α → β) (appβ : β → β → β) :
    AlgebraStructure α β where
  inj := fun B U => appβ (f B) (f U)
  extL := fun t outer => appβ t (f outer)
  extR := fun outer t => appβ (f outer) t

/-- **`state` is a universal-translation instance.** For any
    applicative `app`, `Term.state app` is the universal translation
    from `Term α` into `α` under the algebra where all three
    operations are `app` itself — `state` is the "translate to base
    type under this applicative" case. -/
def stateAlgebra (α : Type u) (app : α → α → α) : AlgebraStructure α α where
  inj := app
  extL := fun prev outer => app prev outer
  extR := fun outer prev => app outer prev

/-- `state` equals `fromTerm` of the state algebra — one concrete
    case of universal translation. -/
theorem state_eq_fromTerm_stateAlgebra
    {α : Type u} (app : α → α → α) (t : Term α) :
    Term.state app t = fromTerm (stateAlgebra α app) t :=
  fromTerm_unique (stateAlgebra α app) (Term.state app)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) t

/-- **Requirement 3: applicative-parametricity.** Two applicatives that agree pointwise produce
    identical states for any term — the syntactic structure is independent of which binary
    operation is plugged in, only its values matter.

    `Term.state app₁` respects the `app₂` state-algebra (that is exactly the hypothesis `h`), so
    initiality identifies it with `fromTerm (stateAlgebra α app₂)`, which is `Term.state app₂`.
    The corpus's name for this is `Polymorphic.state_congr`; one proof, and it is here. -/
theorem universe_syntax_evaluation_parametric
    {α : Type u} (t : Term α) (app₁ app₂ : α → α → α)
    (h : ∀ a b, app₁ a b = app₂ a b) :
    Term.state app₁ t = Term.state app₂ t :=
  (fromTerm_unique (stateAlgebra α app₂) (Term.state app₁)
      (fun B U => h B U) (fun _prev outer => h _ outer)
      (fun outer _ => h outer _) t).trans
    (state_eq_fromTerm_stateAlgebra app₂ t).symm

/-- **Homomorphism factorisation.** `Polymorphic.Term` is the initial
    algebra: if `h : S → T` respects the three operations, the two
    translations commute through it —

      h (fromTerm source t) = fromTerm target t   for all t

    Translating into S then mapping to T equals translating directly
    into T. `Term α` is the unique mediator; any two compatible
    formats are linked by a unique translation, forced by their
    algebraic compatibility.

    `h ∘ fromTerm source` respects the target algebra, so initiality names it. -/
theorem fromTerm_homomorphism_factorization
    {α : Type u} {S : Type v} {T : Type v}
    (source : AlgebraStructure α S) (target : AlgebraStructure α T)
    (h : S → T)
    (hInj : ∀ B U, h (source.inj B U) = target.inj B U)
    (hL : ∀ s outer, h (source.extL s outer) = target.extL (h s) outer)
    (hR : ∀ outer s, h (source.extR outer s) = target.extR outer (h s))
    (t : Term α) :
    h (fromTerm source t) = fromTerm target t :=
  fromTerm_unique target (fun t => h (fromTerm source t))
    hInj (fun _ outer => hL _ outer) (fun outer _ => hR outer _) t

end UniverseSyntax
end Sanguine
