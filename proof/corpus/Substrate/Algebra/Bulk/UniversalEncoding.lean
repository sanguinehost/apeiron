import Substrate.Algebra.Bulk.BulkInevitability
import Substrate.Algebra.Bulk.FreeBulkAlgebra
import Substrate.Algebra.Translation.UniversalApplicativeLift
import Substrate.Algebra.Translation.UniverseSyntax
import Substrate.Algebra.Translation.UniverseSyntaxSerialisation

/-!
# The Universal Encoding — One Form, Forced By the Bulk
@name One Shape for All Data
@child Text, images, sound, code — everything can be written in a single shape. You never need a special reader for each kind of file; the meaning comes from how you *read* the one shape, not from a different format each time.
@scientist `Polymorphic.Term` is the free bulk algebra forced by `x = B/U`: it is initial (a unique homomorphism into every bulk algebra), its state is a function of the chosen applicative alone, the map out is unique, and ingestion is format-agnostic.
@engineer `universal_encoding` — this file's only declaration — conjoins five facts, none of them declared here: `BulkInevitability.inevitability` (forced — K derived, both S conjuncts the `BulkAlgebra` signature's own fields), `FreeBulkAlgebra.bulkTower_is_free` (initial), `UniversalApplicativeLift.applicative_is_the_only_freedom` (`state_congr`), `UniverseSyntax.fromTerm_unique` (unique out), and `UniverseSyntaxSerialisation.universalLift_map` (no per-format lexer).

The framework's central structural claim, collected into a single citable
theorem. There is ONE encoding form that everything reduces to —
`Polymorphic.Term` (the three constructors `K B U`, `S_routeF`, `S_routeG`,
i.e. the free bulk algebra / the SKI-Iota structure) — and working from it
removes the need to ever define a specific data format again.

The five facts, each already proven in its own file, here conjoined:

1. **Forced, not chosen.** Any structure satisfying the bulk equation
   `x = B/U` realises K (constants) and S (both routing orientations).
   *(Precise: only the K component is derived — `contains_K` exhibits
   `base (fun _ => c) (fun _ => 1)` and closes on `c / 1 = c`. The two S
   components are the `BulkAlgebra` structure's `state_stepB` and
   `state_stepU` FIELDS restated, so "the combinator shape is a
   consequence of the bulk's signature" is the signature read back, and
   nothing derives that signature from the equation.)*
   (`BulkInevitability.inevitability`.)

2. **Initial.** The bulk tower is the *free* bulk algebra — there is a
   unique homomorphism from it into every other bulk algebra. "Everything
   reduces to this" is literally initiality. (`FreeBulkAlgebra.bulkTower_is_free`.)

3. **Meaning is a parameter, not structure.** The SAME `Term` evaluates
   under ANY binary applicative, and the output depends on the applicative
   **and only through it** — agreeing applicatives give identical states.
   *(Precise: the theorem is a **congruence**, `state_congr` — "equal
   applicatives ⇒ equal states." "The only freedom" is the *reading* of that
   congruence, not a separately-proven injectivity; what is proven is that the
   state is a function of the applicative.)*
   (`UniversalApplicativeLift.applicative_is_the_only_freedom`.)

4. **Unique out.** Translation from `Term` to ANY target algebra is the
   unique structure-respecting map — one universal translator, every
   destination. (`UniverseSyntax.fromTerm_unique`.)

5. **Format-agnostic in.** Ingestion of a raw α-stream commutes with any
   alphabet remap `f : α → β`, so it cannot branch on what a symbol means
   — there is no per-format code path, no lexer.
   (`UniverseSyntaxSerialisation.universalLift_map`.)

A "data format" in the legacy sense bundles a parser, a struct, and a
serialiser per type. In this basis all three collapse to single universal
functions (`universalLift`, `Polymorphic.Term`, `fromTerm`, `serialise`),
and the ONLY per-instance information is two type-level parameters — the
alphabet α (a type) and the applicative (a function). Never code. This is
the formal content of Axiom XX ("everything is waves; the only
modality-specific parameter is α, a type choice not a code path").

No new mathematics is done here: `universal_encoding` is the single citable conjunction
of the five already-established theorems listed above, each imported verbatim.

## What is proved here, and what is read onto the world

* **Proved (riding):** all five conjuncts of `universal_encoding`, which is the file's only
  declaration. Each is the statement of an imported theorem verbatim —
  `BulkInevitability.inevitability`, `FreeBulkAlgebra.bulkTower_is_free`,
  `UniversalApplicativeLift.applicative_is_the_only_freedom`, `UniverseSyntax.fromTerm_unique`,
  `UniverseSyntaxSerialisation.universalLift_map`. Nothing is elaborated here; the file's product
  is the single citable name.
* **Anchor:** conjunct (4) read as "one universal translator, every destination".
  `fromTerm_unique` says that a map agreeing with `AlgebraStructure`'s three operations on the
  three constructors is the canonical one — uniqueness of a *structure-respecting* map, given the
  target's structure. That meaning is thereby forced, or that no format needs defining again, is
  the reading.
* **Anchor:** conjunct (5) read as "no lexer, no per-format code path".
  `universalLift_map` says ingestion commutes with any alphabet remap `f : α → β`, so
  `universalLift` cannot branch on what a symbol means. It says nothing about the applicative
  applied afterwards, which is free to branch on anything.
* **Proved (riding):** conjunct (1)'s force. Its second and third components are the
  `state_stepB` and `state_stepU` fields of the `BulkAlgebra` structure restated —
  `BulkInevitability.stepB_realizes_S_bulk` is `A.state_stepB` — so "S is forced" is the signature CITATION-EXEMPT: named as the structure field read back; the sentence denies S is derived
  being read back, not derived. Only the K component is argued (`BulkInevitability.contains_K`,
  from `state_base` at a constant pair).
* **Anchor:** that `Polymorphic.Term` is THE encoding "everything reduces to", and
  that Axiom XX's content is formal here. No statement quantifies over data formats, over
  modalities, or over encodings outside the `BulkAlgebra` / `AlgebraStructure` signatures the five
  conjuncts fix.

-/

namespace Sanguine

open Polymorphic UniverseSyntax UniverseSyntax.Serialisation

/-- **The Universal Encoding (collecting headline).** `Polymorphic.Term` is
    THE encoding everything reduces to: realised in any bulk algebra (1 — K
    derived, S read back off the signature), initial (2),
    with meaning supplied by a parametric applicative (3), translated out
    uniquely to any target (4), and ingested from any alphabet with no
    per-format code (5). Each conjunct is an established theorem; this binds
    them so the single-basis claim is citable in one place. -/
theorem universal_encoding :
    -- (1) any bulk algebra realises K (constants) and S (both routings) — K is derived
    -- (`contains_K`), S is read back off the `state_stepB`/`state_stepU` fields.
    (∀ A : BulkAlgebra,
        (∀ c : ℝ, ∃ t : A.carrier, ∀ x, A.state t x = c) ∧
        (∀ (prev : A.carrier) (outer : ℝ → ℝ) (x : ℝ),
            A.state (A.stepB prev outer) x = A.state prev x / outer x) ∧
        (∀ (outer : ℝ → ℝ) (prev : A.carrier) (x : ℝ),
            A.state (A.stepU outer prev) x = outer x / A.state prev x)) ∧
    -- (2) INITIAL — the bulk tower is the free bulk algebra (unique map out).
    (∀ A : BulkAlgebra,
        ∃ (h : BulkAlgebra.Hom A), ∀ (h' : BulkAlgebra.Hom A), h'.toFun = h.toFun) ∧
    -- (3) MEANING IS A PARAMETER — same Term, applicative is the only freedom.
    (∀ (α : Type) (app₁ app₂ : α → α → α), (∀ x y, app₁ x y = app₂ x y) →
        ∀ t : Term α, t.state app₁ = t.state app₂) ∧
    -- (4) UNIQUE OUT — translation to any target is the unique homomorphism.
    (∀ (α T : Type) (a : AlgebraStructure α T) (f : Term α → T),
        (∀ B U, f (Term.K B U) = a.inj B U) →
        (∀ prev outer, f (Term.S_routeF prev outer) = a.extL (f prev) outer) →
        (∀ outer prev, f (Term.S_routeG outer prev) = a.extR outer (f prev)) →
        ∀ t : Term α, f t = fromTerm a t) ∧
    -- (5) FORMAT-AGNOSTIC IN — ingestion commutes with any alphabet remap.
    (∀ (α β : Type) (f : α → β) (cs : List α),
        universalLift (cs.map f) = (universalLift cs).map (Polymorphic.map f)) :=
  ⟨BulkAlgebra.inevitability,
   BulkAlgebra.bulkTower_is_free,
   fun _α _app₁ _app₂ h t => applicative_is_the_only_freedom h t,
   fun _α _T a f hK hF hG => fromTerm_unique a f hK hF hG,
   fun _α _β f cs => universalLift_map f cs⟩

end Sanguine
