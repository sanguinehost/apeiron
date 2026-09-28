import Substrate.Algebra.Bulk.BulkInevitability

/-!
# BulkTower is the Free Bulk Algebra

@name The Master Blueprint Every Copy Comes From
@child There is one original pattern that every version of the building blocks is just a copy of. Pick any version you like, and there is exactly one way to fold the original onto it — so the original is the master blueprint.
@scientist `BulkTower` is the free `BulkAlgebra`: for any target algebra there is a canonical structure-preserving homomorphism from it, and any homomorphism's underlying function equals that canonical one — universality upgraded from existence (K and S exist) to uniqueness.
@engineer `Hom` packages the constructor- and state-preserving maps; `canonical A` builds the unique homomorphism by structural recursion on `BulkTower`, and freeness says any other `Hom`'s `toFun` equals it.

`BulkInevitability` states that any `BulkAlgebra` contains K and S
(universal property as existence) — though only the K half is derived
there; its two S conjuncts are the `state_stepB`/`state_stepU` fields
of the signature restated. This file upgrades to **universality as
uniqueness**: `BulkTower` is the *free* `BulkAlgebra` — for any target
algebra `A`, there is a canonical homomorphism `BulkTower → A`, and any
homomorphism's underlying function equals the canonical one.

Nothing here leans on that S half. `canonical_preserves_state` and
`hom_toFun_unique` are structural inductions over `BulkTower` that use
the `BulkAlgebra` fields directly, so the freeness result stands whole
whatever the inevitability framing is worth.

## What a bulk-algebra homomorphism is

A homomorphism from `BulkTower` to `A : BulkAlgebra` is a function
that:
1. Sends each `base e` to `A.base e.B e.U` (preserves the base
   constructor).
2. Sends each `stepB prev outer` to `A.stepB (image prev) outer`
   (preserves the stepB constructor).
3. Sends each `stepU outer prev` to `A.stepU outer (image prev)`
   (preserves the stepU constructor).
4. Preserves state pointwise: `A.state (image t) p = t.state p`
   for all `t` and `p`.

## The freeness theorem

`canonical A` constructs the unique such homomorphism by
structural recursion on `BulkTower`. It exists for any
`BulkAlgebra A`, and any other homomorphism `h` satisfies
`h.toFun = canonical A`. This is the chain's load-bearing content:
`BulkInevitability` gives one derived fact (`contains_K`) plus two
signature restatements, whereas the two inductions here are earned.
Read the reach exactly — `Hom` already demands `state_preserve`, so
the targets quantified over are those that agree with `BulkTower` on
state, and the uniqueness is of `toFun`, not of `h`.

## What is proved here, and what is read onto the world

* **Proved (native):** `canonical_preserves_state` — by induction on `BulkTower`, the recursive map
  satisfies `A.state (canonical A t) p = BulkTower.state t p` for every `A`, `t`, `p`; each case
  closes on the corresponding `BulkAlgebra` field.
* **Proved (native):** `hom_toFun_unique` — for every `A` and every `h : Hom A`,
  `h.toFun = canonical A`. Induction; the three constructor cases are `h`'s own `*_preserve`
  fields.
* **Proved (native):** `bulkTower_is_free` — `∃ h : Hom A, ∀ h' : Hom A, h'.toFun = h.toFun`, for
  every `A : BulkAlgebra`. Read exactly: existence plus uniqueness **of the underlying function**.
  It is not stated as `h' = h`, and it is initiality in the category of state-preserving `Hom`s,
  not freeness over a named set of generators — there is no generator set in the statement, the
  base constructor takes a whole `UniversalEquation`.
* **Anchor:** that this makes `BulkTower` "the master blueprint every copy comes
  from" / "everything reduces to this". Uniqueness of a structure-respecting map to a target is not
  a statement that the target's meaning is forced; `Hom` already demands `state_preserve`, so the
  targets quantified over are exactly those that agree with `BulkTower` on state.
* **Proved (riding):** the header's premise that `BulkInevitability` "proved that any
  `BulkAlgebra` contains K and S". Its `contains_K` conjunct is derived, but the two S conjuncts of
  `BulkInevitability.inevitability` are the `state_stepB`/`state_stepU` fields of the `BulkAlgebra`
  structure restated — they hold by the signature's definition, not by an argument.

-/

noncomputable section

namespace Sanguine

namespace BulkAlgebra

-- ═══════════════════════════════════════════════════════════════════
-- HOMOMORPHISM STRUCTURE
-- ═══════════════════════════════════════════════════════════════════

/-- A homomorphism from `BulkTower` to a `BulkAlgebra A`: a carrier
    map preserving the three constructors and the state function. -/
structure Hom (A : BulkAlgebra) where
  toFun : BulkTower → A.carrier
  base_preserve : ∀ B U,
    toFun (BulkTower.base ⟨B, U⟩) = A.base B U
  stepB_preserve : ∀ prev outer,
    toFun (BulkTower.stepB prev outer) = A.stepB (toFun prev) outer
  stepU_preserve : ∀ outer prev,
    toFun (BulkTower.stepU outer prev) = A.stepU outer (toFun prev)
  state_preserve : ∀ t p,
    A.state (toFun t) p = BulkTower.state t p

-- ═══════════════════════════════════════════════════════════════════
-- THE CANONICAL HOMOMORPHISM
-- ═══════════════════════════════════════════════════════════════════

/-- The canonical map `BulkTower → A.carrier`, defined by structural
    recursion: each constructor is sent to the corresponding
    constructor of `A`. -/
def canonical (A : BulkAlgebra) : BulkTower → A.carrier
  | .base e => A.base e.B e.U
  | .stepB prev outer => A.stepB (canonical A prev) outer
  | .stepU outer prev => A.stepU outer (canonical A prev)

/-- **State preservation.** The canonical map sends each
    BulkTower's state to the same state in `A`, pointwise. Proved
    by structural induction; each case closes via the
    corresponding `A.state_*` axiom. -/
theorem canonical_preserves_state (A : BulkAlgebra) :
    ∀ (t : BulkTower) (p : ℝ),
      A.state (canonical A t) p = BulkTower.state t p := by
  intro t
  induction t with
  | base e =>
      intro p
      show A.state (A.base e.B e.U) p = e.B p / e.U p
      exact A.state_base e.B e.U p
  | stepB prev outer ih =>
      intro p
      show A.state (A.stepB (canonical A prev) outer) p =
           BulkTower.state prev p / outer p
      rw [A.state_stepB, ih p]
  | stepU outer prev ih =>
      intro p
      show A.state (A.stepU outer (canonical A prev)) p =
           outer p / BulkTower.state prev p
      rw [A.state_stepU, ih p]

/-- The canonical map IS a homomorphism. Existence half of
    universality. -/
def canonicalHom (A : BulkAlgebra) : Hom A where
  toFun := canonical A
  base_preserve _ _ := rfl
  stepB_preserve _ _ := rfl
  stepU_preserve _ _ := rfl
  state_preserve := canonical_preserves_state A

-- ═══════════════════════════════════════════════════════════════════
-- UNIQUENESS
-- ═══════════════════════════════════════════════════════════════════

/-- **Uniqueness of homomorphism's underlying function.** Any
    homomorphism `h : Hom A` has the same underlying function as
    `canonical A`. Proved by structural induction; the constructor
    cases use the corresponding `*_preserve` axioms of `h`. -/
theorem hom_toFun_unique (A : BulkAlgebra) (h : Hom A) :
    h.toFun = canonical A := by
  funext t
  induction t with
  | base e =>
      show h.toFun (BulkTower.base ⟨e.B, e.U⟩) = A.base e.B e.U
      exact h.base_preserve e.B e.U
  | stepB prev outer ih =>
      rw [h.stepB_preserve prev outer, ih]
      rfl
  | stepU outer prev ih =>
      rw [h.stepU_preserve outer prev, ih]
      rfl

-- ═══════════════════════════════════════════════════════════════════
-- THE FREENESS THEOREM
-- ═══════════════════════════════════════════════════════════════════

/-- **BulkTower is the free BulkAlgebra (headline).** For any
    target `BulkAlgebra A`:

    1. There exists a homomorphism `BulkTower → A` (the canonical
       one).
    2. Any homomorphism's underlying function equals the canonical
       map.

    This is the universal property of freeness, upgraded from
    `BulkInevitability`'s "K and S exist" to "BulkTower is the
    universal witness — every bulk algebra is determined by a
    unique map from BulkTower." -/
theorem bulkTower_is_free (A : BulkAlgebra) :
    ∃ (h : Hom A), ∀ (h' : Hom A), h'.toFun = h.toFun :=
  ⟨canonicalHom A, fun h' => by
    rw [hom_toFun_unique A h']
    rfl⟩

end BulkAlgebra
end Sanguine
