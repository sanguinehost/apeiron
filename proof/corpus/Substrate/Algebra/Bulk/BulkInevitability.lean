import Substrate.Algebra.Bulk.BulkTower

/-!
# Bulk Inevitability

@name The Building Blocks That Have To Exist
@child If you build *anything* out of the framework's one rule, three little parts show up on their own — a part that holds a constant, and two parts that pass things along. You never have to add them; they grow out of the rule by themselves.
@scientist Any structure satisfying the bulk recursion contains the universal combinator basis — K (constants), S (routing, two orientations), and the routing selector ι — independently of any particular implementation. Read the weight exactly: only the K half is derived (`contains_K` exhibits `base (fun _ => c) (fun _ => 1)` and closes on `c / 1 = c`); the two S halves are the `BulkAlgebra` structure's own `state_stepB`/`state_stepU` FIELDS restated, so S is realized by what the signature says, not by an argument.
@engineer `BulkAlgebra` abstracts carrier + three constructors + state evaluator; `contains_K`, `stepB_realizes_S_bulk`/`stepU_realizes_S_bulk`, and `iota` lift the realizers to *any* such algebra; `inevitability` is the headline bundle.

**Inevitability program, headline.** Any structure satisfying the
bulk recursion — packaged as the abstract `BulkAlgebra` — contains
the universal combinator pattern: K (constants) and S (routing in
two orientations). The routing selector ι follows by construction.

## Why universality matters

`BulkContainsK`, `BulkContainsS`, and `BulkContainsIota` showed
the realizations exist *in BulkTower*. That's consistency:
BulkTower CAN realize K and S.

This file lifts those realizations to *any* algebra with the
bulk's signature, so the combinator structure is not a feature of
`BulkTower`'s particular inductive implementation.

**How much the lift carries.** K is derived: `contains_K` picks the
witness `base (fun _ => c) (fun _ => 1)` and closes on `c / 1 = c`.
S is not — `state_stepB` and `state_stepU` are FIELDS of the
`BulkAlgebra` structure, so "any bulk algebra MUST realize S" holds
because realizing S is what the signature is DEFINED to mean. The
forcing anyone would want sits one level up — that `x = B/U` compels
this signature — and no theorem here or downstream states it;
`fromBulkTower` is the only model exhibited.

## Contents

* `BulkAlgebra` — the abstract structure: carrier + three
  constructor operations + state evaluator, related by the bulk
  axioms (each constructor's state collapses to division).
* `BulkAlgebra.fromBulkTower` — the concrete `BulkTower` IS a
  bulk algebra.
* `BulkAlgebra.contains_K` — any bulk algebra contains a
  constant-`c` realizer for every `c`.
* `BulkAlgebra.stepB_realizes_S_bulk` — the `state_stepB` FIELD returned unchanged, one orientation of S; a restatement of the signature, not derived content. CITATION-EXEMPT: the line records the field restatement, weight stated on the line
* `BulkAlgebra.stepU_realizes_S_bulk` — the `state_stepU` FIELD returned unchanged, the other orientation; same weight. CITATION-EXEMPT: the line records the field restatement, weight stated on the line
* `BulkAlgebra.iota` — the routing selector lifts to any bulk
  algebra. ι is forced once stepB and stepU are present.
* `BulkAlgebra.inevitability` — the headline: any bulk algebra
  contains K (constants), S (both orientations), and the routing
  selector ι.

## What is proved here — the inevitability direction is CLOSED, and what carries its weight

* **Proved (native):** `inevitability` holds for every `A : BulkAlgebra`. This direction is PROVED,
  not open. `IotaBridge`'s header once read "the inevitability direction ... stays open" while this
  theorem existed, and that header was read back as the state of the framework's central descent
  claim; the record is `theory/THE_ONTOLOGY.md §1` under "understating". Anything citing this file
  should cite `inevitability` as closed.
* **Proved (native):** `contains_K` is the conjunct with a construction — it exhibits
  `A.base (fun _ => c) (fun _ => 1)` and discharges the goal by `A.state_base` and `div_one`. It is
  the one thing here that is derived rather than assumed.
* **Retired:** that S's realization is FORCED. `stepB_realizes_S_bulk` and
  `stepU_realizes_S_bulk` are `A.state_stepB` and `A.state_stepU` returned unchanged — the
  `BulkAlgebra` structure's own axiom FIELDS, with no rewriting between hypothesis and conclusion;
  `iota_true_state` / `iota_false_state` are those same fields again through `iota`'s two-branch
  `match`. So two of `inevitability`'s three conjuncts are hypotheses of `BulkAlgebra`, not
  consequences of it. "Any structure satisfying the bulk recursion MUST realize S" is true because
  realizing S is what `BulkAlgebra` is DEFINED to mean.
* **Open:** that "the combinator structure ... is forced by the bulk equation's
  recursive signature itself", i.e. that `BulkAlgebra` is what `x = B/U` compels. `BulkAlgebra` is
  a hand-written structure with three chosen constructor operations and three chosen state axioms;
  no derivation of that signature from the equation occurs here, and no theorem states it is the
  only such signature. `fromBulkTower` is the sole model exhibited, so "any bulk algebra" quantifies
  over a class this file populates once.
* **Anchor:** `@child`'s "If you build *anything* out of the framework's one rule,
  three little parts show up on their own". "Anything" ranges over `BulkAlgebra`, and the three
  parts show up because two of them are axioms of that structure.

-/

noncomputable section

namespace Sanguine

-- ═══════════════════════════════════════════════════════════════════
-- THE ABSTRACT BULK ALGEBRA
-- ═══════════════════════════════════════════════════════════════════

/-- **The abstract bulk algebra signature.** A `carrier` type with
    three constructor operations and a state evaluator, satisfying
    the bulk axioms that tie state to constructors via division.

    Any concrete implementation of the universal equation
    `x = B/U` with recursive composition is a `BulkAlgebra`. The
    axioms enforce that:
    * `base B U`'s state is `B x / U x` pointwise.
    * `stepB prev U_outer`'s state is `prev.state x / U_outer x`
      (S-distribution with prev in the f-slot).
    * `stepU B_outer prev`'s state is `B_outer x / prev.state x`
      (S-distribution with prev in the g-slot). -/
structure BulkAlgebra where
  carrier : Type
  base : (ℝ → ℝ) → (ℝ → ℝ) → carrier
  stepB : carrier → (ℝ → ℝ) → carrier
  stepU : (ℝ → ℝ) → carrier → carrier
  state : carrier → ℝ → ℝ
  state_base : ∀ B U x, state (base B U) x = B x / U x
  state_stepB : ∀ prev U_outer x,
    state (stepB prev U_outer) x = state prev x / U_outer x
  state_stepU : ∀ B_outer prev x,
    state (stepU B_outer prev) x = B_outer x / state prev x

namespace BulkAlgebra

-- ═══════════════════════════════════════════════════════════════════
-- BULKTOWER IS A BULK ALGEBRA
-- ═══════════════════════════════════════════════════════════════════

/-- **`BulkTower` is a bulk algebra.** The inductive `BulkTower`
    with its constructors and state function satisfies the bulk
    algebra signature; all three axioms close by `rfl`. -/
noncomputable def fromBulkTower : BulkAlgebra where
  carrier := BulkTower
  base B U := BulkTower.base { B := B, U := U }
  stepB prev U_outer := BulkTower.stepB prev U_outer
  stepU B_outer prev := BulkTower.stepU B_outer prev
  state := BulkTower.state
  state_base _ _ _ := rfl
  state_stepB _ _ _ := rfl
  state_stepU _ _ _ := rfl

-- ═══════════════════════════════════════════════════════════════════
-- INEVITABILITY OF K
-- ═══════════════════════════════════════════════════════════════════

/-- **Inevitability of K.** Any bulk algebra contains, for every
    real `c`, an element whose state is the constant function `c`.
    Construction: `A.base (fun _ => c) (fun _ => 1)`. The `state_base`
    axiom reduces to `c / 1 = c`. -/
theorem contains_K (A : BulkAlgebra) (c : ℝ) :
    ∃ t : A.carrier, ∀ x, A.state t x = c := by
  refine ⟨A.base (fun _ => c) (fun _ => 1), fun x => ?_⟩
  rw [A.state_base]
  exact div_one c

-- ═══════════════════════════════════════════════════════════════════
-- INEVITABILITY OF S (BOTH ORIENTATIONS)
-- ═══════════════════════════════════════════════════════════════════

/-- **The `state_stepB` field, restated (prev-as-f orientation).**
    The proof term is `A.state_stepB` returned unchanged — a FIELD
    of `structure BulkAlgebra`, not a derivation. "Any bulk algebra
    realizes S with the inner state in the f-slot" holds because
    realizing it is part of what the signature is defined to mean. -/
theorem stepB_realizes_S_bulk
    (A : BulkAlgebra) (prev : A.carrier) (outer : ℝ → ℝ) (x : ℝ) :
    A.state (A.stepB prev outer) x = A.state prev x / outer x :=
  A.state_stepB prev outer x

/-- **The `state_stepU` field, restated (prev-as-g orientation).**
    The proof term is `A.state_stepU` returned unchanged — a FIELD
    of `structure BulkAlgebra`, not a derivation. Same weight as
    `stepB_realizes_S_bulk`, opposite role-assignment. -/
theorem stepU_realizes_S_bulk
    (A : BulkAlgebra) (outer : ℝ → ℝ) (prev : A.carrier) (x : ℝ) :
    A.state (A.stepU outer prev) x = outer x / A.state prev x :=
  A.state_stepU outer prev x

-- ═══════════════════════════════════════════════════════════════════
-- INEVITABILITY OF ι (THE ROUTING SELECTOR)
-- ═══════════════════════════════════════════════════════════════════

/-- **ι in any bulk algebra.** The routing bifurcation selector
    lifts: given a routing bit `b`, an inner element `prev`, and
    an outer function, produce a new element in the selected
    S-orientation. ι is forced once `stepB` and `stepU` are
    present. -/
def iota (A : BulkAlgebra) (b : Bool) (prev : A.carrier)
    (outer : ℝ → ℝ) : A.carrier :=
  match b with
  | true  => A.stepB prev outer
  | false => A.stepU outer prev

/-- The `state_stepB` field again, through `iota`'s definitional
    `match`: `iota true prev outer` reduces to `stepB prev outer`,
    so this is `stepB_realizes_S_bulk` under a second name — a
    restatement, not a derivation. -/
theorem iota_true_state
    (A : BulkAlgebra) (prev : A.carrier) (outer : ℝ → ℝ) (x : ℝ) :
    A.state (A.iota true prev outer) x = A.state prev x / outer x :=
  A.stepB_realizes_S_bulk prev outer x

/-- The `state_stepU` field again, through `iota`'s definitional
    `match`: `iota false prev outer` reduces to `stepU outer prev`,
    so this is `stepU_realizes_S_bulk` under a second name — a
    restatement, not a derivation. -/
theorem iota_false_state
    (A : BulkAlgebra) (outer : ℝ → ℝ) (prev : A.carrier) (x : ℝ) :
    A.state (A.iota false prev outer) x = outer x / A.state prev x :=
  A.stepU_realizes_S_bulk outer prev x

-- ═══════════════════════════════════════════════════════════════════
-- THE HEADLINE: BULK INEVITABILITY
-- ═══════════════════════════════════════════════════════════════════

/-- **Bulk Inevitability (headline).** Any structure satisfying
    the bulk recursion contains:

    * **K** — for every `c : ℝ`, an element whose state is the
      constant function `c`.
    * **S in prev-as-f orientation** — `stepB` realizes
      `S_bulk prev.state outer`.
    * **S in prev-as-g orientation** — `stepU` realizes
      `S_bulk outer prev.state`.

    The routing selector ι follows by definition once both
    orientations are present (see `BulkAlgebra.iota`).

    The combinator structure is not specific to `BulkTower`'s
    inductive implementation — the statement quantifies over every
    `BulkAlgebra`. It is not *derived* from the bulk equation either:
    conjuncts 2 and 3 are `A.state_stepB` and `A.state_stepU`, the
    signature's own axiom fields returned unchanged. Discount those and
    what this theorem establishes is exactly `contains_K` — every bulk
    algebra has, for each `c : ℝ`, an element of constant state `c`. -/
theorem inevitability (A : BulkAlgebra) :
    (∀ c : ℝ, ∃ t : A.carrier, ∀ x, A.state t x = c) ∧
    (∀ (prev : A.carrier) (outer : ℝ → ℝ) (x : ℝ),
        A.state (A.stepB prev outer) x = A.state prev x / outer x) ∧
    (∀ (outer : ℝ → ℝ) (prev : A.carrier) (x : ℝ),
        A.state (A.stepU outer prev) x = outer x / A.state prev x) :=
  ⟨A.contains_K, A.stepB_realizes_S_bulk, A.stepU_realizes_S_bulk⟩

end BulkAlgebra
end Sanguine

end
