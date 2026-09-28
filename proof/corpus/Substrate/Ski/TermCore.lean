-- Pure Lean core (zero Mathlib): the polymorphic bulk `Term` and its evaluation `state`. This is
-- the free bulk algebra's carrier, used by the K=4 evaluator's HP dispatch, so it carries NO
-- Mathlib — the OS spec stands on Lean core alone. The Mathlib-backed sibling
-- (`PolyIotaTermAlgebra`, the bulk's `ℝ → ℝ` applicative and the `BulkTower` bridge) re-imports this
-- and proves the parity (`Term.state pointwiseDiv = BulkTower.state`); it never sits under it.

/-!
# The polymorphic bulk Term — `Term`, `state` (pure)
@name The One Shape Everything Is Written In
@child Every program, file, or message can be written as the same little tree of three pieces — and you read it by folding one rule over it. What it *means* is the rule you choose.
@scientist `Term α` is the free bulk algebra's carrier over any carrier `α`; `state app` folds it under any binary applicative — the bulk is `α = (ℝ→ℝ)`, `app B U p := B p / U p`.
@engineer The `inductive Term` has three constructors (`K`, `S_routeF`, `S_routeG`); `state` folds them and `map` is the functorial transport — pure Lean, zero Mathlib (the OS spec stands on this alone).

A term over an arbitrary carrier `α`, named with the three combinator-theoretic constructors
(`K`, `S_routeF`, `S_routeG`). Evaluation `state app` folds the term under an arbitrary binary
applicative `app : α → α → α`. The bulk's setting is recovered by `α = (ℝ → ℝ)` and
`app B U p := B p / U p`, proven in `PolyIotaTermAlgebra` (the parity check).

## What this does NOT say

`state` and `map` are two hand-written recursions, not two instances of one fold. The fold is
`UniverseSyntax.fromTerm`, its initiality is `UniverseSyntax.fromTerm_unique`, and the form the
corpus actually uses is `UniverseSyntax.fromTerm_ext` (two structure-respecting maps into one
algebra are equal). All three sit one import hop ABOVE this file, so the root cannot reach them:
`state` is identified with the fold there (`state_eq_fromTerm_stateAlgebra`), `state_congr` and
`map_state` are `fromTerm_ext` at the state- and transport-algebras, and `map` is fed to
`fromTerm_unique` from further up (`UniversalAdapter`). Nothing here is the floor of that descent
— moving `AlgebraStructure` / `fromTerm` / `fromTerm_unique` down to this file, and defining
`state` and `map` as the two instances, is the open consolidation (130 `fromTerm` + 51
`AlgebraStructure` references, and `AlgebraStructure.precompose` is declared inside
`namespace UniverseSyntax`, so a bare `export` breaks its dot-notation).

## What is proved here vs what it rides on

* **Proved (native):** nothing beyond the definitions and their own equations. `Term` (three
  constructors), `state` (the fold under an arbitrary binary applicative) and `map` are `def`s;
  `state_K`, `state_S_routeF`, `state_S_routeG`, `map_K`, `map_S_routeF` and `map_S_routeG` are
  re-exports of the elaborator's equations (`state.eq_1..3`, `map.eq_1..3`) for the `rw` /
  `simp only` sites that call them by name — not one of them is a second proof. Pure Lean core,
  zero Mathlib.
* **Proved (riding):** that `state` IS the universal fold, and that the fold is unique —
  `UniverseSyntax.state_eq_fromTerm_stateAlgebra`, `UniverseSyntax.fromTerm_unique`,
  `UniverseSyntax.fromTerm_ext`, all one import hop ABOVE this file, so this root cannot reach
  them. `state_congr` and `map_state` are `fromTerm_ext` at the state- and transport-algebras,
  and live there too.
* **Proved (riding):** the parity with the bulk's `ℝ → ℝ` applicative —
  `PolyIotaTermAlgebra` proves `Term.state pointwiseDiv = BulkTower.state`. This file states no
  bulk instance and no `ℝ`.
* **Open:** the consolidation — moving `AlgebraStructure` / `fromTerm` / `fromTerm_unique` down
  to this file and defining `state` and `map` as the two instances (130 `fromTerm` + 51
  `AlgebraStructure` references, and `AlgebraStructure.precompose` sits inside
  `namespace UniverseSyntax`, so a bare `export` breaks its dot-notation).

-/

universe u

namespace Sanguine
namespace Polymorphic

/-- A term over an arbitrary carrier type `α`, named with the three
    combinator-theoretic constructors. -/
inductive Term (α : Type u) : Type u where
  | K (B U : α) : Term α
  | S_routeF (prev : Term α) (outer : α) : Term α
  | S_routeG (outer : α) (prev : Term α) : Term α
deriving Hashable, BEq

namespace Term

/-- Evaluation under an arbitrary binary applicative `app : α → α → α`.

    Each constructor reduces to `app` at the corresponding role-assignment. The bulk's setting is
    recovered by taking `α = (ℝ → ℝ)` and `app B U p := B p / U p`. -/
def state {α : Type u} (app : α → α → α) : Term α → α
  | .K B U => app B U
  | .S_routeF prev outer => app (state app prev) outer
  | .S_routeG outer prev => app outer (state app prev)

-- The simp table for `state` and `map` is the elaborator's own equations (`Term.state.eq_1..3`,
-- `map.eq_1..3`), registered below at the definitions. The six named rows in this file are
-- re-exports of those equations for the `rw` / `simp only` sites that call them by name; not one
-- of them is a second proof.
attribute [simp] state

theorem state_K {α : Type u} (app : α → α → α) (B U : α) :
    state app (.K B U) = app B U := state.eq_1 app B U

theorem state_S_routeF {α : Type u} (app : α → α → α)
    (prev : Term α) (outer : α) :
    state app (.S_routeF prev outer) = app (state app prev) outer :=
  state.eq_2 app prev outer

theorem state_S_routeG {α : Type u} (app : α → α → α)
    (outer : α) (prev : Term α) :
    state app (.S_routeG outer prev) = app outer (state app prev) :=
  state.eq_3 app outer prev

end Term

/-- **Functorial transport.** A function `f : α → β` lifts to a structural map on terms. -/
def map {α : Type u} {β : Type u} (f : α → β) : Term α → Term β
  | .K B U => .K (f B) (f U)
  | .S_routeF prev outer => .S_routeF (map f prev) (f outer)
  | .S_routeG outer prev => .S_routeG (f outer) (map f prev)

attribute [simp] map

theorem map_K {α : Type u} {β : Type u} (f : α → β) (B U : α) :
    map f (.K B U) = .K (f B) (f U) := map.eq_1 f B U

theorem map_S_routeF {α : Type u} {β : Type u} (f : α → β)
    (prev : Term α) (outer : α) :
    map f (.S_routeF prev outer) = .S_routeF (map f prev) (f outer) :=
  map.eq_2 f prev outer

theorem map_S_routeG {α : Type u} {β : Type u} (f : α → β)
    (outer : α) (prev : Term α) :
    map f (.S_routeG outer prev) = .S_routeG (f outer) (map f prev) :=
  map.eq_3 f outer prev

end Polymorphic
end Sanguine
