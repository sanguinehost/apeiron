import Substrate.Algebra.Bulk.BulkTower
import Substrate.Algebra.Bulk.FreeBulkAlgebra

/-!
# The Iota Term Algebra (specialized to applicative `/`)
@name One Brick Builds the Rest
@child There's a single building block (called Iota) that, by being applied to itself, generates all the others you need. Here it's pinned to ordinary numbers with division as the way things combine.
@scientist The Iota calculus generates K and S by self-application; this file specializes Iota's term algebra to the bulk setting — ℝ-typed, with `/` as the explicit applicative — built from three constructors `K`/`S_routeF`/`S_routeG`.
@engineer `fromBulkTower`/`toBulkTower` are mutual inverses preserving `state`, proving the BulkTower IS Iota's term algebra at applicative `/` (the Q1 closure); the three constructors are the BulkTower constructors renamed to show the combinator structure.

The Iota calculus generates K and S by self-application. Standard
SKI Iota is type-polymorphic, with generic juxtaposition as the
implicit applicative. This file specializes Iota's term algebra
to the bulk's setting: ℝ-typed, with `/` as the explicit
applicative.

## The Iota term algebra at applicative `/`

A term in this algebra is built from three constructors:

* `K B U` — the Iota K-shape: a "base" carrying a numerator-`B`
  and denominator-`U` pair.
* `S_routeF prev outer` — S in the prev-as-f orientation: the
  inner term's state goes into the numerator, `outer` into the
  denominator.
* `S_routeG outer prev` — S in the prev-as-g orientation: the
  inner term's state goes into the denominator, `outer` into the
  numerator.

These are exactly the three BulkTower constructors with new names
that make the Iota combinator structure visible. The state
function evaluates a term to a real-valued function ℝ → ℝ by
recursive division.

## Q1 equivalence at the term level

`fromBulkTower` and `toBulkTower` are mutual inverses, and the
`state` function is preserved by both. This is the concrete
bijection between BulkTower and Iota's term algebra at applicative
`/` — the formal closure of Q1 in this setting.

What this captures: the BulkTower IS Iota's term algebra
specialized to `/`. The two names denote the same recursive
structure, with the constructor labeling reflecting the
combinator-theoretic reading on one side and the bulk-equation
reading on the other.

What this does NOT directly capture: the type-polymorphic standard
SKI Iota calculus over generic juxtaposition. That object lives at
a different applicative and would require additional infrastructure
(the polymorphic applicative-parametric category) to compare
formally. The present file gives the equivalence at the bulk's
applicative.

## What is proved here vs what the name "Iota" is doing

* **Proved (native):** `fromBulkTower` and `toBulkTower` are mutual inverses
  (`fromBulkTower_toBulkTower`, `toBulkTower_fromBulkTower`) and both preserve `state` at
  every point (`state_eq_BulkTower_state`, `BulkTower_state_eq_state`) — the Q1 closure at
  applicative `/`, by structural induction over the three constructors.
* **Anchor:** that this three-constructor algebra IS "the Iota calculus".
  `K`, `S_routeF` and `S_routeG` are the `BulkTower` constructors renamed to make the
  combinator reading visible; nothing here carries `ι x → x S K`, `K x y → x` or
  `S x y z → x z (y z)`. The reduction-surface facts about iota are
  `GoldbergIotaChains.iota_surface_chains`.
* **Open:** the type-polymorphic standard SKI Iota calculus over generic juxtaposition —
  the file's own "What this does NOT directly capture". `PolyIotaTermAlgebra` pulls the
  carrier and the applicative out, and `SKIReductionBridge` ties `state` to constructor
  dispatch on `Polymorphic.Term`, but that file's own scope note says it does NOT prove
  normalisation of raw SKI/`Comb` expressions. The comparison is still not made.

-/

noncomputable section

namespace Sanguine

namespace IotaTermAlgebra

-- ═══════════════════════════════════════════════════════════════════
-- THE IOTA TERM TYPE
-- ═══════════════════════════════════════════════════════════════════

/-- An Iota term at the bulk's applicative `/`. Three constructors
    name the combinator-theoretic reading of BulkTower's structure:

    * `K B U` — the K-shape carrier (boundary B, drag U).
    * `S_routeF prev outer` — S with the inner term in the f-slot.
    * `S_routeG outer prev` — S with the inner term in the g-slot. -/
inductive IotaTerm : Type where
  | K (B U : ℝ → ℝ) : IotaTerm
  | S_routeF (prev : IotaTerm) (outer : ℝ → ℝ) : IotaTerm
  | S_routeG (outer : ℝ → ℝ) (prev : IotaTerm) : IotaTerm

/-- Evaluation of an Iota term via the bulk's applicative `/`.
    Each constructor reduces to a division at the corresponding
    role-assignment. -/
def state : IotaTerm → ℝ → ℝ
  | .K B U => fun p => B p / U p
  | .S_routeF prev outer => fun p => state prev p / outer p
  | .S_routeG outer prev => fun p => outer p / state prev p

-- ═══════════════════════════════════════════════════════════════════
-- THE BIJECTION WITH BULKTOWER
-- ═══════════════════════════════════════════════════════════════════

/-- Map a `BulkTower` to its Iota-term representation by renaming
    constructors. Structural recursion. -/
def fromBulkTower : BulkTower → IotaTerm
  | .base e => .K e.B e.U
  | .stepB prev outer => .S_routeF (fromBulkTower prev) outer
  | .stepU outer prev => .S_routeG outer (fromBulkTower prev)

/-- Map an Iota term back to a `BulkTower`. The inverse of
    `fromBulkTower`. -/
def toBulkTower : IotaTerm → BulkTower
  | .K B U => .base ⟨B, U⟩
  | .S_routeF prev outer => .stepB (toBulkTower prev) outer
  | .S_routeG outer prev => .stepU outer (toBulkTower prev)

-- ═══════════════════════════════════════════════════════════════════
-- INVERSE THEOREMS
-- ═══════════════════════════════════════════════════════════════════

/-- `fromBulkTower ∘ toBulkTower = id` on `IotaTerm`. Round-trip
    via BulkTower returns the original Iota term. -/
theorem fromBulkTower_toBulkTower (t : IotaTerm) :
    fromBulkTower (toBulkTower t) = t := by
  induction t with
  | K B U => rfl
  | S_routeF prev outer ih =>
      simp only [toBulkTower, fromBulkTower, ih]
  | S_routeG outer prev ih =>
      simp only [toBulkTower, fromBulkTower, ih]

/-- `toBulkTower ∘ fromBulkTower = id` on `BulkTower`. Round-trip
    via the Iota term returns the original tower. -/
theorem toBulkTower_fromBulkTower (t : BulkTower) :
    toBulkTower (fromBulkTower t) = t := by
  induction t with
  | base e => rfl
  | stepB prev outer ih =>
      simp only [fromBulkTower, toBulkTower, ih]
  | stepU outer prev ih =>
      simp only [fromBulkTower, toBulkTower, ih]

-- ═══════════════════════════════════════════════════════════════════
-- STATE PRESERVATION
-- ═══════════════════════════════════════════════════════════════════

/-- The Iota term's state agrees with the corresponding BulkTower's
    state at every point. The bijection is more than syntactic; it
    preserves the semantic function. -/
theorem state_eq_BulkTower_state (t : BulkTower) (p : ℝ) :
    state (fromBulkTower t) p = BulkTower.state t p := by
  induction t with
  | base e => rfl
  | stepB prev outer ih =>
      show state (fromBulkTower prev) p / outer p =
           BulkTower.state prev p / outer p
      rw [ih]
  | stepU outer prev ih =>
      show outer p / state (fromBulkTower prev) p =
           outer p / BulkTower.state prev p
      rw [ih]

/-- Symmetric: the BulkTower's state for `toBulkTower t` agrees
    with the Iota term `t`'s state. -/
theorem BulkTower_state_eq_state (t : IotaTerm) (p : ℝ) :
    BulkTower.state (toBulkTower t) p = state t p := by
  induction t with
  | K B U => rfl
  | S_routeF prev outer ih =>
      show BulkTower.state (toBulkTower prev) p / outer p =
           state prev p / outer p
      rw [ih]
  | S_routeG outer prev ih =>
      show outer p / BulkTower.state (toBulkTower prev) p =
           outer p / state prev p
      rw [ih]

end IotaTermAlgebra
end Sanguine
