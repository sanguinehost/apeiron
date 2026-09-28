import Substrate.Algebra.Bulk.BulkTower
import Substrate.Algebra.Bulk.BulkInevitability
import Substrate.Algebra.Translation.UniversalTranslation

/-!
# The bulk contains K, S, ι — the inevitability program in one file
@name Three Tools, Already Inside
@child Computers are built from a tiny set of basic moves. We show our single rule already *contains* those moves — we don't add them, we point to them already sitting inside the rule.
@scientist The constructive realisation of `BulkInevitability` at the ℝ-instance `BulkTower`: the three combinators K, S, ι are each *exhibited* inside the bulk tower and proven to act as their `UniversalTranslation` counterparts — no combinator basis imported.
@engineer K = `base`, a constant-state tower (`K_tower_realizes_K_comb`, depth 0); S = `stepB`/`stepU`, the routing bifurcation, read off `BulkAlgebra.fromBulkTower`'s axioms; ι is `BulkAlgebra.iota` at that instance. `depth_zero_no_steps` gives both counters from the depth. The basis isn't assumed, it's exhibited.

`BulkInevitability` states the headline ("any bulk algebra realises K, S, ι"); this is the
**constructive realisation** at the ℝ-instance `BulkTower`, the three steps that were one
idea split across three files (`BulkContainsK` + `BulkContainsS` + `BulkContainsIota`, all
in namespace `Sanguine`). No combinator basis is imported — each combinator is *exhibited*
inside `BulkTower` and proven to act as its `UniversalTranslation` counterpart.

The instance is `BulkAlgebra.fromBulkTower`, and §2/§3 are taken through it rather than
re-derived: `stepB_realizes_S_bulk`/`stepU_realizes_S_bulk`/`iota_true_state`/
`iota_false_state` already hold in *any* bulk algebra, and `BulkTower` is one. Restating
them here as `rfl` would record the tower's definition a second time.

## §1 — K is `base` (a constant-state tower)
`K_tower c` is the base equation with constant-`c` boundary and constant-`1` drag; its
state is `c` everywhere (`K_tower_state`), which IS `K_comb c` (`K_tower_realizes_K_comb`).
K sits at the bottom, depth 0 — no recursive machinery. The two transition counters are
not separate facts: `total_steps_eq_depth` partitions the depth, so depth 0 forces both
(`depth_zero_no_steps`).

## §2 — S is `stepB`/`stepU` (the routing bifurcation)
S's shape — distribute `x` to two paths, combine — specialised to the bulk's binary
operation `/`: `S_bulk f g x = f x / g x`. The two recursive constructors **are** S in its
two role-assignments (`stepB_is_S_bulk`, `stepU_is_S_bulk`, both the abstract algebra's
`state_stepB`/`state_stepU` axioms at `fromBulkTower`); there is no third recursive
constructor, so every recursive step is an S-distribution (`recursive_step_is_S_bulk`).

## §3 — ι is the routing selector
`iota_bulk b prev outer` IS `BulkAlgebra.iota` at `fromBulkTower` — one selector, not a
second copy. It chooses the orientation (`b=true → stepB`, `b=false → stepU`): the bulk's
universal single primitive, the routing bit from which both S-orientations derive
(`iota_bulk_true_state`/`iota_bulk_false_state`). With `base`, ι generates every tower
(`iota_bulk_generates_BulkTower`: `BulkTower = base ⊕ iota_bulk`).


*(Consolidation: merges the former `BulkContainsK` + `BulkContainsS` + `BulkContainsIota` —
the K→S→ι inevitability chain, all in namespace `Sanguine`. Every theorem name is unchanged.)*

## What is proved here, and what "contains" ranges over

* **Proved (native):** the K half. `K_tower_state_apply` / `K_tower_state` (`c / 1 = c`),
  `K_tower_realizes_K_comb` / `K_tower_state_eq_K_comb`, `K_tower_depth` (`rfl`),
  `depth_zero_no_steps` (`omega` over `BulkTower.total_steps_eq_depth`) and
  `K_tower_organization`. `K_tower_realizes_K_comb` unfolds `UniversalTranslation.K_comb`, whose
  definition is `K_comb (x : α) (_ : β) := x`, so the theorem is `c / 1 = c` again with a name
  attached.
* **Proved (native):** `S_bulk_swap_eq_iff` (`div_eq_div_iff`), `K_tower_through_stepB` /
  `K_tower_through_stepU`, `iota_bulk_true_is_stepB` / `iota_bulk_false_is_stepU` (`rfl`),
  `iota_bulk_with_K_tower_true` / `iota_bulk_with_K_tower_false`, and
  `iota_bulk_generates_BulkTower`.
* **Proved (riding):** the S and ι half is not earned here and the file says so. `stepB_is_S_bulk`,
  `stepU_is_S_bulk`, `iota_bulk_true_state`, `iota_bulk_false_state` and the bundle
  `recursive_step_is_S_bulk` are, in `BulkInevitability.lean`,
  `BulkAlgebra.stepB_realizes_S_bulk` / `BulkAlgebra.stepU_realizes_S_bulk`, CITATION-EXEMPT: named as the axiom fields this file rides; the bullet says nothing here re-earns them
  `BulkAlgebra.iota_true_state` and CITATION-EXEMPT: named as the axiom fields this file rides; the bullet says nothing here re-earns them
  `BulkAlgebra.iota_false_state` at `fromBulkTower` — and each of CITATION-EXEMPT: named as the axiom fields this file rides; the bullet says nothing here re-earns them
  those is a projection of the `BulkAlgebra` structure's own axiom field. "S is realized" is what
  the abstract signature ASSUMES; nothing derives it.
* **Proved (native):** `iota_bulk_generates_BulkTower` is exhaustiveness over CONSTRUCTORS, not over
  computations. It is `cases t` on the three-constructor inductive `BulkTower`, so
  "`BulkTower = base ⊕ iota_bulk`" says the type has one non-recursive and two recursive
  constructors and a `Bool` indexes the latter. It does not quantify over algebras, programs or
  terms of any other type.
* **Anchor:** that this exhibits the combinators computers are built from — "we show
  our single rule already *contains* those moves". What is exhibited is one point of the algebra
  (`base (fun _ => c) (fun _ => 1)`, a constant state) and two constructor equations. Combinatory
  completeness — that `{K, S}` generates the computable functions — is neither stated nor used
  anywhere in this file, so "contains K and S" is containment of two named shapes, not of what those
  names denote in combinatory logic.

-/

noncomputable section

namespace Sanguine

open BulkTower
open UniversalEquation

-- ═══════════════════════════════════════════════════════════════════
-- §1. K IS `base` — a constant-state tower (was `BulkContainsK`)
-- ═══════════════════════════════════════════════════════════════════

/-- **K realized as a BulkTower.** The K combinator's partial
    application `K c` is realized as a base equation: constant-`c`
    boundary, constant-`1` drag. The trivial drag is what makes the
    ratio collapse to the boundary, exposing `c` directly as the
    tower's state. -/
def K_tower (c : ℝ) : BulkTower :=
  .base { B := fun _ => c, U := fun _ => 1 }

/-- The K-tower's state at any point is `c`. -/
theorem K_tower_state_apply (c p : ℝ) :
    (K_tower c).state p = c := by
  show c / 1 = c
  ring

/-- The K-tower's state, as a function, is the constant function `c`. -/
theorem K_tower_state (c : ℝ) :
    (K_tower c).state = fun _ => c := by
  funext p; exact K_tower_state_apply c p

/-- **K is in BulkTower (pointwise).** The K-tower's state at any
    point equals `K_comb c p` — the K combinator from
    `UniversalTranslation` applied to the constant `c` and the
    point `p`.

    This is the inevitability witness for K at the ℝ-instance: the
    recursive composition framework already contains K as a base
    case. No combinator basis is imported; K emerges from a particular
    choice of `B` and `U`. -/
theorem K_tower_realizes_K_comb (c p : ℝ) :
    (K_tower c).state p = K_comb c p := by
  show c / 1 = K_comb c p
  unfold K_comb; ring

/-- **K is in BulkTower (function-extensional).** The K-tower's state
    function equals `K_comb c` as functions `ℝ → ℝ`. -/
theorem K_tower_state_eq_K_comb (c : ℝ) :
    (K_tower c).state = K_comb c := by
  funext p; exact K_tower_realizes_K_comb c p

/-- The K-tower has depth `0`: it is a base equation with no
    recursive composition. K does not require the tower's recursive
    machinery — it sits at the bottom. -/
theorem K_tower_depth (c : ℝ) : (K_tower c).depth = 0 := rfl

/-- **A depth-0 tower has no transitions of either kind.** `total_steps_eq_depth` says the
    two counters partition the depth, so an empty spine leaves nothing to count. One
    equation (the depth) decides both; neither counter is its own fact. -/
theorem depth_zero_no_steps {t : BulkTower} (h : t.depth = 0) :
    t.stepU_count = 0 ∧ t.stepB_count = 0 := by
  have := BulkTower.total_steps_eq_depth t
  omega

/-- **Organisation of the K-tower is zero.** With no transitions,
    the syntactic Organisation measure is `0` by convention. -/
theorem K_tower_organization (c : ℝ) : (K_tower c).organization = 0 := by
  unfold BulkTower.organization
  simp [K_tower_depth]

-- ═══════════════════════════════════════════════════════════════════
-- §2. S IS `stepB`/`stepU` — the routing bifurcation (was `BulkContainsS`)
-- ═══════════════════════════════════════════════════════════════════

/-- **The S combinator at the bulk's applicative.**

    `S_bulk f g x = (f x) / (g x)`. The combinatory shape of S —
    take a parameter `x`, distribute it to two functional paths
    `f` and `g`, combine the results — specialized to the binary
    operation the bulk equation `x = B/U` already commits to:
    division.

    Same function as `Polymorphic.pointwiseDiv` (`PolyIotaTermAlgebra`),
    declared twice under two names; merging them is a rename across
    `Triad`, `CombinatorScaleInvariance` and `BulkInevitability`, not a
    proof change. -/
def S_bulk (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x / g x

/-- **`stepB` IS S (prev-as-f orientation).**

    `(stepB prev U_outer).state` is `S_bulk` with `prev.state` in
    the f-slot and `U_outer` in the g-slot. The constructor is not
    building something that *contains* S; it *is* S in the bulk
    algebra. Proof term:
    `BulkAlgebra.stepB_realizes_S_bulk` at `fromBulkTower` — the abstract structure's `state_stepB` FIELD restated, so neither module derives it. CITATION-EXEMPT: names the field this proof rides, weight stated on the line -/
theorem stepB_is_S_bulk (prev : BulkTower) (U_outer : ℝ → ℝ) (x : ℝ) :
    (stepB prev U_outer).state x = S_bulk prev.state U_outer x :=
  BulkAlgebra.stepB_realizes_S_bulk BulkAlgebra.fromBulkTower prev U_outer x

/-- **`stepU` IS S (prev-as-g orientation).**

    Same S, opposite role-assignment: `B_outer` in the f-slot,
    `prev.state` in the g-slot. The two constructors enumerate the
    two role-assignments available to S in a structure with one
    "previous level" and one "outer function." Proof term:
    `BulkAlgebra.stepU_realizes_S_bulk` at `fromBulkTower` — the abstract structure's `state_stepU` FIELD restated, so neither module derives it. CITATION-EXEMPT: names the field this proof rides, weight stated on the line -/
theorem stepU_is_S_bulk (B_outer : ℝ → ℝ) (prev : BulkTower) (x : ℝ) :
    (stepU B_outer prev).state x = S_bulk B_outer prev.state x :=
  BulkAlgebra.stepU_realizes_S_bulk BulkAlgebra.fromBulkTower B_outer prev x

/-- **Swap fixed point of `S_bulk`.** The two role-assignments
    `S_bulk f g` and `S_bulk g f` agree at `x` precisely when
    `f x · f x = g x · g x` (with both denominators non-zero).

    This is the structural fixed point of the routing bifurcation:
    the choice between `stepB` (prev-as-f) and `stepU` (prev-as-g)
    collapses exactly when the inner path and outer path stand in
    the swap-symmetric relation. -/
theorem S_bulk_swap_eq_iff (f g : ℝ → ℝ) (x : ℝ)
    (hf : f x ≠ 0) (hg : g x ≠ 0) :
    S_bulk f g x = S_bulk g f x ↔ f x * f x = g x * g x := by
  unfold S_bulk
  exact div_eq_div_iff hg hf

/-- **K through S's f-slot.** When `stepB` is applied to a K-tower,
    the state is `S_bulk (fun _ => c) U_outer x = c / U_outer x`.
    K is the crystallized rest-state occupying S's f-slot when the
    inner level has reached a defect (constant state). -/
theorem K_tower_through_stepB (c : ℝ) (U_outer : ℝ → ℝ) (x : ℝ) :
    (stepB (K_tower c) U_outer).state x = S_bulk (fun _ => c) U_outer x := by
  rw [stepB_is_S_bulk, K_tower_state]

/-- **K through S's g-slot.** Symmetric: when `stepU` is applied to
    a K-tower, the state is `S_bulk B_outer (fun _ => c) x =
    B_outer x / c`. -/
theorem K_tower_through_stepU (B_outer : ℝ → ℝ) (c : ℝ) (x : ℝ) :
    (stepU B_outer (K_tower c)).state x = S_bulk B_outer (fun _ => c) x := by
  rw [stepU_is_S_bulk, K_tower_state]

/-- **Every recursive step in BulkTower is an `S_bulk`
    distribution.** Packages the two constructor theorems: the
    only two recursive constructors of `BulkTower` are `stepB` and
    `stepU`, and each realizes `S_bulk` in one of its two
    role-assignments. There is no third recursive constructor;
    the routing bifurcation IS the S-distribution.

    No extension of BulkTower's typing is needed and no higher-order
    machinery is imported — but that is because `S_bulk` IS
    `BulkTower.state`'s own defining equation at each recursive
    constructor, taken through `fromBulkTower` whose three axiom fields
    are `rfl`. So this restates the definition under a combinatory
    name; it is not a witness that S is forced. The K side
    (`K_tower_realizes_K_comb`) is where an argument is made. -/
theorem recursive_step_is_S_bulk :
    (∀ (prev : BulkTower) (U_outer : ℝ → ℝ) (x : ℝ),
        (stepB prev U_outer).state x = S_bulk prev.state U_outer x) ∧
    (∀ (B_outer : ℝ → ℝ) (prev : BulkTower) (x : ℝ),
        (stepU B_outer prev).state x = S_bulk B_outer prev.state x) :=
  ⟨stepB_is_S_bulk, stepU_is_S_bulk⟩

-- ═══════════════════════════════════════════════════════════════════
-- §3. ι IS THE ROUTING SELECTOR (was `BulkContainsIota`)
-- ═══════════════════════════════════════════════════════════════════

/-- **ι at the bulk's structural level.** The routing bifurcation
    selector: takes a routing bit `b`, an inner tower `prev`, and
    an outer function, and produces a new tower in one of S's two
    orientations.

    * `b = true`  → `stepB prev outer` (S with prev-as-f, S_bulk
      prev.state outer)
    * `b = false` → `stepU outer prev` (S with prev-as-g, S_bulk
      outer prev.state)

    This is `BulkAlgebra.iota` at `fromBulkTower`, not a second
    selector: ι is forced in *any* bulk algebra once both step
    constructors are present, and the tower is one. -/
def iota_bulk (b : Bool) (prev : BulkTower) (outer : ℝ → ℝ) : BulkTower :=
  BulkAlgebra.fromBulkTower.iota b prev outer

/-- ι at `b = true` is `stepB`. Closes by `rfl`. -/
theorem iota_bulk_true_is_stepB (prev : BulkTower) (outer : ℝ → ℝ) :
    iota_bulk true prev outer = stepB prev outer := rfl

/-- ι at `b = false` is `stepU`. Closes by `rfl`. -/
theorem iota_bulk_false_is_stepU (prev : BulkTower) (outer : ℝ → ℝ) :
    iota_bulk false prev outer = stepU outer prev := rfl

/-- ι at `b = true` has state equal to `S_bulk prev.state outer`. Proof term:
    `BulkAlgebra.iota_true_state` at `fromBulkTower` — the `state_stepB` FIELD through `iota`'s definitional `match`, restated, not derived. CITATION-EXEMPT: names the field this proof rides, weight stated on the line -/
theorem iota_bulk_true_state (prev : BulkTower) (outer : ℝ → ℝ) (x : ℝ) :
    (iota_bulk true prev outer).state x = S_bulk prev.state outer x :=
  BulkAlgebra.iota_true_state BulkAlgebra.fromBulkTower prev outer x

/-- ι at `b = false` has state equal to `S_bulk outer prev.state`. Proof term:
    `BulkAlgebra.iota_false_state` at `fromBulkTower` — the `state_stepU` FIELD through `iota`'s definitional `match`, restated, not derived. CITATION-EXEMPT: names the field this proof rides, weight stated on the line -/
theorem iota_bulk_false_state (prev : BulkTower) (outer : ℝ → ℝ) (x : ℝ) :
    (iota_bulk false prev outer).state x = S_bulk outer prev.state x :=
  BulkAlgebra.iota_false_state BulkAlgebra.fromBulkTower outer prev x

/-- **ι routes K through S's f-slot at `b = true`.** When the
    inner tower is a K-tower, ι with bit `true` produces a state
    `c / outer x` — K in the boundary path, outer in the drag. -/
theorem iota_bulk_with_K_tower_true (c : ℝ) (outer : ℝ → ℝ) (x : ℝ) :
    (iota_bulk true (K_tower c) outer).state x =
      S_bulk (fun _ => c) outer x := by
  rw [iota_bulk_true_state, K_tower_state]

/-- **ι routes K through S's g-slot at `b = false`.** Symmetric:
    the K-tower flows into the drag path, outer into the boundary. -/
theorem iota_bulk_with_K_tower_false (c : ℝ) (outer : ℝ → ℝ) (x : ℝ) :
    (iota_bulk false (K_tower c) outer).state x =
      S_bulk outer (fun _ => c) x := by
  rw [iota_bulk_false_state, K_tower_state]

/-- **ι and `base` together generate every BulkTower.** Every tower
    is either a base equation or an `iota_bulk b prev outer` for
    some routing bit, inner tower, and outer function. The
    inductive structure of `BulkTower` is precisely
    `base ⊕ iota_bulk`. -/
theorem iota_bulk_generates_BulkTower (t : BulkTower) :
    (∃ e, t = base e) ∨
    (∃ (b : Bool) (prev : BulkTower) (outer : ℝ → ℝ),
        t = iota_bulk b prev outer) := by
  cases t with
  | base e => left; exact ⟨e, rfl⟩
  | stepB prev U_outer =>
      right; exact ⟨true, prev, U_outer, rfl⟩
  | stepU B_outer prev =>
      right; exact ⟨false, prev, B_outer, rfl⟩

end Sanguine
