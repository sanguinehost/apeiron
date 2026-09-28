import Substrate.Algebra.Bulk.BulkContains

/-!
# Affine Loop Collapse via the Bulk Algebra

@name The Loop That Never Runs
@child A loop that repeats a step over and over usually has to run all the way through. But if each step just multiplies and adds, you can jump straight to where it ends up, without taking a single step.
@scientist An affine recurrence `x_{k+1} = A·x_k + C` (A ≠ 1) has a fixed point `x* = C/(1−A)`; with constant boundary and drag, dB = dU = 0, so the stationary-phase condition holds at every point and the state is a global constant.
@engineer `affineLoopEquation` sets B = C, U = 1−A; `affineLoopEquation_state` gives state = C/(1−A) by `rfl`, and this constant IS the `K_tower` realiser — no `Y_Comb`, no iteration.

An affine loop `x_{k+1} = A * x_k + C` with `A ≠ 1` has fixed
point `x* = C / (1 - A)`. This file proves that this fixed point
IS a K-realiser in the BulkAlgebra — the constant-state tower
`K_tower (C / (1 - A))` from `BulkContains.lean` (this said `BulkContainsK.lean`;
the trio merged into `BulkContains` in `f7251be1`).

## The connection to the proof spine

The existing chain is:

    UniversalEquation.state  →  BulkTower.state  →  K_tower
                                                      ↑
                                              (this file)

An affine loop's boundary is `B = C` (constant), its drag is
`U = 1 - A` (constant). The `UniversalEquation` with these
constant functions has `state p = C / (1 - A)` for all `p`.
That IS `K_tower_state_apply` from `BulkContainsK`.

The stationary phase condition `U · dB = B · dU` holds trivially
because both `dB` and `dU` are zero (constant functions have zero
derivative). This is the formal content of "the loop collapses
without iteration" — the variational principle is satisfied at
every point, meaning the state is a global constant (a K-shape).

## What this tells the translator

The translator should emit `K_tower(C / (1 - A))` — i.e.,
`Extract_Resonance(C)(1-A)` — for any affine loop. No `Y_Comb`,
no iteration. The K=4 evaluator reduces this to the constant
value directly.

## What is proved here vs what is read onto the loop

* **Proved (native):** with constant boundary and constant drag the state is the constant
  `C / (1 - A)` at every point, its derivative vanishes everywhere, and the stationary-phase
  condition `U · dB = B · dU` holds at every point — `affineLoopEquation_state`,
  `affineLoopTower_deriv_zero`, `affineLoop_stationary_phase`. Every one is `rfl` or
  `deriv_const`: the flatness is by construction of `affineLoopEquation`, not a discovery about
  a recurrence.
* **Proved (riding):** the constant state IS the K-realiser — `affineLoopTower_eq_K_tower_state`
  is `BulkContains.K_tower_state_apply` rewritten in the loop's clothing.
* **Anchor:** that the recurrence `x_{k+1} = A · x_k + C` *is* this equation
  (`B = C`, `U = 1 - A`) is the reading. No recurrence, iterate, or fixed point of one appears in
  any statement in this file; the loop enters only through the names.
* **Anchor:** "no `Y_Comb`, no iteration", and that the K=4 evaluator reduces the
  emitted tower to the constant directly. No statement here mentions an evaluator, a step count,
  or a translator.
* **Open:** `A ≠ 1`. The prose and `@scientist` carry that hypothesis; no
  theorem in this file does. At `A = 1` all four statements still hold, with `C / 0 = 0` —
  which is not the loop's fixed point; a theorem carrying `A ≠ 1` as a hypothesis and stating
  the fixed-point character would close it.

-/

noncomputable section

namespace Sanguine

open BulkTower UniversalEquation

-- ═══════════════════════════════════════════════════════════════════
-- THE AFFINE LOOP AS A UNIVERSAL EQUATION
-- ═══════════════════════════════════════════════════════════════════

/-- The universal equation corresponding to an affine loop
    `x_{k+1} = A * x_k + C`. Boundary = C (constant),
    drag = 1 - A (constant). -/
def affineLoopEquation (A C : ℝ) : UniversalEquation where
  B := fun _ => C
  U := fun _ => 1 - A

/-- The state of the affine-loop equation is the constant
    `C / (1 - A)` at every point. This IS the loop's fixed
    point. -/
theorem affineLoopEquation_state (A C p : ℝ) :
    (affineLoopEquation A C).state p = C / (1 - A) := rfl

-- ═══════════════════════════════════════════════════════════════════
-- THE LOOP'S FIXED POINT IS A K-TOWER
-- ═══════════════════════════════════════════════════════════════════

/-- The BulkTower corresponding to the affine loop. -/
def affineLoopTower (A C : ℝ) : BulkTower :=
  .base (affineLoopEquation A C)

/-- The affine-loop tower's state equals the K-tower's state
    at the fixed-point value `C / (1 - A)`. Both are the
    constant function `C / (1 - A)`.

    This is the formal identification: the affine loop's fixed
    point IS the K-realiser from `BulkContainsK`. -/
theorem affineLoopTower_eq_K_tower_state (A C p : ℝ) :
    (affineLoopTower A C).state p = (K_tower (C / (1 - A))).state p := by
  -- LHS: C / (1 - A)  (by definition of affineLoopTower)
  -- RHS: C / (1 - A)  (by K_tower_state_apply)
  show C / (1 - A) = (K_tower (C / (1 - A))).state p
  rw [K_tower_state_apply]

-- ═══════════════════════════════════════════════════════════════════
-- THE STATIONARY PHASE IS TRIVIALLY SATISFIED
-- ═══════════════════════════════════════════════════════════════════

/-- The affine-loop tower's derivative is zero everywhere —
    constant functions have zero derivative. The stationary
    phase condition `d(B/U) = 0` holds globally, not just at
    isolated critical points.

    This is WHY the loop "collapses without iteration": the
    entire state function is flat, so the variational principle
    is trivially satisfied at every point. There is no need to
    search for a critical point — every point is critical. -/
theorem affineLoopTower_deriv_zero (A C : ℝ) (p : ℝ) :
    deriv (affineLoopTower A C).state p = 0 := by
  -- The state is the constant function C / (1 - A).
  -- The derivative of a constant function is zero.
  have h_eq : (affineLoopTower A C).state = fun _ => C / (1 - A) := by
    funext t; rfl
  rw [h_eq]
  exact deriv_const p (C / (1 - A))

/-- **The stationary phase condition for the affine loop.**

    Connecting to the proof spine: `tower_stationary_phase` from
    `BulkTower.lean` characterises the critical points of any
    tower's state as `U · dB = B · dU`. For the affine loop,
    both `dB` and `dU` are zero (constant functions), so the
    condition reduces to `0 = 0` — trivially true everywhere.

    This is the bulk-algebraic proof that affine loops are K-shapes:
    their state has no non-trivial variational structure. The
    translator should emit a K-tower, not a Y_Comb iterator. -/
theorem affineLoop_stationary_phase (A C : ℝ) (p : ℝ) :
    (affineLoopEquation A C).U p * deriv (affineLoopEquation A C).B p =
    (affineLoopEquation A C).B p * deriv (affineLoopEquation A C).U p := by
  -- B is constant C, so deriv B = 0
  -- U is constant (1-A), so deriv U = 0
  -- Both sides reduce to _ * 0 = _ * 0, i.e. 0 = 0
  have hB_const : (affineLoopEquation A C).B = fun _ => C := rfl
  have hU_const : (affineLoopEquation A C).U = fun _ => (1 - A) := rfl
  rw [hB_const, hU_const]
  simp [deriv_const]

end Sanguine
