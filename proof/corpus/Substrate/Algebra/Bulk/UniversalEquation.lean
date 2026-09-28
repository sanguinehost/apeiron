import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Substrate.Algebra.Bulk.AbstractBulkDifferential

/-!
# The Universal Equation x = B / U
@name The One Rule
@child There is a single rule the whole framework grows from: a thing is its *boundary* divided by its *drag* — what it reaches for, over what holds it back. Write that down once, and physics, computation, and the rest grow out of it.
@scientist x = B/U: state is the pointwise ratio of a boundary function B over a drag function U. Apply the derivative and the one featureless ratio shatters into two opposite-sign terms (the quotient rule) — the seed of every decomposition downstream.
@engineer `UniversalEquation` packages B, U : ℝ → ℝ; the state is B/U pointwise. d(B/U) = (U·dB − B·dU)/U² — the two-term bulk differential, opposite signs the shadow of differentiating a quotient. The abstract version (any derivation on any field) is `AbstractBulkDifferential`; this is its concrete real-analysis realisation.

The single line `x = B / U` is the generative equation from which the Sanguine
substrate's structure follows. This file formalises the equation as a pair of
real functions `B, U : ℝ → ℝ`, defines its state `x = B / U` pointwise, and
proves the two foundational consequences that arise from applying the
exterior derivative to the unbound ratio.

**Note on algebraic generality**: This file proves the differential decomposition
using real analysis (`deriv` on `ℝ → ℝ`). The abstract algebraic counterpart
is proven in `AbstractBulkDifferential.lean`, which establishes that this
decomposition holds for ANY derivation on ANY field. The analytic theorem here
is the concrete realisation of that algebraic universal.

## Structure

* **Part I — The Equation.** `UniversalEquation` packages the boundary function
  `B` and the drag function `U`. The state `x = B / U` is the pointwise ratio.

* **Part II — The Bulk Differential Decomposition.** Applying `d` (the exterior
  derivative, realised concretely as `deriv` on `ℝ`) to the ratio `B / U`
  produces, by the quotient rule, the two-term form
  $$d\!\left(\frac{B}{U}\right) = \frac{U \cdot dB - B \cdot dU}{U^2}.$$
  The single featureless ratio shatters into a two-term decomposition. The
  two terms come with opposite sign — the algebraic shadow of having applied
  a differential operator to a quotient.

* **Part III — Stationary Phase Characterisation.** The critical points of
  `B / U` are precisely those where the two terms balance:
  `U · dB = B · dU`. This is the variational principle of the universal
  equation. The boundary's gradient and the drag's gradient stand in the
  same ratio as `B` and `U` themselves.

## What is proved here, and what is read onto the world

* **Proved (Mathlib):** `bulk_differential` — `deriv e.state p = (U·dB − B·dU)/U²`, for `B`, `U`
  differentiable at `p` and `U p ≠ 0`. The mathematical content is Mathlib's `HasDerivAt.div`; this
  file's move is stating it on `state`.
* **Proved (native):** `stationary_phase` — `deriv e.state p = 0 ↔ U p · dB = B p · dU`; and
  `stationary_phase_log_form`, the same condition as `dB/B = dU/U` when additionally `B p ≠ 0`.
  Algebra on top of `bulk_differential`, no further analysis.
* **Proved (native):** `composeB_state` and `composeU_state` — one composition level's state is
  `B₀ / (U₀ · U_outer)` and `B_outer · U₀ / B₀`. Under `composeB` the drags accumulate
  multiplicatively and the boundary survives.
* **Proved (native):** `composeB_differential` and `composeU_differential` — the two-term
  decomposition holds of a composed equation. It covers "every level" because `bulk_differential`
  is already quantified over every `UniversalEquation` and `inner` here is arbitrary; these two add
  the differentiability side conditions, not an induction over tower depth.
* **Anchor:** that `dB` is "the boundary's 1-form, compiled once as a global
  operator" and `dU` "the drag's 1-form, sampled per candidate location", crystallising as the
  per-candidate amplitude `Ψ · φ`. No statement in this file mentions a candidate, an amplitude or
  a filter.
* **Anchor:** that `x = B / U` is "the generative equation from which the Sanguine
  substrate's structure follows". What is proved here is the quotient rule for one pair of real
  functions and its stationary condition. The forcing results are elsewhere and are about the
  `BulkAlgebra` signature, not about this file's `UniversalEquation`
  (`BulkInevitability.inevitability`, `FreeBulkAlgebra.bulkTower_is_free`).

-/

noncomputable section

namespace Sanguine

-- ═══════════════════════════════════════════════════════════════════════════
-- PART I: THE UNIVERSAL EQUATION
-- ═══════════════════════════════════════════════════════════════════════════

/-- A bulk-level expression in the universal equation `x = B / U`.

    * `B : ℝ → ℝ` — the boundary function (constraint, predicate, target).
    * `U : ℝ → ℝ` — the drag function (substrate cost, Madelung resistance).

    The state `x = B / U` is the pointwise ratio (see `state`). -/
structure UniversalEquation where
  B : ℝ → ℝ
  U : ℝ → ℝ

namespace UniversalEquation

/-- The state `x = B / U` of a universal equation at the parameter point `p`. -/
def state (e : UniversalEquation) (p : ℝ) : ℝ :=
  e.B p / e.U p

-- ═══════════════════════════════════════════════════════════════════════════
-- PART II: THE BULK DIFFERENTIAL DECOMPOSITION
-- ═══════════════════════════════════════════════════════════════════════════

/-- **The Bulk Differential Decomposition.**

    Applying the exterior derivative `d` to the unbound ratio `B / U` produces
    the two-term form `(U · dB − B · dU) / U²`. The single featureless ratio
    shatters into a difference of two oppositely-signed terms — the
    boundary's local variation `dB` weighted by the drag `U`, against the
    drag's local variation `dU` weighted by the boundary `B`.

    This is the symmetry-breaking move of the framework: before `d`, `B / U`
    is one scalar with no internal structure. After `d`, it carries an
    irreducible two-term decomposition. The two terms are exactly what
    crystallises at the boundary layer as the per-candidate amplitude
    `Ψ · φ`: `dB` is the boundary's 1-form (compiled once as a global
    operator), `dU` is the drag's 1-form (sampled per candidate location). -/
theorem bulk_differential
    {e : UniversalEquation} {p : ℝ}
    (hB : DifferentiableAt ℝ e.B p) (hU : DifferentiableAt ℝ e.U p)
    (hU_nz : e.U p ≠ 0) :
    deriv e.state p =
      (e.U p * deriv e.B p - e.B p * deriv e.U p) / (e.U p)^2 := by
  show deriv (e.B / e.U) p = _
  rw [(hB.hasDerivAt.div hU.hasDerivAt hU_nz).deriv]
  ring

-- ═══════════════════════════════════════════════════════════════════════════
-- PART III: STATIONARY PHASE CHARACTERISATION
-- ═══════════════════════════════════════════════════════════════════════════

/-- **Stationary Phase Characterisation.**

    The critical points of `B / U` — the points where the differential
    `d(B / U)` vanishes — are precisely those where `U · dB = B · dU`.

    Equivalently: at a stationary point, the boundary's gradient and the
    drag's gradient stand in the same ratio as `B` and `U` themselves
    (`dB / B = dU / U`, when `B ≠ 0`). This is the variational principle
    that the framework's per-candidate filter pattern implements at the
    boundary: candidates near a stationary trajectory have `Ψ · φ` close to
    its peak; candidates far from one have `Ψ · φ` near zero. -/
theorem stationary_phase
    {e : UniversalEquation} {p : ℝ}
    (hB : DifferentiableAt ℝ e.B p) (hU : DifferentiableAt ℝ e.U p)
    (hU_nz : e.U p ≠ 0) :
    deriv e.state p = 0 ↔
      e.U p * deriv e.B p = e.B p * deriv e.U p := by
  rw [bulk_differential hB hU hU_nz, div_eq_zero_iff]
  have h_sq_nz : (e.U p)^2 ≠ 0 := pow_ne_zero 2 hU_nz
  constructor
  · rintro (h | h)
    · linarith
    · exact absurd h h_sq_nz
  · intro h
    left
    linarith

/-- **Stationary Phase in Logarithmic Form.**

    When both `B` and `U` are non-zero at `p`, the stationary phase condition
    `U · dB = B · dU` is equivalent to the equality of logarithmic
    derivatives: `dB / B = dU / U`. This is the symmetric statement of the
    variational principle — at stationarity, `B` and `U` change at the same
    *relative* rate, regardless of their absolute magnitudes. -/
theorem stationary_phase_log_form
    {e : UniversalEquation} {p : ℝ}
    (hB : DifferentiableAt ℝ e.B p) (hU : DifferentiableAt ℝ e.U p)
    (hU_nz : e.U p ≠ 0) (hB_nz : e.B p ≠ 0) :
    deriv e.state p = 0 ↔
      deriv e.B p / e.B p = deriv e.U p / e.U p := by
  rw [stationary_phase hB hU hU_nz, div_eq_div_iff hB_nz hU_nz]
  constructor
  · intro h; linear_combination h
  · intro h; linear_combination h

-- ═══════════════════════════════════════════════════════════════════════════
-- PART IV: RECURSIVE COMPOSITION
-- ═══════════════════════════════════════════════════════════════════════════

/-- Compose two universal equations by routing the inner equation's state
    into the outer equation's *boundary*. The new boundary is `inner.state`;
    the new drag is `U_outer`. This is one of the two coupling modes
    available at each level transition: the inner level's `x` becomes the
    outer level's `B` (the inner's result is treated as a goal to be matched
    against further drag). -/
def composeB (inner : UniversalEquation) (U_outer : ℝ → ℝ) : UniversalEquation :=
  { B := inner.state, U := U_outer }

/-- Compose two universal equations by routing the inner equation's state
    into the outer equation's *drag*. The new boundary is `B_outer`;
    the new drag is `inner.state`. This is the second coupling mode: the
    inner level's `x` becomes the outer level's `U` (the inner's result
    is treated as substrate-given resistance). -/
def composeU (B_outer : ℝ → ℝ) (inner : UniversalEquation) : UniversalEquation :=
  { B := B_outer, U := inner.state }

/-- **Boundary composition collapses to a product of drags.**

    When the inner's state becomes the outer's boundary, the composition's
    state at `p` equals `inner.B p / (inner.U p · U_outer p)`. The two
    drags accumulate multiplicatively; the original boundary survives
    unchanged. -/
theorem composeB_state {inner : UniversalEquation} {U_outer : ℝ → ℝ} {p : ℝ}
    (hU_inner_nz : inner.U p ≠ 0) (hU_outer_nz : U_outer p ≠ 0) :
    (inner.composeB U_outer).state p = inner.B p / (inner.U p * U_outer p) := by
  show inner.state p / U_outer p = _
  unfold state
  field_simp

/-- **Drag composition reweights by the inverse inner-state.**

    When the inner's state becomes the outer's drag, the composition's
    state at `p` equals `B_outer p · inner.U p / inner.B p`. -/
theorem composeU_state {B_outer : ℝ → ℝ} {inner : UniversalEquation} {p : ℝ}
    (hB_inner_nz : inner.B p ≠ 0) (hU_inner_nz : inner.U p ≠ 0) :
    (composeU B_outer inner).state p = B_outer p * inner.U p / inner.B p := by
  show B_outer p / inner.state p = _
  unfold state
  field_simp

/-- **Differentiability propagates under boundary composition.** -/
theorem composeB_differentiableAt {inner : UniversalEquation} {U_outer : ℝ → ℝ}
    {p : ℝ}
    (hB : DifferentiableAt ℝ inner.B p) (hU_in : DifferentiableAt ℝ inner.U p)
    (hU_in_nz : inner.U p ≠ 0) :
    DifferentiableAt ℝ (inner.composeB U_outer).B p := by
  show DifferentiableAt ℝ inner.state p
  exact hB.div hU_in hU_in_nz

/-- **Scale invariance of the differential decomposition.**

    The differential decomposition `d(x) = (U · dB − B · dU) / U²` holds
    at every level of the composition tower. Concretely: when we
    apply `bulk_differential` to a composed equation, the same form
    emerges as at the base level, just with the composed equation's `B`,
    `U` substituted in.

    This is what "scale invariance" cashes out to formally — not a
    philosophical claim about levels looking alike, but a theorem that the
    same equation form is satisfied at every level of recursive composition. -/
theorem composeB_differential
    {inner : UniversalEquation} {U_outer : ℝ → ℝ} {p : ℝ}
    (hB : DifferentiableAt ℝ inner.B p) (hU_in : DifferentiableAt ℝ inner.U p)
    (hU_out : DifferentiableAt ℝ U_outer p)
    (hU_in_nz : inner.U p ≠ 0) (hU_out_nz : U_outer p ≠ 0) :
    deriv (inner.composeB U_outer).state p =
      ((inner.composeB U_outer).U p * deriv (inner.composeB U_outer).B p
       - (inner.composeB U_outer).B p * deriv (inner.composeB U_outer).U p)
      / ((inner.composeB U_outer).U p)^2 :=
  bulk_differential (composeB_differentiableAt hB hU_in hU_in_nz) hU_out hU_out_nz

/-- **Scale invariance for drag composition.**

    Symmetric statement for the second coupling mode: the differential
    decomposition holds at every level when the previous `x` flows into
    the new `U`. -/
theorem composeU_differential
    {B_outer : ℝ → ℝ} {inner : UniversalEquation} {p : ℝ}
    (hB_in : DifferentiableAt ℝ inner.B p) (hU_in : DifferentiableAt ℝ inner.U p)
    (hB_out : DifferentiableAt ℝ B_outer p)
    (hU_in_nz : inner.U p ≠ 0) (hB_in_nz : inner.B p ≠ 0) :
    deriv (composeU B_outer inner).state p =
      ((composeU B_outer inner).U p * deriv (composeU B_outer inner).B p
       - (composeU B_outer inner).B p * deriv (composeU B_outer inner).U p)
      / ((composeU B_outer inner).U p)^2 := by
  have hU_compU : DifferentiableAt ℝ (composeU B_outer inner).U p := by
    show DifferentiableAt ℝ inner.state p
    exact hB_in.div hU_in hU_in_nz
  have hU_compU_nz : (composeU B_outer inner).U p ≠ 0 := by
    show inner.state p ≠ 0
    unfold state
    exact div_ne_zero hB_in_nz hU_in_nz
  exact bulk_differential hB_out hU_compU hU_compU_nz

end UniversalEquation
end Sanguine
