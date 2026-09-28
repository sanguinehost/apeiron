import Mathlib.RingTheory.Derivation.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Ring

/-!
# Abstract Bulk Differential Decomposition
@name The Quotient Rule Without the Calculus
@child The rule for how a fraction changes — top times nothing, minus bottom times nothing, over bottom squared — works in any number-system at all, not just ordinary numbers. It's pure shape.
@scientist The quotient-rule decomposition of `x = B/U` holds for *any* abstract derivation on *any* field, not just `deriv` on `ℝ→ℝ` — making "x = B/U" a structural (coalgebraic) claim.
@engineer `abstract_bulk_differential` proves `d(B/U) = (U·dB − B·dU)/U²` for any `Derivation R F F`; `abstract_stationary_phase` characterises critical points by `b·Da = a·Db`.

This file lifts the universal equation `x = B / U` and its differential
decomposition from the real-analysis instance (`deriv` on `ℝ → ℝ`, proved in
`UniversalEquation`) to the **algebraic level**: the quotient-rule
decomposition holds for *any* abstract derivation on *any* field.

## Why this matters for the bulk basis

The framework's rebasing places `x = B / U` as the generative equation, with
the bulk as its unbound form. For that rebasing to be load-bearing, the
equation's content must not depend on real analysis specifically — it must
hold in any algebraic setting where the operations make sense.

This file proves precisely that. The quotient rule
$$d\!\left(\frac{B}{U}\right) = \frac{U \cdot dB - B \cdot dU}{U^2}$$
is shown to hold for any `Derivation R F F` over any field `F` that is an
`R`-algebra. The previous concrete result (`UniversalEquation.bulk_differential`)
becomes a specialisation of this abstract form to `(R, F) = (ℝ, ℝ)` with
the derivation being Mathlib's `deriv` operator.

This is what makes "x = B/U as the universal equation" a structural claim
about *coalgebraic shape* rather than a real-analysis fact. The shape is:
a derivation on a field, applied to a ratio, producing a two-term signed
decomposition that is the algebraic content of differentiation.

## Structure

* `Derivation.apply_inv`: the derivation-of-inverse formula, `D(b⁻¹) = -b⁻²·Db`.
* `abstract_bulk_differential`: the quotient rule, the algebraic-level
  version of `UniversalEquation.bulk_differential`.
* `abstract_stationary_phase`: critical points of `a/b` characterised by
  `b · D a = a · D b`, the algebraic-level version of
  `UniversalEquation.stationary_phase`.

## What is proved here, and what "algebraic level" does and does not buy

* **Proved (Mathlib):** all four results, native to this file over `[CommRing R] [Field F]
  [Algebra R F]` and any `Derivation R F F` — `Derivation.apply_inv` (`D b⁻¹ = -(D b / b²)`),
  `abstract_bulk_differential` (`D (a/b) = (b·D a − a·D b)/b²`), `abstract_stationary_phase`
  (`D (a/b) = 0 ↔ b·D a = a·D b`) and `abstract_stationary_phase_log_form`. Mathlib supplies
  `Derivation`, `leibniz`, `map_one_eq_zero`, `field_simp` and `linear_combination`.
* **Retired:** that "the previous concrete result
  (`UniversalEquation.bulk_differential`) becomes a specialisation of this abstract form to
  `(R, F) = (ℝ, ℝ)` with the derivation being Mathlib's `deriv` operator". No such specialisation
  occurs in this file — `UniversalEquation` is not imported and `deriv` does not appear — and the
  route named cannot close as written: `UniversalEquation.bulk_differential` is about `deriv` of a
  pointwise ratio of `ℝ → ℝ` functions, and `ℝ → ℝ` under pointwise operations is not a `Field`, so
  it is not an instance of a theorem hypothesising `[Field F]`. The two results are parallel, not
  nested.
* **Anchor:** that this makes "`x = B/U` as the universal equation" a structural claim
  about coalgebraic shape rather than a real-analysis fact, and that the framework's
  per-candidate-amplitude filter is "the boundary-layer crystallisation of this critical-point
  condition". What is proved is the quotient rule for an abstract derivation; no bulk, no universe,
  no amplitude and no coalgebra occurs in any statement here.

-/

namespace Sanguine

variable {R F : Type*} [CommRing R] [Field F] [Algebra R F]

-- ═══════════════════════════════════════════════════════════════════════════
-- PART I: DERIVATION OF AN INVERSE
-- ═══════════════════════════════════════════════════════════════════════════

/-- **Derivation of an inverse.**

    For any derivation `D` on a field `F` and any non-zero element `b`:
    $$D(b^{-1}) = -\frac{D b}{b^2}.$$

    This is the prerequisite for the quotient rule: from this and Leibniz,
    the quotient-rule decomposition follows by algebra. -/
theorem Derivation.apply_inv (D : Derivation R F F) (b : F) (hb : b ≠ 0) :
    D (b⁻¹) = -(D b / b^2) := by
  have hb_sq : b^2 ≠ 0 := pow_ne_zero 2 hb
  -- Establish the key intermediate: b² · D(b⁻¹) = -D(b)
  have key : b^2 * D (b⁻¹) = -D b := by
    have h_one : b * b⁻¹ = 1 := mul_inv_cancel₀ hb
    have h_D_one : D (b * b⁻¹) = 0 := by rw [h_one]; exact D.map_one_eq_zero
    rw [D.leibniz] at h_D_one
    -- h_D_one : b • D b⁻¹ + b⁻¹ • D b = 0
    -- In F-F module, scalar action is multiplication
    have h_smul₁ : b • D (b⁻¹) = b * D (b⁻¹) := rfl
    have h_smul₂ : b⁻¹ • D b = b⁻¹ * D b := rfl
    rw [h_smul₁, h_smul₂] at h_D_one
    -- h_D_one : b * D b⁻¹ + b⁻¹ * D b = 0
    -- Multiply by b, use b * b⁻¹ = 1, get b² · D b⁻¹ + D b = 0
    have h_scaled : b^2 * D (b⁻¹) + b * b⁻¹ * D b = 0 := by
      linear_combination b * h_D_one
    rw [h_one, one_mul] at h_scaled
    -- h_scaled : b² · D(b⁻¹) + D b = 0
    linear_combination h_scaled
  -- Divide: D(b⁻¹) = -D(b) / b² = -(D(b) / b²)
  have h_eq : -(D b / b^2) = -D b / b^2 := by ring
  rw [h_eq, eq_div_iff hb_sq]
  linear_combination key

-- ═══════════════════════════════════════════════════════════════════════════
-- PART II: THE ABSTRACT BULK DIFFERENTIAL DECOMPOSITION
-- ═══════════════════════════════════════════════════════════════════════════

/-- **The Abstract Bulk Differential Decomposition.**

    The quotient-rule decomposition
    $$D\!\left(\frac{a}{b}\right) = \frac{b \cdot D a - a \cdot D b}{b^2}$$
    holds for any derivation `D` on any field `F`, any algebra `R`, and any
    elements `a, b ∈ F` with `b ≠ 0`.

    This is the algebraic-level statement of the framework's universal
    equation: the irreducible two-term signed decomposition is forced by
    the structure of a derivation acting on a ratio, independent of any
    analytic structure on the underlying type. -/
theorem abstract_bulk_differential
    (D : Derivation R F F) (a b : F) (hb : b ≠ 0) :
    D (a / b) = (b * D a - a * D b) / b^2 := by
  -- a / b = a * b⁻¹
  rw [div_eq_mul_inv]
  -- D(a * b⁻¹) = a • D(b⁻¹) + b⁻¹ • D a, by Leibniz
  rw [D.leibniz]
  -- Substitute D(b⁻¹) = -(D b / b^2)
  rw [Derivation.apply_inv D b hb]
  -- Scalar action is multiplication in F
  show a * -(D b / b^2) + b⁻¹ * D a = (b * D a - a * D b) / b^2
  have hb_sq : b^2 ≠ 0 := pow_ne_zero 2 hb
  field_simp
  ring

-- ═══════════════════════════════════════════════════════════════════════════
-- PART III: ABSTRACT STATIONARY PHASE
-- ═══════════════════════════════════════════════════════════════════════════

/-- **Abstract Stationary Phase Characterisation.**

    The critical points of the ratio `a/b` — where any derivation `D`
    vanishes — are precisely those where `b · D a = a · D b`.

    This is the abstract-level version of the variational principle:
    at stationary phase, the two terms of the differential balance, and
    the proportionality is `a/b` itself (the current state). The
    framework's per-candidate-amplitude filter is the boundary-layer
    crystallisation of this critical-point condition. -/
theorem abstract_stationary_phase
    (D : Derivation R F F) (a b : F) (hb : b ≠ 0) :
    D (a / b) = 0 ↔ b * D a = a * D b := by
  rw [abstract_bulk_differential D a b hb, div_eq_zero_iff]
  have hb_sq_nz : b^2 ≠ 0 := pow_ne_zero 2 hb
  constructor
  · rintro (h | h)
    · linear_combination h
    · exact absurd h hb_sq_nz
  · intro h
    left
    linear_combination h

/-- **Abstract Stationary Phase in Logarithmic Form.**

    When `a` is also non-zero, the stationary-phase condition is equivalent
    to the equality of logarithmic derivatives `D a / a = D b / b`. This
    is the symmetric form: at stationarity, `a` and `b` change at the same
    *relative* rate under the derivation. -/
theorem abstract_stationary_phase_log_form
    (D : Derivation R F F) (a b : F) (ha : a ≠ 0) (hb : b ≠ 0) :
    D (a / b) = 0 ↔ D a / a = D b / b := by
  rw [abstract_stationary_phase D a b hb, div_eq_div_iff ha hb]
  constructor
  · intro h; linear_combination h
  · intro h; linear_combination h

end Sanguine
