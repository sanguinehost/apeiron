/-
Transported density along a classical path (`QREG-PATH-DENSITY`, epic
the quantum-register note).

Lohmiller and Slotine (arXiv 2405.06328, eq. 12): along a classical path `x(t)` the density obeys
the continuity equation transported along the flow, `dρ/dt = −ρ · a(t)`, where
`a(t) = Δφ(x(t), t)/m` is the velocity divergence evaluated on the path. This file proves that for
continuous `a` the equation has the unique solution `ρ(t) = ρ₀ exp(−∫₀ᵗ a)`, and that it is
positive when `ρ₀ > 0`.

## Landed here

* `pathDensity_hasDerivAt`: the closed form satisfies `dρ/dt = −ρ a(t)` at every `t`.
* `pathDensity_zero`: the closed form takes the value `ρ₀` at `t = 0`.
* `pathDensity_unique`: any `C¹` solution with the same initial value equals the closed form.
* `pathDensity_pos`: the closed form is strictly positive when `ρ₀ > 0`.

## Not claimed

* That `a` is `Δφ(x(t), t)/m` for a given Hamiltonian: `a` is an arbitrary continuous function here.
* Existence for a nonlinear right-hand side: the equation is linear in `ρ` with a given `a`.
* Any numerical or floating-point statement.

## Complexity

`pathDensity` is one integral and one exponential, counted in evaluations of `a`; it is not
computed by the proofs here.
-/
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Tactic

namespace Numeric.PathDensity

open Real Filter Topology MeasureTheory intervalIntegral

variable (a : ℝ → ℝ)

/-- The transported density `ρ(t) = ρ₀ exp(−∫₀ᵗ a)`. -/
noncomputable def pathDensity (ρ0 : ℝ) (a : ℝ → ℝ) (t : ℝ) : ℝ :=
  ρ0 * exp (-∫ s in (0:ℝ)..t, a s)

/-- The closed form solves `dρ/dt = −ρ a(t)` at every `t`, for continuous `a`. -/
theorem pathDensity_hasDerivAt (ha : Continuous a) (ρ0 t : ℝ) :
    HasDerivAt (pathDensity ρ0 a) (-(pathDensity ρ0 a t) * a t) t := by
  have hI : HasDerivAt (fun u => ∫ s in (0:ℝ)..u, a s) (a t) t :=
    (ha.integral_hasStrictDerivAt 0 t).hasDerivAt
  have h3 : HasDerivAt (fun u => ρ0 * exp (-∫ s in (0:ℝ)..u, a s))
      (ρ0 * (exp (-∫ s in (0:ℝ)..t, a s) * (-a t))) t :=
    (hI.neg.exp).const_mul ρ0
  have e : -(pathDensity ρ0 a t) * a t
      = ρ0 * (exp (-∫ s in (0:ℝ)..t, a s) * (-a t)) := by
    unfold pathDensity
    ring
  rw [e]
  exact h3

/-- The closed form takes the value `ρ₀` at `t = 0`. -/
theorem pathDensity_zero (ρ0 : ℝ) : pathDensity ρ0 a 0 = ρ0 := by
  simp [pathDensity]

/-- Any `C¹` solution of `dρ/dt = −ρ a(t)` with `ρ(0) = ρ₀` equals the closed form. -/
theorem pathDensity_unique (ha : Continuous a) (ρ : ℝ → ℝ) (ρ0 : ℝ)
    (hρ : ∀ t, HasDerivAt ρ (-(ρ t) * a t) t) (h0 : ρ 0 = ρ0) :
    ∀ t, ρ t = pathDensity ρ0 a t := by
  have hu : ∀ t, HasDerivAt (fun u => ρ u * exp (∫ s in (0:ℝ)..u, a s)) 0 t := by
    intro t
    have hI : HasDerivAt (fun u => ∫ s in (0:ℝ)..u, a s) (a t) t :=
      (ha.integral_hasStrictDerivAt 0 t).hasDerivAt
    have h := (hρ t).mul hI.exp
    exact h.congr_deriv (by ring)
  have hc : ∀ t, ρ t * exp (∫ s in (0:ℝ)..t, a s) = ρ0 := by
    intro t
    have hconst : ρ t * exp (∫ s in (0:ℝ)..t, a s)
        = ρ 0 * exp (∫ s in (0:ℝ)..0, a s) :=
      is_const_of_deriv_eq_zero (fun s => (hu s).differentiableAt)
        (fun s => (hu s).deriv) t 0
    rw [hconst, h0, intervalIntegral.integral_same, Real.exp_zero, mul_one]
  intro t
  show ρ t = ρ0 * exp (-∫ s in (0:ℝ)..t, a s)
  rw [Real.exp_neg, ← hc t, mul_assoc, mul_inv_cancel₀ (Real.exp_ne_zero _), mul_one]

/-- The closed form is strictly positive when `ρ₀ > 0`. -/
theorem pathDensity_pos (ρ0 : ℝ) (hρ0 : 0 < ρ0) (t : ℝ) : 0 < pathDensity ρ0 a t := by
  unfold pathDensity
  exact mul_pos hρ0 (Real.exp_pos _)

end Numeric.PathDensity
