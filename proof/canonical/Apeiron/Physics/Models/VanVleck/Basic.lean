import Apeiron.Physics.Models.Schrodinger
import Apeiron.Physics.Quadratic

/-!
# Pointwise Van Vleck semantics

The three quadratic faces share one quantum claim: at a stated point, the polar wave's analytic
Schrödinger residual is zero. This module names that proposition without constructing an evolution
operator or asserting a relation between kernels. Each face supplies its own potential, density,
action, regular domain, derivative witnesses, and residual proof.

## Root

`Physics.Models.Schrodinger` supplies `analyticSchrodingerResidual`, whose zero is the checked
pointwise PDE statement. `Physics.Quadratic` supplies the classical generator classification used
by the face modules. This module proves neither a propagator equality nor a path-integral result.
-/

namespace Apeiron
namespace Physics

/-- The polar Schrödinger equation holds at one specified point.

    (hbar mass : ℝ)                               [the Planck and mass scales]
    (potential density action : ℝ → ℝ → ℝ)        [the potential, density, and action fields]
    (t x : ℝ)                                     [the time and spatial point]
    Prop                                           [a proposition that can hold or fail]
    analyticSchrodingerResidual hbar mass potential density action t x = 0
                                                    [the actual pointwise residual is zero]
    analyticSchrodingerResidual                    [the actual pointwise Schrödinger residual]

    So: this is one residual equation at one point, not an equality of propagators or kernels. -/
noncomputable def PointwiseSchrodingerExact
    (hbar mass : ℝ) (potential density action : ℝ → ℝ → ℝ) (t x : ℝ) : Prop :=
  analyticSchrodingerResidual hbar mass potential density action t x = 0

/-- Pointwise exactness is exactly the analytic residual equality.

    (hbar mass : ℝ)                               [the Planck and mass scales]
    (potential density action : ℝ → ℝ → ℝ)        [the potential, density, and action fields]
    (t x : ℝ)                                     [the time and spatial point]
    PointwiseSchrodingerExact hbar mass potential density action t x
                                                    [the named pointwise exactness proposition]
    ↔ analyticSchrodingerResidual hbar mass potential density action t x = 0
                                                    [the analytic residual equality]
    PointwiseSchrodingerExact                     [the named pointwise exactness proposition]
    analyticSchrodingerResidual                    [the actual pointwise Schrödinger residual]
    ↔                                               [exactly when, true in both directions]

    So: changing the predicate's residual orientation makes this theorem fail. -/
theorem pointwiseSchrodingerExact_iff
    (hbar mass : ℝ) (potential density action : ℝ → ℝ → ℝ) (t x : ℝ) :
    PointwiseSchrodingerExact hbar mass potential density action t x ↔
      analyticSchrodingerResidual hbar mass potential density action t x = 0 := Iff.rfl

end Physics
end Apeiron
