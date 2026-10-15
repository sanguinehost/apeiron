/-
Madelung equivalence with the quantum potential read from the branches (`QREG-BRANCH-MADELUNG-1D`,
epic `docs/epics/quantum-register.md`).

The canonical 1-D theorem `Apeiron.Physics.actual_schrodinger_iff_madelung`
(`proof/canonical/Apeiron/Physics/Models/Schrodinger.lean`) states the Schrödinger residual as the
Hamilton-Jacobi residual with the Bohm potential `Q = -(ℏ²/2M) · R''/R` plus the continuity residual,
for the amplitude `R = √ρ`. This file reads `Q` from finitely many evaluations of `R`: the three
branches `x - h`, `x`, `x + h`. The limit `h → 0`, `h ≠ 0`, is `Numeric.BranchLaplacian.secondDiff_tendsto`.

## Landed here

* `branchQ`: `Q` evaluated from the centre and the two branches `x ± h`. Complexity: 3 evaluations
  of `A`, O(1) in time and space.
* `branchQ_tendsto`: for `A ∈ C²`, `branchQ hbar mass A x → Q(A x, A''(x))` as `h → 0`, `h ≠ 0`.
* `schrodinger_iff_branch_madelung`: under the hypotheses of `actual_schrodinger_iff_madelung` and
  `R = √ρ ∈ C²` in `z`, (a) the branch limit of `Q` is the canonical `Q` at `amplitudeLaplacian x`,
  and (b) the canonical Schrödinger iff, verbatim.

## Not claimed

* A rate of convergence in `h`, or a finite-`h` identity: this is a limit.
* Floating-point behaviour of the difference quotient.
* That the branches are the stencil of any particular evaluator.

## Complexity

`branchQ` evaluates `A` 3 times per `h`; O(1) in time and space.
-/
import Numeric.Quantum.BranchLaplacian
import Apeiron.Physics.Models.Schrodinger

namespace Numeric.BranchMadelung

open Filter Topology
open Apeiron.Physics

/-- The quantum potential read from the branches `x ± h` and the centre `x`:
    `Q = -(ℏ²/2M) · (A(x+h) + A(x-h) - 2 A(x)) / h² / A(x)`.
    Complexity: 3 evaluations of `A`, O(1) in time and space.

    hbar mass : ℝ                                  [the Planck and mass scales]
    A : ℝ → ℝ                                      [the amplitude, a real function of one real]
    x h : ℝ                                        [the centre and the branch offset]
    →                                              [takes the left and gives the right]

    So: `branchQ hbar mass A x h` is the quantum potential built from three samples of `A`. -/
noncomputable def branchQ (hbar mass : ℝ) (A : ℝ → ℝ) (x h : ℝ) : ℝ :=
  Apeiron.Physics.quantumPotential hbar mass (A x)
    ((A (x + h) + A (x - h) - 2 * A x) / h ^ 2)

/-- For `A ∈ C²`, the branch quantum potential tends to the quantum potential of `A`
    with its second derivative, as `h → 0` with `h ≠ 0`.

    ContDiff ℝ 2 A                                 [A is twice continuously differentiable]
    Tendsto f (𝓝[≠] 0) (𝓝 L)                      [f tends to L along h ≠ 0 near 0]
    deriv (deriv A) x                              [the second derivative of A at x]
    →                                              [takes the left and gives the right]

    So: the branch quantum potential converges to the canonical quantum potential. -/
theorem branchQ_tendsto (hbar mass : ℝ) {A : ℝ → ℝ} (hA : ContDiff ℝ 2 A) (x : ℝ) :
    Tendsto (branchQ hbar mass A x) (𝓝[≠] 0)
      (𝓝 (Apeiron.Physics.quantumPotential hbar mass (A x) (deriv (deriv A) x))) := by
  have hL := ((Numeric.BranchLaplacian.secondDiff_tendsto hA x).div_const (A x)).const_mul
    (-(hbar ^ 2 / (2 * mass)))
  unfold branchQ Apeiron.Physics.quantumPotential
  exact hL

/-- The Madelung equivalence with `Q` read from the branches: the branch limit of `Q` for
    `R = √ρ` is the canonical `Q` at `amplitudeLaplacian x`, and the canonical iff holds verbatim.
    The hypotheses are those of `Apeiron.Physics.actual_schrodinger_iff_madelung`, plus
    `R = √ρ(t, ·) ∈ C²`.

    0 < hbar ∧ 0 < mass ∧ 0 < rho t x             [the positive physical domain]
    HasDerivAt f v y                               [f is changing at rate v at y]
    Real.sqrt                                      [the number that squares to give this]
    ∀ y, P y                                       [for every point y]
    ContDiff ℝ 2 R                                 [R is twice continuously differentiable]
    Tendsto f (𝓝[≠] 0) (𝓝 L)                      [f tends to L along h ≠ 0 near 0]
    analyticSchrodingerResidual ... = 0 ↔ P ∧ Q    [the canonical Madelung equivalence]
    ∧                                              [both are true]
    ↔                                              [exactly when, true in both directions]
    →                                              [takes the left and gives the right]
    ℝ                                              [the real numbers]
    <                                              [is smaller than]
    ≠                                              [is not equal to]

    So: the branch-read Q is the canonical Q, and the canonical Madelung iff holds. -/
theorem schrodinger_iff_branch_madelung
    (hbar mass : ℝ) (potential rho action : ℝ → ℝ → ℝ) (t x amplitudeTime actionTime : ℝ)
    (amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ → ℝ)
    (hHbar : 0 < hbar) (hMass : 0 < mass) (hRho : 0 < rho t x)
    (hAmplitudeTime : HasDerivAt (fun s => Real.sqrt (rho s x)) amplitudeTime t)
    (hActionTime : HasDerivAt (fun s => action s x) actionTime t)
    (hAmplitudeSpace : ∀ y, HasDerivAt (fun z => Real.sqrt (rho t z)) (amplitudeSpace y) y)
    (hAmplitudeLaplacian : ∀ y, HasDerivAt amplitudeSpace (amplitudeLaplacian y) y)
    (hActionSpace : ∀ y, HasDerivAt (fun z => action t z) (actionSpace y) y)
    (hActionLaplacian : ∀ y, HasDerivAt actionSpace (actionLaplacian y) y)
    (hC2 : ContDiff ℝ 2 (fun z => Real.sqrt (rho t z))) :
    Tendsto (branchQ hbar mass (fun z => Real.sqrt (rho t z)) x) (𝓝[≠] 0)
        (𝓝 (Apeiron.Physics.quantumPotential hbar mass (Real.sqrt (rho t x))
          (amplitudeLaplacian x))) ∧
    (analyticSchrodingerResidual hbar mass potential rho action t x = 0 ↔
      hamiltonJacobiResidual hbar mass (potential t x) (Real.sqrt (rho t x)) actionTime
          (actionSpace x) (amplitudeLaplacian x) = 0 ∧
      continuityResidual mass (Real.sqrt (rho t x)) amplitudeTime (amplitudeSpace x)
          (actionSpace x) (actionLaplacian x) = 0) := by
  have hderiv : deriv (deriv (fun z => Real.sqrt (rho t z))) x = amplitudeLaplacian x := by
    have h1 : deriv (fun z => Real.sqrt (rho t z)) = amplitudeSpace :=
      funext fun y => (hAmplitudeSpace y).deriv
    rw [h1]
    exact (hAmplitudeLaplacian x).deriv
  have ha := branchQ_tendsto hbar mass hC2 x
  rw [hderiv] at ha
  refine ⟨ha, ?_⟩
  exact actual_schrodinger_iff_madelung hbar mass potential rho action t x amplitudeTime actionTime
    amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian hHbar hMass hRho hAmplitudeTime
    hActionTime hAmplitudeSpace hAmplitudeLaplacian hActionSpace hActionLaplacian

end Numeric.BranchMadelung
