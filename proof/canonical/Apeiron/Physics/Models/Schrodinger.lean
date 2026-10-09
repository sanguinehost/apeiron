import Apeiron.Physics.Madelung
import Mathlib.Analysis.InnerProductSpace.Laplacian

/-!
# Polar Schrödinger residuals

@name Schrodinger polar decomposition
@cite schrodinger-1926
@cite madelung-1927
@cite bohm-1952
@cite vattay-2026-comment
@cite vattay-2026-reply
@grade proved

This module is the algebraic layer of the one-dimensional polar decomposition. Its time and
space data are finite jets, not derivatives constructed in this file. The analytic realization
of the jets is separate. The elliptic flow is the phase carrier, and the factorization below
states exactly which two real residuals make the complex residual zero.

## Root

`polarWave_time_derivative_eq_jet` uses `Trichotomy.hasDerivAt_flow_fst` and
`Trichotomy.hasDerivAt_flow_snd`. `polarWave_space_derivative_eq_jet` uses
`Physics.secondDeriv_waveFst` and `Physics.secondDeriv_waveSnd`. The common elliptic-flow factor
then carries `actual_schrodinger_iff_madelung`; its positive-density premise supplies the nonzero
amplitude required to recover both real coordinates. Mathlib's
`InnerProductSpace.laplacian_eq_iteratedDeriv_real` identifies the named second spatial derivative
with the Laplacian on the real line.

The identity is Madelung's established decomposition. The checked result fixes the exact
one-dimensional hypotheses and exhibits the zero-density failure boundary relevant to Vattay's
criticism. It does not establish novelty, a multidimensional PDE, a propagator, or Van Vleck
exactness.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- The polar wave at one jet point.

    (hbar amplitude action : ℝ)                    [Planck scale, amplitude, and action]
    Cell                                            [the two real slots of the elliptic wave]
    mul (-1) (flow (-1) (action / hbar)) (amplitude, 0)
                                                    [the amplitude carried by the elliptic flow]
    flow                                            [the size-one flow of the algebra]
    mul                                             [multiplication in the elliptic algebra]

    So: the phase is `action / hbar`, and the amplitude is the real point `(amplitude, 0)`. -/
noncomputable def polarWave (hbar amplitude action : ℝ) : Cell :=
  mul (-1) (flow (-1) (action / hbar)) (amplitude, 0)

/-- The polar wave's time jet.

    (hbar amplitude action amplitudeTime actionTime : ℝ)
                                                    [the scale, polar data, and time rates]
    ℝ                                               [the real numbers]
    Cell                                            [the two slots of the time jet]
    mul (-1) (flow (-1) (action / hbar))
      (amplitudeTime, amplitude * actionTime / hbar)
                                                    [the phase carrier times its time coefficient]
    flow                                            [the size-one flow of the algebra]
    mul                                             [multiplication in the elliptic algebra]

    So: this is the first finite jet of `amplitude · exp(i action / hbar)`. -/
noncomputable def polarTimeJet
    (hbar amplitude action amplitudeTime actionTime : ℝ) : Cell :=
  mul (-1) (flow (-1) (action / hbar))
    (amplitudeTime, amplitude * actionTime / hbar)

/-- The polar wave's spatial second jet.

    (hbar amplitude action amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ)
                                                    [the scale and the amplitude and action jets]
    ℝ                                               [the real numbers]
    Cell                                            [the two slots of the spatial second jet]
    mul (-1) (flow (-1) (action / hbar))
      (amplitudeLaplacian - amplitude * (actionSpace / hbar) ^ 2,
        2 * amplitudeSpace * actionSpace / hbar + amplitude * actionLaplacian / hbar)
                                                    [the phase carrier times the
                                                     curvature coefficient]
    flow                                            [the size-one flow of the algebra]
    mul                                             [multiplication in the elliptic algebra]

    So: this is the second finite spatial jet of the polar wave. -/
noncomputable def polarSpaceJet
    (hbar amplitude action amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ) :
    Cell :=
  mul (-1) (flow (-1) (action / hbar))
    (amplitudeLaplacian - amplitude * (actionSpace / hbar) ^ 2,
      2 * amplitudeSpace * actionSpace / hbar + amplitude * actionLaplacian / hbar)

/-- The Bohm quantum-potential coefficient at one jet point.

    (hbar mass amplitude amplitudeLaplacian : ℝ)    [the scales, amplitude, and its Laplacian]
    ℝ                                               [the real quantum-potential value]
    -(hbar ^ 2 / (2 * mass)) * (amplitudeLaplacian / amplitude)
                                                    [minus the scaled curvature-to-amplitude ratio]

    So: the ratio is partial at zero amplitude in the physical reading; Lean division remains
    total. -/
noncomputable def quantumPotential (hbar mass amplitude amplitudeLaplacian : ℝ) : ℝ :=
  -(hbar ^ 2 / (2 * mass)) * (amplitudeLaplacian / amplitude)

/-- The Hamilton-Jacobi residual of a polar jet.

    (hbar mass potential amplitude actionTime actionSpace amplitudeLaplacian : ℝ)
                                                    [the scales, potential, and polar jet values]
    ℝ                                               [the real Hamilton-Jacobi residual]
    actionTime + actionSpace ^ 2 / (2 * mass) + potential
      + quantumPotential hbar mass amplitude amplitudeLaplacian
                                                    [action rate, kinetic, potential, and
                                                     quantum terms]
    quantumPotential                               [the curvature contribution]

    So: zero is the Hamilton-Jacobi equation with the Bohm potential. -/
noncomputable def hamiltonJacobiResidual
    (hbar mass potential amplitude actionTime actionSpace amplitudeLaplacian : ℝ) : ℝ :=
  actionTime + actionSpace ^ 2 / (2 * mass) + potential
    + quantumPotential hbar mass amplitude amplitudeLaplacian

/-- The density-current continuity residual of a polar jet.

    (mass amplitude amplitudeTime amplitudeSpace actionSpace actionLaplacian : ℝ)
                                                    [the mass and the amplitude and action jets]
    ℝ                                               [the real continuity residual]
    2 * amplitude * amplitudeTime
      + (2 * amplitude * amplitudeSpace * actionSpace + amplitude ^ 2 * actionLaplacian) / mass
                                                    [the density rate plus current divergence]

    So: this is `ρ_t + (ρ actionSpace / mass)_x` for `ρ = amplitude²`. -/
noncomputable def continuityResidual
    (mass amplitude amplitudeTime amplitudeSpace actionSpace actionLaplacian : ℝ) : ℝ :=
  2 * amplitude * amplitudeTime
    + (2 * amplitude * amplitudeSpace * actionSpace + amplitude ^ 2 * actionLaplacian) / mass

/-- The Schrödinger residual assembled from finite polar jets.

    (hbar mass potential amplitude action amplitudeTime actionTime : ℝ)
                                                    [the scales, potential, and time jet]
    (amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ)
                                                    [the spatial jet]
    ℝ                                               [the real numbers]
    Cell                                            [the elliptic residual]
    i*hbar*polarTimeJet + hbar²/(2*mass)*polarSpaceJet - potential*polarWave
                                                    [the three Schrödinger terms]
    polarWave                                       [the polar wave value]

    So: this is the finite-jet Schrödinger operator, represented as a two-slot elliptic point. -/
noncomputable def schrodingerResidual
    (hbar mass potential amplitude action amplitudeTime actionTime amplitudeSpace
      amplitudeLaplacian actionSpace actionLaplacian : ℝ) : Cell :=
  let timeTerm := mul (-1) (0, hbar)
    (polarTimeJet hbar amplitude action amplitudeTime actionTime)
  let spaceTerm := polarSpaceJet hbar amplitude action amplitudeSpace amplitudeLaplacian
    actionSpace actionLaplacian
  let waveTerm := polarWave hbar amplitude action
  (timeTerm.1 + hbar ^ 2 / (2 * mass) * spaceTerm.1 - potential * waveTerm.1,
    timeTerm.2 + hbar ^ 2 / (2 * mass) * spaceTerm.2 - potential * waveTerm.2)

/-- The polar wave has squared amplitude as its elliptic norm form.

    (hbar amplitude action : ℝ)                    [the Planck scale, amplitude, and action]
    normForm (-1) (polarWave hbar amplitude action) [the elliptic norm of the polar wave]
    = amplitude ^ 2                                [the squared amplitude]
    normForm                                       [the quadratic size form]
    polarWave                                      [the polar wave value]

    So: the phase flow has norm one, so it does not change the wave's squared amplitude. -/
theorem normForm_polarWave (hbar amplitude action : ℝ) :
    normForm (-1) (polarWave hbar amplitude action) = amplitude ^ 2 := by
  unfold polarWave
  rw [normForm_mul, normForm_flow]
  simp only [normForm]
  ring

/-- The Schrödinger residual factors through the two polar residuals.

    (hbar mass potential amplitude action : ℝ)     [the scales, potential, amplitude, and action]
    (amplitudeTime actionTime amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ)
                                                    [the finite time and space jet values]
    hbar ≠ 0                                       [the phase scale is nonzero]
    mass ≠ 0                                       [the mass scale is nonzero]
    amplitude ≠ 0                                  [the polar amplitude is nonzero]
    schrodingerResidual ...
      = mul (-1) (flow (-1) (action / hbar))
        (-amplitude * hamiltonJacobiResidual ..., hbar / (2 * amplitude) * continuityResidual ...)
                                                    [the factored residual]
    flow                                            [the elliptic phase carrier]
    mul                                             [multiplication in the elliptic algebra]
    Real.sqrt                                      [the number that squares to give this]

    So: the first factor is minus amplitude times Hamilton-Jacobi; the second rescales
    continuity. -/
theorem schrodingerResidual_factorization
    (hbar mass potential amplitude action amplitudeTime actionTime amplitudeSpace
      amplitudeLaplacian actionSpace actionLaplacian : ℝ)
    (hHbar : hbar ≠ 0) (hMass : mass ≠ 0) (hAmplitude : amplitude ≠ 0) :
    schrodingerResidual hbar mass potential amplitude action amplitudeTime actionTime amplitudeSpace
      amplitudeLaplacian actionSpace actionLaplacian =
      mul (-1) (flow (-1) (action / hbar))
        (-amplitude * hamiltonJacobiResidual hbar mass potential amplitude actionTime actionSpace
            amplitudeLaplacian,
          hbar / (2 * amplitude) * continuityResidual mass amplitude amplitudeTime amplitudeSpace
            actionSpace actionLaplacian) := by
  simp only [schrodingerResidual, polarTimeJet, polarSpaceJet, polarWave, mul,
    hamiltonJacobiResidual, quantumPotential, continuityResidual]
  ext <;> field_simp [hHbar, hMass, hAmplitude] <;> ring

/-- The polar Schrödinger residual vanishes exactly when both Madelung residuals vanish.

    (hbar mass potential amplitude action : ℝ)     [the scales, potential, amplitude, and action]
    (amplitudeTime actionTime amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ)
                                                    [the finite time and space jet values]
    0 < hbar ∧ 0 < mass ∧ 0 < amplitude             [the positive polar domain]
    schrodingerResidual ... = 0
      ↔ hamiltonJacobiResidual ... = 0 ∧ continuityResidual ... = 0
                                                    [the equivalent real residual equations]
    ↔                                               [exactly when, true in both directions]
    normForm                                        [the flow's nonzero elliptic norm]
    flow                                            [the elliptic phase carrier]

    So: positivity supplies every denominator and prevents the zero-amplitude false implication. -/
theorem schrodinger_polar_iff_madelung
    (hbar mass potential amplitude action amplitudeTime actionTime amplitudeSpace
      amplitudeLaplacian actionSpace actionLaplacian : ℝ)
    (hHbar : 0 < hbar) (hMass : 0 < mass) (hAmplitude : 0 < amplitude) :
    schrodingerResidual hbar mass potential amplitude action amplitudeTime actionTime amplitudeSpace
      amplitudeLaplacian actionSpace actionLaplacian = 0 ↔
      hamiltonJacobiResidual hbar mass potential amplitude actionTime actionSpace
        amplitudeLaplacian = 0 ∧
      continuityResidual mass amplitude amplitudeTime amplitudeSpace actionSpace
        actionLaplacian = 0 := by
  have hHbar0 : hbar ≠ 0 := ne_of_gt hHbar
  have hMass0 : mass ≠ 0 := ne_of_gt hMass
  have hAmplitude0 : amplitude ≠ 0 := ne_of_gt hAmplitude
  have hFlow : flow (-1) (action / hbar) ≠ 0 := by
    intro hzero
    have hnorm := normForm_flow (-1) (action / hbar)
    rw [hzero] at hnorm
    norm_num [normForm] at hnorm
  rw [schrodingerResidual_factorization hbar mass potential amplitude action amplitudeTime
    actionTime amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian hHbar0 hMass0
    hAmplitude0]
  constructor
  · intro hResidual
    rcases eq_zero_or_eq_zero_of_mul_eq_zero_of_neg (by norm_num : (-1 : ℝ) < 0)
      (flow (-1) (action / hbar))
      (-amplitude * hamiltonJacobiResidual hbar mass potential amplitude actionTime actionSpace
        amplitudeLaplacian,
        hbar / (2 * amplitude) * continuityResidual mass amplitude amplitudeTime amplitudeSpace
          actionSpace actionLaplacian) hResidual with hzero | hzero
    · exact False.elim (hFlow hzero)
    · constructor
      · have hfirst := congrArg Prod.fst hzero
        have hfirst' : -amplitude * hamiltonJacobiResidual hbar mass potential amplitude actionTime
            actionSpace amplitudeLaplacian = 0 := by simpa using hfirst
        have hproduct : amplitude * hamiltonJacobiResidual hbar mass potential amplitude actionTime
            actionSpace amplitudeLaplacian = 0 := by linarith
        exact (mul_eq_zero.mp hproduct).resolve_left hAmplitude0
      · have hsecond := congrArg Prod.snd hzero
        have hcoefficient : hbar / (2 * amplitude) ≠ 0 :=
          div_ne_zero hHbar0 (mul_ne_zero (by norm_num) hAmplitude0)
        exact (mul_eq_zero.mp hsecond).resolve_left hcoefficient
  · rintro ⟨hHamiltonJacobi, hContinuity⟩
    simp [hHamiltonJacobi, hContinuity, mul]

/-- The quantum potential is the Laplacian-to-value ratio of the square-root density jet.

    (hbar mass density laplacianSqrtDensity : ℝ)    [the scales, density, and square-root Laplacian]
    quantumPotential hbar mass (Real.sqrt density) laplacianSqrtDensity
                                                    [the potential evaluated at square-root density]
    = -(hbar ^ 2 / (2 * mass)) * (laplacianSqrtDensity / Real.sqrt density)
                                                    [the Bohm Laplacian quotient]
    quantumPotential                               [the quantum-potential coefficient]
    Real.sqrt                                      [the nonnegative square root]

    So: the derivative information is the supplied Laplacian jet; this theorem is algebraic. -/
theorem quantumPotential_eq_laplacian_sqrt_density
    (hbar mass density laplacianSqrtDensity : ℝ) :
    quantumPotential hbar mass (Real.sqrt density) laplacianSqrtDensity =
      -(hbar ^ 2 / (2 * mass)) * (laplacianSqrtDensity / Real.sqrt density) := rfl

/-- In one spatial dimension the named amplitude Laplacian is Mathlib's actual Laplacian.

    (hbar mass : ℝ)                              [the Planck and mass scales]
    (rho : ℝ → ℝ → ℝ)                           [the density field]
    (t x : ℝ)                                    [the time and spatial point]
    (amplitudeSpace amplitudeLaplacian : ℝ → ℝ)  [the first and second amplitude derivatives]
    ∀ y, HasDerivAt ...                          [the derivative witnesses]
    quantumPotential ... =
      -(hbar²/(2*mass)) * Laplacian(sqrt rho) / sqrt rho
                                                   [the quantum-potential formula]
    Laplacian.laplacian                           [Mathlib's Laplacian]
    Real.sqrt                                     [the nonnegative square root]

    So: the second spatial derivative used by the polar jet is the one-dimensional
    Laplacian, not an unrelated supplied number. -/
theorem quantumPotential_eq_actual_laplacian_sqrt_density
    (hbar mass : ℝ) (rho : ℝ → ℝ → ℝ) (t x : ℝ)
    (amplitudeSpace amplitudeLaplacian : ℝ → ℝ)
    (hAmplitudeSpace : ∀ y,
      HasDerivAt (fun z => Real.sqrt (rho t z)) (amplitudeSpace y) y)
    (hAmplitudeLaplacian : ∀ y,
      HasDerivAt amplitudeSpace (amplitudeLaplacian y) y) :
    quantumPotential hbar mass (Real.sqrt (rho t x)) (amplitudeLaplacian x) =
      -(hbar ^ 2 / (2 * mass)) *
        (Laplacian.laplacian (fun y => Real.sqrt (rho t y)) x / Real.sqrt (rho t x)) := by
  have hFirst : deriv (fun y => Real.sqrt (rho t y)) = amplitudeSpace :=
    funext fun y => (hAmplitudeSpace y).deriv
  have hSecond : deriv (deriv (fun y => Real.sqrt (rho t y))) x = amplitudeLaplacian x := by
    rw [hFirst]
    exact (hAmplitudeLaplacian x).deriv
  have hLaplacian : Laplacian.laplacian (fun y => Real.sqrt (rho t y)) x =
      amplitudeLaplacian x := by
    rw [InnerProductSpace.laplacian_eq_iteratedDeriv_real]
    simpa [iteratedDeriv_succ, iteratedDeriv_zero] using hSecond
  rw [hLaplacian]
  rfl

/-- Zero amplitude makes the Madelung residuals vanish while the Schrödinger residual remains
nonzero.

    schrodingerResidual 1 1 0 0 0 0 0 0 1 0 0 ≠ 0
                                                    [the nonzero Schrödinger residual]
    hamiltonJacobiResidual 1 1 0 0 0 0 1 = 0       [the zero Hamilton-Jacobi residual]
    continuityResidual 1 0 0 0 0 0 = 0             [the zero continuity residual]
    ∧                                               [both are true]
    ≠                                               [is not equal to]
    schrodingerResidual                             [the finite-jet Schrödinger operator]

    So: amplitude positivity in `schrodinger_polar_iff_madelung` is load-bearing. -/
theorem zero_amplitude_counterexample :
    schrodingerResidual 1 1 0 0 0 0 0 0 1 0 0 ≠ 0 ∧
      hamiltonJacobiResidual 1 1 0 0 0 0 1 = 0 ∧
      continuityResidual 1 0 0 0 0 0 = 0 := by
  norm_num [schrodingerResidual, polarTimeJet, polarSpaceJet, polarWave,
    hamiltonJacobiResidual, quantumPotential, continuityResidual, mul, flow]

/-- The componentwise time derivative of a bivariate polar wave.

    (hbar : ℝ)                                    [the Planck scale]
    (rho action : ℝ → ℝ → ℝ)                      [the density and action fields]
    (t x : ℝ)                                     [the time and spatial point]
    Cell                                           [the two componentwise time derivatives]
    deriv                                          [the one-variable derivative]
    polarWave                                      [the polar wave]

    So: the spatial coordinate is fixed while each component is differentiated in time. -/
noncomputable def polarWaveTimeDerivative
    (hbar : ℝ) (rho action : ℝ → ℝ → ℝ) (t x : ℝ) : Cell :=
  (deriv (fun s => (polarWave hbar (Real.sqrt (rho s x)) (action s x)).1) t,
    deriv (fun s => (polarWave hbar (Real.sqrt (rho s x)) (action s x)).2) t)

/-- The componentwise iterated spatial derivative of a bivariate polar wave.

    (hbar : ℝ)                                    [the Planck scale]
    (rho action : ℝ → ℝ → ℝ)                      [the density and action fields]
    (t x : ℝ)                                     [the time and spatial point]
    Cell                                           [the componentwise spatial second derivatives]
    deriv                                          [the one-variable derivative]
    polarWave                                      [the polar wave]

    So: the time coordinate is fixed while each component is differentiated twice in space. -/
noncomputable def polarWaveSpaceDerivative
    (hbar : ℝ) (rho action : ℝ → ℝ → ℝ) (t x : ℝ) : Cell :=
  (deriv (deriv (fun y => (polarWave hbar (Real.sqrt (rho t y)) (action t y)).1)) x,
    deriv (deriv (fun y => (polarWave hbar (Real.sqrt (rho t y)) (action t y)).2)) x)

/-- The Schrödinger residual formed from the actual componentwise derivatives.

    (hbar mass : ℝ)                               [the Planck and mass scales]
    (potential rho action : ℝ → ℝ → ℝ)            [the potential, density, and action fields]
    (t x : ℝ)                                     [the point where the residual is read]
    Cell                                           [the actual pointwise Schrödinger residual]
    polarWaveTimeDerivative                       [the actual time jet]
    polarWaveSpaceDerivative                      [the actual spatial second jet]

    So: this is the same operator as `schrodingerResidual`, with derivatives rather than
    supplied jets. -/
noncomputable def analyticSchrodingerResidual
    (hbar mass : ℝ) (potential rho action : ℝ → ℝ → ℝ) (t x : ℝ) : Cell :=
  let timeTerm := mul (-1) (0, hbar) (polarWaveTimeDerivative hbar rho action t x)
  let spaceTerm := polarWaveSpaceDerivative hbar rho action t x
  let waveTerm := polarWave hbar (Real.sqrt (rho t x)) (action t x)
  (timeTerm.1 + hbar ^ 2 / (2 * mass) * spaceTerm.1 - potential t x * waveTerm.1,
    timeTerm.2 + hbar ^ 2 / (2 * mass) * spaceTerm.2 - potential t x * waveTerm.2)

/-- The actual componentwise time derivative is the supplied polar time jet.

    (hbar : ℝ)                                    [the Planck scale]
    (rho action : ℝ → ℝ → ℝ)                      [the density and action fields]
    (t x amplitudeTime actionTime : ℝ)             [the point and named time-jet values]
    hbar ≠ 0                                       [the phase scale is nonzero]
    HasDerivAt (fun s => Real.sqrt (rho s x)) amplitudeTime t
                                                    [the local amplitude derivative]
    HasDerivAt (fun s => action s x) actionTime t  [the local action derivative]
    polarWaveTimeDerivative hbar rho action t x
      = polarTimeJet hbar (Real.sqrt (rho t x)) (action t x) amplitudeTime actionTime
                                                    [the time jet equality]
    flow                                            [the elliptic phase carrier]
    mul                                             [multiplication in the elliptic algebra]
    HasDerivAt                                      [is changing at that rate at that time]
    Real.sqrt                                        [the number that squares to give this]
    ∀                                               [for every single one]

    So: local time derivatives suffice because this statement differentiates at one fixed point. -/
theorem polarWave_time_derivative_eq_jet
    (hbar : ℝ) (rho action : ℝ → ℝ → ℝ) (t x amplitudeTime actionTime : ℝ)
    (hHbar : hbar ≠ 0)
    (hAmplitude : HasDerivAt (fun s => Real.sqrt (rho s x)) amplitudeTime t)
    (hAction : HasDerivAt (fun s => action s x) actionTime t) :
    polarWaveTimeDerivative hbar rho action t x =
      polarTimeJet hbar (Real.sqrt (rho t x)) (action t x) amplitudeTime actionTime := by
  have hPhase : HasDerivAt (fun s => action s x / hbar) (actionTime / hbar) t :=
    hAction.div_const hbar
  have hFirst := hAmplitude.mul
    ((hasDerivAt_flow_fst (-1) (action t x / hbar)).comp t hPhase)
  have hSecond := hAmplitude.mul
    ((hasDerivAt_flow_snd (-1) (action t x / hbar)).comp t hPhase)
  apply Prod.ext
  · have hValue : amplitudeTime * (flow (-1) (action t x / hbar)).1 +
        Real.sqrt (rho t x) * ((-1) * (flow (-1) (action t x / hbar)).2 * (actionTime / hbar)) =
        (polarTimeJet hbar (Real.sqrt (rho t x)) (action t x) amplitudeTime actionTime).1 := by
      simp only [polarTimeJet, mul]
      field_simp [hHbar]
    change deriv (fun s => (polarWave hbar (Real.sqrt (rho s x)) (action s x)).1) t = _
    have hFunction : ((fun s => Real.sqrt (rho s x)) * (fun u => (flow (-1) u).1) ∘
        fun s => action s x / hbar) =
        (fun s => (polarWave hbar (Real.sqrt (rho s x)) (action s x)).1) := by
      funext s
      simp only [Function.comp_apply, Pi.mul_apply, polarWave, mul]
      ring
    rw [← hFunction]
    exact (hFirst.congr_deriv hValue).deriv
  · have hValue : amplitudeTime * (flow (-1) (action t x / hbar)).2 +
        Real.sqrt (rho t x) * ((flow (-1) (action t x / hbar)).1 * (actionTime / hbar)) =
        (polarTimeJet hbar (Real.sqrt (rho t x)) (action t x) amplitudeTime actionTime).2 := by
      simp only [polarTimeJet, mul]
      field_simp [hHbar]; ring
    change deriv (fun s => (polarWave hbar (Real.sqrt (rho s x)) (action s x)).2) t = _
    have hFunction : ((fun s => Real.sqrt (rho s x)) * (fun u => (flow (-1) u).2) ∘
        fun s => action s x / hbar) =
        (fun s => (polarWave hbar (Real.sqrt (rho s x)) (action s x)).2) := by
      funext s
      simp only [Function.comp_apply, Pi.mul_apply, polarWave, mul]
      ring
    rw [← hFunction]
    exact (hSecond.congr_deriv hValue).deriv

/-- The actual componentwise spatial derivative is the supplied polar space jet.

    (hbar : ℝ)                                    [the Planck scale]
    (rho action : ℝ → ℝ → ℝ)                      [the density and action fields]
    (t x : ℝ)                                     [the point where the spatial jet is read]
    (amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ → ℝ)
                                                    [the named first and second spatial jets]
    hbar ≠ 0                                       [the phase scale is nonzero]
    ∀ y, HasDerivAt (fun z => Real.sqrt (rho t z)) (amplitudeSpace y) y
                                                    [the amplitude first-derivative family]
    ∀ y, HasDerivAt amplitudeSpace (amplitudeLaplacian y) y
                                                    [the amplitude second-derivative family]
    ∀ y, HasDerivAt (fun z => action t z) (actionSpace y) y
                                                    [the action first-derivative family]
    ∀ y, HasDerivAt actionSpace (actionLaplacian y) y
                                                    [the action second-derivative family]
    polarWaveSpaceDerivative hbar rho action t x
      = polarSpaceJet hbar (Real.sqrt (rho t x)) (action t x) (amplitudeSpace x)
          (amplitudeLaplacian x) (actionSpace x) (actionLaplacian x)
                                                    [the spatial jet equality]
    flow                                            [the elliptic phase carrier]
    mul                                             [multiplication in the elliptic algebra]
    HasDerivAt                                      [is changing at that rate at that time]
    Real.sqrt                                        [the number that squares to give this]
    ∀                                               [for every single one]

    So: the full spatial families make the iterated derivative at `x` an actual second
    derivative. -/
theorem polarWave_space_derivative_eq_jet
    (hbar : ℝ) (rho action : ℝ → ℝ → ℝ) (t x : ℝ)
    (amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ → ℝ)
    (hHbar : hbar ≠ 0)
    (hAmplitude : ∀ y, HasDerivAt (fun z => Real.sqrt (rho t z)) (amplitudeSpace y) y)
    (hAmplitudeSpace : ∀ y, HasDerivAt amplitudeSpace (amplitudeLaplacian y) y)
    (hAction : ∀ y, HasDerivAt (fun z => action t z) (actionSpace y) y)
    (hActionSpace : ∀ y, HasDerivAt actionSpace (actionLaplacian y) y) :
    polarWaveSpaceDerivative hbar rho action t x =
      polarSpaceJet hbar (Real.sqrt (rho t x)) (action t x) (amplitudeSpace x)
        (amplitudeLaplacian x) (actionSpace x) (actionLaplacian x) := by
  have hPhase : ∀ y, HasDerivAt (fun z => action t z / hbar) (actionSpace y / hbar) y :=
    fun y => (hAction y).div_const hbar
  have hPhaseSpace : ∀ y, HasDerivAt (fun z => actionSpace z / hbar)
      (actionLaplacian y / hbar) y := fun y => (hActionSpace y).div_const hbar
  have hFirst := secondDeriv_waveFst (-1) (fun y => Real.sqrt (rho t y))
    (fun y => action t y / hbar) amplitudeSpace (fun y => actionSpace y / hbar)
    amplitudeLaplacian (fun y => actionLaplacian y / hbar)
    hAmplitude hAmplitudeSpace hPhase hPhaseSpace x
  have hSecond := secondDeriv_waveSnd (-1) (fun y => Real.sqrt (rho t y))
    (fun y => action t y / hbar) amplitudeSpace (fun y => actionSpace y / hbar)
    amplitudeLaplacian (fun y => actionLaplacian y / hbar)
    hAmplitude hAmplitudeSpace hPhase hPhaseSpace x
  apply Prod.ext
  · change deriv (deriv (fun y => (polarWave hbar (Real.sqrt (rho t y)) (action t y)).1)) x = _
    have hFunction : (fun y => (polarWave hbar (Real.sqrt (rho t y)) (action t y)).1) =
        waveFst (-1) (fun y => Real.sqrt (rho t y)) (fun y => action t y / hbar) := by
      funext y
      simp only [polarWave, waveFst, mul]
      ring
    have hJet : (mul (-1) (flow (-1) (action t x / hbar))
        (curv (-1) (fun y => Real.sqrt (rho t y)) amplitudeSpace amplitudeLaplacian
          (fun y => actionSpace y / hbar) (fun y => actionLaplacian y / hbar) x)).1 =
        (polarSpaceJet hbar (Real.sqrt (rho t x)) (action t x) (amplitudeSpace x)
          (amplitudeLaplacian x) (actionSpace x) (actionLaplacian x)).1 := by
      simp only [curv, ringCurv, polarSpaceJet, mul]
      field_simp [hHbar]; ring
    rw [hFunction]
    exact hFirst.trans hJet
  · change deriv (deriv (fun y => (polarWave hbar (Real.sqrt (rho t y)) (action t y)).2)) x = _
    have hFunction : (fun y => (polarWave hbar (Real.sqrt (rho t y)) (action t y)).2) =
        waveSnd (-1) (fun y => Real.sqrt (rho t y)) (fun y => action t y / hbar) := by
      funext y
      simp only [polarWave, waveSnd, mul]
      ring
    have hJet : (mul (-1) (flow (-1) (action t x / hbar))
        (curv (-1) (fun y => Real.sqrt (rho t y)) amplitudeSpace amplitudeLaplacian
          (fun y => actionSpace y / hbar) (fun y => actionLaplacian y / hbar) x)).2 =
        (polarSpaceJet hbar (Real.sqrt (rho t x)) (action t x) (amplitudeSpace x)
          (amplitudeLaplacian x) (actionSpace x) (actionLaplacian x)).2 := by
      simp only [curv, ringCurv, polarSpaceJet, mul]
      field_simp [hHbar]; ring
    rw [hFunction]
    exact hSecond.trans hJet

/-- Continuity is the derivative of amplitude-squared density plus its current.

    (mass : ℝ)                                    [the mass scale]
    (rho : ℝ → ℝ → ℝ)                             [the density field]
    (t x amplitudeTime : ℝ)                        [the point and amplitude time rate]
    (amplitudeSpace actionSpace actionLaplacian : ℝ → ℝ)
                                                    [the named spatial jets]
    HasDerivAt (fun s => Real.sqrt (rho s x)) amplitudeTime t
                                                    [the local amplitude time derivative]
    ∀ y, HasDerivAt (fun z => Real.sqrt (rho t z)) (amplitudeSpace y) y
                                                    [the amplitude spatial derivative family]
    ∀ y, HasDerivAt actionSpace (actionLaplacian y) y
                                                    [the action spatial second-derivative family]
    continuityResidual mass ...
      = deriv (fun s => Real.sqrt (rho s x) ^ 2) t
        + deriv (fun y => Real.sqrt (rho t y) ^ 2 * actionSpace y / mass) x
                                                    [the amplitude-squared continuity equation]
    continuityResidual                              [the finite-jet continuity residual]
    deriv                                           [the actual derivative]
    HasDerivAt                                      [is changing at that rate at that time]
    Real.sqrt                                        [the number that squares to give this]
    ∀                                               [for every single one]

    So: this uses amplitude-squared density, avoiding an unjustified neighborhood equality for
    arbitrary `rho`. -/
theorem continuityResidual_eq_amplitudeSquared_current_derivative
    (mass : ℝ) (rho : ℝ → ℝ → ℝ) (t x amplitudeTime : ℝ)
    (amplitudeSpace actionSpace actionLaplacian : ℝ → ℝ)
    (hAmplitudeTime : HasDerivAt (fun s => Real.sqrt (rho s x)) amplitudeTime t)
    (hAmplitudeSpace : ∀ y, HasDerivAt (fun z => Real.sqrt (rho t z)) (amplitudeSpace y) y)
    (hActionSpace : ∀ y, HasDerivAt actionSpace (actionLaplacian y) y) :
    continuityResidual mass (Real.sqrt (rho t x)) amplitudeTime (amplitudeSpace x)
      (actionSpace x) (actionLaplacian x) =
      deriv (fun s => Real.sqrt (rho s x) * Real.sqrt (rho s x)) t +
        deriv (fun y => Real.sqrt (rho t y) * Real.sqrt (rho t y) * actionSpace y / mass) x := by
  have hDensityTime := hAmplitudeTime.mul hAmplitudeTime
  have hDensitySpace := (hAmplitudeSpace x).mul (hAmplitudeSpace x)
  have hCurrent := (hDensitySpace.mul (hActionSpace x)).div_const mass
  change continuityResidual mass (Real.sqrt (rho t x)) amplitudeTime (amplitudeSpace x)
      (actionSpace x) (actionLaplacian x) =
    deriv ((fun s => Real.sqrt (rho s x)) * (fun s => Real.sqrt (rho s x))) t +
      deriv (fun y => (((fun z => Real.sqrt (rho t z)) * (fun z => Real.sqrt (rho t z))) *
        actionSpace) y / mass) x
  rw [hDensityTime.deriv, hCurrent.deriv]
  unfold continuityResidual
  simp only [Pi.mul_apply]
  ring

/-- The actual Schrödinger residual is the algebraic residual at the realized jets.

    (hbar mass : ℝ)                               [the Planck and mass scales]
    (potential rho action : ℝ → ℝ → ℝ)            [the potential, density, and action fields]
    (t x amplitudeTime actionTime : ℝ)             [the point and named time-jet values]
    (amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ → ℝ)
                                                    [the named spatial jet families]
    hbar ≠ 0                                       [the phase scale is nonzero]
    HasDerivAt ...                                 [the local amplitude time derivative]
    HasDerivAt ...                                 [the local action time derivative]
    ∀ y, HasDerivAt ...                            [the spatial derivative families]
    analyticSchrodingerResidual ... = schrodingerResidual ...
                                                    [the realized residual equality]
    polarWaveTimeDerivative                        [the actual time jet]
    polarWaveSpaceDerivative                       [the actual spatial second jet]
    Real.sqrt                                      [the number that squares to give this]

    So: the analytic hypotheses identify both actual derivative cells with the finite jets. -/
theorem analyticSchrodingerResidual_eq_jet
    (hbar mass : ℝ) (potential rho action : ℝ → ℝ → ℝ) (t x amplitudeTime actionTime : ℝ)
    (amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ → ℝ)
    (hHbar : hbar ≠ 0)
    (hAmplitudeTime : HasDerivAt (fun s => Real.sqrt (rho s x)) amplitudeTime t)
    (hActionTime : HasDerivAt (fun s => action s x) actionTime t)
    (hAmplitudeSpace : ∀ y, HasDerivAt (fun z => Real.sqrt (rho t z)) (amplitudeSpace y) y)
    (hAmplitudeLaplacian : ∀ y, HasDerivAt amplitudeSpace (amplitudeLaplacian y) y)
    (hActionSpace : ∀ y, HasDerivAt (fun z => action t z) (actionSpace y) y)
    (hActionLaplacian : ∀ y, HasDerivAt actionSpace (actionLaplacian y) y) :
    analyticSchrodingerResidual hbar mass potential rho action t x =
      schrodingerResidual hbar mass (potential t x) (Real.sqrt (rho t x)) (action t x)
        amplitudeTime actionTime (amplitudeSpace x) (amplitudeLaplacian x)
        (actionSpace x) (actionLaplacian x) := by
  unfold analyticSchrodingerResidual
  rw [show polarWaveTimeDerivative hbar rho action t x =
      polarTimeJet hbar (Real.sqrt (rho t x)) (action t x) amplitudeTime actionTime from
    polarWave_time_derivative_eq_jet hbar rho action t x amplitudeTime actionTime hHbar
      hAmplitudeTime hActionTime]
  rw [show polarWaveSpaceDerivative hbar rho action t x =
      polarSpaceJet hbar (Real.sqrt (rho t x)) (action t x) (amplitudeSpace x)
        (amplitudeLaplacian x) (actionSpace x) (actionLaplacian x) from
    polarWave_space_derivative_eq_jet hbar rho action t x amplitudeSpace amplitudeLaplacian
      actionSpace actionLaplacian hHbar hAmplitudeSpace hAmplitudeLaplacian hActionSpace
        hActionLaplacian]
  rfl

/-- The actual polar Schrödinger residual vanishes exactly when both realized Madelung residuals
vanish.

    (hbar mass : ℝ)                               [the Planck and mass scales]
    (potential rho action : ℝ → ℝ → ℝ)            [the potential, density, and action fields]
    (t x amplitudeTime actionTime : ℝ)             [the point and named time-jet values]
    (amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ → ℝ)
                                                    [the named spatial jet families]
    0 < hbar ∧ 0 < mass ∧ 0 < rho t x              [the positive physical domain]
    analyticSchrodingerResidual ... = 0
      ↔ hamiltonJacobiResidual ... = 0 ∧ continuityResidual ... = 0
                                                    [the equivalent realized residual equations]
    analyticSchrodingerResidual                    [the actual pointwise Schrödinger residual]
    continuityResidual                              [the finite-jet continuity residual]
    HasDerivAt                                      [is changing at that rate at that time]
    Real.sqrt                                        [the number that squares to give this]
    ∀                                               [for every single one]
    ↔                                               [exactly when, true in both directions]

    So: `Real.sqrt` turns positive density into the positive amplitude required by the
    algebraic theorem. -/
theorem actual_schrodinger_iff_madelung
    (hbar mass : ℝ) (potential rho action : ℝ → ℝ → ℝ) (t x amplitudeTime actionTime : ℝ)
    (amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian : ℝ → ℝ)
    (hHbar : 0 < hbar) (hMass : 0 < mass) (hRho : 0 < rho t x)
    (hAmplitudeTime : HasDerivAt (fun s => Real.sqrt (rho s x)) amplitudeTime t)
    (hActionTime : HasDerivAt (fun s => action s x) actionTime t)
    (hAmplitudeSpace : ∀ y, HasDerivAt (fun z => Real.sqrt (rho t z)) (amplitudeSpace y) y)
    (hAmplitudeLaplacian : ∀ y, HasDerivAt amplitudeSpace (amplitudeLaplacian y) y)
    (hActionSpace : ∀ y, HasDerivAt (fun z => action t z) (actionSpace y) y)
    (hActionLaplacian : ∀ y, HasDerivAt actionSpace (actionLaplacian y) y) :
    analyticSchrodingerResidual hbar mass potential rho action t x = 0 ↔
      hamiltonJacobiResidual hbar mass (potential t x) (Real.sqrt (rho t x)) actionTime
          (actionSpace x) (amplitudeLaplacian x) = 0 ∧
      continuityResidual mass (Real.sqrt (rho t x)) amplitudeTime (amplitudeSpace x)
          (actionSpace x) (actionLaplacian x) = 0 := by
  rw [analyticSchrodingerResidual_eq_jet hbar mass potential rho action t x amplitudeTime actionTime
    amplitudeSpace amplitudeLaplacian actionSpace actionLaplacian (ne_of_gt hHbar) hAmplitudeTime
    hActionTime hAmplitudeSpace hAmplitudeLaplacian hActionSpace hActionLaplacian]
  exact schrodinger_polar_iff_madelung hbar mass (potential t x) (Real.sqrt (rho t x)) (action t x)
    amplitudeTime actionTime (amplitudeSpace x) (amplitudeLaplacian x) (actionSpace x)
    (actionLaplacian x) hHbar hMass (Real.sqrt_pos.2 hRho)

/-- The constant positive wave supplies every analytic and residual witness jointly.

    (t x : ℝ)                                     [the time and spatial point]
    rho = fun _ _ => 1                            [the constant unit density]
    action = fun _ _ => 0                         [the constant zero action]
    potential = fun _ _ => 0                      [the constant zero potential]
    hbar = mass = 1                               [the unit scales]
    0 < rho t x                                   [the positive density witness]
    HasDerivAt ...                                [the local time derivative witnesses]
    ∀ y, HasDerivAt ...                            [the global spatial derivative witnesses]
    analyticSchrodingerResidual ... = 0            [the actual residual witness]
    hamiltonJacobiResidual ... = 0                 [the Hamilton-Jacobi witness]
    continuityResidual ... = 0                     [the continuity witness]
    HasDerivAt                                     [is changing at that rate at that time]
    Real.sqrt                                      [the number that squares to give this]
    ∀                                               [for every single one]
    ∧                                               [both are true]

    So: this packages one compatible positive-domain witness for every hypothesis of the
    headline theorem. -/
theorem schrodinger_joint_witness (t x : ℝ) :
    0 < (1 : ℝ) ∧ 0 < (1 : ℝ) ∧ 0 < (fun _ _ : ℝ => (1 : ℝ)) t x ∧
    HasDerivAt (fun s => Real.sqrt ((fun _ _ : ℝ => (1 : ℝ)) s x)) 0 t ∧
    HasDerivAt (fun s => (fun _ _ : ℝ => (0 : ℝ)) s x) 0 t ∧
    (∀ y, HasDerivAt (fun z => Real.sqrt ((fun _ _ : ℝ => (1 : ℝ)) t z)) 0 y) ∧
    (∀ y, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 y) ∧
    (∀ y, HasDerivAt (fun z => (fun _ _ : ℝ => (0 : ℝ)) t z) 0 y) ∧
    (∀ y, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 y) ∧
    analyticSchrodingerResidual 1 1 (fun _ _ : ℝ => (0 : ℝ)) (fun _ _ : ℝ => (1 : ℝ))
      (fun _ _ : ℝ => (0 : ℝ)) t x = 0 ∧
    hamiltonJacobiResidual 1 1 0 (Real.sqrt ((fun _ _ : ℝ => (1 : ℝ)) t x)) 0 0 0 = 0 ∧
    continuityResidual 1 (Real.sqrt ((fun _ _ : ℝ => (1 : ℝ)) t x)) 0 0 0 0 = 0 := by
  have hAmplitudeTime : HasDerivAt (fun s => Real.sqrt ((fun _ _ : ℝ => (1 : ℝ)) s x)) 0 t := by
    simpa using (hasDerivAt_const t (c := (1 : ℝ)))
  have hActionTime : HasDerivAt (fun s => (fun _ _ : ℝ => (0 : ℝ)) s x) 0 t := by
    simpa using (hasDerivAt_const t (c := (0 : ℝ)))
  have hAmplitudeSpace : ∀ y, HasDerivAt (fun z => Real.sqrt ((fun _ _ : ℝ => (1 : ℝ)) t z)) 0 y :=
    fun y => by simpa using (hasDerivAt_const y (c := (1 : ℝ)))
  have hZeroDerivative : ∀ y, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 y :=
    fun y => hasDerivAt_const y (c := (0 : ℝ))
  have hActionSpace : ∀ y, HasDerivAt (fun z => (fun _ _ : ℝ => (0 : ℝ)) t z) 0 y :=
    fun y => by simpa using (hasDerivAt_const y (c := (0 : ℝ)))
  refine ⟨by norm_num, by norm_num, by norm_num, hAmplitudeTime, hActionTime,
    hAmplitudeSpace, hZeroDerivative, hActionSpace, hZeroDerivative, ?_,
    by norm_num [hamiltonJacobiResidual, quantumPotential],
    by norm_num [continuityResidual]⟩
  exact (actual_schrodinger_iff_madelung 1 1 (fun _ _ : ℝ => (0 : ℝ))
    (fun _ _ : ℝ => (1 : ℝ)) (fun _ _ : ℝ => (0 : ℝ)) t x 0 0
    (fun _ : ℝ => (0 : ℝ)) (fun _ : ℝ => (0 : ℝ)) (fun _ : ℝ => (0 : ℝ))
    (fun _ : ℝ => (0 : ℝ)) (by norm_num) (by norm_num) (by norm_num) hAmplitudeTime hActionTime
    hAmplitudeSpace hZeroDerivative hActionSpace hZeroDerivative).mpr
    ⟨by norm_num [hamiltonJacobiResidual, quantumPotential], by norm_num [continuityResidual]⟩

/-- A zero wave with nonzero action time refutes the amplitude-free equivalence.

    schrodingerResidual 1 1 0 0 0 0 1 0 0 0 0 = 0
                                                    [the zero-wave Schrödinger residual]
    hamiltonJacobiResidual 1 1 0 0 1 0 0 = 1       [the nonzero Hamilton-Jacobi residual]
    continuityResidual 1 0 0 0 0 0 = 0             [the zero continuity residual]
    ∧                                               [both are true]
    schrodingerResidual                             [the finite-jet Schrödinger operator]

    So: removing the positive-amplitude premise admits this counterexample. -/
theorem zero_wave_nonzero_action_time_counterexample :
    schrodingerResidual 1 1 0 0 0 0 1 0 0 0 0 = 0 ∧
      hamiltonJacobiResidual 1 1 0 0 1 0 0 = 1 ∧
      continuityResidual 1 0 0 0 0 0 = 0 := by
  norm_num [schrodingerResidual, polarTimeJet, polarSpaceJet, polarWave,
    hamiltonJacobiResidual, quantumPotential, continuityResidual, mul, flow]

end Physics
end Apeiron
