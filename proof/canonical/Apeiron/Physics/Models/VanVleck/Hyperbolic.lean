import Apeiron.Physics.Models.VanVleck.Basic

/-!
# Hyperbolic Van Vleck kernel

@cite schrodinger-1926
@cite van-vleck-1928
@cite morette-1951
@cite feynman-hibbs-1965
@cite vattay-2026-reply
@grade proved

This module checks the inverted-oscillator polar formula at a point with positive `hbar`, `mass`,
`kappa`, and `t`. Exactness means only a zero pointwise analytic Schrödinger residual. It does
not define a propagator, evolution operator, kernel composition, unitarity, or a path integral.

## Root

`Physics.Models.VanVleck.Basic.PointwiseSchrodingerExact` names the residual equation.
`Physics.actual_schrodinger_iff_madelung` supplies its polar reduction. `Physics.det_generator`
and `Physics.generator_trichotomy` supply the hyperbolic classical classification. The mixed
derivative identifies the density independently of its definition.
-/

namespace Apeiron
namespace Physics

/-- The inverted harmonic potential.

    (mass kappa t x : ℝ)                          [the scales and point]
    ℝ                                              [the potential value]
    -(mass * kappa ^ 2 * x ^ 2) / 2               [the inverted quadratic potential]

    So: the negative potential sign is the hyperbolic face. -/
noncomputable def hyperbolicVanVleckPotential (mass kappa _t x : ℝ) : ℝ :=
  -(mass * kappa ^ 2 * x ^ 2) / 2

/-- The hyperbolic Van Vleck density.

    (hbar mass kappa t x : ℝ)                     [the scales and point]
    ℝ                                              [the density value]
    mass * kappa / (2 * Real.pi * hbar * Real.sinh (kappa * t))
                                                   [the inverted-oscillator density]
    Real.sinh                                     [the hyperbolic sine]

    So: positivity of time and scale keeps the denominator positive. -/
noncomputable def hyperbolicVanVleckDensity (hbar mass kappa t _x : ℝ) : ℝ :=
  mass * kappa / (2 * Real.pi * hbar * Real.sinh (kappa * t))

/-- The inverted-oscillator Van Vleck action from a fixed source.

    (hbar mass kappa source t x : ℝ)              [the scales, source, and point]
    ℝ                                              [the action value]
    mass * kappa * ((x ^ 2 + source ^ 2) * Real.cosh (kappa * t) - 2 * x * source)
      / (2 * Real.sinh (kappa * t)) - hbar * Real.pi / 4
                                                   [the hyperbolic action and phase offset]
    Real.cosh                                     [the hyperbolic cosine]

    So: the phase offset is constant and the remaining curvature is hyperbolic. -/
noncomputable def hyperbolicVanVleckAction (hbar mass kappa source t x : ℝ) : ℝ :=
  mass * kappa * ((x ^ 2 + source ^ 2) * Real.cosh (kappa * t) - 2 * x * source) /
      (2 * Real.sinh (kappa * t)) - hbar * Real.pi / 4

/-- The hyperbolic density is the negated actual mixed action derivative divided by `2*pi*hbar`.

    (hbar mass kappa source t x : ℝ)              [the scales, source, and point]
    0 < hbar ∧ 0 < mass ∧ 0 < kappa ∧ 0 < t      [the regular domain]
    hyperbolicVanVleckDensity hbar mass kappa t x
      = -deriv (fun position => deriv (fun origin =>
          hyperbolicVanVleckAction hbar mass kappa origin t position) source) x /
          (2 * Real.pi * hbar)                    [the independent mixed-curvature identity]
    deriv                                          [the actual derivative]
    ∧                                              [both are true]
    <                                              [is smaller than]

    So: the named density is checked against source-position curvature rather than introduced
    by it. -/
theorem hyperbolic_vanVleck_density_eq_mixed_action
    (hbar mass kappa source t x : ℝ) (hHbar : 0 < hbar) (hMass : 0 < mass) (hKappa : 0 < kappa)
    (hTime : 0 < t) :
    hyperbolicVanVleckDensity hbar mass kappa t x =
      -deriv (fun position => deriv (fun origin =>
        hyperbolicVanVleckAction hbar mass kappa origin t position) source) x /
          (2 * Real.pi * hbar) := by
  have hArgPos : 0 < kappa * t := mul_pos hKappa hTime
  have hSinhPos : 0 < Real.sinh (kappa * t) := Real.sinh_pos_iff.mpr hArgPos
  have hSinh0 : Real.sinh (kappa * t) ≠ 0 := ne_of_gt hSinhPos
  have hInner : ∀ position : ℝ, HasDerivAt
      (fun origin => hyperbolicVanVleckAction hbar mass kappa origin t position)
      (mass * kappa * (source * Real.cosh (kappa * t) - position) /
        Real.sinh (kappa * t)) source := by
    intro position
    have hId : HasDerivAt (fun origin : ℝ => origin) 1 source := hasDerivAt_id (𝕜 := ℝ) source
    have hSquare : HasDerivAt (fun origin : ℝ => origin ^ 2) (2 * source) source :=
      (hId.fun_pow 2).congr_deriv (by norm_num)
    have hQuadratic : HasDerivAt (fun origin : ℝ => position ^ 2 + origin ^ 2)
        (2 * source) source := hSquare.const_add (position ^ 2)
    have hLinear : HasDerivAt (fun origin : ℝ => 2 * position * origin) (2 * position) source := by
      simpa only [mul_one] using hId.const_mul (2 * position)
    have hNumerator : HasDerivAt
        (fun origin => (position ^ 2 + origin ^ 2) * Real.cosh (kappa * t) - 2 * position * origin)
        (2 * source * Real.cosh (kappa * t) - 2 * position) source :=
      (hQuadratic.mul_const (Real.cosh (kappa * t))).sub hLinear
    have hScaled := (hNumerator.const_mul (mass * kappa)).div_const (2 * Real.sinh (kappa * t))
    unfold hyperbolicVanVleckAction
    exact hScaled.sub_const (hbar * Real.pi / 4) |>.congr_deriv (by
      field_simp [hSinh0])
  have hOuter : HasDerivAt
      (fun position => mass * kappa * (source * Real.cosh (kappa * t) - position) /
        Real.sinh (kappa * t))
      (-mass * kappa / Real.sinh (kappa * t)) x := by
    have hId : HasDerivAt (fun position : ℝ => position) 1 x := hasDerivAt_id (𝕜 := ℝ) x
    exact ((hasDerivAt_const (𝕜 := ℝ) x (c := source * Real.cosh (kappa * t))).sub hId |>.const_mul
      (mass * kappa) |>.div_const (Real.sinh (kappa * t))).congr_deriv (by ring)
  have hInnerDeriv :
      (fun position =>
        deriv (fun origin => hyperbolicVanVleckAction hbar mass kappa origin t position) source) =
        fun position => mass * kappa * (source * Real.cosh (kappa * t) - position) /
          Real.sinh (kappa * t) := by
    funext position
    exact (hInner position).deriv
  rw [hInnerDeriv, hOuter.deriv]
  unfold hyperbolicVanVleckDensity
  field_simp [hHbar.ne', hMass.ne', hKappa.ne', hSinh0, Real.pi_ne_zero]

/-- The inverted generator is hyperbolic at `sigma = kappa^2`.

    (rep (kappa ^ 2) ((0 : ℝ), 1)).det < 0        [the negative determinant]
    ∧ 0 < kappa ^ 2                               [the hyperbolic trichotomy branch]
    rep                                            [the grid representation]
    ∧                                              [both are true]

    So: the carrier in the quantum residual is still elliptic; this is the classical-face sign. -/
theorem hyperbolic_vanVleck_face (kappa : ℝ) (hKappa : 0 < kappa) :
    (rep (kappa ^ 2) ((0 : ℝ), 1)).det < 0 ∧ 0 < kappa ^ 2 := by
  have hTri := generator_trichotomy (kappa ^ 2) 1 (by norm_num : (1 : ℝ) ≠ 0)
  constructor
  · rw [det_generator]
    nlinarith [sq_pos_of_pos hKappa]
  · exact hTri.2.2.mp (by rw [det_generator]; nlinarith [sq_pos_of_pos hKappa])

/-- The inverted Van Vleck formula has zero pointwise Schrödinger residual.

    (hbar mass kappa source t x : ℝ)              [the scales, source, and point]
    0 < hbar ∧ 0 < mass ∧ 0 < kappa ∧ 0 < t      [the regular domain]
    PointwiseSchrodingerExact hbar mass (hyperbolicVanVleckPotential mass kappa)
      (hyperbolicVanVleckDensity hbar mass kappa)
      (hyperbolicVanVleckAction hbar mass kappa source) t x
                                                   [the exactness proposition]
    PointwiseSchrodingerExact                      [the named residual predicate]
    HasDerivAt                                     [the actual derivative witnesses]
    ∧                                              [both are true]
    <                                              [is smaller than]

    So: this is a pointwise PDE theorem; it claims neither propagator nor path-integral
    exactness. -/
theorem hyperbolic_vanVleck_pointwise_exact
    (hbar mass kappa source t x : ℝ) (hHbar : 0 < hbar) (hMass : 0 < mass) (hKappa : 0 < kappa)
    (hTime : 0 < t) :
    PointwiseSchrodingerExact hbar mass (hyperbolicVanVleckPotential mass kappa)
      (hyperbolicVanVleckDensity hbar mass kappa)
      (hyperbolicVanVleckAction hbar mass kappa source) t x := by
  have hArgPos : 0 < kappa * t := mul_pos hKappa hTime
  have hSinhPos : 0 < Real.sinh (kappa * t) := Real.sinh_pos_iff.mpr hArgPos
  have hSinh0 : Real.sinh (kappa * t) ≠ 0 := ne_of_gt hSinhPos
  have hDensityPos : 0 < hyperbolicVanVleckDensity hbar mass kappa t x := by
    unfold hyperbolicVanVleckDensity
    exact div_pos (mul_pos hMass hKappa)
      (mul_pos (mul_pos (mul_pos (by norm_num) Real.pi_pos) hHbar) hSinhPos)
  have hArg : HasDerivAt (fun s : ℝ => kappa * s) kappa t := by
    have hId : HasDerivAt (fun s : ℝ => s) 1 t := hasDerivAt_id (𝕜 := ℝ) t
    simpa only [mul_one] using hId.const_mul kappa
  have hSinh : HasDerivAt (fun s : ℝ => Real.sinh (kappa * s))
      (kappa * Real.cosh (kappa * t)) t := by
    exact (Real.hasDerivAt_sinh (kappa * t)).comp t hArg |>.congr_deriv (by ring)
  have hCosh : HasDerivAt (fun s : ℝ => Real.cosh (kappa * s))
      (kappa * Real.sinh (kappa * t)) t := by
    exact (Real.hasDerivAt_cosh (kappa * t)).comp t hArg |>.congr_deriv (by ring)
  have hInv : HasDerivAt (fun s : ℝ => (Real.sinh (kappa * s))⁻¹)
      (-(kappa * Real.cosh (kappa * t)) / Real.sinh (kappa * t) ^ 2) t :=
    hSinh.fun_inv hSinh0
  have hDensityTime : HasDerivAt (fun s => hyperbolicVanVleckDensity hbar mass kappa s x)
      (-hyperbolicVanVleckDensity hbar mass kappa t x * kappa * Real.cosh (kappa * t) /
        Real.sinh (kappa * t)) t := by
    have hScaled := hInv.const_mul (mass * kappa / (2 * Real.pi * hbar))
    have hFunction : (fun s => hyperbolicVanVleckDensity hbar mass kappa s x) =
        fun s => mass * kappa / (2 * Real.pi * hbar) * (Real.sinh (kappa * s))⁻¹ := by
      funext s
      unfold hyperbolicVanVleckDensity
      field_simp [hHbar.ne', Real.pi_ne_zero]
    rw [hFunction]
    exact hScaled.congr_deriv (by
      unfold hyperbolicVanVleckDensity
      field_simp [hHbar.ne', hSinh0, Real.pi_ne_zero])
  have hAmplitudeTime : HasDerivAt
      (fun s => Real.sqrt (hyperbolicVanVleckDensity hbar mass kappa s x))
      (-Real.sqrt (hyperbolicVanVleckDensity hbar mass kappa t x) * kappa * Real.cosh (kappa * t) /
        (2 * Real.sinh (kappa * t))) t := by
    have h := hDensityTime.sqrt (ne_of_gt hDensityPos)
    have hSquare : Real.sqrt (hyperbolicVanVleckDensity hbar mass kappa t x) ^ 2 =
        hyperbolicVanVleckDensity hbar mass kappa t x := Real.sq_sqrt hDensityPos.le
    exact h.congr_deriv (by
      field_simp [hSinh0, ne_of_gt hDensityPos]
      rw [hSquare])
  have hActionTime : HasDerivAt (fun s => hyperbolicVanVleckAction hbar mass kappa source s x)
      (-mass * kappa ^ 2 * ((x ^ 2 + source ^ 2) - 2 * x * source * Real.cosh (kappa * t)) /
        (2 * Real.sinh (kappa * t) ^ 2)) t := by
    have hNumerator : HasDerivAt
        (fun s => (x ^ 2 + source ^ 2) * Real.cosh (kappa * s) - 2 * x * source)
        (kappa * (x ^ 2 + source ^ 2) * Real.sinh (kappa * t)) t :=
      (hCosh.const_mul (x ^ 2 + source ^ 2)).sub_const (2 * x * source) |>.congr_deriv (by ring)
    have hTrig :
        (x ^ 2 + source ^ 2) * Real.sinh (kappa * t) ^ 2 -
            Real.cosh (kappa * t) *
              ((x ^ 2 + source ^ 2) * Real.cosh (kappa * t) - 2 * x * source) =
          -((x ^ 2 + source ^ 2) - 2 * x * source * Real.cosh (kappa * t)) := by
      calc
        _ = -(x ^ 2 + source ^ 2) *
              (Real.cosh (kappa * t) ^ 2 - Real.sinh (kappa * t) ^ 2) +
                2 * x * source * Real.cosh (kappa * t) := by ring
        _ = -(x ^ 2 + source ^ 2) + 2 * x * source * Real.cosh (kappa * t) := by
              rw [Real.cosh_sq_sub_sinh_sq]
              ring
        _ = _ := by ring
    have hQuotient : HasDerivAt
        (fun s => mass * kappa / 2 *
          (((x ^ 2 + source ^ 2) * Real.cosh (kappa * s) - 2 * x * source) *
            (Real.sinh (kappa * s))⁻¹))
        (-mass * kappa ^ 2 *
          ((x ^ 2 + source ^ 2) - 2 * x * source * Real.cosh (kappa * t)) /
            (2 * Real.sinh (kappa * t) ^ 2)) t := by
      have hScaled := (hNumerator.fun_mul hInv).const_mul (mass * kappa / 2)
      exact hScaled.congr_deriv (by
        field_simp [hSinh0]
        simpa only [sub_eq_add_neg] using hTrig)
    have hFunction : (fun s => hyperbolicVanVleckAction hbar mass kappa source s x) =
        fun s => mass * kappa / 2 *
          (((x ^ 2 + source ^ 2) * Real.cosh (kappa * s) - 2 * x * source) *
            (Real.sinh (kappa * s))⁻¹) - hbar * Real.pi / 4 := by
      funext s
      unfold hyperbolicVanVleckAction
      field_simp
    rw [hFunction]
    exact hQuotient.sub_const (hbar * Real.pi / 4)
  have hAmplitudeSpace : ∀ z, HasDerivAt
      (fun w => Real.sqrt (hyperbolicVanVleckDensity hbar mass kappa t w)) 0 z := by
    intro z
    unfold hyperbolicVanVleckDensity
    simpa using hasDerivAt_const (𝕜 := ℝ) z
      (c := Real.sqrt (mass * kappa / (2 * Real.pi * hbar * Real.sinh (kappa * t))))
  have hZeroSpace : ∀ z, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 z :=
    fun z => hasDerivAt_const (𝕜 := ℝ) z (c := (0 : ℝ))
  have hActionSpace : ∀ z, HasDerivAt (fun w => hyperbolicVanVleckAction hbar mass kappa source t w)
      (mass * kappa * (z * Real.cosh (kappa * t) - source) / Real.sinh (kappa * t)) z := by
    intro z
    have hId : HasDerivAt (fun w : ℝ => w) 1 z := hasDerivAt_id (𝕜 := ℝ) z
    have hSquare : HasDerivAt (fun w : ℝ => w ^ 2) (2 * z) z :=
      (hId.fun_pow 2).congr_deriv (by norm_num)
    have hQuadratic : HasDerivAt (fun w : ℝ => w ^ 2 + source ^ 2) (2 * z) z :=
      hSquare.add_const (source ^ 2)
    have hLinear : HasDerivAt (fun w : ℝ => 2 * w * source) (2 * source) z := by
      simpa only [one_mul, mul_assoc] using (hId.mul_const source).const_mul 2
    have hNumerator := (hQuadratic.mul_const (Real.cosh (kappa * t))).sub hLinear
    have hScaled := (hNumerator.const_mul (mass * kappa)).div_const (2 * Real.sinh (kappa * t))
    unfold hyperbolicVanVleckAction
    exact hScaled.sub_const (hbar * Real.pi / 4) |>.congr_deriv (by
      field_simp [hSinh0])
  have hActionLaplacian : ∀ z, HasDerivAt
      (fun w => mass * kappa * (w * Real.cosh (kappa * t) - source) / Real.sinh (kappa * t))
      (mass * kappa * Real.cosh (kappa * t) / Real.sinh (kappa * t)) z := by
    intro z
    have hId : HasDerivAt (fun w : ℝ => w) 1 z := hasDerivAt_id (𝕜 := ℝ) z
    have hLinear : HasDerivAt (fun w : ℝ => w * Real.cosh (kappa * t) - source)
        (Real.cosh (kappa * t)) z :=
      hId.mul_const (Real.cosh (kappa * t)) |>.sub_const source |>.congr_deriv (by ring)
    exact (hLinear.const_mul (mass * kappa)).div_const (Real.sinh (kappa * t))
  apply (actual_schrodinger_iff_madelung hbar mass (hyperbolicVanVleckPotential mass kappa)
    (hyperbolicVanVleckDensity hbar mass kappa)
    (hyperbolicVanVleckAction hbar mass kappa source) t x
    (-Real.sqrt (hyperbolicVanVleckDensity hbar mass kappa t x) * kappa * Real.cosh (kappa * t) /
      (2 * Real.sinh (kappa * t)))
    (-mass * kappa ^ 2 * ((x ^ 2 + source ^ 2) - 2 * x * source * Real.cosh (kappa * t)) /
      (2 * Real.sinh (kappa * t) ^ 2))
    (fun _ => 0) (fun _ => 0)
    (fun z => mass * kappa * (z * Real.cosh (kappa * t) - source) / Real.sinh (kappa * t))
    (fun _ => mass * kappa * Real.cosh (kappa * t) / Real.sinh (kappa * t))
    hHbar hMass hDensityPos hAmplitudeTime hActionTime hAmplitudeSpace hZeroSpace
    hActionSpace hActionLaplacian).mpr
  constructor
  · unfold hamiltonJacobiResidual quantumPotential hyperbolicVanVleckPotential
    have hSquare := Real.sq_sqrt hDensityPos.le
    field_simp [hSinh0, hMass.ne']
    linear_combination mass ^ 2 * kappa ^ 2 * x ^ 2 *
      Real.sqrt (hyperbolicVanVleckDensity hbar mass kappa t x) *
      (Real.cosh_sq_sub_sinh_sq (kappa * t))
  · unfold continuityResidual
    have hSquare := Real.sq_sqrt hDensityPos.le
    field_simp [hSinh0, hMass.ne']
    ring

/-- A concrete positive-domain witness for the hyperbolic exactness theorem.

    hbar = mass = kappa = t = 1                 [the unit positive parameters]
    x = source = 0                                [the coincident endpoints]
    PointwiseSchrodingerExact                      [the witnessed PDE proposition]
    ℝ                                              [the real numbers]
    ∧                                              [both are true]
    <                                              [is smaller than]

    So: all strict-domain facts and the exactness proposition hold in one witness. -/
theorem hyperbolic_vanVleck_joint_witness :
    0 < (1 : ℝ) ∧
      PointwiseSchrodingerExact 1 1 (hyperbolicVanVleckPotential 1 1)
        (hyperbolicVanVleckDensity 1 1 1) (hyperbolicVanVleckAction 1 1 1 0) 1 0 := by
  refine ⟨by norm_num, ?_⟩
  exact hyperbolic_vanVleck_pointwise_exact 1 1 1 0 1 0
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Physics
end Apeiron
