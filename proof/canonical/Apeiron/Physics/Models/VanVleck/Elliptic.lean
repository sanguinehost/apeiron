import Apeiron.Physics.Models.VanVleck.Basic

/-!
# Elliptic Van Vleck kernel

@cite schrodinger-1926
@cite van-vleck-1928
@cite morette-1951
@cite feynman-hibbs-1965
@cite vattay-2026-reply
@grade proved

This module checks the harmonic-oscillator polar formula before its first caustic: `hbar`,
`mass`, and `omega` are positive and `0 < omega * t < Real.pi`. Exactness means only a zero
pointwise analytic Schrödinger residual. It does not define a propagator, an evolution operator,
kernel composition, unitarity, a path integral, or caustic continuation.

## Root

`Physics.Models.VanVleck.Basic.PointwiseSchrodingerExact` names the residual equation.
`Physics.actual_schrodinger_iff_madelung` provides the polar reduction, while
`Physics.det_generator` and `Physics.generator_trichotomy` provide the elliptic classification.
The mixed derivative identifies the density independently of its definition.
-/

namespace Apeiron
namespace Physics

/-- The harmonic-oscillator potential.

    (mass omega t x : ℝ)                          [the scales and point]
    ℝ                                              [the potential value]
    mass * omega ^ 2 * x ^ 2 / 2                  [the harmonic potential]

    So: positive mass and frequency give the positive quadratic face. -/
noncomputable def ellipticVanVleckPotential (mass omega _t x : ℝ) : ℝ :=
  mass * omega ^ 2 * x ^ 2 / 2

/-- The elliptic Van Vleck density before the first caustic.

    (hbar mass omega t x : ℝ)                     [the scales and point]
    ℝ                                              [the density value]
    mass * omega / (2 * Real.pi * hbar * Real.sin (omega * t))
                                                   [the harmonic density]
    Real.sin                                      [the trigonometric sine]

    So: the density is position-independent and is regular only away from a sine zero. -/
noncomputable def ellipticVanVleckDensity (hbar mass omega t _x : ℝ) : ℝ :=
  mass * omega / (2 * Real.pi * hbar * Real.sin (omega * t))

/-- The harmonic Van Vleck action from a fixed source.

    (hbar mass omega source t x : ℝ)              [the scales, source, and point]
    ℝ                                              [the action value]
    mass * omega * ((x ^ 2 + source ^ 2) * Real.cos (omega * t) - 2 * x * source)
      / (2 * Real.sin (omega * t)) - hbar * Real.pi / 4
                                                   [the harmonic action and phase offset]
    Real.cos                                      [the trigonometric cosine]

    So: the phase offset has zero derivatives and the remaining curvature is harmonic. -/
noncomputable def ellipticVanVleckAction (hbar mass omega source t x : ℝ) : ℝ :=
  mass * omega * ((x ^ 2 + source ^ 2) * Real.cos (omega * t) - 2 * x * source) /
      (2 * Real.sin (omega * t)) - hbar * Real.pi / 4

/-- The elliptic density is the negated mixed action derivative divided by `2*pi*hbar`.

    (hbar mass omega source t x : ℝ)              [the scales, source, and point]
    0 < hbar ∧ 0 < mass ∧ 0 < omega ∧ 0 < omega * t ∧ omega * t < Real.pi
                                                   [the first-caustic domain]
    ellipticVanVleckDensity hbar mass omega t x
      = -deriv (fun position => deriv (fun origin =>
          ellipticVanVleckAction hbar mass omega origin t position) source) x /
          (2 * Real.pi * hbar)                    [the independently checked identity]
    deriv                                          [the actual derivative]
    ∧                                              [both are true]
    <                                              [is smaller than]

    So: the density follows from the real mixed source-position curvature. -/
theorem elliptic_vanVleck_density_eq_mixed_action
    (hbar mass omega source t x : ℝ) (hHbar : 0 < hbar) (hMass : 0 < mass) (hOmega : 0 < omega)
    (hPhase : 0 < omega * t ∧ omega * t < Real.pi) :
    ellipticVanVleckDensity hbar mass omega t x =
      -deriv (fun position => deriv (fun origin =>
        ellipticVanVleckAction hbar mass omega origin t position) source) x /
          (2 * Real.pi * hbar) := by
  have hSinPos : 0 < Real.sin (omega * t) := Real.sin_pos_of_pos_of_lt_pi hPhase.1 hPhase.2
  have hSin0 : Real.sin (omega * t) ≠ 0 := ne_of_gt hSinPos
  have hInner : ∀ position : ℝ, HasDerivAt
      (fun origin => ellipticVanVleckAction hbar mass omega origin t position)
      (mass * omega * (source * Real.cos (omega * t) - position) /
        Real.sin (omega * t)) source := by
    intro position
    have hId : HasDerivAt (fun origin : ℝ => origin) 1 source := hasDerivAt_id (𝕜 := ℝ) source
    have hSquare : HasDerivAt (fun origin : ℝ => origin ^ 2) (2 * source) source :=
      (hId.fun_pow 2).congr_deriv (by norm_num)
    have hQuadratic : HasDerivAt (fun origin : ℝ => position ^ 2 + origin ^ 2)
        (2 * source) source := hSquare.const_add (position ^ 2)
    have hLinear : HasDerivAt (fun origin : ℝ => 2 * position * origin) (2 * position) source := by
      simpa only [mul_one] using hId.const_mul (2 * position)
    have hNumerator : HasDerivAt
        (fun origin => (position ^ 2 + origin ^ 2) * Real.cos (omega * t) - 2 * position * origin)
        (2 * source * Real.cos (omega * t) - 2 * position) source :=
      (hQuadratic.mul_const (Real.cos (omega * t))).sub hLinear
    have hScaled := (hNumerator.const_mul (mass * omega)).div_const (2 * Real.sin (omega * t))
    unfold ellipticVanVleckAction
    exact hScaled.sub_const (hbar * Real.pi / 4) |>.congr_deriv (by
      field_simp [hSin0])
  have hOuter : HasDerivAt
      (fun position => mass * omega * (source * Real.cos (omega * t) - position) /
        Real.sin (omega * t))
      (-mass * omega / Real.sin (omega * t)) x := by
    have hId : HasDerivAt (fun position : ℝ => position) 1 x := hasDerivAt_id (𝕜 := ℝ) x
    exact ((hasDerivAt_const (𝕜 := ℝ) x (c := source * Real.cos (omega * t))).sub hId |>.const_mul
      (mass * omega) |>.div_const (Real.sin (omega * t))).congr_deriv (by ring)
  have hInnerDeriv :
      (fun position =>
        deriv (fun origin => ellipticVanVleckAction hbar mass omega origin t position) source) =
        fun position => mass * omega * (source * Real.cos (omega * t) - position) /
          Real.sin (omega * t) := by
    funext position
    exact (hInner position).deriv
  rw [hInnerDeriv, hOuter.deriv]
  unfold ellipticVanVleckDensity
  field_simp [hHbar.ne', hMass.ne', hOmega.ne', hSin0, Real.pi_ne_zero]

/-- The harmonic generator is elliptic at `sigma = -(omega^2)`.

    0 < (rep (-(omega ^ 2)) ((0 : ℝ), 1)).det    [the positive determinant]
    ∧ -(omega ^ 2) < 0                            [the elliptic trichotomy branch]
    rep                                            [the grid representation]
    ∧                                              [both are true]

    So: the quantum carrier remains `sigma = -1`; this classifies its classical generator. -/
theorem elliptic_vanVleck_face (omega : ℝ) (hOmega : 0 < omega) :
    0 < (rep (-(omega ^ 2)) ((0 : ℝ), 1)).det ∧ -(omega ^ 2) < 0 := by
  have hTri := generator_trichotomy (-(omega ^ 2)) 1 (by norm_num : (1 : ℝ) ≠ 0)
  constructor
  · rw [det_generator]
    nlinarith [sq_pos_of_pos hOmega]
  · exact hTri.1.mp (by rw [det_generator]; nlinarith [sq_pos_of_pos hOmega])

/-- The harmonic Van Vleck formula has zero pointwise Schrödinger residual before the first caustic.

    (hbar mass omega source t x : ℝ)              [the scales, source, and point]
    0 < hbar ∧ 0 < mass ∧ 0 < omega ∧ 0 < omega * t ∧ omega * t < Real.pi
                                                   [the regular domain]
    PointwiseSchrodingerExact hbar mass (ellipticVanVleckPotential mass omega)
      (ellipticVanVleckDensity hbar mass omega) (ellipticVanVleckAction hbar mass omega source) t x
                                                   [the exactness proposition]
    PointwiseSchrodingerExact                      [the named residual predicate]
    HasDerivAt                                     [the actual derivative witness]
    ∧                                              [both are true]
    <                                              [is smaller than]

    So: this proves a pointwise PDE residual only, not propagator or path-integral exactness. -/
theorem elliptic_vanVleck_pointwise_exact
    (hbar mass omega source t x : ℝ) (hHbar : 0 < hbar) (hMass : 0 < mass) (hOmega : 0 < omega)
    (hPhase : 0 < omega * t ∧ omega * t < Real.pi) :
    PointwiseSchrodingerExact hbar mass (ellipticVanVleckPotential mass omega)
      (ellipticVanVleckDensity hbar mass omega)
      (ellipticVanVleckAction hbar mass omega source) t x := by
  have hSinPos : 0 < Real.sin (omega * t) := Real.sin_pos_of_pos_of_lt_pi hPhase.1 hPhase.2
  have hSin0 : Real.sin (omega * t) ≠ 0 := ne_of_gt hSinPos
  have hDensityPos : 0 < ellipticVanVleckDensity hbar mass omega t x := by
    unfold ellipticVanVleckDensity
    exact div_pos (mul_pos hMass hOmega)
      (mul_pos (mul_pos (mul_pos (by norm_num) Real.pi_pos) hHbar) hSinPos)
  have hArg : HasDerivAt (fun s : ℝ => omega * s) omega t := by
    have hId : HasDerivAt (fun s : ℝ => s) 1 t := hasDerivAt_id (𝕜 := ℝ) t
    simpa only [mul_one] using hId.const_mul omega
  have hSin : HasDerivAt (fun s : ℝ => Real.sin (omega * s))
      (omega * Real.cos (omega * t)) t := by
    exact (Real.hasDerivAt_sin (omega * t)).comp t hArg |>.congr_deriv (by ring)
  have hCos : HasDerivAt (fun s : ℝ => Real.cos (omega * s))
      (-omega * Real.sin (omega * t)) t := by
    exact (Real.hasDerivAt_cos (omega * t)).comp t hArg |>.congr_deriv (by ring)
  have hInv : HasDerivAt (fun s : ℝ => (Real.sin (omega * s))⁻¹)
      (-(omega * Real.cos (omega * t)) / Real.sin (omega * t) ^ 2) t :=
    hSin.fun_inv hSin0
  have hDensityTime : HasDerivAt (fun s => ellipticVanVleckDensity hbar mass omega s x)
      (-ellipticVanVleckDensity hbar mass omega t x * omega * Real.cos (omega * t) /
        Real.sin (omega * t)) t := by
    have hScaled := hInv.const_mul (mass * omega / (2 * Real.pi * hbar))
    have hFunction : (fun s => ellipticVanVleckDensity hbar mass omega s x) =
        fun s => mass * omega / (2 * Real.pi * hbar) * (Real.sin (omega * s))⁻¹ := by
      funext s
      unfold ellipticVanVleckDensity
      field_simp [hHbar.ne', Real.pi_ne_zero]
    rw [hFunction]
    exact hScaled.congr_deriv (by
      unfold ellipticVanVleckDensity
      field_simp [hHbar.ne', hSin0, Real.pi_ne_zero])
  have hAmplitudeTime : HasDerivAt
      (fun s => Real.sqrt (ellipticVanVleckDensity hbar mass omega s x))
      (-Real.sqrt (ellipticVanVleckDensity hbar mass omega t x) * omega * Real.cos (omega * t) /
        (2 * Real.sin (omega * t))) t := by
    have h := hDensityTime.sqrt (ne_of_gt hDensityPos)
    have hSquare : Real.sqrt (ellipticVanVleckDensity hbar mass omega t x) ^ 2 =
        ellipticVanVleckDensity hbar mass omega t x := Real.sq_sqrt hDensityPos.le
    exact h.congr_deriv (by
      field_simp [hSin0, ne_of_gt hDensityPos]
      rw [hSquare]
      ring)
  have hActionTime : HasDerivAt (fun s => ellipticVanVleckAction hbar mass omega source s x)
      (-mass * omega ^ 2 * ((x ^ 2 + source ^ 2) - 2 * x * source * Real.cos (omega * t)) /
        (2 * Real.sin (omega * t) ^ 2)) t := by
    have hNumerator : HasDerivAt
        (fun s => (x ^ 2 + source ^ 2) * Real.cos (omega * s) - 2 * x * source)
        (-omega * (x ^ 2 + source ^ 2) * Real.sin (omega * t)) t :=
      (hCos.const_mul (x ^ 2 + source ^ 2)).sub_const (2 * x * source) |>.congr_deriv (by ring)
    have hTrig :
        -((x ^ 2 + source ^ 2) * Real.sin (omega * t) ^ 2) -
            Real.cos (omega * t) *
              ((x ^ 2 + source ^ 2) * Real.cos (omega * t) - 2 * x * source) =
          -((x ^ 2 + source ^ 2) - 2 * x * source * Real.cos (omega * t)) := by
      calc
        _ = -(x ^ 2 + source ^ 2) *
              (Real.sin (omega * t) ^ 2 + Real.cos (omega * t) ^ 2) +
                2 * x * source * Real.cos (omega * t) := by ring
        _ = -(x ^ 2 + source ^ 2) +
                2 * x * source * Real.cos (omega * t) := by
              rw [Real.sin_sq_add_cos_sq]
              ring
        _ = _ := by ring
    have hQuotient : HasDerivAt
        (fun s => mass * omega / 2 *
          (((x ^ 2 + source ^ 2) * Real.cos (omega * s) - 2 * x * source) *
            (Real.sin (omega * s))⁻¹))
        (-mass * omega ^ 2 *
          ((x ^ 2 + source ^ 2) - 2 * x * source * Real.cos (omega * t)) /
            (2 * Real.sin (omega * t) ^ 2)) t := by
      have hScaled := (hNumerator.fun_mul hInv).const_mul (mass * omega / 2)
      exact hScaled.congr_deriv (by
        field_simp [hSin0]
        simpa only [sub_eq_add_neg] using hTrig)
    have hFunction : (fun s => ellipticVanVleckAction hbar mass omega source s x) =
        fun s => mass * omega / 2 *
          (((x ^ 2 + source ^ 2) * Real.cos (omega * s) - 2 * x * source) *
            (Real.sin (omega * s))⁻¹) - hbar * Real.pi / 4 := by
      funext s
      unfold ellipticVanVleckAction
      field_simp
    rw [hFunction]
    exact hQuotient.sub_const (hbar * Real.pi / 4)
  have hAmplitudeSpace : ∀ z, HasDerivAt
      (fun w => Real.sqrt (ellipticVanVleckDensity hbar mass omega t w)) 0 z := by
    intro z
    unfold ellipticVanVleckDensity
    simpa using hasDerivAt_const (𝕜 := ℝ) z
      (c := Real.sqrt (mass * omega / (2 * Real.pi * hbar * Real.sin (omega * t))))
  have hZeroSpace : ∀ z, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 z :=
    fun z => hasDerivAt_const (𝕜 := ℝ) z (c := (0 : ℝ))
  have hActionSpace : ∀ z, HasDerivAt (fun w => ellipticVanVleckAction hbar mass omega source t w)
      (mass * omega * (z * Real.cos (omega * t) - source) / Real.sin (omega * t)) z := by
    intro z
    have hId : HasDerivAt (fun w : ℝ => w) 1 z := hasDerivAt_id (𝕜 := ℝ) z
    have hSquare : HasDerivAt (fun w : ℝ => w ^ 2) (2 * z) z :=
      (hId.fun_pow 2).congr_deriv (by norm_num)
    have hQuadratic : HasDerivAt (fun w : ℝ => w ^ 2 + source ^ 2) (2 * z) z :=
      hSquare.add_const (source ^ 2)
    have hLinear : HasDerivAt (fun w : ℝ => 2 * w * source) (2 * source) z := by
      simpa only [one_mul, mul_assoc] using (hId.mul_const source).const_mul 2
    have hNumerator := (hQuadratic.mul_const (Real.cos (omega * t))).sub hLinear
    have hScaled := (hNumerator.const_mul (mass * omega)).div_const (2 * Real.sin (omega * t))
    unfold ellipticVanVleckAction
    exact hScaled.sub_const (hbar * Real.pi / 4) |>.congr_deriv (by
      field_simp [hSin0])
  have hActionLaplacian : ∀ z, HasDerivAt
      (fun w => mass * omega * (w * Real.cos (omega * t) - source) / Real.sin (omega * t))
      (mass * omega * Real.cos (omega * t) / Real.sin (omega * t)) z := by
    intro z
    have hId : HasDerivAt (fun w : ℝ => w) 1 z := hasDerivAt_id (𝕜 := ℝ) z
    have hLinear : HasDerivAt (fun w : ℝ => w * Real.cos (omega * t) - source)
        (Real.cos (omega * t)) z :=
      hId.mul_const (Real.cos (omega * t)) |>.sub_const source |>.congr_deriv (by ring)
    exact (hLinear.const_mul (mass * omega)).div_const (Real.sin (omega * t))
  apply (actual_schrodinger_iff_madelung hbar mass (ellipticVanVleckPotential mass omega)
    (ellipticVanVleckDensity hbar mass omega) (ellipticVanVleckAction hbar mass omega source) t x
    (-Real.sqrt (ellipticVanVleckDensity hbar mass omega t x) * omega * Real.cos (omega * t) /
      (2 * Real.sin (omega * t)))
    (-mass * omega ^ 2 * ((x ^ 2 + source ^ 2) - 2 * x * source * Real.cos (omega * t)) /
      (2 * Real.sin (omega * t) ^ 2))
    (fun _ => 0) (fun _ => 0)
    (fun z => mass * omega * (z * Real.cos (omega * t) - source) / Real.sin (omega * t))
    (fun _ => mass * omega * Real.cos (omega * t) / Real.sin (omega * t))
    hHbar hMass hDensityPos hAmplitudeTime hActionTime hAmplitudeSpace hZeroSpace
    hActionSpace hActionLaplacian).mpr
  constructor
  · unfold hamiltonJacobiResidual quantumPotential ellipticVanVleckPotential
    have hSquare := Real.sq_sqrt hDensityPos.le
    field_simp [hSin0, hMass.ne']
    linear_combination mass ^ 2 * omega ^ 2 * x ^ 2 *
      Real.sqrt (ellipticVanVleckDensity hbar mass omega t x) *
      (Real.cos_sq_add_sin_sq (omega * t))
  · unfold continuityResidual
    have hSquare := Real.sq_sqrt hDensityPos.le
    field_simp [hSin0, hMass.ne']
    ring

/-- A concrete positive-domain witness for the elliptic exactness theorem.

    hbar = mass = omega = 1                      [the unit scales]
    t = Real.pi / 2                               [the first positive sine maximum]
    x = source = 0                                [the coincident endpoints]
    PointwiseSchrodingerExact                      [the witnessed PDE proposition]
    ℝ                                              [the real numbers]
    ∧                                              [both are true]
    <                                              [is smaller than]

    So: every strict domain premise and the residual proposition hold together. -/
theorem elliptic_vanVleck_joint_witness :
    0 < (1 : ℝ) ∧ 0 < Real.pi / 2 ∧ Real.pi / 2 < Real.pi ∧
      PointwiseSchrodingerExact 1 1 (ellipticVanVleckPotential 1 1)
        (ellipticVanVleckDensity 1 1 1) (ellipticVanVleckAction 1 1 1 0) (Real.pi / 2) 0 := by
  refine ⟨by norm_num, by simpa using half_pos Real.pi_pos, by linarith [Real.pi_pos], ?_⟩
  exact elliptic_vanVleck_pointwise_exact 1 1 1 0 (Real.pi / 2) 0
    (by norm_num) (by norm_num) (by norm_num)
    ⟨by nlinarith [Real.pi_pos], by linarith [Real.pi_pos]⟩

end Physics
end Apeiron
