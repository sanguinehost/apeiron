import Apeiron.Physics.Models.VanVleck.Basic

/-!
# Parabolic Van Vleck kernel

@cite schrodinger-1926
@cite van-vleck-1928
@cite morette-1951
@cite feynman-hibbs-1965
@cite vattay-2026-reply
@grade proved

This module checks the free-particle polar formula at a point in the regular domain `hbar > 0`,
`mass > 0`, and `t > 0`. Exactness means only that the analytic Schrödinger residual is zero.
It does not define an evolution operator, a propagator equality, kernel composition, unitarity,
or a path integral.

## Root

`Physics.Models.VanVleck.Basic.PointwiseSchrodingerExact` names the analytic residual equation.
`Physics.actual_schrodinger_iff_madelung` supplies the checked polar reduction, and
`Physics.det_generator` with `Physics.generator_trichotomy` supplies the classical parabolic
classification. The mixed derivative below independently identifies the density with the action.
-/

namespace Apeiron
namespace Physics

/-- The free-particle potential.

    (t x : ℝ)                                     [the time and spatial point]
    ℝ                                              [the real potential value]
    0                                              [the free-particle potential]

    So: the parabolic face has no spatial potential term. -/
def parabolicVanVleckPotential (_t _x : ℝ) : ℝ := 0

/-- The free-particle Van Vleck density.

    (hbar mass t x : ℝ)                            [the Planck scale, mass, and point]
    ℝ                                              [the real density value]
    mass / (2 * Real.pi * hbar * t)                [the free-particle density]
    Real.pi                                        [the ratio of circumference to diameter]

    So: the density is independent of position and singular at `t = 0`. -/
noncomputable def parabolicVanVleckDensity (hbar mass t _x : ℝ) : ℝ :=
  mass / (2 * Real.pi * hbar * t)

/-- The free-particle Van Vleck action from a fixed source point.

    (hbar mass source t x : ℝ)                     [the scales, source, and evaluation point]
    ℝ                                              [the real action value]
    mass * (x - source) ^ 2 / (2 * t) - hbar * Real.pi / 4
                                                   [the quadratic free action and phase offset]
    Real.pi                                        [the ratio of circumference to diameter]

    So: the phase offset has no derivative; the quadratic term carries the mixed
    source-position curvature. -/
noncomputable def parabolicVanVleckAction (hbar mass source t x : ℝ) : ℝ :=
  mass * (x - source) ^ 2 / (2 * t) - hbar * Real.pi / 4

/-- The free density is the negated mixed action derivative divided by `2*pi*hbar`.

    (hbar mass source t x : ℝ)                     [the scales, source, and evaluation point]
    0 < hbar ∧ 0 < mass ∧ 0 < t                   [the regular physical domain]
    parabolicVanVleckDensity hbar mass t x
      = -deriv (fun position => deriv (fun origin =>
          parabolicVanVleckAction hbar mass origin t position) source) x /
          (2 * Real.pi * hbar)                     [the mixed-action density identity]
    deriv                                          [the actual one-variable derivative]
    parabolicVanVleckDensity                       [the free-particle density]
    parabolicVanVleckAction                        [the free-particle action]

    So: both derivatives are evaluated explicitly; this is not a density definition in disguise. -/
theorem parabolic_vanVleck_density_eq_mixed_action
    (hbar mass source t x : ℝ) (hHbar : 0 < hbar) (hMass : 0 < mass) (hTime : 0 < t) :
    parabolicVanVleckDensity hbar mass t x =
      -deriv (fun position => deriv (fun origin =>
          parabolicVanVleckAction hbar mass origin t position) source) x /
          (2 * Real.pi * hbar) := by
  have hTime0 : t ≠ 0 := ne_of_gt hTime
  have hInner' : ∀ position : ℝ, HasDerivAt
      (fun origin => parabolicVanVleckAction hbar mass origin t position)
      (-mass * (position - source) / t) source := by
    intro position
    have hInnerDiff : HasDerivAt (fun origin : ℝ => position - origin) (-1) source := by
      exact (hasDerivAt_id (𝕜 := ℝ) source).const_sub position
    have hInner : HasDerivAt (fun origin : ℝ => mass * (position - origin) ^ 2 / (2 * t))
        (-mass * (position - source) / t) source := by
      have hSquare : HasDerivAt (fun origin : ℝ => (position - origin) ^ 2)
          (-2 * (position - source)) source := by
        exact (hInnerDiff.fun_pow 2).congr_deriv (by norm_num)
      exact ((hSquare.const_mul mass).div_const (2 * t)).congr_deriv (by
        field_simp [hTime0])
    unfold parabolicVanVleckAction
    exact hInner.sub_const (hbar * Real.pi / 4)
  have hOuter : HasDerivAt (fun position => -mass * (position - source) / t) (-mass / t) x := by
    have hDiff : HasDerivAt (fun position : ℝ => position - source) 1 x :=
      (hasDerivAt_id (𝕜 := ℝ) x).sub_const source
    simpa only [mul_one] using (hDiff.const_mul (-mass)).div_const t
  have hInnerDeriv :
      (fun position =>
        deriv (fun origin => parabolicVanVleckAction hbar mass origin t position) source) =
        fun position => -mass * (position - source) / t := by
    funext position
    exact (hInner' position).deriv
  rw [hInnerDeriv]
  rw [hOuter.deriv]
  unfold parabolicVanVleckDensity
  field_simp [hHbar.ne', hMass.ne', hTime0, Real.pi_ne_zero]

/-- The free generator is the parabolic member of the quadratic trichotomy.

    (rep 0 ((0 : ℝ), 1)).det = 0                  [the generator determinant is zero]
    ∧ ((rep 0 ((0 : ℝ), 1)).det = 0 ↔ (0 : ℝ) = 0)
                                                   [the trichotomy's zero branch]
    rep                                            [the grid representation]
    ∧                                              [both are true]
    ↔                                              [exactly when, true in both directions]

    So: the zero quadratic potential is tied to the existing parabolic generator classification. -/
theorem parabolic_vanVleck_face :
    (rep 0 ((0 : ℝ), 1)).det = 0 ∧ ((rep 0 ((0 : ℝ), 1)).det = 0 ↔ (0 : ℝ) = 0) := by
  have hTrichotomy := generator_trichotomy 0 1 (by norm_num : (1 : ℝ) ≠ 0)
  constructor
  · rw [det_generator]
    norm_num
  · exact hTrichotomy.2.1

/-- The free Van Vleck formula has zero pointwise Schrödinger residual.

    (hbar mass source t x : ℝ)                     [the scales, source, and evaluation point]
    0 < hbar ∧ 0 < mass ∧ 0 < t                   [the regular physical domain]
    PointwiseSchrodingerExact hbar mass parabolicVanVleckPotential
      (parabolicVanVleckDensity hbar mass) (parabolicVanVleckAction hbar mass source) t x
                                                   [the pointwise exactness proposition]
    PointwiseSchrodingerExact                      [the named pointwise exactness proposition]
    HasDerivAt                                     [is changing at that rate at that time]

    So: this discharges the actual componentwise derivative witnesses into the PDE residual only. -/
theorem parabolic_vanVleck_pointwise_exact
    (hbar mass source t x : ℝ) (hHbar : 0 < hbar) (hMass : 0 < mass) (hTime : 0 < t) :
    PointwiseSchrodingerExact hbar mass parabolicVanVleckPotential
      (parabolicVanVleckDensity hbar mass) (parabolicVanVleckAction hbar mass source) t x := by
  unfold PointwiseSchrodingerExact
  have hTime0 : t ≠ 0 := ne_of_gt hTime
  have hDensityPos : 0 < parabolicVanVleckDensity hbar mass t x := by
    unfold parabolicVanVleckDensity
    positivity
  have hDensityTime : HasDerivAt (fun s => parabolicVanVleckDensity hbar mass s x)
      (-parabolicVanVleckDensity hbar mass t x / t) t := by
    have hInv : HasDerivAt (fun s : ℝ => s⁻¹) (-1 / t ^ 2) t :=
      (hasDerivAt_inv hTime0).congr_deriv (by simp [div_eq_mul_inv])
    have hScaled := hInv.const_mul (mass / (2 * Real.pi * hbar))
    have hFunction : (fun s => parabolicVanVleckDensity hbar mass s x) =
        fun s => mass / (2 * Real.pi * hbar) * s⁻¹ := by
      funext s
      unfold parabolicVanVleckDensity
      field_simp [hHbar.ne', Real.pi_ne_zero]
    rw [hFunction]
    exact hScaled.congr_deriv (by
      unfold parabolicVanVleckDensity
      field_simp [hHbar.ne', hTime0, Real.pi_ne_zero])
  have hAmplitudeTime : HasDerivAt (fun s => Real.sqrt (parabolicVanVleckDensity hbar mass s x))
      (-Real.sqrt (parabolicVanVleckDensity hbar mass t x) / (2 * t)) t := by
    have h := hDensityTime.sqrt (ne_of_gt hDensityPos)
    have hSquare : Real.sqrt (parabolicVanVleckDensity hbar mass t x) ^ 2 =
        parabolicVanVleckDensity hbar mass t x := Real.sq_sqrt hDensityPos.le
    exact h.congr_deriv (by
      field_simp [hTime0, ne_of_gt hDensityPos]
      nlinarith)
  have hActionTime : HasDerivAt (fun s => parabolicVanVleckAction hbar mass source s x)
      (-mass * (x - source) ^ 2 / (2 * t ^ 2)) t := by
    have hInv : HasDerivAt (fun s : ℝ => s⁻¹) (-1 / t ^ 2) t :=
      (hasDerivAt_inv hTime0).congr_deriv (by simp [div_eq_mul_inv])
    have hScaled := hInv.const_mul (mass * (x - source) ^ 2 / 2)
    have hFunction : (fun s => parabolicVanVleckAction hbar mass source s x) =
        fun s => mass * (x - source) ^ 2 / 2 * s⁻¹ - hbar * Real.pi / 4 := by
      funext s
      unfold parabolicVanVleckAction
      field_simp
    rw [hFunction]
    exact (hScaled.sub_const (hbar * Real.pi / 4)).congr_deriv (by
      field_simp [hTime0])
  have hAmplitudeSpace : ∀ z, HasDerivAt
      (fun w => Real.sqrt (parabolicVanVleckDensity hbar mass t w)) 0 z := by
    intro z
    unfold parabolicVanVleckDensity
    simpa using (hasDerivAt_const (𝕜 := ℝ) z (c := Real.sqrt (mass / (2 * Real.pi * hbar * t))))
  have hZeroSpace : ∀ z, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 z :=
    fun z => hasDerivAt_const (𝕜 := ℝ) z (c := (0 : ℝ))
  have hActionSpace : ∀ z, HasDerivAt (fun w => parabolicVanVleckAction hbar mass source t w)
      (mass * (z - source) / t) z := by
    intro z
    have hDiff : HasDerivAt (fun w : ℝ => w - source) 1 z := by
      simpa using (hasDerivAt_id (𝕜 := ℝ) z).sub_const source
    have hSquare : HasDerivAt (fun w : ℝ => (w - source) ^ 2) (2 * (z - source)) z := by
      exact (hDiff.fun_pow 2).congr_deriv (by norm_num)
    have hScaled := (hSquare.const_mul mass).div_const (2 * t)
    unfold parabolicVanVleckAction
    exact (hScaled.sub_const (hbar * Real.pi / 4)).congr_deriv (by
      field_simp [hTime0])
  have hActionLaplacian : ∀ z, HasDerivAt (fun w => mass * (w - source) / t) (mass / t) z := by
    intro z
    have hDiff : HasDerivAt (fun w : ℝ => w - source) 1 z :=
      (hasDerivAt_id (𝕜 := ℝ) z).sub_const source
    simpa only [mul_one] using (hDiff.const_mul mass).div_const t
  apply (actual_schrodinger_iff_madelung hbar mass parabolicVanVleckPotential
    (parabolicVanVleckDensity hbar mass) (parabolicVanVleckAction hbar mass source) t x
    (-Real.sqrt (parabolicVanVleckDensity hbar mass t x) / (2 * t))
    (-mass * (x - source) ^ 2 / (2 * t ^ 2)) (fun _ => 0) (fun _ => 0)
    (fun z => mass * (z - source) / t) (fun _ => mass / t) hHbar hMass hDensityPos
    hAmplitudeTime hActionTime hAmplitudeSpace hZeroSpace hActionSpace hActionLaplacian).mpr
  constructor
  · unfold hamiltonJacobiResidual quantumPotential parabolicVanVleckPotential
    have hSquare := Real.sq_sqrt hDensityPos.le
    field_simp [hTime0, hMass.ne']
    ring
  · unfold continuityResidual
    have hSquare := Real.sq_sqrt hDensityPos.le
    field_simp [hTime0, hMass.ne']
    ring

/-- A concrete positive-domain witness for the parabolic exactness theorem.

    0 < (1 : ℝ)                                  [the unit scales and time are positive]
    parabolic_vanVleck_face                        [the parabolic generator classification]
    PointwiseSchrodingerExact 1 1 parabolicVanVleckPotential
      (parabolicVanVleckDensity 1 1) (parabolicVanVleckAction 1 0) 1 0
                                                   [the exactness witness]
    PointwiseSchrodingerExact                      [the named pointwise exactness proposition]
    ∧                                              [both are true]
    ↔                                              [exactly when, true in both directions]

    So: source and evaluation point are both zero, with every positive parameter fixed at one. -/
theorem parabolic_vanVleck_joint_witness :
    0 < (1 : ℝ) ∧
      ((rep 0 ((0 : ℝ), 1)).det = 0 ∧ ((rep 0 ((0 : ℝ), 1)).det = 0 ↔ (0 : ℝ) = 0)) ∧
      PointwiseSchrodingerExact 1 1 parabolicVanVleckPotential
        (parabolicVanVleckDensity 1 1) (parabolicVanVleckAction 1 1 0) 1 0 := by
  refine ⟨by norm_num, parabolic_vanVleck_face, ?_⟩
  exact parabolic_vanVleck_pointwise_exact 1 1 0 1 0 (by norm_num) (by norm_num) (by norm_num)

end Physics
end Apeiron
