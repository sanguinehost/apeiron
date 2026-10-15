/-
# The exact uncertainty relation: the quantum potential is Fisher information

`APEIRON-QARCH-L9`. For a one-dimensional density `ρ` with derivative `ρ'`, the Bohm quantum
potential of the amplitude `√ρ`, weighted by `ρ` and integrated, is `(ℏ²/8m) · I_F` with
`I_F = ∫ ρ'²/ρ` the Fisher information (`integral_rho_Q`). The nonclassical momentum
`N = (ℏ/2)(ρ'/ρ)` has `∫ ρ N² = (ℏ²/4) I_F` (`rho_ncMom_sq`, `quantum_energy_eq_nc_kinetic`), so
the product of the Fisher length `1/√I_F` and the nonclassical spread `√(∫ ρ N²)` is `ℏ/2`
(`exact_uncertainty`). The Cramér–Rao core `1 ≤ V·I` for `c = -1`, `c² ≤ V·I` is
`cramer_rao_core`; `cramer_rao_integral` proves `∫ (x-μ) ρ' = -1` and keeps the Cauchy–Schwarz
step `c² ≤ V·I` as a named hypothesis. `verdict_l9` conjoins the headlines.

Anchors (not proved here): Madelung (1927); Bohm (1952), the quantum potential; Fisher (1925);
Reginatto (1998) and Hall and Reginatto, *J. Phys. A* **35** (2002) 3289, the exact uncertainty
relation; Cramér (1946) and Rao (1945), the Cramér–Rao bound.
-/

import Mathlib
import Apeiron.Physics.Models.Schrodinger

open MeasureTheory Filter Topology

noncomputable section

namespace Quantum
namespace ExactUncertainty

/-- Fisher information of the location family of `ρ`: `I_F = ∫ ρ'² / ρ`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
-/
def fisher (ρ ρ' : ℝ → ℝ) : ℝ := ∫ x, ρ' x ^ 2 / ρ x

/-- Hall–Reginatto nonclassical momentum `N = (ℏ/2)(ρ'/ρ)`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
-/
def ncMom (hbar : ℝ) (ρ ρ' : ℝ → ℝ) (x : ℝ) : ℝ := hbar / 2 * (ρ' x / ρ x)

/-- Fisher length `δx = 1/√I_F`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
-/
def fisherLength (ρ ρ' : ℝ → ℝ) : ℝ := 1 / Real.sqrt (fisher ρ ρ')

/-- Nonclassical spread `ΔN = √(∫ ρ N²)`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∫` [added up continuously over]
`Real.sqrt` [the number that squares to give this]
-/
def ncSpread (hbar : ℝ) (ρ ρ' : ℝ → ℝ) : ℝ :=
  Real.sqrt (∫ x, ρ x * ncMom hbar ρ ρ' x ^ 2)

/-- The derivative of `√ρ` is `ρ'/(2√ρ)`; its derivative is `A''`, the closed form used below.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
`HasDerivAt` [is changing at that rate at that time]
-/
theorem sqrt_second_deriv (ρ ρ' ρ'' : ℝ → ℝ) (hpos : ∀ y, 0 < ρ y)
    (hd : ∀ y, HasDerivAt ρ (ρ' y) y) (hd2 : ∀ y, HasDerivAt ρ' (ρ'' y) y) (x : ℝ) :
    HasDerivAt (fun y => ρ' y / (2 * Real.sqrt (ρ y)))
      (ρ'' x / (2 * Real.sqrt (ρ x)) - ρ' x ^ 2 / (4 * ρ x * Real.sqrt (ρ x))) x := by
  have hs : 0 < Real.sqrt (ρ x) := Real.sqrt_pos.2 (hpos x)
  have hsq : Real.sqrt (ρ x) ^ 2 = ρ x := Real.sq_sqrt (hpos x).le
  have hden : HasDerivAt (fun y => 2 * Real.sqrt (ρ y)) (2 * (ρ' x / (2 * Real.sqrt (ρ x)))) x :=
    ((hd x).sqrt (hpos x).ne').const_mul 2
  have h := (hd2 x).div hden (mul_ne_zero two_ne_zero hs.ne')
  refine h.congr_deriv ?_
  generalize Real.sqrt (ρ x) = s at hs hsq ⊢
  have hs0 : s ≠ 0 := hs.ne'
  rw [← hsq]
  field_simp
  ring

/-- The Laplacian of `√ρ` is the closed form `A''` (Mathlib's one-dimensional Laplacian).
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
`HasDerivAt` [is changing at that rate at that time]
-/
theorem laplacian_sqrt (ρ ρ' ρ'' : ℝ → ℝ) (hpos : ∀ y, 0 < ρ y)
    (hd : ∀ y, HasDerivAt ρ (ρ' y) y) (hd2 : ∀ y, HasDerivAt ρ' (ρ'' y) y) (x : ℝ) :
    Laplacian.laplacian (fun y => Real.sqrt (ρ y)) x =
      ρ'' x / (2 * Real.sqrt (ρ x)) - ρ' x ^ 2 / (4 * ρ x * Real.sqrt (ρ x)) := by
  have hFirst : deriv (fun y => Real.sqrt (ρ y)) = fun y => ρ' y / (2 * Real.sqrt (ρ y)) :=
    funext fun y => ((hd y).sqrt (hpos y).ne').deriv
  have hSecond : deriv (deriv (fun y => Real.sqrt (ρ y))) x =
      ρ'' x / (2 * Real.sqrt (ρ x)) - ρ' x ^ 2 / (4 * ρ x * Real.sqrt (ρ x)) := by
    rw [hFirst]
    exact (sqrt_second_deriv ρ ρ' ρ'' hpos hd hd2 x).deriv
  rw [InnerProductSpace.laplacian_eq_iteratedDeriv_real]
  simpa [iteratedDeriv_succ, iteratedDeriv_zero] using hSecond

/-- Pointwise jet identity: `ρ · Q(√ρ, A'') = (ℏ²/8m) ρ'²/ρ - (ℏ²/4m) ρ''`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
-/
theorem rho_Q_pointwise (hbar m : ℝ) (ρ ρ' ρ'' : ℝ → ℝ) (x : ℝ) (hx : 0 < ρ x) :
    ρ x * Apeiron.Physics.quantumPotential hbar m (Real.sqrt (ρ x))
        (ρ'' x / (2 * Real.sqrt (ρ x)) - ρ' x ^ 2 / (4 * ρ x * Real.sqrt (ρ x)))
      = hbar ^ 2 / (8 * m) * (ρ' x ^ 2 / ρ x) - hbar ^ 2 / (4 * m) * ρ'' x := by
  have hs : 0 < Real.sqrt (ρ x) := Real.sqrt_pos.2 hx
  have hsq : Real.sqrt (ρ x) ^ 2 = ρ x := Real.sq_sqrt hx.le
  unfold Apeiron.Physics.quantumPotential
  generalize Real.sqrt (ρ x) = s at hs hsq ⊢
  have hs0 : s ≠ 0 := hs.ne'
  rw [← hsq]
  field_simp
  ring

/-- `ρ N² = (ℏ²/4) ρ'²/ρ` pointwise; at `ρ x = 0` both sides are `0` in Lean's total division.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
-/
theorem rho_ncMom_sq (hbar : ℝ) (ρ ρ' : ℝ → ℝ) (x : ℝ) :
    ρ x * ncMom hbar ρ ρ' x ^ 2 = hbar ^ 2 / 4 * (ρ' x ^ 2 / ρ x) := by
  unfold ncMom
  by_cases h : ρ x = 0
  · simp [h]
  · field_simp
    ring

/-- `∫ ρ'' = 0` when `ρ'` tends to `0` at both ends.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`∫` [added up continuously over]
`HasDerivAt` [is changing at that rate at that time]
-/
theorem integral_second_zero (ρ' ρ'' : ℝ → ℝ) (hd2 : ∀ y, HasDerivAt ρ' (ρ'' y) y)
    (hint2 : Integrable ρ'') (h0bot : Tendsto ρ' atBot (𝓝 0))
    (h0top : Tendsto ρ' atTop (𝓝 0)) : ∫ x, ρ'' x = 0 := by
  have h := integral_of_hasDerivAt_of_tendsto hd2 hint2 h0bot h0top
  simpa using h

/-- The integrated quantum potential energy weighted by `ρ` is `(ℏ²/8m) I_F`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`∫` [added up continuously over]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
`HasDerivAt` [is changing at that rate at that time]
-/
theorem integral_rho_Q (hbar m : ℝ) (ρ ρ' ρ'' : ℝ → ℝ) (hpos : ∀ y, 0 < ρ y)
    (hd : ∀ y, HasDerivAt ρ (ρ' y) y) (hd2 : ∀ y, HasDerivAt ρ' (ρ'' y) y)
    (hint2 : Integrable ρ'') (hintF : Integrable (fun y => ρ' y ^ 2 / ρ y))
    (h0bot : Tendsto ρ' atBot (𝓝 0)) (h0top : Tendsto ρ' atTop (𝓝 0)) :
    ∫ x, ρ x * Apeiron.Physics.quantumPotential hbar m (Real.sqrt (ρ x))
        (Laplacian.laplacian (fun y => Real.sqrt (ρ y)) x)
      = hbar ^ 2 / (8 * m) * fisher ρ ρ' := by
  have hfun : (fun x => ρ x * Apeiron.Physics.quantumPotential hbar m (Real.sqrt (ρ x))
      (Laplacian.laplacian (fun y => Real.sqrt (ρ y)) x)) =
      fun x => hbar ^ 2 / (8 * m) * (ρ' x ^ 2 / ρ x) - hbar ^ 2 / (4 * m) * ρ'' x := by
    funext x
    rw [laplacian_sqrt ρ ρ' ρ'' hpos hd hd2 x]
    exact rho_Q_pointwise hbar m ρ ρ' ρ'' x (hpos x)
  rw [hfun, integral_sub (hintF.const_mul _) (hint2.const_mul _), integral_const_mul,
    integral_const_mul, integral_second_zero ρ' ρ'' hd2 hint2 h0bot h0top, mul_zero, sub_zero]
  rfl

/-- The quantum potential energy is the kinetic energy of the nonclassical momentum:
`∫ ρ Q = (1/2m) ∫ ρ N²`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`∫` [added up continuously over]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
`HasDerivAt` [is changing at that rate at that time]
-/
theorem quantum_energy_eq_nc_kinetic (hbar m : ℝ) (ρ ρ' ρ'' : ℝ → ℝ) (hpos : ∀ y, 0 < ρ y)
    (hd : ∀ y, HasDerivAt ρ (ρ' y) y) (hd2 : ∀ y, HasDerivAt ρ' (ρ'' y) y)
    (hint2 : Integrable ρ'') (hintF : Integrable (fun y => ρ' y ^ 2 / ρ y))
    (h0bot : Tendsto ρ' atBot (𝓝 0)) (h0top : Tendsto ρ' atTop (𝓝 0)) :
    ∫ x, ρ x * Apeiron.Physics.quantumPotential hbar m (Real.sqrt (ρ x))
        (Laplacian.laplacian (fun y => Real.sqrt (ρ y)) x)
      = 1 / (2 * m) * ∫ x, ρ x * ncMom hbar ρ ρ' x ^ 2 := by
  have hN : ∫ x, ρ x * ncMom hbar ρ ρ' x ^ 2 = hbar ^ 2 / 4 * fisher ρ ρ' := by
    simp_rw [rho_ncMom_sq]
    rw [integral_const_mul]
    rfl
  rw [hN, integral_rho_Q hbar m ρ ρ' ρ'' hpos hd hd2 hint2 hintF h0bot h0top]
  ring

/-- The exact uncertainty relation `δx · ΔN = ℏ/2`, for `ℏ ≥ 0` and `I_F > 0`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`≤` [is smaller than or equal to]
`<` [is smaller than]
-/
theorem exact_uncertainty (ρ ρ' : ℝ → ℝ) (hbar : ℝ) (hF : 0 < fisher ρ ρ') (hh : 0 ≤ hbar) :
    fisherLength ρ ρ' * ncSpread hbar ρ ρ' = hbar / 2 := by
  have hN : ∫ x, ρ x * ncMom hbar ρ ρ' x ^ 2 = hbar ^ 2 / 4 * fisher ρ ρ' := by
    simp_rw [rho_ncMom_sq]
    rw [integral_const_mul]
    rfl
  have hsqrt : Real.sqrt (hbar ^ 2 / 4 * fisher ρ ρ') = hbar / 2 * Real.sqrt (fisher ρ ρ') := by
    rw [show hbar ^ 2 / 4 * fisher ρ ρ' = (hbar / 2) ^ 2 * fisher ρ ρ' by ring,
      Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by positivity)]
  have hpos : 0 < Real.sqrt (fisher ρ ρ') := Real.sqrt_pos.2 hF
  unfold fisherLength ncSpread
  rw [hN, hsqrt]
  field_simp

/-- Cramér–Rao core: `c² ≤ V·I` and `c = -1` give `1 ≤ V·I`.
`ℝ` [the real numbers: every number on the number line]
`≤` [is smaller than or equal to]
-/
theorem cramer_rao_core (V I c : ℝ) (hcs : c ^ 2 ≤ V * I) (hc : c = -1) : 1 ≤ V * I := by
  subst hc
  norm_num at hcs
  linarith

/-- Variance times nonclassical spread is at least `ℏ/2`.
`ℝ` [the real numbers: every number on the number line]
`≤` [is smaller than or equal to]
`Real.sqrt` [the number that squares to give this]
-/
theorem heisenberg_from_cr (V I hbar : ℝ) (hV : 0 ≤ V) (hh : 0 ≤ hbar)
    (h : 1 ≤ V * I) : hbar / 2 ≤ Real.sqrt V * (hbar / 2 * Real.sqrt I) := by
  have h1 : 1 ≤ Real.sqrt V * Real.sqrt I := by
    rw [← Real.sqrt_mul hV]
    exact Real.one_le_sqrt.mpr h
  have h2 : hbar / 2 * 1 ≤ hbar / 2 * (Real.sqrt V * Real.sqrt I) :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  calc hbar / 2 = hbar / 2 * 1 := by ring
    _ ≤ hbar / 2 * (Real.sqrt V * Real.sqrt I) := h2
    _ = Real.sqrt V * (hbar / 2 * Real.sqrt I) := by ring

/-- Integral Cramér–Rao: `∫ (x-μ) ρ' = -1` from `∫ ρ = 1` and the boundary limits; the
Cauchy–Schwarz step `c² ≤ V·I` is the named hypothesis `hcs`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`∧` [both are true]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`HasDerivAt` [is changing at that rate at that time]
-/
theorem cramer_rao_integral (μ : ℝ) (ρ ρ' : ℝ → ℝ)
    (hd : ∀ y, HasDerivAt ρ (ρ' y) y) (hint : Integrable ρ) (hnorm : ∫ x, ρ x = 1)
    (hintr : Integrable (fun x => (x - μ) * ρ' x))
    (htb : Tendsto (fun x => (x - μ) * ρ x) atBot (𝓝 0))
    (htt : Tendsto (fun x => (x - μ) * ρ x) atTop (𝓝 0))
    (hcs : (∫ x, (x - μ) * ρ' x) ^ 2 ≤ (∫ x, (x - μ) ^ 2 * ρ x) * fisher ρ ρ') :
    (∫ x, (x - μ) * ρ' x) = -1 ∧ 1 ≤ (∫ x, (x - μ) ^ 2 * ρ x) * fisher ρ ρ' := by
  have hder : ∀ x, HasDerivAt (fun x => (x - μ) * ρ x) (ρ x + (x - μ) * ρ' x) x := by
    intro x
    have := ((hasDerivAt_id x).sub_const μ).mul (hd x)
    exact this.congr_deriv (by simp)
  have h0 : ∫ x, (ρ x + (x - μ) * ρ' x) = 0 := by
    have h := integral_of_hasDerivAt_of_tendsto hder (hint.add hintr) htb htt
    simpa using h
  rw [integral_add hint hintr, hnorm] at h0
  have hc : ∫ x, (x - μ) * ρ' x = -1 := by linarith
  exact ⟨hc, cramer_rao_core _ _ _ (hc ▸ hcs) hc⟩

/-- Witness at `ρ = 4`, `ρ' = 2`, `ρ'' = 1`, `ℏ = m = 1`: both sides are `-1/8`.
`ℝ` [the real numbers: every number on the number line]
`Real.sqrt` [the number that squares to give this]
-/
theorem witness_pointwise :
    (4 : ℝ) * Apeiron.Physics.quantumPotential 1 1 (Real.sqrt 4)
        (1 / (2 * Real.sqrt 4) - 2 ^ 2 / (4 * 4 * Real.sqrt 4))
      = 1 ^ 2 / (8 * 1) * (2 ^ 2 / 4) - 1 ^ 2 / (4 * 1) * 1 := by
  have h2 : Real.sqrt 4 = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  rw [h2]
  unfold Apeiron.Physics.quantumPotential
  norm_num

/-- Witness for `rho_ncMom_sq` at the same numbers: both sides are `1/4`.
`ℝ` [the real numbers: every number on the number line]
-/
theorem witness_ncMom :
    (4 : ℝ) * ncMom 1 (fun _ => 4) (fun _ => 2) 0 ^ 2 = 1 ^ 2 / 4 * (2 ^ 2 / 4) := by
  norm_num [ncMom]

/-- The verdict: the pointwise jet identity, the Nonclassical identity, the exact uncertainty
relation, the Cramér–Rao core and Heisenberg form, and the two witnesses.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`∧` [both are true]
`≤` [is smaller than or equal to]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
-/
theorem verdict_l9 :
    (∀ (hbar m : ℝ) (ρ ρ' ρ'' : ℝ → ℝ) (x : ℝ), 0 < ρ x →
      ρ x * Apeiron.Physics.quantumPotential hbar m (Real.sqrt (ρ x))
          (ρ'' x / (2 * Real.sqrt (ρ x)) - ρ' x ^ 2 / (4 * ρ x * Real.sqrt (ρ x)))
        = hbar ^ 2 / (8 * m) * (ρ' x ^ 2 / ρ x) - hbar ^ 2 / (4 * m) * ρ'' x) ∧
    (∀ (hbar : ℝ) (ρ ρ' : ℝ → ℝ) (x : ℝ),
      ρ x * ncMom hbar ρ ρ' x ^ 2 = hbar ^ 2 / 4 * (ρ' x ^ 2 / ρ x)) ∧
    (∀ (ρ ρ' : ℝ → ℝ) (hbar : ℝ), 0 < fisher ρ ρ' → 0 ≤ hbar →
      fisherLength ρ ρ' * ncSpread hbar ρ ρ' = hbar / 2) ∧
    (∀ V I c : ℝ, c ^ 2 ≤ V * I → c = -1 → 1 ≤ V * I) ∧
    (∀ V I hbar : ℝ, 0 ≤ V → 0 ≤ hbar → 1 ≤ V * I →
      hbar / 2 ≤ Real.sqrt V * (hbar / 2 * Real.sqrt I)) ∧
    ((4 : ℝ) * Apeiron.Physics.quantumPotential 1 1 (Real.sqrt 4)
        (1 / (2 * Real.sqrt 4) - 2 ^ 2 / (4 * 4 * Real.sqrt 4))
      = 1 ^ 2 / (8 * 1) * (2 ^ 2 / 4) - 1 ^ 2 / (4 * 1) * 1) ∧
    ((4 : ℝ) * ncMom 1 (fun _ => 4) (fun _ => 2) 0 ^ 2 = 1 ^ 2 / 4 * (2 ^ 2 / 4)) :=
  ⟨fun hbar m ρ ρ' ρ'' x hx => rho_Q_pointwise hbar m ρ ρ' ρ'' x hx,
    fun hbar ρ ρ' x => rho_ncMom_sq hbar ρ ρ' x,
    fun ρ ρ' hbar hF hh => exact_uncertainty ρ ρ' hbar hF hh,
    fun V I c hcs hc => cramer_rao_core V I c hcs hc,
    fun V I hbar hV hh h => heisenberg_from_cr V I hbar hV hh h,
    witness_pointwise, witness_ncMom⟩

end ExactUncertainty
end Quantum

end
