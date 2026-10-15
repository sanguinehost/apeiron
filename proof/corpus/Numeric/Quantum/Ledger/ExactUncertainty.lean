/-
# The exact uncertainty relation: the quantum potential is Fisher information

. For a one-dimensional density `ρ` with derivative `ρ'`, the Bohm quantum
potential of the amplitude `√ρ`, weighted by `ρ` and integrated, is `(ℏ²/8m) · I_F` with
`I_F = ∫ ρ'²/ρ` the Fisher information (`integral_rho_Q`). The nonclassical momentum
`N = (ℏ/2)(ρ'/ρ)` has `∫ ρ N² = (ℏ²/4) I_F` (`rho_ncMom_sq`, `quantum_energy_eq_nc_kinetic`), so
the product of the Fisher length `1/√I_F` and the nonclassical spread `√(∫ ρ N²)` is `ℏ/2`
(`exact_uncertainty`). The Cramér–Rao core `1 ≤ V·I` for `c = -1`, `c² ≤ V·I` is
`cramer_rao_core`; `cramer_rao_integral` proves `∫ (x-μ) ρ' = -1` from a Cauchy–Schwarz premise,
which `cauchy_schwarz_fisher` discharges by the discriminant of `∫ ρ (t(x-μ) + ρ'/ρ)² ≥ 0`, so
`cramer_rao` states `1 ≤ V·I_F` with no such premise. `verdict_l9` and `verdict_l9b` conjoin the
headlines.

Anchors (not proved here; full references in `book/src/reference/bibliography.md`): Madelung
(1927); Bohm (1952), the quantum potential; Fisher (1922, 1925), Fisher information; Reginatto
(1998), `∫ρQ = (ℏ²/8m) I_F`; Hall and Reginatto, *J. Phys. A* **35** (2002) 3289, eq. 19
(`δx ΔN = ℏ/2`) and the Cramér–Rao route to Heisenberg. The Cramér–Rao bound was found
independently by Aitken and Silverstone (1942), Fréchet (1943), Rao (1945), Darmois (1945, several
parameters) and Cramér (1946). Cauchy–Schwarz: Cauchy (1821, sums), Bunyakovsky (1859, integrals),
Schwarz (1888, inner products). The position–momentum relation: Heisenberg (1927, heuristic),
Kennard (1927, `σ_x σ_p ≥ ℏ/2`), Weyl (1928), Robertson (1929, any two observables).

`heisenberg_from_cr` bounds the position spread times the nonclassical spread `ΔN`; for the Madelung
wavefunction `ψ = √ρ·e^{iS/ℏ}` (`madelung`, `madelungDeriv`, whose derivative is
`madelung_hasDerivAt`) `momentum_spread_sq` proves Hall and Reginatto's step `Δp² = Var(S') + ΔN²`,
`nc_le_momSpread` lifts it to `ΔN ≤ Δp` (`madelung_normSq`, `madelung_current`,
`weighted_var_nonneg`, `momMean`, `momSq`, `momSpread`), and `heisenberg` gives the full Kennard
bound `Δx·Δp ≥ ℏ/2` for every centre `μ` (`cramer_rao`, `ncSpread_eq`).
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

/-- The discriminant form: for every `t`, `0 ≤ t²·∫(x-μ)²ρ + 2t·∫(x-μ)ρ' + I_F`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`<` [is smaller than]
-/
theorem fisher_quad_nonneg (μ : ℝ) (ρ ρ' : ℝ → ℝ) (hpos : ∀ x, 0 < ρ x)
    (hA : Integrable (fun x => (x - μ) ^ 2 * ρ x)) (hB : Integrable (fun x => (x - μ) * ρ' x))
    (hC : Integrable (fun x => ρ' x ^ 2 / ρ x)) (t : ℝ) :
    0 ≤ t ^ 2 * (∫ x, (x - μ) ^ 2 * ρ x) + 2 * t * (∫ x, (x - μ) * ρ' x) + fisher ρ ρ' := by
  have hpt : ∀ x, ρ x * (t * (x - μ) + ρ' x / ρ x) ^ 2
      = t ^ 2 * ((x - μ) ^ 2 * ρ x) + 2 * t * ((x - μ) * ρ' x) + ρ' x ^ 2 / ρ x := by
    intro x
    have hx : ρ x ≠ 0 := (hpos x).ne'
    field_simp
    ring
  have h1 : Integrable (fun x => t ^ 2 * ((x - μ) ^ 2 * ρ x)) := hA.const_mul _
  have h2 : Integrable (fun x => 2 * t * ((x - μ) * ρ' x)) := hB.const_mul _
  have hnn : 0 ≤ ∫ x, ρ x * (t * (x - μ) + ρ' x / ρ x) ^ 2 :=
    integral_nonneg fun x => mul_nonneg (hpos x).le (sq_nonneg _)
  have h12 : Integrable (fun x => t ^ 2 * ((x - μ) ^ 2 * ρ x) + 2 * t * ((x - μ) * ρ' x)) :=
    h1.add h2
  simp_rw [hpt] at hnn
  rw [integral_add h12 hC, integral_add h1 h2, integral_const_mul,
    integral_const_mul] at hnn
  unfold fisher
  linarith

/-- Cauchy–Schwarz for the Fisher information: `(∫(x-μ)ρ')² ≤ (∫(x-μ)²ρ)·I_F`, from the
nonnegativity of the discriminant form `fisher_quad_nonneg`, with no `L²` space.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`<` [is smaller than]
`∀` [for every single one]
-/
theorem cauchy_schwarz_fisher (μ : ℝ) (ρ ρ' : ℝ → ℝ) (hpos : ∀ x, 0 < ρ x)
    (hA : Integrable (fun x => (x - μ) ^ 2 * ρ x)) (hB : Integrable (fun x => (x - μ) * ρ' x))
    (hC : Integrable (fun x => ρ' x ^ 2 / ρ x)) :
    (∫ x, (x - μ) * ρ' x) ^ 2 ≤ (∫ x, (x - μ) ^ 2 * ρ x) * fisher ρ ρ' := by
  have hq := fisher_quad_nonneg μ ρ ρ' hpos hA hB hC
  have hdisc : discrim (∫ x, (x - μ) ^ 2 * ρ x) (2 * ∫ x, (x - μ) * ρ' x) (fisher ρ ρ') ≤ 0 := by
    apply discrim_le_zero
    intro t
    nlinarith [hq t]
  unfold discrim at hdisc
  nlinarith [hdisc]

/-- Integral Cramér–Rao with the Cauchy–Schwarz step discharged: `1 ≤ V·I_F`, with no
`hcs` hypothesis.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`HasDerivAt` [is changing at that rate at that time]
`<` [is smaller than]
`∀` [for every single one]
-/
theorem cramer_rao (μ : ℝ) (ρ ρ' : ℝ → ℝ) (hpos : ∀ x, 0 < ρ x)
    (hd : ∀ y, HasDerivAt ρ (ρ' y) y) (hint : Integrable ρ) (hnorm : ∫ x, ρ x = 1)
    (hA : Integrable (fun x => (x - μ) ^ 2 * ρ x))
    (hintr : Integrable (fun x => (x - μ) * ρ' x))
    (hC : Integrable (fun x => ρ' x ^ 2 / ρ x))
    (htb : Tendsto (fun x => (x - μ) * ρ x) atBot (𝓝 0))
    (htt : Tendsto (fun x => (x - μ) * ρ x) atTop (𝓝 0)) :
    1 ≤ (∫ x, (x - μ) ^ 2 * ρ x) * fisher ρ ρ' :=
  (cramer_rao_integral μ ρ ρ' hd hint hnorm hintr htb htt
    (cauchy_schwarz_fisher μ ρ ρ' hpos hA hintr hC)).2

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

/-- `exp(i·t)` has unit modulus: its conjugate times it is `1`.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`ℝ` [the real numbers: every number on the number line]
`Complex.normSq` [the squared size a² + b², the square of the modulus a + b·i]
`Complex.exp` [the number e raised to this power]
`Complex.I` [the square root of −1]
`starRingEnd ℂ` [the conjugate a + b·i ↦ a − b·i]
-/
theorem star_exp_I (t : ℝ) :
    starRingEnd ℂ (Complex.exp (Complex.I * (t : ℂ))) * Complex.exp (Complex.I * (t : ℂ)) = 1 := by
  rw [mul_comm, Complex.mul_conj, Complex.normSq_eq_norm_sq, Complex.norm_exp]
  simp

/-- The Madelung wavefunction `ψ = √ρ · exp(i·S/ℏ)` of the density `ρ` and the phase `S`.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`Real.sqrt` [the number that squares to give this]
`Complex.exp` [the number e raised to this power]
`Complex.I` [the square root of −1]
-/
def madelung (hbar : ℝ) (ρ S : ℝ → ℝ) (x : ℝ) : ℂ :=
  (Real.sqrt (ρ x) : ℂ) * Complex.exp (Complex.I * ((S x / hbar : ℝ) : ℂ))

/-- The derivative `ψ' = (ρ'/(2√ρ) + i·√ρ·S'/ℏ) · exp(i·S/ℏ)` of the Madelung wavefunction, with
`ρ'` the derivative of `ρ` and `S'` that of `S`.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`Real.sqrt` [the number that squares to give this]
`Complex.exp` [the number e raised to this power]
`Complex.I` [the square root of −1]
-/
def madelungDeriv (hbar : ℝ) (ρ ρ' S S' : ℝ → ℝ) (x : ℝ) : ℂ :=
  (((ρ' x / (2 * Real.sqrt (ρ x)) : ℝ) : ℂ) +
      Complex.I * ((Real.sqrt (ρ x) * S' x / hbar : ℝ) : ℂ)) *
    Complex.exp (Complex.I * ((S x / hbar : ℝ) : ℂ))

/-- Momentum mean `⟨p⟩ = ∫ Re(conj ψ · (−iℏ ψ'))`, the expectation of `p̂ = −iℏ d/dx`.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`starRingEnd ℂ` [the conjugate a + b·i ↦ a − b·i]
`Complex.re` [the real part a of a + b·i]
`Complex.I` [the square root of −1]
`ℝ` [the real numbers: every number on the number line]
`∫` [added up continuously over]
-/
def momMean (hbar : ℝ) (ψ ψ' : ℝ → ℂ) : ℝ :=
  ∫ x, (starRingEnd ℂ (ψ x) * (-Complex.I * hbar * ψ' x)).re

/-- Momentum second moment `⟨p²⟩ = ∫ ℏ²·normSq ψ'` (`‖p̂ ψ‖²` for `p̂ = −iℏ d/dx`).
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`Complex.normSq` [the squared size a² + b², the square of the modulus a + b·i]
`ℝ` [the real numbers: every number on the number line]
`∫` [added up continuously over]
-/
def momSq (hbar : ℝ) (ψ' : ℝ → ℂ) : ℝ := ∫ x, hbar ^ 2 * Complex.normSq (ψ' x)

/-- Momentum spread `Δp = √(⟨p²⟩ − ⟨p⟩²)`.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`ℝ` [the real numbers: every number on the number line]
`Real.sqrt` [the number that squares to give this]
`∫` [added up continuously over]
-/
def momSpread (hbar : ℝ) (ψ ψ' : ℝ → ℂ) : ℝ :=
  Real.sqrt (momSq hbar ψ' - momMean hbar ψ ψ' ^ 2)

/-- The real derivative of the Madelung wavefunction is `madelungDeriv`: `ψ'` is not assumed, it is
the derivative of the concrete `ψ`.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`≠` [is not equal to]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
`HasDerivAt` [is changing at that rate at that time]
-/
theorem madelung_hasDerivAt (hbar : ℝ) (hh : hbar ≠ 0) (ρ ρ' S S' : ℝ → ℝ) (x : ℝ)
    (hx : 0 < ρ x) (hd : HasDerivAt ρ (ρ' x) x) (hS : HasDerivAt S (S' x) x) :
    HasDerivAt (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S' x) x := by
  have _ := hh
  have hs : HasDerivAt (fun y : ℝ => (Real.sqrt (ρ y) : ℂ))
      ((ρ' x / (2 * Real.sqrt (ρ x)) : ℝ) : ℂ) x :=
    (hd.sqrt hx.ne').ofReal_comp
  have hexp : HasDerivAt (fun y : ℝ => Complex.exp (Complex.I * ((S y / hbar : ℝ) : ℂ)))
      (Complex.exp (Complex.I * ((S x / hbar : ℝ) : ℂ)) *
        (Complex.I * ((S' x / hbar : ℝ) : ℂ))) x :=
    (((hS.div_const hbar).ofReal_comp).const_mul Complex.I).cexp
  convert hs.mul hexp using 1 <;> try rfl
  rw [madelungDeriv]
  push_cast
  ring

/-- Pointwise: `ℏ²·‖ψ'‖² = ρ·S'² + ρ·N²` for the Madelung wavefunction, eliminating the cross term
(`madelungDeriv` is the derivative of `madelung` by `madelung_hasDerivAt`).
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`Complex.normSq` [the squared size a² + b², the square of the modulus a + b·i]
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`≠` [is not equal to]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
-/
theorem madelung_normSq (hbar : ℝ) (hh : hbar ≠ 0) (ρ ρ' S S' : ℝ → ℝ) (x : ℝ) (hx : 0 < ρ x) :
    hbar ^ 2 * Complex.normSq (madelungDeriv hbar ρ ρ' S S' x)
      = ρ x * S' x ^ 2 + ρ x * ncMom hbar ρ ρ' x ^ 2 := by
  rw [madelungDeriv]
  set s : ℝ := Real.sqrt (ρ x) with hs
  have hs0 : 0 < s := by rw [hs]; exact Real.sqrt_pos.2 hx
  have hsq : s ^ 2 = ρ x := by rw [hs]; exact Real.sq_sqrt hx.le
  have hE : Complex.normSq (Complex.exp (Complex.I * ((S x / hbar : ℝ) : ℂ))) = 1 := by
    rw [Complex.normSq_eq_norm_sq, Complex.norm_exp]
    simp
  rw [Complex.normSq_mul, hE, mul_one, Complex.normSq_apply]
  simp only [Complex.add_re, Complex.add_im, Complex.I_mul_re, Complex.I_mul_im, Complex.ofReal_re,
    Complex.ofReal_im]
  simp
  rw [rho_ncMom_sq, ← hsq]
  field_simp
  ring

/-- Pointwise: `Re(conj ψ · (−iℏ ψ')) = ρ·S'`: the probability current of the Madelung wavefunction.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`Complex.re` [the real part a of a + b·i]
`starRingEnd ℂ` [the conjugate a + b·i ↦ a − b·i]
`Complex.I` [the square root of −1]
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`≠` [is not equal to]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
-/
theorem madelung_current (hbar : ℝ) (hh : hbar ≠ 0) (ρ ρ' S S' : ℝ → ℝ) (x : ℝ)
    (hx : 0 < ρ x) :
    (starRingEnd ℂ (madelung hbar ρ S x) *
        (-Complex.I * hbar * madelungDeriv hbar ρ ρ' S S' x)).re = ρ x * S' x := by
  rw [madelung, madelungDeriv, map_mul, Complex.conj_ofReal]
  set s : ℝ := Real.sqrt (ρ x) with hs
  have hsq : s ^ 2 = ρ x := by rw [hs]; exact Real.sq_sqrt hx.le
  set E : ℂ := Complex.exp (Complex.I * ((S x / hbar : ℝ) : ℂ)) with hEdef
  set X : ℂ := ((ρ' x / (2 * s) : ℝ) : ℂ) + Complex.I * ((s * S' x / hbar : ℝ) : ℂ) with hX
  have hE : starRingEnd ℂ E * E = 1 := by rw [hEdef]; exact star_exp_I (S x / hbar)
  have hZ : (s : ℂ) * starRingEnd ℂ E * (-Complex.I * (hbar : ℂ) * (X * E))
      = (s : ℂ) * (-Complex.I * (hbar : ℂ)) * X := by
    calc (s : ℂ) * starRingEnd ℂ E * (-Complex.I * (hbar : ℂ) * (X * E))
        = ((s : ℂ) * (-Complex.I * (hbar : ℂ)) * X) * (starRingEnd ℂ E * E) := by ring
      _ = (s : ℂ) * (-Complex.I * (hbar : ℂ)) * X := by rw [hE, mul_one]
  rw [hZ, hX]
  simp only [Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im, Complex.neg_re,
    Complex.neg_im, Complex.I_re, Complex.I_im, Complex.ofReal_re, Complex.ofReal_im]
  simp
  rw [← hsq]
  field_simp

/-- Weighted variance is nonnegative: `(∫ ρ f)² ≤ ∫ ρ f²` for a unit-mass density `ρ`.
`ℝ` [the real numbers: every number on the number line]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`∀` [for every single one]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
-/
theorem weighted_var_nonneg (ρ f : ℝ → ℝ) (hρ : ∀ x, 0 ≤ ρ x) (hint : Integrable ρ)
    (hnorm : ∫ x, ρ x = 1) (h1 : Integrable (fun x => ρ x * f x))
    (h2 : Integrable (fun x => ρ x * f x ^ 2)) :
    (∫ x, ρ x * f x) ^ 2 ≤ ∫ x, ρ x * f x ^ 2 := by
  set M : ℝ := ∫ y, ρ y * f y with hM
  have hpt : ∀ x, ρ x * (f x - M) ^ 2 = ρ x * f x ^ 2 - 2 * M * (ρ x * f x) + M ^ 2 * ρ x := by
    intro x
    ring
  have hnn : 0 ≤ ∫ x, ρ x * (f x - M) ^ 2 :=
    integral_nonneg fun x => mul_nonneg (hρ x) (sq_nonneg _)
  rw [show (∫ x, ρ x * (f x - M) ^ 2)
      = ∫ x, (ρ x * f x ^ 2 - 2 * M * (ρ x * f x) + M ^ 2 * ρ x)
      from integral_congr_ae (Filter.Eventually.of_forall hpt)] at hnn
  have hA : Integrable (fun x => ρ x * f x ^ 2 - 2 * M * (ρ x * f x)) :=
    h2.sub (h1.const_mul (2 * M))
  rw [integral_add hA (hint.const_mul (M ^ 2)), integral_sub h2 (h1.const_mul (2 * M)),
    integral_const_mul, integral_const_mul, hnorm, mul_one] at hnn
  rw [hM] at hnn ⊢
  nlinarith [hnn]

/-- Hall–Reginatto's step: for the Madelung wavefunction `Δp² = (∫ρS'² − (∫ρS')²) + ∫ρN²`; the
first bracket is `Var(S') ≥ 0`, so the full momentum spread dominates the nonclassical one.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`Complex.normSq` [the squared size a² + b², the square of the modulus a + b·i]
`ℝ` [the real numbers: every number on the number line]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`∀` [for every single one]
`<` [is smaller than]
`≠` [is not equal to]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
-/
theorem momentum_spread_sq (hbar : ℝ) (hh : hbar ≠ 0) (ρ ρ' S S' : ℝ → ℝ)
    (hpos : ∀ x, 0 < ρ x) (hS2 : Integrable (fun x => ρ x * S' x ^ 2))
    (hN2 : Integrable (fun x => ρ x * ncMom hbar ρ ρ' x ^ 2)) :
    momSq hbar (madelungDeriv hbar ρ ρ' S S')
        - momMean hbar (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S') ^ 2
      = ((∫ x, ρ x * S' x ^ 2) - (∫ x, ρ x * S' x) ^ 2)
        + ∫ x, ρ x * ncMom hbar ρ ρ' x ^ 2 := by
  have hpt : ∀ x, hbar ^ 2 * Complex.normSq (madelungDeriv hbar ρ ρ' S S' x)
      = ρ x * S' x ^ 2 + ρ x * ncMom hbar ρ ρ' x ^ 2 :=
    fun x => madelung_normSq hbar hh ρ ρ' S S' x (hpos x)
  have hSq : momSq hbar (madelungDeriv hbar ρ ρ' S S')
      = (∫ x, ρ x * S' x ^ 2) + ∫ x, ρ x * ncMom hbar ρ ρ' x ^ 2 := by
    unfold momSq
    rw [integral_congr_ae (Filter.Eventually.of_forall hpt), integral_add hS2 hN2]
  have hMn : momMean hbar (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S')
      = ∫ x, ρ x * S' x := by
    unfold momMean
    exact integral_congr_ae (Filter.Eventually.of_forall
      fun x => madelung_current hbar hh ρ ρ' S S' x (hpos x))
  rw [hSq, hMn]
  ring

/-- `ncSpread` is `ℏ/2 · √I_F` in closed form, the `hN`/`hsqrt` steps of `exact_uncertainty`.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`Real.sqrt` [the number that squares to give this]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
-/
theorem ncSpread_eq (hbar : ℝ) (ρ ρ' : ℝ → ℝ) (hh : 0 ≤ hbar) :
    ncSpread hbar ρ ρ' = hbar / 2 * Real.sqrt (fisher ρ ρ') := by
  have hN : ∫ x, ρ x * ncMom hbar ρ ρ' x ^ 2 = hbar ^ 2 / 4 * fisher ρ ρ' := by
    simp_rw [rho_ncMom_sq]
    rw [integral_const_mul]
    rfl
  have hsqrt : Real.sqrt (hbar ^ 2 / 4 * fisher ρ ρ') =
      hbar / 2 * Real.sqrt (fisher ρ ρ') := by
    rw [show hbar ^ 2 / 4 * fisher ρ ρ' = (hbar / 2) ^ 2 * fisher ρ ρ' by ring,
      Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by positivity)]
  unfold ncSpread
  rw [hN, hsqrt]

/-- The momentum spread dominates the nonclassical spread: `ΔN ≤ Δp`.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`ℝ` [the real numbers: every number on the number line]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`∀` [for every single one]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
`≠` [is not equal to]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
-/
theorem nc_le_momSpread (hbar : ℝ) (hh : hbar ≠ 0) (ρ ρ' S S' : ℝ → ℝ)
    (hpos : ∀ x, 0 < ρ x) (hint : Integrable ρ) (hnorm : ∫ x, ρ x = 1)
    (hS1 : Integrable (fun x => ρ x * S' x)) (hS2 : Integrable (fun x => ρ x * S' x ^ 2))
    (hN2 : Integrable (fun x => ρ x * ncMom hbar ρ ρ' x ^ 2)) :
    ncSpread hbar ρ ρ' ≤ momSpread hbar (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S') := by
  have hvar : (∫ x, ρ x * S' x) ^ 2 ≤ ∫ x, ρ x * S' x ^ 2 :=
    weighted_var_nonneg ρ S' (fun x => (hpos x).le) hint hnorm hS1 hS2
  have hsq := momentum_spread_sq hbar hh ρ ρ' S S' hpos hS2 hN2
  unfold ncSpread momSpread
  apply Real.sqrt_le_sqrt
  rw [hsq]
  linarith

/-- The full Kennard relation `ℏ/2 ≤ Δx·Δp` for the Madelung wavefunction: Cramér–Rao gives
`ℏ/2 ≤ √V·ΔN`, and `ΔN ≤ Δp`. `μ` is any centre, so in particular the mean of `x`.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`<` [is smaller than]
`Real.sqrt` [the number that squares to give this]
`HasDerivAt` [is changing at that rate at that time]
-/
theorem heisenberg (μ hbar : ℝ) (hh : 0 < hbar) (ρ ρ' S S' : ℝ → ℝ)
    (hpos : ∀ x, 0 < ρ x) (hd : ∀ y, HasDerivAt ρ (ρ' y) y) (hint : Integrable ρ)
    (hnorm : ∫ x, ρ x = 1) (hA : Integrable (fun x => (x - μ) ^ 2 * ρ x))
    (hintr : Integrable (fun x => (x - μ) * ρ' x))
    (hC : Integrable (fun x => ρ' x ^ 2 / ρ x))
    (htb : Tendsto (fun x => (x - μ) * ρ x) atBot (𝓝 0))
    (htt : Tendsto (fun x => (x - μ) * ρ x) atTop (𝓝 0))
    (hS1 : Integrable (fun x => ρ x * S' x)) (hS2 : Integrable (fun x => ρ x * S' x ^ 2))
    (hN2 : Integrable (fun x => ρ x * ncMom hbar ρ ρ' x ^ 2)) :
    hbar / 2 ≤ Real.sqrt (∫ x, (x - μ) ^ 2 * ρ x) *
      momSpread hbar (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S') := by
  have hcr : 1 ≤ (∫ x, (x - μ) ^ 2 * ρ x) * fisher ρ ρ' :=
    cramer_rao μ ρ ρ' hpos hd hint hnorm hA hintr hC htb htt
  have hV : 0 ≤ ∫ x, (x - μ) ^ 2 * ρ x :=
    integral_nonneg fun x => mul_nonneg (sq_nonneg _) (hpos x).le
  have h1 : hbar / 2 ≤ Real.sqrt (∫ x, (x - μ) ^ 2 * ρ x) * ncSpread hbar ρ ρ' := by
    have hcr' := heisenberg_from_cr (∫ x, (x - μ) ^ 2 * ρ x) (fisher ρ ρ') hbar hV hh.le hcr
    rwa [← ncSpread_eq hbar ρ ρ' hh.le] at hcr'
  have h2 := nc_le_momSpread hbar hh.ne' ρ ρ' S S' hpos hint hnorm hS1 hS2 hN2
  calc hbar / 2 ≤ Real.sqrt (∫ x, (x - μ) ^ 2 * ρ x) * ncSpread hbar ρ ρ' := h1
    _ ≤ Real.sqrt (∫ x, (x - μ) ^ 2 * ρ x) *
          momSpread hbar (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S') :=
        mul_le_mul_of_nonneg_left h2 (Real.sqrt_nonneg _)

/-- Witness for `madelung_normSq` and `madelung_current` at `ℏ = 1`, `ρ = 4`, `ρ' = 2`, `S = x`,
`S' = 3`, `x = 0`: the identities at the numbers.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`Complex.normSq` [the squared size a² + b², the square of the modulus a + b·i]
`Complex.re` [the real part a of a + b·i]
`starRingEnd ℂ` [the conjugate a + b·i ↦ a − b·i]
`ℝ` [the real numbers: every number on the number line]
`∧` [both are true]
-/
theorem witness_madelung :
    (1 ^ 2 * Complex.normSq (madelungDeriv 1 (fun _ => 4) (fun _ => 2) (fun x => x) (fun _ => 3) 0)
        = 4 * 3 ^ 2 + 4 * ncMom 1 (fun _ => 4) (fun _ => 2) 0 ^ 2) ∧
    ((starRingEnd ℂ (madelung 1 (fun _ => 4) (fun x => x) 0) *
        (-Complex.I * 1 * madelungDeriv 1 (fun _ => 4) (fun _ => 2) (fun x => x) (fun _ => 3) 0)).re
      = 4 * 3) :=
  ⟨madelung_normSq 1 one_ne_zero (fun _ => 4) (fun _ => 2) (fun x => x) (fun _ => 3) 0
      (by norm_num),
    madelung_current 1 one_ne_zero (fun _ => 4) (fun _ => 2) (fun x => x) (fun _ => 3) 0
      (by norm_num)⟩

/-- The verdict with the Cramér–Rao hypothesis discharged: `verdict_l9`, and the
Cauchy–Schwarz step and the Cramér–Rao bound with no `hcs` hypothesis.
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`∀` [for every single one]
`∧` [both are true]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`HasDerivAt` [is changing at that rate at that time]
`<` [is smaller than]
-/
theorem verdict_l9b :
    type_of% verdict_l9 ∧
    (∀ (μ : ℝ) (ρ ρ' : ℝ → ℝ), (∀ x, 0 < ρ x) →
      Integrable (fun x => (x - μ) ^ 2 * ρ x) → Integrable (fun x => (x - μ) * ρ' x) →
      Integrable (fun x => ρ' x ^ 2 / ρ x) →
      (∫ x, (x - μ) * ρ' x) ^ 2 ≤ (∫ x, (x - μ) ^ 2 * ρ x) * fisher ρ ρ') ∧
    (∀ (μ : ℝ) (ρ ρ' : ℝ → ℝ), (∀ x, 0 < ρ x) → (∀ y, HasDerivAt ρ (ρ' y) y) →
      Integrable ρ → (∫ x, ρ x = 1) →
      Integrable (fun x => (x - μ) ^ 2 * ρ x) → Integrable (fun x => (x - μ) * ρ' x) →
      Integrable (fun x => ρ' x ^ 2 / ρ x) →
      Tendsto (fun x => (x - μ) * ρ x) atBot (𝓝 0) →
      Tendsto (fun x => (x - μ) * ρ x) atTop (𝓝 0) →
      1 ≤ (∫ x, (x - μ) ^ 2 * ρ x) * fisher ρ ρ') :=
  ⟨verdict_l9, fun μ ρ ρ' hpos hA hB hC => cauchy_schwarz_fisher μ ρ ρ' hpos hA hB hC,
    fun μ ρ ρ' hpos hd hint hnorm hA hB hC htb htt =>
      cramer_rao μ ρ ρ' hpos hd hint hnorm hA hB hC htb htt⟩

/-- The verdict of : `verdict_l9b`, the pointwise Madelung norm and current, the
momentum-spread decomposition, the Kennard relation `Δx·Δp ≥ ℏ/2`, and the closed witness.
`ℂ` [the complex numbers: pairs a + b·i with i² = −1]
`Complex.normSq` [the squared size a² + b², the square of the modulus a + b·i]
`Complex.re` [the real part a of a + b·i]
`starRingEnd ℂ` [the conjugate a + b·i ↦ a − b·i]
`ℝ` [the real numbers: every number on the number line]
`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`≠` [is not equal to]
`∀` [for every single one]
`∧` [both are true]
`∫` [added up continuously over]
`≤` [is smaller than or equal to]
`<` [is smaller than]
`HasDerivAt` [is changing at that rate at that time]
`Real.sqrt` [the number that squares to give this]
-/
theorem verdict_l9c :
    type_of% verdict_l9b ∧
    (∀ (hbar : ℝ) (ρ ρ' S S' : ℝ → ℝ) (x : ℝ), hbar ≠ 0 → 0 < ρ x →
      HasDerivAt ρ (ρ' x) x → HasDerivAt S (S' x) x →
      HasDerivAt (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S' x) x) ∧
    (∀ (hbar : ℝ) (ρ ρ' S S' : ℝ → ℝ) (x : ℝ), hbar ≠ 0 → 0 < ρ x →
      hbar ^ 2 * Complex.normSq (madelungDeriv hbar ρ ρ' S S' x)
        = ρ x * S' x ^ 2 + ρ x * ncMom hbar ρ ρ' x ^ 2) ∧
    (∀ (hbar : ℝ) (ρ ρ' S S' : ℝ → ℝ) (x : ℝ), hbar ≠ 0 → 0 < ρ x →
      (starRingEnd ℂ (madelung hbar ρ S x) *
        (-Complex.I * hbar * madelungDeriv hbar ρ ρ' S S' x)).re = ρ x * S' x) ∧
    (∀ (hbar : ℝ) (ρ ρ' S S' : ℝ → ℝ), hbar ≠ 0 → (∀ x, 0 < ρ x) → Integrable ρ →
      (∫ x, ρ x = 1) → Integrable (fun x => ρ x * S' x) →
      Integrable (fun x => ρ x * S' x ^ 2) →
      Integrable (fun x => ρ x * ncMom hbar ρ ρ' x ^ 2) →
      momSq hbar (madelungDeriv hbar ρ ρ' S S')
          - momMean hbar (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S') ^ 2
        = ((∫ x, ρ x * S' x ^ 2) - (∫ x, ρ x * S' x) ^ 2)
          + ∫ x, ρ x * ncMom hbar ρ ρ' x ^ 2) ∧
    (∀ (μ hbar : ℝ) (ρ ρ' S S' : ℝ → ℝ), 0 < hbar → (∀ x, 0 < ρ x) →
      (∀ y, HasDerivAt ρ (ρ' y) y) → Integrable ρ → (∫ x, ρ x = 1) →
      Integrable (fun x => (x - μ) ^ 2 * ρ x) → Integrable (fun x => (x - μ) * ρ' x) →
      Integrable (fun x => ρ' x ^ 2 / ρ x) →
      Tendsto (fun x => (x - μ) * ρ x) atBot (𝓝 0) →
      Tendsto (fun x => (x - μ) * ρ x) atTop (𝓝 0) →
      Integrable (fun x => ρ x * S' x) → Integrable (fun x => ρ x * S' x ^ 2) →
      Integrable (fun x => ρ x * ncMom hbar ρ ρ' x ^ 2) →
      hbar / 2 ≤ Real.sqrt (∫ x, (x - μ) ^ 2 * ρ x) *
        momSpread hbar (madelung hbar ρ S) (madelungDeriv hbar ρ ρ' S S')) ∧
    ((1 ^ 2 * Complex.normSq (madelungDeriv 1 (fun _ => 4) (fun _ => 2) (fun x => x)
          (fun _ => 3) 0) = 4 * 3 ^ 2 + 4 * ncMom 1 (fun _ => 4) (fun _ => 2) 0 ^ 2) ∧
      ((starRingEnd ℂ (madelung 1 (fun _ => 4) (fun x => x) 0) *
        (-Complex.I * 1 * madelungDeriv 1 (fun _ => 4) (fun _ => 2) (fun x => x)
          (fun _ => 3) 0)).re = 4 * 3)) :=
  ⟨verdict_l9b,
    fun hbar ρ ρ' S S' x hh hx hd hS => madelung_hasDerivAt hbar hh ρ ρ' S S' x hx hd hS,
    fun hbar ρ ρ' S S' x hh hx => madelung_normSq hbar hh ρ ρ' S S' x hx,
    fun hbar ρ ρ' S S' x hh hx => madelung_current hbar hh ρ ρ' S S' x hx,
    fun hbar ρ ρ' S S' hh hpos _ _ _ hS2 hN2 =>
      momentum_spread_sq hbar hh ρ ρ' S S' hpos hS2 hN2,
    fun μ hbar ρ ρ' S S' hh hpos hd hint hnorm hA hintr hC htb htt hS1 hS2 hN2 =>
      heisenberg μ hbar hh ρ ρ' S S' hpos hd hint hnorm hA hintr hC htb htt hS1 hS2 hN2,
    witness_madelung⟩

end ExactUncertainty
end Quantum

end
