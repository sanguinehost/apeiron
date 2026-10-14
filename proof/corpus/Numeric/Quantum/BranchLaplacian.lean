/-
Branch stencils for the Laplacian of the amplitude (`QREG-BRANCH-LAPLACIAN`, epic
the quantum-register note, "Building blocks" rows 3c, 3d, 4).

Lohmiller and Slotine (arXiv 2405.06328) compute the quantum wave from the classical action; Vattay
(arXiv 2605.02621, 2606.05197) objects that one classical branch (value and gradient) cannot
supply `∇²R` of the amplitude `R = √ρ`, so the quantum potential `Q = −(ℏ²/2M) ∇²R / R` is missed.
The vendored `Sanguine.QuantumClassicalIsomorphism` proves the 2-D obstruction
(`single_branch_underdetermines`) and the 2-D four-branch reconstruction
(`stencil4_reconstructs_laplacian`). This file extends it.

## Landed here

* `tetra_stencil_reconstructs_laplacian` (row 3c): in `ℝ³` the four tetrahedral branches
  `x + h v`, `v ∈ {(1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)}` give `∇²f = 2 (a11 + a22 + a33)` exactly
  for the general quadratic `quad3` (10 coefficients), every `h ≠ 0`.
* `branch_count_ge_four` (row 3c, minimality `k ≥ d + 1`): a branch family `v : Fin k → ℝ³` with
  weights `w` that has zero first moment (`Σ w v = 0`, cancels the gradient) and isotropic second
  moment (`Σ w (v ⬝ u) v = c u`, `c ≠ 0`, reads the Laplacian) has `k ≥ 4`.
  `tetra_family_attains_four` is the joint witness: the tetrahedron with `w = 1`, `c = 4`.
* `secondDiff_tendsto`, `stencil4_tendsto_laplacian` (row 3d): beyond quadratics, for `C²`
  sections the 1-D second difference tends to `g''` and the 2-D four-branch stencil to
  `∂xx f + ∂yy f` as `h → 0` (L'Hôpital, `deriv.lhopital_zero_nhdsNE`).
* Row 4, the chain-rule term of `R ∘ t`: `second_derivative_comp` (1-D identity
  `(R ∘ t)'' = R'' (t')² + R' t''`), `stencil4_quad_comp_affine`
  (`R'' |∇t|²` is read exactly), `stencil4_affine_comp_quad` (`R' ∇²t` is read exactly).

## Not claimed

* Exactness for non-polynomial amplitudes at finite `h`: row 3d is a limit, with no rate.
* That the branches of a Schrödinger evaluator are these stencils: the file is about the stencils.
* A noise or conditioning statement: the stencil divides by `h²`; no floating-point claim.
* The minimality bound is for the two moment conditions above in `ℝ³`; it is not a statement about
  other stencils (e.g. ones that use the centre value as a branch).

## Complexity

`tetra4` evaluates `f` 5 times (4 branches + the centre) and `stencil4` 5 times (4 + centre); both
are O(1) in time and space, counted in `f`-evaluations.

The axiom cone is checked from a scratch file outside the repository.
-/
import Substrate.Algebra.QuantumField.QuantumClassicalIsomorphism
import Mathlib.Analysis.Calculus.LHopital
import Mathlib.Analysis.Calculus.ContDiff.Deriv
import Mathlib.Analysis.Calculus.Deriv.Shift
import Mathlib.LinearAlgebra.Dimension.OrzechProperty
import Mathlib.Tactic

namespace Numeric.BranchLaplacian

open Sanguine.QuantumClassicalIsomorphism
open Filter Topology

/-! ## Row 3c: the tetrahedral stencil in `ℝ³` -/

/-- General real quadratic on `ℝ³`: constant, 3 linear, 3 diagonal and 3 cross coefficients. -/
noncomputable def quad3 (c g1 g2 g3 a11 a22 a33 a12 a13 a23 x y z : ℝ) : ℝ :=
  c + g1 * x + g2 * y + g3 * z + a11 * x ^ 2 + a22 * y ^ 2 + a33 * z ^ 2
    + a12 * x * y + a13 * x * z + a23 * y * z

/-- Tetrahedral stencil: branches `x + h v`, `v ∈ {(1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)}`,
    divided by `2 h²`. Complexity: 5 evaluations of `f` (4 branches and the centre), O(1) time and
    space. -/
noncomputable def tetra4 (f : ℝ → ℝ → ℝ → ℝ) (x y z h : ℝ) : ℝ :=
  (f (x + h) (y + h) (z + h) + f (x + h) (y - h) (z - h) + f (x - h) (y + h) (z - h)
      + f (x - h) (y - h) (z + h) - 4 * f x y z) / (2 * h ^ 2)

/-- The four tetrahedral branches read `∇²f = 2 (a11 + a22 + a33)` exactly, every `h ≠ 0`. -/
theorem tetra_stencil_reconstructs_laplacian
    (c g1 g2 g3 a11 a22 a33 a12 a13 a23 x y z h : ℝ) (hh : h ≠ 0) :
    tetra4 (quad3 c g1 g2 g3 a11 a22 a33 a12 a13 a23) x y z h = 2 * (a11 + a22 + a33) := by
  simp only [tetra4, quad3]
  field_simp
  ring

/-! ## Row 3c: minimality, `k ≥ d + 1` -/

/-- Two moment conditions need at least four branches in `ℝ³`: zero first moment and isotropic
    second moment with nonzero constant. -/
theorem branch_count_ge_four {k : ℕ} (v : Fin k → (Fin 3 → ℝ)) (w : Fin k → ℝ) (c : ℝ)
    (hc : c ≠ 0)
    (h1 : ∑ i, w i • v i = 0)
    (h2 : ∀ u : Fin 3 → ℝ, ∑ i, (w i * (v i ⬝ᵥ u)) • v i = c • u) :
    4 ≤ k := by
  have hspan : (⊤ : Submodule ℝ (Fin 3 → ℝ)) ≤ Submodule.span ℝ (Set.range v) := by
    intro u _
    have hu : u = c⁻¹ • ∑ i, (w i * (v i ⬝ᵥ u)) • v i := by
      rw [h2 u, smul_smul, inv_mul_cancel₀ hc, one_smul]
    rw [hu]
    refine Submodule.smul_mem _ _ (Submodule.sum_mem _ fun i _ => ?_)
    exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨i, rfl⟩)
  have h3 : 3 ≤ k := by
    have := finrank_range_le_card (R := ℝ) v
    have hr : Module.finrank ℝ (Submodule.span ℝ (Set.range v)) = 3 := by
      have : Submodule.span ℝ (Set.range v) = ⊤ := top_le_iff.mp hspan
      rw [this]
      simp
    have h' : (Set.range v).finrank ℝ = 3 := hr
    simpa [h'] using this
  by_contra hk
  have hk3 : k = 3 := by omega
  subst hk3
  have hli : LinearIndependent ℝ v :=
    linearIndependent_of_top_le_span_of_card_eq_finrank hspan (by simp)
  have hw : ∀ i, w i = 0 := Fintype.linearIndependent_iff.mp hli w h1
  have hz : ∀ u : Fin 3 → ℝ, c • u = 0 := by
    intro u
    rw [← h2 u]
    simp [hw]
  have := congrFun (hz (fun _ => 1)) 0
  simp at this
  exact hc this

/-- The tetrahedron as a branch family. -/
noncomputable def tetraV : Fin 4 → (Fin 3 → ℝ) :=
  ![![1, 1, 1], ![1, -1, -1], ![-1, 1, -1], ![-1, -1, 1]]

/-- Joint witness: the tetrahedron with unit weights has zero first moment and second moment
    `4 u`, so the bound `k ≥ 4` is attained. -/
theorem tetra_family_attains_four :
    (∑ i, (fun _ : Fin 4 => (1 : ℝ)) i • tetraV i = 0) ∧
      (∀ u : Fin 3 → ℝ,
        ∑ i, ((fun _ : Fin 4 => (1 : ℝ)) i * (tetraV i ⬝ᵥ u)) • tetraV i = (4 : ℝ) • u) := by
  refine ⟨?_, fun u => ?_⟩
  · ext j
    fin_cases j <;> simp [tetraV, Fin.sum_univ_four]
  · ext j
    fin_cases j <;> simp [tetraV, Fin.sum_univ_four, dotProduct, Fin.sum_univ_three] <;> ring

/-! ## Row 3d: the limit beyond quadratics -/

/-- The symmetric difference quotient of a function differentiable at `x` tends to the derivative. -/
private lemma slope_neg_tendsto {D : ℝ → ℝ} {x D' : ℝ} (hD : HasDerivAt D D' x) :
    Tendsto (fun t : ℝ => (D (x + t) - D (x - t)) / (2 * t)) (𝓝[≠] 0) (𝓝 D') := by
  have s1 := hD.tendsto_slope_zero
  have hneg : Tendsto (fun t : ℝ => -t) (𝓝[≠] (0 : ℝ)) (𝓝[≠] 0) :=
    tendsto_nhdsWithin_iff.2
      ⟨(continuous_neg.tendsto' 0 0 (by simp)).mono_left nhdsWithin_le_nhds,
        eventually_nhdsWithin_of_forall fun t ht => by simpa using ht⟩
  have s2 := s1.comp hneg
  have s3 := (s1.add s2).const_mul (1 / 2 : ℝ)
  have hlim : (1 / 2 : ℝ) * (D' + D') = D' := by ring
  rw [hlim] at s3
  refine s3.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t ht
  have ht' : t ≠ 0 := ht
  simp only [Function.comp_apply, smul_eq_mul, sub_eq_add_neg]
  field_simp
  ring

/-- `(g(x+h) + g(x−h) − 2 g(x)) / h² → g''(x)` as `h → 0`, `h ≠ 0`, for `g ∈ C²`.
    Complexity: 3 evaluations of `g` per `h`, O(1). -/
theorem secondDiff_tendsto {g : ℝ → ℝ} (hg : ContDiff ℝ 2 g) (x : ℝ) :
    Tendsto (fun h : ℝ => (g (x + h) + g (x - h) - 2 * g x) / h ^ 2) (𝓝[≠] 0)
      (𝓝 (deriv (deriv g) x)) := by
  have hdg : Differentiable ℝ g := hg.differentiable (by norm_num)
  have hD : ContDiff ℝ 1 (deriv g) := by
    have h2 : ContDiff ℝ (1 + 1) g := by
      rw [show ((1 : WithTop ℕ∞) + 1) = 2 by norm_num]; exact hg
    exact ((contDiff_succ_iff_deriv (𝕜 := ℝ) (f := g) (n := 1)).mp h2).2.2
  have hDd : Differentiable ℝ (deriv g) := hD.differentiable (by norm_num)
  have hφ : ∀ h : ℝ, HasDerivAt (fun h : ℝ => g (x + h) + g (x - h) - 2 * g x)
      (deriv g (x + h) - deriv g (x - h)) h := by
    intro h
    have a := ((hdg (x + h)).hasDerivAt).comp_const_add x h
    have b := ((hdg (x - h)).hasDerivAt).comp_const_sub x h
    have := (a.add b).sub_const (2 * g x)
    have e : deriv g (x + h) - deriv g (x - h) = deriv g (x + h) + -deriv g (x - h) := by ring
    rw [e]; exact this
  have hφd : ∀ h : ℝ, deriv (fun h : ℝ => g (x + h) + g (x - h) - 2 * g x) h
      = deriv g (x + h) - deriv g (x - h) := fun h => (hφ h).deriv
  have hsq : ∀ h : ℝ, deriv (fun h : ℝ => h ^ 2) h = 2 * h := by
    intro h; simp
  have hcont : Continuous (fun h : ℝ => g (x + h) + g (x - h) - 2 * g x) :=
    continuous_iff_continuousAt.2 fun h => (hφ h).continuousAt
  refine deriv.lhopital_zero_nhdsNE (f := fun h : ℝ => g (x + h) + g (x - h) - 2 * g x)
    (g := fun h : ℝ => h ^ 2) (a := 0) ?_ ?_ ?_ ?_ ?_
  · exact eventually_nhdsWithin_of_forall fun h _ => (hφ h).differentiableAt
  · refine eventually_nhdsWithin_of_forall fun h hh => ?_
    rw [hsq]
    have : h ≠ 0 := hh
    simpa using this
  · have := hcont.tendsto 0
    simp only [add_zero, sub_zero] at this
    have h0 : g x + g x - 2 * g x = 0 := by ring
    rw [h0] at this
    exact this.mono_left nhdsWithin_le_nhds
  · have := (continuous_pow 2 : Continuous fun h : ℝ => h ^ 2).tendsto 0
    simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow] at this
    exact this.mono_left nhdsWithin_le_nhds
  · simp only [hφd, hsq]
    exact slope_neg_tendsto (hDd x).hasDerivAt

/-- The 2-D four-branch stencil tends to `∂xx f + ∂yy f` for `C²` sections.
    Complexity: 5 evaluations of `f` per `h`, O(1). -/
theorem stencil4_tendsto_laplacian {f : ℝ → ℝ → ℝ} (x y : ℝ)
    (hx : ContDiff ℝ 2 (fun s => f s y)) (hy : ContDiff ℝ 2 (fun s => f x s)) :
    Tendsto (fun h : ℝ => stencil4 f x y h) (𝓝[≠] 0)
      (𝓝 (deriv (deriv (fun s => f s y)) x + deriv (deriv (fun s => f x s)) y)) := by
  have a := secondDiff_tendsto hx x
  have b := secondDiff_tendsto hy y
  refine (a.add b).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with h hh
  have : h ≠ 0 := hh
  simp only [stencil4]
  field_simp
  ring

/-! ## Row 4: the chain-rule term is seen by the four branches -/

/-- `(R ∘ t)'' = R'' (t')² + R' t''` at `x`, from `HasDerivAt` hypotheses. -/
theorem second_derivative_comp (R R' R'' t t' t'' : ℝ → ℝ)
    (hR : ∀ s, HasDerivAt R (R' s) s) (hR' : ∀ s, HasDerivAt R' (R'' s) s)
    (ht : ∀ s, HasDerivAt t (t' s) s) (ht' : ∀ s, HasDerivAt t' (t'' s) s) (x : ℝ) :
    deriv (deriv (fun s => R (t s))) x = R'' (t x) * t' x ^ 2 + R' (t x) * t'' x := by
  have h1 : deriv (fun s => R (t s)) = fun s => R' (t s) * t' s := by
    funext s
    exact ((hR (t s)).comp s (ht s)).deriv
  rw [h1]
  have := ((hR' (t x)).comp x (ht x)).mul (ht' x)
  rw [show deriv (fun s => R' (t s) * t' s) x
      = R'' (t x) * t' x * t' x + R' (t x) * t'' x from this.deriv]
  ring

/-- Quadratic `R` along an affine `t`: the four branches read `R'' |∇t|² = 2 r2 (τ1² + τ2²)`. -/
theorem stencil4_quad_comp_affine (r0 r1 r2 τ0 τ1 τ2 x y h : ℝ) (hh : h ≠ 0) :
    stencil4 (fun x y => r0 + r1 * (τ0 + τ1 * x + τ2 * y)
        + r2 * (τ0 + τ1 * x + τ2 * y) ^ 2) x y h = 2 * r2 * (τ1 ^ 2 + τ2 ^ 2) := by
  simp only [stencil4]
  field_simp
  ring

/-- Affine `R` of a quadratic `t`: the four branches read `R' ∇²t = r1 (p + q)`. -/
theorem stencil4_affine_comp_quad (r0 r1 R0 a b p q c x y h : ℝ) (hh : h ≠ 0) :
    stencil4 (fun x y => r0 + r1 * quad R0 a b p q c x y) x y h = r1 * (p + q) := by
  simp only [stencil4, quad]
  field_simp
  ring

end Numeric.BranchLaplacian
