/-
Tetrahedral four-branch stencil on `C²` fields in `ℝ³` (`QREG-TETRA-LIMIT`).

Builds on `Numeric.BranchLaplacian`: the exact statement for quadratics
(`tetra_stencil_reconstructs_laplacian`, the `tetra4` definition) and the one-dimensional
second-difference limit (`secondDiff_tendsto`, L'Hôpital via `deriv.lhopital_zero_nhdsNE`).

## Landed here

* `oneSided_tendsto` (Step A): for `g ∈ C²(ℝ)`, `(g h − g 0 − h g'(0)) / h² → g''(0) / 2` as
  `h → 0`, `h ≠ 0`.
* `tetra4_tendsto_lines` (Step B): for four lines `Lᵢ` through a point with `Σᵢ Lᵢ'(0) = 0` and
  `Lᵢ ∈ C²`, the tetrahedral stencil of `f` tends to `(Σᵢ Lᵢ''(0)) / 4` as `h → 0`.
* `tetra4_tendsto_field` (Step C): for `F : (Fin 3 → ℝ) → ℝ` with `ContDiff ℝ 2 F`, the stencil of
  `f a b c = F ![a, b, c]` tends to `(Σᵢ ∂²ᵢ) / 4` along the four branch lines `t ↦ F (p + t vᵢ)`.
  The first-order hypothesis of Step B is discharged here: `Σᵢ vᵢ = 0`, so the directional
  derivatives sum to `fderiv ℝ F p 0 = 0`.

## Not claimed

* That `(Σᵢ Lᵢ''(0)) / 4` equals `∂xx F + ∂yy F + ∂zz F`. Identifying the limit with the
  Laplacian needs the symmetric second-derivative identity for the coordinate directions. This is
  the residual obligation of this module.
* A rate of convergence. The statements are limits as `h → 0` with `h ≠ 0`.
* Floating-point behaviour of the stencil.

## Complexity

`tetra4` evaluates `f` 5 times per `h` (4 branches and the centre), O(1) time and space. The
limits are over the filter `𝓝[≠] 0`; no bound is stated.
-/
import Numeric.Quantum.BranchLaplacian
import Mathlib.Analysis.Calculus.LHopital
import Mathlib.Analysis.Calculus.ContDiff.Deriv
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Tactic

namespace Numeric.TetraLimit

open Filter Topology
open Numeric.BranchLaplacian (tetra4)

/-! ## Step A: one-sided second-order Taylor remainder in one variable -/

/-- Symmetric difference-quotient limit for a derivative: `(D(t) − D(0)) / (2t)`-type slope.
    Complexity: O(1), one slope limit. -/
private lemma slope_zero_tendsto {D : ℝ → ℝ} {D' : ℝ} (hD : HasDerivAt D D' 0) :
    Tendsto (fun t : ℝ => (D t - D 0) / t) (𝓝[≠] 0) (𝓝 D') := by
  have s1 := hD.tendsto_slope_zero
  refine s1.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t ht
  simp only [smul_eq_mul, zero_add]
  rw [div_eq_inv_mul]

/-- For `g ∈ C²(ℝ)`, `(g h − g 0 − h g'(0)) / h² → g''(0) / 2` as `h → 0` along `h ≠ 0`.
    Proof: one application of L'Hôpital, `deriv.lhopital_zero_nhdsNE`, to numerator
    `g h − g 0 − h g'(0)` and denominator `h²`. The derivative quotient
    `(g'(h) − g'(0)) / (2h)` tends to `g''(0) / 2` by `slope_zero_tendsto` on `g'`.
    Complexity: 3 evaluations of `g` and 1 of `g'` per `h`, O(1). -/
theorem oneSided_tendsto {g : ℝ → ℝ} (hg : ContDiff ℝ 2 g) :
    Tendsto (fun h : ℝ => (g h - g 0 - h * deriv g 0) / h ^ 2) (𝓝[≠] 0)
      (𝓝 (deriv (deriv g) 0 / 2)) := by
  have hdg : Differentiable ℝ g := hg.differentiable (by norm_num)
  have hD : ContDiff ℝ 1 (deriv g) := by
    have h2 : ContDiff ℝ (1 + 1) g := by
      rw [show ((1 : WithTop ℕ∞) + 1) = 2 by norm_num]; exact hg
    exact ((contDiff_succ_iff_deriv (𝕜 := ℝ) (f := g) (n := 1)).mp h2).2.2
  have hDd : Differentiable ℝ (deriv g) := hD.differentiable (by norm_num)
  have hφ : ∀ h : ℝ, HasDerivAt (fun h : ℝ => g h - g 0 - h * deriv g 0)
      (deriv g h - deriv g 0) h := by
    intro h
    have a := (hdg h).hasDerivAt
    have b := (hasDerivAt_id' h).mul_const (deriv g 0)
    have := (a.sub_const (g 0)).sub b
    have e : deriv g h - deriv g 0 = deriv g h - 1 * deriv g 0 := by ring
    rw [e]; exact this
  have hφd : ∀ h : ℝ, deriv (fun h : ℝ => g h - g 0 - h * deriv g 0) h
      = deriv g h - deriv g 0 := fun h => (hφ h).deriv
  have hsq : ∀ h : ℝ, deriv (fun h : ℝ => h ^ 2) h = 2 * h := by
    intro h; simp
  have hcont : Continuous (fun h : ℝ => g h - g 0 - h * deriv g 0) :=
    continuous_iff_continuousAt.2 fun h => (hφ h).continuousAt
  refine deriv.lhopital_zero_nhdsNE (f := fun h : ℝ => g h - g 0 - h * deriv g 0)
    (g := fun h : ℝ => h ^ 2) (a := 0) ?_ ?_ ?_ ?_ ?_
  · exact eventually_nhdsWithin_of_forall fun h _ => (hφ h).differentiableAt
  · refine eventually_nhdsWithin_of_forall fun h hh => ?_
    rw [hsq]
    have : h ≠ 0 := hh
    simpa using this
  · refine Tendsto.mono_left ?_ nhdsWithin_le_nhds
    have := hcont.tendsto 0
    convert this using 2
    simp
  · have := (continuous_pow 2 : Continuous fun h : ℝ => h ^ 2).tendsto 0
    simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow] at this
    exact this.mono_left nhdsWithin_le_nhds
  · simp only [hφd, hsq]
    have s := slope_zero_tendsto (D := deriv g) (D' := deriv (deriv g) 0) (hDd 0).hasDerivAt
    have s2 := s.const_mul (1 / 2 : ℝ)
    have hl : (1 / 2 : ℝ) * deriv (deriv g) 0 = deriv (deriv g) 0 / 2 := by ring
    rw [hl] at s2
    refine s2.congr' ?_
    filter_upwards [self_mem_nhdsWithin] with t ht
    have t0 : t ≠ 0 := ht
    field_simp

/-! ## Step B: the four-line limit -/

/-- Four-line tetrahedral limit. The branches are `Lᵢ t = f (x + a t) (y + b t) (z + c t)` for the
    sign vectors `(a,b,c) ∈ {(1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)}`; each `Lᵢ` is `C²` and
    the first-order condition `Σᵢ Lᵢ'(0) = 0` holds. Then `tetra4 f x y z h → (Σᵢ Lᵢ''(0)) / 4`.
    Proof: `tetra4 f x y z h = ½ Σᵢ (Lᵢ h − Lᵢ 0 − h Lᵢ'(0)) / h² + h (Σᵢ Lᵢ'(0)) / (2h²)`;
    the last term vanishes by `hfirst`, and each quotient tends to `Lᵢ''(0)/2` by `oneSided_tendsto`.
    Complexity: 5 evaluations of `f` per `h`, O(1). -/
theorem tetra4_tendsto_lines (f : ℝ → ℝ → ℝ → ℝ) (x y z : ℝ)
    (L₁ L₂ L₃ L₄ : ℝ → ℝ)
    (hL₁ : ∀ t, L₁ t = f (x + t) (y + t) (z + t))
    (hL₂ : ∀ t, L₂ t = f (x + t) (y - t) (z - t))
    (hL₃ : ∀ t, L₃ t = f (x - t) (y + t) (z - t))
    (hL₄ : ∀ t, L₄ t = f (x - t) (y - t) (z + t))
    (g₁ : ContDiff ℝ 2 L₁) (g₂ : ContDiff ℝ 2 L₂) (g₃ : ContDiff ℝ 2 L₃) (g₄ : ContDiff ℝ 2 L₄)
    (hfirst : deriv L₁ 0 + deriv L₂ 0 + deriv L₃ 0 + deriv L₄ 0 = 0) :
    Tendsto (tetra4 f x y z) (𝓝[≠] 0)
      (𝓝 ((deriv (deriv L₁) 0 + deriv (deriv L₂) 0 + deriv (deriv L₃) 0 + deriv (deriv L₄) 0) / 4)) := by
  have S₁ := oneSided_tendsto g₁
  have S₂ := oneSided_tendsto g₂
  have S₃ := oneSided_tendsto g₃
  have S₄ := oneSided_tendsto g₄
  have hsum := (((S₁.add S₂).add S₃).add S₄).const_mul (1 / 2 : ℝ)
  have hlim : (1 / 2 : ℝ) * (deriv (deriv L₁) 0 / 2 + deriv (deriv L₂) 0 / 2
      + deriv (deriv L₃) 0 / 2 + deriv (deriv L₄) 0 / 2)
      = (deriv (deriv L₁) 0 + deriv (deriv L₂) 0 + deriv (deriv L₃) 0 + deriv (deriv L₄) 0) / 4 := by
    ring
  rw [hlim] at hsum
  refine hsum.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with h hh
  have h0 : h ≠ 0 := hh
  have key : tetra4 f x y z h = 1 / 2 * ((L₁ h - L₁ 0 - h * deriv L₁ 0) / h ^ 2
      + (L₂ h - L₂ 0 - h * deriv L₂ 0) / h ^ 2 + (L₃ h - L₃ 0 - h * deriv L₃ 0) / h ^ 2
      + (L₄ h - L₄ 0 - h * deriv L₄ 0) / h ^ 2)
      + h * (deriv L₁ 0 + deriv L₂ 0 + deriv L₃ 0 + deriv L₄ 0) / (2 * h ^ 2) := by
    simp only [tetra4, hL₁, hL₂, hL₃, hL₄, add_zero, sub_zero]
    field_simp
    ring
  rw [key, hfirst]
  simp

/-! ## Step C: the four lines of a `C²` field on `ℝ³` -/

/-- The derivative at `0` of the line `t ↦ F (p + t v)` is `DF(p) v`, for differentiable `F`.
    Complexity: O(1), one chain rule. -/
private lemma line_deriv_zero (F : (Fin 3 → ℝ) → ℝ) (hF : Differentiable ℝ F)
    (p v : Fin 3 → ℝ) :
    deriv (fun t : ℝ => F (p + t • v)) 0 = fderiv ℝ F p v := by
  have h1 : HasDerivAt (fun t : ℝ => t • v) v 0 := by
    simpa using (hasDerivAt_id' (0 : ℝ)).smul_const v
  have h2 : HasDerivAt (fun t : ℝ => p + t • v) v 0 := by
    simpa using h1.const_add p
  have hy : p = (fun t : ℝ => p + t • v) 0 := by simp
  have h3 := (hF p).hasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) h2 hy
  exact h3.deriv

/-- Tetrahedral limit for a `C²` field `F` on `ℝ³`, with `f a b c = F ![a, b, c]`: the four branch
    lines `t ↦ F (p + t vᵢ)` are `C²`, their first derivatives sum to `DF(p)(Σ vᵢ) = 0`, and
    Step B applies. The limit is `(Σᵢ ∂²_t F(p + t vᵢ)|₀) / 4`.
    The identification of this limit with `∂xx F + ∂yy F + ∂zz F` is not made here (residual
    obligation: symmetric second derivatives of `F` along the coordinate directions).
    Complexity: 5 evaluations of `F` per `h`, O(1). -/
theorem tetra4_tendsto_field (F : (Fin 3 → ℝ) → ℝ) (hF : ContDiff ℝ 2 F) (x y z : ℝ) :
    Tendsto (tetra4 (fun a b c => F ![a, b, c]) x y z) (𝓝[≠] 0)
      (𝓝 ((deriv (deriv (fun t : ℝ => F (![x, y, z] + t • (![1, 1, 1] : Fin 3 → ℝ)))) 0
        + deriv (deriv (fun t : ℝ => F (![x, y, z] + t • (![1, -1, -1] : Fin 3 → ℝ)))) 0
        + deriv (deriv (fun t : ℝ => F (![x, y, z] + t • (![-1, 1, -1] : Fin 3 → ℝ)))) 0
        + deriv (deriv (fun t : ℝ => F (![x, y, z] + t • (![-1, -1, 1] : Fin 3 → ℝ)))) 0) / 4)) := by
  have hd : Differentiable ℝ F := hF.differentiable (by norm_num)
  have hline : ∀ v : Fin 3 → ℝ, ContDiff ℝ 2 (fun t : ℝ => F (![x, y, z] + t • v)) := by
    intro v
    exact hF.comp (by fun_prop)
  have hsumv : (![1, 1, 1] : Fin 3 → ℝ) + ![1, -1, -1] + ![-1, 1, -1] + ![-1, -1, 1] = 0 := by
    funext i
    fin_cases i <;> norm_num
  have hfirst : deriv (fun t : ℝ => F (![x, y, z] + t • (![1, 1, 1] : Fin 3 → ℝ))) 0
      + deriv (fun t : ℝ => F (![x, y, z] + t • (![1, -1, -1] : Fin 3 → ℝ))) 0
      + deriv (fun t : ℝ => F (![x, y, z] + t • (![-1, 1, -1] : Fin 3 → ℝ))) 0
      + deriv (fun t : ℝ => F (![x, y, z] + t • (![-1, -1, 1] : Fin 3 → ℝ))) 0 = 0 := by
    simp only [line_deriv_zero F hd]
    rw [← map_add, ← map_add, ← map_add, hsumv, map_zero]
  refine tetra4_tendsto_lines (fun a b c => F ![a, b, c]) x y z
    (fun t : ℝ => F (![x, y, z] + t • (![1, 1, 1] : Fin 3 → ℝ)))
    (fun t : ℝ => F (![x, y, z] + t • (![1, -1, -1] : Fin 3 → ℝ)))
    (fun t : ℝ => F (![x, y, z] + t • (![-1, 1, -1] : Fin 3 → ℝ)))
    (fun t : ℝ => F (![x, y, z] + t • (![-1, -1, 1] : Fin 3 → ℝ)))
    ?_ ?_ ?_ ?_ (hline _) (hline _) (hline _) (hline _) hfirst
  · intro t
    have e : (![x, y, z] + t • (![1, 1, 1] : Fin 3 → ℝ)) = ![x + t, y + t, z + t] := by
      funext i; fin_cases i <;> simp
    simp only [e]
  · intro t
    have e : (![x, y, z] + t • (![1, -1, -1] : Fin 3 → ℝ)) = ![x + t, y - t, z - t] := by
      funext i; fin_cases i <;> simp <;> ring
    simp only [e]
  · intro t
    have e : (![x, y, z] + t • (![-1, 1, -1] : Fin 3 → ℝ)) = ![x - t, y + t, z - t] := by
      funext i; fin_cases i <;> simp <;> ring
    simp only [e]
  · intro t
    have e : (![x, y, z] + t • (![-1, -1, 1] : Fin 3 → ℝ)) = ![x - t, y - t, z + t] := by
      funext i; fin_cases i <;> simp <;> ring
    simp only [e]
