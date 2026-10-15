/-
The four-branch tetrahedral limit of a `C²` field on `ℝ³` is the coordinate Laplacian
(`QREG-TETRA-LAPLACIAN`).

`Numeric.Quantum.TetraLimit.tetra4_tendsto_field` computes the four-branch limit of `F : (Fin 3 → ℝ) → ℝ`
along the four tetrahedral directions `v ∈ {(1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)}` as
`(Σₙ ∂²ₜ F(p + t vₙ)|₀) / 4`, and records the identification of that limit with
`∂xx F + ∂yy F + ∂zz F` as its residual obligation. This file discharges that obligation.

## Landed here

* `line_second_deriv`: for `F ∈ C²(ℝ³)` and `p, v : ℝ³`, `(t ↦ F (p + t v))''(0) = D²F(p)(v, v)`.
* `tetra_sum_eq_lap`: the four-direction mean `(Σₙ ∂²ₜ F(p + t vₙ)|₀) / 4` equals the coordinate
  Laplacian `lap F p = Σᵢ ∂²ₜ F(p + t eᵢ)|₀` with `eᵢ = Pi.single i 1`. The four directions satisfy
  `Σₙ vₙ vₙᵀ = 4 · 1`, so the identification needs no symmetry of the second derivative of `F`.
* `tetra4_tendsto_laplacian`: `tetra4 (fun a b c => F ![a, b, c]) x y z → lap F ![x, y, z]` as
  `h → 0` with `h ≠ 0`: `tetra4_tendsto_field` composed with `tetra_sum_eq_lap`.

## Not claimed

* A rate of convergence. The statements are limits over the filter `𝓝[≠] 0`.
* Floating-point behaviour of the stencil, and conditioning near `h = 0`.
* Anything about third and higher derivatives of `F`, or about fields that are only `C¹`.

## Complexity

`tetra4` evaluates `F` 5 times per `h` (4 branches and the centre); `lap` reads the three coordinate
lines. Both are O(1) in `F`-evaluations, time and space. The limits are over `𝓝[≠] 0`; no bound is
stated.

The registered notation used below is glossed as: `deriv` is the derivative of a function of one
real variable; `fderiv ℝ` is the Fréchet derivative; `Pi.single i 1` is the `i`-th canonical basis
vector of `Fin 3 → ℝ`.
-/
import Numeric.Quantum.TetraLimit
import Mathlib.Analysis.Calculus.ContDiff.Comp
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Linear
import Mathlib.Analysis.Normed.Operator.Bilinear
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Tactic

namespace Numeric.TetraLaplacian

open Filter Topology

/-- Coordinate Laplacian of `F : ℝ³ → ℝ` at `p`: the sum of the second derivatives along the three
    coordinate lines `t ↦ F (p + t eᵢ)`, `eᵢ = Pi.single i 1`.
    Complexity: 3 second derivatives of one-variable functions, O(1) time and space. -/
noncomputable def lap (F : (Fin 3 → ℝ) → ℝ) (p : Fin 3 → ℝ) : ℝ :=
  ∑ i : Fin 3, deriv (deriv (fun t : ℝ => F (p + t • Pi.single i 1))) 0

/-- Second directional derivative as the second Fréchet derivative of `F` applied to `v` twice:
    for `F ∈ C²(ℝ³)`, `(t ↦ F (p + t v))''(0) = D(F')(p)(v)(v)`.
    Complexity: O(1), one chain rule composed with the evaluation map at `v`. -/
theorem line_second_deriv (F : (Fin 3 → ℝ) → ℝ) (hF : ContDiff ℝ 2 F) (p v : Fin 3 → ℝ) :
    deriv (deriv (fun t : ℝ => F (p + t • v))) 0 = fderiv ℝ (fderiv ℝ F) p v v := by
  have hdF : Differentiable ℝ F := hF.differentiable (by norm_num)
  have hG : ContDiff ℝ 1 (fderiv ℝ F) := by
    have h2 : ContDiff ℝ (1 + 1) F := by
      rw [show ((1 : WithTop ℕ∞) + 1) = 2 by norm_num]; exact hF
    exact ((contDiff_succ_iff_fderiv (𝕜 := ℝ) (f := F) (n := 1)).mp h2).2.2
  have hGd : Differentiable ℝ (fderiv ℝ F) := hG.differentiable (by norm_num)
  have hline : ∀ t : ℝ, HasDerivAt (fun s : ℝ => p + s • v) v t := by
    intro t
    have h := ((hasDerivAt_id' t).smul_const v).const_add p
    simpa using h
  have hderiv : deriv (fun t : ℝ => F (p + t • v))
      = fun t : ℝ => fderiv ℝ F (p + t • v) v := by
    funext t
    have hl : HasFDerivAt F (fderiv ℝ F (p + t • v)) (p + t • v) :=
      (hdF (p + t • v)).hasFDerivAt
    have h := hl.comp_hasDerivAt t (hline t)
    exact h.deriv
  have hl : HasFDerivAt (fderiv ℝ F) (fderiv ℝ (fderiv ℝ F) p) p := (hGd p).hasFDerivAt
  have hinner : HasDerivAt (fun t : ℝ => fderiv ℝ F (p + t • v))
      ((fderiv ℝ (fderiv ℝ F) p) v) 0 := by
    have h := hl.comp_hasDerivAt_of_eq (0 : ℝ) (hline 0) (by simp)
    exact h
  have hl2 : HasFDerivAt (ContinuousLinearMap.apply ℝ ℝ v) (ContinuousLinearMap.apply ℝ ℝ v)
      ((fun t : ℝ => fderiv ℝ F (p + t • v)) 0) :=
    ContinuousLinearMap.hasFDerivAt (ContinuousLinearMap.apply ℝ ℝ v)
  have houter : HasDerivAt (fun t : ℝ => fderiv ℝ F (p + t • v) v)
      ((fderiv ℝ (fderiv ℝ F) p) v v) 0 := by
    have h := hl2.comp_hasDerivAt (0 : ℝ) hinner
    exact h
  rw [hderiv]
  exact houter.deriv

/-- Expansion of a continuous bilinear form on `ℝ³` in the canonical basis
    `eᵢ = Pi.single i 1`: `B u w = Σᵢ Σⱼ uᵢ wⱼ B eᵢ eⱼ`.
    Complexity: 9 basis evaluations, O(1). -/
private lemma bilin_expand (B : (Fin 3 → ℝ) →L[ℝ] (Fin 3 → ℝ) →L[ℝ] ℝ) (u w : Fin 3 → ℝ) :
    B u w = ∑ i : Fin 3, ∑ j : Fin 3, u i * w j * B (Pi.single i 1) (Pi.single j 1) := by
  have hu : u = u 0 • Pi.single (0 : Fin 3) 1 + u 1 • Pi.single (1 : Fin 3) 1
      + u 2 • Pi.single (2 : Fin 3) 1 := by
    nth_rewrite 1 [pi_eq_sum_univ' u]
    rw [Fin.sum_univ_three]
  have hw : w = w 0 • Pi.single (0 : Fin 3) 1 + w 1 • Pi.single (1 : Fin 3) 1
      + w 2 • Pi.single (2 : Fin 3) 1 := by
    nth_rewrite 1 [pi_eq_sum_univ' w]
    rw [Fin.sum_univ_three]
  nth_rewrite 1 [hu]
  nth_rewrite 1 [hw]
  simp only [Fin.sum_univ_three, ContinuousLinearMap.map_add, ContinuousLinearMap.map_smul,
    add_apply, smul_apply, smul_eq_mul]
  ring

/-- The four tetrahedral directions have second moment `Σₙ vₙ vₙᵀ = 4 · 1`, hence
    `Σₙ B vₙ vₙ = 4 Σᵢ B eᵢ eᵢ` for every continuous bilinear form `B` on `ℝ³`.
    Complexity: 4 bilinear evaluations plus the 9-entry expansion, O(1). -/
private lemma tetra_bilin (B : (Fin 3 → ℝ) →L[ℝ] (Fin 3 → ℝ) →L[ℝ] ℝ) :
    B (![1, 1, 1] : Fin 3 → ℝ) ![1, 1, 1] + B ![1, -1, -1] ![1, -1, -1]
      + B ![-1, 1, -1] ![-1, 1, -1] + B ![-1, -1, 1] ![-1, -1, 1]
    = 4 * (B (Pi.single 0 1) (Pi.single 0 1) + B (Pi.single 1 1) (Pi.single 1 1)
        + B (Pi.single 2 1) (Pi.single 2 1)) := by
  rw [bilin_expand B ![1, 1, 1] ![1, 1, 1], bilin_expand B ![1, -1, -1] ![1, -1, -1],
    bilin_expand B ![-1, 1, -1] ![-1, 1, -1], bilin_expand B ![-1, -1, 1] ![-1, -1, 1]]
  simp only [Fin.sum_univ_three]
  simp
  ring

/-- The tetrahedral four-branch mean is the coordinate Laplacian: the four branch second
    derivatives sum to `4 · lap F p`.
    Proof: `line_second_deriv` identifies each branch with `D²F(p)(v, v)` and `tetra_bilin` applies
    the identity `Σₙ vₙ vₙᵀ = 4 · 1`.
    Complexity: 4 second derivatives of `F` along lines plus 3 for `lap`, O(1). -/
theorem tetra_sum_eq_lap (F : (Fin 3 → ℝ) → ℝ) (hF : ContDiff ℝ 2 F) (x y z : ℝ) :
    (deriv (deriv (fun t : ℝ => F (![x, y, z] + t • (![1, 1, 1] : Fin 3 → ℝ)))) 0
      + deriv (deriv (fun t : ℝ => F (![x, y, z] + t • (![1, -1, -1] : Fin 3 → ℝ)))) 0
      + deriv (deriv (fun t : ℝ => F (![x, y, z] + t • (![-1, 1, -1] : Fin 3 → ℝ)))) 0
      + deriv (deriv (fun t : ℝ => F (![x, y, z] + t • (![-1, -1, 1] : Fin 3 → ℝ)))) 0) / 4
    = lap F ![x, y, z] := by
  have hB : ∀ v : Fin 3 → ℝ, deriv (deriv (fun t : ℝ => F (![x, y, z] + t • v))) 0
      = fderiv ℝ (fderiv ℝ F) ![x, y, z] v v := fun v => line_second_deriv F hF ![x, y, z] v
  rw [hB ![1, 1, 1], hB ![1, -1, -1], hB ![-1, 1, -1], hB ![-1, -1, 1]]
  rw [lap, Fin.sum_univ_three]
  rw [hB (Pi.single 0 1), hB (Pi.single 1 1), hB (Pi.single 2 1)]
  rw [tetra_bilin (fderiv ℝ (fderiv ℝ F) ![x, y, z])]
  ring

/-- The four-branch tetrahedral stencil of a `C²` field tends to the coordinate Laplacian:
    `tetra4 (fun a b c => F ![a, b, c]) x y z → lap F ![x, y, z]` as `h → 0` with `h ≠ 0`.
    Proof: `tetra4_tendsto_field` for the limit, `tetra_sum_eq_lap` for its value.
    Complexity: 5 evaluations of `F` per `h`, O(1). -/
theorem tetra4_tendsto_laplacian (F : (Fin 3 → ℝ) → ℝ) (hF : ContDiff ℝ 2 F) (x y z : ℝ) :
    Tendsto (Numeric.BranchLaplacian.tetra4 (fun a b c => F ![a, b, c]) x y z) (𝓝[≠] 0)
      (𝓝 (lap F ![x, y, z])) := by
  have h := Numeric.TetraLimit.tetra4_tendsto_field F hF x y z
  rw [tetra_sum_eq_lap F hF x y z] at h
  exact h

end Numeric.TetraLaplacian
