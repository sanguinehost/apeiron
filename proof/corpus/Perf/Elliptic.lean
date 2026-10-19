import ApeironWord
import Perf.Iq1s
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The elliptic form of the quantisation distortion, and the transform-coding bound

Board: `quality-118-qwen2-5-0-5b-FbiGTHMTT5s0s2tJZ1GXsj` (QUALITY-118).

`Perf/Iq1s.lean` states the layer error on a pair of weights as the quadratic form of the Gram
`H = XᵀX`, `H_ii a² + 2 H_ij a b + H_jj b²`, and proves the discriminant bound
`H_ij² − H_ii H_jj ≤ 0` (`Iq1s.pair_discriminant_nonpos`, Cauchy–Schwarz). This module states what
that sign decides.

1. `pair_form_psd`: nonnegative diagonal and `hij² ≤ hii*hjj` make the pair form positive
   semidefinite for every `(a, b)` — the elliptic case, and the parabolic case at equality. The
   hyperbolic case `hij² > hii*hjj`, where the form is negative on some direction, is exactly what the
   discriminant bound excludes. `pair_form_psd_of_gram` instantiates the hypotheses from a Gram matrix,
   so the layer error is never hyperbolic; the conical direction is the pair `(H_ij, −H_ii)`
   (`Iq1s.pair_form_on_direction`).

2. `coding_gain_bound`, `coding_gain_eq_iff`, `coding_gain_ge_one`: for `0 < λ₁`, `0 < λ₂`, the mean
   `(λ₁+λ₂)/2` dominates the geometric mean `√(λ₁λ₂)`, with equality exactly at `λ₁ = λ₂`. Reading
   (`model`, not proved): the eigenbasis coding gain is `G = (mean λ)/(geometric mean λ) ≥ 1`, and no
   orthogonal transform has gain above the eigenbasis, so a fixed Hadamard used in place of the
   eigenbasis has gain at most `G(H)`. The general `n`-dimensional form is `Perf.OnebitW13.gm_le_am`
   (`n ∏ xᵢ^{1/n} ≤ Σ xᵢ`); the two-eigenvalue form here is the one the pair error uses.

3. `register_stage_orthogonal`, `register_stage_pi_div_four`: the 2×2 stage of the learned reflection
   register, `M t = [[cos t, sin t], [sin t, −cos t]]`, is orthogonal for every `t`, and at `t = π/4` it
   is `(√2/2)·[[1,1],[1,−1]] = (1/√2)·[[1,1],[1,−1]]`, the fixed Hadamard. `Perf/QuantumRotation.lean`
   measures the family's verdicts (`kan_probe_sum_loses`, `kan_exact_loses_wall`, `kan_mm_loses_wall`);
   those verdicts are about the learned angles, not about this identity, and they are consistent with
   the bound: a learned stage is still one orthogonal transform, so it cannot exceed the eigenbasis
   gain either. `Perf/Trellis.lean` (`trellis2_confirmed`, `trellis_loses_everywhere`) shows the codebook
   generator is not the transform: at one bit per weight the hashed codebook loses on every linear.

Grade `proved` over `ℝ` for each theorem under the stated hypotheses; the gain reading is `model`. No
measurement is cited here; the constants quoted in the receipts live in `Perf/Receipts/Onebit.lean`.
-/

namespace Perf

namespace Elliptic

open Finset

/-- **The pair form is positive semidefinite on the elliptic/parabolic cone.** With `0 ≤ hii`, `0 ≤ hjj`
and `hij² ≤ hii·hjj`, the quadratic form `hii a² + 2 hij a b + hjj b²` is nonnegative for every `(a, b)`.
The excluded case `hij² > hii·hjj` is the hyperbolic one: some direction then has negative value. -/
theorem pair_form_psd (hii hjj hij : ℝ) (hii0 : 0 ≤ hii) (hjj0 : 0 ≤ hjj)
    (hdisc : hij ^ 2 ≤ hii * hjj) (a b : ℝ) :
    0 ≤ hii * a ^ 2 + 2 * hij * a * b + hjj * b ^ 2 := by
  -- `hii` times the form is a sum of two squares.
  have key : hii * (hii * a ^ 2 + 2 * hij * a * b + hjj * b ^ 2)
      = (hii * a + hij * b) ^ 2 + (hii * hjj - hij ^ 2) * b ^ 2 := by ring
  rcases eq_or_lt_of_le hii0 with h0 | hpos
  · -- `hii = 0`: the discriminant bound forces `hij = 0`, and the form is `hjj b²`.
    have hle : hij ^ 2 ≤ 0 := by rw [← h0] at hdisc; simpa using hdisc
    have hz : hij = 0 := by nlinarith [sq_nonneg hij]
    rw [← h0, hz]
    nlinarith [mul_nonneg hjj0 (sq_nonneg b)]
  · have hnn : 0 ≤ hii * (hii * a ^ 2 + 2 * hij * a * b + hjj * b ^ 2) := by
      rw [key]
      have hb : 0 ≤ (hii * hjj - hij ^ 2) * b ^ 2 := mul_nonneg (by linarith) (sq_nonneg b)
      nlinarith [sq_nonneg (hii * a + hij * b)]
    exact nonneg_of_mul_nonneg_right hnn hpos

/-- A Gram diagonal is a sum of squares. -/
theorem gram_diag_nonneg {T n : ℕ} (X : Fin T → Fin n → ℝ) (i : Fin n) :
    0 ≤ Iq1s.gram X i i := by
  unfold Iq1s.gram
  exact Finset.sum_nonneg fun t _ => mul_self_nonneg _

/-- **The layer pair form is never hyperbolic.** Instantiating `pair_form_psd` from the Gram of a data
matrix: the discriminant hypothesis is `Iq1s.pair_discriminant_nonpos`, so the error form of any pair of
weights is positive semidefinite. -/
theorem pair_form_psd_of_gram {T n : ℕ} (X : Fin T → Fin n → ℝ) (i j : Fin n) (a b : ℝ) :
    0 ≤ Iq1s.gram X i i * a ^ 2 + 2 * Iq1s.gram X i j * a * b + Iq1s.gram X j j * b ^ 2 := by
  refine pair_form_psd (Iq1s.gram X i i) (Iq1s.gram X j j) (Iq1s.gram X i j)
    (gram_diag_nonneg X i) (gram_diag_nonneg X j) ?_ a b
  have h := Iq1s.pair_discriminant_nonpos X i j
  linarith

/-! ### The transform-coding bound

Two eigenvalues `λ₁, λ₂ > 0`. The gap between the arithmetic and geometric means is a square. -/

/-- `(λ₁+λ₂)/2 − √(λ₁λ₂) = (√λ₁ − √λ₂)²/2` for positive `λ₁, λ₂`. -/
theorem coding_gain_sq (l1 l2 : ℝ) (h1 : 0 < l1) (h2 : 0 < l2) :
    (l1 + l2) / 2 - Real.sqrt (l1 * l2) = (Real.sqrt l1 - Real.sqrt l2) ^ 2 / 2 := by
  have hs1 : Real.sqrt l1 ^ 2 = l1 := Real.sq_sqrt (le_of_lt h1)
  have hs2 : Real.sqrt l2 ^ 2 = l2 := Real.sq_sqrt (le_of_lt h2)
  rw [Real.sqrt_mul (le_of_lt h1) l2]
  nlinarith [hs1, hs2]

/-- **Arithmetic mean dominates geometric mean.** `√(λ₁λ₂) ≤ (λ₁+λ₂)/2` for `0 < λ₁`, `0 < λ₂`. -/
theorem coding_gain_bound (l1 l2 : ℝ) (h1 : 0 < l1) (h2 : 0 < l2) :
    Real.sqrt (l1 * l2) ≤ (l1 + l2) / 2 := by
  have h := coding_gain_sq l1 l2 h1 h2
  have hnn : 0 ≤ (Real.sqrt l1 - Real.sqrt l2) ^ 2 / 2 := by positivity
  linarith

/-- **Equality holds exactly at equal eigenvalues.** `√(λ₁λ₂) = (λ₁+λ₂)/2 ↔ λ₁ = λ₂` for
`0 < λ₁`, `0 < λ₂`. -/
theorem coding_gain_eq_iff (l1 l2 : ℝ) (h1 : 0 < l1) (h2 : 0 < l2) :
    Real.sqrt (l1 * l2) = (l1 + l2) / 2 ↔ l1 = l2 := by
  constructor
  · intro heq
    have h := coding_gain_sq l1 l2 h1 h2
    have hz : (Real.sqrt l1 - Real.sqrt l2) ^ 2 = 0 := by
      have hzero : (l1 + l2) / 2 - Real.sqrt (l1 * l2) = 0 := by linarith
      rw [h] at hzero
      linarith
    have hs : Real.sqrt l1 = Real.sqrt l2 := by
      have hmul : (Real.sqrt l1 - Real.sqrt l2) * (Real.sqrt l1 - Real.sqrt l2) = 0 := by
        simpa [sq] using hz
      exact sub_eq_zero.mp (mul_self_eq_zero.mp hmul)
    calc l1 = Real.sqrt l1 ^ 2 := (Real.sq_sqrt (le_of_lt h1)).symm
      _ = Real.sqrt l2 ^ 2 := by rw [hs]
      _ = l2 := Real.sq_sqrt (le_of_lt h2)
  · intro heq
    subst heq
    rw [← sq, Real.sqrt_sq (le_of_lt h1)]
    ring

/-- **The eigenbasis coding gain is at least one.** Reading `G = (mean λ)/√(λ₁λ₂)`; the inequality is
`proved`, the gain interpretation is `model`. -/
theorem coding_gain_ge_one (l1 l2 : ℝ) (h1 : 0 < l1) (h2 : 0 < l2) :
    1 ≤ ((l1 + l2) / 2) / Real.sqrt (l1 * l2) := by
  have hs : 0 < Real.sqrt (l1 * l2) := Real.sqrt_pos.2 (mul_pos h1 h2)
  rw [one_le_div hs]
  exact coding_gain_bound l1 l2 h1 h2

/-! ### The reflection-register stage is orthogonal

`Perf/QuantumRotation.lean` composes one `128`-group rotation of seven such stages per (layer, site).
This is the algebra of that family; the receipt's verdicts are about the learned angles. -/

/-- The 2×2 reflection stage `M t = [[cos t, sin t], [sin t, −cos t]]`. -/
noncomputable def regStage (t : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos t, Real.sin t; Real.sin t, -Real.cos t]

/-- **`M t` is orthogonal for every `t`:** `Mᵀ M = 1`. -/
theorem register_stage_orthogonal (t : ℝ) : Matrix.transpose (regStage t) * regStage t = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [regStage, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_two] <;>
    nlinarith [Real.sin_sq_add_cos_sq t]

/-- **At `t = π/4` the stage is the fixed Hadamard**, scaled by `√2/2 = 1/√2`:
`M (π/4) = (√2/2)·[[1,1],[1,−1]]`. -/
theorem register_stage_pi_div_four :
    regStage (Real.pi / 4) = (Real.sqrt 2 / 2) • !![(1 : ℝ), 1; 1, -1] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [regStage, Real.cos_pi_div_four, Real.sin_pi_div_four, smul_eq_mul]

end Elliptic

end Perf
