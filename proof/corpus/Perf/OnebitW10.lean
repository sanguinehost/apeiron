import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Star.Real
import Mathlib.LinearAlgebra.Matrix.PosDef

/-!
# ONEBIT W10: the degrees of freedom of a layer on the inputs it sees

A layer `x ↦ W x` is evaluated on calibration inputs `X` (one row per token). Its function on those inputs is
`X *ᵥ w` for each weight row `w`. This module states, over `ℝ` and the Gram `H = XᵀX`, which part of `w` the
function reads:

- `output_error_eq_quadratic`: the squared output error of a residual `r` is the quadratic form `rᵀ H r`.
  `Sanguine.MLPrimitivesBridge.output_error_is_function_distortion` is the diagonal-`H` case.
- `invisible_iff`: the output error of `r` is zero exactly when `H r = 0`.
- `same_function_iff`: two weight rows give the same outputs on every calibration input exactly when their
  difference lies in the kernel of `H`.

Grade `proved` for the three statements over `ℝ`. A statement about the spectrum of a measured `H` is a
`measured` input recorded in the receipt, not a theorem here.
-/

namespace Perf

namespace OnebitW10

open Matrix

variable {T n : ℕ}

/-- The squared output error of the residual `r` is the quadratic form of the Gram. -/
theorem output_error_eq_quadratic (X : Matrix (Fin T) (Fin n) ℝ) (r : Fin n → ℝ) :
    (X *ᵥ r) ⬝ᵥ (X *ᵥ r) = r ⬝ᵥ ((Xᵀ * X) *ᵥ r) := by
  rw [← Matrix.mulVec_mulVec]
  symm
  rw [Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]

/-- The output error of `r` is zero exactly when the Gram annihilates `r`. -/
theorem invisible_iff (X : Matrix (Fin T) (Fin n) ℝ) (r : Fin n → ℝ) :
    (X *ᵥ r) ⬝ᵥ (X *ᵥ r) = 0 ↔ (Xᵀ * X) *ᵥ r = 0 := by
  constructor
  · intro h
    have hx : X *ᵥ r = 0 := dotProduct_self_eq_zero.mp h
    rw [← Matrix.mulVec_mulVec, hx, Matrix.mulVec_zero]
  · intro h
    rw [output_error_eq_quadratic, h, dotProduct_zero]

/-- Two weight rows compute the same outputs on every calibration input exactly when their difference
lies in the kernel of the Gram. -/
theorem same_function_iff (X : Matrix (Fin T) (Fin n) ℝ) (w₁ w₂ : Fin n → ℝ) :
    X *ᵥ w₁ = X *ᵥ w₂ ↔ (Xᵀ * X) *ᵥ (w₁ - w₂) = 0 := by
  rw [← invisible_iff, Matrix.mulVec_sub, dotProduct_self_eq_zero, sub_eq_zero]

/-! ### Verdict: the measured spectra (`measured`, not derived)

`/home/socol/oc-tmp/escalation/2026-10-11-dof-spectra.json`: Qwen2.5-0.5B f16 weights, C4 Grams of 51200 tokens,
`r99` the number of eigenvalues that carry 99% of the trace, of `H` and of `W H Wᵀ`. The prediction was
`ffn_down` `r99(H) < n/4` and `r99(WHWᵀ) < min(m,n)/2` at blocks 3, 11, 23, and `attn_in` `r99(H) < n/2`. It is
refuted: each clause below is a numeric fact about the receipt, and the failures are the statements that hold. -/

/-- `ffn_down` input width. -/
def ffnN : ℕ := 4864

/-- `ffn_down` `r99(H)` at blocks 3, 11, 23. -/
def ffnR99 : ℕ × ℕ × ℕ := (1037, 4341, 2201)

/-- `ffn_down` `r99(W H Wᵀ)` at blocks 3, 11, 23, against `min(m,n) = 896`. -/
def ffnF99 : ℕ × ℕ × ℕ := (99, 770, 441)

/-- `attn_in` `r99(H)` at blocks 3, 11, 23, against `n = 896`. -/
def attnR99 : ℕ × ℕ × ℕ := (752, 640, 591)

theorem ffn_block3_below_quarter : 4 * ffnR99.1 < ffnN := by decide

theorem ffn_block11_not_below_quarter : ¬ 4 * ffnR99.2.1 < ffnN := by decide

theorem ffn_block23_not_below_quarter : ¬ 4 * ffnR99.2.2 < ffnN := by decide

theorem ffn_function_block11_not_below_half : ¬ 2 * ffnF99.2.1 < 896 := by decide

theorem attn_in_none_below_half :
    ¬ 2 * attnR99.1 < 896 ∧ ¬ 2 * attnR99.2.1 < 896 ∧ ¬ 2 * attnR99.2.2 < 896 := by decide

end OnebitW10

end Perf
