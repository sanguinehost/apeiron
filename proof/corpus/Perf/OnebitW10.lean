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

/-! ### The amplitudes: orientation fixed, the output is linear in them

With the sign pattern fixed, a row is `Φ a` for a `±1` pattern matrix `Φ` and a vector of block amplitudes `a`. On
calibration inputs `X` the output is `Z a` with `Z = X Φ`. The error is the squared distance of `y = X w` from the
column space of `Z`, so the amplitudes that solve the normal equations `Zᵀ (y - Z a*) = 0` lose no more than any
other amplitudes, and the excess of any other `a` is the squared output of `a - a*`. -/

/-- Pythagoras for the amplitude refit: the error of any `a` is the error of `a*` plus the squared output of
`a - a*`, when `a*` solves the normal equations. -/
theorem refit_pythagoras (Z : Matrix (Fin T) (Fin n) ℝ) (y : Fin T → ℝ) (a astar : Fin n → ℝ)
    (h : Zᵀ *ᵥ (y - Z *ᵥ astar) = 0) :
    (y - Z *ᵥ a) ⬝ᵥ (y - Z *ᵥ a)
      = (y - Z *ᵥ astar) ⬝ᵥ (y - Z *ᵥ astar) + (Z *ᵥ (a - astar)) ⬝ᵥ (Z *ᵥ (a - astar)) := by
  have hc : (y - Z *ᵥ astar) ⬝ᵥ (Z *ᵥ (a - astar)) = 0 := by
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, h, zero_dotProduct]
  have hd : y - Z *ᵥ a = (y - Z *ᵥ astar) - Z *ᵥ (a - astar) := by
    rw [Matrix.mulVec_sub]; abel
  rw [hd]
  generalize y - Z *ᵥ astar = u at hc ⊢
  generalize Z *ᵥ (a - astar) = v at hc ⊢
  simp only [sub_dotProduct, dotProduct_sub]
  rw [dotProduct_comm v u, hc]
  ring

/-- The refit never loses: the error at `a*` is at most the error at any `a` (in particular at the scale a
quantizer chose). -/
theorem refit_le (Z : Matrix (Fin T) (Fin n) ℝ) (y : Fin T → ℝ) (a astar : Fin n → ℝ)
    (h : Zᵀ *ᵥ (y - Z *ᵥ astar) = 0) :
    (y - Z *ᵥ astar) ⬝ᵥ (y - Z *ᵥ astar) ≤ (y - Z *ᵥ a) ⬝ᵥ (y - Z *ᵥ a) := by
  rw [refit_pythagoras Z y a astar h]
  refine le_add_of_nonneg_right ?_
  unfold dotProduct
  exact Finset.sum_nonneg fun i _ => mul_self_nonneg _

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

/-! ### Verdict: the amplitude refit (`measured`, not derived)

`/home/socol/oc-tmp/escalation/2026-10-11-amp-refit.json`: `ffn_down` of Qwen2.5-0.5B at blocks 3, 11, 23, weights
rotated by `R₁₂₈`, orientation `sign(W R)`, one amplitude per 128-block (38 per row). Errors are `eᵀ H e / wᵀ H w`
over the 896 rows, in units of `10⁻⁴`, for the scale a quantizer takes (mean `|w|` of the block) and for the joint
per-row refit. The prediction was the order `3 > 23 > 11` of the removed share, at least 15% removed at block 3, and
the scale-only error inside `[0.30, 0.45]` at all three. The order and the 15% hold; the range fails at blocks 3
and 23, so the prediction is refuted by its own falsifier. -/

/-- Scale-only error at blocks 3, 11, 23, in `10⁻⁴`. -/
def rtnErr : ℕ × ℕ × ℕ := (1247, 3164, 2455)

/-- Refit error at blocks 3, 11, 23, in `10⁻⁴`. -/
def refitErr : ℕ × ℕ × ℕ := (56, 2744, 495)

/-- The refit removes more than 95% of the error at block 3. -/
theorem block3_removes_95 : 20 * refitErr.1 < rtnErr.1 := by decide

/-- The removed share orders `3 > 23 > 11`, as the function DOF `99 < 441 < 770` of `ffnF99` does
(cross-multiplied: `e₁/e₀` is smaller where more is removed). -/
theorem removed_share_order :
    refitErr.1 * rtnErr.2.2 < refitErr.2.2 * rtnErr.1 ∧
    refitErr.2.2 * rtnErr.2.1 < refitErr.2.1 * rtnErr.2.2 := by decide

/-- The stated range for the scale-only error (`0.30 ≤ e ≤ 0.45`) fails at blocks 3 and 23. -/
theorem scale_only_range_fails :
    ¬ 3000 ≤ rtnErr.1 ∧ ¬ 3000 ≤ rtnErr.2.2 := by decide

/-! ### Verdict: the whole-model run (`measured`, not derived)

`/home/socol/oc-tmp/escalation/2026-10-11-seqamp-gptq-refit-q.json`: `tools/onebit/seqamp.py --mode gptq --rot 1
--refit 1 --path q`, Qwen2.5-0.5B, all 168 linear tensors at 128-block sign plus one amplitude, each fitted on the
inputs the quantised prefix delivers; paired KLD against the fp32 model on 32768 wikitext-2 tokens, in units of
`10⁻³` nats. The prediction was `[2.5, 6.0]` nats, refuted below `6.85` (W9 `had-gptq1`) and below `2.5`. The
control (`seqamp-control-1`: scale-only, no rotation, teacher inputs) gives `12.905` inside its stated band. The W9
figure came from a different evaluation path, so the order against it is indicative, not like-for-like. -/

/-- Whole-model KLD, in `10⁻³` nats: the run, its standard error, the instrument control, the W9 reference. -/
def runKld : ℕ := 5642
def runSem : ℕ := 22
def controlKld : ℕ := 12905
def w9Kld : ℕ := 6850

/-- The prediction holds: the run lies in the stated band. -/
theorem run_in_predicted_band : 2500 ≤ runKld ∧ runKld ≤ 6000 := by decide

/-- The control lies in its stated band `[11.0, 15.5]`, so the harness is admissible. -/
theorem control_in_band : 11000 ≤ controlKld ∧ controlKld ≤ 15500 := by decide

/-- The run is below the W9 reference by more than ten standard errors. -/
theorem run_below_w9 : runKld + 10 * runSem < w9Kld := by decide

/-- The run is still above one nat: whole-model one-bit by this route is not usable. -/
theorem run_not_usable : 1000 < runKld := by decide

end OnebitW10

end Perf
