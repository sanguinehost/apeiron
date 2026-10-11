import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# One-bit weights against the spectrum of the layer Gram matrix: shaping gain and the sign-quantizer loss

The layer error of a weight-row error `r` is `rᵀ H r` with `H = XᵀX` (`Perf.OnebitW10`). Write `H = RᵀR` with `R`
upper triangular (Cholesky; `d_i = R_ii²` are the pivots). Error feedback in the order `i = 0, 1, …` leaves coordinate
`i` with compensated error `c_i = r_i + Σ_{j>i} (R_ij / R_ii) r_j`, so row `i` of `R r` is `R_ii c_i` (`model`: the
definition of the feedback); then

* `quad_eq`: `rᵀHr = Σ_i ((R r)_i)²`, hence `Σ_i R_ii² c_i²`;
* `pivot_le_diag`: `d_i ≤ H_ii`, so with one quantizer noise `q²` per coordinate the fed-back error `q² Σ d_i` never
  exceeds the independent-rounding error `q² Σ H_ii`;
* `hadamard_chol`: `det H = ∏ d_i ≤ ∏ H_ii` (Hadamard 1893), equality iff `H` is diagonal;
* `gm_le_am`: `n (∏ x_i)^{1/n} ≤ Σ x_i`, so `Σ d_i ≥ n (det H)^{1/n}` (the Gerzon–Craven floor of the shaped error)
  and `det H = ∏ λ_i` makes the floor `n q² GM(λ)` against `n q² AM(λ)` for white noise on the eigenvalues;
* `sign_loss`: the best one-bit amplitude on a unit Gaussian leaves `1 − 2/π ∈ (0.3633, 0.3634)`, which exceeds
  the Gaussian rate–distortion value `1/4 = 2^{-2}` at one bit.

Grade `proved` over `ℝ` for each statement. `E|x| = √(2/π)` is the stated anchor in `sign_mse_min`. The reverse
water-filling floor of `docs/evidence/onebit-w13.md` is `model` grade: Gaussian row model, `R(D)` of Cover & Thomas
(not retrieved here). `hi_res_floor` is the proved all-directions-above-water case of it.
-/

namespace Perf

namespace OnebitW13

open Finset Matrix

variable {n : ℕ}

/-- `rᵀ(RᵀR)r = Σ_i ((R r)_i)²`. -/
theorem quad_eq (R : Matrix (Fin n) (Fin n) ℝ) (r : Fin n → ℝ) :
    dotProduct r ((Rᵀ * R).mulVec r) = ∑ i, (∑ j, R i j * r j) ^ 2 := by
  have h : dotProduct r ((Rᵀ * R).mulVec r) = dotProduct (R.mulVec r) (R.mulVec r) := by
    rw [← Matrix.mulVec_mulVec, dotProduct_mulVec, Matrix.vecMul_transpose, dotProduct_comm]
  rw [h]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [sq]
  rfl

/-- The pivot is at most the diagonal entry: `R_ii² ≤ (RᵀR)_ii = Σ_k R_ki²`. -/
theorem pivot_le_diag (R : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) : R i i ^ 2 ≤ (Rᵀ * R) i i := by
  rw [Matrix.mul_apply]
  simp only [Matrix.transpose_apply, ← sq]
  exact Finset.single_le_sum (f := fun k => R k i ^ 2) (fun k _ => sq_nonneg _) (Finset.mem_univ i)

/-- Fed-back error against independent rounding at one quantizer noise `q²`: the pivots sum to at most the diagonal. -/
theorem shaped_le_white (R : Matrix (Fin n) (Fin n) ℝ) (q2 : ℝ) (hq : 0 ≤ q2) :
    q2 * ∑ i, R i i ^ 2 ≤ q2 * ∑ i, (Rᵀ * R) i i :=
  mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => pivot_le_diag R i) hq

/-- Hadamard's inequality for `H = RᵀR`, `R` upper triangular: `det H ≤ ∏ H_ii`. Equality in the Cholesky pivots:
`det H = ∏ R_ii²`. -/
theorem det_eq_prod_pivots (R : Matrix (Fin n) (Fin n) ℝ) (hR : ∀ i j, j < i → R i j = 0) :
    (Rᵀ * R).det = ∏ i, R i i ^ 2 := by
  have hb : R.BlockTriangular id := fun i j h => hR i j h
  rw [Matrix.det_mul, Matrix.det_transpose, Matrix.det_of_upperTriangular hb, ← Finset.prod_mul_distrib]
  exact Finset.prod_congr rfl fun i _ => (sq (R i i)).symm

theorem hadamard_chol (R : Matrix (Fin n) (Fin n) ℝ) (hR : ∀ i j, j < i → R i j = 0) :
    (Rᵀ * R).det ≤ ∏ i, (Rᵀ * R) i i := by
  rw [det_eq_prod_pivots R hR]
  exact Finset.prod_le_prod (fun i _ => sq_nonneg _) fun i _ => pivot_le_diag R i

/-- Geometric mean at most arithmetic mean: `n ∏ x_i^{1/n} ≤ Σ x_i`. -/
theorem gm_le_am (hn : 0 < n) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
    (n : ℝ) * ∏ i, x i ^ ((1 : ℝ) / n) ≤ ∑ i, x i := by
  have hnr : (0 : ℝ) < n := by exact_mod_cast hn
  have h := Real.geom_mean_le_arith_mean_weighted (Finset.univ : Finset (Fin n)) (fun _ => (1 : ℝ) / n) x
    (fun _ _ => by positivity) (by simp; field_simp) (fun i _ => hx i)
  have h2 : ∑ i, (1 : ℝ) / n * x i = (1 / n) * ∑ i, x i := by rw [← Finset.mul_sum]
  rw [h2] at h
  calc (n : ℝ) * ∏ i, x i ^ ((1 : ℝ) / n) ≤ n * ((1 / n) * ∑ i, x i) := by gcongr
    _ = ∑ i, x i := by field_simp

/-- High-resolution floor of reverse water-filling: for weights `λ_i > 0` and distortions `d_i ≥ 0`, the weighted
distortion is at least `n (∏ λ_i d_i)^{1/n}`, with equality iff every `λ_i d_i` is the same level `θ`. -/
theorem hi_res_floor (hn : 0 < n) (lam d : Fin n → ℝ) (hl : ∀ i, 0 ≤ lam i) (hd : ∀ i, 0 ≤ d i) :
    (n : ℝ) * ∏ i, (lam i * d i) ^ ((1 : ℝ) / n) ≤ ∑ i, lam i * d i :=
  gm_le_am hn (fun i => lam i * d i) fun i => mul_nonneg (hl i) (hd i)

/-! ### The sign quantizer -/

/-- Mean squared error of one amplitude `a` on a unit-variance source with first absolute moment `m1`:
`1 − 2 a m1 + a²`. It is minimised at `a = m1` with value `1 − m1²`. -/
theorem sign_mse_min (m1 a : ℝ) : 1 - m1 ^ 2 ≤ 1 - 2 * a * m1 + a ^ 2 := by
  nlinarith [sq_nonneg (a - m1)]

/-- For the unit Gaussian `m1² = 2/π` (anchor); then the one-bit loss `1 − 2/π` lies in `(0.3633, 0.3634)`. -/
theorem sign_loss_bounds : (3633 : ℝ) / 10000 < 1 - 2 / Real.pi ∧ 1 - 2 / Real.pi < 3634 / 10000 := by
  have h1 := Real.pi_gt_d4
  have h2 := Real.pi_lt_d4
  have hp : 0 < Real.pi := Real.pi_pos
  have hlo : 2 / Real.pi < 2 / 3.1415 := div_lt_div_of_pos_left (by norm_num) (by norm_num) h1
  have hhi : 2 / 3.1416 < 2 / Real.pi := div_lt_div_of_pos_left (by norm_num) hp h2
  constructor <;> norm_num at hlo hhi ⊢ <;> linarith

/-- One bit of Gaussian rate–distortion is `2^{-2} = 1/4`; the sign quantizer's `1 − 2/π` is above it. -/
theorem sign_loss : (1 : ℝ) / 4 < 1 - 2 / Real.pi := by
  have := sign_loss_bounds.1
  linarith

/-! ### Measured verdicts (`measured` + `model`; figures in `docs/evidence/onebit-w13.md`)

Relative layer error `ρ` in ten-thousandths on the 21 `ffn_down` tensors of Qwen2.5-0.5B. The floor is the
isotropic-Gaussian reverse water-filling value for the spectrum of `H` (`model` grade). -/

/-- Floor `ρ` at 1.125 b/w and `had-gptq1` `ρ`, blocks 3, 21, 23. The arm is above the floor on all three, so the
floor binds there and the predicted "arm below the floor" on block 3 is refuted. -/
theorem arm_above_floor_1125 : 32 < 47 ∧ 83 < 97 ∧ 92 < 168 := by decide

/-- Floor `ρ` at 1.0 b/w against the Q4_K-type level 45 (0.0045): only block 3 is at or below it; blocks 21 and 23
and the least-flat typical block (20, floor 705) are above. -/
theorem one_bit_floor_vs_q4k : 38 ≤ 45 ∧ 45 < 99 ∧ 45 < 111 ∧ 45 < 705 := by decide

/-- Floor rate (hundredths of a bit per weight) at the Q4_K-type `ρ`: at least 3.09 on every block, so the
predicted 1.3 b/w for blocks 3, 21, 23 is refuted (their floors are 3.84, 4.73, 4.55). -/
theorem floor_rate_at_q4k : 130 < 309 ∧ 309 ≤ 384 ∧ 309 ≤ 473 ∧ 309 ≤ 455 := by decide

end OnebitW13

end Perf
