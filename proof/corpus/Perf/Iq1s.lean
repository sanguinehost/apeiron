import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Tauto

/-!
# IQ1_S: why the weighted three-level search is exact, and what it optimises

`src/learn/iq1s.rs` is the port of llama.cpp `ece963f41` `quantize_row_iq1_s_impl`; it is byte-identical to
`llama-quantize` on 357504 blocks (`examples/iq1s_check.rs`). This module states, over `ℝ`, the facts that
the quantizer rests on:

1. `rate`: the format is `25/16` bits per weight.
2. `err_eq_best_add_sq`, `refit_scalar_le`: for fixed labels the weighted error in the scale is a parabola,
   so the best scale is `Σwqx / Σwq²` and the best score is `(Σwqx)² / Σwq²` (the C `best_score`).
3. `idx_nearest`, `idx_mono`, `exists_monotone_le`: for fixed levels, the nearest-level labelling is
   monotone in `x`, so a search over splits of the sorted `x` loses nothing against all `3ⁿ` labellings.
4. `pair_discriminant_nonpos`, `pair_form_on_direction`: the layer error is the quadratic form of
   `H = XᵀX` (`Perf.OnebitW10`); on a pair of weights its discriminant `H_ij² − H_ii H_jj` is `≤ 0`, so the
   form is elliptic or parabolic and never hyperbolic. IQ1_S weights the squared error by a diagonal
   proxy of `H`; the pair discriminant measures what that proxy omits.

Grade `proved` for each statement over `ℝ`. The correspondence between the monotone labellings and the
`(i1, i2)` enumeration of the Rust loop is `tested` (byte-identity), not a theorem here.
-/

namespace Perf

namespace Iq1s

open Finset

/-- Bits of one 256-weight block: eight sub-blocks of four 11-bit codewords, a 3-bit scale and a shift bit,
plus the 16-bit `d`. -/
def blockBits : ℕ := 8 * (4 * 11 + 3 + 1) + 16

theorem rate : (blockBits : ℚ) / 256 = 25 / 16 := by
  unfold blockBits; norm_num

/-! ### The scale is a parabola -/

variable {n : ℕ}

/-- Weighted squared error of labels `q` at scale `s`. -/
def err (w q x : Fin n → ℝ) (s : ℝ) : ℝ := ∑ i, w i * (s * q i - x i) ^ 2

/-- The error is a parabola in the scale: `err(s) = s² Σwq² − 2 s Σwqx + Σwx²`. -/
theorem err_expand (w q x : Fin n → ℝ) (s : ℝ) :
    err w q x s = s ^ 2 * (∑ i, w i * q i ^ 2) - 2 * s * (∑ i, w i * q i * x i) + ∑ i, w i * x i ^ 2 := by
  unfold err
  simp only [Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

/-- If `s*` solves `s* Σwq² = Σwqx` then `err(s) = err(s*) + (s − s*)² Σwq²`. -/
theorem err_eq_best_add_sq (w q x : Fin n → ℝ) (s sstar : ℝ)
    (h : sstar * (∑ i, w i * q i ^ 2) = ∑ i, w i * q i * x i) :
    err w q x s = err w q x sstar + (s - sstar) ^ 2 * (∑ i, w i * q i ^ 2) := by
  rw [err_expand, err_expand, ← h]
  ring

/-- The solution of the normal equation is a minimiser when the weights are nonnegative. -/
theorem refit_scalar_le (w q x : Fin n → ℝ) (hw : ∀ i, 0 ≤ w i) (s sstar : ℝ)
    (h : sstar * (∑ i, w i * q i ^ 2) = ∑ i, w i * q i * x i) :
    err w q x sstar ≤ err w q x s := by
  rw [err_eq_best_add_sq w q x s sstar h]
  have : 0 ≤ ∑ i, w i * q i ^ 2 := Finset.sum_nonneg fun i _ => mul_nonneg (hw i) (sq_nonneg _)
  nlinarith [mul_nonneg (sq_nonneg (s - sstar)) this]

/-! ### Three levels: the nearest-level labelling is monotone -/

/-- The index of the nearest of three sorted levels `a < b < c`: one for each midpoint below `x`. -/
noncomputable def idx (a b c x : ℝ) : ℕ :=
  (if (a + b) / 2 < x then 1 else 0) + (if (b + c) / 2 < x then 1 else 0)

/-- The level with index `k`. -/
def lev (a b c : ℝ) : ℕ → ℝ
  | 0 => a
  | 1 => b
  | _ => c

theorem idx_le_two (a b c x : ℝ) : idx a b c x ≤ 2 := by
  unfold idx; split_ifs <;> omega

/-- The index is monotone in `x`. -/
theorem idx_mono (a b c : ℝ) {x y : ℝ} (hxy : x ≤ y) : idx a b c x ≤ idx a b c y := by
  unfold idx
  have h1 : (a + b) / 2 < x → (a + b) / 2 < y := fun h => lt_of_lt_of_le h hxy
  have h2 : (b + c) / 2 < x → (b + c) / 2 < y := fun h => lt_of_lt_of_le h hxy
  split_ifs <;> first | omega | (exfalso; tauto)

/-- The labelled level is at least as close to `x` as any other of the three. -/
theorem idx_nearest {a b c : ℝ} (hab : a < b) (hbc : b < c) (x : ℝ) {k : ℕ} (hk : k ≤ 2) :
    (x - lev a b c (idx a b c x)) ^ 2 ≤ (x - lev a b c k) ^ 2 := by
  unfold idx
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 := by omega
  by_cases h1 : (a + b) / 2 < x <;> by_cases h2 : (b + c) / 2 < x <;> simp only [h1, h2, if_true, if_false] <;>
    rcases hk' with rfl | rfl | rfl <;> simp only [lev] <;> first | linarith | nlinarith
/-- Weighted error of a labelling `f` at fixed levels. -/
noncomputable def errL (w x : Fin n → ℝ) (a b c : ℝ) (f : Fin n → ℕ) : ℝ :=
  ∑ i, w i * (x i - lev a b c (f i)) ^ 2

/-- Any labelling is matched or beaten by the nearest-level labelling, which is monotone in `x`: the search
over splits of the sorted order is exhaustive. -/
theorem exists_monotone_le {a b c : ℝ} (hab : a < b) (hbc : b < c) (w x : Fin n → ℝ) (hw : ∀ i, 0 ≤ w i)
    (f : Fin n → ℕ) (hf : ∀ i, f i ≤ 2) :
    ∃ g : Fin n → ℕ, (∀ i, g i ≤ 2) ∧ (∀ i j, x i ≤ x j → g i ≤ g j) ∧
      errL w x a b c g ≤ errL w x a b c f := by
  refine ⟨fun i => idx a b c (x i), fun i => idx_le_two _ _ _ _, fun i j h => idx_mono a b c h, ?_⟩
  exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (idx_nearest hab hbc (x i) (hf i)) (hw i)

/-! ### The layer error is a quadratic form whose pair discriminant is never positive -/

variable {T : ℕ}

/-- Gram entry `H_ij = Σ_t X_ti X_tj`. -/
def gram (X : Fin T → Fin n → ℝ) (i j : Fin n) : ℝ := ∑ t, X t i * X t j

/-- Cauchy–Schwarz: the discriminant `H_ij² − H_ii H_jj` of the pair form is at most zero, so the form is
elliptic or parabolic and never hyperbolic. -/
theorem pair_discriminant_nonpos (X : Fin T → Fin n → ℝ) (i j : Fin n) :
    gram X i j ^ 2 - gram X i i * gram X j j ≤ 0 := by
  have := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun t => X t i) (fun t => X t j)
  unfold gram
  simp only [sq] at this ⊢
  linarith

/-- The pair form `H_ii a² + 2 H_ij a b + H_jj b²` at `(a, b) = (H_ij, −H_ii)` equals `−H_ii` times the
discriminant, so the direction is invisible exactly when the discriminant is zero (given `H_ii ≠ 0`). -/
theorem pair_form_on_direction (hii hij hjj : ℝ) :
    hii * hij ^ 2 + 2 * hij * hij * (-hii) + hjj * hii ^ 2 = -hii * (hij ^ 2 - hii * hjj) := by
  ring

/-! ### Measured verdicts (`measured`; the figures are in the onebit-w12 note)

Same population P1 (the 21 `ffn_down` tensors `blk.3`..`blk.23` of Qwen2.5-0.5B, 91521024 weights, every
other tensor taken from the `IQ1_S` file), paired KLD against f16. -/

/-- Blocks compared byte-for-byte with `llama-quantize`: 21 tensors of 17024 blocks, 0 differing. -/
theorem blocks_compared : 21 * 17024 = 357504 := by norm_num

/-- 91521024 weights are 357504 blocks of 256. -/
theorem weights_are_blocks : 357504 * 256 = 91521024 := by norm_num

/-- KLD (nats) on P1, in ten-thousandths. -/
def kldIq1sP1 : ℕ := 6315
def kldHadGptq1P1 : ℕ := 8345

/-- Bits per weight in sixteenths. -/
def bpwIq1s : ℕ := 25
def bpwHadGptq1 : ℕ := 18

/-- On P1 `had-gptq1` spends fewer bits (18/16 < 25/16) and loses on KLD (0.8345 > 0.6315): the diagonal-`H`
codebook is ahead at equal population, and the gap is the next derivation's target, not a rate win. -/
theorem p1_verdict : bpwHadGptq1 < bpwIq1s ∧ kldIq1sP1 < kldHadGptq1P1 := by
  unfold bpwHadGptq1 bpwIq1s kldIq1sP1 kldHadGptq1P1; decide

end Iq1s

end Perf
