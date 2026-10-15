import Perf.KLTilt
import Perf.Receipts.Onebit
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Perplexity and Kullback–Leibler divergence

Finite vocabulary `Fin n`, `[NeZero n]`. For a data distribution `d`, a reference conditional `p` (the teacher)
and a student conditional `q`:

- `xent d q = −∑ d log q` is the cross-entropy and `ppl d q = exp (xent d q)` the perplexity.
- `xent_sub`: `xent d q − xent d p = ∑ d (log p − log q)`.
- `decomp`: `∑ d (log p − log q) = KL(p ‖ q) + cross d p q`, with `cross d p q = ∑ (d − p)(log p − log q)`.
- `log_ppl_ratio`: `log (ppl d q / ppl d p) = KL(p ‖ q) + cross d p q`.
- `cross_calibrated`: `cross p p q = 0`, so on teacher-distributed data the log perplexity ratio is exactly
  `KL(p ‖ q)` (`log_ppl_ratio_calibrated`), and `kl_nonneg` (Gibbs) gives `ppl p q ≥ ppl p p`.
- `xent_onehot`, `token_ppl_identity`: for a token sequence `y` the true-label perplexity
  `exp (mean −log q_t(y_t))` satisfies `log (ratio) = (∑ KL_t + ∑ cross_t) / m`.

The cross term carries everything that makes real text differ from the teacher's own distribution: it is
the only place where the data enters, and it is linear in `d − p`.

Standard names: cross-entropy, perplexity (Jelinek, Mercer, Bahl, Baker 1977), Kullback–Leibler divergence
(Kullback and Leibler 1951), Gibbs' inequality (Gibbs 1902).

Receipt link (`measured`, `Receipts.w18_outcomes`): `q4km_gap` and `resq_gap` bound
`KL_heldout − log (ppl_student / ppl_teacher)` on the C4 true-label set by interval, and `resq_kl_ratio`
bounds the ratio of the one-bit arm's held-out KL to Q4_K_M's.
-/

set_option linter.unusedSectionVars false

namespace Perf.PerplexityKL

open Finset Perf.KLTilt

variable {n : ℕ} [NeZero n]

/-- Cross-entropy of the data distribution `d` under the model conditional `q`. -/
noncomputable def xent (d q : Fin n → ℝ) : ℝ := ∑ i, d i * (-Real.log (q i))

/-- Perplexity of `q` on `d`. -/
noncomputable def ppl (d q : Fin n → ℝ) : ℝ := Real.exp (xent d q)

/-- Data-versus-reference mismatch term. -/
noncomputable def cross (d p q : Fin n → ℝ) : ℝ := ∑ i, (d i - p i) * (Real.log (p i) - Real.log (q i))

theorem xent_sub (d p q : Fin n → ℝ) :
    xent d q - xent d p = ∑ i, d i * (Real.log (p i) - Real.log (q i)) := by
  unfold xent
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

theorem decomp (d p q : Fin n → ℝ) :
    ∑ i, d i * (Real.log (p i) - Real.log (q i)) = kl p q + cross d p q := by
  unfold kl cross
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

theorem xent_sub_eq (d p q : Fin n → ℝ) : xent d q - xent d p = kl p q + cross d p q := by
  rw [xent_sub, decomp]

theorem log_ppl_ratio (d p q : Fin n → ℝ) :
    Real.log (ppl d q / ppl d p) = kl p q + cross d p q := by
  unfold ppl
  rw [Real.log_div (Real.exp_pos _).ne' (Real.exp_pos _).ne', Real.log_exp, Real.log_exp, xent_sub_eq]

theorem cross_calibrated (p q : Fin n → ℝ) : cross p p q = 0 := by
  unfold cross
  simp

theorem log_ppl_ratio_calibrated (p q : Fin n → ℝ) :
    Real.log (ppl p q / ppl p p) = kl p q := by
  rw [log_ppl_ratio, cross_calibrated, add_zero]

/-- Gibbs' inequality for positive probability vectors. -/
theorem kl_nonneg (p q : Fin n → ℝ) (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i = 1) : 0 ≤ kl p q := by
  have h : ∀ i, p i - q i ≤ p i * (Real.log (p i) - Real.log (q i)) := by
    intro i
    have hx : 0 < p i / q i := div_pos (hp i) (hq i)
    have h1 := Real.one_sub_inv_le_log_of_pos hx
    rw [Real.log_div (hp i).ne' (hq i).ne', inv_div] at h1
    have h2 := mul_le_mul_of_nonneg_left h1 (hp i).le
    have hne : p i ≠ 0 := (hp i).ne'
    have h3 : p i * (1 - q i / p i) = p i - q i := by
      rw [mul_sub, mul_one, mul_div_cancel₀ _ hne]
    linarith
  have hs : ∑ i, (p i - q i) ≤ ∑ i, p i * (Real.log (p i) - Real.log (q i)) :=
    Finset.sum_le_sum fun i _ => h i
  rw [Finset.sum_sub_distrib, hps, hqs] at hs
  unfold kl
  linarith

theorem one_le_ppl_ratio_calibrated (p q : Fin n → ℝ) (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i = 1) : 1 ≤ ppl p q / ppl p p := by
  have h := log_ppl_ratio_calibrated p q
  have hk := kl_nonneg p q hp hq hps hqs
  have hpos : 0 < ppl p q / ppl p p := div_pos (Real.exp_pos _) (Real.exp_pos _)
  by_contra hlt
  have hlt := not_le.mp hlt
  have := Real.log_neg hpos hlt
  linarith

/-- One-hot data: the true-label negative log-likelihood. -/
theorem xent_onehot (y : Fin n) (q : Fin n → ℝ) : xent (Pi.single y 1) q = -Real.log (q y) := by
  unfold xent
  simp [Pi.single_apply]

/-- Mean true-label perplexity over `m` positions. -/
noncomputable def pplTok {m : ℕ} (q : Fin m → Fin n → ℝ) (y : Fin m → Fin n) : ℝ :=
  Real.exp ((∑ t, -Real.log (q t (y t))) / m)

/-- Token-level identity: the log perplexity ratio on a token sequence is the mean of the per-position
`KL(p_t ‖ q_t)` plus the mean cross term of the one-hot data against the teacher. -/
theorem token_ppl_identity {m : ℕ} (y : Fin m → Fin n) (p q : Fin m → Fin n → ℝ) :
    Real.log (pplTok q y / pplTok p y) =
      ((∑ t, kl (p t) (q t)) + ∑ t, cross (Pi.single (y t) 1) (p t) (q t)) / m := by
  unfold pplTok
  rw [Real.log_div (Real.exp_pos _).ne' (Real.exp_pos _).ne', Real.log_exp, Real.log_exp,
    ← sub_div, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  congr 1
  refine Finset.sum_congr rfl fun t _ => ?_
  have h := xent_sub_eq (Pi.single (y t) 1) (p t) (q t)
  rw [xent_onehot, xent_onehot] at h
  linarith

/-! ## Receipt link -/

lemma log_bounds (x : ℝ) (hx : 0 < x) : 1 - x⁻¹ ≤ Real.log x ∧ Real.log x ≤ x - 1 :=
  ⟨Real.one_sub_inv_le_log_of_pos hx, Real.log_le_sub_one_of_pos hx⟩

/-- `KL_heldout − log (ppl_student / ppl_teacher)` for Q4_K_M on the C4 true-label set lies in
`(1/1000, 2/1000)`: the cross term is at most `2/1000` in magnitude, `1/10` of the KL. -/
theorem q4km_gap :
    (1 / 1000 : ℝ) < ((Receipts.w18_outcomes.q4kmKl : ℚ) : ℝ) -
        Real.log (((Receipts.w18_outcomes.q4kmC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ)) : ℝ) ∧
    ((Receipts.w18_outcomes.q4kmKl : ℚ) : ℝ) -
        Real.log (((Receipts.w18_outcomes.q4kmC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ)) : ℝ) < 2 / 1000 := by
  have hlo : ((10184 / 10000 : ℚ) : ℝ) <
      ((Receipts.w18_outcomes.q4kmC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ) : ℝ) :=
    Rat.cast_lt.mpr (by unfold Receipts.w18_outcomes.q4kmC4Ppl Receipts.w18_outcomes.teacherC4Ppl; norm_num)
  have hhi : ((Receipts.w18_outcomes.q4kmC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ) : ℝ) <
      ((10185 / 10000 : ℚ) : ℝ) :=
    Rat.cast_lt.mpr (by unfold Receipts.w18_outcomes.q4kmC4Ppl Receipts.w18_outcomes.teacherC4Ppl; norm_num)
  have hk : ((Receipts.w18_outcomes.q4kmKl : ℚ) : ℝ) = 19728755950927733 / 1000000000000000000 := by
    unfold Receipts.w18_outcomes.q4kmKl; push_cast; norm_num
  generalize ((Receipts.w18_outcomes.q4kmC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ) : ℝ) = R at *
  push_cast at hlo hhi
  have hR : 0 < R := lt_trans (by norm_num) hlo
  obtain ⟨h1, h2⟩ := log_bounds R hR
  have hinv : R⁻¹ < (10000 / 10184 : ℝ) := by
    rw [inv_lt_comm₀ hR (by norm_num)]
    linarith
  rw [hk]
  constructor <;> linarith

/-- `KL_heldout − log (ppl_student / ppl_teacher)` for the `-resq` one-bit arm on the C4 true-label set lies in
`(2/100, 37/1000)`. -/
theorem resq_gap :
    (2 / 100 : ℝ) < ((Receipts.w18_outcomes.resqKl : ℚ) : ℝ) -
        Real.log (((Receipts.w18_outcomes.resqC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ)) : ℝ) ∧
    ((Receipts.w18_outcomes.resqKl : ℚ) : ℝ) -
        Real.log (((Receipts.w18_outcomes.resqC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ)) : ℝ) < 37 / 1000 := by
  have hlo : ((3532 / 1000 : ℚ) : ℝ) <
      ((Receipts.w18_outcomes.resqC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ) : ℝ) :=
    Rat.cast_lt.mpr (by unfold Receipts.w18_outcomes.resqC4Ppl Receipts.w18_outcomes.teacherC4Ppl; norm_num)
  have hhi : ((Receipts.w18_outcomes.resqC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ) : ℝ) <
      ((3536 / 1000 : ℚ) : ℝ) :=
    Rat.cast_lt.mpr (by unfold Receipts.w18_outcomes.resqC4Ppl Receipts.w18_outcomes.teacherC4Ppl; norm_num)
  have hk : ((Receipts.w18_outcomes.resqKl : ℚ) : ℝ) = 645226614177227 / 500000000000000 := by
    unfold Receipts.w18_outcomes.resqKl; push_cast; norm_num
  generalize ((Receipts.w18_outcomes.resqC4Ppl / Receipts.w18_outcomes.teacherC4Ppl : ℚ) : ℝ) = R at *
  push_cast at hlo hhi
  have hR : 0 < R := lt_trans (by norm_num) hlo
  have hr : 0 < R / 4 := by positivity
  obtain ⟨h1, h2⟩ := log_bounds (R / 4) hr
  have hlog4 : Real.log R = 2 * Real.log 2 + Real.log (R / 4) := by
    have : R = 2 ^ 2 * (R / 4) := by ring
    conv_lhs => rw [this]
    rw [Real.log_mul (by norm_num) hr.ne', Real.log_pow]
    norm_num
  have hinv : (R / 4)⁻¹ < (1000 / 883 : ℝ) := by
    rw [inv_lt_comm₀ hr (by norm_num)]
    linarith
  have l2a := Real.log_two_gt_d9
  have l2b := Real.log_two_lt_d9
  rw [hk]
  constructor <;> linarith

/-- The one-bit arm's held-out KL is between `65` and `66` times Q4_K_M's. -/
theorem resq_kl_ratio :
    65 < Receipts.w18_outcomes.resqKl / Receipts.w18_outcomes.q4kmKl ∧
    Receipts.w18_outcomes.resqKl / Receipts.w18_outcomes.q4kmKl < 66 := by
  unfold Receipts.w18_outcomes.resqKl Receipts.w18_outcomes.q4kmKl
  constructor <;> norm_num

/-- Non-vacuity of the calibrated identity: `n = 2`, `p = (1/2, 1/2)`, `q = (1/4, 3/4)`. -/
theorem witness_calibrated :
    kl (n := 2) (fun _ => 1 / 2) (![1 / 4, 3 / 4]) =
      (1 / 2) * (Real.log (1 / 2) - Real.log (1 / 4)) + (1 / 2) * (Real.log (1 / 2) - Real.log (3 / 4)) := by
  unfold kl
  simp [Fin.sum_univ_two]

end Perf.PerplexityKL
