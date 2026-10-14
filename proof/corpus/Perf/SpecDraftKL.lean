import Perf.Pinsker
import Perf.Receipts.Onebit
import Nous.Model.SpeculativeAccept

/-!
# Speculative-decoding acceptance from the draft's KL

A draft `q` (positive on the vocabulary `Fin n`) proposes tokens for a target `p`. One speculative step
(Leviathan, Kalman and Matias 2022, arXiv 2211.17192; `Nous.Model.specStep_eq_target`) accepts with
probability `∑ min (p x) (q x) = 1 - TV(p, q)` (`Nous.Model.accept_rate`).

- `accept_ge`: `∑ min (p x) (q x) ≥ 1 - √(KL(p ‖ q) / 2)` (Pinsker, `Perf.Pinsker.tv_sq_le`).
- `mean_accept_ge`: for position weights `w` the mean acceptance is at least
  `1 - √(∑ w_t KL_t / 2)`, so the mean held-out KL of a receipt bounds the mean acceptance (Jensen,
  by Cauchy–Schwarz).
- Receipt instances (`measured` KL, `proved` bound): `q4km_accept` (`> 9/10`), `resq_accept` (`> 19/100`).

The bound is a guarantee, not the acceptance rate: the rate is `1 - TV`, measured from logits. The
receipt KL is the mean over held-out positions of `KL(teacher ‖ student)`.
-/

namespace Perf.SpecDraftKL

open Finset Perf.KLTilt

variable {n : ℕ}

/-- **Acceptance lower bound.** `1 - √(KL(p ‖ q) / 2) ≤ ∑ min (p x) (q x)`. -/
theorem accept_ge (p q : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i = 1) :
    1 - Real.sqrt (kl p q / 2) ≤ ∑ x, min (p x) (q x) := by
  rw [Nous.Model.accept_rate p q hps hqs]
  have h := Real.abs_le_sqrt (Perf.Pinsker.tv_sq_le p q hp hq hps hqs)
  have := le_abs_self ((1 / 2 : ℝ) * ∑ i, |p i - q i|)
  linarith

/-- `KL(p ‖ q) ≥ 0` on probability vectors with `q > 0` (consequence of Pinsker). -/
theorem kl_nonneg' (p q : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 < q i)
    (hps : ∑ i, p i = 1) (hqs : ∑ i, q i = 1) : 0 ≤ kl p q := by
  have := Perf.Pinsker.pinsker p q hp hq hps hqs
  nlinarith [sq_nonneg (∑ i, |p i - q i|)]

/-- **Mean acceptance.** Over `m` positions with weights `w` (`w ≥ 0`, `∑ w = 1`), the mean acceptance
is at least `1 - √((∑ w_t KL_t) / 2)`. -/
theorem mean_accept_ge {m : ℕ} (w : Fin m → ℝ) (hw0 : ∀ t, 0 ≤ w t) (hw : ∑ t, w t = 1)
    (p q : Fin m → Fin n → ℝ) (hp : ∀ t i, 0 ≤ p t i) (hq : ∀ t i, 0 < q t i)
    (hps : ∀ t, ∑ i, p t i = 1) (hqs : ∀ t, ∑ i, q t i = 1) :
    1 - Real.sqrt ((∑ t, w t * kl (p t) (q t)) / 2) ≤
      ∑ t, w t * ∑ x, min (p t x) (q t x) := by
  set s : Fin m → ℝ := fun t => Real.sqrt (kl (p t) (q t) / 2) with hs
  have hk0 : ∀ t, 0 ≤ kl (p t) (q t) := fun t => kl_nonneg' _ _ (hp t) (hq t) (hps t) (hqs t)
  have hs2 : ∀ t, s t ^ 2 = kl (p t) (q t) / 2 := fun t => by
    simp only [hs]; exact Real.sq_sqrt (by have := hk0 t; positivity)
  have hs0 : ∀ t, 0 ≤ s t := fun t => Real.sqrt_nonneg _
  have hacc : ∀ t, 1 - s t ≤ ∑ x, min (p t x) (q t x) := fun t =>
    accept_ge _ _ (hp t) (hq t) (hps t) (hqs t)
  have hsum : ∑ t, w t * (1 - s t) ≤ ∑ t, w t * ∑ x, min (p t x) (q t x) :=
    sum_le_sum (fun t _ => mul_le_mul_of_nonneg_left (hacc t) (hw0 t))
  have hexp : ∑ t, w t * (1 - s t) = 1 - ∑ t, w t * s t := by
    simp only [mul_sub, mul_one, sum_sub_distrib, hw]
  have hCS := Finset.sum_mul_sq_le_sq_mul_sq univ (fun t => Real.sqrt (w t))
    (fun t => Real.sqrt (w t) * s t)
  have e1 : ∀ t, Real.sqrt (w t) * (Real.sqrt (w t) * s t) = w t * s t := fun t => by
    rw [← mul_assoc, Real.mul_self_sqrt (hw0 t)]
  have e2 : ∀ t, Real.sqrt (w t) ^ 2 = w t := fun t => Real.sq_sqrt (hw0 t)
  have e3 : ∀ t, (Real.sqrt (w t) * s t) ^ 2 = w t * (kl (p t) (q t) / 2) := fun t => by
    rw [mul_pow, e2, hs2]
  simp only [e1, e2, e3, hw, one_mul] at hCS
  have hsm : ∑ t, w t * (kl (p t) (q t) / 2) = (∑ t, w t * kl (p t) (q t)) / 2 := by
    rw [sum_div]; exact sum_congr rfl (fun t _ => by ring)
  rw [hsm] at hCS
  have hnn : 0 ≤ ∑ t, w t * s t := sum_nonneg (fun t _ => mul_nonneg (hw0 t) (hs0 t))
  have hbd : ∑ t, w t * s t ≤ Real.sqrt ((∑ t, w t * kl (p t) (q t)) / 2) := by
    have := Real.abs_le_sqrt hCS
    rwa [abs_of_nonneg hnn] at this
  linarith

/-- Receipt instance: Q4_K_M's held-out KL guarantees mean acceptance above `9/10`. -/
theorem q4km_accept :
    (9 / 10 : ℝ) < 1 - Real.sqrt (((Receipts.w18_outcomes.q4kmKl : ℚ) : ℝ) / 2) := by
  have hk : ((Receipts.w18_outcomes.q4kmKl : ℚ) : ℝ) = 19728755950927733 / 1000000000000000000 := by
    unfold Receipts.w18_outcomes.q4kmKl; push_cast; norm_num
  rw [hk]
  have : Real.sqrt (19728755950927733 / 1000000000000000000 / 2) < 1 / 10 := by
    rw [Real.sqrt_lt' (by norm_num)]; norm_num
  linarith

/-- Receipt instance: the one-bit arm's held-out KL guarantees mean acceptance above `19/100`
(and the bound is below `1/5`, so the guarantee is the bound, not a rate). -/
theorem resq_accept :
    (19 / 100 : ℝ) < 1 - Real.sqrt (((Receipts.w18_outcomes.resqKl : ℚ) : ℝ) / 2) ∧
    1 - Real.sqrt (((Receipts.w18_outcomes.resqKl : ℚ) : ℝ) / 2) < 1 / 5 := by
  have hk : ((Receipts.w18_outcomes.resqKl : ℚ) : ℝ) = 645226614177227 / 500000000000000 := by
    unfold Receipts.w18_outcomes.resqKl; push_cast; norm_num
  rw [hk]
  constructor
  · have : Real.sqrt (645226614177227 / 500000000000000 / 2) < 81 / 100 := by
      rw [Real.sqrt_lt' (by norm_num)]; norm_num
    linarith
  · have : (4 / 5 : ℝ) < Real.sqrt (645226614177227 / 500000000000000 / 2) := by
      rw [Real.lt_sqrt (by norm_num)]; norm_num
    linarith

/-- Non-vacuity: `p = (1/2, 1/2)`, `q = (1/4, 3/4)` satisfy the hypotheses of `accept_ge`. -/
theorem witness_accept :
    1 - Real.sqrt (kl (n := 2) (fun _ => 1 / 2) (![1 / 4, 3 / 4]) / 2) ≤
      ∑ x, min ((fun _ : Fin 2 => (1 / 2 : ℝ)) x) ((![1 / 4, 3 / 4] : Fin 2 → ℝ) x) := by
  refine accept_ge _ _ (fun _ => by norm_num) ?_ ?_ ?_
  · intro i; fin_cases i <;> norm_num
  · norm_num [Fin.sum_univ_two]
  · norm_num [Fin.sum_univ_two]

end Perf.SpecDraftKL
