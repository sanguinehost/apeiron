import Perf.Pinsker
import Perf.Receipts.Onebit
import Nous.Model.SpeculativeAccept
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

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

/-! ## Block length and step-time model

A speculative block drafts `k` tokens and the verifier scores all `k + 1` positions in one pass. Draft `j` is kept
when drafts `0..j-1` were kept and its own test passes; the block emits the kept prefix plus one corrected or bonus
token. With acceptance `a t` at position `t` the expected emission from a block starting at `t` is
`blockLenAt a t k = ∑_{j ≤ k} ∏_{i < j} a (t + i)` (`spec_stats` in `tools/onebit/fold_transfer.py` computes it
along held-out text); at a constant `a` it is the geometric sum `blockLen a k = (1 - a^(k+1)) / (1 - a)`
(Leviathan, Kalman and Matias 2022, arXiv 2211.17192, §3.1). -/

/-- Expected emission of one block of `k` drafts at constant acceptance `a`. -/
noncomputable def blockLen (a : ℝ) (k : ℕ) : ℝ := ∑ j ∈ range (k + 1), a ^ j

/-- Expected emission of one block of `k` drafts starting at `t`, acceptance `a i` at position `i`. -/
noncomputable def blockLenAt (a : ℕ → ℝ) (t k : ℕ) : ℝ := ∑ j ∈ range (k + 1), ∏ i ∈ range j, a (t + i)

/-- At a constant acceptance the per-position block length is the geometric one. -/
theorem blockLenAt_const (a : ℝ) (t k : ℕ) : blockLenAt (fun _ => a) t k = blockLen a k := by
  simp [blockLenAt, blockLen, prod_const, card_range]

/-- **Geometric closed form.** `(1 - a) · blockLen a k = 1 - a^(k+1)`. -/
theorem blockLen_geom (a : ℝ) (k : ℕ) : (1 - a) * blockLen a k = 1 - a ^ (k + 1) :=
  mul_neg_geom_sum a (k + 1)

/-- Block length rises with acceptance. -/
theorem blockLen_mono {a b : ℝ} (k : ℕ) (h0 : 0 ≤ a) (hab : a ≤ b) : blockLen a k ≤ blockLen b k := by
  unfold blockLen; gcongr

/-- Block length is between `1` and `k + 1` for an acceptance in `[0, 1]`. -/
theorem blockLen_bounds {a : ℝ} (k : ℕ) (h0 : 0 ≤ a) (h1 : a ≤ 1) : 1 ≤ blockLen a k ∧ blockLen a k ≤ k + 1 := by
  constructor
  · have := blockLen_mono k le_rfl h0
    simpa [blockLen, zero_pow_eq] using this
  · have := blockLen_mono k h0 h1
    simpa [blockLen] using this

/-- **When speculation pays.** A verifier pass costs `tV` whether it scores one position or `k + 1` (its weights are
read once: the bandwidth-bound regime), a draft step costs `tD`. Speculation emits `E` tokens per `tV + k tD` against
one per `tV` without it, and is faster exactly when the `E - 1` extra tokens outweigh the drafting time. -/
theorem spec_faster {E tV tD : ℝ} (k : ℕ) (hV : 0 < tV) (hD : 0 ≤ tD) :
    1 / tV < E / (tV + k * tD) ↔ k * tD < (E - 1) * tV := by
  have hk : 0 < tV + k * tD := by positivity
  rw [div_lt_div_iff₀ hV hk]
  constructor <;> intro h <;> nlinarith

/-- Witness: at `a = 1/2`, `k = 2` the block emits `7/4`. -/
theorem blockLen_witness : blockLen (1 / 2) 2 = 7 / 4 := by
  norm_num [blockLen, sum_range_succ]

/-! ## Receipt instances: the finished arm as a draft (`qreg-spec.json`)

Acceptance, block lengths and bandwidths are `measured` (`docs/evidence/receipts/onebit/qreg-spec.json`); the
arithmetic is `proved`. The 27B step times are a `model`: the 27B Q4_K_M verifier carries the 0.5B receipt's `q4kmBpw`
(5.52 bits per weight) and the measured 8B CPU pass time per weight byte (`tPass27`, `qreg-spectv.json`), the draft is
`qwen27b_target_floor`'s 3.9825 GB read at the measured device bandwidth, and the 0.5B pair's acceptance stands for the
27B pair. -/

/-- The finished arm accepts between `59/100` and `3/5` of the teacher's tokens; the Pinsker floor from the same run's
KL is below `7/20`, so the measured rate is the number, not the bound. -/
theorem spec_accept_measured :
    (59 / 100 : ℝ) < (Receipts.qreg_spec.accept : ℝ) ∧ (Receipts.qreg_spec.accept : ℝ) < 3 / 5 ∧
    1 - Real.sqrt ((Receipts.qreg_spec.klHeldoutAfter : ℝ) / 2) < 7 / 20 := by
  refine ⟨by norm_num [Receipts.qreg_spec.accept], by norm_num [Receipts.qreg_spec.accept], ?_⟩
  have : (13 / 20 : ℝ) < Real.sqrt ((Receipts.qreg_spec.klHeldoutAfter : ℝ) / 2) := by
    rw [Real.lt_sqrt (by norm_num)]; norm_num [Receipts.qreg_spec.klHeldoutAfter]
  linarith

/-- **Acceptances cluster along the text.** At `k = 4` and `k = 8` the teacher-forced block length exceeds the
geometric one at the mean acceptance, so rejections bunch together and the iid formula undercounts. -/
theorem spec_runs_cluster :
    blockLen (Receipts.qreg_spec.accept : ℝ) 4 < (Receipts.qreg_spec.eK4 : ℝ) ∧
    blockLen (Receipts.qreg_spec.accept : ℝ) 8 < (Receipts.qreg_spec.eK8 : ℝ) := by
  constructor <;> norm_num [blockLen, sum_range_succ, Receipts.qreg_spec.accept, Receipts.qreg_spec.eK4,
    Receipts.qreg_spec.eK8]

/-- `model`: one 27B draft step at 1.18 bits per weight (3.9825 GB, `qwen27b_target_floor`) read from VRAM, seconds. -/
noncomputable def tDraft27 : ℝ := 3982500000 / ((Receipts.qreg_spec.deviceReadGBs : ℝ) * 10 ^ 9)

/-! ## Training the acceptance directly: the TV objective's logit adjoint

`TVHead` in `tools/onebit/pv_finish.py` (`--objective tv`) minimises `∑_t TV(p_t, q_t) = ∑_t (1 - accept_t)` with
`q = softmax z`. Away from ties `q x = p x`, `TV = (1/2) ∑ s x (q x - p x)` with `s = sign (q - p)` locally constant, so
its derivative along the logit `z y` is half the derivative of the `q`-weighted mean `∑ s x q x`. That derivative is
`q y (s y - ∑ s x q x)` (the softmax Jacobian applied to `s`), which `TVHead` forms in its forward pass. -/

/-- **Softmax adjoint.** Along logit `y`, `t ↦ ∑ s x softmax(z + t e_y) x` has derivative `q y (s y - ⟨s, q⟩)` at `0`,
`q = softmax z`. -/
theorem softmax_adjoint (z s : Fin n → ℝ) (y : Fin n) :
    HasDerivAt (fun t : ℝ => (∑ x, s x * Real.exp (z x + t * (if x = y then 1 else 0))) /
        ∑ x, Real.exp (z x + t * (if x = y then 1 else 0)))
      (Real.exp (z y) / (∑ x, Real.exp (z x)) *
        (s y - (∑ x, s x * Real.exp (z x)) / ∑ x, Real.exp (z x))) 0 := by
  have hD : 0 < ∑ x, Real.exp (z x) := sum_pos (fun _ _ => Real.exp_pos _) ⟨y, mem_univ y⟩
  have hd : ∀ x, HasDerivAt (fun t : ℝ => Real.exp (z x + t * (if x = y then 1 else 0)))
      (Real.exp (z x + 0 * (if x = y then 1 else 0)) * (1 * (if x = y then 1 else 0))) 0 :=
    fun x => (((hasDerivAt_id (0 : ℝ)).mul_const _).const_add (z x)).exp
  have hN := HasDerivAt.fun_sum (u := univ) (fun x _ => (hd x).const_mul (s x))
  have hDd := HasDerivAt.fun_sum (u := univ) (fun x _ => hd x)
  refine (hN.div hDd (by simpa using hD.ne')).congr_deriv ?_
  have hD0 := hD.ne'
  simp only [mul_ite, mul_one, mul_zero, ite_self, add_zero, sum_ite_eq', mem_univ, if_true]
  field_simp

/-! ## Receipt instances: TV-finished against KL-finished (`qreg-spectv.json`)

Three runs per objective at equal wall (`timeout 580`); the KL arm's first run is the `qreg-spec` run. `measured`
values, `proved` comparisons. -/

open Receipts.qreg_spectv in
/-- **Training the acceptance raises it.** Every TV-finished run accepts above `153/250` (0.612), every KL-finished run
below `599/1000`; the mean gain exceeds `7/500` while each arm's run range stays under `1/500`. -/
theorem spectv_accept :
    (153 / 250 : ℚ) < tvAccept1 ∧ (153 / 250 : ℚ) < tvAccept2 ∧ (153 / 250 : ℚ) < tvAccept3 ∧
    klAccept1 < 599 / 1000 ∧ klAccept2 < 599 / 1000 ∧ klAccept3 < 599 / 1000 ∧
    (7 / 500 : ℚ) < (tvAccept1 + tvAccept2 + tvAccept3) / 3 - (klAccept1 + klAccept2 + klAccept3) / 3 ∧
    max tvAccept1 (max tvAccept2 tvAccept3) - min tvAccept1 (min tvAccept2 tvAccept3) < 1 / 2500 ∧
    max klAccept1 (max klAccept2 klAccept3) - min klAccept1 (min klAccept2 klAccept3) < 1 / 500 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [tvAccept1, tvAccept2, tvAccept3, klAccept1, klAccept2, klAccept3]

open Receipts.qreg_spectv in
/-- **The outcomes move with acceptance, the KL does not.** Every TV run's top-1 is above every KL run's (`0.554`
separates them) while its held-out KL is above `47/50` against below `43/50`: KL is not the outcome here. -/
theorem spectv_top1_kl :
    (277 / 500 : ℚ) < tvTop11 ∧ (277 / 500 : ℚ) < tvTop12 ∧ (277 / 500 : ℚ) < tvTop13 ∧
    klTop11 < 277 / 500 ∧ klTop12 < 277 / 500 ∧ klTop13 < 277 / 500 ∧
    (47 / 50 : ℚ) < tvKl1 ∧ (47 / 50 : ℚ) < tvKl2 ∧ (47 / 50 : ℚ) < tvKl3 ∧
    klKl1 < 43 / 50 ∧ klKl2 < 43 / 50 ∧ klKl3 < 43 / 50 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [tvTop11, tvTop12, tvTop13, klTop11, klTop12, klTop13, tvKl1, tvKl2, tvKl3, klKl1, klKl2, klKl3]

/-- `model`: one 27B Q4_K_M verifier pass over `m` tokens on this CPU, seconds: the measured Qwen3-8B Q4_K_M pass of
`m` tokens (`verifyMs<m>`, llama.cpp, 24 threads) scaled by weight bytes (27B at `q4kmBpw` against the 8B GGUF). A
pass of `m` tokens is not one weight read: `t 7 / t 1 = 1.80`. -/
noncomputable def tPass27 (ms : ℚ) : ℝ :=
  27 * 10 ^ 9 * (Receipts.w18_size05.q4kmBpw : ℝ) / 8 * ((ms : ℝ) / 1000) /
    (Receipts.qreg_spectv.verifyModelBytes : ℝ)

open Receipts.qreg_spectv in
/-- **The CPU verify is not one weight read.** A pass of 7 tokens costs above `7/4` of a pass of one, and the cost
rises monotonically in `m`, so the draft length that maximises `E_k / t(k+1)` is not the longest. -/
theorem verify_curve :
    (7 / 4 : ℚ) * verifyMs1 < verifyMs7 ∧ verifyMs1 < verifyMs2 ∧ verifyMs2 < verifyMs3 ∧ verifyMs3 < verifyMs4 ∧
    verifyMs4 < verifyMs5 ∧ verifyMs5 < verifyMs6 ∧ verifyMs6 < verifyMs7 ∧ verifyMs7 < verifyMs8 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [verifyMs1, verifyMs2, verifyMs3, verifyMs4, verifyMs5, verifyMs6, verifyMs7, verifyMs8]

open Receipts.qreg_spectv in
/-- **Draft length 3 is best on this CPU.** With the TV draft's mean block lengths, `E_k / (t(k+1) / t(1))` at
`k = 3` exceeds every other `k = 1..7` and exceeds `7/4`. -/
theorem verify_best_k :
    (7 / 4 : ℚ) < tvEK3Mean / (verifyMs4 / verifyMs1) ∧
    tvEK1Mean / (verifyMs2 / verifyMs1) < tvEK3Mean / (verifyMs4 / verifyMs1) ∧
    tvEK2Mean / (verifyMs3 / verifyMs1) < tvEK3Mean / (verifyMs4 / verifyMs1) ∧
    tvEK4Mean / (verifyMs5 / verifyMs1) < tvEK3Mean / (verifyMs4 / verifyMs1) ∧
    tvEK5Mean / (verifyMs6 / verifyMs1) < tvEK3Mean / (verifyMs4 / verifyMs1) ∧
    tvEK6Mean / (verifyMs7 / verifyMs1) < tvEK3Mean / (verifyMs4 / verifyMs1) ∧
    tvEK7Mean / (verifyMs8 / verifyMs1) < tvEK3Mean / (verifyMs4 / verifyMs1) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [verifyMs1, verifyMs2, verifyMs3, verifyMs4, verifyMs5, verifyMs6, verifyMs7, verifyMs8,
      tvEK1Mean, tvEK2Mean, tvEK3Mean, tvEK4Mean, tvEK5Mean, tvEK6Mean, tvEK7Mean]

/-- **27B at 4 GB VRAM, output exact to the Q4_K_M verifier.** The verifier alone on this CPU decodes below 4 tokens
per second; with the TV-finished 1.18-bit draft at `k = 3` the pair emits above `13/2`, more than `33/20` of the
verifier alone. At `k = 6` the measured verify cost holds the pair below 6: the 1-token-verify step model that gave
above 10 is refuted by `verify_curve`. Streaming the verifier over this box's PCIe link takes above a second per pass. -/
theorem spec27b_cpu :
    1 / tPass27 Receipts.qreg_spectv.verifyMs1 < 4 ∧ tDraft27 < 5 / 1000 ∧
    13 / 2 < (Receipts.qreg_spectv.tvEK3Mean : ℝ) / (tPass27 Receipts.qreg_spectv.verifyMs4 + 3 * tDraft27) ∧
    33 / 20 / tPass27 Receipts.qreg_spectv.verifyMs1 <
      (Receipts.qreg_spectv.tvEK3Mean : ℝ) / (tPass27 Receipts.qreg_spectv.verifyMs4 + 3 * tDraft27) ∧
    (Receipts.qreg_spectv.tvEK6Mean : ℝ) / (tPass27 Receipts.qreg_spectv.verifyMs7 + 6 * tDraft27) < 6 ∧
    1 < 27 * 10 ^ 9 * (Receipts.w18_size05.q4kmBpw : ℝ) / 8 / ((Receipts.qreg_spec.h2dGBs : ℝ) * 10 ^ 9) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [tPass27, tDraft27, Receipts.qreg_spectv.verifyMs1, Receipts.qreg_spectv.verifyMs4,
      Receipts.qreg_spectv.verifyMs7, Receipts.qreg_spectv.verifyModelBytes, Receipts.qreg_spectv.tvEK3Mean,
      Receipts.qreg_spectv.tvEK6Mean, Receipts.qreg_spec.deviceReadGBs, Receipts.qreg_spec.h2dGBs,
      Receipts.w18_size05.q4kmBpw]

end Perf.SpecDraftKL
