import Perf.SpecDraftKL
import Perf.Receipts.Onebit

/-!
# Straight-through KL distillation of a single-sign-plane student: verdicts

Student weight (`tools/onebit/kl_distill.py`): `W'[o,i] = r[o, ⌊i/128⌋] · sign(L[o,i]) · c[i]` with `r, c` in
f16 and `L` an fp32 latent that is not stored. Stored bits per weight: `1 + 16/128 + 16/O`, summed over the
168 linears of Qwen2.5-0.5B. Trained against the teacher by `KL(teacher ‖ student)` (straight-through
estimator on `sign`; Bengio, Léonard and Courville 2013, arXiv 1308.3432) on C4 shard 00001; scored on
`calib-c4` chunks `[100, 140)` in fp32, the instrument of the post-training arm.

All constants are generated from the committed receipts (`Perf/Receipts/Onebit.lean`):
`w18_kd_main1` (520 steps), `w18_kd_main2` (880 steps), `w18_resq_3seed` (the 2-plane E8 post-training arm),
`fold_transfer_1011` (single-plane post-training `fold-local`, `rtn`, and Q4_K_M).

Grades: every statement is a `proved` arithmetic fact about `measured` constants.
-/

namespace Perf.KDScaling

open Receipts

/-- The stored rate is `bits / params` exactly: below the post-training arm's exact `bits / params`, and
within `10⁻¹²` of the receipt's decimal `bpw_linears`. -/
theorem bpw_below_arm :
    (w18_kd_main2.bits : ℚ) / w18_kd_main2.params < (w18_resq_3seed.bits : ℚ) / w18_resq_3seed.params ∧
    |(w18_kd_main2.bits : ℚ) / w18_kd_main2.params - w18_kd_main2.bpwLinears| < 1 / 10 ^ 12 ∧
    w18_kd_main1.bits = w18_kd_main2.bits := by
  refine ⟨?_, ?_, rfl⟩ <;>
    norm_num [w18_kd_main2.bpwLinears, w18_resq_3seed.bits, w18_resq_3seed.params, w18_kd_main2.bits,
      w18_kd_main2.params, abs_lt]

/-- **Training verdict.** From the `sign(W)` start (held-out KL 12.69) both runs improve; the longer run
beats the shorter; both beat single-plane post-training `fold-local` (KL 3.806 at 1.133 b/w) at a lower
rate than the post-training arm; the 2-plane post-training arm still leads; Q4_K_M is two orders away. -/
theorem kd_order :
    w18_kd_main2.klHeldout < w18_kd_main1.klHeldout ∧
    w18_kd_main1.klHeldout < w18_kd_main1.klInit ∧
    w18_kd_main2.klHeldout < fold_transfer_1011.foldLocalKlHeldout ∧
    w18_kd_main1.klHeldout < fold_transfer_1011.foldLocalKlHeldout ∧
    w18_resq_3seed.klHeldoutMean < w18_kd_main2.klHeldout ∧
    fold_transfer_1011.q4kmKlHeldout < w18_resq_3seed.klHeldoutMean := by
  norm_num [w18_kd_main2.klHeldout, w18_kd_main1.klHeldout, w18_kd_main1.klInit,
    fold_transfer_1011.foldLocalKlHeldout, w18_resq_3seed.klHeldoutMean, w18_resq_3seed.klHeldout0,
    w18_resq_3seed.klHeldout1, w18_resq_3seed.klHeldout2, fold_transfer_1011.q4kmKlHeldout]

/-- **Ratios.** Distillation beats `fold-local` by more than 1.8×, beats its own start by more than 6×,
and sits between 101× and 102× above Q4_K_M's held-out KL. -/
theorem kd_ratios :
    (9 / 5 : ℚ) * w18_kd_main2.klHeldout < fold_transfer_1011.foldLocalKlHeldout ∧
    6 * w18_kd_main2.klHeldout < w18_kd_main2.klInit ∧
    101 * fold_transfer_1011.q4kmKlHeldout < w18_kd_main2.klHeldout ∧
    w18_kd_main2.klHeldout < 102 * fold_transfer_1011.q4kmKlHeldout := by
  norm_num [w18_kd_main2.klHeldout, w18_kd_main2.klInit, fold_transfer_1011.foldLocalKlHeldout,
    fold_transfer_1011.q4kmKlHeldout]

/-- Token accounting: both runs see 4096 tokens per step (batch 8 × 512). -/
theorem tokens_per_step :
    w18_kd_main1.tokensTrained = 4096 * w18_kd_main1.steps ∧
    w18_kd_main2.tokensTrained = 4096 * w18_kd_main2.steps := by
  norm_num [w18_kd_main1.tokensTrained, w18_kd_main1.steps, w18_kd_main2.tokensTrained, w18_kd_main2.steps]

/-- The Pinsker guarantee on speculative acceptance (`Perf.SpecDraftKL.accept_ge`) is vacuous at this
student's held-out KL: `1 - √(KL/2) < 0`. Acceptance for this student is a measurement from logits. -/
theorem kd_accept_bound_vacuous :
    1 - Real.sqrt (((w18_kd_main2.klHeldout : ℚ) : ℝ) / 2) < 0 := by
  have hk : ((w18_kd_main2.klHeldout : ℚ) : ℝ) = 20103699624538423 / 10000000000000000 := by
    unfold w18_kd_main2.klHeldout; push_cast; norm_num
  rw [hk]
  have : (1 : ℝ) < Real.sqrt (20103699624538423 / 10000000000000000 / 2) := by
    rw [Real.lt_sqrt (by norm_num)]; norm_num
  linarith

end Perf.KDScaling
