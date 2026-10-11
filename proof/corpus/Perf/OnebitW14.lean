import Mathlib.Tactic.NormNum

/-!
# Sampled-Fisher stiffness of the five `ffn_down` arms against P1 KLD (verdicts)

Per token, in ten-thousandths of a nat, on the 21 `ffn_down` tensors of Qwen2.5-0.5B (P1). `pred` is the
Gauss-Newton second-order KL `Σ_l ½ E[⟨J_l, D_l⟩²]` with `J_l` the gradient of the sequence NLL on `W_l`, one
multinomial target per sample, 400 samples of 512 tokens (`tools/onebit/fisher_capture.py`; `model` grade beyond
the small-perturbation regime). `p1` is the measured paired KLD (`docs/evidence/onebit-w9.md`, `onebit-w12.md`).
Figures: `docs/evidence/onebit-w14.md`. Prediction `stiff-1-1`: rank clause refuted, ratio clause confirmed.
-/

namespace Perf

namespace OnebitW14

/-- Sampled-Fisher sum over blocks, order `q4_k_m < iq1_m < had_gptq1 < iq1_s < rtn`. -/
theorem pred_order : 39 < 3446 ∧ 3446 < 3735 ∧ 3735 < 4564 ∧ 4564 < 67661 := by decide

/-- Measured P1 KLD, order `q4_k_m < iq1_m < iq1_s < had_gptq1 < rtn`. -/
theorem p1_order : 243 < 5191 ∧ 5191 < 6315 ∧ 6315 < 8345 ∧ 8345 < 103846 := by decide

/-- The pair `(iq1_s, had_gptq1)` is ordered oppositely by the two: the predicted rank clause is refuted. -/
theorem rank_inversion : (3735 < 4564 ∧ 6315 < 8345) ∧ ¬ (4564 < 3735) := by decide

/-- Prediction `≥ 0.3 ×` P1 on the three sub-1-nat arms (sum over blocks; `had_gptq1` joint too): the ratio
clause holds. -/
theorem ratio_band :
    3 * 5191 ≤ 10 * 3446 ∧ 3 * 6315 ≤ 10 * 4564 ∧ 3 * 8345 ≤ 10 * 3735 ∧ 3 * 8345 ≤ 10 * 2919 := by decide

/-- Joint over sum of single-block KL (hundredths): `had_gptq1`, `rtn` and `iq1_m` are below 1. -/
theorem additivity : 78 < 100 ∧ 53 < 100 ∧ 96 < 100 := by decide

end OnebitW14

end Perf
