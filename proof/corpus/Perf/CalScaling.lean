import Perf.KDScaling
import Perf.Receipts.Onebit

/-!
# Calibration size and the 2-plane E8 `-resq` arm: verdicts

The arm (`tools/onebit/fold_transfer.py`, `had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-resq-qmlp-nb`,
1.178 b/w over the 168 linears of Qwen2.5-0.5B) was fitted on 100 calibration chunks of 512 tokens
(`w18_resq_3seed`, three chunk orders) and again on 600 chunks: the same 100 plus the first 500 chunks of C4 shard
00001 (`w18_calx600`, `--cal-extra 500`). Held-out KL is scored on `calib-c4` chunks `[100, 140)` in fp32 in both;
in-sample KL on chunks `[0, 20)`, which are calibration chunks in both fits.

Prediction `calx-1` (held-out KL in `[0.80, 1.05]`; falsifier `≥ 1.20`) is refuted: six times the calibration tokens
moves held-out KL by less than 3.4%, while in-sample KL more than doubles. The 3.7× in-sample/held-out gap of the
100-chunk fit is overfitting of the in-sample chunks, not a held-out level that more data recovers.

Grades: every statement is a `proved` arithmetic fact about `measured` constants (`Perf/Receipts/Onebit.lean`).
-/

namespace Perf.CalScaling

open Receipts

/-- Same code, six times the calibration tokens: identical stored bits and parameters, 600 chunks of 512 tokens. -/
theorem calx_same_code :
    w18_calx600.bits = w18_resq_3seed.bits ∧ w18_calx600.params = w18_resq_3seed.params ∧
    w18_calx600.calTokens = 600 * 512 ∧ w18_calx600.calTokens = 6 * (100 * 512) := by
  norm_num [w18_calx600.bits, w18_resq_3seed.bits, w18_calx600.params, w18_resq_3seed.params,
    w18_calx600.calTokens]

/-- **Held-out verdict.** Below every 100-chunk seed, by less than 0.04 against the closest seed and less than
`1/30` of the 3-seed mean; above the registered falsifier `1.20`. -/
theorem calx_heldout :
    w18_calx600.klHeldout < w18_resq_3seed.klHeldout0 ∧
    w18_calx600.klHeldout < w18_resq_3seed.klHeldout1 ∧
    w18_calx600.klHeldout < w18_resq_3seed.klHeldout2 ∧
    w18_resq_3seed.klHeldout1 - w18_calx600.klHeldout < 1 / 25 ∧
    (29 / 30 : ℚ) * w18_resq_3seed.klHeldoutMean < w18_calx600.klHeldout ∧
    (6 / 5 : ℚ) < w18_calx600.klHeldout := by
  norm_num [w18_calx600.klHeldout, w18_resq_3seed.klHeldoutMean, w18_resq_3seed.klHeldout0,
    w18_resq_3seed.klHeldout1, w18_resq_3seed.klHeldout2]

/-- **Overfit verdict.** In-sample KL more than doubles with more calibration data; the 100-chunk fit's held-out
KL exceeds 3.6× its in-sample KL, the 600-chunk fit's is below 1.7×. -/
theorem calx_overfit :
    2 * w18_resq_3seed.klInsample0 < w18_calx600.klInsample ∧
    (18 / 5 : ℚ) * w18_resq_3seed.klInsample0 < w18_resq_3seed.klHeldout0 ∧
    w18_calx600.klHeldout < (17 / 10 : ℚ) * w18_calx600.klInsample := by
  norm_num [w18_calx600.klInsample, w18_calx600.klHeldout, w18_resq_3seed.klInsample0,
    w18_resq_3seed.klHeldout0]

/-- **Standing.** The 600-chunk arm leads straight-through distillation (`w18_kd_main2`, 2.01) and sits between
63× and 64× above Q4_K_M's held-out KL. -/
theorem calx_standing :
    w18_calx600.klHeldout < w18_kd_main2.klHeldout ∧
    63 * fold_transfer_1011.q4kmKlHeldout < w18_calx600.klHeldout ∧
    w18_calx600.klHeldout < 64 * fold_transfer_1011.q4kmKlHeldout := by
  norm_num [w18_calx600.klHeldout, w18_kd_main2.klHeldout, fold_transfer_1011.q4kmKlHeldout]

end Perf.CalScaling
