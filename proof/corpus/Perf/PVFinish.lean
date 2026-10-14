import Perf.Receipts.Onebit

/-!
# End-to-end KL finishing of the 2-plane E8 `-resq` arm: verdicts

`tools/onebit/pv_finish.py` (sha256 `c973b38c…`) builds the arm on Qwen2.5-0.5B exactly as the size-trend receipt
(`w18-size05-n50.json`: 50 calibration chunks, held-out `calib-c4` `[50, 90)`, in-sample `[0, 20)`), with
`fold_transfer.py` (sha256 `f21e276a…`) recording each linear's stored planes `(S_rot, r, c, d, rows)` through
`PACK_SINK`. It then optimises that stored code against forward KL to the fp32 teacher on C4 training tokens disjoint
from calibration, in the manner of PV-Tuning (Malinovskii et al., arXiv 2405.14852):

- P-step: Adam on the group scales `r`, column scales `c` and `d`, through the arm's own quantizers by a straight-through
  estimator (frozen per-row 16-level `r` grid, `quant_c_u8` at 6 and 5 bits), plus the norms and q/k/v biases.
- V-step: every 25 steps, a linearised move of E8 ∪ O8 codebook indices among the 16 nearest rows, accepted only if a
  disjoint training batch's loss falls.

Prediction `pv05-1` (held-out KL in `[0.55, 0.95]`; falsifier `≥ 1.10`) is confirmed: held-out KL falls from 1.1637 to
0.8981 at identical stored bits, while in-sample KL rises. One of 17 V-steps was accepted; the P-step carries the gain,
and it trains under 5% of the stored bits (the scales). The first-order V-step predicted a loss fall for all 136 candidate
moves of the sign codes and realised one, at under 1/45 of its prediction: the code (95% of the bits) is where the next
gain has to come from, and its move needs a curvature term.

Grades: every statement is a `proved` arithmetic fact about `measured` constants (`Perf/Receipts/Onebit.lean`); the
controls are `tested` booleans checked by `tools/rooting/receipt_lean.py (unpublished)` before they are emitted.
-/

namespace Perf.PVFinish

open Receipts

/-- **Same code, same bits.** The finishing run's arm is the size-trend arm: its pre-finishing held-out KL equals the
size-trend receipt's to the last digit, and the stored bit count is unchanged by finishing. -/
theorem pv_same_arm :
    w18_pv05.arm = w18_size05.arm ∧ w18_pv05.params = w18_size05.params ∧
    w18_pv05.bitsBefore = w18_size05.bits ∧ w18_pv05.bitsAfter = w18_pv05.bitsBefore ∧
    w18_pv05.klHeldoutBefore = w18_size05.klHeldout ∧
    w18_pv05.beforeMatchesReference = true ∧ w18_pv05.bitsEqual = true ∧ w18_pv05.codebookExact = true ∧
    w18_pv05.dataDisjoint = true := by
  refine ⟨rfl, rfl, rfl, rfl, ?_, rfl, rfl, rfl, rfl⟩
  norm_num [w18_pv05.klHeldoutBefore, w18_size05.klHeldout]

/-- **Prediction `pv05-1` confirmed.** Held-out KL after finishing lies in `[0.55, 0.95]`, below the `1.10`
falsifier. -/
theorem pv05_confirmed :
    (55 / 100 : ℚ) ≤ w18_pv05.klHeldoutAfter ∧ w18_pv05.klHeldoutAfter ≤ 95 / 100 ∧
    w18_pv05.klHeldoutAfter < 11 / 10 := by
  norm_num [w18_pv05.klHeldoutAfter]

/-- **Held-out gain.** Finishing removes over a fifth of held-out KL and raises held-out top-1 by over 5 points. -/
theorem pv_heldout_drop :
    w18_pv05.klHeldoutAfter < (78 / 100 : ℚ) * w18_pv05.klHeldoutBefore ∧
    w18_pv05.top1HeldoutBefore + 5 / 100 < w18_pv05.top1HeldoutAfter := by
  norm_num [w18_pv05.klHeldoutAfter, w18_pv05.klHeldoutBefore, w18_pv05.top1HeldoutBefore,
    w18_pv05.top1HeldoutAfter]

/-- **The gain is generalisation.** In-sample KL rises while held-out falls; the held-out/in-sample ratio drops from
above 5 to below 4. -/
theorem pv_gap_shrinks :
    w18_pv05.klInsampleBefore < w18_pv05.klInsampleAfter ∧
    5 * w18_pv05.klInsampleBefore < w18_pv05.klHeldoutBefore ∧
    w18_pv05.klHeldoutAfter < 4 * w18_pv05.klInsampleAfter := by
  norm_num [w18_pv05.klInsampleBefore, w18_pv05.klInsampleAfter, w18_pv05.klHeldoutBefore,
    w18_pv05.klHeldoutAfter]

/-- **Standing against Q4_K_M.** The finished arm's held-out KL is between 48 and 49 times Q4_K_M's on the same chunks
(63–64 times before finishing, `Perf.SizeTrend.size_q4km_ratio`). -/
theorem pv_q4km_ratio :
    48 * w18_size05.q4kmKlHeldout < w18_pv05.klHeldoutAfter ∧
    w18_pv05.klHeldoutAfter < 49 * w18_size05.q4kmKlHeldout := by
  norm_num [w18_size05.q4kmKlHeldout, w18_pv05.klHeldoutAfter]

/-- **The P-step carries it.** One of 17 V-steps was accepted. -/
theorem pv_vstep_rare : w18_pv05.vAccepted = 1 ∧ w18_pv05.vSteps = 17 := ⟨rfl, rfl⟩

/-- **Where the gain lives.** The P-step trains the group and column scales `r`, `c`, `d`: under 5% of the stored bits.
The E8 ∪ O8 sign codes hold over 95% and are fixed except at accepted V-steps. -/
theorem pv_scales_small :
    w18_pv05.bitsSign + w18_pv05.bitsScales + w18_pv05.bitsCorr = w18_pv05.bitsBefore ∧
    20 * w18_pv05.bitsScales < w18_pv05.bitsBefore ∧
    100 * w18_pv05.bitsSign > 95 * w18_pv05.bitsBefore := by
  refine ⟨rfl, ?_, ?_⟩ <;> norm_num [w18_pv05.bitsScales, w18_pv05.bitsBefore, w18_pv05.bitsSign]

/-- **Why the codes do not move.** Every one of the 136 V-step candidates had a negative first-order predicted loss
change `⟨g, Δ⟩`; only one lowered the realised loss, and that one realised under `1/45` of its prediction. A codeword
move is a finite jump whose curvature term outweighs the linear term. -/
theorem pv_vstep_first_order_fails :
    w18_pv05.vTries = 136 ∧ w18_pv05.vTriesPredictedFall = w18_pv05.vTries ∧ w18_pv05.vTriesFell = 1 ∧
    w18_pv05.vFellRealised < 0 ∧ 45 * w18_pv05.vFellRealised > w18_pv05.vFellPredicted := by
  refine ⟨rfl, rfl, rfl, ?_, ?_⟩ <;> norm_num [w18_pv05.vFellRealised, w18_pv05.vFellPredicted]

end Perf.PVFinish
