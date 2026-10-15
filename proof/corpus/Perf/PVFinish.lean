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

Two further runs score the V-step candidates by a curvature-corrected (Optimal Brain Surgeon, Hassibi and Stork
(1993), at diagonal order) gain `⟨ḡ, Δ⟩ + (λ/2) Σ F Δ²`, with `ḡ` and the diagonal empirical Fisher `F` averaged over the
window's batch gradients: `w18-pvobs05.json` (`pv_finish.py` sha256 `bab41be3…`, `λ ∈ {1, 4, 16, 64}`) and
`w18-pvlam05.json` (sha256 `3e14c5d8…`, `λ ∈ {1024, …, 65536}`). Predictions `pvobs05-1` and `pvobslam05-1` are
refuted: no candidate in either run lowered the loss, even at `λ = 65536`, sixteen times the tokens per batch gradient.
The three runs reproduce the held-out drop through the scales alone (0.8906, 0.8948, 0.8981; mean under 0.895).

Grades: every statement is a `proved` arithmetic fact about `measured` constants (`Perf/Receipts/Onebit.lean`); the
controls are `tested` booleans checked by `tools/rooting/receipt_lean.py` before they are emitted.
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

/-- **The OBS runs are the same arm.** Both curvature-corrected runs start from the size-trend arm's code and held-out
KL to the last digit, keep the stored bits, and pass every control. -/
theorem pvobs_same_arm :
    w18_pvobs05.arm = w18_pv05.arm ∧ w18_pvlam05.arm = w18_pv05.arm ∧
    w18_pvobs05.klHeldoutBefore = w18_pv05.klHeldoutBefore ∧
    w18_pvlam05.klHeldoutBefore = w18_pv05.klHeldoutBefore ∧
    w18_pvobs05.bitsAfter = w18_pv05.bitsBefore ∧ w18_pvlam05.bitsAfter = w18_pv05.bitsBefore ∧
    w18_pvobs05.beforeMatchesReference = true ∧ w18_pvobs05.bitsEqual = true ∧
    w18_pvobs05.codebookExact = true ∧ w18_pvobs05.dataDisjoint = true ∧
    w18_pvlam05.beforeMatchesReference = true ∧ w18_pvlam05.bitsEqual = true ∧
    w18_pvlam05.codebookExact = true ∧ w18_pvlam05.dataDisjoint = true := by
  refine ⟨rfl, rfl, ?_, ?_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ <;>
    norm_num [w18_pvobs05.klHeldoutBefore, w18_pvlam05.klHeldoutBefore, w18_pv05.klHeldoutBefore]

/-- **Predictions `pvobs05-1` and `pvobslam05-1` refuted.** `pvobs05-1` predicted held-out KL in `[0.72, 0.88]`; it
ended above `0.88`. `pvobslam05-1` predicted at least three accepted V-steps; none was accepted. -/
theorem pvobs_refuted :
    (88 / 100 : ℚ) < w18_pvobs05.klHeldoutAfter ∧ w18_pvobs05.vAccepted = 0 ∧ w18_pvlam05.vAccepted = 0 := by
  refine ⟨?_, rfl, rfl⟩
  norm_num [w18_pvobs05.klHeldoutAfter]

/-- **The OBS gain does not find a falling move.** In both runs every candidate was predicted to lower the loss and none
did; the candidate with the smallest realised change still raised it. -/
theorem pvobs_vstep_fails :
    w18_pvobs05.vTriesPredictedFall = w18_pvobs05.vTries ∧ w18_pvobs05.vTriesFell = 0 ∧
    0 < w18_pvobs05.vBestRealised ∧
    w18_pvlam05.vTriesPredictedFall = w18_pvlam05.vTries ∧ w18_pvlam05.vTriesFell = 0 ∧
    0 < w18_pvlam05.vBestRealised ∧
    w18_pvobs05.vTries + w18_pvlam05.vTries = 280 := by
  refine ⟨rfl, rfl, ?_, rfl, rfl, ?_, rfl⟩ <;>
    norm_num [w18_pvobs05.vBestRealised, w18_pvlam05.vBestRealised]

/-- **Damping is not the missing factor.** Each P-step gradient averages 4096 tokens (`tokens = 4096 · steps`), so the
batch-gradient Fisher sits about 4096 times below the per-token Fisher. At `λ = 65536 = 16 · 4096` every candidate still
raised the loss by over `3/2` of its predicted fall, and the best candidate of the small-`λ` run rose by over three
times its prediction. The diagonal empirical Fisher of the loss gradient does not measure the curvature a codeword move
meets. -/
theorem pvobs_damping_insufficient :
    w18_pvlam05.tokens = 4096 * w18_pvlam05.steps ∧ w18_pvobs05.tokens = 4096 * w18_pvobs05.steps ∧
    w18_pvlam05.obsLamMax = 16 * 4096 ∧ w18_pvlam05.vBestLam = w18_pvlam05.obsLamMax ∧
    (3 / 2 : ℚ) < w18_pvlam05.vRiseOverPredMinAtLamMax ∧
    -w18_pvlam05.vBestPredicted < w18_pvlam05.vBestRealised ∧
    -3 * w18_pvobs05.vBestPredicted < w18_pvobs05.vBestRealised := by
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvlam05.obsLamMax, w18_pvlam05.vBestLam, w18_pvlam05.vRiseOverPredMinAtLamMax,
      w18_pvlam05.vBestPredicted, w18_pvlam05.vBestRealised, w18_pvobs05.vBestPredicted, w18_pvobs05.vBestRealised]

/-- **The scales carry the drop (three runs).** The run with one accepted V-step is not better than either run with
none; across the three runs held-out KL after finishing lies within `1/125` of each other, with mean in
`(0.894, 0.895)`, and each run removes over a fifth of the arm's held-out KL. -/
theorem pv_scales_carry_gain :
    w18_pv05.vAccepted = 1 ∧ w18_pvobs05.vAccepted = 0 ∧ w18_pvlam05.vAccepted = 0 ∧
    w18_pvobs05.klHeldoutAfter < w18_pvlam05.klHeldoutAfter ∧
    w18_pvlam05.klHeldoutAfter < w18_pv05.klHeldoutAfter ∧
    w18_pv05.klHeldoutAfter - w18_pvobs05.klHeldoutAfter < 1 / 125 ∧
    (894 / 1000 : ℚ) < (w18_pv05.klHeldoutAfter + w18_pvobs05.klHeldoutAfter + w18_pvlam05.klHeldoutAfter) / 3 ∧
    (w18_pv05.klHeldoutAfter + w18_pvobs05.klHeldoutAfter + w18_pvlam05.klHeldoutAfter) / 3 < 895 / 1000 ∧
    w18_pvobs05.klHeldoutAfter < (78 / 100 : ℚ) * w18_pvobs05.klHeldoutBefore ∧
    w18_pvlam05.klHeldoutAfter < (78 / 100 : ℚ) * w18_pvlam05.klHeldoutBefore := by
  refine ⟨rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pv05.klHeldoutAfter, w18_pvobs05.klHeldoutAfter, w18_pvlam05.klHeldoutAfter,
      w18_pvobs05.klHeldoutBefore, w18_pvlam05.klHeldoutBefore]

/-- The straight-through run is the same arm on the same protocol: the before-KL reproduces the reference bit for
bit and every stored-code control passes, at the bits the scales-only runs stored. -/
theorem pvste_same_arm :
    w18_pvste05.arm = w18_pv05.arm ∧ w18_pvste05.klHeldoutBefore = w18_pv05.klHeldoutBefore ∧
    w18_pvste05.bitsAfter = w18_pv05.bitsBefore ∧ w18_pvste05.bitsBefore = w18_pv05.bitsBefore ∧
    w18_pvste05.beforeMatchesReference = true ∧ w18_pvste05.bitsEqual = true ∧
    w18_pvste05.codebookExact = true ∧ w18_pvste05.dataDisjoint = true := by
  refine ⟨rfl, ?_, rfl, rfl, rfl, rfl, rfl, rfl⟩
  norm_num [w18_pvste05.klHeldoutBefore, w18_pv05.klHeldoutBefore]

/-- **Prediction `pvste05-1` refuted.** It predicted held-out KL in `[0.65, 0.86]` from training the E8 ∪ O8 codes by a
straight-through projection jointly with the scales. Held-out KL rose above `1.7×` its starting value, past the
scales-only mean `0.894` (`pv_scales_carry_gain`), and in-sample KL rose over eightfold. -/
theorem pvste_refuted :
    (894 / 1000 : ℚ) < w18_pvste05.klHeldoutAfter ∧
    (17 / 10 : ℚ) * w18_pvste05.klHeldoutBefore < w18_pvste05.klHeldoutAfter ∧
    w18_pvste05.top1HeldoutAfter < w18_pvste05.top1HeldoutBefore ∧
    8 * w18_pvste05.klInsampleBefore < w18_pvste05.klInsampleAfter := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvste05.klHeldoutAfter, w18_pvste05.klHeldoutBefore, w18_pvste05.top1HeldoutAfter,
      w18_pvste05.top1HeldoutBefore, w18_pvste05.klInsampleAfter, w18_pvste05.klInsampleBefore]

/-- **The codes were flipped wholesale, from the first window.** At `lr-code = 1/10` under Adam, over a quarter of all
8-blocks changed index; more than one block in 25 changed in steps 1..25, and the train loss over those steps rose more
than threefold. Adam's per-coordinate normalisation moves every latent about `lr` per step whatever its gradient, so
the flips are not ordered by the loss. -/
theorem pvste_flips_wholesale :
    w18_pvste05.lrCode = 1 / 10 ∧ w18_pvste05.codeKinds = 7 ∧
    w18_pvste05.codeBlocks < 4 * w18_pvste05.codeChangedTotal ∧
    w18_pvste05.codeBlocks < 25 * w18_pvste05.codeChangedFirstWindow ∧
    3 * w18_pvste05.trainLossStep1 < w18_pvste05.trainLossStep25 := by
  refine ⟨?_, rfl, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvste05.lrCode, w18_pvste05.codeBlocks, w18_pvste05.codeChangedTotal,
      w18_pvste05.codeChangedFirstWindow, w18_pvste05.trainLossStep1, w18_pvste05.trainLossStep25]

/-- The pre-image run is the same arm on the same protocol, and its latent initialisation reproduces the stored code on
every block at step 0. -/
theorem pvpre_same_arm :
    w18_pvpre05.arm = w18_pv05.arm ∧ w18_pvpre05.klHeldoutBefore = w18_pv05.klHeldoutBefore ∧
    w18_pvpre05.bitsAfter = w18_pv05.bitsBefore ∧ w18_pvpre05.step0ProjectionMismatch = 0 ∧
    w18_pvpre05.beforeMatchesReference = true ∧ w18_pvpre05.bitsEqual = true ∧
    w18_pvpre05.codebookExact = true ∧ w18_pvpre05.dataDisjoint = true := by
  refine ⟨rfl, ?_, rfl, rfl, rfl, rfl, rfl, rfl⟩
  norm_num [w18_pvpre05.klHeldoutBefore, w18_pv05.klHeldoutBefore]

/-- **Prediction `pvpre05-1` refuted.** It predicted held-out KL in `[0.78, 0.88]`, falsified at or above the best
scales-only run. Held-out KL after finishing lies above the three scales-only runs' mean by more than three times their
spread. -/
theorem pvpre_refuted :
    w18_pvobs05.klHeldoutAfter ≤ w18_pvpre05.klHeldoutAfter ∧
    (w18_pv05.klHeldoutAfter + w18_pvobs05.klHeldoutAfter + w18_pvlam05.klHeldoutAfter) / 3 +
      3 * (w18_pv05.klHeldoutAfter - w18_pvobs05.klHeldoutAfter) < w18_pvpre05.klHeldoutAfter ∧
    w18_pvpre05.klHeldoutAfter < w18_pvpre05.klHeldoutBefore := by
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [w18_pvpre05.klHeldoutAfter, w18_pvpre05.klHeldoutBefore, w18_pv05.klHeldoutAfter,
      w18_pvobs05.klHeldoutAfter, w18_pvlam05.klHeldoutAfter]

/-- **Sparse, gradient-ordered flips still cost.** Fewer than one block in 5000 changed index, and fewer than half of
those moved to the pre-image's nearest codeword; the pre-image's nearest codeword is the stored index on under two
blocks in five. -/
theorem pvpre_flips_sparse :
    5000 * w18_pvpre05.codeChangedTotal < w18_pvpre05.codeBlocks ∧
    2 * w18_pvpre05.flipsTowardPreimage < w18_pvpre05.codeChangedTotal ∧
    w18_pvpre05.preimageNearestIsIdx < 2 / 5 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [w18_pvpre05.codeChangedTotal, w18_pvpre05.codeBlocks, w18_pvpre05.flipsTowardPreimage,
      w18_pvpre05.preimageNearestIsIdx]

/-- **No code move has beaten the scales (five runs).** The best held-out KL after finishing is a run that moved no
code; each run that changed a code index (one accepted linear V-step; Adam straight-through; pre-image
straight-through) ends above it. -/
theorem pv_code_moves_cost :
    w18_pvobs05.vAccepted = 0 ∧ w18_pvobs05.klHeldoutAfter < w18_pvlam05.klHeldoutAfter ∧
    w18_pvobs05.klHeldoutAfter < w18_pv05.klHeldoutAfter ∧
    w18_pvobs05.klHeldoutAfter < w18_pvpre05.klHeldoutAfter ∧
    w18_pvobs05.klHeldoutAfter < w18_pvste05.klHeldoutAfter := by
  refine ⟨rfl, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvobs05.klHeldoutAfter, w18_pvlam05.klHeldoutAfter, w18_pv05.klHeldoutAfter,
      w18_pvpre05.klHeldoutAfter, w18_pvste05.klHeldoutAfter]

/-! ## Which tunables carry the drop (`pv_finish.py --train`, V-steps off) -/

/-- The ablation runs are the same arm on the same protocol: the before-KL reproduces the reference, the stored bits
are unchanged, no V-step ran, and every control passes. -/
theorem pvab_same_arm :
    w18_pvabnorm05.arm = w18_pv05.arm ∧ w18_pvabr05.arm = w18_pv05.arm ∧
    w18_pvabnorm05.klHeldoutBefore = w18_pv05.klHeldoutBefore ∧
    w18_pvabr05.klHeldoutBefore = w18_pv05.klHeldoutBefore ∧
    w18_pvabnorm05.bitsAfter = w18_pv05.bitsBefore ∧ w18_pvabr05.bitsAfter = w18_pv05.bitsBefore ∧
    w18_pvabnorm05.vSteps = 0 ∧ w18_pvabr05.vSteps = 0 ∧
    w18_pvabnorm05.trainGroups = "norm" ∧ w18_pvabr05.trainGroups = "r" ∧
    w18_pvabnorm05.beforeMatchesReference = true ∧ w18_pvabnorm05.bitsEqual = true ∧
    w18_pvabnorm05.codebookExact = true ∧ w18_pvabnorm05.dataDisjoint = true ∧
    w18_pvabr05.beforeMatchesReference = true ∧ w18_pvabr05.bitsEqual = true ∧
    w18_pvabr05.codebookExact = true ∧ w18_pvabr05.dataDisjoint = true := by
  refine ⟨rfl, rfl, ?_, ?_, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ <;>
    norm_num [w18_pvabnorm05.klHeldoutBefore, w18_pvabr05.klHeldoutBefore, w18_pv05.klHeldoutBefore]

/-- **Prediction `pvabr05-1` confirmed; `pvabnorm05-1` refuted.** Training only the group scales `r` ends in
`[0.90, 1.00]`, as predicted. Training only the norms and q/k/v biases ends above the predicted `1.07`, below the
`1.12` falsifier. -/
theorem pvab_predictions :
    (90 / 100 : ℚ) ≤ w18_pvabr05.klHeldoutAfter ∧ w18_pvabr05.klHeldoutAfter ≤ 1 ∧
    (107 / 100 : ℚ) < w18_pvabnorm05.klHeldoutAfter ∧ w18_pvabnorm05.klHeldoutAfter < 112 / 100 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> norm_num [w18_pvabr05.klHeldoutAfter, w18_pvabnorm05.klHeldoutAfter]

/-- **The share of `r` against the V-step runs.** `r` alone removes over nine tenths of the three all-group runs' mean
drop (`pv_scales_carry_gain`); the norms and biases alone remove under a third of it. Not exclusive:
`pvab_scales_substitute` shows the column scales alone reach over eight tenths too. -/
theorem pvab_r_share :
    (9 / 10 : ℚ) * (w18_pv05.klHeldoutBefore -
        (w18_pv05.klHeldoutAfter + w18_pvobs05.klHeldoutAfter + w18_pvlam05.klHeldoutAfter) / 3) <
      w18_pvabr05.klHeldoutBefore - w18_pvabr05.klHeldoutAfter ∧
    3 * (w18_pvabnorm05.klHeldoutBefore - w18_pvabnorm05.klHeldoutAfter) <
      w18_pv05.klHeldoutBefore -
        (w18_pv05.klHeldoutAfter + w18_pvobs05.klHeldoutAfter + w18_pvlam05.klHeldoutAfter) / 3 := by
  refine ⟨?_, ?_⟩ <;>
    norm_num [w18_pv05.klHeldoutBefore, w18_pv05.klHeldoutAfter, w18_pvobs05.klHeldoutAfter,
      w18_pvlam05.klHeldoutAfter, w18_pvabr05.klHeldoutBefore, w18_pvabr05.klHeldoutAfter,
      w18_pvabnorm05.klHeldoutBefore, w18_pvabnorm05.klHeldoutAfter]

/-- **The two groups overlap.** The norm-only and `r`-only drops sum to more than the all-group mean drop, so part of
what each fixes the other also fixes: a norm gain and a row of `r` both rescale a row's output. -/
theorem pvab_groups_overlap :
    w18_pv05.klHeldoutBefore -
        (w18_pv05.klHeldoutAfter + w18_pvobs05.klHeldoutAfter + w18_pvlam05.klHeldoutAfter) / 3 <
      (w18_pvabnorm05.klHeldoutBefore - w18_pvabnorm05.klHeldoutAfter) +
        (w18_pvabr05.klHeldoutBefore - w18_pvabr05.klHeldoutAfter) := by
  norm_num [w18_pv05.klHeldoutBefore, w18_pv05.klHeldoutAfter, w18_pvobs05.klHeldoutAfter,
    w18_pvlam05.klHeldoutAfter, w18_pvabr05.klHeldoutBefore, w18_pvabr05.klHeldoutAfter,
    w18_pvabnorm05.klHeldoutBefore, w18_pvabnorm05.klHeldoutAfter]

/-- **The ablation runs saw more tokens.** With V-steps off, each ablation run took over 13/10 the steps of every
all-group run in the same wall-clock box (`tokens = 4096 · steps` in all five). The all-group run without V-steps
(`w18-pvaball05v0`) is the matched denominator. -/
theorem pvab_more_steps :
    w18_pvabr05.tokens = 4096 * w18_pvabr05.steps ∧ w18_pvabnorm05.tokens = 4096 * w18_pvabnorm05.steps ∧
    13 * w18_pvobs05.steps < 10 * w18_pvabr05.steps ∧ 13 * w18_pvobs05.steps < 10 * w18_pvabnorm05.steps ∧
    w18_pv05.steps < w18_pvobs05.steps ∧ w18_pvlam05.steps < w18_pvobs05.steps := by
  refine ⟨rfl, rfl, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvabr05.steps, w18_pvabnorm05.steps, w18_pvobs05.steps, w18_pv05.steps, w18_pvlam05.steps]

/-- The matched denominator is the same arm on the same protocol: every group trains, no V-step runs, the stored bits
are unchanged and every control passes. -/
theorem pvab_matched_same_arm :
    w18_pvaball05v0.arm = w18_pv05.arm ∧ w18_pvaball05v0.trainGroups = "c,d,norm,r" ∧
    w18_pvaball05v0.klHeldoutBefore = w18_pv05.klHeldoutBefore ∧
    w18_pvaball05v0.bitsAfter = w18_pv05.bitsBefore ∧ w18_pvaball05v0.vSteps = 0 ∧
    w18_pvaball05v0.tokens = 4096 * w18_pvaball05v0.steps ∧
    w18_pvaball05v0.beforeMatchesReference = true ∧ w18_pvaball05v0.bitsEqual = true ∧
    w18_pvaball05v0.codebookExact = true ∧ w18_pvaball05v0.dataDisjoint = true := by
  refine ⟨rfl, rfl, ?_, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  norm_num [w18_pvaball05v0.klHeldoutBefore, w18_pv05.klHeldoutBefore]

/-- **Prediction `pvaball05v0-1` confirmed, and V-steps off is the best finishing run.** Every group with no V-step ends
in `[0.85, 0.90]` and below each of the three all-group runs that paused for V-steps; it took more steps than any of
them in the same wall-clock box. One run: the margin under the best of them, `0.8906`, is above their spread `0.0075`
and under three times it. -/
theorem pvab_matched_best :
    (85 / 100 : ℚ) ≤ w18_pvaball05v0.klHeldoutAfter ∧ w18_pvaball05v0.klHeldoutAfter ≤ 90 / 100 ∧
    w18_pvaball05v0.klHeldoutAfter < w18_pvobs05.klHeldoutAfter ∧
    w18_pvaball05v0.klHeldoutAfter < w18_pvlam05.klHeldoutAfter ∧
    w18_pvaball05v0.klHeldoutAfter < w18_pv05.klHeldoutAfter ∧
    w18_pvobs05.steps < w18_pvaball05v0.steps := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvaball05v0.klHeldoutAfter, w18_pvobs05.klHeldoutAfter, w18_pvlam05.klHeldoutAfter,
      w18_pv05.klHeldoutAfter, w18_pvobs05.steps, w18_pvaball05v0.steps]

/-- **The share of `r` against the matched denominator.** With V-steps off in every run, `r` alone removes over nine
tenths of the all-group drop and the norms alone under a third; the two single-group drops sum past it, so they
overlap. Matched in wall clock, not in tokens: the single-group runs step faster and took more steps. -/
theorem pvab_matched_r_share :
    (9 / 10 : ℚ) * (w18_pvaball05v0.klHeldoutBefore - w18_pvaball05v0.klHeldoutAfter) <
      w18_pvabr05.klHeldoutBefore - w18_pvabr05.klHeldoutAfter ∧
    3 * (w18_pvabnorm05.klHeldoutBefore - w18_pvabnorm05.klHeldoutAfter) <
      w18_pvaball05v0.klHeldoutBefore - w18_pvaball05v0.klHeldoutAfter ∧
    w18_pvaball05v0.klHeldoutBefore - w18_pvaball05v0.klHeldoutAfter <
      (w18_pvabnorm05.klHeldoutBefore - w18_pvabnorm05.klHeldoutAfter) +
        (w18_pvabr05.klHeldoutBefore - w18_pvabr05.klHeldoutAfter) ∧
    w18_pvaball05v0.steps < w18_pvabr05.steps ∧ w18_pvabr05.steps < w18_pvabnorm05.steps := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvaball05v0.klHeldoutBefore, w18_pvaball05v0.klHeldoutAfter, w18_pvabr05.klHeldoutBefore,
      w18_pvabr05.klHeldoutAfter, w18_pvabnorm05.klHeldoutBefore, w18_pvabnorm05.klHeldoutAfter,
      w18_pvaball05v0.steps, w18_pvabr05.steps, w18_pvabnorm05.steps]

/-- **Prediction `pvabrcd05-1` confirmed: the norms add nothing once the stored scales train.** `r, c, d` with the norms
frozen ends in `[0.86, 0.92]`, within `1/1000` of the all-group run, on the same arm and controls. -/
theorem pvab_rcd_equals_all :
    w18_pvabrcd05.arm = w18_pv05.arm ∧ w18_pvabrcd05.trainGroups = "c,d,r" ∧ w18_pvabrcd05.vSteps = 0 ∧
    w18_pvabrcd05.bitsAfter = w18_pv05.bitsBefore ∧
    w18_pvabrcd05.beforeMatchesReference = true ∧ w18_pvabrcd05.bitsEqual = true ∧
    w18_pvabrcd05.codebookExact = true ∧ w18_pvabrcd05.dataDisjoint = true ∧
    (86 / 100 : ℚ) ≤ w18_pvabrcd05.klHeldoutAfter ∧ w18_pvabrcd05.klHeldoutAfter ≤ 92 / 100 ∧
    |w18_pvabrcd05.klHeldoutAfter - w18_pvaball05v0.klHeldoutAfter| < 1 / 1000 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩
  · norm_num [w18_pvabrcd05.klHeldoutAfter]
  · norm_num [w18_pvabrcd05.klHeldoutAfter]
  · rw [abs_lt]; constructor <;> norm_num [w18_pvabrcd05.klHeldoutAfter, w18_pvaball05v0.klHeldoutAfter]

/-- **Prediction `pvabcd05-1` refuted: the scale groups substitute for each other.** The per-column scales `c, d` alone
(706,560 latents, under a quarter of `r`'s 3,128,832) end at or below the `0.95` falsifier and remove over eight tenths
of the joint `r, c, d` drop; `r` alone removes over nine tenths. The two single-group drops sum past 17/10 of the joint
drop: a per-row block scale and a per-column scale both rescale the same stored weight, and either recovers most of
the gain. Same arm and controls. -/
theorem pvab_scales_substitute :
    w18_pvabcd05.arm = w18_pv05.arm ∧ w18_pvabcd05.trainGroups = "c,d" ∧ w18_pvabcd05.vSteps = 0 ∧
    w18_pvabcd05.bitsAfter = w18_pv05.bitsBefore ∧
    w18_pvabcd05.beforeMatchesReference = true ∧ w18_pvabcd05.bitsEqual = true ∧
    w18_pvabcd05.codebookExact = true ∧ w18_pvabcd05.dataDisjoint = true ∧
    w18_pvabcd05.klHeldoutAfter ≤ 95 / 100 ∧
    (8 / 10 : ℚ) * (w18_pvabrcd05.klHeldoutBefore - w18_pvabrcd05.klHeldoutAfter) <
      w18_pvabcd05.klHeldoutBefore - w18_pvabcd05.klHeldoutAfter ∧
    (9 / 10 : ℚ) * (w18_pvabrcd05.klHeldoutBefore - w18_pvabrcd05.klHeldoutAfter) <
      w18_pvabr05.klHeldoutBefore - w18_pvabr05.klHeldoutAfter ∧
    (17 / 10 : ℚ) * (w18_pvabrcd05.klHeldoutBefore - w18_pvabrcd05.klHeldoutAfter) <
      (w18_pvabcd05.klHeldoutBefore - w18_pvabcd05.klHeldoutAfter) +
        (w18_pvabr05.klHeldoutBefore - w18_pvabr05.klHeldoutAfter) := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvabcd05.klHeldoutAfter, w18_pvabcd05.klHeldoutBefore, w18_pvabrcd05.klHeldoutBefore,
      w18_pvabrcd05.klHeldoutAfter, w18_pvabr05.klHeldoutBefore, w18_pvabr05.klHeldoutAfter]

/-- **Prediction `pvgrid25-1` confirmed: the `r` grid range does not bind.** Refitting each row's 16-level `r` grid to its
trained latents every 25 steps (zero bits) ends in `[0.86, 0.90]` and within the `0.0075` three-run spread of the same
protocol without the refit; the refit moves under a tenth of the `r` latents to a grid end. Same arm and controls. -/
theorem pvgrid_no_gain :
    w18_pvgrid25.arm = w18_pv05.arm ∧ w18_pvgrid25.trainGroups = "c,d,norm,r" ∧ w18_pvgrid25.vSteps = 0 ∧
    w18_pvgrid25.regrid = 25 ∧ w18_pvgrid25.bitsAfter = w18_pv05.bitsBefore ∧
    w18_pvgrid25.beforeMatchesReference = true ∧ w18_pvgrid25.bitsEqual = true ∧
    w18_pvgrid25.codebookExact = true ∧ w18_pvgrid25.dataDisjoint = true ∧
    w18_pvgrid25.rSatFracBefore = 0 ∧ w18_pvgrid25.rSatFracAfter < 1 / 10 ∧
    (86 / 100 : ℚ) ≤ w18_pvgrid25.klHeldoutAfter ∧ w18_pvgrid25.klHeldoutAfter ≤ 90 / 100 ∧
    |w18_pvgrid25.klHeldoutAfter - w18_pvaball05v0.klHeldoutAfter| < 75 / 10000 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_⟩
  · norm_num [w18_pvgrid25.rSatFracBefore]
  · norm_num [w18_pvgrid25.rSatFracAfter]
  · norm_num [w18_pvgrid25.klHeldoutAfter]
  · norm_num [w18_pvgrid25.klHeldoutAfter]
  · rw [abs_lt]; constructor <;> norm_num [w18_pvgrid25.klHeldoutAfter, w18_pvaball05v0.klHeldoutAfter]

/-- **Predictions `pvtok146-1` and `pvtok292-1` confirmed.** The all-group protocol with the cosine schedule completed at
146 and 292 steps (4096 tokens a step) ends in `[0.92, 0.98]` and `[0.89, 0.93]`. Same arm and controls. -/
theorem pvtok_predictions :
    w18_pvtok146.arm = w18_pv05.arm ∧ w18_pvtok292.arm = w18_pv05.arm ∧
    w18_pvtok146.trainGroups = "c,d,norm,r" ∧ w18_pvtok292.trainGroups = "c,d,norm,r" ∧
    w18_pvtok146.steps = 146 ∧ w18_pvtok292.steps = 292 ∧ w18_pvtok146.stepsCap = 146 ∧ w18_pvtok292.stepsCap = 292 ∧
    w18_pvtok146.tokens = 4096 * w18_pvtok146.steps ∧ w18_pvtok292.tokens = 4096 * w18_pvtok292.steps ∧
    w18_pvtok146.vSteps = 0 ∧ w18_pvtok292.vSteps = 0 ∧
    w18_pvtok146.bitsAfter = w18_pv05.bitsBefore ∧ w18_pvtok292.bitsAfter = w18_pv05.bitsBefore ∧
    w18_pvtok146.beforeMatchesReference = true ∧ w18_pvtok292.beforeMatchesReference = true ∧
    w18_pvtok146.dataDisjoint = true ∧ w18_pvtok292.dataDisjoint = true ∧
    (92 / 100 : ℚ) ≤ w18_pvtok146.klHeldoutAfter ∧ w18_pvtok146.klHeldoutAfter ≤ 98 / 100 ∧
    (89 / 100 : ℚ) ≤ w18_pvtok292.klHeldoutAfter ∧ w18_pvtok292.klHeldoutAfter ≤ 93 / 100 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvtok146.klHeldoutAfter, w18_pvtok292.klHeldoutAfter]

/-- **Finishing is still token-limited at 2.2M tokens.** Held-out KL falls at every step count, by over `4/100` from 146
to 292 steps (a doubling of tokens) and by over eight tenths of that from 292 to 529 steps (`529/292 < 2`, under one
doubling): no flattening between the last two points. The 580-second box, not the scale family, ends the run. -/
theorem pvtok_still_falling :
    w18_pvaball05v0.klHeldoutAfter < w18_pvtok292.klHeldoutAfter ∧
    w18_pvtok292.klHeldoutAfter < w18_pvtok146.klHeldoutAfter ∧
    w18_pvtok146.klHeldoutAfter < w18_pvtok146.klHeldoutBefore ∧
    w18_pvtok292.tokens = 2 * w18_pvtok146.tokens ∧ w18_pvaball05v0.tokens < 2 * w18_pvtok292.tokens ∧
    (4 / 100 : ℚ) < w18_pvtok146.klHeldoutAfter - w18_pvtok292.klHeldoutAfter ∧
    (8 / 10 : ℚ) * (w18_pvtok146.klHeldoutAfter - w18_pvtok292.klHeldoutAfter) <
      w18_pvtok292.klHeldoutAfter - w18_pvaball05v0.klHeldoutAfter := by
  refine ⟨?_, ?_, ?_, rfl, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvaball05v0.klHeldoutAfter, w18_pvtok292.klHeldoutAfter, w18_pvtok146.klHeldoutAfter,
      w18_pvtok146.klHeldoutBefore, w18_pvaball05v0.tokens, w18_pvtok292.tokens, w18_pvtok146.tokens]

/-! ## Finishing throughput: one composition per step and a fused KL head

`w18-pvfast.json` (`pv_finish.py` sha256 `ce1095bf…`, `--compose step --kl fused`) is the all-group protocol with V-steps
off (`w18-pvaball05v0`) on a faster path. `PlaneCompose` composes each coded plane once per step outside the layer
checkpoints, with an analytic backward; `KLHead` saves `(softmax(z_s) − p_t) E` in the forward, so no vocabulary-head
matmul is recomputed. The cost model below counts both changes from the code; the receipt measures the step. -/

/-- Rows per KL chunk and tokens per step (batch 8 of 512-token chunks). -/
def klChunkRows : ℕ := 256
def tokensPerStep : ℕ := 4096
/-- Qwen2.5-0.5B hidden width and vocabulary. -/
def hid : ℕ := 896
def vocab : ℕ := 151936

/-- Vocabulary-head matmuls per KL chunk. Checkpointed `kl_chunk`: teacher and student logits in the forward (2), both
again in the recompute (2), and the student-logit adjoint `G E` in the backward (1). `KLHead`: the two logit matmuls and
the saved adjoint `(q − p) E`, all in the forward (3). -/
def headMatmulsCkpt : ℕ := 5
def headMatmulsFused : ℕ := 3

/-- Plane-composition forwards per coded plane per step: under `--compose layer` the layer checkpoint composes in the
forward and again in the recompute (2); under `--compose step` `PlaneCompose` composes once (1). Each path has one
backward. -/
def composeFwdLayer : ℕ := 2
def composeFwdStep : ℕ := 1

/-- FLOPs of one head matmul, `2 · rows · hid · vocab`. -/
def headMatmulFlops : ℕ := 2 * klChunkRows * hid * vocab

/-- **Cost model.** A step runs 16 KL chunks; the fused head does 48 head matmuls a step against 80, three fifths, and
removes `32 · 2 · 256 · 896 · 151936 ≈ 2.23 · 10¹²` FLOPs a step. Composition forwards halve. -/
theorem pvfast_cost_model :
    tokensPerStep / klChunkRows = 16 ∧
    (tokensPerStep / klChunkRows) * headMatmulsFused = 48 ∧ (tokensPerStep / klChunkRows) * headMatmulsCkpt = 80 ∧
    5 * headMatmulsFused = 3 * headMatmulsCkpt ∧
    (tokensPerStep / klChunkRows) * (headMatmulsCkpt - headMatmulsFused) * headMatmulFlops = 2230430203904 ∧
    2 * composeFwdStep = composeFwdLayer := by
  norm_num [tokensPerStep, klChunkRows, headMatmulsFused, headMatmulsCkpt, headMatmulFlops, hid, vocab,
    composeFwdStep, composeFwdLayer]

/-- **Same arm, same bits, same controls, the fast path.** The receipt's tokens are `4096` a step, its start-of-training
probe passed (losses bit-equal to the `(layer, ckpt)` path in every arm), and no V-step ran. -/
theorem pvfast_same_arm :
    w18_pvfast.arm = w18_pv05.arm ∧ w18_pvfast.trainGroups = "c,d,norm,r" ∧ w18_pvfast.vSteps = 0 ∧
    w18_pvfast.regrid = 0 ∧ w18_pvfast.compose = "step" ∧ w18_pvfast.klMode = "fused" ∧
    w18_pvfast.bitsAfter = w18_pv05.bitsBefore ∧ w18_pvfast.bitsAfter = w18_pvaball05v0.bitsAfter ∧
    w18_pvfast.tokens = tokensPerStep * w18_pvfast.steps ∧
    w18_pvfast.beforeMatchesReference = true ∧ w18_pvfast.bitsEqual = true ∧
    w18_pvfast.codebookExact = true ∧ w18_pvfast.dataDisjoint = true ∧ w18_pvfast.gradEquivOk = true ∧
    w18_pvfast.klHeldoutBefore = w18_pvaball05v0.klHeldoutBefore := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_⟩
  norm_num [w18_pvfast.klHeldoutBefore, w18_pvaball05v0.klHeldoutBefore]

/-- **The fast path computes the same gradient.** On two calibration chunks the trainable gradient differs from the
`(layer, ckpt)` path by a relative L2 under `1/100` and under `3/2` times the rerun floor (the same path run twice);
the composition change alone is within `11/10` of that floor. -/
theorem pvfast_grad_floor :
    w18_pvfast.gradRelRequested < 1 / 100 ∧
    w18_pvfast.gradRelRequested < (3 / 2) * w18_pvfast.gradRelFloor ∧
    w18_pvfast.gradRelComposeOnly < (11 / 10) * w18_pvfast.gradRelFloor ∧
    w18_pvfast.gradRelKlOnly < (3 / 2) * w18_pvfast.gradRelFloor := by
  norm_num [w18_pvfast.gradRelRequested, w18_pvfast.gradRelFloor, w18_pvfast.gradRelComposeOnly,
    w18_pvfast.gradRelKlOnly]

/-- **A step is 1.3 to 1.4 times faster.** Seconds per step fall from `419.1/529` to `416.2/729`, and the same box runs
over `13/10` of the steps. -/
theorem pvfast_faster :
    13 * w18_pvfast.trainS * w18_pvaball05v0.steps < 10 * w18_pvaball05v0.trainS * w18_pvfast.steps ∧
    10 * w18_pvaball05v0.trainS * w18_pvfast.steps < 14 * w18_pvfast.trainS * w18_pvaball05v0.steps ∧
    13 * w18_pvaball05v0.steps < 10 * w18_pvfast.steps := by
  norm_num [w18_pvfast.trainS, w18_pvaball05v0.trainS, w18_pvfast.steps, w18_pvaball05v0.steps]

/-- **Prediction `pvfast-2` refuted on its throughput ranges.** Seconds per step land above the predicted `[0.30, 0.50]`
and steps below the predicted `[800, 1300]`; the held-out KL lands inside its predicted `[0.83, 0.875]`. -/
theorem pvfast_refuted :
    w18_pvfast.trainS / w18_pvfast.steps > 1 / 2 ∧ w18_pvfast.steps < 800 ∧
    (83 / 100 : ℚ) ≤ w18_pvfast.klHeldoutAfter ∧ w18_pvfast.klHeldoutAfter ≤ 875 / 1000 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> norm_num [w18_pvfast.trainS, w18_pvfast.steps, w18_pvfast.klHeldoutAfter]

/-- **More tokens, lower KL at equal bits.** Held-out KL ends `0.0247` below the `(layer, ckpt)` run, over three times
the `0.0075` spread of that protocol (`pvgrid_no_gain`); top-1 rises with it. -/
theorem pvfast_lower :
    w18_pvfast.klHeldoutAfter + 2 / 100 < w18_pvaball05v0.klHeldoutAfter ∧
    w18_pvfast.klHeldoutAfter + 3 * (75 / 10000) < w18_pvaball05v0.klHeldoutAfter ∧
    w18_pvaball05v0.top1HeldoutAfter < w18_pvfast.top1HeldoutAfter := by
  norm_num [w18_pvfast.klHeldoutAfter, w18_pvaball05v0.klHeldoutAfter, w18_pvfast.top1HeldoutAfter,
    w18_pvaball05v0.top1HeldoutAfter]

/-- **The token trend does not flatten at 3.0M tokens.** From 529 to 729 steps is under half a doubling of tokens
(`729² < 2 · 529²`), and the drop over it exceeds half the drop over the first full doubling (146 to 292 steps): the drop
per doubling has not shrunk across the four points 146, 292, 529, 729. -/
theorem pvtok_trend_four :
    w18_pvfast.klHeldoutAfter < w18_pvaball05v0.klHeldoutAfter ∧
    w18_pvaball05v0.klHeldoutAfter < w18_pvtok292.klHeldoutAfter ∧
    w18_pvtok292.klHeldoutAfter < w18_pvtok146.klHeldoutAfter ∧
    w18_pvfast.tokens * w18_pvfast.tokens < 2 * (w18_pvaball05v0.tokens * w18_pvaball05v0.tokens) ∧
    w18_pvtok146.klHeldoutAfter - w18_pvtok292.klHeldoutAfter <
      2 * (w18_pvaball05v0.klHeldoutAfter - w18_pvfast.klHeldoutAfter) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvfast.klHeldoutAfter, w18_pvaball05v0.klHeldoutAfter, w18_pvtok292.klHeldoutAfter,
      w18_pvtok146.klHeldoutAfter, w18_pvfast.tokens, w18_pvaball05v0.tokens]

/-! ### Ten-step profiles of both paths

`w18-pvprof.json` (`(layer, ckpt)`, `pv_finish.py` sha256 `8ece3ced…`) and `w18-pvprof2.json` (`(step, fused)`, sha256
`ce1095bf…`) are `torch.profiler` windows of 10 training steps. `elementwiseS` sums the self device time of the listed
`aten::{mul, cat, sub, add, add_, copy_, fill_, index, gather}` ops, `mmS` of `aten::mm`, `atenSelfS` of every listed aten
op; `selfDeviceS` is kernel time on `pvprof2` and a double-counted op-plus-kernel sum on `pvprof`. -/

/-- **Prediction `pvprof-1` refuted.** The profiled step on the `(layer, ckpt)` path costs over 2 s, outside the
predicted `[0.4, 1.6]`, and the elementwise ops of plane composition take over half the aten device time and more than
every matmul together: the vocabulary head was not the cost. -/
theorem pvprof_refuted :
    w18_pvprof.steps = 10 ∧ 2 * w18_pvprof.steps < w18_pvprof.wallS ∧
    w18_pvprof.atenSelfS < 2 * w18_pvprof.elementwiseS ∧ w18_pvprof.mmS < w18_pvprof.elementwiseS := by
  refine ⟨rfl, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvprof.steps, w18_pvprof.wallS, w18_pvprof.atenSelfS, w18_pvprof.elementwiseS, w18_pvprof.mmS]

/-- **Prediction `pvprof2-1` refuted; its falsifier did not fire.** Kernel time per step lands in the predicted
`[0.35, 0.55]` s, but the profiled step is above the predicted 1.2 s and the elementwise share of kernel time is above the
predicted 35%; it stays under the falsifier's 50%. -/
theorem pvprof2_refuted :
    w18_pvprof2.steps = 10 ∧
    (35 / 100 : ℚ) * w18_pvprof2.steps ≤ w18_pvprof2.selfDeviceS ∧
    w18_pvprof2.selfDeviceS ≤ (55 / 100 : ℚ) * w18_pvprof2.steps ∧
    (6 / 5 : ℚ) * w18_pvprof2.steps < w18_pvprof2.wallS ∧
    (35 / 100 : ℚ) * w18_pvprof2.selfDeviceS < w18_pvprof2.elementwiseS ∧
    w18_pvprof2.elementwiseS < (1 / 2 : ℚ) * w18_pvprof2.selfDeviceS := by
  refine ⟨rfl, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvprof2.steps, w18_pvprof2.selfDeviceS, w18_pvprof2.wallS, w18_pvprof2.elementwiseS]

/-- **What the fast path removed, and what it left.** Elementwise device time falls to under `7/10` of the layer path and
matmul time falls; the `cat` calls of the `fwht128` butterflies rise, because the analytic backward runs `fwht128` twice
per plane. Kernel time covers over `9/10` of the unprofiled step (`416.2 s / 729`): the step is device-bound, so the next
gain has to remove device work, and the butterflies are the largest single block of it. -/
theorem pvprof_fast_path :
    10 * w18_pvprof2.elementwiseS < 7 * w18_pvprof.elementwiseS ∧ w18_pvprof2.mmS < w18_pvprof.mmS ∧
    w18_pvprof.catCalls < w18_pvprof2.catCalls ∧
    (9 / 10 : ℚ) * (w18_pvfast.trainS / w18_pvfast.steps) < w18_pvprof2.selfDeviceS / w18_pvprof2.steps ∧
    w18_pvprof2.selfDeviceS / w18_pvprof2.steps < w18_pvfast.trainS / w18_pvfast.steps := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_pvprof2.elementwiseS, w18_pvprof.elementwiseS, w18_pvprof2.mmS, w18_pvprof.mmS,
      w18_pvprof.catCalls, w18_pvprof2.catCalls, w18_pvfast.trainS, w18_pvfast.steps, w18_pvprof2.selfDeviceS,
      w18_pvprof2.steps]

/-! ### The Hadamard as one matmul (`--hadamard mm`)

`w18-hadmm.json` (`pv_finish.py` sha256 `60b0f6bc…`) is `pvfast`'s protocol with every 128-group's `fwht128` butterfly
replaced by one matmul with the `128 × 128` Sylvester matrix `H^⊗7`, built in float64 and cast. The two are the same
linear map; the start-of-training control checks that in float64 on real planes, then measures what fp32 summation order
and the bf16 cast do to it. -/

/-- The register: a 128-group is `2^7` words, so the Hadamard is seven butterfly stages. -/
def hadWidth : ℕ := 128
def hadStages : ℕ := 7
/-- Bytes per fp32 word. -/
def hadWordBytes : ℕ := 4
/-- Memory traffic per transformed word. Butterfly: each of the 7 stages reads and writes every word, plus the final
`1/√128` scaling pass (read and write). Matmul: the word is read once and written once; the `128 × 128` operand stays in
cache. -/
def butterflyBytesPerWord : ℕ := hadStages * 2 * hadWordBytes + 2 * hadWordBytes
def mmBytesPerWord : ℕ := 2 * hadWordBytes
/-- FLOPs per transformed word under the matmul: one row of `H` against the group, a multiply and an add per entry. -/
def mmFlopsPerWord : ℕ := 2 * hadWidth

/-- **Cost model.** The butterfly moves 64 bytes per word and the matmul 8, eight times fewer; the matmul spends 256 FLOPs
per word, 32 per byte moved, which runs on the tensor cores and is no longer elementwise traffic. -/
theorem hadmm_cost_model :
    2 ^ hadStages = hadWidth ∧ butterflyBytesPerWord = 64 ∧ mmBytesPerWord = 8 ∧
    butterflyBytesPerWord = 8 * mmBytesPerWord ∧ mmFlopsPerWord = 32 * mmBytesPerWord := by
  norm_num [hadStages, hadWidth, butterflyBytesPerWord, mmBytesPerWord, mmFlopsPerWord, hadWordBytes]

/-- **Same arm, same bits, the matmul path.** -/
theorem hadmm_same_arm :
    w18_hadmm.arm = w18_pvfast.arm ∧ w18_hadmm.trainGroups = "c,d,norm,r" ∧ w18_hadmm.vSteps = 0 ∧
    w18_hadmm.compose = "step" ∧ w18_hadmm.klMode = "fused" ∧ w18_hadmm.hadamard = "mm" ∧
    w18_pvfast.hadamard = "butterfly" ∧ w18_hadmm.bitsAfter = w18_pvfast.bitsAfter ∧
    w18_hadmm.tokens = tokensPerStep * w18_hadmm.steps ∧
    w18_hadmm.beforeMatchesReference = true ∧ w18_hadmm.bitsEqual = true ∧ w18_hadmm.codebookExact = true ∧
    w18_hadmm.dataDisjoint = true ∧ w18_hadmm.gradEquivOk = true ∧
    w18_hadmm.klHeldoutBefore = w18_pvfast.klHeldoutBefore := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_⟩
  norm_num [w18_hadmm.klHeldoutBefore, w18_pvfast.klHeldoutBefore]

/-- **The matmul is the butterfly's function.** In float64 on eight coded planes, forward and the `r, c, d` adjoints
agree to a relative `7.1 · 10⁻¹⁶`. With an fp32 student the model loss moves by under `10⁻⁵` relative and the trainable
gradient by under `1/100`, within ten times the fp32 rerun floor (atomic reordering alone). -/
theorem hadmm_same_function :
    w18_hadmm.float64UnitMaxRel < 1 / 10 ^ 12 ∧ w18_hadmm.lossRelFp32Had < 1 / 10 ^ 5 ∧
    w18_hadmm.gradRelFp32Had < 1 / 100 ∧ w18_hadmm.gradRelFp32Had < 10 * w18_hadmm.gradRelFp32Floor := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_hadmm.float64UnitMaxRel, w18_hadmm.lossRelFp32Had, w18_hadmm.gradRelFp32Had,
      w18_hadmm.gradRelFp32Floor]

/-- **What the bf16 student does to an fp32 rounding difference.** The matmul moves no fp32 weight by more than `10⁻⁶`
relative, and its bf16 relative L2 change is below a seeded `2⁻²²` relative jitter's; it still flips over thirteen times
as many bf16 words (under one eighth of them zero to nonzero), because its error is absolute per 128-group and lands on
small words. The bf16 loss moves over 800 times the fp32 loss and over five times the jitter's, the gradient over three
times the jitter's; the bf16 cast alone moves the loss more than the matmul does. -/
theorem hadmm_bf16_amplification :
    w18_hadmm.wRelMaxMm < 1 / 10 ^ 6 ∧ w18_hadmm.bf16RelL2Mm < w18_hadmm.bf16RelL2Jitter ∧
    13 * w18_hadmm.bf16FlipsJitter < w18_hadmm.bf16FlipsMm ∧ 8 * w18_hadmm.zeroToNonzeroMm < w18_hadmm.bf16FlipsMm ∧
    800 * w18_hadmm.lossRelFp32Had < w18_hadmm.lossRelHadOnly ∧
    5 * w18_hadmm.lossRelJitter < w18_hadmm.lossRelHadOnly ∧
    3 * w18_hadmm.gradRelJitter < w18_hadmm.gradRelHadOnly ∧
    w18_hadmm.lossRelHadOnly < w18_hadmm.lossRelBf16VsFp32 ∧ w18_hadmm.lossRelRequested = w18_hadmm.lossRelHadOnly := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, rfl⟩ <;>
    norm_num [w18_hadmm.wRelMaxMm, w18_hadmm.bf16RelL2Mm, w18_hadmm.bf16RelL2Jitter, w18_hadmm.bf16FlipsJitter,
      w18_hadmm.bf16FlipsMm, w18_hadmm.zeroToNonzeroMm, w18_hadmm.lossRelFp32Had, w18_hadmm.lossRelHadOnly,
      w18_hadmm.lossRelJitter, w18_hadmm.gradRelJitter, w18_hadmm.gradRelHadOnly, w18_hadmm.lossRelBf16VsFp32]

/-- **The guessed tolerances were wrong, not the path.** Prediction `hadmm-smoke-1` set the bf16 gradient
tolerance at `1/100` and `hadmm-full-2` the fp32 one at `1/1000`; the measured values exceed both, while the float64 identity
holds (`hadmm_same_function`). The gate now derives from the float64 identity and the measured fp32 floor. -/
theorem hadmm_guessed_tolerances :
    1 / 100 < w18_hadmm.gradRelRequested ∧ 1 / 1000 < w18_hadmm.gradRelFp32Had := by
  refine ⟨?_, ?_⟩ <;> norm_num [w18_hadmm.gradRelRequested, w18_hadmm.gradRelFp32Had]

/-- **A step is 1.1 to 1.2 times faster than `pvfast`.** Seconds per step fall from `416.2/729` to `417.4/871`, a saving
between `9/100` and `1/10` s per step. -/
theorem hadmm_faster :
    11 * w18_hadmm.trainS * w18_pvfast.steps < 10 * w18_pvfast.trainS * w18_hadmm.steps ∧
    10 * w18_pvfast.trainS * w18_hadmm.steps < 12 * w18_hadmm.trainS * w18_pvfast.steps ∧
    9 / 100 < w18_pvfast.trainS / w18_pvfast.steps - w18_hadmm.trainS / w18_hadmm.steps ∧
    w18_pvfast.trainS / w18_pvfast.steps - w18_hadmm.trainS / w18_hadmm.steps < 1 / 10 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> norm_num [w18_hadmm.trainS, w18_pvfast.trainS, w18_hadmm.steps, w18_pvfast.steps]

/-- **More tokens, lower KL; the fifth point lies below the earlier rate.** Held-out KL ends `0.0093` below `pvfast`,
over the `0.0075` protocol spread, and top-1 rises. The step from 729 to 871 is over a quarter of a token doubling
(`871⁴ > 2 · 729⁴`), and its drop is under a quarter of the first doubling's (146 to 292). -/
theorem hadmm_lower :
    w18_hadmm.klHeldoutAfter + 75 / 10000 < w18_pvfast.klHeldoutAfter ∧
    w18_pvfast.top1HeldoutAfter < w18_hadmm.top1HeldoutAfter ∧
    2 * w18_pvfast.tokens ^ 4 < w18_hadmm.tokens ^ 4 ∧
    4 * (w18_pvfast.klHeldoutAfter - w18_hadmm.klHeldoutAfter) <
      w18_pvtok146.klHeldoutAfter - w18_pvtok292.klHeldoutAfter := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_hadmm.klHeldoutAfter, w18_pvfast.klHeldoutAfter, w18_hadmm.top1HeldoutAfter,
      w18_pvfast.top1HeldoutAfter, w18_pvfast.tokens, w18_hadmm.tokens, w18_pvtok146.klHeldoutAfter,
      w18_pvtok292.klHeldoutAfter]

end Perf.PVFinish
