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

end Perf.PVFinish
