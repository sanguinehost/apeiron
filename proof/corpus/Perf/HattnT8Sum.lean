import Perf.HattnT8

/-!
# attention-cut T8 hot/cold page summary on Qwen2.5-7B-Instruct-1M

the hattn-decode-tail note (hot/cold page summary section): the flat arm `grp-fix2048-r8` with every omitted
far page replaced by a summary (page mean value, and a scalar mass), the 64 tail positions before
`t ∈ {262144, 524288}` of the dump `g7-1m`, instrument `attn_cut_probe.hd128qs`
(`KV_TAIL_SUM`, output `g11-t<t>-sum<mode>.out`). Mode 0: mass `64·exp(mean page score)`. Mode 1: exact page mass
`Σ_j w_j` (ceiling for a scalar-mass summary). Mode 2: mass `64·exp(μ + σ²/2)`. All three add `mass · mean(v)`.
Measured figures are `model` inputs: mean whole-model KLD and bytes/dense in ten-thousandths. Baselines are
`Perf.HattnT8Q9`'s flat arm. Predictions: `PREDICTIONS-T8-SUM.md`, sha256 `3396f4ba…`. rustc, LLVM, the f32 dump
kernels and the instrument are outside every statement; one prompt, one model.
-/

namespace Perf

namespace HattnT8Sum

/-- Flat-arm baseline (no summary) at `t = 262144, 524288`. -/
def flat : List ℕ := [386, 3370]

/-- Mode 0 (mass from the mean page score) at `t = 262144, 524288`. -/
def mode0 : List ℕ := [433, 2438]

/-- Mode 1 (exact page mass) at the same `t`. -/
def mode1 : List ℕ := [542, 2389]

/-- Mode 2 (mean plus half the page score variance) at the same `t`. -/
def mode2 : List ℕ := [34111, 21127]

/-- **S1 refuted.** The exact-mass summary does not halve the flat KLD: `0.0542` against `0.0193` at 262144 (worse than
no summary), `0.2389` against `0.1685` at 524288. -/
theorem s1_refuted :
    ¬ (mode1.getD 0 0 ≤ 193) ∧ ¬ (mode1.getD 1 0 ≤ 1685) ∧ flat.getD 0 0 < mode1.getD 0 0 := by decide

/-- **S2 refuted in its second inequality.** At 524288 mode 1 `≤` mode 2 holds, mode 2 `≤` mode 0 does not: the variance
term makes the estimate worse by 8.7×. -/
theorem s2_refuted :
    mode1.getD 1 0 ≤ mode2.getD 1 0 ∧ ¬ (mode2.getD 1 0 ≤ mode0.getD 1 0) ∧ 8 * mode0.getD 1 0 < mode2.getD 1 0 := by
  decide

/-- **S3 held.** No mode passes the `0.05` gate at 524288. -/
theorem s3_held : ∀ k ∈ [mode0.getD 1 0, mode1.getD 1 0, mode2.getD 1 0], 500 < k := by decide

/-- **S4 held (derived).** Bound scan of 4 vectors per page (`pages 524288 = 8187`), refresh 8, near 288: the summary
arm reads `70188` vectors per period against `8 · 2 · 524288` dense, under `0.01`. -/
theorem s4_held :
    HattnT8.pages 524288 = 8187 ∧ HattnT8.period 2048 288 8 (4 * 8187) = 70188 ∧
      100 * 70188 < 8 * 2 * 524288 := by
  decide

/-! ## Findings -/

/-- **The mass is not the bottleneck.** Exact mass (mode 1) and mean-score mass (mode 0) differ by at most `0.005` at
524288 (`0.2389`, `0.2438`): what the omitted pages carry is value mixing inside the page, which a page-mean value
does not hold. -/
theorem mass_not_bottleneck : mode0.getD 1 0 ≤ mode1.getD 1 0 + 50 ∧ mode1.getD 1 0 ≤ mode0.getD 1 0 := by decide

/-- **A mean-value summary buys a quarter at 524288 and costs at 262144.** Modes 0 and 1 lower the flat KLD at
524288 (`0.3370` to `0.2438`, `0.2389`) and raise it at 262144 (`0.0386` to `0.0433`, `0.0542`). -/
theorem summary_gain_and_loss :
    mode0.getD 1 0 < flat.getD 1 0 ∧ mode1.getD 1 0 < flat.getD 1 0 ∧
      flat.getD 0 0 < mode0.getD 0 0 ∧ flat.getD 0 0 < mode1.getD 0 0 ∧ 3 * mode1.getD 1 0 < 2 * flat.getD 1 0 + 1000 := by
  decide

end HattnT8Sum

end Perf
