import Perf.Receipts.Onebit

/-!
# End-loss-weighted Grams in the code solve of the 2-plane E8 `-resq` arm: verdicts

`tools/onebit/fold_transfer.py` builds the arm on Qwen2.5-0.5B exactly as the size-trend receipt
(`w18-size05-n50.json`: 50 calibration chunks, held-out `calib-c4` `[50, 90)`, in-sample `[0, 20)`). The variant token
`-guided[a|b]NN` replaces each layer's input Gram `H = Σ_t x_t x_tᵀ` by the end-loss-weighted Gram of GuidedQuant
(Kim et al., arXiv 2505.07004, §3.2 Eq. 7): `H = Σ_t a_t x_t x_tᵀ`, with token weight `a_t` the squared norm of the
model's output-KL gradient at the layer's output (true Fisher, `K = 4` sampled targets) and row weights `b` the
per-output-row gradient energy in place of the base row proxy. `NN` tempers the token weights to `a_t^(NN/100)`,
renormalised to mean 1. Part `a` keeps token weights only, part `b` row weights only; the default `ab` keeps both.

Three receipts:

- `w18-guided05.json` (sha256 `0a68d4ae…`, seed 0): base, `γ = 1`, `γ = 1/2`. Prediction `guided05-1` confirmed.
- `w18-guidedsweep.json` (sha256 `28b9b4c0…`, which adds only the part parsing, seed 0): `γ ∈ {1/4, 3/4}`, part `a` at
  `γ = 1/2`, part `b`. Prediction `guidedsweep-1` refuted: `γ = 1/4` and part `b` land above its 1.25 falsifier.
- `w18-guidedseed.json` (sha256 `28b9b4c0…`, seeds 1, 2; seed `s` permutes the calibration chunks): base and `γ = 1/2`.
  Prediction `guidedseed-1` refuted: the base arm's seed spread is 0.025, under the predicted 0.05.

Reading: the weighting is a large, unordered perturbation of the code solve, not a smooth lever. Held-out KL alternates
in sign along `γ`, the Kish effective sample size `(Σ a)² / (n Σ a²)` does not order it, neither part alone improves on
the base, and `γ = 1/2` nearly triples the calibration-order spread. Over three seeds `γ = 1/2` is lower by 0.017 on average,
better on two seeds and worse than every base seed on the third.

Grades: every statement is a `proved` arithmetic fact about `measured` constants (`Perf/Receipts/Onebit.lean`); the
receipt checks (calibration count, held-out and in-sample windows, seeds, variant set) are `tested` booleans checked by
`tools/rooting/receipt_lean.py (unpublished)` before the constants are emitted.
-/

namespace Perf.Guided

open Receipts

/-- **Same arm, same bits.** The base variant reproduces the size-trend receipt's held-out KL to the last digit, and
every guided variant in all three receipts stores exactly the base's bits: the weighting changes which code is chosen,
not its size. -/
theorem guided_same_bits :
    w18_guided05.arm = w18_size05.arm ∧ w18_guided05.baseBits = w18_size05.bits ∧
    w18_guided05.gab100Bits = w18_size05.bits ∧ w18_guided05.gab50Bits = w18_size05.bits ∧
    w18_guidedsweep.gab25Bits = w18_size05.bits ∧ w18_guidedsweep.gab75Bits = w18_size05.bits ∧
    w18_guidedsweep.ga50Bits = w18_size05.bits ∧ w18_guidedsweep.gb100Bits = w18_size05.bits ∧
    w18_guidedseed.baseBits = w18_size05.bits ∧ w18_guidedseed.gab50Bits = w18_size05.bits ∧
    w18_guided05.baseKlHeldout = w18_size05.klHeldout := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_⟩
  norm_num [w18_guided05.baseKlHeldout, w18_size05.klHeldout]

/-- **Prediction `guided05-1` confirmed.** Untempered weighting lands in `[1.20, 1.70]`, above the base (the falsifier's
first clause fails); `γ = 1/2` lands in `[1.10, 1.30]`, inside the falsifier's `[1.05, 1.35]`. -/
theorem guided05_confirmed :
    (6 / 5 : ℚ) ≤ w18_guided05.gab100KlHeldout ∧ w18_guided05.gab100KlHeldout ≤ 17 / 10 ∧
    w18_guided05.baseKlHeldout ≤ w18_guided05.gab100KlHeldout ∧
    (11 / 10 : ℚ) ≤ w18_guided05.gab50KlHeldout ∧ w18_guided05.gab50KlHeldout ≤ 13 / 10 := by
  norm_num [w18_guided05.gab100KlHeldout, w18_guided05.baseKlHeldout, w18_guided05.gab50KlHeldout]

/-- **Prediction `guidedsweep-1` refuted.** Its falsifier (any variant above 1.25) fires twice: `γ = 1/4` and row
weights alone. -/
theorem guidedsweep_refuted :
    (5 / 4 : ℚ) < w18_guidedsweep.gab25KlHeldout ∧ (5 / 4 : ℚ) < w18_guidedsweep.gb100KlHeldout := by
  norm_num [w18_guidedsweep.gab25KlHeldout, w18_guidedsweep.gb100KlHeldout]

/-- **Non-monotone in `γ`.** At seed 0 the sign of the change from the base alternates along `γ = 1/4, 1/2, 3/4, 1`:
worse by over 0.13, better, better, worse. -/
theorem guided_nonmonotone :
    w18_guided05.baseKlHeldout + 13 / 100 < w18_guidedsweep.gab25KlHeldout ∧
    w18_guided05.gab50KlHeldout < w18_guided05.baseKlHeldout ∧
    w18_guidedsweep.gab75KlHeldout < w18_guided05.baseKlHeldout ∧
    w18_guided05.baseKlHeldout < w18_guided05.gab100KlHeldout := by
  norm_num [w18_guided05.baseKlHeldout, w18_guidedsweep.gab25KlHeldout, w18_guided05.gab50KlHeldout,
    w18_guidedsweep.gab75KlHeldout, w18_guided05.gab100KlHeldout]

/-- **Effective sample size does not order the outcome.** `γ = 3/4` keeps under 1% Kish effective sample size at its
worst stage and still beats the base; `γ = 1/4` keeps over 3/4 at every stage and loses by over 0.13. -/
theorem guided_ess_not_order :
    w18_guidedsweep.gab75EssMin < 1 / 100 ∧
    w18_guidedsweep.gab75KlHeldout < w18_guided05.baseKlHeldout ∧
    (3 / 4 : ℚ) < w18_guidedsweep.gab25EssMin ∧
    w18_guided05.baseKlHeldout + 13 / 100 < w18_guidedsweep.gab25KlHeldout := by
  norm_num [w18_guidedsweep.gab75EssMin, w18_guidedsweep.gab75KlHeldout, w18_guided05.baseKlHeldout,
    w18_guidedsweep.gab25EssMin, w18_guidedsweep.gab25KlHeldout]

/-- **Neither part alone.** Token weights alone (`γ = 1/2`) and row weights alone each land above the base; only both
together at `γ = 1/2` land below it at seed 0. -/
theorem guided_parts :
    w18_guided05.baseKlHeldout + 6 / 100 < w18_guidedsweep.ga50KlHeldout ∧
    w18_guided05.baseKlHeldout + 18 / 100 < w18_guidedsweep.gb100KlHeldout ∧
    w18_guided05.gab50KlHeldout < w18_guided05.baseKlHeldout ∧
    w18_guidedsweep.ga50Gamma = 1 / 2 ∧ w18_guidedsweep.ga50Part = "a" ∧ w18_guidedsweep.gb100Part = "b" := by
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl⟩ <;>
    norm_num [w18_guided05.baseKlHeldout, w18_guidedsweep.ga50KlHeldout, w18_guidedsweep.gb100KlHeldout,
      w18_guided05.gab50KlHeldout, w18_guidedsweep.ga50Gamma]

/-- **The base solve is stable across calibration order.** Seeds 0, 1, 2 of the base arm lie within 0.025 of each
other (`[1.16, 1.19]`); prediction `guidedseed-1`'s first clause (spread `≥ 0.05`) fails. -/
theorem base_seed_spread :
    (116 / 100 : ℚ) < w18_guided05.baseKlHeldout ∧ w18_guided05.baseKlHeldout < w18_guidedseed.baseKlHeldoutS2 ∧
    w18_guidedseed.baseKlHeldoutS2 < w18_guidedseed.baseKlHeldoutS1 ∧
    w18_guidedseed.baseKlHeldoutS1 < w18_guided05.baseKlHeldout + 25 / 1000 := by
  norm_num [w18_guided05.baseKlHeldout, w18_guidedseed.baseKlHeldoutS1, w18_guidedseed.baseKlHeldoutS2]

/-- **The weighting nearly triples the spread.** At `γ = 1/2` seeds 0 and 2 differ by over 0.07, over 2.9 times the
base arm's 0.0247 spread across the same three seeds. -/
theorem guided_seed_spread :
    w18_guided05.gab50KlHeldout + 7 / 100 < w18_guidedseed.gab50KlHeldoutS2 ∧
    (29 / 10 : ℚ) * (w18_guidedseed.baseKlHeldoutS1 - w18_guided05.baseKlHeldout) <
      w18_guidedseed.gab50KlHeldoutS2 - w18_guided05.gab50KlHeldout := by
  norm_num [w18_guided05.gab50KlHeldout, w18_guidedseed.gab50KlHeldoutS2, w18_guidedseed.baseKlHeldoutS1,
    w18_guided05.baseKlHeldout]

/-- **Mean gain over three seeds is under 0.02, and not robust.** The paired mean difference (`γ = 1/2` minus base) over
seeds 0, 1, 2 lies in `(-0.02, -0.01)`; seeds 0 and 1 are lower than their base, seed 2 is higher than every base seed. -/
theorem guided_seed_mean :
    -(6 / 100 : ℚ) < (w18_guided05.gab50KlHeldout + w18_guidedseed.gab50KlHeldoutS1 + w18_guidedseed.gab50KlHeldoutS2) -
      (w18_guided05.baseKlHeldout + w18_guidedseed.baseKlHeldoutS1 + w18_guidedseed.baseKlHeldoutS2) ∧
    (w18_guided05.gab50KlHeldout + w18_guidedseed.gab50KlHeldoutS1 + w18_guidedseed.gab50KlHeldoutS2) -
      (w18_guided05.baseKlHeldout + w18_guidedseed.baseKlHeldoutS1 + w18_guidedseed.baseKlHeldoutS2) < -(3 / 100) ∧
    w18_guided05.gab50KlHeldout < w18_guided05.baseKlHeldout ∧
    w18_guidedseed.gab50KlHeldoutS1 < w18_guidedseed.baseKlHeldoutS1 ∧
    w18_guidedseed.baseKlHeldoutS1 < w18_guidedseed.gab50KlHeldoutS2 := by
  norm_num [w18_guided05.gab50KlHeldout, w18_guidedseed.gab50KlHeldoutS1, w18_guidedseed.gab50KlHeldoutS2,
    w18_guided05.baseKlHeldout, w18_guidedseed.baseKlHeldoutS1, w18_guidedseed.baseKlHeldoutS2]

/-- **Held-out falls while in-sample rises.** At `γ = 1/2` in-sample KL rises above the base on every seed: the
weighting moves the code away from the unweighted calibration objective. -/
theorem guided_insample_rises :
    w18_guided05.baseKlInsample < w18_guided05.gab50KlInsample ∧
    w18_guidedseed.baseKlInsampleS1 < w18_guidedseed.gab50KlInsampleS1 ∧
    w18_guidedseed.baseKlInsampleS2 < w18_guidedseed.gab50KlInsampleS2 ∧
    w18_guided05.gab50KlInsample < w18_guided05.gab100KlInsample := by
  norm_num [w18_guided05.baseKlInsample, w18_guided05.gab50KlInsample, w18_guidedseed.baseKlInsampleS1,
    w18_guidedseed.gab50KlInsampleS1, w18_guidedseed.baseKlInsampleS2, w18_guidedseed.gab50KlInsampleS2,
    w18_guided05.gab100KlInsample]

end Perf.Guided
