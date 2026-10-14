import Perf.Receipts.Onebit

/-!
# Model size and the 2-plane E8 `-resq` arm: verdicts

The arm (`tools/onebit/fold_transfer.py`, `had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-resq-qmlp-nb`)
is fitted to Qwen2.5-0.5B (24 layers) and Qwen2.5-1.5B (28 layers) on one protocol: 50 calibration chunks of 512
tokens, held-out KL on `calib-c4` chunks `[50, 90)`, in-sample KL on `[0, 20)`, each model against its own Q4_K_M
(imatrix GGUF). The receipt checker (`tools/rooting/receipt_lean.py`) refuses a receipt off that protocol.

The arm's gate/up second plane is a fixed row count (`--qmlp-rows`, 416), so at 1.5B (8960 FFN rows against 4864) it
covers a smaller fraction and the code stores 1.142 b/w against 1.178. A third fit widens it to 792 rows, matching
0.5B's stored sign bits per weight to `10⁻⁴`.

Switch-rule run 2 asks whether the arm's KL gap shrinks with model size (literature basis: one-bit post-training
log-perplexity gaps fall from OPT-1.3B to OPT-2.7B, `docs/evidence/receipts/onebit/w18-literature.tsv`). Prediction
`size15-arm-1` (falsifier: `KL_1.5B / KL_0.5B ≤ 0.95`) is refuted: held-out KL rises with size, at the original and
at the matched bit count, while in-sample KL falls and Q4_K_M stays flat. The extra second-plane rows lower in-sample
KL and raise held-out KL. The arm's held-out gap grows with size on this pair; the switch rule fires.

Grades: every statement is a `proved` arithmetic fact about `measured` constants (`Perf/Receipts/Onebit.lean`).
-/

namespace Perf.SizeTrend

open Receipts

/-- One arm, three fits: the same variant string on both models; 1.5B fitted twice. -/
theorem size_same_arm :
    w18_size05.arm = w18_size15.arm ∧ w18_size15.arm = w18_size15m.arm ∧
    w18_size05.layers = 24 ∧ w18_size15.layers = 28 ∧ w18_size15m.layers = 28 ∧
    w18_size15.linearParams = w18_size15m.linearParams ∧
    w18_size15.qmlpRows = 416 ∧ w18_size15m.qmlpRows = 792 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- **Bits.** The fixed 416-row second plane stores over `1/50` fewer sign bits per weight at 1.5B; 792 rows match
0.5B to `10⁻⁴`. -/
theorem size_bits :
    (w18_size15.bitsSign : ℚ) / w18_size15.params + 1 / 50 <
      (w18_size05.bitsSign : ℚ) / w18_size05.params ∧
    |(w18_size15m.bitsSign : ℚ) / w18_size15m.params - (w18_size05.bitsSign : ℚ) / w18_size05.params| <
      1 / 10000 ∧
    w18_size15.bpwLinears < w18_size15m.bpwLinears ∧ w18_size15m.bpwLinears < w18_size05.bpwLinears := by
  refine ⟨?_, ?_, ?_⟩
  · norm_num [w18_size15.bitsSign, w18_size15.params, w18_size05.bitsSign, w18_size05.params]
  · rw [abs_lt]
    norm_num [w18_size15m.bitsSign, w18_size15m.params, w18_size05.bitsSign, w18_size05.params]
  · norm_num [w18_size15.bpwLinears, w18_size15m.bpwLinears, w18_size05.bpwLinears]

/-- **Prediction `size15-arm-1` refuted.** Held-out KL rises from 0.5B to 1.5B by more than `13/100` at 416 rows and
more than `15/100` at matched bits, far from the `0.95` falsifier. -/
theorem size_heldout_rises :
    (113 / 100 : ℚ) * w18_size05.klHeldout < w18_size15.klHeldout ∧
    (115 / 100 : ℚ) * w18_size05.klHeldout < w18_size15m.klHeldout ∧
    (95 / 100 : ℚ) * w18_size05.klHeldout < w18_size15.klHeldout := by
  norm_num [w18_size05.klHeldout, w18_size15.klHeldout, w18_size15m.klHeldout]

/-- **Bits are not the gap** (prediction `size15-matched-1` refuted). Widening the second plane to 792 rows lowers
in-sample KL and raises held-out KL. -/
theorem size_more_rows_overfit :
    w18_size15m.klInsample < w18_size15.klInsample ∧ w18_size15.klHeldout < w18_size15m.klHeldout ∧
    w18_size15m.top1Heldout < w18_size15.top1Heldout := by
  norm_num [w18_size15m.klInsample, w18_size15.klInsample, w18_size15.klHeldout, w18_size15m.klHeldout,
    w18_size15m.top1Heldout, w18_size15.top1Heldout]

/-- **Generalisation gap grows with size.** In-sample KL falls from 0.5B to 1.5B while held-out rises; the
held-out/in-sample ratio exceeds 5 at 0.5B and 7 at 1.5B. -/
theorem size_gap_grows :
    w18_size15.klInsample < w18_size05.klInsample ∧ w18_size05.klHeldout < w18_size15.klHeldout ∧
    5 * w18_size05.klInsample < w18_size05.klHeldout ∧ w18_size05.klHeldout < 6 * w18_size05.klInsample ∧
    7 * w18_size15.klInsample < w18_size15.klHeldout ∧ 7 * w18_size15m.klInsample < w18_size15m.klHeldout := by
  norm_num [w18_size15.klInsample, w18_size05.klInsample, w18_size05.klHeldout, w18_size15.klHeldout,
    w18_size15m.klInsample, w18_size15m.klHeldout]

/-- **Standing against Q4_K_M.** Q4_K_M's held-out KL moves by under 5%; the arm's ratio to it is between 63 and 64 at
0.5B and between 69 and 70 at 1.5B. -/
theorem size_q4km_ratio :
    w18_size05.q4kmKlHeldout < w18_size15.q4kmKlHeldout ∧
    w18_size15.q4kmKlHeldout < (105 / 100 : ℚ) * w18_size05.q4kmKlHeldout ∧
    63 * w18_size05.q4kmKlHeldout < w18_size05.klHeldout ∧ w18_size05.klHeldout < 64 * w18_size05.q4kmKlHeldout ∧
    69 * w18_size15.q4kmKlHeldout < w18_size15.klHeldout ∧ w18_size15.klHeldout < 70 * w18_size15.q4kmKlHeldout := by
  norm_num [w18_size05.q4kmKlHeldout, w18_size15.q4kmKlHeldout, w18_size05.klHeldout, w18_size15.klHeldout]

end Perf.SizeTrend
