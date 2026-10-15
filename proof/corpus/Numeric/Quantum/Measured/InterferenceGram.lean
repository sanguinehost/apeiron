import Perf.Receipts.Onebit

/-!
# Q1: the interference Gram does not predict group KL (verdict: refuted)

Board: `quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs`. Epic: `docs/epics/quantum-register.md` row Q1.

`examples/qarch_q1.rs` substitutes HadRtn-interpolated linears (weight `(1 − α) W + α Q(W)`, α ∈ {1/8, 1/4, 1/2, 1})
per layer into Qwen2.5-0.5B f16 on the Rust path (`Engine::step_logits`), takes the amplitude error
`δψ_l = √q_l − √p` per layer on three held-out windows of 64 tokens, and predicts each layer group's KL as
`2 ‖Σ_{l ∈ g} δψ_l‖²` (the second-order identity `Quantum.AmplitudeKL.kl_second_order`, summed). The receipt
`docs/evidence/receipts/onebit/qarch-q1.json` (constants `Receipts.qarch_q1`, bound to its ledger sha256 by
`tools/rooting/receipt_lean.py`) gives, per α and window, the measured and predicted KL of three groups: all 24
layers, the prefix of layers 0–19, and the suffix of layers 16–23.

The epic's falsifier states: if the Gram does not predict the measured group ratios within the run spread, the
saturation is softmax nonlinearity and the amplitude reading is refuted. It is refuted at every α and every window:

* `all_overpredicted`: the measured whole-model KL is below `3/4` of the prediction in all 12 cells.
* `prefix_over_suffix_under`: in all 12 cells the prefix is over-predicted and the suffix under-predicted, so no
  single rescaling of the Gram fits both groups.
* `diagonal_share_falls`: the Gram's diagonal share falls with α (`0.583, 0.565, 0.380, 0.285`): the cross-layer
  terms grow with the perturbation, while the measured KL saturates below them.

Not claimed: anything about a quantiser's rate or KL at 1.18 b/w; the receipt is a measurement of the substitution
family above on three windows of the 0.5B model.
-/

namespace Quantum.InterferenceGram

open Receipts.qarch_q1

/-- (measured, predicted) KL of all 24 layers, per (α index, window), row-major. -/
def allCells : List (ℚ × ℚ) :=
  [(a0W0AllMeas, a0W0AllPred), (a0W1AllMeas, a0W1AllPred), (a0W2AllMeas, a0W2AllPred), (a1W0AllMeas, a1W0AllPred),
    (a1W1AllMeas, a1W1AllPred), (a1W2AllMeas, a1W2AllPred), (a2W0AllMeas, a2W0AllPred), (a2W1AllMeas, a2W1AllPred),
    (a2W2AllMeas, a2W2AllPred), (a3W0AllMeas, a3W0AllPred), (a3W1AllMeas, a3W1AllPred), (a3W2AllMeas, a3W2AllPred)]

/-- (measured, predicted) KL of the prefix of layers 0–19. -/
def prefixCells : List (ℚ × ℚ) :=
  [(a0W0Prefix20Meas, a0W0Prefix20Pred), (a0W1Prefix20Meas, a0W1Prefix20Pred), (a0W2Prefix20Meas, a0W2Prefix20Pred),
    (a1W0Prefix20Meas, a1W0Prefix20Pred), (a1W1Prefix20Meas, a1W1Prefix20Pred), (a1W2Prefix20Meas,
    a1W2Prefix20Pred), (a2W0Prefix20Meas, a2W0Prefix20Pred), (a2W1Prefix20Meas, a2W1Prefix20Pred),
    (a2W2Prefix20Meas, a2W2Prefix20Pred), (a3W0Prefix20Meas, a3W0Prefix20Pred), (a3W1Prefix20Meas,
    a3W1Prefix20Pred), (a3W2Prefix20Meas, a3W2Prefix20Pred)]

/-- (measured, predicted) KL of the suffix of layers 16–23. -/
def suffixCells : List (ℚ × ℚ) :=
  [(a0W0Suffix16Meas, a0W0Suffix16Pred), (a0W1Suffix16Meas, a0W1Suffix16Pred), (a0W2Suffix16Meas, a0W2Suffix16Pred),
    (a1W0Suffix16Meas, a1W0Suffix16Pred), (a1W1Suffix16Meas, a1W1Suffix16Pred), (a1W2Suffix16Meas,
    a1W2Suffix16Pred), (a2W0Suffix16Meas, a2W0Suffix16Pred), (a2W1Suffix16Meas, a2W1Suffix16Pred),
    (a2W2Suffix16Meas, a2W2Suffix16Pred), (a3W0Suffix16Meas, a3W0Suffix16Pred), (a3W1Suffix16Meas,
    a3W1Suffix16Pred), (a3W2Suffix16Meas, a3W2Suffix16Pred)]

/-- **The Gram over-predicts the whole-model KL by more than 4/3 in every (α, window) cell.** -/
theorem all_overpredicted : ∀ c ∈ allCells, c.1 < 3 / 4 * c.2 := by
  norm_num [allCells, a0W0AllMeas, a0W1AllMeas, a0W2AllMeas, a1W0AllMeas, a1W1AllMeas, a1W2AllMeas, a2W0AllMeas,
    a2W1AllMeas, a2W2AllMeas, a3W0AllMeas, a3W1AllMeas, a3W2AllMeas, a0W0AllPred, a0W1AllPred, a0W2AllPred,
    a1W0AllPred, a1W1AllPred, a1W2AllPred, a2W0AllPred, a2W1AllPred, a2W2AllPred, a3W0AllPred, a3W1AllPred,
    a3W2AllPred]

/-- **The prefix is over-predicted and the suffix under-predicted in every cell**: the error is not a scale. -/
theorem prefix_over_suffix_under :
    (∀ c ∈ prefixCells, c.1 < c.2) ∧ (∀ c ∈ suffixCells, c.2 < c.1) := by
  norm_num [prefixCells, suffixCells, a0W0Prefix20Meas, a0W1Prefix20Meas, a0W2Prefix20Meas, a1W0Prefix20Meas,
    a1W1Prefix20Meas, a1W2Prefix20Meas, a2W0Prefix20Meas, a2W1Prefix20Meas, a2W2Prefix20Meas, a3W0Prefix20Meas,
    a3W1Prefix20Meas, a3W2Prefix20Meas, a0W0Prefix20Pred, a0W1Prefix20Pred, a0W2Prefix20Pred, a1W0Prefix20Pred,
    a1W1Prefix20Pred, a1W2Prefix20Pred, a2W0Prefix20Pred, a2W1Prefix20Pred, a2W2Prefix20Pred, a3W0Prefix20Pred,
    a3W1Prefix20Pred, a3W2Prefix20Pred, a0W0Suffix16Meas, a0W1Suffix16Meas, a0W2Suffix16Meas, a1W0Suffix16Meas,
    a1W1Suffix16Meas, a1W2Suffix16Meas, a2W0Suffix16Meas, a2W1Suffix16Meas, a2W2Suffix16Meas, a3W0Suffix16Meas,
    a3W1Suffix16Meas, a3W2Suffix16Meas, a0W0Suffix16Pred, a0W1Suffix16Pred, a0W2Suffix16Pred, a1W0Suffix16Pred,
    a1W1Suffix16Pred, a1W2Suffix16Pred, a2W0Suffix16Pred, a2W1Suffix16Pred, a2W2Suffix16Pred, a3W0Suffix16Pred,
    a3W1Suffix16Pred, a3W2Suffix16Pred]

/-- **The Gram's diagonal share falls as α grows.** -/
theorem diagonal_share_falls :
    a3DiagShare < a2DiagShare ∧ a2DiagShare < a1DiagShare ∧ a1DiagShare < a0DiagShare := by
  norm_num [a0DiagShare, a1DiagShare, a2DiagShare, a3DiagShare]

/-- **Q1 verdict (refuted).** The amplitude Gram prediction `2 ‖Σ δψ_l‖²` misses the measured group KL beyond the
three-window spread at every α, with opposite signs on the prefix and suffix groups. -/
theorem verdict_q1 :
    (∀ c ∈ allCells, c.1 < 3 / 4 * c.2) ∧
    ((∀ c ∈ prefixCells, c.1 < c.2) ∧ (∀ c ∈ suffixCells, c.2 < c.1)) ∧
    (a3DiagShare < a2DiagShare ∧ a2DiagShare < a1DiagShare ∧ a1DiagShare < a0DiagShare) :=
  ⟨all_overpredicted, prefix_over_suffix_under, diagonal_share_falls⟩

end Quantum.InterferenceGram
