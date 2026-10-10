import Mathlib.Data.Nat.Basic

/-!
# attention-cut T3: trained one-bit key codes on Qwen2.5 q, k (0.5B and 7B-Instruct-1M)

the hattn-qk-onebit note (trained-projection section). Instrument `tools/hattn/qk_train.py (unpublished)`; data the
32768-token prose prefix of `g7-1m` (`g05-32k`, `g7-32k`); held-out rows `t ∈ [24576, 32768)` over all keys `s ≤ t`.
0.5B: layers {2, 6, 10, 14, 18, 22}, 84 query heads (`t3-05b.jsonl`). 7B: layers {4, 14, 24}, 84 query heads, trained
arms ST and STI only (`t3-7b.jsonl`). KL is the median over heads in ten-thousandths of a nat; mass is the mean float
softmax mass of the arm's top `t/256` keys in ten-thousandths; `oracle` is the exact top `t/256`. All figures are
`model` inputs copied from `--summary`; the instrument, numpy and the dumps are outside every statement.
-/

namespace Perf

namespace HattnT3Train

def gate : ℕ := 500

/-! ## 0.5B population -/

def s0Kl05 : ℕ := 26996
def s1Kl05 : ℕ := 35830
def l4Kl05 : ℕ := 2589
def stKl05 : ℕ := 28432
def ssKl05 : ℕ := 35410
def stiKl05 : ℕ := 22981

def s0Mass05 : ℕ := 5007
def s1Mass05 : ℕ := 4316
def l4Mass05 : ℕ := 7659
def stMass05 : ℕ := 5250
def ssMass05 : ℕ := 3833
def stiMass05 : ℕ := 6205
def oracleMass05 : ℕ := 8134

/-! ## 7B population -/

def s0Kl7 : ℕ := 46442
def s1Kl7 : ℕ := 22592
def l4Kl7 : ℕ := 1533
def stKl7 : ℕ := 13281
def stiKl7 : ℕ := 11247

def s0Mass7 : ℕ := 3355
def s1Mass7 : ℕ := 7278
def l4Mass7 : ℕ := 8668
def stMass7 : ℕ := 7405
def stiMass7 : ℕ := 7913
def oracleMass7 : ℕ := 8781

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T3.md`, sha256 `202b5ac8a192…`) -/

/-- **T3a refuted.** Registered `ST/S1` KL ratio `0.45 [0.25, 0.75]`; measured `2.8432 / 3.5830 = 0.79`. -/
theorem t3a_refuted : 3 * s1Kl05 < 4 * stKl05 := by decide

/-- **T3b held.** Registered gap closure `φ = 0.35 [0.15, 0.60]`; measured `(0.5250 − 0.4316)/(0.8134 − 0.4316)
= 0.245`. -/
theorem t3b_held :
    15 * (oracleMass05 - s1Mass05) ≤ 100 * (stMass05 - s1Mass05) ∧
    100 * (stMass05 - s1Mass05) ≤ 60 * (oracleMass05 - s1Mass05) := by decide

/-- **T3c held.** ST mass stays below L4 by at least `0.05` (`0.5250` against `0.7659`). -/
theorem t3c_held : stMass05 + 500 ≤ l4Mass05 := by decide

/-- **T3d point refuted.** Registered SS KL `2.5 [1.5, 3.5]`; measured `3.5410`, above the interval. -/
theorem t3d_point_refuted : 35000 < ssKl05 := by decide

/-- **T3d second clause held.** SS does not reach `0.05`. -/
theorem t3d_gate_not_reached : gate < ssKl05 := by decide

/-- **T3e held.** ST does not reach `0.05` on either model. -/
theorem t3e_held : gate < stKl05 ∧ gate < stKl7 := by decide

/-- **T3f refuted.** Registered: 7B ST gap closure within `[0.15, 0.60]`; measured `(0.7405 − 0.7278)/(0.8781 −
0.7278) = 0.085`. -/
theorem t3f_refuted : 100 * (stMass7 - s1Mass7) < 15 * (oracleMass7 - s1Mass7) := by decide

/-! ## Decision rules -/

/-- **Port rule not met.** The port to `attn_cut_probe` needs ST mass `≥ L4 − 0.02` on the 7B; ST is `0.1263`
below, and the identity-initialised STI (a labelled extension, no registered prediction) is `0.0755` below. -/
theorem port_rule_not_met : stMass7 + 200 < l4Mass7 ∧ stiMass7 + 200 < l4Mass7 := by decide

/-- **SS conversion rule met.** SS misses `0.05` by more than `10×` (`3.5410`, `70×`): one-bit q and k do not
convert this model by a trained projection alone. -/
theorem ss_conversion_fails : 10 * gate < ssKl05 := by decide

/-! ## Findings -/

/-- On the 7B the Hadamard rotation alone more than doubles the one-bit mass (`0.3355 → 0.7278`); the probe's
`KV_TAIL_LOWBIT=1` score is the unrotated `S0`. On the 0.5B it does not (`0.5007 → 0.4316`). -/
theorem rotation_doubles_7b : 2 * s0Mass7 < s1Mass7 ∧ s1Mass05 < s0Mass05 := by decide

/-- STI is the best one-bit arm on both models, below 4-bit L4 on both. -/
theorem sti_best_onebit :
    stMass05 < stiMass05 ∧ s0Mass05 < stiMass05 ∧ stiMass05 < l4Mass05 ∧
    stMass7 < stiMass7 ∧ s1Mass7 < stiMass7 ∧ stiMass7 < l4Mass7 := by decide

/-- 4-bit L4 is the only arm below `0.5` median KL on either model, and it is not below `0.05`. -/
theorem l4_closest : l4Kl05 < 5000 ∧ l4Kl7 < 5000 ∧ gate < l4Kl7 ∧ l4Kl7 < stiKl7 := by decide

end HattnT3Train

end Perf
