import Mathlib.Data.Nat.Basic

/-!
# attention-cut T8 low-bit key scan: select far tokens by a quantised-key score over all tokens

the hattn-decode-tail note (low-bit section): Qwen2.5-7B-Instruct-1M Q4_K_M, prose dump `g7-1m`,
`t = 524288`, 64 tail positions. `KV_TAIL_LOWBIT=b` ranks every far token by `q . dequant_b(k)` and each query
head keeps its top-`c` (`c` = the count the `grp-fix<B>-r8` page set selects); bytes/dense charges the selected
sets' KV-group union plus the scan, `b/32` of dense (`g23-lb<b>-prose-<B>.out`). Mean whole-model KLD and
bytes/dense in ten-thousandths, copied from the outputs; every figure is a `model` input. rustc, LLVM and the
instrument are outside every statement.
-/

namespace Perf

namespace HattnT8LowBit

def lb8 : ℕ := 552

def lb4 : ℕ := 551

def lb2 : ℕ := 2019

def lb1 : ℕ := 1477

def lb1At8192 : ℕ := 1068

def oracle2048 : ℕ := 553

def flat2048 : ℕ := 3370

def b4 : ℕ := 1407

def b1 : ℕ := 511

def oracleB2048 : ℕ := 176

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8-LOWBIT.md`, sha256 `8a068d4d…`) -/

/-- **Control.** 8 bits reproduces the oracle to within `0.0005`: the scan path ranks like the exact scores. -/
theorem control_held : lb8 ≤ oracle2048 + 5 ∧ oracle2048 ≤ lb8 + 5 := by decide

/-- **LB1 held.** 4 bits at `B = 2048` is under `0.10`. -/
theorem lb1_held : lb4 < 1000 := by decide

/-- **LB2 refuted (by 0.0051).** 4 bits at `B = 2048` is over the gate. -/
theorem lb2_refuted : ¬ lb4 ≤ 500 ∧ lb4 - 500 = 51 := by decide

/-- **LB3 held.** 1 bit at `B = 2048` is under `0.15` (0.1477). -/
theorem lb3_held : lb1 < 1500 := by decide

/-- **LB4 refuted.** 1 bit at `B = 8192` is over the gate (0.1068). -/
theorem lb4_refuted : ¬ lb1At8192 ≤ 500 := by decide

/-- **LB5 refuted.** KLD is not non-increasing in `b`: the 2-bit quantiser, three levels at scale `max|k|`,
ranks worse than 1 bit (0.2019 against 0.1477). -/
theorem lb5_refuted : lb1 < lb2 := by decide

/-- **LB6 held.** 1 bit at `B = 2048` costs at most `0.053` of dense bytes (0.0511). -/
theorem lb6_held : b1 ≤ 530 := by decide

/-! ## Findings -/

/-- **4 bits ranks like the exact scores.** Its KLD is the oracle's (0.0551 against 0.0553), its selected sets'
union is smaller than the oracle's (the bytes above the `b/32 = 0.125` scan are 0.0157 against 0.0176), and the
total is 7× below dense but 2.6× the oracle-8192 byte target. -/
theorem four_bit_matches_oracle :
    lb4 ≤ oracle2048 ∧ 1250 ≤ b4 ∧ b4 - 1250 < oracleB2048 ∧ 7 * b4 < 10000 ∧ ¬ 8 * b4 < 10000 ∧
      ¬ b4 ≤ 530 := by
  decide

/-- 1 bit at `B = 2048` still halves the flat 2048 figure (0.1477 against 0.3370). -/
theorem one_bit_halves_flat : 2 * lb1 < flat2048 := by decide

end HattnT8LowBit

end Perf
