import Mathlib.Tactic

/-!
# attention-cut T1: the registered predictions against the measured constants

the hattn-qk-onebit note: one prompt, Qwen2.5-0.5B Q4_K_M, T = 512, 336 heads
(`~/hattn-t1/run1.txt`; three runs identical). Every figure is a `model` input copied
from the table; no theorem here produces a measurement. KL in thousandths of a nat, TV in
ten-thousandths, fractions in percent. rustc, numpy and the instrument's own correctness are outside
every statement here.
-/

namespace Perf

namespace HattnT1

/-- Best sign arm: A, `b = 64`, `R`. Median held-out KL, milli-nats. -/
def klA64 : ℕ := 1664

/-- Arm B (key norm multiplied in) at `b = 64`: median held-out KL, milli-nats. -/
def klB64 : ℕ := 2381

/-- The registered KL target, milli-nats. -/
def klTarget : ℕ := 50

/-- **P1 held, with margin.** The best sign arm is above the target by at least 33×. -/
theorem p1_holds : klTarget < klA64 ∧ 33 * klTarget ≤ klA64 := by decide

/-- **P2 refuted.** B lowering A's median by at least 2× would need `2 * klB64 ≤ klA64`; B is higher. -/
theorem p2_refuted : ¬ 2 * klB64 ≤ klA64 ∧ klA64 < klB64 := by decide

/-- Fraction (percent) of kv heads whose key codes are all distinct at `t = 512`: `b = 16`, `32`,
`64` (`P`), `64` (`R`). -/
def distinctFrac : List ℕ := [0, 4, 62, 75]

/-- **P3 refuted.** P3 needed at least 90 % at every `b ≥ 16`; none reaches it. -/
theorem p3_refuted : ¬ ∀ x ∈ distinctFrac, 90 ≤ x := by decide

/-- **P3, no cell reaches the bar.** -/
theorem p3_no_cell : ∀ x ∈ distinctFrac, x < 90 := by decide

/-- Median TV (ten-thousandths): arm D at `r = 3`, arm A at `b = 16`. -/
def tvD3 : ℕ := 7461

def tvA16 : ℕ := 7861

/-- **P4 refuted.** P4 needed D(`r = 3`) above A(`b = 16`); it is below. -/
theorem p4_refuted : ¬ tvA16 < tvD3 ∧ tvD3 < tvA16 := by decide

end HattnT1

end Perf
