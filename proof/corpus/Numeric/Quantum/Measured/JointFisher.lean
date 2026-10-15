import Perf.Receipts.Onebit

/-!
# Q2: the joint Fisher-sketch allocation does not beat the separable one (verdict: refuted)

Board: `quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs`. Epic: `docs/epics/quantum-register.md` row Q2.

`examples/qarch_q2.rs` takes a 16-sample Fisher sketch of the Qwen2.5-0.5B f16 model on the Rust path and fills two
allocations to the same parameter budget (`budgetParams`): `separable`, a per-block diagonal greedy, and `joint`, a
greedy on the sketched Gram including the cross-layer terms. The uniform arm is the reference. Each arm's held-out KL
is measured on three windows of 64 tokens. The receipt `docs/evidence/receipts/onebit/qarch-q2.json` (constants
`Receipts.qarch_q2`, bound to its ledger sha256 by `tools/rooting/receipt_lean.py`) gives those KL values, the rates
and the parameter counts.

The prediction is that the joint allocation has lower KL than the separable one on all three windows at equal rate.
It is refuted:

* `separable_below_uniform`: the separable arm is below the uniform arm on all three windows.
* `joint_not_below_separable`: on windows 1 and 2 and on the mean, the joint arm has higher KL than the separable arm.
* `joint_below_on_w0`: the joint arm is below the separable arm on window 0 only. This is stated, not hidden.
* `equal_rate`: the two bit rates differ by less than `1/10000` b/w and both fit the budget.
* `regime_far`: the uniform KL is above 9 nats/token on window 2 and the control `c` is zero. The student is far from
  the teacher, so these results describe this regime and this 16-sample sketch.

Not claimed: anything about 1.18 b/w quality, or about sketches with more samples.
-/

namespace Quantum.JointFisher

open Receipts.qarch_q2

/-- Separable arm is below uniform on all three windows. -/
theorem separable_below_uniform : sepW0 < uniW0 ∧ sepW1 < uniW1 ∧ sepW2 < uniW2 := by
  norm_num [sepW0, sepW1, sepW2, uniW0, uniW1, uniW2]

/-- **The joint arm is not below the separable arm on windows 1, 2 or on the mean.** -/
theorem joint_not_below_separable :
    sepW1 < jointW1 ∧ sepW2 < jointW2 ∧ sepMean < jointMean := by
  norm_num [sepW1, sepW2, sepMean, jointW1, jointW2, jointMean]

/-- The one window where the joint arm is below the separable arm. -/
theorem joint_below_on_w0 : jointW0 < sepW0 := by
  norm_num [jointW0, sepW0]

/-- Both allocations have the same bit rate to within `1/10000` b/w and fit the parameter budget. -/
theorem equal_rate :
    |jointBpw - sepBpw| < 1 / 10000 ∧ sepParams ≤ budgetParams ∧ jointParams ≤ budgetParams := by
  norm_num [abs_lt, jointBpw, sepBpw, sepParams, jointParams, budgetParams]

/-- The regime scope: uniform KL above 9 nats/token on window 2, and control `c` is zero. -/
theorem regime_far : 9 < uniW2 ∧ uniControlC = 0 := by
  norm_num [uniW2, uniControlC]

/-- **Q2 verdict (refuted).** The joint allocation does not undercut the separable one at equal rate on all three
windows: it is higher on windows 1 and 2 and on the mean. -/
theorem verdict_q2 :
    sepW0 < uniW0 ∧ sepW1 < uniW1 ∧ sepW2 < uniW2 ∧
    (sepW1 < jointW1 ∧ sepW2 < jointW2 ∧ sepMean < jointMean) ∧
    jointW0 < sepW0 ∧
    (|jointBpw - sepBpw| < 1 / 10000 ∧ sepParams ≤ budgetParams ∧ jointParams ≤ budgetParams) ∧
    (9 < uniW2 ∧ uniControlC = 0) :=
  ⟨separable_below_uniform.1, separable_below_uniform.2.1, separable_below_uniform.2.2,
    joint_not_below_separable, joint_below_on_w0, equal_rate, regime_far⟩

end Quantum.JointFisher
