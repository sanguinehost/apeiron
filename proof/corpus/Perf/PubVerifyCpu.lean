import Mathlib.Data.Nat.Basic

/-!
# PUB-VERIFY-CPU: perplexity of the published CPU path against llama.cpp (Qwen3.5-0.8B Q8_0)

the pub-verify-cpu note. Same token ids, first 16 chunks of 512, 255 scored tokens per chunk.
llama.cpp `ece963f41` `llama-perplexity` on CPU; apeiron `examples/lm_eval_bridge.rs` with
`APEIRON_KERNEL=q8act APEIRON_F32_CARRIER=1` at `04d2060` and on main. PPL in ten-thousandths. All figures are
`model` inputs; rustc, LLVM, llama.cpp and the CPU are outside every statement.
-/

namespace Perf

namespace PubVerifyCpu

def llamaPpl : ℕ := 118967

def apPpl : ℕ := 119199

def apPrefillPpl : ℕ := 119199

def mainPpl : ℕ := 119199

/-- **Within 1%.** apeiron's perplexity exceeds llama.cpp's by `0.19%`, under the `1%` decision rule. -/
theorem within_one_percent : llamaPpl ≤ apPpl ∧ 100 * (apPpl - llamaPpl) < llamaPpl := by decide

/-- The gap is under `0.2%`. -/
theorem under_point_two_percent : 1000 * (apPpl - llamaPpl) < 2 * llamaPpl := by decide

/-- Batched prefill gives the same perplexity as token-by-token decode, and main equals `04d2060`. -/
theorem prefill_and_main_agree : apPrefillPpl = apPpl ∧ mainPpl = apPpl := by decide

end PubVerifyCpu

end Perf
