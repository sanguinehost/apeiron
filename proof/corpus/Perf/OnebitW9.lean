import Mathlib.Data.Nat.Basic

/-!
# ONEBIT W9: whole-model one-bit conversion of Qwen2.5-0.5B, paired KLD against f16

the onebit-w9 note. wikitext-2 test, 64 chunks of 512, llama.cpp `llama-perplexity` on CPU, paired
against the f16 logits. KLD in millionths of a nat. Bits in bits per 1024 weights. All figures are `model` inputs
read from the logs the receipt names; llama.cpp and the CPU are outside every statement.

- P1: the 21 `ffn_down` tensors (blocks 3–23) that llama.cpp's `IQ1_S`/`IQ1_M` reach; every other tensor is
  byte-identical across the P1 arms.
- P2: all 168 linear tensors at one bit; `token_embd` `q8_0` and the norms identical across arms.
-/

namespace Perf

namespace OnebitW9

/-- Bits per 1024 weights: `Q1_0` 18 B/128, `IQ1_S` 50 B/256, `IQ1_M` 56 B/256. -/
def q1Bits : ℕ := 18 * 8 * 8

def iq1sBits : ℕ := 50 * 8 * 4

def iq1mBits : ℕ := 56 * 8 * 4

def f16SelfKld : ℕ := 0

def q4kmKld : ℕ := 24318

def p1Iq1s : ℕ := 631488

def p1Iq1m : ℕ := 519083

def p1LlamaQ1 : ℕ := 10384596

def p1Rtn : ℕ := 10384596

def p1Gptq1 : ℕ := 5242942

def p1HadGptq1 : ℕ := 834489

def p2LlamaQ1 : ℕ := 13166693

def p2Rtn : ℕ := 13166692

def p2Gptq1 : ℕ := 15300480

def p2HadGptq1 : ℕ := 6849614

/-- **Control.** Identical weights give KLD 0. -/
theorem control_zero : f16SelfKld = 0 := by decide

/-- **Writer.** Our `rtn` written as `Q1_0` is the llama.cpp `Q1_0` file on P1 (same KLD; `cmp` identical), and on
P2 it differs by one millionth (one `f16` scale ulp in one block of 2795520). -/
theorem writer_matches_llama : p1Rtn = p1LlamaQ1 ∧ p2LlamaQ1 - p2Rtn ≤ 1 := by decide

/-- One-bit storage is below both `IQ1` grids: `Q1_0` 1.125, `IQ1_S` 1.5625, `IQ1_M` 1.75 bits per weight. -/
theorem q1_fewest_bits : q1Bits < iq1sBits ∧ iq1sBits < iq1mBits := by decide

/-- On P1, `had-gptq1` beats llama.cpp's `Q1_0` by more than `12×` in KLD at the same 1.125 bits. -/
theorem p1_had_beats_q1_twelvefold : 12 * p1HadGptq1 < p1LlamaQ1 := by decide

/-- On P1, both `IQ1` grids beat `had-gptq1`; `IQ1_S` by less than `1.33×` with `1.39×` the bits. -/
theorem p1_iq1_beats_had : p1Iq1m < p1HadGptq1 ∧ p1Iq1s < p1HadGptq1 ∧ 3 * p1HadGptq1 < 4 * p1Iq1s := by decide

/-- On P1 the arm order is `had-gptq1 < gptq1 < rtn`. -/
theorem p1_order : p1HadGptq1 < p1Gptq1 ∧ p1Gptq1 < p1Rtn := by decide

/-- On P2 `had-gptq1` is the best one-bit arm, and `gptq1` is worse than `rtn`. -/
theorem p2_order : p2HadGptq1 < p2Rtn ∧ p2Rtn < p2Gptq1 := by decide

/-- **Verdict.** Post-training one-bit on every linear tensor is not usable: the best arm's KLD exceeds 6 nats,
over 280× the `Q4_K_M` reference. -/
theorem p2_not_usable : 6000000 < p2HadGptq1 ∧ 280 * q4kmKld < p2HadGptq1 := by decide

end OnebitW9

end Perf
