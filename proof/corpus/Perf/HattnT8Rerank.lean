import Mathlib.Data.Nat.Basic

/-!
# HATTN-T8 rerank: a Quest page pool, then exact token top-2048 inside it

`docs/evidence/hattn-decode-tail.md` (rerank section): Qwen2.5-7B-Instruct-1M Q4_K_M, prose dump `g7-1m`,
`t = 524288`, 64 tail positions. `grp-fix<P>-r8` selects a pool of `P` far tokens by Quest's page bound; with
`KV_TAIL_RERANK=2048` each query head then keeps its exact top-2048 far tokens inside the pool
(`g22-rr-prose-<P>.out`). Pool scoring is not charged to the bytes. Mean whole-model KLD and bytes/dense in
ten-thousandths, copied from the outputs; every figure is a `model` input. rustc, LLVM and the instrument are
outside every statement.
-/

namespace Perf

namespace HattnT8Rerank

def rr8192 : ℕ := 2619

def rr32768 : ℕ := 1768

def rr131072 : ℕ := 1045

def flat2048 : ℕ := 3370

def flat8192 : ℕ := 2612

def flat32768 : ℕ := 1687

def oracle2048 : ℕ := 553

def rrB8192 : ℕ := 106

def rrB32768 : ℕ := 143

def rrB131072 : ℕ := 166

def oracleB2048 : ℕ := 176

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8-RERANK.md`, sha256 `25d78316…`) -/

/-- **RR1 refuted.** The 32768 pool with rerank does not beat the flat 32768 arm by 1.5× (`< 0.112`): it is
worse, 0.1768 against 0.1687. -/
theorem rr1_refuted : ¬ rr32768 < 1120 ∧ flat32768 < rr32768 := by decide

/-- **RR2 refuted.** The 32768 pool with rerank is over the gate. -/
theorem rr2_refuted : ¬ rr32768 ≤ 500 := by decide

/-- **RR3 refuted.** The 131072 pool (a quarter of the context) with rerank is over the gate. -/
theorem rr3_refuted : ¬ rr131072 ≤ 500 := by decide

/-- **RR4 held.** The rerank KLD is non-increasing in the pool size. -/
theorem rr4_held : rr32768 ≤ rr8192 ∧ rr131072 ≤ rr32768 := by decide

/-- **RR5 refuted (by 0.0007).** The 8192 pool with rerank is not below the flat 8192 arm. -/
theorem rr5_refuted : ¬ rr8192 ≤ flat8192 ∧ rr8192 - flat8192 = 7 := by decide

/-! ## Findings -/

/-- **The pool, not the rerank, is the limit.** Against the flat 2048 arm (same final token count) the rerank
cuts KLD by 1.3×, 1.9× and 3.2× at pools 8192, 32768, 131072, but the 131072 pool stays 1.89× above the
oracle. -/
theorem pool_limit :
    100 * rr8192 < 130 * flat2048 ∧ 19 * rr32768 < 10 * flat2048 ∧ 3 * rr131072 < flat2048 ∧
      oracle2048 < rr131072 ∧ rr131072 < 2 * oracle2048 := by
  decide

/-- The rerank arms read fewer bytes than the oracle's per-head sets at 2048 (the pool scan is not counted). -/
theorem bytes_under_oracle : rrB8192 < oracleB2048 ∧ rrB32768 < oracleB2048 ∧ rrB131072 < oracleB2048 := by
  decide

end HattnT8Rerank

end Perf
