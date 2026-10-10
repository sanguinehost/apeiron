import Mathlib.Data.Nat.Basic

/-!
# attention-cut T8 two-stage selector: 1-bit scan pool, exact rerank

the hattn-decode-tail note (two-stage low-bit section): Qwen2.5-7B-Instruct-1M Q4_K_M, prose dump
`g7-1m`, `t = 524288`, 64 tail positions. `KV_TAIL_LOWBIT=1 KV_TAIL_LBPOOL=p` keeps the top-`p` far tokens by the
1-bit key score, then each query head keeps the exact top-`c` of that pool (`c` = the `grp-fix2048-r8` count).
Mean whole-model KLD in ten-thousandths, copied from `g24-lp-prose-<p>.out`; gate `0.05 = 500`. All figures are
`model` inputs; rustc, LLVM and the instrument are outside every statement.
-/

namespace Perf

namespace HattnT8LbPool

def lp8192 : ℕ := 1083

def lp32768 : ℕ := 802

def lp131072 : ℕ := 656

def lb1Alone : ℕ := 1477

def oracle2048 : ℕ := 553

/-- Pool key reads per head in ten-thousandths of dense (`p / (2 n)`, `n = 524288`), a lower bound before the
group union. -/
def poolKeyRead (p : ℕ) : ℕ := p * 10000 / 1048576

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8-LBPOOL.md`, sha256 `8aa46964f4e1…`) -/

/-- **LP1 refuted.** `p = 8192` is not under `0.1000` (0.1083). -/
theorem lp1_refuted : ¬ lp8192 < 1000 := by decide

/-- **LP2 refuted.** `p = 32768` is not within `0.0600` (0.0802). -/
theorem lp2_refuted : ¬ lp32768 ≤ 600 := by decide

/-- **LP3 refuted.** `p = 131072` is not within `0.0570` (0.0656). -/
theorem lp3_refuted : ¬ lp131072 ≤ 570 := by decide

/-- **LP4 held.** KLD is non-increasing in the pool size. -/
theorem lp4_held : lp131072 ≤ lp32768 ∧ lp32768 ≤ lp8192 := by decide

/-- **LP5 held.** `p = 8192` does not pass the gate. -/
theorem lp5_held : ¬ lp8192 ≤ 500 := by decide

/-! ## Findings -/

/-- A pool a quarter of the context still fails the gate; the exact rerank recovers about half of the 1-bit loss
(0.1477 to 0.0656) and no more. -/
theorem sign_pool_misses : ¬ lp131072 ≤ 500 ∧ 2 * lp131072 < 2 * lb1Alone ∧ oracle2048 < lp131072 := by
  decide

/-- Reading the quarter-context pool's keys costs at least `0.125` of dense, over twice the `0.053` byte target. -/
theorem pool_reads_exceed_target : poolKeyRead 131072 = 1250 ∧ 2 * 530 < poolKeyRead 131072 := by decide

end HattnT8LbPool

end Perf
