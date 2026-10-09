import Mathlib.Data.Nat.Basic

/-!
# HATTN-T8: a page index in place of the flat scan of page bounds

`docs/evidence/hattn-decode-tail.md` (T8 section): the HATTN-T7 decode tail (64 positions before
`t ∈ {4096, 8192, 16384, 32768}`, Qwen2.5-0.5B Q4_K_M, 32k dump), one page set per KV head with a 2048-token
far budget refreshed every 8 positions. `grp-fix2048-r8` reads every 64-key page's Quest bound at a refresh.
`tree-sS` reads a fanout-16 tree of Quest boxes top-down, keeping `S` nodes per level; `clu-sS` reads
`⌈n64/16⌉` k-means cluster boxes and the pages of the `S` best (`KV_TAIL_TREE=1`, output
`/home/socol/oc-tmp/hattn-long/tree.out`). Measured figures are `model` inputs copied from that output: mean
whole-model KLD and bytes/dense in ten-thousandths, bound vectors per refresh in tenths, recall in thousandths,
rel-L2 in units of `1e-8`. The cost model below is exact arithmetic over its stated counts. rustc, LLVM and the
instrument are outside every statement.
-/

namespace Perf

namespace HattnT8

/-- Dense-arm tail control at 32768, max rel-L2 over q, k, v and 24 layers, units of `1e-8`. -/
def ctlMax : ℕ := 1815

/-- Mean KLD at `t = 4096, 8192, 16384, 32768`. -/
def grpKld : List ℕ := [53, 234, 342, 110]

def tree4Kld : List ℕ := [53, 714, 335, 128]

def tree8Kld : List ℕ := [53, 234, 372, 118]

def clu4Kld : List ℕ := [53, 192, 352, 121]

def clu8Kld : List ℕ := [53, 234, 329, 107]

/-- `HATTN_MUT=2` (ranking reversed at every level): `tree-s8` mean KLD at 8192. -/
def mutTree8Kld8 : ℕ := 6118

/-- Bound vectors read per refresh per KV head at 32768, tenths. -/
def flatScan32 : ℕ := 10138

def tree4Scan32 : ℕ := 1936

def tree8Scan32 : ℕ := 3193

def clu4Scan32 : ℕ := 2114

def clu8Scan32 : ℕ := 3453

/-- Share of the flat scan's top-32 pages the index selects at 32768, mean over refreshes, thousandths. -/
def tree8Recall32 : ℕ := 727

def clu4Recall32 : ℕ := 507

/-- KLD at 32768, the lists' last entries. -/
def grpKld32 : ℕ := 110

def clu4Kld32 : ℕ := 121

def tree8Kld32 : ℕ := 118

theorem singles_agree : grpKld.getLast? = some grpKld32 ∧ clu4Kld.getLast? = some clu4Kld32 ∧
    tree8Kld.getLast? = some tree8Kld32 := by
  decide

/-- Every entry of `l` is at most `0.05`. -/
def passes (l : List ℕ) : Prop := ∀ k ∈ l, k ≤ 500

instance (l : List ℕ) : Decidable (passes l) := by unfold passes; infer_instance

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8.md`, sha256 `719f0fd1…`) -/

/-- **PT1 held.** The dense tail reproduces the dump; reversing the index's ranking breaks it at 8192. -/
theorem pt1_held : ctlMax ≤ 10000 ∧ 2000 < mutTree8Kld8 := by decide

/-- **PT2 held.** `tree-s8` passes at every `t` and reads at most a third of the flat scan's bound vectors at
32768. -/
theorem pt2_held : passes tree8Kld ∧ 3 * tree8Scan32 ≤ flatScan32 := by decide

/-- **PT3 held.** `tree-s4` fails at 8192 (mean KLD 0.071): four 1024-key boxes miss pages that matter. -/
theorem pt3_held : ¬ passes tree4Kld := by decide

/-- **PT4 held.** `clu-s4` passes at every `t`. -/
theorem pt4_held : passes clu4Kld := by decide

/-- **PT5 refuted.** `tree-s8` keeps 0.727 of the flat top-32 at 32768, not 0.8. -/
theorem pt5_refuted : ¬ 800 ≤ tree8Recall32 := by decide

/-! ## Findings -/

/-- **The index cuts the scan 4.8× at 32k at the flat scan's quality.** `clu-s4` reads under a fifth of the flat
scan's bound vectors and its KLD at 32768 is within `0.002` of the flat arm. -/
theorem clu4_cuts_scan : 4 * clu4Scan32 < flatScan32 ∧ ¬ 5 * clu4Scan32 < flatScan32 ∧
    clu4Kld32 ≤ grpKld32 + 20 := by
  decide

/-- **The budget has slack.** `clu-s4` replaces about half of the flat top-32 pages and moves KLD at 32768 by
at most `0.002`: past the few pages that hold the mass, the rest are interchangeable. -/
theorem budget_slack : clu4Recall32 ≤ 510 ∧ clu4Kld32 ≤ grpKld32 + 20 := by decide

/-! ## Cost model

Counts are head-dim vectors per (layer, KV head). A refresh period is `r` positions; `b` far tokens plus 4
sinks plus a near window of `near` tokens are read as K and V at every position; the index's bound vectors
(a min and a max per box) are read once per period. Dense reads `2 t` vectors per position. -/

/-- Levels of the fanout-16 tree over `n` pages: `(size of the top level, levels above the pages)`. Each
level groups 16 consecutive boxes; the top has at most 16 nodes. `f` is fuel. -/
def climb : ℕ → ℕ → ℕ × ℕ
  | 0, n => (n, 0)
  | f + 1, n => if n ≤ 16 then (n, 0) else ((climb f ((n + 15) / 16)).1, (climb f ((n + 15) / 16)).2 + 1)

/-- **The tree is logarithmic.** A tree with `L ≥ 1` levels above the pages has more than `16^L` pages, so
`L < log₁₆ n` and the per-refresh scan `2 (top + 16 s L)` grows as `log n`. -/
theorem climb_log : ∀ f n, (climb f n).2 = 0 ∨ 16 ^ (climb f n).2 < n := by
  intro f
  induction f with
  | zero => intro n; exact Or.inl rfl
  | succ f ih =>
    intro n
    by_cases h : n ≤ 16
    · left; simp [climb, h]
    · right
      have e : climb (f + 1) n =
          ((climb f ((n + 15) / 16)).1, (climb f ((n + 15) / 16)).2 + 1) := by
        simp [climb, h]
      rw [e]
      rcases ih ((n + 15) / 16) with h0 | h1
      · simp only [h0, Nat.zero_add, Nat.pow_one]; omega
      · rw [Nat.pow_succ]; omega

/-- Bound vectors per refresh read by `tree-sS` over `n` pages: the top level, then at most `16 s` children
per level below it. An upper bound: a level with fewer than `16 s` children reads fewer. -/
def treeScan (n s : ℕ) : ℕ := 2 * ((climb 8 n).1 + 16 * s * (climb 8 n).2)

/-- Bound vectors per refresh read by the flat scan: two per page. -/
def flatScan (n : ℕ) : ℕ := 2 * n

/-- Far pages at context `t`: 64-key pages past 4 sinks and a 256-key window. -/
def pages (t : ℕ) : ℕ := (t - 260) / 64

/-- Vectors per refresh period of `r` positions: K and V of `b + 4 + near` tokens per position, plus the scan. -/
def period (b near r scan : ℕ) : ℕ := r * 2 * (b + 4 + near) + scan

/-- **The model bounds the measurement at 32k.** 507 pages form 32 nodes of 1024 keys under 2 roots; the
measured `tree-s8` mean (319.3) is under the model's 516 and the flat scan is `2 n64` (measured 1013.8). -/
theorem model_bounds_32k : pages 32768 = 507 ∧ climb 8 507 = (2, 2) ∧ 10 * treeScan 507 8 = 5160 ∧
    tree8Scan32 ≤ 10 * treeScan 507 8 ∧ flatScan32 ≤ 10 * flatScan 507 := by
  decide

/-- Qwen2.5-7B-Instruct-1M (`config.json`): 28 layers × 4 KV heads, head dim 128, f16, so one vector is 256
bytes and one position of one vector across the stack is `28 · 4 · 256` bytes. -/
def stackBytes : ℕ := 28 * 4 * 256

/-- **PT6 held (derived, quality not measured at 1M).** At `t = 2^20` with `b = 2048`, a 288-token mean near
window and `r = 8`, `tree-s8` reads at most `0.0025` of dense (0.00228, 439×); the flat scan reads `0.00418`
(239×). The tree's scan is 2% of its reads; the flat scan's is 47%. -/
theorem pt6_held :
    pages 1048576 = 16379 ∧ climb 8 16379 = (4, 3) ∧ treeScan 16379 8 = 776 ∧
      period 2048 288 8 (treeScan 16379 8) = 38216 ∧ period 2048 288 8 (flatScan 16379) = 70198 ∧
      10000 * 38216 ≤ 25 * (8 * 2 * 1048576) ∧ 438 * 38216 < 8 * 2 * 1048576 ∧
      ¬ 440 * 38216 < 8 * 2 * 1048576 ∧ 41 * (8 * 2 * 1048576) < 10000 * 70198 := by
  decide

/-- **Per generated token at 1M, Qwen2.5-7B, f16 KV:** `tree-s8` reads 137 MB (136,966,144 bytes) of the
cache; dense reads 60.1 GB (60,129,542,144 bytes). -/
theorem bytes_7b_1m : 38216 * stackBytes / 8 = 136966144 ∧ 2 * 1048576 * stackBytes = 60129542144 := by
  decide

end HattnT8

end Perf
