import Perf.HattnT8

/-!
# attention-cut T8 on Qwen2.5-7B-Instruct-1M

the hattn-decode-tail note (T8 on the 7B section): the attention-cut T8 arms unchanged (`grp-fix2048-r8`,
`tree-s4`, `tree-s8`, `clu-s4`, `clu-s8`), the same 64 tail positions before `t ∈ {4096, 8192, 16384, 32768}`,
on Qwen2.5-7B-Instruct-1M Q4_K_M (28 layers, 28 query and 4 KV heads, head dim 128), one 32k dump
(`~/hattn-long/t32k-7b`), instrument `attn_cut_probe.hd128t` (`PROBE_HD=128`), one `t` per
invocation (`tree7b-<t>.out`, `tree7b-m2.out`). Measured figures are `model` inputs copied from those outputs:
mean whole-model KLD and bytes/dense in ten-thousandths, bound vectors per refresh in tenths, recall in
thousandths, rel-L2 in units of `1e-8`. The cost model is `Perf.HattnT8`'s. rustc, LLVM and the instrument are
outside every statement.
-/

namespace Perf

namespace HattnT8Q7

/-- Dense-arm tail control, max rel-L2 over q, k, v and 28 layers, at `t = 4096, 8192, 16384, 32768`. -/
def ctl : List ℕ := [242, 502, 1244, 1933]

/-- Mean KLD at `t = 4096, 8192, 16384, 32768`. -/
def flatKld : List ℕ := [30, 70, 103, 38]

def tree4Kld : List ℕ := [30, 262, 148, 49]

def tree8Kld : List ℕ := [30, 70, 115, 47]

def clu4Kld : List ℕ := [30, 84, 163, 43]

def clu8Kld : List ℕ := [30, 70, 117, 40]

/-- `HATTN_MUT=2` (ranking reversed at every level): `tree-s8` mean KLD at 8192. -/
def mutTree8Kld8 : ℕ := 3660

/-- Bound vectors read per refresh per KV head at 32768, tenths. -/
def flatScan32 : ℕ := 10138

def tree4Scan32 : ℕ := 1951

def tree8Scan32 : ℕ := 3219

def clu4Scan32 : ℕ := 2312

def clu8Scan32 : ℕ := 3760

/-- Share of the flat scan's top-32 pages the index selects at 32768, thousandths. -/
def tree8Recall32 : ℕ := 625

def clu4Recall32 : ℕ := 456

/-- `tree-s8` mean bytes/dense at 32768, ten-thousandths. -/
def tree8Bytes32 : ℕ := 722

theorem singles_agree : ctl.getLast? = some 1933 ∧ flatKld.getLast? = some 38 ∧
    tree8Kld.getLast? = some 47 ∧ clu4Kld.getLast? = some 43 := by
  decide

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8-7B.md`, sha256 `360d7ea8…`) -/

/-- **P7-1 held.** The dense tail reproduces the dump at every `t` (largest `1.9e-5`); reversing the index's
ranking breaks `tree-s8` at 8192 (mean KLD 0.366). -/
theorem p71_held : (∀ c ∈ ctl, c ≤ 10000) ∧ 2000 < mutTree8Kld8 := by decide

/-- **P7-2 held.** The flat group budget passes at every `t`. -/
theorem p72_held : HattnT8.passes flatKld := by decide

/-- **P7-3 held.** `clu-s4` passes at every `t` and reads under a quarter of the flat scan's bound vectors at
32768 (4.4× fewer). -/
theorem p73_held : HattnT8.passes clu4Kld ∧ 4 * clu4Scan32 ≤ flatScan32 ∧ ¬ 5 * clu4Scan32 ≤ flatScan32 := by
  decide

/-- **P7-4 held.** `tree-s8` passes at every `t`. -/
theorem p74_held : HattnT8.passes tree8Kld := by decide

/-- **P7-5 held.** The measured `tree-s8` scan at 32768 (321.9) is under the cost model's 516
(`HattnT8.model_bounds_32k`), and the flat scan (1013.8) is under the model's `2 · 507`. -/
theorem p75_held : tree8Scan32 ≤ 10 * HattnT8.treeScan 507 8 ∧ flatScan32 ≤ 10 * HattnT8.flatScan 507 := by
  decide

/-- **P7-6 held (derived; quality not measured at 1M).** `tree-s8` passes on the 7B, and the cost model at
`t = 2^20` with `b = 2048`, a 288-token near window and `r = 8` reads at most `0.0025` of dense
(`HattnT8.pt6_held`: 0.00228). The model at 32768 bounds the measured bytes/dense (0.0722). -/
theorem p76_held : HattnT8.passes tree8Kld ∧
    10000 * HattnT8.period 2048 288 8 (HattnT8.treeScan 16379 8) ≤ 25 * (8 * 2 * 1048576) ∧
    tree8Bytes32 ≤ 10000 * HattnT8.period 2048 288 8 (HattnT8.treeScan 507 8) / (8 * 2 * 32768) := by
  decide

/-! ## Findings -/

/-- **Every arm passes on the 7B at every `t`,** including `tree-s4`, which fails at 8192 on the 0.5B
(`HattnT8.pt3_held`; 7B mean KLD 0.026). -/
theorem all_arms_pass : HattnT8.passes tree4Kld ∧ HattnT8.passes tree8Kld ∧ HattnT8.passes clu4Kld ∧
    HattnT8.passes clu8Kld ∧ HattnT8.passes flatKld ∧ ¬ HattnT8.passes HattnT8.tree4Kld := by
  decide

/-- **The 7B is easier than the 0.5B for the flat arm:** mean KLD is below the 0.5B's at every `t`. -/
theorem flat_below_half_b : ∀ p ∈ flatKld.zip HattnT8.grpKld, p.1 ≤ p.2 := by decide

/-- **The index's recall falls but quality does not.** `tree-s8` keeps `0.625` of the flat top-32 at 32768 on the
7B against `0.727` on the 0.5B, at a lower mean KLD (`0.0047` against `0.0118`): the pages it misses carry less
mass. -/
theorem recall_falls_kld_holds : tree8Recall32 < HattnT8.tree8Recall32 ∧ tree8Kld.getLast? = some 47 ∧
    HattnT8.tree8Kld32 = 118 := by
  decide

/-- **Scan cut at 32k on the 7B:** `tree-s8` reads between a third and a quarter of the flat scan, `clu-s4`
between a quarter and a fifth. -/
theorem scan_cut_7b : 3 * tree8Scan32 ≤ flatScan32 ∧ ¬ 4 * tree8Scan32 ≤ flatScan32 ∧
    4 * clu4Scan32 ≤ flatScan32 ∧ ¬ 5 * clu4Scan32 ≤ flatScan32 := by
  decide

end HattnT8Q7

end Perf
