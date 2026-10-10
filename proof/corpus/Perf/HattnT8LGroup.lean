import Mathlib.Data.Nat.Basic

/-!
# attention-cut T8 per-layer-group far budget on the prose prompt (524288)

the hattn-decode-tail note (per-layer-group section): Qwen2.5-7B-Instruct-1M Q4_K_M, prose dump `g7-1m`,
`t = 524288`, 64 tail positions, `grp-fix2048-r8` with `KV_TAIL_LFARB=<group>:32768` on one group of seven layers and
2048 in every other layer (`attn_cut_probe.hd128qp`, `lgroup.sh`, outputs `g20-lg-<group>.out`). Entries are `model`
inputs copied from those outputs: mean whole-model KLD in units of `1e-4`, reads per token in `1e-4` of dense.
Predictions: `PREDICTIONS-T8-LGROUP.md`, sha256 `a17d8d8e…`. rustc, LLVM and the instrument are outside every
statement; one prompt, one model.
-/

namespace Perf

namespace HattnT8LGroup

/-- Mean KLD with layers `0-6`, `7-13`, `14-20`, `21-27` at 32768 (others 2048). -/
def groupKld : List ℕ := [2531, 2770, 3035, 3133]

/-- Flat 2048 in every layer (`g19-prose-2048.out`) and 32768 in every layer (`g19-prose-32768.out`). -/
def flatKld : ℕ := 3370

def allKld : ℕ := 1687

/-- Mean reads per token of each single-group run, `1e-4` of dense. -/
def groupReads : ℕ := 191

/-- Reduction of each group against the flat arm. -/
def reduction : List ℕ := groupKld.map (fun k => flatKld - k)

/-- **LG1 held.** No single group at 32768 brings the mean KLD to the `0.05` gate. -/
theorem lg1_held : ∀ k ∈ groupKld, 500 < k := by decide

/-- **LG2 held.** The best single group lowers the mean KLD by `0.0839`, at least `0.05`. -/
theorem lg2_held : reduction = [839, 600, 335, 237] ∧ 500 ≤ 839 := by decide

/-- **LG3 held.** Layers 0-6 give the largest reduction, and the reductions fall with depth. -/
theorem lg3_held : reduction = [839, 600, 335, 237] ∧ 600 < 839 ∧ 335 < 600 ∧ 237 < 335 := by decide

/-- **LG4 held.** The four reductions sum to `0.2011`, above the all-layers reduction `0.1683`: the effects overlap,
so the four groups together are not worth four separate runs of the same cost. -/
theorem lg4_held : reduction.sum = 2011 ∧ flatKld - allKld = 1683 ∧ 1683 ≤ reduction.sum := by decide

/-- **Finding: the best single group closes between a quarter and a third of the gap to the gate.** Layers 0-6
at 32768 remove `0.0839` of the `0.2870` that separates flat 2048 from the gate. -/
theorem group_share : 3 * 839 < 2870 ∧ ¬ 4 * 839 < 2870 ∧ flatKld - 500 = 2870 := by decide

end HattnT8LGroup

end Perf
