import Perf.HattnT8

/-!
# attention-cut T8 on Qwen2.5-7B-Instruct-1M at 262144

the hattn-decode-tail note (T8 on the 7B at 256k section): the attention-cut T8 arms unchanged (`grp-fix2048-r8`,
`tree-s4`, `tree-s8`, `clu-s4`, `clu-s8`), the 64 tail positions before `t = 262144`, on Qwen2.5-7B-Instruct-1M
Q4_K_M. The dump is the f32 GPU dump `g7-256k` (`examples/gpu_dump.rs`,
`KV_T=262144`, `KV_QTAIL=131072,262144`), instrument `attn_cut_probe.hd128qa` (`KV_TAIL_ARMS` runs one arm per
invocation to fit the 600 s budget; each arm is its own forward). Outputs `g8a-<arm>.out`, `g8t-131072.out`.
Measured figures are `model` inputs copied from those outputs: mean whole-model KLD and bytes/dense in
ten-thousandths, bound vectors per refresh in tenths, recall in thousandths, rel-L2 in units of `1e-8`. The cost
model is `Perf.HattnT8`'s. rustc, LLVM, the f32 dump kernels and the instrument are outside every statement.
-/

namespace Perf

namespace HattnT8Q8

/-- Dense-arm tail control, max rel-L2 over q, k, v and 28 layers, at `t = 131072` and `t = 262144`. -/
def ctl : List ℕ := [7042, 17880]

/-- Mean KLD at 262144, ten-thousandths. -/
def flatKld : ℕ := 386

def tree4Kld : ℕ := 496

def tree8Kld : ℕ := 468

def clu4Kld : ℕ := 418

def clu8Kld : ℕ := 379

/-- `HATTN_MUT=2` (ranking reversed) on `clu-s4` at 262144. -/
def mutClu4Kld : ℕ := 787

/-- The 131072 tail from this dump (`g8t-131072.out`) and the g7-128k tail (`g7t-131072.out`), arm order flat,
tree-s4, tree-s8, clu-s4, clu-s8. -/
def from256k131 : List ℕ := [242, 611, 549, 270, 227]

def from128k131 : List ℕ := [242, 611, 549, 270, 227]

/-- Bound vectors per refresh per KV head at 262144, tenths. -/
def flatScan : ℕ := 81818

def tree4Scan : ℕ := 2870

def tree8Scan : ℕ := 5421

def clu4Scan : ℕ := 6740

/-- Recall of the flat top-32 pages, thousandths: `clu-s4` at 131072 and 262144. -/
def clu4Recall : List ℕ := [289, 264]

/-- Bytes/dense, ten-thousandths: `clu-s4` at 131072 and 262144. -/
def clu4Bytes : List ℕ := [181, 91]

/-- Every entry is at most `0.05`. -/
def passes (k : ℕ) : Prop := k ≤ 500

instance (k : ℕ) : Decidable (passes k) := by unfold passes; infer_instance

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8-7B-256K.md`, sha256 `02a5f3bc…`) -/

/-- **Q1 refuted in two clauses, held in one.** The 131072 tail from this dump reproduces the g7-128k arms
exactly. The dense control at 262144 is `1.788e-4`, over the `1e-4` bar the probe applies (so the probe prints
`TAIL FAIL`), and the reversed ranking raises `clu-s4` to `0.0787`, `1.88×` its `0.0418`, under the predicted 2×. -/
theorem q1_split :
    from256k131 = from128k131 ∧ (ctl.getD 0 0 ≤ 10000) ∧ ¬ (ctl.getD 1 0 ≤ 10000) ∧
      ¬ 2 * clu4Kld ≤ mutClu4Kld ∧ 18 * clu4Kld ≤ 10 * mutClu4Kld := by
  decide

/-- **Q2 held.** The flat arm (`0.0386`, inside `[0.028, 0.045]`), `clu-s4` and `clu-s8` pass at 262144. -/
theorem q2_held :
    passes flatKld ∧ 280 ≤ flatKld ∧ flatKld ≤ 450 ∧ passes clu4Kld ∧ passes clu8Kld := by
  decide

/-- **Q3 refuted.** `tree-s8` (`0.0468`) and `tree-s4` (`0.0496`) pass at 262144, after failing at 131072
(`0.0549`, `0.0611`). The margin of `tree-s4` is `0.0004` over 64 positions with 19 of them above `0.05`. -/
theorem q3_refuted : passes tree8Kld ∧ passes tree4Kld ∧ 500 < 549 ∧ 500 < 611 := by decide

/-- **Q4 held.** `clu-s4` reads `674.0` bound vectors per refresh against the flat arm's `8181.8`: `12.1×`. The
model's `pages 262144 = 4091`, `flatScan = 8182` and `treeScan 4091 8 = 544`, `treeScan 4091 4 = 288` bound the
measured scans. -/
theorem q4_held :
    HattnT8.pages 262144 = 4091 ∧ HattnT8.climb 8 4091 = (16, 2) ∧ HattnT8.flatScan 4091 = 8182 ∧
      HattnT8.treeScan 4091 8 = 544 ∧ HattnT8.treeScan 4091 4 = 288 ∧ flatScan ≤ 10 * HattnT8.flatScan 4091 ∧
      tree8Scan ≤ 10 * HattnT8.treeScan 4091 8 ∧ tree4Scan ≤ 10 * HattnT8.treeScan 4091 4 ∧
      12 * clu4Scan ≤ flatScan := by
  decide

/-- **Q5 held.** `clu-s4`'s recall of the flat top-32 falls from `0.289` to `0.264`. -/
theorem q5_held : clu4Recall.getD 1 0 < clu4Recall.getD 0 0 := by decide

/-- **Q6 held.** `clu-s4` reads `0.0091` of dense bytes at 262144 (`0.0181` at 131072: reads per position are fixed,
`2t` doubles; the halving holds to the rounding of the printed ten-thousandths, 182 against 181). -/
theorem q6_held : clu4Bytes.getD 1 0 ≤ 100 ∧ 2 * clu4Bytes.getD 1 0 ≤ clu4Bytes.getD 0 0 + 2 := by decide

/-! ## Findings -/

/-- **Every arm passes the `0.05` gate at 262144 on this sample.** The tree arms' pass is within the sample's
noise (19 and 17 of 64 positions above `0.05`, against 29 and 26 at 131072): the gate verdict for the tree is not
monotone in `t`, and the 64-position mean is dominated by a few positions (`max` `0.37`–`0.70`). -/
theorem all_pass_262144 : passes flatKld ∧ passes tree4Kld ∧ passes tree8Kld ∧ passes clu4Kld ∧ passes clu8Kld := by
  decide

/-- **`clu-s4` cuts the flat scan 12.1× at 262144** (9.8× at 131072) at mean KLD `0.0418`; its scan grows 1.6×
(`4141 → 6740` tenths) while the flat scan doubles (`40858 → 81818`). -/
theorem cluster_scan_scaling :
    4141 * 16 < 6740 * 10 ∧ 2 * 40858 < 81818 + 100 ∧ 12 * clu4Scan ≤ flatScan ∧ ¬ 13 * clu4Scan ≤ flatScan := by
  decide

/-- **Per generated token at 262144, Qwen2.5-7B, f16 KV:** dense reads 15.03 GB (`2 · 262144 · stackBytes`);
`clu-s4` reads `0.0091` of it, at most 136.8 MB. -/
theorem bytes_7b_256k :
    2 * 262144 * HattnT8.stackBytes = 15032385536 ∧ 15032385536 * clu4Bytes.getD 1 0 / 10000 = 136794708 := by
  decide

end HattnT8Q8

end Perf
