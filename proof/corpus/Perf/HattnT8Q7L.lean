import Perf.HattnT8

/-!
# attention-cut T8 on Qwen2.5-7B-Instruct-1M at 65536 and 131072

the hattn-decode-tail note (T8 on the 7B at 128k section): the attention-cut T8 arms unchanged (`grp-fix2048-r8`,
`tree-s4`, `tree-s8`, `clu-s4`, `clu-s8`), the 64 tail positions before `t ∈ {32768, 65536, 131072}`, on
Qwen2.5-7B-Instruct-1M Q4_K_M (28 layers, 28 query and 4 KV heads, head dim 128). The dump is the f32 GPU dump
`~/hattn-long/g7-128k` (`examples/gpu_dump.rs`, `KV_T=131072`, `KV_QTAIL=32768,65536,131072`),
instrument `attn_cut_probe.hd128q`, one `t` per invocation (`g7t-<t>.out`, `g7t-m2.out`). Measured figures are
`model` inputs copied from those outputs: mean whole-model KLD and bytes/dense in ten-thousandths, bound vectors
per refresh in tenths, recall in thousandths, rel-L2 in units of `1e-8`. The cost model is `Perf.HattnT8`'s.
rustc, LLVM, the f32 dump kernels and the instrument are outside every statement.
-/

namespace Perf

namespace HattnT8Q7L

/-- Dense-arm tail control, max rel-L2 over q, k, v and 28 layers, at `t = 32768, 65536, 131072`. -/
def ctl : List ℕ := [1999, 4712, 7042]

/-- Mean KLD at `t = 32768, 65536, 131072`. -/
def flatKld : List ℕ := [38, 208, 242]

def tree4Kld : List ℕ := [49, 865, 611]

def tree8Kld : List ℕ := [47, 646, 549]

def clu4Kld : List ℕ := [43, 256, 270]

def clu8Kld : List ℕ := [40, 236, 227]

/-- The CPU-dump result at 32768 (`HattnT8Q7`), in the arm order flat, tree-s4, tree-s8, clu-s4, clu-s8. -/
def cpu32 : List ℕ := [38, 49, 47, 43, 40]

/-- The GPU-dump result at 32768, same order. -/
def gpu32 : List ℕ := [flatKld.head!, tree4Kld.head!, tree8Kld.head!, clu4Kld.head!, clu8Kld.head!]

/-- `HATTN_MUT=2` (ranking reversed at every level), mean KLD at 65536, same order. -/
def mut65 : List ℕ := [1364, 1366, 1331, 1359, 1329]

/-- The unmutated mean KLD at 65536, same order. -/
def base65 : List ℕ := [208, 865, 646, 256, 236]

/-- Bound vectors per refresh per KV head, tenths, at 65536 and 131072. -/
def flatScan65 : ℕ := 20378

def tree8Scan65 : ℕ := 3879

def clu4Scan65 : ℕ := 2927

def flatScan131 : ℕ := 40858

def tree4Scan131 : ℕ := 2708

def tree8Scan131 : ℕ := 5259

def clu4Scan131 : ℕ := 4141

def clu8Scan131 : ℕ := 5661

/-- Share of the flat scan's top-32 pages the index selects, thousandths: tree-s8 at 32768 (`HattnT8Q7`), 65536,
131072. -/
def tree8Recall : List ℕ := [625, 531, 452]

def clu4Recall : List ℕ := [456, 387, 289]

/-- Bytes/dense at 131072, ten-thousandths. -/
def flatBytes131 : ℕ := 198

def tree8Bytes131 : ℕ := 181

def clu4Bytes131 : ℕ := 181

def clu8Bytes131 : ℕ := 182

/-- Every entry of `l` is at most `0.05`. -/
def passes (l : List ℕ) : Prop := ∀ k ∈ l, k ≤ 500

instance (l : List ℕ) : Decidable (passes l) := by unfold passes; infer_instance

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8-7B-128K.md`, sha256 `449ab1b4…`) -/

/-- **Q1 held in part.** The dense tail reproduces the dump to under `1e-4` at every `t`, and the GPU dump gives the
CPU dump's mean KLD at 32768 on every arm. -/
theorem q1_control_held : (∀ c ∈ ctl, c ≤ 10000) ∧ gpu32 = cpu32 := by decide

/-- **Q1 refuted in its threshold.** The reversed ranking moves every arm's mean KLD at 65536 to `0.133`–`0.137`,
not above `0.2`. It is still at least 2× `tree-s8` and 6× the flat arm unmutated, so the control sees the
selection. -/
theorem q1_mut_threshold_refuted :
    (∀ k ∈ mut65, ¬ 2000 < k) ∧ 2 * tree8Kld.getD 1 0 ≤ mut65.getD 2 0 ∧
      6 * flatKld.getD 1 0 ≤ mut65.getD 0 0 ∧ base65 = [flatKld.getD 1 0, tree4Kld.getD 1 0, tree8Kld.getD 1 0,
        clu4Kld.getD 1 0, clu8Kld.getD 1 0] := by
  decide

/-- **Q2 held for the flat arm and `clu-s4`, refuted for `tree-s8`.** At 65536 and 131072 the flat arm and
`clu-s4` pass and `tree-s8` does not (`0.0646`, `0.0549`). -/
theorem q2_split : passes flatKld ∧ passes clu4Kld ∧ passes clu8Kld ∧ ¬ passes tree8Kld := by decide

/-- **Q3 refuted.** `tree-s4` fails: mean KLD `0.0865` at 65536 and `0.0611` at 131072. -/
theorem q3_refuted : ¬ passes tree4Kld ∧ ¬ (tree4Kld.getD 1 0 ≤ 500) ∧ ¬ (tree4Kld.getD 2 0 ≤ 500) := by decide

/-- **Q4 held.** `tree-s8` reads `525.9` bound vectors per refresh at 131072, under the model's 528
(`pages 131072 = 2043`, `climb 8 2043 = (8, 2)`); `clu-s4` reads under a quarter of the flat scan, which sits
under the model's `2 · 2043`. -/
theorem q4_held :
    HattnT8.pages 131072 = 2043 ∧ HattnT8.climb 8 2043 = (8, 2) ∧ HattnT8.treeScan 2043 8 = 528 ∧
      tree8Scan131 ≤ 10 * HattnT8.treeScan 2043 8 ∧ flatScan131 ≤ 10 * HattnT8.flatScan 2043 ∧
      4 * clu4Scan131 < flatScan131 := by
  decide

/-- The model bounds the measurement at 65536 too: `pages 65536 = 1019`, `climb 8 1019 = (4, 2)`. -/
theorem model_bounds_64k :
    HattnT8.pages 65536 = 1019 ∧ HattnT8.climb 8 1019 = (4, 2) ∧ HattnT8.treeScan 1019 8 = 520 ∧
      tree8Scan65 ≤ 10 * HattnT8.treeScan 1019 8 ∧ flatScan65 ≤ 10 * HattnT8.flatScan 1019 := by
  decide

/-- **Q5 held.** `tree-s8`'s recall of the flat top-32 falls with context: `0.625`, `0.531`, `0.452`. -/
theorem q5_held : tree8Recall = [625, 531, 452] ∧ 531 < 625 ∧ 452 < 531 ∧ 289 < 387 ∧ 387 < 456 := by decide

/-- **Q6 held.** `tree-s8` reads `0.0181` of dense bytes at 131072 and the flat arm `0.0198`. -/
theorem q6_held : tree8Bytes131 ≤ 190 ∧ flatBytes131 ≤ 215 := by decide

/-! ## Findings -/

/-- **The tree index stops holding the gate at 65536; the cluster index holds it to 131072.** At 65536 `tree-s4` and
`tree-s8` exceed the `0.05` KLD gate, while the flat arm and both cluster arms stay under it at both contexts. -/
theorem tree_fails_cluster_holds :
    ¬ tree4Kld.getD 1 0 ≤ 500 ∧ ¬ tree8Kld.getD 1 0 ≤ 500 ∧ passes flatKld ∧ passes clu4Kld ∧
      passes clu8Kld := by
  decide

/-- **`clu-s4` cuts the flat scan 9.8× at 131072** (4141 against 40858 tenths) at mean KLD `0.0270`; `clu-s8`
cuts it 7.2× at `0.0227`, which beats the flat arm's `0.0242`. -/
theorem cluster_cuts_scan_128k :
    9 * clu4Scan131 < flatScan131 ∧ ¬ 10 * clu4Scan131 < flatScan131 ∧ 7 * clu8Scan131 < flatScan131 ∧
      ¬ 8 * clu8Scan131 < flatScan131 ∧ clu8Kld.getD 2 0 < flatKld.getD 2 0 := by
  decide

/-- **The flat arm's quality drifts, not jumps.** Its mean KLD grows with `t` (`0.0038`, `0.0208`, `0.0242`) at a
fixed 2048-token far budget and stays under the gate at 131072, where the budget is 1/64 of the context against
1/16 at 32768. -/
theorem flat_drift : flatKld = [38, 208, 242] ∧ 38 < 208 ∧ 208 < 242 ∧ 242 ≤ 500 ∧ 2048 * 64 = 131072 := by
  decide

/-- **Per generated token at 131072, Qwen2.5-7B, f16 KV (measured ratio):** dense reads 7.52 GB
(`2 · 131072 · stackBytes`); the cluster arm that passes reads at most `0.0182` of it, 136.8 MB. -/
theorem bytes_7b_128k :
    2 * 131072 * HattnT8.stackBytes = 7516192768 ∧ 7516192768 * clu8Bytes131 / 10000 = 136794708 := by
  decide

end HattnT8Q7L

end Perf
