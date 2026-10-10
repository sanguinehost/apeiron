import Perf.HattnT8

/-!
# HATTN-T8 on Qwen2.5-7B-Instruct-1M at 524288 and 1048576

`docs/evidence/hattn-decode-tail.md` (T8 on the 7B at 1M section): the HATTN-T8 arms at a fixed 2048-token far budget
(`grp-fix2048-r8` flat, `clu-s4`, `tree-s8`), the 64 tail positions before `t ∈ {262144, 524288, 1048576}`, on
Qwen2.5-7B-Instruct-1M Q4_K_M. The dump is the f32 GPU dump `/home/socol/oc-tmp/hattn-long/g7-1m`
(`examples/gpu_dump.rs`, `KV_T=1048576`, `KV_QTAIL=262144,524288,1048576`), instrument `attn_cut_probe.hd128qf`, one arm
per invocation (`g9-t<t>-a<arm>.out`, `.time`). Measured figures are `model` inputs copied from those outputs: mean
whole-model KLD and bytes/dense in ten-thousandths, bound vectors per refresh in tenths, recall in thousandths, rel-L2
in units of `1e-8`. The cost model is `Perf.HattnT8`'s. rustc, LLVM, the f32 dump kernels and the instrument are outside
every statement. `tree-s4` and `clu-s8` were not run at these contexts.
-/

namespace Perf

namespace HattnT8Q9

/-- Dense-arm tail control, max rel-L2 over q, k, v and 28 layers, at `t = 262144` (read from the 1M dump),
`524288` and `1048576`. -/
def ctl : List ℕ := [17880, 140700, 265000]

/-- The 262144 control from the 256k dump (`HattnT8Q8.ctl`). -/
def ctl256k : ℕ := 17880

/-- Mean KLD at `t = 262144, 524288, 1048576`. -/
def flatKld : List ℕ := [386, 3370, 5874]

def clu4Kld : List ℕ := [418, 3237, 5826]

/-- `tree-s8`: 262144 is `HattnT8Q8.tree8Kld`. -/
def tree8Kld : List ℕ := [468, 3369, 6004]

/-- `HATTN_MUT=2` (ranking reversed) on `clu-s4` at 524288. -/
def mutClu4Kld524 : ℕ := 5193

/-- Median and positions over the gate (of 64) for `clu-s4` at 1048576. -/
def clu4Median1m : ℕ := 5161

def clu4Over1m : ℕ := 62

def flatOver : List ℕ := [12, 62, 62]

/-- Bound vectors per refresh per KV head, tenths, at `t = 262144, 524288, 1048576`. -/
def flatScan : List ℕ := [81818, 163738, 327578]

def clu4Scan : List ℕ := [6740, 11885, 22092]

def tree8Scan524 : ℕ := 5790

def tree8Scan1m : ℕ := 6473

/-- `clu-s4` recall of the flat top-32, thousandths, at the three contexts. -/
def clu4Recall : List ℕ := [264, 219, 242]

/-- Bytes/dense, ten-thousandths. -/
def clu4Bytes : List ℕ := [91, 46, 24]

def flatBytes : List ℕ := [109, 64, 42]

/-- One `clu-s4` invocation at 1048576: wall seconds (`g9-t1048576-a4-rss.time`) and peak resident kB. -/
def secs1m : ℕ := 411

def rssKb1m : ℕ := 20135468

/-- Every entry of `l` is at most `0.05`. -/
def passes (l : List ℕ) : Prop := ∀ k ∈ l, k ≤ 500

instance (l : List ℕ) : Decidable (passes l) := by unfold passes; infer_instance

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8-7B-1M.md`, sha256 `9b221a17…`) -/

/-- **Q1 held at 262144, refuted at 524288 and 1048576.** The 1M dump read at 262144 reproduces the 256k dump's control
exactly. The registered bounds (`≤ 4e-4`, `≤ 8e-4`) fail: the control is `1.407e-3` and `2.65e-3`. The mutation
(`HATTN_MUT=2`) raises `clu-s4` at 524288 by `1.60×`, under the registered `2×`. -/
theorem q1_split :
    ctl.head! = ctl256k ∧ ¬ ctl.getD 1 0 ≤ 40000 ∧ ¬ ctl.getD 2 0 ≤ 80000 ∧
      ¬ 2 * clu4Kld.getD 1 0 ≤ mutClu4Kld524 ∧ clu4Kld.getD 1 0 < mutClu4Kld524 ∧
      10 * mutClu4Kld524 < 17 * clu4Kld.getD 1 0 := by
  decide

/-- **Q2 half refuted, half held.** Flat at 524288 is `0.337`, not in `[0.040, 0.070]`; flat at 1048576 is over the gate. -/
theorem q2_split :
    ¬ (400 ≤ flatKld.getD 1 0 ∧ flatKld.getD 1 0 ≤ 700) ∧ ¬ flatKld.getD 2 0 ≤ 500 := by decide

/-- **Q3 held for the arms run.** `clu-s4` is over the gate at 524288, and flat, `clu-s4` and `tree-s8` are over it at
1048576. -/
theorem q3_held :
    ¬ clu4Kld.getD 1 0 ≤ 500 ∧ ¬ flatKld.getD 2 0 ≤ 500 ∧ ¬ clu4Kld.getD 2 0 ≤ 500 ∧
      ¬ tree8Kld.getD 2 0 ≤ 500 := by
  decide

/-- **Q4 held.** `clu-s4` reads under `1/14` of the flat scan at 1048576 (`14.8×`), and the model bounds both scans:
`pages 1048576 = 16379`, `flatScan 16379 = 32758`, `treeScan 16379 8 = 776`. -/
theorem q4_held :
    14 * clu4Scan.getD 2 0 ≤ flatScan.getD 2 0 ∧ ¬ 15 * clu4Scan.getD 2 0 ≤ flatScan.getD 2 0 ∧
      HattnT8.pages 1048576 = 16379 ∧ flatScan.getD 2 0 ≤ 10 * HattnT8.flatScan 16379 ∧
      tree8Scan1m ≤ 10 * HattnT8.treeScan 16379 8 := by
  decide

/-- The model bounds the 524288 scans: `pages 524288 = 8187`, `climb 8 8187 = (2, 3)`, `treeScan 8187 8 = 772`. -/
theorem model_bounds_512k :
    HattnT8.pages 524288 = 8187 ∧ HattnT8.climb 8 8187 = (2, 3) ∧ HattnT8.treeScan 8187 8 = 772 ∧
      flatScan.getD 1 0 ≤ 10 * HattnT8.flatScan 8187 ∧ tree8Scan524 ≤ 10 * HattnT8.treeScan 8187 8 := by
  decide

/-- **Q5 held, not monotone.** `clu-s4` recall at 1048576 (`0.242`) is under its 262144 value (`0.264`); the 524288 value
(`0.219`) is lower than both. -/
theorem q5_held : clu4Recall.getD 2 0 < clu4Recall.getD 0 0 ∧ clu4Recall.getD 1 0 < clu4Recall.getD 2 0 := by decide

/-- **Q6 held.** `clu-s4` reads `0.0024` of dense bytes at 1048576. -/
theorem q6_held : clu4Bytes.getD 2 0 ≤ 25 := by decide

/-- **Q7 held.** One invocation at 1048576 completes in `411 s` (under 570) at `20.1 GB` peak resident (under 24). -/
theorem q7_held : secs1m ≤ 570 ∧ rssKb1m ≤ 24000000 := by decide

/-! ## Findings -/

/-- **The fixed 2048-token budget stops holding between 262144 and 524288.** Flat and `clu-s4` pass at 262144 and are over
the gate by `6.7×` and `6.5×` at 524288 and `11.7×` at 1048576; `tree-s8` fails at all three of the larger contexts. -/
theorem budget_cliff :
    passes [flatKld.getD 0 0] ∧ passes [clu4Kld.getD 0 0] ∧ ¬ passes [flatKld.getD 1 0] ∧
      ¬ passes [clu4Kld.getD 1 0] ∧ 6 * 500 < flatKld.getD 1 0 ∧ 11 * 500 < flatKld.getD 2 0 ∧
      2048 * 256 = 524288 ∧ 2048 * 512 = 1048576 := by
  decide

/-- **The failure is global at 1048576.** `clu-s4` is over the gate at 62 of 64 positions with median `0.516`; at
262144 flat was over at 12 of 64. -/
theorem global_failure_1m : clu4Over1m = 62 ∧ 10 * 500 < clu4Median1m ∧ flatOver.head! = 12 ∧ 62 ≤ flatOver.getD 2 0 := by
  decide

/-- **The cluster index tracks the flat scan at every context run.** `clu-s4` is `0.0032` above flat at 262144 and
`0.0133` and `0.0048` below it at 524288 and 1048576: a 14.8× smaller scan at the flat arm's quality. -/
theorem clu4_tracks_flat :
    clu4Kld.getD 0 0 ≤ flatKld.getD 0 0 + 32 ∧ clu4Kld.getD 1 0 ≤ flatKld.getD 1 0 ∧
      clu4Kld.getD 2 0 ≤ flatKld.getD 2 0 := by
  decide

/-- **Per generated token at 1048576, Qwen2.5-7B, f16 KV (measured ratio):** dense reads 60.1 GB; `clu-s4` reads at most
`0.0024` of it, 144.3 MB, at a mean KLD that is over the gate. -/
theorem bytes_7b_1m_measured :
    2 * 1048576 * HattnT8.stackBytes = 60129542144 ∧ 60129542144 * 24 / 10000 = 144310901 := by
  decide

end HattnT8Q9

end Perf
