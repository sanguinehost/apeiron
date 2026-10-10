import Perf.HattnT8

/-!
# attention-cut T8 far-budget scaling on Qwen2.5-7B-Instruct-1M at 524288

the hattn-decode-tail note (far-budget scaling section): the flat arm (`grp-fix<b>-r8`) and `clu-s4` at far
budget `b ∈ {4096, 8192, 16384}` in place of 2048 (`KV_TAIL_FARB`), the 64 tail positions before `t = 524288` of the dump
`~/hattn-long/g7-1m`, instrument `attn_cut_probe.hd128qg`, one arm per invocation
(`g10-t524288-b<b>-a<arm>.out`). Measured figures are `model` inputs: mean whole-model KLD and bytes/dense in
ten-thousandths. The `b = 2048` entries are `Perf.HattnT8Q9`'s. Predictions: `PREDICTIONS-T8-FARB.md`, sha256 `492f54aa…`.
rustc, LLVM, the f32 dump kernels and the instrument are outside every statement; one prompt, one model.
-/

namespace Perf

namespace HattnT8Farb

/-- Flat-arm mean KLD at `b = 2048, 4096, 8192, 16384`. -/
def flatKld : List ℕ := [3370, 2979, 2612, 2218]

/-- Flat-arm bytes/dense at the same `b`. -/
def flatBytes : List ℕ := [64, 103, 181, 338]

def clu4Kld16k : ℕ := 2903

def clu4Bytes16k : ℕ := 108

/-- **F1 refuted (above the bracket).** The flat arm at `b = 4096` has mean KLD `0.2979`, over the `0.25` bracket top. -/
theorem f1_refuted : ¬ (flatKld.getD 1 0 ≤ 2500) := by decide

/-- **F2 refuted.** The flat arm at `b = 16384` (1/32 of `t`) has mean KLD `0.2218`, over the `0.05` gate. -/
theorem f2_refuted : ¬ (flatKld.getD 3 0 ≤ 500) := by decide

/-- **F3 refuted.** `clu-s4` at `b = 16384` is `0.0685` above the flat arm, not within `0.02`. -/
theorem f3_refuted : ¬ (clu4Kld16k ≤ flatKld.getD 3 0 + 200) ∧ clu4Kld16k = flatKld.getD 3 0 + 685 := by decide

/-- **F4 held.** Mean KLD falls at every doubling of `b`, by between `0.0366` and `0.0394`. -/
theorem f4_held :
    flatKld = [3370, 2979, 2612, 2218] ∧ 3370 - 2979 = 391 ∧ 2979 - 2612 = 367 ∧ 2612 - 2218 = 394 := by decide

/-- **Budget scaling does not reach the gate.** Each doubling of `b` buys at most `0.04`. Four more doublings from
`b = 16384` at that rate leave the mean KLD at `0.0618`, over the gate; the fifth doubling is `b = 2^19 = 524288 = t`,
the dense read. Extrapolation of four measured steps (grade `model`), not a measurement past `b = 16384`. -/
theorem doublings_to_dense :
    2218 - 4 * 400 = 618 ∧ ¬ (618 ≤ 500) ∧ 16384 * 2 ^ 5 = 524288 := by decide

/-- Bytes/dense rises with `b` (`0.0103`, `0.0181`, `0.0338` at `4096`, `8192`, `16384`), so the KLD bought per doubling is
paid for in bytes. -/
theorem bytes_double : flatBytes.getD 1 0 < flatBytes.getD 2 0 ∧ flatBytes.getD 2 0 < flatBytes.getD 3 0 := by decide

end HattnT8Farb

end Perf
