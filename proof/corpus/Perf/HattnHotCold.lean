import Mathlib.Data.Nat.Basic

/-!
# HATTN-HOTCOLD: 4-bit key scan, top-8192 per KV head, at 524288 and 1048576 (Qwen2.5-7B-Instruct-1M)

`docs/evidence/hattn-decode-tail.md` (hot/cold decode section). Stage 1: `attn_cut_probe.hd128cl` (sha256
`f248f46584d8`), `KV_TAIL_LOWBIT=4 KV_TAIL_FARB=8192`, prose `g7-1m`, 64 tail positions; KLD in ten-thousandths, gate
`0.05 = 500`. Stage 2: `examples/hotcold_decode.rs`, 28 layers, CPU, 32 threads, 5 repetitions; attention time per
decode step in tenths of a millisecond (mean). Non-attention time `90.1 ms` is `engine_profile` on the same GGUF.
All figures are `model` inputs; rustc, LLVM, the CPU and the instruments are outside every statement.
-/

namespace Perf

namespace HattnHotCold

def gate : ℕ := 500

def kld524k : ℕ := 231

def kld1m : ℕ := 471

def top1_524k : ℕ := 59

def top1_1m : ℕ := 56

/-! ## Stage 2, attention per decode step (0.1 ms) -/

def dense131k : ℕ := 6936
def hot131k : ℕ := 3515
def floor131k : ℕ := 1677
def dense524k : ℕ := 30364
def hot524k : ℕ := 4837
def floor524k : ℕ := 6456
def dense1m : ℕ := 64268
def hot1m : ℕ := 6599
def floor1m : ℕ := 12843
def scan1m : ℕ := 1681
def nonAttn : ℕ := 901

/-! ## Controls (rel-L2, ten-thousandths) -/

def selectAllErr : ℕ := 0
def mutationMin : ℕ := 670
def b8192MaxErr : ℕ := 240

/-! ## Verdicts -/

/-- **Stage 1 passes** at both depths. -/
theorem gate_passes : kld524k ≤ gate ∧ kld1m ≤ gate := by decide

/-- At `t = 1048576` the margin is under `0.003`. -/
theorem thin_margin_1m : gate - kld1m < 30 := by decide

/-- Hot/cold attention at `t ≥ 524288` is faster than one streaming read of all K and V, the lower bound for any
dense pass on this CPU. -/
theorem hot_beats_read_floor : hot524k < floor524k ∧ hot1m < floor1m := by decide

/-- At `t = 131072` it is not: hot/cold `351.5 ms` against the read floor `167.7 ms`. -/
theorem hot_loses_floor_131k : floor131k < hot131k := by decide

/-- Against the measured dense arm: `9.7×` at `t = 1048576`. -/
theorem hot_vs_dense_1m : 9 * hot1m < dense1m := by decide

/-- Whole-token time at `t = 1048576` (derived: attention + non-attention) is `750.0 ms`, above `1.3` tokens/s. -/
theorem tok_rate_1m : 13 * (hot1m + nonAttn) < 100000 := by decide

/-- The 4-bit scan is under a third of the hot attention at `t = 1048576`; select, gather and attention are the
rest. -/
theorem scan_under_third_1m : 3 * scan1m < hot1m := by decide

/-- **Controls held.** Select-all equals dense; the lowest-score selection's smallest error exceeds the largest
B = 8192 error. -/
theorem controls_held : selectAllErr = 0 ∧ b8192MaxErr < mutationMin := by decide

end HattnHotCold

end Perf
