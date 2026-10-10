import Perf.HattnT8Layer

/-!
# attention-cut T8 on a code prompt at 524288 (Qwen2.5-7B-Instruct-1M)

the hattn-decode-tail note (code-prompt section): the attention-cut T8 flat arm `grp-fix2048-r8` at `t = 524288` on a
524288-token prompt of Lean and Rust sources (`code8.txt`, sha256 `59eaa07e…`, files in sorted order, each headed by its
path) in place of the web-prose prompt (`g7-1m`), dump `g7-code` (`examples/gpu_dump.rs`, `KV_PROSE=code8.txt`), instrument
`attn_cut_probe.hd128qb`, output `g16-code-524288.out`. Mean whole-model KLD in ten-thousandths, rel-L2 in units of
`1e-8`. Predictions: `PREDICTIONS-T8-CODE.md`, sha256 `df3c6740…`. rustc, LLVM, the f32 dump kernels and the instrument are
outside every statement; one code prompt, one model.
-/

namespace Perf

namespace HattnT8Code

/-- Dense-arm tail control, max rel-L2 over q, k, v and 28 layers. -/
def ctl : ℕ := 66440

/-- Flat arm: mean KLD, median KLD, 90th-percentile KLD (ten-thousandths) and positions of 64 above `0.05`. -/
def flatKld : ℕ := 1273

def flatMedian : ℕ := 15

def flatP90 : ℕ := 3088

def flatOver : ℕ := 15

/-- Flat arm top-1 agreement with dense, of 64 positions. -/
def flatTop1 : ℕ := 60

/-- The prose run at 524288 (`HattnT8Layer.allFlat`, `g9-t524288-a1.out`): mean KLD, median KLD and positions of 64
above `0.05`. -/
def proseKld : ℕ := HattnT8Layer.allFlat

def proseMedian : ℕ := 2386

def proseOver : ℕ := 62

/-- The top-1 threshold registered in `PREDICTIONS-T8-CODE.md` (42 of 64). The prose run's own figure is 39 of 64
(`g9-t524288-a1.out`); the verdict compares against the registered number. -/
def proseTop1 : ℕ := 42

def proseTop1Measured : ℕ := 39

/-- **K1 held.** The dense tail reproduces the dump to under `1e-2`. -/
theorem k1_held : ctl < 1000000 := by decide

/-- **K2 held.** Code is easier than prose: the flat arm's mean KLD is below the prose figure. -/
theorem k2_held : flatKld < proseKld := by decide

/-- **K3 held.** The flat arm at a 2048-token far budget still fails the `0.05` gate on code. -/
theorem k3_held : 500 < flatKld := by decide

/-- **K4 held.** Top-1 agreement is above the prose figure. -/
theorem k4_held : proseTop1 < flatTop1 := by decide

/-- **Finding: the code prompt is 2.6 times lower than prose and 2.5 times over the gate.** -/
theorem between : 2 * flatKld < proseKld ∧ ¬ 3 * 500 < flatKld ∧ 2 * 500 < flatKld := by decide

/-- **Finding: the failure is a tail, not a shift.** The median position passes the gate by a factor of 30, the
90th percentile fails it six-fold, and 15 of 64 positions fail. -/
theorem heavy_tail : 30 * flatMedian ≤ 500 ∧ 6 * 500 < flatP90 ∧ 15 = flatOver ∧ flatOver < 64 := by decide

/-- **Finding: on code most positions pass.** The median KLD is over 100 times below the prose median and 15 positions
fail against 62, while the mean stays over the gate; the prose figure 42 in the prediction is above the prose run's
39 of 64, and code's 60 clears both. -/
theorem code_vs_prose : 100 * flatMedian < proseMedian ∧ flatOver < proseOver ∧ proseTop1Measured < proseTop1 ∧
    proseTop1 < flatTop1 := by
  decide

end HattnT8Code

end Perf
