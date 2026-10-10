import Mathlib.Data.Nat.Basic

/-!
# HATTN-T8 content-cell indexes: F1 (oracle-ranked k-means cells) and F2 (inverted multi-index)

`docs/evidence/hattn-decode-tail.md` (content-cell section): Qwen2.5-7B-Instruct-1M Q4_K_M, prose dump `g7-1m`,
64 tail positions, `t ∈ {262144, 524288, 1048576}`. F1 is `KV_TAIL_CELL=512 KV_TAIL_FARB=2048 KV_TAIL_CELLR=R`
(cells ranked by the true softmax mass, whole cells read until `R` keys, no rerank). F2 is
`KV_TAIL_MI=1024 KV_TAIL_FARB=8192 KV_TAIL_CELLR=R` (top-`R` keys by the two block scores, exact top-8192 rerank).
Mean whole-model KLD and bytes/dense are in ten-thousandths, copied from `g25-<arm>-t<t>.out`; gate `0.05 = 500`.
Interpolated `R*` values are log-linear in `(R, KLD)` between the two bracketing runs. All figures are `model`
inputs; rustc, LLVM and the instrument are outside every statement.
-/

namespace Perf

namespace HattnT8Cells

/-! ## F1, `s = 512` -/

def f1_262k_r4096 : ℕ := 49

def f1_262k_r8192 : ℕ := 27

def f1_262k_r16384 : ℕ := 14

def f1_524k_r8192 : ℕ := 623

def f1_524k_r16384 : ℕ := 317

def f1_1m_r8192 : ℕ := 1221

def f1_1m_r16384 : ℕ := 725

def f1_1m_r32768 : ℕ := 387

def f1_524k_r16384_bytes : ℕ := 877

/-- Log-linear interpolated `R*` (keys), `t = 524288` and `t = 1048576`. -/
def f1Rstar524k : ℕ := 10266

def f1Rstar1m : ℕ := 24694

/-! ## F2, `K = 1024`, `c = 8192` -/

def f2_262k_r8192 : ℕ := 65

def f2_262k_r16384 : ℕ := 38

def f2_262k_r32768 : ℕ := 30

def f2_524k_r16384 : ℕ := 973

def f2_524k_r32768 : ℕ := 634

def f2_524k_r65536 : ℕ := 496

def f2_1m_r16384 : ℕ := 1035

def f2_1m_r32768 : ℕ := 908

def f2_1m_r65536 : ℕ := 890

def f2_524k_r65536_bytes : ℕ := 1563

/-! ## Controls and references -/

def f1Mut2 : ℕ := 5300

def f2Mut1_r32768 : ℕ := 2402

def lowbit4Bytes : ℕ := 1407

def f1a_elapsed : ℕ := 570

def runBudget : ℕ := 570

/-! ## Verdicts on the registered predictions (`PREDICTIONS-T8-CELLS.md`, sha256 `290bbc0db534…`) -/

/-- **F1a unmeasured.** The `s = 64` run reached the 570 s budget (`exit=124`); a timeout is not a result. -/
theorem f1a_unmeasured : runBudget ≤ f1a_elapsed := by decide

/-- **F1b refuted.** Registered KLD `0.12 [0.08, 0.18]`; measured `0.0623` is below the range. -/
theorem f1b_refuted : ¬ 800 ≤ f1_524k_r8192 := by decide

/-- **F1c ratio refuted.** `R*/t ≥ 0.05` is registered. The sweep brackets `R*` in `(8192, 16384]` at
`t = 524288`, so `R*/t ≤ 16384 / 524288 < 0.05` without interpolation. -/
theorem f1c_ratio_refuted_524k : 500 < f1_524k_r8192 ∧ f1_524k_r16384 ≤ 500 ∧ 16384 * 20 < 524288 := by decide

/-- Same at `t = 1048576`: `R* ∈ (16384, 32768]`, so `R*/t ≤ 1/32 < 0.05`. -/
theorem f1c_ratio_refuted_1m : 500 < f1_1m_r16384 ∧ f1_1m_r32768 ≤ 500 ∧ 32768 * 20 < 1048576 := by decide

/-- **F1c exponent refuted under the stated interpolation.** Registered `α ∈ [0.7, 1.0]`; the interpolated
`R*` more than doubles when `t` doubles (`α ≈ 1.27`). The brackets alone do not decide this. -/
theorem f1c_alpha_above_one : 2 * f1Rstar524k < f1Rstar1m := by decide

/-- **F1 control held.** Reverse cell order: KLD `0.5300 > 0.2` at the same read budget. -/
theorem f1_control_held : 2000 < f1Mut2 := by decide

/-- **F2a held.** Registered `R* = 40k [16k, 100k]`; the sweep brackets `R* ∈ (32768, 65536]` at `t = 524288`. -/
theorem f2a_held : 500 < f2_524k_r32768 ∧ f2_524k_r65536 ≤ 500 ∧ 16000 ≤ 32768 ∧ 65536 ≤ 100000 := by decide

/-- **F2b not decided.** `R*` at `t = 1048576` has no upper bracket (`R = 65536` is still above the gate), so
only `α > 0` follows: `R*(1M) > 65536 ≥ R*(524k)`. -/
theorem f2b_lower_bound_only :
    500 < f2_1m_r16384 ∧ 500 < f2_1m_r32768 ∧ 500 < f2_1m_r65536 ∧ f2_524k_r65536 ≤ 500 := by decide

/-- **F2c held.** Registered KLD `0.055 [0.044, 0.066]` at `R = 32768`, `t = 524288`; measured `0.0634`. -/
theorem f2c_held : 440 ≤ f2_524k_r32768 ∧ f2_524k_r32768 ≤ 660 := by decide

/-- **F2d held.** Registered bytes/dense at pass `0.08 [0.03, 0.19]`; first passing run (`R = 65536`) reads
`0.1563`. -/
theorem f2d_held : 300 ≤ f2_524k_r65536_bytes ∧ f2_524k_r65536_bytes ≤ 1900 := by decide

/-- **F2 control held.** Random cell ids at `R = 32768`: KLD `0.2402`, over 3 times the real arm (`0.0634`) and
over 4 times the gate. -/
theorem f2_control_held : 3 * f2_524k_r32768 < f2Mut1_r32768 ∧ 4 * 500 < f2Mut1_r32768 := by decide

/-! ## Decision rules -/

/-- **F1 kill not met.** The rule needs `R*/t ≥ 0.05` at both `s`; at `s = 512` the ratio is under `1/32` at
both lengths with a bracket. -/
theorem f1_kill_not_met : 16384 * 20 < 524288 ∧ 32768 * 20 < 1048576 := by decide

/-- **F2 kill met by its second clause.** `R*(F2) ≥ 2 · R*(F1)` at both bracketed lengths, from the brackets
alone: `t = 524288`, `R*(F2) > 32768 = 2 · 16384 ≥ 2 · R*(F1)`; `t = 1048576`, `R*(F2) > 65536 = 2 · 32768
≥ 2 · R*(F1)`. -/
theorem f2_kill_met :
    (500 < f2_524k_r32768 ∧ f1_524k_r16384 ≤ 500 ∧ 2 * 16384 = 32768) ∧
    (500 < f2_1m_r65536 ∧ f1_1m_r32768 ≤ 500 ∧ 2 * 32768 = 65536) := by decide

/-! ## Findings -/

/-- At the first passing `R`, F2 reads more than the 4-bit scan that passes at `0.1407`. -/
theorem f2_pass_costs_more_than_scan : lowbit4Bytes < f2_524k_r65536_bytes := by decide

/-- F2 stays above the gate across `R ∈ {16384, 32768, 65536}` at `t = 1048576` and its KLD flattens
(`0.0908 → 0.0890` for the last doubling). -/
theorem f2_flat_at_1m : 2 * (f2_1m_r32768 - f2_1m_r65536) < f2_1m_r65536 - 400 ∧ 500 < f2_1m_r65536 := by decide

/-- Oracle-ranked cells are not the cost: F1 passes at `R = 16384` (`t = 524288`) while reading `0.0877`
of dense, more than the `0.053` the exact top-8192 oracle needs. -/
theorem f1_pass_bytes_above_oracle : 530 < f1_524k_r16384_bytes ∧ f1_524k_r16384 ≤ 500 := by decide

end HattnT8Cells

end Perf
