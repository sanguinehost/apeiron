import Perf.HattnT8

/-!
# attention-cut T8 sub-page summary unit on Qwen2.5-7B-Instruct-1M at 524288

the hattn-decode-tail note (sub-page summary section): `Perf.HattnT8Sum` mode 1 (exact mass, mean value row)
with the summary unit `u ∈ {64, 16, 4, 1}` keys (`KV_TAIL_SUMU`), the 64 tail positions before `t = 524288` of the
dump `g7-1m`, instrument `attn_cut_probe.hd128qu`, output
`g12-t524288-sum1-u<u>.out`. Mean whole-model KLD in ten-thousandths; `u = 64` is `Perf.HattnT8Sum.mode1`. Predictions:
`PREDICTIONS-T8-SUMU.md`, sha256 `2f795847…`. rustc, LLVM, the f32 dump kernels and the instrument are outside every
statement; one prompt, one model.
-/

namespace Perf

namespace HattnT8SumU

/-- Mean KLD at `u = 64, 16, 4, 1`. -/
def kld : List ℕ := [2389, 2140, 1416, 0]

/-- **U1 held (instrument control).** `u = 1` is the exact softmax over every key: mean KLD `0.0000`. -/
theorem u1_held : kld.getD 3 1 ≤ 10 := by decide

/-- **U2 held.** The KLD falls with the unit: `0.2389`, `0.2140`, `0.1416`. -/
theorem u2_held : kld.getD 2 0 < kld.getD 1 0 ∧ kld.getD 1 0 < kld.getD 0 0 := by decide

/-- **U3 refuted.** `u = 16` is `0.2140`, not under `0.10`; it also fails the `0.05` gate. -/
theorem u3_refuted : ¬ (kld.getD 1 0 < 1000) ∧ ¬ (kld.getD 1 0 ≤ 500) := by decide

/-- **U4 held (derived).** Bound scan 2 vectors per page plus 2 per unit: `u = 16` reads `119310` vectors per period
against `8 · 2 · 524288` dense, over `0.01` (`0.0142`); `u = 4` reads `315798`, `0.0376`. -/
theorem u4_held :
    HattnT8.period 2048 288 8 (2 * 8187 + 2 * 4 * 8187) = 119310 ∧
      8 * 2 * 524288 < 100 * 119310 ∧ 100 * 119310 < 2 * (8 * 2 * 524288) ∧
      HattnT8.period 2048 288 8 (2 * 8187 + 2 * 16 * 8187) = 315798 ∧
      3 * (8 * 2 * 524288) < 100 * 315798 := by
  decide

/-- **Finding: no summary unit above one key passes the gate.** At 4 keys per unit the KLD is `0.1416`, `2.8×` the
gate, at `3.76%` of dense reads: the omitted keys' values are not interchangeable at any granularity cheaper than
reading them. -/
theorem no_unit_passes : ∀ k ∈ kld.take 3, 500 < k ∧ 2 * 500 < 2 * 1416 := by decide

end HattnT8SumU

end Perf
