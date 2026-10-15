import Perf.Receipts.Onebit

/-!
# Seeded trellis generator vs E8 at 1 b/w: verdicts

`tools/onebit/trellis_gen.py` (sha256 `a364850b…`) codes 21 linears of Qwen2.5-0.5B (layers 0, 11, 23; q, k, v, o,
gate, up, down) against the arm's own objective `J(W, Wq) = tr((W − Wq) H (W − Wq)ᵀ)` in the Hadamard-rotated frame,
with `H` the damped Gram of calibration chunks `[0, 20)`. Two codes at one stored bit per weight plus one f16 scale
per 128-group:

- E8: `gptq_e8` (the arm's lattice step), at beam 1 and beam 4, with a least-squares scale refit.
- Trellis: a bitshift trellis of `L` state bits whose emitted value is the 1MAD hash of the state (QTIP, Tseng et al.,
  arXiv 2406.11235), i.e. a pseudo-random codebook fixed by a seed. Encoding is Viterbi over 16-column blocks with
  per-survivor GPTQ feedback; `L ∈ {8, 12}` plus `L/n` start-state bits per row.

Prediction `trellis-2` (median `J_trellis_L12 / J_e8_beam1 ∈ [1.2, 1.8]`; falsifier `≤ 1.0`) is confirmed: the
hashed generator loses on every linear, by at least 1.49× against beam 1 and 1.60× against beam 4. Lengthening the
trellis from 8 to 12 state bits moves the median by under 1%. The search itself is not the loss: the same Viterbi
over a plain sign alphabet (`v(s) = ±1` by the last bit) beats E8 beam 1 on layer 0 q (0.867), and a one-state
sign Viterbi does too (0.922), while scalar GPTQ signs lose (1.227). At one bit per weight the hashed Gaussian-like
codebook spends its rate on shaping that sign coding gets without it.

Grades: every statement is a `proved` arithmetic fact about `measured` constants (`Perf/Receipts/Onebit.lean`).
-/

namespace Perf.Trellis

open Receipts

/-- Stored bits per weight in a row of `n` weights: one code bit, `L` start-state bits per row, one f16 scale per
128 weights. E8 is the case `L = 0`. -/
def bpw (L n : ℚ) : ℚ := 1 + L / n + 16 / 128

/-- **Cost.** The trellis stores more bits than E8 on every width the receipt codes (896 and 4864 inputs). -/
theorem trellis_bits :
    bpw 0 896 = 9 / 8 ∧ bpw 0 896 < bpw 8 4864 ∧ bpw 8 4864 < bpw 12 4864 ∧ bpw 12 4864 < bpw 12 896 ∧
    bpw 12 896 < 1139 / 1000 := by
  norm_num [bpw]

/-- **Prediction `trellis-2` confirmed.** The L12 median against E8 beam 1 lies in `[1.2, 1.8]` over all 21
linears. -/
theorem trellis2_confirmed :
    w18_trellis2.linears = 21 ∧
    (6 / 5 : ℚ) ≤ w18_trellis2.medianL12Beam1 ∧ w18_trellis2.medianL12Beam1 ≤ 9 / 5 := by
  norm_num [w18_trellis2.linears, w18_trellis2.medianL12Beam1]

/-- **Every linear loses.** The minimum ratio over the 21 linears exceeds 1.49 against E8 beam 1 and 1.6 against
beam 4, at both trellis lengths. -/
theorem trellis_loses_everywhere :
    (149 / 100 : ℚ) < w18_trellis2.minL8Beam1 ∧ (149 / 100 : ℚ) < w18_trellis2.minL12Beam1 ∧
    (8 / 5 : ℚ) < w18_trellis2.minL8Beam4 ∧ (8 / 5 : ℚ) < w18_trellis2.minL12Beam4 := by
  norm_num [w18_trellis2.minL8Beam1, w18_trellis2.minL12Beam1, w18_trellis2.minL8Beam4,
    w18_trellis2.minL12Beam4]

/-- **Trellis length does not close the gap.** Four more state bits (16× the states) improve the median by under
`1/100` and the geometric mean by under `1/50`. -/
theorem trellis_length_flat :
    w18_trellis2.medianL12Beam1 < w18_trellis2.medianL8Beam1 ∧
    w18_trellis2.medianL8Beam1 - w18_trellis2.medianL12Beam1 < 1 / 100 ∧
    w18_trellis2.geomeanL8Beam1 - w18_trellis2.geomeanL12Beam1 < 1 / 50 := by
  norm_num [w18_trellis2.medianL12Beam1, w18_trellis2.medianL8Beam1, w18_trellis2.geomeanL8Beam1,
    w18_trellis2.geomeanL12Beam1]

/-- **The loss is the codebook, not the search** (layer 0 q). Viterbi over a sign alphabet beats E8 beam 1 at
`L = 8` and at one state; scalar GPTQ signs do not. -/
theorem sign_search_wins :
    w18_trellis2.signL8Ratio < w18_trellis2.signL1Ratio ∧ w18_trellis2.signL1Ratio < 1 ∧
    1 < w18_trellis2.gptqSignRatio := by
  norm_num [w18_trellis2.signL8Ratio, w18_trellis2.signL1Ratio, w18_trellis2.gptqSignRatio]

/-- **Instrument.** The Viterbi path cost equals the reported `J` to below `10⁻⁷` relative. -/
theorem trellis_identity : w18_trellis2.identityRel < 1 / 10000000 := by
  norm_num [w18_trellis2.identityRel]

end Perf.Trellis
