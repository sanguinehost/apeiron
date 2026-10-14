import Perf.Receipts.Onebit

/-!
# Learned register angles against the fixed Hadamard: verdicts (`quantum-register` W1-T1)

Board: `quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs`. Epic: `docs/epics/quantum-register.md`.

`tools/onebit/kan_probe.py` (sha256 `981e9fc7…`) writes each 128-group rotation as a register of `2^7` amplitudes:
seven stages of 64 reflections `[[cos t, sin t], [sin t, −cos t]] = cos t · σ_z + sin t · σ_x` (a meridian `Σ·n`,
Lohmiller–Slotine arXiv 2405.06328 eq. 38). At `t = π/4` every stage is the normalised butterfly and the register is
`fwht128`. Angles are tied per input site (q/k/v input, o input, gate/up input, down input) and per butterfly, learned
by Adam on the Hessian-weighted E8 error with the code detached, stored at 16 bits, and scored by the real `gptq_e8`
(act-order, wscale, beam 4) on the calibration Gram and on a held-out Gram of disjoint chunks.

Prediction `kan-probe-1` is refuted. The summed J is larger with the learned angles on both Grams
(`kan_probe_sum_loses`). The geometric mean over the 96 (layer, site) solves is below one (`kan_probe_geo_wins`):
the wins are on small-J solves. By site, the o input is the one win of size (`kan_probe_attn_out_in`: 8.5% on
held-out J); the two FFN inputs lose on both Grams (`kan_probe_ffn_loses`). The probe's surrogate is per-layer; the
next arm trains the angles on end-to-end KL inside the finishing loop.

The angle side information costs `688128` bits, `0.00192` b/w over `357826560` linear weights, within the `0.00194`
b/w slack under 1.18 at the arm's `1.178061` b/w (`kan_probe_angle_budget`).

Not claimed: anything about held-out KL or outcomes. No model was rebuilt with these angles; J is the per-linear
quantization error under the input Gram, not the model loss.
-/

namespace Perf.QuantumRotation

open Receipts

/-- The register: seven stages of 64 butterflies, one 16-bit angle each, four input sites, 24 layers. -/
def stages : ℕ := 7
def butterflies : ℕ := 64
def angleBits : ℕ := 16
def sites : ℕ := 4
def layers : ℕ := 24

/-- **Protocol and identity control.** 96 solves (24 layers × 4 sites); the all-`π/4` register equals `fwht128` to
`10⁻¹²` in float64. -/
theorem kan_probe_protocol :
    qreg_kan_probe.rows = layers * sites ∧
    qreg_kan_probe.angleBitsFullModel = stages * butterflies * angleBits * sites * layers ∧
    qreg_kan_probe.identityMaxAbs < 1 / 10 ^ 12 := by
  refine ⟨rfl, rfl, ?_⟩
  norm_num [qreg_kan_probe.identityMaxAbs]

/-- **Angle budget.** `688128 / 357826560 < 0.00194`, the slack under 1.18 b/w at the arm's 1.178061. -/
theorem kan_probe_angle_budget :
    (qreg_kan_probe.angleBitsFullModel : ℚ) / qreg_kan_probe.linearParams < 194 / 100000 := by
  norm_num [qreg_kan_probe.angleBitsFullModel, qreg_kan_probe.linearParams]

/-- **Summed J loses.** Learned over Hadamard, summed over all 96 solves: above one on both Grams. -/
theorem kan_probe_sum_loses : 1 < qreg_kan_probe.ratioCal ∧ 1 < qreg_kan_probe.ratioEv := by
  norm_num [qreg_kan_probe.ratioCal, qreg_kan_probe.ratioEv]

/-- **Geometric mean wins.** Over the 96 per-solve ratios: below `0.98` on both Grams. -/
theorem kan_probe_geo_wins : qreg_kan_probe.geoCal < 98 / 100 ∧ qreg_kan_probe.geoEv < 98 / 100 := by
  norm_num [qreg_kan_probe.geoCal, qreg_kan_probe.geoEv]

/-- **The o input wins.** Summed over its 24 layers, learned J is below `0.92 ×` Hadamard J on both Grams. -/
theorem kan_probe_attn_out_in :
    qreg_kan_probe.attnOutInJKanCal < 92 / 100 * qreg_kan_probe.attnOutInJHadCal ∧
    qreg_kan_probe.attnOutInJKanEv < 92 / 100 * qreg_kan_probe.attnOutInJHadEv := by
  norm_num [qreg_kan_probe.attnOutInJKanCal, qreg_kan_probe.attnOutInJHadCal, qreg_kan_probe.attnOutInJKanEv,
    qreg_kan_probe.attnOutInJHadEv]

/-- **The q/k/v input wins narrowly.** Below one, above `0.99`, on both Grams. -/
theorem kan_probe_attn_in :
    qreg_kan_probe.attnInJKanCal < qreg_kan_probe.attnInJHadCal ∧
    99 / 100 * qreg_kan_probe.attnInJHadCal < qreg_kan_probe.attnInJKanCal ∧
    qreg_kan_probe.attnInJKanEv < qreg_kan_probe.attnInJHadEv ∧
    99 / 100 * qreg_kan_probe.attnInJHadEv < qreg_kan_probe.attnInJKanEv := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    norm_num [qreg_kan_probe.attnInJKanCal, qreg_kan_probe.attnInJHadCal, qreg_kan_probe.attnInJKanEv,
      qreg_kan_probe.attnInJHadEv]

/-- **The FFN inputs lose.** gate/up input and down input: learned J above Hadamard J on both Grams. -/
theorem kan_probe_ffn_loses :
    qreg_kan_probe.ffnInJHadCal < qreg_kan_probe.ffnInJKanCal ∧
    qreg_kan_probe.ffnInJHadEv < qreg_kan_probe.ffnInJKanEv ∧
    qreg_kan_probe.ffnDownInJHadCal < qreg_kan_probe.ffnDownInJKanCal ∧
    qreg_kan_probe.ffnDownInJHadEv < qreg_kan_probe.ffnDownInJKanEv := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    norm_num [qreg_kan_probe.ffnInJHadCal, qreg_kan_probe.ffnInJKanCal, qreg_kan_probe.ffnInJHadEv,
      qreg_kan_probe.ffnInJKanEv, qreg_kan_probe.ffnDownInJHadCal, qreg_kan_probe.ffnDownInJKanCal,
      qreg_kan_probe.ffnDownInJHadEv, qreg_kan_probe.ffnDownInJKanEv]

end Perf.QuantumRotation
