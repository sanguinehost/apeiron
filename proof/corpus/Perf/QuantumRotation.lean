import Perf.Receipts.Onebit
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Data.Real.Basic

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

Not claimed by the probe: anything about held-out KL or outcomes. No model was rebuilt with the probe's angles; J is the
per-linear quantization error under the input Gram, not the model loss.

## End-to-end KL (`quantum-register` W2)

`tools/onebit/pv_finish.py` (sha256 `17586d6d…`) `--rotation kan` trains the register's offsets `δ = θ − π/4` on the
end-to-end KL inside the finishing loop, with the arm's codes frozen and r, c, d, norms trained (receipt
`qreg-kan-exact.json`). Each stage is the unscaled reflection `[[a, b], [b, −a]]`, `a = cos δ − sin δ`, `b = cos δ + sin δ`,
with one `128^{-1/2}` scale at the end, so at `δ = 0` every composed weight is `fwht128`'s bit for bit
(`kan_exact_protocol`).

Prediction `kan-kl-exact-1` (held-out KL ≤ the butterfly's 0.8577 within the 580 s run) is refuted: the register's step
costs `1.84×` the butterfly's (`kan_exact_step_cost`), so it trains 377 steps against 729 and finishes above
(`kan_exact_loses_wall`). Per step it wins. The butterfly's end-of-run KL at 146, 292, 529 and 729 steps has increasing
secant slopes (`butterfly_curve_convex`); for a convex curve the secant through 529 and 729, extended back to 377, is a
lower bound at 377, and the register's KL is below it (`kan_exact_wins_matched_step`). That reading assumes convexity
between the measured points (`anchor`); the four measured points satisfy it.

## The register as one matmul (`quantum-register` W2)

`pv_finish.py` (sha256 `4611904f…`) `--rotation kan --hadamard mm` builds the register's `128 × 128` matrix
`T(δ) = kan_matrix(δ)` once per (layer, site) per step and composes each weight with one 128-wide matmul against it
(`RegisterComposeMM`, hand adjoint checked against the butterfly path to `10⁻¹⁵` in float64; receipt
`qreg-kan-mm.json`). `T(0)` is the fixed Hadamard matrix bit for bit, so the step-0 weights are exact
(`kan_mm_protocol`). The step costs `0.673×` the butterfly register's and `1.47×` the fixed Hadamard matmul's
(`kan_mm_step_cost`).

Prediction `kan-kl-mm-1` (at least 700 steps and held-out KL below the fixed Hadamard matmul's `0.8484`) is refuted on
both clauses (`kan_mm_loses_wall`). The speedup buys 173 more steps and `0.020` KL over the butterfly register. Per step
the register still wins: at 550 steps its KL is `0.00125` below the convex lower bound from the butterfly curve
(`kan_mm_wins_matched_step`), under the same convexity `anchor`. The remaining gap is the per-(layer, site) build of
`T`: 96 builds of seven stages each, forward and backward.

## The batched register (`quantum-register` W2)

`pv_finish.py` (sha256 `302ae284…`) builds every (layer, site) register matrix of a step in one batched pass,
`kan_matrix_batch`: seven stage launches per call instead of 96 × 7 (receipt `qreg-kan-mmb.json`; selftest control 9:
the batch is `torch.equal` to `kan_matrix` per key in fp32 and fp64, `dL/dδ` rel `2.7 × 10⁻¹⁶`; control 10: the `a ↔ b`
swap mutant moves it by rel `0.95`). Step-0 weights stay exact (`kan_mmb_protocol`). The step costs `0.81×` the
per-key register's and `1.19×` the fixed Hadamard matmul's (`kan_mmb_step_cost`), and the same as the fixed butterfly
`pvfast`'s to within `1%`. That `1.19×` compares runs on two host states. Three interleaved pairs on one host state
(receipt `qreg-step-pair.json`) put the register at `1.03–1.04×` the fixed Hadamard matmul (`kan_pair_step_parity`,
`kan_pair_each`); the matmul arm itself ran more than `1.10×` slower there than in `w18_hadmm` (`pair_host_slowdown`).

Prediction `kan-kl-mmb-1` (step in `[0.48, 0.56]` s, at least 690 steps, held-out KL below the butterfly's `0.8577`) is
refuted on its speed and step clauses; its KL clause holds. At equal wall the learned register beats the fixed butterfly
(`kan_mmb_beats_butterfly_wall`: `0.8547` against `0.8577`) and still loses to the fixed Hadamard matmul's `0.8484`
(`kan_mmb_loses_hadmm`). Per step the margin grows: at 683 steps its KL is `0.0036` below the convex lower bound from the
butterfly curve (`kan_mmb_wins_matched_step`), against `0.00125` at 550 steps, under the same convexity `anchor`.
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

/-- **Protocol and identity.** Step-0 register weights are `torch.equal` to the `fwht128` weights; the stored bits grow
by exactly the probe's angle count; the fp32 student's gradient under the register is within `10⁻²` relative L2 of the
butterfly's; the learned offsets stay within `0.06` rad of `π/4`. -/
theorem kan_exact_protocol :
    qreg_kan_exact.kanStep0Equal = true ∧
    qreg_kan_exact.bitsAfter = qreg_kan_exact.bitsBefore + qreg_kan_exact.angleBits ∧
    qreg_kan_exact.angleBits = qreg_kan_probe.angleBitsFullModel ∧
    qreg_kan_exact.gradRelFp32Kan < 1 / 100 ∧ 0 < qreg_kan_exact.thetaGradL2 ∧
    qreg_kan_exact.thetaDriftMaxAbs < 6 / 100 := by
  refine ⟨rfl, rfl, rfl, ?_, ?_, ?_⟩ <;>
    norm_num [qreg_kan_exact.gradRelFp32Kan, qreg_kan_exact.thetaGradL2, qreg_kan_exact.thetaDriftMaxAbs]

/-- **Step cost.** Seconds per training step, register over butterfly (both `--compose step --kl fused`): between
`1.8` and `1.9`. -/
theorem kan_exact_step_cost :
    18 / 10 * (w18_pvfast.trainS / w18_pvfast.steps) < qreg_kan_exact.trainS / qreg_kan_exact.steps ∧
    qreg_kan_exact.trainS / qreg_kan_exact.steps < 19 / 10 * (w18_pvfast.trainS / w18_pvfast.steps) := by
  norm_num [w18_pvfast.trainS, w18_pvfast.steps, qreg_kan_exact.trainS, qreg_kan_exact.steps]

/-- **Refuted at equal wall.** In the same run budget the register trains fewer steps and ends above the butterfly's
held-out KL. -/
theorem kan_exact_loses_wall :
    qreg_kan_exact.steps < w18_pvfast.steps ∧
    w18_pvfast.klHeldoutAfter < qreg_kan_exact.klHeldoutAfter := by
  refine ⟨by decide, ?_⟩
  norm_num [w18_pvfast.klHeldoutAfter, qreg_kan_exact.klHeldoutAfter]

/-- Slope of the butterfly's end-of-run held-out KL between two step counts. -/
def secant (n₀ n₁ : ℕ) (k₀ k₁ : ℚ) : ℚ := (k₁ - k₀) / ((n₁ : ℚ) - n₀)

/-- **The butterfly curve is convex at its four measured points.** End-of-run held-out KL against step count
(146, 292, 529, 729): the three secant slopes increase. -/
theorem butterfly_curve_convex :
    secant w18_pvtok146.steps w18_pvtok292.steps w18_pvtok146.klHeldoutAfter w18_pvtok292.klHeldoutAfter <
      secant w18_pvtok292.steps w18_pvaball05v0.steps w18_pvtok292.klHeldoutAfter w18_pvaball05v0.klHeldoutAfter ∧
    secant w18_pvtok292.steps w18_pvaball05v0.steps w18_pvtok292.klHeldoutAfter w18_pvaball05v0.klHeldoutAfter <
      secant w18_pvaball05v0.steps w18_pvfast.steps w18_pvaball05v0.klHeldoutAfter w18_pvfast.klHeldoutAfter := by
  norm_num [secant, w18_pvtok146.steps, w18_pvtok292.steps, w18_pvaball05v0.steps, w18_pvfast.steps,
    w18_pvtok146.klHeldoutAfter, w18_pvtok292.klHeldoutAfter, w18_pvaball05v0.klHeldoutAfter, w18_pvfast.klHeldoutAfter]

/-- **Per-step win.** At the register's 377 steps, its held-out KL is below the 529→729 secant of the butterfly curve
extended back to 377 (`0.9012`; a lower bound for a convex curve) and below the 292→529 chord (`0.9062`; an upper bound). -/
theorem kan_exact_wins_matched_step :
    w18_pvtok292.steps < qreg_kan_exact.steps ∧ qreg_kan_exact.steps < w18_pvaball05v0.steps ∧
    qreg_kan_exact.klHeldoutAfter + 3 / 1000 <
      w18_pvaball05v0.klHeldoutAfter +
        secant w18_pvaball05v0.steps w18_pvfast.steps w18_pvaball05v0.klHeldoutAfter w18_pvfast.klHeldoutAfter *
          ((qreg_kan_exact.steps : ℚ) - w18_pvaball05v0.steps) ∧
    qreg_kan_exact.klHeldoutAfter <
      w18_pvtok292.klHeldoutAfter +
        secant w18_pvtok292.steps w18_pvaball05v0.steps w18_pvtok292.klHeldoutAfter w18_pvaball05v0.klHeldoutAfter *
          ((qreg_kan_exact.steps : ℚ) - w18_pvtok292.steps) := by
  refine ⟨by decide, by decide, ?_, ?_⟩ <;>
    norm_num [secant, w18_pvtok292.steps, w18_pvaball05v0.steps, w18_pvfast.steps, qreg_kan_exact.steps,
      w18_pvtok292.klHeldoutAfter, w18_pvaball05v0.klHeldoutAfter, w18_pvfast.klHeldoutAfter, qreg_kan_exact.klHeldoutAfter]

/-- **Protocol and identity, matmul register.** Step-0 weights `torch.equal` to `PlaneCompose`'s; bits grow by exactly the
angle count; the fp32 gradient is within `10⁻²` relative L2 of the fixed Hadamard's; offsets stay within `0.07` rad of
`π/4`. -/
theorem kan_mm_protocol :
    qreg_kan_mm.hadamard = "mm" ∧ qreg_kan_mm.kanStep0Equal = true ∧
    qreg_kan_mm.bitsAfter = qreg_kan_mm.bitsBefore + qreg_kan_mm.angleBits ∧
    qreg_kan_mm.angleBits = qreg_kan_probe.angleBitsFullModel ∧
    qreg_kan_mm.gradRelFp32Kan < 1 / 100 ∧ qreg_kan_mm.thetaDriftMaxAbs < 7 / 100 := by
  refine ⟨rfl, rfl, rfl, rfl, ?_, ?_⟩ <;>
    norm_num [qreg_kan_mm.gradRelFp32Kan, qreg_kan_mm.thetaDriftMaxAbs]

/-- **Step cost, matmul register.** Seconds per step: below `0.68 ×` the butterfly register's, and between `1.4 ×` and
`1.5 ×` the fixed Hadamard matmul's. -/
theorem kan_mm_step_cost :
    qreg_kan_mm.trainS / qreg_kan_mm.steps < 68 / 100 * (qreg_kan_exact.trainS / qreg_kan_exact.steps) ∧
    14 / 10 * (w18_hadmm.trainS / w18_hadmm.steps) < qreg_kan_mm.trainS / qreg_kan_mm.steps ∧
    qreg_kan_mm.trainS / qreg_kan_mm.steps < 15 / 10 * (w18_hadmm.trainS / w18_hadmm.steps) := by
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [qreg_kan_mm.trainS, qreg_kan_mm.steps, qreg_kan_exact.trainS, qreg_kan_exact.steps, w18_hadmm.trainS,
      w18_hadmm.steps]

/-- **Refuted at equal wall** (`kan-kl-mm-1`). Fewer than 700 steps, and held-out KL above the fixed Hadamard matmul's;
below the butterfly register's. -/
theorem kan_mm_loses_wall :
    qreg_kan_mm.steps < 700 ∧ qreg_kan_exact.steps < qreg_kan_mm.steps ∧
    w18_hadmm.klHeldoutAfter < qreg_kan_mm.klHeldoutAfter ∧
    qreg_kan_mm.klHeldoutAfter + 1 / 100 < qreg_kan_exact.klHeldoutAfter := by
  refine ⟨by decide, by decide, ?_, ?_⟩ <;>
    norm_num [w18_hadmm.klHeldoutAfter, qreg_kan_mm.klHeldoutAfter, qreg_kan_exact.klHeldoutAfter]

/-- **Per-step win, matmul register.** At 550 steps (between 529 and 729) its held-out KL is `10⁻³` below the 292→529
secant of the butterfly curve extended forward (`0.8791`; a lower bound for a convex curve) and below the 529→729 chord
(`0.8798`; an upper bound). -/
theorem kan_mm_wins_matched_step :
    w18_pvaball05v0.steps < qreg_kan_mm.steps ∧ qreg_kan_mm.steps < w18_pvfast.steps ∧
    qreg_kan_mm.klHeldoutAfter + 1 / 1000 <
      w18_pvaball05v0.klHeldoutAfter +
        secant w18_pvtok292.steps w18_pvaball05v0.steps w18_pvtok292.klHeldoutAfter w18_pvaball05v0.klHeldoutAfter *
          ((qreg_kan_mm.steps : ℚ) - w18_pvaball05v0.steps) ∧
    qreg_kan_mm.klHeldoutAfter <
      w18_pvaball05v0.klHeldoutAfter +
        secant w18_pvaball05v0.steps w18_pvfast.steps w18_pvaball05v0.klHeldoutAfter w18_pvfast.klHeldoutAfter *
          ((qreg_kan_mm.steps : ℚ) - w18_pvaball05v0.steps) := by
  refine ⟨by decide, by decide, ?_, ?_⟩ <;>
    norm_num [secant, w18_pvtok292.steps, w18_pvaball05v0.steps, w18_pvfast.steps, qreg_kan_mm.steps,
      w18_pvtok292.klHeldoutAfter, w18_pvaball05v0.klHeldoutAfter, w18_pvfast.klHeldoutAfter, qreg_kan_mm.klHeldoutAfter]

/-- **Protocol and identity, batched register.** Step-0 weights `torch.equal` to `PlaneCompose`'s; bits grow by exactly
the angle count; the fp32 gradient is within `10⁻²` relative L2 of the fixed Hadamard's; offsets stay within `0.07` rad of
`π/4`. -/
theorem kan_mmb_protocol :
    qreg_kan_mmb.hadamard = "mm" ∧ qreg_kan_mmb.kanStep0Equal = true ∧
    qreg_kan_mmb.bitsAfter = qreg_kan_mmb.bitsBefore + qreg_kan_mmb.angleBits ∧
    qreg_kan_mmb.angleBits = qreg_kan_probe.angleBitsFullModel ∧
    qreg_kan_mmb.gradRelFp32Kan < 1 / 100 ∧ qreg_kan_mmb.thetaDriftMaxAbs < 7 / 100 := by
  refine ⟨rfl, rfl, rfl, rfl, ?_, ?_⟩ <;>
    norm_num [qreg_kan_mmb.gradRelFp32Kan, qreg_kan_mmb.thetaDriftMaxAbs]

/-- **Step cost, batched register.** Seconds per step: below `0.81 ×` the per-key matmul register's, between `1.18 ×`
and `1.19 ×` the fixed Hadamard matmul's, and within `1%` of the fixed butterfly's. -/
theorem kan_mmb_step_cost :
    qreg_kan_mmb.trainS / qreg_kan_mmb.steps < 81 / 100 * (qreg_kan_mm.trainS / qreg_kan_mm.steps) ∧
    118 / 100 * (w18_hadmm.trainS / w18_hadmm.steps) < qreg_kan_mmb.trainS / qreg_kan_mmb.steps ∧
    qreg_kan_mmb.trainS / qreg_kan_mmb.steps < 119 / 100 * (w18_hadmm.trainS / w18_hadmm.steps) ∧
    99 / 100 * (w18_pvfast.trainS / w18_pvfast.steps) < qreg_kan_mmb.trainS / qreg_kan_mmb.steps ∧
    qreg_kan_mmb.trainS / qreg_kan_mmb.steps < w18_pvfast.trainS / w18_pvfast.steps := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [qreg_kan_mmb.trainS, qreg_kan_mmb.steps, qreg_kan_mm.trainS, qreg_kan_mm.steps, w18_hadmm.trainS,
      w18_hadmm.steps, w18_pvfast.trainS, w18_pvfast.steps]

/-- **Beats the fixed butterfly at equal wall** (the KL clause of `kan-kl-mmb-1`). Fewer than 690 steps (the step clause
fails) and more than the per-key register's; held-out KL below the butterfly `pvfast`'s and `0.02` below the per-key
register's. -/
theorem kan_mmb_beats_butterfly_wall :
    qreg_kan_mmb.steps < 690 ∧ qreg_kan_mm.steps < qreg_kan_mmb.steps ∧
    qreg_kan_mmb.klHeldoutAfter < w18_pvfast.klHeldoutAfter ∧
    qreg_kan_mmb.klHeldoutAfter + 2 / 100 < qreg_kan_mm.klHeldoutAfter := by
  refine ⟨by decide, by decide, ?_, ?_⟩ <;>
    norm_num [qreg_kan_mmb.klHeldoutAfter, w18_pvfast.klHeldoutAfter, qreg_kan_mm.klHeldoutAfter]

/-- **Loses to the fixed Hadamard matmul at equal wall.** Its held-out KL is above `w18_hadmm`'s by more than `0.006`. -/
theorem kan_mmb_loses_hadmm :
    w18_hadmm.klHeldoutAfter + 6 / 1000 < qreg_kan_mmb.klHeldoutAfter := by
  norm_num [w18_hadmm.klHeldoutAfter, qreg_kan_mmb.klHeldoutAfter]

/-- **Per-step win, batched register.** At 683 steps (between 529 and 729) its held-out KL is `3 × 10⁻³` below the
292→529 secant of the butterfly curve extended forward (`0.8583`; a lower bound for a convex curve) and below the 529→729
chord (`0.8634`; an upper bound). -/
theorem kan_mmb_wins_matched_step :
    w18_pvaball05v0.steps < qreg_kan_mmb.steps ∧ qreg_kan_mmb.steps < w18_pvfast.steps ∧
    qreg_kan_mmb.klHeldoutAfter + 3 / 1000 <
      w18_pvaball05v0.klHeldoutAfter +
        secant w18_pvtok292.steps w18_pvaball05v0.steps w18_pvtok292.klHeldoutAfter w18_pvaball05v0.klHeldoutAfter *
          ((qreg_kan_mmb.steps : ℚ) - w18_pvaball05v0.steps) ∧
    qreg_kan_mmb.klHeldoutAfter <
      w18_pvaball05v0.klHeldoutAfter +
        secant w18_pvaball05v0.steps w18_pvfast.steps w18_pvaball05v0.klHeldoutAfter w18_pvfast.klHeldoutAfter *
          ((qreg_kan_mmb.steps : ℚ) - w18_pvaball05v0.steps) := by
  refine ⟨by decide, by decide, ?_, ?_⟩ <;>
    norm_num [secant, w18_pvtok292.steps, w18_pvaball05v0.steps, w18_pvfast.steps, qreg_kan_mmb.steps,
      w18_pvtok292.klHeldoutAfter, w18_pvaball05v0.klHeldoutAfter, w18_pvfast.klHeldoutAfter, qreg_kan_mmb.klHeldoutAfter]

/-- **Step-time parity on one host state** (`qreg-step-pair-1`, confirmed). Three interleaved pairs (ABBA order), steady
s/step over steps 25→100: the summed register step is between `1.03×` and `1.04×` the summed fixed Hadamard matmul step. -/
theorem kan_pair_step_parity :
    103 / 100 * (qreg_step_pair.hadSteady1 + qreg_step_pair.hadSteady2 + qreg_step_pair.hadSteady3) <
      qreg_step_pair.kanSteady1 + qreg_step_pair.kanSteady2 + qreg_step_pair.kanSteady3 ∧
    qreg_step_pair.kanSteady1 + qreg_step_pair.kanSteady2 + qreg_step_pair.kanSteady3 <
      104 / 100 * (qreg_step_pair.hadSteady1 + qreg_step_pair.hadSteady2 + qreg_step_pair.hadSteady3) := by
  refine ⟨?_, ?_⟩ <;>
    norm_num [qreg_step_pair.kanSteady1, qreg_step_pair.kanSteady2, qreg_step_pair.kanSteady3,
      qreg_step_pair.hadSteady1, qreg_step_pair.hadSteady2, qreg_step_pair.hadSteady3]

/-- **Per pair.** The register is slower in every pair, by less than `1.03×` in pairs 1 and 2 and less than `1.06×` in
pair 3 (spread `0.039`). -/
theorem kan_pair_each :
    qreg_step_pair.hadSteady1 < qreg_step_pair.kanSteady1 ∧
    qreg_step_pair.kanSteady1 < 103 / 100 * qreg_step_pair.hadSteady1 ∧
    qreg_step_pair.hadSteady2 < qreg_step_pair.kanSteady2 ∧
    qreg_step_pair.kanSteady2 < 103 / 100 * qreg_step_pair.hadSteady2 ∧
    qreg_step_pair.hadSteady3 < qreg_step_pair.kanSteady3 ∧
    qreg_step_pair.kanSteady3 < 106 / 100 * qreg_step_pair.hadSteady3 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [qreg_step_pair.kanSteady1, qreg_step_pair.kanSteady2, qreg_step_pair.kanSteady3,
      qreg_step_pair.hadSteady1, qreg_step_pair.hadSteady2, qreg_step_pair.hadSteady3]

/-- **The 1.19× of `kan_mmb_step_cost` was host state.** On the pair's host state every fixed Hadamard matmul arm runs
more than `1.10×` the `w18_hadmm` mean step, and the `qreg_kan_mmb` mean step is within `1.02×` of the pair's mean
register step (`trainS / steps` includes the first 25 steps; the pair's is the steady window). -/
theorem pair_host_slowdown :
    11 / 10 * (w18_hadmm.trainS / w18_hadmm.steps) < qreg_step_pair.hadSteady1 ∧
    11 / 10 * (w18_hadmm.trainS / w18_hadmm.steps) < qreg_step_pair.hadSteady2 ∧
    11 / 10 * (w18_hadmm.trainS / w18_hadmm.steps) < qreg_step_pair.hadSteady3 ∧
    (qreg_step_pair.kanSteady1 + qreg_step_pair.kanSteady2 + qreg_step_pair.kanSteady3) / 3 <
      qreg_kan_mmb.trainS / qreg_kan_mmb.steps ∧
    qreg_kan_mmb.trainS / qreg_kan_mmb.steps <
      102 / 100 * ((qreg_step_pair.kanSteady1 + qreg_step_pair.kanSteady2 + qreg_step_pair.kanSteady3) / 3) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [w18_hadmm.trainS, w18_hadmm.steps, qreg_kan_mmb.trainS, qreg_kan_mmb.steps,
      qreg_step_pair.kanSteady1, qreg_step_pair.kanSteady2, qreg_step_pair.kanSteady3,
      qreg_step_pair.hadSteady1, qreg_step_pair.hadSteady2, qreg_step_pair.hadSteady3]

/-- **The trainable corrector's store is the extracted corrector count.** Both arms of the W3-T5 pair carry the same
48-plane corrector, and its factor store `u8(U) @ u8(V)` equals `t3_nb_3seed.bitsCorr`, the constant
`apeiron_word.BitsRefine.qwen_bits_corr` proves equal to `bits_corr` of the extracted Rust on the Qwen configuration;
the two arms have equal total bits. -/
theorem t5_corr_store_refines :
    qreg_t5_corr.corrStoreBits = t3_nb_3seed.bitsCorr ∧
    qreg_t5_corr.baseBitsCorr = qreg_t5_corr.corrStoreBits ∧
    qreg_t5_corr.corrBitsCorr = qreg_t5_corr.corrStoreBits ∧
    qreg_t5_corr.corrBitsAfter = qreg_t5_corr.baseBitsAfter :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- **Training the corrector loses at equal bits and equal wall.** Against the paired arm with the same corrector
frozen, training `u8(U)`, `u8(V)` at `lr_corr = 1e-4` raises held-out KL by between `0.004` and `0.005` nats and lowers
held-out top-1, while running within `2%` of the frozen arm's steps. -/
theorem t5_corr_loses :
    qreg_t5_corr.baseKlHeldoutAfter + 4 / 1000 < qreg_t5_corr.corrKlHeldoutAfter ∧
    qreg_t5_corr.corrKlHeldoutAfter < qreg_t5_corr.baseKlHeldoutAfter + 5 / 1000 ∧
    qreg_t5_corr.corrTop1HeldoutAfter < qreg_t5_corr.baseTop1HeldoutAfter ∧
    98 / 100 * (qreg_t5_corr.baseSteps : ℚ) < qreg_t5_corr.corrSteps ∧
    qreg_t5_corr.corrSteps < qreg_t5_corr.baseSteps := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [qreg_t5_corr.baseKlHeldoutAfter, qreg_t5_corr.corrKlHeldoutAfter, qreg_t5_corr.baseTop1HeldoutAfter,
      qreg_t5_corr.corrTop1HeldoutAfter, qreg_t5_corr.baseSteps, qreg_t5_corr.corrSteps]

open qreg_attrib in
/-- Held-out KL on 8 chunks after finishing with layer `l`'s seven linears restored to the teacher's, `l = 0 … 23`. -/
def attribAfterLayers : List ℚ :=
  [afterLayerKl0, afterLayerKl1, afterLayerKl2, afterLayerKl3, afterLayerKl4, afterLayerKl5, afterLayerKl6, afterLayerKl7,
   afterLayerKl8, afterLayerKl9, afterLayerKl10, afterLayerKl11, afterLayerKl12, afterLayerKl13, afterLayerKl14,
   afterLayerKl15, afterLayerKl16, afterLayerKl17, afterLayerKl18, afterLayerKl19, afterLayerKl20, afterLayerKl21,
   afterLayerKl22, afterLayerKl23]

open qreg_attrib in
/-- As `attribAfterLayers`, before finishing. -/
def attribBeforeLayers : List ℚ :=
  [beforeLayerKl0, beforeLayerKl1, beforeLayerKl2, beforeLayerKl3, beforeLayerKl4, beforeLayerKl5, beforeLayerKl6,
   beforeLayerKl7, beforeLayerKl8, beforeLayerKl9, beforeLayerKl10, beforeLayerKl11, beforeLayerKl12, beforeLayerKl13,
   beforeLayerKl14, beforeLayerKl15, beforeLayerKl16, beforeLayerKl17, beforeLayerKl18, beforeLayerKl19, beforeLayerKl20,
   beforeLayerKl21, beforeLayerKl22, beforeLayerKl23]

open qreg_attrib in
/-- Held-out KL after finishing with one linear kind restored to the teacher's in all 24 layers. -/
def attribAfterKinds : List ℚ :=
  [afterKindQKl, afterKindKKl, afterKindVKl, afterKindOKl, afterKindGateKl, afterKindUpKl, afterKindDownKl]

/-- Sum over layers of the single-layer restore deltas `base − KL(layer l restored)`. -/
def layerDeltaSum (base : ℚ) (layers : List ℚ) : ℚ := (layers.map (base - ·)).sum

/-- **No layer holds the held-out KL after finishing.** Restoring any one layer's seven linears to the teacher's lowers
the 8-chunk held-out KL of the finished 1.178 b/w arm by less than `7%` of it (the largest is layer 23). -/
theorem attrib_no_layer_holds_kl :
    ∀ x ∈ attribAfterLayers, qreg_attrib.afterBase - x < 7 / 100 * qreg_attrib.afterBase := by
  simp only [attribAfterLayers, List.mem_cons, List.not_mem_nil, or_false]
  rintro x (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl) <;>
    norm_num [qreg_attrib.afterBase, qreg_attrib.afterLayerKl0, qreg_attrib.afterLayerKl1, qreg_attrib.afterLayerKl2,
      qreg_attrib.afterLayerKl3, qreg_attrib.afterLayerKl4, qreg_attrib.afterLayerKl5, qreg_attrib.afterLayerKl6,
      qreg_attrib.afterLayerKl7, qreg_attrib.afterLayerKl8, qreg_attrib.afterLayerKl9, qreg_attrib.afterLayerKl10,
      qreg_attrib.afterLayerKl11, qreg_attrib.afterLayerKl12, qreg_attrib.afterLayerKl13, qreg_attrib.afterLayerKl14,
      qreg_attrib.afterLayerKl15, qreg_attrib.afterLayerKl16, qreg_attrib.afterLayerKl17, qreg_attrib.afterLayerKl18,
      qreg_attrib.afterLayerKl19, qreg_attrib.afterLayerKl20, qreg_attrib.afterLayerKl21, qreg_attrib.afterLayerKl22,
      qreg_attrib.afterLayerKl23]

/-- **The held-out KL is not additive over layers, and finishing makes it less so.** The 24 single-layer restore deltas
sum to between `59%` and `60%` of the base KL before finishing and between `40%` and `41%` after: the rest is the
cross-layer part, which no single-layer restore recovers. -/
theorem attrib_kl_nonadditive :
    59 / 100 * qreg_attrib.beforeBase < layerDeltaSum qreg_attrib.beforeBase attribBeforeLayers ∧
    layerDeltaSum qreg_attrib.beforeBase attribBeforeLayers < 60 / 100 * qreg_attrib.beforeBase ∧
    40 / 100 * qreg_attrib.afterBase < layerDeltaSum qreg_attrib.afterBase attribAfterLayers ∧
    layerDeltaSum qreg_attrib.afterBase attribAfterLayers < 41 / 100 * qreg_attrib.afterBase := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    norm_num [layerDeltaSum, attribBeforeLayers, attribAfterLayers, qreg_attrib.beforeBase, qreg_attrib.afterBase,
      qreg_attrib.beforeLayerKl0, qreg_attrib.beforeLayerKl1, qreg_attrib.beforeLayerKl2, qreg_attrib.beforeLayerKl3,
      qreg_attrib.beforeLayerKl4, qreg_attrib.beforeLayerKl5, qreg_attrib.beforeLayerKl6, qreg_attrib.beforeLayerKl7,
      qreg_attrib.beforeLayerKl8, qreg_attrib.beforeLayerKl9, qreg_attrib.beforeLayerKl10, qreg_attrib.beforeLayerKl11,
      qreg_attrib.beforeLayerKl12, qreg_attrib.beforeLayerKl13, qreg_attrib.beforeLayerKl14, qreg_attrib.beforeLayerKl15,
      qreg_attrib.beforeLayerKl16, qreg_attrib.beforeLayerKl17, qreg_attrib.beforeLayerKl18, qreg_attrib.beforeLayerKl19,
      qreg_attrib.beforeLayerKl20, qreg_attrib.beforeLayerKl21, qreg_attrib.beforeLayerKl22, qreg_attrib.beforeLayerKl23,
      qreg_attrib.afterLayerKl0, qreg_attrib.afterLayerKl1, qreg_attrib.afterLayerKl2, qreg_attrib.afterLayerKl3,
      qreg_attrib.afterLayerKl4, qreg_attrib.afterLayerKl5, qreg_attrib.afterLayerKl6, qreg_attrib.afterLayerKl7,
      qreg_attrib.afterLayerKl8, qreg_attrib.afterLayerKl9, qreg_attrib.afterLayerKl10, qreg_attrib.afterLayerKl11,
      qreg_attrib.afterLayerKl12, qreg_attrib.afterLayerKl13, qreg_attrib.afterLayerKl14, qreg_attrib.afterLayerKl15,
      qreg_attrib.afterLayerKl16, qreg_attrib.afterLayerKl17, qreg_attrib.afterLayerKl18, qreg_attrib.afterLayerKl19,
      qreg_attrib.afterLayerKl20, qreg_attrib.afterLayerKl21, qreg_attrib.afterLayerKl22, qreg_attrib.afterLayerKl23]

/-- **The finished kinds are co-adapted.** After finishing, restoring any one linear kind to the teacher's in all 24
layers raises the held-out KL above the student's own; restoring `down` alone costs more than `0.18` nats. Restoring
every linear brings it below `0.011`. -/
theorem attrib_kinds_coadapted :
    (∀ x ∈ attribAfterKinds, qreg_attrib.afterBase < x) ∧
    qreg_attrib.afterBase + 18 / 100 < qreg_attrib.afterKindDownKl ∧
    qreg_attrib.afterAllLinear < 11 / 1000 := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [attribAfterKinds, List.mem_cons, List.not_mem_nil, or_false]
    rintro x (rfl | rfl | rfl | rfl | rfl | rfl | rfl) <;>
      norm_num [qreg_attrib.afterBase, qreg_attrib.afterKindQKl, qreg_attrib.afterKindKKl, qreg_attrib.afterKindVKl,
        qreg_attrib.afterKindOKl, qreg_attrib.afterKindGateKl, qreg_attrib.afterKindUpKl, qreg_attrib.afterKindDownKl]
  · norm_num [qreg_attrib.afterBase, qreg_attrib.afterKindDownKl]
  · norm_num [qreg_attrib.afterAllLinear]

open qreg_iso in
/-- Held-out KL before finishing with layer `l`'s student linears put back over the all-linear (teacher) configuration. -/
def isoBefore : List ℚ :=
  [beforeIsoKl0, beforeIsoKl1, beforeIsoKl2, beforeIsoKl3, beforeIsoKl4, beforeIsoKl5, beforeIsoKl6, beforeIsoKl7,
   beforeIsoKl8, beforeIsoKl9, beforeIsoKl10, beforeIsoKl11, beforeIsoKl12, beforeIsoKl13, beforeIsoKl14,
   beforeIsoKl15, beforeIsoKl16, beforeIsoKl17, beforeIsoKl18, beforeIsoKl19, beforeIsoKl20, beforeIsoKl21,
   beforeIsoKl22, beforeIsoKl23]

open qreg_iso in
/-- Held-out KL after finishing with layer `l`'s student linears put back over the all-linear (teacher) configuration. -/
def isoAfter : List ℚ :=
  [afterIsoKl0, afterIsoKl1, afterIsoKl2, afterIsoKl3, afterIsoKl4, afterIsoKl5, afterIsoKl6, afterIsoKl7,
   afterIsoKl8, afterIsoKl9, afterIsoKl10, afterIsoKl11, afterIsoKl12, afterIsoKl13, afterIsoKl14, afterIsoKl15,
   afterIsoKl16, afterIsoKl17, afterIsoKl18, afterIsoKl19, afterIsoKl20, afterIsoKl21, afterIsoKl22, afterIsoKl23]

/-- Sum over put-back groups of their cost `KL − lin` over the all-linear KL `lin`. O(n) additions in the list length. -/
def isoCostSum (lin : ℚ) (xs : List ℚ) : ℚ := (xs.map (· - lin)).sum

/-- Cost of one put-back group `g − lin` as a fraction of the summed isolated costs of its layers `xs`. -/
def groupRatio (lin g : ℚ) (xs : List ℚ) : ℚ := (g - lin) / isoCostSum lin xs

/-- **The isolated layer costs over-count the held-out KL, and finishing makes the over-count larger.** Putting back one
layer's student linears over the all-linear configuration costs, summed over the 24 layers, between `1.44×` and
`1.45×` the full gap `base − all_linear` before finishing and between `1.65×` and `1.66×` after. -/
theorem iso_kl_superadditive :
    144 / 100 * (qreg_iso.beforeBase - qreg_iso.beforeAllLinear) < isoCostSum qreg_iso.beforeAllLinear isoBefore ∧
    isoCostSum qreg_iso.beforeAllLinear isoBefore < 145 / 100 * (qreg_iso.beforeBase - qreg_iso.beforeAllLinear) ∧
    165 / 100 * (qreg_iso.afterBase - qreg_iso.afterAllLinear) < isoCostSum qreg_iso.afterAllLinear isoAfter ∧
    isoCostSum qreg_iso.afterAllLinear isoAfter < 166 / 100 * (qreg_iso.afterBase - qreg_iso.afterAllLinear) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    norm_num [isoCostSum, isoBefore, isoAfter,
      qreg_iso.beforeIsoKl0, qreg_iso.beforeIsoKl1, qreg_iso.beforeIsoKl2, qreg_iso.beforeIsoKl3,
      qreg_iso.beforeIsoKl4, qreg_iso.beforeIsoKl5, qreg_iso.beforeIsoKl6, qreg_iso.beforeIsoKl7,
      qreg_iso.beforeIsoKl8, qreg_iso.beforeIsoKl9, qreg_iso.beforeIsoKl10, qreg_iso.beforeIsoKl11,
      qreg_iso.beforeIsoKl12, qreg_iso.beforeIsoKl13, qreg_iso.beforeIsoKl14, qreg_iso.beforeIsoKl15,
      qreg_iso.beforeIsoKl16, qreg_iso.beforeIsoKl17, qreg_iso.beforeIsoKl18, qreg_iso.beforeIsoKl19,
      qreg_iso.beforeIsoKl20, qreg_iso.beforeIsoKl21, qreg_iso.beforeIsoKl22, qreg_iso.beforeIsoKl23,
      qreg_iso.beforeBase, qreg_iso.beforeAllLinear, qreg_iso.beforePrefixKl6, qreg_iso.beforePrefixKl12,
      qreg_iso.beforePrefixKl18, qreg_iso.beforeSuffixKl6, qreg_iso.beforeSuffixKl12, qreg_iso.beforeSuffixKl18,
      qreg_iso.afterIsoKl0, qreg_iso.afterIsoKl1, qreg_iso.afterIsoKl2, qreg_iso.afterIsoKl3, qreg_iso.afterIsoKl4,
      qreg_iso.afterIsoKl5, qreg_iso.afterIsoKl6, qreg_iso.afterIsoKl7, qreg_iso.afterIsoKl8, qreg_iso.afterIsoKl9,
      qreg_iso.afterIsoKl10, qreg_iso.afterIsoKl11, qreg_iso.afterIsoKl12, qreg_iso.afterIsoKl13,
      qreg_iso.afterIsoKl14, qreg_iso.afterIsoKl15, qreg_iso.afterIsoKl16, qreg_iso.afterIsoKl17,
      qreg_iso.afterIsoKl18, qreg_iso.afterIsoKl19, qreg_iso.afterIsoKl20, qreg_iso.afterIsoKl21,
      qreg_iso.afterIsoKl22, qreg_iso.afterIsoKl23, qreg_iso.afterBase, qreg_iso.afterAllLinear,
      qreg_iso.afterPrefixKl6, qreg_iso.afterPrefixKl12, qreg_iso.afterPrefixKl18, qreg_iso.afterSuffixKl6,
      qreg_iso.afterSuffixKl12, qreg_iso.afterSuffixKl18]

/-- **Layer errors saturate through depth.** Every prefix (layers `0..k−1`) and suffix (layers `k..23`) put back
together, `k ∈ {6, 12, 18}`, costs less than the sum of its layers' isolated costs: below `0.69` of it for prefixes and
`0.92` for suffixes before finishing, below `0.62` and `0.79` after. Early-layer errors saturate more. -/
theorem iso_groups_saturate :
    groupRatio qreg_iso.beforeAllLinear qreg_iso.beforePrefixKl6 (isoBefore.take 6) < 69 / 100 ∧
    groupRatio qreg_iso.beforeAllLinear qreg_iso.beforePrefixKl12 (isoBefore.take 12) < 69 / 100 ∧
    groupRatio qreg_iso.beforeAllLinear qreg_iso.beforePrefixKl18 (isoBefore.take 18) < 69 / 100 ∧
    groupRatio qreg_iso.beforeAllLinear qreg_iso.beforeSuffixKl6 (isoBefore.drop 6) < 92 / 100 ∧
    groupRatio qreg_iso.beforeAllLinear qreg_iso.beforeSuffixKl12 (isoBefore.drop 12) < 92 / 100 ∧
    groupRatio qreg_iso.beforeAllLinear qreg_iso.beforeSuffixKl18 (isoBefore.drop 18) < 92 / 100 ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterPrefixKl6 (isoAfter.take 6) < 62 / 100 ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterPrefixKl12 (isoAfter.take 12) < 62 / 100 ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterPrefixKl18 (isoAfter.take 18) < 62 / 100 ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterSuffixKl6 (isoAfter.drop 6) < 79 / 100 ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterSuffixKl12 (isoAfter.drop 12) < 79 / 100 ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterSuffixKl18 (isoAfter.drop 18) < 79 / 100 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [groupRatio, isoCostSum, isoBefore, isoAfter, qreg_iso.beforeIsoKl0, qreg_iso.beforeIsoKl1,
      qreg_iso.beforeIsoKl2, qreg_iso.beforeIsoKl3, qreg_iso.beforeIsoKl4, qreg_iso.beforeIsoKl5,
      qreg_iso.beforeIsoKl6, qreg_iso.beforeIsoKl7, qreg_iso.beforeIsoKl8, qreg_iso.beforeIsoKl9,
      qreg_iso.beforeIsoKl10, qreg_iso.beforeIsoKl11, qreg_iso.beforeIsoKl12, qreg_iso.beforeIsoKl13,
      qreg_iso.beforeIsoKl14, qreg_iso.beforeIsoKl15, qreg_iso.beforeIsoKl16, qreg_iso.beforeIsoKl17,
      qreg_iso.beforeIsoKl18, qreg_iso.beforeIsoKl19, qreg_iso.beforeIsoKl20, qreg_iso.beforeIsoKl21,
      qreg_iso.beforeIsoKl22, qreg_iso.beforeIsoKl23, qreg_iso.beforeBase, qreg_iso.beforeAllLinear,
      qreg_iso.beforePrefixKl6, qreg_iso.beforePrefixKl12, qreg_iso.beforePrefixKl18, qreg_iso.beforeSuffixKl6,
      qreg_iso.beforeSuffixKl12, qreg_iso.beforeSuffixKl18, qreg_iso.afterIsoKl0, qreg_iso.afterIsoKl1,
      qreg_iso.afterIsoKl2, qreg_iso.afterIsoKl3, qreg_iso.afterIsoKl4, qreg_iso.afterIsoKl5, qreg_iso.afterIsoKl6,
      qreg_iso.afterIsoKl7, qreg_iso.afterIsoKl8, qreg_iso.afterIsoKl9, qreg_iso.afterIsoKl10, qreg_iso.afterIsoKl11,
      qreg_iso.afterIsoKl12, qreg_iso.afterIsoKl13, qreg_iso.afterIsoKl14, qreg_iso.afterIsoKl15,
      qreg_iso.afterIsoKl16, qreg_iso.afterIsoKl17, qreg_iso.afterIsoKl18, qreg_iso.afterIsoKl19,
      qreg_iso.afterIsoKl20, qreg_iso.afterIsoKl21, qreg_iso.afterIsoKl22, qreg_iso.afterIsoKl23, qreg_iso.afterBase,
      qreg_iso.afterAllLinear, qreg_iso.afterPrefixKl6, qreg_iso.afterPrefixKl12, qreg_iso.afterPrefixKl18,
      qreg_iso.afterSuffixKl6, qreg_iso.afterSuffixKl12, qreg_iso.afterSuffixKl18]

/-- **Finishing makes every group saturate more.** For each prefix and suffix group, `k ∈ {6, 12, 18}`, the ratio of the
group's put-back cost to the sum of its layers' isolated costs is smaller after finishing than before. -/
theorem iso_finishing_saturates :
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterPrefixKl6 (isoAfter.take 6) <
      groupRatio qreg_iso.beforeAllLinear qreg_iso.beforePrefixKl6 (isoBefore.take 6) ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterPrefixKl12 (isoAfter.take 12) <
      groupRatio qreg_iso.beforeAllLinear qreg_iso.beforePrefixKl12 (isoBefore.take 12) ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterPrefixKl18 (isoAfter.take 18) <
      groupRatio qreg_iso.beforeAllLinear qreg_iso.beforePrefixKl18 (isoBefore.take 18) ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterSuffixKl6 (isoAfter.drop 6) <
      groupRatio qreg_iso.beforeAllLinear qreg_iso.beforeSuffixKl6 (isoBefore.drop 6) ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterSuffixKl12 (isoAfter.drop 12) <
      groupRatio qreg_iso.beforeAllLinear qreg_iso.beforeSuffixKl12 (isoBefore.drop 12) ∧
    groupRatio qreg_iso.afterAllLinear qreg_iso.afterSuffixKl18 (isoAfter.drop 18) <
      groupRatio qreg_iso.beforeAllLinear qreg_iso.beforeSuffixKl18 (isoBefore.drop 18) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [groupRatio, isoCostSum, isoBefore, isoAfter, qreg_iso.beforeIsoKl0, qreg_iso.beforeIsoKl1,
      qreg_iso.beforeIsoKl2, qreg_iso.beforeIsoKl3, qreg_iso.beforeIsoKl4, qreg_iso.beforeIsoKl5,
      qreg_iso.beforeIsoKl6, qreg_iso.beforeIsoKl7, qreg_iso.beforeIsoKl8, qreg_iso.beforeIsoKl9,
      qreg_iso.beforeIsoKl10, qreg_iso.beforeIsoKl11, qreg_iso.beforeIsoKl12, qreg_iso.beforeIsoKl13,
      qreg_iso.beforeIsoKl14, qreg_iso.beforeIsoKl15, qreg_iso.beforeIsoKl16, qreg_iso.beforeIsoKl17,
      qreg_iso.beforeIsoKl18, qreg_iso.beforeIsoKl19, qreg_iso.beforeIsoKl20, qreg_iso.beforeIsoKl21,
      qreg_iso.beforeIsoKl22, qreg_iso.beforeIsoKl23, qreg_iso.beforeBase, qreg_iso.beforeAllLinear,
      qreg_iso.beforePrefixKl6, qreg_iso.beforePrefixKl12, qreg_iso.beforePrefixKl18, qreg_iso.beforeSuffixKl6,
      qreg_iso.beforeSuffixKl12, qreg_iso.beforeSuffixKl18, qreg_iso.afterIsoKl0, qreg_iso.afterIsoKl1,
      qreg_iso.afterIsoKl2, qreg_iso.afterIsoKl3, qreg_iso.afterIsoKl4, qreg_iso.afterIsoKl5, qreg_iso.afterIsoKl6,
      qreg_iso.afterIsoKl7, qreg_iso.afterIsoKl8, qreg_iso.afterIsoKl9, qreg_iso.afterIsoKl10, qreg_iso.afterIsoKl11,
      qreg_iso.afterIsoKl12, qreg_iso.afterIsoKl13, qreg_iso.afterIsoKl14, qreg_iso.afterIsoKl15,
      qreg_iso.afterIsoKl16, qreg_iso.afterIsoKl17, qreg_iso.afterIsoKl18, qreg_iso.afterIsoKl19,
      qreg_iso.afterIsoKl20, qreg_iso.afterIsoKl21, qreg_iso.afterIsoKl22, qreg_iso.afterIsoKl23, qreg_iso.afterBase,
      qreg_iso.afterAllLinear, qreg_iso.afterPrefixKl6, qreg_iso.afterPrefixKl12, qreg_iso.afterPrefixKl18,
      qreg_iso.afterSuffixKl6, qreg_iso.afterSuffixKl12, qreg_iso.afterSuffixKl18]

/-- **No single layer is cheap to quantize after finishing.** Every layer's student linears, put back alone over the
all-linear configuration, cost more than `0.02` nats of held-out KL (the smallest is layer 11, `0.0229`). -/
theorem iso_every_layer_costs :
    ∀ x ∈ isoAfter, qreg_iso.afterAllLinear + 2 / 100 < x := by
  simp only [isoAfter, List.mem_cons, List.not_mem_nil, or_false]
  rintro x (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl) <;>
    norm_num [qreg_iso.afterAllLinear, qreg_iso.afterIsoKl0, qreg_iso.afterIsoKl1, qreg_iso.afterIsoKl2, qreg_iso.afterIsoKl3,
      qreg_iso.afterIsoKl4, qreg_iso.afterIsoKl5, qreg_iso.afterIsoKl6, qreg_iso.afterIsoKl7, qreg_iso.afterIsoKl8,
      qreg_iso.afterIsoKl9, qreg_iso.afterIsoKl10, qreg_iso.afterIsoKl11, qreg_iso.afterIsoKl12,
      qreg_iso.afterIsoKl13, qreg_iso.afterIsoKl14, qreg_iso.afterIsoKl15, qreg_iso.afterIsoKl16,
      qreg_iso.afterIsoKl17, qreg_iso.afterIsoKl18, qreg_iso.afterIsoKl19, qreg_iso.afterIsoKl20,
      qreg_iso.afterIsoKl21, qreg_iso.afterIsoKl22, qreg_iso.afterIsoKl23]

/-! ## Block-wise fine-tuning (`docs/evidence/receipts/onebit/qreg-blockft.json`)

`pv_finish.py --block-s 120` spends the first 120 s of the train budget on AQLM's phase 3 (Egiazarian et al., arXiv
2401.06118, Sec. 3.4): each block fits its relative squared residual update on the teacher's hidden-state input, then the
Adam state is cleared and the end-to-end KL phase runs to the same wall. One pair, same transfer, same source. -/

/-- **Prediction `qreg-blockft-1` is refuted.** The block-phase arm ends `0.023` nats of held-out KL above the KL-only arm
at the same wall (`0.8817` against `0.8588`), with lower held-out top-1, although it fits more steps in the budget. -/
theorem blockft_refuted :
    qreg_blockft.baseKlHeldoutAfter + 2 / 100 < qreg_blockft.b120KlHeldoutAfter ∧
    qreg_blockft.b120Top1HeldoutAfter < qreg_blockft.baseTop1HeldoutAfter ∧
    qreg_blockft.baseSteps < qreg_blockft.b120Steps ∧
    qreg_blockft.b120QuickKlStep0 = qreg_blockft.baseQuickKlStep0 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    norm_num [qreg_blockft.baseKlHeldoutAfter, qreg_blockft.b120KlHeldoutAfter, qreg_blockft.b120Top1HeldoutAfter,
      qreg_blockft.baseTop1HeldoutAfter, qreg_blockft.baseSteps, qreg_blockft.b120Steps, qreg_blockft.b120QuickKlStep0,
      qreg_blockft.baseQuickKlStep0]

/-- **The block-local objective is misaligned with the end-to-end KL.** Over the 296 block steps the block loss falls
(`0.3120 → 0.2803` on the same batch) while the quick held-out KL rises by more than `0.1` (`0.7954 → 0.9635`). -/
theorem blockft_phase_raises_kl :
    qreg_blockft.blockLossLast < qreg_blockft.blockLossFirst ∧
    qreg_blockft.b120QuickKlStep0 + 1 / 10 < qreg_blockft.blockQuickKlAtSwitch := by
  refine ⟨?_, ?_⟩ <;>
    norm_num [qreg_blockft.blockLossLast, qreg_blockft.blockLossFirst, qreg_blockft.b120QuickKlStep0,
      qreg_blockft.blockQuickKlAtSwitch]

/-- **Cost model: a block step costs `3/4` to `4/5` of a KL step** (`0.406` s against `0.538` s, ratio `0.754`): it skips the
LM head, so the phase buys steps, not KL. -/
theorem blockft_block_step_cost :
    qreg_blockft.blockWallS / qreg_blockft.blockSteps <
      4 / 5 * (qreg_blockft.baseTrainS / qreg_blockft.baseSteps) ∧
    3 / 4 * (qreg_blockft.baseTrainS / qreg_blockft.baseSteps) <
      qreg_blockft.blockWallS / qreg_blockft.blockSteps := by
  refine ⟨?_, ?_⟩ <;>
    norm_num [qreg_blockft.blockWallS, qreg_blockft.blockSteps, qreg_blockft.baseTrainS, qreg_blockft.baseSteps]

/-! ## Output-side code refinement (`docs/evidence/receipts/onebit/qreg-yaqa.json`)

`fold_transfer.py` arm suffix `-yaqa` re-rounds every E8 plane of the 1.178 b/w arm against
`J_O(D) = tr(Dᵀ H_O D H_l)` (Tseng et al., YAQA, arXiv 2505.22988, eq. 5–6; `H_O` by Sketch A, eq. 11, from
`50 × 512` tokens), a 256-row block at a time, keeping a block only when `J_O` falls (`YaqaBlock` below). Two block-Jacobi
arms ran on a capture whose `H_O` carried atomics noise; two block-solve arms ran on the fixed, bit-reproducible capture. -/

/-- **Predictions `qreg-yaqa-1`, `-3` and `-6` are refuted.** The block solve lowers `J_O` in all seven linear kinds and
raises held-out KL before finishing by more than `0.18` nats (`1.1637 → 1.3471`) at unchanged bits; the repeat on the same
source is bit-identical. -/
theorem yaqa_refuted :
    qreg_yaqa.baseKlHeldoutBefore + 18 / 100 < qreg_yaqa.solveAKlHeldoutBefore ∧
    qreg_yaqa.solveAKlHeldoutBefore = qreg_yaqa.solveBKlHeldoutBefore ∧
    qreg_yaqa.baseKlHeldoutBefore = qreg_yaqa.base2KlHeldoutBefore := by
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [qreg_yaqa.baseKlHeldoutBefore, qreg_yaqa.solveAKlHeldoutBefore, qreg_yaqa.solveBKlHeldoutBefore,
      qreg_yaqa.base2KlHeldoutBefore]

/-- **The proxy falls in every kind.** `J_O` after the accepted blocks is below `J_O` before for `q, k, v, o, gate, up,
down`; `gate` and `up` accept every block. The objective the refine optimises is not the held-out KL it moves. -/
theorem yaqa_jo_falls :
    qreg_yaqa.solveQJoAfter < qreg_yaqa.solveQJoBefore ∧ qreg_yaqa.solveKJoAfter < qreg_yaqa.solveKJoBefore ∧
    qreg_yaqa.solveVJoAfter < qreg_yaqa.solveVJoBefore ∧ qreg_yaqa.solveOJoAfter < qreg_yaqa.solveOJoBefore ∧
    qreg_yaqa.solveGateJoAfter < qreg_yaqa.solveGateJoBefore ∧ qreg_yaqa.solveUpJoAfter < qreg_yaqa.solveUpJoBefore ∧
    qreg_yaqa.solveDownJoAfter < qreg_yaqa.solveDownJoBefore ∧
    qreg_yaqa.solveGateAccepted = qreg_yaqa.solveGateChunks ∧ qreg_yaqa.solveUpAccepted = qreg_yaqa.solveUpChunks := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [qreg_yaqa.solveQJoAfter, qreg_yaqa.solveQJoBefore, qreg_yaqa.solveKJoAfter, qreg_yaqa.solveKJoBefore,
      qreg_yaqa.solveVJoAfter, qreg_yaqa.solveVJoBefore, qreg_yaqa.solveOJoAfter, qreg_yaqa.solveOJoBefore,
      qreg_yaqa.solveGateJoAfter, qreg_yaqa.solveGateJoBefore, qreg_yaqa.solveUpJoAfter, qreg_yaqa.solveUpJoBefore,
      qreg_yaqa.solveDownJoAfter, qreg_yaqa.solveDownJoBefore, qreg_yaqa.solveGateAccepted, qreg_yaqa.solveGateChunks,
      qreg_yaqa.solveUpAccepted, qreg_yaqa.solveUpChunks]

/-- **Held-out KL is hypersensitive to `H_O` estimation noise.** The two block-Jacobi arms ran one source; their captures
differ only by atomics in the attention backward. Their per-kind `J_O` after the refine agree within `1.2%` relative in
every kind, while held-out KL before finishing differs by more than `0.15` nats. -/
theorem yaqa_capture_noise :
    qreg_yaqa.jacobi1KlHeldoutBefore + 15 / 100 < qreg_yaqa.jacobi2KlHeldoutBefore ∧
    |qreg_yaqa.jacobi1QJoAfter - qreg_yaqa.jacobi2QJoAfter| < 12 / 1000 * qreg_yaqa.jacobi1QJoAfter ∧
    |qreg_yaqa.jacobi1KJoAfter - qreg_yaqa.jacobi2KJoAfter| < 12 / 1000 * qreg_yaqa.jacobi1KJoAfter ∧
    |qreg_yaqa.jacobi1VJoAfter - qreg_yaqa.jacobi2VJoAfter| < 12 / 1000 * qreg_yaqa.jacobi1VJoAfter ∧
    |qreg_yaqa.jacobi1OJoAfter - qreg_yaqa.jacobi2OJoAfter| < 12 / 1000 * qreg_yaqa.jacobi1OJoAfter ∧
    |qreg_yaqa.jacobi1GateJoAfter - qreg_yaqa.jacobi2GateJoAfter| < 12 / 1000 * qreg_yaqa.jacobi1GateJoAfter ∧
    |qreg_yaqa.jacobi1UpJoAfter - qreg_yaqa.jacobi2UpJoAfter| < 12 / 1000 * qreg_yaqa.jacobi1UpJoAfter ∧
    |qreg_yaqa.jacobi1DownJoAfter - qreg_yaqa.jacobi2DownJoAfter| < 12 / 1000 * qreg_yaqa.jacobi1DownJoAfter := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [abs_lt, qreg_yaqa.jacobi1KlHeldoutBefore, qreg_yaqa.jacobi2KlHeldoutBefore, qreg_yaqa.jacobi1QJoAfter,
      qreg_yaqa.jacobi2QJoAfter, qreg_yaqa.jacobi1KJoAfter, qreg_yaqa.jacobi2KJoAfter, qreg_yaqa.jacobi1VJoAfter,
      qreg_yaqa.jacobi2VJoAfter, qreg_yaqa.jacobi1OJoAfter, qreg_yaqa.jacobi2OJoAfter, qreg_yaqa.jacobi1GateJoAfter,
      qreg_yaqa.jacobi2GateJoAfter, qreg_yaqa.jacobi1UpJoAfter, qreg_yaqa.jacobi2UpJoAfter,
      qreg_yaqa.jacobi1DownJoAfter, qreg_yaqa.jacobi2DownJoAfter]

end Perf.QuantumRotation

/-!
## YAQA block refinement: the trace identities behind `yaqa_refine`

`tools/onebit/fold_transfer.py` `yaqa_refine` (Tseng et al., YAQA, arXiv 2505.22988, eq. 5–6) re-rounds the
E8 codes of one linear against `J(D) = tr(Dᵀ H_O D H_l)`, `D = Ws − W`, `H_O` the output-side Hessian sketch
(eq. 11, Sketch A) and `H_l` the input metric, a block of rows `C` at a time. With the other rows fixed, the
block objective in `X = D_C` is `tr(Xᵀ A X L) + 2 tr(Xᵀ B L) + const` with `A = H_O[C,C]`,
`B = H_O[C,¬C] D_¬C = H_O[C,:] D − A D_C` and `L = H_l`. The code solves `A E = B`
(`torch.linalg.solve`) and rounds toward `W_C = Ws_C + E`, i.e. `X = −E` (`block_target_attains`), and accepts
a chunk only when the exact increment `dJ_exact` is negative. The earlier per-row target
`(H_O[r,:] D − H_O[r,r] D_r)/H_O[r,r]` is this solve with `A` replaced by its diagonal.

Not covered: the discrete ICM rounding over the code alphabet, the accuracy of the `H_O` sketch, the block
split of `J` itself, and the held-out KL, which are harness measurements.
-/


namespace Perf.QuantumRotation.YaqaBlock

open Matrix

variable {o n c : ℕ}

/-- **Exact increment of the block objective.** For symmetric `H` and `L`, the change of
`J(D) = tr(Dᵀ H D L)` under `D ↦ D + Δ` is the first-order term `2 tr(Δᵀ H D L)` plus the exact
remainder `tr(Δᵀ H Δ L)`; nothing is linearised away. The cross term uses the symmetry
`tr(Xᵀ H Y L) = tr(Yᵀ H X L)` of the bilinear form. -/
theorem dJ_exact (H : Matrix (Fin o) (Fin o) ℝ) (L : Matrix (Fin n) (Fin n) ℝ)
    (D Δ : Matrix (Fin o) (Fin n) ℝ) (hH : Hᵀ = H) (hL : Lᵀ = L) :
    Matrix.trace ((D + Δ)ᵀ * H * (D + Δ) * L) - Matrix.trace (Dᵀ * H * D * L)
      = 2 * Matrix.trace (Δᵀ * (H * D * L)) + Matrix.trace (Δᵀ * H * Δ * L) := by
  have hcross : ∀ X Y : Matrix (Fin o) (Fin n) ℝ,
      Matrix.trace (Xᵀ * H * Y * L) = Matrix.trace (Yᵀ * H * X * L) := by
    intro X Y
    rw [Matrix.trace_mul_comm (Xᵀ * H * Y) L]
    rw [← Matrix.trace_transpose (L * (Xᵀ * H * Y))]
    congr 1
    simp only [Matrix.transpose_mul, Matrix.transpose_transpose, hH, hL, Matrix.mul_assoc]
  have hexpand : Matrix.trace ((D + Δ)ᵀ * H * (D + Δ) * L)
      = Matrix.trace (Dᵀ * H * D * L) + Matrix.trace (Dᵀ * H * Δ * L)
        + Matrix.trace (Δᵀ * H * D * L) + Matrix.trace (Δᵀ * H * Δ * L) := by
    simp only [Matrix.transpose_add, Matrix.add_mul, Matrix.mul_add, Matrix.trace_add,
      Matrix.mul_assoc]
    ring
  rw [hexpand, hcross D Δ]
  simp only [Matrix.mul_assoc]
  ring

/-- **Block complete square.** With `A` and `L` symmetric and the block target `B = A E`, the block
objective `tr(Xᵀ A X L) + 2 tr(Xᵀ B L)` equals the complete square
`tr((X + E)ᵀ A (X + E) L) - tr(Eᵀ A E L)`: the last term is the irreducible cost of the block
target. -/
theorem block_complete_square (A : Matrix (Fin c) (Fin c) ℝ) (L : Matrix (Fin n) (Fin n) ℝ)
    (B E X : Matrix (Fin c) (Fin n) ℝ) (hA : Aᵀ = A) (hL : Lᵀ = L) (hE : A * E = B) :
    Matrix.trace (Xᵀ * A * X * L) + 2 * Matrix.trace (Xᵀ * B * L)
      = Matrix.trace ((X + E)ᵀ * A * (X + E) * L) - Matrix.trace (Eᵀ * A * E * L) := by
  have hcross : ∀ P Q : Matrix (Fin c) (Fin n) ℝ,
      Matrix.trace (Pᵀ * A * Q * L) = Matrix.trace (Qᵀ * A * P * L) := by
    intro P Q
    rw [Matrix.trace_mul_comm (Pᵀ * A * Q) L]
    rw [← Matrix.trace_transpose (L * (Pᵀ * A * Q))]
    congr 1
    simp only [Matrix.transpose_mul, Matrix.transpose_transpose, hA, hL, Matrix.mul_assoc]
  have hexpand : Matrix.trace ((X + E)ᵀ * A * (X + E) * L)
      = Matrix.trace (Xᵀ * A * X * L) + Matrix.trace (Xᵀ * A * E * L)
        + Matrix.trace (Eᵀ * A * X * L) + Matrix.trace (Eᵀ * A * E * L) := by
    simp only [Matrix.transpose_add, Matrix.add_mul, Matrix.mul_add, Matrix.trace_add,
      Matrix.mul_assoc]
    ring
  have hB : Matrix.trace (Xᵀ * A * E * L) = Matrix.trace (Xᵀ * B * L) := by
    rw [Matrix.mul_assoc Xᵀ A E, hE]
  rw [hexpand, hcross E X, hB]
  simp only [Matrix.mul_assoc]
  ring

/-- **The block target is attained.** Under the hypotheses of `block_complete_square`, the code
`X = -E` zeroes the square: the block objective there is exactly `- tr(Eᵀ A E L)`. Minimality over
all `X` needs `A` and `L` positive semidefinite, which is not proved here. A corollary of
`block_complete_square`. -/
theorem block_target_attains (A : Matrix (Fin c) (Fin c) ℝ) (L : Matrix (Fin n) (Fin n) ℝ)
    (B E : Matrix (Fin c) (Fin n) ℝ) (hA : Aᵀ = A) (hL : Lᵀ = L) (hE : A * E = B) :
    Matrix.trace ((-E)ᵀ * A * (-E) * L) + 2 * Matrix.trace ((-E)ᵀ * B * L)
      = - Matrix.trace (Eᵀ * A * E * L) := by
  have h := block_complete_square A L B E (-E) hA hL hE
  simpa using h

end Perf.QuantumRotation.YaqaBlock
