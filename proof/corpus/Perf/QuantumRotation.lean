import Perf.Receipts.Onebit

/-!
# Learned register angles against the fixed Hadamard: verdicts (`quantum-register` )

Board: `quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs`. Epic: the quantum-register note.

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

/-- **The trainable corrector's store is the extracted corrector count.** Both arms of the pair carry the same
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

end Perf.QuantumRotation
