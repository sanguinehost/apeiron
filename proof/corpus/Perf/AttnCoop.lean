import Perf.Chain

/-!
# The ATTN-COOP node: the shared-load term, the staging halving and the node gain

`Perf.Roofline` and `Perf.Chain` give the resource carrier, the roofline lower bound
`timeLB m r = max (fieldMax m r) (r.depth * m.latency)` and the ceiling arithmetic over `Nat`. This
module instantiates them at the ATTN-COOP numbers (`ATTN-COOP-PREDICTION.md`,
committed prediction P18; measured outcome in the perf-predictions note row P18), so the
`open: no Lean row yet (coopmat attention unformalised)` obligation of P18 carries a theorem that
builds.

## What is proved here, and by which theorem

* **The SIMT node is shared-load bound.** `simt_issue_bound_dominates`: the SIMT attention node's
  shared-memory load-issue term (`824 us`) exceeds its FMA term (`236 us`), so the shared port, not
  the FMA pipe, is the binding term.
* **Staging halves with the rows per workgroup.** `staging_halves` is the general `Nat`-division rule:
  with the staging traffic inversely proportional to the query rows per workgroup (`staging R =
  ceil (W / R)`), doubling the rows at most halves the staging term, up to one rounding step, as a
  lemma over symbols; `coop_stage_halves` is the measured instance (`155 us` at `R = 32` to `78 us`
  at `R = 64`).
* **The coopmat node is below the SIMT node and above its floor.** `coop_node_below_simt`: the
  measured coopmat node (`449 us`) is below the SIMT node (`1166 us`), and the `max (staging, mma)`
  floor (`155 us` at `R = 32`, `113 us` at `R = 64`) is below the measured node.
* **The node delta is a pp512 gain.** `attn_node_gain`: the `717 us` node delta divided into the
  pp512 total (`7660.7 us`) lies in `[9%, 10%)`, inside the P18 committed band `+7% [+3, +11]`.

## What is not covered

No hardware speed follows. Every constant is a `model` input copied from the named prediction row or
outcome line, not a measurement; no theorem here produces a byte, a cycle or a microsecond. The
pp512 total's own composition, GPU/SPIR-V codegen, the driver, barriers, bank conflicts, the absence
of an async copy and the per-subgroup softmax serialisation are outside every statement.
-/

namespace Perf

/-! ## 1 · The SIMT node: the shared-load issue term dominates the FMA term -/

/-- The SIMT attention node's FMA term, in us (ATTN-COOP prediction, current-node roofline: `236 us`).
`model`. -/
def simtFma : Nat := 236

/-- The SIMT attention node's shared-memory load-issue term, in us (ATTN-COOP prediction: `5.55 G`
thread-LDS at one warp-LDS per clock per SM gives `824 us`). `model`. -/
def simtLds : Nat := 824

/-- **The SIMT node is shared-load bound, not FMA bound.** The shared-load issue term exceeds the FMA
term: `236 < 824`. The node posts the measured `1166 us` against the `236 us` FMA floor because the
shared port is the binding term. -/
theorem simt_issue_bound_dominates : simtFma < simtLds := by decide

/-! ## 2 · K/V staging is inversely proportional to the rows per workgroup -/

/-- Total K/V staging work, in `us × query-row` (ATTN-COOP prediction: the `R = 32` staging is
`155 us` over `32` rows). `model`. -/
def stageWork : Nat := 4960

/-- K/V staging traffic at `R` query rows per workgroup: one K/V slab is staged per workgroup for its
`R` query rows, so the per-row traffic is the fixed total divided by `R`, rounded up. `model`. -/
def staging (R : Nat) : Nat := ceilDiv stageWork R

/-- **Doubling the rows at most halves the staging.** With the rule `staging R = ceil (W / R)`,
doubling the query rows per workgroup stages at most half the traffic, up to one rounding step:
`staging (2R) ≤ ceil (staging R / 2) + 1`. The rows and the total stay symbols; the only arithmetic is
`Nat` division. -/
theorem staging_halves {R : Nat} (hR : 0 < R) :
    staging (2 * R) ≤ ceilDiv (staging R) 2 + 1 := by
  unfold staging
  have hW : stageWork ≤ ceilDiv stageWork R * R := le_ceilDiv_mul hR
  have h1 : ceilDiv stageWork (2 * R) ≤ ceilDiv (ceilDiv stageWork R * R) (2 * R) :=
    ceilDiv_mono hW
  have hden : 0 < 2 * R := by omega
  have hb : ceilDiv stageWork R ≤ ceilDiv (ceilDiv stageWork R) 2 * 2 :=
    le_ceilDiv_mul (by norm_num : (0 : Nat) < 2)
  have h2 : ceilDiv (ceilDiv stageWork R * R) (2 * R) ≤ ceilDiv (ceilDiv stageWork R) 2 := by
    apply Chain.ceilDiv_le_of_le_mul hden
    calc ceilDiv stageWork R * R ≤ (ceilDiv (ceilDiv stageWork R) 2 * 2) * R :=
          Nat.mul_le_mul_right R hb
      _ = ceilDiv (ceilDiv stageWork R) 2 * (2 * R) := by ring
  omega

/-- **Coopmat staging at R=64 is at most half the R=32 staging, plus one.** The measured staging is
`155 us` at `R = 32` and `78 us` at `R = 64`; instance of `staging_halves` at `R = 32`. -/
theorem coop_stage_halves : staging (2 * 32) ≤ ceilDiv (staging 32) 2 + 1 :=
  staging_halves (R := 32) (by norm_num)

/-! ## 3 · The coopmat node below the SIMT node and above its floor -/

/-- The SIMT attention node, in us: the measured median (ATTN-COOP outcome: `1166.0 us`, six
interleaved rounds). `model`. -/
def simtNode : Nat := 1166

/-- The coopmat attention node, in us: the measured median (ATTN-COOP outcome: `449.5 us`). `model`. -/
def coopNode : Nat := 449

/-- The cooperative-matrix QKᵀ and P·V term, in us (ATTN-COOP prediction: `113 us`). `model`. -/
def coopMma : Nat := 113

/-- The coopmat node's roofline floor at `R` rows per workgroup: the larger of its staging term and
its MMA term. -/
def coopFloor (R : Nat) : Nat := max (staging R) coopMma

/-- **The measured coopmat node is below the SIMT node and above its roofline floor.** `449 < 1166`,
and the `max (staging, mma)` floor is `155 us` at `R = 32` and `113 us` at `R = 64`, both below
`449`. The floor carried, the measured node still carries the residual the model does not attribute. -/
theorem coop_node_below_simt :
    coopNode < simtNode ∧ coopFloor 32 < coopNode ∧ coopFloor 64 < coopNode := by
  refine ⟨by decide, by decide, by decide⟩

/-! ## 4 · The node delta as a pp512 gain -/

/-- The pp512 GPU-active total, in us (ATTN-COOP prediction: `7660.7 us`; the attention node is one
of its terms). `model`. -/
def pp512TotalUs : Nat := 7661

/-- The node delta, in us: `simtNode - coopNode = 1166 - 449 = 717`. -/
def nodeDelta : Nat := simtNode - coopNode

/-- **The node delta is a 9-10% pp512 gain.** `717 us` of `7660.7 us` is at least `9%` and below
`10%`, inside the P18 committed band `+7% [+3, +11]`. `Nat` inequalities only. -/
theorem attn_node_gain :
    9 * pp512TotalUs ≤ 100 * nodeDelta ∧ 100 * nodeDelta < 10 * pp512TotalUs ∧
    3 * pp512TotalUs ≤ 100 * nodeDelta ∧ 100 * nodeDelta ≤ 11 * pp512TotalUs := by
  decide

-- Non-vacuity of the model inputs and the measured instances.
#guard simtFma = 236
#guard simtLds = 824
#guard simtFma < simtLds
#guard staging 32 = 155
#guard staging 64 = 78
#guard staging 64 ≤ ceilDiv (staging 32) 2 + 1
#guard coopFloor 32 = 155
#guard coopFloor 64 = 113
#guard coopFloor 32 < coopNode ∧ coopFloor 64 < coopNode ∧ coopNode < simtNode
#guard nodeDelta = 717
#guard 9 * pp512TotalUs ≤ 100 * nodeDelta ∧ 100 * nodeDelta < 10 * pp512TotalUs

end Perf
