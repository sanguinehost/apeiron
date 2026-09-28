/-
# A kernel-grounded pre-norm transformer block bound

`Nous.Model.BlockGrid` assembles the pre-norm block `fullBlock'` with explicit admissible
domains and the total SiLU stage, but its seven projections enter as abstract `MatvecSpec` values
whose per-coordinate defect `matvecEps` is a field hypothesis, and its two norms are the scalar
`rmsStage` at an externally fixed radicand. `Numeric.MatvecBridgeGrid` and
`Nous.Model.RmsVectorStage` land the two missing halves: a `MatvecSpec` whose defect is *proved*
from the extracted matvec on every admissible grid input, and a vector RMSNorm `Stage` whose
radicand is the rounded mean of the input. This module rebuilds `fullBlock'` with both.

## The grounding ledger

| stage slot | grounded in | defect | expansion |
|---|---|---|---|
| `q`, `k`, `v`, `o`, `gate`, `up`, `down` projections | `MatvecBridgeGrid.matvecSpecGrid` (`Numeric/MatvecBridgeGrid.lean:156`), transported to the block widths by `GridProj.spec` | `matvecEps`, proved in `matvec_stage_grid_proved` (`Numeric/MatvecBridgeGrid.lean:207`) from `matvec_refines_proved` | max row `ℓ¹` norm, `le_maxRowBound` (`Numeric/MatvecBridge.lean:94`) |
| first and second norm | `Nous.Model.rmsVectorStage` (`Nous/Model/RmsVectorStage.lean:405`) | `rmsVecDelta B A0`, from `rms_norm_refines_ideal` through `rmsVecOut_bound` (`Nous/Model/RmsVectorStage.lean:390`) | `(1 + n)/√(toReal eps)`, `rmsIdealVec_lipschitz` (`Nous/Model/RmsVectorStage.lean:62`) |
| SiLU gate branch | `Nous.Model.siluStage` (`Nous/Model/BlockGrid.lean:156`) | `siluErrTotal B`, from `silu_refines_total` (`Numeric/SiluSaturation.lean:150`) | `3`, `siluReal_lipschitz` |
| optional pre-projection rotation | `Nous.Model.rotationStage` (`Nous/Model/RotationStage.lean:294`), wrapped by `RotSlot.ofRotation` | a caller hypothesis: the Rust `fwht_fixed_refines` link has not landed | `√(2^k)`, `hadamardRef_lipschitz_sup` (`Nous/Model/RotationStage.lean:151`) |

Two hypotheses remain unrefined, exactly as `BlockResidual` carries them:

* **attention** (`br.attn`): `attn_head_refines_v2_proved` (`AxiomAudit.lean:1686`) is per-head and
  is not yet a `Stage`, and `br.attn.ref` pins no formula (`BlockComposition.lean:374`);
* **the elementwise gate/up product** (`br.mul`): no kernel refinement exists; only its reference
  form is pinned (`BlockResidual.mul_ref_on`, `BlockComposition.lean:384`).

## What is proved and where the proof stops

`fullBlockGrounded` is the same assembly as `fullBlock'` with `GridProj.spec` in every projection
slot, `rmsVectorStage` at both norm sites, and an optional `RotSlot` before each projection;
`block_error_bound_grounded` is the composed defect bound on the block's own domain, and
`fullBlockGrounded_delta_eq` unfolds `delta` once, into the two residual branches. The rotation's
executed map is a hypothesis inside `rotationStage` itself, so `RotSlot.ofRotation` is grounded no
more firmly than `rotationStage`; the Rust link is the residual obligation, not a claim made here.

Non-vacuity: each component stage's domain is witnessed in its source module
(`matvecSpecGrid_dom_nonempty`, `rmsVectorStage_dom_witness`, `siluStage_dom_witness`). A joint
witness inside the composed block domain is not constructed here, so the bound is proved at the
composed domain while its inhabitedness stays the residual obligation.

Trust boundary: no claim about rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V, the `unsafe`
container, or the dequant-to-ideal-weight reading (the class-(b) `tested` boundary of
`Numeric/EngineMatvecRefinement.lean`). The grid algebra is checked at the abstract real level
under the closed axiom cone `{propext, Classical.choice, Quot.sound}`.

The notation and citation linters of the vendored `apeiron` corpus are switched off, as in
`Nous/Model/RotationStage.lean`, because this bridge module uses the corpus's imported vocabulary
without adopting the corpus's gloss protocol.
-/
import Nous.Model.BlockGrid
import Nous.Model.RmsVectorStage
import Numeric.MatvecBridgeGrid
import Nous.Model.RotationStage

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open apeiron_word Aeneas Aeneas.Std Result Classical
open MatvecBridgeGrid

noncomputable section

/-! ## Grid-grounded projections -/

/-- A projection slot of the block, carried by the `matvecSpecGrid` arguments and the two width
equalities that place them at the block's `Fin n`/`Fin m` widths. `spec` is `matvecSpecGrid` at
those arguments, transported along `h_in` and `h_out`; the defect of `spec.stage` is therefore the
proved `matvecEps` of `matvec_stage_grid_proved`, not a field hypothesis. -/
structure GridProj (n m : ℕ) where
  /-- The GGML type tag of the quantized weight. -/
  ggml_type : U32
  /-- The raw quantized weight bytes. -/
  raw : Slice U8
  /-- The projection's output width, as the extracted kernel takes it. -/
  n_out : Usize
  /-- The projection's input width, as the extracted kernel takes it. -/
  n_in : Usize
  /-- The output width equals the block's `m`. -/
  h_out : n_out.val = m
  /-- The input width equals the block's `n`. -/
  h_in : n_in.val = n
  /-- The number of quantized blocks per row. -/
  qk : Usize
  /-- The quantized block size in bytes. -/
  bb : Usize
  /-- The number of blocks per row. -/
  blocks : Usize
  /-- The row stride in bytes. -/
  row_bytes : Usize
  /-- The required scratch length. -/
  need : Usize
  /-- The dequantizer reading row `i`. -/
  deq : Nat → Slice I64

/-- The `MatvecSpec` of a `GridProj`: `matvecSpecGrid` at the stored arguments, moved to the
block's widths by the two stored equalities. -/
def GridProj.spec {n m : ℕ} (P : GridProj n m) : MatvecSpec n m :=
  Eq.mp (congrArg₂ MatvecSpec P.h_in P.h_out)
    (matvecSpecGrid P.ggml_type P.raw P.n_out P.n_in P.qk P.bb P.blocks P.row_bytes P.need P.deq)

/-- The `delta` of a grid projection's stage is the proved `matvecEps`. -/
theorem GridProj.spec_stage_delta {n m : ℕ} (P : GridProj n m) :
    P.spec.stage.delta = matvecEps := rfl

/-- The `L` of a grid projection's stage is the spec's row bound. -/
theorem GridProj.spec_stage_L {n m : ℕ} (P : GridProj n m) :
    P.spec.stage.L = P.spec.rowBound := rfl

/-- **Grid projection grounding.** On its own domain the stage of a `GridProj` differs from its
reference by at most `matvecEps`; the proof is the transported `matvecSpecGrid` defect, itself
`matvec_refines_proved`. -/
theorem gridProj_stage_defect {n m : ℕ} (P : GridProj n m) :
    ∀ z ∈ P.spec.stage.dom, ‖P.spec.stage.exec z - P.spec.stage.ref z‖ ≤ matvecEps :=
  fun z hz => P.spec.stage.defect z hz

/-! ## Optional Hadamard rotation before a projection -/

/-- A rotation slot: a `Bool` flag and a same-width `Stage`. `ofRotation` builds it from the
`rotationStage` data, so the flag carries a genuinely rotation-shaped stage rather than an
arbitrary one. -/
structure RotSlot (n : ℕ) where
  /-- Whether the rotation applies. -/
  use : Bool
  /-- The rotation stage itself. -/
  rot : Stage (Fin n → ℝ) (Fin n → ℝ)

/-- The rotation slot of a `rotationStage`: the width `n` equals `2^k`, so the stage is transported
to `Fin n` by `hk`. Its `delta` is the caller's `delta` hypothesis and its `L` is `√(2^k)`. -/
def RotSlot.ofRotation {n : ℕ} (k : ℕ) (hk : (2 : ℕ) ^ k = n) (s : Fin (2 ^ k) → ℝ)
    (hs : IsSign s) (B delta : ℝ) (exec : (Fin (2 ^ k) → ℝ) → (Fin (2 ^ k) → ℝ))
    (hdelta : 0 ≤ delta)
    (hdefect : ∀ x, (∀ i, |x i| ≤ B) → ‖exec x - hadamardRef k s x‖ ≤ delta) :
    RotSlot n :=
  { use := true
    rot := Eq.mp (congrArg (fun m => Stage (Fin m → ℝ) (Fin m → ℝ)) hk)
      (rotationStage k s hs B delta exec hdelta hdefect) }

/-- The rotation of `RotSlot.ofRotation` has defect `delta`. -/
theorem RotSlot.ofRotation_rot_delta {n : ℕ} (k : ℕ) (hk : (2 : ℕ) ^ k = n)
    (s : Fin (2 ^ k) → ℝ) (hs : IsSign s) (B delta : ℝ)
    (exec : (Fin (2 ^ k) → ℝ) → (Fin (2 ^ k) → ℝ)) (hdelta : 0 ≤ delta)
    (hdefect : ∀ x, (∀ i, |x i| ≤ B) → ‖exec x - hadamardRef k s x‖ ≤ delta) :
    (RotSlot.ofRotation k hk s hs B delta exec hdelta hdefect).rot.delta = delta := by
  unfold RotSlot.ofRotation
  cases hk
  rfl

/-- The rotation of `RotSlot.ofRotation` has expansion `√(2^k)`, the `ℓ∞` bound of
`hadamardRef_lipschitz_sup`. -/
theorem RotSlot.ofRotation_rot_L {n : ℕ} (k : ℕ) (hk : (2 : ℕ) ^ k = n)
    (s : Fin (2 ^ k) → ℝ) (hs : IsSign s) (B delta : ℝ)
    (exec : (Fin (2 ^ k) → ℝ) → (Fin (2 ^ k) → ℝ)) (hdelta : 0 ≤ delta)
    (hdefect : ∀ x, (∀ i, |x i| ≤ B) → ‖exec x - hadamardRef k s x‖ ≤ delta) :
    (RotSlot.ofRotation k hk s hs B delta exec hdelta hdefect).rot.L
      = Real.sqrt ((2 : ℝ) ^ k) := by
  unfold RotSlot.ofRotation
  cases hk
  rfl

/-- A projection with an optional preceding rotation: `cond` on the slot's flag, so an unused slot
contributes nothing and a used one contributes `Stage.comp`'s `delta₂ + L₂·delta₁`. -/
def rotProj {n m : ℕ} (R : RotSlot n) (s : MatvecSpec n m) : Stage (Fin n → ℝ) (Fin m → ℝ) :=
  cond R.use (R.rot.comp s.stage) s.stage

/-- An active rotation composes the projection's own defect with the rotation's, scaled by the
projection's expansion factor. -/
theorem rotProj_delta_true {n m : ℕ} (R : RotSlot n) (s : MatvecSpec n m) (h : R.use = true) :
    (rotProj R s).delta = s.stage.delta + s.stage.L * R.rot.delta := by
  simp [rotProj, h, Stage.comp]

/-- An inactive rotation leaves the projection's defect alone. -/
theorem rotProj_delta_false {n m : ℕ} (R : RotSlot n) (s : MatvecSpec n m) (h : R.use = false) :
    (rotProj R s).delta = s.stage.delta := by
  simp [rotProj, h]

/-- An active rotation scales the projection's expansion by the rotation's. -/
theorem rotProj_L_true {n m : ℕ} (R : RotSlot n) (s : MatvecSpec n m) (h : R.use = true) :
    (rotProj R s).L = R.rot.L * s.stage.L := by
  simp [rotProj, h, Stage.comp]
  ring

/-- An inactive rotation leaves the projection's expansion alone. -/
theorem rotProj_L_false {n m : ℕ} (R : RotSlot n) (s : MatvecSpec n m) (h : R.use = false) :
    (rotProj R s).L = s.stage.L := by
  simp [rotProj, h]

/-! ## The grounded sub-blocks -/

/-- The `(Q, K, V)` fan-out with grid-grounded projections and optional input rotations. -/
def qkvFanoutGrounded {d p : ℕ} (q k v : GridProj d p) (qR kR vR : RotSlot d) :
    Stage (Fin d → ℝ) ((Fin p → ℝ) × (Fin p → ℝ) × (Fin p → ℝ)) :=
  (rotProj qR q.spec).prod ((rotProj kR k.spec).prod (rotProj vR v.spec))

/-- `qkvFanoutGrounded`'s defect is the pairwise `max` of the three rotated projections'. -/
theorem qkvFanoutGrounded_delta_eq {d p : ℕ} (q k v : GridProj d p) (qR kR vR : RotSlot d) :
    (qkvFanoutGrounded q k v qR kR vR).delta =
      max (rotProj qR q.spec).delta
        (max (rotProj kR k.spec).delta (rotProj vR v.spec).delta) := rfl

/-- The attention sub-block: fan out to `(Q, K, V)`, apply the unrefined `attn`, project back with
the grid-grounded output matvec (optionally rotated). -/
def attnSubBlockGrounded {d p h : ℕ} (q k v : GridProj d p) (o : GridProj p d)
    (qR kR vR : RotSlot d) (oR : RotSlot p) (br : BlockResidual p h) :
    Stage (Fin d → ℝ) (Fin d → ℝ) :=
  (qkvFanoutGrounded q k v qR kR vR).comp (br.attn.comp (rotProj oR o.spec))

/-- `attnSubBlockGrounded`'s defect, closed in the rotated output projection and the `attn`
hypothesis; the three input projections enter only through the fan-out `max`. -/
theorem attnSubBlockGrounded_delta_eq {d p h : ℕ} (q k v : GridProj d p) (o : GridProj p d)
    (qR kR vR : RotSlot d) (oR : RotSlot p) (br : BlockResidual p h) :
    (attnSubBlockGrounded q k v o qR kR vR oR br).delta =
      ((rotProj oR o.spec).delta + (rotProj oR o.spec).L * br.attn.delta)
        + ((rotProj oR o.spec).L * br.attn.L)
            * (qkvFanoutGrounded q k v qR kR vR).delta := rfl

/-- The `(SiLU(gate), up)` fan-out with grid-grounded projections and optional input rotations. -/
def gateUpFanoutGrounded {d h : ℕ} (g u : GridProj d h) (gR uR : RotSlot d)
    (Bs : ℝ) (hBs : 0 ≤ Bs) :
    Stage (Fin d → ℝ) ((Fin h → ℝ) × (Fin h → ℝ)) :=
  ((rotProj gR g.spec).comp (Stage.pi (siluStage Bs hBs))).prod (rotProj uR u.spec)

/-- `gateUpFanoutGrounded`'s defect: `max` of the gate branch's `siluErrTotal Bs + 3·rotProj.delta`
and the up branch's `rotProj.delta`. -/
theorem gateUpFanoutGrounded_delta_eq {d h : ℕ} (g u : GridProj d h) (gR uR : RotSlot d)
    (Bs : ℝ) (hBs : 0 ≤ Bs) :
    (gateUpFanoutGrounded g u gR uR Bs hBs).delta =
      max (siluErrTotal Bs + 3 * (rotProj gR g.spec).delta) (rotProj uR u.spec).delta := rfl

/-- The FFN sub-block with grid-grounded gate, up and down projections, the total SiLU stage and
the unrefined elementwise product. -/
def ffnSubBlockGrounded {d p h : ℕ} (g u : GridProj d h) (dn : GridProj h d)
    (gR uR : RotSlot d) (dnR : RotSlot h) (br : BlockResidual p h) (Bs : ℝ) (hBs : 0 ≤ Bs) :
    Stage (Fin d → ℝ) (Fin d → ℝ) :=
  (gateUpFanoutGrounded g u gR uR Bs hBs).comp (br.mul.comp (rotProj dnR dn.spec))

/-- `ffnSubBlockGrounded`'s defect, closed in the rotated down projection, the `mul` hypothesis and
the gate/up fan-out. -/
theorem ffnSubBlockGrounded_delta_eq {d p h : ℕ} (g u : GridProj d h) (dn : GridProj h d)
    (gR uR : RotSlot d) (dnR : RotSlot h) (br : BlockResidual p h) (Bs : ℝ) (hBs : 0 ≤ Bs) :
    (ffnSubBlockGrounded g u dn gR uR dnR br Bs hBs).delta =
      ((rotProj dnR dn.spec).delta + (rotProj dnR dn.spec).L * br.mul.delta)
        + ((rotProj dnR dn.spec).L * br.mul.L)
            * (gateUpFanoutGrounded g u gR uR Bs hBs).delta := rfl

/-! ## The grounded block -/

/-- The block's arguments: the seven grid projections, the seven rotation slots, the two residual
hypotheses, the two vector-norm data and the SiLU cap. Bundling them keeps the block's `delta`
unfolds readable and the projection-to-slot correspondence explicit. -/
structure GroundedBlock (d p h : ℕ) where
  /-- The query projection. -/
  q : GridProj d p
  /-- The key projection. -/
  k : GridProj d p
  /-- The value projection. -/
  v : GridProj d p
  /-- The attention output projection. -/
  o : GridProj p d
  /-- The gate projection. -/
  g : GridProj d h
  /-- The up projection. -/
  u : GridProj d h
  /-- The down projection. -/
  dn : GridProj h d
  /-- The query input rotation slot. -/
  qR : RotSlot d
  /-- The key input rotation slot. -/
  kR : RotSlot d
  /-- The value input rotation slot. -/
  vR : RotSlot d
  /-- The attention output input rotation slot. -/
  oR : RotSlot p
  /-- The gate input rotation slot. -/
  gR : RotSlot d
  /-- The up input rotation slot. -/
  uR : RotSlot d
  /-- The down input rotation slot. -/
  dnR : RotSlot h
  /-- The two unrefined operations: attention and the elementwise gate/up product. -/
  br : BlockResidual p h
  /-- The residual width is nonempty, so the vector norm's `1 ≤ n` premise holds. -/
  hd : 1 ≤ d
  /-- The first norm's epsilon. -/
  eps1 : I64
  /-- The first norm's epsilon is positive. -/
  heps1 : 0 < toReal eps1
  /-- The first norm's coordinate cap. -/
  B1 : ℝ
  /-- The first norm's radicand floor. -/
  A1 : ℝ
  /-- The first cap is nonnegative. -/
  hB1 : 0 ≤ B1
  /-- The first floor clears the rounding radius. -/
  hA1 : 1 / (2 * q32) < A1
  /-- The second norm's epsilon. -/
  eps2 : I64
  /-- The second norm's epsilon is positive. -/
  heps2 : 0 < toReal eps2
  /-- The second norm's coordinate cap. -/
  B2 : ℝ
  /-- The second norm's radicand floor. -/
  A2 : ℝ
  /-- The second cap is nonnegative. -/
  hB2 : 0 ≤ B2
  /-- The second floor clears the rounding radius. -/
  hA2 : 1 / (2 * q32) < A2
  /-- The SiLU magnitude cap. -/
  Bs : ℝ
  /-- The SiLU cap is nonnegative. -/
  hBs : 0 ≤ Bs

/-- **The kernel-grounded pre-norm transformer block.** The same assembly as `fullBlock'`, with
`GridProj.spec` in every projection slot, `rmsVectorStage` at both norm sites (the radicand is the
rounded input mean, not an external constant), the total `siluStage` in the FFN gate branch, and an
optional `rotationStage` before each projection. Only `br.attn` and `br.mul` remain hypotheses. -/
def fullBlockGrounded {d p h : ℕ} (G : GroundedBlock d p h) :
    Stage (Fin d → ℝ) (Fin d → ℝ) :=
  (Stage.residual ((rmsVectorStage d G.hd G.eps1 G.heps1 G.B1 G.A1 G.hB1 G.hA1).comp
      (attnSubBlockGrounded G.q G.k G.v G.o G.qR G.kR G.vR G.oR G.br))).comp
    (Stage.residual ((rmsVectorStage d G.hd G.eps2 G.heps2 G.B2 G.A2 G.hB2 G.hA2).comp
      (ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs)))

/-- **Grounded block error bound.** On `fullBlockGrounded`'s composed domain the executing block
differs from the reference block by at most its `delta`; every term of that `delta` traces to a
kernel-grounded stage except the two `BlockResidual` hypotheses. -/
theorem block_error_bound_grounded {d p h : ℕ} (G : GroundedBlock d p h) (x : Fin d → ℝ)
    (hx : x ∈ (fullBlockGrounded G).dom) :
    ‖(fullBlockGrounded G).exec x - (fullBlockGrounded G).ref x‖ ≤ (fullBlockGrounded G).delta :=
  (fullBlockGrounded G).defect x hx

/-- One-step unfold of `fullBlockGrounded.delta`, mirroring `fullBlock'_delta_eq`: the top-level
`Stage.comp` recurrence `block₂.delta + block₂.L · block₁.delta` over the two residual branches.
The second branch's expansion factor already contains `1 +` from `Stage.residual`. -/
theorem fullBlockGrounded_delta_eq {d p h : ℕ} (G : GroundedBlock d p h) :
    (fullBlockGrounded G).delta =
      (Stage.residual ((rmsVectorStage d G.hd G.eps2 G.heps2 G.B2 G.A2 G.hB2 G.hA2).comp
          (ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs))).delta
        + (Stage.residual ((rmsVectorStage d G.hd G.eps2 G.heps2 G.B2 G.A2 G.hB2 G.hA2).comp
              (ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs))).L
          * (Stage.residual ((rmsVectorStage d G.hd G.eps1 G.heps1 G.B1 G.A1 G.hB1 G.hA1).comp
              (attnSubBlockGrounded G.q G.k G.v G.o G.qR G.kR G.vR G.oR G.br))).delta :=
  rfl

/-- The first residual branch's `delta`, closed in the first vector norm and the attention
sub-block: `attnSubBlock.delta + attnSubBlock.L · rmsVecDelta B1 A1`. -/
theorem fullBlockGrounded_delta_eq₁ {d p h : ℕ} (G : GroundedBlock d p h) :
    (Stage.residual ((rmsVectorStage d G.hd G.eps1 G.heps1 G.B1 G.A1 G.hB1 G.hA1).comp
        (attnSubBlockGrounded G.q G.k G.v G.o G.qR G.kR G.vR G.oR G.br))).delta =
      (attnSubBlockGrounded G.q G.k G.v G.o G.qR G.kR G.vR G.oR G.br).delta
        + (attnSubBlockGrounded G.q G.k G.v G.o G.qR G.kR G.vR G.oR G.br).L
          * rmsVecDelta G.B1 G.A1 :=
  rfl

/-- The second residual branch's `delta`, closed in the second vector norm and the FFN sub-block:
`ffnSubBlock.delta + ffnSubBlock.L · rmsVecDelta B2 A2`. -/
theorem fullBlockGrounded_delta_eq₂ {d p h : ℕ} (G : GroundedBlock d p h) :
    (Stage.residual ((rmsVectorStage d G.hd G.eps2 G.heps2 G.B2 G.A2 G.hB2 G.hA2).comp
        (ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs))).delta =
      (ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs).delta
        + (ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs).L
          * rmsVecDelta G.B2 G.A2 :=
  rfl

/-- **Closed form.** `fullBlockGrounded.delta` expanded through both residual branches and both
vector norms, in the order the executing block runs them: the second norm's `rmsVecDelta B2 A2`,
then the FFN's defect scaled by the second norm's expansion `(1 + d)/√(toReal eps2)`, then that
branch's residual expansion `1 + …` times the first branch (the first norm's `rmsVecDelta B1 A1`
plus the attention defect scaled by `(1 + d)/√(toReal eps1)`). -/
theorem fullBlockGrounded_delta_closed {d p h : ℕ} (G : GroundedBlock d p h) :
    (fullBlockGrounded G).delta =
      ((ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs).delta
          + (ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs).L
            * rmsVecDelta G.B2 G.A2)
        + (1 + (ffnSubBlockGrounded G.g G.u G.dn G.gR G.uR G.dnR G.br G.Bs G.hBs).L
              * ((1 + (d : ℝ)) / Real.sqrt (toReal G.eps2)))
          * ((attnSubBlockGrounded G.q G.k G.v G.o G.qR G.kR G.vR G.oR G.br).delta
              + (attnSubBlockGrounded G.q G.k G.v G.o G.qR G.kR G.vR G.oR G.br).L
                * rmsVecDelta G.B1 G.A1) :=
  rfl

end

end Nous.Model
