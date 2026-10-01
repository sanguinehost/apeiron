import LaneTrace
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Data.Fintype.Prod

/-!
# Shader Dot Equivalence: Model Scope

Arithmetic equivalence between the GLSL prefill shader (`src/spirv/q8_tile.comp`)
and the CPU block dot (`src/engine/chart/portable.rs` and `LaneTrace.lean`).

## What is proved
1. **Packed-dot split**: GLSL `dotPacked4x8EXT(w, x)` on 32-bit words equals the sum of the
   four signed 8-bit lane products:
   `packed_dot w x = ∑ i : Fin 4, sign8 w i * sign8 x i`
2. **32-lane block dot**: The shader's 8 packed quads equal the 32-lane signed dot
   computed by the CPU block (`cpu_block_dot (lane_weight w) (lane_act x)`).
3. **Canonical link**: Every signed lane product and the overall block dot are connected to
   the canonical `Apeiron.Trichotomy.AnyRing.ringMul σ (a, 0) (b, 0) = (a * b, 0)` diagonal
   channel (instantiating `LaneTrace.ringMul_channel0`).

## What is NOT proved (Trust Boundaries)
- GLSL compiler translation (`glslangValidator` to SPIR-V bytecode).
- The SPIR-V module emitter, descriptor sets, and pipeline layout.
- GPU hardware execution, driver scheduling, workgroup tiling, and barrier semantics.
- The extracted CPU body link (connecting `cpu_block_dot` to the Aeneas-extracted `exact_dot_loop0_loop0`).
Those remain separate trust boundaries for subsequent legs.

## Relation to `Spirv.Semantics`
`proof/corpus/Spirv/Semantics.lean` modeled `OpSUDot` (signed activation by unsigned weight)
for Q4 nibble splitting, pairing nibbles into 2D cells over `Word32 = W32 = ZMod (2^32)` and
showing channel 0 of `ringMul` at `σ = +1`. Here, the Q8 prefill shader uses `dotPacked4x8EXT`
over signed two's-complement 8-bit lanes (both weights and activations signed). This module
supersedes the signed·unsigned 2D cell reading with the direct signed·signed 32-lane dot product
and its link to canonical `ringMul σ` for any parameter `σ`.
-/

set_option autoImplicit false
set_option linter.apeiron.notation false

namespace ShaderDot

open BigOperators

/-- Extraction of byte `i : Fin 4` from a 32-bit word `w : UInt32` as a `Nat` in `[0, 255]`. -/
def byte (w : UInt32) (i : Fin 4) : Nat :=
  (w.toNat / 256 ^ i.val) % 256

/-- Two's-complement signed 8-bit interpretation of byte `i` of word `w : UInt32`.
    Maps `[0, 127]` to `[0, 127]` and `[128, 255]` to `[-128, -1]`. -/
def sign8 (w : UInt32) (i : Fin 4) : Int :=
  let b := byte w i
  if b < 128 then (b : Int) else (b : Int) - 256

/-- Operational model of GLSL `dotPacked4x8EXT(w, x)`:
    The unrolled sum of the four signed 8-bit lane products. -/
def packed_dot (w x : UInt32) : Int :=
  sign8 w 0 * sign8 x 0 +
  sign8 w 1 * sign8 x 1 +
  sign8 w 2 * sign8 x 2 +
  sign8 w 3 * sign8 x 3

/-- Signed two's-complement witness: word `0xFFFFFFFF` represents `-1` in all four byte lanes. -/
theorem sign8_all_ones (i : Fin 4) :
    sign8 (4294967295 : UInt32) i = -1 := by
  fin_cases i <;> rfl

/-- Concrete evaluation of `packed_dot` on all-negative-one and all-one words.
    Serves as an exact arithmetic anchor for mutation control. -/
theorem packed_dot_eval_concrete :
    packed_dot (4294967295 : UInt32) (16843009 : UInt32) = -4 := rfl

/-- Packed-dot split: `dotPacked4x8EXT(w, x)` equals the sum over `Fin 4` of the signed lane products. -/
theorem packed_dot_split (w x : UInt32) :
    packed_dot w x = ∑ i : Fin 4, sign8 w i * sign8 x i := by
  unfold packed_dot
  rw [Fin.sum_univ_four]

/-- The shader's block dot over 8 packed quads (32 lanes total). -/
def shader_block_dot (w x : Fin 8 → UInt32) : Int :=
  ∑ q : Fin 8, packed_dot (w q) (x q)

/-- The 32-lane signed dot computed by the CPU block loop (`portable_i8_dot`). -/
def cpu_block_dot (row code : Fin 32 → Int) : Int :=
  ∑ l : Fin 32, row l * code l

/-- Canonical equivalence between 8 packed quads of 4 bytes and 32 linear lanes. -/
def laneEquiv : Fin 8 × Fin 4 ≃ Fin 32 :=
  finProdFinEquiv

/-- Quad index `q ∈ [0, 7]` containing lane `l ∈ [0, 31]`. -/
def lane_to_quad (l : Fin 32) : Fin 8 :=
  (laneEquiv.symm l).1

/-- Byte index `i ∈ [0, 3]` within the quad for lane `l ∈ [0, 31]`. -/
def lane_to_byte (l : Fin 32) : Fin 4 :=
  (laneEquiv.symm l).2

/-- Unpacked signed weight byte at lane `l : Fin 32` from 8 packed words. -/
def lane_weight (w : Fin 8 → UInt32) (l : Fin 32) : Int :=
  sign8 (w (lane_to_quad l)) (lane_to_byte l)

/-- Unpacked signed activation byte at lane `l : Fin 32` from 8 packed words. -/
def lane_act (x : Fin 8 → UInt32) (l : Fin 32) : Int :=
  sign8 (x (lane_to_quad l)) (lane_to_byte l)

/-- 32-lane block dot equivalence:
    The shader's 8 packed quads equal the 32-lane signed dot computed by the CPU block. -/
theorem shader_block_dot_eq_cpu_block_dot (w x : Fin 8 → UInt32) :
    shader_block_dot w x = cpu_block_dot (lane_weight w) (lane_act x) := by
  unfold shader_block_dot cpu_block_dot
  have h_split : (∑ q : Fin 8, packed_dot (w q) (x q)) =
      ∑ q : Fin 8, ∑ i : Fin 4, sign8 (w q) i * sign8 (x q) i := by
    apply Finset.sum_congr rfl
    intro q _
    exact packed_dot_split (w q) (x q)
  rw [h_split]
  rw [← Finset.sum_product', Finset.univ_product_univ]
  apply Fintype.sum_equiv laneEquiv
  intro p
  dsimp [lane_weight, lane_act, lane_to_quad, lane_to_byte]
  rw [Equiv.symm_apply_apply]

/-- Canonical link for a single lane product:
    Each signed lane product is the channel 0 projection of canonical `ringMul σ (a, 0) (b, 0)`. -/
theorem lane_product_canonical (σ : Int) (a b : Int) :
    a * b = (Apeiron.Trichotomy.AnyRing.ringMul σ (a, 0) (b, 0)).1 := by
  rw [LaneTrace.ringMul_channel0]

/-- Canonical link for the packed dot:
    GLSL `dotPacked4x8EXT(w, x)` is the sum of four channel-0 projections of canonical `ringMul`. -/
theorem packed_dot_canonical (σ : Int) (w x : UInt32) :
    packed_dot w x = ∑ i : Fin 4, (Apeiron.Trichotomy.AnyRing.ringMul σ (sign8 w i, 0) (sign8 x i, 0)).1 := by
  rw [packed_dot_split]
  apply Finset.sum_congr rfl
  intro i _
  exact lane_product_canonical σ (sign8 w i) (sign8 x i)

/-- Canonical link for the 32-lane block dot:
    The 32-lane block dot is the sum of 32 channel-0 projections of canonical `ringMul`. -/
theorem cpu_block_dot_canonical (σ : Int) (row code : Fin 32 → Int) :
    cpu_block_dot row code = ∑ l : Fin 32, (Apeiron.Trichotomy.AnyRing.ringMul σ (row l, 0) (code l, 0)).1 := by
  unfold cpu_block_dot
  apply Finset.sum_congr rfl
  intro l _
  exact lane_product_canonical σ (row l) (code l)

/-- Full equivalence chain:
    Shader 8-quad block dot = 32-lane CPU signed dot = canonical AnyRing diagonal channel sum. -/
theorem shader_cpu_canonical_chain (σ : Int) (w x : Fin 8 → UInt32) :
    shader_block_dot w x = cpu_block_dot (lane_weight w) (lane_act x) ∧
    cpu_block_dot (lane_weight w) (lane_act x) =
      ∑ l : Fin 32, (Apeiron.Trichotomy.AnyRing.ringMul σ (lane_weight w l, 0) (lane_act x l, 0)).1 := by
  refine ⟨shader_block_dot_eq_cpu_block_dot w x, cpu_block_dot_canonical σ (lane_weight w) (lane_act x)⟩

end ShaderDot

/- Audit every owned declaration under the closed dependency cone. -/
open Lean Elab Command in
elab "#shader_dot_audit" : command => do
  let environment ← getEnv
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  let constants := environment.constants.fold (fun acc constant _ =>
    if (`ShaderDot).isPrefixOf constant then constant :: acc else acc) ([] : List Lean.Name)
  if constants.length < 15 then throwError "SHADER_DOT_AUDIT_POPULATION"
  let mut failed := false
  for constant in constants do
    let dependencies ← liftTermElabM <| Lean.collectAxioms constant
    let bad := dependencies.filter (fun value => !allowed.contains value)
    unless bad.isEmpty do
      logError m!"{constant}: forbidden dependencies {bad.toList}"
      failed := true
    logInfo m!"{constant}: dependencies {dependencies.toList}"
  if failed then throwError "SHADER_DOT_CLOSED_CONE_REJECT"
  logInfo m!"SHADER_DOT_KERNEL_OK declarations={constants.length}"

#shader_dot_audit
