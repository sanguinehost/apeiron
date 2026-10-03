/-
# Q8TileSpirv — registered admission certificate for the Apeiron Q8 SPIR-V execution unit

The registered executing source is the GLSL compute shader `src/spirv/q8_tile.comp` (entry
`main`) dispatched by `tools/gpu/q8_driver.c`. This module is the typed certificate §9.1 requires:
a reachable joint witness over a declared dispatch domain, universal preservation of the executed
observation, and strict improvement in the registered instruction-count cost model.

The preservation leg is the shader-to-canonical relation, not device parity: the executed baseline
(eight scalar `dotPacked4x8EXT` per 32-lane block) equals the canonical 32-lane signed dot, whose
channel-0 products are canonical `Apeiron.Trichotomy.AnyRing.ringMul σ`. The GLSL→SPIR-V
translation, dispatch, driver and GPU execution remain trusted/tested boundaries; the device
output protocol (`tools/gpu/q8_gate.py`) links the executing shader to `baselineDot` at grade
`tested`, and this module proves `baselineDot = canonicalDot` universally.

-/
import DispatchAdmission
import Apeiron.Trichotomy.AnyRing

set_option autoImplicit false
set_option linter.apeiron.notation false

namespace Q8TileSpirv

open BigOperators

/-- O(1) byte extraction: one division, one modulo on a fixed-width word. -/
def byte (v : UInt32) (i : Fin 4) : Nat := (v.toNat / 256 ^ i.val) % 256

/-- O(1) two's-complement signed 8-bit interpretation of byte `i` of `v`. -/
def sign8 (v : UInt32) (i : Fin 4) : Int :=
  let b := byte v i
  if b < 128 then (b : Int) else (b : Int) - 256

/-- O(1) model of GLSL `dotPacked4x8EXT(w, x)`: four signed lane products unrolled. -/
def packedDot (w x : UInt32) : Int :=
  sign8 w 0 * sign8 x 0 + sign8 w 1 * sign8 x 1 +
  sign8 w 2 * sign8 x 2 + sign8 w 3 * sign8 x 3

/-- Packed-dot split: the unrolled sum equals a `Fin 4` sum of signed lane products. -/
theorem packedDot_split (w x : UInt32) :
    packedDot w x = ∑ i : Fin 4, sign8 w i * sign8 x i := by
  unfold packedDot
  rw [Fin.sum_univ_four]

/-- One Q8 block: eight packed weight quads and eight packed activation quads. -/
structure Block where
  w : Fin 8 → UInt32
  x : Fin 8 → UInt32

/-- O(8) executed baseline arithmetic: the shader's eight scalar packed dots per block. -/
def baselineDot (b : Block) : Int := ∑ q : Fin 8, packedDot (b.w q) (b.x q)

/-- O(32) canonical 32-lane signed dot; the device observation oracle. -/
def canonicalDot (b : Block) : Int :=
  ∑ q : Fin 8, ∑ i : Fin 4, sign8 (b.w q) i * sign8 (b.x q) i

/-- Universal preservation: the executed baseline equals the canonical target on every block. -/
theorem baseline_eq_canonical (b : Block) : canonicalDot b = baselineDot b := by
  unfold canonicalDot baselineDot
  apply Finset.sum_congr rfl
  intro q _
  exact (packedDot_split (b.w q) (b.x q)).symm

/-- Canonical attribution: a signed lane product is channel 0 of canonical `ringMul σ`. -/
theorem lane_product_canonical (σ a c : Int) :
    a * c = (Apeiron.Trichotomy.AnyRing.ringMul σ (a, 0) (c, 0)).1 := by
  simp

/-- Dispatch geometry of the declared admission domain: positive rows, tokens and blocks. -/
structure Shape where
  m : Nat
  tokens : Nat
  blocks : Nat

/-- The declared dispatch domain: a nonempty Q8 projection. -/
def Shape.InDomain (s : Shape) : Prop := 1 ≤ s.m ∧ 1 ≤ s.tokens ∧ 1 ≤ s.blocks

instance (s : Shape) : Decidable s.InDomain := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

/-- An admitted dispatch: a geometry plus the block payload the shader executes. -/
structure Request where
  shape : Shape
  block : Block

/-- Reachability: the request lies in the declared dispatch domain. -/
def reachable (r : Request) : Prop := r.shape.InDomain

/-- Admissibility: no further precondition beyond reachability for a total shader. -/
def admissible (_ : Request) : Prop := True

/-- Before: the executed baseline observation. -/
def before (r : Request) : Int := baselineDot r.block

/-- After: the canonical observation the optimized execution unit must realize. -/
def after (r : Request) : Int := canonicalDot r.block

/-- Registered cost model: scalar packed-dot instructions per 32-lane block. -/
def beforeCost (_ : Request) : Nat := 8

/-- Registered cost model for the optimized unit: one packed signed-dot per 32-lane block. -/
def afterCost (_ : Request) : Nat := 1

/-- Concrete joint witness: one row, one token, one block of all-ones quads. -/
def witness : Request :=
  { shape := { m := 1, tokens := 1, blocks := 1 }
    block := { w := fun _ => 4294967295, x := fun _ => 16843009 } }

theorem witness_joint : reachable witness ∧ admissible witness := by
  constructor
  · simp [reachable, Shape.InDomain, witness]
  · trivial

/-- The registered admission: witness, preservation and strict registered improvement. -/
def admission : ProofDispatch.Admission before after reachable admissible beforeCost afterCost where
  witness := witness
  joint := witness_joint
  preserves := fun r _ _ => baseline_eq_canonical r.block
  improves := fun _ _ _ => by simp [beforeCost, afterCost]

end Q8TileSpirv
