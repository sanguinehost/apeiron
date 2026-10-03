/-
# Int8CoopmatSpirv — registered admission certificate for the Apeiron int8 cooperative-matrix
Q8 SPIR-V execution unit

The registered executing source is the int8 cooperative-matrix compute shader
`src/spirv/int8_coopmat_q8.comp` (entry `main`) dispatched by `tools/gpu/q8_driver.c`. The optimized
execution unit replaces **only** the per-block 32-lane signed dot of the incumbent `q8_tile.comp`
(eight scalar `dotPacked4x8EXT`) with one `OpCooperativeMatrixMulAddKHR` (M16 N16 K32,
`A=SINT8 B=SINT8 C=SINT32 R=SINT32`, `sat=0`); the checked i128 epilogue is carried unchanged.

This module is the typed certificate §9.1 requires. Its `preserves` leg is the **dot AND the checked
epilogue**, not the dot alone: `before` and `after` are functions into the full observation
(`value : Option Int`, `flags : Nat`), each running the same two-limb accumulation, Q32.32
round-to-nearest narrowing and refusal logic modelled from `src/spirv/q8_tile.comp:25-59,117-150`,
and the theorem `pipeline_eq` proves the optimized observation equals the baseline observation on the
whole declared domain. The dot equality alone (`block_dot_eq`) is the arithmetic content; the
epilogue enters the observation type, the fold `accumulate`, the narrowing `finalize`, and the four
refusal-branch witnesses below.

The GLSL→SPIR-V translation, dispatch, driver and GPU execution remain trusted/tested boundaries;
the device observation protocol (`tools/gpu/q8_gate.py` pattern) links the executing shader to this
model at grade `tested`. No translator (a semantics for GLSL) is used or required: the certificate
fixes the observation the execution unit must realize and proves the replacement preserves it; the
shader-implements-model link is the tested source-identity + device-parity boundary.

## Measured cost anchor (the registered `before`)

The incumbent register holds the measured 909–932 GMAC/s arithmetic plateau and the 500.7 GMAC/s
real-shape peak at grade `measured`
(`docs/evidence/gpu-prefill/gpu-occupancy-analysis.md` §2–§3; command
`timeout 540 python3 tools/gpu/q8_occupancy.py --controls`, exit 0, 11.39 s, RTX 5080 / Vulkan
driver 2580660800). `beforeCost = 8` is the logical instruction count of that incumbent per 32-lane
block; the GMAC/s figures are the hardware anchor, not part of the logical cost model.

## Witness shapes

The declared domain is exercised by the boundary shapes `shapeTail` (`M`, `tokens` not multiples of
16), `shapeSingleBlock` (`blocks = 1`) and `shapeDeepK` (`blocks = 224`, the measured deep-K
boundary of `gpu-occupancy-analysis.md` §2); the joint witness is `shapeTiny` with an all-ones
block. The real Qwen3.5-0.8B projection geometries (`M = 4096` attn_q, `M = 1024` ffn_down at
tokens ∈ {200, 512}) are the admission population these boundary shapes bracket. The refusal-branch
witnesses below cover the non-finite, i128-overflow and i64-narrowing paths.

-/
import DispatchAdmission
import Apeiron.Trichotomy.AnyRing

set_option autoImplicit false
set_option linter.apeiron.notation false

namespace Int8CoopmatSpirv

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

/-- O(8) executed baseline arithmetic: the incumbent shader's eight scalar packed dots per block. -/
def baselineBlockDot (w x : Fin 8 → UInt32) : Int := ∑ q : Fin 8, packedDot (w q) (x q)

/-- O(32) canonical 32-lane signed dot; the int8 cooperative-matrix observation oracle. -/
def canonicalBlockDot (w x : Fin 8 → UInt32) : Int :=
  ∑ q : Fin 8, ∑ i : Fin 4, sign8 (w q) i * sign8 (x q) i

/-- The replaced operation: the scalar baseline dot equals the canonical MMA arithmetic, per block. -/
theorem block_dot_eq (w x : Fin 8 → UInt32) :
    canonicalBlockDot w x = baselineBlockDot w x := by
  unfold canonicalBlockDot baselineBlockDot
  apply Finset.sum_congr rfl
  intro q _
  exact (packedDot_split (w q) (x q)).symm

/-- O(1): a signed byte is in `[-128, 127]`, so one lane product has magnitude at most `16384`. -/
theorem sign8_bounds (v : UInt32) (i : Fin 4) : -128 ≤ sign8 v i ∧ sign8 v i ≤ 127 := by
  have hlt : byte v i < 256 := by
    unfold byte
    exact Nat.mod_lt _ (by decide)
  unfold sign8
  by_cases h : byte v i < 128
  · simp only [h, if_true]
    omega
  · simp only [h, if_false]
    omega

/-- Each lane of a block: a signed byte is in `[-128, 127]`, so `|byte product| ≤ 16384`. -/
theorem lane_product_abs_le (v w : UInt32) (i : Fin 4) :
    |sign8 v i * sign8 w i| ≤ 16384 := by
  rw [abs_mul]
  have h1 : |sign8 v i| ≤ 128 := by
    rw [abs_le]
    exact ⟨(sign8_bounds v i).1, le_trans (sign8_bounds v i).2 (by norm_num)⟩
  have h2 : |sign8 w i| ≤ 128 := by
    rw [abs_le]
    exact ⟨(sign8_bounds w i).1, le_trans (sign8_bounds w i).2 (by norm_num)⟩
  calc |sign8 v i| * |sign8 w i| ≤ 128 * 128 :=
        mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
    _ = 16384 := by norm_num

/-- Exactness bound (§3.2): a 32-lane signed code dot satisfies `|dot| ≤ 32·128² = 524288 < 2³¹`,
so the `sat=0` signed-int8→signed-int32 MMA computes it without saturation or a cross-block carry. -/
theorem canonical_dot_abs_le (w x : Fin 8 → UInt32) : |canonicalBlockDot w x| ≤ 524288 := by
  unfold canonicalBlockDot
  calc |∑ q : Fin 8, ∑ i : Fin 4, sign8 (w q) i * sign8 (x q) i|
      ≤ ∑ q : Fin 8, |∑ i : Fin 4, sign8 (w q) i * sign8 (x q) i| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ q : Fin 8, ∑ i : Fin 4, |sign8 (w q) i * sign8 (x q) i| := by
        apply Finset.sum_le_sum
        intro q _
        exact Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _q : Fin 8, ∑ _i : Fin 4, (16384 : Int) := by
        apply Finset.sum_le_sum
        intro q _
        apply Finset.sum_le_sum
        intro i _
        exact lane_product_abs_le (w q) (x q) i
    _ = 524288 := by norm_num

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

/-- Joint witness geometry: one row, one token, one Q8 block. -/
def shapeTiny : Shape := { m := 1, tokens := 1, blocks := 1 }

/-- Boundary witness: `M` and `tokens` are not multiples of 16, exercising tail masking. -/
def shapeTail : Shape := { m := 17, tokens := 13, blocks := 1 }

/-- Boundary witness: a single Q8 block (`blocks = 1`). -/
def shapeSingleBlock : Shape := { m := 16, tokens := 16, blocks := 1 }

/-- Boundary witness: the measured deep-K end of the work sweep, `blocks = 224`. -/
def shapeDeepK : Shape := { m := 1024, tokens := 256, blocks := 224 }

/-- The boundary witness shapes all lie in the declared dispatch domain, so the preservation and
improvement legs range over a populated, non-degenerate shape set. -/
theorem witness_shapes_in_domain :
    shapeTiny.InDomain ∧ shapeTail.InDomain ∧ shapeSingleBlock.InDomain ∧
      shapeDeepK.InDomain := by
  decide

/-- One admitted dispatch: a geometry plus the per-block Q8 payload the execution unit consumes.
`wExp b = none` models the shader's `NONFINITE` sentinel; `aExp b = none` models `ZERO_SCALE`.
`mantissa b` is the i32 weight mantissa; the codes are the packed `u32` quads of `q8_tile.comp`. -/
structure Request where
  shape : Shape
  mantissa : Fin shape.blocks → Int
  wExp : Fin shape.blocks → Option Nat
  aExp : Fin shape.blocks → Option Nat
  w : Fin shape.blocks → Fin 8 → UInt32
  x : Fin shape.blocks → Fin 8 → UInt32

/-- Reachability: the request lies in the declared dispatch domain. -/
def reachable (r : Request) : Prop := r.shape.InDomain

/-- Admissibility: no further precondition beyond reachability for a total shader. -/
def admissible (_ : Request) : Prop := True

/-- Refusal flag bits, mirroring `q8_tile.comp:20-22`. -/
def REFUSE_NONFINITE : Nat := 1
def REFUSE_I128 : Nat := 2
def REFUSE_I64 : Nat := 4

/-- O(1) two's-complement reduction of an exact integer into the signed 128-bit range. -/
def wrap128 (z : Int) : Int := (z + 2 ^ 127) % 2 ^ 128 - 2 ^ 127

/-- Signed 128-bit range, the domain `addChecked` accepts. -/
def inI128 (z : Int) : Prop := -(2 ^ 127) ≤ z ∧ z ≤ 2 ^ 127 - 1

/-- Signed 64-bit range, the domain the Q32.32 narrowing writes. -/
def inI64 (z : Int) : Prop := -(2 ^ 63) ≤ z ∧ z ≤ 2 ^ 63 - 1

instance (z : Int) : Decidable (inI128 z) := inferInstanceAs (Decidable (_ ∧ _))
instance (z : Int) : Decidable (inI64 z) := inferInstanceAs (Decidable (_ ∧ _))

/-- O(blocks) two-limb accumulation, modelled from `q8_tile.comp:25-59,117-127`: per block, a
non-finite weight refuses; a zero-scale activation is skipped; otherwise the shifted coefficient
`m·dot·2^shift` is formed as a 128-bit term, added two's-complement, and a signed overflow sets
`REFUSE_I128`. -/
def accumulate (r : Request) (d : Fin r.shape.blocks → Int) : Int × Nat :=
  (List.finRange r.shape.blocks).foldl
    (fun (st : Int × Nat) (b : Fin r.shape.blocks) =>
      match r.wExp b with
      | none => (st.1, st.2 ||| REFUSE_NONFINITE)
      | some we =>
        match r.aExp b with
        | none => st
        | some ae =>
          let term := wrap128 (r.mantissa b * d b * 2 ^ (we + ae))
          let s := st.1 + term
          (wrap128 s, if inI128 s then st.2 else st.2 ||| REFUSE_I128))
    ((0 : Int), (0 : Nat))

/-- The full observation the execution unit realizes: the Q32.32-rounded i64 value (if any) and the
cumulative refusal bitmask. -/
structure Observation where
  value : Option Int
  flags : Nat
deriving DecidableEq, Repr

/-- O(1) Q32.32 round-to-nearest with the source's positive `2^31` bias, then the i64 narrowing and
`REFUSE_I64`, modelled from `q8_tile.comp:137-149`. -/
def finalize (acc : Int) (flags : Nat) : Observation :=
  if flags = 0 then
    let biased := wrap128 (acc + 2 ^ 31)
    let q := biased.ediv (2 ^ 32)
    if inI64 q then { value := some q, flags := 0 }
    else { value := none, flags := REFUSE_I64 }
  else { value := none, flags := flags }

/-- Run the epilogue over a per-block dot function. -/
def run (r : Request) (d : Fin r.shape.blocks → Int) : Observation :=
  let st := accumulate r d
  finalize st.1 st.2

/-- O(blocks): the epilogue depends on the dot only pointwise, so equal dot functions give equal
observations. This is what lets a dot replacement preserve the whole checked epilogue. -/
theorem run_congr (r : Request) {d1 d2 : Fin r.shape.blocks → Int}
    (h : ∀ b, d1 b = d2 b) : run r d1 = run r d2 := by
  unfold run accumulate
  rw [funext h]

/-- Before: the incumbent execution unit's observation (eight scalar packed dots per block). -/
def before (r : Request) : Observation :=
  run r (fun b => baselineBlockDot (r.w b) (r.x b))

/-- After: the int8 cooperative-matrix execution unit's observation (one MMA per block). -/
def after (r : Request) : Observation :=
  run r (fun b => canonicalBlockDot (r.w b) (r.x b))

/-- **The preservation leg**: the optimized observation equals the baseline observation, carrying
the dot and the whole checked i128 epilogue (accumulation, Q32.32 round, i64 narrowing, every
refusal flag), on the declared domain. -/
theorem pipeline_eq (r : Request) : after r = before r := by
  unfold after before
  apply run_congr
  intro b
  exact block_dot_eq (r.w b) (r.x b)

/-- Registered cost model: eight scalar packed-dot instructions per 32-lane block (the incumbent). -/
def beforeCost (_ : Request) : Nat := 8

/-- Registered cost model for the optimized unit: one M16N16K32 MMA per 32-lane block. -/
def afterCost (_ : Request) : Nat := 1

/-- Concrete joint witness: one row, one token, one block of all-ones quads (dot = -32). -/
def witness : Request :=
  { shape := shapeTiny
    mantissa := fun _ => 1
    wExp := fun _ => some 0
    aExp := fun _ => some 0
    w := fun _ _ => 4294967295
    x := fun _ _ => 16843009 }

theorem witness_joint : reachable witness ∧ admissible witness := by
  constructor
  · simp [reachable, witness, shapeTiny, Shape.InDomain]
  · trivial

/-- A unit block whose canonical dot is `1` (one signed lane product, the rest zero). -/
def unitW : Fin 8 → UInt32 := fun q => if q = 0 then 1 else 0
def unitX : Fin 8 → UInt32 := fun q => if q = 0 then 1 else 0

/-- Non-finite-weight refusal witness: `wExp = none` sets `REFUSE_NONFINITE` and refuses the cell. -/
def nonfiniteWitness : Request :=
  { shape := { m := 1, tokens := 1, blocks := 1 }
    mantissa := fun _ => 1
    wExp := fun _ => none
    aExp := fun _ => some 0
    w := fun _ => unitW
    x := fun _ => unitX }

theorem nonfinite_refusal_reachable :
    (after nonfiniteWitness).value = none ∧
      (after nonfiniteWitness).flags = REFUSE_NONFINITE := by
  decide

/-- i128-overflow refusal witness: two `2^126` terms sum to `2^127`, outside `addChecked`'s range. -/
def i128Witness : Request :=
  { shape := { m := 1, tokens := 1, blocks := 2 }
    mantissa := fun _ => 1
    wExp := fun _ => some 126
    aExp := fun _ => some 0
    w := fun _ => unitW
    x := fun _ => unitX }

theorem i128_refusal_reachable :
    (after i128Witness).value = none ∧
      (after i128Witness).flags = REFUSE_I128 := by
  decide

/-- i64-narrowing refusal witness: `acc = 2^95` rounds to `2^63`, outside the i64 range. -/
def i64Witness : Request :=
  { shape := { m := 1, tokens := 1, blocks := 1 }
    mantissa := fun _ => 1
    wExp := fun _ => some 95
    aExp := fun _ => some 0
    w := fun _ => unitW
    x := fun _ => unitX }

theorem i64_refusal_reachable :
    (after i64Witness).value = none ∧
      (after i64Witness).flags = REFUSE_I64 := by
  decide

/-- Success witness: the joint witness rounds to the i64 value `0` with no refusal. -/
theorem witness_success :
    (after witness).value = some 0 ∧ (after witness).flags = 0 := by
  decide

/-- The registered admission: witness, dot-plus-epilogue preservation, strict registered
improvement. -/
def admission : ProofDispatch.Admission before after reachable admissible beforeCost afterCost where
  witness := witness
  joint := witness_joint
  preserves := fun r _ _ => pipeline_eq r
  improves := fun _ _ _ => by simp [beforeCost, afterCost]

end Int8CoopmatSpirv
