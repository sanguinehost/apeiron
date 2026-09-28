import Spirv.Fragment
import Apeiron.Trichotomy.AnyRing
import Mathlib.Tactic

/-!
# Denotation of the v0 integer DP4A fragment, at the canonical face

`eval`/`denote` for the v0 executing opcode set (`executingOpcodes`, `Spirv.Fragment`), the byte
order of `PackedVectorFormat4x8Bit`, the face-parameterized `OpSUDot` denotation, and the
layout-versus-representation verdict. The integer core is not a standalone `ZMod (2^w)`
semantics: it is the face instance of the canonical
`Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` the extracted word kernels already
refine (`word_mul_denote`/`word_norm_denote`, `proof/corpus/WordRefinement.lean`).

## Layout verdict (this task's decisive sub-goal)

**Proved layout theorem — the emitted Q4/Q8 packing realizes the hyperbolic face `σ = +1` in the
regular-representation (product) reading.** The realized reduction is named exactly: for the Q4
even/odd nibble split the pair of `OpSUDot` phase results is, per byte lane `i`,
`(ringMul σ u v).1` at `σ = +1` on the cells `u = (sext8 even, sext8 odd)` (activation) and
`v = (zext8 even, zext8 odd)` (weight) — `sdot_split_ringMul` below. The same integer expression is
`ringForm (-1) u v`, the polar of the elliptic norm form; the verdict binds the
`ringMul`/regular-representation reading named by `docs/epics/gpu-spv.md`'s contract. Root witness at
the word face: `r = s = 1`, `r ^ 2 = σ`, `r * s = 1` (`FaceWitness`, `#guard`); the wrong-σ witness
`r = 1` for `σ = -1` is rejected (`wrong_sigma_root`) and turns this verdict red.

Three limits are recorded with the verdict, not hidden:

1. **Only channel 0 of `ringMul` is computed.** The cross channel `(ringMul σ u v).2 = u.1*v.2 +
   u.2*v.1` has no emitted instruction — both `OpSUDot` phases fold into one accumulator
   (`ACCQVAR`, `simt/dp4a.rs` lines 843–847). **Emitter migration item** (named for the record):
   accumulate the cross `OpSUDot` pair into a second accumulator to realize the cell-valued
   `ringMul` instead of its channel-0 projection.
2. **The `OpSUDot` signed·unsigned asymmetry is absorbed by the unpacking maps, not by the face.**
   The activation cell uses `sext8`, the weight cell `zext8`; there is no self-pairing of one cell,
   so the face's own norm form `ringNormForm σ` is not realized. On weight bytes (`< 0x80` for Q4
   nibbles) `sext8 = zext8` and the diagonal would read `ringNormForm (-1) = v.1 ^ 2 + v.2 ^ 2` —
   the elliptic norm form — which is why the dual `ringForm (-1)` reading is recorded.
3. **The 2-adic caveat stands.** `Z/2^w` ramifies at `2`: `ringTrace` (the regular-representation
   trace pairing `tr(ρ(u)ρ(v)) = 2*(u.1*v.1 + σ*u.2*v.2)`) satisfies `ringTrace σ u v =
   2 * (ringMul σ u v).1`, and the halving is unavailable (`two_not_unit`);
   `hyperbolic_decomposition`'s `2 * h = 1` hypothesis does not transport to `W32`. Its unit
   hypotheses do transport at `σ = +1` (`root_witness`).

The face ledger row of `docs/epics/gpu-spv.md` binds this verdict: the trace form transports as the
ring identity `ringTrace = 2 * channel0`, not as a halving.

## Scope of `denote`

`eval` is the result-value semantics of one v0 executing instruction over a word-addressed flat
store; `denote` folds `step` over a straight-line block. The structured-CFG loop nest, the
`GlobalInvocationId` grid assembly and the `OpAccessChain` in-bounds obligations are the
`Spirv.Dp4a` seam (well-formedness premise: Klimis et al., POPL 2023); the decoder obligations
D1–D4 are `Spirv.Decode`. The float epilogue evaluates to `none`: it is a named migration item,
not a v0 premise.
-/

namespace Spirv

open Apeiron.Trichotomy.AnyRing
open BigOperators

/-- The 32-bit word carrier of the fragment's integers.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
abbrev Word32 := W32

/-- `PackedVectorFormat4x8Bit` component `i` of a packed 32-bit word: the byte `i` counting from
the least-significant byte (SPIR-V 1.3 + `SPV_KHR_integer_dot_product` rev 3: component 0 is the
least-significant byte).

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def byte (w : Word32) (i : Fin 4) : Word32 := (((w.val / 256 ^ i.val) % 256 : Nat) : Word32)

/-- Byte order: component `i` is the byte at shift `8*i` from the least-significant end.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem byte_order (w : Word32) (i : Fin 4) :
    byte w i = (((w.val >>> (8 * i.val)) % 256 : Nat) : Word32) := by
  have h : (256 : Nat) ^ i.val = 2 ^ (8 * i.val) := by
    rw [show (256 : Nat) = 2 ^ 8 from rfl, Nat.pow_mul]
  simp only [byte, h, Nat.shiftRight_eq_div_pow]

/-- Zero-extension of a packed byte component to `Word32` (`OpSUDot`'s unsigned operand).

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def zext8 (b : Word32) : Word32 := b

/-- Sign-extension of a packed byte component to `Word32` (`OpSUDot`'s signed operand): values
`≥ 0x80` map to the wrapped negative word.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def sext8 (b : Word32) : Word32 := if b.val < 128 then b else b - 256

/-- `OpSUDot` at `PackedVectorFormat4x8Bit`: the four byte components of the packed words, first
operand signed and second unsigned, summed in `W32` (`SPV_KHR_integer_dot_product` rev 3).

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def evalSUDot (a b : Word32) : Word32 :=
  ∑ i : Fin 4,
    sext8 (((a.val >>> (8 * i.val)) % 256 : Nat) : Word32) *
      zext8 (((b.val >>> (8 * i.val)) % 256 : Nat) : Word32)

/-- The graded `OpSUDot` denotation: `OpSUDot` equals `∑ sext8 (byte i a) * zext8 (byte i b)` in
`ZMod (2 ^ 32)`. The proof goes through `byte_order`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem sdot_denote (a b : Word32) :
    evalSUDot a b = ∑ i : Fin 4, sext8 (byte a i) * zext8 (byte b i) := by
  have h : (256 : Nat) = 2 ^ 8 := rfl
  unfold evalSUDot
  simp only [byte, h, Nat.pow_mul, Nat.shiftRight_eq_div_pow]

/-- The dynamic value of a v0 value instruction: 32-bit words, booleans and `uvec3` words.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
inductive DVal where
  /-- `OpTypeInt 32`; signed and unsigned share the `ZMod (2 ^ 32)` carrier.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
  | word (z : Word32)
  /-- `OpTypeBool`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
  | bool (b : Bool)
  /-- `OpTypeVector <int 32> 3` (`GlobalInvocationId`).

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
  | vec3 (v : Fin 3 → Word32)

/-- The result value of one v0 executing instruction over a word-addressed flat store.
Instructions producing no result (`OpStore`, control), the framing syntax and the float epilogue
surface evaluate to `none`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def eval (env : SpvId → Option DVal) (μ : Nat → Word32) : Inst → Option DVal
  | .iadd _ _ a b => do
      let .word x ← env a | none
      let .word y ← env b | none
      some (.word (x + y))
  | .imul _ _ a b => do
      let .word x ← env a | none
      let .word y ← env b | none
      some (.word (x * y))
  | .bitAnd _ _ a b => do
      let .word x ← env a | none
      let .word y ← env b | none
      some (.word (((x.val &&& y.val) % 2 ^ 32 : Nat) : Word32))
  | .shiftRightLogical _ _ a s => do
      let .word x ← env a | none
      let .word k ← env s | none
      some (.word (((x.val >>> k.val) % 2 ^ 32 : Nat) : Word32))
  | .ult _ _ a b => do
      let .word x ← env a | none
      let .word y ← env b | none
      some (.bool (x.val < y.val))
  | .select _ _ c a b => do
      let .bool t ← env c | none
      let .word x ← env a | none
      let .word y ← env b | none
      some (.word (if t then x else y))
  | .sudot _ _ a b _ => do
      let .word x ← env a | none
      let .word y ← env b | none
      some (.word (evalSUDot x y))
  | .load _ _ p => do
      let .word a ← env p | none
      some (.word (μ a.val))
  | _ => none

/-- One instruction step: evaluate and bind the result id. `OpStore` writes the flat store.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def step (env : SpvId → Option DVal) (μ : Nat → Word32) :
    Inst → Option ((SpvId → Option DVal) × (Nat → Word32))
  | .store p v => do
      let .word a ← env p | none
      let .word x ← env v | none
      some (env, fun n => if n = a.val then x else μ n)
  | i => do
      let v ← eval env μ i
      match i with
      | .iadd _ r .. | .imul _ r .. | .bitAnd _ r .. | .shiftRightLogical _ r ..
      | .ult _ r .. | .select _ r .. | .sudot _ r .. | .load _ r _ =>
        some (fun j => if j = r then some v else env j, μ)
      | _ => none

/-- The value denotation of a straight-line block: the fold of `step` in list order. The
module-level invocation semantics (loop nest, grid assembly) is the `Spirv.Dp4a` seam.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def denote (code : List Inst) (env : SpvId → Option DVal) (μ : Nat → Word32) :
    Option ((SpvId → Option DVal) × (Nat → Word32)) :=
  code.foldl (fun acc i => acc.bind fun p => step p.1 p.2 i) (some (env, μ))

/-- `OpSUDot` through `eval`: the instruction denotes `evalSUDot`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem eval_sudot (env : SpvId → Option DVal) (μ : Nat → Word32) {ty r : SpvId}
    {a b : SpvId} {fmt : PackedFormat} {x y : Word32}
    (ha : env a = some (.word x)) (hb : env b = some (.word y)) :
    eval env μ (.sudot ty r a b fmt) = some (.word (evalSUDot x y)) := by
  simp [eval, ha, hb]

/-! ## The integer core as a face instance of the canonical object -/

/-- The realized face: hyperbolic (`σ = +1`), the coefficient the emitted split pairing forces.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def realizedSigma : Word32 := 1

/-- The root witness `r` of the realized face: `r ^ 2 = σ`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def realizedRoot : Word32 := 1

/-- The co-root witness `s` of the realized face: `r * s = 1`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def realizedCoRoot : Word32 := 1

/-- The word ring embeds in the canonical cell ring at any face: the diagonal channel of `ringMul`
is the ring product. This is the face instance linking `OpIMul`'s `W32` arithmetic to the canonical
`ringMul`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem ringMul_diag (σ x y : Word32) : ringMul σ (x, 0) (y, 0) = (x * y, 0) := by
  simp

/-- `OpIMul` denotes the canonical `ringMul` at the realized face on the diagonal channel.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem evalIMul_face (x y : Word32) :
    x * y = (ringMul realizedSigma (x, 0) (y, 0)).1 := by
  simp [realizedSigma]

/-- A face witness at the word face: the unit hypotheses of
`Apeiron/Trichotomy/.../hyperbolic_decomposition` (`r ^ 2 = σ`, `r * s = 1`).

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def FaceWitness (σ r s : Word32) : Bool := decide (r ^ 2 = σ) && decide (r * s = 1)

-- The layout verdict carries a checked root witness at `σ = +1`.
#guard FaceWitness realizedSigma realizedRoot realizedCoRoot

/-- The witness is recorded at grade `proved`: `r = s = 1` witnesses `σ = +1` at `W32`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem root_witness :
    realizedRoot ^ 2 = realizedSigma ∧ realizedRoot * realizedCoRoot = 1 := by
  decide

/-- The wrong-σ control: `r = 1` does not witness `σ = -1`. A verdict carrying that pair is red.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem wrong_sigma_root : ¬ FaceWitness (-1 : Word32) realizedRoot realizedCoRoot = true := by
  decide

/-- Channel 0 of the canonical `ringMul` at the realized face.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem ringMul_channel0 (u v : RingCell Word32) :
    (ringMul realizedSigma u v).1 = u.1 * v.1 + u.2 * v.2 := by
  simp [realizedSigma]

/-- The dual reading: the realized scalar is also `ringForm (-1)`, the polar of the elliptic norm
form. Recorded so the verdict's σ is not over-claimed.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem channel0_is_ringForm (u v : RingCell Word32) :
    (ringMul realizedSigma u v).1 = ringForm (-1) u v := by
  simp [ringForm, realizedSigma]

/-- The regular-representation trace pairing `tr(ρ(u)ρ(v))` for
`ρ(z) = [[z.1, σ*z.2],[z.2, z.1]]`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def ringTrace (σ : Word32) (u v : RingCell Word32) : Word32 := 2 * (u.1 * v.1 + σ * u.2 * v.2)

/-- The trace form transports as a ring identity: `ringTrace` is `2 * channel0`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem ringTrace_two_channel0 (u v : RingCell Word32) :
    ringTrace realizedSigma u v = 2 * (ringMul realizedSigma u v).1 := by
  simp [ringTrace, realizedSigma]

/-- The 2-adic caveat: `2` is not a unit at `W32`, so the halving `channel0 = ringTrace / 2` and
`hyperbolic_decomposition`'s `2 * h = 1` hypothesis do not transport to the word face.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem two_not_unit : ¬ IsUnit ((2 : Nat) : Word32) := by
  rw [ZMod.isUnit_iff_coprime]
  decide

/-- The per-lane cell pair at byte lane `i`: activation cell (signed unpacking, even/odd phases)
and weight cell (unsigned unpacking, even/odd nibbles).

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def actCell (x0 x1 : Word32) (i : Fin 4) : RingCell Word32 :=
  (sext8 (byte x0 i), sext8 (byte x1 i))

/-- The weight-side cell at byte lane `i`.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
def wgtCell (w0 w1 : Word32) (i : Fin 4) : RingCell Word32 :=
  (zext8 (byte w0 i), zext8 (byte w1 i))

/-- **The layout theorem.** The Q4 even/odd split realizes channel 0 of the canonical `ringMul` at
`σ = +1`: the two `OpSUDot` phase results sum to `(ringMul σ u v).1` on the even/odd cells, per
byte lane. This is the face the emitted packing realizes.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem sdot_split_ringMul (x0 x1 w0 w1 : Word32) :
    evalSUDot x0 w0 + evalSUDot x1 w1
      = ∑ i : Fin 4, (ringMul realizedSigma (actCell x0 x1 i) (wgtCell w0 w1 i)).1 := by
  simp [evalSUDot, actCell, wgtCell, byte_order, realizedSigma, Finset.sum_add_distrib]

/-- The realized reduction, named: the sum of the two phase dots is `ringForm (-1)` on the cells.

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem sdot_split_realized (x0 x1 w0 w1 : Word32) :
    evalSUDot x0 w0 + evalSUDot x1 w1
      = ∑ i : Fin 4, ringForm (-1) (actCell x0 x1 i) (wgtCell w0 w1 i) := by
  rw [sdot_split_ringMul]
  exact Finset.sum_congr rfl fun i _ => by simp [ringForm, realizedSigma]

/-- The cross channel is not realized: the emitter computes channel 0 only. This is the named
emitter migration item (a second accumulator for the cross `OpSUDot` pair).

`→` [takes the left and gives the right; between statements, if the left holds so does the right]
`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`∧` [both are true]
`¬` [this is not true]
`List` [a finite sequence of things, in order]
`Bool` [true or false, nothing else]
`RingCell` [a point: two ring elements side by side, a + b·x]
`ringMul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`IsUnit` [can be divided by, meaning something multiplies with it to give 1] -/
theorem cross_channel_not_realized (u v : RingCell Word32) :
    (ringMul realizedSigma u v).2 = u.1 * v.2 + u.2 * v.1 := by
  simp

end Spirv
