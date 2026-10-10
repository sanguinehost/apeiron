/-
Second half of the SHA-256 proof-of-work cost comparison, namespace `algebra_cost_pow_compare`
(first half: `Icarus.Cost.AlgebraCostPowComparePart1`). This part carries sections 7 to 9: the
twenty-three-field `ShaCost` measure with the padding, loader and store costs, the complete
attempt and range figures, the source-bound corrections, the three faces, and the audit rows.

Depends on: `Icarus.Cost.AlgebraCostPowComparePart1`. Every declaration here is stated over
sections 0 to 6 of Part1 and no declaration there mentions a `ShaCost`, so the cut duplicates
no axiom, instance or binder. The corpus and extracted-side dependencies are those of Part1:
`Apeiron.Trichotomy.AnyRing`, `Apeiron.Trichotomy.ChannelsAnyRing`, `Crypto.Sha2`,
`Crypto.Sha256Certificate` and `WordRefinement`.
Excludes: as Part1. No latency, wall-clock time, cycle, joule, watt, hash-rate, USD or
profitability claim follows from any number in this file.
Grade: as Part1. The split changes no declaration name, namespace or type, and section numbers
are unchanged.
-/

import Icarus.Cost.AlgebraCostPowComparePart1
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sha2
import Crypto.Sha256Certificate
import WordRefinement
import Mathlib.Tactic

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.unusedVariables false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
set_option linter.unnecessarySeqFocus false

universe u

namespace algebra_cost_pow_compare

open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
open apeiron_word

variable {R : Type u} [CommRing R]

/-! ## 7. The complete attempt and range in AR-27's twenty-three-field measure

This is the second measure, kept separate by construction: AR-27's `ShaCost σ` adds the padding's
byte writes, the twelve byte reads and `Nat` operations of the block load, the schedule word
writes, and the digest store's bytes and word reads. **Bytes and operations are different units
and are never summed together** (`ShaCost.eq_of_fields` is a twenty-three-hypothesis theorem, and
no theorem below adds a byte field to an operation field). The first eleven fields are §6's
measure, projected back by `ShaCost.comp`. -/

/-- `ShaCost σ`: AR-27's measure. The first eleven fields are AR-15's `RoundCost σ` in AR-15's
order; the remaining twelve count padding writes, load work and the digest store. -/
structure ShaCost (σ : R) where
  /-- scalar ring multiplications -/
  mul : Nat
  /-- scalar ring additions -/
  add : Nat
  /-- scalar ring subtractions -/
  sub : Nat
  /-- scalar ring negations -/
  neg : Nat
  /-- conversions of a dial value into the ring -/
  cast : Nat
  /-- channel reads -/
  channel : Nat
  /-- bitwise XOR of `Word`s -/
  bitXor : Nat
  /-- bitwise AND of `Word`s -/
  bitAnd : Nat
  /-- bitwise NOT of a `Word` -/
  bitNot : Nat
  /-- logical shifts of a `Word` -/
  shift : Nat
  /-- bitwise OR of `Word`s -/
  bitOr : Nat
  /-- bytes appended by `pad`, message copy included -/
  padByte : Nat
  /-- of those, the zero-fill bytes -/
  zeroByte : Nat
  /-- of those, the eight length-field bytes -/
  lenByte : Nat
  /-- `Nat` shifts extracting the eight length bytes -/
  padShift : Nat
  /-- the eight `% 256` truncations of the length bytes -/
  padMod : Nat
  /-- byte reads of the padded block by the sixteen-word loader -/
  blockByteRead : Nat
  /-- `Nat` multiplications in the big-endian regrouping -/
  regroupMul : Nat
  /-- `Nat` additions in the big-endian regrouping -/
  regroupAdd : Nat
  /-- 32-bit truncations in the big-endian regrouping -/
  regroupTrunc : Nat
  /-- word writes into the 64-word schedule -/
  schedWordWrite : Nat
  /-- bytes written by the big-endian digest store -/
  digestByte : Nat
  /-- chaining-word reads performed by the digest store -/
  digestWordRead : Nat
  deriving Repr

/-- The record of an attempt that performs no operation and moves no byte. -/
def ShaCost.zero (σ : R) : ShaCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
    padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
    blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
    schedWordWrite := 0, digestByte := 0, digestWordRead := 0 }

/-- **Dependent composition over all twenty-three fields.** Temporal and spatial O(1): a closed
term of twenty-three `Nat` additions. -/
def ShaCost.composed (σ : R) (a b : ShaCost σ) : ShaCost σ :=
  { mul := a.mul + b.mul, add := a.add + b.add, sub := a.sub + b.sub, neg := a.neg + b.neg,
    cast := a.cast + b.cast, channel := a.channel + b.channel, bitXor := a.bitXor + b.bitXor,
    bitAnd := a.bitAnd + b.bitAnd, bitNot := a.bitNot + b.bitNot, shift := a.shift + b.shift,
    bitOr := a.bitOr + b.bitOr,
    padByte := a.padByte + b.padByte, zeroByte := a.zeroByte + b.zeroByte,
    lenByte := a.lenByte + b.lenByte, padShift := a.padShift + b.padShift,
    padMod := a.padMod + b.padMod, blockByteRead := a.blockByteRead + b.blockByteRead,
    regroupMul := a.regroupMul + b.regroupMul, regroupAdd := a.regroupAdd + b.regroupAdd,
    regroupTrunc := a.regroupTrunc + b.regroupTrunc,
    schedWordWrite := a.schedWordWrite + b.schedWordWrite,
    digestByte := a.digestByte + b.digestByte, digestWordRead := a.digestWordRead + b.digestWordRead }

instance instAddShaCost (σ : R) : Add (ShaCost σ) := ⟨ShaCost.composed σ⟩

@[simp] theorem ShaCost.composed_apply (σ : R) (a b : ShaCost σ) :
    a + b = ShaCost.composed σ a b := rfl

/-- **Equality of counts is fieldwise equality**, twenty-three fields. -/
theorem ShaCost.eq_of_fields {σ : R} {a b : ShaCost σ}
    (h1 : a.mul = b.mul) (h2 : a.add = b.add) (h3 : a.sub = b.sub) (h4 : a.neg = b.neg)
    (h5 : a.cast = b.cast) (h6 : a.channel = b.channel) (h7 : a.bitXor = b.bitXor)
    (h8 : a.bitAnd = b.bitAnd) (h9 : a.bitNot = b.bitNot) (h10 : a.shift = b.shift)
    (h11 : a.bitOr = b.bitOr) (h12 : a.padByte = b.padByte) (h13 : a.zeroByte = b.zeroByte)
    (h14 : a.lenByte = b.lenByte) (h15 : a.padShift = b.padShift) (h16 : a.padMod = b.padMod)
    (h17 : a.blockByteRead = b.blockByteRead) (h18 : a.regroupMul = b.regroupMul)
    (h19 : a.regroupAdd = b.regroupAdd) (h20 : a.regroupTrunc = b.regroupTrunc)
    (h21 : a.schedWordWrite = b.schedWordWrite) (h22 : a.digestByte = b.digestByte)
    (h23 : a.digestWordRead = b.digestWordRead) : a = b := by
  cases a
  cases b
  cases h1; cases h2; cases h3; cases h4; cases h5; cases h6; cases h7; cases h8; cases h9
  cases h10; cases h11; cases h12; cases h13; cases h14; cases h15; cases h16; cases h17
  cases h18; cases h19; cases h20; cases h21; cases h22; cases h23
  rfl

/-- **Composition is commutative.** -/
theorem ShaCost.composed_comm (σ : R) (a b : ShaCost σ) : a + b = b + a := by
  cases a
  cases b
  simp [ShaCost.composed, Nat.add_comm]

/-- **Composition is associative**. -/
theorem ShaCost.composed_assoc (σ : R) (a b c : ShaCost σ) : a + b + c = a + (b + c) := by
  cases a
  cases b
  cases c
  simp [ShaCost.composed, Nat.add_assoc]

/-- **Composition with the zero record is neutral.** -/
theorem ShaCost.composed_zero (σ : R) (a : ShaCost σ) : a + ShaCost.zero σ = a := by
  cases a
  simp [ShaCost.composed, ShaCost.zero]

/-- **`rep σ n a` is the measure of `n` dependent copies of `a`.** Temporal O(1) and spatial O(1):
twenty-three `Nat` multiplications on a twenty-three-field record. Derivation: fieldwise
`n * a.field`, and `ShaCost.rep_add` proves this image additive. -/
def ShaCost.rep (σ : R) (n : Nat) (a : ShaCost σ) : ShaCost σ :=
  { mul := n * a.mul, add := n * a.add, sub := n * a.sub, neg := n * a.neg, cast := n * a.cast,
    channel := n * a.channel, bitXor := n * a.bitXor, bitAnd := n * a.bitAnd,
    bitNot := n * a.bitNot, shift := n * a.shift, bitOr := n * a.bitOr,
    padByte := n * a.padByte, zeroByte := n * a.zeroByte, lenByte := n * a.lenByte,
    padShift := n * a.padShift, padMod := n * a.padMod,
    blockByteRead := n * a.blockByteRead, regroupMul := n * a.regroupMul,
    regroupAdd := n * a.regroupAdd, regroupTrunc := n * a.regroupTrunc,
    schedWordWrite := n * a.schedWordWrite, digestByte := n * a.digestByte,
    digestWordRead := n * a.digestWordRead }

/-- **Base case.** -/
@[simp] theorem ShaCost.rep_zero (σ : R) (a : ShaCost σ) :
    ShaCost.rep σ 0 a = ShaCost.zero σ := by
  simp [ShaCost.rep, ShaCost.zero]

/-- **Recurrence.** -/
theorem ShaCost.rep_succ (σ : R) (n : Nat) (a : ShaCost σ) :
    ShaCost.rep σ (n + 1) a = a + ShaCost.rep σ n a := by
  cases a
  simp [ShaCost.rep, ShaCost.composed, Nat.succ_mul, Nat.add_comm]

/-- **THE COMPOSITION LAW over an iterated construction, twenty-three fields.** -/
theorem ShaCost.rep_add (σ : R) (n m : Nat) (a : ShaCost σ) :
    ShaCost.rep σ (n + m) a = ShaCost.rep σ n a + ShaCost.rep σ m a := by
  cases a
  simp [ShaCost.rep, ShaCost.composed, Nat.add_mul]

/-- **The compression projection.** The first eleven fields read as AR-15's `RoundCost σ`, so
every `ShaCost` figure yields a §6 figure by forgetting the twelve new fields. Temporal and
spatial O(1): eleven projections. -/
def ShaCost.comp {σ : R} (c : ShaCost σ) : RoundCost σ :=
  { mul := c.mul, add := c.add, sub := c.sub, neg := c.neg, cast := c.cast, channel := c.channel,
    bitXor := c.bitXor, bitAnd := c.bitAnd, bitNot := c.bitNot, shift := c.shift, bitOr := c.bitOr }

/-- **The embedding.** A pure compression cost enters `ShaCost` with all twelve new fields zero.
-/
def ShaCost.ofRound {σ : R} (c : RoundCost σ) : ShaCost σ :=
  { mul := c.mul, add := c.add, sub := c.sub, neg := c.neg, cast := c.cast, channel := c.channel,
    bitXor := c.bitXor, bitAnd := c.bitAnd, bitNot := c.bitNot, shift := c.shift, bitOr := c.bitOr,
    padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
    blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
    schedWordWrite := 0, digestByte := 0, digestWordRead := 0 }

/-- The embedding is the identity on the eleven compression fields. -/
theorem ShaCost.comp_ofRound (σ : R) (c : RoundCost σ) : (ShaCost.ofRound c).comp = c := rfl

/-- The projection commutes with composition. -/
theorem comp_add (σ : R) (a b : ShaCost σ) : (a + b).comp = a.comp + b.comp := by
  cases a
  cases b
  rfl

/-- The projection commutes with the iteration operator. -/
theorem comp_rep (σ : R) (n : Nat) (a : ShaCost σ) :
    (ShaCost.rep σ n a).comp = RoundCost.rep σ n a.comp := by
  cases a
  rfl

/-! ### 7.1 The padding cost

`pad` (`Sha2.lean:133-136`) appends four pieces to the message: the message itself, one `0x80`
merge byte, `padZeroFill n` zero bytes, and the eight bytes of `lengthBlock n`. The padding cost
prices them as bytes: `padByte` (the padded length), `zeroByte`, `lenByte`, plus the eight shifts
and eight moduli of the length bytes (`lengthByte_body_ops`). -/

/-- **The cost of padding a message of `n` bytes.** Temporal O(n) - one append of `n` bytes plus
one `List.replicate` plus eight length bytes; spatial O(n). Counted in: the five padding fields.
Derivation: `pad_spec` displays the four pieces; `pad_length` ties `padByte` to the canonical
body. -/
def padCost (σ : R) (n : Nat) : ShaCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
    padByte := padLength n, zeroByte := padZeroFill n, lenByte := 8, padShift := 8, padMod := 8,
    blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
    schedWordWrite := 0, digestByte := 0, digestWordRead := 0 }

/-- **The padding's `padByte` is the canonical padded length**, not a bare literal. -/
theorem padCost_padByte (σ : R) (m : List (BitVec 8)) :
    (padCost σ m.length).padByte = (pad m).length := by
  rw [padCost, pad_length]

/-- **The padding of an 80-byte header**: 128 bytes, 39 of them zero fill, 8 length bytes, 8 shifts
and 8 moduli. -/
theorem padCost_80 (σ : R) :
    padCost σ 80 =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
        padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
        blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
        schedWordWrite := 0, digestByte := 0, digestWordRead := 0 } := by
  rw [padCost]
  apply ShaCost.eq_of_fields <;> simp [padLength_80, padZeroFill_80]

/-- **The padding of the 32-byte digest**: 64 bytes, 23 of them zero fill, 8 length bytes, 8 shifts
and 8 moduli. -/
theorem padCost_32 (σ : R) :
    padCost σ 32 =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
        padByte := 64, zeroByte := 23, lenByte := 8, padShift := 8, padMod := 8,
        blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
        schedWordWrite := 0, digestByte := 0, digestWordRead := 0 } := by
  rw [padCost]
  apply ShaCost.eq_of_fields <;> simp [padLength_32, padZeroFill_32]

/-! ### 7.2 The block load and the digest store

The canonical loader `paddedBlock` is `private` (`Sha2.lean:139-142`, AR-27's probe); the corpus's
public spelling of the same read is `apeiron_word.bytesToWords`, whose per-word body is
`beWord16`. The store exists only in the extracted body `sha256.sha256_loop2.body`; its occurrence
inventory is `model` (AR-27 residual 1). -/

/-- **One byte read.** `b64 blk j` is a single `getD` of byte `j` of the block. -/
theorem b64_body (blk : Array Std.U8 64#usize) (j : Nat) :
    b64 blk j = (blk.val.getD j (0#u8)).bv := rfl

/-- **The four-byte regrouping body.** `beWord16 blk i` is the big-endian regrouping of bytes
`4i … 4i+3`: four byte reads, three `Nat` multiplications by `2^24`, `2^16`, `2^8`, three `Nat`
additions and one `BitVec.ofNat 32` truncation. -/
theorem beWord16_body (blk : Array Std.U8 64#usize) (i : Fin 16) :
    beWord16 blk i
      = BitVec.ofNat 32
          ((b64 blk (4 * i.val)).toNat * 16777216 + (b64 blk (4 * i.val + 1)).toNat * 65536
            + (b64 blk (4 * i.val + 2)).toNat * 256 + (b64 blk (4 * i.val + 3)).toNat) := rfl

/-- **THE PER-WORD COST OF THE BIG-ENDIAN REGROUPING**: four byte reads, three `Nat`
multiplications, three `Nat` additions, one 32-bit truncation, one word written into the schedule.
The four `Array.index_usize` reads and the single `Array.update` of the extracted body are `model`
(AR-27 residual 1). -/
def regroupCost (σ : R) : ShaCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
    padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
    blockByteRead := 4, regroupMul := 3, regroupAdd := 3, regroupTrunc := 1,
    schedWordWrite := 1, digestByte := 0, digestWordRead := 0 }

/-- **THE PER-BLOCK COST OF THE LOAD.** Sixteen words, each one `regroupCost`: 64 byte reads, 48
`Nat` multiplications, 48 `Nat` additions, 16 truncations, 16 schedule writes. -/
def loadCost (σ : R) : ShaCost σ := ShaCost.rep σ 16 (regroupCost σ)

/-- The loader's closed count. -/
theorem loadCost_fields (σ : R) :
    loadCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
        padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
        blockByteRead := 64, regroupMul := 48, regroupAdd := 48, regroupTrunc := 16,
        schedWordWrite := 16, digestByte := 0, digestWordRead := 0 } := by
  rw [loadCost, regroupCost, ShaCost.rep]
  apply ShaCost.eq_of_fields <;> norm_num

/-- **The loader adds nothing to the eleven compression fields.** -/
theorem loadCost_comp_zero (σ : R) : (loadCost σ).comp = RoundCost.zero σ := by
  rw [loadCost_fields]
  rfl

/-- **THE STORE'S PER-WORD COST**: one chaining-word read, four bytes written. The occurrence
inventory is `model`; the four-byte and eight-word shapes are the types of the extracted body. -/
def storeWordCost (σ : R) : ShaCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
    padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
    blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
    schedWordWrite := 0, digestByte := 4, digestWordRead := 1 }

/-- **THE PER-HASH COST OF THE DIGEST STORE**: eight words, thirty-two bytes. A fixed eight-
iteration loop, temporal and spatial O(1) per digest. -/
def storeCost (σ : R) : ShaCost σ := ShaCost.rep σ 8 (storeWordCost σ)

theorem storeCost_fields (σ : R) :
    storeCost σ =
      { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
        bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
        padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
        blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
        schedWordWrite := 0, digestByte := 32, digestWordRead := 8 } := by
  rw [storeCost, storeWordCost, ShaCost.rep]
  apply ShaCost.eq_of_fields <;> norm_num

/-! ### 7.3 The complete SHA-256d attempt and range -/

/-- **One block, compression and load together.** -/
def blockFullCost (σ : R) : ShaCost σ := ShaCost.ofRound (blockCost σ) + loadCost σ

/-- The completed block's closed count, AR-27's `blockFullCost_eq`. -/
theorem blockFullCost_eq (σ : R) :
    blockFullCost σ =
      { mul := 3456, add := 2376, sub := 0, neg := 0, cast := 0, channel := 576,
        bitXor := 704, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576,
        padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
        blockByteRead := 64, regroupMul := 48, regroupAdd := 48, regroupTrunc := 16,
        schedWordWrite := 16, digestByte := 0, digestWordRead := 0 } := by
  rw [blockFullCost, blockCost_eq, loadCost_fields, ShaCost.ofRound, ShaCost.composed_apply,
    ShaCost.composed]
  apply ShaCost.eq_of_fields <;> norm_num

/-- **THE COMPLETE INNER HASH OF THE 80-BYTE HEADER** in AR-27's measure: the header padding,
two compressed-and-loaded blocks, one digest store. AR-27's `attemptCost_eq`. -/
def innerCostSha (σ : R) : ShaCost σ :=
  padCost σ 80 + ShaCost.rep σ 2 (blockFullCost σ) + storeCost σ

/-- **The inner hash's closed count.** -/
theorem innerCostSha_eq (σ : R) :
    innerCostSha σ =
      { mul := 6912, add := 4752, sub := 0, neg := 0, cast := 0, channel := 1152,
        bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152,
        padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
        blockByteRead := 128, regroupMul := 96, regroupAdd := 96, regroupTrunc := 32,
        schedWordWrite := 32, digestByte := 32, digestWordRead := 8 } := by
  have h0 := padCost_80 σ
  have h1 := blockFullCost_eq σ
  have h2 := storeCost_fields σ
  simp only [innerCostSha, ShaCost.rep, ShaCost.composed_apply, ShaCost.composed]
  simp [h0, h1, h2]

/-- **THE COMPLETE OUTER HASH OF THE 32-BYTE DIGEST**: the digest padding, one
compressed-and-loaded block, one digest store. The outer input is one 64-byte padded block
(`digest_pad_blocks`), so exactly one compression. -/
def outerCostSha (σ : R) : ShaCost σ :=
  padCost σ 32 + ShaCost.rep σ 1 (blockFullCost σ) + storeCost σ

/-- **The outer hash's closed count.** -/
theorem outerCostSha_eq (σ : R) :
    outerCostSha σ =
      { mul := 3456, add := 2376, sub := 0, neg := 0, cast := 0, channel := 576,
        bitXor := 704, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576,
        padByte := 64, zeroByte := 23, lenByte := 8, padShift := 8, padMod := 8,
        blockByteRead := 64, regroupMul := 48, regroupAdd := 48, regroupTrunc := 16,
        schedWordWrite := 16, digestByte := 32, digestWordRead := 8 } := by
  have h0 := padCost_32 σ
  have h1 := blockFullCost_eq σ
  have h2 := storeCost_fields σ
  simp only [outerCostSha, ShaCost.rep, ShaCost.composed_apply, ShaCost.composed]
  simp [h0, h1, h2]

/-- **THE COMPLETE SHA-256d ATTEMPT in AR-27's measure**: inner plus outer, everything paid every
attempt. Temporal and spatial O(1): a closed term. Derivation: two inner blocks and one outer
block (`header_pad_blocks`, `digest_pad_blocks`), two paddings, two stores. -/
def sha256dAttemptSha (σ : R) : ShaCost σ := innerCostSha σ + outerCostSha σ

/-- **THE COMPLETE ATTEMPT FIGURE.** The eleven compression fields are §6's
`⟨10368, 7128, 0, 0, 0, 1728, 2112, 960, 192, 3744, 1728⟩`; the twelve byte and load fields are
192 padded bytes (62 zero fill, 16 length bytes, 16 shifts, 16 moduli), 192 block byte reads, 144
regrouping multiplications, 144 regrouping additions, 48 truncations, 48 schedule word writes, 64
digest bytes and 16 chaining-word reads. -/
theorem sha256dAttemptSha_eq (σ : R) :
    sha256dAttemptSha σ =
      { mul := 10368, add := 7128, sub := 0, neg := 0, cast := 0, channel := 1728,
        bitXor := 2112, bitAnd := 960, bitNot := 192, shift := 3744, bitOr := 1728,
        padByte := 192, zeroByte := 62, lenByte := 16, padShift := 16, padMod := 16,
        blockByteRead := 192, regroupMul := 144, regroupAdd := 144, regroupTrunc := 48,
        schedWordWrite := 48, digestByte := 64, digestWordRead := 16 } := by
  have h1 := innerCostSha_eq σ
  have h2 := outerCostSha_eq σ
  simp only [sha256dAttemptSha, ShaCost.composed_apply, ShaCost.composed]
  simp [h1, h2]

/-- **The eleven compression fields of the complete attempt are exactly §6's attempt figure.**
This is the statement that the byte and load fields changed no operation count. -/
theorem sha256dAttemptSha_comp_eq (σ : R) :
    (sha256dAttemptSha σ).comp =
      { mul := 10368, add := 7128, sub := 0, neg := 0, cast := 0, channel := 1728,
        bitXor := 2112, bitAnd := 960, bitNot := 192, shift := 3744, bitOr := 1728 } := by
  rw [sha256dAttemptSha_eq]
  rfl

/-- **The complete attempt's ring layer**, AR-10's six-field record
`(10368, 7128, 0, 0, 0, 1728)`. -/
theorem sha256dAttemptSha_ringFields (σ : R) :
    (sha256dAttemptSha σ).comp.ringFields = ⟨10368, 7128, 0, 0, 0, 1728⟩ := by
  rw [sha256dAttemptSha_comp_eq]
  rfl

/-- **The cost of a nonce range of `m` complete attempts**, every attempt paying its paddings,
three blocks and two stores. -/
def sha256dRangeSha (σ : R) (m : Nat) : ShaCost σ := ShaCost.rep σ m (sha256dAttemptSha σ)

/-- **The complete range's closed form at an arbitrary range length.** -/
theorem sha256dRangeSha_eq (σ : R) (m : Nat) :
    sha256dRangeSha σ m =
      { mul := m * 10368, add := m * 7128, sub := 0, neg := 0, cast := 0,
        channel := m * 1728, bitXor := m * 2112, bitAnd := m * 960, bitNot := m * 192,
        shift := m * 3744, bitOr := m * 1728,
        padByte := m * 192, zeroByte := m * 62, lenByte := m * 16, padShift := m * 16,
        padMod := m * 16, blockByteRead := m * 192, regroupMul := m * 144,
        regroupAdd := m * 144, regroupTrunc := m * 48, schedWordWrite := m * 48,
        digestByte := m * 64, digestWordRead := m * 16 } := by
  rw [sha256dRangeSha, sha256dAttemptSha_eq, ShaCost.rep]
  apply ShaCost.eq_of_fields <;> norm_num

/-- **THE RECURRENCE IN THE COMPLETE RANGE.** -/
theorem sha256dRangeSha_succ (σ : R) (n : Nat) :
    sha256dRangeSha σ (n + 1) = sha256dAttemptSha σ + sha256dRangeSha σ n := by
  unfold sha256dRangeSha
  rw [ShaCost.rep_succ]

/-- **Additivity of the complete range, twenty-three fields.** -/
theorem sha256dRangeSha_add (σ : R) (m n : Nat) :
    sha256dRangeSha σ (m + n) = sha256dRangeSha σ m + sha256dRangeSha σ n := by
  unfold sha256dRangeSha
  rw [ShaCost.rep_add]

/-- **The complete 2^32 range, closed form with the multiplier explicit.** AR-27's distinction
stands: `sha256dRangeSha σ m` is `m` complete attempts, each paying its own padding; AR-27's
`rangeCost` (one header padding for the whole range) is a different figure and `rangeCost_add` is
false for it (AR-27 §7). -/
theorem sha256dRangeSha_nonceSpace (σ : R) :
    sha256dRangeSha σ nonceSpace =
      { mul := 4294967296 * 10368, add := 4294967296 * 7128, sub := 0, neg := 0, cast := 0,
        channel := 4294967296 * 1728, bitXor := 4294967296 * 2112, bitAnd := 4294967296 * 960,
        bitNot := 4294967296 * 192, shift := 4294967296 * 3744, bitOr := 4294967296 * 1728,
        padByte := 4294967296 * 192, zeroByte := 4294967296 * 62, lenByte := 4294967296 * 16,
        padShift := 4294967296 * 16, padMod := 4294967296 * 16,
        blockByteRead := 4294967296 * 192, regroupMul := 4294967296 * 144,
        regroupAdd := 4294967296 * 144, regroupTrunc := 4294967296 * 48,
        schedWordWrite := 4294967296 * 48, digestByte := 4294967296 * 64,
        digestWordRead := 4294967296 * 16 } := by
  rw [sha256dRangeSha_eq, nonceSpace]

/-- **THE COMPLETE 2^32 RANGE, EVALUATED.** The compression fields are §6's nonce-space figure;
the byte and load fields are 824 633 720 832 padded bytes (266 287 972 352 of them zero fill,
68 719 476 736 length bytes, 68 719 476 736 shifts, 68 719 476 736 moduli), 824 633 720 832 block
byte reads, 618 475 290 624 regrouping multiplications, 618 475 290 624 regrouping additions,
206 158 430 208 truncations, 206 158 430 208 schedule word writes, 274 877 906 944 digest bytes and
68 719 476 736 chaining-word reads. Bytes are bytes and operations are operations: the two halves
of this record are not comparable and are not compared. -/
theorem sha256dRangeSha_nonceSpace_literals (σ : R) :
    sha256dRangeSha σ nonceSpace =
      { mul := 44530220924928, add := 30614526885888, sub := 0, neg := 0, cast := 0,
        channel := 7421703487488, bitXor := 9070970929152, bitAnd := 4123168604160,
        bitNot := 824633720832, shift := 16080357556224, bitOr := 7421703487488,
        padByte := 824633720832, zeroByte := 266287972352, lenByte := 68719476736,
        padShift := 68719476736, padMod := 68719476736,
        blockByteRead := 824633720832, regroupMul := 618475290624,
        regroupAdd := 618475290624, regroupTrunc := 206158430208,
        schedWordWrite := 206158430208, digestByte := 274877906944,
        digestWordRead := 68719476736 } := by
  rw [sha256dRangeSha_nonceSpace]

/-- **Additivity of the complete range split at the half point.** -/
theorem sha256dRangeSha_nonceSpace_split (σ : R) :
    sha256dRangeSha σ (2 ^ 31) + sha256dRangeSha σ (2 ^ 31) = sha256dRangeSha σ nonceSpace := by
  rw [← sha256dRangeSha_add]
  unfold nonceSpace
  norm_num

/-! ### 7.4 AR-41's source-bound corrections, kept as a separate record

AR-41 prices the **executing** compression (`AlgebraCostSource.lean`, bound by the
`ar41-source-bound-cost` evidence block) and its per-block counts decide three fields differently
from the canonical bodies AR-27/this module price: `add` 2328 (the schedule step has three binary
additions, not four operands), `bitXor` 640 (`Ch` is one XOR, not two), `schedWordWrite` 64 (the 48
schedule-extension writes were not counted by AR-27's loader). The record below re-instantiates the
complete attempt at those three source-decided values, so the two readings differ by named numbers
rather than by prose. It is a **reading of that receipt's field table**, not a second derivation;
the arithmetic relating the two is proved here. -/

/-- The complete attempt at AR-41's source-bound per-block counts, with the ring layer unchanged
(the C5 identification is the same in both readings). -/
def sha256dAttemptShaSource (σ : R) : ShaCost σ :=
  { sha256dAttemptSha σ with add := 6984, bitXor := 1920, schedWordWrite := 192 }

/-- **The source-bound attempt's closed count.** -/
theorem sha256dAttemptShaSource_eq (σ : R) :
    sha256dAttemptShaSource σ =
      { mul := 10368, add := 6984, sub := 0, neg := 0, cast := 0, channel := 1728,
        bitXor := 1920, bitAnd := 960, bitNot := 192, shift := 3744, bitOr := 1728,
        padByte := 192, zeroByte := 62, lenByte := 16, padShift := 16, padMod := 16,
        blockByteRead := 192, regroupMul := 144, regroupAdd := 144, regroupTrunc := 48,
        schedWordWrite := 192, digestByte := 64, digestWordRead := 16 } := by
  rw [sha256dAttemptShaSource, sha256dAttemptSha_eq]

/-- **The three divergences between the canonical-body attempt and the source-bound attempt.**
Over three blocks: `add` is 144 lower (48 per block), `bitXor` 192 lower (64 per block), and
`schedWordWrite` 144 higher (48 per block). Each per-block figure is the AR-41 table's. -/
theorem sha256dAttemptShaSource_divergence (σ : R) :
    (sha256dAttemptSha σ).add - (sha256dAttemptShaSource σ).add = 144 ∧
      (sha256dAttemptSha σ).bitXor - (sha256dAttemptShaSource σ).bitXor = 192 ∧
      (sha256dAttemptShaSource σ).schedWordWrite - (sha256dAttemptSha σ).schedWordWrite = 144 := by
  rw [sha256dAttemptSha_eq, sha256dAttemptShaSource_eq]
  norm_num

/-- The range over the source-bound attempt. -/
def sha256dRangeShaSource (σ : R) (m : Nat) : ShaCost σ :=
  ShaCost.rep σ m (sha256dAttemptShaSource σ)

/-- **The source-bound 2^32 range, evaluated.** The three corrected fields and nothing else differ
from `sha256dRangeSha_nonceSpace_literals`. -/
theorem sha256dRangeShaSource_nonceSpace_literals (σ : R) :
    sha256dRangeShaSource σ nonceSpace =
      { mul := 44530220924928, add := 29996051595264, sub := 0, neg := 0, cast := 0,
        channel := 7421703487488, bitXor := 8246337208320, bitAnd := 4123168604160,
        bitNot := 824633720832, shift := 16080357556224, bitOr := 7421703487488,
        padByte := 824633720832, zeroByte := 266287972352, lenByte := 68719476736,
        padShift := 68719476736, padMod := 68719476736,
        blockByteRead := 824633720832, regroupMul := 618475290624,
        regroupAdd := 618475290624, regroupTrunc := 206158430208,
        schedWordWrite := 824633720832, digestByte := 274877906944,
        digestWordRead := 68719476736 } := by
  rw [sha256dRangeShaSource, sha256dAttemptShaSource_eq, ShaCost.rep, nonceSpace]
  apply ShaCost.eq_of_fields <;> norm_num

/-- **The carry-stage inventory of the source-bound reading, kept distinct from the add field.**
AR-41 §5: the executing compression performs 600 binary additions per block (48 schedule steps × 3,
64 rounds × 7, 8 add-back), and a ripple expansion of one 32-bit addition evaluates 32 carry
positions. The stage count is a **model of the adder** (AR-41, grade `model`); the addition count is
read off the extracted body chain. Neither is a field of `ShaCost`, and neither is a time. -/
theorem carryStageInventory :
    48 * 3 + 64 * 7 + 8 = 600 ∧ 600 * 32 = 19200 ∧ 3 * 19200 = 57600 ∧
      nonceSpace * 57600 = 247390116249600 := by
  unfold nonceSpace
  norm_num

/-! ## 8. The three faces

The attempt, the range and the complete figures are computed at all three dial values, `σ = -1`
(elliptic), `σ = 0` (parabolic) and `σ = 1` (hyperbolic), and **the three computations agree in
every field**. The reason is a property of the counted bodies, not of the algebra: every count of
§2-§7 is a closed record of `Nat` literals, and no counted body - `ringMul`, `ringUpper`, `rotr`,
`shrx`, `sigma0`, `sigma1`, `Sigma0`, `Sigma1`, `choose`, `majority`, `pad`, `lengthByte`, `b64`,
`beWord16`, `BitVec` addition - branches on a dial or mentions one except `ringMul`, whose body is
one expression for every `σ` (`ringMul_body_ops`). So the measure is **σ-blind by construction**,
and the honest form of the result is a statement about the counted bodies. It is **not** a
statement that the trichotomy does not affect cost. -/

/-- **The attempt figure does not depend on the dial** (eleven-field vector form). -/
theorem sha256dAttemptCost_vector (σ : W32) :
    (sha256dAttemptCost σ).vector =
      (10368, 7128, 0, 0, 0, 1728, 2112, 960, 192, 3744, 1728) := by
  rw [sha256dAttemptCost_eq]
  rfl

/-- **THE THREE FACES, ATTEMPT FIGURE.** -/
theorem sha256dAttemptCost_threeFaces :
    (sha256dAttemptCost faceElliptic).vector = (sha256dAttemptCost faceParabolic).vector ∧
      (sha256dAttemptCost faceParabolic).vector = (sha256dAttemptCost faceHyperbolic).vector := by
  rw [sha256dAttemptCost_vector, sha256dAttemptCost_vector, sha256dAttemptCost_vector]
  exact ⟨rfl, rfl⟩

/-- **The range figure does not depend on the dial**: the closed form's right-hand side mentions
no dial. -/
theorem sha256dRangeCost_faceBlind (m : Nat) :
    (sha256dRangeCost faceElliptic m).vector = (sha256dRangeCost faceParabolic m).vector ∧
      (sha256dRangeCost faceParabolic m).vector = (sha256dRangeCost faceHyperbolic m).vector := by
  have h1 := sha256dRangeCost_eq faceElliptic m
  have h2 := sha256dRangeCost_eq faceParabolic m
  have h3 := sha256dRangeCost_eq faceHyperbolic m
  rw [h1, h2, h3]
  exact ⟨rfl, rfl⟩

/-- **THE COMPLETE ATTEMPT DOES NOT DEPEND ON THE DIAL**, headline fields: multiplication,
addition, XOR, shift, padded bytes, block bytes read, schedule writes and digest bytes agree across
the three faces. -/
theorem sha256dAttemptSha_threeFaces :
    (sha256dAttemptSha faceElliptic).mul = (sha256dAttemptSha faceParabolic).mul ∧
      (sha256dAttemptSha faceParabolic).mul = (sha256dAttemptSha faceHyperbolic).mul ∧
      (sha256dAttemptSha faceElliptic).add = (sha256dAttemptSha faceParabolic).add ∧
      (sha256dAttemptSha faceParabolic).add = (sha256dAttemptSha faceHyperbolic).add ∧
      (sha256dAttemptSha faceElliptic).bitXor = (sha256dAttemptSha faceParabolic).bitXor ∧
      (sha256dAttemptSha faceParabolic).bitXor = (sha256dAttemptSha faceHyperbolic).bitXor ∧
      (sha256dAttemptSha faceElliptic).shift = (sha256dAttemptSha faceParabolic).shift ∧
      (sha256dAttemptSha faceParabolic).shift = (sha256dAttemptSha faceHyperbolic).shift ∧
      (sha256dAttemptSha faceElliptic).padByte = (sha256dAttemptSha faceParabolic).padByte ∧
      (sha256dAttemptSha faceParabolic).padByte = (sha256dAttemptSha faceHyperbolic).padByte ∧
      (sha256dAttemptSha faceElliptic).blockByteRead =
        (sha256dAttemptSha faceParabolic).blockByteRead ∧
      (sha256dAttemptSha faceParabolic).blockByteRead =
        (sha256dAttemptSha faceHyperbolic).blockByteRead ∧
      (sha256dAttemptSha faceElliptic).schedWordWrite =
        (sha256dAttemptSha faceParabolic).schedWordWrite ∧
      (sha256dAttemptSha faceParabolic).schedWordWrite =
        (sha256dAttemptSha faceHyperbolic).schedWordWrite ∧
      (sha256dAttemptSha faceElliptic).digestByte = (sha256dAttemptSha faceParabolic).digestByte ∧
      (sha256dAttemptSha faceParabolic).digestByte = (sha256dAttemptSha faceHyperbolic).digestByte := by
  rw [sha256dAttemptSha_eq, sha256dAttemptSha_eq, sha256dAttemptSha_eq]
  norm_num

/-- `-1` is not a square mod 4, so it is not a square mod `2^32`: the elliptic dial has no root. -/
theorem no_root_neg_one_w4 : ∀ r : ZMod 4, r ^ 2 ≠ (-1 : ZMod 4) := by decide

/-- **The elliptic face's channel premise is unsatisfiable**, the same fact as AR-15's
`no_root_neg_one_w32`, re-proved through the reduction to `ZMod 4`. -/
theorem no_root_neg_one_w32 : ∀ r : W32, r ^ 2 ≠ (-1 : W32) := by
  intro r h
  have h' := congrArg (ZMod.castHom (n := 2 ^ 32) (m := 4) (by decide : (4 : ℕ) ∣ 2 ^ 32)
    (ZMod 4)) h
  rw [map_pow, map_neg, map_one] at h'
  exact no_root_neg_one_w4 _ h'

/-- The hyperbolic face admits the root `1`. -/
theorem root_hyperbolic : (1 : W32) ^ 2 = faceHyperbolic := rfl

/-- The parabolic face admits the root `0`. -/
theorem root_parabolic : (0 : W32) ^ 2 = faceParabolic := rfl

/-- **Channel multiplicativity is admissible at two faces and refused at the third.** The
difference between the faces is a refusal of the C5 channel reading, not a different number; no
count above uses this premise. -/
theorem channelMul_admissible_faces :
    (ringUpper faceHyperbolic (ringMul faceHyperbolic (0, 1) (0, 1)) =
        ringUpper faceHyperbolic (0, 1) * ringUpper faceHyperbolic (0, 1)) ∧
      (ringUpper faceParabolic (ringMul faceParabolic (0, 1) (0, 1)) =
        ringUpper faceParabolic (0, 1) * ringUpper faceParabolic (0, 1)) ∧
      (∀ r : W32, r ^ 2 ≠ faceElliptic) :=
  ⟨ringUpper_mul (r := faceHyperbolic) (by rfl) _ _,
    ringUpper_mul (r := faceParabolic) (by rfl) _ _, fun r => no_root_neg_one_w32 r⟩

/-! ### 9.1 Audit rows

Every headline of §7-§8 above, printed with its axiom set. The permitted cone is
`{propext, Classical.choice, Quot.sound}` and no row may contain `sorryAx`. The rows of §0-§6 live
in `AlgebraCostPowComparePart1.lean`; together the two lists are the former module's single list,
row for row and in the same order. -/

#print axioms ShaCost.eq_of_fields
#print axioms ShaCost.composed_comm
#print axioms ShaCost.composed_assoc
#print axioms ShaCost.composed_zero
#print axioms ShaCost.rep_zero
#print axioms ShaCost.rep_succ
#print axioms ShaCost.rep_add
#print axioms ShaCost.comp_ofRound
#print axioms comp_add
#print axioms comp_rep
#print axioms padCost_padByte
#print axioms padCost_80
#print axioms padCost_32
#print axioms b64_body
#print axioms beWord16_body
#print axioms loadCost_fields
#print axioms loadCost_comp_zero
#print axioms storeCost_fields
#print axioms blockFullCost_eq
#print axioms innerCostSha_eq
#print axioms outerCostSha_eq
#print axioms sha256dAttemptSha_eq
#print axioms sha256dAttemptSha_comp_eq
#print axioms sha256dAttemptSha_ringFields
#print axioms sha256dRangeSha_eq
#print axioms sha256dRangeSha_succ
#print axioms sha256dRangeSha_add
#print axioms sha256dRangeSha_nonceSpace
#print axioms sha256dRangeSha_nonceSpace_literals
#print axioms sha256dRangeSha_nonceSpace_split
#print axioms sha256dAttemptShaSource_eq
#print axioms sha256dAttemptShaSource_divergence
#print axioms sha256dRangeShaSource_nonceSpace_literals
#print axioms carryStageInventory
#print axioms sha256dAttemptCost_vector
#print axioms sha256dAttemptCost_threeFaces
#print axioms sha256dRangeCost_faceBlind
#print axioms sha256dAttemptSha_threeFaces
#print axioms no_root_neg_one_w4
#print axioms no_root_neg_one_w32
#print axioms root_hyperbolic
#print axioms root_parabolic
#print axioms channelMul_admissible_faces

end algebra_cost_pow_compare
