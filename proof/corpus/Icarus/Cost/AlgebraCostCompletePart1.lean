/-
PART 1 of the former work/proofs/AlgebraCostComplete.lean, split for the apeiron 1,500-line
owned-source cap (repos/apeiron/AGENTS.md:222). Content here: §0 the width and the three dials,
§1 the measures (AR-20's eleven-field `RoundCost`, AR-10's six-field `RingCost`, and AR-15's
twenty-three-field `ShaCost`), §2 the compression counts re-derived from the canonical bodies, and
the audit rows for the declarations owned here. No declaration changed name, namespace or type;
the whole module is re-assembled by the AlgebraCostComplete facade, which imports this part and
AlgebraCostCompletePart2.
-/

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

namespace algebra_cost_complete

open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
open apeiron_word

variable {R : Type u} [CommRing R]

/-! ## 0. The width and the three dials

The SHA-2 word is `BitVec 32` (`Sha2.lean:22`), so the ring this count is read at is `ZMod (2 ^ 32)`;
the extracted `Cell<i64>` arithmetic runs at width 64 (`WordRefinement.lean:38`) and its counts are
kept out of the SHA-2 figures. The three dial values of `cell.Sigma` at width 32 are `denoteSigma`,
`WordRefinement.lean:29-35`: elliptic `-1`, parabolic `0`, hyperbolic `1`. -/

/-- The ring the SHA-2 attempt count is measured in: `w = 32`. -/
abbrev W32 := ZMod (2 ^ 32)

/-- The elliptic face's dial value. -/
abbrev faceElliptic : W32 := -1
/-- The parabolic face's dial value. -/
abbrev faceParabolic : W32 := 0
/-- The hyperbolic face's dial value. -/
abbrev faceHyperbolic : W32 := 1

/-! ## 1. The measures

AR-20's `RoundCost σ` has eleven fields and AR-10's `RingCost σ` has six. Neither module is a root of
any corpus `lean_lib`, so both are re-declared here and their composition laws re-proved. `ShaCost σ`
is the record this task exists for: it **contains** the eleven compression fields unchanged and adds
twelve more, so that the padding, the block load and the digest store are counted in the same
measure as the compression instead of being invisible in it.

### 1.1 The eleven compression fields -/

/-- `RoundCost σ`: AR-15's record. Eleven fields, one counted quantity each - the six ring operations
of `R[x]/(x^2 - sigma)` that AR-10's `RingCost` records, and the five `Word` operations that the
two-slot algebra has no reading for at word width. `σ` is the dial the ring fields are read at; §10
states as a result that no count enters it. -/
structure RoundCost (σ : R) where
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
  /-- channel reads, applications of a root-parameterised linear functional -/
  channel : Nat
  /-- bitwise XOR of `Word`s -/
  bitXor : Nat
  /-- bitwise AND of `Word`s -/
  bitAnd : Nat
  /-- bitwise NOT of a `Word` -/
  bitNot : Nat
  /-- logical shifts of a `Word`, left or right -/
  shift : Nat
  /-- bitwise OR of `Word`s -/
  bitOr : Nat
  deriving Repr

/-- The record of a compression that performs no operation of either kind. -/
def RoundCost.zero (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **Dependent composition.** A computation that runs after another performs both sets of
operations, so every count totals. It is the only composition this module uses. Temporal and
spatial O(1): a closed term of eleven `Nat` additions, no recursion, no allocation. Counted in:
nothing - `composed` is the measure's own operator, not an operation of a hash. Derivation: eleven
independent sums, one per field. -/
def RoundCost.composed (σ : R) (a b : RoundCost σ) : RoundCost σ :=
  { mul := a.mul + b.mul, add := a.add + b.add, sub := a.sub + b.sub, neg := a.neg + b.neg,
    cast := a.cast + b.cast, channel := a.channel + b.channel, bitXor := a.bitXor + b.bitXor,
    bitAnd := a.bitAnd + b.bitAnd, bitNot := a.bitNot + b.bitNot, shift := a.shift + b.shift,
    bitOr := a.bitOr + b.bitOr }

instance instAddRoundCost (σ : R) : Add (RoundCost σ) := ⟨RoundCost.composed σ⟩

/-- The dependent sum, unfolded, so composition rewrites and its laws close. -/
@[simp] theorem RoundCost.composed_apply (σ : R) (a b : RoundCost σ) :
    a + b = RoundCost.composed σ a b := rfl

/-- **Equality of counts is fieldwise equality**: a record with one field per quantity has no room
for a hidden one. -/
theorem RoundCost.eq_of_fields {σ : R} {a b : RoundCost σ}
    (h1 : a.mul = b.mul) (h2 : a.add = b.add) (h3 : a.sub = b.sub) (h4 : a.neg = b.neg)
    (h5 : a.cast = b.cast) (h6 : a.channel = b.channel) (h7 : a.bitXor = b.bitXor)
    (h8 : a.bitAnd = b.bitAnd) (h9 : a.bitNot = b.bitNot) (h10 : a.shift = b.shift)
    (h11 : a.bitOr = b.bitOr) : a = b := by
  cases a
  cases b
  cases h1; cases h2; cases h3; cases h4; cases h5; cases h6; cases h7; cases h8; cases h9
  cases h10; cases h11
  rfl

/-- **Composition is commutative.** -/
theorem RoundCost.composed_comm (σ : R) (a b : RoundCost σ) : a + b = b + a := by
  cases a
  cases b
  simp [RoundCost.composed, Nat.add_comm]

/-- **Composition is associative**: the grouping of a dependent chain does not change its cost. -/
theorem RoundCost.composed_assoc (σ : R) (a b c : RoundCost σ) : a + b + c = a + (b + c) := by
  cases a
  cases b
  cases c
  simp [RoundCost.composed, Nat.add_assoc]

/-- **Composition with the zero record is neutral.** -/
theorem RoundCost.composed_zero (σ : R) (a : RoundCost σ) : a + RoundCost.zero σ = a := by
  cases a
  simp [RoundCost.composed, RoundCost.zero]

/-- The **cost vector**: the eleven counts with the dial index forgotten, so two faces can be
compared on it even though their records carry different indices. Temporal and spatial O(1): a tuple
of eleven projections. -/
def RoundCost.vector {σ : R} (c : RoundCost σ) :
    Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat :=
  (c.mul, c.add, c.sub, c.neg, c.cast, c.channel, c.bitXor, c.bitAnd, c.bitNot, c.shift, c.bitOr)

/-- **`rep σ n a` is the measure of `n` dependent copies of an operation of cost `a`.** Temporal O(1)
and spatial O(1): eleven `Nat` multiplications on an eleven-field record, no recursion, no
allocation. Counted in: nothing - this is the iteration operator of the measure, not an operation of a
hash; the `n` it is given is a count of repetitions, so the value grows while the function does not
recur. Derivation: fieldwise `n * a.field`, and `RoundCost.rep_add` proves this image additive. -/
def RoundCost.rep (σ : R) (n : Nat) (a : RoundCost σ) : RoundCost σ :=
  { mul := n * a.mul, add := n * a.add, sub := n * a.sub, neg := n * a.neg, cast := n * a.cast,
    channel := n * a.channel, bitXor := n * a.bitXor, bitAnd := n * a.bitAnd,
    bitNot := n * a.bitNot, shift := n * a.shift, bitOr := n * a.bitOr }

/-- **Base case: zero copies cost nothing.** -/
@[simp] theorem RoundCost.rep_zero (σ : R) (a : RoundCost σ) :
    RoundCost.rep σ 0 a = RoundCost.zero σ := by
  simp [RoundCost.rep, RoundCost.zero]

/-- **Recurrence: one more copy is one copy plus the copies already made.** -/
theorem RoundCost.rep_succ (σ : R) (n : Nat) (a : RoundCost σ) :
    RoundCost.rep σ (n + 1) a = a + RoundCost.rep σ n a := by
  cases a
  simp [RoundCost.rep, RoundCost.composed, Nat.succ_mul, Nat.add_comm]

/-- **THE COMPOSITION LAW over an iterated construction, eleven fields.** `n + m` copies cost the sum
of `n` copies and `m` copies, field by field. -/
theorem RoundCost.rep_add (σ : R) (n m : Nat) (a : RoundCost σ) :
    RoundCost.rep σ (n + m) a = RoundCost.rep σ n a + RoundCost.rep σ m a := by
  cases a
  simp [RoundCost.rep, RoundCost.composed, Nat.add_mul]

/-- **Iteration distributes over composition.** -/
theorem RoundCost.rep_composed (σ : R) (n : Nat) (a b : RoundCost σ) :
    RoundCost.rep σ n (a + b) = RoundCost.rep σ n a + RoundCost.rep σ n b := by
  cases a
  cases b
  simp [RoundCost.rep, RoundCost.composed, Nat.mul_add]

/-! ### 1.2 AR-10's six-field ring measure, and its composition law -/

/-- `RingCost σ`: AR-10's measure, the six ring quantities of `R[x]/(x^2 - sigma)`. -/
structure RingCost (σ : R) where
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
  deriving Repr

/-- The record of a computation that performs no ring operation. -/
def RingCost.zero (σ : R) : RingCost σ := ⟨0, 0, 0, 0, 0, 0⟩

/-- Dependent composition over the six ring fields. Temporal and spatial O(1): a closed term of six
`Nat` additions. -/
def RingCost.composed (σ : R) (a b : RingCost σ) : RingCost σ :=
  ⟨a.mul + b.mul, a.add + b.add, a.sub + b.sub, a.neg + b.neg, a.cast + b.cast,
    a.channel + b.channel⟩

instance instAddRingCost (σ : R) : Add (RingCost σ) := ⟨RingCost.composed σ⟩

@[simp] theorem RingCost.composed_apply (σ : R) (a b : RingCost σ) :
    a + b = RingCost.composed σ a b := rfl

/-- **Equality of ring counts is fieldwise equality**, six fields. -/
theorem RingCost.eq_of_fields {σ : R} {a b : RingCost σ} (h1 : a.mul = b.mul) (h2 : a.add = b.add)
    (h3 : a.sub = b.sub) (h4 : a.neg = b.neg) (h5 : a.cast = b.cast) (h6 : a.channel = b.channel) :
    a = b := by
  cases a
  cases b
  cases h1; cases h2; cases h3; cases h4; cases h5; cases h6
  rfl

theorem RingCost.composed_comm (σ : R) (a b : RingCost σ) : a + b = b + a := by
  cases a
  cases b
  simp [RingCost.composed, Nat.add_comm]

theorem RingCost.composed_assoc (σ : R) (a b c : RingCost σ) : a + b + c = a + (b + c) := by
  cases a
  cases b
  cases c
  simp [RingCost.composed, Nat.add_assoc]

/-- `n` dependent copies of a ring cost. Temporal and spatial O(1): six `Nat` multiplications. -/
def RingCost.rep (σ : R) (n : Nat) (a : RingCost σ) : RingCost σ :=
  ⟨n * a.mul, n * a.add, n * a.sub, n * a.neg, n * a.cast, n * a.channel⟩

@[simp] theorem RingCost.rep_zero (σ : R) (a : RingCost σ) : RingCost.rep σ 0 a = RingCost.zero σ := by
  simp [RingCost.rep, RingCost.zero]

/-- **THE COMPOSITION LAW over an iterated construction, six fields.** AR-10's `RingCost.rep_add`,
re-proved here. -/
theorem RingCost.rep_add (σ : R) (n m : Nat) (a : RingCost σ) :
    RingCost.rep σ (n + m) a = RingCost.rep σ n a + RingCost.rep σ m a := by
  cases a
  simp [RingCost.rep, RingCost.composed, Nat.add_mul]

theorem RingCost.rep_composed (σ : R) (n : Nat) (a b : RingCost σ) :
    RingCost.rep σ n (a + b) = RingCost.rep σ n a + RingCost.rep σ n b := by
  cases a
  cases b
  simp [RingCost.rep, RingCost.composed, Nat.mul_add]

/-- **The ring projection.** The first six fields of `RoundCost σ` read as AR-10's `RingCost σ`.
Temporal and spatial O(1): six projections, no allocation. -/
def RoundCost.ringFields {σ : R} (c : RoundCost σ) : RingCost σ :=
  ⟨c.mul, c.add, c.sub, c.neg, c.cast, c.channel⟩

/-- The projection commutes with composition. -/
theorem ringFields_add (σ : R) (a b : RoundCost σ) :
    (a + b).ringFields = a.ringFields + b.ringFields := by
  cases a
  cases b
  rfl

/-- The projection commutes with the iteration operator. -/
theorem ringFields_rep (σ : R) (n : Nat) (a : RoundCost σ) :
    (RoundCost.rep σ n a).ringFields = RingCost.rep σ n a.ringFields := by
  cases a
  rfl
/-! ### 1.3 `ShaCost σ`: the record the padding, the load and the store are counted in

AR-20's attempt figure counts the compression function only. Four quantities were named there as
"invisible, not priced": the padding's own list writes, the block loading loop, the digest store,
and every element read. `ShaCost σ` adds twelve fields, one counted quantity each, so that the three
priced ones are visible and the fourth is *named and bounded* rather than absent.

| fields | counted quantity | unit |
|---|---|---|
| `mul` `add` `sub` `neg` `cast` `channel` `bitXor` `bitAnd` `bitNot` `shift` `bitOr` | AR-15's eleven compression counts, unchanged | ring and `Word` operations |
| `padByte` | bytes appended by `Crypto.Sha2.pad`, including the message copy | bytes |
| `zeroByte` | of those, the zero-fill bytes | bytes |
| `lenByte` | of those, the eight length-field bytes | bytes |
| `padShift` | `Nat` shifts extracting the eight length bytes (`lengthByte`) | `Nat` operations |
| `padMod` | the eight `% 256` truncations of those bytes | `Nat` operations |
| `blockByteRead` | byte reads of the padded block by the sixteen-word loader | bytes |
| `regroupMul` | `Nat` multiplications in the big-endian regrouping | `Nat` operations |
| `regroupAdd` | `Nat` additions in the big-endian regrouping | `Nat` operations |
| `regroupTrunc` | `BitVec.ofNat 32` truncations in the regrouping | `Word` truncations |
| `schedWordWrite` | word writes into the 64-word schedule | words |
| `digestByte` | bytes written by the eight-word big-endian digest store | bytes |
| `digestWordRead` | chaining-word reads performed by that store | words |

**A field never mixes two quantities** (`CostAlgebra.lean:34-45`), and `ShaCost.eq_of_fields` makes
equality of counts fieldwise equality, so a record with one field per quantity has no room for a
hidden one. **The bytes and the operations are different units and are never summed together**: no
theorem in this module adds a byte field to an operation field, and `le`/`add`/`rep` are fieldwise,
so a comparison across units cannot be stated at all.

**Excluded with a reason, not silently:** index and address arithmetic on `Nat`/`Std.Usize`
(`4 * i.val`, `64 * k`, `4 * i`) is *address computation*, it is not memory traffic and not a ring or
`BitVec` operation, and it is the quantity `CostAlgebra.Cost` and
`CostBridge.loads_invisible_to_order` (`CostBridge.lean:161`) already own. No constant-time or
address-secrecy claim is made here, so charging it would add a field no consumer of this measure
reads. The full exclusion list is §11. -/

/-- `ShaCost σ`: the measure. Twenty-three fields, one counted quantity each. The first eleven are
AR-15's `RoundCost σ` in AR-15's order and are projected back by `ShaCost.comp`; the remaining twelve
count the padding writes, the block load, the digest store, and the two read scopes named above.
`σ` is the dial the ring fields are read at; §10 states as a result that no count enters it. -/
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

/-- **Dependent composition over all twenty-three fields.** Temporal and spatial O(1): a closed term
of twenty-three `Nat` additions, no recursion, no allocation. Counted in: nothing. Derivation:
twenty-three independent sums, one per field. -/
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

/-- **`rep σ n a` is the measure of `n` dependent copies of a cost `a`.** Temporal O(1) and spatial
O(1): twenty-three `Nat` multiplications on a twenty-three-field record, no recursion, no
allocation. Counted in: nothing - the iteration operator of the measure. Derivation: fieldwise
`n * a.field`, and `ShaCost.rep_add` proves this image additive, which is the closure the §8 comment
requires: the `n`-fold sum telescopes to `n` copies and no field escapes the record. -/
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

/-- **THE COMPOSITION LAW over an iterated construction, twenty-three fields.** Every range figure of
§9 is this operator, so every additivity statement about a range is an instance of it. -/
theorem ShaCost.rep_add (σ : R) (n m : Nat) (a : ShaCost σ) :
    ShaCost.rep σ (n + m) a = ShaCost.rep σ n a + ShaCost.rep σ m a := by
  cases a
  simp [ShaCost.rep, ShaCost.composed, Nat.add_mul]

/-- **Iteration distributes over composition.** -/
theorem ShaCost.rep_composed (σ : R) (n : Nat) (a b : ShaCost σ) :
    ShaCost.rep σ n (a + b) = ShaCost.rep σ n a + ShaCost.rep σ n b := by
  cases a
  cases b
  simp [ShaCost.rep, ShaCost.composed, Nat.mul_add]

/-- **The compression projection.** The first eleven fields read as AR-15's `RoundCost σ`, so every
`ShaCost` figure yields an AR-20 figure by forgetting the twelve new fields, and no new field can
perturb an AR-20 count. Temporal and spatial O(1): eleven projections. -/
def ShaCost.comp {σ : R} (c : ShaCost σ) : RoundCost σ :=
  { mul := c.mul, add := c.add, sub := c.sub, neg := c.neg, cast := c.cast, channel := c.channel,
    bitXor := c.bitXor, bitAnd := c.bitAnd, bitNot := c.bitNot, shift := c.shift, bitOr := c.bitOr }

/-- **The embedding.** A pure compression cost enters `ShaCost` with all twelve new fields zero.
Temporal and spatial O(1): a record literal. -/
def ShaCost.ofRound {σ : R} (c : RoundCost σ) : ShaCost σ :=
  { mul := c.mul, add := c.add, sub := c.sub, neg := c.neg, cast := c.cast, channel := c.channel,
    bitXor := c.bitXor, bitAnd := c.bitAnd, bitNot := c.bitNot, shift := c.shift, bitOr := c.bitOr,
    padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
    blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
    schedWordWrite := 0, digestByte := 0, digestWordRead := 0 }

/-- The embedding is the identity on the eleven compression fields. -/
theorem ShaCost.comp_ofRound (σ : R) (c : RoundCost σ) : (ShaCost.ofRound c).comp = c := rfl

/-- **The compression projection has no padding, load or store content**: an AR-15 cost cannot be
made to see a byte, because the eleven fields it projects to hold no byte. -/
theorem ShaCost.comp_ofRound_zero_new (σ : R) (c : RoundCost σ) :
    (ShaCost.ofRound c).padByte = 0 ∧ (ShaCost.ofRound c).zeroByte = 0 ∧
      (ShaCost.ofRound c).lenByte = 0 ∧ (ShaCost.ofRound c).padShift = 0 ∧
      (ShaCost.ofRound c).padMod = 0 ∧ (ShaCost.ofRound c).blockByteRead = 0 ∧
      (ShaCost.ofRound c).regroupMul = 0 ∧ (ShaCost.ofRound c).regroupAdd = 0 ∧
      (ShaCost.ofRound c).regroupTrunc = 0 ∧ (ShaCost.ofRound c).schedWordWrite = 0 ∧
      (ShaCost.ofRound c).digestByte = 0 ∧ (ShaCost.ofRound c).digestWordRead = 0 :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

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

/-- The six ring fields are a projection of the twenty-three, through the compression projection. -/
def ShaCost.ringFields {σ : R} (c : ShaCost σ) : RingCost σ := c.comp.ringFields

theorem ringFields_add' (σ : R) (a b : ShaCost σ) :
    (a + b).ringFields = a.ringFields + b.ringFields := by
  simp only [ShaCost.ringFields]
  rw [comp_add, ringFields_add]

theorem ringFields_rep' (σ : R) (n : Nat) (a : ShaCost σ) :
    (ShaCost.rep σ n a).ringFields = RingCost.rep σ n a.ringFields := by
  simp only [ShaCost.ringFields, comp_rep, ringFields_rep]

/-- The **cost vector**: the twenty-three counts with the dial index forgotten. Temporal and spatial
O(1): a tuple of twenty-three projections. -/
def ShaCost.vector {σ : R} (c : ShaCost σ) :
    Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat ×
      Nat × Nat × Nat × Nat × Nat × Nat × Nat :=
  (c.mul, c.add, c.sub, c.neg, c.cast, c.channel, c.bitXor, c.bitAnd, c.bitNot, c.shift, c.bitOr,
    c.padByte, c.zeroByte, c.lenByte, c.padShift, c.padMod, c.blockByteRead, c.regroupMul,
    c.regroupAdd, c.regroupTrunc, c.schedWordWrite, c.digestByte, c.digestWordRead)

/-! ## 2. The compression counts, re-derived from the canonical bodies

AR-15 §2 and AR-20 §2 are re-proved here rather than imported, because neither module is a root of any
corpus `lean_lib`. The agreement of these numbers with AR-15's and AR-20's is therefore a reading of
two files, not a kernel check; both carry the sha256 of their bytes and the sha256 is in the header
of this file. What *is* kernel-checked here is that each count follows from a body this module
unfolds with `rfl`. -/

/-- **The canonical multiplication** `(a + bx)(c + dx) = (ac + sigma*bd) + (ad + bc)x`: five
multiplications and two additions. Temporal and spatial O(1): the body is a single expression. -/
def ringMulCost (σ : R) : RoundCost σ :=
  { mul := 5, add := 2, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The canonical channel read** `z.1 + r*z.2`: one multiplication, one addition, one application
of the functional. -/
def ringUpperCost (σ : R) : RoundCost σ :=
  { mul := 1, add := 1, sub := 0, neg := 0, cast := 0, channel := 1,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The canonical norm form** `N(a + bx) = a^2 - sigma*b^2`: three multiplications, one
subtraction. -/
def ringNormFormCost (σ : R) : RoundCost σ :=
  { mul := 3, add := 0, sub := 1, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **One `Word` addition.** `Sha2.lean:108,109,112,116` adds `Word`s and a word addition mod `2^32`
is one addition in the ring `ZMod (2^32)`. Grade `model`: the identification is the named premise
`AddRead` of §7 and the occurrence count is a reading of those four source lines. -/
def wordAddCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 1, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The two-slot reading of one addition**, `RingCell.add` on `R × R`: two scalar additions. Kept
separate so that the additive reading is a stated alternative and not a silent choice. -/
def ringCellAddCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 2, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The rotation reading**, `ringUpper 1 (ringMul 1 (x,0) (2^n, 2^(32-n)))`: one canonical
multiplication plus one canonical channel read. The reading is grade `model` (`RotRead`, §7); the
count is a fact about the two bodies and needs no premise. -/
def rotReadCost (σ : R) : RoundCost σ := ringMulCost σ + ringUpperCost σ

/-- `ROTR`: two shifts and one disjunction. A rotation is not one operation in the canonical body. -/
def rotrCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 2, bitOr := 1 }

/-- `SHR`: one shift and nothing else. -/
def shrxCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 1, bitOr := 0 }

/-- One bitwise XOR, used by the round bodies. Temporal and spatial O(1). -/
def xorCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 1, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- One bitwise AND. -/
def andCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 1, bitNot := 0, shift := 0, bitOr := 0 }

/-- One bitwise NOT. -/
def notCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 1, shift := 0, bitOr := 0 }

/-- `σ₀(x) = ROTR^7 x ⊕ ROTR^18 x ⊕ SHR^3 x`: two rotations, one shift, two XORs. -/
def sigma0Cost (σ : R) : RoundCost σ := rotrCost σ + rotrCost σ + shrxCost σ + xorCost σ + xorCost σ

/-- `σ₁(x) = ROTR^17 x ⊕ ROTR^19 x ⊕ SHR^10 x`: the same inventory. -/
def sigma1Cost (σ : R) : RoundCost σ := rotrCost σ + rotrCost σ + shrxCost σ + xorCost σ + xorCost σ

/-- `Σ₀(x) = ROTR^2 x ⊕ ROTR^13 x ⊕ ROTR^22 x`: three rotations and two XORs, **no shift**. -/
def bigSigma0Cost (σ : R) : RoundCost σ :=
  rotrCost σ + rotrCost σ + rotrCost σ + xorCost σ + xorCost σ

/-- `Σ₁(x) = ROTR^6 x ⊕ ROTR^11 x ⊕ ROTR^25 x`: the same inventory as `Σ₀`. -/
def bigSigma1Cost (σ : R) : RoundCost σ :=
  rotrCost σ + rotrCost σ + rotrCost σ + xorCost σ + xorCost σ

/-- `Ch(x, y, z) = (x &&& y) ⊕ (¬x &&& z)`: two ANDs, one NOT, two XORs. -/
def chooseCost (σ : R) : RoundCost σ :=
  andCost σ + andCost σ + notCost σ + xorCost σ + xorCost σ

/-- `Maj(x, y, z) = (x &&& y) ⊕ (x &&& z) ⊕ (y &&& z)`: three ANDs and two XORs, no NOT. -/
def majorityCost (σ : R) : RoundCost σ :=
  andCost σ + andCost σ + andCost σ + xorCost σ + xorCost σ

/-! ### 2.1 The bodies the counts are read off -/

/-- Unfolding `ringMul` exhibits the expression whose five multiplications and two additions
`ringMulCost` counts. -/
theorem ringMul_body_ops (σ : R) (z w : RingCell R) :
    ringMul σ z w
      = (z.1 * w.1 + σ * z.2 * w.2, z.1 * w.2 + z.2 * w.1) := rfl

/-- Unfolding the channel read exhibits the one multiplication and one addition `ringUpperCost`
counts, and the one application of the functional. -/
theorem ringUpper_body_ops (r : R) (z : RingCell R) : ringUpper r z = z.1 + r * z.2 := rfl

/-- Unfolding the norm form exhibits the three multiplications and one subtraction `ringNormFormCost`
counts. -/
theorem ringNormForm_body_ops (σ : R) (z : RingCell R) :
    ringNormForm σ z = z.1 ^ 2 - σ * z.2 ^ 2 := rfl

/-- The rotation reading's count, read off the two canonical bodies. -/
theorem rotReadCost_fields (σ : R) :
    rotReadCost σ
      = { mul := 6, add := 3, sub := 0, neg := 0, cast := 0, channel := 1,
          bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 } := by
  simp [rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed]

/-- Unfolding `ROTR` exhibits the two shifts and the one disjunction. -/
theorem rotr_body_ops (x : Word) (n : Nat) :
    rotr x n = (x >>> (n % 32)) ||| (x <<< ((32 - n) % 32)) := rfl

/-- Unfolding `SHR` exhibits the one shift. -/
theorem shrx_body_ops (x : Word) (n : Nat) : shrx x n = x >>> (n % 32) := rfl

/-- Unfolding `σ₀` exhibits two rotations, one shift and two XORs. -/
theorem sigma0_body_ops (x : Word) : sigma0 x = rotr x 7 ^^^ rotr x 18 ^^^ shrx x 3 := rfl

/-- Unfolding `σ₁` exhibits two rotations, one shift and two XORs. -/
theorem sigma1_body_ops (x : Word) : sigma1 x = rotr x 17 ^^^ rotr x 19 ^^^ shrx x 10 := rfl

/-- Unfolding `Σ₀` exhibits three rotations and two XORs, with no `SHR`. -/
theorem bigSigma0_body_ops (x : Word) : Sigma0 x = rotr x 2 ^^^ rotr x 13 ^^^ rotr x 22 := rfl

/-- Unfolding `Σ₁` exhibits three rotations and two XORs, with no `SHR`. -/
theorem bigSigma1_body_ops (x : Word) : Sigma1 x = rotr x 6 ^^^ rotr x 11 ^^^ rotr x 25 := rfl

/-- Unfolding `Ch` exhibits two ANDs, one NOT and two XORs. -/
theorem choose_body_ops (x y z : Word) : choose x y z = (x &&& y) ^^^ (~~~x &&& z) := rfl

/-- Unfolding `Maj` exhibits three ANDs and two XORs. -/
theorem majority_body_ops (x y z : Word) : majority x y z = (x &&& y) ^^^ (x &&& z) ^^^ (y &&& z) :=
  rfl

/-- The shift counts of `σ₀` and `σ₁`, `Σ₀` and `Σ₁`, `Ch` and `Maj`, as the record states them. -/
theorem roundFns_shift_counts (σ : R) :
    (sigma0Cost σ).shift = 5 ∧ (sigma1Cost σ).shift = 5 ∧ (bigSigma0Cost σ).shift = 6 ∧
      (bigSigma1Cost σ).shift = 6 ∧ (chooseCost σ).shift = 0 ∧ (majorityCost σ).shift = 0 := by
  simp [sigma0Cost, sigma1Cost, bigSigma0Cost, bigSigma1Cost, chooseCost, majorityCost,
    rotrCost, shrxCost, xorCost, andCost, notCost, RoundCost.composed]

/-- **The shifts of a round come from its rotations; the two message-schedule shifts come from the
small sigmas.** A round's twelve shifts are `6 * 2` (six rotations, two shifts each), and a schedule
step's five shifts per sigma are `2 * 2 + 1`. -/
theorem shifts_come_from_rotations (σ : R) :
    (RoundCost.rep σ 6 (rotrCost σ)).shift = 12 ∧
      (sigma0Cost σ).shift + (sigma1Cost σ).shift = 5 + 5 ∧
      (bigSigma0Cost σ).shift + (bigSigma1Cost σ).shift = 6 + 6 := by
  simp [RoundCost.rep, rotrCost, sigma0Cost, sigma1Cost, bigSigma0Cost, bigSigma1Cost,
    shrxCost, xorCost, RoundCost.composed]

/-- The four-addition shape of `t1` as written at `Sha2.lean:108`: left-associated, four `Word`
additions. -/
theorem t1_arity (hh s c k w : Word) :
    hh + s + c + k + w = ((hh + s) + c) + k + w := rfl

/-! ### 2.2 One round, one schedule step, one block

- **Measure:** `RoundCost σ`, eleven fields, one counted quantity each.
- **Counted quantity:** ring operations of `R[x]/(x^2 - sigma)` and `Word` operations with no ring
reading.
- **Iteration operator:** `RoundCost.rep`; the recurrence `rep_succ` and the closure `rep_add`.
- **Recurrence of the round chain, and why it closes:** the chain recurses on `Nat`
  (`roundChainCost`), so induction is well founded with a checked base case; `roundChainCost_rep`
  proves by induction that the chain is `rep σ n (roundCost σ)`, and `RoundCost.rep_add` proves that
  operator additive, which is the telescoping the §8 comment requires. Temporal `n` steps and one
  stack frame per level for the recursive definition; `RoundCost.rep` is a closed term of eleven
  `Nat` multiplications, so its own temporal and spatial cost is O(1) - the growth is in the counted
  quantity, not in the work the function does. -/

/-- **One round** of `compress` (`Sha2.lean:100-119`), under the C5 rotation reading. The round body
contains six rotations (`Σ₀`'s three at `:109` and `Σ₁`'s three at `:108`), four XORs inside those
two sigmas, `Ch` (`choose e f g`) and `Maj` (`majority a b c`), and **seven** `Word` additions: four
in `t1 := hh + Sigma1 e + choose e f g + K.getD t 0 + w.getD t 0` (`:108`, left-associated, see
`t1_arity`), one in `t2` (`:109`), and two in the state update `0 => t1 + t2` and `4 => d + t1`
(`:112`, `:116`). The count of those seven occurrences is grade `model` (AR-15 §6 item 1: no
machine-checked operation-count mechanism for a term). Temporal and spatial O(1) per round; the
chain of rounds is `roundChainCost`. -/
def roundCost (σ : R) : RoundCost σ :=
  RoundCost.rep σ 6 (rotReadCost σ) + RoundCost.rep σ 6 (rotrCost σ) + RoundCost.rep σ 4 (xorCost σ)
    + chooseCost σ + majorityCost σ + RoundCost.rep σ 7 (wordAddCost σ)

/-- **The round's closed count**: 36 ring multiplications, 25 ring additions, 6 channel reads, 8
XORs, 5 ANDs, 1 NOT, 12 shifts, 6 disjunctions. -/
theorem roundCost_fields (σ : R) :
    roundCost σ
      = { mul := 36, add := 25, sub := 0, neg := 0, cast := 0, channel := 6,
          bitXor := 8, bitAnd := 5, bitNot := 1, shift := 12, bitOr := 6 } := by
  rfl

/-- The round's ring layer alone, as AR-10's six-field record `(36, 25, 0, 0, 0, 6)`. -/
theorem roundCost_ringFields (σ : R) : (roundCost σ).ringFields = ⟨36, 25, 0, 0, 0, 6⟩ := by
  rw [roundCost_fields]
  rfl

/-- **The same round under the two-slot additive reading.** With the seven word additions read as
`RingCell.add` instead of as single-slot additions, `add` rises from 25 to 32 and the other ten
fields are unchanged. -/
def roundCostTwoSlot (σ : R) : RoundCost σ :=
  RoundCost.rep σ 6 (rotReadCost σ) + RoundCost.rep σ 6 (rotrCost σ) + RoundCost.rep σ 4 (xorCost σ)
    + chooseCost σ + majorityCost σ + RoundCost.rep σ 7 (ringCellAddCost σ)

theorem roundCostTwoSlot_fields (σ : R) :
    roundCostTwoSlot σ
      = { mul := 36, add := 32, sub := 0, neg := 0, cast := 0, channel := 6,
          bitXor := 8, bitAnd := 5, bitNot := 1, shift := 12, bitOr := 6 } := by
  rfl

/-- **The two-slot additive reading costs fourteen more scalar additions and changes nothing else.** -/
theorem roundCost_twoSlot_add_only (σ : R) :
    (roundCostTwoSlot σ).mul = (roundCost σ).mul ∧ (roundCostTwoSlot σ).sub = (roundCost σ).sub ∧
      (roundCostTwoSlot σ).neg = (roundCost σ).neg ∧
      (roundCostTwoSlot σ).cast = (roundCost σ).cast ∧
      (roundCostTwoSlot σ).channel = (roundCost σ).channel ∧
      (roundCostTwoSlot σ).bitXor = (roundCost σ).bitXor ∧
      (roundCostTwoSlot σ).bitAnd = (roundCost σ).bitAnd ∧
      (roundCostTwoSlot σ).bitNot = (roundCost σ).bitNot ∧
      (roundCostTwoSlot σ).shift = (roundCost σ).shift ∧
      (roundCostTwoSlot σ).bitOr = (roundCost σ).bitOr := by
  rw [roundCostTwoSlot_fields, roundCost_fields]
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- **One schedule-extension step.** `schedule` (`Sha2.lean:87-88`) writes
`wt := sigma1 (W[t-2]) + W[t-7] + sigma0 (W[t-15]) + W[t-16]`, so one step is **four rotations** (two
per small sigma), two shifts, four XORs and four additions. -/
def schedStepCost (σ : R) : RoundCost σ :=
  RoundCost.rep σ 4 (rotReadCost σ) + sigma0Cost σ + sigma1Cost σ
    + RoundCost.rep σ 4 (wordAddCost σ)

theorem schedStepCost_fields (σ : R) :
    schedStepCost σ
      = { mul := 24, add := 16, sub := 0, neg := 0, cast := 0, channel := 4,
          bitXor := 4, bitAnd := 0, bitNot := 0, shift := 10, bitOr := 4 } := by
  rfl

/-- **The cost of a chain of `n` rounds.** Temporal `n` recursive steps, each one record addition;
spatial `n` stack frames for the recursive definition and O(1) result words. Counted in: the eleven
`RoundCost` fields. Derivation: the recurrence in `roundChainCost` plus `roundChainCost_rep` and
`RoundCost.rep_add`. -/
def roundChainCost (σ : R) : Nat → RoundCost σ
  | 0 => RoundCost.zero σ
  | n + 1 => roundChainCost σ n + roundCost σ

/-- The chain recurrence is the measure's own composition, so `n` rounds cost `n` copies of the
round. Derivation: induction on `Nat`, base `rfl` at zero, step by the definition and
`RoundCost.rep_succ`. -/
theorem roundChainCost_rep (σ : R) (n : Nat) :
    roundChainCost σ n = RoundCost.rep σ n (roundCost σ) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [roundChainCost, ih, RoundCost.rep_succ, RoundCost.composed_comm]

/-- **The chain closed form**, at an arbitrary chain length. -/
theorem roundChainCost_eq (σ : R) (n : Nat) :
    roundChainCost σ n
      = { mul := 36 * n, add := 25 * n, sub := 0, neg := 0, cast := 0, channel := 6 * n,
          bitXor := 8 * n, bitAnd := 5 * n, bitNot := n, shift := 12 * n, bitOr := 6 * n } := by
  rw [roundChainCost_rep]
  simp [RoundCost.rep, RoundCost.composed, roundCost, rotReadCost, ringMulCost, ringUpperCost,
    rotrCost, xorCost, andCost, notCost, wordAddCost, chooseCost, majorityCost, Nat.mul_comm,
    Nat.mul_one]

/-- **The 64-round chain, with the numbers substituted.** -/
theorem roundChain64 (σ : R) :
    roundChainCost σ 64
      = { mul := 2304, add := 1600, sub := 0, neg := 0, cast := 0, channel := 384,
          bitXor := 512, bitAnd := 320, bitNot := 64, shift := 768, bitOr := 384 } := by
  rw [roundChainCost_rep]
  simp [RoundCost.rep, RoundCost.composed, roundCost, rotReadCost, ringMulCost, ringUpperCost,
    rotrCost, xorCost, andCost, notCost, wordAddCost, chooseCost, majorityCost, Nat.mul_comm,
    Nat.mul_one]

/-- **Additivity over the chain**: the measure of the concatenation is the sum of the measures. -/
theorem roundChainCost_add (σ : R) (m n : Nat) :
    roundChainCost σ (m + n) = roundChainCost σ m + roundChainCost σ n := by
  rw [roundChainCost_rep, roundChainCost_rep, roundChainCost_rep, RoundCost.rep_add]

/-- **One block transformation cost.** `compress` (`Sha2.lean:96-120`) is sixty-four rounds **plus one
`schedule` pass of forty-eight steps** (`Sha2.lean:97`) **plus the eight-word add-back**
(`Sha2.lean:120`). Temporal O(64) rounds and O(48) schedule steps; spatial O(64) schedule words.
Counted in: the eleven `RoundCost` fields. Derivation: the recurrence `roundChainCost` plus the
iteration operator, closed by `rep_add`. -/
def blockCost (σ : R) : RoundCost σ :=
  roundChainCost σ 64 + RoundCost.rep σ 48 (schedStepCost σ) + RoundCost.rep σ 8 (wordAddCost σ)

/-- The block is the sum of its three parts, so a reader can see where each field comes from. -/
theorem blockCost_split (σ : R) :
    blockCost σ = roundChainCost σ 64 + RoundCost.rep σ 48 (schedStepCost σ)
      + RoundCost.rep σ 8 (wordAddCost σ) := rfl

/-- **The block's closed count**: 3456 ring multiplications, 2376 ring additions, 576 channel reads,
704 XORs, 320 ANDs, 64 NOTs, 1248 shifts, 576 disjunctions. -/
theorem blockCost_eq (σ : R) :
    blockCost σ
      = { mul := 3456, add := 2376, sub := 0, neg := 0, cast := 0, channel := 576,
          bitXor := 704, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576 } := by
  rw [blockCost_split, roundChain64]
  simp [RoundCost.rep, RoundCost.composed, schedStepCost, rotReadCost, ringMulCost, ringUpperCost,
    rotrCost, shrxCost, xorCost, andCost, notCost, wordAddCost, sigma0Cost, sigma1Cost]

/-- The block's ring layer alone, as AR-10's six-field record `(3456, 2376, 0, 0, 0, 576)`. -/
theorem blockCost_ringFields (σ : R) : (blockCost σ).ringFields = ⟨3456, 2376, 0, 0, 0, 576⟩ := by
  rw [blockCost_eq]
  rfl


/-! ## 12a. Audit rows for this part

Every headline owned by Part 1, with its axiom cone. The closed cone required is
{propext, Classical.choice, Quot.sound} with no `sorryAx`. The remaining rows of the original §12
sit in `AlgebraCostCompletePart2`, which imports this part. -/

#print axioms RoundCost.rep_add
#print axioms RingCost.rep_add
#print axioms ShaCost.rep_add
#print axioms ShaCost.rep_composed
#print axioms ShaCost.eq_of_fields
#print axioms RoundCost.eq_of_fields
#print axioms RingCost.eq_of_fields
#print axioms comp_add
#print axioms comp_rep
#print axioms ringFields_add'
#print axioms ringFields_rep'
#print axioms rotReadCost_fields
#print axioms roundCost_fields
#print axioms roundCostTwoSlot_fields
#print axioms roundCost_twoSlot_add_only
#print axioms schedStepCost_fields
#print axioms roundChainCost_eq
#print axioms roundChain64
#print axioms roundChainCost_add
#print axioms blockCost_eq
#print axioms blockCost_ringFields
