/-
First half of the SHA-256 proof-of-work cost comparison, namespace `algebra_cost_pow_compare`
(second half: `Icarus.Cost.AlgebraCostPowComparePart2`). This part carries sections 0 to 6 and
their audit rows: the width and the three dials, the eleven-field `RoundCost`, the six-field
`RingCost`, the per-operation counts, the round, schedule and block chains, the SHA-256d block
composition, and the attempt and range figures in the eleven-field measure. Part2 carries
sections 7 to 9: the twenty-three-field `ShaCost`, the padding, loader and store costs, the
complete attempt and range, the source-bound corrections, the three faces and their audit rows.
No section 0 to 6 declaration mentions a `ShaCost`.

Figures stated: one SHA-256d proof-of-work attempt is the inner hash of the 80-byte header (two
padded blocks, `blockCount 80 = 2`) plus the outer hash of the 32-byte digest (one padded block,
`blockCount 32 = 1`), three block transformations; the 2^32 nonce-space range as a closed form
with its recurrence; the ranked components (the 48-step message schedule including its 96 SHRs
per block, the 64 rounds, the eight-word add-back, the padding bytes, the block load and the
digest store) with their shares; and the face verdict. Two measures are carried side by side and
never merged: the eleven-field `RoundCost` (operations of the canonical bodies only) and the
twenty-three-field `ShaCost` (the eleven fields plus the byte, `Nat`-operation and read counts of
`pad`, the sixteen-word loader and the eight-word store, with the source-decided corrections
carried as a separate named record).

Depends on: `Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` and
`Apeiron.Trichotomy.ChannelsAnyRing.{ringUpper,ringUpper_mul}` (pinned canonical corpus revision 880aa57e47e782ed7a9d9ee1974cfc87360b446b, `proved`
there); the canonical bodies `Apeiron.Crypto.Sha2.{rotr,shrx,sigma0,sigma1,Sigma0,Sigma1,choose,
majority,schedule,compress,lengthByte,lengthBlock,pad,hash}`; the loaders
`apeiron_word.{b64,beWord16,bytesToWords}` in `Crypto.Sha256Certificate`; the extracted side
`WordRefinement` (`RingCell (ZMod (2^64))`) and `Denotation.denote_wrapping_add` (width 64
only); the extracted store `sha256.sha256_loop2.body` in `ApeironWord`, whose occurrence
inventory is `model`. The measures and counts follow `Icarus.Cost.AlgebraCostRound` (`RoundCost`,
`roundCost`, `schedStepCost`, `blockCost`, `RotRead`), `Icarus.Cost.AlgebraCost` (`RingCost`,
`rep_add`), `Icarus.Cost.AlgebraCostPow` (`attemptCost`, `rangeCost`, `blockCount`, `canonTotal`,
`zeroFill`), `Icarus.Cost.AlgebraCostCompletePart1` and `Icarus.Cost.AlgebraCostCompletePart2`
(`ShaCost`, `padCost`, `loadCost`, `storeCost`, `blockFullCost`), `Icarus.Cost.AlgebraCostSource`
(source-bound counts, with three divergences: add 2328 vs 2376, bitXor 640 vs 704,
schedWordWrite 64 vs 16; the carry-stage model and the wiring model) and
`Icarus.Cost.AlgebraCostKeccak` (the Keccak and SHA3-256 comparison rows). The padding
arithmetic `pad_stateToBytes_length`, `hash_outer_eq_compress_H0`, `sha256d_eq_outer_compress_H0`
(canonical SHA-256d composition) and `canonTotal`, `zeroFill`, `totalLayout`, `pad_length_total`
(padding layout) are the arithmetic this module prices.
Excludes: latency, wall-clock time, throughput, hash rate, bytes per second, energy, joules,
watts, hardware, ASIC or GPU figures, miner economics, profitability, USD, compiler or codegen
cost, the kernel ABI, the `unsafe` container, device dispatch cost, constant-time and security
properties of any hash, and every asymptotic statement. A cost-model proof is not a hardware
speed measurement; this module states counts and nothing else.
Grade: `proved` (kernel-checked, axiom cone {propext, Classical.choice, Quot.sound}, no
`sorryAx`, `#print axioms` rows at the end of the module) for every count read off a canonical
definition body by unfolding it, the padding-length arithmetic and its join to `pad`, the
SHA-256d block composition (2 + 1 = 3), the closed forms of the attempt and the 2^32 range in
both measures, the component decomposition, shares and ranking, the additivity laws and the
face comparisons. `model` for the readings no machine-checked notion of operation count of a
term can discharge: `RotRead` and `AddRead` (candidate C5 and the width-32 additive reading), the
count of the seven `Word` additions of the round body, the loader and store occurrence
inventories, and the ripple carry-stage expansion. `tested`: none.
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

namespace algebra_cost_pow_compare

open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
open apeiron_word

variable {R : Type u} [CommRing R]

/-! ## 0. The width and the three dials

The SHA-2 word is `BitVec 32` (`Sha2.lean:22`), so the ring the SHA-2 figures are read at is
`ZMod (2 ^ 32)`; the extracted `Cell<i64>` arithmetic runs at width 64 (`WordRefinement.lean:38`).
The three dial values of `cell.Sigma` at width 32 are `denoteSigma`, `WordRefinement.lean:29-35`:
elliptic `-1`, parabolic `0`, hyperbolic `1`. -/

/-- The ring the SHA-2 figures are measured in: `w = 32`. -/
abbrev W32 := ZMod (2 ^ 32)

/-- The elliptic face's dial value. -/
abbrev faceElliptic : W32 := -1
/-- The parabolic face's dial value. -/
abbrev faceParabolic : W32 := 0
/-- The hyperbolic face's dial value. -/
abbrev faceHyperbolic : W32 := 1

/-! ## 1. The measures

Two records are re-declared here because `work/proofs` is not a root of any corpus `lean_lib`:
AR-15/AR-20's eleven-field `RoundCost σ` and AR-10's six-field `RingCost σ`, with their composition
laws. AR-27's twenty-three-field `ShaCost σ` is re-declared in §7. Every figure in this module is
one of these records; the per-measure numbers are kept separate and no byte field is ever summed
with an operation field. -/

/-! ### 1.1 The eleven counted quantities -/

/-- `RoundCost σ`: the measure. Eleven fields, one counted quantity each - the six ring operations
of `R[x]/(x^2 - sigma)` that AR-10's `RingCost` records, and the five word operations that the
two-slot algebra has no reading for at word width. `σ` is the dial the ring fields are read at; §8
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
  /-- bitwise XOR -/
  bitXor : Nat
  /-- bitwise AND -/
  bitAnd : Nat
  /-- bitwise NOT -/
  bitNot : Nat
  /-- logical shifts, left or right -/
  shift : Nat
  /-- bitwise OR -/
  bitOr : Nat
  deriving DecidableEq, Repr

/-- The record of a computation that performs no operation of either kind. -/
def RoundCost.zero (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **Dependent composition.** A computation that runs after another performs both sets of
operations, so every count totals. Temporal and spatial O(1): a closed term of eleven `Nat`
additions, no recursion, no allocation. Counted in: nothing - the measure's own operator.
Derivation: eleven independent sums, one per field. -/
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
compared on it even though their records carry different indices. Temporal and spatial O(1): a
tuple of eleven projections. -/
def RoundCost.vector {σ : R} (c : RoundCost σ) :
    Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat :=
  (c.mul, c.add, c.sub, c.neg, c.cast, c.channel, c.bitXor, c.bitAnd, c.bitNot, c.shift, c.bitOr)

/-- **`rep σ n a` is the measure of `n` dependent copies of an operation of cost `a`.** Temporal
O(1) and spatial O(1): eleven `Nat` multiplications on an eleven-field record. Counted in:
nothing - the iteration operator of the measure; the `n` it is given is a count of repetitions, so
the value grows while the function does not recur. Derivation: fieldwise `n * a.field`, and
`RoundCost.rep_add` proves this image additive. -/
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

/-- **THE COMPOSITION LAW over an iterated construction, eleven fields.** `n + m` copies cost the
sum of `n` copies and `m` copies, field by field. Every range figure of §6 and §7 is this operator.
-/
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

/-! ### 1.2 AR-10's six-field ring measure -/

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
  deriving DecidableEq, Repr

/-- The record of a computation that performs no ring operation. -/
def RingCost.zero (σ : R) : RingCost σ := ⟨0, 0, 0, 0, 0, 0⟩

/-- Dependent composition over the six ring fields. Temporal and spatial O(1): six `Nat` additions.
-/
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

/-- `n` dependent copies of a ring cost. Temporal and spatial O(1): six `Nat` multiplications. -/
def RingCost.rep (σ : R) (n : Nat) (a : RingCost σ) : RingCost σ :=
  ⟨n * a.mul, n * a.add, n * a.sub, n * a.neg, n * a.cast, n * a.channel⟩

/-- **THE COMPOSITION LAW over an iterated construction, six fields.** AR-10's `RingCost.rep_add`,
re-proved here. -/
theorem RingCost.rep_add (σ : R) (n m : Nat) (a : RingCost σ) :
    RingCost.rep σ (n + m) a = RingCost.rep σ n a + RingCost.rep σ m a := by
  cases a
  simp [RingCost.rep, RingCost.composed, Nat.add_mul]

/-- **The ring projection.** The first six fields of `RoundCost σ` read as AR-10's `RingCost σ`.
Temporal and spatial O(1): six projections, no allocation. -/
def RoundCost.ringFields {σ : R} (c : RoundCost σ) : RingCost σ :=
  ⟨c.mul, c.add, c.sub, c.neg, c.cast, c.channel⟩

/-- **The projection commutes with composition.** -/
theorem ringFields_add (σ : R) (a b : RoundCost σ) :
    (a + b).ringFields = a.ringFields + b.ringFields := by
  cases a
  cases b
  rfl

/-- **The projection commutes with the iteration operator**, so every eleven-field figure inherits
a six-field one: `rep n` of the eleven is `rep n` of the six. -/
theorem ringFields_rep (σ : R) (n : Nat) (a : RoundCost σ) :
    (RoundCost.rep σ n a).ringFields = RingCost.rep σ n a.ringFields := by
  cases a
  rfl

/-! ## 2. The per-operation counts, re-derived from the canonical bodies

Every count below is read off a displayed definition body and the display is `rfl`-checked in this
module. A count is a reading of a displayed term: nothing in Lean counts the operations of a term,
which is AR-15's residual item 1 and this module's receipt residual 1. -/

/-- **The canonical multiplication** `(a + bx)(c + dx) = (ac + sigma*bd) + (ad + bc)x`: five
multiplications and two additions. -/
def ringMulCost (σ : R) : RoundCost σ :=
  { mul := 5, add := 2, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The canonical channel read** `z.1 + r*z.2`: one multiplication, one addition, one
application counted in `channel`. -/
def ringUpperCost (σ : R) : RoundCost σ :=
  { mul := 1, add := 1, sub := 0, neg := 0, cast := 0, channel := 1,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The rotation reading**, `ringUpper 1 (ringMul 1 (x,0) (2^n, 2^(32-n)))`: one canonical
multiplication followed by one channel read. Its *count* is a fact about two bodies and is
unconditional; its *identification* with a word rotation is candidate C5 at grade `model` and is
carried in the receipt, never as an axiom. -/
def rotReadCost (σ : R) : RoundCost σ := ringMulCost σ + ringUpperCost σ

/-- `ROTR`: two shifts and one disjunction. A rotation is not one operation in the canonical body.
-/
def rotrCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 2, bitOr := 1 }

/-- `SHR`: one shift and nothing else. -/
def shrxCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 1, bitOr := 0 }

/-- `σ₀(x) = ROTR^7 x ⊕ ROTR^18 x ⊕ SHR^3 x`: two rotations, one shift, two XORs. -/
def sigma0Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 5, bitOr := 2 }

/-- `σ₁(x) = ROTR^17 x ⊕ ROTR^19 x ⊕ SHR^10 x`: the same inventory. -/
def sigma1Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 5, bitOr := 2 }

/-- `Σ₀(x) = ROTR^2 x ⊕ ROTR^13 x ⊕ ROTR^22 x`: three rotations and two XORs, **no shift by
itself** - the `SHR` terms of FIPS 180-4 §4.1.2 belong to `σ₀` and `σ₁`, which the message schedule
uses and the round function does not. -/
def bigSigma0Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 6, bitOr := 3 }

/-- `Σ₁(x) = ROTR^6 x ⊕ ROTR^11 x ⊕ ROTR^25 x`: the same inventory as `Σ₀`. -/
def bigSigma1Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 6, bitOr := 3 }

/-- `Ch(x, y, z) = (x &&& y) ⊕ (¬x &&& z)`: two ANDs, one NOT, two XORs. The two-XOR count is
AR-15's reading of this body; AR-41's source-bound count reads one XOR (the two operands `x &&& y`
and `~~~x &&& z` joined by one `^^^`), and both readings are carried separately in the receipt. -/
def chooseCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 2, bitNot := 1, shift := 0, bitOr := 0 }

/-- `Maj(x, y, z) = (x &&& y) ⊕ (x &&& z) ⊕ (y &&& z)`: three ANDs and two XORs, no NOT. -/
def majorityCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 3, bitNot := 0, shift := 0, bitOr := 0 }

/-- **One `Word` addition.** A word addition mod `2^32` is one addition of `ZMod (2^32)`; that
identification is the named premise `AddRead`, at grade `model`, template `Denotation.lean:35`
proved at width 64 only. `cast` is zero because the canonical body holds its words as `Word`, not
as a dial enum variant. -/
def wordAddCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 1, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-! ### 2.1 The bodies the counts are read off -/

/-- Unfolding `ringMul` exhibits the expression whose five multiplications and two additions
`ringMulCost` names. -/
theorem ringMul_body_ops (σ : R) (z w : RingCell R) :
    ringMul σ z w = (z.1 * w.1 + σ * z.2 * w.2, z.1 * w.2 + z.2 * w.1) := rfl

/-- Unfolding the channel read exhibits the one multiplication and one addition `ringUpperCost`
names. -/
theorem ringUpper_body_ops (r : R) (z : RingCell R) : ringUpper r z = z.1 + r * z.2 := rfl

/-- The rotation reading's count, read off the two canonical bodies: six multiplications, three
additions, one channel read. -/
theorem rotReadCost_fields (σ : R) :
    rotReadCost σ =
      { mul := 6, add := 3, sub := 0, neg := 0, cast := 0, channel := 1,
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

/-- **The shifts of a round come from its rotations; the two message-schedule shifts come from the
message schedule.** `Σ₀` and `Σ₁` are three rotations each and a `ROTR` is two shifts, so a
round's `shift` field is exactly `2 * 3 * 2 = 12`. `σ₀` and `σ₁` add one shift each, which is why a
schedule step has five shifts per sigma rather than four. -/
theorem shifts_come_from_rotations (σ : R) :
    (bigSigma0Cost σ).shift = 3 * (rotrCost σ).shift ∧
      (bigSigma1Cost σ).shift = 3 * (rotrCost σ).shift ∧
      (sigma0Cost σ).shift = 2 * (rotrCost σ).shift + (shrxCost σ).shift ∧
      (sigma1Cost σ).shift = 2 * (rotrCost σ).shift + (shrxCost σ).shift := by
  unfold bigSigma0Cost bigSigma1Cost sigma0Cost sigma1Cost rotrCost shrxCost
  exact ⟨rfl, rfl, rfl, rfl⟩

/-! ## 3. The round, the schedule step and one block

The block figure is the composition of three components: the 64-round chain, the 48-step schedule
chain, and the eight-word add-back. They are defined as their own chains so that §4 can rank them
field by field instead of reporting one number. -/

/-- **One round** of `compress` (`Sha2.lean:100-119`): six rotations under the C5 reading, the four
round-function bodies, and seven word additions - four in `t1` (`:108`), one in `t2` (`:109`) and
two in the state update (`:112`, `:116`). The element reads `acc.getD i 0`, `K.getD t 0`,
`w.getD t 0` cost nothing: `RoundCost` has no read field. Temporal and spatial O(1): a closed term
of ten record compositions, no recursion. Derivation: the per-operation inventory of §2 composed
once per occurrence. -/
def roundCost (σ : R) : RoundCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ +
    bigSigma0Cost σ + bigSigma1Cost σ + chooseCost σ + majorityCost σ +
    wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ +
    wordAddCost σ

/-- **The round's closed count**: 36 multiplications, 25 additions, 6 channel reads, 8 XORs, 5
ANDs, 1 NOT, 12 shifts, 6 disjunctions, and no subtraction, no negation, no dial conversion. -/
theorem roundCost_fields (σ : R) :
    roundCost σ =
      { mul := 36, add := 25, sub := 0, neg := 0, cast := 0, channel := 6,
        bitXor := 8, bitAnd := 5, bitNot := 1, shift := 12, bitOr := 6 } := by
  simp [roundCost, rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed, bigSigma0Cost,
    bigSigma1Cost, chooseCost, majorityCost, wordAddCost]

/-- The round's ring layer alone, as AR-10's six-field record: `(36, 25, 0, 0, 0, 6)`. -/
theorem roundCost_ringFields (σ : R) : (roundCost σ).ringFields = ⟨36, 25, 0, 0, 0, 6⟩ := by
  rw [roundCost_fields]
  rfl

/-- **One schedule-extension step.** `schedule` (`Sha2.lean:82-89`) extends the block by one step
per `t = 16 … 63`, that is **48 steps per block**, each
`W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]`: four rotations, two small sigmas, four word
additions. Temporal and spatial O(1): a closed term of nine record compositions. -/
def schedStepCost (σ : R) : RoundCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + sigma0Cost σ + sigma1Cost σ +
    wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ

/-- **The schedule step's closed count**: 24 multiplications, 16 additions, 4 channel reads, 4
XORs, 10 shifts, 4 disjunctions. -/
theorem schedStepCost_fields (σ : R) :
    schedStepCost σ =
      { mul := 24, add := 16, sub := 0, neg := 0, cast := 0, channel := 4,
        bitXor := 4, bitAnd := 0, bitNot := 0, shift := 10, bitOr := 4 } := by
  simp [schedStepCost, rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed, sigma0Cost,
    sigma1Cost, wordAddCost]

/-- **The eight-word add-back** (`Sha2.lean:120`): eight `Word` additions. -/
def addBackCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 8, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The cost of a chain of `n` rounds.** Temporal `n` recursive steps, each one record addition;
spatial one accumulator and one stack frame, O(1) words beyond the result record. Counted in: the
eleven `RoundCost` fields. Derivation: the recurrence `c 0 = zero σ`,
`c (n + 1) = c n + roundCost σ`, one round per step; it closes because `RoundCost.rep_add`
telescopes it to `n` copies of the round and `Nat` induction has no open base case. -/
def roundChainCost (σ : R) : Nat → RoundCost σ
  | 0 => RoundCost.zero σ
  | n + 1 => roundChainCost σ n + roundCost σ

theorem roundChainCost_rep (σ : R) (n : Nat) :
    roundChainCost σ n = RoundCost.rep σ n (roundCost σ) := by
  induction n with
  | zero => simp [roundChainCost, RoundCost.rep_zero, RoundCost.zero]
  | succ n ih => rw [roundChainCost, ih, RoundCost.rep_succ, RoundCost.composed_comm]

/-- The 64-round chain as its own name. -/
def roundChain64Cost (σ : R) : RoundCost σ := roundChainCost σ 64

/-- **The 64-round chain, with the numbers substituted**: 2304 multiplications, 1600 additions, 384
channel reads, 512 XORs, 320 ANDs, 64 NOTs, 768 shifts, 384 disjunctions. -/
theorem roundChain64Cost_eq (σ : R) :
    roundChain64Cost σ =
      { mul := 2304, add := 1600, sub := 0, neg := 0, cast := 0, channel := 384,
        bitXor := 512, bitAnd := 320, bitNot := 64, shift := 768, bitOr := 384 } := by
  rw [roundChain64Cost, roundChainCost_rep, roundCost_fields, RoundCost.rep]
  apply RoundCost.eq_of_fields <;> norm_num

/-- **The cost of a chain of `n` schedule steps**, the same recurrence one step over. -/
def schedChainCost (σ : R) : Nat → RoundCost σ
  | 0 => RoundCost.zero σ
  | n + 1 => schedChainCost σ n + schedStepCost σ

theorem schedChainCost_rep (σ : R) (n : Nat) :
    schedChainCost σ n = RoundCost.rep σ n (schedStepCost σ) := by
  induction n with
  | zero => simp [schedChainCost, RoundCost.rep_zero, RoundCost.zero]
  | succ n ih => rw [schedChainCost, ih, RoundCost.rep_succ, RoundCost.composed_comm]

/-- The 48-step schedule chain as its own name. -/
def schedChain48Cost (σ : R) : RoundCost σ := schedChainCost σ 48

/-- **The 48-step schedule chain, with the numbers substituted**: 1152 multiplications, 768
additions, 192 channel reads, 192 XORs, 480 shifts, 192 disjunctions. -/
theorem schedChain48Cost_eq (σ : R) :
    schedChain48Cost σ =
      { mul := 1152, add := 768, sub := 0, neg := 0, cast := 0, channel := 192,
        bitXor := 192, bitAnd := 0, bitNot := 0, shift := 480, bitOr := 192 } := by
  rw [schedChain48Cost, schedChainCost_rep, schedStepCost_fields, RoundCost.rep]
  apply RoundCost.eq_of_fields <;> norm_num

/-- **One block transformation cost.** `compress` (`Sha2.lean:96-120`) is sixty-four rounds **plus
one `schedule` pass of forty-eight steps** (`:97`) **plus the eight-word add-back** (`:120`).
Temporal and spatial O(1): a closed term of three record compositions, no recursion. Counted in:
the eleven `RoundCost` fields. -/
def blockCost (σ : R) : RoundCost σ :=
  roundChain64Cost σ + schedChain48Cost σ + addBackCost σ

/-- **The block splits into exactly its three priced components.** -/
theorem blockCost_split (σ : R) :
    blockCost σ = roundChain64Cost σ + schedChain48Cost σ + addBackCost σ := rfl

/-- **The block's closed count**: 3456 ring multiplications, 2376 ring additions, 576 channel
reads, 704 XORs, 320 ANDs, 64 NOTs, 1248 shifts and 576 disjunctions, and no subtraction, no
negation and no dial conversion. -/
theorem blockCost_eq (σ : R) :
    blockCost σ =
      { mul := 3456, add := 2376, sub := 0, neg := 0, cast := 0, channel := 576,
        bitXor := 704, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576 } := by
  have h1 := roundChain64Cost_eq σ
  have h2 := schedChain48Cost_eq σ
  simp only [blockCost, RoundCost.composed_apply, RoundCost.composed]
  simp [h1, h2, addBackCost]

/-- **The block's ring layer alone**, as AR-10's six-field record `(3456, 2376, 0, 0, 0, 576)`. -/
theorem blockCost_ringFields (σ : R) : (blockCost σ).ringFields = ⟨3456, 2376, 0, 0, 0, 576⟩ := by
  rw [blockCost_eq]
  rfl

/-! ## 4. Where the work concentrates: the components, ranked

The three components of a block, and the two sub-components of the schedule, are ranked by their
counts in each field of the measure. No timing, no hardware: the ranking is over the `Nat` fields
proved in §3. -/

/-- **The 96 SHRs.** Each schedule step carries the two small sigmas, and each small sigma body
(`sigma0_body_ops`, `sigma1_body_ops`) contains exactly one `SHR`; `SHR` costs one shift
(`shrx_body_ops`), so a 64-byte block has `48 * 2 = 96` shift-by-a-constant operations from the
schedule's `SHR` terms. This is the quantity AR-41's source-bound `shift` field prices alongside
the rotations; it is not a separate field of this measure. -/
theorem schedShr96 (σ : R) :
    48 * ((shrxCost σ).shift + (shrxCost σ).shift) = 96 := by
  simp [shrxCost]

/-- **The schedule's shift field splits into 384 rotation shifts and the 96 SHRs**: four rotations
per step at two shifts each, plus two `SHR` terms at one shift each, over 48 steps. -/
theorem schedShift_split (σ : R) :
    (schedChain48Cost σ).shift
      = 48 * (4 * (rotrCost σ).shift) + 48 * ((shrxCost σ).shift + (shrxCost σ).shift) := by
  rw [schedChain48Cost_eq, rotrCost, shrxCost]
  norm_num

/-- **The rounds have no `SHR`:** both big sigmas are three rotations each, so every shift of a
round comes from a rotation and the schedule owns all 96 `SHR` operations of a block. -/
theorem roundsHaveNoShr (σ : R) :
    (bigSigma0Cost σ).shift = 3 * (rotrCost σ).shift ∧
      (bigSigma1Cost σ).shift = 3 * (rotrCost σ).shift ∧ (shrxCost σ).shift = 1 := by
  unfold bigSigma0Cost bigSigma1Cost rotrCost shrxCost
  exact ⟨rfl, rfl, rfl⟩

/-- **The per-field decomposition of a block**, each field the sum of its three components. -/
theorem blockCost_component_fields (σ : R) :
    (blockCost σ).mul = (roundChain64Cost σ).mul + (schedChain48Cost σ).mul + (addBackCost σ).mul ∧
      (blockCost σ).add = (roundChain64Cost σ).add + (schedChain48Cost σ).add +
        (addBackCost σ).add ∧
      (blockCost σ).channel = (roundChain64Cost σ).channel + (schedChain48Cost σ).channel ∧
      (blockCost σ).bitXor = (roundChain64Cost σ).bitXor + (schedChain48Cost σ).bitXor ∧
      (blockCost σ).bitAnd = (roundChain64Cost σ).bitAnd ∧
      (blockCost σ).bitNot = (roundChain64Cost σ).bitNot ∧
      (blockCost σ).shift = (roundChain64Cost σ).shift + (schedChain48Cost σ).shift ∧
      (blockCost σ).bitOr = (roundChain64Cost σ).bitOr + (schedChain48Cost σ).bitOr := by
  have h1 := roundChain64Cost_eq σ
  have h2 := schedChain48Cost_eq σ
  simp only [blockCost, RoundCost.composed_apply, RoundCost.composed]
  simp [h1, h2, addBackCost]

/-- **THE RANKING.** In every field where more than one component is nonzero the 64 rounds dominate
the 48 schedule steps, which dominate the eight-word add-back; multiplication, channel reads and
disjunctions split 2/3 to 1/3, additions split 200/297 to 96/297 plus 1/297, shifts split 8/13 to
5/13, and XORs split 8/11 to 3/11; ANDs and NOTs are entirely in the rounds. -/
theorem blockCost_ranking (σ : R) :
    (schedChain48Cost σ).mul < (roundChain64Cost σ).mul ∧
      (roundChain64Cost σ).mul < (blockCost σ).mul ∧
      (schedChain48Cost σ).add < (roundChain64Cost σ).add ∧
      (roundChain64Cost σ).add < (blockCost σ).add ∧
      (schedChain48Cost σ).channel < (roundChain64Cost σ).channel ∧
      (roundChain64Cost σ).channel < (blockCost σ).channel ∧
      (schedChain48Cost σ).bitXor < (roundChain64Cost σ).bitXor ∧
      (roundChain64Cost σ).bitXor < (blockCost σ).bitXor ∧
      (schedChain48Cost σ).shift < (roundChain64Cost σ).shift ∧
      (roundChain64Cost σ).shift < (blockCost σ).shift ∧
      (schedChain48Cost σ).bitOr < (roundChain64Cost σ).bitOr ∧
      (roundChain64Cost σ).bitOr < (blockCost σ).bitOr ∧
      (schedChain48Cost σ).bitAnd = 0 ∧ (roundChain64Cost σ).bitAnd = (blockCost σ).bitAnd ∧
      (schedChain48Cost σ).bitNot = 0 ∧ (roundChain64Cost σ).bitNot = (blockCost σ).bitNot := by
  rw [roundChain64Cost_eq, schedChain48Cost_eq, blockCost_eq]
  norm_num

/-- **The shares, exactly.** Each identity is the rank statement's arithmetic in lowest terms: the
rounds are `2` of `3` parts of the multiplication, channel and disjunction fields, `8` of `11` of
the XOR field, `8` of `13` of the shift field; the schedule is the complement; the add-back is the
single remaining addition of the 297. -/
theorem blockCost_shares (σ : R) :
    (roundChain64Cost σ).mul * 3 = (blockCost σ).mul * 2 ∧
      (schedChain48Cost σ).mul * 3 = (blockCost σ).mul ∧
      (roundChain64Cost σ).bitXor * 11 = (blockCost σ).bitXor * 8 ∧
      (schedChain48Cost σ).bitXor * 11 = (blockCost σ).bitXor * 3 ∧
      (roundChain64Cost σ).shift * 13 = (blockCost σ).shift * 8 ∧
      (schedChain48Cost σ).shift * 13 = (blockCost σ).shift * 5 ∧
      (roundChain64Cost σ).add * 297 = (blockCost σ).add * 200 ∧
      (schedChain48Cost σ).add * 297 = (blockCost σ).add * 96 ∧
      (addBackCost σ).add * 297 = (blockCost σ).add ∧
      (roundChain64Cost σ).channel * 3 = (blockCost σ).channel * 2 ∧
      (roundChain64Cost σ).bitOr * 3 = (blockCost σ).bitOr * 2 := by
  rw [roundChain64Cost_eq, schedChain48Cost_eq, blockCost_eq, addBackCost]
  norm_num

/-! ## 5. The SHA-256d block composition: two inner blocks, one outer block

The canonical side is AR-40 (`AlgebraSha256dCompose.lean`): `Sha2.hash` folds one `compress` per
padded block, an 80-byte message pads to two blocks, and `pad (stateToBytes s)` is one 64-byte
block for every digest state, so the double application is three block transformations: two inner
and one outer (`sha256d_eq_outer_compress_H0`). This section re-derives the block counts from
`pad`'s own body so the composition arithmetic is kernel-checked here. -/

/-- The padded length as a `Nat` function of the message length, read off `pad`'s own `let`. -/
def padLength (n : Nat) : Nat :=
  let r := (n + 9) % 64
  if r = 0 then n + 9 else n + 9 + (64 - r)

/-- The zero-fill byte count of a message of `n` bytes: the `List.replicate` argument of `pad`. -/
def padZeroFill (n : Nat) : Nat := padLength n - n - 9

/-- **`pad`'s four pieces, by `rfl`**: the message, one `0x80` merge byte, the zero fill, and the
eight length bytes. -/
theorem pad_spec (m : List (BitVec 8)) :
    pad m = m ++ [0x80] ++ List.replicate (padZeroFill m.length) 0 ++ lengthBlock m.length := rfl

/-- The eight-byte length block has eight bytes. -/
theorem lengthBlock_len (n : Nat) : (lengthBlock n).length = 8 := by
  unfold lengthBlock
  simp

/-- **The layout identity**: message bytes, the `0x80` merge byte, the zero fill and the eight
length bytes total the padded length. -/
theorem pad_layout (n : Nat) : n + 1 + padZeroFill n + 8 = padLength n := by
  unfold padZeroFill
  have hge : n + 9 ≤ padLength n := by
    simp only [padLength]
    split <;> omega
  omega

/-- **The padded length of a message is `padLength` of its byte length**, read off the body of
`pad`. -/
theorem pad_length (m : List (BitVec 8)) : (pad m).length = padLength m.length := by
  rw [pad_spec]
  simp only [List.length_append, List.length_cons, List.length_nil, List.length_replicate,
    lengthBlock_len]
  have h := pad_layout m.length
  omega

/-- **The header pads to 128 bytes.** 80 message bytes, one `0x80` byte, 39 zero bytes and eight
length bytes. -/
theorem padLength_80 : padLength 80 = 128 := by decide

/-- **The zero fill of the header is 39 bytes.** -/
theorem padZeroFill_80 : padZeroFill 80 = 39 := by decide

/-- **The digest pads to 64 bytes.** 32 message bytes, one `0x80` byte, 23 zero bytes and eight
length bytes; the padded digest is exactly one block. -/
theorem padLength_32 : padLength 32 = 64 := by decide

/-- **The zero fill of the 32-byte digest is 23 bytes.** -/
theorem padZeroFill_32 : padZeroFill 32 = 23 := by decide

/-- **THE INNER BLOCK COUNT: 2.** For every 80-byte message, `pad m` has 128 bytes, so `hash`'s
fold runs `(pad m).length / 64 = 2` compressions. -/
theorem header_pad_length (m : List (BitVec 8)) (hm : m.length = 80) :
    (pad m).length = 128 := by
  rw [pad_length, hm, padLength_80]

theorem header_pad_blocks (m : List (BitVec 8)) (hm : m.length = 80) :
    (pad m).length / 64 = 2 := by
  rw [header_pad_length m hm]

/-- **THE OUTER BLOCK COUNT: 1.** For every 32-byte message - the inner digest, by AR-40's
`pad_stateToBytes_length` - `pad` has 64 bytes, one block. -/
theorem digest_pad_length (d : List (BitVec 8)) (hd : d.length = 32) :
    (pad d).length = 64 := by
  rw [pad_length, hd, padLength_32]

theorem digest_pad_blocks (d : List (BitVec 8)) (hd : d.length = 32) :
    (pad d).length / 64 = 1 := by
  rw [digest_pad_length d hd]

/-- **THE SHA-256d BLOCK COMPOSITION: 2 + 1 = 3.** -/
theorem sha256d_block_count :
    (2 : Nat) + 1 = 3 := by norm_num

/-- **The hash fold count of the header is two.** `hash` (`Sha2.lean:146-148`) folds over
`List.range ((pad m).length / 64)`, one `compress` per element; the fold's length is the block
count. -/
theorem header_hash_fold_count (m : List (BitVec 8)) (hm : m.length = 80) :
    (List.range ((pad m).length / 64)).length = 2 := by
  rw [List.length_range, header_pad_blocks m hm]

/-- **The nonce's 4 bytes lie in the second block**: bytes 76–79 of the header are inside the byte
range `[64, 128)` of the padded header's second 64-byte block. A statement about positions, not a
schedule: the measure is content- and position-blind. -/
theorem nonce_bytes_second_block : 64 * 1 ≤ 76 ∧ 76 + 4 ≤ 64 * 2 := by norm_num

/-- The 80-byte header length this module prices. -/
def headerBytes : Nat := 80

/-- The 32-byte digest length of the outer input. -/
def digestBytes : Nat := 32

/-- The size of a 32-bit nonce space: `2 ^ 32` attempts. -/
def nonceSpace : Nat := 4294967296

/-! ## 6. One SHA-256d attempt and the nonce space, in the eleven-field measure

A SHA-256d attempt is three block transformations: the two inner blocks of the header and the one
outer block of the digest. The padding's list operations, the block load and the digest store are
**invisible** in this measure (its residual item 4, AR-20 §9); they are priced in §7 in AR-27's
twenty-three-field measure instead, and the two figures are never summed. -/

/-- **One SHA-256d proof-of-work attempt**: three block transformations. Temporal and spatial O(1):
a closed term of one `RoundCost.rep`. Counted in: the eleven `RoundCost` fields. Derivation: two
inner blocks (`header_pad_blocks`) plus one outer block (`digest_pad_blocks`), each costing
`blockCost σ`. -/
def sha256dAttemptCost (σ : R) : RoundCost σ := RoundCost.rep σ 3 (blockCost σ)

/-- **The attempt splits into the inner two blocks and the outer one**, `2 + 1 = 3`; the split is
`RoundCost.rep_add` and exposes that the composition hides no extra block. -/
theorem sha256dAttemptCost_inner_outer (σ : R) :
    sha256dAttemptCost σ =
      RoundCost.rep σ 2 (blockCost σ) + RoundCost.rep σ 1 (blockCost σ) := by
  unfold sha256dAttemptCost
  rw [← RoundCost.rep_add]

/-- **THE SHA-256d ATTEMPT FIGURE.** Three times one block: 10368 ring multiplications, 7128 ring
additions, 1728 channel reads, 2112 XORs, 960 ANDs, 192 NOTs, 3744 shifts and 1728 disjunctions;
no subtraction, no negation, no dial conversion. These are counts of algebra and word operations
in the measure, not seconds, joules or attempts per second. -/
theorem sha256dAttemptCost_eq (σ : R) :
    sha256dAttemptCost σ =
      { mul := 10368, add := 7128, sub := 0, neg := 0, cast := 0, channel := 1728,
        bitXor := 2112, bitAnd := 960, bitNot := 192, shift := 3744, bitOr := 1728 } := by
  rw [sha256dAttemptCost, blockCost_eq, RoundCost.rep]
  apply RoundCost.eq_of_fields <;> norm_num

/-- **The attempt's ring layer alone**, as AR-10's six-field record
`(10368, 7128, 0, 0, 0, 1728)`. -/
theorem sha256dAttemptCost_ringFields (σ : R) :
    (sha256dAttemptCost σ).ringFields = ⟨10368, 7128, 0, 0, 0, 1728⟩ := by
  rw [sha256dAttemptCost_eq]
  rfl

/-- **The cost of a nonce range of `m` attempts** against one header, every attempt paying its
whole three-block body. Temporal and spatial O(1): a closed term of one `RoundCost.rep`. Counted
in: the eleven fields. Derivation: `RoundCost.rep` at the attempt; its additivity is
`RoundCost.rep_add` instantiated below. -/
def sha256dRangeCost (σ : R) (m : Nat) : RoundCost σ := RoundCost.rep σ m (sha256dAttemptCost σ)

/-- **The range closed form at an arbitrary range length.** -/
theorem sha256dRangeCost_eq (σ : R) (m : Nat) :
    sha256dRangeCost σ m =
      { mul := m * 10368, add := m * 7128, sub := 0, neg := 0, cast := 0,
        channel := m * 1728, bitXor := m * 2112, bitAnd := m * 960, bitNot := m * 192,
        shift := m * 3744, bitOr := m * 1728 } := by
  rw [sha256dRangeCost, sha256dAttemptCost_eq, RoundCost.rep]
  apply RoundCost.eq_of_fields <;> norm_num

/-- **THE RECURRENCE IN THE RANGE**: one more attempt is one attempt plus the attempts already
made, field by field. -/
theorem sha256dRangeCost_succ (σ : R) (n : Nat) :
    sha256dRangeCost σ (n + 1) = sha256dAttemptCost σ + sha256dRangeCost σ n := by
  unfold sha256dRangeCost
  rw [RoundCost.rep_succ]

/-- **Additivity over the range, eleven fields**: the cost of `m + n` attempts is the sum of the
costs of `m` and `n` attempts. This is `RoundCost.rep_add` instantiated at the attempt. -/
theorem sha256dRangeCost_add (σ : R) (m n : Nat) :
    sha256dRangeCost σ (m + n) = sha256dRangeCost σ m + sha256dRangeCost σ n := by
  unfold sha256dRangeCost
  rw [RoundCost.rep_add]

/-- **Additivity over the range, six fields**, read through the projection. -/
theorem sha256dRangeCost_ring_add (σ : R) (m n : Nat) :
    (sha256dRangeCost σ (m + n)).ringFields =
      (sha256dRangeCost σ m).ringFields + (sha256dRangeCost σ n).ringFields := by
  rw [sha256dRangeCost_add, ringFields_add]

/-- **The full 32-bit nonce space, closed form with the multiplier explicit.** -/
theorem sha256dRangeCost_nonceSpace (σ : R) :
    sha256dRangeCost σ nonceSpace =
      { mul := 4294967296 * 10368, add := 4294967296 * 7128, sub := 0, neg := 0, cast := 0,
        channel := 4294967296 * 1728, bitXor := 4294967296 * 2112, bitAnd := 4294967296 * 960,
        bitNot := 4294967296 * 192, shift := 4294967296 * 3744, bitOr := 4294967296 * 1728 } := by
  rw [sha256dRangeCost_eq, nonceSpace]

/-- **THE FULL 32-BIT NONCE SPACE, EVALUATED**: 44 530 220 924 928 ring multiplications,
30 614 526 885 888 ring additions, 7 421 703 487 488 channel reads, 9 070 970 929 152 XORs,
4 123 168 604 160 ANDs, 824 633 720 832 NOTs, 16 080 357 556 224 shifts and 7 421 703 487 488
disjunctions; no subtraction, no negation, no dial conversion. These are counts of operations over
the range, not a rate, a time, an energy or a price. -/
theorem sha256dRangeCost_nonceSpace_literals (σ : R) :
    sha256dRangeCost σ nonceSpace =
      { mul := 44530220924928, add := 30614526885888, sub := 0, neg := 0, cast := 0,
        channel := 7421703487488, bitXor := 9070970929152, bitAnd := 4123168604160,
        bitNot := 824633720832, shift := 16080357556224, bitOr := 7421703487488 } := by
  rw [sha256dRangeCost_nonceSpace]

/-- **The nonce-space figure is the 32-bit instance of the range closed form.** -/
theorem sha256dRangeCost_nonceSpace_eq_pow2 (σ : R) :
    sha256dRangeCost σ nonceSpace = sha256dRangeCost σ (2 ^ 32) := by
  unfold nonceSpace
  norm_num

/-- **A split range costs the sum of its parts**: two ranges of half the nonce space cost the
whole. This is the range additivity at `2^31 + 2^31 = 2^32`. -/
theorem sha256dRangeCost_nonceSpace_split (σ : R) :
    sha256dRangeCost σ (2 ^ 31) + sha256dRangeCost σ (2 ^ 31) =
      sha256dRangeCost σ nonceSpace := by
  rw [← sha256dRangeCost_add]
  unfold nonceSpace
  norm_num

/-! ## 9. Audit rows

Every headline of §0-§6 above, printed with its axiom set. The permitted cone is
`{propext, Classical.choice, Quot.sound}` and no row may contain `sorryAx`. The rows of §7-§9 live
in `AlgebraCostPowComparePart2.lean`, which audits them against the same cone; together the two
lists are the former module's single list, row for row and in the same order. -/

#print axioms RoundCost.eq_of_fields
#print axioms RoundCost.composed_comm
#print axioms RoundCost.composed_assoc
#print axioms RoundCost.composed_zero
#print axioms RoundCost.rep_zero
#print axioms RoundCost.rep_succ
#print axioms RoundCost.rep_add
#print axioms RoundCost.rep_composed
#print axioms RingCost.eq_of_fields
#print axioms RingCost.rep_add
#print axioms ringFields_add
#print axioms ringFields_rep
#print axioms ringMul_body_ops
#print axioms ringUpper_body_ops
#print axioms rotReadCost_fields
#print axioms rotr_body_ops
#print axioms shrx_body_ops
#print axioms sigma0_body_ops
#print axioms sigma1_body_ops
#print axioms bigSigma0_body_ops
#print axioms bigSigma1_body_ops
#print axioms choose_body_ops
#print axioms majority_body_ops
#print axioms shifts_come_from_rotations
#print axioms roundCost_fields
#print axioms roundCost_ringFields
#print axioms schedStepCost_fields
#print axioms roundChainCost_rep
#print axioms roundChain64Cost_eq
#print axioms schedChainCost_rep
#print axioms schedChain48Cost_eq
#print axioms blockCost_split
#print axioms blockCost_eq
#print axioms blockCost_ringFields
#print axioms schedShr96
#print axioms schedShift_split
#print axioms roundsHaveNoShr
#print axioms blockCost_component_fields
#print axioms blockCost_ranking
#print axioms blockCost_shares
#print axioms pad_spec
#print axioms pad_layout
#print axioms pad_length
#print axioms lengthBlock_len
#print axioms padLength_80
#print axioms padZeroFill_80
#print axioms padLength_32
#print axioms padZeroFill_32
#print axioms header_pad_length
#print axioms header_pad_blocks
#print axioms digest_pad_length
#print axioms digest_pad_blocks
#print axioms sha256d_block_count
#print axioms header_hash_fold_count
#print axioms nonce_bytes_second_block
#print axioms sha256dAttemptCost_inner_outer
#print axioms sha256dAttemptCost_eq
#print axioms sha256dAttemptCost_ringFields
#print axioms sha256dRangeCost_eq
#print axioms sha256dRangeCost_succ
#print axioms sha256dRangeCost_add
#print axioms sha256dRangeCost_ring_add
#print axioms sha256dRangeCost_nonceSpace
#print axioms sha256dRangeCost_nonceSpace_literals
#print axioms sha256dRangeCost_nonceSpace_eq_pow2
#print axioms sha256dRangeCost_nonceSpace_split
