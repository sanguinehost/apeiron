/-
SCOPE: work/proofs/AlgebraCostPow.lean, new, and its receipt work/proofs/AlgebraCostPow.md. The
cost, in AR-15's eleven-field ring measure, of one SHA-256 proof-of-work attempt and of a full
mining range: the canonical block-count function `blockCount n = ceil ((n+9)/64)` proved in `Nat`
from the canonical padding arithmetic and joined to `Crypto.Sha2.pad` and `Crypto.Sha2.hash`; the
per-attempt figure for an 80-byte header; the closed-form range figures for the 32-bit nonce space
and for `k` padded blocks, with additivity proved over the eleven fields and over AR-10's six
`RingCost` fields; and an all-three-faces comparison. Complete when the module elaborates with
every headline audited inside {propext, Classical.choice, Quot.sound} and zero sorryAx, a seeded
per-body count defect and a seeded block-count defect each reject at the theorems that state the
closed counts, and this receipt records the counts, their grades, the named premises, the explicit
non-claims and the residual list.
REGISTER: algebra/resource model. No production, pin, API, extraction-root, corpus or gate-driver
edit; no Rust source; no cargo; no regen.sh; no measurement. Excluded by construction: latency,
wall-clock time, throughput, bytes per second, energy, joules, hardware, ASIC or GPU figures,
miner economics, profitability, USD, compiler or codegen cost, the kernel ABI, the `unsafe`
container, device dispatch cost, constant-time and security properties of any hash, and every
asymptotic statement. AGENTS.md:429 - a cost-model proof is not a hardware speed measurement; this
module states counts of algebraic operations and nothing else. This module performs no live target
contact, no probing and no external submission.
CORPUS: `Apeiron.Trichotomy.AnyRing.{RingCell,ringMul,ringNormForm}` and
`Apeiron.Trichotomy.ChannelsAnyRing.{ringUpper,ringUpper_mul}` through
repos/apeiron/proof/corpus/.lake/packages/apeiron/proof/ at the pinned revision
880aa57e47e782ed7a9d9ee1974cfc87360b446b, grade `proved` there. The canonical SHA-2 bodies are
`Apeiron.Crypto.Sha2.{rotr:50,shrx:54,sigma0:60,sigma1:64,Sigma0:67,Sigma1:70,choose:73,
majority:76,schedule:82,compress:96,pad:133,lengthBlock:128,hash:146}` at
repos/apeiron/proof/corpus/Crypto/Sha2.lean, sha256
b87814767012d1306761c31c78a9c429f4382258201b88899598085e237d5c6e, a root of the `apeironWordBridge`
corpus build. Extracted side: proof/corpus/WordRefinement.lean:38 (`RingCell (ZMod (2^64))`),
proof/Denotation.lean:35 (`denote_wrapping_add`, width 64 only).
REFERENCE: repos/apeiron/AGENTS.md:336-341 (§8 per-function complexity: the measure, the recurrence,
why it closes), :156-167 (§4 obligations), :226-240 (§5 grades), :429 (a cost-model proof is not a
hardware speed measurement). work/proofs/AlgebraCostRound.lean/.md (AR-15), sha256
908c7aa729a14164a050490a67b78de83e92398fc72e2e4fa894c8077bdc6c8b: `RoundCost` (eleven fields),
`roundCost`, `schedStepCost`, `blockCost`, `blockCost_eq`, the 64-round closed form, `rep_add`.
work/proofs/AlgebraCost.lean (AR-10), sha256
52e5d6c5c2b4a144df2a99824052521c29f4b24252d68df5096f7870447df4fe: `RingCost` (six fields),
`rep_add`. work/proofs/Sha256Pad.lean/.md: `canonTotal`, `zeroFill`, `tailLen`, `totalLayout`,
`pad_length_total`, `pad_blocks` - the canonical block-count arithmetic, re-derived here rather than
imported (that file is not a root of any corpus `lean_lib`). receipts/campaign-review.md:27 and
:30-34, the owner focus on a formal cost model and the statement that the mining angle is a cost
question first.
GATE: /home/lucas/Workspace/icarus/bin/lane lake env lean
/home/lucas/Workspace/icarus/work/proofs/AlgebraCostPow.lean, cwd
/home/lucas/Workspace/icarus/repos/apeiron/proof/corpus. Pre-flight
repos/apeiron/tools/gate/target/debug/gate hold, cwd /home/lucas/Workspace/icarus, observed
`gate hold: no hold in force`, RC=0, run before the first write of this session. Wall-clock budget
450 s per file elaboration and 590 s per invocation including queue time, under
flock /tmp/opencode/apeiron-lane.lock, MemoryMax=3800M, MemoryHigh=3400M, MemorySwapMax=0,
LEAN_NUM_THREADS=1. No cargo, no `lake build` of the package, no regen.sh, no nohup/setsid, no
retry loop, no raised maxRecDepth, no maxHeartbeats 0, no large enumeration.
GRADE: `proved` for every operation count read off a Lean definition body by unfolding it in this
module, for the block-count function and its join to `pad` and `hash`, for the measure's laws, for
the closed forms, for the additivity laws over eleven and six fields, and for the three-face
comparison; the cone is {propext, Classical.choice, Quot.sound} with no `sorryAx`, checked by the
`#print axioms` rows below. `model` for three readings carried by name and never used as axioms:
`RotRead` (the identification of `ringUpper 1 (ringMul 1 (x,0) (2^n, 2^(32-n)))` with a word
rotation, candidate C5), `AddRead` (the identification of a `Word` addition with a `ZMod (2^32)`
addition, template `Denotation.lean:35`, proved at width 64 only), and the count of the seven `Word`
additions of the round body, read off `Sha2.lean:108,109,112,116` because no machine-checked
operation-count mechanism exists. `tested`: none - no `cargo test` ran, so no row of this receipt
carries that grade. Unmeasured and unclaimed: everything on the REGISTER exclusion list. In
particular NO wall-clock, cycle, joule, watt, hash-per-second, USD or profitability claim follows
from any number in this module.
-/

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sha2
import WordRefinement
import Mathlib.Tactic

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
set_option linter.unnecessarySeqFocus false

universe u

namespace algebra_cost_pow

open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
open apeiron_word

variable {R : Type u} [CommRing R]

/-! ## 0. The width and the three dials

The SHA-2 word is `BitVec 32` (`Sha2.lean:22`), so the ring this count is read at is
`ZMod (2 ^ 32)`; the extracted `Cell<i64>` arithmetic runs at width 64 (`WordRefinement.lean:38`)
and its counts are kept out of the SHA-2 figures. The three dial values of `cell.Sigma` at width 32
are `denoteSigma`, `WordRefinement.lean:29-35`: elliptic `-1`, parabolic `0`, hyperbolic `1`. -/

/-- The ring the SHA-2 attempt count is measured in: `w = 32`. -/
abbrev W32 := ZMod (2 ^ 32)

/-- The elliptic face's dial value. -/
abbrev faceElliptic : W32 := -1
/-- The parabolic face's dial value. -/
abbrev faceParabolic : W32 := 0
/-- The hyperbolic face's dial value. -/
abbrev faceHyperbolic : W32 := 1

/-! ## 1. The measure

AR-15's `RoundCost σ` has eleven fields, one counted quantity each, and AR-10's `RingCost σ` has six.
Neither module is a root of any corpus `lean_lib`, so both structures are re-declared here rather
than imported, and both composition laws are re-proved. §1.3 joins them: the six ring fields are a
projection of the eleven, and the projection commutes with the iteration operator, so every
eleven-field additivity statement here yields a six-field one.

### 1.1 The eleven counted quantities -/

/-- `RoundCost σ`: the measure. Eleven fields, one counted quantity each - the six ring operations of
`R[x]/(x^2 - sigma)` that AR-10's `RingCost` records, and the five word operations that the two-slot
algebra has no reading for at word width. `σ` is the dial the ring fields are read at; §7 states as a
result that no count enters it. -/
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
compared on it even though their records carry different indices. Temporal and spatial O(1): a
tuple of eleven projections. -/
def RoundCost.vector {σ : R} (c : RoundCost σ) :
    Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat :=
  (c.mul, c.add, c.sub, c.neg, c.cast, c.channel, c.bitXor, c.bitAnd, c.bitNot, c.shift, c.bitOr)

/-- **`rep σ n a` is the measure of `n` dependent copies of an operation of cost `a`.** Temporal O(1)
and spatial O(1): eleven `Nat` multiplications on an eleven-field record, no recursion, no
allocation. Counted in: nothing - this is the iteration operator of the measure, not an operation of
a hash; the `n` it is given is a count of repetitions, so the value grows while the function does not
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
of `n` copies and `m` copies, field by field. Every range figure of §6 is this operator, so every
additivity statement about a range is an instance of this theorem. -/
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

/-! ### 1.2 AR-10's six-field ring measure, and its composition law

`RingCost σ` is AR-10's record: the six ring quantities only. It is re-declared here with its own
`rep_add`, so the range figures of §6 carry an additivity statement over six fields as well as over
eleven. -/

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

/-! ### 1.3 The six ring fields are a projection of the eleven -/

/-- **The ring projection.** The first six fields of `RoundCost σ` read as AR-10's `RingCost σ`, and
the other five fields are what that record cannot express. Temporal and spatial O(1): six
projections, no allocation. -/
def RoundCost.ringFields {σ : R} (c : RoundCost σ) : RingCost σ :=
  ⟨c.mul, c.add, c.sub, c.neg, c.cast, c.channel⟩

/-- **The projection commutes with composition.** -/
theorem ringFields_add (σ : R) (a b : RoundCost σ) :
    (a + b).ringFields = a.ringFields + b.ringFields := by
  cases a
  cases b
  rfl

/-- **The projection commutes with the iteration operator**, so every eleven-field figure inherits a
six-field one: `rep n` of the eleven is `rep n` of the six. -/
theorem ringFields_rep (σ : R) (n : Nat) (a : RoundCost σ) :
    (RoundCost.rep σ n a).ringFields = RingCost.rep σ n a.ringFields := by
  cases a
  rfl

/-! ## 2. The per-operation counts, re-derived from the canonical bodies

Every count below is read off a displayed definition body and the display is `rfl`-checked in this
module. A count is a reading of a displayed term: nothing in Lean counts the operations of a term,
which is AR-15's residual item 1 and this module's §8 item 1. -/

/-- **The canonical multiplication** `(a + bx)(c + dx) = (ac + sigma*bd) + (ad + bc)x`: five
multiplications and two additions. -/
def ringMulCost (σ : R) : RoundCost σ :=
  { mul := 5, add := 2, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The canonical channel read** `z.1 + r*z.2`: one multiplication, one addition, one application
counted in `channel`. -/
def ringUpperCost (σ : R) : RoundCost σ :=
  { mul := 1, add := 1, sub := 0, neg := 0, cast := 0, channel := 1,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The canonical norm form** `N(a + bx) = a^2 - sigma*b^2`: three multiplications and one
subtraction. No attempt reads a norm form; the price of face identification is §7.4. -/
def ringNormFormCost (σ : R) : RoundCost σ :=
  { mul := 3, add := 0, sub := 1, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **One `Word` addition.** `Sha2.lean:108,109,112,116` adds `Word`s and a word addition mod `2^32`
is one addition of `ZMod (2^32)`; that identification is the named premise `AddRead`, at grade
`model`, template `Denotation.lean:35` proved at width 64 only. `cast` is zero because the canonical
body holds its words as `Word`, not as a dial enum variant. -/
def wordAddCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 1, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The two-slot reading of one addition**, `RingCell.add` on `R × R`: two scalar additions. Kept
separate so the single-slot and two-slot additive readings differ by a number and not by prose. -/
def ringCellAddCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 2, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0 }

/-- **The rotation reading**, `ringUpper 1 (ringMul 1 (x,0) (2^n, 2^(32-n)))`: one canonical
multiplication followed by one channel read. Its *count* is a fact about two bodies and is
unconditional; its *identification* with a word rotation is candidate C5 at grade `model` and is
carried as the named premise `RotRead`, unused by every count. -/
def rotReadCost (σ : R) : RoundCost σ := ringMulCost σ + ringUpperCost σ

/-- `ROTR`: two shifts and one disjunction. A rotation is not one operation in the canonical body. -/
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

/-- `Σ₀(x) = ROTR^2 x ⊕ ROTR^13 x ⊕ ROTR^22 x`: three rotations and two XORs, **no shift by itself**
- the `SHR` terms of FIPS 180-4 §4.1.2 belong to `σ₀` and `σ₁`, which the message schedule uses and
the round function does not. -/
def bigSigma0Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 6, bitOr := 3 }

/-- `Σ₁(x) = ROTR^6 x ⊕ ROTR^11 x ⊕ ROTR^25 x`: the same inventory as `Σ₀`. -/
def bigSigma1Cost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 0, bitNot := 0, shift := 6, bitOr := 3 }

/-- `Ch(x, y, z) = (x &&& y) ⊕ (¬x &&& z)`: two ANDs, one NOT, two XORs. -/
def chooseCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 2, bitNot := 1, shift := 0, bitOr := 0 }

/-- `Maj(x, y, z) = (x &&& y) ⊕ (x &&& z) ⊕ (y &&& z)`: three ANDs and two XORs, no NOT. -/
def majorityCost (σ : R) : RoundCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 2, bitAnd := 3, bitNot := 0, shift := 0, bitOr := 0 }

/-! ### 2.1 The bodies the counts are read off -/

/-- Unfolding `ringMul` exhibits the expression whose five multiplications and two additions
`ringMulCost` names. -/
theorem ringMul_body_ops (σ : R) (z w : RingCell R) :
    ringMul σ z w = (z.1 * w.1 + σ * z.2 * w.2, z.1 * w.2 + z.2 * w.1) := rfl

/-- Unfolding the channel read exhibits the one multiplication and one addition `ringUpperCost`
names. -/
theorem ringUpper_body_ops (r : R) (z : RingCell R) : ringUpper r z = z.1 + r * z.2 := rfl

/-- Unfolding the norm form exhibits the three multiplications and one subtraction `ringNormFormCost`
names. -/
theorem ringNormForm_body_ops (σ : R) (z : RingCell R) :
    ringNormForm σ z = z.1 * z.1 - σ * (z.2 * z.2) := by
  simp only [ringNormForm, pow_two]

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
message schedule.** `Σ₀` and `Σ₁` are three rotations each and a `ROTR` is two shifts, so a round's
`shift` field is exactly `2 * 3 * 2 = 12`. `σ₀` and `σ₁` add one shift each, which is why a schedule
step has five shifts per sigma rather than four. -/
theorem shifts_come_from_rotations (σ : R) :
    (bigSigma0Cost σ).shift = 3 * (rotrCost σ).shift ∧
      (bigSigma1Cost σ).shift = 3 * (rotrCost σ).shift ∧
      (sigma0Cost σ).shift = 2 * (rotrCost σ).shift + (shrxCost σ).shift ∧
      (sigma1Cost σ).shift = 2 * (rotrCost σ).shift + (shrxCost σ).shift := by
  unfold bigSigma0Cost bigSigma1Cost sigma0Cost sigma1Cost rotrCost shrxCost
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- The four-addition shape of `t1` as written at `Sha2.lean:108`: left-associated, four `Word`
additions. -/
theorem t1_arity (hh s c k w : Word) :
    hh + s + c + k + w = (((hh + s) + c) + k) + w := rfl

/-! ## 3. The per-attempt cost, in one padded block, the round, the schedule and the block

`roundCost` is AR-15's §3.1 figure, re-declared and re-proved here so that every number below is
derived in this module from the bodies of §2 rather than read across modules.

- **Measure:** `RoundCost σ`, eleven fields, one counted quantity each.
- **Recurrence:** `roundChainCost σ 0 = RoundCost.zero σ`,
  `roundChainCost σ (n + 1) = roundChainCost σ n + roundCost σ` (`rfl`), one round per step plus
  what the rounds already made have spent.
- **Why it closes:** the recursion is on `Nat`, so induction is well founded with a checked base
  case; `RoundCost.rep_add` (§1.1) telescopes the recurrence and `roundChainCost_rep` reads the
  telescoped sum off. Temporal `n` steps and one stack frame for the recursive definition;
  `roundChainCost_eq` and `blockCost_eq` are closed terms, so the figures themselves are O(1) to
  compute - the growth is in the counted quantity, not in the work the functions do. -/

/-- **One round** of `compress` (`Sha2.lean:100-119`): six rotations under the C5 reading, the four
round-function bodies, and seven word additions - four in `t1` (`:108`), one in `t2` (`:109`) and two
in the state update (`:112`, `:116`). The element reads `acc.getD i 0`, `K.getD t 0`, `w.getD t 0`
cost nothing: `RoundCost` has no read field. Temporal and spatial O(1): a closed term of ten record
compositions, no recursion. Counted in: the eleven `RoundCost` fields. Derivation: the per-operation
inventory of §2 composed once per occurrence; the recurrence it closes over is `roundChainCost`
below. -/
def roundCost (σ : R) : RoundCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ +
    bigSigma0Cost σ + bigSigma1Cost σ + chooseCost σ + majorityCost σ +
    wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ +
    wordAddCost σ

/-- **The round's closed count**: 36 multiplications, 25 additions, 6 channel reads, 8 XORs, 5 ANDs,
1 NOT, 12 shifts, 6 disjunctions, and no subtraction, no negation, no dial conversion. -/
theorem roundCost_fields (σ : R) :
    roundCost σ =
      { mul := 36, add := 25, sub := 0, neg := 0, cast := 0, channel := 6,
        bitXor := 8, bitAnd := 5, bitNot := 1, shift := 12, bitOr := 6 } := by
  simp [roundCost, rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed, bigSigma0Cost,
    bigSigma1Cost, chooseCost, majorityCost, wordAddCost]

/-- The round's ring layer alone, as AR-10's six-field record: `(36, 25, 0, 0, 0, 6)`. The other five
fields are what a `RingCost σ` cannot express. -/
theorem roundCost_ringFields (σ : R) : (roundCost σ).ringFields = ⟨36, 25, 0, 0, 0, 6⟩ := by
  rw [roundCost_fields]
  rfl

/-- **The same round under the two-slot additive reading.** With the seven word additions read as
`RingCell.add`, each of which is two scalar additions, `add` rises from 25 to 32 and the other ten
fields are unchanged. -/
def roundCostTwoSlot (σ : R) : RoundCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ +
    bigSigma0Cost σ + bigSigma1Cost σ + chooseCost σ + majorityCost σ +
    ringCellAddCost σ + ringCellAddCost σ + ringCellAddCost σ + ringCellAddCost σ +
    ringCellAddCost σ + ringCellAddCost σ + ringCellAddCost σ

theorem roundCostTwoSlot_fields (σ : R) :
    roundCostTwoSlot σ =
      { mul := 36, add := 32, sub := 0, neg := 0, cast := 0, channel := 6,
        bitXor := 8, bitAnd := 5, bitNot := 1, shift := 12, bitOr := 6 } := by
  unfold roundCostTwoSlot rotReadCost ringMulCost ringUpperCost bigSigma0Cost bigSigma1Cost
    chooseCost majorityCost ringCellAddCost
  rfl

/-- **The two-slot additive reading costs 14 more scalar additions and changes nothing else.** -/
theorem roundCost_twoSlot_add_only (σ : R) :
    (roundCostTwoSlot σ).add = (roundCost σ).add + 7 ∧
      (roundCostTwoSlot σ).mul = (roundCost σ).mul ∧
      (roundCostTwoSlot σ).sub = (roundCost σ).sub ∧
      (roundCostTwoSlot σ).neg = (roundCost σ).neg ∧
      (roundCostTwoSlot σ).cast = (roundCost σ).cast ∧
      (roundCostTwoSlot σ).channel = (roundCost σ).channel ∧
      (roundCostTwoSlot σ).bitXor = (roundCost σ).bitXor ∧
      (roundCostTwoSlot σ).bitAnd = (roundCost σ).bitAnd ∧
      (roundCostTwoSlot σ).bitNot = (roundCost σ).bitNot ∧
      (roundCostTwoSlot σ).shift = (roundCost σ).shift ∧
      (roundCostTwoSlot σ).bitOr = (roundCost σ).bitOr := by
  rw [roundCostTwoSlot_fields, roundCost_fields]
  simp

/-- **One schedule-extension step.** `schedule` (`Sha2.lean:82-89`) extends the block by one step per
`t = 16 … 63`, that is **48 steps per block**, each `W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) +
W[t-16]`: four rotations, two small sigmas, four word additions. Temporal and spatial O(1): a closed
term of nine record compositions. Derivation: the §2 inventory of those four bodies, composed once
each. -/
def schedStepCost (σ : R) : RoundCost σ :=
  rotReadCost σ + rotReadCost σ + rotReadCost σ + rotReadCost σ + sigma0Cost σ + sigma1Cost σ +
    wordAddCost σ + wordAddCost σ + wordAddCost σ + wordAddCost σ

theorem schedStepCost_fields (σ : R) :
    schedStepCost σ =
      { mul := 24, add := 16, sub := 0, neg := 0, cast := 0, channel := 4,
        bitXor := 4, bitAnd := 0, bitNot := 0, shift := 10, bitOr := 4 } := by
  simp [schedStepCost, rotReadCost, ringMulCost, ringUpperCost, RoundCost.composed, sigma0Cost,
    sigma1Cost, wordAddCost]

/-- **The cost of a chain of `n` rounds.** Temporal `n` recursive steps, each one record addition;
spatial one accumulator and one stack frame, O(1) words beyond the result record. Counted in: the
eleven `RoundCost` fields. Derivation: the recurrence `c 0 = zero σ`, `c (n + 1) = c n + roundCost σ`,
one round per step; it closes because `RoundCost.rep_add` telescopes it to `n` copies of the round
and `Nat` induction has no open base case. -/
def roundChainCost (σ : R) : Nat → RoundCost σ
  | 0 => RoundCost.zero σ
  | n + 1 => roundChainCost σ n + roundCost σ

theorem roundChainCost_rep (σ : R) (n : Nat) :
    roundChainCost σ n = RoundCost.rep σ n (roundCost σ) := by
  induction n with
  | zero => simp [roundChainCost, RoundCost.rep_zero, RoundCost.zero]
  | succ n ih => rw [roundChainCost, ih, RoundCost.rep_succ, RoundCost.composed_comm]

/-- **The 64-round closed form, at an arbitrary chain length.** -/
theorem roundChainCost_eq (σ : R) (n : Nat) :
    roundChainCost σ n =
      { mul := n * 36, add := n * 25, sub := 0, neg := 0, cast := 0, channel := n * 6,
        bitXor := n * 8, bitAnd := n * 5, bitNot := n, shift := n * 12, bitOr := n * 6 } := by
  have h : RoundCost.rep σ n (roundCost σ) =
      { mul := n * 36, add := n * 25, sub := 0, neg := 0, cast := 0, channel := n * 6,
        bitXor := n * 8, bitAnd := n * 5, bitNot := n, shift := n * 12, bitOr := n * 6 } := by
    rw [roundCost_fields]
    simp [RoundCost.rep]
  rw [roundChainCost_rep, h]

/-- **The 64-round chain, with the numbers substituted.** -/
theorem roundChain64 (σ : R) :
    roundChainCost σ 64 =
      { mul := 2304, add := 1600, sub := 0, neg := 0, cast := 0, channel := 384,
        bitXor := 512, bitAnd := 320, bitNot := 64, shift := 768, bitOr := 384 } :=
  roundChainCost_eq σ 64

/-- **Additivity over the chain**: the measure of the concatenation is the sum of the measures. -/
theorem roundChainCost_add (σ : R) (m n : Nat) :
    roundChainCost σ (m + n) = roundChainCost σ m + roundChainCost σ n := by
  simp only [roundChainCost_eq]
  apply RoundCost.eq_of_fields <;> simp [RoundCost.composed, Nat.add_mul, Nat.mul_add]

/-! ### 3.1 One §6.2.2 block -/

/-- **One block transformation cost.** `compress` (`Sha2.lean:96-120`) is sixty-four rounds **plus
one `schedule` pass of forty-eight steps** (`:97`) **plus the eight-word add-back** (`:120`). Temporal
and spatial O(1): a closed term of three record compositions, no recursion. Counted in: the eleven
`RoundCost` fields. Derivation: the three components above composed once each; closure is
`RoundCost.rep_add` plus `roundChainCost_eq`. -/
def blockCost (σ : R) : RoundCost σ :=
  roundChainCost σ 64 + RoundCost.rep σ 48 (schedStepCost σ) + RoundCost.rep σ 8 (wordAddCost σ)

theorem blockCost_split (σ : R) :
    blockCost σ =
      roundChainCost σ 64 + RoundCost.rep σ 48 (schedStepCost σ) + RoundCost.rep σ 8 (wordAddCost σ) :=
  rfl

/-- **The block's closed count**: 3456 ring multiplications, 2376 ring additions, 576 channel reads,
704 XORs, 320 ANDs, 64 NOTs, 1248 shifts and 576 disjunctions, and no subtraction, no negation and
no dial conversion. -/
theorem blockCost_eq (σ : R) :
    blockCost σ =
      { mul := 3456, add := 2376, sub := 0, neg := 0, cast := 0, channel := 576,
        bitXor := 704, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576 } := by
  rw [blockCost_split, roundChain64, schedStepCost_fields]
  rfl

/-- **The block's ring layer alone**, as AR-10's six-field record `(3456, 2376, 0, 0, 0, 576)`. -/
theorem blockCost_ringFields (σ : R) : (blockCost σ).ringFields = ⟨3456, 2376, 0, 0, 0, 576⟩ := by
  rw [blockCost_eq]
  rfl

/-! ## 4. The block-count function

For a message of `n` bytes the padded message has `blockCount n = ⌈(n + 9) / 64⌉` blocks, and `hash`
performs one `compress` per padded block. The arithmetic is `Nat` throughout, re-derived from the
canonical padding body `Sha2.lean:133-136` and joined to it at `pad_length_total`.

- **Measure:** `Nat`, the count of 64-byte blocks.
- **Recurrence / derivation:** `canonTotal n` is the canonical `total` of `pad`; `canonTotal_ge`
  gives `n + 9 ≤ canonTotal n` and `canonTotal_lt` gives `canonTotal n < n + 73`, so `canonTotal n`
  is the *least* multiple of 64 that is at least `n + 9`. Division of a multiple of 64 by 64 is that
  multiple's quotient, and the least such multiple is `64 * ⌈(n+9)/64⌉`, which is
  `ceilDiv64 (n + 9)` because `64 * ⌈m/64⌉ = m + ((64 - m % 64) % 64)` for every `m`.
- **Why it closes:** every step is `Nat` arithmetic with explicit `omega`/`decide` instances; the
  two structural bounds and the divisibility are proved once and the rest is linear arithmetic over
  them. Temporal and spatial O(1) for each of `canonTotal`, `rest`, `tailLen`, `zeroFill`,
  `ceilDiv64`: each is a closed term with at most one `%`, one `/` and one `ite` on literals, and no
  recursion. -/

/-- Bytes left over after the full 64-byte blocks of a message of `n` bytes. -/
def rest (n : Nat) : Nat := n % 64

/-- Full 64-byte blocks of a message of `n` bytes. -/
def fullBlocks (n : Nat) : Nat := n / 64

/-- The tail buffer length of the canonical padding: 64 when the remainder leaves room for the
`0x80` byte and the eight length bytes, 128 otherwise. -/
def tailLen (n : Nat) : Nat := if rest n < 56 then 64 else 128

/-- **The canonical padded total length**, the `Nat` reading of `Sha2.lean:134-135`: `r` is
`(n + 9) % 64`, and `total` is `n + 9` if `r = 0` and `n + 9 + (64 - r)` otherwise. -/
def canonTotal (n : Nat) : Nat :=
  let r := (n + 9) % 64
  if r = 0 then n + 9 else n + 9 + (64 - r)

/-- Bytes the canonical padding appends after the message: `total - n`. -/
def tailBytes (n : Nat) : Nat := canonTotal n - n

/-- **The canonical zero-fill count**, `total - n - 9`: the zeros between the `0x80` byte and the
eight-byte length field. -/
def zeroFill (n : Nat) : Nat := canonTotal n - n - 9

/-- `ceilDiv64 m` is the least multiple of 64 that is at least `m`, that is `⌈m/64⌉`. -/
def ceilDiv64 (m : Nat) : Nat := (m + 63) / 64

/-- **The block-count function.** A message of `n` bytes pads to `blockCount n` blocks of 64 bytes:
the `1`-byte merge tag `0x80` plus the `8`-byte length field forces `n + 9` bytes before rounding
up. -/
def blockCount (n : Nat) : Nat := ceilDiv64 (n + 9)

/-! ### 4.1 The padding arithmetic, in `Nat` -/

theorem eq_div_mod (n : Nat) : n = 64 * fullBlocks n + rest n :=
  (Nat.div_add_mod n 64).symm

theorem msgSplit (n : Nat) : 64 * fullBlocks n + rest n = n := (eq_div_mod n).symm

theorem rest_lt (n : Nat) : rest n < 64 := Nat.mod_lt n (by omega)

theorem tailLen_cases (n : Nat) : tailLen n = 64 ∨ tailLen n = 128 := by
  rw [tailLen]; split <;> omega

theorem tailLen_mod (n : Nat) : tailLen n % 64 = 0 := by
  rcases tailLen_cases n with h | h <;> simp [h]

/-- **The padded total is a whole number of blocks.** -/
theorem canonTotal_multiple (n : Nat) : 64 ∣ canonTotal n := by
  rw [canonTotal]; split <;> omega

/-- **The padded total is at least `n + 9`**: the message, the `0x80` byte and the length field all
fit. -/
theorem canonTotal_ge (n : Nat) : n + 9 ≤ canonTotal n := by
  rw [canonTotal]; split <;> omega

/-- **The padded total is at most one block more than `n + 9`.** -/
theorem canonTotal_lt (n : Nat) : canonTotal n - 9 < n + 64 := by
  rw [canonTotal]; split <;> omega

/-- The padded total is the tail length on top of the full blocks. -/
theorem canonTotal_eq (n : Nat) : canonTotal n = 64 * fullBlocks n + tailLen n := by
  have h1 : (n + 9) % 64 = (n % 64 + 9) % 64 := by omega
  have h2 : n = 64 * fullBlocks n + rest n := eq_div_mod n
  rw [canonTotal, h1]
  simp only [fullBlocks, tailLen, rest] at *
  split <;> split <;> omega

/-- The zero fill plus the `0x80` byte and the length field is the appended tail. -/
theorem zeroFill_add (n : Nat) : zeroFill n + 9 = canonTotal n - n := by
  have h := canonTotal_ge n
  simp only [zeroFill]; omega

/-- **The layout identity** the whole figure rests on: message, `0x80`, zeros, length field. -/
theorem totalLayout (n : Nat) : n + 1 + zeroFill n + 8 = canonTotal n := by
  have h := zeroFill_add n
  have h2 := canonTotal_ge n
  simp only [zeroFill]
  omega

/-- **The zero fill is bounded by 63** for every `n`, and is non-negative. -/
theorem zeroFill_le (n : Nat) : zeroFill n ≤ 63 := by
  have hb := totalLayout n
  have h := canonTotal_lt n
  have h2 := canonTotal_ge n
  simp only [zeroFill] at *
  omega

theorem zeroFill_nonneg (n : Nat) : 0 ≤ zeroFill n := by
  have h := zeroFill_add n
  have h2 := canonTotal_ge n
  simp only [zeroFill] at *
  omega

/-! ### 4.2 `ceilDiv64` is the ceiling, stated on `Nat` -/

/-- **The ceiling characterisation**: `ceilDiv64 m` is the unique `q` with `m ≤ 64 * q < m + 64`,
that is `64 * ceilDiv64 m` is the least multiple of 64 at least `m`. -/
theorem ceilDiv64_spec (m : Nat) : m ≤ 64 * ceilDiv64 m ∧ 64 * ceilDiv64 m < m + 64 := by
  have h := Nat.div_add_mod (m + 63) 64
  have hr : (m + 63) % 64 < 64 := Nat.mod_lt _ (by omega)
  unfold ceilDiv64
  omega

/-- **The quotient characterisation**: `m / 64 ≤ ⌈m/64⌉ ≤ m / 64 + 1`. Together with
`ceilDiv64_strict` this is the statement that `ceilDiv64 m` is `⌈m / 64⌉`; the weak form is the one
that holds at `m = 0` and at every multiple of 64, where the ceiling equals the quotient exactly. -/
theorem ceilDiv64_quot (m : Nat) : m / 64 ≤ ceilDiv64 m ∧ ceilDiv64 m ≤ m / 64 + 1 := by
  have h := ceilDiv64_spec m
  have hd : 64 * ceilDiv64 m + (m + 63) % 64 = m + 63 := by
    simpa [ceilDiv64] using (Nat.div_add_mod (m + 63) 64)
  have hdiv := Nat.div_add_mod m 64
  have hmod : m % 64 < 64 := Nat.mod_lt _ (by omega)
  omega

/-- **The strict ceiling characterisation**, for `0 < m`:
`64 * (⌈m/64⌉ - 1) < m ≤ 64 * ⌈m/64⌉`, so `⌈m/64⌉` is exactly the least `q` with `m ≤ 64 * q`. -/
theorem ceilDiv64_strict (m : Nat) (hm : 0 < m) : 64 * (ceilDiv64 m - 1) < m := by
  have h := ceilDiv64_spec m
  have hpos : 0 < ceilDiv64 m := by omega
  omega

/-- `ceilDiv64` is monotone in its argument. -/
theorem ceilDiv64_mono {a b : Nat} (h : a ≤ b) : ceilDiv64 a ≤ ceilDiv64 b := by
  have h1 := ceilDiv64_spec a
  have h2 := ceilDiv64_spec b
  omega

/-! ### 4.3 THE BLOCK-COUNT THEOREM

`canonTotal n` is the least multiple of 64 at least `n + 9` (§4.1), and `64 * ceilDiv64 (n + 9)` is
the least multiple of 64 at least `n + 9` (§4.2), so they are equal, and dividing by 64 gives the
block count. -/

/-- **`64 * ceilDiv64 (n + 9) = canonTotal n`**: the canonical padded total is the least multiple of
64 at least `n + 9`. -/
theorem canonTotal_eq_ceil (n : Nat) : 64 * ceilDiv64 (n + 9) = canonTotal n := by
  obtain ⟨k, hk⟩ := canonTotal_multiple n
  have h1 := canonTotal_ge n
  have h2 := canonTotal_lt n
  have h3 := ceilDiv64_spec (n + 9)
  omega

/-- **THE BLOCK-COUNT THEOREM.** The padded length of a message of `n` bytes, divided by 64, is
`⌈(n + 9)/64⌉`. -/
theorem canonTotal_div (n : Nat) : canonTotal n / 64 = blockCount n := by
  have h := canonTotal_eq_ceil n
  calc canonTotal n / 64 = (64 * ceilDiv64 (n + 9)) / 64 := by rw [h]
    _ = ceilDiv64 (n + 9) := Nat.mul_div_cancel_left (ceilDiv64 (n + 9)) (by omega)
    _ = blockCount n := rfl

/-- **The block count is the ceiling, stated on `Nat`**: the largest whole number of blocks strictly
below `n + 9` is `blockCount n - 1`, and `n + 9` fits in `blockCount n` blocks. This is the
`⌈(n+9)/64⌉` characterisation with no division on either side. -/
theorem blockCount_is_ceil (n : Nat) :
    64 * (blockCount n - 1) < n + 9 ∧ n + 9 ≤ 64 * blockCount n := by
  have h := ceilDiv64_strict (n + 9) (by omega)
  have h2 := ceilDiv64_spec (n + 9)
  unfold blockCount
  omega

/-- **The block-count function in closed form**, `⌈m/64⌉ = (m + 63)/64`. -/
theorem blockCount_eq (n : Nat) : blockCount n = (n + 9 + 63) / 64 := by
  unfold blockCount ceilDiv64
  omega

/-- **The block count is at least one and no more than `⌈(n+9)/64⌉` needs at most one block beyond the
message**: `blockCount n ≥ 1` for every `n`, and `64 * blockCount n ≤ n + 72`. -/
theorem blockCount_pos (n : Nat) : 1 ≤ blockCount n := by
  have h := ceilDiv64_spec (n + 9)
  unfold blockCount
  omega

theorem blockCount_le (n : Nat) : 64 * blockCount n ≤ n + 72 := by
  have h := canonTotal_eq_ceil n
  have h2 := canonTotal_lt n
  unfold blockCount
  omega

/-- **Adding one byte can add at most one block**, and it adds one exactly when the message fills a
block to within the nine bytes the padding needs. -/
theorem blockCount_succ_le (n : Nat) : blockCount (n + 1) ≤ blockCount n + 1 := by
  have h := blockCount_eq n
  have h2 := blockCount_eq (n + 1)
  omega

/-! ### 4.4 The join to the canonical bodies

`pad` (`Sha2.lean:133-136`) is `m ++ [0x80] ++ replicate (zeroFill |m|) 0 ++ lengthBlock |m|`, so its
length is `|m| + 1 + zeroFill |m| + 8 = canonTotal |m|`, and `hash` (`Sha2.lean:146-148`) folds
`compress` over `List.range ((pad m).length / 64)`. -/

/-- **The `pad_spec` shape**, definitional at `|m| = n`. -/
theorem pad_spec (m : List (BitVec 8)) :
    pad m = m ++ [0x80] ++ List.replicate (zeroFill m.length) 0 ++ lengthBlock m.length := rfl

/-- **The canonical padded length is `canonTotal`.** This is the join: the `Nat` arithmetic of §4.1
is the length of the canonical `pad`. -/
theorem pad_length_total (m : List (BitVec 8)) : (pad m).length = canonTotal m.length := by
  rw [pad_spec]
  simp only [List.length_append, List.length_replicate, List.length_ofFn, List.length_cons,
    List.length_nil, lengthBlock]
  exact totalLayout m.length

/-- **THE CANONICAL BLOCK COUNT.** The padded message of a message of `n` bytes carries
`⌈(n + 9)/64⌉` blocks, and this is the length of the canonical `pad` divided by 64. -/
theorem pad_blockCount (m : List (BitVec 8)) : (pad m).length / 64 = blockCount m.length := by
  rw [pad_length_total, canonTotal_div]

/-- **One `compress` per padded block.** `hash` folds over `List.range ((pad m).length / 64)`, so it
performs exactly `blockCount m.length` block transformations. `paddedBlock` is `private` in
`Sha2.lean` and is not referenced here; the statement is about the length of the range the fold
walks. -/
theorem hash_folds_over_blockCount (m : List (BitVec 8)) :
    List.range ((pad m).length / 64) = List.range (blockCount m.length) := by
  rw [pad_blockCount m]

/-- **The range a `hash` walks is nonempty for every message**, so one message always costs at least
one block transformation. -/
theorem hash_range_nonempty (m : List (BitVec 8)) :
    0 < (List.range (blockCount m.length)).length := by
  have h := blockCount_pos m.length
  rw [List.length_range]
  exact h

/-! ### 4.5 Boundary witnesses

The block count at the lengths where the padding crosses a block boundary, and at the header length.
Each is a closed arithmetic instance, so each is a kernel decision on explicit `Nat` literals. -/

theorem blockCount_0 : blockCount 0 = 1 := by decide
theorem blockCount_55 : blockCount 55 = 1 := by decide
theorem blockCount_56 : blockCount 56 = 2 := by decide
theorem blockCount_63 : blockCount 63 = 2 := by decide
theorem blockCount_64 : blockCount 64 = 2 := by decide
theorem blockCount_119 : blockCount 119 = 2 := by decide
theorem blockCount_120 : blockCount 120 = 3 := by decide

/-- **The block count of an 80-byte header is 2.** The `0x80` byte and the eight-byte length field
need nine bytes, and `80 + 9 = 89` does not fit in one 64-byte block, so the canonical padding gives
an 80-byte header a second, mostly empty block. -/
theorem blockCount_80 : blockCount 80 = 2 := by decide

/-- **The join at the header length.** -/
theorem pad_blockCount_80 (m : List (BitVec 8)) (hm : m.length = 80) :
    (pad m).length / 64 = 2 := by
  rw [pad_blockCount, hm, blockCount_80]

/-! ## 5. The cost of one proof-of-work attempt

A proof-of-work attempt is one hash of an 80-byte header. The header is 80 bytes, the canonical
padding turns it into `blockCount 80 = 2` blocks, and `hash` performs one `compress` per block, so
the attempt costs `blockCount 80` copies of `blockCost σ`.

**What is not in this figure, by construction:** the padding's own operations (`pad` builds a list;
`RoundCost` has no list-construction, allocation or byte field), the block loading loop
`paddedBlock` and its sixteen-word regrouping, the digest store `Vector.ofFn (h.getD i + v.getD i)`,
and the element traffic of every `getD`. No word-level carry measure is counted either: the additive
layer is `ZMod (2^32)` addition, where the carry is absorbed by the ring's arithmetic before any
operation is counted. -/

/-- **A SHA-256 block-header length in bytes.** The canonical `Crypto.Sha2` module does not carry a
header; 80 is the length this module's attempt figure is stated for, as a named constant so that the
figure is stated at one place. -/
def headerBytes : Nat := 80

/-- **THE COST OF ONE ATTEMPT**, derived from the canonical block count: `blockCount headerBytes`
copies of the block cost. Temporal and spatial O(1): a closed term - one `RoundCost.rep` of eleven
`Nat` multiplications. Counted in: the eleven `RoundCost` fields. Derivation: `hash` performs one
`compress` per padded block (`hash_folds_over_blockCount`), one `compress` costs `blockCost σ`
(§3.1), and the header pads to `blockCount headerBytes` blocks (`blockCount_80`). -/
def attemptCost (σ : R) : RoundCost σ := RoundCost.rep σ (blockCount headerBytes) (blockCost σ)

/-- **The attempt is two padded blocks**, stated so that the derivation is visible in one step. -/
theorem attemptCost_two (σ : R) : attemptCost σ = RoundCost.rep σ 2 (blockCost σ) := by
  rw [attemptCost, headerBytes, blockCount_80]

/-- **The attempt's closed count**: 6912 ring multiplications, 4752 ring additions, 1152 channel
reads, 1408 XORs, 640 ANDs, 128 NOTs, 2496 shifts and 1152 disjunctions, and no subtraction, no
negation and no dial conversion. -/
theorem attemptCost_eq (σ : R) :
    attemptCost σ =
      { mul := 6912, add := 4752, sub := 0, neg := 0, cast := 0, channel := 1152,
        bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152 } := by
  rw [attemptCost_two, blockCost_eq]
  rfl

/-- **The attempt's ring layer alone**, as AR-10's six-field record `(6912, 4752, 0, 0, 0, 1152)`. -/
theorem attemptCost_ringFields (σ : R) : (attemptCost σ).ringFields = ⟨6912, 4752, 0, 0, 0, 1152⟩ := by
  rw [attemptCost_eq]
  rfl

/-- **The attempt figure is exactly twice the block figure, field by field**, so the doubling carries
no hidden field. -/
theorem attemptCost_is_two_blocks (σ : R) :
    (attemptCost σ).mul = 2 * (blockCost σ).mul ∧
      (attemptCost σ).add = 2 * (blockCost σ).add ∧
      (attemptCost σ).sub = 2 * (blockCost σ).sub ∧
      (attemptCost σ).neg = 2 * (blockCost σ).neg ∧
      (attemptCost σ).cast = 2 * (blockCost σ).cast ∧
      (attemptCost σ).channel = 2 * (blockCost σ).channel ∧
      (attemptCost σ).bitXor = 2 * (blockCost σ).bitXor ∧
      (attemptCost σ).bitAnd = 2 * (blockCost σ).bitAnd ∧
      (attemptCost σ).bitNot = 2 * (blockCost σ).bitNot ∧
      (attemptCost σ).shift = 2 * (blockCost σ).shift ∧
      (attemptCost σ).bitOr = 2 * (blockCost σ).bitOr := by
  rw [attemptCost_two, blockCost_eq]
  simp [RoundCost.rep]

/-- **One attempt is never free**: the multiplication field of an attempt is positive. -/
theorem attemptCost_strict (σ : R) : 0 < (attemptCost σ).mul := by
  rw [attemptCost_eq]
  norm_num

/-! ## 6. The cost of a mining range

A range of `m` attempts against a fixed header is `m` dependent copies of `attemptCost σ`, because
each attempt is a fresh `compress` fold over the same padded header: the state carries over but the
sixty-four rounds and the schedule pass do not repeat. So the range cost is
`RoundCost.rep σ m (attemptCost σ)`, and §1.1's `RoundCost.rep_add` makes it additive.

**This is a count of algebraic operations in this measure.** It is not a throughput, a time, an
energy or a currency figure, and no such figure follows from it. -/

/-- **THE COST OF A RANGE OF `m` ATTEMPTS** against one header. Temporal and spatial O(1): a closed
term - `RoundCost.rep` of eleven `Nat` multiplications. Counted in: the eleven `RoundCost` fields.
Derivation: `m` independent folds of `compress` over one padded header, each costing `attemptCost σ`
(§5); the range adds because each attempt re-runs the sixty-four rounds, the forty-eight schedule
steps and the add-back in full, and no operation of one attempt is shared with another. -/
def rangeCost (σ : R) (m : Nat) : RoundCost σ := RoundCost.rep σ m (attemptCost σ)

/-- **The range closed form at an arbitrary range length.** -/
theorem rangeCost_eq (σ : R) (m : Nat) :
    rangeCost σ m =
      { mul := m * 6912, add := m * 4752, sub := 0, neg := 0, cast := 0, channel := m * 1152,
        bitXor := m * 1408, bitAnd := m * 640, bitNot := m * 128, shift := m * 2496,
        bitOr := m * 1152 } := by
  rw [rangeCost, attemptCost_eq]
  rfl

/-- **The range cost is `m` attempts, not fewer and not more**: `rangeCost σ m` is `m` copies of the
attempt cost, field by field. -/
theorem rangeCost_rep_attempt (σ : R) (m : Nat) :
    rangeCost σ m = RoundCost.rep σ m (attemptCost σ) := rfl

/-- **Additivity over a range, eleven fields**: the cost of `m + n` attempts is the sum of the costs
of `m` and `n` attempts. This is `RoundCost.rep_add` instantiated at the attempt. -/
theorem rangeCost_add (σ : R) (m n : Nat) : rangeCost σ (m + n) = rangeCost σ m + rangeCost σ n := by
  simp only [rangeCost_eq]
  apply RoundCost.eq_of_fields <;> simp [RoundCost.composed, Nat.add_mul, Nat.mul_add]

/-- **Additivity over a range, six fields**: the same statement read in AR-10's ring measure, with
its own `rep_add` over six fields rather than eleven. -/
theorem rangeCost_ring_add (σ : R) (m n : Nat) :
    (rangeCost σ (m + n)).ringFields = (rangeCost σ m).ringFields + (rangeCost σ n).ringFields := by
  unfold rangeCost
  rw [ringFields_rep, ringFields_rep, ringFields_rep]
  exact RingCost.rep_add σ m n (attemptCost σ).ringFields

/-- **The range's ring layer, closed form.** -/
theorem rangeCost_ringFields (σ : R) (m : Nat) :
    (rangeCost σ m).ringFields = RingCost.rep σ m ⟨6912, 4752, 0, 0, 0, 1152⟩ := by
  unfold rangeCost
  rw [ringFields_rep, attemptCost_ringFields]

/-- **The range cost grows with the range**: a longer range is never free. -/
theorem rangeCost_strict (σ : R) {m n : Nat} (h : m < n) :
    (rangeCost σ m).mul < (rangeCost σ n).mul := by
  simp only [rangeCost_eq]
  omega

/-! ### 6.1 The 32-bit nonce space

The nonce field of a header is 32 bits wide, so the full space of nonces for one header is
`2 ^ 32 = 4294967296` attempts. The figure below is the algebraic-operation count of that range in
this measure. It says nothing about how long the range takes, how much energy it uses, or what it
costs. -/

/-- The size of a 32-bit nonce space: `2 ^ 32` attempts. -/
def nonceSpace : Nat := 4294967296

/-- The closed-form count of a full 32-bit nonce space, at an arbitrary width `w`: `2^w` attempts. -/
theorem rangeCost_pow2_eq (σ : R) (w : Nat) :
    rangeCost σ (2 ^ w) =
      { mul := 2 ^ w * 6912, add := 2 ^ w * 4752, sub := 0, neg := 0, cast := 0,
        channel := 2 ^ w * 1152, bitXor := 2 ^ w * 1408, bitAnd := 2 ^ w * 640,
        bitNot := 2 ^ w * 128, shift := 2 ^ w * 2496, bitOr := 2 ^ w * 1152 } := by
  rw [rangeCost_eq]

/-- **THE FULL 32-BIT NONCE SPACE AGAINST ONE HEADER**, with the arithmetic evaluated. 29 686 813 949
952 ring multiplications, 20 409 684 590 592 ring additions, 4 947 802 324 992 channel reads,
6 047 313 952 768 XORs, 2 748 779 069 440 ANDs, 549 755 813 888 NOTs, 10 720 238 370 816 shifts and
4 947 802 324 992 disjunctions; no subtraction, no negation, no dial conversion. These are counts of
algebraic operations in the ring measure, not attempts per second, seconds, joules or USD. -/
theorem rangeCost_nonceSpace (σ : R) :
    rangeCost σ nonceSpace =
      { mul := 29686813949952, add := 20409684590592, sub := 0, neg := 0, cast := 0,
        channel := 4947802324992, bitXor := 6047313952768, bitAnd := 2748779069440,
        bitNot := 549755813888, shift := 10720238370816, bitOr := 4947802324992 } := by
  rw [rangeCost_eq, nonceSpace]

/-- **The nonce-space figure is the 32-bit width of `rangeCost_pow2_eq`.** -/
theorem rangeCost_nonceSpace_eq_pow2 (σ : R) :
    rangeCost σ nonceSpace = rangeCost σ (2 ^ 32) := by
  unfold nonceSpace
  norm_num

/-- **A split range costs the sum of its parts**, eleven fields: two ranges of half the nonce space
cost the whole. -/
theorem rangeCost_nonceSpace_split (σ : R) :
    rangeCost σ (2 ^ 31) + rangeCost σ (2 ^ 31) = rangeCost σ nonceSpace := by
  rw [← rangeCost_add]
  unfold nonceSpace
  norm_num

/-- **The same split read in the six-field ring measure**, as the ring projection of the eleven-field
split: the two halves are the operands and the whole is the sum. The six-field additivity itself is
`rangeCost_ring_add`, proved from AR-10's `RingCost.rep_add`. -/
theorem rangeCost_nonceSpace_split_ring (σ : R) :
    (rangeCost σ (2 ^ 31) + rangeCost σ (2 ^ 31)).ringFields =
      (rangeCost σ nonceSpace).ringFields := by
  rw [rangeCost_nonceSpace_split]

/-! ### 6.2 A message of `k` padded blocks

The same measure answers the other half of the question: what a message that pads to `k` blocks
costs. `blocksCost σ k` is `k` block transformations, and `messageCost σ n` is
`blocksCost σ (blockCount n)` - the cost of hashing a message of `n` bytes. -/

/-- **The cost of `k` padded blocks.** Temporal and spatial O(1): a closed term. Counted in: the
eleven `RoundCost` fields. Derivation: `k` `compress` calls in `hash`'s fold (`hash_folds_over_blockCount`),
each costing `blockCost σ`. -/
def blocksCost (σ : R) (k : Nat) : RoundCost σ := RoundCost.rep σ k (blockCost σ)

/-- **The closed form at an arbitrary block count.** -/
theorem blocksCost_eq (σ : R) (k : Nat) :
    blocksCost σ k =
      { mul := k * 3456, add := k * 2376, sub := 0, neg := 0, cast := 0, channel := k * 576,
        bitXor := k * 704, bitAnd := k * 320, bitNot := k * 64, shift := k * 1248,
        bitOr := k * 576 } := by
  rw [blocksCost, blockCost_eq]
  rfl

/-- **Additivity over a block range, eleven fields.** -/
theorem blocksCost_add (σ : R) (k l : Nat) :
    blocksCost σ (k + l) = blocksCost σ k + blocksCost σ l := by
  simp only [blocksCost_eq]
  apply RoundCost.eq_of_fields <;> simp [RoundCost.composed, Nat.add_mul, Nat.mul_add]

/-- **Additivity over a block range, six fields.** -/
theorem blocksCost_ring_add (σ : R) (k l : Nat) :
    (blocksCost σ (k + l)).ringFields = (blocksCost σ k).ringFields + (blocksCost σ l).ringFields := by
  unfold blocksCost
  rw [ringFields_rep, ringFields_rep, ringFields_rep]
  exact RingCost.rep_add σ k l (blockCost σ).ringFields

/-- **The cost of hashing a message of `n` bytes**: `blockCount n` block transformations. Temporal
and spatial O(1): a closed term. Derivation: `hash` performs one `compress` per padded block
(`hash_folds_over_blockCount`) and the padded block count is `blockCount n` (§4.3). -/
def messageCost (σ : R) (n : Nat) : RoundCost σ := RoundCost.rep σ (blockCount n) (blockCost σ)

/-- **The message figure is the block figure at the canonical block count.** -/
theorem messageCost_eq_blocks (σ : R) (n : Nat) : messageCost σ n = blocksCost σ (blockCount n) := by
  rfl

/-- **The message figure, closed through the block count.** -/
theorem messageCost_eq (σ : R) (n : Nat) :
    messageCost σ n =
      { mul := blockCount n * 3456, add := blockCount n * 2376, sub := 0, neg := 0, cast := 0,
        channel := blockCount n * 576, bitXor := blockCount n * 704, bitAnd := blockCount n * 320,
        bitNot := blockCount n * 64, shift := blockCount n * 1248, bitOr := blockCount n * 576 } := by
  rw [messageCost, blockCost_eq]
  rfl

/-- **A message's cost is positive**, because it has at least one padded block. -/
theorem messageCost_strict (σ : R) (n : Nat) : 0 < (messageCost σ n).mul := by
  have h1 := blockCount_pos n
  rw [messageCost, blockCost_eq]
  show 0 < blockCount n * 3456
  omega

/-- **A 1 000 000-byte message pads to 15 626 blocks**, a checked instance of §4.3 at a real
transaction- and header-scale length. -/
theorem blockCount_million : blockCount 1000000 = 15626 := by decide

/-! ## 7. The three faces

### 7.1 What is computed, and what the answer is

The attempt figure, the range figure and the message figure are computed at all three dial values,
`σ = -1` (elliptic), `σ = 0` (parabolic) and `σ = 1` (hyperbolic), and **the three computations agree
in all eleven fields.**

The reason is a property of the counted bodies, not a property of the algebra: every count of §2 is a
closed record of `Nat` literals, and no counted body - `ringMul`, `ringUpper`, `ringNormForm`,
`RingCell.add`, `rotr`, `shrx`, `sigma0`, `sigma1`, `Sigma0`, `Sigma1`, `choose`, `majority`, `BitVec`
addition - has a dial parameter or a branch on one. `ringMul` takes `σ` as an argument and its body
is one expression for every `σ` (`ringMul_body_ops`); the `Word` bodies do not mention a dial at all.
So the measure is **σ-blind by construction**, and the honest form of the result is a statement about
the counted bodies. It is **not** a statement that the trichotomy does not affect cost. -/

/-- **The attempt figure does not depend on the dial.** -/
theorem attemptCost_faceBlind (σ : W32) :
    (attemptCost σ).vector = (6912, 4752, 0, 0, 0, 1152, 1408, 640, 128, 2496, 1152) :=
  by rw [attemptCost_eq]; rfl

/-- **THE THREE FACES, ATTEMPT FIGURE.** -/
theorem attemptCost_threeFaces :
    (attemptCost faceElliptic).vector = (attemptCost faceParabolic).vector ∧
      (attemptCost faceParabolic).vector = (attemptCost faceHyperbolic).vector := by
  rw [attemptCost_eq, attemptCost_eq, attemptCost_eq]
  exact ⟨rfl, rfl⟩

/-- **The attempt figure, field by field, all eleven counts.** -/
theorem attemptCost_threeFaces_fieldwise :
    (attemptCost faceElliptic).mul = (attemptCost faceParabolic).mul ∧
      (attemptCost faceElliptic).mul = (attemptCost faceHyperbolic).mul ∧
      (attemptCost faceElliptic).add = (attemptCost faceParabolic).add ∧
      (attemptCost faceElliptic).add = (attemptCost faceHyperbolic).add ∧
      (attemptCost faceElliptic).sub = (attemptCost faceParabolic).sub ∧
      (attemptCost faceElliptic).sub = (attemptCost faceHyperbolic).sub ∧
      (attemptCost faceElliptic).neg = (attemptCost faceParabolic).neg ∧
      (attemptCost faceElliptic).neg = (attemptCost faceHyperbolic).neg ∧
      (attemptCost faceElliptic).cast = (attemptCost faceParabolic).cast ∧
      (attemptCost faceElliptic).cast = (attemptCost faceHyperbolic).cast ∧
      (attemptCost faceElliptic).channel = (attemptCost faceParabolic).channel ∧
      (attemptCost faceElliptic).channel = (attemptCost faceHyperbolic).channel ∧
      (attemptCost faceElliptic).bitXor = (attemptCost faceParabolic).bitXor ∧
      (attemptCost faceElliptic).bitXor = (attemptCost faceHyperbolic).bitXor ∧
      (attemptCost faceElliptic).bitAnd = (attemptCost faceParabolic).bitAnd ∧
      (attemptCost faceElliptic).bitAnd = (attemptCost faceHyperbolic).bitAnd ∧
      (attemptCost faceElliptic).bitNot = (attemptCost faceParabolic).bitNot ∧
      (attemptCost faceElliptic).bitNot = (attemptCost faceHyperbolic).bitNot ∧
      (attemptCost faceElliptic).shift = (attemptCost faceParabolic).shift ∧
      (attemptCost faceElliptic).shift = (attemptCost faceHyperbolic).shift ∧
      (attemptCost faceElliptic).bitOr = (attemptCost faceParabolic).bitOr ∧
      (attemptCost faceElliptic).bitOr = (attemptCost faceHyperbolic).bitOr := by
  rw [attemptCost_eq faceElliptic, attemptCost_eq faceParabolic, attemptCost_eq faceHyperbolic]
  decide

/-- **The full-nonce-space figure, all three faces.** -/
theorem rangeCost_nonceSpace_threeFaces :
    (rangeCost faceElliptic nonceSpace).vector = (rangeCost faceParabolic nonceSpace).vector ∧
      (rangeCost faceParabolic nonceSpace).vector = (rangeCost faceHyperbolic nonceSpace).vector := by
  simp [rangeCost_eq, nonceSpace, RoundCost.vector]

/-- **The message figure at a fixed length, all three faces.** -/
theorem messageCost_threeFaces (n : Nat) :
    (messageCost faceElliptic n).vector = (messageCost faceParabolic n).vector ∧
      (messageCost faceParabolic n).vector = (messageCost faceHyperbolic n).vector := by
  simp [messageCost_eq, RoundCost.vector]

/-- **The block figure, all three faces.** -/
theorem blockCost_threeFaces :
    (blockCost faceElliptic).vector = (blockCost faceParabolic).vector ∧
      (blockCost faceParabolic).vector = (blockCost faceHyperbolic).vector := by
  simp [blockCost_eq, RoundCost.vector]

/-! ### 7.2 The verdict, stated exactly

**No face yields a different attempt, range or message figure, and this is a result about the
counted bodies only.** The eleven counts are closed records of `Nat` literals over bodies that do
not mention a dial, so the three faces agree field by field
(`attemptCost_threeFaces_fieldwise`, `rangeCost_nonceSpace_threeFaces`,
`messageCost_threeFaces`). -/

/-! ### 7.3 Where the dial actually is: the dial-reading operations, priced separately

The operations that read the dial are not part of an attempt and are not part of a `hash`. Priced
here so that the reader can see what a face-dependent cost record would have to add. -/

/-- **The price of face certification**: one norm read and one channel read on top of an attempt.
6916 ring multiplications, 4753 ring additions, 1 subtraction, 1153 channel reads, and an unchanged
non-ring layer. -/
theorem faceCertifyCost (σ : W32) :
    attemptCost σ + ringNormFormCost σ + ringUpperCost σ =
      { mul := 6916, add := 4753, sub := 1, neg := 0, cast := 0, channel := 1153,
        bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152 } := by
  rw [attemptCost_eq]
  rfl

/-- **Face certification costs the same at all three faces.** -/
theorem faceCertifyCost_threeFaces :
    (attemptCost faceElliptic + ringNormFormCost faceElliptic + ringUpperCost faceElliptic).vector =
        (attemptCost faceParabolic + ringNormFormCost faceParabolic + ringUpperCost faceParabolic).vector
      ∧
      (attemptCost faceParabolic + ringNormFormCost faceParabolic + ringUpperCost faceParabolic).vector =
        (attemptCost faceHyperbolic + ringNormFormCost faceHyperbolic
          + ringUpperCost faceHyperbolic).vector := by
  rw [faceCertifyCost, faceCertifyCost, faceCertifyCost]
  exact ⟨rfl, rfl⟩

/-! ### 7.4 The one face where a dial read is impossible

`ringUpper r z` is defined for every `r`, and `ringUpper_mul`'s premise `r ^ 2 = σ` is satisfiable at
the hyperbolic face (`r = 1`) and at the parabolic face (`r = 0`). At the elliptic face at width 32
it is not. An attempt's own figures do not use that premise, so the elliptic attempt cost is a finite
record like the other two; what fails at the elliptic face is **channel multiplicativity**. The
difference between the faces is therefore a refused reading, not a different number. -/

/-- No root of `-1` in `ZMod 4`, decided over the four elements. -/
theorem no_root_neg_one_w4 : ∀ r : ZMod 4, r ^ 2 ≠ (-1 : ZMod 4) := by decide

/-- No root of `-1` at the SHA-2 word width `ZMod (2 ^ 32)`. -/
theorem no_root_neg_one_w32 : ∀ r : W32, r ^ 2 ≠ (-1 : W32) := by
  intro r h
  have h' := congrArg (ZMod.castHom (n := 2 ^ 32) (m := 4) (by decide : (4 : ℕ) ∣ 2 ^ 32)
    (ZMod 4)) h
  rw [map_pow, map_neg, map_one] at h'
  exact no_root_neg_one_w4 _ h'

theorem root_hyperbolic : (1 : W32) ^ 2 = faceHyperbolic := rfl

theorem root_parabolic : (0 : W32) ^ 2 = faceParabolic := rfl

/-- **Channel multiplicativity is admissible at two faces and refused at the third.** -/
theorem channelMul_admissible_faces :
    (ringUpper faceHyperbolic (ringMul faceHyperbolic (0, 1) (0, 1)) =
        ringUpper faceHyperbolic (0, 1) * ringUpper faceHyperbolic (0, 1)) ∧
      (ringUpper faceParabolic (ringMul faceParabolic (0, 1) (0, 1)) =
        ringUpper faceParabolic (0, 1) * ringUpper faceParabolic (0, 1)) ∧
      (∀ r : W32, r ^ 2 ≠ faceElliptic) :=
  ⟨ringUpper_mul (r := faceHyperbolic) (by rfl) _ _,
    ringUpper_mul (r := faceParabolic) (by rfl) _ _, fun r => no_root_neg_one_w32 r⟩

/-! ## 8. What this module does not give

**It is a count of algebraic operations in a formal measure. It is not a hardware throughput, a
wall-clock time, a cycle count, an energy or joule figure, a hash rate, or a USD or profitability
figure, and none of those follows from any number here.** The mining relevance is exactly this: it
fixes the algebraic work per attempt, and per range, relative to other primitives measured in the
same eleven-field measure - nothing more.

Named exclusions, each a residual rather than a result:

1. **The occurrence count of the round's seven `Word` additions is `model`,** read off canonical
   source lines `Sha2.lean:108,109,112,116`. The `rfl`-checked rows are `t1_arity` and the body
   theorems; nothing in Lean counts the operations of a term. Transcribing the whole `compress` body
   and proving it equals a fold of a named round term by `rfl` hits the default `maxRecDepth` and
   heartbeat guards, and forcing those is forbidden (`AGENTS.md` §6).
2. **The rotation reading is unproved.** `RotRead`, candidate C5 at grade `model`, is carried and
   unused; the *count* is unconditional, the *identification* is not.
3. **The additive identification is `model` at width 32.** `denote_wrapping_add` (`Denotation.lean:35`)
   is proved at width 64 only, and the SHA-2 word is `BitVec 32`.
4. **Padding, loading and storing cost nothing in these figures.** `RoundCost` has no
   list-construction, allocation, byte or read field, so `pad`'s `List.replicate`, `paddedBlock`'s
   sixteen-word regrouping and `Vector.ofFn` digest store are invisible here. The figures are the
   compression figures times a block count, and the *block count* is proved; the rest of `hash` is
   not priced.
5. **Element traffic is invisible**, the same blindness `CostBridge.loads_invisible_to_order`
   (`CostBridge.lean:161`) exhibits for the registered schedule.
6. **No word-level carry measure.** The additive layer absorbs carries before any operation is
   counted; `work/proofs/AlgebraCarry.lean` proves carry facts no cost record reads.
7. **No doubling for SHA-256d.** `hash` is SHA-256; a double application is two `hash` calls and its
   cost is `rep 2`, which this module states as a law (`rangeCost_add`, `blocksCost_add`) but does not
   attach to a Bitcoin figure.
8. **No map to the registered resource record.** `RoundCost` is not connected to
   `CostAlgebra.Cost`, and `CostAlgebra.ops` still collapses six fields into one scalar.
9. **Cross-module identity with AR-15 and AR-10 is a reading, not a check.** Neither
   `AlgebraCost.lean` nor `AlgebraCostRound.lean` is a root of any corpus `lean_lib`, so §3
   re-derives their figures here; the agreement of the two sets of numbers was checked by reading
   and both files carry the sha256 of their bytes.
10. **The nonce-space figure is a range length, not a schedule.** `rangeCost σ nonceSpace` assumes
    `2^32` attempts are made; it says nothing about the order in which they are made, about parallel
    hardware, or about any stopping rule.
11. **The canonical body re-derives the whole schedule per block.** `compress` (`Sha2.lean:97`) calls
    `schedule` afresh for every block, so `blockCost σ` charges the full 48-step extension on every
    attempt. A real miner keeps `W[0 … 15]` fixed across nonces, but that reuse happens in
    `schedule`'s `Vector.ofFn`/`getD` reads and `List.replicate`, and element traffic is invisible
    here (§8 item 5), so the reuse neither appears in these figures nor contradicts them.
-/

/-! ## 9. Audit rows

Every headline above, printed with its axiom set. The permitted cone is
`{propext, Classical.choice, Quot.sound}` and no row may contain `sorryAx`. -/

#print axioms RoundCost.eq_of_fields
#print axioms RoundCost.composed_comm
#print axioms RoundCost.composed_assoc
#print axioms RoundCost.composed_zero
#print axioms RoundCost.rep_zero
#print axioms RoundCost.rep_succ
#print axioms RoundCost.rep_add
#print axioms RoundCost.rep_composed
#print axioms RingCost.eq_of_fields
#print axioms RingCost.composed_comm
#print axioms RingCost.composed_assoc
#print axioms RingCost.rep_zero
#print axioms RingCost.rep_add
#print axioms RingCost.rep_composed
#print axioms ringFields_add
#print axioms ringFields_rep
#print axioms ringMul_body_ops
#print axioms ringUpper_body_ops
#print axioms ringNormForm_body_ops
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
#print axioms t1_arity
#print axioms roundCost_fields
#print axioms roundCost_ringFields
#print axioms roundCostTwoSlot_fields
#print axioms roundCost_twoSlot_add_only
#print axioms schedStepCost_fields
#print axioms roundChainCost_rep
#print axioms roundChainCost_eq
#print axioms roundChain64
#print axioms roundChainCost_add
#print axioms blockCost_split
#print axioms blockCost_eq
#print axioms blockCost_ringFields
#print axioms eq_div_mod
#print axioms msgSplit
#print axioms rest_lt
#print axioms tailLen_cases
#print axioms tailLen_mod
#print axioms canonTotal_multiple
#print axioms canonTotal_ge
#print axioms canonTotal_lt
#print axioms canonTotal_eq
#print axioms zeroFill_add
#print axioms totalLayout
#print axioms zeroFill_le
#print axioms zeroFill_nonneg
#print axioms ceilDiv64_spec
#print axioms ceilDiv64_quot
#print axioms ceilDiv64_strict
#print axioms ceilDiv64_mono
#print axioms canonTotal_eq_ceil
#print axioms canonTotal_div
#print axioms blockCount_is_ceil
#print axioms blockCount_eq
#print axioms blockCount_pos
#print axioms blockCount_le
#print axioms blockCount_succ_le
#print axioms pad_spec
#print axioms pad_length_total
#print axioms pad_blockCount
#print axioms hash_folds_over_blockCount
#print axioms hash_range_nonempty
#print axioms blockCount_0
#print axioms blockCount_55
#print axioms blockCount_56
#print axioms blockCount_63
#print axioms blockCount_64
#print axioms blockCount_119
#print axioms blockCount_120
#print axioms blockCount_80
#print axioms pad_blockCount_80
#print axioms attemptCost_two
#print axioms attemptCost_eq
#print axioms attemptCost_ringFields
#print axioms attemptCost_is_two_blocks
#print axioms attemptCost_strict
#print axioms rangeCost_eq
#print axioms rangeCost_rep_attempt
#print axioms rangeCost_add
#print axioms rangeCost_ring_add
#print axioms rangeCost_ringFields
#print axioms rangeCost_strict
#print axioms rangeCost_pow2_eq
#print axioms rangeCost_nonceSpace
#print axioms rangeCost_nonceSpace_eq_pow2
#print axioms rangeCost_nonceSpace_split
#print axioms rangeCost_nonceSpace_split_ring
#print axioms blocksCost_eq
#print axioms blocksCost_add
#print axioms blocksCost_ring_add
#print axioms messageCost_eq_blocks
#print axioms messageCost_eq
#print axioms messageCost_strict
#print axioms blockCount_million
#print axioms attemptCost_faceBlind
#print axioms attemptCost_threeFaces
#print axioms attemptCost_threeFaces_fieldwise
#print axioms rangeCost_nonceSpace_threeFaces
#print axioms messageCost_threeFaces
#print axioms blockCost_threeFaces
#print axioms faceCertifyCost
#print axioms faceCertifyCost_threeFaces
#print axioms no_root_neg_one_w4
#print axioms no_root_neg_one_w32
#print axioms root_hyperbolic
#print axioms root_parabolic
#print axioms channelMul_admissible_faces

end algebra_cost_pow
